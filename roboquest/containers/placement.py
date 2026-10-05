"""Sparse placement of the panel boxes over a work surface, and the rules that say which panels the
robot can reach. CPU only: no simulator import.

Robot frame
-----------
Everything here is expressed in the *robot frame*: x along the front of the work surface, y away from
the robot's start, z up. It is the task frame of :class:`~roboquest.kitchen.RoboQuestKitchen`,
turned by 180 degrees on the rare layout where RoboCasa starts the robot behind the frame, so that
"front" always means the aisle the robot starts in. A box face is named in the box's own frame
(``front``/``right``/``back``/``left``/``top``); the *side* a panel faces is the direction of its outward
normal in the robot frame, quantised to the nearest quarter turn: ``front`` toward the robot's aisle,
``back`` toward the wall side, ``left`` (-x), ``right`` (+x), and ``top``.

Reachable sides: on an island every side; on a counter against a wall never the wall
side. A panel is reachable when the base can stand on free floor beyond an *accessible edge* of the
surface (the front edge always; on an island every edge) with the panel within the arm's reach from
that stance: ``square`` (the panel faces the edge, the base stands in front of it), ``diagonal`` (the
panel faces along the edge, the base stands on the diagonal between the panel's normal and the edge
normal) or ``top`` (a top panel, the base stands at the nearest accessible edge). A panel facing away
from every accessible edge is unreachable.

Placement: the boxes go anywhere on the fixture's free top rectangles (2 cm from the edges),
decor and tall fixtures are keep-outs, the bowl and Submit stay at the front middle and are keep-outs,
centres at least :data:`SPACING_LADDER` [0] apart (relaxed step by step only when a surface cannot hold
the boxes at that spacing; the achieved spacing is recorded), random order, random facing yaw from
{0, 90, 180, 270} degrees plus a small jitter, and every panel, real or fake, reachable.
"""
import math

import numpy as np

from roboquest.containers import panels as P

SIDES = ('front', 'right', 'back', 'left')       # side index i: outward normal at yaw i * pi/2 (matches panels.FACE_YAW)
SIDE_NORMAL = {'front': (0., -1.), 'right': (1., 0.), 'back': (0., 1.), 'left': (-1., 0.)}
QUARTER = math.pi / 2
MODES = ('square', 'diagonal', 'top')

CASE_HALF = P.W / 2                                   # .13, the square case
PANEL_HALF_X = P.PANEL_X[1]                           # .128, the overlay plate
KNOB_OUT = .015 + P.KNOB_REACH                        # .067, plate proud + capped knob

EDGE_MARGIN = .02                # boxes stay this far inside a top rectangle
# Requested centre spacing first; relaxed step by step only when the surface cannot hold the boxes at
# it (a 0.65 m wall counter with decor, the bowl and Submit at its front middle and panels facing
# along it holds three or four cases at 0.55 m). The ladder ends at 0.35 m so that six
# cases fit and more kitchens pass the layout gate; the floor keeps 9 cm between neighbouring cases and
# the plan-shape checks in ``_compatible`` keep knob blocks and hand space clear at any spacing.
SPACING_LADDER = (.55, .50, .45, .40, .35)
KEEPOUT_MARGIN = .008            # boxes and keep-outs never come closer than this in plan
WORK_MARGIN = .004
WORK_DEPTH, WORK_HALF_WIDTH = .12, .10   # the space the hand needs in front of a side knob
LOW_WORK_DEPTH = .06             # a low prop (bowl, Submit) only blocks this much of it: the 45 degree approach
                                 # passes above a 5 cm rim beyond that
# Base stances, from the box centre: the knob recipes were tuned with the base 0.65-0.70 m in front of a
# side panel and 0.57 m in front of a top panel (oracle constants BASE_Y and NEAR_BASE_Y).
STANDOFF = dict(square=.70, diagonal=.70, top=.57)
# How deep into the surface (box centre from the accessible edge) a panel may still be worked from
# that edge. The base's front stops at the edge, so deeper boxes cost arm reach. Square: the knob sits
# 0.16 m nearer than the centre and the tuned 0.70 m stance fits up to 0.52 m of depth. Top: the tuned
# 0.57 m stance fits 0.39 m; beyond, the base is pushed out and the arm reaches the difference (3 cm
# allowed). Diagonal: the knob is reached across the corner of the case, 0.60 m from the base frame at
# 0.17 m of depth (measured, oracle on layout 4) and 0.67 m at the limit.
MAX_DEPTH = dict(square=.50, diagonal=.25, top=.42)
BASE_AHEAD, BASE_BEHIND, BASE_HALF_WIDTH = .15, .55, .25   # PandaOmron collision box about the base frame
BASE_MARGIN = .02
EDGE_GAP = .01                   # between the base's footprint and the surface edge at a stance
EDGE_STANDOFF = BASE_AHEAD + BASE_MARGIN + EDGE_GAP   # .18: the base frame beyond the edge when it faces it square
# The footprint the *floor* check has to use (job CONTAINERS-REACH, 2026-09-22). BASE_HALF_WIDTH above
# is the collision box the knob recipes were tuned against and it is what :func:`edge_clearance` keeps
# using; ``observe.stance_gate`` -- and with it the oracle's floor planner -- tests a wider rectangle,
# 0.70 x 0.70 m about the base frame less a 0.02 m touch tolerance. Checking a stance against the
# narrower box let the build accept stances the planner then refused to drive to ("base missed the
# stance by 0.135 m", certificate eedd1fc10c1fe8b5): measured over the 54 v4 evaluation instances,
# 36 of them had at least one case whose pick stance stood on the island. These two stay locked to
# ``observe.BASE_HALF_WIDTH_M`` and ``observe.STANCE_TOUCH_M`` (asserted in the tests).
STANCE_HALF_WIDTH = .35
STANCE_TOUCH = .02
# Where the *pick* happens, from the box centre: the oracle drives to the panel's stance to open the
# case and then creeps in to take the item out (oracle constants PICK_REACH_IN_DISTANCE 0.58 for a
# side opening, PICK_TOP_DISTANCE 0.57 into an open top). On a square or diagonal panel that is 0.12 m
# nearer than STANDOFF, so a stance that clears the floor at 0.70 m can stand on the surface at 0.58;
# both have to be free or the case cannot be emptied where it stands.
PICK_STANDOFF = dict(square=.58, diagonal=.58, top=.57)
# ...but the base does not have to arrive: the oracle creeps along its heading and accepts stopping
# short, the arm absorbing the residual (oracle DERIVED_STANCE_TOLERANCE). So the rule is that a free
# spot for the crept stance exists within that tolerance along the approach line, not that the crept
# stance itself is free -- the certificate failures were 0.135 m and 0.163 m beyond the tolerance.
DERIVED_STANCE_TOLERANCE = .12   # locked to the oracle constant of that name (asserted in the tests)
PICK_SEARCH_STEP = .01           # the step of the search along the approach line
PROP_STAND_OFF = .50             # oracle PROP_STAND_OFF: the base in front of the bowl or the button
# ---- a removed plate's park spot (jobs CONTAINERS-PICK and CONTAINERS-CLOSER, 2026-09-22) ------------------
# A lift_lid or twist_lid plate (0.257 m square) has to be set down somewhere: on the case's own surface,
# clear of every case with its knob block, the bowl and Submit, with a top-mode stance for the arm on
# free floor within PARK_REACH of the spot (the top-down reach at counter height is 0.70; a front-band
# case's clear spots are at 0.675). The oracle searches these candidates at run time and the mint
# requires one to exist for every such case (``find_park_spot``); both read the constants here, so the
# generator and the oracle agree. Candidates: beside the case along the worked edge's tangent, both
# sides, nearest first, then the same shifted deeper into the surface by each of LID_PARK_BACK (a
# front-band case needs ~0.23 m to clear the bowl's and Submit's blocks); never toward the edge, where
# a plate hangs over the aisle.
LID_PARK_DISTANCES = (.45, .40, .50, .55, .60, .65, .70)
LID_PARK_BACK = (0., .08, .16, .24, .32)
LID_PARK_CLEARANCE = .02         # round the plate
PARK_REACH = .68
PARK_BLOCK_HALF = CASE_HALF + KNOB_OUT     # .197: a case with its knob block, as the park search sees it
PARK_BOWL_CLEARANCE = .05        # round the bowl's outer radius
PARK_SUBMIT_HALF = .10           # the block round Submit
PARK_STANCE_MODE = 'top'         # the plate is set down top-down, from the edge the case is worked from


# ---- sides and faces ----------------------------------------------------------------------------
def quarter_of(yaw):
    """The quarter turn nearest to ``yaw``, as an index 0-3."""
    return int(round(float(yaw) / QUARTER)) % 4


def normal_yaw(yaw, face):
    """Yaw of the outward normal of ``face`` on a box turned by ``yaw`` (robot frame)."""
    return float(yaw) + P.FACE_YAW[face]


def world_side(yaw, face):
    """The side (robot frame) that ``face`` of a box turned by ``yaw`` looks at."""
    if face == 'top':
        return 'top'
    return SIDES[quarter_of(normal_yaw(yaw, face))]


def face_for_side(yaw, side):
    """The face of a box turned by ``yaw`` whose outward normal points to ``side``."""
    if side == 'top':
        return 'top'
    return SIDES[(SIDES.index(side) - quarter_of(yaw)) % 4]


def reachable_sides(island):
    """Sides a panel may face: all of them on an island, never the wall side on a counter."""
    return SIDES + ('top',) if island else ('front', 'right', 'left', 'top')


def normal_of(side):
    return np.asarray(SIDE_NORMAL[side], float)


# ---- plane geometry -------------------------------------------------------------------------------
def rect(center_xy, yaw, half_x, half_y):
    """Corner list of a rectangle, counter-clockwise."""
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    corners = np.array([(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)])
    return np.asarray(center_xy, float) + corners @ rotation.T


def polygons_overlap(a, b, margin=0.):
    """Separating-axis test on two convex polygons, inflated by ``margin``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.array([-edge[1], edge[0]])
            length = float(np.linalg.norm(axis))
            if length < 1e-12:
                continue
            axis = axis / length
            pa, pb = a @ axis, b @ axis
            if pa.max() + margin <= pb.min() or pb.max() + margin <= pa.min():
                return False
    return True


def circle_polygon(center_xy, radius, sides=12):
    angles = np.linspace(0., 2 * math.pi, sides, endpoint=False)
    return np.asarray(center_xy, float) + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def polygon_inside(polygon, region, margin=0.):
    """Every corner inside the rectangle ``[x0, x1, y0, y1]`` shrunk by ``margin``."""
    x0, x1, y0, y1 = region
    polygon = np.asarray(polygon, float)
    return bool(np.all(polygon[:, 0] >= x0 + margin - 1e-9) and np.all(polygon[:, 0] <= x1 - margin + 1e-9)
                and np.all(polygon[:, 1] >= y0 + margin - 1e-9) and np.all(polygon[:, 1] <= y1 - margin + 1e-9))


def box_plan_shapes(xy, yaw, faces):
    """Plan footprints of one case: the wooden case and each side knob's panel-and-knob block.
    ``faces`` are the box-frame faces carrying a panel; a top panel adds nothing in plan."""
    xy, yaw = np.asarray(xy, float), float(yaw)
    shapes = [rect(xy, yaw, CASE_HALF, CASE_HALF)]
    for face in faces:
        if face is None or face == 'top':
            continue
        normal = normal_yaw(yaw, face)
        offset = np.array([math.sin(normal), -math.cos(normal)]) * (CASE_HALF + KNOB_OUT / 2)
        shapes.append(rect(xy + offset, normal, PANEL_HALF_X, KNOB_OUT / 2))
    return shapes


def knob_work_volume(xy, yaw, face, depth=WORK_DEPTH, half_width=WORK_HALF_WIDTH):
    """The space the hand needs in front of a side knob, as a plan rectangle."""
    normal = normal_yaw(yaw, face)
    direction = np.array([math.sin(normal), -math.cos(normal)])
    start = CASE_HALF + KNOB_OUT
    center = np.asarray(xy, float) + direction * (start + depth / 2)
    return rect(center, normal, half_width, depth / 2)


def base_polygon(xy, heading):
    """Plan footprint of the PandaOmron base standing at ``xy`` facing ``heading``, with a margin."""
    xy, heading = np.asarray(xy, float), float(heading)
    center = xy + np.array([math.cos(heading), math.sin(heading)]) * (BASE_AHEAD - BASE_BEHIND) / 2
    return rect(center, heading, (BASE_AHEAD + BASE_BEHIND) / 2 + BASE_MARGIN, BASE_HALF_WIDTH + BASE_MARGIN)


def stance_polygon(xy, heading, margin=-STANCE_TOUCH):
    """The footprint a *floor* check must use: exactly what ``observe.stance_gate`` tests.

    ``observe`` builds the 0.70 x 0.70 m rectangle about the base frame and allows it to overlap a
    fixture by :data:`STANCE_TOUCH` (the base's front standing at the surface edge), so the polygon
    that may touch nothing is that rectangle shrunk by the tolerance -- the default ``margin``. Wider
    than :func:`base_polygon` across the base, which is the whole point: see :data:`STANCE_HALF_WIDTH`.
    The room bounds are the other half of that gate and it tests them on the *full* rectangle, so a
    bounds check passes ``margin=0.``.
    """
    xy, heading = np.asarray(xy, float), float(heading)
    center = xy + np.array([math.cos(heading), math.sin(heading)]) * (BASE_AHEAD - BASE_BEHIND) / 2
    return rect(center, heading, (BASE_AHEAD + BASE_BEHIND) / 2 + margin, STANCE_HALF_WIDTH + margin)


# ---- stances --------------------------------------------------------------------------------------
def depth_from_edge(xy, region, edge):
    """How far a point lies inside the rectangle from one of its edges (negative: outside past it)."""
    x0, x1, y0, y1 = region
    return {'front': xy[1] - y0, 'back': y1 - xy[1], 'left': xy[0] - x0, 'right': x1 - xy[0]}[edge]


def stance_mode(side, edge):
    """How a panel facing ``side`` is worked from beyond ``edge``: None when it faces away from it."""
    if side == 'top':
        return 'top'
    if side == edge:
        return 'square'
    if float(np.dot(normal_of(side), normal_of(edge))) == 0.:
        return 'diagonal'
    return None


def edge_clearance(heading, edge):
    """How far beyond ``edge`` the base frame must stand, facing ``heading`` (robot frame), for the
    base's footprint (plus margin and gap) to clear the surface: 0.18 m facing it square, 0.32 m on
    the diagonal, where a corner of the 0.72 x 0.54 m footprint sweeps toward the edge."""
    inward = -normal_of(edge)
    forward = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-forward[1], forward[0]])
    cf, cl = float(forward @ inward), float(left @ inward)
    ahead = max(cf * (BASE_AHEAD + BASE_MARGIN), -cf * (BASE_BEHIND + BASE_MARGIN))
    return ahead + abs(cl) * (BASE_HALF_WIDTH + BASE_MARGIN) + EDGE_GAP


def stance_for(xy, side, edge, region, mode, standoff=None):
    """Base position and heading (robot frame) to work a panel facing ``side`` from beyond ``edge``.

    The base frame stands :data:`STANDOFF` (or ``standoff``) from the box centre along the approach
    direction (the panel's normal, the diagonal, or the edge normal for a top panel), pushed out along
    the edge normal until its footprint clears the edge (:func:`edge_clearance`), and faces back along
    the approach direction."""
    xy = np.asarray(xy, float)
    n_edge = normal_of(edge)
    if mode == 'square':
        approach = normal_of(side)
    elif mode == 'diagonal':
        approach = normal_of(side) + n_edge
        approach = approach / np.linalg.norm(approach)
    else:
        approach = n_edge
    point = xy + approach * (STANDOFF[mode] if standoff is None else float(standoff))
    heading = math.atan2(-approach[1], -approach[0])
    beyond = -depth_from_edge(point, region, edge)
    need = edge_clearance(heading, edge)
    if beyond < need:
        point = point + n_edge * (need - beyond)
    return point, heading


def pick_stance_for(xy, side, edge, region, mode):
    """Where the base stands to take the item out of a case at ``xy``: the panel's stance crept in to
    :data:`PICK_STANDOFF` (the oracle's reach-in / top pick distance)."""
    return stance_for(xy, side, edge, region, mode, standoff=PICK_STANDOFF[mode])


def free_pick_stance(point, heading, stance_free, tolerance=DERIVED_STANCE_TOLERANCE,
                     step=PICK_SEARCH_STEP):
    """The nearest point to ``point`` along its own heading line whose footprint clears the floor.

    Returns ``(xy, shortfall)`` -- where the base can really stand and how far that is from the wanted
    crept stance -- or ``(None, None)`` when nothing within ``tolerance`` is free. The oracle creeps to
    a pick stance along the base heading (``creep_ahead``) and its drives allow arriving
    :data:`DERIVED_STANCE_TOLERANCE` short of a derived stance, so a crept stance a few centimetres
    inside the counter still works; one 0.135 m inside it does not.
    """
    point = np.asarray(point, float)
    if stance_free is None or stance_free(point, heading):
        return point, 0.
    forward = np.array([math.cos(heading), math.sin(heading)])
    for k in range(1, int(math.floor(tolerance / step + 1e-9)) + 1):
        for sign in (-1., 1.):        # back out along the approach first, then further in
            candidate = point + forward * (sign * k * step)
            if stance_free(candidate, heading):
                return candidate, float(k * step)
    return None, None


def prop_stance_for(xy):
    """Where the base stands in front of a fixed prop (the bowl, Submit) in the robot frame: in the
    robot's own aisle, :data:`PROP_STAND_OFF` out along -y, facing the frame (oracle ``prop_stance``)."""
    xy = np.asarray(xy, float)
    return np.array([xy[0], xy[1] - PROP_STAND_OFF]), math.pi / 2


# ---- the park spot of a removed lid --------------------------------------------------------------
def park_candidates(centre, n_edge, distances=LID_PARK_DISTANCES, back=LID_PARK_BACK):
    """Plate centres to try (robot frame) for a removed lid: beside the case along the worked edge's tangent,
    both sides, nearest first, then the same shifted away from the edge by each of ``back`` (deeper into the
    surface); never toward the edge, where a plate hangs over the aisle."""
    centre, n = np.asarray(centre, float)[:2], np.asarray(n_edge, float)[:2]
    tangent = np.array([-n[1], n[0]])
    out = []
    for b in back:
        for d in distances:
            for sign in (1., -1.):
                out.append(centre + tangent * (sign * d) - n * b)
    return out


def plate_fits(spot, region, blockers, clearance=LID_PARK_CLEARANCE):
    """Whether a plate centred at ``spot`` lies inside the surface ``region`` (x0, x1, y0, y1) and clear of every
    blocker polygon (cases with their knob blocks, the bowl, Submit)."""
    plate = rect(spot, 0., P.LID_HALF + clearance, P.LID_HALF + clearance)
    if not polygon_inside(plate, region, 0.):
        return False
    return not any(polygons_overlap(plate, b, .01) for b in blockers)


def case_park_block(xy, yaw):
    """The plan block a case keeps a parked plate out of: the case with its knob blocks, square."""
    return rect(xy, yaw, PARK_BLOCK_HALF, PARK_BLOCK_HALF)


def prop_park_blockers(bowl_xy, submit_xy, bowl_radius=P.BOWL_OUTER_RADIUS):
    """The bowl's and Submit's blocks for the park search (robot frame)."""
    return [circle_polygon(bowl_xy, float(bowl_radius) + PARK_BOWL_CLEARANCE),
            rect(submit_xy, 0., PARK_SUBMIT_HALF, PARK_SUBMIT_HALF)]


def park_stance_for(spot, edge, region):
    """The top-mode stance the plate is set down from: :data:`STANDOFF` ['top'] beyond ``spot`` along the
    edge normal, pushed out past the edge (:func:`stance_for`)."""
    point, heading = stance_for(spot, PARK_STANCE_MODE, edge, region, PARK_STANCE_MODE)
    return np.asarray(point, float), float(heading)


def find_park_spot(centre, edge, region, blockers, stance_free, reach=PARK_REACH, candidates=None):
    """The first park candidate whose plate lies on ``region`` clear of every ``blockers`` polygon and whose
    park stance (:func:`park_stance_for`) is within ``reach`` of the spot and on free floor
    (``stance_free(xy, heading)``; None accepts every stance). Returns ``dict(spot, stance_xy, heading,
    where, refused)`` with ``where`` 'beside' (on the case's own line along the edge) or 'deeper', or None
    with nothing found; ``refused`` counts the candidates by the reason they fell. ``candidates`` may be a
    prepared list of ``(spot, stance_xy, heading, where)`` (the static part of the search, see
    :func:`park_options`), in which case only the blockers are tested."""
    centre = np.asarray(centre, float)[:2]
    refused = {'no surface': 0, 'out of reach': 0, 'stance blocked': 0, 'blocked': 0}
    if candidates is None:
        candidates = park_options(centre, edge, region, stance_free, reach, refused)
    for spot, point, heading, where in candidates:
        if not plate_fits(spot, region, blockers):
            refused['blocked'] += 1
            continue
        return dict(spot=np.asarray(spot, float), stance_xy=np.asarray(point, float), heading=float(heading),
                    where=where, refused=refused)
    return None


def park_options(centre, edge, region, stance_free, reach=PARK_REACH, refused=None):
    """The static part of the park search for a case at ``centre`` worked from ``edge``: every candidate
    whose plate lies on the surface and whose park stance is within ``reach`` and on free floor, as
    ``(spot, stance_xy, heading, where)`` in search order. Independent of the other cases, so a placement
    search computes it once per candidate cell and tests only the blockers afterwards."""
    centre = np.asarray(centre, float)[:2]
    n_edge = normal_of(edge)
    out = []
    for spot in park_candidates(centre, n_edge):
        if not plate_fits(spot, region, ()):
            if refused is not None:
                refused['no surface'] += 1
            continue
        point, heading = park_stance_for(spot, edge, region)
        if float(np.linalg.norm(point - spot)) > reach:
            if refused is not None:
                refused['out of reach'] += 1
            continue
        if stance_free is not None and not stance_free(point, heading):
            if refused is not None:
                refused['stance blocked'] += 1
            continue
        where = 'beside' if abs(float((spot - centre) @ n_edge)) < 1e-6 else 'deeper'
        out.append((spot, point, heading, where))
    return out


def panel_stance(xy, yaw, face, region, edges, stance_free=None):
    """The best stance for one panel: the accessible edge it can be worked from at the smallest depth,
    or None when it is unreachable. ``stance_free(xy, heading)`` says whether the base fits there.

    The panel's own standoff stance, where the knob is opened, has to be free. The pick stance it
    creeps to afterwards (:func:`pick_stance_for`) is 0.12 m nearer the surface on a square or
    diagonal panel, so it is the one that ends up on the counter; it only has to have a free spot
    within :data:`DERIVED_STANCE_TOLERANCE` along the approach line (:func:`free_pick_stance`), which
    is the residual the oracle's own drives absorb. Each option records that shortfall.
    """
    side = world_side(yaw, face)
    options = []
    for edge in edges:
        mode = stance_mode(side, edge)
        if mode is None:
            continue
        depth = float(depth_from_edge(xy, region, edge))
        if depth < 0. or depth > MAX_DEPTH[mode] + 1e-9:
            continue
        point, heading = stance_for(xy, side, edge, region, mode)
        if stance_free is not None and not stance_free(point, heading):
            continue
        pick_point, pick_heading = pick_stance_for(xy, side, edge, region, mode)
        pick_point, shortfall = free_pick_stance(pick_point, pick_heading, stance_free)
        if pick_point is None:
            continue
        options.append(dict(mode=mode, side=side, edge=edge, depth=round(depth, 4),
                            xy=[float(point[0]), float(point[1])], heading=float(heading),
                            pick_xy=[float(pick_point[0]), float(pick_point[1])],
                            pick_heading=float(pick_heading), pick_shortfall=round(float(shortfall), 4)))
    if not options:
        return None
    return min(options, key=lambda o: (o['depth'], o['pick_shortfall'], MODES.index(o['mode'])))


# ---- the solver -----------------------------------------------------------------------------------
class PlacementError(ValueError):
    """No arrangement satisfies the rules on this surface (the build fails closed)."""


class _Poly:
    """A convex plan polygon with its bounding box, so most overlap tests end at the box test."""
    __slots__ = ('pts', 'x0', 'x1', 'y0', 'y1')

    def __init__(self, pts):
        self.pts = np.asarray(pts, float)
        self.x0, self.x1 = float(self.pts[:, 0].min()), float(self.pts[:, 0].max())
        self.y0, self.y1 = float(self.pts[:, 1].min()), float(self.pts[:, 1].max())

    def overlaps(self, other, margin):
        if (self.x1 + margin <= other.x0 or other.x1 + margin <= self.x0
                or self.y1 + margin <= other.y0 or other.y1 + margin <= self.y0):
            return False
        return polygons_overlap(self.pts, other.pts, margin)

    def inside(self, region, margin):
        x0, x1, y0, y1 = region
        return (self.x0 >= x0 + margin - 1e-9 and self.x1 <= x1 - margin + 1e-9
                and self.y0 >= y0 + margin - 1e-9 and self.y1 <= y1 - margin + 1e-9)


GRID_PITCH = .025                # candidate centres: a fine grid over every region, visited in random order


def _grid(regions):
    """Every candidate centre (cell centres ``inset`` inside each region) with its region index."""
    cells, owners = [], []
    inset = EDGE_MARGIN + CASE_HALF
    for index, region in enumerate(regions):
        x0, x1, y0, y1 = region['rect']
        if x1 - x0 < 2 * inset or y1 - y0 < 2 * inset:
            continue
        xs = np.arange(x0 + inset, x1 - inset + 1e-9, GRID_PITCH)
        ys = np.arange(y0 + inset, y1 - inset + 1e-9, GRID_PITCH)
        grid = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2)
        cells.append(grid)
        owners.append(np.full(len(grid), index))
    if not cells:
        return np.zeros((0, 2)), np.zeros(0, int)
    return np.vstack(cells), np.concatenate(owners)


SPREAD_CHOICES = 6               # a box first tries the roomiest of this many random valid spots (the first box: any)
OVERHANG = .25                   # how far a knob block may hang past a surface edge (the room bounds still apply)
INTERACT = 2 * (CASE_HALF + KNOB_OUT) + WORK_DEPTH   # nearer than this two cases (or a hand and a case) can touch
SEARCH_BUDGET = 4000             # search nodes per spacing before the next step of the ladder


class _Budget(Exception):
    pass


def _box_cells(box, regions, keepouts, low_keepouts, cells, owners, bounds, stance_free):
    """Every grid cell where this box alone satisfies the rules: both panels reachable from an accessible
    edge (with a free stance), the case 2 cm inside its rectangle, knob blocks over the surface or its
    overhang and inside the room, nothing on a keep-out, the hand's space in front of each side knob
    clear of the keep-outs (and of low props for its first few centimetres)."""
    faces = (box['face'], box['dummy_face'])
    panels = (('real', box['face']), ('fake', box['dummy_face']))
    side_faces = [f for f in faces if f not in (None, 'top')]
    out = []
    for xy, owner in zip(cells, owners):
        region = regions[int(owner)]
        if any(panel_stance(xy, box['yaw'], face, region['rect'], region['edges']) is None for _, face in panels):
            continue
        shapes = [_Poly(s) for s in box_plan_shapes(xy, box['yaw'], faces)]
        if not shapes[0].inside(region['rect'], EDGE_MARGIN):
            continue
        if not all(s.inside(region['overhang'], 0.) for s in shapes[1:]):
            continue
        if bounds is not None and not all(s.inside(bounds, 0.) for s in shapes[1:]):
            continue
        if any(s.overlaps(k, KEEPOUT_MARGIN) for s in shapes for k in keepouts + low_keepouts):
            continue
        volumes = [_Poly(knob_work_volume(xy, box['yaw'], f)) for f in side_faces]
        if bounds is not None and not all(v.inside(bounds, 0.) for v in volumes):
            continue
        if any(v.overlaps(k, WORK_MARGIN) for v in volumes for k in keepouts):
            continue
        near = [_Poly(knob_work_volume(xy, box['yaw'], f, depth=LOW_WORK_DEPTH)) for f in side_faces]
        if any(v.overlaps(k, WORK_MARGIN) for v in near for k in low_keepouts):
            continue
        stances = {label: panel_stance(xy, box['yaw'], face, region['rect'], region['edges'], stance_free)
                   for label, face in panels}
        if any(s is None for s in stances.values()):
            continue
        out.append(dict(index=box['index'], xy=np.asarray(xy, float), shapes=shapes, volumes=volumes,
                        stances=stances, region=int(owner)))
    return out


def _compatible(entry, placed, spacing):
    for other in placed:
        distance = float(np.linalg.norm(entry['xy'] - other['xy']))
        if distance < spacing:
            return False
        if distance < INTERACT:
            if any(s.overlaps(q, KEEPOUT_MARGIN) for s in entry['shapes'] for q in other['shapes']):
                return False
            if any(v.overlaps(q, WORK_MARGIN) for v in entry['volumes'] for q in other['shapes']):
                return False
            if any(v.overlaps(s, WORK_MARGIN) for v in other['volumes'] for s in entry['shapes']):
                return False
    return True


def _room(entry, placed):
    if not placed:
        return 0.
    return min(float(np.linalg.norm(entry['xy'] - other['xy'])) for other in placed)


def _search(valid, spacing, rng, budget, sample, park=None):
    """Randomised backtracking over the boxes in order: each box takes, first, the roomiest of ``sample``
    random compatible cells (spread where there is room), then the rest in random order (the tight
    packings a greedy choice walks past). ``budget`` search nodes at most; None when none is found.

    ``park`` (a :class:`_ParkRule`) is the lid park rule: a cell of a lift/twist-lid case is taken only
    while a park spot survives the cases placed so far, and a complete arrangement only when every
    such case still has one (a later case may have taken the spot). Returns ``(placed, nodes, parks,
    park_rejections)``: the park spots per case index and how many cells or arrangements the rule refused."""
    placed, nodes, rejected = [], [0], [0]
    parks = {}

    def rec(k):
        if k == len(valid):
            if park is None:
                return True
            found = park.spots(placed)
            if found is None:
                rejected[0] += 1
                return False
            parks.clear()
            parks.update(found)
            return True
        options = [e for e in valid[k] if _compatible(e, placed, spacing)]
        if not options:
            return False
        order = [int(i) for i in rng.permutation(len(options))]
        head = sorted(order[:sample], key=lambda i: -_room(options[i], placed))
        for i in head + order[sample:]:
            nodes[0] += 1
            if nodes[0] > budget:
                raise _Budget()
            if park is not None and not park.admits(options[i], placed):
                rejected[0] += 1
                continue
            placed.append(options[i])
            if rec(k + 1):
                return True
            placed.pop()
        return False

    try:
        found = rec(0)
    except _Budget:
        return None, nodes[0], {}, rejected[0]
    return (list(placed) if found else None), nodes[0], dict(parks), rejected[0]


class _ParkRule:
    """The lid park rule inside the placement search: every case whose mechanism is a free lid
    (``panels.FREE_LID_MECHANISMS``) needs a park spot (:func:`find_park_spot`) on its own region, clear of
    every other case's block, the fixed ``blockers`` (the bowl, Submit, an annexed case) and its own block,
    with a free park stance within reach. The static part of a case's search (surface, reach, free floor)
    is computed once per candidate cell; the cases placed so far are the moving part."""

    def __init__(self, boxes, regions, blockers, stance_free):
        self.lid = {int(b['index']) for b in boxes if b.get('mechanism') in P.FREE_LID_MECHANISMS}
        self.yaw = {int(b['index']): float(b['yaw']) for b in boxes}
        self.regions = regions
        self.blockers = [np.asarray(b, float) for b in blockers]
        self.stance_free = stance_free
        self._static = {}

    def _options(self, entry):
        key = (entry['index'], round(float(entry['xy'][0]), 4), round(float(entry['xy'][1]), 4))
        if key not in self._static:
            region = self.regions[int(entry['region'])]
            edge = entry['stances']['real']['edge']
            self._static[key] = park_options(entry['xy'], edge, region['rect'], self.stance_free)
        return self._static[key]

    def _blocks(self, entries):
        return [case_park_block(e['xy'], self.yaw[e['index']]) for e in entries] + self.blockers

    def _find(self, entry, others):
        region = self.regions[int(entry['region'])]
        edge = entry['stances']['real']['edge']
        return find_park_spot(entry['xy'], edge, region['rect'], self._blocks(others + [entry]), self.stance_free,
                              candidates=self._options(entry))

    def admits(self, entry, placed):
        """Whether ``entry`` (about to be placed) still has a park spot among the cases placed so far, and
        whether every lid case already placed keeps one with ``entry`` in."""
        if entry['index'] in self.lid and self._find(entry, placed) is None:
            return False
        return all(self._find(other, [e for e in placed if e is not other] + [entry]) is not None
                   for other in placed if other['index'] in self.lid)

    def spots(self, placed):
        """The park spot of every lid case in a complete arrangement, or None when one has none."""
        out = {}
        for entry in placed:
            if entry['index'] not in self.lid:
                continue
            found = self._find(entry, [e for e in placed if e is not entry])
            if found is None:
                return None
            out[entry['index']] = dict(spot=[float(found['spot'][0]), float(found['spot'][1])],
                                       stance_xy=[float(found['stance_xy'][0]), float(found['stance_xy'][1])],
                                       heading=float(found['heading']), where=found['where'])
        return out


def place_boxes(boxes, regions, keepouts, rng, stance_free=None, bounds=None, low_keepouts=(), ladder=SPACING_LADDER,
                budget=SEARCH_BUDGET, sample=SPREAD_CHOICES, park_blockers=None):
    """Place ``boxes`` (dicts with ``index``, ``yaw``, ``face``, ``dummy_face``, in placement order) on the
    ``regions`` (dicts with ``rect`` = [x0, x1, y0, y1] and ``edges`` = accessible edges), away from the
    ``keepouts`` (plan polygons of things that stand in the hand's way: decor, tall fixtures) and the
    ``low_keepouts`` (low props that only block a case and the first few centimetres in front of a knob),
    inside the room ``bounds`` (a rectangle, or None), at the first spacing of ``ladder`` that admits an
    arrangement. Positions lie on a :data:`GRID_PITCH` grid.

    ``park_blockers`` switches the lid park rule on (:class:`_ParkRule`): every box whose ``mechanism`` is
    a free lid must have a park spot for its plate clear of the other cases and of these polygons (the
    bowl's and Submit's park blocks, :func:`prop_park_blockers`, and an annexed case's block), with a free
    park stance within :data:`PARK_REACH`; the search walks on past arrangements without one.

    Deterministic in ``rng``. Returns ``dict(positions, reach, spacing, ladder_step, nodes, valid_cells,
    park, park_rejections)`` -- ``park`` the park spot per lid case index -- and raises
    :class:`PlacementError` when no spacing of the ladder works (naming the arrangements the park rule
    refused, so a build error can be read for it)."""
    keepouts = [_Poly(k) for k in keepouts]
    low_keepouts = [_Poly(k) for k in low_keepouts]
    regions = [dict(r, overhang=[r['rect'][0] - OVERHANG, r['rect'][1] + OVERHANG,
                                 r['rect'][2] - OVERHANG, r['rect'][3] + OVERHANG]) for r in regions]
    cells, owners = _grid(regions)
    valid = [_box_cells(box, regions, keepouts, low_keepouts, cells, owners, bounds, stance_free) for box in boxes]
    counts = {box['index']: len(v) for box, v in zip(boxes, valid)}
    empty = [index for index, count in counts.items() if count == 0]
    if empty:
        raise PlacementError(f'boxes {empty} have no valid spot on the surface at all (valid cells {counts})')
    park = _ParkRule(boxes, regions, park_blockers, stance_free) if park_blockers is not None else None
    if park is not None and not park.lid:
        park = None
    nodes_total, park_rejected = 0, 0
    for step, spacing in enumerate(ladder):
        placed, nodes, parks, rejected = _search(valid, float(spacing), rng, int(budget), int(sample), park)
        nodes_total += nodes
        park_rejected += rejected
        if placed is not None:
            positions = {e['index']: [float(e['xy'][0]), float(e['xy'][1])] for e in placed}
            reach = {e['index']: dict(e['stances'], region=e['region']) for e in placed}
            return dict(positions=positions, reach=reach, spacing=float(spacing), ladder_step=step,
                        nodes=nodes_total, valid_cells=counts, park=parks, park_rejections=park_rejected)
    park_note = (f'; the lid park rule refused {park_rejected} cells or arrangements (no park spot for a plate)'
                 if park is not None else '')
    raise PlacementError(f'no arrangement of {len(boxes)} boxes on {len(regions)} region(s) at spacings {list(ladder)} '
                         f'({nodes_total} search nodes; valid cells per box {counts}{park_note})')



def settle_prop(polygon_at, nominal_xy, keepouts, region, steps=None):
    """Slide a fixed prop (bowl, Submit) sideways from its nominal spot until it touches no keep-out and
    lies inside ``region``. Returns (xy, shift)."""
    keepouts = [np.asarray(k, float) for k in keepouts]
    steps = list(steps) if steps is not None else [0.] + [s * d for d in np.arange(.05, .501, .05) for s in (-1., 1.)]
    nominal = np.asarray(nominal_xy, float)
    for dx in steps:
        xy = nominal + np.array([dx, 0.])
        polygon = polygon_at(xy)
        if not polygon_inside(polygon, region, EDGE_MARGIN):
            continue
        if any(polygons_overlap(polygon, k, KEEPOUT_MARGIN) for k in keepouts):
            continue
        return [float(xy[0]), float(xy[1])], float(dx)
    raise PlacementError(f'no free spot for a fixed prop near {nominal.tolist()}')
