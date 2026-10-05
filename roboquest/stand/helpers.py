"""The stand's helper containers: native RoboCasa bowls and trays on the work surface.

Every stand scene gets one or two of them, drawn at random from the instance's ``poses`` stream and
never mentioned in the goal: fetching one and parking the ball in it while a leg is lifted is the
robot's own idea. Nothing about the hidden state reaches them, so the draw is uniform over the task
frame and only the geometric constraints reject a pose.

The vocabulary
--------------
:data:`CATALOGUE` lists the assets that were *measured* to hold the 30 mm ball. Each one was built
alone on a ground plane in standalone MuJoCo (1 ms, elliptic cone, noslip 5, the task's own ball
friction), the ball was released at the container's centre just above the rim with a gentle 5 cm/s
sideways drift in each of four directions, and the scene settled for 100 ticks (5 s). An entry is in
this list only when all four trials ended with the ball at rest (< 2 cm/s) inside the container by
the runtime rule below, the container itself had not walked more than 2 mm, and a control ball
parked against the *outside* wall did not read as inside. ``objaverse/tray/tray_9`` is the shape of
thing the rule rejects: a 12 mm lip is not a rim, the ball rolled off it in all four directions.
Flat plates are not in the vocabulary at all for the same reason.

The runtime rule ("the ball is in a helper")
--------------------------------------------
In the helper's own frame: ``|x| <= .8*half_x``, ``|y| <= .8*half_y``, ``z <= height/2 + 1 mm``
(the body origin of an ``MJCFObject`` is its bounding-box centre) and a live contact between the
ball and one of the helper's collision geoms. The 0.8 inset is what separates a ball resting in the
interior from a ball resting against the outside wall, which sits at ``half + r_ball`` and is
rejected for every entry in the catalogue.

The placement constraints
----------------------------------------
1. the whole footprint stays on the work surface, inside the task frame, and at least
   :data:`FRONT_CLEARANCE_M` from the surface's front edge. That is the same 2 cm margin as
   everything else and not the stand's 10 cm rule: the stand needs a wide setback because the ball
   rolls loose on top of it, while a helper is a static container and a ball resting in one is
   parked. The frame's own front edge already sits ``FOOTPRINT_FRONT_MARGIN`` (3 cm) behind the
   counter's, so a helper is never closer than 5 cm to the edge;
2. it clears the shims, the Submit button and the other helper by :data:`HELPER_MARGIN_M` (2 cm),
   and the kitchen's decor and appliances by the same margin (checked at build, where the kitchen
   exists);
3. it stays out of the **roll band**: the disc of radius :data:`ROLL_BAND_M` about the stand's axis,
   which also keeps it clear of the stand itself.

Why that band, and why one radius for every instance
----------------------------------------------------
The ball only leaves the top by climbing the 6 mm rim, which costs a rolling sphere
``sqrt(10 g h / 7) = 0.29 m/s``; the fastest it can be going when it gets there is the run from the
top's centre to the rim at the instance's own tilt, ``sqrt(10 g L sin(theta) / 7)`` with
``L = 99 mm``, i.e. 0.365 m/s at the steepest instance (one short corner leg, 15 mm: 5.51 deg).
What is left after the rim, 0.222 m/s, carries the ball 40.8 cm from the stand's axis before rolling
friction stops it (measured in the bare model, 1 mm physics; the fall off the 12 cm top adds no
distance -- a ball that merely topples off the rim stops at the 13.7 cm it lands at).
:data:`ROLL_BAND_M` rounds that worst case up to 42 cm, and the helper's footprint plus 2 cm must
clear it, so the ball cannot reach a helper by rolling off a stand tilted by its own delta in *any*
direction -- the tilt direction is hidden, so the band has to be the whole disc.

The band is the same for every instance on purpose. A radius derived from the instance's own tilt
would make the helper's distance from the stand a public read-out of ``delta_mm`` (brief ground rule
8), so the steepest instance sets the radius for all of them.

Beyond the band the ball is on its own: a ball thrown by a lifted stand or flung from the gripper can
cross the whole counter and go over its edge, which is the task's documented fatal failure (spec 3,
``left_surface``), not something a placement rule can prevent.
"""
from __future__ import annotations

import math

from roboquest.observe import BOWL_COVERS, bowl_scale_for

ROLL_BAND_M = .42             # see the module docstring; measured worst case 40.8 cm
HELPER_MARGIN_M = .02         # at least 2 cm to the stand, shims, Submit and decor
HELPER_INSET = .80            # of the half extents: the "ball is in this helper" test
HELPER_COUNTS = (1, 2)        # per instance, drawn 50/50 from `poses`
FRONT_CLEARANCE_M = .02       # of the helper footprint to the surface's front edge (see the docstring)
DRAW_ATTEMPTS = 3000          # uniform draws per helper: the allowed area is 1-5 % of the frame

# kind, asset, verified scale, half extents (m, scaled) and height (m, scaled), from the roll-in
# verification described above. Bowls are circular, trays are rectangular; the half extents are the
# scaled bounding box, i.e. half of what MJCFObject reports as `size`, and the build asserts the
# compiled asset still matches them to 1 mm.
#
# The scales are the smallest at which the roll-in test passes, not the cover scale: `bowl_scale_for`
# returns 1.2 for the two profiled bowls because a *cover* has to hide an object, while a parking
# container only has to hold the ball, and at scale 1.0 the interior (radius about 5.4 cm, depth
# 4.3 to 6.1 cm) still clears the 1.5 cm ball by more than 1 cm. Small containers also leave room for
# a second one outside the roll band.
CATALOGUE = (
    dict(kind='bowl', asset='objaverse/bowl/bowl_1', scale=1.0, half=(.060, .060), height=.053),
    dict(kind='bowl', asset='objaverse/bowl/bowl_3', scale=1.0, half=(.065, .065), height=.046),
    dict(kind='bowl', asset='objaverse/bowl/bowl_7', scale=1.0, half=(.065, .065), height=.043),
    dict(kind='bowl', asset='objaverse/bowl/bowl_8', scale=1.0, half=(.060, .060), height=.049),
    dict(kind='bowl', asset='objaverse/bowl/bowl_12', scale=1.0, half=(.060, .060), height=.061),
    dict(kind='bowl', asset='objaverse/bowl/bowl_14', scale=1.0, half=(.057, .057), height=.057),
    dict(kind='tray', asset='objaverse/tray/tray_0', scale=0.8, half=(.053, .088), height=.029),
    dict(kind='tray', asset='objaverse/tray/tray_2', scale=0.8, half=(.055, .084), height=.031),
)


def catalogue():
    """The verified helper containers, each as ``dict(kind, asset, scale, half, height, radius)``.

    ``radius`` is the yaw-invariant bounding radius used where a helper is treated as a disc: the
    footprint of a bowl really is a disc of ``max(half)`` (the meshes are bodies of revolution in a
    square bounding box), a tray's is the half diagonal.
    """
    out = []
    for entry in CATALOGUE:
        half = tuple(float(v) for v in entry['half'])
        radius = max(half) if entry['kind'] == 'bowl' else math.hypot(*half)
        out.append(dict(entry, half=half, radius=radius))
    return tuple(out)


def entry_for(asset):
    """The catalogue entry for ``asset``, or None."""
    return next((e for e in catalogue() if e['asset'] == asset), None)


def fit_scale(asset, ball_radius):
    """``observe.bowl_scale_for`` for the bowls the look job profiled, else None.

    The catalogue scale is the verified one; where a measured interior profile exists this is the
    independent fit check that the ball clears the interior with margin (tested in the CPU suite).
    """
    cover = next((c for c in BOWL_COVERS if c['asset'] == asset), None)
    if cover is None:
        return None
    return bowl_scale_for(cover, float(ball_radius), 2 * float(ball_radius))


# ---- geometry ---------------------------------------------------------------------------------
def corners(xy, yaw_deg, half):
    """The four corners of a helper's footprint, in the frame ``xy`` is given in."""
    yaw = math.radians(float(yaw_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    return [(xy[0] + c * sx * half[0] - s * sy * half[1], xy[1] + s * sx * half[0] + c * sy * half[1])
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _point_gap(xy, yaw_deg, half, point):
    """Distance from ``point`` to the rectangle; 0 when the point is inside it."""
    yaw = math.radians(float(yaw_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = point[0] - xy[0], point[1] - xy[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(lx) - half[0], 0.), max(abs(ly) - half[1], 0.))


def footprint_gap(xy, yaw_deg, half, kind, point):
    """Distance from ``point`` to a helper's footprint: a disc for a bowl, a rectangle for a tray."""
    if kind == 'bowl':
        return max(math.hypot(point[0] - xy[0], point[1] - xy[1]) - max(half), 0.)
    return _point_gap(xy, yaw_deg, half, point)


def placement_problems(xy, yaw_deg, half, *, stand_xy, keepouts, footprint, front_edge_y, kind='tray',
                       front_clearance=FRONT_CLEARANCE_M, label='helper', margin=HELPER_MARGIN_M,
                       band=ROLL_BAND_M):
    """Why a helper of half extents ``half`` at task-frame ``xy``/``yaw_deg`` is not allowed.

    ``keepouts`` are ``(name, (x, y), radius)`` discs: the shims, Submit and the helpers already
    placed. A bowl's footprint is the disc its mesh sweeps, a tray's is its rotated bounding
    rectangle. An empty list of problems means the pose is legal.
    """
    problems = []
    half_x, half_y = footprint[0] / 2, footprint[1] / 2
    if kind == 'bowl':
        radius = max(half)
        outside = abs(xy[0]) + radius > half_x or abs(xy[1]) + radius > half_y
        front = xy[1] - radius - front_edge_y
    else:
        points = corners(xy, yaw_deg, half)
        outside = any(abs(p[0]) > half_x or abs(p[1]) > half_y for p in points)
        front = min(p[1] for p in points) - front_edge_y
    if outside:
        problems.append(f'{label} outside the footprint')
    if front < front_clearance - 1e-9:
        problems.append(f'{label} {front * 100:.1f} cm from the front edge, under the '
                        f'{front_clearance * 100:.0f} cm rule')
    if footprint_gap(xy, yaw_deg, half, kind, stand_xy) - band < margin - 1e-9:
        problems.append(f'{label} inside the ball roll band')
    for name, other_xy, other_radius in keepouts:
        if footprint_gap(xy, yaw_deg, half, kind, other_xy) - other_radius < margin - 1e-9:
            problems.append(f'{label} and {name} overlap')
    return problems


def draw_helpers(rng, *, stand_xy, keepouts, footprint, front_edge_y,
                 front_clearance=FRONT_CLEARANCE_M, counts=HELPER_COUNTS, margin=HELPER_MARGIN_M, band=ROLL_BAND_M,
                 attempts=DRAW_ATTEMPTS):
    """Draw one or two helpers ``dict(kind, asset, scale, xy, yaw)`` at random over the task frame.

    The count, the entry and the pose all come from ``rng`` (the instance's ``poses`` stream) by
    rejection sampling over the whole frame, so the helpers are placed at random and not put
    anywhere in particular. Returns the helpers and the draw statistics. Raises ValueError when a
    helper cannot be placed in ``attempts`` draws, and the mint walks on to the next seed. Consumes
    ``rng`` last, after every other pose draw, so a given seed's stand and shims do not move because
    the helpers were added.
    """
    entries = catalogue()
    half_x, half_y = footprint[0] / 2, footprint[1] / 2
    keepouts = list(keepouts)
    drawn, rejected, tried, dropped = [], 0, 0, 0
    wanted = int(counts[int(rng.integers(len(counts)))])
    for index in range(wanted):
        # A second container is a different asset from the first (the user asked for varied helpers).
        pool = [e for e in entries if e['asset'] not in {h['asset'] for h in drawn}]
        entry = pool[int(rng.integers(len(pool)))]
        for _ in range(int(attempts)):
            tried += 1
            xy = [float(rng.uniform(-half_x, half_x)), float(rng.uniform(-half_y, half_y))]
            yaw = float(rng.uniform(-180., 180.))
            if placement_problems(xy, yaw, entry['half'], kind=entry['kind'], stand_xy=stand_xy,
                                  keepouts=keepouts, footprint=footprint, front_edge_y=front_edge_y,
                                  front_clearance=front_clearance, margin=margin, band=band):
                rejected += 1
                continue
            drawn.append(dict(kind=entry['kind'], asset=entry['asset'], scale=entry['scale'],
                              xy=xy, yaw=yaw))
            keepouts.append((f'helper_{index}', xy, entry['radius']))
            break
        else:
            # The allowed region outside the roll band is small, and a second container does not
            # always fit beside the first one. One helper is the floor, not two: give up on the
            # extra container rather than throwing the seed away.
            if not drawn:
                raise ValueError(f'no room for {entry["asset"]} in {attempts} draws '
                                 f'(stand at {stand_xy}, {len(keepouts)} keep-outs)')
            dropped += 1
    return drawn, dict(draws=tried, rejected=rejected, wanted=wanted, dropped=dropped)


def decor_hits(xy_world, radius, decor_boxes, margin=HELPER_MARGIN_M):
    """Names of the world-aligned decor boxes a helper of bounding ``radius`` comes within ``margin`` of."""
    hits = []
    for box in decor_boxes:
        dx = max(box['x0'] - xy_world[0], 0., xy_world[0] - box['x1'])
        dy = max(box['y0'] - xy_world[1], 0., xy_world[1] - box['y1'])
        if math.hypot(dx, dy) < radius + margin:
            hits.append(box['name'])
    return hits
