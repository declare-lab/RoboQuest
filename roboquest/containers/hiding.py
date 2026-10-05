"""The observability axis for unfamiliar_containers: which case the level hides, and the geometry the
build needs to stand it in an annex and still work its panels.

The look level (spec 2.2)
-------------------------
The boxes *are* the covers of this task, so there is no ``uncover`` level here: the level says whether
the robot has to go and look for a case, not whether it has to take a lid off something. ``look`` puts
exactly one case out of every policy camera at reset, and an occluder cannot do it: a case stands 0.22 m
tall and the tallest occluder in ``observe.OCCLUDERS`` is 0.225 m, so a ray from a camera 1.4-1.75 m up
clears the occluder and lands on the case unless the occluder is within about 1.5 % of the way from the
camera to it. Measured on layouts 4 and 12 (ten cases, 2026-09-21): ``observe.place_occluder`` rejected
every case on its ideal-slab pre-test - a slab as wide and as tall as the largest catalogue occluder,
standing at the closest gap, blocked none of the cameras that see the case. The annex is therefore the
only mechanism, and ``draw_hidden`` only ever returns ``annex`` entries.

Two of those ten cases were already outside every predicted camera frustum at the reset stance: a 1.8 m
island holds cases past the edge of the policy cameras' view. The level therefore gates what it can
promise - the annexed case is pixel-invisible - and reports, without failing, which of the others the
cameras happen not to reach.

Which case is hidden is nuisance, not structure: a fair coin between the cases that hold an item and
the cases that do not (``observe.draw_hidden``), so "the one you cannot see" carries no information
about where the items are. Nothing else in the spec changes with the level.

Reaching an annexed case
------------------------
The case is not the thing the robot has to reach - its knob is. Both of its panels, the working one and
the fake, must be workable from a stance beside the annex surface, so the yaw the case stands at is
chosen here rather than taken from the spec: the quarter turns are tried in the order the spec's drawn
quarter starts, and the first one whose two panels both pass ``placement.panel_stance`` (an accessible
edge, inside ``placement.MAX_DEPTH`` of it, the base's footprint on free floor) and whose two knob
pinches pass ``observe.reach_gate`` from that stance is kept. Only annex surfaces square to the work
surface are used, so the whole of ``containers.placement`` - which is written in one rectangular frame -
holds for the annexed case too.
"""
import math

import numpy as np

from roboquest import observe as OB
from roboquest.containers import panels as P
from roboquest.containers import placement as PL

LEVELS = ('visible', 'look')
LOOK_MODE = 'annex'                        # the only hiding mechanism a 0.22 m case admits
HIDDEN_COUNT = 1
CASE_RADIUS = round(math.hypot(PL.CASE_HALF, PL.CASE_HALF), 4)      # .1838: the corners of the case
CASE_HEIGHT = P.H                                                   # .22: the knob cap tops out below it
KNOB_RADIUS = round(math.hypot(PL.CASE_HALF + PL.KNOB_OUT, PL.CASE_HALF), 4)   # .2358: case and knob block
SQUARE_TOL_RAD = math.radians(3.)          # an annex surface counts as square to the work surface within this
CORNER_GAP = round(CASE_RADIUS - PL.CASE_HALF, 4)      # how far a corner of the square reaches past its inscribed disc
ANNEX_TRIES = 24                           # spots drawn inside one annex cell before the next cell


# ---- the draw (mint time) --------------------------------------------------------------------------
def box_names(spec):
    """The task-object names of the cases, in index order."""
    return [f"box{box['index']}" for box in spec['boxes']]


def box_index(name):
    return int(str(name).replace('box', ''))


def draw_hidden(rng, level, names, qualifying, count=HIDDEN_COUNT):
    """``spec['hidden']`` for one instance: nothing at ``visible``, one annexed case at ``look``.

    ``qualifying`` are the cases that hold an item; the draw is a fair coin between holders and
    non-holders (:func:`observe.draw_hidden`), so the level never says where the items are.
    """
    level = str(level)
    if level not in LEVELS:
        raise ValueError(f'observability {level!r} is not one of {LEVELS}')
    if level == 'visible':
        return []
    drawn = OB.draw_hidden(rng, list(names), list(qualifying), int(count))
    return [OB.hidden_entry(name, LOOK_MODE) for name in drawn]


def hidden_entries(spec):
    return list(spec.get('hidden') or [])


def hidden_box(spec):
    """The name of the annexed case, or None."""
    entries = [entry for entry in hidden_entries(spec) if entry['mode'] == LOOK_MODE]
    return entries[0]['object'] if entries else None


def level_problems(spec, names):
    """Problems with the level and the hidden list of a containers spec; empty when consistent."""
    level = spec.get('observability')
    if level not in LEVELS:
        return [f'observability must be one of {LEVELS}']
    problems = list(OB.validate_hidden(spec, names))
    entries = hidden_entries(spec)
    if level == 'look':
        if len(entries) != HIDDEN_COUNT:
            problems.append(f'level look hides exactly {HIDDEN_COUNT} case, got {len(entries)}')
        if any(entry['mode'] != LOOK_MODE for entry in entries):
            problems.append(f'a case can only be hidden by the annex, not by {[e["mode"] for e in entries]}')
    return problems


# ---- geometry the build needs ----------------------------------------------------------------------
def footprint_radius(jitter=0.):
    """Half-width of the case's square footprint, turned ``jitter`` off square: what has to fit inside an
    annex cell.

    The circumscribed disc (:data:`CASE_RADIUS`, 0.37 m across) is the wrong measure here and costs every
    annex on some kitchens: observe chunks a counter into cells of at most :data:`observe.CELL_M` along
    the edge, and on layout 4 those chunks are 0.338 m wide, so the disc fits nowhere while the 0.26 m
    case fits with room to spare (measured 2026-09-21: 1 of 8 cells took the disc, 12 of 12 the square).
    The case is square to the surface by construction, so the square is the honest footprint; the corners
    that stick out past the inscribed disc are paid for with :data:`CORNER_GAP` of keep-out clearance.
    """
    jitter = abs(float(jitter))
    return float(PL.CASE_HALF * (math.cos(jitter) + math.sin(jitter)))


def is_square(yaw_a, yaw_b, tol=SQUARE_TOL_RAD):
    """Do two frames differ by a quarter turn (so one rectangular frame describes both)?"""
    return abs(math.sin(2. * (float(yaw_a) - float(yaw_b)))) <= math.sin(2. * float(tol))


def rot2(yaw):
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s], [s, c]])


def inscribed_rect(points):
    """An axis-aligned rectangle inside the convex quadrilateral ``points``: exact when it is already
    axis-aligned (the usual case, kitchens being square), conservative when it is turned."""
    points = np.asarray(points, float)
    xs, ys = np.sort(points[:, 0]), np.sort(points[:, 1])
    return [float(xs[1]), float(xs[2]), float(ys[1]), float(ys[2])]


def edge_label(direction):
    """The side of :data:`placement.SIDES` an outward direction points at."""
    direction = np.asarray(direction, float)[:2]
    return max(PL.SIDES, key=lambda side: float(PL.normal_of(side) @ direction))


def yaw_for_side(side, face):
    """The yaw a case must stand at for ``face`` to look at ``side``."""
    return PL.QUARTER * PL.SIDES.index(side) - P.FACE_YAW[face]


def quarter_turns(start):
    """The four quarter turns, beginning at the spec's drawn quarter."""
    return [(int(start) + k) % 4 for k in range(4)]


def knob_grip_world(base, yaw, face, lid_yaw=0., lid_lift=0.):
    """The pinch point of the knob on ``face``, for a case at ``base`` turned by ``yaw`` (world).

    The same points :func:`panels.build_box` writes into ``knob_grip`` and ``dummy_knob_grip``, but
    available before the case is built, so the annex spot can be gated on them.
    """
    side = face != 'top'
    frame_yaw = float(yaw) + (P.FACE_YAW[face] if side else float(lid_yaw))
    offset = P._knob_points(side, lid_lift)[1]
    return (np.asarray(base, float) + P._rot(frame_yaw) @ np.asarray(offset, float)).tolist()


def case_keepout(xy, yaw, faces=()):
    """The plan footprint of a case and its knob blocks, as one convex hull-free list of polygons."""
    return PL.box_plan_shapes(xy, float(yaw), tuple(faces))


def panel_faces(box):
    """The working face and the fake one (None when the case has only one panel)."""
    return (box['face'], box['dummy_face'])


def flicker_tolerant_diff(first, second):
    """Pixels that differ in **both** of two rendered pairs, per camera.

    ``first`` and ``second`` are ``{camera: (before, after)}`` pairs of the same toggle rendered twice.
    This is the rule :func:`observe.visible_in_cameras` already applies to its own diffs: EGL moves a
    handful of pixels between two renders of an unchanged scene on some kitchens, and a single-pass
    exact diff reads that flicker as a leak. Measured on layout 5 style 6 (2026-09-21): three rounds of
    the item toggle gave 20, 18 and 7 changed pixels on ``agentview_left`` against 11 pixels that moved
    with nothing toggled at all, while 5 pixels of ``agentview_right`` changed in every round - the rule
    drops the flicker and keeps the real leak.
    """
    changed = {}
    for camera, (before_a, after_a) in first.items():
        before_b, after_b = second[camera]
        a = np.any(before_a != after_a, axis=2)
        b = np.any(before_b != after_b, axis=2)
        changed[camera] = int((a & b).sum())
    return changed


def unstable_pixels(first, second):
    """Pixels that differ between two renders of the same state, per camera (reported, never fatal)."""
    return {camera: int(np.any(first[camera] != second[camera], axis=2).sum()) for camera in first}


def entry_summary(record):
    """A short line for reports and evidence tables."""
    placed = (record or {}).get('annex_box')
    if not placed:
        return dict(level=(record or {}).get('level', 'visible'), annexed=None)
    return dict(level=record.get('level'), annexed=placed['box'], fixture=placed['fixture'],
                region=placed['region'], drive_m=placed.get('drive_m'),
                modes=[placed['reach'][label]['mode'] for label in ('real', 'fake')],
                yaw_turns=placed.get('yaw_turns'), spots=placed.get('spots'))
