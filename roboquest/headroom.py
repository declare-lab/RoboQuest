"""Headroom above covers, annexed objects and occluders (OBSERVE-HEADROOM, spec 2.2).

:mod:`observe` places covers and hidden objects after checking reach and stance in the plane only.
The skills that undo the hiding (:mod:`uncover`) need room *above*: :func:`~uncover.lift_cloche`
raises the dome by its lift before carrying it, :func:`~uncover.slide_lift_bowl` pinches the rim at
the counter edge and raises the bowl to ``top + 2R + 0.12``, :func:`~uncover.lift_plate_from_pan`
raises the plate to ``top + 0.22``, :func:`~uncover.pour_cup` holds the cup at ``top + 0.20`` and
rolls it, :func:`~uncover.fetch` and :func:`~uncover.move_aside` grasp from ``0.14`` / ``0.12`` above
the grasp point. Above every top-down grasp the Panda's wrist and forearm stand nearly vertical, so
the highest point over the cover is the forearm (link 5, sometimes link 4), about 0.30 m above the
end-effector site. RoboCasa kitchens hang wall cabinets, hoods, microwaves and open shelves 0.45 to
0.60 m above parts of the counter; ``RoboQuestKitchen._decor_boxes`` leaves them out on purpose
because they do not touch the top, so until now a cover placed under one passed every gate and could
not be lifted (the SKILLS job: a cloche under the layout 52 microwave rose 7.7 cm and stopped).

The rule
--------
The clearance volume of a cover (or of an object a skill lifts) is the vertical prism over its
disc inflated by :data:`HAND_PAD_M` (the hand straddles the grasp point, the forearm leans a little
toward the robot), from the surface top up to :func:`headroom_needed`. A placement is refused when an
overhead fixture's underside lies inside that prism (:func:`headroom_gate`). The overhead fixtures
are every RoboCasa fixture whose bounding box starts above the surface top (:func:`overhead_boxes`);
the ones that reach down into the counter band are decor and already footprint keep-outs.

Measurement
-----------
The per-kind allowances below were measured, not guessed: each skill ran on a clear counter (stamps
layout 4 style 4, an island with nothing overhead; ``odd_parcel`` layout 25 for the plate on the pan)
with a callback recording, at every tick, the highest point of any robot geom whose world bounding
box overlaps the cover's column (the disc inflated by 0, 0.03, 0.05 and 0.10 m: the same peak every
time, the forearm and wrist stand over the cover itself) and the highest point of the carried
cover. The phases that work at the cover (pre-grasp waypoint, grasp, lift, carry raise) give the
peak; the tuck at the reset stance before the skill is excluded (the folded arm stands 0.67 m above
the counter at the front edge whatever the cover). ``allowance = peak - hand``, the hand being the
end-effector height the skill commands there (cover height + lift for the lifting skills, the
waypoint or push height for the hand-only motions).
The gate adds
:data:`HEADROOM_MARGIN_M` to every requirement.
"""
import math

import numpy as np

__all__ = ['HAND_PAD_M', 'HEADROOM_MARGIN_M', 'OVERHEAD_MIN_GAP_M', 'OVERHEAD_MAX_M', 'ALLOWANCE_M', 'MEASURED',
           'LIFT_KINDS', 'overhead_boxes', 'lowest_overhead', 'headroom_needed', 'headroom_gate', 'path_headroom_gate',
           'cloche_lift']

HAND_PAD_M = .05             # the clearance column is the cover's disc inflated by this much: the peak geoms (forearm,
                             # wrist) stand inside the disc itself (identical peaks measured with pads 0, .03, .05
                             # and .10), the hand's .10 m half-width overhangs a small cover's edge by 2 cm
HEADROOM_MARGIN_M = .04      # added to every measured requirement (one arm configuration was measured per skill)
OVERHEAD_MIN_GAP_M = .02     # a fixture starting within this of the top touches it: decor, not overhead
OVERHEAD_MAX_M = 1.20        # fixtures starting higher than this above the top never matter (ceiling, lights)
MIN_OVERHEAD_SIDE_M = .03    # a box thinner than this in x or y lies in the wall plane (switches, outlets, sills)
NOT_OVERHEAD_CLASSES = ('Counter', 'Wall', 'Floor', 'Window')

# Skill constants the requirement is built from (uncover.py; kept here so observe.py needs no skill import).
CLOCHE_MIN_LIFT_M = .12      # lift_cloche: lift = max(object height + .05, .12)
CLOCHE_LIFT_OVER_OBJECT_M = .05
BOWL_RAISE_M = .12           # slide_lift_bowl raises the pinched rim to top + 2R + .12 before carrying
BOWL_PUSH_HAND_M = .025      # fingertips while sliding
PLATE_RAISE_M = .22          # lift_plate_from_pan raises the hand to at least top + .22
PLATE_ABOVE_M = .12          # pre-grasp / lift above the rim pinch point
PLATE_RIM_BELOW_TOP_M = .006
CUP_HOLD_M = .20             # pour_cup holds the hand at top + .20 while tilting
PUSH_ABOVE_M = .15           # push_cover's waypoint above the contact point
FETCH_ABOVE_M = .14          # fetch's pre-grasp waypoint above the grasp point
ASIDE_ABOVE_M = .12          # move_aside's pre-grasp waypoint above its grasp point (top - .035)
ASIDE_GRASP_BELOW_TOP_M = .035

# Measured peaks (metres above the surface top) of any robot geom inside the column during the phases at the
# cover, with the cover geometry of the reference scene (the ``robot_cover`` / ``robot_object`` columns of
# the per-phase measurement tables). ``hand`` is the end-effector height the
# skill commands there; ``allowance = peak - hand``. The forearm (link 5) or wrist (link 6) is the peak in
# every top-down case and stands 0.28 to 0.41 m above the end-effector, higher when the hand is lower.
MEASURED = {
    'cloche_small': dict(peak=.559, hand=.2505, geom='robot0_link5/6', phases='cloche_grasp_above..cloche_carry_retract',
                         scene='cloche_small_stamp: knob top .129, stamp .084 so lift .134'),
    'cloche_large': dict(peak=.573, hand=.2805, geom='robot0_link6', phases='cloche_grasp_above..cloche_carry_retract',
                         scene='cloche_large_stamp: knob top .159, lift .134'),
    'plate_lift': dict(peak=.544, hand=.203, geom='robot0_link5', phases='pan_plate_above..pan_plate_lower',
                       scene='pan_plate_parcel25: plate top .089 above the top, pinch .083, waypoint +.12 (the pinch itself '
                             'failed afterwards, SKILLS-2; the raise to .22 would add .017)'),
    'push_contact': dict(peak=.525, hand=.15, geom='robot0_link5', phases='push_above',
                         scene='push_plate_stamps4_free: the waypoint .15 above the contact point'),
    'push_path': dict(peak=.431, hand=.02, geom='robot0_link5', phases='push_lower..push_push_5',
                      scene='push_plate_stamps4_free: fingertips at the plate edge, 20 cm push'),
    'fetch': dict(peak=.543, hand=.224, geom='robot0_link5', phases='fetch_grasp_above..fetch_carry_retract',
                  scene='annex_stamps4: stamp .084 tall grasped at its handle, waypoint +.14 (hand = object height + .14)'),
    'move_aside': dict(peak=.604, hand=.3097, geom='robot0_link6', phases='aside_grasp_above..aside_grasp_lower',
                       scene='occluder_stamps4: a .2247 m occluder grasped .035 below its top, waypoint +.12 (the grasp '
                             'itself stopped short afterwards; the lift is .06, below the waypoint)'),
    'wide_cup': dict(peak=.302, hand=.0975, geom='gripper0_right_hand', phases='pour_pre',
                     scene='cup_cube_stamps4: side grasp, fingers vertical: the hand body tops .205 above the end-effector '
                           '(the pour itself failed at the pre-pose, SKILLS-2)'),
    'bowl': dict(peak=None, hand=None, scene='bowl_cube_stamps4 does not build at this tip (no bowl-feasible cube spot); '
                                             'derived from the cloche lifts (same top-down pinch, hand at 2R + .12) and the '
                                             'push path (the slide, fingertips at .025)'),
}

# Allowance above the commanded hand height, per rule (see :func:`headroom_needed`); rounded up from MEASURED.
ALLOWANCE_M = {
    'cloche': .30,          # .296 small, .280 large
    'bowl_lift': .30,       # derived: the cloche allowance at a higher hand (the forearm rises less than the hand)
    'bowl_slide': .41,      # derived from push_path (.411 at hand .02; the slide hand is at .025)
    'wide_cup': .10,        # hand top .205 above a side-grasp end-effector; the rule's hand is hold (.20) + cup height
                            # (.11), .105 above the hand top itself, so .10 keeps needed >= peak: .45 >= .405
    'plate_lift': .34,      # .341
    'push_contact': .38,    # .375
    'push_path': .41,       # .411
    'fetch': .32,           # .319
    'move_aside': .30,      # .294
}

LIFT_KINDS = ('cloche_small', 'cloche_large', 'bowl', 'wide_cup', 'plate_lift', 'fetch', 'move_aside')


def cloche_lift(object_height):
    """The lift :func:`uncover.lift_cloche` uses over an object of this height."""
    return max(float(object_height) + CLOCHE_LIFT_OVER_OBJECT_M, CLOCHE_MIN_LIFT_M)


def headroom_needed(kind, *, cover_height=0., object_height=0., radius=0., support_above=0., margin=HEADROOM_MARGIN_M):
    """Metres of clear air the skill for ``kind`` needs above the surface top over its column.

    ``kind``: ``cloche_small`` / ``cloche_large`` (``cover_height`` = knob top, ``object_height`` sets the
    lift), ``bowl`` (the lift at the counter edge: ``radius`` = outer radius, ``cover_height`` = bowl
    height), ``bowl_slide`` (the hand pushing the bowl along its slide path), ``wide_cup``
    (``cover_height`` = cup height), ``plate_lift`` (a plate pinched off a pan: ``support_above`` = the
    plate's support height above the top, ``cover_height`` = its thickness), ``push_contact`` /
    ``push_path`` (a flat cover pushed aside; nothing is lifted), ``fetch`` and ``move_aside``
    (``object_height`` = the lifted object's height). Returns ``dict(needed_m, hand_m, lift_m,
    allowance_m, margin_m)``; ``needed_m`` is non-decreasing in every height argument."""
    cover_height, object_height = max(float(cover_height), 0.), max(float(object_height), 0.)
    radius, support_above = max(float(radius), 0.), max(float(support_above), 0.)
    if kind in ('cloche_small', 'cloche_large'):
        lift = cloche_lift(object_height)
        hand = cover_height + lift
        allowance = ALLOWANCE_M['cloche']
    elif kind == 'bowl':
        lift = 2 * radius + BOWL_RAISE_M
        hand = max(lift, cover_height)
        allowance = ALLOWANCE_M['bowl_lift']
    elif kind == 'bowl_slide':
        lift = 0.
        hand = BOWL_PUSH_HAND_M
        allowance = ALLOWANCE_M['bowl_slide']
    elif kind == 'wide_cup':
        lift = CUP_HOLD_M
        hand = cover_height + lift
        allowance = ALLOWANCE_M['wide_cup']
    elif kind == 'plate_lift':
        lift = PLATE_ABOVE_M
        hand = max(PLATE_RAISE_M, support_above + cover_height - PLATE_RIM_BELOW_TOP_M + lift)
        allowance = ALLOWANCE_M['plate_lift']
    elif kind == 'push_contact':
        lift = 0.
        hand = PUSH_ABOVE_M
        allowance = ALLOWANCE_M['push_contact']
    elif kind == 'push_path':
        lift = 0.
        hand = cover_height
        allowance = ALLOWANCE_M['push_path']
    elif kind == 'fetch':
        lift = FETCH_ABOVE_M
        hand = object_height + lift
        allowance = ALLOWANCE_M['fetch']
    elif kind == 'move_aside':
        lift = ASIDE_ABOVE_M
        hand = object_height - ASIDE_GRASP_BELOW_TOP_M + lift
        allowance = ALLOWANCE_M['move_aside']
    else:
        raise ValueError(f'no headroom rule for {kind!r}')
    # the arm above the hand, or the cover itself when it is taller than that (never, for the lifting rules)
    needed = max(hand + allowance, cover_height) + float(margin)
    return dict(kind=kind, needed_m=round(needed, 4), hand_m=round(hand, 4), lift_m=round(lift, 4),
                allowance_m=allowance, margin_m=float(margin))


def overhead_boxes(env, top_z, *, min_gap=OVERHEAD_MIN_GAP_M, max_above=OVERHEAD_MAX_M):
    """World xy boxes and undersides of every fixture hanging above a surface at ``top_z``: wall cabinets,
    hoods, microwaves, open shelves, wall accessories. Fixtures reaching down to within ``min_gap`` of the
    top are decor (footprint keep-outs), the room shell and the counters themselves are skipped."""
    boxes = []
    top_z = float(top_z)
    for name, fixture in (getattr(env, 'fixtures', {}) or {}).items():
        kind = type(fixture).__name__
        if kind in NOT_OVERHEAD_CLASSES:
            continue
        try:
            points = np.asarray(fixture.get_bbox_points(), float)
        except Exception:  # noqa: BLE001  (a fixture without a bounding region)
            continue
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            continue
        z0, z1 = float(points[:, 2].min()), float(points[:, 2].max())
        if z0 <= top_z + min_gap or z0 > top_z + max_above:
            continue
        x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
        y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
        if x1 - x0 < MIN_OVERHEAD_SIDE_M or y1 - y0 < MIN_OVERHEAD_SIDE_M:
            continue                # flat on the wall plane (switches, outlets, pictures): nothing hangs over the top
        boxes.append(dict(name=name, kind=kind, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1))
    boxes.sort(key=lambda b: b['z0'])
    return boxes


def lowest_overhead(boxes, xy, radius):
    """The overhead box with the lowest underside whose xy box overlaps the disc, or None."""
    cx, cy, r = float(xy[0]), float(xy[1]), float(radius)
    best = None
    for box in boxes:
        dx = max(box['x0'] - cx, cx - box['x1'], 0.)
        dy = max(box['y0'] - cy, cy - box['y1'], 0.)
        if dx * dx + dy * dy < r * r and (best is None or box['z0'] < best['z0']):
            best = box
    return best


def headroom_gate(boxes, xy, radius, needed, top_z, *, pad=HAND_PAD_M):
    """Does the column over the disc (``radius`` + ``pad``) have ``needed`` metres of clear air above
    ``top_z``? ``needed`` may be the dict :func:`headroom_needed` returns. Returns ``dict(passed,
    needed_m, available_m, fixture, fixture_kind, column, reasons)``; ``available_m`` is None when nothing
    hangs over the column."""
    need = float(needed['needed_m'] if isinstance(needed, dict) else needed)
    column = dict(xy=[round(float(xy[0]), 4), round(float(xy[1]), 4)], radius=round(float(radius) + float(pad), 4))
    hit = lowest_overhead(boxes, xy, column['radius'])
    available = None if hit is None else round(hit['z0'] - float(top_z), 4)
    passed = hit is None or available >= need
    reasons = []
    if not passed:
        reasons.append(f"no headroom: {hit['name']} ({hit['kind']}) underside {available:.2f} m above the top, "
                       f'{need:.2f} m needed')
    return dict(passed=bool(passed), needed_m=round(need, 4), available_m=available,
                fixture=None if hit is None else hit['name'], fixture_kind=None if hit is None else hit['kind'],
                column=column, reasons=reasons)


def path_headroom_gate(boxes, points, radius, needed, top_z, *, pad=HAND_PAD_M):
    """:func:`headroom_gate` along a path: the tightest of the gates at ``points`` (world xy)."""
    worst = None
    for point in points:
        gate = headroom_gate(boxes, point, radius, needed, top_z, pad=pad)
        if worst is None or (gate['available_m'] is not None
                             and (worst['available_m'] is None or gate['available_m'] < worst['available_m'])):
            worst = gate
    if worst is None:                       # no points: nothing to clear
        worst = headroom_gate([], (0., 0.), radius, needed, top_z, pad=pad)
    return worst


def path_points(start, end, step=.05):
    """World xy samples from ``start`` to ``end`` every ``step`` metres (both ends included)."""
    a, b = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
    n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / float(step))))
    return [a + (b - a) * t for t in np.linspace(0., 1., n + 1)]
