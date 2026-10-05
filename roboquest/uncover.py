"""Uncover and look skills for RoboQuest oracles (spec 2.2, build brief C6, wave 2).

Every skill here is a function taking a bound :class:`~roboquest.motor.Skills`
(``Skills`` exposes them as methods of the same name), works in world coordinates, raises
:class:`~roboquest.motor.SkillFailure` when its postcondition fails and leaves the
hand open, raised and tucked ahead of the base afterwards (``Skills.carry(-1.)``), so the next
skill starts from a sane state. The records the skills read are the ones
:func:`roboquest.observe.apply_observability` stores in
``env.task_spec['observability']`` (``covers[i]`` with ``kind``, ``name``, ``object`` or ``place``,
``radius``, ``half``, ``reach`` = the gate result; ``annex[i]`` with ``stance``; ``occluders[i]``).

Skills
------
* :func:`lift_cloche` - pinch the knob site ``{cover}_knob`` from above, lift the dome clear of
  the covered object, carry it to a free spot on the same surface, set it down, release.
* :func:`slide_lift_bowl` - an inverted bowl has no grasp feature: push it along the gated
  slide direction to the counter edge until a third of it overhangs (the covered object is
  pushed along by the trailing wall and stays inside), pinch the overhanging rim from outside
  the counter with vertical fingers (one pad under the rim), lift, carry to a free spot, set
  down.
* :func:`push_cover` - push a plate, board or tray off a pad or card along the gated
  direction with the closed hand, keeping it on the surface.
* :func:`lift_plate_from_pan` - pinch the rim of a plate lying on a balance pan where it
  overhangs the pan toward the robot, lift it straight up without swinging the beam, carry
  it to a free spot, set it down.
* :func:`pour_cup` - grasp the wide cup by its handle bar from the side, lift, hold it over a
  free spot, roll the wrist about the handle axis past 120 degrees so the cube slides out,
  level the cup, set it down.
* :func:`drive_to_stance` / :func:`look_at` / :func:`fetch` - the look level: drive the base to
  an annex stance ``{xy, yaw, edge, standoff}`` along a planned path, grasp the object there
  and bring it back onto the work frame.
* :func:`move_aside` - take a tall occluder out of what it hides: pinch it across its thinnest
  horizontal side and set it down on a free spot, or, when that side is wider than the gripper
  opens, push it out of the target's line of sight along a free heading (:func:`push_aside_plan`).
* :func:`resolve_observability` - do all of the above for every entry of ``spec['hidden']``,
  so any oracle can call it first and run its normal routine afterwards.

Free spots for setting a cover down come from :func:`free_spot`: a grid search over the work
top (task frame) that keeps the cover clear of every task geom on the surface, of the counter
decor and of the surface edge, inside the band of the counter the arm reaches from the reset
stance, nearest to where the cover was. :func:`set_down_spot` adds the overhead fixtures
(:mod:`headroom`): a spot whose column lacks the skill's headroom, or whose carry path from the
hand passes under a fixture at the carry height, is never chosen; :func:`_tuck` bounds the carry
the same way.

Measured limits: robosuite's Panda fingers
are position actuators (kp 1000 N/m, 20 N cap), so a pinch presses with kp times the half width of
what it holds: 7 N on the cup's 14 mm grip bar, 2 N on its 4 mm wall, 20 N on the bowl's edge. The
fingertip meshes reach 8.8 mm below the pads, which is why a plate's 5-13 mm lip and an inverted
bowl's rim cannot be pinched from above (the inner fingertip rests on the dish or the dome first);
the bowl is pinched from the side instead and the plate is refused with its measured lip.
"""
from copy import deepcopy
import math
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest import headroom as H
from roboquest import observe as O
from roboquest import scatter as S
from roboquest.motor import CARRY_ABOVE_TOP_M, CARRY_AHEAD_M, GRASP_AHEAD_M, ROBOT_PREFIXES, SkillFailure, Submitted, _unit

__all__ = ['free_spot', 'free_spots', 'overhang_fraction', 'tilt_deg', 'pour_landing', 'body_pose', 'body_aabb',
           'surface_keepouts', 'submit_keepout', 'overhead_shapes', 'overhead_keepouts', 'carry_limit',
           'set_down_spot', 'cover_record', 'surface_top_z', 'lift_cloche', 'slide_lift_bowl', 'push_cover',
           'lift_plate_from_pan', 'pour_cup', 'drive_to_stance', 'look_at', 'fetch', 'move_aside',
           'resolve_observability', 'sight_clearance', 'push_aside_plan', 'occluder_record']

SPOT_GAP_M = .015            # clearance between a set-down cover and everything else on the surface
SPOT_EDGE_M = .02            # a set-down cover stays this far inside the surface
SUBMIT_HALF_M = (.05, .045)  # the Submit button's own plan half-extent (the base box plus 2 mm)
SUBMIT_PAD_M = H.HAND_PAD_M  # ... inflated by the hand that has to come down on the cap (submit_keepout)
SUBMIT_PALM_REACH_M = .092   # measured: how far the closed hand's collision hull reaches from the grip site
                             # while it presses the cap (odd_parcel 07fe1c189e3c61f2 85 mm, 2bbb350a46b29e01
                             # 91 mm, 2026-09-22), so a prop whose edge is nearer than this to the button
                             # centre is in the way of every press
REACH_BAND_M = (.28, .60)    # frame-y band ahead of the reset base a set-down spot may use
SLIDE_STEP_M = .04           # one push increment
PUSH_STALL_STEPS = 3         # increments without progress before a push fails
BOWL_TARGET_OVERHANG = 1. / 3.
BOWL_PINCH_INSET_M = .02         # the side pinch of an inverted bowl closes this far inside its rim (bowl_probe, 2026-09-21)
BOWL_PINCH_ABOVE_TOP_M = .02     # ... with the hand this far above the counter top: the lower finger under the rim
POUR_TILT_DEG = 125.
POUR_TILT_MAX_DEG = 145.
POUR_HOLD_M = .30            # the hand this far above the top while the cup pours: its body swings down under the pinch
POUR_BAR_RADIAL_M = .053     # the pinch on the upper handle bar, this far from the cup's axis along the handle (the wall
                             # ends at .046, the grip bar starts at .070; assets.cup)
POUR_BAR_TOP_M = .0925       # the upper bar's top above the cup's base (assets.cup BAR_Z[1] + BAR_HALF[2])
POUR_BAR_PINCH_DZ = -.003    # the pads straddle the bar's top 3 mm and below (the pad reaches 7.9 mm under the grip site)
POUR_TILT_STEPS = (40., 80., 110., 125.)   # tool turns about the finger axis; the cube left at 114-116 deg of cup tilt
SIDE_PINCH_AHEAD_M = .58     # a horizontal tool needs its wrist clear of the robot's body: side pinches stand back
LEFT_SURFACE = 'left_surface'
CARRY_ALLOWANCE_M = H.ALLOWANCE_M['cloche']   # the forearm above a top-down hand while it carries (headroom.MEASURED,
                                              # the cloche phases up to the carry retract: peak .296 above the hand)
CARRY_COLUMN_M = .05         # radius of the hand's own column for the carry headroom (the hand's half width)
DOCK_BACK_M = .45            # a stance is entered square: the base turns to the stance heading this far out and drives in
DOCK_MAX_TURN_RAD = math.radians(35.)
# Panda fingertips (robosuite panda_gripper.xml, measured world boxes at the tool-down rest pose): the pad boxes reach
# .0079 m below the grip site and the finger meshes .0167 m, so a pinch from above needs the lip it closes on to stand
# at least this much above whatever the inner fingertip would rest on (measured 2026-09-21, plate_probe on odd_parcel 25).
FINGERTIP_BELOW_PAD_M = .0088
PAD_BELOW_EEF_M = .0079
PLATE_PITCH_DEG = 55.          # the tool z this far below the horizontal: the pinch that takes the plate
PLATE_PITCH_TRIES = (dict(pitch_deg=55.), dict(pitch_deg=45.), dict(pitch_deg=90.))   # ... and its retries
PLATE_PITCH_GAP_M = .020       # the pads straddle the rim this far apart when the tool is pitched
PLATE_PITCH_INSET_M = .003     # ... around a point this far inside the plate's collision edge
BOX_SAMPLES = 9                # points per axis when a box geom stands in for a hull (geom_points)
PLATE_RIM_CLEAR_M = .008       # the outer finger comes down this far outboard of the plate's collision edge
PLATE_RIM_BITE_M = .024        # ... and the inner finger this far inside it, so the rim sits between the tips
PLATE_MIN_OVERHANG_M = .008    # the plate must stand this far out over its pan for the outer finger to pass it
PLATE_MIN_RIM_DROP_M = .002    # and its rim must stand this far over the surface the inner finger lands on
PLATE_RIM_LAND_M = .0008       # the descent stops with the inner finger this close to the plate
PLATE_PEEL_PULL_M = .004       # the peel pulls the plate this far toward the robot per step, barely rising: the
PLATE_PEEL_RISE_M = .002       # plate slides off its far rim (a plate of radius R on rims 2R apart needs only
PLATE_PEEL_STEPS = 22          # R - rim a few millimetres of travel) instead of levering out of the fingertips
PLATE_LIFT_STEP_M = .008       # the lift then goes up in steps this tall, each with a hold, the fingers closing
PLATE_LIFT_PULL_M = .001
PLATE_LIFT_STEPS = 14
PLATE_OFF_PAN_M = .005         # ... or this far from the pan's rims: a plate dragged out on edge hangs below
PLATE_LIFT_CLEAR_M = .020      # the lift stops here: every further step tips the plate out of the fingertips
PLATE_CARRY_HOP_M = .030      # the carry moves the plate in hops this long: every stop and start slides it a
                              # little in the pinch and the fingertips hold for about twenty of them (certify 02:34)
PLATE_DROP_MARGIN_M = .005     # the set-down spot takes the plate's own radius plus this
PLATE_DROP_GAP_M = .010        # ... and stands this far from everything else on the top (plan footprints)
PLATE_CARRY_ROOM_M = .040      # the plate drops to the low carry with this much room all round it in plan
PLATE_CARRY_MID_M = .075       # ... or this far up while it is only clear of the balance (a parcel is 50 mm)
PLATE_CARRY_LOW_M = .012       # once clear of the balance the plate rides this far over the counter, so a
PLATE_FREE_M = .005            # shear drops it flat instead of over a rim; it must end this far from the balance
PLATE_ON_TOP_M = .030          # ... with its underside no higher than this over the work top
PLATE_FLAT_DEG = 20.           # ... and no more tilted than this
PLATE_RIM_STANCES = (None, .50, .56)   # stances tried for the pinch: the reset one, then closer, then further back
GRIPPER_OPENING_M = .080       # the Panda's finger travel (robosuite panda_gripper.xml: two 0.04 m slides)
# An occluder whose thinnest horizontal side does not fit between the fingers is pushed aside instead of
# pinched (``push_aside_plan`` chooses the heading, ``_push_aside`` runs it). The policy's own skill set has
# ``push_cover``, so a wide occluder is content a policy can handle, not a refusal.
ASIDE_PUSH_HEADINGS = 24       # headings tried around the circle
ASIDE_PUSH_MAX_M = .35         # the longest push one of them may need
ASIDE_PUSH_STEP_M = .02        # distances (and the path samples along them) are searched in these increments
ASIDE_PUSH_PASS_M = .08        # one pass: the hand pushes this far, retreats, re-reads the body and re-aims
ASIDE_PUSH_PASSES = 8          # at most this many passes
ASIDE_PUSH_CONTACT_M = .02     # the fingertips push this far above the top: low, well under a 0.22 m body's
                               # centre of mass, so it slides instead of tipping
ASIDE_PUSH_BEHIND_M = .05      # the hand comes down this far behind the body's hull along the heading: the
                               # closed fingers need room to pass the body on the way down, or they catch its
                               # shoulder and the push tips it (measured on stamps layout 7 style 13)
ASIDE_SIGHT_MARGIN_M = .03     # the pushed body ends this far outside every sight corridor
ASIDE_APPROACH_GAP_M = H.HAND_PAD_M   # ... and this far from the target's own column: the hand comes down on
                                      # the target next (fetch, or the oracle's grasp) and needs that column
ASIDE_TILT_MAX_DEG = 15.       # more lean than this is a topple, not a push
ASIDE_TILT_STOP_DEG = 6.       # ... and this much ends the pass: the hand backs off and lets the body settle


# ---- pure geometry ---------------------------------------------------------------------------------
def overhang_fraction(center_along, edge_along, radius):
    """Fraction of a disc's diameter beyond an edge line, where ``center_along`` and ``edge_along`` are
    coordinates along the outward direction: 0 when the disc is inside, 1/3 when its centre is a third
    of the radius inside the edge (the slide-and-lift stop), 1 when it is fully beyond."""
    radius = float(radius)
    inside = float(edge_along) - float(center_along)          # centre to edge, positive inside
    return float(np.clip((radius - inside) / (2. * radius), 0., 1.))


def tilt_deg(rotation):
    """Angle in degrees between a body's own z axis and world up (0 upright, 180 upside down)."""
    up = np.asarray(rotation, float)[:, 2]
    return float(math.degrees(math.acos(float(np.clip(up[2] / max(np.linalg.norm(up), 1e-9), -1., 1.)))))


def pour_landing(cup_center_xy, mouth_dir_xy, drift=.10):
    """Where a cube leaving a tilted cup lands: the cup's centre pushed ``drift`` along the horizontal
    direction the mouth points to (the cup's half height plus the fall). Inverse of the hold point."""
    d = np.asarray(mouth_dir_xy, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.asarray(cup_center_xy, float).copy()
    return np.asarray(cup_center_xy, float) + d / n * float(drift)


def free_spots(rect, keepouts, radius, *, band=None, prefer=None, grid=.02, gap=SPOT_GAP_M, edge=SPOT_EDGE_M, limit=None):
    """Task-frame centres of discs of ``radius`` inside ``rect`` (a scatter Rect), clear of ``keepouts`` (scatter
    shapes) by ``gap``, ``edge`` inside the rectangle, with their y inside ``band`` when given, nearest to
    ``prefer`` first (grid order without a preference); at most ``limit`` of them."""
    inner = S.Rect(rect.x0 + radius + edge, rect.x1 - radius - edge, rect.y0 + radius + edge, rect.y1 - radius - edge)
    if band is not None:
        inner = S.Rect(inner.x0, inner.x1, max(inner.y0, float(band[0])), min(inner.y1, float(band[1])))
    if inner.empty:
        return []
    keeps = [S.as_shape(k) for k in keepouts]
    xs = np.arange(inner.x0, inner.x1 + 1e-9, grid)
    ys = np.arange(inner.y0, inner.y1 + 1e-9, grid)
    prefer = None if prefer is None else np.asarray(prefer, float)
    points = [(float(x), float(y)) for x in xs for y in ys]
    if prefer is not None:
        points.sort(key=lambda p: float(np.hypot(p[0] - prefer[0], p[1] - prefer[1])))
    found = []
    for x, y in points:
        disc = S.Disc((x, y), float(radius))
        if all(S.clearance(disc, k) >= gap for k in keeps):
            found.append((x, y))
            if limit is not None and len(found) >= int(limit):
                break
    return found


def free_spot(rect, keepouts, radius, *, band=None, prefer=None, grid=.02, gap=SPOT_GAP_M, edge=SPOT_EDGE_M):
    """The task-frame centre of a disc of ``radius`` inside ``rect`` (a scatter Rect), clear of ``keepouts``
    (scatter shapes) by ``gap``, ``edge`` inside the rectangle, with its y inside ``band`` when given,
    nearest to ``prefer``; None when no grid point qualifies."""
    found = free_spots(rect, keepouts, radius, band=band, prefer=prefer, grid=grid, gap=gap, edge=edge, limit=1)
    return found[0] if found else None


def sight_clearance(spot_xy, radius, target_xy, target_radius, sight_dirs):
    """Metres between a body of ``radius`` standing at ``spot_xy`` and the nearest sight corridor: the distance
    from its footprint disc to the ray that leaves the target toward a camera, past the target's own silhouette
    (the corridor is modelled at the target's full width all the way to the camera, which is wider than the real
    wedge, so a positive number is conservative). Negative while the body still stands in a corridor; ``inf``
    when no camera needs the target. All task-frame."""
    spot = np.asarray(spot_xy, float)[:2]
    rel = spot - np.asarray(target_xy, float)[:2]
    best = math.inf
    for direction in sight_dirs:
        d = _unit([float(direction[0]), float(direction[1])])
        along = float(rel @ d)
        across = float(np.linalg.norm(rel - along * d)) if along > 0. else float(np.linalg.norm(rel))
        best = min(best, across - float(radius) - float(target_radius))
    return best


def _push_reach_blocked(base, points):
    """Why the arm cannot work every task-frame point of a push, or None.

    ``base`` is ``(xy, heading)`` of the robot base in the task frame. The base slides along its own lateral
    axis and never drives closer to the surface (:func:`observe.reach_gate`'s model, the one
    :func:`within_top_down_reach` applies point by point), and the push re-stances before every pass, so the
    test is that envelope on each point rather than one stance for the whole push."""
    origin = np.asarray(base[0], float)[:2]
    h = _unit([float(base[1][0]), float(base[1][1])])
    left = np.array([-h[1], h[0]])
    rel = [np.asarray(p, float)[:2] - origin for p in points]
    ahead = [float(r @ h) for r in rel]
    lateral = [float(r @ left) for r in rel]
    if max(ahead) > O.GRASP_REACH_M:
        return 'beyond the arm from where the base can stand'
    if min(ahead) < O.REACH_MIN_AHEAD_M:
        return 'under the base from where it can stand'
    if max(abs(min(lateral)), abs(max(lateral))) > O.MAX_SLIDE_M:
        return 'the base cannot slide that far along the counter'
    return None


def push_aside_plan(occluder_xy, radius, target_xy, target_radius, sight_dirs, rect, keepouts, *, base=None,
                    sight_test=None, footprint=None,
                    headings=ASIDE_PUSH_HEADINGS, max_distance=ASIDE_PUSH_MAX_M, step=ASIDE_PUSH_STEP_M,
                    gap=SPOT_GAP_M, edge=SPOT_EDGE_M, sight_margin=ASIDE_SIGHT_MARGIN_M,
                    approach_gap=ASIDE_APPROACH_GAP_M, behind=ASIDE_PUSH_BEHIND_M):
    """Task-frame heading and distance that push an occluder out of its target's line of sight; pure geometry.

    ``occluder_xy`` / ``radius``: where the body stands and the disc that holds its footprint (the pinch's own
    measure, :func:`footprint_radius`, plus its margin). ``target_xy`` / ``target_radius``: the object it
    hides. ``sight_dirs``: unit directions from the target toward each camera that must see it. ``rect``: the
    work surface (a scatter Rect). ``keepouts``: everything else standing on it (scatter shapes, the occluder
    itself left out), the way :func:`set_down_spot` collects them. ``base``: ``(xy, heading)`` of the robot
    base in the task frame, or None to skip the reach test. ``sight_test``: a callable taking a task-frame
    spot and returning how many of the target's sight rays a body standing there would leave clear
    (:func:`observe.rays_clear` over the cameras that need the target, the body inflated by ``sight_margin``:
    what :func:`_push_aside` passes, and the measure the postcondition uses). Without one the corridor
    estimate :func:`sight_clearance` is used instead, which is conservative by the occluder's own radius.
    ``footprint``: ``(half_x, half_y, yaw)`` of the body's own footprint rectangle in the task frame, used for
    the surface tests in place of the disc. A push slides the body, it never turns it, so the rectangle
    travels rigidly and there is no reason to reserve the disc around it - on a crowded top the difference is
    the whole search (the disc around a 166 mm wide jug is 126 mm, its own half-width is 83).

    A heading qualifies at the first distance where the body ends inside the surface (``edge`` in), ``gap``
    clear of everything else on it, ``approach_gap`` clear of the target's own column (the hand comes down on
    the target next), out of the line of sight (``sight_test`` positive, or ``sight_margin`` outside every
    corridor), with the swept path free and every contact point inside the arm's envelope. The heading stops
    as soon as the swept body would hit something or leave the surface: pushing further only drives it deeper.

    The shortest qualifying push wins, the way the pinch's set-down spot is the free spot nearest to where the
    cover stood; ties go to the clearest view of the target and then to the lower heading index, so the choice
    is deterministic. Returns ``dict(passed, heading, distance_m, spot, contact, sight_clearance_m,
    sight_rays_clear, target_gap_m, surface_gap_m, options, tried, counts, reasons)``."""
    occluder_xy = np.asarray(occluder_xy, float)[:2]
    target_xy = np.asarray(target_xy, float)[:2]
    radius, target_radius = float(radius), float(target_radius)
    keeps = [S.as_shape(k) for k in keepouts]
    dirs = [_unit([float(d[0]), float(d[1])]) for d in (sight_dirs or [])]
    steps = max(int(round(float(max_distance) / float(step))), 1)

    def standing_at(xy):
        # the body's footprint standing at ``xy``: its own rectangle when one was given, else the disc
        if footprint is None:
            return S.Disc((float(xy[0]), float(xy[1])), radius)
        return S.Box((float(xy[0]), float(xy[1])), (float(footprint[0]), float(footprint[1])), float(footprint[2]))

    # the body may already stand closer to one of its neighbours than a set-down would ever be parked (its
    # footprint is the pinch's conservative disc): per keep-out, the path may not make that contact worse,
    # and the spot it ends on is asked for the full ``gap`` only where it had it to begin with
    start = standing_at(occluder_xy)
    floors = [min(float(gap), float(S.clearance(start, k))) for k in keeps]
    walls = [min(0., float(S.clearance(start, k))) for k in keeps]
    options, counts = [], {}

    def note(reason):
        counts[reason] = counts.get(reason, 0) + 1

    for index in range(int(headings)):
        angle = 2 * math.pi * index / int(headings)
        d = np.array([math.cos(angle), math.sin(angle)])
        contact = occluder_xy - d * (radius + float(behind))
        for k in range(1, steps + 1):
            distance = float(step) * k
            spot = occluder_xy + d * distance
            disc = standing_at(spot)
            margin = S.region_margin(disc, rect)
            if margin < 0.:
                note('the surface ends there')
                break
            gaps = [float(S.clearance(disc, k2)) for k2 in keeps]
            surface_gap = min(gaps + [math.inf])
            if any(g < w for g, w in zip(gaps, walls)):
                note('something else stands in the way')
                break
            if any(g < f for g, f in zip(gaps, floors)) or margin < float(edge):
                note('no free surface that far along')
                continue
            clear = sight_clearance(spot, radius, target_xy, target_radius, dirs)
            rays = None if sight_test is None else float(sight_test(spot))
            in_sight = clear < float(sight_margin) if sight_test is None else rays <= 0.
            if in_sight:
                note('still in the line of sight')
                continue
            target_gap = float(np.linalg.norm(spot - target_xy)) - radius - target_radius
            if target_gap < float(approach_gap):
                note("inside the target's own approach column")
                continue
            if base is not None:
                blocked = _push_reach_blocked(base, [contact, contact + d * distance, spot])
                if blocked is not None:
                    note(blocked)
                    continue
            options.append(dict(index=index, heading=[float(d[0]), float(d[1])], distance_m=distance,
                                spot=[float(spot[0]), float(spot[1])],
                                contact=[float(contact[0]), float(contact[1])],
                                sight_clearance_m=float(clear), sight_rays_clear=rays,
                                target_gap_m=float(target_gap), surface_gap_m=float(surface_gap)))
            break
    reasons = [r for r, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:3]
    if not options:
        return dict(passed=False, heading=None, distance_m=None, spot=None, contact=None,
                    sight_clearance_m=None, sight_rays_clear=None, target_gap_m=None, surface_gap_m=None,
                    options=[], tried=int(headings), counts=counts, reasons=reasons)
    best = min(options, key=lambda o: (round(o['distance_m'], 3), -(o['sight_rays_clear'] or 0.),
                                       -round(o['sight_clearance_m'], 3), o['index']))
    return dict(passed=True, options=options, tried=int(headings), counts=counts, reasons=[], **best)


def overhead_shapes(boxes, top_z, needed_m, work_yaw, center_world, *, pad=H.HAND_PAD_M):
    """Task-frame keep-out boxes for the overhead fixtures (:func:`headroom.overhead_boxes` records) whose
    underside is less than ``needed_m`` above ``top_z``: their world xy boxes inflated by ``pad`` (the hand
    straddles the point it works at), expressed in the frame at ``center_world`` yawed by ``work_yaw``."""
    shapes = []
    c, s = math.cos(float(work_yaw)), math.sin(float(work_yaw))
    origin = np.asarray(center_world, float)[:2]
    for box in boxes or []:
        if float(box['z0']) - float(top_z) >= float(needed_m):
            continue
        centre = np.array([(box['x0'] + box['x1']) / 2, (box['y0'] + box['y1']) / 2], float) - origin
        local = np.array([c * centre[0] + s * centre[1], -s * centre[0] + c * centre[1]])
        half = ((box['x1'] - box['x0']) / 2 + float(pad), (box['y1'] - box['y0']) / 2 + float(pad))
        shapes.append(S.Box((float(local[0]), float(local[1])), (float(half[0]), float(half[1])), -float(work_yaw)))
    return shapes


def overhead_keepouts(env, needed_m, boxes=None, *, pad=H.HAND_PAD_M):
    """:func:`overhead_shapes` for the work top of ``env``: what a set-down spot for a skill needing ``needed_m``
    of clear air (a float or a :func:`headroom.headroom_needed` dict) must keep clear of."""
    need = float(needed_m['needed_m'] if isinstance(needed_m, dict) else needed_m)
    top = float(env.work['top_z'])
    boxes = H.overhead_boxes(env, top) if boxes is None else boxes
    return overhead_shapes(boxes, top, need, float(env.work['yaw']), env.work['center_world'], pad=pad)


def carry_limit(boxes, xy, top_z, *, allowance=CARRY_ALLOWANCE_M, margin=H.HEADROOM_MARGIN_M, radius=CARRY_COLUMN_M,
                pad=H.HAND_PAD_M):
    """Highest hand height (world z) at which a top-down hand at world ``xy`` keeps its forearm under the lowest
    overhead fixture over its column; ``inf`` when nothing hangs there."""
    hit = H.lowest_overhead(boxes or [], xy, float(radius) + float(pad))
    if hit is None:
        return math.inf
    return float(hit['z0']) - float(allowance) - float(margin)


# ---- simulator readers -----------------------------------------------------------------------------
def body_pose(env, name):
    """World position and rotation matrix of a task body (a registered object's root body or a body name)."""
    bid = O.body_id(env, name)
    data = env.sim.data
    return np.asarray(data.body_xpos[bid], float).copy(), np.asarray(data.body_xmat[bid], float).reshape(3, 3).copy()


def body_geom_ids(env, name):
    return O._subtree_geoms(env.sim.model, O.body_id(env, name))


def collision_geom_names(env, name):
    model = env.sim.model
    return [model.geom_id2name(g) for g in body_geom_ids(env, name) if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0]


def body_aabb(env, name, collision_only=True):
    """World axis-aligned bounds ``(lo, hi)`` of a body's geoms from MuJoCo's per-geom boxes."""
    model, data = env.sim.model, env.sim.data
    raw = model._model
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in body_geom_ids(env, name):
        if collision_only and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        centre, half = raw.geom_aabb[g, :3], raw.geom_aabb[g, 3:]
        R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g]) + R @ centre
        ext = np.abs(R) @ half
        lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
    if not np.all(np.isfinite(lo)):
        return body_aabb(env, name, collision_only=False) if collision_only else (lo, hi)
    return lo, hi


def footprint_radius(env, name):
    """Radius of the smallest vertical cylinder about a body's origin that holds its collision geoms, the smaller
    of the box estimate (:func:`body_aabb`, which grows by up to root 2 for a yawed cylinder: a stamp's 6 cm sole
    read 8.6 cm and its set-down found no free spot on stamps layout 6 style 11) and the bounding-sphere one."""
    model, data = env.sim.model, env.sim.data
    centre = np.asarray(data.body_xpos[O.body_id(env, name)], float)[:2]
    lo, hi = body_aabb(env, name)
    by_box = float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2)
    by_sphere = 0.
    for g in body_geom_ids(env, name):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        offset = float(np.linalg.norm(np.asarray(data.geom_xpos[g], float)[:2] - centre))
        by_sphere = max(by_sphere, offset + float(model.geom_rbound[g]))
    return min(by_box, by_sphere) if by_sphere > 0. else by_box


def fingertip_drop(skills):
    """How far the lowest gripper geom hangs below the hand site right now (metres)."""
    env = skills.env
    model, data = env.sim.model, env.sim.data
    raw = model._model
    eef_z = float(skills.state()['eef'][2])
    lowest = eef_z
    for g in range(model.ngeom):
        name = model.geom_id2name(g) or ''
        if not name.startswith('gripper0') or model.geom_contype[g] == 0:
            continue
        centre, half = raw.geom_aabb[g, :3], raw.geom_aabb[g, 3:]
        R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        z = float((np.asarray(data.geom_xpos[g]) + R @ centre)[2] - (np.abs(R) @ half)[2])
        lowest = min(lowest, z)
    return eef_z - lowest


def surface_keepouts(env, exclude=(), z_band=(-.03, .45), plan=False):
    """Task-frame shapes of every task geom standing on the work top (frame bodies, registered objects,
    every free prop), the counter decor and the Submit button, leaving out the bodies in ``exclude``
    (names). Discs from MuJoCo's geom bounding radii, so they are conservative.

    ``plan=True`` measures each geom's own plan footprint instead (:func:`geom_plan_box`). A parcel 0.12 by
    0.08 by 0.05 has a bounding disc of radius 0.077 where its footprint is 0.06 by 0.04, and six of those
    plus the answer box close the counter of odd_parcel layout 25 to a plate that in truth fits."""
    model, data = env.sim.model, env.sim.data
    top = float(env.work['top_z'])
    excluded = set()
    for name in exclude:
        try:
            excluded.update(body_geom_ids(env, name))
        except Exception:
            continue
    bodies = set(env.task_body_ids()) | set(env._prop_bodies().values())
    shapes = list(O.decor_shapes_on_work(env))
    for g in range(model.ngeom):
        if g in excluded or int(model.geom_bodyid[g]) not in bodies:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        pos = np.asarray(data.geom_xpos[g], float)
        r = float(model.geom_rbound[g])
        if not (top + z_band[0] <= pos[2] + r and pos[2] - r <= top + z_band[1]):
            continue
        if plan:
            shapes.append(geom_plan_box(env, g))
            continue
        c = O.world_to_frame_xy(env, pos[:2])
        shapes.append(S.Disc((float(c[0]), float(c[1])), r))
    if hasattr(env, 'submit_xy'):
        # The Submit button is the one prop the robot must come down *onto* with the whole hand and cannot
        # move out of the way, so its keep-out is its footprint inflated by the hand, the way
        # :func:`overhead_shapes` inflates the fixtures overhead. Measured 2026-09-22 on odd_parcel
        # 07fe1c189e3c61f2 / 2bbb350a46b29e01: the unpadded box let ``set_down_spot`` park a cloche with its
        # wall 30 mm from the button's footprint, and the palm (``gripper0_right_hand_collision`` reaches
        # 85 mm from the grip site) then hit the dome at 6-10 N and stalled the press 28 mm above the cap -
        # the cap never moved and no submission was recorded.
        shapes.append(submit_keepout(O.world_to_frame_xy(env, env.submit_xy)))
    return shapes


def submit_keepout(center_xy):
    """The Submit button's keep-out for a set-down: its own footprint inflated by :data:`SUBMIT_PAD_M`.

    Nothing may be parked inside it, because the hand that presses the cap needs the column around the
    button, not only the button's own footprint (:data:`SUBMIT_PALM_REACH_M`)."""
    return S.Box((float(center_xy[0]), float(center_xy[1])),
                 (SUBMIT_HALF_M[0] + SUBMIT_PAD_M, SUBMIT_HALF_M[1] + SUBMIT_PAD_M), 0.)


def geom_plan_box(env, geom_id):
    """Task-frame plan footprint of one geom - its own box, capsule, cylinder or sphere projected onto the work
    top, not the bounding disc :func:`surface_keepouts` takes from ``geom_rbound``.

    A balance's hanger rod is a capsule 0.26 m long and 4 mm thick standing on end: ``geom_rbound`` calls it a
    disc of radius 0.136 m, and four of those swallow the whole counter around the pans. Measured on odd_parcel
    layout 25 (2026-09-22): no free spot for the plate within 0.55 m of the pan although the counter beside it
    is empty; with the plan footprints the same search finds one 0.23 m away. Meshes keep the bounding disc.
    """
    model, data = env.sim.model, env.sim.data
    kind, size = int(model.geom_type[geom_id]), np.asarray(model.geom_size[geom_id], float)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        half = size[:3].copy()
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = np.array([size[0], size[0], size[1] + size[0]])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        half = np.array([size[0], size[0], size[1]])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        half = size[:3].copy()
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.array([size[0]] * 3)
    else:                                       # a mesh or a plane: keep MuJoCo's bounding disc
        c = O.world_to_frame_xy(env, np.asarray(data.geom_xpos[geom_id], float)[:2])
        return S.Disc((float(c[0]), float(c[1])), float(model.geom_rbound[geom_id]))
    rot = np.asarray(data.geom_xmat[geom_id], float).reshape(3, 3)
    pos = np.asarray(data.geom_xpos[geom_id], float)
    corners = np.array([O.world_to_frame_xy(env, (pos + rot @ (half * np.array(sign)))[:2])
                        for sign in ((-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
                                     (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1))], float)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return S.Box((float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)),
                 (float((hi[0] - lo[0]) / 2), float((hi[1] - lo[1]) / 2)), 0.)


def plan_keepouts(env, bodies):
    """:func:`geom_plan_box` for every collision geom of ``bodies`` (names), the ones standing over the top."""
    model = env.sim.model
    shapes = []
    for name in bodies:
        for g in body_geom_ids(env, name):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            shapes.append(geom_plan_box(env, g))
    return shapes


def linked_bodies(env, names):
    """Every task body whose name shares the first underscore-separated token with one of ``names``: the pans,
    beam and base of a balance from the pan the cover rests on."""
    model = env.sim.model
    prefixes = {str(n).split('_')[0] for n in names if n}
    found = set()
    for body in set(env.task_body_ids()) | set(env._prop_bodies().values()):
        name = model.body_id2name(int(body)) or ''
        if name.split('_')[0] in prefixes:
            found.add(name)
    return sorted(found | {n for n in names if n})


def plate_landed(free_m, bottom_above_top_m, tilt, *, clear=PLATE_FREE_M):
    """Why a set-down plate is not an acceptable outcome, or None when it is: it has to stand clear of the
    balance (``free_m``, the gap from its hulls to the balance's) and lie flat on the work top. Where the
    fingers lost it on the way does not matter - only where it came to rest (the pinch holds a 3 mm lip, so a
    sideways hop can shear it out and still leave it flat on the counter, clear of the pan)."""
    if free_m is not None and free_m < clear:
        return (f'it came to rest {free_m * 1000:.0f} mm from the balance, less than the '
                f'{clear * 1000:.0f} mm that keeps the pan free to swing')
    if bottom_above_top_m > PLATE_ON_TOP_M:
        return f'it came to rest {bottom_above_top_m * 1000:.0f} mm over the counter, on top of something'
    if bottom_above_top_m < -.02:
        return f'it fell {-bottom_above_top_m * 1000:.0f} mm below the counter top'
    if tilt > PLATE_FLAT_DEG:
        return f'it came to rest tilted {tilt:.0f} degrees, not lying flat'
    return None


def _reach_from(base_xy, base_yaw, point_world):
    """(ahead, lateral) metres of a world point from a base frame at ``base_xy`` facing ``base_yaw``."""
    f = np.array([math.cos(float(base_yaw)), math.sin(float(base_yaw))])
    left = np.array([-f[1], f[0]])
    rel = np.asarray(point_world, float)[:2] - np.asarray(base_xy, float)[:2]
    return float(rel @ f), float(rel @ left)


def within_top_down_reach(base_xy, base_yaw, point_world):
    """Can a top-down grasp at ``point_world`` be made from a base standing on the line through ``base_xy``
    facing ``base_yaw`` (the base slides sideways, never forward: :func:`observe.reach_gate`'s envelope)?"""
    ahead, lateral = _reach_from(base_xy, base_yaw, point_world)
    return O.REACH_MIN_AHEAD_M <= ahead <= O.GRASP_REACH_M and abs(lateral) <= O.MAX_SLIDE_M


def _stance_for(skills, body, *, phase):
    """Drive to a stance from which ``body`` is within the top-down reach when it is not from where the base
    stands. The look level puts an occluded or annexed object where a viewpoint exists from *some* edge of its
    surface (:func:`observe.look_reachable`), on a deep island the back one: measured on stamps layout 4
    styles 11 and 14, the occluder stood 1.27 m ahead of the reset base and the pre-grasp waypoint missed by
    0.44-0.64 m. Returns the drive record, or None when no drive was needed."""
    env = skills.env
    base_xy, base_yaw = skills.base_pose()
    pos = body_pose(env, body)[0]
    if within_top_down_reach(base_xy, base_yaw, pos):
        return None
    look = O.look_reachable(env, body)
    if not look['passed']:
        ahead, lateral = _reach_from(base_xy, base_yaw, pos)
        raise SkillFailure(phase, f'{body} stands {ahead:.2f} m ahead and {lateral:+.2f} m beside the base and no edge stance '
                           'reaches it (' + '; '.join(look['reasons']) + ')')
    return drive_to_stance(skills, dict(look['stance'], standoff=O.STANDOFF_M), phase=phase + '_drive')


def set_down_spot(skills, radius, *, exclude=(), prefer_world=None, band=None, gap=SPOT_GAP_M, headroom=None,
                  carry_from=None, carry_z=None, from_base=False):
    """World xy of a free spot on the work top for a disc of ``radius``, inside the band the arm reaches
    from the reset stance (frame y in ``REACH_BAND_M`` ahead of the base), nearest to ``prefer_world``.

    ``headroom`` (metres of clear air the set-down needs above the top, or the dict
    :func:`headroom.headroom_needed` returns) keeps the spot from under a wall cabinet, shelf or microwave
    whose underside is lower than that (:func:`overhead_keepouts`): a cover set down there could never be
    lifted again and the arm lowering it would hit the fixture. With ``carry_from`` (world xy where the
    hand carries now) and ``carry_z`` (the carry height) the straight path to the spot is gated too
    (:func:`headroom.path_headroom_gate` at the carry allowance) and the nearest spot with a clear path wins.
    ``from_base`` measures the reach from where the base stands now instead of the reset stance's frame band
    (a skill working from another edge of the surface)."""
    env = skills.env
    rect = O.work_surface_rect(env)
    top = float(env.work['top_z'])
    if from_base:
        band = None
    else:
        band = band or (skills.forward_limit_y + REACH_BAND_M[0], skills.forward_limit_y + REACH_BAND_M[1])
    prefer = None if prefer_world is None else O.world_to_frame_xy(env, prefer_world)
    keeps = surface_keepouts(env, exclude=exclude)
    boxes = H.overhead_boxes(env, top) if (headroom is not None or carry_from is not None) else []
    if headroom is not None:
        keeps = keeps + overhead_keepouts(env, headroom, boxes)
    limit = 600 if from_base else 40

    def reachable(spots):
        if not from_base:
            return spots
        base_xy, base_yaw = skills.base_pose()
        kept = []
        for spot in spots:
            ahead, lateral = _reach_from(base_xy, base_yaw, O.frame_to_world_xy(env, spot))
            if REACH_BAND_M[0] <= ahead <= REACH_BAND_M[1] and abs(lateral) <= .30:
                kept.append(spot)
        return kept[:40]

    candidates = reachable(free_spots(rect, keeps, radius, band=band, prefer=prefer, gap=gap, limit=limit))
    if not candidates:
        candidates = reachable(free_spots(rect, keeps, radius, band=band, prefer=prefer, gap=gap * .5, edge=.01, limit=limit))
    if not candidates:
        raise SkillFailure('set_down_spot', f'no free spot of radius {radius:.3f} m on the work top'
                           + (' clear of the overhead fixtures' if headroom is not None else '')
                           + (' within reach of the base' if from_base else ''))
    if carry_from is not None and boxes:
        need = (float(carry_z) if carry_z is not None else top + CARRY_ABOVE_TOP_M) - top + CARRY_ALLOWANCE_M
        blocked = []
        for spot in candidates:
            world = O.frame_to_world_xy(env, spot)
            gate = H.path_headroom_gate(boxes, H.path_points(carry_from, world), CARRY_COLUMN_M, need, top)
            if gate['passed']:
                return np.array([float(world[0]), float(world[1])])
            blocked.append(gate['reasons'][:1])
        raise SkillFailure('set_down_spot', f'no free spot of radius {radius:.3f} m with a clear carry path: '
                           + '; '.join(r[0] for r in blocked[:2] if r))
    world = O.frame_to_world_xy(env, candidates[0])
    return np.array([float(world[0]), float(world[1])])


def surface_top_z(env, body, xy, *, from_above=.30, tries=8):
    """Height (world z) of a body's collision surface under the vertical line through world ``xy``, by a ray
    cast down from ``from_above`` metres over the surface top: MuJoCo's per-geom boxes overestimate a mesh's
    extent (a plate's box top stood 5.4 mm above its lip in the plate probe), the ray does not. Geoms of
    other bodies in the way are skipped; None when the ray never meets the body."""
    model, data = env.sim.model._model, env.sim.data._data
    wanted = set(body_geom_ids(env, body))
    z = float(env.work['top_z']) + float(from_above)
    origin = np.array([float(xy[0]), float(xy[1]), z])
    geomid = np.array([-1], np.int32)
    for _ in range(int(tries)):
        dist = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), None, 1, -1, geomid)
        if dist < 0. or geomid[0] < 0:
            return None
        hit = float(origin[2] - dist)
        if int(geomid[0]) in wanted:
            return hit
        origin = np.array([origin[0], origin[1], hit - .001])
        if origin[2] < float(env.work['top_z']) - .05:
            return None
    return None


def geom_points(env, geom_id):
    """World points of a geom's own shape: a mesh's vertices, else the corners of its own box. The world AABB is
    no use for a finger (its box is dominated by the knuckle) or for a plate (RoboCasa plates carry a visual
    shell several millimetres proud of the hulls that actually touch anything)."""
    model, data = env.sim.model._model, env.sim.data._data
    rot = np.asarray(data.geom_xmat[geom_id], float).reshape(3, 3)
    pos = np.asarray(data.geom_xpos[geom_id], float)
    if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_MESH and int(model.geom_dataid[geom_id]) >= 0:
        mesh = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        verts = np.asarray(model.mesh_vert[start:start + count], float).reshape(-1, 3)
        return verts @ rot.T + pos
    centre, half = np.asarray(model.geom_aabb[geom_id, :3], float), np.asarray(model.geom_aabb[geom_id, 3:], float)
    line = np.linspace(-1., 1., BOX_SAMPLES)          # a lattice, not the eight corners: a long bar's corners
    grid = np.stack(np.meshgrid(line, line, line, indexing='ij'), -1).reshape(-1, 3)   # miss the narrow band a
    return (centre + grid * half) @ rot.T + pos       # profile looks at


def hull_profile(env, geom_ids, origin_xy, direction, *, band=.020, collision_only=True):
    """``(radius, z)`` of every collision-hull vertex of ``geom_ids`` within ``band`` of the line that leaves
    ``origin_xy`` along horizontal ``direction``; radius is measured along that line. Rays cannot do this: with
    ``geomgroup=None`` ``mj_ray`` returns the visual shell (group 1, contype 0) and never the hulls."""
    model = env.sim.model
    d = np.asarray(direction, float)[:2]
    d = d / np.linalg.norm(d)
    across = np.array([-d[1], d[0]])
    origin_xy = np.asarray(origin_xy, float)[:2]
    rows = []
    for g in geom_ids:
        if collision_only and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        points = geom_points(env, g)
        radius = (points[:, :2] - origin_xy) @ d
        lateral = np.abs((points[:, :2] - origin_xy) @ across)
        keep = lateral < band
        if keep.any():
            rows.append(np.column_stack([radius[keep], points[keep][:, 2]]))
    return np.vstack(rows) if rows else np.zeros((0, 2))


def touching_geom_ids(env, body, *, robot=False):
    """Ids of the geoms in contact with a body right now; the robot's own geoms are left out unless asked for."""
    model, data = env.sim.model, env.sim.data._data
    mine = set(body_geom_ids(env, body))
    out = set()
    for i in range(data.ncon):
        con = data.contact[i]
        pair = (int(con.geom1), int(con.geom2))
        for a, b in (pair, pair[::-1]):
            if a in mine and b not in mine:
                name = model.geom_id2name(b) or ''
                if robot or not name.startswith(ROBOT_PREFIXES):
                    out.add(b)
    return sorted(out)


def geom_distance(env, first, second, *, distmax=.10):
    """Smallest surface distance between two sets of geoms and the two witness points, from
    ``mj_geomDistance``: the real shapes, negative when they overlap."""
    model, data = env.sim.model._model, env.sim.data._data
    best, fromto = (float(distmax), -1, -1, None, None), np.zeros(6)
    for a in first:
        for b in second:
            d = float(mujoco.mj_geomDistance(model, data, int(a), int(b), float(distmax), fromto))
            if d < best[0]:
                best = (d, int(a), int(b), fromto[:3].copy(), fromto[3:].copy())
    names = env.sim.model.geom_id2name
    return dict(distance=best[0], first=None if best[1] < 0 else names(best[1]),
                second=None if best[2] < 0 else names(best[2]),
                on_first=None if best[3] is None else [float(v) for v in best[3]],
                on_second=None if best[4] is None else [float(v) for v in best[4]])


def finger_geom_ids(env, *, pads=True):
    """``{name: id}`` of the gripper's finger geoms (the two links and, with ``pads``, the two pads)."""
    model = env.sim.model
    prefix = env.robots[0].gripper['right'].naming_prefix
    names = [prefix + 'finger1_collision', prefix + 'finger2_collision']
    if pads:
        names += [prefix + 'finger1_pad_collision', prefix + 'finger2_pad_collision']
    out = {}
    for name in names:
        gid = mujoco.mj_name2id(env.sim.model._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid is not None and gid >= 0:
            out[name] = int(gid)
    return out


def finger_sides(env, finger_ids, origin_xy, direction, radius):
    """Split the finger geoms into the ones outboard of ``radius`` along ``direction`` and the ones inboard of
    it, by the radius of each geom's own centre."""
    data = env.sim.data
    d = np.asarray(direction, float)[:2]
    d = d / np.linalg.norm(d)
    origin_xy = np.asarray(origin_xy, float)[:2]
    radii = {name: float((np.asarray(data.geom_xpos[g])[:2] - origin_xy) @ d) for name, g in finger_ids.items()}
    outer = {n: finger_ids[n] for n, r in radii.items() if r > radius}
    inner = {n: finger_ids[n] for n, r in radii.items() if r <= radius}
    return inner, outer, radii


def lowest_gripper_z(env, geom_ids):
    """World z of the lowest point of the given gripper geoms' own shapes."""
    return min(float(geom_points(env, g)[:, 2].min()) for g in geom_ids)


def open_fingers(skills, target, rotation=None, *, settle=3, max_steps=240, phase='open'):
    """Bring the fingers ``target`` metres apart from fully open. robosuite integrates the gripper action
    (``current_action += speed * sign(action)`` per step), so a loop that watches the measured gap alone
    overshoots by however far the fingers lag the command; one closing tick with a settle after it does not."""
    for _ in range(int(max_steps)):
        if skills.finger_gap() <= target:
            break
        skills.hold(1, 1., rotation, phase=phase)
        skills.hold(int(settle), 0., rotation, phase=phase + '_settle')
    skills.hold(6, 0., rotation, phase=phase + '_hold')
    return float(skills.finger_gap())


def plate_rim_profile(env, cover_body, toward, *, bite=PLATE_RIM_BITE_M, band=.020):
    """What the near side of a plate resting on a pan is, measured from the collision hulls: ``edge_r`` how far
    its hull reaches toward the robot from its centre and ``edge_z`` how high that point is, ``under_z`` the
    lowest hull point within a centimetre of the edge, ``support_r``/``support_z`` the outermost point of
    whatever it is resting on (the pan's near wall) and ``land_z`` the top of the plate's own surface where the
    inner finger will come down, ``bite`` inside the edge. Heights are world z, radii are from the centre."""
    centre, _ = body_pose(env, cover_body)
    plate = hull_profile(env, body_geom_ids(env, cover_body), centre[:2], toward, band=band)
    if not len(plate):
        return None
    edge_r = float(plate[:, 0].max())
    lip = plate[plate[:, 0] > edge_r - .010]
    land = plate[np.abs(plate[:, 0] - (edge_r - bite)) < .006]
    support_ids = touching_geom_ids(env, cover_body)
    support = hull_profile(env, support_ids, centre[:2], toward, band=band) if support_ids else np.zeros((0, 2))
    near = support[support[:, 0] > float(support[:, 0].max()) - .010] if len(support) else support
    return dict(centre=[float(v) for v in centre], edge_r=edge_r, edge_z=float(lip[:, 1].max()),
                under_z=float(lip[:, 1].min()), land_r=float(edge_r - bite),
                land_z=float(land[:, 1].max()) if len(land) else None,
                support=[env.sim.model.geom_id2name(g) for g in support_ids],
                support_r=float(support[:, 0].max()) if len(support) else None,
                support_z=float(near[:, 1].max()) if len(near) else None)


def cover_record(env, cover_body):
    """The ``covers`` record of ``env.task_spec['observability']`` for a cover body, or an empty dict."""
    record = env.task_spec.get('observability') or {}
    for cover in record.get('covers', []):
        if cover.get('name') == cover_body and cover.get('built'):
            return cover
    return {}


def occluder_record(env, body):
    """The ``occluders`` record of ``env.task_spec['observability']`` for an occluder body (which names the
    object it hides and that object's build-time disc), or an empty dict."""
    record = (getattr(env, 'task_spec', None) or {}).get('observability') or {}
    for occluder in record.get('occluders', []):
        if occluder.get('name') == body:
            return occluder
    return {}


def _events_since(env, count, kind=LEFT_SURFACE):
    return [e for e in list(getattr(env, 'events', []))[count:] if e.get('kind') == kind]


def _no_collateral(env, before, skill):
    left = _events_since(env, before)
    if left:
        raise SkillFailure(skill, 'a prop left the surface: ' + ', '.join(str(e['object']) for e in left), events=left)


def _pitched_side(approach, fingers_up=True, pitch_deg=45.):
    """:func:`_side_rotation` with the tool z pitched ``pitch_deg`` below the horizontal ``approach``, the finger
    axis turning with it: at 0 the pads sit above and below a rim and the hand comes in level, at 90 they sit
    either side of it and the hand comes straight down. In between one pad rides the plate and the other slides
    under its overhang, and the close pulls the rim up as well as together."""
    a = _unit([approach[0], approach[1], 0.])
    th, up = math.radians(float(pitch_deg)), np.array([0., 0., 1.])
    z = a * math.cos(th) - up * math.sin(th)
    x = (a * math.sin(th) + up * math.cos(th)) * (1. if fingers_up else -1.)
    return np.column_stack([x, np.cross(z, x), z])


def _side_rotation(approach, fingers_up=True):
    """Tool z along the horizontal ``approach``, finger axis vertical (pads above and below a rim)."""
    z = _unit([approach[0], approach[1], 0.])
    x = np.array([0., 0., 1. if fingers_up else -1.])
    return np.column_stack([x, np.cross(z, x), z])


def _wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def _tuck(skills, gripper=None, *, height=None, ahead=CARRY_AHEAD_M, phase='tuck', boxes=None):
    """Raise the hand above everything on the surface and bring it back over the base before a drive, keeping
    the hand's current orientation: every waypoint is tolerant, because the arm's home pose does not always
    admit ``Skills.carry``'s tool-down orientation at the tuck point within the waypoint tolerance (seen on the
    stamps kitchens), and a tuck exists only to clear the props.

    Overhead fixtures (:func:`headroom.overhead_boxes`) bound the tuck: the hand never rises into one over
    where it is (the forearm stands :data:`CARRY_ALLOWANCE_M` above a top-down hand), and the retract point
    ``ahead`` of the base moves closer to the base, 4 cm at a time down to 0.22 m, until its column and the
    straight path to it clear the fixtures at the carry height; when nothing clears, the hand stays raised
    where it is. Returns ``dict(z, ahead, limited)``."""
    gripper = skills.gripper if gripper is None else float(gripper)
    top = skills.top_z
    z = top + CARRY_ABOVE_TOP_M if height is None else float(height)
    boxes = H.overhead_boxes(skills.env, top) if boxes is None else boxes
    state = skills.state()
    limited = False
    if boxes:
        ceiling = carry_limit(boxes, state['eef'][:2], top)
        if ceiling < z:
            z, limited = max(ceiling, float(state['eef'][2])), True
    skills.move([state['eef'][0], state['eef'][1], z], state['eef_rotation'], gripper, phase=phase + '_raise', ticks=120,
                tolerance=.01, required=False)
    base_xy, _ = skills.base_pose()
    facing = skills.facing()
    here = skills.state()['eef'][:2]
    chosen = float(ahead)
    if boxes:
        chosen = None
        need = z - top + CARRY_ALLOWANCE_M
        for candidate in np.arange(float(ahead), .22 - 1e-9, -.04):
            point = base_xy + facing * float(candidate)
            if carry_limit(boxes, point, top) < z:
                continue
            if H.path_headroom_gate(boxes, H.path_points(here, point), CARRY_COLUMN_M, need, top)['passed']:
                chosen = float(candidate)
                break
        limited = limited or chosen != float(ahead)
    if chosen is not None:
        target = np.r_[base_xy + facing * chosen, z]
        skills.move(target, skills.state()['eef_rotation'], gripper, phase=phase + '_retract', ticks=160, tolerance=.01,
                    required=False)
    return dict(z=float(z), ahead=chosen, limited=bool(limited))


def _stand_within_reach(skills, target, ahead=GRASP_AHEAD_M, phase='stand', gripper=None, max_short=(.14, .20)):
    """``Skills.stand`` that never asks the base to come closer to the surface than it is: a target that is
    already farther ahead than ``ahead`` is centred laterally only (annex stances have no frame limit)."""
    base_xy, yaw = skills.base_pose()
    f = np.array([math.cos(yaw), math.sin(yaw)])
    now = float((np.asarray(target, float)[:2] - base_xy) @ f)
    return skills.stand(target, ahead=max(ahead, now) if now > ahead else ahead, forward_limit=False, gripper=gripper,
                        max_short=max_short, phase=phase)


# ---- uncover skills ---------------------------------------------------------------------------------
def lift_cloche(skills, cover_body, *, spot=None, alternative=0, above=.12, lift=None, phase='cloche'):
    """Lift a cloche by its knob and set it down on a free spot of the same surface. Returns a record."""
    env = skills.env
    info = cover_record(env, cover_body)
    covered = info.get('object') or (env.task_spec['objects'].get(cover_body) or {}).get('covers')
    radius = float(info.get('radius') or .11)
    object_height = float(info.get('object_height') or .10)
    top = skills.top_z
    site = f'{cover_body}_knob'
    events_before = len(env.events)
    start = skills.tick
    covered_before = body_pose(env, covered)[0] if covered else None
    cover_before = body_pose(env, cover_body)[0]
    lift = float(lift if lift is not None else max(object_height + .05, .12))
    with skills.segment(f'lift the cloche {cover_body} by its knob', kind='uncover', skill='lift_cloche'):
        _tuck(skills, -1., phase=phase + '_clear')
        knob = skills.site_position(site)
        _stand_within_reach(skills, knob, phase=phase + '_stand')
        knob = skills.site_position(site)
        grasp_dz = float(knob[2] - top)
        geoms = [f'{cover_body}_knob_shaft', f'{cover_body}_knob_cap']
        skills.grasp(knob, 'top', alternative=alternative, above=above, geoms=geoms, lower_ticks=160, phase=phase + '_grasp')
        rotation = skills.state()['eef_rotation'].copy()
        skills.move(knob + [0., 0., lift], rotation, 1., phase=phase + '_lift', ticks=200)
        skills.hold(10, 1., rotation, phase=phase + '_lift_hold')
        rise = float(body_pose(env, cover_body)[0][2] - cover_before[2])
        if rise < .6 * lift:
            raise SkillFailure(phase, f'cloche rose only {rise * 1000:.0f} mm of {lift * 1000:.0f}', rise=rise)
        if not skills.grasped(geoms):
            raise SkillFailure(phase, 'the knob slipped out during the lift')
        carry_z = top + grasp_dz + max(lift, .12)
        _tuck(skills, 1., height=carry_z, phase=phase + '_carry')
        if spot is None:
            need = H.headroom_needed(info.get('kind') if info.get('kind') in ('cloche_small', 'cloche_large') else 'cloche_large',
                                     cover_height=float(info.get('height') or grasp_dz + .012), object_height=object_height)
            spot = set_down_spot(skills, radius + .01, exclude=[cover_body], prefer_world=knob[:2], headroom=need,
                                 carry_from=skills.state()['eef'][:2], carry_z=carry_z)
        spot = np.asarray(spot, float)[:2]
        _stand_within_reach(skills, [spot[0], spot[1], top], phase=phase + '_place_stand', gripper=1.)
        skills.place([spot[0], spot[1], top + grasp_dz + .004], skills.state()['eef_rotation'], above=.10,
                     lower_ticks=200, phase=phase + '_place')
        _tuck(skills, -1., phase=phase + '_done')
    after, rot = body_pose(env, cover_body)
    record = dict(cover=cover_body, object=covered, spot_world=spot.tolist(), rise_m=rise,
                  cover_z_error_m=float(after[2] - cover_before[2]), tilt_deg=tilt_deg(rot),
                  ticks=skills.tick - start)
    if abs(after[2] - cover_before[2]) > .03 or tilt_deg(rot) > 10.:
        raise SkillFailure(phase, f'cloche not standing on the surface after the set-down (dz {after[2] - cover_before[2]:+.3f} m, '
                           f'tilt {tilt_deg(rot):.0f} deg)', **record)
    if covered:
        now = body_pose(env, covered)[0]
        record['object_moved_m'] = float(np.linalg.norm(now - covered_before))
        record['object_clear_m'] = float(np.linalg.norm(now[:2] - after[:2]) - radius)
        if record['object_moved_m'] > .03:
            raise SkillFailure(phase, f"the covered object moved {record['object_moved_m'] * 1000:.0f} mm", **record)
        if record['object_clear_m'] < 0.:
            raise SkillFailure(phase, 'the cloche was set down over the object again', **record)
    _no_collateral(env, events_before, phase)
    return record


def slide_lift_bowl(skills, cover_body, edge=None, *, spot=None, target_overhang=BOWL_TARGET_OVERHANG,
                    fingers_up=True, push_height=.025, record=None, phase='bowl'):
    """Slide an inverted bowl to the counter edge until ``target_overhang`` of it overhangs, pinch the
    overhanging rim, lift, carry to a free spot and set it down. ``edge`` picks one of the gate's options
    (default: the gate's choice).

    ``record`` replaces the instance's ``covers`` entry with one the caller gated itself, for a bowl the
    generator never drew as a cover: the painted-cubes face occluders stand over a cube wherever the
    scatter put them, so their slide is gated at oracle time over the live scene
    (:func:`oracles.painted_cubes.bowl_slide_record`) instead of at mint time.
    """
    env = skills.env
    info = dict(record) if record else cover_record(env, cover_body)
    covered = info.get('object') or (env.task_spec['objects'].get(cover_body) or {}).get('covers')
    gate = info.get('reach') or {}
    options = {o['edge']: o for o in gate.get('options', [])}
    if gate.get('chosen') and edge is None:
        edge = gate['chosen']['edge']
    if edge not in options and edge is not None:
        option = None
    else:
        option = options.get(edge)
    if option is None:
        raise SkillFailure(phase, f'no gated slide path for edge {edge!r} (options {sorted(options)})')
    R = float(info.get('radius') or .10)
    top = skills.top_z
    rect = O.work_surface_rect(env)
    d_frame = np.asarray(option['direction_frame'], float)
    d_world = skills.direction_to_world([d_frame[0], d_frame[1], 0.])[:2]
    edge_along = {'front': -rect.y0, 'back': rect.y1, 'left': -rect.x0, 'right': rect.x1}[edge]
    events_before = len(env.events)
    start = skills.tick
    geoms = collision_geom_names(env, cover_body)

    def along(world_xy):
        c = O.world_to_frame_xy(env, world_xy)
        return float(np.dot(c, d_frame))

    def overhang():
        return overhang_fraction(along(body_pose(env, cover_body)[0][:2]), edge_along, R)

    obj_before = body_pose(env, covered)[0] if covered else None
    obj_radius = float(info.get('object_radius') or .04)
    with skills.segment(f'slide the bowl {cover_body} to the {edge} edge', kind='uncover', skill='slide_lift_bowl'):
        _tuck(skills, -1., phase=phase + '_clear')
        if edge != 'front':
            stance = (option.get('reach') or {}).get('stance')
            if stance:
                drive_to_stance(skills, dict(stance, edge=edge, standoff=O.STANDOFF_M), phase=phase + '_drive')
        bowl_xy = body_pose(env, cover_body)[0][:2]
        slide = float(option['slide_m'])
        contact0 = bowl_xy - d_world * (R + .02)
        path_mid = contact0 + d_world * slide / 2
        _stand_within_reach(skills, [path_mid[0], path_mid[1], top], ahead=GRASP_AHEAD_M - .02, phase=phase + '_stand')
        down = skills.hand_down()
        skills.move([contact0[0], contact0[1], top + .15], down, 1., phase=phase + '_above', ticks=200)
        z = top + push_height
        skills.move([contact0[0], contact0[1], z], down, 1., phase=phase + '_lower', ticks=140, tolerance=.006)
        pushed, stalls, steps = 0., 0, 0
        last = along(body_pose(env, cover_body)[0][:2])
        while overhang() < target_overhang - .02 and steps < 40:
            hand = skills.state()['eef']
            step = min(SLIDE_STEP_M, max(.01, slide - pushed + .03))
            skills.move(hand + np.r_[d_world * step, 0.], down, 1., phase=f'{phase}_push_{steps}', ticks=60,
                        tolerance=.004, required=False)
            now = along(body_pose(env, cover_body)[0][:2])
            moved = now - last
            pushed += max(moved, 0.)
            stalls = stalls + 1 if moved < .004 else 0
            last = now
            steps += 1
            if covered:
                obj_inside = edge_along - along(body_pose(env, covered)[0][:2])
                if obj_inside < obj_radius + .005:
                    raise SkillFailure(phase, f'the covered object reached the edge ({obj_inside:.3f} m inside)')
            if stalls >= PUSH_STALL_STEPS:
                raise SkillFailure(phase, f'bowl stopped moving after {pushed * 1000:.0f} mm (overhang {overhang():.2f})')
        final_overhang = overhang()
        if final_overhang < target_overhang - .05:
            raise SkillFailure(phase, f'overhang {final_overhang:.2f} after {steps} pushes')
        skills.move(skills.state()['eef'] + [0., 0., .15], down, -1., phase=phase + '_retreat', ticks=120, required=False)
    with skills.segment('pinch the overhanging rim and lift the bowl', kind='uncover', skill='slide_lift_bowl'):
        # Side pinch with vertical fingers from outside the counter, the way the docstring always described it: the
        # tool points horizontally at the bowl's centre, the lower finger passes under the overhanging rim (free air
        # below it past the edge) and the upper finger comes down on the dome, so the shell's edge is held between
        # them. A top-down pinch across the rim cannot work: the inner fingertip lands on the dome and holds the pads
        # above the rim (measured: the hand stopped 7.4 cm above its target and the pads closed on air). The edge
        # rests on the lower finger's mesh rather than its pad, so the grasp check counts the finger meshes
        # (bowl_probe 2026-09-21: gap 45 mm, three finger contacts at ahead .58, hand .02 above the top).
        bowl_xy = body_pose(env, cover_body)[0][:2]
        a = np.array([-d_world[0], -d_world[1], 0.])            # the approach: from beyond the edge toward the bowl
        rotation = _side_rotation(a[:2], fingers_up=fingers_up)
        rim = np.array([*(bowl_xy + d_world * (R - BOWL_PINCH_INSET_M)), top + BOWL_PINCH_ABOVE_TOP_M])
        _stand_within_reach(skills, rim, ahead=SIDE_PINCH_AHEAD_M, phase=phase + '_pinch_stand')
        bowl_xy = body_pose(env, cover_body)[0][:2]
        rim = np.array([*(bowl_xy + d_world * (R - BOWL_PINCH_INSET_M)), top + BOWL_PINCH_ABOVE_TOP_M])
        skills.move(rim - a * .14 + [0., 0., .05], rotation, -1., phase=phase + '_pre', ticks=220, tolerance=.008, required=False)
        skills.move(rim, rotation, -1., phase=phase + '_rim', ticks=200, tolerance=.004, required=False)
        short = float(np.linalg.norm(skills.state()['eef'] - rim))
        skills.hold(30, 1., rotation, phase=phase + '_close')
        gap = skills.finger_gap()
        if gap < .003:
            raise SkillFailure(phase, f'fingers closed beside the rim (gap {gap * 1000:.1f} mm, approach short by {short:.3f} m)',
                               gap=gap, approach_short_m=short)
        if not skills.grasped(geoms, fingers=True):
            raise SkillFailure(phase, f'the fingers are not both on the bowl (gap {gap * 1000:.0f} mm, approach short by {short:.3f} m)',
                               gap=gap, approach_short_m=short)
        z0 = float(body_pose(env, cover_body)[0][2])
        skills.move(skills.state()['eef'] + [0., 0., .16], rotation, 1., phase=phase + '_lift', ticks=220, tolerance=.01,
                    required=False)
        skills.hold(15, 1., rotation, phase=phase + '_lift_hold')
        rise = float(body_pose(env, cover_body)[0][2] - z0)
        if rise < .06:
            raise SkillFailure(phase, f'bowl rose only {rise * 1000:.0f} mm', rise=rise)
        if not skills.grasped(geoms, fingers=True):
            raise SkillFailure(phase, 'the rim slipped out during the lift')
    with skills.segment('carry the bowl to a free spot and set it down', kind='uncover', skill='slide_lift_bowl'):
        hand = skills.state()['eef']
        skills.move([hand[0], hand[1], top + 2 * R + .12], rotation, 1., phase=phase + '_raise', ticks=160, tolerance=.01,
                    required=False)
        if spot is None:
            spot = set_down_spot(skills, R + .03, exclude=[cover_body], prefer_world=bowl_xy,
                                 headroom=H.headroom_needed('bowl', cover_height=float(info.get('height') or 2 * R), radius=R),
                                 carry_from=skills.state()['eef'][:2], carry_z=top + 2 * R + .12)
        spot = np.asarray(spot, float)[:2]
        _stand_within_reach(skills, [spot[0], spot[1], top], ahead=.44, phase=phase + '_place_stand', gripper=1.)
        hand = skills.state()['eef']
        lo, hi = body_aabb(env, cover_body)
        hang = float(hand[2] - lo[2])                       # how far the bowl hangs below the hand
        skills.move([spot[0], spot[1], hand[2]], rotation, 1., phase=phase + '_over', ticks=200, tolerance=.01)
        skills.move([spot[0], spot[1], top + hang + .004], rotation, 1., phase=phase + '_lower', ticks=200, tolerance=.006,
                    required=False)
        skills.hold(25, -1., rotation, phase=phase + '_release')
        skills.move(skills.state()['eef'] - a * .06 + [0., 0., .12], rotation, -1., phase=phase + '_retreat', ticks=140,
                    required=False)
        skills.hold(30, -1., phase=phase + '_settle')
        _tuck(skills, -1., phase=phase + '_done')
    bowl_after, bowl_rot = body_pose(env, cover_body)
    record = dict(cover=cover_body, object=covered, edge=edge, slide_m=pushed, pushes=steps, overhang=final_overhang,
                  rise_m=rise, spot_world=spot.tolist(), bowl_z_m=float(bowl_after[2] - top), tilt_deg=tilt_deg(bowl_rot),
                  ticks=skills.tick - start)
    c = O.world_to_frame_xy(env, bowl_after[:2])
    if not (rect.x0 - .02 <= c[0] <= rect.x1 + .02 and rect.y0 - .02 <= c[1] <= rect.y1 + .02) or bowl_after[2] < top - .03:
        raise SkillFailure(phase, 'the bowl is not on the work top after the set-down', **record)
    if covered:
        obj_after = body_pose(env, covered)[0]
        record['object_moved_m'] = float(np.linalg.norm(obj_after[:2] - obj_before[:2]))
        record['object_clear_m'] = float(np.linalg.norm(obj_after[:2] - bowl_after[:2]) - R)
        oc = O.world_to_frame_xy(env, obj_after[:2])
        record['object_on_top'] = bool(rect.x0 <= oc[0] <= rect.x1 and rect.y0 <= oc[1] <= rect.y1 and obj_after[2] > top - .02)
        if not record['object_on_top']:
            raise SkillFailure(phase, 'the covered object is no longer on the work top', **record)
        if record['object_clear_m'] < -R * .5:
            raise SkillFailure(phase, 'the bowl still covers the object', **record)
    _no_collateral(env, events_before, phase)
    return record


def push_cover(skills, cover_body, direction=None, distance=None, *, record=None, phase='push'):
    """Push a flat cover (plate, board, tray) off the place it hides along a task-frame ``direction`` for
    ``distance`` metres (defaults: the gate's choice), keeping it on the surface.

    ``record`` stands in for the instance's ``covers`` entry (``half``, ``yaw_frame``, optionally the place)
    for a body the generator never drew as a cover: the painted-cubes face occluders, which the cubes
    oracle's last resort pushes by their own footprint (:func:`oracles.painted_cubes.nudge_cover_off`)."""
    env = skills.env
    info = dict(record) if record else cover_record(env, cover_body)
    gate = info.get('reach') or {}
    chosen = gate.get('chosen') or {}
    if direction is None:
        direction = chosen.get('direction_frame')
    if direction is None:
        raise SkillFailure(phase, 'no gated push direction')
    d_frame = _unit([float(direction[0]), float(direction[1])])
    if distance is None:
        distance = chosen.get('length_m')
    if distance is None:
        half = info.get('half') or (.09, .09)
        place_half = info.get('place_half') or (.08, .06)
        distance = float(max(half)) + float(max(place_half)) + .02
    distance = float(distance)
    d_world = skills.direction_to_world([d_frame[0], d_frame[1], 0.])[:2]
    top = skills.top_z
    rect = O.work_surface_rect(env)
    half = [float(v) for v in (info.get('half') or (.09, .09))]
    yaw_frame = float(info.get('yaw_frame') or 0.)
    c_along = abs(math.cos(yaw_frame) * d_frame[0] + math.sin(yaw_frame) * d_frame[1]) * half[0] \
        + abs(-math.sin(yaw_frame) * d_frame[0] + math.cos(yaw_frame) * d_frame[1]) * half[1]
    events_before = len(env.events)
    start = skills.tick

    def along(world_xy):
        return float(np.dot(O.world_to_frame_xy(env, world_xy), d_frame))

    cover_xy0, _ = body_pose(env, cover_body)
    lo0, hi0 = body_aabb(env, cover_body)
    with skills.segment(f'push the cover {cover_body} aside', kind='uncover', skill='push_cover'):
        _tuck(skills, -1., phase=phase + '_clear')
        contact0 = cover_xy0[:2] - d_world * (c_along + .02)
        path_mid = contact0 + d_world * distance / 2
        _stand_within_reach(skills, [path_mid[0], path_mid[1], top], phase=phase + '_stand')
        down = skills.hand_down()
        skills.move([contact0[0], contact0[1], top + .15], down, 1., phase=phase + '_above', ticks=200)
        drop = fingertip_drop(skills)
        z = float(max(top + drop + .004, lo0[2] + drop * .5))
        z = min(z, float(hi0[2]) + drop - .004)             # the tips must catch the cover's edge
        skills.move([contact0[0], contact0[1], z], down, 1., phase=phase + '_lower', ticks=140, tolerance=.006)
        pushed, stalls, steps = 0., 0, 0
        last = along(cover_xy0[:2])
        while pushed < distance and steps < 40:
            hand = skills.state()['eef']
            step = min(SLIDE_STEP_M, distance - pushed + .02)
            skills.move(hand + np.r_[d_world * step, 0.], down, 1., phase=f'{phase}_push_{steps}', ticks=60,
                        tolerance=.004, required=False)
            now = along(body_pose(env, cover_body)[0][:2])
            moved = now - last
            pushed += max(moved, 0.)
            stalls = stalls + 1 if moved < .004 else 0
            last = now
            steps += 1
            if stalls >= PUSH_STALL_STEPS:
                raise SkillFailure(phase, f'cover stopped moving after {pushed * 1000:.0f} mm of {distance * 1000:.0f}')
        skills.move(skills.state()['eef'] + [0., 0., .15], down, -1., phase=phase + '_retreat', ticks=120, required=False)
        _tuck(skills, -1., phase=phase + '_done')
    after, rot = body_pose(env, cover_body)
    c = O.world_to_frame_xy(env, after[:2])
    record = dict(cover=cover_body, direction_frame=d_frame.tolist(), distance_m=distance, pushed_m=pushed, pushes=steps,
                  cover_z_error_m=float(after[2] - cover_xy0[2]), tilt_deg=tilt_deg(rot), ticks=skills.tick - start)
    if pushed < .9 * distance:
        raise SkillFailure(phase, f'cover moved {pushed * 1000:.0f} mm of {distance * 1000:.0f}', **record)
    if abs(after[2] - cover_xy0[2]) > .03 or tilt_deg(rot) > 15. or not (
            rect.x0 <= c[0] <= rect.x1 and rect.y0 <= c[1] <= rect.y1):
        raise SkillFailure(phase, 'the cover is not lying flat on the work top after the push', **record)
    place_xy = info.get('place_xy')
    if place_xy is not None:
        place = S.Box(tuple(float(v) for v in place_xy), tuple(float(v) for v in (info.get('place_half') or (.08, .06))),
                      float(info.get('place_yaw') or 0.))
        cover_now = S.Box((float(c[0]), float(c[1])), (half[0], half[1]), float(yaw_frame))
        record['place_clearance_m'] = float(S.clearance(cover_now, place))
        if record['place_clearance_m'] < -.01:
            raise SkillFailure(phase, f"the cover still overlaps the place by {-record['place_clearance_m'] * 1000:.0f} mm", **record)
    _no_collateral(env, events_before, phase)
    return record


def lift_plate_from_pan(skills, cover_body, *, spot=None, fingers_up=True, pitch_deg=PLATE_PITCH_DEG,
                        phase='pan_plate'):
    """Take the near rim of a plate lying on a balance pan between the fingertips, peel it off the pan's rims
    and lift it clear, then carry it to a free spot on the counter and set it down.

    Everything the pinch needs is measured on the instance from the collision hulls
    (:func:`plate_rim_profile`): how far the plate's hull reaches toward the robot, how far that is past the
    near wall of whatever holds it up, how deep its rim stands over its own surface a finger's width inside the
    edge. The hand comes down vertically with the fingers straddling the rim - the outer one through the free
    air beyond the overhang, the inner one onto the plate - guided by ``mj_geomDistance`` rather than by any
    modelled fingertip, and closes: the fingertips, which hang ``FINGERTIP_BELOW_PAD_M`` below the pads and
    several millimetres inward of them, take the rim between them. The lift then goes up in small steps that
    also pull toward the robot, peeling the plate off the pan's rims; the pan, unloaded, swings up under it, so
    the postcondition is the plate standing clear of the pan's rim, not the height it rose.

    A horizontal tool at rim height would fit under the overhang but cannot be held there: 0.55 m ahead of the
    base at 8 cm over the counter the wrist pins joint 6 at its limit and the hand settles 36 mm high, driving
    the palm into the plate (side-pinch probes, 2026-09-21).

    Refusals carry their numbers: too little overhang, too flat a rim, a rim wider than the fingers open, no
    stance that reaches it, a descent that cannot straddle the rim, a close on air, a plate that slips.
    """
    env = skills.env
    info = cover_record(env, cover_body)
    half = [float(v) for v in (info.get('half') or (.09, .09))]
    r = min(half)
    top = skills.top_z
    events_before = len(env.events)
    start = skills.tick
    geoms = collision_geom_names(env, cover_body)
    model = env.sim.model
    plate_ids = [g for g in body_geom_ids(env, cover_body)
                 if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0]
    toward_robot = skills.direction_to_world([0., -1., 0.])[:2]
    beam0 = env.beam_angle_deg() if hasattr(env, 'beam_angle_deg') else None
    parcels0 = ({name: body_pose(env, name)[0].copy() for name in env.parcel_ids}
                if hasattr(env, 'parcel_ids') else {})
    with skills.segment(f'pinch the rim of the plate {cover_body} between the fingertips and lift it off the pan',
                        kind='uncover', skill='lift_plate_from_pan'):
        _tuck(skills, -1., phase=phase + '_clear')
        support_ids = touching_geom_ids(env, cover_body)
        rim = plate_rim_profile(env, cover_body, toward_robot)
        if rim is None:
            raise SkillFailure(phase, f'the plate {cover_body} has no collision hull to pinch')
        overhang = None if rim['support_r'] is None else float(rim['edge_r'] - rim['support_r'])
        drop = None if rim['land_z'] is None else float(rim['edge_z'] - rim['land_z'])
        measured = dict(cover=cover_body, edge_r_m=round(rim['edge_r'], 4),
                        edge_above_top_m=round(rim['edge_z'] - top, 4),
                        underside_above_top_m=round(rim['under_z'] - top, 4),
                        lip_m=round(rim['edge_z'] - rim['under_z'], 4),
                        support=[model.geom_id2name(g) for g in support_ids],
                        support_r_m=None if rim['support_r'] is None else round(rim['support_r'], 4),
                        support_above_top_m=None if rim['support_z'] is None else round(rim['support_z'] - top, 4),
                        overhang_m=None if overhang is None else round(overhang, 4),
                        rim_drop_m=None if drop is None else round(drop, 4))
        if overhang is None:
            raise SkillFailure(phase, 'nothing is holding the plate up: it touches no other body, so the pinch has '
                               'no wall to clear', **measured)
        if overhang < PLATE_MIN_OVERHANG_M:
            raise SkillFailure(phase, f'the plate stands only {overhang * 1000:.1f} mm out over the pan wall toward the '
                               f'robot, {PLATE_MIN_OVERHANG_M * 1000:.0f} mm needed for the outer finger to come down '
                               f'past it', **measured)
        pitch = float(pitch_deg)
        measured['pitch_deg'] = round(pitch, 1)
        if pitch >= 89. and (drop is None or drop < PLATE_MIN_RIM_DROP_M):
            raise SkillFailure(phase, f"the plate's rim stands {0. if drop is None else drop * 1000:.1f} mm over its own "
                               f'surface {PLATE_RIM_BITE_M * 1000:.0f} mm inside the edge, {PLATE_MIN_RIM_DROP_M * 1000:.0f} '
                               f'mm needed: the inner fingertip would land level with the rim and the close would slide '
                               f'the plate instead of taking it', **measured)   # a pitched pinch does not care: it
        if pitch >= 89.:                                                        # takes the lip from outside
            gap = PLATE_RIM_CLEAR_M + PLATE_RIM_BITE_M
            axis_r, pinch_z = float(rim['edge_r'] + PLATE_RIM_CLEAR_M - gap / 2.), float(rim['edge_z'])
            finger_yaw = math.atan2(toward_robot[1], toward_robot[0])
            # alternatives 0 and 1 keep the finger axis across the rim (180-degree flips); see axis_keeping_order
            rotation = skills.grasp_rotation('top', alternative=0 if fingers_up else 1, finger_yaw=finger_yaw)
        else:                                       # pads above and below the rim, the hand diving in at an angle
            gap = PLATE_PITCH_GAP_M
            axis_r = float(rim['edge_r'] - PLATE_PITCH_INSET_M)
            pinch_z = float((rim['edge_z'] + rim['under_z']) / 2.)
            rotation = _pitched_side(-toward_robot, fingers_up=fingers_up, pitch_deg=pitch)
        if gap > GRIPPER_OPENING_M - .006:
            raise SkillFailure(phase, f'the rim needs the fingers {gap * 1000:.0f} mm apart, more than the '
                               f'{GRIPPER_OPENING_M * 1000:.0f} mm they open', **measured)
        approach = -np.asarray(rotation, float)[:, 2]        # back along the tool z: straight up when top-down
        centre, _ = body_pose(env, cover_body)
        short, shorts = 1., []
        for ahead in PLATE_RIM_STANCES:
            centre, _ = body_pose(env, cover_body)
            above = np.array([*(centre[:2] + toward_robot * axis_r), pinch_z]) + approach * .07
            if ahead is None:
                _stand_within_reach(skills, above, phase=phase + '_stand')
            else:
                _stand_within_reach(skills, above, ahead=float(ahead), phase=phase + f'_stand{int(ahead * 100)}')
            centre, _ = body_pose(env, cover_body)
            above = np.array([*(centre[:2] + toward_robot * axis_r), pinch_z]) + approach * .07
            skills.move(above, rotation, -1., phase=phase + '_above', ticks=240, tolerance=.006, required=False)
            short = float(np.linalg.norm(skills.state()['eef'] - above))
            shorts.append(round(short, 4))
            if short < .015:
                break
        measured['stand_short_m'] = shorts
        if short >= .015:
            raise SkillFailure(phase, f"the plate's near rim is out of reach: the hand stops {short * 1000:.0f} mm short "
                               f'of it from every stance tried ({", ".join(f"{v * 1000:.0f}" for v in shorts)} mm)',
                               **measured)
        measured['gap_open_mm'] = round(1000 * open_fingers(skills, gap, rotation, phase=phase + '_open'), 2)
        finger_ids = finger_geom_ids(env)
        hand_r = float((skills.state()['eef'][:2] - centre[:2]) @ toward_robot)   # the hand's own radius: the
        inner, outer, radii = finger_sides(env, finger_ids, centre[:2], toward_robot, hand_r)  # pitched hand
        measured['hand_r_m'] = round(hand_r, 4)                                   # stages outboard of the pinch
        measured['finger_r_m'] = {k.split('right_')[-1]: round(v, 4) for k, v in radii.items()}
        if not inner or not outer:
            raise SkillFailure(phase, 'the fingers did not open across the rim: they all came down on the same side of '
                               'it', **measured)
        descent = []
        for k in range(40):
            near_in = geom_distance(env, list(inner.values()), plate_ids)
            near_pan = (geom_distance(env, list(finger_ids.values()), support_ids) if support_ids
                        else dict(distance=.1))
            descent.append(dict(step=k, inner_mm=round(1000 * near_in['distance'], 2),
                                pan_mm=round(1000 * near_pan['distance'], 2),
                                eef_above_top_m=round(float(skills.state()['eef'][2]) - top, 4),
                                tip_above_top_m=round(lowest_gripper_z(env, finger_ids.values()) - top, 4)))
            if near_in['distance'] < PLATE_RIM_LAND_M or near_pan['distance'] < .002:
                break
            step = float(np.clip(near_in['distance'] - PLATE_RIM_LAND_M, .0015, .010))
            skills.move(skills.state()['eef'] - approach * step, rotation, 0., phase=f'{phase}_down{k}',
                        ticks=100, tolerance=.0015, required=False)
        measured['descent'] = descent[-3:]
        tip_z = lowest_gripper_z(env, finger_ids.values())
        rim = plate_rim_profile(env, cover_body, toward_robot) or rim
        measured['tip_below_rim_m'] = round(float(rim['edge_z'] - tip_z), 4)
        if tip_z > rim['edge_z'] - .001:
            raise SkillFailure(phase, f'the descent stopped with the fingertips {(tip_z - rim["edge_z"]) * 1000:.1f} mm '
                               f'over the rim, so nothing can come under it (the inner finger reached '
                               f'{descent[-1]["inner_mm"]:.1f} mm of the plate, the pan {descent[-1]["pan_mm"]:.1f} mm)',
                               **measured)
        for _ in range(40):
            skills.hold(2, 1., rotation, phase=phase + '_close')
            if skills.grasped(geoms, fingers=True):
                break
        skills.hold(8, 0., rotation, phase=phase + '_squeeze')   # hold the fingers where they are: going on
                                                                 # closing extrudes the wedge out of the tips
        measured['gap_closed_mm'] = round(1000 * skills.finger_gap(), 2)
        measured['grasped_pads'] = bool(skills.grasped(geoms))
        measured['grasped_fingers'] = bool(skills.grasped(geoms, fingers=True))
        if skills.finger_gap() < .0012:
            raise SkillFailure(phase, f'the fingers closed on air beside the rim (gap '
                               f'{measured["gap_closed_mm"]:.1f} mm)', **measured)
        if not measured['grasped_fingers']:
            raise SkillFailure(phase, f'the fingertips are not both on the plate after the close (gap '
                               f'{measured["gap_closed_mm"]:.1f} mm)', **measured)
        centre, _ = body_pose(env, cover_body)
        z0, centre0 = float(centre[2]), centre[:2].copy()
        lift = []

        def raise_hand(tag, k, pull, rise_step):
            """One step of the peel or of the lift, with what it did to the plate."""
            goal = skills.state()['eef'] + np.array([*(toward_robot * pull), rise_step])
            skills.move(goal, rotation, 0., phase=f'{phase}_{tag}{k}', ticks=90, tolerance=.004, required=False)
            skills.hold(6, 0., rotation, phase=f'{phase}_{tag}{k}_hold')
            pos, rot = body_pose(env, cover_body)
            lo, hi = body_aabb(env, cover_body)
            row = dict(step=f'{tag}{k}', rise_mm=round(1000 * float(pos[2] - z0), 1), tilt_deg=round(tilt_deg(rot), 1),
                       pulled_mm=round(1000 * float((pos[:2] - centre0) @ toward_robot), 1),
                       gap_mm=round(1000 * skills.finger_gap(), 2),
                       held=bool(skills.grasped(geoms, fingers=True)),
                       off_pan_mm=round(1000 * geom_distance(env, plate_ids, support_ids)['distance'], 1)
                       if support_ids else None,
                       clear_mm=round(1000 * float(lo[2] - max(geom_points(env, g)[:, 2].max() for g in support_ids)), 1)
                       if support_ids else None)
            if beam0 is not None:
                row['beam_deg'] = round(float(env.beam_angle_deg()), 2)
            lift.append(row)
            return row
        # the peel first: a plate of radius R on two rims 2 * rim apart clears the far rim after R - rim of travel
        # toward the robot, which is the overhang the profile measured (11-14 mm on these pans). Watching the
        # contact instead is not enough - the near edge leaves the near rim after a step or two while the far edge
        # is still inside the pan, and the lift then tips the plate against the far rim (certify 01:08, five of
        # eight instances rose 34-36 mm and stayed 22-25 mm below the rim top).
        need = float(overhang) + .003
        measured['peel_need_mm'] = round(1000 * need, 1)
        row = dict(held=True, off_pan_mm=0., pulled_mm=0.)
        for k in range(PLATE_PEEL_STEPS):
            row = raise_hand('peel', k, PLATE_PEEL_PULL_M, PLATE_PEEL_RISE_M)
            if not row['held'] or row['pulled_mm'] >= 1000 * need:
                break
        measured['peel_pulled_mm'] = row['pulled_mm']
        if row['held']:
            for k in range(PLATE_LIFT_STEPS):
                row = raise_hand('lift', k, PLATE_LIFT_PULL_M, PLATE_LIFT_STEP_M)
                if (not row['held'] or (row['clear_mm'] or 0.) >= 1000 * PLATE_LIFT_CLEAR_M
                        or (row['off_pan_mm'] or 0.) >= 1000 * PLATE_OFF_PAN_M):
                    break
        measured['lift'] = lift
        rise = float(body_pose(env, cover_body)[0][2] - z0)
        lo, hi = body_aabb(env, cover_body)
        support_top = max(geom_points(env, g)[:, 2].max() for g in support_ids) if support_ids else top
        clear = float(lo[2] - support_top)
        off_pan = float(geom_distance(env, plate_ids, support_ids)['distance']) if support_ids else .1
        measured['rise_m'], measured['clear_of_pan_m'] = round(rise, 4), round(clear, 4)
        measured['off_pan_m'], measured['tilt_lifted_deg'] = round(off_pan, 4), round(tilt_deg(body_pose(env, cover_body)[1]), 1)
        if beam0 is not None:
            measured['beam_deg_lifted'] = round(float(env.beam_angle_deg()), 2)
        if not skills.grasped(geoms, fingers=True):
            raise SkillFailure(phase, f'the plate slipped out of the fingertips after {rise * 1000:.0f} mm (it tipped '
                               f'{lift[-1]["tilt_deg"]:.0f} degrees and the gap shut to {lift[-1]["gap_mm"]:.1f} mm; the '
                               f'unloaded pan swings up under it)', **measured)
        if clear < .010 and off_pan < PLATE_OFF_PAN_M:      # off the rims counts too: a plate dragged out on
            raise SkillFailure(phase, f'the plate rose {rise * 1000:.0f} mm and came {off_pan * 1000:.0f} mm off the '
                               f'pan rims but hangs {-clear * 1000:.0f} mm below their top, so the pan is not free of '
                               f'it', **measured)                  # edge still hangs below the rim top
        a = -toward_robot                       # "into the pan": the retreat after the set-down backs away from it
        a = np.array([a[0], a[1], 0.])
    with skills.segment('carry the plate off the balance and set it down', kind='uncover', skill='lift_plate_from_pan'):
        place = measured.setdefault('place', [])
        swing = linked_bodies(env, [model.body_id2name(int(model.geom_bodyid[g])) for g in support_ids])
        swing_ids = [g for name in swing for g in body_geom_ids(env, name)
                     if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0]
        swing_shapes = plan_keepouts(env, swing)
        measured['balance_bodies'] = swing

        def plan_clear():
            """Metres between the plate's own disc and the balance's plan footprints, where it stands now."""
            xy = O.world_to_frame_xy(env, body_pose(env, cover_body)[0][:2])
            disc = S.Disc((float(xy[0]), float(xy[1])), float(r))
            return float(min([S.clearance(disc, shape) for shape in swing_shapes], default=1.))

        def free_of_balance():
            """Metres from the plate's hulls to the balance's, and from its disc to the balance in plan."""
            near = geom_distance(env, plate_ids, swing_ids)['distance'] if swing_ids else .1
            return float(near), plan_clear()

        def note(tag, goal=None):
            """Where the hand and the plate stand at one stage of the set-down, and what the move left over."""
            at = skills.state()['eef']
            low = body_aabb(env, cover_body)[0]
            near, plan = free_of_balance()
            place.append(dict(step=tag, eef_above_top_m=round(float(at[2]) - top, 4),
                              plate_bottom_above_top_m=round(float(low[2]) - top, 4),
                              gap_mm=round(1000 * skills.finger_gap(), 2), free_mm=round(1000 * near, 1),
                              plan_mm=round(1000 * plan, 1), held=bool(skills.grasped(geoms, fingers=True)),
                              short_mm=None if goal is None else round(1000 * float(np.linalg.norm(at - goal)), 1)))
            return place[-1]
        hand = skills.state()['eef']
        goal = np.array([hand[0], hand[1], max(hand[2], top + .22)])
        skills.move(goal, rotation, 0., phase=phase + '_raise', ticks=140, tolerance=.01, required=False)
        note('raise', goal)
        standing = surface_keepouts(env, exclude=[cover_body] + swing, plan=True) + swing_shapes

        def props_clear():
            """Metres between the plate's own disc, where it stands now, and everything else on the work top."""
            xy = O.world_to_frame_xy(env, body_pose(env, cover_body)[0][:2])
            disc = S.Disc((float(xy[0]), float(xy[1])), float(r))
            return float(min([S.clearance(disc, shape) for shape in standing], default=1.))
        # Where to: the nearest free patch of counter measured against the balance's *plan* footprints
        # (:func:`geom_plan_box`) instead of its bounding discs, which reach 0.136 m from a 4 mm rod and push
        # every spot beyond half a metre - the carry that shears the plate out of a 3 mm lip.
        if spot is None:
            keeps = (standing + overhead_keepouts(env, H.headroom_needed(
                'plate_lift', cover_height=float(info.get('thickness') or .02))))
            base_xy, base_yaw = skills.base_pose()
            here = O.world_to_frame_xy(env, body_pose(env, cover_body)[0][:2])
            chosen, seen = None, []
            for radius, gap in ((r + PLATE_DROP_MARGIN_M, PLATE_DROP_GAP_M), (r + .002, PLATE_DROP_GAP_M / 2.)):
                for point in free_spots(O.work_surface_rect(env), keeps, radius, prefer=here, gap=gap, edge=.01):
                    world = O.frame_to_world_xy(env, point)
                    ahead, lateral = _reach_from(base_xy, base_yaw, world)
                    if len(seen) < 6:
                        seen.append(dict(frame=[round(float(v), 3) for v in point], ahead_m=round(ahead, 3),
                                         lateral_m=round(lateral, 3)))
                    if REACH_BAND_M[0] <= ahead <= REACH_BAND_M[1] and abs(lateral) <= .30:
                        chosen = np.array([float(world[0]), float(world[1])])
                        break
                if chosen is not None:
                    break
            measured['spots_seen'] = seen
            if chosen is None:
                # Nothing in reach. odd_parcel layout 25 style 3 spreads six parcels over the whole counter in
                # front of the balance and the only free patches stand 0.81 m ahead of the base, behind it; the
                # old fallback took set_down_spot's nearest patch, which only bands the reach *ahead* and handed
                # back a spot 0.56 m sideways. The arm stalls there, the fingers go on squeezing a 3.3 mm lip
                # and extrude the plate onto the parcels (certify 02:01 and 02:07, tilted 37 and 33 degrees).
                # Better to say so: this kitchen has nowhere to put the plate, so the build can drop the
                # pan-plate cover for it rather than have the oracle leave a plate leaning on the goods.
                far = min(seen, key=lambda row: row['ahead_m']) if seen else None
                raise SkillFailure(phase, f'the plate is off the pan but there is nowhere within reach to set it '
                                   f'down: it needs a free patch of counter {2000 * (r + PLATE_DROP_MARGIN_M):.0f} mm '
                                   f'across and the nearest one stands '
                                   + ('nowhere on this top' if far is None else
                                      f"{far['ahead_m']:.2f} m ahead of the base and {abs(far['lateral_m']):.2f} m to "
                                      f'the side, outside the {REACH_BAND_M[0]:.2f}-{REACH_BAND_M[1]:.2f} m ahead and '
                                      f'0.30 m across that the arm reaches'), **measured)
            spot = chosen
        spot = np.asarray(spot, float)[:2]
        measured['place_reach_m'] = [round(v, 3) for v in _reach_from(*skills.base_pose(), spot)]
        measured['spot_from_plate_m'] = round(float(np.linalg.norm(spot - body_pose(env, cover_body)[0][:2])), 3)
        hand = skills.state()['eef']
        lo, hi = body_aabb(env, cover_body)
        hang = float(hand[2] - lo[2])
        # Straight out of the balance first, toward the robot. The pinch holds a 3 mm lip: it takes a pull along
        # the fingers (the peel just did 18 mm of one) far better than a swing across the counter, and while the
        # plate still stands inside the balance's plan footprint that swing drags it through the frame posts and
        # a shear drops it back onto the pan (certify 01:40: sheared at hop 6 of 40, came to rest on the pan).
        base_xy, base_yaw = skills.base_pose()
        plate_xy = body_pose(env, cover_body)[0][:2]
        out, room = 0., None
        for k in range(31):
            d = .01 * k
            ahead, _ = _reach_from(base_xy, base_yaw, np.asarray(hand[:2], float) + toward_robot * d)
            if ahead < REACH_BAND_M[0] + .02:
                break
            xy = O.world_to_frame_xy(env, plate_xy + toward_robot * d)
            disc = S.Disc((float(xy[0]), float(xy[1])), float(r))
            out, room = d, float(min([S.clearance(disc, shape) for shape in swing_shapes], default=1.))
            if room >= PLATE_FREE_M:
                break
        measured['carry_out_m'] = round(out, 3)
        measured['carry_out_plan_mm'] = None if room is None else round(1000 * room, 1)
        # The hand holds the *near rim*: the plate's own centre hangs a radius further from the robot, so a hand
        # sent to the spot sets the plate down a plate's radius beyond it - on layout 24 that put it 59 mm back
        # inside the balance (certify 02:11). The carry aims the plate at the spot and the hand trails it.
        carry_off = np.asarray(hand[:2], float) - np.asarray(plate_xy, float)
        measured['carry_offset_m'] = [round(float(v), 3) for v in carry_off]
        legs = ([np.array([hand[0] + toward_robot[0] * out, hand[1] + toward_robot[1] * out, hand[2]])]
                if out > .005 else [])
        legs.append(np.array([spot[0] + carry_off[0], spot[1] + carry_off[1], hand[2]]))
        was, path, ends = skills.state()['eef'], [], []    # in short hops: one long traverse swings the plate
        for leg in legs:                                   # out of a pinch that only holds a 3 mm lip
            n = max(1, int(math.ceil(float(np.linalg.norm(leg - was)) / PLATE_CARRY_HOP_M)))
            path += [was + (leg - was) * (i / n) for i in range(1, n + 1)]
            was, ends = leg, ends + [len(path)]
        goal, hops = legs[-1], len(path)
        measured['carry_hops'], measured['carry_legs'] = hops, len(legs)
        # The fingers only hold their place through the carry. Going on closing on a 3.3 mm lip extrudes it:
        # certify 02:01 lost 2.5 mm of gap over fourteen hops under a close command and the plate slid 95 mm
        # down the fingers before it let go, where the lift, which holds, drifts half a millimetre in fourteen.
        ride, sheared, lows = None, None, 0
        for i, at in enumerate(path, 1):
            if ride is not None:            # ... and it comes down as it wins room, so a shear drops it a hand's
                hang = float(skills.state()['eef'][2] - body_aabb(env, cover_body)[0][2])   # breadth, not the
                at = np.array([at[0], at[1], top + hang + ride])         # 155 mm that landed it on its rim
            skills.move(at, rotation, 0., phase=f'{phase}_over{i}', ticks=55, tolerance=.008, required=False)
            skills.hold(4, 0., rotation, phase=f'{phase}_over{i}_hold')
            if not skills.grasped(geoms, fingers=True):
                sheared = note(f'shear{i}')     # where it fell is what counts, not that the tips lost it
                break
            # Over empty counter the plate rides a hand's breadth above it; once it is out of the balance it
            # rides over the parcels (50 mm tall); over the balance it keeps the height the raise gave it.
            if i in ends[:-1]:
                note(f'leg{ends.index(i) + 1}')
            free = plan_clear()
            room = min(free, props_clear())
            want = (PLATE_CARRY_LOW_M if room > PLATE_CARRY_ROOM_M
                    else PLATE_CARRY_MID_M if min(free, room) > PLATE_FREE_M else None)
            if want != ride:
                ride, lows = want, lows + 1
                note('ride_raise' if want is None else f'ride{int(1000 * want)}')
        measured['carry_ride_m'], measured['carry_altitude_changes'] = ride, lows
        if sheared is None:
            note('over', goal)
            hang = float(skills.state()['eef'][2] - body_aabb(env, cover_body)[0][2])
            off = np.asarray(skills.state()['eef'][:2], float) - body_pose(env, cover_body)[0][:2]
            goal = np.array([spot[0] + off[0], spot[1] + off[1], top + hang + .004])
            skills.move(goal, rotation, 0., phase=phase + '_lower', ticks=220, tolerance=.005, required=False)
            note('lower', goal)
            skills.hold(25, -1., rotation, phase=phase + '_release')
            note('release')
        skills.move(skills.state()['eef'] - a * .05 + [0., 0., .12], rotation, -1., phase=phase + '_retreat', ticks=140,
                    required=False)
        skills.hold(40, -1., phase=phase + '_settle')
        note('settle')
        _tuck(skills, -1., phase=phase + '_done')
    after, rot = body_pose(env, cover_body)
    lo, hi = body_aabb(env, cover_body)
    near, plan = free_of_balance()
    record = dict(measured, rise_m=rise, spot_world=spot.tolist(), plate_bottom_above_top_m=float(lo[2] - top),
                  free_of_balance_m=round(near, 4), plan_clear_m=round(plan, 4), sheared=sheared,
                  tilt_deg=tilt_deg(rot), ticks=skills.tick - start)
    if parcels0:
        record['parcel_moved_mm'] = {name: round(1000 * float(np.linalg.norm(body_pose(env, name)[0] - was)), 1)
                                     for name, was in parcels0.items()}
    if beam0 is not None:
        record['beam_deg_before'] = beam0
        record['beam_deg_after'] = float(env.beam_angle_deg())
    if hasattr(env, 'balance_displacement_m'):
        record['balance_displacement_m'] = float(env.balance_displacement_m())
        if record['balance_displacement_m'] > .02:
            raise SkillFailure(phase, f"the balance moved {record['balance_displacement_m'] * 1000:.0f} mm", **record)
    wrong = plate_landed(near, float(lo[2] - top), tilt_deg(rot))
    if wrong:
        raise SkillFailure(phase, f'the plate is off the pan but {wrong}'
                           + ('' if sheared is None else f" (it sheared out of the fingertips at hop {sheared['step'][5:]}"
                              f" of {measured['carry_hops']})"), **record)
    _no_collateral(env, events_before, phase)
    return record


def _side_candidates(approach):
    """The two side-grasp rotations with horizontal fingers and the tool z along ``approach`` (horizontal), the
    180-degree flips of each other about the tool z."""
    z = _unit([approach[0], approach[1], 0.])
    x = _unit(np.cross([0., 0., 1.], z))
    first = np.column_stack([x, np.cross(z, x), z])
    return [first, first @ Rotation.from_euler('z', np.pi).as_matrix()]


def _pitched(rotation, pitch_deg):
    """``rotation`` with its tool z turned ``pitch_deg`` toward the floor about the tool x (finger) axis."""
    rotation = np.asarray(rotation, float)
    for sign in (-1., 1.):
        candidate = rotation @ Rotation.from_euler('x', sign * math.radians(pitch_deg)).as_matrix()
        if candidate[2, 2] < rotation[2, 2] - 1e-9:
            return candidate
    return rotation


def pour_cup(skills, cup_body, target_xy=None, *, tilt=POUR_TILT_DEG, max_tilt=POUR_TILT_MAX_DEG, hold_z=POUR_HOLD_M,
             alternative=0, phase='pour'):
    """Pinch the wide cup's upper handle bar from above, lift it, hold it over ``target_xy`` (a free spot by
    default), turn the tool about the finger axis until the cup tilts past ``tilt`` degrees so the cube slides
    out, level it and set it down.

    Why this grasp (measured on stamps layout 4 style 4, SKILLS-2 probes 2026-09-21): robosuite's Panda fingers
    are position actuators, so a pinch presses with kp (1000 N/m) times the half width of what it holds. The
    side pinch of the 14 mm grip bar (7 N per pad) held the lift but the cup twisted out of it at 83-125
    degrees of the roll about the handle axis, when the cup's weight, 8 cm out on the bar, loads the pads in
    shear (capacity about 2 x 7 N x 8 mm); a pinch of the 4 mm rim wall (2 N) dropped the cup at the lift.
    Pinched across the 12 mm upper bar 5 cm from the cup's axis, the same weight loads the pads about their
    normal (torsional friction, 0.05 x 6 N), and turning the tool about that axis, with the cup's top leaning
    away from the handle so its body swings down under the pinch, poured the cube at 114-116 degrees with the
    gap unchanged (12.9 mm) before, during and after (cup_bar_probe2, both wrist flips)."""
    env = skills.env
    info = cover_record(env, cup_body)
    inside = info.get('object') or (env.task_spec['objects'].get(cup_body) or {}).get('covers')
    top = skills.top_z
    events_before = len(env.events)
    start = skills.tick
    handle_geoms = [f'{cup_body}_handle_grip', f'{cup_body}_handle_lower', f'{cup_body}_handle_upper']
    cup_pos0, cup_rot0 = body_pose(env, cup_body)
    cup_height = float(info.get('height') or .11)
    need = H.headroom_needed('wide_cup', cover_height=cup_height)

    def bar_point():
        pos, rot = body_pose(env, cup_body)
        axis = _unit(np.r_[(rot @ np.array([1., 0., 0.]))[:2], 0.])       # cup axis -> handle
        return pos + axis * POUR_BAR_RADIAL_M + np.array([0., 0., POUR_BAR_TOP_M + POUR_BAR_PINCH_DZ]), axis

    with skills.segment(f'pinch the cup {cup_body} by its handle bar from above and lift it', kind='uncover', skill='pour_cup'):
        _tuck(skills, -1., phase=phase + '_clear')
        bar, handle_axis = bar_point()
        _stand_within_reach(skills, bar, phase=phase + '_stand')
        bar, handle_axis = bar_point()
        finger_yaw = math.atan2(handle_axis[0], -handle_axis[1])            # the pads close across the bar's width
        rotation = skills.grasp_rotation('top', alternative=alternative, finger_yaw=finger_yaw)
        skills.grasp(bar, 'top', rotation=rotation, above=.10, geoms=handle_geoms, lower_ticks=160, phase=phase + '_grasp')
        skills.move(skills.state()['eef'] + [0., 0., hold_z - POUR_BAR_TOP_M], rotation, 1., phase=phase + '_lift', ticks=240,
                    tolerance=.008)
        skills.hold(15, 1., rotation, phase=phase + '_lift_hold')
        rise = float(body_pose(env, cup_body)[0][2] - cup_pos0[2])
        if rise < .08:
            raise SkillFailure(phase, f'cup rose only {rise * 1000:.0f} mm', rise=rise)
        if not skills.grasped(handle_geoms):
            raise SkillFailure(phase, 'the handle bar slipped out during the lift')
    with skills.segment('tilt the cup over a free spot so the cube slides out', kind='uncover', skill='pour_cup'):
        axis = rotation[:, 0]                                                # the finger axis: the pads' normal
        sign = 1.
        for candidate in (1., -1.):
            trial = Rotation.from_rotvec(candidate * math.radians(30.) * axis).as_matrix() @ rotation
            up = (trial @ rotation.T)[:, 2]
            if float(np.dot(up[:2], handle_axis[:2])) < 0.:                 # the cup's top leans away from the handle
                sign = candidate
                break

        def tilted(angle):
            return Rotation.from_rotvec(sign * math.radians(angle) * axis).as_matrix() @ rotation

        def cup_up(angle):
            """The cup's own up axis at a tool turn of ``angle`` (the cup is rigid in the hand)."""
            return (tilted(angle) @ rotation.T)[:, 2]

        mouth_dir = _unit(np.r_[cup_up(tilt)[:2], 0.])[:2]
        cup_pos, _ = body_pose(env, cup_body)
        if target_xy is None:
            target_xy = set_down_spot(skills, .07, exclude=[cup_body, inside] if inside else [cup_body],
                                      prefer_world=cup_pos[:2] + mouth_dir * .12, headroom=need)
        target_xy = np.asarray(target_xy, float)[:2]
        _stand_within_reach(skills, [target_xy[0], target_xy[1], top], phase=phase + '_pour_stand', gripper=1.)
        cup_pos, _ = body_pose(env, cup_body)
        hand = skills.state()['eef']
        centre_xy = target_xy - mouth_dir * .10                              # pour_landing inverted
        hold = np.array([*(centre_xy + (hand[:2] - cup_pos[:2])), top + hold_z])
        skills.move(hold, rotation, 1., phase=phase + '_hold_point', ticks=220, tolerance=.008, required=False)
        reached = 0.

        def out():
            if not inside:
                return True
            p = body_pose(env, inside)[0]
            c = body_pose(env, cup_body)[0]
            return bool(p[2] < c[2] - .06 or np.linalg.norm(p[:2] - c[:2]) > .10)

        for index, angle in enumerate(POUR_TILT_STEPS):
            if angle > tilt + 1e-9:
                break
            skills.move(skills.state()['eef'], tilted(angle), 1., phase=f'{phase}_tilt_{index}', ticks=150, tolerance=.01,
                        required=False)
            skills.hold(20, 1., None, phase=f'{phase}_tilt_hold_{index}')
            reached = float(angle)
            if out():
                break
        if not out() and max_tilt > reached:
            skills.move(skills.state()['eef'], tilted(max_tilt), 1., phase=phase + '_tilt_more', ticks=150, tolerance=.01,
                        required=False)
            skills.hold(30, 1., None, phase=phase + '_pour_hold_more')
            reached = float(max_tilt)
        cup_tilt = tilt_deg(body_pose(env, cup_body)[1])
        poured = out()
        for index, angle in enumerate((reached / 2, 0.)):
            skills.move(skills.state()['eef'], tilted(angle), 1., phase=f'{phase}_level_{index}', ticks=150, tolerance=.01,
                        required=False)
        if not poured:
            raise SkillFailure(phase, f'the cube stayed in the cup at {cup_tilt:.0f} deg of tilt', cup_tilt_deg=cup_tilt)
        if not skills.grasped(handle_geoms):
            raise SkillFailure(phase, 'the handle bar slipped out during the pour')
    with skills.segment('set the cup down', kind='uncover', skill='pour_cup'):
        down_spot = set_down_spot(skills, .09, exclude=[cup_body], prefer_world=cup_pos0[:2], headroom=need)
        _stand_within_reach(skills, [down_spot[0], down_spot[1], top], phase=phase + '_down_stand', gripper=1.)
        hand = skills.state()['eef']
        cup_pos, _ = body_pose(env, cup_body)
        lo, hi = body_aabb(env, cup_body)
        hang = float(hand[2] - lo[2])
        target = np.array([*(down_spot + (hand[:2] - cup_pos[:2])), top + hang + .004])
        skills.place(target, rotation, above=.10, lower_ticks=220, phase=phase + '_place')
        _tuck(skills, -1., phase=phase + '_done')
    cup_after, cup_rot = body_pose(env, cup_body)
    record = dict(cup=cup_body, object=inside, target_xy=target_xy.tolist(), tilt_reached_deg=reached, cup_tilt_deg=cup_tilt,
                  rise_m=rise, spot_world=down_spot.tolist(), cup_tilt_after_deg=tilt_deg(cup_rot),
                  cup_z_error_m=float(cup_after[2] - cup_pos0[2]), ticks=skills.tick - start)
    if inside:
        p = body_pose(env, inside)[0]
        rect = O.work_surface_rect(env)
        c = O.world_to_frame_xy(env, p[:2])
        record['object_xy_world'] = p[:2].tolist()
        record['object_z_above_top_m'] = float(p[2] - top)
        record['object_clear_of_cup_m'] = float(np.linalg.norm(p[:2] - cup_after[:2]) - .046)
        record['object_on_top'] = bool(rect.x0 <= c[0] <= rect.x1 and rect.y0 <= c[1] <= rect.y1 and -.02 < p[2] - top < .08)
        if not record['object_on_top']:
            raise SkillFailure(phase, 'the poured object is not on the work top', **record)
        if record['object_clear_of_cup_m'] < 0.:
            raise SkillFailure(phase, 'the object is still inside the cup', **record)
    if tilt_deg(cup_rot) > 15. or abs(cup_after[2] - cup_pos0[2]) > .03:
        raise SkillFailure(phase, 'the cup is not standing upright on the surface', **record)
    _no_collateral(env, events_before, phase)
    return record


# ---- look skills ------------------------------------------------------------------------------------
def _planner(skills):
    """A grid planner over the room's fixture footprints, built once per Skills instance (None without walls)."""
    if not hasattr(skills, '_uncover_planner'):
        from roboquest.search.planner import GridPlanner
        floor = O.floor_map(skills.env)
        planner = None
        if floor.get('bounds'):
            try:
                planner = GridPlanner(floor['bounds'], floor['blockers'])
            except Exception:
                planner = None
        skills._uncover_planner = planner
    return skills._uncover_planner


def drive_to_stance(skills, stance, *, gripper=None, speed=(.20, .40), tolerance=.06, tuck=True, phase='drive'):
    """Drive the base to an observe stance ``{xy, yaw, edge, standoff}`` under closed-loop control: the
    hand is tucked first, the route is planned around the fixture footprints (straight line when no map
    exists), long legs with a big heading change are driven nose first, the final leg turns to the stance
    yaw and re-centres. Fails when the base ends more than ``tolerance`` from the stance."""
    xy = np.asarray(stance['xy'], float)[:2]
    yaw = float(stance['yaw'])
    gripper = skills.gripper if gripper is None else float(gripper)
    start = skills.tick
    if tuck:
        _tuck(skills, gripper, phase=phase + '_tuck')
    start_xy, start_yaw = skills.base_pose()
    planner = _planner(skills)
    legs = []
    if planner is not None and np.linalg.norm(xy - start_xy) > .30:
        try:
            waypoints, _ = planner.plan(start_xy, xy)
            legs = [np.asarray(w, float)[:2] for w in waypoints]
        except ValueError:
            legs = []
    legs = [w for w in legs if np.linalg.norm(w - xy) > .10] + [xy]
    # Enter the stance square. The base is 0.70 m wide and a stance puts its frame 0.20 m from the surface edge,
    # so a final leg that arrives sideways or nose first runs the base's side, or the arm tucked ahead of it,
    # into the fixture before the frame reaches the stance, and the turn on the spot that follows sweeps the
    # arm across it (measured on stamps layout 4, annex_stamps4: both fetch attempts stalled 8-9 cm short of
    # the home stance and the first lost the carried stamp in the turn). From further out than DOCK_BACK_M the
    # route ends at a dock point that far behind the stance along its heading; the base turns to the stance
    # heading there and drives straight in.
    facing = np.array([math.cos(yaw), math.sin(yaw)])
    dock = xy - facing * DOCK_BACK_M
    docking = False
    if np.linalg.norm(xy - start_xy) > DOCK_BACK_M + .05:
        final_heading = math.atan2(*(xy - (legs[-2] if len(legs) > 1 else start_xy))[::-1])
        if abs(_wrap(final_heading - yaw)) > DOCK_MAX_TURN_RAD or abs(_wrap(start_yaw - yaw)) > DOCK_MAX_TURN_RAD:
            cell = planner.cell(dock) if planner is not None else None
            if planner is None or (planner.inside(cell) and planner.free[cell]):
                legs = [w for w in legs[:-1] if np.linalg.norm(w - dock) > .10 and np.linalg.norm(w - xy) > DOCK_BACK_M - .05]
                legs.append(dock)
                docking = True
    for index, waypoint in enumerate(legs):
        cur_xy, cur_yaw = skills.base_pose()
        delta = waypoint - cur_xy
        length = float(np.linalg.norm(delta))
        if length < .03:
            continue
        heading = math.atan2(delta[1], delta[0])
        turn = _wrap(heading - cur_yaw)
        if abs(turn) > math.radians(35.) and length > .35:
            if index == 0 and planner is not None:
                back = cur_xy - .25 * np.array([math.cos(cur_yaw), math.sin(cur_yaw)])
                cell = planner.cell(back)
                if planner.inside(cell) and planner.free[cell]:
                    skills.navigate(back, None, gripper, speed=speed, tolerant=True, phase=f'{phase}_backoff')
            try:
                skills.turn(heading, gripper, phase=f'{phase}_face_{index}')
            except SkillFailure:
                pass
        skills.navigate(waypoint, None, gripper, speed=speed, tolerant=True, phase=f'{phase}_leg_{index}')
    if docking:
        try:
            skills.turn(yaw, gripper, phase=phase + '_dock_turn')
        except SkillFailure:
            pass
    result = skills.navigate(xy, yaw, gripper, speed=speed, tolerant=True, phase=phase + '_final')
    end_xy, end_yaw = skills.base_pose()
    error = float(np.linalg.norm(end_xy - xy))
    yaw_error = abs(_wrap(yaw - end_yaw))
    record = dict(stance=deepcopy(stance), legs=len(legs), docked=docking, position_error_m=error, yaw_error_rad=yaw_error,
                  stalled=bool(result.get('stalled')), ticks=skills.tick - start)
    if error > tolerance or yaw_error > .08:
        raise SkillFailure(phase, f'base ended {error:.3f} m / {yaw_error:.3f} rad from the stance', **record)
    return record


def look_at(skills, body, *, phase='look'):
    """Drive to a stance from which ``body`` is reachable and in view (``observe.look_reachable``) and return
    the object's pose as the privileged oracle reads it, plus the camera check when the env renders."""
    env = skills.env
    look = O.look_reachable(env, body)
    if not look['passed']:
        raise SkillFailure(phase, f'{body}: no reachable viewpoint (' + '; '.join(look['reasons']) + ')')
    stance = dict(look['stance'], standoff=O.STANDOFF_M)
    with skills.segment(f'drive to a viewpoint for {body}', kind='look', skill='look_at'):
        drive = drive_to_stance(skills, stance, phase=phase + '_drive')
    pos, rot = body_pose(env, body)
    visible = None
    if getattr(env, 'has_offscreen_renderer', False):
        try:
            visible = O.visible_in_cameras(env, [body], image_size=256)['visible'][body]
        except Exception:
            visible = None
    return dict(body=body, stance=stance, drive=drive, position=pos.tolist(), rotation=rot.tolist(),
                visible_in_cameras=visible, distance_m=float(np.linalg.norm(pos[:2] - np.asarray(stance['xy']))))


def fetch(skills, body, *, stance=None, offset=(0., 0., 0.), geoms=None, finger_yaw=None, alternative=0, lift=.12,
          spot=None, home=None, phase='fetch', fingers=False, carry_speed=(.20, .40)):
    """Drive to ``stance`` (default: ``look_reachable``'s), grasp ``body`` from above at ``body pose + offset``
    (body frame), carry it back to ``home`` (default: the reset stance) and set it down on a free spot of the
    work frame. Returns a record with the new position. ``fingers`` lets the finger meshes count as the hold
    (``Skills.grasp``; a leaning thin rim such as the marked bowl's). ``carry_speed`` is the base velocity band
    for the drive home with the object in the pinch (the marked mugs lost their handle pinch on the docking
    route home at the default band, three of three look instances; the containers oracle carries at (.12, .25))."""
    env = skills.env
    top = skills.top_z
    start = skills.tick
    events_before = len(env.events)
    if stance is None:
        look = O.look_reachable(env, body)
        if not look['passed']:
            raise SkillFailure(phase, f'{body}: no reachable stance (' + '; '.join(look['reasons']) + ')')
        stance = dict(look['stance'], standoff=O.STANDOFF_M)
    home = home or dict(xy=list(skills.reset_base['xy']), yaw=float(skills.reset_base['yaw']), edge='front', standoff=O.STANDOFF_M)
    geoms = list(geoms) if geoms else collision_geom_names(env, body)
    offset = np.asarray(offset, float)
    with skills.segment(f'drive to the annex and pick up {body}', kind='look', skill='fetch'):
        drive_to_stance(skills, stance, phase=phase + '_drive')
        pos, rot = body_pose(env, body)
        target = pos + rot @ offset
        _stand_within_reach(skills, target, phase=phase + '_stand')
        pos, rot = body_pose(env, body)
        target = pos + rot @ offset
        obj = (getattr(env, 'objects', {}) or {}).get(body)
        if finger_yaw is None and obj is not None and hasattr(obj, 'size'):
            # a box (a parcel) is pinched across its thinner horizontal side: with the default finger axis the
            # yawed 7 x 10 cm parcels closed on air or held a corner that slipped (odd_parcel certificates)
            size = np.asarray(obj.size, float)
            axis = rot[:, int(np.argmin(size[:2]))]
            finger_yaw = math.atan2(axis[1], axis[0])
        rotation = skills.grasp_rotation('top', alternative=alternative, finger_yaw=finger_yaw)
        skills.grasp(target, 'top', rotation=rotation, above=.14, geoms=geoms, lower_ticks=160, phase=phase + '_grasp',
                     fingers=fingers)
        skills.move(target + [0., 0., lift], rotation, 1., phase=phase + '_lift', ticks=200)
        skills.hold(10, 1., rotation, phase=phase + '_lift_hold')
        rise = float(body_pose(env, body)[0][2] - pos[2])
        if rise < .5 * lift:
            raise SkillFailure(phase, f'{body} rose only {rise * 1000:.0f} mm', rise=rise)
        if not skills.grasped(geoms, fingers=fingers):
            raise SkillFailure(phase, f'{body} slipped out during the lift')
        lo, hi = body_aabb(env, body)
        hang = float(skills.state()['eef'][2] - lo[2])
    with skills.segment(f'carry {body} back to the work frame', kind='look', skill='fetch'):
        carry_z = max(top, float(stance.get('top_z', top))) + hang + .12
        _tuck(skills, 1., height=carry_z, phase=phase + '_carry')
        drive_to_stance(skills, home, gripper=1., speed=tuple(carry_speed), phase=phase + '_home')
        if not skills.grasped(geoms, fingers=fingers):
            raise SkillFailure(phase, f'{body} slipped out during the drive')
        if spot is None:
            # nearest to the point straight ahead of the home stance: without a preference the grid's first free point
            # lay at the far end of the counter and the base stalled 0.74 m into its lateral slide (layout 10 style 13)
            home_xy, home_yaw = skills.base_pose()
            ahead_of_home = np.asarray(home_xy, float) + np.array([math.cos(home_yaw), math.sin(home_yaw)]) * GRASP_AHEAD_M
            spot = set_down_spot(skills, footprint_radius(env, body) + .01, exclude=[body], prefer_world=ahead_of_home,
                                 headroom=H.headroom_needed('fetch', object_height=float(hi[2] - lo[2])),
                                 carry_from=skills.state()['eef'][:2], carry_z=float(skills.state()['eef'][2]))
        spot = np.asarray(spot, float)[:2]
        _stand_within_reach(skills, [spot[0], spot[1], top], phase=phase + '_place_stand', gripper=1.)
        hand = skills.state()['eef']
        pos, _ = body_pose(env, body)
        lo, hi = body_aabb(env, body)
        hang = float(hand[2] - lo[2])
        target = np.array([*(spot + (hand[:2] - pos[:2])), top + hang + .004])
        skills.place(target, skills.state()['eef_rotation'], above=.10, lower_ticks=200, phase=phase + '_place')
        _tuck(skills, -1., phase=phase + '_done')
    pos, rot = body_pose(env, body)
    rect = O.work_surface_rect(env)
    c = O.world_to_frame_xy(env, pos[:2])
    lo, hi = body_aabb(env, body)
    record = dict(body=body, stance=deepcopy(stance), rise_m=rise, spot_world=spot.tolist(), position=pos.tolist(),
                  frame_xy=[float(c[0]), float(c[1])], bottom_above_top_m=float(lo[2] - top), tilt_deg=tilt_deg(rot),
                  ticks=skills.tick - start)
    if not (rect.x0 <= c[0] <= rect.x1 and rect.y0 <= c[1] <= rect.y1) or lo[2] - top > .03 or lo[2] - top < -.02:
        raise SkillFailure(phase, f'{body} is not standing on the work top after the fetch', **record)
    _no_collateral(env, events_before, phase)
    return record


def _push_aside(skills, body, thin_side, *, drive=None, alternative=0, start=None, events_before=None,
                phase='aside'):
    """Push an occluder that is too wide to pinch out of its target's line of sight, leaving it standing.

    :func:`move_aside` comes here for the one case its pinch cannot serve: a body whose thinnest horizontal
    side does not fit between the fingers (:data:`GRIPPER_OPENING_M`). The policy's own skill set exposes
    :func:`push_cover`, so such an occluder is content a policy can handle and the library does the same.

    The heading is :func:`push_aside_plan`'s - away from every sight corridor, into free surface chosen the
    way the pinch's set-down spot is (:func:`set_down_spot`: the surface keep-outs, the overhead fixtures, the
    nearest qualifying spot), reachable together with its contact points from one stance. The push is
    :func:`push_cover`'s loop in short passes: the closed hand comes down :data:`ASIDE_PUSH_BEHIND_M` behind the
    body's hull, where the fingers clear its shoulder, to :data:`ASIDE_PUSH_CONTACT_M` over the top - well under
    a tall body's centre of mass, and refused outright when the hand stops short of that - pushes
    :data:`SLIDE_STEP_M` at a time (:data:`ASIDE_PUSH_STEP_M` for a body over 150 mm tall) re-reading the body's
    pose after every increment, ends the pass at :data:`ASIDE_TILT_STOP_DEG` of lean so the body can settle,
    retreats, re-reads where the body now stands and re-aims for the next pass.

    Postconditions (the pinch's, plus the two a push can get wrong): the body still stands on the work top,
    upright within :data:`ASIDE_TILT_MAX_DEG`, out of the line of sight of every policy camera that needs the
    target - measured with the cameras at the reset stance, the ones the occluder was placed against - clear
    of the target's own approach column, and nothing left the surface. Every refusal carries its measured
    numbers."""
    env = skills.env
    top = skills.top_z
    start = skills.tick if start is None else int(start)
    events_before = len(env.events) if events_before is None else int(events_before)
    pos, rot = body_pose(env, body)
    lo, hi = body_aabb(env, body)
    height = float(hi[2] - lo[2])
    radius = footprint_radius(env, body) + .01
    # the push slides the body and never turns it, so the plan walks its own footprint rectangle along the
    # heading instead of the disc the pinch's set-down reserves: on a crowded top that is the whole search
    obj = (getattr(env, 'objects', {}) or {}).get(body)
    footprint = None
    if obj is not None:
        size = np.asarray(obj.size, float)
        footprint = (float(size[0]) / 2 + .01, float(size[1]) / 2 + .01,
                     float(math.atan2(rot[1, 0], rot[0, 0]) - float(env.work['yaw'])))
    refusal = (f'{body} is {thin_side * 1000:.0f} mm across its thinnest side, wider than the '
               f'{GRIPPER_OPENING_M * 1000:.0f} mm the gripper opens')
    record = occluder_record(env, body)
    target = record.get('target')
    if target is None:
        raise SkillFailure(phase, refusal + ', and no occluder record says what it hides, so there is no '
                           'heading to push it out of', thin_side_m=float(thin_side))
    target_pos = body_pose(env, target)[0]
    target_radius = float(record.get('target_radius') or footprint_radius(env, target))
    target_height = float(record.get('target_height') or .10)
    # the policy cameras at the reset stance: the occluder was placed against these, and they do not move with
    # the base the skill has driven to
    frusta = O.camera_frusta_at(skills.reset_base['xy'], float(skills.reset_base['yaw']))
    points = O.box_points(target_pos[:2], (target_radius, target_radius), top, top + target_height, inflate=.01)
    needed = O.seen_by(frusta, points)
    origin = O.world_to_frame_xy(env, target_pos[:2])
    dirs = [_unit(O.world_to_frame_xy(env, f['pos'][:2]) - origin) for f in frusta if needed[f['name']]]
    keeps = surface_keepouts(env, exclude=[body]) \
        + overhead_keepouts(env, H.headroom_needed('push_path', cover_height=height))
    rect = O.work_surface_rect(env)
    base_xy, base_yaw = skills.base_pose()
    base = (O.world_to_frame_xy(env, base_xy),
            skills.rotation[:2, :2].T @ np.array([math.cos(base_yaw), math.sin(base_yaw)]))
    here = O.world_to_frame_xy(env, pos[:2])
    # the line-of-sight test the plan gates on is the library's own (:func:`observe.occludes` reads the same
    # rays), applied to this body inflated by ASIDE_SIGHT_MARGIN_M so the push ends with room to spare
    grown = (hi - lo) / 2 + ASIDE_SIGHT_MARGIN_M
    middle_z = float((lo[2] + hi[2]) / 2)

    def sight_test(spot_frame):
        world = O.frame_to_world_xy(env, spot_frame)
        clear, _ = O.rays_clear(frusta, points, np.array([world[0], world[1], middle_z]), grown, 0.)
        return min([clear[name] for name in needed if needed[name]] or [len(points)])

    plan = push_aside_plan(here, radius, origin, target_radius, dirs, rect, keeps, base=base,
                           sight_test=sight_test, footprint=footprint)
    if not plan['passed']:          # set_down_spot's own relaxation: half the gap, closer to the edge
        relaxed = push_aside_plan(here, radius, origin, target_radius, dirs, rect, keeps, base=base,
                                  sight_test=sight_test, footprint=footprint, gap=SPOT_GAP_M * .5, edge=.01)
        if relaxed['passed']:
            plan = dict(relaxed, relaxed=True)
    if not plan['passed']:
        raise SkillFailure(phase, refusal + ', and no heading pushes it clear: ' + '; '.join(plan['reasons']),
                           thin_side_m=float(thin_side), target=target, push=plan)
    d_frame = np.asarray(plan['heading'], float)
    d_world = skills.direction_to_world([d_frame[0], d_frame[1], 0.])[:2]
    distance = float(plan['distance_m'])
    down = skills.grasp_rotation('top', alternative=alternative,
                                 finger_yaw=math.atan2(d_world[1], d_world[0]) + math.pi / 2)

    def along(world_xy):
        return float(np.dot(O.world_to_frame_xy(env, world_xy), d_frame))

    origin_along = along(pos[:2])
    # a tall body is pushed in finer increments: the lean is read after every one of them
    push_step = SLIDE_STEP_M if height < .15 else ASIDE_PUSH_STEP_M
    pushed, passes, steps, idle, high = 0., 0, 0, 0, 0.
    while pushed < distance - .005 and passes < ASIDE_PUSH_PASSES and steps < 40:
        passes += 1
        now, rot = body_pose(env, body)
        lo, hi = body_aabb(env, body)
        reach = .5 * (abs(d_world[0]) * float(hi[0] - lo[0]) + abs(d_world[1]) * float(hi[1] - lo[1]))
        contact = now[:2] - d_world * (reach + ASIDE_PUSH_BEHIND_M)
        span = ASIDE_PUSH_BEHIND_M + min(ASIDE_PUSH_PASS_M, distance - pushed + .01)   # the hand's own travel
        middle = contact + d_world * span / 2
        _stand_within_reach(skills, [middle[0], middle[1], top], phase=f'{phase}_push{passes}_stand', gripper=1.)
        drop = fingertip_drop(skills)
        z = float(top + drop + min(ASIDE_PUSH_CONTACT_M, max(height * .25, .01)))
        skills.move([contact[0], contact[1], float(hi[2]) + .10], down, 1., phase=f'{phase}_push{passes}_above',
                    ticks=170, tolerance=.008, required=False)
        skills.move([contact[0], contact[1], z], down, 1., phase=f'{phase}_push{passes}_lower', ticks=140,
                    tolerance=.006, required=False)
        # the fingertips have to reach the body's foot: a hand that stopped high pushes above the centre of
        # mass and tips it over instead of sliding it
        above = float(skills.state()['eef'][2]) - drop - top
        high = max(high, above)
        if above > max(.06, height * .35):
            raise SkillFailure(phase, f'the hand stopped {above * 1000:.0f} mm above the top behind {body}, too high '
                               f'to push a {height * 1000:.0f} mm body without tipping it', body=body, mode='push',
                               contact_above_top_m=above, pushed_m=float(pushed), distance_m=distance,
                               passes=passes, heading_frame=[float(d_frame[0]), float(d_frame[1])])
        done, last, stalls, travelled = 0., along(body_pose(env, body)[0][:2]), 0, 0.
        while travelled < span and steps < 40:
            hand = skills.state()['eef']
            step = min(push_step, span - travelled + .005)
            skills.move(hand + np.r_[d_world * step, 0.], down, 1., phase=f'{phase}_push{passes}_{steps}',
                        ticks=60, tolerance=.004, required=False)
            steps += 1
            travelled += step
            after, rot = body_pose(env, body)
            moved = along(after[:2]) - last
            last = along(after[:2])
            done += max(moved, 0.)
            pushed = last - origin_along
            lean = tilt_deg(rot)
            if lean > ASIDE_TILT_MAX_DEG:
                raise SkillFailure(phase, f'{body} leaned {lean:.0f} degrees over after {pushed * 1000:.0f} mm of '
                                   f'the {distance * 1000:.0f} mm push', body=body, mode='push', tilt_deg=lean,
                                   pushed_m=float(pushed), distance_m=distance, passes=passes, pushes=steps,
                                   contact_above_top_m=above,
                                   heading_frame=[float(d_frame[0]), float(d_frame[1])])
            if lean > ASIDE_TILT_STOP_DEG:     # leaning, not toppled: back off and let it settle upright
                break
            stalls = stalls + 1 if moved < .004 else 0
            if stalls >= PUSH_STALL_STEPS:
                break
        skills.move(skills.state()['eef'] + [0., 0., .12], down, 1., phase=f'{phase}_push{passes}_clear',
                    ticks=110, required=False)
        idle = idle + 1 if done < .005 else 0
        if idle >= 2:                      # two passes that re-aimed and still moved nothing: it will not go
            break
    _tuck(skills, -1., phase=phase + '_done')
    after, rot = body_pose(env, body)
    lo, hi = body_aabb(env, body)
    target_now = body_pose(env, target)[0]
    spot = O.world_to_frame_xy(env, after[:2])
    clear = sight_clearance(spot, radius, O.world_to_frame_xy(env, target_now[:2]), target_radius, dirs)
    points = O.box_points(target_now[:2], (target_radius, target_radius), top, top + target_height, inflate=.01)
    # the body's own world bounding box (axis aligned, so yaw 0) is what the rays have to miss
    blocked, needed_after = O.occludes(frusta, points, (lo + hi) / 2, (hi - lo) / 2, 0.)
    still = sorted(name for name, need in needed_after.items() if need and blocked[name])
    seen, _ = O.rays_clear(frusta, points, (lo + hi) / 2, (hi - lo) / 2, 0.)
    lean = tilt_deg(rot)
    out = dict(body=body, mode='push', target=target, thin_side_m=float(thin_side),
               heading_frame=[float(d_frame[0]), float(d_frame[1])], distance_m=distance, pushed_m=float(pushed),
               passes=passes, pushes=steps, spot_world=[float(after[0]), float(after[1])],
               planned_spot_frame=plan['spot'], planned_sight_clearance_m=float(plan['sight_clearance_m']),
               sight_clearance_m=float(clear), planned_sight_rays_clear=plan['sight_rays_clear'],
               sight_rays_clear={name: int(seen[name]) for name in needed_after if needed_after[name]},
               sight_rays_total=len(points), cameras_needed=needed, blocked=blocked, still_blocking=still,
               target_moved_m=float(np.linalg.norm(target_now[:2] - target_pos[:2])),
               contact_above_top_m=float(high), push_step_m=float(push_step),
               footprint=[round(v, 4) for v in footprint] if footprint else None,
               bottom_above_top_m=float(lo[2] - top), tilt_deg=lean, drive=drive,
               relaxed=bool(plan.get('relaxed')), ticks=skills.tick - start)
    if lean > ASIDE_TILT_MAX_DEG:
        raise SkillFailure(phase, f'{body} is leaning {lean:.0f} degrees after the push', **out)
    if lo[2] - top > .03 or lo[2] - top < -.02:
        raise SkillFailure(phase, f'{body} is not standing on the work top', **out)
    if still:
        raise SkillFailure(phase, f'{body} still hides {target} from {", ".join(still)} after {pushed * 1000:.0f} mm '
                           f'of the {distance * 1000:.0f} mm push ({clear * 1000:+.0f} mm clear of the sight '
                           'corridor)', **out)
    _no_collateral(env, events_before, phase)
    return out


def move_aside(skills, body, *, spot=None, alternative=0, phase='aside'):
    """Pinch a tall occluder (a box-like RoboCasa object) across its thinnest horizontal extent near its
    top, lift it, and set it down on a free spot of the work top.

    A body whose thinnest side does not fit between the fingers is pushed aside instead (:func:`_push_aside`,
    the heading from :func:`push_aside_plan`) and only refused when no heading is free; everything the pinch
    can hold behaves exactly as before."""
    env = skills.env
    top = skills.top_z
    start = skills.tick
    events_before = len(env.events)
    geoms = collision_geom_names(env, body)
    with skills.segment(f'move the occluder {body} aside', kind='look', skill='move_aside'):
        _tuck(skills, -1., phase=phase + '_clear')
        drive = _stance_for(skills, body, phase=phase)
        pos, rot = body_pose(env, body)
        lo, hi = body_aabb(env, body)
        obj = (getattr(env, 'objects', {}) or {}).get(body)
        if obj is not None:
            size = np.asarray(obj.size, float)
            thin = int(np.argmin(size[:2]))
            axis = rot[:, thin]
            # the Panda opens 80 mm: a box thicker than that across its thinnest side cannot be pinched (measured on
            # layout 9 style 11, cereal_8 at 94 mm: the pads closed on one face, "not both on the object"), so it is
            # pushed out of the line of sight instead of lifted out of it
            if float(size[thin]) > GRIPPER_OPENING_M - .006:
                return _push_aside(skills, body, float(size[thin]), drive=drive, alternative=alternative,
                                   start=start, events_before=events_before, phase=phase)
        else:
            axis = rot[:, 0]
        finger_yaw = math.atan2(axis[1], axis[0])
        centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float(hi[2]) - .035])
        _stand_within_reach(skills, centre, phase=phase + '_stand')
        pos, rot = body_pose(env, body)
        lo, hi = body_aabb(env, body)
        centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float(hi[2]) - .035])
        rotation = skills.grasp_rotation('top', alternative=alternative, finger_yaw=finger_yaw)
        skills.grasp(centre, 'top', rotation=rotation, above=.12, geoms=geoms, lower_ticks=160, phase=phase + '_grasp')
        skills.move(centre + [0., 0., .06], rotation, 1., phase=phase + '_lift', ticks=160)
        rise = float(body_pose(env, body)[0][2] - pos[2])
        if rise < .03:
            raise SkillFailure(phase, f'{body} rose only {rise * 1000:.0f} mm', rise=rise)
        hang = float(skills.state()['eef'][2] - lo[2]) + .01
        _tuck(skills, 1., height=top + hang + .06, phase=phase + '_carry')
        if spot is None:
            spot = set_down_spot(skills, footprint_radius(env, body) + .01, exclude=[body],
                                 prefer_world=pos[:2], headroom=H.headroom_needed('move_aside', object_height=float(hi[2] - lo[2])),
                                 carry_from=skills.state()['eef'][:2], carry_z=float(skills.state()['eef'][2]),
                                 from_base=drive is not None)
        spot = np.asarray(spot, float)[:2]
        _stand_within_reach(skills, [spot[0], spot[1], top], phase=phase + '_place_stand', gripper=1.)
        hand = skills.state()['eef']
        pos, _ = body_pose(env, body)
        lo, hi = body_aabb(env, body)
        target = np.array([*(spot + (hand[:2] - pos[:2])), top + float(hand[2] - lo[2]) + .004])
        skills.place(target, skills.state()['eef_rotation'], above=.08, lower_ticks=200, phase=phase + '_place')
        _tuck(skills, -1., phase=phase + '_done')
    pos, rot = body_pose(env, body)
    lo, hi = body_aabb(env, body)
    record = dict(body=body, spot_world=spot.tolist(), rise_m=rise, bottom_above_top_m=float(lo[2] - top),
                  tilt_deg=tilt_deg(rot), drive=drive, ticks=skills.tick - start)
    if lo[2] - top > .03 or lo[2] - top < -.02:
        raise SkillFailure(phase, f'{body} is not standing on the work top', **record)
    _no_collateral(env, events_before, phase)
    return record


# ---- the oracle entry point --------------------------------------------------------------------------
def resolve_observability(skills, env, *, grasp=None, event=None, retries=1):
    """Undo every hiding mechanism of the instance before an oracle's normal routine.

    Reads ``env.task_spec['observability']`` (what :func:`observe.apply_observability` realised) and
    ``env.spec['hidden']`` and, per entry: an annexed object is fetched onto the work frame
    (:func:`fetch`; ``grasp[name]`` may give ``offset``, ``geoms``, ``finger_yaw`` for the top-down grasp),
    an occluder is moved aside (:func:`move_aside`), a cover is removed with the matching uncover skill.
    Every skill runs under :meth:`Skills.run_attempt` with ``retries``. Ends at the reset stance with the
    hand open and tucked. Returns ``dict(entries, ticks, left_surface, skipped)``; raises ``AttemptFailed``
    when an entry cannot be resolved."""
    record = env.task_spec.get('observability') or {}
    # What was realised, not what was drawn: the stamps and mugs candidate walks may realise another entry
    # than the spec's (a plate on a pad drawn, the large cloche over a vessel built when no plate candidate
    # passed), and the drawn entry then names nothing in the record. Measured on marked_mugs layouts 4/13 and
    # 14/6 (SKILLS-2 certificates): the cloche was never lifted because pad_red was "not realised".
    hidden = list(record.get('realised') or []) or \
        list((getattr(env, 'spec', None) or {}).get('hidden') or record.get('requested') or [])
    start = skills.tick
    events_before = len(env.events)
    if not record or not hidden:
        return dict(entries=[], ticks=0, left_surface=[], skipped=True, problems=list(record.get('problems', [])))
    grasp = grasp or {}
    log = event or (lambda name, payload: None)
    realised = {r['object']: r for r in record.get('realised', [])}
    annex = {a['object']: a for a in record.get('annex', [])}
    occluders = {o['target']: o for o in record.get('occluders', [])}
    covers = {}
    for cover in record.get('covers', []):
        if cover.get('built'):
            covers[cover.get('object') or cover.get('place')] = cover
    entries = []
    for entry in hidden:
        name = entry['object']
        how = (realised.get(name) or {}).get('realised')
        started = skills.tick
        if how == 'annex':
            params = dict(body=name, stance=annex[name]['stance'], **grasp.get(name, {}))
            params['stance'] = dict(params['stance'], top_z=annex[name].get('top_z', skills.top_z))
            result = skills.run_attempt(f'fetch {name}', fetch, (skills,), params, retries=retries)
        elif how == 'occluder':
            result = skills.run_attempt(f'move aside {occluders[name]["name"]}', move_aside, (skills, occluders[name]['name']),
                                        {}, retries=retries)
            # An occluded object was moved to the back of the surface before the occluder was stood (the record's
            # ``relocated``): once uncovered it may still be out of the reset stance's reach, so it is fetched onto
            # the work frame like an annexed one (measured: stamps layout 4 style 11, the stamp at frame y .86).
            home_xy, home_yaw = skills.reset_base['xy'], float(skills.reset_base['yaw'])
            if not within_top_down_reach(home_xy, home_yaw, body_pose(env, name)[0]):
                params = dict(body=name, stance=None, **grasp.get(name, {}))
                result = dict(move_aside=result,
                              fetch=skills.run_attempt(f'fetch {name}', fetch, (skills,), params, retries=retries))
        elif how == 'cover':
            cover = covers[name]
            kind, body = cover['kind'], cover['name']
            if kind in ('cloche_small', 'cloche_large'):
                result = skills.run_attempt(f'lift {body}', lift_cloche, (skills, body), {}, retries=retries,
                                            jitter=lambda i, rng: dict(alternative=i))
            elif kind == 'bowl':
                result = skills.run_attempt(f'slide and lift {body}', slide_lift_bowl, (skills, body), {}, retries=retries,
                                            jitter=lambda i, rng: dict(fingers_up=bool(i % 2 == 0)))
            elif kind == 'wide_cup':
                result = skills.run_attempt(f'pour {body}', pour_cup, (skills, body), {}, retries=retries,
                                            jitter=lambda i, rng: dict(alternative=i))
            elif cover.get('mode') == 'lift':
                result = skills.run_attempt(f'lift {body} off the pan', lift_plate_from_pan, (skills, body), {},
                                            retries=retries,
                                            jitter=lambda i, rng: PLATE_PITCH_TRIES[i % len(PLATE_PITCH_TRIES)])
            else:
                result = skills.run_attempt(f'push {body} aside', push_cover, (skills, body), {}, retries=retries)
        else:
            result = dict(skipped=True, reason=f'entry not realised ({how})')
        row = dict(object=name, mode=entry['mode'], cover=entry.get('cover'), realised=how, ticks=skills.tick - started,
                   result=result)
        entries.append(row)
        log('observability_resolved', row)
    home = dict(xy=list(skills.reset_base['xy']), yaw=float(skills.reset_base['yaw']), edge='front', standoff=O.STANDOFF_M)
    base_xy, base_yaw = skills.base_pose()
    if np.linalg.norm(base_xy - np.asarray(home['xy'])) > .05 or abs(_wrap(home['yaw'] - base_yaw)) > .06:
        drive_to_stance(skills, home, phase='resolve_home')
    _tuck(skills, -1., phase='resolve_done')
    return dict(entries=entries, ticks=skills.tick - start, left_surface=_events_since(env, events_before), skipped=False,
                problems=list(record.get('problems', [])))
