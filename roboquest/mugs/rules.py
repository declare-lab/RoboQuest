"""Route-independent original-ball and placement score for the marked vessels.

The study-room rule (``active_bench.inspect_cups_rules``) generalised twice:

* the pads live in the task frame, so a vessel's interior centre is transformed
  into frame coordinates (and then into pad-local xy) before the in-pad test;
* the interior is a **truncated cone** - ``floor_radius`` at the interior floor
  widening to ``rim_radius`` at the rim - which is a cylinder for the mug and the
  real shape for the bowl, so one containment test covers both vessel types.

Everything else - uprightness, release, stillness, original-pair identity - is
world frame and carries over as written. ``on_pad`` records which pad actually
supports a vessel, so a vessel parked on the *wrong* pad is visible to
``task_progress`` (spec 1.4) rather than merely "not correct".

Placement credit. A vessel's footprint is the disc of radius ``pad_fit_radius``
(3.95 cm) around its interior centre; f is the area fraction of that disc lying on
its own pad rectangle (pad half size + 2 mm), computed exactly. The vessel earns
credit 1 for f >= 3/4, 0.5 for 1/2 <= f < 3/4 and 0 below, provided it stands
upright, released and stable (its own balls at rest too), is supported from below
by its own pad and does not stand on a wrong pad. ``correct_pad`` is full credit;
a vessel succeeds with full credit and its own two balls inside.
"""
import math

import numpy as np

from roboquest.mugs.geometry import BALL_RADIUS, interior_radius_at, yaw_matrix


def contained_ball(vessel, mug, ball):
    """A ball is in the vessel when it is inside the interior cone, in vessel-local coordinates.

    ``vessel`` is the geometry record (interior floor, rim and the two radii);
    ``mug``/``ball`` are the world states. The test is unambiguous for both types:
    a ball on the counter beside a bowl is below the interior floor, and a ball
    perched on the rim is above ``rim_z - BALL_RADIUS``.
    """
    local = np.asarray(mug['rotation']).T @ (np.asarray(ball['position']) - np.asarray(mug['position']))
    floor, rim = float(vessel['interior_floor_z']), float(vessel['rim_z'])
    if not (floor + BALL_RADIUS - .004 <= local[2] <= rim - BALL_RADIUS + .004):
        return False
    centre = np.asarray(vessel['interior_center'], float)
    radius = interior_radius_at(vessel, float(local[2]))
    return bool(np.linalg.norm(local[:2] - centre) + BALL_RADIUS <= radius + .0025)


PAD_MARGIN = .002          # tolerance on the pad half size
FULL_CREDIT_FRACTION = .75
HALF_CREDIT_FRACTION = .5
_EPS = 1e-9


def _arc_primitive(t, r):
    """Primitive of sqrt(r^2 - t^2) on [-r, r]. The angle is taken from the same s as the first term (atan2, not
    asin(t / r)), so the two terms cancel to rounding near t = +-r."""
    t = max(-r, min(r, t))
    s = math.sqrt(max(0., (r - t) * (r + t)))
    return .5 * (t * s + r * r * math.atan2(t, s))


def disc_rect_area(cx, cy, r, x0, x1, y0, y1):
    """Exact area of the disc (centre (cx, cy), radius r) inside the rectangle [x0, x1] x [y0, y1].

    In disc-centred coordinates the area is the integral over t of min(d, s(t)) - max(c, -s(t)), s = sqrt(r^2-t^2),
    wherever positive. Between the breakpoints where s(t) equals |c| or |d| each bound is a single branch, so every
    piece integrates in closed form."""
    if r <= 0:
        return 0.
    a, b = max(x0 - cx, -r), min(x1 - cx, r)
    c, d = y0 - cy, y1 - cy
    if a >= b or c >= d:
        return 0.
    cuts = {a, b}
    for v in (c, d):
        if abs(v) < r:
            w = math.sqrt(r * r - v * v)
            cuts.update(p for p in (-w, w) if a < p < b)
    pts = sorted(cuts)
    area = 0.
    for t1, t2 in zip(pts, pts[1:]):
        m = (t1 + t2) / 2
        s = math.sqrt(max(0., r * r - m * m))
        if min(d, s) <= max(c, -s):
            continue
        arc = _arc_primitive(t2, r) - _arc_primitive(t1, r)
        upper = arc if s < d else d * (t2 - t1)
        lower = -arc if -s > c else c * (t2 - t1)
        area += upper - lower
    return max(0., area)


def pad_area_fraction(center_xy, half_size_xy, radius, margin=PAD_MARGIN):
    """Fraction of the footprint disc (``radius`` around ``center_xy``, pad-local) on the pad rectangle
    (``half_size_xy`` + ``margin`` on each side)."""
    hx, hy = float(half_size_xy[0]) + margin, float(half_size_xy[1]) + margin
    area = disc_rect_area(float(center_xy[0]), float(center_xy[1]), float(radius), -hx, hx, -hy, hy)
    return min(1., area / (math.pi * float(radius) ** 2))


def fraction_credit(fraction):
    """1 from FULL_CREDIT_FRACTION up, 0.5 from HALF_CREDIT_FRACTION up, else 0 (inclusive, to within 1e-9)."""
    if fraction >= FULL_CREDIT_FRACTION - _EPS:
        return 1.
    if fraction >= HALF_CREDIT_FRACTION - _EPS:
        return .5
    return 0.


def placement_credit(fraction, upright, released, stable, supported, wrong_pad):
    """A vessel's placement credit: ``fraction_credit`` of its footprint fraction on its own pad, when it stands
    upright, released and stable, supported by its own pad and not on a wrong pad; else 0."""
    if upright and released and stable and supported and not wrong_pad:
        return fraction_credit(fraction)
    return 0.


def world_to_frame(geometry, position):
    origin = np.asarray(geometry['frame_origin_world'], float)
    return yaw_matrix(-geometry['frame_yaw']) @ (np.asarray(position, float) - origin)


def pad_local_xy(geometry, pad, position, rotation, centre_xy):
    """The vessel's interior centre in a pad's own xy frame."""
    offset = np.asarray(rotation) @ np.array([centre_xy[0], centre_xy[1], 0.])
    centre = world_to_frame(geometry, np.asarray(position) + offset)
    return (yaw_matrix(-float(pad.get('yaw_rad', 0.)))[:2, :2]
            @ (centre[:2] - np.asarray(pad['center_xy'], float))), centre


def classify_placements(states, geometry):
    rows = {}
    for name, info in geometry['vessels'].items():
        vessel = states[name]
        pad = geometry['pads'][info['label']]
        position = np.asarray(vessel['position'])
        rotation = np.asarray(vessel['rotation'])
        centre_xy = np.asarray(info['interior_center'], float)
        fit = float(info.get('pad_fit_radius', geometry.get('pad_fit_radius', .0395)))
        present = [b for b in geometry['balls'] if contained_ball(info, vessel, states[b])]
        # Pads carry their own yaw in the task frame, so the in-pad test runs in pad-local xy.
        pad_local, centre = pad_local_xy(geometry, pad, position, rotation, centre_xy)
        fraction = pad_area_fraction(pad_local, pad['half_size_xy'], fit)
        upright = bool(rotation[2, 2] > .97)
        released = not vessel.get('robot_contact', False) and not vessel.get('grasped', False)
        stable = vessel.get('linear_speed', 1.) < .025 and vessel.get('angular_speed', 1.) < .15
        supported = info['label'] in vessel.get('supported_pads', [])
        original = sorted(present) == sorted(info['original_balls'])
        balls_stable = all(states[b].get('linear_speed', 1.) < .03 for b in info['original_balls'])
        # Which pad is this vessel actually standing on? A vessel that supports itself on a
        # pad whose colour is not its label is on a *wrong* pad and costs progress.
        on_pad = None
        for colour in vessel.get('supported_pads', []):
            other = geometry['pads'][colour]
            other_local, _ = pad_local_xy(geometry, other, position, rotation, centre_xy)
            if np.all(np.abs(other_local) + fit <= np.asarray(other['half_size_xy']) + PAD_MARGIN):
                on_pad = colour
                break
        wrong_pad = bool(on_pad is not None and on_pad != info['label'])
        credit = placement_credit(fraction, upright, released, stable and balls_stable, supported, wrong_pad)
        rows[name] = dict(correct_pad=credit == 1., placement_credit=credit, pad_area_fraction=round(fraction, 4),
                          supported=bool(supported), upright=upright, released=bool(released),
                          stable=bool(stable and balls_stable), contained_balls=present,
                          original_pair_retained=bool(original), on_pad=on_pad,
                          wrong_pad=wrong_pad,
                          expected_ball_colors=[geometry['balls'][b].get('color') for b in info['original_balls']],
                          contained_ball_colors=[geometry['balls'][b].get('color') for b in present],
                          pad_center_frame_xy=list(pad['center_xy']),
                          pad_yaw_rad=float(pad.get('yaw_rad', 0.)),
                          vessel_center_frame_xy=[float(centre[0]), float(centre[1])],
                          vessel_center_pad_local_xy=[float(pad_local[0]), float(pad_local[1])],
                          success=bool(credit == 1. and original))
    released = all(not s.get('robot_contact', False) and not s.get('grasped', False) for s in states.values())
    return dict(success=bool(released and all(row['success'] for row in rows.values())), vessels=rows,
                vessel=geometry.get('vessel', 'mug'),
                all_objects_released=released, termination_contract='first_physical_submit')


def progress(score, count):
    """Spec 1.4: (vessels upright on their own pad with their own two balls
    - vessels on a wrong pad) / vessel count, clipped to [0, 1]."""
    rows = score['vessels']
    good = sum(1 for row in rows.values()
               if row['correct_pad'] and row['upright'] and row['original_pair_retained'])
    bad = sum(1 for row in rows.values() if row['wrong_pad'])
    return float(min(max((good - bad) / max(int(count), 1), 0.), 1.))


def spilled_balls(states, geometry):
    """Balls that are not inside the vessel they started in (spec 1.5 ``balls_spilled``)."""
    out = []
    for name, info in geometry['vessels'].items():
        vessel = states[name]
        for ball in info['original_balls']:
            if not contained_ball(info, vessel, states[ball]):
                out.append(ball)
    return out
