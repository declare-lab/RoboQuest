"""Pure search-room scoring and diagnostics. No simulator imports.

Terminal predicate for a *set* of targets: every target rests fully inside the
tray square (a task-frame square on the work surface) with a real upward support
contact on that surface, upright, released and still; every other task object is
released and supported somewhere; nothing else overlaps the tray square. The
tray square is tested in its own frame, so rotated work surfaces score the same
as axis-aligned ones. Search diagnostics (which compartments were opened, in
what order) are evaluator-only.

Extended from the room-search worktree (`room_search_rules.py`).
"""
from copy import deepcopy
import math

from roboquest.harness.contract import _overlap, convex_hull
from roboquest.base.scene_contracts import all_resting, support_contact, upright

OPEN_FRACTION = .15   # a door/drawer counts as opened once past this fraction of its travel
IN_TRAY_MARGIN_M = .004
SURFACE_TOLERANCE_M = .007
LEVEL_BELOW_M = .01    # an object's lowest point may sink this far into the tray surface (contact penetration)
LEVEL_ABOVE_M = .06    # or float this far above it (a tilted or stacked object is not resting on the tray)


def tray_local(tray, xy):
    """World xy to tray-frame coordinates (centre at the origin, x along the work-surface front)."""
    c, s = math.cos(tray['yaw']), math.sin(tray['yaw'])
    dx, dy = float(xy[0]) - tray['centre_xy'][0], float(xy[1]) - tray['centre_xy'][1]
    return (c * dx + s * dy, -s * dx + c * dy)


def tray_square(tray, margin=0.):
    h = float(tray['half']) - margin
    return [(-h, -h), (h, -h), (h, h), (-h, h)]


def at_tray_level(state, tray):
    """The object's lowest collision point is at the tray surface (not in a drawer below or held high above)."""
    try:
        bottom = float(state['collision_bounds_xyz'][0][2])
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return tray['top_z'] - LEVEL_BELOW_M <= bottom <= tray['top_z'] + LEVEL_ABOVE_M


def footprint_in_tray(state, tray):
    if not at_tray_level(state, tray):
        return False
    try:
        hull = convex_hull(state['footprint_xy'])
    except (KeyError, TypeError, ValueError):
        return False
    h = float(tray['half']) - IN_TRAY_MARGIN_M
    return all(abs(lx) <= h and abs(ly) <= h for lx, ly in (tray_local(tray, p) for p in hull))


def supported_on_tray(state, tray):
    """A strict upward support contact at the tray's surface height, inside the square."""
    for contact in state.get('support_contacts', []):
        if not support_contact(contact, height_z=tray['top_z']):
            continue
        lx, ly = tray_local(tray, contact['position'][:2])
        if abs(lx) <= tray['half'] and abs(ly) <= tray['half']:
            return True
    return False


def overlaps_tray(state, tray):
    """A non-target intrudes on the tray only when it rests at the tray's level (a hidden object in a
    drawer under the same counter shares the xy footprint but is not on the tray)."""
    if not at_tray_level(state, tray):
        return False
    try:
        hull = [tray_local(tray, p) for p in convex_hull(state['footprint_xy'])]
    except (KeyError, TypeError, ValueError):
        return True
    return _overlap(hull, tray_square(tray))


def terminal_score(spec, snapshot):
    """Current-state score without any stability latch; the Submit press freezes it.

    ``spec`` needs ``objects`` (all task object ids), ``target_ids`` and ``tray``
    (``centre_xy``, ``yaw``, ``half``, ``top_z``).
    """
    tray = spec['tray']
    targets = list(spec['target_ids'])
    objects = snapshot.get('objects', {})
    errors = all_resting(snapshot)
    if set(objects) != set(spec['objects']):
        errors.append('object_set_mismatch')
    per_target = {}
    for name in targets:
        state = objects.get(name)
        inside = state is not None and footprint_in_tray(state, tray)
        supported = state is not None and supported_on_tray(state, tray)
        is_upright = state is not None and upright(state)
        per_target[name] = dict(in_tray=bool(inside), supported=bool(supported), upright=bool(is_upright))
        if not inside:
            errors.append(f'{name}:outside_tray')
        if not supported:
            errors.append(f'{name}:not_on_tray_surface')
        if not is_upright:
            errors.append(f'{name}:not_upright')
    contents = sorted(n for n, s in objects.items() if footprint_in_tray(s, tray))
    for name, state in objects.items():
        if name not in targets and overlaps_tray(state, tray):
            errors.append(f'{name}:unwanted_tray_overlap')
    done = [n for n in targets if all(per_target[n].values())]
    return {
        'success': not errors,
        'score': (len(done) / len(targets)) if targets else 0.,
        'errors': list(errors),
        'targets_on_tray': done,
        'tray_contents': contents,
        'per_target': per_target,
    }


def inside_interior(local_xyz, region, slide_local=(0., 0., 0.), margin=.03):
    """Is a point (fixture-local) inside a compartment's interior region? ``slide_local`` shifts the region
    with an opened drawer (its interior travels with the slide joint); doors leave it in place."""
    ox, oy, oz = region['offset_local']
    sx, sy = region['size_xy']
    x, y, z = (float(v) - float(s) for v, s in zip(local_xyz, slide_local))
    return (abs(x - ox) <= sx / 2 + margin and abs(y - oy) <= sy / 2 + margin
            and oz - margin <= z <= oz + float(region['height']) + margin)


def under_cover(position, cover_position, radius, height, margin=.02):
    """Is an object's centre inside the cylinder a cover encloses (cover origin at its rim plane)?"""
    dx, dy = float(position[0]) - float(cover_position[0]), float(position[1]) - float(cover_position[1])
    z0 = float(cover_position[2])
    return math.hypot(dx, dy) <= float(radius) + margin and z0 - margin <= float(position[2]) <= z0 + float(height) + margin


PROGRESS_CAP = .95   # progress reaches 1 only through the success predicate (base contract C1)


def target_progress(done, out, success):
    """Spec 1.4: per target 0.5 once out of its hiding place, 1 on the tray; mean over targets; 1 exactly on
    success (a non-success state with every target on the tray, say a distractor on it too, is capped)."""
    if success:
        return 1.
    values = [1. if d else (.5 if o else 0.) for d, o in zip(done, out)]
    return min(PROGRESS_CAP, sum(values) / len(values)) if values else 0.


def compartment_open_fraction(joint_value, joint_range):
    """Normalised opening in [0, 1] for a hinge or slide joint whose closed pose is 0."""
    low, high = joint_range
    extent = max(abs(low), abs(high))
    if extent <= 0:
        return 0.
    return min(1., abs(float(joint_value)) / extent)


def compartment_states(compartments, joint_values):
    """Per-compartment opening fraction from the current joint values."""
    states = {}
    for cid, comp in compartments.items():
        fractions = [compartment_open_fraction(joint_values[j], comp['joint_ranges'][j])
                     for j in comp['joints'] if j in joint_values]
        states[cid] = {'open_fraction': max(fractions) if fractions else 0.,
                       'opened': bool(fractions) and max(fractions) >= OPEN_FRACTION}
    return states


class SearchDiagnostics:
    """Evaluator-side record of which compartments were ever opened and in what order."""

    def __init__(self, compartments, target_compartments):
        self.compartments = deepcopy(compartments)
        self.target_compartments = sorted(c for c in target_compartments if c)
        self.opened_order = []
        self.open_events = []
        self._currently_open = set()

    def update(self, step, joint_values):
        states = compartment_states(self.compartments, joint_values)
        for cid, state in states.items():
            if state['opened'] and cid not in self._currently_open:
                self._currently_open.add(cid)
                self.open_events.append({'step': int(step), 'compartment': cid, 'event': 'open'})
                if cid not in self.opened_order:
                    self.opened_order.append(cid)
            elif not state['opened'] and cid in self._currently_open:
                self._currently_open.discard(cid)
                self.open_events.append({'step': int(step), 'compartment': cid, 'event': 'close'})
        return states

    def summary(self):
        opened = list(self.opened_order)
        found = [c for c in opened if c in self.target_compartments]
        return {
            'opened_order': opened,
            'opened_count': len(opened),
            'targets_opened': found,
            'all_target_compartments_opened': all(c in opened for c in self.target_compartments),
            'openings_before_first_target': next((i for i, c in enumerate(opened) if c in self.target_compartments), len(opened)),
            'reopen_count': max(0, sum(1 for e in self.open_events if e['event'] == 'open') - len(opened)),
            'currently_open': sorted(self._currently_open),
        }
