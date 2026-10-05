"""Pure terminal predicates and positive public projection for the vision suite."""
from copy import deepcopy
import hashlib
import json
import math

from roboquest.harness.contract import CAMERAS, PROPRIO_FIELDS, policy_observation as robot_observation, _inside, _overlap, _upright, convex_hull
from roboquest.harness.mobile_native_contract import PUBLIC_CONTROL_VERSION

STABLE_TICKS = 10


def finite_vector(value, length):
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or not all(type(x) in (int, float) and math.isfinite(x) for x in value)):
        raise ValueError('finite numeric vector required')
    return value


def footprint(state):
    # Canonicalize before containment/overlap: repeated or collinear points
    # cannot represent a physically supported object's planar footprint.
    return convex_hull(state['footprint_xy'])


def support_contact(contact, *, support_id=None, height_z=None, bounds_xy=None):
    """One strict upward support witness for both rest and destination checks."""
    try:
        position = finite_vector(contact['position'], 3)
        normal = finite_vector(contact['normal_on_object'], 3)
        force = finite_vector([contact['normal_force_n']], 1)[0]
        if (not isinstance(contact['support_id'], str) or not contact['support_id']
                or force <= 1e-6 or abs(sum(x*x for x in normal)-1) >= .002
                or normal[2] < .5):
            return False
        if support_id is not None and contact['support_id'] != support_id:
            return False
        if height_z is not None and abs(position[2]-height_z) > .007:
            return False
        if bounds_xy is not None and not _inside([position[:2]], bounds_xy):
            return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def policy_observation(rgb, proprio, instruction):
    packet = robot_observation(rgb, proprio)
    packet['instruction'] = instruction
    packet['contract_version'] = PUBLIC_CONTROL_VERSION
    return packet


def in_zone(state, bounds):
    try:
        points = footprint(state)
        finite_vector(bounds, 4)
        return bounds[0] < bounds[1] and bounds[2] < bounds[3] and _inside(points, bounds)
    except (KeyError, TypeError, ValueError):
        return False


def upright(state):
    try:
        return _upright(state['quaternion_xyzw'], [0., 0., 1.])
    except (KeyError, TypeError, ValueError):
        return False


def all_resting(snapshot):
    """Return errors for moving, held, touching-robot or unsupported objects.

Actual upward contacts are required. Fixture identity and permitted support
surfaces can be further restricted by each task's terminal predicate.
"""
    errors = []
    objects = snapshot.get('objects', {})
    if not objects:
        return ['missing_objects']
    for name, state in objects.items():
        try:
            finite_vector(state['position'], 3)
            quat = finite_vector(state['quaternion_xyzw'], 4)
            if abs(sum(x*x for x in quat)-1) > .002:
                errors.append(f'{name}:invalid_quaternion')
            footprint(state)
            for field, limit in [('linear_velocity', .03), ('angular_velocity', .2)]:
                vector = finite_vector(state[field], 3)
                if math.sqrt(sum(x*x for x in vector)) > limit:
                    errors.append(f'{name}:{field}')
            if state['grasped'] is not False or state['robot_contact'] is not False:
                errors.append(f'{name}:not_released')
            if state.get('fixed', False) is True:
                continue
            if not any(support_contact(c) for c in state['support_contacts']):
                errors.append(f'{name}:unsupported')
        except (KeyError, TypeError, ValueError):
            errors.append(f'{name}:invalid_state')
    return errors


def evaluate_pick_place(spec, snapshot):
    """Broad full-footprint placement; no required action sequence or exact pose.

spec: target_ids, destination (bounds or zone name), zones, objects;
optional upright_ids and exclusive (default True).
"""
    try:
        if not isinstance(spec['objects'], dict) or not spec['objects']:
            raise ValueError('objects required')
        targets = spec['target_ids']
        if (not isinstance(targets, list) or not targets
                or not all(isinstance(n, str) and n in spec['objects'] for n in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError('distinct actual targets required')
        destination = spec['destination']
        zone = spec['zones'][destination] if isinstance(destination, str) else {'bounds_xy': destination}
        bounds = finite_vector(zone['bounds_xy'], 4)
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError('ordered destination required')
        if zone.get('height_z') is not None:
            finite_vector([zone['height_z']], 1)
    except (KeyError, TypeError, ValueError):
        return {'state_valid': False, 'errors': ['invalid_task_spec'], 'destination_contents': []}
    errors = all_resting(snapshot)
    states = snapshot.get('objects', {})
    if set(states) != set(spec['objects']):
        errors.append('object_set_mismatch')
    target_ids = set(spec['target_ids'])
    contents = {name for name, state in states.items() if in_zone(state, bounds)}
    for name in target_ids:
        if name not in contents:
            errors.append(f'{name}:outside_destination')
        if isinstance(destination, str) and name in states:
            height = zone.get('height_z')
            if height is not None:
                support = any(support_contact(c, support_id=zone.get('support_id'),
                    height_z=height, bounds_xy=bounds) for c in states[name].get('support_contacts', []))
                if not support:
                    errors.append(f'{name}:not_on_destination_surface')
    if spec.get('exclusive', True):
        for name, state in states.items():
            if name not in target_ids:
                try:
                    x0, x1, y0, y1 = bounds
                    zone_hull = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                    if _overlap(footprint(state), zone_hull):
                        errors.append(f'{name}:unwanted_destination_overlap')
                except (KeyError, TypeError, ValueError):
                    errors.append(f'{name}:invalid_footprint')
    for name in spec.get('upright_ids', []):
        if name not in states or not upright(states[name]):
            errors.append(f'{name}:not_upright')
    return {'state_valid': not errors, 'errors': errors, 'destination_contents': sorted(contents)}


class StableEvaluator:
    """Current-state score with consecutive charged-tick stability, never a latch."""
    def __init__(self, spec, evaluate):
        self.spec = deepcopy(spec)
        self.evaluate = evaluate
        self.stable_ticks = 0
        self.last_step = None
        self.last_digest = None

    @staticmethod
    def _digest(snapshot):
        return hashlib.sha256(json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def update(self, snapshot):
        step = snapshot['step']
        expected = 1 if self.last_step is None else self.last_step + 1
        if type(step) is not int or step != expected:
            raise ValueError('expected exactly one new charged physics tick')
        result = self.evaluate(self.spec, snapshot)
        self.stable_ticks = self.stable_ticks + 1 if result['state_valid'] else 0
        self.last_step = step
        self.last_digest = self._digest(snapshot)
        return {**result, 'stable_steps': self.stable_ticks,
                'success': bool(result['state_valid'] and self.stable_ticks >= STABLE_TICKS)}

    def score(self, snapshot):
        result = self.evaluate(self.spec, snapshot)
        same_step = self.last_step == snapshot['step'] and self.last_digest == self._digest(snapshot)
        return {**result, 'stable_steps': self.stable_ticks if result['state_valid'] and same_step else 0,
                'success': bool(result['state_valid'] and same_step and self.stable_ticks >= STABLE_TICKS)}
