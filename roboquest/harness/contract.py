"""Evaluator-only mesh/contact contract. No simulator truth belongs in policy input.

Footprints are convex hulls of world-transformed native collision geometry, not
full-volume bounding boxes. Conservative hulls may reject near-edge placements;
use generous zones. Contacts must be actual upward support witnesses.
"""
from copy import deepcopy
import math

CONTRACT_VERSION = 'table-setting-physical-v1'
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
PROPRIO_FIELDS = {
    'eef_position_world_m': 3, 'eef_quaternion_xyzw': 4,
    'base_position_world_m': 3, 'base_quaternion_xyzw': 4,
    'joint_positions': None, 'joint_velocities': None,
}
PUBLIC_GOAL = (
    'Finish setting the two marked places: put exactly one plate and one drinking cup at '
    'each place, without overlapping them, and put the clutter in the marked storage area. '
    'Leave unused objects in the marked storage area. Keep cups upright and '
    'plates right-side-up and flat. End with everything released and settled. '
    'Any order and any fitting arrangement are allowed.'
)
MAX_TILT_DEG = 15.0
MAX_LINEAR_SPEED = .03
MAX_ANGULAR_SPEED = .2
STABLE_STEPS = 10
CONTROL_HZ = 20
BOUND_TOLERANCE_M = .001
CONTACT_HEIGHT_TOLERANCE_M = .005


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or (length is not None and len(value) != length):
        raise ValueError('wrong vector shape')
    if not value or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('non-finite or non-numeric vector')
    return value


def policy_observation(rgb, proprio):
    """Positive allowlist projection; caller supplies rendered RGB and robot data.

Image tensors pass through unchanged. Image provenance/absence of overlays is a
renderer integration responsibility, not something this pure function certifies.
No task spec, seed, case ID, scorer output or object state is accepted as a field.
"""
    if set(rgb) != set(CAMERAS):
        raise ValueError('fixed camera set required')
    clean = {}
    for key, length in PROPRIO_FIELDS.items():
        clean[key] = list(_vector(proprio[key], length))
    return {'contract_version': CONTRACT_VERSION, 'instruction': PUBLIC_GOAL,
            'rgb': {name: rgb[name] for name in CAMERAS}, 'proprio': clean}


def convex_hull(points):
    """Conservative planar footprint, vertices in counterclockwise order."""
    points = sorted(set(tuple(_vector(p, 2)) for p in points))
    if len(points) < 3:
        raise ValueError('footprint needs three non-collinear points')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1]+upper[:-1]
    if len(hull) < 3:
        raise ValueError('degenerate footprint')
    return hull


def _inside(hull, bounds):
    xmin, xmax, ymin, ymax = bounds
    t = BOUND_TOLERANCE_M
    return all(xmin-t <= x <= xmax+t and ymin-t <= y <= ymax+t for x, y in hull)


def _overlap(a, b):
    # Positive-area convex overlap via separating axes. Edge touching is allowed.
    for poly in (a, b):
        for p, q in zip(poly, poly[1:]+poly[:1]):
            axis = (-(q[1]-p[1]), q[0]-p[0])
            pa = [x*axis[0]+y*axis[1] for x, y in a]
            pb = [x*axis[0]+y*axis[1] for x, y in b]
            if min(max(pa), max(pb)) <= max(min(pa), min(pb)) + 1e-12:
                return False
    return True


def _upright(quat, local_axis):
    q = _vector(quat, 4)
    norm = math.sqrt(sum(x*x for x in q))
    if not .999 <= norm <= 1.001:
        raise ValueError('quaternion must be normalized xyzw')
    x, y, z, w = [v/norm for v in q]
    axis = _vector(local_axis, 3)
    n = math.sqrt(sum(v*v for v in axis))
    if not .999 <= n <= 1.001:
        raise ValueError('up axis must be normalized')
    vertical = (2*(x*z-w*y)*axis[0] + 2*(y*z+w*x)*axis[1]
                + (1-2*(x*x+y*y))*axis[2]) / n
    return vertical >= math.cos(math.radians(MAX_TILT_DEG))


def validate_spec(spec):
    if len(spec['places']) != 2:
        raise ValueError('exactly two places required')
    supports = spec['supports']
    for support in supports.values():
        b = _vector(support['bounds_xy'], 4)
        if b[0] >= b[1] or b[2] >= b[3] or not math.isfinite(support['height_z']):
            raise ValueError('invalid support geometry')
    if not spec['storage_zones']:
        raise ValueError('explicit spare-stock storage zones required')
    for zone in [*spec['places'].values(), spec['stow'], *spec['storage_zones'].values()]:
        b = _vector(zone['bounds_xy'], 4)
        if b[0] >= b[1] or b[2] >= b[3] or zone['support_id'] not in supports:
            raise ValueError('invalid zone')
        if not _inside([(b[0],b[2]),(b[1],b[3])], supports[zone['support_id']]['bounds_xy']):
            raise ValueError('zone outside support')
    for obj in spec['objects'].values():
        if obj['kind'] not in ('plate','cup','clutter') or not obj['valid_supports']:
            raise ValueError('invalid object role/supports')
        if not set(obj['valid_supports']) <= set(supports):
            raise ValueError('unknown permitted support')
    if sum(o['kind']=='plate' for o in spec['objects'].values()) < 2 or sum(o['kind']=='cup' for o in spec['objects'].values()) < 2:
        raise ValueError('insufficient tableware in spec')


def evaluate_terminal(spec, snapshot):
    """Evaluate *current* state only. Stability history is handled separately.

contacts: support_id, position(world), normal_force_n, normal_on_object(world unit
normal pointing from support into object). Missing/invalid evidence fails closed.
Every task object, including unused stock, must have a valid support and be still.
"""
    validate_spec(spec)
    errors, footprints, supported = [], {}, {}
    states = snapshot.get('objects', {})
    if set(states) != set(spec['objects']):
        return {'state_valid': False, 'errors': ['object_set_mismatch'], 'place_contents': {}}
    for name, role in spec['objects'].items():
        state = states[name]
        try:
            _vector(state['position'], 3)
            hull = convex_hull(state['footprint_xy'])
            footprints[name] = hull
            supported[name] = set()
            for contact in state['support_contacts']:
                sid = contact['support_id']
                if sid not in role['valid_supports']:
                    continue
                support = spec['supports'][sid]
                cp = _vector(contact['position'], 3)
                normal = _vector(contact['normal_on_object'], 3)
                force = contact['normal_force_n']
                if (not isinstance(force, bool) and isinstance(force, (int, float)) and math.isfinite(force) and force > 1e-6
                    and abs(sum(x*x for x in normal)-1) <= .002
                    and normal[2] >= .5
                    and abs(cp[2]-support['height_z']) <= CONTACT_HEIGHT_TOLERANCE_M
                    and _inside([cp[:2]], support['bounds_xy'])
                    and _inside(hull, support['bounds_xy'])):
                    supported[name].add(sid)
            if not supported[name]: errors.append(name+':unsupported')
            if state['grasped'] is not False or state['robot_contact'] is not False:
                errors.append(name+':not_released')
            for key, limit in [('linear_velocity',MAX_LINEAR_SPEED),('angular_velocity',MAX_ANGULAR_SPEED)]:
                if math.sqrt(sum(v*v for v in _vector(state[key], 3))) >= limit:
                    errors.append(name+':'+key)
            if role['kind'] in ('cup','plate') and not _upright(state['quaternion_xyzw'], role.get('up_axis_local',[0,0,1])):
                errors.append(name+':orientation')
            if role['kind']=='clutter':
                zone = spec['stow']
                if zone['support_id'] not in supported[name] or not _inside(hull,zone['bounds_xy']):
                    errors.append(name+':not_stowed')
        except (KeyError, ValueError, TypeError, OverflowError):
            errors.append(name+':invalid_snapshot')
    contents = {}
    for place, zone in spec['places'].items():
        names = [n for n in footprints if zone['support_id'] in supported.get(n,set()) and _inside(footprints[n],zone['bounds_xy'])]
        contents[place] = names
        b = zone['bounds_xy']
        zone_polygon = [(b[0],b[2]),(b[1],b[2]),(b[1],b[3]),(b[0],b[3])]
        for name, hull in footprints.items():
            if zone['support_id'] in supported.get(name, set()) and name not in names and _overlap(hull, zone_polygon):
                errors.append(place+':partial_or_unsupported_intrusion:'+name)
        for kind in ('plate','cup'):
            if sum(spec['objects'][n]['kind']==kind for n in names) != 1:
                errors.append(place+':'+kind+'_count')
        if any(spec['objects'][n]['kind']=='clutter' for n in names): errors.append(place+':clutter')
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if _overlap(footprints[a],footprints[b]): errors.append(place+':overlap')
    assigned = [n for names in contents.values() for n in names]
    if len(assigned) != len(set(assigned)): errors.append('ambiguous_place_membership')
    for name, role in spec['objects'].items():
        if role['kind'] in ('cup', 'plate') and name not in assigned:
            stored = any(zone['support_id'] in supported.get(name, set()) and
                         name in footprints and _inside(footprints[name], zone['bounds_xy'])
                         for zone in spec['storage_zones'].values())
            if not stored: errors.append(name+':unused_not_stored')
    return {'state_valid':not errors, 'errors':errors, 'place_contents':contents}


class ReleasedStableEvaluator:
    """Observe each distinct 20Hz step; score the declared current terminal state."""
    def __init__(self, spec):
        validate_spec(spec)
        self.spec = deepcopy(spec)
        self.last = None
        self.stable_steps = 0

    def update(self, snapshot):
        step = snapshot.get('step')
        if isinstance(step,bool) or not isinstance(step,int) or step < 0:
            raise ValueError('nonnegative integer step required')
        if self.last is not None and step <= self.last['step']:
            raise ValueError('steps must advance; repeated scoring cannot accrue stability')
        result = evaluate_terminal(self.spec,snapshot)
        consecutive = self.last is not None and step == self.last['step']+1
        self.stable_steps = (self.stable_steps+1 if consecutive else 1) if result['state_valid'] else 0
        self.last = deepcopy(snapshot)
        return self.score(snapshot)

    def score(self, snapshot):
        result = evaluate_terminal(self.spec,snapshot)
        exact_current = self.last == snapshot
        stable = self.stable_steps if exact_current and result['state_valid'] else 0
        return {**result, 'success':result['state_valid'] and stable >= STABLE_STEPS,
                'stable_steps':stable, 'contract_version':CONTRACT_VERSION}


def crossed_design():
    """Evaluator-side plan, not a scene generator or proven hiddenness claim."""
    return [{'deficit':kind,'near_stock_available':available,
             'expected_useful_e2':kind+'_stock',
             'expected_after_e2':'retrieve_near' if available else 'search_alternate',
             'information_condition':'unverified',
             'required_audit':'Check fixed RGB before/after actual clearing; visible usable matching stock removes the E2 availability uncertainty.'}
            for kind in ('cup','plate') for available in (False,True)]


def classify_information(*, deficit_visible, relevant_stock_visible, clearing_reveals_deficit):
    """Labels from an external perceptual audit, never evaluator hidden geometry."""
    if deficit_visible and relevant_stock_visible: return 'visible'
    if deficit_visible or relevant_stock_visible: return 'one_hop'
    if clearing_reveals_deficit: return 'conditional_candidate'
    return 'unverified'
