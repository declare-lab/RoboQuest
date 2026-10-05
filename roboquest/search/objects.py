"""Object pool for the search tasks: native RoboCasa assets recoloured to one flat colour each.

Eight objects, two colours per category (mug, can, tin, water bottle), so a
distractor can share the category *or* the colour of a target; targets are named
by colour and category ("the blue mug"). Sizes were measured with MJCFObject
(room-search `measure_pool.py`) so each object fits the gripper's 8 cm opening;
`height` gates which compartment interiors can hold it. The mug is drawers-only
(its flared rim slips out of a front pinch during carries) and the bottle is
too tall for most drawers, so each category lists the hiding-place kinds it may
be assigned to. ``covered`` (v1) is an open counter spot under an opaque cloche
(``search/covers.py``); the bottle (0.135 m) does not fit under the large
cloche's 0.124 m interior and is never covered. No simulator imports here.
"""
from copy import deepcopy

COLOURS = {
    'blue': (.10, .25, .85, 1.),
    'red': (.85, .12, .10, 1.),
    'green': (.12, .65, .20, 1.),
    'yellow': (.92, .80, .10, 1.),
}
PLACE_KINDS = ('drawer', 'door', 'open', 'covered')
COVER_KINDS = ('cloche_large',)   # inverted RoboCasa bowls are 2.7-6.2 cm deep: too shallow for every pool object
CATEGORIES = {
    'mug': {
        'asset': 'objaverse/mug/mug_1', 'label': 'mug',
        'size_xyz_m': (.0786, .115, .0904), 'height': .0904, 'body_diameter_m': .069,
        # Drawers only: the flared rim and handle defeat a front pinch (it slips out within
        # seconds of driving) and a top grasp cannot reach a shelf under a counter.
        'drawers_only': True, 'kinds': ('drawer', 'open', 'covered'),
        'grasp': {'anchor_local_top': (0., .018, .028), 'anchor_local_front': (0., .018, .005), 'handle': True},
    },
    'can': {
        'asset': 'objaverse/can/can_3', 'label': 'can',
        'size_xyz_m': (.0603, .0584, .111), 'height': .111, 'body_diameter_m': .060,
        'drawers_only': False, 'kinds': ('drawer', 'door', 'open', 'covered'),
        # Top anchors sit high on the body (origin = geometric centre) so the object hangs below the
        # pinch axis during carries; pinched at the centre of mass the tin slipped out on base turns.
        'grasp': {'anchor_local_top': (0., 0., .038), 'anchor_local_front': (0., 0., 0.), 'handle': False},
    },
    'tin': {
        'asset': 'objaverse/canned_food/canned_food_1', 'label': 'tin',
        'size_xyz_m': (.0681, .0681, .0958), 'height': .0958, 'body_diameter_m': .068,
        'drawers_only': False, 'kinds': ('drawer', 'door', 'open', 'covered'),
        'grasp': {'anchor_local_top': (0., 0., .032), 'anchor_local_front': (0., 0., 0.), 'handle': False},
    },
    'bottle': {
        'asset': 'objaverse/water_bottle/water_bottle_1', 'label': 'water bottle',
        'size_xyz_m': (.0482, .0524, .135), 'height': .135, 'body_diameter_m': .050,
        # 0.135 m tall: most drawers (0.12 m interiors) cannot hold it, so it is never assigned a drawer.
        'drawers_only': False, 'kinds': ('door', 'open'),
        'grasp': {'anchor_local_top': (0., 0., .040), 'anchor_local_front': (0., 0., 0.), 'handle': False},
    },
}
CATEGORY_COLOURS = {'mug': ('blue', 'red'), 'can': ('red', 'green'), 'tin': ('green', 'yellow'), 'bottle': ('yellow', 'blue')}
HEIGHT_CLEARANCE_M = .02


def _build_pool():
    pool = {}
    for category, colours in CATEGORY_COLOURS.items():
        for colour in colours:
            entry = deepcopy(CATEGORIES[category])
            entry.update(category=category, colour=colour, rgba=COLOURS[colour],
                         description=f"the {colour} {entry['label']}")
            pool[f'{category}_{colour}'] = entry
    return pool


POOL = _build_pool()


def pool_entry(object_id):
    if object_id not in POOL:
        raise ValueError(f'unknown pool object {object_id!r}; choose from {sorted(POOL)}')
    return deepcopy(POOL[object_id])


def fits(object_id, region_height):
    return float(region_height) >= POOL[object_id]['height'] + HEIGHT_CLEARANCE_M


def fits_compartment(object_id, compartment):
    """Height clearance plus the category's kind restrictions (mug: drawers only)."""
    entry = POOL[object_id]
    if entry['drawers_only'] and compartment['kind'] != 'drawer':
        return False
    if compartment['kind'] not in entry['kinds']:
        return False
    return fits(object_id, compartment['region']['height'])


COVER_HEIGHT_CLEARANCE_M = .01   # a cover only has to clear the object's top; the cloche is lifted straight up


def fits_cover(object_id, interior_radius, interior_height):
    """The object stands upright under the cover with its widest extent inside the interior radius."""
    entry = POOL[object_id]
    if 'covered' not in entry['kinds']:
        return False
    half = max(entry['size_xyz_m'][0], entry['size_xyz_m'][1]) / 2
    return half + .01 <= float(interior_radius) and entry['height'] + COVER_HEIGHT_CLEARANCE_M <= float(interior_height)


def confusable(a, b):
    """Two pool objects that share a category or a colour."""
    return a != b and (POOL[a]['category'] == POOL[b]['category'] or POOL[a]['colour'] == POOL[b]['colour'])


def describe_targets(object_ids):
    names = [POOL[o]['description'] for o in object_ids]
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + ' and ' + names[-1]
