"""Placement rules of the fob search and the per-layout table
that lets the mint apply them without building a kitchen.

**Rule 1 (height).** Every place that holds something -- a token, the target, a decoy -- has an interior
floor at least :data:`FLOOR_MIN_M` above the kitchen floor: a drawer whose floor is at that height or
higher, or a counter spot. Bottom drawers (floor about 0.32 m) and SingleCabinets are never used, as a
chain link, a hiding place or a free compartment. Measured: the PandaOmron's gripper cannot get below
about 0.41 m when the object is 0.6-0.8 m ahead of the base (``reach_envelope.py`` in the 2026-09-24
reference folder), so a 14 mm token on a 0.38 m cabinet shelf is out of reach altogether, while the
drawers above the bottom one (floors 0.53 and 0.74 m in RoboCasa's base stacks) are fine.

**Rule 2 (double-door exception).** In a *double-door* instance the final target (a tall pool object
that the pool allows behind a door -- bottle, can or tin; never a token, and not the mug, which the pool
keeps to drawers because its rim slips out of a front pinch) sits in a HingeCabinet base cabinet, inside
the aperture of the leaf that opens (the search family's lateral pose), its centre ``target_depth_m``
behind the cabinet's front face (the door line). The spec records that depth -- :data:`DOOR_TARGET_DEPTH_M`,
the search family's own door pose (the interior starts 0.03-0.04 m behind the front face and the family
hides 0.06 m behind that), the pose the reach certificate grasped from in 0.94 m cabinets -- and the build
places the object at exactly that depth, so the recorded and the actual depth are one number; the rule
caps it at :data:`DOOR_TARGET_DEPTH_MAX_M` (a shallow target keeps reach a non-issue). Door widths of 0.7 and 0.9 m
count the same. Everything else in that instance follows rule 1: tokens in rule-1 drawers or on counters,
mid-chain links and the dead end rule-1 drawers only.

**Rule 3 (counter band).** Every task object on a counter -- a token in the open, a reader
pad, a visible pool object -- sits in the front :data:`COUNTER_WALL_FRONT_FRACTION` of the counter's depth,
measured from the front edge, when the counter's back is against a wall; on an island, which the robot can
approach from both sides, anywhere across the depth. Recorded in the spec (``rules.counter_band``), applied
at build as a filter on every counter spot the fob draws (:func:`counter_band`), recorded per object and
re-asserted by G3. (The edge band the search family samples pads and open tokens from, 0.12-0.24 m inside
the approachable edge, already lies inside the front two thirds of any counter deeper than 0.36 m; the
visible pool objects were sampled anywhere before this rule.)

**Free compartments** (the six enabled compartments less the ones that hold something) are rule-1
drawers or HingeCabinets: a HingeCabinet holding nothing is a decoy the robot may open, and it is the
only door kind left once SingleCabinets are excluded everywhere.

The floor is measured from fixture geometry (``search.layout.interior_region``: fixture z plus the
region's z offset, the number the evaluated episodes' analysis read), never from a handle or a guess.

:data:`LAYOUT_PLACES` is the per-layout supply, probed headless by
``scripts/roboquest_fob_heights.py`` (a property of the layout: styles 1 and 11 agree on layouts 4, 9, 13
and 17): rule-1 drawers and HingeCabinet doors by interior height in millimetres (``{height_mm: count}``),
plus the counts the rule excludes. ``None`` means the layout fails the build outright; a layout the probe
never saw is ``'unknown'`` and the build decides. :func:`placement_problems` is the mint-time hard filter
(a kitchen short of six allowed compartments, or of rule-1 drawers for the chain, the dead end and the
decoys, or of a HingeCabinet for a double-door target, is refused before anything is built).
"""
from roboquest.search.layout import local_from_world, patch_for_spot
from roboquest.search.objects import POOL, fits

FLOOR_MIN_M = .40                 # rule 1: interior floor of any place that holds something
DOOR_TARGET_DEPTH_M = .10         # rule 2: the door target's centre behind the door line (the family's pose)
DOOR_TARGET_DEPTH_MAX_M = .20     # ... and the cap on it
COUNTER_WALL_FRONT_FRACTION = 2. / 3.   # rule 3: on a wall-backed counter, objects in the front two thirds
HINGE_CLASS = 'HingeCabinet'      # the only door class a double-door target (or a free door) may use
SINGLE_CLASS = 'SingleCabinet'
RULES = dict(floor_min_m=FLOOR_MIN_M, door_target_depth_m=DOOR_TARGET_DEPTH_M,
             door_target_depth_max_m=DOOR_TARGET_DEPTH_MAX_M,
             bottom_drawers=False, single_cabinets=False, door_class=HINGE_CLASS,
             counter_band=dict(wall_front_fraction=COUNTER_WALL_FRONT_FRACTION, island_any=True))


def counter_band(spot, records):
    """Where a counter spot sits across its counter's depth, and whether rule 3 allows it there.

    ``front_depth_m`` is measured from the counter region's front edge (RoboCasa's local -y side, the one
    the robot stands at; ``search.layout.spot_standoff`` measures its approachable edge the same way).
    A counter is an island when RoboCasa names it so; every other counter is wall-backed.
    """
    patch = patch_for_spot(records, spot)
    if patch is None or patch.get('is_shelf'):
        return dict(ok=False, reason='not a counter patch')
    rec = records[patch['counter']]
    local = local_from_world(rec['pos'], rec['rot'], [float(spot['xy'][0]), float(spot['xy'][1]), 0.])
    depth = float(patch['size_xy'][1])
    front = float(local[1]) - float(patch['offset_local'][1]) + depth / 2
    island = bool(patch.get('is_island'))
    ok = island or front <= COUNTER_WALL_FRONT_FRACTION * depth + 1e-6
    return dict(counter=patch['counter'], region=patch['region'], is_island=island,
                front_depth_m=round(front, 4), depth_m=round(depth, 4), ok=bool(ok))


def target_depth():
    """The depth a double-door target is recorded with and placed at (behind the door line)."""
    return DOOR_TARGET_DEPTH_M


def depth_allowed(depth):
    return (isinstance(depth, (int, float)) and not isinstance(depth, bool)
            and 0. < float(depth) <= DOOR_TARGET_DEPTH_MAX_M + 1e-6)


def drawer_is_rule1(candidate):
    """A drawer whose interior floor is at least :data:`FLOOR_MIN_M` (a compartment record or probe row)."""
    if candidate.get('kind') != 'drawer':
        return False
    floor = candidate['region']['floor_z'] if 'region' in candidate else candidate['floor_z']
    return float(floor) >= FLOOR_MIN_M - 1e-6


def door_is_hinge(candidate):
    return candidate.get('kind') == 'door' and candidate.get('cls') == HINGE_CLASS


def allowed_compartment(candidate):
    """May this compartment be one of the six? Rule-1 drawers and HingeCabinets; bottom drawers and
    SingleCabinets never."""
    return drawer_is_rule1(candidate) or door_is_hinge(candidate)


def allowed_place(candidate, role, double_door=False):
    """May this compartment *hold* something? Tokens, decoys and mid-chain links: rule-1 drawers only.
    The target: a rule-1 drawer, or (double-door instance only) a HingeCabinet."""
    if drawer_is_rule1(candidate):
        return True
    return bool(double_door) and role == 'target' and door_is_hinge(candidate)


def places_from_rows(rows):
    """The per-layout supply from probe rows (``kind``, ``cls``, ``floor_z``, ``height_mm``)."""
    drawers, hinge, low, single = {}, {}, 0, 0
    for row in rows:
        if row['kind'] == 'drawer':
            if float(row['floor_z']) >= FLOOR_MIN_M - 1e-6:
                drawers[int(row['height_mm'])] = drawers.get(int(row['height_mm']), 0) + 1
            else:
                low += 1
        elif row['cls'] == HINGE_CLASS:
            hinge[int(row['height_mm'])] = hinge.get(int(row['height_mm']), 0) + 1
        elif row['cls'] == SINGLE_CLASS:
            single += 1
    return dict(drawers=dict(sorted(drawers.items())), hinge_doors=dict(sorted(hinge.items())),
                low_drawers=low, single_doors=single)


def layout_places(layout_id):
    """The probe row for ``layout_id``: a dict, ``None`` when the layout fails the build outright, and
    ``'unknown'`` when it was never probed (the caller then leaves the decision to the build)."""
    return LAYOUT_PLACES.get(int(layout_id), 'unknown')


def _take_fitting(pool, object_id):
    """Spend the smallest interior in ``pool`` ({height_mm: count}) that fits the object; the height or None.
    Smallest-fitting-first is the optimal greedy for nested height constraints."""
    for height in sorted(pool):
        if pool[height] > 0 and fits(object_id, height / 1000.):
            pool[height] -= 1
            return height
    return None


def objects_hideable(layout_id, kind, objects=None):
    """Pool objects the kitchen can hide in a rule-1 drawer (``kind='drawer'``) or a HingeCabinet
    (``kind='door'``): the pool's own kind rule plus an interior of that kind tall enough. On a layout the
    probe never saw only the pool rule applies."""
    objects = sorted(POOL) if objects is None else list(objects)
    places = layout_places(layout_id)
    out = []
    for object_id in objects:
        if kind not in POOL[object_id]['kinds']:
            continue
        if places in (None, 'unknown'):
            out.append(object_id)
            continue
        pool = places['drawers'] if kind == 'drawer' else places['hinge_doors']
        if any(count > 0 and fits(object_id, height / 1000.) for height, count in pool.items()):
            out.append(object_id)
    return out


def placement_problems(layout_id, spec, compartment_count=6):
    """Reasons the placement table says this kitchen cannot host the spec under rules 1 and 2 (``[]`` when
    it can). Counts what the build will bind: every chain link, the dead end and every decoy take a
    distinct rule-1 drawer (the last link a HingeCabinet instead in a double-door instance), tall enough
    for what it holds; the six enabled compartments are rule-1 drawers and HingeCabinets."""
    places = layout_places(layout_id)
    if places == 'unknown':
        return []
    if places is None:
        return [f'layout {layout_id} fails the build outright (placement table)']
    problems = []
    drawers, hinge = dict(places['drawers']), dict(places['hinge_doors'])
    allowed = sum(drawers.values()) + sum(hinge.values())
    if allowed < int(compartment_count):
        problems.append(f'layout {layout_id} offers {allowed} compartments allowed by the height rule '
                        f'(rule-1 drawers and HingeCabinets), fewer than {compartment_count} (placement table)')
    double_door = bool(spec.get('double_door'))
    target = (spec.get('targets') or [{}])[0].get('object')
    depth = int(spec.get('chain_depth') or 0)
    # The target's place first (the tallest demand on its kind), then the decoys, then the plain links.
    if target in POOL:
        if double_door:
            if _take_fitting(hinge, target) is None:
                problems.append(f'layout {layout_id} has no HingeCabinet that fits the door target {target!r} '
                                f'(placement table)')
        elif _take_fitting(drawers, target) is None:
            problems.append(f'layout {layout_id} has no rule-1 drawer that fits {target!r} (placement table)')
    for object_id in sorted(spec.get('hidden_distractors') or [], key=lambda o: -POOL[o]['height']):
        if object_id in POOL and _take_fitting(drawers, object_id) is None:
            problems.append(f'layout {layout_id} has no rule-1 drawer left for decoy {object_id!r} (placement table)')
    links = max(0, depth - 1) + int(bool(spec.get('dead_end')))     # mid-chain links and the dead end hold tokens
    if links > sum(drawers.values()):
        problems.append(f'layout {layout_id} has {sum(drawers.values())} rule-1 drawers left for {links} token '
                        f'links (placement table)')
    return problems


# Probed 2026-09-24 by scripts/roboquest_fob_heights.py (style 1, headless, four workers; style 11 agrees on
# layouts 4, 9, 13, 17). Floor rule 0.40 m. Per layout: rule-1 drawers and HingeCabinet doors by interior height
# in millimetres, and how many bottom drawers / SingleCabinets the rule excludes.
LAYOUT_PLACES = {
    1: None,   # search_room: layout 1 has no counter region free of decor for the task frame
    2: dict(drawers={118: 2}, hinge_doors={}, low_drawers=2, single_doors=0),
    3: dict(drawers={118: 6}, hinge_doors={240: 1, 270: 1}, low_drawers=2, single_doors=1),
    4: dict(drawers={118: 4}, hinge_doors={270: 4}, low_drawers=1, single_doors=1),
    5: dict(drawers={118: 12}, hinge_doors={270: 4}, low_drawers=2, single_doors=1),
    6: dict(drawers={118: 10}, hinge_doors={270: 4}, low_drawers=2, single_doors=2),
    7: dict(drawers={118: 9}, hinge_doors={270: 2}, low_drawers=1, single_doors=2),
    8: dict(drawers={118: 9}, hinge_doors={270: 2}, low_drawers=2, single_doors=2),
    9: dict(drawers={118: 9, 208: 2}, hinge_doors={240: 2, 270: 3, 378: 1}, low_drawers=1, single_doors=4),
    10: dict(drawers={118: 9, 208: 1}, hinge_doors={270: 5, 540: 1}, low_drawers=1, single_doors=3),
    11: dict(drawers={118: 7}, hinge_doors={270: 1}, low_drawers=2, single_doors=4),
    12: dict(drawers={118: 2, 177: 1}, hinge_doors={270: 1}, low_drawers=1, single_doors=3),
    13: dict(drawers={177: 4, 185: 2}, hinge_doors={240: 1}, low_drawers=3, single_doors=4),
    14: dict(drawers={160: 1}, hinge_doors={}, low_drawers=0, single_doors=2),
    15: dict(drawers={118: 4, 177: 1}, hinge_doors={270: 3}, low_drawers=1, single_doors=1),
    16: dict(drawers={118: 9, 328: 2}, hinge_doors={270: 4}, low_drawers=0, single_doors=1),
    17: dict(drawers={160: 2, 185: 5, 328: 3}, hinge_doors={240: 2}, low_drawers=5, single_doors=4),
    18: dict(drawers={185: 5}, hinge_doors={240: 1, 432: 1}, low_drawers=6, single_doors=6),
    19: dict(drawers={118: 9}, hinge_doors={270: 3}, low_drawers=2, single_doors=2),
    20: dict(drawers={}, hinge_doors={240: 1, 414: 2}, low_drawers=0, single_doors=5),
    21: dict(drawers={118: 2, 160: 3}, hinge_doors={240: 1, 270: 2}, low_drawers=0, single_doors=2),
    22: dict(drawers={118: 8}, hinge_doors={240: 2, 270: 3}, low_drawers=0, single_doors=3),
    23: dict(drawers={185: 1}, hinge_doors={240: 3}, low_drawers=1, single_doors=1),
    24: dict(drawers={185: 1, 328: 1}, hinge_doors={305: 1, 335: 1}, low_drawers=1, single_doors=2),
    25: dict(drawers={118: 4}, hinge_doors={240: 1}, low_drawers=2, single_doors=3),
    26: dict(drawers={160: 4, 185: 1, 368: 1}, hinge_doors={240: 3, 300: 1}, low_drawers=1, single_doors=0),
    27: dict(drawers={160: 6, 177: 1}, hinge_doors={240: 1, 305: 1}, low_drawers=1, single_doors=2),
    28: dict(drawers={}, hinge_doors={240: 1}, low_drawers=0, single_doors=1),
    29: dict(drawers={177: 4, 185: 1}, hinge_doors={240: 1}, low_drawers=3, single_doors=2),
    30: dict(drawers={118: 3, 160: 6}, hinge_doors={240: 1}, low_drawers=1, single_doors=3),
    31: dict(drawers={160: 4, 202: 3}, hinge_doors={240: 1, 277: 1, 330: 1}, low_drawers=2, single_doors=2),
    32: dict(drawers={160: 3, 202: 1}, hinge_doors={240: 1, 305: 1}, low_drawers=1, single_doors=1),
    33: dict(drawers={160: 3}, hinge_doors={305: 1}, low_drawers=0, single_doors=0),
    34: dict(drawers={202: 1}, hinge_doors={240: 1}, low_drawers=2, single_doors=3),
    35: dict(drawers={202: 1}, hinge_doors={}, low_drawers=1, single_doors=5),
    36: dict(drawers={160: 4}, hinge_doors={}, low_drawers=0, single_doors=0),
    37: dict(drawers={160: 3, 202: 1}, hinge_doors={490: 1}, low_drawers=0, single_doors=0),
    38: dict(drawers={160: 3, 328: 1}, hinge_doors={240: 3, 305: 1}, low_drawers=0, single_doors=3),
    39: dict(drawers={160: 19, 202: 2}, hinge_doors={}, low_drawers=0, single_doors=0),
    40: dict(drawers={160: 2}, hinge_doors={240: 1}, low_drawers=0, single_doors=2),
    41: dict(drawers={160: 4, 328: 1}, hinge_doors={305: 1}, low_drawers=0, single_doors=2),
    42: dict(drawers={202: 1}, hinge_doors={240: 3}, low_drawers=2, single_doors=2),
    43: dict(drawers={160: 3, 202: 2, 328: 1}, hinge_doors={240: 1, 305: 1}, low_drawers=0, single_doors=6),
    44: dict(drawers={160: 2}, hinge_doors={240: 1}, low_drawers=0, single_doors=0),
    45: dict(drawers={160: 2}, hinge_doors={240: 1}, low_drawers=0, single_doors=1),
    46: dict(drawers={160: 4, 202: 2}, hinge_doors={240: 2, 305: 1}, low_drawers=2, single_doors=2),
    47: dict(drawers={118: 2, 160: 2}, hinge_doors={250: 1, 267: 1}, low_drawers=1, single_doors=0),
    48: dict(drawers={160: 4, 202: 3}, hinge_doors={305: 1, 486: 1}, low_drawers=0, single_doors=1),
    49: dict(drawers={160: 11, 202: 2}, hinge_doors={}, low_drawers=0, single_doors=2),
    50: dict(drawers={160: 8, 202: 1}, hinge_doors={305: 1}, low_drawers=0, single_doors=0),
    51: dict(drawers={160: 8, 202: 2}, hinge_doors={240: 1}, low_drawers=0, single_doors=1),
    52: dict(drawers={202: 2}, hinge_doors={}, low_drawers=2, single_doors=2),
    53: dict(drawers={160: 4, 202: 1}, hinge_doors={}, low_drawers=0, single_doors=4),
    54: dict(drawers={160: 3, 202: 1}, hinge_doors={305: 1}, low_drawers=1, single_doors=3),
    55: dict(drawers={160: 4}, hinge_doors={}, low_drawers=0, single_doors=1),
    56: dict(drawers={160: 5, 202: 3}, hinge_doors={240: 3, 305: 1, 528: 1}, low_drawers=2, single_doors=4),
    57: dict(drawers={160: 8, 202: 1}, hinge_doors={305: 1}, low_drawers=0, single_doors=2),
    58: dict(drawers={160: 4, 202: 1}, hinge_doors={305: 1, 403: 1}, low_drawers=0, single_doors=1),
    59: dict(drawers={160: 5}, hinge_doors={305: 1}, low_drawers=0, single_doors=0),
    60: dict(drawers={160: 2, 328: 1}, hinge_doors={324: 1}, low_drawers=0, single_doors=2),
}
