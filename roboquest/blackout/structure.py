"""Blackout Search structure (torch-only): what a spec draws, how it
splits, where a target may hide, and what progress is worth.

The grid is the search room's: ``set_size`` 2 or 3 targets among ``compartment_count`` 4, 6 or 8 openable
compartments, six cells. There is no lighting factor any more: the lantern is lit from the start, it is the
only light in the kitchen, and there is no wall switch (no ``switch`` / ``switch_timed`` / ``lamp_only``
axis).

Hiding places (the reach analysis of 2026-09-24): a target or a decoy hides in a **drawer
whose interior floor is at least** :data:`HIDING_FLOOR_MIN_M` (0.40 m), on an **open** counter spot the
darkness hides (at least ``LAMP_TARGET_MIN_M`` from the lantern's reset spot, see the task module) or
**under a cloche** (``covered``, the search room's kind: the cloche may stand inside the lantern's pool,
which is what makes it different from ``open`` here). No door places at all -- no SingleCabinet, no
HingeCabinet -- and no bottom drawers (floor 0.32 m): the PandaOmron's gripper does not get below about
0.41 m at the distances a base-cabinet shelf or a bottom drawer puts the object at. The rule is applied at
build as a hard filter over the kitchen's compartments (:func:`admissible_hiding_place`), recorded in the
spec (``hiding``) and checked by ``validate_spec``; the *filler* compartments (enabled, empty) keep the
search room's drawer/door mix so the policy still has doors to rule out.

Split key: the search room's ``s{set_size}-{sorted kinds}``. Hold-outs, one per set size:

=============  ==========================  ========================================================
set_size       held out (eval only)        kept in dev
=============  ==========================  ========================================================
2              ``s2-covered+open``         the other five pairs over drawer / open / covered
3              ``s3-covered+drawer+open``  the other nine triples
=============  ==========================  ========================================================

So every one of the six factor cells has eval rows (the key does not name the compartment count), and
eval asks for a composition dev never saw at that set size: at 2, both targets on counters (one in the
dark, one under a cloche); at 3, one of each kind. The search room's ``s3-door+door+drawer`` pattern is
not copied (no doors).

Progress (contract section 3): the search room's per-target steps, mean over the targets,
plus the lantern picked up at least once -- always, because the lantern is now the only light.
"""
from roboquest.search.layout import _take_fitting, layout_capacity
from roboquest.search.objects import POOL, confusable, fits

SET_SIZES = (2, 3)
COMPARTMENT_COUNTS = (4, 6, 8)
FACTORS = {'set_size': SET_SIZES, 'compartment_count': COMPARTMENT_COUNTS}
HIDING_KINDS = ('drawer', 'open', 'covered')       # never 'door'
HIDING_FLOOR_MIN_M = .40                            # interior floor of a drawer that hides a target or a decoy
HIDING_RULE = dict(floor_min_m=HIDING_FLOOR_MIN_M, kinds=list(HIDING_KINDS), doors=False)
# The counter rule: on a counter whose back is against a wall, a target (open or under a
# cloche) and the lantern go only in the front two thirds of the depth, measured from the front edge; on an
# island, approachable from both sides, anywhere across the depth. Applied at build over every sampled counter
# spot (`search.layout.counter_band`), recorded in the spec and checked by `validate_spec` and G3. A low open
# shelf may hold an open target only with its top at HIDING_FLOOR_MIN_M or higher (the same floor rule).
COUNTER_FRONT_FRACTION = round(2 / 3, 4)
COUNTER_BAND_RULE = dict(front_fraction=COUNTER_FRONT_FRACTION, wall_backed_only=True, islands='anywhere')
HIDDEN_DISTRACTORS = 2                              # at most; one when the targets leave a single compartment free
HOLDOUT_KEYS = ('s2-covered+open', 's3-covered+drawer+open')

GOAL_NOUN = 'objects'
GOAL_COUNT = len(POOL) + 1
GOAL_TEMPLATE = ('The kitchen is dark. Collect {targets} on the tray, then press Submit. They may be anywhere in '
                 'this kitchen; some compartments may not open. A lantern with a glowing glass stands on a '
                 'counter.')

# Progress weights: search's three steps rescaled, plus the lantern (the only light, so a policy that never
# lifts it has found nothing on its own); they sum to 1.
LAMP_WEIGHT = .10
PROGRESS_WEIGHTS = dict(out_of_hiding=.45, on_tray=.27, submitted=.18, lamp_picked=LAMP_WEIGHT)


def progress_weights():
    return dict(PROGRESS_WEIGHTS)


def hiding_kinds_of(object_id):
    """The place kinds a pool object may take here: its own kinds without the doors."""
    return tuple(k for k in HIDING_KINDS if k in POOL[object_id]['kinds'])


def decoy_capable(object_id):
    """A decoy hides in a compartment, and the only compartment kind here is the drawer."""
    return 'drawer' in POOL[object_id]['kinds']


def admissible_hiding_place(candidate, floor_min_m=HIDING_FLOOR_MIN_M):
    """A compartment may hide a target or a decoy only if it is a drawer whose interior floor is
    at least ``floor_min_m`` up. Doors (SingleCabinet, HingeCabinet) and bottom drawers never qualify."""
    if candidate.get('kind') != 'drawer' or candidate.get('cls') != 'Drawer':
        return False
    region = candidate.get('region') or {}
    try:
        floor_z = float(region['floor_z'])
    except (KeyError, TypeError, ValueError):
        return False
    return floor_z >= float(floor_min_m)


def hidden_distractor_count(compartment_count, kinds):
    free = int(compartment_count) - sum(1 for k in kinds if k == 'drawer')
    return max(1, min(HIDDEN_DISTRACTORS, free))


def fits_some_drawer(object_id, high):
    """Does some admissible drawer of the layout (``{height_mm: n}``) take the object? Unknown tables admit."""
    if not isinstance(high, dict):
        return True
    return any(n > 0 and fits(object_id, height / 1000.) for height, n in high.items())


def draw_structure(structure, set_size, compartment_count, high=None):
    """Targets with their place kinds, the confusable decoys and the visible rest, from the structure stream.

    Kinds are drawn per target from :func:`hiding_kinds_of` (the bottle is open-counter only: it fits no
    drawer and no cloche); decoys are drawn from the objects that share a category or a colour with a
    target *and* can hide in a drawer. ``high`` is the kitchen's admissible-drawer table
    (:func:`high_drawers`): where it is known, a target draws ``drawer`` and an object becomes a decoy
    only if some admissible drawer of that kitchen takes it (the can needs a 0.131 m interior, which the
    0.118 m drawers of most RoboCasa test layouts do not have), so a seed is not spent on a spec the
    capacity check would refuse anyway. The check after the draw stays the authority.
    """
    ids = sorted(POOL)
    targets = [ids[int(i)] for i in structure.choice(len(ids), int(set_size), replace=False)]
    kinds = []
    for target in targets:
        allowed = tuple(k for k in hiding_kinds_of(target) if k != 'drawer' or fits_some_drawer(target, high))
        kinds.append(allowed[int(structure.integers(len(allowed)))])
    others = [o for o in ids if o not in targets]
    candidates = [o for o in others if decoy_capable(o) and fits_some_drawer(o, high)
                  and any(confusable(o, t) for t in targets)]
    n_hidden = min(hidden_distractor_count(compartment_count, kinds), len(candidates))
    hidden = sorted(candidates[int(i)] for i in structure.choice(len(candidates), n_hidden, replace=False))
    visible = [o for o in others if o not in hidden]
    return [dict(object=t, kind=k) for t, k in zip(targets, kinds)], hidden, visible


# ---- mint-time capacity: which layouts can hide the spec under the floor rule ------------------------
# Drawers with an interior floor >= HIDING_FLOOR_MIN_M per RoboCasa layout, as {interior height mm: how
# many} over the compartments `search.layout.compartment_candidates` admits (reachable interior, openable
# joint, base standoff with room to reverse). A property of the layout, not of the style (the cabinet run
# is the same at every style). Probed 2026-09-24 (results/dark-torch/probe/floors_style1.json, style 1,
# headless). `None`: the layout fails the work-frame build outright; a layout missing here is left to the
# build. Every other number (usable compartments, counter spots) comes from `search.layout.LAYOUT_CAPACITY`.
HIGH_DRAWERS = {
    1: None,   # ValueError: search_room: layout 1 has no counter region free of decor 
    2: {118: 2},   # below the rule: 1 at 0.321 m, 1 at 0.361 m
    3: {118: 6},   # below the rule: 2 at 0.321 m
    4: {118: 4},   # below the rule: 1 at 0.321 m
    5: {118: 12},   # below the rule: 2 at 0.321 m
    6: {118: 10},   # below the rule: 2 at 0.321 m
    7: {118: 9},   # below the rule: 1 at 0.321 m
    8: {118: 9},   # below the rule: 2 at 0.321 m
    9: {118: 9, 208: 2},   # below the rule: 1 at 0.321 m
    10: {118: 9, 208: 1},   # below the rule: 1 at 0.321 m
    11: {118: 7},   # below the rule: 2 at 0.321 m
    12: {118: 2, 177: 1},   # below the rule: 1 at 0.397 m
    13: {177: 4, 185: 2},   # below the rule: 3 at 0.397 m
    14: {160: 1},
    15: {118: 4, 177: 1},   # below the rule: 1 at 0.397 m
    16: {118: 9, 328: 2},
    17: {160: 2, 185: 5, 328: 3},   # below the rule: 5 at 0.397 m
    18: {185: 5},   # below the rule: 6 at 0.397 m
    19: {118: 9},   # below the rule: 2 at 0.321 m
    20: {},
    21: {118: 2, 160: 3},
    22: {118: 8},
    23: {185: 1},   # below the rule: 1 at 0.397 m
    24: {185: 1, 328: 1},   # below the rule: 1 at 0.397 m
    25: {118: 4},   # below the rule: 1 at 0.319 m, 1 at 0.321 m
    26: {160: 4, 185: 1, 368: 1},   # below the rule: 1 at 0.397 m
    27: {160: 6, 177: 1},   # below the rule: 1 at 0.397 m
    28: {},
    29: {177: 4, 185: 1},   # below the rule: 3 at 0.397 m
    30: {118: 3, 160: 6},   # below the rule: 1 at 0.321 m
    31: {160: 4, 202: 3},   # below the rule: 2 at 0.363 m
    32: {160: 3, 202: 1},   # below the rule: 1 at 0.363 m
    33: {160: 3},
    34: {202: 1},   # below the rule: 2 at 0.363 m
    35: {202: 1},   # below the rule: 1 at 0.363 m
    36: {160: 4},
    37: {160: 3, 202: 1},
    38: {160: 3, 328: 1},
    39: {160: 19, 202: 2},
    40: {160: 2},
    41: {160: 4, 328: 1},
    42: {202: 1},   # below the rule: 2 at 0.363 m
    43: {160: 3, 202: 2, 328: 1},
    44: {160: 2},
    45: {160: 2},
    46: {160: 4, 202: 2},   # below the rule: 2 at 0.363 m
    47: {118: 2, 160: 2},   # below the rule: 1 at 0.321 m
    48: {160: 4, 202: 3},
    49: {160: 11, 202: 2},
    50: {160: 8, 202: 1},
    51: {160: 8, 202: 2},
    52: {202: 2},   # below the rule: 2 at 0.363 m
    53: {160: 4, 202: 1},
    54: {160: 3, 202: 1},   # below the rule: 1 at 0.363 m
    55: {160: 4},
    56: {160: 5, 202: 3},   # below the rule: 2 at 0.363 m
    57: {160: 8, 202: 1},
    58: {160: 4, 202: 1},
    59: {160: 5},
    60: {160: 2, 328: 1},
}


def high_drawers(layout_id):
    """``{height_mm: count}`` of the admissible drawers of ``layout_id``, ``None`` when the layout fails the
    build, ``'unknown'`` when it was never probed."""
    return HIGH_DRAWERS.get(int(layout_id), 'unknown')


def capacity_problems(layout_id, compartment_count, targets, hidden_distractors=(), visible=()):
    """Why the tables say this kitchen cannot host the spec under the floor rule; ``[]`` when it can.

    Enough usable compartments for ``compartment_count`` (any kind: fillers are unrestricted), an admissible
    drawer that fits every drawer target and every decoy (tallest first, the order the build binds them),
    and counter spots for the open, covered and visible objects, all from the layout tables.
    """
    cap = layout_capacity(layout_id)
    high = high_drawers(layout_id)
    if cap == 'unknown' or high == 'unknown':
        return []
    if cap is None or high is None:
        return [f'layout {layout_id} has no counter region free of decor for the task frame (capacity table)']
    problems = []
    count = int(compartment_count)
    usable = sum(cap['drawers'].values()) + sum(cap['doors'].values())
    if usable < count:
        problems.append(f'layout {layout_id} offers {usable} usable compartments, fewer than '
                        f'compartment_count={count} (capacity table)')
    pool = dict(high)
    drawer_targets = [t for t in targets if t['kind'] == 'drawer']
    for target in sorted(drawer_targets, key=lambda t: -POOL[t['object']]['height']):
        if _take_fitting(pool, target['object']) is None:
            problems.append(f"layout {layout_id} has no drawer with its floor at {HIDING_FLOOR_MIN_M} m or higher "
                            f"that fits {target['object']!r} (high-drawer table)")
    for object_id in sorted(hidden_distractors, key=lambda o: -POOL[o]['height']):
        if not decoy_capable(object_id) or _take_fitting(pool, object_id) is None:
            problems.append(f'layout {layout_id} has no drawer with its floor at {HIDING_FLOOR_MIN_M} m or higher '
                            f'that fits hidden distractor {object_id!r} (high-drawer table)')
    wanted = {'open': sum(1 for t in targets if t['kind'] == 'open'),
              'cover': sum(1 for t in targets if t['kind'] == 'covered'),
              'free': len(list(visible))}
    for key, need in wanted.items():
        if need and int(cap.get(key, 0)) < need:
            problems.append(f'layout {layout_id} offers {cap.get(key)} {key} counter spots, fewer than '
                            f'the {need} the spec needs (capacity table)')
    return problems
