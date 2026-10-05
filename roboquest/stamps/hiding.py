"""Observability draws and build-time candidate walk for the stamps task (spec 2.2).

The level is a registry factor; everything the level needs beyond it - which stamps
are hidden, which board is covered, which flavour of ``uncover`` - is nuisance drawn
from the ``poses`` stream at the end of ``sample_spec``, so two instances that differ
only in observability keep the same structure class, the same hold-out key and the
same goal text.

``draw`` records an ordered candidate list per flavour; the build walks it with
:func:`resolve` and keeps the first candidate whose geometry passes the observe gates,
so a seed is rejected only when no candidate of either flavour works.
"""
import math

from roboquest import observe, scatter as S
from roboquest.stamps import geometry

LEVELS = observe.MODES                                   # ('visible', 'look', 'uncover')
DEV_ONLY_LEVEL = observe.DEV_ONLY_LEVEL                  # 'look+uncover', dev only (contract C2)
DEV_ONLY_LEVELS = {'observability': (DEV_ONLY_LEVEL,)}
ALL_LEVELS = LEVELS + (DEV_ONLY_LEVEL,)
LOOK_MODES = ('annex', 'occluder')
OBJECT_COVER = 'cloche_large'                            # spec 2.2: stamps go under the large cloche
PLACE_COVER_KINDS = ('board', 'tray')                    # ... or a cutting board / tray goes on a board
LOOK_MAX = 2                                             # 1-2 stamps at the look level
CLOCHE_RADIUS = observe.CLOCHE_SIZES['large']['radius']
COVER_GAP = observe.OCCLUDER_GAP_M                       # the gap cover_footprint_clear asks for
FLAT_GAP = .004
ROUND_COVERS = ('plate',)                                # a plate is a disc inscribed in its bounding box
COVER_OVERHANG_M = .012   # a cover overhangs the board it hides by this much; the print lies flush with
                          # the board's face, so observe's own margin is enough (unlike the proud pads of
                          # marked_mugs, which need .040)
COVER_CORNER_PAD = .008                                  # ... and again, for the rounded corners of a board


def cloche_room(xy, name, shapes, bounds=None, gap=COVER_GAP):
    """Would the large cloche standing over a stamp at ``xy`` clear the other props?

    Mint-time half of :func:`observe.cover_footprint_clear`: the same disc against the same
    task-frame footprints, minus the scene decor the generator cannot see. Candidates that
    fail here are dropped before the build, which is what keeps the uncover seed-rejection
    rate down (brief deliverable 7).
    """
    disc = S.Disc((float(xy[0]), float(xy[1])), CLOCHE_RADIUS)
    if bounds is not None and S.region_margin(disc, bounds) < 0.:
        return False
    return all(S.clearance(disc, shape) >= gap for key, shape in shapes.items() if key != name)


def place_room(place, kind, name, shapes, gap=FLAT_GAP):
    """Mint-time version of the resting-footprint test in :func:`flat_cover_feasible`: every
    asset of ``kind`` the build could draw rests clear of the other props."""
    options = flat_cover_options(kind, place['half'])
    if not options:
        return False
    for half, turn in options:
        box = cover_shape(kind, place['xy'], half, float(place.get('yaw', 0.)) + turn)
        if any(S.clearance(box, shape) < gap for key, shape in shapes.items() if key != name):
            return False
    return True


def stamp_footprints(count, reserved=None):
    """Scatter footprints for the stamps, with the cloche's radius reserved around one of them.

    A .11 m cloche rarely fits over a stamp the plain scatter has already placed - at five
    stamps it fits in fewer than a fifth of the draws - so when no stamp has the room the
    generator re-scatters with the cover's own footprint standing in for that stamp. Every
    other stamp keeps its sole radius, so the placement gate still sees a legal layout.
    """
    return [CLOCHE_RADIUS if name == reserved else geometry.SOLE_RADIUS
            for name in (f'stamp_{index}' for index in range(int(count)))]


def stamp_regions(count, reserved=None):
    """Scatter regions matching :func:`stamp_footprints`. The reserved stamp's region is the
    stamp area grown at the back by the difference between the two radii: its *centre* still
    has to stand exactly where any stamp may stand (the placement gate measures the sole), but
    the cover may overhang the back of that area, which is where the free corridor between the
    scratch board and the final board is."""
    region = geometry.stamp_region()
    grown = S.Rect(region.x0, region.x1, region.y0, region.y1 + CLOCHE_RADIUS - geometry.SOLE_RADIUS)
    return [grown if name == reserved else region
            for name in (f'stamp_{index}' for index in range(int(count)))]


def draw(rng, level, stamps, qualifying, *, shapes, places, rescatter=None, bounds=None):
    """Nuisance draws for one instance at ``level``; returns ``(hidden, plan)``.

    ``stamps`` are the stamp names, ``qualifying`` the ones the rule needs (the recorded
    solution), ``shapes`` the task-frame footprint of every prop by name (stamps included)
    and ``places`` the boards that may be covered, each ``dict(xy, half, yaw)``. ``bounds``
    is the frame the cover must stay on. ``rescatter(name)`` is called only when no stamp has
    room for the cloche: it must re-scatter the stamps with :func:`stamp_footprints` reserving
    that room and return the new ``{name: (x, y)}``, or None when even that does not fit.
    ``hidden`` is the spec's entry list (:func:`observe.hidden_entry`) at the preferred
    candidate; ``plan`` carries the flavour, the ordered fallbacks the build may walk and the
    re-scatter the caller has already applied.
    """
    level = str(level)
    if level not in ALL_LEVELS:
        raise ValueError(f'observability {level!r} is not one of {ALL_LEVELS}')
    hidden, used = [], []
    plan = dict(uncover_flavour=None, look=[], object_candidates=[], place_candidates=[],
                rescatter=None, flavour_drawn=None, flavour_fallback=False)
    if level in ('look', DEV_ONLY_LEVEL):
        count = 1 if level == DEV_ONLY_LEVEL else 1 + int(rng.integers(LOOK_MAX))
        for name in observe.draw_hidden(rng, stamps, qualifying, count):
            mode = LOOK_MODES[int(rng.integers(len(LOOK_MODES)))]
            hidden.append(observe.hidden_entry(name, mode))
            plan['look'].append([name, mode])
            used.append(name)
    if level in ('uncover', DEV_ONLY_LEVEL):
        shapes = dict(shapes)
        free = [name for name in stamps if name not in used]
        roomy = [name for name in free if cloche_room(shapes[name].center, name, shapes, bounds)]
        order = observe.draw_hidden(rng, roomy or free, qualifying, len(roomy or free))
        if not roomy and order and rescatter is not None:
            # Nobody has room for the cloche where the plain scatter left them, so the stamps are
            # drawn again with the cover's footprint reserved around the first candidate.
            moved = rescatter(order[0])
            if moved:
                plan['rescatter'] = [order[0], {name: [float(xy[0]), float(xy[1])]
                                                for name, xy in moved.items()}]
                for name, xy in moved.items():
                    shapes[name] = S.Disc((float(xy[0]), float(xy[1])), shapes[name].radius)
                order = [order[0]] + [name for name in order[1:] if cloche_room(shapes[name].center,
                                                                                name, shapes, bounds)]
            else:
                order = []
        plan['object_candidates'] = [[name, OBJECT_COVER] for name in order]
        kinds = [PLACE_COVER_KINDS[int(i)] for i in rng.permutation(len(PLACE_COVER_KINDS))]
        boards = [list(places)[int(i)] for i in rng.permutation(len(places))]
        plan['place_candidates'] = [[board, kind] for board in boards for kind in kinds
                                    if place_room(places[board], kind, board, shapes)]
        # The two flavours are drawn 50/50 (spec 2.2) and both candidate orders are kept, so an
        # infeasible draw falls back to the other flavour instead of losing the seed.
        flavour = 'object' if int(rng.integers(2)) == 0 else 'place'
        plan['flavour_drawn'] = flavour
        first = plan[f'{flavour}_candidates']
        if not first:
            flavour = 'place' if flavour == 'object' else 'object'
            first = plan[f'{flavour}_candidates']
            plan['flavour_fallback'] = bool(first)
        plan['uncover_flavour'] = flavour if first else None
        if not first:                                     # neither flavour is realisable: G0 rejects it
            return hidden, plan
        name, kind = first[0]
        hidden.append(observe.hidden_entry(name, 'cover', kind))
    return hidden, plan


def cover_half(kind, place_half):
    """Half sizes to ask :func:`observe.cover_place` for so the cover really hides the place.

    ``cover_place`` scales a cover until its *bounding box* overhangs the place by
    :data:`observe.FLAT_COVER_MARGIN_M`, which is not the same as covering it: a plate is a disc
    inscribed in that box, so the corners of a rectangular place stay in view (a covered pad read
    27-86 px per camera in the first G3 render), and the boards and trays have rounded corners.
    Asking for the place's circumradius (round kinds) or for a corner pad (rectangular kinds)
    closes both gaps. Mint and build call this with the same place, so the candidate the draw keeps
    is the cover the build scales.
    """
    hx, hy = abs(float(place_half[0])), abs(float(place_half[1]))
    extra = COVER_OVERHANG_M - observe.FLAT_COVER_MARGIN_M     # cover_place adds its own margin back
    if kind in ROUND_COVERS:
        reach = math.hypot(hx, hy) + extra
        return (reach, reach)
    return (hx + extra + COVER_CORNER_PAD, hy + extra + COVER_CORNER_PAD)


def cover_shape(kind, xy, half, yaw=0.):
    """Resting footprint of a flat cover: a disc for the round kinds, its box for the rest."""
    if kind in ROUND_COVERS:
        return S.Disc((float(xy[0]), float(xy[1])), float(max(half)))
    return S.Box((float(xy[0]), float(xy[1])), (float(half[0]), float(half[1])), float(yaw))


def flat_cover_options(kind, place_half, margin=observe.FLAT_COVER_MARGIN_M,
                       max_scale=observe.FLAT_COVER_MAX_SCALE):
    """Every (half, turn) :func:`observe.cover_place` could draw for this place: one per asset
    of ``kind`` that overhangs :func:`cover_half` by ``margin`` within the scale ceiling."""
    out = []
    need = cover_half(kind, place_half)
    for asset in observe.FLAT_COVERS[kind]:
        sx, sy, _ = asset['size']
        for turn in (0., math.pi / 2):
            cx, cy = (sx / 2, sy / 2) if turn == 0. else (sy / 2, sx / 2)
            scale = max(1., (need[0] + margin) / cx, (need[1] + margin) / cy)
            if scale <= max_scale:
                out.append(((cx * scale, cy * scale), turn))
                break
    return out


def flat_cover_feasible(env, place, kind, *, keepouts=(), bounds=None, floor=None, gap=.004):
    """Conservative dry run of :func:`observe.cover_place` in push mode: *every* asset the draw
    could choose must rest clear of ``keepouts`` and pass :func:`observe.push_gate`, so the
    candidate the build keeps builds whatever the build-time rng picks. ``keepouts`` must exclude
    the place itself. The resting-footprint test is what keeps a cover off a second board."""
    options = flat_cover_options(kind, place['half'])
    if not options:
        return False, [f"no {kind} covers a {tuple(round(v, 3) for v in place['half'])} place "
                       f'within scale {observe.FLAT_COVER_MAX_SCALE}']
    keeps = [S.as_shape(k) for k in keepouts]
    reasons = []
    need = cover_half(kind, place['half'])
    for half, turn in options:
        yaw = float(place.get('yaw', 0.)) + turn
        box = cover_shape(kind, place['xy'], half, yaw)
        blocking = sum(1 for k in keeps if S.clearance(box, k) < gap)
        if blocking:
            reasons.append(f'the {kind} would rest on {blocking} other prop(s)')
            continue
        # `need`, not the bare place: the build hands `cover_place` the same inflated half, so the
        # push it gates here is the push the build gates.
        gate = observe.push_gate(env, place['xy'], half, yaw, need,
                                 keepouts=keepouts, bounds=bounds, floor=floor)
        if not gate['passed']:
            reasons.extend(gate['reasons'][:1])
    return (not reasons), reasons


def resolve(env, spec, *, objects, places, keepouts, bounds=None, floor=None):
    """Pick the uncover entry this scene can realise. Returns ``(hidden, notes)``.

    ``keepouts`` maps a prop name to its task-frame shape; the candidate's own footprint is
    dropped from the list handed to the observe gates (a cover always overlaps what it hides).
    Look entries pass through unchanged: :func:`observe.apply_observability` has its own
    occluder-to-annex fallback for those.
    """
    hidden = [dict(entry) for entry in (spec.get('hidden') or [])]
    plan = spec.get('hidden_plan') or {}
    drawn = plan.get('uncover_flavour')
    notes = dict(flavour=drawn, flavour_realised=drawn, flavour_fallback=False, candidates_tried=[])
    index = next((i for i, entry in enumerate(hidden) if entry['mode'] == 'cover'), None)
    if index is None:
        return hidden, notes
    orders = {'object': list(plan.get('object_candidates') or []),
              'place': list(plan.get('place_candidates') or [])}
    order = ('object', 'place') if drawn != 'place' else ('place', 'object')
    for flavour in order:
        for name, kind in orders[flavour]:
            # Only the candidate's own footprint drops out, which is exactly what the build drops
            # when it hands `observe.apply_observability` its keep-outs. Dropping the *drawn*
            # object as well let a push gate pass here and fail there: the drawn object stays on
            # the counter whenever a later candidate wins.
            keeps = [shape for key, shape in keepouts.items() if key != name]
            if flavour == 'object':
                probe = observe.cover_feasible(env, name, kind, keepouts=keeps, floor=floor,
                                               **(objects.get(name) or {}))
                ok, reasons = bool(probe['ok']), list(probe['reasons'])
            else:
                place = places.get(name)
                if place is None:
                    ok, reasons = False, [f'{name} is not a coverable place']
                else:
                    ok, reasons = flat_cover_feasible(env, place, kind, keepouts=keeps, bounds=bounds,
                                                      floor=floor)
            notes['candidates_tried'].append(dict(flavour=flavour, object=name, cover=kind, ok=ok,
                                                  reasons=reasons[:2]))
            if ok:
                hidden[index] = observe.hidden_entry(name, 'cover', kind)
                notes.update(flavour_realised=flavour, flavour_fallback=bool(flavour != drawn),
                             uncover_object=name, uncover_cover=kind)
                return hidden, notes
    notes['flavour_realised'] = None      # apply_observability records the problem and the build fails
    return hidden, notes
