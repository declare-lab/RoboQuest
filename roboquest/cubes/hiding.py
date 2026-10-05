"""Observability draw for painted cubes (spec 2.2, wave-2 wiring job WIRE-CC).

The level is a registry factor; what it means for one instance is drawn here, at
mint time, from a nuisance stream and written into ``spec['hidden']`` in the
schema :mod:`roboquest.observe` defines (``{'object', 'mode',
'cover'}``). Nothing here touches the structure stream, so the rule, the target
cubes, the split key and the goal text are the same at every level.

Design table (spec 2.2):

``look``
    one or two cubes in the annex or behind an occluder, drawn equally over
    rule-satisfying and non-satisfying cubes (:func:`observe.draw_hidden`).
``uncover``
    one or two cubes under a small cloche or an inverted bowl. The kind is drawn
    per entry. Spec 2.2 also lists the wide cup, and the cup is built and poured
    by the task, but it cannot *hide* a cube: it is open at the top. Measured on
    layout 8 style 13 (2026-09-21, registry candidate 6f83932c97c651af): a cube
    standing in the cup showed 708 pixels of ``robot0_eye_in_hand``, which looks
    straight down into the mouth from the reset stance, and 58 pixels of
    ``robot0_agentview_left`` over the near rim. Every one of the six mini-registry
    candidates that drew a cup failed G3 for exactly this; no other kind failed
    more often than a third of the time. The cup is therefore listed in
    :data:`OPEN_TOPPED_COVERS` and kept out of the hiding draw.
``look+uncover``
    one look entry and one uncover entry, on different cubes (dev only).

Feasibility at mint time: the cover a cube carries is reserved on the counter by
scattering that cube with the cover's footprint disc instead of its own
(:func:`painted_cubes.cube_footprints`), so a drawn cover always fits where its
cube stands and the seed is not rejected at build time for a cover that cannot
be placed. A kind whose interior the cube does not fit is dropped from the pool
before the draw and the fallback is recorded in the entry.

The face-occluding plates and bowls of the task's own nuisance (spec 2.2: they
hide faces, not cubes) are drawn from the same pool of cubes, so a cube named
here is removed from that draw: a plate leaning on a cube that stands under a
cloche or in the annex would be a second, unrecorded hiding mechanism.
"""
import math

from roboquest import observe as OB

# 52 mm cube (marks.SIDE_M): the bounding disc and height every cover is fitted against.
CUBE_RADIUS = .037
CUBE_HEIGHT = .052
LEVELS = ('visible', 'look', 'uncover', 'look+uncover')
LOOK_MODES = ('annex', 'occluder')
# Covers that close over a cube. An open-topped cover hides nothing from a camera that can look into
# its mouth, and at the reset stance the wrist camera does exactly that (see the module docstring), so
# the cup is a manipulation challenge, not a hiding mechanism.
OPEN_TOPPED_COVERS = ('wide_cup',)
COVER_KINDS = ('cloche_small', 'bowl')
HIDDEN_COUNTS = (1, 2)          # how many cubes one level hides
CUP_FOOTPRINT_M = OB.CUP_INNER_RADIUS + .004 + .035 + .006   # cover_feasible's cup clearance disc

# Where a covered cube may stand. A cover has to be lifted off, and ``observe.cover_feasible`` gates the
# knob (or the cup's handle, or the bowl's slide path) against the top-down reach envelope from the
# robot's reset stance: at most ``GRASP_REACH_M`` ahead of the base frame. Measured on the built kitchens
# (layouts 4 and 9, style 4/1, v1 tip), that stance stands at task-frame y = -0.475 whatever the layout,
# because ``place_work_frame`` puts the frame on the fixture's front edge and RoboCasa spawns the base a
# fixed distance in front of it; a cube at frame y is therefore ``y + .475`` m ahead of the base. Cubes
# drawn for a cover are scattered inside this band so that the cover is reachable where its cube stands
# (a cube at the back of the frame is still fine for the task itself: the bare cube is picked, not lifted
# out from under a cloche). The band is the same at every level, so it tells a policy nothing.
BASE_FRAME_Y_M = -.475
COVER_REACH_MARGIN_M = .03
COVER_MAX_Y_M = round(OB.GRASP_REACH_M + BASE_FRAME_Y_M - COVER_REACH_MARGIN_M, 4)      # .195


def cover_fits(kind, radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """Does a cube fit inside this cover at all? Pure geometry, no scene (mint time)."""
    if kind in ('cloche_small', 'cloche_large'):
        spec = OB.CLOCHE_SIZES['small' if kind == 'cloche_small' else 'large']
        return (radius + OB.COVER_CLEARANCE_M <= spec['radius'] - OB.CLOCHE_WALL
                and height + OB.COVER_CLEARANCE_M <= spec['depth'])
    if kind == 'bowl':
        return any(OB.bowl_scale_for(c, radius, height) is not None for c in OB.BOWL_COVERS)
    if kind == 'wide_cup':
        return (radius + OB.CUP_CLEARANCE_M <= OB.CUP_INNER_RADIUS
                and height <= OB.CUP_HEIGHT - OB.CUP_FLOOR - .004)
    raise ValueError(f'unknown cover kind {kind!r}')


def cover_footprint_radius(kind, radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """The disc a cover needs on the counter around its cube: what the scatter reserves.

    The bowl is the widest, and :func:`observe.cover_object` picks whichever of the two bowl covers
    comes first in its own permutation, so the reservation is the larger of the two fitted radii.
    """
    if kind in ('cloche_small', 'cloche_large'):
        return float(OB.CLOCHE_SIZES['small' if kind == 'cloche_small' else 'large']['radius'])
    if kind == 'wide_cup':
        return float(CUP_FOOTPRINT_M)
    if kind == 'bowl':
        radii = [c['radius'] * s for c in OB.BOWL_COVERS
                 for s in (OB.bowl_scale_for(c, radius, height),) if s is not None]
        if not radii:
            raise ValueError('no bowl cover fits')
        return float(max(radii))
    raise ValueError(f'unknown cover kind {kind!r}')


def cover_pool(radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """The cover kinds that hide a cube of this size, in a fixed order (never an open-topped one)."""
    return tuple(kind for kind in COVER_KINDS
                 if kind not in OPEN_TOPPED_COVERS and cover_fits(kind, radius, height))


def draw_hidden(rng, level, names, qualifying, radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """``spec['hidden']`` for one instance: the entries of the level's design table.

    ``rng`` is a nuisance stream (never the structure stream), ``names`` the cubes in spec order and
    ``qualifying`` the cubes that satisfy the rule, so the draw is equal over the two classes
    (:func:`observe.draw_hidden`). Returns (entries, fallbacks): a fallback records a cover kind that
    was drawn and replaced because the cube does not fit inside it.
    """
    if level not in LEVELS:
        raise ValueError(f'observability {level!r} is not one of {LEVELS}')
    if level == 'visible':
        return [], []
    pool = cover_pool(radius, height)
    if not pool:
        raise ValueError('no cover kind fits a task cube')
    fallbacks = []

    def look_entry(name):
        return OB.hidden_entry(name, str(LOOK_MODES[int(rng.integers(len(LOOK_MODES)))]))

    def cover_entry(name):
        kind = str(COVER_KINDS[int(rng.integers(len(COVER_KINDS)))])
        if kind not in pool:
            fallbacks.append(dict(object=name, requested=kind, realised=pool[int(rng.integers(len(pool)))]))
            kind = fallbacks[-1]['realised']
        return OB.hidden_entry(name, 'cover', kind)

    if level == 'look+uncover':
        drawn = OB.draw_hidden(rng, names, qualifying, 2)
        if len(drawn) < 2:
            raise ValueError('look+uncover needs two cubes')
        entries = [look_entry(drawn[0]), cover_entry(drawn[1])]
    else:
        count = int(HIDDEN_COUNTS[int(rng.integers(len(HIDDEN_COUNTS)))])
        drawn = OB.draw_hidden(rng, names, qualifying, min(count, len(names)))
        entries = [(look_entry if level == 'look' else cover_entry)(name) for name in drawn]
    return entries, fallbacks


def cover_zone(zone, y_max=COVER_MAX_Y_M):
    """The part of the cube zone whose covers the robot can reach from its reset stance."""
    return OB.S.Rect(zone.x0, zone.x1, zone.y0, min(zone.y1, float(y_max)))


def scatter_zones(spec, zone):
    """One scatter region per cube: the whole zone, or the reachable band for a cube under a cover."""
    covers = cover_entries(spec)
    return [cover_zone(zone) if cube['name'] in covers else zone for cube in spec['cubes']]


def cover_order(spec, name, radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """The cover kinds to walk for this cube: the drawn one first, then the rest of the pool.

    The drawn kind is the one the scatter reserved room for, so it is walked first and, when the walk can
    prove it, used; the rest are the fallback for what the drawn kind cannot do here. Once the walk has run
    the entry holds the kind it kept rather than the kind that was drawn, so the drawn kind is read back
    from the plan when there is one: that makes :func:`walk_covers` idempotent, which is what lets
    ``validate_spec`` re-derive the plan from the spec it is checking.
    """
    name = str(name)
    drawn = {str(row['object']): str(row['drawn'])
             for row in ((spec.get('hidden_plan') or {}).get('covers') or ())
             if isinstance(row, dict) and row.get('drawn')}.get(name) or cover_entries(spec)[name]
    return (drawn,) + tuple(k for k in cover_pool(radius, height) if k != drawn)


# ---- the mint-time cover walk (job CUBES-REACH) ----------------------------------------------------
# Reserving the cover's footprint is not enough: the build also has to *take the cover off*, and
# ``observe.cover_feasible`` gates that with :func:`observe.reach_gate` (the cloche's knob) or
# :func:`observe.slide_gate` (the bowl slid to a counter edge and lifted at the rim). Until this walk
# the draw knew nothing about those gates, so a cover could be drawn where it could never be lifted and
# the instance was lost at G1 - measured on the WIRE-CC mini registry (2026-09-21): five of the six
# painted_cubes uncover build failures were "cover placed but its reach gate failed".
#
# The walk below applies the same numbers to the same geometry at mint time, over everything the spec
# knows (the cube's spot, the other props). Three facts belong to the kitchen, not to the spec, and are
# recorded in the plan instead of guessed:
BUILD_TIME_UNKNOWNS = (
    'the decor standing on the work surface (observe.decor_shapes_on_work)',
    'the fixtures overhead, so the headroom above the cover (headroom.overhead_boxes)',
    'the floor beside the counter, so the stance the base slides to (observe.stance_gate)',
)
# ... and a fourth for the bowl only: the work surface reaches beyond the task frame, so a bowl the walk
# cannot slide off the frame's front edge may still reach a counter end the build can see.
BOWL_EDGE_UNKNOWN = 'the work-surface edges beyond the task frame (a counter end the bowl could slide to)'


def cover_ahead(xy):
    """How far ahead of the reset stance a task-frame point lies (the ``ahead`` of ``observe.reach_gate``).

    Measured on the built kitchens at the v1 tip (layouts 8, 9 and 47): the base stands at task-frame
    (0, ``BASE_FRAME_Y_M``) with its heading along +y, whatever the layout, so ``ahead`` is one subtraction.
    """
    return float(xy[1]) - BASE_FRAME_Y_M


def footprint_clashes(xy, cover_radius, keepouts, gap=OB.OCCLUDER_GAP_M):
    """Mint-time half of :func:`observe.cover_footprint_clear`: the same disc against the same task-frame
    footprints, minus the scene decor the generator cannot see."""
    disc = OB.S.Disc((float(xy[0]), float(xy[1])), float(cover_radius))
    return sum(1 for k in keepouts if OB.S.clearance(disc, OB.S.as_shape(k)) < gap)


def bowl_front_slide(xy, cover_radius, keepouts, frame_half, radius=CUBE_RADIUS):
    """:func:`observe.slide_gate`'s front-edge option, measured against the task frame.

    The front edge is the only one mint can place: the work surface is at least as large as the frame and
    (layouts 8, 9 and 47 at the v1 tip) its front edge sits 5 mm in front of the frame's, while its ends
    are wherever the counter ends. Returns None when the slide passes every gate, else the reason it does
    not. ``cover_radius`` must be the *widest* bowl the build could pick: :func:`observe.cover_object`
    draws its asset from the build rng, so a candidate is only safe when the worst asset is safe.
    """
    x, y = float(xy[0]), float(xy[1])
    half_x, half_y = float(frame_half[0]), float(frame_half[1])
    slide = (y + half_y) - cover_radius / 3
    if slide <= 0.:
        return 'the bowl already overhangs the front edge'
    if slide > OB.MAX_SLIDE_PATH_M:
        return f'a slide of {slide:.2f} m to the front edge exceeds {OB.MAX_SLIDE_PATH_M:.2f} m'
    strip = OB.S.Box((x, y - slide / 2), (cover_radius + OB.COVER_EDGE_MARGIN_M,
                                          slide / 2 + cover_radius + OB.COVER_EDGE_MARGIN_M), 0.)
    blocking = sum(1 for k in keepouts if OB.S.clearance(strip, OB.S.as_shape(k)) < 0.)
    if blocking:
        return f'the slide path to the front edge is blocked by {blocking} prop(s)'
    if min(x + half_x, half_x - x) < cover_radius + OB.COVER_EDGE_MARGIN_M:
        return 'the bowl stands too close to a side edge of the frame'
    rim_ahead = cover_ahead((x, y - slide - cover_radius + .01))
    if rim_ahead < OB.REACH_MIN_AHEAD_M:
        return (f'the rim beyond the front edge is {rim_ahead:.2f} m ahead of the base, inside the '
                f'{OB.REACH_MIN_AHEAD_M:.2f} m minimum')
    if rim_ahead > OB.GRASP_REACH_M:
        return f'the rim beyond the front edge is {rim_ahead:.2f} m ahead, beyond {OB.GRASP_REACH_M:.2f} m'
    return None


def cover_check(kind, xy, *, keepouts=(), frame_half, radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """Would the build be able to lift this cover off a cube standing at ``xy``? Mint time, task frame.

    Returns ``dict(kind, verdict, reasons, build_time)``. ``verdict`` is ``'ok'`` when every gate mint can
    apply passes, ``'no'`` when one fails and no build-time fact can rescue it (that candidate is dropped),
    and ``'unknown'`` when the answer turns on something only the kitchen knows (the candidate is kept, but
    behind the proved ones). ``keepouts`` are the task-frame shapes of the other props, the candidate's own
    cube already dropped, exactly as ``painted_cubes.hiding_shapes`` hands them to the build.
    """
    out = dict(kind=str(kind), verdict='ok', reasons=[], build_time=list(BUILD_TIME_UNKNOWNS))
    if not cover_fits(kind, radius, height):
        return dict(out, verdict='no', reasons=[f'a cube does not fit inside the {kind}'])
    clashes = footprint_clashes(xy, cover_footprint_radius(kind, radius, height), keepouts)
    if clashes:
        return dict(out, verdict='no', reasons=[f'the {kind} footprint clashes with {clashes} prop(s)'])
    if kind == 'bowl':
        # cover_object picks one of the fitted bowls from the build rng, so the walk gates the widest.
        cover_radius = cover_footprint_radius('bowl', radius, height)
        inside_after = cover_radius / 3 + OB.BOWL_INNER_FRACTION * cover_radius - 2 * float(radius)
        if inside_after < OB.COVER_EDGE_MARGIN_M:
            return dict(out, verdict='no',
                        reasons=[f'the cube would overhang after the slide ({inside_after:+.3f} m)'])
        out['build_time'] = list(BUILD_TIME_UNKNOWNS) + [BOWL_EDGE_UNKNOWN]
        blocked = bowl_front_slide(xy, cover_radius, keepouts, frame_half, radius)
        if blocked is not None:
            out.update(verdict='unknown', reasons=[f'front edge: {blocked}', BOWL_EDGE_UNKNOWN])
        return out
    ahead = cover_ahead(xy)                       # the cloche knob (and the cup handle) stand over the cube
    if ahead < OB.REACH_MIN_AHEAD_M:
        out.update(verdict='no', reasons=[f'the knob is {ahead:.2f} m ahead of the base, inside the '
                                          f'{OB.REACH_MIN_AHEAD_M:.2f} m minimum'])
    elif ahead > OB.GRASP_REACH_M:
        out.update(verdict='no', reasons=[f'the knob is {ahead:.2f} m ahead, beyond {OB.GRASP_REACH_M:.2f} m'])
    return out


def walk_covers(spec, shapes, frame_half, radius=CUBE_RADIUS, height=CUBE_HEIGHT):
    """Gate every cover candidate the way the build will; returns ``(hidden, plan)``.

    ``shapes`` maps a prop name to its task-frame footprint (:func:`painted_cubes.hiding_shapes`); the
    candidate's own cube is dropped from the list handed to the gates, because a cover always stands on
    the disc the scatter reserved for it. The order per entry stays the one :func:`cover_order` already
    defined - the drawn kind, then the rest of the pool - with the candidates the walk *disproved* dropped
    and the ones whose answer needs the kitchen kept, in place, with the reason recorded. The entry the
    spec keeps is the first surviving candidate and the rest are the fallback the build walks
    (:meth:`painted_cubes.PaintedCubes.feasible_cover_kinds`).

    Pure: the same spec always gives the same plan, which is what lets ``validate_spec`` re-derive it.
    """
    entries = [dict(entry) for entry in (spec.get('hidden') or ())]
    plan = dict(covers=[], build_time=list(BUILD_TIME_UNKNOWNS) + [BOWL_EDGE_UNKNOWN])
    places = {str(cube['name']): cube for cube in spec.get('cubes') or ()}
    for index, entry in enumerate(entries):
        if entry.get('mode') != 'cover':
            continue
        name = str(entry['object'])
        keeps = [shape for key, shape in shapes.items() if key != name]
        xy = places[name]['xy']
        order = cover_order(spec, name, radius, height)
        candidates, unproven, rejected = [], [], []
        for kind in order:
            check = cover_check(kind, xy, keepouts=keeps, frame_half=frame_half, radius=radius, height=height)
            row = dict(object=name, cover=kind, reasons=list(check['reasons']))
            if check['verdict'] == 'no':
                rejected.append(row)
                continue
            if check['verdict'] == 'unknown':
                unproven.append(row)
            candidates.append([name, kind])
        plan['covers'].append(dict(object=name, drawn=str(order[0]), candidates=candidates,
                                   unproven=unproven, rejected=rejected))
        if candidates:
            entries[index] = OB.hidden_entry(candidates[0][0], 'cover', candidates[0][1])
    return entries, plan


def hidden_objects(spec):
    """The cubes ``spec['hidden']`` names, in entry order."""
    return [str(entry['object']) for entry in spec.get('hidden') or ()]


def cover_entries(spec):
    """``{cube: kind}`` for the cubes that stand under a cover."""
    return {str(e['object']): str(e['cover']) for e in spec.get('hidden') or () if e['mode'] == 'cover'}


def look_entries(spec):
    """``{cube: mode}`` for the cubes hidden by distance or by an occluder."""
    return {str(e['object']): str(e['mode']) for e in spec.get('hidden') or () if e['mode'] in LOOK_MODES}


def free_band(surface, frame_half, margin=.02, need=.30):
    """The free area of the work surface beyond the task frame, as one task-frame rectangle.

    On a surface larger than the frame the strip beyond it hosts the look-level
    occluders and the objects they hide, so bigger tables get more clutter. The widest of the back,
    left and right strips that can hold ``need`` metres in both directions wins; None when the
    surface is no bigger than the frame.
    """
    half_x, half_y = float(frame_half[0]) + margin, float(frame_half[1]) + margin
    bands = [OB.S.Rect(surface.x0, surface.x1, half_y, surface.y1),
             OB.S.Rect(surface.x0, -half_x, surface.y0, surface.y1),
             OB.S.Rect(half_x, surface.x1, surface.y0, surface.y1)]
    usable = [b for b in bands if not b.empty and b.x1 - b.x0 >= need and b.y1 - b.y0 >= need]
    if not usable:
        return None
    return max(usable, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0))


def frame_keepouts(spec, cubes_xy_yaw, cube_half, bin_shapes, occluder_discs):
    """Task-frame keep-outs of every prop for :func:`observe.apply_observability`.

    Every cube at its realised yaw, both bins, Submit and the face-occluding plates and bowls: the
    occluder search and the cover footprints must clear all of them.
    """
    shapes = [OB.S.shape_at(cube_half, xy, yaw) for xy, yaw in cubes_xy_yaw]
    shapes.extend(bin_shapes)
    shapes.extend(occluder_discs)
    return shapes


def cover_aside_spot(rng, surface, keepouts, radius, gap=.015, tries=200):
    """A free spot on the work surface for a cover the teleport certificate moves aside.

    Task-frame; never off the surface, never on a prop (``keepouts`` are task-frame shapes of the
    cubes, the bins, Submit and the other covers). Returns (x, y) or None.
    """
    region = OB.S.Rect(surface.x0 + radius, surface.x1 - radius, surface.y0 + radius, surface.y1 - radius)
    if region.empty:
        return None
    for _ in range(int(tries)):
        x = float(rng.uniform(region.x0, region.x1))
        y = float(rng.uniform(region.y0, region.y1))
        disc = OB.S.Disc((x, y), float(radius))
        if all(OB.S.clearance(disc, k) >= gap for k in keepouts):
            return x, y
    return None


def flicker_tolerant_diff(first, second):
    """Pixels that differ in both of two rendered pairs (the flicker rule of visible_in_cameras).

    ``first`` and ``second`` are ``{camera: (before, after)}`` image pairs; a pixel counts as changed
    only when it changed in both passes, so a single flickering pixel (observed once in the LOOK job's
    renders) does not abort a valid instance.
    """
    changed = {}
    for camera, (before_a, after_a) in first.items():
        before_b, after_b = second[camera]
        a = (before_a != after_a).any(axis=2)
        b = (before_b != after_b).any(axis=2)
        changed[camera] = int((a & b).sum())
    return changed


def unstable_pixels(first, second):
    """Pixels that differ between two renders of the same state, per camera."""
    return {camera: int((first[camera] != second[camera]).any(axis=2).sum()) for camera in first}


def annex_radius(radius=CUBE_RADIUS, cover=None):
    """Bounding disc an annexed cube needs (a cover would travel with it; none does today)."""
    return float(radius if cover is None else max(radius, cover_footprint_radius(cover)))


def entry_summary(record):
    """One line per realised entry for the build report and the handoff table."""
    out = []
    for item in record.get('realised', ()):
        out.append(f"{item['object']}:{item.get('realised')}"
                   + (f"({item['cover']})" if item.get('cover') else ''))
    return sorted(out)


def degrees(value):
    return round(math.degrees(float(value)), 2)
