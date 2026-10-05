"""Painted-cube marks: vocabulary, decals, hidden-face proof, deterministic realisation, checker.

Adapted from cube inspection v4 (``active_bench.cube_inspection_marks`` and
``vision_tasks.cube_properties``); the originals are unchanged. Differences:

* the vocabulary is shape x colour ({triangle, circle, square, star, cross} x
  {blue, red, green, black}); one geom per shape per face carries the shape,
  and its colour is written into ``geom_rgba`` at reset (alpha selects it);
* decals are larger (half extent 19 mm on a 26 mm half cube) and the proof
  bound follows them;
* marks are realised deterministically from a frozen spec plus the faces
  proved hidden at reset, with **symmetric hiding**: the visible faces of every
  cube are consistent with both classes under the instance's rule, so nothing
  can be classified from the initial views;
* ``check_symmetric_hiding`` is the explicit checker used by the sampler gate,
  the scene at reset and the tests.

CPU only: no simulator imports.
"""
from collections.abc import Mapping
import math
import xml.etree.ElementTree as ET

import numpy as np

SHAPES = ('triangle', 'circle', 'square', 'star', 'cross')
COLOURS = ('blue', 'red', 'green', 'black')
COLOUR_RGB = {'blue': (.015, .12, .92), 'red': (.85, .07, .05), 'green': (.04, .58, .12), 'black': (.02, .02, .02)}
FACE_NORMALS = {'px': (1, 0, 0), 'nx': (-1, 0, 0), 'py': (0, 1, 0), 'ny': (0, -1, 0), 'pz': (0, 0, 1), 'nz': (0, 0, -1)}
FACE_NAMES = tuple(FACE_NORMALS)
CUBE_NAMES = tuple('cube_' + letter for letter in 'abcdefgh')   # v1 size levels go up to eight cubes
# The order is load-bearing: numpy's bounded ``integers`` takes the high bits, so ``integers(4) >> 1``
# is the ``integers(2)`` the two-kind generator drew. With the two new kinds at the odd indices, every
# seed whose draw still lands on ``has_mark`` or ``exactly_one`` keeps the rule (and the whole spec) it
# had before they were added.
RULE_KINDS = ('has_mark', 'has_shape', 'exactly_one', 'has_colour')
# The mark attributes each kind names. The wording of an instance names exactly the attributes that
# separate its targets from the other cubes in that scene (:func:`rule_minimality_problems`): the colour
# and the shape when both are needed, the shape alone when every colour of it counts, the colour alone
# when every shape of it counts. ``exactly_one`` counts marked faces instead and names no attribute.
RULE_ATTRIBUTES = {'has_mark': ('colour', 'shape'), 'has_shape': ('shape',), 'has_colour': ('colour',)}
ATTRIBUTE_KINDS = tuple(RULE_ATTRIBUTES)
NAMED_ATTRIBUTE_WORD = {'has_mark': 'mark', 'has_shape': 'shape', 'has_colour': 'colour'}

SIDE_M = .052
HALF_SIZE_M = SIDE_M / 2
PAINT_OFFSET_M = .00012          # decal mid-plane above the face
PAINT_HALF_THICKNESS_M = .00004
PAINT_HALF_EXTENT_M = .019       # bounds every decal below, tangentially (7 mm inside the cube edge)
CAMERA_JITTER_M = .03            # faces must stay hidden from this neighbourhood of each camera
MIRROR_RANGE_M = 4.0             # reflective surfaces further than this from the work frame are ignored
MIRROR_MIN_REFLECTANCE = .01     # below this MuJoCo's mirror pass contributes nothing a pixel can see
_ROOT2 = math.sqrt(.5)
# MuJoCo wxyz rotations mapping the decal's local +Z normal to each face.
_FACE_ROTATIONS = {'px': (_ROOT2, 0, _ROOT2, 0), 'nx': (_ROOT2, 0, -_ROOT2, 0),
                   'py': (_ROOT2, -_ROOT2, 0, 0), 'ny': (_ROOT2, _ROOT2, 0, 0),
                   'pz': (1, 0, 0, 0), 'nz': (0, 0, 1, 0)}
# Decal outlines in the face plane (metres). Cross bars: half sizes (x, y).
TRIANGLE_XY = ((-.019, -.014), (.019, -.014), (0., .019))
CIRCLE_RADIUS = .017
SQUARE_HALF = .015
CROSS_BARS = {'cross': (.005, .019), 'cross_horizontal': (.019, .005)}
STAR_OUTER, STAR_INNER = .019, .0078


def mark_text(mark):
    """Public name of a mark, e.g. 'blue triangle'."""
    return f"{mark['colour']} {mark['shape']}"


def rule_text(rule):
    """The rule as it appears in the goal (public)."""
    kind = rule['kind']
    if kind == 'has_mark':
        return f"every cube that has a {rule['colour']} {rule['shape']}"
    if kind == 'has_shape':
        return f"every cube that has a {rule['shape']}"
    if kind == 'has_colour':
        return f"every cube that has a {rule['colour']} mark"
    if kind == 'exactly_one':
        return 'every cube with exactly one marked face'
    raise ValueError(f"unknown rule kind {rule.get('kind')!r}")


def validate_rule(rule):
    problems = []
    if not isinstance(rule, Mapping) or rule.get('kind') not in RULE_KINDS:
        return ['rule kind must be one of ' + ', '.join(RULE_KINDS)]
    named = RULE_ATTRIBUTES.get(rule['kind'], ())
    for attribute, vocabulary in (('shape', SHAPES), ('colour', COLOURS)):
        if attribute in named and rule.get(attribute) not in vocabulary:
            problems.append(f"{rule['kind']} rule needs a {attribute} from the vocabulary")
        if attribute not in named and attribute in rule:
            problems.append(f"{rule['kind']} rule must not name a {attribute}")
    return problems


def _same_mark(a, b):
    return a is not None and b is not None and a['shape'] == b['shape'] and a['colour'] == b['colour']


def matches(mark, rule):
    """Whether one mark carries every attribute the rule names (attribute kinds only)."""
    named = RULE_ATTRIBUTES.get(rule.get('kind'))
    if named is None:
        raise ValueError(f"rule kind {rule.get('kind')!r} names no mark attributes")
    return mark is not None and all(mark.get(attribute) == rule.get(attribute) for attribute in named)


def classify(rule, marks):
    """True when a cube with these marks (face -> mark) satisfies the rule."""
    if rule['kind'] in RULE_ATTRIBUTES:
        return any(matches(m, rule) for m in marks.values())
    if rule['kind'] == 'exactly_one':
        return len(marks) == 1
    raise ValueError(f"unknown rule kind {rule.get('kind')!r}")


def completions(rule, marks, hidden_faces):
    """(can_qualify, can_fail): whether the *visible* faces alone admit each class.

    The visible faces are every face not proved hidden; their marks are public
    knowledge in the initial views. A hidden face may carry any mark or none.
    """
    hidden = set(hidden_faces)
    visible_marks = {f: m for f, m in marks.items() if f not in hidden}
    if rule['kind'] in RULE_ATTRIBUTES:
        named_visible = any(matches(m, rule) for m in visible_marks.values())
        return bool(named_visible or hidden), not named_visible
    if rule['kind'] == 'exactly_one':
        v, h = len(visible_marks), len(hidden)
        return (v == 1 or (v == 0 and h >= 1)), (v >= 2 or v == 0 or h >= 1)
    raise ValueError(f"unknown rule kind {rule.get('kind')!r}")


def check_symmetric_hiding(rule, marks_by_cube, hidden_by_cube, targets):
    """Problems with a realised assignment; empty means symmetric hiding holds.

    ``marks_by_cube``: cube -> {face: {'shape', 'colour'}}; ``hidden_by_cube``:
    cube -> faces proved hidden from every settled camera; ``targets``: the
    cubes that must qualify. Checks, for every cube: the class under the rule
    matches its target flag; the visible faces are consistent with both
    classes; every mark that decides the class lies on a proved hidden face.
    Rule-specific symmetry: an attribute rule never shows a mark it names (no
    visible face carries the named mark, the named shape or the named colour)
    and every cube carries the same number (at least one) of marks on hidden
    faces; ``exactly_one`` shows at most one marked visible face on every cube.
    """
    problems = []
    targets = set(targets)
    hidden_counts = {}
    for name in marks_by_cube:
        marks = marks_by_cube[name]
        hidden = set(hidden_by_cube.get(name, ()))
        if not hidden <= set(FACE_NAMES) or not set(marks) <= set(FACE_NAMES):
            problems.append(f'{name}: unknown face name')
            continue
        for face, mark in marks.items():
            if not isinstance(mark, Mapping) or mark.get('shape') not in SHAPES or mark.get('colour') not in COLOURS:
                problems.append(f'{name}: mark on {face} is not in the vocabulary')
        if not hidden:
            problems.append(f'{name}: no face is proved hidden')
        if classify(rule, marks) != (name in targets):
            problems.append(f'{name}: class under the rule does not match its target flag')
        can_qualify, can_fail = completions(rule, marks, hidden)
        if not (can_qualify and can_fail):
            problems.append(f'{name}: visible faces already decide the class')
        visible_marks = {f: m for f, m in marks.items() if f not in hidden}
        hidden_counts[name] = len(marks) - len(visible_marks)
        if rule['kind'] in RULE_ATTRIBUTES:
            if any(matches(m, rule) for m in visible_marks.values()):
                problems.append(f'{name}: the named {NAMED_ATTRIBUTE_WORD[rule["kind"]]} is on a visible face')
            if hidden_counts[name] == 0:
                problems.append(f'{name}: no mark on a hidden face')
        else:
            if len(visible_marks) > 1:
                problems.append(f'{name}: more than one visible marked face')
    if rule['kind'] in RULE_ATTRIBUTES and len(set(hidden_counts.values())) > 1:
        problems.append(f'hidden mark counts differ across cubes: {hidden_counts}')
    missing = targets - set(marks_by_cube)
    if missing:
        problems.append(f'targets are not cubes: {sorted(missing)}')
    return problems


# ---- minimal and sufficient wording -----------------------------------------
def _attribute_candidates():
    """Every wording the vocabulary offers, as an attribute map; ``{}`` is 'every cube that has a mark'."""
    yield {}
    for shape in SHAPES:
        yield dict(shape=shape)
    for colour in COLOURS:
        yield dict(colour=colour)
    for shape in SHAPES:
        for colour in COLOURS:
            yield dict(shape=shape, colour=colour)


def candidate_text(attributes):
    """How a wording over these attributes reads; the same phrasing :func:`rule_text` uses."""
    if not attributes:
        return 'every cube that has a mark'
    if 'shape' not in attributes:
        return f"every cube that has a {attributes['colour']} mark"
    if 'colour' not in attributes:
        return f"every cube that has a {attributes['shape']}"
    return f"every cube that has a {attributes['colour']} {attributes['shape']}"


def _carries(attributes, marks):
    return any(all(mark.get(k) == v for k, v in attributes.items()) for mark in marks.values())


def _separates(attributes, marks_by_cube, targets):
    return all(_carries(attributes, marks) == (name in targets) for name, marks in marks_by_cube.items())


def rule_minimality_problems(rule, marks_by_cube, targets):
    """Why this wording is not exactly the attributes that separate the targets; empty means it is.

    The user's principle - the rule text names exactly the attributes that separate the targets from
    the other cubes in that scene - is three properties of the realised marks:

    * **sufficient**: every target satisfies the rule and no other cube does;
    * **minimal**: no strictly coarser wording (one that drops a named attribute, down to "every cube
      that has a mark") separates the same cubes, so every attribute the text names is needed;
    * **forced**: no other wording of the vocabulary separates them either, so the text is the only one
      that fits.

    Two cases no assignment can avoid are exempt, and only those. A coarser wording is killed by a
    non-target carrying the attribute it keeps, so with fewer marks on the non-targets than the rule
    names attributes (one non-target with one hidden face, at four cubes and three targets) that many
    coarser wordings may survive. And when the targets carry a single mark that the rule names (one
    target, one hidden face), the finer wording spelling out that mark's other attribute picks out the
    same cube; putting that mark anywhere else would break sufficiency.

    ``exactly_one`` counts marked faces instead of naming attributes, so only sufficiency applies.
    """
    targets = set(targets)
    missing = targets - set(marks_by_cube)
    if missing:
        return [f'targets are not cubes: {sorted(missing)}']
    problems = []
    for name, marks in sorted(marks_by_cube.items()):
        if classify(rule, marks) != (name in targets):
            side = 'a target that does not satisfy it' if name in targets else 'a cube that is not a target'
            problems.append(f'{name}: the rule does not separate the targets ({side})')
    if rule['kind'] not in RULE_ATTRIBUTES:
        return problems
    named = {attribute: rule[attribute] for attribute in RULE_ATTRIBUTES[rule['kind']]}
    deciding = sum(sum(bool(matches(m, rule)) for m in marks_by_cube[n].values()) for n in sorted(targets))
    budget = sum(len(marks) for name, marks in marks_by_cube.items() if name not in targets)
    coarser = []
    for candidate in _attribute_candidates():
        if candidate == named or not _separates(candidate, marks_by_cube, targets):
            continue
        if candidate.items() < named.items():
            coarser.append(candidate_text(candidate))
        elif candidate.items() > named.items() and deciding < 2:
            continue      # a single deciding mark: no assignment can stop its own full wording fitting
        else:
            problems.append(f'"{candidate_text(candidate)}" separates the same cubes as well, so the rule '
                            'is not the only wording that fits them')
    for text in sorted(coarser)[:max(0, len(coarser) - max(0, len(named) - budget))]:
        problems.append(f'"{text}" separates the same cubes, so the rule names an attribute it does not need')
    return problems


def rule_is_minimal_and_sufficient(rule, marks_by_cube, targets):
    """True when the rule names exactly the attributes that separate the targets from the other cubes.

    See :func:`rule_minimality_problems`, which says why when it does not.
    """
    return not rule_minimality_problems(rule, marks_by_cube, targets)


# ---- deterministic realisation ---------------------------------------------
def _rng(*words):
    return np.random.default_rng(np.random.SeedSequence([int(w) for w in words]))


def _random_mark(rng, exclude=None):
    while True:
        mark = dict(shape=str(rng.choice(SHAPES)), colour=str(rng.choice(COLOURS)))
        if exclude is None or not _same_mark(mark, exclude):
            return mark


def _mark(shape, colour):
    return dict(shape=str(shape), colour=str(colour))


def _other(rng, options, value):
    """One option other than this one."""
    return str(rng.choice([option for option in options if option != value]))


def _decoy(rng, rule):
    """A mark the rule does not name; for ``has_mark`` half of them are near misses on one attribute."""
    if rule['kind'] == 'has_mark':
        if rng.random() < .5:
            if rng.random() < .5:
                return _mark(_other(rng, SHAPES, rule['shape']), rule['colour'])
            return _mark(rule['shape'], _other(rng, COLOURS, rule['colour']))
        return _random_mark(rng, exclude=rule)
    while True:
        mark = _random_mark(rng)
        if not matches(mark, rule):
            return mark


def _named_target_marks(rng, rule, count):
    """The marks the targets carry that the rule names.

    For the coarse kinds the attribute the rule leaves out differs between them, so naming it as well
    would not separate the same cubes and the coarse wording is the only one that fits.
    """
    if rule['kind'] == 'has_mark':
        return [_mark(rule['shape'], rule['colour']) for _ in range(count)]
    if rule['kind'] == 'has_shape':
        colours = [str(c) for c in rng.permutation(COLOURS)]
        return [_mark(rule['shape'], colours[index % len(colours)]) for index in range(count)]
    shapes = [str(s) for s in rng.permutation(SHAPES)]
    return [_mark(shapes[index % len(shapes)], rule['colour']) for index in range(count)]


def _required_distractors(rng, rule, named_marks):
    """The marks a non-target must carry for every attribute the rule names to be necessary.

    ``has_mark``: one shares the shape in another colour and one shares the colour in another shape, so
    dropping either attribute stops the wording separating the targets. ``has_shape``: another shape in
    a colour a target uses, so naming that colour separates nothing. ``has_colour``: a shape a target
    uses, in another colour.
    """
    if rule['kind'] == 'has_mark':
        return [_mark(rule['shape'], _other(rng, COLOURS, rule['colour'])),
                _mark(_other(rng, SHAPES, rule['shape']), rule['colour'])]
    borrowed = (named_marks or [_random_mark(rng)])[int(rng.integers(len(named_marks) or 1))]
    if rule['kind'] == 'has_shape':
        return [_mark(_other(rng, SHAPES, rule['shape']), borrowed['colour'])]
    return [_mark(borrowed['shape'], _other(rng, COLOURS, rule['colour']))]


def _realize_attribute_marks(rule, names, targets, hidden, plan, seed):
    """Marks for the three attribute rules, built so the wording is minimal, sufficient and forced.

    Every cube carries the same number of marks on hidden faces (so counting them says nothing) and no
    visible face carries a mark the rule names, so the initial views decide nothing. The targets carry
    the named attributes on hidden faces, one mark each and differing in the attribute a coarse rule
    leaves out; every other mark in the scene - hidden or visible, on a target or not - comes from a pool realised on the
    non-targets, so the only wordings that can separate the targets are the rule and the coarser ones,
    and the pool's required near misses stop those. See :func:`rule_minimality_problems`.
    """
    rng = _rng(seed, 7304)
    per_cube = min(2, min(len(hidden[name]) for name in names))
    non_targets = [name for name in names if name not in targets]
    free = {}
    for name in names:
        visible = [f for f in FACE_NAMES if f not in hidden[name]]
        free[name] = ([str(f) for f in rng.permutation(hidden[name])][:per_cube]
                      + [str(f) for f in rng.permutation(visible)][:min(int(plan[name]), len(visible))])
    marks = {name: {} for name in names}
    named_marks = _named_target_marks(rng, rule, len(targets))
    for index, name in enumerate(targets):
        marks[name][free[name].pop(0)] = named_marks[index]
    slots = sum(len(free[name]) for name in non_targets)
    pool = _required_distractors(rng, rule, named_marks)[:max(1, slots)]
    wanted = min(slots, len(pool) + 3)
    while len(pool) < wanted:
        decoy = _decoy(rng, rule)
        if decoy not in pool:
            pool.append(decoy)
    placed = []
    depths = max((len(free[name]) for name in non_targets), default=0)
    for index, (name, depth) in enumerate([(n, d) for d in range(depths) for n in non_targets
                                           if d < len(free[n])]):
        mark = pool[index] if index < len(pool) else placed[int(rng.integers(len(placed)))]
        marks[name][free[name][depth]] = dict(mark)
        placed.append(mark)
    for name in targets:                      # every face a target has left repeats a non-target's mark
        for face in free[name]:
            marks[name][face] = dict(placed[int(rng.integers(len(placed)))] if placed else _decoy(rng, rule))
    return {name: {f: marks[name][f] for f in FACE_NAMES if f in marks[name]} for name in names}


def sample_visible_plan(rng, rule, cube_names):
    """How many *visible* faces each cube shows marked; drawn independently of its class."""
    upper = 2 if rule['kind'] in RULE_ATTRIBUTES else 1
    return {name: int(rng.integers(0, upper + 1)) for name in cube_names}


def realize_marks(spec, hidden_by_cube):
    """Deterministic face-level marks from the frozen spec and the faces proved hidden at reset.

    Returns ``dict(marks={cube: {face: mark}}, targets=[...], hidden_faces=...,
    hidden_marks_per_cube=...)``. Two calls with the same spec and hidden faces
    give the same result; the RNG streams are keyed by ``spec['mark_seed']``
    and the cube index only.
    """
    rule = spec['rule']
    cubes = spec['cubes']
    names = [c['name'] for c in cubes]
    targets = [c['name'] for c in cubes if c['target']]
    plan = spec['visible_plan']
    hidden = {}
    for name in names:
        faces = tuple(f for f in FACE_NAMES if f in set(hidden_by_cube[name]))
        if not faces:
            raise ValueError(f'{name}: no face is proved hidden, cannot realise marks')
        hidden[name] = faces
    seed = int(spec['mark_seed'])
    marks = {}
    if rule['kind'] in RULE_ATTRIBUTES:
        marks = _realize_attribute_marks(rule, names, targets, hidden, plan, seed)
    else:
        for index, name in enumerate(names):
            rng = _rng(seed, 7302, index)
            visible = [f for f in FACE_NAMES if f not in hidden[name]]
            count = min(int(plan[name]), len(visible), 1)
            cube_marks = {}
            for face in [str(f) for f in rng.permutation(visible)][:count]:
                cube_marks[face] = _random_mark(rng)
            n_hidden = len(hidden[name])
            if name in targets:
                extra = 1 - count
            elif count == 1:
                extra = int(rng.integers(1, n_hidden + 1))
            else:
                options = [0] + list(range(2, n_hidden + 1))
                extra = int(rng.choice(options))
            for face in [str(f) for f in rng.permutation(hidden[name])][:extra]:
                cube_marks[face] = _random_mark(rng)
            marks[name] = {f: cube_marks[f] for f in FACE_NAMES if f in cube_marks}
    return dict(marks=marks, targets=targets, hidden_faces={n: list(hidden[n]) for n in names},
                hidden_marks_per_cube={n: sum(f in hidden[n] for f in marks[n]) for n in names},
                mark_seed=seed, rule=dict(rule),
                basis='Faces proved self-occluded from every settled camera (jittered), never paint alpha')


# ---- decals -----------------------------------------------------------------
def _star_mesh():
    points = []
    for k in range(10):
        angle = math.pi / 2 + k * math.pi / 5
        radius = STAR_OUTER if k % 2 == 0 else STAR_INNER
        points.append((radius * math.cos(angle), radius * math.sin(angle)))
    t = PAINT_HALF_THICKNESS_M
    vertices = [(x, y, -t) for x, y in points] + [(x, y, t) for x, y in points] + [(0., 0., -t), (0., 0., t)]
    bottom_c, top_c = 20, 21
    faces = []
    for k in range(10):
        j = (k + 1) % 10
        faces.append((top_c, 10 + k, 10 + j))          # top fan, counter-clockwise seen from +z
        faces.append((bottom_c, j, k))                 # bottom fan, outward -z
        faces.append((k, j, 10 + j))                   # side quads, outward
        faces.append((k, 10 + j, 10 + k))
    return vertices, faces


def _triangle_mesh():
    t = PAINT_HALF_THICKNESS_M
    vertices = [(x, y, z) for z in (-t, t) for x, y in TRIANGLE_XY]
    faces = [(0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)]
    return vertices, faces


def decal_extents():
    """Tangential half extents of every decal outline; all must stay within PAINT_HALF_EXTENT_M."""
    star_vertices, _ = _star_mesh()
    return {'triangle': max(max(abs(x), abs(y)) for x, y in TRIANGLE_XY), 'circle': CIRCLE_RADIUS,
            'square': SQUARE_HALF, 'cross': max(max(CROSS_BARS[k]) for k in CROSS_BARS),
            'star': max(max(abs(x), abs(y)) for x, y, _ in star_vertices)}


def paint_geom_names(cube_name, face, shape):
    names = [f'{cube_name}_paint_{face}_{shape}']
    if shape == 'cross':
        names.append(f'{cube_name}_paint_{face}_cross_horizontal')
    return names


def add_paintings(obj):
    """Append every shape option on every face of a robosuite BoxObject, all invisible (alpha 0).

    Marks are assigned at reset by writing colour and alpha into ``geom_rgba``;
    the decals are massless and non-colliding, so the cube's physics is the
    native solid box.
    """
    name, body = obj.name, obj.get_obj()
    for mesh_name, (vertices, faces) in (('triangle', _triangle_mesh()), ('star', _star_mesh())):
        ET.SubElement(obj.asset, 'mesh', name=f'{name}_paint_{mesh_name}_mesh',
                      vertex=' '.join(f'{v:.6g}' for point in vertices for v in point),
                      face=' '.join(str(v) for face in faces for v in face))
    t = PAINT_HALF_THICKNESS_M
    for face, normal in FACE_NORMALS.items():
        for shape in (*SHAPES, 'cross_horizontal'):
            attrs = dict(name=f'{name}_paint_{face}_{shape}',
                         pos=' '.join(f'{v * (HALF_SIZE_M + PAINT_OFFSET_M):.6g}' for v in normal),
                         quat=' '.join(f'{v:.9g}' for v in _FACE_ROTATIONS[face]),
                         rgba='.5 .5 .5 0', contype='0', conaffinity='0', group='1', mass='0')
            if shape in ('triangle', 'star'):
                attrs.update(type='mesh', mesh=f'{name}_paint_{shape}_mesh')
            elif shape == 'circle':
                attrs.update(type='cylinder', size=f'{CIRCLE_RADIUS} {t}')
            elif shape == 'square':
                attrs.update(type='box', size=f'{SQUARE_HALF} {SQUARE_HALF} {t}')
            else:
                bar = CROSS_BARS[shape]
                attrs.update(type='box', size=f'{bar[0]} {bar[1]} {t}')
            ET.SubElement(body, 'geom', **attrs)


# ---- hidden-face proof (copied from cube inspection v4, extent parameterised) --
def self_occlusion_report(cube_position, cube_rotation, camera_positions, *,
                          side_m=SIDE_M, paint_half_extent_m=PAINT_HALF_EXTENT_M,
                          paint_offset_m=PAINT_OFFSET_M,
                          paint_half_thickness_m=PAINT_HALF_THICKNESS_M,
                          clearance_m=1e-6):
    """Prove entire finite decals hidden behind their own convex opaque cube.

    Rotation is local-to-world 3x3. Cameras may be an Nx3 sequence or name->XYZ
    map; all supplied cameras must pass.

    For a candidate face, let c be the camera in coordinates where its outward
    normal is +n, H the cube half-size, p_n=H+delta the decal point, and |p_t|<=E.
    If c_n<H, the camera-to-decal segment crosses the opaque face plane first.
    The crossing tangent satisfies
        |q_t| <= [delta*|c_t| + (H-c_n)*E] / [H+delta-c_n].
    The maximum over the finite normal thickness occurs at an endpoint in delta.
    Require both tangent bounds strictly inside H, with a positive margin. This
    excludes grazing views around edges of a nominally back-facing face. The
    bound encloses every actual symbol, so it is conservative. Cameras inside
    or on the cube fail the proof; malformed dimensions/poses are rejected.
    """
    position = np.asarray(cube_position, dtype=float)
    rotation = np.asarray(cube_rotation, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError('finite cube position three-vector required')
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-7)):
        raise ValueError('proper local-to-world rotation matrix required')
    if isinstance(camera_positions, Mapping):
        camera_names = list(camera_positions)
        cameras = np.asarray(list(camera_positions.values()), dtype=float)
    else:
        cameras = np.asarray(camera_positions, dtype=float)
        camera_names = list(range(len(cameras))) if cameras.ndim else []
    if cameras.ndim != 2 or cameras.shape[1:] != (3,) or len(cameras) == 0 or not np.all(np.isfinite(cameras)):
        raise ValueError('one or more finite camera positions required')
    dimensions = np.asarray([side_m, paint_half_extent_m, paint_offset_m, paint_half_thickness_m, clearance_m], dtype=float)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError('positive finite box/decal dimensions and clearance required')
    half = float(side_m) / 2
    extent = float(paint_half_extent_m)
    deltas = [float(paint_offset_m - paint_half_thickness_m), float(paint_offset_m + paint_half_thickness_m)]
    if extent + clearance_m >= half or deltas[0] <= 0 or clearance_m >= half:
        raise ValueError('decal must be inset tangentially and outside the opaque face')
    local_cameras = (cameras - position) @ rotation
    faces = {}
    for face, normal in FACE_NORMALS.items():
        normal = np.asarray(normal, dtype=float)
        axis = int(np.flatnonzero(normal)[0])
        tangents = [a for a in range(3) if a != axis]
        checks = []
        for camera_name, camera in zip(camera_names, local_cameras):
            c_normal = float(camera @ normal)
            behind = half - c_normal
            external = bool(np.any(np.abs(camera) > half + clearance_m))
            crossing_margin = None
            if behind > clearance_m and external:
                bounds = [(delta * np.abs(camera[tangents]) + behind * extent) / (behind + delta) for delta in deltas]
                crossing_margin = float(half - np.max(bounds))
            hidden = bool(crossing_margin is not None and crossing_margin > clearance_m)
            checks.append(dict(camera=camera_name, fully_self_occluded=hidden, camera_external_to_cube=external,
                               behind_face_plane_m=behind, minimum_crossing_edge_margin_m=crossing_margin))
        faces[face] = dict(fully_self_occluded=all(c['fully_self_occluded'] for c in checks), cameras=checks)
    return dict(hidden_faces=[face for face in FACE_NAMES if faces[face]['fully_self_occluded']],
                faces=faces, side_m=float(side_m), paint_half_extent_m=extent,
                paint_normal_extent_m=[half + d for d in deltas], clearance_m=float(clearance_m),
                method='Finite-decal camera-segment intersection inside opaque convex cube; independent of alpha')


def eligible_hidden_faces(cube_position, cube_rotation, camera_positions, **kwargs):
    """Tuple of faces conservatively self-occluded from every supplied camera."""
    return tuple(self_occlusion_report(cube_position, cube_rotation, camera_positions, **kwargs)['hidden_faces'])


def mirrored_cameras(camera_positions, planes):
    """Every camera plus its mirror image in each reflective plane: name -> XYZ.

    MuJoCo draws a specular mirror image of the scene in any geom whose material has reflectance, and a
    mirror turns a camera into a virtual camera on the far side of the plane. A face self-occluded from
    the real cameras can still be painted across those pixels - a black star on the back of a cube showed
    29 and 35 pixels in the two agentview cameras of layout 47 style 8, reflected in a kitchen wall whose
    material carries reflectance 0.1 (measured 2026-09-21). The proof therefore has to hold for the
    virtual cameras too.

    ``planes`` are ``(point, normal)`` pairs in world coordinates; the normal need not be a unit vector.
    A virtual camera behind an opaque wall sees more than the mirror really shows, which is the safe
    direction for a hidden-state rule.
    """
    if not isinstance(camera_positions, Mapping):
        camera_positions = {str(i): p for i, p in enumerate(camera_positions)}
    out = {name: np.asarray(p, dtype=float).tolist() for name, p in camera_positions.items()}
    for index, (point, normal) in enumerate(planes):
        point = np.asarray(point, dtype=float)
        normal = np.asarray(normal, dtype=float)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= 0:
            continue
        normal = normal / length
        for name, position in camera_positions.items():
            position = np.asarray(position, dtype=float)
            out[f'{name}~m{index}'] = (position - 2. * float((position - point) @ normal) * normal).tolist()
    return out


def jittered_cameras(camera_positions, jitter_m=CAMERA_JITTER_M):
    """Each camera plus its six axis-displaced neighbours: name -> XYZ, keyed for the report."""
    if not isinstance(camera_positions, Mapping):
        camera_positions = {str(i): p for i, p in enumerate(camera_positions)}
    out = {}
    for name, position in camera_positions.items():
        position = np.asarray(position, dtype=float)
        out[name] = position.tolist()
        for axis, label in enumerate('xyz'):
            for sign, tag in ((-1., '-'), (1., '+')):
                shifted = position.copy()
                shifted[axis] += sign * jitter_m
                out[f'{name}{tag}{label}'] = shifted.tolist()
    return out
