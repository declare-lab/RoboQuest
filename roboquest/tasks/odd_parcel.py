"""Odd Parcel (tasks-v0 task 8): find the parcels that weigh differently with a two-pan balance.

``parcel_count`` visually identical parcels (4, 6 or 8) are **scattered** over the
free area in front of and beside the balance, at random yaws and never in rows;
a few of them start stacked two high (random layout variation at any size level).
``odd_count`` of them (1, 2 or 3, nested in the size level) deviate from the 250 g
standard, each by 120 g or 200 g, up or down. The only instrument is the
purpose-built balance at the back of the frame, which itself starts at a jittered
position and yaw (:mod:`roboquest.balance`).

v1 replaces the flat tray zone with a **column answer box** one parcel wide
(spec 4.2): the interior is the parcel footprint plus 1.5 cm per side and the
walls stand 1 cm above one parcel, so parcels stack in insertion order and the
bottom one has to be pinched out between the walls. Success: exactly the odd
parcels in the box, released and settled, and the balance within 2 cm of its reset
pose, at the first physical Submit press. A parcel left on a pan is reported
(``flags``) but does not fail the episode: the goal never asks to clear the pans.

The structure class is the pair (direction pattern, magnitude pattern), drawn
balanced from the ``structure`` stream and encoded in the split key; mixed
directions and different magnitudes are the hard classes, because there the odd
parcels no longer confirm each other on the balance.

Parcels are robosuite ``BoxObject``s (7 x 10 x 5 cm) sharing one RoboCasa
texture and one tint; only the collision geom's density differs, so nothing
visual or in the goal text distinguishes them.

Observability (spec 2.2, wired in wave 2): the factor ``observability`` has the
levels ``visible``, ``look``, ``uncover`` and the dev-only ``look+uncover``.
``spec['hidden']`` (:func:`draw_observability`, drawn from the ``poses`` stream
after the layout, never from ``structure``) names the parcels hidden at reset:
``look`` puts one or two parcels in an annex region or behind a tall occluder,
``uncover`` covers one parcel with a small cloche or an inverted bowl (hidden
object flavour) or lays a plate on one balance pan (covered place flavour, lifted
off by its rim). :mod:`roboquest.observe` realises the entries at the
end of :meth:`OddParcel.build_task`; a scene that does not realise every entry
refuses to build, so the seed is rejected at G1. The goal text, the structure
class and the hold-out keys never depend on the level.
"""
from copy import deepcopy
import itertools
import math

import mujoco
import numpy as np
from robocasa.models.fixtures import Fixture
from robocasa.models.fixtures.fixture_utils import fixture_is_type
import robocasa.models.fixtures.fixture_utils as FixtureUtils
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial

from roboquest import observe as O
from roboquest import scatter as S
from roboquest.balance import DEFAULT_DESIGN, ROBOCASA_TEXTURES, add_balance, answer_box, interior_xy, sample_textures, textures_are_known, tilt_for_mass_difference, with_textures
from roboquest.balance.builder import GRAVITY
from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz
from roboquest.harness.contract import CAMERAS
from roboquest.base.scene import VisionKitchenScene
from roboquest.base.scene_contracts import all_resting

CONTRACT_VERSION = 'odd-parcel-kitchen-v3'   # v3: balance tolerance on the base, progress capped below 1 until success
GENERATOR_VERSION = 'v4'                     # v3: observability axis (spec 2.2), hidden entries from the poses
                                             # stream; v4: the generator no longer draws the pan-plate cover -
                                             # the uncover level keeps the cloche and the inverted bowl over a
                                             # parcel (the 80 % certificate rule)
# Goal framing (spec 1.3 / contract C1): the scope sentence carries the count only when the
# registry's `wording` stream says so; the middle sentence states the balance rule.
GOAL_NOUN = 'parcels'
GOAL_SCOPE = 'There are {noun} on the counters in this kitchen.'
GOAL_SCOPE_COUNT = 'There are {count} {noun} on the counters in this kitchen.'
GOAL_MIDDLE = ('Most of these parcels weigh the same; fewer than half weigh differently. Put every parcel '
               'that weighs differently in the box and nothing else, leave the balance where it stands, '
               'then press Submit.')
GOAL_CLUTTER = 'Other objects on the counter are not part of the task; you may move them if needed.'

PARCEL_HALF = (.035, .05, .025)        # 7 cm along the frame's x, 10 cm along y, 5 cm tall
BASE_MASS_KG = .25
# Two deviation magnitudes, so "same" and "different" magnitude patterns exist. Both tilt the
# empty-against-one-parcel beam well past the 5 deg readability floor and stay under the 12 deg
# stop, so a magnitude is readable on its own: 120 g -> 6.7 deg, 200 g -> 11.2 deg.
DELTA_KG = {'s': .12, 'b': .20}
DELTA_KEYS = ('s', 'b')
DIRECTION_PATTERNS = ('H', 'L', 'HL')   # all heavy, all light, mixed
MAGNITUDE_PATTERNS = ('same', 'diff')
ODD_COUNT = {4: 1, 6: 2, 8: 3}          # nested size: odd_count is derived, never a factor
# Appearance (spec 4.9): one texture and one tint per instance, shared by every parcel, drawn
# from the `materials` stream. 21 textures x 6 tints, well beyond the four v0 textures.
PARCEL_TEXTURES = ('flat/cream.png', 'flat/warm_white.png', 'flat/light_gray.png', 'flat/gray.png',
                   'flat/blue_gray.png', 'flat/light_green.png', 'ceramic.png', 'cream-plaster.png',
                   'wood/bamboo.png', 'wood/wood_grain_1.png', 'wood/wood_grain_2.png',
                   'wood/wood_grain_3.png', 'wood/light_wood_planks.png', 'wood/light_wood_planks_2.png',
                   'wood/warm_wood_grain.png', 'wood/warm_wood_planks.png', 'wood/walnut_wood_grain.png',
                   'wood/gray_wood_grain.png', 'wood/greenheart_wood_grain.png', 'wood/wood_planks.png',
                   'wood/varnished_wood_planks.png')
PARCEL_TINTS = ((1., 1., 1., 1.), (.93, .90, .84, 1.), (.86, .88, .93, 1.), (.97, .92, .86, 1.),
                (.88, .92, .86, 1.), (.80, .80, .82, 1.))
# Task-frame layout (metres): parcels scattered in front of and beside the balance, the balance at
# the back left, the answer box and Submit to the right. The footprint stays 1.20 x 0.60 m (0.63 m
# with the front margin, inside RoboCasa's 0.65 m counters): the column box is far smaller than the
# v0 tray zone, so eight parcels have more free area than six had in v0, not less.
BALANCE_LOCAL = (-.22, .11)
BALANCE_JITTER = (.05, .05)
BALANCE_JITTER_YAW = math.radians(10.)
PARCEL_REGION_BACK_Y = .01     # parcels stay in front of / beside the balance, never behind it
PARCEL_YAW_RANGE = (-math.pi, math.pi)
BOX_LOCAL = (.40, .02)
SUBMIT_LOCAL = (.40, -.22)
SUBMIT_HALF = (.048, .043)
MIN_PARCEL_GAP = .02           # clear space between two parcels
PARCEL_KEEPOUT_GAP = .02       # clear space from the balance, the answer box or Submit
STACK_OFFSET_M = .005          # how far a stacked parcel may sit off its base
STACK_OFFSET_YAW = math.radians(5.)
PAN_SLOTS_X = (-.08, 0., .08)
BOX_FIT_TOL = .005             # slack on the interior column when deciding "in the box"
BOX_SETTLE_TICKS = 25          # C3: physics between two placements in the answer box
BOX_PLACE_ATTEMPTS = 3         # a parcel stacked above the wall top can bounce out once
BOX_CORNERS = np.array(list(itertools.product((-1., 1.), repeat=3)), float)
BALANCE_RESET_TOL_M = .02      # further displacement is a wrong submit
BUDGET_TICKS = 12000           # provisional (C1): no oracle certificate for this task yet
ALMOST = .99                   # progress reading when the right parcels are boxed but the predicate still fails
# Observability (spec 2.2): the levels, the design table's draws and the mint-time cover geometry.
OBSERVABILITY_LEVELS = ('visible', 'look', 'uncover', O.DEV_ONLY_LEVEL)
LOOK_COUNTS = (1, 2)                      # parcels hidden at the look level
LOOK_MODES = ('annex', 'occluder')        # per hidden parcel, 50/50; the occluder falls back to the annex at build
UNCOVER_FLAVOURS = ('object', 'place')    # drawn 50/50 and recorded, but 'place' is never realised at v4: it
                                          # always falls back to an object cover. The draw
                                          # stays in the stream so the instances that never carried a plate
                                          # keep their v3 ids.
OBJECT_COVER_KINDS = ('cloche_small', 'bowl')
PAN_PLACES = ('pan_left', 'pan_right')
PLACE_COVER_KIND = 'plate'
# Pan-plate set-down feasibility (SKILLS-3 and PARCEL-PANPLATE certificates, 2026-09-22).
# The skill takes the plate off the pan everywhere, and what decides the outcome is the carry: every certified
# pass carried 0.146-0.186 m, every shear came at 0.24-0.26 m. Drawing the cover only where the plate has a
# patch that close still certified 6 of 15 re-minted instances, under the 80 % the rule asks for, so
# the generator no longer draws it at all. The check survives as a CPU gate on a hand-written plate entry: the
# same search the skill runs (uncover.free_spots on plan footprints, the reach band ahead of the base).
PLATE_RADIUS_M = max(max(c['size'][0], c['size'][1]) for c in O.FLAT_COVERS[PLACE_COVER_KIND]) / 2
PLATE_CARRY_MAX_M = .20        # the longest certified carry was 0.186 m, the shortest shear 0.262 m: 0.20 with margin
PLATE_BASE_STANDOFF_M = O.BASE_AHEAD_M   # the reset base spawns with its front at the surface edge, this far ahead
PLATE_SPOT_LATERAL_M = .30     # ... and the set-down spot stands within this much of it across (uncover's own filter)
PARCEL_RADIUS = math.hypot(PARCEL_HALF[0], PARCEL_HALF[1])   # bounding disc of a parcel, 0.061 m
COVER_GAP = O.OCCLUDER_GAP_M              # a cover's footprint stays this clear of the other props
RELOCATE_MARGIN = .06                     # relocation strips for occluded parcels keep this far from the frame and the edges
RELOCATE_MIN_SIDE = .25                   # the smallest strip worth relocating into
OBSERVE_TASK_SALT = 0x0DD9                # third word of the build-time observability RNG seed


def max_stacks(parcel_count):
    """How many parcels may start on top of another one at this size level."""
    return int(parcel_count) // 4


def frame_rect(footprint):
    """The whole task footprint as a scatter rectangle."""
    return S.rect((0., 0.), (footprint[0] / 2, footprint[1] / 2))


def parcel_region(footprint):
    """Where the parcels are scattered: the free band in front of and beside the balance."""
    half = (footprint[0] / 2, footprint[1] / 2)
    return S.Rect(-half[0], half[0], -half[1], PARCEL_REGION_BACK_Y)


def balance_extent(design=DEFAULT_DESIGN):
    """Half sizes of the balance's own bounding box (pan tips to pan tips, hanger depth)."""
    return (design.half_width, design.hanger_back + design.rod_radius)


def box_outer_half(parcel_half=PARCEL_HALF):
    """Half sizes of the answer box's outer shell, from the parcel it has to hold."""
    from roboquest.balance.answer_box import WALL_THICKNESS
    inner = interior_xy((2 * parcel_half[0], 2 * parcel_half[1]))
    return (inner[0] / 2 + WALL_THICKNESS, inner[1] / 2 + WALL_THICKNESS)


def balance_shapes(balance, design=DEFAULT_DESIGN):
    """The balance's keep-out boxes - both pans and the base plate - at its jittered pose.

    ``balance`` is the spec's ``{'xy': [x, y], 'yaw': rad}``; the parts turn with the yaw
    about the balance origin, exactly as the MJCF body does.
    """
    bx, by = (float(v) for v in balance['xy'])
    yaw = float(balance.get('yaw', 0.))
    cos, sin = math.cos(yaw), math.sin(yaw)
    pan_half = (design.pan_half[0], design.hanger_back + design.rod_radius)
    shapes = {}
    for name, dx, half in (('balance_left_pan', -design.arm, pan_half),
                           ('balance_right_pan', design.arm, pan_half),
                           ('balance_base', 0., (design.base_half[0], design.base_half[1]))):
        shapes[name] = S.Box((bx + dx * cos, by + dx * sin), half, yaw)
    return shapes


def parcel_keepouts(spec):
    """Everything a parcel must stay clear of, with their names."""
    shapes = balance_shapes(spec['balance'])
    shapes['answer_box'] = S.rect(spec['box']['xy'], spec['box']['outer_half'])
    shapes['submit'] = S.rect(spec['submit']['xy'], SUBMIT_HALF)
    return shapes


def _assignment(rng, count, values, mixed):
    """``count`` labels drawn uniformly over the assignments that are (not) all the same."""
    options = [tuple(t) for t in itertools.product(values, repeat=count)
               if (len(set(t)) > 1) == bool(mixed)]
    if not options:
        raise ValueError(f'no {"mixed" if mixed else "uniform"} assignment of {values} over {count}')
    return list(options[int(rng.integers(0, len(options)))])


def structure_class(directions, deltas):
    """The (direction pattern, magnitude pattern) class of a realised set of odd parcels."""
    direction = 'HL' if len(set(directions)) > 1 else next(iter(directions))
    magnitude = 'diff' if len(set(deltas)) > 1 else 'same'
    return direction, magnitude


# ---- observability (spec 2.2) ---------------------------------------------------------------------
def pan_place(balance, side, design=DEFAULT_DESIGN):
    """Task-frame centre, half sizes and yaw of a pan when the beam is level: the place a plate covers."""
    bx, by = (float(v) for v in balance['xy'])
    yaw = float(balance.get('yaw', 0.))
    sign = -1. if side == 'pan_left' else 1.
    return dict(xy=[bx + sign * design.arm * math.cos(yaw), by + sign * design.arm * math.sin(yaw)],
                half=[float(design.pan_half[0]), float(design.pan_half[1])], yaw=yaw)


def plate_set_down(spec, side):
    """Where a plate lifted off ``side`` could be set down, as the mint-time feasibility check sees it.

    The skill (:func:`uncover.lift_plate_from_pan`) drags the plate off the pan, carries it out of the balance
    and lowers it onto a free patch of counter its own diameter across, taken from :func:`uncover.free_spots`
    on plan footprints and kept only when it stands 0.28-0.60 m ahead of the base and at most 0.30 m across.
    The carry is what fails: the plate rides on a few millimetres of lip in the fingertips and slides a little
    at every stop, so the certified passes all carried 0.149-0.186 m and every shear came at 0.24-0.26 m
    (SKILLS-3, cert13-cert16). Here the same search runs on the geometry the generator knows - the parcels,
    both pans, the balance base, the answer box and the submit button, inside the task footprint - with the
    base modelled where it spawns: square in front of the pan, its front at the surface's edge.

    Returns ``(spot, distance)``: the task-frame centre of the nearest patch and its distance from the plate's
    start, or ``(None, None)`` when no patch within :data:`PLATE_CARRY_MAX_M` qualifies.
    """
    from roboquest import uncover as U
    place = pan_place(spec['balance'], side)
    plate_xy = (float(place['xy'][0]), float(place['xy'][1]))
    radius = PLATE_RADIUS_M + U.PLATE_DROP_MARGIN_M
    front_y = -(float(spec['footprint'][1]) / 2 + RoboQuestKitchen.FOOTPRINT_FRONT_MARGIN)
    base_y = front_y - PLATE_BASE_STANDOFF_M
    band = (base_y + U.REACH_BAND_M[0], base_y + U.REACH_BAND_M[1])
    hx, hy = float(spec['parcel_half'][0]), float(spec['parcel_half'][1])
    keepouts = [S.shape_at((hx, hy), p['xy'], p['yaw']) for p in spec['parcels']] \
        + list(parcel_keepouts(spec).values())
    for spot in U.free_spots(frame_rect(spec['footprint']), keepouts, radius, band=band, prefer=plate_xy,
                             gap=U.PLATE_DROP_GAP_M, edge=.01):
        if abs(spot[0] - plate_xy[0]) > PLATE_SPOT_LATERAL_M:
            continue
        distance = math.hypot(spot[0] - plate_xy[0], spot[1] - plate_xy[1])
        if distance <= PLATE_CARRY_MAX_M:
            return [float(spot[0]), float(spot[1])], float(distance)
    return None, None


def liftable_pans(spec):
    """The pans a plate may cover here: those whose plate has somewhere within reach to be set down
    (:func:`plate_set_down`). Empty on a crowded counter, where the cover has no working skill."""
    return [side for side in PAN_PLACES if plate_set_down(spec, side)[0] is not None]


def bowl_cover_radius(parcel_half=PARCEL_HALF):
    """Largest outer radius an inverted bowl cover may have over a parcel (the build takes any fitting bowl), or None."""
    r, h = math.hypot(parcel_half[0], parcel_half[1]), 2 * parcel_half[2]
    radii = [float(cover['radius']) * s for cover in O.BOWL_COVERS
             for s in [O.bowl_scale_for(cover, r, h)] if s is not None]
    return max(radii) if radii else None


def cover_radius(kind):
    """Footprint radius of an object cover over a parcel, as the mint-time feasibility check sees it."""
    if kind == 'cloche_small':
        return float(O.CLOCHE_SIZES['small']['radius'])
    if kind == 'bowl':
        return bowl_cover_radius()
    raise ValueError(f'no parcel cover of kind {kind!r}')


def hideable_parcels(parcels):
    """Parcels that may leave their spot or take a cover: nothing stands on them."""
    bases = {p['on'] for p in parcels if p.get('on')}
    return [p['id'] for p in parcels if p['id'] not in bases]


def coverable_parcels(spec, kind):
    """Ground parcels a cover of ``kind`` fits where they stand, from the geometry known at mint time: nothing
    on top, the cover's footprint clear of every other prop by ``COVER_GAP`` and, for the bowl, a straight
    slide path to the front edge that is clear of the props (what ``observe.slide_gate`` checks at build,
    minus the decor and the side edges, which only the kitchen knows)."""
    radius = cover_radius(kind)
    if radius is None:
        return []
    parcels = spec['parcels']
    hx, hy = float(spec['parcel_half'][0]), float(spec['parcel_half'][1])
    hideable = set(hideable_parcels(parcels))
    shapes = {p['id']: S.shape_at((hx, hy), p['xy'], p['yaw']) for p in parcels}
    keepouts = list(parcel_keepouts(spec).values())
    front_y = -(float(spec['footprint'][1]) / 2 + RoboQuestKitchen.FOOTPRINT_FRONT_MARGIN)
    out = []
    for p in parcels:
        if p['level'] or p['id'] not in hideable:
            continue
        others = [shape for name, shape in shapes.items() if name != p['id']] + keepouts
        disc = S.Disc((float(p['xy'][0]), float(p['xy'][1])), radius)
        if any(S.clearance(disc, other) < COVER_GAP for other in others):
            continue
        if kind == 'bowl':
            slide = (float(p['xy'][1]) - front_y) - radius / 3
            if slide <= 0. or slide > O.MAX_SLIDE_PATH_M:
                continue
            strip = S.Box((float(p['xy'][0]), float(p['xy'][1]) - slide / 2),
                          (slide / 2 + radius + O.COVER_EDGE_MARGIN_M, radius + O.COVER_EDGE_MARGIN_M), -math.pi / 2)
            if any(S.clearance(strip, other) < 0. for other in others):
                continue
        out.append(p['id'])
    return out


def draw_observability(rng, level, spec):
    """``spec['hidden']`` for ``level`` from the nuisance stream ``rng``, per the design table (spec 2.2).

    ``look``: one or two parcels, each in the annex or behind an occluder (50/50), drawn equally over odd and
    normal parcels (``observe.draw_hidden``). ``uncover``: a parcel under a small cloche or an inverted bowl
    (the kind drawn among those that fit some parcel here, the parcel drawn equally over odd and normal among
    the fitting ones). The flavour coin is still thrown and recorded as ``flavour_requested``, but the
    ``place`` flavour - a plate on one balance pan - is never realised at v4: the oracle skill certified only
    6 of the 15 instances that kept it even when the mint refused every pan whose plate had no set-down patch
    within reach, under the 80 % the rule asks for, so the fallback applies and every uncover instance takes an object cover. When no cover fits any parcel - a counter
    crowded enough that no parcel takes one - no cover entry is drawn, so :meth:`OddParcel.validate_spec`
    refuses the seed and the mint walks to the next one.
    ``look+uncover``: one look entry and one cover entry on different parcels. Cover entries come first so
    the build places the cover before it stands occluders. Returns ``(entries, record)``."""
    parcels = spec['parcels']
    odd = {p['id'] for p in parcels if p['odd']}
    hideable = hideable_parcels(parcels)
    look, cover = [], []
    record = dict(look_count=0, flavour=None, flavour_requested=None, cover_kind=None)
    if level in ('look', O.DEV_ONLY_LEVEL):
        count = 1 if level == O.DEV_ONLY_LEVEL else LOOK_COUNTS[int(rng.integers(len(LOOK_COUNTS)))]
        for name in O.draw_hidden(rng, hideable, odd, count):
            look.append(O.hidden_entry(name, LOOK_MODES[int(rng.integers(len(LOOK_MODES)))]))
        record['look_count'] = len(look)
    if level in ('uncover', O.DEV_ONLY_LEVEL):
        taken = {e['object'] for e in look}
        flavour = UNCOVER_FLAVOURS[int(rng.integers(len(UNCOVER_FLAVOURS)))]
        record['flavour_requested'] = flavour
        candidates = {kind: [n for n in coverable_parcels(spec, kind) if n not in taken] for kind in OBJECT_COVER_KINDS}
        flavour = 'object' if any(candidates.values()) else None      # 'place' is never realised (item 24)
        if flavour == 'object':
            kinds = [k for k in OBJECT_COVER_KINDS if candidates[k]]
            kind = kinds[int(rng.integers(len(kinds)))]
            cover.append(O.hidden_entry(O.draw_hidden(rng, candidates[kind], odd, 1)[0], 'cover', kind))
            record['cover_kind'] = kind
        record['flavour'] = flavour
    return cover + look, record


class OddParcel(RoboQuestKitchen):
    TASK_NAME = 'odd_parcel'
    CONTRACT_VERSION = CONTRACT_VERSION
    GENERATOR_VERSION = GENERATOR_VERSION   # v2: 4/6/8 nested odd counts, stacks, answer box; v3: observability
    BUDGET_TICKS = BUDGET_TICKS
    # Reported axes (contract C2): the size and the observability level; the combined level is dev-only.
    FACTORS = {'parcel_count': (4, 6, 8), 'observability': OBSERVABILITY_LEVELS}
    DEV_ONLY_LEVELS = {'observability': (O.DEV_ONLY_LEVEL,)}
    FOOTPRINT = (1.20, .60)
    ROBOT_CENTERING_MIN_M = .10     # slide the spawn anchor when the frame sits further than this off the fixture centre
    # Hold-out: {6/2, mixed directions}, {8/3, different magnitudes} and
    # {4/1, light}. Every size level keeps a held-out class and every level of every factor,
    # every direction pattern and every magnitude pattern still appears in dev.
    HOLDOUT_KEYS = frozenset({'p4-o1-L',
                              'p6-o2-HL-same', 'p6-o2-HL-diff',
                              'p8-o3-H-diff', 'p8-o3-L-diff', 'p8-o3-HL-diff'})
    DESIGN = DEFAULT_DESIGN

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        scope = (GOAL_SCOPE_COUNT.format(count=int(spec['parcel_count']), noun=GOAL_NOUN)
                 if spec.get('show_count', False) else GOAL_SCOPE.format(noun=GOAL_NOUN))
        return ' '.join((scope, GOAL_MIDDLE, GOAL_CLUTTER))

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        n = int(factors['parcel_count'])
        k = ODD_COUNT[n]
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        # Structure class first, drawn balanced, then realised: direction pattern for every odd
        # count, magnitude pattern only where two odd parcels can differ.
        direction_pattern = DIRECTION_PATTERNS[int(structure.integers(0, len(DIRECTION_PATTERNS)))] \
            if k > 1 else ('H', 'L')[int(structure.integers(0, 2))]
        magnitude_pattern = MAGNITUDE_PATTERNS[int(structure.integers(0, len(MAGNITUDE_PATTERNS)))] \
            if k > 1 else 'same'
        directions = (_assignment(structure, k, ('H', 'L'), mixed=True) if direction_pattern == 'HL'
                      else [direction_pattern] * k)
        deltas = _assignment(structure, k, DELTA_KEYS, mixed=magnitude_pattern == 'diff')
        odd = sorted(int(i) for i in poses.choice(n, size=k, replace=False))
        # Nuisance: the balance's own pose (+/-5 cm, +/-10 deg), then the parcels scattered over the
        # free area in front of and beside it, clear of the answer box and Submit, then which of them
        # start stacked two high. Every draw raises ValueError when it cannot be satisfied, which
        # rejects the seed.
        frame = frame_rect(cls.FOOTPRINT)
        pose = S.scatter(poses, 1, S.rect(BALANCE_LOCAL, BALANCE_JITTER), balance_extent(cls.DESIGN), 0.,
                         yaw_range=(-BALANCE_JITTER_YAW, BALANCE_JITTER_YAW),
                         region_holds='center', bounds=frame)[0]
        balance = dict(xy=[pose[0], pose[1]], yaw=pose[2], pan_capacity=3)
        box = dict(xy=list(BOX_LOCAL), inner_xy=list(interior_xy((2 * PARCEL_HALF[0], 2 * PARCEL_HALF[1]))),
                   outer_half=list(box_outer_half()), style=int(materials.integers(0, 1 << 30)))
        submit = dict(xy=list(SUBMIT_LOCAL))
        keepouts = parcel_keepouts(dict(balance=balance, box=box, submit=submit))
        stacks = int(poses.integers(0, max_stacks(n) + 1))
        placed = S.scatter(poses, n - stacks, parcel_region(cls.FOOTPRINT),
                           (PARCEL_HALF[0], PARCEL_HALF[1]), MIN_PARCEL_GAP,
                           keepouts=list(keepouts.values()), keepout_gap=PARCEL_KEEPOUT_GAP,
                           yaw_range=PARCEL_YAW_RANGE, max_tries=40000)
        tops = sorted(int(i) for i in poses.choice(n, size=stacks, replace=False))
        ground = [i for i in range(n) if i not in set(tops)]
        bases = [ground[int(i)] for i in poses.choice(len(ground), size=stacks, replace=False)]
        layout = {index: dict(xy=[x, y], yaw=yaw, level=0, on=None)
                  for index, (x, y, yaw) in zip(ground, placed)}
        for top, base in zip(tops, bases):
            under = layout[base]
            draw = (float(poses.uniform(-STACK_OFFSET_M, STACK_OFFSET_M)),
                    float(poses.uniform(-STACK_OFFSET_M, STACK_OFFSET_M)),
                    float(poses.uniform(-STACK_OFFSET_YAW, STACK_OFFSET_YAW)))
            xy = [round(under['xy'][0] + draw[0], 6), round(under['xy'][1] + draw[1], 6)]
            yaw = round(under['yaw'] + draw[2], 6)
            shape = S.shape_at((PARCEL_HALF[0], PARCEL_HALF[1]), xy, yaw)
            others = [S.shape_at((PARCEL_HALF[0], PARCEL_HALF[1]), row['xy'], row['yaw'])
                      for index, row in layout.items() if index != base]
            ok = (S.region_margin(shape, frame) >= 0.
                  and all(S.clearance(shape, other) >= MIN_PARCEL_GAP for other in others)
                  and all(S.clearance(shape, keep) >= PARCEL_KEEPOUT_GAP for keep in keepouts.values()))
            if not ok:                      # fall back to dead centre, which always inherits the base's clearances
                xy, yaw = list(under['xy']), under['yaw']
            layout[top] = dict(xy=xy, yaw=yaw, level=1, on=f'parcel_{base}')
        parcels = []
        directions_by_index = dict(zip(odd, directions))
        deltas_by_index = dict(zip(odd, deltas))
        for index in range(n):
            row = layout[index]
            direction = directions_by_index.get(index)
            delta_key = deltas_by_index.get(index)
            delta = DELTA_KG[delta_key] if delta_key else 0.
            mass = BASE_MASS_KG + (delta if direction == 'H' else -delta if direction == 'L' else 0.)
            parcels.append(dict(id=f'parcel_{index}', xy=row['xy'], yaw=row['yaw'], level=row['level'],
                                on=row['on'], odd=direction is not None, direction=direction,
                                delta_key=delta_key, mass_kg=round(mass, 4)))
        texture = PARCEL_TEXTURES[int(materials.integers(0, len(PARCEL_TEXTURES)))]
        tint = PARCEL_TINTS[int(materials.integers(0, len(PARCEL_TINTS)))]
        # Observability last, from the poses stream: the layout above is identical at every level of the
        # same seed, and the structure stream never sees the level.
        level = str(factors.get('observability', OBSERVABILITY_LEVELS[0]))
        layout_spec = dict(parcels=parcels, parcel_half=list(PARCEL_HALF), balance=balance, box=box,
                           submit=submit, footprint=list(cls.FOOTPRINT))
        hidden, hidden_draw = draw_observability(poses, level, layout_spec)
        return dict(parcel_count=n, odd_count=k, direction_pattern=direction_pattern,
                    magnitude_pattern=magnitude_pattern, odd_indices=odd, parcels=parcels,
                    parcel_half=list(PARCEL_HALF), base_mass_kg=BASE_MASS_KG, delta_kg=dict(DELTA_KG),
                    stacks=stacks, parcel_texture=texture, parcel_tint=list(tint),
                    balance=balance, balance_textures=sample_textures(materials), box=box, submit=submit,
                    footprint=list(cls.FOOTPRINT), observability=level, hidden=hidden, hidden_draw=hidden_draw)

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        n, k = spec.get('parcel_count'), spec.get('odd_count')
        if n not in cls.FACTORS['parcel_count'] or k != ODD_COUNT.get(n):
            problems.append('factor levels out of range, or odd_count not derived from parcel_count')
            return problems
        parcels = spec.get('parcels', [])
        if len(parcels) != n:
            problems.append(f'{len(parcels)} parcels for parcel_count {n}')
        odd = [p for p in parcels if p.get('odd')]
        if len(odd) != k or sorted(spec.get('odd_indices', [])) != [int(p['id'].split('_')[1]) for p in odd]:
            problems.append('odd parcels do not match odd_count / odd_indices')
        if 2 * k >= n:
            problems.append('odd parcels must be fewer than half')
        direction, magnitude = structure_class([p['direction'] for p in odd], [p['delta_key'] for p in odd]) \
            if odd else ('', '')
        if spec.get('direction_pattern') != direction:
            problems.append(f"direction pattern {spec.get('direction_pattern')!r} does not describe the odd parcels")
        if spec.get('magnitude_pattern') != (magnitude if k > 1 else 'same'):
            problems.append(f"magnitude pattern {spec.get('magnitude_pattern')!r} does not describe the odd parcels")
        for p in parcels:
            delta = DELTA_KG.get(p['delta_key'], 0.) if p['delta_key'] else 0.
            want = BASE_MASS_KG + (delta if p['direction'] == 'H' else -delta if p['direction'] == 'L' else 0.)
            if abs(p['mass_kg'] - want) > 1e-6 or (p['direction'] is None) != (p['delta_key'] is None) \
                    or (p['direction'] is None) == bool(p['odd']):
                problems.append(f"{p['id']}: mass/direction/magnitude inconsistent")
        by_id = {p['id']: p for p in parcels}
        tops = [p for p in parcels if p.get('level')]
        if len(tops) != spec.get('stacks') or spec.get('stacks', 0) > max_stacks(n):
            problems.append('the stacked parcels do not match the recorded stack count')
        if len({p['on'] for p in tops}) != len(tops) or any(
                p['on'] not in by_id or by_id[p['on']]['level'] != 0 for p in tops):
            problems.append('a stacked parcel does not sit on its own ground parcel')
        if spec.get('parcel_texture') not in PARCEL_TEXTURES or list(spec.get('parcel_tint', ())) not in \
                [list(t) for t in PARCEL_TINTS]:
            problems.append('unknown parcel texture or tint')
        if not textures_are_known(spec.get('balance_textures')):
            problems.append('unknown balance textures')
        balance = spec.get('balance') or {}
        if abs(float(balance.get('yaw', 0.))) > BALANCE_JITTER_YAW + 1e-9:
            problems.append('the balance yaw is outside the jitter range')
        for axis, nominal, jitter in zip((0, 1), BALANCE_LOCAL, BALANCE_JITTER):
            if abs(float(balance.get('xy', BALANCE_LOCAL)[axis]) - nominal) > jitter + 1e-9:
                problems.append('the balance is outside its jitter box')
        if list(spec.get('box', {}).get('xy', BOX_LOCAL)) != list(BOX_LOCAL) or \
                list(spec.get('submit', {}).get('xy', SUBMIT_LOCAL)) != list(SUBMIT_LOCAL):
            problems.append('the answer box or the Submit button has moved')
        problems.extend(cls.validate_hidden(spec))
        return problems

    @classmethod
    def validate_hidden(cls, spec):
        """Problems with the observability level and its hidden entries (schema per observe.validate_hidden,
        then the task's own rules: hidden parcels carry nothing, covered parcels stand on the counter, pans
        take plates only, the entry counts and the recorded flavour follow the design table)."""
        level = spec.get('observability')
        if level not in OBSERVABILITY_LEVELS:
            return [f'observability {level!r} is not one of {OBSERVABILITY_LEVELS}']
        parcels = spec.get('parcels', [])
        by_id = {p['id']: p for p in parcels}
        problems = O.validate_hidden(spec, objects=list(by_id) + list(PAN_PLACES))
        if problems:
            return problems
        hideable = set(hideable_parcels(parcels))
        hidden = spec.get('hidden', [])
        for entry in hidden:
            name, mode, cover = entry['object'], entry['mode'], entry['cover']
            if name in PAN_PLACES:
                if mode != 'cover' or cover != PLACE_COVER_KIND:
                    problems.append(f'{name}: a pan is only ever covered by a plate')
                continue
            if name not in hideable:
                problems.append(f'{name}: a parcel with another on top cannot be hidden')
            if mode == 'cover' and (cover not in OBJECT_COVER_KINDS or by_id[name]['level']):
                problems.append(f'{name}: a parcel is covered by a small cloche or a bowl, on the counter')
        looks = [e for e in hidden if e['mode'] != 'cover']
        covers = [e for e in hidden if e['mode'] == 'cover']
        draw = spec.get('hidden_draw') or {}
        if level == 'look' and not 1 <= len(looks) <= 2 or level == O.DEV_ONLY_LEVEL and len(looks) != 1:
            problems.append(f'level {level!r} hides {len(looks)} parcel(s) by looking')
        if level in ('uncover', O.DEV_ONLY_LEVEL):
            if len(covers) != 1:
                problems.append(f'level {level!r} needs exactly one cover entry')
            else:
                flavour = 'place' if covers[0]['object'] in PAN_PLACES else 'object'
                if draw.get('flavour') != flavour or draw.get('flavour_requested') not in UNCOVER_FLAVOURS \
                        or draw.get('cover_kind') != covers[0]['cover']:
                    problems.append('the recorded uncover flavour does not describe the cover entry')
        elif covers:
            problems.append(f'level {level!r} covers nothing')
        if draw.get('look_count', 0) != len(looks):
            problems.append('the recorded look count does not match the entries')
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """Geometry only (no MuJoCo), re-deriving exactly what the sampler enforced: every prop inside
        the footprint, the ground parcels inside their region and pairwise clear, every parcel clear of
        the balance, the answer box and Submit, the balance clear of the box and Submit; a stacked
        parcel over its own base; three parcels fit a pan; the box holds one parcel with 1.5 cm per
        side and walls 1 cm above it; the design's stops and sensitivity are consistent."""
        d = cls.DESIGN
        hx, hy = spec['parcel_half'][0], spec['parcel_half'][1]
        frame = frame_rect(cls.FOOTPRINT)
        report = dict(problems=[])
        problems = report['problems']
        keepouts = parcel_keepouts(spec)
        ground = [p for p in spec['parcels'] if not p['level']]
        tops = [p for p in spec['parcels'] if p['level']]
        check = S.check([(p['xy'][0], p['xy'][1], p['yaw']) for p in ground], (hx, hy),
                        bounds=parcel_region(cls.FOOTPRINT), min_gap=MIN_PARCEL_GAP,
                        keepouts=list(keepouts.values()), keepout_gap=PARCEL_KEEPOUT_GAP,
                        names=[p['id'] for p in ground], keepout_names=list(keepouts))
        problems.extend(check['problems'])
        report['min_parcel_gap_m'] = check['min_pair_gap_m']
        report['min_parcel_keepout_gap_m'] = check['min_keepout_gap_m']
        report['min_parcel_region_margin_m'] = check['min_bounds_margin_m']
        shapes = {p['id']: S.shape_at((hx, hy), p['xy'], p['yaw']) for p in spec['parcels']}
        stack_gaps = []
        for p in tops:
            for other in spec['parcels']:
                if other['id'] in (p['id'], p['on']):
                    continue
                gap = S.clearance(shapes[p['id']], shapes[other['id']])
                stack_gaps.append(gap)
                if gap < MIN_PARCEL_GAP:
                    problems.append(f"{p['id']} closer than {MIN_PARCEL_GAP:.3f} m to {other['id']}")
            for name, keep in keepouts.items():
                gap = S.clearance(shapes[p['id']], keep)
                stack_gaps.append(gap)
                if gap < PARCEL_KEEPOUT_GAP:
                    problems.append(f"{p['id']} closer than {PARCEL_KEEPOUT_GAP:.3f} m to {name}")
            offset = math.dist(p['xy'], next(q['xy'] for q in spec['parcels'] if q['id'] == p['on']))
            if offset > STACK_OFFSET_M * math.sqrt(2.) + 1e-6:
                problems.append(f"{p['id']} sits {offset:.3f} m off its base")
        report['stacks'] = len(tops)
        report['min_stacked_gap_m'] = round(min(stack_gaps), 5) if stack_gaps else None
        margins = {name: S.region_margin(shape, frame) for name, shape in keepouts.items()}
        margins.update({name: S.region_margin(shape, frame) for name, shape in shapes.items()})
        report['min_footprint_margin_m'] = round(min(margins.values()), 5)
        for name, margin in sorted(margins.items()):
            if margin < 0.:
                problems.append(f'{name} outside the footprint')
        for a in ('balance_left_pan', 'balance_right_pan', 'balance_base'):
            for b in ('answer_box', 'submit'):
                if S.clearance(keepouts[a], keepouts[b]) < 0.:
                    problems.append(f'{a} overlaps {b}')
        if S.clearance(keepouts['answer_box'], keepouts['submit']) < 0.:
            problems.append('the answer box overlaps Submit')
        inner = d.pan_inner_half
        capacity = int(spec['balance']['pan_capacity'])
        if capacity * 2 * hx + (capacity - 1) * .01 > 2 * inner[0] or 2 * hy > 2 * inner[1] - .02:
            problems.append('three parcels do not fit a pan')
        report['pan_inner_m'] = [2 * inner[0], 2 * inner[1]]
        # The answer box: one parcel wide, 1.5 cm of clearance per side, walls 1 cm above a parcel.
        box_inner = [float(v) for v in spec['box']['inner_xy']]
        report['box_inner_m'] = box_inner
        report['box_clearance_per_side_m'] = [round(box_inner[0] / 2 - hx, 5), round(box_inner[1] / 2 - hy, 5)]
        if any(abs(v - .015) > 1e-6 for v in report['box_clearance_per_side_m']):
            problems.append('the answer box interior is not the parcel footprint plus 1.5 cm per side')
        report['odd_tilt_static_deg'] = {key: tilt_for_mass_difference(delta, d)
                                         for key, delta in sorted(spec['delta_kg'].items())}
        if min(abs(v) for v in report['odd_tilt_static_deg'].values()) < 5.:
            problems.append('an odd parcel would tilt the beam less than 5 degrees')
        if max(abs(v) for v in report['odd_tilt_static_deg'].values()) >= d.stop_deg:
            problems.append('an odd parcel would drive the beam to the stop, so its magnitude is unreadable')
        if tilt_for_mass_difference(spec['base_mass_kg'], d) < d.stop_deg:
            problems.append('an unpaired parcel would not reach the stop')
        # Covers on parcels: the footprint (and the bowl's slide path) is clear where the parcel stands.
        # A pan-plate cover: the plate has a patch of counter within reach to be set down on (item 24).
        report['hidden'] = deepcopy(spec.get('hidden', []))
        for entry in spec.get('hidden', []):
            if entry['mode'] != 'cover':
                continue
            if entry['object'] in PAN_PLACES:
                spot, distance = plate_set_down(spec, entry['object'])
                report['plate_set_down_m'] = None if distance is None else round(distance, 3)
                if spot is None:
                    problems.append(f"{entry['object']}: a plate here has no free patch of counter within "
                                    f'{PLATE_CARRY_MAX_M:.2f} m to be set down on')
            elif entry['object'] not in coverable_parcels(spec, entry['cover']):
                problems.append(f"{entry['object']}: the {entry['cover']} does not fit where it stands")
        report['parcel_count'] = spec['parcel_count']
        report['odd_count'] = spec['odd_count']
        return not problems, report

    @classmethod
    def split_key(cls, spec):
        base = f"p{spec['parcel_count']}-o{spec['odd_count']}-{spec['direction_pattern']}"
        return base if spec['odd_count'] < 2 else f"{base}-{spec['magnitude_pattern']}"

    # ---- scene protocol -----------------------------------------------------
    def _free_slide_offset(self, fixture, region):
        """The base class's sliding rule, evaluated before choosing a fixture: the first lateral offset (0, +-0.1,
        ...) at which the footprint at the front of ``region`` overlaps no decor; None when every offset is blocked."""
        rx, ry = float(region['offset'][0]), float(region['offset'][1])
        sx, sy = float(region['size'][0]), float(region['size'][1])
        top_z = float(fixture.pos[2] + fixture.size[2] / 2)
        cy = ry - sy / 2 + self.FOOTPRINT[1] / 2 + self.FOOTPRINT_FRONT_MARGIN
        decor = self._decor_boxes(top_z)
        slack = max(0., sx / 2 - self.FOOTPRINT[0] / 2)
        for dx in [0.] + [s * d for d in np.arange(.10, slack + 1e-9, .10) for s in (-1, 1)]:
            box = self._footprint_world_box([rx + dx, cy], float(fixture.rot), fixture)
            if not any(self._boxes_overlap(box, d) for d in decor):
                return float(dx)
        return None

    def _choose_work_fixture(self):
        """Islands first, then counters, as in the base class; but among qualifying surfaces prefer the largest
        whose largest region offers a decor-free slide position, so appliances on a bigger counter do not win
        over a clear one (base-class request, see handoff). Deterministic: sorted names, no RNG."""
        for kind in self.WORK_FIXTURE_TYPES:
            candidates = []
            for name in sorted(self.fixtures):
                fixture = self.fixtures[name]
                if not fixture_is_type(fixture, kind):
                    continue
                try:
                    if not FixtureUtils.is_fxtr_valid(self, fixture, self.FOOTPRINT):
                        continue
                    regions = fixture.get_reset_regions(self, top_size=self.FOOTPRINT)
                except Exception:
                    continue
                if not regions:
                    continue
                region = max(regions.values(), key=lambda r: r['size'][0] * r['size'][1])
                area = float(region['size'][0] * region['size'][1])
                blocked = self._free_slide_offset(fixture, region) is None
                candidates.append((blocked, -area, name))
            if candidates:
                candidates.sort()
                return self.fixtures[candidates[0][2]]
        raise ValueError(f'{self.TASK_NAME}: no island or counter offers a free {self.FOOTPRINT} m region')

    def _decor_boxes(self, top_z):
        """Everything standing on the work top counts as occupied, not only ``Accessory`` fixtures: toasters,
        coffee machines, toaster ovens, stand mixers, sinks and stovetops (base-class request, see handoff)."""
        boxes = super()._decor_boxes(top_z)
        seen = {b['name'] for b in boxes}
        for name, fixture in self.fixtures.items():
            if name in seen or fixture is self.work_fixture or not isinstance(fixture, Fixture):
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            if points[:, 2].min() > top_z + .05 or points[:, 2].max() < top_z + .01:
                continue    # above the counter (wall cabinets) or below its top (base cabinets, other counters)
            boxes.append(dict(name=name, x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                              y0=float(points[:, 1].min()), y1=float(points[:, 1].max())))
        return boxes

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._center_robot_on_frame()

    def _center_robot_on_frame(self):
        """RoboCasa parks the robot at the fixture's centre; slide the spawn anchor along the counter front so it
        faces the task frame when the frame slid sideways (base-class request, see handoff)."""
        fixture = self.work_fixture
        rot = yaw_matrix(float(fixture.rot))
        anchor = np.asarray(self.init_robot_base_pos_anchor, float)
        anchor_local = rot.T @ (anchor - np.asarray(fixture.pos, float))
        shift = float(self.work['center_local'][0]) - float(anchor_local[0])
        self.work['robot_anchor_shift_m'] = 0.
        if abs(shift) >= self.ROBOT_CENTERING_MIN_M:
            self.init_robot_base_pos_anchor = (anchor + rot @ np.array([shift, 0., 0.])).tolist()
            self.work['robot_anchor_shift_m'] = shift
            self.task_spec['work'] = deepcopy(self.work)

    def parcel_z(self, level):
        """Height of a parcel's centre above the work top at reset, by stack level.

        Every parcel starts 2 mm above where it will rest and a stacked one only 0.5 mm above
        its base, so the whole scene settles well inside the 5 mm G2 budget.
        """
        half = self.spec['parcel_half'][2]
        return half + .002 + int(level) * (2 * half + .0005)

    def build_task(self):
        self.place_work_frame()
        spec = self.spec
        frame = self.add_frame_body('balance_frame', spec['balance']['xy'])
        # The balance carries its own yaw on top of the work surface's; every reading is taken
        # from the simulated sites and bodies, so the jitter needs no other change.
        frame.set('quat', ' '.join(f'{v:.9g}' for v in
                                   yaw_quat_wxyz(self.work['yaw'] + float(spec['balance'].get('yaw', 0.)))))
        design = with_textures(spec['balance_textures'], self.DESIGN)
        self.balance = add_balance(frame, self.model.asset, self.model.contact, prefix='balance', design=design)
        # The answer box replaces the v0 tray zone: open-top, one parcel wide, welded to the frame.
        box_frame = self.add_frame_body('answer_box_frame', spec['box']['xy'])
        self.box = answer_box(box_frame, 'answer_box', parcel_xy=(2 * spec['parcel_half'][0],
                                                                 2 * spec['parcel_half'][1]),
                              parcel_height=2 * spec['parcel_half'][2],
                              rng=np.random.default_rng(int(spec['box']['style'])))
        self.box['frame_xy'] = list(spec['box']['xy'])
        texture = ROBOCASA_TEXTURES / spec['parcel_texture']
        tint = [float(v) for v in spec['parcel_tint']]
        half = spec['parcel_half']
        volume = 8. * half[0] * half[1] * half[2]
        self.parcel_ids = []
        for p in spec['parcels']:
            material = CustomMaterial(texture=str(texture), tex_name=f"{p['id']}_tex", mat_name=f"{p['id']}_mat",
                                      tex_attrib={'type': 'cube'},
                                      mat_attrib={'texrepeat': '1 1', 'texuniform': 'true', 'specular': '0.1',
                                                  'shininess': '0.05', 'reflectance': '0',
                                                  'rgba': ' '.join(f'{v:.4g}' for v in tint)})
            obj = BoxObject(name=p['id'], size=list(half), density=p['mass_kg'] / volume,
                            friction=[.9, .005, .0001], solref=[.006, 1.], solimp=[.98, .999, .001],
                            material=material)
            self.objects[p['id']] = obj
            pos = self.frame_to_world([p['xy'][0], p['xy'][1], self.parcel_z(p['level'])])
            yaw = self.work['yaw'] + p['yaw']
            self._poses[p['id']] = [float(pos[0]), float(pos[1]), float(pos[2]), math.cos(yaw / 2), 0., 0.,
                                    math.sin(yaw / 2)]
            self.task_spec['objects'][p['id']] = {'kind': 'object', 'fixed': False}
            self.parcel_ids.append(p['id'])
        self.add_submit_button(spec['submit']['xy'])
        box_center = self.frame_to_world([spec['box']['xy'][0], spec['box']['xy'][1], 0.])
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, budget_ticks=BUDGET_TICKS,
            parcel_count=spec['parcel_count'], odd_count=spec['odd_count'],
            odd_parcels_private=[p['id'] for p in spec['parcels'] if p['odd']],
            parcel_masses_private={p['id']: p['mass_kg'] for p in spec['parcels']},
            structure_class_private=[spec['direction_pattern'], spec['magnitude_pattern']],
            balance=dict(frame_xy=spec['balance']['xy'], frame_yaw_rad=float(spec['balance'].get('yaw', 0.)),
                         **self.balance),
            answer_box=dict(self.box, center_world=[float(v) for v in box_center]),
            placement='Parcels scattered over the free area in front of and beside the balance at random '
                      f"yaws (no rows), {spec['stacks']} of them stacked two high; the balance itself "
                      'jittered by +/-5 cm and +/-10 deg; nothing overlapping '
                      '(roboquest.scatter).',
            frame_origin_world=self.work['center_world'], frame_yaw=self.work['yaw'],
            scoring='First physical Submit freezes the score: exactly the odd parcels inside the answer box, '
                    'released and settled, and the balance within 2 cm of its reset pose (a parcel left on a '
                    'pan is flagged, not failed).')
        self.asset_evidence.update(
            parcels=f'robosuite BoxObject 7x10x5 cm, one RoboCasa texture ({texture}) and tint {tint}; '
                    'masses differ only in density',
            balance='Purpose-built two-pan beam balance (hinges, flexure, stops) dressed with RoboCasa metal textures',
            answer_box='Private placeholder with the C4 answer_box signature (balance/answer_box.py); '
                       'open-top column, interior = parcel footprint + 1.5 cm per side, walls = parcel + 1 cm',
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged')
        # Observability last: the hidden parcels move (annex, relocation before an occluder) or take a cover,
        # so every other prop is already in place to be kept clear of.
        self.extra_objects = []
        self._apply_observability()

    # ---- observability (spec 2.2): realised at the end of build_task -------------------------
    def _place_of_pan(self, side):
        """A pan as a coverable place for ``observe.cover_place``: the level pan's task-frame centre, half sizes
        and yaw, the plate resting on the pan's rims, in lift mode (pinched at its rim, never pushed, so the
        beam is not swung)."""
        d = self.DESIGN
        top = float(self.work['top_z'])
        return dict(pan_place(self.spec['balance'], side, d), support_z=top + float(d.pan_floor_z) + float(d.rim_height),
                    mode='lift')

    def _prop_keepouts(self, exclude=()):
        """Task-frame shapes of every prop (parcels, balance pans and base, answer box, Submit) but ``exclude``."""
        spec = self.spec
        hx, hy = float(spec['parcel_half'][0]), float(spec['parcel_half'][1])
        shapes = {name: shape for name, shape in parcel_keepouts(spec).items() if name not in exclude}
        for p in spec['parcels']:
            if p['id'] not in exclude:
                shapes[p['id']] = S.shape_at((hx, hy), p['xy'], p['yaw'])
        return shapes

    def _occluder_relocate_region(self, surface):
        """The largest free strip of the work top behind or beside the frame that a gated base stance reaches,
        or None when the surface has none: nothing standing on the counter hides a parcel in the front band
        under the high policy cameras, so an occluded parcel may move there (``observe.place_occluder``'s
        relocation, recorded per instance). The strips are cut to the reach-gated cells ``observe.annex_regions``
        finds on the work surface (a stance beyond an accessible edge, within reach, not blocked by stools or
        other fixtures), so a relocated parcel always has a viewpoint (``look_reachable`` at G3)."""
        hx, hy = self.FOOTPRINT[0] / 2, self.FOOTPRINT[1] / 2
        m = RELOCATE_MARGIN
        strips = [S.Rect(-hx, hx, hy + m, surface.y1 - m),
                  S.Rect(surface.x0 + m, -hx - m, surface.y0 + m, surface.y1 - m),
                  S.Rect(hx + m, surface.x1 - m, surface.y0 + m, surface.y1 - m)]
        cells = []
        for region in O.annex_regions(self, all_regions=True, min_along=0.):
            if region['kind'] != 'work' or not region['reach']['passed']:
                continue
            pts = np.array([O.world_to_frame_xy(self, c) for c in region['corners_world']])
            cells.append(S.Rect(float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())))
        options = []
        for strip in strips:
            for cell in cells:
                r = S.Rect(max(strip.x0, cell.x0), min(strip.x1, cell.x1), max(strip.y0, cell.y0), min(strip.y1, cell.y1))
                if r.x1 - r.x0 >= RELOCATE_MIN_SIDE and r.y1 - r.y0 >= RELOCATE_MIN_SIDE:
                    options.append(r)
        return max(options, key=lambda r: (r.x1 - r.x0) * (r.y1 - r.y0)) if options else None

    def _apply_observability(self):
        """Realise ``spec['hidden']`` with ``observe.apply_observability`` and refuse the build when the scene
        does not realise every entry exactly as drawn (the occluder -> annex fallback excepted): the seed is
        then rejected at G1. Keep-outs are every prop but the hidden ones; the extra objects (occluders and
        covers) carry ``role`` in ``task_spec['objects']`` and are listed in ``self.extra_objects``."""
        spec = self.spec
        hidden = list(spec.get('hidden') or [])
        exclude = {entry['object'] for entry in hidden}
        exclude |= {'balance_left_pan' if entry['object'] == 'pan_left' else 'balance_right_pan'
                    for entry in hidden if entry['object'] in PAN_PLACES}
        keepouts = self._prop_keepouts(exclude)
        surface = O.work_surface_rect(self)
        relocate = self._occluder_relocate_region(surface)
        rng = np.random.default_rng([int(self.instance.get('seed', 0)) & 0xFFFFFFFF, O.OBSERVE_SALT, OBSERVE_TASK_SALT])
        objects = {p['id']: dict(radius=PARCEL_RADIUS, height=2 * float(spec['parcel_half'][2])) for p in spec['parcels']}
        places = {side: self._place_of_pan(side) for side in PAN_PLACES}
        record = O.apply_observability(self, spec, objects=objects, places=places, keepouts=list(keepouts.values()),
                                       bounds=surface, rng=rng, occluder_relocate=relocate)
        record['relocate_region'] = None if relocate is None else [relocate.x0, relocate.x1, relocate.y0, relocate.y1]
        problems = list(record['problems'])
        realised = {row['object']: row for row in record['realised']}
        for entry in hidden:
            got = realised.get(entry['object'])
            if got is None:
                problems.append(f"{entry['object']}: {entry['mode']} entry not realised")
            elif got['realised'] != entry['mode'] and not (entry['mode'] == 'occluder' and got['realised'] == 'annex'):
                problems.append(f"{entry['object']}: {entry['mode']} entry realised as {got['realised']}")
            elif entry['mode'] == 'cover' and not got.get('reach_passed'):
                problems.append(f"{entry['object']}: the {entry['cover']} reach gate failed")
        if problems:
            raise RuntimeError(f"{self.TASK_NAME}: observability {spec.get('observability')!r} not realised: "
                               + '; '.join(problems))
        self.extra_objects = [name for name, info in self.task_spec['objects'].items()
                              if info.get('role') in ('occluder', 'cover')]
        self.task_spec['hidden_parcels_private'] = [name for name in record['hidden_bodies'] if name in self.parcel_ids]
        return record

    def cover_names(self):
        """The covers built for this instance (cloche, bowl or plate), in build order."""
        return [name for name in getattr(self, 'extra_objects', [])
                if self.task_spec['objects'][name].get('role') == 'cover']

    def setup_task_references(self):
        m = self.sim.model
        self._beam_qadr = m.get_joint_qpos_addr(self.balance['beam_joint'])
        self._pan_qadr = {s: m.get_joint_qpos_addr(p['joint']) for s, p in self.balance['pans'].items()}
        self._pan_bid = {s: m.body_name2id(p['body']) for s, p in self.balance['pans'].items()}
        self._pan_site = {s: m.site_name2id(p['floor_site']) for s, p in self.balance['pans'].items()}
        self._pointer_site = m.site_name2id(self.balance['pointer_tip_site'])
        self._base_bid = m.body_name2id(self.balance['base_body'])
        self._box_bid = m.body_name2id(self.box['body'])
        self._box_site = m.site_name2id(self.box['floor_site'])
        self._balance_gids = {i for i in range(m.ngeom)
                              if (m.geom_id2name(i) or '').startswith('balance_') and (m.geom_contype[i] or m.geom_conaffinity[i])}
        self._pan_gids = {s: {m.geom_name2id(g) for g in self.balance['geoms'][f'pan_{s}']} for s in ('left', 'right')}
        self._parcel_gids = {name: set(self._collision_geom_ids(name)) for name in self.parcel_ids}
        self._cache_parcel_corners()
        self._reset_poses = deepcopy(self._poses)
        self._capture_balance_reference()
        # Observability extras: their geom prefixes are legitimate contacts for the props, and a plate on a
        # pan pre-tilts the beam at reset (see _settle_plates_on_pans).
        self._extra_prefixes = tuple(name + '_' for name in getattr(self, 'extra_objects', []))
        record = self.task_spec.get('observability') or {}
        self._cover_records = {c['name']: c for c in record.get('covers', []) if c.get('built')}
        self._pan_plates = [(c['name'], c['place']) for c in self._cover_records.values()
                            if c['kind'] == PLACE_COVER_KIND and c.get('place') in PAN_PLACES]
        self._annex_supports = {a['object']: a['fixture'] for a in record.get('annex', [])}
        self._plate_tilt = None
        # Joint addresses of the robot, held still while the props settle at reset.
        model = self.sim.model._model
        qpos, dof = [], []
        for j in range(model.njnt):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])) or ''
            if not body.startswith(self.ROBOT_PREFIXES):
                continue
            size = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3)}.get(int(model.jnt_type[j]), (1, 1))
            qpos.extend(range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + size[0]))
            dof.extend(range(int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + size[1]))
        self._robot_qpos_idx, self._robot_dof_idx = np.asarray(qpos, int), np.asarray(dof, int)
        # The goal volume must be free: decor, and equally a cover or an occluder the observability wiring
        # stood on the counter, must not fill the answer box's column (EVAL-KITCHEN-DEFECTS, layouts 2/3/6/9).
        blocked = self._box_column_blocked()
        if blocked:
            raise ValueError(f'{self.TASK_NAME}: RoboCasa decor fills the answer box on layout '
                             f"{self.instance['layout_id']} style {self.instance['style_id']}, so the "
                             'instance has no solved state: ' + '; '.join(blocked[:4]))

    def _box_column_blocked(self):
        """Answer-box levels a parcel cannot occupy because a piece of decor is already there.

        ``foreign_contacts`` only sees the reset scene, where every parcel is on a pan or on
        the work top, so a box built under a toaster passes G1 and the instance fails only
        once it is solved: on layout 3 / style 12 the first parcel lands 15 mm inside
        ``toaster_right_group`` and the solver tips it back out of the box, which no policy
        can undo. Send one parcel through every level the solution needs, keep the foreign
        contacts (covers and occluders count as foreign here), and put the parcels back
        exactly where they were.
        """
        data = self.sim.data
        probe = self.parcel_ids[0]
        levels = max(1, sum(1 for parcel in self.spec['parcels'] if parcel['odd']))
        saved = {name: np.array(data.get_joint_qpos(self.objects[name].joints[0]), float).copy()
                 for name in self.parcel_ids}
        blocked = []
        try:
            for level in range(levels):
                self.place_parcel_in_box(probe, level)
                for row in self.foreign_contacts(allow_extras=False):
                    mine, _, other = row.partition(' | ')
                    if mine.startswith(f'{probe}_'):
                        blocked.append(f'level {level} in {other}')
        finally:
            for name in self.parcel_ids:
                data.set_joint_qpos(self.objects[name].joints[0], saved[name])
                data.set_joint_qvel(self.objects[name].joints[0], np.zeros(6))
            self.sim.forward()
        return sorted(set(blocked))

    def _reset_internal(self):
        super()._reset_internal()
        if getattr(self, '_pan_plates', None):
            self._settle_plates_on_pans()
            self._settle_at_reset()

    def _settle_at_reset(self, seconds=.8):
        """Step the physics with the robot pinned at its reset configuration, so the props are at rest when
        reset returns: a plate lands on the pan rims from its bounding-box height (its underside is curved,
        so it drops about a centimetre) and the beam takes the load. The gates measure drift from the reset
        poses (G2), so the settling happens here rather than in their budget; the tick counter and the
        controllers are untouched."""
        model, d = self.sim.model._model, self.sim.data._data
        idx, dof = self._robot_qpos_idx, self._robot_dof_idx
        held = d.qpos[idx].copy()
        for _ in range(int(round(seconds / model.opt.timestep))):
            mujoco.mj_step(model, d)
            d.qpos[idx] = held
            d.qvel[dof] = 0.
        mujoco.mj_forward(model, d)

    def _settle_plates_on_pans(self):
        """A plate on a pan tilts the beam (the probe measured 12.7 mm at the pan under a 72 g plate), so start
        the beam at the static equilibrium of the load and the plate on the lowered pan: the reset state is
        then already settled and G2, which measures drift from the reset poses, sees millimetres instead of
        the swing. The flexure balances ``M g arm cos(theta)`` against the parts hanging below the pivot."""
        plates = getattr(self, '_pan_plates', None)
        if not plates:
            return
        model, d = self.sim.model._model, self.sim.data
        design = self.DESIGN
        moment = sum((1. if side == 'pan_right' else -1.) * float(model.body_subtreemass[self.obj_body_id[name]])
                     for name, side in plates)
        load = moment * GRAVITY * design.arm
        hang = (design.mass_beam * design.beam_drop + design.mass_pointer * design.pointer_length / 2) * GRAVITY
        angle = 0.
        for _ in range(30):
            angle = (load * math.cos(angle) - hang * math.sin(angle)) / design.beam_stiffness
        limit = design.stop_rad - math.radians(.5)
        angle = max(-limit, min(limit, angle))
        floors = {side: np.array(d.site_xpos[self._pan_site[side]], float) for side in ('left', 'right')}
        d.qpos[self._beam_qadr] = angle
        for side in ('left', 'right'):
            d.qpos[self._pan_qadr[side]] = -angle          # the pans keep hanging vertically
        self.sim.forward()
        for name, side in plates:
            key = 'left' if side == 'pan_left' else 'right'
            delta = np.asarray(d.site_xpos[self._pan_site[key]], float) - floors[key]
            joint = self.objects[name].joints[0]
            pose = np.array(d.get_joint_qpos(joint), float)
            pose[:3] += delta
            d.set_joint_qpos(joint, pose)
            d.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()
        self._plate_tilt = dict(angle_deg=round(math.degrees(angle), 3), load_kg=round(moment, 4),
                                plates=[dict(name=name, pan=side) for name, side in plates])

    def _cache_parcel_corners(self):
        """Local corners of every parcel's collision boxes, so the per-tick checks are state reads.

        ``_world_collision_points`` rescans the model's geom names on every call, which costs
        milliseconds in a RoboCasa kitchen; ``collateral_checks`` runs every tick, so the geom
        ids and their corner offsets are resolved once here. Falls back to the generic routine
        if a parcel ever stops being a box.
        """
        m = self.sim.model
        cache = {}
        for name in self.parcel_ids:
            rows = []
            for gid in sorted(self._parcel_gids[name]):
                if int(m.geom_type[gid]) != mujoco.mjtGeom.mjGEOM_BOX:
                    self._parcel_corners = None
                    return
                rows.append((int(gid), BOX_CORNERS * np.asarray(m.geom_size[gid], float)))
            cache[name] = rows
        self._parcel_corners = cache

    # ---- private state ------------------------------------------------------
    def beam_angle_deg(self):
        return math.degrees(float(self.sim.data.qpos[self._beam_qadr]))

    def pan_angle_deg(self, side):
        return math.degrees(float(self.sim.data.qpos[self._pan_qadr[side]]))

    def _capture_balance_reference(self):
        """World positions of the balance's movable parts at its reset pose.

        Taken from a scratch ``MjData`` at ``qpos0`` rather than from the live state, so the
        reference is the same whenever it is captured. The balance frame is welded to the world,
        so the reference only depends on the compiled model.
        """
        model = self.sim.model._model
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        self._balance_reset = {
            'base': np.array(data.xpos[self._base_bid], float),
            'beam_end_left': np.array(data.xpos[self._pan_bid['left']], float),
            'beam_end_right': np.array(data.xpos[self._pan_bid['right']], float),
            'pan_left_floor': np.array(data.site_xpos[self._pan_site['left']], float),
            'pan_right_floor': np.array(data.site_xpos[self._pan_site['right']], float),
            'pointer_tip': np.array(data.site_xpos[self._pointer_site], float)}

    def balance_displacements_m(self):
        """How far each tracked part of the balance has moved from its reset pose."""
        d = self.sim.data
        now = {'base': np.asarray(d.body_xpos[self._base_bid], float),
               'beam_end_left': np.asarray(d.body_xpos[self._pan_bid['left']], float),
               'beam_end_right': np.asarray(d.body_xpos[self._pan_bid['right']], float),
               'pan_left_floor': np.asarray(d.site_xpos[self._pan_site['left']], float),
               'pan_right_floor': np.asarray(d.site_xpos[self._pan_site['right']], float),
               'pointer_tip': np.asarray(d.site_xpos[self._pointer_site], float)}
        return {name: float(np.linalg.norm(now[name] - reference))
                for name, reference in self._balance_reset.items()}

    def balance_displacement_m(self):
        """How far the instrument itself stands from its reset pose, measured on the base:
        the beam and the pans tilt by design (a parcel or a plate on a pan) and never count as displacement.
        ``balance_displacements_m`` still reports every part for diagnostics."""
        return self.balance_displacements_m()['base']

    def _robot_touches_balance(self):
        m, d = self.sim.model, self.sim.data
        for contact in d.contact[:d.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self._balance_gids or b in self._balance_gids:
                other = b if a in self._balance_gids else a
                if (m.geom_id2name(other) or '').startswith(('robot', 'gripper', 'mobilebase')):
                    return True
        return False

    def foreign_contacts(self, allow_extras=True):
        """Contacts between the props (balance, box, parcels) and anything but the work top, the props, Submit,
        an annexed parcel's own surface and, unless ``allow_extras`` is False, the observability extras (a
        plate resting on a pan)."""
        m, d = self.sim.model, self.sim.data
        own = ('balance_', 'answer_box_', 'parcel_', 'submit_')
        allowed = (self.work['fixture'] + '_',) + own
        if allow_extras:
            allowed += tuple(getattr(self, '_extra_prefixes', ()))
        annexed = getattr(self, '_annex_supports', {}) or {}
        rows = set()
        for contact in d.contact[:d.ncon]:
            a, b = m.geom_id2name(int(contact.geom1)) or '', m.geom_id2name(int(contact.geom2)) or ''
            mine_a, mine_b = a.startswith(own), b.startswith(own)
            if mine_a == mine_b:
                continue
            mine, other = (a, b) if mine_a else (b, a)
            if other.startswith(allowed):
                continue
            if any(mine.startswith(parcel + '_') and other.startswith(fixture + '_') for parcel, fixture in annexed.items()):
                continue
            rows.add(f'{mine} | {other}')
        return sorted(rows)

    def parcels_on_pans(self):
        d = self.sim.data
        on = {'left': set(), 'right': set()}
        for contact in d.contact[:d.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            for side, gids in self._pan_gids.items():
                if a in gids or b in gids:
                    other = b if a in gids else a
                    for name, pgids in self._parcel_gids.items():
                        if other in pgids:
                            on[side].add(name)
        return {side: sorted(v) for side, v in on.items()}

    def pan_frame(self, side):
        """World position of the pan floor centre and the pan body rotation."""
        d = self.sim.data
        return (np.asarray(d.site_xpos[self._pan_site[side]], float),
                np.asarray(d.body_xmat[self._pan_bid[side]], float).reshape(3, 3))

    def parcel_in_pan_local(self, name, side):
        origin, rot = self.pan_frame(side)
        return rot.T @ (np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]], float) - origin)

    def box_frame(self):
        """World origin (on the work top) and rotation of the answer box."""
        d = self.sim.data
        return (np.asarray(d.body_xpos[self._box_bid], float),
                np.asarray(d.body_xmat[self._box_bid], float).reshape(3, 3))

    def parcel_points_world(self, name):
        """World corners of a parcel's collision boxes, from the cache when there is one."""
        rows = getattr(self, '_parcel_corners', None)
        if not rows:
            return np.asarray(self._world_collision_points(name), float)
        d = self.sim.data
        return np.concatenate([np.asarray(d.geom_xpos[gid], float)
                               + local @ np.asarray(d.geom_xmat[gid], float).reshape(3, 3).T
                               for gid, local in rows[name]])

    def parcel_in_box_local(self, name):
        """A parcel's collision points in the box frame, as ``(min_xyz, max_xyz)``."""
        origin, rot = self.box_frame()
        points = (self.parcel_points_world(name) - origin) @ rot
        return points.min(0), points.max(0)

    def parcels_in_box(self):
        """Parcels whose whole collision footprint is inside the box's interior column, bottom first."""
        hx, hy = (float(v) / 2 for v in self.box['inner_xy'])
        rows = []
        for name in self.parcel_ids:
            low, high = self.parcel_in_box_local(name)
            if (max(-low[0], high[0]) <= hx + BOX_FIT_TOL and max(-low[1], high[1]) <= hy + BOX_FIT_TOL
                    and low[2] >= self.box['floor_z'] - BOX_FIT_TOL):
                rows.append((float(low[2] + high[2]) / 2, name))
        return [name for _, name in sorted(rows)]

    # ---- protocol hooks (contract C1) --------------------------------------
    def _box_progress(self):
        """(odd parcels in the box - normal parcels in it) / odd count, clipped to [0, 1] (spec 1.4)."""
        odd = set(self.task_spec['odd_parcels_private'])
        boxed = set(self.parcels_in_box())
        return float(min(1., max(0., (len(boxed & odd) - len(boxed - odd)) / max(1, len(odd)))))

    def task_progress(self):
        """Box progress, but 1 only through the success predicate (spec 1.4): the right parcels in the box
        while one is not yet released and settled, or the balance is displaced, reads ``ALMOST`` (0.99), as
        stamps and the stand do."""
        value = self._box_progress()
        if value < 1.:
            return value
        return 1. if self._current_score()['success'] else ALMOST

    def collateral_checks(self):
        """Cheap state reads the base class turns into ``collateral``/``repaired`` events (spec 1.5)."""
        displacement = self.balance_displacement_m()
        boxed = self.parcels_in_box()
        odd = set(self.task_spec['odd_parcels_private'])
        wrong = [name for name in boxed if name not in odd]
        deepest = wrong[0] if wrong else ''
        above = float(len(boxed) - boxed.index(deepest) - 1) if wrong else 0.
        return {'balance_displaced': dict(bad=bool(displacement > BALANCE_RESET_TOL_M), object='balance',
                                          cost=1.),
                'wrong_parcel_boxed': dict(bad=bool(wrong), object=deepest, cost=above)}

    def _current_score(self, states=None):
        """Exactly the odd parcels in the box, released and settled, the balance within 2 cm of its
        reset pose. Anything else at the press is a wrong submit, so ``outcome`` is the outcome a press
        at this instant would produce. A parcel left on a pan is a flag (``<parcel>:left_on_pan``), not
        an error."""
        snapshot = VisionKitchenScene.evaluator_snapshot(self)
        odd = set(self.task_spec['odd_parcels_private'])
        boxed = self.parcels_in_box()
        on_pans = self.parcels_on_pans()
        displacements = self.balance_displacements_m()
        displacement = displacements['base']
        # Only the parcels must be released and settled: covers and occluders (role in task_spec['objects'])
        # are not part of the task and may lie wherever the policy left them.
        parcels = set(self.parcel_ids)
        errors = list(all_resting(dict(snapshot, objects={name: state for name, state in snapshot['objects'].items()
                                                          if name in parcels})))
        errors += [f'{name}:outside_box' for name in sorted(odd - set(boxed))]
        errors += [f'{name}:unwanted_in_box' for name in boxed if name not in odd]
        flags = [f'{name}:left_on_pan' for side in ('left', 'right') for name in on_pans[side]]
        if displacement > BALANCE_RESET_TOL_M:
            errors.append('balance:displaced')
        success = not errors
        raw = self._box_progress()
        return dict(success=success, outcome='success' if success else 'wrong_submit',
                    progress=raw if success else min(raw, ALMOST), errors=errors, flags=flags, box_contents=list(boxed),
                    odd_in_box=sorted(set(boxed) & odd), extra_in_box=sorted(set(boxed) - odd),
                    missing_odd=sorted(odd - set(boxed)), parcels_on_pans=on_pans,
                    balance_displacement_m=round(displacement, 4),
                    balance_displacements_m={k: round(v, 4) for k, v in sorted(displacements.items())},
                    robot_touches_balance=self._robot_touches_balance(),
                    beam_angle_deg=round(self.beam_angle_deg(), 3))

    def task_snapshot(self):
        return dict(beam_angle_deg=self.beam_angle_deg(),
                    pan_angles_deg={s: self.pan_angle_deg(s) for s in ('left', 'right')},
                    parcels_on_pans=self.parcels_on_pans(), robot_touches_balance=self._robot_touches_balance(),
                    parcels_in_box=self.parcels_in_box(),
                    balance_displacement_m=round(self.balance_displacement_m(), 4),
                    collateral=self.collateral_checks(),
                    parcel_masses_private=deepcopy(self.task_spec['parcel_masses_private']))

    # ---- gates (privileged helpers; never used by a policy) -----------------
    def set_parcel_pose(self, name, pos_world, yaw_world=0.):
        obj = self.objects[name]
        self.sim.data.set_joint_qpos(obj.joints[0], np.r_[np.asarray(pos_world, float),
                                                          math.cos(yaw_world / 2), 0., 0., math.sin(yaw_world / 2)])
        self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()

    def place_parcel_on_pan(self, name, side, dx=0., dy=0.):
        origin, rot = self.pan_frame(side)
        half = self.spec['parcel_half']
        pos = origin + rot @ np.array([dx, dy, half[2] + .008])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        self.set_parcel_pose(name, pos, yaw)

    def place_parcel_in_box(self, name, level):
        """Drop a parcel into the answer box at stack ``level`` (0 is the floor)."""
        origin, rot = self.box_frame()
        half = self.spec['parcel_half'][2]
        z = self.box['floor_z'] + (2 * level + 1) * half + .004
        pos = origin + rot @ np.array([0., 0., z])
        self.set_parcel_pose(name, pos, math.atan2(rot[1, 0], rot[0, 0]))

    def park_parcel(self, name):
        pose = self._reset_poses[name]
        self.sim.data.set_joint_qpos(self.objects[name].joints[0], np.asarray(pose, float))
        self.sim.data.set_joint_qvel(self.objects[name].joints[0], np.zeros(6))
        self.sim.forward()

    def park_all_parcels(self):
        for name in self.parcel_ids:
            self.park_parcel(name)

    def tick(self, n=1):
        action = np.zeros(self.action_dim)
        for _ in range(n):
            self.step(action)

    def parcel_on_pan_check(self, name, side):
        local = self.parcel_in_pan_local(name, side)
        inner = self.balance['pans'][side]['inner_half_xy']
        half = self.spec['parcel_half']
        return bool(abs(local[0]) + half[0] <= inner[0] + .003 and abs(local[1]) + half[1] <= inner[1] + .003
                    and half[2] - .006 <= local[2] <= half[2] + .012)

    # ---- teleport certificate (contract C3) --------------------------------
    def _lowest_point_z(self, body):
        """World z of the lowest point of a body's subtree, from the geoms' axis-aligned boxes."""
        model, d = self.sim.model._model, self.sim.data
        low = math.inf
        for gid in O._subtree_geoms(self.sim.model, body):
            aabb = np.asarray(model.geom_aabb[gid], float)
            rot = np.asarray(d.geom_xmat[gid], float).reshape(3, 3)
            centre = np.asarray(d.geom_xpos[gid], float) + rot @ aabb[:3]
            low = min(low, float(centre[2] - np.abs(rot[2]) @ aabb[3:]))
        return low

    def _cover_footprint_radius(self, name):
        rec = getattr(self, '_cover_records', {}).get(name) or {}
        if rec.get('kind') in O.PLACE_COVERS:
            return float(max(rec.get('half') or (.1, .1)))
        if rec.get('radius'):
            return float(rec['radius'])
        obj = self.objects.get(name)
        return float(obj.horizontal_radius) if obj is not None else .12

    def _live_keepouts(self, exclude=()):
        """Task-frame shapes of everything on the work top right now: the parcels where they stand, the
        balance, the box, Submit, the other extra objects and the decor."""
        d = self.sim.data
        shapes = list(parcel_keepouts(self.spec).values())
        for name in self.parcel_ids:
            c = O.world_to_frame_xy(self, d.body_xpos[self.obj_body_id[name]][:2])
            shapes.append(S.Disc((float(c[0]), float(c[1])), PARCEL_RADIUS))
        for name in getattr(self, 'extra_objects', []):
            if name in exclude:
                continue
            c = O.world_to_frame_xy(self, d.body_xpos[O.body_id(self, name)][:2])
            shapes.append(S.Disc((float(c[0]), float(c[1])), self._cover_footprint_radius(name)))
        return shapes + O.decor_shapes_on_work(self)

    def _cover_joint(self, name):
        obj = self.objects.get(name)
        return obj.joints[0] if obj is not None else name + '_joint'

    def clear_covers(self, rng=None):
        """Move every cover aside onto a free spot of the work top, the way the uncover skills would (the plate
        lifted off its pan, the cloche off its parcel): never off the surface, never onto a task object,
        never onto a pan. Returns certificate steps; settles the beam afterwards."""
        rng = rng if rng is not None else np.random.default_rng([int(self.instance.get('seed', 0)) & 0xFFFFFFFF, 0xC0DE])
        steps = []
        surface = O.work_surface_rect(self)
        top = float(self.work['top_z'])
        for name in self.cover_names():
            radius = self._cover_footprint_radius(name)
            region = S.Rect(surface.x0 + radius + .02, surface.x1 - radius - .02,
                            surface.y0 + radius + .02, surface.y1 - radius - .02)
            body = O.body_id(self, name)
            try:
                x, y, _ = S.scatter(rng, 1, region, radius, 0., keepouts=self._live_keepouts(exclude={name}),
                                    keepout_gap=.02, yaw_range=(0., 0.), max_tries=4000, attempts_per_prop=4000)[0]
            except ValueError:
                steps.append(dict(step=f'clear {name}', ok=False, detail='no free spot on the work top'))
                continue
            joint = self._cover_joint(name)
            pose = np.array(self.sim.data.get_joint_qpos(joint), float)
            pose[:2] = O.frame_to_world_xy(self, (x, y))
            pose[2] += (top + .005) - self._lowest_point_z(body)      # keep its orientation, land it from 5 mm
            self.sim.data.set_joint_qpos(joint, pose)
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
            self.sim.forward()
            self.tick(20)
            c = O.world_to_frame_xy(self, self.sim.data.body_xpos[body][:2])
            lowest = self._lowest_point_z(body) - top
            ok = bool(S.region_margin(S.Disc((float(c[0]), float(c[1])), radius), surface) >= 0. and -.01 <= lowest <= .03)
            steps.append(dict(step=f'clear {name}', ok=ok,
                              detail=dict(frame_xy=[round(float(c[0]), 3), round(float(c[1]), 3)],
                                          lowest_above_top_mm=round(1000 * lowest, 1))))
        if steps:
            self.tick(40)           # the beam swings back level once a plate is lifted off its pan
        return steps

    def teleport_solution(self):
        """Bring the scene to a solved state without the robot: move the covers aside, then drop the odd
        parcels (from wherever they stand: annex, behind an occluder, under a lifted cover) into the answer
        box one at a time, settling between them, and never touch the balance."""
        steps = []
        try:
            steps.extend(self.clear_covers())
            odd = list(self.task_spec['odd_parcels_private'])
            before = self.balance_displacement_m()
            for level, name in enumerate(odd):
                # The box walls stand 1 cm above one parcel (spec 4.2), so from level 1 up the
                # parcel balances on the stack with nothing to guide it and a 4 mm drop is
                # enough to tip it, or the one under it, out now and then. Re-seat the whole
                # stack and settle again rather than certifying a solved state the physics
                # happened to keep on the first try.
                for attempt in range(1, BOX_PLACE_ATTEMPTS + 1):
                    for lower, earlier in enumerate(odd[:level] + [name]):
                        self.place_parcel_in_box(earlier, lower)
                    self.tick(BOX_SETTLE_TICKS)
                    boxed = self.parcels_in_box()
                    if all(parcel in boxed for parcel in odd[:level + 1]):
                        break
                low, high = self.parcel_in_box_local(name)
                steps.append(dict(step=f'box {name} at level {level}', ok=name in boxed,
                                  detail=dict(box_contents=boxed, attempts=attempt,
                                              local_z_mm=[round(1000 * float(low[2]), 1),
                                                          round(1000 * float(high[2]), 1)])))
            self.tick(40)
            # The stack can still shed a parcel during the final settle, so re-seat it and
            # settle again until it holds; report the pass that was needed.
            for settle in range(1, BOX_PLACE_ATTEMPTS + 1):
                boxed = self.parcels_in_box()
                if all(parcel in boxed for parcel in odd):
                    break
                for level, name in enumerate(odd):
                    self.place_parcel_in_box(name, level)
                self.tick(BOX_SETTLE_TICKS + 40)
            steps.append(dict(step='stack settled', ok=all(parcel in self.parcels_in_box() for parcel in odd),
                              detail=dict(passes=settle, box_contents=self.parcels_in_box())))
            moved = self.balance_displacement_m() - before
            steps.append(dict(step='balance untouched', ok=bool(abs(moved) <= BALANCE_RESET_TOL_M),
                              detail=dict(displacement_mm=round(1000 * self.balance_displacement_m(), 2),
                                          beam_angle_deg=round(self.beam_angle_deg(), 3))))
            score = self._current_score()
            steps.append(dict(step='success predicate', ok=bool(score['success']),
                              detail=dict(errors=score['errors'], progress=round(score['progress'], 4),
                                          box_contents=score['box_contents'])))
        except Exception as error:                              # never raises (contract C3)
            steps.append(dict(step='teleport_solution', ok=False, detail=repr(error)))
        return steps

    def physics_gate(self, ticks_per_2s=None):
        """G1/G2 facts on the built CPU env: empty balance level; one odd vs one normal parcel tilts more than
        5 deg the right way; equal loads stay level with nothing sliding off; three parcels per pan reach the
        stop; parcels at rest after 100 zero-action ticks; a parcel settles inside the answer box."""
        ticks = ticks_per_2s or max(1, round(2. / self.control_timestep))
        report = dict(problems=[])
        problems = report['problems']
        spec = self.spec
        normal = [p['id'] for p in spec['parcels'] if not p['odd']]
        odd = [p for p in spec['parcels'] if p['odd']]
        # The instrument is measured with its pans clear: a plate on a pan (uncover) is lifted off first.
        report['covers_cleared'] = self.clear_covers()
        if not all(step['ok'] for step in report['covers_cleared']):
            problems.append('a cover could not be moved aside on the work top')
        self.tick(ticks)
        report['empty_after_2s_deg'] = round(self.beam_angle_deg(), 3)
        if abs(report['empty_after_2s_deg']) > .5:
            problems.append('empty balance not level')
        report['balance_displacement_mm'] = round(1000 * self.balance_displacement_m(), 2)
        if self.balance_displacement_m() > .005:
            problems.append('the balance does not settle back onto its reset pose')
        # Clearance: the props touch only the work top, each other and the Submit button.
        report['foreign_contacts'] = self.foreign_contacts()
        if report['foreign_contacts']:
            problems.append('props touch other fixtures or decor')
        base = np.asarray(self.robot_proprio()['base_position_world_m'], float)
        local = yaw_matrix(-self.work['yaw']) @ (base - np.asarray(self.work['center_world'], float))
        report['robot_base_in_frame'] = [round(float(v), 3) for v in local]
        if abs(local[0]) > .35 or not -.9 < local[1] < -.2:
            problems.append('robot does not stand in front of the task frame')
        # One odd parcel (left pan) against one normal parcel (right pan).
        first = odd[0]
        self.place_parcel_on_pan(first['id'], 'left')
        self.place_parcel_on_pan(normal[0], 'right')
        self.tick(ticks)
        angle = self.beam_angle_deg()
        expected_sign = -1. if first['direction'] == 'H' else 1.   # positive lowers the right pan
        report['odd_vs_normal'] = dict(odd=first['id'], direction=first['direction'],
                                       magnitude=first['delta_key'], angle_deg=round(angle, 3),
                                       left_on=self.parcel_on_pan_check(first['id'], 'left'),
                                       right_on=self.parcel_on_pan_check(normal[0], 'right'))
        if angle * expected_sign < 5. or not report['odd_vs_normal']['left_on'] or not report['odd_vs_normal']['right_on']:
            problems.append('one odd parcel does not tilt the beam more than 5 deg the right way')
        self.park_all_parcels()
        self.tick(ticks)
        # Equal loads stay level: up to three normal parcels per pan.
        per_pan = min(3, len(normal) // 2)
        placed = []
        slots = PAN_SLOTS_X[:per_pan] if per_pan == 3 else ((-.045, .045) if per_pan == 2 else (0.,))
        for side, group in (('left', normal[:per_pan]), ('right', normal[per_pan:2 * per_pan])):
            for name, dx in zip(group, slots):
                self.place_parcel_on_pan(name, side, dx=dx)
                placed.append((name, side))
        self.tick(ticks)
        report['equal_loads'] = dict(per_pan=per_pan, angle_deg=round(self.beam_angle_deg(), 3),
                                     on_pan={name: self.parcel_on_pan_check(name, side) for name, side in placed},
                                     pans=self.parcels_on_pans())
        if abs(report['equal_loads']['angle_deg']) > .5 or not all(report['equal_loads']['on_pan'].values()):
            problems.append(f'{per_pan} normal parcels per pan do not stay level on the pans')
        self.park_all_parcels()
        self.tick(ticks)
        # Capacity: three parcels side by side on one pan ride down to the stop without sliding off.
        trio = normal[:3]
        for name, dx in zip(trio, PAN_SLOTS_X):
            self.place_parcel_on_pan(name, 'left', dx=dx)
        self.tick(ticks)
        report['three_on_one_pan'] = dict(angle_deg=round(self.beam_angle_deg(), 3),
                                          on_pan={name: self.parcel_on_pan_check(name, 'left') for name in trio},
                                          pan_lean_deg=round(self.pan_angle_deg('left') + self.beam_angle_deg(), 2))
        if (report['three_on_one_pan']['angle_deg'] > -(self.DESIGN.stop_deg - .5)
                or not all(report['three_on_one_pan']['on_pan'].values())):
            problems.append('three parcels on one pan do not reach the stop or slide off')
        self.park_all_parcels()
        self.tick(ticks)
        # The answer box holds one parcel: dropped in, it settles on the floor, inside the column.
        probe = normal[0]
        self.place_parcel_in_box(probe, 0)
        self.tick(ticks)
        low, high = self.parcel_in_box_local(probe)
        report['box_probe'] = dict(parcel=probe, in_box=probe in self.parcels_in_box(),
                                   floor_gap_mm=round(1000 * float(low[2] - self.box['floor_z']), 2),
                                   top_below_wall_mm=round(1000 * float(self.box['floor_z']
                                                                        + self.box['wall_height'] - high[2]), 2))
        if not report['box_probe']['in_box'] or abs(report['box_probe']['floor_gap_mm']) > 3.:
            problems.append('a parcel dropped into the answer box does not settle on its floor')
        self.park_all_parcels()
        self.tick(ticks)
        # Rest: 100 zero-action ticks from the reset poses.
        self.tick(50)
        before = {name: np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]], float).copy() for name in self.parcel_ids}
        self.tick(50)
        moved = {name: float(np.linalg.norm(np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]], float) - before[name]))
                 for name in self.parcel_ids}
        speeds = {name: float(np.linalg.norm(self.sim.data.get_body_xvelp(self.objects[name].root_body)))
                  for name in self.parcel_ids}
        report['rest'] = dict(max_move_last_50_ticks_m=round(max(moved.values()), 5),
                              max_speed_m_s=round(max(speeds.values()), 5),
                              beam_deg=round(self.beam_angle_deg(), 3))
        if max(moved.values()) > .005 or max(speeds.values()) > .01 or abs(self.beam_angle_deg()) > .5:
            problems.append('parcels or balance not at rest after 100 zero-action ticks')
        score = self._current_score()
        report['score_after_gate'] = dict(success=score['success'], outcome=score['outcome'],
                                          progress=score['progress'], errors=score['errors'])
        if score['success'] or score['progress'] != 0.:
            problems.append('the score or the progress is not zero without any action')
        return not problems, report

    # ---- G3 hidden-state check --------------------------------------------
    def _project(self, point, camera):
        m, d = self.sim.model, self.sim.data
        cid = m.camera_name2id(camera)
        index = list(self.camera_names).index(camera)
        width, height = int(self.camera_widths[index]), int(self.camera_heights[index])
        rot = np.asarray(d.cam_xmat[cid], float).reshape(3, 3)
        local = rot.T @ (np.asarray(point, float) - np.asarray(d.cam_xpos[cid], float))
        depth = -local[2]
        if depth <= 1e-6:
            return None
        f = height / (2. * math.tan(math.radians(float(m.cam_fovy[cid])) / 2.))
        u = width / 2. + f * local[0] / depth
        v = height / 2. - f * local[1] / depth
        return dict(u=float(u), v=float(v), depth=float(depth), inside=bool(0 <= u < width and 0 <= v < height))

    def _first_hit(self, origin, target):
        m, d = self.sim.model._model, self.sim.data._data
        origin, target = np.asarray(origin, float), np.asarray(target, float)
        vec = target - origin
        dist = float(np.linalg.norm(vec))
        geom = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(m, d, origin, vec / dist, None, 1, -1, geom)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) if geom[0] >= 0 else None
        return dict(hit_geom=name, hit_distance=float(hit), target_distance=dist)

    def hidden_state_check(self, obs=None):
        """G3: policy cameras render; pointer tip, the answer box's interior floor and the Submit cap are
        inside a fixed camera's view and unoccluded along the line of sight (the pointer may be seen against
        its own tick plate, the box floor against its own walls); parcels are visually identical (same visual
        geom sizes, types, rgba and texture)."""
        report = dict(problems=[])
        problems = report['problems']
        if obs is not None:
            images = {}
            for cam in CAMERAS:
                key = cam + '_image'
                image = obs.get(key)
                images[cam] = None if image is None else dict(shape=list(image.shape), std=float(np.std(image)))
                if image is None or image.ndim != 3 or float(np.std(image)) < 1.:
                    problems.append(f'{cam} does not render')
            report['images'] = images
        d = self.sim.data
        targets = {
            'pointer': (np.asarray(d.site_xpos[self._pointer_site], float), ('balance_pointer', 'balance_tick_plate')),
            'answer_box': (np.asarray(d.site_xpos[self._box_site], float), ('answer_box_',)),
            'submit': (np.array(self.task_spec['submit_button']['position'], float), ('submit_cap',)),
        }
        fixed = [c for c in CAMERAS if 'agentview' in c]
        report['views'] = {}
        for name, (point, allowed) in targets.items():
            rows = {}
            for cam in fixed:
                proj = self._project(point, cam)
                cid = self.sim.model.camera_name2id(cam)
                ray = self._first_hit(np.asarray(d.cam_xpos[cid], float), point)
                if allowed is None:
                    clear = ray['hit_geom'] is None or ray['hit_distance'] >= ray['target_distance'] - .02
                else:
                    clear = (ray['hit_geom'] is None or ray['hit_distance'] >= ray['target_distance'] - .02
                             or any(ray['hit_geom'].startswith(a) for a in allowed))
                rows[cam] = dict(projection=proj, ray=ray, visible=bool(proj and proj['inside'] and clear))
            report['views'][name] = rows
            if not any(r['visible'] for r in rows.values()):
                problems.append(f'{name} is not visible in any fixed camera')
        m = self.sim.model
        signatures = {}
        for name in self.parcel_ids:
            parts = []
            for gid in range(m.ngeom):
                gname = m.geom_id2name(gid) or ''
                if not gname.startswith(name + '_'):
                    continue
                mat = int(m.geom_matid[gid])
                tex = int(m.mat_texid[mat][0]) if mat >= 0 and np.ndim(m.mat_texid[mat]) else (int(m.mat_texid[mat]) if mat >= 0 else -1)
                parts.append((int(m.geom_type[gid]), tuple(np.round(m.geom_size[gid], 6)), int(m.geom_group[gid]),
                              tuple(np.round(m.geom_rgba[gid], 4)), tuple(m.tex_width[tex] if tex >= 0 else ()),
                              int(m.tex_height[tex]) if tex >= 0 else -1))
            signatures[name] = tuple(sorted(parts))
        distinct = {sig for sig in signatures.values()}
        report['parcel_visual_signatures_distinct'] = len(distinct)
        report['parcel_masses_distinct'] = len({round(float(m.body_mass[self.obj_body_id[n]]), 4) for n in self.parcel_ids})
        if len(distinct) != 1:
            problems.append('parcels are not visually identical')
        problems.extend(self._observability_check(report))
        return not problems, report

    def _observability_check(self, report):
        """Spec 2.2 at reset: the hidden parcels are in no policy camera while the other parcels, the balance
        and the covers are (observe.hidden_check, a fail-closed render diff; an occluder may itself stand
        outside the cameras' field, the parcel behind it is then hidden all the same and still has its
        viewpoint); every look entry has a reachable viewpoint (observe.look_reachable, plus the annex
        region's reach gate); every cover's reach gate passed at build. Without a renderer a scene with hidden
        parcels fails closed; a visible scene renders every parcel when it can and is otherwise judged by the
        projection checks above."""
        problems = []
        record = self.task_spec.get('observability') or {}
        hidden = [name for name in record.get('hidden_bodies', []) if name in self.parcel_ids]
        extras = list(getattr(self, 'extra_objects', []))
        report['hidden_parcels'] = hidden
        report['extra_objects'] = extras
        renderer = bool(getattr(self, 'has_offscreen_renderer', False))
        if hidden or extras:
            if not renderer:
                problems.append('hidden parcels cannot be verified without the offscreen renderer')
            else:
                visible = [n for n in self.parcel_ids if n not in hidden] + [self.balance['base_body']] + self.cover_names()
                check = O.hidden_check(self, hidden, visible=visible)
                occluders = [n for n in extras if n not in visible]
                seen = O.visible_in_cameras(self, occluders)['pixels'] if occluders else {}
                report['render_check'] = dict(passed=check['passed'], hidden_ok=check['hidden_ok'],
                                              visible_ok=check['visible_ok'], pixels=dict(check['pixels'], **seen),
                                              deterministic=check['deterministic'], flicker=check['flicker'],
                                              unstable_pixels=check.get('unstable_pixels'))
                if not check['passed']:
                    seen = [n for n, ok in check['hidden_ok'].items() if not ok]
                    unseen = [n for n, ok in check['visible_ok'].items() if not ok]
                    problems.append(f'render check: hidden but seen {seen}; expected but unseen {unseen}; '
                                    f"deterministic {check['deterministic']}")
        elif renderer:
            check = O.visible_in_cameras(self, list(self.parcel_ids))
            unseen = [n for n, ok in check['visible'].items() if not ok]
            report['render_check'] = dict(passed=not unseen and all(check['deterministic'].values()),
                                          pixels=check['pixels'], deterministic=check['deterministic'])
            if unseen:
                problems.append(f'parcels in no policy camera at the visible level: {unseen}')
            if not all(check['deterministic'].values()):
                problems.append('the renderer is not deterministic')
        looks = {}
        for placed in record.get('annex', []):
            looks[placed['object']] = dict(mode='annex', region=placed['region'], stance=placed['stance'])
        for placed in record.get('occluders', []):
            looks[placed['target']] = dict(mode='occluder', occluder=placed['name'],
                                           relocated=placed.get('relocated') is not None)
        if looks:
            regions = {r['name']: r for r in O.annex_regions(self, all_regions=True, at_build=False)}
            for name, info in looks.items():
                reach = O.look_reachable(self, name)
                info['look_reachable'] = dict(passed=reach['passed'], stance=reach['stance'], reasons=reach['reasons'])
                if not reach['passed']:
                    problems.append(f"{name}: no reachable viewpoint ({'; '.join(reach['reasons'])})")
                if info['mode'] == 'annex':
                    region = regions.get(info['region'])
                    info['region_reach'] = None if region is None else region['reach']
                    if region is None or not region['reach']['passed']:
                        problems.append(f"{name}: annex region {info['region']} fails the reach gate at reset")
        report['look'] = looks
        covers = {}
        for cover in record.get('covers', []):
            if not cover.get('built'):
                continue
            passed = bool((cover.get('reach') or {}).get('passed'))
            covers[cover['name']] = dict(kind=cover['kind'], covers=cover.get('object') or cover.get('place'),
                                         reach_passed=passed)
            if not passed:
                problems.append(f"{cover['name']}: the {cover['kind']} reach gate failed")
        report['covers'] = covers
        report['plate_tilt'] = getattr(self, '_plate_tilt', None)
        return problems
