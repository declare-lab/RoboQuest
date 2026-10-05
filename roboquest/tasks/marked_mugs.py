"""Marked Mugs (tasks-v0 task 4, RoboQuest v1) as a RoboCasa kitchen task.

``vessel_count`` identical vessels - **all mugs or all bowls**, drawn once per
instance (spec 4.6) - stand scattered over the front of the task frame at random
yaws, each holding two 14 mm balls of its own colour pair and carrying a colour
label decal on its underside. One flat colour pad per vessel sits in a loose row
at the back of the frame - nominal slots, then per-pad xy jitter and a small yaw,
in a shuffled colour order - and the red Submit button is front right. Poses come
from :mod:`roboquest.scatter`, which rejects and re-draws until
nothing overlaps, so no two instances share an arrangement.

The label is the hidden state: an upright vessel hides its own decal, and the
only way to read it is to lift the vessel and tilt it toward a camera. Past about
108 degrees (measured in the study room) both balls fall out of a mug; a bowl is
wide and shallow, so it spills far sooner and the careful strategy is to lift it
straight up and look from below.

v1 design:

* ``mug_count`` became ``vessel_count``; six pads do not fit the 1.00 m frame
  under the v0 spacing rules (see :func:`pad_row_fit`), so the levels are the
  documented fallback **3, 4, 5**;
* the vessel type is a 50/50 nuisance draw from the ``structure`` stream, the
  vessel's tint a draw from ``materials``;
* the ball-to-support **rolling** friction drops tenfold, so a spilled ball runs
  about 20 cm instead of a couple of centimetres;
* the goal is framed by the scope and clutter sentences of spec 1.3, and the
  class carries the protocol hooks ``BUDGET_TICKS``, :meth:`task_progress`,
  :meth:`collateral_checks` and :meth:`teleport_solution`;
* every vessel starts at least 10 cm back from the counter's front edge
  (spec 3), in dev as well as eval.

The props, their tuned contact parameters and the placement rule come from the
0.80 m study-room task (``active_bench.inspect_cups_*``); this module re-derives
every placement in the task frame on RoboCasa's 0.92 m work surface.

The structure class counts the vessels whose **nearest pad by distance** already
carries their own label colour, so it describes the scattered layout the policy
actually sees rather than a column index.
"""
from copy import deepcopy
import itertools
import math

import mujoco
import numpy as np

from roboquest import observe, scatter as S
from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz
from roboquest.mugs import hiding
from roboquest.mugs.geometry import BALL_COLORS, BALL_RADIUS, LABEL_COLORS, MUG_REST_CLEARANCE, PAD_FIT_RADIUS, PAD_HALF_XY, VESSEL_COLORS, VESSEL_KINDS, VESSELS, append_support_ball_contacts, append_vessel_objects, vessel_spec
from roboquest.mugs.rules import classify_placements, progress as rule_progress, spilled_balls
from roboquest.stand import helpers as H
from roboquest.harness.contract import CAMERAS

CONTRACT_VERSION = 'marked-mugs-kitchen-v3'
# Spec 1.3: scope sentence + the v1 middle sentence + clutter sentence. Identical for
# every instance of a vessel type and count, and it names no colour and no hidden state.
GOAL_SCOPE_WITH_COUNT = 'There are {count} {plural} on the counters in this kitchen.'
GOAL_SCOPE = 'There are {plural} on the counters in this kitchen.'
GOAL_MIDDLE = ('Each one has a coloured label on its bottom. Put every one upright on the pad of the '
               'same colour, with its own two balls inside, then press Submit.')
GOAL_CLUTTER = 'Other objects on the counter are not part of the task; you may move them if needed.'

# --- task-frame layout (metres; x along the front, y away from the robot) ----
# Chosen so the whole footprint fits every layout in registry.DEFAULT_LAYOUT_POOL:
# (1.00, .62) is the largest of the probed footprints valid on layouts 1, 2, 7 and 9
# as well as 4, 10, 13 and 17 (probe/footprint_probe.py, 2026-09-20).
FOOTPRINT = (1.00, .62)
# Pads: a loose back row. Nominal slots at PAD_SPACING, then each pad is drawn uniformly
# from a +/-3 cm box about its slot with a +/-10 deg yaw, rejected while it would touch
# another pad, Submit or the frame edge. Five pads of 16 x 12.5 cm at 10 deg span 0.92 m
# of the 1.00 m frame; six span 1.11 m and do not fit (see pad_row_fit).
PAD_Y = .14
PAD_SPACING = {3: .300, 4: .250, 5: .200}
PAD_JITTER_XY = .03
PAD_JITTER_YAW = math.radians(10.)
PAD_GAP = .006                 # clear space between two pads
# Vessels: scattered over the front of the frame at any yaw, clear of the pads and of
# Submit. The footprint is the vessel's yaw-independent bounding circle (mug .0575,
# bowl .060).
VESSEL_REGION_FRACTION = 2. / 3.
# Spec 3: rolling objects start at least 10 cm from the counter's front edge. The task
# frame already starts FOOTPRINT_FRONT_MARGIN (3 cm) behind the front of the work
# region, so 7 cm more of the frame is kept clear of vessels - in dev as well as eval,
# since the generator does not know the split.
VESSEL_FRONT_MARGIN = .07
COUNTER_FRONT_CLEARANCE = .10
VESSEL_GAP = .02               # clear space between two vessels
VESSEL_KEEPOUT_GAP = .015      # clear space from a pad or the Submit button
VESSEL_YAW_RANGE = (-math.pi, math.pi)
SUBMIT_LOCAL = (.40, -.25)
SUBMIT_HALF_XY = (.048, .043)
SUBMIT_RADIUS = math.hypot(*SUBMIT_HALF_XY)     # yaw-invariant bound: what a helper keeps clear of
PAD_RADIUS = math.hypot(*PAD_HALF_XY)           # ... and the pad's circumradius, whatever its jitter yaw
# Helper containers (the stand's rule brought to the vessels). One or two native
# RoboCasa bowls or rimmed trays stand on the work surface with nothing in them. Reading the label
# under a vessel means lifting it, and a shallow bowl spills its balls as soon as it tilts, so the
# careful way is to take the balls out first - and a helper is somewhere to put them. Nothing names
# them in the goal, nothing guarantees one is in reach, and using one is the robot's own idea.
# The vocabulary, the footprint geometry and the placement rule are `stand.helpers`, unchanged and
# unmoved; only the keep-out list below is this task's own.
HELPER_PREFIX = 'helper'
HELPER_MARGIN_M = H.HELPER_MARGIN_M             # 2 cm to every pad, vessel, cover candidate and Submit
HELPER_FRONT_CLEARANCE_M = H.FRONT_CLEARANCE_M  # 2 cm of the footprint to the frame's front edge
# The frame's own front edge. The stand measures its helpers against the *surface* edge, 3 cm further
# forward; spec 3's 10 cm rule is about things that roll, and a container standing still with a ball
# parked in it does not (stand.helpers), so the frame edge plus 2 cm - 5 cm of counter -
# is the rule here and it binds before the footprint test does.
FRAME_FRONT_Y = -FOOTPRINT[1] / 2
HELD_LIFT_M = .05                               # a ball this far over the top, in the robot's hand, is held
ROBOT_PREFIXES = ('robot', 'gripper', 'mobilebase')
PHYSICS_TIMESTEP = .001        # small parts: the balls were tuned at 1 kHz
RESCATTER_TRIES = 8            # draws for the cloche's vessel that keep every nearest pad
SETTLE_TICKS = 100             # G2
TELEPORT_SETTLE_TICKS = 50     # C3 certificate: 2.5 s of physics after the state edits


def _row(count, spacing):
    """Symmetric column centres for ``count`` props spaced ``spacing`` apart."""
    return [round((index - (count - 1) / 2) * spacing, 6) for index in range(count)]


def _yaw_of(quat):
    """Yaw of a wxyz quaternion (the props are only ever turned about z)."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def frame_rect(footprint=FOOTPRINT):
    """The whole task footprint as a scatter rectangle."""
    return S.rect((0., 0.), (footprint[0] / 2, footprint[1] / 2))


def pad_row_fit(count, pad_half=PAD_HALF_XY, yaw=PAD_JITTER_YAW, gap=PAD_GAP, width=FOOTPRINT[0]):
    """Can ``count`` pads stand side by side in the frame under the v0 spacing rules?

    The pads jitter by +/-3 cm in y but their y extent is 15 cm, so two neighbours always
    overlap in y and the row is decided by x alone: every pad needs its yawed x extent
    plus ``gap``. Reported for the record at the maximum jitter yaw and at yaw 0.
    """
    ex_yaw = S.extent(pad_half, yaw)[0]
    ex_flat = S.extent(pad_half, 0.)[0]
    needed = count * 2 * ex_yaw + (count - 1) * gap
    flat = count * 2 * ex_flat + (count - 1) * gap
    return dict(count=int(count), pad_half_xy_m=list(pad_half), yaw_deg=round(math.degrees(yaw), 3),
                pad_extent_x_m=round(ex_yaw, 5), row_width_m=round(needed, 5),
                row_width_at_zero_yaw_m=round(flat, 5), frame_width_m=float(width),
                slack_m=round(width - needed, 5), slack_at_zero_yaw_m=round(width - flat, 5),
                jitter_needed_m=round(2 * PAD_JITTER_XY, 5), fits=bool(needed <= width),
                fits_with_jitter=bool(needed + 2 * PAD_JITTER_XY <= width))


def pad_slots(count):
    """Per-pad jitter boxes: a +/-3 cm square about each nominal back-row slot."""
    if count not in PAD_SPACING:
        raise ValueError(f'no pad spacing for {count} vessels; levels are {sorted(PAD_SPACING)}')
    return [S.rect((x, PAD_Y), (PAD_JITTER_XY, PAD_JITTER_XY)) for x in _row(count, PAD_SPACING[count])]


def vessel_region(footprint=FOOTPRINT):
    """Where the vessels are scattered: the front of the frame, kept back from its front edge."""
    half = (footprint[0] / 2, footprint[1] / 2)
    return S.Rect(-half[0], half[0], -half[1] + VESSEL_FRONT_MARGIN,
                  -half[1] + VESSEL_REGION_FRACTION * footprint[1])


def vessel_radius(spec):
    """The bounding disc of this instance's vessel type."""
    return float(vessel_spec(str(spec.get('vessel', 'mug')))['xy_radius'])


def submit_shape(spec=None):
    xy = (spec or {}).get('placement', {}).get('submit', SUBMIT_LOCAL)
    return S.rect(xy, SUBMIT_HALF_XY)


def pad_shapes(spec):
    """The pads as oriented boxes in the task frame."""
    yaws = spec.get('pad_yaw') or [0.] * len(spec['pad_xy'])
    return [S.shape_at(PAD_HALF_XY, xy, yaw) for xy, yaw in zip(spec['pad_xy'], yaws)]


def nearest_pads(vessel_xy, pad_xy):
    """Index of the pad nearest each vessel's start position (Euclidean, task-frame xy).

    Ties go to the lower pad index, so this is a function of the spec alone.
    """
    pads = np.asarray(pad_xy, float)
    return [int(np.argmin(np.linalg.norm(pads - np.asarray(xy, float), axis=1))) for xy in vessel_xy]


def fixed_points(spec):
    """Vessels whose nearest pad *by distance* already carries their own label colour."""
    nearest = spec.get('nearest_pad') or nearest_pads(spec['vessel_xy'], spec['pad_xy'])
    return int(sum(1 for label, index in zip(spec['labels'], nearest)
                   if label == spec['pad_colors'][int(index)]))


# ---- helper containers ---------------------------------------
def helper_shape(entry, xy, yaw_deg):
    """A helper's task-frame footprint: the disc a bowl's mesh sweeps, a tray's rotated rectangle.

    The same two cases :func:`stand.helpers.footprint_gap` measures, as a scatter shape, so the
    observability keep-outs and the goal-spot check see exactly what the placement rule rejected for.
    """
    half = tuple(float(v) for v in entry['half'])
    if entry['kind'] == 'bowl':
        return S.Disc((float(xy[0]), float(xy[1])), max(half))
    return S.Box((float(xy[0]), float(xy[1])), half, math.radians(float(yaw_deg)))


def helper_names(spec):
    """The scene name of each helper, in spec order."""
    return [f'{HELPER_PREFIX}_{index}' for index in range(len(spec.get('helpers') or []))]


def plan_keepouts(spec):
    """Discs for what the hidden plan may still build, as far as the spec knows it at mint.

    The uncover candidates are known exactly: the large cloche is centred on its vessel, and a flat
    cover on a pad is a disc about the pad centre (``hiding.flat_cover_options`` gives every asset the
    build could draw, and the widest of them bounds the rest). A helper clears all of them, not only
    the one drawn, because the build's candidate walk may fall through to a later one.

    The ``look`` entries are **not** here: the occluder's spot and the annex region are chosen at build
    time from the kitchen, which the mint cannot see. Those are kept off a helper on the build side,
    where the helper footprints join the keep-outs handed to :func:`observe.apply_observability`.
    """
    plan = spec.get('hidden_plan') or {}
    out = []
    for name, _kind in plan.get('object_candidates') or []:
        index = int(str(name).rsplit('_', 1)[-1]) if str(name).startswith('vessel_') else None
        if index is None or index >= len(spec['vessel_xy']):
            continue
        out.append((f'cloche over {name}', list(spec['vessel_xy'][index]), hiding.CLOCHE_RADIUS))
    pads = {f'pad_{colour}': xy for colour, xy in zip(spec['pad_colors'], spec['pad_xy'])}
    for name, kind in plan.get('place_candidates') or []:
        xy = pads.get(name)
        if xy is None:
            continue
        options = hiding.flat_cover_options(kind, PAD_HALF_XY)
        reach = max((max(half) if kind in hiding.ROUND_COVERS else math.hypot(*half)
                     for half, _turn in options), default=0.)
        if reach > 0.:
            out.append((f'{kind} over {name}', list(xy), reach))
    return out


def helper_keepouts(spec):
    """Task-frame discs a helper must clear by :data:`HELPER_MARGIN_M`, as ``(name, xy, radius)``.

    Every pad (its circumradius, so the jitter yaw does not matter), every vessel (its own xy radius:
    a bowl is wider than a mug), Submit, and everything :func:`plan_keepouts` knows about the hidden
    plan. The helpers already drawn are appended to this list as they are drawn, exactly as the stand
    does it.
    """
    out = [(f'pad_{colour}', list(xy), PAD_RADIUS)
           for colour, xy in zip(spec['pad_colors'], spec['pad_xy'])]
    try:
        radius = vessel_radius(spec)
    except ValueError:
        # A tampered vessel type is somebody else's problem to report; keep the widest verified
        # vessel here so the container rule is still checked instead of raising over it.
        radius = max(float(entry['xy_radius']) for entry in VESSELS.values())
    out += [(f'vessel_{index}', list(xy), radius) for index, xy in enumerate(spec['vessel_xy'])]
    out.append(('submit', list(spec['placement']['submit']), SUBMIT_RADIUS))
    out += plan_keepouts(spec)
    return out


def draw_helpers(rng, spec, attempts=H.DRAW_ATTEMPTS, counts=H.HELPER_COUNTS):
    """Draw one or two helpers uniformly over the frame by rejection sampling; never raises.

    The count, the container and the pose all come from ``rng`` - the instance's ``poses`` stream,
    consumed last - and :func:`stand.helpers.placement_problems` decides every proposal, with the roll
    band switched off (there is no stand here; the band's ``stand_xy`` is Submit, which the keep-out
    list already covers with its own radius). Two helpers are two different assets.

    Unlike the stand, a seed is **never** refused for lack of helper room: room for one container is
    the floor, room for none is allowed and recorded. The instance is a marked-vessel instance with or
    without a place to park the balls. Returns ``(helpers, stats)``.
    """
    entries = H.catalogue()
    half_x, half_y = FOOTPRINT[0] / 2, FOOTPRINT[1] / 2
    keepouts = list(helper_keepouts(spec))
    submit = [float(v) for v in spec['placement']['submit']]
    wanted = int(counts[int(rng.integers(len(counts)))])
    drawn, tried, legal = [], 0, 0
    for index in range(wanted):
        pool = [entry for entry in entries if entry['asset'] not in {h['asset'] for h in drawn}]
        entry = pool[int(rng.integers(len(pool)))]
        for _ in range(int(attempts)):
            tried += 1
            xy = [float(rng.uniform(-half_x, half_x)), float(rng.uniform(-half_y, half_y))]
            yaw = float(rng.uniform(-180., 180.))
            if H.placement_problems(xy, yaw, entry['half'], kind=entry['kind'], stand_xy=submit,
                                    keepouts=keepouts, footprint=FOOTPRINT, front_edge_y=FRAME_FRONT_Y,
                                    front_clearance=HELPER_FRONT_CLEARANCE_M, margin=HELPER_MARGIN_M,
                                    band=0.):
                continue
            legal += 1
            drawn.append(dict(kind=entry['kind'], asset=entry['asset'], scale=entry['scale'],
                              xy=xy, yaw=yaw))
            keepouts.append((f'{HELPER_PREFIX}_{index}', xy, entry['radius']))
            break
    return drawn, dict(wanted=wanted, placed=len(drawn), short=wanted - len(drawn),
                       draws=tried, legal=legal)


def helper_problems(spec):
    """Everything wrong with ``spec['helpers']``; empty means the placement is legal.

    The same checks as ``wobbly_stand.helper_problems`` - a container from the verified catalogue, at
    its verified scale, with the recorded kind, a yaw in range and a legal pose - over this task's own
    keep-outs. The one difference is the count: ``helpers_wanted`` records what the draw asked for and
    the list may be shorter, because a scene with no room for a container is still a valid scene.
    """
    helpers = spec.get('helpers')
    wanted = spec.get('helpers_wanted')
    if wanted not in H.HELPER_COUNTS:
        return [f'helpers_wanted {wanted!r} is not one of {H.HELPER_COUNTS}']
    if not isinstance(helpers, list) or len(helpers) > int(wanted):
        return [f'helpers must be a list of at most the {wanted} container(s) drawn']
    problems, keepouts = [], list(helper_keepouts(spec))
    assets = [helper.get('asset') for helper in helpers if isinstance(helper, dict)]
    if len(set(assets)) != len(assets):
        problems.append('two helpers share one asset; the second container must differ from the first')
    for index, helper in enumerate(helpers):
        label = f'{HELPER_PREFIX}_{index}'
        if not isinstance(helper, dict):
            problems.append(f'{label} is not a container record')
            continue
        entry = H.entry_for(helper.get('asset'))
        if entry is None:
            problems.append(f'{label}: unverified container asset {helper.get("asset")!r}')
            continue
        if helper.get('kind') != entry['kind']:
            problems.append(f'{label}: kind {helper.get("kind")!r} is not {entry["kind"]!r}')
        if abs(float(helper.get('scale', 0.)) - entry['scale']) > 1e-9:
            problems.append(f'{label}: scale {helper.get("scale")} is not the verified {entry["scale"]}')
        if not -180. <= float(helper.get('yaw', 999.)) <= 180.:
            problems.append(f'{label}: yaw {helper.get("yaw")} out of range')
        xy = helper.get('xy')
        if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
            problems.append(f'{label}: xy missing')
            continue
        xy = [float(xy[0]), float(xy[1])]
        problems += H.placement_problems(
            xy, float(helper.get('yaw', 0.)), entry['half'], kind=entry['kind'], label=label,
            stand_xy=[float(v) for v in spec['placement']['submit']], keepouts=keepouts,
            footprint=FOOTPRINT, front_edge_y=FRAME_FRONT_Y,
            front_clearance=HELPER_FRONT_CLEARANCE_M, margin=HELPER_MARGIN_M, band=0.)
        keepouts.append((label, xy, entry['radius']))
    return problems


class MarkedMugs(RoboQuestKitchen):
    TASK_NAME = 'marked_mugs'
    CONTRACT_VERSION = CONTRACT_VERSION
    # v2: vessel_count replaces mug_count, bowls join mugs, the goal is re-framed and the
    # rolling friction is cut tenfold. v3 (wave 2): the universal `observability` axis,
    # realised in the scene (spec 2.2).
    GENERATOR_VERSION = 'v4'
    # Spec 2.1 asks for 4, 5, 6 and allows the fallback 3, 4, 5 when six pads do not fit
    # the pad row. They do not: six pads need 1.106 m of the 1.00 m frame (pad_row_fit).
    # Balanced cells are size x observability; `look+uncover` is dev only (contract C2).
    FACTORS = {'vessel_count': (3, 4, 5), 'observability': hiding.LEVELS}
    DEV_ONLY_LEVELS = hiding.DEV_ONLY_LEVELS
    FOOTPRINT = FOOTPRINT
    # Protocol C1: 20 Hz ticks, provisional (no oracle for this task yet), the v0 horizon.
    BUDGET_TICKS = 12000
    # Compositional eval rule: one (vessel_count, fixed-point) cell per size
    # level is held out, so every level and both structure classes still appear in dev.
    HOLDOUT_KEYS = ('v3-f0', 'v4-f1+', 'v5-f0')

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        """Spec 1.3: scope sentence (count only when ``show_count``), middle, clutter."""
        plural = vessel_spec(str(spec.get('vessel', 'mug')))['plural']
        count = int(spec.get('vessel_count', 0))
        scope = (GOAL_SCOPE_WITH_COUNT.format(count=count, plural=plural) if spec.get('show_count')
                 else GOAL_SCOPE.format(plural=plural))
        return ' '.join((scope, GOAL_MIDDLE, GOAL_CLUTTER))

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        count = int(factors['vessel_count'])
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        # Nuisance, drawn once for the whole instance (spec 4.6): mugs or bowls, 50/50.
        kind = VESSEL_KINDS[int(structure.integers(0, len(VESSEL_KINDS)))]
        vessel = vessel_spec(kind)
        # Nuisance: which colours are in play and in which order the pads stand.
        palette = materials.permutation(list(LABEL_COLORS))[:count].tolist()
        pad_colors = materials.permutation(palette).tolist()
        # Nuisance: ball colours, drawn from an independent stream so the pair never predicts the label.
        ball_palette = materials.permutation(list(BALL_COLORS))[:2 * count].tolist()
        ball_colors = [[ball_palette[2 * index], ball_palette[2 * index + 1]] for index in range(count)]
        # Nuisance (spec 4.9): the vessel's own tint, a muted crockery colour over the asset texture.
        vessel_color = str(materials.permutation(sorted(VESSEL_COLORS))[0])
        # Nuisance: the placement. The pads jitter about their nominal back-row slots
        # (shuffled colour order, so a colour never marks a slot), then the vessels are
        # scattered over the front of the frame clear of the pads and of Submit. Both draws
        # raise ValueError if they cannot be satisfied, which rejects the seed.
        frame = frame_rect()
        pads = S.scatter(poses, count, pad_slots(count), PAD_HALF_XY, PAD_GAP,
                         keepouts=[submit_shape()], yaw_range=(-PAD_JITTER_YAW, PAD_JITTER_YAW),
                         region_holds='center', bounds=frame)
        pad_xy = [[x, y] for x, y, _ in pads]
        pad_yaw = [yaw for _, _, yaw in pads]
        keepouts = [S.shape_at(PAD_HALF_XY, xy, yaw) for xy, yaw in zip(pad_xy, pad_yaw)]
        vessels = S.scatter(poses, count, vessel_region(), float(vessel['xy_radius']), VESSEL_GAP,
                            keepouts=keepouts + [submit_shape()], keepout_gap=VESSEL_KEEPOUT_GAP,
                            yaw_range=VESSEL_YAW_RANGE)
        vessel_xy = [[x, y] for x, y, _ in vessels]
        vessel_yaw = [yaw for _, _, yaw in vessels]
        # Structure: the label permutation class, over the *scattered* layout. "Fixed point"
        # means a vessel whose nearest pad by distance already carries its label colour.
        # The class is drawn first and a permutation of that class chosen uniformly, so both
        # classes stay common (a uniform permutation would be a derangement only ~37% of the time).
        nearest = nearest_pads(vessel_xy, pad_xy)
        want_fixed = bool(structure.integers(0, 2))
        wanted, other = [], []
        for order in itertools.permutations(range(count)):
            labels = [pad_colors[index] for index in order]
            hits = sum(1 for label, near in zip(labels, nearest) if label == pad_colors[near])
            (wanted if (hits > 0) == want_fixed else other).append(labels)
        options = wanted or other      # `other` is unreachable for count >= 2; kept as a guard
        labels = options[int(structure.integers(0, len(options)))]
        # Observability (spec 2.2): the level is the second reported factor; which vessels are
        # hidden, which pad is covered and which uncover flavour is used are nuisance drawn from
        # the same poses stream, last, so the structure draws, the hold-out key and the goal text
        # are bit-identical to the same seed at another level (`visible` draws none). The draw sees
        # the placement it must fit into, so a cover candidate this scene cannot hold is dropped
        # before the build instead of costing a seed at G1.
        level = str(factors.get('observability', 'visible'))
        radius = float(vessel['xy_radius'])
        names = [f'vessel_{index}' for index in range(count)]
        shapes = {name: S.Disc((xy[0], xy[1]), radius) for name, xy in zip(names, vessel_xy)}
        shapes.update({f'pad_{colour}': shape for colour, shape in zip(pad_colors, keepouts)})
        shapes['submit'] = submit_shape()
        cover_places = {f'pad_{colour}': dict(xy=list(xy), half=list(PAD_HALF_XY), yaw=float(yaw))
                        for colour, xy, yaw in zip(pad_colors, pad_xy, pad_yaw)}

        def rescatter(index, tries=RESCATTER_TRIES):
            """Draw one vessel again with the cloche's own footprint reserved around it.

            Only that vessel moves: a fresh scatter of the whole front row would almost always
            send some vessel to another pad, and `nearest_pad` decides the label permutation
            class and the hold-out key, which the level may not touch. A draw that changes any
            nearest pad is discarded and the draw retried; the others stand where they were, so
            the layout the placement gate measures is the same one minus a single vessel.
            """
            # The other vessels are keep-outs inflated by the difference between the two gaps, so
            # the vessel-to-vessel clearance the gate measures is still VESSEL_GAP.
            others = [S.Disc((xy[0], xy[1]), radius + VESSEL_GAP - VESSEL_KEEPOUT_GAP)
                      for other, xy in enumerate(vessel_xy) if other != index]
            region = hiding.vessel_regions(1, vessel_region(), radius, 0)[0]
            for _ in range(int(tries)):
                try:
                    drawn = S.scatter(poses, 1, region, hiding.CLOCHE_RADIUS, VESSEL_GAP,
                                      keepouts=keepouts + [submit_shape()] + others,
                                      keepout_gap=VESSEL_KEEPOUT_GAP, yaw_range=VESSEL_YAW_RANGE)
                except ValueError:
                    return None
                x, y, yaw = drawn[0]
                moved = [list(xy) for xy in vessel_xy]
                moved[index] = [x, y]
                if nearest_pads(moved, pad_xy) != nearest:
                    continue
                vessel_xy[:] = moved
                vessel_yaw[index] = yaw
                return {name: tuple(xy) for name, xy in zip(names, moved)}
            return None

        qualifying = [name for name, label, near in zip(names, labels, nearest)
                      if label == pad_colors[near]]
        hidden, hidden_plan = hiding.draw(poses, level, names, qualifying, shapes=shapes,
                                          places=cover_places, rescatter=rescatter, bounds=frame)
        spec = dict(observability=level, hidden=hidden, hidden_plan=hidden_plan,
                    vessel_count=count, vessel=kind, vessel_color=vessel_color,
                    labels=labels, pad_colors=pad_colors, ball_colors=ball_colors,
                    vessel_xy=vessel_xy, vessel_yaw=vessel_yaw, pad_xy=pad_xy, pad_yaw=pad_yaw,
                    nearest_pad=nearest,
                    fixed_points=int(sum(1 for label, near in zip(labels, nearest)
                                         if label == pad_colors[near])),
                    placement=dict(submit=list(SUBMIT_LOCAL)),
                    footprint=list(FOOTPRINT), physics_timestep_s=PHYSICS_TIMESTEP)
        # Helper containers: the very last draw of `poses`, after the pads, the
        # vessels and the hidden plan, so the rest of a seed's scene is untouched by their arrival
        # and a hidden-plan candidate is a keep-out by the time a helper looks for room. `helpers`
        # may be shorter than `helpers_wanted`, or empty: a scene with nowhere to stand a container
        # is still a marked-vessel scene and the seed is never refused for it (the shortfall is the
        # `helpers_short` field, which the cpu gate reports and the registry summary can total).
        helpers, draw_stats = draw_helpers(poses, spec)
        spec.update(helpers=helpers, helpers_wanted=int(draw_stats['wanted']),
                    helpers_short=int(draw_stats['short']))
        return spec

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        count = int(spec['vessel_count'])
        if count not in cls.FACTORS['vessel_count']:
            problems.append(f'vessel_count {count} is not a factor level')
        if spec.get('vessel') not in VESSEL_KINDS:
            problems.append(f"vessel {spec.get('vessel')!r} is not a vessel type")
        if spec.get('vessel_color') not in VESSEL_COLORS:
            problems.append(f"vessel_color {spec.get('vessel_color')!r} is not a vessel tint")
        for key, length in (('labels', count), ('pad_colors', count), ('ball_colors', count),
                            ('vessel_xy', count), ('vessel_yaw', count), ('pad_xy', count),
                            ('pad_yaw', count), ('nearest_pad', count)):
            if len(spec[key]) != length:
                problems.append(f'{key} has {len(spec[key])} entries, expected {length}')
        if sorted(spec['labels']) != sorted(spec['pad_colors']):
            problems.append('labels and pads do not use the same colour set')
        if len(set(spec['pad_colors'])) != count:
            problems.append('pad colours are not distinct')
        if any(colour not in LABEL_COLORS for colour in spec['pad_colors']):
            problems.append('unknown label colour')
        flat = [colour for pair in spec['ball_colors'] for colour in pair]
        if len(set(flat)) != 2 * count:
            problems.append('ball colours are not all distinct')
        if any(colour not in BALL_COLORS for colour in flat):
            problems.append('unknown ball colour')
        if list(spec.get('nearest_pad', [])) != nearest_pads(spec['vessel_xy'], spec['pad_xy']):
            problems.append('nearest_pad disagrees with the vessel and pad positions')
        if spec.get('fixed_points') != fixed_points(spec):
            problems.append('fixed_points disagrees with the label permutation')
        if any(abs(float(yaw)) > PAD_JITTER_YAW + 1e-9 for yaw in spec.get('pad_yaw', [])):
            problems.append('a pad yaw is outside the jitter range')
        problems.extend(cls._observability_problems(spec))
        problems.extend(helper_problems(spec))
        return problems

    @classmethod
    def _observability_problems(cls, spec):
        """Spec 2.2 schema: a known level, entries observe can realise, and a candidate plan
        whose fallbacks name real vessels and pads."""
        level = spec.get('observability', 'visible')
        if level not in hiding.ALL_LEVELS:
            return [f'observability {level!r} is not one of {hiding.ALL_LEVELS}']
        count = int(spec['vessel_count'])
        vessels = [f'vessel_{index}' for index in range(count)]
        pads = [f'pad_{colour}' for colour in spec['pad_colors']]
        problems = list(observe.validate_hidden(spec, objects=set(vessels) | set(pads)))
        plan = spec.get('hidden_plan') or {}
        flavour = plan.get('uncover_flavour')
        wants_cover = level in ('uncover', hiding.DEV_ONLY_LEVEL)
        if wants_cover and flavour not in ('object', 'place'):
            problems.append(f'uncover_flavour {flavour!r} is neither object nor place')
        if not wants_cover and flavour is not None:
            problems.append(f'level {level!r} draws no uncover flavour')
        for name, kind in plan.get('object_candidates') or []:
            if name not in vessels or kind not in observe.OBJECT_COVERS:
                problems.append(f'object candidate {name}/{kind} is not a vessel under an object cover')
        for name, kind in plan.get('place_candidates') or []:
            if name not in pads or kind not in observe.PLACE_COVERS:
                problems.append(f'place candidate {name}/{kind} is not a pad under a flat cover')
        if [name for name, _ in plan.get('look') or []] != [e['object'] for e in spec.get('hidden', [])
                                                            if e['mode'] in hiding.LOOK_MODES]:
            problems.append('hidden_plan.look disagrees with the look entries')
        rescatter = plan.get('rescatter')
        if rescatter is not None:
            name, moved = rescatter
            if name not in vessels:
                problems.append(f'the reserved vessel {name!r} is not a vessel')
            if sorted(moved) != sorted(vessels):
                problems.append('hidden_plan.rescatter does not cover every vessel')
            elif any(not np.allclose(spec['vessel_xy'][int(key.split('_')[1])], xy, atol=1e-9)
                     for key, xy in moved.items()):
                problems.append('the vessels were re-scattered for the cloche but vessel_xy did not follow')
        if wants_cover and not (plan.get('object_candidates') or plan.get('place_candidates')):
            problems.append('no uncover candidate survives the placement')
        if wants_cover and flavour in ('object', 'place'):
            candidates = [tuple(candidate) for candidate in plan.get(f'{flavour}_candidates') or []]
            entry = next((e for e in spec.get('hidden', []) if e['mode'] == 'cover'), None)
            if not candidates:
                problems.append(f'uncover_flavour {flavour!r} has no candidate of its own')
            elif entry is not None and (entry['object'], entry['cover']) != candidates[0]:
                problems.append('the cover entry is not the first candidate of the drawn flavour')
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """Geometry gate, re-deriving exactly what the sampler enforced: pads pairwise clear and
        clear of Submit, vessels pairwise clear and clear of every pad and of Submit, every vessel
        inside its region (which is kept back from the counter's front edge, spec 3), every prop
        inside the footprint, every vessel base fitting a pad with 5 mm to spare."""
        report, problems = {}, []
        count = int(spec['vessel_count'])
        radius = vessel_radius(spec)
        frame, region = frame_rect(), vessel_region()
        submit = submit_shape(spec)
        pads = pad_shapes(spec)
        pad_poses = [(xy[0], xy[1], yaw) for xy, yaw in zip(spec['pad_xy'], spec['pad_yaw'])]
        vessel_poses = [(xy[0], xy[1], yaw) for xy, yaw in zip(spec['vessel_xy'], spec['vessel_yaw'])]
        pad_names = [f'pad_{colour}' for colour in spec['pad_colors']]
        vessel_names = [f'vessel_{index}' for index in range(count)]
        pad_check = S.check(pad_poses, PAD_HALF_XY, bounds=frame, min_gap=PAD_GAP,
                            keepouts=[submit], keepout_gap=PAD_GAP, names=pad_names,
                            keepout_names=['submit'])
        vessel_check = S.check(vessel_poses, radius, bounds=region, min_gap=VESSEL_GAP,
                               keepouts=pads + [submit], keepout_gap=VESSEL_KEEPOUT_GAP,
                               names=vessel_names, keepout_names=pad_names + ['submit'])
        problems.extend(pad_check['problems'])
        problems.extend(vessel_check['problems'])
        report['vessel'] = str(spec.get('vessel'))
        report['vessel_radius_m'] = radius
        report['min_pad_gap_m'] = pad_check['min_pair_gap_m']
        report['min_vessel_gap_m'] = vessel_check['min_pair_gap_m']
        report['min_vessel_pad_gap_m'] = vessel_check['min_keepout_gap_m']
        report['min_vessel_region_margin_m'] = vessel_check['min_bounds_margin_m']
        report['pad_row_fit'] = pad_row_fit(count)
        fit = [float(v) for v in (np.asarray(PAD_HALF_XY) - PAD_FIT_RADIUS)]
        report['pad_placement_tolerance_m'] = [round(v, 5) for v in fit]
        if min(fit) < .005:
            problems.append('a vessel base does not fit its pad with 5 mm to spare')
        # Spec 3: the front-most vessel surface, measured back from the frame's front edge.
        front = min(xy[1] for xy in spec['vessel_xy']) - radius + FOOTPRINT[1] / 2
        report['vessel_front_margin_m'] = round(float(front), 5)
        if front < VESSEL_FRONT_MARGIN - 1e-6:
            problems.append('a vessel starts closer than the front margin to the frame edge')
        margins = [S.region_margin(shape, frame) for shape in pads]
        margins += [S.region_margin(S.shape_at(radius, xy), frame) for xy in spec['vessel_xy']]
        margins.append(S.region_margin(submit, frame))
        report['min_footprint_margin_m'] = round(min(margins), 5)
        if min(margins) < 0:
            problems.append('a prop sticks out of the footprint')
        submit_clear = [S.clearance(S.shape_at(radius, xy), submit) for xy in spec['vessel_xy']]
        submit_clear += [S.clearance(shape, submit) for shape in pads]
        report['min_submit_clearance_m'] = round(min(submit_clear), 5)
        if min(submit_clear) <= 0:
            problems.append('the Submit button overlaps a vessel or a pad')
        # The parking containers. Their legality is `helper_problems` (the same rule
        # the draw sampled against); what the gate *reports* is how the draw went, so the registry
        # summary can total the shortfall per vessel_count and per observability level.
        helpers = spec.get('helpers') or []
        problems.extend(helper_problems(spec))
        report['helpers'] = [helper.get('asset') for helper in helpers]
        report['helper_count'] = len(helpers)
        report['helpers_wanted'] = int(spec.get('helpers_wanted', 0))
        report['helpers_short'] = int(spec.get('helpers_short', 0))
        gaps = []
        for helper in helpers:
            entry = H.entry_for(helper.get('asset'))
            if entry is None:
                continue
            gaps += [H.footprint_gap(helper['xy'], helper['yaw'], entry['half'], entry['kind'], xy) - radius
                     for _name, xy, radius in helper_keepouts(spec)]
        report['min_helper_gap_m'] = round(min(gaps), 5) if gaps else None
        report['problems'] = problems
        report['fixed_points'] = fixed_points(spec)
        report['split_key'] = cls.split_key(spec)
        return not problems, report

    @classmethod
    def split_key(cls, spec):
        hits = fixed_points(spec)
        return f"v{int(spec['vessel_count'])}-f{'0' if hits == 0 else '1+'}"

    # ---- scene protocol -----------------------------------------------------
    def _decor_frame_boxes(self):
        """RoboCasa's counter decor and standing appliances as task-frame boxes.

        Every work-surface yaw seen on the exercised layouts is a multiple of 90
        degrees, so re-boxing a world AABB in the frame is exact there; on a surface
        at another angle it is conservative.
        """
        rotate = yaw_matrix(-self.work['yaw'])
        origin = np.asarray(self.work['center_world'], float)
        boxes = []
        for decor in self._decor_boxes(self.work['top_z']):
            corners = np.array([[decor['x0'], decor['y0'], 0.], [decor['x1'], decor['y0'], 0.],
                                [decor['x1'], decor['y1'], 0.], [decor['x0'], decor['y1'], 0.]])
            local = np.array([rotate @ (corner - origin) for corner in corners])
            boxes.append(dict(name=decor['name'], x0=float(local[:, 0].min()), x1=float(local[:, 0].max()),
                              y0=float(local[:, 1].min()), y1=float(local[:, 1].max())))
        return boxes

    def _decor_near_props(self):
        """Decor boxes that reach a prop's own box; a hint only, the contact gate decides.

        ``get_bbox_points`` is generous (a paper-towel holder's box reaches a pad on
        layout 1 with no collision geometry anywhere near it), so this is recorded,
        not enforced.
        """
        radius = vessel_radius(self.spec)
        props = {name: (info['frame_xy'], (radius, radius))
                 for name, info in self.geometry['vessels'].items()}
        props.update({f'pad_{colour}': (pad['center_xy'], S.extent(PAD_HALF_XY, pad.get('yaw_rad', 0.)))
                      for colour, pad in self.geometry['pads'].items()})
        props['submit'] = (self.spec['placement']['submit'], SUBMIT_HALF_XY)
        near = []
        for decor in self._decor_frame_boxes():
            for name, (centre, half) in props.items():
                box = dict(x0=centre[0] - half[0], x1=centre[0] + half[0],
                           y0=centre[1] - half[1], y1=centre[1] + half[1])
                if self._boxes_overlap(box, decor, margin=0.):
                    near.append(f"{name}/{decor['name']}")
        return sorted(set(near))

    def build_task(self):
        self.place_work_frame()
        pads = self.add_frame_body('marked_mugs_pads')
        self.geometry = append_vessel_objects(self.model.worldbody, self.model.asset, pads, self.spec,
                                              self.frame_to_world, self.work['yaw'], self.work['center_world'])
        self.add_submit_button(self.spec['placement']['submit'])
        self._add_helpers()
        kind = self.geometry['vessel']
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL,
            vessel=kind, vessel_count=self.spec['vessel_count'], vessel_color=self.spec.get('vessel_color'),
            budget_ticks=self.BUDGET_TICKS,
            frame_origin_world=self.work['center_world'],
            frame_yaw=self.work['yaw'], pads=self.geometry['pads'],
            vessels_private={name: dict(label=info['label'], column=info['column'],
                                        original_balls=info['original_balls'],
                                        original_ball_colors=info['original_ball_colors'])
                             for name, info in self.geometry['vessels'].items()},
            balls={name: dict(color=info['color'], original_vessel=info['original_vessel'])
                   for name, info in self.geometry['balls'].items()},
            decor_frame_boxes=self._decor_frame_boxes(), decor_near_props=self._decor_near_props(),
            helpers={name: dict(asset=ref['asset'], kind=ref['entry']['kind'], scale=ref['scale'],
                                local_xy=list(ref['local_xy']), yaw_deg=ref['yaw_deg'], role='helper')
                     for name, ref in self.helpers.items()},
            helpers_wanted=int(self.spec.get('helpers_wanted', 0)),
            helpers_short=int(self.spec.get('helpers_short', 0)),
            helper_rule=dict(margin_m=HELPER_MARGIN_M, inset=H.HELPER_INSET, held_lift_m=HELD_LIFT_M,
                             front_clearance_m=HELPER_FRONT_CLEARANCE_M,
                             rule='Empty parking containers. A ball resting in one, or held '
                                  'by the gripper, is not spilled; a ball loose on the counter is. They are '
                                  'not named in the goal, not part of the success rule and not reachable '
                                  'by guarantee.'),
            placement='Vessels scattered over the front of the frame at random yaws, kept at least '
                      f'{COUNTER_FRONT_CLEARANCE:.2f} m back from the counter edge; pads in a loose back row '
                      '(jittered position and yaw, shuffled colour order); nothing aligned, nothing '
                      'overlapping (roboquest.scatter).',
            hidden_state='The colour label under each vessel; an upright vessel hides it from every policy camera.',
            scoring='First physical Submit freezes the score: every vessel upright on the pad of its own label '
                    'colour with real upward support, released and still, holding exactly its two original '
                    'balls, which are settled; no submission fails.')
        self.asset_evidence.update(
            vessels=(f"{self.spec['vessel_count']} clones of RoboCasa's objaverse mug_1 with the original visual "
                     f"mesh, material, texture and all sixteen hollow collision hulls including the handle"
                     if kind == 'mug' else
                     f"{self.spec['vessel_count']} clones of RoboCasa's objaverse bowl_11 visual mesh with a "
                     f"procedural hollow collision shell (16 thin tilted boxes tracing the interior cone plus a "
                     f"floor cylinder on the foot ring); the asset's own convex hulls are dropped")
                     + f" ({self.geometry['source_asset']})",
            vessel_colour=f"Per-instance tint {self.spec.get('vessel_color')} over the asset's own texture "
                          '(materials stream, spec 4.9); functional colours (pads, balls, Submit) are fixed',
            balls='Free smooth spheres, r=.014 m, two per vessel, distinct colours, native rigid contact only',
            labels='Flat cylinder decals under the vessel floor: visual only, no collision, no mass',
            pads='Flat 16 x 12.5 cm colour boxes, 4 mm proud of the counter top, fixed in the task '
                 'frame at a jittered position and a small yaw',
            helpers='Native RoboCasa bowls and rimmed trays, unmodified and empty, from the vocabulary '
                    'verified to hold a released ball (roboquest.stand.helpers); placed at random over '
                    'the frame, clear of every pad, vessel, cover candidate and Submit',
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged')
        self._apply_observability()

    def _add_helpers(self):
        """The parking containers, native RoboCasa assets with ``role='helper'``.

        The mint drew them clear of the pads, the vessels, Submit, the hidden plan's cover candidates
        and each other inside the task frame; the kitchen is only known here, so the decor check
        happens now and a hit is a build failure (G1) that refuses the seed and lets the mint walk on.
        They are extras, not task objects: they are not in ``geometry['objects']``, so the scoring, the
        progress, ``object_states`` and the settle gate pass over them exactly as they pass over a
        cover or an occluder, while the physics does not.
        """
        spec, top_z = self.spec, self.work['top_z']
        decor = self._decor_boxes(top_z)
        self.helpers = {}
        for name, helper in zip(helper_names(spec), spec.get('helpers') or []):
            entry = H.entry_for(helper['asset'])
            world = self.frame_to_world([helper['xy'][0], helper['xy'][1], 0.])
            hits = H.decor_hits(world[:2], entry['radius'], decor)
            if hits:
                raise ValueError(f'{self.TASK_NAME}: {name} ({helper["asset"]}) would stand on {hits} '
                                 f"(layout {(self.instance or {}).get('layout_id')})")
            yaw = float(self.work['yaw']) + math.radians(float(helper['yaw']))
            quat = yaw_quat_wxyz(yaw)
            obj = self.add_native(name, helper['asset'], [float(world[0]), float(world[1])], top_z,
                                  scale=float(helper['scale']),
                                  quat_xyzw=[float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
            half = [float(obj.size[0]) / 2, float(obj.size[1]) / 2]
            rim = float(obj.top_offset[2])
            if max(abs(half[0] - entry['half'][0]), abs(half[1] - entry['half'][1])) > .001:
                raise ValueError(f'{self.TASK_NAME}: {helper["asset"]} is {half} m half wide, not the '
                                 f'verified {list(entry["half"])} m; re-run the roll-in verification')
            self.helpers[name] = dict(entry=entry, asset=helper['asset'], scale=float(helper['scale']),
                                      half=half, rim_local_z=rim, yaw=yaw, yaw_deg=float(helper['yaw']),
                                      local_xy=[float(helper['xy'][0]), float(helper['xy'][1])],
                                      world_xy=[float(world[0]), float(world[1])])
            self.task_spec['objects'][name].update(
                role='helper', kind=entry['kind'], asset=helper['asset'], scale=float(helper['scale']),
                local_xy=list(helper['xy']), yaw_deg=float(helper['yaw']), half_xy_m=half,
                rim_local_z_m=rim, holds_ball=True,
                note='parking container; not named in the goal, not part of the success rule')
            self.asset_evidence[name].update(
                role='helper', kind=entry['kind'],
                verified='holds a released ball: dropped at the centre above the rim with 5 cm/s of drift '
                         'in four directions, still inside and at rest after a 100-tick settle')

    def helper_shapes(self):
        """The built helpers as task-frame footprints, by name: the keep-outs the build adds."""
        return {name: helper_shape(ref['entry'], ref['local_xy'], ref['yaw_deg'])
                for name, ref in getattr(self, 'helpers', {}).items()}

    # ---- observability (spec 2.2, contract C6) ------------------------------
    def _prop_shapes(self):
        """Task-frame footprint of every prop this task owns, by name: what a cover, an occluder or a
        relocated vessel must keep clear of. Two balls stand inside each vessel, so the vessel's own
        disc already covers them."""
        radius = vessel_radius(self.spec)
        shapes = {name: S.Disc((float(info['frame_xy'][0]), float(info['frame_xy'][1])), radius)
                  for name, info in self.geometry['vessels'].items()}
        shapes.update({f'pad_{colour}': S.shape_at(pad['half_size_xy'], pad['center_xy'],
                                                   float(pad.get('yaw_rad', 0.)))
                       for colour, pad in self.geometry['pads'].items()})
        shapes['submit'] = submit_shape(self.spec)
        # An empty parking container is a prop like any other. Putting it here is what
        # keeps the occluder, the relocated vessel and both flavours of cover off it, in the candidate
        # walk (`hiding.resolve`), in `observe.apply_observability` and in the C3 cover set-down.
        shapes.update(self.helper_shapes())
        return shapes

    def _cover_places(self):
        """Places a flat cover may hide (spec 2.2: one colour pad), task frame. A pad stands 4 mm proud
        of the counter, so the cover rests on the pad, not on the counter."""
        return {f'pad_{colour}': dict(xy=list(pad['center_xy']), half=list(pad['half_size_xy']),
                                      yaw=float(pad.get('yaw_rad', 0.)), mode='push',
                                      support_z=float(self.work['top_z']) + float(pad['floor_z']))
                for colour, pad in self.geometry['pads'].items()}

    def _vessel_objects(self):
        """Geometry overrides observe needs for the mesh vessels: bounding disc and height."""
        shape = vessel_spec(str(self.geometry['vessel']))
        height = float(shape['rim_z']) - float(shape['bottom_z'])
        return {name: dict(radius=float(shape['xy_radius']), height=height)
                for name in self.geometry['vessels']}

    def _observability_rng(self):
        seed = int((getattr(self, 'instance', None) or {}).get('seed', 0)) & 0xFFFFFFFF
        return np.random.default_rng([seed, observe.OBSERVE_SALT])

    def _task_body_elements(self):
        """The XML body element of every prop, by name (build time: there is no sim yet)."""
        wanted = set(self.geometry['objects'])
        return {body.get('name'): body for body in self.model.worldbody.iter('body')
                if body.get('name') in wanted}

    @staticmethod
    def _body_pose(element):
        pos = [float(v) for v in (element.get('pos') or '0 0 0').split()]
        quat = [float(v) for v in (element.get('quat') or '1 0 0 0').split()]
        return np.asarray(pos, float), np.asarray(quat, float)

    def _body_poses(self):
        return {name: self._body_pose(element) for name, element in self._task_body_elements().items()}

    def _follow_moved_vessels(self, before):
        """Deliverable 3: when observability annexes or relocates a vessel, its two balls go with it.

        observe moves the vessel body alone; the balls are separate free bodies that started inside it,
        so the vessel's own rigid transform is applied to them and the recorded poses follow.
        """
        elements = self._task_body_elements()
        moved = {}
        for name, info in self.geometry['vessels'].items():
            pos, quat = self._body_pose(elements[name])
            was_pos, was_quat = before[name]
            if np.allclose(pos, was_pos, atol=1e-12) and np.allclose(quat, was_quat, atol=1e-12):
                continue
            turn = _yaw_of(quat) - _yaw_of(was_quat)
            rotation = yaw_matrix(turn)
            balls = {}
            for ball in info['original_balls']:
                element = elements[ball]
                ball_pos, _ = self._body_pose(element)
                after = pos + rotation @ (ball_pos - was_pos)
                element.set('pos', ' '.join(f'{float(v):.9g}' for v in after))
                element.set('quat', ' '.join(f'{float(v):.9g}' for v in yaw_quat_wxyz(turn)))
                frame_xy = observe.world_to_frame_xy(self, after[:2])
                record = self.geometry['balls'][ball]
                record.update(source_position=[float(v) for v in after],
                              frame_xy=[float(frame_xy[0]), float(frame_xy[1])],
                              moved_by_observability=True)
                balls[ball] = [round(float(v), 4) for v in after]
            frame_xy = observe.world_to_frame_xy(self, pos[:2])
            info.update(frame_xy=[float(frame_xy[0]), float(frame_xy[1])], frame_z=float(pos[2]),
                        frame_yaw_rad=float(_yaw_of(quat) - float(self.work['yaw'])),
                        moved_by_observability=True)
            moved[name] = dict(xy_world=[round(float(v), 4) for v in pos[:2]], balls=balls)
        return moved

    def _apply_observability(self):
        """Realise ``spec['observability']`` at the end of the build (brief deliverable 3).

        The candidate walk in :mod:`..mugs.hiding` fixes the uncover entry first, so the entry handed
        to :func:`observe.apply_observability` is one this scene can build; anything else that fails
        (an annex that holds nothing, a cover that cannot be built, a push gate that fails) leaves a
        problem in the record and raises here, which rejects the seed at G1.
        """
        spec = self.spec
        shapes = self._prop_shapes()
        places = self._cover_places()
        objects = self._vessel_objects()
        surface = observe.work_surface_rect(self)
        relocate = S.Rect(surface.x0 + .06, surface.x1 - .06, max(surface.y0, 0.), surface.y1 - .06)
        floor = observe.floor_map(self)
        entries, notes = hiding.resolve(self, spec, objects=objects, places=places, keepouts=shapes,
                                        bounds=surface, floor=floor)
        # A cover always overlaps what it hides, so the covered prop drops out of the keep-outs (observe
        # requires it); everything else the task owns stays in, which is what keeps a cover, an occluder
        # or a relocated vessel off the other pads and off Submit.
        # The cover must hide the whole place, corners included: `observe.cover_place` scales a cover
        # until its bounding box overhangs `place_half`, so the place it is given is the inflated one
        # (`hiding.cover_half`) the mint-time candidate walk measured.
        covered = next((entry for entry in entries
                        if entry['mode'] == 'cover' and entry.get('cover') in hiding.PLACE_COVER_KINDS), None)
        if covered is not None and covered['object'] in places:
            place = places[covered['object']]
            places = dict(places, **{covered['object']: dict(
                place, half=list(hiding.cover_half(covered['cover'], place['half'])))})
        hidden_names = {entry['object'] for entry in entries if entry['mode'] == 'cover'}
        keepouts = [shape for name, shape in shapes.items() if name not in hidden_names]
        before = self._body_poses()
        # The annex is chosen in world coordinates on another surface, so the task-frame keep-outs
        # above do not reach it; the helpers go over in world shapes as well (nothing
        # observability places may land on or in a container).
        annex_world = [S.Disc((ref['world_xy'][0], ref['world_xy'][1]), float(ref['entry']['radius']))
                       for ref in getattr(self, 'helpers', {}).values()]
        record = observe.apply_observability(self, dict(spec, hidden=entries), objects=objects, places=places,
                                             keepouts=keepouts, bounds=surface, rng=self._observability_rng(),
                                             floor=floor, occluder_relocate=relocate,
                                             annex_keepouts_world=annex_world)
        record.update(notes, resolved=deepcopy(entries))
        record['moved_vessels'] = self._follow_moved_vessels(before)
        record['goal_spot_clashes'] = self._goal_spot_clashes(record, shapes)
        if record['problems'] or record['goal_spot_clashes']:
            # When the walk found no realisable candidate it leaves the drawn entry in place and the
            # build tries it anyway, so the problem above reads as if one cover misbehaved. Say what
            # actually happened: every candidate this scene offered, and why each was refused.
            exhausted = [] if notes.get('flavour_realised') else [
                'no uncover candidate is realisable: ' + ', '.join(
                    f"{row['object']}/{row['cover']} ({'; '.join(row['reasons']) or 'refused'})"
                    for row in notes.get('candidates_tried') or []) or 'the draw offered none']
            raise ValueError(f"{self.TASK_NAME}: observability {record['level']} not realisable on layout "
                             f"{(self.instance or {}).get('layout_id')} style {(self.instance or {}).get('style_id')}: "
                             + '; '.join(exhausted + list(record['problems'])
                                         + list(record['goal_spot_clashes']))[:600])
        return record

    def _goal_spot_clashes(self, record, shapes, gap=.004):
        """Nothing the observability wiring added or moved may stand on a pad or on Submit unless it is
        the recorded uncover entry: a plate on the blue pad is the task, a plate on it by accident is a
        defect. Returns human-readable strings; a non-empty list fails the build."""
        spots = {name: shape for name, shape in shapes.items() if name.startswith('pad_')}
        spots['submit'] = shapes['submit']
        # A helper is a spot too: a cloche dropped over a bowl of parked balls, or an occluder
        # standing in one, would take away the only place the balls can go.
        spots.update({name: shape for name, shape in shapes.items()
                      if name.startswith(f'{HELPER_PREFIX}_')})
        extras = []
        for cover in record.get('covers', []):
            if not cover.get('built'):
                continue
            if cover.get('place'):
                extras.append((cover['name'], hiding.cover_shape(cover['kind'], cover['place_xy'],
                                                                 cover['half'], float(cover['yaw_frame'])),
                               {cover['place']}))
            else:
                xy = observe.world_to_frame_xy(self, cover['xy_world'])
                extras.append((cover['name'], S.Disc((float(xy[0]), float(xy[1])), float(cover['radius'])), set()))
        for occluder in record.get('occluders', []):
            extras.append((occluder['name'],
                           S.Box(tuple(occluder['frame_xy']), (occluder['size'][0] / 2, occluder['size'][1] / 2),
                                 float(occluder['frame_yaw'])), set()))
            moved = occluder.get('relocated')
            if moved:
                xy = observe.world_to_frame_xy(self, moved['to_xy_world'])
                extras.append((occluder['target'], S.Disc((float(xy[0]), float(xy[1])),
                                                          float(occluder['target_radius'])), set()))
        problems = []
        for name, shape, allowed in extras:
            for spot, keep in spots.items():
                if spot not in allowed and S.clearance(shape, keep) < gap:
                    problems.append(f'{name} stands on {spot}')
        return problems

    def _cover_geom_names(self):
        """Geom names of each built flat cover, by the pad it covers: what the pad clearance pass is
        allowed to find over that pad (and nothing else is)."""
        model = self.sim.model
        out = {}
        for cover in (self.task_spec.get('observability') or {}).get('covers', []):
            if not cover.get('built') or not cover.get('place'):
                continue
            root = observe.body_id(self, cover['name'])
            names = set()
            for gid in range(model.ngeom):
                body = int(model.geom_bodyid[gid])
                while body > 0 and body != root:
                    body = int(model.body_parentid[body])
                if body == root:
                    names.add(model.geom_id2name(gid) or '')
            out[cover['place']] = names
        return out

    def _decor_boxes(self, top_z):
        """Also treat small appliances standing on the work top as occupied space.

        The base class only knows RoboCasa ``Accessory`` objects, so a toaster or a
        coffee machine that stands on the counter is invisible to the frame placement
        and a front-row vessel is built inside it (measured on layout 13 / style 4:
        20 mm of penetration). Anything whose bounding box *starts* within 5 cm below
        and 10 cm above the top and rises above it stands on the surface; wall
        cabinets and the base cabinets below start elsewhere and are left out.
        """
        boxes = super()._decor_boxes(top_z)
        known = {box['name'] for box in boxes}
        for name, fixture in sorted(self.fixtures.items()):
            if name in known or fixture is self.work_fixture:
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            low, high = float(points[:, 2].min()), float(points[:, 2].max())
            if not (top_z - .05 <= low <= top_z + .10 and high > top_z + .01):
                continue
            boxes.append(dict(name=name, x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                              y0=float(points[:, 1].min()), y1=float(points[:, 1].max())))
        return boxes

    def _support_surfaces(self):
        """Fixtures a vessel and its balls may stand on: the work surface and every annex surface the
        observability wiring used. At the ``look`` level a vessel starts on another counter, so its top
        slabs need the same tuned ball contacts as the work top (contract C3)."""
        surfaces = [self.work_fixture.name]
        for entry in (self.task_spec.get('observability') or {}).get('annex', []):
            fixture = str(entry.get('fixture') or '')
            if fixture and fixture not in surfaces:
                surfaces.append(fixture)
        return surfaces

    def _support_geom_names(self):
        """Top collision tiles of every support surface: what the vessels and balls stand on."""
        prefix = self.work_fixture.name + '_top'
        prefixes = tuple(f'{fixture}_top' for fixture in self._support_surfaces())
        names = []
        for geom in self.model.root.iter('geom'):
            name = geom.get('name') or ''
            if not name.startswith(prefixes):
                continue
            if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
                continue
            names.append(name)
        if not any(name.startswith(prefix) for name in names):
            raise ValueError(f'no collision top geoms named {prefix}* on {self.work_fixture.name}')
        return names

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        option = self.model.root.find('option')
        if option is None:
            import xml.etree.ElementTree as ET
            option = ET.SubElement(self.model.root, 'option')
        option.set('timestep', str(PHYSICS_TIMESTEP))
        self.task_spec['ball_contacts'] = append_support_ball_contacts(
            self.model.root, self.geometry, self._support_geom_names())
        self.task_spec['physics'] = dict(timestep_s=PHYSICS_TIMESTEP, public_tick_s=.05,
                                         note='Small parts: the ball contact model was tuned at 1 kHz.')
        self._align_robot_base_with_the_frame()

    def _align_robot_base_with_the_frame(self):
        """Stand the robot in front of the task frame, not the middle of a long counter.

        RoboCasa anchors the base on the *fixture*, while the base class slides the
        task frame sideways to dodge decor - 0.60 m on layout 13, which left an end
        pad outside both agentview cameras at reset. Only the component along the
        work surface is changed, so RoboCasa keeps its own standing distance,
        orientation and collision retries.
        """
        anchor = np.asarray(self.init_robot_base_pos_anchor, float)
        along = yaw_matrix(self.work['yaw'])[:, 0]
        shift = float(np.dot(np.asarray(self.work['center_world'], float)[:2] - anchor[:2], along[:2]))
        anchor[:2] += shift * along[:2]
        self.init_robot_base_pos_anchor = anchor.tolist()
        self.task_spec['robot_base_anchor_shift_m'] = round(shift, 4)

    def initialize_time(self, control_freq):
        super().initialize_time(control_freq)
        # Keep each public tick at 50 ms while resolving ball contacts at 1 kHz.
        self.model_timestep = PHYSICS_TIMESTEP
        self.sim.model.opt.timestep = self.model_timestep

    def setup_task_references(self):
        model = self.sim.model
        self.obj_body_id = dict(getattr(self, 'obj_body_id', {}))
        self._object_geoms = {}
        for name, info in self.geometry['objects'].items():
            self.obj_body_id[name] = model.body_name2id(name)
            self._object_geoms[name] = [model.geom_name2id(geom) for geom in info['geom_names']]
        self._pad_geoms = {model.geom_name2id(pad['floor_geom']): colour
                           for colour, pad in self.geometry['pads'].items()}
        self._label_geoms = {name: model.geom_name2id(info['label_geom'])
                             for name, info in self.geometry['vessels'].items()}
        # The parking containers. The inset footprint, the rim height and a live contact
        # are the "ball parked in this one" test (roboquest.stand.helpers). Like the covers and the
        # occluder they stay out of `geometry['objects']` and out of `_object_geoms`, so no scoring,
        # progress or settle loop sees them (RoboCasa does register every built body in
        # `obj_body_id`; the only loop over that dict is `_pose_states`, read back by ball name).
        for name, ref in getattr(self, 'helpers', {}).items():
            bid = model.body_name2id(self.objects[name].root_body)
            ref['bid'] = bid
            ref['gids'] = {gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == bid
                           and (model.geom_contype[gid] or model.geom_conaffinity[gid])}
            ref['inset'] = [H.HELPER_INSET * ref['half'][0], H.HELPER_INSET * ref['half'][1]]
        # G1, as early as the model allows: place_work_frame slides the footprint
        # sideways to dodge decor but falls back to the region centre when every
        # offset is blocked, which on some layout/style pairs builds a front-row vessel
        # inside a toaster. Refuse such an instance instead of evaluating in it.
        mujoco.mj_forward(model._model, self.sim.data._data)
        clashes, _ = self._foreign_penetrations()
        if clashes:
            raise ValueError(f'{self.TASK_NAME}: RoboCasa decor blocks the task frame on layout '
                             f"{self.instance['layout_id']} style {self.instance['style_id']}: "
                             + '; '.join(f"{row['prop']} in {row['other']} by {row['penetration_m']} m"
                                         for row in clashes[:4]))
        blocked = self._pad_clearance_problems()
        if blocked:
            raise ValueError(f'{self.TASK_NAME}: RoboCasa decor stands over a pad on layout '
                             f"{self.instance['layout_id']} style {self.instance['style_id']}, so the "
                             'instance has no solved state: ' + '; '.join(blocked[:4]))

    def _pad_clearance_problems(self, max_penetration=.002):
        """Pads no vessel can stand on because a piece of RoboCasa decor is already there.

        The check above only sees the reset scene, where every vessel is in the front
        row, so a pad that lies *under* a counter accessory passes it and the defect
        surfaces only once the task is solved: on layout 1 / style 15 the mug is set
        15 mm inside ``paper_towel_main_group`` and the solver shoves it 52 mm off the
        pad while that penetration resolves, which no policy can undo. Stand each
        vessel on its own pad for one forward pass, keep the foreign contacts, and put
        the scene back exactly as it was.
        """
        data = self.sim.data
        saved = {name: np.array(data.get_joint_qpos(info['joint_name']), float).copy()
                 for name, info in self.geometry['objects'].items()}
        # The one cover the spec records over a pad is not decor: the solution lifts it off first
        # (see teleport_solution). Anything else standing over a pad still blocks the pad, so the
        # pass runs with the covers in place and only the recorded one is excused.
        covers = self._cover_geom_names()
        worst = {}
        try:
            for name, info in self.geometry['vessels'].items():
                self._place_vessel_on_pad(name, info)
                hits, _ = self._foreign_penetrations(max_penetration)
                pad = f"pad_{info['label']}"
                for row in hits:
                    if not (row['prop'] or '').startswith(f'{name}_'):
                        continue           # another vessel's own reset clash, reported above
                    if row['other'] in covers.get(pad, ()):
                        continue           # the recorded uncover entry
                    key = (pad, row['other'])
                    worst[key] = max(worst.get(key, 0.), float(row['penetration_m']))
                data.set_joint_qpos(info['joint_name'], saved[name])
        finally:
            for name, info in self.geometry['objects'].items():
                data.set_joint_qpos(info['joint_name'], saved[name])
                data.set_joint_qvel(info['joint_name'], np.zeros(6))
            self.sim.forward()
        return [f'{pad} in {other} by {round(depth, 5)} m'
                for (pad, other), depth in sorted(worst.items(), key=lambda item: -item[1])]

    def _foreign_penetrations(self, max_penetration=.002, ignore_robot=True):
        """Contacts between a prop and anything that is not a prop, a pad or the work top."""
        model, data = self.sim.model, self.sim.data
        prop_geoms = {gid for gids in self._object_geoms.values() for gid in gids}
        allowed = set(self._pad_geoms) | {model.geom_name2id(name)
                                          for name in self.task_spec['ball_contacts']['surface_geoms']}
        rows, worst = [], 0.
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if not ({a, b} & prop_geoms):
                continue
            other = b if a in prop_geoms else a
            if other in prop_geoms or other in allowed:
                continue
            other_name = model.geom_id2name(other) or ''
            if ignore_robot and other_name.startswith(('robot', 'gripper', 'mobilebase')):
                continue
            depth = -float(contact.dist)
            worst = max(worst, depth)
            if depth > max_penetration:
                rows.append(dict(prop=model.geom_id2name(a if a in prop_geoms else b),
                                 other=other_name, penetration_m=round(depth, 5)))
        return rows, worst

    def _object_state(self, name):
        model, data = self.sim.model, self.sim.data
        body = self.obj_body_id[name]
        gids = self._object_geoms[name]
        support, robot = set(), []
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in gids and b not in gids:
                continue
            other = b if a in gids else a
            other_name = model.geom_id2name(other) or ''
            if other_name.startswith(('robot', 'gripper', 'mobilebase')):
                robot.append(other_name)
            if other in self._pad_geoms:
                force = np.zeros(6)
                mujoco.mj_contactForce(model._model, data._data, index, force)
                normal = np.asarray(contact.frame).reshape(3, 3)[0] * (1 if b in gids else -1)
                if force[0] > 1e-5 and normal[2] > .5:
                    support.add(self._pad_geoms[other])
        linear = np.asarray(data.get_body_xvelp(name))
        angular = np.asarray(data.get_body_xvelr(name))
        return dict(position=np.asarray(data.body_xpos[body]).tolist(),
                    rotation=np.asarray(data.body_xmat[body]).reshape(3, 3).tolist(),
                    linear_velocity=linear.tolist(), angular_velocity=angular.tolist(),
                    linear_speed=float(np.linalg.norm(linear)), angular_speed=float(np.linalg.norm(angular)),
                    supported_pads=sorted(support), robot_contact=bool(robot), robot_contacts=robot,
                    grasped=bool(self._check_grasp(self.robots[0].gripper['right'],
                                                   [model.geom_id2name(g) for g in gids])))

    def object_states(self):
        return {name: self._object_state(name) for name in self.geometry['objects']}

    def _pose_states(self):
        """Positions and rotations only: no contact scan, for the per-tick collateral check."""
        data = self.sim.data
        return {name: dict(position=np.asarray(data.body_xpos[body]),
                           rotation=np.asarray(data.body_xmat[body]).reshape(3, 3))
                for name, body in self.obj_body_id.items()}

    def _current_score(self, states=None):
        return classify_placements(states or self.object_states(), self.geometry)

    def task_snapshot(self):
        return dict(objects=self.object_states())

    # ---- protocol hooks (contract C1/C3) ------------------------------------
    def task_progress(self):
        """Spec 1.4: (vessels upright on their own pad with their own two balls
        - vessels on a wrong pad) / vessel count, clipped to [0, 1]."""
        return rule_progress(self._current_score(), self.geometry['vessel_count'])

    def _ball_contacts(self, balls):
        """Geom ids each named ball touches, from a single pass over the contact list."""
        data = self.sim.data
        owner = {gid: name for name in balls for gid in self._object_geoms[name]}
        touching = {name: set() for name in balls}
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            for gid, other in ((a, b), (b, a)):
                name = owner.get(gid)
                if name is not None:
                    touching[name].add(other)
        return touching

    def ball_in_helper(self, position, touching):
        """The helper a ball at ``position`` is resting in, or None (the rule in ``stand.helpers``).

        In the helper's own frame: inside the inset footprint, below the rim, and touching one of its
        collision geoms. The inset is what tells a ball in the interior from a ball leaning against
        the outside wall.
        """
        data = self.sim.data
        for name, ref in getattr(self, 'helpers', {}).items():
            centre = np.asarray(data.body_xpos[ref['bid']], float)
            rotation = np.asarray(data.body_xmat[ref['bid']], float).reshape(3, 3)
            local = rotation.T @ (np.asarray(position, float) - centre)
            if abs(local[0]) > ref['inset'][0] or abs(local[1]) > ref['inset'][1]:
                continue
            if local[2] > ref['rim_local_z'] + .001:
                continue
            if touching & ref['gids']:
                return name
        return None

    def parked_balls(self, balls):
        """Of ``balls``, the ones that are not loose: ``{ball: helper}`` or ``{ball: 'gripper'}``.

        As the stand has it: a ball in a container or in the robot's hand is put
        somewhere, not dropped. Held means grasped, or touched by the robot while it is
        :data:`HELD_LIFT_M` clear of the work surface (a ball in mid-carry).
        """
        balls = [name for name in balls if name in self._object_geoms]
        if not balls:
            return {}
        model, data = self.sim.model, self.sim.data
        touching = self._ball_contacts(balls)
        top_z = float(self.work['top_z'])
        out = {}
        for name in balls:
            position = np.asarray(data.body_xpos[self.obj_body_id[name]], float)
            helper = self.ball_in_helper(position, touching[name])
            if helper is not None:
                out[name] = helper
                continue
            names = [model.geom_id2name(gid) or '' for gid in touching[name]]
            if not any(other.startswith(ROBOT_PREFIXES) for other in names):
                continue
            grasped = bool(self._check_grasp(self.robots[0].gripper['right'],
                                             [model.geom_id2name(gid) for gid in self._object_geoms[name]]))
            if grasped or position[2] - top_z > HELD_LIFT_M:
                out[name] = 'gripper'
        return out

    def collateral_checks(self):
        """Spec 1.5: balls outside the vessel they started in; the cost is how many.

        A ball parked in one of the helper containers, or held in the gripper, is not
        spilled. Taking the two balls out before lifting a vessel is the careful way to read a label -
        a shallow bowl tips them out the moment it turns - and the containers are where they go. A
        ball loose on the counter or on the floor still counts, and `bad` goes false again when it is
        picked up or put back, which the base logs as `collateral` and `repaired`.
        """
        spilled = spilled_balls(self._pose_states(), self.geometry)
        parked = self.parked_balls(spilled)
        loose = [ball for ball in spilled if ball not in parked]
        return {'balls_spilled': dict(bad=bool(loose), object=','.join(loose), cost=float(len(loose)))}

    def _place_vessel_on_pad(self, name, info):
        """Set one vessel upright on its own pad with its two balls inside (state edits only)."""
        data = self.sim.data
        pad = self.geometry['pads'][info['label']]
        yaw = float(self.work['yaw'])
        position = np.asarray(pad['center_world'], float) + np.array(
            [0., 0., -float(info['bottom_offset_z']) + MUG_REST_CLEARANCE])
        data.set_joint_qpos(info['joint_name'], np.r_[position, np.asarray(yaw_quat_wxyz(yaw))])
        data.set_joint_qvel(info['joint_name'], np.zeros(6))
        rotation = yaw_matrix(yaw)
        centre = np.asarray(info['interior_center'], float)
        floor_z = float(info['interior_floor_z']) + BALL_RADIUS + float(info['ball_drop'])
        for index, ball in enumerate(info['original_balls']):
            offset = rotation @ np.array([(2 * index - 1) * float(info['ball_spread']) + centre[0],
                                          centre[1], floor_z])
            data.set_joint_qpos(self.geometry['balls'][ball]['joint_name'],
                                np.r_[position + offset, np.array([1., 0., 0., 0.])])
            data.set_joint_qvel(self.geometry['balls'][ball]['joint_name'], np.zeros(6))
        self.sim.forward()
        return position

    def _cover_joint(self, name):
        obj = (getattr(self, 'objects', {}) or {}).get(name)
        joints = list(getattr(obj, 'joints', []) or []) if obj is not None else []
        return joints[0] if joints else f'{name}_joint'

    def _uncover_steps(self):
        """C3: take every cover off before the vessels move.

        A flat cover is pushed along the direction its build-time push gate chose, by the length that
        clears the pad it hides; an object cover is set down on a free spot of the same surface. Both
        stay on the work surface and clear of every pad, vessel and of Submit, and neither is moved by
        the robot: these are state edits, like the rest of the certificate.
        """
        steps = []
        record = self.task_spec.get('observability') or {}
        covers = [cover for cover in record.get('covers', []) if cover.get('built')]
        if not covers:
            return steps
        rng = self._observability_rng()
        shapes = self._prop_shapes()
        bounds = observe.work_surface_rect(self)
        # the task props, the occluders the wiring stood (not task props) and the counter decor: a cover
        # teleported into any of these is resolved by the physics with whatever impulse the overlap takes
        keepouts = list(shapes.values()) + observe.decor_shapes_on_work(self)
        for occluder in record.get('occluders', []):
            keepouts.append(S.Box(tuple(occluder['frame_xy']), (occluder['size'][0] / 2, occluder['size'][1] / 2),
                                  float(occluder['frame_yaw'])))
        for cover in covers:
            name = cover['name']
            try:
                joint = self._cover_joint(name)
                qpos = np.asarray(self.sim.data.get_joint_qpos(joint), float).copy()
                chosen = (cover.get('reach') or {}).get('chosen')
                if cover.get('place') and chosen:
                    target = (np.asarray(cover['place_xy'], float)
                              + np.asarray(chosen['direction_frame'], float) * float(chosen['length_m']))
                else:
                    radius = float(cover.get('radius') or max(cover.get('half') or (.10, .10)))
                    spot = S.scatter(rng, 1, bounds, radius, .01, keepouts=keepouts, keepout_gap=.02,
                                     max_tries=300, attempts_per_prop=300)[0]
                    target = np.asarray(spot[:2], float)
                world = observe.frame_to_world_xy(self, target)
                qpos[:2] = world
                self.sim.data.set_joint_qpos(joint, qpos)
                self.sim.data.set_joint_qvel(joint, np.zeros(6))
                self.sim.forward()
                keepouts.append(S.Disc((float(target[0]), float(target[1])),
                                       float(cover.get('radius') or max(cover.get('half') or (.10, .10)))))
                steps.append(dict(step=f'uncover_{name}', ok=True,
                                  detail=dict(kind=cover['kind'], hides=cover.get('place') or cover.get('object'),
                                              moved_to_frame=[round(float(v), 4) for v in target])))
            except Exception as error:                       # never raises (C3)
                steps.append(dict(step=f'uncover_{name}', ok=False, detail=f'{type(error).__name__}: {error}'))
        return steps

    def teleport_solution(self, settle_ticks=TELEPORT_SETTLE_TICKS):
        """C3: bring the scene to a solved state by state edits alone, then settle.

        Every cover comes off first (nothing is hidden at the ``visible`` level, where there is none),
        then every vessel stands upright on the pad of its own label colour with its own two balls
        inside, wherever the observability wiring left it. The robot is never moved and Submit is never
        pressed.
        """
        steps = self._uncover_steps()
        for name, info in self.geometry['vessels'].items():
            try:
                position = self._place_vessel_on_pad(name, info)
                steps.append(dict(step=f'place {name} on pad_{info["label"]}', ok=True,
                                  detail=dict(position=[round(float(v), 4) for v in position])))
            except Exception as error:                       # never raises (C3)
                steps.append(dict(step=f'place {name} on pad_{info["label"]}', ok=False,
                                  detail=f'{type(error).__name__}: {error}'))
        try:
            action = np.zeros(self.action_dim)
            for _ in range(int(settle_ticks)):
                self.step(action)
            score = self._current_score()
            steps.append(dict(step='settle', ok=bool(score['success']),
                              detail=dict(ticks=int(settle_ticks), success=bool(score['success']),
                                          progress=rule_progress(score, self.geometry['vessel_count']),
                                          failing=[n for n, row in score['vessels'].items()
                                                   if not row['success']])))
        except Exception as error:
            steps.append(dict(step='settle', ok=False, detail=f'{type(error).__name__}: {error}'))
        return steps

    # ---- gates --------------------------------------------------------------
    def build_gate(self, max_penetration=.002):
        """G1: no prop may be built inside a fixture, an appliance or another prop."""
        foreign, worst = self._foreign_penetrations(max_penetration, ignore_robot=False)
        return dict(passed=not foreign, foreign_contacts=foreign,
                    worst_foreign_penetration_m=round(worst, 5),
                    decor_on_surface=list(self.work['decor_on_surface']),
                    decor_near_props=list(self.task_spec['decor_near_props']),
                    decor_blocked=[row[1] for row in self.work['decor_blocked']],
                    front_edge_clearance_m=self.front_edge_clearance(),
                    score_zero_at_reset=not self.evaluate_success()['success'])

    def front_edge_clearance(self):
        """Spec 3: the nearest vessel surface to the work fixture's front edge, in metres.

        The fixture's front face is its local -y side (RoboCasa's convention), so the
        clearance is measured along the frame's +y axis from that face.
        """
        fixture = self.work_fixture
        front_local = -float(fixture.size[1]) / 2
        into = yaw_matrix(float(fixture.rot))[:, 1]           # fixture-local +y in world
        origin = np.asarray(fixture.pos, float)
        radius = vessel_radius(self.spec)
        # A vessel the observability wiring put on another counter is not measured against this one.
        annexed = {entry['object'] for entry in (self.task_spec.get('observability') or {}).get('annex', [])}
        best = None
        for name in self.geometry['vessels']:
            if name in annexed:
                continue
            position = np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]], float)
            along = float(np.dot(position - origin, into)) - front_local - radius
            best = along if best is None else min(best, along)
        return round(float(best if best is not None else 0.), 4)

    def settle_check(self, ticks=SETTLE_TICKS):
        """G2: zero-action ticks must leave every ball inside its own vessel and nothing adrift."""
        start = {name: np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]]).copy()
                 for name in self.geometry['objects']}
        action = np.zeros(self.action_dim)
        for _ in range(int(ticks)):
            self.step(action)
        states = self.object_states()
        score = classify_placements(states, self.geometry)
        moved = {name: float(np.linalg.norm(np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]]) - start[name]))
                 for name in start}
        contained = {name: sorted(row['contained_balls']) == sorted(self.geometry['vessels'][name]['original_balls'])
                     for name, row in score['vessels'].items()}
        return dict(ticks=int(ticks), all_balls_contained=all(contained.values()), contained=contained,
                    max_displacement_m=round(max(moved.values()), 5), displacement_m=moved,
                    upright={name: row['upright'] for name, row in score['vessels'].items()},
                    vessel=self.geometry['vessel'], success=bool(score['success']))

    def _camera_pixel(self, camera, world_xyz, size):
        from robosuite.utils import camera_utils
        transform = camera_utils.get_camera_transform_matrix(self.sim, camera, size, size)
        homogeneous = transform @ np.r_[np.asarray(world_xyz, float), 1.]
        if homogeneous[2] <= 0:
            return None
        pixel = homogeneous[:2] / homogeneous[2]
        return [float(pixel[0]), float(pixel[1])]

    def hidden_state_check(self, cameras=CAMERAS, size=512):
        """G3: the label decals must not move a pixel further than the renderer's own noise.

        MuJoCo's offscreen renderer is not bit-exact in a RoboCasa kitchen: repeating
        the very same render changes 100-200 pixels of a large flat wall by one least
        significant bit on some layouts. So the check renders labels-on, labels-off and
        labels-on again, and demands that hiding the labels change no channel by more
        than that measured repeat-render floor. A visible 40 mm coloured disc is worth
        tens of levels, so this stays fail-closed.

        The visibility part is a frustum test on the pad and button centres, not an
        occlusion test: it says the props are inside a fixed camera's image, not that
        nothing stands in front of them.
        """
        def snap():
            return {camera: self.sim.render(camera_name=camera, width=size, height=size, depth=False).copy()
                    for camera in cameras}

        model = self.sim.model
        snap()  # warm the context so the first comparison is not a cold render
        before = snap()
        saved = {gid: float(model.geom_rgba[gid][3]) for gid in self._label_geoms.values()}
        for gid in saved:
            model.geom_rgba[gid][3] = 0.
        try:
            after = snap()
        finally:
            for gid, alpha in saved.items():
                model.geom_rgba[gid][3] = alpha
        repeat = snap()

        def compare(a, b):
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            return int(np.count_nonzero(np.any(delta > 0, axis=-1))), int(delta.max())

        toggle = {camera: compare(before[camera], after[camera]) for camera in cameras}
        noise = {camera: compare(before[camera], repeat[camera]) for camera in cameras}
        invisible = {camera: bool(toggle[camera][1] <= max(noise[camera][1], 1)) for camera in cameras}
        targets = {f'pad_{colour}': pad['center_world'] for colour, pad in self.geometry['pads'].items()}
        targets['submit'] = list(self.task_spec['submit_button']['position'])
        in_view = {}
        for camera in cameras:
            pixels = {name: self._camera_pixel(camera, xyz, size) for name, xyz in targets.items()}
            in_view[camera] = {name: bool(p is not None and 0 <= p[0] < size and 0 <= p[1] < size)
                               for name, p in pixels.items()}
        fixed_camera = [camera for camera in cameras if all(in_view[camera].values())]
        report = dict(labels_invisible=all(invisible.values()), invisible=invisible,
                      differing_pixels={c: toggle[c][0] for c in cameras},
                      max_channel_delta={c: toggle[c][1] for c in cameras},
                      repeat_render_pixels={c: noise[c][0] for c in cameras},
                      repeat_render_max_delta={c: noise[c][1] for c in cameras},
                      bit_exact={c: bool(toggle[c][0] == 0) for c in cameras},
                      in_view=in_view, cameras_showing_all_pads_and_submit=fixed_camera,
                      passed=bool(all(invisible.values()) and fixed_camera),
                      note=f'Label alpha toggled to 0, the three policy cameras re-rendered at {size} px and '
                           'compared against a repeat render of the unchanged scene.')
        observability = self._observability_check(size, cameras=cameras)
        if observability is not None:
            report['observability'] = observability
            report['passed'] = bool(report['passed'] and observability['passed'])
        return report

    def _observability_check(self, image_size=512, cameras=CAMERAS):
        """G3 for the observability axis (spec 2.2): what the spec hides is in no policy camera, what it
        does not is in one, and every ``look`` vessel stands where a reachable viewpoint sees it. None at
        the ``visible`` level, where there is nothing to check.

        A hidden vessel carries its two balls, so they are checked with it; a covered pad is checked by
        its own geoms, because the colour under the plate is the hidden state there."""
        record = self.task_spec.get('observability') or {}
        entries = record.get('realised') or []
        if not entries:
            return None
        vessels = self.geometry['vessels']
        hidden = [name for name in record.get('hidden_bodies', []) if name in vessels]
        balls = [ball for name in hidden for ball in vessels[name]['original_balls']]
        hidden_geoms = []
        for cover in record.get('covers', []):
            place = cover.get('place') or ''
            if cover.get('built') and place.startswith('pad_'):
                pad = self.geometry['pads'][place[len('pad_'):]]
                hidden_geoms += [pad['floor_geom'], pad['floor_geom'] + '_visual']
        visible = [name for name in vessels if name not in hidden]
        check = observe.hidden_check(self, hidden + balls, visible=visible, image_size=image_size,
                                     cameras=cameras, hidden_geoms=hidden_geoms)
        floor = observe.floor_map(self)
        overrides = self._vessel_objects()
        look = {entry['object']: observe.look_reachable(self, entry['object'], floor=floor, at_build=False,
                                                        **overrides.get(entry['object'], {}))
                for entry in entries if entry['mode'] in hiding.LOOK_MODES}
        reach = {cover['name']: bool((cover.get('reach') or {}).get('passed'))
                 for cover in record.get('covers', []) if cover.get('built')}
        passed = bool(check['passed'] and all(row['passed'] for row in look.values()) and all(reach.values()))
        return dict(passed=passed, level=record.get('level'), hidden=hidden, hidden_balls=balls,
                    hidden_geoms=hidden_geoms, hidden_ok=check['hidden_ok'], visible_ok=check['visible_ok'],
                    deterministic=check['deterministic'], pixels=check['pixels'],
                    look_reachable={name: dict(passed=row['passed'], stance=row['stance'],
                                               reasons=row['reasons'][:2]) for name, row in look.items()},
                    cover_reach=reach, problems=list(record.get('problems') or []),
                    fallbacks=list(record.get('fallbacks') or []))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['roboquest']['split_key'] = self.split_key(self.spec)
        meta['lang'] = self.PUBLIC_GOAL
        return meta

    def frame_pose(self, name):
        """Task-frame pose of a prop, for oracles and diagnostics."""
        state = self._object_state(name)
        local = yaw_matrix(-self.work['yaw']) @ (np.asarray(state['position'])
                                                 - np.asarray(self.work['center_world']))
        rotation = np.asarray(state['rotation'])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0]) - self.work['yaw']
        return dict(frame_xyz=local.tolist(), frame_yaw_rad=float(yaw), state=deepcopy(state))
