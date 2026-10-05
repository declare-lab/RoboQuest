"""Painted cubes (tasks-v0 task 5) as a RoboQuest kitchen task.

White 52 mm cubes stand **scattered** over the cube zone of the task frame, at
uniformly random positions and yaws and never in rows, together with **two**
irreversible capture bins at the frame's front corners and the Submit button;
the instance's rule decides which bin each cube belongs in. A rule names
exactly the attributes that separate its targets from the other cubes in that
scene: "every cube that has a <colour> <shape>" (``has_mark``) when both
attributes are needed, "every cube that has a <shape>" (``has_shape``) when
every colour of that shape counts, "every cube that has a <colour> mark"
(``has_colour``) when every shape in that colour counts, or "every cube with
exactly one marked face" (``exactly_one``). The **blue** lid takes the cubes that satisfy the rule,
the **yellow** lid the cubes that do not (spec 4.1); which side carries which
lid and the bin bodies' colours are nuisance. Marks are face decals from a
shape x colour vocabulary.

**Placement.** Poses come from :mod:`roboquest.scatter`, which
re-draws until nothing touches: each cube is sampled with the bounding disc of
its own *cube plus its occluder* (a flat plate or a bridging plate reaches
further than the cube; a level cover reserves its disc the same way), so the
occluder can be hung on the settled cube and is guaranteed clear of every other
prop, of the bin and of Submit, and inside the frame. The draw raises
``ValueError`` when it cannot be satisfied, which rejects the seed.

**Symmetric hiding.** At reset the faces of every cube that are proved hidden
from the three settled policy cameras (analytic self-occlusion proof from
cube inspection v4, with a 3 cm camera-jitter margin) receive the deciding
marks, and the visible faces of every cube are consistent with both classes
under the rule, so no cube can be classified from the initial views. The
realised assignment is a deterministic function of the frozen instance and
the settled geometry, recorded in ``ep_meta``. A fail-closed render check
(``hidden_state_check``) re-renders the policy cameras with every hidden-face
mark erased and raises on any differing pixel; the CPU build path skips it.

Optional occluders - nuisance clutter, **plates only** from generator v6 - cover
cubes chosen independently of their class; how many is nuisance. A plate lies
flat on the cube's top face or bridges from it to the counter, and the cube
stays visible either way. An inverted bowl centred on a cube is a full cover,
so it is not clutter: the generator keeps
it only as the ``uncover`` level's cover, recorded in ``spec['hidden']``; the
"edge of the bowl on the cube" variant was declined, and a bowl in
``spec['occluders']`` is refused by ``validate_spec``, ``cpu_gate`` and the
build (v4 and v5 drew ``bowl_a``/``bowl_b`` as clutter). Scoring is the v1
predicate: every cube captured (welded) by the bin its class belongs to. The
v0 surface-support and settled predicates for kept cubes are gone, because no
cube is kept on the counter any more. The score is frozen at the first
physical Submit.
"""
from copy import deepcopy
import hashlib
import inspect
from itertools import combinations
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robocasa.models.fixtures.fixture_utils import fixture_is_type
import robocasa.models.fixtures.fixture_utils as FixtureUtils
from robocasa.models.fixtures.blender import Blender
from robocasa.models.fixtures.coffee_machine import CoffeeMachine
from robocasa.models.fixtures.dish_rack import DishRack
from robocasa.models.fixtures.electric_kettle import ElectricKettle
from robocasa.models.fixtures.microwave import Microwave
from robocasa.models.fixtures.sink import Sink
from robocasa.models.fixtures.stand_mixer import StandMixer
from robocasa.models.fixtures.stove import Stove
from robocasa.models.fixtures.toaster import Toaster
from robocasa.models.fixtures.toaster_oven import ToasterOven
from robosuite.models.objects import BoxObject
from scipy.spatial.transform import Rotation

from roboquest import headroom as HR
from roboquest import observe as OB
from roboquest import registry as REG
from roboquest import scatter as S
from roboquest.assets import materials
from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz
from roboquest.cubes import hiding as H
from roboquest.cubes import marks as M
from roboquest.cubes import reach as RCH
from roboquest.cubes.bin import BinCapture, append_bin, append_capture_constraints
from roboquest.harness.contract import CAMERAS
from roboquest.base.scene import VisionKitchenScene
from roboquest.base.scene_contracts import support_contact

CONTRACT_VERSION = 'painted-cubes-kitchen-v3'   # v3: the coarse rule kinds join the goal text family and the observability axis (v2: two bins, sorting predicate)
GOAL_NOUN = 'cubes'
GOAL_MIDDLE = ('Some of them satisfy this rule: {rule}. Put every cube that satisfies it in the blue bin '
               'and every other cube in the yellow bin, then press Submit.')
BUDGET_TICKS = 12000           # provisional (C1): no oracle certificate for this task yet
PHYSICS_TIMESTEP = .001            # v4 tuning for 52 mm cubes and the bin flap
SETTLE_TICKS = 10                  # extra zero-action control ticks before marks are assigned
# Task frame (metres): x along the front of the surface, y away from the robot. The footprint fits RoboCasa's
# 0.65 m deep wall counters (layouts 1, 2, 7, 9 offer at most 1.18 x 0.65 m free; islands there are split).
FOOTPRINT = (1.16, .64)            # unchanged by v1's two bins
FRAME_HALF = (FOOTPRINT[0] / 2, FOOTPRINT[1] / 2)
CUBE_YAW_RANGE = (-math.pi, math.pi)   # the cubes stand at any yaw, so a mark may face anywhere
MIN_GAP_M = .04                    # clear space between two cube (or cube + occluder) footprints
KEEPOUT_GAP_M = .02                # clear space from the bin or the Submit button
OCCLUDER_MIN_CLEARANCE_M = .01     # a placed occluder must not touch another prop
SCATTER_MAX_TRIES = 60000          # sized for v5's tight cell (two bowls and two bridging plates among six cubes); the v6 draw is looser
CUBE_HALF_DIAGONAL = M.HALF_SIZE_M * math.sqrt(2)
CUBE_HALF_XY = (M.HALF_SIZE_M, M.HALF_SIZE_M)
PLATE_ASSET, PLATE_SCALE, PLATE_RADIUS = 'objaverse/plate/plate_1/model.xml', 1.0, .091
BOWL_ASSET, BOWL_SCALE, BOWL_RADIUS = 'objaverse/bowl/bowl_2/model.xml', 1.8, .104
FLAT_OFFSET = .012                 # plate resting on the cube top, slightly off centre
BRIDGE_OFFSET, BRIDGE_TILT_DEG, BRIDGE_Z = .06, 25., .048   # plate leaning from the cube top to the counter
# The clutter pool: plates only (v6). v4/v5 also drew ('bowl', 'bowl_a') and ('bowl', 'bowl_b'),
# an inverted bowl centred on the cube, which the user ruled a full cover, not clutter; a bowl over a cube now
# exists only as the uncover level's cover in spec['hidden']. BOWL_* stay for reading v5 records (occluder_reach).
OCCLUDER_SET = (('plate', 'plate_a'), ('plate', 'plate_b'))
CLUTTER_KINDS = ('plate',)         # the only occluder kind validate_spec, cpu_gate and the build accept
OCCLUDER_SPARE_CUBES = 2           # however many occluders are drawn, this many cubes always stay bare
BIN_HALF_XY = (.135, .115)         # exterior half footprint of one bin (0.264 x 0.224 m) plus 3 mm
SUBMIT_HALF_XY = (.05, .045)
# Two bins at the frame's front corners, mirrored (spec 4.1): the margins below reproduce the v0 bin at
# (0.43, 0.15) and Submit at (0.43, -0.20) on the unchanged 1.16 x 0.64 m frame and re-derive both from the
# footprint, so the layout logic - not a constant - places them. Submit stays in front of the right-hand bin,
# which leaves the middle of the frame and the strips beside and behind the bins to the cubes.
BIN_SIDE_MARGIN_M = .015           # bin to the frame's side edge
BIN_BACK_MARGIN_M = .055           # bin to the frame's back edge
SUBMIT_FRONT_MARGIN_M = .075       # Submit to the frame's front edge
BIN_INTERIOR_HEIGHT_M = .30        # spec 4.1: was 0.50; the mouth ends up about 1.23 m above the floor
BIN_CLASSES = ('blue', 'yellow')   # blue = satisfies the rule, yellow = does not (functional colours, fixed)
BIN_NAMES = {'blue': 'cube_bin_blue', 'yellow': 'cube_bin_yellow'}
LID_RGBA = {'blue': (.09, .32, .78, 1.), 'yellow': (.95, .80, .10, 1.)}
BIN_SIDES = ('bin_left', 'bin_right')
BIN_BODY_MATERIAL = 'plastic'      # the nuisance body colour (spec 4.9); the lids stay functional
HIDING_ROLES = ('occluder', 'cover')   # props observability adds: never scored, never counted as cubes
TELEPORT_SALT = 0x0C3E             # second word of the teleport certificate's RNG seed
COVER_ASIDE_GAP_M = .015           # clear space around a cover the certificate sets down beside its cube
TELEPORT_DROP_M = .055             # how far above the bin mouth a teleported cube starts its drop
TELEPORT_DROP_TICKS = 30           # zero-action ticks allowed per cube for the fall, settle and weld
TELEPORT_FINAL_TICKS = 20
# Counter-top fixtures that are not RoboCasa accessories but occupy the work surface (stovetops, sinks, small
# appliances); the base class only avoids accessories, which spawned props inside a toaster oven (layout 7)
# and on a cooktop (layout 2).
COUNTER_TOP_FIXTURES = (Stove, Sink, ToasterOven, Toaster, CoffeeMachine, Microwave, StandMixer, Blender,
                        ElectricKettle, DishRack)
# Hidden-face sets the CPU gate realises marks under. The counter always hides the bottom face; with the
# free yaw of v1 a cube can additionally turn one or two *adjacent* side faces away from all three policy
# cameras, so the reachable sets are 'nz' plus zero, one or two adjacent sides. 'pz' is never hidden (the
# cameras look down) and two opposite sides are never hidden together, so this enumeration is complete.
HYPOTHETICAL_HIDDEN = (('nz',), ('nz', 'px'), ('nz', 'nx'), ('nz', 'py'), ('nz', 'ny'),
                       ('nz', 'px', 'py'), ('nz', 'px', 'ny'), ('nz', 'nx', 'py'), ('nz', 'nx', 'ny'))
MIXED_HIDDEN_SHIFTS = 3            # extra gate configurations giving neighbouring cubes different hidden sets


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def hidden_configurations(names):
    """The hidden-face configurations the CPU checks work over: one set on every cube, plus the mixtures
    free yaws produce between neighbouring cubes."""
    configurations = [{name: faces for name in names} for faces in HYPOTHETICAL_HIDDEN]
    for shift in range(1, MIXED_HIDDEN_SHIFTS + 1):
        configurations.append({name: HYPOTHETICAL_HIDDEN[(index + shift) % len(HYPOTHETICAL_HIDDEN)]
                               for index, name in enumerate(names)})
    return configurations


def frame_layout(footprint=FOOTPRINT):
    """Where the fixed props stand in the task frame: the two bins and Submit, derived from the footprint.

    The bins sit in the frame's two front corners, mirrored about x = 0 (spec 4.1), and Submit in front of
    the right-hand one. On the 1.16 x 0.64 m frame this is the v0 bin at (0.43, 0.15), its mirror image at
    (-0.43, 0.15) and Submit at (0.43, -0.20).
    """
    half_x, half_y = footprint[0] / 2, footprint[1] / 2
    bin_x = round(half_x - BIN_HALF_XY[0] - BIN_SIDE_MARGIN_M, 6)
    bin_y = round(half_y - BIN_HALF_XY[1] - BIN_BACK_MARGIN_M, 6)
    submit_y = round(-(half_y - SUBMIT_HALF_XY[1] - SUBMIT_FRONT_MARGIN_M), 6)
    return dict(bin_left=[-bin_x, bin_y], bin_right=[bin_x, bin_y], submit=[bin_x, submit_y])


def cube_zone(footprint=FOOTPRINT):
    """Where the cubes are scattered: the whole frame, the two bins and Submit being keep-outs.

    The cubes may use the whole remaining frame, so the middle, the front band and the
    strips beside and behind the bins are all fair game.
    """
    return S.rect((0., 0.), (footprint[0] / 2, footprint[1] / 2))


def cube_keepouts(spec=None, footprint=FOOTPRINT):
    """What a cube (with its occluder) must stay ``KEEPOUT_GAP_M`` clear of, with their names."""
    placement = (spec or {}).get('placement') or frame_layout(footprint)
    return dict(bin_left=S.rect(placement['bin_left'], BIN_HALF_XY),
                bin_right=S.rect(placement['bin_right'], BIN_HALF_XY),
                submit=S.rect(placement['submit'], SUBMIT_HALF_XY))


def occluder_reach(entry):
    """How far an occluder reaches from the centre of the cube it covers."""
    if entry['kind'] == 'bowl':
        return BOWL_RADIUS
    return (BRIDGE_OFFSET if entry['style'] == 'bridge' else FLAT_OFFSET) + PLATE_RADIUS


def cube_footprints(spec):
    """One scatter footprint per cube: its bounding disc, grown to hold the occluder or cover it carries.

    Placing the pair as one disc is what lets the occluder be hung on the settled
    cube afterwards and still be provably clear of everything else. An ``uncover``
    cube reserves its cover's disc the same way (cloche 0.080 m, wide cup 0.087 m,
    the widest fitted bowl 0.111 m), so the cover is feasible where the cube stands
    and the level never rejects a seed at build time for a cover that does not fit.
    """
    reach = {o['cube']: occluder_reach(o) for o in spec.get('occluders', ())}
    for name, kind in H.cover_entries(spec).items():
        reach[name] = max(reach.get(name, 0.), H.cover_footprint_radius(kind))
    return [max(CUBE_HALF_DIAGONAL, reach.get(cube['name'], 0.)) for cube in spec['cubes']]


def cube_poses(spec):
    """``(x, y, yaw)`` per cube, in the spec's order."""
    return [(float(c['xy'][0]), float(c['xy'][1]), float(c['yaw'])) for c in spec['cubes']]


def hiding_shapes(spec, footprint=FOOTPRINT):
    """Task-frame keep-out of every prop, by name: the cubes that stay where they were scattered, both
    bins, Submit and the face-occluding plates (v4/v5 records also carry bowls, read the same way).

    Mint and build share this, so the candidate walk (:func:`hiding.walk_covers`) gates against exactly
    the keep-outs :meth:`PaintedCubes.hiding_keepouts` hands to the build. A hidden cube is left out: a
    cover stands on the disc the scatter already reserved for it (:func:`cube_footprints`), and an annexed
    or relocated cube leaves that disc empty.
    """
    hidden = set(H.hidden_objects(spec))
    shapes = {str(cube['name']): S.shape_at(CUBE_HALF_XY, cube['xy'], cube['yaw'])
              for cube in spec['cubes'] if cube['name'] not in hidden}
    shapes.update(cube_keepouts(spec, footprint))
    for occluder in spec.get('occluders') or ():
        centre, radius = occluder_placement(spec, occluder)
        shapes[str(occluder['name'])] = S.Disc((float(centre[0]), float(centre[1])), float(radius))
    return shapes


def occluder_placement(spec, entry):
    """Frame-local centre and bounding radius of one occluder over its scattered cube."""
    cube = next(c for c in spec['cubes'] if c['name'] == entry['cube'])
    centre = np.asarray(cube['xy'], float)
    if entry['kind'] == 'bowl':
        return centre, BOWL_RADIUS
    offset = BRIDGE_OFFSET if entry['style'] == 'bridge' else FLAT_OFFSET
    return centre + offset * np.asarray(entry['direction'], float), PLATE_RADIUS


class PaintedCubes(RoboQuestKitchen):
    TASK_NAME = 'painted_cubes'
    CONTRACT_VERSION = CONTRACT_VERSION
    # v5: the reach rules (``cubes.reach``, job CUBES-MINT-REACH): the cubes are scattered inside the grasp
    # envelope ahead of the base the layout gives, a kitchen is refused when its floor lets the base slide
    # too little for both bins or no place on it carries a cube to a bin mouth, and the work frame is put
    # where the bin corridors have clear air, so no instance is minted that the robot could never finish.
    # v4: the cover a cube carries is walked at mint time against the gates the build applies
    # (hiding.walk_covers), so a drawn kind the robot could not lift is replaced before the build.
    # v6: clutter occluders are plates only: the inverted
    # bowl leaves the clutter pool, the per-instance cap follows (0 to min(2, n - 2)), and a bowl in
    # spec['occluders'] is refused. The v5 rows that carried one were derived into v6 twins with the bowls
    # removed and re-gated; every other draw, stream and rule is v5's.
    GENERATOR_VERSION = 'v6'       # v3: four rule kinds with minimal wording, and observability (v2: 4/6/8 cubes, two bins, nuisance targets/rule/occluders)
    BUDGET_TICKS = BUDGET_TICKS
    # Spec 2.1: the reported axes are the size and the observability level. ``target_count`` and ``rule``
    # stay in the spec but are drawn balanced from the ``structure`` stream (C2), and so are the occluders
    # and the hidden entries, which spec 2.2 calls random variation at every level.
    FACTORS = {'cube_count': (4, 6, 8), 'observability': ('visible', 'look', 'uncover')}
    DEV_ONLY_LEVELS = {'observability': ('look+uncover',)}
    TARGET_COUNTS = (1, 2, 3)
    FOOTPRINT = FOOTPRINT
    FOOTPRINT_FRONT_MARGIN = .005  # the frame must stay inside 0.65 m deep counters; props keep 2 cm to each edge
    # Hold-out: the diagonal (8, 3), (6, 2) and (4, 1) across all four rule kinds, so
    # every size level keeps a held-out class and every level of every factor still appears in dev. Twelve
    # of the thirty-six structure classes are held out, the same third as with two kinds.
    HOLDOUT_KEYS = frozenset(f'{rule}-c{count}-t{targets}'
                             for rule in M.RULE_KINDS for count, targets in ((8, 3), (6, 2), (4, 1)))

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        """Spec 1.3 / contract C1: the scope sentence (with the count only when ``show_count`` was drawn),
        the v1 middle sentence naming the two lid colours, then the clutter sentence."""
        middle = GOAL_MIDDLE.format(rule=M.rule_text(spec['rule']))
        return cls.frame_goal(spec, GOAL_NOUN, int(spec['cube_count']), middle)

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, material = streams['structure'], streams['poses'], streams['materials']
        cube_count = int(factors['cube_count'])
        # CUBES-MINT-REACH rules (b) and (c): both bins stand where the robot can drop a cube in, from a
        # stance the kitchen's floor allows, and the hand can carry the cube over the counter to each mouth
        # without meeting a wall cabinet (``cubes.reach``, measured). A kitchen that cannot is refused before
        # the draw is made, as a candidate and not as an invalid spec, so ``mint_grid`` walks to the next seed.
        layout_id = int(factors.get('kitchen_layout') or 0)
        style_id = int(factors.get('kitchen_style') or 0)
        bin_spots = frame_layout(cls.FOOTPRINT)
        refused = RCH.reach_problems([(side, bin_spots[side]) for side in BIN_SIDES], layout_id, style_id)
        if refused:
            raise ValueError('the kitchen cannot reach both bins: ' + '; '.join(refused))
        refused = RCH.corridor_problems(layout_id, style_id)
        if refused:
            raise ValueError('the kitchen cannot carry a cube to the bins: ' + '; '.join(refused))
        names = list(M.CUBE_NAMES[:cube_count])
        # Nuisance structure, balanced and recorded in the spec (C2): how many cubes satisfy the rule, which
        # rule it is, which of them, and the frozen plan for the visible faces.
        target_count = int(cls.TARGET_COUNTS[int(structure.integers(len(cls.TARGET_COUNTS)))])
        # The kind is drawn uniformly over the four; the draw sits where the two-kind one did and
        # RULE_KINDS is ordered so that a seed still landing on has_mark or exactly_one keeps its v2 spec.
        rule = dict(kind=str(M.RULE_KINDS[int(structure.integers(len(M.RULE_KINDS)))]))
        if rule['kind'] == 'has_mark':
            rule.update(colour=str(structure.choice(M.COLOURS)), shape=str(structure.choice(M.SHAPES)))
        elif rule['kind'] == 'has_colour':
            rule.update(colour=str(structure.choice(M.COLOURS)))
        elif rule['kind'] == 'has_shape':
            rule.update(shape=str(structure.choice(M.SHAPES)))
        targets = sorted(int(i) for i in structure.choice(cube_count, size=target_count, replace=False))
        mark_seed = int(structure.integers(0, 2**31 - 1))
        visible_plan = M.sample_visible_plan(structure, rule, names)
        # Observability (spec 2.2). The level is a factor; which cubes it hides and how is nuisance, drawn
        # from the poses stream *after* every structure draw above, so the rule, the targets, the plan, the
        # split key and the goal text are byte-identical at every level of the same structure seed.
        level = str(factors.get('observability', 'visible'))
        hidden, hidden_fallbacks = H.draw_hidden(poses, level, names, [names[i] for i in targets])
        hidden_names = {entry['object'] for entry in hidden}
        # Which lid stands on which side is nuisance too, so "the blue bin" is never "the right-hand bin".
        layout = frame_layout(cls.FOOTPRINT)
        blue_side = BIN_SIDES[int(structure.integers(len(BIN_SIDES)))]
        bins = [dict(lid=lid, name=BIN_NAMES[lid],
                     side=blue_side if lid == 'blue' else next(s for s in BIN_SIDES if s != blue_side))
                for lid in BIN_CLASSES]
        for entry in bins:
            entry['xy'] = list(layout[entry['side']])
            entry['body_rgba'] = [round(float(v), 4) for v in materials.rgba(material, BIN_BODY_MATERIAL)]
        # Occluders are decided first (materials stream) because which occluder a cube carries sets the
        # footprint it is scattered with. Which cubes are covered never depends on their class: a uniform
        # draw over all cubes, and the two plate styles and their lean directions are drawn the same way.
        # How many there are is nuisance as well (spec 2.2: random variation at every level), capped so that
        # OCCLUDER_SPARE_CUBES cubes always stay bare - at four cubes v0 covered every one of them. The pool
        # is plates only (v6): 0 to min(2, n - 2) of them, the same draw order as v5 over the shorter set.
        # A cube that observability hides is never given a face occluder as well: a plate leaning on a cube
        # that stands under a cloche or in the annex would be a second, unrecorded hiding mechanism.
        bare = [index for index, name in enumerate(names) if name not in hidden_names]
        occluder_count = int(material.integers(min(len(OCCLUDER_SET), cube_count - OCCLUDER_SPARE_CUBES,
                                                   len(bare)) + 1))
        occluders = []
        if occluder_count:
            covered = [int(i) for i in material.choice(bare, size=occluder_count, replace=False)]
            order = [OCCLUDER_SET[int(i)] for i in material.permutation(len(OCCLUDER_SET))][:occluder_count]
            for (kind, occluder_name), cube_index in zip(order, covered):
                entry = dict(name=occluder_name, kind=kind, cube=names[cube_index])
                if kind == 'plate':
                    entry['style'] = 'bridge' if material.random() < .5 else 'flat'
                    angle = float(material.uniform(0., 2 * math.pi))      # no rows left to lean away from
                    entry['direction'] = [round(math.cos(angle), 4), round(math.sin(angle), 4)]
                occluders.append(entry)
        occluders.sort(key=lambda entry: entry['name'])
        # Nuisance: the placement. Cubes are scattered over the whole cube zone at any yaw, each with the
        # bounding disc of its cube-and-occluder pair, clear of one another, of both bins and of Submit.
        # ValueError (an impossible draw) propagates out of mint and rejects the seed.
        skeleton = dict(cubes=[dict(name=name) for name in names], occluders=occluders, hidden=hidden)
        # A covered cube is scattered in the reachable band (``hiding.cover_zone``) so that its cover can
        # be lifted off from the robot's reset stance; every other cube uses the whole zone.
        # CUBES-MINT-REACH rule (a): the zone is clipped to the band the arm can pick from on this layout
        # (``reach.reach_zone``), so the back row is scattered within the grasp envelope instead of being
        # drawn beyond it and refused; the refusal below stays as the fail-closed check on the result.
        placed = S.scatter(poses, cube_count,
                           H.scatter_zones(skeleton, RCH.reach_zone(cube_zone(cls.FOOTPRINT), layout_id)),
                           cube_footprints(skeleton), MIN_GAP_M,
                           keepouts=list(cube_keepouts(footprint=cls.FOOTPRINT).values()),
                           keepout_gap=KEEPOUT_GAP_M, yaw_range=CUBE_YAW_RANGE, max_tries=SCATTER_MAX_TRIES)
        cubes = [dict(name=name, xy=[x, y], yaw=yaw, target=index in targets)
                 for index, (name, (x, y, yaw)) in enumerate(zip(names, placed))]
        refused = RCH.reach_problems([(cube['name'], cube['xy']) for cube in cubes], layout_id, style_id)
        if refused:
            raise ValueError('the robot cannot reach every cube: ' + '; '.join(refused))
        # Now that the cubes stand somewhere, every cover candidate is gated the way the build will gate
        # it (:func:`hiding.walk_covers`): the knob's reach for a cloche, the slide and the lift at the
        # edge for a bowl, against the keep-outs the build keeps. The entry keeps the first candidate and
        # the rest are the fallback the build walks. The scatter above still reserves the *drawn* kind's
        # disc, which is the widest of the two whenever a bowl was drawn, so the walk never has to move a
        # cube and the placement of a v3 seed is unchanged.
        skeleton = dict(skeleton, cubes=cubes, placement=layout)
        hidden, hidden_plan = H.walk_covers(skeleton, hiding_shapes(skeleton, cls.FOOTPRINT), FRAME_HALF)
        return dict(cube_count=cube_count, target_count=target_count, rule=rule, cubes=cubes, bins=bins,
                    observability=level, hidden=hidden, hidden_fallbacks=hidden_fallbacks,
                    hidden_plan=hidden_plan,
                    occluders=occluders, mark_seed=mark_seed, visible_plan=visible_plan,
                    placement=layout, footprint=list(cls.FOOTPRINT), bin_interior_height=BIN_INTERIOR_HEIGHT_M,
                    assets=dict(plate=PLATE_ASSET, plate_scale=PLATE_SCALE))

    @classmethod
    def validate_spec(cls, spec):
        problems = list(M.validate_rule(spec.get('rule', {})))
        cube_count, target_count = spec.get('cube_count'), spec.get('target_count')
        if cube_count not in cls.FACTORS['cube_count'] or target_count not in cls.TARGET_COUNTS:
            return problems + ['cube_count or target_count outside the factor levels']
        cubes = spec.get('cubes', [])
        names = [c.get('name') for c in cubes]
        if names != list(M.CUBE_NAMES[:cube_count]):
            problems.append('cube names must be the first cube_count of ' + ', '.join(M.CUBE_NAMES))
        if sum(bool(c.get('target')) for c in cubes) != target_count:
            problems.append('the number of target cubes must equal target_count')
        for cube in cubes:
            xy, yaw = cube.get('xy'), cube.get('yaw')
            if not isinstance(xy, list) or len(xy) != 2 or not all(isinstance(v, (int, float)) for v in xy):
                problems.append(f"{cube.get('name')}: xy must be a scattered [x, y] pair")
            elif any(abs(float(v)) > half for v, half in zip(xy, FRAME_HALF)):
                problems.append(f"{cube.get('name')}: centre outside the frame")
            if not isinstance(yaw, (int, float)) or not CUBE_YAW_RANGE[0] - 1e-9 <= float(yaw) <= CUBE_YAW_RANGE[1] + 1e-9:
                problems.append(f"{cube.get('name')}: yaw must be a radian value in {list(CUBE_YAW_RANGE)}")
            if 'slot' in cube:
                problems.append(f"{cube.get('name')}: the v0 slot grid is gone, cubes are scattered")
        if list(spec.get('footprint', cls.FOOTPRINT)) != list(cls.FOOTPRINT):
            problems.append('footprint must match the task layout')
        if not isinstance(spec.get('mark_seed'), int) or spec['mark_seed'] < 0:
            problems.append('mark_seed must be a non-negative integer')
        plan = spec.get('visible_plan', {})
        upper = 2 if spec.get('rule', {}).get('kind') in M.RULE_ATTRIBUTES else 1
        if set(plan) != set(names) or not all(isinstance(v, int) and 0 <= v <= upper for v in plan.values()):
            problems.append('visible_plan must give every cube a visible mark count within the rule limit')
        occluders = spec.get('occluders', [])
        if occluders:
            problems.extend(cls.clutter_kind_problems(occluders))
            covered = [o.get('cube') for o in occluders]
            known = {name for _, name in OCCLUDER_SET}
            if (len(occluders) > min(len(OCCLUDER_SET), int(cube_count) - OCCLUDER_SPARE_CUBES)
                    or len({o.get('name') for o in occluders}) != len(occluders)
                    or not {o.get('name') for o in occluders} <= known
                    or len(set(covered)) != len(covered) or not set(covered) <= set(names)):
                problems.append(f'occluders must be distinct plates ({", ".join(sorted(known))}) on distinct '
                                f'cubes, at most {len(OCCLUDER_SET)} and {OCCLUDER_SPARE_CUBES} cubes always bare')
            names_placed = {c.get('name') for c in cubes}
            for occluder in occluders:
                if occluder.get('cube') not in names_placed:
                    problems.append(f"{occluder.get('name')}: covers no placed cube")
                if occluder.get('kind') == 'plate':
                    direction = occluder.get('direction', [0, 0])
                    if occluder.get('style') not in ('flat', 'bridge') or abs(math.hypot(*direction) - 1) > 1e-3:
                        problems.append(f"{occluder.get('name')}: plate needs a style and a unit direction")
        layout = frame_layout(cls.FOOTPRINT)
        if spec.get('placement') != layout:
            problems.append('the two bins and Submit must stand where the layout logic puts them')
        bins = spec.get('bins', [])
        if ([b.get('lid') for b in bins] != list(BIN_CLASSES)
                or sorted(b.get('side') for b in bins) != sorted(BIN_SIDES)
                or any(b.get('name') != BIN_NAMES[b['lid']] for b in bins)
                or any(b.get('xy') != layout.get(b.get('side')) for b in bins)
                or any(len(b.get('body_rgba', ())) != 4 for b in bins)):
            problems.append('there must be one blue and one yellow bin, one per side, at the layout spots')
        if spec.get('bin_interior_height') != BIN_INTERIOR_HEIGHT_M:
            problems.append('bin interior height must match the task layout')
        # Observability: the level is a factor level, the entries follow observe's schema, and no hidden cube
        # also carries a face-occluding plate.
        levels = tuple(cls.FACTORS['observability']) + tuple(cls.DEV_ONLY_LEVELS['observability'])
        if spec.get('observability') not in levels:
            problems.append(f"observability must be one of {levels}")
        problems.extend(OB.validate_hidden(spec, objects=names))
        hidden_names = {e.get('object') for e in spec.get('hidden') or () if isinstance(e, dict)}
        clash = sorted(hidden_names & {o.get('cube') for o in occluders})
        if clash:
            problems.append('a face occluder may not stand on a hidden cube: ' + ', '.join(clash))
        for entry in spec.get('hidden') or ():
            if isinstance(entry, dict) and entry.get('mode') == 'cover' and entry.get('cover') not in H.COVER_KINDS:
                problems.append(f"{entry.get('object')}: cover must be one of {H.COVER_KINDS}")
        if not problems:
            # Tampering: the covers and the plan must be the ones the mint-time walk produces from this
            # spec's own geometry, so a cover the walk would have rejected (or demoted behind a kind it
            # could prove) cannot be pasted into a spec and carried into a build that cannot lift it.
            walked, plan = H.walk_covers(spec, hiding_shapes(spec, cls.FOOTPRINT), FRAME_HALF)
            covers = [e for e in spec.get('hidden') or () if e.get('mode') == 'cover']
            if [e for e in walked if e.get('mode') == 'cover'] != covers:
                problems.append('every cover must be the first candidate of the cover walk: '
                                + ', '.join(f"{row['object']} {row['candidates'][0][1]}"
                                            for row in plan['covers'] if row['candidates']))
            elif json.loads(json.dumps(spec.get('hidden_plan'))) != plan:
                problems.append('hidden_plan must be the record the cover walk produced')
        if not problems:
            problems.extend(cls.rule_wording_problems(spec))
        return problems

    @classmethod
    def clutter_kind_problems(cls, occluders):
        """v6: a clutter occluder is a plate; an inverted bowl over a cube is a full cover
        and belongs to the uncover level's ``spec['hidden']``, never to ``spec['occluders']``."""
        wrong = [f"{o.get('name')} ({o.get('kind')})" for o in occluders if o.get('kind') not in CLUTTER_KINDS]
        if wrong:
            return ['clutter occluders are plates only (generator v6); a bowl over a cube is '
                    'an uncover-level cover, not clutter: ' + ', '.join(wrong)]
        return []

    @classmethod
    def strip_occluders(cls, spec, kinds=('bowl',)):
        """A copy of ``spec`` without the occluders of these ``kinds`` - how a v5 row that carried a clutter
        bowl is derived into its v6 twin - with everything else kept: cubes, poses, yaws, rule, targets,
        mark seed, visible plan, bins, plates and their lean directions.

        Only the cover walk is re-derived, because :func:`hiding_shapes` handed the removed occluders' discs
        to it as keep-outs and :meth:`validate_spec` re-derives the walk: with fewer keep-outs a candidate's
        verdict can only stay or improve, so nothing tightens; the ``hidden_plan`` record follows (fewer
        clashes or blockers named), and a drawn kind that only the removed disc had refused comes back as
        the first candidate. Returns ``(spec, removed_names, cover_changes)`` where ``cover_changes`` lists
        ``(cube, kind_before, kind_after)`` for every cover the re-walk chose differently, so the caller can
        report them; the scatter and the level's look entries are untouched.
        """
        spec = deepcopy(spec)
        removed = [o['name'] for o in spec.get('occluders') or () if o.get('kind') in kinds]
        spec['occluders'] = [o for o in spec.get('occluders') or () if o.get('kind') not in kinds]
        assets = dict(spec.get('assets') or {})
        for key in tuple(assets):          # the bowl asset served the clutter bowls alone
            if key.split('_')[0] in kinds:
                assets.pop(key)
        spec['assets'] = assets
        before = H.cover_entries(spec)
        hidden, plan = H.walk_covers(spec, hiding_shapes(spec, cls.FOOTPRINT), FRAME_HALF)
        spec['hidden'], spec['hidden_plan'] = hidden, plan
        after = H.cover_entries(spec)
        changes = [(name, before[name], after.get(name)) for name in before if before[name] != after.get(name)]
        return spec, removed, changes

    @classmethod
    def force_cover_kind(cls, spec, kind):
        """A copy of ``spec`` with every cover set to ``kind`` (domes only in dev, 10 dome and 8
        bowl rows in the evaluation uncover cell), everything else kept. The kind is written as the plan's
        drawn kind, so the cover walk tries it first and ``validate_spec`` re-derives the same plan; if the
        walk cannot prove it for a cube (a bowl's wider disc or slide path, a cloche's knob reach) the walk
        keeps another kind and the change is reported, so the caller refuses the row.
        Returns ``(spec, changes, problems)``; ``changes`` lists ``(cube, kind_before, kind_after)``."""
        if kind not in H.COVER_KINDS:
            raise ValueError(f'cover kind must be one of {H.COVER_KINDS}')
        spec = deepcopy(spec)
        before = H.cover_entries(spec)
        spec['hidden'] = [OB.hidden_entry(e['object'], 'cover', kind) if e.get('mode') == 'cover' else e
                          for e in spec.get('hidden') or ()]
        plan = deepcopy(spec.get('hidden_plan') or {})
        for row in plan.get('covers') or ():
            if isinstance(row, dict):
                row['drawn'] = kind
        spec['hidden_plan'] = plan
        hidden, plan = H.walk_covers(spec, hiding_shapes(spec, cls.FOOTPRINT), FRAME_HALF)
        spec['hidden'], spec['hidden_plan'] = hidden, plan
        after = H.cover_entries(spec)
        changes = [(name, before[name], after.get(name)) for name in before if before[name] != after.get(name)]
        problems = [f'{name}: the cover walk cannot keep {kind} (kept {after.get(name)})'
                    for name in before if after.get(name) != kind]
        return spec, changes, problems

    @classmethod
    def rule_wording_problems(cls, spec):
        """The rule text must name exactly the attributes that separate this instance's targets.

        The marks are realised at reset from the frozen spec and the faces the settled geometry proves
        hidden, so the wording is checked over every hidden-face configuration those yaws can produce
        (:func:`hidden_configurations`, the gate's enumeration). A spec whose wording is not minimal and
        sufficient under one of them is rejected by :meth:`validate_spec`, which mint raises on.
        """
        names = [cube['name'] for cube in spec['cubes']]
        targets = [cube['name'] for cube in spec['cubes'] if cube['target']]
        problems = set()
        for hidden in hidden_configurations(names):
            try:
                marks = M.realize_marks(spec, hidden)['marks']
            except ValueError:
                continue          # a realisation that cannot be made is the CPU gate's report, not this one
            problems.update(M.rule_minimality_problems(spec['rule'], marks, targets))
        return sorted(problems)

    @classmethod
    def geometry_report(cls, spec):
        """Clearances in the task frame, re-deriving exactly what the scatter enforced: every cube-and-occluder
        disc inside the cube zone, pairwise clear and clear of both bins and Submit; then the occluders at the
        spots they will actually be hung on, clear of every prop they do not cover."""
        zone = cube_zone(cls.FOOTPRINT)
        keepouts = cube_keepouts(spec, cls.FOOTPRINT)
        names = [c['name'] for c in spec['cubes']]
        check = S.check(cube_poses(spec), cube_footprints(spec), bounds=zone, min_gap=MIN_GAP_M,
                        keepouts=list(keepouts.values()), keepout_gap=KEEPOUT_GAP_M,
                        names=names, keepout_names=list(keepouts))
        problems = list(check['problems'])
        report = dict(problems=problems, min_pair_gap_m=check['min_pair_gap_m'],
                      min_keepout_gap_m=check['min_keepout_gap_m'],
                      min_zone_margin_m=check['min_bounds_margin_m'])
        for name, keepout in keepouts.items():
            if S.region_margin(keepout, zone) < 0.:
                problems.append(f'{name}: outside the footprint')
        # Every covered cube inside the band whose cover the robot can reach from its reset stance.
        band = H.cover_zone(zone)
        for name in H.cover_entries(spec):
            cube = next((c for c in spec['cubes'] if c['name'] == name), None)
            if cube is not None and float(cube['xy'][1]) > band.y1:
                problems.append(f'{name}: covered cube at y {cube["xy"][1]:+.3f} is beyond the '
                                f'cover-reach band (y <= {band.y1:+.3f})')
        report['cover_band_y_max_m'] = round(float(band.y1), 4)
        # Two bins on one frame: the gaps between the fixed props are part of the G1 evidence (spec 4.1).
        fixed = {f'{a}-{b}': round(S.clearance(keepouts[a], keepouts[b]), 5)
                 for a, b in combinations(sorted(keepouts), 2)}
        report['fixed_prop_gaps_m'] = fixed
        report['min_fixed_prop_gap_m'] = round(min(fixed.values()), 5)
        report['min_bin_to_zone_edge_m'] = round(min(S.region_margin(keepouts[side], zone) for side in BIN_SIDES), 5)
        for pair, gap in sorted(fixed.items()):
            if gap < 0.:
                problems.append(f'{pair}: the fixed props overlap')
        # The clearance the user asked for is between the cubes themselves, at their realised yaws.
        boxes = {c['name']: S.shape_at(CUBE_HALF_XY, c['xy'], c['yaw']) for c in spec['cubes']}
        gaps = [S.clearance(boxes[a], boxes[b]) for a, b in combinations(names, 2)]
        report['min_cube_gap_m'] = round(min(gaps), 5) if gaps else None
        for (a, b), gap in zip(combinations(names, 2), gaps):
            if gap < MIN_GAP_M:
                problems.append(f'{a} and {b} closer than {MIN_GAP_M} m')
        # Occluders where build_task will put them: inside the frame and touching nothing but their own cube.
        shapes = dict(boxes, **keepouts)
        covers = {name: name for name in names}          # prop -> the cube it belongs to, if any
        min_clearance = math.inf
        for occluder in spec['occluders']:
            centre, radius = occluder_placement(spec, occluder)
            disc = S.Disc((float(centre[0]), float(centre[1])), radius)
            if S.region_margin(disc, zone) < 0.:
                problems.append(f"{occluder['name']}: outside the footprint")
            for name, other in sorted(shapes.items()):
                if covers.get(name) == occluder['cube']:
                    continue     # the cube this occluder covers, or another occluder on it
                clearance = S.clearance(disc, other)
                min_clearance = min(min_clearance, clearance)
                if clearance < OCCLUDER_MIN_CLEARANCE_M:
                    problems.append(f"{occluder['name']} and {name} would collide")
            shapes[occluder['name']] = disc
            covers[occluder['name']] = occluder['cube']
        report['min_occluder_clearance_m'] = None if min_clearance is math.inf else round(min_clearance, 5)
        return report

    @classmethod
    def cpu_gate(cls, spec):
        """G0: frame geometry plus symmetric hiding under every plausible hidden-face configuration."""
        report = cls.geometry_report(spec)
        report['problems'].extend(cls.clutter_kind_problems(spec.get('occluders') or ()))
        names = [c['name'] for c in spec['cubes']]
        targets = [c['name'] for c in spec['cubes'] if c['target']]
        checked = []
        configurations = [{n: faces for n in names} for faces in HYPOTHETICAL_HIDDEN]
        # Free yaws make neighbouring cubes hide different faces; check mixed configurations too.
        for shift in range(1, MIXED_HIDDEN_SHIFTS + 1):
            configurations.append({n: HYPOTHETICAL_HIDDEN[(index + shift) % len(HYPOTHETICAL_HIDDEN)]
                                   for index, n in enumerate(names)})
        for hidden in configurations:
            try:
                realised = M.realize_marks(spec, hidden)
            except ValueError as error:
                report['problems'].append(f'realisation failed: {error}')
                continue
            problems = M.check_symmetric_hiding(spec['rule'], realised['marks'], hidden, targets)
            if realised['targets'] != targets:
                problems.append('realised targets differ from the spec')
            report['problems'].extend(problems)
            checked.append(sorted({tuple(f) for f in hidden.values()}))
        report['hidden_configurations_checked'] = len(checked)
        return not report['problems'], report

    @classmethod
    def split_key(cls, spec):
        """The structure class the hold-out works on: (cube_count, target_count, rule)."""
        return f"{spec['rule']['kind']}-c{spec['cube_count']}-t{spec['target_count']}"

    @classmethod
    def bin_for(cls, spec, target):
        """Which lid a cube of this class belongs under: blue satisfies the rule, yellow does not."""
        return BIN_CLASSES[0] if target else BIN_CLASSES[1]

    # ---- scene protocol -----------------------------------------------------
    def _frame_offsets(self, fixture):
        """The base offsets, plus the fixtures hanging over this surface so rule (c) can be asked per offset."""
        frame = super()._frame_offsets(fixture)
        frame['overhead'] = HR.overhead_boxes(self, frame['top_z'])
        return frame

    def _corridor_blockers(self, fixture, frame, dx):
        """Rule (c) at a candidate place: wall units hanging into the carry corridor of either bin.

        The frame origin of the candidate is where :meth:`place_work_frame` would put it, so the corridors
        are measured for the placement being judged and not for the one already chosen.
        """
        center = self.fixture_local_to_world(fixture, [frame['rx'] + dx, frame['cy'], fixture.size[2] / 2])
        rotation = yaw_matrix(float(frame['yaw']))[:2, :2]
        spots = frame_layout(self.FOOTPRINT)
        return RCH.corridor_blockers(
            frame['overhead'], lambda xy: np.asarray(center, float)[:2] + rotation @ np.asarray(xy, float),
            frame['top_z'], [spots[side] for side in BIN_SIDES], -self.FOOTPRINT[1] / 2)

    def _find_frame_offset(self, fixture, frame=None):
        """The base search (first offset whose footprint misses the decor) with rule (c) added to it.

        A place under a wall cabinet is refused exactly as a place standing in the decor is: the frame slides
        to the next offset, then to the next surface, and if nothing is left the build refuses the seed and
        ``mint`` walks on. Without this the search took the largest decor-free region and never looked up,
        which is how 25 of the 54 v4 evaluation instances ended up with a bin the arm cannot reach into
        (and how most of the v4 cover-headroom gate failures happened). Deterministic: no RNG, and the decor
        the spec's own scene places is already fixed when the frame is chosen.
        """
        frame = frame or self._frame_offsets(fixture)
        blocked = []
        for dx in frame['offsets']:
            box = self._footprint_world_box([frame['rx'] + dx, frame['cy']], frame['yaw'], fixture)
            hits = [d['name'] for d in frame['decor'] if self._boxes_overlap(box, d)]
            hits += [name for name in self._corridor_blockers(fixture, frame, dx) if name not in hits]
            if not hits:
                return float(dx), blocked
            blocked.append((float(dx), hits))
        return None, blocked

    def _choose_work_fixture(self):
        """Base rule (largest island, else counter) refined: among qualifying surfaces, the largest whose front
        region admits a footprint clear of decor and counter-top fixtures; the base choice when none does."""
        fallback = super()._choose_work_fixture()
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
                if regions:
                    candidates.append((-max(r['size'][0] * r['size'][1] for r in regions.values()), name))
            for _, name in sorted(candidates):
                if self._clear_frame_offset(self.fixtures[name]) is not None:
                    return self.fixtures[name]
        return fallback

    def _clear_frame_offset(self, fixture):
        """The sideways offset place_work_frame() will choose on this fixture, or None if none is free.

        One search, one answer: :meth:`_find_frame_offset` is what the placement itself uses, so a surface is
        only nominated here when the offset it would take misses the decor *and* keeps the bin corridors
        clear.
        """
        return self._find_frame_offset(fixture)[0]

    def _decor_boxes(self, top_z):
        """Accessories (base class) plus counter-top fixtures whose bodies reach the work surface band."""
        boxes = super()._decor_boxes(top_z)
        seen = {box['name'] for box in boxes}
        for name, fixture in self.fixtures.items():
            if name in seen or not isinstance(fixture, COUNTER_TOP_FIXTURES):
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            if points[:, 2].max() < top_z - .02 or points[:, 2].min() > top_z + .25:
                continue     # below the counter, or wall-mounted well above it
            boxes.append(dict(name=name, x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                              y0=float(points[:, 1].min()), y1=float(points[:, 1].max())))
        return boxes

    def build_task(self):
        self.place_work_frame()
        spec = self.spec
        top, yaw = self.work['top_z'], self.work['yaw']
        self.cube_names = [c['name'] for c in spec['cubes']]
        self.occluder_names = [o['name'] for o in spec['occluders']]
        source = inspect.getfile(BoxObject)
        source_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        for cube in spec['cubes']:
            name = cube['name']
            obj = BoxObject(name=name, size=[M.HALF_SIZE_M] * 3, rgba=[.92, .92, .92, 1.], density=1000.,
                            friction=[1., .005, .0001])
            self.objects[name] = obj
            M.add_paintings(obj)
            position = self.frame_to_world([cube['xy'][0], cube['xy'][1], M.HALF_SIZE_M + .002])
            q = Rotation.from_euler('z', yaw + float(cube['yaw'])).as_quat()
            self._poses[name] = [*position.tolist(), float(q[3]), float(q[0]), float(q[1]), float(q[2])]
            self.task_spec['objects'][name] = dict(kind='cube', fixed=False, marks={}, frame_xy=list(cube['xy']),
                                                   frame_yaw_rad=float(cube['yaw']))
            self.asset_evidence[name] = dict(source=source, source_sha256=source_sha, side_m=M.SIDE_M,
                                             collision='Native solid BoxObject; visual-only face decals, every shape option '
                                                       'present on every face and selected by alpha at reset')
        for occluder in spec['occluders']:
            name = occluder['name']
            # The same frame-local spot the CPU gate measured: the occluder rides on the scattered cube.
            local, _ = occluder_placement(spec, occluder)
            if occluder['kind'] not in CLUTTER_KINDS:
                # v4/v5 built an inverted bowl here ('movable_covering_occluder'); v6 refuses it (item 33).
                raise ValueError('; '.join(self.clutter_kind_problems([occluder])))
            else:
                direction = np.array(occluder['direction'], float)
                if occluder['style'] == 'flat':
                    xy = self.frame_to_world([local[0], local[1], 0.])[:2]
                    self.add_native(name, spec['assets']['plate'], xy, top + M.SIDE_M, scale=spec['assets']['plate_scale'])
                else:
                    world = self.frame_to_world([local[0], local[1], BRIDGE_Z])
                    self.add_native(name, spec['assets']['plate'], world[:2], top, scale=spec['assets']['plate_scale'])
                    direction_world = yaw_matrix(yaw) @ np.array([direction[0], direction[1], 0.])
                    axis = np.cross([0., 0., 1.], direction_world)
                    q = Rotation.from_rotvec(axis / np.linalg.norm(axis) * math.radians(BRIDGE_TILT_DEG)).as_quat()
                    self._poses[name] = [*world.tolist(), float(q[3]), float(q[0]), float(q[1]), float(q[2])]
                self.task_spec['objects'][name]['role'] = 'movable_partial_occluder'
                self.task_spec['objects'][name]['style'] = occluder['style']
            self.task_spec['objects'][name]['covers'] = occluder['cube']
            self.task_spec['objects'][name]['frame_xy'] = [round(float(local[0]), 6), round(float(local[1]), 6)]
        # Two bins (spec 4.1): 0.30 m interior, one blue lid and one yellow lid, nuisance body colour, each
        # with its own capture weld per cube, both turned with the frame.
        self.bin_specs = {}
        cube_bodies = {n: self.objects[n].root_body for n in self.cube_names}
        for entry in spec['bins']:
            floor = self.frame_to_world([entry['xy'][0], entry['xy'][1], .008])
            bin_spec = append_bin(self.model.worldbody, floor.tolist(), name=entry['name'],
                                  interior_height=float(spec['bin_interior_height']),
                                  lid_rgba=LID_RGBA[entry['lid']], body_rgba=entry['body_rgba'])
            self.model.worldbody.find(f"body[@name='{bin_spec['body_name']}']").set('quat', _vec(yaw_quat_wxyz(yaw)))
            bin_spec.update(yaw_rad=float(yaw), frame_local_xy=list(entry['xy']), lid=entry['lid'],
                            side=entry['side'], lid_rgba=list(LID_RGBA[entry['lid']]),
                            body_rgba=list(entry['body_rgba']))
            append_capture_constraints(self.model.root, bin_spec, cube_bodies)
            self.bin_specs[entry['lid']] = bin_spec
        self.add_submit_button(spec['placement']['submit'])
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, rule=deepcopy(spec['rule']),
            budget_ticks=BUDGET_TICKS, cube_ids=list(self.cube_names), occluder_ids=list(self.occluder_names),
            bins=deepcopy(self.bin_specs), bin_sides={e['lid']: e['side'] for e in spec['bins']},
            frame_origin_world=self.work['center_world'], frame_yaw=self.work['yaw'],
            placement='Cubes scattered over the cube zone of the frame at uniformly random positions and yaws, '
                      'each sampled with the bounding disc of its cube-and-occluder pair so that nothing is '
                      'aligned, nothing overlaps and every occluder is clear of the other props, of both bins '
                      'and of Submit (roboquest.scatter).',
            scoring='First physical Submit freezes the score: every cube captured (welded) by the bin its class '
                    'belongs to - blue for the cubes that satisfy the rule, yellow for the rest. A cube in the '
                    'wrong bin is permanent. No submission fails.')
        self.asset_evidence.update(
            bins='Two procedural MJCF bins (0.30 m interior) with an inward spring-return flap and a native '
                 'capture weld per cube (cube inspection v4); lids blue and yellow, bodies coloured per instance',
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged')
        self.apply_hiding()

    # ---- observability (spec 2.2, contract C6) ------------------------------
    def hiding_rng(self):
        """Build-time stream for the hiding placement: a child of this instance's ``poses`` stream, so two
        builds of the same instance stand the annex cube, the occluder and the cover in the same places."""
        seed = int((getattr(self, 'instance', None) or {}).get('seed', 0))
        return REG.rng_for(self.TASK_NAME, seed)['poses'].spawn(1)[0]

    def hiding_keepouts(self):
        """Task-frame keep-outs every hidden placement must clear (:func:`hiding_shapes`, without names)."""
        return list(hiding_shapes(self.spec, self.FOOTPRINT).values())

    def apply_hiding(self):
        """Realise ``spec['hidden']`` at the end of the build (contract C6).

        Every prop is a keep-out, the cubes are described with their own geometry, and an occluded cube may
        be moved into the free area beyond the frame. The occluder-to-annex fallback is
        allowed and recorded; anything else observe cannot realise fails the build, because a spec that
        claims a level the scene does not show would be exactly the privileged difference between levels
        that the axis must not have.
        """
        spec = self.spec
        geometry = {cube['name']: dict(radius=H.CUBE_RADIUS, height=H.CUBE_HEIGHT) for cube in spec['cubes']}
        keepouts = self.hiding_keepouts()
        hidden, substitutions = self.feasible_cover_kinds(keepouts)
        record = OB.apply_observability(self, dict(spec, hidden=hidden), objects=geometry, keepouts=keepouts,
                                        rng=self.hiding_rng(),
                                        occluder_relocate=H.free_band(OB.work_surface_rect(self), FRAME_HALF))
        record['kind_fallbacks'] = substitutions
        if record['problems']:
            raise RuntimeError(f"painted_cubes cannot realise observability {spec.get('observability')!r}: "
                               + '; '.join(record['problems']))
        for item in record['realised']:
            entry = self.task_spec['objects'].get(item['object'])
            if entry is not None:
                entry['hidden_by'] = item['realised']
                if item.get('cover'):
                    entry['cover_kind'] = item['cover']
        for placed in record['annex'] + [o for o in record['occluders'] if o['relocated']]:
            name = placed.get('object') or placed.get('target')
            xy = placed.get('xy_world') or placed['relocated']['to_xy_world']
            local = OB.world_to_frame_xy(self, xy)
            self.task_spec['objects'][name]['frame_xy'] = [round(float(local[0]), 6), round(float(local[1]), 6)]
        self.task_spec.update(observability_level=spec.get('observability', 'visible'),
                              hidden_cube_ids=sorted(record['hidden_bodies']),
                              hiding_role_ids=sorted(self.hiding_role_objects()))
        return record

    def feasible_cover_kinds(self, keepouts):
        """``spec['hidden']`` with each cover replaced by the first candidate of the mint-time walk this
        kitchen can really realise.

        The candidate order is the walk's (:func:`hiding.walk_covers`): the drawn kind first, then the rest
        of the pool, with the ones the walk disproved at mint already dropped. Here the kitchen is standing,
        so each candidate gets the full dry run (:meth:`cover_buildable`) and the first that passes is kept;
        nothing is built, so this only decides which cover ``apply_observability`` is asked for. Which
        candidates were tried, and why the earlier ones failed, is recorded.

        Two sweeps. The first asks every asset the build rng could pick to pass, so the kind that is kept
        builds whatever the build draws; only if no candidate survives that does the second sweep fall back
        to the plain ``observe.cover_feasible``, which is what the build did before this walk existed. The
        second sweep can only add instances, never lose one, and it is recorded as ``optimistic``.
        """
        hidden, substitutions = [], []
        floor = OB.floor_map(self)
        plan = {str(row['object']): row for row in ((self.spec.get('hidden_plan') or {}).get('covers') or ())}
        for entry in self.spec.get('hidden') or ():
            if entry['mode'] != 'cover':
                hidden.append(dict(entry))
                continue
            name = str(entry['object'])
            order = [tuple(c) for c in ((plan.get(name) or {}).get('candidates') or ())]
            # A spec minted before the walk existed (generator v3) carries no plan: walk the whole pool for
            # it, exactly as the build did then, so an older registry keeps the covers it used to build.
            order = order or [(name, kind) for kind in H.cover_order(self.spec, name)]
            tried, kept = [], None
            for strict in (True, False):
                for candidate, kind in order:
                    ok, reasons = self.cover_buildable(candidate, kind, keepouts, floor, strict=strict)
                    tried.append(dict(object=candidate, cover=kind, ok=ok, strict=strict, reasons=reasons[:2]))
                    if ok:
                        kept = (candidate, kind, strict)
                        break
                if kept is not None:
                    break
            if kept is None:
                substitutions.append(dict(object=name, requested=entry['cover'], realised=None,
                                          candidates_tried=tried))
                hidden.append(dict(entry))
                continue
            candidate, kind, strict = kept
            if (candidate, kind) != (name, str(entry['cover'])) or not strict:
                substitutions.append(dict(object=name, requested=entry['cover'], realised=kind,
                                          optimistic=not strict, candidates_tried=tried))
            hidden.append(dict(entry, object=candidate, cover=kind))
        return hidden, substitutions

    def cover_buildable(self, name, kind, keepouts, floor, strict=True):
        """Dry run of :func:`observe.cover_object` for one candidate: ``(ok, reasons)``. Nothing is built.

        ``strict`` is the conservative reading, and it matters for the bowl: ``cover_feasible`` measures the
        *smallest* fitted bowl while ``cover_object`` draws one of the two from the build rng, so a kind kept
        on the strength of the small bowl loses the instance whenever the big one comes up. Measured on the
        WIRE-CC mini registry (2026-09-21): the wide bowl (radius .111 against the small one's .090) puts the
        rim .086 m ahead of the base after the slide to the front edge, inside the .10 m minimum, and its
        deeper body needs more headroom; four of the eight drawn bowls built and the other four failed G1
        with "bowl cover placed but its reach gate failed". Under ``strict`` every asset the build could pick
        has to pass, which is what the mugs walk does for its flat covers
        (:func:`mugs.hiding.flat_cover_feasible`). Pinning the asset instead would need ``cover_object`` to
        accept the one ``cover_feasible`` validated, which is observe's to change, not this task's.
        """
        if kind != 'bowl' or not strict:
            check = OB.cover_feasible(self, name, kind, radius=H.CUBE_RADIUS, height=H.CUBE_HEIGHT,
                                      keepouts=keepouts, floor=floor)
            return bool(check['ok']), list(check['reasons'])
        ext = OB.body_extent(self, name, radius=H.CUBE_RADIUS, height=H.CUBE_HEIGHT)
        xy = np.asarray(ext['xy'], float)
        reasons, fitted = [], 0
        for cover in OB.BOWL_COVERS:
            scale = OB.bowl_scale_for(cover, ext['radius'], ext['height'])
            if scale is None:
                continue
            fitted += 1
            radius = float(cover['radius']) * scale
            tag = str(cover['asset']).rsplit('/', 1)[-1]
            gate = OB.slide_gate(self, xy, radius, ext['radius'], keepouts=keepouts, floor=floor)
            # _bowl_headroom drops the options without room overhead and re-decides ``passed``.
            OB._bowl_headroom(self, gate, xy, radius, scale * (OB._bowl_depth_at(cover, 0.) + .01))
            clear, clashes = OB.cover_footprint_clear(self, xy, radius, keepouts)
            if not gate['passed']:
                reasons.append(f'{tag}: ' + '; '.join(gate['reasons'][:1]))
            if not clear:
                reasons.append(f'{tag}: footprint clashes with {clashes} prop(s)')
        if not fitted:
            reasons.append('no bowl cover fits')
        return (not reasons), reasons

    def hiding_role_objects(self):
        """The props observability added (occluders and covers). They are not task objects: every loop that
        scores, measures progress or counts collateral skips them; the generic ``left_surface`` watcher and
        ``prop_clashes()`` keep them, because dropping one is still damage."""
        return {name for name, entry in self.task_spec['objects'].items() if entry.get('role') in HIDING_ROLES}

    def hidden_cube_modes(self):
        """``{cube: 'annex' | 'occluder' | 'cover'}`` as realised, falling back to the spec before the build."""
        record = self.task_spec.get('observability') or {}
        realised = {str(item['object']): str(item['realised']) for item in record.get('realised', ())}
        return realised or {e['object']: e['mode'] for e in (self.spec.get('hidden') or ())}

    def hidden_cube_covers(self):
        """``{cube: cover kind}`` as realised."""
        record = self.task_spec.get('observability') or {}
        return {str(item['object']): str(item['cover']) for item in record.get('realised', ())
                if item.get('realised') == 'cover' and item.get('cover')}

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self.model.root.find('option').set('timestep', str(PHYSICS_TIMESTEP))
        self.add_cube_contact_pairs()

    def add_cube_contact_pairs(self):
        """Stiff explicit contact pairs between the cubes (cube inspection v4), added at most once each."""
        contact = self.model.root.find('contact')
        if contact is None:
            contact = ET.SubElement(self.model.root, 'contact')
        # RoboCasa's Kitchen._load_model retries itself (``self._load_model(attempt_num + 1)``) whenever a
        # fixture or an object cannot be placed, and the retry runs this override again on the model the
        # inner attempt left behind. Adding the same pair twice is a MuJoCo compile error ("repeated name
        # 'cube_a_g0__cube_b_g0_rigid' in pair"), which is what dropped nine candidates of the first v1
        # registry on the layouts that need a retry (34 styles 3/6/8, 46). Adding each pair once is enough.
        present = {pair.get('name') for pair in contact.findall('pair')}
        pairs = []
        for first, second in combinations(self.cube_names, 2):
            for ga in self.objects[first].contact_geoms:
                for gb in self.objects[second].contact_geoms:
                    name = f'{ga}__{gb}_rigid'
                    if name not in present:
                        ET.SubElement(contact, 'pair', name=name, geom1=ga, geom2=gb, condim='3',
                                      friction='1 1 .005 .0001 .0001', solref='.002 1', solimp='.99 .999 .001')
                        present.add(name)
                    pairs.append([ga, gb])
        self.task_spec['cube_impact_physics'] = dict(timestep_s=PHYSICS_TIMESTEP, public_tick_s=.05, cube_pairs=pairs,
                                                     solref=[.002, 1], solimp=[.99, .999, .001])
        return pairs

    def initialize_time(self, control_freq):
        super().initialize_time(control_freq)
        self.model_timestep = PHYSICS_TIMESTEP
        self.sim.model.opt.timestep = self.model_timestep

    def setup_task_references(self):
        m = self.sim.model
        self._cube_bids = {n: self.obj_body_id[n] for n in self.cube_names}
        self._cube_gids = {n: self._collision_geom_ids(n) for n in self.cube_names}
        self._bin_captures = {lid: BinCapture(m, self.sim.data, self._cube_bids, self._cube_gids, spec)
                              for lid, spec in self.task_spec['bins'].items()}
        # Which bin each cube belongs in; filled in by assign_marks once the classes are realised.
        self._cube_bin = {}
        self._paint_gids = {(n, face, shape): [m.geom_name2id(g) for g in M.paint_geom_names(n, face, shape)]
                            for n in self.cube_names for face in M.FACE_NAMES for shape in M.SHAPES}
        self._camera_ids = {camera: m.camera_name2id(camera) for camera in CAMERAS}
        self._hidden_mark_gids = []

    def _reset_internal(self):
        super()._reset_internal()
        for capture in self._bin_captures.values():
            capture.reset()
        self._spawn_xy = {name: np.array(self.sim.data.body_xpos[self.obj_body_id[name]][:2]) for name in self.objects}
        self._settle(SETTLE_TICKS)
        self.assign_marks()
        passed, report = self.build_gate()
        self.task_spec['build_gate'] = report
        if not passed:
            raise RuntimeError('painted_cubes build gate failed: ' + '; '.join(report['problems']))

    def build_gate(self):
        """G1 at reset, fail closed: frame clear of counter decor, every cube resting on the work surface where it
        was spawned, occluders where they were placed, score zero. Returns (passed, report).

        A cube the instance hides in the annex stands on another surface (or far along this one) and a cube
        inside the wide cup rests on the cup's floor, so both are exempt from the work-surface test; their
        placement was gated by observe (reach, cameras, decor) when it was made and ``prop_clashes()`` is the
        physical backstop at G1, counted here as evidence.
        """
        problems = []
        top, fixture = self.work['top_z'], self.fixtures[self.work['fixture']]
        frame_box = self._footprint_world_box(self.work['center_local'], self.work['yaw'], fixture)
        decor = self._decor_boxes(top)
        decor_in_frame = sorted(d['name'] for d in decor if self._boxes_overlap(frame_box, d))
        decor_hits = sorted({f"{name}/{d['name']}" for name, box in self._prop_world_boxes().items() for d in decor
                             if self._boxes_overlap(box, d, margin=.005)})
        if decor_hits:
            problems.append('props overlap counter decor or fixtures: ' + ', '.join(decor_hits))
        d = self.sim.data
        drift = {}
        for name in self.objects:
            drift[name] = float(np.linalg.norm(np.array(d.body_xpos[self.obj_body_id[name]][:2]) - self._spawn_xy[name]))
            limit = .01 if name in self.cube_names else .03
            if drift[name] > limit:
                problems.append(f'{name} moved {drift[name]:.3f} m while settling')
        score = self._current_score()
        modes, covers = self.hidden_cube_modes(), self.hidden_cube_covers()
        off_top = {name for name, mode in modes.items()
                   if mode == 'annex' or covers.get(name) == 'wide_cup'}
        for name, cube in score['cubes'].items():
            if not cube['on_surface'] and name not in off_top:
                problems.append(f'{name} is not resting on the work surface')
        if score['success']:
            problems.append('score is not zero at reset')
        if self.task_progress() != 0.:
            problems.append('progress is not zero at reset')
        clashes = self.prop_clashes()
        return not problems, dict(problems=problems, decor_hits=decor_hits, decor_in_frame=decor_in_frame,
                                  settle_drift_m={k: round(v, 4) for k, v in drift.items()},
                                  observability=self.task_spec.get('observability_level', 'visible'),
                                  hidden_cubes=dict(sorted(modes.items())), cover_kinds=dict(sorted(covers.items())),
                                  hiding_roles=sorted(self.hiding_role_objects()),
                                  prop_clashes=len(clashes), worst_prop_clash=clashes[:1],
                                  cubes_on_surface=all(c['on_surface'] or n in off_top
                                                       for n, c in score['cubes'].items()),
                                  bin_mouth_z_m={lid: round(float(spec['opening_center_world'][2]), 4)
                                                 for lid, spec in self.task_spec['bins'].items()})

    def _prop_world_boxes(self):
        """World xy boxes of every prop at its spawn position: cubes, occluders, both bins and Submit.

        The props observability added and the cubes it hides are left out: they stand where observe put them,
        which is off this frame (the annex, the free area beyond it) and was gated against the work top's
        decor there. This box test knows only one radius per name and one decor height, so it would read a
        cube on a second counter as overlapping the first counter's decor.
        """
        boxes = {}
        skip = self.hiding_role_objects() | set(self.hidden_cube_modes())
        for name in self.objects:
            if name in skip:
                continue
            radius = CUBE_HALF_DIAGONAL if name in self.cube_names else BOWL_RADIUS if name.startswith('bowl') else PLATE_RADIUS
            x, y = self._spawn_xy[name]
            boxes[name] = dict(x0=x - radius, x1=x + radius, y0=y - radius, y1=y + radius)
        for bin_spec in self.task_spec['bins'].values():
            half = np.abs(yaw_matrix(self.work['yaw'])[:2, :2]) @ np.array(bin_spec['exterior_size_xyz'][:2]) / 2
            bx, by = bin_spec['position_xyz_floor'][:2]
            boxes[bin_spec['body_name']] = dict(x0=bx - half[0], x1=bx + half[0], y0=by - half[1], y1=by + half[1])
        sx, sy = self.submit_xy
        boxes['submit'] = dict(x0=sx - .048, x1=sx + .048, y0=sy - .043, y1=sy + .043)
        return boxes

    def _settle(self, ticks):
        """Zero-action control ticks with the arm controller holding pose, as RoboCasa's own reset settle."""
        action = np.zeros(self.action_spec[0].shape)
        policy_step = True
        for _ in range(int(ticks) * int(self.control_timestep / self.model_timestep)):
            self.sim.step1()
            self._pre_action(action, policy_step)
            self.sim.step2()
            policy_step = False
        self.sim.forward()

    # ---- marks ---------------------------------------------------------------
    def camera_positions(self):
        d = self.sim.data
        return {camera: d.cam_xpos[cid].copy() for camera, cid in self._camera_ids.items()}

    def reflective_planes(self, range_m=M.MIRROR_RANGE_M):
        """World ``(point, normal)`` planes of every reflective surface near the work frame.

        MuJoCo's mirror pass draws the scene once more in every reflective plane and in the *front* face
        of every reflective box - its own local +z, the convention a plane follows - so those are the
        only surfaces that can turn a camera into a virtual one. A kitchen wall at reflectance 0.1 is
        enough to show the back of a cube: on layout 47 style 8 a black star self-occluded from all three
        cameras painted 29 and 35 pixels of the two agentview images, reflected in ``wall_right_1_room``
        (measured 2026-09-21). Only surfaces within ``range_m`` of the work frame and facing it are
        collected; the rest cannot put a cube in front of a policy camera.
        """
        m, d = self.sim.model, self.sim.data
        top_z = float(self.work['top_z'])
        centre = np.asarray(self.work['center_world'], dtype=float)
        centre = np.array([centre[0], centre[1], top_z])
        eyes = np.asarray(list(self.camera_positions().values()), dtype=float)
        planes = []
        for gid in range(m.ngeom):
            matid = int(m.geom_matid[gid])
            if matid < 0 or float(m.mat_reflectance[matid]) < M.MIRROR_MIN_REFLECTANCE:
                continue
            kind = int(m.geom_type[gid])
            if kind not in (0, 6):        # mjGEOM_PLANE, mjGEOM_BOX
                continue
            origin = np.asarray(d.geom_xpos[gid], dtype=float)
            rotation = np.asarray(d.geom_xmat[gid], dtype=float).reshape(3, 3)
            normal = rotation[:, 2]
            point = origin + float(m.geom_size[gid][2]) * normal if kind == 6 else origin
            towards = centre - point
            distance = float(np.linalg.norm(towards))
            if distance > range_m or distance <= 1e-6 or float(towards @ normal) <= 0.:
                continue
            # A mirror that puts the camera under the work surface shows nothing that stands on it: the
            # counter is opaque, so the floor's reflection cannot see a cube, let alone the face it rests
            # on. Without this the floor would mirror every camera below the top and no cube would keep a
            # single hidden face (layout 47 style 8, the whole grid).
            mirrored = eyes - 2. * ((eyes - point) @ normal)[:, None] * normal
            if float(mirrored[:, 2].min()) <= top_z:
                continue
            planes.append((point.tolist(), normal.tolist()))
        return planes

    def visibility_reports(self):
        """Analytic self-occlusion proof per cube against the jittered settled cameras and their mirror
        images in the reflective surfaces around the work frame."""
        d = self.sim.data
        cameras = M.mirrored_cameras(M.jittered_cameras(self.camera_positions()), self.reflective_planes())
        return {n: M.self_occlusion_report(d.body_xpos[self._cube_bids[n]], d.body_xmat[self._cube_bids[n]].reshape(3, 3), cameras)
                for n in self.cube_names}

    def assign_marks(self):
        """Reset-only: realise the frozen mark plan on the faces proved hidden from the settled cameras."""
        if self.timestep != 0:
            raise RuntimeError('Cube marks may only be assigned during reset')
        self.sim.forward()
        m = self.sim.model
        reports = self.visibility_reports()
        hidden = {n: tuple(reports[n]['hidden_faces']) for n in self.cube_names}
        assignment = M.realize_marks(self.spec, hidden)
        problems = M.check_symmetric_hiding(self.spec['rule'], assignment['marks'], hidden, assignment['targets'])
        if problems:
            raise RuntimeError('symmetric hiding violated at reset: ' + '; '.join(problems))
        hidden_gids = []
        for n in self.cube_names:
            marks = assignment['marks'][n]
            for face in M.FACE_NAMES:
                mark = marks.get(face)
                for shape in M.SHAPES:
                    on = mark is not None and mark['shape'] == shape
                    for gid in self._paint_gids[(n, face, shape)]:
                        m.geom_rgba[gid, :3] = M.COLOUR_RGB[mark['colour']] if on else (.5, .5, .5)
                        m.geom_rgba[gid, 3] = 1. if on else 0.
                        if on and face in hidden[n]:
                            hidden_gids.append(gid)
            self.task_spec['objects'][n]['marks'] = deepcopy(marks)
            self.task_spec['objects'][n]['qualifies'] = n in assignment['targets']
            self.task_spec['objects'][n]['correct_bin'] = self.bin_for(self.spec, n in assignment['targets'])
        self._hidden_mark_gids = hidden_gids
        self._cube_bin = {n: self.bin_for(self.spec, n in assignment['targets']) for n in self.cube_names}
        self.task_spec['correct_bin'] = dict(self._cube_bin)
        self.task_spec['targets'] = list(assignment['targets'])
        self.task_spec['initial_hidden_faces'] = {n: list(hidden[n]) for n in self.cube_names}
        self.task_spec['mark_assignment'] = assignment
        self.task_spec['camera_positions_at_reset'] = {c: p.tolist() for c, p in self.camera_positions().items()}
        self.task_spec['camera_jitter_m'] = M.CAMERA_JITTER_M
        if getattr(self, 'has_offscreen_renderer', False):
            self.hidden_state_check()
        else:
            self.task_spec['hidden_state_check'] = dict(skipped=True, reason='no offscreen renderer (CPU build)')
        return assignment

    LEAK_TOLERANCE_PX = 4     # stable changed pixels per camera below which a mark or cube is not 'visible':
                              # anti-aliasing at a silhouette edge changes a pixel or two in both passes
                              # when a geom is erased or recoloured; a visible mark face is tens of pixels

    def hidden_state_check(self):
        """G3, fail closed: erasing every hidden-face mark must not change a pixel of any policy camera, and
        every cube the instance hides must be invisible in every policy camera while the bins that prove the
        diff works are visible (contract C6; :meth:`observability_check` says what is gated and what is only
        reported).

        Rendered in the state a policy observes: visualisation sites hidden (robosuite hides them at the
        end of reset, before the first observation). With sites showing, MuJoCo's translucent pass blends
        differently whenever the geom set changes, which would flag pixels unrelated to the marks.

        The erasure is rendered twice and a pixel counts as changed only when it changed in **both** passes,
        which is how :func:`observe.visible_in_cameras` reads its own diffs: EGL flickers by a handful of
        pixels on some kitchens, and demanding a bit-identical repeat aborted two valid eval candidates of
        the first v1 registry (layouts 9 and 10). The flicker is reported, never fatal on its own.
        """
        m = self.sim.model

        def frames():
            return {camera: self.sim.render(width=self.camera_widths[index], height=self.camera_heights[index],
                                            camera_name=camera).copy() for index, camera in enumerate(CAMERAS)}
        gids = list(self._hidden_mark_gids)
        site_alpha = m.site_rgba[:, 3].copy()
        alpha = m.geom_rgba[gids, 3].copy()
        try:
            m.site_rgba[:, 3] = 0.
            frames()   # warm-up: the first frame of a fresh offscreen context is not a valid reference
            original, repeat = frames(), frames()
            m.geom_rgba[gids, 3] = 0.
            erased, erased_again = frames(), frames()
        finally:
            m.geom_rgba[gids, 3] = alpha
            m.site_rgba[:, 3] = site_alpha
        unstable = H.unstable_pixels(original, repeat)
        unstable_erased = H.unstable_pixels(erased, erased_again)
        changed = H.flicker_tolerant_diff({c: (original[c], erased[c]) for c in CAMERAS},
                                          {c: (repeat[c], erased_again[c]) for c in CAMERAS})
        single = {c: int(np.any(original[c] != erased[c], axis=2).sum()) for c in CAMERAS}
        marks_passed = all(v <= self.LEAK_TOLERANCE_PX for v in changed.values())
        report = dict(skipped=False, changed_pixels=changed, changed_pixels_single_pass=single,
                      renderer_unstable_pixels=unstable, renderer_unstable_pixels_erased=unstable_erased,
                      hidden_mark_geoms=len(gids), marks_passed=marks_passed,
                      leak_tolerance_px=self.LEAK_TOLERANCE_PX,
                      image_size=[int(self.camera_widths[0]), int(self.camera_heights[0])])
        report.update(self.observability_check())
        problems = []
        if not marks_passed:
            problems.append('A hidden-face mark is visible in a reset camera: ' + json.dumps(changed))
        problems += ['Observability is not realised in the reset cameras: ' + p
                     for p in report.get('observability_problems', [])]
        report['problems'] = problems
        report['passed'] = bool(marks_passed and report.get('observability_passed', True))
        # Recorded, never raised: the reset must build the same instance every time, and the verdict is
        # G3's to act on (scripts/roboquest_gate_batch.py reads ``passed``). Raising here made a renderer
        # flicker of one pixel a build failure that came and went between runs (2026-09-21).
        self.task_spec['hidden_state_check'] = report
        return report

    def observability_check(self):
        """The observability half of G3: every hidden cube pixel-invisible in every policy camera, both bins
        visible, and a reachable viewpoint for each ``look`` cube (spec 2.2).

        The reach is the point of the level: a cube in the annex or behind an occluder has to be findable by
        driving and looking, so a look entry whose reach gate fails at reset fails the instance, exactly as
        the build-time gate in :func:`observe.annex_regions` would have.

        The cubes the level does **not** hide are measured and reported, never gated, and the bins are the
        control that proves the diff can see anything at all. Demanding that every other cube reach some
        camera failed instances the axis never touched: on layout 9 style 13 (2026-09-21) cube_c stands at
        the far end of a wall counter and reaches no policy camera at reset, and the same kitchen at the
        ``visible`` level - where nothing is checked - has exactly the same cube in exactly the same place.
        It was the single commonest G3 complaint of the first mini registry (18 of 34 problems, 6 instances
        failing on nothing else).

        Renderer instability is likewise reported rather than gated. The whole-image bit-compare in
        :func:`observe.visible_in_cameras` trips on the EGL flicker this job measured elsewhere (11 px with
        nothing toggled) and aborted three candidates with an empty problem list. Per-cube flicker - pixels
        that changed in one of the two colour passes but not the other - is no better as a gate: it is
        exactly the pixels the two-pass rule already refuses to count, and failing on it cost 8 more of the
        42 mini-registry candidates (2026-09-21) without a single one of them showing a stable pixel of a
        hidden cube. What is gated is the stable evidence: a hidden cube that differs in **both** passes of
        any policy camera.
        """
        modes = self.hidden_cube_modes()
        if not modes:
            return dict(observability=self.task_spec.get('observability_level', 'visible'),
                        observability_problems=[], observability_passed=True, hidden_cubes={}, look_reach={})
        hidden = sorted(modes)
        others = [n for n in self.cube_names if n not in modes]
        bins = [spec['body_name'] for spec in self.task_spec['bins'].values()]
        check = OB.hidden_check(self, hidden, visible=others + bins)
        # A hidden cube is a problem when the two-pass diff saw it and its stable pixel count is over the
        # tolerance; without counts (a checker that reports only the verdict) the verdict stands.
        counts = {n: max([v for v in check.get('pixels', {}).get(n, {}).values()] or [-1]) for n in hidden}
        leaked = [n for n in hidden if not check['hidden_ok'].get(n, True)
                  and (counts[n] < 0 or counts[n] > self.LEAK_TOLERANCE_PX)]
        problems = [f'{n} is visible in a policy camera' + (f' ({counts[n]} px)' if counts[n] >= 0 else '')
                    for n in leaked]
        problems += [f'the control {n} is in no policy camera' for n in bins if not check['visible_ok'].get(n, True)]
        unstable = sorted(n for n in hidden if any(check['flicker'].get(n, {}).values()))
        unseen = [n for n in others if not check['visible_ok'].get(n, True)]
        reach = {}
        for name, mode in sorted(modes.items()):
            if mode not in H.LOOK_MODES:
                continue
            gate = OB.look_reachable(self, name, at_build=False, radius=H.CUBE_RADIUS, height=H.CUBE_HEIGHT)
            reach[name] = dict(passed=bool(gate['passed']), surface=gate['surface'], stance=gate['stance'],
                               reasons=gate['reasons'][:2])
            if not gate['passed']:
                problems.append(f'{name}: no reachable viewpoint ({"; ".join(gate["reasons"][:2])})')
        return dict(observability=self.task_spec.get('observability_level', 'visible'),
                    observability_problems=problems,
                    observability_passed=not problems,
                    hidden_cubes=dict(sorted(modes.items())), hidden_pixels=check['pixels'],
                    hidden_flicker=check['flicker'], look_reach=reach, unstable_evidence=unstable,
                    cubes_in_no_camera=unseen, renderer_deterministic=check['deterministic'])

    # ---- scoring -----------------------------------------------------------
    def _post_action(self, action):
        grasped = [n for n in self.cube_names if self._check_grasp(self.robots[0].gripper['right'], self.objects[n])]
        for capture in self._bin_captures.values():
            capture.update(grasped=grasped)
        return super()._post_action(action)

    def captured_by_bin(self):
        """``{cube: lid}`` for every cube a bin has welded. A cube can be welded by at most one bin: the
        weld only fires when the cube is fully inside that bin's interior."""
        return {name: lid for lid, capture in self._bin_captures.items() for name in sorted(capture.captured)}

    def _arrangement_snapshot(self, step=0):
        snapshot = VisionKitchenScene.evaluator_snapshot(self, step)
        captured = self.captured_by_bin()
        snapshot['captured_by_bin'] = captured
        snapshot['discarded_cube_ids'] = sorted(captured)
        snapshot['bin_containment'] = {n: {lid: capture.check(n)['contained']
                                           for lid, capture in self._bin_captures.items()}
                                       for n in self.cube_names}
        return snapshot

    def _on_work_surface(self, state):
        """Upward support from the counter itself (not another prop, a bin or the button) at the counter height.

        Not part of the success predicate any more (spec 4.1: no cube is kept on the counter); the build gate
        still uses it to prove that every cube spawned resting on the work top.
        """
        top = self.work['top_z']
        for contact in state['support_contacts']:
            owner = contact.get('support_id', '')
            if owner in self.objects or owner.startswith(('cube_bin_', self.submit_prefix)):
                continue
            if support_contact(contact, height_z=top):
                return True
        return False

    def _current_score(self, states=None):
        """Spec 4.1: every cube captured (welded) by the bin its class belongs to. A weld only fires on a
        cube that is fully inside, released and settled, so there is nothing left to check besides the bin."""
        snapshot = self._arrangement_snapshot(self.timestep) if states is None else states
        assigned = 'correct_bin' in self.task_spec
        wanted = self.task_spec.get('correct_bin', {})
        captured = dict(snapshot['captured_by_bin'])
        cubes, errors = {}, []
        for n in self.cube_names:
            state = snapshot['objects'][n]
            landed, correct_bin = captured.get(n), wanted.get(n)
            correct = bool(assigned and landed is not None and landed == correct_bin)
            cubes[n] = dict(qualifies=n in set(self.task_spec.get('targets', ())), correct_bin=correct_bin,
                            captured_by=landed, wrong_bin=bool(landed is not None and landed != correct_bin),
                            on_surface=bool(self._on_work_surface(state)),
                            released=bool(state['grasped'] is False and state['robot_contact'] is False),
                            success=correct)
            if not correct:
                errors.append(f'{n}:not_in_the_{correct_bin}_bin' if assigned else f'{n}:unassigned')
        if not assigned:
            errors.append('marks_not_assigned')
        return dict(success=not errors, cubes=cubes, errors=errors, captured_by_bin=captured,
                    discarded_cube_ids=sorted(captured))

    def task_snapshot(self):
        captured = self.captured_by_bin()
        return dict(captured_by_bin=captured, discarded_cube_ids=sorted(captured),
                    capture_events={lid: len(capture.capture_events)
                                    for lid, capture in self._bin_captures.items()})

    # ---- protocol hooks (contract C1) --------------------------------------
    def task_progress(self):
        """Cubes welded into their correct bin / cube count (spec 1.4)."""
        wanted = self.task_spec.get('correct_bin', {})
        if not wanted:
            return 0.
        captured = self.captured_by_bin()
        right = sum(1 for name, lid in wanted.items() if captured.get(name) == lid)
        return float(right) / float(len(self.cube_names))

    def collateral_checks(self):
        """Spec 1.5: a cube in the wrong bin. Permanent - a capture weld is never released - so this check
        never turns good again and the base class never logs a ``repaired`` event for it."""
        wanted = self.task_spec.get('correct_bin', {})
        wrong = sorted(name for name, lid in self.captured_by_bin().items() if wanted.get(name) != lid)
        return {'wrong_bin': dict(bad=bool(wrong), object=wrong[0] if wrong else '', cost=float(len(wrong)))}

    # ---- teleport certificate (contract C3) --------------------------------
    def tick(self, n=1):
        action = np.zeros(self.action_dim)
        for _ in range(int(n)):
            self.step(action)

    def drop_into_bin(self, name, lid, ticks=TELEPORT_DROP_TICKS):
        """Drop one cube through a bin's mouth by a state edit, then settle until the capture weld fires."""
        bin_spec = self.task_spec['bins'][lid]
        mouth = np.asarray(bin_spec['opening_center_world'], float) + np.array([0., 0., TELEPORT_DROP_M])
        joint = self.objects[name].joints[0]
        quat = yaw_quat_wxyz(float(self.work['yaw']))
        self.sim.data.set_joint_qpos(joint, np.array([*mouth, *quat], float))
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()
        for _ in range(int(ticks)):
            if self.captured_by_bin().get(name) == lid:
                break
            self.tick(1)
        return self.captured_by_bin().get(name)

    def free_joint_name(self, body_name):
        """The free joint of a body, whatever built it (procedural cover, native object), or None."""
        m = self.sim.model
        bid = m.body_name2id(body_name)
        for index in range(int(m.body_jntnum[bid])):
            jid = int(m.body_jntadr[bid]) + index
            if int(m.jnt_type[jid]) == 0:            # mjJNT_FREE
                return m.joint_id2name(jid)
        return None

    def move_covers_aside(self):
        """Take every cloche and inverted bowl off its cube and set it down on a free spot of the same work
        surface (contract C3: the certificate passes through the hiding mechanism).

        The spot is drawn inside the work top, clear of every cube, both bins, Submit, the face occluders and
        the covers already moved, so a cover never leaves the surface, never lands on a task object and never
        falls into a bin. The wide cup is left where it stands: its cube is lifted straight out of it.
        """
        steps = []
        record = self.task_spec.get('observability') or {}
        covers = [c for c in record.get('covers', ()) if c.get('built') and c.get('kind') != 'wide_cup']
        if not covers:
            return steps
        rng = np.random.default_rng([int((getattr(self, 'instance', None) or {}).get('seed', 0)), TELEPORT_SALT])
        surface = OB.work_surface_rect(self)
        keepouts = list(self.hiding_keepouts())
        keepouts += [S.shape_at(CUBE_HALF_XY, cube['xy'], cube['yaw']) for cube in self.spec['cubes']
                     if cube['name'] in set(H.hidden_objects(self.spec))]
        for cover in covers:
            name, radius = cover['name'], float(cover.get('radius') or BOWL_RADIUS)
            body = self.objects[name].root_body if name in self.objects else name
            joint = self.free_joint_name(body)
            spot = H.cover_aside_spot(rng, surface, keepouts, radius, gap=COVER_ASIDE_GAP_M)
            if joint is None or spot is None:
                steps.append(dict(step=f'move the {cover["kind"]} off {cover["object"]}', ok=False,
                                  detail='no free spot on the work surface' if joint else f'{name} has no free joint'))
                continue
            bid = self.sim.model.body_name2id(body)
            pose = np.array(self.sim.data.body_xpos[bid], float)
            quat = np.array(self.sim.data.body_xquat[bid], float)
            world = self.frame_to_world([spot[0], spot[1], 0.])
            self.sim.data.set_joint_qpos(joint, np.array([world[0], world[1], pose[2] + .002, *quat], float))
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
            self.sim.forward()
            self.tick(4)
            moved = float(np.linalg.norm(np.array(self.sim.data.body_xpos[bid], float)[:2]
                                         - np.array(cover['xy_world'], float)))
            keepouts.append(S.Disc((float(spot[0]), float(spot[1])), radius))
            steps.append(dict(step=f'move the {cover["kind"]} off {cover["object"]}',
                              ok=bool(moved > radius + M.HALF_SIZE_M),
                              detail=dict(to_frame_xy=[round(spot[0], 4), round(spot[1], 4)],
                                          moved_m=round(moved, 4), radius_m=round(radius, 4))))
        return steps

    def teleport_solution(self):
        """Bring the scene to a solved state without the robot: take the covers off, then drop every cube
        through the mouth of the bin its class belongs to, one at a time, and let the capture weld fire on
        its own (contract C3). An annexed or occluded cube is dropped straight from where it stands."""
        steps = []
        try:
            wanted = dict(self.task_spec.get('correct_bin', {}))
            if not wanted:
                return [dict(step='marks assigned', ok=False, detail='reset has not realised the classes yet')]
            steps.extend(self.move_covers_aside())
            for name in self.cube_names:
                lid = wanted[name]
                landed = self.drop_into_bin(name, lid)
                steps.append(dict(step=f'drop {name} into the {lid} bin', ok=bool(landed == lid),
                                  detail=dict(captured_by=landed,
                                              contained={k: bool(v) for k, v in
                                                         self._arrangement_snapshot()['bin_containment'][name].items()})))
            self.tick(TELEPORT_FINAL_TICKS)
            score = self._current_score()
            steps.append(dict(step='success predicate', ok=bool(score['success']),
                              detail=dict(errors=score['errors'], progress=round(self.task_progress(), 4),
                                          captured_by_bin=score['captured_by_bin'])))
        except Exception as error:                              # never raises (contract C3)
            steps.append(dict(step='teleport_solution', ok=False, detail=repr(error)))
        return steps

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['roboquest'].update(mark_assignment=deepcopy(self.task_spec.get('mark_assignment')),
                                 initial_hidden_faces=deepcopy(self.task_spec.get('initial_hidden_faces')),
                                 hidden_state_check=deepcopy(self.task_spec.get('hidden_state_check')),
                                 observability=self.task_spec.get('observability_level', 'visible'),
                                 hidden_cubes=deepcopy(self.hidden_cube_modes()))
        return meta
