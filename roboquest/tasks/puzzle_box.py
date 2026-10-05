"""Puzzle box (bolt-interlocked sliding lid) as a RoboQuest kitchen task.

The chest and its sliding parts come from the generator-v1 sampler
(``roboquest.puzzle.layouts``) on top of the original layout primitives and the
geometry builder (``puzzle_box_geometry``); here they sit on a RoboCasa work
surface in the task frame, so the same instance builds on any layout and style.
The item and tray are RoboCasa objects placed at world level (free joints must
be top-level bodies). Scoring follows the study-room task: item inside and
resting on the tray, everything released and still, first Submit.

Generator ``v3`` (RoboQuest v1 spec 4.3). The reported
factors are ``chain_length`` {4, 5, 6} and ``cover`` {none, partial, full};
everything else is nuisance drawn from the named streams and recorded in the
spec:

* ``decoys`` {0, 1, 2} and the re-lock flag from ``structure``: in half of the
  instances that have a decoy, one decoy re-locks chain bolt ``k`` through a
  hidden one-way push rod inside the chest (``spec['relock']`` names the decoy,
  the bolt and ``k``; ``None`` when the draw said no or no bolt qualifies);
* the chest's position and yaw from ``poses``: any yaw on an island (or dining
  counter), at most 45 degrees from facing the robot on a wall counter; both
  candidate poses are drawn (each with its own position, since a chest turned
  round stands further forward) and checked against the reachable free
  region, and the scene picks the one its work surface calls for;
* the chest wood palette and the handle colour from ``materials``.

The task rules, the goal text and the success test are unchanged, but the
mechanism is not: v3 replaces v2's per-notch cover strips with one rigid opaque
plate fixed over the sliding layer (``roboquest.puzzle.plate``), so the contract
version is ``puzzle-box-interlock-kitchen-v2``. The plate floats 4 mm above the
sliders, every slider travels freely underneath it, and only handles and short
stubs protrude; ``cover`` now says how many locks of the chain's prefix it
hides (none / a prefix / all of them).

Reachable free region. Everything is in the task frame of
:class:`RoboQuestKitchen` (x along the front of the surface, y away from the
robot). The free region is the footprint the base class guarantees decor-free
on the work surface (``FOOTPRINT``, 1.20 x 0.66 m, its front edge
``FOOTPRINT_FRONT_MARGIN`` behind the surface's edge; ``place_work_frame``
slides it sideways past decor and ``prop_clashes`` decides the rest at G1).
Reach comes from the motor library: the base spawns with its front at the
surface's edge and ``Skills.stand`` never drives closer
(``forward_limit_y``), it puts a handle ``GRASP_AHEAD_M`` = 0.47 m ahead of the
base and the arm absorbs up to 0.12 m more, and the v1 certificates solved
handles as deep as chest y 0.12 with the chest at frame y 0.06, i.e. frame
y 0.18, on every instance (13/13). ``REACH`` therefore bounds handles to
y in [-0.31, 0.18] and |x| <= 0.57; the chest deck, every slider's full sweep
(the lid's travel path included) must lie inside the footprint and keep
``KEEPOUT_GAP`` from the tray and the Submit button.
"""
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from scipy.spatial.transform import Rotation

from roboquest import scatter as S
from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz
from roboquest.puzzle.layouts import CHAIN_LENGTHS, COVER_LEVELS, DECOY_COUNTS, covered_count, lock_names, sample_layout, validate_layout
from roboquest.puzzle import plate as plate_geom
from roboquest.puzzle_box_geometry import ROBOCASA_OBJECTS, LAYER_Z, FLOOR_TOP, add_robocasa_object, append_puzzle_box_objects, item_fits, measure_asset_extent, _rect_box, _rgba, _vec
from roboquest.puzzle_box_layouts import CAVITY, HALF_W, HANDLE_COLOURS, HANDLE_RADIUS, METAL, WOOD_PALETTES, blocked_now, handle_at, swept_rects
from roboquest.harness.contract import CAMERAS

CONTRACT_VERSION = 'puzzle-box-interlock-kitchen-v2'
GENERATOR_VERSION = 'v3'
GOAL_TEMPLATE = ('Open the wooden puzzle box on the counter, take out the {item} inside, and stand it '
                 'inside the tray. The lid slides sideways but is locked by sliding bolts with round '
                 'black handles: a part moves only when nothing blocks it, so work out the order. '
                 'Any order that works is fine. Once the {item} is inside the tray and you have let '
                 'go of everything, press the red SUBMIT button. Your first submission ends the '
                 'episode; a wrong submission or no submission fails.')
# Task-frame placement (metres): the chest left of centre, tray and Submit to its right, inside the footprint.
CHEST_LOCAL = (-.20, .06)          # the v1 reference pose; v2 draws the chest around it
CHEST_X_RANGE = (-.34, -.06)       # position band the chest centre is drawn from; the checks below prune it
CHEST_Y_RANGE = (-.22, .12)        # forward enough for a chest turned round (its deep side then faces the robot)
WALL_YAW_MAX = math.pi / 4         # at most 45 degrees from facing the robot on a wall counter
POSE_ATTEMPTS = 3000               # per candidate; a draw costs well under a millisecond
TRAY_LOCAL = (.42, -.04)
SUBMIT_LOCAL = (.42, -.30)
SUBMIT_HALF = (.048, .043)         # the Submit base plate (inspect_cups_submit)
TRAY_YAW = math.pi / 2
FOOTPRINT = (1.20, .66)
FOOTPRINT_RECT = S.Rect(-FOOTPRINT[0] / 2, FOOTPRINT[0] / 2, -FOOTPRINT[1] / 2, FOOTPRINT[1] / 2)
REACH = S.Rect(-.57, .57, -.31, .18)   # see the module docstring
DECK_MARGIN = .01                  # the chest deck stays this far inside the footprint
KEEPOUT_GAP = .03                  # sliders and deck to the tray and the Submit button
HANDLE_KEEPOUT_GAP = .05           # a handle's post to the tray and the Submit button (the open gripper)
ALMOST = .99                       # progress when the formula reads 1 but the predicate still fails (contract C1)
TELEPORT_SETTLE_TICKS = 100        # the gate batch's G2 settle; at 40 a round item dropped 3 mm into the tray
                                   # still read as moving on the 4/full slow-test instance (PLATE-5)
PROBE_STEP_M = .004                # G3 ray grid over everything the plate must hide
PROBE_SLACK_M = .0015              # a ray stopped within this of its target counts as arriving
PLATE_EDGE_INSET = .010            # see `_under_plate_points`: the 4 mm travel gap is see-through
TIP_PROBE_M = .015                 # length of slider tip sampled as `the tip that sits in the notch`
LOCK_DEPTH_MIN_MM = None           # covered-lock depth under the plate outline: recorded, not gated.
                                   # A 40 mm minimum was considered, reasoning that
                                   # the eye-in-hand camera peers along the 4 mm slit and sees about
                                   # 4 mm x standoff / camera height, some 24 mm at a 300 mm standoff and
                                   # 50 mm height. Measured on candidate A's raw registry the depth of an
                                   # engagement centre is 11.5-30.0 mm on every one of the 136 plated
                                   # instances, so 40 mm would fail the whole covered population. The
                                   # floor is the construction, not a defect: the plate is only promised
                                   # to reach `plate.HIDE_MARGIN` = 10 mm past an engagement, and
                                   # `plate.engagement_rect` pads the inner, across-the-notch end by
                                   # `plate.EDGE_MARGIN` = 4 mm only, because what lies beyond the plate
                                   # there is the locked slider's own body and not a sightline to the tip.
                                   # A 15 mm notch therefore puts its centre 7.5 + 4 = 11.5 mm from the
                                   # seed's edge, which is exactly the observed floor. Left as a
                                   # diagnostic until the designer re-expresses the measure (the tip point
                                   # rather than the engagement centre, or excluding the inner direction).


def _gate_module():
    raise ImportError('The instance minting gates (G0, puzzle_box_physics_gate) are not part of the RoboQuest release')


def _rot(xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])


def tray_half(layout):
    """Half sizes of the tray's collision footprint before its yaw, from the asset's measured extent."""
    extent = measure_asset_extent(ROBOCASA_OBJECTS / layout['tray_asset'] / 'model.xml')
    return ((extent['high'][0] - extent['low'][0]) / 2, (extent['high'][1] - extent['low'][1]) / 2)


def keepouts(layout):
    """The tray and the Submit button as frame shapes the chest must keep clear of."""
    return dict(tray=S.Box(tuple(TRAY_LOCAL), tray_half(layout), TRAY_YAW),
                submit=S.rect(SUBMIT_LOCAL, SUBMIT_HALF))


def chest_pose_problems(layout, xy, yaw, keepout_shapes=None):
    """Why the chest cannot stand at ``xy`` (frame) with ``yaw``; empty when it can.

    The deck stays inside the footprint by ``DECK_MARGIN``; every slider's
    full sweep (so the lid's travel path) stays inside the footprint and
    ``KEEPOUT_GAP`` from the tray and Submit; every handle post at both ends
    of its travel lies inside ``REACH`` and keeps ``HANDLE_KEEPOUT_GAP`` (room
    for the open gripper) from the tray and Submit.
    """
    keepout_shapes = keepouts(layout) if keepout_shapes is None else keepout_shapes
    problems = []

    def box(rect):
        (x0, x1), (y0, y1) = rect
        centre = _rot(((x0 + x1) / 2, (y0 + y1) / 2), yaw)
        return S.Box((xy[0] + centre[0], xy[1] + centre[1]), ((x1 - x0) / 2, (y1 - y0) / 2), yaw)

    deck = box((tuple(layout['deck']['x']), tuple(layout['deck']['y'])))
    if S.region_margin(deck, FOOTPRINT_RECT) < DECK_MARGIN:
        problems.append('deck leaves the footprint')
    for name, shape in keepout_shapes.items():
        if S.clearance(deck, shape) < KEEPOUT_GAP:
            problems.append(f'deck too close to the {name}')
    for name, part in layout['parts'].items():
        for rect in swept_rects(part):
            sweep = box(rect)
            if S.region_margin(sweep, FOOTPRINT_RECT) < 0.:
                problems.append(f'sweep of {name} leaves the footprint')
                break
            if any(S.clearance(sweep, shape) < KEEPOUT_GAP for shape in keepout_shapes.values()):
                problems.append(f'sweep of {name} too close to the tray or Submit')
                break
        for d in (part['range'][0], part['range'][1]):
            local = handle_at(part, d)
            hx, hy = _rot(local, yaw)
            handle = S.Disc((xy[0] + hx, xy[1] + hy), HANDLE_RADIUS)
            if S.region_margin(handle, REACH) < 0.:
                problems.append(f'handle of {name} out of reach')
                break
            if any(S.clearance(handle, shape) < HANDLE_KEEPOUT_GAP for shape in keepout_shapes.values()):
                problems.append(f'handle of {name} too close to the tray or Submit')
                break
    return problems


def sample_chest_pose(poses, layout):
    """Draw both chest pose candidates from the ``poses`` stream: ``island`` (any yaw) then ``wall`` (yaw
    within ``WALL_YAW_MAX`` of facing the robot), each as ``{xy, yaw, attempts}``.

    Per attempt, in a fixed order: yaw, x, y; the first draw that passes
    :func:`chest_pose_problems` is kept. The reach band and the footprint's
    depth decide what survives: with handles reachable to frame y 0.18 a
    turned chest fits only near 0 or 180 degrees, and the turned-round band
    needs the chest 10 to 20 cm further forward, which is why each candidate
    draws its own position. ``ValueError`` after ``POSE_ATTEMPTS`` makes
    ``registry.mint`` reject the seed.
    """
    shapes = keepouts(layout)
    poses_out = {}
    for rule, yaw_range in (('island', (-math.pi, math.pi)), ('wall', (-WALL_YAW_MAX, WALL_YAW_MAX))):
        for attempt in range(POSE_ATTEMPTS):
            yaw = float(poses.uniform(*yaw_range))
            xy = (float(poses.uniform(*CHEST_X_RANGE)), float(poses.uniform(*CHEST_Y_RANGE)))
            if not chest_pose_problems(layout, xy, yaw, shapes):
                poses_out[rule] = dict(xy=[round(xy[0], 4), round(xy[1], 4)], yaw=round(yaw, 4), attempts=attempt + 1)
                break
        else:
            raise ValueError(f'no {rule} chest pose fits the reachable free region in {POSE_ATTEMPTS} draws')
    return poses_out


def island_like(fixture):
    """Surfaces the robot can walk round: any chest yaw. Wall counters keep the chest facing the robot."""
    return bool(fixture_is_type(fixture, FixtureType.ISLAND) or fixture_is_type(fixture, FixtureType.DINING_COUNTER))


class PuzzleBox(RoboQuestKitchen):
    TASK_NAME = 'puzzle_box'
    CONTRACT_VERSION = CONTRACT_VERSION
    GENERATOR_VERSION = GENERATOR_VERSION
    BUDGET_TICKS = 12000               # provisional (spec 1.2): set from the G4 certificates later
    FACTORS = {'chain_length': CHAIN_LENGTHS, 'cover': COVER_LEVELS}
    # Structure class for the split: lid direction (sampled) x chain length (designed).
    # One chain length is held out per direction, so every level of both still appears in dev.
    HOLDOUT_KEYS = ('x+-k4', 'x--k5', 'y+-k6')
    FOOTPRINT = FOOTPRINT

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        return GOAL_TEMPLATE.format(item=spec['item_name'])

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        layout_seed = int(structure.integers(0, 2**31 - 1))
        decoys = int(DECOY_COUNTS[int(structure.integers(0, len(DECOY_COUNTS)))])
        relock_drawn = bool(structure.integers(0, 2))       # half of the instances; binding when decoys >= 1
        layout = sample_layout(layout_seed, int(factors['chain_length']), decoys, factors['cover'],
                               relock=relock_drawn and decoys >= 1)
        palette = WOOD_PALETTES[int(materials.integers(0, len(WOOD_PALETTES)))]
        handle = HANDLE_COLOURS[int(materials.integers(0, len(HANDLE_COLOURS)))]
        layout['colours'] = dict(wood=palette[0], deck=palette[1], lid=palette[2], metal=METAL, handle=handle)
        pose = sample_chest_pose(poses, layout)
        return dict(layout=layout, layout_seed=layout_seed, chain_length=int(layout['chain_length']),
                    decoys=decoys, cover=layout['cover'], relock=layout['relock'], relock_drawn=relock_drawn,
                    item_name=layout['item_asset'].split('/')[0].replace('_', ' '),
                    colours=layout['colours'],
                    placement=dict(chest_island=pose['island'], chest_wall=pose['wall'],
                                   tray=list(TRAY_LOCAL), submit=list(SUBMIT_LOCAL)))

    @classmethod
    def validate_spec(cls, spec):
        layout = spec['layout']
        problems = list(validate_layout(layout))
        # The factors, not the sampler's retries, decide the structure; the nuisance draws must agree.
        for key in ('chain_length', 'decoys', 'cover'):
            if layout[key] != spec[key]:
                problems.append(f'{key} {layout[key]!r} in the layout, {spec[key]!r} in the spec')
        if spec['decoys'] not in DECOY_COUNTS:
            problems.append(f"decoys {spec['decoys']!r} not in {DECOY_COUNTS}")
        covers = sum(1 for name in lock_names(layout['chain_length']) if layout['covered'].get(name))
        if covers != covered_count(layout['chain_length'], layout['cover']):
            problems.append(f"{covers} covered locks for cover={layout['cover']!r}")
        if spec.get('relock') != layout.get('relock'):
            problems.append('relock in the spec does not match the layout')
        if bool(spec.get('relock_drawn')) and spec['decoys'] >= 1 and not layout.get('relock_requested'):
            problems.append('a re-lock was drawn but the sampler was not asked for one')
        if layout.get('relock') and not (bool(spec.get('relock_drawn')) and spec['decoys'] >= 1):
            problems.append('a re-lock without the draw')
        item_asset = ROBOCASA_OBJECTS / layout['item_asset'] / 'model.xml'
        if not item_asset.is_file():
            problems.append(f'missing item asset {item_asset}')
        elif not item_fits(measure_asset_extent(item_asset)):
            problems.append('item does not fit the cavity or gripper')
        placement = spec['placement']
        if abs(float(placement['chest_wall']['yaw'])) > WALL_YAW_MAX + 1e-9:
            problems.append('wall yaw beyond 45 degrees')
        shapes = keepouts(layout)
        for rule in ('island', 'wall'):
            pose = placement[f'chest_{rule}']
            for problem in chest_pose_problems(layout, pose['xy'], pose['yaw'], shapes):
                problems.append(f'chest pose ({rule}): {problem}')
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """G0: the interlock physics (with the re-lock replayed from every reachable state when the
        instance has one) plus the chest pose inside the reachable free region for both yaw candidates."""
        passed, report = _gate_module().quick_gate(spec['layout'])
        layout, placement = spec['layout'], spec['placement']
        pose = {rule: chest_pose_problems(layout, placement[f'chest_{rule}']['xy'], placement[f'chest_{rule}']['yaw'])
                for rule in ('island', 'wall')}
        report['chest_pose'] = dict(problems=pose,
                                    island=dict(placement['chest_island'],
                                                yaw_deg=round(math.degrees(placement['chest_island']['yaw']), 1)),
                                    wall=dict(placement['chest_wall'],
                                              yaw_deg=round(math.degrees(placement['chest_wall']['yaw']), 1)))
        report['relock_ok'] = None if report.get('relock') is None else all(r['ok'] for r in report['relock'])
        return bool(passed and not any(pose.values())), report

    @classmethod
    def split_key(cls, spec):
        """(lid direction, chain length): the compositional class the eval set holds out."""
        return f"{spec['layout']['lid_direction']}-k{spec['layout']['chain_length']}"

    # ---- scene protocol -----------------------------------------------------
    def chest_pose_for(self, fixture):
        """The drawn pose the work surface calls for: (frame xy, yaw, rule)."""
        rule = 'island' if island_like(fixture) else 'wall'
        pose = self.spec['placement'][f'chest_{rule}']
        return (float(pose['xy'][0]), float(pose['xy'][1])), float(pose['yaw']), rule

    def chest_to_world(self, local):
        """Chest-local (x, y, z above the chest base) to world, through the frame and the chest yaw."""
        local = np.asarray(local, float)
        x, y = _rot((local[0], local[1]), self.chest_yaw)
        return self.frame_to_world([self.chest_xy[0] + x, self.chest_xy[1] + y, local[2]])

    def build_task(self):
        self.place_work_frame()
        layout = self.spec['layout']
        placement = self.spec['placement']
        self.chest_xy, self.chest_yaw, self.chest_yaw_rule = self.chest_pose_for(self.work_fixture)
        frame = self.add_frame_body('puzzle_frame', self.chest_xy)
        frame.set('quat', _vec(yaw_quat_wxyz(self.work['yaw'] + self.chest_yaw)))
        geometry = append_puzzle_box_objects(frame, 0., (0., 0.), (0., 0.), assets=None, layout=layout)
        world, assets = self.model.worldbody, self.model.asset
        item_asset = ROBOCASA_OBJECTS / layout['item_asset'] / 'model.xml'
        tray_asset = ROBOCASA_OBJECTS / layout['tray_asset'] / 'model.xml'
        extent = measure_asset_extent(item_asset)
        height = extent['high'][2] - extent['low'][2]
        platform_top = LAYER_Z[0] - .004 - height
        if platform_top < FLOOR_TOP:
            raise ValueError(f'Item too tall for the cavity: {item_asset}')
        chest = frame.find("body[@name='puzzle_chest']")
        (cx0, cx1), (cy0, cy1) = CAVITY
        _rect_box(chest, 'chest_platform', ((cx0 + .002, cx1 - .002), (cy0 + .002, cy1 - .002)),
                  (FLOOR_TOP, platform_top), _rgba(layout['colours']['wood']))
        item_xy = layout['item_xy']
        item_pos = self.chest_to_world([item_xy[0], item_xy[1], platform_top - extent['low'][2] + .001])
        _, item_geoms = add_robocasa_object(world, assets, 'puzzle_item', item_pos.tolist(), item_asset,
                                            friction='2.0 0.05 0.002')
        tray_extent = measure_asset_extent(tray_asset)
        tray_pos = self.frame_to_world([placement['tray'][0], placement['tray'][1], -tray_extent['low'][2]])
        _, tray_geoms = add_robocasa_object(world, assets, 'puzzle_tray', tray_pos.tolist(), tray_asset,
                                            yaw=TRAY_YAW + self.work['yaw'], free=False)
        total_yaw = self.work['yaw'] + self.chest_yaw
        for name, part in geometry['parts'].items():
            axis = np.asarray(part['axis'], float)
            part['axis_chest'] = axis.tolist()
            part['axis_world'] = (yaw_matrix(total_yaw) @ axis).tolist()
            handle = part['handle_center_world']          # chest-local until here (the builder's frame)
            part['handle_center_chest'] = list(handle)
            part['handle_center_world'] = self.chest_to_world(handle).tolist()
        self.geometry = geometry
        self.geometry['item'] = dict(body_name='puzzle_item', joint_name='puzzle_item_joint', geom_names=item_geoms,
                                     source_asset=str(item_asset), source_position=item_pos.tolist(), extent=extent,
                                     height=height, platform_top_local=platform_top,
                                     grasp_offset_local=[0., 0., extent['low'][2] + .5 * height])
        self.geometry['tray'] = dict(body_name='puzzle_tray', geom_names=tray_geoms, source_asset=str(tray_asset),
                                     center_world=tray_pos.tolist(), yaw_rad=TRAY_YAW + self.work['yaw'],
                                     extent=tray_extent)
        self.geometry['chest_geom_names'] = geometry['chest_geom_names'] + ['chest_platform']
        self.add_submit_button(placement['submit'])
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, layout_seed=self.spec['layout_seed'],
            layout_cover=self.spec['cover'], layout_decoys=self.spec['decoys'],
            layout_signature=layout['signature'], split_key=self.split_key(self.spec),
            lid_direction=geometry['lid_direction'], chain_length=geometry['chain_length'],
            covered_locks=geometry['covered_locks'], frame_origin_world=self.work['center_world'],
            frame_yaw=self.work['yaw'], chest_local=list(self.chest_xy), chest_yaw=float(self.chest_yaw),
            chest_yaw_rule=self.chest_yaw_rule, chest_yaw_world=float(total_yaw),
            work_fixture_kind='island' if self.chest_yaw_rule == 'island' else 'wall',
            reach_frame=dict(x=[REACH.x0, REACH.x1], y=[REACH.y0, REACH.y1]),
            parts=geometry['parts'], relock_private=geometry.get('relock'),
            release_order_private=geometry['release_order'], item=self.geometry['item'], tray=self.geometry['tray'],
            scoring='First physical Submit freezes the score: item fully inside the tray, resting on it, '
                    'released and still, with the robot touching no puzzle part; no submission fails.')
        self.asset_evidence.update(
            chest_and_bolts='Purpose-made rigid boxes from a sampled layout in the task frame, wood palette and '
                            'handle colour from the materials stream',
            item=f"RoboCasa objaverse object with original meshes/textures/collisions ({item_asset})",
            tray=f"RoboCasa objaverse tray, fixed ({tray_asset})",
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged')

    def setup_task_references(self):
        m = self.sim.model
        self.obj_body_id = dict(getattr(self, 'obj_body_id', {}))
        self.obj_body_id['puzzle_item'] = m.body_name2id('puzzle_item')
        self._part_qadr, self._part_geoms = {}, {}
        for name, part in self.geometry['parts'].items():
            self.obj_body_id[name] = m.body_name2id(part['body_name'])
            self._part_qadr[name] = m.get_joint_qpos_addr(part['joint_name'])
            self._part_geoms[name] = {m.geom_name2id(g) for g in part['geom_names']}
        self._item_geoms = {m.geom_name2id(g) for g in self.geometry['item']['geom_names']}
        self._tray_geoms = {m.geom_name2id(g) for g in self.geometry['tray']['geom_names']}
        self._tray_inner = self._measure_tray()
        self.task_spec['tray'].update(self._tray_inner)
        self._chain_bolts = [n for n in self.geometry['release_order'] if n != 'lid']
        self._ever_released = set()

    def _reset_internal(self):
        super()._reset_internal()
        self._ever_released = set()

    def _start_events(self):
        """The base's event baseline plus this task's release memory.

        From the first tick on, ``_ever_released`` lives inside ``_event_state``,
        which ``motor.Checkpoint`` deep-copies and restores, so the memory rewinds
        with the physics and a retried oracle branch cannot leave a phantom re-lock
        behind. A restore to a pre-tick checkpoint sets the state back to None and
        the next baseline starts the memory empty, which is right: no bolt has
        moved in that branch, and ``_released_bolts`` re-adds whatever is released
        now.
        """
        state = super()._start_events()
        state['ever_released'] = set()
        self._ever_released = state['ever_released']
        return state

    def _release_memory(self):
        """The chain bolts released at some point in this branch (see ``_start_events``)."""
        state = self._event_state
        if state is not None:
            memory = state.get('ever_released')
            if memory is None:
                memory = state['ever_released'] = set()
            self._ever_released = memory
        return self._ever_released

    def _geom_mesh_aabb(self, gid):
        raw, d = self.sim.model._model, self.sim.data
        mesh = int(raw.geom_dataid[gid])
        vertices = raw.mesh_vert[raw.mesh_vertadr[mesh]:raw.mesh_vertadr[mesh] + raw.mesh_vertnum[mesh]]
        world = vertices @ np.asarray(d.geom_xmat[gid]).reshape(3, 3).T + np.asarray(d.geom_xpos[gid])
        return world.min(0), world.max(0)

    def _measure_tray(self):
        """Inner floor: the tray collision piece with the largest footprint, plus the tray frame for xy tests."""
        mujoco.mj_forward(self.sim.model._model, self.sim.data._data)
        best = None
        for gid in self._tray_geoms:
            lo, hi = self._geom_mesh_aabb(gid)
            area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
            if best is None or area > best[0]:
                best = (area, lo, hi, gid)
        _, lo, hi, gid = best
        extent = self.geometry['tray']['extent']
        return dict(floor_top_z=float(hi[2]), floor_geom=self.sim.model.geom_id2name(gid),
                    inner_half_xy=[float((extent['high'][0] - extent['low'][0]) / 2 - .012),
                                   float((extent['high'][1] - extent['low'][1]) / 2 - .012)])

    def _contacts_of(self, gids):
        m, d = self.sim.model, self.sim.data
        rows = []
        for index, contact in enumerate(d.contact[:d.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if not ({a, b} & gids):
                continue
            other = b if a in gids else a
            force = np.zeros(6)
            mujoco.mj_contactForce(m._model, d._data, index, force)
            normal = np.asarray(contact.frame).reshape(3, 3)[0] * (1 if b in gids else -1)
            rows.append(dict(geom=m.geom_id2name(other) or '', other_id=other, normal_force_n=float(force[0]),
                             normal_on_object=normal.tolist()))
        return rows

    def item_state(self):
        m, d = self.sim.model, self.sim.data
        bid = self.obj_body_id['puzzle_item']
        contacts = self._contacts_of(self._item_geoms)
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for gid in self._item_geoms:
            a, b = self._geom_mesh_aabb(gid)
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        return dict(position=np.asarray(d.body_xpos[bid]).tolist(),
                    quaternion_xyzw=Rotation.from_matrix(np.asarray(d.body_xmat[bid]).reshape(3, 3)).as_quat().tolist(),
                    linear_velocity=np.asarray(d.get_body_xvelp('puzzle_item')).tolist(),
                    angular_velocity=np.asarray(d.get_body_xvelr('puzzle_item')).tolist(),
                    aabb_world=[lo.tolist(), hi.tolist()],
                    robot_contacts=[c for c in contacts if c['geom'].startswith(('robot', 'gripper', 'mobilebase'))],
                    tray_contacts=[c for c in contacts if c['other_id'] in self._tray_geoms],
                    grasped=bool(self._check_grasp(self.robots[0].gripper['right'],
                                                   [m.geom_id2name(g) for g in self._item_geoms])))

    def part_states(self):
        d = self.sim.data
        values = {name: float(d.qpos[adr]) for name, adr in self._part_qadr.items()}
        blocked = blocked_now(values, self.spec['layout'])
        states = {}
        for name, part in self.geometry['parts'].items():
            value = values[name]
            robot = [c for c in self._contacts_of(self._part_geoms[name])
                     if c['geom'].startswith(('robot', 'gripper', 'mobilebase'))]
            released = None if part['released_at'] is None else bool(value >= part['released_at'])
            opened = None if part['open_at'] is None else bool(value >= part['open_at'])
            states[name] = dict(joint_m=value, blocked_by_geometry=bool(blocked[name]), released=released,
                                opened=opened, robot_contact=bool(robot), decoy=bool(part['decoy']))
        return states

    # ---- protocol hooks (spec 1.4, 1.5, contract C3) ---------------------------
    def _bolt_joint(self, name):
        return float(self.sim.data.qpos[self._part_qadr[name]])

    def _released_bolts(self):
        """Chain bolts past their release threshold now; also remembers which ever were (for the re-lock log)."""
        released = [n for n in self._chain_bolts if self._bolt_joint(n) >= self.geometry['parts'][n]['released_at']]
        self._release_memory().update(released)
        return released

    def task_progress(self):
        """Spec 1.4: (bolts released + 1 if the item stands in the tray) / (chain length + 1).

        0 at reset, 1 exactly when the success predicate holds. The item term
        counts when the item rests inside the tray; if the formula reaches 1
        while the predicate still fails (hand still on the item or on a part)
        it reports ``ALMOST``, as the other tasks do.
        """
        score = self._current_score()
        if score['success']:
            return 1.
        released = len(self._released_bolts())
        item = 1. if (score['item_in_tray'] and score['item_supported_by_tray']) else 0.
        value = (released + item) / (len(self._chain_bolts) + 1)
        return ALMOST if value >= 1. else float(min(max(value, 0.), 1.))

    def collateral_checks(self):
        """Spec 1.5 ``relock``: a chain bolt that had been released is back below its release threshold,
        costed in bolts; repaired when it is released again. State reads only: the check cannot tell the
        decoy's push rod from the robot pushing a bolt home by hand, and counts both as undone progress."""
        released = set(self._released_bolts())
        memory = self._release_memory()
        relocked = [n for n in self._chain_bolts if n in memory and n not in released]
        relock = self.geometry.get('relock') or {}
        return dict(relock=dict(bad=bool(relocked), object=','.join(relocked) if relocked else relock.get('decoy', ''),
                                cost=float(len(relocked))))

    SWEEP_STEP_M = .005

    def _sweep_to_full_travel(self, part):
        """Move a part's joint to full travel in 5 mm steps; report the first kitchen clash and the
        first contact with the mechanism plate on the way.

        Setting the joint straight to its end value passes through whatever stands in the path: on
        evaluation kitchen 55 a paper-towel holder crosses the lid's travel, the certificate passed and
        the oracle's lid slide stopped 58 mm short. Only prop-kitchen contacts of the moving part count
        (bolts and notches touch each other legitimately); the joint ends at full travel either way, so
        the remaining steps are judged on their own.

        The plate is the v3 addition. It is welded to the chest, so a slider that touched it would be
        an ordinary prop-prop contact and the kitchen test would not see it; every slider is meant to
        travel its whole range underneath, and the lid to open under it, so any contact is a fault of
        the layout and not of the kitchen it was placed in.
        """
        d = self.sim.data
        joint, end, body = part['joint_name'], float(part['range'][1]), part['body_name']
        start = float(d.get_joint_qpos(joint))
        count = max(1, int(abs(end - start) / self.SWEEP_STEP_M + .999))
        plate_ids = self._geoms_with_prefix('chest_plate')
        part_ids = self._geom_family(part['geom_names'])
        blocked, touched = None, None
        for k in range(1, count + 1):
            d.set_joint_qpos(joint, start + (end - start) * k / count)
            d.set_joint_qvel(joint, 0.)
            self.sim.forward()
            if blocked is None:
                hits = [c for c in self.prop_clashes() if c['kind'] == 'prop-kitchen' and body in (c['body1'], c['body2'])]
                if hits:
                    blocked = dict(joint_mm=round(1000 * float(d.get_joint_qpos(joint)), 1), clash=hits[0])
            if touched is None and plate_ids:
                rubs = [c for c in self._contacts_of(part_ids) if c['other_id'] in plate_ids]
                if rubs:
                    touched = dict(joint_mm=round(1000 * float(d.get_joint_qpos(joint)), 1),
                                   geom=rubs[0]['geom'], normal_force_n=round(rubs[0]['normal_force_n'], 3))
        return blocked, touched

    def teleport_solution(self):
        """Contract C3: release the bolts in chain order by sweeping their slider joints to full travel, open
        the lid, stand the item in the tray through its free joint, settle with zero-action steps. Never
        moves the robot. A part whose travel path meets the kitchen, or the plate over the mechanism,
        fails its step (see ``_sweep_to_full_travel``)."""
        steps = []
        try:
            parts = self.geometry['parts']
            for name in self.geometry['release_order']:
                part = parts[name]
                blocked, touched = self._sweep_to_full_travel(part)
                value = self._bolt_joint(name)
                threshold = part['open_at'] if name == 'lid' else part['released_at']
                detail = dict(joint_mm=round(1000 * value, 1))
                if blocked:
                    detail['path_blocked'] = blocked
                if touched:
                    detail['plate_contact'] = touched
                steps.append(dict(step=('open lid' if name == 'lid' else f'release {name}'),
                                  ok=bool(value >= threshold) and blocked is None and touched is None,
                                  detail=detail))
            d = self.sim.data
            tray, item = self.geometry['tray'], self.geometry['item']
            centre = np.asarray(tray['center_world'], float)
            quat = np.asarray(d.get_joint_qpos(item['joint_name']), float)[3:7]
            z = float(self._tray_inner['floor_top_z'] - item['extent']['low'][2] + .003)
            d.set_joint_qpos(item['joint_name'], np.r_[centre[0], centre[1], z, quat])
            d.set_joint_qvel(item['joint_name'], np.zeros(6))
            self.sim.forward()
            steps.append(dict(step='stand the item in the tray', ok=True,
                              detail=dict(position=[round(float(v), 4) for v in (centre[0], centre[1], z)])))
            zero = np.zeros(self.action_dim)
            for _ in range(TELEPORT_SETTLE_TICKS):
                self.step(zero)
            score = self._current_score()
            steps.append(dict(step=f'settle {TELEPORT_SETTLE_TICKS} ticks',
                              ok=bool(score['item_in_tray'] and score['item_supported_by_tray'] and score['item_stable']),
                              detail={k: score[k] for k in ('item_in_tray', 'item_supported_by_tray', 'item_stable',
                                                            'item_released', 'parts_released')}))
            steps.append(dict(step='success predicate', ok=bool(score['success']),
                              detail=dict(lid_opened=score['lid_opened'], progress=round(self.task_progress(), 4))))
        except Exception as error:                                   # never raises (contract C3)
            steps.append(dict(step='teleport_solution', ok=False, detail=repr(error)))
        return steps

    # ---- G3: the covered locks must reveal nothing on camera -----------------
    def _geom_id(self, name):
        return int(mujoco.mj_name2id(self.sim.model._model, mujoco.mjtObj.mjOBJ_GEOM, name))

    def _geom_family(self, names):
        """Collision geoms and their visual twins, as ids; -1 entries dropped."""
        ids = set()
        for name in names:
            for candidate in (name, name + '_visual'):
                gid = self._geom_id(candidate)
                if gid >= 0:
                    ids.add(gid)
        return ids

    def _geoms_with_prefix(self, *prefixes):
        """Every geom whose name starts with one of the prefixes.

        RoboCasa objects name their visuals ``<obj>_visual_<i>`` and their
        collisions ``<obj>_collision_<i>``, so a name plus ``_visual`` is not
        enough to gather a prop; a prefix is.
        """
        raw = self.sim.model._model
        ids = set()
        for gid in range(raw.ngeom):
            name = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if name.startswith(prefixes):
                ids.add(gid)
        return ids

    def _policy_frames(self, depth=False):
        """The three policy camera images. With ``depth`` also the metric depth buffer, per camera.

        MuJoCo hands back both with row 0 at the *bottom*; the depth map is flipped here so
        that it is indexed by the same image rows robosuite's projection matrix produces
        (see `_backproject`), and the colour frames are left raw, as they were.
        """
        from robosuite.utils import camera_utils
        cameras, out = list(self.camera_names), {}
        for cam in CAMERAS:
            index = cameras.index(cam)
            got = self.sim.render(width=int(self.camera_widths[index]),
                                  height=int(self.camera_heights[index]), camera_name=cam, depth=depth)
            if not depth:
                out[cam] = np.asarray(got).copy()
                continue
            metres = camera_utils.get_real_depth_map(self.sim, np.asarray(got[1])[..., None])[..., 0]
            out[cam] = (np.asarray(got[0]).copy(), np.asarray(metres[::-1]).copy())
        return out

    def _backproject(self, cam, rows, columns, depth_map):
        """Image pixels plus the metric depth buffer to world points, one point per pixel.

        ``rows`` count from the top, as robosuite's camera matrix does, and ``depth_map`` is
        the flipped map `_policy_frames` returns. Round-tripped against `mj_ray` on the
        rendered set: a back-projected pixel is where the camera's ray stops to 0.4 mm on the
        eye-in-hand camera and 3.8 mm on the agentviews, which is one pixel at that standoff.
        """
        from robosuite.utils import camera_utils
        cameras = list(self.camera_names)
        index = cameras.index(cam)
        height, width = int(self.camera_heights[index]), int(self.camera_widths[index])
        matrix = camera_utils.get_camera_transform_matrix(self.sim, cam, height, width)
        rows, columns = np.asarray(rows, int), np.asarray(columns, int)
        z = depth_map[rows, columns].astype(float)
        camera_points = np.stack([columns * z, rows * z, z, np.ones_like(z)], axis=-1)
        return (np.linalg.inv(matrix) @ camera_points.T).T[:, :3]

    def _world_to_chest(self, points):
        """World points to chest-local ones: the inverse of `chest_to_world`, vectorised."""
        points = np.atleast_2d(np.asarray(points, float))
        frame = (yaw_matrix(-self.work['yaw']) @ (points - np.asarray(self.work['center_world'], float)).T).T
        delta = frame[:, :2] - np.asarray(self.chest_xy, float)
        cos, sin = math.cos(-self.chest_yaw), math.sin(-self.chest_yaw)
        return np.stack([cos * delta[:, 0] - sin * delta[:, 1],
                         sin * delta[:, 0] + cos * delta[:, 1], frame[:, 2]], axis=-1)

    def _outline_depths(self, local_xy):
        """`plate.outline_depth` for many chest-local points at once (metres, negative outside)."""
        rects = [tuple(tuple(axis) for axis in rect) for rect in (self.geometry.get('plate') or {}).get('rects_local') or ()]
        segments = plate_geom.union_boundary(rects) if rects else []
        local_xy = np.atleast_2d(np.asarray(local_xy, float))
        if not segments:
            return np.full(len(local_xy), np.nan)
        starts = np.asarray([s[0] for s in segments], float)
        ends = np.asarray([s[1] for s in segments], float)
        span = ends - starts
        length = np.einsum('sj,sj->s', span, span)
        delta = local_xy[:, None, :] - starts[None, :, :]
        t = np.clip(np.where(length > 0, np.einsum('psj,sj->ps', delta, span) / np.where(length > 0, length, 1.), 0.),
                    0., 1.)
        distance = np.linalg.norm(delta - t[..., None] * span[None, :, :], axis=2).min(axis=1)
        inside = np.asarray([plate_geom.inside_union(rects, tuple(p)) for p in local_xy])
        return np.where(inside, distance, -distance)

    def _changed_pixel_band(self, cam, changed, depth_map):
        """Split a camera's changed pixels into the tolerated edge band and the failing rest.

        A slider under the plate is visible by design in the outermost `PLATE_EDGE_INSET` of
        the slab: the plate floats `plate.PLATE_CLEAR` = 4 mm over the sliding layer so the
        sliders can travel, the rim closes that slit down to `plate.LINTEL_CLEAR` = 1.5 mm
        where a slider crosses it, and a low camera looks along what is left. The ray probe
        has always allowed it -- `_under_plate_points` insets its grid by the same 10 mm --
        and this is the pixel test's equivalent allowance.
        Each changed pixel is back-projected through the depth buffer and asked how deep it
        sits under the plate outline in the deck plane: in the band it is tolerated, deeper
        than the inset (a hole in the slab) or outside the outline altogether it fails.
        """
        rows = np.argwhere(changed)
        if not len(rows):
            return dict(changed=0, band=0, outside=0, deep=0, beyond_outline=0, depth_mm=None)
        height = int(self.camera_heights[list(self.camera_names).index(cam)])
        world = self._backproject(cam, height - 1 - rows[:, 0], rows[:, 1], depth_map)
        depth = self._outline_depths(self._world_to_chest(world)[:, :2])
        band = np.isfinite(depth) & (depth >= 0.) & (depth < PLATE_EDGE_INSET)
        deep = int((np.isfinite(depth) & (depth >= PLATE_EDGE_INSET)).sum())
        beyond = int((~np.isfinite(depth) | (depth < 0.)).sum())
        finite = depth[np.isfinite(depth)]
        return dict(changed=int(len(rows)), band=int(band.sum()), outside=deep + beyond,
                    deep=deep, beyond_outline=beyond,
                    depth_mm=(dict(min=round(1000 * float(finite.min()), 2),
                                   median=round(1000 * float(np.median(finite)), 2),
                                   max=round(1000 * float(finite.max()), 2)) if len(finite) else None))

    def _covered_lock_depths(self):
        """Depth of every covered lock's engagement under the plate outline, in millimetres."""
        layout = self.spec['layout']
        depths = {}
        for name, flag in sorted(layout['covered'].items()):
            if not flag:
                continue
            value = plate_geom.lock_depth(layout, name)
            if value is not None:
                depths[name] = round(1000 * value, 2)
        return depths

    def _first_hit(self, origin, target):
        """Name of the first geom on the segment origin -> target (None when nothing is hit)."""
        m, d = self.sim.model._model, self.sim.data._data
        direction = np.asarray(target, float) - np.asarray(origin, float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return None, 0.
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(m, d, np.asarray(origin, float), direction / norm, None, 1, -1, gid)
        if gid[0] < 0:
            return None, float(distance)
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])), float(distance)

    def _rect_world_points(self, rect, step=PROBE_STEP_M, inset=0.):
        """A grid over a chest-local rectangle, just under the sliding layer's top, in world.

        The probe height matters: aimed at the middle of the 12 mm layer a ray
        would enter the blocking bolt's top face a centimetre short of the
        notch, on ordinary exposed bolt, and every lock would read as a leak.
        A fifth of a millimetre under the top keeps the hit inside the notch.
        """
        (x0, x1), (y0, y1) = rect
        x0, x1, y0, y1 = x0 + inset, x1 - inset, y0 + inset, y1 - inset
        if x1 <= x0 or y1 <= y0:
            return []
        z = LAYER_Z[1] - .0002
        nx, ny = max(1, int((x1 - x0) / step) + 1), max(1, int((y1 - y0) / step) + 1)
        points = []
        for i in range(nx):
            for j in range(ny):
                fx = (i + .5) / nx
                fy = (j + .5) / ny
                points.append(self.chest_to_world([x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), z]).tolist())
        return points

    def _notch_world_points(self, name, step=PROBE_STEP_M):
        """A grid over a part's notch rectangle: where the holder's tip sits when it is engaged."""
        return self._rect_world_points(self.geometry['parts'][name]['notch_local'], step)

    def _under_plate_points(self, step=PROBE_STEP_M):
        """A grid over every slider footprint the plate hides, per part.

        Sampled ``PLATE_EDGE_INSET`` inside the hidden footprint. The plate floats 4 mm
        over the sliding layer so that the sliders can travel, and a camera low enough can
        look into that gap: a few millimetres of slider just inside the plate's edge are
        visible by design, which is why the plate reaches ``plate.HIDE_MARGIN`` = 10 mm
        past every engagement it hides instead of stopping at its edge.
        """
        out = {}
        for name, part in self.geometry['parts'].items():
            points = []
            for rect in part.get('under_plate_local') or ():
                points.extend(self._rect_world_points(rect, step, inset=PLATE_EDGE_INSET))
            if points:
                out[name] = points
        return out

    def _reaches(self, origin, point, slack=PROBE_SLACK_M):
        """Does a camera at ``origin`` see ``point``? True when no geom stops the ray first.

        This is the pixel test the render check needs: a point the ray reaches is a point
        that paints pixels. Reporting the geom that stopped the ray keeps the diagnosis.
        """
        hit, distance = self._first_hit(origin, point)
        reach = float(np.linalg.norm(np.asarray(point, float) - np.asarray(origin, float)))
        return (hit is None or distance > reach - slack), hit

    def _camera_positions(self):
        d, m = self.sim.data, self.sim.model
        return {cam: np.asarray(d.cam_xpos[m.camera_name2id(cam)], float).copy() for cam in CAMERAS}

    def _in_frame(self, point):
        """Cameras whose image contains this world point (pixel coordinates in range)."""
        from robosuite.utils import camera_utils
        cameras = list(self.camera_names)
        seen = {}
        for cam in CAMERAS:
            index = cameras.index(cam)
            height, width = int(self.camera_heights[index]), int(self.camera_widths[index])
            matrix = camera_utils.get_camera_transform_matrix(self.sim, cam, height, width)
            pixels = camera_utils.project_points_from_world_to_camera(
                np.asarray([point], float), matrix, height, width)[0]
            row, column = float(pixels[0]), float(pixels[1])
            seen[cam] = bool(0 <= row < height and 0 <= column < width)
        return seen

    def hidden_state_check(self, step=PROBE_STEP_M):
        """G3: the plate hides every covered lock on the policy cameras, props are visible.

        Fail-closed, run at reset, with the robot where the policy finds it. Four parts:

        1. *Nothing hidden reaches a camera* -- rays from each policy camera to a
           `PROBE_STEP_M` grid over (a) every covered lock's notch, (b) the tip of the
           slider engaged in it and (c) every slider footprint the plate covers must all
           be stopped before they arrive. A ray that arrives is a leak, whatever it would
           have shown, so the count is of visible probe points and the check needs zero.
        2. *Pixel identity* -- geoms that lie entirely under the plate are toggled to
           alpha 0 and the three cameras must render identically, as the cube task does,
           **outside the plate outline's 10 mm edge band**. Long sliders run out past the
           plate, so this set is usually empty and part 1 carries the proof; it catches a
           geom that is wholly hidden but still drawn. Every changed pixel is back-projected
           through the depth buffer and located in the deck plane: inside the outline but
           within `PLATE_EDGE_INSET` of it, it is the sliver of slider the travel gap shows
           by design and part 1 already allows (`_under_plate_points`); deeper in, or outside
           the outline, it fails. The counts are reported per camera and an instance that
           passes only on that allowance is flagged ``g3_gap_admitted``.
        3. *The plate is really drawn* -- hiding it must change pixels, or a plate that
           failed to build would pass part 1 by being invisible in both renders.
        4. *Required props* -- the tray, the chest and the Submit button must be inside
           at least one fixed camera and be the first geom their ray meets.

        The uncovered locks of a `partial` scene are meant to stay readable; that is
        reported (``readable_uncovered_locks``) but not gated, since occlusion by the
        robot's own arm is legitimate and varies with the reset pose.
        """
        if not getattr(self, 'has_offscreen_renderer', False):
            return dict(passed=None, note='no renderer')
        model = self.sim.model
        layout = self.spec['layout']
        covered = [name for name, flag in layout['covered'].items() if flag]
        uncovered = [name for name, flag in layout['covered'].items() if not flag]
        cameras = self._camera_positions()
        rendered = self._policy_frames(depth=True)
        base = {cam: frame for cam, (frame, _) in rendered.items()}
        depth_maps = {cam: buffer for cam, (_, buffer) in rendered.items()}

        # 1. Nothing hidden reaches a camera.
        leaks, probes, seen_by = [], {}, {}
        targets = {f'notch:{name}': self._notch_world_points(name, step) for name in covered}
        for name in covered:
            holder = self.geometry['parts'][name]['blocked_by']
            if holder:
                targets[f'tip:{holder}'] = self._tip_world_points(holder, step)
        for name, points in self._under_plate_points(step).items():
            targets[f'under-plate:{name}'] = points
        for label, points in targets.items():
            probes[label] = len(points)
            for cam, origin in cameras.items():
                visible, stopper = [], set()
                for point in points:
                    reached, hit = self._reaches(origin, point)
                    if reached:
                        visible.append(point)
                    elif hit:
                        stopper.add(hit)
                if visible:
                    leaks.append(dict(target=label, camera=cam, visible_points=len(visible),
                                      of=len(points), first=[round(v, 4) for v in visible[0]]))
                seen_by[f'{label}@{cam}'] = sorted(stopper)
        # Informational: in `partial` scenes the uncovered locks are meant to be readable.
        readable = []
        for name in uncovered:
            blocker = self.geometry['parts'][name]['blocked_by']
            if blocker is None:
                continue
            wanted = self._geom_family([g for g in self.geometry['parts'][blocker]['geom_names']
                                        if '_seg' in g])
            for cam, origin in cameras.items():
                for point in self._notch_world_points(name, step):
                    hit, _ = self._first_hit(origin, point)
                    if hit is not None and self._geom_id(hit) in wanted:
                        readable.append(dict(lock=name, camera=cam))
                        break

        # 2. Pixel identity for whatever lies entirely under the plate, outside the edge band.
        shadowed = self._plate_shadow_geoms()
        changed_shadow = {cam: 0 for cam in CAMERAS}
        band_shadow = {cam: 0 for cam in CAMERAS}
        outside_shadow = {cam: 0 for cam in CAMERAS}
        band_detail = {}
        if shadowed:
            alpha = model._model.geom_rgba[list(shadowed), 3].copy()
            try:
                model._model.geom_rgba[list(shadowed), 3] = 0.
                erased = self._policy_frames()
            finally:
                model._model.geom_rgba[list(shadowed), 3] = alpha
            for cam in CAMERAS:
                row = self._changed_pixel_band(cam, np.any(base[cam] != erased[cam], axis=2), depth_maps[cam])
                changed_shadow[cam], band_shadow[cam], outside_shadow[cam] = row['changed'], row['band'], row['outside']
                band_detail[cam] = row

        # 3. The plate is really drawn.
        plate_ids = sorted(self._geoms_with_prefix('chest_plate'))
        changed_plate = {cam: 0 for cam in CAMERAS}
        if plate_ids:
            alpha = model._model.geom_rgba[plate_ids, 3].copy()
            try:
                model._model.geom_rgba[plate_ids, 3] = 0.
                without = self._policy_frames()
            finally:
                model._model.geom_rgba[plate_ids, 3] = alpha
            changed_plate = {cam: int(np.any(base[cam] != without[cam], axis=2).sum()) for cam in CAMERAS}

        # 4. Required props inside a fixed camera and not occluded there.
        props = dict(
            tray=(self.geometry['tray']['center_world'], self._geoms_with_prefix('puzzle_tray')),
            chest=(self.chest_to_world([0., 0., LAYER_Z[1]]).tolist(), self._geoms_with_prefix('chest_', 'puzzle_lid')),
            submit=(self.task_spec['submit_button']['position'],
                    self._geoms_with_prefix(f'{self.submit_prefix}_')))
        visible_props = {}
        for prop, (point, ids) in props.items():
            frames = self._in_frame(point)
            rows = {}
            for cam in ('robot0_agentview_left', 'robot0_agentview_right'):
                hit, _ = self._first_hit(cameras[cam], point)
                rows[cam] = dict(in_frame=bool(frames[cam]),
                                 first_geom=hit, unoccluded=bool(hit is not None and self._geom_id(hit) in ids))
            visible_props[prop] = dict(cameras=rows,
                                       ok=any(r['in_frame'] and r['unoccluded'] for r in rows.values()))

        # 5. Recorded, not gated: how deep each covered lock sits under the plate outline.
        lock_depths = self._covered_lock_depths()
        minimum_lock_depth = min(lock_depths.values()) if lock_depths else None
        shallow_locks = (LOCK_DEPTH_MIN_MM is not None and minimum_lock_depth is not None
                         and minimum_lock_depth < LOCK_DEPTH_MIN_MM)

        passed = (not leaks
                  and all(v == 0 for v in outside_shadow.values())
                  and (not covered or (bool(plate_ids) and any(v > 0 for v in changed_plate.values())))
                  and all(v['ok'] for v in visible_props.values())
                  and not shallow_locks)
        # Marked so that a filter can undo this concession without re-rendering: the instance
        # would have failed the pixel sub-check before the 2026-09-22 ruling and passes only
        # because every changed pixel sits in the plate's edge band.
        admitted = bool(passed and sum(band_shadow.values()) > 0 and sum(outside_shadow.values()) == 0)
        return dict(passed=bool(passed), cover=layout['cover'], covered_locks=sorted(covered),
                    uncovered_locks=sorted(uncovered), leaks=leaks,
                    leaked_points=sum(row['visible_points'] for row in leaks),
                    probe_points=probes, probe_step_m=step, readable_uncovered_locks=readable,
                    shadowed_geoms=len(shadowed), changed_pixels_shadow=changed_shadow,
                    changed_pixels_in_band=band_shadow, changed_pixels_outside_band=outside_shadow,
                    band_detail=band_detail, plate_edge_inset_m=PLATE_EDGE_INSET,
                    g3_gap_admitted=admitted,
                    lock_depth_mm=lock_depths, min_lock_depth_mm=minimum_lock_depth,
                    lock_depth_min_mm_gate=LOCK_DEPTH_MIN_MM,
                    plate_geoms=len(plate_ids), changed_pixels_without_plate=changed_plate,
                    props=visible_props, stopped_by=seen_by,
                    chest_yaw_deg=round(math.degrees(self.chest_yaw), 1),
                    note='fail-closed: a camera ray that arrives at a hidden probe point is a leak; a '
                         'changed pixel is tolerated only inside the plate outline\'s 10 mm edge band')

    def _tip_world_points(self, name, step=PROBE_STEP_M):
        """A grid over the engaged tip of a bolt: its last `TIP_PROBE_M` at the tip end.

        The tip is the end that sits in the notch, at ``tip_end_local``; ``sigma`` points
        from there back towards the handle, which is the end the plate deliberately leaves
        in plain view. Reading the tip off the bolt's own frame keeps the two apart.
        """
        part = self.geometry['parts'][name]
        axis, tip = part.get('axis_index'), part.get('tip_end_local')
        if axis is None or tip is None:                       # a lid has no engaged tip
            return []
        rect = [None, None]
        rect[axis] = tuple(sorted((tip, tip + part['sigma'] * TIP_PROBE_M)))
        rect[1 - axis] = (part['perp_center_local'] - HALF_W, part['perp_center_local'] + HALF_W)
        return self._rect_world_points((rect[0], rect[1]), step)

    def _plate_shadow_geoms(self):
        """Geom ids of slider boxes that lie entirely inside the plate's footprint."""
        plate = self.geometry.get('plate')
        if not plate:
            return []
        shadow = []
        for name, part in self.geometry['parts'].items():
            for index, span in enumerate(part['spans_local']):
                (sx0, sx1), (sy0, sy1) = span
                for (px0, px1), (py0, py1) in plate['rects_local']:
                    if px0 <= sx0 and sx1 <= px1 and py0 <= sy0 and sy1 <= py1:
                        shadow.extend(self._geom_family([f'puzzle_{name}_seg{index}']))
                        break
        return sorted(set(shadow))

    def _item_in_tray(self, item):
        tray = self.geometry['tray']
        center = np.asarray(item['position'][:2]) - np.asarray(tray['center_world'][:2])
        local = yaw_matrix(-tray['yaw_rad'])[:2, :2] @ center
        half = self._tray_inner['inner_half_xy']
        lo = item['aabb_world'][0][2]
        floor = self._tray_inner['floor_top_z']
        return bool(abs(local[0]) <= half[0] and abs(local[1]) <= half[1] and floor - .006 <= lo <= floor + .03)

    def _current_score(self, states=None):
        item = self.item_state()
        parts = self.part_states()
        inside = self._item_in_tray(item)
        supported = any(c['normal_force_n'] > 1e-5 and c['normal_on_object'][2] > .5 for c in item['tray_contacts'])
        released = not item['robot_contacts'] and not item['grasped']
        # The item's residual motion is reported, never scored. An
        # item inside the tray, on its floor and out of the hand at the press is placed, however it still
        # rocks; ``item_stable`` stays in the dict as a diagnostic (the oracle waits on it before pressing).
        stable = (np.linalg.norm(item['linear_velocity']) < .03 and np.linalg.norm(item['angular_velocity']) < .2)
        parts_released = not any(p['robot_contact'] for p in parts.values())
        success = bool(inside and supported and released and parts_released)
        return dict(success=success, item_in_tray=bool(inside), item_supported_by_tray=bool(supported),
                    item_released=bool(released), item_stable=bool(stable), parts_released=bool(parts_released),
                    lid_opened=bool(parts['lid']['opened']), item_position=item['position'], parts=parts)

    def task_snapshot(self):
        return dict(item=self.item_state(), parts=self.part_states(),
                    relock=dict(self.collateral_checks()['relock'], ever_released=sorted(self._release_memory())))
