"""Wobbly stand as a RoboQuest kitchen task (tasks-v0 task 10).

A four-legged wooden stand on the work surface has one short corner leg or two
short adjacent legs, shorter by ``delta_mm``; the top tilts by 1.3-5.5 deg, too
little to see, and a ball placed at the top's centre rolls to the low rim within
about a second. Shim plates of 5/10/15 mm lie nearby. The policy must put the
matching plate(s) under the short leg(s) so the ball stays still, then press
Submit. The prop geometry and the resting-pose model live in ``roboquest.stand``;
the scene here places them in the task frame and scores the episode.

The reported size axis is one ordered ladder of how much is unknown (spec 2.1):
``one_leg_one_shim`` (which leg) < ``one_leg_all_shims`` (leg and thickness) <
``two_legs_all_shims`` (two legs and their thicknesses). ``delta_mm`` (5, 10, 15)
stays a balanced factor but is not reported.

Hidden state: which legs are short and by how much. It is revealed by the
ball's roll, by the gap under the foot that hangs (with one short corner leg
the stand tips onto it and the diagonally *opposite* long foot hangs, so the
gap is not under the short leg), or by lifting the stand.

Every scene also gets one or two native helper containers -- a bowl or a rimmed
tray, drawn from a verified vocabulary -- standing somewhere on the work surface at
random. The goal never mentions them; they are a safe place to park the
ball while a leg is lifted and a shim goes in, and finding that out is the policy's own
job. They are placed out of the ball's roll band (``roboquest.stand.helpers``), so the
ball cannot get into one by accident, and a ball resting in one -- or held in the
gripper -- is not collateral.

Scoring, frozen at the first physical Submit press: top tilt <= 0.4 deg, ball
at rest on the top, stand, shims and ball released and settled, all of it for
the last 2 s (40 ticks). Score is 0 at reset.
"""
from copy import deepcopy
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz
from roboquest.stand import helpers as H
from roboquest.stand import geometry as G
from roboquest.stand import statics as S

CONTRACT_VERSION = 'wobbly-stand-kitchen-v2'
GOAL = 'The stand is not level. Use the shims to level it so the ball stays still on top, then press Submit.'
BUDGET_TICKS = 12000          # provisional (protocol C1): no v0 HORIZON and no oracle certificate yet

# v1 size factor (spec 2.1): one ordered ladder over the v0 `axes` x
# `shim_set` pair, named for what the policy sees. `axes` keeps its v0 meaning --
# the number of *tilt axes*, so 'two' is the single short corner leg and 'one' the
# short adjacent pair -- and never appears in a spec, a key or a factor again.
#   one_leg_one_shim   one short leg, only the matching plate   (unknown: which leg)
#   one_leg_all_shims  one short leg, all three thicknesses     (unknowns: leg, thickness)
#   two_legs_all_shims two short legs, all three thicknesses    (unknowns: two legs, thicknesses)
SIZE_LEVELS = ('one_leg_one_shim', 'one_leg_all_shims', 'two_legs_all_shims')
SIZE_MAP = {'one_leg_one_shim': dict(axes='two', shim_set='exact', short_leg_count=1, plate_count=1),
            'one_leg_all_shims': dict(axes='two', shim_set='mixed', short_leg_count=1, plate_count=3),
            'two_legs_all_shims': dict(axes='one', shim_set='mixed', short_leg_count=2, plate_count=4)}
FACTORS = {'size': SIZE_LEVELS, 'delta_mm': (5, 10, 15)}
# (size, delta) cells, the v0 pattern re-derived: one held-out cell per size level at
# a different delta, so every size level has a held-out class and every delta level
# still appears in development. The first two are v0's ('two-d5' and 'one-d10')
# carried over under their new names; the third level needs the remaining delta.
HOLDOUT_KEYS = ('one_leg_one_shim-d5', 'one_leg_all_shims-d15', 'two_legs_all_shims-d10')
# Spec 3: evaluation placements keep the rolling props this far back from the
# counter's front edge, so a floor drop reflects carelessness, not geometry.
FRONT_EDGE_CLEARANCE = .10
# Progress (spec 1.4) is the fraction of the initial tilt removed; contract C1 makes
# 1.0 mean the success predicate holds, so a level stand still inside the 2 s window
# (or with the ball off the top) reports this instead.
ALMOST = .99
# The simulated resting tilt differs from the rigid model's by a couple of parts in
# ten thousand (measured: 2e-4 of the tilt at reset, i.e. under 0.001 deg), so a dead
# band this wide makes "0 at reset" exact without hiding any real levelling.
PROGRESS_DEADBAND = .01

# Task-frame layout (m): stand left of centre, shims in a row behind it to the right, a free parking area at the
# front right where the oracle sets the stand aside, Submit at the front right corner.
STAND_LOCAL = (-.22, .02)
STAND_JITTER = .02
STAND_YAW_DEG = 15.
SHIM_SLOTS_X = (.02, .13, .24, .35)
SHIM_ROW_Y = .17
SHIM_JITTER = .010
SHIM_YAW_DEG = 25.
PARK_LOCAL = (.18, -.10)
SUBMIT_LOCAL = (.40, -.17)
SUBMIT_RADIUS = .05
FOOTPRINT = (1.00, .56)
PLACE_LIFT = .0001
# The frame sits FOOTPRINT_FRONT_MARGIN behind the front of the chosen top region.
FRONT_EDGE_Y = -FOOTPRINT[1] / 2 - RoboQuestKitchen.FOOTPRINT_FRONT_MARGIN
# The parking bowl, a native RoboCasa bowl placed at random on the surface.
HELPER_PREFIX = 'helper'      # helper_0, helper_1: the parking containers
# A ball this far above the work top while the gripper touches it is being carried, not dropped.
HELD_LIFT_M = .05

# Scoring thresholds.
LEVEL_TILT_DEG = .4
BALL_STILL_SPEED = .02
WINDOW_TICKS = 40
STAND_STILL = (.01, .05)      # m/s, rad/s
SHIM_STILL = .01
ROBOT_PREFIXES = ('robot', 'gripper', 'mobilebase')
# Physics gate thresholds (spec: unshimmed tilt within 10 %, ball at the rim within 3 s, exact shims <= 0.2 deg
# with the ball within 3 cm of the centre for 2 s, wrong shims > 0.6 deg).
GATE_TILT_TOL = .10
GATE_RIM_SECONDS = 3.
GATE_LEVEL_DEG = .2
GATE_WRONG_DEG = .6
GATE_BALL_RADIUS = .03
GATE_PENETRATION_M = .0002

_SIDE_SETS = {frozenset(pair) for pair in G.SIDES.values()}


def _quat_wxyz(rotation):
    q = Rotation.from_matrix(rotation).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def stand_extent(yaw_deg):
    """Half width of the stand's top, in the task frame, at a given yaw."""
    yaw = math.radians(float(yaw_deg))
    return G.TOP_SIZE / 2 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))


def front_clearance(stand):
    """Metres between the stand's front-most extent and the work surface's front edge.

    The frame sits ``FOOTPRINT_FRONT_MARGIN`` behind the front of the chosen top
    region, so the region's front edge is at task-frame ``y = -FOOTPRINT[1]/2 -
    FOOTPRINT_FRONT_MARGIN``.
    """
    return float(stand['xy'][1] - stand_extent(stand['yaw_deg']) - FRONT_EDGE_Y)


def helper_keepouts(spec):
    """Task-frame discs a helper must clear by ``helpers.HELPER_MARGIN_M``: the shims and Submit.

    The stand is not in the list: the roll band around it is wider than the stand itself. The
    helpers already placed are appended to this list as they are drawn.
    """
    out = []
    for shim in spec['shims']:
        yaw = math.radians(shim['yaw_deg'])
        out.append((shim['name'], list(shim['xy']),
                    G.SHIM_SIZE / 2 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))))
    out.append(('submit', list(spec['placement']['submit']), SUBMIT_RADIUS))
    return out


def helper_names(spec):
    """The task-object name of each helper, in spec order."""
    return [f'{HELPER_PREFIX}_{index}' for index in range(len(spec.get('helpers', [])))]


def helper_problems(spec):
    """Everything wrong with ``spec['helpers']``; empty means the placement is legal."""
    helpers = spec.get('helpers')
    if not isinstance(helpers, list) or not 1 <= len(helpers) <= max(H.HELPER_COUNTS):
        return [f'helpers must be a list of 1 to {max(H.HELPER_COUNTS)} containers']
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
            stand_xy=spec['stand']['xy'], keepouts=keepouts, footprint=FOOTPRINT,
            front_edge_y=FRONT_EDGE_Y)
        keepouts.append((label, xy, entry['radius']))
    return problems


class WobblyStand(RoboQuestKitchen):
    TASK_NAME = 'wobbly_stand'
    CONTRACT_VERSION = CONTRACT_VERSION
    BUDGET_TICKS = BUDGET_TICKS
    # v1: `size` replaces the v0 `axes` x `shim_set` pair with one ordered ladder,
    # and the placement keeps the stand back from the counter's front edge.
    # v2: every scene gets one or two helper containers, new objects in every instance.
    GENERATOR_VERSION = 'v2'
    FACTORS = FACTORS
    HOLDOUT_KEYS = HOLDOUT_KEYS
    FOOTPRINT = FOOTPRINT
    SIZE_MAP = SIZE_MAP

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        return GOAL

    @classmethod
    def size_plan(cls, size):
        """The v0 ``axes``/``shim_set`` machinery a size level maps onto."""
        if size not in SIZE_MAP:
            raise ValueError(f'unknown size level {size!r}; choose from {SIZE_LEVELS}')
        return dict(SIZE_MAP[size])

    @classmethod
    def plates_for(cls, size, delta_mm):
        """Plate thicknesses in the scene: one matching plate per short leg, plus the two other thicknesses for 'mixed'."""
        plan = cls.size_plan(size)
        plates = [int(delta_mm)] * plan['short_leg_count']
        if plan['shim_set'] == 'mixed':
            plates += [t for t in G.SHIM_THICKNESSES_MM if t != int(delta_mm)]
        return sorted(plates)

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        size, delta_mm = str(factors['size']), int(factors['delta_mm'])
        plan = cls.size_plan(size)
        axes, shim_set = plan['axes'], plan['shim_set']
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        if axes == 'one':
            side = str(structure.choice(sorted(G.SIDES)))
            short_legs = list(G.SIDES[side])
        else:
            short_legs = [str(structure.choice(G.LEGS))]
        plates = cls.plates_for(size, delta_mm)
        uniform = lambda a, b: float(poses.uniform(a, b))
        stand = dict(xy=[STAND_LOCAL[0] + uniform(-STAND_JITTER, STAND_JITTER),
                         STAND_LOCAL[1] + uniform(-STAND_JITTER, STAND_JITTER)],
                     yaw_deg=uniform(-STAND_YAW_DEG, STAND_YAW_DEG))
        order = [int(i) for i in poses.permutation(len(plates))]
        slots = [int(i) for i in poses.permutation(len(SHIM_SLOTS_X))[:len(plates)]]
        shims = []
        for index, (plate_index, slot) in enumerate(zip(order, slots)):
            shims.append(dict(name=f'shim_{index}', thickness_mm=int(plates[plate_index]),
                              xy=[SHIM_SLOTS_X[slot] + uniform(-SHIM_JITTER, SHIM_JITTER),
                                  SHIM_ROW_Y + uniform(-SHIM_JITTER, SHIM_JITTER)],
                              yaw_deg=uniform(-SHIM_YAW_DEG, SHIM_YAW_DEG)))
        # The helper containers, drawn last from `poses` so that adding them leaves
        # every other draw of a given seed bit-identical, and uniformly over the frame (rejection
        # sampling) so they are placed at random rather than anywhere in particular.
        helpers, _stats = H.draw_helpers(
            poses, stand_xy=stand['xy'],
            keepouts=helper_keepouts(dict(shims=shims, placement=dict(submit=list(SUBMIT_LOCAL)))),
            footprint=FOOTPRINT, front_edge_y=FRONT_EDGE_Y)
        return dict(size=size, delta_mm=delta_mm, short_legs=short_legs,
                    plates_mm=[s['thickness_mm'] for s in shims], stand=stand, shims=shims,
                    helpers=helpers,
                    stand_front_clearance_m=front_clearance(stand),
                    ball=dict(colour=str(materials.choice(sorted(G.BALL_COLOURS)))),
                    materials=dict(stand_wood=str(materials.choice(G.STAND_WOODS)), shim_wood=G.SHIM_WOOD),
                    placement=dict(park=list(PARK_LOCAL), submit=list(SUBMIT_LOCAL)))

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        for name, levels in FACTORS.items():
            if spec.get(name) not in levels:
                problems.append(f'{name}={spec.get(name)!r} not in {levels}')
        if problems:
            return problems
        plan = cls.size_plan(spec['size'])
        short = spec.get('short_legs', [])
        if any(leg not in G.LEGS for leg in short):
            problems.append(f'unknown legs {short}')
        elif len(short) != plan['short_leg_count']:
            problems.append(f"size={spec['size']} needs {plan['short_leg_count']} short leg(s), got {short}")
        elif plan['short_leg_count'] == 2 and frozenset(short) not in _SIDE_SETS:
            problems.append(f"size={spec['size']} needs two *adjacent* short legs, got {short}")
        shims = spec.get('shims', [])
        expected = cls.plates_for(spec['size'], spec['delta_mm'])
        if sorted(s['thickness_mm'] for s in shims) != expected or spec.get('plates_mm') != [s['thickness_mm'] for s in shims]:
            problems.append(f'plates {[s["thickness_mm"] for s in shims]} do not match {expected}')
        if len({s['name'] for s in shims}) != len(shims):
            problems.append('shim names not unique')
        half_x, half_y = FOOTPRINT[0] / 2, FOOTPRINT[1] / 2
        stand = spec['stand']
        yaw = math.radians(stand['yaw_deg'])
        extent = G.TOP_SIZE / 2 * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
        if abs(stand['xy'][0]) + extent > half_x or abs(stand['xy'][1]) + extent > half_y:
            problems.append('stand outside the footprint')
        # Spec 3: the stand -- and with it the ball on its top, whose travel stays
        # inside the top -- keeps clear of the counter's front edge, so a ball on the
        # floor reflects carelessness rather than geometry.
        clearance = front_clearance(stand)
        if clearance < FRONT_EDGE_CLEARANCE - 1e-9:
            problems.append(f'stand {clearance * 100:.1f} cm from the front edge, under the '
                            f'{FRONT_EDGE_CLEARANCE * 100:.0f} cm rule')
        if abs(float(spec.get('stand_front_clearance_m', clearance)) - clearance) > 1e-6:
            problems.append('stand_front_clearance_m disagrees with the stand pose')
        boxes = [('stand', stand['xy'], extent)]
        for shim in shims:
            syaw = math.radians(shim['yaw_deg'])
            reach = G.SHIM_SIZE / 2 * (abs(math.cos(syaw)) + abs(math.sin(syaw)))
            if abs(shim['xy'][0]) + reach > half_x or abs(shim['xy'][1]) + reach > half_y:
                problems.append(f"{shim['name']} outside the footprint")
            boxes.append((shim['name'], shim['xy'], reach))
        sx, sy = spec['placement']['submit']
        boxes.append(('submit', [sx, sy], .05))
        for i, (a, pa, ra) in enumerate(boxes):
            for b, pb, rb in boxes[i + 1:]:
                if max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) < ra + rb + .02:
                    problems.append(f'{a} and {b} overlap')
        try:
            G.texture_path(spec['materials']['stand_wood'])
            G.texture_path(spec['materials']['shim_wood'])
        except (KeyError, ValueError) as error:
            problems.append(str(error))
        if spec.get('ball', {}).get('colour') not in G.BALL_COLOURS:
            problems.append('unknown ball colour')
        problems += helper_problems(spec)
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """G0 by geometry: the unshimmed stand tilts, the exact shims level it, every wrong shim does not."""
        problems = cls.validate_spec(spec)
        outcomes = S.shim_outcomes(spec['short_legs'], spec['delta_mm'], spec['plates_mm'])
        unshimmed = outcomes['unshimmed']
        hanging_mm = {leg: gap * 1000 for leg, gap in unshimmed['hanging'].items()}
        expected_hanging = 0 if cls.size_plan(spec['size'])['short_leg_count'] == 2 else 1
        wrong_min = outcomes['wrong_single_min_tilt_deg']
        checks = dict(
            spec_valid=not problems,
            unshimmed_tilts=1. <= unshimmed['tilt_deg'] <= 6.,
            unshimmed_not_marginal=not unshimmed['marginal'],
            hanging_foot_as_expected=(len(hanging_mm) == expected_hanging
                                      and all(abs(gap - spec['delta_mm']) <= .3 for gap in hanging_mm.values())),
            exact_shims_level=outcomes['exact']['tilt_deg'] <= S.LEVEL_TILT_DEG,
            wrong_thickness_not_level=wrong_min is None or wrong_min > S.WRONG_TILT_DEG,
            wrong_legs_not_level=outcomes['wrong_leg']['tilt_deg'] > S.WRONG_TILT_DEG,
            exact_is_a_level_assignment=any(row['under'] == outcomes['exact']['under']
                                            for row in outcomes['level_assignments']))
        report = dict(checks=checks, problems=problems, geometric_tilt_deg=unshimmed['tilt_deg'],
                      size=spec['size'], size_plan=cls.size_plan(spec['size']),
                      front_edge_clearance_m=front_clearance(spec['stand']),
                      support=unshimmed['support'], hanging_mm=hanging_mm, exact_tilt_deg=outcomes['exact']['tilt_deg'],
                      wrong_single=outcomes['wrong_single'], wrong_single_min_tilt_deg=wrong_min,
                      wrong_leg=outcomes['wrong_leg'], level_assignments=outcomes['level_assignments'])
        return all(checks.values()), report

    @classmethod
    def split_key(cls, spec):
        return f"{spec['size']}-d{spec['delta_mm']}"

    # ---- physics options ------------------------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        option = self.model.root.find('option')
        if option is None:
            import xml.etree.ElementTree as ET
            option = ET.SubElement(self.model.root, 'option')
        option.set('timestep', f'{G.PHYSICS_TIMESTEP:.9g}')
        option.set('noslip_iterations', str(G.NOSLIP_ITERATIONS))
        option.set('cone', G.CONTACT_CONE)

    def initialize_time(self, control_freq):
        super().initialize_time(control_freq)
        # 1 ms physics for the ball (as the cube and stamp scenes); each 20 Hz tick stays 50 ms.
        self.model_timestep = G.PHYSICS_TIMESTEP
        if getattr(self, 'sim', None) is not None:
            self.sim.model.opt.timestep = self.model_timestep

    # ---- scene protocol -----------------------------------------------------
    def _decor_boxes(self, top_z):
        """Everything standing on the work top counts as occupied, not only RoboCasa ``Accessory`` fixtures.

        Toasters, coffee machines and the like are their own fixture classes; a stand foot landing on one
        tilts the top by degrees (seen on layout 13), so they must push the frame sideways like the plant
        and the paper towel do.
        """
        boxes = []
        for name, fixture in self.fixtures.items():
            if fixture is self.work_fixture or not hasattr(fixture, 'get_bbox_points'):
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            if points.size == 0 or not (top_z - .05 <= points[:, 2].min() <= top_z + .15):
                continue
            boxes.append(dict(name=name, x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                              y0=float(points[:, 1].min()), y1=float(points[:, 1].max())))
        return boxes

    def place_work_frame(self):
        """As the base class, but a frame that still overlaps decor (no free offset) is a build failure (G1)."""
        work = super().place_work_frame()
        fixture = self.work_fixture
        box = self._footprint_world_box(work['center_local'], work['yaw'], fixture)
        hits = [d['name'] for d in self._decor_boxes(work['top_z']) if self._boxes_overlap(box, d)]
        if hits:
            raise ValueError(f"{self.TASK_NAME}: no decor-free {self.FOOTPRINT} m frame on {fixture.name} "
                             f"(layout {self.instance['layout_id']}); the frame would overlap {hits}")
        return work

    def _stand_world_pose(self, rotation_body, z0):
        """World position and rotation of the stand body for a stand-frame rest rotation and origin height z0."""
        stand = self.spec['stand']
        yaw = self.work['yaw'] + math.radians(stand['yaw_deg'])
        rotation = yaw_matrix(yaw) @ np.asarray(rotation_body, float)
        pos = self.frame_to_world([stand['xy'][0], stand['xy'][1], z0 + PLACE_LIFT])
        return pos, rotation

    def build_task(self):
        self.place_work_frame()
        spec = self.spec
        world, assets = self.model.worldbody, self.model.asset
        short, delta = spec['short_legs'], spec['delta_mm']
        rest = S.configuration_pose(short, delta)
        pos, rotation = self._stand_world_pose(rest['rotation'], rest['z0'])
        self.stand_geom = G.build_stand(world, assets, 'stand', pos, _quat_wxyz(rotation), short, delta,
                                        spec['materials']['stand_wood'])
        ball_pos = pos + rotation @ np.array([0., 0., G.BALL_RADIUS + PLACE_LIFT])
        self.ball_geom = G.build_ball(world, 'ball', ball_pos, G.BALL_COLOURS[spec['ball']['colour']])
        self.ball_pairs = G.add_ball_contact_pairs(self.model.root, self.ball_geom['geom_name'],
                                                   self.stand_geom['top_geoms'])
        self.shim_geoms = {}
        for shim in spec['shims']:
            spos = self.frame_to_world([shim['xy'][0], shim['xy'][1], shim['thickness_mm'] / 2000. + PLACE_LIFT])
            squat = yaw_quat_wxyz(self.work['yaw'] + math.radians(shim['yaw_deg']))
            self.shim_geoms[shim['name']] = G.build_shim(world, assets, shim['name'], spos, squat,
                                                         shim['thickness_mm'], spec['materials']['shim_wood'])
        self.add_submit_button(spec['placement']['submit'])
        self._add_helpers()
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, size=spec['size'],
            budget_ticks=int(BUDGET_TICKS), front_edge_clearance_m=front_clearance(spec['stand']),
            delta_mm_private=delta, short_legs_private=short,
            predicted_rest_private=dict(tilt_deg=rest['tilt_deg'], support=rest['support'],
                                        hanging_mm={l: g * 1000 for l, g in rest['gaps'].items() if g > 0}),
            frame_origin_world=self.work['center_world'], frame_yaw=self.work['yaw'],
            stand=dict(body_name='stand', joint_name='stand_joint', local_xy=stand_local(spec), yaw_deg=spec['stand']['yaw_deg'],
                       source_position=pos.tolist(), top_size=G.TOP_SIZE, height=G.STAND_HEIGHT, rim_height=G.RIM_HEIGHT,
                       leg_size=G.LEG_SIZE, foot_radius=G.FOOT_RADIUS, mass=self.stand_geom['mass'],
                       wood=spec['materials']['stand_wood'], grasp='top edge (18 mm plate plus rim)'),
            ball=dict(body_name='ball', joint_name='ball_joint', radius=G.BALL_RADIUS, mass=G.BALL_MASS,
                      colour=spec['ball']['colour'], source_position=ball_pos.tolist(),
                      contact_pairs=self.ball_pairs, rolling_friction=G.BALL_TOP_FRICTION[3]),
            shims={name: dict(body_name=name, joint_name=geom['joint_name'], thickness_mm=geom['thickness_mm'],
                              size=G.SHIM_SIZE, local_xy=next(s['xy'] for s in spec['shims'] if s['name'] == name))
                   for name, geom in self.shim_geoms.items()},
            park_local=spec['placement']['park'],
            helpers={name: dict(asset=h['asset'], kind=h['kind'], scale=h['scale'], local_xy=list(h['xy']),
                                yaw_deg=h['yaw'], role='clutter')
                     for name, h in zip(helper_names(spec), spec['helpers'])},
            helper_rule=dict(roll_band_m=H.ROLL_BAND_M, margin_m=H.HELPER_MARGIN_M, inset=H.HELPER_INSET,
                             held_lift_m=HELD_LIFT_M,
                             rule='A ball resting in a helper or held by the gripper is not collateral; '
                                  'a ball on the counter or the floor is.'),
            physics=dict(timestep_s=G.PHYSICS_TIMESTEP, noslip_iterations=G.NOSLIP_ITERATIONS, cone=G.CONTACT_CONE,
                         wood_solref=G.WOOD_SOLREF, wood_solimp=G.WOOD_SOLIMP, ball_solref=G.BALL_SOLREF,
                         ball_top_friction=list(G.BALL_TOP_FRICTION)),
            scoring=dict(level_tilt_deg=LEVEL_TILT_DEG, ball_still_speed=BALL_STILL_SPEED, window_ticks=WINDOW_TICKS,
                         rule='First physical Submit freezes the score: top tilt <= 0.4 deg, ball at rest on the '
                              'top, stand/shims/ball released and settled, all for the last 40 ticks (2 s); '
                              'no submission fails.'))
        self.asset_evidence.update(
            stand='Purpose-built rigid wooden stand: boxes plus spherical foot glides, one free body, RoboCasa '
                  f"wood texture {spec['materials']['stand_wood']}; only the leg lengths encode the hidden state",
            shims=f"Purpose-built free plates, RoboCasa wood texture {spec['materials']['shim_wood']}",
            ball='Plain free sphere with an explicit rolling-friction contact pair against the top',
            helpers='Native RoboCasa bowls and rimmed trays, unmodified, placed clear of the ball roll band; '
                    'the vocabulary is the list verified to hold the ball (roboquest.stand.helpers)',
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged; physics options: '
                 '1 ms timestep, elliptic cone, noslip 5 (task-level, see handoff)')

    def _add_helpers(self):
        """The parking containers, native RoboCasa assets with ``role='clutter'``.

        The mint drew them clear of the stand's roll band, the shims, Submit and each other inside
        the task frame; the kitchen is only known here, so the decor check happens now and a hit is
        a build failure (G1) that refuses the seed and lets the mint walk on. ``place_work_frame``
        already refuses a frame that overlaps decor, so this is the second lock on the same door.
        """
        spec, top_z = self.spec, self.work['top_z']
        decor = self._decor_boxes(top_z)
        self.helpers = {}
        for name, helper in zip(helper_names(spec), spec['helpers']):
            entry = H.entry_for(helper['asset'])
            world = self.frame_to_world([helper['xy'][0], helper['xy'][1], 0.])
            hits = H.decor_hits(world[:2], entry['radius'], decor)
            if hits:
                raise ValueError(f'{self.TASK_NAME}: {name} ({helper["asset"]}) would stand on {hits} '
                                 f"(layout {self.instance['layout_id']})")
            yaw = self.work['yaw'] + math.radians(float(helper['yaw']))
            obj = self.add_native(name, helper['asset'], [float(world[0]), float(world[1])], top_z,
                                  scale=float(helper['scale']),
                                  quat_xyzw=Rotation.from_euler('z', yaw).as_quat().tolist())
            half = [float(obj.size[0]) / 2, float(obj.size[1]) / 2]
            rim = float(obj.top_offset[2])
            if max(abs(half[0] - entry['half'][0]), abs(half[1] - entry['half'][1])) > .001:
                raise ValueError(f'{self.TASK_NAME}: {helper["asset"]} is {half} m half wide, not the '
                                 f'verified {list(entry["half"])} m; re-run the roll-in verification')
            self.helpers[name] = dict(entry=entry, half=half, rim_local_z=rim, yaw=yaw,
                                      world_xy=[float(world[0]), float(world[1])])
            self.task_spec['objects'][name].update(
                role='clutter', kind=entry['kind'], asset=helper['asset'], scale=float(helper['scale']),
                local_xy=list(helper['xy']), yaw_deg=float(helper['yaw']), half_xy_m=half,
                rim_local_z_m=rim, holds_ball=True,
                note='parking container; not named in the goal, not part of the success rule')
            self.asset_evidence[name].update(
                role='clutter', kind=entry['kind'],
                verified='holds the 30 mm ball: released at the centre above the rim with 5 cm/s of drift '
                         'in four directions, still inside and at rest after a 100-tick settle')

    def setup_task_references(self):
        m = self.sim.model
        self._stand_bid = m.body_name2id('stand')
        self._ball_bid = m.body_name2id('ball')
        self._shim_bids = {name: m.body_name2id(name) for name in self.shim_geoms}
        self._stand_qadr, self._stand_vadr = m.get_joint_qpos_addr('stand_joint'), m.get_joint_qvel_addr('stand_joint')
        self._ball_qadr, self._ball_vadr = m.get_joint_qpos_addr('ball_joint'), m.get_joint_qvel_addr('ball_joint')
        self._shim_qadr = {name: m.get_joint_qpos_addr(geom['joint_name']) for name, geom in self.shim_geoms.items()}
        self._shim_vadr = {name: m.get_joint_qvel_addr(geom['joint_name']) for name, geom in self.shim_geoms.items()}
        self._stand_gids = {m.geom_name2id(g) for g in self.stand_geom['geom_names']}
        self._top_gids = {m.geom_name2id(g) for g in self.stand_geom['top_geoms']}
        self._rim_gids = {m.geom_name2id(g) for g in self.stand_geom['top_geoms'] if 'rim' in g}
        self._foot_gids = {leg: m.geom_name2id(g) for leg, g in self.stand_geom['foot_geoms'].items()}
        self._ball_gid = m.geom_name2id(self.ball_geom['geom_name'])
        self._shim_gids = {name: m.geom_name2id(geom['geom_name']) for name, geom in self.shim_geoms.items()}
        self._prop_gids = self._stand_gids | {self._ball_gid} | set(self._shim_gids.values())
        # The helper containers. The inset footprint, the rim height and a live contact
        # are the "ball parked in this one" test (roboquest.stand.helpers).
        for name, ref in getattr(self, 'helpers', {}).items():
            bid = m.body_name2id(self.objects[name].root_body)
            ref['bid'] = bid
            ref['gids'] = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                           and (m.geom_contype[g] or m.geom_conaffinity[g])}
            ref['inset'] = [H.HELPER_INSET * ref['half'][0], H.HELPER_INSET * ref['half'][1]]
        self.obj_body_id = dict(getattr(self, 'obj_body_id', {}))
        self.obj_body_id.update(stand=self._stand_bid, ball=self._ball_bid, **self._shim_bids)
        self.obj_body_id.update({name: ref['bid'] for name, ref in getattr(self, 'helpers', {}).items()})
        self._level_ticks = 0
        self.task_spec['physics']['compiled'] = dict(timestep_s=float(m.opt.timestep),
                                                     noslip_iterations=int(m.opt.noslip_iterations),
                                                     cone=int(m.opt.cone))

    def _reset_internal(self):
        super()._reset_internal()
        self._level_ticks = 0
        self._state_cache = None
        # RoboCasa settles the scene for 0.5 s inside reset; put the ball back at the top's centre so the
        # policy sees the roll from its first observation.
        self._recentre_ball()
        self.sim.forward()

    # ---- state ---------------------------------------------------------------
    def _body_pose(self, bid):
        d = self.sim.data
        return np.asarray(d.body_xpos[bid], float).copy(), np.asarray(d.body_xmat[bid], float).reshape(3, 3).copy()

    def _body_speeds(self, name):
        d = self.sim.data
        return float(np.linalg.norm(d.get_body_xvelp(name))), float(np.linalg.norm(d.get_body_xvelr(name)))

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
                             normal_on_object=normal.tolist(), dist=float(contact.dist)))
        return rows

    def _robot_touching(self, contacts):
        return any(c['geom'].startswith(ROBOT_PREFIXES) for c in contacts)

    def _grasped(self, geom_names):
        return bool(self._check_grasp(self.robots[0].gripper['right'], list(geom_names)))

    def _set_free_joint(self, qadr, vadr, pos, quat_wxyz):
        d = self.sim.data
        d.qpos[qadr[0]:qadr[1]] = [*pos, *quat_wxyz]
        d.qvel[vadr[0]:vadr[1]] = 0.

    def _recentre_ball(self):
        pos, rotation = self._body_pose(self._stand_bid)
        ball = pos + rotation @ np.array([0., 0., G.BALL_RADIUS + PLACE_LIFT])
        self._set_free_joint(self._ball_qadr, self._ball_vadr, ball, [1., 0., 0., 0.])

    def stand_state(self):
        pos, rotation = self._body_pose(self._stand_bid)
        contacts = self._contacts_of(self._stand_gids)
        top_z = self.work['top_z']
        clearance = {}
        for leg, foot in self.stand_geom['feet_local'].items():
            clearance[leg] = float((pos + rotation @ np.asarray(foot))[2] - G.FOOT_RADIUS - top_z)
        foot_gids = set(self._foot_gids.values())
        foot_contacts = [c for c in contacts if c['other_id'] not in self._prop_gids or c['other_id'] in self._shim_gids.values()]
        penetration = -min((c['dist'] for c in self._contacts_of(foot_gids)), default=0.)
        lin, ang = self._body_speeds('stand')
        return dict(position=pos.tolist(), quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
                    tilt_deg=math.degrees(math.acos(min(1., max(-1., float(rotation[2, 2]))))),
                    linear_speed=lin, angular_speed=ang, robot_contact=self._robot_touching(contacts),
                    grasped=self._grasped(self.stand_geom['geom_names']), foot_clearance_m=clearance,
                    foot_penetration_m=max(0., float(penetration)),
                    support_geoms=sorted({c['geom'] for c in foot_contacts if not c['geom'].startswith(ROBOT_PREFIXES)}))

    def ball_in_helper(self, pos, contacts):
        """The helper the ball is resting in, or None (the rule in ``stand.helpers``).

        In the helper's frame: inside the inset footprint, below the rim, and touching one of its
        collision geoms. The inset is what tells a ball in the interior from a ball leaning against
        the outside wall.
        """
        for name, ref in getattr(self, 'helpers', {}).items():
            hpos, rotation = self._body_pose(ref['bid'])
            local = rotation.T @ (pos - hpos)
            if abs(local[0]) > ref['inset'][0] or abs(local[1]) > ref['inset'][1]:
                continue
            if local[2] > ref['rim_local_z'] + .001:
                continue
            if any(c['other_id'] in ref['gids'] for c in contacts):
                return name
        return None

    def ball_state(self):
        pos, _ = self._body_pose(self._ball_bid)
        spos, rotation = self._body_pose(self._stand_bid)
        local = rotation.T @ (pos - spos)
        contacts = self._contacts_of({self._ball_gid})
        travel = self.stand_geom['ball_travel']
        inside = bool(abs(local[0]) <= travel + .004 and abs(local[1]) <= travel + .004
                      and abs(local[2] - G.BALL_RADIUS) <= .006)
        supported = any(c['other_id'] in self._top_gids and c['normal_force_n'] > 1e-6 for c in contacts)
        at_rim = bool(inside and (abs(local[0]) >= travel - .002 or abs(local[1]) >= travel - .002
                                  or any(c['other_id'] in self._rim_gids for c in contacts)))
        lin, ang = self._body_speeds('ball')
        grasped = self._grasped([self.ball_geom['geom_name']])
        touched = self._robot_touching(contacts)
        return dict(position=pos.tolist(), local_xy=[float(local[0]), float(local[1])],
                    distance_from_centre=float(np.linalg.norm(local[:2])), speed=lin, angular_speed=ang,
                    # `within_top` is geometry alone: the ball is over the top, at resting height.
                    # `on_top` also needs a live top contact, which the success predicate wants but a
                    # collateral check must not (at reset the ball is set PLACE_LIFT clear of the top,
                    # and a ball in flight over its own stand has not left it).
                    within_top=bool(inside), on_top=bool(inside and supported), at_rim=at_rim,
                    robot_contact=touched, grasped=grasped,
                    # Neither of these is the ball "off the stand".
                    in_helper=self.ball_in_helper(pos, contacts),
                    held=bool(grasped or (touched and pos[2] - self.work['top_z'] > HELD_LIFT_M)))

    def shim_states(self):
        spos, srot = self._body_pose(self._stand_bid)
        feet = {leg: spos + srot @ np.asarray(foot) for leg, foot in self.stand_geom['feet_local'].items()}
        states = {}
        for name, geom in self.shim_geoms.items():
            pos, rotation = self._body_pose(self._shim_bids[name])
            contacts = self._contacts_of({self._shim_gids[name]})
            touching_feet = {leg for leg, gid in self._foot_gids.items() if any(c['other_id'] == gid for c in contacts)}
            under = None
            for leg, foot in feet.items():
                local = rotation.T @ (foot - pos)
                if abs(local[0]) <= geom['half_size'] and abs(local[1]) <= geom['half_size'] and leg in touching_feet:
                    under = leg
            lin, ang = self._body_speeds(name)
            states[name] = dict(position=pos.tolist(), thickness_mm=geom['thickness_mm'], linear_speed=lin,
                                angular_speed=ang, under_leg=under, flat=bool(rotation[2, 2] > .995),
                                on_surface=bool(abs(pos[2] - (self.work['top_z'] + geom['thickness'] / 2)) < .003),
                                robot_contact=self._robot_touching(contacts), grasped=self._grasped([geom['geom_name']]))
        return states

    def _instant_state(self):
        stand, ball, shims = self.stand_state(), self.ball_state(), self.shim_states()
        level = stand['tilt_deg'] <= LEVEL_TILT_DEG
        ball_still = ball['on_top'] and ball['speed'] < BALL_STILL_SPEED
        released = not (stand['robot_contact'] or stand['grasped'] or ball['robot_contact'] or ball['grasped']
                        or any(s['robot_contact'] or s['grasped'] for s in shims.values()))
        settled = (stand['linear_speed'] < STAND_STILL[0] and stand['angular_speed'] < STAND_STILL[1]
                   and all(s['linear_speed'] < SHIM_STILL for s in shims.values()))
        return dict(ok=bool(level and ball_still and released and settled), level=bool(level),
                    ball_still=bool(ball_still), released=bool(released), settled=bool(settled),
                    stand=stand, ball=ball, shims=shims)

    def _update_level_window(self):
        state = self._instant_state()
        self._level_ticks = self._level_ticks + 1 if state['ok'] else 0
        self._state_cache = (int(self.timestep), state)
        return state

    def _cached_state(self):
        """The tick's own ``_instant_state``; ``_post_action`` computes it once per step."""
        cached = getattr(self, '_state_cache', None)
        if cached is not None and cached[0] == int(self.timestep):
            return cached[1]
        return self._instant_state()

    # ---- protocol hooks (spec 1.4, 1.5; contracts C1, C3) --------------------
    def initial_tilt_deg(self):
        """The unshimmed tilt this instance starts at, from the rigid resting-pose model.

        ``physics_gate`` measures the simulated tilt within 0.001 deg of it, and it
        does not depend on where the episode has got to, so progress is comparable
        across an episode and across instances.
        """
        value = getattr(self, '_initial_tilt_deg', None)
        if value is None:
            value = float(S.configuration_pose(self.spec['short_legs'], self.spec['delta_mm'])['tilt_deg'])
            self._initial_tilt_deg = value
        return value

    def task_progress(self):
        """Fraction of the initial tilt removed, clipped to [0, 1].

        1 only through the success predicate (spec 1.4): a stand that is level but
        has not held the 2 s window, lost the ball off the top or is still in the
        gripper reports ``ALMOST``.
        """
        state = self._cached_state()
        initial = self.initial_tilt_deg()
        removed = 1. - float(state['stand']['tilt_deg']) / initial if initial > 1e-9 else 0.
        value = 0. if removed < PROGRESS_DEADBAND else float(min(removed, 1.))
        if self._current_score()['success']:
            return 1.
        return min(value, ALMOST)

    def collateral_checks(self):
        """Spec 1.5: the ball loose on the counter or the floor, costed as the one ball. Repairable.

        A ball parked in one of the helper containers, or held in the gripper, is not
        collateral -- parking it while a leg is lifted is what the containers are for. Everything
        else off the top is: `bad` turns true the tick the ball comes to rest anywhere else and
        false again when it is put back, which the base logs as `collateral` and `repaired`.
        """
        ball = self._cached_state()['ball']
        bad = not (ball['within_top'] or ball['in_helper'] or ball['held'])
        return dict(ball_off_stand=dict(bad=bool(bad), object='ball', cost=1.))

    def teleport_solution(self):
        """Contract C3: shim the short legs and settle the ball, without the robot.

        The stand is set at the rigid resting pose for the exact shim assignment and
        each matching plate is placed under its foot through their free joints
        (``_place_configuration``, the same routine the physics gate uses), then the
        scene settles with zero-action steps until the success predicate's 2 s window
        has passed. Nothing about the score is short-circuited: the tilt, the ball and
        the settling are all measured from the simulated state.
        """
        steps = []
        try:
            delta = int(self.spec['delta_mm'])
            short = list(self.spec['short_legs'])
            table = self._shims_by_thickness()
            if len(table.get(delta, ())) < len(short):
                steps.append(dict(step='find_shims', ok=False,
                                  detail=f'{len(table.get(delta, ()))} plates of {delta} mm for {len(short)} legs'))
                return steps
            exact = {leg: table[delta][index] for index, leg in enumerate(short)}
            pose = self._place_configuration(exact)
            steps.append(dict(step='shims_under_short_legs', ok=True,
                              detail=dict(under=exact, predicted_tilt_deg=pose['tilt_deg'])))
            action = np.zeros(self.action_dim)
            for _ in range(WINDOW_TICKS + 10):
                self.step(action)
            score = self._current_score()
            steps.append(dict(step='settle', ok=bool(score['level'] and score['settled'] and score['released']),
                              detail=dict(tilt_deg=score['tilt_deg'], released=score['released'],
                                          settled=score['settled'], level=score['level'])))
            steps.append(dict(step='ball_on_top', ok=bool(score['ball_on_top'] and score['ball_still']),
                              detail=dict(distance_from_centre=score['ball_distance_from_centre'],
                                          speed=score['ball_speed'])))
            steps.append(dict(step='success', ok=bool(score['success']),
                              detail=dict(level_ticks=score['level_ticks'], window_ticks=score['window_ticks'],
                                          progress=self.task_progress())))
        except Exception as error:               # never raises (C3)
            steps.append(dict(step='teleport_solution', ok=False, detail=repr(error)))
        return steps

    def _post_action(self, action):
        self._update_level_window()
        return super()._post_action(action)

    def _current_score(self, states=None):
        state = self._instant_state()
        ticks = int(getattr(self, '_level_ticks', 0))
        return dict(success=bool(state['ok'] and ticks >= WINDOW_TICKS), level=state['level'],
                    ball_still=state['ball_still'], released=state['released'], settled=state['settled'],
                    level_ticks=ticks, window_ticks=WINDOW_TICKS, tilt_deg=state['stand']['tilt_deg'],
                    ball_speed=state['ball']['speed'], ball_distance_from_centre=state['ball']['distance_from_centre'],
                    ball_on_top=state['ball']['on_top'], ball_at_rim=state['ball']['at_rim'],
                    shims_under_legs={name: s['under_leg'] for name, s in state['shims'].items()},
                    stand_position=state['stand']['position'])

    def task_snapshot(self):
        return dict(stand=self.stand_state(), ball=self.ball_state(), shims=self.shim_states(),
                    level_ticks=int(getattr(self, '_level_ticks', 0)),
                    progress=self.task_progress(), collateral=self.collateral_checks(),
                    initial_tilt_deg=self.initial_tilt_deg())

    # ---- physics gate (G2) -----------------------------------------------------
    def _tick_physics(self, ticks, on_tick=None):
        """Advance whole 20 Hz ticks with a zero action (the robot holds still), as RoboCasa's reset settle does."""
        action = np.zeros(self.action_spec[0].shape)
        substeps = int(round(self.control_timestep / self.model_timestep))
        for tick in range(ticks):
            policy_step = True
            for _ in range(substeps):
                self.sim.step1()
                self._pre_action(action, policy_step)
                self.sim.step2()
                policy_step = False
            self._update_level_window()
            if on_tick is not None:
                on_tick(tick + 1)

    def _place_configuration(self, shims_under):
        """Put the stand at the rigid resting pose for ``shims_under`` ({leg: shim name}), each named shim
        centred under its foot, the ball at the top centre. Sets joint states directly (CPU gate only)."""
        short, delta = self.spec['short_legs'], self.spec['delta_mm']
        thickness = {leg: self.shim_geoms[name]['thickness_mm'] for leg, name in shims_under.items()}
        pose = S.configuration_pose(short, delta, thickness)
        pos, rotation = self._stand_world_pose(pose['rotation'], pose['z0'])
        self._set_free_joint(self._stand_qadr, self._stand_vadr, pos, _quat_wxyz(rotation))
        yaw = self.work['yaw'] + math.radians(self.spec['stand']['yaw_deg'])
        for leg, name in shims_under.items():
            foot = pos + rotation @ np.asarray(self.stand_geom['feet_local'][leg])
            geom = self.shim_geoms[name]
            self._set_free_joint(self._shim_qadr[name], self._shim_vadr[name],
                                 [foot[0], foot[1], self.work['top_z'] + geom['thickness'] / 2 + PLACE_LIFT / 2],
                                 yaw_quat_wxyz(yaw))
        ball = pos + rotation @ np.array([0., 0., G.BALL_RADIUS + PLACE_LIFT])
        self._set_free_joint(self._ball_qadr, self._ball_vadr, ball, [1., 0., 0., 0.])
        self.sim.forward()
        return pose

    def _shims_by_thickness(self):
        table = {}
        for name, geom in self.shim_geoms.items():
            table.setdefault(int(geom['thickness_mm']), []).append(name)
        return table

    def physics_gate(self):
        """G2 on the built CPU env after reset(): unshimmed tilt and roll, exact shims level, wrong shims do not.

        Leaves the scene modified (shims moved); call reset() before using the env for anything else.
        """
        short, delta = self.spec['short_legs'], int(self.spec['delta_mm'])
        predicted = S.configuration_pose(short, delta)
        state0 = self.sim.get_state()
        checks, report = {}, {}

        def restore():
            self.sim.set_state(state0)
            self.sim.forward()
            self._level_ticks = 0

        # 1. Unshimmed: the stand tilts as predicted and the ball reaches the rim within 3 s.
        origin = np.asarray(self.stand_state()['position'])
        rim = {}

        def watch_rim(tick):
            if 'tick' not in rim and self.ball_state()['at_rim']:
                rim['tick'] = tick
        self._tick_physics(int(GATE_RIM_SECONDS * 20), watch_rim)
        stand = self.stand_state()
        drift = float(np.linalg.norm(np.asarray(stand['position'])[:2] - origin[:2]))
        report['unshimmed'] = dict(predicted_tilt_deg=predicted['tilt_deg'], tilt_deg=stand['tilt_deg'],
                                   rim_time_s=rim['tick'] / 20 if 'tick' in rim else None,
                                   foot_penetration_um=stand['foot_penetration_m'] * 1e6,
                                   foot_clearance_mm={l: v * 1000 for l, v in stand['foot_clearance_m'].items()},
                                   support_geoms=stand['support_geoms'], stand_drift_mm=drift * 1000,
                                   stand_speed=[stand['linear_speed'], stand['angular_speed']],
                                   score_after=self._current_score()['success'])
        checks['unshimmed_tilt_within_10pct'] = abs(stand['tilt_deg'] - predicted['tilt_deg']) <= GATE_TILT_TOL * predicted['tilt_deg']
        checks['ball_reaches_rim_within_3s'] = 'tick' in rim
        checks['foot_penetration_below_0.2mm'] = stand['foot_penetration_m'] <= GATE_PENETRATION_M
        checks['stand_stays_put'] = drift <= .005
        checks['score_not_success_unshimmed'] = not report['unshimmed']['score_after']

        # 2. Exact shims under the short legs: level, ball within 3 cm for 2 s, and the score reports success.
        restore()
        table = self._shims_by_thickness()
        exact = {leg: table[delta][i] for i, leg in enumerate(short)}
        self._place_configuration(exact)
        self._tick_physics(WINDOW_TICKS)
        tilts, dists, speeds = [], [], []

        def watch_level(_):
            tilts.append(self.stand_state()['tilt_deg'])
            ball = self.ball_state()
            dists.append(ball['distance_from_centre'])
            speeds.append(ball['speed'])
        self._tick_physics(WINDOW_TICKS, watch_level)
        score = self._current_score()
        stand = self.stand_state()
        report['exact'] = dict(shims_under=exact, tilt_deg_max=max(tilts), ball_max_distance_mm=max(dists) * 1000,
                               ball_speed_max=max(speeds), foot_penetration_um=stand['foot_penetration_m'] * 1e6,
                               level_ticks=score['level_ticks'], score_success=score['success'],
                               shims_under_legs=score['shims_under_legs'])
        checks['exact_tilt_below_0.2deg'] = max(tilts) <= GATE_LEVEL_DEG
        checks['exact_ball_within_3cm'] = max(dists) <= GATE_BALL_RADIUS
        checks['exact_score_success'] = bool(score['success'])
        checks['exact_shims_detected_under_short_legs'] = all(score['shims_under_legs'][name] == leg for leg, name in exact.items())

        # 3. Wrong thickness under a short leg (mixed sets only): not level.
        report['wrong_thickness'] = {}
        for thickness, names in sorted(table.items()):
            if thickness == delta:
                continue
            restore()
            assignment = {short[0]: names[0]}
            for i, leg in enumerate(short[1:]):
                assignment[leg] = table[delta][i]
            expected = S.configuration_pose(short, delta, {l: self.shim_geoms[n]['thickness_mm'] for l, n in assignment.items()})
            self._place_configuration(assignment)
            rim.clear()
            self._tick_physics(int(GATE_RIM_SECONDS * 20), watch_rim)
            tilt = self.stand_state()['tilt_deg']
            report['wrong_thickness'][thickness] = dict(shims_under=assignment, predicted_tilt_deg=expected['tilt_deg'],
                                                        tilt_deg=tilt, rim_time_s=rim['tick'] / 20 if 'tick' in rim else None)
            checks[f'wrong_{thickness}mm_not_level'] = tilt > GATE_WRONG_DEG

        # 4. The right plates under the wrong legs (the hanging foot, or the long pair): not level.
        restore()
        wrong_legs = S.opposite_legs(short)
        assignment = {leg: table[delta][i] for i, leg in enumerate(wrong_legs)}
        expected = S.configuration_pose(short, delta, {l: delta for l in wrong_legs})
        self._place_configuration(assignment)
        rim.clear()
        self._tick_physics(int(GATE_RIM_SECONDS * 20), watch_rim)
        tilt = self.stand_state()['tilt_deg']
        report['wrong_legs'] = dict(shims_under=assignment, predicted_tilt_deg=expected['tilt_deg'], tilt_deg=tilt,
                                    rim_time_s=rim['tick'] / 20 if 'tick' in rim else None)
        checks['wrong_legs_not_level'] = tilt > GATE_WRONG_DEG

        restore()
        report['checks'] = checks
        return all(checks.values()), report


def stand_local(spec):
    return list(spec['stand']['xy'])
