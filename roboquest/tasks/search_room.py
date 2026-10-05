"""Search Room (tasks-v0 task 1, v1 build): a set of recoloured objects hidden in a RoboCasa kitchen.

Port of the room-search task onto the RoboQuest base class. The work surface,
task frame, tray zone (0.34 m outlined square) and Submit come from the base
class; the tray sits in the frame. ``compartment_count`` openable compartments
are chosen from the kitchen's drawers and cabinet doors with room-search's
discovery rules; **every other drawer, cabinet door and appliance door is
disabled at model build time** (handle geoms hidden, joints locked). Hiding
places have four kinds: ``drawer``, ``door``, ``open`` (a counter patch or
low open shelf that none of the three policy cameras sees from the start pose)
and, since v1, ``covered`` (an open counter spot under an opaque cloche that
must be lifted by its knob; spec 4.4 and the cover rules in 2.2).

v1 additions (spec 4.4): ``set_size`` 2 or 3, the ``covered``
place kind in the compositional split key, passive closers on one or two of the
enabled compartments (drawn in ``sample_spec`` from the ``structure`` stream,
never announced; ``closers.apply_closer`` after ``_setup_references``), the
protocol hooks ``task_progress``, ``collateral_checks`` (``autoclose_closed``)
and the teleport certificate ``teleport_solution`` (contract C3), and a G3 check
that repaints hidden objects in two opaque colours instead of toggling alpha.

Generation is CPU only: the spec fixes the targets, their place kinds, the
hidden distractors, the visible objects, the closer picks and the build seeds;
binding those places to a kitchen's real compartments happens at build time
(deterministic per instance) and fails with a clear error when the layout
cannot host the spec. Policies receive only the fixed cameras, proprioception
and the goal.
"""
from copy import deepcopy
import hashlib
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from robocasa.models.fixtures import Drawer, SingleCabinet, HingeCabinet, Counter, OpenCabinet

from roboquest.closers import CLOSED_FRACTION, apply_closer
from roboquest.kitchen import RoboQuestKitchen
from roboquest.search.covers import COVER_SIZES, build_cover, cover_dims
from roboquest.search.layout import HIGH_FLOOR_M, TRAY_STANDOFF_M, assign_places, base_blockers, capacity_problems, compartment_candidates, cover_fits_spot, cover_setdown_options, floor_bounds_from_walls, footprint_polygon, counter_front_fraction, hidden_pose, high_drawer_capacity, layout_capacity, local_from_world, sample_counter_spots
from roboquest.search.objects import POOL, PLACE_KINDS, confusable, describe_targets, fits, fits_cover, pool_entry
from roboquest.search.skin import skin_geom
from roboquest.search.rules import SearchDiagnostics, compartment_states, inside_interior, target_progress, terminal_score, under_cover
from roboquest.search.visibility import camera_frusta, project, rank_unseen_spots
from roboquest.harness.contract import CAMERAS
from roboquest.base.scene import ASSET_ROOT, VisionKitchenScene

CONTRACT_VERSION = 'search-room-kitchen-v2'
GENERATOR_VERSION = 'v1.1'     # mint-time capacity gate + decor/appliance clearance on every counter spot
GOAL_TEMPLATE = ('Collect {targets} on the tray, then press Submit. They may be anywhere in this kitchen; '
                 'only compartments with handles can be opened. Other objects are not part of the task; '
                 'you may move them if needed.')
HIDDEN_DISTRACTORS = 2         # at most; one when the targets leave a single compartment free (3 targets, 4 compartments)
TRAY_HALF_M = .17
TRAY_LOCAL = (-.14, -.03)      # task frame: tray centre 0.21 m behind the surface's front edge (room-search used 0.22)
SUBMIT_LOCAL = (.20, -.03)     # 0.34 m beside the tray centre; the footprint (0.66 x 0.42) leaves 2 cm at both ends
VISIBLE_SPACING_M = .22        # visible objects keep this far apart and from open hiding spots (objects are <= 0.115 m wide)
START_EDGE_GAP_M = .15         # base frame to the work surface's front edge at the start (RoboCasa's own distance)
DOOR_JOINT_TOKENS = ('doorhinge', 'slidejoint', 'door_joint', 'microjoint', 'drawer')
# Tray slots for the oracle and the teleport certificate (frame-local offsets from the tray centre; objects are
# <= 0.115 m wide, the square is 0.34 m).
TRAY_SLOTS = {1: [(0., 0.)], 2: [(-.075, 0.), (.075, 0.)], 3: [(-.09, -.07), (.09, -.07), (0., .08)]}
# Covers (spec 4.4): the large cloche; keep-outs so the cover can be lifted and put down beside the target.
COVER_SIZE = 'large'
COVER_KEEPOUT_M = .40          # visible objects and open hiding spots keep this far from a covered spot's centre
SETDOWN_KEEPOUT_M = .22        # ... and from each cover set-down centre (cloche radius 0.11 + object + margin)
COVER_SPACING_M = .55          # between two covered spots
COVER_EDGE_DEPTH_MIN_M = .14   # the cover's rim stays at least 3 cm inside the approachable edge
# Passive closers (spec 4.4): one or two enabled compartments per instance, drawn from `structure`.
CLOSER_COUNT = (1, 2)
# Design closing time from fully open: the single knob. Drawers use the *preloaded* spring the container
# mechanisms use (containers/springs.py) instead of the shared exponential recipe. The rest position sits
# CLOSER_PRELOAD_TRAVELS travels past the closed limit and the joint's dry friction is removed, so the force
# k (x + R) changes by only 1 + L/R over the pull and the over-damped return is nearly a straight line,
# x(t) = (L + R) exp(-t/tau) - R with tau = d / k (see `drawer_closer_design`). An exponential spring (rest at
# the limit) is fastest exactly where it must not be -- it covers half the travel in a tenth of the closing
# time -- which is why the first design needed 90 s to stay out of the way: the oracle releases the handle and
# is back inside the drawer 8 to 14 s later. Measured 2026-09-21 on layout 13 style 4, headless, with
# `closers.closing_time` (see the handoff and scripts/roboquest_search_closer.py).
CLOSER_TIME_S = {'drawer': 25., 'door': 2.}
CLOSER_PRELOAD_TRAVELS = 3.        # drawer spring rest: this many travels past the closed limit (the preload)
CLOSER_PRELOAD_N = .12             # least drawer spring force at the closed limit, so a weak spring never stalls
CLOSER_DRAWER_FRICTION = 0.        # a closer drawer's dry friction: none, the damper alone sets the speed
# RoboCasa cabinet leaves rub their frames (about 1 mm of penetration through the swing) and jam near 0.7 rad,
# so a door's closing is friction- and geometry-bound and keeps the measured shared-recipe design: the 2 s
# design (3.9 N m/rad) closes a leaf from 0.45 rad in about 20 s, the 10 s design never does.
CLOSER_REST_PAST = {'drawer': .01, 'door': .03}    # spring rest this far past the closed limit (m, rad): shuts firmly
PROGRESS_OUT_OF_SPOT_M = .15                       # an `open` target counts as out of its place this far from the spot
SETTLE_TICKS = 40                                  # zero-action ticks after the teleport edits
# Decor avoidance (build finding 2026-09-21): every counter placement the task makes outside a compartment
# keeps the object's radius plus this margin from the kitchen's decor and counter-top appliances. The first
# v1 registry clashed 3 to 35 mm deep with toaster ovens, paper-towel holders, plants and coffee machines
# because the spot samplers only saw the fixture *records*, whose `size` is not an accessory's bounding box.
DECOR_MARGIN_M = .02
POOL_RADIUS_M = max(max(e['size_xyz_m'][0], e['size_xyz_m'][1]) for e in POOL.values()) / 2   # 0.0575 (the mug)
SPOT_CLEARANCE_M = POOL_RADIUS_M + DECOR_MARGIN_M
DECOR_BAND_M = .40                                 # decor is read once per counter-top height, as observe does
# Drawer-type structure (search room top-up of 2026-09-24): a recorded variant for set_size 3
# whose three targets and every decoy hide only in drawers with an interior floor >= 0.40 m or on the counter
# under a cloche -- no door targets, no bottom drawers (floor 0.32 m), no open places (an open target can fall
# back to any compartment, a door included). The two compositions are the ones no development row has, and
# both are held out, so a drawer-type row is an evaluation row by the task's own compositional rule. A spec
# records the variant in `structure_variant` and the floor in `hide_min_floor_m`; specs without the fields
# (every row minted before it) validate and build exactly as before.
DRAWER_TYPE = 'drawer-type'
DRAWER_TYPE_CLASSES = (('drawer', 'drawer', 'drawer'), ('covered', 'covered', 'drawer'))
DRAWER_TYPE_MIN_FLOOR_M = HIGH_FLOOR_M
# Counter places of a drawer-type spec: on a counter whose back is against a wall the
# object stands in the front two thirds of the counter's depth, measured from the front edge; on an island
# (approached from both sides) anywhere across the depth. Recorded as `counter_front_max_fraction`.
DRAWER_TYPE_COUNTER_FRONT_MAX = .6667
DRAWER_TYPE_HOLDOUT_KEYS = tuple('s3-' + '+'.join(sorted(c)) for c in DRAWER_TYPE_CLASSES)


def drawer_closer_design(travel, joint_damping, close_time_s, closed_tol,
                         preload_travels=CLOSER_PRELOAD_TRAVELS, preload_n=CLOSER_PRELOAD_N):
    """(rest_past, stiffness, damping) for a preloaded drawer spring that shuts ``travel`` in ``close_time_s``.

    The rest position sits ``R = preload_travels * travel`` past the closed limit, so the force k (x + R)
    changes by only 1 + travel / R over the pull. Over-damped, the slide runs at its terminal velocity
    k (x + R) / d, i.e. x(t) = (travel + R) exp(-t / tau) - R with tau = d / k, and asking x(T) = closed_tol
    fixes tau. The stiffness is then the smallest that keeps RoboCasa's own joint damping in place and still
    presses with ``preload_n`` at the closed limit; the damping follows as k tau. Raising ``preload_n`` makes
    the drawer press harder (and, at the same tau, damps proportionally more) without changing the timing.
    """
    rest_past = float(preload_travels) * float(travel)
    tau = float(close_time_s) / math.log((travel + rest_past) / (float(closed_tol) + rest_past))
    stiffness = max(float(joint_damping) / tau, float(preload_n) / rest_past)
    return rest_past, stiffness, stiffness * tau


def drawer_closer_profile(travel, stiffness, damping, rest_past, seconds):
    """Where the modelled drawer is at each of ``seconds`` after release: the over-damped terminal-velocity
    solution above, as a fraction of the pull still open. A check on the measurement, not a substitute."""
    tau = float(damping) / float(stiffness)
    return [max(0., ((travel + rest_past) * math.exp(-float(t) / tau) - rest_past) / travel) for t in seconds]


def object_name(pool_id):
    return f'obj_{pool_id}'


def region_geom_names(asset):
    """Names of a native pool asset's ``class="region"`` geoms: invisible markers (reg_bbox, reg_int, the
    mug's ``liquid``) that MJCFObject recolours along with the visual meshes."""
    root = ET.parse(ASSET_ROOT / asset / 'model.xml').getroot()
    return sorted(g.get('name') for g in root.iter('geom') if g.get('class') == 'region' and g.get('name'))


def hide_region_markers(obj, asset):
    """Set alpha 0 on the object's region-class geoms; returns the geom names touched. Render only: they keep
    contype/conaffinity 0, their size and density, so the model's mass and contacts are unchanged."""
    markers = {obj.naming_prefix + n for n in region_geom_names(asset)}
    touched = []
    for geom in obj.worldbody.iter('geom'):
        if geom.get('name') in markers:
            rgba = geom.get('rgba', '0 0 0 0').split()
            geom.set('rgba', ' '.join(rgba[:3] + ['0']))
            touched.append(geom.get('name'))
    return touched


def cover_name(place_key):
    return f'cover_{place_key}'


def compartment_kinds(kinds):
    return [k for k in kinds if k in ('drawer', 'door')]


def hidden_distractor_count(compartment_count, kinds):
    """Two confusable distractors hide in compartments, or one when the targets leave only one free."""
    free = int(compartment_count) - len(compartment_kinds(kinds))
    return max(1, min(HIDDEN_DISTRACTORS, free))


class SearchRoom(RoboQuestKitchen):
    TASK_NAME = 'search_room'
    CONTRACT_VERSION = CONTRACT_VERSION
    GENERATOR_VERSION = GENERATOR_VERSION
    FACTORS = {'set_size': (2, 3), 'compartment_count': (4, 6, 8)}
    FOOTPRINT = (.66, .42)     # tray square + Submit; small enough for RoboCasa's 0.70 m counter regions
    # Compositional eval holdout; the drawer-type variant adds two compositions (no development
    # row has either, so no row changes split).
    HOLDOUT_KEYS = ('s2-covered+open', 's3-door+door+drawer') + DRAWER_TYPE_HOLDOUT_KEYS
    USE_CAPACITY_TABLE = True                     # the capacity probe turns this off while it measures the table
    HORIZON = 20000                               # ticks at 20 Hz (v0); kept for the oracle's default
    BUDGET_TICKS = 20000                          # provisional (C1): revisit from the G4 certificates
    submit_prefix = 'submit'
    # Handle geoms hidden with a locked fixture, by the names RoboCasa gives them under the fixture's own
    # name (cabinets and microwaves: door_handle_, left_door_handle_, right_door_handle_). A subclass extends
    # the tuple: the fridge names its handles fridge_door_handle_ and freezer_door_handle_, which this rule
    # misses, so a locked fridge keeps its handles in view here (the fob search hides them).
    HIDDEN_HANDLE_PATTERNS = ('door_handle_', 'left_door_handle_', 'right_door_handle_')
    # Fixture classes whose every handle geom is hidden when the fixture is locked, whatever the geom is called
    # (RoboCasa numbers the fridge's: fridge0_door_handle_2, freezer0_door_handle_main). Empty here, so the
    # search room builds as before; the v1.1 variants name the fridges.
    HIDE_ALL_HANDLES_OF = ()
    # Whether the compartments named by spec['closers'] get their passive springs at build. The v1.1 variants
    # turn it off: their compartments stay where the robot leaves them.
    APPLY_CLOSERS = True

    # ---- generator protocol (CPU only) --------------------------------------
    @classmethod
    def goal(cls, spec):
        return GOAL_TEMPLATE.format(targets=describe_targets([t['object'] for t in spec['targets']]))

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        set_size, count = int(factors['set_size']), int(factors['compartment_count'])
        ids = sorted(POOL)
        targets = [ids[int(i)] for i in structure.choice(len(ids), set_size, replace=False)]
        kinds = [POOL[t]['kinds'][int(structure.integers(len(POOL[t]['kinds'])))] for t in targets]
        others = [o for o in ids if o not in targets]
        candidates = [o for o in others if any(confusable(o, t) for t in targets)]
        n_hidden = min(hidden_distractor_count(count, kinds), len(candidates))
        hidden = sorted(candidates[int(i)] for i in structure.choice(len(candidates), n_hidden, replace=False))
        visible = [o for o in others if o not in hidden]
        build_seed = int(structure.integers(0, 2**31 - 1))
        pose_seed = int(poses.integers(0, 2**31 - 1))
        # Passive closers: which of the enabled compartments (by rank in their sorted ids, bound at build time).
        closer_count = int(CLOSER_COUNT[int(structure.integers(len(CLOSER_COUNT)))])
        closers = sorted(int(i) for i in structure.choice(count, closer_count, replace=False))
        cover_seed = int(materials.integers(0, 2**31 - 1))
        # Reject a kitchen that cannot host this cell before anything is built, so `registry.mint` raises and
        # `mint_grid` walks straight on to the next seed. The build keeps its own check as the safety net.
        problems = cls.capacity_problems(int(factors['kitchen_layout']), count,
                                         [dict(object=t, kind=k) for t, k in zip(targets, kinds)], hidden, visible)
        if problems:
            raise ValueError('; '.join(problems))
        return dict(
            set_size=set_size, compartment_count=count, layout_id=int(factors['kitchen_layout']),
            targets=[dict(object=t, kind=k) for t, k in zip(targets, kinds)],
            hidden_distractors=hidden, visible=visible,
            build_seed=build_seed, pose_seed=pose_seed, closers=closers,
            cover_size=COVER_SIZE, cover_seed=cover_seed,
            placement=dict(tray=list(TRAY_LOCAL), submit=list(SUBMIT_LOCAL)), tray_half_m=TRAY_HALF_M)

    @classmethod
    def structure_variant(cls, name):
        """The generator class for a recorded structure variant (``replace_instances --structure``)."""
        if name != DRAWER_TYPE or cls is not SearchRoom:
            raise ValueError(f'{cls.TASK_NAME}: no structure variant {name!r} (known: {DRAWER_TYPE!r})')
        return SearchRoomDrawerType

    @classmethod
    def hide_rule(cls, spec):
        """``assign_places``' hiding restriction for this spec: ``{}`` (none) unless it records the variant."""
        if spec.get('structure_variant') != DRAWER_TYPE:
            return {}
        return dict(hide_kinds=('drawer',), min_floor_z=float(spec['hide_min_floor_m']))

    @classmethod
    def drawer_type_problems(cls, spec):
        """The drawer-type variant's invariants for a spec; ``[]`` for a spec without the variant fields."""
        variant = spec.get('structure_variant')
        if variant is None and 'hide_min_floor_m' not in spec:
            return []
        problems = []
        if variant != DRAWER_TYPE:
            return [f'unknown structure_variant {variant!r}']
        if spec.get('hide_min_floor_m') != DRAWER_TYPE_MIN_FLOOR_M:
            problems.append(f'a drawer-type spec records hide_min_floor_m = {DRAWER_TYPE_MIN_FLOOR_M}')
        if spec.get('counter_front_max_fraction') != DRAWER_TYPE_COUNTER_FRONT_MAX:
            problems.append(f'a drawer-type spec records counter_front_max_fraction = {DRAWER_TYPE_COUNTER_FRONT_MAX}')
        targets = spec.get('targets') or []
        kinds = tuple(sorted(t.get('kind') for t in targets))
        if spec.get('set_size') != 3 or kinds not in {tuple(sorted(c)) for c in DRAWER_TYPE_CLASSES}:
            problems.append(f'a drawer-type spec has three targets as one of {DRAWER_TYPE_CLASSES}, not {kinds}')
        for o in spec.get('hidden_distractors') or []:
            if o in POOL and 'drawer' not in POOL[o]['kinds']:
                problems.append(f'drawer-type decoy {o!r} cannot hide in a drawer')
        return problems

    @classmethod
    def validate_spec(cls, spec):
        problems = list(cls.drawer_type_problems(spec))
        targets = spec.get('targets') or []
        if spec.get('set_size') not in cls.FACTORS['set_size'] or len(targets) != spec.get('set_size'):
            problems.append('set_size must be 2 or 3 and match the target list')
        count = spec.get('compartment_count')
        if count not in cls.FACTORS['compartment_count']:
            problems.append('compartment_count must be 4, 6 or 8')
        objects = [t.get('object') for t in targets]
        if len(set(objects)) != len(objects) or any(o not in POOL for o in objects):
            problems.append('targets must be distinct pool objects')
        kinds = [t.get('kind') for t in targets]
        for t in targets:
            if t.get('kind') not in PLACE_KINDS:
                problems.append(f"unknown place kind {t.get('kind')!r}")
            elif t.get('object') in POOL and t['kind'] not in POOL[t['object']]['kinds']:
                problems.append(f"{t['object']} may not hide in a {t['kind']}")
        hidden = spec.get('hidden_distractors') or []
        expected_hidden = hidden_distractor_count(count, kinds) if isinstance(count, int) else HIDDEN_DISTRACTORS
        if not 1 <= len(hidden) <= expected_hidden or len(set(hidden)) != len(hidden) or set(hidden) & set(objects):
            problems.append(f'between 1 and {expected_hidden} distinct hidden distractors, none of them a target')
        for o in hidden:
            if o not in POOL or not any(confusable(o, t) for t in objects if t in POOL):
                problems.append(f'hidden distractor {o!r} shares neither category nor colour with a target')
        expected_visible = sorted(o for o in POOL if o not in objects and o not in hidden)
        if sorted(spec.get('visible') or []) != expected_visible:
            problems.append('visible objects must be the rest of the pool')
        needed = len(compartment_kinds(kinds)) + len(hidden)
        if isinstance(count, int) and needed > count:
            problems.append('more hidden objects than compartments')
        for key in ('build_seed', 'pose_seed', 'cover_seed'):
            if not isinstance(spec.get(key), int):
                problems.append(f'{key} must be an int')
        closers = spec.get('closers')
        if (not isinstance(closers, list) or len(closers) not in CLOSER_COUNT or len(set(closers)) != len(closers)
                or not all(isinstance(c, int) and isinstance(count, int) and 0 <= c < count for c in closers)):
            problems.append('closers must list one or two distinct compartment ranks below compartment_count')
        if spec.get('cover_size') not in COVER_SIZES:
            problems.append('cover_size must name a cloche size')
        placement = spec.get('placement') or {}
        if not all(isinstance(placement.get(k), list) and len(placement[k]) == 2 for k in ('tray', 'submit')):
            problems.append('placement needs tray and submit frame positions')
        return problems

    @classmethod
    def capacity_problems(cls, layout_id, compartment_count, targets, hidden_distractors=(), visible=()):
        """Reasons the capacity table says this layout cannot host the spec (``[]`` when it can).

        A hook so a probe can turn the table off (``USE_CAPACITY_TABLE = False``) while measuring it.
        """
        if not cls.USE_CAPACITY_TABLE:
            return []
        return capacity_problems(layout_id, compartment_count, targets, hidden_distractors, visible)

    @classmethod
    def spec_capacity_problems(cls, spec):
        """``capacity_problems`` for a whole spec. A drawer-type spec (41c) must also hide every target and decoy
        in the high drawers; its empty filler compartments may be any kind, so the count is checked once, on the
        full table."""
        targets = spec.get('targets') or []
        layout = spec['layout_id']
        hidden, visible = spec.get('hidden_distractors') or [], spec.get('visible') or []
        problems = cls.capacity_problems(layout, int(spec.get('compartment_count') or 0), targets, hidden, visible)
        if not problems and cls.USE_CAPACITY_TABLE and spec.get('structure_variant') == DRAWER_TYPE:
            problems = [p.replace('(capacity table)', f'(high drawers, floor >= {spec["hide_min_floor_m"]} m)')
                        for p in capacity_problems(layout, 0, targets, hidden, visible,
                                                   capacity=high_drawer_capacity(layout))]
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """CPU-only checks: spec invariants plus the layout capacity table (search/layout.py)."""
        problems = cls.validate_spec(spec)
        count = int(spec.get('compartment_count') or 0)
        targets = spec.get('targets') or []
        kinds = [t.get('kind') for t in targets]
        hidden = spec.get('hidden_distractors') or []
        capacity = layout_capacity(spec.get('layout_id')) if spec.get('layout_id') is not None else 'unknown'
        report = dict(hidden_compartments_needed=len(compartment_kinds(kinds)) + len(hidden),
                      target_kinds=kinds, capacity_layout=spec.get('layout_id'), closers=spec.get('closers'),
                      capacity=None if capacity in (None, 'unknown') else dict(capacity))
        if not problems and all(t.get('object') in POOL for t in targets) and all(o in POOL for o in hidden):
            problems = problems + cls.spec_capacity_problems(spec)
        report['problems'] = problems
        return not problems, report

    @classmethod
    def split_key(cls, spec):
        return f"s{spec['set_size']}-" + '+'.join(sorted(t['kind'] for t in spec['targets']))

    # ---- construction -------------------------------------------------------
    def __init__(self, instance, **kwargs):
        self._records = None
        self._floor = None
        self._blockers = None
        self._keepouts = None
        self.compartments = {}
        self.places = {}
        self.covers = {}
        self.tray = None
        self._open_candidates = []
        self._open_choice = {}
        self._open_bindings = {}      # the place fields the first reset bound (spot, fallback), replayed after
        self._covered_spots = []
        self._closer_ids = []
        self._closers = {}
        self._disabled = dict(locked_joints=[], hidden_handle_geoms=[], fixtures=[])
        self._joint_addresses = {}
        self._object_geoms = {}
        self._search = None
        self._camera_size = None
        super().__init__(instance, **kwargs)

    # ---- build-time discovery -----------------------------------------------
    def _hiding_admissible(self, candidate):
        """May this compartment hide a target or a hidden distractor? Every candidate here; the dark search
        keeps only drawers with their interior floor at 0.40 m or higher. Fillers are not
        filtered: ``assign_places`` draws them from every candidate."""
        return True

    def _counter_spot_admissible(self, spot):
        """May this sampled counter spot hold a target (open or under a cloche)? Every spot here; the dark
        search applies the counter band (the front two thirds of a wall-backed counter)
        and the floor rule to open shelves."""
        return True

    def _fixture_records(self):
        records = {}
        for name, fixture in self.fixtures.items():
            try:
                pos = [float(v) for v in fixture.pos]
            except Exception:
                pos = None
            record = {'name': name, 'cls': type(fixture).__name__, 'pos': pos,
                      'rot': float(getattr(fixture, 'rot', 0.) or 0.),
                      'size': [float(v) for v in getattr(fixture, 'size', [0., 0., 0.])]}
            if isinstance(fixture, (Drawer, SingleCabinet, HingeCabinet)):
                record['panel_type'] = getattr(fixture, 'panel_type', None)
                record['is_corner'] = getattr(fixture, 'is_corner_cab', None)
                record['orientation'] = getattr(fixture, 'orientation', None)
                regions = fixture.get_reset_regions(reset_region_names=None, z_range=None)
                record['regions'] = {k: {'offset': [float(x) for x in v['offset']],
                                         'size': [float(x) for x in v['size']],
                                         'height': float(v['height'])} for k, v in regions.items()}
                record['door_joints'] = list(fixture.door_joint_names)
                record['joint_ranges'] = {j: [float(x) for x in fixture._joint_infos[j]['range']]
                                          for j in record['door_joints']}
                record['handles'] = [getattr(fixture, a) for a in ('handle_name', 'left_handle_name', 'right_handle_name')
                                     if hasattr(type(fixture), a)]
            elif isinstance(fixture, OpenCabinet):
                try:
                    regions = fixture.get_reset_regions(self, z_range=None)
                except Exception:
                    regions = {}
                record['regions'] = {k: {'offset': [float(x) for x in v['offset']],
                                         'size': [float(x) for x in v['size']],
                                         'height': float(v['height'])} for k, v in regions.items()}
            elif isinstance(fixture, Counter):
                try:
                    regions = fixture.get_reset_regions(self)
                except Exception:
                    regions = {}
                record['regions'] = {k: {'offset': [float(x) for x in v['offset']],
                                         'size': [float(x) for x in v['size']]} for k, v in regions.items()}
            records[name] = record
        return records

    def _fixture_door_joints(self):
        """Every door/drawer-like joint of every fixture, by fixture (cabinets, drawers, appliances)."""
        joints = {}
        for name, fixture in self.fixtures.items():
            infos = getattr(fixture, '_joint_infos', None) or {}
            found = []
            try:
                found += list(fixture.door_joint_names)
            except Exception:
                pass
            found += [j for j in infos if any(tok in j.lower() for tok in DOOR_JOINT_TOKENS)]
            found = sorted(set(found))
            if found:
                joints[name] = found
        return joints

    # ---- scene construction -------------------------------------------------
    def _decor_boxes(self, top_z):
        """Counter decor plus every other fixture standing on the work top (toasters, coffee machines,
        stand mixers...). The base class only avoids ``Accessory`` decor; on layout 13 the toaster
        would otherwise sit inside the tray square. Base-class request: adopt this rule."""
        boxes = super()._decor_boxes(top_z)
        named = {b['name'] for b in boxes}
        for name, fixture in self.fixtures.items():
            if name in named or isinstance(fixture, (Counter, Drawer, SingleCabinet, HingeCabinet, OpenCabinet)):
                continue
            if type(fixture).__name__ in ('Wall', 'Floor', 'Window', 'Box', 'Hood', 'Stool', 'WallAccessory'):
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            bottom = float(points[:, 2].min())
            if not (top_z - .10 <= bottom <= top_z + .40):
                continue
            boxes.append(dict(name=name, x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                              y0=float(points[:, 1].min()), y1=float(points[:, 1].max())))
        return boxes

    def _decor_keepouts(self):
        """The kitchen's real decor and appliance footprints, per counter-top height.

        Returns the callable ``search.layout.sample_counter_spots`` and the cover helpers take: given a
        surface's ``top_z`` it hands back the boxes standing on that surface, read from
        ``get_bbox_points()`` exactly as ``observe.decor_shapes_on_work`` reads them for the work top.
        The fixture *records* the layout rules work from cannot see this: an ``Accessory``'s ``size`` is
        not its bounding box, so plants, paper-towel holders, toasters, toaster ovens and coffee machines
        were invisible to ``counter_top_occupants`` and the task dropped objects into them (G1, first v1
        registry). Cached per height because a kitchen has only two or three counter-top levels.
        """
        cache = {}

        def boxes(top_z):
            key = round(float(top_z), 3)
            if key not in cache:
                cache[key] = self._decor_boxes(key)
            return cache[key]
        return boxes

    def _work_fixture_candidates(self):
        """The base class's work fixture first, then every other qualifying surface (islands, then counters, by area)."""
        from robocasa.models.fixtures.fixture_utils import fixture_is_type
        import robocasa.models.fixtures.fixture_utils as FixtureUtils
        ordered = [self.work_fixture]
        for kind in self.WORK_FIXTURE_TYPES:
            rows = []
            for name in sorted(self.fixtures):
                fixture = self.fixtures[name]
                if fixture is self.work_fixture or not fixture_is_type(fixture, kind):
                    continue
                try:
                    if not FixtureUtils.is_fxtr_valid(self, fixture, self.FOOTPRINT):
                        continue
                    regions = fixture.get_reset_regions(self, top_size=self.FOOTPRINT)
                except Exception:
                    continue
                if regions:
                    rows.append((-max(r['size'][0] * r['size'][1] for r in regions.values()), name))
            ordered += [self.fixtures[name] for _, name in sorted(rows)]
        return ordered

    def place_work_frame(self):
        """Base placement, extended: when every sideways offset of the largest region is blocked by decor or
        a counter-top appliance, try the surface's other regions, then the other surfaces; fail the build
        rather than lay the frame under an appliance. Base-class request: adopt this search."""
        tried = []
        for fixture in self._work_fixture_candidates():
            regions = fixture.get_reset_regions(self, top_size=self.FOOTPRINT)
            top_z = float(fixture.pos[2] + fixture.size[2] / 2)
            yaw = float(fixture.rot)
            decor = self._decor_boxes(top_z)
            for region_name, region in sorted(regions.items(), key=lambda kv: -kv[1]['size'][0] * kv[1]['size'][1]):
                rx, ry = float(region['offset'][0]), float(region['offset'][1])
                sx, sy = float(region['size'][0]), float(region['size'][1])
                cy = ry - sy / 2 + self.FOOTPRINT[1] / 2 + self.FOOTPRINT_FRONT_MARGIN
                slack = max(0., sx / 2 - self.FOOTPRINT[0] / 2)
                offsets = [0.] + [s * d for d in np.arange(.10, slack + 1e-9, .10) for s in (-1, 1)]
                blocked_by = []
                for dx in offsets:
                    box = self._footprint_world_box([rx + dx, cy], yaw, fixture)
                    hits = [d['name'] for d in decor if self._boxes_overlap(box, d)]
                    if hits:
                        blocked_by.append((float(dx), hits))
                        continue
                    self.work_fixture = fixture
                    self.init_robot_base_ref = fixture
                    center_local = [rx + dx, cy]
                    center_world = self.fixture_local_to_world(fixture, [center_local[0], center_local[1], fixture.size[2] / 2])
                    self.work = dict(fixture=fixture.name, region=region_name, top_z=top_z, yaw=yaw,
                                     center_local=[float(v) for v in center_local],
                                     center_world=[float(center_world[0]), float(center_world[1]), top_z],
                                     footprint=list(self.FOOTPRINT), decor_blocked=blocked_by,
                                     decor_on_surface=[d['name'] for d in decor], regions_tried=tried)
                    self.task_spec['work'] = deepcopy(self.work)
                    self.task_spec['supports'][fixture.name] = {
                        'bounds_xy': [float(center_world[0] - sx / 2), float(center_world[0] + sx / 2),
                                      float(center_world[1] - sy / 2), float(center_world[1] + sy / 2)],
                        'height_z': top_z, 'fixture': fixture.name, 'rotation_z_rad': yaw}
                    return self.work
                tried.append((fixture.name, region_name, blocked_by))
        raise ValueError(f'{self.TASK_NAME}: layout {self.layout_id} has no counter region free of decor and appliances '
                         f'for the {self.FOOTPRINT} m frame; tried {[(f, r) for f, r, _ in tried]}')

    def _work_polygon(self, margin=.10):
        hx, hy = self.FOOTPRINT[0] / 2 + margin, self.FOOTPRINT[1] / 2 + margin
        return [self.frame_to_world([sx * hx, sy * hy, 0.])[:2].tolist() for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    def _build_tray(self):
        tx, ty = self.spec['placement']['tray']
        half = float(self.spec['tray_half_m'])
        body = self.add_frame_body('tray_frame', (tx, ty), z=0.)
        marks = []
        # White outline: no pool colour appears on anything but the pool objects and the red button.
        for i, (x, y, sx, sy) in enumerate([(0., -half, half, .003), (0., half, half, .003),
                                            (-half, 0., .003, half), (half, 0., .003, half)]):
            name = f'tray_mark_{i}'
            ET.SubElement(body, 'geom', name=name, type='box', size=f'{sx} {sy} .0005', pos=f'{x} {y} .0015',
                          rgba='.96 .96 .96 1', contype='0', conaffinity='0', group='1')
            marks.append(name)
        centre = self.frame_to_world([tx, ty, 0.])
        front_y = -self.FOOTPRINT[1] / 2 - self.FOOTPRINT_FRONT_MARGIN     # the surface's front edge, task frame
        facing = float(self.work['yaw'] + math.pi / 2)                     # base yaw that looks along frame +y
        standoff = self.frame_to_world([tx, front_y - TRAY_STANDOFF_M, 0.])
        sx, sy = self.spec['placement']['submit']
        submit_standoff = self.frame_to_world([sx, front_y - TRAY_STANDOFF_M, 0.])
        corners = [self.frame_to_world([tx + ax * half, ty + ay * half, 0.]).tolist() for ax, ay in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        self.tray = dict(centre_xy=[float(centre[0]), float(centre[1])], top_z=float(self.work['top_z']),
                         yaw=float(self.work['yaw']), half=half, centre_local=[float(tx), float(ty)],
                         corners_world=corners, mark_geoms=marks, support=self.work['fixture'],
                         standoff=dict(xy=[float(standoff[0]), float(standoff[1])], yaw=facing),
                         submit_standoff=dict(xy=[float(submit_standoff[0]), float(submit_standoff[1])], yaw=facing))
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        self.task_spec['zones']['tray'] = {'bounds_xy': [min(xs), max(xs), min(ys), max(ys)], 'height_z': self.tray['top_z'],
                                           'support_id': self.work['fixture'], 'note': 'axis-aligned bounds; scoring uses the frame square'}
        return self.tray

    def _add_object(self, pool_id, xy, support_z, yaw, role):
        entry = pool_entry(pool_id)
        name = object_name(pool_id)
        quat = Rotation.from_euler('z', yaw).as_quat()
        obj = self.add_native(name, entry['asset'], [float(xy[0]), float(xy[1])], float(support_z),
                              scale=1., quat_xyzw=quat.tolist(), rgba=entry['rgba'])
        # Declared visual override: the native texture (brand artwork) is dropped on visual geoms and each
        # visual geom gets the skin in the object's pool colour (search/skin.py: generic label + silver ends on
        # can/tin, black bottle cap, speckled bottle body, two-tone ceramic mug); render only, collision
        # geometry and mass are untouched.
        for geom in obj.worldbody.iter('geom'):
            if geom.get('group') == '1' and geom.get('material') is not None:
                native = geom.attrib.pop('material')
                geom.set('rgba', ' '.join(map(str, entry['rgba'])))
                skin_geom(obj, geom, entry, native)
        # MJCFObject's rgba paints every group-1 geom and RoboCasa re-hides only the reg_* ones, so the mug's
        # region-class ``liquid`` cylinder would render as an opaque disc filling the cup. Hide the region
        # markers again (render only).
        hide_region_markers(obj, entry['asset'])
        self.task_spec['objects'][name].update(kind=entry['category'], pool_id=pool_id, colour=entry['colour'],
                                               description=entry['description'], grasp=entry['grasp'],
                                               height_m=entry['height'], size_xyz_m=list(entry['size_xyz_m']),
                                               role=role, yaw=float(yaw))
        self.asset_evidence[name].update(colour=entry['colour'], pool_id=pool_id, role=role,
                                         visual_override='skin in the pool colour replaces native texture on visual geoms')
        return obj

    def _add_cover(self, key, place, spot, rng):
        """The shared cloche (assets, C4) over the covered target at ``spot``: rim on the counter, centred on the
        object, dressed from the instance's ``cover_seed`` (materials stream)."""
        name = cover_name(key)
        pool_id = place['object']
        size = self.spec['cover_size']
        dims = cover_dims(size)
        if not fits_cover(pool_id, dims['inner_radius'], dims['inner_height']):
            raise ValueError(f'{self.TASK_NAME}: {pool_id!r} does not fit under the {size} cloche')
        yaw = float(rng.uniform(-np.pi, np.pi))
        z = float(spot['top_z']) + .002
        pos = (float(spot['xy'][0]), float(spot['xy'][1]), z)
        obj, built = build_cover(name, size, rng, self.model.asset, pos, yaw)
        self.objects[name] = obj
        q = Rotation.from_euler('z', yaw).as_quat()
        self._poses[name] = [pos[0], pos[1], z, float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        record = {k: built[k] for k in ('radius', 'height', 'inner_radius', 'inner_height', 'knob_site', 'knob_grasp_z',
                                        'knob_top_z', 'mass', 'material', 'rgba', 'size')}
        record['material_kind'] = built['kind']
        self.task_spec['objects'][name] = dict(kind='cover', fixed=False, cover_kind=f"cloche_{built['size']}",
                                               covers=object_name(pool_id), role='cover', yaw=yaw, **record)
        self.asset_evidence[name] = dict(source='roboquest.assets.cloche',
                                         collision=f"hollow ring of {built['segments']} thin boxes plus a thin top disk; "
                                                   f"knob pinched at site {built['knob_site']}",
                                         mass_kg=built['mass'], appearance=built['material'] or built['rgba'])
        self.covers[name] = dict(target=object_name(pool_id), place=key, spot=spot, yaw=yaw,
                                 setdown_options=place.get('setdown_options', []), **record)
        place.update(cover=name, kind_effective='covered')
        return built

    def _choose_covered_spot(self, radius, exclude):
        """The first edge-approachable counter spot (with a base standoff) that keeps its distance from the other
        covered spots and offers at least one place to put the lifted cover down."""
        front_max = self.spec.get('counter_front_max_fraction')
        for spot in self._open_candidates:
            if spot.get('is_shelf') or spot.get('standoff') is None or spot['edge_depth'] < COVER_EDGE_DEPTH_MIN_M:
                continue
            if front_max is not None:
                # A wall-backed counter's front two thirds only; an island anywhere
                island, fraction = counter_front_fraction(self._records, spot)
                if not island and not 0. <= fraction <= float(front_max):
                    continue
                spot = dict(spot, island=bool(island), front_fraction=round(fraction, 4))
            if any(np.linalg.norm(np.subtract(spot['xy'], c['xy'])) < COVER_SPACING_M for c in self._covered_spots):
                continue
            if not cover_fits_spot(spot, self._records, radius, exclude=exclude, keepouts=self._keepouts):
                continue
            options = cover_setdown_options(spot, self._records, radius, exclude=exclude, keepouts=self._keepouts)
            if not options:
                continue
            return spot, options
        raise ValueError(f'{self.TASK_NAME}: no counter spot with room for a covered target on layout {self.layout_id}')

    def _clear_of_covers(self, xy, keepout=COVER_KEEPOUT_M, setdown_keepout=SETDOWN_KEEPOUT_M):
        for spot in self._covered_spots:
            if np.linalg.norm(np.subtract(xy, spot['xy'])) < keepout:
                return False
            if any(np.linalg.norm(np.subtract(xy, o['xy'])) < setdown_keepout for o in spot.get('setdown_options', [])):
                return False
        return True

    def build_task(self):
        self.place_work_frame()
        self._records = self._fixture_records()
        self._floor = floor_bounds_from_walls(self._records)
        self._blockers = base_blockers(self._records)
        candidates = compartment_candidates(self._records, self._floor)
        count = int(self.spec['compartment_count'])
        if len(candidates) < count:
            raise ValueError(f'{self.TASK_NAME}: layout {self.layout_id} offers {len(candidates)} usable compartments, '
                             f'fewer than compartment_count={count}')
        rng = np.random.default_rng([int(self.spec['build_seed']), 20260920])
        pose_rng = np.random.default_rng([int(self.spec['pose_seed']), 20260920])
        cover_rng = np.random.default_rng([int(self.spec['cover_seed']), 20260921])
        assignment = assign_places(candidates, self.spec, rng, hiding=self._hiding_admissible,
                                   **self.hide_rule(self.spec))
        self.compartments = assignment['compartments']
        self.places = assignment['places']
        ordered = sorted(self.compartments)
        self._closer_ids = ([ordered[i] for i in self.spec['closers'] if i < len(ordered)]
                            if self.APPLY_CLOSERS else [])
        self._build_tray()
        self.add_submit_button(self.spec['placement']['submit'])
        # Counter spots: `open`/`covered` candidates near approachable edges (with standoffs) are drawn first so the
        # hidden world does not depend on how many visible objects there are; covered spots are bound now (the cover
        # is part of the model), open spots at reset (camera-dependent); visible objects then take any interior
        # counter spot clear of the work frame, the candidates, the covers and each other.
        exclude = [self._work_polygon()]
        self._keepouts = self._decor_keepouts()
        self._open_candidates = [s for s in sample_counter_spots(self._records, self._floor, rng, exclude=exclude,
                                                                 edge_only=True, keepouts=self._keepouts,
                                                                 clearance=SPOT_CLEARANCE_M)
                                 if self._counter_spot_admissible(s)]
        anywhere = sample_counter_spots(self._records, self._floor, rng, per_patch=14, include_shelves=False,
                                        exclude=exclude, edge_only=False,
                                        keepouts=self._keepouts, clearance=SPOT_CLEARANCE_M)
        radius = cover_dims(self.spec['cover_size'])['radius']
        self._covered_spots = []
        for key in sorted(self.places):
            if self.places[key]['kind'] != 'covered':
                continue
            spot, options = self._choose_covered_spot(radius, exclude)
            spot = dict(spot, setdown_options=options)
            self._covered_spots.append(spot)
            self._open_candidates = [s for s in self._open_candidates if s is not spot and s['xy'] != spot['xy']]
            self.places[key].update(spot=spot, setdown_options=options)
        visible_spots, taken = [], []

        def clear_of(spot, others):
            return all(np.linalg.norm(np.subtract(spot['xy'], o['xy'])) >= VISIBLE_SPACING_M for o in others)
        for pool_id in self.spec['visible']:
            spot = next((s for s in anywhere if s not in taken and clear_of(s, taken) and clear_of(s, self._open_candidates)
                         and self._clear_of_covers(s['xy'])), None)
            if spot is None:   # small kitchens: accept a spot that crowds out some open candidates
                spot = next((s for s in anywhere if s not in taken and clear_of(s, taken)
                             and self._clear_of_covers(s['xy'], keepout=VISIBLE_SPACING_M + radius, setdown_keepout=0.)), None)
            if spot is None:
                raise ValueError(f'{self.TASK_NAME}: no counter spot left for visible object {pool_id!r} on layout {self.layout_id}')
            taken.append(spot)
            visible_spots.append((pool_id, spot))
        self._open_candidates = [s for s in self._open_candidates
                                 if all(np.linalg.norm(np.subtract(s['xy'], t['xy'])) >= VISIBLE_SPACING_M for t in taken)
                                 and self._clear_of_covers(s['xy'])]
        hidden_ids, open_index = [], 0
        for key in sorted(self.places):
            place = self.places[key]
            pool_id = place['object']
            if place['kind'] == 'open':
                if not self._open_candidates:
                    raise ValueError(f'{self.TASK_NAME}: no counter spot for an open hiding place on layout {self.layout_id}')
                # Provisional pose only (one candidate per open target so they never overlap); replaced at
                # reset by an unseen spot, or by the fallback compartment.
                provisional = self._open_candidates[min(open_index, len(self._open_candidates) - 1)]
                open_index += 1
                place['provisional_spot'] = provisional
                self._add_object(pool_id, provisional['xy'], provisional['top_z'], float(pose_rng.uniform(-np.pi, np.pi)), place['role'])
            elif place['kind'] == 'covered':
                spot = place['spot']
                self._add_object(pool_id, spot['xy'], spot['top_z'], float(pose_rng.uniform(-np.pi, np.pi)), place['role'])
                self._add_cover(key, place, spot, cover_rng)
            else:
                comp = self.compartments[place['compartment']]
                pose = hidden_pose(comp, pool_id, pose_rng, POOL[pool_id])
                place.update(hide_local=pose['hide_local'], hide_world=pose['hide_world'])
                self._add_object(pool_id, pose['xy'], pose['z'], pose['yaw'], place['role'])
            hidden_ids.append(pool_id)
        for pool_id, spot in visible_spots:
            self._add_object(pool_id, spot['xy'], spot['top_z'], float(pose_rng.uniform(-np.pi, np.pi)), 'visible')
        targets = [object_name(t['object']) for t in self.spec['targets']]
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, family='search',
            layout_id=int(self.layout_id), style_id=int(self.style_id),
            target_ids=targets, targets=deepcopy(self.spec['targets']),
            compartments={cid: {k: c[k] for k in ('kind', 'fixture', 'cls', 'joints', 'joint_ranges', 'handles', 'standoff',
                                                  'region', 'front_normal', 'pos', 'rot', 'size', 'open_side', 'handle_side')}
                          for cid, c in self.compartments.items()},
            compartment_mix=assignment['mix'], places=deepcopy(self.places),
            hidden_object_ids=[object_name(p) for p in hidden_ids],
            visible_objects={object_name(p): s for p, s in visible_spots},
            candidate_count=len(candidates), candidate_ids=[c['id'] for c in candidates],
            tray=deepcopy(self.tray), floor_bounds=self._floor, pool=deepcopy(POOL),
            open_candidate_count=len(self._open_candidates),
            covers=deepcopy(self.covers), closer_ids=list(self._closer_ids),
            scoring='First physical Submit freezes the score: every target upright inside the tray square with a real '
                    'upward support contact, nothing else on the tray, all objects released and settled.')
        self.asset_evidence['layout'] = {
            'layout_id': int(self.layout_id), 'style_id': int(self.style_id),
            'pool_assets': {k: {'source': str(ASSET_ROOT / v['asset'] / 'model.xml'),
                                'xml_sha256': hashlib.sha256((ASSET_ROOT / v['asset'] / 'model.xml').read_bytes()).hexdigest()}
                            for k, v in POOL.items()},
            'selection_rules': 'roboquest.search.layout',
            'disabled_fixtures': 'all other drawer/door/appliance-door joints locked, their handle geoms hidden',
            'closers': 'joint springs on the compartments in task_spec["closer_ids"], applied to the compiled model'}

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._disable_other_fixtures()

    def _anchor_base_for_task(self):
        """The base's stance gate must judge the start pose this task sets, so the re-anchoring runs
        inside the base ``_load_model``, before the gate, not after it."""
        self._start_in_front_of_frame()

    def _start_in_front_of_frame(self):
        """Start the robot facing the task frame, START_EDGE_GAP_M from the surface's front edge.

        RoboCasa parks the robot at the work fixture's lateral centre; on counters
        whose largest free region is off-centre the tray would then sit beside the
        robot, outside both fixed cameras. Same edge distance RoboCasa uses.
        """
        front_y = -self.FOOTPRINT[1] / 2 - self.FOOTPRINT_FRONT_MARGIN
        anchor = self.frame_to_world([0., front_y - START_EDGE_GAP_M, 0.])
        yaw = float(self.work['yaw'] + math.pi / 2)
        self.init_robot_base_pos_anchor = np.array([float(anchor[0]), float(anchor[1]), float(self.init_robot_base_pos_anchor[2])])
        self.init_robot_base_ori_anchor = np.array([0., 0., yaw])
        self.robots[0].robot_model.set_base_ori(self.init_robot_base_ori_anchor)
        self.task_spec['start_pose'] = dict(xy=[float(anchor[0]), float(anchor[1])], yaw=yaw,
                                            note='base frame in front of the task frame, RoboCasa edge distance')

    def _disable_other_fixtures(self):
        """Lock every door/drawer joint outside the chosen compartments and hide those handles.

        Locked joints keep a 0.1 mm range around their closed value (MuJoCo needs
        range[0] < range[1]); hidden handle geoms get alpha 0, group 3 (never
        rendered) and no collision, so nothing invisible can be bumped. Idempotent:
        RoboCasa may run ``_load_model`` more than once.
        """
        keep_joints = {j for c in self.compartments.values() for j in c['joints']}
        door_joints = self._fixture_door_joints()
        locked, prefixes, fixtures, every_handle = [], [], [], []
        for name, joints in sorted(door_joints.items()):
            if name in self.compartments:
                continue
            lock = [j for j in joints if j not in keep_joints]
            if not lock:
                continue
            fixtures.append(dict(fixture=name, cls=type(self.fixtures[name]).__name__, joints=lock))
            locked += lock
            prefixes += [f'{name}_{pattern}' for pattern in self.HIDDEN_HANDLE_PATTERNS]
            if any(k in type(self.fixtures[name]).__name__ for k in self.HIDE_ALL_HANDLES_OF):
                every_handle.append(f'{name}_')
        locked = set(locked)
        for joint in self.model.root.iter('joint'):
            if joint.get('name') in locked:
                joint.set('limited', 'true')
                joint.set('range', '-0.0001 0.0001')
        hidden = []
        for geom in self.model.root.iter('geom'):
            gname = geom.get('name') or ''
            if gname and (any(gname.startswith(p) for p in prefixes)
                          or ('handle' in gname.lower() and any(gname.startswith(p) for p in every_handle))):
                geom.set('rgba', '0 0 0 0')
                geom.set('group', '3')
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                hidden.append(gname)
        for row in fixtures:
            row['handle_geoms_hidden'] = sum(1 for g in hidden if g.startswith(row['fixture'] + '_'))
        self._disabled = dict(locked_joints=sorted(locked), hidden_handle_geoms=sorted(hidden), fixtures=fixtures,
                              handles_not_found=[r['fixture'] for r in fixtures if r['handle_geoms_hidden'] == 0])
        self.task_spec['disabled_fixtures'] = deepcopy(self._disabled)

    # ---- runtime --------------------------------------------------------------
    def setup_task_references(self):
        m = self.sim.model
        self._joint_addresses = {j: m.get_joint_qpos_addr(j) for c in self.compartments.values() for j in c['joints']}
        body_prefix = {name: name + '_' for name in self.objects}
        self._object_geoms = {name: [] for name in self.objects}
        for gid in range(m.ngeom):
            body = m.body_id2name(int(m.geom_bodyid[gid])) or ''
            for name, prefix in body_prefix.items():
                if body == name or body.startswith(prefix):
                    self._object_geoms[name].append(gid)
        widths = getattr(self, 'camera_widths', None) or [512]
        heights = getattr(self, 'camera_heights', None) or [512]
        self._camera_size = (int(widths[0]), int(heights[0]))
        self._closers = self._apply_closers()

    def _apply_closers(self):
        """Passive closers (C5) on the compiled model: ``closers.apply_closer`` sets the spring rest past the
        closed limit and sizes stiffness and damping for ``CLOSER_TIME_S`` from fully open against RoboCasa's own
        joint damping; on a drawer the rest goes far past the limit and `drawer_closer_design` rewrites the pair
        for the constant-speed return (module docstring above `CLOSER_TIME_S`). Nothing is written to the MJCF;
        re-applied on every model load."""
        m = self.sim.model
        applied = {}
        for cid in self._closer_ids:
            comp = self.compartments[cid]
            kind = comp['kind']
            rows = []
            for joint in comp['joints']:
                jid = m.joint_name2id(joint)
                dadr = int(m.jnt_dofadr[jid])
                lo, hi = (float(v) for v in m.jnt_range[jid])
                open_qpos = hi if abs(hi) >= abs(lo) else lo
                travel = abs(open_qpos - (lo if open_qpos == hi else hi))
                tol = CLOSED_FRACTION * travel
                rest_past = CLOSER_REST_PAST[kind]
                design = None
                if kind == 'drawer':
                    rest_past, stiffness, damping = drawer_closer_design(travel, float(m.dof_damping[dadr]),
                                                                         CLOSER_TIME_S[kind], tol)
                    design = dict(stiffness=stiffness, damping=damping, frictionloss=CLOSER_DRAWER_FRICTION)
                rest = -math.copysign(rest_past, open_qpos)
                params = apply_closer(self.sim, joint, rest, close_time_s=CLOSER_TIME_S[kind], open_qpos=open_qpos)
                if design is not None:
                    m.jnt_stiffness[jid] = design['stiffness']
                    m.dof_damping[dadr] = design['damping']
                    m.dof_frictionloss[dadr] = design['frictionloss']
                    params.update(stiffness=round(design['stiffness'], 5), damping=round(design['damping'], 4),
                                  frictionloss=design['frictionloss'], design='preloaded')
                # The closing time is measured against the pull, not against the travel to the spring rest.
                params.update(kind=kind, compartment=cid, travel=travel, closed_tol=tol, rest_past=rest_past)
                rows.append(params)
            applied[cid] = rows
        self.task_spec['closers'] = deepcopy(applied)
        return applied

    def _joint_values(self):
        return {j: float(self.sim.data.qpos[adr]) for j, adr in self._joint_addresses.items()}

    def base_pose(self):
        pos, rot = self.robots[0].part_controllers['base'].get_base_pose()
        return np.asarray(pos)[:2].copy(), float(math.atan2(rot[1, 0], rot[0, 0]))

    def _set_object_pose(self, name, xy, support_z, yaw):
        obj = self.objects[name]
        z = float(support_z) - float(obj.bottom_offset[2]) + .002
        q = Rotation.from_euler('z', yaw).as_quat()
        pose = [float(xy[0]), float(xy[1]), z, float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        self._poses[name] = pose
        self.sim.data.set_joint_qpos(obj.joints[0], np.asarray(pose))
        self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.task_spec['objects'][name]['yaw'] = float(yaw)

    def _choose_open_spots(self):
        """Bind each `open` target to a counter spot no policy camera sees from the start pose.

        Runs after the robot is at its start pose (cameras compiled and posed). Prefers
        spots behind, then beside, the robot. Without any unseen spot the target goes
        to its reserved fallback compartment and ``open_fallback`` is recorded.
        """
        open_keys = [k for k in sorted(self.places) if self.places[k]['kind'] == 'open']
        if not open_keys:
            return
        if self._open_choice:
            for key, choice in self._open_choice.items():
                self._set_object_pose(*choice)
                # A hard reset rebuilds `places`: put back the binding the first reset recorded (the spot with
                # its standoff, or the fallback compartment), which the certificate and the gates read.
                self.places[key].update(deepcopy(self._open_bindings.get(key, {})))
            self.task_spec['places'] = deepcopy(self.places)
            self.task_spec['open_fallback'] = any(p.get('open_fallback') for p in self.places.values())
            return
        frusta = camera_frusta(self.sim, CAMERAS, aspect=self._camera_size[0] / self._camera_size[1])
        base_xy, base_yaw = self.base_pose()
        pose_rng = np.random.default_rng([int(self.spec['pose_seed']), 20260921])
        chosen_spots = []
        for key in open_keys:
            place = self.places[key]
            pool_id = place['object']
            name = object_name(pool_id)
            half = np.asarray(POOL[pool_id]['size_xyz_m'], float) / 2
            candidates = [s for s in self._open_candidates
                          if all(np.linalg.norm(np.subtract(s['xy'], c['xy'])) >= VISIBLE_SPACING_M for c in chosen_spots)]
            ranked, report = rank_unseen_spots(candidates, frusta, base_xy, base_yaw, half)
            yaw = float(pose_rng.uniform(-np.pi, np.pi))
            if ranked:
                spot = ranked[0]
                chosen_spots.append(spot)
                binding = dict(spot=spot, open_fallback=False, kind_effective='open', visibility_report=report)
                place.update(binding)
                choice = (name, spot['xy'], spot['top_z'], yaw)
            else:
                fallback = place.get('fallback_compartment')
                if fallback is None:
                    raise ValueError(f'{self.TASK_NAME}: no unseen open spot for {pool_id!r} on layout {self.layout_id} '
                                     'and no fallback compartment fits it')
                comp = self.compartments[fallback]
                pose = hidden_pose(comp, pool_id, pose_rng, POOL[pool_id])
                binding = dict(compartment=fallback, open_fallback=True, kind_effective=comp['kind'],
                               hide_local=pose['hide_local'], hide_world=pose['hide_world'], visibility_report=report)
                place.update(binding)
                choice = (name, pose['xy'], pose['z'], pose['yaw'])
            self._open_choice[key] = choice
            self._open_bindings[key] = deepcopy(binding)
            self._set_object_pose(*choice)
        self.task_spec['places'] = deepcopy(self.places)
        self.task_spec['open_fallback'] = any(p.get('open_fallback') for p in self.places.values())

    def _reset_internal(self):
        super()._reset_internal()
        for cid, comp in self.compartments.items():
            self.get_fixture(cid).close_door(self)
            for joint in comp['joints']:
                self.sim.data.set_joint_qvel(joint, 0.)
        self.sim.forward()
        self._choose_open_spots()
        self.sim.forward()
        self._search = SearchDiagnostics(self.compartments, self.target_compartments())

    def target_compartments(self):
        return [p.get('compartment') for p in self.places.values() if p['role'] == 'target' and p.get('compartment')]

    def _post_action(self, action):
        if self._search is not None:
            self._search.update(self.timestep, self._joint_values())
        return super()._post_action(action)

    def _score_spec(self):
        return dict(objects=self.task_spec['objects'], target_ids=self.task_spec['target_ids'], tray=self.tray)

    def _snapshot(self):
        return VisionKitchenScene.evaluator_snapshot(self, self.timestep)

    def _current_score(self, states=None):
        return terminal_score(self._score_spec(), self._snapshot())

    def task_snapshot(self):
        joints = self._joint_values()
        return dict(compartments={cid: {'joints': {j: joints[j] for j in comp['joints']}}
                                  for cid, comp in self.compartments.items()},
                    search=self._search.summary() if self._search is not None else None)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['roboquest']['search'] = dict(
            places=deepcopy(self.places), compartments=sorted(self.compartments), closers=list(self._closer_ids),
            covers={n: dict(target=c['target'], place=c['place'], kind=c['size']) for n, c in self.covers.items()},
            open_fallback=any(p.get('open_fallback') for p in self.places.values()),
            disabled=dict(locked_joints=len(self._disabled['locked_joints']),
                          hidden_handle_geoms=len(self._disabled['hidden_handle_geoms']),
                          handles_not_found=list(self._disabled.get('handles_not_found', []))))
        return meta

    # ---- protocol hooks (C1) --------------------------------------------------
    def _body_position(self, name):
        return np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]], float)

    def _slide_offset_local(self, comp):
        """A drawer's interior travels with its slide joint (fixture-local y); doors leave it in place."""
        if comp['kind'] != 'drawer':
            return (0., 0., 0.)
        q = float(self.sim.data.qpos[self._joint_addresses[comp['joints'][0]]])
        return (0., q, 0.)

    def _inside_compartment(self, name, cid, margin=.03):
        comp = self.compartments[cid]
        local = local_from_world(comp['pos'], comp['rot'], self._body_position(name))
        return inside_interior(local, comp['region'], self._slide_offset_local(comp), margin=margin)

    def _out_of_hiding(self, key, place, snapshot):
        """Spec 1.4: the target left its hiding place (compartment interior, cover, or unseen counter spot)."""
        name = object_name(place['object'])
        if place.get('compartment'):
            return not self._inside_compartment(name, place['compartment'])
        if place.get('cover'):
            cover = self.covers[place['cover']]
            cover_pos = snapshot['objects'][place['cover']]['position']
            return not under_cover(snapshot['objects'][name]['position'], cover_pos, cover['radius'], cover['height'])
        spot = place.get('spot')
        if spot is None:
            return False
        position = np.asarray(snapshot['objects'][name]['position'], float)
        return (np.linalg.norm(position[:2] - np.asarray(spot['xy'], float)) > PROGRESS_OUT_OF_SPOT_M
                or position[2] > float(spot['top_z']) + .10)

    def task_progress(self):
        """Per target: 0.5 once out of its hiding place, 1 released and upright on the tray; mean; 1 on success."""
        snapshot = self._snapshot()
        score = terminal_score(self._score_spec(), snapshot)
        done, out = [], []
        for key in sorted(self.places):
            place = self.places[key]
            if place['role'] != 'target':
                continue
            name = object_name(place['object'])
            state = snapshot['objects'].get(name, {})
            per = score['per_target'].get(name)
            done.append(bool(per and all(per.values()) and not state.get('grasped') and not state.get('robot_contact')))
            out.append(bool(self._out_of_hiding(key, place, snapshot)))
        return float(target_progress(done, out, bool(score['success'])))

    def collateral_checks(self):
        """``autoclose_closed``: a closer compartment that was opened and has shut again with a target inside
        (cost: one per target). Cheap state reads only."""
        inside = []
        if self._search is not None and self._closer_ids:
            joints = self._joint_values()
            states = compartment_states({cid: self.compartments[cid] for cid in self._closer_ids}, joints)
            opened = set(self._search.opened_order)
            for cid in self._closer_ids:
                if cid not in opened or states[cid]['opened']:
                    continue
                for place in self.places.values():
                    if place['role'] == 'target' and place.get('compartment') == cid:
                        name = object_name(place['object'])
                        if self._inside_compartment(name, cid):
                            inside.append(name)
        return {'autoclose_closed': dict(bad=bool(inside), object=','.join(inside), cost=float(len(inside)))}

    # ---- teleport certificate (C3) ------------------------------------------------
    def teleport_solution(self):
        """Bring the scene to a solved state without the robot (contract C3).

        Opens each target's compartment by setting its joints, moves covers to their set-down spots, moves the
        targets onto the tray slots, closes the compartments again (an opened drawer under the work top would
        otherwise stand inside the robot base during the settle), and settles with zero-action steps. Never
        moves the robot, never presses Submit, never raises.
        """
        steps = []
        started = time.monotonic()

        def record(step, ok, detail=None):
            steps.append(dict(step=step, ok=bool(ok), detail=detail))
            return ok
        data = self.sim.data
        targets = [(k, p) for k, p in sorted(self.places.items()) if p['role'] == 'target']
        slots = TRAY_SLOTS.get(len(targets)) or [(0., 0.)] * len(targets)
        opened = {}
        for index, (key, place) in enumerate(targets):
            name = object_name(place['object'])
            try:
                cid = place.get('compartment')
                if cid:
                    comp = self.compartments[cid]
                    values = {}
                    for joint in comp['joints']:
                        jid = self.sim.model.joint_name2id(joint)
                        lo, hi = (float(v) for v in self.sim.model.jnt_range[jid])
                        open_qpos = hi if abs(hi) >= abs(lo) else lo
                        data.set_joint_qpos(joint, open_qpos)
                        data.set_joint_qvel(joint, 0.)
                        values[joint] = open_qpos
                    self.sim.forward()
                    opened[cid] = values
                    record(f'open {cid}', True, dict(joints=values, kind=comp['kind'], closer=cid in self._closer_ids))
                elif place.get('cover'):
                    cover = self.covers[place['cover']]
                    options = cover.get('setdown_options') or []
                    if not options:
                        record(f"lift {place['cover']}", False, 'no set-down option recorded for this cover')
                    else:
                        option = options[0]
                        self._set_object_pose(place['cover'], option['xy'], option['top_z'], float(cover['yaw']))
                        self.sim.forward()
                        record(f"lift {place['cover']}", True, dict(setdown_xy=option['xy'], side=option['side']))
                else:
                    record(f'expose {name}', True, dict(kind=place.get('kind_effective', place['kind']), spot=place.get('spot', {}).get('xy')))
                slot = slots[index]
                xy = self.frame_to_world([self.tray['centre_local'][0] + slot[0], self.tray['centre_local'][1] + slot[1], 0.])[:2]
                self._set_object_pose(name, xy, self.tray['top_z'], float(self.work['yaw']))
                self.sim.forward()
                record(f'place {name} on the tray', True, dict(slot=list(slot), xy=[float(v) for v in xy]))
            except Exception as exc:  # noqa: BLE001
                record(f'target {key}', False, f'{type(exc).__name__}: {exc}')
        for cid, values in opened.items():
            try:
                for joint in values:
                    data.set_joint_qpos(joint, 0.)
                    data.set_joint_qvel(joint, 0.)
                self.sim.forward()
                record(f'reclose {cid}', True, 'closed again before settling (a drawer under the work top would stand in the robot base)')
            except Exception as exc:  # noqa: BLE001
                record(f'reclose {cid}', False, f'{type(exc).__name__}: {exc}')
        try:
            zero = np.zeros(self.action_dim)
            for _ in range(SETTLE_TICKS):
                self.step(zero)
            score = self._current_score()
            record('settle', score['success'], dict(ticks=SETTLE_TICKS, errors=score['errors'], score=score['score'],
                                                    progress=self.task_progress(), seconds=round(time.monotonic() - started, 2)))
        except Exception as exc:  # noqa: BLE001
            record('settle', False, f'{type(exc).__name__}: {exc}')
        return steps

    # ---- gates ----------------------------------------------------------------
    def disabled_fixture_report(self):
        """Compiled-model facts for the build gate: which door joints move, which are locked, hidden handles."""
        m = self.sim.model
        movable, locked = [], []
        for name, joints in self._fixture_door_joints().items():
            for joint in joints:
                jid = m.joint_name2id(joint)
                lo, hi = m.jnt_range[jid]
                (movable if hi - lo > 1e-3 else locked).append(joint)
        fixtures_movable = sorted({n for n, js in self._fixture_door_joints().items() if any(j in movable for j in js)})
        hidden = self._disabled['hidden_handle_geoms']
        visible_hidden = [g for g in hidden if m.geom_rgba[m.geom_name2id(g)][3] > 0 or m.geom_group[m.geom_name2id(g)] < 3
                          or m.geom_contype[m.geom_name2id(g)] != 0 or m.geom_conaffinity[m.geom_name2id(g)] != 0]
        return dict(movable_joints=sorted(movable), locked_joints=sorted(locked), fixtures_with_movable_doors=fixtures_movable,
                    chosen_compartments=sorted(self.compartments), hidden_handle_geoms=len(hidden),
                    hidden_handles_still_visible=visible_hidden,
                    closers={cid: [dict(joint=r['joint'], stiffness=r['stiffness'], damping=r['damping'], design=r['design'])
                                   for r in rows] for cid, rows in self._closers.items()})

    def hidden_state_check(self):
        """Gate G3: hidden objects are pixel-invisible at reset; tray and Submit are visible in a fixed camera.

        Renders the three policy cameras with every hidden object's geoms repainted in one opaque colour and
        again in another (alpha untouched, so MuJoCo's transparent-geom draw order does not change, unlike an
        alpha toggle) and requires pixel-identical images (fail-closed, with a render-determinism check). Covered
        targets are hidden objects too, so the cloche must occlude them completely. Tray and Submit must project
        inside at least one fixed camera image *and* change pixels there when toggled.
        """
        if getattr(self.sim, '_render_context_offscreen', None) is None:
            raise RuntimeError('hidden_state_check needs an env built with render=True')
        width, height = self._camera_size
        m = self.sim.model

        def render_all():
            return {cam: np.array(self.sim.render(width=width, height=height, camera_name=cam), copy=True) for cam in CAMERAS}

        def toggled(gids):
            saved = m.geom_rgba[gids].copy()
            m.geom_rgba[gids, 3] = 0.
            try:
                return render_all()
            finally:
                m.geom_rgba[gids] = saved

        def repainted(gids, rgb):
            saved = m.geom_rgba[gids].copy()
            m.geom_rgba[gids, :3] = rgb
            try:
                return render_all()
            finally:
                m.geom_rgba[gids] = saved

        base, again = render_all(), render_all()
        deterministic = {cam: bool(np.array_equal(base[cam], again[cam])) for cam in CAMERAS}
        hidden_names = list(self.task_spec['hidden_object_ids'])
        hidden_gids = np.array([g for n in hidden_names for g in self._object_geoms[n] if m.geom_rgba[g, 3] > 0.], int)
        red, green = repainted(hidden_gids, (1., 0., 0.)), repainted(hidden_gids, (0., 1., 0.))
        identical = {cam: bool(np.array_equal(red[cam], green[cam])) for cam in CAMERAS}
        changed = {cam: int(np.any(red[cam] != green[cam], axis=-1).sum()) for cam in CAMERAS}
        fixed = list(CAMERAS[:2])
        frusta = {f['name']: f for f in camera_frusta(self.sim, fixed, aspect=width / height)}

        def inside(cam, point):
            uv = project(frusta[cam], point, width, height)
            return uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height

        tray_points = [np.r_[c[:2], self.tray['top_z']] for c in self.tray['corners_world']]
        button = np.asarray(self.task_spec['submit_button']['position'], float)
        tray_gids = np.array([m.geom_name2id(g) for g in self.tray['mark_geoms']], int)
        submit_gids = np.array([g for g in range(m.ngeom) if (m.geom_id2name(g) or '').startswith(self.submit_prefix + '_')], int)
        without_tray, without_submit = toggled(tray_gids), toggled(submit_gids)
        tray_visible = {cam: bool(all(inside(cam, p) for p in tray_points) and not np.array_equal(base[cam], without_tray[cam]))
                        for cam in fixed}
        submit_visible = {cam: bool(inside(cam, button) and not np.array_equal(base[cam], without_submit[cam])) for cam in fixed}
        covers = {}
        for name, cover in self.covers.items():
            visual = [g for g in self._object_geoms[name] if m.geom_group[g] == 1]
            covers[name] = dict(target=cover['target'], opaque=bool(visual) and bool(all(m.geom_rgba[g, 3] >= 1. for g in visual)),
                                target_under_cover=bool(under_cover(self._body_position(cover['target']), self._body_position(name),
                                                                    cover['radius'], cover['height'])))
        covers_ok = all(c['opaque'] and c['target_under_cover'] for c in covers.values())
        passed = (all(deterministic.values()) and all(identical.values()) and covers_ok
                  and any(tray_visible.values()) and any(submit_visible.values()))
        return dict(passed=bool(passed), render_deterministic=deterministic, hidden_objects=hidden_names,
                    hidden_pixel_identical=identical, hidden_changed_pixels=changed, method='opaque red/green repaint',
                    tray_visible=tray_visible, submit_visible=submit_visible, covers=covers,
                    open_places={k: dict(band=p.get('spot', {}).get('band'), open_fallback=p.get('open_fallback'))
                                 for k, p in self.places.items() if p['kind'] == 'open'},
                    images=base)


class SearchRoomDrawerType(SearchRoom):
    """Generator of the drawer-type structure variant: set_size 3, the targets as one of
    ``DRAWER_TYPE_CLASSES`` and every decoy drawer-capable, recorded in the spec so the plain
    :class:`SearchRoom` (the manifest's class) builds, validates and scores it. Only ``sample_spec`` differs;
    the task name, versions and id scheme are the search room's."""

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        set_size, count = int(factors['set_size']), int(factors['compartment_count'])
        if set_size != 3:
            raise ValueError(f'the {DRAWER_TYPE} structure is set_size 3 only')
        kinds = list(DRAWER_TYPE_CLASSES[int(structure.integers(len(DRAWER_TYPE_CLASSES)))])
        kinds = [kinds[int(i)] for i in structure.permutation(len(kinds))]
        # A drawer place (target or decoy) takes an object that fits the kitchen's tallest high drawer (the
        # 0.118 m drawers of most kitchens hold mugs and tins, not cans; the bottle never hides in a drawer);
        # a covered place takes any object the cloche covers (mugs, cans, tins).
        cap = high_drawer_capacity(int(factors['kitchen_layout']))
        tallest = max(cap['drawers']) / 1000. if isinstance(cap, dict) and cap['drawers'] else None

        def drawer_ok(o):
            return 'drawer' in POOL[o]['kinds'] and (tallest is None or fits(o, tallest))
        targets = []
        for kind in kinds:
            able = sorted(o for o in POOL if o not in targets and (drawer_ok(o) if kind == 'drawer'
                                                                    else 'covered' in POOL[o]['kinds']))
            targets.append(able[int(structure.integers(len(able)))])
        others = [o for o in sorted(POOL) if o not in targets]
        candidates = [o for o in others if drawer_ok(o) and any(confusable(o, t) for t in targets)]
        if not candidates:
            raise ValueError(f'no decoy fits a high drawer and shares a category or colour with {targets}')
        n_hidden = min(hidden_distractor_count(count, kinds), len(candidates))
        hidden = sorted(candidates[int(i)] for i in structure.choice(len(candidates), n_hidden, replace=False))
        visible = [o for o in others if o not in hidden]
        build_seed = int(structure.integers(0, 2**31 - 1))
        pose_seed = int(poses.integers(0, 2**31 - 1))
        closer_count = int(CLOSER_COUNT[int(structure.integers(len(CLOSER_COUNT)))])
        closers = sorted(int(i) for i in structure.choice(count, closer_count, replace=False))
        cover_seed = int(materials.integers(0, 2**31 - 1))
        spec = dict(
            set_size=set_size, compartment_count=count, layout_id=int(factors['kitchen_layout']),
            targets=[dict(object=t, kind=k) for t, k in zip(targets, kinds)],
            hidden_distractors=hidden, visible=visible,
            build_seed=build_seed, pose_seed=pose_seed, closers=closers,
            cover_size=COVER_SIZE, cover_seed=cover_seed,
            placement=dict(tray=list(TRAY_LOCAL), submit=list(SUBMIT_LOCAL)), tray_half_m=TRAY_HALF_M,
            structure_variant=DRAWER_TYPE, hide_min_floor_m=DRAWER_TYPE_MIN_FLOOR_M,
            counter_front_max_fraction=DRAWER_TYPE_COUNTER_FRONT_MAX)
        problems = cls.spec_capacity_problems(spec)
        if problems:
            raise ValueError('; '.join(problems))
        return spec
