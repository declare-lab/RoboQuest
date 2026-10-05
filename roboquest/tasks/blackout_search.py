"""Blackout Search (v1.1 contract section 3, torch-only): search_room with the lights out.

The kitchen is unlit and stays unlit. At reset the arena light is off and MuJoCo's headlight is disabled;
there is no wall switch and no timer (no ``switch`` / ``switch_timed`` / ``lamp_only`` lighting
axis). A lantern (a real hurricane lantern, ``blackout/lamp.py``)
stands lit within 1.2 m of the robot and lights about a metre around itself, so the robot sees the
counter it starts at; beyond that the only things any policy camera can see are the emissive markers:
the lantern's glowing glass and the Submit cap. The robot finds the targets by carrying the lantern, and
because the gripper that holds the lantern cannot also open a drawer, it must set the lantern down beside
whatever it works on.

The grid is the search room's: ``set_size`` 2 or 3 targets among ``compartment_count`` 4, 6 or 8 openable
compartments, with the search room's hold-out scheme (one key per set size, so every cell has eval rows;
``blackout/structure.py``). Targets and decoys hide only in drawers whose interior floor is at least
``HIDING_FLOOR_MIN_M`` (0.40 m), on an open counter spot at least ``LAMP_TARGET_MIN_M`` from the lantern's
reset spot (the darkness hides it), or under a cloche (no door places, no bottom drawers; the
reach analysis of 2026-09-24). The rule is a hard filter over the kitchen's compartments at build
(``_hiding_admissible``), recorded in the spec and asserted by ``validate_spec`` and G3.

Everything not about light is inherited from :class:`SearchRoom`: compartment discovery, the pool and its
confusable distractors, the work frame, the tray, Submit, the disabled fixtures, scoring; not the passive
closers (none here). This module adds the lantern, the progress weights, the darkness gate
(G3) and the lantern certificate (G3.5). The privileged state -- where the targets are -- reaches the
evaluator trace only, never an observation and never the goal.
"""
from copy import deepcopy
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.tasks.search_room import COVER_SIZE, SETTLE_TICKS, SearchRoom, SPOT_CLEARANCE_M, TRAY_HALF_M, TRAY_LOCAL, TRAY_SLOTS, SUBMIT_LOCAL, START_EDGE_GAP_M, VISIBLE_SPACING_M, object_name
from roboquest.blackout.darkness import room_light_ids, set_room_light
from roboquest.blackout.lamp import COLUMN_XY, FILL, GLOW_MATERIAL, HANDLE_GRASP_Z, RADIUS as LAMP_RADIUS, RING_Z, SPOT, aim_yaw, build_lamp
from roboquest.blackout.structure import COMPARTMENT_COUNTS, COUNTER_BAND_RULE, COUNTER_FRONT_FRACTION, FACTORS, GOAL_COUNT, GOAL_NOUN, GOAL_TEMPLATE, HIDING_FLOOR_MIN_M, HIDING_KINDS, HIDING_RULE, HOLDOUT_KEYS, SET_SIZES, admissible_hiding_place, capacity_problems, decoy_capable, draw_structure, hidden_distractor_count, hiding_kinds_of, high_drawers, progress_weights
from roboquest import headroom as H
from roboquest.search.covers import COVER_SIZES
from roboquest.search.layout import FLOOR_MARGIN_M, _blocked, _clearance, counter_band, patch_for_spot, point_in_floor, sample_counter_spots
from roboquest.search.objects import POOL, confusable, describe_targets
from roboquest.search.rules import PROGRESS_CAP, terminal_score
from roboquest.search.visibility import camera_frusta, project
from roboquest.harness.contract import CAMERAS

CONTRACT_VERSION = 'blackout-search-kitchen-v2'   # v2: torch-only, the search room's grid, the drawer rule
GENERATOR_VERSION = 'v3'                           # v3: set_size x compartment_count, no
                                                   # lighting factor, hiding places filtered by floor height
# Passive closers: none. Fresh specs record closers=[]; nothing else is accepted since v3.
CLOSER_COUNT = (0,)
LAMP_NAME = 'work_lamp'
WRIST_CAMERA = CAMERAS[2]      # the eye-in-hand camera: exempt from the darkness sanity cap
NIGHT_SKY_TEXTURE = 'rq_night_sky'
LAMP_START_DIST_M = 1.2        # the lamp stands within this of the start stance (contract)
LAMP_SPOT_PER_PATCH = 40       # lamp spot candidates sampled per counter patch
LAMP_SPACING_M = .25           # ... and this far from every object, hiding spot and cover set-down spot
LAMP_CLEARANCE_M = LAMP_RADIUS + .02
LAMP_LIFT_M = .06              # body centre this far above its resting height counts as picked up
LAMP_STILL_TICKS = 10          # ... and this many still ticks after that counts as set down
LAMP_STILL_SPEED = .01
LAMP_TIP_COS = .5              # cos 60 degrees: the lamp axis is tipped past this while released
LAMP_FLOOR_Z_M = .30           # the lamp body below this is on the floor, not on any counter
# G3 thresholds (contract's rendering note). The glow counts are pixels at the gate's render size,
# scaled from 512 x 512; measured values go in the handoff.
# The lantern lights the robot's own counter at reset (policy-camera means of 16-41/255
# measured on two kitchens), so the whole-frame bar is a sanity cap that only says "the room light is
# off" (the room light gives 100-150); what hides a target is the pixel test and the keep-out below.
DARK_TARGET_MAX_PX = 8         # target pixels tolerated in the dark per camera at 512 x 512,
                               # the lantern lights compartment fronts and 1-4 px of a target show through drawer
                               # gaps (re-gate of 2026-09-24); the room-light check keeps the family's zero rule
DARK_MEAN_MAX = 60.            # mean luminance outside the emissive markers, per agentview camera (sanity cap:
                               # the room light is off; the wrist camera is exempt, it looks into the lantern's
                               # pool from 0.6 m and read 62-76 on two rows with the room dark)
LAMP_TARGET_MIN_M = 2.0        # an open-counter target keeps this far from the lantern's reset spot: at the
                               # lantern's 1/(0.5 + 2.5 d^2) falloff that is under a tenth of its light
GLOW_MIN_PX = 40               # the lamp glass, best policy camera
LAMP_WRIST_M = .108            # the wrist (link 7) tops this far above the end-effector site with the hand down
                               # (measured on a development kitchen: the wrist against a wall unit's underside
                               # at 1.42 m with the hand at 1.312 m)
LAMP_PINCH_LIFT_M = .10        # the oracle's pre-grasp point, lift and set-down lift above the bail
                               # (oracles.blackout_search LAMP_ABOVE_M / LAMP_LIFT_M / SETDOWN_LIFT_M)
LAMP_HEADROOM_M = round(HANDLE_GRASP_Z + LAMP_PINCH_LIFT_M + LAMP_WRIST_M + H.HEADROOM_MARGIN_M, 4)
                               # clear air over a lamp spot or a counter set-down spot for the pinch, the lift
                               # and the set-down: 0.483 m, under the 0.50 m of a standard wall unit
SUBMIT_GLOW_MIN_PX = 40
# The lit-place measurement of the certificate:
# an absolute floor that only says "there is light on it" and the contrast ratio that carries the claim;
# both are recorded per target, neither is fatal (LIT_STEP).
LIT_INTERIOR_MIN = 2.          # mean luminance of the lit place (certificate)
LIT_TARGET_MIN_PX = 40         # target pixels under the lamp's light (certificate)
SETDOWN_RANGE_M = (.40, .70)   # the certificate sets the lamp down this far from the place
SETDOWN_FALLBACK_M = (.30, 1.80)   # ... and this far when the sampled spots miss the contract band
SETDOWN_PER_PATCH = 40             # spots sampled per counter patch when looking for one
SETDOWN_TRIES = 8                  # ... and this many are rendered before the best one is kept
SETDOWN_GRID_M = .25               # candidates closer together than this are the same candidate
LIT_PAD_MIN_PX = 6                 # the lit region is the target's bounding box padded by at least this
LIT_STEP = 'the lamp lights the place (measured, not fatal)'
LIT_CONTRAST_X = 8.                # ... and this much brighter than before the lamp was aimed (measured 38-89)
SUBMIT_CAP_RGBA = '.78 .06 .04 1'


class BlackoutSearch(SearchRoom):
    TASK_NAME = 'blackout_search'
    CONTRACT_VERSION = CONTRACT_VERSION
    GENERATOR_VERSION = GENERATOR_VERSION
    FACTORS = FACTORS
    HOLDOUT_KEYS = HOLDOUT_KEYS
    APPLY_CLOSERS = False      # no passive closers in the dark search
    # The locked fridge's handles hide with the cabinets': RoboCasa names them
    # fridge_door_handle_* and freezer_door_handle_*, outside the pattern the search task hides.
    HIDDEN_HANDLE_PATTERNS = SearchRoom.HIDDEN_HANDLE_PATTERNS + ('fridge_door_handle_', 'freezer_door_handle_')
    HIDE_ALL_HANDLES_OF = ('Fridge',)       # the patterns above miss RoboCasa's numbered fridge handles

    # ---- generator protocol (CPU only) --------------------------------------
    @classmethod
    def goal(cls, spec):
        targets = describe_targets([t['object'] for t in spec['targets']])
        return cls.frame_goal(spec, GOAL_NOUN, GOAL_COUNT, GOAL_TEMPLATE.format(targets=targets), clutter=True,
                              scope=False)

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        set_size, count = int(factors['set_size']), int(factors['compartment_count'])
        if set_size not in SET_SIZES or count not in COMPARTMENT_COUNTS:
            raise ValueError(f'unknown cell set_size {set_size!r} x compartment_count {count!r}')
        high = high_drawers(int(factors['kitchen_layout']))
        targets, hidden, visible = draw_structure(structure, set_size, count,
                                                  high=high if isinstance(high, dict) else None)
        build_seed = int(structure.integers(0, 2 ** 31 - 1))
        pose_seed = int(poses.integers(0, 2 ** 31 - 1))
        lamp_seed = int(materials.integers(0, 2 ** 31 - 1))
        cover_seed = int(materials.integers(0, 2 ** 31 - 1))
        problems = cls.capacity_problems(int(factors['kitchen_layout']), count, targets, hidden, visible)
        if problems:
            raise ValueError('; '.join(problems))
        return dict(
            set_size=set_size, compartment_count=count, layout_id=int(factors['kitchen_layout']),
            targets=targets, hidden_distractors=hidden, visible=visible,
            build_seed=build_seed, pose_seed=pose_seed, closers=[],
            lamp_seed=lamp_seed, cover_size=COVER_SIZE, cover_seed=cover_seed,
            hiding=deepcopy(HIDING_RULE), counter_band=deepcopy(COUNTER_BAND_RULE),
            placement=dict(tray=list(TRAY_LOCAL), submit=list(SUBMIT_LOCAL)), tray_half_m=TRAY_HALF_M)

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        targets = spec.get('targets') or []
        set_size, count = spec.get('set_size'), spec.get('compartment_count')
        if set_size not in SET_SIZES or len(targets) != set_size:
            problems.append('set_size must be 2 or 3 and match the target list')
        if count not in COMPARTMENT_COUNTS:
            problems.append('compartment_count must be 4, 6 or 8')
        objects = [t.get('object') for t in targets]
        if len(set(objects)) != len(objects) or any(o not in POOL for o in objects):
            problems.append('targets must be distinct pool objects')
        kinds = [t.get('kind') for t in targets]
        for t in targets:
            if t.get('kind') not in HIDING_KINDS:
                problems.append(f"place kind {t.get('kind')!r} is not one of {HIDING_KINDS} (no door places)")
            elif t.get('object') in POOL and t['kind'] not in hiding_kinds_of(t['object']):
                problems.append(f"{t['object']} may not hide in a {t['kind']}")
        hidden = spec.get('hidden_distractors') or []
        expected = hidden_distractor_count(count, kinds) if isinstance(count, int) else 2
        if not 1 <= len(hidden) <= expected or len(set(hidden)) != len(hidden) or set(hidden) & set(objects):
            problems.append(f'between 1 and {expected} distinct hidden distractors, none of them a target')
        for o in hidden:
            if o not in POOL or not any(confusable(o, t) for t in objects if t in POOL):
                problems.append(f'hidden distractor {o!r} shares neither category nor colour with a target')
            elif not decoy_capable(o):
                problems.append(f'hidden distractor {o!r} cannot hide in a drawer, the only compartment kind here')
        expected_visible = sorted(o for o in POOL if o not in objects and o not in hidden)
        if sorted(spec.get('visible') or []) != expected_visible:
            problems.append('visible objects must be the rest of the pool')
        needed = sum(1 for k in kinds if k == 'drawer') + len(hidden)
        if isinstance(count, int) and needed > count:
            problems.append('more hidden objects than compartments')
        for key in ('build_seed', 'pose_seed', 'lamp_seed', 'cover_seed'):
            if not isinstance(spec.get(key), int):
                problems.append(f'{key} must be an int')
        if spec.get('closers') != []:
            problems.append('closers must be an empty list (no passive closers)')
        if spec.get('cover_size') not in COVER_SIZES:
            problems.append('cover_size must name a cloche size')
        if spec.get('hiding') != HIDING_RULE:
            problems.append(f'hiding must record the placement rule {HIDING_RULE}')
        if spec.get('counter_band') != COUNTER_BAND_RULE:
            problems.append(f"counter_band must record the counter rule {COUNTER_BAND_RULE}")
        placement = spec.get('placement') or {}
        if not all(isinstance(placement.get(k), list) and len(placement[k]) == 2 for k in ('tray', 'submit')):
            problems.append('placement needs tray and submit frame positions')
        return problems

    @classmethod
    def capacity_problems(cls, layout_id, compartment_count, targets, hidden_distractors=(), visible=()):
        """The search room's table plus the high-drawer table (``blackout/structure.py``): a drawer with its
        floor at ``HIDING_FLOOR_MIN_M`` or higher for every drawer target and every decoy."""
        if not cls.USE_CAPACITY_TABLE:
            return []
        return capacity_problems(layout_id, compartment_count, targets, hidden_distractors, visible)

    @classmethod
    def cpu_gate(cls, spec):
        passed, report = super().cpu_gate(spec)
        high = high_drawers(spec['layout_id']) if spec.get('layout_id') is not None else 'unknown'
        report['high_drawers'] = None if high in (None, 'unknown') else dict(high)
        report['hiding_rule'] = deepcopy(HIDING_RULE)
        return passed, report

    # ---- construction -------------------------------------------------------
    def __init__(self, instance, **kwargs):
        self.lamp = None
        self._lamp_candidates = []
        self._lamp_choice = None
        self._lamp_state = None
        self._room_lights = []
        self._marker_materials = {}
        super().__init__(instance, **kwargs)

    def _hiding_admissible(self, candidate):
        """A target or a decoy hides only in a drawer with its floor at HIDING_FLOOR_MIN_M or higher."""
        return admissible_hiding_place(candidate, HIDING_FLOOR_MIN_M)

    def _counter_spot_admissible(self, spot):
        """The counter rule: a counter spot for a target or the lantern lies in
        the front two thirds of a wall-backed counter's depth (anywhere on an island); an open shelf only with
        its top at HIDING_FLOOR_MIN_M or higher. The verdict is written onto the spot (``counter_band``; the
        visibility ranking owns the key ``band``) for the record."""
        if spot.get('is_shelf'):
            ok = float(spot['top_z']) >= HIDING_FLOOR_MIN_M
            spot['counter_band'] = dict(shelf=True, top_z=round(float(spot['top_z']), 4), floor_min_m=HIDING_FLOOR_MIN_M,
                                        ok=ok)
            return ok
        patch = patch_for_spot(self._records, spot)
        if patch is None:
            spot['counter_band'] = dict(ok=False, error='no counter patch for this spot')
            return False
        spot['counter_band'] = counter_band(spot, patch, self._records, COUNTER_FRONT_FRACTION)
        return bool(spot['counter_band']['ok'])

    def _apply_render_profile(self):
        """The base profile, MuJoCo's headlight off, and a black sky. Both would defeat the darkness.

        The headlight is the light that would follow the camera into every cupboard. The sky is the
        subtler leak: the kitchen arena's skybox is a .98 .98 1. to .06 .08 .12 gradient, so wherever
        the room has no wall -- layout 10 is open along most of two sides -- the cameras look
        straight out at a lit sky (42.8 and 37.1 of 255 mean in the two agentview cameras on one
        instance with every room light off). The skybox is therefore painted black: the night belongs
        outside the room as well as inside it.
        """
        super()._apply_render_profile()
        visual = self.model.root.find('visual')
        headlight = visual.find('headlight')
        if headlight is None:
            headlight = ET.SubElement(visual, 'headlight')
        headlight.set('active', '0')
        sky = self.model.asset.find("texture[@type='skybox']")
        if sky is None:
            sky = ET.SubElement(self.model.asset, 'texture', name=NIGHT_SKY_TEXTURE, type='skybox',
                                width='32', height='192')
        for key in [k for k in sky.attrib if k.startswith('file')]:
            sky.attrib.pop(key)
        sky.set('builtin', sky.get('builtin') or 'flat')
        sky.set('rgb1', '0 0 0')
        sky.set('rgb2', '0 0 0')

    def _start_anchor_xy(self):
        """Where ``_start_in_front_of_frame`` will park the base; known as soon as the frame is placed."""
        front_y = -self.FOOTPRINT[1] / 2 - self.FOOTPRINT_FRONT_MARGIN
        return np.asarray(self.frame_to_world([0., front_y - START_EDGE_GAP_M, 0.]), float)[:2]

    def build_task(self):
        super().build_task()
        # No passive closers; the spec's `closers` list is empty and inert.
        self.task_spec['closer_rule'] = 'none: the dark search has no passive closers'
        self.asset_evidence['layout']['closers'] = 'none: no compartment carries a spring'
        self._record_hiding_rule()
        self._make_submit_emissive()
        self._build_lamp()
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, task=self.TASK_NAME,
            darkness=dict(arena_light='off for the whole episode (model.light_active); no wall switch, no timer '
                                      '(torch-only)',
                          headlight='disabled in the MJCF', lamp_lights=list(self.lamp['lights']),
                          note='at least one light stays active: with none, MuJoCo renders flat unlit colour'),
            scoring=self.task_spec['scoring'] + ' The lamp is a task object: it must be released and settled '
                                                'and off the tray like everything else.')

    def _record_hiding_rule(self):
        """The placement rule as applied on this kitchen, asserted here and reported by G3."""
        rows = {}
        for key in sorted(self.places):
            place = self.places[key]
            cid = place.get('compartment')
            if cid is None:
                continue
            comp = self.compartments[cid]
            rows[key] = dict(role=place['role'], compartment=cid, kind=comp['kind'], cls=comp['cls'],
                             floor_z=round(float(comp['region']['floor_z']), 4),
                             ok=bool(self._hiding_admissible(comp)))
        bad = [k for k, r in rows.items() if not r['ok']]
        if bad:
            raise ValueError(f'{self.TASK_NAME}: hiding places {bad} break the placement rule {HIDING_RULE} on '
                             f'layout {self.layout_id}')
        admissible = [cid for cid, c in self.compartments.items() if self._hiding_admissible(c)]
        self.task_spec['hiding_rule'] = dict(HIDING_RULE, places=rows, admissible_compartments=sorted(admissible),
                                             counter_band=deepcopy(COUNTER_BAND_RULE), counter_places={},
                                             note='targets and decoys only; fillers keep the drawer/door mix; '
                                                  'counter_places is filled at reset (open spots and the lamp)')

    def _counter_places(self):
        """The counter rule on the bound scene: every target on a counter (open or under a cloche) and the lantern."""
        keys = ('counter', 'island', 'shelf', 'depth_m', 'from_front_m', 'limit_m', 'top_z', 'ok')
        rows = {}
        for key in sorted(self.places):
            place = self.places[key]
            spot = place.get('spot')
            if place['role'] != 'target' or not spot:
                continue
            band = spot.get('counter_band') or self._band_of(spot)
            rows[key] = dict(role='target', kind=place.get('kind_effective', place['kind']), object=place['object'],
                             **{k: band.get(k) for k in keys if k in band})
        if self.lamp and self.lamp.get('spot'):
            spot = self.lamp['spot']
            band = spot.get('counter_band') or self._band_of(spot)
            rows['lamp'] = dict(role='lamp', **{k: band.get(k) for k in keys if k in band})
        return rows

    def _band_of(self, spot):
        probe = dict(spot)
        self._counter_spot_admissible(probe)
        return probe['counter_band']

    def _make_submit_emissive(self):
        """The Submit cap glows in the dark: emission on its label material and a twin material, same colour."""
        prefix = self.submit_prefix
        assets = self.model.asset
        label = assets.find(f"material[@name='{prefix}_material']")
        if label is not None:
            label.set('emission', '1')
        cap_material = f'{prefix}_cap_material'
        if assets.find(f"material[@name='{cap_material}']") is None:
            ET.SubElement(assets, 'material', name=cap_material, rgba=SUBMIT_CAP_RGBA, emission='1',
                              specular='0', shininess='0')
        for geom in self.model.worldbody.iter('geom'):
            if geom.get('name') == f'{prefix}_cap_visual':
                geom.set('material', cap_material)
                geom.attrib.pop('rgba', None)
        self.task_spec['submit_button']['emissive'] = dict(materials=[f'{prefix}_material', cap_material],
                                                           emission=1., rgba=SUBMIT_CAP_RGBA)

    # ---- the lamp ------------------------------------------------------------
    def _lamp_spot_candidates(self):
        """Counter spots for the lamp: within 1.2 m of the start stance, clear of everything already placed."""
        rng = np.random.default_rng([int(self.spec['lamp_seed']), 20260923])
        # LAMP_SPOT_PER_PATCH candidates per counter patch: a spot must be seen by a policy camera and keep the
        # pinch headroom; 10 per patch left 8 of 156 rows without a seen spot at the re-gate.
        spots = sample_counter_spots(self._records, self._floor, rng, per_patch=LAMP_SPOT_PER_PATCH,
                                     include_shelves=False, exclude=[self._work_polygon()], edge_only=True,
                                     keepouts=self._keepouts, clearance=max(LAMP_CLEARANCE_M, SPOT_CLEARANCE_M))
        start = self._start_anchor_xy()
        taken = [np.asarray(s['xy'], float) for s in self.task_spec['visible_objects'].values()]
        for place in self.places.values():
            if place.get('spot'):
                taken.append(np.asarray(place['spot']['xy'], float))
            # A cloche is lifted onto one of its set-down spots: the lantern must not stand on them.
            taken += [np.asarray(o['xy'], float) for o in place.get('setdown_options') or []]
        out = []
        for spot in spots:
            xy = np.asarray(spot['xy'], float)
            distance = float(np.linalg.norm(xy - start))
            if distance > LAMP_START_DIST_M or not self._counter_spot_admissible(spot):
                continue
            if any(float(np.linalg.norm(xy - t)) < LAMP_SPACING_M for t in taken):
                continue
            out.append(dict(spot, distance_from_start_m=distance,
                            on_work_counter=bool(spot.get('counter') == self.work['fixture']),
                            headroom=self._lamp_headroom(spot)))
        # The pinch needs LAMP_HEADROOM_M of clear air (the wrist under the wall units),
        # so a spot with it comes first; without one the build keeps the old order and G3 says so.
        out.sort(key=lambda s: (not s['headroom']['passed'], not s['on_work_counter'], s['distance_from_start_m']))
        return out

    def _lamp_headroom(self, spot):
        """The clear air over ``spot`` against LAMP_HEADROOM_M (:func:`headroom.headroom_gate`, the column
        the lantern's disc plus the hand pad), from the fixtures hanging over that counter top."""
        if not getattr(self, 'fixtures', None):
            raise RuntimeError('the lamp headroom rule needs the kitchen fixtures (called before they exist)')
        cache = self.__dict__.setdefault('_overhead_boxes_cache', {})
        top_z = round(float(spot['top_z']), 3)
        if top_z not in cache:
            cache[top_z] = H.overhead_boxes(self, top_z)
        return H.headroom_gate(cache[top_z], spot['xy'], LAMP_RADIUS, LAMP_HEADROOM_M, top_z)

    def _build_lamp(self):
        self._lamp_candidates = self._lamp_spot_candidates()
        if not self._lamp_candidates:
            raise self._refuse_frame('no counter spot within 1.2 m of the start stance for the lamp',
                                     surfaces=[self.work['fixture']], reason='no_lamp_spot')
        spot = self._lamp_candidates[0]
        yaw = float(aim_yaw(spot['xy'], self._start_anchor_xy()))
        pos = (float(spot['xy'][0]), float(spot['xy'][1]), float(spot['top_z']) + .002)
        rng = np.random.default_rng([int(self.spec['lamp_seed']), 20260922])
        obj, built = build_lamp(LAMP_NAME, rng, self.model.asset, pos, yaw)
        self.objects[LAMP_NAME] = obj
        q = Rotation.from_euler('z', yaw).as_quat()
        self._poses[LAMP_NAME] = [pos[0], pos[1], pos[2], float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        self.lamp = dict(name=LAMP_NAME, spot=deepcopy(spot), yaw=yaw, headroom=deepcopy(spot['headroom']),
                         lamp_headroom_short=not spot['headroom']['passed'],
                         **{k: built[k] for k in ('handle_site', 'handle_grasp_z', 'top_z', 'radius', 'mass',
                                                  'beam_local', 'beam_tilt_deg', 'ring_z', 'ring_outer_r',
                                                  'lights', 'glow_material', 'glow_rgba', 'material', 'rgba',
                                                  'kind')})
        self.task_spec['objects'][LAMP_NAME] = dict(
            kind='lamp', fixed=False, role='lamp', yaw=yaw, mass=built['mass'], radius=built['radius'],
            top_z=built['top_z'], handle_site=built['handle_site'], lights=list(built['lights']),
            # The bar handle is the pinch point, recorded in the shape `search.objects` gives the pool so
            # `search.motor.grasp_top_down` works on the lamp unchanged; `handle` keeps the two
            # along-the-bar yaws (a bar can only be pinched across itself).
            grasp=dict(anchor_local_top=(0., 0., float(built['handle_grasp_z'])),
                       anchor_local_front=(0., 0., float(built['handle_grasp_z'])), handle=True))
        self.asset_evidence[LAMP_NAME] = dict(
            source='roboquest.blackout.lamp',
            collision='cylinder foot, body and cap carrying 0.34 kg, the bail apex a 1 cm bar along the lantern '
                      'axis the gripper pinches top-down; the meshes only draw; the lamp spot and '
                      f'the counter set-down spots keep {LAMP_HEADROOM_M} m of clear air for the pinch',
            mesh='Poly Haven Lantern_01 (CC0) at scale 0.8: assets/meshes/lantern_01/ (PROVENANCE.md)',
            mass_kg=built['mass'],
            appearance=built['material'] or built['rgba'],
            lights=f"omni fill cutoff {FILL['cutoff']} diffuse {FILL['diffuse']} attenuation {FILL['attenuation']} "
                   f"ambient {FILL['ambient']} at the glass centre; a downward cone cutoff {SPOT['cutoff']} "
                   f"exponent {SPOT['exponent']} diffuse {SPOT['diffuse']} that casts the shadows; no beam; "
                   f"both children of the lamp body; lit from the start, the only light")
        self.task_spec['lamp'] = deepcopy(self.lamp)

    def _open_target_count(self):
        return sum(1 for p in self.places.values() if p['role'] == 'target' and p['kind'] == 'open')

    def _choose_lamp_spot(self):
        """Stand the lamp where a policy camera sees its glass, and keep the open places away from it.

        Runs before ``_choose_open_spots`` (same reset, cameras already posed), so an ``open`` target can
        never be bound to the spot the lamp occupies. With several open targets the lantern stands where at
        least that many open candidates, LAMP_TARGET_MIN_M away, are left for them.
        """
        if self._lamp_choice is not None:
            self._set_object_pose(*self._lamp_choice)
            return
        frusta = {f['name']: f for f in camera_frusta(self.sim, CAMERAS, aspect=self._camera_size[0] / self._camera_size[1])}
        width, height = self._camera_size
        start = self._start_anchor_xy()
        ranked = []                           # (spot, cameras that see the glow), the seen ones first
        for spot in self._lamp_candidates:
            yaw = float(aim_yaw(spot['xy'], start))
            glow = np.array([spot['xy'][0] + math.cos(yaw) * (COLUMN_XY + .002),
                             spot['xy'][1] + math.sin(yaw) * (COLUMN_XY + .002),
                             float(spot['top_z']) + RING_Z])
            cams = []
            for cam in CAMERAS:
                uv = project(frusta[cam], glow, width, height)
                if uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height:
                    cams.append(cam)
            ranked.append((spot, cams))
        # stable: headroom-clear spots first, the seen ones within, candidate order within those
        ranked.sort(key=lambda r: (not r[0]['headroom']['passed'], not r[1]))
        # An open-counter target must stay out of the lantern's pool, so the lantern stands where
        # enough open spots are LAMP_TARGET_MIN_M away; the other hiding spots keep the ordinary spacing.
        n_open = self._open_target_count()
        keepout = LAMP_TARGET_MIN_M if n_open else LAMP_SPACING_M

        def far_spots(spot):
            return [s for s in self._open_candidates
                    if float(np.linalg.norm(np.subtract(s['xy'], spot['xy']))) >= keepout]

        def enough(spots):
            """At least n_open of them, mutually far enough apart to hold one open target each."""
            chosen = []
            for s in spots:
                if all(float(np.linalg.norm(np.subtract(s['xy'], c['xy']))) >= VISIBLE_SPACING_M for c in chosen):
                    chosen.append(s)
                if len(chosen) >= n_open:
                    return True
            return n_open == 0
        best, seen_by, short = None, [], False
        if n_open:
            for spot, cams in ranked:
                if enough(far_spots(spot)):
                    best, seen_by = spot, cams
                    break
        if best is None:
            best, seen_by = ranked[0]
            short = n_open > 0   # no lamp spot leaves enough open spots far enough: G3 says so
        yaw = float(aim_yaw(best['xy'], start))
        self.lamp.update(spot=deepcopy(best), yaw=yaw, ring_seen_by=list(seen_by), open_keepout_m=keepout,
                         open_keepout_short=short, headroom=deepcopy(best['headroom']),
                         lamp_headroom_short=not best['headroom']['passed'])
        self.task_spec['lamp'] = deepcopy(self.lamp)
        self._lamp_choice = (LAMP_NAME, best['xy'], float(best['top_z']), yaw)
        self._set_object_pose(*self._lamp_choice)
        self._open_candidates = far_spots(best) or self._open_candidates

    def _choose_open_spots(self):
        self._choose_lamp_spot()
        super()._choose_open_spots()
        self.task_spec['hiding_rule']['counter_places'] = self._counter_places()

    # ---- runtime -------------------------------------------------------------
    def setup_task_references(self):
        super().setup_task_references()
        m = self.sim.model
        self._room_lights = room_light_ids(m, keep_names=self.lamp['lights'])
        self._marker_materials = {'lamp_glow': GLOW_MATERIAL,
                                  'submit_cap': f'{self.submit_prefix}_cap_material',
                                  'submit_label': f'{self.submit_prefix}_material'}
        self.task_spec['darkness']['room_light_ids'] = list(self._room_lights)

    def _lamp_pose(self):
        """(body xyz, world z axis) of the lamp, from the compiled data."""
        bid = self.obj_body_id[LAMP_NAME]
        return (np.asarray(self.sim.data.body_xpos[bid], float),
                np.asarray(self.sim.data.body_xmat[bid], float).reshape(3, 3)[:, 2])

    def _reset_internal(self):
        super()._reset_internal()
        set_room_light(self.sim.model, self._room_lights, False)
        position, _ = self._lamp_pose()
        self._lamp_state = dict(up=False, picked=False, rest_z=float(position[2]), last_z=float(position[2]),
                                still=0, picks=0)
        self.sim.forward()

    def _track_lamp(self):
        """``lamp_picked`` / ``lamp_set_down`` from the body height and the free joint's speed (cheap reads)."""
        state = self._lamp_state
        if state is None:
            return
        position, _ = self._lamp_pose()
        z = float(position[2])
        speed = float(np.linalg.norm(self.sim.data.get_joint_qvel(self.objects[LAMP_NAME].joints[0])[:3]))
        if not state['up']:
            if z > state['rest_z'] + LAMP_LIFT_M:
                state.update(up=True, picked=True, still=0, picks=state['picks'] + 1)
                self.log_event('lamp_picked', LAMP_NAME, height_m=round(z - state['rest_z'], 4))
        else:
            if speed < LAMP_STILL_SPEED and abs(z - state['last_z']) < .001:
                state['still'] += 1
            else:
                state['still'] = 0
            if state['still'] >= LAMP_STILL_TICKS:
                state.update(up=False, still=0, rest_z=z)
                self.log_event('lamp_set_down', LAMP_NAME,
                               xy=[round(float(position[0]), 4), round(float(position[1]), 4)])
        state['last_z'] = z

    def _post_action(self, action):
        self._track_lamp()
        return super()._post_action(action)

    # ---- protocol hooks (C1) --------------------------------------------------
    def _target_places(self):
        """``[(key, place)]`` of every target, in key order."""
        return [(key, self.places[key]) for key in sorted(self.places) if self.places[key]['role'] == 'target']

    def _target_place(self):
        """The first target's place (the oracle carries the lantern there first)."""
        places = self._target_places()
        if not places:
            raise KeyError('no target place')
        return places[0]

    def task_progress(self):
        """Weighted steps (blackout/structure.py), mean over the targets: out of hiding, on the tray; plus the
        lantern picked up at least once -- always, because the lantern is the only light.

        The lamp term is a latch on the episode, so unlike the rest of C1's progress it does depend on the
        trajectory; the contract asks for it explicitly (a policy that never lifts the lantern has found
        nothing on its own, whatever the scene looks like).
        """
        snapshot = self._snapshot()
        score = terminal_score(self._score_spec(), snapshot)
        if score['success']:
            return 1.
        weights = progress_weights()
        values = []
        for key, place in self._target_places():
            name = object_name(place['object'])
            state = snapshot['objects'].get(name, {})
            per = score['per_target'].get(name)
            value = 0.
            if self._out_of_hiding(key, place, snapshot):
                value += weights['out_of_hiding']
            if per and all(per.values()) and not state.get('grasped') and not state.get('robot_contact'):
                value += weights['on_tray']
            values.append(value)
        value = sum(values) / len(values) if values else 0.
        if (self._lamp_state or {}).get('picked'):
            value += weights['lamp_picked']
        return float(min(PROGRESS_CAP, value))

    def collateral_checks(self):
        checks = super().collateral_checks()
        if self._lamp_state is None:
            return checks
        position, axis = self._lamp_pose()
        released = not self._lamp_state['up']
        checks['lamp_on_floor'] = dict(bad=bool(float(position[2]) < LAMP_FLOOR_Z_M), object=LAMP_NAME, cost=1.)
        checks['lamp_tipped'] = dict(bad=bool(released and float(axis[2]) < LAMP_TIP_COS), object=LAMP_NAME,
                                     cost=1.)
        return checks

    def task_snapshot(self):
        snapshot = super().task_snapshot()
        snapshot['lamp'] = dict(self._lamp_state or {})
        return snapshot

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['roboquest']['blackout'] = dict(
            set_size=self.spec['set_size'], compartment_count=self.spec['compartment_count'],
            lighting='torch-only: the lantern is lit from the start and is the only light',
            hiding_rule=deepcopy(HIDING_RULE),
            lamp=dict(spot=self.lamp['spot']['xy'], yaw=self.lamp['yaw'],
                      ring_seen_by=list(self.lamp.get('ring_seen_by') or []),
                      distance_from_start_m=round(float(self.lamp['spot'].get('distance_from_start_m', 0.)), 3)))
        return meta

    # ---- gates ----------------------------------------------------------------
    def _material_emission(self, keys, value):
        """Set the emission of the named marker materials on the compiled model; returns the old values."""
        raw = self.sim.model._model
        saved = {}
        for key in keys:
            name = self._marker_materials.get(key)
            if name is None:
                continue
            mid = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if mid < 0:
                continue
            saved[key] = (mid, float(raw.mat_emission[mid]))
            raw.mat_emission[mid] = float(value)
        return saved

    def _restore_emission(self, saved):
        raw = self.sim.model._model
        for mid, value in saved.values():
            raw.mat_emission[mid] = value

    def _render_all(self, cameras=CAMERAS):
        width, height = self._camera_size
        return {cam: np.array(self.sim.render(width=width, height=height, camera_name=cam), copy=True)
                for cam in cameras}

    def _placement_report(self):
        """The drawer rule on the built scene: every target's and decoy's compartment is an admissible drawer."""
        rows = {}
        for key in sorted(self.places):
            place = self.places[key]
            cid = place.get('compartment')
            if cid is None:
                continue
            comp = self.compartments[cid]
            rows[key] = dict(role=place['role'], compartment=cid, kind=comp['kind'], cls=comp['cls'],
                             floor_z=round(float(comp['region']['floor_z']), 4),
                             ok=bool(self._hiding_admissible(comp)))
        counters = self._counter_places()
        return dict(floor_min_m=HIDING_FLOOR_MIN_M, places=rows, counter_band=deepcopy(COUNTER_BAND_RULE),
                    counter_places=counters,
                    ok=bool(all(r['ok'] for r in rows.values()) and all(r['ok'] for r in counters.values())))

    def hidden_state_check(self):
        """Gate G3: the darkness is real in the policy observations, and only the markers are visible.

        In the dark, with the arena light off: repainting each target's geoms red and then green changes at
        most DARK_TARGET_MAX_PX pixels in any camera (search's pixel test, so "no target pixel" is measured,
        not assumed; a sliver through a drawer gap is tolerated); every open-counter target lies at
        least LAMP_TARGET_MIN_M from the lantern's reset spot (the lantern lights about a metre
        around itself, and that keep-out is what keeps the target in the dark once the robot turns to look);
        every agentview camera's mean luminance outside the emissive markers is below DARK_MEAN_MAX, a sanity
        cap that only says the room light is off; each marker is visible above a pixel threshold in at least
        one camera; the lantern's spot has the pinch headroom; every hiding compartment obeys the drawer rule.
        The targets and decoys that are *not* in the open additionally run search's own check with the room
        light turned on, which proves the compartment or the cloche -- not the darkness -- hides them.
        """
        if getattr(self.sim, '_render_context_offscreen', None) is None:
            raise RuntimeError('hidden_state_check needs an env built with render=True')
        m = self.sim.model
        set_room_light(m, self._room_lights, False)
        self.sim.forward()
        base, again = self._render_all(), self._render_all()
        deterministic = {cam: bool(np.array_equal(base[cam], again[cam])) for cam in CAMERAS}

        def repainted(gids, rgb):
            saved = m.geom_rgba[gids].copy()
            m.geom_rgba[gids, :3] = rgb
            try:
                return self._render_all()
            finally:
                m.geom_rgba[gids] = saved

        places = self._target_places()
        target_pixels, keepouts, kinds = {}, {}, {}
        for key, place in places:
            target = object_name(place['object'])
            kinds[target] = place.get('kind_effective', place['kind'])
            gids = np.array([g for g in self._object_geoms[target] if m.geom_rgba[g, 3] > 0.], int)
            red, green = repainted(gids, (1., 0., 0.)), repainted(gids, (0., 1., 0.))
            target_pixels[target] = {cam: int(np.any(red[cam] != green[cam], axis=-1).sum()) for cam in CAMERAS}
            # An open-counter target outside the lantern's pool at reset.
            if kinds[target] == 'open' and place.get('spot'):
                distance = float(np.linalg.norm(np.subtract(place['spot']['xy'], self.lamp['spot']['xy'])))
                keepouts[target] = dict(distance_m=round(distance, 3), min_m=LAMP_TARGET_MIN_M,
                                        ok=bool(distance >= LAMP_TARGET_MIN_M - 1e-6))
        # The marker mask: pixels that change when every emissive material stops emitting.
        saved = self._material_emission(list(self._marker_materials), 0.)
        unlit = self._render_all()
        self._restore_emission(saved)
        marks, glow = {}, {}
        for name in self._marker_materials:
            one = self._material_emission([name], 0.)
            frame = self._render_all()
            self._restore_emission(one)
            marks[name] = {cam: np.any(frame[cam] != base[cam], axis=-1) for cam in CAMERAS}
            glow[name] = {cam: int(marks[name][cam].sum()) for cam in CAMERAS}
        luminance = {}
        for cam in CAMERAS:
            mask = np.any(unlit[cam] != base[cam], axis=-1)
            rest = base[cam].mean(axis=2)[~mask]
            luminance[cam] = dict(mean_outside_markers=round(float(rest.mean()) if rest.size else 0., 3),
                                  mean_all=round(float(base[cam].mean()), 3), max=int(base[cam].max()),
                                  marker_pixels=int(mask.sum()))
        scale = (self._camera_size[0] * self._camera_size[1]) / (512. * 512.)
        thresholds = dict(dark_mean_max=DARK_MEAN_MAX, lamp_glow_px=GLOW_MIN_PX * scale,
                          submit_px=SUBMIT_GLOW_MIN_PX * scale,
                          open_target_min_m=LAMP_TARGET_MIN_M, lamp_headroom_m=LAMP_HEADROOM_M,
                          dark_target_max_px=DARK_TARGET_MAX_PX * scale, hiding_floor_min_m=HIDING_FLOOR_MIN_M)
        # Up to dark_target_max_px of a target may show in the dark (a sliver through a
        # compartment gap the lantern lights); the room-light check below keeps the family's zero rule.
        target_ok = all(v <= thresholds['dark_target_max_px'] for per in target_pixels.values() for v in per.values())
        dark_ok = all(v['mean_outside_markers'] < DARK_MEAN_MAX for cam, v in luminance.items()
                      if cam != WRIST_CAMERA)
        glow_ok = max(glow['lamp_glow'].values()) >= thresholds['lamp_glow_px']
        keepout_ok = all(k['ok'] for k in keepouts.values())
        # ... and the lantern where the pinch has its headroom (the wrist under the wall units).
        headroom = self.lamp.get('headroom') or self._lamp_headroom(self.lamp['spot'])
        headroom_ok = bool(headroom['passed'])
        submit_ok = (max(glow['submit_cap'].values()) + max(glow['submit_label'].values())) >= thresholds['submit_px']
        placement = self._placement_report()
        # The room-light half: what a compartment or a cloche hides must stay hidden with the light on. The
        # open targets are left out of it (the darkness is what hides them, and that is measured above).
        open_names = [t for t, k in kinds.items() if k == 'open']
        hidden_ids = list(self.task_spec['hidden_object_ids'])
        lit = None
        covered_or_compartment = [n for n in hidden_ids if n not in open_names]
        if covered_or_compartment:
            set_room_light(m, self._room_lights, True)
            self.task_spec['hidden_object_ids'] = covered_or_compartment
            self.sim.forward()
            try:
                lit = SearchRoom.hidden_state_check(self)
                lit.pop('images', None)
            finally:
                self.task_spec['hidden_object_ids'] = hidden_ids
                set_room_light(m, self._room_lights, False)
                self.sim.forward()
        passed = bool(all(deterministic.values()) and dark_ok and glow_ok and submit_ok and keepout_ok
                      and headroom_ok and target_ok and placement['ok'] and (lit is None or lit['passed']))
        return dict(passed=passed, render_deterministic=deterministic, dark_luminance=luminance,
                    targets=list(target_pixels), target_changed_pixels=target_pixels, glow_pixels=glow,
                    open_target_keepout=keepouts or None, lamp_headroom=headroom, placement=placement,
                    thresholds=thresholds, set_size=self.spec['set_size'],
                    compartment_count=self.spec['compartment_count'], place_kinds=kinds,
                    lit_check=lit, lamp=dict(spot=self.lamp['spot']['xy'], yaw=self.lamp['yaw'],
                                             ring_seen_by=list(self.lamp.get('ring_seen_by') or [])),
                    method='room light off; emission toggled per material for the marker masks; the room-light '
                           'check runs over the targets and decoys that are not in the open',
                    images=base)

    # ---- teleport certificate (C3 / G3.5) --------------------------------------
    def _base_joint_names(self):
        names = [f'{p}joint_mobile_{axis}' for p in ('mobilebase0_', 'robot0_', '') for axis in
                 ('forward', 'side', 'yaw')]
        have = []
        for name in names:
            try:
                self.sim.model.joint_name2id(name)
            except Exception:
                continue
            have.append(name)
        return have[:3] if len(have) >= 3 else []

    def _teleport_base(self, xy, yaw, iterations=3):
        """Drive the Omron base's own slide/hinge joints until the base frame is at (xy, yaw).

        The contract's certificate for this task asks for the base at the place's standoff stance, so
        the lantern's light and the wrist camera see the same thing the robot would; C3's "never move the
        robot" is overridden here on purpose and recorded in the step detail.
        """
        joints = self._base_joint_names()
        if not joints:
            return dict(ok=False, detail='no mobile base joints in this model')
        forward, side, yaw_joint = joints
        data = self.sim.data
        target_xy = np.asarray(xy, float)
        for _ in range(iterations):
            have_xy, have_yaw = self.base_pose()
            error = target_xy - have_xy
            heading = float(data.qpos[self.sim.model.get_joint_qpos_addr(yaw_joint)])
            base_yaw = have_yaw - heading
            step = np.array([math.cos(base_yaw) * error[0] + math.sin(base_yaw) * error[1],
                             -math.sin(base_yaw) * error[0] + math.cos(base_yaw) * error[1]])
            for joint, delta in ((forward, step[0]), (side, step[1])):
                adr = self.sim.model.get_joint_qpos_addr(joint)
                data.qpos[adr] = float(data.qpos[adr]) + float(delta)
                data.set_joint_qvel(joint, 0.)
            adr = self.sim.model.get_joint_qpos_addr(yaw_joint)
            data.qpos[adr] = float(data.qpos[adr]) + float(_wrap(yaw - self.base_pose()[1]))
            data.set_joint_qvel(yaw_joint, 0.)
            self.sim.forward()
        have_xy, have_yaw = self.base_pose()
        error = float(np.linalg.norm(have_xy - target_xy))
        return dict(ok=bool(error < .08), detail=dict(target_xy=[float(v) for v in target_xy],
                                                      reached_xy=[float(v) for v in have_xy],
                                                      error_m=round(error, 4),
                                                      yaw_error_deg=round(math.degrees(abs(_wrap(have_yaw - yaw))), 2),
                                                      note='C3 override: the base is moved, the arm is not'))

    def _place_anchor(self, place):
        """(world xy, standoff stance) of whatever the target hides in."""
        cid = place.get('compartment')
        if cid:
            comp = self.compartments[cid]
            return np.asarray(comp['pos'], float)[:2], comp.get('standoff')
        spot = place.get('spot') or {}
        return np.asarray(spot.get('xy', [0., 0.]), float), spot.get('standoff')

    def teleport_solution(self):
        """G3.5: for each target, the lantern makes its place visible, then the target goes on the tray (C3).

        Per target: move the base to the place's standoff stance; open the compartment or lift the cloche
        onto its set-down spot; set the lantern down 0.4-0.7 m from the place and measure the target's pixels
        and the lit interior's luminance under the lantern alone (recorded, not fatal); stand the lantern
        back on its reset spot; put the target on its tray slot. Then every opened compartment is reclosed
        and the scene settles with zero actions; success must be True and progress 0 -> 1.
        """
        steps = []
        started = time.monotonic()

        def record(step, ok, detail=None):
            steps.append(dict(step=step, ok=bool(ok), detail=detail))
            return ok
        places = self._target_places()
        slots = TRAY_SLOTS.get(len(places)) or [(0., 0.)] * len(places)
        opened = {}
        for index, (key, place) in enumerate(places):
            target = object_name(place['object'])
            anchor, stance = self._place_anchor(place)
            # The base and the compartment come first: every camera is on the robot, so the set-down search
            # below can only be judged once the robot stands where it would stand to work the place.
            if stance:
                moved = self._teleport_base(stance['xy'], stance['yaw'])
                record(f'move the base to the standoff stance of {target}', moved['ok'], moved['detail'])
            else:
                record(f'move the base to the standoff stance of {target}', False, 'the place records no standoff')
            cid = place.get('compartment')
            if cid:
                try:
                    values = {}
                    for joint in self.compartments[cid]['joints']:
                        jid = self.sim.model.joint_name2id(joint)
                        lo, hi = (float(v) for v in self.sim.model.jnt_range[jid])
                        open_qpos = hi if abs(hi) >= abs(lo) else lo
                        self.sim.data.set_joint_qpos(joint, open_qpos)
                        self.sim.data.set_joint_qvel(joint, 0.)
                        values[joint] = open_qpos
                    self.sim.forward()
                    opened[cid] = values
                    record(f'open {cid}', True, dict(joints=values, kind=self.compartments[cid]['kind'],
                                                     floor_z=round(float(self.compartments[cid]['region']['floor_z']), 4)))
                except Exception as exc:  # noqa: BLE001
                    record(f'open {cid}', False, f'{type(exc).__name__}: {exc}')
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
                record(f'expose {target}', True, dict(kind=place.get('kind_effective', place['kind']),
                                                      spot=(place.get('spot') or {}).get('xy')))
            try:
                aim = self._aim_point(target, anchor)
                chosen, tried = self._best_setdown(aim, target)
                record(f'set the lamp down at the place of {target}', True,
                       dict(xy=[round(float(v), 3) for v in chosen['spot']['xy']],
                            yaw=round(chosen['yaw'], 4),
                            distance_m=round(float(chosen['spot']['setdown_distance_m']), 3),
                            range_m=list(SETDOWN_RANGE_M), band_used_m=chosen['spot']['band_m'],
                            within_contract_band=chosen['spot']['within_contract_band'],
                            spots_tried=tried,
                            method='the certificate renders each allowed set-down spot and keeps the one '
                                   'that lights the place best; a policy only has to find one of them'))
                record(f'{LIT_STEP}: {target}', chosen['ok'], chosen['detail'])
            except Exception as exc:  # noqa: BLE001
                record(f'set the lamp down at the place of {target}', False, f'{type(exc).__name__}: {exc}')
                record(f'{LIT_STEP}: {target}', False, f'{type(exc).__name__}: {exc}')
            if self._lamp_choice is not None:
                # Back to the reset spot before the success half. The set-down spot exists to prove the
                # light reaches the place; leaving the lamp there would make the settle step judge the
                # lamp's resting pose rather than the targets on the tray.
                self._set_object_pose(*self._lamp_choice)
                self.sim.forward()
                record(f'stand the lamp back up on its reset spot after {target}', True,
                       dict(xy=[round(float(v), 3) for v in self._lamp_choice[1]]))
            try:
                slot = slots[index]
                xy = self.frame_to_world([self.tray['centre_local'][0] + slot[0],
                                          self.tray['centre_local'][1] + slot[1], 0.])[:2]
                self._set_object_pose(target, xy, self.tray['top_z'], float(self.work['yaw']))
                self.sim.forward()
                record(f'place {target} on the tray', True, dict(slot=list(slot), xy=[float(v) for v in xy]))
            except Exception as exc:  # noqa: BLE001
                record(f'place {target} on the tray', False, f'{type(exc).__name__}: {exc}')
        for cid, values in opened.items():
            try:
                for joint in values:
                    self.sim.data.set_joint_qpos(joint, 0.)
                    self.sim.data.set_joint_qvel(joint, 0.)
                self.sim.forward()
                record(f'reclose {cid}', True, 'closed again before settling')
            except Exception as exc:  # noqa: BLE001
                record(f'reclose {cid}', False, f'{type(exc).__name__}: {exc}')
        try:
            zero = np.zeros(self.action_dim)
            for _ in range(SETTLE_TICKS):
                self.step(zero)
            score = self._current_score()
            record('settle', score['success'], dict(ticks=SETTLE_TICKS, errors=score['errors'],
                                                    score=score['score'], progress=self.task_progress(),
                                                    seconds=round(time.monotonic() - started, 2)))
        except Exception as exc:  # noqa: BLE001
            record('settle', False, f'{type(exc).__name__}: {exc}')
        return steps

    def _aim_point(self, target, anchor):
        """World point the lamp is aimed at: the target itself, once the compartment is open."""
        try:
            return self._body_position(target).copy()
        except Exception:  # noqa: BLE001
            return np.array([float(anchor[0]), float(anchor[1]), float(self.work['top_z'])])

    def _floor_setdown_spots(self, aim, exclude_xy=None):
        """Free floor points around ``aim``: where a lantern goes when the place is below a counter.

        A drawer's interior is lit best from close by and from above the counter it sits under; the floor
        ring is the fallback for places no counter spot reaches (kept from the work-lamp contract). Spots
        are ranked by the caller; here they only have to be on free floor, clear of the fixtures and not
        under the robot's own base.
        """
        aim = np.asarray(aim, float)
        out = []
        for radius in np.arange(.30, 1.01, .05):
            for k in range(24):
                angle = 2 * math.pi * k / 24
                xy = aim[:2] + np.array([math.cos(angle), math.sin(angle)]) * radius
                if not point_in_floor(xy, self._floor, FLOOR_MARGIN_M):
                    continue
                if _blocked(xy, self._blockers) or _clearance(xy, self._blockers) < LAMP_CLEARANCE_M:
                    continue
                if exclude_xy is not None and float(np.linalg.norm(xy - np.asarray(exclude_xy, float))) < .45:
                    continue
                out.append(dict(xy=np.asarray(xy, float), top_z=0., surface='floor'))
        return out

    def _lamp_setdown(self, aim, limit=SETDOWN_TRIES, stance_xy=None):
        """Up to ``limit`` spots where the lamp may stand to light ``aim``, nearest first.

        ``aim`` is the target's own world position and the distance is measured in three dimensions,
        from the lamp's glow to it -- the contract's 0.4-0.7 m band, but only meaningful if the height
        is counted, because a compartment's interior is below the counter the lamp might stand on.
        Candidates come from the counter patches *and* from the free floor (:meth:`_floor_setdown_spots`).
        Counter spots are sampled rather than enumerated, so the band can still come up empty by
        accident; the search then widens once to ``SETDOWN_FALLBACK_M`` and records which band each
        spot came from. The caller picks between them by measurement.
        """
        rng = np.random.default_rng([int(self.spec['lamp_seed']), 20260924])
        spots = []
        for s in sample_counter_spots(self._records, self._floor, rng, per_patch=SETDOWN_PER_PATCH,
                                     include_shelves=True, exclude=[], edge_only=True,
                                     keepouts=self._keepouts, clearance=LAMP_CLEARANCE_M):
            gate = self._lamp_headroom(s)      # the set-down lowers the pinched lantern: same wrist, same air
            if gate['passed']:
                spots.append(dict(s, surface='counter', headroom=gate))
        aim = np.asarray(aim, float)
        spots += self._floor_setdown_spots(aim, exclude_xy=stance_xy)
        measured = [(float(np.linalg.norm(np.array([float(s['xy'][0]), float(s['xy'][1]),
                                                    float(s['top_z']) + RING_Z]) - aim)), s) for s in spots]
        out, seen = [], set()
        for band in (SETDOWN_RANGE_M, SETDOWN_FALLBACK_M):
            for distance, spot in sorted((r for r in measured if band[0] <= r[0] <= band[1]),
                                         key=lambda r: r[0]):
                # A coarse grid, not the exact point: the floor ring puts two dozen near-identical
                # candidates in the band and rendering six of them would prove nothing.
                key = (spot['surface'], round(float(spot['xy'][0]) / SETDOWN_GRID_M),
                       round(float(spot['xy'][1]) / SETDOWN_GRID_M))
                if key in seen:
                    continue
                seen.add(key)
                out.append(dict(spot, setdown_distance_m=distance, band_m=list(band),
                                within_contract_band=bool(band == SETDOWN_RANGE_M)))
                if len(out) >= limit:
                    return out
        if not out:
            raise ValueError(f'no counter or floor spot {SETDOWN_FALLBACK_M[0]}-{SETDOWN_FALLBACK_M[1]} m '
                             f'from the place among {len(spots)} sampled spots')
        return out

    def _best_setdown(self, aim, target):
        """Stand the lamp on the allowed set-down spot that lights the place best. Returns (best, tried).

        A policy only has to find *one* good spot, so the certificate is allowed to look: it renders
        the place from each candidate and keeps the brightest. The lamp is left standing on the winner.
        """
        candidates = self._lamp_setdown(aim, stance_xy=self.base_pose()[0])
        dark = self._reference_frames()
        best, tried = None, []
        for spot in candidates:
            yaw = float(aim_yaw(spot['xy'], aim[:2]))
            self._set_object_pose(LAMP_NAME, spot['xy'], spot['top_z'], yaw)
            self.sim.forward()
            ok, detail = self._lit_measurement(target, dark=dark)
            row = dict(spot=spot, yaw=yaw, ok=ok, detail=detail)
            tried.append(dict(xy=[round(float(v), 3) for v in spot['xy']],
                              distance_m=round(float(spot['setdown_distance_m']), 3),
                              luminance=None if isinstance(detail, str) else detail['measured_luminance'],
                              pixels=None if isinstance(detail, str) else detail['measured_pixels'], ok=ok))
            if best is None or _setdown_rank(row) > _setdown_rank(best):
                best = row
            if ok:
                break
        yaw = best['yaw']
        self._set_object_pose(LAMP_NAME, best['spot']['xy'], best['spot']['top_z'], yaw)
        self.sim.forward()
        return best, tried

    def _reference_frames(self):
        """The policy frames with the lamp back on its reset spot: the "before" of the lit check.

        The comparison has to be made over the *same pixels* as the "after", and those pixels are only
        known once the target has been found in the aimed render, so this hands back whole frames and
        :meth:`_lit_measurement` slices the region it ends up using out of them.
        """
        if self._lamp_choice is None:
            return None
        saved = self.sim.data.get_joint_qpos(self.objects[LAMP_NAME].joints[0]).copy()
        try:
            self._set_object_pose(*self._lamp_choice)
            self.sim.forward()
            return self._render_all()
        except Exception:  # noqa: BLE001
            return None
        finally:
            self.sim.data.set_joint_qpos(self.objects[LAMP_NAME].joints[0], saved)
            self.sim.forward()

    def _lit_measurement(self, target, dark=None):
        """Does the lamp alone make the place readable? Target pixels and the *interior's* luminance.

        The interior is the target's own pixels' bounding box padded by half its size (at least
        ``LIT_PAD_MIN_PX``): the target and the surface it sits on, which is exactly what a policy would
        have to read. The whole-frame mean is reported alongside it so the darkness of the rest of the
        room stays visible in the record. Two things are asserted about that region: an absolute floor
        (``LIT_INTERIOR_MIN``) and, when the reference frames are available, the factor by which bringing
        the lamp raised it (``LIT_CONTRAST_X``), measured over the *same pixels* with the lamp back on its
        reset spot.
        """
        if getattr(self.sim, '_render_context_offscreen', None) is None:
            return False, 'the certificate needs an env built with render=True'
        m = self.sim.model
        set_room_light(m, self._room_lights, False)
        self.sim.forward()
        base = self._render_all()
        gids = np.array([g for g in self._object_geoms[target] if m.geom_rgba[g, 3] > 0.], int)
        saved = m.geom_rgba[gids].copy()
        m.geom_rgba[gids, :3] = (1., 0., 0.)
        red = self._render_all()
        m.geom_rgba[gids, :3] = (0., 1., 0.)
        green = self._render_all()
        m.geom_rgba[gids] = saved
        pixels, interior, frame_mean, before_all = {}, {}, {}, {}
        for cam in CAMERAS:
            mask = np.any(red[cam] != green[cam], axis=-1)
            pixels[cam] = int(mask.sum())
            frame_mean[cam] = round(float(base[cam].mean()), 3)
            before_all[cam] = None
            if not mask.any():
                interior[cam] = None
                continue
            rows, cols = np.where(mask)
            r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
            pad_r, pad_c = max(LIT_PAD_MIN_PX, (r1 - r0) // 2), max(LIT_PAD_MIN_PX, (c1 - c0) // 2)
            box = (slice(max(0, r0 - pad_r), r1 + pad_r + 1), slice(max(0, c0 - pad_c), c1 + pad_c + 1))
            interior[cam] = round(float(base[cam][box].mean()), 3)
            if dark is not None and cam in dark:
                before_all[cam] = round(float(dark[cam][box].mean()), 3)
        wrist = CAMERAS[2]
        best = max(CAMERAS, key=lambda cam: pixels[cam])
        chosen = wrist if pixels[wrist] >= LIT_TARGET_MIN_PX else best
        value = interior[chosen] or 0.
        before = before_all.get(chosen)
        gain = None if before is None else round(value / max(before, .05), 2)
        ok = bool(pixels[chosen] >= LIT_TARGET_MIN_PX and value >= LIT_INTERIOR_MIN
                  and (gain is None or gain >= LIT_CONTRAST_X))
        return ok, dict(target_pixels=pixels, interior_luminance=interior, frame_luminance=frame_mean,
                        interior_before_aiming=before_all, contrast_gain=gain,
                        wrist_camera=wrist, measured_camera=chosen, measured_pixels=pixels[chosen],
                        measured_luminance=interior[chosen],
                        thresholds=dict(target_px=LIT_TARGET_MIN_PX, interior_luminance=LIT_INTERIOR_MIN,
                                        contrast_gain=LIT_CONTRAST_X),
                        note="interior = the target's pixel bounding box padded by half its size, on the "
                             'lamp-only render; the wrist camera carries the verdict when it sees the '
                             'target, otherwise the camera with the most target pixels does. '
                             '`contrast_gain` is that region against the same region with the lamp back '
                             'on its reset spot: what bringing the lamp actually bought.')


def _setdown_rank(row):
    """Order set-down candidates: a passing one first, then by how bright it made the place."""
    detail = row['detail']
    if isinstance(detail, str):
        return (0, 0., 0)
    return (1 if row['ok'] else 0, float(detail['measured_luminance'] or 0.), int(detail['measured_pixels']))


def _wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi
