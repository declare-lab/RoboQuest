"""RoboQuest tasks in RoboCasa's own task pattern: one ``Kitchen`` subclass per task.

What is RoboCasa's and stays untouched: the kitchen layouts and styles, the
fixtures and objects, the robot and its three cameras, and the render recipe
(4x MSAA, no cast shadows, the arena light and headlight). What this base adds:

* a **frozen instance** (kitchen layout and style plus the task's own spec)
  instead of per-reset sampling, so an evaluation set is a list of files;
* a deterministic **work surface** choice (largest qualifying island, else
  counter) and a **footprint** on it that avoids RoboCasa's counter decor;
* a build-time **stance gate** (:meth:`RoboQuestKitchen.frame_stance`): the
  base must be able to stand in front of the frame at its reset standoff and
  RoboCasa must spawn it there facing the frame, else the kitchen is refused
  like a decor-blocked one (``WorkFrameRefused``);
* task props appended to the model in :meth:`build_task`, in a local frame
  whose +y points away from the robot on any layout;
* the physical **Submit** button and evaluator-side scoring frozen at the first
  press (``CupSubmitMixin``), and an ``ep_meta`` that records the instance.

Policies receive only the fixed cameras, proprioception and the public goal.
"""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import Accessory, FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
import robocasa.models.fixtures.fixture_utils as FixtureUtils

from roboquest.base.submit_button import CupSubmitMixin
from roboquest.harness.contract import CAMERAS
from roboquest.base.scene import VisionKitchenScene
from roboquest import lean as lean_mod   # ``lean`` is make_env's parameter name

ROBOCASA_OFFSAMPLES = 4  # RoboCasa keeps MuJoCo's default multisampling; its lights cast no shadows.
CLUTTER_SENTENCE = ('Other objects on the counter are not part of the task; '
                    'you may move them if needed.')   # spec 1.3, last sentence of the framed goal


class WorkFrameRefused(ValueError):
    """This kitchen cannot host the task's footprint decor-free, so the instance is refused at build time.

    Raised by :meth:`RoboQuestKitchen._choose_work_fixture` and
    :meth:`RoboQuestKitchen.place_work_frame` instead of laying the task frame
    on top of RoboCasa decor. A ``ValueError``, so ``registry.mint_grid``'s
    seed walk treats it as a rejected seed and ``roboquest_gate_batch`` records
    it as a G1 failure carrying the message rather than as a crashed worker.

    Attributes: ``task``, ``layout_id``, ``style_id``, ``surfaces`` (the surface
    names tried), ``blockers`` (the decor/fixture names that blocked them) and
    ``reason``: ``'decor'`` (no decor-free footprint), ``'stance_blocked'`` (a
    fixture stands where the base must, :meth:`RoboQuestKitchen.frame_stance`)
    or ``'stance_disconnected'`` (RoboCasa spawns the base behind or beside the
    surface, so the heading-kept skills never reach the front stance).
    """

    def __init__(self, message, task='', layout_id=None, style_id=None, surfaces=(), blockers=(), reason='decor'):
        super().__init__(message)
        self.task, self.layout_id, self.style_id = task, layout_id, style_id
        self.surfaces, self.blockers = list(surfaces), list(blockers)
        self.reason = str(reason)


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def yaw_quat_wxyz(yaw):
    return [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _poly_penetration(a, b):
    """Depth of overlap of two convex polygons (the shortest translation that separates them); 0 when apart."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    depth = math.inf
    for poly in (a, b):
        for edge in np.roll(poly, -1, axis=0) - poly:
            axis = np.array([-edge[1], edge[0]])
            length = float(np.linalg.norm(axis))
            if length < 1e-12:
                continue
            axis = axis / length
            pa, pb = a @ axis, b @ axis
            overlap = float(min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
            if overlap <= 0.:
                return 0.
            depth = min(depth, overlap)
    return float(depth)


# The frame's front stance (build-time gate, see RoboQuestKitchen.frame_stance).
STANCE_HEADING_TOL_RAD = math.radians(10.)   # the reset heading must face the frame this closely
STANCE_STEP_M = .05                          # spacing of the informational stance samples along the front


class RoboQuestKitchen(CupSubmitMixin, VisionKitchenScene):
    """Base class for RoboQuest tasks. Subclasses set the class attributes and
    implement the generator protocol (classmethods, CPU only) and the scene
    protocol (instance methods, simulator)."""

    TASK_NAME = ''
    CONTRACT_VERSION = ''
    GENERATOR_VERSION = 'v0'
    FACTORS = {}                  # factor name -> tuple of levels; the first level is the default
    WORK_FIXTURE_TYPES = (FixtureType.ISLAND, FixtureType.COUNTER)
    FOOTPRINT = (.60, .50)        # free top region the props need: along the front, depth
    FOOTPRINT_FRONT_MARGIN = .03  # footprint starts this far behind the surface's front edge
    EXCLUDED_LAYOUTS = ()         # RoboCasa layout ids the layout gate must not offer this task (see
                                  # scripts/roboquest_layout_gate.py --exclude); empty by default.
    submit_prefix = 'submit'
    BUDGET_TICKS = 12000          # spec 1.2: one fixed budget in 20 Hz ticks per task, never exposed
    SURFACE_DROP_M = .05          # spec 1.5: a prop this far below the work top has left the surface
    # Memory footprint (lean.py). ``make_env(lean=True)`` sets the flag; the state names describe the
    # current compile and are reset by every compile (a hard reset compiles again).
    release_after_compile = False   # free MuJoCo's ~1 GB compiler state right after every compile
    compiler_state = 'kept'         # 'kept' | 'released'
    compiled_scene_xml = None       # MuJoCo's saved XML of the current compile, taken before a release
    SURFACE_START_TOL_M = .02     # how far below the top a prop may start and still count as on it

    def __init_subclass__(cls, **kwargs):
        """A task that still declares the v0 ``HORIZON`` keeps it as its budget until it sets its own."""
        super().__init_subclass__(**kwargs)
        if 'HORIZON' in cls.__dict__ and 'BUDGET_TICKS' not in cls.__dict__:
            cls.BUDGET_TICKS = int(cls.__dict__['HORIZON'])

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def default_factors(cls):
        return {name: levels[0] for name, levels in cls.FACTORS.items()}

    @classmethod
    def goal(cls, spec):
        """Public instruction for this instance; must not leak hidden state."""
        raise NotImplementedError

    @classmethod
    def frame_goal(cls, spec, noun, count, middle, clutter=True, scope=True):
        """Frame the task's v0 middle sentence per spec 1.3.

        The wording is identical at every observability level, so the prompt
        never reveals whether anything is hidden. The scope sentence carries the
        count only when the generator drew ``spec['show_count']`` (wording
        variation, never analysed); ``clutter=False`` drops the closing sentence
        for the tasks that must not license moving other objects.
        """
        # ``scope=False`` drops the opening sentence (the search variants: it told nothing
        # and the dark search's count included objects hidden in compartments).
        opening = ('' if not scope else
                   f'There are {count} {noun} on the counters in this kitchen.' if spec.get('show_count')
                   else f'There are {noun} on the counters in this kitchen.')
        parts = [opening, str(middle).strip()] + ([CLUTTER_SENTENCE] if clutter else [])
        return ' '.join(part for part in parts if part)

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        """Draw the task's own spec (structure, hidden state, placements) from named RNG streams."""
        raise NotImplementedError

    @classmethod
    def validate_spec(cls, spec):
        """Return a list of problems; empty when the spec is acceptable."""
        return []

    @classmethod
    def cpu_gate(cls, spec):
        """Optional CPU physics/geometry gate: (passed, report)."""
        return True, {}

    @classmethod
    def split_key(cls, spec):
        """Coarse structure class used to hold out evaluation instances by factor, not seed."""
        return ''

    # ---- construction -------------------------------------------------------
    def __init__(self, instance, **kwargs):
        instance = deepcopy(instance)
        if instance.get('task') != self.TASK_NAME:
            raise ValueError(f"instance is for task {instance.get('task')!r}, not {self.TASK_NAME!r}")
        if instance['spec'].get('contract_version') != self.CONTRACT_VERSION:
            raise ValueError('instance contract does not match this task class')
        self.instance = instance
        self.spec = instance['spec']
        self.PUBLIC_GOAL = self.goal(self.spec)
        self.work_fixture = None
        self.work = None
        self.submission = None
        self.timed_out = False
        self.events = []
        self._event_state = None
        self._work_surface_survey = []   # (surface name, blocking decor) per surface _choose_work_fixture tried
        kwargs.setdefault('layout_ids', int(instance['layout_id']))
        kwargs.setdefault('style_ids', int(instance['style_id']))
        super().__init__(variant=0, **kwargs)

    def _setup_kitchen_references(self):
        Kitchen._setup_kitchen_references(self)
        self.work_fixture = self._choose_work_fixture()
        self.init_robot_base_ref = self.work_fixture

    def _choose_work_fixture(self):
        """Largest qualifying surface whose front region admits a decor-free footprint, islands first;
        deterministic (sorted by name), no RNG. Surfaces blocked by decor on every offset are skipped.

        **No fallback.** Up to v1 this returned the least-bad blocked surface when
        every candidate was blocked, so the frame ended up under RoboCasa decor
        and the instance only failed at the teleport certificate or in a rollout.
        It now raises :class:`WorkFrameRefused` naming the layout, the style, the
        surfaces tried and their blockers, so the build fails at G1 and the seed
        walk moves on. A subclass that runs its own, wider frame search (it
        overrides :meth:`place_work_frame`) still gets the least-bad surface as a
        base anchor; the refusal is stashed and raised by the base
        ``place_work_frame`` if that search reaches it.
        """
        blocked = []          # (kind order, -area, name, blockers) of surfaces with decor on every offset
        survey = []
        for order, kind in enumerate(self.WORK_FIXTURE_TYPES):
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
                area = max(r['size'][0] * r['size'][1] for r in regions.values())
                offset, blocked_by = self._find_frame_offset(fixture)
                if offset is None:
                    hits = sorted({n for _, names in blocked_by for n in names})
                    blocked.append((order, -area, name, hits))
                    survey.append((name, hits))
                    continue
                candidates.append((-area, name))
            if candidates:
                candidates.sort()
                self._work_surface_survey = survey
                return self.fixtures[candidates[0][1]]
        self._work_surface_survey = survey
        if blocked:
            blocked.sort()
            if self._frame_search_is_delegated():
                # The subclass searches further (other regions, other surfaces) and refuses by itself;
                # hand it the least-bad anchor and let its own placement decide.
                return self.fixtures[blocked[0][2]]
            raise self._refuse_frame('no decor-free {footprint} m work frame on any work surface',
                                     surfaces=[row[2] for row in blocked],
                                     blockers=sorted({n for row in blocked for n in row[3]}),
                                     detail='; '.join(f'{row[2]} blocked by {", ".join(row[3]) or "decor"}'
                                                      for row in blocked[:4]))
        raise self._refuse_frame('no island or counter offers a free {footprint} m region')

    def _frame_search_is_delegated(self):
        """True when this subclass replaces :meth:`place_work_frame` and therefore runs its own frame
        search (search room walks every region of every surface). Its placement raises the refusal."""
        return type(self).place_work_frame is not RoboQuestKitchen.place_work_frame

    def _refuse_frame(self, message, surfaces=(), blockers=(), detail='', reason='decor'):
        """Build a :class:`WorkFrameRefused` naming the task, layout, style, surfaces and blockers."""
        footprint = 'x'.join(f'{float(v):.2f}' for v in self.FOOTPRINT)
        layout = (self.instance or {}).get('layout_id', getattr(self, 'layout_id', None))
        style = (self.instance or {}).get('style_id', getattr(self, 'style_id', None))
        text = (f'{self.TASK_NAME}: ' + message.format(footprint=footprint)
                + f' on layout {layout} style {style}' + (f'; {detail}' if detail else ''))
        return WorkFrameRefused(text, task=self.TASK_NAME, layout_id=layout, style_id=style,
                                surfaces=surfaces, blockers=blockers, reason=reason)

    def _initialize_sim(self, xml_string=None):
        super()._initialize_sim(xml_string)
        self.compiler_state, self.compiled_scene_xml = 'kept', None
        if self.release_after_compile:
            self.release_compiler_state()

    def release_compiler_state(self):
        """Capture the saved scene XML, then free MuJoCo's ~1 GB compiler state (see lean.py).

        Idempotent per compile: True when this call freed the state, False when it was
        already released. After a release ``sim.model.get_xml()`` raises until the next
        compile (``mujoco.FatalError: No XML model loaded``); :meth:`scene_xml` serves the
        copy taken here, so no caller can lose the scene. Callers that export the scene
        themselves (``episode_runner.save_initial``) do so first, then call this.
        """
        if self.compiler_state == 'released':
            return False
        self.compiled_scene_xml = self.sim.model.get_xml()
        lean_mod.release_compiler_state()
        self.compiler_state = 'released'
        return True

    def scene_xml(self):
        """MuJoCo's saved XML of the current compile: live, or the copy taken before a release."""
        if self.compiler_state == 'released':
            return self.compiled_scene_xml
        return self.sim.model.get_xml()

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._apply_render_profile()
        # RoboCasa computes the robot base anchor at the end of its _load_model (after the task's
        # objects are created), so the frame alignment must run here, not inside place_work_frame.
        self._align_robot_base_with_frame()
        self._anchor_base_for_task()
        # Every task passes here, including the ones with their own place_work_frame: the stance gate
        # needs the finished floor (all fixtures) and the anchor the episode really starts from.
        self._check_frame_stance()

    def _anchor_base_for_task(self):
        """Hook for a task that starts the robot somewhere other than RoboCasa's aligned anchor
        (search_room stands it in front of the frame). It runs after the alignment and before the
        stance gate, so the gate judges the reset pose the episode starts from, not the default."""
        return None

    def _apply_render_profile(self):
        """RoboCasa's recipe, made explicit: default MSAA, no cast shadows."""
        visual = self.model.root.find('visual')
        if visual is None:
            visual = ET.SubElement(self.model.root, 'visual')
        quality = visual.find('quality')
        if quality is None:
            quality = ET.SubElement(visual, 'quality')
        quality.set('offsamples', str(ROBOCASA_OFFSAMPLES))
        for light in self.model.root.iter('light'):
            light.set('castshadow', 'false')

    # ---- work surface geometry ---------------------------------------------
    def fixture_local_to_world(self, fixture, local):
        return np.asarray(fixture.pos, float) + yaw_matrix(float(fixture.rot)) @ np.asarray(local, float)

    # Fixture classes that never occupy a work top: the surfaces themselves and the room shell.
    _NOT_DECOR_CLASSES = ('Counter', 'Wall', 'Floor', 'Window', 'Hood', 'Fridge', 'Dishwasher', 'Oven')

    def _decor_boxes(self, top_z):
        """World xy boxes of everything standing on or reaching into the work-top band (kept, treated as
        occupied): RoboCasa accessories (plants, utensil crocks, paper towels) and counter-top fixtures
        (toasters, coffee machines, toaster ovens, cooktops, sinks). Wall cabinets and hoods sit above
        the band, base cabinets and drawers below it."""
        boxes = []
        for name, fixture in self.fixtures.items():
            if type(fixture).__name__ in self._NOT_DECOR_CLASSES or fixture_is_type(fixture, FixtureType.COUNTER):
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            if isinstance(fixture, Accessory):
                if points[:, 2].max() < top_z - .05:
                    continue
            elif points[:, 2].max() < top_z - .02 or points[:, 2].min() > top_z + .25:
                continue     # below the counter, or wall-mounted well above it
            boxes.append(dict(name=name, x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                              y0=float(points[:, 1].min()), y1=float(points[:, 1].max()),
                              kind=type(fixture).__name__))
        return boxes

    def _frame_offsets(self, fixture):
        """Candidate sideways offsets of the footprint on the fixture's largest region, and the frame data."""
        regions = fixture.get_reset_regions(self, top_size=self.FOOTPRINT)
        region_name, region = max(regions.items(), key=lambda kv: kv[1]['size'][0] * kv[1]['size'][1])
        rx, ry = float(region['offset'][0]), float(region['offset'][1])
        sx, sy = float(region['size'][0]), float(region['size'][1])
        top_z = float(fixture.pos[2] + fixture.size[2] / 2)
        cy = ry - sy / 2 + self.FOOTPRINT[1] / 2 + self.FOOTPRINT_FRONT_MARGIN
        slack = max(0., sx / 2 - self.FOOTPRINT[0] / 2)
        offsets = [0.] + [s * d for d in np.arange(.10, slack + 1e-9, .10) for s in (-1, 1)]
        return dict(region_name=region_name, rx=rx, ry=ry, sx=sx, sy=sy, top_z=top_z, cy=cy,
                    yaw=float(fixture.rot), offsets=offsets, decor=self._decor_boxes(top_z))

    def _find_frame_offset(self, fixture, frame=None):
        """The first sideways offset whose footprint touches no decor, with the blocked ones; None if all are."""
        frame = frame or self._frame_offsets(fixture)
        blocked = []
        for dx in frame['offsets']:
            box = self._footprint_world_box([frame['rx'] + dx, frame['cy']], frame['yaw'], fixture)
            hits = [d['name'] for d in frame['decor'] if self._boxes_overlap(box, d)]
            if not hits:
                return float(dx), blocked
            blocked.append((float(dx), hits))
        return None, blocked

    def _frame_clearance_m(self, fixture, frame, dx):
        """Free distance in metres between the footprint at ``dx`` and the nearest decor on the surface.

        ``None`` when the surface carries no decor at all. Negative when the boxes
        overlap; the :meth:`_boxes_overlap` margin is *not* subtracted, so the
        value is the plain gap the layout gate reports as remaining clearance.
        """
        box = self._footprint_world_box([frame['rx'] + dx, frame['cy']], frame['yaw'], fixture)
        gaps = []
        for d in frame['decor']:
            gx = max(d['x0'] - box['x1'], box['x0'] - d['x1'])
            gy = max(d['y0'] - box['y1'], box['y0'] - d['y1'])
            gaps.append(max(gx, gy))
        return round(float(min(gaps)), 5) if gaps else None

    def _footprint_world_box(self, center_local, yaw, fixture):
        half = np.array(self.FOOTPRINT) / 2
        corners = [self.fixture_local_to_world(fixture, [center_local[0] + sx * half[0], center_local[1] + sy * half[1], 0.])
                   for sx in (-1, 1) for sy in (-1, 1)]
        corners = np.asarray(corners)
        return dict(x0=corners[:, 0].min(), x1=corners[:, 0].max(), y0=corners[:, 1].min(), y1=corners[:, 1].max())

    @staticmethod
    def _boxes_overlap(a, b, margin=.02):
        return not (a['x1'] + margin < b['x0'] or b['x1'] + margin < a['x0']
                    or a['y1'] + margin < b['y0'] or b['y1'] + margin < a['y0'])

    def place_work_frame(self):
        """Choose the task frame on the work surface.

        The frame origin is the footprint centre on the top surface; +x runs
        along the front of the surface, +y away from the robot (RoboCasa's
        fixture convention: the front face is local -y). The footprint sits at
        the front of the largest free top region and slides sideways to avoid
        counter decor; decor is never removed.

        **No fallback.** Up to v1, when every offset was blocked this took the
        least-bad one (``min(blocked_by, ...)``) and left the frame under the
        decor; the reset-time clash check only sees where the props *start*, so
        the instance passed G1 and failed at the teleport certificate or in a
        rollout instead. It now raises :class:`WorkFrameRefused` naming the
        layout, the style, the surface (and the other surfaces
        :meth:`_choose_work_fixture` tried) and the blockers, so the build fails
        at G1 with that message and ``mint`` walks on to the next seed.
        ``work['decor_overlap']`` therefore stays empty on every built instance.
        """
        fixture = self.work_fixture
        frame = self._frame_offsets(fixture)
        chosen, blocked_by = self._find_frame_offset(fixture, frame)
        decor_overlap = []
        if chosen is None:
            hits = sorted({n for _, names in blocked_by for n in names})
            others = [(name, names) for name, names in self._work_surface_survey if name != fixture.name]
            detail = (f'{fixture.name} region {frame["region_name"]}: all {len(blocked_by)} offsets blocked by '
                      f'{", ".join(hits) or "decor"}')
            if others:
                detail += '; also tried ' + '; '.join(f'{name} blocked by {", ".join(names) or "decor"}'
                                                      for name, names in others[:3])
            raise self._refuse_frame('no decor-free {footprint} m work frame',
                                     surfaces=[fixture.name] + [name for name, _ in others],
                                     blockers=sorted(set(hits).union(n for _, names in others for n in names)),
                                     detail=detail)
        rx, sx, sy, top_z, yaw, cy = frame['rx'], frame['sx'], frame['sy'], frame['top_z'], frame['yaw'], frame['cy']
        center_local = [rx + chosen, cy]
        center_world = self.fixture_local_to_world(fixture, [center_local[0], center_local[1], fixture.size[2] / 2])
        self.work = dict(fixture=fixture.name, region=frame['region_name'], top_z=top_z, yaw=yaw,
                         center_local=[float(v) for v in center_local],
                         center_world=[float(center_world[0]), float(center_world[1]), top_z],
                         footprint=list(self.FOOTPRINT), offset_along=float(chosen), region_size=[sx, sy],
                         region_front_local=float(frame['ry'] - sy / 2), decor_blocked=blocked_by,
                         decor_overlap=list(decor_overlap), decor_on_surface=[d['name'] for d in frame['decor']],
                         decor_clearance_m=self._frame_clearance_m(fixture, frame, chosen))
        self.task_spec['work'] = deepcopy(self.work)
        self.task_spec['supports'][fixture.name] = {
            'bounds_xy': [float(center_world[0] - sx / 2), float(center_world[0] + sx / 2),
                          float(center_world[1] - sy / 2), float(center_world[1] + sy / 2)],
            'height_z': top_z, 'fixture': fixture.name, 'rotation_z_rad': yaw}
        self._align_robot_base_with_frame()
        return self.work

    def _align_robot_base_with_frame(self):
        """Stand the robot in front of the task frame instead of the fixture centre.

        RoboCasa anchors the base on the fixture; the frame may have slid sideways
        to dodge decor (0.6 m on some layouts, leaving props outside the fixed
        cameras). Only the component along the surface changes, so RoboCasa keeps
        its own standing distance, orientation and collision retries. Idempotent.
        """
        anchor = getattr(self, 'init_robot_base_pos_anchor', None)
        if anchor is None or not self.work:
            return
        anchor = np.asarray(anchor, float).copy()
        along = yaw_matrix(self.work['yaw'])[:, 0]
        shift = float(np.dot(np.asarray(self.work['center_world'], float)[:2] - anchor[:2], along[:2]))
        anchor[:2] += shift * along[:2]
        self.init_robot_base_pos_anchor = anchor.tolist()
        self.work['robot_base_shift_m'] = round(shift, 4)
        self.task_spec['work']['robot_base_shift_m'] = round(shift, 4)

    # ---- the frame's front stance: can the base stand there, and get there from its reset pose? ----------
    def _floor_map(self):
        """The floor the base drives on (:func:`observe.floor_map`): room bounds and the footprints of every
        fixture, appliance, stool and floor decor the base cannot cross. Overridden by the CPU tests."""
        from roboquest import observe
        return observe.floor_map(self)

    def _reset_base_pose(self):
        """(world xy, yaw) where RoboCasa stands the base at reset: the aligned anchor once ``_load_model``
        has it, else the build-time prediction (:func:`observe.expected_base_pose`). Overridden by the tests."""
        anchor = getattr(self, 'init_robot_base_pos_anchor', None)
        ori = getattr(self, 'init_robot_base_ori_anchor', None)
        if anchor is not None and ori is not None:
            yaw = float(np.asarray(ori, float).ravel()[-1])
            return np.asarray(anchor, float)[:2].copy(), _wrap(yaw)
        from roboquest import observe
        xy, yaw = observe.expected_base_pose(self, at_build=True)
        return np.asarray(xy, float)[:2].copy(), _wrap(float(yaw))

    def frame_stance(self):
        """Can the base stand in front of the work frame, and get there from its reset pose?

        The front stance is the base rectangle (:func:`observe.base_rectangle`) at
        the frame centre on the reset pose's standoff line, facing the frame's +y:
        where RoboCasa stands the base once the frame alignment has slid it, and
        the line the skills (``Skills.stand``: heading kept, never closer to the
        surface than at reset) drive along. It must clear the floor map
        (:func:`observe.stance_gate`: fixtures, appliances, stools, floor decor,
        the room; touching by ``STANCE_TOUCH_M``), as must the reset pose itself,
        and the reset pose must be on that line facing the frame (heading within
        ``STANCE_HEADING_TOL_RAD`` of the frame's +y, base frame at least its own
        forward extent in front of the footprint's front edge, alongside the
        footprint): a base RoboCasa spawns behind or beside the surface (an island
        whose front abuts another counter, layout 23) never reaches the front
        stance without turning, which the skills do not do; the stance is then
        judged at :data:`observe.STANDOFF_M` for the report.

        The rest of the footprint's front is swept for information only:
        ``samples`` holds one base rectangle per ``STANCE_STEP_M`` of frame x over
        the footprint's width and ``free_range`` the contiguous interval of
        base-centre offsets around the frame centre that clear the floor, the
        lateral slide the skills have there. Whether a task's props at the
        footprint's ends are workable from that range is left to the oracle
        certificate: the interim puzzle registry solved every instance on layouts
        13, 19 and 46, whose ends this sweep marks. Returns a dict; ``passed`` is
        False with ``reason`` ``'blocked'`` (``blockers``: fixture name -> overlap
        depth in metres at the front stance, ``outside_room``) or
        ``'disconnected'``.
        """
        from roboquest import observe
        work = self.work
        floor = self._floor_map()
        reset_xy, reset_yaw = self._reset_base_pose()
        heading = _wrap(float(work['yaw']) + math.pi / 2)
        rotation = yaw_matrix(work['yaw'])[:2, :2]
        origin = np.asarray(work['center_world'], float)[:2]
        reset_local = rotation.T @ (np.asarray(reset_xy, float) - origin)
        half_w = float(self.FOOTPRINT[0]) / 2
        front_y = -(float(self.FOOTPRINT[1]) / 2 + self.FOOTPRINT_FRONT_MARGIN)   # footprint front edge, frame y
        standoff = float(front_y - reset_local[1])                                # base frame beyond that edge
        heading_error = abs(_wrap(float(reset_yaw) - heading))
        facing = heading_error <= STANCE_HEADING_TOL_RAD
        in_front = (standoff >= observe.BASE_AHEAD_M - observe.STANCE_TOUCH_M
                    and abs(float(reset_local[0])) <= half_w + observe.BASE_HALF_WIDTH_M)
        connected = bool(facing and in_front)
        line_y = float(reset_local[1]) if connected else front_y - observe.STANDOFF_M
        required = {0.} | ({round(float(reset_local[0]), 3)} if connected else set())
        xs = sorted({round(float(x), 3) for x in np.arange(-half_w, half_w + 1e-9, STANCE_STEP_M)}
                    | required | {round(half_w, 3)})
        blockers, outside, samples = {}, False, []
        for x in xs:
            xy = origin + rotation @ np.array([x, line_y])
            gate = observe.stance_gate(xy, heading, floor)
            samples.append(dict(x=x, passed=bool(gate['passed']), in_room=bool(gate['in_room']),
                                blockers=list(gate['blockers'])))
            if x in required:
                outside = outside or not gate['in_room']
                corners = observe.base_rectangle(xy, heading)
                for name in gate['blockers']:
                    blockers[name] = max(blockers.get(name, 0.), _poly_penetration(corners, floor['blockers'][name]))
        free_range = None
        passed_x = sorted(s['x'] for s in samples if s['passed'])
        if 0. in passed_x:
            lo = hi = passed_x.index(0.)
            while lo > 0 and passed_x[lo] - passed_x[lo - 1] <= STANCE_STEP_M + 1e-6:
                lo -= 1
            while hi < len(passed_x) - 1 and passed_x[hi + 1] - passed_x[hi] <= STANCE_STEP_M + 1e-6:
                hi += 1
            free_range = [passed_x[lo], passed_x[hi]]
        passed = connected and not blockers and not outside
        if connected:
            where = ''
        elif not in_front:
            where = 'behind' if reset_local[1] > front_y else 'beside'
        else:
            where = 'facing away from'
        return dict(passed=bool(passed), reason='' if passed else ('disconnected' if not connected else 'blocked'),
                    heading=float(heading), standoff_m=round(standoff, 4), line_y=round(line_y, 4), where=where,
                    reset=dict(xy=[float(v) for v in reset_xy], yaw=float(reset_yaw),
                               frame_xy=[round(float(v), 4) for v in reset_local],
                               heading_error_deg=round(math.degrees(heading_error), 2), in_front=bool(in_front)),
                    blockers={name: round(depth, 4) for name, depth in sorted(blockers.items(), key=lambda kv: -kv[1])},
                    outside_room=bool(outside), free_range=free_range, samples=samples)

    def _check_frame_stance(self):
        """Refuse the frame (:class:`WorkFrameRefused`, reason ``stance_blocked`` / ``stance_disconnected``) when
        :meth:`frame_stance` fails; record the summary in ``work['stance']`` either way."""
        if not self.work:
            return None
        report = self.frame_stance()
        summary = {key: deepcopy(report[key]) for key in ('passed', 'reason', 'standoff_m', 'free_range', 'blockers',
                                                          'outside_room', 'reset')}
        self.work['stance'] = summary
        if isinstance(getattr(self, 'task_spec', None), dict) and isinstance(self.task_spec.get('work'), dict):
            self.task_spec['work']['stance'] = deepcopy(summary)
        if report['passed']:
            return report
        hits = ', '.join(f'{name} by {depth:.3f} m' for name, depth in report['blockers'].items())
        if report['outside_room']:
            hits = (hits + '; ' if hits else '') + 'the room ends'
        if report['reason'] == 'disconnected':
            reset = report['reset']
            message = 'the base cannot reach the front of the {footprint} m work frame'
            detail = (f"RoboCasa stands it {report['where']} the frame (frame x {reset['frame_xy'][0]:+.2f}, "
                      f"y {reset['frame_xy'][1]:+.2f} m, heading {reset['heading_error_deg']:.0f} deg off) and "
                      f"the skills keep heading" + (f'; the front stance overlaps {hits}' if hits else ''))
            reason = 'stance_disconnected'
        else:
            message = 'the base cannot stand in front of the {footprint} m work frame'
            detail = f"stance rectangle at the frame centre (standoff {report['standoff_m']:.2f} m) overlaps {hits}"
            reason = 'stance_blocked'
        raise self._refuse_frame(message, surfaces=[self.work['fixture']], blockers=sorted(report['blockers']),
                                 detail=detail, reason=reason)

    def frame_to_world(self, local):
        """Task-frame (x along the front, y away from the robot, z up from the top) to world."""
        local = np.asarray(local, float)
        origin = np.asarray(self.work['center_world'], float)
        return origin + yaw_matrix(self.work['yaw']) @ local

    def add_frame_body(self, name, local_xy=(0., 0.), z=0.):
        """A fixed body at a task-frame position, rotated with the surface; append props to it."""
        pos = self.frame_to_world([local_xy[0], local_xy[1], z])
        self._frame_bodies = getattr(self, '_frame_bodies', []) + [name]
        return ET.SubElement(self.model.worldbody, 'body', name=name, pos=_vec(pos),
                             quat=_vec(yaw_quat_wxyz(self.work['yaw'])))

    # ---- physics step ------------------------------------------------------------------------
    PHYSICS_TIMESTEP = None   # seconds; set e.g. 0.001 for tasks with balls or small sliding parts

    def initialize_time(self, control_freq):
        """robosuite takes the physics step from its macros; honour the task's own value when set."""
        super().initialize_time(control_freq)
        if self.PHYSICS_TIMESTEP:
            self.model_timestep = float(self.PHYSICS_TIMESTEP)
            self.sim.model.opt.timestep = self.model_timestep

    # ---- protocol: outcomes, progress and the event log (spec 1.1, 1.4, 1.5) -------------------
    def task_progress(self):
        """Progress in [0, 1] read off the current state (spec 1.4).

        0 at reset, 1 exactly when the success predicate holds. It never depends
        on the trajectory, so irreversible damage simply shows up as low
        progress. Tasks override it with their own partial credit.
        """
        return 1. if self._current_score()['success'] else 0.

    def collateral_checks(self):
        """``{name: {'bad': bool, 'object': str, 'cost': float}}``, evaluated every tick (spec 1.5).

        The base logs ``collateral`` when a check turns bad and ``repaired``
        when it turns good again. Keep the checks cheap: state reads only.
        """
        return {}

    def log_event(self, kind, obj=None, cost=0., **extra):
        """Append one ``{tick, kind, object, cost}`` record to the episode's event log."""
        event = dict(tick=int(self.timestep), kind=str(kind),
                     object=None if obj is None else str(obj), cost=float(cost), **extra)
        self.events.append(event)
        return event

    @staticmethod
    def _progress_value(value):
        """A task's progress, clipped into [0, 1]; a broken value reads as 0."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.
        return 0. if not math.isfinite(value) else min(1., max(0., value))

    def _prop_bodies(self):
        """``{name: body id}`` of the movable task props the generic checks watch.

        Every free-jointed body that is not part of the robot: the props a task
        can lose off the surface, whatever it called them. Tasks append their
        props in different ways (``self.objects``, or straight into the
        worldbody as the puzzle box does), so the model, not a task-side
        register, is the source of truth; the task's own name is used as the
        label where there is one. Welded parts (a tray, a board) and the parts
        of a rigid mechanism (bolts and lids on slide or hinge joints) cannot
        leave the surface and are left out. Resolved once per episode.
        """
        model = self.sim.model
        labels = {}
        for name, obj in (getattr(self, 'objects', {}) or {}).items():
            try:
                labels[int(model.body_name2id(obj.root_body))] = str(name)
            except Exception:
                continue
        bodies = {}
        for body in range(model.nbody):
            if int(model.body_jntnum[body]) != 1:
                continue
            if int(model.jnt_type[int(model.body_jntadr[body])]) != mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = model.body_id2name(body) or f'body{body}'
            if name.startswith(self.ROBOT_PREFIXES):
                continue
            bodies[labels.get(body, name)] = body
        return bodies

    def _prop_heights(self, bodies):
        positions = self.sim.data.body_xpos
        return {name: float(positions[body][2]) for name, body in bodies.items()}

    def _start_events(self):
        """Baseline for the generic checks, taken at the first tick so the task has placed its props."""
        top = float(self.work['top_z']) if self.work else None
        bodies = self._prop_bodies()
        heights = self._prop_heights(bodies)
        on_surface = sorted(name for name, z in heights.items()
                            if top is not None and z >= top - self.SURFACE_START_TOL_M)
        self._event_state = dict(top_z=top, bodies=bodies, on_surface=on_surface, left=set(), bad=set(),
                                 cost={})
        return self._event_state

    def _log_events(self):
        """One generic tick of the event log: props that left the surface, then the task's checks."""
        state = self._event_state or self._start_events()
        top = state['top_z']
        if top is not None and len(state['left']) < len(state['on_surface']):
            heights = self._prop_heights(state['bodies'])
            for name in state['on_surface']:
                if name in state['left'] or heights.get(name, top) >= top - self.SURFACE_DROP_M:
                    continue
                state['left'].add(name)
                self.log_event('left_surface', name, 1.)
        checks = self.collateral_checks() or {}
        costs = state.setdefault('cost', {})     # last logged total of every check that is bad
        bad = {name for name, check in checks.items() if check.get('bad')}
        for name in sorted(bad):
            check = checks[name]
            cost = float(check.get('cost') or 0.)
            if name not in state['bad']:
                self.log_event('collateral', check.get('object'), cost, check=str(name))
            elif cost > costs.get(name, cost) + 1e-9:
                # A running total that rose (a second cube in the wrong bin): log the increase, so the
                # collateral events of an episode sum to its cost.
                self.log_event('collateral', check.get('object'), cost - costs[name], check=str(name))
            elif cost < costs.get(name, cost) - 1e-9:
                # Partly repaired while still bad: log the decrease.
                self.log_event('repaired', check.get('object'), costs[name] - cost, check=str(name))
            costs[name] = cost
        for name in sorted(state['bad'] - bad):
            check = checks.get(name) or {}
            costs.pop(name, None)
            self.log_event('repaired', check.get('object'), check.get('cost') or 0., check=str(name))
        state['bad'] = bad

    def _reset_internal(self):
        super()._reset_internal()
        self.events = []
        self._event_state = None

    def _post_action(self, action):
        """After the mixin's Submit check: log the tick's events and freeze progress at the press."""
        reward, done, info = super()._post_action(action)
        self._log_events()
        if self.submission is not None and 'progress' not in self.submission:
            self.submission['progress'] = self._progress_value(self.task_progress())
        return reward, done, info

    def evaluate_success(self, states=None):
        """The mixin's frozen score plus the episode ``outcome`` (spec 1.1) and ``progress`` (spec 1.4)."""
        result = super().evaluate_success(states)
        if result.get('submitted'):
            frozen = (self.submission or {}).get('progress')
            result['outcome'] = 'success' if result.get('success') else 'wrong_submit'
            result['progress'] = self._progress_value(self.task_progress() if frozen is None else frozen)
        else:
            result['outcome'] = 'timeout' if self.timed_out else None
            result['progress'] = self._progress_value(self.task_progress())
        return result

    # ---- generic scene gate (G1): task props must not penetrate the kitchen or each other -------
    ROBOT_PREFIXES = ('robot0', 'gripper0', 'mount0', 'mobilebase0', 'base0')

    def task_body_ids(self):
        """Ids of every body belonging to the task: frame bodies and task objects with their descendants."""
        model = self.sim.model
        roots = set()
        for name in getattr(self, '_frame_bodies', []):
            try:
                roots.add(model.body_name2id(name))
            except Exception:
                continue
        for obj in (getattr(self, 'objects', {}) or {}).values():
            try:
                roots.add(model.body_name2id(obj.root_body))
            except Exception:
                continue
        ids = set()
        for body in range(model.nbody):
            b = body
            while b > 0:
                if b in roots:
                    ids.add(body)
                    break
                b = int(model.body_parentid[b])
        return ids

    def prop_clashes(self, depth_m=.002, prop_depth_m=.005):
        """Penetrating contacts at the current state between a task prop and the kitchen (fixtures, decor,
        appliances), or between two props (deeper than ``prop_depth_m``). Robot contacts are ignored.
        Returns a list of dicts; empty means the scene is physically clean."""
        model, data = self.sim.model, self.sim.data
        task = self.task_body_ids()
        out = []
        for i in range(data.ncon):
            con = data.contact[i]
            b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
            n1, n2 = model.body_id2name(b1) or '', model.body_id2name(b2) or ''
            if n1.startswith(self.ROBOT_PREFIXES) or n2.startswith(self.ROBOT_PREFIXES):
                continue
            t1, t2 = b1 in task, b2 in task
            if not (t1 or t2):
                continue
            depth = -float(con.dist)
            kind = 'prop-prop' if (t1 and t2) else 'prop-kitchen'
            if depth > (prop_depth_m if kind == 'prop-prop' else depth_m):
                out.append(dict(kind=kind, depth_mm=round(1000 * depth, 2),
                                geom1=model.geom_id2name(con.geom1), geom2=model.geom_id2name(con.geom2),
                                body1=n1, body2=n2))
        out.sort(key=lambda c: -c['depth_mm'])
        return out

    def add_submit_button(self, local_xy):
        pos = self.frame_to_world([local_xy[0], local_xy[1], 0.])
        self.submit_xy = (float(pos[0]), float(pos[1]))
        button = self._add_submit_button(self.model.worldbody, self.model.asset, self.work['top_z'])
        button['frame_local_xy'] = [float(local_xy[0]), float(local_xy[1])]
        self.task_spec['submit_button'] = button
        return button

    # ---- scene protocol -----------------------------------------------------
    def build_task(self):
        """Called by VisionKitchenScene._create_objects. Subclasses call place_work_frame() first."""
        raise NotImplementedError

    def setup_task_references(self):
        """Resolve ids/addresses after compilation (called from _setup_references)."""

    def task_snapshot(self):
        """Task-specific private state for the evaluator trace."""
        return {}

    def _current_score(self, states=None):
        raise NotImplementedError

    def _setup_references(self):
        super()._setup_references()
        self._setup_submit()
        self.setup_task_references()

    def evaluator_snapshot(self, step=0):
        snapshot = VisionKitchenScene.evaluator_snapshot(self, step)
        snapshot.update(self.task_snapshot())
        snapshot['score'] = self.evaluate_success()
        snapshot['submission'] = deepcopy(self.submission)
        snapshot['roboquest'] = dict(events=deepcopy(self.events))   # logging only (spec 1.5)
        return snapshot

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['roboquest'] = dict(task=self.TASK_NAME, contract_version=self.CONTRACT_VERSION,
                                 generator_version=self.GENERATOR_VERSION,
                                 instance_id=self.instance.get('instance_id'),
                                 factors=deepcopy(self.instance.get('factors')),
                                 spec=deepcopy(self.spec), work=deepcopy(self.work),
                                 events=deepcopy(self.events))
        return meta


def make_env(task_cls, instance, image_size=512, gpu=0, horizon=None, render=True, lean=None, camera_obs=None):
    """Construct a task env the way RoboCasa's create_env does, from a frozen instance.

    ``horizon=None`` takes the task's own budget (``BUDGET_TICKS``, spec 1.2);
    nothing about it reaches the policy. ``render=False`` builds physics only
    (no EGL context) for CPU gates and tests.

    ``lean`` (default ``not render``) builds the smaller worker of ``lean.py``:
    MuJoCo's compiler state is released after every compile, and a headless
    build also swaps the asset-pack textures for flat placeholders. A rendered
    build never swaps textures, whatever ``lean`` says. Pass ``lean=False`` for
    the untouched build (the memory identity check) or when a caller needs
    ``sim.model.get_xml()`` after the first reset instead of ``env.scene_xml()``.
    """
    horizon = int(task_cls.BUDGET_TICKS if horizon is None else horizon)
    camera_obs = bool(render) if camera_obs is None else bool(camera_obs)
    if camera_obs and not render:
        raise ValueError('camera observations require render=True')
    controller = load_composite_controller_config(controller=None, robot='PandaOmron')
    env = robosuite.make(
        env_name=task_cls.__name__, robots='PandaOmron', controller_configs=controller,
        instance=instance, seed=int(instance['seed']),
        layout_ids=int(instance['layout_id']), style_ids=int(instance['style_id']),
        camera_names=list(CAMERAS), camera_widths=image_size, camera_heights=image_size,
        has_renderer=False, has_offscreen_renderer=render, use_camera_obs=camera_obs,
        use_object_obs=True, camera_depths=False, ignore_done=True, clutter_mode=0,
        robot_spawn_deviation_pos_x=0, robot_spawn_deviation_pos_y=0, robot_spawn_deviation_rot=0,
        initialization_noise=None, render_gpu_device_id=gpu, horizon=horizon)
    lean_build = (not render) if lean is None else bool(lean)
    if lean_build and isinstance(env, RoboQuestKitchen):   # fakes in tests pass through untouched
        if not render:
            env.set_xml_processor(lean_mod.flatten_asset_textures)
        env.release_after_compile = True
    return env
