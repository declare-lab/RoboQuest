"""Shared motion skills, checkpoint/retry harness and headless record / rendered replay for RoboQuest oracles.

Built on :class:`KeyedFitMotor` (bounded world-frame OSC servo with the native
PandaOmron base) and made frame-aware through the task's work frame
(``env.frame_to_world``, ``env.work``): every skill takes world targets, and
the natural hand-down orientation follows the base heading rather than a
fixed world matrix (fixed world-frame grasp poses were the leading cause of
Panda wrist/elbow joint-limit failures in the study-room oracles).

Harness semantics
-----------------
* **One step.** Every physical action goes through :meth:`Skills.step`, which
  records it in :attr:`Skills.actions` (tick ``i`` holds the action applied at
  ``env.step`` number ``i``).
* **Checkpoint.** :class:`Checkpoint` saves MuJoCo's integration state (positions,
  velocities, actuator state, solver warm start, time), every part
  controller's plain state (OSC goal, origin and reference, base and gripper
  velocity goals, goal-update mode), the Panda gripper's integrated command,
  the robot's recent-value buffers and the episode counters (timestep,
  cur_time, submission). Restoring is bit-exact, so re-running the same
  actions reproduces the same trajectory.
* **Attempt.** :meth:`Skills.attempt` (decorator) and :meth:`Skills.run_attempt`
  run a skill up to ``1 + retries`` times; a failed postcondition
  (:class:`SkillFailure` or any ``RuntimeError`` raised by a waypoint) restores
  the checkpoint, truncates the action log and the open segments to the
  checkpoint tick, and retries with the parameters returned by ``jitter``.
  The log therefore holds the successful branch only, so a replay stays valid.
* **Replay.** :func:`save_actions` writes the log (plus the segment captions and
  the final poses of every movable body); :func:`replay` re-executes it on a
  fresh env, checks the final poses (5 mm / 3 deg by default) and can write a
  20 fps video of the three policy cameras side by side with the segment
  captions burned in.
"""
from contextlib import contextmanager
from copy import deepcopy
import functools
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.keyed_fit_motor import KeyedFitMotor
from roboquest.harness.contract import CAMERAS
from roboquest.kitchen import yaw_matrix

DOWN = np.diag([-1., 1., -1.])   # KeyedFitMotor's tool-down matrix: finger axis along world -x
GRASP_AHEAD_M = .47              # top-down grasp anchor distance ahead of the base frame (room search: 0.58 hit limits)
CARRY_ABOVE_TOP_M = .24          # carry height above the work surface
CARRY_AHEAD_M = .34              # carry point ahead of the base frame
ROBOT_PREFIXES = ('robot', 'gripper', 'mobilebase')
ROBOT_BUFFERS = ('recent_qpos', 'recent_actions', 'recent_torques', 'recent_ee_forcetorques', 'recent_ee_pose',
                 'recent_ee_vel', 'recent_ee_vel_buffer', 'recent_ee_acc')
DEJAVU_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ACTIONS_FORMAT = 'roboquest-actions-v1'
STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)   # everything mj_step integrates from

# Room-search drawer/door recipe constants (ported; see open_drawer / open_door).
DRAWER_FRONT_UP = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])  # north-facing bar handle, tool x up
DRAWER_PULL_M = .40
DRAWER_VIEW_M = .71
UNHOOK_M = .10
DOOR_TILT_RAD = .90
DOOR_GRIP_BELOW_TOP_M = .05
DOOR_LOW_BAR_Z = .85
DOOR_MID_BAR_Z = .70
DOOR_OPEN_RAD = 1.30
DOOR_PULL_RAD = .70
DOOR_MIN_OPEN_RAD = 1.0


class Submitted(Exception):
    """Raised by the action that physically pressed Submit: the episode is over."""


class SkillFailure(RuntimeError):
    """A skill's postcondition failed (fail fast; the attempt harness restores and retries)."""

    def __init__(self, skill, reason, **details):
        super().__init__(f'{skill}: {reason}')
        self.skill, self.reason, self.details = skill, reason, details


class AttemptFailed(SkillFailure):
    """All attempts of one skill failed; ``errors`` lists every attempt's reason."""

    def __init__(self, name, errors):
        super().__init__(name, f"{len(errors)} attempt(s) failed; last: {errors[-1]['error']}", errors=errors)
        self.errors = errors


# ---- plain-state capture ----------------------------------------------------------------------
_PLAIN = (int, float, bool, str, type(None), np.generic, np.ndarray)


def _is_plain(value, depth=0):
    if isinstance(value, _PLAIN):
        return True
    if depth > 3:
        return False
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int)) and _is_plain(v, depth + 1) for k, v in value.items())
    return False


def plain_state(obj, skip=('sim',)):
    """Deep copies of an object's plain data attributes (arrays, numbers, strings and containers of those)."""
    return {k: deepcopy(v) for k, v in vars(obj).items() if k not in skip and _is_plain(v)}


def restore_plain(obj, state):
    for key, value in state.items():
        setattr(obj, key, deepcopy(value))


def movable_body_poses(env):
    """World pose of every non-robot body that has a joint (parts, items, doors, the Submit cap)."""
    model, data = env.sim.model, env.sim.data
    raw_m, raw_d = model._model, data._data
    poses = {}
    for bid in range(raw_m.nbody):
        name = model.body_id2name(bid) or ''
        if not name or name.startswith(ROBOT_PREFIXES) or int(raw_m.body_jntnum[bid]) == 0:
            continue
        poses[name] = dict(pos=np.asarray(raw_d.xpos[bid]).tolist(), quat=np.asarray(raw_d.xquat[bid]).tolist())
    return poses


def axis_keeping_order(ranked):
    """Reorder :meth:`Skills.wrist_alternatives`'s ranking so the candidates that keep the requested finger
    axis (``k`` 0 and 2, the 180-degree flips) come first, best margin first, then the perpendicular pair
    (``k`` 1 and 3) the same way. Measured need (SKILLS-2): ``move_aside`` asked for the fingers across a
    6.5 cm occluder and the margin ranking turned them across its 22.75 cm side, so the hand stood on the
    box top 4.4 cm above the grasp point; the plate and bowl rim pinches need the radial axis the same way."""
    keep = [r for r in ranked if r['k'] in (0, 2)]
    turn = [r for r in ranked if r['k'] not in (0, 2)]
    return keep + turn


def _unit(vector):
    vector = np.asarray(vector, float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError('zero direction')
    return vector / norm


class Checkpoint:
    """A bit-exact rewind point of one episode (valid until the env is reset again)."""

    def __init__(self, skills, name=''):
        env = skills.env
        robot = env.robots[0]
        self._skills = skills
        self.name, self.tick = name, len(skills.actions)
        # MuJoCo's integration state: time, qpos, qvel, act, ctrl, applied forces, mocap, user data, warm start.
        model, data = env.sim.model._model, env.sim.data._data
        self.state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
        mujoco.mj_getState(model, data, self.state, STATE_SPEC)
        self.warmstart = np.array(data.qacc_warmstart, copy=True)
        self.controllers = {k: plain_state(c) for k, c in robot.part_controllers.items()}
        self.grippers = {arm: np.array(g.current_action, copy=True) for arm, g in robot.gripper.items()
                         if getattr(g, 'current_action', None) is not None}
        self.buffers = {k: deepcopy(getattr(robot, k)) for k in ROBOT_BUFFERS if hasattr(robot, k)}
        # The event log rewinds with the physics: a branch that is thrown away must not leave
        # its collateral behind. ``_event_state`` is the per-object memory that dedups the log,
        # so it rewinds too, or the replayed branch would skip events it logged the first time.
        self.episode = dict(timestep=env.timestep, cur_time=env.cur_time, done=env.done,
                            submission=deepcopy(getattr(env, 'submission', None)),
                            timed_out=getattr(env, 'timed_out', False),
                            events=deepcopy(getattr(env, 'events', None)),
                            event_state=deepcopy(getattr(env, '_event_state', None)),
                            obs_cache=deepcopy(getattr(env, '_obs_cache', {})))
        self.skills = dict(steps=skills.steps, gripper=skills.gripper, phases=len(skills.phases),
                           segments=len(skills.segments), obs=skills.obs)

    def restore(self):
        skills, env = self._skills, self._skills.env
        robot = env.robots[0]
        from_tick = len(skills.actions)
        model, data = env.sim.model._model, env.sim.data._data
        mujoco.mj_setState(model, data, self.state, STATE_SPEC)
        # Refresh the derived quantities (poses, contacts) that skills read, then re-impose the saved warm
        # start that mj_forward overwrote: the next step then solves from exactly the checkpoint's inputs.
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = self.warmstart
        for key, state in self.controllers.items():
            restore_plain(robot.part_controllers[key], state)
        for arm, action in self.grippers.items():
            robot.gripper[arm].current_action = action.copy()
        for key, value in self.buffers.items():
            setattr(robot, key, deepcopy(value))
        env.timestep, env.cur_time, env.done = self.episode['timestep'], self.episode['cur_time'], self.episode['done']
        if hasattr(env, 'submission'):
            env.submission = deepcopy(self.episode['submission'])
            env.timed_out = self.episode['timed_out']
        if hasattr(env, 'events'):
            env.events = deepcopy(self.episode['events'])
        if hasattr(env, '_event_state'):
            env._event_state = deepcopy(self.episode['event_state'])
        if hasattr(env, '_obs_cache'):
            env._obs_cache = deepcopy(self.episode['obs_cache'])
        skills.steps, skills.gripper, skills.obs = self.skills['steps'], self.skills['gripper'], self.skills['obs']
        del skills.actions[self.tick:]
        del skills.phases[self.skills['phases']:]
        skills._truncate_segments(self.tick, self.skills['segments'])
        skills.restores.append(dict(name=self.name, tick=self.tick, from_tick=from_tick))
        return self


class Skills(KeyedFitMotor):
    """Frame-aware motion skills with an action log, checkpoints and retries.

    Every skill is a world-frame recipe; the task frame is available through
    :meth:`to_world` / :meth:`to_frame` / :meth:`direction_to_world`. Skills
    raise :class:`SkillFailure` when their postcondition fails within a short
    tick budget, so :meth:`attempt` can restore and retry with jitter.
    """

    def __init__(self, env, horizon=None, anchor=None, seed=0):
        super().__init__(env, None, horizon=int(horizon or env.horizon))
        self.work = env.work
        self.frame_yaw = float(env.work['yaw'])
        self.top_z = float(env.work['top_z'])
        self.rotation = yaw_matrix(self.frame_yaw)          # task frame -> world
        self.dim = int(env.action_dim)
        self.actions, self.segments, self.attempts, self.restores = [], [], [], []
        self.gripper = -1.
        self.anchor = anchor
        self.rng = np.random.default_rng(seed)
        base_xy, base_yaw = self.base_pose()
        self.reset_base = dict(xy=base_xy.tolist(), yaw=base_yaw)
        # The base spawns with its front at the work surface's edge: never drive closer than that.
        self.forward_limit_y = float(self.to_frame([base_xy[0], base_xy[1], 0.])[1])

    # ---- frames and state ----------------------------------------------------------------------
    @property
    def tick(self):
        return len(self.actions)

    def to_world(self, local):
        return self.env.frame_to_world(local)

    def to_frame(self, world):
        origin = np.asarray(self.work['center_world'], float)
        return self.rotation.T @ (np.asarray(world, float) - origin)

    def direction_to_world(self, vector):
        return self.rotation @ np.asarray(vector, float)

    def site_position(self, name):
        sim = self.env.sim
        return np.asarray(sim.data.site_xpos[sim.model.site_name2id(name)]).copy()

    def state(self, name=None):
        name = name or self.anchor
        if name is not None:
            return super().state(name)
        sim, robot = self.env.sim, self.env.robots[0]
        sid = robot.eef_site_id['right']
        return {'position': None, 'rotation': None, 'eef': sim.data.site_xpos[sid].copy(),
                'eef_rotation': sim.data.site_xmat[sid].reshape(3, 3).copy()}

    def base_pose(self):
        pos, rot = self.env.robots[0].part_controllers['base'].get_base_pose()
        return np.asarray(pos)[:2].copy(), float(math.atan2(rot[1, 0], rot[0, 0]))

    def facing(self):
        _, yaw = self.base_pose()
        return np.array([math.cos(yaw), math.sin(yaw)])

    def arm_joint_margin(self):
        """Smallest distance (rad) of any arm joint to its limit, and that joint's name."""
        robot, model = self.env.robots[0], self.env.sim.model
        if getattr(robot, '_ref_arm_joint_pos_indexes', None) is None:
            return None, None
        q = np.asarray(self.env.sim.data.qpos[robot._ref_arm_joint_pos_indexes], float)
        ids = [model.joint_name2id(j) for j in robot.robot_model.arm_joints]
        lo, hi = model.jnt_range[ids, 0], model.jnt_range[ids, 1]
        margins = np.minimum(q - lo, hi - q)
        index = int(np.argmin(margins))
        return float(margins[index]), robot.robot_model.arm_joints[index]

    def finger_gap(self):
        robot = self.env.robots[0]
        q = self.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes['right']]
        return float(q[0] - q[1])

    def grasped(self, geoms, fingers=False):
        """Both finger pads touch one of ``geoms``. With ``fingers`` the finger meshes count as well: they reach
        8.8 mm past the pads and 5-7 mm inward of them at the tip (robosuite panda_gripper.xml, measured world
        boxes), so a thick edge held between the fingers, as the inverted bowl's rim is, can rest on a finger
        without touching its pad (SKILLS-2 bowl probe: gap 45 mm, three finger contacts, no pad on one side)."""
        gripper = self.env.robots[0].gripper['right']
        if fingers:
            groups = [[f'{gripper.naming_prefix}finger1_pad_collision', f'{gripper.naming_prefix}finger1_collision'],
                      [f'{gripper.naming_prefix}finger2_pad_collision', f'{gripper.naming_prefix}finger2_collision']]
            return bool(self.env._check_grasp(groups, list(geoms)))
        return bool(self.env._check_grasp(gripper, list(geoms)))

    # ---- the one physical step ---------------------------------------------------------------
    def step(self, action):
        action = np.array(action, dtype=float, copy=True)
        if action.shape != (self.dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f'action must be a finite {self.dim}-vector')
        self.actions.append(action.copy())
        return self.env.step(action)

    def act(self, dp, dr, gripper, base=(0., 0., 0.), base_mode=False):
        self.gripper = float(gripper)
        super().act(dp, dr, gripper, base, base_mode)
        if getattr(self.env, 'submission', None) is not None:
            raise Submitted()

    # ---- orientations ---------------------------------------------------------------------------
    def hand_down(self, finger_yaw=None):
        """Tool z down with the finger axis at world yaw ``finger_yaw`` (default: the base's left, the
        Panda's home-configuration branch; the mirrored branch winds joint 7 toward its limit)."""
        if finger_yaw is None:
            _, yaw = self.base_pose()
            finger_yaw = yaw + np.pi / 2
        c, s = math.cos(finger_yaw), math.sin(finger_yaw)
        return np.array([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])

    def wrist_alternatives(self, base_rotation):
        """The four wrist yaws (90 deg apart about the tool z axis) ranked by the distance the wrist joint
        (joint 7) would keep from its limits; ties prefer the smaller turn from the current pose."""
        robot, model = self.env.robots[0], self.env.sim.model
        q7 = float(self.env.sim.data.qpos[robot._ref_arm_joint_pos_indexes[-1]])
        lo, hi = model.jnt_range[model.joint_name2id(robot.robot_model.arm_joints[-1])]
        current = self.state()['eef_rotation']
        ranked = []
        for k in range(4):
            candidate = np.asarray(base_rotation, float) @ Rotation.from_euler('z', k * np.pi / 2).as_matrix()
            rotvec = Rotation.from_matrix(candidate @ current.T).as_rotvec()
            delta = float(np.dot(rotvec, current[:, 2]))       # turn about the current tool z (joint 7 axis)
            q7c = q7 + delta
            ranked.append(dict(k=k, rotation=candidate, delta=delta, q7=float(q7c),
                               margin=float(min(q7c - lo, hi - q7c))))
        ranked.sort(key=lambda r: (-round(r['margin'], 2), abs(r['delta'])))
        return ranked

    def grasp_rotation(self, kind='top', approach=None, alternative=0, finger_yaw=None):
        """Top grasp (tool down) or side grasp (tool z along ``approach``), wrist yaw chosen among the
        four alternatives by joint-limit distance; ``alternative`` picks the next-best ones for retries.

        With an explicit ``finger_yaw`` the finger axis is part of the request (a rim pinched across its
        lip, a box pinched across its thin side), so the two candidates that keep that axis (the 180-degree
        flips, ``k`` 0 and 2) come first and the perpendicular pair only after them: ``alternative`` 0 and 1
        keep the axis, 2 and 3 turn it. Without ``finger_yaw`` the four are ranked by margin alone."""
        if kind == 'top':
            base = self.hand_down(finger_yaw)
        elif kind == 'side':
            z = _unit([approach[0], approach[1], 0.])
            x = np.array([0., 0., 1.])
            base = np.column_stack([x, np.cross(z, x), z])
        else:
            raise ValueError(f'unknown grasp kind {kind!r}')
        ranked = self.wrist_alternatives(base)
        if kind == 'top' and finger_yaw is not None:
            ranked = axis_keeping_order(ranked)
        return ranked[int(alternative) % 4]['rotation']

    # ---- arm skills ------------------------------------------------------------------------------
    def move(self, position, rotation=None, gripper=None, *, phase='move', ticks=140, tolerance=.005,
             required=True, name=None, scale=True):
        """Servo the hand to a world position (and rotation; ``None`` keeps the current one). ``scale`` is
        accepted for call sites written against the reach-certificate branch's speed factor (``scale=False``
        marks a duration that is a measurement); this motor has no factor, every duration runs as written."""
        if isinstance(position, (tuple, list)) and len(position) == 2 and np.shape(position[1]) == (3, 3):
            position, rotation = position
        rotation = self.state(name)['eef_rotation'].copy() if rotation is None else np.asarray(rotation, float)
        gripper = self.gripper if gripper is None else float(gripper)
        return super().move(name, np.asarray(position, float), rotation, gripper, phase,
                            ticks=int(ticks), tolerance=tolerance, required=required)

    def hold(self, ticks, gripper=None, rotation=None, *, phase='hold', required=False, name=None, scale=True):
        return self.move(self.state(name)['eef'], rotation, gripper, phase=phase, ticks=int(ticks),
                         tolerance=0., required=required, name=name)

    def carry(self, gripper=None, *, height=None, ahead=CARRY_AHEAD_M, phase='carry'):
        """Raise the hand above everything on the surface and tuck it ahead of the base before driving."""
        gripper = self.gripper if gripper is None else float(gripper)
        z = self.top_z + CARRY_ABOVE_TOP_M if height is None else float(height)
        state = self.state()
        base_xy, _ = self.base_pose()
        self.move([state['eef'][0], state['eef'][1], z], state['eef_rotation'], gripper, phase=phase + '_raise', ticks=120)
        target = np.r_[base_xy + self.facing() * ahead, z]
        self.move(target, self.hand_down(), gripper, phase=phase + '_retract', ticks=180)

    def grasp(self, target, kind='top', *, rotation=None, approach=None, alternative=0, above=.14, lower_ticks=140,
              close_ticks=25, min_gap=.004, name=None, geoms=None, lift=None, min_lift=.08, phase='grasp', fingers=False):
        """Top or side grasp at a world point. Postconditions: the fingers stopped on something
        (gap > ``min_gap``), both finger pads touch ``geoms`` when given, and the object rose by
        ``min_lift`` when ``lift`` is requested (needs ``name``). With ``fingers`` the finger meshes count
        as a hold too (:meth:`grasped`): a thin wall that leans, like the marked bowl's rim (31 degrees from
        vertical), stops the fingertips 5 mm before the pads reach it (MUGS-ORACLE bowl probe: gap 10.2 mm,
        pads clear, both finger meshes on the wall)."""
        target = np.asarray(target, float)
        if rotation is None:
            rotation = self.grasp_rotation(kind, approach, alternative)
        rotation = np.asarray(rotation, float)
        if kind == 'top':
            self.move(target + [0., 0., above], rotation, -1., phase=phase + '_above', ticks=200)
            self.move(target, rotation, -1., phase=phase + '_lower', ticks=lower_ticks)
        else:
            a = rotation[:, 2]
            self.move(target - a * .12 + [0., 0., .03], rotation, -1., phase=phase + '_pre', ticks=200)
            self.move(target, rotation, -1., phase=phase + '_approach', ticks=lower_ticks, tolerance=.008)
        before = self.state(name)['position'].copy() if name else None
        self.hold(close_ticks, 1., rotation, phase=phase + '_close')
        gap = self.finger_gap()
        if gap < min_gap:
            raise SkillFailure(phase, f'fingers closed on nothing (gap {gap * 1000:.1f} mm)', gap=gap)
        if geoms is not None and not self.grasped(geoms, fingers=fingers):
            raise SkillFailure(phase, 'finger pads are not both on the object')
        result = dict(gap=gap, rotation=rotation.tolist())
        if lift:
            self.move(self.state(name)['eef'] + [0., 0., float(lift)], rotation, 1., phase=phase + '_lift', ticks=160)
            self.hold(10, 1., rotation, phase=phase + '_lift_hold')
            if name:
                rise = float(self.state(name)['position'][2] - before[2])
                result['lift'] = rise
                if rise < min_lift:
                    raise SkillFailure(phase, f'object rose only {rise * 1000:.0f} mm', lift=rise)
            if geoms is not None and not self.grasped(geoms, fingers=fingers):
                raise SkillFailure(phase, 'object slipped during the lift')
        return result

    def place(self, position, rotation=None, *, above=.12, lower_ticks=140, release_ticks=25, retreat=.15,
              settle=30, name=None, phase='place'):
        """Lower the held object to a world hand position, release, retreat and settle."""
        position = np.asarray(position, float)
        rotation = self.state(name)['eef_rotation'].copy() if rotation is None else np.asarray(rotation, float)
        self.move(position + [0., 0., above], rotation, 1., phase=phase + '_above', ticks=200)
        self.move(position, rotation, 1., phase=phase + '_lower', ticks=lower_ticks, tolerance=.006)
        self.hold(release_ticks, -1., rotation, phase=phase + '_release')
        # Tolerant: at the reach limit over a counter the hand may stop short of the full rise (harmless).
        self.move(self.state(name)['eef'] + [0., 0., retreat], rotation, -1., phase=phase + '_retreat', ticks=140,
                  tolerance=.06, required=False)
        if settle:
            self.hold(settle, -1., phase=phase + '_settle')
        return self.state(name) if name else None

    def press(self, site, *, above=.16, approach=.040, depth=.020, close_ticks=25, press_ticks=100, phase='press'):
        """Press the Submit cap with the closed hand (recipe from the cup/puzzle references). Returns True
        when the press registered the physical submission."""
        position = self.site_position(site) if isinstance(site, str) else np.asarray(site, float)
        down = self.hand_down()
        self.move(position + [0., 0., above], down, -1., phase=phase + '_above', ticks=200)
        self.hold(close_ticks, 1., down, phase=phase + '_close_empty_hand')
        try:
            self.move(position + [0., 0., approach], down, 1., phase=phase + '_approach', ticks=120)
            self.move(position + [0., 0., -depth], down, 1., phase=phase + '_press', ticks=press_ticks,
                      tolerance=0., required=False)
        except Submitted:
            return True
        raise SkillFailure(phase, 'the press did not register a submission')

    def slide(self, handle, axis_world, distance, *, above=.14, grip_dz=.006, rotation=None, alternative=0,
              pull_ticks=90, pull_tolerance=.003, close_ticks=25, min_gap=.004, measure=None, phase='slide'):
        """Grasp a round handle post from above and pull it ``distance`` along ``axis_world``; returns the
        motion measured by ``measure()`` (e.g. the part's joint) or of the hand. No postcondition of its own:
        whether the part should have moved is the task's decision (a blocked part is an observation)."""
        handle = np.asarray(handle, float)
        axis = _unit(axis_world)
        rotation = self.grasp_rotation('top', alternative=alternative) if rotation is None else np.asarray(rotation, float)
        self.move(handle + [0., 0., above], rotation, -1., phase=phase + '_above')
        self.move(handle + [0., 0., grip_dz], rotation, -1., phase=phase + '_lower')
        self.hold(close_ticks, 1., rotation, phase=phase + '_grip')
        gap = self.finger_gap()
        if gap < min_gap:
            raise SkillFailure(phase, f'fingers closed beside the handle (gap {gap * 1000:.1f} mm)', gap=gap)
        before = float(measure()) if measure else None
        eef0 = self.state()['eef'].copy()
        self.move(eef0 + axis * float(distance), rotation, 1., phase=phase + '_pull', ticks=pull_ticks,
                  tolerance=pull_tolerance, required=False)
        self.hold(10, 1., rotation, phase=phase + '_pull_settle')
        hand_moved = float(np.dot(self.state()['eef'] - eef0, axis))
        moved = float(measure()) - before if measure else hand_moved
        self.hold(20, -1., rotation, phase=phase + '_release')
        self.move(self.state()['eef'] + [0., 0., .12], rotation, -1., phase=phase + '_retreat', required=False)
        return dict(moved=moved, hand_moved=hand_moved, commanded=float(distance), gap=gap)

    # ---- base skills -----------------------------------------------------------------------------
    def _drive(self, target, gripper, *, speed, ticks, stop, phase, stall_ticks=25, stall_m=.004):
        """KeyedFitMotor's base velocity loop (body-frame command ``low + 1.2 * error`` clipped to
        ``[low, high]``, full 0.5 command when ``low >= .5``) with stall detection: the loop ends when the
        base moved less than ``stall_m`` over ``stall_ticks`` while commanded (feet against a cabinet
        door, a counter lip), instead of pushing on the fixture for the whole tick budget."""
        target = np.asarray(target, float)[:2]
        low, high = speed
        start, history, stalled = self.steps, [], False
        for _ in range(int(ticks)):
            pos, rot = self.env.robots[0].part_controllers['base'].get_base_pose()
            xy = np.asarray(pos)[:2]
            diff = target - xy
            if np.linalg.norm(diff) < stop:
                break
            history.append(xy.copy())
            if len(history) > stall_ticks and np.linalg.norm(xy - history[-stall_ticks - 1]) < stall_m:
                stalled = True
                break
            body = np.asarray(rot)[:2, :2].T @ diff
            if low >= .5:
                velocity = np.where(np.abs(body) > .01, np.sign(body) * .5, 0.)
            else:
                velocity = np.where(np.abs(body) > .008, np.sign(body) * np.clip(low + 1.2 * np.abs(body), low, high), 0.)
            self.act(np.zeros(3), np.zeros(3), gripper, np.r_[velocity, 0.], True)
        for _ in range(12):
            self.act(np.zeros(3), np.zeros(3), gripper, base_mode=True)
        pos, _ = self.base_pose()
        row = {'phase_private': phase, 'start_step': start, 'end_step': self.steps, 'target_xy_world_m': target.tolist(),
               'position_error_m': float(np.linalg.norm(target - pos)), 'stalled': stalled, 'required_waypoint': False}
        self.phases.append(row)
        if self.phase_callback:
            self.phase_callback(self, row)
        return dict(error=row['position_error_m'], stalled=stalled, ticks=self.steps - start)

    def navigate(self, xy, yaw=None, gripper=None, *, speed=(.20, .40), ticks=None, phase='navigate', tolerant=False,
                 tolerance=.035):
        """Drive the base to a world xy (holonomic, heading kept), then turn to ``yaw`` when given and
        re-centre (a turn displaces the frame by up to 0.4 m). Raises SkillFailure when the base ends more
        than ``tolerance`` from the goal unless ``tolerant``; returns the drive result."""
        xy = np.asarray(xy, float)[:2]
        gripper = self.gripper if gripper is None else float(gripper)
        start_xy, _ = self.base_pose()
        distance = float(np.linalg.norm(xy - start_xy))
        if ticks is None:
            ticks = int(120 + distance / .004)
        result = dict(error=distance, stalled=False, ticks=0)
        if distance > .02:
            result = self._drive(xy, gripper, speed=speed, ticks=ticks, stop=.018, phase=phase)
        if yaw is not None:
            for attempt in range(3):
                self.turn(yaw, gripper, phase=f'{phase}_turn{attempt}')
                current, _ = self.base_pose()
                if np.linalg.norm(xy - current) < .03:
                    break
                result = self._drive(xy, gripper, speed=speed, ticks=400, stop=.018, phase=f'{phase}_recentre{attempt}')
        if result['error'] > tolerance and not tolerant:
            raise SkillFailure(phase, f"native base missed by {result['error']:.4f} m"
                               + (' (stalled against something)' if result['stalled'] else ''), **result)
        return result

    def turn(self, yaw, gripper=None, *, phase='turn', ticks=400, tolerance=.03):
        """Turn the base in place to a world yaw (room-search recipe)."""
        gripper = self.gripper if gripper is None else float(gripper)
        start = self.steps
        for _ in range(int(ticks)):
            _, current = self.base_pose()
            error = (yaw - current + np.pi) % (2 * np.pi) - np.pi
            if abs(error) < tolerance:
                break
            velocity = np.sign(error) * np.clip(.22 + .8 * abs(error), .25, .5)
            self.act(np.zeros(3), np.zeros(3), gripper, [0., 0., velocity], True)
        for _ in range(10):
            self.act(np.zeros(3), np.zeros(3), gripper, base_mode=True)
        _, current = self.base_pose()
        error = abs((yaw - current + np.pi) % (2 * np.pi) - np.pi)
        self.phases.append({'phase_private': phase, 'start_step': start, 'end_step': self.steps, 'target_yaw': float(yaw),
                            'yaw_error_rad': float(error), 'required_waypoint': True})
        if error > .06:
            raise SkillFailure(phase, f'base yaw missed by {error:.3f} rad')
        return error

    def crawl(self, xy, gripper=None, *, phase='crawl', ticks=None, stop=.012):
        """Short base move at full command (the velocity loop creeps below a 0.3 command); stops within
        about a centimetre or at a stall, and never raises."""
        target = np.asarray(xy, float)[:2]
        gripper = self.gripper if gripper is None else float(gripper)
        if ticks is None:
            ticks = int(np.linalg.norm(target - self.base_pose()[0]) / .006) + 100
        return self._drive(target, gripper, speed=(.5, .5), ticks=ticks, stop=stop, phase=phase)

    def stand(self, target, ahead=GRASP_AHEAD_M, lateral=0., *, gripper=None, forward_limit=True, speed=(.20, .40),
              max_short=(.12, .20), phase='stand'):
        """Drive the base (heading kept) so ``target`` lies ``ahead`` metres straight ahead and ``lateral``
        metres to the left. The base never comes closer to the work surface than at reset. A stance the base
        cannot reach (feet against a neighbouring fixture) is accepted while the residual stays within
        ``max_short`` (ahead, lateral) metres, which the arm absorbs; beyond that the skill fails."""
        target = np.asarray(target, float)[:2]
        base_xy, yaw = self.base_pose()
        facing = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-facing[1], facing[0]])
        rel = target - base_xy
        goal = base_xy + facing * (float(rel @ facing) - ahead) + left * (float(rel @ left) - lateral)
        clamped = 0.
        if forward_limit:
            goal_y = float(self.to_frame([goal[0], goal[1], 0.])[1])
            if goal_y > self.forward_limit_y:
                clamped = goal_y - self.forward_limit_y
                goal = goal - self.direction_to_world([0., clamped, 0.])[:2]
        distance = float(np.linalg.norm(goal - base_xy))
        drive = dict(error=distance, stalled=False, ticks=0)
        if distance >= .015:
            if distance < .25:
                drive = self.crawl(goal, gripper, phase=phase + '_crawl')
            else:
                drive = self.navigate(goal, None, gripper, speed=speed, phase=phase, tolerant=True)
        base_xy, _ = self.base_pose()
        residual = goal - base_xy
        short = (float(residual @ facing), float(residual @ left))
        if abs(short[0]) > max_short[0] or abs(short[1]) > max_short[1]:
            raise SkillFailure(phase, f'base stopped {drive["error"]:.3f} m from its stance (ahead {short[0]:+.2f}, '
                               f'left {short[1]:+.2f} m){" after a stall" if drive["stalled"] else ""}', **drive)
        rel = target - base_xy
        return dict(ahead=float(rel @ facing), lateral=float(rel @ left), moved=distance, clamped=clamped,
                    error=drive['error'], stalled=drive['stalled'], short_ahead=short[0], short_lateral=short[1])

    # ---- fixtures (ported from the room-search motor; not verified in the puzzle-box job) ------------
    def _fixture_frame(self, fixture):
        """Outward front normal, lateral unit vector and bounds of a RoboCasa fixture."""
        rot = float(fixture.rot)
        n = np.r_[(yaw_matrix(rot) @ np.array([0., -1., 0.]))[:2], 0.]
        t = np.cross(n, [0., 0., 1.])
        return n, t, np.asarray(fixture.pos, float), np.asarray(fixture.size, float)

    def handle_centre(self, prefix):
        """Centre and bounds of the collision geoms whose names start with ``prefix`` (a bar handle)."""
        m, d = self.env.sim.model, self.env.sim.data
        points = []
        for g in range(m.ngeom):
            gn = m.geom_id2name(g) or ''
            if gn.startswith(prefix) and m.geom_contype[g] != 0:
                half = np.asarray(m.geom_size[g])[:3]
                if m.geom_type[g] == 6:
                    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1])
                                        for sz in (-half[2], half[2])])
                    points.extend(corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
                else:
                    r = float(half[0])
                    points.extend([d.geom_xpos[g] + np.array([sx, sy, sz]) * r for sx in (-1, 1) for sy in (-1, 1)
                                   for sz in (-1, 1)])
        if not points:
            raise SkillFailure('handle', f'no collision handle geoms with prefix {prefix}')
        points = np.asarray(points)
        return (points.min(0) + points.max(0)) / 2, points.min(0), points.max(0)

    def open_drawer(self, fixture, opening=DRAWER_PULL_M, *, phase='drawer'):
        """Grasp the bar handle of a RoboCasa drawer from the front (tool x up), pull with the arm while the
        hand stays 0.40 m from the base, reverse the base for the remainder, unhook, rise, return.
        Ported from room search (verified there on layouts 1/13/17); the base must already face the
        drawer front at its standoff. Status: ported, not verified in this job."""
        n, _, pos, size = self._fixture_frame(fixture)
        front = yaw_matrix(float(fixture.rot)) @ DRAWER_FRONT_UP
        joint = fixture.door_joint_names[0]
        handle, _, _ = self.handle_centre(fixture.name + '_door_handle_')
        self.move(handle + n * .18 + [0., 0., .12], front, -1., phase=phase + '_prehandle', ticks=250)
        self.move(handle + n * .12, front, -1., phase=phase + '_align', ticks=160)
        self.move(handle + n * .004, front, -1., phase=phase + '_approach', ticks=120)
        self.hold(20, 1., front, phase=phase + '_grasp')
        if self.finger_gap() < .004:
            raise SkillFailure(phase, 'fingers closed beside the bar')
        q = float(self.env.sim.data.get_joint_qpos(joint))
        closed = handle - n * abs(q)
        base_xy, _ = self.base_pose()
        facing = self.facing()
        distance_from_front = float(np.dot(base_xy - (pos[:2] + n[:2] * size[1] / 2), n[:2]))
        arm_pull = float(np.clip(distance_from_front - .40, 0., opening))
        self.move(closed + n * (arm_pull + .004), front, 1., phase=phase + '_arm_pull', ticks=220, tolerance=.01, required=False)
        after_arm = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        remaining = opening - after_arm
        moved_back = 0.
        if remaining > .02:
            self.crawl(base_xy - facing * remaining, 1., phase=phase + '_base_pull', ticks=int(remaining / .006) + 150)
            moved_back = float(np.linalg.norm(self.base_pose()[0] - base_xy))
        self.hold(20, 1., front, phase=phase + '_hold')
        achieved = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        self.hold(15, -1., front, phase=phase + '_release')
        self.crawl(self.base_pose()[0] - facing * UNHOOK_M, -1., phase=phase + '_unhook', ticks=80)
        free = self.state()['eef'].copy()
        clear_z = float(pos[2]) + float(size[2]) / 2 + .12
        self.move(np.r_[free[:2], clear_z], front, -1., phase=phase + '_retreat', ticks=160, tolerance=.015, required=False)
        view_xy = base_xy - facing * max(0., DRAWER_VIEW_M - distance_from_front)
        if np.linalg.norm(view_xy - self.base_pose()[0]) > .02:
            self.crawl(view_xy, -1., phase=phase + '_base_return', ticks=200)
        if achieved < min(.34, opening - .06):
            raise SkillFailure(phase, f'drawer opened only {achieved:.3f} m (arm {after_arm:.3f}, base {moved_back:.3f})')
        return {'joint': joint, 'achieved_open_m': achieved, 'arm_pull_m': after_arm, 'base_reverse_m': moved_back}

    def tilted_down(self, n, angle):
        down = self.hand_down()
        axis = down[:, 0]
        for sign in (1., -1.):
            tilted = Rotation.from_rotvec(sign * angle * axis).as_matrix() @ down
            if float(np.dot(tilted[:, 2], -n)) > 0.:
                return tilted
        return down

    def open_door(self, fixture, open_rad=DOOR_OPEN_RAD, *, side=None, phase='door'):
        """Vertical bar handle of a RoboCasa cabinet door: tilted top-down grasp of the bar's top, base
        crawl along the hinge arc in chords while the compliant arm holds the bar, release, unhook, then
        push the leaf the rest of the way with the closed hand. Ported from room search (verified there on
        layout 1); the base must already face the door at its standoff. Status: ported, not verified here."""
        n, t, pos, size = self._fixture_frame(fixture)
        if side is None:
            side = 'right' if type(fixture).__name__ == 'HingeCabinet' else ''
        joints = list(fixture.door_joint_names)
        joint = next(j for j in joints if (side + 'doorhinge') in j) if side else joints[0]
        model, data = self.env.sim.model, self.env.sim.data
        lo, hi = model.jnt_range[model.joint_name2id(joint)]
        direction = 1. if hi > 0 else -1.
        pivot = np.asarray(data.xanchor[model.joint_name2id(joint)]).copy()
        prefix = fixture.name + (f'_{side}_door_handle_' if side else '_door_handle_')
        handle, low, high = self.handle_centre(prefix)
        grip = handle.copy()
        grip[2] = float(high[2] - DOOR_GRIP_BELOW_TOP_M)
        if high[2] < DOOR_LOW_BAR_Z:
            if high[2] >= DOOR_MID_BAR_Z:
                base_xy0, _ = self.base_pose()
                self.crawl(base_xy0 - self.facing() * .10, -1., phase=phase + '_back_off', ticks=60)
            down = self.tilted_down(n, DOOR_TILT_RAD)
            above = grip + n * .012 - down[:, 2] * .26
        else:
            down = self.hand_down()
            above = grip + n * .06 + [0., 0., .22]
        self.move(above, down, -1., phase=phase + '_above', ticks=250)
        self.move(grip + n * .012, down, -1., phase=phase + '_descend', ticks=160, tolerance=.008)
        self.hold(25, 1., down, phase=phase + '_grasp')
        if self.finger_gap() < .004:
            raise SkillFailure(phase, 'fingers closed beside the bar')
        angle0 = abs(float(data.get_joint_qpos(joint)))
        closed_span = Rotation.from_euler('z', -direction * angle0).as_matrix() @ (handle - pivot)
        width = float(np.linalg.norm(closed_span[:2]))
        sign = float(np.sign(np.dot(closed_span[:2], t[:2]))) or 1.
        hinge_dir = -sign * t[:2]
        base_xy, _ = self.base_pose()
        for index, theta in enumerate(np.linspace(DOOR_PULL_RAD / 2., DOOR_PULL_RAD, 2)):
            delta = n[:2] * width * math.sin(theta) + hinge_dir * width * (1. - math.cos(theta))
            self.crawl(base_xy + delta, 1., phase=f'{phase}_base_pull_{index}', ticks=int(np.linalg.norm(delta) / .006) + 100)
        pulled = abs(float(data.get_joint_qpos(joint)))
        rotation = self.state()['eef_rotation'].copy()
        self.hold(20, -1., rotation, phase=phase + '_release')
        leaf_normal = Rotation.from_euler('z', direction * pulled).as_matrix() @ n
        self.crawl(self.base_pose()[0] + leaf_normal[:2] * UNHOOK_M, -1., phase=phase + '_unhook', ticks=80)
        free = self.state()['eef'].copy()
        self.move(free + [0., 0., .20], rotation, -1., phase=phase + '_up', ticks=140, required=False)
        # Push the leaf the rest of the way from the free-edge side, hand in the wedge behind the leaf.
        push_point = pivot[:2] + t[:2] * sign * .30
        goal = push_point + n[:2] * .70
        if np.linalg.norm(goal - self.base_pose()[0]) > .02:
            self.crawl(goal, -1., phase=phase + '_push_stand', ticks=300)
        self.hold(15, 1., self.state()['eef_rotation'].copy(), phase=phase + '_close_hand')
        edge_vector = handle - pivot
        radial = edge_vector * (1. - .15 / max(width, .15))
        push_z = min(grip[2] - .10, float(pos[2]) + float(size[2]) / 2 - .12)
        vertical = self.hand_down()
        current = abs(float(data.get_joint_qpos(joint)))
        angles = [current - .30] + list(np.linspace(current + .20, open_rad, 3))
        for index, angle in enumerate(angles):
            rot = Rotation.from_euler('z', direction * angle).as_matrix()
            target = pivot + rot @ radial
            if index == 0:
                self.move(np.r_[target[:2], push_z + .30], rot @ vertical, 1., phase=f'{phase}_push_over', ticks=140,
                          tolerance=.015, required=False)
            self.move(np.r_[target[:2], push_z], rot @ vertical, 1., phase=f'{phase}_push_{index}', ticks=110,
                      tolerance=.012, required=False)
        achieved = abs(float(data.get_joint_qpos(joint)))
        self.move(self.state()['eef'] + [0., 0., .20], None, -1., phase=phase + '_clear', ticks=140, required=False)
        self.carry(-1., phase=phase + '_relax')
        if achieved < DOOR_MIN_OPEN_RAD:
            raise SkillFailure(phase, f'door opened only {achieved:.3f} rad (after the arc pull {pulled:.3f})')
        return {'joint': joint, 'achieved_open_rad': achieved, 'direction': direction, 'base_pull_rad': pulled}

    # ---- uncover and look skills (roboquest.uncover; thin delegates) --------------------
    def lift_cloche(self, cover_body, **kwargs):
        """Lift a cloche by its knob and set it down on a free spot (see :func:`uncover.lift_cloche`)."""
        from roboquest.uncover import lift_cloche
        return lift_cloche(self, cover_body, **kwargs)

    def slide_lift_bowl(self, cover_body, edge=None, **kwargs):
        """Slide an inverted bowl to the counter edge, pinch the overhanging rim, lift and set it down."""
        from roboquest.uncover import slide_lift_bowl
        return slide_lift_bowl(self, cover_body, edge, **kwargs)

    def push_cover(self, cover_body, direction=None, distance=None, **kwargs):
        """Push a plate, board or tray off the place it hides, keeping it on the surface."""
        from roboquest.uncover import push_cover
        return push_cover(self, cover_body, direction, distance, **kwargs)

    def lift_plate_from_pan(self, cover_body, **kwargs):
        """Pinch a plate's rim where it overhangs a balance pan, lift it straight up, set it down aside."""
        from roboquest.uncover import lift_plate_from_pan
        return lift_plate_from_pan(self, cover_body, **kwargs)

    def pour_cup(self, cup_body, target_xy=None, **kwargs):
        """Grasp the wide cup by its handle, tilt it past 120 degrees over a free spot, level it, set it down."""
        from roboquest.uncover import pour_cup
        return pour_cup(self, cup_body, target_xy, **kwargs)

    def drive_to_stance(self, stance, **kwargs):
        """Drive the base to an observe stance ``{xy, yaw, edge, standoff}`` along a planned route."""
        from roboquest.uncover import drive_to_stance
        return drive_to_stance(self, stance, **kwargs)

    def look_at(self, body, **kwargs):
        """Drive to a reachable viewpoint of ``body`` and return its pose (privileged)."""
        from roboquest.uncover import look_at
        return look_at(self, body, **kwargs)

    def fetch(self, body, **kwargs):
        """Drive to an annex stance, grasp ``body`` from above, bring it back onto the work frame."""
        from roboquest.uncover import fetch
        return fetch(self, body, **kwargs)

    def resolve_observability(self, **kwargs):
        """Undo every hiding mechanism of the instance (annex, occluder, cover) before the normal routine."""
        from roboquest.uncover import resolve_observability
        return resolve_observability(self, self.env, **kwargs)

    # ---- segments, checkpoints and attempts ------------------------------------------------------
    @contextmanager
    def segment(self, caption, **meta):
        """Label the actions taken inside the block (for captions and reports)."""
        record = dict(caption=str(caption), start=self.tick, end=None, **meta)
        self.segments.append(record)
        try:
            yield record
        finally:
            record['end'] = self.tick

    def _truncate_segments(self, tick, count):
        """Drop the segments opened after the checkpoint was taken; reopen those that closed after its tick."""
        del self.segments[count:]
        for record in self.segments:
            if record['end'] is not None and record['end'] > tick:
                record['end'] = None

    def checkpoint(self, name=''):
        return Checkpoint(self, name)

    def run_attempt(self, name, fn, args=(), kwargs=None, retries=3, jitter=None):
        """Run ``fn(*args, **kwargs)`` up to ``1 + retries`` times from the same checkpoint. On a
        SkillFailure/RuntimeError the checkpoint is restored (action log truncated) and the next try adds
        ``jitter(attempt_index, rng)`` to the keyword arguments. Submitted always propagates."""
        checkpoint = self.checkpoint(name)
        errors = []
        total = 1 + max(0, int(retries))
        for index in range(total):
            params = dict(kwargs or {})
            if index and jitter is not None:
                params.update(jitter(index, self.rng) or {})
            started = time.monotonic()
            try:
                result = fn(*args, **params)
            except Submitted:
                raise
            except (SkillFailure, RuntimeError, ValueError) as error:
                margin, joint = self.arm_joint_margin()
                errors.append(dict(attempt=index + 1, params={k: _jsonable(v) for k, v in params.items()},
                                   error=str(error), details=_jsonable(getattr(error, 'details', None) or {}),
                                   tick=self.tick, joint_margin_rad=margin, joint=joint))
                self.attempts.append(dict(name=name, attempt=index + 1, ok=False, error=str(error),
                                          start_tick=checkpoint.tick, end_tick=self.tick,
                                          wall_seconds=time.monotonic() - started))
                if index == total - 1:
                    raise AttemptFailed(name, errors) from error
                checkpoint.restore()
                continue
            self.attempts.append(dict(name=name, attempt=index + 1, ok=True, start_tick=checkpoint.tick,
                                      end_tick=self.tick, wall_seconds=time.monotonic() - started))
            return result

    def attempt(self, name, retries=3, jitter=None):
        """Decorator form of :meth:`run_attempt`: the wrapped function's keyword arguments are the
        parameters the jitter may override."""
        def decorator(fn):
            @functools.wraps(fn)
            def run(*args, **kwargs):
                return self.run_attempt(name, fn, args, kwargs, retries=retries, jitter=jitter)
            return run
        return decorator

    # ---- reports ----------------------------------------------------------------------------------
    def summary(self):
        failed = [a for a in self.attempts if not a['ok']]
        return dict(ticks=self.tick, attempts=len(self.attempts), failed_attempts=len(failed),
                    restores=len(self.restores), segments=[dict(s) for s in self.segments])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


# ---- record and replay ---------------------------------------------------------------------------
def save_actions(skills, path, instance=None, extra=None):
    """Write the action log with its captions and the final movable-body poses (the replay check)."""
    env = skills.env
    score = env.evaluate_success() if hasattr(env, 'evaluate_success') else None
    payload = dict(
        format=ACTIONS_FORMAT, task=getattr(env, 'TASK_NAME', None),
        instance_id=(instance or {}).get('instance_id') or getattr(env, 'instance', {}).get('instance_id'),
        instance=deepcopy(instance) if instance is not None else deepcopy(getattr(env, 'instance', None)),
        control_hz=int(env.control_freq), action_dim=skills.dim, ticks=skills.tick,
        actions=[a.tolist() for a in skills.actions], segments=[dict(s) for s in skills.segments],
        attempts=_jsonable(skills.attempts), restores=_jsonable(skills.restores),
        submitted=getattr(env, 'submission', None) is not None,
        final=dict(bodies=movable_body_poses(env), qpos=np.asarray(env.sim.data.qpos).tolist(),
                   sim_time=float(env.sim.data.time), score=_jsonable(score)),
        extra=_jsonable(extra or {}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    # The caption list also goes next to the log as a small file editors and video tools can read directly.
    path.with_suffix('.segments.json').write_text(json.dumps(payload['segments'], indent=1) + '\n')
    return path


def load_actions(path):
    log = json.loads(Path(path).read_text())
    if log.get('format') != ACTIONS_FORMAT:
        raise ValueError(f'{path} is not a {ACTIONS_FORMAT} file')
    return log


def caption_at(segments, tick):
    """The innermost (last started) segment caption covering ``tick``."""
    caption = ''
    for record in segments:
        end = record['end'] if record['end'] is not None else float('inf')
        if record['start'] <= tick < end:
            caption = record['caption']
    return caption


def compose_frame(obs, cameras=CAMERAS, caption='', tick=None, fps=20):
    """Three policy cameras side by side with a caption bar and a time stamp burned in."""
    from PIL import Image, ImageDraw, ImageFont
    frame = np.concatenate([obs[camera + '_image'][::-1] for camera in cameras], axis=1)
    frame = np.ascontiguousarray(frame)
    if not caption and tick is None:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, 'RGBA')
    h, w = frame.shape[:2]
    size = max(16, int(h * .05))
    try:
        font = ImageFont.truetype(DEJAVU_BOLD, size)
    except OSError:
        font = ImageFont.load_default()
    if caption:
        bar = int(size * 1.8)
        draw.rectangle([0, h - bar, w, h], fill=(0, 0, 0, 160))
        draw.text((int(size * .6), h - bar + int(size * .35)), caption, fill=(255, 255, 255, 255), font=font)
    if tick is not None:
        text = f't = {tick / fps:6.1f} s   tick {tick}'
        draw.rectangle([0, 0, int(size * 11), int(size * 1.6)], fill=(0, 0, 0, 120))
        draw.text((int(size * .4), int(size * .2)), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image)


def pose_errors(recorded, current):
    """Largest position (m) and angle (deg) difference between two body-pose dicts."""
    worst = dict(position_m=0., angle_deg=0., position_body=None, angle_body=None)
    for name, pose in recorded.items():
        if name not in current:
            continue
        dp = float(np.linalg.norm(np.subtract(pose['pos'], current[name]['pos'])))
        qa, qb = np.asarray(pose['quat'], float), np.asarray(current[name]['quat'], float)
        dot = float(np.clip(abs(np.dot(qa, qb)) / max(np.linalg.norm(qa) * np.linalg.norm(qb), 1e-12), -1., 1.))
        da = float(np.degrees(2. * math.acos(dot)))
        if dp > worst['position_m']:
            worst.update(position_m=dp, position_body=name)
        if da > worst['angle_deg']:
            worst.update(angle_deg=da, angle_body=name)
    return worst


def replay(instance, path, render=True, video=None, gpu=0, image_size=512, fps=20, cameras=CAMERAS, task_cls=None,
           position_tol=.005, angle_tol_deg=3., progress=None, segments=None):
    """Re-execute a recorded action log on a fresh env of the same instance.

    Returns a report with the final-pose agreement against the recording (``ok`` when every movable body
    is within ``position_tol`` / ``angle_tol_deg``) and, when ``video`` is given, writes a ``fps`` mp4 of
    the three policy cameras side by side with the segment captions burned in. ``segments`` overrides
    the captions stored in the log (a list of ``{'start', 'end', 'caption'}``).
    """
    from roboquest.kitchen import make_env
    from roboquest.manifest import task_class
    log = load_actions(path) if not isinstance(path, dict) else path
    actions = np.asarray(log['actions'], float)
    captions = segments if segments is not None else log.get('segments', [])
    cls = task_cls or task_class(instance['task'])
    env = make_env(cls, instance, image_size=image_size, gpu=gpu, horizon=max(len(actions) + 20, 100), render=render)
    writer = None
    started = time.monotonic()
    try:
        obs = env.reset()
        if video:
            import imageio.v2 as imageio
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video), fps=fps, codec='libx264', quality=7, macro_block_size=None)
            writer.append_data(compose_frame(obs, cameras, caption_at(captions, 0), 0, fps))
        submitted_at = None
        replayed = 0
        for tick, action in enumerate(actions):
            obs, *_ = env.step(action.copy())
            replayed = tick + 1
            if writer is not None:
                writer.append_data(compose_frame(obs, cameras, caption_at(captions, tick), tick + 1, fps))
            if progress and replayed % 500 == 0:
                progress(replayed, len(actions))
            if getattr(env, 'submission', None) is not None:
                submitted_at = replayed
                break
        if writer is not None and submitted_at is not None:
            for _ in range(fps * 2):   # two-second terminal hold on the frozen scene
                writer.append_data(compose_frame(obs, cameras, caption_at(captions, replayed - 1), replayed, fps))
        errors = pose_errors(log['final']['bodies'], movable_body_poses(env))
        qpos = np.asarray(env.sim.data.qpos)
        recorded_qpos = np.asarray(log['final']['qpos'], float)
        score = env.evaluate_success() if hasattr(env, 'evaluate_success') else None
        report = dict(
            ticks_replayed=replayed, ticks_recorded=len(actions), submitted_at=submitted_at,
            submitted_recorded=log.get('submitted'), max_position_error_m=errors['position_m'],
            max_angle_error_deg=errors['angle_deg'], worst_position_body=errors['position_body'],
            worst_angle_body=errors['angle_body'],
            max_qpos_difference=float(np.max(np.abs(qpos - recorded_qpos))) if qpos.shape == recorded_qpos.shape else None,
            ok=bool(replayed == len(actions) and errors['position_m'] <= position_tol
                    and errors['angle_deg'] <= angle_tol_deg and bool(submitted_at) == bool(log.get('submitted'))),
            score=_jsonable(score), video=str(video) if video else None, rendered=bool(render),
            wall_seconds=time.monotonic() - started)
        return report
    finally:
        if writer is not None:
            writer.close()
        env.close()
