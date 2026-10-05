"""Public physical policy adapter, separate from privileged task evaluation.

Uses native workroom delta/absolute-arm and mobile-base semantics. No semantic
object tools, simulator object poses, intermediate score, or success termination.
Python in-process introspection is not a security sandbox; tool outputs are the
policy boundary. Trace, evaluator, callbacks and score() are evaluator-only APIs.
"""
from copy import deepcopy
import hashlib
import random
import time
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.harness.contract import CAMERAS, ReleasedStableEvaluator, policy_observation
from roboquest.harness.command_checks import finite_vector, bounded_ticks

# Fixed across variants, intentionally broad. Not a reachability/collision promise.
ARM_COMMAND_MIN = np.array([-6., -6., .3])
ARM_COMMAND_MAX = np.array([6., 6., 2.2])


class TableSettingAdapter:
    def __init__(self, variant='cup_deficit', seed=0, image_size=256, horizon=6000,
                 gpu=2, near_clip_m=.001, env=None, *,
                 evaluator_factory=ReleasedStableEvaluator, instruction=None,
                 control_version=None):
        if isinstance(horizon,bool) or not isinstance(horizon,int) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if isinstance(near_clip_m, bool) or not np.isfinite(near_clip_m) or near_clip_m <= 0:
            raise ValueError('near clip must be finite and positive')
        if env is None:
            from roboquest.base.kitchen_scene import make_table_setting_scene
            env = make_table_setting_scene(variant, seed, image_size, gpu, horizon+20)
        self.env = env
        self.seed, self.horizon = seed, horizon
        self.near_clip_m = near_clip_m
        if env.control_freq != 20:
            raise ValueError('contract requires native 20 Hz control')
        self._evaluator_factory = evaluator_factory
        self._instruction, self._control_version = instruction, control_version
        # Lazy native scenes create task_spec during reset, after model loading.
        self._evaluator = None
        self._obs = None
        self.steps = 0
        self.stopped = False
        self._closed = False
        self.termination = None
        self.trace, self.commands, self.reset_trace = [], [], []
        self.callback = None
        self._evaluation_error = None
        self._timing_s = {}

    def _add_timing(self, name, elapsed):
        self._timing_s[name] = self._timing_s.get(name, 0.0) + float(elapsed)

    def timing_summary(self):
        """Evaluator-side phase timings; never projected into policy observations."""
        total = float(self._timing_s.get('total', 0.0))
        phases = {name: float(value) for name, value in self._timing_s.items() if name != 'total'}
        measured = sum(phases.values())
        return {
            'steps': int(self.steps),
            'total_s': total,
            **{f'{name}_s': value for name, value in phases.items()},
            'unattributed_s': max(0.0, total - measured),
        }

    def _tick_evaluator_snapshot(self, step):
        """Private evidence consumed after one physical tick; subclasses may make this cheaper."""
        return self.env.evaluator_snapshot(step)

    @property
    def robot(self):
        return self.env.robots[0]

    def _native(self, arm, base, base_mode):
        ctrl = self.robot.part_controllers['right']
        local = np.r_[ctrl.origin_ori.T @ arm[:3] / .05,
                      ctrl.origin_ori.T @ arm[3:6] / .5]
        return self.robot.composite_controller.create_action_vector(
            {'right':local,'right_gripper':[arm[6]],'base':base.copy(),
             'torso':[0.], 'base_mode':1 if base_mode else -1})

    def reset(self):
        if self._closed:
            raise RuntimeError('Adapter closed')
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.seed = self.seed
        self.env.rng = np.random.default_rng(self.seed)
        self._obs = self.env.reset()
        # Prevent a model-registry error from exposing object freejoint vectors.
        robot_joints = set(self.robot.robot_model.all_joints)
        for gripper in self.robot.gripper.values():
            robot_joints.update(gripper.joints)
        object_joints = {j for obj in self.env.objects.values() for j in obj.joints}
        if not robot_joints or robot_joints & object_joints:
            raise ValueError('invalid robot proprioception joint manifest')
        self.env.sim.model.vis.map.znear = self.near_clip_m / self.env.sim.model.stat.extent
        self.reset_trace = []
        # Fixed initialization, not policy-time success seeking. No evaluator credit.
        for _ in range(10):
            raw = self._native(np.array([0.,0.,0.,0.,0.,0.,-1.]),np.zeros(3),False)
            self.reset_trace.append(np.array(raw,copy=True).tolist())
            self._obs = self.env.step(np.array(raw,copy=True))[0]
        self.steps, self.stopped, self.termination = 0, False, None
        self._evaluation_error = None
        self.trace, self.commands = [], []
        self._timing_s = {}
        self._evaluator = self._evaluator_factory(self.env.task_spec)
        self.initial_time = float(self.env.sim.data.time)
        self.started = time.monotonic()
        self.initial_qpos = np.array(self.env.sim.data.qpos,copy=True)
        self.initial_qvel = np.array(self.env.sim.data.qvel,copy=True)
        packet = self.observe()
        self.initial_rgb_hashes = {
            camera: hashlib.sha256(np.ascontiguousarray(packet['rgb'][camera]).tobytes()).hexdigest()
            for camera in CAMERAS
        }
        return packet

    def observe(self):
        if self._obs is None:
            raise RuntimeError('Call reset first')
        # Reproject even if an injected env helper accidentally adds private keys.
        observed = self.env.policy_observe(self._obs)
        clean = policy_observation(observed['rgb'],observed['proprio'])
        if self._instruction is not None:
            clean['instruction'] = self._instruction
        if self._control_version is not None:
            clean['contract_version'] = self._control_version
        clean.update(step=self.steps,
                     sim_time_s=float(self.env.sim.data.time)-self.initial_time,
                     remaining_ticks=max(0,self.horizon-self.steps),
                     episode_ended=bool(self.stopped or self.steps>=self.horizon))
        return clean

    def rgb_hashes(self, capture_missing=True):
        images = {c: self._obs.get(c + '_image') for c in CAMERAS}
        if any(image is None for image in images.values()):
            if not capture_missing:
                return None
            rgb = self.env.policy_observe(self._obs)['rgb']
        else:
            rgb = {c: images[c][::-1] for c in CAMERAS}
        return {c: hashlib.sha256(np.ascontiguousarray(rgb[c]).tobytes()).hexdigest() for c in CAMERAS}

    def _check_active(self):
        if self._closed or self._obs is None:
            raise RuntimeError('Adapter closed or not reset')
        if self.stopped or self.steps>=self.horizon:
            raise RuntimeError('Episode ended')

    def _physical_termination(self, snapshot):
        """Versioned task hook; scenes without it never terminate on intermediate success."""
        return None

    def act(self, arm, base=(0,0,0), base_mode=False):
        """One charged20Hz native step; gripper -1 open,+1 close,0 retains setpoint."""
        self._check_active()
        arm=finite_vector(arm,7,'arm');base=finite_vector(base,3,'base')
        if np.any(np.abs(arm)>np.array([.025]*3+[.15]*3+[1])+1e-12):
            raise ValueError('arm delta exceeds limits')
        if np.any(np.abs(base)>.5+1e-12):
            raise ValueError('base input exceeds 0.5')
        if not isinstance(base_mode,(bool,np.bool_)):
            raise ValueError('base_mode must be boolean')
        if base_mode and np.any(arm[:6]!=0):
            raise ValueError('base mode requires zero arm delta')
        raw=self._native(arm,base,bool(base_mode))
        return self._step_native(raw, {'arm':arm.tolist(),'base':base.tolist(),'base_mode':bool(base_mode)})

    def act_native(self, action, *, observe=True):
        """Execute one finite native controller action as one charged 20 Hz tick.

        This is the public low-level VLA path. It keeps the evaluator,
        physical-Submit, timeout, trace, and callback behavior of :meth:`act`;
        callers remain responsible for task-specific action sanitization. Set
        ``observe=False`` only when the caller will explicitly observe after an
        action chunk; physics and evaluator state still advance for this tick.
        """
        self._check_active()
        if not isinstance(observe, (bool, np.bool_)):
            raise ValueError('observe must be boolean')
        dimension = int(getattr(self.env, 'action_dim', 12))
        raw = finite_vector(action, dimension, 'native_action')
        spec = getattr(self.env, 'action_spec', None)
        if spec is not None:
            low, high = (np.asarray(bound, float) for bound in spec)
            if low.shape != (dimension,) or high.shape != (dimension,):
                raise ValueError('invalid native action specification')
            if np.any(raw < low - 1e-12) or np.any(raw > high + 1e-12):
                raise ValueError('native action exceeds controller limits')
        return self._step_native(raw, {'native':raw.tolist()}, observe=observe)

    def _step_native(self, raw, public_action, *, observe=True):
        """Shared single-tick execution after a public action is validated."""
        total_started = time.perf_counter()
        submitted=np.array(raw,copy=True)
        phase_started = time.perf_counter()
        try:
            self._obs=self.env.step(np.array(raw,copy=True))[0]
        except Exception as exc:
            self.stopped=True;self.termination='runtime_error'
            self._evaluation_error=repr(exc)
            raise RuntimeError('Physical execution interrupted') from None
        finally:
            self._add_timing('env_step', time.perf_counter() - phase_started)
        self.steps+=1
        phase_started = time.perf_counter()
        record={'step':self.steps,'action':public_action,
                'native_action':submitted.tolist(),'sim_time':float(self.env.sim.data.time),
                'qpos':self.env.sim.data.qpos.tolist(),'qvel':self.env.sim.data.qvel.tolist(),
                'rgb_sha256':self.rgb_hashes(capture_missing=False)}
        self._add_timing('trace_record', time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        try:
            snapshot=self._tick_evaluator_snapshot(self.steps)
            # Preserve the evaluation already computed for this physical tick.
            # This private trace is never projected into policy observations.
            record['score']=self._evaluator.update(snapshot)
            record['snapshot']=snapshot
        except Exception as exc:
            self._evaluation_error=repr(exc)
            self.stopped=True;self.termination='evaluation_error'
            self.trace.append(record)
            raise RuntimeError('Episode evaluation unavailable') from None
        finally:
            self._add_timing('evaluation', time.perf_counter() - phase_started)
        self.trace.append(record)
        terminal = self._physical_termination(snapshot)
        if terminal is not None:
            self.stopped=True;self.termination=terminal
        elif self.steps>=self.horizon:
            self.stopped=True;self.termination='horizon'
        phase_started = time.perf_counter()
        packet = self.observe() if observe else None
        if packet is not None and record['rgb_sha256'] is None:
            record['rgb_sha256'] = {
                camera: hashlib.sha256(np.ascontiguousarray(packet['rgb'][camera]).tobytes()).hexdigest()
                for camera in CAMERAS
            }
        self._add_timing('observation', time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        if self.callback is not None:
            self.callback(self)  # evaluator-owned recorder only, never tool result
        self._add_timing('callback', time.perf_counter() - phase_started)
        self._add_timing('total', time.perf_counter() - total_started)
        return packet

    def execute(self, command):
        """workroom-compatible arm/base/wait/stop mapping; maximum200ticks per chunk.

arm: position_world_m, optional quaternion_xyzw/gripper/ticks(default120).
base: velocity_body normalized3vector, ticks(default20); arm held relative to base.
wait: optional gripper/ticks(default20). stop: no extra physical step.
"""
        self._check_active()
        if not isinstance(command,dict):raise ValueError('command must be a mapping')
        command=deepcopy(command);kind=command.get('type')
        fields={'arm':{'type','position_world_m','quaternion_xyzw','gripper','ticks'},
                'base':{'type','velocity_body','ticks'},'wait':{'type','gripper','ticks'},'stop':{'type'}}
        if not isinstance(kind,str) or kind not in fields or set(command)-fields[kind]:
            raise ValueError('unknown command type or fields')
        if kind=='stop':
            self.stopped=True;self.termination='declared_stop'
            self.commands.append({'start_step':self.steps,'end_step':self.steps,'command':command})
            return self.observe()
        ticks=bounded_ticks(command.get('ticks',120 if kind=='arm' else 20))
        if ticks>self.horizon-self.steps:raise ValueError('command exceeds remaining physical budget')
        try:gripper=float(command.get('gripper',0))
        except (ValueError,TypeError):raise ValueError('invalid gripper') from None
        if not np.isfinite(gripper) or abs(gripper)>1:raise ValueError('gripper must be finite in [-1,1]')
        if kind=='arm':
            if 'position_world_m' not in command:raise ValueError('position_world_m required')
            target=finite_vector(command['position_world_m'],3,'position')
            if np.any(target<ARM_COMMAND_MIN) or np.any(target>ARM_COMMAND_MAX):raise ValueError('arm target outside command envelope')
            quat=finite_vector(command.get('quaternion_xyzw',self.observe()['proprio']['eef_quaternion_xyzw']),4,'quaternion')
            if abs(np.linalg.norm(quat)-1)>1e-3:raise ValueError('quaternion must be normalized')
            rotation=Rotation.from_quat(quat).as_matrix()
        elif kind=='base':
            if 'velocity_body' not in command:raise ValueError('velocity_body required')
            velocity=finite_vector(command['velocity_body'],3,'velocity_body')
            if np.any(np.abs(velocity)>.5):raise ValueError('base input exceeds 0.5')
        before=self.steps
        for _ in range(ticks):
            if kind=='arm':
                p=self.observe()['proprio']
                dp=.8*(target-np.asarray(p['eef_position_world_m']))
                dr=.5*Rotation.from_matrix(rotation @ Rotation.from_quat(p['eef_quaternion_xyzw']).as_matrix().T).as_rotvec()
                self.act(np.r_[np.clip(dp,-.015,.015),np.clip(dr,-.12,.12),gripper])
            elif kind=='base':self.act(np.zeros(7),velocity,True)
            else:self.act([0,0,0,0,0,0,gripper])
            if self.stopped:
                break
        self.commands.append({'start_step':before,'end_step':self.steps,'command':command})
        return self.observe()

    def score(self):
        """Evaluator only; fresh terminal snapshot, never included in observations."""
        if self._obs is None:raise RuntimeError('Call reset first')
        if self._evaluation_error is not None:
            return {'success':False,'scored':False,'termination':self.termination,
                    'private_error':self._evaluation_error,'steps':self.steps}
        result=self._evaluator.score(self.env.evaluator_snapshot(self.steps))
        return {**result,'scored':True,'termination':self.termination,'steps':self.steps,
                'sim_time_s':float(self.env.sim.data.time)-self.initial_time,
                'wall_time_s':time.monotonic()-self.started}

    def close(self):
        if not self._closed:
            self.env.close();self._closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
