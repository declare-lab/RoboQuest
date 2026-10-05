"""Privileged keyed-fitting demonstration through ordinary native actions.

Object poses choose scripted waypoints. This is not an observation-limited
policy, and the route deliberately tries two known wrong candidates first.
There are no object setters, attachments, collision changes or camera changes.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class KeyedFitMotor:
    """Bounded world-frame OSC servo with native base motion and held torso."""

    def __init__(self, env, observation, horizon=5000, step_callback=None,
                 phase_callback=None):
        self.env, self.obs, self.horizon = env, observation, horizon
        self.steps, self.phases = 0, []
        self.step_callback, self.phase_callback = step_callback, phase_callback

    def state(self, name):
        robot, sim = self.env.robots[0], self.env.sim
        bid = self.env.obj_body_id[name]
        sid = robot.eef_site_id['right']
        return {'position': sim.data.body_xpos[bid].copy(),
                'rotation': sim.data.body_xmat[bid].reshape(3, 3).copy(),
                'eef': sim.data.site_xpos[sid].copy(),
                'eef_rotation': sim.data.site_xmat[sid].reshape(3, 3).copy()}

    def step(self, action):
        """The one physical step; subclasses override it to record or gate actions."""
        return self.env.step(action)

    def act(self, dp, dr, gripper, base=(0., 0., 0.), base_mode=False):
        if self.steps >= self.horizon:
            raise RuntimeError('Native physical horizon exhausted')
        dp, dr, base = (np.asarray(v, dtype=float) for v in (dp, dr, base))
        if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (dp, dr, base)):
            raise ValueError('Native action requires finite three-vectors')
        if np.any(np.abs(dp) > .015+1e-12) or np.any(np.abs(dr) > .12+1e-12):
            raise ValueError('Native arm delta exceeds demonstration bounds')
        if np.any(np.abs(base) > .5+1e-12) or not np.isfinite(gripper) or abs(gripper) > 1:
            raise ValueError('Native base/gripper action exceeds bounds')
        if base_mode and (np.any(dp) or np.any(dr)):
            raise ValueError('Base mode requires zero arm increment')
        robot = self.env.robots[0]
        ctrl = robot.part_controllers['right']
        arm = np.r_[ctrl.origin_ori.T @ dp / .05, ctrl.origin_ori.T @ dr / .5]
        raw = robot.composite_controller.create_action_vector({
            'right': arm, 'right_gripper': [gripper], 'base': base.copy(),
            'torso': [0.], 'base_mode': 1 if base_mode else -1})
        submitted = np.array(raw, copy=True)
        self.obs, *_ = self.step(np.array(raw, copy=True))
        self.steps += 1
        if self.step_callback:
            self.step_callback(self, submitted)

    def move(self, name, target, rotation, gripper, phase, ticks=140,
             tolerance=.005, required=True):
        target, rotation = np.asarray(target), np.asarray(rotation)
        start = self.steps
        for _ in range(ticks):
            state = self.state(name)
            dp = target-state['eef']
            dr = Rotation.from_matrix(rotation @ state['eef_rotation'].T).as_rotvec()
            self.act(np.clip(.8*dp, -.015, .015), np.clip(.5*dr, -.12, .12), gripper)
            if np.linalg.norm(dp) < tolerance and np.linalg.norm(dr) < .04:
                break
        state = self.state(name)
        error = float(np.linalg.norm(target-state['eef']))
        angle = float(np.linalg.norm(Rotation.from_matrix(rotation @ state['eef_rotation'].T).as_rotvec()))
        row = {'phase_private': phase, 'start_step': start, 'end_step': self.steps,
               'target_position_world_m': target.tolist(), 'target_rotation_world': rotation.tolist(),
               'gripper': gripper, 'position_error_m': error,
               'orientation_error_rad': angle, 'required_waypoint': required,
               'object_private': name, 'state_private': state}
        self.phases.append(row)
        if self.phase_callback:
            self.phase_callback(self, row)
        if required and (error > .028 or angle > .15):
            raise RuntimeError(f'{phase}: waypoint error {error:.4f}m / {angle:.4f}rad')
        return row

    def hold(self, name, rotation, gripper, ticks, phase, required=True):
        return self.move(name, self.state(name)['eef'], rotation, gripper,
                         phase, ticks=ticks, tolerance=0., required=required)

    def navigate(self, name, xy, gripper=-1., phase='navigate', ticks=160, speed=(.20, .40)):
        """Drive the base to xy; `speed` bounds the commanded velocity (lower it when carrying)."""
        target, start = np.asarray(xy), self.steps
        low, high = speed
        for _ in range(ticks):
            pos, rot = self.env.robots[0].part_controllers['base'].get_base_pose()
            diff = target-np.asarray(pos)[:2]
            if np.linalg.norm(diff) < .018:
                break
            body = np.asarray(rot)[:2, :2].T @ diff
            velocity = np.where(np.abs(body) > .008,
                np.sign(body)*np.clip(low+1.2*np.abs(body), low, high), 0.)
            self.act(np.zeros(3), np.zeros(3), gripper, np.r_[velocity, 0.], True)
        for _ in range(12):
            self.act(np.zeros(3), np.zeros(3), gripper, base_mode=True)
        pos, _ = self.env.robots[0].part_controllers['base'].get_base_pose()
        error = float(np.linalg.norm(target-np.asarray(pos)[:2]))
        row = {'phase_private': phase, 'start_step': start, 'end_step': self.steps,
               'target_xy_world_m': target.tolist(), 'position_error_m': error,
               'required_waypoint': True, 'object_private': name}
        self.phases.append(row)
        if self.phase_callback:
            self.phase_callback(self, row)
        if error > .035:
            raise RuntimeError(f'{phase}: native base missed by {error:.4f}m')


def run_wrong_first_demo(motor, metadata, event, base_y=-.70,
                         carry_height=1.13, trial_ticks=60, stop_after_pick=False):
    """Run known wrong A, wrong B, correct C, recording every attempt boundary.

metadata contains candidates keyed by native body name (grasp_offset_local),
candidate_order, seated_root_position, and optional wrong_first_count. A trial
uses the same nominal seated target for every candidate. It does not infer the
candidate from private success; only the prerecorded route order knows the key.
"""
    order = list(metadata['candidate_order'])
    candidates = metadata['candidates']
    seated = np.asarray(metadata['seated_root_position'], dtype=float)
    source_roots = {name: motor.state(name)['position'].copy() for name in order}
    down = np.diag([-1., 1., -1.])
    attempts = []

    def carry(name, grip, phase):
        # Retract at a height that clears all handles before lateral base motion.
        current = motor.state(name)['eef'].copy()
        motor.move(name, [*current[:2], carry_height], down, grip, phase+'_raise')
        base, _ = motor.env.robots[0].part_controllers['base'].get_base_pose()
        motor.move(name, [base[0], base[1]+.35, carry_height], down, grip, phase+'_retract')

    def acquire(name, phase):
        state = motor.state(name)
        local = np.asarray(candidates[name]['grasp_offset_local'], dtype=float)
        anchor = state['position']+state['rotation'] @ local
        # A slight passive tilt after a failed fit is followed at its real pose.
        rotation = state['rotation'] @ down
        motor.move(name, anchor+[0., 0., .15], rotation, -1., phase+'_pregrasp')
        motor.move(name, anchor, rotation, -1., phase+'_approach')
        motor.hold(name, rotation, 1., 25, phase+'_close')
        before = motor.state(name)['position'].copy()
        motor.move(name, anchor+[0., 0., .16], rotation, 1., phase+'_lift')
        motor.hold(name, rotation, 1., 15, phase+'_lift_hold')
        lift = float(motor.state(name)['position'][2]-before[2])
        if lift < .075:
            raise RuntimeError(f'{phase}: object lift only {lift:.4f}m')
        # Reorient only while the object is safely above every neighbouring top.
        motor.move(name, motor.state(name)['eef'], down, 1., phase+'_upright')
        event(phase+'_acquired', {'candidate': name, 'lift_m': lift})
        return lift

    motor.hold(order[0], motor.state(order[0])['eef_rotation'], -1., 30, 'initial_settle')
    event('initial_settled', {'candidate_order_private': order})
    for index, name in enumerate(order):
        prefix = f'attempt_{index+1}'
        is_last = index == len(order)-1
        carry(name, -1., prefix+'_source_clearance')
        motor.navigate(name, [source_roots[name][0], base_y], phase=prefix+'_source_base')
        acquire(name, prefix)
        if stop_after_pick:
            event('pick_gate_complete', {'candidate': name, 'object_remains_held': True})
            return []
        carry(name, 1., prefix+'_transport_clearance')
        motor.navigate(name, [seated[0], base_y], gripper=1., phase=prefix+'_receiver_base')
        state = motor.state(name)
        offset = state['eef']-state['position']
        target = seated+offset
        motor.move(name, target+[0., 0., .15], down, 1., prefix+'_above_receiver')
        motor.move(name, target+[0., 0., .040], down, 1., prefix+'_near_receiver')
        # A wrong key blocks physical descent. A bounded compliant target is
        # allowed to remain unreached, and this failure is recorded explicitly.
        motor.move(name, target+[0., 0., .001], down, 1., prefix+'_fit_trial',
                   ticks=trial_ticks, tolerance=0., required=False)
        event(prefix+'_held_fit', {'candidate': name,
            'root_gap_m': float(motor.state(name)['position'][2]-seated[2])})
        motor.hold(name, down, -1., 25, prefix+'_release', required=False)
        motor.move(name, motor.state(name)['eef']+[0., 0., .15], down, -1., prefix+'_inspect_clearance')
        motor.hold(name, down, -1., 50, prefix+'_observe_fit')
        state = motor.state(name)
        attempt = {'candidate': name, 'index': index+1, 'final_candidate_private': is_last,
                   'released_root_gap_m': float(state['position'][2]-seated[2]),
                   'released_root_position': state['position'].tolist(),
                   'step': motor.steps}
        attempts.append(attempt)
        event(prefix+'_released_fit', attempt)
        if is_last:
            continue
        acquire(name, prefix+'_recover')
        carry(name, 1., prefix+'_return_clearance')
        motor.navigate(name, [source_roots[name][0], base_y], gripper=1., phase=prefix+'_return_base')
        offset = motor.state(name)['eef']-motor.state(name)['position']
        target = source_roots[name]+offset
        motor.move(name, target+[0., 0., .15], down, 1., prefix+'_return_above')
        motor.move(name, target+[0., 0., .003], down, 1., prefix+'_return_lower')
        motor.hold(name, down, -1., 25, prefix+'_return_release')
        motor.move(name, motor.state(name)['eef']+[0., 0., .15], down, -1., prefix+'_return_retreat')
        event(prefix+'_returned', {'candidate': name})
    motor.hold(order[-1], down, -1., 80, 'final_released_stability')
    event('terminal', {'attempts': attempts})
    return attempts
