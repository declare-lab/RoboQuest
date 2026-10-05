"""Privileged scripted motor for the search tasks through ordinary native actions.

Motor waypoints use simulator poses (fixture handles, hinge pivots, object poses,
the private fixture map for base planning). No object setters, attachments,
joint setters or camera changes. Not a policy result: a feasibility witness.

Ported from the room-search worktree (`room_search_motor.py`, recipes kept) onto
the RoboQuest base class: the tray and Submit live in the task frame
(``env.tray``), counter spots (``open`` places) are grasped top-down from a
standoff, and the wrist-image look is optional (headless runs have no images).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.keyed_fit_motor import KeyedFitMotor
from roboquest.search.layout import DRAWER_PULL_M, base_blockers, front_normal, yaw_matrix
from roboquest.search.planner import GridPlanner

DECISION_DECLARATION = dict(
    task_branch='Privileged: the visit order and the contents of every place are read from the private spec.',
    motor_privileges='Native fixture poses, handle geometry, hinge pivots, object poses and the private fixture map for base planning.',
    limitation='Privileged scripted motor; a feasibility certificate, not a learned policy evaluation.')

DOWN = np.diag([-1., 1., -1.])
DRAWER_FRONT = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])   # north-facing drawer bar handle, tool x down
DRAWER_FRONT_UP = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])  # same bar, tool x up (reach probe: 3 mm vs 18 mm)
DOOR_FRONT = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])     # north-facing vertical door bar
COLOUR_HSV = {  # OpenCV-style hue 0..180, saturation/value 0..255
    'blue': ((100, 130), 120, 60), 'red': ((0, 8), 120, 60), 'green': ((45, 85), 90, 50), 'yellow': ((20, 35), 120, 90),
}
CARRY_Z = 1.13
COVER_LIFT_M = .20      # the cloche rim clears every pool object (<= 0.111 m tall under the large cloche) at this lift
COVER_SETDOWN_GAP_M = .03   # the put-down rim keeps this clear of every other object's radius (build keep-out is 0.22 m)
DOOR_TILT_RAD = .90     # forward tilt of the hand-down pose for bars under a counter overhang (door probe 02: 7 mm, no contacts)
DOOR_GRIP_BELOW_TOP_M = .05   # grip site this far below the top of the vertical bar (palm clearance ~0.04 m)
DOOR_LOW_BAR_Z = .85          # bars whose top is below this sit under a counter lip and need the tilted grasp
DOOR_MID_BAR_Z = .70          # tilted grasps of bars topping above this start 0.10 m further from the front
UNHOOK_M = .10          # base back-off with the hand open that slides a handle bar out of the finger gap
DRAWER_VIEW_M = .71     # base frame to fixture front while inspecting an opened drawer (direct-11 geometry)
GRASP_AHEAD_M = .47     # top-down grasp anchor distance ahead of the base frame (0.49 m worked, 0.58 m hit joint limits)
GRASP_AHEAD_TOL_M = .03 # re-centring crawl outside this band (0.07 m left the above point 4.6 cm out of reach)
SHELF_REACH_M = .63     # base frame to the rim anchor for the horizontal shelf pinch (0.70 m left the hand 4 cm high, l1-19)
DOOR_LOOK_M = .62       # base frame to the cabinet front line while looking into an opened door
DOOR_OPEN_RAD = 1.30    # leaf angle the push aims for (about 75 degrees)
DOOR_PULL_RAD = .70     # leaf angle reached by the chorded base pull before the grip twists off the bar
DOOR_MIN_OPEN_RAD = .65 # smallest opening accepted before looking inside (RoboCasa leaves jam on their frame near 0.75 rad)
PRESS_AHEAD_M = .55     # base frame to the Submit button when the standoff beside the tray is boxed in


class Submitted(Exception):
    pass


def colour_blob(rgb, colour, min_pixels=400, centre_fraction=.6, max_aspect=5., min_fill=.3):
    """Largest compact blob of the requested colour in the central part of an RGB image."""
    import cv2
    h, w = rgb.shape[:2]
    m = int((1 - centre_fraction) / 2 * min(h, w))
    crop = np.ascontiguousarray(rgb[m:h - m, m:w - m])
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    (h0, h1), s_min, v_min = COLOUR_HSV[colour]
    mask = ((hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) & (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)).astype(np.uint8)
    count = int(mask.sum())
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_box = 0, None
    for label in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        fill = area / max(bw * bh, 1)
        if area >= min_pixels and aspect <= max_aspect and fill >= min_fill and area > best:
            best, best_box = area, [x + m, y + m, bw, bh]
    return {'pixels': count, 'largest_blob': best, 'blob_box': best_box, 'visible': best >= min_pixels,
            'threshold': min_pixels, 'centre_fraction': centre_fraction}


class Motor(KeyedFitMotor):
    def __init__(self, env, observation, horizon=20000, **kwargs):
        super().__init__(env, observation, horizon=horizon, **kwargs)
        self.planner = GridPlanner(env.task_spec['floor_bounds'], base_blockers(env._records))

    def act(self, *args, **kwargs):
        super().act(*args, **kwargs)
        if self.env.submission is not None:
            raise Submitted()

    # ---- base ----------------------------------------------------------------
    def base_pose(self):
        pos, rot = self.env.robots[0].part_controllers['base'].get_base_pose()
        return np.asarray(pos)[:2].copy(), float(math.atan2(rot[1, 0], rot[0, 0]))

    def turn(self, name, yaw, phase, gripper=-1., ticks=400, tolerance=.03):
        start = self.steps
        for _ in range(ticks):
            _, current = self.base_pose()
            error = (yaw - current + np.pi) % (2 * np.pi) - np.pi
            if abs(error) < tolerance:
                break
            # Gentler in-place turns while carrying: a friction-held cylinder slipped out of the top grasp
            # during the final turn at the tray standoff (layout 17, tin), landing on the base platform.
            top = .30 if gripper > 0 else .5
            velocity = np.sign(error) * np.clip(.22 + .8 * abs(error), .25, top)
            self.act(np.zeros(3), np.zeros(3), gripper, [0., 0., velocity], True)
        for _ in range(10):
            self.act(np.zeros(3), np.zeros(3), gripper, base_mode=True)
        _, current = self.base_pose()
        error = abs((yaw - current + np.pi) % (2 * np.pi) - np.pi)
        row = {'phase_private': phase, 'start_step': start, 'end_step': self.steps, 'target_yaw': float(yaw),
               'yaw_error_rad': float(error), 'required_waypoint': True, 'object_private': name}
        self.phases.append(row)
        if self.phase_callback:
            self.phase_callback(self, row)
        if error > .06:
            raise RuntimeError(f'{phase}: base yaw missed by {error:.3f} rad')

    def goto(self, name, xy, yaw, phase, gripper=-1., speed=(.20, .40)):
        """Plan, then drive each leg nose-first.

        The PandaOmron footprint box is 0.15 m ahead of the base frame but 0.55 m
        behind it, so legs are driven forward after turning to face them, and a
        large first turn at a standoff is preceded by a short reverse so the long
        rear does not sweep into the furniture the robot was facing.
        """
        start_xy, _ = self.base_pose()
        waypoints, info = self.planner.plan(start_xy, xy)
        for index, waypoint in enumerate(waypoints):
            cur_xy, cur_yaw = self.base_pose()
            delta_xy = np.subtract(waypoint, cur_xy)
            if np.linalg.norm(delta_xy) < .03:
                continue
            heading = math.atan2(delta_xy[1], delta_xy[0])
            turn_needed = (heading - cur_yaw + np.pi) % (2 * np.pi) - np.pi
            # Short legs are driven holonomically without turning: each in-place turn displaces the
            # frame by up to 0.4 m, which costs more than a 0.35 m sideways or backward crawl.
            if abs(turn_needed) > math.radians(35) and np.linalg.norm(delta_xy) > .35:
                if index == 0:
                    back = cur_xy - .25 * np.array([math.cos(cur_yaw), math.sin(cur_yaw)])
                    cell = self.planner.cell(back)
                    if self.planner.inside(cell) and self.planner.free[cell]:
                        # Tolerant: a start pose boxed in behind (layout 13 style 6) must not abort the route.
                        if gripper > 0:
                            try:
                                self.navigate(name, back, gripper=gripper, phase=f'{phase}_backoff', ticks=400, speed=speed)
                            except RuntimeError:
                                pass
                        else:
                            self.crawl(name, back, gripper=gripper, phase=f'{phase}_backoff', ticks=200)
                self.turn(name, heading, f'{phase}_face{index}', gripper=gripper)
            ticks = int(400 + 900 * np.linalg.norm(delta_xy) / max(speed[0], .05))
            self.navigate(name, waypoint, gripper=gripper, phase=f'{phase}_leg{index}', ticks=min(ticks, 3000), speed=speed)
        # The base rotates about a point about 0.20 m behind its frame, so a turn displaces the frame
        # by up to 2*0.20*sin(angle/2). Re-centre after turning; the second turn is small.
        for attempt in range(3):
            self.turn(name, yaw, f'{phase}_turn{attempt}', gripper=gripper)
            cur_xy, _ = self.base_pose()
            if np.linalg.norm(np.subtract(xy, cur_xy)) < .03:
                break
            self.navigate(name, xy, gripper=gripper, phase=f'{phase}_recentre{attempt}', ticks=400, speed=speed)
        return {'waypoints': waypoints, **info}

    def down(self):
        """Natural downward tool orientation for the current base heading (home wrist branch)."""
        _, yaw = self.base_pose()
        return Rotation.from_euler('z', yaw - np.pi / 2).as_matrix() @ DOWN

    def tilted_down(self, n, angle):
        """Hand-down pose pitched `angle` about the finger axis so the tool z leans toward -n
        (into the fixture front) and the wrist column leans back toward the robot."""
        down = self.down()
        axis = down[:, 0]
        for sign in (1., -1.):
            tilted = Rotation.from_rotvec(sign * angle * axis).as_matrix() @ down
            if float(np.dot(tilted[:, 2], -n)) > 0.:
                return tilted
        return down

    def tuck(self, name, phase, gripper=-1.):
        """Bring the hand to a carry pose ahead of the base before driving."""
        xy, yaw = self.base_pose()
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        current = self.state(name)['eef_rotation']
        rotation = current if gripper > 0 else self.down()
        if gripper <= 0 and abs(float(current[2, 2])) < .5:
            # From a forward hand (door look pose) the straight move to the carry point winds joint
            # 6: pitch to hand-down first at a farther, mid-height point.
            self.move(name, np.r_[xy + .45 * forward, .95], rotation, gripper, phase + '_tuck_pitch', ticks=160, tolerance=.03, required=False)
        if gripper > 0 and abs(float(rotation[2, 2])) < .3:
            # Object pinched from the front with a horizontal hand (shelf grasp): keep it upright and
            # carry it 0.40 m ahead just above counter height.
            target = np.r_[xy + .40 * forward, CARRY_Z - .08]
        else:
            target = np.r_[xy + .25 * forward, CARRY_Z]
        self.move(name, target, rotation, gripper, phase + '_tuck', ticks=200, tolerance=.01)
        if gripper <= 0:
            self.move(name, self.state(name)['eef'], self.down(), gripper, phase + '_tuck_upright', ticks=120)

    def relax(self, name, phase, gripper=-1.):
        """Tolerant move to the carry pose: lets the arm leave a joint-limit posture."""
        xy, yaw = self.base_pose()
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        target = np.r_[xy + .25 * forward, CARRY_Z]
        self.move(name, target, self.down(), gripper, phase + '_relax', ticks=160, tolerance=.02, required=False)

    # ---- geometry helpers ---------------------------------------------------------
    def handle_centre(self, prefix):
        m, d = self.env.sim.model, self.env.sim.data
        points = []
        for g in range(m.ngeom):
            gn = m.geom_id2name(g) or ''
            if gn.startswith(prefix) and m.geom_contype[g] != 0:
                half = np.asarray(m.geom_size[g])[:3]
                if m.geom_type[g] == 6:  # box
                    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])])
                    points.extend(corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
                else:
                    r = float(half[0])
                    points.extend([d.geom_xpos[g] + np.array([sx, sy, sz]) * r for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        if not points:
            raise RuntimeError(f'no collision handle geoms with prefix {prefix}')
        points = np.asarray(points)
        return (points.min(0) + points.max(0)) / 2, points.min(0), points.max(0)

    # ---- compartments ----------------------------------------------------------
    def open_drawer(self, name, comp, opening=DRAWER_PULL_M, phase='drawer', return_to_view=True):
        """Grasp the bar handle, pull with the arm to 0.40 m from the base frame, then reverse the base
        for the remainder while holding it; unhook, rise, return to the look distance (``return_to_view``;
        skipped when the grasp follows at once, which also keeps a self-closing drawer from shutting)."""
        n = np.r_[front_normal(comp['rot']), 0.]
        front = yaw_matrix(comp['rot']) @ DRAWER_FRONT_UP
        joint = comp['joints'][0]
        handle, _, _ = self.handle_centre(comp['fixture'] + '_door_handle_')
        self.move(name, handle + n * .18 + [0., 0., .12], front, -1., phase + '_prehandle', ticks=250)
        self.move(name, handle + n * .12, front, -1., phase + '_align', ticks=160)
        self.move(name, handle + n * .004, front, -1., phase + '_approach', ticks=120)
        self.hold(name, front, 1., 20, phase + '_grasp')
        q = float(self.env.sim.data.get_joint_qpos(joint))
        closed = handle - n * abs(q)
        arm_pull = float(np.clip(comp['standoff']['distance_from_front'] - .40, 0., opening))
        self.move(name, closed + n * (arm_pull + .004), front, 1., phase + '_arm_pull', ticks=220, tolerance=.01, required=False)
        after_arm = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        base_xy, yaw = self.base_pose()
        facing = np.array([math.cos(yaw), math.sin(yaw)])
        remaining = opening - after_arm
        moved_back = 0.
        if remaining > .02:
            self.crawl(name, base_xy - facing * remaining, gripper=1., phase=phase + '_base_pull',
                       ticks=int(remaining / .006) + 150)
            moved_back = float(np.linalg.norm(self.base_pose()[0] - base_xy))
        self.hold(name, front, 1., 20, phase + '_hold')
        achieved = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        self.hold(name, front, -1., 15, phase + '_release')
        # The opened fingers still straddle the bar: back the base off with the hand open, then rise.
        unhook = UNHOOK_M if return_to_view else .06
        self.crawl(name, self.base_pose()[0] - facing * unhook, gripper=-1., phase=phase + '_unhook', ticks=80 if return_to_view else 20)
        free = self.state(name)['eef'].copy()
        clear_z = float(comp['pos'][2]) + float(comp['size'][2]) / 2 + .12
        # Straight up in the current orientation: chasing the bar orientation here never met the break test, so
        # the move ran out its budget short of the top and the base's next crawl pushed the drawer shut with the
        # hand (10 cm in 1.4 s, measured). Position-only convergence clears the front panel in about a second.
        self.move(name, np.r_[free[:2], clear_z], self.state(name)['eef_rotation'].copy(), -1., phase + '_retreat',
                  ticks=80, tolerance=.02, required=False)
        view_xy = base_xy - facing * max(0., DRAWER_VIEW_M - float(comp['standoff']['distance_from_front']))
        if return_to_view and np.linalg.norm(view_xy - self.base_pose()[0]) > .02:
            self.crawl(name, view_xy, gripper=-1., phase=phase + '_base_return', ticks=200)
        if achieved < .34:
            raise RuntimeError(f'{phase}: drawer opened only {achieved:.3f} m (arm {after_arm:.3f}, base {moved_back:.3f})')
        return {'joint': joint, 'achieved_open_m': achieved, 'arm_pull_m': after_arm, 'base_reverse_m': moved_back}

    def crawl(self, name, xy, gripper, phase, ticks, stop=.012, speed=.5):
        """Drive the base toward xy at ``speed`` (full command by default) without raising on a stall;
        a carried object that hangs from a pinch (the dark search's lantern) asks for a gentler one."""
        target, start = np.asarray(xy, float), self.steps
        for _ in range(ticks):
            pos, rot = self.env.robots[0].part_controllers['base'].get_base_pose()
            diff = target - np.asarray(pos)[:2]
            if np.linalg.norm(diff) < stop:
                break
            body = np.asarray(rot)[:2, :2].T @ diff
            velocity = np.where(np.abs(body) > .01, np.sign(body) * float(speed), 0.)
            self.act(np.zeros(3), np.zeros(3), gripper, np.r_[velocity, 0.], True)
        for _ in range(12):
            self.act(np.zeros(3), np.zeros(3), gripper, base_mode=True)
        pos, _ = self.env.robots[0].part_controllers['base'].get_base_pose()
        row = {'phase_private': phase, 'start_step': start, 'end_step': self.steps,
               'target_xy_world_m': target.tolist(), 'position_error_m': float(np.linalg.norm(target - np.asarray(pos)[:2])),
               'required_waypoint': False, 'object_private': name}
        self.phases.append(row)
        if self.phase_callback:
            self.phase_callback(self, row)

    def door_state(self, comp):
        side = comp.get('open_side', '')
        joint = next(j for j in comp['joints'] if (side + 'doorhinge') in j) if side else comp['joints'][0]
        lo, hi = comp['joint_ranges'][joint]
        direction = 1. if hi > 0 else -1.
        jid = self.env.sim.model.joint_name2id(joint)
        pivot = self.env.sim.data.xanchor[jid].copy()
        prefix = comp['fixture'] + (f'_{side}_door_handle_' if side else '_door_handle_')
        return joint, direction, pivot, prefix

    def door_frame(self, comp):
        n = np.r_[front_normal(comp['rot']), 0.]
        t = np.cross(n, [0., 0., 1.])
        joint, direction, pivot, prefix = self.door_state(comp)
        handle, _, _ = self.handle_centre(prefix)
        angle = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        closed_span = Rotation.from_euler('z', -direction * angle).as_matrix() @ (handle - pivot)
        width = float(np.linalg.norm(closed_span[:2]))
        sign = float(np.sign(np.dot(closed_span[:2], t[:2]))) or 1.
        edge_t = float(np.dot(pivot[:2], t[:2])) + sign * width * math.cos(angle)
        return {'n': n, 't': t, 'sign': sign, 'pivot': pivot, 'width': width, 'angle': angle,
                'edge_t': edge_t, 'joint': joint, 'direction': direction}

    def stand_before(self, name, comp, point, ahead, phase, margin=.35):
        """Crawl the base so `point` (on or near the fixture front) is `ahead` metres straight ahead,
        keeping `margin` lateral distance from the open leaf's free edge."""
        f = self.door_frame(comp)
        n, t = f['n'][:2], f['t'][:2]
        goal = np.asarray(point, float)[:2] + n * ahead
        if margin is not None:
            goal_t = float(np.dot(goal, t))
            limit = f['edge_t'] + f['sign'] * margin
            if f['sign'] * (goal_t - limit) < 0.:
                goal = goal + t * (limit - goal_t)
        if np.linalg.norm(goal - self.base_pose()[0]) > .02:
            self.crawl(name, goal, gripper=-1., phase=phase, ticks=300)
        return goal

    def open_door(self, name, comp, open_rad=DOOR_OPEN_RAD, phase='door'):
        """Vertical bar handle: tilted top-down grasp of the bar's top, chorded base pull along the hinge
        arc, release, unhook, rise, then a closed-hand push of the leaf to `open_rad`."""
        n = np.r_[front_normal(comp['rot']), 0.]
        joint, direction, pivot, prefix = self.door_state(comp)
        handle, low, high = self.handle_centre(prefix)
        grip = handle.copy()
        grip[2] = float(high[2] - DOOR_GRIP_BELOW_TOP_M)
        if high[2] < DOOR_LOW_BAR_Z:
            if high[2] >= DOOR_MID_BAR_Z:
                base_xy0, yaw0 = self.base_pose()
                facing0 = np.array([math.cos(yaw0), math.sin(yaw0)])
                self.crawl(name, base_xy0 - facing0 * .10, gripper=-1., phase=phase + '_back_off', ticks=60)
            down = self.tilted_down(n, DOOR_TILT_RAD)
            above = grip + n * .012 - down[:, 2] * .26
        else:
            down = self.down()
            above = grip + n * .06 + [0., 0., .22]
        self.move(name, above, down, -1., phase + '_above', ticks=250)
        self.move(name, grip + n * .012, down, -1., phase + '_descend', ticks=160, tolerance=.008)
        self.hold(name, down, 1., 25, phase + '_grasp')
        f = self.door_frame(comp)
        hinge_dir = -f['sign'] * f['t'][:2]
        base_xy, _ = self.base_pose()
        for index, theta in enumerate(np.linspace(DOOR_PULL_RAD / 2., DOOR_PULL_RAD, 2)):
            delta = n[:2] * f['width'] * math.sin(theta) + hinge_dir * f['width'] * (1. - math.cos(theta))
            self.crawl(name, base_xy + delta, gripper=1., phase=f'{phase}_base_pull_{index}',
                       ticks=int(np.linalg.norm(delta) / .006) + 100)
        pulled = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        rotation = self.state(name)['eef_rotation'].copy()
        self.hold(name, rotation, -1., 20, phase + '_release')
        leaf_normal = Rotation.from_euler('z', direction * pulled).as_matrix() @ n
        self.crawl(name, self.base_pose()[0] + leaf_normal[:2] * UNHOOK_M, gripper=-1., phase=phase + '_unhook', ticks=80)
        free = self.state(name)['eef'].copy()
        self.move(name, free + [0., 0., .20], rotation, -1., phase + '_up', ticks=140, required=False)
        push_point = f['pivot'][:2] + f['t'][:2] * f['sign'] * .30
        self.stand_before(name, comp, push_point, .70, phase + '_push_stand', margin=None)
        self.hold(name, self.state(name)['eef_rotation'].copy(), 1., 15, phase + '_close_hand')
        edge_vector = handle - pivot
        radial = edge_vector * (1. - .15 / max(f['width'], .15))
        push_z = min(grip[2] - .10, float(comp['pos'][2]) + float(comp['size'][2]) / 2 - .12)
        vertical = self.down()
        current = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        angles = [current - .30] + list(np.linspace(current + .20, open_rad, 3))
        for index, angle in enumerate(angles):
            rot = Rotation.from_euler('z', direction * angle).as_matrix()
            target = pivot + rot @ radial
            if index == 0:
                self.move(name, np.r_[target[:2], push_z + .30], rot @ vertical, 1., f'{phase}_push_over', ticks=140, tolerance=.015, required=False)
            self.move(name, np.r_[target[:2], push_z], rot @ vertical, 1., f'{phase}_push_{index}', ticks=110, tolerance=.012, required=False)
        achieved = abs(float(self.env.sim.data.get_joint_qpos(joint)))
        self.move(name, self.state(name)['eef'] + [0., 0., .20], self.state(name)['eef_rotation'], -1., phase + '_clear', ticks=140, required=False)
        self.relax(name, phase + '_push')
        if achieved < DOOR_MIN_OPEN_RAD:
            raise RuntimeError(f'{phase}: door opened only {achieved:.3f} rad (after the arc pull {pulled:.3f})')
        return {'joint': joint, 'achieved_open_rad': achieved, 'direction': direction, 'base_pull_rad': pulled}

    def look_inside(self, name, comp, phase):
        """Aim the wrist camera at the opened compartment; returns the wrist image when the env renders."""
        env = self.env
        n = np.r_[front_normal(comp['rot']), 0.]
        cid = env.sim.model.camera_name2id('robot0_eye_in_hand')
        state = self.state(name)
        eef_to_camera_rot = state['eef_rotation'].T @ env.sim.data.cam_xmat[cid].reshape(3, 3)
        eef_to_camera_pos = state['eef_rotation'].T @ (env.sim.data.cam_xpos[cid] - state['eef'])
        centre = np.asarray(comp['region']['centre_world'])
        floor_z = comp['region']['floor_z']
        if comp['kind'] == 'drawer':
            look_at = centre + n * (comp['region']['size_xy'][1] / 2 + .20)
            _, base_yaw = self.base_pose()
            camera_rotation = Rotation.from_euler('z', base_yaw - np.pi / 2).as_matrix()
            camera_target = np.r_[look_at[:2], floor_z + .45]
        else:
            f = self.door_frame(comp)
            front_point = f['pivot'][:2] + f['t'][:2] * f['sign'] * f['width'] / 2
            self.stand_before(name, comp, front_point, DOOR_LOOK_M, phase + '_stand')
            look_at = np.r_[front_point + n[:2] * .22, 0.]
            camera_target = np.r_[look_at[:2], floor_z + .17]
            camera_rotation = None
        if camera_rotation is None:
            rotation = yaw_matrix(comp['rot']) @ DRAWER_FRONT_UP
        else:
            rotation = camera_rotation @ eef_to_camera_rot.T
        target = camera_target - rotation @ eef_to_camera_pos
        self.move(name, target, rotation, -1., phase + '_view', ticks=200, tolerance=.01, required=False)
        self.hold(name, rotation, -1., 12, phase + '_settle', required=False)
        key = 'robot0_eye_in_hand_image'
        if isinstance(self.obs, dict) and key in self.obs:
            return np.asarray(self.obs[key])[::-1].copy()
        return None

    # ---- objects ------------------------------------------------------------------
    def grasp_top_down(self, name, phase, min_above_z=None, above_m=.16, lift_m=.18):
        """Top-down grasp of an exposed object (opened drawer or counter spot), re-centring the base so
        the anchor sits GRASP_AHEAD_M ahead of the base frame. ``min_above_z`` lifts the approach point
        above a drawer's front panel (the hand brushing it pushes a self-closing drawer shut).
        ``above_m`` / ``lift_m`` are the pre-grasp point and the lift above the anchor: the defaults suit
        the pool objects; a tall handle under a wall unit (the dark search's lantern) passes smaller ones,
        because the wrist tops about 0.11 m above the hand."""
        state = self.state(name)
        grasp = self.env.task_spec['objects'][name]['grasp']
        yaw0 = Rotation.from_matrix(state['rotation']).as_euler('zyx')[0]
        options = [yaw0, yaw0 + np.pi] if grasp.get('handle') else [yaw0 + k * np.pi / 4 for k in range(8)]
        natural = self.down()

        def twist(yaw):
            candidate = Rotation.from_euler('z', yaw).as_matrix() @ DOWN
            return float(np.linalg.norm(Rotation.from_matrix(candidate @ natural.T).as_rotvec()))
        yaw = min(options, key=twist)
        rotation = Rotation.from_euler('z', yaw).as_matrix() @ DOWN
        anchor = state['position'] + state['rotation'] @ np.asarray(grasp['anchor_local_top'], float)
        base_xy, base_yaw = self.base_pose()
        facing = np.array([math.cos(base_yaw), math.sin(base_yaw)])
        ahead = float(np.dot(anchor[:2] - base_xy, facing))
        if abs(ahead - GRASP_AHEAD_M) > GRASP_AHEAD_TOL_M:
            self.crawl(name, base_xy + facing * (ahead - GRASP_AHEAD_M), gripper=-1., phase=phase + '_base_centre', ticks=200)
        # 0.16 m above the anchor clears a base-cabinet drawer's front panel (0.12 m did not: the hand pushed the
        # drawer shut on its way in).
        above = anchor + [0., 0., float(above_m)]
        if min_above_z is not None:
            above[2] = max(above[2], float(min_above_z))
        self.move(name, above, rotation, -1., phase + '_above', ticks=120, tolerance=.01)
        self.move(name, anchor, rotation, -1., phase + '_lower', ticks=140, tolerance=.006)
        self.hold(name, rotation, 1., 25, phase + '_close')
        before = state['position'][2]
        self.move(name, anchor + [0., 0., float(lift_m)], rotation, 1., phase + '_lift', ticks=160)
        lift = float(self.state(name)['position'][2] - before)
        if lift < min(.08, .8 * float(lift_m)):
            raise RuntimeError(f'{phase}: object lift only {lift:.3f} m')
        return lift

    grasp_from_drawer = grasp_top_down

    # ---- covers (v1: `covered` places) ---------------------------------------------
    def setdown_choice(self, cover_name):
        """The first recorded set-down option for a cover that is clear of every other object (privileged
        positions) and whose matching base shift stays on planner-free floor; None when there is none."""
        env = self.env
        cover = env.covers[cover_name]
        radius = float(cover['radius'])
        others = [n for n in env.objects if n not in (cover_name, cover['target'])]
        base_xy, _ = self.base_pose()
        cover_xy = self.state(cover_name)['position'][:2]
        for option in cover.get('setdown_options', []):
            xy = np.asarray(option['xy'], float)
            clear = all(np.linalg.norm(self.state(n)['position'][:2] - xy)
                        >= radius + float(env.objects[n].horizontal_radius) + COVER_SETDOWN_GAP_M for n in others)
            if clear and self._free(base_xy + (xy - cover_xy)):
                return option
        return None

    def lift_cover(self, cover_name, phase):
        """Pinch the cover's knob top-down, lift it clear of the object, shift the base sideways with the cover
        in hand, put it down at a free set-down spot, release and come back. Returns the lift and set-down."""
        env = self.env
        cover = env.covers[cover_name]
        option = self.setdown_choice(cover_name)
        if option is None:
            raise RuntimeError(f'{phase}: no clear set-down spot for {cover_name}')
        m, d = env.sim.model, env.sim.data
        knob = d.site_xpos[m.site_name2id(cover['knob_site'])].copy()
        rotation = self.down()
        base_xy, base_yaw = self.base_pose()
        facing = np.array([math.cos(base_yaw), math.sin(base_yaw)])
        ahead = float(np.dot(knob[:2] - base_xy, facing))
        if abs(ahead - GRASP_AHEAD_M) > .03:
            self.crawl(cover_name, base_xy + facing * (ahead - GRASP_AHEAD_M), gripper=-1., phase=phase + '_base_centre', ticks=200)
        knob = d.site_xpos[m.site_name2id(cover['knob_site'])].copy()
        self.move(cover_name, knob + [0., 0., .14], rotation, -1., phase + '_above', ticks=200)
        self.move(cover_name, knob, rotation, -1., phase + '_descend', ticks=140, tolerance=.006)
        self.hold(cover_name, rotation, 1., 25, phase + '_close')
        before = float(self.state(cover_name)['position'][2])
        self.move(cover_name, knob + [0., 0., COVER_LIFT_M], rotation, 1., phase + '_lift', ticks=200)
        lift = float(self.state(cover_name)['position'][2] - before)
        if lift < .12 or not self.state_grasped(cover_name):
            raise RuntimeError(f'{phase}: cover lifted only {lift:.3f} m')
        start_xy, _ = self.base_pose()
        delta = np.asarray(option['xy'], float) - self.state(cover_name)['position'][:2]
        self.navigate(cover_name, start_xy + delta, gripper=1., phase=phase + '_shift', ticks=500, speed=(.08, .20))
        eef = self.state(cover_name)['eef'].copy()
        rest_z = eef[2] - (float(self.state(cover_name)['position'][2]) - (float(option['top_z']) + .004))
        self.move(cover_name, np.r_[eef[:2], rest_z], rotation, 1., phase + '_lower', ticks=200, tolerance=.006, required=False)
        self.hold(cover_name, rotation, -1., 20, phase + '_release')
        self.move(cover_name, self.state(cover_name)['eef'] + [0., 0., .15], rotation, -1., phase + '_retreat', ticks=140,
                  tolerance=.03, required=False)
        self.navigate(cover_name, start_xy, gripper=-1., phase=phase + '_return', ticks=500, speed=(.08, .25))
        settled = self.state(cover_name)['position']
        error = float(np.linalg.norm(settled[:2] - np.asarray(option['xy'])))
        return dict(lift_m=lift, setdown_xy=[float(v) for v in settled[:2]], setdown_error_m=error, side=option['side'])

    def grasp_from_shelf(self, name, comp, phase):
        """Horizontal front pinch on the object's body centre, extract along the front normal, raise."""
        n = np.r_[front_normal(comp['rot']), 0.]
        front = yaw_matrix(comp['rot']) @ DOOR_FRONT @ np.diag([-1., -1., 1.])
        base_xy, yaw = self.base_pose()
        facing = np.array([math.cos(yaw), math.sin(yaw)])
        park_z = min(float(comp['pos'][2]) + float(comp['size'][2]) / 2 + .17, .85)
        self.move(name, np.r_[base_xy + facing * .40, park_z], self.state(name)['eef_rotation'].copy(), -1.,
                  phase + '_park', ticks=140, tolerance=.02, required=False)
        state = self.state(name)
        grasp = self.env.task_spec['objects'][name]['grasp']
        anchor = state['position'] + state['rotation'] @ np.asarray(grasp['anchor_local_front'], float)
        self.stand_before(name, comp, anchor[:2], SHELF_REACH_M, phase + '_stand')
        self.move(name, anchor + n * .18 + [0., 0., .17], front, -1., phase + '_front_high', ticks=200)
        self.move(name, anchor + n * .10 + [0., 0., .02], front, -1., phase + '_front', ticks=160)
        self.move(name, anchor, front, -1., phase + '_approach', ticks=160, tolerance=.012)
        self.hold(name, front, 1., 25, phase + '_close')
        before = state['position'].copy()
        self.move(name, anchor + [0., 0., .05], front, 1., phase + '_lift', ticks=120)
        self.move(name, anchor + n * .24 + [0., 0., .05], front, 1., phase + '_extract', ticks=180)
        lifted = self.state(name)['position']
        moved = float(np.dot(lifted - before, n))
        if moved < .20 or not self.state_grasped(name):
            raise RuntimeError(f'{phase}: object not extracted ({moved:.3f} m)')
        self.move(name, self.state(name)['eef'] + [0., 0., .25], front, 1., phase + '_raise', ticks=160)
        return moved

    def state_grasped(self, name):
        robot = self.env.robots[0]
        return bool(self.env._check_grasp(robot.gripper['right'], self.env.objects[name]))

    def place_on_tray(self, name, phase, slot=(0., 0.)):
        """Lower the held object onto the tray square at a frame-local slot offset and release."""
        tray = self.env.tray
        state = self.state(name)
        offset = state['eef'] - state['position']
        bottom = float(self.env.objects[name].bottom_offset[2])
        centre = np.asarray(tray['centre_xy'], float) + (yaw_matrix(tray['yaw']) @ np.r_[slot, 0.])[:2]
        target = np.r_[centre, tray['top_z'] - bottom + .006] + offset
        rotation = state['eef_rotation'].copy()
        self.move(name, target + [0., 0., .15], rotation, 1., phase + '_above', ticks=200)
        self.move(name, target, rotation, 1., phase + '_lower', ticks=160, tolerance=.006)
        self.hold(name, rotation, -1., 25, phase + '_release')
        self.move(name, self.state(name)['eef'] + [0., 0., .15], rotation, -1., phase + '_retreat', ticks=140, tolerance=.06, required=False)

    def _free(self, xy):
        cell = self.planner.cell(xy)
        return self.planner.inside(cell) and bool(self.planner.free[cell])

    def press_submit(self, name, phase='submit'):
        """Stand so the button is straight ahead, then a top-down press with the closed empty hand.

        The sideways shift from the tray standoff to the Submit standoff is clipped to
        planner-free floor (on layout 13 style 6 the next counter group boxes it in);
        when the button is still more than 0.12 m off-axis, the base tucks the hand,
        turns to face the button and crawls to PRESS_AHEAD_M from it.
        """
        button = np.asarray(self.env.task_spec['submit_button']['position'])
        tray = self.env.tray
        base_xy, _ = self.base_pose()
        goal = np.asarray(tray['submit_standoff']['xy'], float)
        shift = goal - base_xy
        distance = float(np.linalg.norm(shift))
        best = base_xy.copy()
        if distance > 1e-6:
            if self._free(goal):
                best = goal
            else:
                for s in np.arange(.05, distance, .05):
                    point = base_xy + shift / distance * s
                    if not self._free(point):
                        break
                    best = point
        if np.linalg.norm(best - base_xy) > .02:
            self.crawl(name, best, gripper=-1., phase=phase + '_shift', ticks=300)
        base_xy, yaw = self.base_pose()
        facing = np.array([math.cos(yaw), math.sin(yaw)])
        relative = button[:2] - base_xy
        lateral = float(facing[0] * relative[1] - facing[1] * relative[0])
        if abs(lateral) > .12:
            self.tuck(name, phase + '_reaim')
            away = base_xy - button[:2]
            away = away / max(float(np.linalg.norm(away)), 1e-6)
            target = button[:2] + away * PRESS_AHEAD_M
            heading = math.atan2(-away[1], -away[0])
            self.turn(name, heading, phase + '_face', gripper=-1.)
            self.crawl(name, target, gripper=-1., phase=phase + '_approach_base', ticks=300)
            self.turn(name, heading, phase + '_face_again', gripper=-1.)
        down = self.down()
        self.move(name, button + [0., 0., .16], down, -1., phase + '_above', ticks=200)
        self.hold(name, down, 1., 25, phase + '_close_empty_hand')
        try:
            self.move(name, button + [0., 0., .040], down, 1., phase + '_approach', ticks=120)
            self.move(name, button + [0., 0., -.020], down, 1., phase + '_press', ticks=100, tolerance=0., required=False)
        except Submitted:
            return True
        raise RuntimeError('Physical button press did not submit')
