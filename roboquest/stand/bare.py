"""Bare-MuJoCo harness for the stand: place a configuration at its predicted resting pose, settle, measure.

Used by the CPU tests (contact tuning facts without a kitchen) and by the tuning CLI in the job's tmp dir.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.stand import geometry as G
from roboquest.stand import statics as S

PLACE_LIFT = .0001
CONTROL_HZ = 20


def quat_wxyz(rotation):
    q = Rotation.from_matrix(rotation).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def build_bare(short_legs, delta_mm, shims_under_mm=None, yaw=0., ball=True, **options):
    """The stand at its predicted resting pose for the configuration, shims centred under the given legs,
    the ball at the top centre. Returns (model, data, meta, predicted pose)."""
    shims_under_mm = dict(shims_under_mm or {})
    pose = S.configuration_pose(short_legs, delta_mm, shims_under_mm)
    Rz = yaw_matrix(yaw)
    Rw = Rz @ pose['rotation']
    pos = np.array([0., 0., pose['z0'] + PLACE_LIFT])
    shims = []
    for leg, thickness in shims_under_mm.items():
        foot = pos + Rw @ np.asarray(G.foot_centre(leg, short_legs, delta_mm))
        shims.append(dict(pos=[float(foot[0]), float(foot[1]), thickness / 1000. / 2 + PLACE_LIFT / 2],
                          quat_wxyz=quat_wxyz(Rz), thickness_mm=thickness))
    ball_spec = dict(pos=(pos + Rw @ np.array([0., 0., G.BALL_RADIUS + PLACE_LIFT])).tolist()) if ball else None
    xml, meta = G.bare_model_xml(dict(pos=pos.tolist(), quat_wxyz=quat_wxyz(Rw), short_legs=short_legs,
                                      delta_mm=delta_mm), shims=shims, ball=ball_spec, textures=False, **options)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, meta, pose


class BareProbe:
    def __init__(self, model, data, meta):
        self.m, self.d, self.meta = model, data, meta
        geom = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, meta['stand']['body_name'])
        self.ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, meta['ball']['body_name']) if meta['ball'] else None
        self.foot_gids = {geom(g) for g in meta['stand']['foot_geoms'].values()}
        self.rim_gids = {geom(g) for g in meta['stand']['top_geoms'] if 'rim' in g}
        self.ball_gid = geom(meta['ball']['geom_name']) if meta['ball'] else None
        self.travel = meta['stand']['ball_travel']
        self.origin = np.array(data.xpos[self.stand])

    def tilt_deg(self):
        R = self.d.xmat[self.stand].reshape(3, 3)
        return math.degrees(math.acos(min(1., max(-1., float(R[2, 2])))))

    def ball_local(self):
        R = self.d.xmat[self.stand].reshape(3, 3)
        return R.T @ (self.d.xpos[self.ball] - self.d.xpos[self.stand])

    def _speed(self, bid):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
        return float(np.linalg.norm(v[3:])), float(np.linalg.norm(v[:3]))

    def ball_speed(self):
        return self._speed(self.ball)[0]

    def stand_speed(self):
        return self._speed(self.stand)

    def foot_penetration(self):
        worst = 0.
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 in self.foot_gids or c.geom2 in self.foot_gids) and c.dist < worst:
                worst = float(c.dist)
        return -worst

    def ball_at_rim(self):
        local = self.ball_local()
        if abs(local[0]) >= self.travel - .001 or abs(local[1]) >= self.travel - .001:
            return True
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.ball_gid in (c.geom1, c.geom2) and ({c.geom1, c.geom2} & self.rim_gids):
                return True
        return False

    def stand_drift(self):
        return float(np.linalg.norm(self.d.xpos[self.stand][:2] - self.origin[:2]))


def settle_bare(short_legs, delta_mm, shims_under_mm=None, seconds=3., yaw=.3, ball=True, measure_after=2.,
                **options):
    """Simulate a configuration and report what the gate checks look at.

    ``tilt_deg`` is the final tilt; ``tilt_max_after`` and ``ball_max_distance_after`` are taken over the time
    after ``measure_after`` seconds (a settled window); ``rim_time_s`` is when the ball first reaches the rim.
    """
    model, data, meta, pose = build_bare(short_legs, delta_mm, shims_under_mm, yaw=yaw, ball=ball, **options)
    probe = BareProbe(model, data, meta)
    steps = int(round(seconds / model.opt.timestep))
    every = int(round(1. / CONTROL_HZ / model.opt.timestep))
    rim_time, tilt_after, dist_after, penetration = None, [], [], []
    for step in range(steps):
        mujoco.mj_step(model, data)
        if (step + 1) % every:
            continue
        t = (step + 1) * model.opt.timestep
        penetration.append(probe.foot_penetration())
        if ball and rim_time is None and probe.ball_at_rim():
            rim_time = t
        if t >= measure_after:
            tilt_after.append(probe.tilt_deg())
            if ball:
                dist_after.append(float(np.linalg.norm(probe.ball_local()[:2])))
    return dict(predicted_tilt_deg=pose['tilt_deg'], support=pose['support'], tilt_deg=probe.tilt_deg(),
                tilt_max_after=max(tilt_after) if tilt_after else None, rim_time_s=rim_time,
                ball_max_distance_after=max(dist_after) if dist_after else None,
                ball_speed=probe.ball_speed() if ball else None,
                foot_penetration_m=max(penetration[len(penetration) // 2:]) if penetration else 0.,
                stand_speed=probe.stand_speed(), stand_drift_m=probe.stand_drift())
