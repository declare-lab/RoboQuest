"""Native RoboCasa365 room/object construction and evaluator-only contact state.

All object placements are construction/reset operations. The policy gets the
original fixed three RGB cameras and robot proprioception only. Rooms, textures,
meshes, mass/inertia and collisions remain native unless a task records a change.
"""
from pathlib import Path
import hashlib
import itertools
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils import env_utils as EnvUtils

from roboquest.base.kitchen_scene import TableSettingScene
from roboquest.base.scene_contracts import CAMERAS, policy_observation, all_resting, in_zone, upright, evaluate_pick_place

ASSET_ROOT = Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime'))) / 'src/robocasa/robocasa/models/assets/objects'
_CORNERS = np.asarray(list(itertools.product((-1., 1.), repeat=3)))


class VisionKitchenScene(Kitchen):
    PUBLIC_GOAL = ''
    CATEGORY = ''
    ROBOT_FIXTURE = 'island_island_group'

    def __init__(self, variant=0, **kwargs):
        if isinstance(variant, bool) or not isinstance(variant, int) or not 0 <= variant < 100:
            raise ValueError('variant must be an integer in [0,99]')
        self.variant = variant
        self.instruction = self.PUBLIC_GOAL
        self.asset_evidence = {}
        self.task_spec = {'objects': {}, 'zones': {}, 'supports': {}, 'instruction': self.instruction}
        self._poses = {}
        super().__init__(**kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_ref = self.get_fixture(self.ROBOT_FIXTURE)

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        Kitchen._create_objects(self)
        self._poses = {}
        self.task_spec = {'objects': {}, 'zones': {}, 'supports': {}, 'instruction': self.instruction}
        self.asset_evidence = {}
        self.build_task()
        # Deferred merge permits task-owned visual marks on native objects.
        for name, obj in self.objects.items():
            if name in self._poses:
                pose = self._poses[name]
                obj.get_obj().set('pos', ' '.join(map(str, pose[:3])))
                obj.get_obj().set('quat', ' '.join(map(str, pose[3:])))
            self.model.merge_objects([obj])

    def build_task(self):
        raise NotImplementedError

    def fixture_top(self, name):
        fixture = self.get_fixture(name)
        center, local_size = np.asarray(fixture.pos).copy(), np.asarray(fixture.size).copy()
        rotation = Rotation.from_euler('z', float(fixture.rot)).as_matrix()
        size = np.abs(rotation) @ local_size
        top = float(center[2] + size[2]/2)
        # Native support identity is retained for terminal and asset audits.
        self.task_spec['supports'][fixture.name] = {
            'bounds_xy': [float(center[0]-size[0]/2), float(center[0]+size[0]/2),
                          float(center[1]-size[1]/2), float(center[1]+size[1]/2)],
            'height_z': top, 'fixture': fixture.name, 'rotation_z_rad': float(fixture.rot),
        }
        return center, size, top

    def add_native(self, name, relative_asset_path, xy, support_z, scale=1., quat_xyzw=None, rgba=None):
        path = ASSET_ROOT / relative_asset_path
        if path.suffix != '.xml':
            path = path / 'model.xml'
        arguments = {} if rgba is None else {'rgba': rgba}
        obj = MJCFObject(name=name, mjcf_path=str(path), scale=scale, **arguments)
        self.objects[name] = obj
        q = [0., 0., 0., 1.] if quat_xyzw is None else list(quat_xyzw)
        z = float(support_z) - float(obj.bottom_offset[2]) + .002
        self._poses[name] = np.r_[xy, z, q[3], q[:3]].tolist()
        self.task_spec['objects'][name] = {'kind': 'object', 'fixed': False}
        self.asset_evidence[name] = {'source': str(path), 'xml_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'uniform_scale': float(scale), 'size_m': list(obj.size), 'bottom_offset_m': list(obj.bottom_offset),
            'native_collision': True}
        return obj

    def add_zone(self, name, bounds_xy, z, rgba=(.05, .2, .85, 1.)):
        x0, x1, y0, y1 = map(float, bounds_xy)
        if x0 >= x1 or y0 >= y1:
            raise ValueError('zone bounds must be ordered')
        self.task_spec['zones'][name] = {'bounds_xy': [x0,x1,y0,y1], 'height_z': float(z)}
        for support_id, support in self.task_spec['supports'].items():
            a,b,c,d = support['bounds_xy']
            if (abs(support['height_z']-z) < 1e-6
                    and a <= x0 < x1 <= b and c <= y0 < y1 <= d):
                self.task_spec['zones'][name]['support_id'] = support_id
                break
        for i,(x,y,sx,sy) in enumerate([
            ((x0+x1)/2,y0,(x1-x0)/2,.003), ((x0+x1)/2,y1,(x1-x0)/2,.003),
            (x0,(y0+y1)/2,.003,(y1-y0)/2), (x1,(y0+y1)/2,.003,(y1-y0)/2),
        ]):
            ET.SubElement(self.model.worldbody, 'geom', name=f'vision_{name}_mark_{i}', type='box',
                size=f'{sx} {sy} .0005', pos=f'{x} {y} {z+.001}',
                rgba=' '.join(map(str, rgba)), contype='0', conaffinity='0', group='1')

    def _reset_internal(self):
        Kitchen._reset_internal(self)
        for name, pose in self._poses.items():
            obj = self.objects[name]
            if obj.joints:
                self.sim.data.set_joint_qpos(obj.joints[0], pose)
                self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.sim.forward()

    _collision_geom_ids = TableSettingScene._collision_geom_ids
    robot_proprio = TableSettingScene.robot_proprio

    def _world_collision_points(self, name):
        m, d = self.sim.model, self.sim.data
        points = []
        for g in self._collision_geom_ids(name):
            typ = int(m.geom_type[g])
            if typ == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[g]); start = int(m.mesh_vertadr[mesh]); count = int(m.mesh_vertnum[mesh])
                local = np.asarray(m.mesh_vert[start:start+count])
            elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                local = _CORNERS*m.geom_size[g]
            elif typ in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                r, half = m.geom_size[g][:2]
                # Conservative bounding box for uncommon analytic additions.
                if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    half += r
                local = _CORNERS*np.asarray([r,r,half])
            elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                local = _CORNERS*m.geom_size[g][0]
            else:
                raise ValueError(f'unsupported collision geometry {typ} for {name}')
            points.extend(local @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g])
        if not points:
            raise ValueError(f'no collision geometry for {name}')
        return np.asarray(points)

    def evaluator_snapshot(self, step=0):
        """Native truth for the evaluator; never returned by a policy action."""
        m, d = self.sim.model, self.sim.data
        groups = {name: set(self._collision_geom_ids(name)) for name in self.objects}
        geom_owner = {g:name for name, geoms in groups.items() for g in geoms}
        fixtures = sorted(self.fixtures, key=len, reverse=True)
        contacts = {name: [] for name in groups}
        robot_contacts = {name: [] for name in groups}
        for i, con in enumerate(d.contact[:d.ncon]):
            a, b = int(con.geom1), int(con.geom2)
            for mine, other, sign in ((a,b,-1),(b,a,1)):
                if mine not in geom_owner:
                    continue
                name = geom_owner[mine]
                if geom_owner.get(other) == name:
                    continue
                other_name = m.geom_id2name(other) or ''
                if other_name.startswith(('robot', 'gripper', 'mobilebase')):
                    robot_contacts[name].append(other_name)
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(m._model, d._data, i, force)
                support = geom_owner.get(other)
                if support is None:
                    support = next((n for n in fixtures if other_name.startswith(n+'_')), other_name)
                contacts[name].append({'support_id': support, 'geom': other_name,
                    'position': np.asarray(con.pos).tolist(), 'normal_force_n': float(force[0]),
                    'normal_on_object': (np.asarray(con.frame).reshape(3,3)[0]*sign).tolist()})
        result = {'step': int(step), 'objects': {},
                  'robot_base_position_world_m': self.robot_proprio()['base_position_world_m']}
        for name, obj in self.objects.items():
            body = self.obj_body_id[name]
            points = self._world_collision_points(name)
            xy = points[:,:2]
            hull = xy[ConvexHull(xy).vertices]
            result['objects'][name] = {
                'position': np.asarray(d.body_xpos[body]).tolist(),
                'quaternion_xyzw': Rotation.from_matrix(d.body_xmat[body].reshape(3,3)).as_quat().tolist(),
                'linear_velocity': np.asarray(d.get_body_xvelp(obj.root_body)).tolist(),
                'angular_velocity': np.asarray(d.get_body_xvelr(obj.root_body)).tolist(),
                'grasped': bool(self._check_grasp(self.robots[0].gripper['right'], obj)),
                'robot_contact': bool(robot_contacts[name]), 'robot_contacts': robot_contacts[name],
                'fixed': bool(self.task_spec['objects'][name].get('fixed', False)),
                'footprint_xy': hull.tolist(), 'collision_bounds_xyz': [points.min(0).tolist(), points.max(0).tolist()],
                'support_contacts': contacts[name],
            }
        return result

    def policy_observe(self, obs):
        if all(c + '_image' in obs for c in CAMERAS):
            rgb = {c: obs[c + '_image'][::-1].copy() for c in CAMERAS}
        else:
            cameras = list(self.camera_names)
            rgb = {}
            for camera in CAMERAS:
                index = cameras.index(camera)
                rgb[camera] = np.asarray(self.sim.render(
                    width=int(self.camera_widths[index]),
                    height=int(self.camera_heights[index]),
                    camera_name=camera,
                )).copy()
        return policy_observation(rgb, self.robot_proprio(), self.instruction)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['lang'] = self.instruction
        return meta

    def _check_success(self):
        return False


def make_scene(scene_class, variant=0, seed=0, image_size=256, gpu=2, horizon=6000):
    return EnvUtils.create_env(scene_class.__name__, variant=variant, seed=seed,
        layout_ids=4, style_ids=4, clutter_mode=0, camera_names=list(CAMERAS),
        camera_widths=image_size, camera_heights=image_size, robot_spawn_deviation_pos_x=0,
        robot_spawn_deviation_pos_y=0, robot_spawn_deviation_rot=0, initialization_noise=None,
        render_gpu_device_id=gpu, horizon=horizon)
