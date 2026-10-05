"""Furnished RoboCasa breakfast-island scene; evaluator truth stays private.

Native meshes/collisions; object state is assigned only during reset. No task
score or oracle controller is implemented here. Geometry metadata is NOT policy
observation. A real kitchen layout is retained, with an added native counter.
"""
from copy import deepcopy
import os
from pathlib import Path
import itertools
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures.counter import Counter
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils import env_utils as EnvUtils
from roboquest.base.robocasa_task import CAMERAS

ASSET_ROOT = Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime')) + '/src/robocasa/robocasa/models/assets/objects/objaverse')
VARIANTS = ('cup_deficit', 'plate_deficit', 'visible_deficit')
from roboquest.harness.contract import CONTRACT_VERSION, PUBLIC_GOAL, policy_observation
INSTRUCTION = PUBLIC_GOAL


class TableSettingScene(Kitchen):
    def __init__(self, variant='cup_deficit', **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.instruction = INSTRUCTION
        self.asset_evidence = {}
        self.task_spec = {
            'contract_version': CONTRACT_VERSION,
            'instruction': INSTRUCTION,
            'objects': {n: {'kind': k, 'valid_supports': ['breakfast']} for n,k in
                        [('plate_a','plate'),('plate_b','plate'),('cup_a','cup'),('cup_b','cup'),('carton','clutter')]},
            'supports': {'breakfast': {'bounds_xy': [1.55,3.95,-2.70,-2.00], 'height_z': .92}},
            'places': {'place_left': {'support_id':'breakfast','bounds_xy':[1.69,2.35,-2.53,-2.04]},
                       'place_right': {'support_id':'breakfast','bounds_xy':[2.49,3.15,-2.53,-2.04]}},
            'stow': {'support_id':'breakfast','bounds_xy':[3.29,3.88,-2.57,-2.04]},
            'storage_zones': {'right_end': {'support_id':'breakfast','bounds_xy':[3.29,3.88,-2.57,-2.04]}},
        }
        super().__init__(**kwargs)

    def _setup_model(self):
        super()._setup_model()
        native = Counter(name='breakfast_counter', size=(2.4,.70,.92),
                         pos=(2.75,-2.35,.46), overhang=.025,
                         top_texture=str(Path(__file__).resolve().parents[2]/'assets/textures/source_marble.png'),
                         base_texture=str(Path(__file__).resolve().parents[2]/'assets/textures/source_marble.png'),
                         base_color=[.25,.28,.31,1], hollow=[False,False], rng=self.rng)
        native.set_euler([0,0,np.pi])  # approach its north face from the kitchen aisle
        self.fixtures['breakfast_counter'] = native
        self.fixture_cfgs.append({'name':'breakfast_counter','type':'fixture','model':native})
        self.model.merge_objects([native])
        # Thin visual-only corner marks identify broad places; they are neither
        # support nor hidden clues, and introduce no collision geometry.
        import xml.etree.ElementTree as ET
        for name, region in {**self.task_spec['places'], **self.task_spec['storage_zones']}.items():
            x0,x1,y0,y1 = region['bounds_xy']
            borders = [((x0+x1)/2,y0,(x1-x0)/2,.003),
                       ((x0+x1)/2,y1,(x1-x0)/2,.003),
                       (x0,(y0+y1)/2,.003,(y1-y0)/2),
                       (x1,(y0+y1)/2,.003,(y1-y0)/2)]
            for i,(x,y,sx,sy) in enumerate(borders):
                ET.SubElement(self.model.worldbody,'geom',name=f'{name}_mark_{i}',
                              type='box',size=f'{sx} {sy} .0005',pos=f'{x} {y} .921',
                              rgba='.16 .19 .23 1',contype='0',conaffinity='0',group='1')


    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.init_robot_base_ref = self.fixtures['breakfast_counter']
        # Kitchen computes native anchor placement from this rotated fixture.

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        for name,rel in [('plate_a','plate/plate_0'),('plate_b','plate/plate_0'),
                         ('cup_a','mug/mug_2'),('cup_b','mug/mug_2'),
                         ('carton','boxed_food/boxed_food_2')]:
            path = ASSET_ROOT/rel/'model.xml'
            obj = MJCFObject(name=name,mjcf_path=str(path),scale=1.0)
            self.objects[name] = obj
            self.model.merge_objects([obj])
            self.asset_evidence[name] = {'mjcf_path':str(path),'uniform_scale':1.,
                                        'size':list(obj.size),'bottom_offset':list(obj.bottom_offset)}

    def _reset_internal(self):
        # Reset-time robot placement only, facing the island from the clear aisle.
        self.init_robot_base_pos_anchor = np.array([2.4,-1.45,0.])
        super()._reset_internal()
        positions = {'plate_a':[1.92,-2.30], 'cup_a':[2.18,-2.32],
                     'plate_b':[2.72,-2.30], 'cup_b':[2.98,-2.30],
                     'carton':[2.18,-2.12]}
        if self.variant in ('cup_deficit','visible_deficit'):
            positions['cup_a'] = [3.62,-2.27]
        else:
            positions['plate_a'] = [3.58,-2.30]
        if self.variant == 'visible_deficit':
            positions['carton'] = [3.38,-2.16]
        for name,xy in positions.items():
            obj = self.objects[name]
            quat = Rotation.from_euler('z',np.pi/2 if name=='carton' else 0).as_quat()
            quat_wxyz = np.r_[quat[3],quat[:3]]
            z = .923-float(obj.bottom_offset[2])
            self.sim.data.set_joint_qpos(obj.joints[0],np.r_[xy,z,quat_wxyz])
            self.sim.data.set_joint_qvel(obj.joints[0],np.zeros(6))
        self.sim.forward()

    def _collision_geom_ids(self,name):
        model=self.sim.model
        return [i for i in range(model.ngeom) if
                (model.geom_id2name(i) or '').startswith(name+'_') and
                (model.geom_contype[i] or model.geom_conaffinity[i])]

    def _world_collision_points(self,name):
        m,d=self.sim.model,self.sim.data
        points=[]
        for g in self._collision_geom_ids(name):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
                local=np.array(m.mesh_vert[start:start+count])
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                local=np.array(list(itertools.product((-1,1),repeat=3)))*m.geom_size[g]
            else:
                raise ValueError(f'Unsupported collision footprint geom type for {name}: {m.geom_type[g]}')
            points.extend(local @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g])
        return np.asarray(points)

    def evaluator_snapshot(self,step=0):
        """Privileged physical snapshot; never include this in policy observations."""
        m,d=self.sim.model,self.sim.data
        result={'step':int(step),'objects':{}}
        for name in self.task_spec['objects']:
            obj=self.objects[name];body=self.obj_body_id[name]
            points=self._world_collision_points(name)
            xy=points[:,:2];xy=xy[ConvexHull(xy).vertices]
            gids=set(self._collision_geom_ids(name));contacts=[];robot_contacts=[]
            for i,con in enumerate(d.contact[:d.ncon]):
                a,b=int(con.geom1),int(con.geom2)
                if not ({a,b}&gids):continue
                other=b if a in gids else a
                other_name=m.geom_id2name(other) or ''
                if other_name.startswith(('robot','gripper','mobilebase')):
                    robot_contacts.append({'geom':other_name,'distance_m':float(con.dist)})
                if not other_name.startswith('breakfast_counter_'):continue
                force=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,i,force)
                normal=np.array(con.frame).reshape(3,3)[0]*(1 if b in gids else -1)
                contacts.append({'support_id':'breakfast','position':np.array(con.pos).tolist(),
                                 'normal_force_n':float(force[0]),'normal_on_object':normal.tolist()})
            result['objects'][name]={'position':d.body_xpos[body].tolist(),
                'quaternion_xyzw':Rotation.from_matrix(d.body_xmat[body].reshape(3,3)).as_quat().tolist(),
                'linear_velocity':np.array(d.get_body_xvelp(obj.root_body)).tolist(),
                'angular_velocity':np.array(d.get_body_xvelr(obj.root_body)).tolist(),
                'grasped':bool(self._check_grasp(self.robots[0].gripper['right'],obj)),
                'footprint_xy':xy.tolist(),'support_contacts':contacts,
                'robot_contact':bool(robot_contacts),'robot_contacts':robot_contacts,
                'collision_bounds_xyz':[points.min(0).tolist(),points.max(0).tolist()]}
        return result

    def robot_proprio(self):
        """Only robot joints/odometry; no fixture or object qpos/qvel slices."""
        robot=self.robots[0];m,d=self.sim.model,self.sim.data
        def addresses(getter):
            out=[]
            for joint in list(robot.robot_model.all_joints) + list(robot.gripper['right'].joints):
                address=getter(joint)
                out.extend(range(*address) if isinstance(address,tuple) else [address])
            return out
        qpos=addresses(m.get_joint_qpos_addr);qvel=addresses(m.get_joint_qvel_addr)
        sid=robot.eef_site_id['right'];base_pos,base_rot=robot.part_controllers['base'].get_base_pose()
        return {'eef_position_world_m':d.site_xpos[sid].tolist(),
                'eef_quaternion_xyzw':Rotation.from_matrix(d.site_xmat[sid].reshape(3,3)).as_quat().tolist(),
                'base_position_world_m':np.array(base_pos).tolist(),
                'base_quaternion_xyzw':Rotation.from_matrix(base_rot).as_quat().tolist(),
                'joint_positions':d.qpos[qpos].tolist(),'joint_velocities':d.qvel[qvel].tolist()}

    def policy_observe(self,obs):
        return policy_observation({c:obs[c+'_image'][::-1].copy() for c in CAMERAS},self.robot_proprio())

    def get_ep_meta(self):
        # Existing env metadata is evaluator-owned, not the policy allowlist.
        meta=super().get_ep_meta();meta['lang']=self.instruction
        return meta

    def _check_success(self):
        return False  # evaluator owner implements final scorer, never a scene latch


def make_table_setting_scene(variant='cup_deficit',seed=0,image_size=256,gpu=2,horizon=6000):
    return EnvUtils.create_env('TableSettingScene',variant=variant,seed=seed,layout_ids=1,style_ids=1,
        camera_names=list(CAMERAS),camera_widths=image_size,camera_heights=image_size,
        robot_spawn_deviation_pos_x=0,robot_spawn_deviation_pos_y=0,initialization_noise=None,
        render_gpu_device_id=gpu,horizon=horizon)
