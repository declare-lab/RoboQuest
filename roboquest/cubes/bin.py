# Copied verbatim from active_bench/cube_inspection_bin.py (cube inspection v4, 2026-09-19) for the
# RoboQuest painted-cubes task; the original module is unchanged. Only this header was added.
"""Visible reject bin and explicit irreversible, native constraint commitment.

A 52 mm cube becomes committed only after its entire collision geometry is below
the rim and inside the bin, it is released, and native upward support plus low
velocity persist for a dwell. The latch activates a predefined weld at the
CURRENT pose: no qpos/qvel assignments, collision disabling, or pose snapping.
This is a task-specific capture abstraction, not a claim of passive bin material.
"""
from __future__ import annotations

from collections.abc import Mapping
import itertools
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def _vec(values):
    return ' '.join(str(float(v)) for v in values)


def append_bin(worldbody, position_xyz_floor, name='cube_reject_bin', interior_height=.50, lid_rgba=None,
               body_rgba=None):
    """Append a tall trash can with an inward-only spring-return drop flap.

    Position is interior floor TOP in world. Interior is 240 x 200 mm by
    ``interior_height`` (500 mm by default, the cube inspection bin; RoboQuest
    v1 passes 300 mm), with a 110 mm square mouth. The flap sweeps 107 mm below
    the roof: at 500 mm the lowest sweep is above 393 mm, leaving room for even
    a six-cube 312 mm stack; at 300 mm it is above 196 mm, room for three
    stacked 52 mm cubes (156 mm). ``lid_rgba`` colours the four roof pieces
    (the class colour of the v1 bins) and ``body_rgba`` the floor and walls; the
    defaults reproduce the original bin exactly. MJCF compiler angles must be
    radians. Geometry enables entry; the separately declared native weld
    supplies the irreversible task commitment after a cube settles inside and
    is released.
    """
    position = np.asarray(position_xyz_floor, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Bin position must be a finite XYZ vector')
    if not name or worldbody.find(f"body[@name='{name}']") is not None:
        raise ValueError('Bin name must be nonempty and unique')
    interior_height = float(interior_height)
    if not math.isfinite(interior_height) or interior_height < .16:
        raise ValueError('Bin interior height must leave room for the 107 mm flap sweep over a cube')
    for label, rgba in (('lid_rgba', lid_rgba), ('body_rgba', body_rgba)):
        if rgba is not None and (len(rgba) != 4 or not np.isfinite(np.asarray(rgba, dtype=float)).all()):
            raise ValueError(f'{label} must be four finite numbers')
    half, height, thickness, mouth_half = (.12, .10), interior_height, .012, .055
    body = ET.SubElement(worldbody, 'body', name=name, pos=_vec(position))
    geoms = []

    def box(parent, suffix, pos, size, color, **collision_attributes):
        geom_name = name+'_'+suffix
        common = dict(type='box', pos=_vec(pos), size=_vec(size), rgba=_vec(color))
        contact = dict(priority='1', condim='4', friction='1 .01 .001', solref='.002 1', solimp='.99 .999 .001')
        contact.update(collision_attributes)
        ET.SubElement(parent, 'geom', name=geom_name, group='0', contype='1', conaffinity='1', **contact, **common)
        ET.SubElement(parent, 'geom', name=geom_name+'_visual', group='1', contype='0', conaffinity='0',
                      density='0', **common)
        geoms.append(geom_name)
        return geom_name

    charcoal = (.105, .12, .14, 1) if body_rgba is None else tuple(float(v) for v in body_rgba)
    lid_color = (.16, .18, .21, 1) if lid_rgba is None else tuple(float(v) for v in lid_rgba)
    trim_color = (.49, .53, .57, 1)
    floor = box(body, 'floor', (0, 0, -thickness/2),
                (half[0]+thickness, half[1]+thickness, thickness/2), charcoal)
    # The walls end at the roof's underside: a wall reaching ``height`` would share the roof's outer
    # side faces over its top 6 mm and z-fight there (a jagged band under the lid). The solid is the same.
    roof_z, roof_half_z = height, .006
    wall_half_z = (height-roof_half_z)/2
    for sign in (-1, 1):
        box(body, f'wall_x_{sign:+d}', (sign*(half[0]+thickness/2), 0, wall_half_z),
            (thickness/2, half[1]+thickness, wall_half_z), charcoal)
        box(body, f'wall_y_{sign:+d}', (0, sign*(half[1]+thickness/2), wall_half_z),
            (half[0], thickness/2, wall_half_z), charcoal)
    # Four disjoint roof boxes leave a real square aperture, not a texture hole.
    outer_x, outer_y = half[0]+thickness, half[1]+thickness
    for sign in (-1, 1):
        box(body, f'roof_x_{sign:+d}', (sign*(outer_x+mouth_half)/2, 0, roof_z),
            ((outer_x-mouth_half)/2, outer_y, roof_half_z), lid_color)
        box(body, f'roof_y_{sign:+d}', (0, sign*(outer_y+mouth_half)/2, roof_z),
            (mouth_half, (outer_y-mouth_half)/2, roof_half_z), lid_color)
        # Raised matching collision/visual trim frames the opening without
        # reducing its 110 mm clear dimensions or extending into the flap sweep.
        box(body, f'trim_x_{sign:+d}', (sign*(mouth_half+.004), 0, height+.009),
            (.004, mouth_half+.008, .003), trim_color)
        box(body, f'trim_y_{sign:+d}', (0, sign*(mouth_half+.004), height+.009),
            (mouth_half, .004, .003), trim_color)
    flap_body_name, flap_joint_name = name+'_flap_body', name+'_flap_hinge'
    hinge = np.array([0., .051, height+.003])
    flap = ET.SubElement(body, 'body', name=flap_body_name, pos=_vec(hinge))
    # Positive rotation sends the panel down into the chamber. Spring preload
    # exceeds empty-panel gravity, while a 52 mm cube's weight opens it.
    ET.SubElement(flap, 'joint', name=flap_joint_name, type='hinge', axis='1 0 0',
                  limited='true', range='0 1.5', stiffness='.012', springref='-.65',
                  damping='.0025', armature='.00001', frictionloss='.0001',
                  solreflimit='.004 1', solimplimit='.99 .999 .001')
    flap_geom = box(flap, 'flap', (0, -.052, 0), (.053, .052, .003), (.31, .35, .40, 1),
                    mass='.012', priority='1', friction='.25 .005 .0001')
    return dict(body_name=name, position_xyz_floor=position.tolist(), floor_geom_name=floor,
                geom_names=geoms, interior_half_size_xy=list(half), interior_height=height,
                interior_bounds_world_min=(position+np.array([-half[0], -half[1], 0.])).tolist(),
                interior_bounds_world_max=(position+np.array([half[0], half[1], height])).tolist(),
                wall_thickness=thickness, containment_tolerance_m=.001, rim_clearance_m=.002,
                settling_dwell_s=.15, max_linear_speed_m_s=.015, max_angular_speed_rad_s=.12,
                min_support_force_n=.0001, support_normal_z_min=.5,
                six_cube_centers_local_xy=[[x, y] for y in (-.045, .045) for x in (-.07, 0., .07)],
                recommended_physics_timestep_s=.001,
                contact_parameters=dict(priority=1, solref=[.002, 1.], solimp=[.99, .999, .001],
                    cube_cube_pair_solref=[.002, 1.], cube_cube_pair_solimp=[.99, .999, .001],
                    cube_cube_pair_condim=3, cube_cube_pair_friction=[1., 1., .005, .0001, .0001]),
                exterior_size_xyz=[2*outer_x, 2*outer_y, height+thickness+.012],
                opening_center_world=(position+np.array([0., 0., height+.012])).tolist(),
                opening_size_xy=[2*mouth_half, 2*mouth_half], opening_roof_z_local=height+.006,
                flap=dict(body_name=flap_body_name, joint_name=flap_joint_name, geom_name=flap_geom,
                    hinge_position_local=hinge.tolist(), axis_local=[1., 0., 0.], joint_range_rad=[0., 1.5],
                    mass_kg=.012, spring_stiffness_nm_rad=.012, spring_reference_rad=-.65,
                    damping_nm_s_rad=.0025, panel_size_xyz=[.106, .104, .006],
                    minimum_sweep_z_local=float(hinge[2]-.104*np.sin(1.5)-.003),
                    mechanism='Native inward-only limited hinge with passive preloaded return spring; normal soft joint-limit tolerance.'),
                capture_semantics='Irreversible current-pose native weld after fully inside, below rim, released and supported/settled for dwell; geometry/flap alone is not claimed irreversible; reset clears capture.')


def append_capture_constraints(root, spec, cube_body_names):
    """Predefine one inactive bin-to-cube weld; save its name in spec.

    cube_body_names can be a sequence of body names or a name -> body-name map.
    The mapping form allows task labels to differ from native body names.
    """
    bodies = dict(cube_body_names) if isinstance(cube_body_names, Mapping) else {n: n for n in cube_body_names}
    if not bodies or len(set(bodies.values())) != len(bodies):
        raise ValueError('Capture requires distinct cube bodies')
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    names = {}
    for key, body_name in bodies.items():
        weld_name = spec['body_name']+'__capture__'+key
        if equality.find(f"*[@name='{weld_name}']") is not None:
            raise ValueError('Capture constraint already exists: '+weld_name)
        ET.SubElement(equality, 'weld', name=weld_name, body1=spec['body_name'], body2=body_name,
                      active='false', relpose='0 0 0 1 0 0 0',
                      solref='.004 1', solimp='.99 .999 .001', torquescale='.1')
        names[key] = weld_name
    spec['capture_welds'] = names
    return names


def bounds_fully_inside(bounds_min_local, bounds_max_local, spec):
    """Box bounds in bin coordinates; 1 mm contact tolerance, 2 mm rim margin."""
    low, high = (np.asarray(v, dtype=float) for v in (bounds_min_local, bounds_max_local))
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low > high):
        return False
    half = np.asarray(spec['interior_half_size_xy'])
    tolerance = spec['containment_tolerance_m']
    return bool(np.all(low[:2] >= -half-tolerance) and np.all(high[:2] <= half+tolerance)
                and low[2] >= -tolerance and high[2] <= spec['interior_height']-spec['rim_clearance_m'])


class BinCapture:
    """Evaluator/physics helper for raw MuJoCo or robosuite model/data wrappers.

    Call update(grasped=<names>) after each native step or control step, using
    current physical grasp detection. robot_geom_ids may explicitly list every
    robot/gripper collision geom; otherwise conventional native name prefixes
    robot/gripper/mobilebase are detected. update returns newly captured names.
    Dwell means consecutive eligible observations spanning the stated simulation
    time; call at native physics frequency if substep continuity is required.
    Capture status never depends on cube labels, hidden inspection results or
    task correctness. reset changes only this helper's constraints/history.
    """
    def __init__(self, model, data, cube_body_ids, cube_geom_ids, spec, robot_geom_ids=None):
        self.model = getattr(model, '_model', model)
        self.data = getattr(data, '_data', data)
        self.spec = spec
        self.body_ids = {name: int(value) for name, value in cube_body_ids.items()}
        self.geom_ids = {name: tuple(int(g) for g in cube_geom_ids[name]) for name in self.body_ids}
        self.bin_id = self._id(mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
        self.floor_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, spec['floor_geom_name'])
        if 'flap' in spec:
            hinge_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, spec['flap']['joint_name'])
            if not np.allclose(self.model.jnt_range[hinge_id], spec['flap']['joint_range_rad']):
                raise ValueError('Trash-bin hinge requires MJCF compiler angles in radians')
        self.weld_ids = {name: self._id(mujoco.mjtObj.mjOBJ_EQUALITY, spec['capture_welds'][name]) for name in self.body_ids}
        for name, bid in self.body_ids.items():
            if not self.geom_ids[name] or any(self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in self.geom_ids[name]):
                raise ValueError('Solid cube capture requires box collision geoms: '+name)
            if any(self.model.geom_bodyid[g] != bid for g in self.geom_ids[name]):
                raise ValueError('Cube collision geoms must belong to its free rigid body: '+name)
            eid = self.weld_ids[name]
            if self.model.eq_type[eid] != mujoco.mjtEq.mjEQ_WELD or self.model.eq_obj1id[eid] != self.bin_id or self.model.eq_obj2id[eid] != bid:
                raise ValueError('Capture weld does not join expected bin/cube: '+name)
        if robot_geom_ids is None:
            robot_geom_ids = [g for g in range(self.model.ngeom) if
                (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith(('robot', 'gripper', 'mobilebase'))]
        self.robot_geom_ids = frozenset(int(g) for g in robot_geom_ids)
        self._original_weld_data = {name: self.model.eq_data[eid].copy() for name, eid in self.weld_ids.items()}
        self._eligible_since = {}
        self._last_time = float(self.data.time)
        self.capture_events = []

    def _id(self, kind, name):
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError('Missing native capture object: '+name)
        return value

    @property
    def captured(self):
        return frozenset(name for name, eid in self.weld_ids.items() if self.data.eq_active[eid])

    def check(self, name, grasped=()):
        """Read current candidate conditions without activating anything."""
        m, d = self.model, self.data
        gids = self.geom_ids[name]
        signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
        corners = np.concatenate([(signs*m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in gids])
        local = (corners-d.xpos[self.bin_id]) @ d.xmat[self.bin_id].reshape(3, 3)
        low, high = local.min(0), local.max(0)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.body_ids[name], velocity, 0)
        captured_geoms = {g for key in self.captured if key != name for g in self.geom_ids[key]}
        support, robot_contact = [], False
        up = d.xmat[self.bin_id].reshape(3, 3)[:, 2]
        for index, contact in enumerate(d.contact[:d.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in gids and b not in gids:
                continue
            other = b if a in gids else a
            if other in self.robot_geom_ids and contact.dist <= .001:
                robot_contact = True
            if other != self.floor_id and other not in captured_geoms:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, force)
            normal = np.asarray(contact.frame).reshape(3, 3)[0]*(1 if b in gids else -1)
            if force[0] > self.spec['min_support_force_n'] and normal @ up > self.spec['support_normal_z_min']:
                support.append(other)
        contained = bounds_fully_inside(low, high, self.spec)
        linear, angular = float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))
        settled = math.isfinite(linear+angular) and linear <= self.spec['max_linear_speed_m_s'] and angular <= self.spec['max_angular_speed_rad_s']
        released = not robot_contact and name not in grasped
        return dict(eligible=bool(contained and settled and released and support), contained=contained,
                    released=released, settled=settled, supported=bool(support), robot_contact=robot_contact,
                    grasped=name in grasped, linear_speed_m_s=linear, angular_speed_rad_s=angular,
                    local_bounds_min=low.tolist(), local_bounds_max=high.tolist(),
                    world_bounds_min=corners.min(0).tolist(), world_bounds_max=corners.max(0).tolist(),
                    support_geom_ids=sorted(set(support)), captured=name in self.captured)

    def update(self, grasped=()):
        """Commit eligible cubes at their existing pose; return newly captured names."""
        now = float(self.data.time)
        if now < self._last_time:
            self.reset()
        self._last_time = now
        grasped = frozenset(grasped)
        captured_before = self.captured
        ready = []
        for name in self.body_ids:
            if name in captured_before:
                continue
            conditions = self.check(name, grasped)
            if not conditions['eligible']:
                self._eligible_since.pop(name, None)
                continue
            since = self._eligible_since.setdefault(name, now)
            if now-since+1e-12 >= self.spec['settling_dwell_s']:
                ready.append((name, conditions))
        for name, conditions in ready:
            bid, eid = self.body_ids[name], self.weld_ids[name]
            rotation = self.data.xmat[self.bin_id].reshape(3, 3)
            relative_position = rotation.T @ (self.data.xpos[bid]-self.data.xpos[self.bin_id])
            relative_rotation = rotation.T @ self.data.xmat[bid].reshape(3, 3)
            relative_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.ravel())
            self.model.eq_data[eid, :3] = 0.
            self.model.eq_data[eid, 3:6] = relative_position
            self.model.eq_data[eid, 6:10] = relative_quaternion
            self.data.eq_active[eid] = True
            self._eligible_since.pop(name, None)
            self.capture_events.append(dict(cube=name, time_s=now, relative_position=relative_position.tolist(),
                relative_quaternion_wxyz=relative_quaternion.tolist(), conditions=conditions))
        return [name for name, _ in ready]

    def reset(self):
        """Clear only capture constraints and history; never move a body."""
        for name, eid in self.weld_ids.items():
            self.data.eq_active[eid] = False
            self.model.eq_data[eid] = self._original_weld_data[name]
        self._eligible_since.clear()
        self.capture_events.clear()
        self._last_time = float(self.data.time)
