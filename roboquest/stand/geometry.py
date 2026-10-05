"""MJCF builder for the wobbly stand, its shim plates and the ball.

Stand body frame: origin at the centre of the top surface, x/y along the top's
edges, z up; the four legs hang below the top plate and the rim stands on the
top's edge. Every part is a rigid body on a free joint (the stand is one body;
each shim and the ball are their own bodies). Collision geoms are plain boxes
and spheres (group 0) with a group-1 visual twin that carries a RoboCasa wood
texture. Nothing in the MJCF encodes which leg is short beyond the legs'
actual lengths.

Each leg ends in a spherical glide. A square foot touches a flat surface at
its corner once the stand tilts, which moves the pivot line about 14 mm toward
the tilt and makes a stand with one short corner leg rest *level* on its three
long legs (measured on the bare model); a sphere always touches directly under
its centre, so the resting pose follows the point-foot statics in ``statics``
and the stand tips onto the short leg as the task needs.

Contact tuning (measured in ``tests/test_roboquest_wobbly_stand.py`` on the
bare model and by ``WobblyStand.physics_gate`` in a kitchen): the wood parts
use a 5 ms soft-contact time constant with high impedance, so a resting foot
penetrates the counter by microns rather than the ~0.4 mm of MuJoCo's
defaults, which would otherwise blur a 5 mm shim delta; ``priority="1"`` makes
those parameters win against RoboCasa's counter geoms. The ball rolls on the
top through an explicit contact pair with a small rolling-friction
coefficient: it starts rolling above ~0.4 deg of tilt and stays put below.
"""
from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

TEXTURE_DIR = Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime'))) \
    / 'src/robocasa/robocasa/models/assets/textures/wood'
STAND_WOODS = ('walnut_wood_grain', 'warm_wood_grain', 'wood_grain_1', 'red_wood_planks', 'gray_wood_grain')
SHIM_WOOD = 'light_wood_planks_2'
BALL_COLOURS = {'orange': (1., .45, .05, 1.), 'blue': (.10, .35, .95, 1.), 'green': (.15, .75, .20, 1.)}

# Stand dimensions (m): 24 x 24 cm top with a 6 mm rim, 12 cm tall, 2 cm square legs flush with the top's edge,
# each ending in a spherical glide whose lowest point defines the leg length.
TOP_SIZE = .24
TOP_THICKNESS = .012
RIM_HEIGHT = .006
RIM_THICKNESS = .006
STAND_HEIGHT = .12
LEG_SIZE = .02
FOOT_RADIUS = .009
LEG_LENGTH = STAND_HEIGHT - TOP_THICKNESS          # nominal leg length below the top plate, glide included
LEG_CENTRE = TOP_SIZE / 2 - LEG_SIZE / 2           # leg centre distance from the stand's axis, per coordinate
LEGS = ('sw', 'se', 'ne', 'nw')                    # compass names in the stand frame (x east, y north)
LEG_SIGNS = {'sw': (-1, -1), 'se': (1, -1), 'ne': (1, 1), 'nw': (-1, 1)}
SIDES = {'s': ('sw', 'se'), 'e': ('se', 'ne'), 'n': ('ne', 'nw'), 'w': ('nw', 'sw')}
DELTA_LEVELS_MM = (5, 10, 15)

# Masses (kg): about 0.5 kg in all; every leg gets the same mass whatever its length, so shortening a leg
# does not move the centre of mass. The centre of mass is then biased by COM_BIAS toward the low side
# (see statics: a stand with one short corner leg has its centre of mass close to the diagonal between the
# two resting configurations, and the bias decides that it tips onto the short leg, not onto the long ones).
LEG_MASS = .05
FOOT_MASS = .008
RIM_MASS = .005
STAND_MASS = .5
TOP_MASS = STAND_MASS - 4 * LEG_MASS - 4 * RIM_MASS
COM_BIAS = .006

BALL_RADIUS = .015
BALL_MASS = .03
SHIM_SIZE = .05
SHIM_THICKNESSES_MM = (5, 10, 15)
SHIM_DENSITY = 700.

# Contact parameters (see the module docstring).
WOOD_SOLREF = '0.005 1'
WOOD_SOLIMP = '0.98 0.999 0.001'
WOOD_FRICTION = '0.8 0.005 0.0001'
BALL_SOLREF = '0.005 1'
BALL_SOLIMP = '0.95 0.99 0.001'
BALL_FRICTION = '0.8 0.005 0.0001'
BALL_TOP_FRICTION = (.8, .8, .005, 1e-4, 1e-4)     # slide x2, spin, roll x2 for the ball-top contact pair
PHYSICS_TIMESTEP = .001
# MuJoCo's default soft friction has no static regime: a ball on a 0.1 deg slope creeps or rolls whatever the
# rolling coefficient. The noslip post-pass with the elliptic cone enforces the friction cone, and the rolling
# threshold becomes the physical one, tan(tilt) = roll / radius = 0.38 deg for this ball (measured on the bare
# model: still at 0.26 deg, 8 mm of creep in 4 s at 0.39 deg, rolls to the rim from 0.52 deg). With the
# pyramidal cone the ball still creeps 47 mm in 4 s at 0.13 deg.
NOSLIP_ITERATIONS = 5
CONTACT_CONE = 'elliptic'


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def leg_foot_xy(leg):
    sx, sy = LEG_SIGNS[leg]
    return (sx * LEG_CENTRE, sy * LEG_CENTRE)


def leg_length(leg, short_legs, delta_mm):
    return LEG_LENGTH - (float(delta_mm) / 1000. if leg in short_legs else 0.)


def foot_centre(leg, short_legs, delta_mm):
    """Centre of the leg's spherical glide in the stand frame; its lowest point is FOOT_RADIUS lower."""
    x, y = leg_foot_xy(leg)
    return (x, y, -TOP_THICKNESS - leg_length(leg, short_legs, delta_mm) + FOOT_RADIUS)


def com_bias_xy(short_legs):
    """Horizontal centre-of-mass offset (stand frame) toward the low side: the short corner, or the short side's middle."""
    if not short_legs:
        return (0., 0.)
    direction = np.mean([LEG_SIGNS[leg] for leg in short_legs], axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return (0., 0.)
    return tuple((COM_BIAS * direction / norm).tolist())


def texture_path(wood):
    path = TEXTURE_DIR / f'{wood}.png'
    if not path.is_file():
        raise ValueError(f'RoboCasa wood texture not found: {path}')
    return path


def add_wood_material(assets, name, wood, texrepeat=(1., 1.), specular=.15, shininess=.1):
    """A RoboCasa wood texture as a material; idempotent per name."""
    if assets.find(f"material[@name='{name}']") is not None:
        return name
    ET.SubElement(assets, 'texture', name=name + '_tex', type='2d', file=str(texture_path(wood)))
    ET.SubElement(assets, 'material', name=name, texture=name + '_tex', texrepeat=_vec(texrepeat),
                  texuniform='true', specular=str(specular), shininess=str(shininess))
    return name


def _geom_pair(body, name, kind, pos, size, mass, material=None, rgba=(1., 1., 1., 1.), friction=WOOD_FRICTION,
               solref=WOOD_SOLREF, solimp=WOOD_SOLIMP, condim='3', priority='1'):
    """Collision geom (group 0) plus a visual twin (group 1) carrying the material. Returns the collision name."""
    ET.SubElement(body, 'geom', name=name, type=kind, pos=_vec(pos), size=_vec(size), group='0', rgba=_vec(rgba),
                  mass=f'{mass:.9g}', friction=friction, condim=condim, solref=solref, solimp=solimp, margin='0',
                  priority=priority)
    visual = dict(name=name + '_visual', type=kind, pos=_vec(pos), size=_vec(size), group='1', contype='0',
                  conaffinity='0', density='0', rgba=_vec(rgba))
    if material is not None:
        visual['material'] = material
    ET.SubElement(body, 'geom', **visual)
    return name


def _inertia_about_origin(kind, pos, size, mass):
    if kind == 'box':
        hx, hy, hz = size
        local = mass / 3. * np.diag([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])
    elif kind == 'sphere':
        local = 2. / 5. * mass * size[0] ** 2 * np.eye(3)
    else:
        raise ValueError(kind)
    d = np.asarray(pos, float)
    return local + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def composite_inertial(parts, com_shift=(0., 0., 0.)):
    """Mass, centre of mass and inertia (about that centre) of ``(kind, pos, size, mass)`` boxes and spheres.

    ``com_shift`` moves the reported centre of mass without changing the tensor: a small deliberate bias
    (millimetres) is physically a slightly uneven wood density and keeps the tensor positive definite.
    """
    mass = float(sum(p[3] for p in parts))
    com = sum(np.asarray(p[1], float) * p[3] for p in parts) / mass
    inertia = sum(_inertia_about_origin(*p) for p in parts) - mass * (float(com @ com) * np.eye(3) - np.outer(com, com))
    com = com + np.asarray(com_shift, float)
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return dict(mass=mass, com=com.tolist(), fullinertia=[float(v) for v in full])


def stand_parts(short_legs, delta_mm):
    """(name, kind, pos, size, mass) of every collision geom of the stand, in the stand frame."""
    parts = [('top', 'box', (0., 0., -TOP_THICKNESS / 2), (TOP_SIZE / 2, TOP_SIZE / 2, TOP_THICKNESS / 2), TOP_MASS)]
    r = TOP_SIZE / 2 - RIM_THICKNESS / 2
    for side, (axis, sign) in {'s': (1, -1), 'n': (1, 1), 'w': (0, -1), 'e': (0, 1)}.items():
        pos = [0., 0., RIM_HEIGHT / 2]
        pos[axis] = sign * r
        half = [TOP_SIZE / 2, TOP_SIZE / 2, RIM_HEIGHT / 2]
        half[axis] = RIM_THICKNESS / 2
        parts.append((f'rim_{side}', 'box', tuple(pos), tuple(half), RIM_MASS))
    for leg in LEGS:
        x, y, zc = foot_centre(leg, short_legs, delta_mm)
        shaft = -TOP_THICKNESS - zc                     # leg box from the top plate down to the glide's centre
        parts.append((f'leg_{leg}', 'box', (x, y, -TOP_THICKNESS - shaft / 2), (LEG_SIZE / 2, LEG_SIZE / 2, shaft / 2),
                      LEG_MASS - FOOT_MASS))
        parts.append((f'foot_{leg}', 'sphere', (x, y, zc), (FOOT_RADIUS,), FOOT_MASS))
    return parts


def build_stand(world, assets, name, pos, quat_wxyz, short_legs, delta_mm, wood, free=True):
    """Append the stand body. Returns names and stand-frame measurements for the evaluator."""
    short_legs = tuple(short_legs)
    for leg in short_legs:
        if leg not in LEGS:
            raise ValueError(f'unknown leg {leg!r}')
    material = add_wood_material(assets, f'{name}_wood', wood, texrepeat=(2., 2.))
    body = ET.SubElement(world, 'body', name=name, pos=_vec(pos), quat=_vec(quat_wxyz))
    if free:
        ET.SubElement(body, 'freejoint', name=f'{name}_joint')
    parts = stand_parts(short_legs, delta_mm)
    bias = com_bias_xy(short_legs)
    inertial = composite_inertial([p[1:] for p in parts], com_shift=(bias[0], bias[1], 0.))
    ET.SubElement(body, 'inertial', pos=_vec(inertial['com']), mass=f"{inertial['mass']:.9g}",
                  fullinertia=_vec(inertial['fullinertia']))
    geoms, top_geoms, leg_geoms, foot_geoms = [], [], {}, {}
    for part, kind, ppos, size, mass in parts:
        gname = _geom_pair(body, f'{name}_{part}', kind, ppos, size, mass,
                           material=None if kind == 'sphere' else material,
                           rgba=(.25, .25, .27, 1.) if kind == 'sphere' else (1., 1., 1., 1.))
        geoms.append(gname)
        if part == 'top' or part.startswith('rim_'):
            top_geoms.append(gname)
        elif part.startswith('leg_'):
            leg_geoms[part[len('leg_'):]] = gname
        else:
            foot_geoms[part[len('foot_'):]] = gname
    feet = {leg: list(foot_centre(leg, short_legs, delta_mm)) for leg in LEGS}
    return dict(body_name=name, joint_name=f'{name}_joint' if free else None, geom_names=geoms, top_geoms=top_geoms,
                leg_geoms=leg_geoms, foot_geoms=foot_geoms, feet_local=feet, foot_radius=FOOT_RADIUS,
                short_legs=list(short_legs), delta_mm=float(delta_mm), com_local=inertial['com'],
                mass=inertial['mass'], top_half=TOP_SIZE / 2, rim_height=RIM_HEIGHT, rim_thickness=RIM_THICKNESS,
                leg_size=LEG_SIZE, ball_travel=TOP_SIZE / 2 - RIM_THICKNESS - BALL_RADIUS, material=material,
                wood=wood)


def build_shim(world, assets, name, pos, quat_wxyz, thickness_mm, wood=SHIM_WOOD, free=True):
    thickness = float(thickness_mm) / 1000.
    material = add_wood_material(assets, f'{name}_wood', wood, texrepeat=(1., 1.))
    body = ET.SubElement(world, 'body', name=name, pos=_vec(pos), quat=_vec(quat_wxyz))
    if free:
        ET.SubElement(body, 'freejoint', name=f'{name}_joint')
    mass = SHIM_DENSITY * SHIM_SIZE * SHIM_SIZE * thickness
    geom = _geom_pair(body, f'{name}_plate', 'box', (0., 0., 0.), (SHIM_SIZE / 2, SHIM_SIZE / 2, thickness / 2), mass,
                      material=material)
    return dict(body_name=name, joint_name=f'{name}_joint' if free else None, geom_name=geom,
                thickness_mm=float(thickness_mm), thickness=thickness, half_size=SHIM_SIZE / 2, mass=mass, wood=wood)


def build_ball(world, name, pos, rgba, free=True):
    body = ET.SubElement(world, 'body', name=name, pos=_vec(pos))
    if free:
        ET.SubElement(body, 'freejoint', name=f'{name}_joint')
    common = dict(type='sphere', size=f'{BALL_RADIUS:.9g}', rgba=_vec(rgba))
    ET.SubElement(body, 'geom', name=f'{name}_sphere', group='0', mass=f'{BALL_MASS:.9g}', friction=BALL_FRICTION,
                  condim='6', solref=BALL_SOLREF, solimp=BALL_SOLIMP, margin='0', priority='1', **common)
    ET.SubElement(body, 'geom', name=f'{name}_sphere_visual', group='1', contype='0', conaffinity='0', density='0',
                  **common)
    return dict(body_name=name, joint_name=f'{name}_joint' if free else None, geom_name=f'{name}_sphere',
                radius=BALL_RADIUS, mass=BALL_MASS, rgba=list(rgba))


def add_ball_contact_pairs(root, ball_geom, surface_geoms):
    """Explicit ball-surface pairs: rolling friction is what lets the ball rest on a nearly level top."""
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    names = []
    for surface in surface_geoms:
        pair = f'{ball_geom}__{surface}'
        if contact.find(f"pair[@name='{pair}']") is not None:
            raise ValueError(f'duplicate contact pair {pair}')
        ET.SubElement(contact, 'pair', name=pair, geom1=ball_geom, geom2=surface, condim='6',
                      friction=_vec(BALL_TOP_FRICTION), solref=BALL_SOLREF, solimp=BALL_SOLIMP, margin='0')
        names.append(pair)
    return names


def bare_model_xml(stand, shims=(), ball=None, ground_half=(.6, .6, .02), textures=True, timestep=PHYSICS_TIMESTEP,
                   noslip_iterations=NOSLIP_ITERATIONS, cone=CONTACT_CONE):
    """A standalone MuJoCo model: a flat ground box whose top is z = 0, the stand, shims and the ball.

    ``stand``: dict(pos, quat_wxyz, short_legs, delta_mm, wood); ``shims``: dicts(pos, quat_wxyz, thickness_mm);
    ``ball``: dict(pos, rgba) or None. Used for contact tuning and CPU tests without RoboCasa.
    """
    root = ET.Element('mujoco', model='wobbly_stand_bare')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep=f'{timestep:.9g}', gravity='0 0 -9.81',
                  noslip_iterations=str(int(noslip_iterations)), cone=cone)
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1', castshadow='false')
    ET.SubElement(world, 'geom', name='ground', type='box', size=_vec(ground_half), pos=_vec((0, 0, -ground_half[2])),
                  rgba='.7 .7 .7 1', friction='1 0.005 0.0001')
    meta = dict(stand=build_stand(world, assets, 'stand', stand['pos'], stand['quat_wxyz'], stand['short_legs'],
                                  stand['delta_mm'], stand.get('wood', STAND_WOODS[0])), shims=[], ball=None)
    for index, shim in enumerate(shims):
        meta['shims'].append(build_shim(world, assets, f'shim_{index}', shim['pos'], shim['quat_wxyz'],
                                        shim['thickness_mm'], shim.get('wood', SHIM_WOOD)))
    if ball is not None:
        meta['ball'] = build_ball(world, 'ball', ball['pos'], ball.get('rgba', BALL_COLOURS['orange']))
        add_ball_contact_pairs(root, meta['ball']['geom_name'], meta['stand']['top_geoms'])
    if not textures:
        for texture in list(assets.findall('texture')):
            assets.remove(texture)
        for material in assets.findall('material'):
            material.attrib.pop('texture', None)
            material.set('rgba', '.7 .5 .3 1')
    return ET.tostring(root, encoding='unicode'), meta
