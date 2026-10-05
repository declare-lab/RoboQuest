"""Primitive builders and the touch-latch rule shared by every panel box.

Copied out of the open-containers worktree so this suite owns its geometry:
``open_containers_geometry`` (``_vec``/``_box``/``_span``/``add_robocasa_object``
and the RoboCasa asset paths) and ``mech_boxes_geometry`` (the capped-knob
cylinder, the material colours, the latch constants and the declared touch-latch
release rule). Nothing in the source worktree is modified.

Box frame: origin at the surface top under the box centre, x right, y into the
box (away from the robot at yaw 0), z up.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

RUNTIME_ROOT = Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime')))
ROBOCASA_ASSETS = RUNTIME_ROOT / 'src/robocasa/robocasa/models/assets'
BOWL_ASSET = ROBOCASA_ASSETS / 'objects/objaverse/bowl/bowl_1/model.xml'
ITEM_ASSETS = {
    # Small native objaverse objects that fit the 20 x 20 x 18 cm box interior, the pulled-out tray
    # and the parallel gripper (80 mm open); the half extents are the assets' reg_bbox. One asset per
    # category, so a category name in the goal is one shape. Verified by the item tests, which also
    # hold every item to the pinch rule below (the garlic, 7 cm tall, failed it and was dropped:
    # the pads closed above its bulb on separate cloves and it crept out of every squeeze).
    'lemon': ('objects/objaverse/lemon/lemon_0/model.xml',
        (0.0290092, 0.0373494, 0.0322674)),
    'lime': ('objects/objaverse/lime/lime_1/model.xml',
        (0.0325756, 0.0375, 0.0351646)),
    'orange': ('objects/objaverse/orange/orange_2/model.xml',
        (0.0337337, 0.0343972, 0.0329717)),
    'peach': ('objects/objaverse/peach/peach_0/model.xml',
        (0.030831, 0.032729, 0.0345035)),
    'kiwi': ('objects/objaverse/kiwi/kiwi_0/model.xml',
        (0.0320111, 0.0403017, 0.0284731)),
    'tomato': ('objects/objaverse/tomato/tomato_0/model.xml',
        (0.0319362, 0.030445, 0.027975)),
    'apple': ('objects/objaverse/apple/apple_1/model.xml',
        (0.0353495, 0.0337567, 0.0344545)),
    'egg': ('objects/objaverse/egg/egg_1/model.xml',
        (0.0281117, 0.0374667, 0.0279041)),
}
ITEM_HALF = np.array(ITEM_ASSETS['lemon'][1])
# The top-down pick: the descent ends when the gripper's housing lands on the item's top, and the pad
# centres are then PINCH_BELOW_TOP under that top (housing underside 3.0 cm above the eef site, pads
# 2.6 cm below it; oracle traces, 2026-09-22). An item is pinchable only if its collision shape at that
# height is still nearly as wide as its girth, PINCH_WIDTH_SHARE of its full width along both plan axes:
# the pool's items keep 0.91-1.00 of it, the garlic kept 0.27-0.32 (its cloves taper at 60 degrees
# there) and slid out under any squeeze. The item test measures this on the collision meshes.
PINCH_BELOW_TOP = .026
PINCH_WIDTH_SHARE = .85

WOOD = (.60, .44, .28, 1.)
PANEL = (.66, .50, .32, 1.)
METAL = (.72, .73, .75, 1.)
DARK = (.20, .20, .22, 1.)

LATCH_RELEASE_N = 6.             # declared: an inward press of this force releases the touch latch
GUIDE_FRICTION = '0.15 0.005 0.0001'      # slide-lid runners and slot: waxed, so a high push does not jam them
LATCH_POP_M = .04                # spring rest position of the released panel
POP_SPRING = 90.                 # N/m, press-drawer pop spring (dead band beyond LATCH_POP_M)
DOOR_POP_SPRING, DOOR_POP_RAD = 3., .25   # N.m/rad dead-band spring on the press door


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _box(body, name, pos, half, rgba, mass=None, friction='0.7 0.005 0.0001', condim='3',
         solref='0.006 1', solimp='0.98 0.999 0.001'):
    attributes = dict(name=name, type='box', pos=_vec(pos), size=_vec(half), group='0', rgba='0 0 0 0',
                      friction=friction, condim=condim, solref=solref, solimp=solimp, margin='0')
    if mass is not None:
        attributes['mass'] = str(mass)
    ET.SubElement(body, 'geom', **attributes)
    ET.SubElement(body, 'geom', name=name + '_visual', type='box', pos=_vec(pos), size=_vec(half),
                  rgba=_vec(rgba), group='1', contype='0', conaffinity='0', density='0')
    return name


def _span(body, name, x, y, z, rgba, mass=None, **kw):
    pos = [(a + b) / 2 for a, b in (x, y, z)]
    half = [(b - a) / 2 for a, b in (x, y, z)]
    return _box(body, name, pos, half, rgba, mass, **kw)



def add_robocasa_object(world, assets, name, source_asset, pos, yaw=0., free=True, density='500',
                        friction='0.8 0.005 0.0001', condim='4', opaque=False, scale=1.):
    """Copy one RoboCasa object (meshes, textures, all original convex collision
    pieces) into the arena under a prefixed name. Returns (body, collision names).
    `scale` multiplies every mesh uniformly (visual and collision alike).
    """
    source_asset = Path(source_asset).resolve(strict=True)
    source = ET.parse(source_asset).getroot()
    mapping = {e.get('name'): name + '_' + e.get('name') for e in source.find('asset') if e.get('name')}
    for original in source.find('asset'):
        element = deepcopy(original)
        for key in ('name', 'texture', 'material', 'mesh'):
            if element.get(key) in mapping:
                element.set(key, mapping[element.get(key)])
        if element.get('file'):
            element.set('file', str((source_asset.parent / element.get('file')).resolve(strict=True)))
        if opaque and element.tag == 'material' and element.get('rgba'):
            rgba = element.get('rgba').split()
            rgba[3] = '1'
            element.set('rgba', ' '.join(rgba))
        if element.tag == 'mesh' and float(scale) != 1.:
            existing = [float(v) for v in element.get('scale', '1 1 1').split()]
            element.set('scale', _vec([v * float(scale) for v in existing]))
        assets.append(element)
    body = ET.SubElement(world, 'body', name=name, pos=_vec(pos),
                         quat=_vec([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    if free:
        ET.SubElement(body, 'freejoint', name=name + '_joint')
    collisions, visuals = [], []
    for index, original in enumerate(source.find(".//body[@name='object']").findall('geom')):
        if original.get('class') == 'region':
            continue
        element = deepcopy(original)
        # lightwheel assets tag geoms by class; objaverse assets by group/contype.
        if element.get('class') in ('collision', 'visual'):
            collision = element.get('class') == 'collision'
        else:
            collision = element.get('group', '0') == '0' and element.get('contype', '1') != '0'
        element.attrib.pop('class', None)
        element.set('name', name + ('_collision_' if collision else '_visual_') + str(index))
        for key in ('mesh', 'material'):
            if element.get(key):
                element.set(key, mapping[element.get(key)])
        if collision:
            element.attrib.update(group='0', rgba='0 0 0 0', density=density, contype='1', conaffinity='1',
                                  friction=friction, condim=condim, solref='0.006 1', solimp='0.95 0.99 0.001')
            collisions.append(element.get('name'))
        else:
            element.attrib.update(group='1', contype='0', conaffinity='0', density='0')
            visuals.append(element.get('name'))
        body.append(element)
    return body, collisions



def _cyl(body, name, pos, radius, half_length, rgba, axis='z', mass=None, friction='1.0 0.005 0.0001'):
    quat = {'z': '1 0 0 0', 'x': '0.7071068 0 0.7071068 0', 'y': '0.7071068 0.7071068 0 0'}[axis]
    attributes = dict(name=name, type='cylinder', pos=_vec(pos), quat=quat, size=_vec((radius, half_length)),
                      group='0', rgba='0 0 0 0', friction=friction, condim='4', solref='0.006 1',
                      solimp='0.98 0.999 0.001')
    if mass is not None:
        attributes['mass'] = str(mass)
    ET.SubElement(body, 'geom', **attributes)
    ET.SubElement(body, 'geom', name=name + '_visual', type='cylinder', pos=_vec(pos), quat=quat,
                  size=_vec((radius, half_length)), rgba=_vec(rgba), group='1', contype='0', conaffinity='0',
                  density='0')
    return name



def add_latch_constraints(root, spec):
    """Press-drawer parts that live outside the worldbody: the joint equality
    that holds the tray shut until released, and the pop spring (a fixed tendon
    on the slide joint with a dead band above LATCH_POP_M)."""
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    tendon = root.find('tendon')
    if tendon is None:
        tendon = ET.SubElement(root, 'tendon')
    names = []
    for info in spec['boxes'].values():
        if info.get('latch_equality'):
            ET.SubElement(equality, 'joint', name=info['latch_equality'], joint1=info['joint_name'], polycoef='0 0 0 0 0',
                          solref='0.004 1')   # stiff: the preloaded latch deflects well under 0.1 mm
            names.append(info['latch_equality'])
        if info.get('pop_tendon'):
            pop = info.get('pop', dict(stiffness=POP_SPRING, dead_band_from=LATCH_POP_M))
            fixed = ET.SubElement(tendon, 'fixed', name=info['pop_tendon'], stiffness=str(pop['stiffness']),
                                  springlength=_vec((pop['dead_band_from'], 10.)))
            ET.SubElement(fixed, 'joint', joint=info['joint_name'], coef='1')
    return names


def latch_released(model, data, eq_id, rule):
    """Declared release rule for a touch latch: the press the latch resists,
    beyond its spring preload, exceeds the threshold (newtons for the drawer,
    newton-metres about the hinge for the flap). At rest the equality reads
    -preload (it holds the part against its spring); a press moves the reading
    by `sign` times the press."""
    return rule['sign'] * (latch_press_force(model, data, eq_id) + rule['preload']) >= rule['threshold']


def latch_press_force(model, data, eq_id):
    """Constraint force the latch exerts against an inward press, in newtons
    (positive when the panel is being pushed in). Declared release rule:
    release once it exceeds LATCH_RELEASE_N."""
    import mujoco
    total = 0.
    for row in range(data.nefc):
        if data.efc_type[row] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and data.efc_id[row] == eq_id:
            total += float(data.efc_force[row])
    return total
