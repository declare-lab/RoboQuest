"""Shared MJCF primitives for the procedural assets.

Every asset is built from ``xml.etree`` elements the way the suite's other props are
(``cubes/bin.py``, ``containers/panels.py``, ``balance/builder.py``): a collision geom in
group 0 that carries the mass, an opaque visual twin in group 1 that carries the material or
colour (robosuite's offscreen renderer hides group 0), and hollow shells made of thin boxes
rather than convex hulls.
"""
import math
import xml.etree.ElementTree as ET

# The suite's safe contact softness: stiffer pairs launched objects bridging counter seams.
CONTACT = dict(solref='0.006 1', solimp='0.98 0.999 0.001', margin='0')
SITE = dict(size='0.002', rgba='0 0 0 0', group='5')


def vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def yaw_quat(yaw):
    """wxyz quaternion string of a rotation about z."""
    return vec((math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)))


def draw(rng, n):
    """A deterministic index in [0, n): one draw of a numpy Generator or RandomState; 0 without an rng."""
    n = int(n)
    if n <= 0:
        raise ValueError('nothing to draw from')
    if rng is None:
        return 0
    if hasattr(rng, 'integers'):
        return int(rng.integers(n))
    if hasattr(rng, 'randint'):
        return int(rng.randint(n))
    raise TypeError(f'rng must be a numpy Generator or RandomState, not {type(rng).__name__}')


def collision(body, name, geom_type, pos, size, mass, quat=None, friction='1 0.005 0.0001', condim='3',
              priority=None):
    """A group-0 collision geom carrying ``mass``; invisible, the visual twin shows the shape."""
    attributes = dict(name=name, type=geom_type, pos=vec(pos), size=vec(size), group='0', contype='1',
                      conaffinity='1', rgba='0 0 0 0', mass=f'{float(mass):.9g}', friction=friction,
                      condim=condim, **CONTACT)
    if quat is not None:
        attributes['quat'] = quat
    if priority is not None:
        attributes['priority'] = str(int(priority))
    ET.SubElement(body, 'geom', **attributes)
    return name


def visual(body, name, geom_type, pos, size, quat=None, rgba=None, material=None):
    """An opaque group-1 visual geom with no mass and no collision; a material wins over a colour."""
    attributes = dict(name=name, type=geom_type, pos=vec(pos), size=vec(size), group='1', contype='0',
                      conaffinity='0', density='0')
    if quat is not None:
        attributes['quat'] = quat
    if material is not None:
        attributes['material'] = material
    elif rgba is not None:
        attributes['rgba'] = vec(rgba)
    ET.SubElement(body, 'geom', **attributes)
    return name


def pair(body, name, geom_type, pos, size, mass, quat=None, rgba=None, material=None, **contact):
    """Collision geom plus its visual twin of the same shape. Returns the collision name."""
    collision(body, name, geom_type, pos, size, mass, quat=quat, **contact)
    visual(body, name + '_visual', geom_type, pos, size, quat=quat, rgba=rgba, material=material)
    return name


def site(body, name, pos):
    """An invisible site marking a grasp or reference point."""
    ET.SubElement(body, 'site', name=name, pos=vec(pos), **SITE)
    return name
