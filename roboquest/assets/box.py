"""The odd-parcel answer box (spec 4.2): an open-top column one parcel wide, fixed to its parent.

Four walls stand on the parent's z = 0 plane, which is the box's floor (``floor_z`` is 0: a parcel
inside rests on the counter the box stands on). The interior is the parcel footprint plus 1.5 cm
of clearance on every side; the walls are 8 mm thick and reach 1 cm above one parcel, so parcels
stack in insertion order and the bottom one needs a precise grasp between the walls. The body
has no joint, so it can hang under a frame body from ``add_frame_body`` or sit in the worldbody
at ``pos``/``yaw``. Appearance as for the other assets (``material``, ``assets`` or a plain colour;
``kind`` among paper, wood, plastic, drawn from ``rng`` when None).
"""
import xml.etree.ElementTree as ET

from roboquest.assets import materials
from roboquest.assets._mjcf import draw, pair, vec, yaw_quat

CLEARANCE = .015
WALL = .008
EXTRA_HEIGHT = .010
WALL_MASS = .05
WALL_FRICTION = '0.7 0.005 0.0001'
KINDS = ('paper', 'wood', 'plastic')


def answer_box(parent, name, parcel_xy, parcel_height, rng=None, pos=(0., 0., 0.), yaw=0., assets=None,
               material=None, kind=None):
    """Append the answer box under ``parent``; see the module docstring.

    Returns a dict: ``body`` (the element), ``inner_xy``, ``wall_height``, ``floor_z``, and ``name``,
    ``outer_xy``, ``wall_thickness``, ``clearance``, ``parcel_xy``, ``parcel_height``, ``geoms``,
    ``material``, ``rgba``, ``kind``.
    """
    if not name:
        raise ValueError('answer_box needs a name')
    lx, ly = (float(v) for v in parcel_xy)
    height = float(parcel_height)
    if not (lx > 0 and ly > 0 and height > 0):
        raise ValueError('parcel footprint and height must be positive')
    inner = (lx + 2 * CLEARANCE, ly + 2 * CLEARANCE)
    wall_height = height + EXTRA_HEIGHT
    body = ET.SubElement(parent, 'body', name=name, pos=vec(pos), quat=yaw_quat(yaw))
    kind = kind or KINDS[draw(rng, len(KINDS))]
    material, rgba = materials.dress(rng, kind, assets, material)
    look = dict(rgba=rgba, material=material, friction=WALL_FRICTION)
    geoms = []
    for sign, label in ((-1., 'xn'), (1., 'xp')):
        geoms.append(pair(body, f'{name}_wall_{label}', 'box', (sign * (inner[0] / 2 + WALL / 2), 0., wall_height / 2),
                          (WALL / 2, inner[1] / 2 + WALL, wall_height / 2), WALL_MASS, **look))
    for sign, label in ((-1., 'yn'), (1., 'yp')):
        geoms.append(pair(body, f'{name}_wall_{label}', 'box', (0., sign * (inner[1] / 2 + WALL / 2), wall_height / 2),
                          (inner[0] / 2, WALL / 2, wall_height / 2), WALL_MASS, **look))
    return dict(body=body, name=name, inner_xy=inner, wall_height=wall_height, floor_z=0.,
                outer_xy=(inner[0] + 2 * WALL, inner[1] + 2 * WALL), wall_thickness=WALL, clearance=CLEARANCE,
                parcel_xy=(lx, ly), parcel_height=height, geoms=geoms, material=material, rgba=rgba, kind=kind)
