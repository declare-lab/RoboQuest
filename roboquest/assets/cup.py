"""A wide cup that hides a 52 mm cube and has to be poured out (spec 4.10): 84 mm inside,
110 mm tall, with a handle.

Walls: a ring of 16 thin boxes standing on a 5 mm floor disk, the ring lifted 1 mm so only the
disk touches the counter. The inside is glazed: the walls carry friction 0.35 with contact
priority 1, so the cup's own coefficient governs whatever is inside it, and a cube slides out
once the cup is tilted past about 110 degrees (the wall's incline then exceeds the friction
angle) while it stays put with the mouth horizontal. The 120 degrees the spec asks for pours it
in well under a second.

Handle: a D loop on +x, 24 mm of finger room between the wall and the 12 x 14 x 70 mm grip bar.
The site ``{name}_handle`` marks the pinch on the bar (30 mm out from the wall, mid-height) and
``{name}_mouth`` the rim's centre. The tilt that pours is a rotation about the cup-to-handle
axis (+x in the body frame), the wrist roll of a gripper holding the bar from the side.

With ``free=True`` ``parent`` must be the worldbody (MuJoCo allows free joints on top-level
bodies only); ``pos`` and ``yaw`` give the world pose. Mass 0.20 kg. Appearance as for the cloche
(``material``, ``assets`` or a plain colour; ``kind`` among ceramic, plastic, metal).
"""
import math
import xml.etree.ElementTree as ET

from roboquest.assets import materials
from roboquest.assets._mjcf import draw, pair, site, vec, yaw_quat

INNER_RADIUS, WALL, HEIGHT, FLOOR, SEGMENTS = .042, .004, .110, .005, 16
OUTER_RADIUS = INNER_RADIUS + WALL
RING_LIFT = .001                    # the ring starts just above the base: only the floor disk touches the counter
GRIP_X = OUTER_RADIUS + .030        # grip bar centre, 24 mm of finger room between the wall and the bar
GRIP_HALF = (.006, .007, .035)
GRIP_Z = .0575
BAR_X = OUTER_RADIUS + .017         # the two horizontal bars reach from inside the wall to the grip bar
BAR_HALF = (.019, .006, .005)
BAR_Z = (.0275, .0875)
MASSES = dict(ring=.11, floor=.04, bar=.015, grip=.02)   # 0.20 kg in all
MASS = MASSES['ring'] + MASSES['floor'] + 2 * MASSES['bar'] + MASSES['grip']
GLAZE = '0.35 0.005 0.0001'
FLOOR_FRICTION = '1 0.005 0.0001'
HANDLE_FRICTION = '1 0.01 0.0001'
KINDS = ('ceramic', 'plastic', 'metal')


def wide_cup(parent, name, rng=None, free=True, pos=(0., 0., 0.), yaw=0., assets=None, material=None, kind=None):
    """Append the cup under ``parent``; see the module docstring.

    Returns a dict: ``body`` (the element), ``inner_radius``, ``height``, ``handle_site``, ``mass``,
    and ``name``, ``joint``, ``outer_radius``, ``inner_height``, ``floor_z`` (the inside floor),
    ``handle_grasp`` (body-frame pinch point), ``handle_reach`` (outermost x of the handle),
    ``handle_axis``, ``mouth_site``, ``segments``, ``geoms``, ``material``, ``rgba``, ``kind``.
    """
    if not name:
        raise ValueError('wide_cup needs a name')
    body = ET.SubElement(parent, 'body', name=name, pos=vec(pos), quat=yaw_quat(yaw))
    joint = None
    if free:
        joint = name + '_joint'
        ET.SubElement(body, 'freejoint', name=joint)
    kind = kind or KINDS[draw(rng, len(KINDS))]
    material, rgba = materials.dress(rng, kind, assets, material)
    look = dict(rgba=rgba, material=material)
    geoms = [pair(body, f'{name}_floor', 'cylinder', (0., 0., FLOOR / 2), (INNER_RADIUS + .002, FLOOR / 2),
                  MASSES['floor'], friction=FLOOR_FRICTION, **look)]
    half_chord = OUTER_RADIUS * math.tan(math.pi / SEGMENTS) + .0005
    mid = INNER_RADIUS + WALL / 2
    wall_half_z = (HEIGHT - RING_LIFT) / 2
    for i in range(SEGMENTS):
        angle = 2 * math.pi * i / SEGMENTS
        geoms.append(pair(body, f'{name}_wall_{i}', 'box',
                          (mid * math.cos(angle), mid * math.sin(angle), RING_LIFT + wall_half_z),
                          (WALL / 2, half_chord, wall_half_z), MASSES['ring'] / SEGMENTS, quat=yaw_quat(angle),
                          friction=GLAZE, priority=1, **look))
    for label, z in (('lower', BAR_Z[0]), ('upper', BAR_Z[1])):
        geoms.append(pair(body, f'{name}_handle_{label}', 'box', (BAR_X, 0., z), BAR_HALF, MASSES['bar'],
                          friction=HANDLE_FRICTION, **look))
    geoms.append(pair(body, f'{name}_handle_grip', 'box', (GRIP_X, 0., GRIP_Z), GRIP_HALF, MASSES['grip'],
                      friction=HANDLE_FRICTION, condim='4', **look))
    handle_site = site(body, f'{name}_handle', (GRIP_X, 0., GRIP_Z))
    mouth_site = site(body, f'{name}_mouth', (0., 0., HEIGHT))
    return dict(body=body, name=name, joint=joint, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                height=HEIGHT, inner_height=HEIGHT - FLOOR, floor_z=FLOOR, handle_site=handle_site,
                handle_grasp=(GRIP_X, 0., GRIP_Z), handle_reach=GRIP_X + GRIP_HALF[0], handle_axis=(1., 0., 0.),
                mouth_site=mouth_site, mass=MASS, segments=SEGMENTS, geoms=geoms, material=material, rgba=rgba,
                kind=kind)
