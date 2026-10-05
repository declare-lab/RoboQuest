"""Opaque dome covers with a knob the gripper pinches (spec 4.10).

The small cloche (16 cm across, 10 cm of headroom) hides cubes, parcels and small items; the
large one (22 cm across, 13 cm of headroom) hides vessels and stamps. The shell is hollow: a
ring of thin boxes closed by a thin top disk, so a covered object stands inside untouched and
the cover lifts straight up. The visual is an opaque skirt with an ellipsoid dome, drawn inside
the collision envelope, plus the knob. The masses are those of a steel dish cover, 0.30 kg
(small) and 0.38 kg (large); the rim's friction of 1.0 keeps it put on a level counter.

Knob: a 20 mm shaft under a 25 mm cap, 25 mm tall in all, so two Panda pads straddle the shaft
and the cap stops them slipping off when the cover is lifted. The site ``{name}_knob`` marks the
pinch point, 12.5 mm above the shell top.

With ``free=True`` (a cover that is lifted) ``parent`` must be the worldbody, since MuJoCo allows
free joints on top-level bodies only; ``pos`` and ``yaw`` then give the world pose
(``env.frame_to_world(...)`` and ``env.work['yaw']``). Appearance: ``material`` if given, else a
RoboCasa texture registered on ``assets`` when passed, else a plain colour; ``kind`` picks the
texture family (metal, ceramic or plastic, drawn from ``rng`` when None).
"""
import math
import xml.etree.ElementTree as ET

from roboquest.assets import materials
from roboquest.assets._mjcf import collision, draw, pair, site, vec, visual, yaw_quat

SIZES = {'small': dict(radius=.080, depth=.100, segments=16, mass=.30),
         'large': dict(radius=.110, depth=.130, segments=20, mass=.38)}
WALL = .004                                  # shell thickness: the ring boxes and the top disk
KNOB_SHAFT_RADIUS, KNOB_SHAFT_HEIGHT = .010, .020
KNOB_CAP_RADIUS, KNOB_CAP_HEIGHT = .0125, .005
KNOB_HEIGHT = KNOB_SHAFT_HEIGHT + KNOB_CAP_HEIGHT
KNOB_GRASP = .0125                           # pinch point above the shell top
KNOB_MASS = .02
KINDS = ('metal', 'ceramic', 'plastic')
KNOB_RGBA = ((.16, .16, .17, 1.), (.80, .68, .40, 1.), (.85, .86, .88, 1.))   # black, brass, chrome
RIM_FRICTION = '1 0.005 0.0001'
KNOB_FRICTION = '1 0.01 0.0001'


def cloche(parent, name, size='small', rng=None, free=True, pos=(0., 0., 0.), yaw=0., assets=None,
           material=None, kind=None):
    """Append a dome cover under ``parent``; see the module docstring.

    Returns a dict: ``body`` (the element), ``radius`` (outer), ``height`` (the shell top, where the
    knob starts), ``knob_site``, ``mass``, and ``name``, ``joint``, ``size``, ``inner_radius``,
    ``inner_height``, ``knob_grasp_z``, ``knob_top_z``, ``knob_diameter``, ``segments``, ``geoms``
    (collision names), ``material``, ``rgba``, ``kind``.
    """
    if size not in SIZES:
        raise ValueError(f'cloche size must be one of {tuple(SIZES)}, not {size!r}')
    if not name:
        raise ValueError('cloche needs a name')
    p = SIZES[size]
    radius, depth, segments = p['radius'], p['depth'], p['segments']
    height = depth + WALL
    body = ET.SubElement(parent, 'body', name=name, pos=vec(pos), quat=yaw_quat(yaw))
    joint = None
    if free:
        joint = name + '_joint'
        ET.SubElement(body, 'freejoint', name=joint)
    kind = kind or KINDS[draw(rng, len(KINDS))]
    material, rgba = materials.dress(rng, kind, assets, material)
    knob_rgba = KNOB_RGBA[draw(rng, len(KNOB_RGBA))]

    shell_mass = p['mass'] - KNOB_MASS
    ring_area, top_area = 2 * math.pi * radius * height, math.pi * radius ** 2
    ring_mass = shell_mass * ring_area / (ring_area + top_area)
    top_mass = shell_mass - ring_mass
    geoms = []
    half_chord = radius * math.tan(math.pi / segments) + .0005    # adjacent boxes meet at the outer face
    mid = radius - WALL / 2
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        geoms.append(collision(body, f'{name}_wall_{i}', 'box',
                               (mid * math.cos(angle), mid * math.sin(angle), height / 2),
                               (WALL / 2, half_chord, height / 2), ring_mass / segments, quat=yaw_quat(angle),
                               friction=RIM_FRICTION))
    geoms.append(collision(body, f'{name}_top', 'cylinder', (0., 0., height - WALL / 2),
                           (radius - .0005, WALL / 2), top_mass, friction=RIM_FRICTION))
    geoms.append(pair(body, f'{name}_knob_shaft', 'cylinder', (0., 0., height + KNOB_SHAFT_HEIGHT / 2),
                      (KNOB_SHAFT_RADIUS, KNOB_SHAFT_HEIGHT / 2), .7 * KNOB_MASS, rgba=knob_rgba,
                      friction=KNOB_FRICTION, condim='4'))
    geoms.append(pair(body, f'{name}_knob_cap', 'cylinder',
                      (0., 0., height + KNOB_SHAFT_HEIGHT + KNOB_CAP_HEIGHT / 2),
                      (KNOB_CAP_RADIUS, KNOB_CAP_HEIGHT / 2), .3 * KNOB_MASS, rgba=knob_rgba,
                      friction=KNOB_FRICTION, condim='4'))
    # Opaque visual: a skirt cylinder and an ellipsoid dome whose apex meets the top disk; the
    # ellipsoid's lower half is hidden inside the skirt.
    skirt = .55 * height
    visual(body, f'{name}_skirt_visual', 'cylinder', (0., 0., skirt / 2), (radius - .0003, skirt / 2),
           rgba=rgba, material=material)
    visual(body, f'{name}_dome_visual', 'ellipsoid', (0., 0., skirt),
           (radius - .0003, radius - .0003, height - skirt), rgba=rgba, material=material)
    knob_site = site(body, f'{name}_knob', (0., 0., height + KNOB_GRASP))
    return dict(body=body, name=name, joint=joint, size=size, radius=radius, inner_radius=radius - WALL,
                height=height, inner_height=depth, knob_site=knob_site, knob_grasp_z=height + KNOB_GRASP,
                knob_top_z=height + KNOB_HEIGHT, knob_diameter=2 * KNOB_SHAFT_RADIUS, mass=p['mass'],
                segments=segments, geoms=geoms, material=material, rgba=rgba, kind=kind)
