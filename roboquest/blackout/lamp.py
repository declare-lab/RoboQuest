"""The lantern of Blackout Search (v1.1 contract section 3, 2026-09-24).

A battery lantern the robot carries through a dark kitchen, a real hurricane
lantern: the CC0 Poly Haven model ``Lantern_01`` (``assets/meshes/lantern_01/``, provenance and hashes in
``PROVENANCE.md`` there), drawn at :data:`MESH_SCALE` so it stands :data:`BAIL_Z` high, its glass globe
glowing warm white in the dark. The mesh only draws; the physics is the primitive stack below (a heavy
foot, the body, the cap and the bail apex thickened into a bar), 0.35 kg with about 60 % of it in the
foot so the lantern stands up again after a nudge instead of toppling.

Grasp. The gripper pinches the bail's flat apex (:data:`BAIL_Z` above the foot) top-down, the same
``grasp`` record the object pool's mug uses (``handle=True``: the hand yaw equals the object yaw or
its opposite, and the fingers close along the hand's x axis, so the bar runs along the lantern's local
**y**; the mesh is turned a quarter turn inside the body so its bail lies that way). The lantern then
hangs under the hand with its weight below the pinch, which is the stable way to carry it.

Light. Two ``<light>`` children of the body, so they travel with it (``light_bodyid``):

* ``{name}_fill`` -- the lantern itself: an omni light at the glass centre, warm white, with the
  inverse-square-like falloff ``attenuation="0.5 0 2.5"`` (full at the glass, a third at 1 m, a tenth
  at 2 m: a "small pool"), plus a little warm ambient so
  the surfaces beside the lamp read as lit rather than black;
* ``{name}_spot`` -- a wide downward cone from the same point that casts shadows (the cage and the
  lantern's own foot throw the pattern a real lantern throws); it adds light only below the lamp.

There is no beam and no aiming: the lamp's yaw is cosmetic. :func:`aim_yaw` and
:func:`beam_direction` serve callers that set a yaw. Both lights stay active for the whole
episode: with *no* active light MuJoCo renders flat unlit colour (about 45/255 in the policy cameras,
measured on a real search_room instance), which is not darkness -- the darkness of this task needs at
least one active light in the model, and the lantern's own lights are it.

The lantern lights its surroundings, not only itself; the reset requirement on the darkness is in
``tasks/blackout_search.py``.

Marker. The glowing glass is the emissive marker G3 measures (``GLOW_MATERIAL``, ``emission`` 0.5 with
``rgba`` alpha 0.45: visible with no light on it, and translucent so the bulb shows through). Its warm
white is nothing like the pool green ``(0.12, 0.65, 0.20)`` of spec 4.9.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from roboquest.assets._mjcf import collision, site, vec, visual, yaw_quat
from roboquest.search.covers import ElementObject

MESH_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'meshes' / 'lantern_01'
MESH_FILES = dict(brass='lantern_01_brass.obj', glass='lantern_01_glass.obj', texture='lantern_01_brass_diff_1k.png')
MESH_SCALE = .8
# Source extents at scale 1 (PROVENANCE.md): bail apex z 0.294, glass z 0.061-0.130 (radius 0.035), foot
# radius 0.061, the body about 0.045 across the cage, the cap at about z 0.20-0.22.
BAIL_Z = round(.294 * MESH_SCALE, 4)          # 0.2352: the bail's flat apex, where the gripper pinches
GLASS_Z = round(.0955 * MESH_SCALE, 4)        # 0.0764: the glass centre, where the lights sit
GLASS_R = round(.035 * MESH_SCALE, 4)         # 0.028
FOOT_R = round(.061 * MESH_SCALE, 4)          # 0.0488
BODY_R = round(.045 * MESH_SCALE, 4)          # 0.036
CAP_Z0, CAP_Z1 = round(.20 * MESH_SCALE, 4), round(.225 * MESH_SCALE, 4)   # 0.16 .. 0.18
FOOT_H = .024
BAIL_BAR_R, BAIL_BAR_HALF = .005, .011        # the pinch bar: half-thickness along the finger axis (1 cm, thicker
                                              # than the 2 mm wire), half-length along local y (2.2 cm long)
BAIL_BAR_HALF_Z = .006                        # ... and half-height: a flat-faced box the pads pinch squarely
MASS = .35
FOOT_MASS, BODY_MASS, CAP_MASS, BAIL_MASS = .23, .09, .02, .01
HANDLE_GRASP_Z = BAIL_Z                       # the pinch point, for callers that read it by that name
TOP_Z = BAIL_Z + BAIL_BAR_HALF_Z
RADIUS = FOOT_R + .002
BODY_H = CAP_Z1                               # the solid body's top (the bail rises above it)
COLUMN_XY = 0.                                # the glow sits on the lamp's own axis
RING_Z, RING_OUTER_R = GLASS_Z, GLASS_R       # the glow marker's centre and radius
HANDLE_FRICTION = '2 0.02 0.0002'             # a grippy bail apex (the carry must survive turns)
FOOT_FRICTION = '1 0.005 0.0001'

BEAM_LOCAL = (1., 0.)              # no beam leaves the lantern; the yaw is cosmetic
BEAM_TILT_DEG = 0.
GLOW_RGBA = (1., .97, .88, .45)    # the warm glass; not the pool green (.12 .65 .20)
GLOW_MATERIAL = 'rq_lamp_glass'
GLOW_EMISSION = .5
BULB_MATERIAL = 'rq_lamp_bulb'
BRASS_MATERIAL = 'rq_lantern_brass'
BRASS_TEXTURE = 'rq_lantern_brass_tex'
MESH_NAMES = dict(brass='rq_lantern_brass_mesh', glass='rq_lantern_glass_mesh')
BODY_KINDS = ('lantern_01',)

# Photometry ("E with the small pool" of the lighting sweep):
# an omni light with 1/(0.5 + 2.5 d^2) falloff lights the robot's own counter at reset (policy-camera
# means 16-41/255 on the two kitchens tried), a target 2 m away reads about 2/255 before the lamp is
# carried over and 30/255 once it is set down beside it, and at the far end of the kitchen the cameras
# read 2-6/255: silhouettes. The downward cone carries the shadows; its light stays under the lamp.
FILL = dict(cutoff='180', exponent='0', attenuation='.5 0 2.5', diffuse='.85 .75 .6', specular='0 0 0',
            ambient='.06 .05 .04')
SPOT = dict(cutoff='70', exponent='1', attenuation='.5 0 2.5', diffuse='.5 .44 .35', specular='.1 .1 .1',
            ambient='0 0 0')
AXIS_X_QUAT = vec((math.cos(math.pi / 4), 0., math.sin(math.pi / 4), 0.))   # cylinder axis along local +x (the switch's dot)
AXIS_Y_QUAT = vec((math.cos(math.pi / 4), math.sin(math.pi / 4), 0., 0.))   # capsule axis along local +y
MESH_QUAT = yaw_quat(math.pi / 2)                                            # the mesh bail into local y


def glow_material(assets, name=GLOW_MATERIAL, rgba=GLOW_RGBA, emission='1', specular='0', shininess='0'):
    """Register (once) an emissive material: visible with no light on it (the lantern's glass by default;
    the switch's dot passes its own name, colour and emission). Returns the name."""
    if assets is not None and assets.find(f"material[@name='{name}']") is None:
        ET.SubElement(assets, 'material', name=name, rgba=vec(rgba), emission=str(emission),
                      specular=str(specular), shininess=str(shininess), reflectance='0')
    return name


def lantern_assets(assets):
    """Register (once) the lantern's meshes, texture and materials on ``assets``."""
    if assets is None:
        return
    if assets.find(f"mesh[@name='{MESH_NAMES['brass']}']") is None:
        scale = vec((MESH_SCALE, MESH_SCALE, MESH_SCALE))
        ET.SubElement(assets, 'mesh', name=MESH_NAMES['brass'], file=str(MESH_DIR / MESH_FILES['brass']), scale=scale)
        ET.SubElement(assets, 'mesh', name=MESH_NAMES['glass'], file=str(MESH_DIR / MESH_FILES['glass']), scale=scale)
    if assets.find(f"texture[@name='{BRASS_TEXTURE}']") is None:
        ET.SubElement(assets, 'texture', name=BRASS_TEXTURE, type='2d', file=str(MESH_DIR / MESH_FILES['texture']))
        ET.SubElement(assets, 'material', name=BRASS_MATERIAL, texture=BRASS_TEXTURE, specular='.45', shininess='.5',
                      reflectance='.05')
    if assets.find(f"material[@name='{BULB_MATERIAL}']") is None:
        ET.SubElement(assets, 'material', name=BULB_MATERIAL, rgba='1 .96 .82 1', emission='1', specular='0',
                      shininess='0', reflectance='0')
    glow_material(assets, emission=GLOW_EMISSION, specular='.6', shininess='.8')


def aim_yaw(lamp_xy, target_xy):
    """The lamp yaw that faces ``target_xy`` (cosmetic: the lantern has no beam)."""
    return math.atan2(float(target_xy[1]) - float(lamp_xy[1]), float(target_xy[0]) - float(lamp_xy[0]))


def beam_direction(yaw):
    """Unit world direction the lamp faces at ``yaw`` (no beam leaves the lantern)."""
    tilt = math.radians(BEAM_TILT_DEG)
    return (math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), -math.sin(tilt))


def lamp(parent, name, rng=None, free=True, pos=(0., 0., 0.), yaw=0., assets=None, material=None, kind=None):
    """Append the lantern under ``parent`` (the worldbody when ``free``); see the module docstring.

    Returns a dict with ``body``, ``name``, ``joint``, ``handle_site``, ``handle_grasp_z``, ``top_z``,
    ``radius``, ``mass``, ``beam_local``, ``beam_tilt_deg``, ``ring_z``, ``ring_outer_r``, ``lights``,
    ``glow_material``, ``glow_rgba``, ``geoms``, ``material``, ``rgba``, ``kind``, ``meshes``.
    """
    if not name:
        raise ValueError('lamp needs a name')
    lantern_assets(assets)
    body = ET.SubElement(parent, 'body', name=name, pos=vec(pos), quat=yaw_quat(yaw))
    joint = None
    if free:
        joint = name + '_joint'
        ET.SubElement(body, 'freejoint', name=joint)
    # The physics: a foot, the body, the cap, the bail apex as a bar along local y (see the docstring).
    geoms = [collision(body, f'{name}_foot', 'cylinder', (0., 0., FOOT_H / 2), (FOOT_R, FOOT_H / 2), FOOT_MASS,
                       friction=FOOT_FRICTION),
             collision(body, f'{name}_column', 'cylinder', (0., 0., (FOOT_H + CAP_Z0) / 2),
                       (BODY_R, (CAP_Z0 - FOOT_H) / 2), BODY_MASS, friction=FOOT_FRICTION),
             collision(body, f'{name}_cap', 'cylinder', (0., 0., (CAP_Z0 + CAP_Z1) / 2),
                       (round(BODY_R * .7, 4), (CAP_Z1 - CAP_Z0) / 2), CAP_MASS, friction=FOOT_FRICTION),
             # A box, not a capsule: the pads meet flat faces, so the pinch holds the lantern's swing by
             # geometry instead of by torsional friction on a 1 cm round bar (the capsule slipped out during
             # the carry's base turns in grasp tests).
             collision(body, f'{name}_handle_bar', 'box', (0., 0., BAIL_Z), (BAIL_BAR_R, BAIL_BAR_HALF, BAIL_BAR_HALF_Z),
                       BAIL_MASS, friction=HANDLE_FRICTION, condim='4')]
    # The looks: the two meshes (turned so the bail runs along local y) and a bulb inside the glass.
    meshes = []
    for part, mesh_name in MESH_NAMES.items():
        attributes = dict(name=f'{name}_{part}_mesh', type='mesh', mesh=mesh_name, quat=MESH_QUAT, group='1',
                          contype='0', conaffinity='0', density='0',
                          material=BRASS_MATERIAL if part == 'brass' else GLOW_MATERIAL)
        ET.SubElement(body, 'geom', **attributes)
        meshes.append(attributes['name'])
    visual(body, f'{name}_bulb', 'capsule', (0., 0., GLASS_Z), (.009, .011), material=BULB_MATERIAL)
    ET.SubElement(body, 'light', name=f'{name}_spot', pos=vec((0., 0., GLASS_Z)), dir='0 0 -1',
                  directional='false', active='true', castshadow='true', **SPOT)
    ET.SubElement(body, 'light', name=f'{name}_fill', pos=vec((0., 0., GLASS_Z)), dir='0 0 -1',
                  directional='false', active='true', castshadow='false', **FILL)
    handle_site = site(body, f'{name}_handle', (0., 0., BAIL_Z))
    return dict(body=body, name=name, joint=joint, handle_site=handle_site,
                handle_grasp_z=HANDLE_GRASP_Z, top_z=TOP_Z, radius=RADIUS, mass=MASS,
                height=BODY_H, beam_local=BEAM_LOCAL, beam_tilt_deg=BEAM_TILT_DEG, ring_z=RING_Z,
                ring_outer_r=RING_OUTER_R, lights=(f'{name}_spot', f'{name}_fill'), glow_material=GLOW_MATERIAL,
                glow_rgba=GLOW_RGBA, geoms=geoms, meshes=meshes, material=BRASS_MATERIAL, rgba=None,
                kind=kind or BODY_KINDS[0])


def build_lamp(name, rng, assets, pos, yaw):
    """A free lantern at world ``pos`` (base plane) and ``yaw``. Returns ``(ElementObject, spec)``."""
    holder = ET.Element('holder')
    spec = lamp(holder, name, rng=rng, free=True, pos=pos, yaw=yaw, assets=assets)
    obj = ElementObject(name, spec['body'], top_z=spec['top_z'], horizontal_radius=spec['radius'])
    return obj, spec
