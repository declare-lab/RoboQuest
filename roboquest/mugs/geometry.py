"""Vessel clones (mug or bowl), coloured balls, underside label decals and colour pads.

Per instance the vessel type is drawn once for the whole scene (spec 4.6), so an
instance holds ``vessel_count`` identical mugs **or** identical bowls:

* **mug** - a clone of RoboCasa's native ``objaverse/mug/mug_1``: the original
  visual mesh, material and texture and all sixteen hollow collision hulls,
  including the real handle;
* **bowl** - the visual mesh of RoboCasa's ``objaverse/bowl/bowl_11`` with a
  **procedural hollow collision shell** in place of its convex hulls: a ring of
  sixteen thin tilted boxes tracing the interior cone plus a floor cylinder on
  the foot ring (contract C4: collision shells are hollow rings of thin boxes,
  never a convex mesh). The bowl is wide and shallow, so tilting one to read the
  label spills the balls far more easily than tilting a mug.

Both carry the same extras: two spheres inside, a flat label decal under the
floor (visual only, no collision, no mass) and nothing welded, latched or set at
runtime. The flat colour pads are the same for both.

Measured constants
------------------
Mug (standalone compile, ``probe/mug_extent.py``): collision AABB x +/-.03928,
y -.0575..+.0575 (the handle reaches -y), z +/-.04519; the bowl - and so the base
disc - is centred on the interior centre (0, .018) with outer radius .0395, and
the whole mug fits a circle of radius .0575 about its body origin at any yaw.

Bowl (visual mesh vertices at RoboCasa's own 0.12 scale, ``probe/bowl_probe.py``):
z -.0309..+.0309, outer radius .060 at the rim, foot ring radius .027, interior
floor at z=-.0200 with radius .027 widening to .0576 at the rim - a truncated
cone of half angle 31 degrees, 4.9 cm deep. Two 14 mm balls need 2.8 cm of floor
and have 3.5 cm at their centre height, so both settle inside.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

RUNTIME_ROOT = Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime')))
ASSET_ROOT = RUNTIME_ROOT / 'src/robocasa/robocasa/models/assets/objects/objaverse'
SOURCE_ASSET = ASSET_ROOT / 'mug/mug_1/model.xml'
BOWL_SOURCE_ASSET = ASSET_ROOT / 'bowl/bowl_11/model.xml'

# --- mug geometry, measured on the asset (metres, mug-local) -----------------
MUG_BOTTOM = -.04519450
MUG_RIM = .04519450
INTERIOR_FLOOR = -.03284
INTERIOR_CENTER = (0., .018)
INNER_RADIUS = .0345
BASE_RADIUS = .0395
MUG_XY_RADIUS = .0575          # bounding circle about the body origin: yaw-independent
BALL_RADIUS = .014
BALL_SPREAD = .0155            # the two balls sit this far either side of the interior centre
MUG_REST_CLEARANCE = .0015     # the vessel is dropped this far above its support
LABEL_RADIUS = .020
LABEL_HALF_THICKNESS = .0003
LABEL_DROP = .0005             # the decal sits this far below the vessel's bottom face
GRASP_OFFSET_LOCAL = (0., -.046, .027)   # the genuine handle arch; no added grasp geometry

# --- bowl geometry, measured on the visual mesh (metres, bowl-local) ---------
BOWL_BOTTOM = -.0309           # the foot ring's underside: the lowest point of the mesh
BOWL_RIM = .0309
BOWL_INTERIOR_FLOOR = -.0200   # flat interior floor inside the foot ring
BOWL_FLOOR_RADIUS = .027       # interior radius at the floor (= the foot ring radius)
BOWL_RIM_RADIUS = .0576        # interior radius at the rim
BOWL_OUTER_RADIUS = .0600      # widest point of the visual mesh
BOWL_XY_RADIUS = .0600         # bounding circle about the body origin (a bowl is round)
BOWL_WALL_THICKNESS = .004
BOWL_WALL_SEGMENTS = 16
BOWL_BALL_SPREAD = .0145       # a shade under two ball diameters: they start clear of each other
BOWL_BALL_DROP = .0005        # the balls start essentially at rest on the flat interior floor
BOWL_WALL_MASS = .18           # kg over the whole ring
BOWL_FLOOR_MASS = .12          # kg; keeps the centre of mass low, as in a real bowl
BOWL_GRASP_OFFSET_LOCAL = (0., -(BOWL_OUTER_RADIUS + BOWL_RIM_RADIUS) / 2, .020)   # pinch the rim

# --- pads --------------------------------------------------------------------
# 16 x 12.5 cm; the mug base disc (r=.0395) leaves +/-.0405 x +/-.023 of placement
# tolerance. The same disc is required of a bowl (its foot ring is only .027 across,
# so the rule is conservative there) so one number describes both vessel types.
PAD_HALF_XY = (.080, .0625)
PAD_HALF_THICKNESS = .002
PAD_FIT_RADIUS = BASE_RADIUS

# --- palettes ----------------------------------------------------------------
# Label/pad colours: the three study-room colours plus two more, so vessel_count <= 5 fits.
LABEL_COLORS = {'red': (.86, .035, .025, 1), 'blue': (.025, .14, .90, 1), 'green': (.035, .68, .16, 1),
                'yellow': (.94, .78, .04, 1), 'purple': (.46, .12, .74, 1)}
# Ball colours: the six study-room colours first, then four more so ten balls stay distinct.
BALL_COLORS = {'red': (.96, .025, .02, 1), 'blue': (.015, .12, .98, 1), 'green': (.02, .80, .13, 1),
               'yellow': (1., .84, .015, 1), 'magenta': (.94, .025, .75, 1), 'cyan': (.01, .86, .96, 1),
               'orange': (.99, .45, .02, 1), 'violet': (.52, .10, .86, 1),
               'white': (.95, .95, .95, 1), 'black': (.05, .05, .05, 1)}
# Vessel tints (spec 4.9, drawn from the `materials` stream): muted crockery colours that
# modulate the asset's own texture. Deliberately far from the saturated functional colours
# above, so a vessel's own colour can never be read as a label or a ball colour.
VESSEL_COLORS = {'porcelain': (.93, .92, .89, 1), 'cream': (.91, .85, .70, 1),
                 'sand': (.80, .72, .58, 1), 'stone': (.68, .69, .70, 1),
                 'slate': (.44, .48, .52, 1), 'clay': (.72, .52, .43, 1)}

# --- tuned contact parameters ------------------------------------------------
# Native pair friction: sliding axes, torsional, rolling axes. Explicit pairs isolate
# the support's rolling resistance from the smooth vessel interior. Measured at .002 s in
# the study room; this task runs at .001 s (see the task module).
# v1 (spec 4.6): the **rolling** terms drop tenfold, .003 -> .0003, so a spilled ball runs
# about 20 cm across the counter instead of a couple of centimetres; the sliding and
# torsional terms are unchanged.
SUPPORT_BALL_FRICTION = (.8, .8, .005, .0003, .0003)
SUPPORT_BALL_SOLREF = (.006, 1.)
SUPPORT_BALL_SOLIMP = (.95, .99, .001)
MUG_COLLISION = dict(friction='.65 .003 .00003', condim='6', solref='.006 1', solimp='.95 .99 .001')
# The bowl's own rolling term is raised to .002: inside a cone a ball with the mug's .00003
# oscillates about the centre for seconds, and the G2 settle gate asks for stillness in 100
# ticks. MuJoCo mixes geom frictions by the elementwise maximum, so this sets ball-to-bowl
# rolling resistance without touching the explicit ball-to-counter pairs below.
BOWL_COLLISION = dict(friction='.65 .003 .002', condim='6', solref='.006 1', solimp='.95 .99 .001')
BALL_COLLISION = dict(friction='.15 .001 .00001', condim='6', solref='.006 1', solimp='.95 .99 .001')
PAD_FRICTION = '.8 .003 .00003'
BALL_MASS = '.008'

# --- the two vessel types ----------------------------------------------------
# Everything the placement rule, the containment rule and the oracle need, per type.
# ``floor_radius``/``rim_radius`` describe the interior as a truncated cone (a cylinder
# when they are equal), so one containment test covers both vessels.
VESSELS = {
    'mug': dict(kind='mug', noun='mug', plural='mugs', source_asset=str(SOURCE_ASSET),
                bottom_z=MUG_BOTTOM, rim_z=MUG_RIM, interior_floor_z=INTERIOR_FLOOR,
                interior_center=list(INTERIOR_CENTER), floor_radius=INNER_RADIUS,
                rim_radius=INNER_RADIUS, inner_radius=INNER_RADIUS, base_radius=BASE_RADIUS,
                xy_radius=MUG_XY_RADIUS, pad_fit_radius=PAD_FIT_RADIUS,
                ball_spread=BALL_SPREAD, ball_drop=.003,
                grasp_offset_local=list(GRASP_OFFSET_LOCAL),
                grasp_description='Pinch the genuine top handle arch across local X; no added grasp geometry'),
    'bowl': dict(kind='bowl', noun='bowl', plural='bowls', source_asset=str(BOWL_SOURCE_ASSET),
                 bottom_z=BOWL_BOTTOM, rim_z=BOWL_RIM, interior_floor_z=BOWL_INTERIOR_FLOOR,
                 interior_center=[0., 0.], floor_radius=BOWL_FLOOR_RADIUS,
                 rim_radius=BOWL_RIM_RADIUS, inner_radius=BOWL_FLOOR_RADIUS,
                 base_radius=BOWL_FLOOR_RADIUS, xy_radius=BOWL_XY_RADIUS,
                 pad_fit_radius=PAD_FIT_RADIUS,
                 ball_spread=BOWL_BALL_SPREAD, ball_drop=BOWL_BALL_DROP,
                 grasp_offset_local=list(BOWL_GRASP_OFFSET_LOCAL),
                 grasp_description='Pinch the rim between finger and thumb; the wide shallow bowl has no handle'),
}
VESSEL_KINDS = tuple(sorted(VESSELS))


def vessel_spec(kind):
    """The geometry record of a vessel type; raises on an unknown type."""
    if kind not in VESSELS:
        raise ValueError(f'unknown vessel type {kind!r}; choose from {list(VESSEL_KINDS)}')
    return deepcopy(VESSELS[kind])


def interior_radius_at(vessel, z):
    """Interior radius of a vessel at local height ``z`` (a cone; a cylinder for the mug)."""
    floor, rim = float(vessel['interior_floor_z']), float(vessel['rim_z'])
    r0, r1 = float(vessel['floor_radius']), float(vessel['rim_radius'])
    if rim <= floor:
        return r0
    t = min(max((float(z) - floor) / (rim - floor), 0.), 1.)
    return r0 + (r1 - r0) * t


def vec(values):
    return ' '.join(str(float(v)) for v in values)


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def yaw_quat_wxyz(yaw):
    return [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]


def matrix_quat_wxyz(rotation):
    """Rotation matrix (columns = local axes) to a MuJoCo w x y z quaternion."""
    m = np.asarray(rotation, float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = [.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s]
    q = np.asarray(q, float)
    return (q / np.linalg.norm(q)).tolist()


def _clone_assets(source_root, source_path, assets, name):
    """Copy an asset's meshes, textures and materials under a per-vessel prefix; returns the name map."""
    mapping = {e.get('name'): name + '_' + e.get('name') for e in source_root.find('asset') if e.get('name')}
    cloned = []
    for original in source_root.find('asset'):
        element = deepcopy(original)
        for key in ('name', 'texture', 'material', 'mesh'):
            if element.get(key) in mapping:
                element.set(key, mapping[element.get(key)])
        if element.get('file'):
            element.set('file', str((source_path.parent / element.get('file')).resolve(strict=True)))
        assets.append(element)
        cloned.append(element)
    return mapping, cloned


def _tint_materials(cloned, rgba):
    """Per-instance vessel colour (spec 4.9): the asset's own texture, modulated by this rgba."""
    if rgba is None:
        return
    for element in cloned:
        if element.tag == 'material':
            element.set('rgba', vec(rgba))


def add_mug(world, assets, name, pos, yaw=0., source_asset=SOURCE_ASSET, rgba=None):
    """Clone the native mug into ``world`` as a free body; returns (body, collision geom names)."""
    source_asset = Path(source_asset).resolve(strict=True)
    source = ET.parse(source_asset).getroot()
    mapping, cloned = _clone_assets(source, source_asset, assets, name)
    _tint_materials(cloned, rgba)
    body = ET.SubElement(world, 'body', name=name, pos=vec(pos), quat=vec(yaw_quat_wxyz(yaw)))
    ET.SubElement(body, 'freejoint', name=name + '_joint')
    collisions = []
    for index, original in enumerate(source.find(".//body[@name='object']").findall('geom')):
        if original.get('class') == 'region':
            continue
        element = deepcopy(original)
        collision = element.get('class') == 'collision'
        element.attrib.pop('class', None)
        element.set('name', name + ('_collision_' if collision else '_visual_') + str(index))
        for key in ('mesh', 'material'):
            if element.get(key):
                element.set(key, mapping[element.get(key)])
        if collision:
            element.attrib.update(group='0', rgba='0 0 0 0', density='1000', contype='1', conaffinity='1',
                                  **MUG_COLLISION)
            collisions.append(element.get('name'))
        else:
            element.set('density', '0')
        body.append(element)
    return body, collisions


def bowl_shell_geoms():
    """The procedural hollow collision shell of a bowl, as MJCF geom attribute dicts.

    A floor cylinder on the foot ring plus a ring of ``BOWL_WALL_SEGMENTS`` thin boxes,
    each tilted by the wall angle so their inner faces trace the interior cone of the
    visual mesh. No convex hull anywhere: a ball can only be inside or outside.
    """
    wall_dz = BOWL_RIM - BOWL_INTERIOR_FLOOR
    wall_dr = BOWL_RIM_RADIUS - BOWL_FLOOR_RADIUS
    slant = math.hypot(wall_dr, wall_dz)
    alpha = math.atan2(wall_dr, wall_dz)                    # tilt from vertical
    mid_z = .5 * (BOWL_INTERIOR_FLOOR + BOWL_RIM)
    mid_r = .5 * (BOWL_FLOOR_RADIUS + BOWL_RIM_RADIUS)
    half_t = BOWL_WALL_THICKNESS / 2
    half_w = BOWL_RIM_RADIUS * math.sin(math.pi / BOWL_WALL_SEGMENTS) + .0005   # slight overlap at the rim
    floor_half = .5 * (BOWL_INTERIOR_FLOOR - BOWL_BOTTOM)
    geoms = [dict(name='floor', type='cylinder', size=vec((BOWL_FLOOR_RADIUS, floor_half)),
                  pos=vec((0., 0., BOWL_BOTTOM + floor_half)), mass=str(BOWL_FLOOR_MASS))]
    for index in range(BOWL_WALL_SEGMENTS):
        phi = 2 * math.pi * index / BOWL_WALL_SEGMENTS
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        normal = np.array([math.cos(alpha) * cos_p, math.cos(alpha) * sin_p, -math.sin(alpha)])
        tangent = np.array([-sin_p, cos_p, 0.])
        along = np.array([math.sin(alpha) * cos_p, math.sin(alpha) * sin_p, math.cos(alpha)])
        centre = np.array([mid_r * cos_p, mid_r * sin_p, mid_z]) + half_t * normal
        geoms.append(dict(name=f'wall_{index}', type='box',
                          size=vec((half_t, half_w, slant / 2)), pos=vec(centre),
                          quat=vec(matrix_quat_wxyz(np.column_stack([normal, tangent, along]))),
                          mass=str(round(BOWL_WALL_MASS / BOWL_WALL_SEGMENTS, 6))))
    return geoms


def add_bowl(world, assets, name, pos, yaw=0., source_asset=BOWL_SOURCE_ASSET, rgba=None):
    """Clone the bowl's *visual* mesh and give it a procedural hollow collision shell.

    RoboCasa ships the bowl with sixteen convex collision hulls that fill its interior,
    so a ball could never be inside it; they are dropped and replaced by
    :func:`bowl_shell_geoms`. Returns (body, collision geom names).
    """
    source_asset = Path(source_asset).resolve(strict=True)
    source = ET.parse(source_asset).getroot()
    mapping, cloned = _clone_assets(source, source_asset, assets, name)
    _tint_materials(cloned, rgba)
    body = ET.SubElement(world, 'body', name=name, pos=vec(pos), quat=vec(yaw_quat_wxyz(yaw)))
    ET.SubElement(body, 'freejoint', name=name + '_joint')
    for index, original in enumerate(source.find(".//body[@name='object']").findall('geom')):
        if original.get('class') in ('region', 'collision'):
            continue
        element = deepcopy(original)
        element.attrib.pop('class', None)
        element.set('name', f'{name}_visual_{index}')
        for key in ('mesh', 'material'):
            if element.get(key):
                element.set(key, mapping[element.get(key)])
        element.set('density', '0')
        element.attrib.update(contype='0', conaffinity='0', group='1')
        body.append(element)
    collisions = []
    for shell in bowl_shell_geoms():
        attrs = dict(shell)
        geom_name = f"{name}_collision_{attrs.pop('name')}"
        ET.SubElement(body, 'geom', name=geom_name, group='0', rgba='0 0 0 0',
                      contype='1', conaffinity='1', **attrs, **BOWL_COLLISION)
        collisions.append(geom_name)
    return body, collisions


VESSEL_BUILDERS = {'mug': add_mug, 'bowl': add_bowl}


def append_vessel_objects(world, assets, pad_body, spec, to_world, frame_yaw, frame_origin_world):
    """Build one instance's vessels, balls, labels and pads.

    ``world``/``assets``   the model's worldbody and asset elements: free bodies must be top level.
    ``pad_body``           a fixed body already placed at the task-frame origin; pads are its geoms.
    ``to_world(local)``    task frame -> world (the base class' :meth:`frame_to_world`).
    ``frame_yaw``          the work surface's yaw, added to every prop's own yaw.
    """
    count = int(spec['vessel_count'])
    kind = str(spec.get('vessel', 'mug'))
    vessel = vessel_spec(kind)
    tint = spec.get('vessel_color')
    rgba = list(VESSEL_COLORS[tint]) if tint in VESSEL_COLORS else None
    build = VESSEL_BUILDERS[kind]
    vessels, balls, pads, objects = {}, {}, {}, {}
    bottom_z = float(vessel['bottom_z'])
    body_z = -bottom_z + MUG_REST_CLEARANCE
    spread, drop = float(vessel['ball_spread']), float(vessel['ball_drop'])
    centre_xy = [float(v) for v in vessel['interior_center']]
    for index in range(count):
        name = f'vessel_{index}'
        local_xy = [float(v) for v in spec['vessel_xy'][index]]
        yaw = float(spec['vessel_yaw'][index])
        pos = to_world([local_xy[0], local_xy[1], body_z])
        body, collision_geoms = build(world, assets, name, pos, yaw + frame_yaw, rgba=rgba)
        label = spec['labels'][index]
        label_center = [centre_xy[0], centre_xy[1], bottom_z - LABEL_DROP]
        label_geom = name + '_bottom_label'
        ET.SubElement(body, 'geom', name=label_geom, type='cylinder',
                      size=f'{LABEL_RADIUS} {LABEL_HALF_THICKNESS}', pos=vec(label_center),
                      rgba=vec(LABEL_COLORS[label]), group='1', contype='0', conaffinity='0', density='0')
        info = dict(vessel, body_name=name, joint_name=name + '_joint', geom_names=collision_geoms,
                    label=label, label_geom=label_geom, label_rgba=list(LABEL_COLORS[label]),
                    label_center_local=label_center, label_normal_local=[0., 0., -1.],
                    column=index, frame_xy=local_xy, frame_yaw_rad=yaw, frame_z=body_z,
                    source_position=pos.tolist(), source_yaw_rad=yaw + frame_yaw,
                    original_balls=[f'ball_{index}_{j}' for j in range(2)],
                    original_ball_colors=list(spec['ball_colors'][index]),
                    interior_center_local=list(centre_xy), bottom_offset_z=bottom_z,
                    vessel_color=tint, vessel_rgba=list(rgba) if rgba else None, uniform_scale=1.0)
        vessels[name] = info
        objects[name] = info
        rotation = yaw_matrix(yaw)
        for j in range(2):
            ball = f'ball_{index}_{j}'
            offset = rotation @ np.array([(2 * j - 1) * spread + centre_xy[0], centre_xy[1],
                                          float(vessel['interior_floor_z']) + BALL_RADIUS + drop])
            ball_local = [local_xy[0] + offset[0], local_xy[1] + offset[1], body_z + offset[2]]
            ball_pos = to_world(ball_local)
            element = ET.SubElement(world, 'body', name=ball, pos=vec(ball_pos))
            ET.SubElement(element, 'freejoint', name=ball + '_joint')
            colour = spec['ball_colors'][index][j]
            common = dict(type='sphere', size=str(BALL_RADIUS), rgba=vec(BALL_COLORS[colour]))
            geom = ball + '_sphere'
            ET.SubElement(element, 'geom', name=geom, group='0', mass=BALL_MASS, **BALL_COLLISION, **common)
            ET.SubElement(element, 'geom', name=geom + '_visual', group='1', density='0',
                          contype='0', conaffinity='0', **common)
            ball_info = dict(body_name=ball, joint_name=ball + '_joint', geom_names=[geom], original_vessel=name,
                             radius=BALL_RADIUS, source_position=ball_pos.tolist(), frame_xy=ball_local[:2],
                             grasp_offset_local=[0., 0., 0.], color=colour, rgba=list(BALL_COLORS[colour]))
            balls[ball] = ball_info
            objects[ball] = ball_info
    pad_yaws = spec.get('pad_yaw') or [0.] * len(spec['pad_colors'])
    for column, colour in enumerate(spec['pad_colors']):
        x, y = (float(v) for v in spec['pad_xy'][column])
        pad_yaw = float(pad_yaws[column])
        geom = f'pad_{colour}_floor'
        # The pad carries its own yaw inside the frame body, so a jittered back row is
        # not a set of parallel rectangles; the placement rule tests in pad-local xy.
        common = dict(type='box', size=f'{PAD_HALF_XY[0]} {PAD_HALF_XY[1]} {PAD_HALF_THICKNESS}',
                      pos=vec((x, y, PAD_HALF_THICKNESS)), quat=vec(yaw_quat_wxyz(pad_yaw)),
                      rgba=vec(LABEL_COLORS[colour]))
        ET.SubElement(pad_body, 'geom', name=geom, group='0', friction=PAD_FRICTION, **common)
        ET.SubElement(pad_body, 'geom', name=geom + '_visual', group='1', contype='0', conaffinity='0', **common)
        pads[colour] = dict(center_xy=[x, y], half_size_xy=list(PAD_HALF_XY), column=column,
                            yaw_rad=pad_yaw, floor_z=2 * PAD_HALF_THICKNESS, floor_geom=geom,
                            center_world=to_world([x, y, 2 * PAD_HALF_THICKNESS]).tolist())
    return dict(objects=objects, vessels=vessels, balls=balls, pads=pads,
                frame_origin_world=[float(v) for v in frame_origin_world], frame_yaw=float(frame_yaw),
                vessel_count=count, vessel=kind, vessel_geometry=vessel, vessel_color=tint,
                ball_radius=BALL_RADIUS, pad_fit_radius=PAD_FIT_RADIUS,
                source_asset=vessel['source_asset'],
                material_effect='Native rigid contact only: no lids, catches, welds, attachments or object setters.'
                                + (' The bowl keeps its RoboCasa visual mesh; its collision is a procedural hollow '
                                   'shell (ring of thin tilted boxes plus a floor cylinder).' if kind == 'bowl'
                                   else ''),
                ball_color_assignment='Ball colours are drawn from the materials stream, independently of the '
                                      'label permutation drawn from the structure stream.',
                label_visibility='Underside decal: visual only, radius .020 inside the base disc and '
                                 '.0005 m below the bottom face, so an upright vessel hides it from every camera.',
                pad_placement='Pads are scattered in a loose back row: per-pad xy jitter and a small yaw, '
                              'so the row is not a fixed pitch and the pads are not all parallel.')


def append_support_ball_contacts(root, geometry, support_geoms):
    """Construction-only native friction pairs between every ball and every support surface.

    ``support_geoms`` are the work fixture's own top collision tiles; the pads are
    added here as well. No trapping, damping or state edits: a ball in flight has
    no contact resistance and may leave the counter, exactly as on the study table.
    """
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    surfaces, seen = [], set()
    for surface in list(support_geoms) + [pad['floor_geom'] for pad in geometry['pads'].values()]:
        if surface not in seen:         # order-preserving dedupe
            seen.add(surface)
            surfaces.append(surface)
    if not surfaces:
        raise ValueError('no support geoms: the ball rolling pairs would be missing')
    # Idempotent: RoboCasa re-runs ``_load_model`` on the same XML tree when its first robot
    # spawn collides (seen on the stamps dining-table layout 19), and a second append would
    # give MuJoCo a repeated ``<pair>`` name. Rewrite the pair that is already there.
    existing = {pair.get('name'): pair for pair in contact.findall('pair')}
    names = []
    for ball, info in geometry['balls'].items():
        for surface in surfaces:
            name = f'{ball}__{surface}_rolling'
            element = existing.get(name)
            if element is None:
                element = ET.SubElement(contact, 'pair', name=name)
                existing[name] = element
            element.set('geom1', info['geom_names'][0])
            element.set('geom2', surface)
            element.set('condim', '6')
            element.set('friction', vec(SUPPORT_BALL_FRICTION))
            element.set('solref', vec(SUPPORT_BALL_SOLREF))
            element.set('solimp', vec(SUPPORT_BALL_SOLIMP))
            names.append(name)
    return dict(pair_names=names, surface_geoms=surfaces, condim=6,
                friction_5d=list(SUPPORT_BALL_FRICTION), solref=list(SUPPORT_BALL_SOLREF),
                solimp=list(SUPPORT_BALL_SOLIMP),
                scope='Ball-to-counter-top and ball-to-pad pairs only; vessel contacts are left to MuJoCo geom '
                      'mixing, which on a RoboCasa counter top (friction 1 .005 .0001, condim 3, solref .02 1) '
                      'gives the vessel condim 6 and friction 1 .005 .0001 - the same combination the study '
                      'room table produced.',
                rolling='Rolling terms .0003 (v1, spec 4.6): a spilled ball runs about 20 cm.')
