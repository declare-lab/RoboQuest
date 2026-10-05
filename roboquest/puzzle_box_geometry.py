"""Mechanical puzzle box MJCF: a lid locked by a chain of sliding bolts.

The geometry is driven by a layout from `puzzle_box_layouts` (sampled per seed
or the hand-designed default). Every moving part is a rigid body on one
prismatic joint whose axis is its withdrawal direction. Blocking is ordinary
contact: a bolt tip resting inside a notch cut into the next part. Nothing in
the MJCF encodes the release order. Item and tray are RoboCasa objects imported
with their original meshes, textures and convex collisions.

Re-lock coupling (RoboQuest v1 spec 4.3). A layout may name one decoy and one
chain bolt in ``layout['relock']``. The bolt then carries a *striker*, a box
hanging from its underside into the chest wall, and the decoy carries a rigid
*push rod* whose *head* sits ``BOLT_TRAVEL + ROD_CLEARANCE`` behind the
striker while the decoy is at rest. Pulling the decoy (its handle points the
opposite way from the bolt's) drives the head against the striker and pushes
the bolt back to within ``ROD_CLEARANCE`` of its locked position.

Why the decoy's joint carries a hidden return spring: the striker presses the
head in the same direction whether the decoy is pushing the bolt home or the
bolt is being withdrawn against a pulled decoy, so no passive rod can
transmit the first and yield to the second (a compliant rod fails to re-lock
against the bolt's friction, measured; a rigid one lets the bolt's withdrawal
shove the decoy home, the two-way betrayal spec 4.3 forbids). With the return
spring the decoy slides home by itself as soon as the robot lets go, taking
the rod out of the bolt's path: releasing the bolt afterwards meets no rod
and never moves the decoy, and the pulled decoy is never a stop for the
bolt. The spring is invisible; its friction loss is lowered so it returns to
within ``RELOCK_DECOY_FRICTION / RELOCK_DECOY_STIFFNESS`` (2 mm) of home,
inside the head's clearance. Every coupling geom is collision-only with a
private contact bit, so it meets nothing but its counterpart, lies inside the
chest's solid walls and floor (5 mm under the deck top; the rod's shaft runs
inside the floor slab) and has no visual twin: from outside the re-lock decoy
is any other slider.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from roboquest.puzzle_box_layouts import BOLT_TRAVEL, CAVITY, HANDLE_RADIUS, default_layout, release_order

ROBOCASA_OBJECTS = (Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime')))
                    / 'src/robocasa/robocasa/models/assets/objects/objaverse')

DECK_TOP = .075             # chest height; the sliding layer floats just above it
PART_HALF_Z = .006          # sliding layer half thickness
LAYER_GAP = .001            # a horizontal slide joint cannot fall; a resting contact
                            # would only add sticky friction to every slide
HANDLE_HALF_Z = .03
COVER_HALF_Z = .002
LAYER_Z = (DECK_TOP + LAYER_GAP, DECK_TOP + LAYER_GAP + 2 * PART_HALF_Z)
HANDLE_Z = LAYER_Z[1] + HANDLE_HALF_Z
COVER_Z = LAYER_Z[1] + .0005 + COVER_HALF_Z
FLOOR_TOP = .01
ITEM_MAX_FOOTPRINT = .064   # parallel gripper opens 80 mm
ITEM_HEIGHT_RANGE = (.03, .062)
PART_MASS = {'lid': .16, 'bolt': .07, 'decoy': .06}
# Re-lock coupling (see the module docstring). Positions are chest-local; z above the chest base.
COUPLING_BITS = 1 << 5      # private contype/conaffinity bit: striker and rod head meet only each other
STRIKER_Q = .035            # striker centre from the bolt's tip end, along the bolt: inside the chest wall
COUPLING_HALF = (.005, .010)  # half thickness along the bolt, half width across it (striker and head alike)
COUPLING_Z = (.045, .070)   # striker and head band: inside the wall, 5 mm under the deck top
ROD_Z = (.003, .008)        # the rod's shaft runs inside the 10 mm floor slab
ROD_CLEARANCE = .007        # head-to-striker gap with the decoy home and the bolt fully withdrawn: room for
                            # the bolt's soft-limit overshoot under a hard pull (3 mm at 12 N) and the decoy's
                            # 2 mm return residual; the re-lock stroke ends this far from the bolt's limit
RELOCK_DECOY_STIFFNESS = 100.   # N/m return spring on the re-lock decoy's joint: 5 N at full pull
RELOCK_DECOY_FRICTION = .2      # its friction loss (other sliders: 1.2 N) so the spring parks it within 2 mm
COUPLING_MASS = .004
_EXTENT_CACHE = {}


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _box(body, name, pos, half, rgba, mass=None, friction='0.6 0.005 0.0001', condim='3'):
    attributes = dict(name=name, type='box', pos=_vec(pos), size=_vec(half), group='0',
                      rgba=_vec(rgba), friction=friction, condim=condim,
                      solref='0.006 1', solimp='0.98 0.999 0.001', margin='0')
    if mass is not None:
        attributes['mass'] = str(mass)
    ET.SubElement(body, 'geom', **attributes)
    ET.SubElement(body, 'geom', name=name + '_visual', type='box', pos=_vec(pos), size=_vec(half),
                  rgba=_vec(rgba), group='1', contype='0', conaffinity='0', density='0')
    return name


def _rect_box(body, name, rect, z, rgba, mass=None, **kw):
    (x0, x1), (y0, y1) = rect
    pos = [(x0 + x1) / 2, (y0 + y1) / 2, (z[0] + z[1]) / 2]
    half = [(x1 - x0) / 2, (y1 - y0) / 2, (z[1] - z[0]) / 2]
    return _box(body, name, pos, half, rgba, mass, **kw)


def _rgba(rgb):
    return (*rgb, 1.)


def _coupling_geom(body, name, pos, half, quat=None):
    """A hidden coupling piece: collision-only, private contact bit, fully transparent, no visual twin."""
    attributes = dict(name=name, type='box', pos=_vec(pos), size=_vec(half), group='0', rgba='0 0 0 0',
                      contype=str(COUPLING_BITS), conaffinity=str(COUPLING_BITS), mass=str(COUPLING_MASS),
                      friction='0.3 0.005 0.0001', condim='3', solref='0.006 1', solimp='0.98 0.999 0.001',
                      margin='0')
    if quat is not None:
        attributes['quat'] = _vec(quat)
    ET.SubElement(body, 'geom', **attributes)
    return name


def _along_across(part, along, across, z_half):
    """Half sizes of a box measured along and across a part's axis."""
    half = [0., 0., z_half]
    half[part['axis_index']], half[1 - part['axis_index']] = along, across
    return half


def _tip_point(part):
    point = [0., 0.]
    point[part['axis_index']] = part['tip_end']
    point[1 - part['axis_index']] = part['perp_center']
    return point


def coupling_points(bolt):
    """Chest-local xy of the striker's centre and of the rod head's centre (decoy at rest, bolt locked)."""
    tip = _tip_point(bolt)
    along = bolt['axis']
    striker = [tip[0] + along[0] * STRIKER_Q, tip[1] + along[1] * STRIKER_Q]
    reach = STRIKER_Q + COUPLING_HALF[0] + BOLT_TRAVEL + ROD_CLEARANCE + COUPLING_HALF[0]
    head = [tip[0] + along[0] * reach, tip[1] + along[1] * reach]
    return striker, head


def _striker(body, name, bolt):
    striker, _ = coupling_points(bolt)
    z0, z1 = COUPLING_Z
    return _coupling_geom(body, f'puzzle_{name}_striker', (striker[0], striker[1], (z0 + z1) / 2),
                          _along_across(bolt, COUPLING_HALF[0], COUPLING_HALF[1], (z1 - z0) / 2))


def _push_rod(body, name, decoy, bolt):
    """The decoy's rigid push rod: head behind the striker, post under the decoy, shaft in the floor slab."""
    _, head = coupling_points(bolt)
    post = _tip_point(decoy)
    post[decoy['axis_index']] += decoy['sigma'] * decoy['length'] / 2    # under the decoy's middle
    tall = ((ROD_Z[0] + COUPLING_Z[1]) / 2, (COUPLING_Z[1] - ROD_Z[0]) / 2)
    geoms = [_coupling_geom(body, f'puzzle_{name}_rod_head', (head[0], head[1], tall[0]),
                            _along_across(bolt, COUPLING_HALF[0], COUPLING_HALF[1], tall[1])),
             _coupling_geom(body, f'puzzle_{name}_rod_post', (post[0], post[1], tall[0]), (.005, .005, tall[1]))]
    dx, dy = head[0] - post[0], head[1] - post[1]
    length = math.hypot(dx, dy)
    if length > .012:
        yaw = math.atan2(dy, dx)
        geoms.append(_coupling_geom(body, f'puzzle_{name}_rod_shaft',
                                    ((head[0] + post[0]) / 2, (head[1] + post[1]) / 2, (ROD_Z[0] + ROD_Z[1]) / 2),
                                    (length / 2, .005, (ROD_Z[1] - ROD_Z[0]) / 2),
                                    quat=(math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))))
    return geoms


def _chest(world, base_pos, layout):
    colours = layout['colours']
    body = ET.SubElement(world, 'body', name='puzzle_chest', pos=_vec(base_pos))
    (x0, x1), (y0, y1) = layout['deck']['x'], layout['deck']['y']
    (cx0, cx1), (cy0, cy1) = CAVITY
    names = [_rect_box(body, 'chest_floor', ((x0, x1), (y0, y1)), (0., FLOOR_TOP), _rgba(colours['wood']))]
    walls = {'chest_front': ((x0, x1), (y0, cy0)), 'chest_back': ((x0, x1), (cy1, y1)),
             'chest_left': ((x0, cx0), (cy0, cy1)), 'chest_right': ((cx1, x1), (cy0, cy1))}
    for name, rect in walls.items():
        names.append(_rect_box(body, name, rect, (FLOOR_TOP, DECK_TOP),
                               _rgba(colours['deck'] if name == 'chest_front' else colours['wood'])))
    names.extend(_plate(body, layout))
    return body, names


def _plate(body, layout):
    """The mechanism plate and its legs, welded to the chest body.

    One rigid opaque slab a few millimetres over the sliding layer, with its legs standing
    on the deck in columns no slider sweep enters (`roboquest.puzzle.plate`). It collides
    like any other chest geom: a slider that tried to rise would hit it, the lid passes
    under it, and a hand that reaches for a hidden tip meets it.
    """
    plate = layout.get('plate')
    if not plate:
        return []
    colours = layout['colours']
    z, leg_z, half = tuple(plate['z']), tuple(plate['leg_z']), plate['leg_half']
    names = []
    for index, ((x0, x1), (y0, y1)) in enumerate(plate['rects']):
        names.append(_rect_box(body, f'chest_plate{index}', ((x0, x1), (y0, y1)), z,
                               _rgba(colours['deck'])))
    for index, (lx, ly) in enumerate(plate['legs']):
        names.append(_rect_box(body, f'chest_plate_leg{index}',
                               ((lx - half, lx + half), (ly - half, ly + half)), leg_z,
                               _rgba(colours['wood'])))
    wall_z = tuple(plate.get('wall_z') or leg_z)
    for index, ((x0, x1), (y0, y1)) in enumerate(plate.get('walls') or ()):
        names.append(_rect_box(body, f'chest_plate_wall{index}', ((x0, x1), (y0, y1)), wall_z,
                               _rgba(colours['deck'])))
    skirt_z = tuple(plate.get('skirt_z') or z)
    for index, ((x0, x1), (y0, y1)) in enumerate(plate.get('skirts') or ()):
        names.append(_rect_box(body, f'chest_plate_skirt{index}', ((x0, x1), (y0, y1)), skirt_z,
                               _rgba(colours['deck'])))
    lintel_z = tuple(plate.get('lintel_z') or z)
    for index, ((x0, x1), (y0, y1)) in enumerate(plate.get('lintels') or ()):
        names.append(_rect_box(body, f'chest_plate_lintel{index}', ((x0, x1), (y0, y1)), lintel_z,
                               _rgba(colours['deck'])))
    return names


def _under_plate(part, layout):
    """The part's resting footprint clipped to the plate, i.e. what the plate hides of it.

    The evaluator's render check (G3) samples these rectangles: no pixel of them may reach
    a policy camera. They are computed at rest because that is the state at reset.
    """
    plate = layout.get('plate')
    if not plate:
        return []
    out = []
    for (x0, x1), (y0, y1) in part['spans']:
        for (px0, px1), (py0, py1) in plate['rects']:
            ix, iy = (max(x0, px0), min(x1, px1)), (max(y0, py0), min(y1, py1))
            if ix[1] - ix[0] > 1e-9 and iy[1] - iy[0] > 1e-9:
                out.append([list(ix), list(iy)])
    return out


def _part(world, name, part, base_pos, layout):
    colours = layout['colours']
    body = ET.SubElement(world, 'body', name=f'puzzle_{name}', pos=_vec(base_pos))
    axis = (part['axis'][0], part['axis'][1], 0.)
    relock = layout.get('relock')
    joint = dict(name=f'puzzle_{name}_slide', type='slide', axis=_vec(axis), range=_vec(part['range']),
                 limited='true', damping='6', frictionloss='1.2')
    if relock and relock['decoy'] == name:
        # Hidden return spring (see the module docstring): the same damping, a lower friction loss.
        joint.update(stiffness=str(RELOCK_DECOY_STIFFNESS), springref='0', frictionloss=str(RELOCK_DECOY_FRICTION))
    ET.SubElement(body, 'joint', **joint)
    rgba = _rgba(colours['lid'] if part['kind'] == 'lid' else colours['metal'])
    handle_mass = .02
    mass_each = (PART_MASS[part['kind']] - handle_mass) / len(part['spans'])
    geoms = []
    for index, rect in enumerate(part['spans']):
        geoms.append(_rect_box(body, f'puzzle_{name}_seg{index}', rect, LAYER_Z, rgba, mass_each))
    if part.get('cover') is not None:
        geoms.append(_rect_box(body, f'puzzle_{name}_cover', part['cover'],
                               (COVER_Z - COVER_HALF_Z, COVER_Z + COVER_HALF_Z), rgba, .005))
    hx, hy = part['handle']
    handle = f'puzzle_{name}_handle'
    # Round post: graspable at any wrist yaw; slides are driven through grip friction.
    ET.SubElement(body, 'geom', name=handle, type='cylinder', pos=_vec((hx, hy, HANDLE_Z)),
                  size=_vec((HANDLE_RADIUS, HANDLE_HALF_Z)), group='0', rgba=_vec(_rgba(colours['handle'])),
                  mass=str(handle_mass), friction='1.5 0.01 0.001', condim='4', solref='0.006 1',
                  solimp='0.98 0.999 0.001', margin='0')
    ET.SubElement(body, 'geom', name=handle + '_visual', type='cylinder', pos=_vec((hx, hy, HANDLE_Z)),
                  size=_vec((HANDLE_RADIUS, HANDLE_HALF_Z)), rgba=_vec(_rgba(colours['handle'])), group='1',
                  contype='0', conaffinity='0', density='0')
    geoms.append(handle)
    ET.SubElement(body, 'site', name=f'puzzle_{name}_grasp', pos=_vec((hx, hy, HANDLE_Z)),
                  size='0.002', rgba='0 0 0 0', group='5')
    relock = layout.get('relock')
    if relock and relock['bolt'] == name:
        _striker(body, name, part)
    if relock and relock['decoy'] == name:
        _push_rod(body, name, part, layout['parts'][relock['bolt']])
    return body, geoms


def measure_asset_extent(source_asset):
    """Collision AABB of a RoboCasa object in its own frame, from a standalone compile."""
    source_asset = str(Path(source_asset).resolve(strict=True))
    if source_asset in _EXTENT_CACHE:
        return _EXTENT_CACHE[source_asset]
    import mujoco
    import numpy as np
    root = ET.parse(source_asset).getroot()
    for element in root.iter():
        if element.get('file'):
            element.set('file', str((Path(source_asset).parent / element.get('file')).resolve(strict=True)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        low, high = np.minimum(low, world.min(0)), np.maximum(high, world.max(0))
    extent = dict(low=low.tolist(), high=high.tolist())
    _EXTENT_CACHE[source_asset] = extent
    return extent


def item_fits(extent):
    size = [extent['high'][i] - extent['low'][i] for i in range(3)]
    return max(size[0], size[1]) <= ITEM_MAX_FOOTPRINT and ITEM_HEIGHT_RANGE[0] <= size[2] <= ITEM_HEIGHT_RANGE[1]


def add_robocasa_object(world, assets, name, pos, source_asset, yaw=0., free=True,
                        friction='0.8 0.005 0.0001', condim='4', density='600'):
    """Copy one RoboCasa objaverse object (meshes, textures, collisions) into the arena.

    The original visual mesh, texture, material and every original convex
    collision piece are retained unchanged; only names are prefixed. Source files
    are read, never modified. Returns (body, collision geom names).
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
        assets.append(element)
    body = ET.SubElement(world, 'body', name=name, pos=_vec(pos),
                         quat=_vec([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    if free:
        ET.SubElement(body, 'freejoint', name=name + '_joint')
    collisions = []
    for index, original in enumerate(source.find(".//body[@name='object']").findall('geom')):
        if original.get('class') == 'region':
            continue
        element = deepcopy(original)
        # RoboCasa objects mark collisions by class, group 0 or mesh name; visuals are group 1.
        collision = (element.get('class') == 'collision' or element.get('group') == '0'
                     or 'collision' in (element.get('mesh') or '').lower())
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
            element.set('density', '0')
        body.append(element)
    return body, collisions


def append_puzzle_box_objects(world, table_z, box_xy, tray_xy, assets=None, submit_xy=None, layout=None):
    """Append the chest and sliding parts from `layout` and, when `assets` is
    given, the RoboCasa item inside the cavity and the RoboCasa tray on the table.

    Returns evaluator/oracle metadata. The robot receives only the public goal
    and its ordinary cameras; nothing here is policy observation.
    """
    layout = layout or default_layout()
    for xy in (box_xy, tray_xy):
        if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
            raise ValueError('Positions must contain two finite coordinates')
    if not math.isfinite(float(table_z)):
        raise ValueError('table_z must be finite')
    base = [float(box_xy[0]), float(box_xy[1]), float(table_z)]
    _, chest_geoms = _chest(world, base, layout)
    parts = {}
    for name, part in layout['parts'].items():
        _, geoms = _part(world, name, part, base, layout)
        parts[name] = dict(
            body_name=f'puzzle_{name}', joint_name=f'puzzle_{name}_slide', geom_names=geoms,
            handle_geom=f'puzzle_{name}_handle', kind=part['kind'], axis=[part['axis'][0], part['axis'][1], 0.],
            range=list(part['range']), rest=0., released_at=part['released_at'], open_at=part['open_at'],
            blocked_by=part['blocked_by'], decoy=bool(part['decoy']),
            handle_center_world=[base[0] + part['handle'][0], base[1] + part['handle'][1], base[2] + HANDLE_Z],
            handle_radius=HANDLE_RADIUS, handle_half_height=HANDLE_HALF_Z, grasp_yaw_deg=0.,
            spans_local=[list(map(list, r)) for r in part['spans']],
            notch_local=list(map(list, part['notch'])) if part['notch'] else None,
            # The engaged end of a bolt, in the chest frame: the tip sits at `tip_end_local`
            # and the body runs from there towards the handle along `sigma`. A lid has none.
            axis_index=part.get('axis_index'), sigma=part.get('sigma'),
            tip_end_local=part.get('tip_end'), perp_center_local=part.get('perp_center'),
            cover_local=list(map(list, part['cover'])) if part.get('cover') else None,
            under_plate_local=_under_plate(part, layout),
            travel=part['range'][1] - part['range'][0])
    plate = layout.get('plate')
    plate_meta = None
    if plate:
        plate_meta = dict(
            rects_local=[list(map(list, rect)) for rect in plate['rects']],
            legs_local=[list(leg) for leg in plate['legs']], leg_half=plate['leg_half'],
            walls_local=[list(map(list, wall)) for wall in plate.get('walls') or ()],
            wall_z_local=list(plate.get('wall_z') or plate['leg_z']),
            skirts_local=[list(map(list, box)) for box in plate.get('skirts') or ()],
            skirt_z_local=list(plate.get('skirt_z') or plate['z']),
            lintels_local=[list(map(list, bar)) for bar in plate.get('lintels') or ()],
            lintel_z_local=list(plate.get('lintel_z') or plate['z']),
            z_local=list(plate['z']), leg_z_local=list(plate['leg_z']),
            z_world=[base[2] + plate['z'][0], base[2] + plate['z'][1]],
            clearance_m=plate['clearance_m'], hides=list(plate['hides']),
            geom_names=[n for n in chest_geoms if n.startswith('chest_plate')],
            construction='one rigid opaque slab welded to the chest over the sliding layer, on legs '
                         'that stand in columns no slider sweep enters; sliders travel underneath it')
    relock = layout.get('relock')
    relock_meta = None
    if relock:
        striker, head = coupling_points(layout['parts'][relock['bolt']])
        relock_meta = dict(relock, striker_geom=f"puzzle_{relock['bolt']}_striker",
                           head_geom=f"puzzle_{relock['decoy']}_rod_head",
                           striker_local=striker, head_local=head, clearance_m=ROD_CLEARANCE,
                           decoy_return_stiffness=RELOCK_DECOY_STIFFNESS, decoy_frictionloss=RELOCK_DECOY_FRICTION,
                           coupling_bits=COUPLING_BITS,
                           construction='rigid hidden push rod on the decoy pushes the bolt home; the decoy joint '
                                        'carries a hidden return spring, so letting go retracts the rod and the '
                                        'bolt is never blocked nor the decoy moved by its release')
    metadata = dict(
        construction='Fixed rigid chest; prismatic sliding parts blocked only by notch contact; '
                     'free RoboCasa item inside; fixed RoboCasa tray. No attachment, latch logic or pose setter.',
        relock=relock_meta, plate=plate_meta,
        layout_seed=layout.get('seed'), layout_signature=layout.get('signature'),
        lid_direction=layout['lid_direction'], chain_length=layout['chain_length'],
        covered_locks={k: bool(v) for k, v in layout.get('covered', {}).items()},
        box_origin_world=base, chest_geom_names=chest_geoms, deck_top_z=base[2] + DECK_TOP,
        deck_local=dict(x=list(layout['deck']['x']), y=list(layout['deck']['y'])),
        layer_z_world=[base[2] + LAYER_Z[0], base[2] + LAYER_Z[1]],
        cavity_local=dict(x=list(CAVITY[0]), y=list(CAVITY[1]), floor_z=FLOOR_TOP),
        parts=parts, release_order=release_order(layout), colours=layout['colours'])
    if assets is not None:
        item_asset = ROBOCASA_OBJECTS / layout['item_asset'] / 'model.xml'
        tray_asset = ROBOCASA_OBJECTS / layout['tray_asset'] / 'model.xml'
        extent = measure_asset_extent(item_asset)
        if not item_fits(extent):
            raise ValueError(f'Item asset does not fit the cavity or gripper: {item_asset} {extent}')
        item_xy = layout['item_xy']
        height = extent['high'][2] - extent['low'][2]
        # A fitted inner platform lifts the item so its top sits 4 mm under the lid.
        # Grasping at 42 percent of its height then keeps the hand above the sliding layer.
        platform_top = LAYER_Z[0] - .004 - height
        if platform_top < FLOOR_TOP:
            raise ValueError(f'Item too tall for the cavity: {item_asset} height {height:.4f}')
        chest = world.find("body[@name='puzzle_chest']")
        (cx0, cx1), (cy0, cy1) = CAVITY
        _rect_box(chest, 'chest_platform', ((cx0 + .002, cx1 - .002), (cy0 + .002, cy1 - .002)),
                  (FLOOR_TOP, platform_top), _rgba(layout['colours']['wood']))
        item_pos = [base[0] + item_xy[0], base[1] + item_xy[1], base[2] + platform_top - extent['low'][2] + .001]
        # High sliding and torsional friction: a round item is held by two flat pads only.
        _, item_geoms = add_robocasa_object(world, assets, 'puzzle_item', item_pos, item_asset,
                                            friction='2.0 0.05 0.002')
        tray_extent = measure_asset_extent(tray_asset)
        tray_pos = [float(tray_xy[0]), float(tray_xy[1]), float(table_z) - tray_extent['low'][2]]
        _, tray_geoms = add_robocasa_object(world, assets, 'puzzle_tray', tray_pos, tray_asset, yaw=math.pi / 2,
                                            free=False)
        metadata['item'] = dict(body_name='puzzle_item', joint_name='puzzle_item_joint', geom_names=item_geoms,
                                source_asset=str(item_asset), source_position=item_pos, extent=extent,
                                bottom_offset=extent['low'][2], height=height, platform_top_local=platform_top,
                                # Pinch at mid-height: on a round item an off-centre pinch squeezes it
                                # out of the pads (down when above, up into the finger links when below).
                                grasp_offset_local=[0., 0., extent['low'][2] + .5 * height])
        metadata['chest_geom_names'] = chest_geoms + ['chest_platform']
        metadata['tray'] = dict(body_name='puzzle_tray', geom_names=tray_geoms, source_asset=str(tray_asset),
                                center_world=tray_pos, yaw_rad=math.pi / 2, extent=tray_extent,
                                note='Inner floor height and inner bounds are measured by the compiled mesh')
    if submit_xy is not None:
        metadata['submit_xy'] = [float(submit_xy[0]), float(submit_xy[1])]
    return metadata
