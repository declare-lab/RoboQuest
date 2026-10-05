"""Layout-generic compartment discovery, place assignment and counter spots for the search tasks.

Pure geometry over plain fixture records (dicts), so every rule is CPU-testable
without a simulator. A fixture record carries the fields the scene extracts from
RoboCasa at build time: name, class, world position, yaw, size [w, d, h],
interior regions, door joints and their ranges, handle names and panel type.

RoboCasa cabinet/drawer/counter local frames: +x along the front, -y points out
of the front face (doors and handles sit at local -y), +z up; `rot` is a yaw.
Counters and cabinets report full sizes; wall records report half extents and
are only used for the room bounds.

Copied from the room-search worktree (`room_search_layout.py`, logic kept) and
extended with the place assignment for a *set* of targets and the counter-spot
sampling used for `open` hiding places and visible objects.
"""
from copy import deepcopy
import math

import numpy as np

from roboquest.search.objects import POOL, fits, fits_compartment

STORAGE_CLASSES = {'Drawer': 'drawer', 'SingleCabinet': 'door', 'HingeCabinet': 'door'}
GLASS_PANELS = {'full_window', 'divided_window'}
REACH_Z = (.22, 1.10)          # region floor height band the held-torso arm can reach into
MIN_INTERIOR = (.13, .13, .11)  # width, depth, height needed to hide the target mug
DRAWER_STANDOFF_M = .62        # base frame to drawer front when grasping the handle (arm reach at counter height
                               # ends near 0.60 m; the drawer is then pulled by driving the base backward)
DOOR_STANDOFF_M = .55          # base frame to cabinet front: the bar is grasped from above (reach ~0.6 m),
                               # then the leaf is pulled by reversing the base and pushed the rest of the way
STANDOFF_LOW_GAIN = .5         # extra distance per metre the handle sits below 0.80 m (arm reach, not clearance)
TRAY_STANDOFF_M = .50          # base frame to counter front for placing on the tray (T1 placements used .45)
DRAWER_PULL_M = .40            # opening reached by backing the base up while holding the handle
REAR_ROOM_M = DRAWER_PULL_M + .15   # free floor needed behind a drawer standoff for that reverse
DOOR_LATERAL_M = .25           # doors: stand in front of the handle side so the swung leaf clears the base
FLOOR_MARGIN_M = .38           # base centre must stay this far inside the room bounds
BLOCKER_MAX_BOTTOM_Z = .90     # fixtures whose underside is higher than this never block the base
TRAY_SIZE_M = .34
NON_BLOCKING_CLASSES = ('Wall', 'Floor', 'Window', 'Hood')
BASE_CLEARANCE_M = .27         # half base width plus margin: a standoff must keep this from every other footprint
SPOT_EDGE_BAND_M = (.12, .24)  # open/visible spots sit this far inside an approachable counter edge (top grasp reach)
SPOT_STANDOFF_M = .50          # base frame to the counter edge when approaching a counter spot
OPEN_SHELF_MAX_Z = 1.15        # open shelves above this are out of the arm's reach


# ---- layout capacity (mint-time gate) -----------------------------------------------------------
# What every RoboCasa layout can host, so `sample_spec` can reject a (layout, compartment_count,
# set_size) before a kitchen is ever built (the build discovering the shortfall wasted a third of the
# level-8 candidates in the first v1 registry). Per layout: the interior height in millimetres of
# every usable compartment as {height: how many} by kind, and the counts of edge-approachable counter
# spots (`open`), spots that can take the cloche with a set-down position (`cover`) and interior spots
# for the visible objects (`free`). `None` means the layout fails the build outright.
#
# The compartment histograms are a property of the layout, not of the style (checked on layouts 4, 13
# and 26 under styles 1 and 12: identical). The three spot counts are style-dependent within a few
# percent (style 1 vs 12 on those layouts: open 28/28, 18/17, 30/28; cover 13/11, 7/6, 11/15; free
# 78/84, 86/88, 84/81), so the gate treats them as hints with slack and the build stays authoritative.
# Every probed layout offers at least 12 open, 3 cover and 42 free spots, well above the 2, 2 and 6 a
# specification can ask for, so in v1 the spot counts never decide a rejection on their own.
# Probed 2026-09-21 by scripts/roboquest_search_capacity.py (style 1, headless, two workers): the
# compartments are a property of the layout, the spot counts are this style with the build's own clearance.
LAYOUT_CAPACITY = {
    1: None,   # ValueError: search_room: layout 1 has no counter region free of de
    2: dict(drawers={118: 3, 158: 1}, doors={}, open=30, cover=19, free=70),
    3: dict(drawers={118: 8}, doors={240: 2, 270: 1}, open=16, cover=4, free=57),
    4: dict(drawers={118: 5}, doors={240: 1, 270: 4}, open=28, cover=13, free=78),
    5: dict(drawers={118: 14}, doors={270: 5}, open=21, cover=6, free=56),
    6: dict(drawers={118: 12}, doors={270: 6}, open=28, cover=5, free=98),
    7: dict(drawers={118: 10}, doors={270: 4}, open=48, cover=32, free=140),
    8: dict(drawers={118: 11}, doors={270: 4}, open=25, cover=9, free=112),
    9: dict(drawers={118: 10, 208: 2}, doors={240: 3, 270: 4, 337: 2, 378: 1}, open=25, cover=12, free=112),
    10: dict(drawers={118: 10, 208: 1}, doors={240: 1, 270: 5, 347: 2, 540: 1}, open=26, cover=19, free=112),
    11: dict(drawers={118: 9}, doors={270: 4, 335: 1}, open=26, cover=9, free=70),
    12: dict(drawers={118: 2, 177: 1, 194: 1}, doors={240: 2, 270: 2}, open=26, cover=17, free=88),
    13: dict(drawers={177: 4, 185: 3, 194: 2}, doors={237: 1, 240: 4}, open=18, cover=7, free=86),
    14: dict(drawers={160: 1}, doors={240: 2}, open=29, cover=12, free=84),
    15: dict(drawers={118: 4, 177: 1, 194: 1}, doors={270: 4}, open=39, cover=22, free=98),
    16: dict(drawers={118: 9, 328: 2}, doors={240: 1, 270: 4}, open=23, cover=10, free=70),
    17: dict(drawers={160: 2, 185: 10, 328: 3}, doors={240: 6}, open=36, cover=20, free=112),
    18: dict(drawers={185: 11}, doors={240: 7, 432: 1}, open=21, cover=9, free=74),
    19: dict(drawers={118: 11}, doors={270: 5}, open=18, cover=15, free=70),
    20: dict(drawers={}, doors={240: 6, 414: 2}, open=45, cover=22, free=140),
    21: dict(drawers={118: 2, 160: 3}, doors={240: 3, 270: 2}, open=27, cover=4, free=95),
    22: dict(drawers={118: 8}, doors={240: 2, 270: 6}, open=33, cover=8, free=84),
    23: dict(drawers={185: 2}, doors={240: 4}, open=34, cover=22, free=125),
    24: dict(drawers={185: 2, 328: 1}, doors={240: 2, 305: 1, 335: 1}, open=30, cover=18, free=70),
    25: dict(drawers={118: 6}, doors={240: 4}, open=42, cover=23, free=126),
    26: dict(drawers={160: 4, 185: 2, 368: 1}, doors={240: 3, 300: 1}, open=30, cover=11, free=84),
    27: dict(drawers={160: 6, 177: 1, 194: 1}, doors={240: 2, 305: 1, 343: 1}, open=30, cover=12, free=84),
    28: dict(drawers={}, doors={240: 1, 270: 1}, open=25, cover=13, free=80),
    29: dict(drawers={177: 4, 185: 1, 194: 3}, doors={240: 3}, open=36, cover=19, free=84),
    30: dict(drawers={118: 4, 160: 6}, doors={240: 4}, open=30, cover=20, free=70),
    31: dict(drawers={160: 4, 202: 5}, doors={240: 3, 277: 1, 330: 1}, open=30, cover=20, free=70),
    32: dict(drawers={160: 3, 202: 2}, doors={240: 2, 305: 1}, open=20, cover=6, free=70),
    33: dict(drawers={160: 3}, doors={305: 1}, open=39, cover=22, free=98),
    34: dict(drawers={202: 3}, doors={240: 4}, open=24, cover=15, free=62),
    35: dict(drawers={202: 2}, doors={240: 5}, open=36, cover=18, free=112),
    36: dict(drawers={160: 4}, doors={}, open=20, cover=4, free=51),
    37: dict(drawers={160: 3, 202: 1}, doors={490: 1}, open=24, cover=14, free=63),
    38: dict(drawers={160: 3, 328: 1}, doors={240: 6, 305: 1}, open=30, cover=20, free=70),
    39: dict(drawers={160: 19, 202: 2}, doors={}, open=36, cover=28, free=84),
    40: dict(drawers={160: 2}, doors={240: 3}, open=12, cover=4, free=47),
    41: dict(drawers={160: 4, 328: 1}, doors={240: 2, 305: 1}, open=30, cover=15, free=70),
    42: dict(drawers={202: 3}, doors={240: 5}, open=41, cover=17, free=112),
    43: dict(drawers={160: 3, 202: 2, 328: 1}, doors={240: 5, 270: 1, 305: 1, 350: 1}, open=32, cover=4, free=112),
    44: dict(drawers={160: 2}, doors={240: 1}, open=19, cover=4, free=63),
    45: dict(drawers={160: 2}, doors={240: 2}, open=27, cover=11, free=74),
    46: dict(drawers={160: 4, 202: 4}, doors={240: 4, 305: 1}, open=32, cover=21, free=78),
    47: dict(drawers={118: 3, 160: 2}, doors={250: 1, 267: 1}, open=27, cover=10, free=99),
    48: dict(drawers={160: 4, 202: 3}, doors={240: 1, 305: 1, 486: 1}, open=30, cover=14, free=84),
    49: dict(drawers={160: 11, 202: 2}, doors={240: 2}, open=30, cover=20, free=98),
    50: dict(drawers={160: 8, 202: 1}, doors={305: 1}, open=36, cover=19, free=98),
    51: dict(drawers={160: 8, 202: 2}, doors={240: 2}, open=42, cover=23, free=112),
    52: dict(drawers={202: 4}, doors={240: 2}, open=13, cover=3, free=57),
    53: dict(drawers={160: 4, 202: 1}, doors={240: 4}, open=27, cover=11, free=98),
    54: dict(drawers={160: 3, 202: 2}, doors={240: 3, 305: 1}, open=29, cover=16, free=84),
    55: dict(drawers={160: 4}, doors={418: 1}, open=16, cover=11, free=42),
    56: dict(drawers={160: 5, 202: 5}, doors={240: 7, 305: 1, 528: 1}, open=42, cover=33, free=112),
    57: dict(drawers={160: 8, 202: 1}, doors={240: 2, 305: 1}, open=36, cover=22, free=98),
    58: dict(drawers={160: 4, 202: 1}, doors={240: 1, 305: 1, 403: 1}, open=24, cover=13, free=56),
    59: dict(drawers={160: 5}, doors={305: 1}, open=30, cover=19, free=70),
    60: dict(drawers={160: 2, 328: 1}, doors={240: 2, 324: 1}, open=12, cover=5, free=92),
}


def opening_standoff(handle_z, kind='drawer'):
    if kind != 'drawer':
        return DOOR_STANDOFF_M
    return float(np.clip(.55 + STANDOFF_LOW_GAIN * (.80 - float(handle_z)), DRAWER_STANDOFF_M, .85))


def yaw_matrix(rot):
    c, s = math.cos(rot), math.sin(rot)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def front_normal(rot):
    """World unit vector pointing out of a RoboCasa fixture's front face."""
    return (yaw_matrix(rot) @ np.array([0., -1., 0.]))[:2]


def world_from_local(pos, rot, local):
    return np.asarray(pos, float) + yaw_matrix(rot) @ np.asarray(local, float)


def local_from_world(pos, rot, world):
    return yaw_matrix(rot).T @ (np.asarray(world, float) - np.asarray(pos, float))


def footprint_polygon(record, local_centre=(0., 0.), size_xy=None):
    """Counter-clockwise world footprint of a fixture, or of one of its local patches."""
    w, d = (float(record['size'][0]), float(record['size'][1])) if size_xy is None else size_xy
    cx, cy = local_centre
    corners = [[cx - w/2, cy - d/2, 0.], [cx + w/2, cy - d/2, 0.], [cx + w/2, cy + d/2, 0.], [cx - w/2, cy + d/2, 0.]]
    return [world_from_local(record['pos'], record['rot'], c)[:2].tolist() for c in corners]


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xs = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xs:
                inside = not inside
    return inside


def point_in_floor(point, floor_bounds, margin=0.):
    x0, x1, y0, y1 = floor_bounds
    return x0 + margin <= point[0] <= x1 - margin and y0 + margin <= point[1] <= y1 - margin


def floor_bounds_from_walls(fixtures):
    """Room bounding box from the thin (non-backing) wall fixtures."""
    xs, ys = [], []
    for name, rec in fixtures.items():
        if rec['cls'] != 'Wall' or 'backing' in name or rec.get('pos') is None:
            continue
        xs.append(float(rec['pos'][0]))
        ys.append(float(rec['pos'][1]))
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError('need at least two walls on each axis for the room bounds')
    return [min(xs), max(xs), min(ys), max(ys)]


def base_blockers(fixtures):
    """World footprints of everything a mobile base cannot drive through."""
    blockers = {}
    for name, rec in fixtures.items():
        if rec['cls'] in NON_BLOCKING_CLASSES or rec.get('pos') is None or not rec.get('size'):
            continue
        w, d, h = (float(v) for v in rec['size'][:3])
        if w < .05 or d < .05 or h <= .10:
            # Kick plates (`*_base` boxes, 5 cm tall) report unplaced positions and never block the base.
            continue
        if float(rec['pos'][2]) - h / 2 > BLOCKER_MAX_BOTTOM_Z:
            continue
        blockers[name] = footprint_polygon(rec)
    return blockers


def _blocked(point, blockers, ignore=()):
    return any(name not in ignore and point_in_polygon(point, poly) for name, poly in blockers.items())


def box_polygon(box):
    """World polygon of an axis-aligned ``{x0, x1, y0, y1}`` box (the shape ``kitchen._decor_boxes`` returns)."""
    return [[float(box['x0']), float(box['y0'])], [float(box['x1']), float(box['y0'])],
            [float(box['x1']), float(box['y1'])], [float(box['x0']), float(box['y1'])]]


def as_polygons(shapes):
    """Accept boxes (dicts) or polygons (point lists) and return polygons."""
    return [box_polygon(s) if isinstance(s, dict) else [list(p) for p in s] for s in (shapes or ())]


def resolve_keepouts(keepouts, top_z):
    """Keep-out polygons for a surface at ``top_z``.

    ``keepouts`` is either a fixed sequence of boxes/polygons or a callable ``top_z -> sequence``
    (the scene's decor reader: which decor stands on a surface depends on the surface's height).
    """
    if keepouts is None:
        return []
    if callable(keepouts):
        return as_polygons(keepouts(top_z))
    return as_polygons(keepouts)


def clear_by(point, polygons, gap=0.):
    """Is ``point`` further than ``gap`` from every polygon? ``gap=0`` means simply outside them."""
    return all(polygon_distance(point, poly) > gap for poly in polygons)


def _segment_distance(point, a, b):
    ap, ab = np.subtract(point, a), np.subtract(b, a)
    t = float(np.clip(np.dot(ap, ab) / max(float(np.dot(ab, ab)), 1e-12), 0., 1.))
    return float(np.linalg.norm(ap - t * ab))


def polygon_distance(point, polygon):
    """Planar distance from a point to a polygon; zero inside."""
    if point_in_polygon(point, polygon):
        return 0.
    return min(_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def _clearance(point, blockers, ignore=()):
    return min((polygon_distance(point, poly) for name, poly in blockers.items() if name not in ignore), default=float('inf'))


def counter_top_occupants(fixtures, top_z, band=.40):
    """Footprints of fixtures standing on a counter top at `top_z` (sinks, stoves, toasters, decor).

    Under-counter appliances and wall cabinets do not occupy the top surface, so
    only fixtures whose vertical extent crosses the band above the top count.
    Wall cabinets start about 0.47 m above the counter, hence the 0.40 m band.
    """
    occupied = {}
    for name, rec in fixtures.items():
        if rec['cls'] in NON_BLOCKING_CLASSES or rec['cls'] == 'Counter' or rec.get('pos') is None or not rec.get('size'):
            continue
        w, d, h = (float(v) for v in rec['size'][:3])
        if w < .05 or d < .05:
            continue
        bottom, top = float(rec['pos'][2]) - h / 2, float(rec['pos'][2]) + h / 2
        if top > top_z + .01 and bottom < top_z + band:
            occupied[name] = footprint_polygon(rec)
    return occupied


def interior_region(record):
    """Pick the interior region the target can rest in; None when nothing qualifies.

    Drawers expose one 'int' region. Cabinets expose level0..levelN shelves; the
    lowest shelf whose floor is inside the reach band is used.
    """
    regions = record.get('regions') or {}
    if not isinstance(regions, dict):
        return None
    kind = STORAGE_CLASSES.get(record['cls'])
    names = ['int'] if kind == 'drawer' else sorted(n for n in regions if n.startswith('level'))
    for name in names:
        region = regions.get(name)
        if not region:
            continue
        offset, size, height = region['offset'], region['size'], region['height']
        floor_z = float(record['pos'][2]) + float(offset[2])
        if not REACH_Z[0] <= floor_z <= REACH_Z[1]:
            continue
        if float(size[0]) < MIN_INTERIOR[0] or float(size[1]) < MIN_INTERIOR[1] or float(height) < MIN_INTERIOR[2]:
            continue
        centre = world_from_local(record['pos'], record['rot'], [offset[0], offset[1], offset[2]])
        return {'name': name, 'centre_world': centre.tolist(), 'floor_z': floor_z,
                'size_xy': [float(size[0]), float(size[1])], 'height': float(height),
                'offset_local': [float(v) for v in offset]}
    return None


def door_plan(record):
    """Which leaf to open and on which local-x side the handle sits (+1 = local +x)."""
    if record['cls'] == 'HingeCabinet':
        return {'open_side': 'right', 'handle_side': -1.}      # right leaf hinges at +x, handle near the seam
    if record['cls'] == 'SingleCabinet':
        orientation = record.get('orientation', 'right')     # hinge on the named side, handle opposite
        return {'open_side': '', 'handle_side': -1. if orientation == 'right' else 1.}
    return {'open_side': '', 'handle_side': 0.}


def standoff_pose(record, floor_bounds, blockers, side=1., lateral=0., distance=DRAWER_STANDOFF_M):
    """Base pose facing a fixture front, or None if it is off-floor or inside furniture."""
    normal = front_normal(record['rot']) * side
    tangent = (yaw_matrix(record['rot']) @ np.array([1., 0., 0.]))[:2]
    depth = float(record['size'][1])
    centre = np.asarray(record['pos'][:2], float) + tangent * lateral
    for distance in (distance, distance + .12, distance + .24):
        point = centre + normal * (depth / 2 + distance)
        if (point_in_floor(point, floor_bounds, FLOOR_MARGIN_M)
                and _clearance(point, blockers, ignore=(record['name'],)) >= BASE_CLEARANCE_M):
            yaw = math.atan2(-normal[1], -normal[0])   # face the fixture
            return {'xy': point.tolist(), 'yaw': float(yaw), 'distance_from_front': float(distance),
                    'side': float(side)}
    return None


def compartment_candidates(fixtures, floor_bounds=None):
    """Every drawer or door the robot could plausibly open and reach into, sorted by name."""
    if floor_bounds is None:
        floor_bounds = floor_bounds_from_walls(fixtures)
    blockers = base_blockers(fixtures)
    out = []
    for name, rec in sorted(fixtures.items()):
        kind = STORAGE_CLASSES.get(rec['cls'])
        if kind is None or rec.get('pos') is None:
            continue
        if rec.get('panel_type') in GLASS_PANELS or rec.get('is_corner') is True:
            continue
        region = interior_region(rec)
        if region is None:
            continue
        joints = list(rec.get('door_joints') or [])
        if not joints:
            continue
        plan = door_plan(rec)
        lateral = plan['handle_side'] * DOOR_LATERAL_M if kind == 'door' else 0.
        handle_z = float(rec['pos'][2])   # bar handles sit at the drawer/door centre height
        pose = standoff_pose(rec, floor_bounds, blockers, lateral=lateral, distance=opening_standoff(handle_z, kind))
        if pose is None:
            continue
        # The base reverses to pull drawers and doors: that floor must be inside the room and clear too.
        rear = np.asarray(pose['xy']) + front_normal(rec['rot']) * REAR_ROOM_M
        if not (point_in_floor(rear, floor_bounds, FLOOR_MARGIN_M)
                and _clearance(rear, blockers, ignore=(rec['name'],)) >= BASE_CLEARANCE_M):
            continue
        pose = {**pose, 'rear_room_m': REAR_ROOM_M}
        out.append({
            'id': name, 'kind': kind, 'fixture': name, 'cls': rec['cls'],
            'pos': [float(v) for v in rec['pos']], 'rot': float(rec['rot']),
            'size': [float(v) for v in rec['size']],
            'front_normal': front_normal(rec['rot']).tolist(),
            'region': region, 'joints': joints,
            'joint_ranges': {j: [float(v) for v in rec['joint_ranges'][j]] for j in joints},
            'handles': list(rec.get('handles') or []),
            'standoff': pose, 'open_side': plan['open_side'], 'handle_side': plan['handle_side'],
            'handle_z_estimate': handle_z,
        })
    return out


def hide_local(candidate):
    """Where a hidden object rests inside a compartment, in the fixture's local frame.

    Objects sit a short way behind the interior's front edge so an opened
    drawer exposes them from above and an opened door exposes them to a
    front pinch. Hinge cabinets hide the object behind the leaf that is opened.
    """
    x0, y0, z0 = candidate['region']['offset_local']
    sy = candidate['region']['size_xy'][1]
    # Drawer contents sit 0.13 m behind the interior front: after a 0.40 m pull the hand has about
    # 0.18 m to the front panel and 0.19 m to the handle of the drawer above (runs 07 and l17-04).
    # Cabinet contents sit 0.06 m behind the front: the forward pinch reaches about 0.70 m ahead
    # of a base that must stay 0.6 m from the front, or its elbow lands on the counter lip (l1-18).
    inset = .13 if candidate['kind'] == 'drawer' else .06
    dx = .14 if candidate['cls'] == 'HingeCabinet' else 0.
    return [x0 + dx, y0 - (sy / 2 - inset), z0]


def hidden_pose(candidate, object_id, rng, pool_entry):
    """World xy, support z and yaw for an object hidden in `candidate`; handled objects on a shelf
    point their handle inward so a front pinch closes on the body."""
    local = hide_local(candidate)
    world = world_from_local(candidate['pos'], candidate['rot'], local)
    yaw = float(rng.uniform(-np.pi, np.pi))
    if candidate['kind'] == 'door' and pool_entry['grasp']['handle']:
        n = candidate['front_normal']
        yaw = float(np.arctan2(-n[0], n[1]) + rng.uniform(-.26, .26))
    return {'xy': world[:2].tolist(), 'z': candidate['region']['floor_z'], 'yaw': yaw,
            'hide_local': [float(v) for v in local], 'hide_world': world.tolist()}


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def select_fillers(candidates, chosen, rng, count, min_drawers=2, min_doors=2):
    """Seeded choice of further compartments so that `count` are chosen in total, with at least
    `min_drawers` drawers and `min_doors` doors when the layout allows it."""
    chosen = {c['id']: c for c in chosen}
    if len(candidates) < count:
        raise ValueError(f'layout offers {len(candidates)} usable compartments, fewer than {count}')
    remaining = [c for c in candidates if c['id'] not in chosen]
    for kind, minimum in (('drawer', min_drawers), ('door', min_doors)):
        have = sum(1 for c in chosen.values() if c['kind'] == kind)
        pool = [c for c in remaining if c['kind'] == kind]
        for _ in range(min(minimum - have, len(pool), count - len(chosen))):
            pick = _pick(rng, pool)
            chosen[pick['id']] = pick
            pool = [c for c in pool if c['id'] != pick['id']]
            remaining = [c for c in remaining if c['id'] != pick['id']]
    while len(chosen) < count:
        pick = _pick(rng, remaining)
        chosen[pick['id']] = pick
        remaining = [c for c in remaining if c['id'] != pick['id']]
    return [chosen[k] for k in sorted(chosen)]


def layout_capacity(layout_id):
    """The probe row for ``layout_id``: a dict, ``None`` when the layout fails the build outright, and
    ``'unknown'`` when it was never probed (the caller then leaves the decision to the build)."""
    return LAYOUT_CAPACITY.get(int(layout_id), 'unknown')


def _take_fitting(pool, object_id):
    """Spend the *smallest* interior in ``pool`` ({height_mm: how many}) that fits the object; height or None.

    Smallest-fitting-first is the optimal greedy for nested height constraints, so the table only says
    "cannot host" when no assignment exists at all.
    """
    for height in sorted(pool):
        if pool[height] > 0 and fits(object_id, height / 1000.):
            pool[height] -= 1
            return height
    return None


# High drawers: per layout, the interior heights in millimetres of the usable
# drawers whose interior floor is at least ``HIGH_FLOOR_M`` above the floor, as {height: how many}. The
# PandaOmron cannot lift an object off a lower floor (bottom drawers sit at 0.32 m; base-cabinet shelves at
# 0.35-0.38 m), so the search room's drawer-type instances hide every target and decoy in one of these.
# Probed 2026-09-24 on the evaluation layouts (style 11, `compartment_candidates` floor_z; the drawer floors
# are 0.321 / 0.411 / 0.531 / 0.741 m). A layout not listed is unknown and the build decides.
HIGH_FLOOR_M = .40
HIGH_DRAWERS = {
    3: {118: 6}, 4: {118: 4}, 5: {118: 12}, 6: {118: 10}, 7: {118: 9}, 8: {118: 9},
    9: {118: 9, 208: 2}, 10: {118: 9, 208: 1},
}


def counter_front_fraction(fixtures, spot):
    """``(is_island, fraction)`` for a counter spot: where across its counter's depth it lies, measured from
    the front edge (RoboCasa counters face local -y), 0 at the front edge and 1 at the back. An island
    (``'island'`` in the counter's name) is approached from both sides."""
    rec = fixtures[spot['counter']]
    local = local_from_world(rec['pos'], rec['rot'], [spot['xy'][0], spot['xy'][1], 0.])
    depth = float(rec['size'][1])
    return 'island' in str(spot['counter']), float((local[1] + depth / 2) / depth)


def high_drawer_capacity(layout_id):
    """The capacity row restricted to drawers with a floor >= ``HIGH_FLOOR_M`` (no doors); ``'unknown'`` for
    a layout never probed, ``None`` for one that fails the build outright."""
    cap = layout_capacity(layout_id)
    if cap in (None, 'unknown') or int(layout_id) not in HIGH_DRAWERS:
        return cap if cap is None else 'unknown'
    return dict(cap, drawers=dict(HIGH_DRAWERS[int(layout_id)]), doors={})


def capacity_problems(layout_id, compartment_count, targets, hidden_distractors=(), visible=(), capacity=None):
    """Why the capacity table says this kitchen cannot host the spec; an empty list when it can.

    Checked without building anything: enough usable compartments for ``compartment_count``, a
    compartment of the right *kind* and interior height for every target that hides in one, a
    compartment for every hidden distractor, and enough counter spots for the ``open`` and ``covered``
    targets and for the visible objects. ``targets`` are the spec's ``{object, kind}`` entries.
    A layout the probe never saw passes, so the build-time check stays the safety net.
    """
    cap = layout_capacity(layout_id) if capacity is None else capacity
    if cap == 'unknown':
        return []
    if cap is None:
        return [f'layout {layout_id} has no counter region free of decor for the task frame (capacity table)']
    problems = []
    count = int(compartment_count)
    pools = {'drawer': dict(cap['drawers']), 'door': dict(cap['doors'])}
    usable = sum(pools['drawer'].values()) + sum(pools['door'].values())
    if usable < count:
        problems.append(f'layout {layout_id} offers {usable} usable compartments, fewer than '
                        f'compartment_count={count} (capacity table)')
    # Tallest demand first, the way `assign_places` now binds them, so a short object never spends the
    # only tall interior; fixed-kind targets before the distractors, which may take either kind.
    hiding = [t for t in targets if t['kind'] in ('drawer', 'door')]
    for target in sorted(hiding, key=lambda t: -POOL[t['object']]['height']):
        if _take_fitting(pools[target['kind']], target['object']) is None:
            problems.append(f"layout {layout_id} has no usable {target['kind']} that fits "
                            f"{target['object']!r} (capacity table)")
    for object_id in sorted(hidden_distractors, key=lambda o: -POOL[o]['height']):
        kinds = [k for k in ('drawer', 'door') if k in POOL[object_id]['kinds']]
        if not any(_take_fitting(pools[k], object_id) is not None for k in kinds):
            problems.append(f'layout {layout_id} has no usable compartment that fits hidden distractor '
                            f'{object_id!r} (capacity table)')
    wanted = {'open': sum(1 for t in targets if t['kind'] == 'open'),
              'cover': sum(1 for t in targets if t['kind'] == 'covered'),
              'free': len(list(visible))}
    for key, need in wanted.items():
        if need and int(cap.get(key, 0)) < need:
            problems.append(f'layout {layout_id} offers {cap.get(key)} {key} counter spots, fewer than '
                            f'the {need} the spec needs (capacity table)')
    return problems


def assign_places(candidates, spec, rng, hide_kinds=None, min_floor_z=None, hiding=None):
    """Bind the spec's abstract places to this kitchen's compartments.

    ``hide_kinds`` / ``min_floor_z`` (both ``None`` for every row minted before it) restrict
    where targets and hidden distractors may go: only compartments of those kinds whose interior floor is
    at least ``min_floor_z``. The filter only removes options, so with both ``None`` the draws are the
    ones this function always made. The empty filler compartments are not restricted.

    Targets with kind ``drawer``/``door`` get a fitting compartment of that kind;
    hidden distractors get fitting compartments of any kind; the remaining
    compartments are filled with the drawer/door mix rule; every ``open`` target
    gets a *fallback* compartment reserved among the unused ones (used only when
    no unseen counter spot exists); ``covered`` targets take no compartment. Raises ValueError with a clear message when
    the layout cannot host the spec, so a build gate can reject the layout.

    ``hiding`` (optional) is a predicate over a candidate: only compartments it admits may hide a target,
    a hidden distractor or an ``open`` target's fallback (the dark search's floor rule);
    the fillers are drawn from every candidate as before. ``None`` admits everything.
    """
    admits = hiding if hiding is not None else (lambda candidate: True)
    count = int(spec['compartment_count'])
    if len(candidates) < count:
        raise ValueError(f'layout offers {len(candidates)} usable compartments, fewer than compartment_count={count}')
    used = {}
    places = {}

    def may_hide(c):
        return ((hide_kinds is None or c['kind'] in hide_kinds)
                and (min_floor_z is None or float(c['region']['floor_z']) >= float(min_floor_z)))
    rule = '' if hide_kinds is None and min_floor_z is None else (
        f' (hiding rule: kinds {sorted(hide_kinds) if hide_kinds else "any"}, floor >= {min_floor_z} m)')
    # Tallest object first: a mug that takes the only tall drawer leaves nothing for the bottle behind it,
    # and the mint-time capacity table assumes this order (`capacity_problems`). The place keys keep the
    # spec's own indices, so nothing downstream depends on the binding order.
    hiding = [(i, t) for i, t in enumerate(spec['targets']) if t['kind'] not in ('open', 'covered')]
    for index, target in sorted(hiding, key=lambda it: -POOL[it[1]['object']]['height']):
        obj, kind = target['object'], target['kind']
        options = [c for c in candidates if c['id'] not in used and c['kind'] == kind and fits_compartment(obj, c)
                   and may_hide(c) and admits(c)]
        if not options:
            raise ValueError(f'layout has no unused {kind} compartment that fits target {obj!r}{rule}')
        pick = _pick(rng, options)
        used[pick['id']] = pick
        places[f'target_{index}'] = dict(role='target', object=obj, kind=kind, compartment=pick['id'])
    for index, obj in sorted(enumerate(spec['hidden_distractors']), key=lambda it: -POOL[it[1]]['height']):
        options = [c for c in candidates if c['id'] not in used and fits_compartment(obj, c) and may_hide(c)
                   and admits(c)]
        if not options:
            raise ValueError(f'layout has no unused compartment that fits hidden distractor {obj!r}{rule}')
        pick = _pick(rng, options)
        used[pick['id']] = pick
        places[f'distractor_{index}'] = dict(role='hidden_distractor', object=obj, kind=pick['kind'], compartment=pick['id'])
    if len(used) > count:
        raise ValueError(f'spec needs {len(used)} hidden compartments but compartment_count={count}')
    chosen = select_fillers(candidates, list(used.values()), rng, count)
    chosen_ids = {c['id']: c for c in chosen}
    spare = [c for c in chosen if c['id'] not in used]
    for index, target in enumerate(spec['targets']):
        if target['kind'] == 'covered':
            # An open counter spot under a cover, chosen by the scene at build time; no compartment fallback
            # (a layout without a suitable spot fails the build gate).
            places[f'target_{index}'] = dict(role='target', object=target['object'], kind='covered', compartment=None)
            continue
        if target['kind'] != 'open':
            continue
        obj = target['object']
        fallback = [c for c in spare if fits_compartment(obj, c) and admits(c)]
        if not fallback:
            # Swap one spare filler for an unchosen compartment that fits, when one exists.
            outside = [c for c in candidates if c['id'] not in chosen_ids and fits_compartment(obj, c) and admits(c)]
            if outside and spare:
                swap_in, swap_out = _pick(rng, outside), _pick(rng, spare)
                chosen_ids.pop(swap_out['id'])
                chosen_ids[swap_in['id']] = swap_in
                spare = [c for c in spare if c['id'] != swap_out['id']] + [swap_in]
                fallback = [swap_in]
        pick = _pick(rng, fallback) if fallback else None
        if pick is not None:
            spare = [c for c in spare if c['id'] != pick['id']]
        places[f'target_{index}'] = dict(role='target', object=obj, kind='open', compartment=None,
                                         fallback_compartment=None if pick is None else pick['id'])
    chosen = [chosen_ids[k] for k in sorted(chosen_ids)]
    kinds = [c['kind'] for c in chosen]
    return dict(compartments={c['id']: c for c in chosen}, places=places,
                mix=dict(drawers=kinds.count('drawer'), doors=kinds.count('door')))


def counter_patches(fixtures):
    """Every counter-top patch (RoboCasa 'geom_i' region) with its world polygon, largest first."""
    patches = []
    for name, rec in fixtures.items():
        if rec['cls'] != 'Counter' or rec.get('pos') is None or not isinstance(rec.get('regions'), dict):
            continue
        for region_name, region in rec['regions'].items():
            offset, size = region['offset'], region['size']
            patches.append({
                'counter': name, 'region': region_name, 'is_island': 'island' in name,
                'offset_local': [float(v) for v in offset], 'size_xy': [float(size[0]), float(size[1])],
                'top_z': float(rec['pos'][2]) + float(rec['size'][2]) / 2,
                'polygon': footprint_polygon(rec, local_centre=(float(offset[0]), float(offset[1])),
                                             size_xy=(float(size[0]), float(size[1]))),
                'area': float(size[0]) * float(size[1]),
            })
    patches.sort(key=lambda p: (not p['is_island'], -p['area'], p['counter'], p['region']))
    return patches


def open_shelves(fixtures):
    """Open-cabinet shelves low enough for the arm, as patch-like records (surface = shelf top)."""
    shelves = []
    for name, rec in fixtures.items():
        if rec['cls'] != 'OpenCabinet' or rec.get('pos') is None or not isinstance(rec.get('regions'), dict):
            continue
        for region_name, region in rec['regions'].items():
            offset, size = region['offset'], region['size']
            top_z = float(rec['pos'][2]) + float(offset[2])
            if top_z > OPEN_SHELF_MAX_Z or top_z < REACH_Z[0]:
                continue
            shelves.append({
                'counter': name, 'region': region_name, 'is_island': False, 'is_shelf': True,
                'offset_local': [float(offset[0]), float(offset[1]), 0.], 'size_xy': [float(size[0]), float(size[1])],
                'top_z': top_z,
                'polygon': footprint_polygon(rec, local_centre=(float(offset[0]), float(offset[1])),
                                             size_xy=(float(size[0]), float(size[1]))),
                'area': float(size[0]) * float(size[1]),
            })
    return shelves


def spot_standoff(spot_xy, patch, fixtures, floor_bounds, blockers):
    """Base pose facing a counter spot from the nearest approachable edge, or None.

    The spot must lie within SPOT_EDGE_BAND_M of that edge so a top-down grasp
    0.47 m ahead of the base frame reaches it after a short crawl.
    """
    rec = fixtures[patch['counter']]
    ox, oy = patch['offset_local'][0], patch['offset_local'][1]
    sx, sy = patch['size_xy']
    local = local_from_world(rec['pos'], rec['rot'], [spot_xy[0], spot_xy[1], 0.])
    lx, ly = float(local[0]) - ox, float(local[1]) - oy
    if abs(lx) > sx / 2 or abs(ly) > sy / 2:
        return None
    options = []
    for side, depth in ((1., ly + sy / 2), (-1., sy / 2 - ly)):
        if not SPOT_EDGE_BAND_M[0] - .02 <= depth <= SPOT_EDGE_BAND_M[1] + .02:
            continue
        normal = front_normal(rec['rot']) * side
        for distance in (SPOT_STANDOFF_M, SPOT_STANDOFF_M + .10):
            point = np.asarray(spot_xy, float) + normal * (depth + distance)
            if (point_in_floor(point, floor_bounds, FLOOR_MARGIN_M)
                    and _clearance(point, blockers, ignore=(patch['counter'],)) >= BASE_CLEARANCE_M
                    and not _blocked(point, blockers)):
                yaw = math.atan2(-normal[1], -normal[0])
                options.append({'xy': point.tolist(), 'yaw': float(yaw), 'distance_from_edge': float(distance),
                                'edge_depth': float(depth), 'side': float(side)})
                break
    return options[0] if options else None


def sample_counter_spots(fixtures, floor_bounds, rng, per_patch=6, include_shelves=True, exclude=(), edge_only=True,
                         keepouts=None, clearance=0.):
    """Candidate resting spots on counter patches (and low open shelves). With ``edge_only`` (hiding
    places the robot must reach) spots sit near an approachable edge and carry a base standoff;
    without it (visible distractors) any interior point of a patch qualifies. Spots avoid counter-top
    occupants (appliances, decor) and the polygons in `exclude`. Shuffled by `rng`.

    ``keepouts`` carries the kitchen's *real* decor and appliance footprints (boxes or polygons, or a
    callable ``top_z -> boxes``; the scene reads them the way ``observe.decor_shapes_on_work`` does).
    The fixture records alone under-report them: an accessory's ``size`` is not its bounding box, so
    plants, paper-towel holders, toasters and coffee machines slip through ``counter_top_occupants``.
    A spot must keep ``clearance`` (the object's radius plus a margin) from every keep-out and from
    every counter-top occupant, so the object standing there does not touch them."""
    blockers = base_blockers(fixtures)
    surfaces = counter_patches(fixtures) + (open_shelves(fixtures) if include_shelves else [])
    gap = float(clearance)
    spots = []
    for patch in surfaces:
        rec = fixtures[patch['counter']]
        sx, sy = patch['size_xy']
        if sx < .30 or sy < .30:
            continue
        ox, oy = patch['offset_local'][0], patch['offset_local'][1]
        occupied = list(counter_top_occupants(fixtures, patch['top_z']).values())
        occupied += resolve_keepouts(keepouts, patch['top_z'])
        tries = 0
        found = 0
        while tries < per_patch * 4 and found < per_patch:
            tries += 1
            side = 1. if rng.uniform() < .5 else -1.
            if edge_only:
                depth = rng.uniform(*SPOT_EDGE_BAND_M)
                ly = -side * (sy / 2 - depth)
            else:
                ly = rng.uniform(-sy / 2 + .10, sy / 2 - .10)
                depth = min(ly + sy / 2, sy / 2 - ly)
            lx = rng.uniform(-sx / 2 + .12, sx / 2 - .12)
            point = world_from_local(rec['pos'], rec['rot'], [ox + lx, oy + ly, 0.])[:2]
            if not clear_by(point, occupied, gap) or any(point_in_polygon(point, poly) for poly in exclude):
                continue
            if patch.get('is_shelf') or not edge_only:
                standoff = None
            else:
                standoff = spot_standoff(point, patch, fixtures, floor_bounds, blockers)
                if standoff is None:
                    continue
            spots.append({'counter': patch['counter'], 'region': patch['region'], 'xy': point.tolist(),
                          'top_z': patch['top_z'], 'standoff': standoff, 'is_shelf': bool(patch.get('is_shelf')),
                          'edge_depth': float(depth)})
            found += 1
    order = rng.permutation(len(spots))
    return [spots[i] for i in order]


def counter_band(spot, patch, fixtures, fraction):
    """Where a counter spot sits across its counter's depth, measured from the edge it is approached from.

    RoboCasa counters are rectangles whose local y is the depth (local -y is the front face). A spot
    sampled with ``edge_only`` carries a base standoff on one side (``side`` +1: the front edge, -1: the
    back edge); the distance from that edge is ``from_front_m``. The counter rule: on a
    counter whose back is against a wall, objects go only in the front ``fraction`` (two thirds) of the
    depth measured from the front edge; on an island, approachable from both sides, anywhere across the
    depth is fine. Returns the record the spec and the render gate carry, with ``ok``.
    """
    rec = fixtures[patch['counter']]
    ox, oy = float(patch['offset_local'][0]), float(patch['offset_local'][1])
    sy = float(patch['size_xy'][1])
    local = local_from_world(rec['pos'], rec['rot'], [float(spot['xy'][0]), float(spot['xy'][1]), 0.])
    ly = float(local[1]) - oy
    side = float((spot.get('standoff') or {}).get('side', 1.))
    from_front = ly + sy / 2 if side >= 0 else sy / 2 - ly
    island = bool(patch.get('is_island'))
    limit = float(fraction) * sy
    return dict(counter=patch['counter'], region=patch['region'], island=island, depth_m=round(sy, 4),
                from_front_m=round(float(from_front), 4), limit_m=round(limit, 4), fraction=float(fraction),
                approach_side=side, ok=bool(island or from_front <= limit + 1e-6))


def patch_for_spot(fixtures, spot):
    """The counter patch (or shelf) record a sampled spot came from, or None."""
    for patch in counter_patches(fixtures) + open_shelves(fixtures):
        if patch['counter'] == spot.get('counter') and patch['region'] == spot.get('region'):
            return patch
    return None


def disc_clear(centre, patch, fixtures, radius, exclude=(), margin=.02, keepouts=None):
    """Does a disc of ``radius`` (plus ``margin``) at ``centre`` lie inside the counter ``patch``, clear of the
    counter-top occupants (appliances, decor), the ``keepouts`` (the kitchen's real decor boxes, see
    :func:`sample_counter_spots`) and of the ``exclude`` polygons? Eight rim points and the centre, and the
    rim also keeps ``margin`` from every occupant and keep-out."""
    occupied = list(counter_top_occupants(fixtures, patch['top_z']).values())
    occupied += resolve_keepouts(keepouts, patch['top_z'])
    reach = float(radius) + float(margin)
    centre = np.asarray(centre, float)
    points = [centre] + [centre + reach * np.array([math.cos(a), math.sin(a)]) for a in np.arange(0., 2 * math.pi, math.pi / 4)]
    if not all(point_in_polygon(p, patch['polygon']) for p in points):
        return False
    if not clear_by(centre, occupied, reach):
        return False
    return not any(point_in_polygon(p, poly) for p in points for poly in exclude)


def cover_fits_spot(spot, fixtures, radius, exclude=(), margin=.02, keepouts=None):
    """A cover of ``radius`` standing on ``spot`` stays on its counter patch and clear of appliances and decor
    (the spot itself was sampled as a point)."""
    patch = patch_for_spot(fixtures, spot)
    if patch is None or patch.get('is_shelf'):
        return False
    return disc_clear(spot['xy'], patch, fixtures, radius, exclude=exclude, margin=margin, keepouts=keepouts)


def cover_setdown_options(spot, fixtures, radius, exclude=(), offsets=(.30, .36, .42), margin=.02, keepouts=None):
    """Where a cover lifted off ``spot`` can be put down: points along the counter front, either side of the
    spot, whose disc of ``radius`` (plus ``margin``) lies inside the spot's counter patch, clear of counter-top
    occupants (appliances, decor, the ``keepouts``) and of the ``exclude`` polygons (the task frame). Nearest
    first per side, sides interleaved; empty when the patch is too small."""
    patch = patch_for_spot(fixtures, spot)
    if patch is None or patch.get('is_shelf'):
        return []
    rec = fixtures[patch['counter']]
    tangent = (yaw_matrix(rec['rot']) @ np.array([1., 0., 0.]))[:2]
    options = []
    for offset in offsets:
        for side in (1., -1.):
            centre = np.asarray(spot['xy'], float) + side * offset * tangent
            if disc_clear(centre, patch, fixtures, radius, exclude=exclude, margin=margin, keepouts=keepouts):
                options.append({'xy': centre.tolist(), 'side': side, 'offset_m': float(offset), 'top_z': patch['top_z']})
    return options


def as_record(name, cls, pos, rot, size, **extra):
    """Helper for tests and the scene: build one plain fixture record."""
    record = {'name': name, 'cls': cls, 'pos': [float(v) for v in pos], 'rot': float(rot),
              'size': [float(v) for v in size]}
    record.update(deepcopy(extra))
    return record


def records_from_probe(layout_info):
    """Adapt the development probe JSON (tests/fixtures/room_search_probe_style4.json) into fixture records."""
    fixtures = {}
    for name, entry in layout_info['fixtures'].items():
        rec = {'name': name, 'cls': entry['cls'], 'pos': entry.get('pos'), 'rot': float(entry.get('rot', 0.)),
               'size': entry.get('size')}
        if entry.get('regions') is not None and isinstance(entry['regions'], dict):
            rec['regions'] = entry['regions']
        if entry.get('panel_type') is not None:
            rec['panel_type'] = entry['panel_type']
        if entry.get('is_corner') is not None:
            rec['is_corner'] = entry['is_corner']
        if entry.get('orientation') is not None:
            rec['orientation'] = entry['orientation']
        if entry.get('door_joints'):
            rec['door_joints'] = list(entry['door_joints'])
            rec['joint_ranges'] = {j: entry['joint_infos'][j]['range'] for j in entry['door_joints']}
        handles = [entry[k] for k in ('handle_name', 'left_handle_name', 'right_handle_name') if entry.get(k)]
        if handles:
            rec['handles'] = handles
        fixtures[name] = rec
    return fixtures
