"""Observability machinery for RoboQuest v1 (spec 2.2, build brief contract C6).

Everything here is a function taking the task env; task modules call them from
``build_task`` (wave 2 wires each task) and the G3 gate calls the render check
after reset. Nothing here touches the policy observation.

Three levels (``spec['observability']``): ``visible`` (every task object in a
policy camera), ``look`` (an object outside every camera at reset but visible
from a reachable viewpoint without moving anything: an **annex** region on the
work surface at least 0.8 m along the counter from the frame or on another
surface, or behind a tall opaque **occluder**), ``uncover`` (a task object under
a **cover** that must be lifted, slid or poured, or a place under a flat cover
that must be pushed aside). ``look+uncover`` is dev-only.

Frames
------
Regions and stances are world xy; each region also carries its fixture-local
rectangle. Keep-outs and bounds that a task passes in are task-frame shapes
(:mod:`scatter` conventions: x along the front of the work surface, y away
from the robot) because that is how tasks describe their own props. Places for
flat covers are task-frame xy.

Reach
-----
The PandaOmron base is a 0.70 x 0.70 m rectangle about its base frame (0.15 m
ahead, 0.55 m behind, 0.35 m to each side); a stance passes when that rectangle
overlaps no fixture footprint and stays inside the room floor. RoboCasa spawns
the base 0.20 m beyond the surface edge; an annex stance uses the same
standoff, so a region centre within :data:`REGION_REACH_M` of the stance is
workable. The top-down grasp envelope from a stance is :data:`GRASP_REACH_M`
ahead of the base frame (motor's nominal anchor is 0.47 m) and the base may
slide sideways along the counter to centre a grasp.

Cameras at build time
---------------------
The three policy cameras are robot-mounted, so their reset poses are fixed
offsets from the base frame (:data:`CAMERA_MOUNTS`, measured on three kitchens
with ``robot_spawn_deviation_* = 0`` and no initialisation noise). At build
time the base pose is RoboCasa's anchor for the work fixture shifted along the
frame, exactly what ``RoboQuestKitchen._align_robot_base_with_frame`` does, so
the frusta can be predicted before the model is compiled. The geometric checks
choose placements; :func:`visible_in_cameras` (render diff, opaque colour
swap) is the truth at reset and fails closed.

Covers
------
The cloches and the wide cup come from :mod:`roboquest.assets`
(contract C4): free-joint bodies appended to the worldbody at their world pose
and registered in ``env._frame_bodies`` so ``prop_clashes()`` and the settle
gate see them. The inverted bowl is a RoboCasa bowl at a fit-derived scale;
flat covers are RoboCasa plates, cutting boards and trays.
"""
from copy import deepcopy
import math

import numpy as np

from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.utils import env_utils as EnvUtils

from roboquest.harness.contract import CAMERAS
from roboquest import headroom as H
from roboquest import scatter as S
from roboquest.assets.cloche import KNOB_HEIGHT as CLOCHE_KNOB_HEIGHT, SIZES as CLOCHE_SIZES, WALL as CLOCHE_WALL, cloche as build_cloche
from roboquest.assets.cup import FLOOR as CUP_FLOOR, HEIGHT as CUP_HEIGHT, INNER_RADIUS as CUP_INNER_RADIUS, wide_cup as build_wide_cup
from roboquest.kitchen import yaw_matrix, yaw_quat_wxyz
from roboquest.search.layout import base_blockers, floor_bounds_from_walls
from roboquest.search.visibility import camera_frusta, point_in_frustum

__all__ = ['MODES', 'DEV_ONLY_LEVEL', 'HIDE_MODES', 'OBJECT_COVERS', 'PLACE_COVERS', 'LEVEL_MODES',
           'annex_regions', 'place_in_annex', 'occluder_candidates', 'place_occluder', 'cover_object',
           'cover_place', 'cover_feasible', 'cover_footprint_clear', 'push_gate', 'reach_gate', 'stance_gate',
           'slide_gate', 'visible_in_cameras', 'hidden_check', 'frames_agree', 'body_id', 'annex_regions_from',
           'look_reachable', 'spot_reachable', 'draw_hidden', 'hidden_entry', 'validate_hidden', 'apply_observability',
           'expected_base_pose', 'expected_camera_frusta', 'floor_map', 'surface_regions', 'body_extent',
           'set_body_pose', 'cover_headroom', 'fixture_aabb', 'enclosing_boxes', 'rays_clear']

# ---- schema ---------------------------------------------------------------------------------------
MODES = ('visible', 'look', 'uncover')
DEV_ONLY_LEVEL = 'look+uncover'
HIDE_MODES = ('annex', 'occluder', 'cover')          # spec['hidden'][i]['mode']
OBJECT_COVERS = ('cloche_small', 'cloche_large', 'bowl', 'wide_cup')
PLACE_COVERS = ('plate', 'board', 'tray')
COVER_KINDS = OBJECT_COVERS + PLACE_COVERS
LEVEL_MODES = {'visible': (), 'look': ('annex', 'occluder'), 'uncover': ('cover',),
               DEV_ONLY_LEVEL: ('annex', 'occluder', 'cover')}
OBSERVE_SALT = 0x0B5E                                # second word of the build-time RNG seed

# ---- base, stances and reach (metres) ---------------------------------------------------------------
BASE_AHEAD_M, BASE_BEHIND_M, BASE_HALF_WIDTH_M = .15, .55, .35   # the 0.70 x 0.70 m base rectangle
STANDOFF_M = .20              # base frame beyond the surface edge at a stance (RoboCasa's spawn distance)
STANCE_STEPS_M = (0., .15, -.15, .30, -.30, .45, -.45)   # stances tried along an accessible edge, nearest first
REGION_REACH_M = .75          # region centre within this horizontal distance of its stance
GRASP_REACH_M = .70           # top-down grasp reach ahead of the base frame at counter height (nominal .47)
REACH_MIN_AHEAD_M = .10       # the arm pinches from above down to here ahead of the base frame (a rim overhanging
                              # the counter edge toward the robot stands above the base's own front)
MAX_SLIDE_PATH_M = .60        # a bowl is slid to a counter edge within one arm sweep from its stance
LATERAL_REACH_M = .25         # residual lateral offset the arm absorbs after the base has slid
MAX_SLIDE_M = 1.20            # how far the base may slide along the counter from its reset stance
FLOOR_MARGIN_M = .05          # the base rectangle stays this far inside the room bounds
STANCE_TOUCH_M = .02          # the base rectangle may touch a fixture by this much (its front at the edge)
ANNEX_MIN_ALONG_M = .80       # annex on the work surface: at least this far along the counter from the frame centre
ANNEX_REACH_TRIES = 8         # spots of one annex cell that may fail the spot reach check before the next cell
SURFACE_Z_BAND = (.60, 1.10)  # top heights a surface may have (dining tables to tall counters)
MIN_REGION_M = .25            # a region must hold at least this square
CELL_M = .50                  # regions are chunked along their accessible edge into cells of this length
DEPTH_MAX_M = REGION_REACH_M - STANDOFF_M   # usable depth of a region from its accessible edge
CAMERA_MARGIN_RAD = math.radians(3.)        # frustum margin for the geometric hidden test
OCCLUDER_GAP_M = .01          # clearance between an occluder and the props around it
OCCLUDER_ROOM_RISE_M = .01    # a fixture whose box reaches this far above a top stands in what is put there
OCCLUDER_INSET = .85          # fraction of an occluder's bounding box trusted to block a ray
COVER_CLEARANCE_M = .01       # object to cover interior
COVER_EDGE_MARGIN_M = .01     # a pushed or slid cover / object stays this far inside the surface
FLAT_COVER_MARGIN_M = .012    # a flat cover overhangs the place it hides by at least this much
FLAT_COVER_MAX_SCALE = 1.7

EDGES = {'front': (0., -1.), 'back': (0., 1.), 'left': (-1., 0.), 'right': (1., 0.)}   # outward normals, fixture local

# Camera mounts relative to the base frame at reset (pos in metres, rotation world-from-camera columns,
# fovy in degrees); measured on suite/v0 stamps (layout 52), marked_mugs (24) and odd_parcel (25).
CAMERA_MOUNTS = {
    'robot0_agentview_left': dict(pos=(-.5, .35, 1.75), fovy=60.,
                                  mat=((-.20197, .52813, -.82479), (-.9793, -.09726, .17753), (.01354, .84358, .53684))),
    'robot0_agentview_right': dict(pos=(-.5, -.35, 1.75), fovy=60.,
                                   mat=((.20197, .52813, -.82479), (-.9793, .09726, -.17753), (-.01354, .84358, .53684))),
    'robot0_eye_in_hand': dict(pos=(.2686, -.0063, 1.403), fovy=75.,
                               mat=((.07087, .95933, -.27324), (-.99694, .05904, -.05130), (-.03308, .27603, .96058))),
}

# ---- assets ---------------------------------------------------------------------------------------
# Occluders: RoboCasa objects at a fixed scale that brings them to 0.21-0.23 m (none of the cereal boxes,
# boxed food, jugs or bottles reaches 0.20 m at native scale: cereal tops out at 0.17 m, jugs at 0.12 m).
# Every entry rendered leak-free in a standalone calibration (a 5.2 cm cube 6 cm behind the occluder,
# two agentview-like cameras, opaque colour swap of the cube: zero differing pixels); the objaverse jugs,
# kettles and water bottles leak through handles, spouts and transparent walls and are left out.
# ``toward``: the occluder's local axis that must point from the cameras toward the hidden object.
OCCLUDERS = [
    *[dict(id=f'cereal_{i}', kind='cereal', asset=f'objaverse/cereal/cereal_{i}', scale=s, toward=t)
      for i, s, t in ((0, 1.55, 'x'), (1, 1.5, 'y'), (2, 1.3, 'x'), (3, 1.45, 'x'), (4, 1.65, 'x'), (5, 1.4, 'x'),
                      (6, 1.5, 'x'), (7, 1.45, 'x'), (8, 1.4, 'x'), (9, 1.45, 'x'), (10, 1.4, 'x'), (11, 1.5, 'x'),
                      (12, 1.45, 'x'), (13, 1.6, 'x'))],
    *[dict(id=f'cereal_a{i}', kind='cereal', asset=f'aigen_objs/cereal/cereal_{i}', scale=1.7, toward=t)
      for i, t in ((0, 'y'), (1, 'x'), (2, 'x'), (3, 'y'), (4, 'x'), (5, 'x'), (6, 'x'), (7, 'x'), (8, 'x'))],
    *[dict(id=f'boxed_food_{i}', kind='boxed_food', asset=f'objaverse/boxed_food/boxed_food_{i}', scale=s, toward='x')
      for i, s in ((1, 1.7), (2, 1.5), (4, 1.7), (6, 1.7), (8, 1.75), (9, 1.7), (10, 1.7), (11, 2.0))],
    *[dict(id=f'boxed_food_a{i}', kind='boxed_food', asset=f'aigen_objs/boxed_food/boxed_food_{i}', scale=2.2, toward='y')
      for i in (0, 1, 2, 3, 4, 5)],
    *[dict(id=f'boxed_drink_{i}', kind='boxed_food', asset=f'objaverse/boxed_drink/boxed_drink_{i}', scale=s, toward='y')
      for i, s in ((1, 1.75), (2, 1.65), (4, 1.75))],
    *[dict(id=f'boxed_drink_a{i}', kind='boxed_food', asset=f'aigen_objs/boxed_drink/boxed_drink_{i}', scale=2.2, toward='y')
      for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14)],
    *[dict(id=f'jug_a{i}', kind='jug', asset=f'aigen_objs/jug/jug_{i}', scale=2.2, toward='y') for i in (0, 3, 4, 5, 7)],
    *[dict(id=f'kettle_a{i}', kind='kettle', asset=f'aigen_objs/kettle/kettle_{i}', scale=s, toward='y')
      for i, s in ((0, 2.0), (1, 2.2), (2, 2.0), (3, 2.0), (4, 2.2), (8, 1.85), (10, 2.2), (11, 1.85))],
    *[dict(id=f'bottle_{n}', kind='bottle', asset=a, scale=s, toward=t)
      for n, a, s, t in (('milk_a2', 'aigen_objs/milk/milk_2', 2.2, 'y'), ('milk_a3', 'aigen_objs/milk/milk_3', 2.0, 'y'),
                         ('milk_a5', 'aigen_objs/milk/milk_5', 1.85, 'y'), ('milk_9', 'objaverse/milk/milk_9', 1.35, 'x'),
                         ('condiment_a0', 'aigen_objs/condiment/condiment_0', 2.0, 'y'),
                         ('condiment_a1', 'aigen_objs/condiment/condiment_1', 2.0, 'y'),
                         ('condiment_a9', 'aigen_objs/condiment/condiment_9', 2.0, 'y'),
                         ('ketchup_a7', 'aigen_objs/ketchup/ketchup_7', 1.9, 'y'))],
]
OCCLUDER_MIN_HEIGHT_M = .20
OCCLUDER_MAX_THIN_M = .074    # widest thinnest side ``uncover.move_aside`` can pinch: the Panda's 80 mm
                              # opening less the margin it keeps (``uncover.GRIPPER_OPENING_M`` - .006)
OCCLUDER_PINCHABLE_MINTS = False   # whether a placement draws from the pinchable pool only. The occluder is
                              # chosen when the scene is built, not recorded in the instance, so narrowing the
                              # pool re-rolls the occluder of every instance already minted (measured: stamps
                              # 5b7508dd74b85e47 rebuilt with a 50 mm cereal box in place of its 166 mm jug).
                              # It therefore ships off and turns on with the next generator version.
IDEAL_OCCLUDER = (.23, .10, .225)      # width, depth, height of the largest catalogue entry (the pruning slab)

# Flat covers (RoboCasa assets, full sizes at scale 1 from their bounding boxes).
FLAT_COVERS = {
    'plate': [dict(asset=f'objaverse/plate/plate_{i}', size=s) for i, s in
              ((1, (.1818, .1818, .0111)), (7, (.1831, .182, .0096)), (8, (.1837, .1818, .0123)),
               (10, (.1813, .1813, .0172)), (11, (.1825, .1825, .0117)), (12, (.1815, .1815, .0199)),
               (14, (.183, .1818, .015)), (19, (.183, .183, .0159)))],
    'board': [dict(asset=f'objaverse/cutting_board/cutting_board_{i}', size=s) for i, s in
              ((2, (.1557, .25, .0188)), (5, (.1462, .26, .0091)), (7, (.1564, .2465, .0109)),
               (9, (.16, .23, .01)), (11, (.1526, .2626, .0112)), (12, (.156, .2093, .0101)),
               (15, (.1875, .25, .0125)), (16, (.1429, .25, .0143)))],
    'tray': [dict(asset=f'objaverse/tray/tray_{i}', size=s) for i, s in
             ((1, (.1478, .2081, .018)), (3, (.1484, .21, .0114)), (5, (.1333, .2, .0195)),
              (7, (.1398, .2096, .012)), (9, (.1255, .2225, .0117)), (10, (.1623, .21, .0084)),
              (11, (.1428, .21, .0223)), (12, (.1492, .2081, .0211)))],
}

# Inverted RoboCasa bowls as object covers. Interior depth profiles (radius, depth below the rim) at scale
# 1 were measured by dropping a cube and small balls into the upright bowl in standalone MuJoCo (every
# RoboCasa bowl is hollow: 16-32 convex collision pieces, cube 18-42 mm below the rim; none is
# convex-hulled). These two have the deepest, flattest interiors.
BOWL_COVERS = [
    dict(asset='objaverse/bowl/bowl_8', radius=.060, profile=((0., .0423), (.033, .0436), (.036, .0443), (.041, .0316), (.060, 0.))),
    dict(asset='objaverse/bowl/bowl_7', radius=.065, profile=((0., .0370), (.0325, .0382), (.0353, .0372), (.0368, .0307), (.065, 0.))),
]
BOWL_SCALES = tuple(round(1.2 + .05 * k, 2) for k in range(29))   # 1.2 .. 2.6
BOWL_INNER_FRACTION = .90     # inner radius at the rim as a fraction of the outer radius

# Cloches (assets.cloche): small 16 cm across with 10 cm of headroom, large 22 cm with 13 cm; the wide cup
# (assets.cup) is 84 mm inside and 110 mm tall. The cup was designed around the 52 mm cube, so its
# clearance is tighter than a cloche's.
CUP_CLEARANCE_M = .001
COVER_LIFT_M = .001           # covers start this far above the counter so nothing interpenetrates at reset
FLICKER_TOLERANCE_PX = 32     # two renders of the same state may differ in this many pixels (GPU flicker) and
                              # still count as deterministic; a hidden body needs pixels in both colour passes anyway


# ---- small geometry ---------------------------------------------------------------------------------
def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _rot2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def _quat_xyzw(yaw):
    return [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]


def _yaw_of_wxyz(q):
    return _wrap(2 * math.atan2(float(q[3]), float(q[0])))


def _rect_corners(center, half, yaw):
    R = _rot2(yaw)
    c = np.asarray(center, float)
    return np.array([c + R @ (sx * half[0], sy * half[1]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _polys_overlap(a, b, margin=0.):
    """Separating-axis test on two convex polygons; ``margin`` > 0 inflates them, < 0 tolerates overlap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for edge in edges:
            axis = np.array([-edge[1], edge[0]])
            length = float(np.linalg.norm(axis))
            if length < 1e-12:
                continue
            axis = axis / length
            pa, pb = a @ axis, b @ axis
            if pa.max() + margin <= pb.min() or pb.max() + margin <= pa.min():
                return False
    return True


def _corners_inside(corners, bounds, margin=0.):
    x0, x1, y0, y1 = bounds
    corners = np.asarray(corners, float)
    return bool(np.all(corners[:, 0] >= x0 + margin) and np.all(corners[:, 0] <= x1 - margin)
                and np.all(corners[:, 1] >= y0 + margin) and np.all(corners[:, 1] <= y1 - margin))


def box_points(center_xy, half_xy, z0, z1, yaw=0., inflate=0.):
    """The eight corners of an upright box, for frustum and occlusion tests."""
    half = (float(half_xy[0]) + inflate, float(half_xy[1]) + inflate)
    corners = _rect_corners(center_xy, half, yaw)
    return [np.array([x, y, z]) for x, y in corners for z in (float(z0) - inflate, float(z1) + inflate)]


def seen_by(frusta, points, margin_rad=CAMERA_MARGIN_RAD):
    """Per camera: does any of the points fall inside its frustum (plus a margin)?"""
    return {f['name']: bool(any(point_in_frustum(f, p, margin_rad) for p in points)) for f in frusta}


def hidden_from_all(frusta, points, margin_rad=CAMERA_MARGIN_RAD):
    return not any(seen_by(frusta, points, margin_rad).values())


def _segment_hits_box(a, b, center, half, yaw):
    """Does the segment a-b pass through the upright oriented box (slab test in the box frame)?"""
    R = yaw_matrix(yaw)
    la = R.T @ (np.asarray(a, float) - np.asarray(center, float))
    lb = R.T @ (np.asarray(b, float) - np.asarray(center, float))
    d = lb - la
    t0, t1 = 0., 1.
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if abs(la[i]) > half[i]:
                return False
            continue
        ta, tb = (-half[i] - la[i]) / d[i], (half[i] - la[i]) / d[i]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return True


def rays_clear(frusta, target_points, center, half, yaw, margin_rad=CAMERA_MARGIN_RAD):
    """Per camera: how many of ``target_points`` a ray from the camera reaches without crossing the box, and
    whether that camera needs the target at all. :func:`occludes` is this with the count read as a flag (a
    camera is blocked when no ray gets through); the count itself says how far from hidden the target is."""
    needed = seen_by(frusta, target_points, margin_rad)
    clear = {f['name']: sum(not _segment_hits_box(f['pos'], p, center, half, yaw) for p in target_points)
             for f in frusta}
    return clear, needed


def occludes(frusta, target_points, center, half, yaw, margin_rad=CAMERA_MARGIN_RAD):
    """Per camera: the target is outside the frustum, or every ray from the camera to a target corner
    crosses the box. Returns (blocked per camera, needed per camera)."""
    needed = seen_by(frusta, target_points, margin_rad)
    blocked = {}
    for f in frusta:
        if not needed[f['name']]:
            blocked[f['name']] = True
            continue
        blocked[f['name']] = bool(all(_segment_hits_box(f['pos'], p, center, half, yaw) for p in target_points))
    return blocked, needed


# ---- fixtures, floor, base --------------------------------------------------------------------------
def fixture_records(env):
    records = {}
    for name, fixture in env.fixtures.items():
        try:
            pos = [float(v) for v in fixture.pos]
        except Exception:
            pos = None
        records[name] = dict(name=name, cls=type(fixture).__name__, pos=pos,
                             rot=float(getattr(fixture, 'rot', 0.) or 0.),
                             size=[float(v) for v in getattr(fixture, 'size', [0., 0., 0.])])
    return records


def fixture_aabb(fixture):
    """World axis-aligned bounds ``[x0, x1, y0, y1, z0, z1]`` of a fixture, or None when it has none.

    ``Fixture.get_bbox_points`` reads ``self.rot``, which the room shell does not have (RoboCasa's ``Wall``
    and ``Floor`` are ``BoxObject``s turned by a fixed side quaternion), so the walls and their backings —
    the only fixtures that stand on a counter top's own footprint — fall back to RoboCasa's corner sites."""
    for name, kwargs in (('get_bbox_points', {}), ('get_ext_sites', dict(all_points=True, relative=False))):
        getter = getattr(fixture, name, None)
        if getter is None:
            continue
        try:
            points = np.asarray(getter(**kwargs), float)
        except Exception:      # noqa: BLE001  (no bounding region, or no rot)
            continue
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            continue
        return [float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()),
                float(points[:, 1].max()), float(points[:, 2].min()), float(points[:, 2].max())]
    return None


def enclosing_boxes(env, top_z, height, *, skip=(), min_rise=OCCLUDER_ROOM_RISE_M, min_side=.001):
    """World xy boxes of everything standing in the air over a surface at ``top_z``: every fixture whose
    bounding box reaches into the band from ``top_z + min_rise`` to ``top_z + height`` — the walls, their
    backings and the fixtures beside the surface (``skip`` leaves the surface's own fixture out).

    Nothing else keeps these out of a placement on the top: ``RoboQuestKitchen._decor_boxes`` skips the
    walls and the counters on purpose, and :func:`headroom.overhead_boxes` only sees what starts clear
    above the top, so an occluder at the end of a counter run or in front of a backing was built inside
    it (measured: stamps layout 16 style 4, 35.8 mm into wall_left_5_backing_room, caught by
    ``prop_clashes`` at G1 and the seed lost)."""
    out = []
    lo, hi = float(top_z) + float(min_rise), float(top_z) + float(height)
    for name in sorted(getattr(env, 'fixtures', {}) or {}):
        if name in skip:
            continue
        box = fixture_aabb(env.fixtures[name])
        if box is None:
            continue
        x0, x1, y0, y1, z0, z1 = box
        if z1 <= lo or z0 >= hi or x1 - x0 < min_side or y1 - y0 < min_side:
            continue           # below or above the band, or flat in the wall plane (switches, sills)
        out.append(dict(name=name, kind=type(env.fixtures[name]).__name__, x0=x0, x1=x1, y0=y0, y1=y1,
                        z0=z0, z1=z1))
    return out


def floor_map(env):
    """Room bounds from the walls (None when they cannot be found) and the plan footprints the base cannot
    drive through, from :mod:`search.layout` (fixtures whose underside is below 0.9 m; kick plates ignored)."""
    records = fixture_records(env)
    try:
        bounds = [float(v) for v in floor_bounds_from_walls(records)]
    except ValueError:
        bounds = None
    blockers = {name: [[float(x), float(y)] for x, y in poly] for name, poly in base_blockers(records).items()}
    return dict(bounds=bounds, blockers=blockers)


def _decor_at(env, top_z, cache=None):
    """``env._decor_boxes(top_z)``, memoised in ``cache`` (a dict) when one is given: a caller that tests
    many spots on one kitchen measures each surface's decor once."""
    if cache is None:
        return env._decor_boxes(top_z)
    key = round(float(top_z), 6)
    if key not in cache:
        cache[key] = env._decor_boxes(top_z)
    return cache[key]


def base_rectangle(xy, yaw):
    """Corners of the 0.70 x 0.70 m base rectangle for a base frame at ``xy`` facing ``yaw``."""
    f = np.array([math.cos(yaw), math.sin(yaw)])
    center = np.asarray(xy, float) + f * (BASE_AHEAD_M - BASE_BEHIND_M) / 2
    return _rect_corners(center, ((BASE_AHEAD_M + BASE_BEHIND_M) / 2, BASE_HALF_WIDTH_M), yaw)


def stance_gate(xy, yaw, floor, ignore=()):
    """Reach gate for a base stance: the base rectangle overlaps no fixture footprint (touching by
    :data:`STANCE_TOUCH_M` is allowed, the base's front stands at the edge) and stays inside the room."""
    corners = base_rectangle(xy, yaw)
    bounds = floor.get('bounds')
    in_room = True if bounds is None else _corners_inside(corners, bounds, FLOOR_MARGIN_M)
    hits = sorted(name for name, poly in floor['blockers'].items()
                  if name not in ignore and _polys_overlap(corners, poly, -STANCE_TOUCH_M))
    return dict(passed=bool(in_room and not hits), in_room=bool(in_room), blockers=hits,
                xy=[float(xy[0]), float(xy[1])], yaw=float(yaw), corners=corners.round(4).tolist())


def expected_base_pose(env, at_build=False):
    """(xy, yaw) of the base at reset. After reset the compiled sim answers; at build time RoboCasa's anchor
    for the work fixture is shifted along the frame the way ``_align_robot_base_with_frame`` will do it."""
    if not at_build:
        try:
            pos, rot = env.robots[0].part_controllers['base'].get_base_pose()
            return np.asarray(pos, float)[:2].copy(), _wrap(math.atan2(rot[1, 0], rot[0, 0]))
        except Exception:
            pass
    pos, ori = EnvUtils.init_robot_base_pose(env)
    xy = np.asarray(pos, float)[:2].copy()
    yaw = _wrap(float(np.asarray(ori, float).ravel()[-1]))
    work = getattr(env, 'work', None)
    if work:
        along = yaw_matrix(work['yaw'])[:2, 0]
        xy = xy + float(np.dot(np.asarray(work['center_world'], float)[:2] - xy, along)) * along
    return xy, yaw


def camera_frusta_at(base_xy, base_yaw, aspect=1., cameras=CAMERAS):
    R = yaw_matrix(base_yaw)
    base = np.array([float(base_xy[0]), float(base_xy[1]), 0.])
    return [dict(name=name, pos=base + R @ np.asarray(m['pos'], float), rot=R @ np.asarray(m['mat'], float),
                 fovy=math.radians(m['fovy']), aspect=float(aspect))
            for name, m in CAMERA_MOUNTS.items() if name in cameras]


def expected_camera_frusta(env, at_build=False, aspect=1.):
    """Policy-camera frusta at reset: from the compiled sim when it exists, else predicted from the base."""
    if not at_build and getattr(env, 'sim', None) is not None:
        try:
            return camera_frusta(env.sim, list(CAMERAS), aspect=aspect)
        except Exception:
            pass
    xy, yaw = expected_base_pose(env, at_build=True)
    return camera_frusta_at(xy, yaw, aspect)


# ---- task-frame helpers -----------------------------------------------------------------------------
def frame_to_world_xy(env, xy):
    return env.frame_to_world([float(xy[0]), float(xy[1]), 0.])[:2]


def world_to_frame_xy(env, xy):
    origin = np.asarray(env.work['center_world'], float)[:2]
    return _rot2(env.work['yaw']).T @ (np.asarray(xy, float)[:2] - origin)


def _shape_to_frame(env, shape):
    """A world-xy scatter shape (Disc, Box, Rect) as a task-frame shape."""
    shape = S.as_shape(shape)
    yaw = float(env.work['yaw'])
    if isinstance(shape, S.Disc):
        c = world_to_frame_xy(env, shape.center)
        return S.Disc((float(c[0]), float(c[1])), shape.radius)
    c = world_to_frame_xy(env, shape.center)
    return S.Box((float(c[0]), float(c[1])), shape.half, float(shape.yaw - yaw))


def decor_shapes_on_work(env):
    """RoboCasa decor and counter-top appliances on the work top, as task-frame boxes."""
    top_z = float(env.work['top_z'])
    yaw = float(env.work['yaw'])
    shapes = []
    for box in env._decor_boxes(top_z):
        center = world_to_frame_xy(env, ((box['x0'] + box['x1']) / 2, (box['y0'] + box['y1']) / 2))
        half = ((box['x1'] - box['x0']) / 2, (box['y1'] - box['y0']) / 2)
        shapes.append(S.Box((float(center[0]), float(center[1])), (float(half[0]), float(half[1])), -yaw))
    return shapes


def work_surface_rect(env):
    """The work fixture's top region as a task-frame :class:`Rect`."""
    work = env.work
    sx, sy = work['region_size']
    off = float(work['offset_along'])
    y_front = float(work['region_front_local']) - float(work['center_local'][1])
    return S.Rect(-sx / 2 - off, sx / 2 - off, y_front, y_front + sy)


# ---- task bodies ------------------------------------------------------------------------------------
def _find_body(root, name):
    for body in root.iter('body'):
        if body.get('name') == name:
            return body
    return None


def _geom_extent(body):
    """Bounding radius and top height of a body's analytic geoms (meshes contribute nothing)."""
    radius, top = 0., 0.
    for geom in body.iter('geom'):
        kind = geom.get('type', 'sphere')
        size = [float(v) for v in (geom.get('size') or '0').split()]
        pos = [float(v) for v in (geom.get('pos') or '0 0 0').split()]
        if kind == 'box':
            r, h = math.hypot(size[0], size[1]), size[2]
        elif kind in ('cylinder', 'capsule'):
            r, h = size[0], size[1] + (size[0] if kind == 'capsule' else 0.)
        elif kind == 'sphere':
            r, h = size[0], size[0]
        else:
            continue
        radius = max(radius, math.hypot(pos[0], pos[1]) + r)
        top = max(top, pos[2] + h)
    return radius, top


def body_extent(env, name, radius=None, height=None):
    """World pose and a bounding disc + height of a task body at build time.

    Objects registered in ``env.objects`` (MJCFObject, BoxObject) come with their pose in ``env._poses``
    and their bounding box; other free bodies are read from the model tree (analytic geoms only, so
    pass ``radius`` and ``height`` for mesh bodies such as the mugs)."""
    objects = getattr(env, 'objects', {}) or {}
    poses = getattr(env, '_poses', {}) or {}
    if name in objects and name in poses:
        p = poses[name]
        obj = objects[name]
        r = float(obj.horizontal_radius) if radius is None else float(radius)
        h = float(obj.top_offset[2] - obj.bottom_offset[2]) if height is None else float(height)
        return dict(name=name, xy=np.array([float(p[0]), float(p[1])]), z=float(p[2]),
                    yaw=_yaw_of_wxyz(p[3:7]), radius=r, height=h, source='object')
    body = _find_body(env.model.worldbody, name)
    if body is None:
        raise KeyError(f'no task body named {name!r}')
    pos = [float(v) for v in (body.get('pos') or '0 0 0').split()]
    quat = [float(v) for v in (body.get('quat') or '1 0 0 0').split()]
    r_geom, h_geom = _geom_extent(body)
    r = r_geom if radius is None else float(radius)
    h = h_geom if height is None else float(height)
    if r <= 0. or h <= 0.:
        raise ValueError(f'{name}: pass radius and height (no analytic geoms to measure)')
    return dict(name=name, xy=np.array(pos[:2]), z=pos[2], yaw=_yaw_of_wxyz(quat), radius=r, height=h, source='body')


def set_body_pose(env, name, xy_world, yaw_world, top_z, lift=0.):
    """Move a task body to a new world xy and yaw on a surface at ``top_z`` (plus ``lift``), keeping its
    own height above the support. Objects update ``env._poses``; XML bodies update their attributes."""
    objects = getattr(env, 'objects', {}) or {}
    poses = getattr(env, '_poses', {}) or {}
    x, y = float(xy_world[0]), float(xy_world[1])
    q = yaw_quat_wxyz(float(yaw_world))
    if name in objects and name in poses:
        obj = objects[name]
        z = float(top_z) - float(obj.bottom_offset[2]) + .002 + float(lift)
        before = list(poses[name])
        poses[name] = [x, y, z, float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        return dict(kind='object', before=before, after=list(poses[name]))
    body = _find_body(env.model.worldbody, name)
    if body is None:
        raise KeyError(f'no task body named {name!r}')
    pos = [float(v) for v in (body.get('pos') or '0 0 0').split()]
    above = pos[2] - float(env.work['top_z'])
    after = [x, y, float(top_z) + above + float(lift)]
    body.set('pos', _vec(after))
    body.set('quat', _vec(q))
    return dict(kind='body', before=pos, after=after)


# ---- surfaces and annex regions ---------------------------------------------------------------------
def _is_surface(fixture):
    return (fixture_is_type(fixture, FixtureType.COUNTER) or fixture_is_type(fixture, FixtureType.ISLAND)
            or fixture_is_type(fixture, FixtureType.DINING_COUNTER))


def surface_regions(env, min_size=MIN_REGION_M):
    """Top regions of every counter, island and dining counter (fixture-local rectangles with world frames)."""
    out = []
    work_name = env.work_fixture.name if getattr(env, 'work_fixture', None) is not None else None
    for name in sorted(env.fixtures):
        fixture = env.fixtures[name]
        if not _is_surface(fixture):
            continue
        try:
            regions = fixture.get_reset_regions(env, top_size=(min_size, min_size))
        except Exception:
            continue
        top_z = float(fixture.pos[2] + fixture.size[2] / 2)
        if not SURFACE_Z_BAND[0] <= top_z <= SURFACE_Z_BAND[1]:
            continue
        island = bool(fixture_is_type(fixture, FixtureType.ISLAND) or fixture_is_type(fixture, FixtureType.DINING_COUNTER))
        kind = 'work' if name == work_name else ('dining' if fixture_is_type(fixture, FixtureType.DINING_COUNTER)
                                                 else 'island' if island else 'counter')
        for reg_name in sorted(regions):
            reg = regions[reg_name]
            ox, oy = float(reg['offset'][0]), float(reg['offset'][1])
            sx, sy = float(reg['size'][0]), float(reg['size'][1])
            out.append(dict(fixture=name, region=reg_name, kind=kind, island=island, top_z=top_z,
                            origin=[float(fixture.pos[0]), float(fixture.pos[1])], yaw=float(fixture.rot),
                            rect=[ox - sx / 2, ox + sx / 2, oy - sy / 2, oy + sy / 2]))
    return out


def region_to_world(region, local_xy):
    return np.asarray(region['origin'], float) + _rot2(region['yaw']) @ np.asarray(local_xy, float)[:2]


def region_to_local(region, world_xy):
    return _rot2(region['yaw']).T @ (np.asarray(world_xy, float)[:2] - np.asarray(region['origin'], float))


def accessible_edges(region):
    """Edges the base may stand beyond: all four on an island or dining counter, never the wall side."""
    return ('front', 'back', 'left', 'right') if region['island'] else ('front', 'left', 'right')


def edge_chunks(rect, edge, cell=CELL_M):
    x0, x1, y0, y1 = rect
    if edge in ('front', 'back'):
        n = max(1, int(math.ceil((x1 - x0) / cell - 1e-9)))
        step = (x1 - x0) / n
        return [[x0 + i * step, x0 + (i + 1) * step, y0, y1] for i in range(n)]
    n = max(1, int(math.ceil((y1 - y0) / cell - 1e-9)))
    step = (y1 - y0) / n
    return [[x0, x1, y0 + i * step, y0 + (i + 1) * step] for i in range(n)]


def clip_depth(rect, edge, depth=DEPTH_MAX_M):
    x0, x1, y0, y1 = rect
    if edge == 'front':
        return [x0, x1, y0, min(y1, y0 + depth)]
    if edge == 'back':
        return [x0, x1, max(y0, y1 - depth), y1]
    if edge == 'left':
        return [x0, min(x1, x0 + depth), y0, y1]
    return [max(x0, x1 - depth), x1, y0, y1]


def edge_stance(region, rect, edge, standoff=STANDOFF_M):
    """Base frame (world xy, yaw) beyond ``edge`` of a local rectangle, facing the surface."""
    x0, x1, y0, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    point = {'front': (cx, y0), 'back': (cx, y1), 'left': (x0, cy), 'right': (x1, cy)}[edge]
    n = np.asarray(EDGES[edge], float)
    local = np.asarray(point, float) + n * standoff
    xy = region_to_world(region, local)
    heading = _rot2(region['yaw']) @ (-n)
    return xy, _wrap(math.atan2(heading[1], heading[0]))


def _local_keepouts(decor_boxes, region, rect, extra_world=()):
    """Decor boxes (world aabb dicts) and caller shapes (world), as fixture-local scatter shapes near ``rect``."""
    out = []
    probe = S.Rect(*rect)
    for box in decor_boxes:
        corners = [region_to_local(region, (x, y)) for x in (box['x0'], box['x1']) for y in (box['y0'], box['y1'])]
        corners = np.asarray(corners)
        shape = S.Rect(float(corners[:, 0].min()), float(corners[:, 0].max()),
                       float(corners[:, 1].min()), float(corners[:, 1].max()))
        if S.clearance(shape, probe) < .05:
            out.append(dict(name=box['name'], shape=shape))
    for index, item in enumerate(extra_world):
        shape = S.as_shape(item)
        if isinstance(shape, S.Disc):
            c = region_to_local(region, shape.center)
            local = S.Disc((float(c[0]), float(c[1])), shape.radius)
        else:
            c = region_to_local(region, shape.center)
            local = S.Box((float(c[0]), float(c[1])), shape.half, float(shape.yaw - region['yaw']))
        if S.clearance(local, probe) < .05:
            out.append(dict(name=f'extra_{index}', shape=local))
    return out


def annex_regions(env, *, floor=None, frusta=None, at_build=True, keepouts_world=(), object_height=.10,
                  object_radius=.05, min_along=ANNEX_MIN_ALONG_M, all_regions=False):
    """Candidate regions for task objects outside the policy cameras' view, gated for base reach.

    On the work surface only the parts at least ``min_along`` along the counter from the frame centre
    qualify; every other counter, island and dining counter qualifies whole. Each region is chunked
    along an accessible edge into cells (:data:`CELL_M`) clipped to :data:`DEPTH_MAX_M` from that edge,
    and carries: world corners and bounds, ``top_z``, the fixture name, a base stance (xy, yaw) beyond
    the edge and its reach gate (0.70 x 0.70 m base rectangle free of fixture footprints and inside the
    room; region centre within :data:`REGION_REACH_M`), the decor keep-outs on it (fixture-local) and
    ``hidden_fraction``: the share of a 5 x 5 grid of object-sized boxes in the cell that lie outside every
    predicted camera frustum. Regions that pass the gate and have some hidden area are returned, nearest
    stance first (work-surface annex first); ``all_regions`` returns everything with the gate results."""
    floor = floor or floor_map(env)
    frusta = frusta or expected_camera_frusta(env, at_build=at_build)
    base_xy, _ = expected_base_pose(env, at_build=at_build)
    surfaces = surface_regions(env)
    decor = {}
    for region in surfaces:
        if region['top_z'] not in decor:
            decor[region['top_z']] = env._decor_boxes(region['top_z'])
    return annex_regions_from(surfaces, env.work, floor, frusta, base_xy, decor, keepouts_world=keepouts_world,
                              object_height=object_height, object_radius=object_radius, min_along=min_along,
                              all_regions=all_regions)


def annex_regions_from(surfaces, work, floor, frusta, base_xy, decor, *, keepouts_world=(), object_height=.10,
                       object_radius=.05, min_along=ANNEX_MIN_ALONG_M, all_regions=False):
    """The geometry of :func:`annex_regions` on plain records: ``surfaces`` from :func:`surface_regions`,
    ``work`` the env's work dict, ``floor`` from :func:`floor_map`, ``frusta`` camera records, ``decor`` a
    map top_z -> world aabb boxes of the decor on that surface."""
    out = []
    for region in surfaces:
        rect = region['rect']
        if region['kind'] == 'work':
            fx = float(work['center_local'][0])
            parts = [[rect[0], min(rect[1], fx - min_along), rect[2], rect[3]],
                     [max(rect[0], fx + min_along), rect[1], rect[2], rect[3]]]
        else:
            parts = [rect]
        for part in parts:
            if part[1] - part[0] < MIN_REGION_M or part[3] - part[2] < MIN_REGION_M:
                continue
            for edge in accessible_edges(region):
                for index, chunk in enumerate(edge_chunks(part, edge)):
                    cell = clip_depth(chunk, edge)
                    if cell[1] - cell[0] < MIN_REGION_M or cell[3] - cell[2] < MIN_REGION_M:
                        continue
                    xy, yaw = edge_stance(region, cell, edge)
                    gate = stance_gate(xy, yaw, floor)
                    centre = region_to_world(region, ((cell[0] + cell[1]) / 2, (cell[2] + cell[3]) / 2))
                    distance = float(np.linalg.norm(centre - xy))
                    corners = np.array([region_to_world(region, (x, y)) for x, y in
                                        ((cell[0], cell[2]), (cell[1], cell[2]), (cell[1], cell[3]), (cell[0], cell[3]))])
                    far = float(max(np.linalg.norm(c - xy) for c in corners))
                    hidden = 0
                    xs = np.linspace(cell[0] + object_radius, cell[1] - object_radius, 5)
                    ys = np.linspace(cell[2] + object_radius, cell[3] - object_radius, 5)
                    for x in xs:
                        for y in ys:
                            w = region_to_world(region, (x, y))
                            points = box_points(w, (object_radius, object_radius), region['top_z'],
                                                region['top_z'] + object_height, region['yaw'])
                            hidden += int(hidden_from_all(frusta, points))
                    record = dict(name=f"{region['fixture']}:{region['region']}:{edge}:{index}",
                                  fixture=region['fixture'], region=region['region'], kind=region['kind'],
                                  island=region['island'], top_z=region['top_z'], origin=list(region['origin']),
                                  yaw=region['yaw'], rect=[round(float(v), 4) for v in cell], edge=edge,
                                  corners_world=corners.round(4).tolist(),
                                  bounds_world=[float(corners[:, 0].min()), float(corners[:, 0].max()),
                                                float(corners[:, 1].min()), float(corners[:, 1].max())],
                                  center_world=[float(centre[0]), float(centre[1])],
                                  stance=dict(xy=[float(xy[0]), float(xy[1])], yaw=float(yaw), edge=edge, standoff=STANDOFF_M),
                                  reach=dict(passed=bool(gate['passed'] and distance <= REGION_REACH_M),
                                             distance_m=round(distance, 4), far_corner_m=round(far, 4),
                                             in_room=gate['in_room'], blockers=gate['blockers'],
                                             base_corners=gate['corners']),
                                  keepouts=_local_keepouts(decor.get(region['top_z'], ()), region, cell, keepouts_world),
                                  hidden_fraction=hidden / 25.,
                                  drive_m=round(float(np.linalg.norm(xy - base_xy)), 4))
                    if all_regions or (record['reach']['passed'] and hidden > 0):
                        out.append(record)
    out.sort(key=lambda r: (r['kind'] != 'work', r['drive_m'], r['name']))
    return out


def place_in_annex(env, region, rng, *, radius, height, frusta=None, at_build=True, tries=40, min_gap=.01,
                   headroom=None, boxes=None, reach=None, reach_tries=ANNEX_REACH_TRIES, diagnostics=None):
    """A spot for an object (bounding disc ``radius``, ``height``) inside an annex region, clear of the
    region's keep-outs by ``min_gap`` and outside every camera frustum. With ``headroom`` (metres, or the
    dict :func:`headroom.headroom_needed` returns) the spot's column must also be clear of overhead fixtures,
    because :func:`uncover.fetch` lifts the object from there. Returns None when no draw works.

    ``reach`` is called as ``reach(world_xy, top_z)`` for a spot that is otherwise good and must return a
    :func:`spot_reachable` record: :func:`annex_regions` gates a *cell centre* against the cell's own
    stance, but the object may sit up to half a cell along the edge from that centre, so a spot the gate
    let through can still have no reachable stance and the instance builds and then fails the look gate
    (measured: marked_mugs ``look`` on island kitchens, the far cells of island_island_group_1). A spot
    whose walk fails is skipped; after ``reach_tries`` such spots the region is abandoned for the next one.
    ``diagnostics`` (a dict) receives the counters of the draws."""
    frusta = frusta or expected_camera_frusta(env, at_build=at_build)
    rect = S.Rect(*region['rect'])
    keeps = [k['shape'] for k in region.get('keepouts', [])]
    if headroom is not None and boxes is None:
        boxes = H.overhead_boxes(env, region['top_z'])
    diag = diagnostics if diagnostics is not None else {}
    diag.update(attempts=0, no_spot=False, seen=0, headroom_blocked=0, reach_blocked=0, reasons=[], placed=False)
    blocked = 0
    for attempt in range(int(tries)):
        try:
            x, y, yaw = S.scatter(rng, 1, rect, float(radius), 0., keepouts=keeps, keepout_gap=min_gap,
                                  max_tries=200, attempts_per_prop=200)[0]
        except ValueError:
            diag['no_spot'] = True
            return None
        diag['attempts'] = attempt + 1
        world = region_to_world(region, (x, y))
        points = box_points(world, (radius, radius), region['top_z'], region['top_z'] + height, region['yaw'] + yaw)
        if not hidden_from_all(frusta, points):
            diag['seen'] += 1
            continue
        gate = None
        if headroom is not None:
            gate = H.headroom_gate(boxes, world, radius, headroom, region['top_z'])
            if not gate['passed']:
                blocked += 1
                diag['headroom_blocked'] = blocked
                continue
        walk = None
        if reach is not None:
            walk = reach(world, region['top_z'])
            if not walk['passed']:
                diag['reach_blocked'] += 1
                for why in walk['reasons']:
                    if why not in diag['reasons']:
                        diag['reasons'].append(why)
                if diag['reach_blocked'] >= int(reach_tries):
                    return None
                continue
        diag['placed'] = True
        return dict(region=region['name'], fixture=region['fixture'], top_z=region['top_z'],
                    local=[round(x, 4), round(y, 4), round(yaw, 4)],
                    xy_world=[float(world[0]), float(world[1])], yaw_world=_wrap(region['yaw'] + yaw),
                    stance=deepcopy(region['stance']), attempts=attempt + 1, headroom=gate, headroom_blocked=blocked,
                    look_reach=walk)
    return None


# ---- occluders --------------------------------------------------------------------------------------
def occluder_candidates(kinds=None, min_height=OCCLUDER_MIN_HEIGHT_M, rng=None, max_thin=OCCLUDER_MAX_THIN_M):
    """Tall opaque RoboCasa objects usable as look occluders (copies; shuffled when ``rng`` is given).

    A prop whose thinnest horizontal side is wider than ``max_thin`` is dropped: the gripper cannot close on
    it, so :func:`uncover.move_aside` can only push it out of the line of sight, and on a crowded top there
    may be no heading that does (measured on stamps 5b7508dd74b85e47, a 166 mm jug wedged between a prop at
    0 mm and a fixture the body already stands 6 mm under: all 24 headings refuse). Pass ``max_thin=None``
    for the whole pool - what an instance minted before this filter has to be rebuilt with."""
    out = [deepcopy(o) for o in OCCLUDERS if (kinds is None or o['kind'] in kinds)]
    if rng is not None:
        order = rng.permutation(len(out))
        out = [out[int(i)] for i in order]
    return [o for o in out if _occluder_size(o)[2] >= min_height - 1e-9
            and (max_thin is None or min(_occluder_size(o)[:2]) <= float(max_thin) + 1e-9)]


_SIZE_CACHE = {}


def _occluder_size(candidate):
    """Full sizes (x, y, z) of an occluder at its scale, from the asset's bounding box (cached)."""
    key = (candidate['asset'], float(candidate['scale']))
    if key not in _SIZE_CACHE:
        from robocasa.models.objects.objects import MJCFObject
        from roboquest.base.scene import ASSET_ROOT
        obj = MJCFObject(name='probe', mjcf_path=str(ASSET_ROOT / candidate['asset'] / 'model.xml'),
                         scale=float(candidate['scale']))
        _SIZE_CACHE[key] = tuple(float(v) for v in obj.size)
    return _SIZE_CACHE[key]


def place_occluder(env, target_body, rng, *, radius=None, height=None, keepouts=(), bounds=None, frusta=None,
                   at_build=True, candidates=None, gaps=(.02, .04, .06, .08, .10, .13, .16),
                   laterals=(0., .03, -.03, .06, -.06), name=None, margin_rad=CAMERA_MARGIN_RAD, build=True,
                   diagnostics=None, relocate_region=None, relocate_tries=40, reach=None, pinchable=None):
    """Stand a tall opaque object between the cameras and ``target_body`` so the target is in no policy
    camera at reset. ``keepouts`` and ``bounds`` are task-frame scatter shapes (the task's props, pads,
    Submit; the work top's decor is added here).

    The policy cameras sit high on the robot (1.40 to 1.75 m) and look down steeply, so an object within
    about 0.45 m of the base cannot be hidden by anything standing on the counter (measured: the front
    rows of stamps, vessels and parcels block at most one camera at any occluder position). With
    ``relocate_region`` (a task-frame Rect, normally the back of the task's free area) the target is moved
    to a spot inside it, clear of ``keepouts``, where an occluder does hide it. Returns the placement
    record (``relocated`` tells whether and where the target moved; the occluder is added to
    ``env.objects`` when ``build``) or None when no candidate blocks every camera with clearance.
    ``diagnostics`` (a dict) receives rejection counters.

    A candidate also keeps :func:`enclosing_boxes` out by :data:`OCCLUDER_GAP_M` — the walls, the backings
    and the neighbouring fixtures that stand in the air over the work top, which are in neither the decor
    keep-outs nor the overhead boxes — and the ones near the top are listed in the record under
    ``fixture_keepouts``.

    ``pinchable`` (default :data:`OCCLUDER_PINCHABLE_MINTS`) draws from the part of the pool the gripper can
    close on and, for a wide candidate handed in explicitly, takes it only where
    :func:`uncover.push_aside_plan` finds a heading that pushes it out of the line of sight (counted under
    ``no_push``). It is off: which occluder stands in a scene is decided here, at build time, and is not
    recorded in the instance, so narrowing the pool would re-roll the occluder of every instance already
    minted rather than only the mints to come.

    ``reach`` (a callable taking the world xy of a relocation spot, normally :func:`spot_reachable`) gates
    the spots the target is moved to: a spot the base cannot work at is skipped and the walk draws the next
    one, since the look gate measures the object where the relocation leaves it. What the walk cost is
    recorded under ``relocate_walk``."""
    work = env.work
    top_z = float(work['top_z'])
    ext = body_extent(env, target_body, radius=radius, height=height)
    frusta = frusta or expected_camera_frusta(env, at_build=at_build)
    keeps = [S.as_shape(k) for k in keepouts] + decor_shapes_on_work(env)
    bounds = bounds if bounds is not None else work_surface_rect(env)
    pinchable = OCCLUDER_PINCHABLE_MINTS if pinchable is None else bool(pinchable)
    candidates = candidates if candidates is not None else \
        occluder_candidates(rng=rng, max_thin=OCCLUDER_MAX_THIN_M if pinchable else None)
    diag = diagnostics if diagnostics is not None else {}
    diag.update(positions=0, not_blocked=0, out_of_bounds=0, target_overlap=0, keepout_clash=0, best_blocked=0,
                relocations=0, already_hidden=False, no_headroom=0, fixture_clash=0, reach_blocked=0,
                no_push=0, reach_reasons=[])
    overhead = H.overhead_boxes(env, top_z)          # move_aside lifts the occluder: its column needs headroom
    # the room itself: a candidate may not stand inside a wall, a backing or a fixture beside the top
    work_name = env.work_fixture.name if getattr(env, 'work_fixture', None) is not None else None
    tallest = max([_occluder_size(c)[2] for c in candidates] or [0.])
    enclosure = []
    for box in enclosing_boxes(env, top_z, tallest, skip=(work_name,) if work_name else ()):
        centre = world_to_frame_xy(env, ((box['x0'] + box['x1']) / 2, (box['y0'] + box['y1']) / 2))
        shape = S.Box((float(centre[0]), float(centre[1])),
                      ((box['x1'] - box['x0']) / 2, (box['y1'] - box['y0']) / 2), -float(work['yaw']))
        if S.clearance(shape, bounds) > OCCLUDER_GAP_M:
            continue                                 # nowhere near the surface a candidate stands on
        enclosure.append(dict(box, shape=shape))

    def search(target_xy):
        """The first (candidate, spot) hiding a target that stands at ``target_xy`` (world), or None."""
        target_points = box_points(target_xy, (ext['radius'], ext['radius']), top_z, top_z + ext['height'], inflate=.01)
        needed = seen_by(frusta, target_points, margin_rad)
        if not any(needed.values()):
            return dict(already_hidden=True, cameras_needed=needed)
        cams = np.mean([f['pos'][:2] for f in frusta if needed[f['name']]], axis=0)
        u = _unit(cams - target_xy)
        u_frame = _rot2(float(work['yaw'])).T @ u          # target toward the cameras, in the task frame
        v = np.array([-u[1], u[0]])
        target_frame = world_to_frame_xy(env, target_xy)
        # a spot no occluder can serve is dropped after one test with an ideal slab as wide and tall as the
        # largest candidate at the closest gap: the per-candidate search below is then never entered
        ideal_yaw = _wrap(math.atan2(-u[1], -u[0]) - math.pi / 2)
        ideal_centre = target_xy + u * (ext['radius'] + IDEAL_OCCLUDER[1] / 2 + gaps[0])
        ideal_blocked, _ = occludes(frusta, target_points, np.array([ideal_centre[0], ideal_centre[1], top_z + IDEAL_OCCLUDER[2] / 2]),
                                    (IDEAL_OCCLUDER[0] / 2, IDEAL_OCCLUDER[1] / 2, IDEAL_OCCLUDER[2] / 2), ideal_yaw, margin_rad)
        diag['positions'] += 1
        if not all(ideal_blocked[c] for c in needed if needed[c]):
            diag['not_blocked'] += 1
            return None
        for cand in candidates:
            sx, sy, sz = _occluder_size(cand)
            if cand['toward'] == 'y':          # local +y points from the cameras toward the target
                yaw_world = _wrap(math.atan2(-u[1], -u[0]) - math.pi / 2)
                hd = sy / 2
            else:                              # local +x points toward the target
                yaw_world = _wrap(math.atan2(-u[1], -u[0]))
                hd = sx / 2
            yaw_frame = _wrap(yaw_world - float(work['yaw']))
            for lateral in laterals:
                for gap in gaps:
                    centre = target_xy + u * (ext['radius'] + hd + gap) + v * lateral
                    # the blocking box is the occluder's own bounding box (its axes, its yaw), shrunk a little
                    blocked, _ = occludes(frusta, target_points, np.array([centre[0], centre[1], top_z + sz / 2]),
                                          (sx / 2 * OCCLUDER_INSET, sy / 2 * OCCLUDER_INSET, sz / 2 * .95),
                                          yaw_world, margin_rad)
                    diag['positions'] += 1
                    diag['best_blocked'] = max(diag['best_blocked'], sum(blocked[c] for c in needed if needed[c]))
                    if not all(blocked[c] for c in needed if needed[c]):
                        diag['not_blocked'] += 1
                        continue
                    cf = world_to_frame_xy(env, centre)
                    shape = S.Box((float(cf[0]), float(cf[1])), (sx / 2, sy / 2), yaw_frame)
                    if S.region_margin(shape, bounds) < 0.:
                        diag['out_of_bounds'] += 1
                        continue
                    if S.clearance(shape, S.Disc((float(target_frame[0]), float(target_frame[1])), ext['radius'])) < 0.:
                        diag['target_overlap'] += 1
                        continue
                    if any(S.clearance(shape, k) < OCCLUDER_GAP_M for k in keeps):
                        diag['keepout_clash'] += 1
                        continue
                    if any(e['z0'] < top_z + sz and S.clearance(shape, e['shape']) < OCCLUDER_GAP_M
                           for e in enclosure):
                        diag['fixture_clash'] += 1
                        continue
                    room = H.headroom_gate(overhead, centre, max(sx, sy) / 2,
                                           H.headroom_needed('move_aside', object_height=sz), top_z)
                    if not room['passed']:
                        diag['no_headroom'] += 1
                        continue
                    push = None
                    if pinchable and min(sx, sy) > OCCLUDER_MAX_THIN_M:
                        from roboquest import uncover as U           # deferred: uncover imports this module
                        push = U.push_aside_plan(cf, math.hypot(sx, sy) / 2 + .01, target_frame, ext['radius'],
                                                 [u_frame], bounds, keeps + [e['shape'] for e in enclosure],
                                                 footprint=(sx / 2 + .01, sy / 2 + .01, yaw_frame))
                        if not push['passed']:
                            diag['no_push'] += 1
                            continue
                    return dict(id=cand['id'], kind=cand['kind'], asset=cand['asset'], scale=float(cand['scale']),
                                size=[round(sx, 4), round(sy, 4), round(sz, 4)],
                                xy_world=[float(centre[0]), float(centre[1])], yaw_world=float(yaw_world),
                                frame_xy=[float(cf[0]), float(cf[1])], frame_yaw=float(yaw_frame),
                                gap_m=float(gap), lateral_m=float(lateral), cameras_needed=needed, blocked=blocked,
                                headroom=room, push=push)
        return None

    target_xy = np.asarray(ext['xy'], float)
    target_yaw = float(ext['yaw'])
    found = search(target_xy)
    relocated = None
    if (found is None or found.get('already_hidden')) and relocate_region is not None:
        found = None
        for attempt in range(int(relocate_tries)):
            try:
                x, y, yaw = S.scatter(rng, 1, relocate_region, ext['radius'], OCCLUDER_GAP_M, keepouts=keeps,
                                      keepout_gap=OCCLUDER_GAP_M, max_tries=300, attempts_per_prop=300)[0]
            except ValueError:
                break
            diag['relocations'] += 1
            spot = frame_to_world_xy(env, (x, y))
            walk = reach(spot) if reach is not None else None
            if walk is not None and not walk['passed']:
                diag['reach_blocked'] += 1        # the look gate measures the object where this leaves it
                for reason in walk['reasons'][:1]:
                    if reason not in diag['reach_reasons']:
                        diag['reach_reasons'].append(reason)
                continue
            result = search(spot)
            if result is not None and not result.get('already_hidden'):
                relocated = dict(from_xy_world=[float(target_xy[0]), float(target_xy[1])],
                                 to_xy_world=[float(spot[0]), float(spot[1])],
                                 to_frame=[round(x, 4), round(y, 4), round(yaw, 4)], attempts=attempt + 1,
                                 look_reach=walk)
                target_xy, target_yaw = spot, _wrap(float(work['yaw']) + yaw)
                found = result
                break
    if found is None or found.get('already_hidden'):
        diag['already_hidden'] = bool(found and found.get('already_hidden'))
        return None
    record = dict(name=name or f'occluder_{target_body}', target=target_body, target_radius=ext['radius'],
                  target_height=ext['height'], relocated=relocated,
                  relocate_walk=None if relocate_region is None else dict(
                      spots=diag['relocations'], reach_blocked=diag['reach_blocked'],
                      reasons=list(diag['reach_reasons'])[:2]),
                  fixture_keepouts=[{k: v for k, v in e.items() if k != 'shape'} for e in enclosure], **found)
    if build:
        if relocated is not None:
            record['target_moved'] = set_body_pose(env, target_body, target_xy, target_yaw, top_z)
        env.add_native(record['name'], record['asset'], list(record['xy_world']), top_z, scale=record['scale'],
                       quat_xyzw=_quat_xyzw(record['yaw_world']))
        env.task_spec['objects'][record['name']]['role'] = 'occluder'
    return record


# ---- reach gates ------------------------------------------------------------------------------------
def reach_gate(env, point_world, *, floor=None, at_build=True, base_pose=None):
    """Is a world point inside the top-down reach envelope from the reset stance, letting the base slide
    along the counter (its lateral axis) to centre it? Sliding stances pass :func:`stance_gate`."""
    base_xy, base_yaw = base_pose if base_pose is not None else expected_base_pose(env, at_build=at_build)
    f = np.array([math.cos(base_yaw), math.sin(base_yaw)])
    left = np.array([-f[1], f[0]])
    rel = np.asarray(point_world, float)[:2] - np.asarray(base_xy, float)
    ahead, lateral = float(rel @ f), float(rel @ left)
    shift = float(np.clip(lateral, -MAX_SLIDE_M, MAX_SLIDE_M))
    stance = np.asarray(base_xy, float) + left * shift
    residual = lateral - shift
    if abs(shift) > 1e-6:
        gate = stance_gate(stance, base_yaw, floor or floor_map(env))
    else:
        gate = dict(passed=True, in_room=True, blockers=[], xy=[float(stance[0]), float(stance[1])], yaw=float(base_yaw))
    passed = bool(REACH_MIN_AHEAD_M <= ahead <= GRASP_REACH_M and abs(residual) <= LATERAL_REACH_M and gate['passed'])
    reasons = []
    if ahead > GRASP_REACH_M:
        reasons.append(f'{ahead:.2f} m ahead exceeds {GRASP_REACH_M:.2f}')
    if ahead < REACH_MIN_AHEAD_M:
        reasons.append(f'{ahead:.2f} m ahead is inside the base')
    if abs(residual) > LATERAL_REACH_M:
        reasons.append(f'{residual:+.2f} m lateral residual after sliding {shift:+.2f}')
    if not gate['passed']:
        reasons.append('slid stance blocked: ' + (', '.join(gate['blockers']) or 'outside the room'))
    return dict(passed=passed, ahead_m=round(ahead, 4), lateral_m=round(lateral, 4), slide_m=round(shift, 4),
                residual_lateral_m=round(residual, 4), point=[round(float(v), 4) for v in np.asarray(point_world, float)],
                stance=dict(xy=[float(stance[0]), float(stance[1])], yaw=float(base_yaw)), stance_gate=gate,
                reasons=reasons)


def _frame_edges(env):
    """Task-frame outward directions and coordinates of the work top's edges."""
    rect = work_surface_rect(env)
    return dict(front=((0., -1.), rect.y0), back=((0., 1.), rect.y1), left=((-1., 0.), rect.x0), right=((1., 0.), rect.x1)), rect


def slide_gate(env, bowl_xy_world, bowl_radius, object_radius, *, keepouts=(), floor=None, at_build=True,
               edges=('front', 'left', 'right'), base_pose=None):
    """Gate for the slide-and-lift removal of an inverted bowl: a straight slide to a reachable counter
    edge, stopping once a third of the bowl overhangs, with the path clear of ``keepouts`` (task-frame
    shapes), the rim beyond the edge inside the reach envelope from a gated stance, and the covered
    object (pushed along by the trailing wall in the worst case) still on the counter at the end."""
    work = env.work
    floor = floor or floor_map(env)
    edges_info, rect = _frame_edges(env)
    bowl = world_to_frame_xy(env, bowl_xy_world)
    keeps = [S.as_shape(k) for k in keepouts] + decor_shapes_on_work(env)
    inner = BOWL_INNER_FRACTION * float(bowl_radius)
    inside_after = float(bowl_radius) / 3 + inner - 2 * float(object_radius)
    options, reasons = [], []
    if inside_after < COVER_EDGE_MARGIN_M:
        reasons.append(f'object would overhang after the slide ({inside_after:+.3f} m)')
    for edge in edges:
        direction, _ = edges_info[edge]
        d = np.asarray(direction, float)
        # distance from the bowl centre to the edge line along the outward direction
        dist = {'front': bowl[1] - rect.y0, 'back': rect.y1 - bowl[1], 'left': bowl[0] - rect.x0, 'right': rect.x1 - bowl[0]}[edge]
        slide = float(dist - bowl_radius / 3)
        if slide <= 0.:
            reasons.append(f'{edge}: bowl already at the edge')
            continue
        if slide > MAX_SLIDE_PATH_M:
            reasons.append(f'{edge}: slide of {slide:.2f} m exceeds {MAX_SLIDE_PATH_M:.2f}')
            continue
        angle = math.atan2(d[1], d[0])
        strip = S.Box((float(bowl[0] + d[0] * slide / 2), float(bowl[1] + d[1] * slide / 2)),
                      (slide / 2 + bowl_radius + COVER_EDGE_MARGIN_M, bowl_radius + COVER_EDGE_MARGIN_M), angle)
        blocking = [i for i, k in enumerate(keeps) if S.clearance(strip, k) < 0.]
        if blocking:
            reasons.append(f'{edge}: slide path blocked by {len(blocking)} keep-out(s)')
            continue
        # the bowl must not cross a side edge while it slides; the target edge may be crossed by design
        lateral_ok = True
        for other, (od, oc) in edges_info.items():
            if other == edge:
                continue
            od = np.asarray(od, float)
            if abs(float(od @ d)) > .5:      # the opposite edge: irrelevant
                continue
            margin = {'front': bowl[1] - rect.y0, 'back': rect.y1 - bowl[1], 'left': bowl[0] - rect.x0, 'right': rect.x1 - bowl[0]}[other]
            if margin < bowl_radius + COVER_EDGE_MARGIN_M:
                lateral_ok = False
        if not lateral_ok:
            reasons.append(f'{edge}: bowl too close to a side edge')
            continue
        rim_frame = np.array([bowl[0] + d[0] * (slide + bowl_radius - .01), bowl[1] + d[1] * (slide + bowl_radius - .01)])
        rim_world = frame_to_world_xy(env, rim_frame)
        if edge == 'front':
            reach = reach_gate(env, rim_world, floor=floor, at_build=at_build, base_pose=base_pose)
        else:
            edge_point = frame_to_world_xy(env, [bowl[0] + d[0] * dist, bowl[1] + d[1] * dist])
            stance_xy = edge_point + (_rot2(work['yaw']) @ d) * STANDOFF_M
            heading = _rot2(work['yaw']) @ (-d)
            stance_yaw = _wrap(math.atan2(heading[1], heading[0]))
            gate = stance_gate(stance_xy, stance_yaw, floor)
            reach = reach_gate(env, rim_world, floor=floor, base_pose=(stance_xy, stance_yaw))
            reach['edge_stance'] = gate
            if not gate['passed']:
                reach['passed'] = False
                reach['reasons'].append('edge stance blocked: ' + (', '.join(gate['blockers']) or 'outside the room'))
        if not reach['passed']:
            reasons.append(f'{edge}: rim out of reach ({"; ".join(reach["reasons"])})')
            continue
        options.append(dict(edge=edge, slide_m=round(slide, 4), direction_frame=[float(d[0]), float(d[1])],
                            strip=dict(center=list(strip.center), half=list(strip.half), yaw=strip.yaw),
                            rim_world=[float(rim_world[0]), float(rim_world[1])], reach=reach,
                            object_inside_after_m=round(inside_after, 4)))
    best = min(options, key=lambda o: o['slide_m']) if options and inside_after >= COVER_EDGE_MARGIN_M else None
    return dict(passed=best is not None, chosen=best, options=options, reasons=reasons,
                bowl_frame=[float(bowl[0]), float(bowl[1])])


def push_gate(env, cover_xy_frame, cover_half, cover_yaw, place_half, *, keepouts=(), bounds=None, floor=None,
              at_build=True, directions=((-1., 0.), (1., 0.), (0., 1.)), base_pose=None):
    """Gate for pushing a flat cover aside: the push path (cover plus place extent plus 2 cm) is clear of
    ``keepouts`` and inside ``bounds`` (task-frame), and the contact point behind the cover is reachable."""
    keeps = [S.as_shape(k) for k in keepouts] + decor_shapes_on_work(env)
    bounds = bounds if bounds is not None else work_surface_rect(env)
    options, reasons = [], []
    for direction in directions:
        d = _unit(direction)
        angle = math.atan2(d[1], d[0])
        c_along = abs(float(np.dot(d, (math.cos(cover_yaw), math.sin(cover_yaw))))) * cover_half[0] \
            + abs(float(np.dot(d, (-math.sin(cover_yaw), math.cos(cover_yaw))))) * cover_half[1]
        c_across = abs(float(np.dot((-d[1], d[0]), (math.cos(cover_yaw), math.sin(cover_yaw))))) * cover_half[0] \
            + abs(float(np.dot((-d[1], d[0]), (-math.sin(cover_yaw), math.cos(cover_yaw))))) * cover_half[1]
        p_along = abs(d[0]) * place_half[0] + abs(d[1]) * place_half[1]
        # the cover clears the place once it has moved its own half extent plus the place's, plus 2 cm
        length = c_along + p_along + .02
        centre = np.asarray(cover_xy_frame, float) + d * length / 2
        strip = S.Box((float(centre[0]), float(centre[1])), (length / 2 + c_along, c_across + COVER_EDGE_MARGIN_M), angle)
        if S.region_margin(strip, bounds) < 0.:
            reasons.append(f'{tuple(round(float(v), 1) for v in d)}: path leaves the surface')
            continue
        blocking = [i for i, k in enumerate(keeps) if S.clearance(strip, k) < 0.]
        if blocking:
            reasons.append(f'{tuple(round(float(v), 1) for v in d)}: path blocked by {len(blocking)} keep-out(s)')
            continue
        contact = np.asarray(cover_xy_frame, float) - d * (c_along + .01)
        reach = reach_gate(env, frame_to_world_xy(env, contact), floor=floor, at_build=at_build, base_pose=base_pose)
        if not reach['passed']:
            reasons.append(f'{tuple(round(float(v), 1) for v in d)}: contact point out of reach')
            continue
        options.append(dict(direction_frame=[float(d[0]), float(d[1])], length_m=round(length, 4),
                            strip=dict(center=list(strip.center), half=list(strip.half), yaw=strip.yaw),
                            contact_frame=[float(contact[0]), float(contact[1])], reach=reach))
    return dict(passed=bool(options), chosen=options[0] if options else None, options=options, reasons=reasons)


# ---- covers -----------------------------------------------------------------------------------------
def _bowl_depth_at(cover, radius):
    """Conservative interior depth below the rim at ``radius`` (scale 1): the profile made monotone."""
    profile = sorted((float(r), float(d)) for r, d in cover['profile'])
    depths, floor = [], math.inf
    for r, d in profile:
        floor = min(floor, d)
        depths.append((r, floor))
    if radius <= depths[0][0]:
        return depths[0][1]
    for (r0, d0), (r1, d1) in zip(depths, depths[1:]):
        if r0 <= radius <= r1:
            t = (radius - r0) / max(r1 - r0, 1e-9)
            return d0 + t * (d1 - d0)
    return 0.


def bowl_scale_for(cover, object_radius, object_height, clearance=COVER_CLEARANCE_M, slide_margin=COVER_EDGE_MARGIN_M):
    """Smallest scale at which the inverted bowl clears the object (interior depth at the object's radius,
    inner radius at the rim) and the object still stays ``slide_margin`` inside the counter edge after
    the slide-and-lift (see :func:`slide_gate`), or None."""
    for s in BOWL_SCALES:
        R = cover['radius'] * s
        inner = BOWL_INNER_FRACTION * R
        depth = s * _bowl_depth_at(cover, object_radius / s)
        inside_after = R / 3 + inner - 2 * object_radius
        if object_radius + clearance <= inner and object_height + clearance <= depth and inside_after >= slide_margin:
            return float(s)
    return None


def cover_footprint_clear(env, xy_world, radius, keepouts=(), gap=OCCLUDER_GAP_M):
    """Does a round cover of ``radius`` standing at ``xy_world`` clear the task-frame ``keepouts`` (and the
    work top's decor) by ``gap``? Returns (ok, number of clashing shapes)."""
    c = world_to_frame_xy(env, xy_world)
    disc = S.Disc((float(c[0]), float(c[1])), float(radius))
    keeps = [S.as_shape(k) for k in keepouts] + decor_shapes_on_work(env)
    clashes = sum(1 for k in keeps if S.clearance(disc, k) < gap)
    return clashes == 0, clashes


# ---- headroom (OBSERVE-HEADROOM): the room above a cover its uncover skill needs, see :mod:`headroom` ------
def cover_headroom(env, kind, xy_world, radius, *, cover_height=0., object_height=0., support_above=0., boxes=None,
                   top_z=None):
    """Headroom gate for a cover (or lifted object) of ``kind`` standing at ``xy_world`` on the work top: the
    column over its disc (``radius`` + :data:`headroom.HAND_PAD_M`) must hold :func:`headroom.headroom_needed`
    metres of clear air under every overhead fixture (wall cabinets, hoods, microwaves, shelves)."""
    top_z = float(env.work['top_z']) if top_z is None else float(top_z)
    boxes = H.overhead_boxes(env, top_z) if boxes is None else boxes
    need = H.headroom_needed(kind, cover_height=cover_height, object_height=object_height, radius=radius,
                             support_above=support_above)
    gate = H.headroom_gate(boxes, xy_world, radius, need, top_z)
    gate.update(kind=kind, hand_m=need['hand_m'], lift_m=need['lift_m'], allowance_m=need['allowance_m'])
    return gate


def _bowl_headroom(env, gate, bowl_xy_world, bowl_radius, bowl_height, boxes=None):
    """Drop the slide options of a :func:`slide_gate` result whose slide path (the hand pushing low) or lift
    point at the edge (the bowl raised to ``top + 2R + .12``) lacks headroom, re-pick ``chosen`` (shortest
    slide) and return the tightest gate that was applied. Mutates ``gate``."""
    top_z = float(env.work['top_z'])
    boxes = H.overhead_boxes(env, top_z) if boxes is None else boxes
    lift = H.headroom_needed('bowl', cover_height=bowl_height, radius=bowl_radius)
    slide = H.headroom_needed('bowl_slide', cover_height=bowl_height)
    kept, applied = [], []
    for option in gate.get('options', []):
        rim = np.asarray(option['rim_world'], float)
        at_rim = H.headroom_gate(boxes, rim, bowl_radius, lift, top_z)
        along = H.path_headroom_gate(boxes, H.path_points(bowl_xy_world, rim), bowl_radius, slide, top_z)
        option['headroom'] = dict(lift=at_rim, slide=along)
        applied.extend([at_rim, along])
        if at_rim['passed'] and along['passed']:
            kept.append(option)
        else:
            gate['reasons'].append(f"{option['edge']}: " + '; '.join(at_rim['reasons'] + along['reasons']))
    gate['options'] = kept
    gate['chosen'] = min(kept, key=lambda o: o['slide_m']) if kept else None
    gate['passed'] = bool(gate['passed'] and kept)
    worst = min(applied, key=lambda g: (g['passed'], g['available_m'] if g['available_m'] is not None else math.inf),
                default=H.headroom_gate(boxes, bowl_xy_world, bowl_radius, lift, top_z))
    return dict(worst, kind='bowl', lift_m=lift['lift_m'], hand_m=lift['hand_m'], allowance_m=lift['allowance_m'],
                slide_needed_m=slide['needed_m'])


def _push_headroom(env, gate, cover_xy_frame, cover_half, cover_thickness, boxes=None):
    """Drop the push options of a :func:`push_gate` result whose contact point (the hand's waypoint 0.15 m up)
    or push path (the hand down at the cover's edge) lacks headroom; re-pick ``chosen``. Mutates ``gate``."""
    top_z = float(env.work['top_z'])
    boxes = H.overhead_boxes(env, top_z) if boxes is None else boxes
    radius = float(max(cover_half))
    contact_need = H.headroom_needed('push_contact', cover_height=cover_thickness)
    path_need = H.headroom_needed('push_path', cover_height=cover_thickness)
    start = frame_to_world_xy(env, cover_xy_frame)
    kept, applied = [], []
    for option in gate.get('options', []):
        contact = frame_to_world_xy(env, option['contact_frame'])
        end = frame_to_world_xy(env, np.asarray(cover_xy_frame, float) + np.asarray(option['direction_frame'], float)
                                * float(option['length_m']))
        at_contact = H.headroom_gate(boxes, contact, .05, contact_need, top_z)
        along = H.path_headroom_gate(boxes, H.path_points(start, end), radius, path_need, top_z)
        option['headroom'] = dict(contact=at_contact, path=along)
        applied.extend([at_contact, along])
        if at_contact['passed'] and along['passed']:
            kept.append(option)
        else:
            gate['reasons'].append(f"{tuple(round(float(v), 1) for v in option['direction_frame'])}: "
                                   + '; '.join(at_contact['reasons'] + along['reasons']))
    gate['options'] = kept
    gate['chosen'] = kept[0] if kept else None
    gate['passed'] = bool(kept)
    worst = min(applied, key=lambda g: (g['passed'], g['available_m'] if g['available_m'] is not None else math.inf),
                default=H.headroom_gate(boxes, start, radius, path_need, top_z))
    return dict(worst, kind='push', hand_m=contact_need['hand_m'], lift_m=0., allowance_m=contact_need['allowance_m'],
                path_needed_m=path_need['needed_m'])


def cover_feasible(env, obj_body, kind, *, radius=None, height=None, keepouts=(), floor=None, at_build=True):
    """Dry run of :func:`cover_object`: does ``kind`` fit ``obj_body`` where it stands and pass its reach
    gate (knob or handle reach, or the bowl's slide path)? Nothing is built. Returns
    ``dict(ok, kind, object, reasons, scale)``; wave-2 wiring uses it to define the objects that qualify
    for a cover before :func:`draw_hidden`."""
    if kind not in OBJECT_COVERS:
        raise ValueError(f'unknown object cover {kind!r}; choose from {OBJECT_COVERS}')
    work = env.work
    top_z = float(work['top_z'])
    ext = body_extent(env, obj_body, radius=radius, height=height)
    xy = np.asarray(ext['xy'], float)
    out = dict(ok=False, kind=kind, object=obj_body, reasons=[], scale=None)
    if kind in ('cloche_small', 'cloche_large'):
        spec = CLOCHE_SIZES['small' if kind == 'cloche_small' else 'large']
        if ext['radius'] + COVER_CLEARANCE_M > spec['radius'] - CLOCHE_WALL:
            out['reasons'].append('too wide for the cloche')
        if ext['height'] + COVER_CLEARANCE_M > spec['depth']:
            out['reasons'].append('too tall for the cloche')
        knob = np.array([xy[0], xy[1], top_z + COVER_LIFT_M + spec['depth'] + CLOCHE_WALL + .0125])
        gate = reach_gate(env, knob, floor=floor, at_build=at_build)
        out['reach'] = gate
        if not gate['passed']:
            out['reasons'].append('knob out of reach: ' + '; '.join(gate['reasons']))
        clear, n = cover_footprint_clear(env, xy, spec['radius'], keepouts)
        if not clear:
            out['reasons'].append(f'cloche footprint clashes with {n} prop(s)')
        out['headroom'] = cover_headroom(env, kind, xy, spec['radius'], object_height=ext['height'],
                                         cover_height=spec['depth'] + CLOCHE_WALL + CLOCHE_KNOB_HEIGHT)
        out['reasons'].extend(out['headroom']['reasons'])
    elif kind == 'bowl':
        scales = [bowl_scale_for(c, ext['radius'], ext['height']) for c in BOWL_COVERS]
        fitting = [s for s in scales if s is not None]
        if not fitting:
            out['reasons'].append('no bowl cover fits')
        else:
            out['scale'] = min(fitting)
            cover = BOWL_COVERS[scales.index(out['scale'])]
            R = cover['radius'] * out['scale']
            gate = slide_gate(env, xy, R, ext['radius'], keepouts=keepouts, floor=floor, at_build=at_build)
            out['reach'] = gate
            out['headroom'] = _bowl_headroom(env, gate, xy, R, out['scale'] * (_bowl_depth_at(cover, 0.) + .01))
            if not gate['passed']:
                out['reasons'].extend(gate['reasons'])
            clear, n = cover_footprint_clear(env, xy, R, keepouts)
            if not clear:
                out['reasons'].append(f'bowl footprint clashes with {n} prop(s)')
    else:
        if ext['radius'] + CUP_CLEARANCE_M > CUP_INNER_RADIUS:
            out['reasons'].append('too wide for the cup')
        if ext['height'] > CUP_HEIGHT - CUP_FLOOR - .004:
            out['reasons'].append('too tall for the cup')
        handle = np.array([xy[0], xy[1], top_z + COVER_LIFT_M + .0575])
        gate = reach_gate(env, handle, floor=floor, at_build=at_build)
        out['reach'] = gate
        if not gate['passed']:
            out['reasons'].append('handle out of reach: ' + '; '.join(gate['reasons']))
        clear, n = cover_footprint_clear(env, xy, CUP_INNER_RADIUS + .004 + .035 + .006, keepouts)
        if not clear:
            out['reasons'].append(f'cup footprint clashes with {n} prop(s)')
        out['headroom'] = cover_headroom(env, 'wide_cup', xy, CUP_INNER_RADIUS + .004 + .035 + .006, cover_height=CUP_HEIGHT)
        out['reasons'].extend(out['headroom']['reasons'])
    out['ok'] = not out['reasons']
    return out


def cover_object(env, obj_body, kind, rng, *, radius=None, height=None, keepouts=(), floor=None, at_build=True,
                 name=None):
    """Hide a task object under a cover: ``cloche_small`` / ``cloche_large`` (lifted by the knob), ``bowl``
    (an inverted RoboCasa bowl, removed by slide-and-lift) or ``wide_cup`` (the object goes inside the
    cup, which must be poured out). Returns a record with ``built`` and the reach gate; ``keepouts`` are
    task-frame shapes of the other props (for the bowl's slide path)."""
    if kind not in OBJECT_COVERS:
        raise ValueError(f'unknown object cover {kind!r}; choose from {OBJECT_COVERS}')
    work = env.work
    top_z = float(work['top_z'])
    ext = body_extent(env, obj_body, radius=radius, height=height)
    name = name or f'cover_{obj_body}'
    xy = np.asarray(ext['xy'], float)
    record = dict(kind=kind, name=name, object=obj_body, xy_world=[float(xy[0]), float(xy[1])], built=False,
                  object_radius=ext['radius'], object_height=ext['height'], placeholder=False)
    # A covered object starts at rest on the surface: a task that drops its props from a height (odd
    # parcel starts them 5 cm up) would otherwise start them inside the cover's ceiling.
    if ext['source'] == 'object':
        record['object_reposed'] = set_body_pose(env, obj_body, xy, ext['yaw'], top_z)
    if kind in ('cloche_small', 'cloche_large'):
        size = 'small' if kind == 'cloche_small' else 'large'
        spec = CLOCHE_SIZES[size]
        fit = dict(radius_ok=ext['radius'] + COVER_CLEARANCE_M <= spec['radius'] - CLOCHE_WALL,
                   height_ok=ext['height'] + COVER_CLEARANCE_M <= spec['depth'])
        record['fit'] = fit
        if not all(fit.values()):
            record['reason'] = f'{obj_body} does not fit under the {size} cloche'
            return record
        clear, n = cover_footprint_clear(env, xy, spec['radius'], keepouts)
        record['footprint_clear'] = clear
        if not clear:
            record['reason'] = f'{size} cloche footprint clashes with {n} prop(s)'
            return record
        record['headroom'] = cover_headroom(env, kind, xy, spec['radius'], object_height=ext['height'],
                                            cover_height=spec['depth'] + CLOCHE_WALL + CLOCHE_KNOB_HEIGHT)
        if not record['headroom']['passed']:
            record['reason'] = f'{size} cloche cannot be lifted here: ' + '; '.join(record['headroom']['reasons'])
            return record
        yaw = float(rng.uniform(-math.pi, math.pi)) if rng is not None else float(work['yaw'])
        built = build_cloche(env.model.worldbody, name, size, rng=rng, free=True,
                             pos=(float(xy[0]), float(xy[1]), top_z + COVER_LIFT_M), yaw=yaw, assets=env.model.asset)
        knob = np.array([xy[0], xy[1], top_z + COVER_LIFT_M + built['knob_grasp_z']])
        record.update(built=True, grasp_site=built['knob_site'], grasp_world=knob.round(4).tolist(),
                      radius=built['radius'], height=built['knob_top_z'], mass=built['mass'], yaw_world=yaw,
                      material=built['material'], material_kind=built['kind'],
                      reach=reach_gate(env, knob, floor=floor, at_build=at_build))
        env._frame_bodies = getattr(env, '_frame_bodies', []) + [name]
        env.task_spec['objects'][name] = dict(kind='cover', fixed=False, role='cover', covers=obj_body)
    elif kind == 'bowl':
        order = list(range(len(BOWL_COVERS)))
        if rng is not None:
            order = [int(i) for i in rng.permutation(len(order))]
        chosen = None
        for i in order:
            s = bowl_scale_for(BOWL_COVERS[i], ext['radius'], ext['height'])
            if s is not None:
                chosen = (BOWL_COVERS[i], s)
                break
        if chosen is None:
            record['reason'] = f'no bowl cover fits {obj_body} (radius {ext["radius"]:.3f}, height {ext["height"]:.3f})'
            return record
        cover, s = chosen
        clear, n = cover_footprint_clear(env, xy, float(cover['radius']) * s, keepouts)
        record['footprint_clear'] = clear
        if not clear:
            record['reason'] = f'bowl footprint clashes with {n} prop(s)'
            return record
        env.add_native(name, cover['asset'], [float(xy[0]), float(xy[1])], top_z, scale=s, quat_xyzw=[1., 0., 0., 0.])
        obj = env.objects[name]
        env._poses[name][2] = top_z + float(obj.top_offset[2]) + .002      # rim down on the counter
        env.task_spec['objects'][name].update(role='cover', covers=obj_body)
        R = float(cover['radius']) * s
        record.update(built=True, asset=cover['asset'], scale=s, radius=R, height=float(obj.size[2]), placeholder=False,
                      reach=slide_gate(env, xy, R, ext['radius'], keepouts=keepouts, floor=floor, at_build=at_build))
        record['headroom'] = _bowl_headroom(env, record['reach'], xy, R, float(obj.size[2]))
    else:  # wide_cup
        fit = dict(radius_ok=ext['radius'] + CUP_CLEARANCE_M <= CUP_INNER_RADIUS,
                   height_ok=ext['height'] <= CUP_HEIGHT - CUP_FLOOR - .004)
        record['fit'] = fit
        if not all(fit.values()):
            record['reason'] = f'{obj_body} does not fit inside the wide cup'
            return record
        clear, n = cover_footprint_clear(env, xy, CUP_INNER_RADIUS + .004 + .035 + .006, keepouts)
        record['footprint_clear'] = clear
        if not clear:
            record['reason'] = f'cup footprint clashes with {n} prop(s)'
            return record
        record['headroom'] = cover_headroom(env, 'wide_cup', xy, CUP_INNER_RADIUS + .004 + .035 + .006, cover_height=CUP_HEIGHT)
        if not record['headroom']['passed']:
            record['reason'] = 'wide cup cannot be poured here: ' + '; '.join(record['headroom']['reasons'])
            return record
        # the handle points toward the robot (frame -y) with a little scatter, so a side pinch is natural
        cup_yaw = _wrap(float(work['yaw']) - math.pi / 2 + (float(rng.uniform(-.6, .6)) if rng is not None else 0.))
        built = build_wide_cup(env.model.worldbody, name, rng=rng, free=True,
                               pos=(float(xy[0]), float(xy[1]), top_z + COVER_LIFT_M), yaw=cup_yaw, assets=env.model.asset)
        moved = set_body_pose(env, obj_body, xy, ext['yaw'], top_z, lift=COVER_LIFT_M + built['floor_z'])
        hx, hy, hz = built['handle_grasp']
        handle = (np.array([xy[0], xy[1], 0.]) + yaw_matrix(cup_yaw) @ np.array([hx, hy, 0.])
                  + np.array([0., 0., top_z + COVER_LIFT_M + hz]))
        record.update(built=True, grasp_site=built['handle_site'], grasp_world=handle.round(4).tolist(),
                      radius=built['handle_reach'], height=built['height'], mass=built['mass'], object_moved=moved,
                      yaw_world=cup_yaw, material=built['material'], material_kind=built['kind'],
                      reach=reach_gate(env, handle, floor=floor, at_build=at_build))
        env._frame_bodies = getattr(env, '_frame_bodies', []) + [name]
        env.task_spec['objects'][name] = dict(kind='cover', fixed=False, role='cover', covers=obj_body)
    return record


def cover_place(env, place_xy, kind, rng, *, place_half, place_yaw=0., support_z=None, keepouts=(), bounds=None,
                mode='push', floor=None, at_build=True, name=None, directions=((-1., 0.), (1., 0.), (0., 1.)),
                margin=None, max_scale=None):
    """Hide a place (a pad, card, board or pan; task-frame centre ``place_xy`` and half sizes) under a flat
    RoboCasa cover: ``plate``, ``board`` (cutting board) or ``tray``. In ``push`` mode the cover is scaled
    (at most ``max_scale``, default :data:`FLAT_COVER_MAX_SCALE`) to overhang the place by ``margin``
    (default :data:`FLAT_COVER_MARGIN_M`) and the push path is gated (:func:`push_gate`). In ``lift`` mode
    (a plate on a balance pan, pinched at its rim) the cover stays at native size by default, so it passes
    between the balance's hanger rods, and the rim nearest the robot is gated for reach. ``keepouts``
    (task-frame) must not include the place itself."""
    if kind not in PLACE_COVERS:
        raise ValueError(f'unknown place cover {kind!r}; choose from {PLACE_COVERS}')
    work = env.work
    top_z = float(work['top_z'])
    support_z = top_z + .004 if support_z is None else float(support_z)
    margin = (0. if mode == 'lift' else FLAT_COVER_MARGIN_M) if margin is None else float(margin)
    max_scale = (1. if mode == 'lift' else FLAT_COVER_MAX_SCALE) if max_scale is None else float(max_scale)
    name = name or f'cover_{kind}_{round(float(place_xy[0]) * 1000)}_{round(float(place_xy[1]) * 1000)}'
    place_half = (float(place_half[0]), float(place_half[1]))
    options = list(FLAT_COVERS[kind])
    if rng is not None:
        options = [options[int(i)] for i in rng.permutation(len(options))]
    chosen = None
    for asset in options:
        sx, sy, sz = asset['size']
        for turn in (0., math.pi / 2):
            cx, cy = (sx / 2, sy / 2) if turn == 0. else (sy / 2, sx / 2)
            scale = max(1., (place_half[0] + margin) / cx, (place_half[1] + margin) / cy) if margin > 0. or mode != 'lift' \
                else 1.
            if mode == 'lift' and margin <= 0.:
                scale = 1.
            if scale <= max_scale:
                chosen = (asset, turn, round(float(scale), 3), (cx * scale, cy * scale), sz * scale)
                break
        if chosen:
            break
    record = dict(kind=kind, name=name, place_xy=[float(place_xy[0]), float(place_xy[1])], place_half=list(place_half),
                  built=False, mode=mode, placeholder=False, margin=margin, max_scale=max_scale)
    if chosen is None:
        record['reason'] = f'no {kind} covers a {place_half} place within scale {max_scale}'
        return record
    asset, turn, scale, half, thickness = chosen
    yaw_frame = _wrap(float(place_yaw) + turn)
    yaw_world = _wrap(yaw_frame + float(work['yaw']))
    xy_world = frame_to_world_xy(env, place_xy)
    if mode == 'lift':
        record['headroom'] = cover_headroom(env, 'plate_lift', xy_world, max(half), cover_height=thickness,
                                            support_above=support_z - top_z)
        if not record['headroom']['passed']:
            record['reason'] = f'{kind} cannot be lifted off the place: ' + '; '.join(record['headroom']['reasons'])
            return record
    env.add_native(name, asset['asset'], [float(xy_world[0]), float(xy_world[1])], support_z, scale=scale,
                   quat_xyzw=_quat_xyzw(yaw_world))
    env.task_spec['objects'][name].update(role='cover', covers_place=list(record['place_xy']))
    record.update(built=True, asset=asset['asset'], scale=scale, half=[round(half[0], 4), round(half[1], 4)],
                  thickness=round(thickness, 4), yaw_frame=yaw_frame, yaw_world=yaw_world,
                  xy_world=[float(xy_world[0]), float(xy_world[1])], support_z=support_z)
    if mode == 'lift':
        rim = np.asarray(place_xy, float) + np.array([0., -1.]) * (min(half) - .005)
        record['reach'] = reach_gate(env, frame_to_world_xy(env, rim), floor=floor, at_build=at_build)
        record['reach']['rim_frame'] = [float(rim[0]), float(rim[1])]
    else:
        record['reach'] = push_gate(env, place_xy, half, yaw_frame, place_half, keepouts=keepouts, bounds=bounds,
                                    floor=floor, at_build=at_build, directions=directions)
        record['headroom'] = _push_headroom(env, record['reach'], place_xy, half, thickness)
    return record


# ---- render check (G3) ------------------------------------------------------------------------------
def body_id(env, name):
    """Model id of a task body: a registered object's root body (``{name}_main`` for RoboCasa assets) or the
    body of that name."""
    objects = getattr(env, 'objects', {}) or {}
    if name in objects:
        return env.sim.model.body_name2id(objects[name].root_body)
    return env.sim.model.body_name2id(name)


def _subtree_geoms(model, root_id):
    ids = []
    for gid in range(model.ngeom):
        b = int(model.geom_bodyid[gid])
        while b > 0:
            if b == root_id:
                ids.append(gid)
                break
            b = int(model.body_parentid[b])
    return ids


def visible_in_cameras(env, body_names, image_size=512, cameras=CAMERAS, geom_names=()):
    """Fail-closed render diff over the policy cameras: how many pixels of each body (its whole subtree)
    reach each camera at the current state.

    Every geom of a body is repainted an opaque red then an opaque green with its material detached, the
    cameras are rendered after each repaint and the differing pixels are counted; a hidden body differs
    nowhere. Repainting rather than hiding keeps MuJoCo's transparent-geom draw order untouched (alpha
    toggling changes it and false-positives on glass appliance doors). The renderer is warmed up first and
    a repeat render must agree with the first within :data:`FLICKER_TOLERANCE_PX` pixels (``deterministic``,
    with the count in ``unstable_pixels``: a GPU flicker of a few pixels, seen once in twelve kitchens,
    must not fail a scene closed); ``visible[name]`` is True when any camera saw the body. ``geom_names``
    adds single geoms checked the same way. Raises when the env has no renderer."""
    sim = env.sim
    if getattr(sim, '_render_context_offscreen', None) is None and not getattr(env, 'has_offscreen_renderer', False):
        raise RuntimeError('visible_in_cameras needs an env built with render=True')
    m = sim.model
    targets = {}
    for name in body_names:
        targets[name] = _subtree_geoms(m, body_id(env, name))
    for gname in geom_names:
        targets[f'geom:{gname}'] = [m.geom_name2id(gname)]

    def render_all():
        return {c: np.array(sim.render(width=image_size, height=image_size, camera_name=c), copy=True) for c in cameras}

    render_all()
    base, again = render_all(), render_all()
    agreement = {c: frames_agree(base[c], again[c]) for c in cameras}
    deterministic = {c: agreement[c][0] for c in cameras}
    unstable = {c: agreement[c][1] for c in cameras}
    pixels, boxes, flicker = {}, {}, {}
    for name, gids in targets.items():
        gids = np.asarray(gids, int)
        flicker[name] = {}
        if gids.size == 0:
            pixels[name] = {c: -1 for c in cameras}
            boxes[name] = {c: None for c in cameras}
            continue
        saved_rgba, saved_mat = m.geom_rgba[gids].copy(), m.geom_matid[gids].copy()
        passes = []
        try:
            m.geom_matid[gids] = -1
            for _ in range(2):
                m.geom_rgba[gids] = [1., 0., 0., 1.]
                red = render_all()
                m.geom_rgba[gids] = [0., 1., 0., 1.]
                green = render_all()
                passes.append((red, green))
        finally:
            m.geom_rgba[gids] = saved_rgba
            m.geom_matid[gids] = saved_mat
        pixels[name], boxes[name] = {}, {}
        for c in cameras:
            # A pixel counts when it differs in both colour passes: a stray GPU glitch in one pair
            # (6 px seen once on a hidden stamp) does not; the flicker count reports such pixels.
            masks = [np.any(red[c] != green[c], axis=-1) for red, green in passes]
            mask = masks[0] & masks[1]
            flicker[name][c] = int(np.count_nonzero(masks[0] ^ masks[1]))
            pixels[name][c] = int(np.count_nonzero(mask))
            if pixels[name][c]:
                rows, cols = np.nonzero(mask)
                # rows count from the top of the image as a viewer sees it (MuJoCo renders bottom-up)
                boxes[name][c] = [int(image_size - 1 - rows.max()), int(image_size - 1 - rows.min()),
                                  int(cols.min()), int(cols.max())]
            else:
                boxes[name][c] = None
    visible = {name: bool(any(n > 0 for n in per.values())) for name, per in pixels.items()}
    return dict(pixels=pixels, visible=visible, deterministic=deterministic, unstable_pixels=unstable,
                image_size=int(image_size), cameras=list(cameras), geoms={name: len(g) for name, g in targets.items()},
                bbox=boxes, flicker=flicker)


def frames_agree(a, b, tolerance=FLICKER_TOLERANCE_PX):
    """Do two renders of the same state agree? ``(True, n)`` when at most ``tolerance`` pixels differ in any
    channel (``n`` differing pixels); shape mismatches never agree."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, int(max(a.size, b.size))
    differing = int(np.count_nonzero(np.any(a != b, axis=-1))) if a.ndim >= 3 else int(np.count_nonzero(a != b))
    return differing <= int(tolerance), differing


def hidden_check(env, hidden, visible=(), image_size=512, cameras=CAMERAS, hidden_geoms=()):
    """G3 for observability: every ``hidden`` body is pixel-invisible in every policy camera, every
    ``visible`` body reaches some camera, and the renderer is deterministic. Fails closed."""
    report = visible_in_cameras(env, list(hidden) + list(visible), image_size, cameras, geom_names=hidden_geoms)
    hidden_ok = {name: not report['visible'][name] for name in hidden}
    hidden_ok.update({f'geom:{g}': not report['visible'][f'geom:{g}'] for g in hidden_geoms})
    visible_ok = {name: report['visible'][name] for name in visible}
    passed = all(report['deterministic'].values()) and all(hidden_ok.values()) and all(visible_ok.values())
    return dict(passed=bool(passed), hidden_ok=hidden_ok, visible_ok=visible_ok, **report)


# ---- look: reachable viewpoint ----------------------------------------------------------------------
def look_reachable(env, body, *, floor=None, at_build=False, radius=None, height=None, surfaces=None, decor=None):
    """For ``look``: the object stands on an open surface (not inside decor or a compartment) and lies
    within :data:`REGION_REACH_M` of a base stance beyond an accessible edge of that surface that passes
    the stance gate, so a viewpoint exists from which it is seen without moving anything."""
    sim = getattr(env, 'sim', None)
    if not at_build and sim is not None:
        pos = np.asarray(sim.data.body_xpos[body_id(env, body)], float)
        xy, z = pos[:2], float(pos[2])
    else:
        ext = body_extent(env, body, radius=radius, height=height)
        xy, z = np.asarray(ext['xy'], float), float(ext['z'])
    return spot_reachable(env, xy, z, floor=floor, surfaces=surfaces, decor=decor)


def spot_reachable(env, xy, z, *, floor=None, surfaces=None, decor=None):
    """The rule of :func:`look_reachable` for a bare world point (``xy``, ``z``): it stands on an open
    surface (not inside decor or a compartment) and a base stance beyond an accessible edge of that
    surface passes the stance gate within :data:`REGION_REACH_M` of it.

    ``surfaces`` (:func:`surface_regions`) and ``decor`` (a dict memo for :func:`_decor_at`) may be passed
    in so a caller that tests many spots on one kitchen measures the surfaces once; this is how
    :func:`place_in_annex` checks a spot before the object is moved there."""
    floor = floor or floor_map(env)
    xy = np.asarray(xy, float)[:2]
    z = float(z)
    best, reasons, surface = None, [], None
    for region in (surface_regions(env) if surfaces is None else surfaces):
        local = region_to_local(region, xy)
        x0, x1, y0, y1 = region['rect']
        if not (x0 - .02 <= local[0] <= x1 + .02 and y0 - .02 <= local[1] <= y1 + .02):
            continue
        if not (region['top_z'] - .03 <= z <= region['top_z'] + .40):
            reasons.append(f"{region['fixture']}: object z {z:.2f} is not on the top ({region['top_z']:.2f})")
            continue
        enclosed = [d['name'] for d in _decor_at(env, region['top_z'], decor)
                    if d['x0'] - .01 <= xy[0] <= d['x1'] + .01 and d['y0'] - .01 <= xy[1] <= d['y1'] + .01]
        if enclosed:
            reasons.append(f"inside {', '.join(enclosed)}")
            continue
        surface = region['fixture']
        for edge in accessible_edges(region):
            n = np.asarray(EDGES[edge], float)
            axis = 1 if edge in ('left', 'right') else 0        # the coordinate that runs along the edge
            lo, hi = (y0, y1) if axis else (x0, x1)
            point = np.array([np.clip(local[0], x0, x1), np.clip(local[1], y0, y1)])
            if edge == 'front':
                point[1] = y0
            elif edge == 'back':
                point[1] = y1
            elif edge == 'left':
                point[0] = x0
            else:
                point[0] = x1
            # The base may stand anywhere along the edge, not only straight out from the object, so a
            # stance blocked by one stool does not make the object unviewable: step along the edge
            # (:data:`STANCE_STEPS_M`, nearest first) and keep the first stance that passes. Without this
            # an annex placement that :func:`annex_regions` gated on the cell's own stance can still be
            # called unreachable here, and the instance fails G3 instead of never being built.
            failure = None
            for step in STANCE_STEPS_M:
                here = point.copy()
                here[axis] = float(point[axis]) + step
                if not lo - 1e-9 <= here[axis] <= hi + 1e-9:
                    continue
                stance_xy = region_to_world(region, here + n * STANDOFF_M)
                heading = _rot2(region['yaw']) @ (-n)
                stance_yaw = _wrap(math.atan2(heading[1], heading[0]))
                gate = stance_gate(stance_xy, stance_yaw, floor)
                distance = float(np.linalg.norm(stance_xy - xy))
                if gate['passed'] and distance <= REGION_REACH_M:
                    if best is None or distance < best['distance_m']:
                        best = dict(edge=edge, xy=[float(stance_xy[0]), float(stance_xy[1])], yaw=float(stance_yaw),
                                    distance_m=round(distance, 4), fixture=region['fixture'], step_m=round(step, 3))
                    failure = None
                    break
                if failure is None:
                    failure = (f"{region['fixture']} {edge}: stance blocked by {gate['blockers'] or 'the room bounds'}"
                               if not gate['passed'] else
                               f"{region['fixture']} {edge}: {distance:.2f} m from the stance")
            if failure is not None:
                reasons.append(failure)
    if surface is None and not reasons:
        reasons.append('object is on no counter, island or dining top')
    return dict(passed=best is not None, stance=best, surface=surface, xy=[float(xy[0]), float(xy[1])], z=z, reasons=reasons)


# ---- spec schema ------------------------------------------------------------------------------------
def draw_hidden(rng, objects, qualifying, count):
    """Draw ``count`` object names to hide, equally over qualifying and non-qualifying objects: each draw
    picks the pool by a fair coin while both still have members, then uniformly inside the pool, without
    replacement. Deterministic in ``rng``."""
    objects = list(objects)
    qualifying = set(qualifying)
    pools = [[o for o in objects if o in qualifying], [o for o in objects if o not in qualifying]]
    drawn = []
    for _ in range(int(count)):
        available = [p for p in pools if p]
        if not available:
            break
        pool = available[0] if len(available) == 1 else available[int(rng.integers(2))]
        drawn.append(pool.pop(int(rng.integers(len(pool)))))
    return drawn


def hidden_entry(object_name, mode, cover=None):
    """One ``spec['hidden']`` entry: {'object', 'mode', 'cover'}."""
    if mode not in HIDE_MODES:
        raise ValueError(f'mode {mode!r} is not one of {HIDE_MODES}')
    if mode == 'cover' and cover not in COVER_KINDS:
        raise ValueError(f'cover {cover!r} is not one of {COVER_KINDS}')
    if mode != 'cover' and cover is not None:
        raise ValueError(f'mode {mode!r} takes no cover')
    return {'object': str(object_name), 'mode': str(mode), 'cover': cover}


def validate_hidden(spec, objects=None):
    """Problems with ``spec['observability']`` and ``spec['hidden']``; empty when consistent."""
    problems = []
    level = spec.get('observability', 'visible')
    if level not in MODES + (DEV_ONLY_LEVEL,):
        problems.append(f'observability {level!r} is not one of {MODES + (DEV_ONLY_LEVEL,)}')
        return problems
    hidden = spec.get('hidden', [])
    if not isinstance(hidden, list):
        return [f'hidden must be a list, got {type(hidden).__name__}']
    allowed = LEVEL_MODES[level]
    seen = set()
    for index, entry in enumerate(hidden):
        if not isinstance(entry, dict) or set(entry) != {'object', 'mode', 'cover'}:
            problems.append(f'hidden[{index}] must have exactly the keys object, mode, cover')
            continue
        if entry['mode'] not in HIDE_MODES:
            problems.append(f"hidden[{index}]: mode {entry['mode']!r} unknown")
        elif entry['mode'] not in allowed:
            problems.append(f"hidden[{index}]: mode {entry['mode']!r} not allowed at level {level!r}")
        if entry['mode'] == 'cover' and entry['cover'] not in COVER_KINDS:
            problems.append(f"hidden[{index}]: cover {entry['cover']!r} unknown")
        if entry['mode'] != 'cover' and entry['cover'] is not None:
            problems.append(f"hidden[{index}]: mode {entry['mode']!r} takes no cover")
        if objects is not None and entry['object'] not in objects:
            problems.append(f"hidden[{index}]: object {entry['object']!r} is not a task object")
        if entry['object'] in seen:
            problems.append(f"hidden[{index}]: object {entry['object']!r} listed twice")
        seen.add(entry['object'])
    modes = {e['mode'] for e in hidden if isinstance(e, dict)}
    if level == 'look' and not modes & {'annex', 'occluder'}:
        problems.append("level 'look' needs an annex or occluder entry")
    if level == 'uncover' and 'cover' not in modes:
        problems.append("level 'uncover' needs a cover entry")
    if level == DEV_ONLY_LEVEL and not (modes & {'annex', 'occluder'} and 'cover' in modes):
        problems.append(f"level {DEV_ONLY_LEVEL!r} needs both a look entry and a cover entry")
    if level == 'visible' and hidden:
        problems.append("level 'visible' hides nothing")
    return problems


# ---- realisation at build time ----------------------------------------------------------------------
def apply_observability(env, spec, *, objects=None, places=None, keepouts=(), bounds=None, rng=None, floor=None,
                        annex_keepouts_world=(), occluder_relocate=None, annex_lift=True):
    """Realise ``spec['hidden']`` at the end of a task's ``build_task``.

    Every cover, occluder and (with ``annex_lift``, the default: the oracle fetches annexed objects with a
    top-down grasp) annex spot passes the headroom gate of :mod:`headroom`, recorded per entry under
    ``headroom`` (needed metres, the lowest overhead fixture over its column, whether it passed).

    ``objects`` maps a task object name to optional geometry overrides ``dict(radius, height)`` (needed for
    mesh bodies); ``places`` maps a place name to ``dict(xy, half, yaw, support_z, mode)`` (task frame)
    for flat covers; ``keepouts`` and ``bounds`` are task-frame scatter shapes of the task's props and its
    free area; ``annex_keepouts_world`` are world shapes to avoid on other surfaces; ``occluder_relocate`` is
    the task-frame Rect an occluded object may be moved into (see :func:`place_occluder`). The build-time
    RNG is derived from the instance seed and :data:`OBSERVE_SALT` unless ``rng`` is given. An occluder
    that cannot be placed falls back to the annex (recorded under ``fallbacks``); an annex or cover
    that cannot be realised is recorded under ``problems`` and the gate fails the instance. Every spot the
    hidden object is moved to — an annex spot, or the spot an occluded object is relocated to — must pass
    :func:`spot_reachable` first, and what the walk tried is recorded under ``annex_walk`` and the
    occluder's ``relocate_walk``. The record is stored in
    ``env.task_spec['observability']`` and returned."""
    hidden = list(spec.get('hidden') or [])
    record = dict(level=spec.get('observability', 'visible'), requested=deepcopy(hidden), realised=[],
                  hidden_bodies=[], covers=[], occluders=[], annex=[], annex_walk=[], fallbacks=[], problems=[])
    env.task_spec['observability'] = record
    if not hidden:
        return record
    problems = validate_hidden(spec)
    if problems:
        record['problems'].extend(problems)
        return record
    if rng is None:
        seed = int(env.instance.get('seed', 0)) & 0xFFFFFFFF
        rng = np.random.default_rng([seed, OBSERVE_SALT])
    floor = floor or floor_map(env)
    frusta = expected_camera_frusta(env, at_build=True)
    objects = objects or {}
    places = places or {}
    keepouts = list(keepouts)
    occupied = []               # task-frame shapes of what this function added
    annex_world = list(annex_keepouts_world)
    regions = None
    surfaces = None             # measured once: the spot reach check below walks them for every draw
    decor = {}

    def geometry(name):
        override = objects.get(name, {})
        return dict(radius=override.get('radius'), height=override.get('height'))

    def reach_at(world_xy, top_z):
        # the rule the look gate applies after reset, on the spot itself: an object standing there is
        # 0.01 m above the top, and only its xy decides whether a stance can reach it
        nonlocal surfaces
        if surfaces is None:
            surfaces = surface_regions(env)
        return spot_reachable(env, world_xy, float(top_z) + .01, floor=floor, surfaces=surfaces, decor=decor)

    def do_annex(name, entry):
        nonlocal regions
        geo = geometry(name)
        ext = body_extent(env, name, **geo)
        if regions is None:
            regions = annex_regions(env, floor=floor, frusta=frusta, keepouts_world=annex_world,
                                    object_height=ext['height'], object_radius=ext['radius'])
        if not regions:
            return None
        lift = H.headroom_needed('fetch', object_height=ext['height']) if annex_lift else None
        start = int(rng.integers(len(regions)))
        walk = []
        for k in range(len(regions)):
            region = regions[(start + k) % len(regions)]
            region = dict(region, keepouts=region['keepouts'] + _local_keepouts((), region, region['rect'], annex_world))
            diag = {}
            placed = place_in_annex(env, region, rng, radius=ext['radius'], height=ext['height'], frusta=frusta,
                                    headroom=lift, reach=reach_at, diagnostics=diag)
            walk.append(dict(region=region['name'], edge=region['edge'], attempts=diag.get('attempts', 0),
                             seen=diag.get('seen', 0), headroom_blocked=diag.get('headroom_blocked', 0),
                             reach_blocked=diag.get('reach_blocked', 0), no_spot=bool(diag.get('no_spot')),
                             reasons=list(diag.get('reasons', []))[:2], placed=bool(diag.get('placed'))))
            if placed is None:
                continue
            moved = set_body_pose(env, name, placed['xy_world'], placed['yaw_world'], region['top_z'])
            annex_world.append(S.Disc(tuple(placed['xy_world']), ext['radius']))
            placed.update(object=name, moved=moved, radius=ext['radius'], height=ext['height'])
            record['annex'].append(placed)
            record['annex_walk'].append(dict(object=name, region=region['name'], tried=walk))
            record['hidden_bodies'].append(name)
            record['realised'].append(dict(entry, realised='annex', region=region['name']))
            return placed
        record['annex_walk'].append(dict(object=name, region=None, tried=walk))
        return None

    for entry in hidden:
        name, mode, cover = entry['object'], entry['mode'], entry.get('cover')
        if mode == 'annex':
            if do_annex(name, entry) is None:
                record['problems'].append(f'{name}: no annex region holds it out of every camera')
        elif mode == 'occluder':
            geo = geometry(name)
            placed = place_occluder(env, name, rng, keepouts=keepouts + occupied, bounds=bounds, frusta=frusta,
                                    relocate_region=occluder_relocate,
                                    reach=lambda xy: reach_at(xy, env.work['top_z']), **geo)
            if placed is not None:
                occupied.append(S.Box(tuple(placed['frame_xy']), (placed['size'][0] / 2, placed['size'][1] / 2), placed['frame_yaw']))
                if placed['relocated'] is not None:
                    moved_to = world_to_frame_xy(env, placed['relocated']['to_xy_world'])
                    occupied.append(S.Disc((float(moved_to[0]), float(moved_to[1])), placed['target_radius']))
                record['occluders'].append(placed)
                record['hidden_bodies'].append(name)
                record['realised'].append(dict(entry, realised='occluder', occluder=placed['name']))
            elif do_annex(name, entry) is not None:
                record['fallbacks'].append(dict(object=name, requested='occluder', realised='annex'))
            else:
                record['problems'].append(f'{name}: no occluder blocks every camera and no annex region holds it')
        elif mode == 'cover':
            if cover in OBJECT_COVERS:
                geo = geometry(name)
                result = cover_object(env, name, cover, rng, keepouts=keepouts + occupied, floor=floor, **geo)
                if result['built']:
                    centre = world_to_frame_xy(env, result['xy_world'])
                    occupied.append(S.Disc((float(centre[0]), float(centre[1])), result['radius']))
                    record['hidden_bodies'].append(name)
            else:
                place = places.get(name)
                if place is None:
                    record['problems'].append(f'{name}: covered place has no geometry in places')
                    continue
                result = cover_place(env, place['xy'], cover, rng, place_half=place['half'], place_yaw=place.get('yaw', 0.),
                                     support_z=place.get('support_z'), keepouts=keepouts + occupied, bounds=bounds,
                                     mode=place.get('mode', 'push'), floor=floor)
                result['place'] = name
                if result['built']:
                    occupied.append(S.Box(tuple(place['xy']), tuple(result['half']), result['yaw_frame']))
            record['covers'].append(result)
            if result['built']:
                record['realised'].append(dict(entry, realised='cover', cover_body=result['name'],
                                               reach_passed=bool(result.get('reach', {}).get('passed'))))
                if not result.get('reach', {}).get('passed'):
                    why = '; '.join((result.get('headroom') or {}).get('reasons') or [])
                    record['problems'].append(f"{name}: {cover} cover placed but its reach gate failed"
                                              + (f' ({why})' if why else ''))
            else:
                record['problems'].append(f"{name}: {cover} cover not built ({result.get('reason')})")
    return record
