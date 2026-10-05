"""Camera-frustum visibility tests for hiding places. Pure geometry; the caller passes camera poses.

MuJoCo cameras look along their own -z axis with +y up; ``cam_xmat`` is the
world-from-camera rotation. A point is *seen* by a camera when it lies inside
the pinhole frustum (no occlusion reasoning: this is the conservative test used
to choose `open` hiding places, and the pixel-identity gate G3 confirms the
choice on the rendered images).
"""
import math

import numpy as np


def camera_frusta(sim, cameras, aspect=1.):
    """Frustum records for named cameras from a compiled simulator (after forward kinematics)."""
    frusta = []
    for name in cameras:
        cid = sim.model.camera_name2id(name)
        frusta.append(dict(name=name, pos=np.asarray(sim.data.cam_xpos[cid], float).copy(),
                           rot=np.asarray(sim.data.cam_xmat[cid], float).reshape(3, 3).copy(),
                           fovy=math.radians(float(sim.model.cam_fovy[cid])), aspect=float(aspect)))
    return frusta


def camera_coords(frustum, point):
    return frustum['rot'].T @ (np.asarray(point, float) - frustum['pos'])


def point_in_frustum(frustum, point, margin_rad=0.):
    pc = camera_coords(frustum, point)
    depth = -pc[2]
    if depth <= 1e-6:
        return False
    ty = math.tan(frustum['fovy'] / 2 + margin_rad)
    return abs(pc[1]) <= depth * ty and abs(pc[0]) <= depth * ty * frustum['aspect']


def project(frustum, point, width, height):
    """Pixel coordinates (u right, v down) of a world point, or None when behind the camera."""
    pc = camera_coords(frustum, point)
    depth = -pc[2]
    if depth <= 1e-6:
        return None
    f = (height / 2) / math.tan(frustum['fovy'] / 2)
    return (width / 2 + f * pc[0] / depth, height / 2 - f * pc[1] / depth)


def box_corners(centre, half_xyz):
    c = np.asarray(centre, float)
    h = np.asarray(half_xyz, float)
    return [c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def seen_by_any(frusta, points, margin_rad=0.):
    """True when any of the points is inside any camera frustum."""
    return any(point_in_frustum(f, p, margin_rad) for f in frusta for p in points)


def relative_bearing(base_xy, base_yaw, point_xy):
    """Signed angle (rad) from the robot's forward direction to a point; |angle| > pi/2 is behind."""
    dx, dy = float(point_xy[0]) - float(base_xy[0]), float(point_xy[1]) - float(base_xy[1])
    return (math.atan2(dy, dx) - base_yaw + math.pi) % (2 * math.pi) - math.pi


def rank_unseen_spots(spots, frusta, base_xy, base_yaw, half_xyz, margin_rad=math.radians(3.), clearance_m=.06):
    """Spots whose object box (plus clearance) is outside every frustum, behind-first then beside.

    Each spot needs ``xy`` and ``top_z``; ``half_xyz`` is the object's half extent.
    Returns (unseen spots in preference order, per-spot report).
    """
    report = []
    unseen = []
    for spot in spots:
        centre = np.r_[spot['xy'], spot['top_z'] + half_xyz[2]]
        points = box_corners(centre, np.asarray(half_xyz, float) + clearance_m)
        seen = seen_by_any(frusta, points, margin_rad)
        bearing = relative_bearing(base_xy, base_yaw, spot['xy'])
        band = 'behind' if abs(bearing) > math.radians(110.) else ('beside' if abs(bearing) > math.radians(60.) else 'front')
        report.append(dict(xy=list(spot['xy']), counter=spot.get('counter'), seen=bool(seen), band=band,
                           bearing_deg=round(math.degrees(bearing), 1)))
        if not seen:
            unseen.append(({'behind': 0, 'beside': 1, 'front': 2}[band], len(unseen), spot, band))
    unseen.sort(key=lambda row: (row[0], row[1]))
    return [dict(spot, band=band) for _, _, spot, band in unseen], report
