"""Rigid-body resting pose of the stand: which feet touch, how far the top tilts, which foot hangs.

A shim lying flat under a foot lengthens that leg by its thickness, so every
configuration (short legs, shim stacks) is a set of four effective leg lengths.
The stand rests on the three feet whose plane has the centre-of-mass line of
action inside their triangle while the fourth foot stays above the ground; when
all four feet are coplanar it rests on all four. Two adjacent short legs keep
the feet coplanar (the top tilts about one axis, no foot hangs); one short
corner leg leaves two feasible triples that share the diagonal through the
short leg's neighbours, and the centre of mass (biased toward the short corner,
and lowered by the tilt) picks the one that tips onto the short leg, leaving
the diagonally opposite *long* leg hanging by the delta. The tilt about the
diagonal is larger than the one-axis tilt for the same delta: the lever is the
half diagonal instead of the leg spacing.

No MuJoCo here; ``WobblyStand.physics_gate`` checks that the simulated stand
agrees with this model within 10 %.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from roboquest.stand.geometry import LEGS, LEG_SIGNS, LEG_CENTRE, FOOT_RADIUS, TOP_THICKNESS, STAND_HEIGHT, com_bias_xy, composite_inertial, foot_centre, leg_length, stand_parts

COPLANAR_TOL = 1e-6     # m: feet closer than this to a plane count as on it
MARGINAL_BARY = 1e-3    # barycentric coordinate below which the support is called marginal
LEVEL_TILT_DEG = .2     # a configuration is "level" (for the gate) below this
WRONG_TILT_DEG = .6     # and "clearly not level" above this


def stand_com(short_legs, delta_mm):
    """Centre of mass in the stand frame, including the deliberate bias toward the low side."""
    bias = com_bias_xy(short_legs)
    return np.asarray(composite_inertial([p[1:] for p in stand_parts(short_legs, delta_mm)],
                                         com_shift=(bias[0], bias[1], 0.))['com'], float)


def effective_leg_lengths(short_legs, delta_mm, shims_under_mm=None):
    """Leg length plus the shim stack (mm) under that foot."""
    shims_under_mm = shims_under_mm or {}
    return {leg: leg_length(leg, short_legs, delta_mm) + float(shims_under_mm.get(leg, 0.)) / 1000. for leg in LEGS}


def feet_points(lengths):
    """Glide centres in the stand frame; the ground plane lies FOOT_RADIUS below the supporting centres' plane."""
    return {leg: np.array([*foot_centre(leg, (), 0.)[:2], -TOP_THICKNESS - lengths[leg] + FOOT_RADIUS]) for leg in LEGS}


def _rotation_to_up(normal):
    """Rotation matrix taking the body-frame unit ``normal`` to world +z (minimal rotation)."""
    z = np.array([0., 0., 1.])
    c = float(np.clip(normal @ z, -1., 1.))
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def resting_pose(lengths, com):
    """Resting configuration for effective leg lengths (m) and a body-frame centre of mass.

    Returns tilt_deg, the body-frame up normal, the supporting legs, the gap (m) under every foot, the
    rotation (world from body, no yaw) and z0 (body origin height above the ground), plus ``coplanar`` and
    ``marginal`` flags. ``marginal`` means the centre of mass sits on an edge of the best support triangle.
    """
    P = feet_points(lengths)
    com = np.asarray(com, float)
    candidates = []
    for triple in itertools.combinations(LEGS, 3):
        a, b, c = (P[t] for t in triple)
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        fourth = next(l for l in LEGS if l not in triple)
        gap = float((P[fourth] - a) @ n)       # positive: the fourth foot hangs above the ground
        if gap < -COPLANAR_TOL:
            continue                            # the fourth foot would be below the ground
        q = com - float((com - a) @ n) * n      # the centre of mass's line of action meets the feet plane
        uv = np.linalg.lstsq(np.stack([b - a, c - a], axis=1), q - a, rcond=None)[0]
        bary = (1. - uv[0] - uv[1], float(uv[0]), float(uv[1]))
        candidates.append(dict(legs=triple, normal=n, gap=max(gap, 0.), fourth=fourth, min_bary=float(min(bary))))
    best = max(candidates, key=lambda cand: cand['min_bary'])
    coplanar = all(cand['gap'] <= COPLANAR_TOL for cand in candidates) and len(candidates) == 4
    n = best['normal']
    rotation = _rotation_to_up(n)
    support = list(LEGS) if coplanar else list(best['legs'])
    gaps = {leg: 0. for leg in LEGS}
    if not coplanar:
        gaps[best['fourth']] = best['gap']
    z0 = -float((rotation @ P[support[0]])[2]) + FOOT_RADIUS
    return dict(tilt_deg=math.degrees(math.acos(float(np.clip(n[2], -1., 1.)))), normal=n.tolist(),
                support=support, gaps=gaps, rotation=rotation, z0=z0, coplanar=bool(coplanar),
                marginal=bool(best['min_bary'] <= MARGINAL_BARY), min_bary=best['min_bary'])


def configuration_pose(short_legs, delta_mm, shims_under_mm=None):
    return resting_pose(effective_leg_lengths(short_legs, delta_mm, shims_under_mm), stand_com(short_legs, delta_mm))


def geometric_tilt_deg(axes, delta_mm):
    """Unshimmed tilt for a factor cell (the same for every leg choice within the cell)."""
    short = ['sw', 'se'] if axes == 'one' else ['ne']
    return configuration_pose(short, delta_mm)['tilt_deg']


def level_pose_z0():
    """Body origin height above the counter when all four feet rest on their (shimmed) supports, level."""
    return STAND_HEIGHT


def opposite_legs(short_legs):
    """The 'wrong' legs for a shim: the diagonal opposite of a short corner, or the long pair opposite a short side."""
    short = set(short_legs)
    if len(short) == 1:
        sx, sy = LEG_SIGNS[next(iter(short))]
        return [leg for leg in LEGS if LEG_SIGNS[leg] == (-sx, -sy)]
    return [leg for leg in LEGS if leg not in short]


def shim_outcomes(short_legs, delta_mm, plates_mm):
    """Every way of putting the available plates under the legs, and what the stand does then.

    ``plates_mm`` lists the plates in the scene. Returns the unshimmed pose, the exact fix (one matching plate
    per short leg), the worst single-plate wrong-thickness fix, the fix applied to the wrong legs, and every
    plate assignment (stacks allowed) that levels the stand.
    """
    short_legs = list(short_legs)
    delta_mm = float(delta_mm)
    plates = [float(p) for p in plates_mm]
    unshimmed = configuration_pose(short_legs, delta_mm)
    exact_under = {leg: delta_mm for leg in short_legs}
    exact = configuration_pose(short_legs, delta_mm, exact_under)
    wrong_single = []
    seen = set()
    for chosen in itertools.permutations(range(len(plates)), len(short_legs)):
        thickness = tuple(plates[i] for i in chosen)
        if thickness in seen or all(t == delta_mm for t in thickness):
            continue
        seen.add(thickness)
        pose = configuration_pose(short_legs, delta_mm, dict(zip(short_legs, thickness)))
        wrong_single.append(dict(under=dict(zip(short_legs, thickness)), tilt_deg=pose['tilt_deg']))
    wrong_legs = opposite_legs(short_legs)
    wrong_leg = configuration_pose(short_legs, delta_mm, {leg: delta_mm for leg in wrong_legs})
    level = []
    places = [None] + list(LEGS)
    for assignment in itertools.product(places, repeat=len(plates)):
        under = {}
        for plate, leg in zip(plates, assignment):
            if leg is not None:
                under[leg] = under.get(leg, 0.) + plate
        if not under:
            continue
        key = tuple(sorted(under.items()))
        if any(row['key'] == key for row in level):
            continue
        pose = configuration_pose(short_legs, delta_mm, under)
        if pose['tilt_deg'] <= LEVEL_TILT_DEG:
            level.append(dict(key=key, under=under, tilt_deg=pose['tilt_deg']))
    return dict(
        unshimmed=dict(tilt_deg=unshimmed['tilt_deg'], support=unshimmed['support'], marginal=unshimmed['marginal'],
                       coplanar=unshimmed['coplanar'],
                       hanging={leg: gap for leg, gap in unshimmed['gaps'].items() if gap > COPLANAR_TOL}),
        exact=dict(under=exact_under, tilt_deg=exact['tilt_deg'], coplanar=exact['coplanar']),
        wrong_single=wrong_single,
        wrong_single_min_tilt_deg=min((row['tilt_deg'] for row in wrong_single), default=None),
        wrong_leg=dict(under={leg: delta_mm for leg in wrong_legs}, tilt_deg=wrong_leg['tilt_deg'],
                       coplanar=wrong_leg['coplanar']),
        level_assignments=[dict(under=row['under'], tilt_deg=row['tilt_deg']) for row in level],
        plates_mm=plates)


def leg_spacing():
    return 2 * LEG_CENTRE
