"""Proprioception-only Cartesian servo over the calibrated seven-dimensional OSC API."""

import numpy as np
from scipy.spatial.transform import Rotation


LABELS = ("x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z", "gripper")
# A command envelope, not a claim that every pose in this box is reachable.
LOW = np.array([-0.85, -0.95, 0.75] + [-1.] * 6 + [0.])
HIGH = np.array([0.75, 0.35, 1.65] + [1.] * 6 + [1.])
MAX_STEP = (0.004,) * 3 + (0.025,) * 6 + (0.05,)
SETTLE_STEPS = 20


def command_bounds(scene="reduced"):
    """Keep the same envelope relative to each scene's fixed robot spawn.

    Both robots face world +y. Stock moves the spawn from [0, -.72, 0]
    to [1.02, -1.05, 0], so only the position coordinates are translated.
    This is a command limit, not a reachability or collision certificate.
    """
    offsets = {"reduced": (0., 0., 0.), "stock": (1.02, -0.33, 0.)}
    if scene not in offsets:
        raise ValueError("scene must be reduced or stock")
    low, high = LOW.copy(), HIGH.copy()
    low[:3] += offsets[scene]
    high[:3] += offsets[scene]
    return low, high


def encode_rotation(matrix):
    """First two COLUMNS of a rotation matrix, concatenated (not row-major)."""
    return np.asarray(matrix)[:, :2].T.reshape(6).copy()


def decode_rotation(values):
    """Project an interpolated rot6d basis to SO(3), rejecting singular paths."""
    values = np.asarray(values, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("rot6d requires six finite numbers")
    first, second = values[:3], values[3:]
    length = np.linalg.norm(first)
    if length < 0.2:
        raise ValueError("rot6d first column is near zero; split the rotation")
    x = first / length
    y = second - np.dot(x, second) * x
    length = np.linalg.norm(y)
    if length < 0.2:
        raise ValueError("rot6d columns are near parallel; split the rotation")
    y /= length
    return np.column_stack((x, y, np.cross(x, y)))


def validate_waypoints(waypoints):
    """Upstream pre_check: reject invalid final bases and ill-conditioned chords."""
    try:
        rotations = [decode_rotation(row[3:9]) for row in waypoints]
        final = waypoints[-1, 3:9]
        if not np.allclose(final, encode_rotation(rotations[-1]), atol=2e-3, rtol=0):
            return "rotation targets must be two orthonormal columns; supply all six components"
        for before, after in zip(rotations, rotations[1:]):
            if Rotation.from_matrix(after @ before.T).magnitude() > 0.15:
                return "rotation interpolation is too fast near a singularity; split the rotation"
    except ValueError as exc:
        return str(exc)
    return None


def _bounded_norm(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-12))


class CartesianServo:
    """Track targets from achieved grip-site pose; never integrate requested deltas.

    The Panda's sign-only actuator changes its normalized opening command by
    0.1 per control tick. Track that command independently of measured aperture,
    so contact does not integrate a partial-opening request all the way closed.
    """

    def __init__(self, scene="reduced"):
        self.low, self.high = command_bounds(scene)
        self.reference_rotation = None
        self.opening_command = 1.0  # DrawerAdapter.reset ends with ten open commands.
        self.last_target = None

    def measured(self, proprio):
        rotation = Rotation.from_quat(proprio["eef_quaternion_xyzw"]).as_matrix()
        if self.reference_rotation is None:
            self.reference_rotation = rotation.copy()
        fingers = proprio["gripper_joint_positions_m"]
        aperture = float(np.clip((fingers[0] - fingers[1]) / 0.08, 0, 1))
        relative = rotation @ self.reference_rotation.T
        return np.r_[proprio["eef_position_world_m"], encode_rotation(relative), aperture]

    def action(self, target, proprio):
        target = np.asarray(target, dtype=float)
        if target.shape != (10,) or not np.isfinite(target).all():
            raise ValueError("Cartesian target must contain ten finite numbers")
        if np.any(target < self.low - 1e-12) or np.any(target > self.high + 1e-12):
            raise ValueError("Cartesian target exceeds command bounds")
        desired_relative = decode_rotation(target[3:9])
        measured = self.measured(proprio)
        current_rotation = Rotation.from_quat(proprio["eef_quaternion_xyzw"]).as_matrix()
        desired_rotation = desired_relative @ self.reference_rotation
        dp = _bounded_norm(0.8 * (target[:3] - measured[:3]), 0.025)
        dr = _bounded_norm(0.5 * Rotation.from_matrix(
            desired_rotation @ current_rotation.T).as_rotvec(), 0.15)
        # Nearest available setpoint, half steps rounded upward.
        opening = np.floor(target[-1] * 10 + 0.5 + 1e-10) / 10
        direction = float(-np.sign(round(opening - self.opening_command, 8)))
        self.opening_command = float(np.clip(self.opening_command - 0.1 * direction, 0, 1))
        self.last_target = target.copy()
        return np.r_[dp, dr, direction]

    def tracking_error(self, measured):
        if self.last_target is None:
            return np.zeros(3)
        desired = decode_rotation(self.last_target[3:9])
        current = decode_rotation(measured[3:9])
        return np.array([np.linalg.norm(self.last_target[:3] - measured[:3]),
                         Rotation.from_matrix(desired @ current.T).magnitude(),
                         abs(self.last_target[-1] - measured[-1])])
