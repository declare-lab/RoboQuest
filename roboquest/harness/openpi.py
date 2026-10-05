"""OpenPI WebSocket protocol and RoboQuest evaluation boundary helpers.

This module intentionally depends only on NumPy at import time. ``msgpack`` and
``websockets`` are optional simulator-runtime dependencies loaded when a client
is constructed, so OpenPI and RoboCasa do not need to share an environment.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

# From lerobot_export.py (the training-data exporter): the native action size, the camera feature
# names and the 16-D state the openpi RoboQuest policies are trained on.
ACTION_DIM = 12
STATE_DIM = 16

CAMERA_TO_FEATURE = {
    "robot0_agentview_left": "image",
    "robot0_eye_in_hand": "wrist_image",
    "robot0_agentview_right": "right_image",
}


def _normalize_quaternion(quaternion) -> np.ndarray:
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    return value / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz, scalar = quaternion[:3], quaternion[3]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + scalar * vector)


def proprio_to_state(proprio: dict) -> np.ndarray:
    """Return the RoboCasa/openpi PandaOmron 16-D state in the base frame."""
    eef_position = np.asarray(proprio["eef_position_world_m"], dtype=float)
    base_position = np.asarray(proprio["base_position_world_m"], dtype=float)
    if eef_position.shape != (3,) or base_position.shape != (3,):
        raise ValueError("EEF and base positions must be 3-D")
    eef_quaternion = _normalize_quaternion(proprio["eef_quaternion_xyzw"])
    base_quaternion = _normalize_quaternion(proprio["base_quaternion_xyzw"])
    base_inverse = np.r_[-base_quaternion[:3], base_quaternion[3]]
    relative_position = _quat_rotate(base_inverse, eef_position - base_position)
    relative_quaternion = _normalize_quaternion(_quat_multiply(base_inverse, eef_quaternion))
    joints = np.asarray(proprio["joint_positions"], dtype=float)
    if joints.ndim != 1 or len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError("joint_positions must end with two finite gripper joints")
    state = np.concatenate([relative_position, relative_quaternion, base_position, base_quaternion, joints[-2:]])
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"constructed state has invalid shape or values: {state.shape}")
    return state.astype(np.float32)


ACTION_HORIZON = 20
REQUEST_SEED_KEY = "__openpi_rng_seed__"


class ClientSetupError(RuntimeError):
    """The simulator process cannot construct the protocol client."""


class PolicyServerError(RuntimeError):
    """The policy server reported an inference/model failure."""


class PolicyOutputError(ValueError):
    """The policy server returned a malformed joint-model response."""


def pack_array(obj: Any) -> Any:
    """MessagePack adapter identical to ``openpi_client.msgpack_numpy``."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict) -> Any:
    """Decode NumPy values emitted by OpenPI's MessagePack policy server."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class OpenPIWebsocketClient:
    """Small synchronous client for OpenPI's uncompressed MessagePack protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, *, timeout_s: float = 120.0):
        try:
            import msgpack
            import websockets.sync.client
        except ImportError as exc:
            raise ClientSetupError(
                "OpenPI evaluation requires optional packages msgpack and websockets; "
                "rerun scripts/bootstrap_agent.sh"
            ) from exc
        self._msgpack = msgpack
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None and not uri.rsplit(":", 1)[-1].isdigit():
            uri += f":{port}"
        deadline = time.monotonic() + float(timeout_s)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._ws = websockets.sync.client.connect(
                    uri, compression=None, max_size=None, open_timeout=min(10.0, timeout_s), ping_interval=None,
                )
                metadata = self._ws.recv(timeout=max(1.0, deadline - time.monotonic()))
                if isinstance(metadata, str):
                    raise PolicyServerError(f"OpenPI server returned text instead of metadata: {metadata}")
                self.metadata = msgpack.unpackb(metadata, object_hook=unpack_array)
                return
            except (OSError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"OpenPI server at {uri} was unavailable for {timeout_s:g}s") from last_error

    def infer(self, observation: dict, *, timeout_s: float = 120.0) -> dict:
        payload = self._msgpack.packb(observation, default=pack_array)
        self._ws.send(payload)
        response = self._ws.recv(timeout=timeout_s)
        if isinstance(response, str):
            raise PolicyServerError(f"OpenPI inference server error:\n{response}")
        try:
            decoded = self._msgpack.unpackb(response, object_hook=unpack_array)
        except Exception as exc:
            raise PolicyOutputError("OpenPI response is not valid MessagePack") from exc
        if not isinstance(decoded, dict):
            raise PolicyOutputError("OpenPI response must be a mapping")
        return decoded

    def close(self) -> None:
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def policy_seed(evaluation_seed: int, instance_id: str, inference_index: int) -> int:
    """Stable uint32 request seed, independent of batch ordering and resume."""
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, (int, np.integer)):
        raise ValueError("evaluation_seed must be an integer")
    if isinstance(inference_index, bool) or not isinstance(inference_index, (int, np.integer)) or inference_index < 0:
        raise ValueError("inference_index must be a nonnegative integer")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id must be a nonempty string")
    material = f"roboquest-openpi-v1\0{int(evaluation_seed)}\0{instance_id}\0{int(inference_index)}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def make_policy_observation(
    packet: dict,
    request_seed: int,
    *,
    image_size: int = 256,
    prompt_override: str | None = None,
) -> dict:
    """Project one public RoboQuest packet into the OpenPI policy's input keys."""
    prompt = prompt_override if prompt_override is not None else packet.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("public observation has no nonempty goal prompt")
    rgb = packet.get("rgb")
    if not isinstance(rgb, dict):
        raise ValueError("public observation has no RGB mapping")
    result: dict[str, Any] = {
        "observation/state": np.asarray(proprio_to_state(packet["proprio"]), dtype=np.float32),
        "prompt": prompt,
        REQUEST_SEED_KEY: np.uint32(request_seed),
    }
    for camera, feature in CAMERA_TO_FEATURE.items():
        image = np.asarray(rgb.get(camera))
        if image.shape != (image_size, image_size, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"camera {camera} has shape/dtype {image.shape}/{image.dtype}, "
                f"expected {(image_size, image_size, 3)}/uint8"
            )
        result[f"observation/{feature}"] = np.ascontiguousarray(image)
    return result


def validate_policy_response(
    response: dict,
    *,
    action_horizon: int = ACTION_HORIZON,
    require_subtask: bool = True,
) -> tuple[np.ndarray, str | None]:
    """Require one finite action chunk and, for joint policies, a predicted subtask."""
    if not isinstance(response, dict):
        raise PolicyOutputError("OpenPI response must be a mapping")
    try:
        actions = np.asarray(response["actions"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyOutputError("OpenPI response has no numerical actions") from exc
    if actions.shape != (action_horizon, ACTION_DIM):
        raise PolicyOutputError(f"OpenPI actions have shape {actions.shape}, expected {(action_horizon, ACTION_DIM)}")
    if not np.isfinite(actions).all():
        raise PolicyOutputError("OpenPI actions contain non-finite values")
    subtask = response.get("subtask")
    if not isinstance(subtask, str) or not subtask.strip():
        if require_subtask:
            raise PolicyOutputError("OpenPI response has no nonempty subtask")
        return actions.copy(), None
    return actions.copy(), subtask.strip()


def sanitize_native_action(action: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert a continuous policy row to RoboQuest's native hybrid contract."""
    raw = np.asarray(action, dtype=np.float32)
    if raw.shape != (ACTION_DIM,) or not np.isfinite(raw).all():
        raise ValueError(f"native policy action must be finite with shape {(ACTION_DIM,)}")
    executed = raw.copy()
    base_mode = bool(raw[11] >= 0)
    executed[6] = 1.0 if raw[6] >= 0 else -1.0
    executed[10] = 0.0
    executed[11] = 1.0 if base_mode else -1.0
    before_clip = executed.copy()
    if base_mode:
        executed[:6] = 0.0
        executed[7:10] = np.clip(executed[7:10], -0.5, 0.5)
        active = slice(7, 10)
    else:
        executed[7:10] = 0.0
        executed[:3] = np.clip(executed[:3], -0.5, 0.5)
        executed[3:6] = np.clip(executed[3:6], -0.3, 0.3)
        active = slice(0, 6)
    clipped_mask = before_clip[active] != executed[active]
    diagnostics = {
        "mode": "base" if base_mode else "arm",
        "clipped_elements": int(np.count_nonzero(clipped_mask)),
        "active_elements": int(executed[active].size),
        "raw_outside_native_range": int(np.count_nonzero(np.abs(raw) > 1.0)),
    }
    return executed, diagnostics
