"""Select an NVIDIA physical GPU's EGL device before renderer imports.

``configure_egl(k)`` maps nvidia-smi index ``k`` to the index MuJoCo's EGL backend expects
(``MUJOCO_EGL_DEVICE_ID``), because the two enumerations differ on multi-GPU hosts. Three
matching methods, tried in order, or forced with ``ROBOQUEST_EGL_MATCH=drm|cuda|order``:

- ``drm``: the EGL device's DRM render node (or card node) resolves in ``/sys/class/drm`` to the
  GPU's PCI address. Exact; needs the nvidia-drm module and readable ``/dev/dri`` nodes.
- ``cuda``: the EGL device reports its CUDA device id (``EGL_NV_device_cuda``); CUDA numbers
  identical GPUs in PCI bus order, so it is compared with the GPU's rank by PCI address.
  Used on hosts without DRM nodes, such as headless compute nodes.
- ``order``: last resort when EGL exposes as many NVIDIA devices as nvidia-smi lists: assume
  the same order.
"""

import ctypes
import os
from pathlib import Path
import subprocess

EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_CUDA_DEVICE_NV = 0x323A
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377


def _prepare_environment():
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"


def nvidia_gpus():
    """nvidia-smi's GPUs as ``[{'index', 'pci'}]`` with the 12-character lower-case PCI address."""
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], text=True)
    rows = []
    for line in text.strip().splitlines():
        index, pci = [part.strip() for part in line.split(",")]
        rows.append(dict(index=int(index), pci=pci.lower()[-12:]))
    return rows


def egl_devices():
    """EGL's devices as ``[{'index', 'render_node', 'drm_file', 'cuda_device', 'pci'}]`` (fields None when unknown)."""
    _prepare_environment()
    from mujoco.egl import egl_ext as egl

    query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        egl.eglGetProcAddress(b"eglQueryDeviceStringEXT"))
    query_attrib = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(
        egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
    rows = []
    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        row = dict(index=index, render_node=None, drm_file=None, cuda_device=None, pci=None)
        for attribute, key in ((EGL_DRM_RENDER_NODE_FILE_EXT, "render_node"), (EGL_DRM_DEVICE_FILE_EXT, "drm_file")):
            try:
                raw = query_string(device, attribute)
            except Exception:  # noqa: BLE001
                raw = None
            if raw:
                row[key] = raw.decode()
        value = ctypes.c_ssize_t(-1)
        try:
            if query_attrib(device, EGL_CUDA_DEVICE_NV, ctypes.byref(value)) and value.value >= 0:
                row["cuda_device"] = int(value.value)
        except Exception:  # noqa: BLE001
            pass
        for node in (row["render_node"], row["drm_file"]):
            if node:
                try:
                    row["pci"] = (Path("/sys/class/drm") / Path(node).name / "device").resolve().name[-12:]
                    break
                except OSError:
                    pass
        rows.append(row)
    return rows


def match_egl_device(physical_gpu, gpus=None, devices=None, method=None):
    """The EGL device row for nvidia-smi index ``physical_gpu`` and the method that found it."""
    gpus = nvidia_gpus() if gpus is None else gpus
    devices = egl_devices() if devices is None else devices
    by_index = {g["index"]: g for g in gpus}
    if physical_gpu not in by_index:
        raise RuntimeError(f"nvidia-smi lists no GPU {physical_gpu}; it lists {sorted(by_index)}")
    pci = by_index[physical_gpu]["pci"]
    methods = (method,) if method else ("drm", "cuda", "order")
    for candidate in methods:
        if candidate == "drm":
            for row in devices:
                if row["pci"] == pci:
                    return row, "drm"
        elif candidate == "cuda":
            rank = sorted(g["pci"] for g in gpus).index(pci)
            for row in devices:
                if row["cuda_device"] == rank:
                    return row, "cuda"
        elif candidate == "order":
            nvidia_rows = [row for row in devices if row["cuda_device"] is not None or row["render_node"] or row["drm_file"]]
            nvidia_rows = nvidia_rows or devices
            if len(nvidia_rows) == len(gpus):
                return nvidia_rows[physical_gpu], "order"
    table = "; ".join(f"egl {r['index']}: node {r['render_node'] or r['drm_file']} cuda {r['cuda_device']} pci {r['pci']}"
                      for r in devices) or "no EGL devices"
    raise RuntimeError(f"No EGL device matched GPU {physical_gpu} ({pci}) by {'/'.join(methods)}. EGL reports: {table}. "
                       f"Force a method with ROBOQUEST_EGL_MATCH=cuda|order.")


def configure_egl(physical_gpu):
    """Point MuJoCo's EGL backend at nvidia-smi GPU ``physical_gpu``; call before MuJoCo is imported.

    Returns ``dict(physical_gpu, egl_index, pci, render_node, method)``.
    """
    _prepare_environment()
    gpus = nvidia_gpus()
    row, method = match_egl_device(int(physical_gpu), gpus=gpus, method=os.environ.get("ROBOQUEST_EGL_MATCH") or None)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(row["index"])
    pci = next(g["pci"] for g in gpus if g["index"] == int(physical_gpu))
    return dict(physical_gpu=int(physical_gpu), egl_index=row["index"], pci=pci, render_node=row["render_node"], method=method)
