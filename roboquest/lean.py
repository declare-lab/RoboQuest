"""Smaller kitchen workers, physics untouched.

A RoboCasa kitchen worker holds about 2.3 GB of host memory, of which physics
is 1.5 MB (``mjData``): the compiled model is 0.58 GB (0.45 GB of it texture
pixels) and about 1.4 GB is the compiler state MuJoCo keeps after every compile
so that ``mj_saveLastXML`` can reproduce the scene. Two measures, verified on
one instance of every task: a SHA-256 over ``qpos``/``qvel`` at each of 150
ticks identical to the untouched build, identical scores and teleport
certificates.

* :func:`flatten_asset_textures`: an XML processor for **headless** builds
  only. Every texture loaded from the RoboCasa/robosuite asset packs becomes a
  4x4 flat builtin of the same name and type, so material and texture ids stay
  valid and task-made textures (the stamps papers, the Submit label) are kept.
  Model buffer 578 MB -> 131 MB, compiles faster. Never for rendered builds:
  the policy, the hidden-state render check (G3) and the asset manifest need
  the real textures. Physics never reads textures.
* :func:`release_compiler_state`: frees MuJoCo's retained last-XML state and
  returns the heap to the OS (about 0.93 GB per kitchen). Afterwards
  ``sim.model.get_xml()`` raises ``mujoco.FatalError('No XML model loaded')``
  until the next compile, so :meth:`RoboQuestKitchen.release_compiler_state`
  captures the saved XML first and :meth:`RoboQuestKitchen.scene_xml` serves
  the copy. Disabling MuJoCo's asset cache was tried and changed nothing.

Measured per worker: headless 2.1-2.7 GB -> 0.82-0.91 GB, rendered (release
only) 2.37 -> 1.48 GB. The renderer's 1.6 GB lives in GPU memory either way.
"""
import ctypes
import ctypes.util
import glob
import os
import xml.etree.ElementTree as ET

import mujoco

ASSET_TEXTURE_MARKERS = ('/robocasa/', '/robosuite/')   # texture files that come from the asset packs
PLACEHOLDER = {'builtin': 'flat', 'rgb1': '0.5 0.5 0.5', 'width': '4', 'height': '4'}


def is_asset_texture(texture):
    """True for a ``<texture>`` element whose image file(s) come from the asset packs."""
    files = [value for key, value in texture.attrib.items() if key.startswith('file')]
    return any(marker in path for path in files for marker in ASSET_TEXTURE_MARKERS)


def flatten_asset_textures(xml):
    """robosuite XML processor: asset-pack textures become flat 4x4 builtins; everything else is kept.

    Names and types survive, so materials keep their references and the compiled
    model has the same ``ntex``/``nmat`` and the same ids as the textured build.
    """
    root = ET.fromstring(xml)
    for asset in root.iter('asset'):
        for texture in asset.findall('texture'):
            if is_asset_texture(texture):
                kept = {key: texture.get(key) for key in ('name', 'type') if texture.get(key) is not None}
                texture.attrib.clear()
                texture.attrib.update(kept)
                texture.attrib.update(PLACEHOLDER)
    return ET.tostring(root, encoding='unicode')


_LIBMUJOCO = None      # ctypes handle of the bindings' own libmujoco; False once looked up and missing


def libmujoco():
    """The MuJoCo shared library the Python bindings load, as a ctypes handle (None when not found).

    The bindings expose ``mj_saveLastXML`` but not ``mj_freeLastXML``; loading the
    same library file again returns the already-loaded copy, so its global
    last-XML state is the one the bindings filled.
    """
    global _LIBMUJOCO
    if _LIBMUJOCO is None:
        _LIBMUJOCO = False
        here = os.path.dirname(mujoco.__file__)
        candidates = glob.glob(os.path.join(here, 'libmujoco*.so*')) + glob.glob(os.path.join(here, 'libmujoco*.dylib'))
        for path in sorted(candidates):
            try:
                lib = ctypes.CDLL(path)
                lib.mj_freeLastXML.restype = None
                lib.mj_freeLastXML.argtypes = []
            except (OSError, AttributeError):
                continue
            _LIBMUJOCO = lib
            break
    return _LIBMUJOCO or None


def free_last_xml():
    """Free MuJoCo's retained last-XML compiler state. False when the library is not reachable."""
    lib = libmujoco()
    if lib is None:
        return False
    lib.mj_freeLastXML()
    return True


def trim_heap():
    """Return freed heap pages to the OS (glibc ``malloc_trim``); False where that is not available."""
    try:
        ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        return False
    return True


def release_compiler_state():
    """Free the retained compiler state and trim the heap. True when the state was freed."""
    freed = free_last_xml()
    trim_heap()
    return freed
