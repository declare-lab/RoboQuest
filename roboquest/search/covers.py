"""Object covers for the search room's ``covered`` hiding places (spec 4.4, cover rules in 2.2).

The cover itself is the shared procedural cloche (``roboquest/assets/cloche.py``,
contract C4): an opaque hollow shell (a ring of thin boxes closed by a thin disk) with a knob the
gripper pinches at the ``{name}_knob`` site. ``build_cover`` builds one at a world pose and wraps it in
``ElementObject``, a robosuite ``MujocoObject`` over the builder's element, so the base class merges
it into the worldbody (free joints must sit on top-level bodies), resolves its body id, includes it in
``prop_clashes()``, the G2 settle and the evaluator snapshot, and scores it like every other task
object: released, settled and off the tray at the Submit press.

Inverted RoboCasa bowls were measured for this task (``objaverse/bowl/bowl_0..19``): 10 to 14 cm
across and 2.7 to 6.2 cm deep, so none can cover a pool object (9 to 13.5 cm tall); the bowl cover
kind is therefore not offered here even though every bowl's collision shell is hollow.
"""
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.objects import MujocoObject

from roboquest.assets import cloche
from roboquest.assets.cloche import SIZES as CLOCHE_SIZES, WALL as CLOCHE_WALL

COVER_SIZES = tuple(CLOCHE_SIZES)


def cover_dims(size):
    """Outer radius and the interior a covered object must fit (metres), without building anything."""
    if size not in CLOCHE_SIZES:
        raise ValueError(f'cover size must be one of {COVER_SIZES}, not {size!r}')
    p = CLOCHE_SIZES[size]
    return dict(radius=p['radius'], inner_radius=p['radius'] - CLOCHE_WALL, inner_height=p['depth'],
                height=p['depth'] + CLOCHE_WALL, mass=p['mass'])


class ElementObject(MujocoObject):
    """A body element built by a C4-style builder, registered like a native object.

    The builder already names the body ``name`` and every geom, joint and site ``{name}_*``, so
    nothing is prefixed; ``root_body``, ``joints`` (including a ``<freejoint>``) and
    ``contact_geoms`` read the element as is. ``bottom_offset`` is the body origin (the rim plane
    for a cover), ``top_offset`` the given top.
    """

    def __init__(self, name, body, top_z, horizontal_radius):
        super().__init__(obj_type='all', duplicate_collision_geoms=False)
        self._name = name
        self._obj = body
        self._top = np.array([0., 0., float(top_z)])
        self._radius = float(horizontal_radius)
        self._get_object_properties()

    def _get_object_properties(self):
        super()._get_object_properties()
        if not self._joints:      # robosuite's element filter only knows <joint>; the assets use <freejoint>
            self._joints = [e.get('name') for e in self._obj.iter('freejoint') if e.get('name')]

    def exclude_from_prefixing(self, inp):
        return True

    def _get_object_subtree(self):
        return self._obj

    @property
    def bottom_offset(self):
        return np.zeros(3)

    @property
    def top_offset(self):
        return self._top.copy()

    @property
    def horizontal_radius(self):
        return self._radius

    def get_bounding_box_half_size(self):
        return np.array([self._radius, self._radius, self._top[2] / 2])


def build_cover(name, size, rng, assets, pos, yaw):
    """A free cloche of ``size`` at world ``pos`` (rim plane) and ``yaw``, dressed from ``rng`` with a RoboCasa
    texture registered on ``assets``. Returns ``(ElementObject, spec)`` with the builder's spec (``radius``,
    ``height``, ``inner_radius``, ``inner_height``, ``knob_site``, ``knob_grasp_z``, ``knob_top_z``, ``mass``,
    ``kind``, ``material``, ``rgba``, ``geoms``)."""
    holder = ET.Element('holder')
    spec = cloche(holder, name, size=size, rng=rng, free=True, pos=pos, yaw=yaw, assets=assets)
    obj = ElementObject(name, spec['body'], top_z=spec['knob_top_z'], horizontal_radius=spec['radius'])
    return obj, spec
