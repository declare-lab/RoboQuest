"""The bowl containment rule, copied from ``open_containers_rules``.

Success needs only the items settled inside the bowl, released. Box states are
diagnostics: any legal way of reaching an item is accepted and no box has to
be closed again. With several items in one bowl an item may rest on another
item instead of on the bowl; the task decides support (bowl, or an item that
is itself in the bowl), this rule decides the geometry.
"""
import numpy as np

RIM_MARGIN = .02     # an item piled on another may stand this far above the rim (bottom of the item)


def item_in_bowl(item_state, bowl, item_half_height):
    p = np.asarray(item_state['position'], dtype=float)
    center = np.asarray(bowl['center_xy'], dtype=float)
    radial = float(np.linalg.norm(p[:2] - center))
    inside_radius = radial <= bowl['inner_radius']
    above_floor = p[2] >= bowl['floor_z'] - .005
    below_rim = p[2] - item_half_height <= bowl['rim_z'] + RIM_MARGIN
    return dict(radial_m=radial, inside_radius=bool(inside_radius), above_floor=bool(above_floor),
                below_rim=bool(below_rim), contained=bool(inside_radius and above_floor and below_rim))
