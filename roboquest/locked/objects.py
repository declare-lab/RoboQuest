"""Lock hardware for ``locked_storage``: reader pads, lock plates and token fobs (contract v1.1 section 2).

Three procedural pieces, built the way the C4 assets are (``assets/_mjcf.py``): a group-0 collision
geom carrying the mass and an opaque group-1 visual twin.

* **Reader pad** -- a flat 12 x 9 x 0.6 cm slab welded to a counter top, coloured with its lock colour.
  Welded (no free joint) so the robot cannot carry the reader away; a token is *placed on* it.
* **Lock plate** -- a 4 x 4 cm coloured square on the compartment's moving front panel, next to the
  handle, so the colour mapping is observable without the goal stating it. Visual only (group 1,
  ``contype=0``): it is a painted marker, not a part that can catch on the frame.
* **Token fob** -- a free 5.0 x 3.2 x 1.4 cm box, 40 g, high friction, in the lock colour, with a
  1 cm white dot on its top face so it does not read as a plain block.

The lock palette is disjoint from the search pool's functional colours (blue, red, green, yellow),
so a token is never confusable with a pool object (spec 4.9 keeps functional colours fixed).
"""
import xml.etree.ElementTree as ET

from roboquest.assets._mjcf import pair, site, vec, visual, yaw_quat

# Disjoint from search/objects.py COLOURS (blue, red, green, yellow).
LOCK_COLOURS = {
    'orange': (.95, .48, .10, 1.),
    'magenta': (.86, .12, .62, 1.),
    'cyan': (.10, .78, .85, 1.),
    'white': (.95, .95, .95, 1.),
    'purple': (.49, .21, .76, 1.),
}
COLOUR_NAMES = tuple(sorted(LOCK_COLOURS))

PAD_SIZE_M = (.12, .09, .006)        # slab: along the counter edge, into the counter, thick
PAD_MASS_KG = .18
PAD_RADIUS_M = .075                  # half-diagonal of the slab, for the counter clearance discs
PLATE_SIZE_M = (.04, .04, .006)
TOKEN_SIZE_M = (.050, .032, .014)
TOKEN_MASS_KG = .040
TOKEN_DOT_RADIUS_M = .005            # the 1 cm white dot
TOKEN_FRICTION = '1.4 0.02 0.001'    # high friction: the fob stays where it is set down on a pad
PAD_FRICTION = '1.2 0.01 0.0005'


def pad_name(cid):
    return f'pad_{cid}'


def token_name(index):
    return f'token_{int(index)}'


def plate_name(cid):
    return f'lock_plate_{cid}'


def reader_pad(parent, name, colour, pos, yaw=0.):
    """A welded reader pad slab whose *top* face sits ``PAD_SIZE_M[2]`` above ``pos`` (the counter top).

    ``parent`` is the worldbody; the pad has no joint, so it belongs to the world and cannot be moved.
    Returns the record the task and the lock rule read (``top_z``, ``half_xy``, ``centre``).
    """
    rgba = LOCK_COLOURS[colour]
    sx, sy, sz = PAD_SIZE_M
    body = ET.SubElement(parent, 'body', name=name, pos=vec(pos), quat=yaw_quat(yaw))
    pair(body, f'{name}_slab', 'box', (0., 0., sz / 2), (sx / 2, sy / 2, sz / 2), PAD_MASS_KG,
         rgba=rgba, friction=PAD_FRICTION)
    # A thin dark inlay so the pad reads as a reader, not as a coloured tile.
    visual(body, f'{name}_inlay', 'box', (0., 0., sz + .0004), (sx / 2 - .012, sy / 2 - .010, .0004),
           rgba=(.12, .12, .13, 1.))
    site(body, f'{name}_centre', (0., 0., sz))
    return dict(body=body, name=name, colour=colour, rgba=list(rgba), yaw=float(yaw),
                centre=[float(v) for v in pos], top_z=float(pos[2]) + sz,
                half_xy=[sx / 2, sy / 2], size_xyz_m=list(PAD_SIZE_M), mass=PAD_MASS_KG)


def token(parent, name, colour, pos, yaw=0.):
    """A free token fob at world ``pos`` (its *bottom* plane) and ``yaw``; see the module docstring."""
    rgba = LOCK_COLOURS[colour]
    sx, sy, sz = TOKEN_SIZE_M
    body = ET.SubElement(parent, 'body', name=name, pos=vec(pos), quat=yaw_quat(yaw))
    joint = f'{name}_joint'
    ET.SubElement(body, 'freejoint', name=joint)
    pair(body, f'{name}_body', 'box', (0., 0., sz / 2), (sx / 2, sy / 2, sz / 2), TOKEN_MASS_KG,
         rgba=rgba, friction=TOKEN_FRICTION, condim='4')
    visual(body, f'{name}_dot', 'cylinder', (0., 0., sz + .0003), (TOKEN_DOT_RADIUS_M, .0003),
           rgba=(.97, .97, .97, 1.))
    site(body, f'{name}_grasp', (0., 0., sz / 2))
    return dict(body=body, name=name, joint=joint, colour=colour, rgba=list(rgba),
                size_xyz_m=list(TOKEN_SIZE_M), mass=TOKEN_MASS_KG, height=sz,
                half_xy=[sx / 2, sy / 2], grasp_site=f'{name}_grasp')


def lock_plate(panel_body, name, colour, pos, normal_axis, thickness=PLATE_SIZE_M[2]):
    """Append a visual-only 4 x 4 cm plate to a compartment's moving front panel element.

    ``pos`` is in the panel body's frame and ``normal_axis`` (0 or 1) is the panel's outward axis, so
    the plate is thin in that direction and square in the other two. Visual only: a painted marker
    cannot catch on the cabinet frame or change how the door swings.
    """
    axis = int(normal_axis)
    size = [PLATE_SIZE_M[0] / 2, PLATE_SIZE_M[1] / 2, PLATE_SIZE_M[1] / 2]
    size[axis] = float(thickness) / 2
    rim = [s * 1.18 if i != axis else s * .5 for i, s in enumerate(size)]
    visual(panel_body, name + '_rim', 'box', pos, rim, rgba=(.10, .10, .11, 1.))
    visual(panel_body, name, 'box', pos, size, rgba=LOCK_COLOURS[colour])
    return dict(name=name, colour=colour, rgba=list(LOCK_COLOURS[colour]), pos=[float(v) for v in pos],
                normal_axis=axis, size_xyz_m=[2 * s for s in size])
