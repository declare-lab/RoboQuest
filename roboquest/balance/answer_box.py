"""Open-top column answer box, one parcel wide (spec 4.2).

**Private placeholder.** :func:`answer_box` carries the exact signature the v1
build brief gives the procedural-asset job (contract C4)::

    answer_box(parent, name, parcel_xy=(lx, ly), parcel_height=h, rng=None)
        -> dict(body, inner_xy, wall_height, floor_z)

so the integrator can drop the ASSETS package's builder in its place without
touching :mod:`roboquest.tasks.odd_parcel`. Until then this module
builds the box, following the same conventions as the rest of the suite: every
geom is named ``{name}_*``, the shell is a hollow ring of thin boxes (never a
convex mesh), each mass-carrying group-0 collision geom has an opaque group-1
visual twin (robosuite's offscreen renderer hides group 0), and the masses are
realistic for a small wooden crate.

Geometry (spec 4.2 and 4.10): the interior is one parcel footprint plus
**1.5 cm of clearance per side**, and the walls stand **1 cm above one parcel**,
so parcels can only stack in insertion order and the bottom one has to be
pinched out between the walls. The box has a thin floor plate of its own, so
"in the box" is a property of the box, not of the counter under it.
"""
import xml.etree.ElementTree as ET

CLEARANCE_PER_SIDE = .015      # interior = parcel footprint + this on each side
WALL_ABOVE_PARCEL = .01        # wall top above one parcel standing on the floor
WALL_THICKNESS = .008
FLOOR_THICKNESS = .008
# Neutral crate colours; the box is not a functional colour (the goal says only "the box"),
# so it varies per instance from the materials stream (spec 4.9).
BODY_RGBA = ((.72, .66, .56, 1.), (.62, .60, .58, 1.), (.55, .58, .62, 1.), (.78, .74, .68, 1.),
             (.46, .48, .50, 1.), (.68, .58, .46, 1.), (.50, .54, .48, 1.), (.80, .78, .72, 1.))
MASS_FLOOR = .25
MASS_WALL = .10


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _pair(body, name, pos, half, rgba, mass):
    """A group-0 collision box carrying the mass plus its opaque group-1 visual twin."""
    common = dict(name=name, type='box', pos=_vec(pos), size=_vec(half))
    ET.SubElement(body, 'geom', **dict(common, group='0', contype='1', conaffinity='1',
                                       rgba='0.5 0.5 0.5 1', mass=f'{mass:.9g}',
                                       friction='0.9 0.005 0.0001', solref='0.006 1',
                                       solimp='0.98 0.999 0.001', condim='3', margin='0'))
    ET.SubElement(body, 'geom', **dict(common, name=name + '_visual', group='1', contype='0',
                                       conaffinity='0', density='0', rgba=_vec(rgba)))


def interior_xy(parcel_xy, clearance=CLEARANCE_PER_SIDE):
    """Interior lengths (x, y) for a parcel footprint: the footprint plus ``clearance`` per side."""
    return (float(parcel_xy[0]) + 2 * clearance, float(parcel_xy[1]) + 2 * clearance)


def answer_box(parent, name, parcel_xy=(.07, .10), parcel_height=.05, rng=None):
    """Append an open-top box one parcel wide under ``parent`` (a body on the support plane).

    ``parcel_xy`` are the parcel's full footprint lengths and ``parcel_height`` its
    full height. ``rng`` (a ``numpy.random.Generator``) only picks the body colour;
    the geometry is fully determined by the parcel, so two instances of the same
    task always have the same box.

    Returns ``dict(body, inner_xy, wall_height, floor_z)``: the body name, the
    **full** interior lengths, the wall height above the interior floor, and the
    interior floor height above ``parent``'s origin.
    """
    inner = interior_xy(parcel_xy)
    wall_height = float(parcel_height) + WALL_ABOVE_PARCEL
    floor_z = FLOOR_THICKNESS
    t, hx, hy = WALL_THICKNESS, inner[0] / 2, inner[1] / 2
    rgba = BODY_RGBA[0] if rng is None else BODY_RGBA[int(rng.integers(0, len(BODY_RGBA)))]
    body = ET.SubElement(parent, 'body', name=f'{name}_body', pos='0 0 0')
    _pair(body, f'{name}_floor', (0, 0, floor_z / 2), (hx + t, hy + t, floor_z / 2), rgba, MASS_FLOOR)
    zc = floor_z + wall_height / 2
    for wall, pos, half in (('left', (-(hx + t / 2), 0, zc), (t / 2, hy + t, wall_height / 2)),
                            ('right', ((hx + t / 2), 0, zc), (t / 2, hy + t, wall_height / 2)),
                            ('front', (0, -(hy + t / 2), zc), (hx, t / 2, wall_height / 2)),
                            ('back', (0, (hy + t / 2), zc), (hx, t / 2, wall_height / 2))):
        _pair(body, f'{name}_wall_{wall}', pos, half, rgba, MASS_WALL)
    ET.SubElement(body, 'site', name=f'{name}_floor_site', pos=_vec((0, 0, floor_z)), size='0.002',
                  rgba='0 0 0 0', group='5')
    return dict(body=f'{name}_body', inner_xy=[inner[0], inner[1]], wall_height=wall_height,
                floor_z=floor_z, floor_site=f'{name}_floor_site', rgba=list(rgba),
                outer_xy=[inner[0] + 2 * t, inner[1] + 2 * t], wall_thickness=t,
                placeholder='private placeholder with the C4 answer_box signature; swap for the ASSETS builder')
