"""Scattered, non-overlapping prop placement in a task frame.

Props used to stand in rows at a fixed pitch, which made every instance of a
task look like the same shop display. This module places them the way objects
sit on a real counter: sampled uniformly over a frame-local rectangle, at a
random yaw, rejected and re-drawn whenever a draw would touch another prop, a
keep-out (pads, boards, the balance, the tray zone, Submit) or the edge of the
region.

Conventions
-----------
Everything is in the task frame of :class:`~roboquest.kitchen.RoboQuestKitchen`
(x along the front of the work surface, y away from the robot, metres, yaw in
radians about +z). A prop's **footprint** is either

* a radius (a float) — a yaw-independent bounding disc, for round props and for
  assets whose bounding circle is the honest description (a mug with a handle);
* a pair ``(hx, hy)`` — the half sizes of a box that turns with the prop.

A **region** is an axis-aligned :class:`Rect` that the prop's whole footprint
must stay inside; a **keep-out** is a ``Rect``, a :class:`Box` (oriented) or a
:class:`Disc` that it must stay ``keepout_gap`` away from.

Determinism
-----------
:func:`scatter` draws from the caller's ``numpy`` generator (the task's ``poses``
stream) in a fixed order — yaw, x, y per attempt — so the same generator state
and the same arguments give the same poses, rejections included. It never falls
back to a degenerate layout: when the region cannot hold the props it raises
``ValueError``, which makes ``registry.mint`` reject that seed and the generator
walk on to the next one.

Clearances
----------
:func:`clearance` returns the signed gap between two shapes in metres (negative
when they overlap). Disc-disc and disc-box are exact; box-box is the separating
axis value over the four edge normals, which is exact whenever the closest
features are a face and a vertex and a *lower bound* (never an over-estimate) in
the vertex-vertex case, so a gate built on it is conservative. :func:`check`
re-derives the same numbers for a finished placement, so a task's CPU gate
measures exactly what the sampler enforced.
"""
from dataclasses import dataclass
import itertools
import math

import numpy as np

__all__ = ['Rect', 'Box', 'Disc', 'rect', 'as_shape', 'shape_at', 'extent', 'clearance',
           'region_margin', 'check', 'scatter']


@dataclass(frozen=True)
class Rect:
    """Axis-aligned frame-local rectangle, used for regions and keep-outs."""
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def center(self):
        return (.5 * (self.x0 + self.x1), .5 * (self.y0 + self.y1))

    @property
    def half(self):
        return (.5 * (self.x1 - self.x0), .5 * (self.y1 - self.y0))

    def shrunk(self, ex, ey):
        return Rect(self.x0 + ex, self.x1 - ex, self.y0 + ey, self.y1 - ey)

    @property
    def empty(self):
        return self.x1 < self.x0 or self.y1 < self.y0


@dataclass(frozen=True)
class Box:
    """Oriented box: centre, half sizes, yaw about +z."""
    center: tuple
    half: tuple
    yaw: float = 0.

    @property
    def corners(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = self.half
        cx, cy = self.center
        return [(cx + sx * hx * c - sy * hy * s, cy + sx * hx * s + sy * hy * c)
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


@dataclass(frozen=True)
class Disc:
    """Bounding disc: centre and radius; yaw-independent."""
    center: tuple
    radius: float


def rect(center, half):
    """A :class:`Rect` from a centre and half sizes."""
    return Rect(center[0] - half[0], center[0] + half[0], center[1] - half[1], center[1] + half[1])


def footprint_of(footprint):
    """Normalise a footprint to ``('disc', radius)`` or ``('box', (hx, hy))``."""
    if np.isscalar(footprint) or isinstance(footprint, (int, float, np.floating, np.integer)):
        return 'disc', float(footprint)
    values = [float(v) for v in footprint]
    if len(values) != 2:
        raise ValueError(f'a footprint is a radius or a (hx, hy) pair, got {footprint!r}')
    return 'box', (values[0], values[1])


def extent(footprint, yaw=0.):
    """Axis-aligned half extents of a footprint held at ``yaw``."""
    kind, value = footprint_of(footprint)
    if kind == 'disc':
        return (value, value)
    hx, hy = value
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (hx * c + hy * s, hx * s + hy * c)


def shape_at(footprint, xy, yaw=0.):
    """The :class:`Disc` or :class:`Box` a footprint occupies at a pose."""
    kind, value = footprint_of(footprint)
    center = (float(xy[0]), float(xy[1]))
    return Disc(center, value) if kind == 'disc' else Box(center, value, float(yaw))


def as_shape(item):
    """A keep-out given as a ``Rect``, ``Box``, ``Disc`` or ``(center, half)`` pair as a shape."""
    if isinstance(item, (Box, Disc)):
        return item
    if isinstance(item, Rect):
        return Box(item.center, item.half, 0.)
    center, half = item
    return Box((float(center[0]), float(center[1])), (float(half[0]), float(half[1])), 0.)


def _project(corners, axis):
    values = [corner[0] * axis[0] + corner[1] * axis[1] for corner in corners]
    return min(values), max(values)


def _box_box_gap(a, b):
    """Separating-axis gap over both boxes' edge normals (a lower bound on the distance)."""
    best = -math.inf
    for shape in (a, b):
        c, s = math.cos(shape.yaw), math.sin(shape.yaw)
        for axis in ((c, s), (-s, c)):
            a0, a1 = _project(a.corners, axis)
            b0, b1 = _project(b.corners, axis)
            best = max(best, max(b0 - a1, a0 - b1))
    return best


def _disc_box_gap(disc, box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    dx = disc.center[0] - box.center[0]
    dy = disc.center[1] - box.center[1]
    local = (dx * c + dy * s, -dx * s + dy * c)
    outside = (max(abs(local[0]) - box.half[0], 0.), max(abs(local[1]) - box.half[1], 0.))
    if outside[0] > 0. or outside[1] > 0.:
        return math.hypot(*outside) - disc.radius
    # Centre inside the box: the (negative) distance to the nearest face.
    return -min(box.half[0] - abs(local[0]), box.half[1] - abs(local[1])) - disc.radius


def clearance(a, b):
    """Signed gap in metres between two shapes; negative when they overlap."""
    a, b = as_shape(a), as_shape(b)
    if isinstance(a, Disc) and isinstance(b, Disc):
        return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) - a.radius - b.radius
    if isinstance(a, Disc):
        return _disc_box_gap(a, b)
    if isinstance(b, Disc):
        return _disc_box_gap(b, a)
    return _box_box_gap(a, b)


def region_margin(shape, region):
    """Signed distance from a shape's bounding box to the region's edge; negative when it sticks out."""
    shape = as_shape(shape)
    if isinstance(shape, Disc):
        ex = ey = shape.radius
    else:
        ex, ey = extent(shape.half, shape.yaw)
    cx, cy = shape.center
    return min(cx - ex - region.x0, region.x1 - cx - ex, cy - ey - region.y0, region.y1 - cy - ey)


def _numeric(value):
    return isinstance(value, (int, float, np.floating, np.integer))


def _regions(region, count):
    if isinstance(region, Rect):
        return [region] * count
    values = list(region)
    if len(values) != count:
        raise ValueError(f'region has {len(values)} entries, expected {count}')
    return values


def _footprints(footprints, count):
    """One shared footprint (a radius, or a ``tuple`` of half sizes) or a ``list`` with one per prop."""
    if _numeric(footprints) or (isinstance(footprints, tuple) and len(footprints) == 2
                                and all(_numeric(v) for v in footprints)):
        return [footprints] * count
    values = list(footprints)
    if len(values) != count:
        raise ValueError(f'footprints has {len(values)} entries, expected {count}')
    return values


def check(poses, footprints, bounds=None, min_gap=0., keepouts=(), keepout_gap=None,
          names=None, keepout_names=None):
    """Re-derive a finished placement's clearances. Returns a report dict with ``problems``.

    ``poses`` are ``(x, y, yaw)`` triples and ``bounds`` the rectangle (normally
    the task footprint) every prop must stay inside. The numbers are the ones
    :func:`scatter` enforced, so a task's CPU gate and the sampler agree.
    """
    region = bounds
    poses = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.) for p in poses]
    prints = _footprints(footprints, len(poses))
    shapes = [shape_at(f, (x, y), yaw) for f, (x, y, yaw) in zip(prints, poses)]
    labels = list(names) if names is not None else [f'prop_{i}' for i in range(len(poses))]
    keeps = [as_shape(k) for k in keepouts]
    keep_labels = (list(keepout_names) if keepout_names is not None
                   else [f'keepout_{i}' for i in range(len(keeps))])
    keep_gap = float(min_gap) if keepout_gap is None else float(keepout_gap)
    problems, pair_gaps, keep_gaps, margins = [], [], [], []
    for i, j in itertools.combinations(range(len(shapes)), 2):
        gap = clearance(shapes[i], shapes[j])
        pair_gaps.append(gap)
        if gap < float(min_gap):
            problems.append(f'{labels[i]} and {labels[j]} closer than {float(min_gap):.3f} m')
    for index, shape in enumerate(shapes):
        for keep, keep_label in zip(keeps, keep_labels):
            gap = clearance(shape, keep)
            keep_gaps.append(gap)
            if gap < keep_gap:
                problems.append(f'{labels[index]} closer than {keep_gap:.3f} m to {keep_label}')
        if region is not None:
            margin = region_margin(shape, region)
            margins.append(margin)
            if margin < 0.:
                problems.append(f'{labels[index]} outside the bounds')
    return dict(problems=problems,
                min_pair_gap_m=round(min(pair_gaps), 5) if pair_gaps else None,
                min_keepout_gap_m=round(min(keep_gaps), 5) if keep_gaps else None,
                min_bounds_margin_m=round(min(margins), 5) if margins else None)


def scatter(rng, count, region, footprints, min_gap, keepouts=(), yaw_range=(-math.pi, math.pi),
            max_tries=6000, keepout_gap=None, attempts_per_prop=300,
            region_holds='footprint', bounds=None):
    """Sample ``count`` non-overlapping ``(x, y, yaw)`` poses inside ``region``.

    ``rng``                a ``numpy.random.Generator`` (the task's ``poses`` stream).
    ``region``             a :class:`Rect`, or a list with one per prop.
    ``footprints``         a radius or a ``(hx, hy)`` tuple shared by every prop,
                           or a ``list`` with one per prop.
    ``min_gap``            metres of clear space required between two props.
    ``keepouts``           shapes the props must stay ``keepout_gap`` (default ``min_gap``) from.
    ``yaw_range``          closed interval the yaw is drawn from.
    ``max_tries``          total draws before giving up; then ``ValueError``.
    ``attempts_per_prop``  draws for one prop before the whole layout restarts.
    ``region_holds``       ``'footprint'``: the prop's whole footprint stays inside its
                           region (a scatter area); ``'center'``: the region bounds the
                           prop's *centre* only (a per-prop jitter box).
    ``bounds``             an extra rectangle — normally the task footprint — the whole
                           footprint must stay inside, whatever ``region_holds`` says.

    Props are placed in order, each drawn uniformly over its region; a draw that
    clashes is discarded. When a prop cannot be placed the layout restarts from
    scratch, so a greedy first prop never dooms the rest. Raises ``ValueError``
    when a region cannot hold its prop at all or when ``max_tries`` runs out.
    """
    count = int(count)
    if count < 0:
        raise ValueError('count must not be negative')
    if count == 0:
        return []
    if region_holds not in ('footprint', 'center'):
        raise ValueError(f"region_holds is 'footprint' or 'center', got {region_holds!r}")
    regions = _regions(region, count)
    prints = _footprints(footprints, count)
    keeps = [as_shape(k) for k in keepouts]
    gap = float(min_gap)
    keep_gap = gap if keepout_gap is None else float(keepout_gap)
    low, high = float(yaw_range[0]), float(yaw_range[1])
    if high < low:
        raise ValueError(f'yaw_range {yaw_range!r} is empty')
    yaws = np.linspace(low, high, 9) if high > low else np.array([low])

    def inner_rect(own, footprint, yaw):
        return own.shrunk(*extent(footprint, yaw)) if region_holds == 'footprint' else own

    # Fail loudly (not after max_tries) when a prop cannot fit its region at any yaw.
    for index, (own, footprint) in enumerate(zip(regions, prints)):
        if all(inner_rect(own, footprint, float(yaw)).empty for yaw in yaws):
            raise ValueError(f'scatter: prop {index} with footprint {footprint!r} does not fit region {own}')
    tries, restarts = 0, 0
    while tries < max_tries:
        restarts += 1
        placed, poses = [], []
        for index in range(count):
            own, footprint = regions[index], prints[index]
            chosen = None
            for _ in range(attempts_per_prop):
                if tries >= max_tries:
                    break
                tries += 1
                yaw = round(float(rng.uniform(low, high)) if high > low else low, 6)
                inner = inner_rect(own, footprint, yaw)
                if inner.empty:
                    continue                     # this yaw does not fit; draw another
                x = round(float(rng.uniform(inner.x0, inner.x1)), 6)
                y = round(float(rng.uniform(inner.y0, inner.y1)), 6)
                shape = shape_at(footprint, (x, y), yaw)
                if bounds is not None and region_margin(shape, bounds) < 0.:
                    continue
                if any(clearance(shape, other) < gap for other in placed):
                    continue
                if any(clearance(shape, keep) < keep_gap for keep in keeps):
                    continue
                chosen = (shape, (x, y, yaw))
                break
            if chosen is None:
                break
            placed.append(chosen[0])
            poses.append(chosen[1])
        if len(poses) == count:
            return poses
    raise ValueError(f'scatter: could not place {count} props with a {gap:.3f} m gap in {region} '
                     f'within {max_tries} draws ({restarts} layout attempts)')
