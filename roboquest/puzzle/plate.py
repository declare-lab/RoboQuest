"""The opaque mechanism plate that hides a puzzle chest's interlocks (generator v3).

Why a plate. Generator v1/v2 hid a lock with a thin strip lying on the locked
part over its own notch. Measured on the interim v1 registry (162 instances,
396 covered locks) every covered lock had *exactly one* slider whose footprint
entered its strip, never two: the holder's body ran visibly into the strip and
only the last centimetres of its tip were hidden, so following each slider to
where it disappeared read the whole blocking chain off a single frame. The
strips are gone. In their place one rigid opaque plate is fixed to the chest a
few millimetres above the sliding layer, over the region where tips meet
notches. Sliders travel underneath it; only their handles and a stub of bare
slider stick out past its edge, so several sliders disappear under the same
plate and which of them ends in which notch has to be discovered by pulling.

Geometry (chest-local metres, the layout frame: x right, y away from the robot).

* The sliding layer is ``LAYER_Z`` = (0.076, 0.088) above the chest base. The
  plate floats ``PLATE_CLEAR`` = 4 mm over it and is ``PLATE_THICK`` = 6 mm
  thick, so its underside is at 0.092 and nothing in the sliding layer can
  touch it. It stands on at least ``MIN_LEGS`` ``LEG_HALF``-square legs that
  rest on the deck top (0.075) in columns no slider sweep ever enters.
* A handle post rises from the sliding layer to 0.148, so a handle would hit
  the plate: the plate therefore keeps ``HANDLE_KEEP`` = ``HANDLE_CLEAR`` +
  ``HANDLE_RADIUS`` = 53 mm from every handle centre *over that handle's whole
  travel*, which leaves at least ``HANDLE_CLEAR`` = 4 cm of bare slider between
  the plate's edge and the grasp region for the gripper and the wrist. The
  lid's handle travels the lid's full opening path, so the plate never blocks
  the lid: the lid slides *under* it and its handle stays 53 mm clear.
* The hand reaches into the cavity for the item, so the plate keeps clear of
  the column ``ITEM_CLEAR`` around the item.
* At ``partial`` the locks that stay visible must stay readable, so the plate
  keeps ``SHOW_MARGIN`` from their engagements.

The plate is the *maximal* free region around the engagements it hides: each
covered engagement seeds a rectangle that grows in 5 mm steps until it meets a
keep-out or the deck, rectangles whose bounding box is free are merged, and
what is left is one plate made of one to a few boxes. Growing it is what makes
the puzzle hard: the further a slider runs before it disappears, the less its
visible part says about where it ends.

Hardness (G0). For a covered lock, a *candidate* holder is a slider the
observer cannot rule out from the visible geometry:

* ``simple``  -- its footprint enters the plate, so the frame does not say
  where it ends.
* ``strict``  -- additionally its tip is hidden under the plate (a slider whose
  tip is in plain sight holds nothing hidden), it is perpendicular to the
  covered lock (a tip can only enter a notch from the side), its straight
  extrapolated body crosses that lock's body somewhere under the plate on the
  tip side of its own handle, and it is not already *committed*: a slider the
  observer can watch holding an uncovered lock holds nothing hidden as well,
  because in a chain every slider holds at most one lock.

:func:`hardness_problems` rejects a layout whose covered locks do not all keep
at least :data:`MIN_CANDIDATES` candidates; the sampler resamples, it never
patches a layout that fails.

The gate runs on :data:`GATE_MEASURE`. It is ``simple``, for a measured reason.
Committed elimination is exhaustive: a chain of length L drawn with ``d`` decoys
has L + d sliders and L locks, of which L - c stay visible when c are covered, so
the deepest covered lock has at most (L + d) - (L - c) - 1 = d + c - 1 strict
candidates whatever the geometry -- two of them at ``partial`` with a chain of
four and no decoy, and only if the geometry delivers both. It does not: over 200
seeds in each of the nine cells the strict count of some covered lock is 1 in
every accepted chain-4 ``partial`` instance and in 9-15 % of the instances of the
other covered cells, at every decoy count. Gating on strict would therefore empty
one factor cell outright and make the others depend on a nuisance factor, so
``strict`` is a diagnostic: both measures are recorded on every instance
(:func:`hardness`) and an analysis can condition on the strict count afterwards.
"""
from __future__ import annotations

import math

from roboquest.puzzle_box_geometry import DECK_TOP, LAYER_Z
from roboquest.puzzle_box_layouts import CAVITY, HALF_W, HANDLE_RADIUS, handle_at, inside, overlap, swept_rects

# --- the plate's own numbers ------------------------------------------------
PLATE_CLEAR = .004                  # air between the sliding layer's top and the plate's underside
PLATE_THICK = .006
PLATE_Z = (LAYER_Z[1] + PLATE_CLEAR, LAYER_Z[1] + PLATE_CLEAR + PLATE_THICK)
LEG_Z = (DECK_TOP, PLATE_Z[0])      # legs stand on the deck top and carry the plate
LEG_HALF = .009
LEG_GAP = .004                      # a leg column keeps this from every slider sweep
MIN_LEGS = 3                        # legs carrying the plate, so it is not a floating slab
HANDLE_CLEAR = .040                 # bare slider between the plate's edge and the handle's grasp region
HANDLE_KEEP = HANDLE_CLEAR + HANDLE_RADIUS
HANDLE_SAMPLES = 11                 # handle positions sampled over a part's travel
HIDE_MARGIN = .010                  # the plate reaches this far around an engagement it hides
EDGE_MARGIN = .004                  # except past the far end of the notch, which is solid lock
SHOW_MARGIN = .015                  # and stays this far from an engagement it must leave visible
ITEM_CLEAR = (.106, .045)           # half extents of the slot the plate leaves over the item, and
                                    # the one number in v3 that is the gripper's and not the box's.
                                    # Measured on the model in the grasp site's frame: the open
                                    # fingers reach 52 mm along the axis they close on, 11 mm across
                                    # it and from 13 mm below the site to 54 mm above it, and the
                                    # hand behind them is a 211 x 64 mm box hanging 22 to 124 mm
                                    # above the site. With the site down at an item standing in the
                                    # cavity both are level with a plate 17 mm over the deck, so the
                                    # plate has to leave the whole hand free along the axis the
                                    # ranked-first top grasp closes on -- the chest's x -- and as
                                    # much as it can across it. 45 mm across is the mechanism's
                                    # limit, not a choice: at 50 mm the hand's slot reaches the lid's
                                    # own notch, no plate can hide the first lock of the prefix and
                                    # the sampler draws nothing at all in any covered cell. The first
                                    # v3 draft kept 140 x 80 mm and failed G4 with the plate standing
                                    # in the fingers' way: the oracle opened the lid and freed the
                                    # item on 9 of 9 plate instances and could not pick it up on 7.
DECK_INSET = .005                   # the plate stays this far inside the deck
GROW_STEP = .005
MIN_CANDIDATES = 2                  # G0: at least two sliders stay possible for every covered lock
MEASURES = ('simple', 'strict')
GATE_MEASURE = 'simple'             # the measure the G0 gate runs on; see the module docstring
WALL_THICK = .004                   # the rim wall that closes the gap under the plate's outline
WALL_GAP = .002                     # air the wall leaves around a slider that crosses it
WALL_MIN = .003                     # a wall piece shorter than this is not worth a geom
WALL_Z = (DECK_TOP, PLATE_Z[0])     # it stands on the deck and carries nothing
LINTEL_CLEAR = .0015                # a doorway is closed down to this much air over the slider
LINTEL_Z = (LAYER_Z[1] + LINTEL_CLEAR, PLATE_Z[0])
SKIRT_SKIN = .0005                  # air a skirt leaves in front of a slider's outermost face
SKIRT_STEP = .004                   # a doorway is offered to the skirt in pieces this long
SKIRT_Z = (DECK_TOP, PLATE_Z[1])    # a skirt is outside the outline, so it may rise to the top
GEOM_TOL = 1e-9                     # rectangle arithmetic slack, well under a micrometre


def _bbox(rects):
    return ((min(r[0][0] for r in rects), max(r[0][1] for r in rects)),
            (min(r[1][0] for r in rects), max(r[1][1] for r in rects)))


def _pad(rect, margin):
    return tuple((rect[i][0] - margin, rect[i][1] + margin) for i in range(2))


def _rect_hits_disc(rect, centre, radius):
    """True when any point of the rectangle is closer than `radius` to `centre`."""
    dx = max(rect[0][0] - centre[0], 0., centre[0] - rect[0][1])
    dy = max(rect[1][0] - centre[1], 0., centre[1] - rect[1][1])
    return dx * dx + dy * dy < radius * radius - 1e-12


def _interval_minus(span, cuts, tol=GEOM_TOL):
    """`span` with the open intervals in `cuts` removed, as a list of closed pieces."""
    pieces = [(float(span[0]), float(span[1]))]
    for c0, c1 in cuts:
        out = []
        for lo, hi in pieces:
            if c1 <= lo + tol or c0 >= hi - tol:
                out.append((lo, hi))
                continue
            if c0 > lo + tol:
                out.append((lo, min(c0, hi)))
            if c1 < hi - tol:
                out.append((max(c1, lo), hi))
        pieces = out
    return [(lo, hi) for lo, hi in pieces if hi - lo > tol]


def union_boundary(rects, tol=GEOM_TOL):
    """The outline of the union of axis-aligned rectangles, as ``(p0, p1)`` segments.

    Not simply every rectangle's four edges: :func:`build_plate` grows and merges the
    plate out of overlapping boxes, so an edge that another box covers from the outside
    runs through the *inside* of the plate and is not outline at all. A point sitting on
    such a seam is as deeply covered as the slab around it, which is the whole question
    both callers ask (:func:`outline_depth`).
    """
    segments = []
    boxes = [((float(r[0][0]), float(r[0][1])), (float(r[1][0]), float(r[1][1]))) for r in rects]
    for index, ((x0, x1), (y0, y1)) in enumerate(boxes):
        for axis, value, outward in ((0, x0, -1), (0, x1, 1), (1, y0, -1), (1, y1, 1)):
            span = (y0, y1) if axis == 0 else (x0, x1)
            cuts = []
            for other, box in enumerate(boxes):
                if other == index:
                    continue
                (ox0, ox1), (oy0, oy1) = box
                near, far = (ox0, ox1) if axis == 0 else (oy0, oy1)
                # `other` has to cover the strip just outside this edge for the edge to be interior
                covers = (near < value - tol and far > value - tol) if outward < 0 else \
                         (far > value + tol and near < value + tol)
                if covers:
                    cuts.append((oy0, oy1) if axis == 0 else (ox0, ox1))
            for lo, hi in _interval_minus(span, cuts, tol):
                segments.append(((value, lo), (value, hi)) if axis == 0 else ((lo, value), (hi, value)))
    return segments


def _point_segment_distance(point, segment):
    (ax, ay), (bx, by) = segment
    px, py = point
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    t = 0. if length <= 0. else max(0., min(1., ((px - ax) * dx + (py - ay) * dy) / length))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def inside_union(rects, point, tol=GEOM_TOL):
    """True when the point is inside one of the rectangles or on its edge."""
    px, py = point
    return any(r[0][0] - tol <= px <= r[0][1] + tol and r[1][0] - tol <= py <= r[1][1] + tol
               for r in rects)


def outline_depth(rects, point, tol=GEOM_TOL):
    """How deep `point` sits under the plate: metres to the outline, negative outside it.

    The deck-plane distance from the point to the nearest stretch of :func:`union_boundary`,
    signed so that a point on the outline reads 0. Both the G3 band rule (a changed pixel is
    tolerated only in the outermost ``PLATE_EDGE_INSET`` of the slab, where the travel gap
    shows a sliver of slider by design) and the covered-lock depth gate are this one number.
    """
    if not rects:
        return None
    segments = union_boundary(rects, tol)
    if not segments:
        return None
    distance = min(_point_segment_distance(point, s) for s in segments)
    return distance if inside_union(rects, point, tol) else -distance


def deck_rect(layout):
    return (tuple(layout['deck']['x']), tuple(layout['deck']['y']))


def part_bbox(part):
    return _bbox(part['spans'])


def engagement_rect(layout, lock, margin=0.):
    """The rectangle that shows which tip sits in ``lock``'s notch, padded by `margin`.

    Not the whole notch, and not padded evenly.

    * Along the notch (4 cm) only the holder's bar matters: the bar is ``2 * HALF_W``
      = 3 cm wide and covers the middle, and the 5 mm of empty slot at each end shows
      nothing but the notch floor. Both ends are padded by `margin`, which is where the
      bar would otherwise be read as it runs into the notch.
    * Across the notch the holder comes in from the part's edge and stops inside it, so
      the *outer* end is padded by `margin` and the inner end, which is solid lock
      already, only by ``EDGE_MARGIN``. The lid's notch is cut at the rim of the lid,
      a couple of centimetres from the hand's column over the item, and the
      asymmetry is what lets the plate hide it without reaching into that column.
    """
    part = layout['parts'][lock]
    notch = tuple(tuple(axis) for axis in part['notch'])
    body = part_bbox(part)
    long_axis = 0 if notch[0][1] - notch[0][0] >= notch[1][1] - notch[1][0] else 1
    across = 1 - long_axis
    mid = .5 * (notch[long_axis][0] + notch[long_axis][1])
    rect = [None, None]
    rect[long_axis] = (max(notch[long_axis][0], mid - HALF_W) - margin,
                       min(notch[long_axis][1], mid + HALF_W) + margin)
    outer_low = abs(notch[across][0] - body[across][0]) <= abs(notch[across][1] - body[across][1])
    inner = min(margin, EDGE_MARGIN)
    rect[across] = ((notch[across][0] - (margin if outer_low else inner)),
                    (notch[across][1] + (inner if outer_low else margin)))
    return (rect[0], rect[1])


def engagement_point(layout, lock):
    """The centre of `lock`'s engagement: where the holder's tip sits in the notch."""
    rect = engagement_rect(layout, lock)
    return (.5 * (rect[0][0] + rect[0][1]), .5 * (rect[1][0] + rect[1][1]))


def lock_depth(layout, lock):
    """:func:`outline_depth` of `lock`'s engagement point, or None when there is no plate."""
    plate = layout.get('plate')
    if not plate:
        return None
    return outline_depth([tuple(tuple(axis) for axis in rect) for rect in plate['rects']],
                         engagement_point(layout, lock))


def handle_keepouts(layout):
    """(centre, radius) discs the plate must avoid: every handle over its whole travel."""
    discs = []
    for part in layout['parts'].values():
        lo, hi = part['range']
        for i in range(HANDLE_SAMPLES):
            discs.append((handle_at(part, lo + (hi - lo) * i / (HANDLE_SAMPLES - 1)), HANDLE_KEEP))
    return discs


def item_column(item_xy):
    """The column the hand comes down in to pick the item out of the open chest."""
    return ((item_xy[0] - ITEM_CLEAR[0], item_xy[0] + ITEM_CLEAR[0]),
            (item_xy[1] - ITEM_CLEAR[1], item_xy[1] + ITEM_CLEAR[1]))


def handle_path_clear(part, rects):
    """True when `part`'s handle keeps HANDLE_KEEP from every rectangle over its whole travel."""
    lo, hi = part['range']
    for i in range(HANDLE_SAMPLES):
        centre = handle_at(part, lo + (hi - lo) * i / (HANDLE_SAMPLES - 1))
        if any(_rect_hits_disc(rect, centre, HANDLE_KEEP) for rect in rects):
            return False
    return True


def show_rect(parts, lock):
    """The engagement of a lock the instance leaves visible, with the clearance the plate
    has to keep from it: covering it would hide a lock that the cover level says is shown."""
    return engagement_rect(dict(parts=parts), lock, SHOW_MARGIN)


def hideable(parts, lock, item_xy, deck, movers=None, avoid=()):
    """The rectangle the plate has to cover for `lock`, or None when no plate could.

    The sampler calls this while it places a chain, so that a draw whose engagement
    no plate can reach dies on the spot instead of after the whole layout is built.
    ``avoid`` are the engagements this instance leaves visible (:func:`show_rect`).
    """
    seed = engagement_rect(dict(parts=parts), lock, HIDE_MARGIN)
    if not inside(seed, _pad(deck, -DECK_INSET)):
        return None
    if overlap(seed, item_column(item_xy)):
        return None
    if any(overlap(seed, rect) for rect in avoid):
        return None
    movers = parts.values() if movers is None else movers
    return seed if all(handle_path_clear(part, (seed,)) for part in movers) else None


def keepouts(layout):
    """Everything the plate must not reach over: handles (travelling), the hand's column over the
    item, and the engagements of the locks this instance leaves visible."""
    blocks = [('disc', centre, radius) for centre, radius in handle_keepouts(layout)]
    blocks.append(('rect', item_column(layout['item_xy'])))
    for name, flag in layout['covered'].items():
        if not flag:
            blocks.append(('rect', engagement_rect(layout, name, SHOW_MARGIN)))
    return blocks


def _free(rect, deck, blocks):
    if not inside(rect, deck):
        return False
    for block in blocks:
        if block[0] == 'rect':
            if overlap(rect, block[1]):
                return False
        elif _rect_hits_disc(rect, block[1], block[2]):
            return False
    return True


def _grow(rect, deck, blocks, step=GROW_STEP):
    """Push every side outwards in `step` increments until it meets a keep-out or the deck."""
    current = [list(rect[0]), list(rect[1])]
    sides = [(0, 0, -1.), (0, 1, 1.), (1, 0, -1.), (1, 1, 1.)]
    live = set(range(len(sides)))
    while live:
        for index, (axis, end, sign) in enumerate(sides):
            if index not in live:
                continue
            trial = [list(current[0]), list(current[1])]
            trial[axis][end] += sign * step
            if _free((tuple(trial[0]), tuple(trial[1])), deck, blocks):
                current = trial
            else:
                live.discard(index)
    return (tuple(current[0]), tuple(current[1]))


def _merge(rects, deck, blocks):
    """Merge any two rectangles whose bounding box is free, until none can be merged."""
    rects = list(rects)
    merged = True
    while merged and len(rects) > 1:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                box = _bbox([rects[i], rects[j]])
                if _free(box, deck, blocks):
                    rects = [r for k, r in enumerate(rects) if k not in (i, j)] + [box]
                    merged = True
                    break
            if merged:
                break
    keep = [r for i, r in enumerate(rects)
            if not any(k != i and inside(r, other, 1e-9) for k, other in enumerate(rects))]
    return sorted({tuple(tuple(axis) for axis in rect) for rect in keep})


def covers_rect(rects, rect):
    """True when the rectangle is wholly inside one plate box (the plate is a union of boxes, so a
    rectangle straddling two of them counts only if one of them contains it)."""
    return any(inside(rect, box, 1e-9) for box in rects)


def hits_plate(rects, rect):
    return any(overlap(rect, box) for box in rects)


def _leg_columns(layout, rects):
    """Grid of free columns under the plate: no slider sweep, no cavity, wholly under a plate box."""
    sweeps = [s for part in layout['parts'].values() for s in swept_rects(part)]
    cavity = _pad(CAVITY, LEG_GAP)
    out = []
    for rect in rects:
        x0, x1 = rect[0][0] + LEG_HALF, rect[0][1] - LEG_HALF
        y0, y1 = rect[1][0] + LEG_HALF, rect[1][1] - LEG_HALF
        if x1 < x0 or y1 < y0:
            continue
        nx = max(1, int((x1 - x0) / GROW_STEP) + 1)
        ny = max(1, int((y1 - y0) / GROW_STEP) + 1)
        for i in range(nx):
            for j in range(ny):
                cx = x0 if nx == 1 else x0 + (x1 - x0) * i / (nx - 1)
                cy = y0 if ny == 1 else y0 + (y1 - y0) * j / (ny - 1)
                column = ((cx - LEG_HALF - LEG_GAP, cx + LEG_HALF + LEG_GAP),
                          (cy - LEG_HALF - LEG_GAP, cy + LEG_HALF + LEG_GAP))
                if overlap(column, cavity) or any(overlap(column, s) for s in sweeps):
                    continue
                out.append((round(cx, 4), round(cy, 4)))
    return out


def _free_intervals(span, blocked, minimum=WALL_MIN):
    """What is left of the interval ``span`` once every interval in ``blocked`` is cut out."""
    out = [tuple(span)]
    for lo, hi in blocked:
        nxt = []
        for a, b in out:
            if hi <= a or lo >= b:
                nxt.append((a, b))
                continue
            if a < lo:
                nxt.append((a, lo))
            if hi < b:
                nxt.append((hi, b))
        out = nxt
    return [(a, b) for a, b in out if b - a >= minimum]


def wall_boxes(layout, rects):
    """The rim wall and its lintels: what closes the gap under the plate's outline.

    Four millimetres of air between the plate and the sliders is four millimetres a camera
    can look through: at the shallow angle of the policy cameras a sliver of whatever lies
    just inside the edge comes back into view, which is how a chest that passed the ray
    probes still leaked a few pixels of slider body. A wall from the deck to the plate
    closes that line of sight wherever the outline is solid. Where a slider crosses the
    outline the wall cannot stand -- that is the slider's doorway -- so there a *lintel*
    hangs from the plate down to ``LINTEL_CLEAR`` over the sliding layer instead, leaving
    a millimetre and a half of air the slider never touches and a camera cannot see past.

    Not every doorway is a crossing. A slider whose outermost face stops exactly at the
    plate's edge -- the lid, most often, since the plate grows until something stops it --
    leaves that face standing in the open, coplanar with the plate's own side and lit from
    the same angle; the ray probes never see it, a grazing camera sees a pixel or two of
    it. Nothing can stand inside there, so a *skirt* stands outside instead: a box in the
    free deck just beyond the edge, half a millimetre off the face, rising to the top of
    the plate. A doorway is offered to the skirt in ``SKIRT_STEP`` pieces and it takes the
    runs it can stand on -- deck free of every slider, clear of the cavity, and
    ``HANDLE_KEEP`` from every handle over its travel -- so a corner too near a handle
    costs that corner and not the whole stretch. The lintel hangs over the doorway either
    way: the two close different gaps, one over the slider and one in front of it, and
    where a skirt stands the lintel is carried out over the half millimetre between the
    two so that nothing can be read down the slot from overhead either.

    Interior edges (where one box of the plate abuts another) need none of the three, and
    nor does the cavity, which has no deck to stand on.

    Returns ``(walls, skirts, lintels)``, all lists of chest-local rectangles.
    """
    raw = [s for part in layout['parts'].values() for s in swept_rects(part)]
    sweeps = [_pad(s, WALL_GAP) for s in raw]
    solid = [_pad(CAVITY, WALL_GAP)]
    discs = handle_keepouts(layout)
    deck = deck_rect(layout)
    walls, skirts, lintels = [], [], []
    for index, rect in enumerate(rects):
        for axis in (0, 1):
            other = 1 - axis
            for side in (0, 1):
                edge = rect[axis][side]
                inner = edge + WALL_THICK if side == 0 else edge - WALL_THICK
                strip = [None, None]
                strip[axis] = tuple(sorted((edge, inner)))
                strip[other] = tuple(rect[other])
                strip = (strip[0], strip[1])
                outside = [None, None]                      # a sliver just beyond the edge: when
                outside[axis] = ((edge - WALL_THICK, edge) if side == 0     # another box of the
                                 else (edge, edge + WALL_THICK))            # plate covers it the
                outside[other] = tuple(rect[other])                         # edge is interior
                hard = [(n[other][0], n[other][1]) for j, n in enumerate(rects)
                        if j != index and overlap((outside[0], outside[1]), n)]
                hard += [(b[other][0], b[other][1]) for b in solid if overlap(strip, b)]
                doors = [(s[other][0], s[other][1]) for s in sweeps if overlap(strip, s)]

                def box(lo, hi, strip=strip, axis=axis, other=other):
                    out = [None, None]
                    out[axis] = tuple(round(v, 4) for v in strip[axis])
                    out[other] = (round(lo, 4), round(hi, 4))
                    return (out[0], out[1])

                def skirt(lo, hi, edge=edge, side=side, axis=axis, other=other):
                    out = [None, None]
                    out[axis] = ((edge - SKIRT_SKIN - WALL_THICK, edge - SKIRT_SKIN) if side == 0
                                 else (edge + SKIRT_SKIN, edge + SKIRT_SKIN + WALL_THICK))
                    out[other] = (lo, hi)
                    return tuple(tuple(round(v, 4) for v in span) for span in out)

                def bridge(lo, hi, edge=edge, side=side, axis=axis, other=other):
                    out = [None, None]                      # over the slot between plate and skirt,
                    out[axis] = ((edge - SKIRT_SKIN - WALL_THICK, edge) if side == 0   # at lintel
                                 else (edge, edge + SKIRT_SKIN + WALL_THICK))          # height
                    out[other] = (lo, hi)
                    return tuple(tuple(round(v, 4) for v in span) for span in out)

                def handles_clear(candidate):
                    return not any(_rect_hits_disc(candidate, centre, radius)
                                   for centre, radius in discs)

                def standable(candidate):
                    return (inside(candidate, deck) and not overlap(candidate, CAVITY)
                            and not any(overlap(candidate, s) for s in raw)
                            and not any(overlap(candidate, r) for r in rects)
                            and handles_clear(candidate))

                for lo, hi in _free_intervals(rect[other], hard, minimum=0.):
                    standing = _free_intervals((lo, hi), doors, WALL_MIN)
                    walls.extend(box(a, b) for a, b in standing)
                    for a, b in _free_intervals((lo, hi), standing, WALL_MIN):
                        lintels.append(box(a, b))
                        steps = max(1, int(round((b - a) / SKIRT_STEP)))
                        cut = [a + (b - a) * i / steps for i in range(steps + 1)]
                        free = [standable(skirt(cut[i], cut[i + 1]))
                                and handles_clear(bridge(cut[i], cut[i + 1])) for i in range(steps)]
                        start = None
                        for i in range(steps + 1):
                            if i < steps and free[i]:
                                start = i if start is None else start
                            elif start is not None:
                                if cut[i] - cut[start] >= WALL_MIN:
                                    skirts.append(skirt(cut[start], cut[i]))
                                    lintels.append(bridge(cut[start], cut[i]))
                                start = None
    return sorted(set(walls)), sorted(set(skirts)), sorted(set(lintels))


def _spread(candidate, legs):
    return min((max(abs(candidate[0] - leg[0]), abs(candidate[1] - leg[1])) for leg in legs),
               default=float('inf'))


def _pick_legs(rects, columns, wanted=4):
    """Up to ``wanted`` legs, spread out: one per corner of the plate's bounding box, then
    whichever free column stands furthest from the legs already chosen.

    The corner pass alone leaves an L-shaped plate on one or two legs, because several of
    its bounding box's corners hang over the notch of the L and resolve to the same column.
    """
    if not columns:
        return []
    box = _bbox(rects)
    legs = []
    for corner in ((box[0][0], box[1][0]), (box[0][1], box[1][0]),
                   (box[0][0], box[1][1]), (box[0][1], box[1][1])):
        ranked = sorted(columns, key=lambda c: ((c[0] - corner[0]) ** 2 + (c[1] - corner[1]) ** 2, c))
        for candidate in ranked:
            if _spread(candidate, legs) > 4 * LEG_HALF:
                legs.append(candidate)
                break
    while len(legs) < wanted:
        best = max(columns, key=lambda c: (_spread(c, legs), c))
        if _spread(best, legs) <= 4 * LEG_HALF:
            break
        legs.append(best)
    return sorted(legs)


def build_plate(layout):
    """The plate for this layout, or ``None`` when the covered engagements cannot be hidden.

    Returns a JSON-plain dict: ``rects`` (chest-local rectangles), ``legs``
    (leg centres), the z bands of the plate and its legs, and the clearance it
    leaves over the sliding layer. ``None`` is not an error, it is a rejection:
    the sampler draws another layout.
    """
    covered = sorted(name for name, flag in layout['covered'].items() if flag)
    if not covered:
        return None
    deck = _pad(deck_rect(layout), -DECK_INSET)
    blocks = keepouts(layout)
    seeds = [engagement_rect(layout, name, HIDE_MARGIN) for name in covered]
    if not all(_free(seed, deck, blocks) for seed in seeds):
        return None
    rects = _merge([_grow(seed, deck, blocks) for seed in seeds], deck, blocks)
    rects = _merge([_grow(rect, deck, blocks) for rect in rects], deck, blocks)
    if not all(covers_rect(rects, seed) for seed in seeds):
        return None
    legs = _pick_legs(rects, _leg_columns(layout, rects))
    if len(legs) < MIN_LEGS:
        return None
    walls, skirts, lintels = wall_boxes(layout, rects)
    return dict(rects=[[list(rect[0]), list(rect[1])] for rect in rects],
                legs=[list(leg) for leg in legs], z=list(PLATE_Z), leg_z=list(LEG_Z),
                leg_half=LEG_HALF, clearance_m=PLATE_CLEAR, hides=covered,
                walls=[[list(wall[0]), list(wall[1])] for wall in walls],
                skirts=[[list(box[0]), list(box[1])] for box in skirts],
                lintels=[[list(bar[0]), list(bar[1])] for bar in lintels],
                wall_z=list(WALL_Z), skirt_z=list(SKIRT_Z), lintel_z=list(LINTEL_Z))


def plate_rects(layout):
    """The plate's rectangles as tuples (empty when the layout has no plate)."""
    plate = layout.get('plate')
    if not plate:
        return []
    return [tuple(tuple(axis) for axis in rect) for rect in plate['rects']]


# --- G0 hardness: how many sliders stay possible for every covered lock -------------------
def sliders(layout):
    return [name for name, part in layout['parts'].items() if part['kind'] != 'lid']


def components(rects):
    """The plate's connected pieces: boxes that touch or overlap are one opaque region.

    Hardness is counted per piece, not over the plate as a whole. Between two pieces the
    deck is in plain view, so a slider that runs under one of them and never reappears in
    the gap cannot reach the other: the observer can rule it out for the locks the far
    piece hides. Counting over the whole plate would let a disconnected plate -- the v1
    cover strips again -- claim candidates it does not really leave open.
    """
    groups = []
    for rect in rects:
        merged, rest = [rect], []
        for group in groups:
            (merged if any(overlap(rect, other, 1e-6) for other in group) else rest).append(group)
        groups = rest + [[rect] + [r for g in merged[1:] for r in g]]
    return groups


def region(layout, lock, rects=None):
    """The connected piece of the plate that hides ``lock``'s engagement (empty when none does)."""
    rects = plate_rects(layout) if rects is None else rects
    seed = engagement_rect(layout, lock)
    for group in components(rects):
        if any(overlap(seed, box) for box in group):
            return group
    return []


def enters_plate(layout, name, rects=None):
    rects = plate_rects(layout) if rects is None else rects
    return any(hits_plate(rects, span) for span in layout['parts'][name]['spans'])


def tip_hidden(layout, name, rects=None):
    """True when the slider's tip is under the plate, i.e. the observer cannot see where it ends."""
    rects = plate_rects(layout) if rects is None else rects
    part = layout['parts'][name]
    axis = part['axis_index']
    tip = [None, None]
    tip[axis] = tuple(sorted((part['tip_end'], part['tip_end'] + part['sigma'] * .002)))
    tip[1 - axis] = (part['perp_center'] - HALF_W, part['perp_center'] + HALF_W)
    return covers_rect(rects, (tip[0], tip[1]))


def committed(layout):
    """Sliders the observer can see holding an uncovered lock: they hold nothing hidden."""
    out = set()
    for name, flag in layout['covered'].items():
        holder = layout['parts'][name].get('blocked_by')
        if not flag and holder:
            out.add(holder)
    return out


def crossing_rect(layout, lock, name):
    """Where ``name``'s straight extrapolated body crosses ``lock``'s body, or None."""
    parts = layout['parts']
    lock_part, part = parts[lock], parts[name]
    axis = part['axis_index']
    if lock_part['kind'] != 'lid' and lock_part['axis_index'] == axis:
        return None                                     # parallel bars never cross
    target = part_bbox(lock_part)
    band = (part['perp_center'] - HALF_W, part['perp_center'] + HALF_W)
    across = (max(band[0], target[1 - axis][0]), min(band[1], target[1 - axis][1]))
    if across[1] - across[0] <= 1e-9:
        return None
    rect = [None, None]
    rect[axis], rect[1 - axis] = target[axis], across
    near = rect[axis][1] if part['sigma'] < 0 else rect[axis][0]
    if part['sigma'] * (near - part['handle'][axis]) > -1e-9:
        return None                                     # not on the tip side of its own handle
    return (rect[0], rect[1])


def candidates(layout, lock, measure=GATE_MEASURE, rects=None):
    """Sliders the observer cannot rule out as the holder of ``lock`` (see the module docstring)."""
    if measure not in MEASURES:
        raise ValueError(f'unknown hardness measure {measure!r}; choose from {MEASURES}')
    rects = plate_rects(layout) if rects is None else rects
    piece = region(layout, lock, rects)          # only the plate's own piece hides this lock
    blocked = committed(layout) if measure == 'strict' else ()
    out = []
    for name in sliders(layout):
        if name == lock or name in blocked or not enters_plate(layout, name, piece):
            continue
        if measure == 'strict':
            cross = crossing_rect(layout, lock, name)
            if not tip_hidden(layout, name, piece) or cross is None or not hits_plate(piece, cross):
                continue
        out.append(name)
    return sorted(out)


def hardness(layout, rects=None):
    """Candidate counts per covered lock under both measures, plus the worst case."""
    rects = plate_rects(layout) if rects is None else rects
    covered = sorted(name for name, flag in layout['covered'].items() if flag)
    report = {}
    for measure in MEASURES:
        per_lock = {lock: candidates(layout, lock, measure, rects) for lock in covered}
        report[measure] = dict(per_lock=per_lock, counts={k: len(v) for k, v in per_lock.items()},
                               worst=min((len(v) for v in per_lock.values()), default=None),
                               holder_missing=sorted(lock for lock in covered
                                                     if layout['parts'][lock]['blocked_by'] not in per_lock[lock]))
    return report


def hardness_problems(layout, measure=GATE_MEASURE, minimum=MIN_CANDIDATES, rects=None):
    """G0: every covered lock keeps at least ``minimum`` candidate holders."""
    rects = plate_rects(layout) if rects is None else rects
    problems = []
    for lock, flag in sorted(layout['covered'].items()):
        if not flag:
            continue
        names = candidates(layout, lock, measure, rects)
        holder = layout['parts'][lock]['blocked_by']
        if holder not in names:
            problems.append(f'hardness: the holder {holder} of {lock} is not a {measure} candidate')
        if len(names) < minimum:
            problems.append(f'hardness: {lock} has {len(names)} {measure} candidate(s) '
                            f'({", ".join(names) or "none"}), fewer than {minimum}')
    return problems


# --- validation -------------------------------------------------------------

def plate_problems(layout, measure=GATE_MEASURE, minimum=MIN_CANDIDATES):
    """Problems with the layout's recorded plate, checked from the layout alone.

    Independent of :func:`build_plate`: a plate that came from anywhere has to
    hide every covered engagement in one box, stay off every handle over its
    travel, off the hand's column over the item and off the engagements the
    instance leaves visible, stay inside the deck, stand on at least
    ``MIN_LEGS`` legs in columns no slider sweep enters, and pass G0.
    """
    covered = sorted(name for name, flag in layout['covered'].items() if flag)
    plate = layout.get('plate')
    if not covered:
        return ['plate: an uncovered layout carries a plate'] if plate else []
    if not plate:
        return ['plate: covered locks but no plate']
    problems = []
    rects = plate_rects(layout)
    deck = _pad(deck_rect(layout), -DECK_INSET)
    blocks = keepouts(layout)
    if sorted(plate.get('hides', ())) != covered:
        problems.append(f"plate: hides {sorted(plate.get('hides', ()))}, the covered locks are {covered}")
    if list(plate.get('z', ())) != list(PLATE_Z) or plate.get('clearance_m') != PLATE_CLEAR:
        problems.append(f"plate: z {plate.get('z')} is not {list(PLATE_Z)} over the sliding layer")
    for rect in rects:
        if not inside(rect, deck, 1e-9):
            problems.append(f'plate: box {rect} leaves the deck')
        if not _free(rect, deck, blocks):
            problems.append(f'plate: box {rect} reaches over a handle, the item or a visible engagement')
    for lock in covered:
        if not covers_rect(rects, engagement_rect(layout, lock, HIDE_MARGIN)):
            problems.append(f'plate: the engagement of {lock} is not covered')
    legs = [tuple(leg) for leg in plate.get('legs', ())]
    if len(legs) < MIN_LEGS:
        problems.append(f'plate: {len(legs)} legs, fewer than {MIN_LEGS}')
    sweeps = [s for part in layout['parts'].values() for s in swept_rects(part)]
    cavity = _pad(CAVITY, LEG_GAP)
    for leg in legs:
        column = ((leg[0] - LEG_HALF - LEG_GAP, leg[0] + LEG_HALF + LEG_GAP),
                  (leg[1] - LEG_HALF - LEG_GAP, leg[1] + LEG_HALF + LEG_GAP))
        if not covers_rect(rects, ((leg[0] - LEG_HALF, leg[0] + LEG_HALF), (leg[1] - LEG_HALF, leg[1] + LEG_HALF))):
            problems.append(f'plate: the leg at {leg} does not stand under the plate')
        if overlap(column, cavity) or any(overlap(column, s) for s in sweeps):
            problems.append(f'plate: the leg at {leg} stands in a slider sweep or the cavity')
    problems.extend(hardness_problems(layout, measure, minimum, rects))
    return problems
