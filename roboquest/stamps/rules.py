"""Stamp composition rules: mask bank, ink geometry, board bitmaps, scoring and the
uniqueness gate. Pure numpy; no simulator imports (the CPU gate runs on this).

Conventions (unchanged from ``family5_stamps_rules``/``stamps_v2_rules``):
a mask is a 3x3 boolean array indexed ``[row, col]``; rows advance in board
local +y, columns in +x, so cell (row, col) prints a dot at
``((col-1), (row-1)) * CELL_SPACING``. ``rotate_mask(mask, k)`` is a +z rotation
by ``k`` quarter turns in those coordinates. Ink bitmaps are boolean rasters
over a square board of half size ``half`` with row index advancing in +y.
"""
import itertools

import numpy as np

CELL_SPACING = .034
DOT_RADIUS = .008
PAPER_HALF = .084             # final board and reference card print area (16.8 cm square)
RESOLUTION = 384              # final/reference raster (pixels per side)
SCRATCH_RESOLUTION = 512
SCRATCH_MARGIN = .032         # clear margin around the scratch workspace (holds the corner tags)
MARKER_OFFSET = .068
MARKER_HALF = .008
PAPER_RGB = 245
INK_RGB = np.array([12, 30, 150], dtype=np.uint8)
GRID_RGB = np.array([207, 210, 213], dtype=np.uint8)
MARKER_RGB = ((225, 30, 25), (25, 185, 35), (25, 45, 225), (225, 175, 20))
MARKER_XY = ((-1, -1), (1, -1), (1, 1), (-1, 1))   # SW, SE, NE, NW corner tags
CELL_INK_FRACTION = .035      # a cell counts as printed above this ink fraction
MAX_OFF_GRID_PIXELS = 35      # antialias/raster-edge tolerance, not another dot
MIN_DOTS, MAX_DOTS = 2, 4
BANK_SIZE = 12
# Paper stock: a per-instance appearance draw (spec 4.9). Every entry is a light,
# unprinted paper so the blue ink and the grey grid keep their contrast; the ink
# model and the scoring read the boolean bitmap, never a colour.
PAPER_PALETTE = ((245, 245, 245), (243, 238, 226), (238, 240, 244), (246, 241, 232),
                 (235, 236, 231), (248, 244, 238))

# Twelve die patterns, (row, col) cells, 2-4 dots each. Chosen by hand for print
# legibility; no two are equal under rotation and none is invariant under any
# rotation, so every stamp has four distinct impressions. Checked by the tests.
MASK_BANK = (
    ((0, 0), (0, 1)),                        # 0  corner + adjacent edge
    ((0, 0), (1, 1)),                        # 1  corner + centre
    ((0, 1), (1, 2)),                        # 2  two edges, knight apart
    ((0, 0), (1, 2)),                        # 3  corner + far edge
    ((0, 0), (0, 1), (1, 1)),                # 4  corner, edge, centre
    ((0, 0), (0, 2), (1, 1)),                # 5  two corners + centre
    ((0, 1), (1, 0), (1, 1)),                # 6  L of two edges around the centre
    ((0, 0), (0, 1), (0, 2)),                # 7  full side row
    ((0, 0), (0, 1), (1, 2)),                # 8  corner, edge, side edge
    ((0, 0), (0, 1), (1, 0), (1, 1)),        # 9  2x2 block in a corner
    ((0, 0), (0, 2), (1, 1), (2, 1)),        # 10 two corners, centre, opposite edge
    ((0, 0), (0, 1), (0, 2), (1, 1)),        # 11 T: side row + centre
)


# ---- masks -----------------------------------------------------------------
def mask_array(cells):
    result = np.zeros((3, 3), dtype=bool)
    for row, col in cells:
        result[int(row), int(col)] = True
    return result


def mask_cells(mask):
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(np.asarray(mask, dtype=bool)))]


def mask_bits(mask):
    """Nine-bit integer encoding, bit 3*row+col."""
    mask = np.asarray(mask, dtype=bool)
    return int(sum(1 << (3 * r + c) for r, c in zip(*np.nonzero(mask))))


def bits_mask(bits):
    return np.array([[(int(bits) >> (3 * r + c)) & 1 for c in range(3)] for r in range(3)], dtype=bool)


def rotate_mask(mask, quarter_turns):
    """World +z rotation by ``quarter_turns`` * 90 deg in row=+y array coordinates."""
    return np.rot90(np.asarray(mask, dtype=bool), -int(quarter_turns) % 4)


def points_for_mask(mask, spacing=CELL_SPACING):
    """Dot centres (x, y) in board/stamp-local metres, row-major over the die's cells."""
    row, col = np.nonzero(np.asarray(mask, dtype=bool))
    return np.c_[(col - 1) * float(spacing), (row - 1) * float(spacing)]


def bank_mask(index):
    return mask_array(MASK_BANK[int(index)])


def distinct_rotations(mask):
    """(quarter_turns, bits) for each distinct impression of ``mask``; symmetric masks repeat fewer."""
    seen, out = set(), []
    for k in range(4):
        bits = mask_bits(rotate_mask(mask, k))
        if bits not in seen:
            seen.add(bits)
            out.append((k, bits))
    return out


def canonical_bits(mask):
    """Smallest bit encoding over the four rotations: the rotation class of a pattern."""
    return min(mask_bits(rotate_mask(mask, k)) for k in range(4))


# All 140 rotation classes of a 3x3 pattern (Burnside: (512 + 8 + 8 + 32) / 4), sorted.
TARGET_CLASSES = tuple(sorted({canonical_bits(bits_mask(b)) for b in range(512)}))
TARGET_CLASS_COUNT = len(TARGET_CLASSES)


def target_class_id(mask):
    """Index of the target's rotation class in ``TARGET_CLASSES``; the structure class of an instance."""
    return TARGET_CLASSES.index(canonical_bits(mask))


# ---- composition and uniqueness --------------------------------------------
def union_of(masks, turns):
    result = np.zeros((3, 3), dtype=bool)
    for mask, k in zip(masks, turns):
        result |= rotate_mask(mask, k)
    return result


def solutions(masks, target):
    """Every way to print ``target`` as a union of impressions, each stamp used at most once.

    Rotations that give the same impression (symmetric masks) are counted once.
    Returns a sorted list of tuples of (stamp index, quarter turns).
    """
    target_bits = mask_bits(target)
    options = [distinct_rotations(mask) for mask in masks]
    found = []
    for size in range(1, len(masks) + 1):
        for subset in itertools.combinations(range(len(masks)), size):
            for choice in itertools.product(*(options[i] for i in subset)):
                bits = 0
                for _, image in choice:
                    bits |= image
                if bits == target_bits:
                    found.append(tuple((i, k) for i, (k, _) in zip(subset, choice)))
    return sorted(found)


def unique_solution(masks, target):
    """G0: exactly one (subset, rotations) prints the target; then no smaller or larger subset does."""
    found = solutions(masks, target)
    return len(found) == 1, found


def unique_targets(masks, size):
    """Every target a subset of exactly ``size`` dies prints that *no other* combination prints.

    One enumeration of all ``sum_s C(n, s) * 4**s`` impressions (3124 of them for
    five dies) yields both the printable patterns and how many ways each is
    reachable, so the generator can pick a uniquely solvable target directly
    instead of drawing (subset, rotations) at random and testing them. The
    acceptance rule is exactly :func:`unique_solution`'s — a target is kept only
    when the whole enumeration reaches it once — but the cost per instance falls
    from the reciprocal of the raw uniqueness rate (1 in 20 000 draws at
    composition 4) to a single pass over the die set.

    Returns a list of ``(bits, solution)`` sorted by ``bits``, where ``solution``
    is the tuple of ``(stamp index, quarter turns)`` that prints it.
    """
    size = int(size)
    options = [distinct_rotations(mask) for mask in masks]
    counts, owners = {}, {}
    for subset_size in range(1, len(masks) + 1):
        for subset in itertools.combinations(range(len(masks)), subset_size):
            for choice in itertools.product(*(options[i] for i in subset)):
                bits = 0
                for _, image in choice:
                    bits |= image
                counts[bits] = counts.get(bits, 0) + 1
                if subset_size == size and bits not in owners:
                    owners[bits] = tuple((i, k) for i, (k, _) in zip(subset, choice))
    return sorted((bits, owners[bits]) for bits in owners if counts[bits] == 1)


# ---- ink rasters -------------------------------------------------------------
def raster_coordinates(resolution, half):
    centers = ((np.arange(int(resolution)) + .5) / int(resolution) * 2 - 1) * float(half)
    return np.meshgrid(centers, centers)


def deposit(ink, mask, translation_xy, yaw, half):
    """Copy of ``ink`` with circular dots for ``mask`` at the real rigid pose on a board of ``half`` size."""
    ink = np.asarray(ink, dtype=bool).copy()
    x, y = raster_coordinates(ink.shape[0], half)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    centers = points_for_mask(mask) @ rotation.T + np.asarray(translation_xy, dtype=float)
    for px, py in centers:
        ink |= (x - px) ** 2 + (y - py) ** 2 <= DOT_RADIUS ** 2
    return ink


def paper_rgb(ink, paper=PAPER_RGB):
    """Final board / reference card image: light grid, four colour corner tags, blue ink.

    ``paper`` is the instance's paper stock (a scalar or an RGB triple): appearance
    variation only. The grid, the corner tags and the ink keep their fixed colours.
    """
    x, y = raster_coordinates(ink.shape[0], PAPER_HALF)
    image = np.empty((*ink.shape, 3), dtype=np.uint8)
    image[:] = np.asarray(paper, dtype=np.uint8)
    for value in (-1.5, -.5, .5, 1.5):
        line = value * CELL_SPACING
        grid = (((np.abs(x - line) < .0005) & (np.abs(y) < 1.5 * CELL_SPACING)) |
                ((np.abs(y - line) < .0005) & (np.abs(x) < 1.5 * CELL_SPACING)))
        image[grid] = GRID_RGB
    for (sx, sy), rgb in zip(MARKER_XY, MARKER_RGB):
        image[(np.abs(x - sx * MARKER_OFFSET) <= MARKER_HALF) & (np.abs(y - sy * MARKER_OFFSET) <= MARKER_HALF)] = rgb
    image[ink] = INK_RGB
    return image


def scratch_half(cells):
    """Half size of the unlined scratch board with ``cells`` cell widths of clear workspace."""
    return float(int(cells) / 2 * CELL_SPACING + SCRATCH_MARGIN)


def scratch_rgb(ink, half, paper=PAPER_RGB):
    """Unlined scratch board: four corner tags orient it, no grid, cumulative ink."""
    x, y = raster_coordinates(ink.shape[0], half)
    image = np.empty((*ink.shape, 3), dtype=np.uint8)
    image[:] = np.asarray(paper, dtype=np.uint8)
    tag = float(half) - .016
    for (sx, sy), rgb in zip(MARKER_XY, MARKER_RGB):
        image[(np.abs(x - sx * tag) < .007) & (np.abs(y - sy * tag) < .007)] = rgb
    image[ink] = INK_RGB
    return image


def ink_cells(ink):
    """Occupied 3x3 cells of a final-board raster, per-cell ink fractions and off-grid pixels."""
    x, y = raster_coordinates(ink.shape[0], PAPER_HALF)
    cells = np.zeros((3, 3), dtype=bool)
    fractions = np.zeros((3, 3), dtype=float)
    allowed = np.zeros_like(ink, dtype=bool)
    for row, col in itertools.product(range(3), repeat=2):
        area = ((np.abs(x - (col - 1) * CELL_SPACING) < CELL_SPACING / 2 - .001) &
                (np.abs(y - (row - 1) * CELL_SPACING) < CELL_SPACING / 2 - .001))
        allowed |= area
        fractions[row, col] = np.mean(ink[area])
        cells[row, col] = fractions[row, col] >= CELL_INK_FRACTION
    return cells, fractions, int(np.count_nonzero(ink & ~allowed))


def contact_print_eligible(rotation, normal_on_stamp, sole_height_error, normal_force):
    """Physical face-contact gate for an impression; no identity or target condition."""
    return bool(rotation[2, 2] >= np.cos(np.deg2rad(8)) and normal_force > 1e-5
                and normal_on_stamp[2] >= .9 and abs(sole_height_error) <= .003)


# ---- dot-level scoring -------------------------------------------------------
# The score comes from the impression list and the die masks, not from the board bitmap:
# ``ink_cells`` only says which cells carry ink, so a cell printed twice 7 mm apart reads
# exactly like one centred dot. The geometry below is ``deposit``'s, so every graded dot is
# a dot the raster really printed.
PERFECT_DOT, EDGE_DOT, NO_DOT = 1., .5, 0.
STRAY_DOT_COST = 1.           # a dot on the paper outside the target costs a whole cell
FINISH_WEIGHT = .1            # progress = (1 - FINISH_WEIGHT) * pattern + FINISH_WEIGHT * finish


def dot_centres(impressions, die_masks, spacing=CELL_SPACING):
    """Board-frame centres of every dot an impression list prints, in print order.

    ``impressions`` is a sequence of ``(stamp, (x, y), yaw)`` in the board frame and
    ``die_masks`` maps a stamp key to its 3x3 die mask. The convention is
    :func:`deposit`'s: die cell (row, col) prints at ``((col - 1), (row - 1)) * spacing``
    in stamp-local metres, turned by ``yaw`` about +z and translated. Returns
    ``(x, y, impression index)`` triples.
    """
    out = []
    for index, (stamp, translation, yaw) in enumerate(impressions):
        c, s = np.cos(float(yaw)), np.sin(float(yaw))
        rotation = np.array([[c, -s], [s, c]])
        centres = points_for_mask(die_masks[stamp], spacing) @ rotation.T + np.asarray(translation, float)
        out.extend((float(x), float(y), int(index)) for x, y in centres)
    return out


def dot_cell(x, y, spacing=CELL_SPACING):
    """(row, col) of the 3x3 cell holding a dot centre, or ``None`` when it is off the grid."""
    col = int(round(float(x) / float(spacing))) + 1
    row = int(round(float(y) / float(spacing))) + 1
    return (row, col) if 0 <= row <= 2 and 0 <= col <= 2 else None


def dot_on_paper(x, y, half=PAPER_HALF, radius=DOT_RADIUS):
    """Whether a dot reaches the print area at all; one that does not leaves no ink and costs nothing."""
    return bool(max(abs(float(x)), abs(float(y))) <= float(half) + float(radius))


def solution_dot_counts(masks, solution):
    """How many of a solution's dies print each cell: the target's own dot multiplicity.

    ``masks`` are the instance's die masks in stamp order and ``solution`` its
    ``(stamp, quarter turns)`` pairs. :func:`union_of` ORs exactly these images, so the
    boolean of this count is the target, and a cell with a count of k carries k dots when
    the solution is pressed exactly - the design's own double.
    """
    counts = np.zeros((3, 3), dtype=int)
    for index, turns in solution:
        counts += rotate_mask(masks[int(index)], turns).astype(int)
    return counts


def grade_impressions(impressions, die_masks, target, expected_counts=None, half=PAPER_HALF,
                      spacing=CELL_SPACING, radius=DOT_RADIUS):
    """Dot-level score of an impression list against a 3x3 target.

    Every printed dot belongs to the cell holding its centre, and a dot is clean when it is
    fully inside that cell (centre within ``spacing / 2 - radius``, 9 mm, of the cell centre
    on both axes). ``expected_counts`` is the number of dots the design itself puts in each
    cell - :func:`solution_dot_counts` of the private solution - and defaults to one
    everywhere. A target cell grades 1 when it holds any 1 to k clean dots and nothing else,
    0 when it holds no dot, and 0.5 for anything else (more dots than the design prints, so a
    policy-made double where k is 1, an edge-crossing dot, or k clean dots plus a partial).
    The ladder is monotone in the dots that land right. Every dot whose centre lands
    on the paper but not in a target cell - an empty cell or the grid margin - costs
    ``STRAY_DOT_COST``; a dot pressed past the print area puts nothing on the board and
    costs nothing. ::

        pattern = max(0, (sum of the target grades - stray dots) / number of target cells)

    Pure: the caller supplies the impressions, the die masks, the target and the geometry.
    Returns ``dict(pattern, target_grades, stray_dots, dots)`` where ``target_grades`` has one
    entry per target cell (row-major, ``cell``/``grade``/``verdict``/``dots``/``expected``) and
    ``dots`` one per printed dot in print order (``x_m``/``y_m``/``impression``/``cell``/
    ``inside``/``grade`` of the cell it lands in/``stray``/``off_paper``).
    """
    target = np.asarray(target, dtype=bool)
    expected = (np.ones((3, 3), dtype=int) if expected_counts is None
                else np.asarray(expected_counts, dtype=int))
    margin = float(spacing) / 2 - float(radius)
    hits, dots, strays = {}, [], 0
    for x, y, index in dot_centres(impressions, die_masks, spacing):
        cell = dot_cell(x, y, spacing)
        on_target = cell is not None and bool(target[cell])
        inside = bool(on_target and abs(x - (cell[1] - 1) * spacing) <= margin
                      and abs(y - (cell[0] - 1) * spacing) <= margin)
        off_paper = not dot_on_paper(x, y, half, radius)
        stray = bool(not on_target and not off_paper)
        strays += int(stray)
        if on_target:
            hits.setdefault(cell, []).append(len(dots))
        dots.append(dict(x_m=x, y_m=y, impression=index,
                         cell=None if cell is None else [cell[0], cell[1]],
                         inside=inside, grade=None, stray=stray, off_paper=bool(off_paper)))
    grades = []
    for row, col in itertools.product(range(3), repeat=2):
        if not target[row, col]:
            continue
        here = hits.get((row, col), [])
        want = int(expected[row, col])
        clean = all(dots[i]['inside'] for i in here)
        if not here:
            verdict, value = 'missing', NO_DOT
        elif clean and len(here) == 1:
            verdict, value = 'perfect', PERFECT_DOT
        elif clean and len(here) <= want:
            verdict, value = 'design_double', PERFECT_DOT
        elif len(here) == 1:
            verdict, value = 'partial', EDGE_DOT
        else:
            verdict, value = 'overpainted', EDGE_DOT
        for i in here:
            dots[i]['grade'] = float(value)
        grades.append(dict(cell=[row, col], grade=float(value), verdict=verdict, dots=len(here),
                           expected=want))
    count = int(np.count_nonzero(target))
    total = float(sum(g['grade'] for g in grades))
    pattern = max(0., (total - STRAY_DOT_COST * strays) / count) if count else 0.
    return dict(pattern=float(pattern), target_grades=grades, stray_dots=int(strays), dots=dots)
