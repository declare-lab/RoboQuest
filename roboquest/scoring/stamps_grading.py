"""Per-cell grading of the dots printed on a stamps board.

Every print on the final board is recorded with its stamp, translation and yaw, so the centre of every printed dot
follows from the stamp's die mask. Each dot belongs to the cell holding its centre. The instance's solution
prescribes a dot count k for every target cell. A target cell grades 1 with only clean dots (each centre within
CELL_SPACING/2 - DOT_RADIUS of the cell centre on both axes) and no more than k of them, 0.5 with more than k or
any dot crossing an edge, 0 when empty; every dot on the paper outside the target cells is a stray (a dot beyond
the paper's print area leaves nothing and costs nothing).

roboquest/scoring/progress.py scores stamps progress from these grades; the task's live scorer applies the same
grading for success (every target cell perfect, no stray, stamps released and still, off-grid ink within 35 px).
"""
import math

# The scorer's geometry (roboquest/stamps/rules.py), copied so the grading needs no numpy.
CELL_SPACING = .034
DOT_RADIUS = .008
PAPER_HALF = .084
MAX_OFF_GRID_PIXELS = 35

PERFECT, IMPERFECT, MISSING = 1., .5, 0.
STRAY_COST = 1.


def dot_centres(spec, impressions):
    """Board-frame centres of every printed dot, with the impression index each came from (the scorer's
    convention: mask cell (row, col) prints at ((col-1), (row-1)) * CELL_SPACING, rotated by yaw about +z, then
    translated)."""
    centres = []
    for index, (stamp, translation, yaw, _) in enumerate(impressions):
        mask = spec['stamps'][stamp]['die_mask_private']
        c, s = math.cos(yaw), math.sin(yaw)
        for row in range(3):
            for col in range(3):
                if not mask[row][col]:
                    continue
                lx, ly = (col - 1) * CELL_SPACING, (row - 1) * CELL_SPACING
                centres.append((c * lx - s * ly + translation[0], s * lx + c * ly + translation[1], index))
    return centres


def cell_of(x, y):
    """(row, col) of the cell holding a centre, or None when the centre is off the 3x3 grid."""
    col = int(round(x / CELL_SPACING)) + 1
    row = int(round(y / CELL_SPACING)) + 1
    if 0 <= row <= 2 and 0 <= col <= 2:
        return row, col
    return None


def fully_inside(x, y, row, col):
    margin = CELL_SPACING / 2 - DOT_RADIUS
    return abs(x - (col - 1) * CELL_SPACING) <= margin and abs(y - (row - 1) * CELL_SPACING) <= margin


def off_paper(x, y):
    """A dot printed beyond the paper's print area leaves no ink on the board."""
    return max(abs(x), abs(y)) > PAPER_HALF + DOT_RADIUS


def prescribed_counts(spec):
    """Dots per cell of the minted solution at exact placement (the design's k), or None when the solution is
    missing or does not reproduce ``target_cells`` under the rotation convention (rotation index times +90 deg)."""
    solution, names = spec.get('solution_private'), spec.get('stamp_order') or []
    if not solution or not names:
        return None
    target = [[bool(v) for v in row] for row in spec['target_cells']]
    counts = [[0] * 3 for _ in range(3)]
    for stamp_index, rotation in solution:
        placed = [(names[stamp_index], [0., 0.], rotation * math.pi / 2, None)]
        for x, y, _ in dot_centres(spec, placed):
            cell = cell_of(x, y)
            if cell is None:
                return None
            counts[cell[0]][cell[1]] += 1
    return counts if [[c > 0 for c in row] for row in counts] == target else None


def grade(centres, target, count_off_paper=False, prescribed=None):
    """Per-cell verdicts and the pattern score for a list of dot centres against the target mask; ``prescribed``
    holds the design's dot count per cell (every k is 1 without it)."""
    target = [[bool(v) for v in row] for row in target]
    per_cell = {}
    strays = strays_off_paper = 0
    for x, y, _ in centres:
        cell = cell_of(x, y)
        if cell is None or not target[cell[0]][cell[1]]:
            if cell is None and off_paper(x, y):
                strays_off_paper += 1
                if not count_off_paper:
                    continue
            strays += 1
            if cell is not None:
                per_cell.setdefault(cell, []).append(False)
            continue
        per_cell.setdefault(cell, []).append(fully_inside(x, y, *cell))
    grades, verdicts = 0., {}
    for row in range(3):
        for col in range(3):
            if not target[row][col]:
                continue
            dots = per_cell.get((row, col), [])
            k = max(1, prescribed[row][col]) if prescribed else 1
            if not dots:
                verdict, value = 'missing', MISSING
            elif all(dots) and len(dots) <= k:          # one clean dot, or the design's own multiple
                verdict, value = ('perfect' if len(dots) == 1 else 'design_multiple'), PERFECT
            elif len(dots) == 1:                        # a single dot crossing an edge
                verdict, value = 'partial', IMPERFECT
            else:                                       # several dots: more than the design's, or not all clean
                verdict, value = 'overpainted', IMPERFECT
            verdicts[(row, col)] = verdict
            grades += value
    n_target = sum(v for row in target for v in row)
    pattern = max(0., (grades - STRAY_COST * strays) / n_target) if n_target else 0.
    counts = dict(perfect=sum(v == 'perfect' for v in verdicts.values()),
                  design_multiple=sum(v == 'design_multiple' for v in verdicts.values()),
                  partial=sum(v == 'partial' for v in verdicts.values()),
                  overpainted=sum(v == 'overpainted' for v in verdicts.values()),
                  missing=sum(v == 'missing' for v in verdicts.values()), stray=strays,
                  stray_off_paper=strays_off_paper)
    return pattern, counts
