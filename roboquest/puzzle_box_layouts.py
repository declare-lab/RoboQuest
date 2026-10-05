"""Procedural puzzle box layouts: sampled lock chains, decoys, covers and props.

A layout is plain data in the box frame (origin at the chest centre on the
table top, X right, Y away from the robot). Every sliding part is a set of
axis-aligned rectangles in one 12 mm layer plus a round handle post. A lock is
one bolt whose tip rests inside a notch of the next part; withdrawing the bolt
along its own axis frees that part. Chains alternate axes because a tip can
only enter a notch from the side.

Sampling is deterministic per seed. `validate_layout` rejects layouts whose
parts could touch at rest or while sliding, whose handles are too close for the
open gripper, or whose handles leave the robot's reach band. The evaluator-side
truth (chain, release order, covers) lives here; the robot only sees the scene.
"""
from __future__ import annotations

import math
import random

HALF_W = .015                 # bolt half width
BOLT_TRAVEL = .05
NOTCH_LEN = .04               # notch width along the target
SIDE_NOTCH_DEPTH = HALF_W     # side notch reaches the target's centre line
LID_NOTCH_DEPTH = .03
TIP_ENGAGEMENT = .013         # bolt tip inside a side notch
LID_TIP_END_PLAY = .004
BOLT_RELEASED_AT = .02
LID_BOLT_RELEASED_AT = .03
LID_HALF = (.12, .08)
CAVITY = ((-.11, .11), (-.07, .07))
LID_TRAVEL = {'x+': .17, 'x-': .17, 'y+': .14}
DECK_X = (-.30, .30)
DECK_Y = {'x+': (-.30, .10), 'x-': (-.30, .10), 'y+': (-.30, .25)}
OVERHANG = .035
REACH_X = (-.33, .33)
REACH_Y = (-.29, .12)
HANDLE_SEP = (.068, .033)     # no other handle inside |dx|<.068 and |dy|<.033 of a grasped one
HAND_OVER_ITEM = (.125, .05)  # no handle inside this box around the item while the hand reaches into the cavity
HANDLE_INSET = .02            # handle centre from a bolt's outer end
HANDLE_RADIUS = .013
GAP = .004                    # minimum clearance between unrelated parts, at rest and while sliding
LOCK_GAP = .0015              # a tip inside its notch keeps 2 mm end play and 5 mm side play
COVER_MARGIN = .012
COVER_PROBABILITY = .4
# Difficulty tiers: covers hide which tip sits in whose notch. 'visible' has none.
TIERS = {'visible': dict(cover_probability=0.), 'mixed': dict(cover_probability=COVER_PROBABILITY)}
CHAIN_LENGTHS = (2, 3, 4)
SLIDERS = 5
BOLT_LENGTHS = (.10, .12, .14)
FIRST_BOLT_LENGTHS = (.11, .13, .15)
DECOY_LENGTHS = (.09, .11, .13)
LID_DIRECTIONS = ('x+', 'x-', 'y+')
# RoboCasa objects whose measured collision extent fits the cavity and an 80 mm gripper.
ITEM_CANDIDATES = ('candle/candle_0', 'tangerine/tangerine_0', 'tangerine/tangerine_1',
                   'tangerine/tangerine_2', 'tangerine/tangerine_3', 'tomato/tomato_2')
TRAY_CANDIDATES = ('tray/tray_0', 'tray/tray_2', 'tray/tray_4')
WOOD_PALETTES = (((.55, .38, .22), (.66, .50, .32), (.74, .56, .33)),
                 ((.42, .28, .16), (.52, .36, .22), (.60, .42, .26)),
                 ((.62, .48, .30), (.72, .58, .38), (.80, .66, .44)),
                 ((.36, .30, .26), (.46, .40, .34), (.56, .50, .42)))
HANDLE_COLOURS = ((.16, .16, .17), (.10, .12, .30), (.30, .10, .10), (.12, .26, .14))
METAL = (.58, .60, .63)


# --- rectangles -----------------------------------------------------------

def overlap(a, b, gap=0.):
    """True when the rectangles overlap or come closer than `gap` on both axes."""
    return all(min(a[i][1], b[i][1]) - max(a[i][0], b[i][0]) > -gap + 1e-9 for i in range(2))


def pair_gap(p, q):
    """Required clearance for a pair of parts: locked pairs keep only their design play."""
    locked = p.get('blocked_by') == q['name'] or q.get('blocked_by') == p['name']
    return LOCK_GAP if locked else GAP


def shift(rect, axis, d):
    """Translate a rect by the 2D vector axis*d."""
    return tuple((rect[i][0] + axis[i] * d, rect[i][1] + axis[i] * d) for i in range(2))


def sweep(rect, axis, lo, hi):
    """Union of rect translated by axis*d for all d in [lo, hi]."""
    out = []
    for i in range(2):
        shifts = (axis[i] * lo, axis[i] * hi)
        out.append((rect[i][0] + min(shifts), rect[i][1] + max(shifts)))
    return tuple(out)


def inside(rect, bounds, slack=0.):
    return all(rect[i][0] >= bounds[i][0] - slack and rect[i][1] <= bounds[i][1] + slack for i in range(2))


# --- part constructors ----------------------------------------------------

def _rect(axis_index, along, across):
    r = [None, None]
    r[axis_index], r[1 - axis_index] = tuple(along), tuple(across)
    return (tuple(r[0]), tuple(r[1]))


def make_bolt(name, axis_index, sigma, tip_end, perp_center, length, notch_q=None, notch_side=None,
              decoy=False):
    """A bolt lying along axis `axis_index`; `sigma` is its withdrawal sign.

    The tip end sits at coordinate `tip_end` along the axis; the body extends
    `length` in the withdrawal direction. An optional side notch (for the next
    bolt) is centred `notch_q` from the tip end on side `notch_side`.
    """
    a, b = axis_index, 1 - axis_index
    if notch_q is not None and not (NOTCH_LEN <= notch_q <= length - HANDLE_INSET - HANDLE_RADIUS - .012):
        raise ValueError(f'{name}: notch position {notch_q} outside the usable body')
    lo, hi = sorted((tip_end, tip_end + sigma * length))
    across = (perp_center - HALF_W, perp_center + HALF_W)
    spans, notch = [], None
    if notch_q is not None:
        centre = tip_end + sigma * notch_q
        n_along = (centre - NOTCH_LEN / 2, centre + NOTCH_LEN / 2)
        n_across = tuple(sorted((perp_center + notch_side * HALF_W, perp_center)))
        notch = _rect(a, n_along, n_across)
        spans.append(_rect(a, (lo, hi), tuple(sorted((perp_center, perp_center - notch_side * HALF_W)))))
        spans.append(_rect(a, (lo, n_along[0]), n_across))
        spans.append(_rect(a, (n_along[1], hi), n_across))
    else:
        spans.append(_rect(a, (lo, hi), across))
    handle = [None, None]
    handle[a], handle[b] = tip_end + sigma * (length - HANDLE_INSET), perp_center
    axis = [0., 0.]
    axis[a] = float(sigma)
    return dict(name=name, kind='decoy' if decoy else 'bolt', axis_index=a, sigma=sigma, tip_end=tip_end,
                perp_center=perp_center, length=length, axis=tuple(axis),
                range=(-BOLT_TRAVEL, BOLT_TRAVEL) if decoy else (0., BOLT_TRAVEL),
                spans=spans, notch=notch, notch_side=notch_side, notch_q=notch_q, cover=None,
                handle=(handle[0], handle[1]), blocked_by=None, released_at=None, open_at=None,
                decoy=decoy)


def make_lid(direction, slot, side=None, travel=None):
    """The lid; ``travel`` overrides ``LID_TRAVEL[direction]`` (generator v3 opens x lids farther)."""
    hx, hy = LID_HALF
    if direction in ('x+', 'x-'):
        sigma = 1 if direction == 'x+' else -1
        xs = slot
        notch = ((xs - NOTCH_LEN / 2, xs + NOTCH_LEN / 2), (-hy, -hy + LID_NOTCH_DEPTH))
        spans = [((-hx, hx), (-hy + LID_NOTCH_DEPTH, hy)),
                 ((-hx, xs - NOTCH_LEN / 2), (-hy, -hy + LID_NOTCH_DEPTH)),
                 ((xs + NOTCH_LEN / 2, hx), (-hy, -hy + LID_NOTCH_DEPTH))]
        handle = (.05 * sigma, -.02)
        axis = (float(sigma), 0.)
        blocker = dict(axis_index=1, sigma=-1, tip_end=-hy + LID_NOTCH_DEPTH - LID_TIP_END_PLAY, perp_center=xs)
    else:
        sigma, ys = 1, slot
        nx = tuple(sorted((side * hx, side * (hx - LID_NOTCH_DEPTH))))
        notch = (nx, (ys - NOTCH_LEN / 2, ys + NOTCH_LEN / 2))
        spans = [(tuple(sorted((-side * hx, side * (hx - LID_NOTCH_DEPTH)))), (-hy, hy)),
                 (nx, (-hy, ys - NOTCH_LEN / 2)), (nx, (ys + NOTCH_LEN / 2, hy))]
        handle = (0., -.05)
        axis = (0., 1.)
        blocker = dict(axis_index=0, sigma=side, tip_end=side * (hx - LID_NOTCH_DEPTH + LID_TIP_END_PLAY), perp_center=ys)
    travel = LID_TRAVEL[direction] if travel is None else float(travel)
    for span in spans:                    # a notch flush with an edge is an open corner: the
        if min(span[0][1] - span[0][0], span[1][1] - span[1][0]) <= 0.:   # bolt tip would leave
            raise ValueError(f'lid slot {slot} leaves no solid lid beside the notch')
    return dict(name='lid', kind='lid', direction=direction, axis=axis, range=(0., travel), spans=spans,
                notch=notch, cover=None, handle=handle, blocked_by=None, released_at=None,
                open_at=travel - .02, decoy=False, blocker_seed=blocker, slot=slot, side=side)


def make_blocker(name, target, length, notch_q=None, notch_side=None):
    """The bolt whose tip rests in `target`'s notch; it withdraws outward."""
    if target['kind'] == 'lid':
        seed = target['blocker_seed']
        bolt = make_bolt(name, seed['axis_index'], seed['sigma'], seed['tip_end'], seed['perp_center'],
                         length, notch_q, notch_side)
        bolt['released_at'] = LID_BOLT_RELEASED_AT
        bolt['tip_engagement'] = LID_NOTCH_DEPTH - LID_TIP_END_PLAY
    else:
        a_t, s_t = target['axis_index'], target['notch_side']
        notch_centre = target['tip_end'] + target['sigma'] * target['notch_q']
        tip_end = target['perp_center'] + s_t * (HALF_W - TIP_ENGAGEMENT)
        bolt = make_bolt(name, 1 - a_t, s_t, tip_end, notch_centre, length, notch_q, notch_side)
        bolt['released_at'] = BOLT_RELEASED_AT
        bolt['tip_engagement'] = TIP_ENGAGEMENT
    return bolt


def add_cover(part):
    """A thin strip on top of the part hiding whose tip sits in its notch."""
    (x0, x1), (y0, y1) = part['notch']
    if part['kind'] == 'lid':
        if part['direction'] in ('x+', 'x-'):
            part['cover'] = ((x0 - COVER_MARGIN, x1 + COVER_MARGIN), (y0, y1 + COVER_MARGIN))
        else:
            edge = part['side'] * LID_HALF[0]
            bottom = part['side'] * (LID_HALF[0] - LID_NOTCH_DEPTH - COVER_MARGIN)
            part['cover'] = (tuple(sorted((edge, bottom))), (y0 - COVER_MARGIN, y1 + COVER_MARGIN))
    else:
        a = part['axis_index']
        # Never reach back over the engaged tip: that strip lies under the target's cover.
        clear = part['tip_end'] + part['sigma'] * (part.get('tip_engagement', TIP_ENGAGEMENT) + GAP + .002)
        along = [part['notch'][a][0] - COVER_MARGIN, part['notch'][a][1] + COVER_MARGIN]
        if part['sigma'] > 0:
            along[0] = max(along[0], clear)
        else:
            along[1] = min(along[1], clear)
        across = (part['perp_center'] - HALF_W, part['perp_center'] + HALF_W)
        part['cover'] = _rect(a, tuple(along), across)
    return part


# --- geometry queries -----------------------------------------------------

def rects_at(part, d):
    return [shift(r, part['axis'], d) for r in part['spans']]


def swept_rects(part, lo=None, hi=None):
    lo = part['range'][0] if lo is None else lo
    hi = part['range'][1] if hi is None else hi
    return [sweep(r, part['axis'], lo, hi) for r in part['spans']]


def handle_at(part, d):
    return (part['handle'][0] + part['axis'][0] * d, part['handle'][1] + part['axis'][1] * d)


def release_order(layout):
    """Bolts in the order they must be withdrawn, ending with the lid."""
    parts = layout['parts']
    chain, name = [], 'lid'
    while parts[name]['blocked_by'] is not None:
        name = parts[name]['blocked_by']
        chain.append(name)
    return list(reversed(chain)) + ['lid']


def blocked_now(values, layout):
    parts = layout['parts']
    result = {}
    for name, part in parts.items():
        blocker = part['blocked_by']
        result[name] = blocker is not None and values[blocker] < parts[blocker]['released_at']
    return result


# --- validation -----------------------------------------------------------

def _pair_conflicts(p, q, states_p, states_q, gap=None):
    gap = pair_gap(p, q) if gap is None else gap
    for dp in states_p:
        for dq in states_q:
            for r in rects_at(p, dp):
                for s in rects_at(q, dq):
                    if overlap(r, s, gap):
                        return f'{p["name"]}@{dp:.3f} overlaps {q["name"]}@{dq:.3f}'
    return None


def validate_layout(layout, strict=True):
    """Return a list of problems (empty when the layout is acceptable)."""
    parts = layout['parts']
    deck = (DECK_X, DECK_Y[layout['lid_direction']])
    problems = []
    names = list(parts)
    order = release_order(layout)
    chain = set(order)
    # 1. Rest positions never touch.
    for i, n in enumerate(names):
        for m in names[i + 1:]:
            issue = _pair_conflicts(parts[n], parts[m], (0.,), (0.,))
            if issue:
                problems.append('rest: ' + issue)
    # 2. Release sequence: each moving part sweeps through free space.
    state = {n: 0. for n in names}
    for mover in order:
        p = parts[mover]
        sw = swept_rects(p)
        for other in names:
            if other == mover:
                continue
            gap = pair_gap(p, parts[other])
            for s in rects_at(parts[other], state[other]):
                for r in sw:
                    if overlap(r, s, gap):
                        problems.append(f'sweep: {mover} sweeps into {other} at {state[other]:.3f}')
        state[mover] = p['range'][1]
    # 3. Decoys may be anywhere in their range at any time: their sweeps must
    #    be clear of every other part's sweep, and vice versa.
    for n in names:
        if not parts[n]['decoy']:
            continue
        for m in names:
            if m == n:
                continue
            for r in swept_rects(parts[n]):
                for s in swept_rects(parts[m]):
                    if overlap(r, s, GAP):
                        problems.append(f'decoy sweep: {n} and {m} sweeps overlap')
    # 4. Handles: the open gripper around any handle must not meet another handle,
    #    at any combination of the two parts' positions.
    grid = lambda p: [p['range'][0] + t * (p['range'][1] - p['range'][0]) / 4 for t in range(5)]
    for i, n in enumerate(names):
        for m in names[i + 1:]:
            close = any(abs(hn[0] - hm[0]) < HANDLE_SEP[0] and abs(hn[1] - hm[1]) < HANDLE_SEP[1]
                        for hn in (handle_at(parts[n], d) for d in grid(parts[n]))
                        for hm in (handle_at(parts[m], d) for d in grid(parts[m])))
            if close:
                problems.append(f'handles: {n} and {m} too close')
    # 5. Reach band and deck bounds.
    for n in names:
        p = parts[n]
        for d in (p['range'][0], p['range'][1]):
            h = handle_at(p, d)
            if not (REACH_X[0] <= h[0] <= REACH_X[1] and REACH_Y[0] <= h[1] <= REACH_Y[1]):
                problems.append(f'reach: {n} handle at {h} outside band')
        for r in swept_rects(p):
            if not inside(r, deck, OVERHANG):
                problems.append(f'deck: {n} sweep {r} leaves the deck')
        if p['cover'] is not None:
            bbox = ((min(r[0][0] for r in p['spans']), max(r[0][1] for r in p['spans'])),
                    (min(r[1][0] for r in p['spans']), max(r[1][1] for r in p['spans'])))
            if not inside(p['cover'], bbox, 1e-6):
                problems.append(f'cover: {n} cover leaves its part')
    # 6. Structure.
    sliders = layout.get('sliders', SLIDERS)
    if len(parts) != sliders + 1:
        problems.append(f'structure: {len(parts) - 1} sliders, expected {sliders}')
    if not 1 <= len(chain) - 1 <= 4:
        problems.append(f'structure: chain length {len(chain) - 1}')
    # 7. The cavity stays covered at rest; when open, the fingers' descent footprint
    #    is clear of the lid and every handle stays out of the hand's footprint.
    item = layout['item_xy']
    item_rect = ((item[0] - .03, item[0] + .03), (item[1] - .03, item[1] + .03))
    finger_rect = ((item[0] - .05, item[0] + .05), (item[1] - .02, item[1] + .02))
    lid = parts['lid']
    if not any(overlap(item_rect, r) for r in rects_at(lid, 0.)):
        problems.append('item: not covered by the lid at rest')
    if any(overlap(finger_rect, r, GAP) for r in rects_at(lid, lid['range'][1])):
        problems.append('item: fingers would meet the open lid')
    for n in names:
        states = (parts[n]['range'][1],) if n == 'lid' else (parts[n]['range'][0], parts[n]['range'][1])
        for d in states:
            h = handle_at(parts[n], d)
            if abs(h[0] - item[0]) < HAND_OVER_ITEM[0] and abs(h[1] - item[1]) < HAND_OVER_ITEM[1]:
                problems.append(f'item: handle of {n} under the hand while grasping the item')
                break
    return problems if strict else problems[:1]


# --- sampling -------------------------------------------------------------

def _candidate_notch_positions(length):
    lo, hi = NOTCH_LEN, length - HANDLE_INSET - HANDLE_RADIUS - .012
    return [round(lo + t * (hi - lo) / 2, 4) for t in range(3)]


def _try_place(part, placed, conservative=False):
    """Incremental check of one new part against the already placed ones.

    Chain bolts are checked at rest and for handle spacing; their sweeps are
    checked in release order by `_chain_ok`. Decoys may sit anywhere in their
    range at any time, so their sweeps must be clear of every other sweep.
    """
    for other in placed.values():
        if _pair_conflicts(part, other, (0.,), (0.,)):
            return False
        if conservative:
            for r in swept_rects(part):
                for s in swept_rects(other):
                    if overlap(r, s, GAP):
                        return False
        for dn in (part['range'][0], part['range'][1]):
            hn = handle_at(part, dn)
            for dm in (other['range'][0], other['range'][1]):
                hm = handle_at(other, dm)
                if abs(hn[0] - hm[0]) < HANDLE_SEP[0] and abs(hn[1] - hm[1]) < HANDLE_SEP[1]:
                    return False
    return True


def _chain_ok(parts, order):
    """Sweep check along the release order for the chain built so far."""
    state = {n: 0. for n in parts}
    for mover in order:
        if mover not in parts:
            continue
        for other in parts:
            if other == mover:
                continue
            gap = pair_gap(parts[mover], parts[other])
            for s in rects_at(parts[other], state[other]):
                for r in swept_rects(parts[mover]):
                    if overlap(r, s, gap):
                        return False
        state[mover] = parts[mover]['range'][1]
    return True


def sample_layout(seed, attempts=400, stats=None, tier='mixed', cover_probability=None, chain_lengths=CHAIN_LENGTHS):
    if tier not in TIERS:
        raise ValueError(f'Unknown tier {tier!r}; choose from {sorted(TIERS)}')
    cover_probability = TIERS[tier]['cover_probability'] if cover_probability is None else float(cover_probability)

    def note(reason):
        if stats is not None:
            stats[reason] = stats.get(reason, 0) + 1

    # Direction and chain length are fixed by the seed so that placement retries
    # do not bias instances toward the easiest structures.
    structure = random.Random(seed)
    direction = structure.choice(LID_DIRECTIONS)
    k = structure.choice(chain_lengths)
    for attempt in range(attempts):
        rng = random.Random(seed * 100003 + attempt)
        if attempt >= attempts // 2:
            # Fall back to a fresh structure if this one keeps failing placement.
            direction = rng.choice(LID_DIRECTIONS)
            k = rng.choice(chain_lengths)
        if direction == 'y+':
            lid = make_lid(direction, rng.choice((-.04, .02)), side=rng.choice((-1, 1)))
        else:
            lid = make_lid(direction, rng.choice((-.06, 0., .06)))
        parts = {'lid': lid}
        target, ok = lid, True
        for i in range(1, k + 1):
            length = rng.choice(FIRST_BOLT_LENGTHS if i == 1 else BOLT_LENGTHS)
            if i < k:
                notch_q = rng.choice(_candidate_notch_positions(length))
                side = rng.choice((-1, 1))
            else:
                notch_q, side = None, None
            bolt = make_blocker(f'bolt_{i}', target, length, notch_q, side)
            parts[target['name']]['blocked_by'] = bolt['name']   # `blocked_by` names a part's blocker
            if not _try_place(bolt, parts):
                ok = False
                break
            parts[bolt['name']] = bolt
            target = bolt
        if not ok:
            note('chain placement')
            continue
        order = release_order({'parts': parts})
        if not _chain_ok(parts, order):
            note('chain sweep')
            continue
        deck = (DECK_X, DECK_Y[direction])
        decoys_needed = SLIDERS - k
        placed_decoys = 0
        for j in range(decoys_needed):
            success = False
            for _ in range(80):
                axis_index = rng.choice((0, 1))
                sigma = rng.choice((-1, 1))
                length = rng.choice(DECOY_LENGTHS)
                cx = round(rng.uniform(deck[0][0] + .03, deck[0][1] - .03), 3)
                cy = round(rng.uniform(deck[1][0] + .03, deck[1][1] - .03), 3)
                tip_end = cx if axis_index == 0 else cy
                perp = cy if axis_index == 0 else cx
                empty_notch = rng.random() < .5
                notch_q = rng.choice(_candidate_notch_positions(length)) if empty_notch else None
                side = rng.choice((-1, 1)) if empty_notch else None
                decoy = make_bolt(f'decoy_{j + 1}', axis_index, sigma, tip_end, perp, length, notch_q, side, decoy=True)
                if not all(inside(r, deck, OVERHANG) for r in swept_rects(decoy)):
                    continue
                for d in decoy['range']:
                    h = handle_at(decoy, d)
                    if not (REACH_X[0] <= h[0] <= REACH_X[1] and REACH_Y[0] <= h[1] <= REACH_Y[1]):
                        break
                else:
                    if _try_place(decoy, parts, conservative=True):
                        parts[decoy['name']] = decoy
                        success = True
                        break
            if not success:
                break
            placed_decoys += 1
        if placed_decoys != decoys_needed:
            note('decoy placement')
            continue
        covered = {}
        for name in order[:-1] + ['lid']:
            target_name = name
            # a lock exists on every part that has a blocker
            if parts[target_name]['blocked_by'] is not None:
                covered[target_name] = rng.random() < cover_probability
                if covered[target_name]:
                    add_cover(parts[target_name])
        sigma_l = parts['lid']['axis']
        item_xy = ((-.04 * sigma_l[0], 0.) if direction != 'y+' else (0., -.02))
        palette = rng.choice(WOOD_PALETTES)
        layout = dict(seed=seed, attempt=attempt, tier=tier, cover_probability=cover_probability,
                      lid_direction=direction, chain_length=k, sliders=SLIDERS,
                      parts=parts, release_order=order, covered=covered, item_xy=item_xy,
                      deck=dict(x=DECK_X, y=DECK_Y[direction]),
                      item_asset=rng.choice(ITEM_CANDIDATES), tray_asset=rng.choice(TRAY_CANDIDATES),
                      colours=dict(wood=palette[0], deck=palette[1], lid=palette[2], metal=METAL,
                                   handle=rng.choice(HANDLE_COLOURS)))
        problems = validate_layout(layout)
        if not problems:
            layout['signature'] = layout_signature(layout)
            return layout
        note('validation: ' + problems[0].split(':')[0])
    raise RuntimeError(f'No valid layout for seed {seed} in {attempts} attempts')


def layout_signature(layout):
    """Coarse structure class used for development/evaluation splits."""
    start = layout['parts'][layout['release_order'][0]]
    hx, hy = start['handle']
    xclass = 'L' if hx < -.1 else ('R' if hx > .1 else 'C')
    yclass = 'front' if hy < -.15 else 'mid'
    covers = sum(1 for v in layout['covered'].values() if v)
    return f"{layout['lid_direction']}-k{layout['chain_length']}-start{xclass}{yclass}-cov{covers}"


def default_layout():
    """The hand-designed instance demonstrated first (native-01), in this representation."""
    lid = make_lid('x+', -.07)
    lid['handle'] = (.06, 0.)
    parts = {'lid': lid}
    a = make_blocker('bolt_1', lid, .136, notch_q=.071, notch_side=1)
    lid['blocked_by'] = 'bolt_1'
    b = make_blocker('bolt_2', a, .12, notch_q=.075, notch_side=-1)
    a['blocked_by'] = 'bolt_2'
    c = make_blocker('bolt_3', b, .128)
    b['blocked_by'] = 'bolt_3'
    d = make_bolt('decoy_1', 1, -1, -.09, -.19, .11, decoy=True)
    for p in (a, b, c, d):
        parts[p['name']] = p
    layout = dict(seed=None, attempt=0, tier='visible', cover_probability=0., lid_direction='x+', chain_length=3, sliders=4, parts=parts,
                  release_order=release_order({'parts': parts}), covered={}, item_xy=(-.04, 0.),
                  deck=dict(x=DECK_X, y=DECK_Y['x+']), item_asset='candle/candle_0', tray_asset='tray/tray_0',
                  colours=dict(wood=WOOD_PALETTES[0][0], deck=WOOD_PALETTES[0][1], lid=WOOD_PALETTES[0][2],
                               metal=METAL, handle=HANDLE_COLOURS[0]))
    layout['signature'] = layout_signature(layout)
    return layout
