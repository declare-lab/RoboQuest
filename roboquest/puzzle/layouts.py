"""Parameterised puzzle-box layouts (generator v1) for the RoboQuest suite.

The mechanism, the part naming (``lid``, ``bolt_i``, ``decoy_j``), the layout
dict format and every geometric primitive are the original ones in
:mod:`roboquest.puzzle_box_layouts` (the study-room sampler, generator v0 of
this task); this module only changes *who chooses the structure*. There the
seed drew the lid direction, the chain length and the number of decoys, and
placement retries were allowed to fall back to a different structure. Here the
experiment grid fixes

* ``chain_length`` in {4, 5, 6} -- bolts in the release chain, hence locks;
* ``decoys`` in {0, 1, 2} -- extra sliders that block nothing;
* ``cover`` in {none, partial, full} -- how many locks are hidden by a cover
  strip, so the blocking relation has to be discovered by trying.

The number of sliders is ``chain_length + decoys`` (at most
:data:`MAX_SLIDERS`). A retry re-draws positions, notch offsets, lengths and
nuisance choices only: the structure a call returns is always the structure its
arguments asked for. The lid direction stays a *seed* draw (it is structure,
not a factor), so the split class ``(direction, chain_length)`` mixes one
sampled and one designed coordinate.

``release_order``, ``blocked_now``, ``layout_signature`` and the layout dict
are unchanged, so :mod:`roboquest.puzzle_box_geometry`, the CPU interlock
gate and the oracle read layouts from either generator the same way.
``validate_layout`` is re-implemented only because the original hard-codes the
deck rectangle and a chain length of at most four; every rule inside it is the
original rule, plus a check that the structure is the one the factors asked
for. Layouts carry ``generator = 'roboquest-puzzle-v1'`` so the gate picks the
matching validator.

Re-lock decoy (v1 spec 4.3, generator ``v2`` of the task)
---------------------------------------------------------
When the caller asks for one (``relock=True``, drawn by the task from its
``structure`` stream in half of the instances that have a decoy), one decoy is
coupled to one chain bolt ``k`` through a hidden, one-way push rod inside the
chest (built by :mod:`roboquest.puzzle_box_geometry`): pulling the decoy
drives bolt ``k`` back into its notch, releasing bolt ``k`` again never moves
the decoy. The sampler only records a re-lock that the blocking model accepts
from **every chain state the robot can reach** -- the prefixes of the release
order (:func:`chain_states`): from each, the bolt's return sweep must be free
and the puzzle must still open afterwards (:func:`relock_state_checks`).

Measured consequence of the mechanism's own numbers: a withdrawn successor
bolt (travel ``BOLT_TRAVEL`` = 5 cm) always covers the 3 cm wide tip of the
bolt that used to hold it (its notch sits at least ``NOTCH_LEN`` = 4 cm from
its own tip), so a bolt whose successor is another bolt can never re-engage
once that successor has moved. The only bolt that can satisfy the rule is
the one that holds the lid, ``bolt_1``, and only when the open lid clears its
tip; the sampler therefore restricts the lid's notch slot to the clearing ones
when a re-lock is requested (:func:`relock_lid_slots`) and still runs the
general enumeration over the chain bolts in a seed-drawn preference order, so
a change of the mechanism's constants would widen the choice automatically.
The layout records ``relock = {decoy, bolt, k}`` (``None`` when no bolt
qualifies) and ``relock_requested``.
"""
from __future__ import annotations

import math
import random

from roboquest.puzzle_box_layouts import BOLT_TRAVEL, GAP, HANDLE_COLOURS, HANDLE_SEP, LID_HALF, HAND_OVER_ITEM, ITEM_CANDIDATES, LID_DIRECTIONS, METAL, OVERHANG, REACH_X, REACH_Y, TRAY_CANDIDATES, WOOD_PALETTES, HALF_W, HANDLE_INSET, HANDLE_RADIUS, LID_NOTCH_DEPTH, LID_TIP_END_PLAY, NOTCH_LEN, TIP_ENGAGEMENT, _chain_ok, _pair_conflicts, _try_place, blocked_now, handle_at, inside, layout_signature, make_blocker, make_bolt, make_lid, overlap, pair_gap, rects_at, release_order, swept_rects
from roboquest.puzzle import plate

GENERATOR = 'roboquest-puzzle-v1'
CHAIN_LENGTHS = (4, 5, 6)
DECOY_COUNTS = (0, 1, 2)
COVER_LEVELS = ('none', 'partial', 'full')
MAX_SLIDERS = 8
# One attempt is now a bounded backtracking search (:func:`_place_chain`), 30-90 ms for a
# chain of six, not the v1 sampler's single 0.15 ms draw, so the budget is counted in
# hundreds rather than tens of thousands: it buys STRUCTURE_REDRAW attempts on each of
# about two dozen structures, and the cells that need a particular lid direction find one
# with probability 1 - (2/3)**24.
ATTEMPTS = 600
# Attempts per structure draw (:func:`_structure_draw`). A structure the plate cannot
# accommodate is not unlucky, it is unbuildable, so the sampler moves on to another one.
STRUCTURE_REDRAW = 25
# Nodes the chain placer may expand per attempt before giving up and letting the caller
# draw again (see :func:`_place_chain`). Measured over the hard cells, 3000 is where the
# restart yield stops improving: a chain of five with every lock covered goes from 11/20
# at 600 nodes to 19/20 at 3000, and 8000 buys nothing for three times the time.
CHAIN_NODES = 3000
# Grid the placer draws notch offsets from (:func:`notch_grid`), and how many of the
# resulting options it may try per bolt. Trying them all turns the search depth-first into
# a few thousand nodes spent on bolt_1 and bolt_2 of a chain of six; drawing a handful per
# bolt and restarting the whole attempt keeps it shallow and diverse.
NOTCH_STEP = .004
MAX_BRANCH = 8
# Notch offsets along a bolt. The v0 sampler used three per bolt; a chain of six has
# to thread a much tighter space, and five offsets raise the yield by about 1.6x at
# every chain length while widening the design space. Bolt lengths are unchanged: shorter
# bolts fold the chain onto itself and do far worse (measured).
NOTCH_POSITIONS = 5
# Slider lengths. The plate keeps `plate.HANDLE_KEEP` = 53 mm clear of every handle, so an
# engagement it hides sits at least HANDLE_INSET + HANDLE_KEEP + HALF_W + HIDE_MARGIN =
# 9.8 cm back from the slider's outer end, while the smallest notch offset from the tip is
# NOTCH_LEN = 4 cm: a slider whose lock can be covered at all is at least 14 cm long (see
# :func:`max_notch_q`). Covered locks therefore draw from their own pool. The uncovered
# pool is the v0 range on a 1 cm grid and reaches down to 10 cm, which is what makes a long
# chain fit at all: the short sliders pack around the long ones, and measured on a chain of
# six with three covered locks the yield goes from 0 to a third of attempts when the
# uncovered pool is (.10 ... .14) rather than (.12, .14, .16). The pools overlap and the
# decoys draw from the union, so a slider's length never says whether its lock is covered
# or whether it is a decoy at all.
FIRST_BOLT_LENGTHS = (.10, .11, .12, .13, .14)
BOLT_LENGTHS = (.10, .11, .12, .13, .14)
COVERED_FIRST_BOLT_LENGTHS = (.14, .15, .16, .17, .18, .19, .20)
COVERED_BOLT_LENGTHS = (.14, .15, .16, .17, .18, .19, .20)
DECOY_LENGTHS = (.10, .12, .14, .16, .18, .20)
# A covered lock costs 14 cm of slider, so a chain cannot have many of them: measured with
# a beam search over the whole option set (see `MAX_COVERED_LOCKS`), four covered bolt
# locks are the most that fit inside the reach band, and the fifth never does.
LID_SLOT_MARGIN = .01               # solid lid kept outside either end of its notch
MIN_PARTIAL_LOCKS = 3               # see :func:`covered_count`: fewer cannot pass the G0 gate
MAX_COVERED_LOCKS = 4
PARTIAL_LOCKS = {4: 3, 5: 3, 6: 4}  # see :func:`covered_count`: a chain of six cannot be placed with three
# Deck (chest top) rectangle in the box frame, enlarged from the v0 sampler's
# (-.30, .30) x (-.30, .10) so the reach band, not the deck, bounds the sliders: a
# handle at the edge of REACH needs HANDLE_INSET more deck behind it, and the deck
# tolerates OVERHANG of that. Cavity, chest walls and REACH_X/REACH_Y are unchanged
# and `puzzle_box_geometry` sizes the chest from the layout's `deck`.
DECK_X = (-.32, .32)
DECK_Y = {'x+': (-.30, .11), 'x-': (-.30, .11), 'y+': (-.30, .26)}
# A re-locked bolt counts as back in its locked position when its joint is within this of zero: the push rod
# stops ROD_CLEARANCE (7 mm) short of the joint limit by design, and the tip is then still 4 mm inside the
# notch (TIP_ENGAGEMENT 13 mm), far below BOLT_RELEASED_AT (20 mm); the physics gate proves the lid stays
# blocked after a re-lock.
RELOCK_LOCKED_TOL = .009

__all__ = ['GENERATOR', 'CHAIN_LENGTHS', 'DECOY_COUNTS', 'COVER_LEVELS', 'MAX_SLIDERS', 'ATTEMPTS',
           'DECK_X', 'DECK_Y', 'RELOCK_LOCKED_TOL', 'blocked_now', 'chain_states', 'covered_count', 'deck_for',
           'layout_signature', 'lock_names', 'release_order', 'relock_feasible', 'relock_lid_slots',
           'relock_state_checks', 'sample_layout', 'validate_layout']


def notch_positions(length, count=None):
    """Candidate notch offsets from a bolt's tip end (the v0 rule on a finer grid)."""
    count = NOTCH_POSITIONS if count is None else count
    # floor to the 0.1 mm grid: `make_bolt` rejects an offset a float ulp past its bound,
    # and with some lengths the top offset lands exactly there.
    lo = NOTCH_LEN
    hi = math.floor((length - HANDLE_INSET - HANDLE_RADIUS - .012) * 1e4) / 1e4
    if hi <= lo:
        return [lo]
    return [round(lo + t * (hi - lo) / (count - 1), 4) for t in range(count)]


def bolt_lengths(index, covered):
    """Length pool for chain bolt `index`: only a bolt whose own notch the plate has to
    hide needs the extra length (:func:`max_notch_q`), and the pools overlap so that a
    slider's length never says on its own whether its lock is covered. Decoys draw from
    the union, so a long slider is as likely to be a decoy as a covered lock."""
    if covered:
        return COVERED_FIRST_BOLT_LENGTHS if index == 1 else COVERED_BOLT_LENGTHS
    return FIRST_BOLT_LENGTHS if index == 1 else BOLT_LENGTHS


def max_notch_q(length):
    """Largest notch offset from the tip whose engagement the plate can still hide.

    A per-notch strip would need a covered notch *far enough from the tip*,
    because it could not reach back over the part's own engaged tip. The plate
    needs the opposite: the notch far
    enough from the slider's own **handle**, because the plate stops
    ``plate.HANDLE_KEEP`` short of every handle and still has to reach
    ``plate.HIDE_MARGIN`` past the holder's bar inside the notch. The handle
    sits ``HANDLE_INSET`` in from the outer end, so the offset from the tip is
    bounded by ``length - HANDLE_INSET - HANDLE_KEEP - (HALF_W + HIDE_MARGIN)``.
    """
    return length - HANDLE_INSET - plate.HANDLE_KEEP - (HALF_W + plate.HIDE_MARGIN)


ITEM_X_OFFSET = 0.045                # x lids: the item stands this far from the chest's centre, away
                                    # from the lid's opening side (v1/v2: 0.04)
ITEM_Y_PLUS = (0., -0.035)            # y+ lids: the item's standing position (v1/v2: (0, -0.02))
LID_TRAVEL_V3 = {'x+': .20, 'x-': .20, 'y+': .14}   # x lids open 3 cm farther than v0/v1 (0.17); see item_xy_for


def item_xy_for(direction, lid):
    """Where the item sits in the cavity: off-centre, away from the lid's opening side.

    The distances are the gripper's, measured in job PLATE-5, and they come with
    ``LID_TRAVEL_V3``. The pick closes its fingers along the chest's x (the long side of the
    plate's slot, `plate.ITEM_CLEAR`), so along an x lid's travel the hand needs the outer
    fingertip (52 mm from the grasp site) inside the cavity wall (|x| = 0.11) and the far end
    of its 211 mm body (underside 22 mm above the site) clear of the fully open lid's near
    edge, whose top (0.088) is above that underside. With the v0 travel of 0.17 the open edge
    was 5 cm from the centre and the two needs left a 2 mm window: at 4 cm the hand rested on
    the lid's edge (`gripper0_right_hand_collision <-> puzzle_lid_seg0`, finger bodies instead
    of pads on the item, 4 of 6 trial-a picks) and at 6 cm the fingertip landed on the chest
    top outside the cavity (every trial-b x-lid pick stopped 47-49 mm short, tracking the
    grasp-height jitter exactly). x lids therefore open 0.20 (edge at 8 cm) and the item
    stands 4.5 cm from the centre: 13 mm inside the wall for the fingertip, 19 mm clear of the
    open lid for the hand. For a y+ lid the item stands lower so that the plate can still hide
    the lid's own notch: the slot's 106 mm half-length along x reaches the rim notch
    (x >= 0.086), so the notch has to clear the slot's y band instead, which it does at the
    rim slot y = 0.04 once the item is at y = -0.035 (its far side 3 mm inside the cavity wall
    for the widest item); the hand's short side lies along y there and clears the open lid.
    """
    if direction == 'y+':
        return ITEM_Y_PLUS
    return (-ITEM_X_OFFSET * lid['axis'][0], 0.)


def deck_for(direction):
    """The deck rectangle for a lid direction, in the layout's own dict form."""
    return dict(x=DECK_X, y=DECK_Y[direction])


def deck_for_rect(direction):
    """The same rectangle as a pair of ranges."""
    return (DECK_X, DECK_Y[direction])


def lock_names(chain_length):
    """Parts that carry a lock, i.e. that something else blocks, nearest the lid first."""
    return ['lid'] + [f'bolt_{i}' for i in range(1, chain_length)]


def covered_count(chain_length, cover):
    """How many of the ``chain_length`` locks the cover level hides.

    Both levels hide a prefix of the lock order, which starts at the lid, so what ``partial``
    leaves in the open is the deep end of the chain, where the first sliders the robot has to
    pull are. ``full`` means every lock, capped at ``MAX_COVERED_LOCKS`` = 4; ``partial`` is one
    engagement fewer where the geometry allows it, and the two are the same on a chain of six:
    ``PARTIAL_LOCKS`` = {4: 3, 5: 3, 6: 4}. Every number here is a measurement (job PLATE-5,
    2026-09-22, `/tmp/plate5/*.json` and the progress note), not a preference.

    * The floor ``MIN_PARTIAL_LOCKS`` = 3 is G0 arithmetic: the covered locks being a prefix,
      hiding ``c`` of them puts exactly ``c`` sliders under the plate and the deepest covered
      lock is itself one of them, leaving it ``c - 1`` candidate holders; two candidates need
      ``c >= 3`` in any instance the decoys do not happen to help.
    * The cap of 4 is the gripper's. The plate has to leave the hand's measured footprint free
      over the item (`plate.ITEM_CLEAR`, 106 x 45 mm half extents, long along the chest's x),
      and with that slot in the deck no chain of six hides five locks: 0 of 12 seeds in every
      one of five item positions, all 7200 attempts failing in chain placement, against
      12 of 12 with the 70 x 40 mm slot the first v3 draft had; a chain of five hides five only
      with a y+ lid and only when the item stands lower in the cavity (5 of 12), which would
      confound the cover level with the held-out lid direction. Chains of four and five hide
      all of theirs, a chain of six the four nearest the lid.
    * The chain-6 amendment: a chain of six with three covered locks is unplaceable with this
      slot -- 0 of 12 seeds at 88 s each, with the committed item position, five other item
      positions and three longer length pools for the visible bolts (the placer trace shows the
      three short trailing bolts curling back over the item; every such chain the old slot
      built used a y+ lid, and the slot's 106 mm half-length along x removed the y+ lid's rim
      notch from reach). With four covered locks the same cell is 12 of 12, so on a chain of
      six ``partial`` and ``full`` coincide at four. The rule (partial one
      fewer than full everywhere) therefore holds for chains of four and five only.
    """
    if cover not in COVER_LEVELS:
        raise ValueError(f'unknown cover level {cover!r}; choose from {COVER_LEVELS}')
    if cover == 'none':
        return 0
    full = min(chain_length, MAX_COVERED_LOCKS)
    if cover == 'full':
        return full
    return min(full, max(MIN_PARTIAL_LOCKS, PARTIAL_LOCKS.get(chain_length, full - 1)))


# --- re-lock: the blocking model over every reachable chain state ------------------------

def chain_states(layout):
    """Every chain state the robot can reach: the prefixes of the release order.

    State ``j`` has ``order[:j]`` released at their full travel and everything
    else at rest. Decoys may be anywhere in their range at any time, but their
    sweeps are clear of every other sweep by construction (rule 3), so they
    never enter the blocking model.
    """
    order = layout['release_order']
    return [list(order[:j]) for j in range(len(order) + 1)]


def _state_positions(layout, released):
    parts = layout['parts']
    return {name: (parts[name]['range'][1] if name in released else 0.) for name in parts}


def _sweep_clear(layout, mover, positions):
    """True when ``mover``'s full-travel sweep is free of every chain part where ``positions`` puts it."""
    parts = layout['parts']
    p = parts[mover]
    for other, q in parts.items():
        if other == mover or q['decoy']:
            continue
        gap = pair_gap(p, q)
        for s in rects_at(q, positions[other]):
            for r in swept_rects(p):
                if overlap(r, s, gap):
                    return False
    return True


def relock_state_checks(layout, bolt):
    """The blocking model's verdict on re-locking ``bolt`` from every reachable chain state.

    One row per state: whether the bolt is released there (else there is
    nothing to re-lock and the pull is harmless), whether its return sweep --
    the release sweep in reverse -- is free of every chain part where the
    state puts it (``return_clear``: this is what a withdrawn successor
    defeats), and whether the puzzle still opens afterwards (``solvable``: the
    bolt releases again and every lock that was not yet open still opens in
    order).
    """
    parts = layout['parts']
    order = layout['release_order']
    rows = []
    for released in chain_states(layout):
        positions = _state_positions(layout, released)
        row = dict(released=list(released), bolt_released=bolt in released)
        if bolt not in released:
            row.update(return_clear=True, solvable=True, note='bolt still locked: the pull re-locks nothing')
            rows.append(row)
            continue
        row['return_clear'] = _sweep_clear(layout, bolt, positions)
        after = dict(positions)
        after[bolt] = 0.
        done = set(released) - {bolt}
        solvable = True
        for name in order:
            if name in done:
                continue                       # already released and untouched by the re-lock
            if not _sweep_clear(layout, name, after):
                solvable = False
                break
            after[name] = parts[name]['range'][1]
            done.add(name)
        row['solvable'] = solvable
        rows.append(row)
    return rows


def relock_feasible(layout, bolt):
    return all(r['return_clear'] and r['solvable'] for r in relock_state_checks(layout, bolt))


def lid_slots(direction):
    """Where the lid's notch may sit along its edge (v1 offered three slots for x lids and
    two for y+). More slots matter when the lid's own lock is covered: the plate has to
    reach over the notch while staying `HANDLE_KEEP` from the lid's handle, which only the
    slots away from the handle allow. Every slot keeps at least ``LID_SLOT_MARGIN`` of
    solid lid outside either end of the notch -- a notch flush with the lid's edge is an
    open corner, and the bolt tip would leave it sideways instead of holding the lid.
    """
    half = LID_HALF[0] if direction in ('x+', 'x-') else LID_HALF[1]
    step = .03 if direction in ('x+', 'x-') else .02
    reach = int((half - NOTCH_LEN / 2 - LID_SLOT_MARGIN) / step + 1e-9)
    return tuple(round(step * k, 4) for k in range(-reach, reach + 1))


def lid_sides(direction):
    return (-1, 1) if direction == 'y+' else (None,)


def _lid_probe(direction, slot, side):
    """A lid at this slot with a first bolt of the shortest length in its notch, as a two-part layout."""
    lid = make_lid(direction, slot, side=side, travel=LID_TRAVEL_V3[direction])
    bolt = make_blocker('bolt_1', lid, FIRST_BOLT_LENGTHS[0])
    lid['blocked_by'] = bolt['name']
    return dict(parts={'lid': lid, 'bolt_1': bolt}, release_order=['bolt_1', 'lid'])


_RELOCK_SLOTS = {}


def relock_lid_slots(direction):
    """The lid notch slots at which ``bolt_1`` can re-engage with the lid fully open, for any lid side.

    Computed from the same sweep model as :func:`relock_state_checks` (a probe
    lid with a first bolt), never tabulated by hand. With the v3 numbers (x lids opening
    ``LID_TRAVEL_V3`` = 0.20): ``x+`` every slot but 0.09, ``x-`` every slot but -0.09,
    ``y+`` every slot (with the v0 travel of 0.17 it was ``x+`` (-0.06, 0), ``x-`` (0, 0.06)).
    """
    if direction not in _RELOCK_SLOTS:
        keep = [slot for slot in lid_slots(direction)
                if all(relock_feasible(_lid_probe(direction, slot, side), 'bolt_1')
                       for side in lid_sides(direction))]
        _RELOCK_SLOTS[direction] = tuple(keep)
    return _RELOCK_SLOTS[direction]


# --- validation -----------------------------------------------------------

def validate_layout(layout, strict=True):
    """Return a list of problems (empty when the layout is acceptable).

    Identical in substance to :func:`roboquest.puzzle_box_layouts.validate_layout`;
    the deck comes from the layout instead of module constants and the structure
    check uses the layout's own slider, chain and cover counts.
    """
    parts = layout['parts']
    deck = (tuple(layout['deck']['x']), tuple(layout['deck']['y']))
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
    # 3. Decoys may be anywhere in their range at any time.
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
    # 4. Handles: the open gripper around any handle must not meet another handle.
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
            problems.append(f'cover: {n} carries a per-notch cover strip; v3 hides locks '
                            f'under the chest plate instead')
    # 6. Structure: the sliders, chain and covers the factors asked for, and nothing else.
    sliders = layout['sliders']
    chain_length = layout['chain_length']
    if len(parts) != sliders + 1:
        problems.append(f'structure: {len(parts) - 1} sliders, expected {sliders}')
    if len(chain) - 1 != chain_length:
        problems.append(f'structure: chain length {len(chain) - 1}, expected {chain_length}')
    if sliders != chain_length + layout['decoys']:
        problems.append(f"structure: {sliders} sliders is not {chain_length} + {layout['decoys']}")
    if sliders > MAX_SLIDERS:
        problems.append(f'structure: {sliders} sliders exceeds {MAX_SLIDERS}')
    level = layout['cover']
    expected_covers = covered_count(chain_length, level)
    covered = layout['covered']
    if sorted(covered) != sorted(lock_names(chain_length)):
        problems.append(f'structure: covers keyed on {sorted(covered)}, not the locks')
    elif sum(bool(v) for v in covered.values()) != expected_covers:
        problems.append(f"structure: {sum(bool(v) for v in covered.values())} covers, "
                        f"expected {expected_covers} for cover={layout['cover']!r}")
    has_plate = bool(layout.get('plate'))
    if bool(expected_covers) != has_plate:
        problems.append(f'structure: cover level {level!r} wants {expected_covers} covered lock(s) '
                        f'but has_plate is {has_plate}')
    # 7. Cavity covered at rest, fingers clear when open, no handle under the hand.
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
    # 8. Re-lock decoy: a real decoy pulling against a chain bolt's withdrawal, and a re-lock the
    #    blocking model accepts from every reachable chain state.
    relock = layout.get('relock')
    if relock is not None:
        decoy_name, bolt_name = relock.get('decoy'), relock.get('bolt')
        if decoy_name not in parts or not parts[decoy_name]['decoy']:
            problems.append(f'relock: {decoy_name!r} is not a decoy of this layout')
        elif bolt_name not in parts or bolt_name not in chain or parts[bolt_name]['kind'] != 'bolt':
            problems.append(f'relock: {bolt_name!r} is not a chain bolt of this layout')
        else:
            d, b = parts[decoy_name], parts[bolt_name]
            if d['axis_index'] != b['axis_index'] or d['sigma'] != -b['sigma']:
                problems.append(f'relock: {decoy_name} does not pull against the withdrawal of {bolt_name}')
            if relock.get('k') != int(bolt_name.split('_')[1]):
                problems.append(f"relock: k={relock.get('k')!r} does not name {bolt_name}")
            for row in relock_state_checks(layout, bolt_name):
                if not (row['return_clear'] and row['solvable']):
                    what = 'cannot re-engage' if not row['return_clear'] else 'leaves the puzzle unsolvable'
                    problems.append(f"relock: {bolt_name} {what} once {row['released']} are released")
                    break
    # 9. The plate: it hides every covered engagement, clears every handle over its travel, the
    #    hand's column over the item and the engagements this instance leaves visible, stands on
    #    legs outside every slider sweep, and leaves at least two candidate holders for every
    #    covered lock (G0 hardness).
    problems.extend(plate.plate_problems(layout))
    return problems if strict else problems[:1]


# --- sampling -------------------------------------------------------------

def _structure_draw(seed, chain_length, cover, decoys=0, relock=False):
    """Structure draw: lid direction, which locks are covered and, when a re-lock is
    requested, the preference order over the chain bolts and which decoy carries the rod.

    Drawn before the placement loop so that ordinary retries never move them; after
    ``STRUCTURE_REDRAW`` fruitless attempts :func:`sample_layout` draws a new structure
    from a derived seed, because with the plate a structure can be unbuildable rather than
    unlucky -- four covered locks need a y+ lid, since an x lid pins the whole chain into
    the 22 cm strip between its near rim and the edge of the reach band. The first draw is
    the same the v0 sampler made, so an instance that never needs a redraw keeps its lid
    direction across generator versions; the re-lock draws come last and only when asked
    for, so a seed without a re-lock keeps its v1 layout.
    """
    structure = random.Random(seed)
    direction = structure.choice(LID_DIRECTIONS)
    locks = lock_names(chain_length)
    n = covered_count(chain_length, cover)
    # Both cover levels hide the locks nearest the lid, so `partial` is a prefix of `full`
    # and one plate covers one connected region: a covered engagement scattered among
    # visible ones leaves the plate no way to reach it without hiding a lock the level says
    # is shown (`plate.SHOW_MARGIN`), and chains for such a draw were rejected far more
    # often than they were placed. It also puts the deep end of the chain -- where the
    # short sliders have to go, and the one lock a chain of six leaves visible at `full`
    # (:func:`covered_count`) -- on the side the plate does not reach.
    covered = set(locks[:n])
    if not relock:
        return direction, covered, None, None
    preference = [f'bolt_{i}' for i in range(1, chain_length + 1)]
    structure.shuffle(preference)
    decoy_index = structure.randrange(decoys) + 1
    return direction, covered, preference, decoy_index


def reach_ok(part):
    """True when the part's handle stays in the robot's reach band over its whole travel."""
    for d in part['range']:
        h = handle_at(part, d)
        if not (REACH_X[0] <= h[0] <= REACH_X[1] and REACH_Y[0] <= h[1] <= REACH_Y[1]):
            return False
    return True


def notch_grid(length, covered, step=NOTCH_STEP):
    """Every notch offset the placer may use for a bolt of this length, on a `step` grid.

    The v1 sampler drew one of five offsets per bolt. A covered lock's usable range is
    much shorter (:func:`max_notch_q` caps it at the handle end) and a v3 chain has more
    to satisfy, so the placer searches a 4 mm grid instead: same interval, denser, which
    is what makes a chain of six with three covered locks findable at all.
    """
    hi = math.floor((max_notch_q(length) if covered else
                     length - HANDLE_INSET - HANDLE_RADIUS - .012) * 1e4) / 1e4
    if hi < NOTCH_LEN:
        return []
    return [round(NOTCH_LEN + k * step, 4) for k in range(int((hi - NOTCH_LEN) / step) + 1)]


def _blocker_options(rng, index, last, covered):
    """(length, notch_q, notch_side) draws for chain bolt `index`, in random order.

    The v1 sampler drew one of these per attempt and threw the whole layout away when it
    did not fit. A v3 chain is longer (a covered lock needs a long slider) and has more to
    satisfy -- reach band, no clash, handles off the engagements the plate must hide -- so
    the placer walks its own small option set in seed-driven random order and keeps the
    first option that fits. Same draws, searched instead of guessed; measured on chain 6
    it is the difference between one layout in 10^5 attempts and one in 30.
    """
    options = []
    for length in bolt_lengths(index, covered):
        if last:
            options.append((length, None, None))
            continue
        for notch_q in notch_grid(length, covered):
            options.extend([(length, notch_q, -1), (length, notch_q, 1)])
    rng.shuffle(options)
    return options[:MAX_BRANCH]


def hand_ok(part, item):
    """True when the part's handle stays out of the hand's footprint over the item.

    The same rule as `validate_layout` check 7: a handle the hand would meet while it
    reaches into the open chest is a placement failure, so the placer backtracks on it
    instead of building a whole layout that check 7 then throws away.
    """
    states = (part['range'][1],) if part['kind'] == 'lid' else part['range']
    for d in states:
        h = handle_at(part, d)
        if abs(h[0] - item[0]) < HAND_OVER_ITEM[0] and abs(h[1] - item[1]) < HAND_OVER_ITEM[1]:
            return False
    return True


def lid_ok(lid, item):
    """`validate_layout` check 7 for the lid alone: it covers the item at rest, its open
    position is clear of the fingers, and its handle is not over the item when open."""
    item_rect = ((item[0] - .03, item[0] + .03), (item[1] - .03, item[1] + .03))
    finger_rect = ((item[0] - .05, item[0] + .05), (item[1] - .02, item[1] + .02))
    if not any(overlap(item_rect, r) for r in rects_at(lid, 0.)):
        return False
    if any(overlap(finger_rect, r, GAP) for r in rects_at(lid, lid['range'][1])):
        return False
    return hand_ok(lid, item)


def _extend_chain(rng, parts, seeds, shows, target, index, chain_length, covered_names, item,
                  deck, budget):
    """Depth-first search for the rest of the chain from `target`; True when it is placed.

    ``seeds`` are the engagements the plate will have to cover and ``shows`` the ones it
    must keep off, both grown as the chain grows: a covered engagement that lands next to
    a visible one leaves the plate no way to reach the first without hiding the second, so
    the placer backtracks there rather than building a layout `build_plate` then rejects.
    ``budget`` is a one-element list of remaining nodes, so a hopeless subtree costs a
    bounded amount of work and the caller simply draws again.
    """
    if index > chain_length:
        return True
    name = f'bolt_{index}'
    covered = index < chain_length and name in covered_names
    target['blocked_by'] = name     # `blocked_by` names a part's blocker, and a locked pair
    for length, notch_q, side in _blocker_options(  # keeps only its design play (`pair_gap`)
            rng, index, index == chain_length, covered):
        if budget[0] <= 0:
            break
        budget[0] -= 1
        bolt = make_blocker(name, target, length, notch_q, side)
        if not reach_ok(bolt) or not hand_ok(bolt, item):
            continue
        if not _try_place(bolt, parts) or not plate.handle_path_clear(bolt, seeds):
            continue
        seed = show = None
        whole = dict(parts, **{name: bolt})
        if covered:
            seed = plate.hideable(whole, name, item, deck, avoid=shows)
            if seed is None:
                continue
            seeds.append(seed)
        elif index < chain_length:
            show = plate.show_rect(whole, name)
            if any(overlap(show, rect) for rect in seeds):
                continue
            shows.append(show)
        parts[name] = bolt
        if _extend_chain(rng, parts, seeds, shows, bolt, index + 1, chain_length, covered_names,
                         item, deck, budget):
            return True
        del parts[name]
        if seed is not None:
            seeds.pop()
        if show is not None:
            shows.pop()
    target['blocked_by'] = None
    return False


def _place_chain(rng, direction, chain_length, covered_names=(), relock=False, budget=CHAIN_NODES):
    """The lid plus its chain of blocking bolts, or None when placement failed.

    ``covered_names`` are the locks this instance will cover. Two rules apply to them,
    both about the plate that will have to hide them: a covered bolt's notch sits far
    enough from its own *handle* for the plate to reach over the bar inside it
    (:func:`max_notch_q`), and every covered engagement stays off the hand's column over
    the item and off every handle's path, checked as the chain grows so that a hopeless
    draw dies here instead of after a whole layout is built. Uncovered locks keep the full
    range of offsets. With ``relock`` the lid's notch slot is drawn from
    :func:`relock_lid_slots` only. Returns ``(parts, seeds)``: the parts and the
    rectangles the plate will have to cover.

    The v1 sampler drew one length, notch offset and side per bolt and threw the whole
    layout away when they did not fit. A v3 chain is longer (a covered lock needs a long
    slider) and has more to satisfy, and one bad offset early poisons every later bolt, so
    the placer walks its own small option set in seed-driven random order and backtracks
    within a node budget. Same draws, searched instead of guessed.
    """
    slots = relock_lid_slots(direction) if relock else lid_slots(direction)
    lids = [make_lid(direction, slot, side=side, travel=LID_TRAVEL_V3[direction])
            for slot in slots for side in lid_sides(direction)]
    rng.shuffle(lids)
    nodes = [budget]
    for lid in lids:
        item = item_xy_for(direction, lid)
        if not lid_ok(lid, item):
            continue
        seeds, shows = [], []
        if 'lid' in covered_names:
            seed = plate.hideable({'lid': lid}, 'lid', item, deck_for_rect(direction))
            if seed is None:
                continue
            seeds.append(seed)
        else:
            shows.append(plate.show_rect({'lid': lid}, 'lid'))
        parts = {'lid': lid}
        if _extend_chain(rng, parts, seeds, shows, lid, 1, chain_length, covered_names, item,
                         deck_for_rect(direction), nodes):
            return parts, seeds
        if nodes[0] <= 0:
            break
    return None


def _place_decoys(rng, parts, deck, decoys, item, tries=120, forced=None, seeds=()):
    """Add `decoys` free sliders; True when every one of them found a spot.

    ``forced`` maps a decoy index (1-based) to the ``(axis_index, sigma)`` it
    must have: the re-lock decoy pulls against the re-locked bolt's withdrawal.
    ``seeds`` are the engagements the plate will have to hide: a decoy whose
    handle would travel over one of them is re-drawn, since the plate has to
    keep clear of every handle.
    """
    forced = forced or {}
    for j in range(decoys):
        for _ in range(tries):
            if j + 1 in forced:
                axis_index, sigma = forced[j + 1]
            else:
                axis_index = rng.choice((0, 1))
                sigma = rng.choice((-1, 1))
            length = rng.choice(DECOY_LENGTHS)
            cx = round(rng.uniform(deck[0][0] + .03, deck[0][1] - .03), 3)
            cy = round(rng.uniform(deck[1][0] + .03, deck[1][1] - .03), 3)
            tip_end = cx if axis_index == 0 else cy
            perp = cy if axis_index == 0 else cx
            empty_notch = rng.random() < .5
            notch_q = rng.choice(notch_positions(length)) if empty_notch else None
            side = rng.choice((-1, 1)) if empty_notch else None
            decoy = make_bolt(f'decoy_{j + 1}', axis_index, sigma, tip_end, perp, length, notch_q, side,
                              decoy=True)
            if not all(inside(r, deck, OVERHANG) for r in swept_rects(decoy)):
                continue
            if not hand_ok(decoy, item):
                continue
            for d in decoy['range']:
                h = handle_at(decoy, d)
                if not (REACH_X[0] <= h[0] <= REACH_X[1] and REACH_Y[0] <= h[1] <= REACH_Y[1]):
                    break
            else:
                if _try_place(decoy, parts, conservative=True) and plate.handle_path_clear(decoy, seeds):
                    parts[decoy['name']] = decoy
                    break
        else:
            return False
    return True


def sample_layout(seed, chain_length, decoys, cover, attempts=ATTEMPTS, stats=None, relock=False):
    """A valid layout with exactly this structure, or ``RuntimeError``.

    ``chain_length`` bolts lock the lid in a chain, ``decoys`` further sliders
    block nothing, and the ``cover`` level decides how many of the
    ``chain_length`` locks are hidden. Retries re-draw positions only. With
    ``relock`` (needs ``decoys >= 1``) one decoy re-locks a chain bolt: the
    bolt is the first of a seed-drawn preference order that
    :func:`relock_feasible` accepts, the decoy is seed-drawn and placed
    pulling against that bolt's withdrawal; ``layout['relock']`` is ``None``
    when no chain bolt qualifies.
    """
    if chain_length not in CHAIN_LENGTHS:
        raise ValueError(f'chain_length {chain_length!r} not in {CHAIN_LENGTHS}')
    if decoys not in DECOY_COUNTS:
        raise ValueError(f'decoys {decoys!r} not in {DECOY_COUNTS}')
    if cover not in COVER_LEVELS:
        raise ValueError(f'cover {cover!r} not in {COVER_LEVELS}')
    sliders = chain_length + decoys
    if sliders > MAX_SLIDERS:
        raise ValueError(f'{sliders} sliders exceeds {MAX_SLIDERS}')

    def note(reason):
        if stats is not None:
            stats[reason] = stats.get(reason, 0) + 1

    relock = bool(relock)
    if relock and decoys < 1:
        raise ValueError('a re-lock decoy needs at least one decoy')
    for attempt in range(attempts):
        if attempt % STRUCTURE_REDRAW == 0:
            direction, covered_names, preference, decoy_index = _structure_draw(
                seed if attempt == 0 else seed * 7919 + attempt, chain_length, cover, decoys, relock)
            deck = (DECK_X, DECK_Y[direction])
        rng = random.Random(seed * 100003 + attempt)
        placed = _place_chain(rng, direction, chain_length, covered_names, relock=relock)
        if placed is None:
            note('chain placement')
            continue
        parts, seeds = placed
        item_xy = item_xy_for(direction, parts['lid'])
        order = release_order({'parts': parts})
        if not _chain_ok(parts, order):
            note('chain sweep')
            continue
        relock_spec, forced = None, None
        if relock:
            probe = dict(parts=parts, release_order=order)
            bolt = next((b for b in preference if relock_feasible(probe, b)), None)
            if bolt is not None:
                forced = {decoy_index: (parts[bolt]['axis_index'], -parts[bolt]['sigma'])}
                relock_spec = dict(decoy=f'decoy_{decoy_index}', bolt=bolt, k=int(bolt.split('_')[1]))
        if not _place_decoys(rng, parts, deck, decoys, item_xy, forced=forced, seeds=seeds):
            note('decoy placement')
            continue
        covered = {name: name in covered_names for name in lock_names(chain_length)}
        palette = rng.choice(WOOD_PALETTES)
        layout = dict(seed=seed, attempt=attempt, generator=GENERATOR, tier=cover, cover=cover,
                      cover_probability=None, lid_direction=direction, chain_length=chain_length,
                      decoys=decoys, sliders=sliders, parts=parts, release_order=order, covered=covered,
                      relock=relock_spec, relock_requested=relock,
                      item_xy=item_xy, deck=deck_for(direction),
                      item_asset=rng.choice(ITEM_CANDIDATES), tray_asset=rng.choice(TRAY_CANDIDATES),
                      colours=dict(wood=palette[0], deck=palette[1], lid=palette[2], metal=METAL,
                                   handle=rng.choice(HANDLE_COLOURS)))
        layout['plate'] = plate.build_plate(layout)
        if covered_names and layout['plate'] is None:
            note('plate')
            continue
        problems = validate_layout(layout)
        if not problems:
            layout['signature'] = layout_signature(layout)
            return layout
        note('validation: ' + problems[0].split(':')[0])
    raise RuntimeError(f'No valid layout for seed {seed} (chain {chain_length}, decoys {decoys}, '
                       f'cover {cover}, relock {relock}) in {attempts} attempts')
