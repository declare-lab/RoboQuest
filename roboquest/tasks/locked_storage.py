"""Locked Storage (tasks-v0 task 2, "fob search"; RoboQuest v1.1): a target behind a chain of locks.

A sibling of :class:`~roboquest.tasks.search_room.SearchRoom`, which it subclasses: the
work frame, the tray, Submit, compartment discovery, the disabling of every other fixture, the counter
spot samplers, the search object pool and the terminal score all come from there unchanged; the passive
closers do not (none in this task) and the locked fridge's handles hide with the rest. What
this task adds is the lock chain (package ``roboquest/locked/``).

**Story.** One target hides in a chain of locked compartments. A locked compartment opens only while
the token of its colour rests on its reader pad; the first token lies in the open, each later token
sits inside the previous compartment. Closing a compartment re-locks it unless its token is on the
pad, so a token or the target shut inside a re-locked compartment is locked away.

**Factors.** ``chain_depth`` 1/2/3 (the size axis) x ``dead_end`` 0/1 (one further locked compartment
whose token is reachable but which holds only a distractor). ``set_size`` is fixed at 1 and
``compartment_count`` at 6; both stay recorded spec fields so they can be raised later. ``double_door``
(``OPTIONAL_FACTORS``, default False, recorded in the spec) is a mint-time mix, not a grid cell: the
registry asks every evaluation cell for 2 of its 9 rows and every development cell for about the same
share.

**Placement rules (generator v3; ``locked/heights.py``).** Every place that holds
something -- a token, the target, a decoy -- is a drawer whose interior floor is at least 0.40 m above
the kitchen floor, or a counter spot; bottom drawers (0.32 m) and SingleCabinets are never among the six
compartments. A *double-door* instance puts the final target (bottle, can or tin; never a token, never
the mug the pool keeps to drawers) in a HingeCabinet base cabinet, inside the opened leaf's aperture, its
centre ``target_depth_m`` behind the door line (recorded in the spec -- 0.10 m, the search family's door
pose -- and placed exactly by the build; the rule caps it at 0.20 m); everything else in
it follows the height rule. Every task object on a counter (open tokens, pads, visible objects) keeps to
the front two thirds of a wall-backed counter's depth; islands are free (rule 3). The mint applies the
rules as a hard filter from the per-layout placement table (a kitchen short of six allowed compartments
or of tall-enough rule-1 drawers for the chain, the dead end and the decoys is refused), the build
binds only allowed compartments and records every place's floor, and G3 re-asserts the placement.
Hidden decoys are the confusable pool objects the kitchen can hide in a rule-1 drawer, at most two and
at least one (a 118 mm drawer kitchen leaves one: the can needs 131 mm and the bottle never hides in a
drawer).

**Split key and hold-outs.** ``d{chain_depth}-{drawers|door}-{lock colours}``, e.g.
``d2-drawers-cyan+purple``; the dead-end level is not in the key. The held-out class is a **purple
lock** anywhere in the instance (:func:`~roboquest.locked.structure.is_holdout`): no
development row uses purple, every evaluation row does, on a lock the structure stream chose. Under the
placement rules the chain's kinds are fixed (drawers, or drawers then one HingeCabinet) and the
double-door mix must appear in both splits, so no kind pattern can be held out; the lock colour is the
one chain choice that exists in every cell, kitchen and mix. Development keeps every kind pattern and
both dead-end levels at every depth, with 4 of the 5 lock colours
(:func:`~roboquest.locked.structure.dev_composition_counts`).

**Lock mechanics.** A locked compartment's door or drawer joint range is collapsed to
``closed +- 1e-4`` in ``sim.model.jnt_range`` -- the technique ``_disable_other_fixtures`` uses
statically -- with a stiff limit solver beside it (a MuJoCo limit is soft: with RoboCasa's own values a
locked door still swings 2 degrees under the contract's 4 N.m), and both are written back to unlock. The rule (``locked/rules.py``) is
evaluated every tick in :meth:`_post_action`: token on its pad -> unlocked; else closed -> locked;
else unchanged, so an open compartment stays open until it is shut again.

**Privilege.** Which compartment holds the target, where the tokens are, the chain order and the dead
end never reach the goal text or a policy observation. The lock colours are visible by design (plate,
pad and token share a colour) but which colour leads to the target is not stated. The scope
text carries no count; five or six pool objects stand on the counters depending on how
many decoys the kitchen can hide, which says nothing about where the target or the tokens are.
"""
from copy import deepcopy
import hashlib
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from roboquest.locked.objects import COLOUR_NAMES, LOCK_COLOURS, PAD_RADIUS_M, PAD_SIZE_M, PLATE_SIZE_M, TOKEN_SIZE_M, lock_plate, pad_name, plate_name, reader_pad, token as build_token, token_name
from roboquest.locked.rules import LockRule, is_closed, locked_range, token_on_pad
from roboquest.locked.heights import DOOR_TARGET_DEPTH_MAX_M, RULES, allowed_compartment, allowed_place, counter_band, depth_allowed, door_is_hinge, drawer_is_rule1, objects_hideable, placement_problems, target_depth
from roboquest.locked.structure import CHAIN_DEPTHS, COMPARTMENT_COUNT, DEAD_END_LEVELS, HOLDOUT_KEYS, PAD_FRONT_MAX_M, PAD_FRONT_MIN_M, PAD_LATERAL_MAX_M, PAD_SPACING_M, SET_SIZE, chain_binding, dev_composition_counts, front_centre, is_holdout as structure_holdout, pad_offsets, pad_spot, split_key as structure_key
from roboquest.search.covers import ElementObject
from roboquest.search.layout import base_blockers, compartment_candidates, floor_bounds_from_walls, hide_local, hidden_pose, sample_counter_spots, select_fillers, world_from_local
from roboquest.search.objects import POOL, confusable, fits_compartment
from roboquest.search.rules import PROGRESS_CAP, terminal_score
from roboquest.search.visibility import box_corners, camera_frusta, seen_by_any
from roboquest.harness.contract import CAMERAS
from roboquest.base.scene import ASSET_ROOT
from roboquest.tasks.search_room import DECOR_MARGIN_M, SETTLE_TICKS, SPOT_CLEARANCE_M, SUBMIT_LOCAL, SearchRoom, TRAY_HALF_M, TRAY_LOCAL, VISIBLE_SPACING_M, object_name

CONTRACT_VERSION = 'locked-storage-kitchen-v2'    # v2: placement rules, double-door mix, colour hold-out
GENERATOR_VERSION = 'v3'     # v3: height rule at mint, double_door, purple held out;
                             # v2: closer_count 0, no passive closers
GOAL_MIDDLE = ('Collect {target} on the tray, then press Submit. It may be anywhere in this kitchen; '
               'some compartments may not open, and a compartment with a coloured lock plate opens '
               'only while the token of that colour rests on its reader pad.')
GOAL_NOUN = 'objects'
HIDDEN_DISTRACTORS = 2         # at most: the confusable objects the kitchen can hide in a rule-1 drawer, at least
                               # one (the goal carries no count, so the number leaks nothing)
# Passive closers: none. The spec keeps `closer_count` (0 since generator v2; 1 or 2
# on v1 rows, accepted and ignored) and `closer_seed`, so older rows stay valid and reproducible; the build
# applies no spring to any compartment whatever the value.
CLOSER_COUNT = (0,)
CLOSER_COUNT_ACCEPTED = (0, 1, 2)
PAD_SPOT_TRIES = 24            # counter spots sampled per patch for the pad search (search uses 6 for hiding)
PAD_CLEARANCE_M = PAD_RADIUS_M + DECOR_MARGIN_M
PLATE_INSET_M = .11            # plate centre this far from the handle, towards the panel's middle
PLATE_EDGE_MARGIN_M = .02      # ... and never closer than this to the panel's edge
TOKEN_SLOT_M = .065          # two tokens sharing a compartment sit this far either side of its hide point
TOKEN_HALF_M = tuple(v / 2 for v in TOKEN_SIZE_M)
TOKEN_GAP_M = .012           # token face to a pool object's footprint, inside a shared compartment
TOKEN_FLOOR_Z = .15            # a token whose body drops below this is on the kitchen floor
LOCK_LOAD_N = 40.              # the drawer pull the lock is verified against (contract section 2)
LOCK_LOAD_NM = 4.              # ... and the door torque
LOCK_LOAD_TICKS = 40           # 2 s at the 20 Hz control rate
# A MuJoCo joint limit is a soft constraint: with RoboCasa's defaults a locked door still swings
# 0.0364 rad (2.1 deg) under the contract's 4 N.m and a locked drawer 1.2 mm under 40 N. Locked joints
# therefore also get a stiff limit solver (2 ms reference, near-unit impedance), which takes those to
# 0.00037 rad and 0.10 mm; the compartment's own values are restored together with its range.
LOCK_SOLREF = (.002, 1.)
LOCK_SOLIMP = (.9999, .99999, .0001, .5, 2.)
PROGRESS_WEIGHTS = dict(unlocked=.6, out=.2, tray=.2)


def lock_colours(count, rng):
    """``count`` distinct lock colours drawn from the palette, in draw order (one per locked compartment)."""
    if int(count) > len(COLOUR_NAMES):
        raise ValueError(f'the lock palette has {len(COLOUR_NAMES)} colours, fewer than the {count} needed')
    return [COLOUR_NAMES[int(i)] for i in rng.choice(len(COLOUR_NAMES), int(count), replace=False)]


def element_box(body):
    """Axis-aligned (centre, half) of a body's *own* geoms, in the body frame; ``None`` if unmeasurable.

    Sizes are read as written and rotated by the geom's ``euler``/``quat`` -- RoboCasa writes a cabinet
    panel's slab with a 90 degree x rotation, so a naive read of ``size`` names the wrong axes.
    """
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for geom in body.findall('geom'):
        raw = geom.get('size')
        if not raw:
            continue
        size = np.array([float(v) for v in raw.split()], float)
        kind = geom.get('type', 'sphere')
        if kind == 'box' and size.size == 3:
            half = size
        elif size.size >= 2:                 # cylinder / capsule: radius, half length
            half = np.array([size[0], size[0], size[1]])
        else:
            half = np.repeat(size[0], 3)
        pos = np.array([float(v) for v in (geom.get('pos') or '0 0 0').split()], float)
        rot = np.eye(3)
        if geom.get('euler'):
            rot = Rotation.from_euler('xyz', [float(v) for v in geom.get('euler').split()]).as_matrix()
        elif geom.get('quat'):
            q = [float(v) for v in geom.get('quat').split()]
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        extent = np.abs(rot) @ half
        low, high = np.minimum(low, pos - extent), np.maximum(high, pos + extent)
    if not np.isfinite(low).all():
        return None
    return (low + high) / 2, (high - low) / 2


class LockedStorage(SearchRoom):
    TASK_NAME = 'locked_storage'
    CONTRACT_VERSION = CONTRACT_VERSION
    GENERATOR_VERSION = GENERATOR_VERSION
    FACTORS = {'chain_depth': CHAIN_DEPTHS, 'dead_end': DEAD_END_LEVELS}
    OPTIONAL_FACTORS = {'double_door': False}     # a mint-time mix, never a grid cell
    HOLDOUT_KEYS = HOLDOUT_KEYS                   # empty: the hold-out is the `is_holdout` predicate below
    HORIZON = 20000
    BUDGET_TICKS = 20000       # provisional, the search value (contract section 1)
    # The fridge's handles hide with its locked doors: RoboCasa names them fridge_door_handle_*
    # and freezer_door_handle_*, outside the cabinet pattern the search task hides.
    HIDDEN_HANDLE_PATTERNS = SearchRoom.HIDDEN_HANDLE_PATTERNS + ('fridge_door_handle_', 'freezer_door_handle_')
    HIDE_ALL_HANDLES_OF = ('Fridge',)       # the patterns above miss RoboCasa's numbered fridge handles

    # ---- generator protocol (CPU only) ---------------------------------------
    @classmethod
    def goal(cls, spec):
        target = POOL[spec['targets'][0]['object']]['description']
        return cls.frame_goal(spec, GOAL_NOUN, len(spec.get('visible') or []),
                              GOAL_MIDDLE.format(target=target), scope=False)

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        depth, dead_end = int(factors['chain_depth']), int(factors['dead_end'])
        double_door = bool(factors.get('double_door', False))
        layout_id = int(factors['kitchen_layout'])
        ids = sorted(POOL)
        # The target: behind the HingeCabinet door of a double-door instance (an object the pool allows behind a
        # door and one of this kitchen's cabinets can hold), otherwise in a rule-1 drawer this kitchen has and
        # the object fits; the chain's kinds follow.
        eligible = objects_hideable(layout_id, 'door' if double_door else 'drawer', ids)
        if not eligible:
            raise ValueError(f'layout {layout_id} can hide no pool object in a '
                             f'{"HingeCabinet" if double_door else "rule-1 drawer"} (placement table)')
        target = eligible[int(structure.integers(len(eligible)))]
        chain_kinds = ['drawer'] * (depth - 1) + ['door' if double_door else 'drawer']
        dead_end_kind = 'drawer' if dead_end else None
        others = [o for o in ids if o != target]
        # Decoys: the confusable objects this kitchen can hide in a rule-1 drawer, at most two, at least one.
        candidates = objects_hideable(layout_id, 'drawer', sorted(o for o in others if confusable(o, target)))
        if not candidates:
            raise ValueError(f'layout {layout_id} can hide no decoy confusable with {target!r} in a rule-1 drawer '
                             f'(placement table)')
        n_hidden = min(HIDDEN_DISTRACTORS, len(candidates))
        hidden = sorted(candidates[int(i)] for i in structure.choice(len(candidates), n_hidden, replace=False))
        visible = [o for o in others if o not in hidden]
        colours = lock_colours(depth + dead_end, materials)
        # The door target's depth behind the door line (rule 2), recorded here and placed exactly by the build.
        door_depth = target_depth() if double_door else None
        # Where the dead end's token lies: in the open with token 1, or inside a chain *drawer* (never the door
        # of a double-door instance: a token obeys the height rule). Recorded in the spec, never announced.
        dead_end_token = None
        if dead_end:
            options = ['open'] + [f'chain_{i}' for i in range(depth) if chain_kinds[i] == 'drawer']
            dead_end_token = options[int(structure.integers(len(options)))]
        spec = dict(
            chain_depth=depth, dead_end=dead_end, set_size=SET_SIZE, compartment_count=COMPARTMENT_COUNT,
            layout_id=layout_id, double_door=double_door, rules=dict(RULES), target_depth_m=door_depth,
            targets=[dict(object=target, kind=chain_kinds[-1])],
            chain_kinds=chain_kinds, dead_end_kind=dead_end_kind, dead_end_token=dead_end_token,
            lock_colours=colours, hidden_distractors=hidden, visible=visible,
            build_seed=int(structure.integers(0, 2**31 - 1)),
            pose_seed=int(poses.integers(0, 2**31 - 1)),
            token_seed=int(materials.integers(0, 2**31 - 1)),
            closer_count=int(CLOSER_COUNT[int(structure.integers(len(CLOSER_COUNT)))]),
            closer_seed=int(structure.integers(0, 2**31 - 1)),
            placement=dict(tray=list(TRAY_LOCAL), submit=list(SUBMIT_LOCAL)), tray_half_m=TRAY_HALF_M)
        # Reject a kitchen that cannot host this cell before anything is built (the search capacity table and
        # the placement table), so `registry.mint` raises and `mint_grid` walks on to the next seed. The build
        # keeps its own checks as the safety net.
        problems = (cls.capacity_problems(layout_id, COMPARTMENT_COUNT, spec['targets'], hidden, visible)
                    + placement_problems(layout_id, spec, COMPARTMENT_COUNT))
        if problems:
            raise ValueError('; '.join(problems))
        return spec

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        depth, dead_end = spec.get('chain_depth'), spec.get('dead_end')
        double_door = spec.get('double_door')
        if depth not in CHAIN_DEPTHS:
            problems.append('chain_depth must be 1, 2 or 3')
        if dead_end not in DEAD_END_LEVELS:
            problems.append('dead_end must be 0 or 1')
        if not isinstance(double_door, bool):
            problems.append('double_door must be true or false')
        if spec.get('rules') != RULES:
            problems.append(f'rules must record the placement rules {RULES}')
        door_depth = spec.get('target_depth_m')
        if double_door is True and not depth_allowed(door_depth):
            problems.append(f'a double-door instance records target_depth_m, at most '
                            f'{RULES["door_target_depth_max_m"]} m behind the door line')
        if double_door is False and door_depth is not None:
            problems.append('target_depth_m must be null without a door target')
        if spec.get('set_size') != SET_SIZE:
            problems.append('set_size is fixed at 1')
        if spec.get('compartment_count') != COMPARTMENT_COUNT:
            problems.append('compartment_count is fixed at 6')
        targets = spec.get('targets') or []
        if len(targets) != 1 or targets[0].get('object') not in POOL:
            problems.append('exactly one target, drawn from the pool')
        kinds = spec.get('chain_kinds')
        expected_kinds = (['drawer'] * (depth - 1) + ['door' if double_door else 'drawer']
                          if depth in CHAIN_DEPTHS and isinstance(double_door, bool) else None)
        if not isinstance(kinds, list) or (expected_kinds is not None and kinds != expected_kinds):
            problems.append('chain_kinds must be rule-1 drawers, with a door as the last link only in a '
                            'double-door instance')
        elif targets and targets[0].get('object') in POOL:
            if targets[0].get('kind') != kinds[-1]:
                problems.append("the target's kind must be the last chain compartment's kind")
            elif kinds[-1] not in POOL[targets[0]['object']]['kinds']:
                problems.append(f"{targets[0]['object']} may not hide in a {kinds[-1]}")
        dead_kind = spec.get('dead_end_kind')
        if dead_end and dead_kind != 'drawer':
            problems.append('a dead end is a rule-1 drawer')
        if not dead_end and dead_kind is not None:
            problems.append('dead_end_kind must be null without a dead end')
        where = spec.get('dead_end_token')
        if dead_end:
            allowed = ['open'] + [f'chain_{i}' for i in range(depth if depth in CHAIN_DEPTHS else 0)
                                  if isinstance(kinds, list) and i < len(kinds) and kinds[i] == 'drawer']
            if where not in allowed:
                problems.append(f'dead_end_token must be one of {allowed} (never the door link)')
        elif where is not None:
            problems.append('dead_end_token must be null without a dead end')
        colours = spec.get('lock_colours')
        expected = (depth if isinstance(depth, int) else 0) + (dead_end if isinstance(dead_end, int) else 0)
        if (not isinstance(colours, list) or len(colours) != expected or len(set(colours)) != expected
                or any(c not in LOCK_COLOURS for c in colours or [])):
            problems.append(f'lock_colours must name {expected} distinct palette colours')
        objects = [t.get('object') for t in targets]
        hidden = spec.get('hidden_distractors') or []
        if (not 1 <= len(hidden) <= HIDDEN_DISTRACTORS or len(set(hidden)) != len(hidden)
                or set(hidden) & set(objects)):
            problems.append(f'between 1 and {HIDDEN_DISTRACTORS} distinct hidden distractors, none of them the target')
        for pool_id in hidden:
            if pool_id not in POOL or not any(confusable(pool_id, t) for t in objects if t in POOL):
                problems.append(f'hidden distractor {pool_id!r} shares neither category nor colour with the target')
            elif 'drawer' not in POOL[pool_id]['kinds']:
                problems.append(f'hidden distractor {pool_id!r} cannot hide in a drawer, the only place the height '
                                f'rule allows a decoy')
        expected_visible = sorted(o for o in POOL if o not in objects and o not in hidden)
        if sorted(spec.get('visible') or []) != expected_visible:
            problems.append('visible objects must be the rest of the pool')
        if (isinstance(depth, int) and isinstance(dead_end, int)
                and depth + dead_end + max(0, len(hidden) - dead_end) > COMPARTMENT_COUNT):
            problems.append('more locked and hidden compartments than compartment_count')
        for key in ('build_seed', 'pose_seed', 'token_seed', 'closer_seed'):
            if not isinstance(spec.get(key), int):
                problems.append(f'{key} must be an int')
        if spec.get('closer_count') not in CLOSER_COUNT_ACCEPTED:
            problems.append('closer_count must be 0, 1 or 2 (0 since generator v2; the build applies no closers)')
        placement = spec.get('placement') or {}
        if not all(isinstance(placement.get(k), list) and len(placement[k]) == 2 for k in ('tray', 'submit')):
            problems.append('placement needs tray and submit frame positions')
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """CPU-only checks: the spec invariants, the search capacity table and the placement table (rules 1-2)."""
        problems = cls.validate_spec(spec)
        report = dict(chain_depth=spec.get('chain_depth'), dead_end=spec.get('dead_end'),
                      double_door=spec.get('double_door'), target_depth_m=spec.get('target_depth_m'),
                      chain_kinds=spec.get('chain_kinds'),
                      dead_end_kind=spec.get('dead_end_kind'), lock_colours=spec.get('lock_colours'),
                      capacity_layout=spec.get('layout_id'), rules=spec.get('rules'),
                      split_key=None if problems else cls.split_key(spec))
        if not problems:
            problems = (cls.capacity_problems(spec['layout_id'], COMPARTMENT_COUNT, spec['targets'],
                                              spec.get('hidden_distractors') or [], spec.get('visible') or [])
                        + placement_problems(spec['layout_id'], spec, COMPARTMENT_COUNT))
        report['problems'] = problems
        return not problems, report

    @classmethod
    def split_key(cls, spec):
        return structure_key(spec['chain_depth'], spec.get('double_door'), spec['lock_colours'])

    @classmethod
    def is_holdout(cls, split_key):
        """Held out for evaluation: a purple lock somewhere in the instance (``structure.is_holdout``)."""
        return structure_holdout(split_key)

    @classmethod
    def dev_compositions(cls):
        """``{depth: (kept, total)}`` split keys development keeps; quoted in the module docstring."""
        return dev_composition_counts()

    # ---- construction --------------------------------------------------------
    def __init__(self, instance, **kwargs):
        self.chain_ids, self.locked_ids, self.dead_end_id = [], [], None
        self.pads, self.tokens, self.plates = {}, {}, {}
        self.pad_refusals = []
        self._lock = None
        self._saved_ranges, self._saved_solver, self._lock_joints = {}, {}, {}
        self._token_bodies, self._token_dofs, self._pad_geoms = {}, {}, {}
        self._open_token_choice, self._open_token_spots = {}, {}
        self._locked_away = {}
        self._max_seat_snap = 0.
        self.door_target = None
        self._excluded = []
        self._counter_places = {}
        super().__init__(instance, **kwargs)

    def build_task(self):
        self.place_work_frame()
        self._records = self._fixture_records()
        self._floor = floor_bounds_from_walls(self._records)
        self._blockers = base_blockers(self._records)
        usable = compartment_candidates(self._records, self._floor)
        # The height rule: only drawers with a floor at FLOOR_MIN_M or above and HingeCabinets
        # may be among the six; bottom drawers and SingleCabinets are disabled with the rest of the kitchen.
        candidates = [c for c in usable if allowed_compartment(c)]
        self._excluded = [dict(id=c['id'], kind=c['kind'], cls=c['cls'],
                               floor_z=round(float(c['region']['floor_z']), 4))
                          for c in usable if not allowed_compartment(c)]
        self.door_target = None
        self._counter_places = {}
        count = int(self.spec['compartment_count'])
        if len(candidates) < count:
            raise ValueError(f'{self.TASK_NAME}: layout {self.layout_id} offers {len(candidates)} compartments '
                             f'allowed by the height rule (of {len(usable)} usable), fewer than '
                             f'compartment_count={count}')
        rng = np.random.default_rng([int(self.spec['build_seed']), 20260922])
        pose_rng = np.random.default_rng([int(self.spec['pose_seed']), 20260922])
        token_rng = np.random.default_rng([int(self.spec['token_seed']), 20260922])

        self._build_tray()
        self.add_submit_button(self.spec['placement']['submit'])
        exclude = [self._work_polygon()]
        self._keepouts = self._decor_keepouts()
        # Pad spots are drawn first, from the same edge band search uses for its hiding spots, so the lock
        # hardware never depends on how many visible objects there are.
        pad_pool = sample_counter_spots(self._records, self._floor, rng, per_patch=PAD_SPOT_TRIES,
                                        include_shelves=False, exclude=exclude, edge_only=True,
                                        keepouts=self._keepouts, clearance=PAD_CLEARANCE_M)
        # Rule 3 (counter band): pads and open tokens only in the front two thirds of a wall-backed counter.
        pad_pool = [s for s in pad_pool if counter_band(s, self._records)['ok']]
        taken_pads, too_far = [], {}

        def pad_for(comp):
            # `chain_binding` commits the first spot this returns, so a hit is recorded as taken right
            # here; otherwise every lock of the chain would be offered the same first free spot.
            rejected = []
            spot = pad_spot(comp, pad_pool, self._records, exclude=exclude, keepouts=self._keepouts,
                            taken=taken_pads, rejected=rejected)
            if spot is not None:
                taken_pads.append(spot['xy'])
            elif rejected:
                # The nearest spot the forward bound turned away: a refusal is only actionable if the
                # report says how far the closest usable counter actually was.
                too_far[comp['id']] = min(f for _, f in rejected)
            return spot

        target = self.spec['targets'][0]['object']
        depth = int(self.spec['chain_depth'])
        double_door = bool(self.spec.get('double_door'))

        def fits(index, comp):
            # Token links and the dead end: rule-1 drawers. The last link: a rule-1 drawer the target fits, or
            # the HingeCabinet of a double-door instance (rule 2).
            if index != depth - 1:
                return allowed_place(comp, 'link')
            return allowed_place(comp, 'target', double_door) and fits_compartment(target, comp)

        by_id = {c['id']: c for c in candidates}
        self.pad_refusals = []
        try:
            self.chain_ids, self.dead_end_id, pad_spots = chain_binding(
                by_id, self.spec['chain_kinds'], self.spec.get('dead_end_kind'), rng, pad_for,
                self.pad_refusals, fits=fits)
        except ValueError as exc:
            # `chain_binding` formats its message before this runs, so re-raise with the distance to the
            # nearest counter the forward bound turned away: that is what makes a refusal actionable.
            for row in self.pad_refusals:
                row['nearest_spot_forward_m'] = round(too_far[row['compartment']], 4) \
                    if row['compartment'] in too_far else None
            raise ValueError(f'{exc} [nearest usable counter, m forward: '
                             f'{ {r["compartment"]: r["nearest_spot_forward_m"] for r in self.pad_refusals} }]'
                             ) from exc
        assert len(taken_pads) == len(pad_spots)      # every committed pad spot was recorded as taken
        self.locked_ids = list(self.chain_ids) + ([self.dead_end_id] if self.dead_end_id is not None else [])

        # Hidden distractors: one in the dead end when there is one, the rest in free compartments.
        used = {cid: by_id[cid] for cid in self.locked_ids}
        self.places = {'target_0': dict(role='target', object=target, kind=by_id[self.chain_ids[-1]]['kind'],
                                        compartment=self.chain_ids[-1], chain_index=depth - 1)}
        distractors = list(self.spec['hidden_distractors'])
        if self.dead_end_id is not None:
            if not fits_compartment(distractors[0], by_id[self.dead_end_id]):
                distractors = distractors[::-1]
            if not fits_compartment(distractors[0], by_id[self.dead_end_id]):
                raise ValueError(f'{self.TASK_NAME}: neither hidden distractor fits the dead-end compartment '
                                 f'{self.dead_end_id} on layout {self.layout_id}')
            self.places['distractor_0'] = dict(role='hidden_distractor', object=distractors[0],
                                               kind=by_id[self.dead_end_id]['kind'],
                                               compartment=self.dead_end_id, dead_end=True)
            distractors = distractors[1:]
        for index, pool_id in enumerate(distractors, start=len(self.places) - 1):
            options = [c for c in candidates if c['id'] not in used and allowed_place(c, 'decoy')
                       and fits_compartment(pool_id, c)]
            if not options:
                raise ValueError(f'{self.TASK_NAME}: layout {self.layout_id} has no unused compartment that '
                                 f'fits hidden distractor {pool_id!r}')
            pick = options[int(rng.integers(len(options)))]
            used[pick['id']] = pick
            self.places[f'distractor_{index}'] = dict(role='hidden_distractor', object=pool_id, kind=pick['kind'],
                                                      compartment=pick['id'], dead_end=False)
        chosen = select_fillers(candidates, list(used.values()), rng, count)
        self.compartments = {c['id']: c for c in chosen}

        # Reader pads on the counter, one per locked compartment, in that compartment's lock colour.
        colours = list(self.spec['lock_colours'])
        self.pads = {}
        for index, cid in enumerate(self.locked_ids):
            spot = pad_spots[cid]
            pos = (float(spot['xy'][0]), float(spot['xy'][1]), float(spot['top_z']))
            record = reader_pad(self.model.worldbody, pad_name(cid), colours[index], pos,
                                float(self.compartments[cid]['rot']))
            record.pop('body')
            record.update(compartment=cid, lateral_m=spot.get('pad_lateral_m'), forward_m=spot.get('pad_forward_m'),
                          standoff=spot.get('standoff'),
                          role='dead_end' if cid == self.dead_end_id else f'chain_{index}')
            self.pads[cid] = record
            self._counter_places[pad_name(cid)] = dict(role='pad', compartment=cid, **counter_band(spot, self._records))
            self.asset_evidence[pad_name(cid)] = dict(
                source='roboquest.locked.objects.reader_pad', colour=colours[index],
                size_m=list(PAD_SIZE_M), welded='no joint: the reader belongs to the counter')

        # Tokens: token k+1 lives inside chain compartment k, token 1 in the open; the dead end's token sits
        # with token 1 or inside a chain compartment.
        contents = {cid: [] for cid in self.compartments}
        plans = []
        for index, cid in enumerate(self.chain_ids):
            host = 'open' if index == 0 else self.chain_ids[index - 1]
            plans.append(dict(index=index + 1, colour=colours[index], compartment=cid, where=host, dead_end=False))
            if host != 'open':
                contents[host].append(token_name(index + 1))
        if self.dead_end_id is not None:
            where = self.spec['dead_end_token']
            host = 'open' if where == 'open' else self.chain_ids[int(where.split('_')[1])]
            plans.append(dict(index=len(self.chain_ids) + 1, colour=colours[-1], compartment=self.dead_end_id,
                              where=host, dead_end=True))
            if host != 'open':
                contents[host].append(token_name(len(self.chain_ids) + 1))

        slots = {cid: self._token_slots(cid, names) for cid, names in contents.items() if names}
        self.tokens = {}
        holder = ET.Element('holder')      # the deferred merge in `_create_objects` attaches these bodies
        for plan in plans:
            name = token_name(plan['index'])
            if plan['where'] == 'open':
                pos = (0., 0., float(self.work['top_z']))      # provisional; bound to a seen spot at reset
            else:
                pose = self._token_pose_in(self.compartments[plan['where']], *slots[plan['where']][name])
                pos = (pose['xy'][0], pose['xy'][1], pose['z'])
            yaw = float(token_rng.uniform(-np.pi, np.pi))
            record = build_token(holder, name, plan['colour'], pos, yaw)
            body = record.pop('body')
            self.objects[name] = ElementObject(name, body, top_z=TOKEN_SIZE_M[2],
                                               horizontal_radius=float(max(record['half_xy'])))
            quat = Rotation.from_euler('z', yaw).as_quat()
            self._poses[name] = [float(pos[0]), float(pos[1]), float(pos[2]),
                                 float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]
            record.update(compartment=plan['compartment'], where=plan['where'], index=plan['index'],
                          dead_end=bool(plan['dead_end']), yaw=yaw)
            self.tokens[name] = record
            self.task_spec['objects'][name] = dict(
                kind='token', fixed=False, role='token', colour=plan['colour'], opens=plan['compartment'],
                where=plan['where'], yaw=yaw, size_xyz_m=list(TOKEN_SIZE_M), mass_kg=record['mass'],
                description=f'{plan["colour"]} token', height_m=float(TOKEN_SIZE_M[2]),
                # Same schema as a pool entry's, so the search motor's `grasp_top_down` can take a token:
                # the body origin is the fob's bottom plane, so the finger pads close at its mid height.
                grasp=dict(anchor_local_top=(0., 0., TOKEN_SIZE_M[2] / 2),
                           anchor_local_front=(0., 0., TOKEN_SIZE_M[2] / 2), handle=False))
            self.asset_evidence[name] = dict(source='roboquest.locked.objects.token',
                                             colour=plan['colour'], mass_kg=record['mass'],
                                             appearance='flat lock colour with a 1 cm white dot')
            self.places[f'token_{plan["index"]}'] = dict(
                role='token', object=name, kind='token_open' if plan['where'] == 'open' else 'token_hidden',
                compartment=None if plan['where'] == 'open' else plan['where'], opens=plan['compartment'])

        # Pool objects: the target and the hidden distractors, inside their compartments.
        for key in sorted(self.places):
            place = self.places[key]
            if place['role'] not in ('target', 'hidden_distractor'):
                continue
            comp = self.compartments[place['compartment']]
            if place['role'] == 'target' and comp['kind'] == 'door':
                pose = self._door_target_pose(comp, place['object'], pose_rng)
            else:
                pose = hidden_pose(comp, place['object'], pose_rng, POOL[place['object']])
            place.update(hide_local=pose['hide_local'], hide_world=pose['hide_world'],
                         floor_z=round(float(comp['region']['floor_z']), 4))
            if place['role'] == 'target' and comp['kind'] == 'door':
                self.door_target = self._door_target_record(comp, pose)
            self._add_object(place['object'], pose['xy'], pose['z'], pose['yaw'], place['role'])

        # Visible pool objects on the counters, clear of the pads, of the open-token candidates and of each other.
        anywhere = sample_counter_spots(self._records, self._floor, rng, per_patch=14, include_shelves=False,
                                        exclude=exclude, edge_only=False, keepouts=self._keepouts,
                                        clearance=SPOT_CLEARANCE_M)
        anywhere = [s for s in anywhere if counter_band(s, self._records)['ok']]      # rule 3

        def clear_of(xy, others, gap=VISIBLE_SPACING_M):
            return all(np.linalg.norm(np.subtract(xy, other)) >= gap for other in others)
        visible_spots, taken = [], []
        for pool_id in self.spec['visible']:
            spot = next((s for s in anywhere if clear_of(s['xy'], taken)
                         and clear_of(s['xy'], taken_pads, VISIBLE_SPACING_M + PAD_RADIUS_M)), None)
            if spot is None:
                raise ValueError(f'{self.TASK_NAME}: no counter spot left for visible object {pool_id!r} '
                                 f'on layout {self.layout_id}')
            taken.append(spot['xy'])
            visible_spots.append((pool_id, spot))
            self._counter_places[object_name(pool_id)] = dict(role='visible', **counter_band(spot, self._records))
            self._add_object(pool_id, spot['xy'], spot['top_z'], float(pose_rng.uniform(-np.pi, np.pi)), 'visible')
        # Tokens in the open go on counter spots at least one policy camera sees; the candidates are edge spots
        # clear of the pads and of the visible objects, ranked at reset once the cameras are posed.
        self._open_candidates = [s for s in pad_pool if clear_of(s['xy'], taken)
                                 and clear_of(s['xy'], taken_pads, VISIBLE_SPACING_M + PAD_RADIUS_M)]
        open_tokens = [n for n, t in sorted(self.tokens.items()) if t['where'] == 'open']
        if len(self._open_candidates) < len(open_tokens):
            raise ValueError(f'{self.TASK_NAME}: layout {self.layout_id} offers {len(self._open_candidates)} '
                             f'counter spots for {len(open_tokens)} tokens in the open')

        # Passive closers: none. `spec.closer_count` and `spec.closer_seed` are inert; every
        # compartment is a plain drawer or door that stays where the robot leaves it.
        self._closer_ids = []

        placement = self._placement_report()
        if not placement['ok']:
            bad = ([k for k, r in placement['places'].items() if not r['ok']]
                   + [c for c, r in placement['compartments'].items() if not r['ok']])
            raise ValueError(f'{self.TASK_NAME}: the placement rules are broken at build on layout '
                             f'{self.layout_id}: {bad}')
        hidden_ids = ([object_name(self.spec['targets'][0]['object'])]
                      + [object_name(o) for o in self.spec['hidden_distractors']]
                      + [n for n, t in sorted(self.tokens.items()) if t['where'] != 'open'])
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, family='search',
            layout_id=int(self.layout_id), style_id=int(self.style_id),
            double_door=double_door, placement_rules=dict(RULES), placement=placement,
            door_target=deepcopy(self.door_target), excluded_compartments=deepcopy(self._excluded),
            target_ids=[object_name(self.spec['targets'][0]['object'])],
            targets=deepcopy(self.spec['targets']),
            compartments={cid: {k: c[k] for k in ('kind', 'fixture', 'cls', 'joints', 'joint_ranges', 'handles',
                                                  'standoff', 'region', 'front_normal', 'pos', 'rot', 'size',
                                                  'open_side', 'handle_side')}
                          for cid, c in self.compartments.items()},
            places=deepcopy(self.places), chain=list(self.chain_ids), dead_end=self.dead_end_id,
            locked_compartments=list(self.locked_ids),
            lock_colours={cid: self.pads[cid]['colour'] for cid in self.locked_ids},
            pads=deepcopy(self.pads), pad_refusals=deepcopy(self.pad_refusals), tokens=deepcopy(self.tokens),
            hidden_object_ids=hidden_ids,
            visible_objects={object_name(p): s for p, s in visible_spots},
            candidate_count=len(candidates), candidate_ids=[c['id'] for c in candidates],
            tray=deepcopy(self.tray), floor_bounds=self._floor, pool=deepcopy(POOL),
            closer_ids=[],
            closer_rule='none: the fob search has no passive closers; spec.closer_count and closer_seed are inert',
            scoring='First physical Submit freezes the score: the target upright inside the tray square with a real '
                    'upward support contact, nothing else on the tray, all objects released and settled.')
        self.asset_evidence['layout'] = {
            'layout_id': int(self.layout_id), 'style_id': int(self.style_id),
            'pool_assets': {k: {'source': str(ASSET_ROOT / v['asset'] / 'model.xml'),
                                'xml_sha256': hashlib.sha256((ASSET_ROOT / v['asset'] / 'model.xml').read_bytes()).hexdigest()}
                            for k, v in POOL.items()},
            'selection_rules': 'roboquest.search.layout + roboquest.locked.structure + '
                               'roboquest.locked.heights (floor >= 0.40 m for every place that holds '
                               'something; a HingeCabinet target only in a double-door instance; no bottom drawer, '
                               'no SingleCabinet)',
            'disabled_fixtures': 'all other drawer/door/appliance-door joints locked, their handle geoms hidden',
            'locks': 'locked compartments collapsed to closed +- 1e-4 in sim.model.jnt_range at runtime, '
                     f'with a stiff limit solver (solref {LOCK_SOLREF}, solimp {LOCK_SOLIMP}) while locked',
            'closers': 'none: no compartment carries a spring',
            'fridge_handles': 'hidden with the locked fridge (HIDDEN_HANDLE_PATTERNS covers fridge_door_handle_ and '
                              'freezer_door_handle_)'}

    def _door_target_pose(self, comp, object_id, rng):
        """The double-door target's pose (rule 2): the search family's lateral door pose (``hide_local``: inside
        the aperture of the leaf that opens) with the object's centre exactly ``spec.target_depth_m`` behind the
        cabinet's front face (local y = -depth/2, the door line). Raises when the interior cannot hold it that
        deep, so the build gate rejects the kitchen."""
        depth = float(self.spec['target_depth_m'])
        local = list(hide_local(comp))
        front_y = -float(comp['size'][1]) / 2
        local[1] = front_y + depth
        y0, sy = float(comp['region']['offset_local'][1]), float(comp['region']['size_xy'][1])
        room = max(POOL[object_id]['size_xyz_m'][:2]) / 2 + .01
        if not (y0 - sy / 2 + room <= local[1] <= y0 + sy / 2 - room):
            raise ValueError(f'{self.TASK_NAME}: {comp["id"]} on layout {self.layout_id} cannot hold {object_id!r} '
                             f'{depth} m behind its door line (interior y {y0 - sy / 2:.3f}..{y0 + sy / 2:.3f})')
        world = world_from_local(comp['pos'], comp['rot'], local)
        yaw = float(rng.uniform(-np.pi, np.pi))
        return dict(xy=world[:2].tolist(), z=float(comp['region']['floor_z']), yaw=yaw,
                    hide_local=[float(v) for v in local], hide_world=world.tolist())

    def _door_target_record(self, comp, pose):
        """The double-door target's geometry (rule 2): in a HingeCabinet, inside the aperture of the leaf that
        opens, its centre ``spec.target_depth_m`` behind the cabinet's front face (the door line), within the
        rule's limits. Raises when the rule is broken, so the build gate rejects the kitchen."""
        front_y = -float(comp['size'][1]) / 2                       # the fixture's front face, local y
        depth_behind = float(pose['hide_local'][1]) - front_y
        seam = float(pose['hide_local'][0]) - float(comp['region']['offset_local'][0])
        record = dict(compartment=comp['id'], cls=comp['cls'], double_door=bool(self.spec.get('double_door')),
                      width_m=round(float(comp['size'][0]), 3),
                      interior_width_m=round(float(comp['region']['size_xy'][0]), 3),
                      floor_z=round(float(comp['region']['floor_z']), 4),
                      target_depth_m=self.spec.get('target_depth_m'),
                      depth_behind_door_line_m=round(depth_behind, 4), lateral_from_seam_m=round(seam, 4),
                      leaf=comp.get('open_side') or '', depth_max_m=DOOR_TARGET_DEPTH_MAX_M)
        if (not record['double_door'] or not door_is_hinge(comp) or not depth_allowed(depth_behind)
                or abs(depth_behind - float(self.spec['target_depth_m'])) > 1e-3):
            raise ValueError(f'{self.TASK_NAME}: the door target on layout {self.layout_id} breaks rule 2: {record}')
        return record

    def _placement_report(self):
        """Every place that holds something against rule 1 (rule 2 for the door target) and every enabled
        compartment against the allowed classes; ``ok`` is the conjunction (the build and G3 fail closed on it)."""
        double_door = bool(self.spec.get('double_door'))
        rows = {}
        for key, place in sorted(self.places.items()):
            cid = place.get('compartment')
            if cid is None:
                band = self._counter_places.get(place.get('object'))     # an open token is bound at reset
                rows[key] = dict(role=place['role'], where='counter', ok=True if band is None else bool(band['ok']),
                                 counter_band=deepcopy(band))
                continue
            comp = self.compartments[cid]
            if place['role'] == 'target' and comp['kind'] == 'door':
                ok = (double_door and door_is_hinge(comp) and self.door_target is not None
                      and depth_allowed(self.door_target['depth_behind_door_line_m'])
                      and abs(self.door_target['depth_behind_door_line_m'] - float(self.spec['target_depth_m'])) <= 1e-3)
            else:
                ok = drawer_is_rule1(comp)
            rows[key] = dict(role=place['role'], compartment=cid, kind=comp['kind'], cls=comp['cls'],
                             floor_z=round(float(comp['region']['floor_z']), 4), ok=bool(ok))
        compartments = {cid: dict(kind=c['kind'], cls=c['cls'], floor_z=round(float(c['region']['floor_z']), 4),
                                  ok=bool(allowed_compartment(c)))
                        for cid, c in sorted(self.compartments.items())}
        counters = deepcopy(self._counter_places)
        ok = (all(r['ok'] for r in rows.values()) and all(c['ok'] for c in compartments.values())
              and all(c['ok'] for c in counters.values()))
        return dict(ok=bool(ok), rules=dict(RULES), places=rows, compartments=compartments, counters=counters,
                    door_target=deepcopy(self.door_target))

    def _token_pose_in(self, comp, lateral=0., depth=0.):
        """World pose for a token hidden inside ``comp``, offset from the compartment's hide point.

        ``lateral``/``depth`` are metres along the fixture's local x/y, from :meth:`_token_slots`, and
        are clipped to the interior so a token never ends up inside a wall of the compartment.
        """
        local = list(hide_local(comp))
        half = np.asarray(comp['region']['size_xy'], float) / 2
        centre = np.asarray(comp['region']['offset_local'][:2], float)
        for axis, delta in ((0, lateral), (1, depth)):
            room = max(float(half[axis]) - TOKEN_HALF_M[axis], 0.)
            local[axis] = float(np.clip(local[axis] + delta, centre[axis] - room, centre[axis] + room))
        world = world_from_local(comp['pos'], comp['rot'], local)
        return dict(xy=[float(world[0]), float(world[1])], z=float(comp['region']['floor_z']),
                    hide_local=[float(v) for v in local], hide_world=[float(v) for v in world])

    def _token_slots(self, cid, names):
        """``{token name: (lateral, depth)}`` offsets from compartment ``cid``'s hide point.

        The target and the hidden distractors hide at the *centre* of that point (``search.layout``'s
        ``hidden_pose``, which this task must not change), so when a token shares their compartment it
        steps out sideways past their footprint, alternating sides. A first gate batch failed G1 on 4 of
        24 instances with 20 token-versus-can overlaps up to 12 mm deep -- and G2 with the token then
        squirted up to 94 mm out of the drawer -- because the token sat exactly on the can. If the step
        does not fit across the interior the token goes deeper instead, behind the pool object; with no
        pool object in the compartment the tokens simply share the hide point symmetrically.
        """
        radius = max([max(POOL[p['object']]['size_xyz_m'][:2]) / 2 for p in self.places.values()
                      if p.get('compartment') == cid and p['role'] in ('target', 'hidden_distractor')],
                     default=0.)
        comp, out = self.compartments[cid], {}
        base = np.asarray(hide_local(comp), float) - np.asarray(comp['region']['offset_local'], float)
        for slot, name in enumerate(names):
            if radius <= 0.:
                out[name] = ((slot - (len(names) - 1) / 2) * 2 * TOKEN_SLOT_M, 0.)
                continue
            reach = radius + TOKEN_GAP_M + (slot // 2) * 2 * TOKEN_SLOT_M
            lateral = (reach + TOKEN_HALF_M[0]) * (1. if slot % 2 == 0 else -1.)
            room = float(comp['region']['size_xy'][0]) / 2 - TOKEN_HALF_M[0]
            out[name] = ((lateral, 0.) if abs(base[0] + lateral) <= room
                         else (0., reach + TOKEN_HALF_M[1]))
        return out

    # ---- lock plates on the moving fronts -------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._attach_lock_plates()

    def _moving_body(self, cid):
        """The MJCF body carrying a compartment's own joint (the drawer box or the door leaf)."""
        joints = set(self.compartments[cid]['joints'])
        for body in self.model.root.iter('body'):
            for joint in body.findall('joint'):
                if joint.get('name') in joints:
                    return body
        return None

    def _panel_elements(self, cid):
        """(panel body, handle body) of a compartment's moving front, or ``(None, None)``.

        RoboCasa builds the front as a ``CabinetPanel`` object named ``{fixture}_{door_name}``, so its root
        body is ``..._door_main`` with the handle object ``..._handle_main`` under it, and the whole panel
        is a child of the body the joint moves. Panel-local axes: x = width, y = depth (outward), z = height.
        """
        moving = self._moving_body(cid)
        if moving is None:
            return None, None
        panel = next((b for b in moving.iter('body') if (b.get('name') or '').endswith('_door_main')), None)
        if panel is None:
            return None, None
        handle = next((b for b in panel.iter('body') if (b.get('name') or '').endswith('_handle_main')), None)
        return panel, handle

    def _attach_lock_plates(self):
        """A 4 x 4 cm coloured plate beside each locked compartment's handle, on its moving front.

        Falls back (contract section 2) to a plate standing on the counter edge directly above the
        compartment's front when the moving panel or its handle is not in the MJCF. The mode is recorded
        per compartment in ``task_spec['plates']``.
        """
        if not self.pads:
            return
        self.plates = {}
        for cid in self.locked_ids:
            colour = self.pads[cid]['colour']
            panel, handle = self._panel_elements(cid)
            box = element_box(panel) if panel is not None else None
            if panel is None or handle is None or box is None:
                self.plates[cid] = self._counter_plate(cid, colour)
                continue
            centre, half = box
            handle_pos = np.array([float(v) for v in (handle.get('pos') or '0 0 0').split()], float)
            outward = 1. if handle_pos[1] >= centre[1] else -1.
            reach = max(0., half[0] - PLATE_SIZE_M[0] / 2 - PLATE_EDGE_MARGIN_M)
            away = -math.copysign(1., handle_pos[0]) if abs(handle_pos[0]) > 1e-6 else 1.
            rise = max(0., half[2] - PLATE_SIZE_M[1] / 2 - PLATE_EDGE_MARGIN_M)
            pos = [float(np.clip(handle_pos[0] + away * PLATE_INSET_M, centre[0] - reach, centre[0] + reach)),
                   float(centre[1] + outward * (half[1] + PLATE_SIZE_M[2] / 2 + .001)),
                   float(np.clip(handle_pos[2], centre[2] - rise, centre[2] + rise))]
            # RoboCasa may run _load_model more than once on the same fixture elements: never a second plate
            # on the panel (the gate batch saw "repeated name 'lock_plate_...' in geom" on two rows).
            for stale in [g for g in panel.findall('geom') if (g.get('name') or '').startswith(plate_name(cid))]:
                panel.remove(stale)
            record = lock_plate(panel, plate_name(cid), colour, pos, 1)
            record.update(attached_to=panel.get('name'), mode='moving_panel', compartment=cid,
                          handle=handle.get('name'), outward=float(outward),
                          panel_half=[float(v) for v in half])
            self.plates[cid] = record
        self.task_spec['plates'] = deepcopy(self.plates)

    def _counter_plate(self, cid, colour):
        """Fallback: the plate on a short post at the counter edge above the compartment's front."""
        comp = self.compartments[cid]
        xy = front_centre(comp)
        name = plate_name(cid)
        z = float(self.pads[cid]['centre'][2])
        for stale in [b for b in self.model.worldbody.findall('body') if b.get('name') == name + '_body']:
            self.model.worldbody.remove(stale)
        body = ET.SubElement(self.model.worldbody, 'body', name=name + '_body',
                             pos=f'{float(xy[0]):.9g} {float(xy[1]):.9g} {z:.9g}')
        record = lock_plate(body, name, colour, (0., 0., PLATE_SIZE_M[1] / 2 + .002), 1)
        record.update(attached_to=name + '_body', mode='counter_edge_fallback', compartment=cid)
        return record

    # ---- runtime ---------------------------------------------------------------
    def setup_task_references(self):
        # Put RoboCasa's own ranges back before the base sizes the passive closers from them.
        self._restore_ranges()
        super().setup_task_references()
        m = self.sim.model
        self._lock_joints, self._saved_ranges, self._saved_solver = {}, {}, {}
        for cid in self.locked_ids:
            rows = []
            for joint in self.compartments[cid]['joints']:
                jid = int(m.joint_name2id(joint))
                low, high = (float(v) for v in m.jnt_range[jid])
                closed = low if abs(low) <= abs(high) else high
                self._saved_ranges[joint] = (low, high)
                self._saved_solver[joint] = (np.array(m.jnt_solref[jid], float).copy(),
                                             np.array(m.jnt_solimp[jid], float).copy())
                rows.append(dict(joint=joint, jid=jid, qadr=int(m.jnt_qposadr[jid]), dadr=int(m.jnt_dofadr[jid]),
                                 closed=closed, travel=abs(high - low)))
            self._lock_joints[cid] = rows
        self._token_bodies = {name: int(m.body_name2id(name)) for name in self.tokens}
        self._token_dofs = {name: int(m.jnt_dofadr[int(m.joint_name2id(self.tokens[name]['joint']))])
                            for name in self.tokens}
        self._pad_geoms = {cid: [g for g in range(m.ngeom)
                                 if (m.geom_id2name(g) or '').startswith(pad_name(cid) + '_')]
                           for cid in self.locked_ids}
        self._lock = LockRule(self.locked_ids)
        self._apply_lock_state(force=True, seat=False)
        self.task_spec['lock_joints'] = {cid: [{k: r[k] for k in ('joint', 'closed', 'travel')} for r in rows]
                                         for cid, rows in self._lock_joints.items()}

    def _restore_ranges(self):
        """RoboCasa's own ranges back into the compiled model, so a second ``setup_task_references`` never
        sizes a passive closer against a collapsed (zero-travel) range."""
        if not self._saved_ranges or getattr(self, 'sim', None) is None:
            return
        m = self.sim.model
        for joint, (low, high) in self._saved_ranges.items():
            try:
                jid = int(m.joint_name2id(joint))
            except Exception:      # noqa: BLE001 - a freshly compiled model need not carry the old joints
                continue
            m.jnt_range[jid] = (low, high)
            solver = self._saved_solver.get(joint)
            if solver is not None:
                m.jnt_solref[jid], m.jnt_solimp[jid] = solver

    def _apply_lock_state(self, force=False, seat=True):
        """Write the lock state into ``sim.model.jnt_range``; seat a re-locking compartment at its limit.

        A compartment re-locks within ``CLOSED_FRACTION`` of its travel, so the bolt has to take up that last
        slack: qpos snaps to the closed limit and qvel to zero, which is what a real latch does. The largest
        snap seen in an episode is kept in ``_max_seat_snap`` and reported in the snapshot.
        """
        m, data = self.sim.model, self.sim.data
        snaps = []
        for cid, rows in self._lock_joints.items():
            locked = self._lock.locked[cid]
            for row in rows:
                low, high = locked_range(row['closed']) if locked else self._saved_ranges[row['joint']]
                current = (float(m.jnt_range[row['jid']][0]), float(m.jnt_range[row['jid']][1]))
                if not force and abs(current[0] - low) <= 1e-12 and abs(current[1] - high) <= 1e-12:
                    continue
                if locked and seat:
                    snaps.append(abs(float(data.qpos[row['qadr']]) - row['closed']))
                    data.qpos[row['qadr']] = row['closed']
                    data.qvel[row['dadr']] = 0.
                m.jnt_range[row['jid']] = (low, high)
                # A soft limit lets a locked door swing 2 degrees under the contract load; a locked
                # joint gets the stiff solver, an unlocked one RoboCasa's own feel back.
                if locked:
                    m.jnt_solref[row['jid']], m.jnt_solimp[row['jid']] = LOCK_SOLREF, LOCK_SOLIMP
                else:
                    m.jnt_solref[row['jid']], m.jnt_solimp[row['jid']] = self._saved_solver[row['joint']]
        if snaps:
            self._max_seat_snap = float(max(snaps + [self._max_seat_snap]))
        return snaps

    def _token_state(self, name):
        data = self.sim.data
        position = np.asarray(data.body_xpos[self._token_bodies[name]], float)
        dof = self._token_dofs[name]
        return position, float(np.linalg.norm(data.qvel[dof:dof + 3]))

    def _open_fraction(self, cid):
        """How far a locked compartment stands open, as a fraction of its *live* travel."""
        fraction = 0.
        for row in self._lock_joints[cid]:
            if row['travel'] > 0:
                fraction = max(fraction, min(1., abs(float(self.sim.data.qpos[row['qadr']]) - row['closed'])
                                             / row['travel']))
        return fraction

    def _token_for(self, cid):
        return next(n for n, t in sorted(self.tokens.items()) if t['compartment'] == cid)

    def _lock_observations(self):
        out = {}
        for cid in self.locked_ids:
            position, speed = self._token_state(self._token_for(cid))
            out[cid] = (token_on_pad(self.pads[cid], position, float(position[2]), speed),
                        is_closed(self._open_fraction(cid)))
        return out

    def _update_locks(self):
        if self._lock is None:
            return []
        events = self._lock.update(self._lock_observations())
        for kind, cid in events:
            if kind == 'relock':
                self._note_locked_away(cid)
            self.log_event(kind, cid, colour=self.pads[cid]['colour'])
        if events:
            self._apply_lock_state()
        return events

    def _note_locked_away(self, cid):
        """What a re-locking compartment shut in: the tokens and the target inside it at that moment."""
        shut = [name for name in sorted(self.tokens) if self._inside_compartment(name, cid)]
        target = self.task_spec['target_ids'][0]
        if self._inside_compartment(target, cid):
            shut.append(target)
        if shut:
            self._locked_away[cid] = sorted(shut)
        else:
            self._locked_away.pop(cid, None)

    def _reset_internal(self):
        self._locked_away = {}
        self._max_seat_snap = 0.
        super()._reset_internal()
        self._choose_open_token_spots()
        if self._lock is not None:
            self._lock = LockRule(self.locked_ids)
            self._apply_lock_state(force=True, seat=False)
        self.sim.forward()

    def _choose_open_token_spots(self):
        """Bind every token that starts in the open to a spot a policy camera sees from the start pose.

        The mirror image of search's ``_choose_open_spots``: these tokens must be *findable*, so the spots
        are ranked by camera coverage (G3 then checks them pixel-wise) and, among the seen ones, by distance
        from the robot. When no seen spot is left the nearest candidate is used and the miss is recorded in
        ``task_spec['open_tokens']``.
        """
        names = [n for n, t in sorted(self.tokens.items()) if t['where'] == 'open']
        if not names:
            return
        if self._open_token_choice:
            for choice in self._open_token_choice.values():
                self._set_object_pose(*choice)
            return
        frusta = camera_frusta(self.sim, CAMERAS, aspect=self._camera_size[0] / self._camera_size[1])
        half = np.asarray(TOKEN_SIZE_M, float) / 2
        pose_rng = np.random.default_rng([int(self.spec['pose_seed']), 20260923])
        base_xy, _ = self.base_pose()
        seen, unseen = [], []
        for spot in self._open_candidates:
            centre = np.r_[np.asarray(spot['xy'], float), float(spot['top_z']) + half[2]]
            (seen if seen_by_any(frusta, box_corners(centre, half)) else unseen).append(spot)
        seen.sort(key=lambda s: float(np.linalg.norm(np.subtract(s['xy'], base_xy))))
        unseen.sort(key=lambda s: float(np.linalg.norm(np.subtract(s['xy'], base_xy))))
        taken = []
        for name in names:
            ranked = [s for s in seen + unseen
                      if all(np.linalg.norm(np.subtract(s['xy'], t)) >= VISIBLE_SPACING_M for t in taken)]
            if not ranked:
                raise ValueError(f'{self.TASK_NAME}: no counter spot left for {name} on layout {self.layout_id}')
            spot = ranked[0]
            taken.append(spot['xy'])
            self._open_token_spots[name] = dict(spot, seen=any(spot is s for s in seen))
            self._open_token_choice[name] = (name, spot['xy'], float(spot['top_z']),
                                             float(pose_rng.uniform(-np.pi, np.pi)))
            self._set_object_pose(*self._open_token_choice[name])
            self.tokens[name]['spot'] = self._open_token_spots[name]
            self._counter_places[name] = dict(role='token', **counter_band(spot, self._records))
        self.task_spec['open_tokens'] = {n: dict(xy=[float(v) for v in s['xy']], top_z=float(s['top_z']),
                                                 seen_at_reset=bool(s['seen']))
                                         for n, s in self._open_token_spots.items()}
        self.task_spec['open_token_candidates'] = dict(seen=len(seen), total=len(self._open_candidates))

    def _post_action(self, action):
        self._update_locks()
        return super()._post_action(action)

    def target_compartments(self):
        return [self.chain_ids[-1]] if self.chain_ids else []

    # ---- protocol hooks (C1) -----------------------------------------------------
    def task_progress(self):
        """0.6 x the chain links unlocked at least once, + 0.2 target out of its compartment, + 0.2 on the tray."""
        snapshot = self._snapshot()
        score = terminal_score(self._score_spec(), snapshot)
        if score['success']:
            return 1.
        depth = max(1, len(self.chain_ids))
        unlocked = len([cid for cid in self.chain_ids if self._lock is not None and cid in self._lock.unlocked_ever])
        place = self.places['target_0']
        name = object_name(place['object'])
        state = snapshot['objects'].get(name, {})
        per = score['per_target'].get(name)
        on_tray = bool(per and all(per.values()) and not state.get('grasped') and not state.get('robot_contact'))
        out = bool(on_tray or self._out_of_hiding('target_0', place, snapshot))
        value = (PROGRESS_WEIGHTS['unlocked'] * unlocked / depth + PROGRESS_WEIGHTS['out'] * float(out)
                 + PROGRESS_WEIGHTS['tray'] * float(on_tray))
        return float(min(PROGRESS_CAP, value))

    def collateral_checks(self):
        """On top of search's ``autoclose_closed``: ``locked_away`` -- a compartment that re-locked with a token
        or the target inside and is still locked -- and ``token_on_floor``."""
        checks = super().collateral_checks()
        shut = set()
        if self._lock is not None:
            for cid, names in sorted(self._locked_away.items()):
                if self._lock.locked[cid]:
                    shut.update(n for n in names if self._inside_compartment(n, cid))
        dropped = sorted(name for name in self.tokens
                         if float(self.sim.data.body_xpos[self._token_bodies[name]][2]) < TOKEN_FLOOR_Z)
        checks['locked_away'] = dict(bad=bool(shut), object=','.join(sorted(shut)), cost=float(len(shut)))
        checks['token_on_floor'] = dict(bad=bool(dropped), object=','.join(dropped), cost=float(len(dropped)))
        return checks

    def task_snapshot(self):
        snapshot = super().task_snapshot()
        snapshot['locks'] = self._lock.summary() if self._lock is not None else None
        snapshot['locked_away'] = deepcopy(self._locked_away)
        snapshot['max_relock_seat_m'] = float(self._max_seat_snap)
        return snapshot

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['roboquest']['locked'] = dict(
            chain=list(self.chain_ids), dead_end=self.dead_end_id, double_door=bool(self.spec.get('double_door')),
            door_target=None if self.door_target is None else self.door_target['compartment'],
            colours={cid: self.pads[cid]['colour'] for cid in self.locked_ids},
            tokens={n: dict(opens=t['compartment'], where=t['where']) for n, t in sorted(self.tokens.items())},
            pad_refusals=len(self.pad_refusals),
            plates={cid: p.get('mode') for cid, p in sorted((self.plates or {}).items())})
        return meta

    # ---- lock verification --------------------------------------------------------
    def lock_excursion(self, cid, force_n=LOCK_LOAD_N, torque_nm=LOCK_LOAD_NM, ticks=LOCK_LOAD_TICKS):
        """Peak joint excursion of a *locked* compartment under a pull (drawer) or a torque (door).

        Applies ``force_n`` along the compartment's outward normal to the moving body (drawer) or
        ``torque_nm`` about the vertical hinge axis (door) for ``ticks`` control steps and returns the largest
        ``|qpos - closed|`` seen. Verification only; nothing calls it during an episode.
        """
        m, data = self.sim.model, self.sim.data
        comp = self.compartments[cid]
        rows = self._lock_joints[cid]
        element = self._moving_body(cid)
        name = element.get('name') if element is not None else None
        try:
            body = int(m.body_name2id(name))
        except Exception:      # noqa: BLE001
            return dict(compartment=cid, error=f'no compiled body for the moving element {name!r}')
        wrench = np.zeros(6)
        if comp['kind'] == 'drawer':
            normal = np.asarray(comp['front_normal'], float).ravel()   # the records carry it as xy
            wrench[:normal.size] = normal * float(force_n)
        else:
            wrench[5] = float(torque_nm)
        peak, zero = 0., np.zeros(self.action_dim)
        saved = data.xfrc_applied[body].copy()
        try:
            for _ in range(int(ticks)):
                data.xfrc_applied[body] = wrench
                self.step(zero)
                for row in rows:
                    peak = max(peak, abs(float(data.qpos[row['qadr']]) - row['closed']))
        finally:
            data.xfrc_applied[body] = saved
        return dict(compartment=cid, kind=comp['kind'], body=name,
                    load=float(force_n) if comp['kind'] == 'drawer' else float(torque_nm),
                    unit='N' if comp['kind'] == 'drawer' else 'N.m', ticks=int(ticks),
                    seconds=round(float(ticks) / float(self.control_freq), 3), peak_excursion=float(peak),
                    locked=bool(self._lock.locked[cid]), range_width=float(self._range_width(cid)))

    def _range_width(self, cid):
        m = self.sim.model
        return max(float(m.jnt_range[row['jid']][1] - m.jnt_range[row['jid']][0])
                   for row in self._lock_joints[cid])

    # ---- teleport certificate (C3) --------------------------------------------------
    def teleport_solution(self):
        """Walk the chain by state edits alone: token onto its pad, the range opens, the compartment opens,
        the next token, ... and the target onto the tray.

        Also demonstrates the re-lock: a mid-chain compartment whose token leaves its pad and which is then
        closed locks again. Never moves the robot, never presses Submit, never raises (contract C3).
        """
        steps = []
        started = time.monotonic()

        def record(step, ok, detail=None):
            steps.append(dict(step=step, ok=bool(ok), detail=detail))
            return ok

        def onto_pad(cid, name):
            pad = self.pads[cid]
            self._set_object_pose(name, pad['centre'][:2], pad['top_z'], 0.)
            self.sim.forward()
            for _ in range(3):
                self.step(zero)
            return record(f'token {name} onto pad {cid}', not self._lock.locked[cid],
                          dict(colour=pad['colour'], range_width=round(self._range_width(cid), 6),
                               on_pad=bool(self._lock.on_pad[cid])))

        def set_joints(cid, opened):
            values = {}
            for row in self._lock_joints[cid]:
                low, high = self._saved_ranges[row['joint']]
                value = (high if abs(high) >= abs(low) else low) if opened else row['closed']
                self.sim.data.set_joint_qpos(row['joint'], value)
                self.sim.data.set_joint_qvel(row['joint'], 0.)
                values[row['joint']] = float(value)
            self.sim.forward()
            return values
        zero = np.zeros(self.action_dim)
        opened = []
        try:
            for cid in self.chain_ids:
                if not onto_pad(cid, self._token_for(cid)):
                    return steps
                values = set_joints(cid, True)
                opened.append(cid)
                record(f'open {cid}', True, dict(joints=values, kind=self.compartments[cid]['kind'],
                                                 closer=cid in self._closer_ids))
            if self.dead_end_id is not None:
                onto_pad(self.dead_end_id, self._token_for(self.dead_end_id))
            if len(self.chain_ids) >= 2:
                # The re-lock: take the first link's token off its pad and shut that compartment again.
                cid = self.chain_ids[0]
                name = self._token_for(cid)
                spot = self.tokens[name].get('spot')
                pad = self.pads[cid]
                if spot is not None:
                    self._set_object_pose(name, spot['xy'], float(spot['top_z']), 0.)
                else:
                    self._set_object_pose(name, [pad['centre'][0], pad['centre'][1] + .40], pad['centre'][2], 0.)
                set_joints(cid, False)
                for _ in range(3):
                    self.step(zero)
                record(f're-lock {cid} once its token leaves the pad', bool(self._lock.locked[cid]),
                       dict(range_width=round(self._range_width(cid), 6), on_pad=bool(self._lock.on_pad[cid]),
                            relocks=int(self._lock.relocks[cid]),
                            note='a mid-chain compartment closed without its token locks again'))
                opened = [c for c in opened if c != cid]
            target = object_name(self.spec['targets'][0]['object'])
            xy = [float(v) for v in self.tray['centre_xy']]
            self._set_object_pose(target, xy, self.tray['top_z'], float(self.work['yaw']))
            self.sim.forward()
            record(f'place {target} on the tray', True, dict(xy=xy))
            for cid in opened:
                set_joints(cid, False)
            if opened:
                record('reclose the compartments left open', True,
                       'an open drawer under the work top would stand inside the robot base while settling')
        except Exception as exc:  # noqa: BLE001
            record('chain', False, f'{type(exc).__name__}: {exc}')
        try:
            for _ in range(SETTLE_TICKS):
                self.step(zero)
            score = self._current_score()
            record('settle', score['success'], dict(ticks=SETTLE_TICKS, errors=score['errors'], score=score['score'],
                                                    progress=self.task_progress(),
                                                    locks=self._lock.summary() if self._lock else None,
                                                    seconds=round(time.monotonic() - started, 2)))
        except Exception as exc:  # noqa: BLE001
            record('settle', False, f'{type(exc).__name__}: {exc}')
        return steps

    # ---- gates -----------------------------------------------------------------------
    def hidden_state_check(self):
        """Gate G3: search's pixel check over the hidden set, plus the open tokens, the pads and the locks.

        The hidden set is the target, the hidden distractors and every token inside a compartment (the base
        check, which reads ``task_spec['hidden_object_ids']``). On top of it G3 requires that

        * token 1 changes pixels in at least one policy camera, so the chain has a findable first step
          (``_choose_open_token_spots`` ranks the spots for exactly this, token 1 first);
        * every locked compartment is locked at reset;
        * the placement rules hold: every place that holds something is a drawer with its floor
          at 0.40 m or above, or the HingeCabinet of a double-door instance with the target exactly the spec's
          ``target_depth_m`` (at most 0.20 m) behind the door line, no enabled compartment is a bottom
          drawer or a SingleCabinet, and every task object on a counter (open tokens, pads, visible objects)
          keeps to the front two thirds of a wall-backed counter's depth (rule 3);
        * every reader pad is *sited* on its compartment: on an approachable counter edge with a base
          standoff, within :data:`~...locked.structure.PAD_LATERAL_MAX_M` of the compartment's front
          centre and on its own side of the fixture line, and no two pads within ``PAD_SPACING_M``.

        Pad visibility **at reset** is measured and reported (``pads_visible``, ``pads_seen_at_reset``) but
        is not required. The contract asked for it; it cannot hold and keep the task a search. A pad sits at
        its own compartment, which the sampler may put anywhere in the kitchen, while the robot starts facing
        the work frame: on the 24-instance trial registry only 1 instance had every pad in a policy camera at
        reset (7 of 33 pads over the 20 that built). The discoverability the rule was after is the *local*
        one -- the robot that has driven to a compartment and sees its coloured lock plate sees the
        same-coloured pad beside it -- and that is what the siting rule guarantees by construction.
        """
        report = super().hidden_state_check()
        m = self.sim.model
        width, height = self._camera_size
        base = report['images']

        def visible(gids):
            gids = np.asarray([g for g in gids if m.geom_rgba[g, 3] > 0.], int)
            if not gids.size:
                return {cam: False for cam in CAMERAS}, {cam: 0 for cam in CAMERAS}
            saved = m.geom_rgba[gids].copy()
            m.geom_rgba[gids, 3] = 0.
            try:
                without = {cam: np.array(self.sim.render(width=width, height=height, camera_name=cam), copy=True)
                           for cam in CAMERAS}
            finally:
                m.geom_rgba[gids] = saved
            changed = {cam: int(np.any(base[cam] != without[cam], axis=-1).sum()) for cam in CAMERAS}
            return {cam: bool(changed[cam] > 0) for cam in CAMERAS}, changed

        tokens = {}
        for name, token in sorted(self.tokens.items()):
            if token['where'] != 'open':
                continue
            seen, pixels = visible(self._object_geoms.get(name, []))
            tokens[name] = dict(colour=token['colour'], seen=seen, pixels=pixels, visible=any(seen.values()),
                                dead_end=bool(token['dead_end']))
        # The contract asks for *token 1* at reset: the chain has to have a findable first step. A dead-end
        # token that also starts in the open is a decoy, and a decoy the robot has to come across while
        # searching is a better decoy than one laid out in front of it, so its visibility is only recorded.
        first = token_name(1)
        tokens_ok = bool(tokens.get(first, {}).get('visible'))
        pads, pads_ok = {}, True
        for cid in self.locked_ids:
            seen, pixels = visible(self._pad_geoms[cid])
            pads[cid] = dict(colour=self.pads[cid]['colour'], seen=seen, pixels=pixels, visible=any(seen.values()))
            pads_ok = pads_ok and pads[cid]['visible']
        locked = {cid: bool(self._lock.locked[cid]) for cid in self.locked_ids}
        siting, siting_ok = self._pad_siting()
        placement = self._placement_report()
        report.update(placement=placement, placement_ok=placement['ok'],
                      open_tokens=tokens, first_token_visible=tokens_ok, pads=pads, pads_visible=pads_ok,
                      open_tokens_visible=all(t['visible'] for t in tokens.values()),
                      pads_seen_at_reset=sum(1 for p in pads.values() if p['visible']),
                      pad_siting=siting, pads_sited=siting_ok,
                      locked_at_reset=locked, lock_range_width={cid: round(self._range_width(cid), 6)
                                                                for cid in self.locked_ids},
                      hidden_tokens=[n for n, t in sorted(self.tokens.items()) if t['where'] != 'open'])
        report['passed'] = bool(report['passed'] and tokens_ok and siting_ok and all(locked.values())
                                and placement['ok'])
        # The renders are megabytes of uint8 and the gate batch stores whatever comes back: without this the
        # row is a 4000-character repr of the pixel arrays and every flag above is truncated away.
        report.pop('images', None)
        return report

    def _pad_siting(self):
        """Is every reader pad on its compartment's counter edge, and are the pads apart? (G3, above.)"""
        rows, ok = {}, True
        centres = {cid: np.asarray(self.pads[cid]['centre'][:2], float) for cid in self.locked_ids}
        for cid in self.locked_ids:
            pad = self.pads[cid]
            lateral, forward = pad_offsets(self.compartments[cid], centres[cid])
            nearest = min([float(np.linalg.norm(centres[cid] - centres[other]))
                           for other in self.locked_ids if other != cid] or [float('inf')])
            row = dict(colour=pad['colour'], lateral_m=round(lateral, 4), forward_m=round(forward, 4),
                       standoff=pad['standoff'] is not None,
                       nearest_pad_m=None if nearest == float('inf') else round(nearest, 4))
            row['ok'] = bool(abs(lateral) <= PAD_LATERAL_MAX_M and PAD_FRONT_MIN_M <= forward
                             <= PAD_FRONT_MAX_M and row['standoff'] and nearest >= PAD_SPACING_M)
            rows[cid] = row
            ok = ok and row['ok']
        return rows, ok

    def disabled_fixture_report(self):
        report = super().disabled_fixture_report()
        report.update(locked_compartments=list(self.locked_ids),
                      excluded_by_height_rule=deepcopy(self._excluded),
                      lock_range_width={cid: round(self._range_width(cid), 6) for cid in self.locked_ids},
                      pad_refusals=deepcopy(self.pad_refusals),
                      plate_modes={cid: p.get('mode') for cid, p in sorted((self.plates or {}).items())})
        return report
