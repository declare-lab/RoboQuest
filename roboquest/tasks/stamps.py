"""Stamp Composition (tasks-v0 task 7) as a RoboQuest kitchen task.

Round self-inking stamps stand face down, scattered over the free front and side
area of a RoboCasa work surface around an unlined scratch board, a reference card
showing a 3x3 target and the 3x3 final board (each drawn at a small jitter about
its functional place, so nothing is aligned). Each stamp prints a fixed pattern
of dots at whatever position and yaw it is actually held at; the die is inside
the unmarked round housing, so neither the pattern nor the stamp's orientation
can be seen. The target is the
union of ``composition`` of the stamps at multiples of 90 degrees, and the
generator keeps only instances where that combination is the *only* one that
prints it. The episode ends at the first physical Submit press.

Ink is a declared material model, ported unchanged from ``stamps_v2``: an actual
flat sole-to-board contact transfers circular dots at the real relative x/y/yaw
into that board's bitmap, which is uploaded into the board's texture. No object
is snapped, attached or moved by the task, and nothing about the score triggers
ink.
"""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboquest import observe, scatter as S
from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz
from roboquest.stamps import geometry, hiding, rules

CONTRACT_VERSION = 'stamps-composition-kitchen-v3'
PHYSICS_TIMESTEP = .001       # small light parts on thin boards; stamps_v2's tuned value
BUDGET_TICKS = 12000          # provisional (protocol C1): no v0 HORIZON, so the shared default
MAX_SAMPLER_ATTEMPTS = 400    # die sets, not (subset, rotation) draws: see ``sample_spec``
YAW_JITTER = math.pi / 2      # the round housing hides the yaw, so it is drawn over +/- 90 degrees
COMPOSITION_LEVELS = (2, 3, 4)
RELEASE_SETTLE_TICKS = 25     # C3: let the stamps land at home before the score reads them
SCRATCH_CELL_LEVELS = (5, 4)  # nuisance in v1: the scratch board's workspace, drawn from `structure`
NOUN = 'stamps'
# Spec 1.3: the scope sentence (with the count in a random half of the instances,
# drawn by the registry's `wording` stream), the v0 middle wording unchanged, then
# the clutter sentence. The wording is identical at every observability level.
GOAL_SCOPE = 'There are {count}{noun} on the counters in this kitchen.'
GOAL_MIDDLE = (
    'Stamp the final board so it shows exactly the pattern on the reference card, then press Submit. '
    'The final board is the blue-bordered one with the printed 3x3 grid; the green-bordered reference '
    'card is not for stamping. The round self-inking stamps stand face down on the work surface. Each '
    'prints a fixed pattern of dots in whatever direction it is held, and the unmarked housings show '
    'neither the pattern nor which way it points, so try them on the tan scratch board. Ink stays where '
    'it lands, on the scratch board and on the final board alike. Fill exactly the reference card\'s grid '
    'cells with no ink outside the final board\'s grid, put every stamp back down on the counter and let '
    'go, then press the red SUBMIT button. Your first submission ends the episode; a wrong submission or '
    'no submission fails.')
GOAL_CLUTTER = 'Other objects on the counter are not part of the task; you may move them if needed.'
# Progress (spec 1.4) is 0.9 * the dot-rule pattern + 0.1 * the finish term,
# but contract C1 makes 1.0 mean the success predicate holds;
# a board that reads 1 with stray ink past the raster tolerance reports this instead.
ALMOST = .99


def stamp_count_for(composition):
    """Stamps offered at a composition level: one more than the target needs (spec 2.1)."""
    return int(composition) + 1


def _split_key(target_id, composition):
    return f't{int(target_id)}-k{int(composition)}'


def _holdout_keys():
    """The compositional evaluation hold-out (see ``Stamps.is_holdout``)."""
    return frozenset(_split_key(target_id, composition)
                     for target_id in range(rules.TARGET_CLASS_COUNT)
                     for composition in COMPOSITION_LEVELS
                     if Stamps.is_holdout(_split_key(target_id, composition)))


class Stamps(RoboQuestKitchen):
    TASK_NAME = 'stamps'
    CONTRACT_VERSION = CONTRACT_VERSION
    BUDGET_TICKS = BUDGET_TICKS
    # v2: `composition` is the single reported size factor, `stamp_count` is derived
    # and `scratch_cells` became nuisance; the target is chosen by enumeration
    # instead of rejection sampling and the boards carry a paper-stock colour.
    # v3 (wave 2): the universal `observability` axis, realised in the scene.
    GENERATOR_VERSION = 'v3'
    FACTORS = {'composition': COMPOSITION_LEVELS, 'observability': hiding.LEVELS}
    DEV_ONLY_LEVELS = hiding.DEV_ONLY_LEVELS
    FOOTPRINT = geometry.FOOTPRINT

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        count = f"{stamp_count_for(spec['composition'])} " if spec.get('show_count', False) else ''
        return ' '.join((GOAL_SCOPE.format(count=count, noun=NOUN), GOAL_MIDDLE, GOAL_CLUTTER))

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        composition = int(factors['composition'])
        if composition not in COMPOSITION_LEVELS:
            raise ValueError(f'composition {composition} is not a level of this task')
        stamp_count = stamp_count_for(composition)
        # Nuisance (v1): how much clear workspace the scratch board offers, drawn
        # from the structure stream and recorded, never a factor and never named.
        scratch_cells = int(structure.choice(SCRATCH_CELL_LEVELS))
        # Distinct dies for the stamps, then a target that exactly one (subset,
        # rotations) combination prints (gate G0, re-checked by cpu_gate). The
        # acceptance rule is v0's; the search is not. Drawing (subset, rotations)
        # at random and testing it accepts 38 % of draws at composition 2, 3.5 %
        # at composition 3 and 0.005 % at composition 4 (measured over 120 000
        # draws), so the largest level would never mint. ``rules.unique_targets``
        # enumerates every impression of the die set once and returns the targets
        # only one combination reaches; the draw is then uniform over exactly the
        # combinations rejection sampling would have accepted for that die set.
        attempts, mask_ids, chosen, turns, target = 0, None, None, None, None
        while attempts < MAX_SAMPLER_ATTEMPTS and target is None:
            attempts += 1
            mask_ids = [int(v) for v in structure.choice(rules.BANK_SIZE, size=stamp_count, replace=False)]
            masks = [rules.bank_mask(i) for i in mask_ids]
            options = rules.unique_targets(masks, composition)
            if not options:
                continue                        # this die set prints nothing uniquely; draw another
            bits, solution = options[int(structure.integers(0, len(options)))]
            chosen = [int(index) for index, _ in solution]
            turns = [int(k) for _, k in solution]
            target = rules.bits_mask(bits)
        unique = target is not None
        if not unique:                          # recorded, then rejected by validate_spec
            mask_ids = mask_ids or [int(v) for v in structure.choice(rules.BANK_SIZE, size=stamp_count,
                                                                     replace=False)]
            chosen, turns = list(range(composition)), [0] * composition
            target = rules.union_of([rules.bank_mask(mask_ids[c]) for c in chosen], turns)
        palette = materials.permutation(len(geometry.HANDLE_PALETTE))[:stamp_count]
        sole_rgba = geometry.SOLE_PALETTE[int(materials.integers(0, len(geometry.SOLE_PALETTE)))]
        paper_rgb = {name: [int(v) for v in rules.PAPER_PALETTE[int(materials.integers(0, len(rules.PAPER_PALETTE)))]]
                     for name in geometry.BOARD_ORDER}
        # Nuisance: the placement. The boards jitter about their functional centres, then
        # the stamps are scattered over the free front and side area around them; a draw
        # that would rest a stamp on a board (which would print) or on Submit is rejected.
        # The stamp's yaw is hidden state and keeps its own +/- 90 degree sampling.
        frame = geometry.frame_rect()
        submit = geometry.submit_shape()
        halves = [geometry.board_outer_half(name, scratch_cells) for name in geometry.BOARD_ORDER]
        boards = S.scatter(poses, len(geometry.BOARD_ORDER), geometry.board_slots(scratch_cells),
                           halves, geometry.BOARD_GAP, keepouts=[submit], keepout_gap=geometry.BOARD_GAP,
                           yaw_range=(-geometry.BOARD_JITTER_YAW, geometry.BOARD_JITTER_YAW),
                           region_holds='center', bounds=frame)
        placement = {name: [x, y] for name, (x, y, _) in zip(geometry.BOARD_ORDER, boards)}
        board_yaw = {name: yaw for name, (_, _, yaw) in zip(geometry.BOARD_ORDER, boards)}
        keepouts = [S.shape_at(half, (x, y), yaw) for half, (x, y, yaw) in zip(halves, boards)]
        stamps = S.scatter(poses, stamp_count, geometry.stamp_region(), geometry.SOLE_RADIUS,
                           geometry.STAMP_GAP, keepouts=keepouts + [submit],
                           keepout_gap=geometry.STAMP_KEEPOUT_GAP, yaw_range=(-YAW_JITTER, YAW_JITTER))
        # Observability (spec 2.2): the level is the second reported factor; which stamps are
        # hidden, which board is covered and which uncover flavour is used are nuisance drawn
        # from the same poses stream, last, so the structure draws, the hold-out key and the
        # goal text are bit-identical to the same seed at another level (`visible` draws none).
        # The draw sees the placement it must fit into, so a cover candidate that the scene
        # geometry cannot hold is dropped before the build instead of costing a seed at G1.
        level = str(factors.get('observability', 'visible'))
        plan = geometry.board_plan(scratch_cells)
        shapes = {f'stamp_{index}': S.Disc((x, y), geometry.SOLE_RADIUS)
                  for index, (x, y, _) in enumerate(stamps)}
        shapes.update(dict(zip(geometry.BOARD_ORDER, keepouts)), submit=submit)
        cover_places = {name: dict(xy=list(placement[name]), half=list(plan[name]['half']),
                                   yaw=float(board_yaw[name])) for name in geometry.BOARD_ORDER}

        def rescatter(name):
            """Draw the stamps again, reserving the cloche's own footprint around ``name``."""
            try:
                drawn = S.scatter(poses, stamp_count, hiding.stamp_regions(stamp_count, name),
                                  hiding.stamp_footprints(stamp_count, name), geometry.STAMP_GAP,
                                  keepouts=keepouts + [submit], keepout_gap=geometry.STAMP_KEEPOUT_GAP,
                                  yaw_range=(-YAW_JITTER, YAW_JITTER))
            except ValueError:
                return None
            stamps[:] = drawn
            return {f'stamp_{index}': (x, y) for index, (x, y, _) in enumerate(drawn)}

        hidden, hidden_plan = hiding.draw(poses, level, [f'stamp_{index}' for index in range(stamp_count)],
                                          [f'stamp_{int(index)}' for index in chosen], shapes=shapes,
                                          places=cover_places, rescatter=rescatter, bounds=frame)
        return dict(
            observability=level, hidden=hidden, hidden_plan=hidden_plan,
            stamp_count=stamp_count, composition=composition, scratch_cells=scratch_cells,
            mask_ids=mask_ids, target=rules.mask_array(rules.mask_cells(target)).astype(int).tolist(),
            target_bits=rules.mask_bits(target), target_id=rules.target_class_id(target),
            solution=[[int(c), int(k)] for c, k in zip(chosen, turns)],
            unique=bool(unique), sampler_attempts=int(attempts),
            initial_yaw_rad=[yaw for _, _, yaw in stamps],
            stamp_xy=[[x, y] for x, y, _ in stamps],
            handle_rgba=[list(geometry.HANDLE_PALETTE[int(i)]) for i in palette],
            sole_rgba=list(sole_rgba), paper_rgb=paper_rgb,
            placement=dict(submit=list(geometry.SUBMIT_XY), **placement), board_yaw=board_yaw)

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        stamp_count, composition = int(spec['stamp_count']), int(spec['composition'])
        if composition not in COMPOSITION_LEVELS:
            problems.append(f'composition {composition} is not a level of this task')
        if stamp_count != stamp_count_for(composition):
            problems.append(f'stamp_count {stamp_count} is not composition + 1')
        if int(spec.get('scratch_cells', 0)) not in SCRATCH_CELL_LEVELS:
            problems.append(f'scratch_cells {spec.get("scratch_cells")} is not one of {SCRATCH_CELL_LEVELS}')
        if not spec.get('unique'):
            problems.append(f'no unique composition found in {spec.get("sampler_attempts")} die sets')
        if len(set(spec['mask_ids'])) != stamp_count:
            problems.append('stamps must carry distinct dies')
        if len(spec['initial_yaw_rad']) != stamp_count or len(spec['stamp_xy']) != stamp_count:
            problems.append('per-stamp nuisance draws do not match stamp_count')
        if len(spec['solution']) != composition:
            problems.append('the recorded solution does not use `composition` stamps')
        if len({s for s, _ in spec['solution']}) != composition:
            problems.append('the recorded solution reuses a stamp')
        masks = [rules.bank_mask(i) for i in spec['mask_ids']]
        for index, mask in enumerate(masks):
            if not rules.MIN_DOTS <= int(mask.sum()) <= rules.MAX_DOTS:
                problems.append(f'die {index} has {int(mask.sum())} dots')
        target = np.asarray(spec['target'], bool)
        union = rules.union_of([masks[s] for s, _ in spec['solution']], [k for _, k in spec['solution']])
        if not np.array_equal(union, target):
            problems.append('the recorded solution does not print the target')
        if rules.mask_bits(target) != int(spec['target_bits']):
            problems.append('target_bits disagrees with the target')
        if rules.target_class_id(target) != int(spec['target_id']):
            problems.append('target_id disagrees with the target')
        if sorted(spec.get('board_yaw', {})) != sorted(geometry.BOARD_ORDER):
            problems.append('board_yaw does not carry one yaw per board')
        elif any(abs(float(yaw)) > geometry.BOARD_JITTER_YAW + 1e-9
                 for yaw in spec['board_yaw'].values()):
            problems.append('a board yaw is outside the jitter range')
        for name in geometry.BOARD_ORDER:
            nominal = geometry.BOARD_NOMINAL_XY[name]
            xy = spec['placement'].get(name)
            if xy is None or max(abs(xy[index] - nominal[index]) for index in (0, 1)) > \
                    geometry.BOARD_JITTER_XY + 1e-9:
                problems.append(f'{name} is outside its jitter box')
        if list(spec['placement']['submit']) != list(geometry.SUBMIT_XY):
            problems.append('the Submit button has moved')
        if any(abs(float(yaw)) > YAW_JITTER + 1e-9 for yaw in spec['initial_yaw_rad']):
            problems.append('a stamp yaw is outside the +/- 90 degree range')
        problems.extend(geometry.placement_problems(spec, cls.FOOTPRINT)[0])
        problems.extend(cls._observability_problems(spec))
        return problems

    @classmethod
    def _observability_problems(cls, spec):
        """Spec 2.2 schema: a known level, entries observe can realise, and a candidate plan
        whose fallbacks name real stamps and boards."""
        level = spec.get('observability', 'visible')
        if level not in hiding.ALL_LEVELS:
            return [f'observability {level!r} is not one of {hiding.ALL_LEVELS}']
        stamps = [f'stamp_{index}' for index in range(int(spec['stamp_count']))]
        problems = list(observe.validate_hidden(spec, objects=set(stamps) | set(geometry.BOARD_ORDER)))
        plan = spec.get('hidden_plan') or {}
        flavour = plan.get('uncover_flavour')
        wants_cover = level in ('uncover', hiding.DEV_ONLY_LEVEL)
        if wants_cover and flavour not in ('object', 'place'):
            problems.append(f'uncover_flavour {flavour!r} is neither object nor place')
        if not wants_cover and flavour is not None:
            problems.append(f'level {level!r} draws no uncover flavour')
        for name, kind in plan.get('object_candidates') or []:
            if name not in stamps or kind not in observe.OBJECT_COVERS:
                problems.append(f'object candidate {name}/{kind} is not a stamp under an object cover')
        for name, kind in plan.get('place_candidates') or []:
            if name not in geometry.BOARD_ORDER or kind not in observe.PLACE_COVERS:
                problems.append(f'place candidate {name}/{kind} is not a board under a flat cover')
        if [name for name, _ in plan.get('look') or []] != [e['object'] for e in spec.get('hidden', [])
                                                            if e['mode'] in hiding.LOOK_MODES]:
            problems.append('hidden_plan.look disagrees with the look entries')
        rescatter = plan.get('rescatter')
        if rescatter is not None:
            name, moved = rescatter
            if name not in stamps:
                problems.append(f'the reserved stamp {name!r} is not a stamp')
            if sorted(moved) != sorted(stamps):
                problems.append('hidden_plan.rescatter does not cover every stamp')
            elif any(not np.allclose(spec['stamp_xy'][int(key.split('_')[1])], xy, atol=1e-9)
                     for key, xy in moved.items()):
                problems.append('the stamps were re-scattered for the cloche but stamp_xy did not follow')
        if wants_cover and not (plan.get('object_candidates') or plan.get('place_candidates')):
            problems.append('no uncover candidate survives the placement')
        if wants_cover and flavour in ('object', 'place'):
            candidates = [tuple(candidate) for candidate in plan.get(f'{flavour}_candidates') or []]
            entry = next((e for e in spec.get('hidden', []) if e['mode'] == 'cover'), None)
            if not candidates:
                problems.append(f'uncover_flavour {flavour!r} has no candidate of its own')
            elif entry is not None and (entry['object'], entry['cover']) != candidates[0]:
                problems.append('the cover entry is not the first candidate of the drawn flavour')
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """G0: exactly one (subset of stamps, rotations) prints the target, each stamp used at most once.

        ``rules.solutions`` searches every subset size, so a single solution also
        means no smaller subset and no larger one prints it.
        """
        masks = [rules.bank_mask(i) for i in spec['mask_ids']]
        target = np.asarray(spec['target'], bool)
        found = rules.solutions(masks, target)
        recorded = tuple((int(s), int(k)) for s, k in spec['solution'])
        # Symmetric dies would make two quarter-turns interchangeable; the bank has none.
        matches_recorded = any(sorted(option) == sorted(recorded) for option in found)
        geometry_problems, placement = geometry.placement_problems(spec, cls.FOOTPRINT)
        report = dict(solution_count=len(found), solutions=[[list(pair) for pair in option] for option in found],
                      target_dots=int(target.sum()), die_dots=[int(m.sum()) for m in masks],
                      target_id=int(spec['target_id']), sampler_attempts=int(spec.get('sampler_attempts', 0)),
                      recorded_solution_reachable=bool(matches_recorded),
                      placement=placement, problems=geometry_problems)
        return bool(len(found) == 1 and matches_recorded and not geometry_problems), report

    @classmethod
    def split_key(cls, spec):
        """Structure class: the target's rotation class and how many stamps compose it.

        ``stamp_count`` is no longer in the key because it is ``composition + 1``;
        the target's *rotation* class is used so that two instances whose
        reference cards differ only by a turn are one class.
        """
        return _split_key(spec['target_id'], spec['composition'])

    @classmethod
    def is_holdout(cls, split_key):
        """Evaluation hold-out, the v0 pattern re-derived at the new levels: a third of
        the target bank (every class id divisible by three) at every composition, and a
        second third (``target_id % 3 == 1``) at the largest cell, composition 4.

        The largest cell is therefore two-thirds evaluation-only, which is the v0
        intent ("the largest cell plus a third of the bank"), while the remaining
        third keeps composition 4 present in development: contract C2 requires every
        level of every factor to appear there, and ``registry.mint_grid`` would
        otherwise walk 400 seeds for an empty development cell.
        """
        target_id, composition = (int(part[1:]) for part in str(split_key).split('-'))
        return bool(target_id % 3 == 0 or (composition == max(COMPOSITION_LEVELS) and target_id % 3 == 1))

    # ---- scene protocol -----------------------------------------------------
    def build_task(self):
        self.place_work_frame()
        spec = self.spec
        world, assets = self.model.worldbody, self.model.asset
        work_yaw = self.work['yaw']
        top_z = self.work['top_z']
        self.boards, self.stamps, self.stamp_names = {}, {}, []
        board_yaw = spec.get('board_yaw') or {}
        for name, plan in geometry.board_plan(spec['scratch_cells']).items():
            xy = spec['placement'][name]
            own_yaw = float(board_yaw.get(name, 0.))
            body = self.add_frame_body(f'board_{name}', xy)
            # The board carries its own small yaw on top of the work surface's, so the
            # three boards are not parallel; every ink and scoring transform reads
            # ``yaw_rad`` below, so the jitter is exact rather than approximated.
            body.set('quat', geometry.vec(yaw_quat_wxyz(work_yaw + own_yaw)))
            built = geometry.add_board(body, assets, f'paper_{name}', plan['half'],
                                       geometry.BORDER_RGBA[name], plan['resolution'])
            center = self.frame_to_world([xy[0], xy[1], geometry.PAPER_THICKNESS])
            self.boards[name] = dict(built, name=name, body_name=f'board_{name}', frame_xy=list(xy),
                                     position=center.tolist(), yaw_rad=float(work_yaw + own_yaw),
                                     frame_yaw_rad=own_yaw,
                                     half_size=[float(plan['half'][0]), float(plan['half'][1])],
                                     shape=[int(plan['resolution']), int(plan['resolution'])],
                                     paper_rgb=[int(v) for v in (spec.get('paper_rgb') or {}).get(
                                         name, (rules.PAPER_RGB,) * 3)],
                                     inkable=bool(plan['inkable']), gridded=bool(plan['gridded']))
        for index in range(int(spec['stamp_count'])):
            name = f'stamp_{index}'
            xy = spec['stamp_xy'][index]
            yaw = work_yaw + float(spec['initial_yaw_rad'][index])
            position = self.frame_to_world([xy[0], xy[1], .001])
            body = ET.SubElement(world, 'body', name=name, pos=geometry.vec(position),
                                 quat=geometry.vec(yaw_quat_wxyz(yaw)))
            ET.SubElement(body, 'freejoint', name=f'{name}_joint')
            mask = rules.bank_mask(spec['mask_ids'][index])
            built = geometry.add_stamp(body, name, mask, spec['handle_rgba'][index],
                                       sole_rgba=spec.get('sole_rgba') or geometry.SOLE_RGBA)
            self.stamps[name] = dict(built, name=name, body_name=name, joint_name=f'{name}_joint',
                                     frame_xy=list(xy), source_position=position.tolist(),
                                     initial_yaw_rad=float(yaw), grasp_offset_local=list(geometry.GRASP_OFFSET),
                                     die_mask_private=mask.astype(int).tolist(),
                                     die_id_private=int(spec['mask_ids'][index]))
            self.stamp_names.append(name)
        self.add_submit_button(spec['placement']['submit'])
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL,
            frame_origin_world=self.work['center_world'], frame_yaw=work_yaw, work_top_z=top_z,
            stamp_order=list(self.stamp_names), stamps=deepcopy(self.stamps), boards=deepcopy(self.boards),
            target_cells=deepcopy(spec['target']), target_id=int(spec['target_id']),
            composition=int(spec['composition']), scratch_cells=int(spec['scratch_cells']),
            stamp_count=int(spec['stamp_count']), budget_ticks=int(BUDGET_TICKS),
            solution_private=deepcopy(spec['solution']),
            physics_timestep=PHYSICS_TIMESTEP,
            ink_model='Flat native sole-board contact transfers circular ink dots at the actual relative '
                      'x/y/yaw; a printed pair must separate before it prints again.',
            placement='Stamps scattered over the free front and side area at yaws over +/- 90 degrees; the '
                      'three boards jittered about their functional centres (+/-2 cm, +/-5 deg); nothing '
                      'aligned, nothing overlapping (roboquest.scatter).',
            scoring='First physical Submit freezes the score: final-board cells equal the reference target, '
                    'at most 35 off-grid ink pixels, and every stamp released, still and resting on the work '
                    'surface; no submission fails.')
        self.asset_evidence.update(
            stamps='Purpose-made round rigid stamps (sole disc, handle) with the die enclosed inside the '
                   'sole; the housing colour is a per-instance draw and the handle colours a permutation, '
                   'both independent of the dies',
            boards='Purpose-made bordered paper boards whose printed faces are textured planes; the paper '
                   'stock colour is a per-instance draw, the borders (blue final, green reference, tan '
                   'scratch), the grid and the ink are fixed functional colours',
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged')
        self.material_effect_model = dict(
            name='contact-triggered-self-inking-kitchen-v1',
            trigger='Actual native sole-board contact with upward normal > 0.9, positive normal force, '
                    'sole-bottom height within 3 mm of the board and sole tilt below 8 degrees',
            footprint='All die dots transformed by the actual stamp-to-board x/y/yaw at first contact',
            rearm='The contact pair must separate before another impression',
            mutations=['Per-board ink bitmap', 'Board texture tex_data', 'GPU texture upload'],
            excluded=['No object pose setters', 'No attachment', 'No score-triggered ink'])
        self._apply_observability()

    # ---- observability (spec 2.2, contract C6) ------------------------------
    def _prop_shapes(self):
        """Task-frame footprint of every prop this task owns, by name: what a cover, an occluder
        or a relocated stamp must keep clear of."""
        cells = int(self.spec['scratch_cells'])
        shapes = {name: S.shape_at(geometry.board_outer_half(name, cells), board['frame_xy'],
                                   board['frame_yaw_rad'])
                  for name, board in self.boards.items()}
        shapes.update({name: S.Disc((float(stamp['frame_xy'][0]), float(stamp['frame_xy'][1])),
                                    geometry.SOLE_RADIUS)
                       for name, stamp in self.stamps.items()})
        shapes['submit'] = geometry.submit_shape(tuple(self.spec['placement']['submit']))
        return shapes

    def _cover_places(self):
        """Places a flat cover may hide (spec 2.2: the reference card or a board), task frame."""
        return {name: dict(xy=list(board['frame_xy']), half=list(board['half_size']),
                           yaw=float(board['frame_yaw_rad']),
                           support_z=float(self.work['top_z']) + geometry.PAPER_THICKNESS, mode='push')
                for name, board in self.boards.items()}

    def _observability_rng(self):
        seed = int((getattr(self, 'instance', None) or {}).get('seed', 0)) & 0xFFFFFFFF
        return np.random.default_rng([seed, observe.OBSERVE_SALT])

    def _apply_observability(self):
        """Realise ``spec['observability']`` at the end of the build (brief deliverable 3).

        The candidate walk in :mod:`..stamps.hiding` fixes the uncover entry first, so the entry
        handed to :func:`observe.apply_observability` is one this scene can build; anything else
        that fails (an annex that holds nothing, a cover that cannot be built, a push gate that
        fails) leaves a problem in the record and raises here, which rejects the seed at G1.
        """
        spec = self.spec
        shapes = self._prop_shapes()
        places = self._cover_places()
        objects = {name: dict(radius=geometry.SOLE_RADIUS, height=geometry.STAMP_HEIGHT)
                   for name in self.stamp_names}
        surface = observe.work_surface_rect(self)
        relocate = S.Rect(surface.x0 + .06, surface.x1 - .06, max(surface.y0, 0.), surface.y1 - .06)
        floor = observe.floor_map(self)
        entries, notes = hiding.resolve(self, spec, objects=objects, places=places, keepouts=shapes,
                                        bounds=surface, floor=floor)
        # A cover always overlaps what it hides, so the covered prop drops out of the keep-outs
        # (observe requires it); everything else the task owns stays in, which is what keeps a
        # cover, an occluder or a relocated stamp off the other boards and off Submit.
        # The cover must hide the whole place, corners included: `observe.cover_place` scales a cover
        # until its bounding box overhangs `place_half`, so the place it is given is the inflated one
        # (`hiding.cover_half`) the mint-time candidate walk measured.
        covered = next((entry for entry in entries
                        if entry['mode'] == 'cover' and entry.get('cover') in hiding.PLACE_COVER_KINDS), None)
        if covered is not None and covered['object'] in places:
            place = places[covered['object']]
            places = dict(places, **{covered['object']: dict(
                place, half=list(hiding.cover_half(covered['cover'], place['half'])))})
        hidden_names = {entry['object'] for entry in entries if entry['mode'] == 'cover'}
        keepouts = [shape for name, shape in shapes.items() if name not in hidden_names]
        record = observe.apply_observability(self, dict(spec, hidden=entries), objects=objects, places=places,
                                             keepouts=keepouts, bounds=surface, rng=self._observability_rng(),
                                             floor=floor, occluder_relocate=relocate)
        record.update(notes, resolved=deepcopy(entries))
        record['goal_spot_clashes'] = self._goal_spot_clashes(record, shapes)
        self._refresh_stamp_poses(record)
        if record['problems'] or record['goal_spot_clashes']:
            # When the walk found no realisable candidate it leaves the drawn entry in place and the
            # build tries it anyway, so the problem above reads as if one cover misbehaved. Say what
            # actually happened: every candidate this scene offered, and why each was refused.
            exhausted = [] if notes.get('flavour_realised') else [
                'no uncover candidate is realisable: ' + ', '.join(
                    f"{row['object']}/{row['cover']} ({'; '.join(row['reasons']) or 'refused'})"
                    for row in notes.get('candidates_tried') or []) or 'the draw offered none']
            raise ValueError(f"{self.TASK_NAME}: observability {record['level']} not realisable on layout "
                             f"{(self.instance or {}).get('layout_id')} style {(self.instance or {}).get('style_id')}: "
                             + '; '.join(exhausted + list(record['problems'])
                                         + list(record['goal_spot_clashes']))[:600])
        return record

    def _goal_spot_clashes(self, record, shapes, gap=.004):
        """Nothing the observability wiring added or moved may stand on a board or on Submit unless
        it is the recorded uncover entry: a cover on the reference card is the task, a cover on it by
        accident is a defect. Returns human-readable strings; a non-empty list fails the build."""
        spots = {name: shapes[name] for name in list(self.boards) + ['submit']}
        extras = []
        for cover in record.get('covers', []):
            if not cover.get('built'):
                continue
            if cover.get('place'):
                extras.append((cover['name'], hiding.cover_shape(cover['kind'], cover['place_xy'],
                                                                 cover['half'], float(cover['yaw_frame'])),
                               {cover['place']}))
            else:
                xy = observe.world_to_frame_xy(self, cover['xy_world'])
                extras.append((cover['name'], S.Disc((float(xy[0]), float(xy[1])), float(cover['radius'])), set()))
        for occluder in record.get('occluders', []):
            extras.append((occluder['name'],
                           S.Box(tuple(occluder['frame_xy']), (occluder['size'][0] / 2, occluder['size'][1] / 2),
                                 float(occluder['frame_yaw'])), set()))
            moved = occluder.get('relocated')
            if moved:
                xy = observe.world_to_frame_xy(self, moved['to_xy_world'])
                extras.append((occluder['target'], S.Disc((float(xy[0]), float(xy[1])), geometry.SOLE_RADIUS), set()))
        problems = []
        for name, shape, allowed in extras:
            for spot, keep in spots.items():
                if spot not in allowed and S.clearance(shape, keep) < gap:
                    problems.append(f'{name} stands on {spot}')
        return problems

    def _refresh_stamp_poses(self, record):
        """Keep the recorded stamp poses in step with whatever observability moved."""
        moves = {entry['object']: entry for entry in record.get('annex', [])}
        for occluder in record.get('occluders', []):
            if occluder.get('relocated'):
                moves[occluder['target']] = dict(xy_world=occluder['relocated']['to_xy_world'])
        for name, move in moves.items():
            stamp = self.stamps.get(name)
            if stamp is None:
                continue
            frame_xy = observe.world_to_frame_xy(self, move['xy_world'])
            stamp['frame_xy'] = [float(frame_xy[0]), float(frame_xy[1])]
            stamp['moved_by_observability'] = True
        self.task_spec['stamps'] = deepcopy(self.stamps)

    # ---- model and timing ---------------------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        option = self.model.root.find('option')
        if option is None:
            option = ET.SubElement(self.model.root, 'option')
        option.set('timestep', repr(PHYSICS_TIMESTEP))
        supports = [board['collision_geom'] for board in self.boards.values()] + self._work_top_geom_names()
        self.task_spec['contact_pairs'] = geometry.add_contact_pairs(self.model.root, self.stamp_names, supports)
        self.task_spec['support_geoms'] = list(supports)

    def _support_surfaces(self):
        """Surfaces a released stamp may rest on, by fixture name: the work surface and any annex
        surface the observability wiring used. The goal text asks for every stamp back down on the
        counter, and at the ``look`` level a stamp starts on another counter, so its top slabs need
        the same tuned contacts and the same support test as the work top (contract C3)."""
        surfaces = {str(self.work['fixture']): float(self.work['top_z'])}
        for entry in (self.task_spec.get('observability') or {}).get('annex', []):
            surfaces[str(entry['fixture'])] = float(entry['top_z'])
        return surfaces

    def _work_top_geom_names(self):
        """RoboCasa names a counter's top slabs ``<fixture>_top_<i>``; empty if a
        layout names them otherwise, in which case the default contact applies."""
        prefixes = tuple(f'{fixture}_top' for fixture in self._support_surfaces())
        return sorted({geom.get('name') for geom in self.model.root.iter('geom')
                       if (geom.get('name') or '').startswith(prefixes)
                       and geom.get('contype') != '0' and geom.get('group') != '1'})

    def initialize_time(self, control_freq):
        super().initialize_time(control_freq)
        # Public ticks stay at the control rate while contacts resolve at 1 kHz.
        self.model_timestep = PHYSICS_TIMESTEP
        self.sim.model.opt.timestep = PHYSICS_TIMESTEP

    # ---- references and ink -------------------------------------------------
    def setup_task_references(self):
        m = self.sim.model
        self.obj_body_id = dict(getattr(self, 'obj_body_id', {}))
        self._stamp_geoms, self._die_geoms, self._stamp_masks = {}, {}, {}
        for name, stamp in self.stamps.items():
            self.obj_body_id[name] = m.body_name2id(name)
            self._stamp_geoms[name] = {m.geom_name2id(g) for g in stamp['collision_geoms']}
            self._die_geoms[name] = [m.geom_name2id(g) for g in stamp['die_geoms']]
            self._stamp_masks[name] = np.asarray(stamp['die_mask_private'], bool)
        self._sole_ids = {m.geom_name2id(stamp['sole_geom']): name for name, stamp in self.stamps.items()}
        self._board_ids = {m.geom_name2id(board['collision_geom']): name
                           for name, board in self.boards.items() if board['inkable']}
        self._texture_ids = {name: mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_TEXTURE, board['texture_name'])
                             for name, board in self.boards.items()}
        self._support_ids = self._resolve_support_geoms()
        self.task_spec['support_geom_ids'] = sorted(self._support_ids)
        # How many dots the private solution itself puts in each cell. Dies in a solution may
        # overlap (the uniqueness gate ORs their images), so a target cell can legitimately
        # carry two or three dots; the scorer needs the count to tell the design's own double
        # from a policy's misprint.
        self._expected_counts = rules.solution_dot_counts(
            [rules.bank_mask(i) for i in self.spec['mask_ids']], self.spec['solution'])
        self._reset_ink()

    def _resolve_support_geoms(self):
        """Geoms a released stamp may rest on: the boards and the top slabs of the work surface and
        of every annex surface an observability draw used."""
        m, d = self.sim.model, self.sim.data
        raw = m._model
        surfaces = self._support_surfaces()
        ids = {m.geom_name2id(board['collision_geom']) for board in self.boards.values()}
        for gid in range(raw.ngeom):
            name = m.geom_id2name(gid) or ''
            fixture = next((f for f in surfaces if name.startswith(f)), None)
            if fixture is None or not (raw.geom_contype[gid] or raw.geom_conaffinity[gid]):
                continue
            size = np.asarray(raw.geom_size[gid])
            top = float(d.geom_xpos[gid][2] + (size[2] if int(raw.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_BOX else 0.))
            if abs(top - surfaces[fixture]) <= .02:
                ids.add(gid)
        return ids

    def _reset_ink(self):
        self.ink = {name: np.zeros(tuple(board['shape']), dtype=bool) for name, board in self.boards.items()}
        self.ink['reference'] = rules.deposit(self.ink['reference'], np.asarray(self.spec['target'], bool),
                                              (0., 0.), 0., self.boards['reference']['half_size'][0])
        # The card's printed target is its baseline; any impression past it is collateral.
        self._reference_impressions = 0
        self.ink_events, self._active_contacts = [], set()
        for name in self.boards:
            self._upload_board(name)

    def _upload_board(self, name):
        m = self.sim.model
        board = self.boards[name]
        ink = self.ink[name]
        paper = board.get('paper_rgb') or rules.PAPER_RGB
        rgb = (rules.paper_rgb(ink, paper) if board['gridded']
               else rules.scratch_rgb(ink, board['half_size'][0], paper))
        tid = self._texture_ids[name]
        address = int(m.tex_adr[tid])
        # Ink rows advance in board-local +y; MuJoCo's plane maps texture rows the
        # other way, so display is row-reversed while scoring stays metric.
        m.tex_data[address:address + rgb.size] = rgb[::-1].ravel()
        context = getattr(self.sim, '_render_context_offscreen', None)
        if context is not None:
            context.upload_texture(tid)

    def _reset_internal(self):
        super()._reset_internal()
        if getattr(self, 'boards', None) and hasattr(self, '_texture_ids'):
            self._reset_ink()

    def _update_ink(self):
        """Deposit an impression for every newly touching flat sole-board pair."""
        m, d = self.sim.model, self.sim.data
        active, changed = set(), False
        for index, contact in enumerate(d.contact[:d.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self._board_ids and b in self._sole_ids:
                a, b = b, a
            if a not in self._sole_ids or b not in self._board_ids:
                continue
            stamp, board_name = self._sole_ids[a], self._board_ids[b]
            pair = (stamp, board_name)
            active.add(pair)                       # side contacts rearm too: only separation does
            if pair in self._active_contacts:
                continue
            board = self.boards[board_name]
            bid = self.obj_body_id[stamp]
            rotation = np.asarray(d.body_xmat[bid]).reshape(3, 3)
            force = np.zeros(6)
            mujoco.mj_contactForce(m._model, d._data, index, force)
            normal_on_stamp = np.asarray(contact.frame).reshape(3, 3)[0] * (1 if int(contact.geom2) == a else -1)
            height_error = float(d.body_xpos[bid][2] - board['position'][2])
            if not rules.contact_print_eligible(rotation, normal_on_stamp, height_error, force[0]):
                continue
            delta = np.asarray(d.body_xpos[bid])[:2] - np.asarray(board['position'])[:2]
            offset = yaw_matrix(-board['yaw_rad'])[:2, :2] @ delta
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]) - board['yaw_rad'])
            before = int(np.count_nonzero(self.ink[board_name]))
            self.ink[board_name] = rules.deposit(self.ink[board_name], self._stamp_masks[stamp],
                                                 offset, yaw, board['half_size'][0])
            self._upload_board(board_name)
            self.ink_events.append(dict(
                step=int(self.timestep), stamp=stamp, board=board_name, translation_xy_m=offset.tolist(),
                yaw_rad=yaw, actual_contact_geoms=[m.geom_id2name(a), m.geom_id2name(b)],
                contact_position_world=np.asarray(contact.pos).tolist(),
                contact_normal_on_stamp=normal_on_stamp.tolist(), sole_height_error_m=height_error,
                normal_force_n_private=float(force[0]),
                new_ink_pixels=int(np.count_nonzero(self.ink[board_name])) - before))
            self._active_contacts.add(pair)
            if board_name == 'reference':
                self._reference_impressions += 1
            changed = True
        self._active_contacts.intersection_update(active)
        return changed

    def _post_action(self, action):
        # Ink first: a Submit press in the same tick must freeze the board it made.
        if self._update_ink():
            self._update_observables(force=True)
        return super()._post_action(action)

    # ---- state and scoring ---------------------------------------------------
    def stamp_state(self, name):
        m, d = self.sim.model, self.sim.data
        bid, gids = self.obj_body_id[name], self._stamp_geoms[name]
        robot_contacts, support = [], []
        for index, contact in enumerate(d.contact[:d.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if not ({a, b} & gids):
                continue
            other = b if a in gids else a
            other_name = m.geom_id2name(other) or ''
            force = np.zeros(6)
            mujoco.mj_contactForce(m._model, d._data, index, force)
            normal = np.asarray(contact.frame).reshape(3, 3)[0] * (1 if b in gids else -1)
            row = dict(geom=other_name, normal_force_n=float(force[0]), normal_on_object=normal.tolist())
            if other_name.startswith(('robot', 'gripper', 'mobilebase')):
                robot_contacts.append(row)
            if other in self._support_ids:
                support.append(row)
        rotation = np.asarray(d.body_xmat[bid]).reshape(3, 3)
        return dict(position=np.asarray(d.body_xpos[bid]).tolist(),
                    quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
                    yaw_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                    frame_yaw_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0]) - self.work['yaw']),
                    linear_velocity=np.asarray(d.get_body_xvelp(name)).tolist(),
                    angular_velocity=np.asarray(d.get_body_xvelr(name)).tolist(),
                    robot_contacts=robot_contacts, support_contacts=support,
                    grasped=bool(self._check_grasp(self.robots[0].gripper['right'],
                                                   [m.geom_id2name(g) for g in gids])))

    def _current_score(self, states=None):
        states = states or {name: self.stamp_state(name) for name in self.stamp_names}
        cells, fractions, stray = rules.ink_cells(self.ink['final'])
        target = np.asarray(self.spec['target'], bool)
        checks = {}
        for name, state in states.items():
            checks[name] = dict(
                released=bool(not state['robot_contacts'] and not state['grasped']),
                stable=bool(np.linalg.norm(state['linear_velocity']) < .03
                            and np.linalg.norm(state['angular_velocity']) < .2),
                supported=bool(any(c['normal_force_n'] > 1e-5 and c['normal_on_object'][2] > .5
                                   for c in state['support_contacts'])))
        # One entry per print event, in the board frame, so the analysis can redo the grading from
        # the record. ``task_snapshot`` keeps the full events with contact details for the trace.
        impressions = [dict(stamp=e['stamp'], board=e['board'], x_m=float(e['translation_xy_m'][0]),
                            y_m=float(e['translation_xy_m'][1]), yaw_rad=float(e['yaw_rad']), step=int(e['step']))
                       for e in self.ink_events]
        # The board is graded dot by dot from the impressions and the die masks,
        # not from the raster - two half impressions fill a cell exactly as one centred dot does.
        # The raster stays for the picture the policy sees and for the off-grid guard below.
        # A cell the solution prints k times is graded against that count - any 1 to k clean dots
        # read as the design - so pressing the solution exactly scores 1.
        graded = rules.grade_impressions([(e['stamp'], e['translation_xy_m'], e['yaw_rad'])
                                          for e in self.ink_events if e['board'] == 'final'],
                                         self._stamp_masks, target,
                                         getattr(self, '_expected_counts', None))
        match = bool(graded['pattern'] >= 1.)
        finish = bool(all(all(v.values()) for v in checks.values()))
        return dict(success=bool(match and finish and int(stray) <= rules.MAX_OFF_GRID_PIXELS),
                    pattern_match=match, pattern=float(graded['pattern']), finish=finish,
                    target_grades=graded['target_grades'], stray_dots=graded['stray_dots'],
                    dots=graded['dots'],
                    final_cells=cells.tolist(), target_cells=target.astype(int).tolist(),
                    ink_fraction_per_cell=fractions.tolist(), off_grid_ink_pixels=int(stray),
                    stamps=checks, ink_event_count=len(self.ink_events), impressions=impressions)

    def task_snapshot(self):
        return dict(stamps={name: self.stamp_state(name) for name in self.stamp_names},
                    ink_events=list(self.ink_events), progress=self.task_progress(),
                    collateral=self.collateral_checks(),
                    board_cells={name: rules.ink_cells(self.ink[name])[0].tolist()
                                 for name, board in self.boards.items() if board['gridded']})

    # ---- protocol hooks (spec 1.4, 1.5; contracts C1, C3) --------------------
    def task_progress(self):
        """0.9 * the dot-rule pattern score + 0.1 * the finish term.

        ``pattern`` grades the printed dots against the target cells (any 1 to k clean
        dots in a cell the solution prints k times 1, an edge-crossing cell or one with
        more dots than the design puts there 0.5, a missing cell 0, every dot that
        lands on the paper outside the target -1, floored at 0);
        ``finish`` is 1 when every stamp is released, still and supported. A perfect
        board with a stamp still in hand reads 0.90, and an untouched scene reads 0.10,
        the stamps being down and still already. 1 exactly when the success predicate
        holds: a board that reads 1 while the predicate fails - stray ink past the
        35-pixel raster tolerance - reports ``ALMOST`` instead, per contract C1.
        """
        score = self._current_score()
        value = ((1. - rules.FINISH_WEIGHT) * float(score['pattern'])
                 + rules.FINISH_WEIGHT * float(score['finish']))
        if value >= 1.:
            return 1. if score['success'] else ALMOST
        return float(min(max(value, 0.), 1.))

    def collateral_checks(self):
        """Spec 1.5: ink on the reference card, costed in cells. Permanent (paper)."""
        cells, _, _ = rules.ink_cells(self.ink['reference'])
        target = np.asarray(self.spec['target'], bool)
        extra = int(np.count_nonzero(cells & ~target))
        return dict(reference_inked=dict(bad=bool(extra or getattr(self, '_reference_impressions', 0)),
                                         object='reference', cost=float(extra)))

    def _cover_joint(self, name):
        obj = (getattr(self, 'objects', {}) or {}).get(name)
        joints = list(getattr(obj, 'joints', []) or []) if obj is not None else []
        return joints[0] if joints else f'{name}_joint'

    def _uncover_steps(self):
        """C3: take every cover off before the stamps move.

        A flat cover is pushed along the direction its build-time push gate chose, by the length
        that clears the place it hides; an object cover is set down on a free spot of the same
        surface. Both stay on the work surface and clear of every task prop, and neither is moved
        by the robot: these are state edits, like the rest of the certificate.
        """
        steps = []
        record = self.task_spec.get('observability') or {}
        covers = [c for c in record.get('covers', []) if c.get('built')]
        if not covers:
            return steps
        rng = self._observability_rng()
        shapes = self._prop_shapes()
        bounds = observe.work_surface_rect(self)
        # Everything a set-down cover must clear: the task props, the occluders the wiring stood (they are not
        # task props, so `_prop_shapes` does not know them) and the counter decor; a cover teleported into any
        # of these is resolved by the physics with whatever impulse the overlap takes.
        keepouts = list(shapes.values()) + observe.decor_shapes_on_work(self)
        for occluder in record.get('occluders', []):
            keepouts.append(S.Box(tuple(occluder['frame_xy']), (occluder['size'][0] / 2, occluder['size'][1] / 2),
                                  float(occluder['frame_yaw'])))
        for cover in covers:
            name = cover['name']
            try:
                joint = self._cover_joint(name)
                qpos = np.asarray(self.sim.data.get_joint_qpos(joint), float).copy()
                chosen = (cover.get('reach') or {}).get('chosen')
                if cover.get('place') and chosen:
                    target = (np.asarray(cover['place_xy'], float)
                              + np.asarray(chosen['direction_frame'], float) * float(chosen['length_m']))
                else:
                    radius = float(cover.get('radius') or max(cover.get('half') or (.10, .10)))
                    spot = S.scatter(rng, 1, bounds, radius, .01, keepouts=keepouts, keepout_gap=.02,
                                     max_tries=300, attempts_per_prop=300)[0]
                    target = np.asarray(spot[:2], float)
                world = observe.frame_to_world_xy(self, target)
                qpos[:2] = world
                self.sim.data.set_joint_qpos(joint, qpos)
                self.sim.data.set_joint_qvel(joint, np.zeros(6))
                self.sim.forward()
                keepouts.append(S.Disc((float(target[0]), float(target[1])),
                                       float(cover.get('radius') or max(cover.get('half') or (.10, .10)))))
                steps.append(dict(step=f'uncover_{name}', ok=True,
                                  detail=dict(kind=cover['kind'], hides=cover.get('place') or cover.get('object'),
                                              moved_to_frame=[round(float(v), 4) for v in target])))
            except Exception as error:                   # never raises (C3)
                steps.append(dict(step=f'uncover_{name}', ok=False, detail=f'{type(error).__name__}: {error}'))
        for _ in range(RELEASE_SETTLE_TICKS):
            self.step(np.zeros(self.action_dim))
        return steps

    def teleport_solution(self):
        """Contract C3: print the target by moving the solution stamps, without the robot.

        Each solution stamp is set 4 mm above the final board's face at its
        quarter turn through its free joint, dropped with zero-action steps so the
        real sole-to-board contact deposits the impression (nothing about the ink
        is short-circuited), and then returned to its reset pose so it ends
        released, still and supported. The scratch board is left alone: this
        certificate prints straight onto the final board.
        """
        steps = []
        try:
            board = self.boards['final']
            action = np.zeros(self.action_dim)
            steps.extend(self._uncover_steps())

            def place(name, pose):
                self.sim.data.set_joint_qpos(f'{name}_joint', np.asarray(pose, float))
                self.sim.data.set_joint_qvel(f'{name}_joint', np.zeros(6))
                self.sim.forward()

            for stamp_index, turns in self.spec['solution']:
                name = f'stamp_{int(stamp_index)}'
                before = len(self.ink_events)
                home = np.asarray(self.sim.data.get_joint_qpos(f'{name}_joint'), float).copy()
                yaw = board['yaw_rad'] + int(turns) * math.pi / 2
                try:
                    place(name, np.r_[board['position'][:2], board['position'][2] + .004,
                                      math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
                    for _ in range(12):
                        self.step(action)
                    place(name, home)
                    for _ in range(3):
                        self.step(action)
                except Exception as error:      # never raises: report the step that failed
                    steps.append(dict(step=f'press_{name}', ok=False, detail=repr(error)))
                    continue
                printed = len(self.ink_events) - before
                steps.append(dict(step=f'press_{name}', ok=bool(printed == 1),
                                  detail=dict(turns=int(turns), impressions=int(printed))))
            # A stamp put back home has only had three ticks to land, so the last one pressed
            # was still moving (or still in the air) when the score read it and the certificate
            # failed at 0.99 with a perfect board (layout 3 / style 14, both candidates).
            for _ in range(RELEASE_SETTLE_TICKS):
                self.step(action)
            score = self._current_score()
            steps.append(dict(step='final_board', ok=bool(score['pattern_match']),
                              detail=dict(final_cells=score['final_cells'], target_cells=score['target_cells'],
                                          off_grid_ink_pixels=score['off_grid_ink_pixels'])))
            steps.append(dict(step='stamps_released', ok=bool(all(all(v.values()) for v in score['stamps'].values())),
                              detail=score['stamps']))
            steps.append(dict(step='success', ok=bool(score['success']),
                              detail=dict(progress=self.task_progress())))
        except Exception as error:               # pragma: no cover - defensive, C3 forbids raising
            steps.append(dict(step='teleport_solution', ok=False, detail=repr(error)))
        return steps

    # ---- gate G3 -------------------------------------------------------------
    def hidden_state_check(self, image_size=512):
        """G3: the dies never reach a policy camera, and the public props are in view.

        Renders the three policy cameras twice with every die geom repainted a
        different opaque colour, and requires the images to be pixel-identical
        (fail-closed): if a single die pixel reached a camera the two renders
        would differ there. Repainting rather than hiding is deliberate. Setting
        a die to alpha zero changes the scene's transparent-geom set, and MuJoCo
        blends overlapping semi-transparent surfaces in that order, so a kitchen
        holding a glass-doored appliance reports tens of differing pixels *on
        the appliance* for a die that is nowhere near it (measured: 54 px on the
        toaster oven, layout 7 style 2). Two opaque colours leave the draw order
        untouched.

        Also checks that the reference card, the final board and Submit fall
        inside some fixed camera's frame. That second test is a frustum test,
        not an occlusion test: the arm passes in front of the right-hand props
        in the left camera, and it moves, so requiring an unoccluded pixel at
        reset would fail for reasons that do not concern the scene's layout.
        """
        from roboquest.harness.contract import CAMERAS
        m = self.sim.model
        die_ids = [gid for ids in self._die_geoms.values() for gid in ids]
        saved = {gid: m.geom_rgba[gid].copy() for gid in die_ids}
        try:
            for gid in die_ids:
                m.geom_rgba[gid] = [1., 0., 0., 1.]
            before = {c: self.sim.render(width=image_size, height=image_size, camera_name=c).copy()
                      for c in CAMERAS}
            for gid in die_ids:
                m.geom_rgba[gid] = [0., 1., 0., 1.]
            after = {c: self.sim.render(width=image_size, height=image_size, camera_name=c).copy()
                     for c in CAMERAS}
        finally:
            for gid, rgba in saved.items():
                m.geom_rgba[gid] = rgba
        identical = {c: bool(np.array_equal(before[c], after[c])) for c in CAMERAS}
        differing = {c: int(np.count_nonzero(np.any(before[c] != after[c], axis=-1))) for c in CAMERAS}
        in_frame = {}
        for name in ('reference', 'final', 'submit'):
            if name == 'submit':
                point = np.asarray(self.task_spec['submit_button']['position'], float)
                corners = [point + [dx, dy, 0.] for dx in (-.036, .036) for dy in (-.036, .036)]
            else:
                board = self.boards[name]
                half = board['half_size']
                corners = [np.asarray(board['position'], float)
                           + yaw_matrix(board['yaw_rad']) @ [sx * half[0], sy * half[1], 0.]
                           for sx in (-1, 1) for sy in (-1, 1)]
            in_frame[name] = {c: bool(all(self._point_in_camera(c, p, image_size) for p in corners))
                              for c in CAMERAS if c != 'robot0_eye_in_hand'}
        passed = all(identical.values()) and all(any(v.values()) for v in in_frame.values())
        report = dict(die_geoms=len(die_ids), identical=identical, differing_pixels=differing,
                      inside_fixed_camera_frame=in_frame, image_size=int(image_size))
        observability = self._observability_check(image_size)
        if observability is not None:
            report['observability'] = observability
            passed = passed and bool(observability['passed'])
        return bool(passed), report

    def _observability_check(self, image_size=512):
        """G3 for the observability axis (spec 2.2): what the spec hides is in no policy camera,
        what it does not is in one, and every ``look`` object stands where a reachable viewpoint
        sees it. None at the ``visible`` level, where there is nothing to check."""
        record = self.task_spec.get('observability') or {}
        entries = record.get('realised') or []
        if not entries:
            return None
        hidden = [name for name in record.get('hidden_bodies', []) if name in self.stamps]
        hidden_geoms = [self.boards[cover['place']]['print_geom'] for cover in record.get('covers', [])
                        if cover.get('built') and cover.get('place') in self.boards]
        visible = [name for name in self.stamp_names if name not in hidden]
        check = observe.hidden_check(self, hidden, visible=visible, image_size=image_size,
                                     hidden_geoms=hidden_geoms)
        floor = observe.floor_map(self)
        look = {entry['object']: observe.look_reachable(self, entry['object'], floor=floor, at_build=False,
                                                        radius=geometry.SOLE_RADIUS,
                                                        height=geometry.STAMP_HEIGHT)
                for entry in entries if entry['mode'] in hiding.LOOK_MODES}
        reach = {name: bool((row.get('reach') or {}).get('passed'))
                 for name, row in [(c['name'], c) for c in record.get('covers', []) if c.get('built')]}
        passed = bool(check['passed'] and all(row['passed'] for row in look.values()) and all(reach.values()))
        return dict(passed=passed, level=record.get('level'), hidden=hidden, hidden_geoms=hidden_geoms,
                    hidden_ok=check['hidden_ok'], visible_ok=check['visible_ok'],
                    deterministic=check['deterministic'], pixels=check['pixels'],
                    look_reachable={name: dict(passed=row['passed'], stance=row['stance'],
                                               reasons=row['reasons'][:2]) for name, row in look.items()},
                    cover_reach=reach, problems=list(record.get('problems') or []),
                    fallbacks=list(record.get('fallbacks') or []))

    def _point_in_camera(self, camera_name, point, image_size):
        m, d = self.sim.model, self.sim.data
        cid = m.camera_name2id(camera_name)
        rotation = np.asarray(d.cam_xmat[cid]).reshape(3, 3)
        local = rotation.T @ (np.asarray(point, float) - np.asarray(d.cam_xpos[cid]))
        if local[2] >= -1e-6:                       # MuJoCo cameras look down -z
            return False
        focal = .5 * image_size / math.tan(math.radians(float(m._model.cam_fovy[cid])) / 2)
        px = image_size / 2 + focal * local[0] / -local[2]
        py = image_size / 2 - focal * local[1] / -local[2]
        return bool(0 <= px < image_size and 0 <= py < image_size)


# The exact key set, for registries that prefer a frozenset to the predicate. Built
# after the class body because it applies ``Stamps.is_holdout`` to every key.
Stamps.HOLDOUT_KEYS = _holdout_keys()
