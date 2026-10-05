"""Frozen task instances and their registry. CPU only: no simulator imports.

An instance is one fully specified episode start, in the sense of RoboCasa's
``ep_meta`` record: the kitchen layout and style, the task's own specification
(hidden state, structure, placements) and the factor values the generator
varied. Seeds *mint* instances; instances, not seeds, identify episodes, so a
generator change never silently changes an evaluation set.

Registry layout::

    <out>/registry.json            summary + one row per instance (no full spec)
    <out>/instances/<id>.json      the instance, loadable by the adapter
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

# RoboCasa's own scene split: layouts and styles 1-10 are its test set, 11-60 its train set.
ROBOCASA_TEST_IDS = tuple(range(1, 11))
ROBOCASA_TRAIN_IDS = tuple(range(11, 61))
# Layouts whose work surfaces have been exercised here (room search and cube/puzzle renders).
# Widening the pool is a generation-time decision that the build gate must confirm.
DEFAULT_LAYOUT_POOL = (1, 2, 4, 7, 9, 10, 13, 17)
DEFAULT_STYLE_POOL = tuple(range(1, 11))
# Suite pools: development kitchens are RoboCasa train layouts that pass the build gate with styles
# 1-10; evaluation kitchens are RoboCasa test layouts that pass, with five held-out styles. The layout
# lists are filled from the build gate.
DEV_LAYOUT_POOL = (13, 17)
EVAL_LAYOUT_POOL = (1, 2, 4, 7, 9, 10)
DEV_STYLE_POOL = tuple(range(1, 11))
EVAL_STYLE_POOL = (11, 12, 13, 14, 15)
INSTANCE_ID_CHARS = 16


def _salt(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f'not JSON serialisable: {type(value).__name__}')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=_jsonable)


def instance_id(task, generator_version, layout_id, style_id, spec):
    digest = hashlib.sha256(canonical_json(dict(task=task, generator_version=generator_version,
        layout_id=int(layout_id), style_id=int(style_id), spec=spec)).encode()).hexdigest()
    return digest[:INSTANCE_ID_CHARS]


def rng_for(task_name, seed):
    """Independent, named substreams so adding a factor never reshuffles the others.

    ``spawn`` hands out children in order, so appending ``wording`` (spec 1.3
    wording variation) leaves the first four streams bit-identical to v0.
    """
    root = np.random.SeedSequence([_salt(task_name), int(seed)])
    scene, structure, poses, materials, wording = root.spawn(5)
    return dict(scene=np.random.default_rng(scene), structure=np.random.default_rng(structure),
                poses=np.random.default_rng(poses), materials=np.random.default_rng(materials),
                wording=np.random.default_rng(wording))


def styles_for_layout(layout_styles, layout_id):
    """The styles the layout gate allows on this layout, or ``None`` when it named no set for it.

    ``layouts.json`` comes back from JSON with string keys, so both are accepted.
    """
    if not layout_styles:
        return None
    styles = layout_styles.get(int(layout_id), layout_styles.get(str(int(layout_id))))
    return [int(s) for s in styles] if styles else None


def mint(task, seed, factors=None, layout_pool=DEFAULT_LAYOUT_POOL, style_pool=DEFAULT_STYLE_POOL,
         layout_styles=None):
    """Derive one instance from a seed.

    ``task`` is a RoboQuestKitchen subclass (or any object with its generator
    protocol). Explicit ``factors`` fix levels; ``kitchen_layout`` and
    ``kitchen_style`` may be fixed the same way.

    ``layout_styles`` is the layout gate's optional per-layout style set
    (``layouts.json`` ``dev_styles`` / ``eval_styles``, keys may be ints or
    strings): when the drawn layout appears in it, the style is drawn from that
    layout's own set instead of ``style_pool``, so no instance is minted on a
    (layout, style) pair whose kitchen cannot host the task's footprint. The
    draw order is unchanged, so passing nothing reproduces v0 byte for byte.
    """
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    # ``OPTIONAL_FACTORS`` ({name: default}): mint-time choices a task accepts as fixed factors without
    # making them grid cells (the fob search's ``double_door``); the default is recorded on every row.
    optional = dict(getattr(task, 'OPTIONAL_FACTORS', None) or {})
    factors = {**task.default_factors(), **optional, **dict(factors or {})}
    unknown = set(factors) - set(task.FACTORS) - set(optional) - {'kitchen_layout', 'kitchen_style'}
    if unknown:
        raise ValueError(f'unknown factors for {task.TASK_NAME}: {sorted(unknown)}')
    for name, value in factors.items():
        levels = task.FACTORS.get(name)
        if levels is not None and value not in levels and value not in dev_only_levels(task).get(name, ()):
            raise ValueError(f'{task.TASK_NAME}: factor {name}={value!r} not in {levels}'
                             + (f' nor in the development-only levels {dev_only_levels(task)[name]}'
                                if name in dev_only_levels(task) else ''))
    streams = rng_for(task.TASK_NAME, seed)
    layout_id = int(factors.get('kitchen_layout') or streams['scene'].choice(layout_pool))
    allowed = styles_for_layout(layout_styles, layout_id) or style_pool
    style_id = int(factors.get('kitchen_style') or streams['scene'].choice(allowed))
    factors['kitchen_layout'], factors['kitchen_style'] = layout_id, style_id
    spec = task.sample_spec(streams, factors, seed)
    spec = json.loads(canonical_json(spec))  # plain JSON types only, so ids are stable
    spec['contract_version'] = task.CONTRACT_VERSION
    # Spec 1.3: a random half of the instances state the count in the goal's scope sentence. Wording
    # variation only, drawn from its own stream so no task ever spends the draw itself.
    spec['show_count'] = bool(streams['wording'].integers(2))
    problems = task.validate_spec(spec)
    if problems:
        raise ValueError(f'{task.TASK_NAME} seed {seed}: ' + '; '.join(problems))
    return dict(task=task.TASK_NAME, contract_version=task.CONTRACT_VERSION,
                generator_version=task.GENERATOR_VERSION, seed=seed, factors=factors,
                layout_id=layout_id, style_id=style_id, spec=spec,
                split_key=task.split_key(spec), split=None, goal=str(task.goal(spec)),
                instance_id=instance_id(task.TASK_NAME, task.GENERATOR_VERSION, layout_id, style_id, spec))


def task_holdout(task):
    """The task's own held-out structure classes: exact keys in ``HOLDOUT_KEYS`` and/or a
    ``is_holdout(split_key)`` classmethod. Returns (keys, predicate_or_None)."""
    keys = set(getattr(task, 'HOLDOUT_KEYS', ()) or ())
    predicate = getattr(task, 'is_holdout', None)
    return keys, (predicate if callable(predicate) else None)


def key_is_held_out(split_key, holdout_keys=(), predicate=None):
    if split_key is None:
        return False
    if split_key in set(holdout_keys or ()):
        return True
    return bool(predicate(split_key)) if predicate is not None else False


def assign_splits(instances, holdout_every=3, holdout_keys=None, holdout_layouts=(), holdout_styles=(),
                  predicate=None):
    """Development/evaluation split by factor level, never by seed.

    ``holdout_keys`` (exact structure classes) and ``predicate`` (a rule over
    the split key) come from the task; when neither is given every
    ``holdout_every``-th structure class (sorted) is held out. Any instance in a
    held-out kitchen layout or style is evaluation as well. Returns the exact
    held-out keys seen.
    """
    keys = sorted({row['split_key'] for row in instances if row['split_key']})
    if holdout_keys is None and predicate is None:
        holdout_keys = {k for i, k in enumerate(keys) if holdout_every and i % holdout_every == holdout_every - 1}
    holdout_keys = set(holdout_keys or ())
    for row in instances:
        held = (key_is_held_out(row['split_key'], holdout_keys, predicate)
                or row['layout_id'] in set(holdout_layouts) or row['style_id'] in set(holdout_styles))
        row['split'] = 'eval' if held else 'dev'
    return holdout_keys | {k for k in keys if key_is_held_out(k, holdout_keys, predicate)}


def dev_only_levels(task):
    """``DEV_ONLY_LEVELS``: ``{factor: (level, ...)}`` minted for development only.

    A development-only level is a real level of a real factor (for example the
    combined ``look+uncover`` observability level) that no evaluation instance
    may claim, so ``mint`` accepts it but the balanced grid excludes it.
    Default ``{}``: a task without the attribute behaves exactly as before.
    """
    raw = getattr(task, 'DEV_ONLY_LEVELS', None) or {}
    return {name: tuple(levels) for name, levels in raw.items() if levels}


def even_quotas(total, cell_count):
    """``total`` spread evenly over ``cell_count`` cells; the remainder goes round-robin to the first cells."""
    if cell_count <= 0:
        return []
    base, extra = divmod(max(0, int(total)), int(cell_count))
    return [base + (1 if i < extra else 0) for i in range(cell_count)]


def grid_cells(task):
    """Every reported factor combination of the task, in ``FACTORS`` order.

    Development-only levels are excluded: the grid is what evaluation is
    balanced over, and development gets those levels from :func:`dev_only_cells`.
    """
    dev_only = dev_only_levels(task)
    cells = [dict()]
    for name in task.FACTORS:
        levels = [level for level in task.FACTORS[name] if level not in dev_only.get(name, ())]
        cells = [dict(cell, **{name: level}) for cell in cells for level in levels]
    return cells


def dev_only_cells(task):
    """One cell per development-only level, balanced over the other factors' levels.

    ``{'observability': ('look+uncover',)}`` with a three-level size factor
    gives three cells, so the extra development instances spread over the size
    levels instead of piling up on one. A development-only level named for a
    factor the task does not declare cannot be balanced and is ignored here
    (``mint`` still accepts it when it is asked for explicitly).
    """
    dev_only = dev_only_levels(task)
    cells = []
    for factor in task.FACTORS:
        for level in dev_only.get(factor, ()):
            others = [dict()]
            for name in task.FACTORS:
                if name == factor:
                    continue
                levels = [lv for lv in task.FACTORS[name] if lv not in dev_only.get(name, ())]
                others = [dict(cell, **{name: lv}) for cell in others for lv in levels]
            cells.extend(dict(cell, **{factor: level}) for cell in others)
    return cells


DEV_ONLY_SEED_OFFSET = 500_000   # development-only instances walk their own seed range, below the evaluation one
CELL_SEED_STRIDE = 1_000       # each factor cell walks its own seed range: a seed the hold-out or
                               # validity filter rejects then differs between cells, so the kitchen
                               # walk (mint_grid.CellWalk) is not shifted the same way in every cell
KITCHEN_MISS_LIMIT = 3         # straight kitchen-blamed rejections before a cell's walk leaves that kitchen
HOLDOUT_SKIP = object()        # try_mint's verdict for a seed the hold-out filter skips: not the kitchen's fault
EVAL_SEED_OFFSET = 1_000_000   # evaluation instances never share a seed (hence a task spec) with development ones


def mint_grid(task, per_cell_dev, eval_total, dev_pools, eval_pools, first_seed=0, gate=None,
              max_attempts_per_cell=400, log=None, dev_only_fraction=0.25, dev_mix=(), eval_mix=()):
    """Balanced generation over the factor grid.

    ``dev_mix`` / ``eval_mix`` are optional per-cell mixes over a task's ``OPTIONAL_FACTORS``: a
    sequence of ``(overrides, count)`` pairs, e.g. ``[({'double_door': True}, 2)]`` asks every cell for
    two instances minted with that override before the rest of its quota is minted plain (the
    optional factor at its default). One attempt budget per cell covers the whole quota; the empty mix
    reproduces the plain walk byte for byte.

    Development: for every factor cell, walk seeds from ``first_seed`` until
    ``per_cell_dev`` instances whose structure class is *not* held out have
    passed ``gate``; kitchens from ``dev_pools``.

    ``dev_pools`` / ``eval_pools`` are the keyword arguments of :func:`mint`:
    ``layout_pool``, ``style_pool`` and the layout gate's optional
    ``layout_styles`` (per-layout allowed styles, see :func:`styles_for_layout`),
    so a layout that only hosts the footprint at some styles is never minted at
    the others.

    Development-only levels (``DEV_ONLY_LEVELS``) are minted afterwards, from
    their own seed range (``first_seed + DEV_ONLY_SEED_OFFSET``), balanced over
    :func:`dev_only_cells`, enough of them that they are ``dev_only_fraction``
    of the development split (0.25 of dev means one extra instance for every
    three balanced ones). They carry ``dev_only=True``.

    Evaluation: ``eval_total`` split into per-cell quotas by :func:`even_quotas`
    (54 over 9 cells is 6 each; a remainder goes round-robin to the first
    cells). Each cell walks its own seed counter from
    ``first_seed + EVAL_SEED_OFFSET + cell * CELL_SEED_STRIDE`` (own seed range
    per cell, so a seed rejected by the hold-out filter does not shift the
    kitchen walk identically in every cell); its kitchens come from the layout pool
    walked cyclically, each cell starting further along (``CellWalk``), so
    a split spreads evenly over its kitchens instead of letting the same few
    seed draws dominate every cell (with the matched draw a 54-instance eval
    split used five kitchens, one of them 24 times). It keeps only
    held-out structure classes with kitchens from ``eval_pools``, until its
    quota passed the gate or ``max_attempts_per_cell`` seeds are spent. A cell
    that falls short is
    recorded in ``stats['eval_skipped_cells']`` with its shortfall and the run
    continues; quotas are never moved to another cell, because a balanced grid
    matters more than the total. Returns (instances, stats).
    """
    holdout_keys, predicate = task_holdout(task)
    stats = dict(cells=[], dev_rejected=0, eval_rejected=0, gate_failed=0, invalid=0, eval_skipped_cells=[],
                 dev_only_cells=[], dev_only_total=0, dev_only_fraction=float(dev_only_fraction or 0.0))
    accepted = []

    def try_mint(seed, cell, pools, want_holdout, layout=None, overrides=None):
        factors = dict(cell)
        if overrides:
            factors.update(overrides)
        if layout is not None:
            factors['kitchen_layout'] = int(layout)
        try:
            instance = mint(task, seed, factors, **pools)
        except (ValueError, RuntimeError) as error:     # invalid spec, or a sampler that gave up on this seed
            stats['invalid'] += 1
            if log:
                log(f'seed {seed} {cell}: invalid: {error}')
            return None
        held = key_is_held_out(instance['split_key'], holdout_keys, predicate)
        if held != want_holdout:
            stats['eval_rejected' if want_holdout else 'dev_rejected'] += 1
            return HOLDOUT_SKIP
        passed, report = (True, {'skipped': True}) if gate is None else gate(instance['spec'])
        instance['gate'] = dict(passed=bool(passed), report=report)
        if not passed:
            stats['gate_failed'] += 1
            return None
        instance['split'] = 'eval' if want_holdout else 'dev'
        instance['cell'] = dict(cell)
        return instance

    class CellWalk:
        """One cell's seed walk: next seed, attempts, accepted count and the kitchen slot. The slot,
        a position in the layout pool offset per cell by ``stride``, advances on every accepted
        instance and after ``KITCHEN_MISS_LIMIT`` straight kitchen-blamed rejections, so a split's
        accepted instances spread evenly over its kitchens while a kitchen that keeps rejecting a
        cell's seeds is left behind; a seed the hold-out filter skips moves neither."""

        def __init__(self, seed, cell_index, stride, pools):
            self.seed, self.attempts, self.got, self.misses = int(seed), 0, 0, 0
            self.pool = list(pools.get('layout_pool') or DEFAULT_LAYOUT_POOL)
            self.slot = cell_index * max(int(stride), 1)

        def kitchen(self):
            return self.pool[self.slot % len(self.pool)]

        def step(self, result):
            """Book one attempt; returns the accepted instance or None."""
            self.seed += 1
            self.attempts += 1
            if result is HOLDOUT_SKIP:
                return None
            if result is None:
                self.misses += 1
                if self.misses >= KITCHEN_MISS_LIMIT:
                    self.slot, self.misses = self.slot + 1, 0
                return None
            self.got += 1
            self.slot, self.misses = self.slot + 1, 0
            return result

    def fill(w, cell, pools, want_holdout, quota, mix, on_accept):
        """Walk the cell's seeds until ``quota`` instances are accepted: each mix entry's count with its
        overrides first, then the rest plain, all on the cell's one attempt budget."""
        plan = [(dict(overrides), int(count)) for overrides, count in (mix or ()) if int(count) > 0]
        plan.append(({}, max(0, int(quota) - sum(n for _, n in plan))))
        for overrides, count in plan:
            start = w.got
            while w.got - start < count and w.attempts < max_attempts_per_cell:
                instance = w.step(try_mint(w.seed, cell, pools, want_holdout, layout=w.kitchen(),
                                           overrides=overrides or None))
                if instance is not None:
                    on_accept(instance)

    cells = grid_cells(task)
    dev_walk = [CellWalk(first_seed + ci * CELL_SEED_STRIDE, ci, per_cell_dev, dev_pools) for ci in range(len(cells))]
    for ci, cell in enumerate(cells):
        w = dev_walk[ci]
        fill(w, cell, dev_pools, False, per_cell_dev, dev_mix, accepted.append)
        stats['cells'].append(dict(cell=cell, dev=w.got, dev_attempts=w.attempts))
    # A cell whose every structure class is held out (the stand's held-out (size, delta) cells) yields
    # no development instance at all: hand its quota round-robin to the cells that did produce some.
    shortfall = per_cell_dev * len(cells) - sum(w.got for w in dev_walk)
    stats['dev_redistributed'] = 0
    productive = [i for i, w in enumerate(dev_walk) if w.got > 0]
    while shortfall > 0 and productive:
        for i in list(productive):
            if shortfall <= 0:
                break
            w = dev_walk[i]
            instance = None
            while instance is None and w.attempts < max_attempts_per_cell:
                instance = w.step(try_mint(w.seed, cells[i], dev_pools, want_holdout=False, layout=w.kitchen()))
            if instance is None:
                productive.remove(i)
                continue
            accepted.append(instance)
            stats['cells'][i]['dev'] += 1
            stats['cells'][i]['dev_attempts'] = w.attempts
            stats['dev_redistributed'] += 1
            shortfall -= 1
    # Development-only levels: enough that they are `dev_only_fraction` of the development split.
    dev_got = sum(c['dev'] for c in stats['cells'])
    extra_cells = dev_only_cells(task)
    fraction = float(dev_only_fraction or 0.0)
    wanted = int(round(dev_got * fraction / (1.0 - fraction))) if 0.0 < fraction < 1.0 else 0
    for cj, (cell, quota) in enumerate(zip(extra_cells, even_quotas(wanted, len(extra_cells)))):
        w = CellWalk(first_seed + DEV_ONLY_SEED_OFFSET + cj * CELL_SEED_STRIDE, cj, quota, dev_pools)
        while w.got < quota and w.attempts < max_attempts_per_cell:
            instance = w.step(try_mint(w.seed, cell, dev_pools, want_holdout=False, layout=w.kitchen()))
            if instance is not None:
                instance['dev_only'] = True
                accepted.append(instance)
        stats['dev_only_cells'].append(dict(cell=cell, dev=w.got, quota=quota, shortfall=quota - w.got,
                                            dev_attempts=w.attempts))
        stats['dev_only_total'] += w.got
    # Evaluation: a fixed quota per cell, each cell walked on its own seed counter.
    quotas = even_quotas(eval_total, len(cells))
    eval_got = 0
    walk = [CellWalk(first_seed + EVAL_SEED_OFFSET + i * CELL_SEED_STRIDE, i, quotas[i], eval_pools)
            for i in range(len(cells))]
    for i, (cell, quota) in enumerate(zip(cells, quotas)):
        w = walk[i]

        def on_eval(instance, i=i):
            nonlocal eval_got
            accepted.append(instance)
            stats['cells'][i]['eval'] = stats['cells'][i].get('eval', 0) + 1
            eval_got += 1
        fill(w, cell, eval_pools, True, quota, eval_mix, on_eval)
        stats['cells'][i]['eval_quota'] = quota
        if w.got < quota:
            stats['eval_skipped_cells'].append(dict(cell=cell, quota=quota, eval=w.got, shortfall=quota - w.got,
                                                    eval_attempts=w.attempts))
            if log:
                log(f'eval cell {cell}: {w.got}/{quota} after {w.attempts} seeds')
    # Cells without a held-out structure class (a task whose hold-out is a subset of its factor cells,
    # like the stand) can never fill their quota: hand the shortfall round-robin to the cells that did
    # produce evaluation instances, so the total is met and stays balanced over the productive cells.
    shortfall = sum(quotas) - eval_got
    stats['eval_redistributed'] = 0
    productive = [i for i, w in enumerate(walk) if w.got > 0]
    while shortfall > 0 and productive:
        for i in list(productive):
            if shortfall <= 0:
                break
            w = walk[i]
            instance = None
            while instance is None and w.attempts < max_attempts_per_cell:
                instance = w.step(try_mint(w.seed, cells[i], eval_pools, want_holdout=True, layout=w.kitchen()))
            if instance is None:
                productive.remove(i)
                continue
            accepted.append(instance)
            stats['cells'][i]['eval'] = stats['cells'][i].get('eval', 0) + 1
            stats['eval_redistributed'] += 1
            eval_got += 1
            shortfall -= 1
    for i, w in enumerate(walk):
        stats['cells'][i]['eval_attempts'] = w.attempts
    stats['eval_total'] = eval_got
    stats['eval_wanted'] = sum(quotas)
    stats['eval_quotas'] = quotas
    stats['holdout_keys'] = sorted(holdout_keys)
    stats['holdout_predicate'] = predicate is not None
    stats['dev_mix'] = [[dict(o), int(n)] for o, n in (dev_mix or ())]
    stats['eval_mix'] = [[dict(o), int(n)] for o, n in (eval_mix or ())]
    return accepted, stats


def write_registry(out_dir, instances, summary=None):
    out = Path(out_dir)
    (out / 'instances').mkdir(parents=True, exist_ok=True)
    rows = []
    for row in instances:
        path = out / 'instances' / f"{row['instance_id']}.json"
        path.write_text(canonical_json(row) + '\n')
        rows.append({k: v for k, v in row.items() if k != 'spec'})
    payload = dict(summary=dict(summary or {}), instances=rows)
    (out / 'registry.json').write_text(json.dumps(payload, indent=1, sort_keys=True, default=_jsonable) + '\n')
    return out / 'registry.json'


def load_registry(out_dir):
    return json.loads((Path(out_dir) / 'registry.json').read_text())


def load_instance(source):
    """An instance dict, a path to one, or a registry directory plus id ('dir#id')."""
    if isinstance(source, dict):
        return deepcopy(source)
    text = str(source)
    if '#' in text:
        directory, identifier = text.rsplit('#', 1)
        source = Path(directory) / 'instances' / f'{identifier}.json'
    path = Path(source)
    if not path.is_file():
        raise ValueError(f'instance file not found: {path}')
    instance = json.loads(path.read_text())
    for key in ('task', 'spec', 'layout_id', 'style_id', 'seed', 'instance_id', 'generator_version'):
        if key not in instance:
            raise ValueError(f'instance file {path} lacks {key!r}')
    return instance


def public_fields(instance):
    """What a public catalog may show: identity and split, never the spec or factors."""
    return dict(task=instance['task'], instance_id=instance['instance_id'], split=instance['split'],
                contract_version=instance['contract_version'], goal=instance.get('goal'))
