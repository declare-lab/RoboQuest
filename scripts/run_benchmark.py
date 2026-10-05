"""Screen RoboQuest task difficulty with an API policy on gate-verified development instances.

Each episode is one ``scripts/run_episode.sh --agent api`` subprocess on ``roboquest_<task>@<instance_id>``.
Instances come from the suite registry's ``--split`` (development by default; ``eval`` and ``all`` run the
episodes with ``--evaluation-instances``, the only way the runner binds an evaluation instance) with G1 and
G2 passed and, where the task has a render check, G3 passed; they are spread over distinct split keys.
``--per-task N`` takes N instances per task; ``--per-cell N`` takes N per factor cell instead, so every
level combination of the task's design is screened; ``--all`` takes every eligible instance, ordered
round-robin over the cells so that any prefix of the plan stays balanced; ``--factors key=value`` restricts
the pool first, ``--exclude-file`` names instance ids to leave out and ``--defer-file`` names ids to run last
within the plan. An episode's horizon is its task's
``BUDGET_TICKS`` unless ``--horizon`` overrides it. Episodes run in parallel across the given GPUs, because
the provider round trip, not physics, bounds the wall time; an integer written to ``--out/max_parallel``
changes the concurrency of a running launch, and one written to ``--out/max_started`` caps the episodes of
the plan the launch may have started (scored earlier, finished or running): at the cap nothing new starts
and the launch waits for the file to change or go away, so several lanes can be held at a common count.
``--resume`` reuses the plan in ``--out/launch.json`` and
reruns only the episodes without a scored result. ``--summarize`` folds the result files under ``--out``
into a per-task table (``summary.md`` / ``summary.json``). Models connect as in
``scripts/run_episode.py`` (README.md, Connecting a model).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboquest.catalog import registry_dir  # noqa: E402
from roboquest.manifest import TASKS  # noqa: E402

DEFAULT_MODEL = 'openai/gpt-6-astra'
OPENROUTER_URL = 'https://openrouter.ai/api/v1'
SPLITS = ('dev', 'eval', 'all')
INSTANCE_ID = re.compile(r'^[0-9a-f]{16}$')
# Sampled per instance to vary the kitchen, not levels of the task design: they never form cells.
NUISANCE_FACTORS = ('kitchen_layout', 'kitchen_style')


def verified_instances(task, root=None, split='dev'):
    """Rows of ``split`` (``dev``, ``eval`` or ``all``) whose gate report passed G1 and G2 and did not fail
    G3 (None: no render check), in registry order."""
    reg = registry_dir(task, root)
    try:
        payload = json.loads((reg / 'registry.json').read_text())
    except OSError:
        return []
    try:
        reports = {r['instance_id']: r for r in json.loads((reg / 'gates.json').read_text()).get('reports', [])}
    except OSError:
        reports = {}
    rows = []
    for row in payload['instances']:
        report = reports.get(row['instance_id'])
        if (split != 'all' and row.get('split') != split) or report is None:
            continue
        if report.get('g1') is not True or report.get('g2') is not True or report.get('g3') is False:
            continue
        rows.append(dict(task=task, instance_id=row['instance_id'], split=row.get('split'),
                         split_key=row.get('split_key'), layout=row.get('layout_id'), style=row.get('style_id'),
                         g3=report.get('g3'), factors=row.get('factors') or {}))
    return rows


def verified_dev_instances(task, root=None):
    """Development rows whose gate report passed G1 and G2 and did not fail G3 (None: no render check)."""
    return verified_instances(task, root, 'dev')


def excluded_ids(paths):
    """Instance ids named in exclusion files: every 16-hex-digit token outside ``#`` comments. Other tokens
    (the task names written in front of the ids) are ignored, so the build's lists are read as they are."""
    ids = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            ids.update(token for token in line.split('#', 1)[0].split() if INSTANCE_ID.match(token))
    return ids


def task_budget(task):
    """The task class ``BUDGET_TICKS``: an episode's horizon unless ``--horizon`` overrides it."""
    from roboquest.manifest import task_class
    return int(task_class(task).BUDGET_TICKS)


def spread(rows, count, keys=('split_key',)):
    """Take ``count`` rows covering as many distinct values of ``keys`` as possible, in registry order.

    One pass per key: the first covers as many split keys as the count allows, each further pass
    fills the remainder with rows whose value of the next key has not been taken yet, and whatever
    is still missing is taken in registry order. With the default single key this is the split-key
    spread used by ``--per-task``; ``--per-cell`` also spreads over layouts inside a cell.
    """
    chosen = []
    for key in keys:
        seen = {row[key] for row in chosen}
        for row in rows:
            if len(chosen) >= count:
                break
            if row[key] not in seen and row not in chosen:
                chosen.append(row)
                seen.add(row[key])
    for row in rows:
        if len(chosen) >= count:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def matches_factors(row, factors):
    """Keep rows whose factor values equal the requested ones; tasks without that factor are unaffected."""
    return all(str(row['factors'].get(key, value)) == value for key, value in factors.items())


def task_cells(task):
    """Every factor cell of a task: the Cartesian product of its class ``FACTORS``, in ``FACTORS`` order.

    The task class is imported through the manifest, so the levels are the design's own rather than a
    copy kept here. The nuisance factors (kitchen layout and style) vary the scene rather than the
    task, and are left out of the cells exactly as they are when a registry is minted.
    """
    factors = cell_factors(task)
    return [dict(zip(factors, values)) for values in product(*factors.values())]


def cell_factors(task):
    """The task class ``FACTORS`` without the nuisance factors, in ``FACTORS`` order."""
    from roboquest.manifest import task_class
    return {name: levels for name, levels in task_class(task).FACTORS.items() if name not in NUISANCE_FACTORS}


def cell_of(row, names):
    """The row's level for each cell factor, in ``names`` order; nuisance factors are ignored."""
    factors = row.get('factors') or {}
    return tuple(factors.get(name) for name in names)


def pick_per_cell(rows, cells, count):
    """``count`` verified instances for every cell, spread over split keys and layouts within it.

    Returns ``(chosen, short)``. A cell with too few verified instances contributes what it has and
    is listed in ``short``: a thin cell is a fact about the registry to report, not a reason to fail.
    """
    names = list(cells[0]) if cells else []
    chosen, short = [], []
    for cell in cells:
        wanted = tuple(cell[name] for name in names)
        picked = spread([row for row in rows if cell_of(row, names) == wanted], count,
                        keys=('split_key', 'layout'))
        for row in picked:
            row['cell'] = dict(cell)
        chosen += picked
        if len(picked) < count:
            short.append(dict(cell=dict(cell), verified=len(picked), wanted=count))
    return chosen, short


def interleave_cells(rows, names):
    """Every row, ordered round-robin over the cells present, so that any prefix of the plan covers the
    cells as evenly as the registry allows; inside a cell the spread order (split keys, then layouts) is
    kept, so the first round is the pick ``--per-cell 1`` would make. Each row gets its ``cell``.
    """
    buckets = {}
    for row in rows:
        key = cell_of(row, names)
        row['cell'] = dict(zip(names, key))
        buckets.setdefault(key, []).append(row)
    queues = [spread(bucket, len(bucket), keys=('split_key', 'layout')) for bucket in buckets.values()]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def agent_deps_dir():
    configured = os.environ.get('ROBOQUEST_AGENT_DEPS')
    candidates = [Path(configured)] if configured else []
    candidates.append(ROOT / 'artifacts' / 'agent-deps')
    candidates += [base / '.worktrees' / 'agent-interface' / 'artifacts' / 'agent-deps'
                   for base in (ROOT, *ROOT.parents)]
    for path in candidates:
        if path.is_dir():
            return path
    raise SystemExit('Provider dependencies not found; set ROBOQUEST_AGENT_DEPS')


def episode_command(args, episode):
    command = ['bash', str(ROOT / 'scripts' / 'run_episode.sh'),
               '--task', f"roboquest_{episode['task']}@{episode['instance_id']}", '--agent', args.agent,
               '--physical-gpu', str(episode['gpu']),
               '--horizon', str(episode.get('horizon') or args.horizon), '--max-decisions', str(args.max_decisions),
               '--max-request-attempts', str(args.max_request_attempts),
               '--request-timeout-s', str(args.request_timeout_s), '--out', episode['out']]
    if args.agent == 'policy':
        # a policy server instead of a model API: no decision cap or request options, the task's tick budget
        for flag in ('--max-decisions', '--max-request-attempts', '--request-timeout-s'):
            at = command.index(flag)
            del command[at:at + 2]
        command += ['--policy-url', args.policy_url, '--policy-seed', str(args.policy_seed),
                    '--action-filter', args.action_filter]
        if args.replan_every is not None:
            command += ['--replan-every', str(args.replan_every)]
        if args.require_subtask:
            command.append('--require-subtask')
    else:
        command += ['--model', args.model]
    if args.max_attempts is not None:
        command += ['--max-attempts', str(args.max_attempts)]
    for flag, value in (('--effort', args.effort), ('--max-output-tokens', args.max_output_tokens)):
        if value is not None:
            command += [flag, str(value)]
    if args.retry_backoff_s is not None:     # a provider with per-minute limits: wait before retrying a 429
        command += ['--retry-backoff-s', str(args.retry_backoff_s)]
    if args.honor_retry_after:
        command.append('--honor-retry-after')
    if args.seed is not None:     # by default every registry instance keeps its own seed (the gated scene)
        command += ['--seed', str(args.seed)]
    if args.split != 'dev':       # the runner binds evaluation instances only when told so explicitly
        command.append('--evaluation-instances')
    if args.offline:
        command.append('--offline')
    else:
        for flag, value in (('--base-url', args.base_url), ('--env-file', args.env_file),
                            ('--api-key-env', args.api_key_env)):
            if value is not None:     # otherwise run_episode.py picks the default; a key may come from the environment
                command += [flag, str(value)]
    return command


def plan_episodes(args):
    episodes, unfilled, excluded = [], [], {}
    for task in args.tasks:
        pool = [r for r in verified_instances(task, args.suite_dir, args.split) if matches_factors(r, args.factors)]
        if args.included_ids is not None:      # a release tranche: only the ids the certificate pass validated
            pool = [r for r in pool if r['instance_id'] in args.included_ids]
            print(f'{task}: {len(pool)} of the {len(args.included_ids)} included ids are eligible', file=sys.stderr)
        rows = [r for r in pool if r['instance_id'] not in args.excluded_ids]
        if len(rows) < len(pool):
            excluded[task] = sorted(r['instance_id'] for r in pool if r['instance_id'] in args.excluded_ids)
            print(f'{task}: {len(excluded[task])} of {len(pool)} eligible instances excluded', file=sys.stderr)
        if args.all:
            chosen = interleave_cells(rows, list(cell_factors(task)))
            cells_present = {json.dumps(row['cell'], sort_keys=True) for row in chosen}
            print(f'{task}: all {len(chosen)} eligible {args.split} instances over {len(cells_present)} cells',
                  file=sys.stderr)
        elif args.per_cell:
            # a --factors level fixes that factor, so only the cells at that level are asked for
            cells = [cell for cell in task_cells(task) if matches_factors(dict(factors=cell), args.factors)]
            chosen, short = pick_per_cell(rows, cells, args.per_cell)
            for gap in short:
                print(f"warning: {task} cell {gap['cell']} has {gap['verified']} of "
                      f"{gap['wanted']} verified development instances", file=sys.stderr)
            print(f'{task}: {len(chosen)} instances over {len(cells)} cells '
                  f'({len(short)} short)', file=sys.stderr)
            unfilled += [dict(gap, task=task) for gap in short]
        else:
            chosen = spread(rows, args.per_task)
            if len(chosen) < args.per_task:
                print(f'warning: {task} has only {len(chosen)} verified development instances', file=sys.stderr)
        horizon = args.horizon if args.horizon is not None else task_budget(task)
        for row in chosen:
            row['horizon'] = horizon
        episodes.extend(chosen)
    args.unfilled_cells = unfilled
    args.excluded = excluded
    for index, episode in enumerate(episodes):
        episode['gpu'] = args.gpus[index % len(args.gpus)]
        episode['out'] = str(args.out / episode['task'] / episode['instance_id'])
        episode['log'] = str(args.out / episode['task'] / (episode['instance_id'] + '.log'))
    return episodes


def read_result(out):
    try:
        result = json.loads((Path(out) / 'result.json').read_text())
    except (OSError, ValueError):
        return None
    score = result.get('score') or {}
    return dict(completed=result.get('completed'), termination=result.get('termination'),
                success=score.get('success'), submitted=score.get('submitted'),
                decisions=result.get('successful_model_decisions'), steps=result.get('physical_steps'),
                wall_min=round((result.get('total_wall_s') or 0.) / 60., 1),
                generation_min=round((result.get('generation_wall_s') or 0.) / 60., 1),
                returned_models=result.get('returned_model_ids'))


def summarize(args):
    launch = json.loads((args.out / 'launch.json').read_text())
    try:
        exit_codes = {r['out']: r['exit_code'] for r in json.loads((args.out / 'status.json').read_text())['finished']}
    except (OSError, ValueError, KeyError):
        exit_codes = {}
    rows = [dict(episode, exit_code=exit_codes.get(episode['out']),
                 **(read_result(episode['out']) or dict(completed=None))) for episode in launch['episodes']]
    for r in rows:   # progress, written by the runner beside result.json
        try:
            r['progress_v2'] = json.loads((Path(r['out']) / 'task-progress.json').read_text()).get('P')
        except (OSError, ValueError, TypeError):
            r['progress_v2'] = None
    per_task = {}
    for task in dict.fromkeys(r['task'] for r in rows):
        done = [r for r in rows if r['task'] == task and r.get('completed') is not None]
        per_task[task] = dict(
            episodes=sum(1 for r in rows if r['task'] == task), finished=len(done),
            unscored=sum(1 for r in rows if r['task'] == task and r.get('completed') is None
                         and r.get('exit_code') is not None),
            success=sum(1 for r in done if r.get('success') is True),
            mean_progress=(round(sum(r['progress_v2'] for r in done if r.get('progress_v2') is not None)
                                    / max(1, sum(1 for r in done if r.get('progress_v2') is not None)), 3)
                              if done else None),
            submitted=sum(1 for r in done if r.get('submitted') is True),
            terminations=dict(Counter(str(r.get('termination')) for r in done)),
            mean_decisions=round(sum(r.get('decisions') or 0 for r in done) / len(done), 1) if done else None,
            mean_wall_min=round(sum(r.get('wall_min') or 0 for r in done) / len(done), 1) if done else None)
    summary = dict(model=launch['model'], offline=launch.get('offline', False), out=str(args.out),
                   per_task=per_task, episodes=rows, summarized_utc=datetime.now(timezone.utc).isoformat())
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['| task | episodes | finished | unscored | success | mean progress | submitted | mean decisions | mean wall min | terminations |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for task, s in per_task.items():
        lines.append(f"| {task} | {s['episodes']} | {s['finished']} | {s['unscored']} | {s['success']} | {s['mean_progress']} | "
                     f"{s['submitted']} | {s['mean_decisions']} | {s['mean_wall_min']} | {s['terminations']} |")
    table = '\n'.join(lines)
    (args.out / 'summary.md').write_text(f"# API screen: {launch['model']}\n\n{table}\n")
    print(table)
    return summary


def set_aside(path, stamp):
    """Keep the partial evidence of a killed episode next to its fresh rerun."""
    path = Path(path)
    if path.exists():
        path.rename(path.with_name(path.name + '.aborted-' + stamp))


def live_parallel(args):
    """``--max-parallel``, or the integer in ``<out>/max_parallel`` while that file exists: the concurrency
    of a running launch can be changed without restarting it and losing the episodes in flight. ``0`` drains
    the launch: nothing new starts, the running episodes finish, and the launch waits for a new value."""
    try:
        return max(0, int((args.out / 'max_parallel').read_text().strip()))
    except (OSError, ValueError):
        return args.max_parallel


def live_started_cap(args):
    """The integer in ``<out>/max_started`` while that file exists: how many episodes of the plan the launch
    may have started (scored earlier, finished or running). At the cap nothing new starts and the launch
    waits, like a drain, until the file changes or goes away; absent or unreadable means no cap."""
    try:
        return max(0, int((args.out / 'max_started').read_text().strip()))
    except (OSError, ValueError):
        return None


def launch_all(args):
    if args.resume:
        launch = json.loads((args.out / 'launch.json').read_text())
        episodes = launch['episodes']
        for key in ('model', 'seed', 'horizon', 'max_decisions', 'offline'):   # the plan's settings win
            setattr(args, key, launch[key])
        args.split, args.all = launch.get('split', 'dev'), launch.get('all', False)
        if args.max_parallel is None:
            args.max_parallel = launch.get('max_parallel')
        # A narrower plan on resume: unscored episodes whose instance an --exclude-file names, or that an
        # --include-file leaves out, leave the plan (a release withdrawn after the launch); scored ones stay.
        # The plan file records what was pruned, so the run's history stays readable.
        pruned = [e for e in episodes if not (read_result(e['out']) or {}).get('completed')
                  and (e['instance_id'] in args.excluded_ids
                       or (args.included_ids is not None and e['instance_id'] not in args.included_ids))]
        if pruned:
            episodes = [e for e in episodes if e not in pruned]
            launch['episodes'] = episodes
            launch.setdefault('pruned', []).extend(
                dict(task=e['task'], instance_id=e['instance_id'], at=datetime.now(timezone.utc).isoformat())
                for e in pruned)
            print(f'{len(pruned)} unscored episodes pruned from the plan on resume', file=sys.stderr)
    else:
        episodes = plan_episodes(args)
    if not episodes:
        raise SystemExit('No verified instances to run')
    if args.deferred_ids:          # listed instances run last (ids awaiting replacement in a registry)
        episodes = ([e for e in episodes if e['instance_id'] not in args.deferred_ids]
                    + [e for e in episodes if e['instance_id'] in args.deferred_ids])
    deferral = dict(defer_files=[str(path) for path in args.defer_file],
                    deferred=[e['instance_id'] for e in episodes if e['instance_id'] in args.deferred_ids])
    if args.max_parallel is None:      # --per-cell: the cell count is only known once the plan exists
        args.max_parallel = 8 if args.all else max(1, len(episodes))
    if args.dry_run:
        for episode in episodes:
            print(json.dumps(dict(episode, command=' '.join(episode_command(args, episode)))))
        return 0
    environment = dict(os.environ, ROBOQUEST_AGENT_DEPS=str(agent_deps_dir()))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if args.resume:
        launch.setdefault('resumed_utc', []).append(datetime.now(timezone.utc).isoformat())
        if args.deferred_ids:      # a deferral given at resume reorders the plan's episodes the same way
            launch.update(episodes=episodes, **deferral)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        launch = dict(model=args.model, offline=args.offline, seed=args.seed, horizon=args.horizon,
                      max_decisions=args.max_decisions, max_parallel=args.max_parallel, gpus=args.gpus,
                      split=args.split, all=args.all, per_task=args.per_task, per_cell=args.per_cell,
                      factors=args.factors, exclude_files=[str(path) for path in args.exclude_file],
                      include_files=[str(path) for path in args.include_file],
                      included=sorted(args.included_ids) if args.included_ids is not None else None,
                      excluded=getattr(args, 'excluded', {}), unfilled_cells=getattr(args, 'unfilled_cells', []),
                      retry_backoff_s=args.retry_backoff_s, honor_retry_after=args.honor_retry_after,
                      started_utc=datetime.now(timezone.utc).isoformat(), episodes=episodes, **deferral)
    (args.out / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    pending, running, finished = [], {}, []
    for episode in episodes:
        result = read_result(episode['out'])
        if result is not None and result.get('completed'):   # scored earlier; its exit code is unknown
            finished.append(dict(task=episode['task'], instance_id=episode['instance_id'], out=episode['out'],
                                 exit_code=None, **result))
            continue
        # unscored (killed or provider-interrupted) episodes are rerun
        set_aside(episode['out'], stamp)
        set_aside(episode['log'], stamp)
        pending.append(episode)

    def stop_children(signum, frame):
        for process, episode in running.values():
            process.terminate()
        raise SystemExit(f'stopped by signal {signum}; {len(running)} episodes terminated')
    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    def write_status():
        status = dict(running=[e['out'] for _, e in running.values()], finished=finished,
                      pending=[e['out'] for e in pending], max_parallel=args.live_parallel,
                      max_started=args.live_started_cap, updated_utc=datetime.now(timezone.utc).isoformat())
        (args.out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')

    args.live_parallel, args.live_started_cap, held = args.max_parallel, None, False
    while pending or running:
        wanted = live_parallel(args)
        if wanted != args.live_parallel:
            print(json.dumps(dict(event='max_parallel', value=wanted, was=args.live_parallel)), flush=True)
            args.live_parallel = wanted
        cap = live_started_cap(args)
        if cap != args.live_started_cap:
            print(json.dumps(dict(event='max_started', value=cap, was=args.live_started_cap)), flush=True)
            args.live_started_cap = cap
        while pending and len(running) < args.live_parallel and (cap is None or len(finished) + len(running) < cap):
            episode = pending.pop(0)
            Path(episode['log']).parent.mkdir(parents=True, exist_ok=True)
            with open(episode['log'], 'w') as log:
                process = subprocess.Popen(episode_command(args, episode), cwd=str(ROOT), env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
            episode['pid'] = process.pid
            running[process.pid] = (process, episode)
            print(json.dumps(dict(event='started', task=episode['task'], instance=episode['instance_id'],
                                  gpu=episode['gpu'], pid=process.pid)), flush=True)
            write_status()
            time.sleep(args.stagger_s)
        for pid, (process, episode) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[pid]
            record = dict(task=episode['task'], instance_id=episode['instance_id'], out=episode['out'],
                          exit_code=code, **(read_result(episode['out']) or dict(completed=None)))
            finished.append(record)
            print(json.dumps(dict(event='finished', **record)), flush=True)
        write_status()
        holding = bool(pending) and not running and cap is not None and len(finished) >= cap
        if holding and not held:      # at the cap with nothing running: the launch waits for the file to change
            print(json.dumps(dict(event='held', max_started=cap, started=len(finished), pending=len(pending))), flush=True)
        held = holding
        if running or pending:        # a drained or held launch waits too, instead of spinning
            time.sleep(args.poll_s)
    summarize(args)
    return 1 if any(record['exit_code'] not in (0, None) for record in finished) else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', default=','.join(TASKS), help='Comma-separated manifest task names')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--per-task', type=int, help='Verified development instances per task (default 3)')
    selection.add_argument('--per-cell', type=int, help='Verified development instances per factor cell: the '
                           'Cartesian product of the task class FACTORS, kitchen layout and style excluded')
    selection.add_argument('--all', action='store_true', help='Every eligible instance of the split, ordered '
                           'round-robin over the factor cells')
    parser.add_argument('--split', choices=SPLITS, default='dev', help='Registry split to draw from; eval and '
                        'all run the episodes with --evaluation-instances')
    parser.add_argument('--factors', default='', help='Comma-separated key=value instance factors to require, '
                        'e.g. cover=none,occluders=off; tasks without that factor are unaffected')
    parser.add_argument('--exclude-file', type=Path, action='append', default=[], help='File naming instance '
                        'ids to leave out (16-hex-digit tokens, # comments); repeatable')
    parser.add_argument('--include-file', type=Path, action='append', default=[], help='File naming the only '
                        'instance ids to draw from (same format), e.g. a validated release tranche; repeatable')
    parser.add_argument('--defer-file', type=Path, action='append', default=[], help='File naming instance ids '
                        'to run last within the plan (same format), e.g. ids awaiting replacement; repeatable')
    parser.add_argument('--model', help='Model to evaluate, <provider>/<model id> (scripts/run_episode.py --help)')
    parser.add_argument('--effort', help='Reasoning effort to request (passed to run_episode.py)')
    parser.add_argument('--max-output-tokens', type=int, help='Output token limit (passed to run_episode.py)')
    parser.add_argument('--agent', choices=('api', 'policy'), default='api', help='api: a model through its '
                        'API (--model); policy: a robot policy served over the openpi WebSocket protocol')
    parser.add_argument('--policy-url', help='--agent policy: the policy server, e.g. ws://localhost:8000')
    parser.add_argument('--policy-seed', type=int, default=0, help='--agent policy: evaluation seed')
    parser.add_argument('--replan-every', type=int, help='--agent policy: actions executed per chunk')
    parser.add_argument('--require-subtask', action='store_true', help='--agent policy: require predicted subtasks')
    parser.add_argument('--action-filter', choices=('none', 'demos'), default='none',
                        help='--agent policy: action handling (see scripts/run_episode.py --help)')
    parser.add_argument('--set', choices=('eval50',), help='eval50: the benchmark, the 500 evaluation '
                        'instances (50 per task, suite/v1/eval50); --tasks narrows it')
    parser.add_argument('--base-url', help='Endpoint of an openai-compatible/ model')
    parser.add_argument('--env-file', type=Path, help='Private KEY=VALUE file holding the provider key')
    parser.add_argument('--api-key-env', help='Variable holding the key (default: the provider\'s)')
    parser.add_argument('--gpus', default='0,1', help='Physical GPU ids used round-robin')
    parser.add_argument('--max-parallel', type=int, help='Concurrent episodes; default: all of them')
    parser.add_argument('--seed', type=int, help='Scene seed override; default: each registry instance keeps its '
                        'own seed, the scene its gates and certificates were run on')
    parser.add_argument('--horizon', type=int, help="Physical ticks at 20 Hz; default for API models "
                        "1000000 (no tick budget: the 200-decision cap ends an episode), the benchmark's "
                        "setting, and 36000 (30 simulated minutes) for --agent policy. Pass 0 "
                        "for each task's provisional BUDGET_TICKS")
    parser.add_argument('--max-decisions', type=int, default=200, help='Model decisions per episode (200 in the '
                        'benchmark)')
    parser.add_argument('--max-attempts', type=int, help='Optional total provider HTTP attempt cap per episode')
    parser.add_argument('--max-request-attempts', type=int, default=6, help='Immediate retries per frozen request (max 6)')
    parser.add_argument('--retry-backoff-s', type=float, help='Exponential wait before each retry of a transient '
                        'provider error (for providers with per-minute limits); unset retries at once')
    parser.add_argument('--honor-retry-after', action='store_true', help='Wait at least the delay the provider asks '
                        'for before retrying')
    parser.add_argument('--request-timeout-s', type=float, default=600., help='Per provider request (max 600)')
    parser.add_argument('--stagger-s', type=float, default=5., help='Delay between launches; kitchen builds are CPU heavy')
    parser.add_argument('--poll-s', type=float, default=30.)
    parser.add_argument('--suite-dir', type=Path, help='Suite root holding <task>/registry; default suite/v1')
    parser.add_argument('--out', type=Path, help='Run directory (default runs/<model>)')
    parser.add_argument('--offline', action='store_true', help='Mocked provider; a wiring check, not a policy')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan and commands only')
    parser.add_argument('--summarize', action='store_true', help='Only fold existing results under --out')
    parser.add_argument('--resume', action='store_true', help='Reuse the plan in --out/launch.json and rerun only '
                        'the episodes without a scored result; partial evidence is set aside')
    args = parser.parse_args(argv)
    args.tasks = [task for task in args.tasks.split(',') if task]
    unknown = [task for task in args.tasks if task not in TASKS]
    if unknown:
        parser.error('Unknown tasks: ' + ', '.join(unknown))
    if args.agent == 'policy':
        if not args.policy_url and not args.summarize:
            parser.error('--agent policy needs --policy-url')
        if args.model:
            parser.error('--model applies to --agent api')
        args.model = 'policy'   # the run's label: runs/policy unless --out names another folder
    if not args.model and not args.summarize:
        parser.error('--model is required')
    if args.out is None:
        if not args.model:
            parser.error('--summarize needs --out')
        args.out = ROOT / 'runs' / re.sub(r'[^A-Za-z0-9._-]+', '_', args.model)
    if args.set == 'eval50':
        if args.per_task is not None or args.per_cell is not None or args.include_file:
            parser.error('--set eval50 chooses the instances; drop --per-task/--per-cell/--include-file')
        args.split, args.all = 'eval', True
        args.include_file = [ROOT / 'suite/v1/eval50' / f'{task}.txt' for task in args.tasks]
    factors = {}
    for item in [item for item in args.factors.split(',') if item]:
        key, separator, value = item.partition('=')
        if not separator or not key or not value:
            parser.error('--factors items must be key=value')
        factors[key] = value
    args.factors = factors
    try:
        args.gpus = [int(gpu) for gpu in args.gpus.split(',') if gpu != '']
    except ValueError:
        parser.error('--gpus must be integers')
    if not args.gpus or any(gpu < 0 for gpu in args.gpus):
        parser.error('--gpus needs nonnegative ids')
    if args.per_cell is None and args.per_task is None and not args.all:
        args.per_task = 3
    if (args.per_task is not None and args.per_task < 1) or (args.per_cell is not None and args.per_cell < 1):
        parser.error('--per-task and --per-cell must be positive')
    if args.horizon is None:
        # API models: no tick budget, the decision cap ends an episode; policies: one budget for every task
        args.horizon = 36_000 if args.agent == 'policy' else 1_000_000
    if args.horizon == 0:
        args.horizon = None   # each task's BUDGET_TICKS
    if args.max_decisions < 1 or (args.horizon is not None and args.horizon < 1):
        parser.error('--max-decisions and --horizon must be positive')
    missing = [str(path) for path in args.exclude_file + args.include_file if not path.is_file()]
    if missing:
        parser.error('--exclude-file/--include-file not found: ' + ', '.join(missing))
    missing = [str(path) for path in args.defer_file if not path.is_file()]
    if missing:
        parser.error('--defer-file not found: ' + ', '.join(missing))
    args.excluded_ids = excluded_ids(args.exclude_file)
    args.included_ids = excluded_ids(args.include_file) if args.include_file else None
    args.deferred_ids = excluded_ids(args.defer_file)
    if args.include_file and not args.included_ids:
        parser.error('--include-file names no instance ids')
    if not (args.summarize or args.offline or args.dry_run):
        if args.env_file is not None:
            if not args.env_file.is_file():
                parser.error('--env-file not found')
            args.env_file = args.env_file.resolve()
    args.out = args.out.resolve()
    if args.resume and not (args.out / 'launch.json').is_file():
        parser.error('--resume needs an existing --out with launch.json')
    if args.max_parallel is None and args.per_task is not None:
        args.max_parallel = max(1, len(args.tasks) * args.per_task)
    if args.max_parallel is None and args.all:
        args.max_parallel = 8
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error('--max-parallel must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        summarize(args)
        return 0
    return launch_all(args)


if __name__ == '__main__':
    raise SystemExit(main())
