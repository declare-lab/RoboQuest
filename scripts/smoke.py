#!/usr/bin/env python3
"""Smoke test for a RoboQuest simulator runtime: build registry instances headless, render the cameras, run one oracle.

Run through scripts/run.sh so the pinned interpreter and PYTHONPATH are used::

    scripts/run.sh scripts/smoke.py                         # every release task builds and steps (CPU only)
    scripts/run.sh scripts/smoke.py --tasks puzzle_box      # one task
    scripts/run.sh scripts/smoke.py --render smoke.png --gpu 0   # three policy cameras at reset (EGL)
    scripts/run.sh scripts/smoke.py --oracle puzzle_box --option probe=true   # one full rollout, headless
    scripts/run.sh scripts/smoke.py --tasks search_room --every-layout        # one instance per kitchen layout

Without --ids the first instance of the task's newest rollout list (suite/v1/rollout_lists/<date>/<task>.json)
is used, else the first development instance of the registry. Exit status is non-zero when any check fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RELEASE_TASKS = ('puzzle_box', 'wobbly_stand', 'odd_parcel', 'search_room', 'stamps',
                 'marked_mugs', 'painted_cubes', 'unfamiliar_containers')


def registry_dir(root, task):
    return Path(root) / task / 'registry'


def registry_rows(root, task):
    return json.loads((registry_dir(root, task) / 'registry.json').read_text())['instances']


def default_instance_id(root, task):
    lists = sorted((Path(root) / 'rollout_lists').glob(f'*/{task}.json'))
    if lists:
        manifest = json.loads(lists[-1].read_text())
        if manifest.get('instances'):
            return manifest['instances'][0]['instance_id']
    rows = registry_rows(root, task)
    dev = [r for r in rows if r.get('split') == 'dev'] or rows
    return dev[0]['instance_id']


def one_per_layout(root, task):
    seen = {}
    for row in registry_rows(root, task):
        seen.setdefault(row['layout_id'], row['instance_id'])
    return [seen[k] for k in sorted(seen)]


def load(root, task, instance_id):
    from roboquest import registry
    return registry.load_instance(f'{registry_dir(root, task)}#{instance_id}')


def build_check(root, task, instance_id, steps):
    import numpy as np
    from roboquest.kitchen import make_env
    from roboquest.manifest import task_class
    started = time.monotonic()
    instance = load(root, task, instance_id)
    env = make_env(task_class(task), instance, render=False, horizon=max(steps + 10, 50))
    try:
        env.reset()
        for _ in range(steps):
            env.step(np.zeros(env.action_dim))
        score = env.evaluate_success()
    finally:
        env.close()
    return dict(task=task, instance_id=instance_id, layout=instance['layout_id'], style=instance['style_id'],
                split=instance.get('split'), score_at_start=score, seconds=round(time.monotonic() - started, 1))


def render_check(root, task, instance_id, out, gpu, tile):
    from roboquest.harness.egl import configure_egl
    mapping = configure_egl(gpu)                      # before MuJoCo is imported
    import numpy as np
    from PIL import Image
    from roboquest.kitchen import CAMERAS, make_env
    from roboquest.manifest import task_class
    started = time.monotonic()
    instance = load(root, task, instance_id)
    env = make_env(task_class(task), instance, image_size=tile, gpu=mapping['egl_index'], horizon=50)
    try:
        obs = env.reset()
        images = [np.ascontiguousarray(obs[camera + '_image'][::-1]) for camera in CAMERAS]
    finally:
        env.close()
    Image.fromarray(np.hstack(images)).save(out)
    return dict(task=task, instance_id=instance_id, out=str(out), cameras=list(CAMERAS), egl=mapping,
                seconds=round(time.monotonic() - started, 1))


def oracle_check(root, task, instance_id, options, horizon):
    from roboquest.kitchen import make_env
    from roboquest.manifest import task_class
    from roboquest.motor import AttemptFailed, Skills, SkillFailure, Submitted
    raise ImportError('The scripted oracles are not part of the RoboQuest release')
    started = time.monotonic()
    instance = load(root, task, instance_id)
    env = make_env(task_class(task), instance, render=False, horizon=horizon)
    result = dict(task=task, instance_id=instance_id, options=options)
    try:
        env.reset()
        skills = Skills(env)
        try:
            load_oracle(task)(env, instance, skills, **options)
            result['outcome'] = 'returned'
        except Submitted:
            result['outcome'] = 'submitted'
        except (AttemptFailed, SkillFailure) as error:
            result['outcome'] = f'skill failure: {error!r}'[:300]
        result['success'] = bool(env.evaluate_success())
        result['ticks'] = skills.summary().get('ticks')
    finally:
        env.close()
    result['seconds'] = round(time.monotonic() - started, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--registry-root', default='suite/v1', help='directory holding <task>/registry (default suite/v1)')
    parser.add_argument('--tasks', nargs='*', default=None, help='tasks to build (default: the eight release tasks)')
    parser.add_argument('--ids', nargs='*', default=[], metavar='TASK=ID', help='instance to use for a task')
    parser.add_argument('--every-layout', action='store_true', help='build one instance per kitchen layout of each task')
    parser.add_argument('--steps', type=int, default=20, help='zero-action steps after reset (default 20)')
    parser.add_argument('--render', type=Path, help='write the three policy cameras of the first task to this PNG')
    parser.add_argument('--gpu', type=int, default=0, help='physical NVIDIA GPU for --render (nvidia-smi index)')
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--oracle', help='also run this task\'s oracle once, headless, on its instance')
    parser.add_argument('--option', action='append', default=[], help='oracle option name=json (with --oracle)')
    parser.add_argument('--horizon', type=int, default=20000)
    args = parser.parse_args()

    tasks = list(args.tasks) if args.tasks else list(RELEASE_TASKS)
    if args.oracle and args.oracle not in tasks:
        tasks.append(args.oracle)
    chosen = dict(item.split('=', 1) for item in args.ids)
    ids = {task: chosen.get(task) or default_instance_id(args.registry_root, task) for task in tasks}
    failures = 0

    for task in tasks:
        builds = one_per_layout(args.registry_root, task) if args.every_layout else [ids[task]]
        for instance_id in builds:
            try:
                print('BUILD OK', json.dumps(build_check(args.registry_root, task, instance_id, args.steps)), flush=True)
            except Exception:  # noqa: BLE001
                failures += 1
                print(f'BUILD FAILED {task} {instance_id}', flush=True)
                traceback.print_exc()

    if args.oracle:
        options = {}
        for item in args.option:
            name, value = item.split('=', 1)
            try:
                options[name] = json.loads(value)
            except json.JSONDecodeError:
                options[name] = value
        try:
            report = oracle_check(args.registry_root, args.oracle, ids[args.oracle], options, args.horizon)
            print('ORACLE', json.dumps(report), flush=True)
            if not report.get('success'):
                failures += 1
        except Exception:  # noqa: BLE001
            failures += 1
            print(f'ORACLE FAILED {args.oracle} {ids[args.oracle]}', flush=True)
            traceback.print_exc()

    if args.render:
        # last, because selecting the EGL device must happen before MuJoCo's renderer is first imported
        task = tasks[0]
        try:
            print('RENDER OK', json.dumps(render_check(args.registry_root, task, ids[task], args.render, args.gpu, args.tile)),
                  flush=True)
        except Exception:  # noqa: BLE001
            failures += 1
            print(f'RENDER FAILED {task} {ids[task]}', flush=True)
            traceback.print_exc()

    print('ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
