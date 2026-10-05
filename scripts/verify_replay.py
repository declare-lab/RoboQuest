#!/usr/bin/env python3
"""Replay recorded RoboQuest API-evaluation episodes through this checkout's harness and compare.

For an episode directory recorded with this repository (``scripts/run_episode.py`` or a benchmark run) the run is
rebuilt from its ``config.json`` through ``scripts/run_episode.py``'s own argument handling (task and
instance, seed, image size, horizon, decision and retry limits) and executed by
``roboquest.harness.episode_runner.run_episode`` with a single substitution: the provider transport answers
attempt k with the response recorded at attempt k (``provider/wire/trial/episode/calls.jsonl``). Then this
checkout's run is compared with the recorded one:

- every request the harness sends, field by field (system prompt, conversation text, tool schemas,
  generation settings); camera images as pixel differences, where rasterization noise (at most 2 levels on
  under 1% of the pixels) is tolerated and reported;
- the number of requests, and result fields (completed, termination, decisions, attempts, physical steps);
- the score (success, progress, submitted, outcome);
- goal, system prompt and tool schemas; initial state (qpos, qvel, reset actions), task spec; every command
  and the step it ended on; the final physical state (last line of the private trace).

Reported but not failures: keys the current code adds to the task spec with a null value, EGL flicker and
rasterization-sized pixel counts in the reset render check, and differences in that check's verdict
(``render_check_differences``; it is recorded at reset, never raised, and never read by the episode).

    scripts/run_agent.sh scripts/verify_replay.py --one EPISODE_DIR --gpu 1 --out DIR
    scripts/run_agent.sh scripts/verify_replay.py --episodes EP ... --gpu 1 --workers 4 --out DIR

Run through ``scripts/run_agent.sh`` (provider packages and the Inspect Robots source on the path), with
``ROBOQUEST_AGENT_DEPS`` set when the packages are not in ``<runtime>/agent-deps``. Writes one JSON report per
episode and ``summary.json``; exit status 1 when anything differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CALLS = Path('provider/wire/trial/episode/calls.jsonl')
BLOBS = Path('provider/wire/trial/blobs')
TAIL = 8_000_000


def recorded_rows(ep):
    return [json.loads(line) for line in (ep / CALLS).read_text().splitlines() if line.strip()]


class RecordedResponses:
    """An httpx transport that answers attempt k with the k-th recorded attempt's status and body."""

    def __init__(self, rows):
        self.rows, self.index = rows, 0

    def __call__(self, request):
        import httpx
        if self.index >= len(self.rows):
            raise RuntimeError('the harness sent more requests than the recording')
        row = self.rows[self.index]
        self.index += 1
        if row.get('status') is None:
            raise httpx.ConnectError(row.get('error') or 'recorded transport error', request=request)
        body = row.get('response')
        if isinstance(body, (dict, list)):
            return httpx.Response(row['status'], json=body)
        return httpx.Response(row['status'], text=str(body or ''))


def replay_policy_factory(rows):
    """``make_api_policy`` with the provider transport replaced by the recorded responses."""

    def make_replay_policy(args, config, capture):
        import httpx
        from roboquest.harness.agent_runtime import validate_endpoint
        from roboquest.harness.agent import MODEL_CONFIGS, OBSERVATION_TEXT_VERSION, TableSettingPolicy
        wire = MODEL_CONFIGS[args.model]['wire']
        if args.vertex:
            endpoint = 'https://gemini.invalid/v1' if wire == 'gemini' else 'https://vertex.invalid/v1'
            config['vertex_region'] = args.vertex_region
            config['vertex_publisher'] = 'google' if wire == 'gemini' else 'anthropic'
        elif wire == 'gemini':
            endpoint = 'https://gemini.invalid/v1'
        else:
            endpoint = args.base_url
        validate_endpoint(endpoint)
        config['model_check'] = {'method': 'each_response_before_action', 'extra_generation_attempts': 0}
        config['observation_text'] = OBSERVATION_TEXT_VERSION
        policy = TableSettingPolicy(args.model, endpoint, '', transport=httpx.MockTransport(RecordedResponses(rows)),
            capture=capture, scene=args.scene, image_size=args.image_size, max_decisions=args.max_decisions,
            max_attempts=args.max_attempts, max_request_attempts=args.max_request_attempts,
            max_restarts=args.max_restarts, timeout_s=args.request_timeout_s, verify_model_identity=True,
            retry_backoff_s=0., honor_retry_after=False)
        config['http_timeouts_s'] = policy.client._http.timeout.as_dict()
        return policy

    return make_replay_policy


def replay_argv(config, gpu, out):
    model = config['model']
    argv = ['--task', config['task'], '--agent', 'api', '--model', model, '--out', str(out),
            '--physical-gpu', str(gpu), '--image-size', str(config['image_size']),
            '--horizon', str(config['horizon']), '--max-request-attempts', str(config['max_request_attempts']),
            '--max-restarts', str(config['max_restarts']), '--request-timeout-s', str(config['request_timeout_s'])]
    if '@' in str(config['task']):
        argv.append('--evaluation-instances')
    if model.startswith('openai-compatible/'):
        # never contacted: every response comes from the recording (the endpoint does not change a request body)
        argv += ['--base-url', 'https://replay.invalid/v1']
    if config.get('seed') is not None:
        argv += ['--seed', str(config['seed'])]
    if config.get('max_decisions') is not None:
        argv += ['--max-decisions', str(config['max_decisions'])]
    if config.get('max_attempts') is not None:
        argv += ['--max-attempts', str(config['max_attempts'])]
    return argv


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            yield from differences(a.get(key), b.get(key), f'{path}.{key}')
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, f'{path}[{i}]')
    elif a != b:
        yield path, a, b


ABSENT = object()
RENDER_NOISE = __import__('re').compile(r'^\.hidden_state_check\..*(flicker|unstable_pixels|single_pass)')
RENDER_COUNT = __import__('re').compile(r'^\.hidden_state_check\.')


def render_noise(path, replayed, recorded):
    """A difference in the reset render check that rasterization alone explains: flicker/unstable counts (always
    noise by the check's own design), or another pixel count within 4 px or 1%. Verdicts are never noise."""
    if RENDER_NOISE.search(path):
        return True
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (replayed, recorded))
    if RENDER_COUNT.search(path) and numbers:
        return abs(replayed - recorded) <= max(4, .01 * max(abs(replayed), abs(recorded)))
    return False


def spec_paths(a, b, path=''):
    """Every (path, replayed, recorded) that differs; a missing key reads as ABSENT."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            yield from spec_paths(a.get(key, ABSENT), b.get(key, ABSENT), f'{path}.{key}')
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from spec_paths(x, y, f'{path}[{i}]')
    elif a is not b and a != b:
        yield path, a, b


def pixel_difference(ours, theirs):
    """(pixels that differ, largest per-channel difference) between two PNG files."""
    import numpy as np
    from PIL import Image
    try:
        a = np.asarray(Image.open(ours)).astype(int)
        b = np.asarray(Image.open(theirs)).astype(int)
    except OSError:
        return (10 ** 9, 255)
    if a.shape != b.shape:
        return (10 ** 9, 255)
    d = np.abs(a - b)
    return (int((d.sum(-1) > 0).sum()), int(d.max()))


def last_record(path, tail=TAIL):
    with open(path, 'rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - tail))
        data = handle.read()
    for line in reversed([line for line in data.split(b'\n') if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def blob_name(value):
    return str(value).split('$blob:')[-1][:64]


def compare(ep, out, task, report):
    import numpy as np
    problems = report['problems']
    rows, mine = recorded_rows(ep), (recorded_rows(out) if (out / CALLS).is_file() else [])
    report.update(recorded_requests=len(rows), replayed_requests=len(mine))
    if len(mine) != len(rows):
        problems.append(f'{len(mine)} requests sent, {len(rows)} recorded')
    first, images = None, {}
    for k, (a, b) in enumerate(zip(mine, rows)):
        if (a.get('endpoint'), a.get('attempt')) != (b.get('endpoint'), b.get('attempt')) and first is None:
            first = (k, 'endpoint/attempt', f"{a.get('endpoint')}#{a.get('attempt')}", f"{b.get('endpoint')}#{b.get('attempt')}")
        for path, x, y in differences(a.get('request'), b.get('request')):
            if '$blob:' in str(x) and '$blob:' in str(y):
                pair = (blob_name(x), blob_name(y))
                if pair[0] != pair[1] and pair not in images:
                    images[pair] = pixel_difference(out / BLOBS / f'{pair[0]}.png', ep / BLOBS / f'{pair[1]}.png')
            elif first is None:
                first = (k, path, str(x)[:160], str(y)[:160])
    if first is not None:
        report['first_request_difference'] = first
        problems.append(f'request {first[0]} differs at {first[1]}')
    report['images_not_identical'] = len(images)
    report['image_worst'] = max(images.values(), default=(0, 0))
    if any(level > 2 or pixels > .01 * 512 * 512 for pixels, level in images.values()):
        problems.append('a camera image differs beyond rasterization noise')
    for name in ('instruction.txt', 'system-prompt.txt', 'tool-schemas.json'):
        if (out / name).read_text() != (ep / name).read_text():
            problems.append(f'{name} differs')
    ini_a, ini_b = (json.loads((d / 'initial-private.json').read_text()) for d in (out, ep))
    report['initial_qpos_max_abs_diff'] = float(np.max(np.abs(np.asarray(ini_a['qpos']) - np.asarray(ini_b['qpos']))))
    report['initial_qvel_max_abs_diff'] = float(np.max(np.abs(np.asarray(ini_a['qvel']) - np.asarray(ini_b['qvel']))))
    report['initial_rgb_equal'] = ini_a['rgb_sha256'] == ini_b['rgb_sha256']
    if report['initial_qpos_max_abs_diff'] > 0 or report['initial_qvel_max_abs_diff'] > 0:
        problems.append('initial state differs')
    if ini_a['reset_actions'] != ini_b['reset_actions']:
        problems.append('reset actions differ')
    spec_a, spec_b = (json.loads((d / 'task-spec-private.json').read_text()) for d in (out, ep))
    if spec_a != spec_b:
        paths = list(spec_paths(spec_a, spec_b))
        # A key the current code adds with a null value (a newer record schema, e.g. an occluder's ``push``)
        # changes nothing about the scene; any other difference is a real one.
        added_null = [p for p, x, y in paths if x is None and y is ABSENT]
        # The reset render check (hidden_state_check) records how many pixels EGL flickered between repeated
        # renders; that count is renderer noise, reported and never fatal (see its docstring). Its verdicts
        # (changed_pixels, passed, observability) stay compared.
        noise = [(p, x, y) for p, x, y in paths if render_noise(p, x, y)]
        real = [(p, x, y) for p, x, y in paths if not (x is None and y is ABSENT) and not render_noise(p, x, y)]
        # Asset paths name the runtime folder they were recorded in; the same asset file under another runtime
        # (another installation) is the same scene.
        tail = '/robocasa/models/assets/'
        relocated = [(p, x, y) for p, x, y in real if isinstance(x, str) and isinstance(y, str) and tail in x
                     and tail in y and x.split(tail, 1)[1] == y.split(tail, 1)[1]]
        real = [r for r in real if r not in relocated]
        if relocated:
            report['spec_relocated_asset_paths'] = [p for p, x, y in relocated[:10]]
        if added_null:
            report['spec_added_null_keys'] = added_null[:20]
        if noise:
            report['spec_renderer_noise'] = [f'{p}: {x} vs recorded {y}' for p, x, y in noise[:10]]
        # The reset render check is recorded, never raised and never read by the episode; on a marginal instance
        # its counts vary between runs (its own render path, not the policy cameras). A difference there is a
        # warning; everything else in the spec must match.
        check = [(p, x, y) for p, x, y in real if RENDER_COUNT.search(p)]
        real = [(p, x, y) for p, x, y in real if not RENDER_COUNT.search(p)]
        if check:
            report['render_check_differences'] = [f'{p}: {str(x)[:80]} vs recorded {str(y)[:80]}' for p, x, y in check[:10]]
        if real:
            report['spec_differences'] = [f'{p}: {str(x)[:80]} vs recorded {str(y)[:80]}' for p, x, y in real[:10]]
            problems.append('task spec differs')
    cmd_a, cmd_b = (json.loads((d / 'commands.json').read_text()) if (d / 'commands.json').is_file() else []
                    for d in (out, ep))
    report['commands'] = [len(cmd_a), len(cmd_b)]
    for i, (x, y) in enumerate(zip(cmd_a, cmd_b)):
        if x != y:
            report['first_command_difference'] = dict(index=i, replayed=str(x)[:200], recorded=str(y)[:200])
            problems.append(f'command {i} differs')
            break
    if len(cmd_a) != len(cmd_b):
        problems.append(f'{len(cmd_a)} commands replayed, {len(cmd_b)} recorded')
    last_a, last_b = (last_record(d / 'physical-trace-private.jsonl') if (d / 'physical-trace-private.jsonl').is_file()
                      else None for d in (out, ep))
    if last_a is not None and last_b is not None:
        report['final_step'] = [last_a.get('step'), last_b.get('step')]
        if 'qpos' in last_a and 'qpos' in last_b:
            report['final_qpos_max_abs_diff'] = float(np.max(np.abs(np.asarray(last_a['qpos']) - np.asarray(last_b['qpos']))))
        if last_a.get('step') != last_b.get('step') or report.get('final_qpos_max_abs_diff', 0) > 0:
            problems.append('final state differs')
    res_a, res_b = (json.loads((d / 'result.json').read_text()) for d in (out, ep))
    for key in ('completed', 'termination', 'successful_model_decisions', 'generation_attempts', 'physical_steps'):
        if res_a.get(key) != res_b.get(key):
            problems.append(f'{key}: {res_a.get(key)!r} vs recorded {res_b.get(key)!r}')
    ours, theirs = res_a.get('score') or {}, res_b.get('score') or {}
    report['score'] = {k: [ours.get(k), theirs.get(k)] for k in ('success', 'progress', 'submitted', 'outcome')}
    for key in ('success', 'progress', 'submitted', 'outcome'):
        if ours.get(key) != theirs.get(key):
            problems.append(f'score {key}: {ours.get(key)!r} vs recorded {theirs.get(key)!r}')


def replay_one(ep, gpu, out_dir):
    started = time.monotonic()
    ep = Path(ep).resolve()
    config = json.loads((ep / 'config.json').read_text())
    task = config['task'].split('@')[0].removeprefix('roboquest_')
    model_dir, lane = ep.parts[-4], ep.parts[-3]
    label = f'{model_dir}-{lane}-{task}-{ep.name}'
    out = Path(out_dir).resolve() / 'episodes' / label
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(episode=str(ep), label=label, model=model_dir, lane=lane, task=task, instance=ep.name, problems=[])
    source = json.loads((ep / 'source-manifest.json').read_text())
    report['recorded_revision'] = source.get('revision')
    import os
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '1')           # as scripts/run.sh, under which the evaluation lanes ran
    from roboquest.harness.egl import configure_egl
    report['rendering'] = configure_egl(gpu)       # MuJoCo picks its GL backend at import: EGL before any scene loads
    try:
        from scripts.run_episode import parse_args
        import roboquest.harness.episode_runner as runner
        for attempt in (1, 2):
            if out.exists():
                import shutil
                shutil.rmtree(out)
            args = parse_args(replay_argv(config, gpu, out))
            args.vertex_region = config.get('vertex_region', args.vertex_region)
            from roboquest.harness.agent import MODEL_CONFIGS
            MODEL_CONFIGS[args.model] = dict(config['model_settings'])     # the model settings the episode recorded
            runner.make_api_policy = replay_policy_factory(recorded_rows(ep))
            runner.run_episode(args, ROOT)
            if (out / 'initial-private.json').is_file():
                break
            # Setup failed before the first request (e.g. no EGL context on a GPU another job has filled): once more.
            result = json.loads((out / 'result.json').read_text()) if (out / 'result.json').is_file() else {}
            report['setup_failures'] = report.get('setup_failures', []) + [result.get('error_type')]
            time.sleep(30)
        compare(ep, out, task, report)
    except Exception as error:          # a crash is a finding; keep the traceback in the report
        import traceback
        report['problems'].append(f'replay crashed: {type(error).__name__}: {error}')
        report['traceback'] = traceback.format_exc()[-3000:]
    report.update(ok=not report['problems'], wall_s=round(time.monotonic() - started, 1))
    (Path(out_dir) / f'{label}.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--one', help='replay this raw episode directory')
    parser.add_argument('--episodes', nargs='*', help='raw episode directories')
    parser.add_argument('--episodes-file', type=Path, help='file with one raw episode directory per line')
    parser.add_argument('--gpu', type=int, default=1, help='physical GPU (nvidia-smi index) for --one')
    parser.add_argument('--gpus', default=None, help='physical GPUs with per-GPU worker caps, e.g. 0:3,2:2 (a replay '
                        'renders through EGL and needs about 2 GB on its GPU); without caps --workers is spread evenly')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stagger-s', type=float, default=20., help='wait between worker starts (EGL/scene load)')
    parser.add_argument('--skip-ok', action='store_true', help='skip episodes whose report already says ok')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.one:
        report = replay_one(args.one, args.gpu, args.out)
        print(json.dumps({k: report[k] for k in ('label', 'ok', 'problems', 'wall_s')}))
        return 0 if report['ok'] else 1
    episodes = list(args.episodes or [])
    if args.episodes_file:
        episodes += [line.strip() for line in args.episodes_file.read_text().splitlines() if line.strip()]
    if args.skip_ok:
        done = {r.get('episode') for r in (json.loads(p.read_text()) for p in args.out.glob('*.json')
                                           if p.name != 'summary.json') if r.get('ok')}
        episodes = [ep for ep in episodes if str(Path(ep).resolve()) not in done]
    caps = {}
    for item in (args.gpus or str(args.gpu)).split(','):
        gpu, _, cap = item.partition(':')
        caps[int(gpu)] = int(cap) if cap else None
    if any(cap is None for cap in caps.values()):
        per = max(1, args.workers // len(caps))
        caps = {gpu: cap or per for gpu, cap in caps.items()}
    queue, running, codes = list(episodes), {}, {}
    last_start = 0.

    def free_gpu():
        load = {gpu: sum(1 for _, _, g in running.values() if g == gpu) for gpu in caps}
        open_gpus = [gpu for gpu in caps if load[gpu] < caps[gpu]]
        return min(open_gpus, key=lambda g: load[g]) if open_gpus else None

    while queue or running:
        while queue and free_gpu() is not None and time.monotonic() - last_start >= args.stagger_s:
            ep, gpu = queue.pop(0), free_gpu()
            last_start = time.monotonic()
            log = (args.out / (hashlib.sha256(ep.encode()).hexdigest()[:12] + '.log')).open('w')
            process = subprocess.Popen([sys.executable, __file__, '--one', ep, '--gpu', str(gpu),
                                        '--out', str(args.out)], stdout=log, stderr=subprocess.STDOUT)
            running[process] = (ep, log, gpu)
        time.sleep(2)
        for process in [p for p in running if p.poll() is not None]:
            ep, log, _ = running.pop(process)
            log.close()
            codes[ep] = process.returncode
            print(time.strftime('%H:%M:%S'), f'{len(codes)}/{len(episodes)}', ep,
                  'ok' if process.returncode == 0 else 'DIFFERS/ERROR', flush=True)
    reports = [json.loads(p.read_text()) for p in sorted(args.out.glob('*.json')) if p.name != 'summary.json']
    summary = dict(episodes=len(episodes), reports=len(reports), ok=sum(r['ok'] for r in reports),
                   problems={r['label']: r['problems'] for r in reports if not r['ok']},
                   render_check_warnings={r['label']: r['render_check_differences'][:3] for r in reports
                                          if r.get('render_check_differences')},
                   crashed=[ep for ep, code in codes.items() if code not in (0, 1)])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    return 0 if summary['ok'] == len(episodes) else 1


if __name__ == '__main__':
    sys.exit(main())
