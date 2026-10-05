#!/usr/bin/env python3
"""convert_lerobot_v21.py: RoboQuest oracle demonstrations -> LeRobot v2.1 dataset in VIDEO mode, rendered from the
recorded simulator states (no re-simulation), with per-frame subtask/stage indices from the v3.1 caption timelines.

Same feature keys / shapes / state / action definitions as the RoboQuest image-mode LeRobot export: image = robot0_agentview_left, right_image = robot0_agentview_right,
wrist_image = robot0_eye_in_hand (256x256x3), state = 16-D PandaOmron base-frame state (proprio_to_state),
actions = the logged 12-D command as float32, fps 20. Frame t = observation of recorded state t (before action t) and
action t; t = 0 .. len(actions)-1 (the post-final-action state and the old 2 s hold are not written).
Only the camera features change: dtype "video", one mp4 per episode and camera under videos/chunk-XXX/<key>/.
Extra per-frame int64 features subtask_index / stage_index index meta/subtasks.jsonl / meta/stages.jsonl.

Pythons: $ROBOCASA_PY (the simulator env; default: the running python) for `render`, `replaycheck` and the
children `drive` launches; $LEROBOT_PY (an env with the pinned openpi LeRobot; default: the running python) for
`encode`, `write`, `check` and the encoder children `render` launches. $ROBOQUEST_SOURCE / $ROBOQUEST_SKIN_PY as
in rerender.py (the code tree, the render-only object skin).

Modes (each runs in a specific Python):
  pick      <selection.json> <per_task> <captions_root> <out.json>       any python
  drive     <picks.json> <stage_root> --gpus 0,1,2,3 --workers N [--sim-size N] [--out-size M]
                                                                          any python; launches `render` per episode
  render    <run_dir> <caption.json> <stage_dir> <gpu> [--sim-size N] [--out-size M]   robocasa env (MuJoCo/EGL)
            --sim-size: MuJoCo camera render size (make_env image_size); --out-size: frame size written to the videos
            and PNGs, downscaled when sim > out (exact area average for integer factors, else Lanczos; none when equal).
            Both default to 256 (the published dataset). They are recorded in info.json features.<cam>.info
            (roboquest.sim_size / roboquest.out_size / roboquest.resize_filter) and meta/roboquest_conversion.json.
            The state renderer itself lives in rerender.py (standalone, no LeRobot) next to this file.
  encode    <out.mp4> <width> <height> <fps>                              lerobot env; raw rgb24 frames on stdin
  write     <picks.json> <stage_root> <dataset_root>                      lerobot env (pinned openpi LeRobot)
  check     <picks.json> <stage_root> <dataset_root> <out_dir>            lerobot env
  replaycheck <run_dir> <stage_dir> <gpu> <out.json>                      robocasa env: live replay vs states render
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LEROBOT_PY = os.environ.get('LEROBOT_PY') or sys.executable
ROBOCASA_PY = os.environ.get('ROBOCASA_PY') or sys.executable
FPS = 20
SIZE = 256
STATE_DIM, ACTION_DIM = 16, 12
CAMERA_TO_FEATURE = {  # identical to the LeRobot exporter's CAMERA_TO_FEATURE
    'robot0_agentview_left': 'image',
    'robot0_eye_in_hand': 'wrist_image',
    'robot0_agentview_right': 'right_image',
}
VIDEO_KEYS = ('image', 'wrist_image', 'right_image')
SELF = Path(__file__).resolve()
sys.path.insert(0, str(SELF.parent))
from rerender import (StateRenderer, build_env, proprio_to_state, downscale, resize_filter_name,  # noqa: E402
                      NO_OBJSKIN_TASKS as _NOSKIN)


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- LeRobot stats sampling
# copies of lerobot.common.datasets.compute_stats.{estimate_num_samples, sample_indices} (pinned 0cf86487); `write`
# asserts they agree with the library so the stats PNGs written by `render` are exactly the frames LeRobot samples.
def estimate_num_samples(n, min_num_samples=100, max_num_samples=10_000, power=0.75):
    if n < min_num_samples:
        min_num_samples = n
    return max(min_num_samples, min(int(n ** power), max_num_samples))


def sample_indices(n):
    import numpy as np
    return np.round(np.linspace(0, n - 1, estimate_num_samples(n))).astype(int).tolist()


def check_ticks(n):
    return sorted({0, min(50, n - 1), n // 4, n // 2, (3 * n) // 4, n - 1})


# ------------------------------------------------------------------------------------------------------------ pick
def cmd_pick(selection, per_task, captions_root, out):
    sel = json.loads(Path(selection).read_text())
    picks = []
    for task in sel:                         # frozen file order; first `per_task` entries that have a caption file
        n = 0
        for e in sel[task]:
            cap = Path(captions_root) / task / f"{e['id']}.json"
            if not cap.is_file():
                continue
            picks.append({'task': task, 'id': e['id'], 'run': e['run'], 'caption': str(cap)})
            n += 1
            if n == int(per_task):
                break
    Path(out).write_text(json.dumps({'selection': str(selection), 'selection_sha256': sha256(selection),
                                     'per_task': int(per_task), 'rule': 'first N per task in frozen file order with a caption file',
                                     'episodes': picks}, indent=1) + '\n')
    log(f'picked {len(picks)} -> {out}')


# ------------------------------------------------------------------------------------------------------------ drive
def gpu_free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return {int(a): int(b) for a, b in (l.split(',') for l in out.strip().splitlines())}


def cmd_drive(picks, stage_root, gpus, workers, min_free_mib=6144, sim_size=SIZE, out_size=SIZE):
    eps = json.loads(Path(picks).read_text())['episodes']
    stage_root = Path(stage_root); stage_root.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in gpus.split(',')]
    assert len(gpus) <= 4
    pending = [e for e in eps if not (stage_root / e['id'] / 'episode.json').is_file()]
    log(f'{len(eps)} episodes, {len(pending)} pending, gpus {gpus}, workers {workers}')
    running = {}   # pid -> (proc, ep, gpu, logfile)
    pidfile = stage_root / 'drive_children.pids'
    failures = []
    while pending or running:
        for pid, (p, e, g, lf) in list(running.items()):
            if p.poll() is not None:
                ok = p.returncode == 0 and (stage_root / e['id'] / 'episode.json').is_file()
                log(f"{'done' if ok else 'FAILED'} {e['task']} {e['id']} gpu {g} rc {p.returncode}")
                if not ok:
                    failures.append(e['id'])
                del running[pid]
        while pending and len(running) < workers:
            free = gpu_free_mib()
            load = {g: sum(1 for v in running.values() if v[2] == g) for g in gpus}
            ok = [g for g in gpus if free.get(g, 0) >= min_free_mib]
            if not ok:
                log(f'no GPU in {gpus} with >= {min_free_mib} MiB free ({free}); waiting'); break
            g = min(ok, key=lambda x: (load[x], -free[x]))
            e = pending.pop(0)
            sd = stage_root / e['id']
            lf = stage_root / f"{e['id']}.log"
            p = subprocess.Popen([ROBOCASA_PY, str(SELF), 'render', e['run'], e['caption'], str(sd), str(g),
                                  '--sim-size', str(sim_size), '--out-size', str(out_size)],
                                 stdout=open(lf, 'w'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[p.pid] = (p, e, g, lf)
            with open(pidfile, 'a') as f:
                f.write(f"{p.pid} {e['id']} gpu{g} {time.strftime('%F %T')}\n")
            log(f"start {e['task']} {e['id']} gpu {g} (free {free[g]} MiB) pid {p.pid}")
        time.sleep(5)
    log(f'drive finished; failures: {failures}')
    return 1 if failures else 0


# ---------------------------------------------------------------------------------------------------------- render
def lerobot_env():
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'ROBOQUEST_RUNTIME', 'MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID')}
    return env


def cmd_render(run, caption, stage_dir, gpu, sim_size=SIZE, out_size=SIZE):
    t0 = time.time()
    run, stage_dir, gpu, sim_size, out_size = Path(run), Path(stage_dir), int(gpu), int(sim_size), int(out_size)
    cap = json.loads(Path(caption).read_text())
    task = json.loads((run / 'result.json').read_text())['task']
    partial = stage_dir.with_name(stage_dir.name + '.partial')
    if stage_dir.exists():
        raise SystemExit(f'{stage_dir} exists')
    if partial.exists():
        import shutil; shutil.rmtree(partial)
    partial.mkdir(parents=True)
    R = StateRenderer(run, gpu, sim_size, out_size, objskin='auto', visfix=True, mirror_root=stage_dir.parent.parent / 'render-assets')
    import numpy as np
    info, lg, iid = R.info, R.log, R.info['instance_id']
    A = R.actions
    N = len(A)
    assert int(lg['ticks']) == N and A.shape == (N, ACTION_DIM) and np.isfinite(A).all()
    S, spec = R.states, R.spec
    # caption timeline: rows cover [0, N) contiguously
    assert cap['id'] == iid and cap['task'] == task and int(cap['n_ticks']) == N, (cap['id'], cap['task'], cap['n_ticks'], N)
    tl = cap['timeline']
    assert tl[0]['start'] == 0 and tl[-1]['end'] == N and all(a['end'] == b['start'] for a, b in zip(tl, tl[1:])), 'timeline gaps'
    assert all(r['end'] > r['start'] and r['subtask'] and r['stage'] for r in tl)
    row = np.empty(N, np.int32)
    for k, r in enumerate(tl):
        row[r['start']:r['end']] = k
    hook = R.hook
    keep = set(sample_indices(N)) | set(check_ticks(N))
    for cam, key in CAMERA_TO_FEATURE.items():
        (partial / 'frames' / key).mkdir(parents=True)
    from PIL import Image
    enc = {}
    for cam, key in CAMERA_TO_FEATURE.items():
        enc[key] = subprocess.Popen([LEROBOT_PY, str(SELF), 'encode', str(partial / f'{key}.mp4'), str(out_size), str(out_size), str(FPS)],
                                    stdin=subprocess.PIPE, stdout=open(partial / f'{key}.encode.json', 'w'), env=lerobot_env(),
                                    stderr=open(partial / f'{key}.encode.log', 'w'))
    states = np.empty((N, STATE_DIM), np.float32)
    build_s = time.time() - t0
    t_r = time.time()
    for i in range(N):
        R.restore(i)
        states[i] = R.state()
        imgs = R.images()
        for key, im in imgs.items():
            assert im.shape == (out_size, out_size, 3) and im.dtype == np.uint8
            enc[key].stdin.write(im.tobytes())
            if i in keep:
                Image.fromarray(im).save(partial / 'frames' / key / f'frame_{i:06d}.png', compress_level=1)
        if (i + 1) % 1000 == 0:
            log(f'{iid} {i + 1}/{N} ticks, {time.time() - t_r:.0f} s')
    loop_s = time.time() - t_r
    t_e = time.time()
    enc_out = {}
    for key, p in enc.items():
        p.stdin.close()
    for key, p in enc.items():
        rc = p.wait()
        assert rc == 0, f'encoder {key} rc {rc}: {(partial / (key + ".encode.log")).read_text()[-2000:]}'
        enc_out[key] = json.loads((partial / f'{key}.encode.json').read_text())
        assert enc_out[key]['frames'] == N, (key, enc_out[key], N)
    enc_wait_s = time.time() - t_e
    render_s = R.render_s
    R.close()
    np.savez(partial / 'arrays.npz', state=states, actions=A.astype(np.float32), timeline_row=row)
    meta = {
        'task': task, 'instance_id': iid, 'run': str(run), 'caption': str(caption), 'caption_sha256': sha256(caption),
        'caption_version': cap.get('version'), 'goal': cap['goal'], 'timeline': tl, 'frames': N,
        'recorded_states': int(len(S)), 'state_spec': spec, 'ink_hook': hook is not None,
        'sample_indices': sorted(set(sample_indices(N))), 'check_ticks': check_ticks(N),
        'render': {**info, 'gpu': gpu, 'image_size': out_size, 'fps': FPS},
        'encode': enc_out,
        'timing_s': {'build': round(build_s, 1), 'loop': round(loop_s, 1), 'render_calls': round(render_s, 1),
                     'encoder_drain': round(enc_wait_s, 1), 'total': round(time.time() - t0, 1)},
        'video_bytes': {k: (partial / f'{k}.mp4').stat().st_size for k in VIDEO_KEYS},
        'finished': time.strftime('%F %T %z'),
    }
    (partial / 'episode.json').write_text(json.dumps(meta, indent=1) + '\n')
    partial.rename(stage_dir)
    log('REPORT ' + json.dumps({'id': iid, 'task': task, 'frames': N, 'timing_s': meta['timing_s'], 'video_bytes': meta['video_bytes']}))


# ---------------------------------------------------------------------------------------------------------- encode
def cmd_encode(out, width, height, fps):
    """LeRobot's encode_video_frames (pinned 0cf86487), fed raw frames instead of PNG files: same PyAV build, codec,
    pix_fmt, g, crf, fast_decode -- read from the library's own defaults."""
    import inspect
    import av, numpy as np
    from lerobot.common.datasets.video_utils import encode_video_frames
    dflt = {k: v.default for k, v in inspect.signature(encode_video_frames).parameters.items() if v.default is not inspect._empty}
    vcodec, pix_fmt, g, crf, fast_decode = dflt['vcodec'], dflt['pix_fmt'], dflt['g'], dflt['crf'], dflt['fast_decode']
    width, height, fps = int(width), int(height), int(fps)
    opts = {}
    if g is not None:
        opts['g'] = str(g)
    if crf is not None:
        opts['crf'] = str(crf)
    if fast_decode:
        opts['svtav1-params' if vcodec == 'libsvtav1' else 'tune'] = f'fast-decode={fast_decode}' if vcodec == 'libsvtav1' else 'fastdecode'
    import logging
    logging.getLogger('libav').setLevel(av.logging.ERROR)
    n, nbytes = 0, width * height * 3
    src = sys.stdin.buffer
    t0 = time.time()
    with av.open(str(out), 'w') as output:
        st = output.add_stream(vcodec, fps, options=opts)
        st.pix_fmt = pix_fmt; st.width = width; st.height = height
        while True:
            buf = src.read(nbytes)
            if not buf:
                break
            while len(buf) < nbytes:
                more = src.read(nbytes - len(buf))
                if not more:
                    raise SystemExit(f'truncated frame {n}')
                buf += more
            frame = av.VideoFrame.from_ndarray(np.frombuffer(buf, np.uint8).reshape(height, width, 3), format='rgb24')
            pkt = st.encode(frame)
            if pkt:
                output.mux(pkt)
            n += 1
        pkt = st.encode()
        if pkt:
            output.mux(pkt)
    print(json.dumps({'frames': n, 'vcodec': vcodec, 'pix_fmt': pix_fmt, 'g': g, 'crf': crf, 'fast_decode': fast_decode,
                      'options': opts, 'pyav': av.__version__, 'libavcodec': list(av.library_versions['libavcodec']),
                      'bytes': Path(out).stat().st_size, 'encode_wall_s': round(time.time() - t0, 1)}))


# ----------------------------------------------------------------------------------------------------------- write
def features(size=SIZE):
    vid = {'dtype': 'video', 'shape': (size, size, 3), 'names': ['height', 'width', 'channel']}
    return {
        'image': dict(vid), 'wrist_image': dict(vid), 'right_image': dict(vid),
        'state': {'dtype': 'float32', 'shape': (STATE_DIM,), 'names': ['state']},
        'actions': {'dtype': 'float32', 'shape': (ACTION_DIM,), 'names': ['actions']},
        'subtask_index': {'dtype': 'int64', 'shape': (1,), 'names': None},
        'stage_index': {'dtype': 'int64', 'shape': (1,), 'names': None},
    }


def cmd_write(picks, stage_root, root, repo_id='roboquest/oracle-v21-video'):
    import shutil
    import numpy as np
    from lerobot.common.datasets import lerobot_dataset as LD
    from lerobot.common.datasets import compute_stats as CS
    assert LD.CODEBASE_VERSION == 'v2.1', LD.CODEBASE_VERSION
    eps = json.loads(Path(picks).read_text())['episodes']
    stage_root, root = Path(stage_root), Path(root)
    t0 = time.time()
    sizes = {(json.loads((stage_root / ep['id'] / 'episode.json').read_text())['render'].get('sim_size', SIZE),
              json.loads((stage_root / ep['id'] / 'episode.json').read_text())['render'].get('out_size', SIZE),
              json.loads((stage_root / ep['id'] / 'episode.json').read_text())['render'].get('resize_filter', 'none')) for ep in eps}
    assert len(sizes) == 1, f'stage episodes were rendered at different sizes: {sizes}'
    sim_size, out_size, resize_filter = sizes.pop()
    ds = LD.LeRobotDataset.create(repo_id=repo_id, fps=FPS, root=root, robot_type='panda_omron', features=features(out_size),
                                  use_videos=True, image_writer_processes=0, image_writer_threads=0)
    subtasks, stages, rows = {}, {}, []
    for ep in eps:
        sd = stage_root / ep['id']
        meta = json.loads((sd / 'episode.json').read_text())
        arr = np.load(sd / 'arrays.npz')
        N = meta['frames']
        assert sample_indices(N) == CS.sample_indices(N)
        lg = json.load(open(Path(meta['run']) / 'actions' / f"{meta['instance_id']}.json"))
        assert len(lg['actions']) == N == len(arr['state']) == len(arr['actions'])
        assert np.array_equal(arr['actions'], np.asarray(lg['actions'], np.float64).astype(np.float32))
        tl = meta['timeline']
        sub_ids = [subtasks.setdefault(r['subtask'], len(subtasks)) for r in tl]
        stg_ids = [stages.setdefault(r['stage'], len(stages)) for r in tl]
        row = arr['timeline_row']
        ep_idx = ds.meta.total_episodes
        buf = ds.create_episode_buffer()
        buf['size'] = N
        buf['task'] = [meta['goal']] * N
        buf['frame_index'] = list(range(N))
        buf['timestamp'] = [i / FPS for i in range(N)]
        buf['state'] = list(arr['state'])
        buf['actions'] = list(arr['actions'])
        buf['subtask_index'] = [sub_ids[k] for k in row]
        buf['stage_index'] = [stg_ids[k] for k in row]
        for key in VIDEO_KEYS:
            buf[key] = [str(sd / 'frames' / key / f'frame_{i:06d}.png') for i in range(N)]
            for i in CS.sample_indices(N):
                assert Path(buf[key][i]).is_file(), buf[key][i]
            dst = root / ds.meta.get_video_file_path(ep_idx, key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sd / f'{key}.mp4', dst)   # encode_episode_videos() skips files that already exist
        ds.episode_buffer = buf
        ds.save_episode()
        rows.append({'episode_index': ep_idx, 'task_name': meta['task'], 'instance_id': meta['instance_id'], 'run': meta['run'],
                     'source_tree': meta['render']['source'], 'caption_file': meta['caption'], 'caption_sha256': meta['caption_sha256'],
                     'caption_version': meta['caption_version'], 'length': N,
                     'timeline': [{**r, 'subtask_index': si, 'stage_index': gi} for r, si, gi in zip(tl, sub_ids, stg_ids)],
                     'render': {k: meta['render'][k] for k in ('visfix', 'objskin', 'bin_fix_geoms', 'mug_liquid_hidden', 'objskin_geoms') if k in meta['render']},
                     'ink_hook': meta['ink_hook']})
        log(f"episode {ep_idx} {meta['task']} {meta['instance_id']} {N} frames")
    with open(root / 'meta' / 'subtasks.jsonl', 'w') as f:
        for t, i in subtasks.items():
            f.write(json.dumps({'subtask_index': i, 'subtask': t}) + '\n')
    with open(root / 'meta' / 'stages.jsonl', 'w') as f:
        for t, i in stages.items():
            f.write(json.dumps({'stage_index': i, 'stage': t}) + '\n')
    with open(root / 'meta' / 'roboquest_episodes.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    enc0 = json.loads((stage_root / eps[0]['id'] / 'episode.json').read_text())['encode']['image']
    # how the frames were made, beside LeRobot's own video info (extra keys; the v2.1 loader ignores them)
    from lerobot.common.datasets.utils import write_info, INFO_PATH
    info = json.loads((root / INFO_PATH).read_text())
    for key in VIDEO_KEYS:
        info['features'][key].setdefault('info', {}).update({'roboquest.sim_size': sim_size, 'roboquest.out_size': out_size,
                                                             'roboquest.resize_filter': resize_filter})
    write_info(info, root)
    (root / 'meta' / 'roboquest_conversion.json').write_text(json.dumps({
        'converter': str(SELF), 'converter_sha256': sha256(SELF), 'picks': str(picks), 'written': time.strftime('%F %T %z'),
        'lerobot_commit': '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5', 'codebase_version': 'v2.1',
        'sim_size': sim_size, 'out_size': out_size, 'resize_filter': resize_filter,
        'video_encoding': {k: enc0[k] for k in ('vcodec', 'pix_fmt', 'g', 'crf', 'fast_decode', 'options', 'pyav', 'libavcodec')},
        'frame_convention': 'frame t = recorded state t (before action t) + action t, t < len(actions); no terminal hold',
        'cameras': CAMERA_TO_FEATURE, 'task': 'episode goal (captions_v31 goal), raw text',
        'subtask_stage': 'per-frame int64 subtask_index / stage_index -> meta/subtasks.jsonl / meta/stages.jsonl (captions_v31 timeline)',
    }, indent=1) + '\n')
    log(f'wrote {len(rows)} episodes to {root} in {time.time() - t0:.0f} s')


# ----------------------------------------------------------------------------------------------------------- check
def cmd_check(picks, stage_root, root, out_dir, repo_id='roboquest/oracle-v21-video'):
    import numpy as np, torch
    from PIL import Image, ImageDraw
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import decode_video_frames
    eps = json.loads(Path(picks).read_text())['episodes']
    stage_root, root, out_dir = Path(stage_root), Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # LeRobot's default backend (torchcodec when importable; $LR_VIDEO_BACKEND overrides). It needs a torchcodec build
    # that matches torch (e.g. torch 2.14+cpu with torchcodec 0.16.0+cpu); torchvision 0.29 has no VideoReader, so the
    # 'pyav' backend is not available there.
    ds = LeRobotDataset(repo_id, root=root, video_backend=os.environ.get('LR_VIDEO_BACKEND') or None)
    load_s = time.time() - t0
    rep = {'dataset': str(root), 'load_s': round(load_s, 1), 'num_frames': ds.num_frames, 'num_episodes': ds.num_episodes,
           'video_backend': ds.video_backend, 'features': {k: {'dtype': v['dtype'], 'shape': list(v['shape'])} for k, v in ds.features.items()},
           'episodes': []}
    subtasks = [json.loads(l) for l in open(root / 'meta' / 'subtasks.jsonl')]
    stages = [json.loads(l) for l in open(root / 'meta' / 'stages.jsonl')]
    sub_text = {r['subtask_index']: r['subtask'] for r in subtasks}
    stg_text = {r['stage_index']: r['stage'] for r in stages}
    hf = ds.hf_dataset.with_format(None)
    col_ep = np.asarray(hf['episode_index']); col_sub = np.asarray(hf['subtask_index']); col_stg = np.asarray(hf['stage_index'])
    col_fi = np.asarray(hf['frame_index']); col_task = np.asarray(hf['task_index'])
    col_state = np.asarray(hf['state'], np.float32); col_act = np.asarray(hf['actions'], np.float32)
    all_mad = []
    comps = []
    for ep_idx, ep in enumerate(eps):
        sd = stage_root / ep['id']
        meta = json.loads((sd / 'episode.json').read_text())
        lg = json.load(open(Path(meta['run']) / 'actions' / f"{meta['instance_id']}.json"))
        cap = json.loads(Path(ep['caption']).read_text())
        sel = np.nonzero(col_ep == ep_idx)[0]
        n = len(sel)
        r = {'episode_index': ep_idx, 'task': meta['task'], 'id': meta['instance_id'], 'frames': n, 'actions': len(lg['actions']),
             'frames_eq_actions': n == len(lg['actions']) == ds.meta.episodes[ep_idx]['length'],
             'frame_index_ok': bool(np.array_equal(col_fi[sel], np.arange(n)))}
        # subtask / stage text per frame vs caption timeline (every frame)
        exp_sub = np.empty(n, object); exp_stg = np.empty(n, object)
        for row in cap['timeline']:
            exp_sub[row['start']:row['end']] = row['subtask']; exp_stg[row['start']:row['end']] = row['stage']
        got_sub = np.asarray([sub_text[int(i)] for i in col_sub[sel]], object)
        got_stg = np.asarray([stg_text[int(i)] for i in col_stg[sel]], object)
        r['subtask_mismatch_frames'] = int((got_sub != exp_sub).sum()); r['stage_mismatch_frames'] = int((got_stg != exp_stg).sum())
        r['task_is_goal'] = all(ds.meta.tasks[int(t)] == cap['goal'] for t in np.unique(col_task[sel]))
        r['actions_match_log'] = bool(np.array_equal(col_act[sel], np.asarray(lg['actions'], np.float64).astype(np.float32)))
        arr = np.load(sd / 'arrays.npz')
        r['state_match_stage'] = bool(np.array_equal(col_state[sel], arr['state']))
        # video stream frame count
        import av
        nfr = {}
        for key in VIDEO_KEYS:
            with av.open(str(root / ds.meta.get_video_file_path(ep_idx, key))) as c:
                nfr[key] = sum(1 for _ in c.decode(video=0))
        r['video_frames'] = nfr
        r['video_frames_ok'] = all(v == n for v in nfr.values())
        # decoded frames via the dataset __getitem__ vs the lossless PNGs of the same ticks
        mads = {}
        for t in meta['check_ticks']:
            item = ds[int(sel[t])]
            assert int(item['frame_index']) == t
            assert sub_text[int(item['subtask_index'])] == exp_sub[t]
            for key in VIDEO_KEYS:
                dec = (item[key].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                png = np.asarray(Image.open(sd / 'frames' / key / f'frame_{t:06d}.png').convert('RGB'))
                mad = float(np.abs(dec.astype(np.int16) - png.astype(np.int16)).mean())
                mads[f'{key}@{t}'] = round(mad, 3); all_mad.append(mad)
                comps.append((mad, meta['task'], meta['instance_id'], key, t, png, dec))
        r['mad'] = mads
        r['mad_mean'] = round(float(np.mean(list(mads.values()))), 3)
        # sizes
        vb = {k: (root / ds.meta.get_video_file_path(ep_idx, k)).stat().st_size for k in VIDEO_KEYS}
        pb = (root / ds.meta.get_data_file_path(ep_idx)).stat().st_size
        r['bytes'] = {'videos': vb, 'parquet': pb, 'total': sum(vb.values()) + pb, 'per_frame': round((sum(vb.values()) + pb) / n, 1)}
        r['timing_s'] = meta['timing_s']
        rep['episodes'].append(r)
        log(json.dumps({k: r[k] for k in ('task', 'id', 'frames', 'frames_eq_actions', 'subtask_mismatch_frames', 'stage_mismatch_frames', 'video_frames_ok', 'mad_mean')}))
    rep['mad_overall_mean'] = round(float(np.mean(all_mad)), 3); rep['mad_overall_max'] = round(float(np.max(all_mad)), 3)
    # 3 side-by-side comparisons: one per camera, the frame with that camera's median MAD
    comps.sort(key=lambda c: c[0])
    pick = []
    for key in VIDEO_KEYS:
        ck = [c for c in comps if c[3] == key]
        rep.setdefault('mad_by_camera', {})[key] = {'mean': round(float(np.mean([c[0] for c in ck])), 3), 'max': round(float(ck[-1][0]), 3)}
        pick.append(ck[len(ck) // 2])
    paths = []
    for k, (mad, task, iid, key, t, png, dec) in enumerate(pick):
        diff = np.clip(np.abs(dec.astype(np.int16) - png.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
        sz = png.shape[0]
        canvas = Image.new('RGB', (3 * sz + 20, sz + 22), 'white')
        for j, im in enumerate((png, dec, diff)):
            canvas.paste(Image.fromarray(im), (j * (sz + 10), 22))
        ImageDraw.Draw(canvas).text((4, 4), f'{task} {iid} {key} tick {t}: PNG render | decoded AV1 frame | |diff| x8   MAD {mad:.2f}/255', fill='black')
        p = out_dir / f'compare_{k + 1}_{task}_{key}_t{t}.png'
        canvas.save(p); paths.append(str(p))
    rep['comparison_pngs'] = paths
    # random-access decode timing through the dataset (one frame = 3 videos)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, ds.num_frames, 60)
    t1 = time.time()
    for i in idx:
        ds[int(i)]
    rep['random_getitem_ms'] = round((time.time() - t1) / len(idx) * 1000, 1)
    tot_b = sum(e['bytes']['total'] for e in rep['episodes']); tot_f = sum(e['frames'] for e in rep['episodes'])
    rep['totals'] = {'bytes': tot_b, 'frames': tot_f, 'bytes_per_frame': round(tot_b / tot_f, 1),
                     'bytes_per_episode': round(tot_b / len(rep['episodes']))}
    (out_dir / 'check_report.json').write_text(json.dumps(rep, indent=1) + '\n')
    log('CHECK ' + json.dumps({k: rep[k] for k in ('num_frames', 'num_episodes', 'mad_overall_mean', 'mad_overall_max', 'mad_by_camera', 'random_getitem_ms', 'totals', 'comparison_pngs')}))


# ----------------------------------------------------------------------------------------------------- replaycheck
def cmd_replaycheck(run, stage_dir, gpu, out):
    """Independent path: live replay of the logged actions (env.step, as the trunk image-mode exporter does) with the
    same render-only fixes, compared at the stage's check ticks with the states render (PNG) and state vector."""
    run, stage_dir = Path(run), Path(stage_dir)
    meta = json.loads((stage_dir / 'episode.json').read_text())
    objskin = meta['render']['objskin'] or ''
    sim_size, out_size = meta['render'].get('sim_size', SIZE), meta['render'].get('out_size', SIZE)
    e, m, d, info, ctx = build_env(run, int(gpu), sim_size, objskin, visfix=True, mirror_root=stage_dir.parent.parent / 'render-assets')
    np, CAMERAS, lg, obs = ctx['np'], ctx['CAMERAS'], ctx['log'], ctx['obs']
    from PIL import Image
    arr = np.load(stage_dir / 'arrays.npz')
    ticks = meta['check_ticks']
    gids = ctx['bin_fix']
    res = {}
    t0 = time.time()
    A = np.asarray(lg['actions'], np.float64)

    def grab():
        # render-only bin fix without touching dynamics: shorten wall boxes, shift their world position, render, undo
        saved = [(g, d.geom_xpos[g].copy(), m.geom_size[g, 2]) for g in gids]
        for g in gids:
            b = int(m.geom_bodyid[g])   # the model fix moves geom_pos (body frame) down 3 mm
            d.geom_xpos[g] = d.geom_xpos[g] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, -0.003]); m.geom_size[g, 2] -= 0.003
        imgs = {CAMERA_TO_FEATURE[c]: downscale(np.ascontiguousarray(np.asarray(e.sim.render(camera_name=c, width=sim_size, height=sim_size))[::-1]), out_size)
                for c in CAMERAS}
        for g, xp, sz in saved:
            d.geom_xpos[g] = xp; m.geom_size[g, 2] = sz
        return imgs
    for t in range(max(ticks) + 1):
        if t in ticks:
            imgs = grab()
            st = proprio_to_state(e.robot_proprio())
            r = {'state_max_abs_diff': float(np.abs(st - arr['state'][t]).max())}
            for key, im in imgs.items():
                png = np.asarray(Image.open(stage_dir / 'frames' / key / f'frame_{t:06d}.png').convert('RGB'))
                r[f'{key}_mad'] = round(float(np.abs(im.astype(np.int16) - png.astype(np.int16)).mean()), 4)
                if t == ticks[len(ticks) // 2] and key == 'image':
                    Image.fromarray(np.concatenate([png, im], 1)).save(stage_dir.parent / f"replaycheck_{meta['task']}_{t}.png")
            res[t] = r
        if t < len(A):
            obs, *_ = e.step(A[t].copy())
    e.close()
    rep = {'task': meta['task'], 'id': meta['instance_id'], 'ticks': res, 'wall_s': round(time.time() - t0, 1)}
    Path(out).write_text(json.dumps(rep, indent=1) + '\n')
    log('REPLAYCHECK ' + json.dumps(rep))


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'pick':
        cmd_pick(*args)
    elif mode == 'drive':
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument('picks'); ap.add_argument('stage_root')
        ap.add_argument('--gpus', default='0,1,2,3'); ap.add_argument('--workers', type=int, default=4)
        ap.add_argument('--min-free-mib', type=int, default=6144)
        ap.add_argument('--sim-size', type=int, default=SIZE, help='MuJoCo camera render size (make_env image_size)')
        ap.add_argument('--out-size', type=int, default=SIZE, help='frame size written to video/PNG (area/Lanczos downscale)')
        a = ap.parse_args(args)
        resize_filter_name(a.sim_size, a.out_size)   # rejects out > sim
        raise SystemExit(cmd_drive(a.picks, a.stage_root, a.gpus, a.workers, a.min_free_mib, a.sim_size, a.out_size))
    elif mode == 'render':
        import argparse
        ap = argparse.ArgumentParser(); [ap.add_argument(k) for k in ('run', 'caption', 'stage_dir', 'gpu')]
        ap.add_argument('--sim-size', type=int, default=SIZE); ap.add_argument('--out-size', type=int, default=SIZE)
        a = ap.parse_args(args)
        cmd_render(a.run, a.caption, a.stage_dir, a.gpu, a.sim_size, a.out_size)
    elif mode == 'encode':
        cmd_encode(*args)
    elif mode == 'write':
        cmd_write(*args)
    elif mode == 'check':
        cmd_check(*args)
    elif mode == 'replaycheck':
        cmd_replaycheck(*args)
    else:
        raise SystemExit(__doc__)
