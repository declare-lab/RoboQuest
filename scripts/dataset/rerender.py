#!/usr/bin/env python3
"""rerender.py: re-render a RoboQuest episode from its recorded simulator states at any resolution.
No LeRobot needed: only the RoboQuest/RoboCasa runtime (MuJoCo + EGL) and the episode's run directory
(result.json, actions/<id>.json with the instance and the actions, actions/<id>.states.npz with the states).
The task code comes from this repository (``--source`` / ``$ROBOQUEST_SOURCE`` for another checkout;
``--provenance-source`` uses the tree named in the run's provenance.json).

Every tick is restored with mj_setState + mj_forward (+ the task's per-step visual update, stamps: _update_ink) and the
three policy cameras are rendered at --sim-size (the MuJoCo render / make_env image_size), then downscaled to
--out-size when sim > out (exact area average for integer factors, Lanczos otherwise; no resize when equal).
Frame t = recorded state t (before action t), t = 0 .. len(actions)-1, same as the LeRobot export.

  MUJOCO_GL=egl python scripts/dataset/rerender.py <run_dir> <out_dir> [--gpu 0] [--sim-size 512] [--out-size 256]
        [--mp4] [--png] [--ticks 0:3040:1 | --ticks 0,1500,3039] [--cameras image,wrist_image,right_image]
        [--objskin auto|final|none] [--skin-py skin.py] [--no-visfix] [--crf 18] [--source DIR]
--sim-size and --out-size both default to 256, the published dataset's sizes.
--objskin final (auto: when --skin-py / $ROBOQUEST_SKIN_PY names a skin module, except marked_mugs) applies the
render-only release skin of the search-pool objects; without a skin module those objects keep their simulator look.
Outputs: <out_dir>/<camera>.mp4 (libx264 yuv420p via imageio, fps 20; for viewing, not the LeRobot codec) and/or
<out_dir>/frames/<camera>/frame_XXXXXX.png, plus <out_dir>/rerender.json (sizes, filter, fixes, timings).
Camera names: image = robot0_agentview_left, wrist_image = robot0_eye_in_hand, right_image = robot0_agentview_right.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

FPS = 20
STATE_DIM = 16
CAMERA_TO_FEATURE = {  # identical to the LeRobot exporter's CAMERA_TO_FEATURE
    'robot0_agentview_left': 'image',
    'robot0_eye_in_hand': 'wrist_image',
    'robot0_agentview_right': 'right_image',
}
NO_OBJSKIN_TASKS = {'marked_mugs'}   # its mugs carry task-relevant marks
SIZE = 256   # the published dataset: sim 256 / out 256
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_source():
    """The code tree the task classes are imported from: $ROBOQUEST_SOURCE, else this repository."""
    return Path(os.environ.get('ROBOQUEST_SOURCE') or REPO_ROOT)


def default_skin():
    """The render-only object skin module: $ROBOQUEST_SKIN_PY, else none."""
    value = os.environ.get('ROBOQUEST_SKIN_PY')
    return Path(value) if value else None


def resize_filter_name(sim_size: int, out_size: int) -> str:
    if sim_size == out_size:
        return 'none'
    if sim_size < out_size:
        raise ValueError(f'out-size {out_size} > sim-size {sim_size}: render at least at the output size')
    return f'area x{sim_size // out_size} (PIL Image.reduce)' if sim_size % out_size == 0 else 'lanczos (PIL)'


def downscale(img, out_size: int):
    """uint8 HxWx3 -> out_size x out_size. Integer factor: exact box/area average; otherwise Lanczos."""
    import numpy as np
    from PIL import Image
    n = img.shape[0]
    if n == out_size:
        return img
    im = Image.fromarray(img)
    im = im.reduce(n // out_size) if n % out_size == 0 else im.resize((out_size, out_size), Image.LANCZOS)
    return np.asarray(im)


def proprio_to_state(proprio):
    """Copy of the LeRobot exporter's proprio_to_state: RoboCasa/openpi PandaOmron 16-D state in the base frame."""
    import numpy as np

    def nq(q):
        q = np.asarray(q, float); return q / np.linalg.norm(q)

    def qmul(l, r):
        lx, ly, lz, lw = l; rx, ry, rz, rw = r
        return np.asarray([lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx,
                           lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz])

    def qrot(q, v):
        xyz, s = q[:3], q[3]
        return v + 2 * np.cross(xyz, np.cross(xyz, v) + s * v)
    ep = np.asarray(proprio['eef_position_world_m'], float); bp = np.asarray(proprio['base_position_world_m'], float)
    eq = nq(proprio['eef_quaternion_xyzw']); bq = nq(proprio['base_quaternion_xyzw'])
    binv = np.r_[-bq[:3], bq[3]]
    rel_p = qrot(binv, ep - bp); rel_q = nq(qmul(binv, eq))
    joints = np.asarray(proprio['joint_positions'], float)
    s = np.concatenate([rel_p, rel_q, bp, bq, joints[-2:]])
    assert s.shape == (STATE_DIM,) and np.isfinite(s).all()
    return s.astype(np.float32)


def resolve_source(run, source=None, provenance_source=False):
    """``source``, else the run's provenance.json source tree (``provenance_source``), else :func:`default_source`."""
    if source:
        return Path(source)
    prov = Path(run) / 'provenance.json'
    if provenance_source:
        if not prov.is_file():
            raise SystemExit(f'--provenance-source: {prov} not found')
        return Path(json.loads(prov.read_text())['source'])
    return default_source()


def build_env(run, gpu, sim_size, objskin, visfix=True, mirror_root=None, skin_py=None, source=None,
              provenance_source=False):
    """The recording's environment build (make_env from the instance in actions/<id>.json) + render-only visual
    fixes, parameterised. objskin: 'final' or ''. Returns (env, mujoco model, data, info, ctx)."""
    os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    os.environ['ROBOQUEST_RECORD_STATES'] = '0'
    run = Path(run)
    source = resolve_source(run, source, provenance_source)
    sys.path.insert(0, str(source)); sys.path.insert(0, str(source / 'scripts'))
    import numpy as np, mujoco
    from robocasa.models.objects.objects import MJCFObject
    mirror_root = Path(mirror_root or (Path(tempfile.gettempdir()) / 'roboquest-render-assets'))
    _orig = MJCFObject.__init__

    def _obj(self, name, mjcf_path, *a, **k):
        original = Path(mjcf_path).resolve()
        mirror = mirror_root / hashlib.sha256(str(original.parent).encode()).hexdigest()[:16] / str(os.getpid())
        mirror.mkdir(parents=True, exist_ok=True)
        for e in original.parent.iterdir():
            link = mirror / e.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(e, target_is_directory=e.is_dir())
        _orig(self, name, str(mirror / original.name), *a, **k)
    MJCFObject.__init__ = _obj
    SKIN = None
    if objskin == 'final':
        import importlib.util as _ilu
        import robosuite.utils.binding_utils as _bu
        skin_py = skin_py or default_skin()
        if skin_py is None or not Path(skin_py).is_file():
            raise SystemExit(f'--objskin final needs a skin module (--skin-py / $ROBOQUEST_SKIN_PY): {skin_py}')
        _spec = _ilu.spec_from_file_location('_skin', str(skin_py))
        SKIN = _ilu.module_from_spec(_spec); _spec.loader.exec_module(SKIN)
        cats = {'mug': ('blue', 'red'), 'can': ('red', 'green'), 'tin': ('green', 'yellow'), 'bottle': ('yellow', 'blue')}
        mats = []
        for cat, cols in cats.items():
            for col in cols:
                for part in ({SKIN.BODY_PART[cat]} | ({'black'} if cat == 'bottle' else {'silver'} if cat == 'tin' else set())):
                    tex = SKIN.texture_path(cat, col) if part in ('label', 'speckle', 'ceramic') else None
                    mats.append((f'sk_{cat}_{col}_{part}', part, tex))
        _skin_orig = _bu.MjSim.from_xml_string.__func__

        def _skin_inject(cls, xml):
            add = ''
            for n, part, tex in mats:
                spec, shin, refl = SKIN.FINISH[part]
                if tex is not None:
                    add += f'<texture name="{n}_tex" type="2d" file="{tex}"/>'
                add += f'<material name="{n}" specular="{spec}" shininess="{shin}" reflectance="{refl}"' + (f' texture="{n}_tex"' if tex else '') + '/>'
            return _skin_orig(cls, xml.replace('</asset>', add + '</asset>', 1))
        _bu.MjSim.from_xml_string = classmethod(_skin_inject)
    from roboquest.kitchen import make_env, CAMERAS
    from roboquest.manifest import task_class
    result = json.loads((run / 'result.json').read_text())
    task, iid = result['task'], result['instance_id']
    lg = json.load(open(run / 'actions' / f'{iid}.json'))
    e = make_env(task_class(task), lg['instance'], render=True, image_size=int(sim_size), gpu=gpu, horizon=len(lg['actions']) + 20)
    obs = e.reset()
    m, d = e.sim.model._model, e.sim.data._data
    info = {'source': str(source), 'task': task, 'instance_id': iid, 'visfix': bool(visfix), 'objskin': objskin or None,
            'sim_size': int(sim_size)}
    bin_fix = []   # render-only wall shortening (RENDER_VISFIX bin fix)
    if visfix:
        for g in range(m.ngeom):
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            mt = re.match(r'(.+)_wall_[xy]_[+-]1(_visual)?$', gn)
            _r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, mt.group(1) + '_roof_x_+1') if mt else -1
            if mt and _r >= 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX \
                    and m.geom_pos[g, 2] + m.geom_size[g, 2] > m.geom_pos[_r, 2] - m.geom_size[_r, 2] + 1e-6:
                bin_fix.append(g)
        liq = []
        for g in range(m.ngeom):
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
            if gn.endswith('_liquid') and bn.startswith('obj_mug') and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                m.geom_rgba[g, 3] = 0.; liq.append(gn)
        info['bin_fix_geoms'] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in bin_fix]
        info['mug_liquid_hidden'] = liq
    if objskin == 'final':
        sk = []
        for g in range(m.ngeom):
            mt = re.match(r'obj_(mug|can|tin|bottle)_(blue|red|green|yellow)_(g\d+)$', mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '')
            if not (mt and m.geom_contype[g] == 0 and m.geom_group[g] == 1):
                continue
            cat, col, gpart = mt.groups()
            part = 'black' if (cat == 'bottle' and gpart == 'g0') else 'silver' if (cat == 'tin' and gpart == 'g0') else SKIN.BODY_PART[cat]
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f'sk_{cat}_{col}_{part}')
            m.geom_matid[g] = k; m.geom_rgba[g] = SKIN.part_rgba(part, col); sk.append(f'{cat}_{col}_{part}')
        info['objskin_geoms'] = len(sk)
    ctx = {'np': np, 'mujoco': mujoco, 'CAMERAS': CAMERAS, 'log': lg, 'obs': obs, 'bin_fix': bin_fix}
    return e, m, d, info, ctx


def apply_bin_fix_model(m, gids):
    """render_states behaviour: shorten the walls in the model (states are restored; nothing is simulated)."""
    for g in gids:
        m.geom_pos[g, 2] -= 0.003; m.geom_size[g, 2] -= 0.003


class StateRenderer:
    """Restores recorded states tick by tick and returns {feature key: out_size image} and the 16-D state."""

    def __init__(self, run, gpu=0, sim_size=SIZE, out_size=SIZE, objskin='auto', visfix=True, mirror_root=None, skin_py=None,
                 source=None, provenance_source=False):
        run = Path(run)
        self.sim_size, self.out_size = int(sim_size), int(out_size)
        self.filter = resize_filter_name(self.sim_size, self.out_size)
        task = json.loads((run / 'result.json').read_text())['task']
        if objskin == 'auto':
            skin = skin_py or default_skin()
            objskin = '' if task in NO_OBJSKIN_TASKS or skin is None or not Path(skin).is_file() else 'final'
        elif objskin == 'none':
            objskin = ''
        self.e, self.m, self.d, self.info, ctx = build_env(run, gpu, self.sim_size, objskin, visfix, mirror_root, skin_py,
                                                           source, provenance_source)
        self.np, self.mujoco, self.cameras, self.log = ctx['np'], ctx['mujoco'], ctx['CAMERAS'], ctx['log']
        apply_bin_fix_model(self.m, ctx['bin_fix'])
        iid = self.info['instance_id']
        self.actions = self.np.asarray(self.log['actions'], dtype=self.np.float64)
        rec = self.np.load(run / 'actions' / f'{iid}.states.npz')
        self.states, self.spec = rec['states'], int(rec['spec'])
        assert len(self.states) >= len(self.actions), (len(self.states), len(self.actions))
        self.hook = getattr(self.e, '_update_ink', None)
        self.info.update(out_size=self.out_size, resize_filter=self.filter, ink_hook=self.hook is not None)
        self.render_s = 0.

    @property
    def n_frames(self):
        return len(self.actions)

    def restore(self, i):
        self.mujoco.mj_setState(self.m, self.d, self.states[i], self.spec)
        self.mujoco.mj_forward(self.m, self.d)
        if self.hook is not None and i > 0:
            self.hook()   # stamps ink is derived visual state; render_states calls it after every restore but tick 0

    def state(self):
        return proprio_to_state(self.e.robot_proprio())

    def images(self, keys=None, raw=False):
        np = self.np
        t = time.time()
        out = {}
        for c in self.cameras:
            k = CAMERA_TO_FEATURE[c]
            if keys is not None and k not in keys:
                continue
            im = np.ascontiguousarray(np.asarray(self.e.sim.render(camera_name=c, width=self.sim_size, height=self.sim_size))[::-1])
            out[k] = im if raw else np.ascontiguousarray(downscale(im, self.out_size))
        self.render_s += time.time() - t
        return out

    def close(self):
        self.e.close()


def parse_ticks(text, n):
    if not text:
        return list(range(n))
    if ':' in text:
        a, b, *s = text.split(':')
        return list(range(int(a or 0), min(int(b or n), n), int(s[0]) if s else 1))
    return sorted(int(x) for x in text.split(','))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('out_dir')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--sim-size', type=int, default=SIZE, help='MuJoCo camera render size (make_env image_size); default 256')
    ap.add_argument('--out-size', type=int, default=SIZE,
                    help='written frame size, downscaled from --sim-size (area average for integer factors, else Lanczos); '
                         'default 256')
    ap.add_argument('--mp4', action='store_true'); ap.add_argument('--png', action='store_true')
    ap.add_argument('--ticks', default='', help='a:b[:step] or a comma list (default: every action tick)')
    ap.add_argument('--cameras', default='image,wrist_image,right_image')
    ap.add_argument('--objskin', default='auto', choices=('auto', 'final', 'none'))
    ap.add_argument('--skin-py', default=None, help='render-only object skin module (default $ROBOQUEST_SKIN_PY)')
    ap.add_argument('--source', default=None, help='code tree with the task classes (default $ROBOQUEST_SOURCE, else this repo)')
    ap.add_argument('--provenance-source', action='store_true', help="use the source tree named in the run's provenance.json")
    ap.add_argument('--no-visfix', action='store_true')
    ap.add_argument('--crf', type=int, default=18, help='libx264 crf for --mp4')
    a = ap.parse_args(argv)
    if not (a.mp4 or a.png):
        ap.error('pick --mp4 and/or --png')
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in a.cameras.split(',') if k]
    t0 = time.time()
    try:
        resize_filter_name(a.sim_size, a.out_size)
    except ValueError as exc:
        ap.error(str(exc))
    r = StateRenderer(a.run_dir, a.gpu, a.sim_size, a.out_size, a.objskin, not a.no_visfix,
                      mirror_root=out / '.render-assets', skin_py=a.skin_py, source=a.source,
                      provenance_source=a.provenance_source)
    build_s = time.time() - t0
    ticks = parse_ticks(a.ticks, r.n_frames)
    writers = {}
    if a.mp4:
        import imageio.v2 as imageio
        writers = {k: imageio.get_writer(str(out / f'{k}.mp4'), fps=FPS, codec='libx264', macro_block_size=None,
                                         ffmpeg_params=['-crf', str(a.crf), '-pix_fmt', 'yuv420p']) for k in keys}
    if a.png:
        from PIL import Image
        for k in keys:
            (out / 'frames' / k).mkdir(parents=True, exist_ok=True)
    states = []
    start = 0 if r.hook is not None else min(ticks)   # stamps ink accumulates: restore every tick from 0 in order
    for i in range(start, max(ticks) + 1):
        r.restore(i)
        if i not in ticks:
            continue
        states.append(r.state())
        for k, im in r.images(keys).items():
            if a.mp4:
                writers[k].append_data(im)
            if a.png:
                Image.fromarray(im).save(out / 'frames' / k / f'frame_{i:06d}.png')
    for w in writers.values():
        w.close()
    r.close()
    rep = {**r.info, 'run': str(a.run_dir), 'frames_written': len(ticks), 'first_tick': ticks[0], 'last_tick': ticks[-1],
           'cameras': keys, 'mp4': bool(a.mp4), 'png': bool(a.png), 'crf': a.crf if a.mp4 else None,
           'timing_s': {'build': round(build_s, 1), 'render_calls': round(r.render_s, 1), 'total': round(time.time() - t0, 1)},
           'render_ms_per_frame_all_cameras': round(1000 * r.render_s / max(len(ticks), 1), 2)}
    import numpy as np
    np.save(out / 'states.npy', np.asarray(states, np.float32))
    (out / 'rerender.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
