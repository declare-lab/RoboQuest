# Dataset tools: re-rendering at another resolution

The RoboQuest demonstrations record the simulator state of every tick, so their camera frames can be rendered
again at any resolution without re-simulating. The published dataset uses `--sim-size 256 --out-size 256`.

- `--sim-size N`: the MuJoCo camera render size (the `image_size` the environment is built with).
- `--out-size M`: the size of the frames that are written. When `N > M` they are downscaled (an exact area
  average when `N` is a multiple of `M`, Lanczos otherwise); `M > N` is refused. Both default to 256.

Both tools need the simulator runtime (`scripts/bootstrap_simulator.sh` in the release, or an existing RoboCasa
environment), an EGL-capable GPU (`MUJOCO_GL=egl`), and an episode run directory:

    <run_dir>/result.json                      task and instance_id
    <run_dir>/actions/<instance_id>.json       the instance and the logged actions
    <run_dir>/actions/<instance_id>.states.npz the recorded simulator states (states, spec)

The task code comes from this repository; `--source DIR` (or `$ROBOQUEST_SOURCE`) uses another checkout, and
`--provenance-source` the tree named in the run's `provenance.json`.

## One episode: `rerender.py`

    MUJOCO_GL=egl python scripts/dataset/rerender.py <run_dir> <out_dir> --sim-size 512 --out-size 256 --mp4
    MUJOCO_GL=egl python scripts/dataset/rerender.py <run_dir> <out_dir> --sim-size 1024 --out-size 512 --png --ticks 1500:1540

Every tick is restored with `mj_setState` + `mj_forward` (plus the task's own per-step visual update, the stamps
ink) and the three policy cameras are rendered: `image` (left agent view), `right_image` (right agent view) and
`wrist_image` (eye in hand). Frame `t` is the recorded state before action `t`. Outputs: `<camera>.mp4` (H.264 for
viewing, `--mp4`) and/or `frames/<camera>/frame_XXXXXX.png` (`--png`), `states.npy` (the 16-D state of each written
frame) and `rerender.json` (sizes, filter, render-only fixes, timings). `--cameras` picks cameras, `--ticks a:b[:step]`
or a comma list picks ticks, `--gpu` the EGL device.

The scenes are built by this repository's task code, which already includes the cube-bin walls, the hidden mug
marker and the search-pool object skin, so the frames look like the published dataset's. The `--no-visfix`,
`--objskin` and `--skin-py` options only matter when `--source` points at another checkout of the task code; note
that with the default `--objskin auto`, a `ROBOQUEST_SKIN_PY` variable set in your shell applies that external skin
on top of the in-scene one, so leave it unset. `rerender.py` keeps a mirror of symlinks to the scene assets in
`<out_dir>/.render-assets`.

## A LeRobot v2.1 dataset: `convert_lerobot_v21.py`

Video-mode LeRobot v2.1 with the same features as the image-mode export (`image`, `wrist_image`, `right_image`,
16-D `state`, 12-D `actions`, fps 20) plus per-frame `subtask_index` / `stage_index` from the caption timelines.
The modes run in two Pythons: `$ROBOCASA_PY` (simulator) and `$LEROBOT_PY` (the pinned openpi LeRobot); both default
to the running python.

    python scripts/dataset/convert_lerobot_v21.py pick  <selection.json> <per_task> <captions_root> picks.json
    python scripts/dataset/convert_lerobot_v21.py drive picks.json stage --gpus 0,1 --workers 4 --sim-size 512 --out-size 256
    $LEROBOT_PY scripts/dataset/convert_lerobot_v21.py write picks.json stage dataset roboquest/<task>
    $LEROBOT_PY scripts/dataset/convert_lerobot_v21.py check picks.json stage dataset check roboquest/<task>

`drive` launches one `render` per episode (each streams raw frames into three `encode` children with LeRobot's own
video settings); `write` builds the dataset and records `roboquest.sim_size` / `roboquest.out_size` /
`roboquest.resize_filter` in `meta/info.json` and `meta/roboquest_conversion.json`; `check` decodes frames through
the dataset and compares them with the lossless PNGs of the same ticks.
