<div align="center">

<h1>RoboQuest</h1>

<h3>Generalist Physical Agents that Search, Inspect and Test</h3>

<h4><a href="https://declare-lab.github.io/RoboQuest/">Project Page</a> | <a href="https://arxiv.org/abs/2610.10388">Paper</a> | <a href="https://huggingface.co/datasets/declare-lab/RoboQuest">Dataset</a></h4>

</div>

RoboQuest is a benchmark for goal-directed embodied exploration, in which a mobile manipulator must gather
task-relevant information through physical interaction, act on the evidence, and decide for itself when the task is
done.

## ✨ Highlights

- 🏠 **Ten mobile-manipulation tasks** in RoboCasa365 kitchens, in three families: Search & Explore, Object
  Inspect and Testing.
- 🔍 **Hidden information by design**: every task hides what it needs (where the targets are, a label under a
  vessel, which parcel is odd, the order in which bolts block each other, ...), and the agent ends an episode by
  physically pressing a Submit button.
- 🎯 **500 evaluation instances**, 50 per task, with held-out kitchen styles; scored by success rate and a staged
  progress measure.
- 🤖 **Five frontier multimodal models** evaluated zero-shot through one visuomotor interface (GPT-6 Astra, Claude
  Opus 5.5, GPT-6.1 Sol, Claude Fable 5.1, Gemini 3.8 Flash); the best succeeds in 23.2% of episodes.
- 🔌 **Any model or policy**: API models from the major providers or any OpenAI-compatible server, external CLI
  agents, and robot policies over openpi's WebSocket protocol.
- 💽 **5,000 verified demonstrations** (500 per task, 366 hours at 20 Hz) with two-level language annotations, in
  LeRobot format and as replayable simulator recordings.

## 🧩 Tasks

| Family | Task | Hidden information | What the agent must do |
|---|---|---|---|
| Search & Explore | Locked Storage (`locked_storage`) | the target behind locks | Find an object behind compartments that open only while a matching coloured token rests on their reader. |
| | Search Room (`search_room`) | where the targets are | Find objects hidden anywhere in the kitchen, in compartments with handles, and collect them on a tray. |
| | Blackout Search (`blackout_search`) | targets in the dark | The same in a dark kitchen, with a lantern to carry. |
| Object Inspect | Painted Cubes (`painted_cubes`) | marks on unseen faces | Sort cubes into a blue and a yellow bin by a rule about marks on their faces, some of which are hidden. |
| | Marked Mugs (`marked_mugs`) | labels under vessels | Put each mug or bowl upright on the pad matching the coloured label on its bottom, with its own two balls inside. |
| | Unfamiliar Containers (`unfamiliar_containers`) | how containers open | Open unfamiliar boxes (lids, drawers, knobs, doors) to get the items inside into a bowl. |
| Testing | Puzzle Box (`puzzle_box`) | bolt blocking order | Open a wooden puzzle box whose lid is locked by sliding bolts, take out the item inside and stand it in the tray. |
| | Stamp Composition (`stamps`) | patterns and rotations | Reproduce a reference dot pattern on a board with stamps whose patterns and orientations are unknown until tried. |
| | Wobbly Stand (`wobbly_stand`) | short legs and gaps | Level a wobbly stand with shims so a ball stays still on top. |
| | Odd Parcel (`odd_parcel`) | which parcel is odd | Find the parcels that weigh differently (a balance stands nearby) and put exactly those in the box. |

Each instance's exact goal text is in its registry row (`suite/v1/<task>/registry/registry.json`).

## 🛠️ Installation

### Requirements

- Linux (x86-64) with an NVIDIA GPU and its driver. Cameras render headless through EGL; each running episode needs
  about 2 GB of GPU memory. No root access is needed.
- `git` and `curl`; `ffmpeg` for episode videos (episodes run without it).
- Disk: about 2 GB for the runtime, plus RoboCasa's kitchen assets: 7 GB when made from assets you already have,
  24 GB when downloaded (RoboCasa's downloader fetches its packs whole).

Python and every package are installed by the setup scripts with [uv](https://docs.astral.sh/uv/) into one runtime
folder you choose, as a uv virtual environment. Nothing is installed system-wide and no conda is involved.

### Install

```bash
git clone <repository URL> roboquest && cd roboquest
# simulator: uv, Python 3.11 and a virtual environment at ~/roboquest-runtime/envs/robocasa with MuJoCo, robosuite
# and RoboCasa at pinned versions, the kitchen assets, then one task built as a check (a few minutes plus the download)
scripts/bootstrap_simulator.sh --runtime ~/roboquest-runtime --assets download
# model client: Inspect Robots (pinned) and its packages in ~/roboquest-runtime/agent-deps (under a minute)
scripts/bootstrap_agent.sh --runtime ~/roboquest-runtime
# in every new shell: tells the scripts where the runtime is (sets ROBOQUEST_RUNTIME)
source ~/roboquest-runtime/env.sh
```

Every step can be rerun after a failure. Instead of the download:

```bash
# you already have RoboCasa's assets: copy out the 7 GB RoboQuest uses and set up from that copy
mkdir -p ~/roboquest-assets
python3 scripts/asset_subset.py --assets-root /path/to/robocasa/robocasa/models/assets --tar - | tar -C ~/roboquest-assets -xf -
scripts/bootstrap_simulator.sh --runtime ~/roboquest-runtime --assets link:$HOME/roboquest-assets
# slow connection to PyPI: install from a mirror
scripts/bootstrap_simulator.sh --runtime ~/roboquest-runtime --assets download --index-url https://mirrors.aliyun.com/pypi/simple/
```

You do not need to activate the environment: the scripts in `scripts/` run its Python themselves and also put the
repository and the model client on the path, render headless (`MUJOCO_GL=egl`) and use one thread per math library.
To use the simulator side by hand, for example in a notebook (tasks, scenes, cameras; run models through the scripts):

```bash
source ~/roboquest-runtime/envs/robocasa/bin/activate
export PYTHONPATH=$PWD MUJOCO_GL=egl
```

### Check the installation

```bash
# every task builds; ends with "ALL CHECKS PASSED"
scripts/run.sh scripts/smoke.py
# a whole episode against a mocked model (no key, no API use); ends with "completed": true,
# the episode is saved under artifacts/episodes/
bash scripts/run_episode.sh --agent api --model anthropic/claude-opus-5-5 --task roboquest_puzzle_box --offline
```

## 🧑‍💻 Usage

### Add an API key

```bash
# copy the template, keep it private, then fill in the keys you use (which provider reads which: [Connecting a model](#connecting-a-model))
cp keys.env.example keys.env && chmod 600 keys.env
# every run then takes --env-file keys.env; or export the variable instead, for example
export OPENROUTER_API_KEY=...
```

### Run a model

```bash
# one episode: a task's default development instance (--list-tasks lists the tasks), on GPU 0 (--physical-gpu N)
bash scripts/run_episode.sh --agent api --model openrouter/qwen/qwen3-vl-235b-a22b-instruct --env-file keys.env \
  --task roboquest_puzzle_box
# the benchmark: 500 episodes, then the per-task table (options: [Run the benchmark](#run-the-benchmark))
bash scripts/benchmark.sh --model openrouter/qwen/qwen3-vl-235b-a22b-instruct --env-file keys.env
```

### Connecting a model

`--model <provider>/<model id>` names the provider and the model's id there. The provider fixes the API format, the
endpoint and the variable the key is read from:

| Provider | Example | Key | API format |
|---|---|---|---|
| `openai` | `openai/gpt-6-astra` | `OPENAI_API_KEY` | OpenAI Responses |
| `openrouter` | `openrouter/qwen/qwen3-vl-235b-a22b-instruct` | `OPENROUTER_API_KEY` | OpenAI Chat Completions |
| `anthropic` | `anthropic/claude-opus-5-5` | `ANTHROPIC_API_KEY` | Anthropic Messages |
| `google` | `google/gemini-3.8-flash` | `GEMINI_API_KEY` | Gemini API (native; keeps Gemini's thought signatures across turns) |
| `vertex/anthropic`, `vertex/google` | `vertex/anthropic/claude-opus-5-5` | `GOOGLE_APPLICATION_CREDENTIALS` | as `anthropic` and `google`, through Google Cloud Vertex AI |
| `openai-compatible` | `openai-compatible/Qwen/Qwen3-VL-32B-Instruct` | `OPENAI_API_KEY`, if the server needs one | OpenAI Chat Completions at `--base-url` |

```bash
# any server that speaks OpenAI Chat Completions: vLLM, SGLang, Ollama, LM Studio, Together, Fireworks, DeepSeek,
# Groq, Azure OpenAI, ...; for a provider without such an endpoint (AWS Bedrock, ...), an OpenAI-compatible proxy
# such as LiteLLM (https://docs.litellm.ai/) in front of it
bash scripts/run_episode.sh --agent api --model openai-compatible/Qwen/Qwen3-VL-32B-Instruct \
  --base-url http://localhost:8000/v1 --task roboquest_puzzle_box

# Google Cloud Vertex AI: a service-account file instead of a key; region from GOOGLE_CLOUD_LOCATION (default global)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
bash scripts/run_episode.sh --agent api --model vertex/anthropic/claude-opus-5-5 --task roboquest_puzzle_box

# reasoning effort and output limit; without them the provider's defaults apply (Anthropic and Gemini need a limit
# and get 16000 and 32768)
bash scripts/run_episode.sh --agent api --model anthropic/claude-opus-5-5 --effort medium --max-output-tokens 16000 \
  --env-file keys.env --task roboquest_puzzle_box

# transient provider errors (429, 5xx, timeouts) are retried, never changing what the model sees: attempts per
# request (default 6), a growing wait between attempts, and the wait the provider asks for
bash scripts/run_episode.sh --agent api --model anthropic/claude-opus-5-5 --env-file keys.env --task roboquest_puzzle_box \
  --max-request-attempts 6 --retry-backoff-s 5 --honor-retry-after

# an external CLI agent instead of an API model: the runner writes a session folder (instructions, tool definitions,
# observations) with act.py, through which the agent submits one action per decision
bash scripts/run_episode.sh --agent codex --task roboquest_puzzle_box
```

### Running a robot policy

A learned policy such as π0.5 runs the same episodes through [openpi](https://github.com/Physical-Intelligence/openpi)'s
WebSocket protocol: serve the checkpoint with openpi, then point RoboQuest at the server. Each inference gets the
three cameras, the robot state and the task's goal, and returns a chunk of actions that run one per 20 Hz tick;
then the policy is asked again with a new observation. An episode ends at the physical Submit press or at the
tick budget: 36,000 ticks, 30 minutes of simulated time, for every task (`--horizon` changes it).

```bash
# check the wiring with the built-in stand-in server: it keeps the robot still, so the episode runs to its budget
bash scripts/run_agent.sh scripts/serve_test_policy.py --port 8000 &
bash scripts/run_episode.sh --agent policy --policy-url ws://localhost:8000 --task roboquest_puzzle_box --horizon 400

# a real policy: serve the checkpoint in openpi's own environment (openpi listens on port 8000) ...
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint dir>
# ... then run one episode, or the benchmark
bash scripts/run_episode.sh --agent policy --policy-url ws://localhost:8000 --task roboquest_puzzle_box
bash scripts/benchmark.sh --agent policy --policy-url ws://localhost:8000 --out runs/my-policy
```

What the policy receives and returns:

| | Key | Contents |
|---|---|---|
| request | `observation/image`, `observation/right_image`, `observation/wrist_image` | the left and right agent-view cameras and the wrist camera, 256×256×3 uint8, upright (`--image-size 512` for 512 px) |
| | `observation/state` | 16 floats: end-effector position (3) and quaternion xyzw (4) in the robot base frame, base position (3) and quaternion xyzw (4) in the world, the two gripper finger joints (2) |
| | `prompt` | the task's goal text |
| | `__openpi_rng_seed__` | a sampling seed for this request (from `--policy-seed`, the instance and the inference index); servers that do not use it ignore it |
| reply | `actions` | a chunk of N × 12 native RoboCasa PandaOmron actions: arm motion (6, controller deltas in the base frame), gripper (1: > 0 closes, < 0 opens, 0 holds), base velocity (3), torso (1, held), mode (1: > 0 the arm's target follows the moving base) |
| | `subtask` | optional: a predicted subtask text, logged per inference (`--require-subtask` makes it mandatory) |

Each action runs as the policy sent it, clipped to the controller's range [-1, 1], with the torso held (as for every
RoboQuest agent): the arm and the base may move in the same tick, and the gripper closes on a positive value, opens
on a negative one and holds on zero. `--action-filter demos` instead shapes actions like RoboQuest's own
demonstrations, for policies trained on them: the arm or the base per tick (the mode picks one), an open or
closed gripper, and the API agents' per-tick speed limits (arm position ±0.5, rotation ±0.3, base ±0.5).
`--replan-every N` executes only the first N actions of each chunk. Every inference is recorded in the episode's
`inference.jsonl`.

The public openpi checkpoints are trained on other robots, cameras and action spaces, and have never seen a
mobile base, these kitchens or a Submit button: a policy needs fine-tuning on RoboCasa PandaOmron data with this
observation and action layout before it can do anything useful here.

### Run the benchmark

```bash
M=openrouter/qwen/qwen3-vl-235b-a22b-instruct
# the 500 evaluation instances (50 per task), 200 decisions each and no tick budget; every episode goes to
# runs/<model>/ and the per-task table of success and mean progress is printed at the end
bash scripts/benchmark.sh --model $M --env-file keys.env
# only some tasks
bash scripts/benchmark.sh --model $M --env-file keys.env --tasks puzzle_box,stamps
# episodes at once and the GPUs they share (each episode needs about 2 GB of GPU memory)
bash scripts/benchmark.sh --model $M --env-file keys.env --gpus 0,1 --max-parallel 8
# finish an interrupted run (reruns only the episodes without a result)
bash scripts/benchmark.sh --model $M --env-file keys.env --resume
# print the plan without running anything; another output folder: --out DIR
bash scripts/benchmark.sh --model $M --dry-run
# the per-task table again
bash scripts/benchmark.sh --model $M --summarize
```

### What an episode saves

Each episode writes a directory (`--out`; default `artifacts/episodes/<run id>` for one episode,
`runs/<model>/<task>/<instance>` in a benchmark run), on average about 475 MB, so a full run of 500 needs about
240 GB:

| File | Contents | Typical size |
|---|---|---|
| `result.json` | success, termination, decisions, timings, token use | small |
| `task-progress.json` | progress `P` with its stages and flags | small |
| `config.json`, `instruction.txt`, `system-prompt.txt`, `tool-schemas.json` | the run settings and what the model was told | small |
| `transcripts.json`, `turns.json`, `commands.json`, `usage.json` | the conversation, each decision and robot command, token use | a few MB |
| `rollout.mp4` | the three cameras side by side, start to finish, 10 fps | ~40 MB |
| `initial-*.png`, `final-*.png`, `observations/` | first and last camera frames; the frames the model saw at every decision | ~150 MB |
| `provider/` | every API request and response, as sent and received | ~200 MB |
| `physical-trace-private.jsonl` | the simulator state at every 20 Hz tick (progress is computed from it) | ~130 MB |
| `inference.jsonl` (`--agent policy`) | every inference: its seed, latency, the actions received and the actions executed | grows with the episode |

### Replay a recorded episode

```bash
# rerun an episode recorded with this repository from its recorded provider responses, and compare every request,
# the commands, the physics and the score with the recording (for example on another machine or installation)
bash scripts/run_agent.sh scripts/verify_replay.py --one <episode dir> --gpu 0 --out replay-check
```

## 📏 Scoring

- **Success rate**: the share of episodes whose goal state holds at a valid press of the physical Submit button.
  Submitting ends the episode; not submitting fails. The task code decides it at the press (`score.success` in
  `result.json`).
- **Progress** (`P`, 0 to 1): a per-task staged measure of how much of the goal state the task's own objects
  reach, computed on the final state of every episode however it ended, with a 0.1 tidiness deduction in four
  tasks (a non-target object on the tray in Locked Storage, Search Room and Blackout Search; the balance moved in
  Odd Parcel), clipped to [0, 1]. Every finished episode gets `task-progress.json` beside its `result.json`.
  (`score.progress` in `result.json` is the task code's own running reading, not the benchmark's progress.)
- Two task rules in brief: in Odd Parcel a parcel left on a balance pan is reported but neither fails the episode
  nor costs progress (the goal does not ask to clear the pans; moving the balance does count); in Marked Mugs a
  vessel's placement credit is 1 when at least 3/4 of its footprint disc lies on its own pad and 0.5 when at least
  1/2 does, and success needs full credit and the vessel's own two balls for every vessel.

The complete definitions are in [docs/progress.md](docs/progress.md).

## 🏆 Results

We ran five frontier models on the same 500 evaluation episodes (`suite/v1/eval50/`), 50 for each task, with no
task-specific training or examples. GPT-6 Astra, Claude Opus 5.5, GPT-6.1 Sol and Claude Fable 5.1 ran with medium
thinking (`--effort medium`) and Gemini 3.8 Flash with high thinking (`--effort high`).

<p>
  <img src="assets/readme/leaderboard.png" width="100%" alt="Bar charts of the overall results. Success rate: GPT-6 Astra 23.2%, Claude Opus 5.5 13.8%, GPT-6.1 Sol 12.2%, Claude Fable 5.1 11.4%, Gemini 3.8 Flash 2.0%. Task progress: 48.0%, 37.3%, 34.0%, 31.0% and 9.2% in the same order. Cost per episode at list prices: GPT-6.1 Sol $1.99, Gemini 3.8 Flash $3.37, Claude Opus 5.5 $6.83, GPT-6 Astra $11.29, Claude Fable 5.1 $19.22.">
</p>

GPT-6 Astra leads, but even it solves fewer than a quarter of the episodes. Puzzle Box is by far the easiest task:
without it, Astra's success rate falls to 15.1% and every other model's to under 8%.

### Results by task

<p>
  <img src="assets/readme/success-by-task.png" width="100%" alt="Dot plot of each model's success rate on the ten tasks. The best result per task: Locked Storage 28.0%, Search Room 2.0%, Blackout Search 0.0%, Painted Cubes 30.0%, Marked Mugs 38.0%, Unfamiliar Containers 6.0%, Puzzle Box 96.0%, Stamp Composition 14.0%, Wobbly Stand 24.0%, Odd Parcel 10.0%.">
</p>

<p>
  <img src="assets/readme/progress-by-task.png" width="100%" alt="Dot plot of each model's average task progress on the ten tasks. The best result per task: Locked Storage 52.9%, Search Room 31.0%, Blackout Search 23.7%, Painted Cubes 74.8%, Marked Mugs 69.7%, Unfamiliar Containers 32.2%, Puzzle Box 99.6%, Stamp Composition 55.3%, Wobbly Stand 32.2%, Odd Parcel 22.3%.">
</p>

The two search tasks are almost untouched: across all five models, Search Room and Blackout Search were solved once
in 500 episodes, even though most models get part of the way. GPT-6 Astra makes the most progress on every task
except Wobbly Stand, where Claude Opus 5.5 leads.

### Success and cost

<p>
  <img src="assets/readme/success-vs-cost.png" width="100%" alt="Scatter plot of success rate against cost per episode on a log scale. GPT-6.1 Sol: $1.99 and 12.2%. Gemini 3.8 Flash: $3.37 and 2.0%. Claude Opus 5.5: $6.83 and 13.8%. GPT-6 Astra: $11.29 and 23.2%. Claude Fable 5.1: $19.22 and 11.4%. GPT-6.1 Sol, Claude Opus 5.5 and GPT-6 Astra lie on the Pareto line.">
</p>

GPT-6.1 Sol is the cheapest model to run, at $1.99 an episode and $16.31 for each solved one. GPT-6 Astra's lead
costs $11.29 an episode, and Claude Fable 5.1 is the most expensive at $19.22.

### How episodes end

<p>
  <img src="assets/readme/episode-ends.png" width="100%" alt="Stacked bars of how each model's 500 episodes ended, next to bars of submission precision. Submitted correctly: GPT-6 Astra 23%, Claude Opus 5.5 14%, GPT-6.1 Sol 12%, Claude Fable 5.1 11%, Gemini 3.8 Flash 2%. Submitted wrongly: 62%, 32%, 63%, 54% and 56%. Claude Opus 5.5 gave up in 49% of episodes and Claude Fable 5.1 in 29%; Gemini 3.8 Flash ran out of decisions in 42% and GPT-6.1 Sol in 22%. Submission precision: Claude Opus 5.5 30.4%, GPT-6 Astra 27.3%, Claude Fable 5.1 17.5%, GPT-6.1 Sol 16.3%, Gemini 3.8 Flash 3.5%.">
</p>

An episode ends when the robot presses SUBMIT, when the model gives up or answers in text, or when it runs out of
its 200 decisions. The models fail in different ways: GPT-6 Astra and GPT-6.1 Sol press SUBMIT in most episodes,
often before the task is done; Claude Opus 5.5 gives up in about half of its episodes; and Gemini 3.8 Flash often
runs out of decisions.

The numbers behind these charts are in [results/eval50/](results/eval50/): `summary.md` and `summary.csv` per task
and model, and [episodes.csv](results/eval50/episodes.csv) with one row per episode (success, progress, how it
ended, decisions, time and tokens). Costs use the providers' list prices. The
[project page](https://declare-lab.github.io/RoboQuest/) has interactive versions of these charts.

## 💽 Demonstration Dataset

5,000 verified demonstrations, 500 per task, recorded at 20 Hz (about 366 hours), with a two-level annotation of
every episode (stages and subtasks). Hosted on Hugging Face at [declare-lab/RoboQuest](https://huggingface.co/datasets/declare-lab/RoboQuest), in two forms:

- `lerobot/<task>/`: LeRobot v2.1 datasets with three 256×256 camera streams (left and right scene cameras, wrist),
  the 16-D state and 12-D action at every frame, the goal text as the task, and per-frame stage and subtask indices.
- `raw/<task>/<id>/`: the recorded episodes themselves: the simulator state at every tick, the actions, the stage
  segments and the captions, enough to replay an episode exactly and render it again at any resolution.

The scene of every demonstration is in `suite/v1/<task>/demos/` (development scenes only; none is an evaluation
scene). The tools that made the LeRobot datasets are in `scripts/dataset/`:

```bash
# render a recorded episode again from its simulator states: 512-pixel renders downscaled to 256, as MP4
MUJOCO_GL=egl python scripts/dataset/rerender.py <raw episode dir> <out dir> --sim-size 512 --out-size 256 --mp4
```

`scripts/dataset/README.md` covers re-rendering and the LeRobot v2.1 conversion (`convert_lerobot_v21.py`).

## 🗂️ Repository Structure

```text
roboquest/tasks/          one RoboCasa kitchen task per benchmark task
roboquest/<family>/       task geometry, objects, rules, scene building, visibility and reach checks
roboquest/base/           scene base classes: the kitchen scene, support and resting contracts, the Submit button
roboquest/harness/        episode runner, tool protocol and prompt, model providers, recording
roboquest/scoring/        progress and the per-cell grading of printed stamp dots
suite/v1/<task>/registry/ frozen instances (development and evaluation rows) with their gate reports
suite/v1/<task>/demos/    the scenes of the demonstration dataset (500 per task)
suite/v1/eval50/          the 50 evaluation instance ids per task
results/eval50/           results per episode (episodes.csv) and per task and model (summary.csv, summary.md)
docs/progress.md          the scoring: success and progress, task by task
scripts/                  episode and benchmark launchers, replay verifier, setup scripts
scripts/dataset/          re-rendering recorded demonstrations and the LeRobot v2.1 conversion
```

## 📝 Citation

```bibtex
@article{liu2026roboquest,
  title   = {{RoboQuest}: Generalist Physical Agents that Search, Inspect and Test},
  author  = {Liu, Renhang and Majumder, Navonil and Pala, Tej Deep and Poria, Soujanya},
  journal = {arXiv preprint arXiv:2610.10388},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.10388}
}
```

## 🙏 Acknowledgements

RoboQuest builds on [RoboCasa](https://robocasa.ai/) (kitchens and assets, including RoboCasa365),
[robosuite](https://robosuite.ai/) and [MuJoCo](https://mujoco.org/) (simulation),
[Inspect Robots](https://github.com/robocurve/inspect-robots) (the agent harness),
[openpi](https://github.com/Physical-Intelligence/openpi) (the robot-policy protocol and the π0.5 baseline) and
[LeRobot](https://github.com/huggingface/lerobot) (the dataset format). The lantern in Blackout Search is
[Lantern 01](https://polyhaven.com/a/Lantern_01) from Poly Haven (CC0). We thank their authors and maintainers.

## ⚖️ License

To be announced.

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="MuJoCo" src="https://img.shields.io/badge/Simulator-MuJoCo-475569?style=flat-square&labelColor=64748b" height="22"/>&nbsp;
  <img alt="RoboCasa" src="https://img.shields.io/badge/Kitchens-RoboCasa365-475569?style=flat-square&labelColor=64748b" height="22"/>&nbsp;
  <img alt="LeRobot v2.1" src="https://img.shields.io/badge/Dataset-LeRobot_v2.1-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>
