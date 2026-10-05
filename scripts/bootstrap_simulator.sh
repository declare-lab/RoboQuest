#!/usr/bin/env bash
# Build the RoboQuest simulator runtime on a fresh Ubuntu machine (no root needed).
#
#   scripts/bootstrap_simulator.sh --runtime ~/robot-agent-runtime --assets link:~/robocasa-assets
#   scripts/bootstrap_simulator.sh --runtime ~/robot-agent-runtime --assets download
#   options: [--python 3.11] [--index-url URL] [--no-smoke]
#
# What it does, in order (every step is idempotent, rerun after a failure):
#   1. uv        installs the uv package manager into ~/.local/bin when it is missing
#   2. python    uv fetches a managed CPython (default 3.11, the version the pins were made with)
#   3. venv      <runtime>/envs/robocasa, the interpreter scripts/run.sh uses
#   4. pins      requirements.sim.txt (MuJoCo 3.3.1, NumPy 2.2.5, numba 0.61.2, ...)
#   5. sources   robosuite and RoboCasa cloned into <runtime>/src at the pinned commits, installed
#                editable with --no-deps (RoboCasa's setup.py would otherwise pull torch and lerobot)
#   6. assets    link:DIR   symlinks an asset tree you own (scripts/asset_subset.py makes it, about
#                           7.5 GB) into the RoboCasa source tree, pack by pack
#                copy:DIR   copies that tree into <runtime>/robocasa-assets first, then links it; use this
#                           when the tree belongs to another user, because RoboCasa writes a temporary XML
#                           next to every object model it loads and needs write access to the tree
#                download   the six official RoboCasa packs from Box (about 24 GB; blocked from some networks)
#                skip       nothing (rerun later)
#   7. dirs      <runtime>/cache/numba, results, logs, and <runtime>/env.sh to source before working
#   8. smoke     builds one task instance headless through scripts/run.sh (skipped with --no-smoke or
#                when assets were skipped)
#
# Afterwards:  source <runtime>/env.sh   (exports ROBOQUEST_RUNTIME so scripts/run.sh finds the env)
#              scripts/run.sh scripts/smoke.py                          # every task builds
#              scripts/run.sh scripts/smoke.py --render smoke.png --gpu 0   # cameras via EGL
#
# Behind a slow link to PyPI pass --index-url (for example https://mirrors.aliyun.com/pypi/simple/).
set -euo pipefail

ROBOSUITE_URL=https://github.com/ARISE-Initiative/robosuite.git
ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c   # robosuite 1.5.2
ROBOCASA_URL=https://github.com/robocasa/robocasa.git
ROBOCASA_COMMIT=4f8a2980def75a55dff96b990745b83540425f09    # RoboCasa 1.0.1
ASSET_PACKS="tex tex_generative fixtures_lw objs_objaverse objs_aigen objs_lw"
LINK_DIRS="textures generative_textures fixtures objects/objaverse objects/aigen_objs objects/lightwheel"
NEEDED_DIRS="textures fixtures objects/objaverse objects/aigen_objs objects/lightwheel"

RUNTIME=""
ASSETS="download"
PYVER="3.11"
INDEX_URL="${UV_INDEX_URL:-}"
SMOKE=1

usage() { sed -n '2,27p' "$0"; exit "${1:-0}"; }
log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2 ;;
    --assets) ASSETS="$2"; shift 2 ;;
    --python) PYVER="$2"; shift 2 ;;
    --index-url) INDEX_URL="$2"; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    -h|--help) usage 0 ;;
    *) usage 1 ;;
  esac
done
[ -n "$RUNTIME" ] || usage 1
case "$ASSETS" in download|skip|link:*|copy:*) ;; *) die "--assets must be download, skip, link:DIR or copy:DIR" ;; esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$RUNTIME"
RUNTIME="$(cd "$RUNTIME" && pwd)"
PY="$RUNTIME/envs/robocasa/bin/python"
ASSET_ROOT="$RUNTIME/src/robocasa/robocasa/models/assets"
export PATH="$HOME/.local/bin:$PATH"

log "repo $REPO"
log "runtime $RUNTIME"

# 0. prerequisites
for tool in git curl; do command -v "$tool" >/dev/null || die "$tool is required"; done
command -v ffmpeg >/dev/null || log "WARNING: ffmpeg not found; rollouts run, but videos need it (apt install ffmpeg)"
if command -v nvidia-smi >/dev/null; then
  log "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
else
  log "no nvidia-smi: headless rollouts still work, video rendering needs an NVIDIA GPU with EGL"
fi

# 1. uv
if ! command -v uv >/dev/null; then
  log "installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null || die "uv did not install; see https://docs.astral.sh/uv/"
fi
log "uv $(uv --version)"

# 2. python
log "python $PYVER via uv"
uv python install "$PYVER"

# 3. venv
if [ ! -x "$PY" ]; then
  log "creating venv $RUNTIME/envs/robocasa"
  uv venv --python "$PYVER" "$RUNTIME/envs/robocasa"
fi
log "interpreter $("$PY" -V)"

# 4. pins
log "installing requirements.sim.txt"
uv pip install --python "$PY" ${INDEX_URL:+--index-url "$INDEX_URL"} -r "$REPO/requirements.sim.txt"

# 5. simulator sources at the pinned commits
clone_at() {
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then
    log "cloning $url"
    git clone --quiet "$url" "$dir"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dir" fetch --quiet origin "$commit" || git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
  fi
  log "$(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}
mkdir -p "$RUNTIME/src"
clone_at "$ROBOSUITE_URL" "$RUNTIME/src/robosuite" "$ROBOSUITE_COMMIT"
clone_at "$ROBOCASA_URL" "$RUNTIME/src/robocasa" "$ROBOCASA_COMMIT"
log "installing robosuite and robocasa editable, no dependencies"
uv pip install --python "$PY" ${INDEX_URL:+--index-url "$INDEX_URL"} --no-deps -e "$RUNTIME/src/robosuite" -e "$RUNTIME/src/robocasa"

# 6. assets
if [ "${ASSETS#copy:}" != "$ASSETS" ]; then
  SRC="${ASSETS#copy:}"
  SRC="${SRC/#\~/$HOME}"
  [ -d "$SRC" ] || die "asset source $SRC is not a directory"
  if [ ! -d "$RUNTIME/robocasa-assets" ]; then
    log "copying $SRC to $RUNTIME/robocasa-assets (about 7.5 GB; RoboCasa needs a writable tree)"
    cp -rL "$SRC" "$RUNTIME/robocasa-assets.partial" && mv "$RUNTIME/robocasa-assets.partial" "$RUNTIME/robocasa-assets"
  else
    log "using the existing copy at $RUNTIME/robocasa-assets"
  fi
  ASSETS="link:$RUNTIME/robocasa-assets"
fi
case "$ASSETS" in
  download)
    log "downloading the RoboCasa asset packs ($ASSET_PACKS), about 24 GB, into $ASSET_ROOT"
    ok=0
    for attempt in 1 2 3; do
      # shellcheck disable=SC2086
      if "$PY" -m robocasa.scripts.download_kitchen_assets --type $ASSET_PACKS; then ok=1; break; fi
      log "download attempt $attempt failed, retrying in 30 s"
      sleep 30
    done
    [ "$ok" = 1 ] || die "asset download failed three times; get the asset tree another way and rerun with --assets link:DIR"
    ;;
  link:*)
    SRC="${ASSETS#link:}"
    SRC="${SRC/#\~/$HOME}"
    [ -d "$SRC" ] || die "asset source $SRC is not a directory"
    # The RoboCasa clone already tracks a few files inside fixtures/, textures/ and objects/ (identical copies of
    # pack files), so a pack cannot be linked as one directory: merge instead, linking every entry the clone
    # lacks and descending only into directories both sides have.
    LINKED=0
    link_merge() {
      local src="$1" dst="$2" child name
      mkdir -p "$dst"
      for child in "$src"/*; do
        [ -e "$child" ] || continue
        name="$(basename "$child")"
        if [ -L "$dst/$name" ]; then
          # a link from an earlier run: repoint it, the asset tree may have moved since
          if [ "$(readlink "$dst/$name")" != "$(readlink -f "$child")" ]; then
            ln -sfn "$(readlink -f "$child")" "$dst/$name"
            LINKED=$((LINKED + 1))
          fi
        elif [ ! -e "$dst/$name" ]; then
          ln -s "$(readlink -f "$child")" "$dst/$name"
          LINKED=$((LINKED + 1))
        elif [ -d "$child" ] && [ -d "$dst/$name" ]; then
          link_merge "$child" "$dst/$name"
        fi
      done
    }
    for sub in $LINK_DIRS; do
      [ -d "$SRC/$sub" ] || continue
      LINKED=0
      link_merge "$SRC/$sub" "$ASSET_ROOT/$sub"
      log "linked $LINKED new entries into $sub"
    done
    probe="$SRC/objects/objaverse"
    if [ -d "$probe" ] && [ ! -w "$probe" ]; then
      log "WARNING: $probe is not writable by you; RoboCasa writes a temporary XML next to each object it loads, so scenes will fail with PermissionError. Rerun with --assets copy:$SRC"
    fi
    ;;
  skip) log "assets skipped" ;;
esac
missing=0
for sub in $NEEDED_DIRS; do
  if [ ! -d "$ASSET_ROOT/$sub" ] || [ -z "$(ls -A "$ASSET_ROOT/$sub" 2>/dev/null)" ]; then
    log "NOTE: asset pack $sub is not installed yet"; missing=1
  fi
done

# 7. runtime directories and the env file
mkdir -p "$RUNTIME/cache/numba" "$RUNTIME/results" "$RUNTIME/logs"
cat > "$RUNTIME/env.sh" <<EOF
# source this file before using scripts/run.sh from the RoboQuest checkout
export ROBOQUEST_RUNTIME="$RUNTIME"
export PATH="\$HOME/.local/bin:\$PATH"
EOF
log "wrote $RUNTIME/env.sh"

# 8. smoke
if [ "$SMOKE" = 1 ] && [ "$ASSETS" != skip ] && [ "$missing" = 0 ]; then
  log "smoke: building one puzzle_box instance headless"
  ROBOQUEST_RUNTIME="$RUNTIME" "$REPO/scripts/run.sh" "$REPO/scripts/smoke.py" --tasks puzzle_box
fi
log "done. Next: source $RUNTIME/env.sh; then see README.md"
