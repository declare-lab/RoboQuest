#!/usr/bin/env bash
# The agent side of the runtime, after scripts/bootstrap_simulator.sh built the simulator side:
#   <runtime>/src/inspect-robots   Inspect Robots at the pinned commit (the harness checks the pin at start)
#   <runtime>/agent-deps           httpx, websockets and the inspect-robots-agent plugin, installed into a
#                                  separate target so the simulator environment stays untouched
# Usage: scripts/bootstrap_agent.sh [--runtime DIR]   (default $ROBOQUEST_RUNTIME or ~/robot-agent-runtime)
set -euo pipefail
cd "$(dirname "$0")/.."
RUNTIME="${ROBOQUEST_RUNTIME:-$HOME/robot-agent-runtime}"
while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
PY="$RUNTIME/envs/robocasa/bin/python"
[ -x "$PY" ] || { echo "no simulator runtime at $RUNTIME (run scripts/bootstrap_simulator.sh first)" >&2; exit 2; }
URL=https://github.com/robocurve/inspect-robots.git
COMMIT=7e4d1b7aee1c0d3cfc3a05a7492b9d12cda666f9
SRC="$RUNTIME/src/inspect-robots"
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet "$URL" "$SRC"
fi
if [ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]; then
  git -C "$SRC" fetch --quiet origin "$COMMIT" || git -C "$SRC" fetch --quiet origin
  git -C "$SRC" checkout --quiet "$COMMIT"
fi
uv pip install --python "$PY" --target "$RUNTIME/agent-deps" -r requirements.agent.txt \
  "$SRC/plugins/inspect-robots-agent" --no-deps
echo "agent runtime ready: $SRC @ $COMMIT, packages in $RUNTIME/agent-deps"
