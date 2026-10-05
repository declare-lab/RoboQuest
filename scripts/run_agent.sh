#!/usr/bin/env bash
# run_inspect.sh plus the provider packages (httpx, websockets, inspect_robots_agent) on the path.
set -euo pipefail
cd "$(dirname "$0")/.."
agent_deps="${ROBOQUEST_AGENT_DEPS:-${ROBOQUEST_RUNTIME:-$HOME/robot-agent-runtime}/agent-deps}"
export PYTHONPATH="$agent_deps${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${ROBOQUEST_NUMBA_CACHE:-/tmp/active-bench-agent-numba}"
exec bash scripts/run_inspect.sh "$@"
