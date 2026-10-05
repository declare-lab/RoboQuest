#!/usr/bin/env bash
# Run a Python entry point in the simulator runtime with the pinned Inspect Robots sources on the path.
set -euo pipefail
cd "$(dirname "$0")/.."
runtime_root="${ROBOQUEST_RUNTIME:-$HOME/robot-agent-runtime}"
if [ ! -x "$runtime_root/envs/robocasa/bin/python" ]; then
  echo "No simulator runtime at $runtime_root: build it with scripts/bootstrap_simulator.sh or set ROBOQUEST_RUNTIME" >&2
  exit 2
fi
inspect_source="$runtime_root/src/inspect-robots"
expected_commit=7e4d1b7aee1c0d3cfc3a05a7492b9d12cda666f9
actual_commit="$(git -C "$inspect_source" rev-parse HEAD 2>/dev/null || echo missing)"
if [ "$actual_commit" != "$expected_commit" ]; then
  echo "Inspect Robots source pin mismatch at $inspect_source (run scripts/bootstrap_agent.sh)" >&2
  exit 1
fi
export ROBOQUEST_RUNTIME="$runtime_root"
export PYTHONPATH="$PWD:$inspect_source/src:$inspect_source/plugins/inspect-robots-agent/src${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$runtime_root/cache/numba}"
exec "$runtime_root/envs/robocasa/bin/python" "$@"
