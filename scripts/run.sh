#!/usr/bin/env bash
# Run a Python entry point in the simulator runtime only (no provider packages), rendering through EGL.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNTIME_ROOT="${ROBOQUEST_RUNTIME:-$HOME/robot-agent-runtime}"
if [ ! -x "$RUNTIME_ROOT/envs/robocasa/bin/python" ]; then
  echo "scripts/run.sh: no simulator runtime at $RUNTIME_ROOT; build it with scripts/bootstrap_simulator.sh or set ROBOQUEST_RUNTIME" >&2
  exit 2
fi
export ROBOQUEST_RUNTIME="$RUNTIME_ROOT"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$RUNTIME_ROOT/cache/numba"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$RUNTIME_ROOT/envs/robocasa/bin/python" "$@"
