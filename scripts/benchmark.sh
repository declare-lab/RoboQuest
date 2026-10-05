#!/usr/bin/env bash
# Run the benchmark: the 500 evaluation instances (50 per task) for one model, then print the
# per-task table. Results go to runs/<model> unless --out says otherwise; rerun with --resume to finish an
# interrupted run. All other options are scripts/run_benchmark.py's (--tasks, --gpus, --max-parallel, ...).
#
#   bash scripts/benchmark.sh --model openrouter/qwen/qwen3-vl-235b-a22b-instruct
#   bash scripts/benchmark.sh --model anthropic/claude-opus-5-5 --effort medium
#   bash scripts/benchmark.sh --model openai-compatible/Qwen/Qwen3-VL-32B-Instruct --base-url http://localhost:8000/v1
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_agent.sh scripts/run_benchmark.py --set eval50 "$@"
