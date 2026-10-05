#!/usr/bin/env bash
# One episode: scripts/run_episode.py in the agent runtime (see --help).
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_agent.sh scripts/run_episode.py "$@"
