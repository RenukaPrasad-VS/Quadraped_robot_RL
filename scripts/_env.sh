#!/usr/bin/env bash
# Locate Isaac Sim / Isaac Lab and put this project on PYTHONPATH.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_DIR="${ISAACSIM_DIR:-$HOME/issacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
if [[ ! -x "$ISAACSIM_DIR/python.sh" ]]; then
  echo "Isaac Sim not found at $ISAACSIM_DIR (set ISAACSIM_DIR)" >&2; exit 2
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PY="$ISAACSIM_DIR/python.sh"
