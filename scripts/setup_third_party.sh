#!/usr/bin/env bash
# Clone Unitree's repositories into third_party/ at the exact commits this project was built and tested with,
# then install unitree_rl_lab into the Isaac Sim Python (skip that step with --no-install).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

RL_LAB_URL=https://github.com/unitreerobotics/unitree_rl_lab
RL_LAB_COMMIT=4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
ROS_URL=https://github.com/unitreerobotics/unitree_ros
ROS_COMMIT=157ff950351e5f535e03c5d85a16b7d594b3dbcc

fetch() {  # fetch <dir> <url> <commit> [sparse path]
  local dir="$TP/$1" url=$2 commit=$3 sparse=${4:-}
  if [[ -d "$dir/.git" ]]; then
    echo "[setup] $1 already present ($(git -C "$dir" rev-parse --short HEAD))"; return
  fi
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  if [[ -n "$sparse" ]]; then
    git -C "$dir" sparse-checkout set "$sparse"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
  echo "[setup] $1 @ ${commit:0:7}"
}

fetch unitree_rl_lab "$RL_LAB_URL" "$RL_LAB_COMMIT"
fetch unitree_ros "$ROS_URL" "$ROS_COMMIT" robots/go2_description  # only the Go2 URDF/meshes (Gazebo)

if [[ "${1:-}" != "--no-install" ]]; then
  source "$ROOT/scripts/_env.sh"
  "$PY" -m pip install -e "$TP/unitree_rl_lab/source/unitree_rl_lab"
fi
