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

# --ros2: use the ROS 2 Humble libraries bundled with Isaac Sim (built for its Python 3.11).
# A system ROS 2 sourced in this terminal is built for Python 3.10, so its Python paths are dropped.
if [[ " $* " == *" --ros2 "* ]]; then
  ROS_BRIDGE="$ISAACSIM_DIR/exts/isaacsim.ros2.bridge/humble"
  PYTHONPATH="$(tr ':' '\n' <<<"$PYTHONPATH" | grep -v '^/opt/ros/' | paste -sd: -)"
  export PYTHONPATH="$ROS_BRIDGE/rclpy:$PYTHONPATH"
  export LD_LIBRARY_PATH="$ROS_BRIDGE/lib:${LD_LIBRARY_PATH:-}"
  export ROS_DISTRO=humble
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
fi
