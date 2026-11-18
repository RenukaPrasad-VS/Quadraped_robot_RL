#!/usr/bin/env bash
# Usage: ./run.sh train [args]   |   ./run.sh play [args]   |   ./run.sh tensorboard
set -euo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM_DIR="${ISAACSIM_DIR:-$HOME/issacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
PY="$ISAACSIM_DIR/python.sh"
[[ -x "$PY" ]] || { echo "Isaac Sim not found at $ISAACSIM_DIR (set ISAACSIM_DIR)" >&2; exit 2; }
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PKG_ROOT:${PYTHONPATH:-}"
cd "$PKG_ROOT"

cmd="${1:-}"; shift || true
case "$cmd" in
  train)
    # with --headless: Ctrl+C stops + saves, then the checkpoint opens in Isaac Sim (skip with --no-play)
    show=0; targs=(); task=(); prev=""
    for a in "$@"; do
      case "$a" in
        --no-play) show=-1; continue ;;
        --headless) [[ $show == 0 ]] && show=1 ;;
      esac
      [[ "$prev" == "--task" ]] && task=(--task "$a")
      targs+=("$a"); prev="$a"
    done
    rm -f "$PKG_ROOT/logs/.last_checkpoint"
    "$PY" scripts/train_live.py "${targs[@]}" || { echo "training failed (exit $?)"; exit 1; }
    if [[ $show == 1 && -f "$PKG_ROOT/logs/.last_checkpoint" ]]; then
      ckpt="$(cat "$PKG_ROOT/logs/.last_checkpoint")"
      echo "[run.sh] opening $ckpt in Isaac Sim"
      exec "$PY" scripts/play.py "${task[@]}" --checkpoint "$ckpt"
    fi ;;
  play)        exec "$PY" scripts/play.py "$@" ;;
  tensorboard) exec "$PY" -m tensorboard.main --logdir "$PKG_ROOT/logs" "$@" ;;
  *) echo "usage: $0 {train|play|tensorboard} [args]  (add --help for options)"; exit 1 ;;
esac
