#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$PROJECT_ROOT"
exec "$PY" "$PROJECT_ROOT/scripts/course.py" "$@"
