#!/usr/bin/env bash
set -euo pipefail
TRACKER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRACKER_DIR"
# Match inherited shell/subprocess settings to the interpreter used below.
source "$TRACKER_DIR/.venv/bin/activate"
exec "$TRACKER_DIR/.venv/bin/python" "$TRACKER_DIR/jewel_tracker_5.py" "$@"
