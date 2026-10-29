#!/usr/bin/env bash
set -euo pipefail
TRACKER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRACKER_DIR"
if [[ ! -x "$TRACKER_DIR/.venv-slic/bin/python" ]]; then
    echo "The SLIC test environment .venv-slic is not installed." >&2
    exit 1
fi
source "$TRACKER_DIR/.venv-slic/bin/activate"
exec "$TRACKER_DIR/.venv-slic/bin/python" "$TRACKER_DIR/jewel_tracker_5.py" "$@"
