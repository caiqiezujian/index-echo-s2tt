#!/usr/bin/env bash
# Use the container's existing python3 and the checked-out source.
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/.." && pwd)"
CONFIG_PATH="${1:-$REPOSITORY_ROOT/configs/5090_2b.json}"
if (( $# > 0 )); then shift; fi
exec python3 -u "$REPOSITORY_ROOT/run_s2tt.py" serve --config "$CONFIG_PATH" "$@"
