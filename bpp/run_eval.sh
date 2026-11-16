#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
python3 "$BPP_ROOT/scripts/configure.py"
exec python3 "$BPP_ROOT/scripts/evaluate.py" "$@"
