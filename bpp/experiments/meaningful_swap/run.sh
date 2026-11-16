#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$experiment_dir/../../scripts/common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${BPP_PYTHON:-$BPP_ROOT/env/bpp/bin/python}" "$experiment_dir/run.py" "$@"
