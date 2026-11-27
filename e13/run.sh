#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E13_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 2 & GPU 5 (2x empty RTX 6000D, 85 GB each) by UUID per AGENTS.md
GPU2_UUID="GPU-925a799e-d7db-61b5-8d1e-2e97919fba76"
GPU5_UUID="GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec "$PYTHON" -m unittest discover -s e13/tests -v "$@"
    ;;
  exp1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e13.exp1_training_free "$@"
    ;;
  exp2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU5_UUID}"
    exec "$PYTHON" -u -m e13.exp2_action_baselines "$@"
    ;;
  exp3)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e13.exp3_factor_composition "$@"
    ;;
  exp4)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU5_UUID}"
    exec "$PYTHON" -u -m e13.exp4_minimal_icl "$@"
    ;;
  exp5-gpu2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e13.exp5_e12_bc "$@"
    ;;
  exp5-gpu5)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU5_UUID}"
    exec "$PYTHON" -u -m e13.exp5_e12_bc "$@"
    ;;
  report)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e13.report_e13 "$@"
    ;;
  *)
    echo "Usage: bash e13/run.sh {test|exp1|exp2|exp3|exp4|exp5-gpu2|exp5-gpu5|report} [args...]" >&2
    exit 2
    ;;
esac
