#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E14_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 4 & GPU 5 (2x idle RTX 6000D, 85 GB each) by UUID per user request & AGENTS.md
GPU4_UUID="GPU-129d1a29-e94e-1067-e547-41ef6ea6df69"
GPU5_UUID="GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec "$PYTHON" -m unittest discover -s e14/tests -v "$@"
    ;;
  exp1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -u -m e14.exp1_cross_transfer "$@"
    ;;
  exp2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU5_UUID}"
    exec "$PYTHON" -u -m e14.exp2_mechanism_diagnosis "$@"
    ;;
  exp3)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -u -m e14.exp3_complementary_repr "$@"
    ;;
  report)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -u -m e14.report_e14 "$@"
    ;;
  *)
    echo "Usage: bash e14/run.sh {test|exp1|exp2|exp3|report} [args...]" >&2
    exit 2
    ;;
esac
