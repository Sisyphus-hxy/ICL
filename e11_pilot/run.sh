#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E11_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 4 & GPU 5 (2x NVIDIA RTX 6000D, 85 GB, sm_120) per AGENTS.md
GPU4_UUID="GPU-129d1a29-e94e-1067-e547-41ef6ea6df69"
GPU5_UUID="GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_KERNEL_CACHE_PATH="${TORCH_KERNEL_CACHE_PATH:-/tmp/torch_kernels}"
mkdir -p "$TORCH_KERNEL_CACHE_PATH"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -m unittest discover -s e11_pilot/tests -v "$@"
    ;;
  oracle)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -u -m e11_pilot.oracle_opt "$@"
    ;;
  learnability)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -u -m e11_pilot.learnability "$@"
    ;;
  report)
    exec "$PYTHON" -u -m e11_pilot.plot_and_report "$@"
    ;;
  *)
    echo "Usage: bash e11_pilot/run.sh {test|oracle|learnability|report} [args...]" >&2
    exit 2
    ;;
esac
