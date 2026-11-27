#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E11_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 0 & GPU 1 (2x NVIDIA RTX A6000, 48 GB) per AGENTS.md & user specification
GPU0_UUID="GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID="GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TORCH_KERNEL_CACHE_PATH="${TORCH_KERNEL_CACHE_PATH:-/tmp/torch_kernels}"
mkdir -p "$TORCH_KERNEL_CACHE_PATH" "$MPLCONFIGDIR"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -m unittest e11_pilot/tests/test_e11_3_decoder_aware.py -v "$@"
    ;;
  prep-cache)
    exec "$PYTHON" -u -m e11_pilot.e11_3_decoder_aware --mode prep-cache "$@"
    ;;
  train-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e11_pilot.e11_3_decoder_aware --mode train "$@"
    ;;
  train-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e11_pilot.e11_3_decoder_aware --mode train "$@"
    ;;
  eval-report)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e11_pilot.e11_3_decoder_aware --mode eval-report "$@"
    ;;
  *)
    echo "Usage: bash e11_pilot/run_e11_3.sh {test|prep-cache|train-gpu0|train-gpu1|eval-report} [args...]" >&2
    exit 2
    ;;
esac
