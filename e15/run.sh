#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E15_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Strictly physical GPU 0 & GPU 1 (2x RTX A6000, 48 GB each) by UUID per user request & AGENTS.md
GPU0_UUID="GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID="GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec "$PYTHON" -m unittest discover -s e15/tests -v "$@"
    ;;
  exp1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e15.exp1_three_component "$@"
    ;;
  noise-bank)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e15.residual_noise "$@"
    ;;
  train-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e15.train_e15 "$@"
    ;;
  train-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e15.train_e15 "$@"
    ;;
  eval-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e15.eval_e15 "$@"
    ;;
  eval-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e15.eval_e15 "$@"
    ;;
  report)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e15.report_e15 "$@"
    ;;
  *)
    echo "Usage: bash e15/run.sh {test|exp1|noise-bank|train-gpu0|train-gpu1|eval-gpu0|eval-gpu1|report} [args...]" >&2
    exit 2
    ;;
esac
