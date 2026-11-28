#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E18_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 0 & GPU 1 (2x idle NVIDIA RTX A6000, 48 GB) strictly selected by UUID per AGENTS.md
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
    exec "$PYTHON" -m unittest discover -s e18/tests -v "$@"
    ;;
  cache-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.cache_generalization "$@"
    ;;
  cache-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e18.cache_generalization "$@"
    ;;
  geo-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.geometry_relations "$@"
    ;;
  geo-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e18.geometry_relations "$@"
    ;;
  pretrain-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.pretrain_e18 "$@"
    ;;
  pretrain-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e18.pretrain_e18 "$@"
    ;;
  decoder-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e17.train_unified_decoder "$@"
    ;;
  decoder-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e17.train_unified_decoder "$@"
    ;;
  eval-e15-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e15.eval_e15 "$@"
    ;;
  eval-e15-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e15.eval_e15 "$@"
    ;;
  rlb3d-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.eval_rlbench_3d_geo "$@"
    ;;
  rlb3d-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e18.eval_rlbench_3d_geo "$@"
    ;;
  gen-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.eval_generalization "$@"
    ;;
  gen-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e18.eval_generalization "$@"
    ;;
  suite-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.eval_e18_all "$@"
    ;;
  suite-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e18.eval_e18_all "$@"
    ;;
  report)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e18.report_e18 "$@"
    ;;
  *)
    echo "Usage: bash e18/run.sh {test|cache-gpu0|cache-gpu1|geo-gpu0|pretrain-gpu0|pretrain-gpu1|decoder-gpu0|decoder-gpu1|eval-e15-gpu0|rlb3d-gpu0|gen-gpu0|suite-gpu0|report} [args...]" >&2
    exit 2
    ;;
esac
