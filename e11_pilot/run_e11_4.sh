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
    exec "$PYTHON" -m unittest e11_pilot/tests/test_e11_4_latent_causal_audit.py -v "$@"
    ;;
  seen-gpu0)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e11_pilot.e11_4_latent_causal_audit --mode eval-split --split seen_val "$@"
    ;;
  unseen-gpu1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU1_UUID}"
    exec "$PYTHON" -u -m e11_pilot.e11_4_latent_causal_audit --mode eval-split --split unseen_chain "$@"
    ;;
  aggregate)
    exec "$PYTHON" -u -m e11_pilot.e11_4_latent_causal_audit --mode aggregate "$@"
    ;;
  run-dual)
    # Run seen_val on Physical GPU 0 and unseen_chain on Physical GPU 1 in parallel, then aggregate
    CUDA_VISIBLE_DEVICES="$GPU0_UUID" "$PYTHON" -u -m e11_pilot.e11_4_latent_causal_audit --mode eval-split --split seen_val "$@" &
    pid0=$!
    CUDA_VISIBLE_DEVICES="$GPU1_UUID" "$PYTHON" -u -m e11_pilot.e11_4_latent_causal_audit --mode eval-split --split unseen_chain "$@" &
    pid1=$!
    wait "$pid0"
    wait "$pid1"
    exec "$PYTHON" -u -m e11_pilot.e11_4_latent_causal_audit --mode aggregate
    ;;
  *)
    echo "Usage: bash e11_pilot/run_e11_4.sh {test|seen-gpu0|unseen-gpu1|aggregate|run-dual} [args...]" >&2
    exit 2
    ;;
esac
