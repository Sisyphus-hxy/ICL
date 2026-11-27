#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E9_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 0 & GPU 1 (2x NVIDIA RTX A6000, 48 GB each) by UUID per AGENTS.md
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
    exec "$PYTHON" -m unittest discover -s e9/tests -v "$@"
    ;;
  audit-cache)
    exec "$PYTHON" -u -m e9.cache_dense_clip --audit-only "$@"
    ;;
  build-cache)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    exec "$PYTHON" -u -m e9.cache_dense_clip "$@"
    ;;
  smoke)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e9.train_stage1_local --smoke "$@"
    ;;
  train)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E9_NPROC:-2}"
    port="${E9_MASTER_PORT:-29691}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e9.train_stage1_local "$@"
    ;;
  train-1gpu)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e9.train_stage1_local "$@"
    ;;
  eval-d1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e9.eval_d1_local "$@"
    ;;
  cache-stage2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e9.cache_stage2_local "$@"
    ;;
  smoke-stage2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e9.train_stage2_direct --smoke "$@"
    ;;
  train-stage2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E9_NPROC:-2}"
    port="${E9_STAGE2_MASTER_PORT:-29692}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e9.train_stage2_direct "$@"
    ;;
  eval-stage2-chain)
    exec bash "$ROOT/e9/eval_chain_stage2.sh" "$@"
    ;;
  diagnose-stage2-d1-d4)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e9.diagnose_stage2_d1_d4 "$@"
    ;;
  analyze-stage2)
    exec "$PYTHON" -u -m e9.analyze_stage2_direct "$@"
    ;;
  *)
    echo "Usage: bash e9/run.sh {test|audit-cache|build-cache|smoke|train|eval-d1|cache-stage2|smoke-stage2|train-stage2|eval-stage2-chain|diagnose-stage2-d1-d4|analyze-stage2} [args...]" >&2
    exit 2
    ;;
esac
