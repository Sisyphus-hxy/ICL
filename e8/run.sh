#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E8_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Default GPUs by UUID per AGENTS.md (2x RTX A6000: physical GPU 0 & GPU 1)
GPU0_UUID="GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID="GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "$ROOT/e8/outputs/source_conditioned/.tmp"
export TMPDIR="${TMPDIR:-$ROOT/e8/outputs/source_conditioned/.tmp}"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec "$PYTHON" -m unittest discover -s e8/tests -v "$@"
    ;;
  smoke)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E8_NPROC:-2}"
    port="${E8_MASTER_PORT:-29811}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e8.train --smoke "$@"
    ;;
  train)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E8_NPROC:-2}"
    port="${E8_MASTER_PORT:-29812}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e8.train "$@"
    ;;
  eval-offline)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e8.train "$@"
    ;;
  eval-chain)
    exec bash "$ROOT/e8/eval_chain.sh" "$@"
    ;;
  analyze)
    exec "$PYTHON" -u -m e8.analyze_results "$@"
    ;;
  *)
    echo "Usage: bash e8/run.sh {test|smoke|train|eval-offline|eval-chain|analyze} [args...]" >&2
    exit 2
    ;;
esac
