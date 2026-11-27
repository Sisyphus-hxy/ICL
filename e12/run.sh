#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E12_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 2 (primary empty RTX 6000D, 85 GB), plus GPU 4 & GPU 5 by UUID per AGENTS.md
GPU2_UUID="GPU-925a799e-d7db-61b5-8d1e-2e97919fba76"
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
    exec "$PYTHON" -m unittest discover -s e12/tests -v "$@"
    ;;
  analyze-e9)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e12.analyze_e9 "$@"
    ;;
  eval)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e12.eval_suite "$@"
    ;;
  compare)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e12.compare_all "$@"
    ;;
  train-gpu2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU2_UUID}"
    exec "$PYTHON" -u -m e12.train_e12 "$@"
    ;;
  train-gpu4)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID}"
    exec "$PYTHON" -u -m e12.train_e12 "$@"
    ;;
  train-gpu5)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU5_UUID}"
    exec "$PYTHON" -u -m e12.train_e12 "$@"
    ;;
  train-ddp)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU4_UUID,$GPU5_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    port="${E12_MASTER_PORT:-29712}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e12.train_e12 "$@"
    ;;
  *)
    echo "Usage: bash e12/run.sh {test|analyze-e9|eval|train-gpu4|train-gpu5|train-ddp} [args...]" >&2
    exit 2
    ;;
esac
