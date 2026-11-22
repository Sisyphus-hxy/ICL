#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/train/common.sh"

# Default to 2 idle Blackwell GPUs unless CUDA_VISIBLE_DEVICES is explicitly set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ -z "${NCCL_RAS_ADDR:-}" ]]; then
  ras_port="$("$E4_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  export NCCL_RAS_ADDR="127.0.0.1:${ras_port}"
fi

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec "$E4_PYTHON" -m unittest discover -s e5/tests -v "$@"
    ;;
  cache)
    exec "$E4_PYTHON" -u -m e5.cache "$@"
    ;;
  verify-cache)
    exec "$E4_PYTHON" -u -m e5.cache --verify-only "$@"
    ;;
  overfit)
    CUDA_VISIBLE_DEVICES="${E5_SINGLE_GPU:-1}" exec "$E4_PYTHON" -u -m e5.train --overfit "$@"
    ;;
  train-e5)
    nproc="${E5_NPROC:-2}"
    exec "$E4_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$nproc" -m e5.train --mode e5 "$@"
    ;;
  train-baseline)
    nproc="${E5_NPROC:-2}"
    exec "$E4_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$nproc" -m e5.train --mode bpp_rgb_only "$@"
    ;;
  cache-proprio)
    exec "$E4_PYTHON" -u -m e5.cache_proprio "$@"
    ;;
  train-bpp-full)
    nproc="${E5_NPROC:-2}"
    exec "$E4_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$nproc" -m e5.bpp_full "$@"
    ;;
  *)
    echo "Usage: bash e5/run.sh {test|cache|verify-cache|cache-proprio|overfit|train-e5|train-baseline|train-bpp-full} [args...]" >&2
    exit 2
    ;;
esac
