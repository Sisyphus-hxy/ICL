#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E6_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 0 & GPU 1 (2x NVIDIA RTX A6000, 48 GB each) by UUID per AGENTS.md
GPU0_UUID="GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID="GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$ROOT"
cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec "$PYTHON" -m unittest discover -s e6/tests -v "$@"
    ;;
  smoke)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage1 --smoke "$@"
    ;;
  train-stage1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage1 "$@"
    ;;
  cache-stage2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.cache_stage2 "$@"
    ;;
  smoke-stage2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29611}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2 --smoke "$@"
    ;;
  train-stage2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29612}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2 "$@"
    ;;
  eval-chain-stage2)
    exec bash "$ROOT/e6/eval_chain_stage2.sh" "$@"
    ;;
  smoke-stage2-1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29621}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2_1 --smoke "$@"
    ;;
  train-stage2-1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29622}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2_1 "$@"
    ;;
  train-stage2-2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 --output-dir e6/outputs/stage2_2 "$@"
    ;;
  smoke-stage2-2-a)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_2 \
      --run-name smoke_action_only \
      --lambda-z-schedule constant \
      --lambda-z 0.0 \
      --lambda-anchor 0.1 \
      --smoke "$@"
    ;;
  smoke-stage2-2-c)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_2 \
      --run-name smoke_scheduled_latent \
      --lambda-z-schedule hold_linear \
      --lambda-z-start 0.5 \
      --lambda-z-end 0.1 \
      --lambda-z-hold-frac 0.2 \
      --lambda-anchor 0.1 \
      --smoke "$@"
    ;;
  train-stage2-2-a)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_2 \
      --run-name action_only \
      --lambda-z-schedule constant \
      --lambda-z 0.0 \
      --lambda-anchor 0.1 "$@"
    ;;
  train-stage2-2-c)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_2 \
      --run-name scheduled_latent \
      --lambda-z-schedule hold_linear \
      --lambda-z-start 0.5 \
      --lambda-z-end 0.1 \
      --lambda-z-hold-frac 0.2 \
      --lambda-anchor 0.1 "$@"
    ;;
  smoke-stage2-2-d)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_2 \
      --run-name smoke_latent_only \
      --lambda-diff 0.0 \
      --lambda-z-schedule constant \
      --lambda-z 0.5 \
      --lambda-anchor 0.1 \
      --smoke "$@"
    ;;
  train-stage2-2-d)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29624}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_2 \
      --run-name latent_only \
      --lambda-diff 0.0 \
      --lambda-z-schedule constant \
      --lambda-z 0.5 \
      --lambda-anchor 0.1 "$@"
    ;;
  smoke-stage2-3-f)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_3 \
      --run-name smoke_stable_correction \
      --lambda-diff 1.0 \
      --lambda-z 0.5 \
      --lambda-anchor 0.1 \
      --lambda-id 1.0 \
      --lambda-refine 0.2 \
      --smoke "$@"
    ;;
  train-stage2-3-f)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29631}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_3 \
      --run-name stable_correction \
      --lambda-diff 1.0 \
      --lambda-z 0.5 \
      --lambda-anchor 0.1 \
      --lambda-id 1.0 \
      --lambda-refine 0.2 "$@"
    ;;
  smoke-stage2-3-e)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
    exec "$PYTHON" -u -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_3 \
      --run-name smoke_identity_only \
      --lambda-diff 1.0 \
      --lambda-z 0.5 \
      --lambda-anchor 0.1 \
      --lambda-id 1.0 \
      --lambda-refine 0.0 \
      --smoke "$@"
    ;;
  train-stage2-3-e)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID,$GPU1_UUID}"
    export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    nproc="${E6_NPROC:-2}"
    port="${E6_MASTER_PORT:-29632}"
    exec "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
      --master_addr=127.0.0.1 --master_port="$port" \
      -m e6.train_stage2_1 \
      --output-dir e6/outputs/stage2_3 \
      --run-name identity_only \
      --lambda-diff 1.0 \
      --lambda-z 0.5 \
      --lambda-anchor 0.1 \
      --lambda-id 1.0 \
      --lambda-refine 0.0 "$@"
    ;;
  test-dinov3)
    exec "$PYTHON" -m unittest discover -s e6/tests -p 'test_stage1_dinov3.py' -v "$@"
    ;;
  check-dinov3)
    exec "$PYTHON" -u -m e6.train_stage1_dinov3 --check-only --cpu "$@"
    ;;
  smoke-stage1-dinov3)
    exec "$PYTHON" -u -m e6.train_stage1_dinov3 --smoke "$@"
    ;;
  train-stage1-dinov3)
    exec "$PYTHON" -u -m e6.train_stage1_dinov3 "$@"
    ;;
  eval-stage1-dinov3)
    exec "$PYTHON" -u -m e6.train_stage1_dinov3 --eval-only "$@"
    ;;
  *)
    echo "Usage: bash e6/run.sh {test|smoke|train-stage1|cache-stage2|smoke-stage2|train-stage2|eval-chain-stage2|test-dinov3|check-dinov3|smoke-stage1-dinov3|train-stage1-dinov3|eval-stage1-dinov3} [args...]" >&2
    exit 2
    ;;
esac
