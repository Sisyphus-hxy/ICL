#!/bin/bash
# ContextFlow unseen-test-set rollout with pi0.5 (no demo input, prompt only).
# Usage: bash pi05_eval/run_contextflow.sh <PHYSICAL gpu index> [port] [suite] [extra args...]
# NOTE: this client only does EGL rendering — no CUDA. mujoco/robosuite pick
# the EGL device by PHYSICAL index via MUJOCO_EGL_DEVICE_ID; unlike the policy
# server this script does NOT take a CUDA_VISIBLE_DEVICES value.
cd "$(dirname "$0")/.."
source pi05_eval/env.sh

GPU="${1:?PHYSICAL gpu index (nvidia-smi index) required}"
PORT="${2:-8000}"
SUITE="${3:-all}"
shift 3 2>/dev/null || shift $#

unset CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH="$PI05_CLIENT_PYTHONPATH"
OUT=ContextFlow/data/pi05_demos/rollouts
mkdir -p "$OUT" logs
exec "$LIBERO_PY" pi05_eval/rollout.py \
    --mode contextflow \
    --host 127.0.0.1 --port "$PORT" \
    --task-suite-name "$SUITE" \
    --num-trials-per-task 50 \
    --output "$OUT" \
    "$@"
