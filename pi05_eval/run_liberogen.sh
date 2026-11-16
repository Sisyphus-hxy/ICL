#!/bin/bash
# LIBERO-Gen rollout with pi0.5 -> BPP prompt demo.hdf5 on success.
# Usage: bash pi05_eval/run_liberogen.sh <PHYSICAL gpu index> [port] [benchmark] [tasks_csv] [queries] [extra...]
# NOTE: EGL device is selected by PHYSICAL index (MUJOCO_EGL_DEVICE_ID);
# this client uses no CUDA.
cd "$(dirname "$0")/.."
source pi05_eval/env.sh

GPU="${1:?PHYSICAL gpu index (nvidia-smi index) required}"
PORT="${2:-8000}"
BENCH="${3:-liberogen_spatial_combination}"
TASKS="${4:-}"
QUERIES="${5:-0-49}"

unset CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH="$PI05_CLIENT_PYTHONPATH"
OUT=bpp/experiments/pi05_demos/outputs
mkdir -p "$OUT" logs

ARGS=(--mode liberogen --host 127.0.0.1 --port "$PORT"
      --benchmark "$BENCH" --queries "$QUERIES" --output "$OUT")
if [ -n "$TASKS" ]; then ARGS+=(--tasks "$TASKS"); fi
shift 5 2>/dev/null || shift $#
exec "$LIBERO_PY" pi05_eval/rollout.py "${ARGS[@]}" "$@"
