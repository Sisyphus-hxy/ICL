#!/bin/bash
# ContextFlow in-context eval client (unseen tasks), py3.8 LIBERO env.
# Usage: bash pi05_eval/cf_eval.sh <PHYSICAL gpu index> [port] [video_dir] [extra args...]
#   EGL device follows the PHYSICAL nvidia-smi index (MUJOCO_EGL_DEVICE_ID).
cd "$(dirname "$0")/.."
source pi05_eval/env.sh

GPU="${1:?PHYSICAL gpu index (nvidia-smi index) required}"
PORT="${2:-8010}"
VIDEO_DIR="${3:-ContextFlow/data/pi05_demos/ic_eval_videos}"
shift 3 2>/dev/null || shift $#

unset CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH="$PI05_CLIENT_PYTHONPATH"
mkdir -p "$VIDEO_DIR" logs
exec "$LIBERO_PY" ContextFlow/examples/libero/main_incontext.py \
    --host 127.0.0.1 --port "$PORT" \
    --video-out-path "$VIDEO_DIR" \
    "$@"
