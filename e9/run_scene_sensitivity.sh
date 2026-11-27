#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E9_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"
OUT_DIR="${1:-$ROOT/e9/outputs/scene_sensitivity}"

# Physical GPU 2 (RTX 6000D) per AGENTS.md:
#   CUDA_VISIBLE_DEVICES=GPU-925a799e-d7db-61b5-8d1e-2e97919fba76
#   MUJOCO_EGL_DEVICE_ID=2 (with CUDA_VISIBLE_DEVICES="" in MuJoCo process)
GPU2_UUID="GPU-925a799e-d7db-61b5-8d1e-2e97919fba76"
EGL_ID="2"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "$ROOT"

echo "=== [Stage 1/2] MuJoCo EGL Simulation on GPU 2 (EGL=$EGL_ID) ==="
CUDA_VISIBLE_DEVICES="" MUJOCO_GL="egl" MUJOCO_EGL_DEVICE_ID="$EGL_ID" \
  "$PYTHON" -u -m e9.scene_sensitivity simulate --out-dir "$OUT_DIR"

echo "=== [Stage 2/2] CLIP + E9 Latent Extraction & Analysis on GPU 2 ($GPU2_UUID) ==="
CUDA_VISIBLE_DEVICES="$GPU2_UUID" \
  "$PYTHON" -u -m e9.scene_sensitivity analyze --out-dir "$OUT_DIR"
