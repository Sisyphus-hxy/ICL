#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E10_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

PROMPT_MODE="${1:?Usage: bash e10/eval_chain.sh <default|shifted|oracle_gt|shuffle> <e10_ckpt> <output_dir> [extra args...]}"
CKPT="${2:?Provide E10 checkpoint path}"
OUT_DIR="${3:?Provide output directory}"
shift 3

# Policy GPU UUID & MuJoCo EGL device (per AGENTS.md: GPU 0 -> EGL 0, GPU 1 -> EGL 1)
POLICY_GPU_UUID="${E10_EVAL_POLICY_GPU:-GPU-e0914423-1b44-589b-fdb3-0360e8f79b43}"
EGL_ID="${E10_EVAL_EGL_DEVICE:-0}"
WORKERS="${E10_EVAL_WORKERS:-10}"
MAX_BATCH="${E10_EVAL_MAX_BATCH:-20}"
MODE="${E10_EVAL_MODE:-e10_lowrank}"

TMP_ROOT="${E10_TMP_ROOT:-$ROOT/e10/outputs/.tmp}"
mkdir -p "$TMP_ROOT"
SOCK="/tmp/e10_${MODE}_${PROMPT_MODE}_$$.sock"
READY="$TMP_ROOT/e10_chain_${MODE}_${PROMPT_MODE}_$$.ready"
LIBERO_CFG_DIR="$TMP_ROOT/e10_libero_cfg_$$"
export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
rm -f "$SOCK" "$READY"

cd "$ROOT"
echo "[E10 ChainEval] mode=$MODE prompt=$PROMPT_MODE ckpt=$CKPT out=$OUT_DIR policy_gpu=$POLICY_GPU_UUID egl=$EGL_ID workers=$WORKERS"

(
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU_UUID"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PYTHON" -u -m e10.policy_server \
        --mode "$MODE" \
        --prompt-mode "$PROMPT_MODE" \
        --checkpoint "$CKPT" \
        --socket "$SOCK" \
        --ready-file "$READY" \
        --max-batch "$MAX_BATCH" \
        --coalesce-ms 4.0
) &
SRV_PID=$!

cleanup() {
    if kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    rm -rf "$SOCK" "$READY" "$LIBERO_CFG_DIR"
}
trap cleanup EXIT

for _ in $(seq 1 1800); do
    if [[ -f "$READY" ]]; then
        break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "[ERROR] E10 policy server exited before becoming ready!" >&2
        wait "$SRV_PID"
        exit 1
    fi
    sleep 0.5
done
if [[ ! -f "$READY" ]]; then
    echo "[ERROR] E10 policy server timed out waiting for ready file $READY" >&2
    exit 1
fi

LIBERO_ROOT="$ROOT/bpp/src/deps/LIBERO/libero/libero"
export LIBERO_CONFIG_PATH="$LIBERO_CFG_DIR"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $ROOT/bpp/datasets/libero
assets: $LIBERO_ROOT/assets
EOF

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="$EGL_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ "$PROMPT_MODE" == "default" ]]; then
    "$PYTHON" -u e5/eval_libero_chain.py \
        --socket "$SOCK" \
        --output "$OUT_DIR" \
        --mode "$MODE" \
        --checkpoint "$CKPT" \
        --workers "$WORKERS" \
        --egl-device "$EGL_ID" \
        "$@"
else
    "$PYTHON" -u -m e6.eval_libero_chain_demoswap \
        --socket "$SOCK" \
        --output "$OUT_DIR" \
        --mode "$MODE" \
        --prompt-mode "$PROMPT_MODE" \
        --delta-scale 1.0 \
        --checkpoint "$CKPT" \
        --workers "$WORKERS" \
        --egl-device "$EGL_ID" \
        "$@"
fi
