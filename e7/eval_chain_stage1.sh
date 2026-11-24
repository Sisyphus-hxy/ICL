#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E7_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

PROMPT_MODE="${1:?Usage: bash e7/eval_chain_stage1.sh <default|shifted|oracle_gt> <stage1_ckpt> <output_dir> [extra eval args...]}"
STAGE1_CKPT="${2:?Provide Stage 1 checkpoint path}"
OUT_DIR="${3:?Provide output directory}"
shift 3

POLICY_GPU_UUID="${S1_EVAL_POLICY_GPU:-GPU-e0914423-1b44-589b-fdb3-0360e8f79b43}"
EGL_ID="${S1_EVAL_EGL_DEVICE:-1}"
WORKERS="${S1_EVAL_WORKERS:-10}"
MAX_BATCH="${S1_EVAL_MAX_BATCH:-20}"

SOCK="/tmp/stage1_diag_${PROMPT_MODE}_$$.sock"
READY="$ROOT/e7/outputs/.tmp/stage1_diag_${PROMPT_MODE}_$$.ready"
LIBERO_CFG_DIR="$ROOT/e7/outputs/.tmp/stage1_libero_cfg_${PROMPT_MODE}_$$"
mkdir -p "$ROOT/e7/outputs/.tmp"
export TMPDIR="${TMPDIR:-$ROOT/e7/outputs/.tmp}"
rm -f "$SOCK" "$READY"

cd "$ROOT"
echo "[Stage1 ChainEval] prompt=$PROMPT_MODE ckpt=$STAGE1_CKPT out=$OUT_DIR policy_gpu=$POLICY_GPU_UUID egl=$EGL_ID workers=$WORKERS extra=$*"

(
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU_UUID"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PYTHON" -u -m e7.policy_server_stage1 \
        --prompt-mode "$PROMPT_MODE" \
        --stage1-ckpt "$STAGE1_CKPT" \
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

for _ in $(seq 1 1200); do
    if [[ -f "$READY" ]]; then
        break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "[ERROR] Stage 1 policy server exited before becoming ready!" >&2
        wait "$SRV_PID"
        exit 1
    fi
    sleep 0.5
done
if [[ ! -f "$READY" ]]; then
    echo "[ERROR] Stage 1 policy server timed out waiting for ready file $READY" >&2
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
        --mode "stage1_teacher" \
        --checkpoint "$STAGE1_CKPT" \
        --workers "$WORKERS" \
        --egl-device "$EGL_ID" \
        "$@"
else
    "$PYTHON" -u -m e6.eval_libero_chain_demoswap \
        --socket "$SOCK" \
        --output "$OUT_DIR" \
        --mode "stage1_teacher" \
        --prompt-mode "$PROMPT_MODE" \
        --delta-scale 0.0 \
        --checkpoint "$STAGE1_CKPT" \
        --workers "$WORKERS" \
        --egl-device "$EGL_ID" \
        "$@"
fi
