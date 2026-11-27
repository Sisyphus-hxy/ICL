#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E9_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

MODE="${1:?Usage: bash e9/eval_chain_stage2.sh <e9_direct|e9_no_transfer> <default|shifted|oracle_gt|shuffle> <e9_stage2_ckpt> <output_dir> [extra eval args...]}"
PROMPT_MODE="${2:?Provide prompt_mode: default|shifted|oracle_gt|shuffle}"
CKPT="${3:?Provide E9 Stage 2 checkpoint path}"
OUT_DIR="${4:?Provide output directory}"
shift 4

# Default: physical GPU 0 (A6000) for policy server, physical GPU 1 (EGL 1) for MuJoCo
POLICY_GPU_UUID="${E9_EVAL_POLICY_GPU:-GPU-e0914423-1b44-589b-fdb3-0360e8f79b43}"
EGL_ID="${E9_EVAL_EGL_DEVICE:-1}"
WORKERS="${E9_EVAL_WORKERS:-10}"
MAX_BATCH="${E9_EVAL_MAX_BATCH:-20}"
DELTA_SCALE="${E9_DELTA_SCALE:-1.0}"
READER_TEMP="${E9_READER_TEMP:-1.0}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --delta-scale)
            DELTA_SCALE="${2:?Missing value for --delta-scale}"
            shift 2
            ;;
        --delta-scale=*)
            DELTA_SCALE="${1#*=}"
            shift
            ;;
        --reader-temperature)
            READER_TEMP="${2:?Missing value for --reader-temperature}"
            shift 2
            ;;
        --reader-temperature=*)
            READER_TEMP="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

TMP_ROOT="${E9_TMP_ROOT:-$ROOT/e9/outputs/stage2_direct/.tmp}"
mkdir -p "$TMP_ROOT"
SOCK="/tmp/e9_${MODE}_$$.sock"
READY="$TMP_ROOT/e9_chain_${MODE}_${PROMPT_MODE}_$$.ready"
LIBERO_CFG_DIR="$TMP_ROOT/e9_libero_cfg_$$"
export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
rm -f "$SOCK" "$READY"

cd "$ROOT"
echo "[E9 Stage2 ChainEval] mode=$MODE prompt=$PROMPT_MODE delta_scale=$DELTA_SCALE reader_tau=$READER_TEMP ckpt=$CKPT out=$OUT_DIR policy_gpu=$POLICY_GPU_UUID egl=$EGL_ID workers=$WORKERS extra=${EXTRA_ARGS[*]:-}"

(
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU_UUID"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PYTHON" -u -m e9.policy_server_stage2 \
        --mode "$MODE" \
        --prompt-mode "$PROMPT_MODE" \
        --delta-scale "$DELTA_SCALE" \
        --reader-temperature "$READER_TEMP" \
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
        echo "[ERROR] E9 Stage 2 policy server exited before becoming ready!" >&2
        wait "$SRV_PID"
        exit 1
    fi
    sleep 0.5
done
if [[ ! -f "$READY" ]]; then
    echo "[ERROR] E9 Stage 2 policy server timed out waiting for ready file $READY" >&2
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
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    "$PYTHON" -u -m e6.eval_libero_chain_demoswap \
        --socket "$SOCK" \
        --output "$OUT_DIR" \
        --mode "$MODE" \
        --prompt-mode "$PROMPT_MODE" \
        --delta-scale "$DELTA_SCALE" \
        --checkpoint "$CKPT" \
        --workers "$WORKERS" \
        --egl-device "$EGL_ID" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
