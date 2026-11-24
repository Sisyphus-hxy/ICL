#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E7_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

MODE="${1:?Usage: bash e7/eval_chain.sh <e7|e7_no_delta|e7_adjacent|e7_adjacent_no_delta> <default|shifted|oracle_gt|shuffle> <e7_ckpt> <output_dir> [extra eval args...]}"
PROMPT_MODE="${2:?Provide prompt_mode: default|shifted|oracle_gt|shuffle}"
CKPT="${3:?Provide E7 checkpoint path}"
OUT_DIR="${4:?Provide output directory}"
shift 4

# Default: physical GPU 0 (A6000) for policy server, physical GPU 1 (EGL 1) for MuJoCo
POLICY_GPU_UUID="${E7_EVAL_POLICY_GPU:-GPU-e0914423-1b44-589b-fdb3-0360e8f79b43}"
EGL_ID="${E7_EVAL_EGL_DEVICE:-1}"
WORKERS="${E7_EVAL_WORKERS:-10}"
MAX_BATCH="${E7_EVAL_MAX_BATCH:-20}"
DELTA_SCALE="${E7_DELTA_SCALE:-1.0}"

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
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

SOCK="/tmp/e7_${MODE}_$$.sock"
READY="$ROOT/e7/outputs/.tmp/e7_chain_${MODE}_${PROMPT_MODE}_$$.ready"
LIBERO_CFG_DIR="$ROOT/e7/outputs/.tmp/e7_libero_cfg_$$"
mkdir -p "$ROOT/e7/outputs/.tmp"
export TMPDIR="${TMPDIR:-$ROOT/e7/outputs/.tmp}"
rm -f "$SOCK" "$READY"

cd "$ROOT"
echo "[E7 ChainEval] mode=$MODE prompt=$PROMPT_MODE delta_scale=$DELTA_SCALE ckpt=$CKPT out=$OUT_DIR policy_gpu=$POLICY_GPU_UUID egl=$EGL_ID workers=$WORKERS extra=${EXTRA_ARGS[*]:-}"

(
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU_UUID"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PYTHON" -u -m e7.policy_server \
        --mode "$MODE" \
        --prompt-mode "$PROMPT_MODE" \
        --delta-scale "$DELTA_SCALE" \
        --checkpoint "$CKPT" \
        --socket "$SOCK" \
        --ready-file "$READY" \
        --stats-out "$OUT_DIR/reader_online_stats.json" \
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
        echo "[ERROR] E7 policy server exited before becoming ready!" >&2
        wait "$SRV_PID"
        exit 1
    fi
    sleep 0.5
done
if [[ ! -f "$READY" ]]; then
    echo "[ERROR] E7 policy server timed out waiting for ready file $READY" >&2
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
