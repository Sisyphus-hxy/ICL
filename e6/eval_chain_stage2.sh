#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${E6_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

MODE="${1:?Usage: bash e6/eval_chain_stage2.sh <e6_stage2|e6_stage2_no_delta> <stage2_ckpt> <output_dir> [extra eval_libero_chain args...]}"
CKPT="${2:?Provide Stage 2 checkpoint path}"
OUT_DIR="${3:?Provide output directory}"
shift 3

# Default: policy server on physical GPU 0 (A6000), MuJoCo EGL on physical GPU 1 (A6000)
# (Per AGENTS.md: physical GPU 0 -> EGL 0, physical GPU 1 -> EGL 1)
POLICY_GPU_UUID="${E6_EVAL_POLICY_GPU:-GPU-e0914423-1b44-589b-fdb3-0360e8f79b43}"
EGL_ID="${E6_EVAL_EGL_DEVICE:-1}"
WORKERS="${E6_EVAL_WORKERS:-10}"
DELTA_SCALE="${E6_DELTA_SCALE:-1.0}"

SOCK="/tmp/e6_stage2_chain_${MODE}_$$.sock"
READY="/tmp/e6_stage2_chain_${MODE}_$$.ready"
LIBERO_CFG_DIR="/tmp/e6_libero_cfg_$$"
rm -f "$SOCK" "$READY"

cd "$ROOT"
echo "[E6 ChainEval] mode=$MODE delta_scale=$DELTA_SCALE ckpt=$CKPT out=$OUT_DIR policy_gpu=$POLICY_GPU_UUID egl=$EGL_ID workers=$WORKERS extra=$*"

(
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU_UUID"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PYTHON" -u -m e6.policy_server_stage2 \
        --mode "$MODE" \
        --delta-scale "$DELTA_SCALE" \
        --checkpoint "$CKPT" \
        --socket "$SOCK" \
        --ready-file "$READY" \
        --max-batch 20 \
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

for _ in $(seq 1 360); do
    if [[ -f "$READY" ]]; then
        break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "[ERROR] E6 Stage 2 policy server exited before becoming ready!" >&2
        wait "$SRV_PID"
        exit 1
    fi
    sleep 0.5
done
if [[ ! -f "$READY" ]]; then
    echo "[ERROR] E6 Stage 2 policy server timed out waiting for ready file $READY" >&2
    exit 1
fi

# Prepare writable LIBERO config.yaml in /tmp so MuJoCo workers never touch ~/.libero
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

# Per AGENTS.md: MuJoCo subprocesses set CUDA_VISIBLE_DEVICES= and MUJOCO_EGL_DEVICE_ID=<egl>
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="$EGL_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -u e5/eval_libero_chain.py \
    --socket "$SOCK" \
    --output "$OUT_DIR" \
    --mode "$MODE" \
    --checkpoint "$CKPT" \
    --workers "$WORKERS" \
    --egl-device "$EGL_ID" \
    "$@"
