#!/usr/bin/env bash
#SBATCH --job-name=e5-chain-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=e5/outputs/logs/chain-eval-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

MODE="${1:?Usage: sbatch e5/eval_chain.sbatch <e5|bpp_rgb_only|bpp_full|bpp_official> <checkpoint_or_none> <output_dir> [extra eval_libero_chain args...]}"
CKPT="${2:?Provide checkpoint path (or 'none' for bpp_official)}"
OUT_DIR="${3:?Provide output directory}"
shift 3

SOCK="/tmp/e5_chain_srv_${SLURM_JOB_ID:-$$}.sock"
READY="/tmp/e5_chain_srv_${SLURM_JOB_ID:-$$}.ready"
rm -f "$SOCK" "$READY"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} mode=$MODE ckpt=$CKPT out=$OUT_DIR extra=$*"

# 1. Start batched GPU policy server in env/train/e4_chain
SRV_CMD=(env/train/e4_chain/bin/python -u -m e5.policy_server --mode "$MODE" --socket "$SOCK" --ready-file "$READY" --max-batch 20 --coalesce-ms 4.0)
if [[ "$CKPT" != "none" ]]; then
    SRV_CMD+=(--checkpoint "$CKPT")
fi

(
    source env/train/common.sh
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "${SRV_CMD[@]}"
) &
SRV_PID=$!

cleanup() {
    if kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    rm -f "$SOCK" "$READY"
}
trap cleanup EXIT

# Wait up to 180s for server startup + 10x10 prompt pre-encoding
for _ in $(seq 1 360); do
    if [[ -f "$READY" ]]; then
        break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "[ERROR] Policy server exited before becoming ready!" >&2
        wait "$SRV_PID"
        exit 1
    fi
    sleep 0.5
done
if [[ ! -f "$READY" ]]; then
    echo "[ERROR] Policy server timed out waiting for ready file $READY" >&2
    exit 1
fi

# 2. Resolve EGL device matching CUDA ordinal 0 in env/libero/.conda and run parallel rollouts
source env/libero/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EGL_ID=$(python -c '
import ctypes
try:
    from mujoco.egl import egl_ext as egl
    q = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(egl.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    for idx, h in enumerate(egl.eglQueryDevicesEXT()):
        ord_v = ctypes.c_ssize_t()
        if q(h, 0x323A, ctypes.byref(ord_v)) and ord_v.value == 0:
            print(idx)
            raise SystemExit(0)
except Exception:
    pass
print("")
')
echo "Resolved MUJOCO_EGL_DEVICE_ID=${EGL_ID:-default}"

EVAL_ARGS=(--socket "$SOCK" --output "$OUT_DIR" --mode "$MODE" --checkpoint "$CKPT" --workers 10)
if [[ -n "$EGL_ID" ]]; then
    EVAL_ARGS+=(--egl-device "$EGL_ID")
fi
python -u e5/eval_libero_chain.py "${EVAL_ARGS[@]}" "$@"
