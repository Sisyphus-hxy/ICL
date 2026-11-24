#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="e6/outputs/stage2_1/stage2_1_joint_reader_delta/best.pt"
OUT_ROOT="e6/outputs/delta_scale_sweep"

# Strictly use Physical GPU 0 and Physical GPU 1 (2x RTX A6000)
GPU0_UUID="GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID="GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"
WORKERS="${E6_SWEEP_WORKERS:-27}"
MAX_BATCH="${E6_SWEEP_MAX_BATCH:-32}"

ALPHAS=("$@")
if [[ ${#ALPHAS[@]} -eq 0 ]]; then
    ALPHAS=(0.5 0.75 1.25 1.5 2.0)
fi

mkdir -p "$OUT_ROOT/logs"

for alpha in "${ALPHAS[@]}"; do
    def_dir="$OUT_ROOT/alpha_${alpha}/default"
    ora_dir="$OUT_ROOT/alpha_${alpha}/oracle_gt"
    def_log="$OUT_ROOT/logs/alpha_${alpha}_default.log"
    ora_log="$OUT_ROOT/logs/alpha_${alpha}_oracle_gt.log"

    echo "================================================================"
    echo "[DeltaScaleSweep] alpha=${alpha} | Default (GPU0+EGL1) & Oracle-GT (GPU1+EGL0) | workers=${WORKERS}"
    echo "================================================================"

    (
        E6_EVAL_POLICY_GPU="$GPU0_UUID" \
        E6_EVAL_EGL_DEVICE=1 \
        E6_EVAL_WORKERS="$WORKERS" \
        E6_EVAL_MAX_BATCH="$MAX_BATCH" \
        bash "$ROOT/e6/eval_chain_demoswap.sh" \
            e6_stage2_1 default_paired "$CKPT" "$def_dir" \
            --delta-scale "$alpha"
    ) >"$def_log" 2>&1 &
    PID_DEF=$!

    (
        E6_EVAL_POLICY_GPU="$GPU1_UUID" \
        E6_EVAL_EGL_DEVICE=0 \
        E6_EVAL_WORKERS="$WORKERS" \
        E6_EVAL_MAX_BATCH="$MAX_BATCH" \
        bash "$ROOT/e6/eval_chain_demoswap.sh" \
            e6_stage2_1 oracle_gt "$CKPT" "$ora_dir" \
            --delta-scale "$alpha"
    ) >"$ora_log" 2>&1 &
    PID_ORA=$!

    FAIL=0
    if ! wait "$PID_DEF"; then
        echo "[ERROR] alpha=${alpha} default_paired failed! See $def_log" >&2
        tail -n 30 "$def_log" >&2 || true
        FAIL=1
    fi
    if ! wait "$PID_ORA"; then
        echo "[ERROR] alpha=${alpha} oracle_gt failed! See $ora_log" >&2
        tail -n 30 "$ora_log" >&2 || true
        FAIL=1
    fi
    if [[ $FAIL -ne 0 ]]; then
        exit 1
    fi

    echo "[DeltaScaleSweep DONE] alpha=${alpha}:"
    tail -n 2 "$def_log" || true
    tail -n 2 "$ora_log" || true
done

echo "[DeltaScaleSweep ALL DONE] Completed alphas: ${ALPHAS[*]}"
