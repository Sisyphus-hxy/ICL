#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${1:-e6/outputs/demoswap}"

run_one() {
    local mode="$1"
    local prompt_mode="$2"
    local ckpt="$3"
    local base_csv="$4"
    local out_dir="$OUT_ROOT/$prompt_mode/$mode"
    echo "============================================================"
    echo "[Suite] mode=$mode prompt_mode=$prompt_mode -> $out_dir"
    echo "============================================================"
    bash "$ROOT/e6/eval_chain_demoswap.sh" \
        "$mode" "$prompt_mode" "$ckpt" "$out_dir" \
        --baseline-csv "$base_csv"
}

BPP_RGB_CKPT="e5/outputs/training/bpp_rgb_only_fullpass/best.pt"
BPP_RGB_BASE="e5/outputs/evaluation/bpp_rgb_only_fullpass/liberogen_goal_chain/rollout_results.csv"

E5_CKPT="e5/outputs/training/e5_fullpass/best.pt"
E5_BASE="e5/outputs/evaluation/e5_fullpass/liberogen_goal_chain/rollout_results.csv"

E6_CKPT="e6/outputs/stage2/stage2_latent_transfer/best.pt"
E6_BASE="e6/outputs/stage2/eval_chain/e6_stage2/rollout_results.csv"

# 1. Oracle-GT same-layout non-default demos (331 paired trials, matches bpp/experiments/oracle_gt)
run_one bpp_rgb_only oracle_gt "$BPP_RGB_CKPT" "$BPP_RGB_BASE"
run_one e5           oracle_gt "$E5_CKPT"      "$E5_BASE"
run_one e6_stage2    oracle_gt "$E6_CKPT"      "$E6_BASE"

# 2. Shifted same-task unseen-HDF5 demos: demo_10..19 instead of default demo_0..9 (500 paired trials)
run_one bpp_rgb_only shifted   "$BPP_RGB_CKPT" "$BPP_RGB_BASE"
run_one e5           shifted   "$E5_CKPT"      "$E5_BASE"
run_one e6_stage2    shifted   "$E6_CKPT"      "$E6_BASE"

echo "[Suite DONE] All 6 non-default demo evaluations finished -> $OUT_ROOT"
