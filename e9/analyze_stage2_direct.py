"""Aggregate E9 Stage 2 Direct offline validation, closed-loop rollouts (Default 500,
Oracle-GT 331 paired, Shifted 500 paired), and D1-D4 offline decomposition, and compare
head-to-head against E7-Direct Best with full paired W/L/T and init_sha256 verification.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from e5.policy_server import get_chain_tasks
from e5.train import atomic_write_json
from e7.analyze_behavior_space_diagnosis import (
    SHORT_TASK_NAMES,
    _load_rollout_csv,
    _paired_stats,
    _verify_init_sha256,
)
from e7.diagnose_behavior_space import MODES

from . import DEFAULT_STAGE2_DIRECT_DIR, ROOT

E7_DIRECT_ROOT = ROOT / "e7/outputs/direct_latent"
E7_D1_D4_SUMMARY = ROOT / "e7/outputs/behavior_space_diagnosis/offline/offline_summary.json"


def _extract_offline_summary(final_full: Dict[str, Any]) -> Dict[str, Any]:
    ev = final_full["final_eval"]
    return {
        "best_epoch": final_full["best_epoch"],
        "best_global_step": final_full.get("best_global_step"),
        "s_z": ev["latent"]["s_z"],
        "direct_latent_bias_rms": final_full.get("direct_latent_bias_rms"),
        "self_teacher_ddim_mse10": ev["ddim"]["self_teacher"]["action10_mse"],
        "cross_no_delta_ddim_mse10": ev["ddim"]["cross_no_delta"]["action10_mse"],
        "cross_with_delta_ddim_mse10": ev["ddim"]["cross_with_delta"]["action10_mse"],
        "cross_with_delta_ddim_mae7": ev["ddim"]["cross_with_delta"]["action7_mae"],
        "cross_with_delta_exec12_mae7": ev["ddim"]["cross_with_delta"]["exec12_action7_mae"],
        "cross_with_delta_grip_acc": ev["ddim"]["cross_with_delta"]["grip_sign_acc"],
        "cross_no_delta_L_z": ev["latent"]["cross_no_delta_L_z"],
        "cross_with_delta_L_z": ev["latent"]["cross_with_delta_L_z"],
        "cross_no_delta_cos": ev["latent"]["cross_no_delta_cos"],
        "cross_with_delta_cos": ev["latent"]["cross_with_delta_cos"],
        "cross_delta_rms": ev["latent"]["cross_delta_rms"],
        "correction_direction_cosine": ev["latent"]["correction_direction_cosine_mean"],
        "wrong_demo_sens_diff_no_delta": ev["diff_loss"]["demo_sensitivity_no_delta"],
        "wrong_demo_sens_diff_with_delta": ev["diff_loss"]["demo_sensitivity_with_delta"],
        "wrong_demo_sens_ddim_mse10_no_delta": ev["ddim"]["demo_sensitivity_no_delta_action10_mse"],
        "wrong_demo_sens_ddim_mse10_with_delta": ev["ddim"]["demo_sensitivity_action10_mse"],
        "wrong_demo_sens_ddim_mae7_with_delta": ev["ddim"]["demo_sensitivity_action7_mae"],
        "reader_attention": ev["reader_attention"],
        "multi_demo_consistency": ev["multi_demo_consistency"],
    }


def _plot_closed_loop_and_d1_d4(
    summary: Dict[str, Any],
    e9_d1_d4: Optional[Dict[str, Any]],
    e7_d1_d4: Optional[Dict[str, Any]],
    out_dir: Path,
) -> None:
    cl = summary["closed_loop"]
    # 1. Closed-loop SR overview + per-task bar chart
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), dpi=160, gridspec_kw={"width_ratios": [1.0, 2.2]})
    ax0, ax1 = axes

    benches = ["Default\n(500)", "Oracle-GT\n(331)", "Shifted\n(500)", "3-Group\nMean"]
    e7_srs = [
        100.0 * cl["default_500"]["b_sr"],
        100.0 * cl["oracle_gt_331"]["b_sr"],
        100.0 * cl["shifted_500"]["b_sr"],
        100.0 * cl["three_group_mean"]["e7_direct_sr"],
    ]
    e9_srs = [
        100.0 * cl["default_500"]["a_sr"],
        100.0 * cl["oracle_gt_331"]["a_sr"],
        100.0 * cl["shifted_500"]["a_sr"],
        100.0 * cl["three_group_mean"]["e9_stage2_sr"],
    ]
    x = np.arange(len(benches))
    w = 0.35
    b1 = ax0.bar(x - w / 2, e7_srs, w, label="E7-Direct (Old S1)", color="#7f8c8d", alpha=0.9)
    b2 = ax0.bar(x + w / 2, e9_srs, w, label="E9 Stage 2 (Local S1)", color="#1f77b4", alpha=0.92)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax0.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax0.set_ylim(0, 105)
    ax0.set_xticks(x)
    ax0.set_xticklabels(benches, fontsize=9)
    ax0.set_ylabel("Closed-Loop Success Rate (%)")
    ax0.set_title("Closed-Loop Benchmark Comparison", fontweight="bold")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(fontsize=8.5, loc="lower right")

    tasks = get_chain_tasks()
    pt = cl["per_task"]
    tx = np.arange(len(tasks))
    tw = 0.2
    e7_def = [100.0 * pt[t]["e7_direct_default"]["sr"] for t in tasks]
    e9_def = [100.0 * pt[t]["e9_direct_default"]["sr"] for t in tasks]
    e7_ogt = [100.0 * pt[t]["e7_direct_oracle_gt"]["sr"] if pt[t]["e7_direct_oracle_gt"] else 0.0 for t in tasks]
    e9_ogt = [100.0 * pt[t]["e9_direct_oracle_gt"]["sr"] if pt[t]["e9_direct_oracle_gt"] else 0.0 for t in tasks]
    ax1.bar(tx - 1.5 * tw, e7_def, tw, label="E7-Direct Default", color="#95a5a6")
    ax1.bar(tx - 0.5 * tw, e9_def, tw, label="E9 Stage 2 Default", color="#3498db")
    ax1.bar(tx + 0.5 * tw, e7_ogt, tw, label="E7-Direct Oracle-GT", color="#e67e22", alpha=0.85)
    ax1.bar(tx + 1.5 * tw, e9_ogt, tw, label="E9 Stage 2 Oracle-GT", color="#27ae60", alpha=0.9)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(tx)
    ax1.set_xticklabels([SHORT_TASK_NAMES.get(t, t) for t in tasks], rotation=26, ha="right", fontsize=7.8)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Per-Task Closed-Loop SR: Default & Oracle-GT", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 2. D1-D4 offline decomposition comparison (E9 vs E7-Direct)
    if e9_d1_d4 is not None and e7_d1_d4 is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)
        dts = ["default", "shifted", "oracle_gt"]
        titles = ["Default (9 shared)", "Shifted (9 shared)", "Oracle-GT (9 shared)"]
        mode_labels = ["D1\nTeacher", "D2\nRetrieval", "D3\nTransfer", "D4\nFull"]
        mx = np.arange(len(MODES))
        mw = 0.35
        for ax, dt, title in zip(axes, dts, titles):
            v_e7 = [e7_d1_d4["by_demo_type_macro_shared_9"][dt]["modes"][m]["ddim_10d_mse"] for m in MODES]
            v_e9 = [e9_d1_d4["by_demo_type_macro_shared_9"][dt]["modes"][m]["ddim_10d_mse"] for m in MODES]
            b1 = ax.bar(mx - mw / 2, v_e7, mw, label="E7-Direct (Old S1)", color="#7f8c8d", alpha=0.9)
            b2 = ax.bar(mx + mw / 2, v_e9, mw, label="E9 Stage 2 (Local S1)", color="#1f77b4", alpha=0.92)
            for bars in (b1, b2):
                for b in bars:
                    v = b.get_height()
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=20)
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(mx)
            ax.set_xticklabels(mode_labels, fontsize=8.5)
            ax.set_ylabel("DDIM 10D MSE (↓)")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
        axes[0].legend(fontsize=8.5, loc="upper right")
        fig.suptitle("Offline D1–D4 Decomposition: E7-Direct vs E9 Stage 2 Direct (9-Task Equal-Weight Macro)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "d1_d4_comparison.png", bbox_inches="tight")
        plt.close(fig)


def build_stage2_analysis(
    *,
    stage2_root: Path = DEFAULT_STAGE2_DIRECT_DIR,
) -> Dict[str, Any]:
    stage2_root = Path(stage2_root)
    e9_final = json.loads((stage2_root / "train/final_eval.json").read_text())
    e7_final = json.loads(
        (E7_DIRECT_ROOT / "train/e7_direct_latent/final_eval.json").read_text()
    )

    e9_def_dir = stage2_root / "closed_loop/default"
    e9_ogt_dir = stage2_root / "closed_loop/oracle_gt"
    e9_shf_dir = stage2_root / "closed_loop/shifted"

    e7_def_dir = E7_DIRECT_ROOT / "eval_chain/e7_direct"
    e7_ogt_dir = E7_DIRECT_ROOT / "demoswap/oracle_gt/e7_direct"
    e7_shf_dir = E7_DIRECT_ROOT / "demoswap/shifted/e7_direct"

    e9_def = _load_rollout_csv(e9_def_dir / "rollout_results.csv")
    e9_ogt = _load_rollout_csv(e9_ogt_dir / "rollout_results.csv")
    e9_shf = _load_rollout_csv(e9_shf_dir / "rollout_results.csv")

    e7_def = _load_rollout_csv(e7_def_dir / "rollout_results.csv")
    e7_ogt = _load_rollout_csv(e7_ogt_dir / "rollout_results.csv")
    e7_shf = _load_rollout_csv(e7_shf_dir / "rollout_results.csv")

    sha_def = _verify_init_sha256(e9_def_dir, e7_def_dir, sorted(set(e9_def) & set(e7_def)))
    sha_ogt = _verify_init_sha256(e9_ogt_dir, e7_ogt_dir, sorted(set(e9_ogt) & set(e7_ogt)))
    sha_shf = _verify_init_sha256(e9_shf_dir, e7_shf_dir, sorted(set(e9_shf) & set(e7_shf)))

    def_pair = _paired_stats(e9_def, e7_def)
    ogt_pair = _paired_stats(e9_ogt, e7_ogt)
    shf_pair = _paired_stats(e9_shf, e7_shf)

    e9_ogt_vs_def = _paired_stats(e9_ogt, e9_def)
    e9_shf_vs_def = _paired_stats(e9_shf, e9_def)
    e7_ogt_vs_def = _paired_stats(e7_ogt, e7_def)
    e7_shf_vs_def = _paired_stats(e7_shf, e7_def)

    e9_mean3 = (def_pair["a_sr"] + ogt_pair["a_sr"] + shf_pair["a_sr"]) / 3.0
    e7_mean3 = (def_pair["b_sr"] + ogt_pair["b_sr"] + shf_pair["b_sr"]) / 3.0

    tasks = get_chain_tasks()
    per_task: Dict[str, Any] = {}
    for t in tasks:
        def _tsr(m: Dict[Tuple[str, int], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            sub = [v["success"] for (tk, _), v in m.items() if tk == t]
            if not sub:
                return None
            ok = sum(1 for v in sub if v)
            return {"ok": ok, "n": len(sub), "sr": ok / len(sub)}

        def _tpair(ma, mb):
            sa = {k: v for k, v in ma.items() if k[0] == t}
            sb = {k: v for k, v in mb.items() if k[0] == t}
            return _paired_stats(sa, sb) if (sa and sb) else None

        per_task[t] = {
            "e7_direct_default": _tsr(e7_def),
            "e9_direct_default": _tsr(e9_def),
            "default_paired": _tpair(e9_def, e7_def),
            "e7_direct_oracle_gt": _tsr(e7_ogt),
            "e9_direct_oracle_gt": _tsr(e9_ogt),
            "oracle_gt_paired": _tpair(e9_ogt, e7_ogt),
            "e9_ogt_vs_def_paired": _tpair(e9_ogt, e9_def),
            "e7_direct_shifted": _tsr(e7_shf),
            "e9_direct_shifted": _tsr(e9_shf),
            "shifted_paired": _tpair(e9_shf, e7_shf),
        }

    e9_d1_d4_path = stage2_root / "d1_d4_diagnosis/offline_summary.json"
    e9_d1_d4 = json.loads(e9_d1_d4_path.read_text()) if e9_d1_d4_path.is_file() else None
    e7_d1_d4 = json.loads(E7_D1_D4_SUMMARY.read_text()) if E7_D1_D4_SUMMARY.is_file() else None

    summary: Dict[str, Any] = {
        "offline_val_6248": {
            "e7_direct": _extract_offline_summary(e7_final),
            "e9_stage2_direct": _extract_offline_summary(e9_final),
        },
        "closed_loop": {
            "init_sha256_verification": {
                "default_500": sha_def,
                "oracle_gt_331": sha_ogt,
                "shifted_500": sha_shf,
                "all_1331_match": bool(
                    sha_def["all_match"] and sha_ogt["all_match"] and sha_shf["all_match"]
                ),
            },
            "default_500": def_pair,
            "oracle_gt_331": ogt_pair,
            "shifted_500": shf_pair,
            "three_group_mean": {
                "e9_stage2_sr": e9_mean3,
                "e7_direct_sr": e7_mean3,
                "delta_pp": 100.0 * (e9_mean3 - e7_mean3),
            },
            "e9_oracle_gt_vs_default_331": e9_ogt_vs_def,
            "e9_shifted_vs_default_500": e9_shf_vs_def,
            "e7_oracle_gt_vs_default_331": e7_ogt_vs_def,
            "e7_shifted_vs_default_500": e7_shf_vs_def,
            "per_task": per_task,
        },
        "d1_d4_macro_shared_9": (
            {
                "e9_stage2_direct": e9_d1_d4["by_demo_type_macro_shared_9"],
                "e7_direct": e7_d1_d4["by_demo_type_macro_shared_9"],
            }
            if (e9_d1_d4 is not None and e7_d1_d4 is not None)
            else None
        ),
        "d1_d4_w_self_ge_0_9": (
            {
                dt: {
                    "e9": e9_d1_d4["by_demo_type_micro"][dt]["d3_stratification"].get("w_self_ge_0.9"),
                    "e7": e7_d1_d4["by_demo_type_micro"][dt]["d3_stratification"].get("w_self_ge_0.9"),
                }
                for dt in ("default", "shifted", "oracle_gt")
            }
            if (e9_d1_d4 is not None and e7_d1_d4 is not None)
            else None
        ),
    }

    atomic_write_json(summary, stage2_root / "comparison_vs_e7_direct.json")
    _plot_closed_loop_and_d1_d4(summary, e9_d1_d4, e7_d1_d4, stage2_root)
    print(json.dumps(summary["closed_loop"]["three_group_mean"], indent=2))
    print(
        f"init_sha256 all_1331_match={summary['closed_loop']['init_sha256_verification']['all_1331_match']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-root", type=Path, default=DEFAULT_STAGE2_DIRECT_DIR)
    args = parser.parse_args()
    build_stage2_analysis(stage2_root=args.stage2_root)


if __name__ == "__main__":
    main()
