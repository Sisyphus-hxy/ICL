"""Aggregate Experiment B (`e9/outputs/direct_identity/`) offline validation, gradient
contribution logs, D1-D4 offline decomposition, and paired closed-loop rollouts
(Default 500, Oracle-GT 331 paired, Shifted 500 paired), comparing head-to-head against
E9 Stage 2 Direct (baseline) and E7-Direct with full init_sha256 verification and W/L/T.
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

from . import DEFAULT_STAGE2_DIRECT_DIR, OUTPUT_ROOT, ROOT

DEFAULT_EXP_B_DIR = OUTPUT_ROOT / "direct_identity"
E7_DIRECT_ROOT = ROOT / "e7/outputs/direct_latent"
E7_D1_D4_SUMMARY = ROOT / "e7/outputs/behavior_space_diagnosis/offline/offline_summary.json"


def _extract_offline(final_full: Dict[str, Any]) -> Dict[str, Any]:
    ev = final_full["final_eval"]
    lt = ev["latent"]
    return {
        "best_epoch": final_full["best_epoch"],
        "best_global_step": final_full.get("best_global_step"),
        "s_z": lt["s_z"],
        "self_teacher_ddim_mse10": ev["ddim"]["self_teacher"]["action10_mse"],
        "cross_no_delta_ddim_mse10": ev["ddim"]["cross_no_delta"]["action10_mse"],
        "cross_with_delta_ddim_mse10": ev["ddim"]["cross_with_delta"]["action10_mse"],
        "cross_with_delta_ddim_mae7": ev["ddim"]["cross_with_delta"]["action7_mae"],
        "cross_with_delta_exec12_mae7": ev["ddim"]["cross_with_delta"]["exec12_action7_mae"],
        "cross_with_delta_grip_acc": ev["ddim"]["cross_with_delta"]["grip_sign_acc"],
        "cross_no_delta_L_z": lt["cross_no_delta_L_z"],
        "cross_with_delta_L_z": lt["cross_with_delta_L_z"],
        "exact_self_identity_L_id": lt.get(
            "exact_self_identity_L_id",
            (lt["self_teacher_delta_rms"] / max(lt["s_z"], 1e-12)) ** 2,
        ),
        "self_teacher_delta_rms": lt["self_teacher_delta_rms"],
        "cross_no_delta_cos": lt["cross_no_delta_cos"],
        "cross_with_delta_cos": lt["cross_with_delta_cos"],
        "wrong_demo_sens_ddim_mse10_with_delta": ev["ddim"]["demo_sensitivity_action10_mse"],
        "wrong_demo_sens_diff_with_delta": ev["diff_loss"]["demo_sensitivity_with_delta"],
        "reader_attention": ev["reader_attention"],
        "multi_demo_consistency": ev.get("multi_demo_consistency"),
    }


def _plot_grad_contributions(grad_csv: Path, metrics_json: Path, out_path: Path) -> None:
    if not grad_csv.is_file():
        return
    rows = list(csv.DictReader(grad_csv.open()))
    if not rows:
        return
    steps = np.array([int(r["global_step"]) for r in rows], dtype=np.float64)
    gn_cross = np.array([float(r["transfer_gn_cross"]) for r in rows], dtype=np.float64)
    gn_id = np.array([float(r["transfer_gn_id"]) for r in rows], dtype=np.float64)
    gn_diff = np.array([float(r["transfer_gn_diff"]) for r in rows], dtype=np.float64)
    gn_z = np.array([float(r["transfer_gn_z"]) for r in rows], dtype=np.float64)
    ratio = np.array([float(r["grad_ratio_id_over_cross"]) for r in rows], dtype=np.float64)
    share = 100.0 * np.array([float(r["grad_share_id"]) for r in rows], dtype=np.float64)
    cos_ci = np.array([float(r["grad_cos_cross_id"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=160)
    axes[0].plot(steps, gn_cross, label="||g_cross|| (L_diff+0.5L_z)", color="#1f77b4", lw=1.6)
    axes[0].plot(steps, gn_diff, label="||g_diff||", color="#3498db", lw=1.1, alpha=0.7)
    axes[0].plot(steps, gn_z, label="||g_z|| (0.5L_z)", color="#9b59b6", lw=1.1, alpha=0.7)
    axes[0].plot(steps, gn_id, label="||g_id|| (0.1L_id)", color="#e67e22", lw=1.8)
    axes[0].set_title("Transfer Gradient Norms over Training", fontweight="bold")
    axes[0].set_xlabel("Global Step")
    axes[0].set_ylabel("L2 Gradient Norm")
    axes[0].grid(alpha=0.3, ls="--")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps, ratio, color="#e67e22", lw=1.8, label="||g_id|| / ||g_cross||")
    ax1_r = axes[1].twinx()
    ax1_r.plot(steps, share, color="#27ae60", lw=1.5, ls="--", label="Identity Share % (right)")
    axes[1].axhline(1.0, color="#c0392b", ls=":", lw=1.0, label="Parity (1.0)")
    axes[1].set_title("Identity vs Cross Gradient Contribution", fontweight="bold")
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Norm Ratio ||g_id|| / ||g_cross||")
    ax1_r.set_ylabel("Share %", color="#27ae60")
    axes[1].grid(alpha=0.3, ls="--")
    axes[1].legend(loc="upper left", fontsize=8)
    ax1_r.legend(loc="upper right", fontsize=8)

    axes[2].plot(steps, cos_ci, color="#8e44ad", lw=1.7, label="cos(g_cross, g_id)")
    axes[2].axhline(0.0, color="#7f8c8d", ls=":", lw=1.0)
    axes[2].set_title("Gradient Alignment cos(g_cross, g_id)", fontweight="bold")
    axes[2].set_xlabel("Global Step")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].grid(alpha=0.3, ls="--")
    axes[2].legend(fontsize=8.5)

    fig.suptitle("Experiment B (E9-Direct + 0.1 Exact Self Identity): Gradient Dynamics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_comparisons(
    summary: Dict[str, Any],
    exp_b_d1_d4: Optional[Dict[str, Any]],
    e9_d1_d4: Optional[Dict[str, Any]],
    e7_d1_d4: Optional[Dict[str, Any]],
    out_dir: Path,
) -> None:
    cl = summary["closed_loop"]
    # 1. Closed-loop 3-way comparison (E7-Direct vs E9 Stage 2 vs Exp B)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.2), dpi=160, gridspec_kw={"width_ratios": [1.05, 2.2]})
    ax0, ax1 = axes
    benches = ["Default\n(500)", "Oracle-GT\n(331)", "Shifted\n(500)", "3-Group\nMean"]
    e7_srs = [
        100.0 * cl["vs_e7_direct"]["default_500"]["b_sr"],
        100.0 * cl["vs_e7_direct"]["oracle_gt_331"]["b_sr"],
        100.0 * cl["vs_e7_direct"]["shifted_500"]["b_sr"],
        100.0 * cl["three_group_mean"]["e7_direct_sr"],
    ]
    e9_srs = [
        100.0 * cl["vs_e9_direct"]["default_500"]["b_sr"],
        100.0 * cl["vs_e9_direct"]["oracle_gt_331"]["b_sr"],
        100.0 * cl["vs_e9_direct"]["shifted_500"]["b_sr"],
        100.0 * cl["three_group_mean"]["e9_direct_sr"],
    ]
    eb_srs = [
        100.0 * cl["vs_e9_direct"]["default_500"]["a_sr"],
        100.0 * cl["vs_e9_direct"]["oracle_gt_331"]["a_sr"],
        100.0 * cl["vs_e9_direct"]["shifted_500"]["a_sr"],
        100.0 * cl["three_group_mean"]["exp_b_sr"],
    ]
    x = np.arange(len(benches))
    w = 0.26
    b1 = ax0.bar(x - w, e7_srs, w, label="E7-Direct", color="#7f8c8d", alpha=0.9)
    b2 = ax0.bar(x, e9_srs, w, label="E9-Direct (Base)", color="#3498db", alpha=0.9)
    b3 = ax0.bar(x + w, eb_srs, w, label="Exp B (+0.1 L_id)", color="#27ae60", alpha=0.95)
    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            ax0.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax0.set_ylim(0, 108)
    ax0.set_xticks(x)
    ax0.set_xticklabels(benches, fontsize=9)
    ax0.set_ylabel("Closed-Loop Success Rate (%)")
    ax0.set_title("Closed-Loop Benchmark Comparison", fontweight="bold")
    ax0.grid(axis="y", ls="--", alpha=0.35)
    ax0.legend(fontsize=8, loc="lower right")

    tasks = get_chain_tasks()
    pt = cl["per_task"]
    tx = np.arange(len(tasks))
    tw = 0.25
    e7_ogt = [100.0 * pt[t]["e7_direct_oracle_gt"]["sr"] if pt[t]["e7_direct_oracle_gt"] else 0.0 for t in tasks]
    e9_ogt = [100.0 * pt[t]["e9_direct_oracle_gt"]["sr"] if pt[t]["e9_direct_oracle_gt"] else 0.0 for t in tasks]
    eb_ogt = [100.0 * pt[t]["exp_b_oracle_gt"]["sr"] if pt[t]["exp_b_oracle_gt"] else 0.0 for t in tasks]
    ax1.bar(tx - tw, e7_ogt, tw, label="E7-Direct Oracle-GT", color="#95a5a6")
    ax1.bar(tx, e9_ogt, tw, label="E9-Direct Oracle-GT", color="#3498db")
    ax1.bar(tx + tw, eb_ogt, tw, label="Exp B (+0.1 L_id) Oracle-GT", color="#27ae60")
    ax1.set_ylim(0, 110)
    ax1.set_xticks(tx)
    ax1.set_xticklabels([SHORT_TASK_NAMES.get(t, t) for t in tasks], rotation=26, ha="right", fontsize=7.8)
    ax1.set_ylabel("Oracle-GT Success Rate (%)")
    ax1.set_title("Per-Task Oracle-GT Closed-Loop SR (331 paired)", fontweight="bold")
    ax1.grid(axis="y", ls="--", alpha=0.35)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "closed_loop_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 2. D1-D4 comparison (E7-Direct vs E9-Direct vs Exp B)
    if exp_b_d1_d4 is not None and e9_d1_d4 is not None and e7_d1_d4 is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), dpi=160)
        dts = ["default", "shifted", "oracle_gt"]
        titles = ["Default (9 shared)", "Shifted (9 shared)", "Oracle-GT (9 shared)"]
        mode_labels = ["D1\nTeacher", "D2\nRetrieval", "D3\nTransfer", "D4\nFull"]
        mx = np.arange(len(MODES))
        mw = 0.26
        for ax, dt, title in zip(axes, dts, titles):
            v_e7 = [e7_d1_d4["by_demo_type_macro_shared_9"][dt]["modes"][m]["ddim_10d_mse"] for m in MODES]
            v_e9 = [e9_d1_d4["by_demo_type_macro_shared_9"][dt]["modes"][m]["ddim_10d_mse"] for m in MODES]
            v_eb = [exp_b_d1_d4["by_demo_type_macro_shared_9"][dt]["modes"][m]["ddim_10d_mse"] for m in MODES]
            b1 = ax.bar(mx - mw, v_e7, mw, label="E7-Direct", color="#7f8c8d", alpha=0.9)
            b2 = ax.bar(mx, v_e9, mw, label="E9-Direct (Base)", color="#3498db", alpha=0.9)
            b3 = ax.bar(mx + mw, v_eb, mw, label="Exp B (+0.1 L_id)", color="#27ae60", alpha=0.95)
            for bars in (b1, b2, b3):
                for b in bars:
                    v = b.get_height()
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=6.4, rotation=25)
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(mx)
            ax.set_xticklabels(mode_labels, fontsize=8.5)
            ax.set_ylabel("DDIM 10D MSE (↓)")
            ax.grid(axis="y", ls="--", alpha=0.35)
        axes[0].legend(fontsize=8, loc="upper right")
        fig.suptitle("Offline D1–D4 Decomposition: E7-Direct vs E9-Direct vs Exp B (+0.1 Exact Self Identity)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "d1_d4_comparison.png", bbox_inches="tight")
        plt.close(fig)


def build_exp_b_analysis(*, exp_b_root: Path = DEFAULT_EXP_B_DIR) -> Dict[str, Any]:
    exp_b_root = Path(exp_b_root)
    eb_final = json.loads((exp_b_root / "train/final_eval.json").read_text())
    e9_final = json.loads((DEFAULT_STAGE2_DIRECT_DIR / "train/final_eval.json").read_text())
    e7_final = json.loads((E7_DIRECT_ROOT / "train/e7_direct_latent/final_eval.json").read_text())

    eb_def_dir = exp_b_root / "closed_loop/default"
    eb_ogt_dir = exp_b_root / "closed_loop/oracle_gt"
    eb_shf_dir = exp_b_root / "closed_loop/shifted"

    e9_def_dir = DEFAULT_STAGE2_DIRECT_DIR / "closed_loop/default"
    e9_ogt_dir = DEFAULT_STAGE2_DIRECT_DIR / "closed_loop/oracle_gt"
    e9_shf_dir = DEFAULT_STAGE2_DIRECT_DIR / "closed_loop/shifted"

    e7_def_dir = E7_DIRECT_ROOT / "eval_chain/e7_direct"
    e7_ogt_dir = E7_DIRECT_ROOT / "demoswap/oracle_gt/e7_direct"
    e7_shf_dir = E7_DIRECT_ROOT / "demoswap/shifted/e7_direct"

    eb_def = _load_rollout_csv(eb_def_dir / "rollout_results.csv")
    eb_ogt = _load_rollout_csv(eb_ogt_dir / "rollout_results.csv")
    eb_shf = _load_rollout_csv(eb_shf_dir / "rollout_results.csv")

    e9_def = _load_rollout_csv(e9_def_dir / "rollout_results.csv")
    e9_ogt = _load_rollout_csv(e9_ogt_dir / "rollout_results.csv")
    e9_shf = _load_rollout_csv(e9_shf_dir / "rollout_results.csv")

    e7_def = _load_rollout_csv(e7_def_dir / "rollout_results.csv")
    e7_ogt = _load_rollout_csv(e7_ogt_dir / "rollout_results.csv")
    e7_shf = _load_rollout_csv(e7_shf_dir / "rollout_results.csv")

    sha_e9_def = _verify_init_sha256(eb_def_dir, e9_def_dir, sorted(set(eb_def) & set(e9_def)))
    sha_e9_ogt = _verify_init_sha256(eb_ogt_dir, e9_ogt_dir, sorted(set(eb_ogt) & set(e9_ogt)))
    sha_e9_shf = _verify_init_sha256(eb_shf_dir, e9_shf_dir, sorted(set(eb_shf) & set(e9_shf)))

    sha_e7_def = _verify_init_sha256(eb_def_dir, e7_def_dir, sorted(set(eb_def) & set(e7_def)))
    sha_e7_ogt = _verify_init_sha256(eb_ogt_dir, e7_ogt_dir, sorted(set(eb_ogt) & set(e7_ogt)))
    sha_e7_shf = _verify_init_sha256(eb_shf_dir, e7_shf_dir, sorted(set(eb_shf) & set(e7_shf)))

    vs_e9_def = _paired_stats(eb_def, e9_def)
    vs_e9_ogt = _paired_stats(eb_ogt, e9_ogt)
    vs_e9_shf = _paired_stats(eb_shf, e9_shf)

    vs_e7_def = _paired_stats(eb_def, e7_def)
    vs_e7_ogt = _paired_stats(eb_ogt, e7_ogt)
    vs_e7_shf = _paired_stats(eb_shf, e7_shf)

    eb_ogt_vs_def = _paired_stats(eb_ogt, eb_def)
    eb_shf_vs_def = _paired_stats(eb_shf, eb_def)

    eb_mean3 = (vs_e9_def["a_sr"] + vs_e9_ogt["a_sr"] + vs_e9_shf["a_sr"]) / 3.0
    e9_mean3 = (vs_e9_def["b_sr"] + vs_e9_ogt["b_sr"] + vs_e9_shf["b_sr"]) / 3.0
    e7_mean3 = (vs_e7_def["b_sr"] + vs_e7_ogt["b_sr"] + vs_e7_shf["b_sr"]) / 3.0

    tasks = get_chain_tasks()
    per_task: Dict[str, Any] = {}
    for t in tasks:
        def _tsr(m):
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
            "exp_b_default": _tsr(eb_def),
            "e9_direct_default": _tsr(e9_def),
            "e7_direct_default": _tsr(e7_def),
            "default_vs_e9": _tpair(eb_def, e9_def),
            "default_vs_e7": _tpair(eb_def, e7_def),
            "exp_b_oracle_gt": _tsr(eb_ogt),
            "e9_direct_oracle_gt": _tsr(e9_ogt),
            "e7_direct_oracle_gt": _tsr(e7_ogt),
            "oracle_gt_vs_e9": _tpair(eb_ogt, e9_ogt),
            "oracle_gt_vs_e7": _tpair(eb_ogt, e7_ogt),
            "exp_b_shifted": _tsr(eb_shf),
            "e9_direct_shifted": _tsr(e9_shf),
            "e7_direct_shifted": _tsr(e7_shf),
            "shifted_vs_e9": _tpair(eb_shf, e9_shf),
            "shifted_vs_e7": _tpair(eb_shf, e7_shf),
        }

    eb_d1_d4_path = exp_b_root / "d1_d4_diagnosis/offline_summary.json"
    e9_d1_d4_path = DEFAULT_STAGE2_DIRECT_DIR / "d1_d4_diagnosis/offline_summary.json"
    eb_d1_d4 = json.loads(eb_d1_d4_path.read_text()) if eb_d1_d4_path.is_file() else None
    e9_d1_d4 = json.loads(e9_d1_d4_path.read_text()) if e9_d1_d4_path.is_file() else None
    e7_d1_d4 = json.loads(E7_D1_D4_SUMMARY.read_text()) if E7_D1_D4_SUMMARY.is_file() else None

    summary: Dict[str, Any] = {
        "offline_val_6248": {
            "exp_b_direct_identity": _extract_offline(eb_final),
            "e9_stage2_direct": _extract_offline(e9_final),
            "e7_direct": _extract_offline(e7_final),
        },
        "closed_loop": {
            "init_sha256_vs_e9": {
                "default_500": sha_e9_def,
                "oracle_gt_331": sha_e9_ogt,
                "shifted_500": sha_e9_shf,
                "all_1331_match": bool(
                    sha_e9_def["all_match"] and sha_e9_ogt["all_match"] and sha_e9_shf["all_match"]
                ),
            },
            "init_sha256_vs_e7": {
                "default_500": sha_e7_def,
                "oracle_gt_331": sha_e7_ogt,
                "shifted_500": sha_e7_shf,
                "all_1331_match": bool(
                    sha_e7_def["all_match"] and sha_e7_ogt["all_match"] and sha_e7_shf["all_match"]
                ),
            },
            "vs_e9_direct": {
                "default_500": vs_e9_def,
                "oracle_gt_331": vs_e9_ogt,
                "shifted_500": vs_e9_shf,
            },
            "vs_e7_direct": {
                "default_500": vs_e7_def,
                "oracle_gt_331": vs_e7_ogt,
                "shifted_500": vs_e7_shf,
            },
            "three_group_mean": {
                "exp_b_sr": eb_mean3,
                "e9_direct_sr": e9_mean3,
                "e7_direct_sr": e7_mean3,
                "delta_vs_e9_pp": 100.0 * (eb_mean3 - e9_mean3),
                "delta_vs_e7_pp": 100.0 * (eb_mean3 - e7_mean3),
            },
            "exp_b_oracle_gt_vs_default_331": eb_ogt_vs_def,
            "exp_b_shifted_vs_default_500": eb_shf_vs_def,
            "per_task": per_task,
        },
        "d1_d4_macro_shared_9": (
            {
                "exp_b_direct_identity": eb_d1_d4["by_demo_type_macro_shared_9"],
                "e9_stage2_direct": e9_d1_d4["by_demo_type_macro_shared_9"],
                "e7_direct": e7_d1_d4["by_demo_type_macro_shared_9"],
            }
            if (eb_d1_d4 is not None and e9_d1_d4 is not None and e7_d1_d4 is not None)
            else None
        ),
        "d1_d4_w_self_ge_0_9": (
            {
                dt: {
                    "exp_b": eb_d1_d4["by_demo_type_micro"][dt]["d3_stratification"].get("w_self_ge_0.9"),
                    "e9": e9_d1_d4["by_demo_type_micro"][dt]["d3_stratification"].get("w_self_ge_0.9"),
                    "e7": e7_d1_d4["by_demo_type_micro"][dt]["d3_stratification"].get("w_self_ge_0.9"),
                }
                for dt in ("default", "shifted", "oracle_gt")
            }
            if (eb_d1_d4 is not None and e9_d1_d4 is not None and e7_d1_d4 is not None)
            else None
        ),
    }

    atomic_write_json(summary, exp_b_root / "comparison_summary.json")
    _plot_grad_contributions(
        exp_b_root / "train/grad_contributions.csv",
        exp_b_root / "train/metrics.json",
        exp_b_root / "grad_contributions.png",
    )
    _plot_comparisons(summary, eb_d1_d4, e9_d1_d4, e7_d1_d4, exp_b_root)
    print(json.dumps(summary["closed_loop"]["three_group_mean"], indent=2))
    print(
        f"init_sha256 vs_e9={summary['closed_loop']['init_sha256_vs_e9']['all_1331_match']} "
        f"vs_e7={summary['closed_loop']['init_sha256_vs_e7']['all_1331_match']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-b-root", type=Path, default=DEFAULT_EXP_B_DIR)
    args = parser.parse_args()
    build_exp_b_analysis(exp_b_root=args.exp_b_root)


if __name__ == "__main__":
    main()
