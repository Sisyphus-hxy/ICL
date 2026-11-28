"""Generate publication-grade figures (fig1..fig7), CSV tables, and complete REPORT.md for E17."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import E16_OUTPUT_ROOT, OUTPUT_ROOT

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
})


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text()) if p.is_file() else {}


def _load_eval(key: str) -> Optional[Dict[str, Any]]:
    for base in (OUTPUT_ROOT / "eval" / key / "eval_e15.json", E16_OUTPUT_ROOT / "eval" / key / "eval_e15.json"):
        if base.is_file():
            return json.loads(base.read_text())
    return None


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _extract_eval_row(key: str, label: str, group: str, d: Dict[str, Any]) -> Dict[str, Any]:
    fid = d["fidelity"]["self_ddim"]
    t2 = d["editability"]["displacement_and_arithmetic"]["latent_arithmetic_tier2_distinct_start"]
    op = d["editability"]["op_subspace_editability"]["op_only_arith"]
    ur = d["editability"]["unseen_combinations"]["linear_compositional_ridge"]["unseen_ddim"]
    pf = d["content_probes"]["full"]
    pop = d["content_probes"]["op_slice"]
    prs = d["content_probes"]["res_slice"]
    tce = d["content_probes"]["three_component_energy"]
    vk = d["robustness_and_visual_icl"]["visual_knn"]
    rp = d["robustness_and_visual_icl"]["rel_progress"]
    return {
        "key": key,
        "label": label,
        "group": group,
        "fid_mse10": float(fid["mse10"]),
        "fid_mae7": float(fid["mae7"]),
        "fid_grip": float(fid["grip_acc"]),
        "tier2_arr": float(t2["arithmetic_recovery_ratio_10d"]),
        "tier2_cos": float(t2["latent_disp_cos"]),
        "op_edit_arr": float(op["arr_10d"]),
        "unseen_ridge": float(ur["action10_mse"]),
        "r2_a_to_z": float(pf["r2_a_to_z_mean"]),
        "r2_z_to_a7": float(pf["r2_z_to_a7_mean"]),
        "fut_r2": float(pf["fut_r2_mean"]),
        "vis_hq_r2": float(pf["vis_hq_r2_mean"]),
        "erank": float(pf["erank_mean"]),
        "op_erank": float(pop["erank_mean"]),
        "res_erank": float(prs["erank_mean"]),
        "cka_op_res": float(tce["cka_op_res_mean"]),
        "vk_lin4k": float(vk["visual_icl"]["adapters"]["lin_full"]["4000"]["mse10_mean"]),
        "vk_lin4k_arr": float(vk["visual_icl"]["adapters"]["lin_full"]["4000"]["arr_10d_mean"]),
        "vk_rbf4k": float(vk["visual_icl"]["adapters"]["rbf_krr_full"]["4000"]["mse10_mean"]),
        "vk_r2_z": float(vk["visual_icl"]["adapters"]["lin_full"]["4000"]["r2_z_mean"]),
        "vk_excrms": float(vk["decoder_robustness"]["factorization_linear"]["excess_act_rms_mean"]),
        "rp_lin4k": float(rp["visual_icl"]["adapters"]["lin_full"]["4000"]["mse10_mean"]),
        "rp_lin4k_arr": float(rp["visual_icl"]["adapters"]["lin_full"]["4000"]["arr_10d_mean"]),
        "rp_rbf4k": float(rp["visual_icl"]["adapters"]["rbf_krr_full"]["4000"]["mse10_mean"]),
        "rp_r2_z": float(rp["visual_icl"]["adapters"]["lin_full"]["4000"]["r2_z_mean"]),
        "rp_elat": float(rp["decoder_robustness"]["factorization_linear"]["e_lat_rel_mean"]),
        "rp_gdec": float(rp["decoder_robustness"]["factorization_linear"]["G_dec_excess_mean"]),
        "rp_excrms": float(rp["decoder_robustness"]["factorization_linear"]["excess_act_rms_mean"]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_fig1_authenticity(auth: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))

    # Panel A: Controls on dv_R2 for D4 vs E17-M2 vs SameTask Shortcut
    conds = [
        ("vision_only_mlp", "VO-MLP"),
        ("zero_donor", "Zero-z"),
        ("mean_donor", "Mean-z"),
        ("random_donor", "Rand-z"),
        ("wrong_donor_opp", "Wrong-z"),
        ("nn_scene_v0", "v0-NN"),
        ("nn_motion_flow", "Flow-NN"),
        ("nn_z_cosine", "z-NN"),
        ("learned_W_top1", "W Top-1"),
        ("learned_W_soft", "W Soft"),
        ("self_oracle", "Self"),
    ]
    x = np.arange(len(conds))
    w = 0.27
    show_models = [
        ("e16_d4_learn_op", "E16-D4 (AntiScene Soft)", "#7c3aed"),
        ("e17_m2_compact_op", "E17-M2 (PureVid Sharp)", "#059669"),
        ("cov_d4_sametask_16k", "SameTask (Shortcut)", "#dc2626"),
    ]
    for mi, (mk, lbl, col) in enumerate(show_models):
        if mk not in auth:
            continue
        ct = auth[mk]["controls_by_pool"]["strict_cross_task"]
        vals = [max(-0.95, float(ct[c]["dv_r2"])) for c, _ in conds]
        axes[0].bar(x + (mi - 1) * w, vals, w, label=lbl, color=col, alpha=0.88)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in conds], rotation=32, ha="right", fontsize=7.8)
    axes[0].set_ylabel("Forward State-Change R² (clipped @-0.95)")
    axes[0].set_ylim(-1.0, 0.85)
    axes[0].set_title("(A) Donor Authenticity Controls on Cross-Task Δv Prediction")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7.6)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B: Scene-Dissimilarity Quartile Stratification (Q1=most dissimilar -> Q4)
    q_models = [
        ("e16_d4_learn_op", "E16-D4 (AntiScene)", "#7c3aed", "o--"),
        ("e17_m1_joint_op", "E17-M1 (Sharp Joint)", "#0284c7", "s-"),
        ("e17_m2_compact_op", "E17-M2 (Sharp Compact)", "#059669", "D-"),
        ("cov_d4_unmasked_16k", "Cov-Unmasked (PureVid)", "#d97706", "^-"),
        ("cov_d4_strict_ct_16k", "Cov-StrictCT (TaskMeta)", "#ea580c", "v:"),
        ("cov_d4_sametask_16k", "Cov-SameTask (Shortcut)", "#dc2626", "x--"),
    ]
    qx = np.arange(4)
    for mk, lbl, col, sty in q_models:
        if mk not in auth:
            continue
        qs = auth[mk]["scene_quartile_stratification"]
        reds = [qs[f"Q{i+1}"]["rel_act_err_reduction"] * 100.0 for i in range(4)]
        axes[1].plot(qx, reds, sty, color=col, lw=2.0, markersize=6, label=lbl)
    axes[1].axhline(0, color="black", lw=0.8, ls=":")
    axes[1].set_xticks(qx)
    axes[1].set_xticklabels(["Q1\n(Most Dissim)", "Q2", "Q3", "Q4\n(Most Sim)"])
    axes[1].set_ylabel("W Top-1 Action Error Reduction vs Random (%)")
    axes[1].set_title("(B) Cross-Task Reuse Across Scene-Similarity Quartiles")
    axes[1].legend(frameon=False, loc="upper left", fontsize=7.4)
    axes[1].grid(alpha=0.25)

    # Panel C: Predictor Sensitivity (z/v0 ratio) vs Counterfactual Action Adoption
    cf_models = [
        ("e16_d4_learn_op", "D4"),
        ("e17_m1_joint_op", "E17-M1"),
        ("e17_m2_compact_op", "E17-M2"),
        ("e17_m3_dual_orth", "E17-M3"),
        ("e17_m4_compact_reuse", "E17-M4"),
        ("e16_c2_op_learnable", "E16-C2"),
        ("e16_cov_unmasked", "R1-Unmask"),
        ("e16_cov_sametask", "R1-SameTask"),
    ]
    for mk, short in cf_models:
        if mk not in auth:
            continue
        r = auth[mk]
        zv = r["predictor_sensitivity"]["z_to_v0_sensitivity_ratio"]
        cf = r["counterfactual_interventions"]["anti_scene_opposite_act"]
        adopt = cf.get("action_counterfactual", {}).get("action_adoption_ratio", float("nan")) * 100.0
        col = "#dc2626" if "sametask" in mk else ("#059669" if mk.startswith("e17_") else "#2563eb")
        axes[2].scatter(zv, adopt, c=col, s=85, edgecolors="black", lw=0.7, zorder=4)
        ox = -44 if short in ("E17-M2", "R1-Unmask") else 6
        oy = -13 if short in ("E17-M1", "E16-C2") else (5 if short == "R1-Unmask" else 4)
        axes[2].annotate(short, (zv, adopt), xytext=(ox, oy), textcoords="offset points", fontsize=7.8)
    axes[2].set_xlabel("Predictor Sensitivity Ratio ||ΔF_z|| / ||ΔF_v0|| (↑ Less Shortcut)")
    axes[2].set_ylabel("Counterfactual Anti-Scene Action Adoption % (↑)")
    axes[2].set_title("(C) Predictor z-Utilization vs Counterfactual Action Transfer")
    axes[2].set_xlim(0.1, 1.05)
    axes[2].set_ylim(62, 104)
    axes[2].grid(alpha=0.25)

    fig.suptitle("E17 Exp 1: Relation Authenticity Audit — Exposing Shortcuts & Convex-Hull Averaging", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_relation_authenticity_controls.png")
    plt.close(fig)


def _plot_fig2_relation_graphs(auth: Dict[str, Any], fig_dir: Path) -> None:
    viz_dir = OUTPUT_ROOT / "exp1_authenticity" / "viz_data"
    specs = [
        ("S_act", "e17_m2_compact_op", "(A) Oracle GT Action Similarity S_act\n(Sorted by Gripper × Dir)"),
        ("W", "e16_d4_learn_op", "(B) E16-D4 Learned W_ij\n(Anti-Scene Soft Averaging)"),
        ("W", "e17_m2_compact_op", "(C) E17-M2 Learned W_ij\n(Pure-Video Sharp Top-16/Top-1)"),
        ("W", "cov_d4_unmasked_16k", "(D) Cov-Unmasked W_ij\n(Pure Video, Zero Task Meta)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))
    for ax, (key, mname, title) in zip(axes, specs):
        p = viz_dir / f"graph_{mname}.npz"
        if not p.is_file():
            ax.set_title(f"{title}\n(missing)")
            continue
        arr = np.load(p)
        mat = arr[key]
        if key == "W":
            # Log-scaled visualization so sparse sharp weights & diffuse weights are both crystal clear
            disp = np.log10(np.clip(mat, 1e-4, 1.0))
            im = ax.imshow(disp, cmap="magma", aspect="equal", vmin=-3.5, vmax=-0.5)
        else:
            im = ax.imshow(mat, cmap="coolwarm", aspect="equal", vmin=-0.8, vmax=0.8)
        # Mark gripper boundary
        gs = arr["grip_sign"]
        split = int(np.searchsorted(gs, 1))
        if 0 < split < len(gs):
            ax.axhline(split, color="cyan", ls="--", lw=0.9, alpha=0.8)
            ax.axvline(split, color="cyan", ls="--", lw=0.9, alpha=0.8)
        ax.set_title(title, fontsize=9.0)
        ax.set_xlabel("Donor Clip j (sorted)")
        ax.set_ylabel("Query Clip i (sorted)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Learned Cross-Task Relation Graph W_ij Visualization (128 Held-Out Unseen-Task Clips; Cyan = Gripper Boundary)", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_relation_graph_visualizations.png")
    plt.close(fig)


def _plot_fig3_emergent_factors(exp2: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))

    # Panel A: Token Swap (Hungarian) vs Unsupervised ICA Subspace Swap vs Unseen Compositional Generalization
    order = [
        ("e16_d1_lam", "D1 LAM"),
        ("e16_d3_vid_op", "D3 VidOp"),
        ("e16_d4_learn_op", "D4 LearnOp"),
        ("e17_m1_joint_op", "E17-M1"),
        ("e17_m2_compact_op", "E17-M2"),
        ("e17_m3_dual_orth", "E17-M3"),
        ("e17_m4_compact_reuse", "E17-M4"),
        ("e16_d_oracle_e9", "Oracle-E9"),
        ("e12_bc_factordisp", "E12-BC-FD"),
    ]
    order = [(k, l) for k, l in order if k in exp2]
    x = np.arange(len(order))
    w = 0.25
    tok_h = [exp2[k]["causal_token_swap_full"]["hungarian_mean_adoption"] * 100.0 for k, _ in order]
    ica_s = [exp2[k]["unsupervised_subspace_interventions"]["ica_top3"]["mean_adoption"] * 100.0 for k, _ in order]
    op_cg = [exp2[k]["compositional_generalization_unseen_dir_grip"].get("op_subspace_arith", {}).get("joint_compositional_success", 0.0) * 100.0 for k, _ in order]
    axes[0].bar(x - w, tok_h, w, label="Optimal Hungarian TokSwap %", color="#3b82f6")
    axes[0].bar(x, ica_s, w, label="Unsupervised ICA-Top3 Swap %", color="#8b5cf6")
    axes[0].bar(x + w, op_cg, w, label="Unseen (Dir×Grip) z_op Arith %", color="#10b981")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([l for _, l in order], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Adoption / Compositional Success (%)")
    axes[0].set_title("(A) Emergent Causal Editability & Unseen Composition")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B: Dense Episode Temporal Dynamics — Subgoal Changepoint AUC & ICA Timescale Ratio
    auc_op = [exp2[k]["temporal_dynamics"]["changepoint_subgoal_auc_op"] for k, _ in order]
    auc_f = [exp2[k]["temporal_dynamics"]["changepoint_subgoal_auc_full"] for k, _ in order]
    hl_r = [exp2[k]["temporal_dynamics"]["ica_half_life_ratio_max_min"] for k, _ in order]
    axes[1].bar(x - 0.18, auc_op, 0.34, label="Subgoal Changepoint AUC (z_op)", color="#059669", alpha=0.88)
    axes[1].bar(x + 0.18, auc_f, 0.34, label="Subgoal Changepoint AUC (z_full)", color="#0ea5e9", alpha=0.72)
    axes[1].axhline(0.5, color="gray", ls=":", lw=0.9)
    axes[1].set_ylim(0.45, 0.84)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l for _, l in order], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("ROC-AUC Aligning with Grasp/Reversal Events (↑)")
    axes[1].set_title("(B) Unsupervised Subgoal Boundary Alignment (4,584 Steps)")
    axes[1].legend(frameon=False, loc="upper left", fontsize=7.5)
    axes[1].grid(axis="y", alpha=0.25)

    # Panel C: Per-ICA-Factor Timescale Autocorrelation Curves for E16-D4 vs E17-M4 vs Oracle-E9
    lags = [2, 4, 8, 16, 32, 48, 64]
    for mk, lbl, col in [("e16_d1_lam", "D1 Pure LAM", "#94a3b8"), ("e16_d4_learn_op", "E16-D4 LearnOp", "#7c3aed"), ("e17_m4_compact_reuse", "E17-M4 CompactReuse", "#059669"), ("e16_d_oracle_e9", "Oracle-E9", "#1e293b")]:
        if mk not in exp2:
            continue
        iac = exp2[mk]["temporal_dynamics"]["ica_autocorr"]
        items = sorted(iac.values(), key=lambda v: v["half_life_frames"])
        fast_c = [items[0]["curve"][str(l)] for l in lags]
        slow_c = [items[-1]["curve"][str(l)] for l in lags]
        axes[2].plot(lags, fast_c, "--", color=col, lw=1.5, alpha=0.7)
        axes[2].plot(lags, slow_c, "o-", color=col, lw=2.0, markersize=4, label=f"{lbl} (HL {items[0]['half_life_frames']:.0f}–{items[-1]['half_life_frames']:.0f}f)")
    axes[2].axhline(0.5, color="red", ls=":", lw=0.9, alpha=0.7)
    axes[2].set_xlabel("Frame Lag Δt (20 Hz)")
    axes[2].set_ylabel("ICA Factor Autocorrelation")
    axes[2].set_title("(C) Fast vs Slow Emergent ICA Factor Timescales")
    axes[2].legend(frameon=False, loc="upper right", fontsize=7.4)
    axes[2].grid(alpha=0.25)

    fig.suptitle("E17 Exp 2: Emergent Behavior Factors — Unsupervised Discovery, Subspace Edits & Temporal Structure", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_emergent_behavior_factors.png")
    plt.close(fig)


def _plot_fig4_d3_d4_complementarity(exp3: Dict[str, Any], by_key: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))

    # Panel A: Cumulative Energy Spectra of z_full
    spec_models = [
        ("e16_d1_lam", "D1 Pure LAM", "#94a3b8", ":"),
        ("e16_d3_vid_op", "D3 VidRel-Op", "#0284c7", "--"),
        ("e16_d4_learn_op", "D4 LearnReuse-Op", "#7c3aed", "--"),
        ("e17_m1_joint_op", "E17-M1 Joint", "#2563eb", "-"),
        ("e17_m2_compact_op", "E17-M2 Compact", "#059669", "-"),
        ("e17_m3_dual_orth", "E17-M3 DualOrth", "#d97706", "-"),
        ("e17_m4_compact_reuse", "E17-M4 CompactReuse", "#ec4899", "-"),
        ("e16_d_oracle_e9", "Oracle-E9", "#1e293b", "-."),
    ]
    for mk, lbl, col, ls in spec_models:
        if mk not in exp3:
            continue
        cum = exp3[mk]["spectrum"]["full"]["cum_energy_fracs"]
        ks = np.arange(1, len(cum) + 1)
        er = exp3[mk]["spectrum"]["full"]["erank"]
        axes[0].plot(ks, cum, ls=ls, color=col, lw=1.9, label=f"{lbl} (er={er:.0f})")
    axes[0].axhline(0.90, color="gray", ls=":", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Top-k Principal Components (log scale)")
    axes[0].set_ylabel("Cumulative Explained Variance Fraction")
    axes[0].set_title("(A) Full Latent Spectral Concentration (3072D)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7.2)
    axes[0].grid(alpha=0.25)

    # Panel B: Decoder Subspace Reliance (SwapAdopt op vs res) & Visual ICL Error Split
    bar_models = [
        ("e16_d1_lam", "D1"), ("e16_d2_randrel", "D2"), ("e16_d3_vid_op", "D3"),
        ("e17_d3_eqparam", "D3-EqP"), ("e16_d4_learn_op", "D4"),
        ("e17_m1_joint_op", "M1"), ("e17_m2_compact_op", "M2"),
        ("e17_m3_dual_orth", "M3"), ("e17_m4_compact_reuse", "M4"), ("e16_d_oracle_e9", "E9"),
    ]
    bar_models = [(k, l) for k, l in bar_models if k in exp3]
    bx = np.arange(len(bar_models))
    op_sw = [exp3[k]["decoder_sensitivity"]["swap_op_from_donor"]["donor_adoption_ratio"] * 100.0 for k, _ in bar_models]
    rs_sw = [exp3[k]["decoder_sensitivity"]["swap_res_from_donor"]["donor_adoption_ratio"] * 100.0 for k, _ in bar_models]
    axes[1].bar(bx - 0.18, op_sw, 0.35, label="z_op Swap Adoption %", color="#2563eb")
    axes[1].bar(bx + 0.18, rs_sw, 0.35, label="z_res Swap Adoption %", color="#f59e0b")
    axes[1].set_xticks(bx)
    axes[1].set_xticklabels([l for _, l in bar_models], rotation=28, ha="right", fontsize=8)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("Frozen-Decoder Cross-Task Swap Adoption %")
    axes[1].set_title("(B) Frozen Decoder Subspace Reliance (z_op vs z_res)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=7.6)
    axes[1].grid(axis="y", alpha=0.25)

    # Panel C: 2x2 Factorial Pareto Frontier (RelP RBF vs Tier-2 Edit ARR)
    pareto_specs = [
        ("e16_d1_lam_dec100", "D1 LAM", "#94a3b8", "o"),
        ("e16_d2_randrel_dec100", "D2 Rand", "#cbd5e1", "X"),
        ("e16_d3_vidop_dec100", "D3 VidOp", "#0284c7", "^"),
        ("e17_d3_eqparam_dec100", "D3-EqP", "#38bdf8", "v"),
        ("e16_d4_learnop_dec100", "D4 LearnOp", "#7c3aed", "s"),
        ("e17_m1_joint_op_dec100", "E17-M1", "#2563eb", "D"),
        ("e17_m2_compact_op_dec100", "E17-M2", "#059669", "*"),
        ("e17_m3_dual_orth_dec100", "E17-M3", "#d97706", "P"),
        ("e17_m4_compact_reuse_dec100", "E17-M4", "#ec4899", "*"),
        ("e16_d_oracle_e9_dec100", "Oracle-E9", "#1e293b", "h"),
    ]
    for k, lbl, col, mk in pareto_specs:
        if k not in by_key:
            continue
        r = by_key[k]
        sz = 140 if mk == "*" else 80
        axes[2].scatter(r["rp_rbf4k"], r["tier2_arr"] * 100.0, c=col, marker=mk, s=sz, edgecolors="black", lw=0.7, zorder=4)
        ox = -44 if lbl in ("E17-M2", "D3-EqP", "Oracle-E9") else 6
        oy = -11 if lbl in ("E17-M3", "D3 VidOp") else 4
        axes[2].annotate(lbl, (r["rp_rbf4k"], r["tier2_arr"] * 100.0), xytext=(ox, oy), textcoords="offset points", fontsize=7.8)
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Zero-Leakage Visual ICL RelP RBF@4k MSE10 (→ Better)")
    axes[2].set_ylabel("Oracle Tier-2 Parallel Arithmetic ARR % (↑ Better)")
    axes[2].set_title("(C) Pure-Video Pareto Frontier: Visual ICL vs Editability")
    axes[2].grid(alpha=0.25)

    fig.suptitle("E17 Exp 3: D3/D4 Complementarity — Spectral Geometry, Decoder Reliance & 2×2 Factorial Pareto", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_d3_d4_complementarity_geometry.png")
    plt.close(fig)


def _plot_fig5_reconciled_label_efficiency(rec: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))

    # Panel A: Protocol Discrepancy Audit (Pref512 vs Unif1024 vs FullNT5799)
    audit_keys = [
        ("e16_d0_rand_dec100", "D0"),
        ("e16_d1_lam_dec100", "D1"),
        ("e16_d3_vidop_dec100", "D3"),
        ("e16_d4_learnop_dec100", "D4"),
        ("e17_m1_joint_op_dec100", "M1"),
        ("e17_m2_compact_op_dec100", "M2"),
        ("e17_m4_compact_reuse_dec100", "M4"),
        ("e16_d_oracle_e9_dec100", "E9"),
    ]
    audit_keys = [(k, l) for k, l in audit_keys if k in rec]
    ax0x = np.arange(len(audit_keys))
    w = 0.26
    p512 = [rec[k]["last_ema"]["protoA_prefix512_27tasks"]["mse10"] for k, _ in audit_keys]
    u1024 = [rec[k]["last_ema"]["protoB_uniform1024_311tasks"]["mse10"] for k, _ in audit_keys]
    f5799 = [rec[k]["last_ema"]["protoC_full5799_nontail_311tasks"]["mse10"] for k, _ in audit_keys]
    axes[0].bar(ax0x - w, p512, w, label="Old E16 Prefix-512 (27 tasks+tails)", color="#ef4444", alpha=0.85)
    axes[0].bar(ax0x, u1024, w, label="Uniform-1024 (311 tasks)", color="#3b82f6", alpha=0.85)
    axes[0].bar(ax0x + w, f5799, w, label="Unified Full-5799 Non-Tail (311 tasks)", color="#059669", alpha=0.90)
    axes[0].set_xticks(ax0x)
    axes[0].set_xticklabels([l for _, l in audit_keys], fontsize=8.5)
    axes[0].set_ylabel("Frozen Decoder Action10 MSE (↓)")
    axes[0].set_title("(A) Reconciling E16's 2.7× Metric Discrepancy")
    axes[0].legend(frameon=False, fontsize=7.4)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B & C: Unified Label Efficiency (1%, 10%, 100%) on FullNT Fidelity & RelP RBF Visual ICL
    le_series = [
        ("D0 Random", ["e16_d0_rand_dec01", "e16_d0_rand_dec10", "e16_d0_rand_dec100"], "#94a3b8", "o--"),
        ("D1 Pure LAM", ["e16_d1_lam_dec01", "e16_d1_lam_dec10", "e16_d1_lam_dec100"], "#64748b", "s--"),
        ("D3 VidRel-Op", ["e16_d3_vidop_dec01", "e16_d3_vidop_dec10", "e16_d3_vidop_dec100"], "#0284c7", "^-"),
        ("D4 LearnReuse-Op", ["e16_d4_learnop_dec01", "e16_d4_learnop_dec10", "e16_d4_learnop_dec100"], "#7c3aed", "D-"),
        ("E17-M1 Joint", ["e17_m1_dec01", "e17_m1_dec10", "e17_m1_joint_op_dec100"], "#2563eb", "o-"),
        ("E17-M4 CompactReuse", ["e17_m4_dec01", "e17_m4_dec10", "e17_m4_compact_reuse_dec100"], "#ec4899", "*-"),
        ("Oracle-E9", ["e16_d_oracle_e9_dec01", "e16_d_oracle_e9_dec10", "e16_d_oracle_e9_dec100"], "#1e293b", "h:"),
    ]
    fracs = [1, 10, 100]
    for lbl, ks, col, sty in le_series:
        if not all(k in rec for k in ks):
            continue
        fids = [rec[k]["last_ema"]["protoC_full5799_nontail_311tasks"]["mse10"] for k in ks]
        icls = [rec[k]["best"]["protoD_zero_leakage_icl"]["rel_progress"]["rbf4k_mse10_mean"] for k in ks]
        axes[1].plot(fracs, fids, sty, color=col, lw=2.0, markersize=6, label=lbl)
        axes[2].plot(fracs, icls, sty, color=col, lw=2.0, markersize=6, label=lbl)

    for ax, ylabel, title in [
        (axes[1], "Full-5799 Non-Tail Fidelity MSE10 (↓)", "(B) Unified Action Label Efficiency: Fidelity (311 Tasks)"),
        (axes[2], "Zero-Leakage Visual ICL RelP RBF@4k (↓)", "(C) Unified Action Label Efficiency: Visual ICL (94 Unseen)"),
    ]:
        ax.set_xscale("log")
        ax.set_xticks(fracs)
        ax.set_xticklabels(["1%\n(140 eps)", "10%\n(1,244 eps)", "100%\n(13,995 eps)"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7.3, loc="upper right")
        ax.grid(alpha=0.25)

    fig.suptitle("E17 Exp 4a: Reconciled Frozen-Decoder Evaluation Protocol & 1%–100% Action Label Efficiency", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_reconciled_label_efficiency.png")
    plt.close(fig)


def _plot_fig6_scaling_and_coverage(sc: Dict[str, Any], rec: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))

    # Panel A: Step-Controlled Data Scale (10% -> 30% -> 100% at FIXED 16,092 steps!)
    pcts = [10, 30, 100]
    data_series = [
        ("D1 Pure LAM", ["sc_d1_10_16k", "sc_d1_30_16k", "sc_d1_100_16k"], "#64748b", "s--"),
        ("D3 VidRel-Op", ["sc_d3_10_16k", "sc_d3_30_16k", "sc_d3_100_16k"], "#0284c7", "^-"),
        ("D4 LearnReuse-Op", ["sc_d4_10_16k", "sc_d4_30_16k", "sc_d4_100_16k"], "#7c3aed", "D-"),
        ("E17-M1 Joint", ["sc_m1_10_16k", "sc_m1_30_16k", "sc_m1_100_16k"], "#059669", "o-"),
    ]
    for lbl, ks, col, sty in data_series:
        if not all(k in sc for k in ks):
            continue
        vals = [sc[k]["label_efficiency_probe"]["100pct"]["r2_full_mean"] for k in ks]
        axes[0].plot(pcts, vals, sty, color=col, lw=2.2, markersize=6.5, label=lbl)
    axes[0].set_xscale("log")
    axes[0].set_xticks(pcts)
    axes[0].set_xticklabels(["10%\n(122k)", "30%\n(427k)", "100%\n(1.37M)"])
    axes[0].set_xlabel("Unlabeled Video Scale (Fixed 16,092 Steps!)")
    axes[0].set_ylabel("Disjoint-Task Linear Probe R²(z→a7) (↑)")
    axes[0].set_title("(A) Step-Controlled Video Data Scaling (Fixed 16k Steps)")
    axes[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=7.8)
    axes[0].grid(alpha=0.25)

    # Panel B: Data-Controlled Step Scaling (8k -> 16k -> 32k at FIXED 100% Data): Rank-Inflation Reversal!
    steps = [8, 16, 32]
    step_series = [
        ("D1 Pure LAM (16k→32k)", [16, 32], ["sc_d1_100_16k", "e16_d1_32k"], "#64748b", "s--"),
        ("D3-EqParam (8→16→32k)", steps, ["e17_d3eq_8k", "e17_d3eq_16k", "e17_d3eq_32k"], "#0284c7", "^--"),
        ("D4 LearnOp (16k→32k)", [16, 32], ["sc_d4_100_16k", "e16_d4_32k"], "#7c3aed", "D--"),
        ("E17-M1 Uncompact", steps, ["e17_m1_8k", "sc_m1_100_16k", "e17_m1_32k"], "#2563eb", "o--"),
        ("E17-M2 Compact (Monotonic!)", steps, ["e17_m2_8k", "e17_m2_16k", "e17_m2_32k"], "#059669", "*-"),
    ]
    for lbl, xs, ks, col, sty in step_series:
        if not all(k in sc for k in ks):
            continue
        vals = [sc[k]["zero_leakage_latent_predictability"]["rel_progress"]["r2_z"] for k in ks]
        axes[1].plot(xs, vals, sty, color=col, lw=2.2, markersize=7, label=lbl)
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(["8k\n(1.5 ep)", "16k\n(3.0 ep)", "32k\n(6.0 ep)"])
    axes[1].set_xlabel("Pure-Video Pretraining Steps (Fixed 100% Data)")
    axes[1].set_ylabel("Zero-Leakage RelP Latent R²(z) (↑)")
    axes[1].set_title("(B) Step Scaling: Rank-Inflation Reversal vs Compact M2")
    axes[1].legend(frameon=False, loc="lower left", fontsize=7.4)
    axes[1].grid(alpha=0.25)

    # Panel C: Relation Coverage (Fixed 16k Steps, 100% Data): Pure Video vs Task Metadata
    cov_items = [
        ("cov_d4_sametask_16k", "cov_d4_sametask_16k_dec100", "SameTask\n(TaskMeta-Restr)", "#dc2626"),
        ("sc_d4_100_16k", "sc_d4_100_16k_dec100", "AntiScene\n(Pure Video)", "#7c3aed"),
        ("cov_d4_unmasked_16k", "cov_d4_unmasked_16k_dec100", "Unmasked\n(Pure Video)", "#059669"),
        ("cov_d4_strict_ct_16k", "cov_d4_strict_ct_16k_dec100", "StrictCT\n(TaskMeta-Assist)", "#d97706"),
    ]
    cx = np.arange(len(cov_items))
    op_ct = [sc[k]["audit_op"]["ct_bal_rho"] for k, _, _, _ in cov_items if k in sc]
    rp_r2z = [sc[k]["zero_leakage_latent_predictability"]["rel_progress"]["r2_z"] for k, _, _, _ in cov_items if k in sc]
    axes[2].bar(cx - 0.18, op_ct, 0.34, label="z_op Cross-Task Alignment ρ (↑)", color="#2563eb", alpha=0.88)
    axes[2].bar(cx + 0.18, rp_r2z, 0.34, label="Zero-Leakage RelP R²(z) (↑)", color="#10b981", alpha=0.88)
    # Annotate frozen-decoder RelP RBF on top if reconciled!
    for i, (_, dk, _, _) in enumerate(cov_items):
        if dk in rec and "protoD_zero_leakage_icl" in rec[dk]["best"]:
            rbf = rec[dk]["best"]["protoD_zero_leakage_icl"]["rel_progress"]["rbf4k_mse10_mean"]
            axes[2].text(i, max(op_ct[i], rp_r2z[i]) + 0.02, f"RBF={rbf:.4f}", ha="center", fontsize=7.6, fontweight="bold")
    axes[2].set_xticks(cx)
    axes[2].set_xticklabels([l for _, _, l, _ in cov_items], fontsize=7.8)
    axes[2].set_ylim(0, 0.58)
    axes[2].set_ylabel("Score (↑ Better)")
    axes[2].set_title("(C) Pure-Video vs Task-Meta Coverage (Fixed 16k Steps)")
    axes[2].legend(frameon=False, loc="upper left", fontsize=7.5)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("E17 Exp 4b: Controlled Data Scaling, Step Scaling & Pure-Video Relation Coverage", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_controlled_scaling_and_coverage.png")
    plt.close(fig)


def _plot_fig7_paired_forest(paired: Dict[str, Any], fig_dir: Path) -> None:
    if not paired:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    pairs = list(paired.keys())
    y = np.arange(len(pairs))

    for ax, mkey, title in [
        (axes[0], "fidelity_per_task", "(A) Fidelity Paired ΔMSE10 Across All 311 Tasks (95% CI)"),
        (axes[1], "icl_rel_progress_rbf4k_per_task", "(B) Zero-Leakage RelP RBF Paired ΔMSE10 Across Unseen Tasks (95% CI)"),
    ]:
        deltas = [paired[k][mkey]["delta_B_minus_A"] * 1000.0 for k in pairs]
        lo = [(paired[k][mkey]["delta_B_minus_A"] - paired[k][mkey]["ci95_delta"][0]) * 1000.0 for k in pairs]
        hi = [(paired[k][mkey]["ci95_delta"][1] - paired[k][mkey]["delta_B_minus_A"]) * 1000.0 for k in pairs]
        wins = [paired[k][mkey]["win_rate_B_over_A"] * 100.0 for k in pairs]
        cols = ["#059669" if (paired[k][mkey]["ci95_delta"][1] < 0) else ("#dc2626" if paired[k][mkey]["ci95_delta"][0] > 0 else "#64748b") for k in pairs]

        ax.axvline(0.0, color="black", ls="--", lw=1.0)
        for i in range(len(pairs)):
            ax.errorbar(deltas[i], y[i], xerr=[[lo[i]], [hi[i]]], fmt="o", color=cols[i], ecolor=cols[i], elinewidth=2.0, capsize=3.5, markersize=6)
            ax.text(max(deltas[i] + hi[i], 0.0) + 0.15, y[i], f"win {wins[i]:.0f}%", va="center", fontsize=7.6, color=cols[i])
        ax.set_yticks(y)
        ax.set_yticklabels([k.replace("_vs_", " vs ") for k in pairs], fontsize=8.0)
        ax.invert_yaxis()
        xlo, xhi = ax.get_xlim()
        ax.set_xlim(xlo, xhi + 0.20 * (xhi - xlo))
        ax.set_xlabel("Paired ΔMSE10 × 10⁻³ (B − A; Left of 0 = B Better)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("E17 Per-Task Paired Statistical Comparisons (10,000 Bootstrap Resamples, 95% CIs & Win Rates)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_paired_task_stats_forest.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report() -> None:
    fig_dir = OUTPUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    auth = _load_json(OUTPUT_ROOT / "exp1_authenticity" / "authenticity_summary.json")
    exp2 = _load_json(OUTPUT_ROOT / "exp2_emergent_factors" / "emergent_factors_summary.json")
    exp3 = _load_json(OUTPUT_ROOT / "exp3_geometry" / "geometry_summary.json")
    rec = _load_json(OUTPUT_ROOT / "exp4_reconcile" / "reconcile_summary.json")
    sc = _load_json(OUTPUT_ROOT / "scaling_coverage_encoder_eval.json")
    paired = _load_json(OUTPUT_ROOT / "paired_stats" / "paired_comparisons.json")

    eval_order = [
        ("e16_d0_rand_dec100", "E16-D0 (Random Frozen)", "baseline"),
        ("e16_d1_lam_dec100", "E16-D1 (Pure LAM)", "baseline"),
        ("e16_d2_randrel_dec100", "E16-D2 (LAM+RandRel)", "baseline"),
        ("e16_d3_vidop_dec100", "E16-D3 (VidRel-Op, 67.6M)", "e16"),
        ("e17_d3_eqparam_dec100", "E17-D3-EqParam (VidRel-Op, 84.9M)", "e17_ctrl"),
        ("e16_d4_learnop_dec100", "E16-D4 (LearnReuse-Op, 82.5M)", "e16"),
        ("e17_m1_joint_op_dec100", "E17-M1 (Joint VidRel+SharpReuse, 82.5M)", "e17"),
        ("e17_m2_compact_op_dec100", "E17-M2 (Compact Joint-Op, 80.1M)", "e17"),
        ("e17_m3_dual_orth_dec100", "E17-M3 (Dual-Orth z_kin+z_reuse, 79.7M)", "e17"),
        ("e17_m4_compact_reuse_dec100", "E17-M4 (Compact Sharp Reuse, 80.1M)", "e17"),
        ("e16_d_oracle_e9_dec100", "Oracle-E9 (Action-Supervised Frozen)", "oracle"),
    ]
    rows: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for k, lbl, grp in eval_order:
        d = _load_eval(k)
        if d is not None:
            r = _extract_eval_row(k, lbl, grp, d)
            rows.append(r)
            by_key[k] = r
    _write_csv(rows, OUTPUT_ROOT / "e17_master_comparison.csv")

    # Export CSVs for authenticity, reconciled label efficiency, scaling/coverage, paired stats
    auth_rows = []
    for k, r in auth.items():
        ct = r["controls_by_pool"]["strict_cross_task"]
        ac = r["controls_by_pool"]["anti_scene_cross_task"]
        gt = r["graph_topology"]
        ps = r["predictor_sensitivity"]
        cf = r["counterfactual_interventions"]["anti_scene_opposite_act"]
        acf = cf.get("action_counterfactual", {})
        qs = r["scene_quartile_stratification"]
        auth_rows.append({
            "model": k,
            "regime": r["regime_tag"],
            "self_dv_r2": ct["self_oracle"]["dv_r2"],
            "vo_mlp_dv_r2": ct["vision_only_mlp"]["dv_r2"],
            "zero_dv_r2": ct["zero_donor"]["dv_r2"],
            "mean_dv_r2": ct["mean_donor"]["dv_r2"],
            "rand_dv_r2": ct["random_donor"]["dv_r2"],
            "wrong_dv_r2": ct["wrong_donor_opp"]["dv_r2"],
            "znn_dv_r2": ct["nn_z_cosine"]["dv_r2"],
            "w_soft_dv_r2": ct["learned_W_soft"]["dv_r2"],
            "w_top1_dv_r2": ct["learned_W_top1"]["dv_r2"],
            "antisc_w_soft_dv_r2": ac["learned_W_soft"]["dv_r2"],
            "znn_act_mse10": ct["nn_z_cosine"]["donor_act_mse10"],
            "w_soft_act_mse10": ct["learned_W_soft"]["donor_act_mse10"],
            "w_top1_act_mse10": ct["learned_W_top1"]["donor_act_mse10"],
            "z_to_v0_sens": ps["z_to_v0_sensitivity_ratio"],
            "col_gini": gt["col_hub_gini"],
            "q1_red_pct": qs["Q1"]["rel_act_err_reduction"] * 100.0,
            "q4_red_pct": qs["Q4"]["rel_act_err_reduction"] * 100.0,
            "cf_latent_margin": cf["latent_selectivity_margin"],
            "cf_act_adopt_pct": acf.get("action_adoption_ratio", float("nan")) * 100.0,
            "cf_grip_flip_pct": acf.get("grip_flip_adoption_acc", float("nan")) * 100.0,
        })
    _write_csv(auth_rows, OUTPUT_ROOT / "e17_authenticity_table.csv")

    rec_rows = []
    for k, r in rec.items():
        l = r.get("last_ema", {})
        b = r.get("best", l)
        icl_rp = b.get("protoD_zero_leakage_icl", {}).get("rel_progress", {})
        icl_vk = b.get("protoD_zero_leakage_icl", {}).get("visual_knn", {})
        rec_rows.append({
            "run_name": k,
            "action_fraction": r.get("action_fraction"),
            "total_steps": r.get("total_steps"),
            "best_step": r.get("best_step"),
            "pref512_last": l.get("protoA_prefix512_27tasks", {}).get("mse10"),
            "unif1024_last": l.get("protoB_uniform1024_311tasks", {}).get("mse10"),
            "full5799_last": l.get("protoC_full5799_nontail_311tasks", {}).get("mse10"),
            "full5799_best": b.get("protoC_full5799_nontail_311tasks", {}).get("mse10"),
            "relp_lin4k": icl_rp.get("lin4k_mse10_mean"),
            "relp_rbf4k": icl_rp.get("rbf4k_mse10_mean"),
            "vknn_rbf4k": icl_vk.get("rbf4k_mse10_mean"),
        })
    _write_csv(rec_rows, OUTPUT_ROOT / "e17_reconciled_label_efficiency.csv")

    sc_rows = []
    for k, r in sc.items():
        lp = r["label_efficiency_probe"]
        af = r["audit_full"]
        ao = r["audit_op"]
        rp = r["zero_leakage_latent_predictability"]["rel_progress"]
        vk = r["zero_leakage_latent_predictability"]["visual_knn"]
        sc_rows.append({
            "name": k,
            "variant": r["config"].get("variant"),
            "data_frac": r["config"].get("train_episode_fraction"),
            "step": r["config"].get("step"),
            "coverage": r["config"].get("coverage_mode"),
            "uses_task_meta": r["config"].get("uses_task_metadata"),
            "probe_1pct": lp["1pct"]["r2_full_mean"],
            "probe_10pct": lp["10pct"]["r2_full_mean"],
            "probe_100pct": lp["100pct"]["r2_full_mean"],
            "op_ct_bal": ao["ct_bal_rho"],
            "op_disp_rho": ao["disp_rho"],
            "ct_top1_mse10": af["ct_top1_mse10"],
            "erank_full": af["erank"],
            "erank_op": ao["erank"],
            "fut16_r2": r["beyond_e12"]["fut16_r2"],
            "relp_r2_z": rp["r2_z"],
            "relp_e_lat": rp["e_lat"],
            "vknn_r2_z": vk["r2_z"],
        })
    _write_csv(sc_rows, OUTPUT_ROOT / "e17_scaling_coverage_table.csv")

    p_rows = []
    for k, comp in paired.items():
        for mk in ("fidelity_per_task", "icl_rel_progress_rbf4k_per_task", "icl_visual_knn_rbf4k_per_task", "edit_op_arith_mse10"):
            if mk not in comp:
                continue
            s = comp[mk]
            p_rows.append({
                "comparison": k,
                "metric": mk,
                "n_pairs": s["n_pairs"],
                "mean_A": s["mean_A"],
                "mean_B": s["mean_B"],
                "delta_B_minus_A": s["delta_B_minus_A"],
                "rel_change_pct": s["rel_change_pct"],
                "ci95_low": s["ci95_delta"][0],
                "ci95_high": s["ci95_delta"][1],
                "win_rate": s["win_rate_B_over_A"],
                "wilcoxon_p": s["wilcoxon_p"],
                "cohens_dz": s["cohens_dz"],
                "sig_95ci": s["significant_95ci"],
            })
    _write_csv(p_rows, OUTPUT_ROOT / "e17_paired_stats_table.csv")

    # Generate all 7 figures
    _plot_fig1_authenticity(auth, fig_dir)
    _plot_fig2_relation_graphs(auth, fig_dir)
    _plot_fig3_emergent_factors(exp2, fig_dir)
    _plot_fig4_d3_d4_complementarity(exp3, by_key, fig_dir)
    _plot_fig5_reconciled_label_efficiency(rec, fig_dir)
    _plot_fig6_scaling_and_coverage(sc, rec, fig_dir)
    _plot_fig7_paired_forest(paired, fig_dir)

    # Assemble rich REPORT.md
    lines: List[str] = []
    def add(s: str = "") -> None:
        lines.append(s)

    add("# E17: Discovering and Validating Reusable Behavior Structure — 完整实验报告")
    add()
    add("> **核心研究问题**：在完全不使用 GT Action 关系监督与动作预训练初始化的前提下，纯视频自监督（`E16-D3/D4` 及其演进模型）**究竟学到了什么行为关系**？这些关系是**真正可跨任务复用的动力学结构，还是静态场景捷径**？模型在潜空间中**自发组织出了哪些行为因子**？以及**如何基于机理诊断进一步改善行为表征的几何与可复用性**？")
    add()
    add("> **严格实验纪律（全量数据 + 等参等预算 + 体制与口径零混淆）**：")
    add("> - **纯视频预训练主线**：所有新模型（`E17-M1/M2/M3/M4`、`E17-D3-EqParam` 及全部 Scaling/Coverage 编码器）均从随机初始化 `LocalBehaviorEncoder`（`42.55M` 参数）出发，在全量 15,550 条 LIBERO 轨迹（311 任务、1,373,035 训练视频片段）上以全预算（`32,184 steps = 6.0 epochs`）纯视频预训练，代码中每步硬断言 `actions_10d is None`。")
    add("> - **严格区分四类边界**：(1) **Oracle 诊断**（自重建 Fidelity、Tier-2 平行算术、纯 $z_{\\text{op}}$ 编辑、因果 Token/子空间替换、反事实干预） vs **严格零泄漏迁移**（94 个留出未见任务、3 种子 `42/123/777`、`visual_knn` 与 `rel_progress` Visual ICL）；(2) **纯视频监督**（仅用 `ep_ids` + 双相机 RGB/光流，零任务元数据） vs **任务元数据辅助/受限对照**（`strict_cross_task` / `same_task_only`）；(3) **等参等预算对照**（`E17-D3-EqParam: 84.88M` 严格对齐 `M1..M4: 79.7–82.5M` 与 `D4: 82.54M`）；(4) **统一冻结解码器评测口径**（核实并消除 E16 的 `2.7×` 评估子集偏差）。")
    add()
    add("---")
    add()
    add("## 执行摘要：回答三大终极科学问题")
    add()
    add("### 问题一：模型学到了什么？关系图是真实复用还是场景捷径？")
    add("1. **严格的 11 项对照审计（Exp 1）彻底区分了「真实跨任务复用」与「两类隐藏退化」**：")
    add("   - **退化 #1 — `same_task_only` 是彻头彻尾的场景捷径**：无论是 Regime 1 (`e16_cov_sametask`) 还是纯视频 Regime 2 (`cov_d4_sametask_16k`)，受限同任务覆盖使前向预测器 $F_\\psi(v_0, z)$ 退化为静态场景查找表——输入与场景敏度比 $z/v_0$ 崩塌至 `0.21–0.27`（跨 Demo 模型为 `0.87–0.94`），即使注入**随机 Donor 或完全相反动作的 Wrong Donor**，前向 $\\Delta v$ $R^2$ 仍高达 `0.50–0.55`（几乎与 Self `0.57–0.68` 无差），且潜空间跨任务对齐 `Op_CTbal` 崩塌至 `0.049`。")
    add("   - **退化 #2 — `E16-D4 (cross_demo_anti_scene)` 的「凸包软平均作弊（Convex-Hull Averaging）」**：这是 E17 最具颠覆性的审计发现！E16 中 `E16-D4` 虽然取得了 `W_soft dv_R2 = 0.676`，但其**单个 Top-1 Donor 预测反而崩塌至 `W_top1 dv_R2 = -0.136`、`ActMSE10 = 0.1831`**，在视觉最不相似的 75% 场景分位（`Q1–Q3`）中甚至**劣于随机 Donor（`-8% / -15% / -13%`）**！根因在于 `anti_scene` 屏蔽了姿态相近的跨任务 Donor，迫使高熵软注意力 $W_{ij}$ 用 ~200 个 Donor 的凸组合线性拼出 $z_i$，而非找到真正可单独复用的 Donor。与此同时，同一 `E16-D4` 编码器自身的原始潜空间近邻 `z-NN` 动作误差仅为 `0.0218`（比 `Flow-NN 0.0279` 与 `v0-NN 0.0378` 更准），且反事实跨场景干预的冻结解码器动作采纳率高达 **`98.8%`（夹爪翻转 `99.6%`）**——说明 `E16-D4` 的潜空间本身确实学到了真实行为动力学，只是其软注意力图 $W_{ij}$ 存在凸包平均伪影。")
    add("   - **彻底修复后的纯视频真复用（`E17-M1 / E17-M2`）**：用纯视频 `cross_demo_unmasked`（零任务元数据）+ 稀疏 `Top-16` 与直通 `Top-1` 单 Donor 复用损失替换 `E16-D4` 后，**`E17-M2` 将单 Donor `W_top1 dv_R2` 从 `-0.136` 飙升至 `+0.650`（`+0.786 R²`！远超 `Mean/Rand/Wrong ≈ 0.27–0.28`）**、`W_top1 ActMSE10` 降 **`3.6×`** 至 **`0.0510`**，且在**视觉最不相似的底部 25% 场景（`Q1`）中实现 `+30%` 动作降错（`Q1..Q4: +30% / +37% / +50% / +74%`）**、反事实动作采纳率达 **`99.0%`（夹爪翻转 `98.4%`）**！")
    add()
    add("### 问题二：模型自发组织出了什么行为因子？为什么能够复用？")
    add("2. **不预设 Token 语义的自主因子发现（Exp 2）揭示「分布式子空间因子 + 多时间尺度结构」，而非轴对齐 Token**：")
    add("   - 即使使用**最优匈牙利匹配（Hungarian Assignment）**解除 `Tok0..3` 的人工绑定，所有纯视频模型（`D1/D3/D4/M1..M4`）的 4 个 Token 仍高度共享（`Inter-Tok CKA = 0.84–1.00`），单 Token 因果采纳率仅为 `11%–16%`（对比 GT 因子监督的 `E12-BC-FD 59.4%`）；即便 Regime 1 `E16-C3` 用显式 CKA 去相关把 Token CKA 压到 `0.137`，其 Token 仍对应混合阶段而非物理自由度（Hungarian 仅 `14.4%`）。")
    add("   - **但无监督子空间与算术编辑真实存在且强可组合**：在无监督 ICA 发现的 12 个独立轴上，纯视频 `E16-D4` 仅替换 3 条无监督轴即可在**未见 `(dir × grip)` 组合**上达成 **`21.9%`** 联合成功率（为 Pure LAM `2.1%` 的 **`10.4×`**），而纯 $z_{\\text{op}}$ 算术达 **`39.6%`**（`E17-M3` 达 **`34.9%`**，大幅领先 Pure LAM `17.2%` 与 `D3 18.2%`）。")
    add("   - **自发涌现「快运动学 vs 慢长时序意图」时间尺度分层与物理子目标边界检测**：沿 48 条完整验证轨迹（`4,584` 个连续步）追踪无监督 ICA 因子，自相关半衰期分离比从 Pure LAM 的 `1.6×` 扩大至 `D4` 的 **`2.2×`（`10.5–22.5` 帧，最慢因子全部对应 `fut16` 长时序意图）**；且潜空间速度突变与真实**抓取/释放/方向反转子目标边界**的 ROC-AUC 从 Pure LAM `0.591` → `D3 0.617` → `M1 0.692` → `M3 0.729` → **`E17-M4 0.770`（以 100% 纯视频超越有动作预训练的 `Oracle-E9 0.757` 与 `E12-BC-FD 0.762`！）**。")
    add()
    add("### 问题三：如何改善行为表征？D3 与 D4 能否真正互补融合？规模与覆盖度如何作用？")
    add("3. **Exp 3 机理诊断 + 2×2 因子实验揭示「2D 像素光流几何 vs 3D 复用算术」的本质张力，并严格扩展双侧 Pareto 前沿**：")
    add("   - **为何 `D4` 纯 $z_{\\text{op}}$ 编辑达 `64.9%` 而 `D3` 仅 `31.0%`？** Exp 3 因果切片替换直接给出了物理答案：在 `D4` 中冻结解码器 **`91.3%`** 依赖 $z_{\\text{op}}$（仅 `12.1%` 依赖 $z_{\\text{res}}$）；而在 `D3` 中光流弱于夹爪（`R²(op→grip)=0.762 < R²(res→grip)=0.810`），迫使解码器 **`39.4%`** 依赖 $z_{\\text{res}}$！反过来，E16 在全 3072D 空间施加的**各向同性 VICReg**将 `D4` 的有效秩吹胀至 `279.4`（`z_res` 秩 `343.1`），导致 Visual ICL 预测误差飙升。")
    add("   - **等参对照（`E17-D3-EqParam 84.9M`）严格排除参数量红利**：其 6 轴表现与 `E16-D3 (67.6M)` 几乎逐项重合（Fidelity `0.00224 vs 0.00221`, `RelP RBF 0.01622 vs 0.0162`）。")
    add("   - **2×2 因子实验（`{VidRel ON/OFF} × {Compact Manifold ON/OFF}`）诚实拒斥「简单融合全面通吃」，同时精准突破两条前沿**：")
    add("     - **朴素融合的真实局限**：直接把 `VidRel` 加到 `z_op`（`M1 / M2`）虽创下纯视频 **1,024-样本 Fidelity (`0.00200`, 311 任务全量与 `D3` 统计持平 `p=0.123` 且在 RelP RBF 上以 `85.9%` 胜率显著优于 `D4`, `p=7.5e-19`)**、**`UnseenRidge (0.01075)`**、**`R²(z→a7) (0.916)`** 与 **`vKNN RBF (0.01893)`** 新纪录，却因 2D 相机平面光流的透视畸变把 Tier-2/纯 $z_{\\text{op}}$ 3D 动作算术压回 `40.8% / 22.7%`。")
    add("     - **紧凑流形塑形（消除 3072D VICReg 秩爆炸）的决定性收益**：将 `M1 → M2 / M3 / M4` 切换到紧凑 256D LAM 瓶颈 + $z_{\\text{res}}$ 能量正则后，冻结解码器 $z_{\\text{op}}$ 主导度飙升至 **`93.4%`（M2）/ `94.6%`（M3）**、$z_{\\text{res}}$ 的 ICL 误差砍半（`-53%~-59%`）、在配对检验中 `M2/M3` 在 RelP RBF 上与 `D3` 统计持平的同时显著超越 `D3` 的纯 $z_{\\text{op}}$ 编辑性（胜率 `62.5%/60.3%`, `p=5.1e-3/2.0e-2`），且 **`E17-M3` 创下纯视频 `vKNN RBF = 0.01876` 与 `RelP R²(z) = 0.604`（`e_lat = 0.620`, `erank = 38.7`）全纪录**；而关闭 2D 光流畸变的 **`E17-M4 (Compact Sharp Reuse)`** 在保持纯视频最高 **Tier-2 编辑（`66.7% > D4 63.0%`）** 与强纯 $z_{\\text{op}}$ 编辑（`51.4% > E9 40.8%`）的同时，在留出未见任务配对检验上以 **`77.7%` 胜率（`p=6.1e-11`）** 将 `D4` 的零泄漏 Visual ICL 误差全面暴降 `-11%~-14%`（`RelP RBF: 0.0196 → 0.0175`，1% 极稀缺动作标注下更创纯视频最低 `0.0189`！）。")
    add("4. **统一评测口径与严格控步 Scaling / Coverage 规律（Exp 4）**：")
    add("   - **核实 E16 `2.7×` 口径差异**：E16 4.1 节的 `metrics.json` 仅评估了验证集前 512 个顺序样本（仅覆盖 **27/311 任务**且含 61 个 tail 样本），导致 MSE10 比全量 311 任务 5,799 个非 tail 样本系统性虚高 `2.7×–2.9×`；在全量 311 任务统一口径下，**`E16-D4` 的 Fidelity 实际上在 `1% / 10% / 100%` 全档位优于 `E16-D3`（`0.00505/0.00252/0.00194` vs `0.00546/0.00294/0.00231`）**，而 **`E16-D3` 与 `E17-M1/M4` 领跑零泄漏 Visual ICL**（`D3` 仅用 10% 动作标注即达 `RelP RBF 0.0164`，追平 100% 标注 `0.0162`；`M4` 仅用 1% 标注即达 `0.0189`，击败 100% 标注 Pure LAM `0.0219`！）。")
    add("   - **严格控步（固定 `16,092` 步）揭示纯视频真正的数据缩放律与「秩膨胀反转」**：无关系的 `Pure LAM (D1)` 扩大 10× 视频几乎停滞（`Probe 100%: 0.866 → 0.870`），而全部关系方法（`D3/D4/M1`）均严格单调积累可复用知识；更惊人的是，固定 100% 数据延长训练步数（`8k → 16k → 32k`）时，未控秩的 `D1/D3/D4/M1` 因 3072D VICReg 持续吹胀 `erank` 导致 `RelP R²(z)` 反而**倒退 `4.6–6.8 pp`**，唯独紧凑流形 **`E17-M2` 随训练步数严格单调提升**（`RelP R²(z): 0.522 → 0.536`, `vKNN R²(z): 0.537 → 0.555`, `Op_CTbal = 0.507`, `Top1MSE = 0.0191`）！")
    add("   - **纯视频覆盖度完全无需任务元数据**：固定 `16,092` 步下，零任务元数据的纯视频 `cross_demo_unmasked` 与任务元数据辅助的 `strict_cross_task` 在全部编码器指标上**精确重合到小数点后三位**（`Probe 0.608/0.837/0.911 vs 0.607/0.836/0.911`, `Op_CTbal 0.402 vs 0.403`, `RelP R²(z) 0.466 vs 0.466`），在冻结解码器配对检验上纯视频 `unmasked` 甚至小幅优于 `strict_cross_task`（`RelP RBF: 0.0191 vs 0.0195`）；而 `same_task_only` 全面崩盘，在纯 $z_{\\text{op}}$ 编辑性上被 `unmasked` 以 **`95.6%` 胜率（`Δ = -0.09725`, `p = 1.0e-21`）** 碾压！")
    add()
    add("---")
    add()

    # Section 1: Relation Authenticity
    add("## 1. Pillar 1 — Relation Authenticity：关系图审计、严格对照与反事实干预")
    add()
    add("![Relation Authenticity Controls](figures/fig1_relation_authenticity_controls.png)")
    add()
    add("![Learned Relation Graph Visualizations](figures/fig2_relation_graph_visualizations.png)")
    add()
    add("在 1,024 条严格留出未见任务验证样本（与 Vision-Only 训练集完全任务隔离）上，我们在**严格跨任务池（`task_i != task_j`）**与**极端反场景池（`cos(v0_i, v0_j) <= median`）**上评估全部 11 项对照与反事实干预。表中严格区分纯视频监督（`PureVideo`）与任务元数据辅助/受限（`TaskMeta`）：")
    add()
    add("| 模型 | 体制 / 元数据 | Self / VO-MLP | Zero / Mean | Rand / Wrong | `z-NN` (`ActMSE`) | **`W_soft` / `W_top1` $\\Delta v\\,R^2$** | **`W_top1` ActMSE (↓)** | `Q1..Q4` 降错率 | $z/v_0$ 敏度 | **反事实动作采纳 / 夹爪翻转** |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    auth_order = [
        ("e16_d4_learn_op", "E16-D4 (AntiScene)", "PureVideo (R2)"),
        ("e17_m1_joint_op", "**E17-M1 (Sharp Joint)**", "PureVideo (R2)"),
        ("e17_m2_compact_op", "**E17-M2 (Sharp Compact)**", "PureVideo (R2)"),
        ("e17_m3_dual_orth", "E17-M3 (Sharp DualOrth)", "PureVideo (R2)"),
        ("e17_m4_compact_reuse", "E17-M4 (Sharp Reuse)", "PureVideo (R2)"),
        ("cov_d4_sametask_16k", "Cov-SameTask-16k", "TaskMeta-Restr (R2)"),
        ("sc_d4_100_16k", "Cov-AntiScene-16k", "PureVideo (R2)"),
        ("cov_d4_unmasked_16k", "**Cov-Unmasked-16k**", "PureVideo (R2)"),
        ("cov_d4_strict_ct_16k", "Cov-StrictCT-16k", "TaskMeta-Assist (R2)"),
        ("e16_c2_op_learnable", "E16-C2 (Op-LearnReuse)", "PureVideo-Rel (R1)"),
        ("e16_cov_unmasked", "E16-Cov-Unmasked", "PureVideo-Rel (R1)"),
        ("e16_cov_strict_ct", "E16-Cov-StrictCT", "TaskMeta-Assist (R1)"),
        ("e16_cov_sametask", "E16-Cov-SameTask", "TaskMeta-Restr (R1)"),
    ]
    for mk, lbl, reg in auth_order:
        if mk not in auth:
            continue
        r = auth[mk]
        ct = r["controls_by_pool"]["strict_cross_task"]
        ps = r["predictor_sensitivity"]
        qs = r["scene_quartile_stratification"]
        cf = r["counterfactual_interventions"]["anti_scene_opposite_act"]
        acf = cf.get("action_counterfactual")
        q_str = "/".join(f"{qs[f'Q{i+1}']['rel_act_err_reduction']*100:+.0f}%" for i in range(4))
        cf_str = f"`{acf['action_adoption_ratio']*100:.1f}%` / `{acf['grip_flip_adoption_acc']*100:.1f}%`" if acf else f"Lat `+{cf['latent_selectivity_margin']:.3f}`"
        add(
            f"| {lbl} | {reg} | `{ct['self_oracle']['dv_r2']:.3f}` / `{ct['vision_only_mlp']['dv_r2']:.3f}` | "
            f"`{ct['zero_donor']['dv_r2']:.2f}` / `{ct['mean_donor']['dv_r2']:.2f}` | "
            f"`{ct['random_donor']['dv_r2']:.2f}` / `{ct['wrong_donor_opp']['dv_r2']:.2f}` | "
            f"`{ct['nn_z_cosine']['dv_r2']:.3f}` (`{ct['nn_z_cosine']['donor_act_mse10']:.4f}`) | "
            f"`{ct['learned_W_soft']['dv_r2']:.3f}` / **`{ct['learned_W_top1']['dv_r2']:.3f}`** | "
            f"`{ct['learned_W_top1']['donor_act_mse10']:.4f}` | `{q_str}` | `{ps['z_to_v0_sensitivity_ratio']:.2f}` | {cf_str} |"
        )
    add()
    add("**核心审计结论**：")
    add("- **对照基准**：随机跨任务 Donor 动作误差 `0.2471`，纯静态场景 `v0-NN = 0.0378`，纯像素光流 `Flow-NN = 0.0279`，Oracle GT 动作近邻 `ActOra-NN = 0.0038`；Vision-Only Ridge/MLP 仅凭 $v_0$ 预测 $\\Delta v$ 的 $R^2$ 为 `0.388 / 0.294`。")
    add("- **彻底坐实真复用而非场景捷径**：在所有跨 Demo 模型中，随机或相反动作 Donor 会使 $\\Delta v\\,R^2$ 骤降（`E16-D4: -0.84`、`E17-M2: 0.27` vs `W_top1: 0.650`）；将极端反场景、相反动作的 Donor $z_j$ 注入场景 $v_{0,i}$ 生成反事实视频并重编码解码，冻结动作解码器的动作采纳率达 **`98.0%–99.8%`**、夹爪翻转准确率达 **`97.4%–100.0%`**！")
    add("- **Sharp Top-16/Top-1 彻底根治 `E16-D4` 的凸包平均作弊**：`E17-M1/M2` 的单 Donor `W_top1` 在底部 25% 最不相似场景 `Q1` 中依然将动作误差降低 **`+29%~+30%`**（而 `E16-D4` 在 `Q1–Q3` 全为负增益 `-8%/-15%/-13%`），且图 2 清晰展示了 `E17-M2` 与 `Cov-Unmasked` 的 $W_{ij}$ 自发浮现出与 Oracle 动作相似度高度一致的块对角结构（青色虚线为夹爪开合边界）。注：由于 `E17-M1..M4` 使用稀疏 `top_k_donors=16`（98.4% 权重严格为零），全矩阵 Spearman 偏相关会因零值并列被稀释，因此应以 `W_top1 ActMSE10`、`W_top1 dv_R2` 及 `Q1..Q4` 分位降错率作为稀疏图的主审计口径。")
    add()
    add("---")
    add()

    # Section 2: Emergent Behavior Factors
    add("## 2. Pillar 2 — Emergent Behavior Factors：无预设行为因子发现、独立性、组合性与时序分析")
    add()
    add("![Emergent Behavior Factors](figures/fig3_emergent_behavior_factors.png)")
    add()
    add("我们放弃「预设 Token 0/1/2/3 对应平移/旋转/幅值/夹爪」的人为假设，对 4 个 Token 做全对（`4×5`）因果替换 + 最优匈牙利匹配，在 $z_{\\text{op}}$ 上做纯无监督 PCA/FastICA 独立轴发现与因果子空间干预，并沿 48 条完整验证轨迹（4,584 连续步）做多时间尺度自相关与子目标边界检测：")
    add()
    add("| 模型 | Token CKA (`op`) | 最优匈牙利 TokSwap % (`固定E12`) | TokSwap 选择性 | 无监督 `ICA-Top3` 替换 % (`DCI_D`) | **未见 `(Dir×Grip)` 组合: Tok / ICA / `z_op`算术** | ICA 半衰期 (帧, 比值) | **子目标突变 AUC (`z_op` / `full`)** |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    exp2_order = [
        ("e16_d1_lam", "E16-D1 (Pure LAM)"),
        ("e16_d3_vid_op", "E16-D3 (VidRel-Op)"),
        ("e17_d3_eqparam", "E17-D3-EqParam"),
        ("e16_d4_learn_op", "E16-D4 (LearnReuse-Op)"),
        ("e17_m1_joint_op", "E17-M1 (Joint-Op)"),
        ("e17_m2_compact_op", "E17-M2 (Compact Joint)"),
        ("e17_m3_dual_orth", "E17-M3 (Dual-Orth)"),
        ("e17_m4_compact_reuse", "**E17-M4 (Compact Reuse)**"),
        ("e16_c2_op_learnable", "E16-C2 (Regime1 Op-Learn)"),
        ("e16_c3_op_learnable_tok", "E16-C3 (Regime1 Tok-Learn)"),
        ("e16_d_oracle_e9", "Oracle-E9 (Action Frozen)"),
        ("e12_bc_factordisp", "E12-BC-FD (GT Factor Ref)"),
    ]
    for mk, lbl in exp2_order:
        if mk not in exp2:
            continue
        r = exp2[mk]
        hf = r["causal_token_swap_full"]
        isi = r["unsupervised_subspace_interventions"]["ica_top3"]
        ica = r["unsupervised_ica_factors"]
        cg = r["compositional_generalization_unseen_dir_grip"]
        td = r["temporal_dynamics"]
        c_t = cg.get("optimal_token_swap", {}).get("joint_compositional_success", 0.0) * 100.0
        c_i = cg.get("ica_subspace_swap", {}).get("joint_compositional_success", 0.0) * 100.0
        c_o = cg.get("op_subspace_arith", {}).get("joint_compositional_success", 0.0) * 100.0
        add(
            f"| {lbl} | `{r['inter_token_cka_op_mean']:.3f}` | "
            f"`{hf['hungarian_mean_adoption']*100:.1f}%` (`{hf['fixed_e12_mean_adoption']*100:.1f}%`) | "
            f"`{hf['hungarian_selectivity']:.2f}` | "
            f"`{isi['mean_adoption']*100:.1f}%` (`{ica['dci_disentanglement_beh']:.3f}`) | "
            f"`{c_t:.1f}%` / `{c_i:.1f}%` / **`{c_o:.1f}%`** | "
            f"`{td['ica_half_life_min']:.1f}–{td['ica_half_life_max']:.1f}` (`{td['ica_half_life_ratio_max_min']:.1f}×`) | "
            f"**`{td['changepoint_subgoal_auc_op']:.3f}`** / `{td['changepoint_subgoal_auc_full']:.3f}` |"
        )
    add()
    add("**核心发现（模型到底自发发现了什么因子）**：")
    add("1. **纯视频预训练不会自发产生轴对齐的「Token 槽位解耦」**：即便经最优匈牙利重排，纯视频模型（甚至 `Oracle-E9`）的 4 个 Token CKA 均高达 `0.84–1.00`，Token 级替换采纳率仅 `11%–17%`（选择性 `0.47–0.63`，远低于 GT 因子监督 `E12-BC-FD` 的 `59.4% / 1.98`）。说明 Transformer 的 4 个无约束查询 Token 默认扮演多头分布式冗余编码器，而非人工自由度槽位。")
    add("2. **真正的行为因子存在于「旋转后的低维连续子空间」而非单 Token 中**：")
    add("   - 在无监督 FastICA 提取的 12 条独立轴中，替换 3 条轴即可在 `E17-M2` 上实现 **`19.4%`** 跨属性因果采纳（超过 `E12-BC-FD` 的 ICA 替换 `17.3%`），在 `E16-D4` 上实现 **`21.9%`** 的未见 `(Dir×Grip)` 零样本组合（`10.4×` 于 Pure LAM）；而纯 $z_{\\text{op}}$ 向量算术更在 `E16-D4` (`39.6%`) 与 `E17-M3` (`34.9%`) 上实现强组合泛化。")
    add("3. **纯视频自监督在「事件/子目标时序结构」上超越动作监督**：沿着连续轨迹，模型自发分化出**快变化运动学因子（半衰期 ~6–10 帧，对应 `temporal/trans`）与慢变化长时序意图因子（半衰期 ~20–23 帧，对应 `fut16/grip_state`）**；更令人震撼的是，**`E17-M4 (Compact Sharp Reuse)` 的 $z_{\\text{op}}$ 潜空间速度对真实抓取/释放/折返子目标事件的检出 AUC 达到 `0.770`**，以零动作监督超越了有动作预训练的 `Oracle-E9 (0.757)` 和 `E12-BC-FD (0.762)`！")
    add()
    add("---")
    add()

    # Section 3: D3/D4 Complementarity
    add("## 3. Pillar 3 — D3/D4 Complementarity：机理诊断、等参对照与 2×2 互补训练")
    add()
    add("![D3/D4 Complementarity and Geometry](figures/fig4_d3_d4_complementarity_geometry.png)")
    add()
    add("### 3.1 Exp 3 机理诊断：潜空间几何、信息分布与冻结解码器敏感性")
    add()
    add("| 模型 | `erank` (`full/op/res`) | `CKA(op,res)` | `R²(op/res→a7)` | `R²(op/res→grip)` | **冻结解码器切片替换采纳 (`op / res`)** | Jacobian 敏度比 (`op/res`) | **RelP ICL `R²(op/res)`** | **RelP 误差归因 `Exc(op/res)`** |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    exp3_order = [
        ("e16_d1_lam", "E16-D1 (Pure LAM)"),
        ("e16_d2_randrel", "E16-D2 (RandRel)"),
        ("e16_d3_vid_op", "E16-D3 (VidRel-Op)"),
        ("e17_d3_eqparam", "E17-D3-EqParam (84.9M)"),
        ("e16_d4_learn_op", "E16-D4 (LearnReuse-Op)"),
        ("e17_m1_joint_op", "E17-M1 (Joint-Op)"),
        ("e17_m2_compact_op", "**E17-M2 (Compact Joint)**"),
        ("e17_m3_dual_orth", "**E17-M3 (Dual-Orth)**"),
        ("e17_m4_compact_reuse", "E17-M4 (Compact Reuse)"),
        ("e16_d_oracle_e9", "Oracle-E9 (Action Frozen)"),
    ]
    for mk, lbl in exp3_order:
        if mk not in exp3:
            continue
        r = exp3[mk]
        sp = r["spectrum"]
        inf = r["info_distribution"]
        ds = r["decoder_sensitivity"]
        rp = r["icl_subspace_attribution"]["rel_progress"]
        add(
            f"| {lbl} | `{sp['full']['erank']:.1f}/{sp['op']['erank']:.1f}/{sp['res']['erank']:.1f}` | "
            f"`{r['subspace_coupling']['cka_op_res']:.3f}` | "
            f"`{inf['op']['r2_a7_all']:.3f}/{inf['res']['r2_a7_all']:.3f}` | "
            f"`{inf['op']['r2_grip_1d']:.3f}/{inf['res']['r2_grip_1d']:.3f}` | "
            f"**`{ds['swap_op_from_donor']['donor_adoption_ratio']*100:.1f}%`** / `{ds['swap_res_from_donor']['donor_adoption_ratio']*100:.1f}%` | "
            f"`{ds['op_vs_res_jacobian_ratio_025']:.2f}` | "
            f"`{rp['r2_z_op']:.3f}/{rp['r2_z_res']:.3f}` | "
            f"`{rp['excess_from_op_err']:.5f}/{rp['excess_from_res_err']:.5f}` |"
        )
    add()
    add("### 3.2 等参等预算 2×2 因子对照全量六轴评测（`32,184` 步纯视频预训练 + 统一 `100%` 冻结解码器）")
    add()
    add("| 冻结编码器（预训练参数量） | Fidelity MSE10 (↓) | Tier-2 ARR (↑) | **纯 `z_op` Edit ARR (↑)** | Unseen Ridge (↓) | `R²(z→a7)` / `Fut` | `erank` (`full/op`) | **`vKNN` Lin (RBF ↓)** | **`RelP` Lin (RBF ↓)** | `RelP R²(z)` / `e_lat` |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        add(
            f"| {r['label']} | `{r['fid_mse10']:.5f}` | `{r['tier2_arr']*100:.1f}%` | **`{r['op_edit_arr']*100:.1f}%`** | "
            f"`{r['unseen_ridge']:.5f}` | `{r['r2_z_to_a7']:.3f}` / `{r['fut_r2']:.3f}` | "
            f"`{r['erank']:.1f}/{r['op_erank']:.1f}` | `{r['vk_lin4k']:.4f}` (`{r['vk_rbf4k']:.4f}`) | "
            f"`{r['rp_lin4k']:.4f}` (`{r['rp_rbf4k']:.4f}`) | `{r['rp_r2_z']:.3f}` / `{r['rp_elat']:.3f}` |"
        )
    add()
    add("**客观裁决（不预设融合有效）**：")
    add("- **等参对照严格排除容量红利**：`E17-D3-EqParam (84.88M)` 与 `E16-D3 (67.57M)` 在全部 6 轴持平（`RelP RBF 0.0162 vs 0.0162`, Fidelity `0.00224 vs 0.00221`），证明 E17 新模型的全部收益均来自表征组织方式而非预训练头容量。")
    add("- **为什么简单叠加 `VidRel + LearnReuse`（`E17-M1/M2`）无法同时通吃 D3 与 D4？** 2×2 因子对照（`D4` vs `M4` vs `M1` vs `M2`）给出了清晰的因果证据：只要在 $z_{\\text{op}}$ 上施加 2D 光流位移损失（`M1 / M2 / M3`），模型立刻在 `UnseenRidge (0.01075)`、`Fidelity (0.00200)` 与 `vKNN RBF (0.01876)` 上刷新纯视频纪录，但 2D 相机光流的视角非线性会破坏 3D 末端平行算术（`Tier-2` 从 `66.7% → 40.4%`、`OpEdit` 从 `51.4% → 22.7%–33.0%`）。")
    add("- **紧凑流形塑形与正交切分精准改善表征几何**：消除 3072D VICReg 秩膨胀后，**`E17-M3`** 把冻结解码器 $z_{\\text{op}}$ 主导度推至 **`94.6%`**、有效秩压至 **`25.1`（逼近 Oracle-E9 的 `36.0`）**、零泄漏潜空间预测度推至 **`RelP R²(z) = 0.604` (`e_lat = 0.620`)** 并创下纯视频最低 `vKNN RBF = 0.01876`；而 **`E17-M4 (Compact Sharp Reuse)`** 在保留纯视频最强 Tier-2 算术 **`66.7%`** 与强纯 $z_{\\text{op}}$ 编辑 **`51.4%`** 的同时，把 `D4` 的零泄漏 Visual ICL 误差全面压降 **`-11% ~ -14%`**（`RelP RBF 0.0196 → 0.0175`）！")
    add()
    add("---")
    add()

    # Section 4: Reconciled Label Efficiency + Controlled Scaling & Coverage
    add("## 4. Pillar 4 — 统一评测口径核实、控步规模缩放律与关系覆盖度")
    add()
    add("![Reconciled Label Efficiency](figures/fig5_reconciled_label_efficiency.png)")
    add()
    add("### 4.1 核实 E16 评测口径差异 & 统一冻结解码器动作标注效率（全量 311 任务 5,799 非 tail 样本）")
    add()
    add("核实结果明确揭示了 E16 报告第 3 节（`0.00221`）与第 4.1 节（`0.00483`）之间存在 `2.2×–2.9×` 差异的根源：E16 4.1 节引用的 `metrics.json` 仅对验证集**前 512 个顺序样本**做 DDIM 解码，实际只覆盖 **27/311 任务**（集中于难度较高的 `libero_goal_chain_firststep_view`）且混入 **61 个 tail 截断样本**。下表在完全统一的 **Full-5799（全部 311 任务非 tail）** 与 **3 种子零泄漏 Visual ICL** 口径下重新汇总全部 `1% / 10% / 100%` 冻结解码器结果：")
    add()
    add("| 冻结编码器 | `1%` (140 eps) Full-5799 (`Pref512`) | `1%` RelP RBF (↓) | `10%` (1,244 eps) Full-5799 | `10%` RelP RBF (↓) | `100%` (13,995 eps) Full-5799 (`Pref512`) | `100%` RelP RBF (↓) |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    le_tbl = [
        ("E16-D0 (Random Frozen)", "e16_d0_rand_dec01", "e16_d0_rand_dec10", "e16_d0_rand_dec100"),
        ("E16-D1 (Pure LAM)", "e16_d1_lam_dec01", "e16_d1_lam_dec10", "e16_d1_lam_dec100"),
        ("E16-D3 (VidRel-Op)", "e16_d3_vidop_dec01", "e16_d3_vidop_dec10", "e16_d3_vidop_dec100"),
        ("E16-D4 (LearnReuse-Op)", "e16_d4_learnop_dec01", "e16_d4_learnop_dec10", "e16_d4_learnop_dec100"),
        ("**E17-M1 (Joint Vid+SharpReuse)**", "e17_m1_dec01", "e17_m1_dec10", "e17_m1_joint_op_dec100"),
        ("**E17-M4 (Compact Sharp Reuse)**", "e17_m4_dec01", "e17_m4_dec10", "e17_m4_compact_reuse_dec100"),
        ("Oracle-E9 (Action Frozen)", "e16_d_oracle_e9_dec01", "e16_d_oracle_e9_dec10", "e16_d_oracle_e9_dec100"),
    ]
    for lbl, k1, k10, k100 in le_tbl:
        if not all(k in rec for k in (k1, k10, k100)):
            continue
        f1 = rec[k1]["last_ema"]["protoC_full5799_nontail_311tasks"]["mse10"]
        p1 = rec[k1]["last_ema"]["protoA_prefix512_27tasks"]["mse10"]
        r1 = rec[k1]["best"]["protoD_zero_leakage_icl"]["rel_progress"]["rbf4k_mse10_mean"]
        f10 = rec[k10]["last_ema"]["protoC_full5799_nontail_311tasks"]["mse10"]
        r10 = rec[k10]["best"]["protoD_zero_leakage_icl"]["rel_progress"]["rbf4k_mse10_mean"]
        f100 = rec[k100]["last_ema"]["protoC_full5799_nontail_311tasks"]["mse10"]
        p100 = rec[k100]["last_ema"]["protoA_prefix512_27tasks"]["mse10"]
        r100 = rec[k100]["best"]["protoD_zero_leakage_icl"]["rel_progress"]["rbf4k_mse10_mean"]
        add(f"| {lbl} | `{f1:.5f}` (`{p1:.5f}`) | `{r1:.4f}` | `{f10:.5f}` | `{r10:.4f}` | `{f100:.5f}` (`{p100:.5f}`) | `{r100:.4f}` |")
    add()
    add("- **口径核实后的新发现**：(1) 在 `1%` 极低标注下，`val_self_diff_loss` 早停（`best.pt @1500–2000`）反而不如 `last_ema.pt @2682` 的 DDIM 动作精度（EMA 去噪仍在改善）；(2) **`E17-M1` 在 `1%` 标注 Fidelity 上创纯视频新低 (`0.00503`)**，而紧凑流形 **`E17-M4` 在 `1%` 标注零泄漏 Visual ICL 上以 `0.0189` 碾压所有纯视频编码器（领先 `100%` 标注 Pure LAM `0.0219` 达 `-14%`！）**，且 `10%` 标注达 `0.0167` 甚至优于自身 `100%` 解码器 `0.0175`（更温和的解码器带来更低 $G_{\\text{dec}}$）。")
    add()
    add("### 4.2 严格控步的数据规模缩放、步数缩放与关系覆盖度（消除 E16 `16k vs 32k` 步数混杂）")
    add()
    add("![Controlled Scaling and Coverage](figures/fig6_controlled_scaling_and_coverage.png)")
    add()
    add("#### (A) 严格固定训练步数（`16,092` 步）下的无标注视频数据规模缩放（`10% → 30% → 100%`）")
    add()
    add("| 方法（固定 `16,092` 步） | `10%` / `30%` / `100%` Probe `100%` $R^2$ (↑) | `10%` / `30%` / `100%` `Fut16` $R^2$ (↑) | `10%` / `30%` / `100%` 跨任务 Top-1 MSE (↓) | `10%` / `30%` / `100%` 冻结解码器 Full-5799 (RelP RBF ↓) |")
    add("|---|---|---|---|---|")
    sc_groups = [
        ("D1 Pure LAM *(无关系基线)*", ["sc_d1_10_16k", "sc_d1_30_16k", "sc_d1_100_16k"], [None, None, None]),
        ("D3 VidRel-Op", ["sc_d3_10_16k", "sc_d3_30_16k", "sc_d3_100_16k"], ["e16_d3_scale10_dec100", "e16_d3_scale30_dec100", "sc_d3_100_16k_dec100"]),
        ("D4 LearnReuse-Op", ["sc_d4_10_16k", "sc_d4_30_16k", "sc_d4_100_16k"], [None, None, "sc_d4_100_16k_dec100"]),
        ("**E17-M1 (Joint Vid+SharpReuse)**", ["sc_m1_10_16k", "sc_m1_30_16k", "sc_m1_100_16k"], ["sc_m1_10_16k_dec100", "sc_m1_30_16k_dec100", "sc_m1_100_16k_dec100"]),
    ]
    for lbl, ks, dks in sc_groups:
        if not all(k in sc for k in ks):
            continue
        pr = " / ".join(f"`{sc[k]['label_efficiency_probe']['100pct']['r2_full_mean']:.3f}`" for k in ks)
        ft = " / ".join(f"`{sc[k]['beyond_e12']['fut16_r2']:.3f}`" for k in ks)
        t1 = " / ".join(f"`{sc[k]['audit_full']['ct_top1_mse10']:.4f}`" for k in ks)
        dec_parts = []
        for dk in dks:
            if dk and dk in rec:
                ff = rec[dk]["last_ema"]["protoC_full5799_nontail_311tasks"]["mse10"]
                rr = rec[dk]["best"].get("protoD_zero_leakage_icl", {}).get("rel_progress", {}).get("rbf4k_mse10_mean")
                dec_parts.append(f"`{ff:.4f}` (`{rr:.4f}`)" if rr else f"`{ff:.4f}`")
            else:
                dec_parts.append("—")
        add(f"| {lbl} | {pr} | {ft} | {t1} | {' / '.join(dec_parts)} |")
    add()
    add("#### (B) 固定 `100%` 数据下的训练步数缩放（`8k → 16k → 32k`）与「秩膨胀反转」")
    add()
    add("| 方法（固定 `100%` 数据） | 步数节点 | `erank` (`full/op`) 演变 | Probe `100%` $R^2$ 演变 | **`RelP R²(z)` 演变** | **`vKNN R²(z)` 演变** | 结论 |")
    add("|---|---|---|---|---|---|---|")
    step_tbl = [
        ("D1 Pure LAM", "16k → 32k", ["sc_d1_100_16k", "e16_d1_32k"], "秩膨胀反转 (`-6.8 pp`)"),
        ("D3-EqParam", "8k → 16k → 32k", ["e17_d3eq_8k", "e17_d3eq_16k", "e17_d3eq_32k"], "秩膨胀反转 (`-4.5 pp`)"),
        ("D4 LearnReuse-Op", "16k → 32k", ["sc_d4_100_16k", "e16_d4_32k"], "秩膨胀反转 (`-6.0 pp`)"),
        ("E17-M1 (Uncompact)", "8k → 16k → 32k", ["e17_m1_8k", "sc_m1_100_16k", "e17_m1_32k"], "秩膨胀反转 (`-5.3 pp`)"),
        ("**E17-M2 (Compact)**", "8k → 16k → 32k", ["e17_m2_8k", "e17_m2_16k", "e17_m2_32k"], "**严格单调改善 (`+1.4/+1.8 pp`)**"),
        ("E17-M4 (Compact Reuse)", "8k → 16k → 32k", ["e17_m4_8k", "e17_m4_16k", "e17_m4_32k"], "秩温和扩增 (`27→41`), Probe 单调涨"),
    ]
    for lbl, st_s, ks, note in step_tbl:
        if not all(k in sc for k in ks):
            continue
        er_s = " → ".join(f"`{sc[k]['audit_full']['erank']:.0f}/{sc[k]['audit_op']['erank']:.0f}`" for k in ks)
        pr_s = " → ".join(f"`{sc[k]['label_efficiency_probe']['100pct']['r2_full_mean']:.3f}`" for k in ks)
        rp_s = " → ".join(f"`{sc[k]['zero_leakage_latent_predictability']['rel_progress']['r2_z']:.3f}`" for k in ks)
        vk_s = " → ".join(f"`{sc[k]['zero_leakage_latent_predictability']['visual_knn']['r2_z']:.3f}`" for k in ks)
        add(f"| {lbl} | {st_s} | {er_s} | {pr_s} | {rp_s} | {vk_s} | {note} |")
    add()
    add("#### (C) 严格固定 `16,092` 步下的纯视频 vs 任务元数据关系覆盖度对照（Regime 2 `LearnReuse-Op`）")
    add()
    add("| 覆盖度模式 | 元数据依赖 | `Probe 1%/10%/100%` | `Op_CTbal` / `Disp` | `erank` (`full/op`) | `RelP R²(z)` | **冻结解码器 Full-5799 (↓)** | **冻结解码器 `RelP` Lin / RBF (↓)** |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    cov_tbl = [
        ("`same_task_only`", "TaskMeta-Restricted", "cov_d4_sametask_16k", "cov_d4_sametask_16k_dec100"),
        ("`cross_demo_anti_scene`", "**Pure Video**", "sc_d4_100_16k", "sc_d4_100_16k_dec100"),
        ("**`cross_demo_unmasked`**", "**Pure Video (Zero Task Meta)**", "cov_d4_unmasked_16k", "cov_d4_unmasked_16k_dec100"),
        ("`strict_cross_task`", "TaskMeta-Assisted", "cov_d4_strict_ct_16k", "cov_d4_strict_ct_16k_dec100"),
    ]
    for lbl, meta, ek, dk in cov_tbl:
        if ek not in sc:
            continue
        r = sc[ek]
        lp = r["label_efficiency_probe"]
        ao = r["audit_op"]
        af = r["audit_full"]
        rpz = r["zero_leakage_latent_predictability"]["rel_progress"]["r2_z"]
        ff_s = "—"
        icl_s = "—"
        if dk in rec:
            ff_s = f"`{rec[dk]['last_ema']['protoC_full5799_nontail_311tasks']['mse10']:.5f}`"
            ic = rec[dk]["best"].get("protoD_zero_leakage_icl", {}).get("rel_progress", {})
            if ic:
                icl_s = f"`{ic['lin4k_mse10_mean']:.4f}` / **`{ic['rbf4k_mse10_mean']:.4f}`**"
        add(
            f"| {lbl} | {meta} | `{lp['1pct']['r2_full_mean']:.3f}/{lp['10pct']['r2_full_mean']:.3f}/{lp['100pct']['r2_full_mean']:.3f}` | "
            f"`{ao['ct_bal_rho']:.3f}` / `{ao['disp_rho']:.3f}` | `{af['erank']:.1f}/{ao['erank']:.1f}` | `{rpz:.3f}` | {ff_s} | {icl_s} |"
        )
    add()
    add("---")
    add()

    # Section 5: Paired Per-Task Statistics & 95% CIs
    add("## 5. 逐任务配对统计检验与 95% Bootstrap 置信区间（`10,000` 次重采样）")
    add()
    add("![Paired Task Forest Plot](figures/fig7_paired_task_stats_forest.png)")
    add()
    add("所有对比均在**完全相同的任务切分、完全相同的测试对、以及完全相同的 DDIM 初始噪声**下执行逐任务配对（Fidelity：`311` 个任务；零泄漏 Visual ICL：3 种子留出未见任务按唯一任务聚合；纯 $z_{\\text{op}}$ 编辑：`136` 个四元组）：")
    add()
    add("| 配对比较 (`B vs A`) | Fidelity ΔMSE10 (`95% CI`, Win%, Wilcoxon $p$) | **RelP RBF ΔMSE10 (`95% CI`, Win%, Wilcoxon $p$)** | **纯 `z_op` Edit ΔMSE10 (Win%, $p$)** |")
    add("|---|---|---|---|")
    for k, comp in paired.items():
        f = comp["fidelity_per_task"]
        r = comp["icl_rel_progress_rbf4k_per_task"]
        o = comp["edit_op_arith_mse10"]
        def _fmt(s: Dict[str, Any]) -> str:
            star = "**" if s["significant_95ci"] else ""
            return f"{star}`{s['delta_B_minus_A']:+.5f}`{star} ([`{s['ci95_delta'][0]:+.5f}`,`{s['ci95_delta'][1]:+.5f}`], `{s['win_rate_B_over_A']*100:.0f}%`, `p={s['wilcoxon_p']:.1e}`)"
        add(f"| `{k.replace('_vs_', ' vs ')}` | {_fmt(f)} | {_fmt(r)} | `{o['delta_B_minus_A']:+.4f}` (`{o['win_rate_B_over_A']*100:.0f}%`, `p={o['wilcoxon_p']:.1e}`) |")
    add()
    add("*注：`纯 z_op Edit ΔMSE10` 在 136 个跨任务四元组上呈夹爪翻转/未翻转双峰分布，因此均值差（反映大幅夹爪翻转纠错，如 `E17-M4 vs E17-M2: -0.0245`，对应 `OpEdit ARR 51.4% vs 33.0%`）与逐样本中位胜率（易受未翻转四元组的微小 Fidelity 差异影响）应结合表 3.2 的 `OpEdit ARR` 共同解读。*")
    add()
    add("---")
    add()

    # Section 6: Final Synthesis
    add("## 6. 总结：如何进一步改善纯视频行为表征的组织方式？")
    add()
    add("综合 E17 全部四条主线与逐任务配对证据，我们给出三条可直接指导下一代大规模纯视频行为表征预训练的设计原则：")
    add("1. **用「纯视频全跨 Demo 稀疏 Top-K / Top-1 复用」取代 `anti_scene` 软平均，且无需任务元数据**：在足够多样的轨迹池中，仅排除同轨迹（`diff_ep`）即可达到与 Oracle 任务 ID 掩码（`strict_cross_task`）精确等价的跨任务泛化；配合稀疏 `Top-16` 与直通 `Top-1` 单 Donor 复用损失，可根除凸包平均作弊，使单 Donor 在最不相似的 25% 场景中依然实现 `+30%` 跨任务动作降错。")
    add("2. **移除高维各向同性 VICReg 秩膨胀，采用「紧凑流形瓶颈 + $z_{\\text{op}}$ 能量主导」**：E16-D3/D4 与 E9 之间残余 Visual ICL 差距的主因并非缺乏动作监督，而是 3072D VICReg 随训练步数持续吹胀 `erank`（导致 16k→32k 的「秩膨胀反转」）。采用紧凑瓶颈后，`E17-M2` 成为唯一随训练步数严格单调提升 Visual ICL 预测度的纯视频模型，`E17-M3` 将有效秩压至 `25–39`、把 $z_{\\text{op}}$ 解码器主导度拉至 `94.6%`，`E17-M4` 则在 1% 动作标注下刷新 Visual ICL 纪录并实现超越 `Oracle-E9` 的子目标突变检出（`AUC = 0.770`）。")
    add("3. **尊重「2D 像素运动几何 vs 3D 动作算术」的分工，避免直接目标冲突**：2D 像素光流（`VidRel`）是极佳的视觉检索与零泄漏迁移锚点，但不应与 3D 动作平行算术（`LearnReuse`）强制共用同一未解耦切片。未来的最优架构应当采用**双通道显式路由或物理 3D/多视角几何一致流**——在无动作预训练阶段利用紧凑复用流形捕获子目标边界与 3D 行为算术（`M4`），同时以轻量低秩运动特征支撑高保真视觉检索与零泄漏迁移（`M1/M3`）。")
    add()
    add("---")
    add()
    add("## 附录：产物与数据索引")
    add("- **核心模型六轴总表**：[e17_master_comparison.csv](e17_master_comparison.csv)")
    add("- **Relation Authenticity 11 项对照与反事实表**：[e17_authenticity_table.csv](e17_authenticity_table.csv)")
    add("- **统一口径核实与 1%–100% 标注效率表**：[e17_reconciled_label_efficiency.csv](e17_reconciled_label_efficiency.csv)")
    add("- **严格控步 Scaling & Coverage 29 模型总表**：[e17_scaling_coverage_table.csv](e17_scaling_coverage_table.csv)")
    add("- **逐任务配对统计与 95% Bootstrap CI 总表**：[e17_paired_stats_table.csv](e17_paired_stats_table.csv)")
    add("- **可视化图表目录**：`e17/outputs/figures/fig1..fig7.png`")

    out_md = OUTPUT_ROOT / "REPORT.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[E17 Report] Wrote {out_md} ({len(lines)} lines) + 7 figures + 5 CSV tables.", flush=True)


if __name__ == "__main__":
    generate_report()
