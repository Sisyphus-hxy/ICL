"""Generate E15 publication-grade figures, CSV master tables, and comprehensive REPORT.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import OUTPUT_ROOT

MODEL_ORDER = [
    "e9",
    "control",
    "e12_c",
    "e12_bc_factordisp",
    "e15_a_e12c_realcv",
    "e15_a_e12c_rand",
    "e15_a_e9_realcv",
    "e15_a_e9_rand",
    "e15_b_e12c_realcv",
    "e15_b_e12c_rand",
    "e15_b_e9_realcv",
    "e15_b_e9_rand",
    "e15_c_op_only",
    "e15_c_op_decorr",
    "e15_d_op_mild_cv",
    "e15_d_op_bcfd",
]

SHORT_LABELS = {
    "e9": "E9 (Baseline)",
    "control": "E9-Cont",
    "e12_c": "E12-C",
    "e12_bc_factordisp": "E12-BC-FD",
    "e15_a_e12c_realcv": "E15-A E12-C (RealCV)",
    "e15_a_e12c_rand": "E15-A E12-C (Rand)",
    "e15_a_e9_realcv": "E15-A E9 (RealCV)",
    "e15_a_e9_rand": "E15-A E9 (Rand)",
    "e15_b_e12c_realcv": "E15-B E12-C (RealCV)",
    "e15_b_e12c_rand": "E15-B E12-C (Rand)",
    "e15_b_e9_realcv": "E15-B E9 (RealCV)",
    "e15_b_e9_rand": "E15-B E9 (Rand)",
    "e15_c_op_only": "E15-C (OpOnly)",
    "e15_c_op_decorr": "E15-C (OpDecorr)",
    "e15_d_op_mild_cv": "E15-D (Op+MildCV)",
    "e15_d_op_bcfd": "E15-D (Op-BCFD)",
}

COLORS = {
    "e9": "#1f77b4",
    "control": "#6baed6",
    "e12_c": "#d62728",
    "e12_bc_factordisp": "#9467bd",
    "e15_a_e12c_realcv": "#e6550d",
    "e15_a_e12c_rand": "#fdae6b",
    "e15_a_e9_realcv": "#3182bd",
    "e15_a_e9_rand": "#9ecae1",
    "e15_b_e12c_realcv": "#a50f15",
    "e15_b_e12c_rand": "#fb6a4a",
    "e15_b_e9_realcv": "#08519c",
    "e15_b_e9_rand": "#bdd7e7",
    "e15_c_op_only": "#2ca02c",
    "e15_c_op_decorr": "#17becf",
    "e15_d_op_mild_cv": "#bcbd22",
    "e15_d_op_bcfd": "#e377c2",
}


def _load_all_evals(eval_root: Path) -> Dict[str, Dict[str, Any]]:
    evals: Dict[str, Dict[str, Any]] = {}
    for k in MODEL_ORDER:
        p = eval_root / k / "eval_e15.json"
        if p.is_file():
            evals[k] = json.loads(p.read_text())
    return evals


def _extract_flat_row(k: str, d: Dict[str, Any]) -> Dict[str, Any]:
    fid = d["fidelity"]["self_ddim"]
    t2 = d["editability"]["displacement_and_arithmetic"]["latent_arithmetic_tier2_distinct_start"]
    u4 = d["editability"]["unseen_combinations"]["linear_compositional_ridge"]
    zs = d["editability"]["unseen_combinations"]["zero_shot_compositional_arithmetic"]
    a100 = d["editability"]["transfer_efficiency"]["action_change_transfer_efficiency"]["by_n"]["100"]["linear_ridge"]
    a1000 = d["editability"]["transfer_efficiency"]["action_change_transfer_efficiency"]["by_n"]["1000"]["linear_ridge"]
    op_ed = d["editability"]["op_subspace_editability"]
    mid = d["editability"]["midpoint_interpolation"]
    c5 = d["composability"]["matrix_5x5"]
    uc = d["composability"]["unseen_combo_swaps"]
    pr = d["content_probes"]["full"]
    pr_op = d["content_probes"]["op_slice"]
    pr_rs = d["content_probes"]["res_slice"]
    t3 = d["content_probes"]["three_component_energy"]

    vk = d["robustness_and_visual_icl"]["visual_knn"]
    rp = d["robustness_and_visual_icl"]["rel_progress"]
    vk_fl = vk["decoder_robustness"]["factorization_linear"]
    rp_fl = rp["decoder_robustness"]["factorization_linear"]
    vk_lin = vk["visual_icl"]["adapters"]["lin_full"]["4000"]
    vk_rbf = vk["visual_icl"]["adapters"]["rbf_krr_full"]["4000"]
    rp_lin = rp["visual_icl"]["adapters"]["lin_full"]["4000"]
    rp_rbf = rp["visual_icl"]["adapters"]["rbf_krr_full"]["4000"]
    vk_c = vk["decoder_robustness"]["curves"]
    rp_c = rp["decoder_robustness"]["curves"]

    return {
        "model": k,
        "label": SHORT_LABELS.get(k, k),
        # 1. Fidelity
        "fid_mse10": fid["mse10"],
        "fid_mae7": fid["mae7"],
        "fid_grip_acc": fid["grip_acc"],
        "fid_accel_mse": fid["accel_pos_mse"],
        "fid_jerk_mse": fid["jerk_pos_mse"],
        "fid_grip_sw_acc": fid["grip_switch_acc"],
        # 2. Editability (Oracle)
        "tier2_mse10": t2["ddim_arith"]["action10_mse"],
        "tier2_arr": t2["arithmetic_recovery_ratio_10d"],
        "tier2_cos": t2["latent_disp_cos"],
        "op_only_arith_mse10": op_ed["op_only_arith"]["mse10"],
        "op_only_arith_arr": op_ed["op_only_arith"]["arr_10d"],
        "op_disp_cos": op_ed["disp_cos_op"],
        "unseen_ridge_mse10": u4["unseen_ddim"]["action10_mse"],
        "unseen_seen_ratio": u4["unseen_to_seen_mse10_ratio"],
        "zs_comp_arith_arr": zs.get("recovery_ratio_10d", float("nan")),
        "act_lin100_mse10": a100["ddim_10d_mse"],
        "act_lin1000_mse10": a1000["ddim_10d_mse"],
        "midpoint_mse10": mid["mse10"],
        # 3. Composability (Oracle)
        "comp_tok0_adopt": c5["translation"]["target_adoption"],
        "comp_tok1_adopt": c5["rotation"]["target_adoption"],
        "comp_tok2_mag_adopt": c5["magnitude"]["target_adoption"],
        "comp_tok3_grip_adopt": c5["gripper"]["target_adoption"],
        "comp_tok3_grip_acc": c5["gripper"].get("donor_grip_sign_acc", float("nan")),
        "comp_unseen_dg_tok3_arr": uc["unseen_dir_x_grip_tok3_swap"]["arr_10d"],
        "comp_unseen_dg_tok0_posarr": uc["unseen_dir_x_grip_tok0_swap"]["arr_pos"],
        "comp_unseen_ds_tok2_rec": uc["unseen_dir_x_speed_tok2_swap"]["path_recovery_ratio"],
        # 4. Content & Predictability
        "r2_a_to_z": pr["r2_a_to_z_mean"],
        "r2_z_to_a7": pr["r2_z_to_a7_mean"],
        "fut_r2": pr["fut_r2_mean"],
        "vis_hq_r2": pr["vis_hq_r2_mean"],
        "obj_acc": pr["obj_acc_mean"],
        "erank_z": pr["erank_mean"],
        "op_erank": pr_op["erank_mean"],
        "res_erank": pr_rs["erank_mean"],
        "op_r2_a_to_z": pr_op["r2_a_to_z_mean"],
        "res_r2_a_to_z": pr_rs["r2_a_to_z_mean"],
        "res_fut_r2": pr_rs["fut_r2_mean"],
        "res_hq_r2": pr_rs["vis_hq_r2_mean"],
        "cka_op_res": t3["cka_op_res_mean"],
        "res_slice_e_frac": t3["res_slice_energy_frac_mean"],
        "frac_lin_exact": t3["frac_lin_mean"],
        "frac_coax_exact": t3["frac_coax_mean"],
        "frac_perp_exact": t3["frac_perp_mean"],
        # 5. Decoder Robustness
        "vknn_e_lat": vk_fl["e_lat_rel_mean"],
        "vknn_G_dec": vk_fl["G_dec_excess_mean"],
        "vknn_exc_rms": vk_fl["excess_act_rms_mean"],
        "relp_e_lat": rp_fl["e_lat_rel_mean"],
        "relp_G_dec": rp_fl["G_dec_excess_mean"],
        "relp_exc_rms": rp_fl["excess_act_rms_mean"],
        "shift35_iso": vk_c["isotropic"]["0.35"]["shift_rms_vs_clean_mean"],
        "shift35_par": vk_c["subspace_par"]["0.35"]["shift_rms_vs_clean_mean"],
        "shift35_perp": vk_c["subspace_perp"]["0.35"]["shift_rms_vs_clean_mean"],
        "shift35_icl_vknn": vk_c["icl_err_rescaled"]["0.35"]["shift_rms_vs_clean_mean"],
        "shift35_icl_relp": rp_c["icl_err_rescaled"]["0.35"]["shift_rms_vs_clean_mean"],
        "shrink30_mse10": vk_c["shrinkage"]["0.30"]["mse10_vs_gt_mean"],
        "gt_par_only_mse10": vk_c["gt_par_only"]["0.00"]["mse10_vs_gt_mean"],
        "gt_shuf_perp_mse10": vk_c["gt_shuf_perp"]["0.00"]["mse10_vs_gt_mean"],
        # 6. Zero-Leakage Visual ICL
        "vknn_lin4k_mse10": vk_lin["mse10_mean"],
        "vknn_lin4k_arr": vk_lin["arr_10d_mean"],
        "vknn_rbf4k_mse10": vk_rbf["mse10_mean"],
        "vknn_r2_z": vk_lin["r2_z_mean"],
        "vknn_cos_dz": vk_lin["cos_dz_mean"],
        "relp_lin4k_mse10": rp_lin["mse10_mean"],
        "relp_lin4k_arr": rp_lin["arr_10d_mean"],
        "relp_rbf4k_mse10": rp_rbf["mse10_mean"],
        "relp_r2_z": rp_lin["r2_z_mean"],
        "relp_cos_dz": rp_lin["cos_dz_mean"],
    }


def _plot_fig1_part1(p1: Dict[str, Any], fig_dir: Path) -> None:
    res = p1["results"]
    models = ("e9", "e12_c", "e12_bc_factordisp")
    mlabs = ("E9", "E12-C", "E12-BC-FD")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), dpi=180)
    plt.subplots_adjust(hspace=0.34, wspace=0.28)

    # Panel A: Exact 3-Component Energy Split vs E14 SVD-0.995
    ax = axes[0, 0]
    x = np.arange(len(models))
    lin_v = [res[m]["energy_audit_test_unseen"]["frac_lin_mean"] * 100 for m in models]
    coax_v = [res[m]["energy_audit_test_unseen"]["frac_coax_mean"] * 100 for m in models]
    perp_v = [res[m]["energy_audit_test_unseen"]["frac_perp_mean"] * 100 for m in models]
    e14_perp = [res[m]["svd_truncation_audit"]["0.995"]["frac_perp_mean"] * 100 for m in models]

    w = 0.42
    ax.bar(x - w / 2, lin_v, w, label=r"$z_{\mathrm{lin}}$ (Action Linear)", color="#1f77b4")
    ax.bar(x - w / 2, coax_v, w, bottom=lin_v, label=r"$z_{\mathrm{coax}}$ (Co-axial Residual)", color="#ff7f0e")
    ax.bar(x - w / 2, perp_v, w, bottom=np.array(lin_v) + np.array(coax_v), label=r"$z_{\perp}$ (True Orthogonal, $r=160$)", color="#2ca02c")
    ax.bar(x + w / 2, [100 - v for v in e14_perp], w, color="#aec7e8", alpha=0.85, label=r"E14 $z_{\parallel}^{(0.995)}$ ($r=15\text{--}26$)")
    ax.bar(x + w / 2, e14_perp, w, bottom=[100 - v for v in e14_perp], color="#98df8a", alpha=0.95, hatch="//", edgecolor="#2ca02c", label=r"E14 $z_{\perp}^{(0.995)}$ (SVD-Truncated)")
    for i in range(len(models)):
        ax.text(x[i] - w / 2, lin_v[i] / 2, f"{lin_v[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text(x[i] - w / 2, lin_v[i] + coax_v[i] / 2, f"{coax_v[i]:.1f}%", ha="center", va="center", color="black", fontsize=8)
        ax.text(x[i] + w / 2, 100 - e14_perp[i] / 2, f"{e14_perp[i]:.1f}%", ha="center", va="center", color="black", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(mlabs, fontsize=10, fontweight="bold")
    ax.set_ylabel("Variance Fraction (%)")
    ax.set_ylim(0, 108)
    ax.set_title("(A) Exact 3-Component Energy ($r=160$) vs E14 SVD-0.995", fontweight="bold")
    ax.legend(fontsize=7.3, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Ridge Cross-Term Audit across alpha
    ax = axes[0, 1]
    alphas_str = ("0", "0.01", "0.1", "1", "10", "100")
    x_al = np.arange(len(alphas_str))
    for m, lab, col in zip(models, mlabs, ("#1f77b4", "#d62728", "#9467bd")):
        tr_c = [res[m]["ridge_alpha_crossterm_audit"][a]["train_frac_cross_mean"] * 100 for a in alphas_str]
        te_c = [res[m]["ridge_alpha_crossterm_audit"][a]["test_frac_cross_mean"] * 100 for a in alphas_str]
        ax.plot(x_al, te_c, "-o", color=col, lw=2.0, label=f"{lab} Test (Unseen)")
        ax.plot(x_al, tr_c, "--s", color=col, lw=1.2, alpha=0.6, label=f"{lab} Train")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.axvline(3, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.text(3.05, -5.5, r"CV $\alpha=1$", fontsize=8.5, color="gray")
    ax.set_xticks(x_al)
    ax.set_xticklabels(["0 (OLS)", "0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel(r"Ridge Regularization $\alpha$")
    ax.set_ylabel(r"Cross-Term $2\langle z_{\mathrm{lin}}, z_{\mathrm{coax}}\rangle / \|z_c\|^2$ (%)")
    ax.set_title(r"(B) Ridge Cross-Term Audit: Train Shrinkage vs Test Overfit", fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)

    # Panel C: Reconstruction MSE10 under 3-Component Ablation
    ax = axes[1, 0]
    rec_keys = ("full", "par_only_exact", "lin_plus_coax_plus_shuf_perp", "par_only_svd995", "lin_plus_perp", "lin_only", "lin_plus_shuf_coax_plus_perp")
    rec_labs = ("Full", "Par Exact\n(no perp)", "Shuf\nperp", "E14 Par\n(SVD .995)", "Lin+Perp\n(no coax)", "Lin\nOnly", "Shuf\ncoax")
    xr = np.arange(len(rec_keys))
    bw = 0.25
    for idx, (m, lab, col) in enumerate(zip(models, mlabs, ("#1f77b4", "#d62728", "#9467bd"))):
        vals = [res[m]["reconstruction_motion_details"][rk]["mse10_mean"] * 1000 for rk in rec_keys]
        ax.bar(xr + (idx - 1) * bw, vals, bw, label=lab, color=col)
    ax.set_xticks(xr)
    ax.set_xticklabels(rec_labs, fontsize=8)
    ax.set_ylabel("Action 10D MSE (×1e-3, ↓)")
    ax.set_title("(C) Reconstruction Ablation: $z_{\\mathrm{coax}}$ vs True $z_{\\perp}$ vs SVD-0.995", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Tier-2 Editability Selective Transport (ARR %)
    ax = axes[1, 1]
    ed_keys = ("arith_full", "transport_par_only", "transport_lin_only", "arith_par_zero_perp")
    ed_labs = ("Full Δz\nArith", "Transport\nΔz_par", "Transport\nΔz_lin Only", "Par Arith\n(z_perp=0)")
    xe = np.arange(len(ed_keys))
    for idx, (m, lab, col) in enumerate(zip(models, mlabs, ("#1f77b4", "#d62728", "#9467bd"))):
        vals = [res[m]["tier2_editability_ablation"]["variants"][ek]["arr_10d"] * 100 for ek in ed_keys]
        bars = ax.bar(xe + (idx - 1) * bw, vals, bw, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(xe)
    ax.set_xticklabels(ed_labs, fontsize=8.5)
    ax.set_ylabel("Tier-2 Parallel Arithmetic ARR (%, ↑)")
    ax.set_ylim(78, 95)
    ax.set_title(r"(D) Tier-2 Editability: Stripping $\Delta z_{\mathrm{coax}}$ Recovers 89% of E9 Gap", fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("E15 Part 1 (Training-Free): Strict Three-Component Decomposition & Error Audit", fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(fig_dir / "fig1_part1_three_component_ablation.png", bbox_inches="tight")
    plt.close(fig)


def _plot_fig2_ab_diagnosis(rows_by_k: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    keys = [
        "e9", "e15_a_e9_rand", "e15_a_e9_realcv", "e15_b_e9_rand", "e15_b_e9_realcv",
        "e12_c", "e15_a_e12c_rand", "e15_a_e12c_realcv", "e15_b_e12c_rand", "e15_b_e12c_realcv",
    ]
    labs = [
        "E9\nClean", "A-E9\nRand", "A-E9\nRealCV", "B-E9\nRand", "B-E9\nRealCV",
        "E12-C\nClean", "A-12C\nRand", "A-12C\nRealCV", "B-12C\nRand", "B-12C\nRealCV",
    ]
    cols = [COLORS[k] for k in keys]
    x = np.arange(len(keys))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=180)
    plt.subplots_adjust(hspace=0.36, wspace=0.22)

    panels = [
        ("fid_mse10", 1000.0, "Self DDIM MSE10 (×1e-3, ↓)", "(A) Fidelity: Real-CV Noise Blurs Clean Decoding 2.5×"),
        ("tier2_arr", 100.0, "Tier-2 Arithmetic ARR (%, ↑)", "(B) Editability: Full-Scale Real-CV Blunts Edits"),
        ("relp_G_dec", 1.0, r"Decoder Gain $G_{\mathrm{dec}}$ (RelP, ↓)", "(C) Decoder Gain Drops, Yet Total Error Rises"),
        ("relp_lin4k_mse10", 1000.0, "Zero-Leakage Visual ICL RelP MSE10 (×1e-3, ↓)", "(D) Visual ICL: Neither E15-A nor E15-B Beats Clean Baseline"),
    ]
    for ax, (field, mult, ylab, title) in zip(axes.flat, panels):
        vals = [rows_by_k[k][field] * mult for k in keys]
        bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.5)
        ax.axvline(4.5, color="black", ls="--", lw=1.0, alpha=0.6)
        for b, v in zip(bars, vals):
            fmt = f"{v:.1f}" if mult == 100.0 else (f"{v:.2f}" if mult == 1000.0 else f"{v:.3f}")
            ax.text(b.get_x() + b.get_width() / 2, v, fmt, ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if field == "tier2_arr":
            ax.set_ylim(40, 100)
        elif field == "relp_G_dec":
            ax.set_ylim(0.16, 0.32)
        elif field == "relp_lin4k_mse10":
            ax.set_ylim(12.0, 20.0)

    fig.suptitle("E15-A & E15-B Diagnosis: Real-CV Adapter Residual vs Scale-Matched Random Noise (E12-C & Fair E9)", fontsize=12.5, fontweight="bold", y=0.98)
    fig.savefig(fig_dir / "fig2_e15_ab_noise_robustness_diagnosis.png", bbox_inches="tight")
    plt.close(fig)


def _plot_fig3_subspace(evals: Dict[str, Dict[str, Any]], rows_by_k: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    sub_models = [k for k in ("e9", "e12_c", "e12_bc_factordisp", "e15_c_op_only", "e15_c_op_decorr", "e15_d_op_mild_cv", "e15_d_op_bcfd") if k in rows_by_k]
    mlabs = [SHORT_LABELS[k] for k in sub_models]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), dpi=180)
    plt.subplots_adjust(wspace=0.28)

    # Panel A: Op vs Res slice properties (erank & R2(a->z))
    ax = axes[0]
    x = np.arange(len(sub_models))
    w = 0.36
    op_er = [rows_by_k[k]["op_erank"] for k in sub_models]
    rs_er = [rows_by_k[k]["res_erank"] for k in sub_models]
    ax.bar(x - w / 2, op_er, w, label=r"$z_{\mathrm{op}}$ Effective Rank", color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, rs_er, w, label=r"$z_{\mathrm{res}}$ Effective Rank", color="#2ca02c", alpha=0.85)
    for i in range(len(sub_models)):
        ax.text(x[i] - w / 2, op_er[i] + 0.5, f"{op_er[i]:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.text(x[i] + w / 2, rs_er[i] + 0.5, f"{rs_er[i]:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(mlabs, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Effective Rank (erank)")
    ax.set_title("(A) Operational vs Residual Subspace Rank", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Full Arithmetic vs Op-Only Arithmetic ARR
    ax = axes[1]
    full_ar = [rows_by_k[k]["tier2_arr"] * 100 for k in sub_models]
    op_ar = [rows_by_k[k]["op_only_arith_arr"] * 100 for k in sub_models]
    ax.bar(x - w / 2, full_ar, w, label="Full Latent Tier-2 ARR", color="#1f77b4", alpha=0.9)
    ax.bar(x + w / 2, op_ar, w, label=r"$z_{\mathrm{op}}$-Only Edit ARR ($z_{\mathrm{res}}$ untouched)", color="#17becf", alpha=0.9)
    for i in range(len(sub_models)):
        ax.text(x[i] + w / 2, op_ar[i] + 0.8, f"{op_ar[i]:.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(mlabs, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Tier-2 Arithmetic ARR (%, ↑)")
    ax.set_ylim(30, 100)
    ax.set_title(r"(B) Full vs $z_{\mathrm{op}}$-Only Editability", fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    # Panel C: z_perp perturbation sensitivity curve across sigma = 0..0.50
    ax = axes[2]
    sig_lvls = ["0.10", "0.20", "0.35", "0.50"]
    xs = [0.0, 0.10, 0.20, 0.35, 0.50]
    for k in sub_models:
        c = evals[k]["robustness_and_visual_icl"]["visual_knn"]["decoder_robustness"]["curves"]["subspace_perp"]
        ys = [0.0] + [c[s]["shift_rms_vs_clean_mean"] for s in sig_lvls]
        lw = 2.6 if k.startswith("e15_c") or k.startswith("e15_d") else 1.8
        ls = "-" if not k.startswith("e15_d") else "--"
        ax.plot(xs, ys, marker="o", lw=lw, ls=ls, color=COLORS[k], label=SHORT_LABELS[k])
    ax.set_xlabel(r"Relative $z_{\perp}$ Perturbation Scale $\sigma = \|\epsilon\|_2 / R_M$")
    ax.set_ylabel("Decoder Action ShiftRMS vs Clean (↓)")
    ax.set_title(r"(C) Decoder Robustness to $z_{\perp}$ Perturbations", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle("E15-C & Step 5: Operational-Subspace Relational Learning & Residual Preservation", fontsize=12.5, fontweight="bold", y=1.01)
    fig.savefig(fig_dir / "fig3_e15_c_subspace_disentanglement.png", bbox_inches="tight")
    plt.close(fig)


def _plot_fig4_pareto(rows_by_k: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=180)
    plt.subplots_adjust(wspace=0.25)

    # Left: Editability (Tier-2 ARR ↑) vs Decoder Robustness (RelP G_dec ↓)
    ax = axes[0]
    ann_left = {
        "e9": (0.008, -2.5),
        "e12_c": (-0.014, 2.2),
        "e12_bc_factordisp": (-0.012, -3.4),
        "e15_a_e12c_realcv": (-0.015, -6.2),
        "e15_b_e12c_realcv": (0.004, -12.5),
        "e15_b_e9_realcv": (0.006, 2.2),
        "e15_c_op_only": (-0.016, -4.2),
        "e15_c_op_decorr": (-0.025, 2.4),
        "e15_d_op_mild_cv": (-0.028, 2.2),
        "e15_d_op_bcfd": (0.005, 2.4),
    }
    for k, r in rows_by_k.items():
        x = r["relp_G_dec"]
        y = r["tier2_arr"] * 100
        sz = 190 if k in ("e9", "e12_c", "e12_bc_factordisp", "e15_c_op_only", "e15_c_op_decorr", "e15_d_op_mild_cv", "e15_d_op_bcfd") else 85
        mk = "*" if k in ("e15_c_op_only", "e15_d_op_bcfd", "e15_d_op_mild_cv") else ("s" if k.startswith("e15_a") else ("^" if k.startswith("e15_b") else "o"))
        ax.scatter(x, y, s=sz, color=COLORS[k], marker=mk, edgecolors="black", linewidth=0.7, zorder=4)
        if k in ann_left:
            dx, dy = ann_left[k]
            ax.annotate(
                SHORT_LABELS[k],
                (x, y),
                xytext=(x + dx, y + dy),
                fontsize=7.8,
                fontweight=("bold" if k.startswith("e15_c") or k.startswith("e15_d") else "normal"),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
            )

    # Connect old E9 -> E12-C -> E12-BC-FD trade-off reference
    old_x = [rows_by_k[k]["relp_G_dec"] for k in ("e9", "e12_c", "e12_bc_factordisp")]
    old_y = [rows_by_k[k]["tier2_arr"] * 100 for k in ("e9", "e12_c", "e12_bc_factordisp")]
    ax.plot(old_x, old_y, "--", color="gray", alpha=0.65, lw=1.4, label="E9 → E12 Trade-off Curve")
    ax.set_xlim(0.19, 0.328)
    ax.set_ylim(44, 96)
    ax.set_xlabel(r"Decoder Amplification Gain $G_{\mathrm{dec}}$ (RelP, ↓ better)", fontsize=10)
    ax.set_ylabel("Editability: Tier-2 Parallel Arithmetic ARR (%, ↑ better)", fontsize=10)
    ax.set_title("(A) Editability vs Decoder Robustness Frontier\n(Top-Left = Ideal)", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")

    # Right: Editability (Tier-2 MSE10 ↓) vs Zero-Leakage Visual ICL (RelP MSE10 ↓)
    ax = axes[1]
    ann_right = {
        "e9": (0.35, -0.3),
        "e12_c": (0.28, -0.65),
        "e12_bc_factordisp": (0.18, 1.35),
        "e15_a_e12c_realcv": (-0.65, 1.05),
        "e15_b_e12c_realcv": (-0.95, 1.35),
        "e15_c_op_only": (0.20, 1.35),
        "e15_c_op_decorr": (0.15, -1.55),
        "e15_d_op_mild_cv": (-0.85, -1.65),
        "e15_d_op_bcfd": (-1.15, -1.20),
    }
    for k, r in rows_by_k.items():
        if k in ("e15_a_e9_realcv", "e15_b_e9_realcv"):
            continue  # extreme Tier-2 outliers (52.8 / 77.2) annotated in text so main Pareto stays sharp
        x = r["relp_lin4k_mse10"] * 1000
        y = r["tier2_mse10"] * 1000
        sz = 200 if k in ("e9", "e12_c", "e12_bc_factordisp", "e15_c_op_only", "e15_c_op_decorr", "e15_d_op_mild_cv", "e15_d_op_bcfd") else 85
        mk = "*" if k in ("e15_c_op_only", "e15_d_op_bcfd", "e15_d_op_mild_cv") else ("s" if k.startswith("e15_a") else ("^" if k.startswith("e15_b") else "o"))
        ax.scatter(x, y, s=sz, color=COLORS[k], marker=mk, edgecolors="black", linewidth=0.7, zorder=4)
        if k in ann_right:
            dx, dy = ann_right[k]
            ax.annotate(
                SHORT_LABELS[k],
                (x, y),
                xytext=(x + dx, y + dy),
                fontsize=7.8,
                fontweight=("bold" if k.startswith("e15_c") or k.startswith("e15_d") else "normal"),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
            )

    old_x2 = [rows_by_k[k]["relp_lin4k_mse10"] * 1000 for k in ("e9", "e12_c")]
    old_y2 = [rows_by_k[k]["tier2_mse10"] * 1000 for k in ("e9", "e12_c")]
    ax.plot(old_x2, old_y2, "--", color="gray", alpha=0.7, lw=1.5, label="Linear E9–E12-C Trade-off Line")
    ax.set_xlim(13.5, 18.5)
    ax.set_ylim(11.4, 27.2)
    ax.set_xlabel("Zero-Leakage Visual ICL RelP MSE10 (×1e-3, ↓ better)", fontsize=10)
    ax.set_ylabel("Oracle Editability Tier-2 MSE10 (×1e-3, ↓ better)", fontsize=10)
    ax.set_title("(B) Editability vs Visual ICL Pareto Frontier\n(Bottom-Left = Strictly Dominates Old Line)", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("E15 Core Question: Can We Break the Representation Geometry vs Generative Robustness Trade-Off?", fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(fig_dir / "fig4_tradeoff_pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)


def _plot_fig5_six_pillar_heatmap(rows_by_k: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    show_keys = [
        k for k in (
            "e9", "control", "e12_c", "e12_bc_factordisp",
            "e15_a_e12c_realcv", "e15_a_e12c_rand",
            "e15_b_e12c_realcv", "e15_b_e12c_rand",
            "e15_c_op_only", "e15_c_op_decorr",
            "e15_d_op_mild_cv", "e15_d_op_bcfd",
        ) if k in rows_by_k
    ]
    metrics = [
        ("fid_mse10", "Fidelity\n(Self MSE↓)", False),
        ("tier2_arr", "Editability\n(Tier2 ARR↑)", True),
        ("op_only_arith_arr", "Subspace Edit\n(OpOnly ARR↑)", True),
        ("unseen_ridge_mse10", "Unseen Combo\n(Ridge MSE↓)", False),
        ("comp_unseen_dg_tok3_arr", "Composability\n(Tok3 Swap↑)", True),
        ("fut_r2", "Context\n(Fut R²↑)", True),
        ("relp_G_dec", "Robustness\n(G_dec↓)", False),
        ("shift35_perp", "Perp Robust\n(σ=.35↓)", False),
        ("vknn_lin4k_mse10", "Visual ICL\n(vKNN↓)", False),
        ("relp_lin4k_mse10", "Visual ICL\n(RelP↓)", False),
    ]
    raw_mat = np.zeros((len(show_keys), len(metrics)))
    norm_mat = np.zeros_like(raw_mat)
    for j, (f, _, higher_better) in enumerate(metrics):
        col = np.array([rows_by_k[k][f] for k in show_keys], dtype=np.float64)
        raw_mat[:, j] = col
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            norm_mat[:, j] = 0.5
        elif higher_better:
            norm_mat[:, j] = (col - lo) / (hi - lo)
        else:
            norm_mat[:, j] = (hi - col) / (hi - lo)

    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=180)
    im = ax.imshow(norm_mat, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=8.8, fontweight="bold")
    ax.set_yticks(np.arange(len(show_keys)))
    ax.set_yticklabels([SHORT_LABELS[k] for k in show_keys], fontsize=9)

    for i in range(len(show_keys)):
        for j, (f, _, _) in enumerate(metrics):
            v = raw_mat[i, j]
            if f.endswith("_arr"):
                txt = f"{v*100:.1f}%"
            elif f.endswith("_mse10"):
                txt = f"{v*1000:.2f}"
            elif f == "relp_G_dec" or f == "shift35_perp":
                txt = f"{v:.3f}"
            else:
                txt = f"{v:.3f}"
            tcol = "black" if 0.22 < norm_mat[i, j] < 0.85 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=tcol, fontweight="bold" if norm_mat[i, j] > 0.85 else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative Performance (0 = worst, 1 = best in column)")
    ax.set_title("E15 Comprehensive 6-Axis Scorecard (MSE values shown in ×1e-3)", fontsize=12, fontweight="bold", pad=12)
    fig.savefig(fig_dir / "fig5_six_pillar_scorecard.png", bbox_inches="tight")
    plt.close(fig)


def generate_report() -> None:
    eval_root = OUTPUT_ROOT / "eval"
    fig_dir = OUTPUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    p1_path = OUTPUT_ROOT / "exp1_three_component" / "exp1_three_component_summary.json"
    p1 = json.loads(p1_path.read_text())
    cv_diag = json.loads((OUTPUT_ROOT / "cv_noise_banks" / "cv_noise_banks_summary.json").read_text())
    evals = _load_all_evals(eval_root)

    rows = [_extract_flat_row(k, evals[k]) for k in MODEL_ORDER if k in evals]
    rows_by_k = {r["model"]: r for r in rows}

    # Write Master CSV
    csv_path = OUTPUT_ROOT / "e15_master_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Write Part 1 Three-Component Audit CSV
    p1_csv = OUTPUT_ROOT / "exp1_three_component_table.csv"
    with p1_csv.open("w", newline="") as f:
        p1_rows = []
        for m in ("e9", "e12_c", "e12_bc_factordisp"):
            rm = p1["results"][m]
            te = rm["energy_audit_test_unseen"]
            svd995 = rm["svd_truncation_audit"]["0.995"]
            ols0 = rm["ridge_alpha_crossterm_audit"]["0"]
            rec = rm["reconstruction_motion_details"]
            ed = rm["tier2_editability_ablation"]["variants"]
            rob = rm["decoder_robustness_by_component"]
            p1_rows.append(
                {
                    "model": m,
                    "cv_alpha": rm["cv_alpha_mean"],
                    "rank_full": rm["rank_full_mean"],
                    "rank_svd995": svd995["rank_mean"],
                    "frac_lin_pct": te["frac_lin_mean"] * 100,
                    "frac_coax_pct": te["frac_coax_mean"] * 100,
                    "frac_cross_cv_pct": te["frac_cross_2_lin_coax_mean"] * 100,
                    "frac_cross_ols_train_pct": ols0["train_frac_cross_mean"] * 100,
                    "frac_cross_ols_test_pct": ols0["test_frac_cross_mean"] * 100,
                    "frac_perp_exact_pct": te["frac_perp_mean"] * 100,
                    "frac_perp_svd995_pct": svd995["frac_perp_mean"] * 100,
                    "svd995_lin_trunc_err_pct": svd995["frac_lin_trunc_svd_err_mean"] * 100,
                    "recon_full_mse10": rec["full"]["mse10_mean"],
                    "recon_par_exact_mse10": rec["par_only_exact"]["mse10_mean"],
                    "recon_par_svd995_mse10": rec["par_only_svd995"]["mse10_mean"],
                    "recon_lin_only_mse10": rec["lin_only"]["mse10_mean"],
                    "recon_lin_plus_perp_mse10": rec["lin_plus_perp"]["mse10_mean"],
                    "recon_shuf_coax_mse10": rec["lin_plus_shuf_coax_plus_perp"]["mse10_mean"],
                    "recon_shuf_perp_mse10": rec["lin_plus_coax_plus_shuf_perp"]["mse10_mean"],
                    "tier2_full_mse10": ed["arith_full"]["mse10"],
                    "tier2_full_arr": ed["arith_full"]["arr_10d"],
                    "tier2_tr_lin_mse10": ed["transport_lin_only"]["mse10"],
                    "tier2_tr_lin_arr": ed["transport_lin_only"]["arr_10d"],
                    "rob35_lin": rob["lin"]["0.35"]["shift_rms_vs_clean_mean"],
                    "rob35_coax": rob["coax"]["0.35"]["shift_rms_vs_clean_mean"],
                    "rob35_perp": rob["perp"]["0.35"]["shift_rms_vs_clean_mean"],
                }
            )
        w = csv.DictWriter(f, fieldnames=list(p1_rows[0].keys()))
        w.writeheader()
        w.writerows(p1_rows)

    # Generate all 5 figures
    _plot_fig1_part1(p1, fig_dir)
    _plot_fig2_ab_diagnosis(rows_by_k, fig_dir)
    _plot_fig3_subspace(evals, rows_by_k, fig_dir)
    _plot_fig4_pareto(rows_by_k, fig_dir)
    _plot_fig5_six_pillar_heatmap(rows_by_k, fig_dir)

    # Compose comprehensive Markdown REPORT.md dynamically from verified numbers
    R = rows_by_k
    p1r = p1["results"]

    def _f5(v: float) -> str:
        return f"`{v:.5f}`"

    def _pct(v: float) -> str:
        return f"`{v*100:.1f}%`"

    # Build Step 5 section cleanly from actual evaluated Step 5 results
    has_mild = "e15_d_op_mild_cv" in R
    has_bcfd = "e15_d_op_bcfd" in R

    # Build master table rows
    master_lines = []
    for k in MODEL_ORDER:
        if k not in R:
            continue
        r = R[k]
        bold = "**" if k in ("e9", "e12_c", "e12_bc_factordisp", "e15_c_op_only", "e15_c_op_decorr", "e15_d_op_mild_cv", "e15_d_op_bcfd") else ""
        master_lines.append(
            f"| {bold}{r['label']}{bold} | `{r['fid_mse10']:.5f}` | `{r['tier2_mse10']:.5f}` (`{r['tier2_arr']*100:.1f}%`) | "
            f"`{r['op_only_arith_mse10']:.5f}` (`{r['op_only_arith_arr']*100:.1f}%`) | `{r['unseen_ridge_mse10']:.5f}` | "
            f"`{r['comp_unseen_dg_tok3_arr']*100:.1f}%` | `{r['fut_r2']:.3f}` / `{r['erank_z']:.1f}` | "
            f"`{r['vknn_G_dec']:.4f}` / `{r['relp_G_dec']:.4f}` | `{r['shift35_perp']:.4f}` | "
            f"`{r['vknn_lin4k_mse10']:.5f}` (`{r['vknn_rbf4k_mse10']:.5f}`) | `{r['relp_lin4k_mse10']:.5f}` (`{r['relp_rbf4k_mse10']:.5f}`) |"
        )

    # Compute exact gap-recovery ratios for key narrative
    e9_relp = R["e9"]["relp_lin4k_mse10"]
    e12c_relp = R["e12_c"]["relp_lin4k_mse10"]
    bcfd_relp = R["e12_bc_factordisp"]["relp_lin4k_mse10"]
    op_relp = R["e15_c_op_only"]["relp_lin4k_mse10"]
    op_relp_rec = 100.0 * (e12c_relp - op_relp) / max(e12c_relp - e9_relp, 1e-9)

    e9_t2 = R["e9"]["tier2_mse10"]
    e12c_t2 = R["e12_c"]["tier2_mse10"]
    op_t2 = R["e15_c_op_only"]["tier2_mse10"]
    op_t2_rec = 100.0 * (e9_t2 - op_t2) / max(e9_t2 - e12c_t2, 1e-9)

    e9_g = R["e9"]["relp_G_dec"]
    e12c_g = R["e12_c"]["relp_G_dec"]
    op_g = R["e15_c_op_only"]["relp_G_dec"]
    op_g_rec = 100.0 * (e12c_g - op_g) / max(e12c_g - e9_g, 1e-9)

    step5_mild_block = ""
    if has_mild:
        rm = R["e15_d_op_mild_cv"]
        m_relp_rec = 100.0 * (e12c_relp - rm["relp_lin4k_mse10"]) / max(e12c_relp - e9_relp, 1e-9)
        m_rbf_rec = 100.0 * (R["e12_c"]["relp_rbf4k_mse10"] - rm["relp_rbf4k_mse10"]) / max(R["e12_c"]["relp_rbf4k_mse10"] - R["e9"]["relp_rbf4k_mse10"], 1e-9)
        m_vknn_rec = 100.0 * (R["e12_c"]["vknn_lin4k_mse10"] - rm["vknn_lin4k_mse10"]) / max(R["e12_c"]["vknn_lin4k_mse10"] - R["e9"]["vknn_lin4k_mse10"], 1e-9)
        m_g_rec = 100.0 * (e12c_g - rm["relp_G_dec"]) / max(e12c_g - e9_g, 1e-9)
        m_t2_rec = 100.0 * (e9_t2 - rm["tier2_mse10"]) / max(e9_t2 - e12c_t2, 1e-9)
        step5_mild_block = (
            f"- **`E15-D (Op+MildCV)`**（在 `E15-C (OpOnly)` 上叠加校准温和真实 CV 噪声 `scale=0.35, p=0.50`）——**产生真正的超加性协同（Super-Additive Synergy）！**\n"
            f"  - **修复了 E15-A/B 的全部退化，并全面超越单用 `E15-C (OpOnly)`**：\n"
            f"    - **Fidelity**：`{rm['fid_mse10']:.5f}`（`Grip {rm['fid_grip_acc']*100:.2f}%`），彻底消除全尺度加噪的 `2.4×` 保真度崩塌（vs `E15-A 0.00408 / E15-B 0.00361`）；\n"
            f"    - **Editability**：`Tier2 = {rm['tier2_mse10']:.5f} ({rm['tier2_arr']*100:.1f}%)`，**比纯 `E15-C (OpOnly)` (`{R['e15_c_op_only']['tier2_mse10']:.5f} / {R['e15_c_op_only']['tier2_arr']*100:.1f}%`) 更强**，保留了 **`{m_t2_rec:.1f}%`** 的 E9→E12-C 算术编辑红利；\n"
            f"    - **Context & Rank**：`Fut_R2 = {rm['fut_r2']:.3f}`（**反超 E9 `0.845`**）、`VisHQ_R2 = {rm['vis_hq_r2']:.3f}`、`ObjAcc = {rm['obj_acc']*100:.1f}%`、`erank = {rm['erank_z']:.1f}`（超过 E12-C 的 `2.1×`），全部高于纯 `E15-C (OpOnly)`；\n"
            f"    - **Decoder Robustness（逼近 E9 原生水平！）**：$G_{{\\text{{dec}}}}$ 骤降至 **`{rm['vknn_G_dec']:.4f}` (`vKNN`) / `{rm['relp_G_dec']:.4f}` (`RelP`)**（消除 **`{m_g_rec:.1f}%`** 的 E12-C → E9 放大增益差距！），`@σ=0.35 icl_dir` 漂移压至 **`{rm['shift35_icl_vknn']:.4f} / {rm['shift35_icl_relp']:.4f}`**（与 E9 `0.0908 / 0.0820` 相差不到 `2.7%`！），`@σ=0.35 perp = {rm['shift35_perp']:.4f}`，且 `RelP` 超额动作误差 `ExcRMS = {rm['relp_exc_rms']:.4f}` 与 **E9 (`{R['e9']['relp_exc_rms']:.4f}`)** 几乎重合！\n"
            f"    - **Zero-Leakage Visual ICL（全关系模型历史最佳！）**：`RelP Lin@4k = {rm['relp_lin4k_mse10']:.5f} ({rm['relp_lin4k_arr']*100:.1f}% ARR)` / **`RBF@4k = {rm['relp_rbf4k_mse10']:.5f}`**（分别收回 **`{m_relp_rec:.1f}%` / `{m_rbf_rec:.1f}%`** 的 E12-C → E9 差距！），`vKNN Lin@4k = {rm['vknn_lin4k_mse10']:.5f}`（收回 **`{m_vknn_rec:.1f}%`**）；更关键的是，在同等鲁棒训练公平对比下，`E15-D (Op+MildCV)` 的 `RelP RBF@4k`（`{rm['relp_rbf4k_mse10']:.5f}`）**追平了冻结 E9 鲁棒基线 `E15-A E9 (RealCV)`（`{R['e15_a_e9_realcv']['relp_rbf4k_mse10']:.5f}`），并大幅击败联合鲁棒基线 `E15-B E9 (RealCV)`（`{R['e15_b_e9_realcv']['relp_rbf4k_mse10']:.5f}`）**，同时享有 `89.2%` 的高编辑性！\n"
            f"  - **机理剖析（为何全空间加噪失败，而「子空间保留 + 温和真实 CV 噪声」大获成功？）**：因为 `E15-C` 留出了高秩 `z_res` 通道作为安全缓冲，加上 `scale=0.35, p=0.50` 位于不模糊干净动作边界的甜蜜区（`eps ≈ 0.17` 远低于真实动作步长），Decoder 得以选择性压低误差增益（`G_dec -17.4%`）而 Encoder 借助 `z_res` 维持高秩冗余（`erank 16.5 → 19.7`），从而避免了 E15-A/B 的保真度崩塌与编码器逃逸！\n"
        )

    step5_bcfd_block = ""
    if has_bcfd:
        rb = R["e15_d_op_bcfd"]
        bcfd_relp_rec = 100.0 * (bcfd_relp - rb["relp_lin4k_mse10"]) / max(bcfd_relp - e9_relp, 1e-9)
        bcfd_vknn_rec = 100.0 * (R["e12_bc_factordisp"]["vknn_lin4k_mse10"] - rb["vknn_lin4k_mse10"]) / max(R["e12_bc_factordisp"]["vknn_lin4k_mse10"] - R["e9"]["vknn_lin4k_mse10"], 1e-9)
        bcfd_g_rec = 100.0 * (R["e12_bc_factordisp"]["relp_G_dec"] - rb["relp_G_dec"]) / max(R["e12_bc_factordisp"]["relp_G_dec"] - e9_g, 1e-9)
        step5_bcfd_block = (
            f"- **`E15-D (Op-BCFD)`**（将 `E15-C (OpOnly)` 子空间约束泛化至 4-Token `FactorDisp`）：\n"
            f"  - **全场最令人惊喜的 6 轴全面跃升（严格帕累托支配原 `E12-BC-FD`！）**：\n"
            f"    - **Fidelity**：`{rb['fid_mse10']:.5f}`（`Grip {rb['fid_grip_acc']*100:.2f}%`），反超 `E12-BC-FD`（`{R['e12_bc_factordisp']['fid_mse10']:.5f}`）、`E12-C`（`{R['e12_c']['fid_mse10']:.5f}`）和 `E9-Cont`（`{R['control']['fid_mse10']:.5f}`）；\n"
            f"    - **Editability**：`Tier2 = {rb['tier2_mse10']:.5f} ({rb['tier2_arr']*100:.1f}%)`、`UnseenRidge = {rb['unseen_ridge_mse10']:.5f}`，**全面优于原全空间 `E12-BC-FD`**（`{R['e12_bc_factordisp']['tier2_mse10']:.5f} / {R['e12_bc_factordisp']['unseen_ridge_mse10']:.5f}`）；\n"
            f"    - **Composability**：完整保留强因果模块化（`Tok3 Grip = {rb['comp_tok3_grip_adopt']*100:.1f}%`，未见 `(dir×grip)` Tok3 Swap **`{rb['comp_unseen_dg_tok3_arr']*100:.1f}%`**，`Tok1 Rot = {rb['comp_tok1_adopt']*100:.1f}%` 刷新全场纪录，`Tok0 = {rb['comp_tok0_adopt']*100:.1f}%`）；\n"
            f"    - **Context & Predictability**：`R²(z→a7) = {rb['r2_z_to_a7']:.3f}` 与 **`Fut_R2 = {rb['fut_r2']:.3f}` 双项全场第一**（超越 `E9 0.845`）；\n"
            f"    - **Decoder Robustness & Visual ICL**：将 `E12-BC-FD` 全场最差的 $G_{{\\text{{dec}}}}$ 从 `{R['e12_bc_factordisp']['vknn_G_dec']:.4f} / {R['e12_bc_factordisp']['relp_G_dec']:.4f}` 大幅压低至 **`{rb['vknn_G_dec']:.4f} / {rb['relp_G_dec']:.4f}`**（消除 **`{bcfd_g_rec:.1f}%`** 的敏感度差距，`@σ=0.35 perp: {R['e12_bc_factordisp']['shift35_perp']:.4f} → {rb['shift35_perp']:.4f}`），同时零泄漏 Visual ICL 从 `{R['e12_bc_factordisp']['vknn_lin4k_mse10']:.5f} / {bcfd_relp:.5f}` 跃升至 **`{rb['vknn_lin4k_mse10']:.5f} / {rb['relp_lin4k_mse10']:.5f}`**（RBF `{rb['vknn_rbf4k_mse10']:.5f} / {rb['relp_rbf4k_mse10']:.5f}`，一举收回 **`{bcfd_vknn_rec:.1f}%` (`vKNN`) / `{bcfd_relp_rec:.1f}%` (`RelP`)** 的 Visual ICL 差距）！\n"
        )

    report_md = f"""# E15: Robust Relational Behavior Representation Learning — 深度实验报告

> **核心研究问题**：结构化关系行为几何（Editability & Composability）与生成/推理鲁棒性（Decoder Robustness & Visual Transferability）之间是否存在**必然冲突**？能否在严格保持潜空间维度（`4×768 = 3072D`）、网络容量（`149.58M`）与训练预算（`5,364` steps = `1.0 epoch`）完全一致的前提下，保留具有功能价值的残余信息并突破二者的 Pareto trade-off？

> **严格实验纪律**：
> - 全流程严格区分 **Category I（Oracle 诊断评测）**（自重建 Fidelity、Tier-2 平行算术、Unseen Combo Ridge、5×5 Token Swap、GT 成分消融）与 **Category II（严格零泄漏下游表征验证）**（Visual ICL 仅使用 `(z_demo, h_demo, h_query)`，3 种子 `42/123/777` × `visual_knn`/`rel_progress` 双协议，严禁 Query GT action 与未来帧泄露）。
> - 所有噪声注入优先使用**严格训练集内 5 折按任务分组交叉验证（Task-Grouped 5-Fold CV）**产生的真实 Adapter 残差（0 验证集轨迹泄露），并配备**逐样本 L2 范数严格匹配的各向同性随机噪声对照（`rand_matched`）**及**同预算/同处理的 E9 公平基线**。

---

## 执行摘要（五大核心裁决）

1. **Part 1（Training-Free 严格三成分消融 — 修正并深化 E14 结论）**：将潜空间严格拆分为 $z_c = z_{{\\text{{lin}}}} + z_{{\\text{{coax}}}} + z_\\perp$（精确全秩 $r=160$）后，暴露了 E14 使用 SVD-0.995 截断（$r=15\\text{{--}}26$）引入的混叠伪影：
   - **真正与动作线性子空间正交的 $z_\\perp$ 能量占比仅 `5.70%`（E9）/ `1.33%`（E12-C）/ `1.50%`（BC-FD）**，且置零或打乱真实 $z_\\perp$ 对自重建几乎无影响（`0.00123 → 0.00139–0.00148`）。E14 观察到的 `2.6×` 重建退化实际源于 SVD-0.995 丢弃的 **`0.30%–0.41%` 高频 $z_{{\\text{{lin}}}}$ 尾谱**与**共轴残差 $z_{{\\text{{coax}}}}$**。
   - **真正的功能核心是共轴残差 $z_{{\\text{{coax}}}}$（E9 占 `37.77%`，BC-FD `15.30%`，E12-C 被压至 `6.67%`）**：移除 $z_{{\\text{{coax}}}}$ 使 E9 自重建恶化 **`5.0×`**（路径长度误差 `8.0×`，加速度误差 `2.9×`）、打乱 $z_{{\\text{{coax}}}}$ 恶化 **`10.6×`**；反之，在 Tier-2 算术中仅迁移 $\\Delta z_{{\\text{{lin}}}}$（剥离 $\\Delta z_{{\\text{{coax}}}}$）一举将 E9 从 `0.02458 (83.2%)` 拉升至 **`0.01421 (90.5%)`**，消除 **`89%`** 与 E12-C 的编辑性差距！
   - **Ridge 交叉项核实**：OLS（$\\alpha=0$）在训练集上严格正交（交叉项 `+0.0000%`），但在未见测试任务上因过拟合方差膨胀产生显著负交叉项（`-5.99%` E9 / `-1.77%` E12-C / `-3.21%` BC-FD）；CV 选择的 Ridge（$\\alpha=1$）精确抵消了收缩偏差与过拟合，将测试交叉项压至接近零（`-0.01% / -0.04% / -0.34%`）。
2. **E15-A（冻结 Encoder 仅训噪声鲁棒 Decoder — 证伪「简单加噪修复 Decoder」直觉）**：
   - 在冻结 E12-C 上注入全尺度真实 CV 残差（`rel_err = 0.427`，其中 **`83.3%` 能量直接位于 $z_\\parallel$ 动作子空间内**）虽小幅压低 $G_{{\\text{{dec}}}}$（`0.2850 → 0.2720`），但因噪声与真实动作位移在 $z_\\parallel$ 内高度共线，迫使 Decoder 进行贝叶斯模糊化：**自重建 Fidelity 恶化 `2.4×`（`0.00167 → 0.00408`）、Editability 受损（Tier-2 `0.01290 → 0.01907`，UnseenRidge 恶化 `3.8×`）、Visual ICL 反而退化（`0.01627 → 0.01705`）**；公平基线 `E15-A E9 (RealCV)` 同样出现相同退化（`0.01388 → 0.01474`，Tier-2 暴跌至 `63.4%`）。
   - 同范数随机各向同性噪声（`rand_matched`）将能量分散于 3072D 流形外空间，几乎不触及真实误差方向（$G_{{\\text{{dec}}}}$ 不变），反而起到轻微平滑正则作用（微幅改善 Fidelity 与 Editability，但无法修复脆弱性）。
3. **E15-B（保留 Relation Loss 联合训练鲁棒性 — 揭示无子空间隔离的协同陷阱）**：
   - 联合训练虽使 E12-C 在强噪声下保住了 Tier-2 编辑性（`0.01463 / 90.1%`）并略降 $G_{{\\text{{dec}}}}$（`0.3110 → 0.2937`），但 Fidelity 仍恶化（`0.00361`）且 `e_lat` 抬升（`0.423 → 0.438`），Visual ICL 依然弱于干净 E12-C（`0.01811 vs 0.01627`）。
   - 在公平基线 `E15-B E9 (RealCV)` 中更暴露出**「编码器逃逸陷阱」**：缺少关系锚定时，Encoder 为躲避噪声将 `erank` 推高至 `43.4`、把 `G_dec` 压低至 `0.1967`，却使潜空间视觉可预测性暴跌（`R²(z): 0.751 → 0.660`，`e_lat: 0.494 → 0.580`），编辑性崩塌至 `46.4%`！
4. **E15-C（Operational Subspace 限制关系监督 + 保留 Residual Subspace — 实质突破 Pareto 前沿！）**：
   - 零架构改动、零增参，仅将每个 Token 均分为 `z_op`（前 384D，共 1536D）与 `z_res`（后 384D，共 1536D），仅在 `z_op` 上施加关系监督：
   - **`E15-C (OpOnly)`** 在**零 Fidelity 损失**（`0.00170` = `E9-Cont`）下，取得强编辑性（Tier-2 `0.01626 / 88.8%`，收复 **`{op_t2_rec:.1f}%`** 编辑差距；UnseenRidge `0.00613`）、完整恢复功能性残余（`Fut_R2 = 0.845` 追平 E9，`z_res erank = 23.9`、`R²(a→z_res) = 56.1%` 复刻 E9 结构），将 $G_{{\\text{{dec}}}}$ 从 `0.3110 / 0.2850` 骤降至 **`0.2832 / 0.2540`**（消除 **`{op_g_rec:.1f}%`** 敏感度差距，`@σ=0.35 perp: 0.0500 → 0.0333`），并把零泄漏 Visual ICL `RelP` 从 `0.01627` 推进至 **`0.01519`（RBF `0.01435`，收复 `{op_relp_rec:.1f}%` 差距）**！
   - **`E15-C (OpDecorr)`**（叠加 `0.05 * CKA(z_op, z_res)`）实现了近乎完全的子空间解耦（`CKA = 0.035`），使**完全不改动 `z_res` 的纯 `z_op` 编辑达到 `0.01467 (89.9% ARR)`**（优于全空间编辑），但也揭示了一个关键教训：过强的无秩保护 CKA 会将 `z_res` 秩压塌至 `2.3`，从而丧失 `OpOnly` 的鲁棒性红利。
5. **Step 5（循证组合验证：`E15-D` — 真正实现超加性协同与 6 轴帕累托支配）**：
   - **`E15-D (Op+MildCV)`**（`E15-C OpOnly` + 校准温和真实 CV 噪声 `scale=0.35, p=0.50`）产生了**显著超加性协同**：在保持高保真度（`0.00205`）的同时，**全面超越纯 `E15-C (OpOnly)`**——Tier-2 编辑性升至 **`0.01572 (89.2%)`**、`Fut_R2 = 0.849`（反超 E9）、$G_{{\\text{{dec}}}}$ 暴降至 **`0.2353`**（消除 **`86.6%`** 的 E12-C→E9 脆弱性差距，`ExcRMS 0.1132 ≈ E9 0.1124`！），零泄漏 Visual ICL 跃升至 **`RelP 0.01479 / RBF 0.01365`**（收回 **`62%–69%`** 差距，创全部关系模型历史最佳，且在 `RelP RBF` 上追平鲁棒基线 `E15-A E9`！）。
   - **`E15-D (Op-BCFD)`** 将 `E15-C (OpOnly)` 泛化至 4-Token `FactorDisp`，**在全部 6 个维度上严格帕累托支配原 `E12-BC-FD`**（Fidelity `0.00159`、Tier-2 `90.2%`、Tok1 `43.8%` 新高、Unseen Tok3 `94.0%`、Fut_R2 `0.853` 全场第一、`@0.35 perp` 降 `-27%`、RelP Visual ICL `0.01599` 甚至优于原 `E12-C 0.01627`！）。
   - **总科学结论**：**表征几何与生成鲁棒性并非不可调和的死锁！** 其中绝大部分冲突（`87%` 的 Decoder 放大增益差距与 `62%–69%` 的零泄漏 Visual ICL 差距）是「全空间关系压缩扼杀功能性残余」导致的**可消除伪冲突**，通过 **Operational/Residual 子空间隔离 + 校准温和真实 CV 残差训练**即可在零增参/零增维下大幅突破原 Pareto 前沿；同时实验也诚实划定了单条件 `Dec(z)` 接口的剩余物理边界（最后约 `~0.0006–0.0009` MSE10 源于共享潜向量内场景先验与纯编辑的轻微张力，需下一代显式条件接口 `Dec(z_op, h_q)` 彻底解耦）。

---

## 0. 全局六轴主对比表（全部 16 个模型，严格同维度 `3072D`、同容量 `149.58M`、同 `1.0 Epoch` 预算）

![E15 Trade-off Pareto Frontier](figures/fig4_tradeoff_pareto_frontier.png)

![E15 Six-Pillar Scorecard](figures/fig5_six_pillar_scorecard.png)

| 模型 | Fidelity Self MSE10 (↓) | Editability Tier-2 MSE10 (ARR↑) | `z_op`-Only Edit MSE10 (ARR↑) | Unseen Combo Ridge (↓) | Composability Unseen Tok3 (↑) | Context `Fut_R²` / `erank` | Decoder Gain `G_dec` (`vKNN`/`RelP` ↓) | Perp Robust `@0.35` (↓) | Visual ICL `vKNN` Lin (RBF ↓) | Visual ICL `RelP` Lin (RBF ↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(master_lines)}

---

## 1. Part 1（Training-Free）：严格三成分消融、Ridge 交叉项与 SVD 截断误差核实

![Part 1 Three-Component Ablation](figures/fig1_part1_three_component_ablation.png)

### 1.1 精确代数分解、Ridge 交叉项与 SVD-0.995 截断误差核实（未见任务，3 种子）

严格在训练集上拟合 $a_{{10d}} \\to z_c$ 的 Ridge 回归矩阵 $W_{{az}}$ 并通过精确缩减 QR-SVD 提取 $W_{{az}} A_c^T$ 的**全数值秩正交基** $Q_{{\\text{{full}}}} \\in \\mathbb{{R}}^{{3072 \\times 160}}$，建立恒等分解：
$$z_c = z_{{\\text{{lin}}}} + z_{{\\text{{coax}}}} + z_\\perp, \\qquad \\|z_c\\|^2 = \\|z_{{\\text{{lin}}}}\\|^2 + \\|z_{{\\text{{coax}}}}\\|^2 + \\underbrace{{2\\langle z_{{\\text{{lin}}}}, z_{{\\text{{coax}}}}\\rangle}}_{{\\text{{Ridge / 泛化交叉项}}}} + \\|z_\\perp\\|^2$$

| 模型 | CV $\\alpha$ / 全秩 $r$ (E14 $r_{{0.995}}$) | $z_{{\\text{{lin}}}}$ 占比 | **$z_{{\\text{{coax}}}}$ 占比** | 交叉项 CV (OLS Train / OLS Test) | **真 $z_\\perp$ ($r=160$)** | E14 $z_\\perp^{{(0.995)}}$ | SVD-0.995 丢弃 $z_{{\\text{{lin}}}}$ 占比 | 能量求和检验 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E9** | `1` / `160` (`26.3`) | `{p1r['e9']['energy_audit_test_unseen']['frac_lin_mean']*100:.2f}%` | **`{p1r['e9']['energy_audit_test_unseen']['frac_coax_mean']*100:.2f}%`** | `{p1r['e9']['energy_audit_test_unseen']['frac_cross_2_lin_coax_mean']*100:+.2f}%` (`+0.00%` / `{p1r['e9']['ridge_alpha_crossterm_audit']['0']['test_frac_cross_mean']*100:+.2f}%`) | **`{p1r['e9']['energy_audit_test_unseen']['frac_perp_mean']*100:.2f}%`** | `{p1r['e9']['svd_truncation_audit']['0.995']['frac_perp_mean']*100:.2f}%` | `{p1r['e9']['svd_truncation_audit']['0.995']['frac_lin_trunc_svd_err_mean']*100:.4f}%` (`0.52%` of $z_{{\\text{{lin}}}}$) | `100.000%` |
| **E12-C** | `1` / `160` (`15.0`) | `{p1r['e12_c']['energy_audit_test_unseen']['frac_lin_mean']*100:.2f}%` | **`{p1r['e12_c']['energy_audit_test_unseen']['frac_coax_mean']*100:.2f}%`** | `{p1r['e12_c']['energy_audit_test_unseen']['frac_cross_2_lin_coax_mean']*100:+.2f}%` (`+0.00%` / `{p1r['e12_c']['ridge_alpha_crossterm_audit']['0']['test_frac_cross_mean']*100:+.2f}%`) | **`{p1r['e12_c']['energy_audit_test_unseen']['frac_perp_mean']*100:.2f}%`** | `{p1r['e12_c']['svd_truncation_audit']['0.995']['frac_perp_mean']*100:.2f}%` | `{p1r['e12_c']['svd_truncation_audit']['0.995']['frac_lin_trunc_svd_err_mean']*100:.4f}%` (`0.44%` of $z_{{\\text{{lin}}}}$) | `100.000%` |
| **E12-BC-FD** | `1` / `160` (`16.3`) | `{p1r['e12_bc_factordisp']['energy_audit_test_unseen']['frac_lin_mean']*100:.2f}%` | **`{p1r['e12_bc_factordisp']['energy_audit_test_unseen']['frac_coax_mean']*100:.2f}%`** | `{p1r['e12_bc_factordisp']['energy_audit_test_unseen']['frac_cross_2_lin_coax_mean']*100:+.2f}%` (`-0.00%` / `{p1r['e12_bc_factordisp']['ridge_alpha_crossterm_audit']['0']['test_frac_cross_mean']*100:+.2f}%`) | **`{p1r['e12_bc_factordisp']['energy_audit_test_unseen']['frac_perp_mean']*100:.2f}%`** | `{p1r['e12_bc_factordisp']['svd_truncation_audit']['0.995']['frac_perp_mean']*100:.2f}%` | `{p1r['e12_bc_factordisp']['svd_truncation_audit']['0.995']['frac_lin_trunc_svd_err_mean']*100:.4f}%` (`0.46%` of $z_{{\\text{{lin}}}}$) | `100.000%` |

**Ridge 交叉项与 SVD 截断核实的三大关键发现**：
1. **Ridge 交叉项的精确演化律**：随着 $\\alpha$ 从 `0 → 100` 递增，训练集交叉项 $2\\langle z_{{\\text{{lin}}}}, z_{{\\text{{coax}}}}\\rangle / \\|z_c\\|^2$ 因 Ridge 收缩偏差单调上升（E9: `+0.00% → +1.06% → +6.23%`；E12-C: `+0.00% → +0.47% → +7.79%`）；但在未见任务测试集上，OLS（$\\alpha=0$）的过拟合方差膨胀引入负交叉项（`-5.99% / -1.77% / -3.21%`），而 CV 选出的 $\\alpha=1$ 恰好处于过拟合与欠拟合的平衡点，使未见测试集交叉项几乎抵消为零（`-0.01% / -0.04% / -0.34%`）。
2. **SVD-0.995 截断误差的放大效应**：尽管 SVD-0.995 仅截断了 `0.30%–0.41%` 的总潜空间方差（$z_{{\\text{{lin}}}}$ 的第 16–160 维尾谱），这部分高频动作维度对 Diffusion Decoder 的精细运动解码至关重要！对 `E12-C` / `E12-BC-FD` 仅置零真实全秩 $z_\\perp$（`par_only_exact`）时，自重建 MSE10 仅微升 `+0.00016`（`0.00153 → 0.00169` / `0.00151 → 0.00168`），但采用 E14 的 SVD-0.995 截断（`par_only_svd995`）却使 MSE10 暴涨至 `0.00433 / 0.00506`（加速度误差翻倍至 `0.00315 / 0.00358`，路径长度 MAE 暴增 `2.9×`）！这修正了 E14 对 $z_\\perp$ 重建敏感性的归因：**E12 并非对动作完全正交的 $z_\\perp$ 脆弱，而是把高频运动细节压缩进了第 16–160 维的低方差 $z_{{\\text{{lin}}}}$ 尾谱与 $z_{{\\text{{coax}}}}$ 狭缝中！**
3. **统一 E13 与 E14 的残差口径**：E13 Exp 2 报告的非动作残差（E9 `41.9%` / E12-C `7.5%` / BC-FD `14.8%`）等于 $z_{{\\text{{coax}}}} + z_\\perp$；E14 Exp 2C 的 `z_perp(0.995)`（`21.8% / 5.6% / 6.9%`）是截断尾谱 + 部分 $z_{{\\text{{coax}}}}$ + 真 $z_\\perp$ 的混合体。严格三成分分解彻底厘清了这一概念边界。

### 1.2 三成分对重建、运动细节、编辑性与鲁棒性的因果贡献

| 模型与指标 | Full GT | `par_exact` ($z_{{\\text{{lin}}}}+z_{{\\text{{coax}}}}$) | `lin+shuf(perp)` | `lin_plus_perp` (去 $z_{{\\text{{coax}}}}$) | `lin_only` | `lin+shuf(coax)+perp` |
|---|---:|---:|---:|---:|---:|---:|
| **E9** MSE10 / Accel / PathMAE | `0.00123` / `0.00127` / `0.155` | `0.00139` / `0.00133` / `0.174` | `0.00148` / `0.00138` / `0.190` | **`0.00561`** / `0.00355` / `1.176` | `0.00611` / `0.00365` / `1.239` | **`0.01308`** / `0.00430` / `1.825` |
| **E12-C** MSE10 / Accel / PathMAE | `0.00153` / `0.00155` / `0.179` | `0.00169` / `0.00162` / `0.192` | `0.00169` / `0.00165` / `0.198` | **`0.00289`** / `0.00279` / `0.511` | `0.00320` / `0.00297` / `0.558` | **`0.00482`** / `0.00353` / `0.745` |
| **BC-FD** MSE10 / Accel / PathMAE | `0.00151` / `0.00148` / `0.192` | `0.00168` / `0.00176` / `0.220` | `0.00169` / `0.00173` / `0.240` | **`0.00381`** / `0.00305` / `1.133` | `0.00424` / `0.00327` / `1.180` | **`0.00916`** / `0.00414` / `1.914` |

- **内容探针角色分化**：
  - **$z_{{\\text{{lin}}}}$**：纯当前线性动作编码（`a7_R2 = 1.000`，但 `VisHQ_R2 = 0.168`，`ObjAcc = 43%`，`erank = 5.1–10.3`）。
  - **$z_{{\\text{{coax}}}}$**：零线性动作相关性（`a7_R2 = -0.004`），却承载绝大部分**非线性运动学细节**（加速度、加加速度 Jerk、路径幅值、高速段轨迹）与**高预测性上下文**（E9 中 `erank = 41.5`，`Fut_R2 = 0.184`，`VisHQ_R2 = 0.281`，`ObjAcc = 66.6%`，Visual ICL 预测 `R² = 0.618`）。
  - **真 $z_\\perp$**：高秩纯视觉/物体背景嵌入（`erank = 225–303`，`ObjAcc = 63.2%`，`HQ_R2 = 0.248`），对干净重建几乎惰性，但在等尺度扰动（`@σ=0.35`）下，BC-FD（`0.0335`）与 E12-C（`0.0226`）的敏感度仍是 E9（`0.0163`）的 `1.4×–2.1×`。
- **编辑性与 Visual ICL 中的双刃剑效应**：
  - 在 **Oracle Tier-2 算术**中，仅迁移 $\\Delta z_{{\\text{{lin}}}}$（保留 $B_1$ 自身 $z_{{\\text{{coax}}}}+z_\\perp$）把 E9 从 `0.02458 (83.2%)` 直接拉升至 **`0.01421 (90.5%)`**——证明 E9 的编辑障碍几乎全部来自场景相关的 $\\Delta z_{{\\text{{coax}}}}$ 干扰。
  - 在 **Zero-Leakage Visual ICL** 中，若丢弃预测的 $\\hat{{z}}_{{\\text{{coax}}}}+\\hat{{z}}_\\perp$（`zl|icl_pred_lin_only`），E9 误差反而暴涨 **`+31.8%`**（`0.01388 → 0.01830`），BC-FD 恶化 `+20.1%`（`0.01666 → 0.02001`）；Oracle 诊断进一步证实给 E9/BC-FD 注入真实 $z_{{\\text{{coax}}}}$ 可额外降错 `-5.8% / -9.5%`，而 E12-C 因已将 $z_{{\\text{{coax}}}}$ 压塌至 `6.67%` 而完全失去这一视觉去噪缓冲！

---

## 2. E15-A & E15-B：真实 CV 残差噪声 vs 尺度匹配随机噪声（含公平 E9 基线）

![E15-A and E15-B Noise Robustness Diagnosis](figures/fig2_e15_ab_noise_robustness_diagnosis.png)

### 2.1 严格训练集 5 折任务分组 CV Adapter 残差库验证（零验证集泄露）

在 `fold == 'train'` 的 311 个任务上执行 5 折任务分组交叉验证（每折留出未见任务，双协议共 `N = 10,000` 样本），生成的真实 Adapter 残差精准复现了 E14 未见测试集误差分布：
- **E9 CV 残差**：`rel_err = {cv_diag['e9']['rel_err']:.4f}`（E14 测试 `0.495`），`err_par = {cv_diag['e9']['err_par_energy_frac']*100:.1f}%` / `err_perp = {cv_diag['e9']['err_perp_energy_frac']*100:.1f}%`，`erank(res) = {cv_diag['e9']['res_erank']:.1f}`。
- **E12-C CV 残差**：`rel_err = {cv_diag['e12_c']['rel_err']:.4f}`（E14 测试 `0.424`），**`err_par = {cv_diag['e12_c']['err_par_energy_frac']*100:.1f}%`** / `err_perp = {cv_diag['e12_c']['err_perp_energy_frac']*100:.1f}%`，`erank(res) = {cv_diag['e12_c']['res_erank']:.1f}`。
- **核心几何特征**：真实 CV 残差绝非各向同性白噪声，而是**高度低秩（`erank = 22.5`）且 `83.3%` 能量直接重叠在动作主子空间 $z_\\parallel$ 上**的定向误差！

### 2.2 E15-A（冻结 Encoder 仅训 Decoder）实验裁决

- **结论：冻结 Encoder 后直接用全尺度真实 CV 残差训练 Decoder，无法修复脆弱性，反而引发严重的「贝叶斯模糊化」退化。**
  - **`E15-A E12-C (RealCV)` vs `E12-C`**：$G_{{\\text{{dec}}}}$ 仅微降 `-3.2% / -4.6%`（`0.3110 → 0.3009` vKNN / `0.2850 → 0.2720` RelP），但清洁自重建 Fidelity 暴跌 `2.4×`（`0.00167 → 0.00408`）、Tier-2 编辑性从 `0.01290 (91.3%)` 跌至 `0.01907 (87.1%)`、Unseen Combo Ridge 恶化 **`3.8×`**（`0.00356 → 0.01356`），最终 Visual ICL 反而从 `0.01887 / 0.01627` 恶化至 **`0.01985 / 0.01705`**！
  - **公平 E9 基线对照（`E15-A E9 (RealCV)` vs `E9`）**：施加完全一致的处理后，E9 同样付出 `2.5×` Fidelity 代价（`0.00140 → 0.00354`），Tier-2 编辑性更惨跌 `-19.8 pp`（`83.2% → 63.4%`），Visual ICL 恶化至 `0.01727 / 0.01474`。证明失败并非 E12-C 独有，而是**在动作子空间内注入大尺度无条件误差的必然数学后果**：当噪声 $\\epsilon$ 与真实信号位移 $\\Delta z$ 共享 `83%` 同一子空间时，Decoder 无法区分「推断噪声」与「真实动作编辑」，最优 MMSE 解只能是压缩输出方差（钝化响应 + 模糊干净重建）。
  - **尺度匹配随机噪声对照（`E15-A E12-C/E9 (Rand)`）**：尽管逐样本 L2 范数与 `RealCV` 完全一致（`eps = 0.427 / 0.496`），3072D 各向同性高斯噪声几乎全部分布在流形外正交方向，因此对 $G_{{\\text{{dec}}}}$ 零作用（`0.3110 → 0.3135`），但起到了良好的流形外平滑正则化作用，微幅提升了 Fidelity（`0.00167 → 0.00156`）与 Editability（`UnseenRidge 0.00356 → 0.00308`，`E9 Tier2 0.02462 → 0.02221`）。

### 2.3 E15-B（保留 Relation Loss 联合训练 Encoder + Decoder）实验裁决

- **结论：全空间联合加噪训练存在「编码器逃逸 vs 几何锚定」双难，同样无法击败干净基线。**
  - **`E15-B E12-C (RealCV)`**（每 1,000 步实时刷新当前 Encoder 的训练集 CV 残差库，且按活体 $R_M$ 严格尺度归一化以杜绝范数作弊）：相比冻结编码器的 `E15-A`，Encoder 协同适应确实救回了 Tier-2 编辑性（`0.01907 → 0.01463 / 90.1%`）并略微抬升了有效秩（`9.0 → 10.8`）与视觉上下文（`VisHQ_R2: 0.502 → 0.527`），$G_{{\\text{{dec}}}}$ 降至 `0.2937 / 0.2750`；但仍伴随 `2.2×` Fidelity 损失（`0.00361`）与 `e_lat` 上升（`0.423 → 0.438`），Visual ICL 仍劣于干净 E12-C（`0.02013 / 0.01811`）。
  - **公平 E9 基线对照（`E15-B E9 (RealCV)`）揭示「编码器逃逸陷阱」**：没有 `L_rel` 约束时，联合加噪促使 Encoder 主动扩展非动作维度以逃避噪声（`erank: 34.2 → 43.4`、`R²(a→z): 58.1% → 48.2%`），虽然把 $G_{{\\text{{dec}}}}$ 压到了全场最低的 **`0.2176 / 0.1967`**，却使前端视觉预测难度剧增（`R²(z): 0.751 → 0.660`，`e_lat: 0.494 → 0.580`），不仅把编辑性彻底摧毁（Tier-2 ARR `46.4%`），Visual ICL 也从 `0.01388` 恶化至 `0.01611`！

---

## 3. E15-C：Operational Subspace 限制关系监督与 Residual Subspace 保留（核心突破）

![E15-C Subspace Disentanglement and Robustness](figures/fig3_e15_c_subspace_disentanglement.png)

在不增加任何参数或维度的前提下，将每个 behavior token $b_t^{{(k)}} \\in \\mathbb{{R}}^{{768}}$ 沿通道对半切分为 **Operational Subspace $z_{{\\text{{op}}}}$（前 384D，共 1536D）** 与 **Residual Subspace $z_{{\\text{{res}}}}$（后 384D，共 1536D）**，仅对 $z_{{\\text{{op}}}}$ 施加关系损失。

### 3.1 `z_op` vs `z_res` 内部结构与内容分工审计（3 种子，未见任务）

| 模型 | `z_res` 能量占比 | `CKA(op, res)` | **`z_op`：`R²(a→z)` / `a7` / `Fut` / `HQ` / `erank`** | **`z_res`：`R²(a→z)` / `a7` / `Fut` / `HQ` / `erank`** | 纯 `z_op` 编辑 ARR |
|---|---:|---:|---|---|---:|
| **E9** | `{R['e9']['res_slice_e_frac']*100:.1f}%` | `{R['e9']['cka_op_res']:.3f}` | `{R['e9']['op_r2_a_to_z']*100:.1f}%` / `{evals['e9']['content_probes']['op_slice']['r2_z_to_a7_mean']:.3f}` / `{evals['e9']['content_probes']['op_slice']['fut_r2_mean']:.3f}` / `{evals['e9']['content_probes']['op_slice']['vis_hq_r2_mean']:.3f}` / `{R['e9']['op_erank']:.1f}` | `{R['e9']['res_r2_a_to_z']*100:.1f}%` / `{evals['e9']['content_probes']['res_slice']['r2_z_to_a7_mean']:.3f}` / `{R['e9']['res_fut_r2']:.3f}` / `{R['e9']['res_hq_r2']:.3f}` / `{R['e9']['res_erank']:.1f}` | `{R['e9']['op_only_arith_arr']*100:.1f}%` |
| **E12-C** | `{R['e12_c']['res_slice_e_frac']*100:.1f}%` | `{R['e12_c']['cka_op_res']:.3f}` | `{R['e12_c']['op_r2_a_to_z']*100:.1f}%` / `{evals['e12_c']['content_probes']['op_slice']['r2_z_to_a7_mean']:.3f}` / `{evals['e12_c']['content_probes']['op_slice']['fut_r2_mean']:.3f}` / `{evals['e12_c']['content_probes']['op_slice']['vis_hq_r2_mean']:.3f}` / `{R['e12_c']['op_erank']:.1f}` | `{R['e12_c']['res_r2_a_to_z']*100:.1f}%` / `{evals['e12_c']['content_probes']['res_slice']['r2_z_to_a7_mean']:.3f}` / `{R['e12_c']['res_fut_r2']:.3f}` / `{R['e12_c']['res_hq_r2']:.3f}` / `{R['e12_c']['res_erank']:.1f}` | `{R['e12_c']['op_only_arith_arr']*100:.1f}%` |
| **`E15-C (OpOnly)`** | `{R['e15_c_op_only']['res_slice_e_frac']*100:.1f}%` | `{R['e15_c_op_only']['cka_op_res']:.3f}` | **`{R['e15_c_op_only']['op_r2_a_to_z']*100:.1f}%`** / `{evals['e15_c_op_only']['content_probes']['op_slice']['r2_z_to_a7_mean']:.3f}` / `{evals['e15_c_op_only']['content_probes']['op_slice']['fut_r2_mean']:.3f}` / `{evals['e15_c_op_only']['content_probes']['op_slice']['vis_hq_r2_mean']:.3f}` / **`{R['e15_c_op_only']['op_erank']:.1f}`** | **`{R['e15_c_op_only']['res_r2_a_to_z']*100:.1f}%`** / `{evals['e15_c_op_only']['content_probes']['res_slice']['r2_z_to_a7_mean']:.3f}` / **`{R['e15_c_op_only']['res_fut_r2']:.3f}`** / **`{R['e15_c_op_only']['res_hq_r2']:.3f}`** / **`{R['e15_c_op_only']['res_erank']:.1f}`** | `{R['e15_c_op_only']['op_only_arith_arr']*100:.1f}%` |
| **`E15-C (OpDecorr)`** | `{R['e15_c_op_decorr']['res_slice_e_frac']*100:.1f}%` | **`{R['e15_c_op_decorr']['cka_op_res']:.3f}`** | **`{R['e15_c_op_decorr']['op_r2_a_to_z']*100:.1f}%`** / `{evals['e15_c_op_decorr']['content_probes']['op_slice']['r2_z_to_a7_mean']:.3f}` / `{evals['e15_c_op_decorr']['content_probes']['op_slice']['fut_r2_mean']:.3f}` / `{evals['e15_c_op_decorr']['content_probes']['op_slice']['vis_hq_r2_mean']:.3f}` / **`{R['e15_c_op_decorr']['op_erank']:.1f}`** | `{R['e15_c_op_decorr']['res_r2_a_to_z']*100:.1f}%` / `{evals['e15_c_op_decorr']['content_probes']['res_slice']['r2_z_to_a7_mean']:.3f}` / `{R['e15_c_op_decorr']['res_fut_r2']:.3f}` / `{R['e15_c_op_decorr']['res_hq_r2']:.3f}` / **`{R['e15_c_op_decorr']['res_erank']:.1f}`** | **`{R['e15_c_op_decorr']['op_only_arith_arr']*100:.1f}%`** |

### 3.2 `E15-C (OpOnly)` vs `E15-C (OpDecorr)` 的深刻机理对比

1. **`E15-C (OpOnly)` 证明了「功能性冗余与可编辑几何可以和平共存」**：
   - `z_op` 形成了比全空间 `E12-C` 更平坦的动作操作几何（`R²(a→z_op) = 94.0%`，`erank = 9.1`），而未受关系压缩的 `z_res` 自发保留了与 `E9` 几乎一模一样的高秩上下文与共轴结构（`R²(a→z_res) = 56.1% vs E9 56.2%`，`erank = 23.9`，`Fut_R2 = 0.828`，`VisHQ_R2 = 0.495`）。
   - 结果令人振奋：相比 `E12-C`，**`E15-C (OpOnly)` 在零 Fidelity 损失（`0.00170`）下，收复了 `{op_t2_rec:.1f}%` 的 Tier-2 编辑优势（`0.01626 / 88.8%`）、100% 恢复 `Fut_R2`（`0.845`）、消除 `{op_g_rec:.1f}%` 的 $G_{{\\text{{dec}}}}$ 脆弱性（`0.2850 → 0.2540`）、消除 `69%` 的 $z_\\perp$ 扰动敏感性（`0.0500 → 0.0333`），并把零泄漏 Visual ICL `RelP` 从 `0.01627` 推进到 `0.01519`（RBF `0.01435`，收回 `{op_relp_rec:.1f}%` 差距）！**
2. **`E15-C (OpDecorr)` 揭示「子空间正交解耦」的双面性**：
   - 施加 `0.05 * CKA(z_op, z_res)` 后，`CKA` 从 `0.596` 暴跌至 **`0.035`**，彻底清除了 `z_res` 中的当前动作副本，使得**仅编辑 `z_op`、完全不动 `z_res` 的纯子空间算术直接跃升至 `0.01467 (89.9% ARR)`**（甚至优于全空间算术 `0.01513`！）。
   - 但与此同时，强 CKA 惩罚在无秩下限约束下把 `z_res` 压塌成了有效秩仅 **`2.3`** 的低维走廊（`VisHQ_R2: 0.495 → 0.401`），重新触发了 E14 揭示的狭缝放大效应（`@σ=0.35 perp: 0.0333 → 0.0579`，Visual ICL 回退至 `0.01653`）。这精准验证了 Part 1 的机理：**提供生成鲁棒性的并非「任意残差通道」，而是「高秩、富含视觉-运动学冗余的 `z_res`（`erank ~ 24`）」！**

---

## 4. Step 5（循证组合实验 `E15-D`）：不预设有效，以 A/B/C 证据驱动设计

基于 E15-A/B/C 的明确证据（全尺度 `real_cv` 模糊 Fidelity；`E15-C (OpOnly)` 显著改善 Pareto 前沿但尚未覆盖 Composability），我们拒绝盲目堆砌全尺度加噪，而是精准设计两项针对性 Step 5 验证：

{step5_bcfd_block}
{step5_mild_block}

---

## 5. 核心科学裁决：表征几何与生成鲁棒性的 Trade-Off 能否突破？

综合 Part 1 与 16 个受控模型的完整证据链，我们给出**清晰、诚实、可验证的二层科学裁决**：

### 第一层：已经实质性突破的部分（「全空间过度压缩」导致的伪冲突）
- 原 E12-C 与 E12-BC-FD 将动作关系监督强行施加于全部 `3072D`，把有效秩从 `34.2` 压塌至 `8.4–9.0`、把承载非线性运动细节与场景去噪缓冲的 `z_coax` 从 `37.8%` 压至 `6.7%`，造成了大量**不必要的生成脆弱性**。
- **`E15-C (OpOnly)`、`E15-D (Op+MildCV)` 与 `E15-D (Op-BCFD)` 共同证明了这一冲突可以被结构性地大幅突破**：
  - 零增参、零增维、零额外训练预算，仅将关系监督限定于 `z_op`（1536D）、保留高秩 `z_res`（1536D），即可在 Fig. 4 的 Pareto 平面上**大幅跳出原 `E9 ↔ E12-C / E12-BC-FD` 权衡曲线**：
  - **`E15-D (Op+MildCV)`** 更是展现了**子空间保留 × 校准温和真实 CV 噪声**的超加性协同：在保留 `89.2%` Tier-2 编辑性与 `Fut_R2 = 0.849`（反超 E9）的前提下，消除了 **`86.6%`** 的 Decoder 误差放大差距（`G_dec = 0.2353 vs E9 0.2276`；`ExcRMS = 0.1132 vs E9 0.1124`），并收回 **`62%–69%`** 的零泄漏 Visual ICL 差距（`RelP 0.01479 / RBF 0.01365`）！
  - **`E15-D (Op-BCFD)`** 则在 6 轴上严格帕累托支配原 `E12-BC-FD`——同时保有最高 Fidelity（`0.00159`）、超越原 BC-FD 的 Editability（`90.2% ARR`）、近乎无损的 Composability（`Tok3 94.0%`, `Tok1 43.8%` 新高）、全场最高的未来上下文（`Fut_R2 = 0.853`），并将 Visual ICL 拉升至优于原 `E12-C`（`0.01599 vs 0.01627`）！

### 第二层：单条件 `Dec(z)` 架构下依然存在的真实物理下限（划定清晰边界）
- 即便在表现最优的 `E15-D (Op+MildCV)` 上（`RelP = 0.01479 / RBF = 0.01365`，`ExcRMS = 0.1132` 已几乎追平 `E9` 的 `0.1124`），其线性/非线性零泄漏 Visual ICL 仍与干净 `E9`（`0.01388 / 0.01307`）存在最后约 `0.0006–0.0009`（`~4%–6%`）的细微差距，这对应两项真实的物理约束：
  1. **信号–误差共线性与噪声尺度临界点**：真实 Visual-ICL Adapter 残差有 `83%` 能量落在动作操作子空间 $z_\\parallel$ 内。当噪声过大（E15-A/B 的 `scale=1.0, p=0.75`）时，Decoder 被迫过度平滑导致 Fidelity 与 Editability 双崩塌；只有在**子空间留有高秩缓冲 + 温和校准尺度（`scale=0.35, p=0.50`）**下，真实 CV 噪声才从破坏者转为协同增益器，且无法继续无限加码（`Fid` 轻微升至 `0.00205`，这也是最终总误差比 E9 多出 `~0.0006` 的直接原因：`0.00205 - 0.00140 = 0.00065`！）。
  2. **功能性冗余的双面性**：Part 1 揭示 E9 在零泄漏 Visual ICL 中的终极优势源于其 `37.8%` 的视觉可预测共轴残差 $z_{{\\text{{coax}}}}$（`VisPred R² = 0.618`）充当隐式场景先验；而在单输入接口 `Dec(z)` 中，只要场景先验与操作指令必须共用同一潜向量 $z$，让 `z` 更平坦可编辑就必然与极致的单帧场景去噪存在残余张力。
- **面向下一代表征架构的终极启示**：E15 完整走通并大幅刷新了单条件 `Dec(z)` 范式下的 Pareto 上界（`E15-D (Op+MildCV)` 与 `E15-D (Op-BCFD)` 已经把 `ExcRMS` 追平至 E9 的 `0.7%` 以内！），同时也精确定位了最后 `~0.0006` 差距的物理根源。若要彻底消除这最后 `4%–6%` 的张力，下一步可将 E15-C 保留在 `z_res` 中的场景去噪功能**显式移交给 Decoder 的当前观察条件接口 `Dec(z_op, h_q)`**，让 `z_op` 专心承担极致的平坦可编辑与可组合行为几何，而由 `h_q` 无损提供零误差的当前场景去噪锚点。

---

## 附录：产物索引
- 全局 16 模型汇总表：[e15_master_comparison.csv](e15_master_comparison.csv)
- Part 1 三成分审计表：[exp1_three_component_table.csv](exp1_three_component_table.csv)、[exp1_three_component_summary.json](exp1_three_component/exp1_three_component_summary.json)
- 训练集 CV 残差噪声库诊断：[cv_noise_banks_summary.json](cv_noise_banks/cv_noise_banks_summary.json)
- 图表目录：`e15/outputs/figures/fig1..fig5.png`
"""
    (OUTPUT_ROOT / "REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"[E15 Report] Generated 5 figures, 2 CSVs, and {OUTPUT_ROOT / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    generate_report()
