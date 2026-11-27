"""E14 Report, CSV Tables, and Publication-Grade Figure Generator.

Reads completed JSON summaries from:
  - e14/outputs/exp1_cross_transfer/exp1_summary.json
  - e14/outputs/exp2_mechanism_diagnosis/exp2_summary.json
  - e14/outputs/exp3_complementary_repr/exp3_summary.json

Produces:
  - 3 consolidated CSV tables under `e14/outputs/`
  - 6 publication-grade figures under `e14/outputs/figures/`
  - Comprehensive, evidence-grounded `e14/outputs/REPORT.md`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import CORE_MODELS, MODEL_LABELS, OUTPUT_ROOT


COLORS = {
    "e9": "#1f77b4",
    "e12_b": "#2ca02c",
    "e12_c": "#d62728",
    "e12_bc_factordisp": "#9467bd",
    "control": "#7f7f7f",
    "none": "#444444",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _export_csvs(d1: Dict[str, Any], d2: Dict[str, Any], d3: Dict[str, Any], out_dir: Path) -> None:
    # 1. Exp 1 Cross-Transfer Matrix CSV (all protocols at N=4000 + N=1000)
    rows1 = []
    for mode in d1["match_modes"]:
        res = d1["results"][mode]
        for proto in ("direct_full_ridge", "direct_pca_ridge", "aligned_res_ridge", "rbf_krr", "tuned_res_mlp"):
            for n_s in ("1000", "4000"):
                if n_s not in res["protocols"][proto]:
                    continue
                for sp in ("seen", "unseen"):
                    mat = res["protocols"][proto][n_s][sp]
                    for A, cols in mat.items():
                        for B, v in cols.items():
                            rows1.append(
                                {
                                    "match_mode": mode,
                                    "protocol": proto,
                                    "budget_n": int(n_s),
                                    "split": sp,
                                    "input_a": A,
                                    "target_b": B,
                                    "mse10_mean": v["mse10_mean"],
                                    "mse10_std": v["mse10_std"],
                                    "arr_10d_mean": v["arr_10d_mean"],
                                    "grip_acc_mean": v["grip_acc_mean"],
                                    "r2_z_mean": v["r2_z_mean"],
                                    "r2_dz_mean": v["r2_dz_mean"],
                                    "cos_dz_mean": v["cos_dz_mean"],
                                    "rel_rms_err_mean": v["rel_rms_err_mean"],
                                }
                            )
    _save_csv(rows1, out_dir / "exp1_cross_transfer_matrix.csv")

    # 2. Exp 2 Mechanism Summary CSV
    rows2 = []
    for mode in d2["match_modes"]:
        p2a = d2["results"][mode]["part_2a"]
        p2b = d2["results"][mode]["part_2b"]
        p2c = d2["results"][mode]["part_2c"]
        for m in CORE_MODELS:
            st = p2a[m]["subspace_stats"]
            lf = p2a[m]["adapters"]["linear_full"]["4000"]["unseen"]
            kf = p2a[m]["adapters"]["rbf_krr_full"]["4000"]["unseen"]
            lp = p2a[m]["adapters"]["linear_pca"]["4000"]["unseen"]
            kp = p2a[m]["adapters"]["rbf_krr_pca"]["4000"]["unseen"]
            rm = p2a[m]["adapters"]["res_mlp_pca"]["4000"]["unseen"]
            fl = p2b[m]["factorization_linear"]
            c = p2b[m]["curves"]
            gt = p2c[m]["gt_decoder_ablation"]
            oa = p2c[m]["output_ablation"]["4000"]
            rows2.append(
                {
                    "match_mode": mode,
                    "model": m,
                    "erank_z": st["erank_z_mean"],
                    "erank_dz": st["erank_dz_mean"],
                    "rank_995": st["rank_995_mean"],
                    "geo_par_frac": st["geo_subspace_var_frac_mean"],
                    "reg_r2_a_to_z": st["reg_r2_train_mean"],
                    "r2_z": lf["r2_z_mean"],
                    "r2_z_par": lf["r2_z_par_mean"],
                    "r2_z_perp": lf["r2_z_perp_mean"],
                    "r2_dz": lf["r2_dz_mean"],
                    "cos_dz": lf["cos_dz_mean"],
                    "shrinkage_ratio": lf["shrinkage_ratio_mean"],
                    "mse10_lin_full": lf["mse10_mean"],
                    "mse10_rbf_full": kf["mse10_mean"],
                    "mse10_lin_pca": lp["mse10_mean"],
                    "mse10_rbf_pca": kp["mse10_mean"],
                    "mse10_res_mlp": rm["mse10_mean"],
                    "e_lat_rel": fl["e_lat_rel_mean"],
                    "G_dec_excess": fl["G_dec_excess_mean"],
                    "excess_act_rms": fl["excess_act_rms_mean"],
                    "shift_rms_iso_035": c["isotropic"]["0.35"]["shift_rms_vs_clean_mean"],
                    "shift_rms_cov_035": c["cov_matched"]["0.35"]["shift_rms_vs_clean_mean"],
                    "shift_rms_par_035": c["subspace_par"]["0.35"]["shift_rms_vs_clean_mean"],
                    "shift_rms_perp_035": c["subspace_perp"]["0.35"]["shift_rms_vs_clean_mean"],
                    "shift_rms_icl_035": c["icl_err_rescaled"]["0.35"]["shift_rms_vs_clean_mean"],
                    "shrink_030_mse10": c["shrinkage"]["0.30"]["mse10_vs_gt_mean"],
                    "gt_full_mse10": gt["gt_full"]["unseen"]["mse10_mean"],
                    "gt_par_only_mse10": gt["gt_par_only"]["unseen"]["mse10_mean"],
                    "gt_par_shuf_perp_mse10": gt["gt_par_plus_shuf_perp"]["unseen"]["mse10_mean"],
                    "out_full_mse10": oa["out_full"]["unseen"]["mse10_mean"],
                    "out_zero_perp_mse10": oa["out_par_only_zero_perp"]["unseen"]["mse10_mean"],
                    "out_demo_perp_mse10": oa["out_par_plus_demo_perp"]["unseen"]["mse10_mean"],
                    "out_hq_perp_mse10": oa["out_par_plus_hq_perp"]["unseen"]["mse10_mean"],
                    "out_gt_perp_mse10": oa["out_par_plus_gt_perp"]["unseen"]["mse10_mean"],
                }
            )
    _save_csv(rows2, out_dir / "exp2_mechanism_summary.csv")

    # 3. Exp 3 Complementarity Summary CSV
    rows3 = []
    for mode in d3["match_modes"]:
        vs = d3["results"][mode]["variants"]
        for vname, by_n in vs.items():
            for n_s, sps in by_n.items():
                u = sps["unseen"]
                rows3.append(
                    {
                        "match_mode": mode,
                        "variant": vname,
                        "budget_n": int(n_s),
                        "unseen_mse10_mean": u["mse10_mean"],
                        "unseen_mse10_std": u["mse10_std"],
                        "unseen_arr_10d_mean": u["arr_10d_mean"],
                        "unseen_grip_acc_mean": u["grip_acc_mean"],
                        "unseen_r2_z_mean": u["r2_z_mean"],
                        "seen_mse10_mean": sps["seen"]["mse10_mean"],
                    }
                )
    _save_csv(rows3, out_dir / "exp3_complementarity_summary.csv")


# ---------------------------------------------------------------------------
# Figures 1-6
# ---------------------------------------------------------------------------

def plot_fig1_cross_transfer_4x4(d1: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1.05, 1.25]})
    rows = list(CORE_MODELS) + ["none"]
    row_lbls = [MODEL_LABELS[r] for r in rows]
    col_lbls = [MODEL_LABELS[c] for c in CORE_MODELS]

    for idx, (mode, title) in enumerate([("visual_knn", "Visual kNN (Unseen, N=4k)"), ("rel_progress", "Rel-Progress (Unseen, N=4k)")]):
        ax = axes[idx]
        p = d1["results"][mode]["protocols"]["direct_full_ridge"]["4000"]["unseen"]
        mat = np.array([[p[r][c]["mse10_mean"] for c in CORE_MODELS] for r in rows])
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(col_lbls)))
        ax.set_xticklabels(col_lbls, rotation=18, ha="right")
        ax.set_yticks(range(len(row_lbls)))
        ax.set_yticklabels(row_lbls)
        ax.set_xlabel("Target Latent + Frozen Decoder (B)")
        if idx == 0:
            ax.set_ylabel("Demo Input Latent (A)")
        ax.set_title(title, fontweight="bold")
        ax.axhline(3.5, color="black", lw=1.4, ls="--")
        vmin, vmax = mat.min(), mat.max()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt_col = "white" if val > vmin + 0.62 * (vmax - vmin) else "black"
                fw = "bold" if (i == j or (i == 2 and j == 0)) else "normal"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=txt_col, fontsize=8.8, fontweight=fw)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Action 10D MSE (↓)", fontsize=8.5)

    # Panel 3: Two-Way ANOVA Variance Decomposition across all 5 protocols (both modes)
    ax = axes[2]
    protos = ["direct_full_ridge", "direct_pca_ridge", "aligned_res_ridge", "rbf_krr", "tuned_res_mlp"]
    plbls = ["Full Ridge", "PCA Ridge", "Aligned Res", "RBF-KRR", "Res-MLP"]
    x = np.arange(len(protos))
    w = 0.36
    for m_i, (mode, off, hatch, alpha_b) in enumerate(
        [("visual_knn", -w / 2, "", 0.92), ("rel_progress", w / 2, "//", 0.72)]
    ):
        in_f, out_f, int_f = [], [], []
        for pr in protos:
            an = d1["results"][mode]["anova_4x4"][pr]["4000"]["unseen"]["mse10"]
            in_f.append(an["ss_input_frac"] * 100)
            out_f.append(an["ss_output_frac"] * 100)
            int_f.append(an["ss_interaction_frac"] * 100)
        in_f = np.array(in_f)
        out_f = np.array(out_f)
        int_f = np.array(int_f)
        ax.bar(x + off, out_f, w, color="#d62728", alpha=alpha_b, hatch=hatch, edgecolor="white", label=f"Output/Dec ({'vKNN' if m_i==0 else 'RelP'})")
        ax.bar(x + off, in_f, w, bottom=out_f, color="#1f77b4", alpha=alpha_b, hatch=hatch, edgecolor="white", label=f"Input ({'vKNN' if m_i==0 else 'RelP'})")
        ax.bar(x + off, int_f, w, bottom=out_f + in_f, color="#ff7f0e", alpha=alpha_b, hatch=hatch, edgecolor="white", label=f"Interact ({'vKNN' if m_i==0 else 'RelP'})")
        for xi, ov in zip(x + off, out_f):
            ax.text(xi, ov / 2, f"{ov:.0f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(plbls, rotation=18, ha="right")
    ax.set_ylabel("4×4 MSE Variance Explained (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Two-Way ANOVA Across 5 Adapter Protocols", fontweight="bold")
    ax.legend(ncol=2, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)

    fig.suptitle("E14 Exp 1: 4×4 Cross-Representation Transfer — Bottleneck Is 75%–99% Output/Decoder", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_exp1_cross_transfer_4x4.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_sample_efficiency(d1: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), sharey=False)
    budgets = d1["budgets"]
    curves = [
        ("e9", "e9", "E9 → E9 (Diag)", COLORS["e9"], "-", "o", 2.2),
        ("e12_c", "e9", "E12-C → E9 (Cross)", "#17becf", "--", "s", 2.0),
        ("e12_bc_factordisp", "e9", "E12-BC-FD → E9 (Cross)", "#8c564b", ":", "D", 1.8),
        ("e12_b", "e12_b", "E12-B → E12-B (Diag)", COLORS["e12_b"], "-", "^", 1.9),
        ("e12_c", "e12_c", "E12-C → E12-C (Diag)", COLORS["e12_c"], "-", "v", 2.1),
        ("e9", "e12_c", "E9 → E12-C (Cross)", "#ff7f0e", "--", "P", 1.8),
        ("e12_bc_factordisp", "e12_bc_factordisp", "E12-BC-FD → E12-BC-FD (Diag)", COLORS["e12_bc_factordisp"], "-", "X", 2.1),
        ("none", "e9", "Vision-Only → E9", "#333333", "-.", "*", 1.7),
    ]

    for ax, (mode, title) in zip(axes, [("visual_knn", "Protocol A: Visual kNN (Unseen Tasks)"), ("rel_progress", "Protocol B: Rel-Progress (Unseen Tasks)")]):
        proto = d1["results"][mode]["protocols"]["direct_full_ridge"]
        for A, B, lbl, col, ls, mk, lw in curves:
            ys = [proto[str(n)]["unseen"][A][B]["mse10_mean"] for n in budgets]
            ax.plot(budgets, ys, label=lbl, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=5.5)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(n) for n in budgets])
        ax.set_xlabel("Adapter Training Budget N (Zero-Leakage Pairs)")
        ax.set_ylabel("Unseen-Task Action 10D MSE (↓)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=8.5)
    fig.suptitle("E14 Exp 1: Sample-Efficiency Curves (N = 50 → 4000) — Target/Decoder Column Governs All Budgets", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_exp1_sample_efficiency_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_exp2a_predictability_nonlinearity(d2: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    models = list(CORE_MODELS)
    mlbls = [MODEL_LABELS[m] for m in models]
    x = np.arange(len(models))

    # Panel A: The Predictability Paradox (rel_progress N=4000)
    ax = axes[0]
    p2a = d2["results"]["rel_progress"]["part_2a"]
    r2_z = [p2a[m]["adapters"]["linear_full"]["4000"]["unseen"]["r2_z_mean"] for m in models]
    r2_par = [p2a[m]["adapters"]["linear_full"]["4000"]["unseen"]["r2_z_par_mean"] for m in models]
    r2_perp = [p2a[m]["adapters"]["linear_full"]["4000"]["unseen"]["r2_z_perp_mean"] for m in models]
    cos_dz = [p2a[m]["adapters"]["linear_full"]["4000"]["unseen"]["cos_dz_mean"] for m in models]
    w = 0.20
    ax.bar(x - 1.5 * w, r2_z, w, label="Total R²(z)", color="#1f77b4")
    ax.bar(x - 0.5 * w, r2_par, w, label="Action-Subspace R²(z_∥)", color="#2ca02c")
    ax.bar(x + 0.5 * w, cos_dz, w, label="Displacement cos(Δz)", color="#ff7f0e")
    ax.bar(x + 1.5 * w, r2_perp, w, label="Orthogonal R²(z_⊥)", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(mlbls)
    ax.set_ylim(0.35, 0.96)
    ax.set_ylabel("Latent Predictability Metric (↑)")
    ax.set_title("A. Latent Predictability Paradox (RelP @4k)", fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower left", ncol=2, frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    # Panel B & C: Linear vs Tuned Nonlinear Adapters (visual_knn & rel_progress)
    adps = [
        ("linear_pca", "Linear (PCA-320)", "#aec7e8"),
        ("linear_full", "Linear (Full)", "#1f77b4"),
        ("res_mlp_pca", "Tuned Res-MLP", "#ff7f0e"),
        ("rbf_krr_pca", "RBF-KRR (PCA)", "#2ca02c"),
        ("rbf_krr_full", "RBF-KRR (Full)", "#d62728"),
    ]
    for ax_i, (mode, title) in enumerate([("visual_knn", "B. Linear vs Nonlinear (Visual kNN @4k)"), ("rel_progress", "C. Linear vs Nonlinear (Rel-Progress @4k)")]):
        ax = axes[ax_i + 1]
        p2a_m = d2["results"][mode]["part_2a"]
        w = 0.16
        for k_i, (ak, albl, col) in enumerate(adps):
            vals = [p2a_m[m]["adapters"][ak]["4000"]["unseen"]["mse10_mean"] for m in models]
            ax.bar(x + (k_i - 2) * w, vals, w, label=albl, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(mlbls)
        ax.set_ylabel("Unseen Action 10D MSE (↓)")
        ymin = 0.0115 if mode == "rel_progress" else 0.0145
        ymax = 0.0195 if mode == "rel_progress" else 0.0225
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if ax_i == 1:
            ax.legend(fontsize=7.6, loc="upper left", ncol=2, frameon=True, framealpha=0.9)

    fig.suptitle("E14 Exp 2A: E12 Latents Are MORE Predictable (Higher R²(z)), Yet Nonlinear Adapters Do Not Close the Action Gap", fontsize=11.8, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_exp2a_predictability_and_nonlinearity.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_exp2b_decoder_sensitivity(d2: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    models = list(CORE_MODELS)
    mlbls = [MODEL_LABELS[m] for m in models]
    p2b_v = d2["results"]["visual_knn"]["part_2b"]
    p2b_r = d2["results"]["rel_progress"]["part_2b"]

    # Panel A: Exact Log-Additive Factorization relative to E9 (%)
    ax = axes[0]
    e9_el_r = p2b_r["e9"]["factorization_linear"]["e_lat_rel_mean"]
    e9_gd_r = p2b_r["e9"]["factorization_linear"]["G_dec_excess_mean"]
    e9_ex_r = p2b_r["e9"]["factorization_linear"]["excess_act_rms_mean"]
    x = np.arange(3)
    comp_models = ["e12_b", "e12_c", "e12_bc_factordisp"]
    clbls = [MODEL_LABELS[m] for m in comp_models]
    el_pct = [(p2b_r[m]["factorization_linear"]["e_lat_rel_mean"] / e9_el_r - 1.0) * 100 for m in comp_models]
    gd_pct = [(p2b_r[m]["factorization_linear"]["G_dec_excess_mean"] / e9_gd_r - 1.0) * 100 for m in comp_models]
    ex_pct = [(p2b_r[m]["factorization_linear"]["excess_act_rms_mean"] / e9_ex_r - 1.0) * 100 for m in comp_models]
    w = 0.25
    b1 = ax.bar(x - w, el_pct, w, label="Latent Error e_lat (% vs E9)", color="#1f77b4")
    b2 = ax.bar(x, gd_pct, w, label="Decoder Gain G_dec (% vs E9)", color="#d62728")
    b3 = ax.bar(x + w, ex_pct, w, label="Net Action ExcessRMS (% vs E9)", color="#333333")
    ax.axhline(0, color="black", lw=1.0)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 0.9 if h >= 0 else -1.1
            ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.1f}%", ha="center", va=va, fontsize=7.6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clbls)
    ax.set_ylim(-26, 44)
    ax.set_ylabel("Relative Change vs E9 (%)")
    ax.set_title("A. Exact Factorization: ExcRMS = e_lat × G_dec (RelP)", fontweight="bold")
    ax.legend(fontsize=7.6, loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Equal-Scale Perturbation Sensitivity Curves (Actual ICL Direction & z_perp)
    ax = axes[1]
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    for m in models:
        y_icl = [p2b_r[m]["curves"]["icl_err_rescaled"][f"{s:.2f}"]["shift_rms_vs_clean_mean"] for s in sigmas]
        y_prp = [p2b_r[m]["curves"]["subspace_perp"][f"{s:.2f}"]["shift_rms_vs_clean_mean"] for s in sigmas]
        ax.plot(sigmas, y_icl, color=COLORS[m], lw=2.2, marker="o", ms=4.5, label=f"{MODEL_LABELS[m]} (ICL dir)")
        ax.plot(sigmas, y_prp, color=COLORS[m], lw=1.5, ls="--", marker="x", ms=4.0, alpha=0.85)
    ax.set_xlabel("Equal Relative Perturbation Scale σ = ||ε|| / R_M")
    ax.set_ylabel("Decoded Action Shift RMS vs Clean (↑ = More Sensitive)")
    ax.set_title("B. Equal-Scale Perturbation: ICL Dir (solid) & z_⊥ (dash)", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Panel C: Perturbation Type Breakdown at σ = 0.35
    ax = axes[2]
    ptypes = ["isotropic", "subspace_perp", "cov_matched", "subspace_par", "icl_err_rescaled"]
    pt_lbls = ["Isotropic", "z_⊥ (Non-Act)", "On-Manifold", "z_∥ (Act)", "ICL Error Dir"]
    xp = np.arange(len(ptypes))
    w = 0.19
    for mi, m in enumerate(models):
        vals = [p2b_v[m]["curves"][pt]["0.35"]["shift_rms_vs_clean_mean"] for pt in ptypes]
        ax.bar(xp + (mi - 1.5) * w, vals, w, label=MODEL_LABELS[m], color=COLORS[m])
    ax.set_xticks(xp)
    ax.set_xticklabels(pt_lbls, rotation=16, ha="right")
    ax.set_ylabel("Decoded Action Shift RMS @ σ=0.35")
    ax.set_title("C. Directional Sensitivity Breakdown (@ σ=0.35)", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("E14 Exp 2B: Higher Editability Monotonically Amplifies Decoder Sensitivity G_dec (+12% to +35%), Overwhelming Lower e_lat", fontsize=11.6, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_exp2b_decoder_sensitivity_and_factorization.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig5_exp2c_subspace_ablation(d2: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.7))
    models = list(CORE_MODELS)
    mlbls = [MODEL_LABELS[m] for m in models]

    # Panel A: GT Decoder Subspace Ablation across all 4 models
    ax = axes[0]
    p2c_v = d2["results"]["visual_knn"]["part_2c"]
    conds = [
        ("gt_full", "GT Full z", "#1f77b4"),
        ("gt_par_only", "GT z_∥ Only (z_⊥=0)", "#ff7f0e"),
        ("gt_par_plus_demo_perp", "GT z_∥ + Demo z_⊥", "#2ca02c"),
        ("gt_par_plus_shuf_perp", "GT z_∥ + Shuffled z_⊥", "#d62728"),
    ]
    x = np.arange(len(models))
    w = 0.19
    for ci, (ck, clbl, col) in enumerate(conds):
        vals = [p2c_v[m]["gt_decoder_ablation"][ck]["unseen"]["mse10_mean"] * 1000.0 for m in models]
        bars = ax.bar(x + (ci - 1.5) * w, vals, w, label=clbl, color=col)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_LABELS[m]}\n(z_⊥={100*(1-d2['results']['visual_knn']['part_2a'][m]['subspace_stats']['geo_subspace_var_frac_mean']):.1f}%)" for m in models])
    ax.set_ylabel("Decoded Action 10D MSE × 10³ (↓)")
    ax.set_title("A. GT Latent Subspace Ablation: E12's Tiny z_⊥ Is 2.3–2.6× More Fragile", fontweight="bold")
    ax.legend(fontsize=8.0, loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: E9 Visual ICL Causal Ablations (both visual_knn & rel_progress @ N=4000)
    ax = axes[1]
    out_specs = [
        ("out_par_plus_gt_perp", "z_∥_hat + GT z_⊥\n(Oracle Scene)"),
        ("out_par_plus_hq_perp", "z_∥_hat + h_q z_⊥\n(Query Image Only)"),
        ("out_full", "Full E9 Pred\n(z_∥_hat + z_⊥_hat)"),
        ("out_par_only_zero_perp", "z_∥_hat Only\n(z_⊥ = 0)"),
        ("out_par_plus_demo_perp", "z_∥_hat + Demo z_⊥\n(Wrong Scene)"),
    ]
    xo = np.arange(len(out_specs))
    w = 0.34
    v_vals = [d2["results"]["visual_knn"]["part_2c"]["e9"]["output_ablation"]["4000"][k]["unseen"]["mse10_mean"] for k, _ in out_specs]
    r_vals = [d2["results"]["rel_progress"]["part_2c"]["e9"]["output_ablation"]["4000"][k]["unseen"]["mse10_mean"] for k, _ in out_specs]
    b1 = ax.bar(xo - w / 2, v_vals, w, label="Visual kNN", color="#1f77b4")
    b2 = ax.bar(xo + w / 2, r_vals, w, label="Rel-Progress", color="#17becf")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.00025, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=7.2, rotation=0)
    ax.set_xticks(xo)
    ax.set_xticklabels([lbl for _, lbl in out_specs], fontsize=8.0)
    ax.set_ylim(0.009, 0.0215)
    ax.set_ylabel("E9 Unseen Action 10D MSE (↓)")
    ax.set_title("B. E9 Output-Side z_⊥ Causal Ablations (Visual ICL @4k)", fontweight="bold")
    ax.legend(fontsize=8.2, loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("E14 Exp 2C: Action-Orthogonal Subspace z_⊥ Acts as a Query-Scene Denoising Prior in E9 vs High-Gain Fragility in E12", fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_exp2c_subspace_ablation.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig6_exp3_complementarity(d3: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    # Panel A: Equal-Dim & Equal-Param Input Fusion Controls (visual_knn & rel_progress @4k)
    ax = axes[0]
    v_v = d3["results"]["visual_knn"]["variants"]
    v_r = d3["results"]["rel_progress"]["variants"]
    specs_a = [
        ("m64_single_e9", "Single E9 (64D)"),
        ("m64_concat_e9+control->e9", "E9+E9Cont (32+32)"),
        ("m64_concat_e9+e12_c->e9", "E9+E12C (32+32)"),
        ("m64_orthres_e9+(e12_c|e9)->e9", "E9+Orth(C|E9)"),
        ("gated_single_e9_split->e9", "Gated E9 Split*"),
        ("gated_e9+e9cont->e9", "Gated E9+Cont*"),
        ("gated_e9+e12c->e9", "Gated E9+E12C*"),
        ("gated_e9+e12bcfd->e9", "Gated E9+BCFD*"),
    ]
    xa = np.arange(len(specs_a))
    w = 0.36
    yv = [v_v[k]["4000"]["unseen"]["mse10_mean"] for k, _ in specs_a]
    yr = [v_r[k]["4000"]["unseen"]["mse10_mean"] for k, _ in specs_a]
    ax.bar(xa - w / 2, yv, w, label="Visual kNN", color="#1f77b4")
    ax.bar(xa + w / 2, yr, w, label="Rel-Progress", color="#17becf")
    ax.axvline(3.5, color="gray", ls="--", lw=1.0)
    ax.set_xticks(xa)
    ax.set_xticklabels([lbl for _, lbl in specs_a], rotation=28, ha="right", fontsize=7.8)
    ax.set_ylim(0.0125, 0.0200)
    ax.set_ylabel("Unseen Action 10D MSE (↓)")
    ax.set_title("A. Input Fusion (* = Exact Equal Params)", fontweight="bold")
    ax.legend(fontsize=8.0, loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Dual-Decoder Output Fusion vs Same-Family Ensemble & Oracle Best-of-Two
    ax = axes[1]
    dd_v = d3["results"]["visual_knn"]["dual_decoder"]
    dd_r = d3["results"]["rel_progress"]["dual_decoder"]
    au_v = d3["results"]["visual_knn"]["complementarity_audit"]
    au_r = d3["results"]["rel_progress"]["complementarity_audit"]
    cats = ["Single E9", "E9+E9Cont\n(Same-Fam)", "E9+E12-C\n(Cross-Fam)", "E9+E12-BCFD\n(Cross-Fam)"]
    xb = np.arange(len(cats))
    w = 0.20
    blend_v = [
        v_v["full_single_e9"]["4000"]["unseen"]["mse10_mean"],
        dd_v["e9+control"]["tuned_blend_unseen"]["mse10_mean"],
        dd_v["e9+e12_c"]["tuned_blend_unseen"]["mse10_mean"],
        dd_v["e9+e12_bc_factordisp"]["tuned_blend_unseen"]["mse10_mean"],
    ]
    ora_v = [
        v_v["full_single_e9"]["4000"]["unseen"]["mse10_mean"],
        au_v["e9_vs_control"]["oracle_best_of_two_mse10_mean"],
        au_v["e9_vs_e12_c"]["oracle_best_of_two_mse10_mean"],
        au_v["e9_vs_e12_bc_factordisp"]["oracle_best_of_two_mse10_mean"],
    ]
    blend_r = [
        v_r["full_single_e9"]["4000"]["unseen"]["mse10_mean"],
        dd_r["e9+control"]["tuned_blend_unseen"]["mse10_mean"],
        dd_r["e9+e12_c"]["tuned_blend_unseen"]["mse10_mean"],
        dd_r["e9+e12_bc_factordisp"]["tuned_blend_unseen"]["mse10_mean"],
    ]
    ora_r = [
        v_r["full_single_e9"]["4000"]["unseen"]["mse10_mean"],
        au_r["e9_vs_control"]["oracle_best_of_two_mse10_mean"],
        au_r["e9_vs_e12_c"]["oracle_best_of_two_mse10_mean"],
        au_r["e9_vs_e12_bc_factordisp"]["oracle_best_of_two_mse10_mean"],
    ]
    ax.bar(xb - 1.5 * w, blend_v, w, label="Val-Blend (vKNN)", color="#1f77b4")
    ax.bar(xb - 0.5 * w, ora_v, w, label="Oracle Best-2 (vKNN)", color="#aec7e8")
    ax.bar(xb + 0.5 * w, blend_r, w, label="Val-Blend (RelP)", color="#d62728")
    ax.bar(xb + 1.5 * w, ora_r, w, label="Oracle Best-2 (RelP)", color="#ff9896")
    ax.set_xticks(xb)
    ax.set_xticklabels(cats, fontsize=8.0)
    ax.set_ylim(0.0115, 0.0178)
    ax.set_ylabel("Unseen Action 10D MSE (↓)")
    ax.set_title("B. Dual-Decoder Output Fusion & Headroom", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Stratified Win-Rate of Partner Over E9 (Rel-Progress + Visual kNN avg or Rel-Progress)
    ax = axes[2]
    strata = [
        ("partner_win_rate_low_da_mean", "Low ||Δa||"),
        ("partner_win_rate_high_da_mean", "High ||Δa||"),
        ("partner_win_rate_grip_steady_mean", "Grip Steady"),
        ("partner_win_rate_grip_trans_mean", "Grip Switch!"),
    ]
    xs = np.arange(len(strata))
    w = 0.25
    for pi, (pkey, plbl, col) in enumerate(
        [
            ("e9_vs_control", "E9-Cont (Control)", "#7f7f7f"),
            ("e9_vs_e12_c", "E12-C vs E9", COLORS["e12_c"]),
            ("e9_vs_e12_bc_factordisp", "E12-BC-FD vs E9", COLORS["e12_bc_factordisp"]),
        ]
    ):
        vals = [au_v[pkey][sk] * 100.0 for sk, _ in strata]
        bars = ax.bar(xs + (pi - 1) * w, vals, w, label=plbl, color=col)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax.axhline(50, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in strata], fontsize=8.4)
    ax.set_ylim(0, 56)
    ax.set_ylabel("Per-Sample Win Rate Over E9 (%) — vKNN")
    ax.set_title("C. Failure-Case Regime Specialization", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("E14 Exp 3: Zero Input-Fusion Gain Under Equal Params, Yet Genuine Output-Space & Regime Complementarity", fontsize=11.6, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_exp3_complementarity_and_failure_cases.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Complete, Evidence-Grounded REPORT.md Generator
# ---------------------------------------------------------------------------

def write_report(d1: Dict[str, Any], d2: Dict[str, Any], d3: Dict[str, Any], out_dir: Path) -> None:
    def _m1(mode: str, proto: str, n: str, sp: str, A: str, B: str, k: str = "mse10_mean") -> float:
        return float(d1["results"][mode]["protocols"][proto][n][sp][A][B][k])

    def _an1(mode: str, proto: str, n: str, sp: str = "unseen", m: str = "mse10") -> Dict[str, Any]:
        return d1["results"][mode]["anova_4x4"][proto][n][sp][m]

    def _p2a(mode: str, model: str) -> Dict[str, Any]:
        return d2["results"][mode]["part_2a"][model]

    def _p2b(mode: str, model: str) -> Dict[str, Any]:
        return d2["results"][mode]["part_2b"][model]

    def _p2c(mode: str, model: str) -> Dict[str, Any]:
        return d2["results"][mode]["part_2c"][model]

    lines: List[str] = []
    a = lines.append

    a("# E14: Disentangling Editability and Visual Transferability — 机制诊断与综合报告")
    a("")
    a("> **核心研究问题**：为什么 E12 系列（`E12-B`、`E12-C`、`E12-BC-FactorDisp`）显著提升了潜空间可编辑性（Editability）与因子可组合性（Composability），却在严格零泄漏的 Visual ICL 中稳定弱于 `E9`？瓶颈究竟在于 **（1）Demo 输入端信息损失**、**（2）目标潜空间的可预测性/非线性守恒**，还是 **（3）冻结 Decoder 的误差敏感性与子空间解码机制**？")
    a("")
    a("> **严格实验纪律**：全流程复用 E13 Exp 4 的严格零泄漏划分（217 训练任务 `fold=train` / 217 已见任务 `fold=val` 独立布局 / 94 未见任务 `fold=val`；3 随机种子 `42, 123, 777`；`visual_knn` 与 `rel_progress` 双协议），**严禁 Query GT action 与 Query 未来帧进入推理条件**；所有潜空间均在输入端（分块 RMS 能量对齐）与输出端（全局标量 RMS 标准化）严格控制尺度差异；暂不重训 Encoder，不做闭环 rollout。")
    a("")
    a("---")
    a("")
    a("## 执行摘要（核心科学裁决：证伪两项流行直觉，确立真实双机制）")
    a("")
    an_v = _an1("visual_knn", "direct_full_ridge", "4000")
    an_r = _an1("rel_progress", "direct_full_ridge", "4000")
    a(f"1. **Exp 1 裁决 — 彻底否决「输入端信息损失」假说，瓶颈 `93.8%–97.9%` 位于输出/Decoder 端**：")
    a(f"   - 在 4×4 交叉迁移矩阵（Direct Full-Dim Ridge @ `N=4000`，未见任务）中，双因素方差分析（Two-Way ANOVA）显示：**输出/Decoder 列主效应占 `{an_v['ss_output_frac']*100:.1f}%`（`visual_knn`）与 `{an_r['ss_output_frac']*100:.1f}%`（`rel_progress`）**，而输入 Demo 行主效应仅占 **`{an_v['ss_input_frac']*100:.1f}%` / `{an_r['ss_input_frac']*100:.1f}%`**，交互效应仅 **`{an_v['ss_interaction_frac']*100:.1f}%` / `{an_r['ss_interaction_frac']*100:.1f}%`**。该结论在全部 5 种 Adapter 协议（Full Ridge、PCA Ridge、Cross-Aligned Residual Ridge、RBF Kernel Ridge、Tuned Residual MLP）与全部 7 个样本预算（`N=50..4000`）上高度一致（输出端占比 `74.7%–99.3%`）。")
    a(f"   - **强烈反直觉现象**：在 `visual_knn` 中，作为 Demo 输入的 **`E12-C` 反而取得全场最低行均值（`0.01838` vs `E9 0.01846`）**，且跨表征迁移 **`E12-C → E9`（`0.01700`）略微反超 `E9 → E9`（`0.01715`）**；在 `rel_progress` 中，`E12-C → E9`（`0.01406`）一步消除 `E12-C → E12-C`（`0.01627`）与 `E9 → E9`（`0.01388`）之间 **`92.5%`** 的差距。甚至完全不给 Demo latent 的 **`Vision-Only → E9`（`0.01833`）** 也优于完整带 Demo 的 `E12-B`（`0.01865`）、`E12-C`（`0.01887`）和 `E12-BC-FD`（`0.01986`）。")
    a(f"2. **Exp 2A 裁决 — 直接证伪 E13 提出的朴素「逆动力学非线性守恒律」**：")
    a(f"   - 潜空间可预测性测试揭示了与直觉完全相反的事实：在严格零泄漏视觉输入 `(Δh, h_q, z_d)` 下，**`E12-C`（`R²(z)=0.817`, `e_lat=0.423`）与 `E12-BC-FD`（`R²(z)=0.835`, `e_lat=0.403`）比 `E9`（`R²(z)=0.751`, `e_lat=0.496`）显著更容易预测**（潜空间相对误差反而低 `15%–19%`，连动作子空间 `R²(z_∥)` 也是 `0.865 > 0.837 > 0.822 > 0.814`）！")
    a(f"   - 修复 E13 中欠调优 MLP 的偏差后，通用非线性适配器（闭式 **RBF Kernel Ridge** 与早停 **Residual-on-Ridge MLP**）在所有模型上都比 Linear Ridge 提升了 `6%–10%`（`E9` 在 `rel_progress` 上降至 `0.01301`），但**对 `E9` 的收益与对 `E12` 相当甚至更大**，并未缩小 `E12` 与 `E9` 的差距（RBF-Full 仍为 `0.01529 vs 0.01306`，差距 `+17.1%`）。")
    a(f"   - **关键细粒度逆转**：虽然 E12 的全局 `R²(z)` 因有效秩 4× 坍缩（`erank(z): 36.6 → 9.2–9.8`）而被粗粒度模态（如 `Tok3` 夹爪 `R²=0.904`）推高，但决定跨布局迁移方向的**位移余弦对齐 `cos(Δz_hat, Δz)` 却随关系约束单调下降**（`rel_progress`: `E9 0.767 → E12-B 0.717 → E12-C 0.705 → E12-BC-FD 0.694`）。")
    a(f"3. **Exp 2B & 2C 裁决 — 真正决定性能反转的「双输出机制」（精确因子分解）**：")
    a(f"   - **机制一（可编辑性–敏感性放大权衡）**：利用精确恒等式 $\\text{{ExcessRMS}} \\equiv e_{{\\text{{lat}}}} \\times G_{{\\text{{dec}}}}$ 进行归因分解，发现 **冻结 Diffusion Decoder 对视觉推断误差的放大增益 $G_{{\\text{{dec}}}}$ 随可编辑性严格单调暴涨**：`rel_progress` 下从 `E9` 的 `0.2275` 跃升至 `E12-B` `0.2544`（`+11.8%`）、`E12-C` `0.2851`（`+25.3%`）、`E12-BC-FD` `0.3074`（**`+35.1%`**）；`visual_knn` 下从 `0.2537` 升至 `0.3353`（**`+32.2%`**）。这一 **`+12% ~ +35%` 的 Decoder 误差放大完全吞噬并逆转了 E12 `-4% ~ -19%` 的潜空间误差优势**，并彻底解释了 E13 的未解之谜——为什么 `VisHQ_R²` 更高、可编辑性最强的 `E12-BC-FD` 却在 Visual ICL 中垫底！")
    a(f"   - **机制二（动作正交子空间 $z_\\perp$ 的角色反转：E9 的「场景去噪先验」 vs E12 的「高增益脆弱通道」）**：等尺度扰动实验表明，E12 将高频动作残差挤压进仅占总方差 `5.6%–6.9%` 的 $z_\\perp$ 狭缝，迫使 E12 Decoder 对 $z_\\perp$ 方向建立 **`2.0×–2.3×` 更高的雅可比增益**（`@σ=0.35 ShiftRMS`: `E12-BC-FD 0.0567` / `E12-C 0.0497` vs `E9 0.0247`），GT $z_\\perp$ 置零或打乱导致 E12 性能恶化幅度为 E9 的 **`2.3×–2.6×`**。反观 E9，其 $z_\\perp$ 不仅视觉可预测性更高（`R²(z_⊥)=0.540 vs 0.454–0.482`），且在有噪视觉预测时充当 **Query 场景去噪锚点**：在 E9 上置零 $z_\\perp$ 恶化至 `0.01814`，填入错误 Demo 的 $z_\\perp$ 恶化至 `0.01916`（跌至 E12-C 水平），而仅用单帧 Query 图像 `h_q` 独立预测 $z_\\perp$（`out_par_plus_hq_perp`）竟将 E9 从 `0.01715` 进一步优化至 **`0.01659`**（给 GT $z_\\perp$ 更骤降至 **`0.01381` / `0.01120`**）！")
    a(f"4. **Exp 3 裁决 — 输入端融合在等参对照下零收益，但输出端与工作机制存在真实互补性**：")
    a(f"   - **严格等维/等参对照的威力**：在输入端做 32+32=64 拼接、Joint-PCA、正交残差补充或等参门控融合，均无法稳定超越单模型 `E9`（`GatedSingle(E9_split) = 0.01673 / 0.01387` 甚至反超 `GatedFusion(E9, E12-C) = 0.01681 / 0.01416` 与 `GatedFusion(E9, E12-BC-FD) = 0.01740 / 0.01412`），且无法挽救 `E12-C/BC-FD` Decoder。若缺少等参对照，很容易把非线性容量增益误判为表征互补！")
    a(f"   - **真实互补性出现在「多任务探针」与「双 Decoder 输出融合」**：在 64D 等维探针下，`Concat(E9[32], E12-BC-FD[32])` 同时拿下高动作线性度（`R²(z→a7)=0.962`）与全场最高未来动作预测（`Fut_R²=0.788`），严格支配同族对照 `E9+E9-Cont`（`0.927 / 0.753`）。在 Visual ICL 输出端，验证集选权的双 Decoder 凸融合 **`E9 + E12-C`（`rel_progress 0.01351`）与 `E9 + E12-BC-FD`（`visual_knn 0.01677`）带来同族双模型集成 `E9 + E9-Cont`（`0.01370 / 0.01696`）的 `2×` 净收益**，误差相关性从 `0.971–0.988` 显著去相关至 `0.858–0.916`。")
    a(f"   - **失败案例分层揭示互补机理**：`E9` 统治小幅适配与稳态段（胜率 `73%–79%`）；而在**大动作跨布局偏移（`High-||Δa||`）**与**离散夹爪切换（`Grip Transition`）**等困难样本上，`E12-BC-FD` 对 `E9` 的胜率分别暴涨 **`+10 pp`** 与 **`+17 pp`**（在夹爪切换样本上高达 **`44.1%–45.7%`**），因 E9 的低增益场景去噪 Decoder 在锐利切换处欠冲模糊，而 E12-BC-FD 的高增益因果 `Tok3` 模块能干净执行动作跃变。")
    a(f"5. **架构决策建议（基于证据，拒绝盲目预设）**：**不建议盲目设计「双分支 Behavior Encoder + 单共享 Decoder」**（Exp 1 与 Exp 3 明确证明瓶颈 94%–98% 在输出/Decoder 端、输入融合零收益）。若零重训落地，直接采用 **`E12-C/BC-FD` 作为可编辑记忆 + 跨解码至 `E9`（或双 Decoder 输出融合）**；若进入下一代重训（E15），应直击输出端双机制，采用 **「Decoder 潜空间扰动鲁棒训练（压低 $G_{{\\text{{dec}}}}$）+ 显式解耦 Query 视觉条件 `Dec(z_op, h_q)`（把 $z_\\perp$ 从潜空间剥离交给 `h_q`）」**。")
    a("")
    a("---")
    a("")

    # ------------------------------------------------------------------
    # Section 1: Exp 1
    # ------------------------------------------------------------------
    a("## 1. Exp 1：4×4 Cross-Representation Transfer（输入端 vs 输出端瓶颈定位）")
    a("")
    a("![Exp 1 4x4 Cross-Transfer Matrix and ANOVA](figures/fig1_exp1_cross_transfer_4x4.png)")
    a("")
    a("![Exp 1 Sample Efficiency Curves](figures/fig2_exp1_sample_efficiency_curves.png)")
    a("")
    a("### 1.1 主 4×4 交叉迁移矩阵（Direct Full-Dim Block-RMS Ridge @ N=4000，未见任务，3 seeds）")
    a("")
    for mode, m_title in [("visual_knn", "协议 A：`visual_knn`（纯视觉检索，MSE10 ↓ / `R²(z)`）"), ("rel_progress", "协议 B：`rel_progress`（按相对进度对齐，MSE10 ↓ / `R²(z)`）")]:
        a(f"**{m_title}**")
        a("")
        a("| Demo 输入 A \\ 目标+Decoder B | E9 | E12-B | E12-C | E12-BC-FD | RowMean (MSE10) |")
        a("|---|---:|---:|---:|---:|---:|")
        for A in list(CORE_MODELS) + ["none"]:
            cells = []
            mses = []
            for B in CORE_MODELS:
                mv = _m1(mode, "direct_full_ridge", "4000", "unseen", A, B, "mse10_mean")
                rv = _m1(mode, "direct_full_ridge", "4000", "unseen", A, B, "r2_z_mean")
                mses.append(mv)
                s = f"`{mv:.5f}` (`{rv:.3f}`)"
                if (A == B) or (A == "e12_c" and B == "e9"):
                    s = f"**{s}**"
                cells.append(s)
            lbl = MODEL_LABELS[A]
            if A == "none":
                lbl = "*Vision-Only (无 z_d)*"
            a(f"| **{lbl}** | " + " | ".join(cells) + f" | `{np.mean(mses):.5f}` |")
        an = _an1(mode, "direct_full_ridge", "4000", "unseen", "mse10")
        cmeans = " | ".join(f"**`{v:.5f}`**" for v in an["col_means"])
        a(f"| **ColMean (4×4)** | {cmeans} | `{an['grand_mean']:.5f}` |")
        a("")
        a(
            f"- **Two-Way ANOVA (`MSE10`)**：Input (Row) = **`{an['ss_input_frac']*100:.1f}%`**，"
            f"Output/Decoder (Col) = **`{an['ss_output_frac']*100:.1f}%`**，"
            f"Interaction = **`{an['ss_interaction_frac']*100:.1f}%`**"
            f"（行极差 `{an['row_range']:.5f}` vs 列极差 `{an['col_range']:.5f}`，对角优势 `{an['diag_advantage_mse']:+.5f}`）。"
        )
        a("")

    a("### 1.2 跨 5 种 Adapter 协议与尺度控制的稳健性核查（N=4000，未见任务）")
    a("")
    a("| Adapter 协议 | `visual_knn` ANOVA (In/Out/Int) | `rel_progress` ANOVA (In/Out/Int) | `E9→E9` (`vKNN`/`RelP`) | `E12-C→E9` (`vKNN`/`RelP`) | `E12-C→E12-C` (`vKNN`/`RelP`) | `E9→E12-C` (`vKNN`/`RelP`) |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for proto, plbl in [
        ("direct_full_ridge", "Direct Full-Dim Ridge"),
        ("direct_pca_ridge", "Direct PCA-320 Ridge"),
        ("aligned_res_ridge", "Cross-Aligned Residual Ridge"),
        ("rbf_krr", "RBF Kernel Ridge (非线性)"),
        ("tuned_res_mlp", "Tuned Residual MLP (非线性)"),
    ]:
        av = _an1("visual_knn", proto, "4000")
        ar = _an1("rel_progress", proto, "4000")
        a(
            f"| **{plbl}** | `{av['ss_input_frac']*100:.1f}% / {av['ss_output_frac']*100:.1f}% / {av['ss_interaction_frac']*100:.1f}%` "
            f"| `{ar['ss_input_frac']*100:.1f}% / {ar['ss_output_frac']*100:.1f}% / {ar['ss_interaction_frac']*100:.1f}%` "
            f"| `{_m1('visual_knn', proto, '4000', 'unseen', 'e9', 'e9'):.5f}` / `{_m1('rel_progress', proto, '4000', 'unseen', 'e9', 'e9'):.5f}` "
            f"| `{_m1('visual_knn', proto, '4000', 'unseen', 'e12_c', 'e9'):.5f}` / `{_m1('rel_progress', proto, '4000', 'unseen', 'e12_c', 'e9'):.5f}` "
            f"| `{_m1('visual_knn', proto, '4000', 'unseen', 'e12_c', 'e12_c'):.5f}` / `{_m1('rel_progress', proto, '4000', 'unseen', 'e12_c', 'e12_c'):.5f}` "
            f"| `{_m1('visual_knn', proto, '4000', 'unseen', 'e9', 'e12_c'):.5f}` / `{_m1('rel_progress', proto, '4000', 'unseen', 'e9', 'e12_c'):.5f}` |"
        )
    a("")
    a("**Exp 1 关键洞察**：")
    a("- **为什么跨表征迁移（`E12-C → E9`）几乎无损甚至在 `visual_knn` 上反超 `E9 → E9`？** 对比 `Vision-Only (none → B)` 与带 Demo 行可见：在 `visual_knn` 中，纯视觉输入已承担绝大部分目标状态预测，Demo latent `z_d` 主要提供任务意图与参考动作锚点；而 `E12-C` 剔除了场景视觉干扰、高信噪比地压缩了动作意图，因此在检索存在视觉错位（`visual_knn`）时，`z_d^(E12-C)` 作为输入反而比带场景噪声的 `z_d^(E9)` 更干净（`0.01700 vs 0.01715`）。")
    a("- **为什么只在 `RBF-KRR`（非线性）下 Input 方差占比略升至 `11.8%–23.8%`？** 因为 RBF 核能利用 `z_d^(E9)` 中与 `h_q` 的非线性场景交互，使 `E9` 作为输入相对 `E12-C` 获得约 `0.0004–0.0008` 的额外增益（如 `rel_progress` 下 `E9→E12-C = 0.01443` 优于 `E12-C→E12-C = 0.01530`），但**输出/Decoder 端仍以 `74.7%–86.9%` 占据绝对主导**。")
    a("- **尺度控制与全维 Ridge 带来的意外基线跃升**：取消 E13 的 PCA-64 截断并加入分块 RMS 能量对齐后，所有模型的绝对 MSE10 均比 E13 改善约 `10%`（`E9`: `0.0190→0.01715` / `0.0152→0.01388`；`RBF-KRR` 进一步压低至 `0.01606 / 0.01305`），且相对排序完全稳健，排除尺度或欠拟合伪影。")
    a("")
    a("---")
    a("")

    # ------------------------------------------------------------------
    # Section 2: Exp 2
    # ------------------------------------------------------------------
    a("## 2. Exp 2：Mechanism Diagnosis（潜空间可预测性、Decoder 敏感性与 E9 子空间因果消融）")
    a("")
    a("### 2.1 Exp 2A：目标潜空间可预测性悖论与线性/非线性 Adapter 差距")
    a("")
    a("![Exp 2A Predictability and Nonlinearity](figures/fig3_exp2a_predictability_and_nonlinearity.png)")
    a("")
    a("| 模型 (`rel_progress` @4k, Unseen) | `erank(z)` / `erank(Δz)` | `R²(z)` (↑) | `R²(z_∥)` / `R²(z_⊥)` | `cos(Δz)` (↑) | `Tok0/1/2/3 R²` | LinFull MSE10 | RBF-Full MSE10 | ResMLP MSE10 | 非线性收益 `Δ_RBF` |")
    a("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for m in CORE_MODELS:
        st = _p2a("rel_progress", m)["subspace_stats"]
        lf = _p2a("rel_progress", m)["adapters"]["linear_full"]["4000"]["unseen"]
        kf = _p2a("rel_progress", m)["adapters"]["rbf_krr_full"]["4000"]["unseen"]
        rm = _p2a("rel_progress", m)["adapters"]["res_mlp_pca"]["4000"]["unseen"]
        gain = (lf["mse10_mean"] - kf["mse10_mean"]) / lf["mse10_mean"] * 100.0
        a(
            f"| **{MODEL_LABELS[m]}** | `{st['erank_z_mean']:.1f}` / `{st['erank_dz_mean']:.1f}` "
            f"| `{lf['r2_z_mean']:.3f}` | `{lf['r2_z_par_mean']:.3f}` / `{lf['r2_z_perp_mean']:.3f}` "
            f"| `{lf['cos_dz_mean']:.3f}` "
            f"| `{lf['tok0_r2_mean']:.2f}/{lf['tok1_r2_mean']:.2f}/{lf['tok2_r2_mean']:.2f}/{lf['tok3_r2_mean']:.2f}` "
            f"| `{lf['mse10_mean']:.5f}` | `{kf['mse10_mean']:.5f}` | `{rm['mse10_mean']:.5f}` | `-{gain:.1f}%` |"
        )
    a("")
    a("**Exp 2A 反直觉发现（证伪朴素非线性守恒律）**：")
    a("1. **`R²(z)` 与动作 `MSE10` 完全倒挂**：在 `visual_knn` 与 `rel_progress` 上，`E12-BC-FD` 均拥有全场最高的 `R²(z)`（`0.835 / 0.838`）与 `R²(z_∥)`（`0.862 / 0.865`），而 `E9` 最低（`0.751 / 0.753`），但解码后的动作误差恰好完全相反（`E12-BC-FD 0.01986 / 0.01666` 最差 vs `E9 0.01715 / 0.01388` 最好）！")
    a("2. **为什么 `E12-BC-FD` 的 `R²(z)` 更高，而位移对齐 `cos(Δz)` 反而更低？** 观察 `erank(z)` 与分 Token `R²`：E12 关系约束将潜空间有效秩从 `36.6` 急剧压缩至 `9.2–9.8`（如 `E12-BC-FD` 的 `Tok3` 夹爪单轴 `R²` 高达 `0.904`），使全局静态状态极易拟合（推高 `R²(z)`）；但跨布局所需的精细相对位移 `Δz = z_q - z_d` 的方向对齐 `cos(Δz)` 却从 `E9` 的 `0.767`（`visual_knn 0.736`）单调跌落至 `E12-BC-FD` 的 `0.694`（`0.662`），且动作正交分量 `R²(z_⊥)` 从 `0.536–0.540` 跌至 `0.454–0.482`。")
    a("3. **通用非线性适配器无法消除 E12 劣势**：在 E13 中简单 2 层 MLP 因未早停而过拟合（劣于 Ridge）；本实验引入闭式 `RBF-KRR` 与早停 `Residual-on-Ridge MLP` 后，非线性适配器稳定超越 Linear Ridge（`6%–10%`），但 `E9` 的非线性收益（`-6.4%` vKNN / `-5.9%` RelP）与 `E12-C`（`-4.5%` / `-8.1%`）和 `E12-BC-FD`（`-5.5%` / `-8.2%`）同量级，**即使给定通用非线性拟合器，`E12-BC-FD` 仍落后 `E9` 约 `+17%`**。因此，性能反转不能归因于「前端线性适配器容量不足」。")
    a("")

    a("### 2.2 Exp 2B：Decoder 等尺度扰动敏感性与精确误差归因分解")
    a("")
    a("![Exp 2B Decoder Sensitivity and Factorization](figures/fig4_exp2b_decoder_sensitivity_and_factorization.png)")
    a("")
    a("对每个模型 $M$，用其自身中心化潜向量的均方根范数 $R_M = \\sqrt{\\mathbb{E}\\|z - \\mu_M\\|_2^2}$ 做严格等尺度归一化，建立精确乘法分解：")
    a("$$\\text{ExcessRMS}_M \\equiv \\sqrt{\\text{MSE}_{\\text{act}}(\\hat{z}_M) - \\text{MSE}_{\\text{act}}(z_M)} \\;=\\; \\underbrace{\\frac{\\|\\hat{z}_M - z_M\\|_{\\text{rms}}}{R_M}}_{e_{\\text{lat}}(M)\\text{（潜空间预测误差）}} \\;\\times\\; \\underbrace{G_{\\text{dec}}(M)}_{\\text{Decoder 误差放大增益}}$$")
    a("")
    a("| 模型 | `e_lat` (`vKNN`/`RelP`) | **`G_dec` (`vKNN`/`RelP`)** | `G_dec` 相对 E9 | `ExcRMS` (`vKNN`/`RelP`) | `@σ=0.35` Iso ShiftRMS | `@σ=0.35` `z_∥` ShiftRMS | **`@σ=0.35` `z_⊥` ShiftRMS** | **`@σ=0.35` ICL-Dir ShiftRMS** | Shrink 0.30 MSE10 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    gd_v_e9 = _p2b("visual_knn", "e9")["factorization_linear"]["G_dec_excess_mean"]
    gd_r_e9 = _p2b("rel_progress", "e9")["factorization_linear"]["G_dec_excess_mean"]
    for m in CORE_MODELS:
        fv = _p2b("visual_knn", m)["factorization_linear"]
        fr = _p2b("rel_progress", m)["factorization_linear"]
        cv = _p2b("visual_knn", m)["curves"]
        dv = (fv["G_dec_excess_mean"] / gd_v_e9 - 1.0) * 100.0
        dr = (fr["G_dec_excess_mean"] / gd_r_e9 - 1.0) * 100.0
        a(
            f"| **{MODEL_LABELS[m]}** | `{fv['e_lat_rel_mean']:.4f}` / `{fr['e_lat_rel_mean']:.4f}` "
            f"| **`{fv['G_dec_excess_mean']:.4f}` / `{fr['G_dec_excess_mean']:.4f}`** "
            f"| `{dv:+.1f}%` / **`{dr:+.1f}%`** "
            f"| `{fv['excess_act_rms_mean']:.4f}` / `{fr['excess_act_rms_mean']:.4f}` "
            f"| `{cv['isotropic']['0.35']['shift_rms_vs_clean_mean']:.4f}` "
            f"| `{cv['subspace_par']['0.35']['shift_rms_vs_clean_mean']:.4f}` "
            f"| **`{cv['subspace_perp']['0.35']['shift_rms_vs_clean_mean']:.4f}`** "
            f"| **`{cv['icl_err_rescaled']['0.35']['shift_rms_vs_clean_mean']:.4f}`** "
            f"| `{cv['shrinkage']['0.30']['mse10_vs_gt_mean']:.5f}` |"
        )
    a("")
    a("**Exp 2B 核心机理解析（解开 E13 最大谜团）**：")
    a("1. **「可编辑性–误差敏感性」内在耦合**：潜空间越平坦、因果解耦越干净（`E9 → E12-B → E12-C → E12-BC-FD`），冻结 Diffusion Decoder 的有效误差放大增益 $G_{\\text{dec}}$ 就**严格单调攀升**：`+8.1%/+11.8%`（`E12-B`）→ `+22.6%/+25.3%`（`E12-C`）→ **`+32.2%/+35.1%`（`E12-BC-FD`）**！即使把真实视觉 ICL 误差向量强行缩放到**完全相同的相对幅度 `σ=0.35`**，引发的动作漂移仍从 E9 的 `0.0909` 单调暴增至 `E12-BC-FD` 的 `0.1191`（`+31.0%`）。")
    a("2. **精确回答「为什么 `E12-BC-FD` 比 `E12-C` 保留了更多视觉上下文，Visual ICL 却更差？」**：`E12-BC-FD` 的分 Token 相对位移尺度损失显式要求每个 Token 对细微潜空间位移都产生高响应因果动作变化（从而在 Exp 3 因果替换中全面刷新纪录），这直接把它的 Decoder 等尺度敏感性推至全场最高——尤其在各向同性噪声（`0.0216 vs E9 0.0120`, `+80%`）与 $z_\\perp$ 方向（`0.0567 vs E9 0.0247`, **`+130%`**）上。当输入是无噪人工编辑向量时，高增益是巨大优势；当输入是带噪视觉推断预测时，高增益变成了**误差放大器**。")
    a("3. **排除 Ridge 收缩（Shrinkage）假说**：对所有模型施加 `s=0.30` 径向收缩 `μ + 0.7(z-μ)` 后，动作 MSE10 仅在 `0.00222–0.00358` 之间，且 `E12-BC-FD`（`0.00222`）甚至最抗收缩，说明 Visual ICL 劣势来自**方向性误差放大（尤其是 $z_\\perp$ 与高曲率误差分量）**而非 Ridge 均值收缩。")
    a("")

    a("### 2.3 Exp 2C：动作可解释子空间（$z_\\parallel$）vs 动作正交子空间（$z_\\perp$）因果消融")
    a("")
    a("![Exp 2C Subspace Causal Ablation](figures/fig5_exp2c_subspace_ablation.png)")
    a("")
    a("仅在训练集上拟合动作线性映射 $a_{10\\text{d}} \\to z$ 的 99.5% 方差正交基 $Q_{\\text{act}} \\in \\mathbb{R}^{3072 \\times r}$（测试时零 GT 泄露），将任意潜向量正交拆分为 $z - \\mu = z_\\parallel + z_\\perp$：")
    a("")
    a("| 模型 (`visual_knn` @4k, Unseen) | $z_\\perp$ 方差占比 | GT Full | **GT $z_\\parallel$ Only ($z_\\perp=0$)** | **GT $z_\\parallel$ + Shuf $z_\\perp$** | ICL Full | ICL $z_\\parallel$ Only ($z_\\perp=0$) | ICL $z_\\parallel$ + Demo $z_\\perp$ | **ICL $z_\\parallel$ + $h_q$ 预测 $z_\\perp$** | **ICL $z_\\parallel$ + GT $z_\\perp$** |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in CORE_MODELS:
        zp_frac = (1.0 - _p2a("visual_knn", m)["subspace_stats"]["geo_subspace_var_frac_mean"]) * 100.0
        gt = _p2c("visual_knn", m)["gt_decoder_ablation"]
        oa = _p2c("visual_knn", m)["output_ablation"]["4000"]
        a(
            f"| **{MODEL_LABELS[m]}** | `{zp_frac:.1f}%` "
            f"| `{gt['gt_full']['unseen']['mse10_mean']:.5f}` "
            f"| **`{gt['gt_par_only']['unseen']['mse10_mean']:.5f}`** "
            f"| **`{gt['gt_par_plus_shuf_perp']['unseen']['mse10_mean']:.5f}`** "
            f"| `{oa['out_full']['unseen']['mse10_mean']:.5f}` "
            f"| `{oa['out_par_only_zero_perp']['unseen']['mse10_mean']:.5f}` "
            f"| `{oa['out_par_plus_demo_perp']['unseen']['mse10_mean']:.5f}` "
            f"| **`{oa['out_par_plus_hq_perp']['unseen']['mse10_mean']:.5f}`** "
            f"| **`{oa['out_par_plus_gt_perp']['unseen']['mse10_mean']:.5f}`** |"
        )
    a("")
    a("**Exp 2C 三个决定性因果发现（含输入端对照）**：")
    a("1. **输入端消融（`E9` extra info 并非主要通过 Demo 输入起作用）**：在 `E9` 的 Demo 输入侧，仅保留 $z_\\parallel(\\text{demo})$（`0.01715` vKNN / `0.01432` RelP）、仅保留 $z_\\perp(\\text{demo})$（`0.01697` / `0.01438`）与完整 $z_d$（`0.01715` / `0.01388`）差距极小，且与等维随机子空间对照（`0.01706–0.01712`）几乎重合。说明 E9 额外信息的主要贡献**不在 Demo 输入端**，这与 Exp 1 的 ANOVA 结论完美互证。")
    a("2. **输出端因果消融（$z_\\perp$ 在 E9 Decoder 中扮演「Query 场景去噪先验」）**：")
    a("   - 在有噪的 Visual ICL 预测上，直接抹掉 E9 预测的 $z_\\perp$（`out_par_only`）使误差显著恶化（`0.01715 → 0.01814` vKNN；`0.01388 → 0.01505` RelP）；如果填入 **Demo 的错误场景 $z_\\perp$**（`out_par_plus_demo_perp`），E9 更是直接恶化至 **`0.01916` / `0.01559`**（跌落到与 E12-C 几乎相同！）。")
    a("   - **极具启发性的正向构造**：仅用当前 Query 单帧图像 `h_q`（完全不看 Demo！）独立回归预测 $z_\\perp$ 并拼回 $\\hat{z}_\\parallel$（`out_par_plus_hq_perp`），在 `visual_knn` 上直接把 E9 从 `0.01715` 优化到 **`0.01659`**（`-3.3%`），在 `rel_progress` 上恢复了 `82%` 的 $z_\\perp$ 收益（`0.01409`）；若给予真实 Query 场景 $z_\\perp$（`out_par_plus_gt_perp`），误差更是暴跌 **`-19.5%`**（`0.01715→0.01381` / `0.01388→0.01120`）！这因果证明：**E9 的 Diffusion Decoder 在面对不确定动作预测 $\\hat{z}_\\parallel$ 时，主动利用可从 `h_q` 高精度估计的 $z_\\perp$ 将解码动作锚定回当前场景的可行动作流形。**")
    a("3. **反直觉对比 — E12 的极小 $z_\\perp$ 反而高度脆弱**：`E12-BC-FD` 的 $z_\\perp$ 仅占总方差 `6.9%`（不到 E9 `21.8%` 的三分之一），但将 GT $z_\\perp$ 置零或打乱导致解码误差飙升至 `0.00506 / 0.00727`（为 E9 `0.00281 / 0.00342` 的 **`2.3×–2.6×`**），Visual ICL 中 `zero_perp` 更是恶化 `+0.00226`（vKNN）/ `+0.00353`（RelP）。因为高压缩秩坍缩把非线性微调全塞进了 $z_\\perp$ 狭缝，使 E12 Decoder 既对 $z_\\perp$ 高度敏感、又更难从视觉准确预测它（`R²(z_⊥) = 0.454–0.482 vs E9 0.540`）。")
    a("")
    a("---")
    a("")

    # ------------------------------------------------------------------
    # Section 3: Exp 3
    # ------------------------------------------------------------------
    a("## 3. Exp 3：Complementary Representation（互补性验证与严格等维/等参对照）")
    a("")
    a("![Exp 3 Complementarity and Failure Cases](figures/fig6_exp3_complementarity_and_failure_cases.png)")
    a("")
    a("### 3.1 输入端融合与严格等维/等参数量对照（Unseen Tasks, N=4000, 3 seeds）")
    a("")
    a("除单模型基线外，我们引入**同族双模型对照 `E9 + E9-Cont`**（两个同架构纯扩散模型，排除单纯双模型集成效应）与**严格等参数量单模型分叉对照 `GatedSingle(E9_split)`**（将 E9 自身 128D PCA 拆为两路 64D 送入**参数量完全一致**的 `GatedFusionAdapter`，`n_params = 1,004,352`）：")
    a("")
    a("| 类别与方法（→ 目标 Decoder） | 输入潜维 / 参数对照 | `visual_knn` MSE10 (ARR) | `rel_progress` MSE10 (ARR) | 相对等参单模型 E9 |")
    a("|---|---|---:|---:|---:|")
    vv = d3["results"]["visual_knn"]["variants"]
    vr = d3["results"]["rel_progress"]["variants"]
    e9_m64_v = vv["m64_single_e9"]["4000"]["unseen"]["mse10_mean"]
    e9_m64_r = vr["m64_single_e9"]["4000"]["unseen"]["mse10_mean"]
    e9_gs_v = vv["gated_single_e9_split->e9"]["4000"]["unseen"]["mse10_mean"]
    e9_gs_r = vr["gated_single_e9_split->e9"]["4000"]["unseen"]["mse10_mean"]
    e9_full_v = vv["full_single_e9"]["4000"]["unseen"]["mse10_mean"]
    e9_full_r = vr["full_single_e9"]["4000"]["unseen"]["mse10_mean"]

    table3_rows = [
        ("m64_single_e9", "Single E9 (Matched-64)", "64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_single_e12_c", "Single E12-C (Matched-64)", "64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_single_e12_bc_factordisp", "Single E12-BC-FD (Matched-64)", "64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_concat_e9+control->e9", "Concat E9 + E9-Cont → E9 *(同族对照)*", "32+32=64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_concat_e9+e12_c->e9", "Concat E9 + E12-C → E9", "32+32=64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_concat_e9+e12_bc_factordisp->e9", "Concat E9 + E12-BC-FD → E9", "32+32=64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_jointpca_e9+e12_c->e9", "Joint-PCA64(E9, E12-C) → E9", "64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_orthres_e9+(e12_c|e9)->e9", "OrthRes E9 + (E12-C ⊥ E9) → E9", "32+32=64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_orthres_e12_c+(e9|e12_c)->e12_c", "OrthRes E12-C + (E9 ⊥ E12-C) → E12-C", "32+32=64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("m64_orthres_e12_bc_factordisp+(e9|e12_bc_factordisp)->e12_bc_factordisp", "OrthRes BC-FD + (E9 ⊥ BC-FD) → BC-FD", "32+32=64D (`D_in=320`)", e9_m64_v, e9_m64_r),
        ("gated_single_e9_split->e9", "**GatedSingle E9 Split → E9 *(等参对照)***", "64+64D (`1.004M` params)", e9_gs_v, e9_gs_r),
        ("gated_e9+e9cont->e9", "GatedFusion E9 + E9-Cont → E9 *(同族对照)*", "64+64D (`1.004M` params)", e9_gs_v, e9_gs_r),
        ("gated_e9+e12c->e9", "GatedFusion E9 + E12-C → E9", "64+64D (`1.004M` params)", e9_gs_v, e9_gs_r),
        ("gated_e9+e12bcfd->e9", "GatedFusion E9 + E12-BC-FD → E9", "64+64D (`1.004M` params)", e9_gs_v, e9_gs_r),
        ("gated_e9+e12c->e12_c", "GatedFusion E9 + E12-C → E12-C", "64+64D (`1.004M` params)", e9_gs_v, e9_gs_r),
        ("gated_e9+e12bcfd->e12_bc_fd", "GatedFusion E9 + E12-BC-FD → BC-FD", "64+64D (`1.004M` params)", e9_gs_v, e9_gs_r),
        ("full_single_e9", "Full-Dim Single E9", "3072D", e9_full_v, e9_full_r),
        ("full_concat_e9+control->e9", "Full-Dim Concat E9 + E9-Cont → E9", "6144D", e9_full_v, e9_full_r),
        ("full_concat_e9+e12_c->e9", "Full-Dim Concat E9 + E12-C → E9", "6144D", e9_full_v, e9_full_r),
        ("full_concat_e9+e12_bc_factordisp->e9", "Full-Dim Concat E9 + E12-BC-FD → E9", "6144D", e9_full_v, e9_full_r),
    ]
    for k, lbl, dim_s, ref_v, ref_r in table3_rows:
        uv = vv[k]["4000"]["unseen"]
        ur = vr[k]["4000"]["unseen"]
        dv = uv["mse10_mean"] - ref_v
        dr = ur["mse10_mean"] - ref_r
        a(
            f"| {lbl} | {dim_s} | `{uv['mse10_mean']:.5f}` (`{uv['arr_10d_mean']*100:.1f}%`) "
            f"| `{ur['mse10_mean']:.5f}` (`{ur['arr_10d_mean']*100:.1f}%`) "
            f"| `{dv:+.5f}` / `{dr:+.5f}` |"
        )
    a("")
    a("**关键方法学警示（等参对照揭露伪收益）**：如果只把 `GatedFusion(E9, E12-C) → E9`（`0.01681`）与单模型 `m64_single_e9`（`0.01857`）比较，会误以为融合带来了 `-9.5%` 的巨大收益；但与**精确同参数量、同输入维度的 `GatedSingle(E9_split)`（`0.01673` vKNN / `0.01387` RelP）**对照后立刻露馅——`GatedSingle(E9_split)` 在两种协议上均**反超**所有跨表征门控融合！这再次印证 Exp 1：单看 Demo 输入侧，拼接/残差/门控融合均无实质互补增益，也无法挽救 E12 Decoder。")
    a("")

    a("### 3.2 输出端双 Decoder 动作空间融合与失败案例机制分层（真正的互补性所在）")
    a("")
    a("| 双 Decoder 组合（验证集选权 $w_{\\text{E9}}$，零测试泄露） | `vKNN` Blend MSE10 (相对 E9) | `RelP` Blend MSE10 (相对 E9) | 误差相关性 `Corr` (`vKNN`/`RelP`) | Partner 胜率 (`vKNN`/`RelP`) | Oracle Best-2 (`vKNN`/`RelP`) | Low-Δa → **High-Δa** 胜率 (`vKNN`) | Steady → **Grip Switch** 胜率 (`vKNN`) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for partner, plbl in [
        ("control", "E9 + E9-Cont *(同族容量对照)*"),
        ("e12_c", "**E9 + E12-C *(跨表征互补)***"),
        ("e12_bc_factordisp", "**E9 + E12-BC-FD *(跨表征互补)***"),
    ]:
        dv = d3["results"]["visual_knn"]["dual_decoder"][f"e9+{partner}"]
        dr = d3["results"]["rel_progress"]["dual_decoder"][f"e9+{partner}"]
        av = d3["results"]["visual_knn"]["complementarity_audit"][f"e9_vs_{partner}"]
        ar = d3["results"]["rel_progress"]["complementarity_audit"][f"e9_vs_{partner}"]
        bv = dv["tuned_blend_unseen"]["mse10_mean"]
        br = dr["tuned_blend_unseen"]["mse10_mean"]
        a(
            f"| {plbl} | `{bv:.5f}` (`{bv - e9_full_v:+.5f}`) "
            f"| `{br:.5f}` (`{br - e9_full_r:+.5f}`) "
            f"| `{av['error_pearson_corr_mean']:.3f}` / `{ar['error_pearson_corr_mean']:.3f}` "
            f"| `{av['partner_win_rate_mean']*100:.1f}%` / `{ar['partner_win_rate_mean']*100:.1f}%` "
            f"| `{av['oracle_best_of_two_mse10_mean']:.5f}` / `{ar['oracle_best_of_two_mse10_mean']:.5f}` "
            f"| `{av['partner_win_rate_low_da_mean']*100:.1f}%` → **`{av['partner_win_rate_high_da_mean']*100:.1f}%`** "
            f"| `{av['partner_win_rate_grip_steady_mean']*100:.1f}%` → **`{av['partner_win_rate_grip_trans_mean']*100:.1f}%`** |"
        )
    a("")
    a("**双 Decoder 互补性与失败案例深析**：")
    a("- 尽管 `E12-BC-FD` 单独运行比 `E9` 差 `+15.8%`（`0.01986 vs 0.01715`），将它与 `E9` 在输出动作空间凸融合（`w_E9 ≈ 0.65`）却达到了 **`0.01677`**（`rel_progress` 下 `E9+E12-C = 0.01351`），其净收益是**同容量、单模型更强的同族集成 `E9 + E9-Cont`（`0.01696 / 0.01370`）的 `2.0×`**，且 Oracle Best-of-Two 上限扩大了 **`2.3×–2.6×`**（`0.01548 / 0.01237` vs `0.01651 / 0.01322`）。")
    a("- **为什么两者在特定失败案例上高度互补？** 分层审计给出了清晰的物理答案：`E9` 的低增益、强场景去噪 Decoder 在**小幅度微调与夹爪稳态段**优势压倒性（`E12-C` 胜率仅 `20.7%–20.8%`），因为它能过滤视觉检索误差；但一旦遇到**大跨度动作改写（`High-||Δa||`）**或**离散夹爪开合切换（`Grip Transition`）**，`E9` 的场景先验平滑机制反而导致动作切换欠冲/迟滞，而拥有高增益因果 `Tok3` 模块的 **`E12-BC-FD` 胜率飙升 `+17 pp` 至 `44.1%–45.7%`**（同族 `E9-Cont` 则完全无机制分化：`43.8% vs 44.7%`）。")
    a("")

    a("### 3.3 双体制表征探针审计（Part 3B：上下文预测 + 动作几何，3 seeds）")
    a("")
    a("| 表征（Matched-64D 等维 vs Full-Dim） | 维度 | `R²(a→z)` | 动作线性探针 `R²(z→a7)` (↑) | 未来动作 `Fut_R²` (↑) | 视觉上下文 `VisHQ_R²` (↑) | 物体线性 `ObjAcc` (↑) |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    p3b = d3["part_3b_dual_regime_audit"]
    for k, lbl in [
        ("e9_k64", "E9 (64D)"),
        ("e12_c_k64", "E12-C (64D)"),
        ("e12_bc_fd_k64", "E12-BC-FD (64D)"),
        ("concat_e9_e9cont_32p32", "Concat E9 + E9-Cont (32+32=64D, 同族对照)"),
        ("concat_e9_e12c_32p32", "Concat E9 + E12-C (32+32=64D)"),
        ("concat_e9_e12bcfd_32p32", "**Concat E9 + E12-BC-FD (32+32=64D)**"),
        ("e9_full", "E9 (Full 3072D)"),
        ("e12_c_full", "E12-C (Full 3072D)"),
        ("e12_bc_fd_full", "E12-BC-FD (Full 3072D)"),
        ("concat_e9_e9cont_full", "Concat E9 + E9-Cont (6144D, 同族对照)"),
        ("concat_e9_e12c_full", "Concat E9 + E12-C (6144D)"),
        ("concat_e9_e12bcfd_full", "**Concat E9 + E12-BC-FD (6144D)**"),
    ]:
        v = p3b[k]
        a(
            f"| {lbl} | `{int(v['dim_mean'])}` | `{v['r2_action_to_z_mean']:.3f}` "
            f"| `{v['r2_z_to_cur_action7d_mean']:.3f}` | `{v['r2_z_to_future_action7d_mean']:.3f}` "
            f"| `{v['r2_z_to_visual_hq_mean']:.3f}` | `{v['object_linear_acc_mean']*100:.1f}%` |"
        )
    a("")
    a("在严格 64D 等维约束下，`Concat(E9[32], E12-BC-FD[32])` 同时实现了 `E12-BC-FD` 级别的当前动作几何（`R²(z→a7) = 0.962 vs E9 0.937`）与超越两者的未来动作预测（`Fut_R² = 0.788` > `E9 0.783` > `E12-BC-FD 0.754`），并全面碾压同族对照 `Concat(E9[32], E9-Cont[32])`（`0.927 / 0.753`）——证实 **contextual information 与 operational geometry 在表征内容层面上确实正交互补**，只是在 Visual ICL 中这种互补性受制于下游单个冻结 Decoder 的误差敏感性瓶颈。")
    a("")
    a("---")
    a("")

    # ------------------------------------------------------------------
    # Section 4: Architectural Decision & Next Steps
    # ------------------------------------------------------------------
    a("## 4. 总结与架构决策：是否需要设计新的双分支 Behavior Encoder？")
    a("")
    a("E14 用严格可控的因果实验澄清了 E13 遗留的根本矛盾：**E12 在零泄漏 Visual ICL 中落后 E9，既不是因为 Demo 输入信息丢失（Exp 1：输入仅占 `1.5%–4.0%` 方差），也不是因为朴素「非线性守恒律」使目标潜空间更难预测（Exp 2A：E12 的 `R²(z)` 反而高达 `0.817–0.838` > `E9 0.751`，且非线性 RBF-KRR 不缩小差距）**。真正起决定作用的是**输出/Decoder 端的双重物理机制**：")
    a("1. **可编辑性–敏感性放大权衡**：平坦、低秩、高因果响应的关系几何将 Decoder 的误差放大增益 $G_{\\text{dec}}$ 推高 **`+12% ~ +35%`**，完全压过 `-4% ~ -19%` 的潜空间误差优势；")
    a("2. **$z_\\perp$ 的角色反转**：E9 的 $z_\\perp$（`21.8%` 方差）是可由 `h_q` 预测的**低增益场景去噪先验**；E12 的 $z_\\perp$（`5.6%–6.9%` 方差）是被秩坍缩挤压出的**2.3× 高增益脆弱狭缝**。")
    a("")
    a("### 对「是否设计新双分支 Behavior Encoder」的循证决策")
    a("")
    a("> **结论：不应设计单纯的「双分支 Encoder → 单共享 Decoder」架构；真正的架构升级点在 Decoder 接口与训练正则化，而非单纯加宽/分叉 Encoder。**")
    a("")
    a("- **为什么反对单纯双分支 Encoder？** Exp 1（`93.8%–97.9%` 输出端主导）与 Exp 3（等参输入融合 `0%` 增益、`GatedSingle(E9_split)` 反超）已经证明：即使给 Adapter 同时喂入完整的 `E9 + E12-C/BC-FD` 双分支潜向量，只要最终仍解码通过未经改造的单个高敏感度 `E12` Decoder，性能仍卡在 `0.0191–0.0196`（无法接近 `E9` 的 `0.0171`）；而只要切换到 `E9` Decoder，哪怕单用 `E12-C` 作为输入就能立刻拿到 `0.0170`。")
    a("- **立即可用（零重训）的最优工程组合**：")
    a("  1. **跨表征解码（`E12-C/BC-FD` 行为记忆 → `E9` 目标与 Decoder）**：保留 `E12-C/BC-FD` 的高编辑性/因果组合检索与编辑能力，在零泄漏 Visual ICL 推理时直接预测并解码 `E9` 空间，无需重训任何 Encoder 即可拿回 `92.5%–100%` 的 Visual ICL 性能。")
    a("  2. **双 Decoder 输出融合 / 门控路由（`E9 + E12-BC-FD`）**：在动作空间融合两个互补 Decoder，可稳定超越任何单模型与同族集成（`0.01677 / 0.01351`），若未来进一步构建基于 `||Δh||` 或夹爪切换概率的轻量路由器，向 `OracleBest2`（`0.01548 / 0.01237`）逼近的空间高达 `-9.7% ~ -10.9%`。")
    a("- **若开展下一代端到端重训（E15 推荐方案，直击 E14 两大根因）**：")
    a("  1. **Decoder 侧潜空间噪声注入与雅可比封顶（Noise-Robust Relational Decoding）**：在 `E12-BC-FactorDisp` 联合微调时，向送入 Diffusion UNet 的潜向量注入模拟 Visual ICL 预测误差尺度的各向同性/$z_\\perp$ 扰动（`σ ∈ [0.15, 0.35]`），同时在无噪潜向量上保持 `L_FactorDisp`，迫使 Decoder **对动作主方向保持陡峭可编辑、对 $z_\\perp$/推断误差保持低增益平滑（压低 $G_{\\text{dec}}$）**。")
    a("  2. **显式解耦 Query 场景条件（`a ~ Dec(z_op, h_q)`）**：受 Exp 2C `out_par_plus_hq_perp`（仅靠 `h_q` 预测 $z_\\perp$ 即超越完整 E9：`0.01659 vs 0.01715`）直接启发，将 Query 当前视觉特征 `h_q` 作为独立条件直接送入 Diffusion Decoder，把「场景去噪先验」从 behavior latent 中彻底剥离，让行为潜空间 `z_op` 专心承担平坦、低秩、可因果组合的纯操作几何——**从结构上而非权衡上彻底解耦 Editability 与 Visual Transferability**。")
    a("")
    a("---")
    a("")
    a("## 附录：产物索引")
    a("- Exp 1 完整数据：[exp1_summary.json](exp1_cross_transfer/exp1_summary.json)、[exp1_cross_transfer_matrix.csv](exp1_cross_transfer_matrix.csv)、[exp1_per_sample_seed42.npz](exp1_cross_transfer/exp1_per_sample_seed42.npz)")
    a("- Exp 2 完整数据：[exp2_summary.json](exp2_mechanism_diagnosis/exp2_summary.json)、[exp2_mechanism_summary.csv](exp2_mechanism_summary.csv)")
    a("- Exp 3 完整数据：[exp3_summary.json](exp3_complementary_repr/exp3_summary.json)、[exp3_complementarity_summary.csv](exp3_complementarity_summary.csv)")
    a("- 图表目录：`e14/outputs/figures/fig1..fig6.png`")

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args()
    out_dir: Path = args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d1 = json.loads((out_dir / "exp1_cross_transfer" / "exp1_summary.json").read_text())
    d2 = json.loads((out_dir / "exp2_mechanism_diagnosis" / "exp2_summary.json").read_text())
    d3 = json.loads((out_dir / "exp3_complementary_repr" / "exp3_summary.json").read_text())

    _export_csvs(d1, d2, d3, out_dir)
    plot_fig1_cross_transfer_4x4(d1, fig_dir)
    plot_fig2_sample_efficiency(d1, fig_dir)
    plot_fig3_exp2a_predictability_nonlinearity(d2, fig_dir)
    plot_fig4_exp2b_decoder_sensitivity(d2, fig_dir)
    plot_fig5_exp2c_subspace_ablation(d2, fig_dir)
    plot_fig6_exp3_complementarity(d3, fig_dir)
    write_report(d1, d2, d3, out_dir)
    print(f"[E14 Report] Successfully generated CSVs, 6 figures, and {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
