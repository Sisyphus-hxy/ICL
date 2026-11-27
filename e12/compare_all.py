"""Unified evaluation and comparative reporting for E9 vs E9-Cont vs E12-A vs E12-B vs E12-C.

Evaluates every model checkpoint through the exact same 5-pillar `e12.eval_suite.run_full_eval`,
generates publication-quality comparison figures, exports consolidated CSVs, and writes
`e12/outputs/comparison/REPORT.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import DEFAULT_STAGE1_LOCAL_CKPT, OUTPUT_ROOT, STAGE1_ANALYSIS_DIR
from .eval_suite import run_full_eval


MODEL_SPECS_DEFAULT: List[Tuple[str, str, Path, Path]] = [
    ("e9", "E9 (Stage 1)", DEFAULT_STAGE1_LOCAL_CKPT, STAGE1_ANALYSIS_DIR),
    ("control", "E9-Cont (L_diff)", OUTPUT_ROOT / "control" / "last_ema.pt", OUTPUT_ROOT / "control" / "eval"),
    ("e12_a", "E12-A (Global Sim)", OUTPUT_ROOT / "e12_a" / "last_ema.pt", OUTPUT_ROOT / "e12_a" / "eval"),
    ("e12_b", "E12-B (Multi-Attr)", OUTPUT_ROOT / "e12_b" / "last_ema.pt", OUTPUT_ROOT / "e12_b" / "eval"),
    ("e12_c", "E12-C (Disp Cons.)", OUTPUT_ROOT / "e12_c" / "last_ema.pt", OUTPUT_ROOT / "e12_c" / "eval"),
]

COLORS = {
    "e9": "#4d4d4d",
    "control": "#8c8c8c",
    "e12_a": "#1f77b4",
    "e12_b": "#2ca02c",
    "e12_c": "#d62728",
}


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _extract_headline_row(key: str, display: str, d: Dict[str, Any]) -> Dict[str, Any]:
    s1 = d["suite1_reconstruction"]
    s2 = d["suite2_relational"]
    s3 = d["suite3_displacement_arithmetic"]
    s4 = d["suite4_unseen_combinations"]
    s5 = d["suite5_transfer_efficiency"]

    ct = s2["correlations"]["latent_cos"]["cross_task"]
    ct_c = s2["correlations"]["latent_centered_cos"]["cross_task"]
    rq = s2["knn"]["cross_task_retrieval_quality"]["latent_cos"]
    knn_ct = s2["knn"]["cross_task"]["latent_cos"]
    pa = s2["partial_attribute"]
    sc = s2.get("scene_sensitivity", {})
    dc = s3["displacement_consistency"]
    t1 = s3["latent_arithmetic_tier1_matched_start"]
    t2 = s3["latent_arithmetic_tier2_distinct_start"]
    lcr = s4["linear_compositional_ridge"]
    zsa = s4["zero_shot_compositional_arithmetic"]
    lp = s5["linear_probing_cross_task"]["latent"]
    tc = s5["action_change_transfer_efficiency"]["by_n"]
    clg = s5["cross_layout_geometry"]

    # Best-token cross-task Spearman for each attribute (captures E12-B token specialization)
    best_tok = {}
    for attr in ("translation", "rotation_active", "magnitude", "temporal", "gripper"):
        best_tok[attr] = max(
            s2["correlations"][f"token_{k}_cos"]["cross_task"][attr]["spearman"]
            for k in range(4)
        )

    return {
        "model_key": key,
        "model": display,
        # Suite 1: Fidelity
        "val_self_diff": s1["val_self_diff_loss"],
        "ddim_10d_mse": s1["ddim_self"]["action10_mse"],
        "ddim_pos_mse": s1["ddim_self"]["pos_mse"],
        "ddim_rot6d_mse": s1["ddim_self"]["rot6d_mse"],
        "ddim_grip_mse": s1["ddim_self"]["grip_mse"],
        "ddim_7d_mae": s1["ddim_self"]["action7_mae"],
        "exec12_mae": s1["ddim_self"]["exec12_action7_mae"],
        "grip_acc": s1["ddim_self"]["grip_sign_acc"],
        "ts_ratio": s1["timestep_distinctiveness"]["mismatch_to_match_mse10_ratio"],
        # Suite 2: Relational & Cross-Scene
        "ct_rho_bal7d": ct["global_balanced7d"]["spearman"],
        "ct_rho_bal7d_centered": ct_c["global_balanced7d"]["spearman"],
        "ct_rho_trans": ct["translation"]["spearman"],
        "ct_rho_rot_act": ct["rotation_active"]["spearman"],
        "ct_rho_mag": ct["magnitude"]["spearman"],
        "ct_rho_temp": ct["temporal"]["spearman"],
        "ct_rho_grip": ct["gripper"]["spearman"],
        "best_tok_rho_trans": best_tok["translation"],
        "best_tok_rho_rot_act": best_tok["rotation_active"],
        "best_tok_rho_mag": best_tok["magnitude"],
        "best_tok_rho_temp": best_tok["temporal"],
        "best_tok_rho_grip": best_tok["gripper"],
        "ct_knn10_bal7d": knn_ct["global_balanced7d"]["recall@10"],
        "ct_knn10_trans": knn_ct["translation"]["recall@10"],
        "ct_knn10_rot": knn_ct["rotation"]["recall@10"],
        "ct_knn10_mag": knn_ct["magnitude"]["recall@10"],
        "ct_knn10_temp": knn_ct["temporal"]["recall@10"],
        "ct_top1_mse10": rq["top1"]["action10_mse"],
        "ct_top5_mse10": rq["top5"]["action10_mse"],
        "same_task_leak": s2["knn"]["leakage"]["latent_cos"]["top10_same_task_rate"],
        "partial_sep_mean": float(
            np.mean([pa[k]["latent_cos_separation"] for k in ("translation", "rotation", "magnitude", "temporal", "gripper") if k in pa])
        ),
        "scene_cos_mean": sc.get("variants_vs_base_mean_cos", float("nan")),
        "scene_nl2_ratio": sc.get("scene_to_action_nl2_ratio", float("nan")),
        "scene_dec_mse_vs_gt": sc.get("variants_vs_gt_mean_decoder_mse", float("nan")),
        # Suite 3: Displacement & Arithmetic
        "disp_rho_bal7d": dc["delta_balanced7d"]["latent_disp_spearman"],
        "disp_matched_cos_bal7d": dc["delta_balanced7d"]["matched_latent_disp_cos"],
        "disp_matched_cos_grip": dc["delta_gripper"]["matched_latent_disp_cos"],
        "arith_t1_cos": t1["latent_disp_cos"],
        "arith_t1_mse10": t1["ddim_arith"]["action10_mse"],
        "arith_t1_arr": t1["arithmetic_recovery_ratio_10d"],
        "arith_t2_cos": t2["latent_disp_cos"],
        "arith_t2_mse10": t2["ddim_arith"]["action10_mse"],
        "arith_t2_arr": t2["arithmetic_recovery_ratio_10d"],
        "arith_t2_grip_acc": t2["ddim_arith"]["grip_sign_acc"],
        # Suite 4: Unseen Combinations
        "unseen_ridge_mse10": lcr["unseen_ddim"]["action10_mse"],
        "unseen_to_seen_ratio": lcr["unseen_to_seen_mse10_ratio"],
        "unseen_zsa_mse10": zsa["ddim_arith"]["action10_mse"],
        "unseen_zsa_arr": zsa["recovery_ratio_10d"],
        "unseen_zsa_grip_gain": zsa["grip_acc_gain"],
        # Suite 5: Transfer Efficiency
        "probe_r2_bal7d": lp["global_balanced7d"]["r2"],
        "probe_r2_temp": lp["temporal"]["r2"],
        "lin_n50_mse10": tc["50"]["linear_ridge"]["ddim_10d_mse"],
        "lin_n100_mse10": tc["100"]["linear_ridge"]["ddim_10d_mse"],
        "lin_n500_mse10": tc["500"]["linear_ridge"]["ddim_10d_mse"],
        "lin_n2000_mse10": tc["2000"]["linear_ridge"]["ddim_10d_mse"],
        "mlp_n100_mse10": tc["100"]["mlp_2layer"]["ddim_10d_mse"],
        "mlp_n2000_mse10": tc["2000"]["mlp_2layer"]["ddim_10d_mse"],
        "erank_Z": clg["latent_Z_rank"]["erank_energy"],
        "erank_dz": clg["cross_layout_delta_z_rank"]["erank_energy"],
        "dz_to_Z_rms": clg["delta_z_to_Z_rms_ratio"],
    }


def generate_comparison_figures_and_report(
    results: List[Tuple[str, str, Dict[str, Any]]], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = [_extract_headline_row(k, lbl, d) for k, lbl, d in results]
    _save_csv(rows, out_dir / "headline_comparison.csv")

    keys = [k for k, _, _ in results]
    labels = [lbl for _, lbl, _ in results]
    n_m = len(results)
    colors = [COLORS.get(k, "#333333") for k in keys]

    # --- Fig 1: Cross-Task Attribute Alignment + Token Specialization Heatmaps ---
    attrs = ["global_balanced7d", "translation", "rotation_active", "magnitude", "temporal", "gripper"]
    attr_lbls = ["Global\nBal 7D", "Trans", "Rot\n(Active)", "Mag", "Temporal", "Gripper"]
    x = np.arange(len(attrs))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), dpi=160)
    ax = axes[0]
    w = 0.8 / max(n_m, 1)
    for i, (k, lbl, d) in enumerate(results):
        ct = d["suite2_relational"]["correlations"]["latent_cos"]["cross_task"]
        vals = [ct[a]["spearman"] for a in attrs]
        ax.bar(x + (i - (n_m - 1) / 2.0) * w, vals, width=w, label=lbl, color=colors[i])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attr_lbls, fontsize=9)
    ax.set_ylabel("Cross-Task Spearman ρ (Full Latent Cos)")
    ax.set_title("Cross-Task Action–Latent Alignment (Full 3072D Latent)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Right: Best-Token Attribute Alignment (shows Multi-Attribute Factorization in E12-B!)
    ax2 = axes[1]
    t_attrs = ["translation", "rotation_active", "magnitude", "temporal", "gripper"]
    t_lbls = ["Trans", "Rot (Active)", "Mag", "Temporal", "Gripper"]
    xt = np.arange(len(t_attrs))
    for i, (k, lbl, d) in enumerate(results):
        vals = [
            max(
                d["suite2_relational"]["correlations"][f"token_{tk}_cos"]["cross_task"][a]["spearman"]
                for tk in range(4)
            )
            for a in t_attrs
        ]
        ax2.bar(xt + (i - (n_m - 1) / 2.0) * w, vals, width=w, label=lbl, color=colors[i])
    ax2.set_xticks(xt)
    ax2.set_xticklabels(t_lbls, fontsize=9)
    ax2.set_ylabel("Best-Token Cross-Task Spearman ρ")
    ax2.set_title("Attribute-Specialized Token Alignment (Max over 4 Tokens)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_relational_alignment_comparison.png")
    plt.close(fig)

    # --- Fig 2: Token Specialization Matrix for E9 vs E12-A vs E12-B vs E12-C ---
    show_models = [(k, lbl, d) for k, lbl, d in results if k != "control"]
    fig, axes = plt.subplots(1, len(show_models), figsize=(4.2 * len(show_models), 3.8), dpi=160)
    if len(show_models) == 1:
        axes = [axes]
    for ax, (k, lbl, d) in zip(axes, show_models):
        mat = np.array(
            [
                [
                    d["suite2_relational"]["correlations"][f"token_{tk}_cos"]["cross_task"][a]["spearman"]
                    for a in t_attrs
                ]
                for tk in range(4)
            ]
        )
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=0.95, aspect="auto")
        ax.set_xticks(np.arange(len(t_lbls)))
        ax.set_xticklabels(["Tr", "Rot", "Mag", "Tmp", "Grp"], fontsize=8.5)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(["Tok 0", "Tok 1", "Tok 2", "Tok 3"], fontsize=8.5)
        ax.set_title(lbl, fontsize=10)
        for r_i in range(4):
            for c_i in range(len(t_attrs)):
                ax.text(c_i, r_i, f"{mat[r_i, c_i]:.2f}", ha="center", va="center", fontsize=8)
    fig.suptitle("Per-Token Behavioral Attribute Specialization (Cross-Task Spearman ρ)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_token_specialization_heatmaps.png")
    plt.close(fig)

    # --- Fig 3: Cross-Scene Retrieval, Scene Invariance, Latent Arithmetic & Unseen Combos ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)
    xm = np.arange(n_m)

    # Left: Reconstruction Fidelity vs Cross-Task Top-1 Retrieval MSE10
    ax = axes[0]
    rec_mse = [r["ddim_10d_mse"] * 1000.0 for r in rows]  # x1e-3
    ret_mse = [r["ct_top1_mse10"] * 1000.0 for r in rows]
    ax.bar(xm - 0.18, rec_mse, 0.35, label="Self Recon MSE10 (×1e-3)", color="#2ca02c")
    ax.bar(xm + 0.18, ret_mse, 0.35, label="Cross-Task Top-1 MSE10 (×1e-3)", color="#1f77b4")
    for i in range(n_m):
        ax.text(i - 0.18, rec_mse[i] + 0.15, f"{rec_mse[i]:.2f}", ha="center", fontsize=7.5)
        ax.text(i + 0.18, ret_mse[i] + 0.15, f"{ret_mse[i]:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xm)
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel("10D Action MSE (×1e-3, ↓)")
    ax.set_title("Fidelity Preservation vs Cross-Task Retrieval")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Middle: Tier-2 Parallel Transport Arithmetic & Zero-Shot Unseen Combo Arithmetic ARR
    ax = axes[1]
    t2_arr = [r["arith_t2_arr"] * 100.0 for r in rows]
    zsa_arr = [r["unseen_zsa_arr"] * 100.0 for r in rows]
    ax.bar(xm - 0.18, t2_arr, 0.35, label="Tier-2 Parallel Arith ARR (%)", color="#d62728")
    ax.bar(xm + 0.18, zsa_arr, 0.35, label="Unseen Combo Arith ARR (%)", color="#ff7f0e")
    for i in range(n_m):
        ax.text(i - 0.18, t2_arr[i] + 1.0, f"{t2_arr[i]:.1f}%", ha="center", fontsize=7.5)
        ax.text(i + 0.18, zsa_arr[i] + 1.0, f"{zsa_arr[i]:.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(xm)
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel("Action Recovery Ratio (%, ↑)")
    ax.set_ylim(0, 108)
    ax.set_title("Latent Arithmetic & Compositional Transfer")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Right: Scene-Sensitivity Invariance (S1..S10 vs S0_base)
    ax = axes[2]
    sc_mse = [r["scene_dec_mse_vs_gt"] * 1000.0 for r in rows]
    sc_ratio = [r["scene_nl2_ratio"] * 100.0 for r in rows]
    ax2_r = ax.twinx()
    b1 = ax.bar(xm - 0.18, sc_mse, 0.35, label="Scene Drift Decoder MSE vs GT (×1e-3, ↓)", color="#9467bd")
    b2 = ax2_r.bar(xm + 0.18, sc_ratio, 0.35, label="Scene/Action nL2 Ratio (%, ↓)", color="#8c564b", alpha=0.8)
    ax.set_xticks(xm)
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel("Decoder 10D MSE vs GT (×1e-3, ↓)", color="#9467bd")
    ax2_r.set_ylabel("Scene-Shift / Action-Shift nL2 (%, ↓)", color="#8c564b")
    ax.set_title("Controlled Scene-Shift Invariance (S0..S10)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[b1, b2], fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_transfer_arithmetic_scene_invariance.png")
    plt.close(fig)

    # --- Fig 4: Transfer Efficiency Curves (Linear Ridge vs MLP) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=160)
    ns = [50, 100, 250, 500, 1000, 2000]
    for i, (k, lbl, d) in enumerate(results):
        tc = d["suite5_transfer_efficiency"]["action_change_transfer_efficiency"]["by_n"]
        lin_v = [tc[str(n)]["linear_ridge"]["ddim_10d_mse"] for n in ns]
        mlp_v = [tc[str(n)]["mlp_2layer"]["ddim_10d_mse"] for n in ns]
        axes[0].plot(ns, lin_v, "o-", label=lbl, color=colors[i], linewidth=2)
        axes[1].plot(ns, mlp_v, "s-", label=lbl, color=colors[i], linewidth=2)
    for ax, title in [(axes[0], "Linear Ridge Transfer (z_src + W Δa)"), (axes[1], "2-Layer MLP Transfer (z_src + MLP(Δa))")]:
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Training Transfer Samples N (Disjoint Tasks)")
        ax.set_ylabel("Held-Out Task DDIM 10D MSE (↓)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_transfer_efficiency_curves.png")
    plt.close(fig)

    # --- Write Consolidated Markdown REPORT.md ---
    lines = [
        "# E12: Relational Behavior Representation Learning — 完整对比实验报告",
        "",
        "## 1. 实验设计总览（严格单变量对照）",
        "",
        "所有变体保持 E9 Stage 1 完全一致的 `LocalBehaviorEncoder`（6-layer Pre-LN Transformer, 4×768=3072D latent）、`ConditionalUnet1D` 动作解码器与 DDPM-IP 动作重建损失 `L_diff`，均从 `e9/outputs/stage1_local/best.pt` 同一起点出发，以相同 1.0 Epoch 预算（`5,364` steps, `batch=256`, `lr=1e-4` cosine, `EMA=0.999`）在 GPU 4 & GPU 5 上训练，不叠加多个辅助损失：",
        "",
        "- **`E9 (Stage 1)`**：原始冻结基线检查点。",
        "- **`E9-Cont (L_diff)`**：同预算纯扩散损失对照（`λ_rel = 0`），用于严格分离额外训练步数与关系监督的因果贡献。",
        "- **`E12-A (Global Sim)`**：E9 + Global Action Similarity Supervision（参考 arXiv:2609.19846，`λ_rel = 0.15`，对齐全局平衡 7D 动作余弦相似度矩阵）。",
        "- **`E12-B (Multi-Attr)`**：E9 + Multi-Attribute Relational Supervision（`λ_rel = 0.15`，将 4 个 behavior tokens 分别对齐 Translation / Rotation / Magnitude+Temporal / Gripper 关系矩阵，在不改架构的前提下建立局部属性跨场景连接）。",
        "- **`E12-C (Disp Cons.)`**：E9 + Behavior Displacement Consistency（`λ_rel = 0.15`，对轨迹内与跨轨迹行为变化对 `Δz = z(y)-z(x)` 施加方向与相对尺度一致性约束）。",
        "",
        "## 2. 动作重建保真度对比（Suite 1）",
        "",
        "| 模型 | Val Self Diff (↓) | DDIM 10D MSE (↓) | Pos / Rot6D / Grip MSE | 7D MAE (↓) | Exec-12 MAE (↓) | Grip Acc (↑) | TS Mismatch/Match (↑) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | `{r['val_self_diff']:.5f}` | `{r['ddim_10d_mse']:.5f}` | "
            f"`{r['ddim_pos_mse']:.5f}` / `{r['ddim_rot6d_mse']:.5f}` / `{r['ddim_grip_mse']:.5f}` | "
            f"`{r['ddim_7d_mae']:.5f}` | `{r['exec12_mae']:.5f}` | `{r['grip_acc']*100:.2f}%` | `{r['ts_ratio']:.1f}×` |"
        )

    lines += [
        "",
        "## 3. 跨场景行为邻域一致性与场景不变性（Suite 2）",
        "",
        "![Relational Alignment](figures/fig1_relational_alignment_comparison.png)",
        "",
        "![Token Specialization](figures/fig2_token_specialization_heatmaps.png)",
        "",
        "| 模型 | CT ρ Bal7D (Cent.) | CT ρ Trans / Rot / Mag / Temp / Grip | Best-Tok ρ Tr/Rot/Mag/Tmp/Grp | CT kNN@10 Bal7D | CT Top-1 / Top-5 MSE10 (↓) | Task Leak (↓) | Scene S1..10 Cos (↑) / Ratio (↓) / Dec MSE vs GT (↓) |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | `{r['ct_rho_bal7d']:.3f}` (`{r['ct_rho_bal7d_centered']:.3f}`) | "
            f"`{r['ct_rho_trans']:.2f}` / `{r['ct_rho_rot_act']:.2f}` / `{r['ct_rho_mag']:.2f}` / `{r['ct_rho_temp']:.2f}` / `{r['ct_rho_grip']:.2f}` | "
            f"`{r['best_tok_rho_trans']:.2f}`/`{r['best_tok_rho_rot_act']:.2f}`/`{r['best_tok_rho_mag']:.2f}`/`{r['best_tok_rho_temp']:.2f}`/`{r['best_tok_rho_grip']:.2f}` | "
            f"`{r['ct_knn10_bal7d']:.3f}` | `{r['ct_top1_mse10']:.4f}` / `{r['ct_top5_mse10']:.4f}` | "
            f"`{r['same_task_leak']*100:.2f}%` | `{r['scene_cos_mean']:.3f}` / `{r['scene_nl2_ratio']*100:.1f}%` / `{r['scene_dec_mse_vs_gt']:.4f}` |"
        )

    lines += [
        "",
        "## 4. Latent Displacement 一致性、Latent Arithmetic Transfer 与未见行为组合泛化（Suite 3 & Suite 4）",
        "",
        "![Transfer & Arithmetic](figures/fig3_transfer_arithmetic_scene_invariance.png)",
        "",
        "| 模型 | Disp ρ (`Δbal7d`) | Matched `cos(Δz)` (Bal7D / Grip) | Tier-1 Arith MSE10 (ARR) | **Tier-2 Parallel Arith MSE10 (ARR / cos / Grip)** | Unseen Combo Ridge MSE10 (Unseen/Seen) | **Unseen Combo Zero-Shot Arith MSE10 (ARR / Grip Gain)** |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | `{r['disp_rho_bal7d']:.3f}` | `{r['disp_matched_cos_bal7d']:.3f}` / `{r['disp_matched_cos_grip']:.3f}` | "
            f"`{r['arith_t1_mse10']:.4f}` (`{r['arith_t1_arr']*100:.1f}%`) | "
            f"**`{r['arith_t2_mse10']:.4f}`** (`{r['arith_t2_arr']*100:.1f}%` / `{r['arith_t2_cos']:.3f}` / `{r['arith_t2_grip_acc']*100:.1f}%`) | "
            f"`{r['unseen_ridge_mse10']:.5f}` (`{r['unseen_to_seen_ratio']:.2f}×`) | "
            f"**`{r['unseen_zsa_mse10']:.4f}`** (`{r['unseen_zsa_arr']*100:.1f}%` / `{r['unseen_zsa_grip_gain']*100:+.1f}pp`) |"
        )

    lines += [
        "",
        "## 5. 迁移效率（样本效率 × 映射复杂度 × 潜空间有效秩，Suite 5）",
        "",
        "![Transfer Efficiency](figures/fig4_transfer_efficiency_curves.png)",
        "",
        "| 模型 | Linear Probe R² (Bal7D / Temp) | Linear@50 MSE10 | Linear@100 MSE10 | Linear@500 MSE10 | Linear@2000 MSE10 | MLP@100 / MLP@2000 | `erank(Z)` / `erank(Δz)` / `RMS(Δz)/RMS(Z)` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | `{r['probe_r2_bal7d']:.3f}` / `{r['probe_r2_temp']:.3f}` | "
            f"`{r['lin_n50_mse10']:.5f}` | `{r['lin_n100_mse10']:.5f}` | `{r['lin_n500_mse10']:.5f}` | `{r['lin_n2000_mse10']:.5f}` | "
            f"`{r['mlp_n100_mse10']:.5f}` / `{r['mlp_n2000_mse10']:.5f}` | "
            f"`{r['erank_Z']:.1f}` / `{r['erank_dz']:.1f}` / `{r['dz_to_Z_rms']:.3f}` |"
        )

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[E12 Compare] Written consolidated report to {out_dir / 'REPORT.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "comparison")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--ddim-samples", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[Tuple[str, str, Dict[str, Any]]] = []
    for key, display, ckpt_path, eval_dir in MODEL_SPECS_DEFAULT:
        if not ckpt_path.is_file():
            print(f"[E12 Compare] Skipping {key}: missing checkpoint {ckpt_path}", flush=True)
            continue
        summary_file = eval_dir / "eval_summary.json"
        if summary_file.is_file() and not args.force_eval:
            print(f"[E12 Compare] Reusing existing eval for {key}: {summary_file}", flush=True)
            summary = json.loads(summary_file.read_text())
        else:
            summary = run_full_eval(
                ckpt_path,
                eval_dir,
                label=key,
                device=device,
                ddim_samples=args.ddim_samples,
            )
        results.append((key, display, summary))

    generate_comparison_figures_and_report(results, args.out_dir)


if __name__ == "__main__":
    main()
