"""Stage 1: Training-Free Relational Analysis of Existing E9 Stage 1 Latent Space.

Runs the full 5-pillar evaluation suite on frozen `e9/outputs/stage1_local/best.pt`,
generates diagnostic visualizations, and writes a structured `REPORT.md` to guide the
design of E12-A / E12-B / E12-C.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import DEFAULT_STAGE1_LOCAL_CKPT, STAGE1_ANALYSIS_DIR
from .eval_suite import run_full_eval


ATTR_DISPLAY = [
    ("global_raw7d", "Global\n(Raw 7D)"),
    ("global_balanced7d", "Global\n(Bal 7D)"),
    ("global_10d", "Global\n(10D RBF)"),
    ("translation", "Translation\n(3D Seq)"),
    ("translation_net_dir", "Translation\n(Net Dir)"),
    ("rotation", "Rotation\n(All)"),
    ("rotation_active", "Rotation\n(Active)"),
    ("magnitude", "Motion\nMagnitude"),
    ("temporal", "Temporal\nStructure"),
    ("gripper", "Gripper\nState"),
]


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def generate_stage1_figures_and_report(summary: Dict[str, Any], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    s1 = summary["suite1_reconstruction"]
    s2 = summary["suite2_relational"]
    s3 = summary["suite3_displacement_arithmetic"]
    s4 = summary["suite4_unseen_combinations"]
    s5 = summary["suite5_transfer_efficiency"]

    # --- Fig 1: Action-Latent Spearman Correlation across Attributes & Strata (E9 vs CLIP) ---
    attrs = [a for a, _ in ATTR_DISPLAY]
    labels = [lbl for _, lbl in ATTR_DISPLAY]
    x = np.arange(len(attrs))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=160)
    # Left panel: E9 Latent vs CLIP across Cross-Task vs Same-Episode
    ax = axes[0]
    w = 0.2
    e9_all = [s2["correlations"]["latent_cos"]["all_pairs"][a]["spearman"] for a in attrs]
    e9_ct = [s2["correlations"]["latent_cos"]["cross_task"][a]["spearman"] for a in attrs]
    e9_se = [s2["correlations"]["latent_cos"]["same_episode"][a]["spearman"] for a in attrs]
    clip_ct = [s2["correlations"]["clip_cos"]["cross_task"][a]["spearman"] for a in attrs]

    ax.bar(x - 1.5 * w, e9_se, width=w, label="E9 Latent (Same-Episode)", color="#1f77b4")
    ax.bar(x - 0.5 * w, e9_all, width=w, label="E9 Latent (All Pairs)", color="#2ca02c")
    ax.bar(x + 0.5 * w, e9_ct, width=w, label="E9 Latent (Cross-Task)", color="#d62728")
    ax.bar(x + 1.5 * w, clip_ct, width=w, label="CLIP Input (Cross-Task)", color="#7f7f7f", alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Spearman Rank Correlation ρ")
    ax.set_title("E9 Action–Latent Similarity Alignment by Attribute & Stratum")
    ax.set_ylim(-0.08, 0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Right panel: Per-Token & Subspace (P64 vs Q3008) specialization on Cross-Task pairs
    ax2 = axes[1]
    rep_rows = ["latent_cos", "latent_centered_cos", "token_0_cos", "token_1_cos", "token_2_cos", "token_3_cos"]
    rep_lbls = ["E9 Full", "E9 Centered", "Token 0", "Token 1", "Token 2", "Token 3"]
    if "subspace_P64_cos" in s2["correlations"]:
        rep_rows += ["subspace_P64_cos", "subspace_Q3008_cos"]
        rep_lbls += ["Subspace P(64D)", "Subspace Q(3008D)"]
    mat = np.array(
        [[s2["correlations"][r]["cross_task"][a]["spearman"] for a in attrs] for r in rep_rows]
    )
    im = ax2.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(0.65, float(np.nanmax(mat))))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticks(np.arange(len(rep_lbls)))
    ax2.set_yticklabels(rep_lbls, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax2.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color="black")
    ax2.set_title("Cross-Task Alignment Heatmap (Tokens & P/Q Subspaces)")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_e9_attribute_alignment.png")
    plt.close(fig)

    # --- Fig 2: kNN Neighborhood Consistency, Scene Leakage, and Partial-Attribute Separation ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)
    knn_attrs = ["global_balanced7d", "global_10d", "translation", "rotation", "magnitude", "temporal", "gripper"]
    knn_lbls = ["Bal 7D", "10D RBF", "Trans", "Rot", "Mag", "Temp", "Grip"]
    xk = np.arange(len(knn_attrs))

    ax = axes[0]
    r_unc = [s2["knn"]["unconstrained"]["latent_cos"][a]["recall@10"] for a in knn_attrs]
    r_ct = [s2["knn"]["cross_task"]["latent_cos"][a]["recall@10"] for a in knn_attrs]
    r_ct_c = [s2["knn"]["cross_task"]["latent_centered_cos"][a]["recall@10"] for a in knn_attrs]
    r_clip = [s2["knn"]["cross_task"]["clip_cos"][a]["recall@10"] for a in knn_attrs]
    w = 0.2
    ax.bar(xk - 1.5 * w, r_unc, w, label="E9 Unconstrained", color="#1f77b4")
    ax.bar(xk - 0.5 * w, r_ct, w, label="E9 Cross-Task", color="#d62728")
    ax.bar(xk + 0.5 * w, r_ct_c, w, label="E9 Centered CT", color="#ff7f0e")
    ax.bar(xk + 1.5 * w, r_clip, w, label="CLIP Cross-Task", color="#7f7f7f")
    ax.set_xticks(xk)
    ax.set_xticklabels(knn_lbls, fontsize=8.5)
    ax.set_ylabel("kNN Recall@10 vs Action-kNN")
    ax.set_title("kNN Neighborhood Consistency (k=10)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

    # Middle: Cross-Task Retrieval Action 10D MSE & Same-Task Leakage
    ax = axes[1]
    rq = s2["knn"]["cross_task_retrieval_quality"]
    methods = ["oracle_action_10d", "latent_centered_cos", "latent_cos", "clip_cos", "random_cross_task"]
    m_lbls = ["Oracle\nAct-kNN", "E9 Cent.\nCos", "E9 Raw\nCos", "CLIP\nCos", "Random\nCross-Task"]
    top1_mse = [rq[m]["top1"]["action10_mse"] for m in methods]
    top5_mse = [rq[m]["top5"]["action10_mse"] for m in methods]
    xm = np.arange(len(methods))
    ax.bar(xm - 0.18, top1_mse, 0.35, label="Top-1 Neighbor MSE10", color="#2ca02c")
    ax.bar(xm + 0.18, top5_mse, 0.35, label="Top-5 Neighbors MSE10", color="#1f77b4")
    for i, v in enumerate(top1_mse):
        ax.text(i - 0.18, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xm)
    ax.set_xticklabels(m_lbls, fontsize=8.5)
    ax.set_ylabel("Cross-Task Retrieved Action 10D MSE (↓)")
    ax.set_title("Cross-Task Retrieval Fidelity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Right: Partial-Attribute Separation (Globally Dissimilar Pairs)
    ax = axes[2]
    pa = s2["partial_attribute"]
    pa_keys = [k for k in ("translation", "rotation", "magnitude", "temporal", "gripper") if k in pa]
    xp = np.arange(len(pa_keys))
    sep_e9 = [pa[k]["latent_cos_separation"] for k in pa_keys]
    sep_c9 = [pa[k]["centered_cos_separation"] for k in pa_keys]
    sep_cl = [pa[k]["clip_cos_separation"] for k in pa_keys]
    ax.bar(xp - 0.25, sep_e9, 0.25, label="E9 Cos Gap", color="#d62728")
    ax.bar(xp, sep_c9, 0.25, label="E9 Centered Gap", color="#ff7f0e")
    ax.bar(xp + 0.25, sep_cl, 0.25, label="CLIP Cos Gap", color="#7f7f7f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xp)
    ax.set_xticklabels([k.capitalize() for k in pa_keys], fontsize=8.5)
    ax.set_ylabel("Latent Cos(Partial-Match) − Cos(Non-Match)")
    ax.set_title("Partial-Attribute Connectivity (Global Sim < 0.25)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_e9_knn_and_partial_attributes.png")
    plt.close(fig)

    # --- Fig 3: Displacement Consistency, Latent Arithmetic Transfer, & Transfer Efficiency ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)

    # Left: Displacement Spearman correlation across Δa components
    ax = axes[0]
    dc = s3["displacement_consistency"]
    d_keys = ["delta_balanced7d", "delta_translation", "delta_rotation", "delta_magnitude", "delta_gripper"]
    d_lbls = ["ΔBal 7D", "ΔTrans", "ΔRot", "ΔMag", "ΔGrip"]
    xd = np.arange(len(d_keys))
    ax.bar(xd - 0.18, [dc[k]["latent_disp_spearman"] for k in d_keys], 0.35, label="E9 Δz", color="#1f77b4")
    ax.bar(xd + 0.18, [dc[k]["clip_disp_spearman"] for k in d_keys], 0.35, label="CLIP Δc", color="#7f7f7f")
    ax.set_xticks(xd)
    ax.set_xticklabels(d_lbls, fontsize=8.5)
    ax.set_ylabel("Cross-Task Displacement Spearman ρ")
    ax.set_title("Action-Change vs Latent-Displacement Alignment")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Middle: DDIM-Decoded Latent Arithmetic Transfer (Tier 1 vs Tier 2)
    ax = axes[1]
    t1 = s3["latent_arithmetic_tier1_matched_start"]
    t2 = s3["latent_arithmetic_tier2_distinct_start"]
    cats = ["Base (z_B1)", "Copy (z_A2)", "Arith (z_B1+Δz_A)", "Oracle (z_B2)"]
    xc = np.arange(len(cats))
    v1 = [
        t1["ddim_base_zB1"]["action10_mse"],
        t1["ddim_copy_zA2"]["action10_mse"],
        t1["ddim_arith"]["action10_mse"],
        t1["ddim_oracle"]["action10_mse"],
    ]
    v2 = [
        t2["ddim_base_zB1"]["action10_mse"],
        t2["ddim_copy_zA2"]["action10_mse"],
        t2["ddim_arith"]["action10_mse"],
        t2["ddim_oracle"]["action10_mse"],
    ]
    ax.bar(xc - 0.18, v1, 0.35, label=f"Tier 1 Matched-Start (ARR={t1['arithmetic_recovery_ratio_10d']*100:.1f}%)", color="#2ca02c")
    ax.bar(xc + 0.18, v2, 0.35, label=f"Tier 2 Distinct-Start (ARR={t2['arithmetic_recovery_ratio_10d']*100:.1f}%)", color="#d62728")
    ax.set_xticks(xc)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("DDIM Decoded 10D Action MSE (↓)")
    ax.set_title("Cross-Scene Latent Arithmetic Transfer")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

    # Right: Transfer Sample-Efficiency Curve (Linear Ridge vs 2-Layer MLP)
    ax = axes[2]
    tc = s5["action_change_transfer_efficiency"]
    ns = sorted(int(k) for k in tc["by_n"].keys())
    lin_mse = [tc["by_n"][str(n)]["linear_ridge"]["ddim_10d_mse"] for n in ns]
    mlp_mse = [tc["by_n"][str(n)]["mlp_2layer"]["ddim_10d_mse"] for n in ns]
    ax.plot(ns, lin_mse, "o-", label="Linear Ridge (z_src + W Δa)", color="#1f77b4", linewidth=2)
    ax.plot(ns, mlp_mse, "s--", label="2-Layer MLP (z_src + MLP(Δa))", color="#ff7f0e", linewidth=2)
    ax.axhline(tc["oracle_ddim"]["action10_mse"], color="#2ca02c", linestyle=":", label="Oracle DDIM")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Training Transfer Samples N (Disjoint Tasks)")
    ax.set_ylabel("Held-Out Task DDIM 10D MSE (↓)")
    ax.set_title("E9 Transfer Sample-Efficiency Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_e9_displacement_arithmetic_efficiency.png")
    plt.close(fig)

    # --- Save tabular CSVs ---
    corr_rows = []
    for a, _ in ATTR_DISPLAY:
        corr_rows.append(
            {
                "attribute": a,
                "e9_all_spearman": s2["correlations"]["latent_cos"]["all_pairs"][a]["spearman"],
                "e9_same_ep_spearman": s2["correlations"]["latent_cos"]["same_episode"][a]["spearman"],
                "e9_same_task_diff_layout_spearman": s2["correlations"]["latent_cos"]["same_task_diff_layout"][a]["spearman"],
                "e9_cross_task_spearman": s2["correlations"]["latent_cos"]["cross_task"][a]["spearman"],
                "e9_centered_cross_task_spearman": s2["correlations"]["latent_centered_cos"]["cross_task"][a]["spearman"],
                "e9_negl2_cross_task_spearman": s2["correlations"]["latent_neg_l2"]["cross_task"][a]["spearman"],
                "clip_cross_task_spearman": s2["correlations"]["clip_cos"]["cross_task"][a]["spearman"],
            }
        )
    _save_csv(corr_rows, out_dir / "attribute_correlations.csv")

    # --- Write comprehensive Stage 1 REPORT.md ---
    leak = s2["knn"]["leakage"]
    rq = s2["knn"]["cross_task_retrieval_quality"]
    sc = s2["scene_sensitivity"]
    lp_e9 = s5["linear_probing_cross_task"]["latent"]
    lp_cl = s5["linear_probing_cross_task"]["clip"]
    cl_g = s5["cross_layout_geometry"]

    lines = [
        "# E12 Stage 1 — 现有 E9 Stage 1 潜空间关系结构诊断报告（Training-Free）",
        "",
        f"- **冻结检查点**：`{summary['checkpoint']}`（未做任何微调或重训）",
        f"- **评测样本**：E9 原始 Validation Split 全量 `6,248` starts（`311` 任务 × `50` layouts / `1,549` 独立 val 轨迹），关系矩阵子集 `N = {s2['num_samples']}` 非末尾窗口样本，控制场景敏感性 `12` 场景（`S0..S10`），跨场景行为迁移转置对 `M = {s3['num_transitions']}` 条轨迹内行为变化。",
        "",
        "## 1. 基线动作重建保真度（Suite 1）",
        "",
        "| 指标 | E9 Stage 1 Baseline |",
        "|---|---:|",
        f"| Val Self Diff Loss (6,248) | `{s1['val_self_diff_loss']:.5f}` |",
        f"| Val Wrong-Clip Diff Loss | `{s1['val_wrong_clip_diff_loss']:.5f}` (sens `+{s1['val_clip_sensitivity_diff_loss']:.5f}`) |",
        f"| DDIM 10D MSE (`N={s1['ddim_samples']}`) | `{s1['ddim_self']['action10_mse']:.5f}` (`pos={s1['ddim_self']['pos_mse']:.5f}`, `rot6d={s1['ddim_self']['rot6d_mse']:.5f}`, `grip={s1['ddim_self']['grip_mse']:.5f}`) |",
        f"| DDIM 7D MAE / Exec-12 MAE | `{s1['ddim_self']['action7_mae']:.5f}` / `{s1['ddim_self']['exec12_action7_mae']:.5f}` |",
        f"| Gripper Sign Acc | `{s1['ddim_self']['grip_sign_acc']*100:.2f}%` |",
        f"| Timestep Distinctiveness (`z_cos` / Mismatch÷Match) | `{s1['timestep_distinctiveness']['latent_cosine_sim']:.3f}` / `{s1['timestep_distinctiveness']['mismatch_to_match_mse10_ratio']:.1f}×` |",
        "",
        "## 2. Action Space 与 E9 Latent Space 距离/相似度一致性（分属性 × 分层次）",
        "",
        "![Attribute Alignment](figures/fig1_e9_attribute_alignment.png)",
        "",
        "| 行为属性 | E9 Same-Ep ρ | E9 Same-Task Diff-Layout ρ | **E9 Cross-Task ρ (Cos)** | **E9 Cross-Task ρ (Centered)** | E9 Cross-Task ρ (−L2) | CLIP Cross-Task ρ | Cross-Task Linear Probe R² (E9 vs CLIP) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, _ in ATTR_DISPLAY:
        r_se = s2["correlations"]["latent_cos"]["same_episode"][a]["spearman"]
        r_st = s2["correlations"]["latent_cos"]["same_task_diff_layout"][a]["spearman"]
        r_ct = s2["correlations"]["latent_cos"]["cross_task"][a]["spearman"]
        r_cc = s2["correlations"]["latent_centered_cos"]["cross_task"][a]["spearman"]
        r_l2 = s2["correlations"]["latent_neg_l2"]["cross_task"][a]["spearman"]
        r_cl = s2["correlations"]["clip_cos"]["cross_task"][a]["spearman"]
        lp_str = (
            f"`{lp_e9[a]['r2']:.3f}` vs `{lp_cl[a]['r2']:.3f}`"
            if a in lp_e9
            else "—"
        )
        lines.append(
            f"| `{a}` | `{r_se:.3f}` | `{r_st:.3f}` | **`{r_ct:.3f}`** | **`{r_cc:.3f}`** | `{r_l2:.3f}` | `{r_cl:.3f}` | {lp_str} |"
        )

    lines += [
        "",
        "### 2.1 kNN 邻域一致性、场景泄漏与跨场景检索质量",
        "",
        "![kNN and Partial Attributes](figures/fig2_e9_knn_and_partial_attributes.png)",
        "",
        "- **无约束 Top-10 kNN 场景/任务泄漏率**：",
        f"  - E9 Raw Cos：同轨迹 `{leak['latent_cos']['top10_same_episode_rate']*100:.2f}%`，同任务 `{leak['latent_cos']['top10_same_task_rate']*100:.2f}%`",
        f"  - E9 Centered Cos：同轨迹 `{leak['latent_centered_cos']['top10_same_episode_rate']*100:.2f}%`，同任务 `{leak['latent_centered_cos']['top10_same_task_rate']*100:.2f}%`",
        f"  - CLIP Input：同轨迹 `{leak['clip_cos']['top10_same_episode_rate']*100:.2f}%`，同任务 `{leak['clip_cos']['top10_same_task_rate']*100:.2f}%`",
        f"  - Oracle Action-kNN（自然基准）：同轨迹 `{leak['oracle_action_bal7d']['top10_same_episode_rate']*100:.2f}%`，同任务 `{leak['oracle_action_bal7d']['top10_same_task_rate']*100:.2f}%`",
        "",
        "| 跨任务检索方式 (Top-1 / Top-5) | Action 10D MSE (↓) | 7D MAE (↓) | Pos Cos (↑) | Rot MSE (↓) | Mag MAE (↓) | Temp Cos (↑) | Grip Acc (↑) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m_key, m_name in [
        ("oracle_action_10d", "Oracle Action-10D kNN"),
        ("oracle_action_bal7d", "Oracle Action-Bal7D kNN"),
        ("latent_cos", "**E9 Latent Cos**"),
        ("latent_centered_cos", "**E9 Centered Cos**"),
        ("clip_cos", "CLIP Input Cos"),
        ("random_cross_task", "Random Cross-Task"),
    ]:
        t1m = rq[m_key]["top1"]
        t5m = rq[m_key]["top5"]
        lines.append(
            f"| {m_name} | `{t1m['action10_mse']:.4f}` / `{t5m['action10_mse']:.4f}` | "
            f"`{t1m['action7_mae']:.4f}` / `{t5m['action7_mae']:.4f}` | "
            f"`{t1m['pos_cos']:.3f}` / `{t5m['pos_cos']:.3f}` | "
            f"`{t1m['rot_mse']:.4f}` / `{t5m['rot_mse']:.4f}` | "
            f"`{t1m['mag_path_mae']:.2f}` / `{t5m['mag_path_mae']:.2f}` | "
            f"`{t1m['temporal_cos']:.3f}` / `{t5m['temporal_cos']:.3f}` | "
            f"`{t1m['grip_sign_acc']*100:.1f}%` / `{t5m['grip_sign_acc']*100:.1f}%` |"
        )

    lines += [
        "",
        "### 2.2 局部属性跨场景连接性（全局不同 `global_bal7d_cos < 0.25`，单属性 Top-5% 相似）",
        "",
        "| 单一属性 | E9 Cos (Partial-Match vs Non-Match) | E9 Cos Gap | E9 Centered Gap | CLIP Gap | Partial Spearman ρ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in s2["partial_attribute"].items():
        lines.append(
            f"| `{k}` | `{v['latent_cos_partial_match']:.3f}` vs `{v['latent_cos_non_match']:.3f}` | "
            f"`{v['latent_cos_separation']:+.4f}` | `{v['centered_cos_separation']:+.4f}` | "
            f"`{v['clip_cos_separation']:+.4f}` | `{v['partial_spearman_latent_cos']:.3f}` |"
        )

    if sc:
        lines += [
            "",
            "### 2.3 同动作跨场景不变性（10 视觉扰动场景 vs `S0_base`）",
            "",
            f"- 10 场景均值 `cos = {sc['variants_vs_base_mean_cos']:.4f}`，`norm_L2 = {sc['variants_vs_base_mean_nl2']:.4f}`（占同轨迹异动作距离比例 **`{sc['scene_to_action_nl2_ratio']*100:.1f}%`**），解码漂移 `10D MSE vs Base = {sc['variants_vs_base_mean_decoder_mse']:.5f}`（`vs GT = {sc['variants_vs_gt_mean_decoder_mse']:.5f}`，基准 `S0_base vs GT = {sc['base_gt_decoder_mse']:.5f}`）。",
        ]

    # Section 3: Displacement & Arithmetic
    t1a = s3["latent_arithmetic_tier1_matched_start"]
    t2a = s3["latent_arithmetic_tier2_distinct_start"]
    lines += [
        "",
        "## 3. 行为变化与 Latent Displacement 一致性及 Latent Arithmetic Transfer（Suite 3）",
        "",
        "![Displacement Arithmetic Efficiency](figures/fig3_e9_displacement_arithmetic_efficiency.png)",
        "",
        "### 3.1 跨任务行为变化 `Δa` 与潜空间位移 `Δz` 一致性",
        "",
        "| 行为变化分量 | E9 `Δz` Spearman ρ | CLIP `Δc` Spearman ρ | 强匹配跨场景类比 `cos(Δz_1, Δz_2)` (随机基线) | 强匹配相对 L2 误差 (随机基线) | 匹配对数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in s3["displacement_consistency"].items():
        lines.append(
            f"| `{k}` | **`{v['latent_disp_spearman']:.3f}`** | `{v['clip_disp_spearman']:.3f}` | "
            f"`{v['matched_latent_disp_cos']:.3f}` (`{v['random_latent_disp_cos']:.3f}`) | "
            f"`{v['matched_latent_disp_rel_l2']:.3f}` (`{v['random_latent_disp_rel_l2']:.3f}`) | `{v['n_matched_cross_task_analogies']}` |"
        )

    lines += [
        "",
        "### 3.2 跨场景 DDIM 解码 Latent Arithmetic Transfer (`z_hat = z_B1 + (z_A2 - z_A1)`)",
        "",
        "| 层级 | 四元组数 | `cos(Δz_A, Δz_B)` | Base `z_B1` MSE10 | Copy `z_A2` MSE10 | **Arith MSE10** | Oracle `z_B2` MSE10 | **ARR 10D** (Pos / Grip) | Arith Grip Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| **Tier 1 (Matched-Start)** | `{t1a['num_quadruples']}` | `{t1a['latent_disp_cos']:.3f}` | `{t1a['ddim_base_zB1']['action10_mse']:.4f}` | `{t1a['ddim_copy_zA2']['action10_mse']:.4f}` | **`{t1a['ddim_arith']['action10_mse']:.4f}`** | `{t1a['ddim_oracle']['action10_mse']:.4f}` | **`{t1a['arithmetic_recovery_ratio_10d']*100:.1f}%`** (`{t1a['arithmetic_recovery_ratio_pos']*100:.1f}%` / `{t1a['arithmetic_recovery_ratio_grip']*100:.1f}%`) | `{t1a['ddim_arith']['grip_sign_acc']*100:.1f}%` |",
        f"| **Tier 2 (Distinct-Start Parallel Transport)** | `{t2a['num_quadruples']}` | `{t2a['latent_disp_cos']:.3f}` | `{t2a['ddim_base_zB1']['action10_mse']:.4f}` | `{t2a['ddim_copy_zA2']['action10_mse']:.4f}` | **`{t2a['ddim_arith']['action10_mse']:.4f}`** | `{t2a['ddim_oracle']['action10_mse']:.4f}` | **`{t2a['arithmetic_recovery_ratio_10d']*100:.1f}%`** (`{t2a['arithmetic_recovery_ratio_pos']*100:.1f}%` / `{t2a['arithmetic_recovery_ratio_grip']*100:.1f}%`) | `{t2a['ddim_arith']['grip_sign_acc']*100:.1f}%` |",
        "",
        "## 4. 未见行为组合泛化与迁移效率基线（Suite 4 & Suite 5）",
        "",
        f"- **未见 `(dir × grip)` 组合**（留出 6 组合：`{', '.join(s4['held_out_combos'])}`）：",
        f"  - 线性组合 Ridge (`a_7d → z`)：Seen DDIM 10D MSE = `{s4['linear_compositional_ridge']['seen_ddim']['action10_mse']:.5f}` vs **Unseen DDIM 10D MSE = `{s4['linear_compositional_ridge']['unseen_ddim']['action10_mse']:.5f}`**（比值 `{s4['linear_compositional_ridge']['unseen_to_seen_mse10_ratio']:.2f}×`，Oracle = `{s4['linear_compositional_ridge']['unseen_oracle_ddim']['action10_mse']:.5f}`）。",
        f"  - Zero-Shot 潜空间因子算术（同方向换夹爪迁移，`N={s4['zero_shot_compositional_arithmetic']['num_valid']}`）：Base MSE10 = `{s4['zero_shot_compositional_arithmetic']['ddim_base']['action10_mse']:.4f}` → **Arith MSE10 = `{s4['zero_shot_compositional_arithmetic']['ddim_arith']['action10_mse']:.4f}`**（ARR = `{s4['zero_shot_compositional_arithmetic']['recovery_ratio_10d']*100:.1f}%`，Grip Acc 提升 `{s4['zero_shot_compositional_arithmetic']['grip_acc_gain']*100:+.1f} pp`）。",
        f"- **跨 Layout 残差几何**：潜空间 `Z` 有效秩 `erank = {cl_g['latent_Z_rank']['erank_energy']:.1f}`（`d_90 = {cl_g['latent_Z_rank']['d_90']}`），跨 Layout 残差 `Δz` 有效秩 **`erank = {cl_g['cross_layout_delta_z_rank']['erank_energy']:.1f}`**（`d_90 = {cl_g['cross_layout_delta_z_rank']['d_90']}`），`RMS(Δz)/RMS(Z) = {cl_g['delta_z_to_Z_rms_ratio']:.3f}`。",
        "",
        "### 4.1 跨任务行为变化迁移样本效率曲线（Linear Ridge vs 2-Layer MLP）",
        "",
        "| 训练样本数 N | Linear Ridge MSE10 (ARR) | Linear Grip Acc | 2-Layer MLP MSE10 (ARR) | MLP Grip Acc |",
        "|---:|---:|---:|---:|---:|",
    ]
    tc = s5["action_change_transfer_efficiency"]
    for n_str, vals in sorted(tc["by_n"].items(), key=lambda kv: int(kv[0])):
        lr_v = vals["linear_ridge"]
        ml_v = vals["mlp_2layer"]
        lines.append(
            f"| `{n_str}` | `{lr_v['ddim_10d_mse']:.5f}` (`{lr_v['arr_10d']*100:.1f}%`) | `{lr_v['grip_acc']*100:.1f}%` | "
            f"`{ml_v['ddim_10d_mse']:.5f}` (`{ml_v['arr_10d']*100:.1f}%`) | `{ml_v['grip_acc']*100:.1f}%` |"
        )
    lines.append(
        f"\n> 参考基准（无迁移 Identity `z_src`）：MSE10 = `{tc['identity_no_transfer_ddim']['action10_mse']:.5f}`；Oracle `z_tgt`：MSE10 = `{tc['oracle_ddim']['action10_mse']:.5f}`。"
    )

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument("--output-dir", type=Path, default=STAGE1_ANALYSIS_DIR)
    parser.add_argument("--ddim-samples", type=int, default=2048)
    parser.add_argument("--max-rel-samples", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = run_full_eval(
        args.ckpt,
        args.output_dir,
        label="e9_stage1_baseline",
        device=device,
        ddim_samples=args.ddim_samples,
        max_rel_samples=args.max_rel_samples,
        seed=args.seed,
    )
    generate_stage1_figures_and_report(summary, args.output_dir)
    print(f"[E12 Stage 1] Report written to {args.output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
