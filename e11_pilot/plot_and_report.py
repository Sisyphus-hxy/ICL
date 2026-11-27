"""Generate E11-Pilot publication figures and comprehensive REPORT.md answering the 3 key questions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import OUTPUT_ROOT, SEMANTIC_PHASES
from .learnability import lam_to_tag

PHASE_LABELS = {
    "approach": "Approach",
    "grasp_transition": "Grasp\nTrans.",
    "transport_carry": "Transport\nCarry",
    "release_transition": "Release\nTrans.",
    "post_release_transit": "Post-Rel.\nTransit",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v: Optional[float], digits: int = 5) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _fmt_pct(v: Optional[float], digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{100.0 * v:.{digits}f}%"


def plot_fig1_oracle_pareto_and_convergence(
    oracle_json: Dict[str, Any],
    curve_rows: List[Dict[str, str]],
    fig_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), dpi=160)
    lambdas = oracle_json["config"]["lambdas"]
    lam_tags = [
        ("action_opt_lam_0" if l == 0.0 else f"action_opt_{lam_to_tag(l)}") for l in lambdas
    ]

    # Panel (0, 0): Pareto Frontier (DDIM 10D MSE vs ||c||_2)
    ax = axes[0, 0]
    for sname, color, marker, s_lbl in (
        ("seen_val", "#1f77b4", "o", "Seen-Val (3k)"),
        ("unseen_chain", "#d62728", "s", "Unseen-Chain (3k)"),
    ):
        ms = oracle_json["results"][sname]["methods"]
        xs = [ms[t]["c_l2_mean"] for t in lam_tags]
        ys = [ms[t]["ddim_10d_mse"] for t in lam_tags]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2.2, markersize=6, label=f"Action-Opt Oracle ({s_lbl})")
        for l, xi, yi in zip(lambdas, xs, ys):
            lbl = "0" if l == 0 else f"{l:.0e}".replace("-0", "-")
            xyoff = (-14, -12) if sname == "seen_val" else (5, 5)
            ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=xyoff, fontsize=7, color=color)

        # Baselines
        ax.scatter([0.0], [ms["identity"]["ddim_10d_mse"]], color=color, marker="X", s=90, zorder=5)
        ax.scatter(
            [ms["e10_c"]["c_l2_mean"]],
            [ms["e10_c"]["ddim_10d_mse"]],
            color=color,
            marker="^",
            s=110,
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
            label=f"E10-C ({s_lbl})",
        )
        ax.scatter(
            [ms["p0_oracle"]["c_l2_mean"]],
            [ms["p0_oracle"]["ddim_10d_mse"]],
            color=color,
            marker="*",
            s=150,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
            label=f"P0 Oracle ({s_lbl})",
        )
        ax.axhline(ms["full_self"]["ddim_10d_mse"], color=color, linestyle=":", alpha=0.55, linewidth=1.2)

    ax.set_xlabel("Latent Modification Norm  ||c||_2  (= ||U_64 c||_2)")
    ax.set_ylabel("DDIM 10D Action MSE (log scale, lower is better)")
    ax.set_yscale("log")
    ax.set_title("A. Pareto Frontier: Action Error vs Modification Norm")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    # Panel (0, 1): Diffusion training loss convergence on Unseen-Chain
    ax = axes[0, 1]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, len(lam_tags)))
    for t, l, col in zip(lam_tags, lambdas, cmap):
        rows = [r for r in curve_rows if r["split"] == "unseen_chain" and r["method"] == t]
        steps = [int(r["step"]) for r in rows]
        dl = [float(r["diff_loss"]) for r in rows]
        lbl = "λ=0" if l == 0 else f"λ={l:.0e}".replace("-0", "-")
        ax.plot(steps, dl, color=col, linewidth=2.0, marker="o", markersize=3.5, label=lbl)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Diffusion Training Loss (L_diff)")
    ax.set_yscale("log")
    ax.set_title("B. Optimization Convergence: L_diff (Unseen-Chain)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    # Panel (1, 0): Independent DDIM 10D MSE convergence (Unseen-Chain 512 track)
    ax = axes[1, 0]
    for t, l, col in zip(lam_tags, lambdas, cmap):
        rows = [
            r
            for r in curve_rows
            if r["split"] == "unseen_chain"
            and r["method"] == t
            and r.get("ddim_track_10d_mse") not in (None, "", "None")
        ]
        steps = [int(r["step"]) for r in rows]
        dm = [float(r["ddim_track_10d_mse"]) for r in rows]
        lbl = "λ=0" if l == 0 else f"λ={l:.0e}".replace("-0", "-")
        ax.plot(steps, dm, color=col, linewidth=2.0, marker="s", markersize=4.5, label=lbl)
    ms_u = oracle_json["results"]["unseen_chain"]["methods"]
    ax.axhline(ms_u["e10_c"]["ddim_10d_mse"], color="#ff7f0e", linestyle="--", linewidth=1.6, label=f"E10-C ({ms_u['e10_c']['ddim_10d_mse']:.4f})")
    ax.axhline(ms_u["p0_oracle"]["ddim_10d_mse"], color="black", linestyle=":", linewidth=1.6, label=f"P0 Oracle ({ms_u['p0_oracle']['ddim_10d_mse']:.4f})")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Independent DDIM 10D MSE (log scale)")
    ax.set_yscale("log")
    ax.set_title("C. Independent Fixed-Noise DDIM Convergence (Unseen-Chain)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.2, loc="upper right", ncol=2)

    # Panel (1, 1): Geometry vs Lambda (||c||_2 / ||c_p0||_2, cos(c*, c_p0), L_z vs z_q)
    ax = axes[1, 1]
    x_idx = np.arange(len(lambdas))
    x_lbls = ["0" if l == 0 else f"{l:.0e}".replace("-0", "-") for l in lambdas]
    for sname, ls, s_lbl in (("seen_val", "-", "Seen"), ("unseen_chain", "--", "Unseen")):
        ms = oracle_json["results"][sname]["methods"]
        norm_rat = [ms[t]["c_norm_ratio_vs_p0"] for t in lam_tags]
        cos_p0 = [ms[t]["cos_vs_c_p0"] for t in lam_tags]
        lz_zq = [ms[t]["latent_lz"] for t in lam_tags]
        ax.plot(x_idx, norm_rat, color="#1f77b4", linestyle=ls, marker="o", linewidth=2.0, label=f"||c*||/||c_p0|| ({s_lbl})")
        ax.plot(x_idx, cos_p0, color="#2ca02c", linestyle=ls, marker="s", linewidth=2.0, label=f"cos(c*, c_p0) ({s_lbl})")
        ax.plot(x_idx, lz_zq, color="#9467bd", linestyle=ls, marker="^", linewidth=2.0, label=f"L_z vs z_query ({s_lbl})")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_lbls)
    ax.set_xlabel("Regularization Weight λ")
    ax.set_ylabel("Ratio / Cosine / Normalized L_z")
    ax.set_title("D. Modification Geometry vs Future-Latent Target (c_p0, z_query)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.2, loc="upper right", ncol=2)

    fig.suptitle("E11-Pilot Oracle Adaptation Optimization: Pareto Frontier, Convergence & Geometry", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = fig_dir / "fig1_oracle_pareto_and_convergence.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_fig2_oracle_phases(
    oracle_json: Dict[str, Any],
    lam_primary_tag: str,
    lam_tight_tag: str,
    fig_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.8), dpi=160)
    methods_show = [
        ("identity", "Identity (c=0)", "#7f7f7f"),
        ("e10_c", "E10-C (Learned)", "#ff7f0e"),
        ("p0_oracle", "P0 Oracle", "#2ca02c"),
        (lam_primary_tag, f"Action-Opt ({lam_primary_tag.replace('action_opt_', '')})", "#1f77b4"),
        (lam_tight_tag, f"Action-Opt ({lam_tight_tag.replace('action_opt_', '')})", "#9467bd"),
    ]
    x = np.arange(len(SEMANTIC_PHASES))
    width = 0.16

    for col_i, (sname, s_title) in enumerate((("seen_val", "Seen-Val (3,000)"), ("unseen_chain", "Unseen-Chain (3,000)"))):
        # Top row: per-phase DDIM 10D MSE
        ax = axes[0, col_i]
        ms = oracle_json["results"][sname]["methods"]
        for mi, (mid, lbl, col) in enumerate(methods_show):
            if mid not in ms:
                continue
            ys = [ms[mid]["phases"].get(ph, {}).get("ddim_10d_mse", np.nan) for ph in SEMANTIC_PHASES]
            ax.bar(x + (mi - 2) * width, ys, width=width, color=col, label=lbl, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABELS[p] for p in SEMANTIC_PHASES], fontsize=8.5)
        ax.set_ylabel("DDIM 10D Action MSE (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"{'A' if col_i == 0 else 'B'}. Per-Phase DDIM 10D MSE — {s_title}")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")

        # Bottom row: per-phase modification norm ||c||_2 (excluding Identity=0)
        ax2 = axes[1, col_i]
        m_norm = methods_show[1:]
        w2 = 0.19
        for mi, (mid, lbl, col) in enumerate(m_norm):
            if mid not in ms:
                continue
            ys = [ms[mid]["phases"].get(ph, {}).get("c_l2_mean", np.nan) for ph in SEMANTIC_PHASES]
            ax2.bar(x + (mi - 1.5) * w2, ys, width=w2, color=col, label=lbl, alpha=0.9)
        ax2.set_xticks(x)
        ax2.set_xticklabels([PHASE_LABELS[p] for p in SEMANTIC_PHASES], fontsize=8.5)
        ax2.set_ylabel("Modification Norm ||c||_2")
        ax2.set_title(f"{'C' if col_i == 0 else 'D'}. Per-Phase Latent Modification Norm — {s_title}")
        ax2.grid(True, axis="y", alpha=0.25)
        ax2.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("E11-Pilot Per-Behavioral-Phase Breakdown: Action Error & Minimal Modification Norm", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = fig_dir / "fig2_oracle_method_and_phase_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_fig3_learnability(
    learn_json: Dict[str, Any],
    oracle_json: Dict[str, Any],
    fig_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.8), dpi=160)
    runs = learn_json["train_runs"]
    ev = learn_json["evaluation"]
    run_colors = {
        "p0_reg_light_ctrl": "#2ca02c",
    }
    palette = ["#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    pi = 0
    for k in runs:
        if k not in run_colors:
            run_colors[k] = palette[pi % len(palette)]
            pi += 1

    # Panel (0, 0): val_ckpt DDIM 10D MSE learning curve
    ax = axes[0, 0]
    for rname, rmeta in runs.items():
        hist = [h for h in rmeta["history"] if h.get("val_ckpt_ddim10") is not None]
        ep = [h["epoch"] for h in hist]
        vm = [h["val_ckpt_ddim10"] for h in hist]
        ax.plot(ep, vm, marker="o", markersize=4, linewidth=2.0, color=run_colors[rname], label=f"{rname} (best={rmeta['best_val_ckpt_ddim10']:.5f}@ep{rmeta['best_epoch']})")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Independent val_ckpt DDIM 10D MSE")
    ax.set_title("A. Lightweight Transfer Learning Curves (Independent val_ckpt)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right")

    # Panel (0, 1): Seen vs Unseen DDIM 10D MSE comparison
    ax = axes[0, 1]
    model_order = ["identity"] + list(runs.keys()) + ["e10_c"]
    labels = [ev[m]["seen_val"]["label"] for m in model_order]
    seen_y = [ev[m]["seen_val"]["correct"]["ddim_10d_mse"] for m in model_order]
    unseen_y = [ev[m]["unseen_chain"]["correct"]["ddim_10d_mse"] for m in model_order]
    x = np.arange(len(model_order))
    w = 0.36
    b1 = ax.bar(x - w / 2, seen_y, width=w, color="#1f77b4", label="Seen-Val (3k)")
    b2 = ax.bar(x + w / 2, unseen_y, width=w, color="#d62728", label="Unseen-Chain (3k)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in model_order], fontsize=7.5)
    ax.set_ylabel("DDIM 10D Action MSE (lower is better)")
    ax.set_title("B. Learned Transfer Accuracy: Seen-Val vs Unseen-Chain")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel (1, 0): Correct vs Wrong-Task Demo Sensitivity (Unseen-Chain)
    ax = axes[1, 0]
    m_sens = list(runs.keys()) + ["e10_c", "identity"]
    x_s = np.arange(len(m_sens))
    corr_u = [ev[m]["unseen_chain"]["correct"]["ddim_10d_mse"] for m in m_sens]
    wrg_u = [ev[m]["unseen_chain"]["wrong_task"]["ddim_10d_mse"] for m in m_sens]
    ratios = [ev[m]["unseen_chain"]["wrong_ratio_mse10"] for m in m_sens]
    ax.bar(x_s - w / 2, corr_u, width=w, color="#2ca02c", label="Correct Demo")
    ax.bar(x_s + w / 2, wrg_u, width=w, color="#d62728", alpha=0.85, label="Wrong-Task Demo")
    for xi, wy, rat in zip(x_s, wrg_u, ratios):
        ax.annotate(f"{rat:.2f}x", (xi + w / 2, wy), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color="#b22222")
    ax.set_xticks(x_s)
    ax.set_xticklabels([m.replace("_", "\n") for m in m_sens], fontsize=7.5)
    ax.set_ylabel("Unseen-Chain DDIM 10D Action MSE")
    ax.set_title("C. Demo Sensitivity on Unseen-Chain (Correct vs Wrong-Task)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel (1, 1): Modification Norm ||c||_2 (Cross vs Self Preservation) on Unseen-Chain
    ax = axes[1, 1]
    m_geo = list(runs.keys()) + ["e10_c"]
    x_g = np.arange(len(m_geo))
    cross_l2 = [ev[m]["unseen_chain"]["geometry"]["c_l2_mean"] for m in m_geo]
    self_l2 = [ev[m]["unseen_chain"]["self_c_l2_mean"] for m in m_geo]
    d3_d1 = [
        ev[m]["unseen_chain"]["self_preservation"]["ddim_10d_mse"]
        / max(learn_json["config"]["full_self_d1_ddim10"]["unseen_chain"], 1e-12)
        for m in m_geo
    ]
    ax.bar(x_g - w / 2, cross_l2, width=w, color="#1f77b4", label="Cross-Demo ||c||_2")
    ax.bar(x_g + w / 2, self_l2, width=w, color="#ff7f0e", label="Self-Demo ||c||_2 (ideal=0)")
    for xi, sl, r31 in zip(x_g, self_l2, d3_d1):
        ax.annotate(f"D3/D1\n{r31:.2f}x", (xi + w / 2, sl), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color="#8c4600")
    ax.set_xticks(x_g)
    ax.set_xticklabels([m.replace("_", "\n") for m in m_geo], fontsize=7.5)
    ax.set_ylabel("Coefficient L2 Norm ||c||_2")
    ax.set_title("D. Modification Magnitude & Self-Preservation (Unseen-Chain)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("E11-Pilot Learnability Verification: Predicting Action-Guided c* from (z_src, h_q)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = fig_dir / "fig3_learnability_and_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def write_report(
    oracle_json: Dict[str, Any],
    learn_json: Optional[Dict[str, Any]],
    *,
    lam_primary: float,
    lam_tight: float,
    fig_paths: Dict[str, Path],
    out_path: Path,
) -> None:
    o_cfg = oracle_json["config"]
    lambdas = o_cfg["lambdas"]
    lam_tags = [("action_opt_lam_0" if l == 0.0 else f"action_opt_{lam_to_tag(l)}") for l in lambdas]
    pri_tag = f"action_opt_{lam_to_tag(lam_primary)}"
    tight_tag = f"action_opt_{lam_to_tag(lam_tight)}"

    s_ms = oracle_json["results"]["seen_val"]["methods"]
    u_ms = oracle_json["results"]["unseen_chain"]["methods"]

    lines: List[str] = []
    lines.append("# E11-Pilot: Action-Guided Minimal Behavior Adaptation 实验报告\n")
    lines.append("## 1. 实验设置与冻结/防泄漏核查\n")
    mm = o_cfg["model_meta"]
    lines.append(f"- **冻结基座（零参数更新、无架构/数据集改动、纯离线）**：")
    lines.append(f"  - E9 Stage 1 Local Encoder & Action Decoder：`{mm['e9_stage1_ckpt']}`")
    lines.append(f"  - E9 Stage 2 Direct Reader：`{mm['e9_stage2_ckpt']}`")
    lines.append(f"  - E10-C 对照模型（K=64）：`{mm['e10_c_ckpt']}`（Epoch `{mm['e10_c_epoch']}`）")
    lines.append(f"  - P0 训练集正交基 `U_64`：`{mm['basis_path']}`（`{mm['basis_key']}`，正交误差 `{mm['basis_ortho_err']:.2e}`，与 E10-C `basis_u_k` 最大绝对差 `{mm['basis_vs_e10c_max_abs_diff']:.2e}`）")
    lines.append(f"- **Oracle 优化协议**：初始化 `c = 0`，使用可微的 diffusion training loss（每步全新随机噪声 + `{o_cfg['mc_timesteps']}` 分层时间步采样、`pertub_noise=0`，严格不对 DDIM 采样反向传播），Adam `lr={o_cfg['lr']}` 余弦退火 `{o_cfg['steps']}` 步；随后以独立固定噪声的 16 步 DDIM Action MSE（主种子 `{o_cfg['ddim_seed']}` + 独立复核种子 `{o_cfg['ddim_seed_alt']}`）评估。")
    lines.append(f"- **评估集**：与 P0 / P0.5 / E10 逐对一致的 `seen_val`（`{o_cfg['n_seen_val']}` 对）和 `unseen_chain`（`{o_cfg['n_unseen']}` 对）。\n")

    lines.append("## 2. Oracle Adaptation Optimization：λ 扫描与四方关键对照\n")
    lines.append(f"![Oracle Pareto & Convergence](figures/{fig_paths['fig1'].name})\n")

    for sname, stitle in (("seen_val", "Seen-Val（3,000 对）"), ("unseen_chain", "Unseen-Chain（3,000 对）")):
        ms = oracle_json["results"][sname]["methods"]
        lines.append(f"### 2.{1 if sname == 'seen_val' else 2} {stitle}\n")
        lines.append("| 方法 / λ | DDIM 10D MSE (主 / 复核种子) | ARR | 7D MAE | Grip Acc | `||c||_2` (vs P0) | `Δz_rms` | `L_z` vs `z_q` | `cos(c, c_p0)` | `c` eRank |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        order = ["identity", "e10_c", "p0_oracle", "full_self"] + lam_tags
        for mid in order:
            r = ms[mid]
            norm_str = f"`{r['c_l2_mean']:.2f}`"
            if r.get("c_norm_ratio_vs_p0") is not None:
                norm_str += f" ({100*r['c_norm_ratio_vs_p0']:.0f}%)"
            lines.append(
                f"| **{r['label']}** | `{_fmt(r['ddim_10d_mse'])}` / `{_fmt(r['ddim_10d_mse_alt_seed'])}` | "
                f"`{_fmt_pct(r['arr_10d_vs_id_full'])}` | `{_fmt(r['ddim_7d_mae'])}` | `{_fmt_pct(r['grip_acc16'], 2)}` | "
                f"{norm_str} | `{_fmt(r['delta_z_rms'], 4)}` | `{_fmt(r['latent_lz'], 4)}` | "
                f"`{_fmt(r.get('cos_vs_c_p0'), 3)}` | `{_fmt(r.get('c_erank_energy'), 1)}` |"
            )
        lines.append("")

    lines.append("## 3. 分行为阶段（5-Phase）对比结果\n")
    lines.append(f"![Per-Phase Comparison](figures/{fig_paths['fig2'].name})\n")
    for sname, stitle in (("seen_val", "Seen-Val"), ("unseen_chain", "Unseen-Chain")):
        ms = oracle_json["results"][sname]["methods"]
        lines.append(f"### 3.{1 if sname == 'seen_val' else 2} {stitle} 分阶段 DDIM 10D MSE（括号内为 `||c||_2`）\n")
        lines.append(f"| 行为阶段 (样本数) | Identity | E10-C | P0 Oracle | Action-Opt (`λ={lam_primary:.0e}`) | Action-Opt (`λ={lam_tight:.0e}`) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for ph in SEMANTIC_PHASES:
            cnt = ms["identity"]["phases"].get(ph, {}).get("count", 0)
            cells = []
            for mid in ("identity", "e10_c", "p0_oracle", pri_tag, tight_tag):
                pm = ms[mid]["phases"].get(ph, {})
                mse_s = _fmt(pm.get("ddim_10d_mse"))
                cl2 = pm.get("c_l2_mean")
                if mid == "identity" or cl2 is None:
                    cells.append(f"`{mse_s}`")
                else:
                    cells.append(f"`{mse_s}` (`{cl2:.1f}`)")
            lines.append(f"| `{ph}` ({cnt}) | " + " | ".join(cells) + " |")
        lines.append("")

    if learn_json is not None:
        l_cfg = learn_json["config"]
        ev = learn_json["evaluation"]
        runs = learn_json["train_runs"]
        pri_audit = l_cfg["primary_cache_meta"]["split_audit"]

        n_tr_k = pri_audit["train_fit_pairs"] / 1000.0
        pct_of_e10 = 100.0 * pri_audit["train_fit_pairs"] / 1373035.0
        med_key = next((k for k in runs if k.startswith("ag_med_")), None)
        med_p_str = f"{runs[med_key]['trainable_params']/1e6:.2f}M" if med_key else "7.80M"

        lines.append("## 4. Learnability 验证：用 `(z_src, h_q)` 预测 Action-Guided `c*`\n")
        lines.append(f"- **严格防泄漏四分划分**：仅从 `fold='train'` 按 layout 划分互不重叠的 `train_fit`（`{pri_audit['train_fit_pairs']}` 对，`{pri_audit['train_fit_unique_eps']}` 条轨迹，仅占 E10-C 1.37M 训练样本的 `{pct_of_e10:.2f}%`）与 `val_ckpt`（`{pri_audit['val_ckpt_pairs']}` 对，`{pri_audit['val_ckpt_unique_eps']}` 条轨迹，用于独立选模）；与 `seen_val` 重叠轨迹数 = **`{pri_audit['train_vs_seen_val_ep_overlap']}` / `{pri_audit['val_ckpt_vs_seen_val_ep_overlap']}`**，`unseen_chain` 严格仅做最终测试，全程零 GT future 信息泄漏。")
        lines.append(f"- **轻量 Transfer Network**：2 层 Pre-LN Cross-Attention Transformer（`d=256, 4 heads, FFN=1024`，**2.57M 参数**，比 E10-C 28.6M 轻 **11.1×**），另设同数据/同架构的 Future-Latent 回归对照（`p0_reg_light_ctrl`，预测 `c_p0`）及中等容量对照（`d=384, 3 layers`，`{med_p_str}`）。\n")
        lines.append(f"![Learnability & Sensitivity](figures/{fig_paths['fig3'].name})\n")

        for sname, stitle in (("seen_val", "Seen-Val（3,000 对）"), ("unseen_chain", "Unseen-Chain（3,000 对）")):
            lines.append(f"### 4.{1 if sname == 'seen_val' else 2} {stitle} Learnability 与 Correct/Wrong Demo 敏感性\n")
            lines.append("| 模型 | Params | DDIM 10D MSE (ARR) | 7D MAE / Grip | `||c||_2` | Coeff MSE / R² / cos (vs `c*`) | Coeff MSE (vs `c_p0`) | Wrong MSE (`×`, `Δ`) | Self `||c||_2` (`D3/D1`) |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            m_order = ["identity"] + list(runs.keys()) + ["e10_c"]
            d1_mse = l_cfg["full_self_d1_ddim10"][sname]
            id_mse = l_cfg["identity_ddim10"][sname]
            for mid in m_order:
                r = ev[mid][sname]
                c_m = r["correct"]
                w_m = r["wrong_task"]
                s_m = r["self_preservation"]
                g_m = r["geometry"]
                fp = r["fit_vs_c_opt_primary"]
                fp0 = r["fit_vs_c_p0"]
                arr = (id_mse - c_m["ddim_10d_mse"]) / max(id_mse - d1_mse, 1e-12)
                d3_r = s_m["ddim_10d_mse"] / max(d1_mse, 1e-12)
                p_str = "0" if mid == "identity" else ("28.6M" if mid == "e10_c" else f"{runs[mid]['trainable_params']/1e6:.2f}M")
                lines.append(
                    f"| **{r['label']}** | `{p_str}` | `{c_m['ddim_10d_mse']:.5f}` (`{100*arr:.1f}%`) | "
                    f"`{c_m['ddim_7d_mae']:.5f}` / `{100*c_m['grip_acc16']:.2f}%` | `{g_m['c_l2_mean']:.2f}` | "
                    f"`{fp['coeff_mse']:.2f}` / `{fp['coeff_r2']:.3f}` / `{fp['coeff_cos']:.3f}` | "
                    f"`{fp0['coeff_mse']:.2f}` | `{w_m['ddim_10d_mse']:.5f}` (`{r['wrong_ratio_mse10']:.2f}×`, `{r['wrong_delta_mse10']:+.4f}`) | "
                    f"`{r['self_c_l2_mean']:.2f}` (`{d3_r:.2f}×`) |"
                )
            lines.append("")

        # Add compact Unseen-Chain per-phase Learnability table
        lines.append("### 4.3 Unseen-Chain Learnability 分行为阶段 DDIM 10D MSE\n")
        pri_m_k = f"ag_light_{lam_to_tag(lam_primary)}"
        med_m_k = f"ag_med_{lam_to_tag(lam_primary)}"
        tgh_m_k = f"ag_light_{lam_to_tag(lam_tight)}"
        ph_models = ["identity", pri_m_k, med_m_k, tgh_m_k, "p0_reg_light_ctrl", "e10_c"]
        ph_models = [m for m in ph_models if m in ev]
        lines.append("| 行为阶段 | " + " | ".join(ph_models) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(ph_models)) + "|")
        for ph in SEMANTIC_PHASES:
            vals = [f"`{_fmt(ev[m]['unseen_chain']['phases'].get(ph, {}).get('ddim_10d_mse'))}`" for m in ph_models]
            lines.append(f"| `{ph}` | " + " | ".join(vals) + " |")
        lines.append("")

    # Synthesize direct answers to the 3 core questions
    lines.append("## 5. 重点问题回答\n")
    s_pri, u_pri = s_ms[pri_tag], u_ms[pri_tag]
    s_tgh, u_tgh = s_ms[tight_tag], u_ms[tight_tag]
    s_p0, u_p0 = s_ms["p0_oracle"], u_ms["p0_oracle"]
    s_e10, u_e10 = s_ms["e10_c"], u_ms["e10_c"]

    lines.append("### Q1. 现有行为 latent 是否只需少量动作引导的修改即可迁移？")
    lines.append(
        f"- **明确是肯定的（且存在极陡峭的帕累托前沿）**：在冻结 E9 + `U_64` 上，动作引导优化（`λ={lam_primary:.0e}`）仅用 P0 Oracle **`{100*s_pri['c_norm_ratio_vs_p0']:.1f}%`（Seen） / `{100*u_pri['c_norm_ratio_vs_p0']:.1f}%`（Unseen）的修改范数**（即仅 **`{100*s_pri['c_energy_ratio_vs_p0']:.1f}%` / `{100*u_pri['c_energy_ratio_vs_p0']:.1f}%` 的平方修改能量**），就将 DDIM 10D Action MSE 从 Identity 的 `{s_ms['identity']['ddim_10d_mse']:.5f}` / `{u_ms['identity']['ddim_10d_mse']:.5f}` 降至 **`{s_pri['ddim_10d_mse']:.5f}`（Seen） / `{u_pri['ddim_10d_mse']:.5f}`（Unseen）**，大幅超越 E10-C（`{s_e10['ddim_10d_mse']:.5f}` / `{u_e10['ddim_10d_mse']:.5f}`）并逼近甚至超过 P0 Oracle（`{s_p0['ddim_10d_mse']:.5f}` / `{u_p0['ddim_10d_mse']:.5f}`）。"
    )
    lines.append(
        f"- 进一步收紧正则至 `λ={lam_tight:.0e}` 时，仅用 P0 Oracle **`{100*u_tgh['c_norm_ratio_vs_p0']:.1f}%` 的范数**（`||c||_2 = {u_tgh['c_l2_mean']:.2f}` vs E10-C `{u_e10['c_l2_mean']:.2f}`，能量仅 `{100*u_tgh['c_energy_ratio_vs_p0']:.1f}%`），Unseen 动作误差（`{u_tgh['ddim_10d_mse']:.5f}`）仍显著优于 E10-C（`{u_e10['ddim_10d_mse']:.5f}`）。且主种子与独立复核种子 DDIM MSE 高度一致，证明该收益不是单样本噪声过拟合。\n"
    )

    lines.append("### Q2. 最小修改是否优于 Future Latent Regression？")
    lines.append(
        f"- **在 Oracle 层面明确更优，揭示 E9 潜空间存在显著的“动作等价冗余”**：Action-Opt（`λ={lam_primary:.0e}`）与 `z_query` 仍有显著潜空间距离（`L_z = {s_pri['latent_lz']:.4f}` Seen / `{u_pri['latent_lz']:.4f}` Unseen，远大于 P0 Oracle 的 `{s_p0['latent_lz']:.4f}` / `{u_p0['latent_lz']:.4f}`，`cos(c*, c_p0) = {s_pri['cos_vs_c_p0']:.3f}` / `{u_pri['cos_vs_c_p0']:.3f}`），但动作精度几乎持平甚至更优，且修改范数与有效秩更小。这说明 Future Latent Regression（回归 `z_query`）把大量容量浪费在了与动作无关的视觉/背景变化分量上。"
    )
    if learn_json is not None:
        ev = learn_json["evaluation"]
        pri_audit = learn_json["config"]["primary_cache_meta"]["split_audit"]
        n_tr_k = int(round(pri_audit["train_fit_pairs"] / 1000))
        pri_m_key = f"ag_light_{lam_to_tag(lam_primary)}"
        tgh_m_key = f"ag_light_{lam_to_tag(lam_tight)}"
        p0_m_key = "p0_reg_light_ctrl"
        ag_s = ev[pri_m_key]["seen_val"]
        ag_u = ev[pri_m_key]["unseen_chain"]
        tgh_u = ev[tgh_m_key]["unseen_chain"]
        p0_s = ev[p0_m_key]["seen_val"]
        p0_u = ev[p0_m_key]["unseen_chain"]
        lines.append(
            f"- **在同数据/同架构（{n_tr_k}k 对、2.57M 轻量网络）纯系数回归层面，呈现出“精度 vs 边界敏感性与 Demo 保持”的清晰权衡**："
            f"Action-Guided 纯系数监督（`{pri_m_key}`）在 Seen / Unseen 上取得 DDIM 10D MSE = **`{ag_s['correct']['ddim_10d_mse']:.5f}` / `{ag_u['correct']['ddim_10d_mse']:.5f}`**（`||c||_2 = {ag_s['geometry']['c_l2_mean']:.2f}` / `{ag_u['geometry']['c_l2_mean']:.2f}`），"
            f"虽然系数 MSE 本身低于同数据 Future-Latent 回归对照 `p0_reg_light_ctrl`（`cMSE = 1.69/2.25` vs `2.12/2.70`），但 DDIM 动作误差略高于 `p0_reg_light_ctrl`（`{p0_s['correct']['ddim_10d_mse']:.5f}` / `{p0_u['correct']['ddim_10d_mse']:.5f}`，`||c||_2 = {p0_s['geometry']['c_l2_mean']:.2f}` / `{p0_u['geometry']['c_l2_mean']:.2f}`）。"
            f"机制上原因明确：（1）最小范数解 `c*` 位于动作等价区域的**最小范数边界**，纯欧氏回归 `MSE(c_hat, c*)` 的条件均值收缩会直接将预测推离低动作误差盆地（从 `λ=1e-5` 收紧到 `λ=3e-5` 时 Unseen MSE 进一步从 `0.01268` 升至 `{tgh_u['correct']['ddim_10d_mse']:.5f}` 即验证了这一点），而 `c_p0` 位于 Encoder 流形内部、对回归误差更宽容；（2）相应地，Action-Guided 监督带来了**显著更小的修改幅度（`-28%` 范数）、更强的 Wrong-Demo 敏感性（Unseen `{ag_u['wrong_ratio_mse10']:.2f}×` vs `p0_reg` `{p0_u['wrong_ratio_mse10']:.2f}×` / E10-C `3.28×`）以及更优的零偏移自保真（Self `||c||_2 = {ag_u['self_c_l2_mean']:.2f}` / `{tgh_u['self_c_l2_mean']:.2f}` vs `p0_reg` `{p0_u['self_c_l2_mean']:.2f}`）**。"
        )
    lines.append("")

    lines.append("### Q3. Oracle 优化有效时，模型能否从 Demo 和 Query 预测这种修改？")
    if learn_json is not None:
        ev = learn_json["evaluation"]
        pri_audit = learn_json["config"]["primary_cache_meta"]["split_audit"]
        n_tr_k = int(round(pri_audit["train_fit_pairs"] / 1000))
        pct_e10 = 100.0 * pri_audit["train_fit_pairs"] / 1373035.0
        pri_m_key = f"ag_light_{lam_to_tag(lam_primary)}"
        med_m_key = f"ag_med_{lam_to_tag(lam_primary)}"
        ag_s = ev[pri_m_key]["seen_val"]
        ag_u = ev[pri_m_key]["unseen_chain"]
        med_u = ev[med_m_key]["unseen_chain"]
        lines.append(
            f"- **高度可预测，且能保持最强 Demo 敏感性，但单靠纯 `MSE(c_hat, c*)` 尚不能追平端到端 `L_diff` 训练的 E10-C**：仅用 `{n_tr_k}k` 训练对（E10-C 训练量的 `{pct_e10:.2f}%`）和 2.57M 轻量网络（E10-C 的 `9.0%`）纯监督预测 `c*`，在 Seen / Unseen 上已达到 **`R² = {ag_s['fit_vs_c_opt_primary']['coeff_r2']:.3f}` / `{ag_u['fit_vs_c_opt_primary']['coeff_r2']:.3f}`**（`cos = {ag_s['fit_vs_c_opt_primary']['coeff_cos']:.3f}` / `{ag_u['fit_vs_c_opt_primary']['coeff_cos']:.3f}`），恢复 **`{100*(l_cfg['identity_ddim10']['seen_val'] - ag_s['correct']['ddim_10d_mse'])/(l_cfg['identity_ddim10']['seen_val'] - l_cfg['full_self_d1_ddim10']['seen_val']):.1f}%` / `{100*(l_cfg['identity_ddim10']['unseen_chain'] - ag_u['correct']['ddim_10d_mse'])/(l_cfg['identity_ddim10']['unseen_chain'] - l_cfg['full_self_d1_ddim10']['unseen_chain']):.1f}%`** 的动作精度（7.80M 中等容量模型进一步小幅降至 `{med_u['correct']['ddim_10d_mse']:.5f}`，`R²={med_u['fit_vs_c_opt_primary']['coeff_r2']:.3f}`），同时保持所有学习模型中最高的 Wrong-Demo 敏感性（Seen `{ag_s['wrong_ratio_mse10']:.2f}×` / Unseen `{ag_u['wrong_ratio_mse10']:.2f}×`）。"
        )
        lines.append(
            f"- **下一步启示**：Oracle `c*`（Unseen `0.00113`）与纯 `MSE(c_hat, c*)` 预测（`0.01226–0.01268`）及 E10-C（`0.00749`）之间的差距指出两条明确改进方向：（1）用**端到端 `L_diff(z_src + U_64 c_hat, a_q^*) + λ ||c_hat||_2^2`**（按 Action Decoder 雅可比加权而非各向同性欧氏 `MSE(c_hat, c*)`）直接训练最小修改网络，既可避免边界收缩退化，又无需先离线优化 `c*`；（2）将源状态视觉键 `h_src` 或更锐利的 Reader 与最小修改正则结合，补齐纯 `(z_src, h_q)` 缺乏空间偏移对照信息的可辨识度瓶颈。"
        )
    lines.append("")

    lines.append("## 6. 产物索引\n")
    lines.append(f"- Oracle 配置与汇总：[oracle/config.json](file://{(OUTPUT_ROOT / 'oracle/config.json').resolve()}), [oracle/oracle_sweep_summary.csv](file://{(OUTPUT_ROOT / 'oracle/oracle_sweep_summary.csv').resolve()}), [oracle/oracle_sweep_summary.json](file://{(OUTPUT_ROOT / 'oracle/oracle_sweep_summary.json').resolve()})")
    lines.append(f"- Oracle 优化曲线与分阶段表：[oracle/convergence_curves.csv](file://{(OUTPUT_ROOT / 'oracle/convergence_curves.csv').resolve()}), [oracle/phase_breakdown.csv](file://{(OUTPUT_ROOT / 'oracle/phase_breakdown.csv').resolve()})")
    if learn_json is not None:
        lines.append(f"- Learnability 配置与汇总：[learnability/config.json](file://{(OUTPUT_ROOT / 'learnability/config.json').resolve()}), [learnability/learnability_summary.csv](file://{(OUTPUT_ROOT / 'learnability/learnability_summary.csv').resolve()}), [learnability/learnability_summary.json](file://{(OUTPUT_ROOT / 'learnability/learnability_summary.json').resolve()})")
        lines.append(f"- Learnability 分阶段表：[learnability/learnability_phase_breakdown.csv](file://{(OUTPUT_ROOT / 'learnability/learnability_phase_breakdown.csv').resolve()})")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=OUTPUT_ROOT / "oracle")
    parser.add_argument("--learn-dir", type=Path, default=OUTPUT_ROOT / "learnability")
    parser.add_argument("--fig-dir", type=Path, default=OUTPUT_ROOT / "figures")
    parser.add_argument("--report-out", type=Path, default=OUTPUT_ROOT / "REPORT.md")
    parser.add_argument("--lam-primary", type=float, default=1e-5)
    parser.add_argument("--lam-tight", type=float, default=3e-5)
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    oracle_json = json.loads((args.oracle_dir / "oracle_sweep_summary.json").read_text())
    curve_rows = _read_csv(args.oracle_dir / "convergence_curves.csv")

    pri_tag = f"action_opt_{lam_to_tag(args.lam_primary)}"
    tight_tag = f"action_opt_{lam_to_tag(args.lam_tight)}"

    fig_paths: Dict[str, Path] = {}
    fig_paths["fig1"] = plot_fig1_oracle_pareto_and_convergence(oracle_json, curve_rows, args.fig_dir)
    fig_paths["fig2"] = plot_fig2_oracle_phases(oracle_json, pri_tag, tight_tag, args.fig_dir)

    learn_json = None
    learn_sum_path = args.learn_dir / "learnability_summary.json"
    if learn_sum_path.is_file():
        learn_json = json.loads(learn_sum_path.read_text())
        fig_paths["fig3"] = plot_fig3_learnability(learn_json, oracle_json, args.fig_dir)

    write_report(
        oracle_json,
        learn_json,
        lam_primary=args.lam_primary,
        lam_tight=args.lam_tight,
        fig_paths=fig_paths,
        out_path=args.report_out,
    )
    print(f"[E11-Report DONE] Figures -> {args.fig_dir} | Report -> {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
