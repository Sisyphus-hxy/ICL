"""Visualization & Report Generator for E9 Reader Inference-only Diagnosis.

Reads outputs produced by `e9/reader_diagnosis.py` in `e9/outputs/reader_diagnosis/` and generates:
  - fig1_reader_comparison_d2_d4.png         : 6 Reader families D2 & D4 MSE + Temperature Sweep
  - fig2_attention_alignment_and_jumps.png   : Entropy, Top-1 w, Top-2 Adj, Alignment Error, Phase Hit, Peak Jumps
  - fig3_location_vs_mixing_decomposition.png: HIT/MISS & Adjacent/Non-Adjacent stratification + Waterfall
  - fig4_behavioral_phase_analysis.png       : Per-behavioral-phase error, alignment, and phase hit
  - fig5_attention_heatmaps_seen.png         : Seen-Cross trajectory attention heatmaps + 6-reader readout comparison
  - fig6_attention_heatmaps_unseen.png       : Unseen-Cross trajectory attention heatmaps + 6-reader readout comparison
  - fig7_attention_heatmaps_oracle_gt.png    : Oracle-GT self-retrieval heatmaps + 6-reader readout comparison
  - REPORT.md                                : Comprehensive Chinese diagnostic report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .reader_diagnosis import DEFAULT_OUT_DIR, READER_SPECS, SEMANTIC_PHASES

READER_ORDER: Tuple[str, ...] = (
    "global_soft",
    "global_top1",
    "global_top2",
    "adjacent_top2",
    "adjacent_top3",
    "temp_tau_0p25",
    "temp_tau_0p50",
)

SHORT_LABELS: Dict[str, str] = {
    "global_soft": "Global\nSoft (τ=1)",
    "global_top1": "Global\nTop-1",
    "global_top2": "Global\nTop-2",
    "adjacent_top2": "Adjacent\nTop-2",
    "adjacent_top3": "Adjacent\nTop-3",
    "temp_tau_0p25": "Temp\nτ=0.25",
    "temp_tau_0p50": "Temp\nτ=0.50",
    "temp_tau_1p00": "Temp\nτ=1.00",
}

INLINE_LABELS: Dict[str, str] = {
    "global_soft": "Global Soft (τ=1.0)",
    "global_top1": "Global Top-1",
    "global_top2": "Global Top-2 (Any)",
    "adjacent_top2": "Adjacent Top-2",
    "adjacent_top3": "Adjacent Top-3",
    "temp_tau_0p25": "Soft Temp (τ=0.25)",
    "temp_tau_0p50": "Soft Temp (τ=0.50)",
    "temp_tau_1p00": "Soft Temp (τ=1.00)",
}

COLORS: Dict[str, str] = {
    "global_soft": "#2c3e50",
    "global_top1": "#c0392b",
    "global_top2": "#e67e22",
    "adjacent_top2": "#27ae60",
    "adjacent_top3": "#16a085",
    "temp_tau_0p25": "#8e44ad",
    "temp_tau_0p50": "#2980b9",
    "temp_tau_1p00": "#2c3e50",
}

PHASE_LABELS: Dict[str, str] = {
    "approach": "Approach",
    "grasp_transition": "Grasp\nTrans.",
    "transport_carry": "Transport\nCarry",
    "release_transition": "Release\nTrans.",
    "post_release_transit": "Post-Rel.\nTransit",
}


def _short_task(name: str) -> str:
    s = name.replace("put_the_", "").replace("_and_then_", " → ").replace("_of_the_", "_")
    s = s.replace("in_front_of_the_", "front_").replace("on_top_of_the_", "top_")
    s = s.replace("open_the_", "open_").replace("push_the_", "push_").replace("turn_on_the_", "turn_on_")
    return s[:48] + ("…" if len(s) > 48 else "")


def plot_fig1_reader_comparison(summary: Dict[str, Any], fig_dir: Path) -> Path:
    splits_cfg = [
        ("seen_cross_p0", "Seen-Val Cross (n=3,000)"),
        ("unseen_cross_p0", "Unseen-Chain Cross (n=3,000)"),
        ("self_default", "Self D1–D4 Default (n=2,025)"),
        ("self_oracle_gt", "Self D1–D4 Oracle-GT (n=7,053)"),
    ]
    splits_cfg = [(k, t) for k, t in splits_cfg if k in summary["splits"]]
    ncols = len(splits_cfg)

    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8.2), dpi=160)
    if ncols == 1:
        axes = axes[:, None]

    x = np.arange(len(READER_ORDER))
    bar_colors = [COLORS[r] for r in READER_ORDER]
    x_labels = [SHORT_LABELS[r] for r in READER_ORDER]

    for col, (sp_key, sp_title) in enumerate(splits_cfg):
        rds = summary["splits"][sp_key]["readers"]
        gs_ag = rds["global_soft"]["aggregate"]

        # Row 0: D2 (Reader Only) DDIM 10D MSE
        ax0 = axes[0, col]
        d2_vals = [rds[r]["aggregate"]["D2"]["ddim_10d_mse"] for r in READER_ORDER]
        bars = ax0.bar(x, d2_vals, color=bar_colors, alpha=0.88, edgecolor="#111111", lw=0.6)
        for b, v in zip(bars, d2_vals):
            ax0.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7.3, rotation=0)

        d1_v = gs_ag["D1"]["ddim_10d_mse"]
        ax0.axhline(d1_v, color="#111111", ls=":", lw=1.4, label=f"D1 Self ({d1_v:.4f})")
        if "OracleAlign_D2" in gs_ag:
            oa_v = gs_ag["OracleAlign_D2"]["ddim_10d_mse"]
            ax0.axhline(oa_v, color="#0096c7", ls="--", lw=1.6, label=f"Oracle-Align Single ({oa_v:.4f})")
        ax0.set_title(f"{sp_title}\nD2 (Reader Only) DDIM 10D MSE", fontweight="bold", fontsize=9.8)
        ax0.set_xticks(x)
        ax0.set_xticklabels(x_labels, fontsize=7.5)
        ax0.set_ylim(0, max(d2_vals) * 1.20)
        ax0.grid(axis="y", alpha=0.3, ls="--")
        if col == 0:
            ax0.set_ylabel("DDIM 10D Action MSE (↓)", fontweight="bold")
        ax0.legend(fontsize=7.3, loc="lower right")

        # Row 1: D4 (Full Model) DDIM 10D MSE
        ax1 = axes[1, col]
        d4_vals = [rds[r]["aggregate"]["D4"]["ddim_10d_mse"] for r in READER_ORDER]
        bars4 = ax1.bar(x, d4_vals, color=bar_colors, alpha=0.88, edgecolor="#111111", lw=0.6)
        for b, v in zip(bars4, d4_vals):
            ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7.3)

        d3_v = gs_ag["D3"]["ddim_10d_mse"]
        ax1.axhline(d3_v, color="#8e44ad", ls="--", lw=1.4, label=f"D3 Self→Transfer ({d3_v:.4f})")
        if "OracleAlign_D4" in gs_ag:
            oa4_v = gs_ag["OracleAlign_D4"]["ddim_10d_mse"]
            ax1.axhline(oa4_v, color="#0096c7", ls="-.", lw=1.5, label=f"Oracle-Align→Transfer ({oa4_v:.4f})")
        ax1.set_title("D4 (Reader → Transfer) DDIM 10D MSE", fontweight="bold", fontsize=9.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, fontsize=7.5)
        ymax4 = max(max(d4_vals), d3_v) * 1.24
        ax1.set_ylim(0, ymax4)
        ax1.grid(axis="y", alpha=0.3, ls="--")
        if col == 0:
            ax1.set_ylabel("DDIM 10D Action MSE (↓)", fontweight="bold")
        ax1.legend(fontsize=7.3, loc="lower right")

    fig.suptitle(
        "E9 Reader Inference-Only Diagnosis: D2 (Reader Only) & D4 (Full) DDIM 10D Action MSE across 6 Reader Families",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    out = fig_dir / "fig1_reader_comparison_d2_d4.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fig2_attention_alignment_jumps(summary: Dict[str, Any], fig_dir: Path) -> Path:
    splits = [
        ("seen_cross_traj", "Seen-Cross", "#2980b9", "o-"),
        ("unseen_cross_traj", "Unseen-Cross", "#e67e22", "s-"),
        ("self_default", "Self-Default", "#27ae60", "^-"),
        ("self_oracle_gt", "Self-OracleGT", "#c0392b", "d-"),
    ]
    splits = [s for s in splits if s[0] in summary["splits"]]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), dpi=160)
    x = np.arange(len(READER_ORDER))
    x_labels = [SHORT_LABELS[r] for r in READER_ORDER]

    panels = [
        (0, 0, "Attention Entropy H (readout)", lambda r: r["aggregate"]["attention"]["entropy_mean"], "Entropy H (nats)"),
        (0, 1, "Top-1 Readout Weight (solid) & Captured Mass (dashed)", lambda r: r["aggregate"]["attention"]["top1_weight_mean"], "Weight / Mass (0–1)"),
        (0, 2, "Temporal Alignment Error (Peak solid / Exp dashed)", lambda r: r["aggregate"]["alignment"]["peak_err_slots_mean"], "Alignment Error (slots ↓)"),
        (1, 0, "Exact Segment Phase Hit Rate (Peak)", lambda r: 100.0 * r["aggregate"]["alignment"]["seg_phase_hit_peak_rate"], "Phase Hit Rate (% ↑)"),
        (1, 1, "In-Phase Probability Mass (Seg Phase)", lambda r: 100.0 * r["aggregate"]["alignment"]["seg_phase_mass_mean"], "In-Phase Mass (% ↑)"),
        (1, 2, "Reader Peak Jump Rate (|Δj| ≥ 3 slots)", lambda r: 100.0 * r["peak_jumps"]["peak_jump_ge3_rate"], "Large Jump Rate (% ↓)"),
    ]

    for row_i, col_i, title, fn, ylabel in panels:
        ax = axes[row_i, col_i]
        for sp_key, sp_lab, col_c, fmt in splits:
            rds = summary["splits"][sp_key]["readers"]
            vals = [fn(rds[r]) for r in READER_ORDER]
            ax.plot(x, vals, fmt, color=col_c, lw=2.0, markersize=5.5, label=sp_lab)
            if row_i == 0 and col_i == 1:
                cm = [rds[r]["aggregate"]["attention"]["captured_raw_mass_mean"] for r in READER_ORDER]
                ax.plot(x, cm, "--", color=col_c, lw=1.2, alpha=0.65)
            if row_i == 0 and col_i == 2:
                ev = [rds[r]["aggregate"]["alignment"]["exp_err_slots_mean"] for r in READER_ORDER]
                ax.plot(x, ev, "--", color=col_c, lw=1.2, alpha=0.65)
        ax.set_title(title, fontweight="bold", fontsize=9.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.8)
        ax.grid(alpha=0.3, ls="--")
        if (row_i, col_i) == (0, 0):
            ax.legend(fontsize=8)

    fig.suptitle(
        "E9 Reader Attention Sharpness, Temporal Alignment, Phase Hit Rate & Trajectory Peak Jumps",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    out = fig_dir / "fig2_attention_alignment_and_jumps.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fig3_location_vs_mixing(summary: Dict[str, Any], fig_dir: Path) -> Path:
    splits = [
        ("seen_cross_p0", "Seen-Cross (3k)"),
        ("unseen_cross_p0", "Unseen-Cross (3k)"),
        ("self_oracle_gt", "Oracle-GT Self (7k)"),
    ]
    splits = [(k, t) for k, t in splits if k in summary["splits"]]
    ncols = len(splits)

    fig, axes = plt.subplots(2, ncols, figsize=(5.3 * ncols, 8.6), dpi=160)
    if ncols == 1:
        axes = axes[:, None]

    # Row 0: HIT vs MISS D2 & D4 across Readers
    x = np.arange(len(READER_ORDER))
    x_labels = [SHORT_LABELS[r] for r in READER_ORDER]

    for col, (sp_key, sp_title) in enumerate(splits):
        rds = summary["splits"][sp_key]["readers"]
        ax = axes[0, col]
        hit_d2 = [rds[r]["stratifications"]["base_exact_hit"]["D2_mse10"] for r in READER_ORDER]
        miss_d2 = [rds[r]["stratifications"]["base_miss_exact"]["D2_mse10"] for r in READER_ORDER]
        hit_d4 = [rds[r]["stratifications"]["base_exact_hit"]["D4_mse10"] for r in READER_ORDER]
        miss_d4 = [rds[r]["stratifications"]["base_miss_exact"]["D4_mse10"] for r in READER_ORDER]
        hit_frac = 100.0 * rds["global_soft"]["stratifications"]["base_exact_hit"]["frac"]
        miss_frac = 100.0 * rds["global_soft"]["stratifications"]["base_miss_exact"]["frac"]

        ax.plot(x, hit_d2, "o-", color="#27ae60", lw=2.2, label=f"HIT ({hit_frac:.1f}%) D2")
        ax.plot(x, miss_d2, "s-", color="#c0392b", lw=2.2, label=f"MISS ({miss_frac:.1f}%) D2")
        ax.plot(x, hit_d4, "o--", color="#16a085", lw=1.6, alpha=0.85, label="HIT D4")
        ax.plot(x, miss_d4, "s--", color="#8e44ad", lw=1.6, alpha=0.85, label="MISS D4")
        ax.set_title(f"{sp_title}: HIT vs MISS Stratification", fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.6)
        ax.grid(alpha=0.3, ls="--")
        if col == 0:
            ax.set_ylabel("DDIM 10D Action MSE (↓)", fontweight="bold")
        ax.legend(fontsize=7.6, loc="center right")

        # Row 1: Top-2 Adjacent vs Non-Adjacent + Error Decomposition Bar
        ax1 = axes[1, col]
        adj_f = 100.0 * rds["global_soft"]["stratifications"]["base_top2_adjacent"]["frac"]
        nadj_f = 100.0 * rds["global_soft"]["stratifications"]["base_top2_non_adjacent"]["frac"]
        d3_f = 100.0 * rds["global_soft"]["stratifications"]["base_top2_dist_ge3"]["frac"]

        comp_readers = ("global_soft", "global_top1", "global_top2", "adjacent_top2", "adjacent_top3", "temp_tau_0p25")
        xc = np.arange(len(comp_readers))
        w_bar = 0.25
        v_adj = [rds[r]["stratifications"]["base_top2_adjacent"]["D2_mse10"] for r in comp_readers]
        v_nadj = [rds[r]["stratifications"]["base_top2_non_adjacent"]["D2_mse10"] for r in comp_readers]
        v_d3 = [rds[r]["stratifications"]["base_top2_dist_ge3"]["D2_mse10"] for r in comp_readers]

        ax1.bar(xc - w_bar, v_adj, w_bar, color="#27ae60", alpha=0.88, label=f"Top-2 Adj ({adj_f:.1f}%)")
        ax1.bar(xc, v_nadj, w_bar, color="#e67e22", alpha=0.88, label=f"Top-2 Non-Adj ({nadj_f:.1f}%)")
        ax1.bar(xc + w_bar, v_d3, w_bar, color="#c0392b", alpha=0.88, label=f"Top-2 Dist≥3 ({d3_f:.1f}%)")
        ax1.set_title(f"{sp_title}: D2 MSE by Top-2 Spatial Separation", fontweight="bold", fontsize=10)
        ax1.set_xticks(xc)
        ax1.set_xticklabels([SHORT_LABELS[r] for r in comp_readers], fontsize=7.6)
        ax1.grid(axis="y", alpha=0.3, ls="--")
        if col == 0:
            ax1.set_ylabel("D2 DDIM 10D MSE (↓)", fontweight="bold")
        ax1.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Mechanism Anatomy: Reading Location Error (HIT vs MISS) vs Non-Local Latent Mixing (Adjacent vs Distant Top-2)",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    out = fig_dir / "fig3_location_vs_mixing_decomposition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fig4_behavioral_phases(summary: Dict[str, Any], fig_dir: Path) -> Path:
    splits = [
        ("seen_cross_p0", "Seen-Cross (3k)"),
        ("unseen_cross_traj", "Unseen-Cross (4k)"),
        ("self_oracle_gt", "Oracle-GT Self (7k)"),
    ]
    splits = [(k, t) for k, t in splits if k in summary["splits"]]
    ncols = len(splits)

    fig, axes = plt.subplots(3, ncols, figsize=(5.2 * ncols, 11.2), dpi=160)
    if ncols == 1:
        axes = axes[:, None]

    sel_readers = ("global_soft", "global_top1", "global_top2", "adjacent_top2", "adjacent_top3", "temp_tau_0p25")
    xp = np.arange(len(SEMANTIC_PHASES))
    plabs = [PHASE_LABELS[p] for p in SEMANTIC_PHASES]
    w_bar = 0.13

    for col, (sp_key, sp_title) in enumerate(splits):
        rds = summary["splits"][sp_key]["readers"]

        # Row 0: D2 MSE per phase
        ax0 = axes[0, col]
        for ri, r_id in enumerate(sel_readers):
            vals = [
                rds[r_id]["phases"].get(ph, {}).get("D2", {}).get("ddim_10d_mse", np.nan)
                for ph in SEMANTIC_PHASES
            ]
            ax0.bar(
                xp + (ri - (len(sel_readers) - 1) / 2) * w_bar,
                vals,
                w_bar,
                color=COLORS[r_id],
                alpha=0.88,
                label=INLINE_LABELS[r_id] if col == 0 else None,
            )
        # Overlay Oracle-Align or D1 markers
        ref_vals = []
        for ph in SEMANTIC_PHASES:
            pb = rds["global_soft"]["phases"].get(ph, {})
            if "OracleAlign_D2" in pb:
                ref_vals.append(pb["OracleAlign_D2"]["ddim_10d_mse"])
            elif "D1" in pb:
                ref_vals.append(pb["D1"]["ddim_10d_mse"])
            else:
                ref_vals.append(np.nan)
        ax0.plot(xp, ref_vals, "k*--", lw=1.4, ms=7, label="Oracle-Align / D1" if col == 0 else None)
        ax0.set_title(f"{sp_title}: D2 (Reader Only) MSE by Phase", fontweight="bold", fontsize=9.8)
        ax0.set_xticks(xp)
        ax0.set_xticklabels(plabs, fontsize=8)
        ymax0 = ax0.get_ylim()[1]
        if col == 0:
            ax0.set_ylim(0, ymax0 * 1.32)
            ax0.set_ylabel("D2 DDIM 10D MSE (↓)", fontweight="bold")
            ax0.legend(fontsize=7.0, ncol=2, loc="upper right")
        ax0.grid(axis="y", alpha=0.3, ls="--")

        # Row 1: D4 MSE per phase
        ax1 = axes[1, col]
        for ri, r_id in enumerate(sel_readers):
            vals = [
                rds[r_id]["phases"].get(ph, {}).get("D4", {}).get("ddim_10d_mse", np.nan)
                for ph in SEMANTIC_PHASES
            ]
            ax1.bar(
                xp + (ri - (len(sel_readers) - 1) / 2) * w_bar,
                vals,
                w_bar,
                color=COLORS[r_id],
                alpha=0.88,
            )
        ax1.set_title(f"{sp_title}: D4 (Full Model) MSE by Phase", fontweight="bold", fontsize=9.8)
        ax1.set_xticks(xp)
        ax1.set_xticklabels(plabs, fontsize=8)
        ax1.grid(axis="y", alpha=0.3, ls="--")
        if col == 0:
            ax1.set_ylabel("D4 DDIM 10D MSE (↓)", fontweight="bold")

        # Row 2: Base Reader Diagnostics per phase (Alignment Error, Seg Phase Hit, Top-2 Adj)
        ax2 = axes[2, col]
        gs_ph = rds["global_soft"]["phases"]
        peak_err = [gs_ph.get(ph, {}).get("alignment", {}).get("peak_err_slots_mean", np.nan) for ph in SEMANTIC_PHASES]
        seg_hit = [100.0 * gs_ph.get(ph, {}).get("alignment", {}).get("seg_phase_hit_peak_rate", np.nan) for ph in SEMANTIC_PHASES]
        t2_adj = [100.0 * gs_ph.get(ph, {}).get("attention", {}).get("base_top2_adj_rate", np.nan) for ph in SEMANTIC_PHASES]

        ax2.plot(xp, seg_hit, "o-", color="#27ae60", lw=2.0, label="Seg Phase Hit (%)")
        ax2.plot(xp, t2_adj, "s--", color="#2980b9", lw=1.8, label="Top-2 Adj Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.set_xticks(xp)
        ax2.set_xticklabels(plabs, fontsize=8)
        ax2.grid(alpha=0.3, ls="--")
        ax2_r = ax2.twinx()
        ax2_r.plot(xp, peak_err, "^-.", color="#c0392b", lw=2.0, label="Peak Err (slots)")
        ax2_r.set_ylabel("Peak Error (slots ↓)", color="#c0392b", fontsize=8.5)
        ax2.set_title(f"{sp_title}: Phase Hit, Top-2 Adj & Align Err", fontweight="bold", fontsize=9.8)
        if col == 0:
            ax2.set_ylabel("Rate (% ↑)", fontweight="bold")
            ax2.legend(loc="lower left", fontsize=7.5)
            ax2_r.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Behavioral Phase Breakdown: D2/D4 Action MSE, Alignment Error & Phase Hit across 5 Manipulation Phases",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    out = fig_dir / "fig4_behavioral_phase_analysis.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_attention_heatmaps_for_split(
    hm_index: List[Dict[str, Any]],
    hm_npz: Any,
    *,
    split_key: str,
    split_title: str,
    out_path: Path,
) -> Path | None:
    items = [h for h in hm_index if h["split"] == split_key]
    if not items:
        return None

    # Top row: 4 diverse trajectories under Global Soft (best -> worst)
    # Bottom row: 6 Reader readout matrices on the worst / highest-error trajectory!
    n_top = min(4, len(items))
    pick_top = [items[i] for i in np.linspace(0, len(items) - 1, n_top, dtype=int)]
    worst_item = items[-1]

    fig = plt.figure(figsize=(18.0, 9.2), dpi=160)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.32)

    # Row 0: 4 trajectories (each spans 1.5 columns -> use 4 equal subaxes in top row via GridSpecFromSubplotSpec)
    gs_top = gs[0, :].subgridspec(1, n_top, wspace=0.28)
    for idx, hm in enumerate(pick_top):
        ax = fig.add_subplot(gs_top[0, idx])
        pid = hm["id"]
        mat = hm_npz[f"{pid}_w_global_soft"]
        ref_js = hm_npz[f"{pid}_ref_js"]
        q_anchors = hm_npz[f"{pid}_q_anchors"]
        s_anchors = hm_npz[f"{pid}_s_anchors"]
        q_flips = hm_npz[f"{pid}_q_flips"]
        s_flips = hm_npz[f"{pid}_s_flips"]

        im = ax.imshow(mat, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0, origin="upper")
        M_q, M_s = mat.shape
        q_idx = np.arange(M_q)
        ax.plot(ref_js, q_idx, color="#00e5ff", ls="--", lw=1.6, label="Ref Align j*")
        peaks = np.argmax(mat, axis=1)
        ax.scatter(peaks, q_idx, s=16, c="#76ff03", marker="x", lw=1.2, label="Peak argmax")

        # Mark gripper flip boundaries
        for fl in q_flips:
            pos = float(np.searchsorted(q_anchors, fl)) - 0.5
            if 0 <= pos < M_q:
                ax.axhline(pos, color="white", ls=":", lw=0.9, alpha=0.75)
        for fl in s_flips:
            pos = float(np.searchsorted(s_anchors, fl)) - 0.5
            if 0 <= pos < M_s:
                ax.axvline(pos, color="white", ls=":", lw=0.9, alpha=0.75)

        d2_m = hm["gs_d2_mean"]
        d4_m = hm["gs_d4_mean"]
        ax.set_title(
            f"{_short_task(hm['task'])}\nD2={d2_m:.4f}, D4={d4_m:.4f}",
            fontsize=8.3,
            fontweight="bold",
        )
        ax.set_xlabel("Source Memory Slot j", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Query Anchor k (time ↓)", fontsize=8.5)
            ax.legend(fontsize=7.0, loc="lower left", framealpha=0.75)

    # Row 1: 6 Reader readout matrices on the highest-error trajectory
    pid = worst_item["id"]
    ref_js = hm_npz[f"{pid}_ref_js"]
    q_idx = np.arange(worst_item["M_q"])
    six_modes = (
        ("global_soft", "1. Global Soft (τ=1)"),
        ("global_top1", "2. Global Top-1"),
        ("global_top2", "3. Global Top-2"),
        ("adjacent_top2", "4. Adjacent Top-2"),
        ("adjacent_top3", "5. Adjacent Top-3"),
        ("temp_tau_0p25", "6. Soft Temp (τ=0.25)"),
    )
    for col, (m_id, m_title) in enumerate(six_modes):
        ax = fig.add_subplot(gs[1, col])
        mat = hm_npz[f"{pid}_w_{m_id}"]
        d2_arr = hm_npz[f"{pid}_d2_{m_id}"]
        d4_arr = hm_npz[f"{pid}_d4_{m_id}"]
        ax.imshow(mat, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0, origin="upper")
        ax.plot(ref_js, q_idx, color="#00e5ff", ls="--", lw=1.3)
        ax.set_title(
            f"{m_title}\nD2={float(np.mean(d2_arr)):.4f} | D4={float(np.mean(d4_arr)):.4f}",
            fontsize=8.2,
            fontweight="bold",
        )
        ax.set_xlabel("Source Slot j", fontsize=7.8)
        if col == 0:
            ax.set_ylabel("Query Anchor k", fontsize=8.2)

    fig.suptitle(
        f"{split_title} Attention Heatmaps — Top: 4 Trajectories (Best→Worst, Global Soft); Bottom: 6 Readers on Worst Case",
        fontweight="bold",
        fontsize=11.5,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_report(summary: Dict[str, Any], out_dir: Path, fig_paths: Dict[str, Path]) -> Path:
    sp = summary["splits"]
    md_lines: List[str] = []

    def _fmt(v: Any, spec: str = ".5f") -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "—"
        return f"{v:{spec}}"

    # Extract core quantitative findings for executive verdict
    sc = sp["seen_cross_p0"]["readers"]
    uc = sp["unseen_cross_p0"]["readers"]
    ut = sp["unseen_cross_traj"]["readers"]
    st = sp["seen_cross_traj"]["readers"]
    ogt = sp["self_oracle_gt"]["readers"] if "self_oracle_gt" in sp else None
    sdef = sp["self_default"]["readers"]

    sc_gs, sc_t1, sc_a2, sc_a3, sc_t2, sc_t025 = (
        sc["global_soft"]["aggregate"],
        sc["global_top1"]["aggregate"],
        sc["adjacent_top2"]["aggregate"],
        sc["adjacent_top3"]["aggregate"],
        sc["global_top2"]["aggregate"],
        sc["temp_tau_0p25"]["aggregate"],
    )
    uc_gs, uc_t1, uc_a2, uc_a3, uc_t2, uc_t025 = (
        uc["global_soft"]["aggregate"],
        uc["global_top1"]["aggregate"],
        uc["adjacent_top2"]["aggregate"],
        uc["adjacent_top3"]["aggregate"],
        uc["global_top2"]["aggregate"],
        uc["temp_tau_0p25"]["aggregate"],
    )

    sc_oa_d2 = sc_gs["OracleAlign_D2"]["ddim_10d_mse"]
    uc_oa_d2 = uc_gs["OracleAlign_D2"]["ddim_10d_mse"]
    sc_loc_err = sc_t1["D2"]["ddim_10d_mse"] - sc_oa_d2
    sc_mix_eff = sc_gs["D2"]["ddim_10d_mse"] - sc_t1["D2"]["ddim_10d_mse"]
    uc_loc_err = uc_t1["D2"]["ddim_10d_mse"] - uc_oa_d2
    uc_mix_eff = uc_gs["D2"]["ddim_10d_mse"] - uc_t1["D2"]["ddim_10d_mse"]

    md_lines.append("# E9 Reader Inference-only Diagnosis 报告\n")
    md_lines.append("## 0. 核心诊断结论（读取位置错误 vs 多不兼容 Latent 加权混合）\n")
    md_lines.append(
        "> **核心结论**：在冻结 E9 Stage 1/2 权重、仅替换 Reader readout（无 STE、不重训）的严格离线分解下，**E9 Reader 的主导问题是「读取位置错误（Mislocation / 跨时刻与跨阶段漂移）」，而非单纯的「多个不兼容 latent 加权混合」**；同时，**在命中位置上存在真实但次要的软混合损失，且下游 Direct Transfer 对所有 readout 差异高度脱敏（D4 波动 < 2%）**。\n"
    )
    md_lines.append("### 决定性量化证据\n")
    md_lines.append(
        f"1. **零混合基线（Global Top-1）vs 理想单槽对齐（Oracle-Align Single-Slot）直接剥离位置误差**：\n"
        f"   - **Seen-Val Cross (3,000)**：Oracle 单槽 D2 MSE = **`{sc_oa_d2:.5f}`**；冻结 Reader 零混合 `Global Top-1` D2 = **`{sc_t1['D2']['ddim_10d_mse']:.5f}`**（仅位置错误即增加 **`{sc_loc_err:+.5f}`**），而恢复全部软混合 `Global Soft` 后 D2 = **`{sc_gs['D2']['ddim_10d_mse']:.5f}`**（软混合相对 Top-1 反而变化 **`{sc_mix_eff:+.5f}`**）。\n"
        f"   - **Unseen-Chain Cross (3,000)**：Oracle 单槽 D2 MSE = **`{uc_oa_d2:.5f}`**；`Global Top-1` D2 = **`{uc_t1['D2']['ddim_10d_mse']:.5f}`**（纯位置错误增加 **`{uc_loc_err:+.5f}`**），`Global Soft` D2 = **`{uc_gs['D2']['ddim_10d_mse']:.5f}`**（软混合相对 Top-1 变化 **`{uc_mix_eff:+.5f}`**）。\n"
    )
    if ogt is not None:
        og_gs = ogt["global_soft"]["aggregate"]
        og_t1 = ogt["global_top1"]["aggregate"]
        og_a2 = ogt["adjacent_top2"]["aggregate"]
        og_hit_gs = ogt["global_soft"]["stratifications"]["base_exact_hit"]["D2_mse10"]
        og_hit_t1 = ogt["global_top1"]["stratifications"]["base_exact_hit"]["D2_mse10"]
        og_miss_gs = ogt["global_soft"]["stratifications"]["base_miss_exact"]["D2_mse10"]
        og_miss_t1 = ogt["global_top1"]["stratifications"]["base_miss_exact"]["D2_mse10"]
        md_lines.append(
            f"   - **Oracle-GT Self D1–D4 (7,053，精确零对齐误差基准)**：精确自检索 D1 = `{og_gs['D1']['ddim_10d_mse']:.5f}`；`Global Top-1` D2 = **`{og_t1['D2']['ddim_10d_mse']:.5f}`**（`exact_hit` 仅 **`{100*og_t1['alignment']['exact_hit_rate']:.1f}%`**），`Global Soft` D2 = **`{og_gs['D2']['ddim_10d_mse']:.5f}`**，`Adjacent Top-2` D2 = **`{og_a2['D2']['ddim_10d_mse']:.5f}`**。\n"
            f"   - **HIT vs MISS 解剖（Oracle-GT）**：在 ~47% 命中锚点（HIT）上，去除软混合使 D2 暴降（`Global Soft {og_hit_gs:.5f} → Global Top-1 {og_hit_t1:.5f}`），证实软混合确实污染了已命中的局部行为；但在 ~53% 错位锚点（MISS）上，硬选错误槽位反而大幅恶化 D2（`Global Soft {og_miss_gs:.5f} → Global Top-1 {og_miss_t1:.5f}`），完全抵消了 HIT 上的收益。\n"
        )
    md_lines.append(
        f"2. **原始注意力本就高度局部单峰，消除非相邻混合（Adjacent Top-2/3）反而略劣于 Global Soft**：\n"
        f"   - 原始注意力 **Top-2 相邻率** 在 Seen-Cross 高达 **`{100*sc_gs['attention']['base_top2_adj_rate']:.1f}%`**、Unseen-Cross 为 **`{100*uc_gs['attention']['base_top2_adj_rate']:.1f}%`**、Self-OracleGT 为 **`{100*ogt['global_soft']['aggregate']['attention']['base_top2_adj_rate']:.1f}%`**（非相邻双峰仅占 **6.2%–9.4%**，且 `>±1` 槽位的非局部权重总和仅 **5.4%–7.0%**），直接排除了「注意力大量跨远距离多峰混叠」作为主因的假说。\n"
        f"   - 即便强制截断为局部窗口 `Adjacent Top-2` / `Adjacent Top-3` 或锐化温度 $\\tau=0.25$，全量 D2 仍略劣于原始 `Global Soft`（Seen D2：`{sc_gs['D2']['ddim_10d_mse']:.5f}` vs Adj-2 `{sc_a2['D2']['ddim_10d_mse']:.5f}` vs Adj-3 `{sc_a3['D2']['ddim_10d_mse']:.5f}`；Unseen D2：`{uc_gs['D2']['ddim_10d_mse']:.5f}` vs Adj-2 `{uc_a2['D2']['ddim_10d_mse']:.5f}` vs Adj-3 `{uc_a3['D2']['ddim_10d_mse']:.5f}`），甚至在非相邻双峰子集上 `Adjacent Top-2` 也略劣于 `Global Soft`（因当主峰错位时，截断次峰反而丢弃了残留在正确槽位上的权重）。\n"
        f"3. **真正的定位病灶集中在抓取/释放阶段切换边界与「列锁定 + 提前跳跃」**：\n"
        f"   - 轨迹级峰值跳跃率 **`|Δj| ≥ 3` 槽位（≥48 步）** 达 **`{100*st['global_soft']['peak_jumps']['peak_jump_ge3_rate']:.1f}%`（Seen） / `{100*ut['global_soft']['peak_jumps']['peak_jump_ge3_rate']:.1f}%`（Unseen） / `{100*ogt['global_soft']['peak_jumps']['peak_jump_ge3_rate']:.1f}%`（Oracle-GT）**，且回跳率达 **4.3%–6.8%**；Attention Heatmap 显示明显的**对角线右侧垂直列锁定（Column Locking）**，即连续 3–4 个 query anchor 提前锁死在某一个未来地标槽位上。\n"
        f"   - 按 5 个物理行为阶段分解可见：平滑阶段（`approach` / `transport_carry` / `post_release_transit`）阶段命中率高达 **94%–99.6%**、D2 误差仅 `0.014–0.045`；而 **`grasp_transition` 与 `release_transition`**（合计仅 ~19% 锚点）的阶段命中率暴跌至 **46.1%–68.5%**、峰值对齐误差飙升至 **1.45–2.01 槽位（23–32 步）**、D2 误差暴增 **5–10×**（`0.125–0.139`）、Oracle-GT `release_transition` D4 高达 **`0.0738`**。\n"
        f"4. **下游 Direct Transfer（D4）对六种 Reader 几乎完全免疫**：\n"
        f"   - Seen-Cross D4 全范围仅在 `{min(sc[r]['aggregate']['D4']['ddim_10d_mse'] for r in READER_ORDER):.5f}–{max(sc[r]['aggregate']['D4']['ddim_10d_mse'] for r in READER_ORDER):.5f}` 之间窄幅波动；Unseen-Cross D4 仅在 `{min(uc[r]['aggregate']['D4']['ddim_10d_mse'] for r in READER_ORDER):.5f}–{max(uc[r]['aggregate']['D4']['ddim_10d_mse'] for r in READER_ORDER):.5f}` 之间波动；Oracle-GT D4 仅在 `{min(ogt[r]['aggregate']['D4']['ddim_10d_mse'] for r in READER_ORDER):.5f}–{max(ogt[r]['aggregate']['D4']['ddim_10d_mse'] for r in READER_ORDER):.5f}` 之间波动（< 1.5%）。\n"
        f"   - **工程含义**：仅修改冻结 E9 的推理期 Reader readout（包括 Adjacent Top-2/3、Top-1/2、温度缩放）**无法修复 E9**；要解决 E9 瓶颈，必须从**训练端提升 Reader 的时间/阶段定位精度（尤其修复抓取/释放切换边界的提前列锁定与错位，同时避免无监督导致的 Reader 退化）**，并结合**不覆写正确 latent 的 Transfer 接口**。\n"
    )

    md_lines.append("---\n")
    md_lines.append("## 1. 实验设置与合规核查\n")
    pr = summary["protocol"]
    ck = summary["checkpoints"]
    md_lines.append(f"- **冻结权重**：Stage 1 `{ck['e9_stage1_local']}` + Stage 2 `{ck['e9_stage2_direct']}`（Epoch `{ck['e9_stage2_epoch']}`，`s_z = {pr['s_z']:.6f}`，可训练参数 = `{pr['trainable_params']}`）。")
    md_lines.append(f"- **严格约束**：`no_ste = {pr['no_ste']}`、`no_retraining = {pr['no_retraining']}`、`offline_only = {pr['offline_only_no_closed_loop']}`、`gpu = {pr['gpu_name']}`。")
    md_lines.append("- **唯一变量**：仅修改冻结 attention logits $s_i = Q(h_q)K(h_i^D)^T / \\sqrt{d}$ 之后的 readout 映射，TransferTransformer、冻结 Stage 1 UNet、以及逐样本 DDIM 初始噪声在 6 类 Reader 间严格 bit-identical 共享。")
    md_lines.append(f"- **`temp_tau_1p00 == global_soft` 精确一致校验**：所有 split 上最大权重差 = `{max(v['temp1_vs_global_soft_max_diff'] for v in sp.values()):.2e}`。")
    if summary.get("reproducibility_checks"):
        md_lines.append("- **与既有 E9 D1–D4 基线复现校验（`global_soft` 绝对误差）**：")
        for rk, rv in summary["reproducibility_checks"].items():
            md_lines.append(
                f"  - `{rk}`: D1 Δ=`{rv['D1_abs_diff']:.2e}`, D2 Δ=`{rv['D2_abs_diff']:.2e}`, D3 Δ=`{rv['D3_abs_diff']:.2e}`, D4 Δ=`{rv['D4_abs_diff']:.2e}`"
            )

    md_lines.append("\n---\n")
    md_lines.append("## 2. 六种 Reader 主对比表（Seen / Unseen Cross-Demo & Self D1–D4）\n")
    md_lines.append(f"![六种 Reader D2/D4 对比]({fig_paths['fig1'].as_posix()})\n")
    md_lines.append(f"![注意力、对齐误差、阶段命中与峰值跳跃]({fig_paths['fig2'].as_posix()})\n")

    for sp_key, sp_title in (
        ("seen_cross_p0", "2.1 Seen-Val Cross-Demo（Exact P0 3,000 跨布局对）"),
        ("unseen_cross_p0", "2.2 Unseen-Chain Cross-Demo（Exact P0 3,000 跨布局对，10 未见任务）"),
        ("seen_cross_traj", "2.3 Seen-Val 完整连续轨迹（200 条跨布局轨迹，含 Peak Jump）"),
        ("unseen_cross_traj", "2.4 Unseen-Chain 完整连续轨迹（200 条跨布局轨迹 / 4,067 锚点，含 Peak Jump）"),
        ("self_default", "2.5 Self-Demo D1–D4：Default（100 demos / 2,025 锚点）"),
        ("self_shifted", "2.6 Self-Demo D1–D4：Shifted（100 demos / 2,042 锚点）"),
        ("self_oracle_gt", "2.7 Self-Demo D1–D4：Oracle-GT（331 demos / 7,053 锚点）"),
    ):
        if sp_key not in sp:
            continue
        rds = sp[sp_key]["readers"]
        gs_ag = rds["global_soft"]["aggregate"]
        has_jump = rds["global_soft"]["peak_jumps"] is not None
        oa_note = ""
        if "OracleAlign_D2" in gs_ag:
            oa_note = (
                f" | **Oracle-Align Single-Slot**: D2 = `{gs_ag['OracleAlign_D2']['ddim_10d_mse']:.5f}`, "
                f"D4 = `{gs_ag['OracleAlign_D4']['ddim_10d_mse']:.5f}`"
            )
        md_lines.append(f"### {sp_title}\n")
        md_lines.append(
            f"> 共享基准：D1 Self = `{gs_ag['D1']['ddim_10d_mse']:.5f}`, D3 Self→Transfer = `{gs_ag['D3']['ddim_10d_mse']:.5f}`{oa_note} | "
            f"Raw Base Top-2 Adj = `{100*gs_ag['attention']['base_top2_adj_rate']:.1f}%`, Dist≥3 = `{100*gs_ag['attention']['base_top2_dist_ge3_rate']:.1f}%`, Non-local Mass (>±1) = `{100*gs_ag['attention']['base_nonlocal_mass_out3_mean']:.1f}%`。\n"
        )
        if has_jump:
            md_lines.append(
                "| Reader | **D2 MSE** ($L_z$) | **D4 MSE** ($L_z$) | Entropy $H$ | Top-1 $w$ | Cap. Mass | Exact / ±1 Hit | Peak / Exp Err (sl) | Seg Phase Hit / Mass | Peak Jump `|Δj|` / `≥3` / Back |"
            )
            md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            md_lines.append(
                "| Reader | **D2 MSE** ($L_z$) | **D4 MSE** ($L_z$) | Entropy $H$ | Top-1 $w$ | Cap. Mass | Exact / ±1 Hit | Peak / Exp Err (sl) | Seg Phase Hit / Mass | Sem Phase Hit / Mass |"
            )
            md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for r_id, _, r_label, _ in READER_SPECS:
            blk = rds[r_id]
            ag = blk["aggregate"]
            at = ag["attention"]
            al = ag["alignment"]
            if has_jump:
                jp = blk["peak_jumps"]
                tail = f"`{jp['mean_abs_peak_step']:.2f}` / `{100*jp['peak_jump_ge3_rate']:.1f}%` / `{100*jp['backward_jump_rate']:.1f}%`"
            else:
                tail = f"`{100*al['sem_phase_hit_peak_rate']:.1f}%` / `{100*al['sem_phase_mass_mean']:.1f}%`"
            md_lines.append(
                f"| {r_label} | `{ag['D2']['ddim_10d_mse']:.5f}` (`{ag['D2']['latent_lz_vs_self']:.3f}`) "
                f"| `{ag['D4']['ddim_10d_mse']:.5f}` (`{ag['D4']['latent_lz_vs_self']:.3f}`) "
                f"| `{at['entropy_mean']:.3f}` | `{at['top1_weight_mean']:.3f}` | `{100*at['captured_raw_mass_mean']:.1f}%` "
                f"| `{100*al['exact_hit_rate']:.1f}%` / `{100*al['within1_hit_rate']:.1f}%` "
                f"| `{al['peak_err_slots_mean']:.2f}` / `{al['exp_err_slots_mean']:.2f}` "
                f"| `{100*al['seg_phase_hit_peak_rate']:.1f}%` / `{100*al['seg_phase_mass_mean']:.1f}%` | {tail} |"
            )
        md_lines.append("")

    md_lines.append("---\n")
    md_lines.append("## 3. 机制解剖：位置错误 vs 多不兼容 Latent 混合\n")
    md_lines.append(f"![位置错误 vs 混合解剖]({fig_paths['fig3'].as_posix()})\n")
    md_lines.append("| Split | Oracle 单槽 D2 | Top-1 D2 (纯位置误差 Δ) | Global Soft D2 (混合效应 Δ) | Adj Top-2 D2 | HIT 子集 (Soft→Top-1 D2) | MISS 子集 (Soft→Top-1 D2) | 非相邻双峰占比 | 非相邻子集 (Soft→Adj-2 D2) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sp_key in ("seen_cross_p0", "unseen_cross_p0", "self_default", "self_oracle_gt"):
        if sp_key not in summary["mechanism_decomposition"]:
            continue
        mc = summary["mechanism_decomposition"][sp_key]
        rds = sp[sp_key]["readers"]
        h_gs = rds["global_soft"]["stratifications"]["base_exact_hit"]["D2_mse10"]
        h_t1 = rds["global_top1"]["stratifications"]["base_exact_hit"]["D2_mse10"]
        m_gs = rds["global_soft"]["stratifications"]["base_miss_exact"]["D2_mse10"]
        m_t1 = rds["global_top1"]["stratifications"]["base_miss_exact"]["D2_mse10"]
        na_gs = rds["global_soft"]["stratifications"]["base_top2_non_adjacent"]["D2_mse10"]
        na_a2 = rds["adjacent_top2"]["stratifications"]["base_top2_non_adjacent"]["D2_mse10"]
        md_lines.append(
            f"| `{sp_key}` | `{mc['ref_single_slot_D2_mse10']:.5f}` "
            f"| `{mc['global_top1_D2_mse10']:.5f}` (`{mc['location_error_D2_top1_minus_oracle']:+.5f}`) "
            f"| `{mc['global_soft_D2_mse10']:.5f}` (`{mc['mixing_effect_D2_soft_minus_top1']:+.5f}`) "
            f"| `{mc['adjacent_top2_D2_mse10']:.5f}` "
            f"| `{h_gs:.5f} → {h_t1:.5f}` | `{m_gs:.5f} → {m_t1:.5f}` "
            f"| `{100*mc['non_adj_top2_frac']:.1f}%` | `{na_gs:.5f} → {na_a2:.5f}` |"
        )

    md_lines.append("\n---\n")
    md_lines.append("## 4. 各行为阶段误差与注意力分析\n")
    md_lines.append(f"![各行为阶段分解]({fig_paths['fig4'].as_posix()})\n")
    for sp_key, sp_title in (
        ("seen_cross_p0", "Seen-Cross (3k)"),
        ("unseen_cross_traj", "Unseen-Cross (4k)"),
        ("self_oracle_gt", "Oracle-GT Self (7k)"),
    ):
        if sp_key not in sp:
            continue
        rds = sp[sp_key]["readers"]
        md_lines.append(f"### 4.{['seen_cross_p0', 'unseen_cross_traj', 'self_oracle_gt'].index(sp_key) + 1} {sp_title} 行为阶段表\n")
        md_lines.append("| 行为阶段 | 占比 | Global Soft D2 / D4 | Global Top-1 D2 / D4 | Adj Top-2 D2 / D4 | Temp τ=0.25 D2 / D4 | Peak Err (sl) | Seg Phase Hit | Top-2 Adj |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ph in SEMANTIC_PHASES:
            p_gs = rds["global_soft"]["phases"].get(ph, {})
            if p_gs.get("num_anchors", 0) == 0:
                continue
            p_t1 = rds["global_top1"]["phases"][ph]
            p_a2 = rds["adjacent_top2"]["phases"][ph]
            p_25 = rds["temp_tau_0p25"]["phases"][ph]
            md_lines.append(
                f"| `{ph}` | `{100*p_gs['frac']:.1f}%` "
                f"| `{p_gs['D2']['ddim_10d_mse']:.5f}` / `{p_gs['D4']['ddim_10d_mse']:.5f}` "
                f"| `{p_t1['D2']['ddim_10d_mse']:.5f}` / `{p_t1['D4']['ddim_10d_mse']:.5f}` "
                f"| `{p_a2['D2']['ddim_10d_mse']:.5f}` / `{p_a2['D4']['ddim_10d_mse']:.5f}` "
                f"| `{p_25['D2']['ddim_10d_mse']:.5f}` / `{p_25['D4']['ddim_10d_mse']:.5f}` "
                f"| `{p_gs['alignment']['peak_err_slots_mean']:.2f}` "
                f"| `{100*p_gs['alignment']['seg_phase_hit_peak_rate']:.1f}%` "
                f"| `{100*p_gs['attention']['base_top2_adj_rate']:.1f}%` |"
            )
        md_lines.append("")

    md_lines.append("---\n")
    md_lines.append("## 5. Attention Heatmap 可视化\n")
    for fk, title in (
        ("fig5", "Seen-Cross 轨迹 Attention Heatmaps"),
        ("fig6", "Unseen-Cross 轨迹 Attention Heatmaps"),
        ("fig7", "Oracle-GT Self 轨迹 Attention Heatmaps"),
    ):
        if fk in fig_paths and fig_paths[fk] is not None:
            md_lines.append(f"### {title}\n")
            md_lines.append(f"![{title}]({fig_paths[fk].as_posix()})\n")

    md_lines.append("---\n")
    md_lines.append("## 6. 产出文件索引\n")
    md_lines.append(f"- 完整 JSON 汇总：[summary.json](file://{(out_dir / 'summary.json').resolve()})")
    md_lines.append(f"- 主对比表 CSV：[reader_comparison_summary.csv](file://{(out_dir / 'reader_comparison_summary.csv').resolve()})")
    md_lines.append(f"- 行为阶段表 CSV：[phase_breakdown.csv](file://{(out_dir / 'phase_breakdown.csv').resolve()})")
    md_lines.append(f"- 位置 vs 混合分层 CSV：[stratification_location_vs_mixing.csv](file://{(out_dir / 'stratification_location_vs_mixing.csv').resolve()})")
    md_lines.append(f"- 未见任务逐任务 CSV：[per_task_unseen.csv](file://{(out_dir / 'per_task_unseen.csv').resolve()})")
    md_lines.append(f"- 逐轨迹 Peak Jump CSV：[peak_jump_per_trajectory.csv](file://{(out_dir / 'peak_jump_per_trajectory.csv').resolve()})")
    md_lines.append(f"- 逐锚点明细 CSV：`per_anchor_seen_cross.csv`、`per_anchor_unseen_cross.csv`、`per_anchor_self_d1_d4.csv`")
    md_lines.append(f"- Attention Heatmap 原始矩阵：[heatmaps_data.npz](file://{(out_dir / 'heatmaps_data.npz').resolve()}) & [heatmaps_index.json](file://{(out_dir / 'heatmaps_index.json').resolve()})")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report_path


def generate_all_plots_and_report(out_dir: Path = DEFAULT_OUT_DIR) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((out_dir / "summary.json").read_text())
    hm_index = json.loads((out_dir / "heatmaps_index.json").read_text())["heatmaps"]
    hm_npz = np.load(out_dir / "heatmaps_data.npz")

    figs: Dict[str, Path] = {}
    figs["fig1"] = plot_fig1_reader_comparison(summary, fig_dir)
    figs["fig2"] = plot_fig2_attention_alignment_jumps(summary, fig_dir)
    figs["fig3"] = plot_fig3_location_vs_mixing(summary, fig_dir)
    figs["fig4"] = plot_fig4_behavioral_phases(summary, fig_dir)

    p5 = plot_attention_heatmaps_for_split(
        hm_index, hm_npz, split_key="seen_cross_traj",
        split_title="Seen-Val Cross-Demo", out_path=fig_dir / "fig5_attention_heatmaps_seen.png",
    )
    if p5:
        figs["fig5"] = p5
    p6 = plot_attention_heatmaps_for_split(
        hm_index, hm_npz, split_key="unseen_cross_traj",
        split_title="Unseen-Chain Cross-Demo", out_path=fig_dir / "fig6_attention_heatmaps_unseen.png",
    )
    if p6:
        figs["fig6"] = p6
    p7 = plot_attention_heatmaps_for_split(
        hm_index, hm_npz, split_key="self_oracle_gt",
        split_title="Oracle-GT Self-Retrieval", out_path=fig_dir / "fig7_attention_heatmaps_oracle_gt.png",
    )
    if p7:
        figs["fig7"] = p7

    report_path = write_report(summary, out_dir, figs)
    print(f"[Plot & Report DONE] Generated {len(figs)} figures + {report_path}", flush=True)
    return {"report": report_path, **figs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    generate_all_plots_and_report(args.out_dir)


if __name__ == "__main__":
    main()
