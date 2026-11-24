"""Aggregate Stage 1 Behavior Space Diagnosis (Offline D1-D4 + Stage 1 Closed-Loop)
and compare paired trials against E7-Direct Best (Default 500, Oracle-GT 331, Shifted 500).
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

from . import OUTPUT_ROOT
from .diagnose_behavior_space import MODES

DIAG_ROOT = OUTPUT_ROOT / "behavior_space_diagnosis"
DIRECT_ROOT = OUTPUT_ROOT / "direct_latent"

SHORT_TASK_NAMES = {
    "open_the_middle_drawer_of_the_cabinet_and_then_put_the_cream_cheese_on_the_stove": "open_mid_drawer->cheese_stove",
    "open_the_top_drawer_and_then_put_the_wine_bottle_on_top_of_the_cabinet": "open_top_drawer->wine_cabinet",
    "push_the_plate_to_the_front_of_the_stove_and_then_put_the_bowl_on_the_plate": "push_plate->bowl_plate",
    "put_the_bowl_on_the_plate_and_then_put_the_cream_cheese_in_the_bowl": "bowl_plate->cheese_bowl",
    "put_the_bowl_on_top_of_the_cabinet_and_then_put_the_cream_cheese_on_the_plate": "bowl_cabinet->cheese_plate",
    "put_the_cream_cheese_on_the_plate_and_then_put_the_bowl_on_the_cream_cheese": "cheese_plate->bowl_cheese",
    "put_the_cream_cheese_on_top_of_the_cabinet_and_then_put_the_wine_bottle_in_the_bowl": "cheese_cabinet->wine_bowl",
    "put_the_wine_bottle_in_front_of_the_stove_and_then_put_the_bowl_on_the_stove": "wine_stove->bowl_stove",
    "put_the_wine_bottle_on_the_rack_and_then_put_the_bowl_on_top_of_the_cabinet": "wine_rack->bowl_cabinet",
    "turn_on_the_stove_and_then_put_the_bowl_on_the_stove": "turn_on_stove->bowl_stove",
}


def _load_rollout_csv(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["task"], int(row["layout"]))] = {
                "success": str(row["success"]).lower() == "true",
                "steps": int(row["steps"]),
                "replans": int(row["replans"]),
                "donor": row.get("donor", ""),
            }
    return out


def _verify_init_sha256(dir_a: Path, dir_b: Path, keys: List[Tuple[str, int]]) -> Dict[str, Any]:
    checked = 0
    matched = 0
    mismatches = []
    for task, qi in keys:
        pa = dir_a / task / f"layout_{qi:02d}" / "result.json"
        pb = dir_b / task / f"layout_{qi:02d}" / "result.json"
        if not pa.is_file() or not pb.is_file():
            continue
        ja = json.loads(pa.read_text())
        jb = json.loads(pb.read_text())
        checked += 1
        if ja.get("init_sha256") == jb.get("init_sha256"):
            matched += 1
        else:
            mismatches.append((task, qi))
    return {
        "checked": checked,
        "matched": matched,
        "all_match": (checked == len(keys) and matched == checked),
        "mismatches": mismatches[:10],
    }


def _paired_stats(
    map_a: Dict[Tuple[str, int], Dict[str, Any]],
    map_b: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    n = len(keys)
    if n == 0:
        return {"n": 0}
    a_ok = sum(1 for k in keys if map_a[k]["success"])
    b_ok = sum(1 for k in keys if map_b[k]["success"])
    w = sum(1 for k in keys if map_a[k]["success"] and not map_b[k]["success"])
    l = sum(1 for k in keys if not map_a[k]["success"] and map_b[k]["success"])
    a_steps = [map_a[k]["steps"] for k in keys if map_a[k]["success"]]
    b_steps = [map_b[k]["steps"] for k in keys if map_b[k]["success"]]
    return {
        "n": n,
        "a_ok": a_ok,
        "b_ok": b_ok,
        "a_sr": a_ok / n,
        "b_sr": b_ok / n,
        "delta_pp": 100.0 * (a_ok - b_ok) / n,
        "wins": w,
        "losses": l,
        "ties": n - w - l,
        "a_mean_success_steps": round(float(np.mean(a_steps)), 1) if a_steps else None,
        "b_mean_success_steps": round(float(np.mean(b_steps)), 1) if b_steps else None,
    }


def _plot_offline_decomposition(offline: Dict[str, Any], out_path: Path) -> None:
    macro9 = offline["by_demo_type_macro_shared_9"]
    dts = ["default", "shifted", "oracle_gt"]
    dt_labels = ["Default\n(9 shared)", "Shifted\n(9 shared)", "Oracle-GT\n(9 shared)"]
    mode_labels = ["D1 Teacher", "D2 Retrieval", "D3 Transfer", "D4 Full"]
    colors = ["#2b5c8f", "#e08214", "#7b3294", "#1b7837"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=160)
    panels = [
        ("ddim_10d_mse", "DDIM 10D Action MSE (↓)", False),
        ("ddim_7d_mae", "DDIM 7D Action MAE (↓)", False),
        ("exec12_7d_mae", "Exec-12 7D Action MAE (↓)", False),
        ("grip_acc", "Gripper Sign Accuracy % (↑)", True),
        ("latent_mse_vs_self", "Latent MSE vs Exact Self b_k (↓)", False),
        ("latent_cos_vs_self", "Latent Cosine vs Exact Self b_k (↑)", False),
    ]
    x = np.arange(len(dts))
    width = 0.19

    for ax, (metric, title, pct) in zip(axes.flat, panels):
        for i, (m, ml, col) in enumerate(zip(MODES, mode_labels, colors)):
            vals = [
                macro9[dt]["modes"][m][metric] * (100.0 if pct else 1.0)
                for dt in dts
            ]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=ml, color=col, alpha=0.9)
            for b, v in zip(bars, vals):
                if metric == "latent_cos_vs_self":
                    lbl = f"{v:.4f}"
                elif pct:
                    lbl = f"{v:.1f}"
                elif metric == "latent_mse_vs_self":
                    lbl = f"{v:.2f}"
                else:
                    lbl = f"{v:.4f}"
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height(),
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    rotation=25,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(dt_labels, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if metric == "latent_cos_vs_self":
            lo = min(
                macro9[dt]["modes"][m][metric]
                for dt in dts
                for m in MODES
            )
            ax.set_ylim(max(0.0, lo - 0.012), 1.006)
        elif pct:
            lo = min(
                macro9[dt]["modes"][m][metric] * 100.0
                for dt in dts
                for m in MODES
            )
            ax.set_ylim(max(0.0, lo - 5.0), 102.5)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.14)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Stage 1 Behavior Space Offline Decomposition (9-Task Equal-Weight Macro Average)",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.905])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_task_overview(
    offline: Dict[str, Any],
    closed_loop: Dict[str, Any],
    tasks: List[str],
    out_path: Path,
) -> None:
    short = [SHORT_TASK_NAMES.get(t, t[:24]) for t in tasks]
    y = np.arange(len(tasks))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=160, sharey=True)

    # Left: Offline D1 vs D4 DDIM 10D MSE for Default & Oracle-GT
    ax = axes[0]
    pt_def = offline["per_task_by_demo_type"]["default"]
    pt_ogt = offline["per_task_by_demo_type"]["oracle_gt"]
    d1_def = [pt_def[t]["modes"]["D1_teacher"]["ddim_10d_mse"] for t in tasks]
    d4_def = [pt_def[t]["modes"]["D4_full"]["ddim_10d_mse"] for t in tasks]
    d1_ogt = [
        pt_ogt[t]["modes"]["D1_teacher"]["ddim_10d_mse"] if t in pt_ogt else np.nan
        for t in tasks
    ]
    d4_ogt = [
        pt_ogt[t]["modes"]["D4_full"]["ddim_10d_mse"] if t in pt_ogt else np.nan
        for t in tasks
    ]
    h = 0.2
    ax.barh(y - 1.5 * h, d1_def, h, label="Default D1 (Teacher)", color="#2b5c8f")
    ax.barh(y - 0.5 * h, d4_def, h, label="Default D4 (Full)", color="#92c5de")
    ax.barh(y + 0.5 * h, np.nan_to_num(d1_ogt, nan=0.0), h, label="Oracle-GT D1 (Teacher)", color="#b2182b")
    ax.barh(y + 1.5 * h, np.nan_to_num(d4_ogt, nan=0.0), h, label="Oracle-GT D4 (Full)", color="#f4a582")
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Offline DDIM 10D Action MSE (↓)", fontsize=9.5)
    ax.set_title("Per-Task Offline Action MSE: D1 vs D4", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    # Right: Closed-loop SR: Stage 1 Teacher vs E7-Direct on Default & Oracle-GT
    ax = axes[1]
    pt_cl = closed_loop["per_task"]
    s1_def_sr = [100.0 * pt_cl[t]["stage1_default"]["sr"] for t in tasks]
    s1_ogt_sr = [
        100.0 * pt_cl[t]["stage1_oracle_gt"]["sr"] if pt_cl[t]["stage1_oracle_gt"] else 0.0
        for t in tasks
    ]
    d_def_sr = [100.0 * pt_cl[t]["e7_direct_default"]["sr"] for t in tasks]
    d_ogt_sr = [
        100.0 * pt_cl[t]["e7_direct_oracle_gt"]["sr"] if pt_cl[t]["e7_direct_oracle_gt"] else 0.0
        for t in tasks
    ]
    ax.barh(y - 1.5 * h, s1_def_sr, h, label="Stage 1 Teacher — Default", color="#2b5c8f")
    ax.barh(y - 0.5 * h, s1_ogt_sr, h, label="Stage 1 Teacher — Oracle-GT", color="#b2182b")
    ax.barh(y + 0.5 * h, d_def_sr, h, label="E7-Direct — Default", color="#1b7837")
    ax.barh(y + 1.5 * h, d_ogt_sr, h, label="E7-Direct — Oracle-GT", color="#a6dba0")
    ax.set_xlabel("Closed-Loop Success Rate % (↑)", fontsize=9.5)
    ax.set_xlim(0, 108)
    ax.set_title("Per-Task Closed-Loop SR: Stage 1 Teacher vs E7-Direct", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_reader_and_d3_strat(offline: Dict[str, Any], out_path: Path) -> None:
    dts = ["default", "shifted", "oracle_gt"]
    dt_labels = ["Default", "Shifted", "Oracle-GT"]
    micro = offline["by_demo_type_micro"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=160)

    # Left: Reader metrics across demo types
    ax = axes[0]
    x = np.arange(len(dts))
    w = 0.25
    max_w = [micro[dt]["reader"]["max_weight_mean"] for dt in dts]
    self_w = [micro[dt]["reader"]["self_weight_mean"] for dt in dts]
    exact_hit = [micro[dt]["reader"]["exact_hit_rate"] for dt in dts]
    ax.bar(x - w, max_w, w, label="Max Weight (w_max)", color="#2b5c8f")
    ax.bar(x, self_w, w, label="Self Anchor Weight (w_k)", color="#e08214")
    ax.bar(x + w, exact_hit, w, label="Exact Anchor Hit Rate", color="#1b7837")
    for xi, (a, b, c) in enumerate(zip(max_w, self_w, exact_hit)):
        ax.text(xi - w, a + 0.015, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi, b + 0.015, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(xi + w, c + 0.015, f"{c:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dt_labels, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Probability / Weight")
    ax.set_title("Reader Self-Retrieval Accuracy by Demo Source", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8.5)

    # Right: D1/D2/D3/D4 10D MSE stratified by self_weight bucket on Oracle-GT
    ax = axes[1]
    buckets = ["w_self_ge_0.9", "w_self_0.5_to_0.9", "w_self_lt_0.5"]
    b_labels = ["w_self ≥ 0.9", "0.5 ≤ w_self < 0.9", "w_self < 0.5"]
    strat = micro["oracle_gt"]["d3_stratification"]
    x2 = np.arange(len(buckets))
    w2 = 0.19
    cols = ["#2b5c8f", "#e08214", "#7b3294", "#1b7837"]
    for i, (key, lbl, col) in enumerate(
        zip(
            ("D1_mse10", "D2_mse10", "D3_mse10", "D4_mse10"),
            ("D1 Teacher", "D2 Retrieval", "D3 Transfer", "D4 Full"),
            cols,
        )
    ):
        vals = [strat[b].get(key, 0.0) for b in buckets]
        ax.bar(x2 + (i - 1.5) * w2, vals, w2, label=lbl, color=col)
    x_ticks_lbl = [
        f"{bl}\n(n={strat[b]['num_anchors']}, {100*strat[b]['frac']:.1f}%)"
        for bl, b in zip(b_labels, buckets)
    ]
    ax.set_xticks(x2)
    ax.set_xticklabels(x_ticks_lbl, fontsize=8.5)
    ax.set_ylabel("DDIM 10D Action MSE (↓)")
    ax.set_title("Oracle-GT: D1–D4 Stratified by Reader Self-Weight w_k", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_diagnosis_summary(
    *,
    diag_root: Path = DIAG_ROOT,
    direct_root: Path = DIRECT_ROOT,
) -> Dict[str, Any]:
    tasks = get_chain_tasks()
    offline = json.loads((diag_root / "offline/offline_summary.json").read_text())

    s1_root = diag_root / "closed_loop/stage1_teacher"
    s1_def_dir = s1_root / "default"
    s1_ogt_dir = s1_root / "oracle_gt"
    s1_shf_dir = s1_root / "shifted"

    d_def_dir = direct_root / "eval_chain/e7_direct"
    d_ogt_dir = direct_root / "demoswap/oracle_gt/e7_direct"
    d_shf_dir = direct_root / "demoswap/shifted/e7_direct"

    s1_def = _load_rollout_csv(s1_def_dir / "rollout_results.csv")
    s1_ogt = _load_rollout_csv(s1_ogt_dir / "rollout_results.csv")
    s1_shf = _load_rollout_csv(s1_shf_dir / "rollout_results.csv")

    d_def = _load_rollout_csv(d_def_dir / "rollout_results.csv")
    d_ogt = _load_rollout_csv(d_ogt_dir / "rollout_results.csv")
    d_shf = _load_rollout_csv(d_shf_dir / "rollout_results.csv")

    # Verify exact init_sha256 alignment across all 500 + 331 + 500 trials
    sha_checks = {
        "default_500": _verify_init_sha256(s1_def_dir, d_def_dir, sorted(s1_def.keys())),
        "oracle_gt_331": _verify_init_sha256(s1_ogt_dir, d_ogt_dir, sorted(s1_ogt.keys())),
        "shifted_500": _verify_init_sha256(s1_shf_dir, d_shf_dir, sorted(s1_shf.keys())),
    }

    s1_vs_d_def = _paired_stats(s1_def, d_def)
    s1_vs_d_ogt = _paired_stats(s1_ogt, d_ogt)
    s1_vs_d_shf = _paired_stats(s1_shf, d_shf)

    s1_ogt_vs_def = _paired_stats(s1_ogt, s1_def)
    s1_shf_vs_def = _paired_stats(s1_shf, s1_def)
    d_ogt_vs_def = _paired_stats(d_ogt, d_def)
    d_shf_vs_def = _paired_stats(d_shf, d_def)

    s1_mean3 = (s1_vs_d_def["a_sr"] + s1_vs_d_ogt["a_sr"] + s1_vs_d_shf["a_sr"]) / 3.0
    d_mean3 = (s1_vs_d_def["b_sr"] + s1_vs_d_ogt["b_sr"] + s1_vs_d_shf["b_sr"]) / 3.0

    per_task: Dict[str, Any] = {}
    pt_csv_rows: List[Dict[str, Any]] = []
    for task in tasks:
        def _tsr(m: Dict[Tuple[str, int], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            sub = [v for (t, _), v in m.items() if t == task]
            if not sub:
                return None
            ok = sum(1 for v in sub if v["success"])
            steps_ok = [v["steps"] for v in sub if v["success"]]
            return {
                "ok": ok,
                "n": len(sub),
                "sr": ok / len(sub),
                "mean_success_steps": round(float(np.mean(steps_ok)), 1) if steps_ok else None,
            }

        entry = {
            "stage1_default": _tsr(s1_def),
            "e7_direct_default": _tsr(d_def),
            "stage1_oracle_gt": _tsr(s1_ogt),
            "e7_direct_oracle_gt": _tsr(d_ogt),
            "stage1_shifted": _tsr(s1_shf),
            "e7_direct_shifted": _tsr(d_shf),
        }
        # Paired Oracle-GT vs Default on this task's Oracle-GT subset
        ogt_keys_t = [k for k in s1_ogt.keys() if k[0] == task]
        if ogt_keys_t:
            s1_def_sub = {k: s1_def[k] for k in ogt_keys_t}
            s1_ogt_sub = {k: s1_ogt[k] for k in ogt_keys_t}
            d_def_sub = {k: d_def[k] for k in ogt_keys_t}
            d_ogt_sub = {k: d_ogt[k] for k in ogt_keys_t}
            entry["stage1_ogt_vs_default_paired"] = _paired_stats(s1_ogt_sub, s1_def_sub)
            entry["e7_direct_ogt_vs_default_paired"] = _paired_stats(d_ogt_sub, d_def_sub)
        per_task[task] = entry

        pt_csv_rows.append(
            {
                "task": task,
                "s1_default_ok": entry["stage1_default"]["ok"],
                "s1_default_n": entry["stage1_default"]["n"],
                "s1_default_sr": entry["stage1_default"]["sr"],
                "e7d_default_sr": entry["e7_direct_default"]["sr"],
                "s1_ogt_ok": entry["stage1_oracle_gt"]["ok"] if entry["stage1_oracle_gt"] else "",
                "s1_ogt_n": entry["stage1_oracle_gt"]["n"] if entry["stage1_oracle_gt"] else 0,
                "s1_ogt_sr": entry["stage1_oracle_gt"]["sr"] if entry["stage1_oracle_gt"] else "",
                "e7d_ogt_sr": entry["e7_direct_oracle_gt"]["sr"] if entry["e7_direct_oracle_gt"] else "",
                "s1_shifted_ok": entry["stage1_shifted"]["ok"],
                "s1_shifted_n": entry["stage1_shifted"]["n"],
                "s1_shifted_sr": entry["stage1_shifted"]["sr"],
                "e7d_shifted_sr": entry["e7_direct_shifted"]["sr"],
            }
        )

    closed_loop = {
        "init_sha256_verification": sha_checks,
        "stage1_vs_e7_direct_default_500": s1_vs_d_def,
        "stage1_vs_e7_direct_oracle_gt_331": s1_vs_d_ogt,
        "stage1_vs_e7_direct_shifted_500": s1_vs_d_shf,
        "three_group_mean": {
            "stage1_teacher_sr": s1_mean3,
            "e7_direct_sr": d_mean3,
            "delta_pp": 100.0 * (s1_mean3 - d_mean3),
        },
        "stage1_oracle_gt_vs_default_331": s1_ogt_vs_def,
        "stage1_shifted_vs_default_500": s1_shf_vs_def,
        "e7_direct_oracle_gt_vs_default_331": d_ogt_vs_def,
        "e7_direct_shifted_vs_default_500": d_shf_vs_def,
        "per_task": per_task,
    }

    # Write closed-loop CSVs
    with (diag_root / "per_task_closed_loop.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pt_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pt_csv_rows)

    cl_summary_rows = [
        {
            "benchmark": "default_500",
            "n_trials": s1_vs_d_def["n"],
            "stage1_ok": s1_vs_d_def["a_ok"],
            "stage1_sr": s1_vs_d_def["a_sr"],
            "e7_direct_ok": s1_vs_d_def["b_ok"],
            "e7_direct_sr": s1_vs_d_def["b_sr"],
            "s1_minus_e7d_pp": s1_vs_d_def["delta_pp"],
            "wins": s1_vs_d_def["wins"],
            "losses": s1_vs_d_def["losses"],
            "ties": s1_vs_d_def["ties"],
        },
        {
            "benchmark": "oracle_gt_331",
            "n_trials": s1_vs_d_ogt["n"],
            "stage1_ok": s1_vs_d_ogt["a_ok"],
            "stage1_sr": s1_vs_d_ogt["a_sr"],
            "e7_direct_ok": s1_vs_d_ogt["b_ok"],
            "e7_direct_sr": s1_vs_d_ogt["b_sr"],
            "s1_minus_e7d_pp": s1_vs_d_ogt["delta_pp"],
            "wins": s1_vs_d_ogt["wins"],
            "losses": s1_vs_d_ogt["losses"],
            "ties": s1_vs_d_ogt["ties"],
        },
        {
            "benchmark": "shifted_500",
            "n_trials": s1_vs_d_shf["n"],
            "stage1_ok": s1_vs_d_shf["a_ok"],
            "stage1_sr": s1_vs_d_shf["a_sr"],
            "e7_direct_ok": s1_vs_d_shf["b_ok"],
            "e7_direct_sr": s1_vs_d_shf["b_sr"],
            "s1_minus_e7d_pp": s1_vs_d_shf["delta_pp"],
            "wins": s1_vs_d_shf["wins"],
            "losses": s1_vs_d_shf["losses"],
            "ties": s1_vs_d_shf["ties"],
        },
    ]
    with (diag_root / "closed_loop_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cl_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cl_summary_rows)

    # Generate diagnostic figures
    _plot_offline_decomposition(offline, diag_root / "offline_decomposition.png")
    _plot_per_task_overview(offline, closed_loop, tasks, diag_root / "per_task_offline_and_closed_loop.png")
    _plot_reader_and_d3_strat(offline, diag_root / "reader_and_d3_diagnostics.png")

    full_summary = {
        "offline_decomposition": offline,
        "closed_loop": closed_loop,
    }
    atomic_write_json(full_summary, diag_root / "diagnosis_summary.json")
    print(json.dumps(closed_loop, indent=2), flush=True)
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-root", type=Path, default=DIAG_ROOT)
    parser.add_argument("--direct-root", type=Path, default=DIRECT_ROOT)
    args = parser.parse_args()
    build_diagnosis_summary(diag_root=args.diag_root, direct_root=args.direct_root)


if __name__ == "__main__":
    main()
