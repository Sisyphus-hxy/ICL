"""D1 Diagnostic Evaluation for E9 Stage 1 Local Behavior Encoder (CLIP).

Reuses the EXACT D1 diagnostic protocol from `e7/diagnose_behavior_space.py`:
  - Default (100 demos / 2,025 anchors across 10 LIBERO-Gen Goal Chain tasks)
  - Shifted (100 demos / 2,042 anchors across 10 tasks)
  - Oracle-GT (331 demos / 7,053 anchors across 9 shared tasks)
  - Stride-16 Behavior Memory anchors `t_k = 0, 16, 32, ... < T`
  - Frozen CLIP ViT-B/16 (`BPPVisualEncoder`, fp16 autocast)
  - Local clip encoding `b_k = E_phi(V_{t_k:t_k+16}, clip_pad_mask)` (bf16 autocast)
    strictly over the provided Demo's own video frames
  - Identical deterministic per-(demo, anchor) DDIM initial noise (`base_seed=12345`)
  - Identical tail-window valid-step masking (`valid_step = t_k + s < T`), primary
    9-task equal-weight macro average, micro average, oracle-weighted average,
    sensitivity checks, and per-task comparison against original E6 Stage 1 D1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from e5 import ACTION_DIM_10D, ACTION_HORIZON, BPP_CKPT, EXEC_HORIZON
from e5.actions import action_10d_to_7d, action_7d_to_10d
from e5.policy_server import get_chain_tasks
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder
from e7.analyze_behavior_space_diagnosis import SHORT_TASK_NAMES
from e7.diagnose_behavior_space import (
    _collect_demo_specs,
    _compute_per_anchor_action_metrics,
    _stable_seed,
)
from e7.model import compute_behavior_anchor_steps

from . import BEHAVIOR_STRIDE, DEFAULT_STAGE1_LOCAL_DIR, ROOT
from .stage1_local_model import load_frozen_e9_stage1_teacher

OLD_D1_SUMMARY_PATH = (
    ROOT / "e7/outputs/behavior_space_diagnosis/offline/offline_summary.json"
)


def _aggregate_d1_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Exact D1 aggregation matching `e7/diagnose_behavior_space.py::_aggregate_rows`."""
    n = len(rows)
    if n == 0:
        return {"num_anchors": 0, "num_demos": 0}

    demo_keys = {(r["demo_type"], r["task"], r["demo_label"]) for r in rows}
    n_tail = sum(1 for r in rows if r["is_tail_window"])
    tot_v16 = float(sum(r["n_valid_16"] for r in rows))
    tot_v12 = float(sum(r["n_valid_12"] for r in rows))

    mse10 = float(sum(r["mse10_valid_sum"] for r in rows) / max(tot_v16, 1.0))
    pos_mse = float(sum(r["pos_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0))
    rot_mse = float(sum(r["rot6d_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0))
    grip_mse = float(sum(r["grip_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0))
    mae7 = float(sum(r["mae7_valid_sum"] for r in rows) / max(tot_v16, 1.0))
    exec12_mae7 = float(sum(r["exec12_mae7_valid_sum"] for r in rows) / max(tot_v12, 1.0))
    grip_acc16 = float(sum(r["grip_acc16_valid_sum"] for r in rows) / max(tot_v16, 1.0))
    grip_acc12 = float(sum(r["grip_acc12_valid_sum"] for r in rows) / max(tot_v12, 1.0))

    sample_mse10 = float(np.mean([r["mse10_masked"] for r in rows]))
    unmasked_mse10 = float(np.mean([r["mse10_unmasked"] for r in rows]))
    non_tail_rows = [r for r in rows if not r["is_tail_window"]]
    tail_rows = [r for r in rows if r["is_tail_window"]]
    non_tail_mse10 = (
        float(np.mean([r["mse10_masked"] for r in non_tail_rows]))
        if non_tail_rows
        else mse10
    )
    tail_only_mse10 = (
        float(np.mean([r["mse10_masked"] for r in tail_rows])) if tail_rows else mse10
    )
    lat_rms = float(math.sqrt(np.mean([r["latent_sq_mean"] for r in rows])))

    return {
        "num_demos": len(demo_keys),
        "num_anchors": n,
        "num_tail_anchors": n_tail,
        "tail_anchor_frac": n_tail / n,
        "total_valid_steps_16": int(round(tot_v16)),
        "total_valid_steps_12": int(round(tot_v12)),
        "ddim_10d_mse": mse10,
        "ddim_pos_mse": pos_mse,
        "ddim_rot6d_mse": rot_mse,
        "ddim_grip_mse": grip_mse,
        "ddim_7d_mae": mae7,
        "exec12_7d_mae": exec12_mae7,
        "grip_acc": grip_acc16,
        "grip_acc_exec12": grip_acc12,
        "latent_rms": lat_rms,
        "sensitivity_checks": {
            "sample_masked_10d_mse": sample_mse10,
            "unmasked_repeat_last_10d_mse": unmasked_mse10,
            "non_tail_only_10d_mse": non_tail_mse10,
            "tail_only_10d_mse": tail_only_mse10,
        },
    }


def _macro_average_d1(
    per_task_aggs: Dict[str, Dict[str, Any]],
    task_list: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    active = [t for t in task_list if t in per_task_aggs and per_task_aggs[t]["num_anchors"] > 0]
    if not active:
        return {"tasks": [], "num_tasks": 0}
    if weights is None:
        w = {t: 1.0 / len(active) for t in active}
    else:
        tot = sum(weights[t] for t in active)
        w = {t: weights[t] / max(tot, 1e-12) for t in active}

    def _wavg(key: str) -> float:
        return float(sum(w[t] * per_task_aggs[t][key] for t in active))

    return {
        "num_tasks": len(active),
        "tasks": active,
        "weights": w,
        "ddim_10d_mse": _wavg("ddim_10d_mse"),
        "ddim_pos_mse": _wavg("ddim_pos_mse"),
        "ddim_rot6d_mse": _wavg("ddim_rot6d_mse"),
        "ddim_grip_mse": _wavg("ddim_grip_mse"),
        "ddim_7d_mae": _wavg("ddim_7d_mae"),
        "exec12_7d_mae": _wavg("exec12_7d_mae"),
        "grip_acc": _wavg("grip_acc"),
        "grip_acc_exec12": _wavg("grip_acc_exec12"),
        "latent_rms": _wavg("latent_rms"),
    }


def _plot_d1_comparison(
    e9_summary: Dict[str, Any],
    old_summary: Optional[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Plot Old Stage 1 vs E9 Stage 1 Local D1 comparison (overview + per-task)."""
    dts = ["default", "shifted", "oracle_gt"]
    dt_labels = ["Default\n(9 shared)", "Shifted\n(9 shared)", "Oracle-GT\n(9 shared)"]
    e9_m9 = e9_summary["by_demo_type_macro_shared_9"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=160)
    panels = [
        ("ddim_10d_mse", "DDIM 10D MSE (↓)", False),
        ("ddim_7d_mae", "DDIM 7D MAE (↓)", False),
        ("exec12_7d_mae", "Exec-12 7D MAE (↓)", False),
        ("grip_acc", "Gripper Acc % (↑)", True),
    ]
    x = np.arange(len(dts))
    w = 0.34

    for ax, (metric, title, pct) in zip(axes, panels):
        scale = 100.0 if pct else 1.0
        new_vals = [e9_m9[dt][metric] * scale for dt in dts]
        if old_summary is not None:
            old_m9 = old_summary["by_demo_type_macro_shared_9"]
            old_vals = [old_m9[dt]["modes"]["D1_teacher"][metric] * scale for dt in dts]
            b1 = ax.bar(x - w / 2, old_vals, w, label="E6 Stage 1 (Full Demo)", color="#7f8c8d", alpha=0.9)
            b2 = ax.bar(x + w / 2, new_vals, w, label="E9 Stage 1 (Local Clip)", color="#1f77b4", alpha=0.92)
            for bars in (b1, b2):
                for b in bars:
                    v = b.get_height()
                    lbl = f"{v:.2f}" if pct else f"{v:.4f}"
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v,
                        lbl,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
        else:
            b2 = ax.bar(x, new_vals, w, label="E9 Stage 1 (Local Clip)", color="#1f77b4", alpha=0.92)
            for b in b2:
                v = b.get_height()
                lbl = f"{v:.2f}" if pct else f"{v:.4f}"
                ax.text(b.get_x() + b.get_width() / 2, v, lbl, ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(dt_labels, fontsize=8.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if pct:
            ax.set_ylim(90, 100.5)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Stage 1 D1 Self-Reconstruction: Old (Full Demo) vs E9 (Local Behavior Encoder)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "d1_comparison_overview.png", bbox_inches="tight")
    plt.close(fig)

    # Per-task Oracle-GT vs Default 10D MSE comparison
    shared_9 = e9_summary["shared_9_tasks"]
    if shared_9 and old_summary is not None:
        fig, ax = plt.subplots(figsize=(14, 5.2), dpi=160)
        tx = np.arange(len(shared_9))
        tw = 0.2
        old_def = [old_summary["per_task_by_demo_type"]["default"][t]["modes"]["D1_teacher"]["ddim_10d_mse"] for t in shared_9]
        old_ogt = [old_summary["per_task_by_demo_type"]["oracle_gt"][t]["modes"]["D1_teacher"]["ddim_10d_mse"] for t in shared_9]
        new_def = [e9_summary["per_task_by_demo_type"]["default"][t]["ddim_10d_mse"] for t in shared_9]
        new_ogt = [e9_summary["per_task_by_demo_type"]["oracle_gt"][t]["ddim_10d_mse"] for t in shared_9]

        ax.bar(tx - 1.5 * tw, old_def, tw, label="Old Stage 1 — Default", color="#95a5a6")
        ax.bar(tx - 0.5 * tw, old_ogt, tw, label="Old Stage 1 — Oracle-GT", color="#e74c3c", alpha=0.85)
        ax.bar(tx + 0.5 * tw, new_def, tw, label="E9 Local — Default", color="#3498db")
        ax.bar(tx + 1.5 * tw, new_ogt, tw, label="E9 Local — Oracle-GT", color="#27ae60", alpha=0.9)
        ax.set_xticks(tx)
        ax.set_xticklabels([SHORT_TASK_NAMES.get(t, t) for t in shared_9], rotation=24, ha="right", fontsize=8)
        ax.set_ylabel("DDIM 10D Action MSE (↓)")
        ax.set_title("Per-Task D1 10D MSE: Old Stage 1 vs E9 Local (Default & Oracle-GT)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(ncol=2, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(out_dir / "d1_per_task_comparison.png", bbox_inches="tight")
        plt.close(fig)


@torch.inference_mode()
def run_d1_local_evaluation(
    *,
    stage1_local_ckpt: Path = DEFAULT_STAGE1_LOCAL_DIR / "best.pt",
    out_dir: Path = DEFAULT_STAGE1_LOCAL_DIR / "d1_eval",
    old_d1_summary_path: Optional[Path] = OLD_D1_SUMMARY_PATH,
    device: torch.device,
    use_bf16: bool = True,
    base_seed: int = 12345,
    behavior_stride: int = BEHAVIOR_STRIDE,
    demo_types: Tuple[str, ...] = ("default", "shifted", "oracle_gt"),
    max_demos_per_type: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = get_chain_tasks()

    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    ckpt_payload = torch.load(Path(stage1_local_ckpt), map_location="cpu", weights_only=False)
    policy = load_frozen_e9_stage1_teacher(stage1_local_ckpt, device=device)
    trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    trainable_cnt += sum(p.numel() for p in visual.parameters() if p.requires_grad)
    if trainable_cnt != 0:
        raise RuntimeError(f"Expected all modules frozen, found {trainable_cnt} trainable params")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    specs = _collect_demo_specs(tasks, demo_types=demo_types)
    if max_demos_per_type is not None:
        filtered: List[Dict[str, Any]] = []
        counts: Dict[str, int] = defaultdict(int)
        for sp in specs:
            dt = sp["demo_type"]
            if counts[dt] < max_demos_per_type:
                filtered.append(sp)
                counts[dt] += 1
        specs = filtered

    print(
        f"[E9 D1 Eval] Starting D1 evaluation: {len(specs)} demos "
        f"(types={demo_types}, stride={behavior_stride}, bf16={use_bf16}, seed={base_seed}, "
        f"ckpt={stage1_local_ckpt} ep={ckpt_payload.get('epoch')}) on {device}",
        flush=True,
    )

    all_rows: List[Dict[str, Any]] = []
    max_rot_roundtrip_err = 0.0

    for idx, sp in enumerate(specs, start=1):
        demo_type = sp["demo_type"]
        task = sp["task"]
        task_idx = sp["task_idx"]
        demo_label = sp["demo_label"]

        with h5py.File(sp["h5_path"], "r") as f:
            conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
            d = f[sp["group_key"]]
            av_all = d["obs/agentview_rgb"][:]
            eh_all = d["obs/eye_in_hand_rgb"][:]
            actions_7d = d["actions"][:].astype(np.float32)

        T = len(av_all)
        if len(eh_all) != T or len(actions_7d) != T:
            raise ValueError(f"Length mismatch in {sp['h5_path']}:{sp['group_key']}")

        # Encode all T frames of the provided Demo video with frozen CLIP (fp16 autocast)
        demo_all_feats = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if device.type == "cuda" else None,
        )  # (T, 2, 768) float32 on GPU

        # Extract local behavior latents b_k = E_phi(V_{t_k:t_k+16}) at exact stride-16 anchors
        with amp_ctx:
            _, b_mem, anchors = policy.extract_behavior_memory_from_demo_features(
                demo_all_feats, stride=behavior_stride
            )
            M = len(anchors)

            # Exact same per-(demo, anchor) deterministic DDIM noise as E7 D1 protocol
            noises = []
            for t_k in anchors:
                s_k = _stable_seed(
                    f"{demo_type}/{task}/{demo_label}/t{int(t_k)}", base_seed=base_seed
                )
                gen = torch.Generator(device="cpu").manual_seed(s_k)
                noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
            init_noise = torch.stack(noises, dim=0).to(device=device, dtype=torch.float32)

            pred_10d = policy.sample_from_z_self(b_mem, init_noise=init_noise)

        # Ground truth 16-step action chunks + tail-window mask (exact E7 D1 match)
        actions_10d_np = action_7d_to_10d(actions_7d)
        rt_7d_np = action_10d_to_7d(actions_10d_np)
        rt_err = float(np.max(np.abs(rt_7d_np - actions_7d)))
        if rt_err > max_rot_roundtrip_err:
            max_rot_roundtrip_err = rt_err

        step_grid = anchors[:, None] + np.arange(ACTION_HORIZON, dtype=np.int32)[None, :]
        valid_16_np = step_grid < T
        clipped_grid = np.minimum(step_grid, T - 1)
        gt_10d = torch.from_numpy(actions_10d_np[clipped_grid]).to(
            device=device, dtype=torch.float32
        )
        valid_16 = torch.from_numpy(valid_16_np).to(device=device, dtype=torch.bool)

        am = _compute_per_anchor_action_metrics(pred_10d, gt_10d, valid_16)
        b_sq = b_mem.float().square().mean(dim=(1, 2)).cpu().numpy()

        for k in range(M):
            all_rows.append(
                {
                    "demo_type": demo_type,
                    "task_idx": task_idx,
                    "task": task,
                    "demo_label": demo_label,
                    "demo_len": T,
                    "mem_len": M,
                    "anchor_idx": k,
                    "anchor_step": int(anchors[k]),
                    "is_tail_window": bool(not valid_16_np[k].all()),
                    "n_valid_16": int(am["n_valid_16"][k]),
                    "n_valid_12": int(am["n_valid_12"][k]),
                    "mse10_masked": float(am["mse10_masked"][k]),
                    "mae7_masked": float(am["mae7_masked"][k]),
                    "exec12_mae7_masked": float(am["exec12_mae7_masked"][k]),
                    "grip_acc16_masked": float(am["grip_acc16_masked"][k]),
                    "grip_acc12_masked": float(am["grip_acc12_masked"][k]),
                    "mse10_valid_sum": float(am["mse10_valid_sum"][k]),
                    "pos_mse_valid_sum": float(am["pos_mse_valid_sum"][k]),
                    "rot6d_mse_valid_sum": float(am["rot6d_mse_valid_sum"][k]),
                    "grip_mse_valid_sum": float(am["grip_mse_valid_sum"][k]),
                    "mae7_valid_sum": float(am["mae7_valid_sum"][k]),
                    "exec12_mae7_valid_sum": float(am["exec12_mae7_valid_sum"][k]),
                    "grip_acc16_valid_sum": float(am["grip_acc16_valid_sum"][k]),
                    "grip_acc12_valid_sum": float(am["grip_acc12_valid_sum"][k]),
                    "mse10_unmasked": float(am["mse10_unmasked"][k]),
                    "latent_sq_mean": float(b_sq[k]),
                }
            )

        if idx % 50 == 0 or idx == len(specs):
            dt = time.monotonic() - t0
            print(
                f"[E9 D1 Eval] {idx}/{len(specs)} demos ({len(all_rows):,} anchors) "
                f"in {dt:.1f}s ({idx / max(dt, 1e-6):.1f} demos/s)",
                flush=True,
            )

    # Aggregate results
    by_demo_type: Dict[str, Any] = {}
    per_task_by_type: Dict[str, Dict[str, Any]] = {}
    for dt in demo_types:
        dt_rows = [r for r in all_rows if r["demo_type"] == dt]
        by_demo_type[dt] = _aggregate_d1_rows(dt_rows)
        per_task_by_type[dt] = {}
        for task in tasks:
            t_rows = [r for r in dt_rows if r["task"] == task]
            if t_rows:
                per_task_by_type[dt][task] = _aggregate_d1_rows(t_rows)

    shared_9_tasks = [t for t in tasks if t in per_task_by_type.get("oracle_gt", {})]
    ogt_demo_weights = {
        t: float(per_task_by_type["oracle_gt"][t]["num_demos"]) for t in shared_9_tasks
    }
    ogt_anchor_weights = {
        t: float(per_task_by_type["oracle_gt"][t]["num_anchors"]) for t in shared_9_tasks
    }

    macro_all_10: Dict[str, Any] = {}
    macro_shared_9: Dict[str, Any] = {}
    oracle_demo_weighted_9: Dict[str, Any] = {}
    oracle_anchor_weighted_9: Dict[str, Any] = {}
    for dt in demo_types:
        macro_all_10[dt] = _macro_average_d1(per_task_by_type[dt], tasks)
        if shared_9_tasks:
            macro_shared_9[dt] = _macro_average_d1(per_task_by_type[dt], shared_9_tasks)
            oracle_demo_weighted_9[dt] = _macro_average_d1(
                per_task_by_type[dt], shared_9_tasks, weights=ogt_demo_weights
            )
            oracle_anchor_weighted_9[dt] = _macro_average_d1(
                per_task_by_type[dt], shared_9_tasks, weights=ogt_anchor_weights
            )

    # Load old Stage 1 D1 summary for direct comparison
    old_summary = (
        json.loads(Path(old_d1_summary_path).read_text())
        if (old_d1_summary_path is not None and Path(old_d1_summary_path).is_file())
        else None
    )

    comparison_vs_old: Dict[str, Any] = {}
    if old_summary is not None and shared_9_tasks:
        old_m9 = old_summary["by_demo_type_macro_shared_9"]
        for dt in demo_types:
            if dt in macro_shared_9 and dt in old_m9:
                cur = macro_shared_9[dt]
                ref = old_m9[dt]["modes"]["D1_teacher"]
                comparison_vs_old[dt] = {
                    "old_10d_mse": ref["ddim_10d_mse"],
                    "new_10d_mse": cur["ddim_10d_mse"],
                    "delta_10d_mse_abs": cur["ddim_10d_mse"] - ref["ddim_10d_mse"],
                    "delta_10d_mse_rel_pct": 100.0
                    * (cur["ddim_10d_mse"] - ref["ddim_10d_mse"])
                    / max(ref["ddim_10d_mse"], 1e-12),
                    "old_7d_mae": ref["ddim_7d_mae"],
                    "new_7d_mae": cur["ddim_7d_mae"],
                    "delta_7d_mae_rel_pct": 100.0
                    * (cur["ddim_7d_mae"] - ref["ddim_7d_mae"])
                    / max(ref["ddim_7d_mae"], 1e-12),
                    "old_exec12_mae": ref["exec12_7d_mae"],
                    "new_exec12_mae": cur["exec12_7d_mae"],
                    "delta_exec12_mae_rel_pct": 100.0
                    * (cur["exec12_7d_mae"] - ref["exec12_7d_mae"])
                    / max(ref["exec12_7d_mae"], 1e-12),
                    "old_grip_acc": ref["grip_acc"],
                    "new_grip_acc": cur["grip_acc"],
                    "delta_grip_acc_pp": 100.0 * (cur["grip_acc"] - ref["grip_acc"]),
                }
        if "default" in macro_shared_9 and "oracle_gt" in macro_shared_9:
            old_ratio = old_m9["oracle_gt"]["modes"]["D1_teacher"]["ddim_10d_mse"] / max(
                old_m9["default"]["modes"]["D1_teacher"]["ddim_10d_mse"], 1e-12
            )
            new_ratio = macro_shared_9["oracle_gt"]["ddim_10d_mse"] / max(
                macro_shared_9["default"]["ddim_10d_mse"], 1e-12
            )
            comparison_vs_old["gt_over_default_mse_ratio"] = {
                "old": old_ratio,
                "new": new_ratio,
            }

    # Save CSVs
    if all_rows:
        with (out_dir / "per_anchor_records.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summary_rows: List[Dict[str, Any]] = []
    for agg_kind, container in (
        ("micro_all", by_demo_type),
        ("macro_all_10", macro_all_10),
        ("macro_shared_9", macro_shared_9),
        ("oracle_demo_weighted_9", oracle_demo_weighted_9),
    ):
        for dt in demo_types:
            blk = container.get(dt)
            if not blk or "ddim_10d_mse" not in blk:
                continue
            summary_rows.append(
                {
                    "aggregation": agg_kind,
                    "demo_type": dt,
                    "ddim_10d_mse": blk["ddim_10d_mse"],
                    "ddim_pos_mse": blk["ddim_pos_mse"],
                    "ddim_rot6d_mse": blk["ddim_rot6d_mse"],
                    "ddim_grip_mse": blk["ddim_grip_mse"],
                    "ddim_7d_mae": blk["ddim_7d_mae"],
                    "exec12_7d_mae": blk["exec12_7d_mae"],
                    "grip_acc": blk["grip_acc"],
                    "grip_acc_exec12": blk["grip_acc_exec12"],
                    "latent_rms": blk["latent_rms"],
                }
            )
    if summary_rows:
        with (out_dir / "summary_by_demo_type.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    pt_rows: List[Dict[str, Any]] = []
    for task in tasks:
        for dt in demo_types:
            if task not in per_task_by_type[dt]:
                continue
            blk = per_task_by_type[dt][task]
            old_mse = None
            if (
                old_summary is not None
                and task in old_summary["per_task_by_demo_type"].get(dt, {})
            ):
                old_mse = old_summary["per_task_by_demo_type"][dt][task]["modes"][
                    "D1_teacher"
                ]["ddim_10d_mse"]
            pt_rows.append(
                {
                    "task": task,
                    "demo_type": dt,
                    "num_demos": blk["num_demos"],
                    "num_anchors": blk["num_anchors"],
                    "ddim_10d_mse": blk["ddim_10d_mse"],
                    "old_stage1_10d_mse": old_mse,
                    "ddim_7d_mae": blk["ddim_7d_mae"],
                    "exec12_7d_mae": blk["exec12_7d_mae"],
                    "grip_acc": blk["grip_acc"],
                    "grip_acc_exec12": blk["grip_acc_exec12"],
                    "latent_rms": blk["latent_rms"],
                }
            )
    if pt_rows:
        with (out_dir / "per_task_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pt_rows)

    elapsed = round(time.monotonic() - t0, 2)
    summary: Dict[str, Any] = {
        "experiment": "e9_stage1_local_d1_evaluation",
        "checkpoints": {
            "clip_ckpt": str(BPP_CKPT),
            "stage1_local_ckpt": str(Path(stage1_local_ckpt).resolve()),
            "best_epoch": ckpt_payload.get("epoch"),
            "best_val_self_diff_loss": ckpt_payload.get("best_val_self_diff_loss"),
            "old_stage1_d1_summary": str(old_d1_summary_path) if old_summary else None,
        },
        "protocol": {
            "behavior_stride": int(behavior_stride),
            "action_horizon": int(ACTION_HORIZON),
            "exec_horizon": int(EXEC_HORIZON),
            "base_seed": int(base_seed),
            "use_bf16": bool(use_bf16 and device.type == "cuda"),
            "tail_window_mask": "valid_step = (t_k + s < T); primary metrics step-weighted over valid steps",
            "max_rot_roundtrip_err": max_rot_roundtrip_err,
            "all_modules_frozen": True,
        },
        "elapsed_seconds": elapsed,
        "shared_9_tasks": shared_9_tasks,
        "by_demo_type_micro": by_demo_type,
        "by_demo_type_macro_all_10": macro_all_10,
        "by_demo_type_macro_shared_9": macro_shared_9,
        "by_demo_type_oracle_demo_weighted_9": oracle_demo_weighted_9,
        "by_demo_type_oracle_anchor_weighted_9": oracle_anchor_weighted_9,
        "per_task_by_demo_type": per_task_by_type,
        "comparison_vs_old_stage1_macro_shared_9": comparison_vs_old,
    }
    atomic_write_json(summary, out_dir / "d1_summary.json")

    if "default" in macro_shared_9 and "shifted" in macro_shared_9 and "oracle_gt" in macro_shared_9:
        _plot_d1_comparison(summary, old_summary, out_dir)

    print("\n=== E9 Stage 1 Local D1 Summary (9-Task Equal-Weight Macro Average) ===", flush=True)
    for dt in demo_types:
        blk = macro_shared_9.get(dt, by_demo_type[dt])
        cmp_str = ""
        if dt in comparison_vs_old:
            c = comparison_vs_old[dt]
            cmp_str = f" | Old={c['old_10d_mse']:.6f} ({c['delta_10d_mse_rel_pct']:+.1f}%)"
        print(
            f"  {dt:10s}: 10D_MSE={blk['ddim_10d_mse']:.6f}{cmp_str} | "
            f"7D_MAE={blk['ddim_7d_mae']:.5f} | Exec12={blk['exec12_7d_mae']:.5f} | "
            f"Grip={100*blk['grip_acc']:.2f}% | b_rms={blk['latent_rms']:.3f}",
            flush=True,
        )
    if "gt_over_default_mse_ratio" in comparison_vs_old:
        r = comparison_vs_old["gt_over_default_mse_ratio"]
        print(
            f"  GT / Default 10D MSE ratio: Old={r['old']:.2f}x -> E9 Local={r['new']:.2f}x",
            flush=True,
        )
    print(f"[E9 D1 Eval DONE] Saved to {out_dir} in {elapsed}s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_DIR / "best.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_STAGE1_LOCAL_DIR / "d1_eval"
    )
    parser.add_argument("--old-d1-summary", type=Path, default=OLD_D1_SUMMARY_PATH)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--stride", type=int, default=BEHAVIOR_STRIDE)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--demo-types",
        nargs="+",
        default=["default", "shifted", "oracle_gt"],
        choices=("default", "shifted", "oracle_gt"),
    )
    parser.add_argument("--max-demos-per-type", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_d1_local_evaluation(
        stage1_local_ckpt=args.ckpt,
        out_dir=args.output,
        old_d1_summary_path=args.old_d1_summary,
        device=device,
        use_bf16=not args.no_bf16,
        base_seed=args.seed,
        behavior_stride=args.stride,
        demo_types=tuple(args.demo_types),
        max_demos_per_type=args.max_demos_per_type,
    )


if __name__ == "__main__":
    main()
