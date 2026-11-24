"""Stage 1 Behavior Space Training-Free Offline Decomposition Diagnosis (D1 / D2 / D3 / D4).

Uses frozen CLIP, frozen E6 Stage 1 Teacher, and frozen E7-Direct Best checkpoint
(zero training, zero architecture/data modifications) to decompose offline action &
latent behavior on Default (100 demos), Shifted (100 demos), and Oracle-GT (331 demos):

  - D1 (Teacher Only)   : exact self latent b_k = R_T(M_D, h_k^D) -> Frozen UNet
  - D2 (Retrieval Only) : Behavior Memory -> Reader -> z_c         -> Frozen UNet
  - D3 (Transfer Only)  : exact self latent b_k -> Direct Transfer -> Frozen UNet
  - D4 (Full Model)     : Behavior Memory -> Reader -> Direct Transfer -> Frozen UNet

All evaluated at actual stride-16 Behavior Memory anchor positions t_k = 0, 16, 32, ... < T
using each Demo's own observation h_k^D as Query and comparing against that Demo's own GT
action chunk under unified action normalization, history observations, deterministic per-sample
DDIM noise seeds, and tail-window masking (valid steps t_k + s < T).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from e5 import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    BPP_CKPT,
    DEMO_STRIDE,
    EMBED_DIM,
    EXEC_HORIZON,
    ROOT,
)
from e5.actions import action_10d_to_7d, action_7d_to_10d
from e5.policy_server import CHAIN_HDF5_DIR, DEFAULT_PROMPT_SLOTS, get_chain_tasks
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder

from . import BEHAVIOR_STRIDE, DEFAULT_STAGE1_CKPT, OUTPUT_ROOT
from .model import compute_attention_stats, compute_behavior_anchor_steps, load_e7_policy
from .policy_server import ORACLE_GT_EXPERTS_DIR, SHIFT_OFFSET

DEFAULT_E7_DIRECT_CKPT = (
    OUTPUT_ROOT / "direct_latent/train/e7_direct_latent/best.pt"
)
DEFAULT_DIAG_OUT_DIR = OUTPUT_ROOT / "behavior_space_diagnosis/offline"
MODES = ("D1_teacher", "D2_retrieval", "D3_transfer", "D4_full")


def _stable_seed(key: str, base_seed: int = 12345) -> int:
    h = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False) & 0x7FFFFFFF


def _collect_demo_specs(
    tasks: List[str],
    demo_types: Tuple[str, ...] = ("default", "shifted", "oracle_gt"),
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for demo_type in demo_types:
        if demo_type == "default":
            for ti, task in enumerate(tasks):
                h5_path = CHAIN_HDF5_DIR / f"{task}_demo.hdf5"
                for di in range(DEFAULT_PROMPT_SLOTS):
                    specs.append(
                        {
                            "demo_type": "default",
                            "task_idx": ti,
                            "task": task,
                            "demo_label": f"demo_{di}",
                            "h5_path": h5_path,
                            "group_key": f"data/demo_{di}",
                        }
                    )
        elif demo_type == "shifted":
            for ti, task in enumerate(tasks):
                h5_path = CHAIN_HDF5_DIR / f"{task}_demo.hdf5"
                for di in range(DEFAULT_PROMPT_SLOTS):
                    src_di = SHIFT_OFFSET + di
                    specs.append(
                        {
                            "demo_type": "shifted",
                            "task_idx": ti,
                            "task": task,
                            "demo_label": f"demo_{src_di}",
                            "h5_path": h5_path,
                            "group_key": f"data/demo_{src_di}",
                        }
                    )
        elif demo_type == "oracle_gt":
            for ti, task in enumerate(tasks):
                tdir = ORACLE_GT_EXPERTS_DIR / task
                if not tdir.is_dir():
                    continue
                for qi in range(50):
                    h5_path = tdir / f"query_{qi:02d}" / "demo.hdf5"
                    if h5_path.is_file():
                        specs.append(
                            {
                                "demo_type": "oracle_gt",
                                "task_idx": ti,
                                "task": task,
                                "demo_label": f"query_{qi:02d}",
                                "h5_path": h5_path,
                                "group_key": "data/demo_0",
                            }
                        )
        else:
            raise ValueError(f"Unsupported demo_type: {demo_type}")
    return specs


def _compute_per_anchor_action_metrics(
    pred_10d: torch.Tensor,
    gt_10d: torch.Tensor,
    valid_16: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Compute per-anchor action metrics under:
      1. Primary tail-window mask (valid_16[k, s] = (t_k + s < T), valid_12 = valid_16[:, :12])
      2. Unmasked repeat-last padding (matching E6/E7 training evaluation convention)
    """
    pred_10d = pred_10d.float()
    gt_10d = gt_10d.float()
    m16 = valid_16.float()  # (M, 16)
    m12 = valid_16[:, :EXEC_HORIZON].float()  # (M, 12)
    n16 = m16.sum(dim=1).clamp_min(1.0)  # (M,)
    n12 = m12.sum(dim=1).clamp_min(1.0)  # (M,)

    diff10 = pred_10d - gt_10d  # (M, 16, 10)
    sq10_step = diff10.square().mean(dim=-1)  # (M, 16)
    abs10_step = diff10.abs().mean(dim=-1)  # (M, 16)
    pos_sq_step = diff10[..., :3].square().mean(dim=-1)
    rot_sq_step = diff10[..., 3:9].square().mean(dim=-1)
    grip_sq_step = diff10[..., 9:10].square().mean(dim=-1)

    pred_7d = action_10d_to_7d(pred_10d)
    gt_7d = action_10d_to_7d(gt_10d)
    diff7 = pred_7d - gt_7d  # (M, 16, 7)
    abs7_step = diff7.abs().mean(dim=-1)  # (M, 16)
    sq7_step = diff7.square().mean(dim=-1)  # (M, 16)

    grip_ok_step = ((pred_7d[..., 6] >= 0.0) == (gt_7d[..., 6] >= 0.0)).float()  # (M, 16)

    out = {
        "n_valid_16": n16.cpu().numpy(),
        "n_valid_12": n12.cpu().numpy(),
        # Per-sample masked means (average over valid steps in each window)
        "mse10_masked": ((sq10_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "mae10_masked": ((abs10_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "pos_mse_masked": ((pos_sq_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "rot6d_mse_masked": ((rot_sq_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "grip_mse_masked": ((grip_sq_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "mae7_masked": ((abs7_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "mse7_masked": ((sq7_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "exec12_mae7_masked": ((abs7_step[:, :EXEC_HORIZON] * m12).sum(dim=1) / n12).cpu().numpy(),
        "exec12_mse7_masked": ((sq7_step[:, :EXEC_HORIZON] * m12).sum(dim=1) / n12).cpu().numpy(),
        "grip_acc16_masked": ((grip_ok_step * m16).sum(dim=1) / n16).cpu().numpy(),
        "grip_acc12_masked": ((grip_ok_step[:, :EXEC_HORIZON] * m12).sum(dim=1) / n12).cpu().numpy(),
        # Step sums for exact step-weighted masked aggregation across samples
        "mse10_valid_sum": (sq10_step * m16).sum(dim=1).cpu().numpy(),
        "pos_mse_valid_sum": (pos_sq_step * m16).sum(dim=1).cpu().numpy(),
        "rot6d_mse_valid_sum": (rot_sq_step * m16).sum(dim=1).cpu().numpy(),
        "grip_mse_valid_sum": (grip_sq_step * m16).sum(dim=1).cpu().numpy(),
        "mae7_valid_sum": (abs7_step * m16).sum(dim=1).cpu().numpy(),
        "exec12_mae7_valid_sum": (abs7_step[:, :EXEC_HORIZON] * m12).sum(dim=1).cpu().numpy(),
        "grip_acc16_valid_sum": (grip_ok_step * m16).sum(dim=1).cpu().numpy(),
        "grip_acc12_valid_sum": (grip_ok_step[:, :EXEC_HORIZON] * m12).sum(dim=1).cpu().numpy(),
        # Unmasked repeat-last (cross-check against E6/E7 training evaluation convention)
        "mse10_unmasked": sq10_step.mean(dim=1).cpu().numpy(),
        "mae7_unmasked": abs7_step.mean(dim=1).cpu().numpy(),
        "exec12_mae7_unmasked": abs7_step[:, :EXEC_HORIZON].mean(dim=1).cpu().numpy(),
        "grip_acc16_unmasked": grip_ok_step.mean(dim=1).cpu().numpy(),
    }
    return out


def _aggregate_rows(rows: List[Dict[str, Any]], s_z: float) -> Dict[str, Any]:
    """Aggregate per-anchor diagnostic rows into step-masked, sample-masked, latent, and reader stats."""
    n = len(rows)
    if n == 0:
        return {"num_anchors": 0, "num_demos": 0}

    demo_keys = {(r["demo_type"], r["task"], r["demo_label"]) for r in rows}
    n_tail = sum(1 for r in rows if r["is_tail_window"])
    tot_v16 = float(sum(r["n_valid_16"] for r in rows))
    tot_v12 = float(sum(r["n_valid_12"] for r in rows))
    s_z2 = s_z * s_z

    mode_stats: Dict[str, Any] = {}
    for m in MODES:
        # Primary: step-weighted masked average over all valid (anchor, step) pairs
        mse10 = float(sum(r[f"{m}_mse10_valid_sum"] for r in rows) / max(tot_v16, 1.0))
        pos_mse = float(sum(r[f"{m}_pos_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0))
        rot_mse = float(sum(r[f"{m}_rot6d_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0))
        grip_mse = float(sum(r[f"{m}_grip_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0))
        mae7 = float(sum(r[f"{m}_mae7_valid_sum"] for r in rows) / max(tot_v16, 1.0))
        exec12_mae7 = float(
            sum(r[f"{m}_exec12_mae7_valid_sum"] for r in rows) / max(tot_v12, 1.0)
        )
        grip_acc16 = float(
            sum(r[f"{m}_grip_acc16_valid_sum"] for r in rows) / max(tot_v16, 1.0)
        )
        grip_acc12 = float(
            sum(r[f"{m}_grip_acc12_valid_sum"] for r in rows) / max(tot_v12, 1.0)
        )

        # Secondary sensitivity checks: per-anchor masked mean, unmasked repeat-last, non-tail only
        sample_mse10 = float(np.mean([r[f"{m}_mse10_masked"] for r in rows]))
        unmasked_mse10 = float(np.mean([r[f"{m}_mse10_unmasked"] for r in rows]))
        non_tail_rows = [r for r in rows if not r["is_tail_window"]]
        non_tail_mse10 = (
            float(np.mean([r[f"{m}_mse10_masked"] for r in non_tail_rows]))
            if non_tail_rows
            else mse10
        )

        lat_mse = float(np.mean([r[f"{m}_latent_mse_vs_self"] for r in rows]))
        lat_cos = float(np.mean([r[f"{m}_latent_cos_vs_self"] for r in rows]))
        lat_rms = float(math.sqrt(np.mean([r[f"{m}_latent_sq_mean"] for r in rows])))

        mode_stats[m] = {
            "ddim_10d_mse": mse10,
            "ddim_pos_mse": pos_mse,
            "ddim_rot6d_mse": rot_mse,
            "ddim_grip_mse": grip_mse,
            "ddim_7d_mae": mae7,
            "exec12_7d_mae": exec12_mae7,
            "grip_acc": grip_acc16,
            "grip_acc_exec12": grip_acc12,
            "latent_mse_vs_self": lat_mse,
            "latent_lz_vs_self": lat_mse / s_z2,
            "latent_cos_vs_self": lat_cos,
            "latent_rms": lat_rms,
            "sensitivity_checks": {
                "sample_masked_10d_mse": sample_mse10,
                "unmasked_repeat_last_10d_mse": unmasked_mse10,
                "non_tail_only_10d_mse": non_tail_mse10,
            },
        }

    # Deltas relative to D1
    d1 = mode_stats["D1_teacher"]
    deltas_vs_d1: Dict[str, Any] = {}
    for m in MODES[1:]:
        cur = mode_stats[m]
        deltas_vs_d1[m] = {
            "ddim_10d_mse_abs": cur["ddim_10d_mse"] - d1["ddim_10d_mse"],
            "ddim_10d_mse_rel_pct": 100.0
            * (cur["ddim_10d_mse"] - d1["ddim_10d_mse"])
            / max(d1["ddim_10d_mse"], 1e-12),
            "ddim_7d_mae_abs": cur["ddim_7d_mae"] - d1["ddim_7d_mae"],
            "ddim_7d_mae_rel_pct": 100.0
            * (cur["ddim_7d_mae"] - d1["ddim_7d_mae"])
            / max(d1["ddim_7d_mae"], 1e-12),
            "exec12_7d_mae_abs": cur["exec12_7d_mae"] - d1["exec12_7d_mae"],
            "exec12_7d_mae_rel_pct": 100.0
            * (cur["exec12_7d_mae"] - d1["exec12_7d_mae"])
            / max(d1["exec12_7d_mae"], 1e-12),
            "grip_acc_pp": 100.0 * (cur["grip_acc"] - d1["grip_acc"]),
            "grip_acc_exec12_pp": 100.0 * (cur["grip_acc_exec12"] - d1["grip_acc_exec12"]),
        }

    # Reader retrieval diagnostics
    reader_stats = {
        "mem_len_mean": float(np.mean([r["mem_len"] for r in rows])),
        "max_weight_mean": float(np.mean([r["reader_max_weight"] for r in rows])),
        "max_weight_median": float(np.median([r["reader_max_weight"] for r in rows])),
        "self_weight_mean": float(np.mean([r["reader_self_weight"] for r in rows])),
        "self_weight_median": float(np.median([r["reader_self_weight"] for r in rows])),
        "exact_hit_rate": float(np.mean([r["reader_exact_hit"] for r in rows])),
        "within1_hit_rate": float(np.mean([r["reader_within1_hit"] for r in rows])),
        "pos_err_argmax_slots_mean": float(
            np.mean([r["reader_pos_err_argmax_slots"] for r in rows])
        ),
        "pos_err_argmax_steps_mean": float(
            np.mean([r["reader_pos_err_argmax_steps"] for r in rows])
        ),
        "pos_err_argmax_norm_mean": float(
            np.mean([r["reader_pos_err_argmax_norm"] for r in rows])
        ),
        "signed_err_argmax_slots_mean": float(
            np.mean([r["reader_signed_err_argmax_slots"] for r in rows])
        ),
        "pos_err_expected_slots_mean": float(
            np.mean([r["reader_pos_err_expected_slots"] for r in rows])
        ),
        "pos_err_expected_steps_mean": float(
            np.mean([r["reader_pos_err_expected_steps"] for r in rows])
        ),
        "expected_abs_dist_slots_mean": float(
            np.mean([r["reader_expected_abs_dist_slots"] for r in rows])
        ),
        "expected_abs_dist_steps_mean": float(
            np.mean([r["reader_expected_abs_dist_steps"] for r in rows])
        ),
        "attn_entropy_mean": float(np.mean([r["reader_entropy"] for r in rows])),
        "attn_norm_entropy_mean": float(np.mean([r["reader_norm_entropy"] for r in rows])),
        "attn_eff_pos_exp_mean": float(np.mean([r["reader_eff_pos_exp"] for r in rows])),
    }

    # Extra cross-stage latent comparison: D4 vs D3 (how much retrieval error persists through Transfer)
    d4_vs_d3 = {
        "latent_mse_d4_vs_d3": float(np.mean([r["latent_mse_d4_vs_d3"] for r in rows])),
        "latent_cos_d4_vs_d3": float(np.mean([r["latent_cos_d4_vs_d3"] for r in rows])),
    }

    # D3 input-distribution stratification: split by self_weight >= 0.9 vs < 0.9 and exact_hit
    strat: Dict[str, Any] = {}
    for label, pred in (
        ("w_self_ge_0.9", lambda r: r["reader_self_weight"] >= 0.9),
        ("w_self_0.5_to_0.9", lambda r: 0.5 <= r["reader_self_weight"] < 0.9),
        ("w_self_lt_0.5", lambda r: r["reader_self_weight"] < 0.5),
        ("exact_hit_true", lambda r: bool(r["reader_exact_hit"])),
        ("exact_hit_false", lambda r: not bool(r["reader_exact_hit"])),
    ):
        sub = [r for r in rows if pred(r)]
        if not sub:
            strat[label] = {"num_anchors": 0, "frac": 0.0}
            continue
        sv16 = float(sum(r["n_valid_16"] for r in sub))
        sv12 = float(sum(r["n_valid_12"] for r in sub))
        strat[label] = {
            "num_anchors": len(sub),
            "frac": len(sub) / n,
            "self_weight_mean": float(np.mean([r["reader_self_weight"] for r in sub])),
            "D1_mse10": float(sum(r["D1_teacher_mse10_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D2_mse10": float(sum(r["D2_retrieval_mse10_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D3_mse10": float(sum(r["D3_transfer_mse10_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D4_mse10": float(sum(r["D4_full_mse10_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D1_mae7": float(sum(r["D1_teacher_mae7_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D2_mae7": float(sum(r["D2_retrieval_mae7_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D3_mae7": float(sum(r["D3_transfer_mae7_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D4_mae7": float(sum(r["D4_full_mae7_valid_sum"] for r in sub) / max(sv16, 1.0)),
            "D1_exec12": float(
                sum(r["D1_teacher_exec12_mae7_valid_sum"] for r in sub) / max(sv12, 1.0)
            ),
            "D2_exec12": float(
                sum(r["D2_retrieval_exec12_mae7_valid_sum"] for r in sub) / max(sv12, 1.0)
            ),
            "D3_exec12": float(
                sum(r["D3_transfer_exec12_mae7_valid_sum"] for r in sub) / max(sv12, 1.0)
            ),
            "D4_exec12": float(
                sum(r["D4_full_exec12_mae7_valid_sum"] for r in sub) / max(sv12, 1.0)
            ),
            "D2_lat_mse": float(np.mean([r["D2_retrieval_latent_mse_vs_self"] for r in sub])),
            "D3_lat_mse": float(np.mean([r["D3_transfer_latent_mse_vs_self"] for r in sub])),
            "D4_lat_mse": float(np.mean([r["D4_full_latent_mse_vs_self"] for r in sub])),
        }

    # Demo trajectory / GT action characteristics (to detect distribution shifts in Oracle-GT demos)
    demo_len_map = {(r["demo_type"], r["task"], r["demo_label"]): r["demo_len"] for r in rows}
    gt_stats = {
        "demo_len_mean": float(np.mean(list(demo_len_map.values()))),
        "demo_len_min": int(min(demo_len_map.values())),
        "demo_len_max": int(max(demo_len_map.values())),
        "gt_pos_l2_mean": float(np.mean([r["gt_pos_l2"] for r in rows])),
        "gt_grip_pos_frac": float(np.mean([r["gt_grip_pos_frac"] for r in rows])),
    }

    return {
        "num_demos": len(demo_keys),
        "num_anchors": n,
        "num_tail_anchors": n_tail,
        "tail_anchor_frac": n_tail / n,
        "total_valid_steps_16": int(round(tot_v16)),
        "total_valid_steps_12": int(round(tot_v12)),
        "modes": mode_stats,
        "deltas_vs_D1": deltas_vs_d1,
        "reader": reader_stats,
        "d4_vs_d3": d4_vs_d3,
        "d3_stratification": strat,
        "gt_demo_stats": gt_stats,
    }


def _macro_average_tasks(
    per_task_aggs: Dict[str, Dict[str, Any]],
    task_list: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute equal-weight (macro) or custom-weighted average across `task_list` to eliminate
    task-distribution confounding when comparing Default vs Shifted vs Oracle-GT."""
    active = [t for t in task_list if t in per_task_aggs and per_task_aggs[t]["num_anchors"] > 0]
    if not active:
        return {"tasks": [], "num_tasks": 0}
    if weights is None:
        w = {t: 1.0 / len(active) for t in active}
    else:
        tot = sum(weights[t] for t in active)
        w = {t: weights[t] / max(tot, 1e-12) for t in active}

    def _wavg(fn) -> float:
        return float(sum(w[t] * fn(per_task_aggs[t]) for t in active))

    modes_out: Dict[str, Any] = {}
    for m in MODES:
        modes_out[m] = {
            "ddim_10d_mse": _wavg(lambda a: a["modes"][m]["ddim_10d_mse"]),
            "ddim_pos_mse": _wavg(lambda a: a["modes"][m]["ddim_pos_mse"]),
            "ddim_rot6d_mse": _wavg(lambda a: a["modes"][m]["ddim_rot6d_mse"]),
            "ddim_grip_mse": _wavg(lambda a: a["modes"][m]["ddim_grip_mse"]),
            "ddim_7d_mae": _wavg(lambda a: a["modes"][m]["ddim_7d_mae"]),
            "exec12_7d_mae": _wavg(lambda a: a["modes"][m]["exec12_7d_mae"]),
            "grip_acc": _wavg(lambda a: a["modes"][m]["grip_acc"]),
            "grip_acc_exec12": _wavg(lambda a: a["modes"][m]["grip_acc_exec12"]),
            "latent_mse_vs_self": _wavg(lambda a: a["modes"][m]["latent_mse_vs_self"]),
            "latent_lz_vs_self": _wavg(lambda a: a["modes"][m]["latent_lz_vs_self"]),
            "latent_cos_vs_self": _wavg(lambda a: a["modes"][m]["latent_cos_vs_self"]),
            "latent_rms": _wavg(lambda a: a["modes"][m]["latent_rms"]),
        }

    d1 = modes_out["D1_teacher"]
    deltas_vs_d1: Dict[str, Any] = {}
    for m in MODES[1:]:
        cur = modes_out[m]
        deltas_vs_d1[m] = {
            "ddim_10d_mse_abs": cur["ddim_10d_mse"] - d1["ddim_10d_mse"],
            "ddim_10d_mse_rel_pct": 100.0
            * (cur["ddim_10d_mse"] - d1["ddim_10d_mse"])
            / max(d1["ddim_10d_mse"], 1e-12),
            "ddim_7d_mae_abs": cur["ddim_7d_mae"] - d1["ddim_7d_mae"],
            "ddim_7d_mae_rel_pct": 100.0
            * (cur["ddim_7d_mae"] - d1["ddim_7d_mae"])
            / max(d1["ddim_7d_mae"], 1e-12),
            "exec12_7d_mae_abs": cur["exec12_7d_mae"] - d1["exec12_7d_mae"],
            "exec12_7d_mae_rel_pct": 100.0
            * (cur["exec12_7d_mae"] - d1["exec12_7d_mae"])
            / max(d1["exec12_7d_mae"], 1e-12),
            "grip_acc_pp": 100.0 * (cur["grip_acc"] - d1["grip_acc"]),
            "grip_acc_exec12_pp": 100.0 * (cur["grip_acc_exec12"] - d1["grip_acc_exec12"]),
        }

    reader_keys = [
        "mem_len_mean",
        "max_weight_mean",
        "self_weight_mean",
        "exact_hit_rate",
        "within1_hit_rate",
        "pos_err_argmax_slots_mean",
        "pos_err_argmax_steps_mean",
        "pos_err_argmax_norm_mean",
        "signed_err_argmax_slots_mean",
        "pos_err_expected_slots_mean",
        "pos_err_expected_steps_mean",
        "expected_abs_dist_slots_mean",
        "expected_abs_dist_steps_mean",
        "attn_entropy_mean",
        "attn_norm_entropy_mean",
        "attn_eff_pos_exp_mean",
    ]
    reader_out = {k: _wavg(lambda a, kk=k: a["reader"][kk]) for k in reader_keys}

    return {
        "num_tasks": len(active),
        "tasks": active,
        "weights": w,
        "modes": modes_out,
        "deltas_vs_D1": deltas_vs_d1,
        "reader": reader_out,
    }


@torch.inference_mode()
def run_offline_diagnosis(
    *,
    e7_direct_ckpt: Path = DEFAULT_E7_DIRECT_CKPT,
    stage1_ckpt: Path = DEFAULT_STAGE1_CKPT,
    out_dir: Path = DEFAULT_DIAG_OUT_DIR,
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

    # Load frozen CLIP, frozen Stage 1 Teacher, and frozen E7-Direct Best
    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    policy, payload = load_e7_policy(
        e7_direct_ckpt, stage1_ckpt=stage1_ckpt, device=device
    )
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()

    if policy.prediction_mode != "direct":
        raise RuntimeError(
            f"Expected E7-Direct checkpoint (prediction_mode='direct'), got {policy.prediction_mode!r}"
        )
    trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    trainable_cnt += sum(p.numel() for p in visual.parameters() if p.requires_grad)
    if trainable_cnt != 0:
        raise RuntimeError(f"Expected all modules frozen, found {trainable_cnt} trainable params")

    s_z = float(policy.latent_scale_s_z.item())
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
        f"[OfflineDiag] Starting D1/D2/D3/D4 diagnosis: {len(specs)} demos "
        f"(types={demo_types}, stride={behavior_stride}, bf16={use_bf16}, seed={base_seed}, "
        f"ckpt={e7_direct_ckpt} ep={payload.get('epoch')}) on {device}",
        flush=True,
    )

    # Group specs by HDF5 file path to avoid re-opening the same HDF5 file 20 times
    all_rows: List[Dict[str, Any]] = []
    # Check 7D -> 10D -> 7D roundtrip max error as sanity check
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
            raise ValueError(
                f"Length mismatch in {sp['h5_path']}:{sp['group_key']}: "
                f"av={T}, eh={len(eh_all)}, actions={len(actions_7d)}"
            )

        # 1. Exact same CLIP encoding as E7BatchedEngine._encode_single_demo_video
        d_frames = np.arange(0, T, DEMO_STRIDE, dtype=np.int32)
        anchors = compute_behavior_anchor_steps(T, stride=behavior_stride)
        prev_anchors = np.maximum(anchors - 1, 0)
        M = len(anchors)

        union_frames = np.unique(np.concatenate([d_frames, prev_anchors, anchors]))
        feats_union = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all[union_frames]),
            np.ascontiguousarray(eh_all[union_frames]),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if device.type == "cuda" else None,
        )  # (U, 2, 768) float32 on GPU

        d_pos = torch.as_tensor(
            np.searchsorted(union_frames, d_frames), dtype=torch.long, device=device
        )
        p0_pos = torch.as_tensor(
            np.searchsorted(union_frames, prev_anchors), dtype=torch.long, device=device
        )
        p1_pos = torch.as_tensor(
            np.searchsorted(union_frames, anchors), dtype=torch.long, device=device
        )

        demo_rgb_1hz = feats_union[d_pos]  # (P, 2, 768)
        anchor_query_rgb = torch.stack(
            [feats_union[p0_pos], feats_union[p1_pos]], dim=1
        )  # (M, 2, 2, 768)

        # 2. Extract Behavior Memory (h_mem, b_mem) and evaluate D1, D2, D3, D4
        with amp_ctx:
            h_mem, b_mem = policy.extract_behavior_memory_from_single_demo(
                demo_rgb_1hz, anchor_query_rgb
            )
            h_mem_f = h_mem.float().contiguous()
            b_mem_f = b_mem.float().contiguous()

            # Query = Demo's own observation at each stride-16 anchor
            h_q = h_mem_f  # (M, 4, 768)

            # D1 — Teacher Only: exact self latent b_k
            z_d1 = b_mem_f  # (M, 4, 768)

            # D2 — Retrieval Only: Behavior Memory -> Reader
            h_mem_exp = h_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            b_mem_exp = b_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            z_d2, attn_w = policy.reader(h_q, h_mem_exp, b_mem_exp, mem_pad_mask=None)

            # D3 — Transfer Only: exact self latent b_k -> Direct Transfer
            z_d3, _ = policy.adapt_latent(z_d1, h_q, delta_scale=1.0)

            # D4 — Full Model: Behavior Memory -> Reader -> Direct Transfer
            z_d4, _ = policy.adapt_latent(z_d2, h_q, delta_scale=1.0)

            # Deterministic per-(demo, anchor) DDIM initial noise shared across D1/D2/D3/D4
            noises = []
            for k, t_k in enumerate(anchors):
                s_k = _stable_seed(
                    f"{demo_type}/{task}/{demo_label}/t{int(t_k)}", base_seed=base_seed
                )
                gen = torch.Generator(device="cpu").manual_seed(s_k)
                noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
            init_noise = torch.stack(noises, dim=0).to(device=device, dtype=torch.float32)

            pred_d1 = policy.teacher.sample_from_z_self(z_d1, init_noise=init_noise)
            pred_d2 = policy.teacher.sample_from_z_self(z_d2, init_noise=init_noise)
            pred_d3 = policy.teacher.sample_from_z_self(z_d3, init_noise=init_noise)
            pred_d4 = policy.teacher.sample_from_z_self(z_d4, init_noise=init_noise)

        # 3. Ground truth 16-step action chunks + tail-window mask
        actions_10d_np = action_7d_to_10d(actions_7d)
        rt_7d_np = action_10d_to_7d(actions_10d_np)
        rt_err = float(np.max(np.abs(rt_7d_np - actions_7d)))
        if rt_err > max_rot_roundtrip_err:
            max_rot_roundtrip_err = rt_err

        step_grid = anchors[:, None] + np.arange(ACTION_HORIZON, dtype=np.int32)[None, :]  # (M, 16)
        valid_16_np = step_grid < T
        clipped_grid = np.minimum(step_grid, T - 1)
        gt_10d = torch.from_numpy(actions_10d_np[clipped_grid]).to(
            device=device, dtype=torch.float32
        )
        valid_16 = torch.from_numpy(valid_16_np).to(device=device, dtype=torch.bool)

        preds = {
            "D1_teacher": pred_d1,
            "D2_retrieval": pred_d2,
            "D3_transfer": pred_d3,
            "D4_full": pred_d4,
        }
        latents = {
            "D1_teacher": z_d1.float(),
            "D2_retrieval": z_d2.float(),
            "D3_transfer": z_d3.float(),
            "D4_full": z_d4.float(),
        }

        act_metrics = {
            m: _compute_per_anchor_action_metrics(preds[m], gt_10d, valid_16)
            for m in MODES
        }

        # 4. Latent metrics vs exact self latent z_d1
        z_ref = latents["D1_teacher"]
        z_ref_flat = z_ref.reshape(M, -1)
        lat_metrics: Dict[str, Dict[str, np.ndarray]] = {}
        for m in MODES:
            zm = latents[m]
            lat_metrics[m] = {
                "mse": (zm - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
                "cos": F.cosine_similarity(zm.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
                "sq_mean": zm.square().mean(dim=(1, 2)).cpu().numpy(),
            }
        lat_mse_d4_d3 = (
            (latents["D4_full"] - latents["D3_transfer"]).square().mean(dim=(1, 2)).cpu().numpy()
        )
        lat_cos_d4_d3 = (
            F.cosine_similarity(
                latents["D4_full"].reshape(M, -1),
                latents["D3_transfer"].reshape(M, -1),
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        # 5. Reader retrieval diagnostics (using known self anchor index k)
        w_np = attn_w.float().cpu().numpy()  # (M, M)
        ast = compute_attention_stats(attn_w, mem_pad_mask=None)
        ent_np = ast["entropy"].cpu().numpy()
        nent_np = ast["norm_entropy"].cpu().numpy()
        eff_np = ast["eff_pos_exp"].cpu().numpy()
        wmax_np = ast["max_weight"].cpu().numpy()

        idx_grid = np.arange(M, dtype=np.float64)
        anchors_f = anchors.astype(np.float64)

        # GT action chunk stats per anchor (valid steps only)
        gt_chunks_7d = actions_7d[clipped_grid]  # (M, 16, 7)

        for k in range(M):
            wk = w_np[k]
            argmax_j = int(np.argmax(wk))
            signed_slots = argmax_j - k
            err_slots = abs(signed_slots)
            err_steps = abs(int(anchors[argmax_j]) - int(anchors[k]))
            err_norm = float(err_slots) / float(max(M - 1, 1))
            exp_j = float(np.sum(wk * idx_grid))
            exp_step = float(np.sum(wk * anchors_f))
            exp_err_slots = abs(exp_j - float(k))
            exp_err_steps = abs(exp_step - float(anchors[k]))
            exp_abs_slots = float(np.sum(wk * np.abs(idx_grid - float(k))))
            exp_abs_steps = float(np.sum(wk * np.abs(anchors_f - float(anchors[k]))))

            v_mask = valid_16_np[k]
            gt_pos_l2 = float(
                np.mean(np.linalg.norm(gt_chunks_7d[k, v_mask, :3], axis=-1))
            )
            gt_grip_pos = float(np.mean(gt_chunks_7d[k, v_mask, 6] >= 0.0))

            row: Dict[str, Any] = {
                "demo_type": demo_type,
                "task_idx": task_idx,
                "task": task,
                "demo_label": demo_label,
                "demo_len": T,
                "mem_len": M,
                "anchor_idx": k,
                "anchor_step": int(anchors[k]),
                "is_tail_window": bool(not valid_16_np[k].all()),
                "n_valid_16": int(act_metrics["D1_teacher"]["n_valid_16"][k]),
                "n_valid_12": int(act_metrics["D1_teacher"]["n_valid_12"][k]),
                "reader_max_weight": float(wmax_np[k]),
                "reader_self_weight": float(wk[k]),
                "reader_argmax_idx": argmax_j,
                "reader_exact_hit": int(argmax_j == k),
                "reader_within1_hit": int(err_slots <= 1),
                "reader_pos_err_argmax_slots": int(err_slots),
                "reader_pos_err_argmax_steps": int(err_steps),
                "reader_pos_err_argmax_norm": err_norm,
                "reader_signed_err_argmax_slots": int(signed_slots),
                "reader_pos_err_expected_slots": exp_err_slots,
                "reader_pos_err_expected_steps": exp_err_steps,
                "reader_expected_abs_dist_slots": exp_abs_slots,
                "reader_expected_abs_dist_steps": exp_abs_steps,
                "reader_entropy": float(ent_np[k]),
                "reader_norm_entropy": float(nent_np[k]),
                "reader_eff_pos_exp": float(eff_np[k]),
                "latent_mse_d4_vs_d3": float(lat_mse_d4_d3[k]),
                "latent_cos_d4_vs_d3": float(lat_cos_d4_d3[k]),
                "gt_pos_l2": gt_pos_l2,
                "gt_grip_pos_frac": gt_grip_pos,
            }
            for m in MODES:
                am = act_metrics[m]
                lm = lat_metrics[m]
                row[f"{m}_mse10_masked"] = float(am["mse10_masked"][k])
                row[f"{m}_mae7_masked"] = float(am["mae7_masked"][k])
                row[f"{m}_exec12_mae7_masked"] = float(am["exec12_mae7_masked"][k])
                row[f"{m}_grip_acc16_masked"] = float(am["grip_acc16_masked"][k])
                row[f"{m}_grip_acc12_masked"] = float(am["grip_acc12_masked"][k])
                row[f"{m}_mse10_valid_sum"] = float(am["mse10_valid_sum"][k])
                row[f"{m}_pos_mse_valid_sum"] = float(am["pos_mse_valid_sum"][k])
                row[f"{m}_rot6d_mse_valid_sum"] = float(am["rot6d_mse_valid_sum"][k])
                row[f"{m}_grip_mse_valid_sum"] = float(am["grip_mse_valid_sum"][k])
                row[f"{m}_mae7_valid_sum"] = float(am["mae7_valid_sum"][k])
                row[f"{m}_exec12_mae7_valid_sum"] = float(am["exec12_mae7_valid_sum"][k])
                row[f"{m}_grip_acc16_valid_sum"] = float(am["grip_acc16_valid_sum"][k])
                row[f"{m}_grip_acc12_valid_sum"] = float(am["grip_acc12_valid_sum"][k])
                row[f"{m}_mse10_unmasked"] = float(am["mse10_unmasked"][k])
                row[f"{m}_latent_mse_vs_self"] = float(lm["mse"][k])
                row[f"{m}_latent_cos_vs_self"] = float(lm["cos"][k])
                row[f"{m}_latent_sq_mean"] = float(lm["sq_mean"][k])
            all_rows.append(row)

        if idx % 50 == 0 or idx == len(specs):
            dt = time.monotonic() - t0
            print(
                f"[OfflineDiag] {idx}/{len(specs)} demos ({len(all_rows):,} anchors) "
                f"in {dt:.1f}s ({idx / max(dt, 1e-6):.1f} demos/s)",
                flush=True,
            )

    # Aggregate results
    by_demo_type: Dict[str, Any] = {}
    per_task_by_type: Dict[str, Dict[str, Any]] = {}
    for dt in demo_types:
        dt_rows = [r for r in all_rows if r["demo_type"] == dt]
        by_demo_type[dt] = _aggregate_rows(dt_rows, s_z=s_z)
        per_task_by_type[dt] = {}
        for task in tasks:
            t_rows = [r for r in dt_rows if r["task"] == task]
            if t_rows:
                per_task_by_type[dt][task] = _aggregate_rows(t_rows, s_z=s_z)

    # Shared 9 tasks (those that have Oracle-GT demos) for unconfounded cross-type comparison
    shared_9_tasks = [
        t for t in tasks if t in per_task_by_type.get("oracle_gt", {})
    ]
    ogt_demo_weights = {
        t: float(per_task_by_type["oracle_gt"][t]["num_demos"])
        for t in shared_9_tasks
    }
    ogt_anchor_weights = {
        t: float(per_task_by_type["oracle_gt"][t]["num_anchors"])
        for t in shared_9_tasks
    }

    macro_shared_9: Dict[str, Any] = {}
    oracle_demo_weighted_9: Dict[str, Any] = {}
    oracle_anchor_weighted_9: Dict[str, Any] = {}
    macro_all_10: Dict[str, Any] = {}
    for dt in demo_types:
        macro_all_10[dt] = _macro_average_tasks(per_task_by_type[dt], tasks)
        if shared_9_tasks:
            macro_shared_9[dt] = _macro_average_tasks(per_task_by_type[dt], shared_9_tasks)
            oracle_demo_weighted_9[dt] = _macro_average_tasks(
                per_task_by_type[dt], shared_9_tasks, weights=ogt_demo_weights
            )
            oracle_anchor_weighted_9[dt] = _macro_average_tasks(
                per_task_by_type[dt], shared_9_tasks, weights=ogt_anchor_weights
            )

    # Write per-anchor CSV
    anchor_csv = out_dir / "per_anchor_records.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with anchor_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # Write summary_by_demo_type.csv (micro, macro_shared_9, oracle_weighted_9)
    summary_csv = out_dir / "summary_by_demo_type.csv"
    summary_csv_rows: List[Dict[str, Any]] = []
    for agg_kind, container in (
        ("micro_all", by_demo_type),
        ("macro_all_10", macro_all_10),
        ("macro_shared_9", macro_shared_9),
        ("oracle_demo_weighted_9", oracle_demo_weighted_9),
    ):
        for dt in demo_types:
            if dt not in container or "modes" not in container[dt]:
                continue
            blk = container[dt]
            for m in MODES:
                ms = blk["modes"][m]
                dl = blk["deltas_vs_D1"].get(m, {})
                rd = blk["reader"]
                summary_csv_rows.append(
                    {
                        "aggregation": agg_kind,
                        "demo_type": dt,
                        "mode": m,
                        "ddim_10d_mse": ms["ddim_10d_mse"],
                        "ddim_7d_mae": ms["ddim_7d_mae"],
                        "exec12_7d_mae": ms["exec12_7d_mae"],
                        "grip_acc": ms["grip_acc"],
                        "grip_acc_exec12": ms["grip_acc_exec12"],
                        "delta_vs_D1_10d_mse_abs": dl.get("ddim_10d_mse_abs", 0.0),
                        "delta_vs_D1_10d_mse_rel_pct": dl.get("ddim_10d_mse_rel_pct", 0.0),
                        "delta_vs_D1_7d_mae_abs": dl.get("ddim_7d_mae_abs", 0.0),
                        "delta_vs_D1_exec12_mae_abs": dl.get("exec12_7d_mae_abs", 0.0),
                        "delta_vs_D1_grip_acc_pp": dl.get("grip_acc_pp", 0.0),
                        "latent_mse_vs_self": ms["latent_mse_vs_self"],
                        "latent_lz_vs_self": ms["latent_lz_vs_self"],
                        "latent_cos_vs_self": ms["latent_cos_vs_self"],
                        "latent_rms": ms["latent_rms"],
                        "reader_max_weight": rd["max_weight_mean"],
                        "reader_self_weight": rd["self_weight_mean"],
                        "reader_exact_hit_rate": rd["exact_hit_rate"],
                        "reader_pos_err_argmax_slots": rd["pos_err_argmax_slots_mean"],
                        "reader_pos_err_argmax_steps": rd["pos_err_argmax_steps_mean"],
                        "reader_pos_err_expected_slots": rd["pos_err_expected_slots_mean"],
                        "reader_expected_abs_dist_slots": rd["expected_abs_dist_slots_mean"],
                    }
                )
    if summary_csv_rows:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_csv_rows)

    # Write per_task_summary.csv
    per_task_csv = out_dir / "per_task_summary.csv"
    pt_rows: List[Dict[str, Any]] = []
    for task in tasks:
        for dt in demo_types:
            if task not in per_task_by_type[dt]:
                continue
            blk = per_task_by_type[dt][task]
            rd = blk["reader"]
            for m in MODES:
                ms = blk["modes"][m]
                dl = blk["deltas_vs_D1"].get(m, {})
                pt_rows.append(
                    {
                        "task": task,
                        "demo_type": dt,
                        "num_demos": blk["num_demos"],
                        "num_anchors": blk["num_anchors"],
                        "mode": m,
                        "ddim_10d_mse": ms["ddim_10d_mse"],
                        "ddim_7d_mae": ms["ddim_7d_mae"],
                        "exec12_7d_mae": ms["exec12_7d_mae"],
                        "grip_acc": ms["grip_acc"],
                        "delta_vs_D1_10d_mse_abs": dl.get("ddim_10d_mse_abs", 0.0),
                        "delta_vs_D1_10d_mse_rel_pct": dl.get("ddim_10d_mse_rel_pct", 0.0),
                        "latent_mse_vs_self": ms["latent_mse_vs_self"],
                        "latent_cos_vs_self": ms["latent_cos_vs_self"],
                        "reader_max_weight": rd["max_weight_mean"],
                        "reader_self_weight": rd["self_weight_mean"],
                        "reader_exact_hit_rate": rd["exact_hit_rate"],
                        "reader_pos_err_argmax_slots": rd["pos_err_argmax_slots_mean"],
                        "reader_pos_err_expected_slots": rd["pos_err_expected_slots_mean"],
                    }
                )
    if pt_rows:
        with per_task_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pt_rows)

    elapsed = round(time.monotonic() - t0, 2)
    summary: Dict[str, Any] = {
        "experiment": "stage1_behavior_space_offline_decomposition",
        "checkpoints": {
            "clip_ckpt": str(BPP_CKPT),
            "stage1_ckpt": str(Path(stage1_ckpt).resolve()),
            "e7_direct_ckpt": str(Path(e7_direct_ckpt).resolve()),
            "e7_direct_epoch": payload.get("epoch"),
            "e7_direct_prediction_mode": policy.prediction_mode,
        },
        "protocol": {
            "behavior_stride": int(behavior_stride),
            "demo_stride_1hz": int(DEMO_STRIDE),
            "action_horizon": int(ACTION_HORIZON),
            "exec_horizon": int(EXEC_HORIZON),
            "base_seed": int(base_seed),
            "use_bf16": bool(use_bf16 and device.type == "cuda"),
            "s_z": s_z,
            "tail_window_mask": "valid_step = (t_k + s < T); primary metrics step-weighted over valid steps",
            "action_normalizer_scale": policy.teacher.action_normalizer.scale.cpu().tolist(),
            "action_normalizer_offset": policy.teacher.action_normalizer.offset.cpu().tolist(),
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
    }
    atomic_write_json(summary, out_dir / "offline_summary.json")

    print("\n=== Offline Decomposition Summary (9-Task Equal-Weight Macro Average) ===", flush=True)
    for dt in demo_types:
        blk = macro_shared_9.get(dt, by_demo_type[dt])
        rd = blk["reader"]
        print(
            f"\n[{dt.upper()}] Reader: max_w={rd['max_weight_mean']:.4f}, "
            f"self_w={rd['self_weight_mean']:.4f}, exact_hit={100*rd['exact_hit_rate']:.2f}%, "
            f"argmax_err={rd['pos_err_argmax_slots_mean']:.3f} slots ({rd['pos_err_argmax_steps_mean']:.1f} steps), "
            f"exp_abs_err={rd['expected_abs_dist_slots_mean']:.3f} slots",
            flush=True,
        )
        for m in MODES:
            ms = blk["modes"][m]
            dl = blk["deltas_vs_D1"].get(m)
            dstr = (
                f" (Δ10D={dl['ddim_10d_mse_abs']:+.6f} / {dl['ddim_10d_mse_rel_pct']:+.1f}%)"
                if dl
                else ""
            )
            print(
                f"  {m:13s}: 10D_MSE={ms['ddim_10d_mse']:.6f}{dstr} | "
                f"7D_MAE={ms['ddim_7d_mae']:.5f} | Exec12={ms['exec12_7d_mae']:.5f} | "
                f"Grip={100*ms['grip_acc']:.2f}% | "
                f"LatMSE={ms['latent_mse_vs_self']:.4f} | Cos={ms['latent_cos_vs_self']:.6f}",
                flush=True,
            )

    print(f"\n[OfflineDiag DONE] Saved to {out_dir} in {elapsed}s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e7-direct-ckpt", type=Path, default=DEFAULT_E7_DIRECT_CKPT)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIAG_OUT_DIR)
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
    run_offline_diagnosis(
        e7_direct_ckpt=args.e7_direct_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        out_dir=args.output,
        device=device,
        use_bf16=not args.no_bf16,
        base_seed=args.seed,
        behavior_stride=args.stride,
        demo_types=tuple(args.demo_types),
        max_demos_per_type=args.max_demos_per_type,
    )


if __name__ == "__main__":
    main()
