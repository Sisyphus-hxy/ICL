"""E9 Reader Inference-only Diagnosis (Frozen E9 Checkpoint, Zero Retraining, Zero STE).

Reuses frozen E9 Stage 1 Local Teacher (`e9/outputs/stage1_local/best.pt`),
frozen E9 Stage 2 Direct Policy (`e9/outputs/stage2_direct/train/best.pt`),
existing Behavior Memory caches (`e9/outputs/stage2_direct/cache/`,
`p0_lowrank/outputs/unseen_cache/`), and D1-D4 offline decomposition code
(`e9/diagnose_stage2_d1_d4.py`, `e7/diagnose_behavior_space.py`).

Compares 6 Reader readout families (8 configs, where temp_tau_1p00 == global_soft):
  1. global_soft    : Original Global Softmax (tau=1.0)
  2. global_top1    : Global Top-1 hard selection
  3. global_top2    : Global Top-2 (non-adjacent allowed), renormalized within top-2
  4. adjacent_top2  : Adjacent Top-2 (max adjacent pair weight sum), renormalized within pair
  5. adjacent_top3  : Adjacent Top-3 (max contiguous 3-slot window weight sum), renormalized within window
  6. temp_tau_0p25 / temp_tau_0p50 / temp_tau_1p00 : Global Soft Temperature Sweep

ONLY the Reader readout from frozen attention logits s_i is modified; attention logits,
Direct TransferTransformer, frozen Stage 1 Action Decoder, and per-sample DDIM noise
are held strictly identical across all variants. No STE, no retraining, no closed-loop.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from e5 import ACTION_DIM_10D, ACTION_HORIZON, BPP_CKPT, EXEC_HORIZON
from e5.actions import action_10d_to_7d, action_7d_to_10d
from e5.policy_server import get_chain_tasks
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder
from e7.diagnose_behavior_space import (
    _collect_demo_specs,
    _compute_per_anchor_action_metrics,
    _stable_seed,
)
from e7.model import compute_attention_stats
from p0_lowrank import (
    E5_CACHE_DIR,
    E9_BEHAVIOR_MEM_DIR,
    E9_STAGE1_CKPT,
    E9_STAGE2_CKPT,
    EMBED_DIM,
    LATENT_DIM,
    LATENT_TOKENS,
)
from p0_lowrank.alignment import (
    align_demo_pair_offline,
    detect_debounced_gripper_flips,
    sample_cached_split_pairs,
    sample_unseen_chain_pairs,
)
from p0_lowrank.cache_unseen import DEFAULT_UNSEEN_CACHE_DIR
from p0_lowrank.run_diagnosis import _build_init_noises, _extract_gt_chunk, _write_csv

from . import BEHAVIOR_STRIDE, OUTPUT_ROOT
from .stage2_direct_model import load_e9_stage2_direct_policy

DEFAULT_OUT_DIR = OUTPUT_ROOT / "reader_diagnosis"

# Canonical ordered list of reader specs (temp_tau_1p00 is verified bit-exact with global_soft)
READER_SPECS: Tuple[Tuple[str, str, str, float], ...] = (
    ("global_soft", "1_global_soft", "Global Soft (τ=1.0)", 1.0),
    ("global_top1", "2_global_top1", "Global Top-1", 1.0),
    ("global_top2", "3_global_top2", "Global Top-2 (Any)", 1.0),
    ("adjacent_top2", "4_adjacent_top2", "Adjacent Top-2", 1.0),
    ("adjacent_top3", "5_adjacent_top3", "Adjacent Top-3", 1.0),
    ("temp_tau_0p25", "6_temp_sweep", "Soft Temp (τ=0.25)", 0.25),
    ("temp_tau_0p50", "6_temp_sweep", "Soft Temp (τ=0.50)", 0.50),
    ("temp_tau_1p00", "6_temp_sweep", "Soft Temp (τ=1.00)", 1.00),
)

UNIQUE_READER_IDS: Tuple[str, ...] = (
    "global_soft",
    "global_top1",
    "global_top2",
    "adjacent_top2",
    "adjacent_top3",
    "temp_tau_0p25",
    "temp_tau_0p50",
)

SEMANTIC_PHASES: Tuple[str, ...] = (
    "approach",
    "grasp_transition",
    "transport_carry",
    "release_transition",
    "post_release_transit",
)


# ---------------------------------------------------------------------------
# 1. Pure Inference-only Reader Readout Functions (NO STE, NO training)
# ---------------------------------------------------------------------------


def apply_reader_readout(
    base_logits: torch.Tensor,
    mode: str,
    mem_pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform frozen Reader attention logits (B, M) into normalized readout weights (B, M).

    Strictly inference-only, deterministic, NO STE, NO learnable parameters.
    Padded slots (`mem_pad_mask == True`) always receive exact weight 0.0.
    """
    if base_logits.ndim != 2:
        raise ValueError(f"Expected 2D base_logits (B, M), got {tuple(base_logits.shape)}")

    B, M = base_logits.shape
    device = base_logits.device
    logits = base_logits.float()

    if mem_pad_mask is not None:
        pad = mem_pad_mask.to(device=device, dtype=torch.bool)
        if pad.shape != (B, M):
            raise ValueError(f"Expected mem_pad_mask ({B}, {M}), got {tuple(pad.shape)}")
        if pad.all(dim=-1).any():
            raise ValueError("mem_pad_mask contains a row with zero valid entries")
        valid = ~pad
        logits = logits.masked_fill(pad, float("-inf"))
    else:
        valid = torch.ones((B, M), device=device, dtype=torch.bool)

    if mode in ("global_soft", "temp_tau_1p00"):
        w = torch.softmax(logits, dim=-1)
    elif mode == "temp_tau_0p25":
        w = torch.softmax(logits / 0.25, dim=-1)
    elif mode == "temp_tau_0p50":
        w = torch.softmax(logits / 0.50, dim=-1)
    elif mode == "global_top1":
        idx = torch.argmax(logits, dim=-1)  # (B,)
        w = F.one_hot(idx, num_classes=M).to(dtype=torch.float32)
    elif mode == "global_top2":
        if M == 1:
            w = valid.float()
        else:
            k_eff = min(2, M)
            top_vals, top_idx = torch.topk(logits, k=k_eff, dim=-1)  # (B, 2)
            # If a row has only 1 valid slot, top_vals[:, 1] == -inf; softmax handles it cleanly
            top_w = torch.softmax(top_vals, dim=-1)
            w = torch.zeros((B, M), device=device, dtype=torch.float32)
            w.scatter_(1, top_idx, top_w)
    elif mode == "adjacent_top2":
        w = _adjacent_window_readout(logits, valid, window_size=2)
    elif mode == "adjacent_top3":
        w = _adjacent_window_readout(logits, valid, window_size=3)
    else:
        raise ValueError(f"Unknown reader readout mode: {mode!r}")

    w = w.masked_fill(~valid, 0.0)
    # Guard exact normalization
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return w


def _adjacent_window_readout(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    window_size: int,
) -> torch.Tensor:
    """Select the contiguous window [i, ..., i + K - 1] with maximum sum of base softmax weights
    (equivalently maximum logsumexp of logits) and renormalize strictly inside that window.

    Handles short episodes where M_valid < K by gracefully keeping all M_valid slots.
    """
    B, M = logits.shape
    K = int(window_size)
    device = logits.device
    valid_lens = valid.sum(dim=-1)  # (B,)

    if M < K:
        return torch.softmax(logits, dim=-1)

    # Build sliding window views: (B, M - K + 1, K)
    win_logits = logits.unfold(dimension=1, size=K, step=1)
    win_valid = valid.unfold(dimension=1, size=K, step=1).all(dim=-1)  # (B, M - K + 1)

    # logsumexp(s_i .. s_{i+K-1}) is strictly monotone in sum_{j=i}^{i+K-1} softmax(s)_j
    # Use base softmax sum in float64? Even cleaner: logsumexp has zero underflow and exact
    # rank equivalence to w_i + ... + w_{i+K-1}.
    win_scores = torch.logsumexp(win_logits, dim=-1)
    win_scores = win_scores.masked_fill(~win_valid, float("-inf"))

    has_full_win = win_valid.any(dim=-1)  # (B,)
    best_start = torch.argmax(win_scores.masked_fill(~has_full_win.unsqueeze(-1), 0.0), dim=-1)

    w = torch.zeros((B, M), device=device, dtype=torch.float32)
    if has_full_win.any():
        rows = torch.nonzero(has_full_win, as_tuple=False).squeeze(-1)
        starts = best_start[rows]  # (n_ok,)
        offsets = torch.arange(K, device=device, dtype=torch.long).unsqueeze(0)  # (1, K)
        idx = starts.unsqueeze(1) + offsets  # (n_ok, K)
        sel_logits = torch.gather(logits[rows], dim=1, index=idx)  # (n_ok, K)
        sel_w = torch.softmax(sel_logits, dim=-1)
        w[rows] = w[rows].scatter(1, idx, sel_w)

    if (~has_full_win).any():
        # Episode shorter than K valid slots -> renormalize over all valid slots
        short_rows = torch.nonzero(~has_full_win, as_tuple=False).squeeze(-1)
        w[short_rows] = torch.softmax(logits[short_rows], dim=-1)

    return w


def compute_base_reader_logits(
    policy: torch.nn.Module,
    h_q: torch.Tensor,
    h_mem: torch.Tensor,
    mem_pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact frozen SimpleBehaviorReader Q*K^T / sqrt(d) logits (B, M) in float32."""
    reader = policy.reader
    q = reader.encode_query(h_q)       # (B, d)
    k = reader.encode_keys(h_mem)      # (B, M, d)
    logits = torch.einsum("bd,bmd->bm", q.float(), k.float()) * float(reader.scale)
    if mem_pad_mask is not None:
        pad = mem_pad_mask.to(device=logits.device, dtype=torch.bool)
        if pad.all(dim=-1).any():
            raise ValueError("mem_pad_mask has a row with zero valid entries")
        logits = logits.masked_fill(pad, float("-inf"))
    return logits


# ---------------------------------------------------------------------------
# 2. Behavioral Phase Labeling (Universal 5-Phase + Exact Segment Phase ID)
# ---------------------------------------------------------------------------


def label_episode_behavioral_phases(
    grip_cmd: np.ndarray,
    anchors: np.ndarray,
    *,
    window_len: int = ACTION_HORIZON,
    min_run_len: int = 6,
) -> Dict[str, Any]:
    """Label each stride-16 anchor in an episode with:
      - `seg_phase_ids`: exact gripper-flip segment index in [0 .. n_flips]
      - `semantic_phases`: one of SEMANTIC_PHASES:
          * 'approach'             : open gripper before first close flip (no flip in window)
          * 'grasp_transition'     : close flip (0 -> 1) inside [t_k, t_k + window_len)
          * 'transport_carry'      : closed gripper throughout window (no flip in window)
          * 'release_transition'   : open flip (1 -> 0) inside [t_k, t_k + window_len)
          * 'post_release_transit' : open gripper after a release flip (no flip in window)
      - `flips`: list of debounced flip step indices
      - `flip_types`: list of 'close' or 'open'
      - `init_grip`: 0 (open) or 1 (closed)
    """
    T = len(grip_cmd)
    flips, init_grip = detect_debounced_gripper_flips(grip_cmd, min_run_len=min_run_len)
    bounds = [0] + flips + [T]
    n_segs = len(bounds) - 1

    # Alternating state per segment
    seg_states = [(init_grip + s) % 2 for s in range(n_segs)]
    flip_types = ["close" if seg_states[i + 1] == 1 else "open" for i in range(len(flips))]

    M = len(anchors)
    seg_phase_ids = np.empty(M, dtype=np.int32)
    semantic_phases: List[str] = []

    first_close_step = next(
        (flips[i] for i, ft in enumerate(flip_types) if ft == "close"), None
    )

    for k, tk in enumerate(anchors):
        t0 = int(tk)
        t1 = min(t0 + int(window_len), T)

        # Segment ID at anchor start t0
        seg_id = n_segs - 1
        for s in range(n_segs):
            if bounds[s] <= t0 < bounds[s + 1]:
                seg_id = s
                break
        seg_phase_ids[k] = seg_id

        # Check whether a debounced close or open event falls inside [t0, t1)
        # (or within 2 steps before t0 if transition straddles anchor boundary)
        in_win = [
            (abs(f - (t0 + 4)), f, ft)
            for f, ft in zip(flips, flip_types)
            if (t0 - 2) <= f < t1
        ]
        if in_win:
            in_win.sort(key=lambda x: x[0])
            ft = in_win[0][2]
            semantic_phases.append("grasp_transition" if ft == "close" else "release_transition")
        else:
            st = seg_states[seg_id]
            if st == 1:
                semantic_phases.append("transport_carry")
            else:
                if first_close_step is None or t0 < first_close_step:
                    semantic_phases.append("approach")
                else:
                    semantic_phases.append("post_release_transit")

    return {
        "seg_phase_ids": seg_phase_ids,
        "semantic_phases": semantic_phases,
        "flips": flips,
        "flip_types": flip_types,
        "init_grip": int(init_grip),
        "n_segs": int(n_segs),
    }


# ---------------------------------------------------------------------------
# 3. Base Attention Geometry & Readout Metrics Helpers
# ---------------------------------------------------------------------------


def compute_base_attention_geometry(w_base_row: np.ndarray) -> Dict[str, Any]:
    """Compute intrinsic geometry of raw Global Soft attention weights (1D array, length M_valid)."""
    M = len(w_base_row)
    w = np.asarray(w_base_row, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)

    order = np.argsort(-w)
    i1 = int(order[0])
    w1 = float(w[i1])
    if M >= 2:
        i2 = int(order[1])
        w2 = float(w[i2])
        top2_dist = abs(i1 - i2)
        top2_adj = int(top2_dist == 1)
    else:
        i2 = i1
        w2 = 0.0
        top2_dist = 0
        top2_adj = 1

    if M >= 3:
        i3 = int(order[2])
        w3 = float(w[i3])
        top3_span = max(i1, i2, i3) - min(i1, i2, i3)
        top3_contig = int(top3_span == 2)
    else:
        i3 = i2
        w3 = 0.0
        top3_span = top2_dist
        top3_contig = 1

    # Non-local mass outside [i1 - 1, i1 + 1]
    idx = np.arange(M)
    local3_mask = np.abs(idx - i1) <= 1
    nonlocal_mass_out3 = float(w[~local3_mask].sum())

    # Entropy
    ent = float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0))))
    norm_ent = float(ent / math.log(M)) if M > 1 else 0.0
    eff_pos = float(math.exp(ent))

    return {
        "top1_idx": i1,
        "top2_idx": i2,
        "top3_idx": i3,
        "top1_weight": w1,
        "top2_weight": w2,
        "top2_sum": w1 + w2,
        "top3_sum": w1 + w2 + w3,
        "top2_dist": int(top2_dist),
        "top2_adj": int(top2_adj),
        "top3_span": int(top3_span),
        "top3_contig": int(top3_contig),
        "nonlocal_mass_out3": nonlocal_mass_out3,
        "entropy": ent,
        "norm_entropy": norm_ent,
        "eff_pos": eff_pos,
    }


def compute_readout_row_stats(
    w_row: np.ndarray,
    w_base_row: np.ndarray,
    *,
    ref_j: int,
    anchors_s: np.ndarray,
    q_step: int,
    Tq: int,
    Ts: int,
    q_seg_phase: int,
    s_seg_phases: np.ndarray,
    exact_phase_match: bool,
    q_sem_phase: str,
    s_sem_phases: Sequence[str],
) -> Dict[str, Any]:
    """Compute per-anchor attention, alignment, phase-hit, and mixing stats for ONE reader readout."""
    M = len(w_row)
    w = np.asarray(w_row, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    wb = np.asarray(w_base_row, dtype=np.float64)
    wb = wb / max(float(wb.sum()), 1e-12)

    idx_grid = np.arange(M, dtype=np.float64)
    anchors_f = np.asarray(anchors_s, dtype=np.float64)

    order = np.argsort(-w)
    peak_j = int(order[0])
    top1_w = float(w[peak_j])

    # Active support (> 1e-7)
    active = w > 1e-7
    support_size = int(active.sum())
    captured_raw_mass = float(wb[active].sum())

    if support_size >= 2:
        i2 = int(order[1])
        eff_top2_dist = abs(peak_j - i2)
        eff_top2_adj = int(eff_top2_dist == 1)
    else:
        eff_top2_dist = 0
        eff_top2_adj = 1  # single slot has no non-adjacent mixing

    ent = float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0))))
    norm_ent = float(ent / math.log(M)) if M > 1 else 0.0
    eff_pos = float(math.exp(ent))

    # Temporal alignment relative to ref_j
    signed_peak_slots = peak_j - int(ref_j)
    peak_err_slots = abs(signed_peak_slots)
    peak_err_steps = abs(int(anchors_s[peak_j]) - int(anchors_s[ref_j]))
    peak_err_norm = float(peak_err_slots) / float(max(M - 1, 1))

    exp_j = float(np.sum(w * idx_grid))
    exp_step = float(np.sum(w * anchors_f))
    signed_exp_slots = exp_j - float(ref_j)
    exp_err_slots = abs(signed_exp_slots)
    exp_err_steps = abs(exp_step - float(anchors_s[ref_j]))
    exp_abs_dist_slots = float(np.sum(w * np.abs(idx_grid - float(ref_j))))
    exp_abs_dist_steps = float(np.sum(w * np.abs(anchors_f - float(anchors_s[ref_j]))))

    # Normalized trajectory time error |t_s / (Ts-1) - t_q / (Tq-1)|
    q_norm_t = float(q_step) / float(max(Tq - 1, 1))
    s_peak_norm_t = float(anchors_s[peak_j]) / float(max(Ts - 1, 1))
    s_exp_norm_t = exp_step / float(max(Ts - 1, 1))
    peak_norm_time_err = abs(s_peak_norm_t - q_norm_t)
    exp_norm_time_err = abs(s_exp_norm_t - q_norm_t)

    # Exact segment phase hit & in-phase probability mass (valid when exact_phase_match)
    if exact_phase_match:
        seg_match_mask = s_seg_phases == int(q_seg_phase)
        seg_phase_hit_peak = int(bool(seg_match_mask[peak_j]))
        seg_phase_mass = float(w[seg_match_mask].sum())
    else:
        seg_phase_hit_peak = -1
        seg_phase_mass = float("nan")

    # Universal 5-phase semantic phase hit & in-phase probability mass (100% coverage)
    sem_match_mask = np.array([sp == q_sem_phase for sp in s_sem_phases], dtype=bool)
    sem_phase_hit_peak = int(bool(sem_match_mask[peak_j]))
    sem_phase_mass = float(w[sem_match_mask].sum())

    return {
        "peak_j": peak_j,
        "exp_j": exp_j,
        "top1_weight": top1_w,
        "entropy": ent,
        "norm_entropy": norm_ent,
        "eff_pos": eff_pos,
        "support_size": support_size,
        "captured_raw_mass": captured_raw_mass,
        "eff_top2_dist": int(eff_top2_dist),
        "eff_top2_adj": int(eff_top2_adj),
        "ref_weight": float(w[ref_j]),
        "window_contains_ref": int(bool(active[ref_j])),
        "exact_hit": int(peak_j == int(ref_j)),
        "within1_hit": int(peak_err_slots <= 1),
        "peak_err_slots": int(peak_err_slots),
        "peak_err_steps": int(peak_err_steps),
        "peak_err_norm": peak_err_norm,
        "signed_peak_slots": int(signed_peak_slots),
        "exp_err_slots": exp_err_slots,
        "exp_err_steps": exp_err_steps,
        "signed_exp_slots": signed_exp_slots,
        "exp_abs_dist_slots": exp_abs_dist_slots,
        "exp_abs_dist_steps": exp_abs_dist_steps,
        "peak_norm_time_err": peak_norm_time_err,
        "exp_norm_time_err": exp_norm_time_err,
        "seg_phase_hit_peak": int(seg_phase_hit_peak),
        "seg_phase_mass": seg_phase_mass,
        "sem_phase_hit_peak": int(sem_phase_hit_peak),
        "sem_phase_mass": sem_phase_mass,
    }


def compute_trajectory_peak_jumps(
    traj_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Compute step-to-step Reader peak jump metrics along complete consecutive trajectories.

    Groups records by `(traj_id)` sorted by `anchor_idx` (requiring consecutive 0..M_q-1).
    """
    by_traj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in traj_records:
        by_traj[r["traj_id"]].append(r)

    all_abs_dj: List[float] = []
    all_excess_dj: List[float] = []
    all_jump_ge3: List[int] = []
    all_jump_ge5: List[int] = []
    all_back: List[int] = []
    all_back_le_m2: List[int] = []
    all_abs_dexp: List[float] = []
    all_exp_jump_ge3: List[int] = []
    all_non_mono_vs_ref: List[float] = []
    per_traj_rows: List[Dict[str, Any]] = []

    for tid in sorted(by_traj.keys()):
        seq = sorted(by_traj[tid], key=lambda x: int(x["anchor_idx"]))
        if len(seq) < 2:
            continue
        peaks = np.array([int(x["peak_j"]) for x in seq], dtype=np.int32)
        exps = np.array([float(x["exp_j"]) for x in seq], dtype=np.float64)
        refs = np.array([int(x["ref_j"]) for x in seq], dtype=np.int32)

        dj = np.diff(peaks)
        dexp = np.diff(exps)
        dref = np.diff(refs)
        abs_dj = np.abs(dj)
        abs_dexp = np.abs(dexp)
        excess_dj = np.abs(dj - 1)
        err_vs_dref = np.abs(dj - dref)

        all_abs_dj.extend(abs_dj.tolist())
        all_excess_dj.extend(excess_dj.tolist())
        all_jump_ge3.extend((abs_dj >= 3).astype(int).tolist())
        all_jump_ge5.extend((abs_dj >= 5).astype(int).tolist())
        all_back.extend((dj < 0).astype(int).tolist())
        all_back_le_m2.extend((dj <= -2).astype(int).tolist())
        all_abs_dexp.extend(abs_dexp.tolist())
        all_exp_jump_ge3.extend((abs_dexp >= 3.0).astype(int).tolist())
        all_non_mono_vs_ref.extend(err_vs_dref.tolist())

        per_traj_rows.append(
            {
                "traj_id": tid,
                "task": seq[0]["task"],
                "n_anchors": len(seq),
                "n_transitions": len(dj),
                "mean_abs_peak_step": float(np.mean(abs_dj)),
                "max_abs_peak_jump": int(np.max(abs_dj)),
                "jump_ge3_rate": float(np.mean(abs_dj >= 3)),
                "jump_ge5_rate": float(np.mean(abs_dj >= 5)),
                "backward_rate": float(np.mean(dj < 0)),
                "backward_le_m2_rate": float(np.mean(dj <= -2)),
                "mean_abs_exp_step": float(np.mean(abs_dexp)),
                "max_abs_exp_jump": float(np.max(abs_dexp)),
                "mean_peak_err_slots": float(np.mean([x["peak_err_slots"] for x in seq])),
                "mean_d2_mse10": float(np.mean([x["D2_mse10_masked"] for x in seq])),
                "mean_d4_mse10": float(np.mean([x["D4_mse10_masked"] for x in seq])),
            }
        )

    agg = {
        "num_trajectories": len(per_traj_rows),
        "num_transitions": len(all_abs_dj),
        "mean_abs_peak_step": float(np.mean(all_abs_dj)) if all_abs_dj else float("nan"),
        "mean_abs_peak_jump_excess": float(np.mean(all_excess_dj)) if all_excess_dj else float("nan"),
        "mean_abs_peak_step_vs_ref": float(np.mean(all_non_mono_vs_ref)) if all_non_mono_vs_ref else float("nan"),
        "peak_jump_ge3_rate": float(np.mean(all_jump_ge3)) if all_jump_ge3 else float("nan"),
        "peak_jump_ge5_rate": float(np.mean(all_jump_ge5)) if all_jump_ge5 else float("nan"),
        "backward_jump_rate": float(np.mean(all_back)) if all_back else float("nan"),
        "backward_le_m2_rate": float(np.mean(all_back_le_m2)) if all_back_le_m2 else float("nan"),
        "max_abs_peak_jump_traj_mean": (
            float(np.mean([r["max_abs_peak_jump"] for r in per_traj_rows]))
            if per_traj_rows
            else float("nan")
        ),
        "mean_abs_exp_step": float(np.mean(all_abs_dexp)) if all_abs_dexp else float("nan"),
        "exp_jump_ge3_rate": float(np.mean(all_exp_jump_ge3)) if all_exp_jump_ge3 else float("nan"),
    }
    return agg, per_traj_rows


# ---------------------------------------------------------------------------
# 4. Batched DDIM Decoding & Aggregation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def batched_ddim_metrics(
    policy: torch.nn.Module,
    z_all: np.ndarray,
    z_ref_all: np.ndarray,
    gt_chunks: np.ndarray,
    valid_16: np.ndarray,
    init_noise_cpu: torch.Tensor,
    *,
    device: torch.device,
    use_bf16: bool,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Decode (N, 4, 768) latents through frozen E9 Stage 1 UNet in large batches and return
    exact per-anchor action & latent metrics matching `e7/diagnose_behavior_space.py`."""
    N = len(z_all)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    out_lists: Dict[str, List[np.ndarray]] = defaultdict(list)
    lat_mse = np.empty(N, dtype=np.float32)
    lat_cos = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        z_t = torch.from_numpy(z_all[start:end]).to(device=device, dtype=torch.float32)
        zr_t = torch.from_numpy(z_ref_all[start:end]).to(device=device, dtype=torch.float32)
        noise_t = init_noise_cpu[start:end].to(device=device, dtype=torch.float32)
        gt_t = torch.from_numpy(gt_chunks[start:end]).to(device=device, dtype=torch.float32)
        v16_t = torch.from_numpy(valid_16[start:end]).to(device=device, dtype=torch.bool)

        with amp_ctx:
            pred_10d = policy.teacher.sample_from_z_self(z_t, init_noise=noise_t)

        am = _compute_per_anchor_action_metrics(pred_10d, gt_t, v16_t)
        for k, v in am.items():
            out_lists[k].append(v)

        zf = z_t.float()
        zrf = zr_t.float()
        lat_mse[start:end] = (zf - zrf).square().mean(dim=(1, 2)).cpu().numpy()
        lat_cos[start:end] = (
            F.cosine_similarity(zf.reshape(end - start, -1), zrf.reshape(end - start, -1), dim=-1)
            .cpu()
            .numpy()
        )

    out = {k: np.concatenate(v, axis=0) for k, v in out_lists.items()}
    out["latent_mse_vs_self"] = lat_mse
    out["latent_cos_vs_self"] = lat_cos
    return out


def aggregate_reader_rows(
    rows: List[Dict[str, Any]],
    *,
    s_z: float,
    include_task_macro: bool = False,
) -> Dict[str, Any]:
    """Aggregate per-anchor records for one (split, reader) into step-weighted action,
    latent, attention, alignment, and phase metrics."""
    n = len(rows)
    if n == 0:
        return {"num_anchors": 0}

    s_z2 = float(s_z) * float(s_z)
    tot_v16 = float(sum(r["n_valid_16"] for r in rows))
    tot_v12 = float(sum(r["n_valid_12"] for r in rows))

    def _agg_mode(prefix: str) -> Dict[str, float]:
        lmse = float(np.mean([r[f"{prefix}_latent_mse"] for r in rows]))
        return {
            "ddim_10d_mse": float(sum(r[f"{prefix}_mse10_valid_sum"] for r in rows) / max(tot_v16, 1.0)),
            "ddim_pos_mse": float(sum(r[f"{prefix}_pos_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0)),
            "ddim_rot6d_mse": float(sum(r[f"{prefix}_rot6d_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0)),
            "ddim_grip_mse": float(sum(r[f"{prefix}_grip_mse_valid_sum"] for r in rows) / max(tot_v16, 1.0)),
            "ddim_7d_mae": float(sum(r[f"{prefix}_mae7_valid_sum"] for r in rows) / max(tot_v16, 1.0)),
            "exec12_7d_mae": float(sum(r[f"{prefix}_exec12_mae7_valid_sum"] for r in rows) / max(tot_v12, 1.0)),
            "grip_acc16": float(sum(r[f"{prefix}_grip_acc16_valid_sum"] for r in rows) / max(tot_v16, 1.0)),
            "grip_acc12": float(sum(r[f"{prefix}_grip_acc12_valid_sum"] for r in rows) / max(tot_v12, 1.0)),
            "latent_mse_vs_self": lmse,
            "latent_lz_vs_self": lmse / s_z2,
            "latent_cos_vs_self": float(np.mean([r[f"{prefix}_latent_cos"] for r in rows])),
        }

    seg_valid_rows = [r for r in rows if int(r["seg_phase_hit_peak"]) >= 0]

    res: Dict[str, Any] = {
        "num_anchors": n,
        "D1": _agg_mode("D1"),
        "D2": _agg_mode("D2"),
        "D3": _agg_mode("D3"),
        "D4": _agg_mode("D4"),
        "attention": {
            "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
            "norm_entropy_mean": float(np.mean([r["norm_entropy"] for r in rows])),
            "eff_pos_mean": float(np.mean([r["eff_pos"] for r in rows])),
            "top1_weight_mean": float(np.mean([r["top1_weight"] for r in rows])),
            "top1_weight_median": float(np.median([r["top1_weight"] for r in rows])),
            "support_size_mean": float(np.mean([r["support_size"] for r in rows])),
            "captured_raw_mass_mean": float(np.mean([r["captured_raw_mass"] for r in rows])),
            "eff_top2_adj_rate": float(np.mean([r["eff_top2_adj"] for r in rows])),
            "eff_top2_dist_mean": float(np.mean([r["eff_top2_dist"] for r in rows])),
            # Base (raw logits) geometry (same across readers, included for convenience)
            "base_entropy_mean": float(np.mean([r["base_entropy"] for r in rows])),
            "base_norm_entropy_mean": float(np.mean([r["base_norm_entropy"] for r in rows])),
            "base_top1_weight_mean": float(np.mean([r["base_top1_weight"] for r in rows])),
            "base_top2_sum_mean": float(np.mean([r["base_top2_sum"] for r in rows])),
            "base_top3_sum_mean": float(np.mean([r["base_top3_sum"] for r in rows])),
            "base_top2_adj_rate": float(np.mean([r["base_top2_adj"] for r in rows])),
            "base_top2_dist_mean": float(np.mean([r["base_top2_dist"] for r in rows])),
            "base_top2_dist_ge3_rate": float(np.mean([r["base_top2_dist"] >= 3 for r in rows])),
            "base_top2_dist_ge5_rate": float(np.mean([r["base_top2_dist"] >= 5 for r in rows])),
            "base_top3_contig_rate": float(np.mean([r["base_top3_contig"] for r in rows])),
            "base_nonlocal_mass_out3_mean": float(np.mean([r["base_nonlocal_mass_out3"] for r in rows])),
        },
        "alignment": {
            "exact_hit_rate": float(np.mean([r["exact_hit"] for r in rows])),
            "within1_hit_rate": float(np.mean([r["within1_hit"] for r in rows])),
            "window_contains_ref_rate": float(np.mean([r["window_contains_ref"] for r in rows])),
            "ref_weight_mean": float(np.mean([r["ref_weight"] for r in rows])),
            "peak_err_slots_mean": float(np.mean([r["peak_err_slots"] for r in rows])),
            "peak_err_slots_median": float(np.median([r["peak_err_slots"] for r in rows])),
            "peak_err_steps_mean": float(np.mean([r["peak_err_steps"] for r in rows])),
            "peak_err_norm_mean": float(np.mean([r["peak_err_norm"] for r in rows])),
            "signed_peak_slots_mean": float(np.mean([r["signed_peak_slots"] for r in rows])),
            "exp_err_slots_mean": float(np.mean([r["exp_err_slots"] for r in rows])),
            "exp_err_steps_mean": float(np.mean([r["exp_err_steps"] for r in rows])),
            "signed_exp_slots_mean": float(np.mean([r["signed_exp_slots"] for r in rows])),
            "exp_abs_dist_slots_mean": float(np.mean([r["exp_abs_dist_slots"] for r in rows])),
            "exp_abs_dist_steps_mean": float(np.mean([r["exp_abs_dist_steps"] for r in rows])),
            "peak_norm_time_err_mean": float(np.mean([r["peak_norm_time_err"] for r in rows])),
            "exp_norm_time_err_mean": float(np.mean([r["exp_norm_time_err"] for r in rows])),
            "seg_phase_valid_frac": float(len(seg_valid_rows) / max(n, 1)),
            "seg_phase_hit_peak_rate": (
                float(np.mean([r["seg_phase_hit_peak"] for r in seg_valid_rows]))
                if seg_valid_rows
                else float("nan")
            ),
            "seg_phase_mass_mean": (
                float(np.mean([r["seg_phase_mass"] for r in seg_valid_rows]))
                if seg_valid_rows
                else float("nan")
            ),
            "sem_phase_hit_peak_rate": float(np.mean([r["sem_phase_hit_peak"] for r in rows])),
            "sem_phase_mass_mean": float(np.mean([r["sem_phase_mass"] for r in rows])),
        },
    }

    if "O2_mse10_valid_sum" in rows[0]:
        res["OracleAlign_D2"] = _agg_mode("O2")
        res["OracleAlign_D4"] = _agg_mode("O4")

    if include_task_macro:
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        t_aggs = {t: aggregate_reader_rows(tr, s_z=s_z, include_task_macro=False) for t, tr in sorted(by_task.items())}
        macro: Dict[str, Any] = {"num_tasks": len(t_aggs)}
        for mkey in ("D1", "D2", "D3", "D4"):
            macro[mkey] = {
                k: float(np.mean([ta[mkey][k] for ta in t_aggs.values()]))
                for k in next(iter(t_aggs.values()))[mkey].keys()
            }
        if "OracleAlign_D2" in next(iter(t_aggs.values())):
            for mkey in ("OracleAlign_D2", "OracleAlign_D4"):
                macro[mkey] = {
                    k: float(np.mean([ta[mkey][k] for ta in t_aggs.values()]))
                    for k in next(iter(t_aggs.values()))[mkey].keys()
                }
        for grp in ("attention", "alignment"):
            macro[grp] = {
                k: float(np.nanmean([ta[grp][k] for ta in t_aggs.values()]))
                for k in next(iter(t_aggs.values()))[grp].keys()
            }
        res["task_macro"] = macro
        res["per_task"] = t_aggs

    return res


def compute_stratifications(rows: List[Dict[str, Any]], *, s_z: float) -> Dict[str, Any]:
    """Stratify rows by HIT/MISS, Top-2 adjacency, Top-2 distance, and Phase Hit to
    directly separate Location Error from Multi-Latent Mixing Error."""
    strata_defs = (
        ("all", lambda r: True),
        ("base_exact_hit", lambda r: int(r["base_exact_hit"]) == 1),
        ("base_within1_hit", lambda r: int(r["base_within1_hit"]) == 1),
        ("base_miss_exact", lambda r: int(r["base_exact_hit"]) == 0),
        ("base_miss_gt1", lambda r: int(r["base_within1_hit"]) == 0),
        ("base_top2_adjacent", lambda r: int(r["base_top2_adj"]) == 1),
        ("base_top2_non_adjacent", lambda r: int(r["base_top2_adj"]) == 0),
        ("base_top2_dist_ge3", lambda r: int(r["base_top2_dist"]) >= 3),
        ("base_top2_dist_ge5", lambda r: int(r["base_top2_dist"]) >= 5),
        ("hit_and_top2_non_adj", lambda r: int(r["base_exact_hit"]) == 1 and int(r["base_top2_adj"]) == 0),
        ("hit_and_top2_adj", lambda r: int(r["base_exact_hit"]) == 1 and int(r["base_top2_adj"]) == 1),
        ("miss_and_top2_non_adj", lambda r: int(r["base_exact_hit"]) == 0 and int(r["base_top2_adj"]) == 0),
        ("base_seg_phase_hit", lambda r: int(r["base_seg_phase_hit"]) == 1),
        ("base_seg_phase_miss", lambda r: int(r["base_seg_phase_hit"]) == 0),
    )
    out: Dict[str, Any] = {}
    n_tot = max(len(rows), 1)
    for name, fn in strata_defs:
        sub = [r for r in rows if fn(r)]
        if not sub:
            out[name] = {"num_anchors": 0, "frac": 0.0}
            continue
        ag = aggregate_reader_rows(sub, s_z=s_z, include_task_macro=False)
        out[name] = {
            "num_anchors": len(sub),
            "frac": float(len(sub) / n_tot),
            "D1_mse10": ag["D1"]["ddim_10d_mse"],
            "D2_mse10": ag["D2"]["ddim_10d_mse"],
            "D3_mse10": ag["D3"]["ddim_10d_mse"],
            "D4_mse10": ag["D4"]["ddim_10d_mse"],
            "D2_lz": ag["D2"]["latent_lz_vs_self"],
            "D4_lz": ag["D4"]["latent_lz_vs_self"],
            "D2_mae7": ag["D2"]["ddim_7d_mae"],
            "D4_mae7": ag["D4"]["ddim_7d_mae"],
            "D2_grip16": ag["D2"]["grip_acc16"],
            "D4_grip16": ag["D4"]["grip_acc16"],
            "entropy": ag["attention"]["entropy_mean"],
            "top1_weight": ag["attention"]["top1_weight_mean"],
            "captured_raw_mass": ag["attention"]["captured_raw_mass_mean"],
            "peak_err_slots": ag["alignment"]["peak_err_slots_mean"],
            "exp_err_slots": ag["alignment"]["exp_err_slots_mean"],
            "exact_hit_rate": ag["alignment"]["exact_hit_rate"],
            "within1_hit_rate": ag["alignment"]["within1_hit_rate"],
            "seg_phase_hit_rate": ag["alignment"]["seg_phase_hit_peak_rate"],
            "seg_phase_mass": ag["alignment"]["seg_phase_mass_mean"],
        }
        if "OracleAlign_D2" in ag:
            out[name]["OracleAlign_D2_mse10"] = ag["OracleAlign_D2"]["ddim_10d_mse"]
            out[name]["OracleAlign_D4_mse10"] = ag["OracleAlign_D4"]["ddim_10d_mse"]
    return out


def compute_phase_breakdown(rows: List[Dict[str, Any]], *, s_z: float) -> Dict[str, Any]:
    """Break down D1/D2/D3/D4, attention, and alignment metrics across the 5 behavioral phases."""
    out: Dict[str, Any] = {}
    n_tot = max(len(rows), 1)
    for ph in SEMANTIC_PHASES:
        sub = [r for r in rows if r["q_sem_phase"] == ph]
        if not sub:
            out[ph] = {"num_anchors": 0, "frac": 0.0}
            continue
        ag = aggregate_reader_rows(sub, s_z=s_z, include_task_macro=False)
        out[ph] = {
            "num_anchors": len(sub),
            "frac": float(len(sub) / n_tot),
            "D1": ag["D1"],
            "D2": ag["D2"],
            "D3": ag["D3"],
            "D4": ag["D4"],
            "attention": ag["attention"],
            "alignment": ag["alignment"],
        }
        if "OracleAlign_D2" in ag:
            out[ph]["OracleAlign_D2"] = ag["OracleAlign_D2"]
            out[ph]["OracleAlign_D4"] = ag["OracleAlign_D4"]
    return out


# ---------------------------------------------------------------------------
# 5. Core Evaluator for a List of Anchor Pairs (Cross-Demo OR Self-Demo)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_anchor_dataset(
    policy: torch.nn.Module,
    *,
    split_name: str,
    pairs: List[Dict[str, Any]],
    mem_h: np.ndarray,
    mem_b: np.ndarray,
    mem_anchor_steps: np.ndarray,
    mem_offsets: np.ndarray,
    mem_lengths: np.ndarray,
    actions_10d: np.ndarray,
    ee_pos_all: np.ndarray,
    action_offsets: np.ndarray,
    init_noise_cpu: torch.Tensor,
    device: torch.device,
    use_bf16: bool,
    batch_size: int = 512,
    is_self_demo: bool = False,
    compute_oracle_align: bool = True,
    collect_heatmaps_n: int = 8,
) -> Dict[str, Any]:
    """Evaluate all 6 Reader families (7 unique + temp_tau_1p00 mirror) on a dataset of
    `(q_ep, q_anchor_idx, s_ep)` anchor pairs with shared logits, Transfer, Decoder, and DDIM noise."""
    t0 = time.monotonic()
    N = len(pairs)
    s_z = float(policy.latent_scale_s_z.item())
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # 1. Precompute episode-level behavioral phase labels & pairwise offline alignment
    ep_phase_cache: Dict[int, Dict[str, Any]] = {}
    pair_align_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _get_ep_phases(ep: int) -> Dict[str, Any]:
        if ep not in ep_phase_cache:
            a0, a1 = int(action_offsets[ep]), int(action_offsets[ep + 1])
            m0, m1 = int(mem_offsets[ep]), int(mem_offsets[ep] + mem_lengths[ep])
            ep_phase_cache[ep] = label_episode_behavioral_phases(
                actions_10d[a0:a1, 9], mem_anchor_steps[m0:m1]
            )
        return ep_phase_cache[ep]

    q_steps = np.empty(N, dtype=np.int32)
    ref_js = np.empty(N, dtype=np.int32)
    exact_phase_matches = np.empty(N, dtype=bool)
    gt_chunks = np.empty((N, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
    valid_16 = np.empty((N, ACTION_HORIZON), dtype=bool)
    z_d1_all = np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
    z_oracle_all = np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32) if compute_oracle_align else None
    h_q_all = np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)

    for i, p in enumerate(pairs):
        q_ep = int(p["q_ep"])
        s_ep = int(p["s_ep"])
        k = int(p["q_anchor_idx"])
        q_ph = _get_ep_phases(q_ep)
        _ = _get_ep_phases(s_ep)

        q_a0, q_a1 = int(action_offsets[q_ep]), int(action_offsets[q_ep + 1])
        q_m0 = int(mem_offsets[q_ep])
        s_m0 = int(mem_offsets[s_ep])
        Tq = q_a1 - q_a0

        q_step = int(mem_anchor_steps[q_m0 + k])
        q_steps[i] = q_step

        if is_self_demo:
            ref_j = k
            exact_phase_matches[i] = True
        else:
            key = (q_ep, s_ep)
            if key not in pair_align_cache:
                s_a0, s_a1 = int(action_offsets[s_ep]), int(action_offsets[s_ep + 1])
                q_m1 = q_m0 + int(mem_lengths[q_ep])
                s_m1 = s_m0 + int(mem_lengths[s_ep])
                pair_align_cache[key] = align_demo_pair_offline(
                    ee_pos_q=ee_pos_all[q_a0:q_a1],
                    grip_q=actions_10d[q_a0:q_a1, 9],
                    ee_pos_s=ee_pos_all[s_a0:s_a1],
                    grip_s=actions_10d[s_a0:s_a1, 9],
                    anchors_q=mem_anchor_steps[q_m0:q_m1],
                    anchors_s=mem_anchor_steps[s_m0:s_m1],
                )
            al = pair_align_cache[key]
            ref_j = int(al["matched_j"][k])
            exact_phase_matches[i] = bool(al["exact_phase_match"])

        ref_js[i] = ref_j
        gq, vq = _extract_gt_chunk(actions_10d, q_a0, Tq, q_step)
        gt_chunks[i] = gq
        valid_16[i] = vq

        q_flat = q_m0 + k
        z_d1_all[i] = mem_b[q_flat].astype(np.float32)
        h_q_all[i] = mem_h[q_flat].astype(np.float32)
        if z_oracle_all is not None:
            z_oracle_all[i] = mem_b[s_m0 + ref_j].astype(np.float32)

    # 2. Compute D3 (Transfer on exact query self-latent) & OracleAlign D4 ONCE
    z_d3_all = np.empty_like(z_d1_all)
    z_o4_all = np.empty_like(z_d1_all) if compute_oracle_align else None
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        hq_t = torch.from_numpy(h_q_all[start:end]).to(device=device, dtype=torch.float32)
        z1_t = torch.from_numpy(z_d1_all[start:end]).to(device=device, dtype=torch.float32)
        with amp_ctx:
            z3_t, _ = policy.adapt_latent(z1_t, hq_t, delta_scale=1.0)
        z_d3_all[start:end] = z3_t.float().cpu().numpy()
        if z_o4_all is not None and z_oracle_all is not None:
            zo_t = torch.from_numpy(z_oracle_all[start:end]).to(device=device, dtype=torch.float32)
            with amp_ctx:
                zo4_t, _ = policy.adapt_latent(zo_t, hq_t, delta_scale=1.0)
            z_o4_all[start:end] = zo4_t.float().cpu().numpy()

    # Decode shared D1, D3, and OracleAlign D2/D4 ONCE
    m_d1 = batched_ddim_metrics(
        policy, z_d1_all, z_d1_all, gt_chunks, valid_16, init_noise_cpu,
        device=device, use_bf16=use_bf16, batch_size=batch_size,
    )
    m_d3 = batched_ddim_metrics(
        policy, z_d3_all, z_d1_all, gt_chunks, valid_16, init_noise_cpu,
        device=device, use_bf16=use_bf16, batch_size=batch_size,
    )
    m_o2 = None
    m_o4 = None
    if compute_oracle_align and z_oracle_all is not None and z_o4_all is not None:
        m_o2 = batched_ddim_metrics(
            policy, z_oracle_all, z_d1_all, gt_chunks, valid_16, init_noise_cpu,
            device=device, use_bf16=use_bf16, batch_size=batch_size,
        )
        m_o4 = batched_ddim_metrics(
            policy, z_o4_all, z_d1_all, gt_chunks, valid_16, init_noise_cpu,
            device=device, use_bf16=use_bf16, batch_size=batch_size,
        )

    # 3. Single pass over frozen Reader Q/K logits -> generate z_d2, z_d4, and weights for ALL unique readers
    z_d2_by_mode: Dict[str, np.ndarray] = {
        m: np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32) for m in UNIQUE_READER_IDS
    }
    z_d4_by_mode: Dict[str, np.ndarray] = {
        m: np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32) for m in UNIQUE_READER_IDS
    }
    row_stats_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in UNIQUE_READER_IDS}
    base_geom_all: List[Dict[str, Any]] = []
    # Keep per-anchor raw weight arrays only for heatmap episodes (if requested)
    # Identify trajectories with full consecutive anchors
    traj_anchor_counts: Dict[str, int] = defaultdict(int)
    for p in pairs:
        if "traj_id" in p:
            traj_anchor_counts[p["traj_id"]] += 1
    eligible_hm_trajs = sorted(t for t, c in traj_anchor_counts.items() if c >= 6)
    hm_traj_set: set[str] = set()
    if collect_heatmaps_n > 0 and eligible_hm_trajs:
        # Pick evenly spaced trajectories across tasks for diverse coverage
        step_hm = max(1, len(eligible_hm_trajs) // max(collect_heatmaps_n * 3, 1))
        hm_traj_set = set(eligible_hm_trajs[::step_hm][: max(collect_heatmaps_n * 4, 24)])
    hm_weights_store: Dict[str, Dict[str, Dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )  # traj_id -> mode -> k -> w (M_s,)

    temp1_max_diff = 0.0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        batch_pairs = pairs[start:end]

        s_lens = [int(mem_lengths[int(p["s_ep"])]) for p in batch_pairs]
        max_m = max(s_lens)

        h_mem_np = np.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        b_mem_np = np.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        pad_mask_np = np.ones((B, max_m), dtype=bool)
        for bi, p in enumerate(batch_pairs):
            s_ep = int(p["s_ep"])
            s_m0 = int(mem_offsets[s_ep])
            m_len = s_lens[bi]
            h_mem_np[bi, :m_len] = mem_h[s_m0 : s_m0 + m_len].astype(np.float32)
            b_mem_np[bi, :m_len] = mem_b[s_m0 : s_m0 + m_len].astype(np.float32)
            pad_mask_np[bi, :m_len] = False

        hq_t = torch.from_numpy(h_q_all[start:end]).to(device=device, dtype=torch.float32)
        hm_t = torch.from_numpy(h_mem_np).to(device=device, dtype=torch.float32)
        bm_t = torch.from_numpy(b_mem_np).to(device=device, dtype=torch.float32)
        pad_t = torch.from_numpy(pad_mask_np).to(device=device, dtype=torch.bool)

        with amp_ctx:
            base_logits = compute_base_reader_logits(policy, hq_t, hm_t, mem_pad_mask=pad_t)

        # Verify temp_tau_1p00 == global_soft bit-exact
        w_gs = apply_reader_readout(base_logits, "global_soft", mem_pad_mask=pad_t)
        w_t1 = apply_reader_readout(base_logits, "temp_tau_1p00", mem_pad_mask=pad_t)
        temp1_max_diff = max(temp1_max_diff, float((w_gs - w_t1).abs().max().item()))
        w_base_np = w_gs.cpu().numpy()

        # Precompute base geometry for this batch
        batch_base_geom: List[Dict[str, Any]] = []
        for bi, p in enumerate(batch_pairs):
            idx = start + bi
            m_len = s_lens[bi]
            g = compute_base_attention_geometry(w_base_np[bi, :m_len])
            ref_j = int(ref_js[idx])
            g["base_exact_hit"] = int(g["top1_idx"] == ref_j)
            g["base_within1_hit"] = int(abs(g["top1_idx"] - ref_j) <= 1)
            s_ep = int(p["s_ep"])
            q_ep = int(p["q_ep"])
            k = int(p["q_anchor_idx"])
            q_seg = int(_get_ep_phases(q_ep)["seg_phase_ids"][k])
            s_segs = _get_ep_phases(s_ep)["seg_phase_ids"]
            g["base_seg_phase_hit"] = (
                int(s_segs[g["top1_idx"]] == q_seg) if bool(exact_phase_matches[idx]) else -1
            )
            batch_base_geom.append(g)
            base_geom_all.append(g)

        for mode in UNIQUE_READER_IDS:
            w_m = w_gs if mode == "global_soft" else apply_reader_readout(
                base_logits, mode, mem_pad_mask=pad_t
            )
            with amp_ctx:
                z2_t = torch.einsum("bm,bmkd->bkd", w_m.to(dtype=bm_t.dtype), bm_t)
                z4_t, _ = policy.adapt_latent(z2_t, hq_t, delta_scale=1.0)
            z_d2_by_mode[mode][start:end] = z2_t.float().cpu().numpy()
            z_d4_by_mode[mode][start:end] = z4_t.float().cpu().numpy()

            w_m_np = w_m.cpu().numpy()
            for bi, p in enumerate(batch_pairs):
                idx = start + bi
                q_ep = int(p["q_ep"])
                s_ep = int(p["s_ep"])
                k = int(p["q_anchor_idx"])
                m_len = s_lens[bi]
                s_m0 = int(mem_offsets[s_ep])
                q_a0, q_a1 = int(action_offsets[q_ep]), int(action_offsets[q_ep + 1])
                s_a0, s_a1 = int(action_offsets[s_ep]), int(action_offsets[s_ep + 1])
                q_ph = _get_ep_phases(q_ep)
                s_ph = _get_ep_phases(s_ep)

                rs = compute_readout_row_stats(
                    w_m_np[bi, :m_len],
                    w_base_np[bi, :m_len],
                    ref_j=int(ref_js[idx]),
                    anchors_s=mem_anchor_steps[s_m0 : s_m0 + m_len],
                    q_step=int(q_steps[idx]),
                    Tq=q_a1 - q_a0,
                    Ts=s_a1 - s_a0,
                    q_seg_phase=int(q_ph["seg_phase_ids"][k]),
                    s_seg_phases=s_ph["seg_phase_ids"],
                    exact_phase_match=bool(exact_phase_matches[idx]),
                    q_sem_phase=q_ph["semantic_phases"][k],
                    s_sem_phases=s_ph["semantic_phases"],
                )
                row_stats_by_mode[mode].append(rs)

                tid = p.get("traj_id")
                if tid in hm_traj_set:
                    hm_weights_store[tid][mode][k] = w_m_np[bi, :m_len].astype(np.float32)

    # 4. Batched DDIM decoding for D2 & D4 across all unique reader modes
    per_mode_rows: Dict[str, List[Dict[str, Any]]] = {}
    for mode in UNIQUE_READER_IDS:
        tm0 = time.monotonic()
        m_d2 = batched_ddim_metrics(
            policy, z_d2_by_mode[mode], z_d1_all, gt_chunks, valid_16, init_noise_cpu,
            device=device, use_bf16=use_bf16, batch_size=batch_size,
        )
        m_d4 = batched_ddim_metrics(
            policy, z_d4_by_mode[mode], z_d1_all, gt_chunks, valid_16, init_noise_cpu,
            device=device, use_bf16=use_bf16, batch_size=batch_size,
        )

        rows: List[Dict[str, Any]] = []
        rs_list = row_stats_by_mode[mode]
        for i, p in enumerate(pairs):
            q_ep = int(p["q_ep"])
            s_ep = int(p["s_ep"])
            k = int(p["q_anchor_idx"])
            q_ph = _get_ep_phases(q_ep)
            bg = base_geom_all[i]
            rs = rs_list[i]

            row: Dict[str, Any] = {
                "split": split_name,
                "reader": mode,
                "task": p["task"],
                "traj_id": p.get("traj_id", f"q{q_ep}_s{s_ep}"),
                "q_ep": q_ep,
                "s_ep": s_ep,
                "anchor_idx": k,
                "q_step": int(q_steps[i]),
                "ref_j": int(ref_js[i]),
                "mem_len": int(mem_lengths[s_ep]),
                "q_seg_phase": int(q_ph["seg_phase_ids"][k]),
                "q_sem_phase": q_ph["semantic_phases"][k],
                "exact_phase_match": int(bool(exact_phase_matches[i])),
                "n_valid_16": int(m_d1["n_valid_16"][i]),
                "n_valid_12": int(m_d1["n_valid_12"][i]),
                # Base raw attention geometry (for consistent stratification across readers)
                "base_top1_idx": bg["top1_idx"],
                "base_top1_weight": bg["top1_weight"],
                "base_top2_sum": bg["top2_sum"],
                "base_top3_sum": bg["top3_sum"],
                "base_top2_dist": bg["top2_dist"],
                "base_top2_adj": bg["top2_adj"],
                "base_top3_contig": bg["top3_contig"],
                "base_nonlocal_mass_out3": bg["nonlocal_mass_out3"],
                "base_entropy": bg["entropy"],
                "base_norm_entropy": bg["norm_entropy"],
                "base_exact_hit": bg["base_exact_hit"],
                "base_within1_hit": bg["base_within1_hit"],
                "base_seg_phase_hit": bg["base_seg_phase_hit"],
                # Subset flag (if provided, e.g., P0 3,000 subset inside full unseen 4,067)
                "in_p0_subset": int(bool(p.get("in_p0_subset", False))),
            }
            row.update(rs)

            for prefix, am in (("D1", m_d1), ("D2", m_d2), ("D3", m_d3), ("D4", m_d4)):
                row[f"{prefix}_mse10_masked"] = float(am["mse10_masked"][i])
                row[f"{prefix}_mae7_masked"] = float(am["mae7_masked"][i])
                row[f"{prefix}_mse10_valid_sum"] = float(am["mse10_valid_sum"][i])
                row[f"{prefix}_pos_mse_valid_sum"] = float(am["pos_mse_valid_sum"][i])
                row[f"{prefix}_rot6d_mse_valid_sum"] = float(am["rot6d_mse_valid_sum"][i])
                row[f"{prefix}_grip_mse_valid_sum"] = float(am["grip_mse_valid_sum"][i])
                row[f"{prefix}_mae7_valid_sum"] = float(am["mae7_valid_sum"][i])
                row[f"{prefix}_exec12_mae7_valid_sum"] = float(am["exec12_mae7_valid_sum"][i])
                row[f"{prefix}_grip_acc16_valid_sum"] = float(am["grip_acc16_valid_sum"][i])
                row[f"{prefix}_grip_acc12_valid_sum"] = float(am["grip_acc12_valid_sum"][i])
                row[f"{prefix}_latent_mse"] = float(am["latent_mse_vs_self"][i])
                row[f"{prefix}_latent_cos"] = float(am["latent_cos_vs_self"][i])

            if m_o2 is not None and m_o4 is not None:
                for prefix, am in (("O2", m_o2), ("O4", m_o4)):
                    row[f"{prefix}_mse10_masked"] = float(am["mse10_masked"][i])
                    row[f"{prefix}_mse10_valid_sum"] = float(am["mse10_valid_sum"][i])
                    row[f"{prefix}_pos_mse_valid_sum"] = float(am["pos_mse_valid_sum"][i])
                    row[f"{prefix}_rot6d_mse_valid_sum"] = float(am["rot6d_mse_valid_sum"][i])
                    row[f"{prefix}_grip_mse_valid_sum"] = float(am["grip_mse_valid_sum"][i])
                    row[f"{prefix}_mae7_valid_sum"] = float(am["mae7_valid_sum"][i])
                    row[f"{prefix}_exec12_mae7_valid_sum"] = float(am["exec12_mae7_valid_sum"][i])
                    row[f"{prefix}_grip_acc16_valid_sum"] = float(am["grip_acc16_valid_sum"][i])
                    row[f"{prefix}_grip_acc12_valid_sum"] = float(am["grip_acc12_valid_sum"][i])
                    row[f"{prefix}_latent_mse"] = float(am["latent_mse_vs_self"][i])
                    row[f"{prefix}_latent_cos"] = float(am["latent_cos_vs_self"][i])

            rows.append(row)

        per_mode_rows[mode] = rows
        ag_tmp = aggregate_reader_rows(rows, s_z=s_z, include_task_macro=False)
        print(
            f"  [{split_name:15s} | {mode:14s}] "
            f"D2={ag_tmp['D2']['ddim_10d_mse']:.5f} (Lz={ag_tmp['D2']['latent_lz_vs_self']:.4f}) | "
            f"D4={ag_tmp['D4']['ddim_10d_mse']:.5f} (Lz={ag_tmp['D4']['latent_lz_vs_self']:.4f}) | "
            f"H={ag_tmp['attention']['entropy_mean']:.3f} w1={ag_tmp['attention']['top1_weight_mean']:.3f} "
            f"hit={100*ag_tmp['alignment']['exact_hit_rate']:.1f}% "
            f"err={ag_tmp['alignment']['peak_err_slots_mean']:.2f}sl "
            f"segHit={100*ag_tmp['alignment']['seg_phase_hit_peak_rate']:.1f}% "
            f"({time.monotonic() - tm0:.1f}s)",
            flush=True,
        )

    # Mirror temp_tau_1p00 from global_soft (already verified max_diff == 0.0)
    per_mode_rows["temp_tau_1p00"] = [
        {**r, "reader": "temp_tau_1p00"} for r in per_mode_rows["global_soft"]
    ]

    # Build trajectory heatmap payloads for selected trajectories
    heatmap_payloads: List[Dict[str, Any]] = []
    if hm_weights_store:
        # Score trajectories by mean D2 MSE and peak jumps so we include both good & failure cases
        gs_by_traj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in per_mode_rows["global_soft"]:
            if r["traj_id"] in hm_weights_store:
                gs_by_traj[r["traj_id"]].append(r)

        scored: List[Tuple[float, str]] = []
        for tid, seq in gs_by_traj.items():
            seq_s = sorted(seq, key=lambda x: int(x["anchor_idx"]))
            peaks = np.array([int(x["peak_j"]) for x in seq_s])
            jump_rate = float(np.mean(np.abs(np.diff(peaks)) >= 3)) if len(peaks) > 1 else 0.0
            d2_mse = float(np.mean([x["D2_mse10_masked"] for x in seq_s]))
            scored.append((d2_mse + 0.05 * jump_rate, tid))
        scored.sort(key=lambda x: x[0])

        # Select mix of best (well-aligned), median, and worst (high error/jump) trajectories
        n_pick = min(collect_heatmaps_n, len(scored))
        if n_pick > 0:
            pick_idxs = sorted(
                set(np.linspace(0, len(scored) - 1, n_pick, dtype=int).tolist())
            )
            for pi in pick_idxs:
                tid = scored[pi][1]
                seq_s = sorted(gs_by_traj[tid], key=lambda x: int(x["anchor_idx"]))
                q_ep = int(seq_s[0]["q_ep"])
                s_ep = int(seq_s[0]["s_ep"])
                M_q = len(seq_s)
                M_s = int(mem_lengths[s_ep])
                q_ph = _get_ep_phases(q_ep)
                s_ph = _get_ep_phases(s_ep)
                q_m0 = int(mem_offsets[q_ep])
                s_m0 = int(mem_offsets[s_ep])

                mats: Dict[str, np.ndarray] = {}
                for mode in UNIQUE_READER_IDS:
                    mat = np.zeros((M_q, M_s), dtype=np.float32)
                    for row_i, r in enumerate(seq_s):
                        k = int(r["anchor_idx"])
                        if k in hm_weights_store[tid][mode]:
                            mat[row_i] = hm_weights_store[tid][mode][k]
                    mats[mode] = mat

                # Per-anchor D2/D4 MSE for global_soft, global_top1, adjacent_top2
                mode_d2 = {}
                mode_d4 = {}
                for mode in ("global_soft", "global_top1", "global_top2", "adjacent_top2", "adjacent_top3", "temp_tau_0p25"):
                    m_seq = sorted(
                        [r for r in per_mode_rows[mode] if r["traj_id"] == tid],
                        key=lambda x: int(x["anchor_idx"]),
                    )
                    mode_d2[mode] = np.array([r["D2_mse10_masked"] for r in m_seq], dtype=np.float32)
                    mode_d4[mode] = np.array([r["D4_mse10_masked"] for r in m_seq], dtype=np.float32)

                heatmap_payloads.append(
                    {
                        "split": split_name,
                        "traj_id": tid,
                        "task": seq_s[0]["task"],
                        "q_ep": q_ep,
                        "s_ep": s_ep,
                        "M_q": M_q,
                        "M_s": M_s,
                        "ref_js": np.array([int(r["ref_j"]) for r in seq_s], dtype=np.int32),
                        "q_anchors": mem_anchor_steps[q_m0 : q_m0 + M_q].astype(np.int32),
                        "s_anchors": mem_anchor_steps[s_m0 : s_m0 + M_s].astype(np.int32),
                        "q_flips": np.array(q_ph["flips"], dtype=np.int32),
                        "s_flips": np.array(s_ph["flips"], dtype=np.int32),
                        "q_sem_phases": [r["q_sem_phase"] for r in seq_s],
                        "mats": mats,
                        "mode_d2": mode_d2,
                        "mode_d4": mode_d4,
                        "gs_d2_mean": float(np.mean(mode_d2["global_soft"])),
                        "gs_d4_mean": float(np.mean(mode_d4["global_soft"])),
                    }
                )

    return {
        "split": split_name,
        "num_anchors": N,
        "temp1_vs_global_soft_max_diff": float(temp1_max_diff),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "per_mode_rows": per_mode_rows,
        "heatmaps": heatmap_payloads,
    }


# ---------------------------------------------------------------------------
# 6. Sampling Complete Seen-Val Cross-Layout Trajectories
# ---------------------------------------------------------------------------


def sample_seen_val_trajectories(
    episodes: Sequence[Dict[str, Any]],
    mem_lengths: np.ndarray,
    *,
    num_trajectories: int = 200,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """Sample `num_trajectories` COMPLETE consecutive cross-layout episode pairs from fold='val'
    (same task & procedure, different layout), returning all anchors k = 0..M_q-1 per pair."""
    groups: Dict[Tuple[str, str], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for ep_idx, row in enumerate(episodes):
        if row["fold"] == "val":
            groups[(row["task"], row["procedure"])][int(row["layout"])].append(ep_idx)

    eligible = [
        (key, by_lay) for key, by_lay in sorted(groups.items()) if len(by_lay) >= 2
    ]
    rng = np.random.default_rng(seed)

    # Round-robin pick query episodes across all eligible (task, procedure) groups
    pools: List[List[int]] = []
    for _, by_lay in eligible:
        all_eps = [ep for eps_l in by_lay.values() for ep in eps_l]
        order = rng.permutation(len(all_eps))
        pools.append([all_eps[int(i)] for i in order])

    cursors = [0] * len(eligible)
    picked_eps: List[Tuple[int, int, str, str]] = []
    order_g = list(range(len(eligible)))
    while len(picked_eps) < num_trajectories:
        rng.shuffle(order_g)
        progressed = False
        for gi in order_g:
            if len(picked_eps) >= num_trajectories:
                break
            if cursors[gi] >= len(pools[gi]):
                continue
            q_ep = pools[gi][cursors[gi]]
            cursors[gi] += 1
            progressed = True
            (task, proc), by_lay = eligible[gi]
            q_lay = int(episodes[q_ep]["layout"])
            cands = [e for lay, eps_l in by_lay.items() if lay != q_lay for e in eps_l]
            s_ep = int(cands[int(rng.integers(len(cands)))])
            assert q_ep != s_ep and int(episodes[s_ep]["layout"]) != q_lay
            picked_eps.append((q_ep, s_ep, task, proc))
        if not progressed:
            break

    pairs: List[Dict[str, Any]] = []
    for q_ep, s_ep, task, proc in picked_eps:
        m_q = int(mem_lengths[q_ep])
        tid = f"seen_q{q_ep}_s{s_ep}"
        for k in range(m_q):
            pairs.append(
                {
                    "split": "seen_cross_traj",
                    "task": task,
                    "procedure": proc,
                    "q_ep": q_ep,
                    "s_ep": s_ep,
                    "q_layout": int(episodes[q_ep]["layout"]),
                    "s_layout": int(episodes[s_ep]["layout"]),
                    "q_anchor_idx": k,
                    "traj_id": tid,
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# 7. Oracle-GT Single-Pass Feature Cache (for Full 531-Demo Self D1-D4)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def load_or_encode_oracle_gt_latents(
    policy: torch.nn.Module,
    *,
    cache_dir: Path,
    device: torch.device,
    use_bf16: bool = True,
    max_demos: Optional[int] = None,
) -> Dict[str, Any]:
    """Encode 331 Oracle-GT demos ONCE with frozen CLIP + E9 Stage 1 Local Encoder
    (matching `e9/diagnose_stage2_d1_d4.py`) and cache to disk for zero-cost reuse."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "oracle_gt_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if max_demos is None or int(meta.get("num_demos", -1)) == int(max_demos):
            print(f"[OracleGT Cache] Reusing existing cache at {cache_dir}", flush=True)
            return {
                "meta": meta,
                "mem_h": np.load(cache_dir / "mem_h.npy", mmap_mode="r"),
                "mem_b": np.load(cache_dir / "mem_b.npy", mmap_mode="r"),
                "mem_anchor_steps": np.load(cache_dir / "mem_anchor_steps.npy"),
                "mem_offsets": np.load(cache_dir / "mem_offsets.npy"),
                "mem_lengths": np.load(cache_dir / "mem_lengths.npy"),
                "actions_10d": np.load(cache_dir / "actions_10d.npy", mmap_mode="r"),
                "ee_pos": np.load(cache_dir / "ee_pos.npy", mmap_mode="r"),
                "action_offsets": np.load(cache_dir / "action_offsets.npy"),
            }

    t0 = time.monotonic()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_chain_tasks()
    specs = _collect_demo_specs(tasks, demo_types=("oracle_gt",))
    if max_demos is not None:
        specs = specs[:max_demos]

    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    demos_meta: List[Dict[str, Any]] = []
    mem_h_list: List[np.ndarray] = []
    mem_b_list: List[np.ndarray] = []
    mem_anchors_list: List[np.ndarray] = []
    mem_offsets: List[int] = [0]
    mem_lengths: List[int] = []
    actions_10d_list: List[np.ndarray] = []
    ee_pos_list: List[np.ndarray] = []
    action_offsets: List[int] = [0]

    print(f"[OracleGT Cache] Encoding {len(specs)} Oracle-GT demos once on {device}...", flush=True)
    for idx, sp in enumerate(specs, start=1):
        with h5py.File(sp["h5_path"], "r") as f:
            conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
            d = f[sp["group_key"]]
            av_all = d["obs/agentview_rgb"][:]
            eh_all = d["obs/eye_in_hand_rgb"][:]
            actions_7d = d["actions"][:].astype(np.float32)
            ee_pos = d["obs/ee_pos"][:].astype(np.float32)
        T = len(av_all)

        feats = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if device.type == "cuda" else None,
        )
        with amp_ctx:
            h_mem, b_mem, anchors = policy.teacher.extract_behavior_memory_from_demo_features(
                feats, stride=BEHAVIOR_STRIDE
            )

        M = len(anchors)
        mem_h_list.append(h_mem.float().cpu().numpy().astype(np.float16))
        mem_b_list.append(b_mem.float().cpu().numpy().astype(np.float16))
        mem_anchors_list.append(anchors.astype(np.int32))
        mem_lengths.append(M)
        mem_offsets.append(mem_offsets[-1] + M)

        a10 = action_7d_to_10d(actions_7d).astype(np.float32)
        actions_10d_list.append(a10)
        ee_pos_list.append(ee_pos)
        action_offsets.append(action_offsets[-1] + T)

        demos_meta.append(
            {
                "ep_idx": idx - 1,
                "task_idx": sp["task_idx"],
                "task": sp["task"],
                "demo_label": sp["demo_label"],
                "frames": int(T),
                "mem_len": int(M),
            }
        )
        if idx % 60 == 0 or idx == len(specs):
            dt = time.monotonic() - t0
            print(
                f"  [OracleGT Cache] {idx}/{len(specs)} demos encoded in {dt:.1f}s "
                f"({idx / max(dt, 1e-6):.1f} demos/s)",
                flush=True,
            )

    del visual
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mem_h = np.concatenate(mem_h_list, axis=0)
    mem_b = np.concatenate(mem_b_list, axis=0)
    mem_anchors = np.concatenate(mem_anchors_list, axis=0)
    mem_offsets_arr = np.asarray(mem_offsets, dtype=np.int64)
    mem_lengths_arr = np.asarray(mem_lengths, dtype=np.int32)
    actions_10d = np.concatenate(actions_10d_list, axis=0)
    ee_pos_arr = np.concatenate(ee_pos_list, axis=0)
    action_offsets_arr = np.asarray(action_offsets, dtype=np.int64)

    np.save(cache_dir / "mem_h.npy", mem_h)
    np.save(cache_dir / "mem_b.npy", mem_b)
    np.save(cache_dir / "mem_anchor_steps.npy", mem_anchors)
    np.save(cache_dir / "mem_offsets.npy", mem_offsets_arr)
    np.save(cache_dir / "mem_lengths.npy", mem_lengths_arr)
    np.save(cache_dir / "actions_10d.npy", actions_10d)
    np.save(cache_dir / "ee_pos.npy", ee_pos_arr)
    np.save(cache_dir / "action_offsets.npy", action_offsets_arr)

    meta = {
        "num_demos": len(demos_meta),
        "total_anchors": int(len(mem_anchors)),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "demos": demos_meta,
    }
    atomic_write_json(meta, meta_path)
    return {
        "meta": meta,
        "mem_h": mem_h,
        "mem_b": mem_b,
        "mem_anchor_steps": mem_anchors,
        "mem_offsets": mem_offsets_arr,
        "mem_lengths": mem_lengths_arr,
        "actions_10d": actions_10d,
        "ee_pos": ee_pos_arr,
        "action_offsets": action_offsets_arr,
    }


def _build_d1_d4_self_noise(
    pairs: List[Dict[str, Any]],
    q_steps: np.ndarray,
    *,
    demo_type: str,
    base_seed: int = 12345,
) -> torch.Tensor:
    """Exact bit-for-bit DDIM noise matching `e9/diagnose_stage2_d1_d4.py`."""
    noises = []
    for p, tk in zip(pairs, q_steps):
        s_k = _stable_seed(f"{demo_type}/{p['task']}/{p['demo_label']}/t{int(tk)}", base_seed=base_seed)
        gen = torch.Generator(device="cpu").manual_seed(s_k)
        noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
    return torch.stack(noises, dim=0)


# ---------------------------------------------------------------------------
# 8. Main End-to-End Diagnostic Runner
# ---------------------------------------------------------------------------


def run_e9_reader_diagnosis(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    stage1_ckpt: Path = E9_STAGE1_CKPT,
    stage2_ckpt: Path = E9_STAGE2_CKPT,
    e5_cache_dir: Path = E5_CACHE_DIR,
    e9_cache_dir: Path = E9_BEHAVIOR_MEM_DIR,
    unseen_cache_dir: Path = DEFAULT_UNSEEN_CACHE_DIR,
    n_seen_p0: int = 3000,
    n_seen_trajs: int = 200,
    n_unseen_p0: int = 3000,
    seed: int = 42,
    ddim_seed: int = 12345,
    batch_size: int = 512,
    use_bf16: bool = True,
    skip_oracle_gt: bool = False,
    device: torch.device,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load strictly frozen E9 policy
    policy, payload = load_e9_stage2_direct_policy(
        stage2_ckpt, stage1_ckpt=stage1_ckpt, device=device
    )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if trainable_cnt != 0:
        raise RuntimeError(f"Expected 0 trainable params, found {trainable_cnt}")
    s_z = float(policy.latent_scale_s_z.item())

    print(
        f"[E9 Reader Diag] Frozen E9 Stage 2 loaded (ep={payload.get('epoch')}, "
        f"s_z={s_z:.6f}, trainable=0) on {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})",
        flush=True,
    )

    # Load existing Seen-Val & Unseen-Chain caches
    episodes = json.loads((e5_cache_dir / "episodes.json").read_text())
    actions_10d_e5 = np.load(e5_cache_dir / "actions_10d.npy", mmap_mode="r")
    ee_pos_e5 = np.load(e5_cache_dir / "proprio_11d.npy", mmap_mode="r")[:, :3]
    action_offsets_e5 = np.load(e5_cache_dir / "action_offsets.npy")

    e9_mem_h = np.load(e9_cache_dir / "mem_h.npy", mmap_mode="r")
    e9_mem_b = np.load(e9_cache_dir / "mem_b.npy", mmap_mode="r")
    e9_mem_anchors = np.load(e9_cache_dir / "mem_anchor_steps.npy")
    e9_mem_offsets = np.load(e9_cache_dir / "mem_offsets.npy")
    e9_mem_lengths = np.load(e9_cache_dir / "mem_lengths.npy")

    unseen_meta = json.loads((unseen_cache_dir / "unseen_meta.json").read_text())
    unseen_demos = unseen_meta["demos"]
    u_mem_h = np.load(unseen_cache_dir / "mem_h.npy", mmap_mode="r")
    u_mem_b = np.load(unseen_cache_dir / "mem_b_e9.npy", mmap_mode="r")
    u_mem_anchors = np.load(unseen_cache_dir / "mem_anchor_steps.npy")
    u_mem_offsets = np.load(unseen_cache_dir / "mem_offsets.npy")
    u_mem_lengths = np.load(unseen_cache_dir / "mem_lengths.npy")
    u_actions_10d = np.load(unseen_cache_dir / "actions_10d.npy", mmap_mode="r")
    u_ee_pos = np.load(unseen_cache_dir / "ee_pos.npy", mmap_mode="r")
    u_action_offsets = np.load(unseen_cache_dir / "action_offsets.npy")

    all_split_results: Dict[str, Any] = {}
    all_heatmaps: List[Dict[str, Any]] = []
    per_traj_jump_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Suite 1A: Seen-Val Cross-Demo (Exact P0 3,000 pairs)
    # ------------------------------------------------------------------
    print("\n=== [1/5] Seen-Val Cross-Demo (Exact P0 3,000 Pairs) ===", flush=True)
    seen_p0_pairs = sample_cached_split_pairs(
        episodes, e9_mem_lengths, fold="val", target_pairs=n_seen_p0, seed=seed + 1001
    )
    seen_p0_qsteps = np.array(
        [int(e9_mem_anchors[int(e9_mem_offsets[p["q_ep"]]) + int(p["q_anchor_idx"])]) for p in seen_p0_pairs],
        dtype=np.int32,
    )
    seen_p0_noise = _build_init_noises(seen_p0_pairs, seen_p0_qsteps, base_seed=ddim_seed)
    res_seen_p0 = evaluate_anchor_dataset(
        policy,
        split_name="seen_cross_p0",
        pairs=seen_p0_pairs,
        mem_h=e9_mem_h,
        mem_b=e9_mem_b,
        mem_anchor_steps=e9_mem_anchors,
        mem_offsets=e9_mem_offsets,
        mem_lengths=e9_mem_lengths,
        actions_10d=actions_10d_e5,
        ee_pos_all=ee_pos_e5,
        action_offsets=action_offsets_e5,
        init_noise_cpu=seen_p0_noise,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
        is_self_demo=False,
        compute_oracle_align=True,
        collect_heatmaps_n=0,
    )
    all_split_results["seen_cross_p0"] = res_seen_p0

    # ------------------------------------------------------------------
    # Suite 1B: Seen-Val Complete Cross-Demo Trajectories (200 trajs, peak jumps + heatmaps)
    # ------------------------------------------------------------------
    print(f"\n=== [2/5] Seen-Val Complete Cross-Demo Trajectories ({n_seen_trajs} trajs) ===", flush=True)
    seen_traj_pairs = sample_seen_val_trajectories(
        episodes, e9_mem_lengths, num_trajectories=n_seen_trajs, seed=seed + 2026
    )
    seen_traj_qsteps = np.array(
        [int(e9_mem_anchors[int(e9_mem_offsets[p["q_ep"]]) + int(p["q_anchor_idx"])]) for p in seen_traj_pairs],
        dtype=np.int32,
    )
    seen_traj_noise = _build_init_noises(seen_traj_pairs, seen_traj_qsteps, base_seed=ddim_seed)
    res_seen_traj = evaluate_anchor_dataset(
        policy,
        split_name="seen_cross_traj",
        pairs=seen_traj_pairs,
        mem_h=e9_mem_h,
        mem_b=e9_mem_b,
        mem_anchor_steps=e9_mem_anchors,
        mem_offsets=e9_mem_offsets,
        mem_lengths=e9_mem_lengths,
        actions_10d=actions_10d_e5,
        ee_pos_all=ee_pos_e5,
        action_offsets=action_offsets_e5,
        init_noise_cpu=seen_traj_noise,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
        is_self_demo=False,
        compute_oracle_align=True,
        collect_heatmaps_n=8,
    )
    all_split_results["seen_cross_traj"] = res_seen_traj
    all_heatmaps.extend(res_seen_traj["heatmaps"])

    # ------------------------------------------------------------------
    # Suite 2: Unseen-Chain Complete Cross-Demo (All 200 demos = 4,067 anchors;
    #          strictly contains exact P0 3,000 Unseen-Chain pairs!)
    # ------------------------------------------------------------------
    print("\n=== [3/5] Unseen-Chain Cross-Demo (All 200 Complete Trajectories + Exact P0 3,000 Subset) ===", flush=True)
    p0_unseen_pairs = sample_unseen_chain_pairs(
        unseen_demos, target_pairs=n_unseen_p0, seed=seed + 2002
    )
    p0_unseen_key_set = {
        (int(p["q_ep"]), int(p["q_anchor_idx"]), int(p["s_ep"])) for p in p0_unseen_pairs
    }

    by_task_u: Dict[str, List[int]] = defaultdict(list)
    for idx, d in enumerate(unseen_demos):
        by_task_u[d["task"]].append(idx)

    unseen_full_pairs: List[Dict[str, Any]] = []
    for task in sorted(by_task_u.keys()):
        d_idxs = by_task_u[task]
        half = max(1, len(d_idxs) // 2)
        for local_i, q_idx in enumerate(d_idxs):
            s_idx = d_idxs[(local_i + half) % len(d_idxs)]
            m_q = int(unseen_demos[q_idx]["mem_len"])
            tid = f"unseen_{task}_q{q_idx}_s{s_idx}"
            for k in range(m_q):
                unseen_full_pairs.append(
                    {
                        "split": "unseen_chain",
                        "task": task,
                        "procedure": task,
                        "q_ep": int(q_idx),
                        "s_ep": int(s_idx),
                        "q_layout": int(unseen_demos[q_idx]["demo_idx"]),
                        "s_layout": int(unseen_demos[s_idx]["demo_idx"]),
                        "q_anchor_idx": k,
                        "traj_id": tid,
                        "in_p0_subset": (int(q_idx), int(k), int(s_idx)) in p0_unseen_key_set,
                    }
                )
    assert sum(1 for p in unseen_full_pairs if p["in_p0_subset"]) == len(p0_unseen_pairs), (
        "Exact P0 unseen subset mismatch!"
    )
    unseen_full_qsteps = np.array(
        [int(u_mem_anchors[int(u_mem_offsets[p["q_ep"]]) + int(p["q_anchor_idx"])]) for p in unseen_full_pairs],
        dtype=np.int32,
    )
    # Uses exact P0 noise key format (`split='unseen_chain'`) so the 3,000 subset matches P0/E10 bit-for-bit!
    unseen_full_noise = _build_init_noises(unseen_full_pairs, unseen_full_qsteps, base_seed=ddim_seed)
    res_unseen_full = evaluate_anchor_dataset(
        policy,
        split_name="unseen_cross_traj",
        pairs=unseen_full_pairs,
        mem_h=u_mem_h,
        mem_b=u_mem_b,
        mem_anchor_steps=u_mem_anchors,
        mem_offsets=u_mem_offsets,
        mem_lengths=u_mem_lengths,
        actions_10d=u_actions_10d,
        ee_pos_all=u_ee_pos,
        action_offsets=u_action_offsets,
        init_noise_cpu=unseen_full_noise,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
        is_self_demo=False,
        compute_oracle_align=True,
        collect_heatmaps_n=10,
    )
    all_split_results["unseen_cross_traj"] = res_unseen_full
    all_heatmaps.extend(res_unseen_full["heatmaps"])

    # Slice exact P0 3,000 Unseen-Chain subset directly (zero extra compute!)
    all_split_results["unseen_cross_p0"] = {
        "split": "unseen_cross_p0",
        "num_anchors": len(p0_unseen_pairs),
        "temp1_vs_global_soft_max_diff": res_unseen_full["temp1_vs_global_soft_max_diff"],
        "elapsed_seconds": 0.0,
        "per_mode_rows": {
            m: [r for r in rows if int(r["in_p0_subset"]) == 1]
            for m, rows in res_unseen_full["per_mode_rows"].items()
        },
        "heatmaps": [],
    }

    # ------------------------------------------------------------------
    # Suite 3A: Self-Demo D1-D4 Default (100 demos) & Shifted (100 demos)
    #           reusing `unseen_cache` (exact same demos as `diagnose_stage2_d1_d4.py`!)
    # ------------------------------------------------------------------
    print("\n=== [4/5] Self-Demo D1-D4: Default (100 demos) & Shifted (100 demos) ===", flush=True)
    for dt_name, idx_cond in (
        ("self_default", lambda di: 0 <= di < 10),
        ("self_shifted", lambda di: 10 <= di < 20),
    ):
        dt_short = dt_name.replace("self_", "")
        self_pairs: List[Dict[str, Any]] = []
        for ep_i, d in enumerate(unseen_demos):
            if not idx_cond(int(d["demo_idx"])):
                continue
            m_q = int(d["mem_len"])
            tid = f"{dt_name}_{d['task']}_{d['demo_label']}"
            for k in range(m_q):
                self_pairs.append(
                    {
                        "split": dt_name,
                        "task": d["task"],
                        "demo_label": d["demo_label"],
                        "q_ep": ep_i,
                        "s_ep": ep_i,
                        "q_anchor_idx": k,
                        "traj_id": tid,
                    }
                )
        qsteps_s = np.array(
            [int(u_mem_anchors[int(u_mem_offsets[p["q_ep"]]) + int(p["q_anchor_idx"])]) for p in self_pairs],
            dtype=np.int32,
        )
        noise_s = _build_d1_d4_self_noise(self_pairs, qsteps_s, demo_type=dt_short, base_seed=ddim_seed)
        res_s = evaluate_anchor_dataset(
            policy,
            split_name=dt_name,
            pairs=self_pairs,
            mem_h=u_mem_h,
            mem_b=u_mem_b,
            mem_anchor_steps=u_mem_anchors,
            mem_offsets=u_mem_offsets,
            mem_lengths=u_mem_lengths,
            actions_10d=u_actions_10d,
            ee_pos_all=u_ee_pos,
            action_offsets=u_action_offsets,
            init_noise_cpu=noise_s,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
            is_self_demo=True,
            compute_oracle_align=False,  # self-demo oracle == D1/D3
            collect_heatmaps_n=4 if dt_name == "self_default" else 0,
        )
        all_split_results[dt_name] = res_s
        all_heatmaps.extend(res_s["heatmaps"])

    # ------------------------------------------------------------------
    # Suite 3B: Self-Demo D1-D4 Oracle-GT (331 demos, 7,053 anchors)
    # ------------------------------------------------------------------
    if not skip_oracle_gt:
        print("\n=== [5/5] Self-Demo D1-D4: Oracle-GT (331 demos / 7,053 anchors) ===", flush=True)
        ogt = load_or_encode_oracle_gt_latents(
            policy,
            cache_dir=out_dir / "oracle_gt_cache",
            device=device,
            use_bf16=use_bf16,
        )
        ogt_pairs: List[Dict[str, Any]] = []
        for ep_i, d in enumerate(ogt["meta"]["demos"]):
            m_q = int(d["mem_len"])
            tid = f"self_oracle_gt_{d['task']}_{d['demo_label']}"
            for k in range(m_q):
                ogt_pairs.append(
                    {
                        "split": "self_oracle_gt",
                        "task": d["task"],
                        "demo_label": d["demo_label"],
                        "q_ep": ep_i,
                        "s_ep": ep_i,
                        "q_anchor_idx": k,
                        "traj_id": tid,
                    }
                )
        qsteps_o = np.array(
            [int(ogt["mem_anchor_steps"][int(ogt["mem_offsets"][p["q_ep"]]) + int(p["q_anchor_idx"])]) for p in ogt_pairs],
            dtype=np.int32,
        )
        noise_o = _build_d1_d4_self_noise(ogt_pairs, qsteps_o, demo_type="oracle_gt", base_seed=ddim_seed)
        res_ogt = evaluate_anchor_dataset(
            policy,
            split_name="self_oracle_gt",
            pairs=ogt_pairs,
            mem_h=ogt["mem_h"],
            mem_b=ogt["mem_b"],
            mem_anchor_steps=ogt["mem_anchor_steps"],
            mem_offsets=ogt["mem_offsets"],
            mem_lengths=ogt["mem_lengths"],
            actions_10d=ogt["actions_10d"],
            ee_pos_all=ogt["ee_pos"],
            action_offsets=ogt["action_offsets"],
            init_noise_cpu=noise_o,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
            is_self_demo=True,
            compute_oracle_align=False,
            collect_heatmaps_n=8,
        )
        all_split_results["self_oracle_gt"] = res_ogt
        all_heatmaps.extend(res_ogt["heatmaps"])

    # ------------------------------------------------------------------
    # 9. Aggregate All Metrics, Peak Jumps, Stratifications, Phase Breakdown, & Save CSV/JSON/NPZ
    # ------------------------------------------------------------------
    print("\n=== Aggregating & Saving Complete CSV / JSON / Heatmap Artifacts ===", flush=True)
    summary_splits: Dict[str, Any] = {}
    main_table_rows: List[Dict[str, Any]] = []
    phase_table_rows: List[Dict[str, Any]] = []
    strat_table_rows: List[Dict[str, Any]] = []
    per_task_unseen_rows: List[Dict[str, Any]] = []

    traj_capable_splits = {
        "seen_cross_traj",
        "unseen_cross_traj",
        "self_default",
        "self_shifted",
        "self_oracle_gt",
    }

    for split_name, sres in all_split_results.items():
        include_macro = split_name in (
            "unseen_cross_p0",
            "unseen_cross_traj",
            "self_default",
            "self_shifted",
            "self_oracle_gt",
        )
        is_traj_split = split_name in traj_capable_splits

        split_block: Dict[str, Any] = {
            "num_anchors": sres["num_anchors"],
            "temp1_vs_global_soft_max_diff": sres["temp1_vs_global_soft_max_diff"],
            "readers": {},
        }

        for r_id, r_family, r_label, r_tau in READER_SPECS:
            rows = sres["per_mode_rows"][r_id]
            ag = aggregate_reader_rows(rows, s_z=s_z, include_task_macro=include_macro)
            strat = compute_stratifications(rows, s_z=s_z)
            phases = compute_phase_breakdown(rows, s_z=s_z)
            jump_agg = None
            if is_traj_split:
                jump_agg, tj_rows = compute_trajectory_peak_jumps(rows)
                if r_id != "temp_tau_1p00":
                    for tr in tj_rows:
                        per_traj_jump_rows.append(
                            {"split": split_name, "reader": r_id, **tr}
                        )

            split_block["readers"][r_id] = {
                "family": r_family,
                "label": r_label,
                "tau": r_tau,
                "aggregate": ag,
                "peak_jumps": jump_agg,
                "stratifications": strat,
                "phases": phases,
            }

            # Main CSV row
            at = ag["attention"]
            al = ag["alignment"]
            t_row: Dict[str, Any] = {
                "split": split_name,
                "reader": r_id,
                "family": r_family,
                "label": r_label,
                "tau": r_tau,
                "num_anchors": ag["num_anchors"],
                "D1_mse10": ag["D1"]["ddim_10d_mse"],
                "D2_mse10": ag["D2"]["ddim_10d_mse"],
                "D3_mse10": ag["D3"]["ddim_10d_mse"],
                "D4_mse10": ag["D4"]["ddim_10d_mse"],
                "D2_pos_mse": ag["D2"]["ddim_pos_mse"],
                "D2_rot6d_mse": ag["D2"]["ddim_rot6d_mse"],
                "D2_grip_mse": ag["D2"]["ddim_grip_mse"],
                "D4_pos_mse": ag["D4"]["ddim_pos_mse"],
                "D4_rot6d_mse": ag["D4"]["ddim_rot6d_mse"],
                "D4_grip_mse": ag["D4"]["ddim_grip_mse"],
                "D2_mae7": ag["D2"]["ddim_7d_mae"],
                "D4_mae7": ag["D4"]["ddim_7d_mae"],
                "D2_grip_acc16": ag["D2"]["grip_acc16"],
                "D4_grip_acc16": ag["D4"]["grip_acc16"],
                "D2_latent_lz": ag["D2"]["latent_lz_vs_self"],
                "D4_latent_lz": ag["D4"]["latent_lz_vs_self"],
                "D2_latent_cos": ag["D2"]["latent_cos_vs_self"],
                "D4_latent_cos": ag["D4"]["latent_cos_vs_self"],
                "OracleAlign_D2_mse10": ag["OracleAlign_D2"]["ddim_10d_mse"] if "OracleAlign_D2" in ag else float("nan"),
                "OracleAlign_D4_mse10": ag["OracleAlign_D4"]["ddim_10d_mse"] if "OracleAlign_D4" in ag else float("nan"),
                "D2_mse10_task_macro": ag["task_macro"]["D2"]["ddim_10d_mse"] if "task_macro" in ag else float("nan"),
                "D4_mse10_task_macro": ag["task_macro"]["D4"]["ddim_10d_mse"] if "task_macro" in ag else float("nan"),
                "attn_entropy": at["entropy_mean"],
                "attn_norm_entropy": at["norm_entropy_mean"],
                "attn_eff_pos": at["eff_pos_mean"],
                "top1_weight": at["top1_weight_mean"],
                "support_size": at["support_size_mean"],
                "captured_raw_mass": at["captured_raw_mass_mean"],
                "eff_top2_adj_rate": at["eff_top2_adj_rate"],
                "base_top2_adj_rate": at["base_top2_adj_rate"],
                "base_top2_dist_mean": at["base_top2_dist_mean"],
                "base_top2_dist_ge3_rate": at["base_top2_dist_ge3_rate"],
                "base_top3_contig_rate": at["base_top3_contig_rate"],
                "base_nonlocal_mass_out3": at["base_nonlocal_mass_out3_mean"],
                "exact_hit_rate": al["exact_hit_rate"],
                "within1_hit_rate": al["within1_hit_rate"],
                "window_contains_ref_rate": al["window_contains_ref_rate"],
                "ref_weight": al["ref_weight_mean"],
                "peak_err_slots": al["peak_err_slots_mean"],
                "peak_err_steps": al["peak_err_steps_mean"],
                "exp_err_slots": al["exp_err_slots_mean"],
                "exp_err_steps": al["exp_err_steps_mean"],
                "exp_abs_dist_slots": al["exp_abs_dist_slots_mean"],
                "signed_peak_slots": al["signed_peak_slots_mean"],
                "peak_norm_time_err": al["peak_norm_time_err_mean"],
                "seg_phase_hit_rate": al["seg_phase_hit_peak_rate"],
                "seg_phase_mass": al["seg_phase_mass_mean"],
                "sem_phase_hit_rate": al["sem_phase_hit_peak_rate"],
                "sem_phase_mass": al["sem_phase_mass_mean"],
                "peak_jump_abs_mean": jump_agg["mean_abs_peak_step"] if jump_agg else float("nan"),
                "peak_jump_excess_mean": jump_agg["mean_abs_peak_jump_excess"] if jump_agg else float("nan"),
                "peak_jump_ge3_rate": jump_agg["peak_jump_ge3_rate"] if jump_agg else float("nan"),
                "peak_jump_ge5_rate": jump_agg["peak_jump_ge5_rate"] if jump_agg else float("nan"),
                "backward_jump_rate": jump_agg["backward_jump_rate"] if jump_agg else float("nan"),
                "exp_jump_abs_mean": jump_agg["mean_abs_exp_step"] if jump_agg else float("nan"),
                "exp_jump_ge3_rate": jump_agg["exp_jump_ge3_rate"] if jump_agg else float("nan"),
            }
            main_table_rows.append(t_row)

            # Phase CSV rows
            for ph, pb in phases.items():
                if pb.get("num_anchors", 0) == 0:
                    continue
                phase_table_rows.append(
                    {
                        "split": split_name,
                        "reader": r_id,
                        "phase": ph,
                        "num_anchors": pb["num_anchors"],
                        "frac": pb["frac"],
                        "D1_mse10": pb["D1"]["ddim_10d_mse"],
                        "D2_mse10": pb["D2"]["ddim_10d_mse"],
                        "D3_mse10": pb["D3"]["ddim_10d_mse"],
                        "D4_mse10": pb["D4"]["ddim_10d_mse"],
                        "D2_lz": pb["D2"]["latent_lz_vs_self"],
                        "D4_lz": pb["D4"]["latent_lz_vs_self"],
                        "OracleAlign_D2_mse10": pb["OracleAlign_D2"]["ddim_10d_mse"] if "OracleAlign_D2" in pb else float("nan"),
                        "OracleAlign_D4_mse10": pb["OracleAlign_D4"]["ddim_10d_mse"] if "OracleAlign_D4" in pb else float("nan"),
                        "entropy": pb["attention"]["entropy_mean"],
                        "top1_weight": pb["attention"]["top1_weight_mean"],
                        "base_top2_adj_rate": pb["attention"]["base_top2_adj_rate"],
                        "exact_hit_rate": pb["alignment"]["exact_hit_rate"],
                        "within1_hit_rate": pb["alignment"]["within1_hit_rate"],
                        "peak_err_slots": pb["alignment"]["peak_err_slots_mean"],
                        "exp_err_slots": pb["alignment"]["exp_err_slots_mean"],
                        "seg_phase_hit_rate": pb["alignment"]["seg_phase_hit_peak_rate"],
                        "seg_phase_mass": pb["alignment"]["seg_phase_mass_mean"],
                        "sem_phase_hit_rate": pb["alignment"]["sem_phase_hit_peak_rate"],
                        "sem_phase_mass": pb["alignment"]["sem_phase_mass_mean"],
                    }
                )

            # Stratification CSV rows
            for st_name, sb in strat.items():
                if sb.get("num_anchors", 0) == 0:
                    continue
                strat_table_rows.append(
                    {"split": split_name, "reader": r_id, "stratum": st_name, **sb}
                )

            # Per-task unseen CSV rows
            if split_name in ("unseen_cross_traj", "self_oracle_gt") and "per_task" in ag:
                for task_name, ta in ag["per_task"].items():
                    per_task_unseen_rows.append(
                        {
                            "split": split_name,
                            "reader": r_id,
                            "task": task_name,
                            "num_anchors": ta["num_anchors"],
                            "D1_mse10": ta["D1"]["ddim_10d_mse"],
                            "D2_mse10": ta["D2"]["ddim_10d_mse"],
                            "D3_mse10": ta["D3"]["ddim_10d_mse"],
                            "D4_mse10": ta["D4"]["ddim_10d_mse"],
                            "D2_lz": ta["D2"]["latent_lz_vs_self"],
                            "D4_lz": ta["D4"]["latent_lz_vs_self"],
                            "entropy": ta["attention"]["entropy_mean"],
                            "top1_weight": ta["attention"]["top1_weight_mean"],
                            "base_top2_adj_rate": ta["attention"]["base_top2_adj_rate"],
                            "exact_hit_rate": ta["alignment"]["exact_hit_rate"],
                            "peak_err_slots": ta["alignment"]["peak_err_slots_mean"],
                            "seg_phase_hit_rate": ta["alignment"]["seg_phase_hit_peak_rate"],
                        }
                    )

        summary_splits[split_name] = split_block

    # Reproducibility cross-check against saved E9 baselines (P0 & D1-D4)
    repro_checks: Dict[str, Any] = {}
    saved_d1d4_path = OUTPUT_ROOT / "stage2_direct/d1_d4_diagnosis/offline_summary.json"
    if saved_d1d4_path.is_file():
        saved = json.loads(saved_d1d4_path.read_text())
        for dt_s, sp_s in (("default", "self_default"), ("shifted", "self_shifted"), ("oracle_gt", "self_oracle_gt")):
            if sp_s in summary_splits and dt_s in saved["by_demo_type_micro"]:
                ref_m = saved["by_demo_type_micro"][dt_s]["modes"]
                cur_m = summary_splits[sp_s]["readers"]["global_soft"]["aggregate"]
                repro_checks[f"{sp_s}_vs_saved_e9_d1_d4"] = {
                    "D1_abs_diff": abs(cur_m["D1"]["ddim_10d_mse"] - ref_m["D1_teacher"]["ddim_10d_mse"]),
                    "D2_abs_diff": abs(cur_m["D2"]["ddim_10d_mse"] - ref_m["D2_retrieval"]["ddim_10d_mse"]),
                    "D3_abs_diff": abs(cur_m["D3"]["ddim_10d_mse"] - ref_m["D3_transfer"]["ddim_10d_mse"]),
                    "D4_abs_diff": abs(cur_m["D4"]["ddim_10d_mse"] - ref_m["D4_full"]["ddim_10d_mse"]),
                }

    # Location vs Mixing quantitative decomposition summary
    mech_decomp: Dict[str, Any] = {}
    for sp_name in ("seen_cross_p0", "unseen_cross_p0", "unseen_cross_traj", "self_default", "self_oracle_gt"):
        if sp_name not in summary_splits:
            continue
        rds = summary_splits[sp_name]["readers"]
        gs = rds["global_soft"]["aggregate"]
        t1 = rds["global_top1"]["aggregate"]
        t2 = rds["global_top2"]["aggregate"]
        a2 = rds["adjacent_top2"]["aggregate"]
        a3 = rds["adjacent_top3"]["aggregate"]
        st_gs = rds["global_soft"]["stratifications"]
        st_t1 = rds["global_top1"]["stratifications"]
        st_a2 = rds["adjacent_top2"]["stratifications"]

        ref_d2 = gs["OracleAlign_D2"]["ddim_10d_mse"] if "OracleAlign_D2" in gs else gs["D1"]["ddim_10d_mse"]
        loc_err_d2 = t1["D2"]["ddim_10d_mse"] - ref_d2
        mix_delta_d2 = gs["D2"]["ddim_10d_mse"] - t1["D2"]["ddim_10d_mse"]
        adj2_vs_soft_d2 = a2["D2"]["ddim_10d_mse"] - gs["D2"]["ddim_10d_mse"]
        adj2_vs_t2_d2 = a2["D2"]["ddim_10d_mse"] - t2["D2"]["ddim_10d_mse"]
        adj2_vs_soft_d4 = a2["D4"]["ddim_10d_mse"] - gs["D4"]["ddim_10d_mse"]
        t1_vs_soft_d4 = t1["D4"]["ddim_10d_mse"] - gs["D4"]["ddim_10d_mse"]

        mech_decomp[sp_name] = {
            "ref_single_slot_D2_mse10": ref_d2,
            "global_top1_D2_mse10": t1["D2"]["ddim_10d_mse"],
            "global_soft_D2_mse10": gs["D2"]["ddim_10d_mse"],
            "adjacent_top2_D2_mse10": a2["D2"]["ddim_10d_mse"],
            "adjacent_top3_D2_mse10": a3["D2"]["ddim_10d_mse"],
            "location_error_D2_top1_minus_oracle": loc_err_d2,
            "mixing_effect_D2_soft_minus_top1": mix_delta_d2,
            "adj2_minus_soft_D2": adj2_vs_soft_d2,
            "adj2_minus_global_top2_D2": adj2_vs_t2_d2,
            "adj2_minus_soft_D4": adj2_vs_soft_d4,
            "top1_minus_soft_D4": t1_vs_soft_d4,
            "hit_pure_mixing_penalty_soft_minus_top1_D2": (
                st_gs["base_exact_hit"]["D2_mse10"] - st_t1["base_exact_hit"]["D2_mse10"]
                if st_gs["base_exact_hit"].get("num_anchors", 0) > 0
                else float("nan")
            ),
            "miss_mixing_effect_soft_minus_top1_D2": (
                st_gs["base_miss_exact"]["D2_mse10"] - st_t1["base_miss_exact"]["D2_mse10"]
                if st_gs["base_miss_exact"].get("num_anchors", 0) > 0
                else float("nan")
            ),
            "non_adj_top2_frac": st_gs["base_top2_non_adjacent"].get("frac", 0.0),
            "non_adj_adj2_minus_soft_D2": (
                st_a2["base_top2_non_adjacent"]["D2_mse10"] - st_gs["base_top2_non_adjacent"]["D2_mse10"]
                if st_gs["base_top2_non_adjacent"].get("num_anchors", 0) > 0
                else float("nan")
            ),
            "non_adj_adj2_minus_soft_D4": (
                st_a2["base_top2_non_adjacent"]["D4_mse10"] - st_gs["base_top2_non_adjacent"]["D4_mse10"]
                if st_gs["base_top2_non_adjacent"].get("num_anchors", 0) > 0
                else float("nan")
            ),
        }

    # Write CSV files
    _write_csv(main_table_rows, out_dir / "reader_comparison_summary.csv")
    _write_csv(phase_table_rows, out_dir / "phase_breakdown.csv")
    _write_csv(strat_table_rows, out_dir / "stratification_location_vs_mixing.csv")
    _write_csv(per_task_unseen_rows, out_dir / "per_task_unseen.csv")
    _write_csv(per_traj_jump_rows, out_dir / "peak_jump_per_trajectory.csv")

    # Write per-anchor compact CSVs for Seen-Cross, Unseen-Cross, and Self D1-D4
    compact_cols = [
        "split", "reader", "task", "traj_id", "anchor_idx", "q_step", "ref_j", "mem_len",
        "q_seg_phase", "q_sem_phase", "exact_phase_match", "in_p0_subset",
        "base_top1_idx", "base_top1_weight", "base_top2_adj", "base_top2_dist", "base_exact_hit",
        "peak_j", "exp_j", "top1_weight", "entropy", "norm_entropy", "captured_raw_mass",
        "exact_hit", "within1_hit", "peak_err_slots", "exp_err_slots",
        "seg_phase_hit_peak", "seg_phase_mass", "sem_phase_hit_peak", "sem_phase_mass",
        "D1_mse10_masked", "D2_mse10_masked", "D3_mse10_masked", "D4_mse10_masked",
        "D2_latent_mse", "D4_latent_mse",
    ]
    for fname, sp_keys in (
        ("per_anchor_seen_cross.csv", ("seen_cross_p0", "seen_cross_traj")),
        ("per_anchor_unseen_cross.csv", ("unseen_cross_traj",)),
        ("per_anchor_self_d1_d4.csv", ("self_default", "self_shifted", "self_oracle_gt")),
    ):
        flat_rows = []
        for sk in sp_keys:
            if sk not in all_split_results:
                continue
            for m in UNIQUE_READER_IDS:
                for r in all_split_results[sk]["per_mode_rows"][m]:
                    flat_rows.append({c: r.get(c, "") for c in compact_cols})
        if flat_rows:
            _write_csv(flat_rows, out_dir / fname)

    # Save heatmap matrices & metadata cleanly to npz + json index
    hm_index: List[Dict[str, Any]] = []
    hm_arrays: Dict[str, np.ndarray] = {}
    for hi, hm in enumerate(all_heatmaps):
        prefix = f"hm_{hi:02d}"
        hm_index.append(
            {
                "id": prefix,
                "split": hm["split"],
                "traj_id": hm["traj_id"],
                "task": hm["task"],
                "q_ep": hm["q_ep"],
                "s_ep": hm["s_ep"],
                "M_q": hm["M_q"],
                "M_s": hm["M_s"],
                "q_sem_phases": hm["q_sem_phases"],
                "gs_d2_mean": hm["gs_d2_mean"],
                "gs_d4_mean": hm["gs_d4_mean"],
            }
        )
        hm_arrays[f"{prefix}_ref_js"] = hm["ref_js"]
        hm_arrays[f"{prefix}_q_anchors"] = hm["q_anchors"]
        hm_arrays[f"{prefix}_s_anchors"] = hm["s_anchors"]
        hm_arrays[f"{prefix}_q_flips"] = hm["q_flips"]
        hm_arrays[f"{prefix}_s_flips"] = hm["s_flips"]
        for mode, mat in hm["mats"].items():
            hm_arrays[f"{prefix}_w_{mode}"] = mat
        for mode, arr in hm["mode_d2"].items():
            hm_arrays[f"{prefix}_d2_{mode}"] = arr
        for mode, arr in hm["mode_d4"].items():
            hm_arrays[f"{prefix}_d4_{mode}"] = arr

    np.savez_compressed(out_dir / "heatmaps_data.npz", **hm_arrays)
    atomic_write_json({"heatmaps": hm_index}, out_dir / "heatmaps_index.json")

    elapsed = round(time.monotonic() - t0, 2)
    full_summary: Dict[str, Any] = {
        "experiment": "e9_reader_inference_only_diagnosis",
        "protocol": {
            "all_modules_frozen": True,
            "trainable_params": 0,
            "no_ste": True,
            "no_retraining": True,
            "offline_only_no_closed_loop": True,
            "only_reader_readout_modified": True,
            "shared_attention_logits_transfer_decoder_ddim_noise": True,
            "seed": int(seed),
            "ddim_seed": int(ddim_seed),
            "use_bf16": bool(use_bf16 and device.type == "cuda"),
            "s_z": s_z,
            "behavior_stride": int(BEHAVIOR_STRIDE),
            "action_horizon": int(ACTION_HORIZON),
            "exec_horizon": int(EXEC_HORIZON),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "checkpoints": {
            "e9_stage1_local": str(Path(stage1_ckpt).resolve()),
            "e9_stage2_direct": str(Path(stage2_ckpt).resolve()),
            "e9_stage2_epoch": payload.get("epoch"),
        },
        "reader_specs": [
            {"id": r[0], "family": r[1], "label": r[2], "tau": r[3]} for r in READER_SPECS
        ],
        "reproducibility_checks": repro_checks,
        "mechanism_decomposition": mech_decomp,
        "splits": summary_splits,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(full_summary, out_dir / "summary.json")

    print(f"\n[E9 Reader Diag DONE] Saved all CSV/JSON/NPZ to {out_dir} in {elapsed}s", flush=True)
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stage1-ckpt", type=Path, default=E9_STAGE1_CKPT)
    parser.add_argument("--stage2-ckpt", type=Path, default=E9_STAGE2_CKPT)
    parser.add_argument("--n-seen-p0", type=int, default=3000)
    parser.add_argument("--n-seen-trajs", type=int, default=200)
    parser.add_argument("--n-unseen-p0", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-seed", type=int, default=12345)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--skip-oracle-gt", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_e9_reader_diagnosis(
        out_dir=args.out_dir,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        n_seen_p0=args.n_seen_p0,
        n_seen_trajs=args.n_seen_trajs,
        n_unseen_p0=args.n_unseen_p0,
        seed=args.seed,
        ddim_seed=args.ddim_seed,
        batch_size=args.batch_size,
        use_bf16=not args.no_bf16,
        skip_oracle_gt=args.skip_oracle_gt,
        device=device,
    )


if __name__ == "__main__":
    main()
