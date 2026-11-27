"""P0: Low-dimensional Behavior Transfer Diagnosis (Frozen E7 vs E9).

Executes:
  1. Residual pair construction (Train 8k, Seen-Val 3k, Unseen-Chain 3k) under both
     Offline Phase Alignment (trajectory + gripper events) and Frozen Reader Alignment.
  2. Strict temporal alignment quality & anti-leakage audit.
  3. Subspace SVD analysis: un-centered SVD on train residuals, comparison with raw &
     centered behavior latents, centering/scaling/coordinate-standardization checks,
     effective ranks, and generalization to independent Seen & Unseen splits.
  4. Oracle Low-rank Transfer via frozen Action Decoder (identical DDIM settings & noise)
     for K in {0, 8, 16, 32, 64, 128, 256, Full} + nested random orthogonal control +
     frozen Stage 2 Direct (D4) reference.

No training, no architecture modification, no closed-loop evaluation.
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

import numpy as np
import torch
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e7.diagnose_behavior_space import _stable_seed
from e7.model import compute_attention_stats, load_e7_policy
from e9.stage2_direct_model import load_e9_stage2_direct_policy

from . import (
    ACTION_10D_NAMES,
    ACTION_DIM_10D,
    ACTION_HORIZON,
    E5_CACHE_DIR,
    E7_BEHAVIOR_MEM_DIR,
    E7_STAGE1_CKPT,
    E7_STAGE2_CKPT,
    E9_BEHAVIOR_MEM_DIR,
    E9_STAGE1_CKPT,
    E9_STAGE2_CKPT,
    EXEC_HORIZON,
    LATENT_DIM,
    LATENT_TOKENS,
    EMBED_DIM,
    NONTRIVIAL_RANKS,
    OUTPUT_ROOT,
    RANK_GRID,
)
from .alignment import (
    align_demo_pair_offline,
    sample_cached_split_pairs,
    sample_unseen_chain_pairs,
)
from .cache_unseen import DEFAULT_UNSEEN_CACHE_DIR, build_unseen_chain_cache

ENERGY_THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)
SPECTRUM_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, LATENT_DIM)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compute_svd_stats(X: np.ndarray, *, center: bool = False) -> Dict[str, Any]:
    """Compute exact float64 thin SVD and spectral/effective-rank statistics for X (N, D)."""
    Xf = np.asarray(X, dtype=np.float64)
    N, D = Xf.shape
    mean_vec = Xf.mean(axis=0)
    total_sq_raw = float(np.sum(Xf * Xf))
    mean_sq = float(N * np.sum(mean_vec * mean_vec))
    mean_energy_frac = mean_sq / max(total_sq_raw, 1e-18)

    M = Xf - mean_vec[None, :] if center else Xf
    # Full thin SVD: M = U @ diag(S) @ Vt; Vt rows are orthonormal right singular vectors
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    eig = S * S
    tot_eig = float(np.sum(eig))
    p = eig / max(tot_eig, 1e-18)
    q = S / max(float(np.sum(S)), 1e-18)
    cum_e = np.cumsum(p)

    # Effective ranks
    p_pos = p[p > 1e-18]
    q_pos = q[q > 1e-18]
    erank_energy = float(math.exp(-float(np.sum(p_pos * np.log(p_pos)))))
    erank_sv = float(math.exp(-float(np.sum(q_pos * np.log(q_pos)))))
    participation_ratio = float((tot_eig * tot_eig) / max(float(np.sum(eig * eig)), 1e-30))

    dims_at = {}
    for thr in ENERGY_THRESHOLDS:
        idx = int(np.searchsorted(cum_e, thr))
        dims_at[f"d_{int(round(thr * 100))}"] = int(min(idx + 1, len(S)))

    cum_at = {}
    for k in SPECTRUM_KS:
        kk = min(k, len(S))
        cum_at[f"cum_e_{k}"] = float(cum_e[kk - 1])

    rms = float(math.sqrt(float(np.mean(M * M))))
    rms_raw = float(math.sqrt(total_sq_raw / max(N * D, 1)))
    mean_rms = float(math.sqrt(float(np.mean(mean_vec * mean_vec))))

    # Token-level (4 tokens x 768) energy fractions of M
    M_tok = M.reshape(N, LATENT_TOKENS, EMBED_DIM)
    tok_e = np.sum(M_tok * M_tok, axis=(0, 2))
    tok_frac = (tok_e / max(float(np.sum(tok_e)), 1e-18)).tolist()

    return {
        "N": int(N),
        "D": int(D),
        "centered": bool(center),
        "rms": rms,
        "rms_raw": rms_raw,
        "mean_rms": mean_rms,
        "mean_energy_frac": float(mean_energy_frac),
        "sigma_max": float(S[0]),
        "sigma_10": float(S[min(9, len(S) - 1)]),
        "sigma_64": float(S[min(63, len(S) - 1)]),
        "sigma_256": float(S[min(255, len(S) - 1)]),
        "erank_energy": erank_energy,
        "erank_sv": erank_sv,
        "participation_ratio": participation_ratio,
        "dims_at_threshold": dims_at,
        "cum_energy_at_k": cum_at,
        "token_energy_frac": [float(x) for x in tok_frac],
        # Arrays (kept in memory, stripped before JSON dump)
        "_S": S,
        "_Vt": Vt,
        "_cum_e": cum_e,
    }


def _strip_arrays(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _energy_recovery(delta_mat: np.ndarray, U_K: np.ndarray) -> float:
    """Compute ||delta_mat @ U_K||_F^2 / ||delta_mat||_F^2 in float64."""
    X = np.asarray(delta_mat, dtype=np.float64)
    tot = float(np.sum(X * X))
    if tot <= 1e-18:
        return 0.0
    if U_K.shape[1] == 0:
        return 0.0
    proj = X @ np.asarray(U_K, dtype=np.float64)
    return float(np.sum(proj * proj) / tot)


def _subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """Average squared cosine of principal angles: ||U1^T U2||_F^2 / K in [0, 1]."""
    K = U1.shape[1]
    if K == 0:
        return 0.0
    M = np.asarray(U1, dtype=np.float64).T @ np.asarray(U2, dtype=np.float64)
    return float(np.sum(M * M) / float(K))


def _make_nested_random_ortho(dim: int, max_k: int, seed: int) -> np.ndarray:
    """Generate Haar-random orthonormal columns Q in R^(dim x max_k) via QR."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, max_k), dtype=np.float64)
    Q, R = np.linalg.qr(G, mode="reduced")
    # Fix column signs for exact Haar invariance
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs[None, :]
    return Q


def _extract_gt_chunk(
    actions_10d: np.ndarray, ep_start: int, ep_len: int, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (chunk_10d (16, 10), valid_16 (16,)) with repeat-last tail padding."""
    grid = int(step) + np.arange(ACTION_HORIZON, dtype=np.int64)
    valid = grid < int(ep_len)
    clipped = np.minimum(grid, int(ep_len) - 1)
    chunk = actions_10d[int(ep_start) + clipped].astype(np.float32, copy=True)
    return chunk, valid


def _compute_alignment_and_gt_for_split(
    split_name: str,
    pairs: List[Dict[str, Any]],
    *,
    actions_10d: np.ndarray,
    ee_pos_all: np.ndarray,
    action_offsets: np.ndarray,
    mem_anchor_steps: np.ndarray,
    mem_offsets: np.ndarray,
    mem_lengths: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """Compute offline phase alignment, GT action chunks, and alignment quality audit
    for a split (`train`, `seen_val`, or `unseen_chain`). Shared identically by E7 & E9."""
    N = len(pairs)
    pair_align_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    q_steps = np.empty(N, dtype=np.int32)
    s_offline_j = np.empty(N, dtype=np.int32)
    s_offline_step = np.empty(N, dtype=np.int32)
    quant_errs = np.empty(N, dtype=np.float64)
    norm_time_diffs = np.empty(N, dtype=np.float64)
    exact_phase = np.empty(N, dtype=bool)
    methods: List[str] = []

    gt_chunks = np.empty((N, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
    valid_16 = np.empty((N, ACTION_HORIZON), dtype=bool)

    anchor_grip_agree = np.empty(N, dtype=np.float64)
    chunk_grip_agree = np.empty(N, dtype=np.float64)
    pair_action_mse = np.empty(N, dtype=np.float64)

    # Random source anchor baseline for sanity-checking alignment quality
    rng = np.random.default_rng(seed + 555)
    rand_grip_agree = np.empty(N, dtype=np.float64)
    rand_action_mse = np.empty(N, dtype=np.float64)

    for i, p in enumerate(pairs):
        q_ep = int(p["q_ep"])
        s_ep = int(p["s_ep"])
        k = int(p["q_anchor_idx"])

        key = (q_ep, s_ep)
        if key not in pair_align_cache:
            q_a0, q_a1 = int(action_offsets[q_ep]), int(action_offsets[q_ep + 1])
            s_a0, s_a1 = int(action_offsets[s_ep]), int(action_offsets[s_ep + 1])
            q_m0, q_m1 = int(mem_offsets[q_ep]), int(mem_offsets[q_ep] + mem_lengths[q_ep])
            s_m0, s_m1 = int(mem_offsets[s_ep]), int(mem_offsets[s_ep] + mem_lengths[s_ep])
            pair_align_cache[key] = align_demo_pair_offline(
                ee_pos_q=ee_pos_all[q_a0:q_a1],
                grip_q=actions_10d[q_a0:q_a1, 9],
                ee_pos_s=ee_pos_all[s_a0:s_a1],
                grip_s=actions_10d[s_a0:s_a1, 9],
                anchors_q=mem_anchor_steps[q_m0:q_m1],
                anchors_s=mem_anchor_steps[s_m0:s_m1],
            )
        al = pair_align_cache[key]

        q_a0, q_a1 = int(action_offsets[q_ep]), int(action_offsets[q_ep + 1])
        s_a0, s_a1 = int(action_offsets[s_ep]), int(action_offsets[s_ep + 1])
        Tq = q_a1 - q_a0
        Ts = s_a1 - s_a0

        q_step = int(mem_anchor_steps[int(mem_offsets[q_ep]) + k])
        j_star = int(al["matched_j"][k])
        s_step = int(mem_anchor_steps[int(mem_offsets[s_ep]) + j_star])

        q_steps[i] = q_step
        s_offline_j[i] = j_star
        s_offline_step[i] = s_step
        quant_errs[i] = float(al["quant_err_steps"][k])
        norm_time_diffs[i] = abs(float(q_step) / max(Tq - 1, 1) - float(s_step) / max(Ts - 1, 1))
        exact_phase[i] = bool(al["exact_phase_match"])
        methods.append(al["method"])

        gq, vq = _extract_gt_chunk(actions_10d, q_a0, Tq, q_step)
        gs, vs = _extract_gt_chunk(actions_10d, s_a0, Ts, s_step)
        gt_chunks[i] = gq
        valid_16[i] = vq

        v_both = vq & vs
        if not np.any(v_both):
            v_both = vq
        anchor_grip_agree[i] = float((gq[0, 9] >= 0.0) == (gs[0, 9] >= 0.0))
        chunk_grip_agree[i] = float(np.mean((gq[v_both, 9] >= 0.0) == (gs[v_both, 9] >= 0.0)))
        pair_action_mse[i] = float(np.mean((gq[v_both] - gs[v_both]) ** 2))

        # Random anchor comparison
        j_rand = int(rng.integers(int(mem_lengths[s_ep])))
        s_step_r = int(mem_anchor_steps[int(mem_offsets[s_ep]) + j_rand])
        gr, vr = _extract_gt_chunk(actions_10d, s_a0, Ts, s_step_r)
        v_br = vq & vr
        if not np.any(v_br):
            v_br = vq
        rand_grip_agree[i] = float(np.mean((gq[v_br, 9] >= 0.0) == (gr[v_br, 9] >= 0.0)))
        rand_action_mse[i] = float(np.mean((gq[v_br] - gr[v_br]) ** 2))

    method_counts = defaultdict(int)
    for m in methods:
        method_counts[m] += 1

    audit = {
        "split": split_name,
        "num_pairs": int(N),
        "unique_demo_pairs": int(len(pair_align_cache)),
        "exact_gripper_phase_match_rate": float(np.mean(exact_phase)),
        "method_fractions": {k: float(v) / max(N, 1) for k, v in sorted(method_counts.items())},
        "quant_err_steps_mean": float(np.mean(quant_errs)),
        "quant_err_steps_median": float(np.median(quant_errs)),
        "quant_err_steps_p95": float(np.percentile(quant_errs, 95)),
        "norm_time_diff_mean": float(np.mean(norm_time_diffs)),
        "anchor_grip_agree_rate": float(np.mean(anchor_grip_agree)),
        "chunk16_grip_agree_rate": float(np.mean(chunk_grip_agree)),
        "gt_action_10d_mse_aligned": float(np.mean(pair_action_mse)),
        "random_anchor_chunk16_grip_agree": float(np.mean(rand_grip_agree)),
        "random_anchor_gt_action_10d_mse": float(np.mean(rand_action_mse)),
    }

    return {
        "q_steps": q_steps,
        "s_offline_j": s_offline_j,
        "s_offline_step": s_offline_step,
        "gt_chunks": gt_chunks,
        "valid_16": valid_16,
        "audit": audit,
    }


@torch.inference_mode()
def _extract_model_latents_for_split(
    policy: torch.nn.Module,
    pairs: List[Dict[str, Any]],
    split_data: Dict[str, Any],
    *,
    mem_h: np.ndarray,
    mem_b: np.ndarray,
    mem_anchor_steps: np.ndarray,
    mem_offsets: np.ndarray,
    mem_lengths: np.ndarray,
    actions_10d: np.ndarray,
    action_offsets: np.ndarray,
    device: torch.device,
    use_bf16: bool,
    batch_size: int = 512,
    compute_d4: bool = False,
) -> Dict[str, Any]:
    """Extract z_query, z_source_offline, z_source_reader (+ optional z_d4) and Reader audit."""
    N = len(pairs)
    s_offline_j = split_data["s_offline_j"]
    gt_chunks = split_data["gt_chunks"]
    valid_16 = split_data["valid_16"]

    z_query = np.empty((N, LATENT_DIM), dtype=np.float32)
    z_src_off = np.empty((N, LATENT_DIM), dtype=np.float32)
    z_src_rdr = np.empty((N, LATENT_DIM), dtype=np.float32)
    z_d4 = np.empty((N, LATENT_DIM), dtype=np.float32) if compute_d4 else None

    max_w_all = np.empty(N, dtype=np.float32)
    norm_ent_all = np.empty(N, dtype=np.float32)
    eff_pos_all = np.empty(N, dtype=np.float32)
    argmax_j_all = np.empty(N, dtype=np.int32)
    slot_diff_all = np.empty(N, dtype=np.int32)
    step_diff_all = np.empty(N, dtype=np.int32)
    rdr_argmax_grip_agree = np.empty(N, dtype=np.float64)
    rdr_argmax_act_mse = np.empty(N, dtype=np.float64)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        batch_pairs = pairs[start:end]

        s_eps = [int(p["s_ep"]) for p in batch_pairs]
        lens = [int(mem_lengths[ep]) for ep in s_eps]
        max_m = max(lens)

        h_q_np = np.empty((B, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        h_mem_np = np.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        b_mem_np = np.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        pad_mask_np = np.ones((B, max_m), dtype=bool)

        for bi, p in enumerate(batch_pairs):
            q_ep = int(p["q_ep"])
            s_ep = int(p["s_ep"])
            k = int(p["q_anchor_idx"])
            j_off = int(s_offline_j[start + bi])

            q_flat = int(mem_offsets[q_ep]) + k
            s_m0 = int(mem_offsets[s_ep])
            m_len = int(mem_lengths[s_ep])

            z_query[start + bi] = mem_b[q_flat].astype(np.float32).reshape(LATENT_DIM)
            z_src_off[start + bi] = mem_b[s_m0 + j_off].astype(np.float32).reshape(LATENT_DIM)
            h_q_np[bi] = mem_h[q_flat].astype(np.float32)
            h_mem_np[bi, :m_len] = mem_h[s_m0 : s_m0 + m_len].astype(np.float32)
            b_mem_np[bi, :m_len] = mem_b[s_m0 : s_m0 + m_len].astype(np.float32)
            pad_mask_np[bi, :m_len] = False

        h_q_t = torch.from_numpy(h_q_np).to(device=device, dtype=torch.float32)
        h_mem_t = torch.from_numpy(h_mem_np).to(device=device, dtype=torch.float32)
        b_mem_t = torch.from_numpy(b_mem_np).to(device=device, dtype=torch.float32)
        pad_t = torch.from_numpy(pad_mask_np).to(device=device, dtype=torch.bool)

        with amp_ctx:
            z_c, attn_w = policy.reader(h_q_t, h_mem_t, b_mem_t, mem_pad_mask=pad_t)
            if compute_d4:
                z_d4_t, _ = policy.adapt_latent(z_c, h_q_t, delta_scale=1.0)
                z_d4[start:end] = z_d4_t.float().cpu().numpy().reshape(B, LATENT_DIM)

        z_src_rdr[start:end] = z_c.float().cpu().numpy().reshape(B, LATENT_DIM)
        ast = compute_attention_stats(attn_w, mem_pad_mask=pad_t)
        max_w_all[start:end] = ast["max_weight"].cpu().numpy()
        norm_ent_all[start:end] = ast["norm_entropy"].cpu().numpy()
        eff_pos_all[start:end] = ast["eff_pos_exp"].cpu().numpy()

        w_np = attn_w.float().cpu().numpy()
        for bi, p in enumerate(batch_pairs):
            idx = start + bi
            s_ep = int(p["s_ep"])
            m_len = int(mem_lengths[s_ep])
            am_j = int(np.argmax(w_np[bi, :m_len]))
            j_off = int(s_offline_j[idx])
            argmax_j_all[idx] = am_j
            slot_diff_all[idx] = abs(am_j - j_off)

            s_m0 = int(mem_offsets[s_ep])
            s_step_am = int(mem_anchor_steps[s_m0 + am_j])
            s_step_off = int(mem_anchor_steps[s_m0 + j_off])
            step_diff_all[idx] = abs(s_step_am - s_step_off)

            s_a0, s_a1 = int(action_offsets[s_ep]), int(action_offsets[s_ep + 1])
            g_am, v_am = _extract_gt_chunk(actions_10d, s_a0, s_a1 - s_a0, s_step_am)
            vb = valid_16[idx] & v_am
            if not np.any(vb):
                vb = valid_16[idx]
            rdr_argmax_grip_agree[idx] = float(
                np.mean((gt_chunks[idx, vb, 9] >= 0.0) == (g_am[vb, 9] >= 0.0))
            )
            rdr_argmax_act_mse[idx] = float(np.mean((gt_chunks[idx, vb] - g_am[vb]) ** 2))

    reader_audit = {
        "max_weight_mean": float(np.mean(max_w_all)),
        "norm_entropy_mean": float(np.mean(norm_ent_all)),
        "eff_pos_mean": float(np.mean(eff_pos_all)),
        "slot_diff_vs_offline_mean": float(np.mean(slot_diff_all)),
        "step_diff_vs_offline_mean": float(np.mean(step_diff_all)),
        "exact_slot_match_vs_offline_rate": float(np.mean(slot_diff_all == 0)),
        "within1_slot_match_vs_offline_rate": float(np.mean(slot_diff_all <= 1)),
        "argmax_chunk16_grip_agree_rate": float(np.mean(rdr_argmax_grip_agree)),
        "argmax_gt_action_10d_mse": float(np.mean(rdr_argmax_act_mse)),
    }

    return {
        "z_query": z_query,
        "z_src_offline": z_src_off,
        "z_src_reader": z_src_rdr,
        "z_d4": z_d4,
        "reader_audit": reader_audit,
    }


def _build_init_noises(
    pairs: List[Dict[str, Any]], q_steps: np.ndarray, *, base_seed: int
) -> torch.Tensor:
    """Deterministic per-sample DDIM initial noise shared across E7/E9 and all K."""
    noises = []
    for p, t_k in zip(pairs, q_steps):
        key = f"{p['split']}/{p['task']}/q{p['q_ep']}_t{int(t_k)}_s{p['s_ep']}"
        s_k = _stable_seed(key, base_seed=base_seed)
        gen = torch.Generator(device="cpu").manual_seed(s_k)
        noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
    return torch.stack(noises, dim=0)  # (N, 16, 10) float32 CPU


@torch.inference_mode()
def _decode_and_score_latents(
    policy: torch.nn.Module,
    z_hat: np.ndarray,
    z_query: np.ndarray,
    z_source: np.ndarray,
    gt_chunks: np.ndarray,
    valid_16: np.ndarray,
    init_noise_cpu: torch.Tensor,
    tasks: List[str],
    *,
    s_z: float,
    device: torch.device,
    use_bf16: bool,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Decode `z_hat` (N, 3072) through frozen UNet with identical DDIM init_noise and
    compute latent MSE, residual energy recovery, overall + per-dim DDIM action MSE."""
    N = len(z_hat)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Latent metrics in float64 for accuracy
    zq64 = z_query.astype(np.float64)
    zs64 = z_source.astype(np.float64)
    zh64 = z_hat.astype(np.float64)
    err_z = zh64 - zq64
    res_z = zq64 - zs64
    lat_mse_raw = float(np.mean(err_z * err_z))
    lat_lz = float(lat_mse_raw / (s_z * s_z))
    res_tot_sq = float(np.sum(res_z * res_z))
    err_tot_sq = float(np.sum(err_z * err_z))
    energy_recovery = float(1.0 - err_tot_sq / max(res_tot_sq, 1e-18))

    zh_norm = np.linalg.norm(zh64, axis=-1)
    zq_norm = np.linalg.norm(zq64, axis=-1)
    lat_cos = float(
        np.mean(np.sum(zh64 * zq64, axis=-1) / np.maximum(zh_norm * zq_norm, 1e-12))
    )

    # Batched DDIM decoding
    tot_v16 = 0.0
    tot_v12 = 0.0
    sum_sq_per_dim = np.zeros(ACTION_DIM_10D, dtype=np.float64)
    sum_abs7 = 0.0
    sum_exec12_abs7 = 0.0
    sum_grip16_ok = 0.0
    sum_grip12_ok = 0.0

    # Per-task accumulators for macro average (especially unseen 10 tasks)
    task_v16: Dict[str, float] = defaultdict(float)
    task_sq10: Dict[str, float] = defaultdict(float)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        z_t = torch.from_numpy(
            z_hat[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)
        ).to(device=device, dtype=torch.float32)
        noise_t = init_noise_cpu[start:end].to(device=device, dtype=torch.float32)

        with amp_ctx:
            pred_10d = policy.teacher.sample_from_z_self(z_t, init_noise=noise_t).float()

        gt_t = torch.from_numpy(gt_chunks[start:end]).to(device=device, dtype=torch.float32)
        v16_t = torch.from_numpy(valid_16[start:end]).to(device=device, dtype=torch.float32)
        v12_t = v16_t[:, :EXEC_HORIZON]

        diff10 = pred_10d - gt_t  # (B, 16, 10)
        sq10 = diff10.square()  # (B, 16, 10)
        sum_sq_per_dim += (sq10 * v16_t.unsqueeze(-1)).sum(dim=(0, 1)).cpu().numpy()

        # Per-sample sums for task macro-average
        sample_v16 = v16_t.sum(dim=1).cpu().numpy()
        sample_sq10 = (sq10.mean(dim=-1) * v16_t).sum(dim=1).cpu().numpy()
        for bi in range(B):
            tk = tasks[start + bi]
            task_v16[tk] += float(sample_v16[bi])
            task_sq10[tk] += float(sample_sq10[bi])

        pred_7d = action_10d_to_7d(pred_10d)
        gt_7d = action_10d_to_7d(gt_t)
        abs7_step = (pred_7d - gt_7d).abs().mean(dim=-1)  # (B, 16)
        grip_ok = ((pred_7d[..., 6] >= 0.0) == (gt_7d[..., 6] >= 0.0)).float()

        tot_v16 += float(v16_t.sum().item())
        tot_v12 += float(v12_t.sum().item())
        sum_abs7 += float((abs7_step * v16_t).sum().item())
        sum_exec12_abs7 += float((abs7_step[:, :EXEC_HORIZON] * v12_t).sum().item())
        sum_grip16_ok += float((grip_ok * v16_t).sum().item())
        sum_grip12_ok += float((grip_ok[:, :EXEC_HORIZON] * v12_t).sum().item())

    per_dim_mse = (sum_sq_per_dim / max(tot_v16, 1.0)).tolist()
    ddim_10d_mse = float(np.mean(per_dim_mse))
    ddim_pos_mse = float(np.mean(per_dim_mse[:3]))
    ddim_rot6d_mse = float(np.mean(per_dim_mse[3:9]))
    ddim_grip_mse = float(per_dim_mse[9])

    task_mses = [task_sq10[t] / max(task_v16[t], 1.0) for t in sorted(task_v16.keys())]
    macro_10d_mse = float(np.mean(task_mses)) if task_mses else ddim_10d_mse

    return {
        "latent_mse_raw": lat_mse_raw,
        "latent_lz": lat_lz,
        "latent_cos": lat_cos,
        "residual_energy_recovery": energy_recovery,
        "ddim_10d_mse": ddim_10d_mse,
        "ddim_10d_mse_macro": macro_10d_mse,
        "ddim_pos_mse": ddim_pos_mse,
        "ddim_rot6d_mse": ddim_rot6d_mse,
        "ddim_grip_mse": ddim_grip_mse,
        "ddim_7d_mae": float(sum_abs7 / max(tot_v16, 1.0)),
        "exec12_7d_mae": float(sum_exec12_abs7 / max(tot_v12, 1.0)),
        "grip_acc16": float(sum_grip16_ok / max(tot_v16, 1.0)),
        "grip_acc12": float(sum_grip12_ok / max(tot_v12, 1.0)),
        "per_dim_mse": {
            ACTION_10D_NAMES[d]: float(per_dim_mse[d]) for d in range(ACTION_DIM_10D)
        },
    }


def _safe_arr(mse_0: float, mse_k: float, mse_full: float) -> Tuple[Optional[float], bool]:
    """Compute Action Recovery Ratio (MSE_0 - MSE_K) / (MSE_0 - MSE_Full) ONLY when
    denominator is clearly positive: > max(1e-5, 0.05 * MSE_Full)."""
    denom = float(mse_0 - mse_full)
    thresh = max(1e-5, 0.05 * float(mse_full))
    if denom > thresh:
        return float((mse_0 - mse_k) / denom), True
    return None, False


def _attach_arr(
    rows: List[Dict[str, Any]], rec_0: Dict[str, Any], rec_full: Dict[str, Any]
) -> None:
    """Attach Action Recovery Ratio (overall, pos, rot6d, grip, and per-dim) in place."""
    for r in rows:
        arr_all, ok_all = _safe_arr(rec_0["ddim_10d_mse"], r["ddim_10d_mse"], rec_full["ddim_10d_mse"])
        arr_macro, _ = _safe_arr(
            rec_0["ddim_10d_mse_macro"], r["ddim_10d_mse_macro"], rec_full["ddim_10d_mse_macro"]
        )
        arr_pos, _ = _safe_arr(rec_0["ddim_pos_mse"], r["ddim_pos_mse"], rec_full["ddim_pos_mse"])
        arr_rot, _ = _safe_arr(rec_0["ddim_rot6d_mse"], r["ddim_rot6d_mse"], rec_full["ddim_rot6d_mse"])
        arr_grip, _ = _safe_arr(rec_0["ddim_grip_mse"], r["ddim_grip_mse"], rec_full["ddim_grip_mse"])
        r["arr_10d"] = arr_all
        r["arr_10d_valid"] = ok_all
        r["arr_10d_denom"] = float(rec_0["ddim_10d_mse"] - rec_full["ddim_10d_mse"])
        r["arr_10d_macro"] = arr_macro
        r["arr_pos"] = arr_pos
        r["arr_rot6d"] = arr_rot
        r["arr_grip"] = arr_grip
        per_dim_arr = {}
        for name in ACTION_10D_NAMES:
            v, _ = _safe_arr(
                rec_0["per_dim_mse"][name],
                r["per_dim_mse"][name],
                rec_full["per_dim_mse"][name],
            )
            per_dim_arr[name] = v
        r["per_dim_arr"] = per_dim_arr


def _flatten_oracle_row(r: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "model": r["model"],
        "alignment": r["alignment"],
        "split": r["split"],
        "basis_type": r["basis_type"],
        "rank_k": r["rank_k"],
        "rank_label": r["rank_label"],
        "latent_mse_raw": r["latent_mse_raw"],
        "latent_lz": r["latent_lz"],
        "latent_cos": r["latent_cos"],
        "residual_energy_recovery": r["residual_energy_recovery"],
        "ddim_10d_mse": r["ddim_10d_mse"],
        "ddim_10d_mse_macro": r["ddim_10d_mse_macro"],
        "ddim_pos_mse": r["ddim_pos_mse"],
        "ddim_rot6d_mse": r["ddim_rot6d_mse"],
        "ddim_grip_mse": r["ddim_grip_mse"],
        "ddim_7d_mae": r["ddim_7d_mae"],
        "exec12_7d_mae": r["exec12_7d_mae"],
        "grip_acc16": r["grip_acc16"],
        "grip_acc12": r["grip_acc12"],
        "arr_10d": r["arr_10d"],
        "arr_10d_valid": r["arr_10d_valid"],
        "arr_10d_denom": r["arr_10d_denom"],
        "arr_10d_macro": r["arr_10d_macro"],
        "arr_pos": r["arr_pos"],
        "arr_rot6d": r["arr_rot6d"],
        "arr_grip": r["arr_grip"],
    }
    for name in ACTION_10D_NAMES:
        flat[f"mse_{name}"] = r["per_dim_mse"][name]
        flat[f"arr_{name}"] = r["per_dim_arr"][name]
    return flat


def run_p0_diagnosis(
    *,
    out_dir: Path = OUTPUT_ROOT,
    unseen_cache_dir: Path = DEFAULT_UNSEEN_CACHE_DIR,
    n_train: int = 8000,
    n_seen_val: int = 3000,
    n_unseen: int = 3000,
    demos_per_unseen_task: int = 20,
    seed: int = 42,
    ddim_seed: int = 12345,
    batch_size: int = 512,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure unseen cache exists
    build_unseen_chain_cache(
        out_dir=unseen_cache_dir,
        demos_per_task=demos_per_unseen_task,
        device=device,
        use_bf16=use_bf16,
    )

    # 2. Load shared metadata & trajectories
    print("[P0] Loading E5 cached metadata, actions, proprio, and unseen cache...", flush=True)
    episodes = json.loads((E5_CACHE_DIR / "episodes.json").read_text())
    actions_10d_e5 = np.load(E5_CACHE_DIR / "actions_10d.npy", mmap_mode="r")
    proprio_11d_e5 = np.load(E5_CACHE_DIR / "proprio_11d.npy", mmap_mode="r")
    ee_pos_e5 = proprio_11d_e5[:, :3]
    action_offsets_e5 = np.load(E5_CACHE_DIR / "action_offsets.npy")

    e7_mem_anchors = np.load(E7_BEHAVIOR_MEM_DIR / "mem_anchor_steps.npy")
    e7_mem_offsets = np.load(E7_BEHAVIOR_MEM_DIR / "mem_offsets.npy")
    e7_mem_lengths = np.load(E7_BEHAVIOR_MEM_DIR / "mem_lengths.npy")

    e9_mem_anchors = np.load(E9_BEHAVIOR_MEM_DIR / "mem_anchor_steps.npy")
    e9_mem_offsets = np.load(E9_BEHAVIOR_MEM_DIR / "mem_offsets.npy")
    e9_mem_lengths = np.load(E9_BEHAVIOR_MEM_DIR / "mem_lengths.npy")
    assert np.array_equal(e7_mem_anchors, e9_mem_anchors), "E7/E9 anchor grid mismatch"
    assert np.array_equal(e7_mem_offsets, e9_mem_offsets), "E7/E9 mem_offsets mismatch"
    assert np.array_equal(e7_mem_lengths, e9_mem_lengths), "E7/E9 mem_lengths mismatch"

    unseen_meta = json.loads((unseen_cache_dir / "unseen_meta.json").read_text())
    unseen_demos = unseen_meta["demos"]
    unseen_mem_h = np.load(unseen_cache_dir / "mem_h.npy", mmap_mode="r")
    unseen_mem_b_e7 = np.load(unseen_cache_dir / "mem_b_e7.npy", mmap_mode="r")
    unseen_mem_b_e9 = np.load(unseen_cache_dir / "mem_b_e9.npy", mmap_mode="r")
    unseen_mem_anchors = np.load(unseen_cache_dir / "mem_anchor_steps.npy")
    unseen_mem_offsets = np.load(unseen_cache_dir / "mem_offsets.npy")
    unseen_mem_lengths = np.load(unseen_cache_dir / "mem_lengths.npy")
    unseen_actions_10d = np.load(unseen_cache_dir / "actions_10d.npy", mmap_mode="r")
    unseen_ee_pos = np.load(unseen_cache_dir / "ee_pos.npy", mmap_mode="r")
    unseen_action_offsets = np.load(unseen_cache_dir / "action_offsets.npy")

    # 3. Sample identical pairs for Train / Seen-Val / Unseen-Chain + strict anti-leakage checks
    train_pairs = sample_cached_split_pairs(
        episodes, e7_mem_lengths, fold="train", target_pairs=n_train, seed=seed
    )
    seen_val_pairs = sample_cached_split_pairs(
        episodes, e7_mem_lengths, fold="val", target_pairs=n_seen_val, seed=seed + 1001
    )
    unseen_pairs = sample_unseen_chain_pairs(
        unseen_demos, target_pairs=n_unseen, seed=seed + 2002
    )

    train_ep_set = {p["q_ep"] for p in train_pairs} | {p["s_ep"] for p in train_pairs}
    seen_ep_set = {p["q_ep"] for p in seen_val_pairs} | {p["s_ep"] for p in seen_val_pairs}
    assert train_ep_set.isdisjoint(seen_ep_set), "LEAKAGE: train and seen_val share episodes!"

    train_task_set = {p["task"] for p in train_pairs}
    train_stem_set = {t.split("/")[-1] for t in train_task_set}
    unseen_task_set = {p["task"] for p in unseen_pairs}
    assert unseen_task_set.isdisjoint(train_task_set), "LEAKAGE: unseen task in train!"
    assert unseen_task_set.isdisjoint(train_stem_set), "LEAKAGE: unseen task stem in train!"

    leakage_audit = {
        "train_pairs": len(train_pairs),
        "seen_val_pairs": len(seen_val_pairs),
        "unseen_chain_pairs": len(unseen_pairs),
        "train_unique_episodes": len(train_ep_set),
        "seen_val_unique_episodes": len(seen_ep_set),
        "train_seen_episode_overlap": len(train_ep_set & seen_ep_set),
        "train_unique_tasks": len(train_task_set),
        "seen_val_unique_tasks": len({p["task"] for p in seen_val_pairs}),
        "unseen_unique_tasks": len(unseen_task_set),
        "train_unseen_task_overlap": len(unseen_task_set & train_task_set),
        "train_unseen_stem_overlap": len(unseen_task_set & train_stem_set),
        "all_pairs_diff_layout": all(
            p["q_layout"] != p["s_layout"]
            for p in (train_pairs + seen_val_pairs + unseen_pairs)
        ),
        "subspace_fitted_exclusively_on_train": True,
        "query_latent_oracle_only": True,
        "no_extra_source_state_input": True,
    }
    print(f"[P0] Anti-leakage audit passed: {json.dumps(leakage_audit)}", flush=True)

    # 4. Offline phase alignment + GT chunks (shared identically by E7 & E9)
    print("[P0] Computing offline phase alignment for train / seen_val / unseen_chain...", flush=True)
    split_bundles: Dict[str, Dict[str, Any]] = {}
    for sname, spairs, a10, eep, aoff, ma, mo, ml, s_off in (
        (
            "train",
            train_pairs,
            actions_10d_e5,
            ee_pos_e5,
            action_offsets_e5,
            e7_mem_anchors,
            e7_mem_offsets,
            e7_mem_lengths,
            10,
        ),
        (
            "seen_val",
            seen_val_pairs,
            actions_10d_e5,
            ee_pos_e5,
            action_offsets_e5,
            e7_mem_anchors,
            e7_mem_offsets,
            e7_mem_lengths,
            20,
        ),
        (
            "unseen_chain",
            unseen_pairs,
            unseen_actions_10d,
            unseen_ee_pos,
            unseen_action_offsets,
            unseen_mem_anchors,
            unseen_mem_offsets,
            unseen_mem_lengths,
            30,
        ),
    ):
        sb = _compute_alignment_and_gt_for_split(
            sname,
            spairs,
            actions_10d=a10,
            ee_pos_all=eep,
            action_offsets=aoff,
            mem_anchor_steps=ma,
            mem_offsets=mo,
            mem_lengths=ml,
            seed=seed + s_off,
        )
        sb["pairs"] = spairs
        sb["tasks"] = [p["task"] for p in spairs]
        if sname != "train":
            sb["init_noise_cpu"] = _build_init_noises(
                spairs, sb["q_steps"], base_seed=ddim_seed
            )
        split_bundles[sname] = sb
        au = sb["audit"]
        print(
            f"  [OfflineAlign {sname:12s}] n={au['num_pairs']}, "
            f"exact_phase={100*au['exact_gripper_phase_match_rate']:.1f}%, "
            f"quant_err={au['quant_err_steps_mean']:.2f}st, "
            f"chunk_grip_agree={100*au['chunk16_grip_agree_rate']:.1f}% "
            f"(rand={100*au['random_anchor_chunk16_grip_agree']:.1f}%), "
            f"pair_act_mse={au['gt_action_10d_mse_aligned']:.4f} "
            f"(rand={au['random_anchor_gt_action_10d_mse']:.4f})",
            flush=True,
        )

    # Pre-generate nested Haar-random orthogonal basis for DDIM control + 5 seeds for spectral check
    Q_ddim = _make_nested_random_ortho(LATENT_DIM, max(NONTRIVIAL_RANKS), seed=seed + 9001)
    Q_spec_seeds = [
        _make_nested_random_ortho(LATENT_DIM, max(NONTRIVIAL_RANKS), seed=seed + 9100 + s)
        for s in range(5)
    ]

    # 5. Run E7 and E9 sequentially (same pairs, same DDIM noise, strictly frozen)
    all_oracle_rows: List[Dict[str, Any]] = []
    subspace_summary_rows: List[Dict[str, Any]] = []
    full_summary: Dict[str, Any] = {
        "experiment": "p0_low_dimensional_behavior_transfer_diagnosis",
        "config": {
            "n_train": len(train_pairs),
            "n_seen_val": len(seen_val_pairs),
            "n_unseen": len(unseen_pairs),
            "ranks": list(RANK_GRID),
            "nontrivial_ranks": list(NONTRIVIAL_RANKS),
            "seed": seed,
            "ddim_seed": ddim_seed,
            "use_bf16": bool(use_bf16 and device.type == "cuda"),
            "behavior_stride": 16,
            "action_horizon": ACTION_HORIZON,
            "exec_horizon": EXEC_HORIZON,
        },
        "checkpoints": {
            "e7_stage1": str(E7_STAGE1_CKPT),
            "e7_stage2": str(E7_STAGE2_CKPT),
            "e9_stage1": str(E9_STAGE1_CKPT),
            "e9_stage2": str(E9_STAGE2_CKPT),
        },
        "leakage_audit": leakage_audit,
        "offline_alignment_audit": {k: v["audit"] for k, v in split_bundles.items()},
        "models": {},
    }

    model_specs = [
        (
            "e7",
            E7_BEHAVIOR_MEM_DIR,
            unseen_mem_b_e7,
            lambda: load_e7_policy(E7_STAGE2_CKPT, stage1_ckpt=E7_STAGE1_CKPT, device=device),
        ),
        (
            "e9",
            E9_BEHAVIOR_MEM_DIR,
            unseen_mem_b_e9,
            lambda: load_e9_stage2_direct_policy(
                E9_STAGE2_CKPT, stage1_ckpt=E9_STAGE1_CKPT, device=device
            ),
        ),
    ]

    for model_name, bmem_dir, u_mem_b, loader_fn in model_specs:
        tm0 = time.monotonic()
        print(f"\n==================== [{model_name.upper()}] ====================", flush=True)
        policy, payload = loader_fn()
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        assert trainable_cnt == 0, f"{model_name} has {trainable_cnt} trainable params!"
        s_z = float(policy.latent_scale_s_z.item())
        print(
            f"[{model_name.upper()}] Loaded frozen policy (epoch={payload.get('epoch')}, "
            f"s_z={s_z:.6f}, trainable={trainable_cnt})",
            flush=True,
        )

        e5_mem_h = np.load(bmem_dir / "mem_h.npy", mmap_mode="r")
        e5_mem_b = np.load(bmem_dir / "mem_b.npy", mmap_mode="r")

        # Extract latents for train / seen_val / unseen_chain
        lat_by_split: Dict[str, Dict[str, Any]] = {}
        for sname in ("train", "seen_val", "unseen_chain"):
            is_unseen = sname == "unseen_chain"
            lat_by_split[sname] = _extract_model_latents_for_split(
                policy,
                split_bundles[sname]["pairs"],
                split_bundles[sname],
                mem_h=unseen_mem_h if is_unseen else e5_mem_h,
                mem_b=u_mem_b if is_unseen else e5_mem_b,
                mem_anchor_steps=unseen_mem_anchors if is_unseen else e7_mem_anchors,
                mem_offsets=unseen_mem_offsets if is_unseen else e7_mem_offsets,
                mem_lengths=unseen_mem_lengths if is_unseen else e7_mem_lengths,
                actions_10d=unseen_actions_10d if is_unseen else actions_10d_e5,
                action_offsets=unseen_action_offsets if is_unseen else action_offsets_e5,
                device=device,
                use_bf16=use_bf16,
                batch_size=batch_size,
                compute_d4=(sname != "train"),
            )
            ra = lat_by_split[sname]["reader_audit"]
            print(
                f"  [{model_name.upper()} Reader {sname:12s}] max_w={ra['max_weight_mean']:.4f}, "
                f"H_n={ra['norm_entropy_mean']:.4f}, eff_pos={ra['eff_pos_mean']:.2f}, "
                f"slot_diff_vs_off={ra['slot_diff_vs_offline_mean']:.2f} "
                f"(within1={100*ra['within1_slot_match_vs_offline_rate']:.1f}%), "
                f"argmax_grip_agree={100*ra['argmax_chunk16_grip_agree_rate']:.1f}%",
                flush=True,
            )

        # Behavior latent (Z_query) SVD on train: uncentered, centered, coord-standardized
        zq_tr = lat_by_split["train"]["z_query"].astype(np.float64)
        zq_uncentered_svd = _compute_svd_stats(zq_tr, center=False)
        zq_centered_svd = _compute_svd_stats(zq_tr, center=True)
        zq_coord_std = np.maximum(zq_tr.std(axis=0), 1e-4)
        zq_std_svd = _compute_svd_stats((zq_tr - zq_tr.mean(axis=0)) / zq_coord_std[None, :], center=False)

        print(
            f"  [{model_name.upper()} Z_query Train] "
            f"uncentered: d90={zq_uncentered_svd['dims_at_threshold']['d_90']}, "
            f"erank_E={zq_uncentered_svd['erank_energy']:.1f}, PR={zq_uncentered_svd['participation_ratio']:.1f}, "
            f"mean_E_frac={100*zq_uncentered_svd['mean_energy_frac']:.1f}% | "
            f"centered: d90={zq_centered_svd['dims_at_threshold']['d_90']}, "
            f"erank_E={zq_centered_svd['erank_energy']:.1f}, PR={zq_centered_svd['participation_ratio']:.1f} | "
            f"coord_std: d90={zq_std_svd['dims_at_threshold']['d_90']}, PR={zq_std_svd['participation_ratio']:.1f}",
            flush=True,
        )

        # Record Z_query rows in subspace summary
        for label, sv in (
            ("z_query_uncentered", zq_uncentered_svd),
            ("z_query_centered", zq_centered_svd),
            ("z_query_coord_std", zq_std_svd),
        ):
            subspace_summary_rows.append(
                {
                    "model": model_name,
                    "alignment": "behavior_latent",
                    "matrix": label,
                    "s_z": s_z,
                    "rms": sv["rms"],
                    "rms_over_sz": sv["rms"] / s_z,
                    "mean_energy_frac": sv["mean_energy_frac"],
                    "erank_energy": sv["erank_energy"],
                    "erank_sv": sv["erank_sv"],
                    "participation_ratio": sv["participation_ratio"],
                    **sv["dims_at_threshold"],
                    **sv["cum_energy_at_k"],
                }
            )

        model_res: Dict[str, Any] = {
            "epoch": payload.get("epoch"),
            "s_z": s_z,
            "reader_audit": {s: lat_by_split[s]["reader_audit"] for s in ("train", "seen_val", "unseen_chain")},
            "z_query_train_svd": {
                "uncentered": _strip_arrays(zq_uncentered_svd),
                "centered": _strip_arrays(zq_centered_svd),
                "coord_standardized": _strip_arrays(zq_std_svd),
            },
            "alignments": {},
        }

        train_U_by_align: Dict[str, np.ndarray] = {}
        deltas_by_align_split: Dict[str, Dict[str, np.ndarray]] = {}

        for align_name, src_key in (
            ("offline_align", "z_src_offline"),
            ("reader_align", "z_src_reader"),
        ):
            align_dir = out_dir / model_name / align_name
            align_dir.mkdir(parents=True, exist_ok=True)

            dz_tr = (
                lat_by_split["train"]["z_query"].astype(np.float64)
                - lat_by_split["train"][src_key].astype(np.float64)
            )
            dz_seen = (
                lat_by_split["seen_val"]["z_query"].astype(np.float64)
                - lat_by_split["seen_val"][src_key].astype(np.float64)
            )
            dz_unseen = (
                lat_by_split["unseen_chain"]["z_query"].astype(np.float64)
                - lat_by_split["unseen_chain"][src_key].astype(np.float64)
            )
            deltas_by_align_split[align_name] = {
                "train": dz_tr,
                "seen_val": dz_seen,
                "unseen_chain": dz_unseen,
            }

            # Primary: UN-CENTERED SVD on train residuals (strictly train-only!)
            dz_uncentered_svd = _compute_svd_stats(dz_tr, center=False)
            # Diagnostics: centered residual SVD & coordinate-standardized residual SVD
            dz_centered_svd = _compute_svd_stats(dz_tr, center=True)
            dz_std_svd = _compute_svd_stats(dz_tr / zq_coord_std[None, :], center=False)

            # In-domain SVD on seen/unseen ONLY to measure subspace transfer gap & principal angles
            dz_seen_indom = _compute_svd_stats(dz_seen, center=False)
            dz_unseen_indom = _compute_svd_stats(dz_unseen, center=False)

            U_train = dz_uncentered_svd["_Vt"].T  # (3072, 3072) columns are basis vectors
            U_seen_indom = dz_seen_indom["_Vt"].T
            U_unseen_indom = dz_unseen_indom["_Vt"].T
            train_U_by_align[align_name] = U_train

            print(
                f"  [{model_name.upper()} {align_name:13s}] "
                f"Δz_rms={dz_uncentered_svd['rms']:.4f} (Δz/s_z={dz_uncentered_svd['rms']/s_z:.4f}, "
                f"Δz/std_z={dz_uncentered_svd['rms']/max(zq_centered_svd['rms'],1e-12):.3f}, "
                f"mean_E={100*dz_uncentered_svd['mean_energy_frac']:.2f}%) | "
                f"uncentered: d50={dz_uncentered_svd['dims_at_threshold']['d_50']}, "
                f"d80={dz_uncentered_svd['dims_at_threshold']['d_80']}, "
                f"d90={dz_uncentered_svd['dims_at_threshold']['d_90']}, "
                f"d95={dz_uncentered_svd['dims_at_threshold']['d_95']}, "
                f"erank_E={dz_uncentered_svd['erank_energy']:.1f}, PR={dz_uncentered_svd['participation_ratio']:.1f} | "
                f"centered_PR={dz_centered_svd['participation_ratio']:.1f}, "
                f"coord_std_PR={dz_std_svd['participation_ratio']:.1f}",
                flush=True,
            )

            for label, sv in (
                ("residual_uncentered", dz_uncentered_svd),
                ("residual_centered", dz_centered_svd),
                ("residual_coord_std", dz_std_svd),
                ("residual_seen_indomain", dz_seen_indom),
                ("residual_unseen_indomain", dz_unseen_indom),
            ):
                subspace_summary_rows.append(
                    {
                        "model": model_name,
                        "alignment": align_name,
                        "matrix": label,
                        "s_z": s_z,
                        "rms": sv["rms"],
                        "rms_over_sz": sv["rms"] / s_z,
                        "mean_energy_frac": sv["mean_energy_frac"],
                        "erank_energy": sv["erank_energy"],
                        "erank_sv": sv["erank_sv"],
                        "participation_ratio": sv["participation_ratio"],
                        **sv["dims_at_threshold"],
                        **sv["cum_energy_at_k"],
                    }
                )

            # Save full spectrum CSV (all 3072 components) for plotting
            spec_rows = []
            for idx_sv in range(LATENT_DIM):
                spec_rows.append(
                    {
                        "rank": idx_sv + 1,
                        "sv_zq_uncentered": float(zq_uncentered_svd["_S"][idx_sv]),
                        "cum_zq_uncentered": float(zq_uncentered_svd["_cum_e"][idx_sv]),
                        "sv_zq_centered": float(zq_centered_svd["_S"][idx_sv]),
                        "cum_zq_centered": float(zq_centered_svd["_cum_e"][idx_sv]),
                        "sv_zq_coord_std": float(zq_std_svd["_S"][idx_sv]),
                        "cum_zq_coord_std": float(zq_std_svd["_cum_e"][idx_sv]),
                        "sv_dz_uncentered": float(dz_uncentered_svd["_S"][idx_sv]),
                        "cum_dz_uncentered": float(dz_uncentered_svd["_cum_e"][idx_sv]),
                        "sv_dz_centered": float(dz_centered_svd["_S"][idx_sv]),
                        "cum_dz_centered": float(dz_centered_svd["_cum_e"][idx_sv]),
                        "sv_dz_coord_std": float(dz_std_svd["_S"][idx_sv]),
                        "cum_dz_coord_std": float(dz_std_svd["_cum_e"][idx_sv]),
                    }
                )
            _write_csv(spec_rows, align_dir / "spectrum.csv")

            # Train -> Seen / Unseen subspace generalization table
            gen_rows = []
            for k in RANK_GRID:
                Uk = U_train[:, :k]
                e_tr = _energy_recovery(dz_tr, Uk)
                e_seen = _energy_recovery(dz_seen, Uk)
                e_unseen = _energy_recovery(dz_unseen, Uk)
                e_seen_indom = (
                    float(dz_seen_indom["_cum_e"][min(k, len(dz_seen_indom["_cum_e"])) - 1])
                    if k > 0
                    else 0.0
                )
                e_unseen_indom = (
                    float(dz_unseen_indom["_cum_e"][min(k, len(dz_unseen_indom["_cum_e"])) - 1])
                    if k > 0
                    else 0.0
                )
                ks = min(k, U_seen_indom.shape[1])
                ku = min(k, U_unseen_indom.shape[1])
                ov_seen = (
                    _subspace_overlap(Uk[:, :ks], U_seen_indom[:, :ks])
                    if 0 < k < LATENT_DIM
                    else (1.0 if k == LATENT_DIM else 0.0)
                )
                ov_unseen = (
                    _subspace_overlap(Uk[:, :ku], U_unseen_indom[:, :ku])
                    if 0 < k < LATENT_DIM
                    else (1.0 if k == LATENT_DIM else 0.0)
                )

                if 0 < k < LATENT_DIM:
                    r_seen = [_energy_recovery(dz_seen, Q[:, :k]) for Q in Q_spec_seeds]
                    r_unseen = [_energy_recovery(dz_unseen, Q[:, :k]) for Q in Q_spec_seeds]
                    rand_seen_m, rand_seen_s = float(np.mean(r_seen)), float(np.std(r_seen))
                    rand_unseen_m, rand_unseen_s = float(np.mean(r_unseen)), float(np.std(r_unseen))
                else:
                    val_edge = 0.0 if k == 0 else 1.0
                    rand_seen_m = rand_unseen_m = val_edge
                    rand_seen_s = rand_unseen_s = 0.0

                gen_rows.append(
                    {
                        "rank_k": int(k),
                        "energy_train": e_tr,
                        "energy_seen_val": e_seen,
                        "energy_seen_indomain_oracle": e_seen_indom,
                        "energy_seen_gap": e_seen_indom - e_seen,
                        "energy_seen_random_mean": rand_seen_m,
                        "energy_seen_random_std": rand_seen_s,
                        "energy_unseen_chain": e_unseen,
                        "energy_unseen_indomain_oracle": e_unseen_indom,
                        "energy_unseen_gap": e_unseen_indom - e_unseen,
                        "energy_unseen_random_mean": rand_unseen_m,
                        "energy_unseen_random_std": rand_unseen_s,
                        "principal_overlap_train_vs_seen": ov_seen,
                        "principal_overlap_train_vs_unseen": ov_unseen,
                    }
                )
            _write_csv(gen_rows, align_dir / "generalization.csv")

            # Oracle Low-Rank Transfer DDIM Action Decoding on seen_val and unseen_chain
            align_oracle_rows: List[Dict[str, Any]] = []
            oracle_by_split: Dict[str, List[Dict[str, Any]]] = {}
            U_train_f32 = U_train.astype(np.float32)
            Q_ddim_f32 = Q_ddim.astype(np.float32)

            for eval_split in ("seen_val", "unseen_chain"):
                sb = split_bundles[eval_split]
                zq = lat_by_split[eval_split]["z_query"]
                zs = lat_by_split[eval_split][src_key]
                dz = zq - zs
                split_rows: List[Dict[str, Any]] = []

                # SVD train-fitted basis for K in {0, 8, 16, 32, 64, 128, 256, Full}
                for k in RANK_GRID:
                    if k == 0:
                        z_hat = zs
                    elif k == LATENT_DIM:
                        z_hat = zq
                    else:
                        Uk32 = U_train_f32[:, :k]
                        z_hat = zs + (dz @ Uk32) @ Uk32.T

                    metrics = _decode_and_score_latents(
                        policy,
                        z_hat,
                        zq,
                        zs,
                        sb["gt_chunks"],
                        sb["valid_16"],
                        sb["init_noise_cpu"],
                        sb["tasks"],
                        s_z=s_z,
                        device=device,
                        use_bf16=use_bf16,
                        batch_size=batch_size,
                    )
                    row = {
                        "model": model_name,
                        "alignment": align_name,
                        "split": eval_split,
                        "basis_type": "svd_train",
                        "rank_k": int(k),
                        "rank_label": "Full" if k == LATENT_DIM else str(k),
                        **metrics,
                    }
                    split_rows.append(row)

                # Random orthogonal control for K in {8, 16, 32, 64, 128, 256}
                for k in NONTRIVIAL_RANKS:
                    Qk32 = Q_ddim_f32[:, :k]
                    z_hat_r = zs + (dz @ Qk32) @ Qk32.T
                    metrics_r = _decode_and_score_latents(
                        policy,
                        z_hat_r,
                        zq,
                        zs,
                        sb["gt_chunks"],
                        sb["valid_16"],
                        sb["init_noise_cpu"],
                        sb["tasks"],
                        s_z=s_z,
                        device=device,
                        use_bf16=use_bf16,
                        batch_size=batch_size,
                    )
                    row_r = {
                        "model": model_name,
                        "alignment": align_name,
                        "split": eval_split,
                        "basis_type": "random_ortho",
                        "rank_k": int(k),
                        "rank_label": str(k),
                        **metrics_r,
                    }
                    split_rows.append(row_r)

                # Frozen Stage 2 Direct (D4) reference on reader_align
                if align_name == "reader_align" and lat_by_split[eval_split]["z_d4"] is not None:
                    metrics_d4 = _decode_and_score_latents(
                        policy,
                        lat_by_split[eval_split]["z_d4"],
                        zq,
                        zs,
                        sb["gt_chunks"],
                        sb["valid_16"],
                        sb["init_noise_cpu"],
                        sb["tasks"],
                        s_z=s_z,
                        device=device,
                        use_bf16=use_bf16,
                        batch_size=batch_size,
                    )
                    split_rows.append(
                        {
                            "model": model_name,
                            "alignment": align_name,
                            "split": eval_split,
                            "basis_type": "frozen_stage2_d4",
                            "rank_k": -1,
                            "rank_label": "D4_Stage2",
                            **metrics_d4,
                        }
                    )

                rec_0 = next(r for r in split_rows if r["basis_type"] == "svd_train" and r["rank_k"] == 0)
                rec_full = next(
                    r for r in split_rows if r["basis_type"] == "svd_train" and r["rank_k"] == LATENT_DIM
                )
                _attach_arr(split_rows, rec_0, rec_full)

                oracle_by_split[eval_split] = split_rows
                align_oracle_rows.extend(split_rows)
                all_oracle_rows.extend(split_rows)

                # Compact progress print
                by_k = {
                    (r["basis_type"], r["rank_k"]): r for r in split_rows
                }
                k64 = by_k[("svd_train", 64)]
                k128 = by_k[("svd_train", 128)]
                k256 = by_k[("svd_train", 256)]
                r64 = by_k[("random_ortho", 64)]
                print(
                    f"    [{model_name.upper()} {align_name:13s} {eval_split:12s}] "
                    f"MSE0={rec_0['ddim_10d_mse']:.5f} -> "
                    f"K64={k64['ddim_10d_mse']:.5f} (ARR={100*(k64['arr_10d'] or 0):.1f}%, E={100*k64['residual_energy_recovery']:.1f}%, rand64={r64['ddim_10d_mse']:.5f}) -> "
                    f"K128={k128['ddim_10d_mse']:.5f} (ARR={100*(k128['arr_10d'] or 0):.1f}%) -> "
                    f"K256={k256['ddim_10d_mse']:.5f} (ARR={100*(k256['arr_10d'] or 0):.1f}%) -> "
                    f"Full={rec_full['ddim_10d_mse']:.5f}",
                    flush=True,
                )

            _write_csv([_flatten_oracle_row(r) for r in align_oracle_rows], align_dir / "oracle_transfer.csv")

            model_res["alignments"][align_name] = {
                "train_svd_uncentered": _strip_arrays(dz_uncentered_svd),
                "train_svd_centered": _strip_arrays(dz_centered_svd),
                "train_svd_coord_standardized": _strip_arrays(dz_std_svd),
                "seen_indomain_svd": _strip_arrays(dz_seen_indom),
                "unseen_indomain_svd": _strip_arrays(dz_unseen_indom),
                "generalization": gen_rows,
                "oracle_transfer": oracle_by_split,
            }

        # Cross-alignment subspace overlap (offline_align vs reader_align)
        cross_align = {}
        U_off = train_U_by_align["offline_align"]
        U_rdr = train_U_by_align["reader_align"]
        for k in NONTRIVIAL_RANKS:
            cross_align[str(k)] = {
                "principal_overlap_offline_vs_reader": _subspace_overlap(U_off[:, :k], U_rdr[:, :k]),
                "offline_U_on_reader_seen_energy": _energy_recovery(
                    deltas_by_align_split["reader_align"]["seen_val"], U_off[:, :k]
                ),
                "reader_U_on_offline_seen_energy": _energy_recovery(
                    deltas_by_align_split["offline_align"]["seen_val"], U_rdr[:, :k]
                ),
            }
        model_res["cross_alignment_subspace_comparison"] = cross_align
        model_res["elapsed_seconds"] = round(time.monotonic() - tm0, 2)
        full_summary["models"][model_name] = model_res

        del policy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    full_summary["total_elapsed_seconds"] = round(time.monotonic() - t0, 2)
    atomic_write_json(full_summary, out_dir / "summary.json")
    atomic_write_json(
        {
            "leakage_audit": leakage_audit,
            "offline_alignment_audit": full_summary["offline_alignment_audit"],
            "reader_audit": {
                m: full_summary["models"][m]["reader_audit"] for m in full_summary["models"]
            },
        },
        out_dir / "alignment_audit.json",
    )
    _write_csv([_flatten_oracle_row(r) for r in all_oracle_rows], out_dir / "oracle_transfer_all.csv")
    _write_csv(subspace_summary_rows, out_dir / "subspace_summary.csv")

    print(
        f"\n[P0 Diagnosis DONE] Total elapsed: {full_summary['total_elapsed_seconds']}s -> {out_dir}",
        flush=True,
    )
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--unseen-cache", type=Path, default=DEFAULT_UNSEEN_CACHE_DIR)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-seen-val", type=int, default=3000)
    parser.add_argument("--n-unseen", type=int, default=3000)
    parser.add_argument("--demos-per-unseen-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_p0_diagnosis(
        out_dir=args.output,
        unseen_cache_dir=args.unseen_cache,
        n_train=args.n_train,
        n_seen_val=args.n_seen_val,
        n_unseen=args.n_unseen,
        demos_per_unseen_task=args.demos_per_unseen_task,
        seed=args.seed,
        ddim_seed=args.ddim_seed,
        batch_size=args.batch_size,
        device=device,
        use_bf16=not args.no_bf16,
    )


if __name__ == "__main__":
    main()
