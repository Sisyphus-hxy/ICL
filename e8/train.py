"""Training, smoke verification, offline validation, and inference-time source-state shuffle
diagnostic for E8: Source-conditioned Latent Behavior Transfer (CLIP).

Hyperparameters & protocol strictly match E7-Direct:
  - Reuses E7 Behavior Memory cache (stride=16) and E7TransferBank splits/pairings
  - Frozen Stage 1 Teacher (`e6/outputs/stage1/stage1_self_latent/best.pt`), CLIP, UNet
  - Reader and TransferTransformer trained from scratch (NO E7 weights loaded)
  - Direct Prediction init: `direct_latent_bias = mean(z_self_train)`,
    `out_proj.weight ~ N(0, 0.002^2)`, `output_scale = s_z = 7.822748`
  - Loss: L = L_diff(A_q | z_q) + 0.5 * L_z, fixed s_z = 7.822748
  - Reader / Transformer LR = 1e-4 -> 1e-5 (Cosine, warmup=500)
  - AdamW, betas=(0.9, 0.95), weight_decay=0.01, grad_clip=1.0, BF16, EMA=0.999
  - Global batch = 256, 8 full-pass / 42,912 steps, seed=42, val_seed=12345
  - Best checkpoint selected by Validation Cross DDIM 10D Action MSE
  - All outputs saved independently under `e8/outputs/source_conditioned/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from e5.diffusion import EMAModel
from e5.model import E5Policy
from e5.train import (
    _action_error_sums,
    _all_reduce_sum,
    _rank,
    _unpack_action_errors,
    _world,
    atomic_torch_save,
    atomic_write_json,
    seed_everything,
)
from e6.stage1_model import load_frozen_stage1_teacher
from e7.data import E7TransferBank
from e7.model import compute_attention_stats
from e7.train import compute_train_z_self_mean, get_cosine_schedule_with_min_ratio

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    DEFAULT_BEHAVIOR_MEM_DIR,
    DEFAULT_STAGE1_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    FIXED_LATENT_SCALE_S_Z,
    OUTPUT_ROOT,
)
from .model import E8Policy, load_e8_policy


def _build_cross_episode_derangement(
    query_episodes: np.ndarray,
    demo_episodes: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Build deterministic permutation `perm` of `0..N-1` such that `perm[i]` comes from a
    different query episode AND different demo episode than `i` whenever possible."""
    n = len(query_episodes)
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(np.int64)
    for i in range(n):
        j = int(perm[i])
        if query_episodes[j] == query_episodes[i] or demo_episodes[j] == demo_episodes[i]:
            # Find a valid swap partner deterministically
            for offset in range(1, n):
                k = (i + offset) % n
                pk = int(perm[k])
                if (
                    query_episodes[pk] != query_episodes[i]
                    and demo_episodes[pk] != demo_episodes[i]
                    and query_episodes[j] != query_episodes[k]
                    and demo_episodes[j] != demo_episodes[k]
                ):
                    perm[i], perm[k] = pk, j
                    break
    return perm


def _shift_weights_within_demo(
    weights: torch.Tensor,
    mem_pad_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Circularly shift attention weights by `max(1, M_valid // 2)` across valid entries of
    each Demo, creating a maximally phase-mismatched source state from the SAME Demo."""
    B, M = weights.shape
    device = weights.device
    shifted = torch.zeros_like(weights)
    if mem_pad_mask is None:
        valid_lens = torch.full((B,), M, device=device, dtype=torch.long)
    else:
        valid_lens = (~mem_pad_mask.to(device=device, dtype=torch.bool)).long().sum(dim=-1)
    for b in range(B):
        L = int(valid_lens[b].item())
        if L <= 1:
            shifted[b, :L] = weights[b, :L]
        else:
            shift = max(1, L // 2)
            shifted[b, :L] = torch.roll(weights[b, :L], shifts=shift, dims=0)
    return shifted


@torch.inference_mode()
def evaluate_e8_validation(
    policy: E8Policy,
    bank: E7TransferBank,
    *,
    batch_size: int = 128,
    max_val_samples: Optional[int] = None,
    run_ddim: bool = True,
    ddim_max_samples: Optional[int] = None,
    run_multi_demo: bool = True,
    multi_demo_max_samples: Optional[int] = 1024,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Comprehensive distributed offline validation for E8, directly comparable to E7-Direct."""
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    s_z = float(policy.latent_scale_s_z.item())
    s_z2 = s_z * s_z
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    n_pool = int(bank.val_valid_cross_mask.sum())
    n_val = n_pool if max_val_samples is None else min(int(max_val_samples), n_pool)

    gen = torch.Generator(device="cpu").manual_seed(val_seed)
    all_noise = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen)
    all_pertub = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen)
    all_t = torch.randint(
        0,
        policy.teacher.noise_scheduler.config.num_train_timesteps,
        (n_val,),
        generator=gen,
        dtype=torch.long,
    )
    local_idx = np.arange(n_val)[rank::world]
    local_noise = all_noise[local_idx]
    local_pertub = all_pertub[local_idx]
    local_t = all_t[local_idx]

    # 64 float64 accumulators (58 from E7 + 6 source-state similarity diagnostics)
    acc = torch.zeros(64, device=device, dtype=torch.float64)

    c_it = bank.iter_e7_val_batches(
        "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    s_it = bank.iter_e7_val_batches(
        "self", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    w_it = bank.iter_e7_val_batches(
        "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    s1_it = bank.iter_val_batches(
        "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_val
    )

    rr_teacher = policy.teacher.reference_reader
    cur = 0
    for (
        (hm_c, bm_c, mpm_c, q_c, z_self, a_c, _),
        (hm_s, bm_s, mpm_s, q_s, _, _, _),
        (hm_w, bm_w, mpm_w, q_w, _, a_w, _),
        (dm_s1, pm_s1, _, _, _, _),
    ) in zip(c_it, s_it, w_it, s1_it):
        B = len(hm_c)
        noise = local_noise[cur : cur + B].to(device, non_blocking=True)
        pertub = local_pertub[cur : cur + B].to(device, non_blocking=True)
        t = local_t[cur : cur + B].to(device, non_blocking=True)
        cur += B

        h_q = E5Policy.format_query_tokens(q_c)
        with amp_ctx:
            z_c0 = rr_teacher.prompt_with_obs_decoder(
                tgt=h_q + rr_teacher.prompt_current_obs_pos_emb,
                memory=dm_s1,
                memory_key_padding_mask=pm_s1,
                need_weights=False,
            )
            # E8 Cross retrieval (shared weights -> z_c, h_ref_c) + direct transfer
            z_c, h_ref_c, _, w_c = policy.read_cross_latent_and_source(
                hm_c, bm_c, mpm_c, q_c, h_q=h_q
            )
            z_q, delta_c = policy.transfer_latent(z_c, h_ref_c, h_q)

            # E8 Self sparse-memory retrieval + transfer
            z_s_ret, h_ref_s, _, w_s = policy.read_cross_latent_and_source(
                hm_s, bm_s, mpm_s, q_s, h_q=h_q
            )
            z_s_q, delta_s_ret = policy.transfer_latent(z_s_ret, h_ref_s, h_q)
            _, delta_s_teacher = policy.transfer_latent(z_self, h_q, h_q)

            # E8 Wrong-task Demo retrieval + transfer
            z_w, h_ref_w, _, w_w = policy.read_cross_latent_and_source(
                hm_w, bm_w, mpm_w, q_w, h_q=h_q
            )
            z_w_q, delta_w = policy.transfer_latent(z_w, h_ref_w, h_q)

            l_self = policy.teacher.diffusion_loss_from_z_self(
                z_self, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_s1_c = policy.teacher.diffusion_loss_from_z_self(
                z_c0, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_nd = policy.teacher.diffusion_loss_from_z_self(
                z_c, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_d = policy.teacher.diffusion_loss_from_z_self(
                z_q, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_s_nd = policy.teacher.diffusion_loss_from_z_self(
                z_s_ret, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_s_d = policy.teacher.diffusion_loss_from_z_self(
                z_s_q, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_w_nd = policy.teacher.diffusion_loss_from_z_self(
                z_w, a_w, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_w_d = policy.teacher.diffusion_loss_from_z_self(
                z_w_q, a_w, noise=noise, pertub_noise=pertub, timesteps=t
            )

        zs_f = z_self.float()
        zc0_f = z_c0.float()
        zc_f = z_c.float()
        zq_f = z_q.float()
        zs_ret_f = z_s_ret.float()
        zs_q_f = z_s_q.float()
        zw_f = z_w.float()
        zwq_f = z_w_q.float()
        dc_f = delta_c.float()
        hq_f = h_q.float()
        href_c_f = h_ref_c.float()
        href_s_f = h_ref_s.float()
        href_w_f = h_ref_w.float()

        gap_c = zs_f - zc_f
        gap_sq_i = gap_c.square().mean(dim=(1, 2))
        mse_zq_i = (zq_f - zs_f).square().mean(dim=(1, 2))
        g_rms_i = gap_sq_i.clamp_min(0.0).sqrt()
        d_rms_i = dc_f.square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()
        dir_cos_i = F.cosine_similarity(dc_f.reshape(B, -1), gap_c.reshape(B, -1), dim=-1)
        proj_frac_i = (dc_f * gap_c).sum(dim=(1, 2)) / gap_c.square().sum(dim=(1, 2)).clamp_min(1e-12)
        mag_ratio_i = d_rms_i / g_rms_i.clamp_min(1e-6)

        ast_c = compute_attention_stats(w_c, mpm_c)
        ast_s = compute_attention_stats(w_s, mpm_s)
        ast_w = compute_attention_stats(w_w, mpm_w)

        acc[0] += float(B)
        acc[1] += l_self.double().sum()
        acc[2] += l_s1_c.double().sum()
        acc[3] += l_c_nd.double().sum()
        acc[4] += l_c_d.double().sum()
        acc[5] += l_s_nd.double().sum()
        acc[6] += l_s_d.double().sum()
        acc[7] += l_w_nd.double().sum()
        acc[8] += l_w_d.double().sum()

        acc[9] += (zc0_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[10] += gap_sq_i.double().sum()
        acc[11] += mse_zq_i.double().sum()
        acc[12] += (zs_ret_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[13] += (zs_q_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[14] += (zw_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[15] += (zwq_f - zs_f).square().mean(dim=(1, 2)).double().sum()

        acc[16] += F.cosine_similarity(zc0_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[17] += F.cosine_similarity(zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[18] += F.cosine_similarity(zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[19] += F.cosine_similarity(zs_ret_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[20] += F.cosine_similarity(zs_q_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()

        acc[21] += dc_f.square().mean(dim=(1, 2)).double().sum()
        acc[22] += zc_f.square().mean(dim=(1, 2)).double().sum()
        acc[23] += zs_f.square().mean(dim=(1, 2)).double().sum()
        acc[24] += delta_s_ret.float().square().mean(dim=(1, 2)).double().sum()
        acc[25] += delta_s_teacher.float().square().mean(dim=(1, 2)).double().sum()
        acc[26] += delta_w.float().square().mean(dim=(1, 2)).double().sum()

        acc[27] += dir_cos_i.double().sum()
        acc[28] += proj_frac_i.double().sum()
        acc[29] += mag_ratio_i.double().sum()
        acc[30] += g_rms_i.double().sum()
        acc[31] += d_rms_i.double().sum()
        acc[32] += (g_rms_i * g_rms_i).double().sum()
        acc[33] += (d_rms_i * d_rms_i).double().sum()
        acc[34] += (g_rms_i * d_rms_i).double().sum()
        acc[35] += (mse_zq_i < gap_sq_i).double().sum()

        acc[36] += ast_c["entropy"].double().sum()
        acc[37] += ast_c["uniform_entropy"].double().sum()
        acc[38] += ast_c["norm_entropy"].double().sum()
        acc[39] += ast_c["max_weight"].double().sum()
        acc[40] += ast_c["uniform_max_weight"].double().sum()
        acc[41] += ast_c["eff_pos_exp"].double().sum()
        acc[42] += ast_c["eff_pos_ipr"].double().sum()
        acc[43] += ast_c["eff_frac"].double().sum()
        acc[44] += ast_c["valid_pos"].double().sum()

        acc[45] += ast_s["entropy"].double().sum()
        acc[46] += ast_s["norm_entropy"].double().sum()
        acc[47] += ast_s["max_weight"].double().sum()
        acc[48] += ast_s["eff_pos_exp"].double().sum()
        acc[49] += ast_s["eff_pos_ipr"].double().sum()
        acc[50] += ast_s["valid_pos"].double().sum()

        acc[51] += ast_w["entropy"].double().sum()
        acc[52] += ast_w["norm_entropy"].double().sum()
        acc[53] += ast_w["max_weight"].double().sum()
        acc[54] += ast_w["eff_pos_exp"].double().sum()
        acc[55] += ast_w["eff_pos_ipr"].double().sum()
        acc[56] += ast_w["valid_pos"].double().sum()
        acc[57] += zq_f.square().mean(dim=(1, 2)).double().sum()

        # Source state vs Target state diagnostics (Cross, Self, Wrong-Demo)
        acc[58] += F.cosine_similarity(href_c_f.reshape(B, -1), hq_f.reshape(B, -1), dim=-1).double().sum()
        acc[59] += (href_c_f - hq_f).square().mean(dim=(1, 2)).double().sum()
        acc[60] += F.cosine_similarity(href_s_f.reshape(B, -1), hq_f.reshape(B, -1), dim=-1).double().sum()
        acc[61] += (href_s_f - hq_f).square().mean(dim=(1, 2)).double().sum()
        acc[62] += F.cosine_similarity(href_w_f.reshape(B, -1), hq_f.reshape(B, -1), dim=-1).double().sum()
        acc[63] += (href_w_f - hq_f).square().mean(dim=(1, 2)).double().sum()

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)

    def _pearson(sx: float, sy: float, sxx: float, syy: float, sxy: float, n: float) -> float:
        mx = sx / n
        my = sy / n
        vx = max(sxx / n - mx * mx, 0.0)
        vy = max(syy / n - my * my, 0.0)
        denom = math.sqrt(vx * vy)
        return float((sxy / n - mx * my) / denom) if denom > 1e-12 else 0.0

    diff_self = float(acc[1].item() / total)
    diff_s1_c = float(acc[2].item() / total)
    diff_c_nd = float(acc[3].item() / total)
    diff_c_d = float(acc[4].item() / total)
    diff_s_nd = float(acc[5].item() / total)
    diff_s_d = float(acc[6].item() / total)
    diff_w_nd = float(acc[7].item() / total)
    diff_w_d = float(acc[8].item() / total)

    mse_s1_c = float(acc[9].item() / total)
    mse_c_nd = float(acc[10].item() / total)
    mse_c_d = float(acc[11].item() / total)
    mse_s_nd = float(acc[12].item() / total)
    mse_s_d = float(acc[13].item() / total)
    mse_w_nd = float(acc[14].item() / total)
    mse_w_d = float(acc[15].item() / total)

    cross_delta_rms = math.sqrt(float(acc[21].item() / total))
    cross_z_rms = math.sqrt(float(acc[22].item() / total))
    self_z_rms = math.sqrt(float(acc[23].item() / total))
    zq_rms = math.sqrt(float(acc[57].item() / total))
    self_ret_delta_rms = math.sqrt(float(acc[24].item() / total))
    self_teacher_delta_rms = math.sqrt(float(acc[25].item() / total))
    wrong_delta_rms = math.sqrt(float(acc[26].item() / total))
    gap_rms = math.sqrt(mse_c_nd)

    mag_gap_r = _pearson(
        float(acc[30].item()),
        float(acc[31].item()),
        float(acc[32].item()),
        float(acc[33].item()),
        float(acc[34].item()),
        total,
    )

    reader_diff_gain = diff_s1_c - diff_c_nd
    delta_diff_gain = diff_c_nd - diff_c_d
    total_diff_gain = diff_s1_c - diff_c_d

    reader_latent_gain = mse_s1_c - mse_c_nd
    delta_latent_gain = mse_c_nd - mse_c_d
    total_latent_gain = mse_s1_c - mse_c_d

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "diff_loss": {
            "self_teacher": diff_self,
            "stage1_reader_cross": diff_s1_c,
            "cross_no_delta": diff_c_nd,
            "cross_with_delta": diff_c_d,
            "self_memory_no_delta": diff_s_nd,
            "self_memory_with_delta": diff_s_d,
            "wrong_demo_no_delta": diff_w_nd,
            "wrong_demo_with_delta": diff_w_d,
            "reader_gain_over_stage1": reader_diff_gain,
            "delta_gain_over_cross_no_delta": delta_diff_gain,
            "total_gain_over_stage1": total_diff_gain,
            "demo_sensitivity_no_delta": diff_w_nd - diff_c_nd,
            "demo_sensitivity_with_delta": diff_w_d - diff_c_d,
        },
        "latent": {
            "s_z": s_z,
            "stage1_reader_cross_mse_raw": mse_s1_c,
            "cross_no_delta_mse_raw": mse_c_nd,
            "cross_with_delta_mse_raw": mse_c_d,
            "self_memory_no_delta_mse_raw": mse_s_nd,
            "self_memory_with_delta_mse_raw": mse_s_d,
            "wrong_demo_no_delta_mse_raw": mse_w_nd,
            "wrong_demo_with_delta_mse_raw": mse_w_d,
            "stage1_reader_cross_L_z": mse_s1_c / s_z2,
            "cross_no_delta_L_z": mse_c_nd / s_z2,
            "cross_with_delta_L_z": mse_c_d / s_z2,
            "self_memory_no_delta_L_z": mse_s_nd / s_z2,
            "self_memory_with_delta_L_z": mse_s_d / s_z2,
            "wrong_demo_no_delta_L_z": mse_w_nd / s_z2,
            "wrong_demo_with_delta_L_z": mse_w_d / s_z2,
            "reader_latent_mse_reduction_pct": 100.0
            * (1.0 - mse_c_nd / max(mse_s1_c, 1e-12)),
            "total_latent_mse_reduction_pct": 100.0
            * (1.0 - mse_c_d / max(mse_s1_c, 1e-12)),
            "delta_latent_mse_reduction_over_reader_pct": 100.0
            * (1.0 - mse_c_d / max(mse_c_nd, 1e-12)),
            "delta_improved_sample_frac": float(acc[35].item() / total),
            "stage1_reader_cross_cos": float(acc[16].item() / total),
            "cross_no_delta_cos": float(acc[17].item() / total),
            "cross_with_delta_cos": float(acc[18].item() / total),
            "self_memory_no_delta_cos": float(acc[19].item() / total),
            "self_memory_with_delta_cos": float(acc[20].item() / total),
            "cross_z_rms": cross_z_rms,
            "adapted_z_rms": zq_rms,
            "self_z_rms": self_z_rms,
            "cross_delta_rms": cross_delta_rms,
            "cross_delta_rel_rms_zc": cross_delta_rms / max(cross_z_rms, 1e-12),
            "cross_delta_rel_rms_sz": cross_delta_rms / max(s_z, 1e-12),
            "self_memory_delta_rms": self_ret_delta_rms,
            "self_teacher_delta_rms": self_teacher_delta_rms,
            "wrong_demo_delta_rms": wrong_delta_rms,
            "correction_direction_cosine_mean": float(acc[27].item() / total),
            "correction_projection_fraction_mean": float(acc[28].item() / total),
            "delta_to_gap_ratio_global": cross_delta_rms / max(gap_rms, 1e-12),
            "delta_to_gap_ratio_mean": float(acc[29].item() / total),
            "delta_mag_gap_pearson_r": mag_gap_r,
        },
        "source_state_alignment": {
            "cross_cos_href_hq": float(acc[58].item() / total),
            "cross_mse_href_hq": float(acc[59].item() / total),
            "self_cos_href_hq": float(acc[60].item() / total),
            "self_mse_href_hq": float(acc[61].item() / total),
            "wrong_demo_cos_href_hq": float(acc[62].item() / total),
            "wrong_demo_mse_href_hq": float(acc[63].item() / total),
        },
        "reader_attention": {
            "cross": {
                "valid_positions_mean": float(acc[44].item() / total),
                "entropy_mean": float(acc[36].item() / total),
                "uniform_entropy_mean": float(acc[37].item() / total),
                "normalized_entropy_mean": float(acc[38].item() / total),
                "max_weight_mean": float(acc[39].item() / total),
                "uniform_max_weight_mean": float(acc[40].item() / total),
                "max_weight_over_uniform_ratio": float(acc[39].item())
                / max(float(acc[40].item()), 1e-12),
                "effective_positions_exp_entropy": float(acc[41].item() / total),
                "effective_positions_ipr": float(acc[42].item() / total),
                "effective_fraction_mean": float(acc[43].item() / total),
            },
            "self": {
                "valid_positions_mean": float(acc[50].item() / total),
                "entropy_mean": float(acc[45].item() / total),
                "normalized_entropy_mean": float(acc[46].item() / total),
                "max_weight_mean": float(acc[47].item() / total),
                "effective_positions_exp_entropy": float(acc[48].item() / total),
                "effective_positions_ipr": float(acc[49].item() / total),
            },
            "wrong_demo": {
                "valid_positions_mean": float(acc[56].item() / total),
                "entropy_mean": float(acc[51].item() / total),
                "normalized_entropy_mean": float(acc[52].item() / total),
                "max_weight_mean": float(acc[53].item() / total),
                "effective_positions_exp_entropy": float(acc[54].item() / total),
                "effective_positions_ipr": float(acc[55].item() / total),
            },
        },
        "contributions": {
            "latent_mse_reader_gain": reader_latent_gain,
            "latent_mse_delta_gain": delta_latent_gain,
            "latent_mse_total_gain": total_latent_gain,
            "latent_mse_reader_share_pct": (
                100.0 * reader_latent_gain / total_latent_gain
                if abs(total_latent_gain) > 1e-12
                else 0.0
            ),
            "latent_mse_delta_share_pct": (
                100.0 * delta_latent_gain / total_latent_gain
                if abs(total_latent_gain) > 1e-12
                else 0.0
            ),
            "diff_loss_reader_gain": reader_diff_gain,
            "diff_loss_delta_gain": delta_diff_gain,
            "diff_loss_total_gain": total_diff_gain,
            "diff_loss_reader_share_pct": (
                100.0 * reader_diff_gain / total_diff_gain
                if abs(total_diff_gain) > 1e-12
                else 0.0
            ),
            "diff_loss_delta_share_pct": (
                100.0 * delta_diff_gain / total_diff_gain
                if abs(total_diff_gain) > 1e-12
                else 0.0
            ),
        },
    }

    if run_ddim:
        n_ddim = n_val if ddim_max_samples is None else min(int(ddim_max_samples), n_val)
        gen_ddim = torch.Generator(device="cpu").manual_seed(val_seed + 777)
        init_noise_all = torch.randn(
            (n_ddim, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_ddim
        )
        local_z0 = init_noise_all[np.arange(n_ddim)[rank::world]]

        modes = [
            "self_teacher",
            "stage1_reader_cross",
            "cross_no_delta",
            "cross_with_delta",
            "self_memory_no_delta",
            "self_memory_with_delta",
            "wrong_demo_no_delta",
            "wrong_demo_with_delta",
        ]
        err = {m: torch.zeros(11, device=device, dtype=torch.float64) for m in modes}

        c_it2 = bank.iter_e7_val_batches(
            "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        s_it2 = bank.iter_e7_val_batches(
            "self", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        w_it2 = bank.iter_e7_val_batches(
            "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        s1_it2 = bank.iter_val_batches(
            "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )

        cur = 0
        for (
            (hm_c, bm_c, mpm_c, q_c, z_self, a_c, _),
            (hm_s, bm_s, mpm_s, q_s, _, _, _),
            (hm_w, bm_w, mpm_w, q_w, _, a_w, _),
            (dm_s1, pm_s1, _, _, _, _),
        ) in zip(c_it2, s_it2, w_it2, s1_it2):
            B = len(hm_c)
            z0 = local_z0[cur : cur + B].to(device, non_blocking=True)
            cur += B
            h_q = E5Policy.format_query_tokens(q_c)
            with amp_ctx:
                z_c0 = rr_teacher.prompt_with_obs_decoder(
                    tgt=h_q + rr_teacher.prompt_current_obs_pos_emb,
                    memory=dm_s1,
                    memory_key_padding_mask=pm_s1,
                    need_weights=False,
                )
                z_c, h_ref_c, _, _ = policy.read_cross_latent_and_source(
                    hm_c, bm_c, mpm_c, q_c, h_q=h_q
                )
                z_q, _ = policy.transfer_latent(z_c, h_ref_c, h_q)
                z_s_ret, h_ref_s, _, _ = policy.read_cross_latent_and_source(
                    hm_s, bm_s, mpm_s, q_s, h_q=h_q
                )
                z_s_q, _ = policy.transfer_latent(z_s_ret, h_ref_s, h_q)
                z_w, h_ref_w, _, _ = policy.read_cross_latent_and_source(
                    hm_w, bm_w, mpm_w, q_w, h_q=h_q
                )
                z_w_q, _ = policy.transfer_latent(z_w, h_ref_w, h_q)

                pred_self = policy.teacher.sample_from_z_self(z_self, init_noise=z0)
                pred_s1_c = policy.teacher.sample_from_z_self(z_c0, init_noise=z0)
                pred_c_nd = policy.teacher.sample_from_z_self(z_c, init_noise=z0)
                pred_c_d = policy.teacher.sample_from_z_self(z_q, init_noise=z0)
                pred_s_nd = policy.teacher.sample_from_z_self(z_s_ret, init_noise=z0)
                pred_s_d = policy.teacher.sample_from_z_self(z_s_q, init_noise=z0)
                pred_w_nd = policy.teacher.sample_from_z_self(z_w, init_noise=z0)
                pred_w_d = policy.teacher.sample_from_z_self(z_w_q, init_noise=z0)

            err["self_teacher"] += _action_error_sums(pred_self, a_c)
            err["stage1_reader_cross"] += _action_error_sums(pred_s1_c, a_c)
            err["cross_no_delta"] += _action_error_sums(pred_c_nd, a_c)
            err["cross_with_delta"] += _action_error_sums(pred_c_d, a_c)
            err["self_memory_no_delta"] += _action_error_sums(pred_s_nd, a_c)
            err["self_memory_with_delta"] += _action_error_sums(pred_s_d, a_c)
            err["wrong_demo_no_delta"] += _action_error_sums(pred_w_nd, a_w)
            err["wrong_demo_with_delta"] += _action_error_sums(pred_w_d, a_w)

        ddim_out: Dict[str, Any] = {
            m: _unpack_action_errors(_all_reduce_sum(err[m])) for m in modes
        }
        s1_mse10 = ddim_out["stage1_reader_cross"]["action10_mse"]
        nd_mse10 = ddim_out["cross_no_delta"]["action10_mse"]
        cd_mse10 = ddim_out["cross_with_delta"]["action10_mse"]
        reader_mse10_gain = s1_mse10 - nd_mse10
        delta_mse10_gain = nd_mse10 - cd_mse10
        total_mse10_gain = s1_mse10 - cd_mse10

        s1_mae7 = ddim_out["stage1_reader_cross"]["action7_mae"]
        nd_mae7 = ddim_out["cross_no_delta"]["action7_mae"]
        cd_mae7 = ddim_out["cross_with_delta"]["action7_mae"]

        ddim_out["reader_gain_action10_mse"] = reader_mse10_gain
        ddim_out["delta_gain_action10_mse"] = delta_mse10_gain
        ddim_out["total_gain_action10_mse"] = total_mse10_gain
        ddim_out["reader_gain_action7_mae"] = s1_mae7 - nd_mae7
        ddim_out["delta_gain_action7_mae"] = nd_mae7 - cd_mae7
        ddim_out["total_gain_action7_mae"] = s1_mae7 - cd_mae7
        ddim_out["demo_sensitivity_no_delta_action10_mse"] = (
            ddim_out["wrong_demo_no_delta"]["action10_mse"] - nd_mse10
        )
        ddim_out["demo_sensitivity_action10_mse"] = (
            ddim_out["wrong_demo_with_delta"]["action10_mse"] - cd_mse10
        )
        ddim_out["demo_sensitivity_action7_mae"] = (
            ddim_out["wrong_demo_with_delta"]["action7_mae"] - cd_mae7
        )
        result["ddim"] = ddim_out
        result["contributions"].update(
            {
                "ddim_action10_mse_reader_gain": reader_mse10_gain,
                "ddim_action10_mse_delta_gain": delta_mse10_gain,
                "ddim_action10_mse_total_gain": total_mse10_gain,
                "ddim_action10_mse_reader_share_pct": (
                    100.0 * reader_mse10_gain / total_mse10_gain
                    if abs(total_mse10_gain) > 1e-12
                    else 0.0
                ),
                "ddim_action10_mse_delta_share_pct": (
                    100.0 * delta_mse10_gain / total_mse10_gain
                    if abs(total_mse10_gain) > 1e-12
                    else 0.0
                ),
            }
        )

    if run_multi_demo:
        pool2 = np.flatnonzero(bank.val_has_two_demos)
        n_md = (
            len(pool2)
            if multi_demo_max_samples is None
            else min(int(multi_demo_max_samples), len(pool2))
        )
        if n_md > 0:
            gen_md = torch.Generator(device="cpu").manual_seed(val_seed + 3030)
            z0_all = torch.randn((n_md, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_md)
            sel2 = pool2[:n_md][rank::world]
            local_z0_md = z0_all[np.arange(n_md)[rank::world]]

            md_acc = torch.zeros(7, device=device, dtype=torch.float64)
            err_d1 = torch.zeros(11, device=device, dtype=torch.float64)
            err_d2 = torch.zeros(11, device=device, dtype=torch.float64)
            err_d1_vs_d2 = torch.zeros(11, device=device, dtype=torch.float64)
            err_d1_vs_d2_nd = torch.zeros(11, device=device, dtype=torch.float64)

            for start in range(0, len(sel2), batch_size):
                end = min(start + batch_size, len(sel2))
                s_idx = sel2[start:end]
                z0 = local_z0_md[start:end].to(device, non_blocking=True)
                B = len(s_idx)
                hm1, bm1, pm1, q1, zs1, a1, _ = bank.gather_e7_batch(
                    split="val",
                    sample_indices=s_idx,
                    demo_episodes=bank.val_cross_a[s_idx],
                    is_self=np.zeros(B, dtype=bool),
                )
                hm2, bm2, pm2, q2, _, _, _ = bank.gather_e7_batch(
                    split="val",
                    sample_indices=s_idx,
                    demo_episodes=bank.val_cross_b[s_idx],
                    is_self=np.zeros(B, dtype=bool),
                )
                with amp_ctx:
                    zc1, href1, hq, _ = policy.read_cross_latent_and_source(hm1, bm1, pm1, q1)
                    zc2, href2, _, _ = policy.read_cross_latent_and_source(
                        hm2, bm2, pm2, q2, h_q=hq
                    )
                    za1, _ = policy.transfer_latent(zc1, href1, hq)
                    za2, _ = policy.transfer_latent(zc2, href2, hq)
                    p1_nd = policy.teacher.sample_from_z_self(zc1, init_noise=z0)
                    p2_nd = policy.teacher.sample_from_z_self(zc2, init_noise=z0)
                    p1 = policy.teacher.sample_from_z_self(za1, init_noise=z0)
                    p2 = policy.teacher.sample_from_z_self(za2, init_noise=z0)

                zc1_f, zc2_f = zc1.float(), zc2.float()
                za1_f, za2_f = za1.float(), za2.float()
                zs1_f = zs1.float()

                md_acc[0] += float(B)
                md_acc[1] += F.cosine_similarity(
                    zc1_f.reshape(B, -1), zc2_f.reshape(B, -1), dim=-1
                ).double().sum()
                md_acc[2] += F.cosine_similarity(
                    za1_f.reshape(B, -1), za2_f.reshape(B, -1), dim=-1
                ).double().sum()
                md_acc[3] += (zc1_f - zc2_f).square().mean(dim=(1, 2)).double().sum()
                md_acc[4] += (za1_f - za2_f).square().mean(dim=(1, 2)).double().sum()
                md_acc[5] += ((za1_f - zs1_f) / s_z).square().mean(dim=(1, 2)).double().sum()
                md_acc[6] += ((za2_f - zs1_f) / s_z).square().mean(dim=(1, 2)).double().sum()

                err_d1 += _action_error_sums(p1, a1)
                err_d2 += _action_error_sums(p2, a1)
                err_d1_vs_d2 += _action_error_sums(p1, p2)
                err_d1_vs_d2_nd += _action_error_sums(p1_nd, p2_nd)

            md_acc = _all_reduce_sum(md_acc)
            cnt_md = max(float(md_acc[0].item()), 1.0)
            u_d1 = _unpack_action_errors(_all_reduce_sum(err_d1))
            u_d2 = _unpack_action_errors(_all_reduce_sum(err_d2))
            u_diff = _unpack_action_errors(_all_reduce_sum(err_d1_vs_d2))
            u_diff_nd = _unpack_action_errors(_all_reduce_sum(err_d1_vs_d2_nd))
            result["multi_demo_consistency"] = {
                "count": int(round(cnt_md)),
                "latent_cos_d1_d2_no_delta": float(md_acc[1].item() / cnt_md),
                "latent_cos_d1_d2_with_delta": float(md_acc[2].item() / cnt_md),
                "latent_mse_d1_d2_no_delta": float(md_acc[3].item() / cnt_md),
                "latent_mse_d1_d2_with_delta": float(md_acc[4].item() / cnt_md),
                "demo1_L_z": float(md_acc[5].item() / cnt_md),
                "demo2_L_z": float(md_acc[6].item() / cnt_md),
                "demo1_action10_mse": u_d1["action10_mse"],
                "demo2_action10_mse": u_d2["action10_mse"],
                "demo1_action7_mae": u_d1["action7_mae"],
                "demo2_action7_mae": u_d2["action7_mae"],
                "pred_d1_vs_d2_action10_mse_no_delta": u_diff_nd["action10_mse"],
                "pred_d1_vs_d2_action10_mse_with_delta": u_diff["action10_mse"],
            }

    return result


@torch.inference_mode()
def evaluate_source_state_shuffle(
    policy: E8Policy,
    bank: E7TransferBank,
    *,
    batch_size: int = 128,
    max_val_samples: Optional[int] = None,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Offline inference-time source-state shuffle diagnostic.

    Fixes `(z_c, h_q)` from normal Cross retrieval and replaces ONLY `h_ref`:
      1. `orig`                  : matched `(z_c, h_ref, h_q)`
      2. `cross_sample_shuffle`  : `h_ref` from a different val sample/episode (deterministic derangement)
      3. `same_demo_phase_shift` : `h_ref` read from the SAME Demo with circularly shifted weights
      4. `wrong_demo_href`       : `h_ref` read from the wrong-task Demo
      5. `target_copy_href`      : `h_ref <- h_q` (pretending zero source-to-target scene shift)

    Uses the exact same `val_seed=12345` noise & DDIM init noise as `evaluate_e8_validation`.
    """
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    s_z = float(policy.latent_scale_s_z.item())
    s_z2 = s_z * s_z
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    pool = np.flatnonzero(bank.val_valid_cross_mask)
    n_val = len(pool) if max_val_samples is None else min(int(max_val_samples), len(pool))
    sel_all = pool[:n_val]

    # Deterministic cross-episode derangement over `0..n_val-1`
    perm = _build_cross_episode_derangement(
        bank.val_episodes[sel_all], bank.val_cross_a[sel_all], seed=val_seed + 9090
    )
    shuf_sel_all = sel_all[perm]

    gen_diff = torch.Generator(device="cpu").manual_seed(val_seed)
    all_noise = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_diff)
    all_pertub = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_diff)
    all_t = torch.randint(
        0,
        policy.teacher.noise_scheduler.config.num_train_timesteps,
        (n_val,),
        generator=gen_diff,
        dtype=torch.long,
    )
    gen_ddim = torch.Generator(device="cpu").manual_seed(val_seed + 777)
    all_z0 = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_ddim)

    local_pos = np.arange(n_val)[rank::world]
    local_sel = sel_all[local_pos]
    local_shuf_sel = shuf_sel_all[local_pos]
    local_noise = all_noise[local_pos]
    local_pertub = all_pertub[local_pos]
    local_t = all_t[local_pos]
    local_z0 = all_z0[local_pos]

    conds = (
        "orig",
        "cross_sample_shuffle",
        "same_demo_phase_shift",
        "wrong_demo_href",
        "target_copy_href",
    )
    # Per-condition scalar sums: [count, diff_loss, latent_mse_raw, latent_cos, zq_change_sq, href_change_sq, href_cos_orig]
    sc = {c: torch.zeros(7, device=device, dtype=torch.float64) for c in conds}
    err_gt = {c: torch.zeros(11, device=device, dtype=torch.float64) for c in conds}
    err_vs_orig = {c: torch.zeros(11, device=device, dtype=torch.float64) for c in conds}

    for start in range(0, len(local_sel), batch_size):
        end = min(start + batch_size, len(local_sel))
        s_idx = local_sel[start:end]
        sh_idx = local_shuf_sel[start:end]
        B = len(s_idx)
        noise = local_noise[start:end].to(device, non_blocking=True)
        pertub = local_pertub[start:end].to(device, non_blocking=True)
        t = local_t[start:end].to(device, non_blocking=True)
        z0 = local_z0[start:end].to(device, non_blocking=True)

        hm_c, bm_c, mpm_c, q_c, z_self, a_c, _ = bank.gather_e7_batch(
            split="val",
            sample_indices=s_idx,
            demo_episodes=bank.val_cross_a[s_idx],
            is_self=np.zeros(B, dtype=bool),
        )
        hm_sh, bm_sh, mpm_sh, q_sh, _, _, _ = bank.gather_e7_batch(
            split="val",
            sample_indices=sh_idx,
            demo_episodes=bank.val_cross_a[sh_idx],
            is_self=np.zeros(B, dtype=bool),
        )
        hm_w, bm_w, mpm_w, q_w, _, _, _ = bank.gather_e7_batch(
            split="val",
            sample_indices=s_idx,
            demo_episodes=bank.val_wrong_demos[s_idx],
            is_self=np.zeros(B, dtype=bool),
        )

        h_q = E5Policy.format_query_tokens(q_c)
        with amp_ctx:
            # Fixed z_c and h_q from normal Cross retrieval
            z_c, h_ref_orig, _, w_c = policy.read_cross_latent_and_source(
                hm_c, bm_c, mpm_c, q_c, h_q=h_q
            )
            # Replacement source states
            _, h_ref_shuf, _, _ = policy.read_cross_latent_and_source(hm_sh, bm_sh, mpm_sh, q_sh)
            w_phase = _shift_weights_within_demo(w_c, mpm_c)
            h_ref_phase = torch.einsum("bm,bmkd->bkd", w_phase.to(dtype=hm_c.dtype), hm_c)
            _, h_ref_wrong, _, _ = policy.read_cross_latent_and_source(
                hm_w, bm_w, mpm_w, q_w, h_q=h_q
            )
            h_ref_map = {
                "orig": h_ref_orig,
                "cross_sample_shuffle": h_ref_shuf,
                "same_demo_phase_shift": h_ref_phase,
                "wrong_demo_href": h_ref_wrong,
                "target_copy_href": h_q,
            }

            zq_map: Dict[str, torch.Tensor] = {}
            pred_map: Dict[str, torch.Tensor] = {}
            diff_map: Dict[str, torch.Tensor] = {}
            for c in conds:
                zq_c, _ = policy.transfer_latent(z_c, h_ref_map[c], h_q)
                zq_map[c] = zq_c
                diff_map[c] = policy.teacher.diffusion_loss_from_z_self(
                    zq_c, a_c, noise=noise, pertub_noise=pertub, timesteps=t
                )
                pred_map[c] = policy.teacher.sample_from_z_self(zq_c, init_noise=z0)

        zs_f = z_self.float()
        zq_orig_f = zq_map["orig"].float()
        href_orig_f = h_ref_orig.float()
        pred_orig = pred_map["orig"]

        for c in conds:
            zq_f = zq_map[c].float()
            hr_f = h_ref_map[c].float()
            sc[c][0] += float(B)
            sc[c][1] += diff_map[c].double().sum()
            sc[c][2] += (zq_f - zs_f).square().mean(dim=(1, 2)).double().sum()
            sc[c][3] += F.cosine_similarity(
                zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).double().sum()
            sc[c][4] += (zq_f - zq_orig_f).square().mean(dim=(1, 2)).double().sum()
            sc[c][5] += (hr_f - href_orig_f).square().mean(dim=(1, 2)).double().sum()
            sc[c][6] += F.cosine_similarity(
                hr_f.reshape(B, -1), href_orig_f.reshape(B, -1), dim=-1
            ).double().sum()

            err_gt[c] += _action_error_sums(pred_map[c], a_c)
            err_vs_orig[c] += _action_error_sums(pred_map[c], pred_orig)

    out_conds: Dict[str, Any] = {}
    for c in conds:
        s_red = _all_reduce_sum(sc[c])
        n = max(float(s_red[0].item()), 1.0)
        mse_raw = float(s_red[2].item() / n)
        gt_u = _unpack_action_errors(_all_reduce_sum(err_gt[c]))
        vo_u = _unpack_action_errors(_all_reduce_sum(err_vs_orig[c]))
        out_conds[c] = {
            "val_samples": int(round(n)),
            "diff_loss": float(s_red[1].item() / n),
            "latent_mse_raw": mse_raw,
            "L_z": mse_raw / s_z2,
            "latent_cos": float(s_red[3].item() / n),
            "zq_rms_change_vs_orig": math.sqrt(max(float(s_red[4].item() / n), 0.0)),
            "href_rms_change_vs_orig": math.sqrt(max(float(s_red[5].item() / n), 0.0)),
            "href_cos_vs_orig": float(s_red[6].item() / n),
            "ddim_gt": gt_u,
            "pred_shift_vs_orig_action10_mse": vo_u["action10_mse"],
            "pred_shift_vs_orig_action7_mae": vo_u["action7_mae"],
        }

    base = out_conds["orig"]
    deltas: Dict[str, Any] = {}
    for c in conds:
        if c == "orig":
            continue
        cur_c = out_conds[c]
        deltas[c] = {
            "delta_diff_loss": cur_c["diff_loss"] - base["diff_loss"],
            "delta_diff_loss_pct": 100.0
            * (cur_c["diff_loss"] - base["diff_loss"])
            / max(base["diff_loss"], 1e-12),
            "delta_latent_mse_raw": cur_c["latent_mse_raw"] - base["latent_mse_raw"],
            "delta_latent_mse_pct": 100.0
            * (cur_c["latent_mse_raw"] - base["latent_mse_raw"])
            / max(base["latent_mse_raw"], 1e-12),
            "delta_L_z": cur_c["L_z"] - base["L_z"],
            "delta_ddim_action10_mse": cur_c["ddim_gt"]["action10_mse"]
            - base["ddim_gt"]["action10_mse"],
            "delta_ddim_action10_mse_pct": 100.0
            * (cur_c["ddim_gt"]["action10_mse"] - base["ddim_gt"]["action10_mse"])
            / max(base["ddim_gt"]["action10_mse"], 1e-12),
            "delta_ddim_action7_mae": cur_c["ddim_gt"]["action7_mae"]
            - base["ddim_gt"]["action7_mae"],
            "delta_ddim_action7_mae_pct": 100.0
            * (cur_c["ddim_gt"]["action7_mae"] - base["ddim_gt"]["action7_mae"])
            / max(base["ddim_gt"]["action7_mae"], 1e-12),
            "delta_exec12_action7_mae": cur_c["ddim_gt"]["exec12_action7_mae"]
            - base["ddim_gt"]["exec12_action7_mae"],
            "delta_grip_sign_acc_pp": 100.0
            * (cur_c["ddim_gt"]["grip_sign_acc"] - base["ddim_gt"]["grip_sign_acc"]),
            "zq_rms_change_vs_orig": cur_c["zq_rms_change_vs_orig"],
            "pred_shift_vs_orig_action10_mse": cur_c["pred_shift_vs_orig_action10_mse"],
            "pred_shift_vs_orig_action7_mae": cur_c["pred_shift_vs_orig_action7_mae"],
        }

    return {
        "val_samples": int(n_val),
        "val_seed": int(val_seed),
        "shuffle_seed": int(val_seed + 9090),
        "conditions": out_conds,
        "degradation_vs_orig": deltas,
    }


def run_training(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if use_ddp:
        dist.init_process_group(backend="nccl", device_id=device)

    try:
        rank = _rank()
        world = _world()
        if args.global_batch % world != 0:
            raise ValueError(
                f"global_batch ({args.global_batch}) must be divisible by world_size ({world})"
            )
        per_rank_batch = args.global_batch // world

        if args.smoke:
            if args.run_name is None:
                args.run_name = "smoke"
            if args.epochs == 8:
                args.epochs = 2
            if args.max_train_samples is None:
                args.max_train_samples = args.global_batch * 60
            if args.max_val_samples is None:
                args.max_val_samples = 512
            if args.ddim_val_samples is None:
                args.ddim_val_samples = 256
            if args.multi_demo_samples == 1024:
                args.multi_demo_samples = 256
            args.final_ddim_samples = args.ddim_val_samples
            args.final_multi_demo_samples = args.multi_demo_samples
            if args.warmup_steps == 500:
                args.warmup_steps = 10

        run_name = args.run_name if args.run_name else "e8_source_conditioned"
        run_dir = args.output_dir / run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        seed_everything(args.seed)
        bank = E7TransferBank(
            clip_cache_dir=args.clip_cache_dir,
            z_self_cache_dir=args.z_self_cache_dir,
            behavior_mem_dir=args.behavior_mem_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            z_self_on_device=(device.type == "cuda" and not args.z_self_on_cpu),
            self_ratio=args.self_ratio,
            seed=args.seed,
        )

        s_z = float(args.latent_scale_s_z)
        if abs(s_z - bank.latent_scale_s_z) > 1e-4:
            raise RuntimeError(
                f"Configured s_z={s_z} diverges from cached train_z_self_rms={bank.latent_scale_s_z}"
            )

        if args.direct_bias_mode == "train_z_self_mean":
            direct_latent_bias = compute_train_z_self_mean(bank)
        else:
            direct_latent_bias = torch.zeros(
                (1, bank.z_self_train.shape[1], EMBED_DIM), dtype=torch.float32
            )

        # Load and freeze Stage 1 Teacher ONLY; NEVER load any E7/Stage 2 Reader or Transformer!
        teacher = load_frozen_stage1_teacher(args.stage1_ckpt, device=device)
        seed_everything(args.seed)
        policy = E8Policy(
            teacher,
            latent_scale_s_z=s_z,
            reader_attn_dim=args.reader_attn_dim,
            transformer_layers=args.transformer_layers,
            transformer_dropout=args.transformer_dropout,
            direct_latent_bias=direct_latent_bias,
            direct_out_proj_init_std=args.direct_out_proj_init_std,
        ).to(device)

        # Strict gradient & freeze audit on rank 0 before training
        if rank == 0 and args.check_grads:
            policy.train()
            hm_g, bm_g, mpm_g, q_g, zs_g, a_g, is_s_g = next(
                bank.iter_e7_train_batches(1, min(8, per_rank_batch), max_samples=16)
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                out_g = policy(
                    hm_g,
                    bm_g,
                    mpm_g,
                    q_g,
                    zs_g,
                    a_g,
                    is_s_g,
                    lambda_a=1.0,
                    lambda_z=0.0,
                )
            out_g["loss"].backward()

            out_proj_gn = float(
                policy.transfer_transformer.out_proj.weight.grad.float().norm().item()
            )
            src_norm_gn = float(
                policy.transfer_transformer.source_in_norm.weight.grad.float().norm().item()
            )
            beh_type_gn = float(
                policy.transfer_transformer.behavior_type_emb.grad.float().norm().item()
            )
            src_type_gn = float(
                policy.transfer_transformer.source_type_emb.grad.float().norm().item()
            )
            tgt_type_gn = float(
                policy.transfer_transformer.target_type_emb.grad.float().norm().item()
            )
            q_gn = float(policy.reader.q_proj.weight.grad.float().norm().item())
            k_gn = float(policy.reader.k_proj.weight.grad.float().norm().item())
            teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
            policy.zero_grad(set_to_none=True)

            print(
                f"[E8 Grad Check (pure L_diff via frozen UNet)] "
                f"xf.out_proj_gn={out_proj_gn:.6f} | xf.src_norm_gn={src_norm_gn:.6e} | "
                f"type_emb_gn=(beh:{beh_type_gn:.3e}, src:{src_type_gn:.3e}, tgt:{tgt_type_gn:.3e}) | "
                f"reader.q_gn={q_gn:.6e} | reader.k_gn={k_gn:.6e} | "
                f"teacher_any_grad={teacher_has_grad} | diff_loss={float(out_g['diff_loss'].item()):.4f}",
                flush=True,
            )
            checks = [
                out_proj_gn,
                src_norm_gn,
                beh_type_gn,
                src_type_gn,
                tgt_type_gn,
                q_gn,
                k_gn,
            ]
            if any(v <= 0.0 or not math.isfinite(v) for v in checks) or teacher_has_grad:
                raise RuntimeError("E8 gradient / freeze audit FAILED!")
        if use_ddp:
            dist.barrier()

        optim_groups = policy.get_optim_groups(
            reader_lr=args.reader_lr,
            transformer_lr=args.transformer_lr,
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(args.beta1, args.beta2),
        )

        epoch_len = (
            min(args.max_train_samples, bank.num_train_samples())
            if args.max_train_samples is not None
            else bank.num_train_samples()
        )
        steps_per_epoch = math.ceil(epoch_len / args.global_batch)
        total_steps = steps_per_epoch * args.epochs
        lr_scheduler = get_cosine_schedule_with_min_ratio(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        trainable_params = policy.trainable_parameters()
        ema = (
            EMAModel(
                trainable_params,
                decay=args.ema_decay,
                min_decay=0.0,
                update_after_step=0,
                use_ema_warmup=False,
                inv_gamma=1.0,
                power=0.75,
                foreach=True,
            ).to(device)
            if args.use_ema
            else None
        )

        config = {
            "stage": "e8_source_conditioned_transfer",
            "prediction_mode": "direct",
            "source_conditioned": True,
            "direct_bias_mode": args.direct_bias_mode,
            "direct_out_proj_init_std": float(args.direct_out_proj_init_std),
            "direct_latent_bias_rms": float(
                policy.direct_latent_bias.float().square().mean().sqrt().item()
            ),
            "run_name": run_name,
            "smoke": bool(args.smoke),
            "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
            "clip_cache_dir": str(Path(args.clip_cache_dir).resolve()),
            "z_self_cache_dir": str(Path(args.z_self_cache_dir).resolve()),
            "behavior_mem_dir": str(Path(args.behavior_mem_dir).resolve()),
            "behavior_stride": int(bank.behavior_mem_meta["behavior_stride"]),
            "latent_scale_s_z": s_z,
            "lambda_a": args.lambda_a,
            "lambda_diff": args.lambda_a,
            "lambda_z": args.lambda_z,
            "self_ratio": args.self_ratio,
            "cross_ratio": round(1.0 - args.self_ratio, 6),
            "reader_attn_dim": args.reader_attn_dim,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": 8,
            "transformer_ffn_dim": 3072,
            "transformer_dropout": args.transformer_dropout,
            "samples_per_epoch": epoch_len,
            "total_cached_train_starts": bank.num_train_samples(),
            "total_cached_val_cross_starts": int(bank.val_valid_cross_mask.sum()),
            "val_multi_demo_starts": int(bank.val_has_two_demos.sum()),
            "seed": args.seed,
            "val_seed": args.val_seed,
            "epochs": args.epochs,
            "world_size": world,
            "global_batch": args.global_batch,
            "per_rank_batch": per_rank_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "reader_lr": args.reader_lr,
            "reader_min_lr": args.reader_lr * args.min_lr_ratio,
            "transformer_lr": args.transformer_lr,
            "transformer_min_lr": args.transformer_lr * args.min_lr_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_steps": args.warmup_steps,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "clip_grad_norm": args.clip_grad_norm,
            "mixed_precision": "bf16",
            "use_ema": bool(args.use_ema),
            "ema_decay": args.ema_decay if args.use_ema else None,
            "best_selection_metric": "ddim.cross_with_delta.action10_mse",
            "parameter_summary": policy.parameter_summary(),
        }

        last_ckpt = run_dir / "last.pt"
        best_ckpt = run_dir / "best.pt"
        history: List[Dict[str, Any]] = []
        best_val_mse = float("inf")
        best_epoch = 0
        start_epoch = 1
        global_step = 0

        if args.resume and last_ckpt.is_file():
            ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            if ck.get("stage") != "e8_source_conditioned_transfer":
                raise RuntimeError(f"Refusing to resume non-E8 checkpoint: {ck.get('stage')}")
            if ck.get("direct_latent_bias") is not None:
                policy.set_direct_latent_bias(ck["direct_latent_bias"])
            policy.reader.load_state_dict(ck["reader"], strict=True)
            policy.transfer_transformer.load_state_dict(ck["transfer_transformer"], strict=True)
            if ema is not None and ck.get("ema") is not None:
                ema.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["optimizer"])
            lr_scheduler.load_state_dict(ck["lr_scheduler"])
            history = ck["history"]
            best_val_mse = ck["best_val_cross_ddim_mse10"]
            best_epoch = ck["best_epoch"]
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
            if rank == 0:
                print(
                    f"[E8] Resumed from epoch {ck['epoch']} "
                    f"(best_epoch={best_epoch}, best_val_cross_ddim_mse10={best_val_mse:.5f})",
                    flush=True,
                )

        ddp_model: torch.nn.Module = (
            DistributedDataParallel(
                policy, device_ids=[device.index], find_unused_parameters=False
            )
            if use_ddp
            else policy
        )
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type=device.type, enabled=False)
        )

        if rank == 0:
            atomic_write_json(config, run_dir / "config.json")
            ps = config["parameter_summary"]
            print(
                f"[E8] run={run_name} | source_conditioned=True (direct) | "
                f"world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={args.warmup_steps}) | "
                f"reader_lr={args.reader_lr:.2e}->{args.reader_lr * args.min_lr_ratio:.2e} | "
                f"xf_lr={args.transformer_lr:.2e}->{args.transformer_lr * args.min_lr_ratio:.2e} | "
                f"weights=(diff:{args.lambda_a}, z:{args.lambda_z}) s_z={s_z:.6f} "
                f"bias_rms={config['direct_latent_bias_rms']:.4f} | "
                f"trainable={ps['total_trainable']:,} "
                f"(reader={ps['reader_trainable']:,}, "
                f"xf={ps['transfer_transformer_trainable']:,}, "
                f"teacher_frozen={ps['stage1_teacher_frozen']:,})",
                flush=True,
            )

        # Epoch-0 reference validation
        init_val = None
        if start_epoch == 1:
            init_val = evaluate_e8_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=True,
                ddim_max_samples=args.ddim_val_samples,
                run_multi_demo=True,
                multi_demo_max_samples=args.multi_demo_samples,
                val_seed=args.val_seed,
            )
            if rank == 0:
                atomic_write_json(init_val, run_dir / "init_val.json")
                dl = init_val["diff_loss"]
                lt = init_val["latent"]
                dm = init_val["ddim"]
                at = init_val["reader_attention"]["cross"]
                print(
                    f"[E8 Init Val] "
                    f"zq_rms={lt['adapted_z_rms']:.4f} zc_rms={lt['cross_z_rms']:.4f} "
                    f"zs_rms={lt['self_z_rms']:.4f} | "
                    f"diff self={dl['self_teacher']:.4f} c_nd={dl['cross_no_delta']:.4f} "
                    f"c+d={dl['cross_with_delta']:.4f} | Lz={lt['cross_with_delta_L_z']:.4f} | "
                    f"attn H_norm={at['normalized_entropy_mean']:.4f} max_w={at['max_weight_mean']:.4f} "
                    f"eff_pos={at['effective_positions_exp_entropy']:.2f}/{at['valid_positions_mean']:.2f} | "
                    f"DDIM mse10 self={dm['self_teacher']['action10_mse']:.4f} "
                    f"c_nd={dm['cross_no_delta']['action10_mse']:.4f} "
                    f"c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"(mae7={dm['cross_with_delta']['action7_mae']:.4f})",
                    flush=True,
                )

        if start_epoch == 1:
            seed_everything(args.seed + 1000 + rank)

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            policy.teacher.eval()

            tr_acc = torch.zeros(20, device=device, dtype=torch.float64)

            for (
                h_mem,
                b_mem,
                mem_pad_mask,
                query_rgb,
                z_self,
                actions_10d,
                is_self,
            ) in bank.iter_e7_train_batches(
                epoch,
                per_rank_batch,
                rank=rank,
                world_size=world,
                max_samples=args.max_train_samples,
            ):
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(
                        h_mem,
                        b_mem,
                        mem_pad_mask,
                        query_rgb,
                        z_self,
                        actions_10d,
                        is_self,
                        lambda_a=args.lambda_a,
                        lambda_z=args.lambda_z,
                    )
                out["loss"].backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, args.clip_grad_norm
                    ).item()
                )
                optimizer.step()
                lr_scheduler.step()
                if ema is not None:
                    ema.step(trainable_params)
                global_step += 1

                B = float(len(h_mem))
                ns = float(out["n_self"].item())
                nc = float(out["n_cross"].item())
                tr_acc[0] += B
                tr_acc[1] += float(out["loss"].item()) * B
                tr_acc[2] += float(out["diff_loss"].item()) * B
                tr_acc[3] += float(out["latent_loss_norm"].item()) * B
                tr_acc[4] += float(out["nodelta_loss_norm"].item()) * B
                tr_acc[5] += float(out["cos_nodelta_self"].item()) * B
                tr_acc[6] += float(out["cos_adapted_self"].item()) * B
                tr_acc[7] += float(out["dir_cos"].item()) * B
                tr_acc[8] += (float(out["cross_delta_rms"].item()) ** 2) * nc
                tr_acc[9] += nc
                tr_acc[10] += (float(out["self_delta_rms"].item()) ** 2) * ns
                tr_acc[11] += ns
                tr_acc[12] += grad_norm
                tr_acc[13] += 1.0
                tr_acc[14] += float(out["attn_entropy"].item()) * B
                tr_acc[15] += float(out["attn_norm_entropy"].item()) * B
                tr_acc[16] += float(out["attn_max_weight"].item()) * B
                tr_acc[17] += float(out["attn_eff_pos"].item()) * B
                tr_acc[18] += float(out["attn_eff_ipr"].item()) * B
                tr_acc[19] += float(out["source_target_cos"].item()) * B

            tr_acc = _all_reduce_sum(tr_acc)
            t_train = time.monotonic() - t_ep

            n_tot = max(float(tr_acc[0].item()), 1.0)
            n_cross_tot = max(float(tr_acc[9].item()), 1.0)
            n_self_tot = max(float(tr_acc[11].item()), 1.0)
            n_steps_tot = max(float(tr_acc[13].item()), 1.0)
            last_lrs = lr_scheduler.get_last_lr()
            train_metrics = {
                "samples": int(round(n_tot)),
                "self_fraction": float(tr_acc[11].item() / n_tot),
                "loss": float(tr_acc[1].item() / n_tot),
                "diff_loss": float(tr_acc[2].item() / n_tot),
                "L_z": float(tr_acc[3].item() / n_tot),
                "nodelta_L_z": float(tr_acc[4].item() / n_tot),
                "cos_nodelta_self": float(tr_acc[5].item() / n_tot),
                "cos_adapted_self": float(tr_acc[6].item() / n_tot),
                "dir_cos": float(tr_acc[7].item() / n_tot),
                "source_target_cos": float(tr_acc[19].item() / n_tot),
                "cross_delta_rms": math.sqrt(float(tr_acc[8].item() / n_cross_tot)),
                "self_delta_rms": math.sqrt(float(tr_acc[10].item() / n_self_tot)),
                "attn_entropy": float(tr_acc[14].item() / n_tot),
                "attn_norm_entropy": float(tr_acc[15].item() / n_tot),
                "attn_max_weight": float(tr_acc[16].item() / n_tot),
                "attn_eff_pos_exp": float(tr_acc[17].item() / n_tot),
                "attn_eff_pos_ipr": float(tr_acc[18].item() / n_tot),
                "grad_norm_mean": float(tr_acc[12].item() / n_steps_tot),
                "reader_lr": float(last_lrs[0]),
                "transformer_lr": float(last_lrs[2]),
                "ema_decay": float(ema.cur_decay_value or 0.0) if ema is not None else None,
                "train_seconds": round(t_train, 2),
                "steps_per_sec": round((n_steps_tot / world) / max(t_train, 1e-6), 2),
            }

            t_val0 = time.monotonic()
            if ema is not None:
                ema.store(trainable_params)
                ema.copy_to(trainable_params)

            val_metrics = evaluate_e8_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=True,
                ddim_max_samples=args.ddim_val_samples,
                run_multi_demo=(epoch == 1 or epoch == args.epochs or epoch % 2 == 0),
                multi_demo_max_samples=args.multi_demo_samples,
                val_seed=args.val_seed,
            )
            val_metrics["val_seconds"] = round(time.monotonic() - t_val0, 2)

            cur_monitor = val_metrics["ddim"]["cross_with_delta"]["action10_mse"]
            is_best = cur_monitor < best_val_mse
            if is_best:
                best_val_mse = cur_monitor
                best_epoch = epoch

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_cross_ddim_mse10": best_val_mse,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "stage": "e8_source_conditioned_transfer",
                        "prediction_mode": "direct",
                        "source_conditioned": True,
                        "direct_latent_bias": policy.direct_latent_bias.detach().cpu(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_cross_ddim_mse10": best_val_mse,
                        "latent_scale_s_z": s_z,
                        "reader": {
                            k: v.detach().cpu() for k, v in policy.reader.state_dict().items()
                        },
                        "transfer_transformer": {
                            k: v.detach().cpu()
                            for k, v in policy.transfer_transformer.state_dict().items()
                        },
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    best_ckpt,
                )

            if ema is not None:
                ema.restore(trainable_params)

            if rank == 0:
                if epoch % args.save_last_every == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        {
                            "stage": "e8_source_conditioned_transfer",
                            "prediction_mode": "direct",
                            "source_conditioned": True,
                            "direct_latent_bias": policy.direct_latent_bias.detach().cpu(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_cross_ddim_mse10": best_val_mse,
                            "latent_scale_s_z": s_z,
                            "reader": {
                                k: v.detach().cpu()
                                for k, v in policy.reader.state_dict().items()
                            },
                            "transfer_transformer": {
                                k: v.detach().cpu()
                                for k, v in policy.transfer_transformer.state_dict().items()
                            },
                            "ema": ema.state_dict() if ema is not None else None,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "history": history,
                            "config": config,
                        },
                        last_ckpt,
                    )
                atomic_write_json(
                    {
                        "stage": "e8_source_conditioned_transfer",
                        "prediction_mode": "direct",
                        "source_conditioned": True,
                        "run_name": run_name,
                        "best_epoch": best_epoch,
                        "best_val_cross_ddim_mse10": best_val_mse,
                        "init_val": init_val,
                        "history": history,
                    },
                    run_dir / "metrics.json",
                )
                dl = val_metrics["diff_loss"]
                lt = val_metrics["latent"]
                dm = val_metrics["ddim"]
                at = val_metrics["reader_attention"]["cross"]
                sa = val_metrics["source_state_alignment"]
                print(
                    f"[E8 Ep {epoch:02d}/{args.epochs}] "
                    f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, "
                    f"Lz={train_metrics['L_z']:.4f}) | "
                    f"val diff nd={dl['cross_no_delta']:.4f} -> c+d={dl['cross_with_delta']:.4f} | "
                    f"Lz nd={lt['cross_no_delta_L_z']:.4f} -> c+d={lt['cross_with_delta_L_z']:.4f} | "
                    f"attn H_n={at['normalized_entropy_mean']:.3f} max_w={at['max_weight_mean']:.3f} "
                    f"eff={at['effective_positions_exp_entropy']:.2f}/{at['valid_positions_mean']:.1f} "
                    f"cos(href,hq)={sa['cross_cos_href_hq']:.3f} | "
                    f"DDIM mse10 nd={dm['cross_no_delta']['action10_mse']:.4f} -> "
                    f"c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"(mae7={dm['cross_with_delta']['action7_mae']:.4f}) | "
                    f"lr={train_metrics['reader_lr']:.1e} t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s){'*' if is_best else ''}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Reload best.pt and run full final offline evaluation + source-state shuffle diagnostic
        reloaded, best_payload = load_e8_policy(
            best_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        final_eval = evaluate_e8_validation(
            reloaded,
            bank,
            batch_size=args.val_batch_size,
            max_val_samples=args.max_val_samples,
            run_ddim=True,
            ddim_max_samples=args.final_ddim_samples,
            run_multi_demo=True,
            multi_demo_max_samples=args.final_multi_demo_samples,
            val_seed=args.val_seed,
        )
        shuffle_diag = evaluate_source_state_shuffle(
            reloaded,
            bank,
            batch_size=args.val_batch_size,
            max_val_samples=args.final_ddim_samples or args.max_val_samples,
            val_seed=args.val_seed,
        )
        if rank == 0:
            final_report = {
                "stage": "e8_source_conditioned_transfer",
                "prediction_mode": "direct",
                "source_conditioned": True,
                "direct_bias_mode": config["direct_bias_mode"],
                "direct_out_proj_init_std": config["direct_out_proj_init_std"],
                "direct_latent_bias_rms": config["direct_latent_bias_rms"],
                "run_name": run_name,
                "seed": args.seed,
                "val_seed": args.val_seed,
                "effective_global_steps": global_step,
                "best_epoch": best_payload["epoch"],
                "best_global_step": best_payload.get("global_step"),
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_cross_ddim_mse10": best_payload["best_val_cross_ddim_mse10"],
                "lambda_diff": config["lambda_a"],
                "lambda_z": config["lambda_z"],
                "parameter_summary": config["parameter_summary"],
                "init_val_summary": (
                    {
                        "initial_output_z_q_rms": init_val["latent"]["adapted_z_rms"],
                        "initial_cross_z_c_rms": init_val["latent"]["cross_z_rms"],
                        "initial_self_z_rms": init_val["latent"]["self_z_rms"],
                        "initial_diff_loss": init_val["diff_loss"]["cross_with_delta"],
                        "initial_L_z": init_val["latent"]["cross_with_delta_L_z"],
                        "initial_ddim_action10_mse": init_val["ddim"]["cross_with_delta"]["action10_mse"],
                        "initial_ddim_action7_mae": init_val["ddim"]["cross_with_delta"]["action7_mae"],
                    }
                    if init_val is not None
                    else None
                ),
                "final_eval": final_eval,
                "source_state_shuffle_diag": shuffle_diag,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            atomic_write_json(shuffle_diag, run_dir / "source_shuffle_diag.json")
            print(
                f"\n[E8 FINAL BEST EVAL (ep={best_payload['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
            print(
                f"\n[E8 SOURCE-STATE SHUFFLE DIAG (ep={best_payload['epoch']})] "
                + json.dumps(shuffle_diag["degradation_vs_orig"], indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


def run_eval_only(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    bank = E7TransferBank(
        clip_cache_dir=args.clip_cache_dir,
        z_self_cache_dir=args.z_self_cache_dir,
        behavior_mem_dir=args.behavior_mem_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        z_self_on_device=(device.type == "cuda" and not args.z_self_on_cpu),
        self_ratio=args.self_ratio,
        seed=args.seed,
    )
    policy, payload = load_e8_policy(
        args.eval_only_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
    )
    res = evaluate_e8_validation(
        policy,
        bank,
        batch_size=args.val_batch_size,
        max_val_samples=args.max_val_samples,
        run_ddim=True,
        ddim_max_samples=args.final_ddim_samples,
        run_multi_demo=True,
        multi_demo_max_samples=args.final_multi_demo_samples,
        val_seed=args.val_seed,
    )
    shuffle_diag = evaluate_source_state_shuffle(
        policy,
        bank,
        batch_size=args.val_batch_size,
        max_val_samples=args.final_ddim_samples or args.max_val_samples,
        val_seed=args.val_seed,
    )
    report = {
        "checkpoint": str(Path(args.eval_only_ckpt).resolve()),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "val_seed": args.val_seed,
        "eval": res,
        "source_state_shuffle_diag": shuffle_diag,
    }
    if args.eval_out is not None:
        args.eval_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report, args.eval_out)
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--clip-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--z-self-cache-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument("--behavior-mem-dir", type=Path, default=DEFAULT_BEHAVIOR_MEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "train")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--reader-lr", type=float, default=1.0e-4)
    parser.add_argument("--transformer-lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-a", "--lambda-diff", dest="lambda_a", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=0.5)
    parser.add_argument("--latent-scale-s-z", type=float, default=FIXED_LATENT_SCALE_S_Z)
    parser.add_argument("--self-ratio", type=float, default=0.025)
    parser.add_argument("--reader-attn-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--direct-bias-mode",
        choices=("train_z_self_mean", "zero"),
        default="train_z_self_mean",
    )
    parser.add_argument("--direct-out-proj-init-std", type=float, default=0.002)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--ddim-val-samples", type=int, default=None)
    parser.add_argument("--multi-demo-samples", type=int, default=1024)
    parser.add_argument("--final-ddim-samples", type=int, default=None)
    parser.add_argument("--final-multi-demo-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--z-self-on-cpu", action="store_true")
    parser.add_argument("--save-last-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--eval-only-ckpt", type=Path, default=None)
    parser.add_argument("--eval-out", type=Path, default=None)
    parser.add_argument("--no-check-grads", dest="check_grads", action="store_false")
    parser.set_defaults(check_grads=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only_ckpt is not None:
        run_eval_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
