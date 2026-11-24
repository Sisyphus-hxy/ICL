"""Training, smoke verification, and offline validation for E7.1:
Adjacent-Pair Behavior Retrieval + Latent Adaptation (CLIP).

Strictly preserves all E7 training hyperparameters, data splits, and epoch pairing schedules,
modifying ONLY the Behavior Reader (`AdjacentPairBehaviorReader` with Straight-Through Softmax
during training and deterministic argmax at inference).

All outputs saved independently under `e7/outputs/adjacent_pair/`.
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
from .adjacent_model import (
    E7AdjacentPolicy,
    compute_pair_selection_stats,
    load_e7_adjacent_policy,
)
from .data import E7TransferBank
from .train import get_cosine_schedule_with_min_ratio

ADJACENT_OUTPUT_ROOT = OUTPUT_ROOT / "adjacent_pair"


def _compute_spearman_approx(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


def _compute_pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xm = x.astype(np.float64) - float(x.mean())
    ym = y.astype(np.float64) - float(y.mean())
    denom = math.sqrt(float((xm * xm).sum() * (ym * ym).sum()))
    return float((xm * ym).sum() / denom) if denom > 1e-12 else 0.0


def _summarize_selection_arrays(
    norm_pos: np.ndarray,
    norm_idx: np.ndarray,
    beta_sel: np.ndarray,
    is_first: np.ndarray,
    is_last: np.ndarray,
    query_time_frac: np.ndarray,
    p_soft_ent: np.ndarray,
    p_soft_unif_ent: np.ndarray,
    p_soft_norm_ent: np.ndarray,
    p_soft_max_w: np.ndarray,
    p_soft_unif_max_w: np.ndarray,
    p_soft_eff_exp: np.ndarray,
    p_soft_eff_ipr: np.ndarray,
    valid_pairs: np.ndarray,
) -> Dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 11)
    pos_hist, _ = np.histogram(norm_pos, bins=bins)
    pos_hist_frac = (pos_hist / max(len(norm_pos), 1)).tolist()
    beta_hist, _ = np.histogram(beta_sel, bins=bins)
    beta_hist_frac = (beta_hist / max(len(beta_sel), 1)).tolist()

    # Execution-time decile profile: retrieved Demo norm_pos vs Query execution progress
    deciles: List[Dict[str, Any]] = []
    for d in range(10):
        lo, hi = d * 0.1, (d + 1) * 0.1
        if d < 9:
            m = (query_time_frac >= lo) & (query_time_frac < hi)
        else:
            m = (query_time_frac >= lo) & (query_time_frac <= 1.0 + 1e-9)
        cnt = int(m.sum())
        deciles.append(
            {
                "query_time_bin": f"{lo:.1f}-{hi:.1f}",
                "count": cnt,
                "mean_query_time_frac": float(query_time_frac[m].mean()) if cnt > 0 else None,
                "mean_selected_pos": float(norm_pos[m].mean()) if cnt > 0 else None,
                "std_selected_pos": float(norm_pos[m].std()) if cnt > 0 else None,
                "mean_beta": float(beta_sel[m].mean()) if cnt > 0 else None,
            }
        )

    return {
        "count": int(len(norm_pos)),
        "valid_pairs_mean": float(valid_pairs.mean()),
        "p_soft_entropy_mean": float(p_soft_ent.mean()),
        "p_soft_uniform_entropy_mean": float(p_soft_unif_ent.mean()),
        "p_soft_normalized_entropy_mean": float(p_soft_norm_ent.mean()),
        "p_soft_max_weight_mean": float(p_soft_max_w.mean()),
        "p_soft_uniform_max_weight_mean": float(p_soft_unif_max_w.mean()),
        "p_soft_max_weight_over_uniform_ratio": float(p_soft_max_w.mean())
        / max(float(p_soft_unif_max_w.mean()), 1e-12),
        "p_soft_effective_pairs_exp_entropy": float(p_soft_eff_exp.mean()),
        "p_soft_effective_pairs_ipr": float(p_soft_eff_ipr.mean()),
        "pair_selection_distribution": {
            "norm_pos_mean": float(norm_pos.mean()),
            "norm_pos_std": float(norm_pos.std()),
            "norm_idx_mean": float(norm_idx.mean()),
            "first_pair_frac": float(is_first.mean()),
            "last_pair_frac": float(is_last.mean()),
            "hist_10bins_edges": [round(float(x), 2) for x in bins],
            "hist_10bins_frac": [round(float(x), 5) for x in pos_hist_frac],
        },
        "beta_distribution": {
            "mean": float(beta_sel.mean()),
            "std": float(beta_sel.std()),
            "p05": float(np.quantile(beta_sel, 0.05)),
            "p25": float(np.quantile(beta_sel, 0.25)),
            "p50": float(np.quantile(beta_sel, 0.50)),
            "p75": float(np.quantile(beta_sel, 0.75)),
            "p95": float(np.quantile(beta_sel, 0.95)),
            "near_zero_frac_lt_005": float((beta_sel < 0.05).mean()),
            "near_one_frac_gt_095": float((beta_sel > 0.95).mean()),
            "interior_frac_01_to_09": float(((beta_sel >= 0.1) & (beta_sel <= 0.9)).mean()),
            "hist_10bins_frac": [round(float(x), 5) for x in beta_hist_frac],
        },
        "position_vs_execution_time": {
            "pearson_r": _compute_pearson_np(query_time_frac, norm_pos),
            "spearman_rho": _compute_spearman_approx(query_time_frac, norm_pos),
            "mean_abs_time_align_err": float(np.abs(norm_pos - query_time_frac).mean()),
            "decile_profile": deciles,
        },
    }


def _gather_1d_np(arr: np.ndarray, device: torch.device) -> np.ndarray:
    """Gather variable-length 1D float32 arrays across DDP ranks onto all ranks in order."""
    world = _world()
    if world == 1 or not dist.is_initialized():
        return arr
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)
    n_local = torch.tensor([len(t)], device=device, dtype=torch.long)
    n_list = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world)]
    dist.all_gather(n_list, n_local)
    max_n = int(max(x.item() for x in n_list))
    padded = torch.zeros(max_n, device=device, dtype=torch.float32)
    padded[: len(t)] = t
    gathered = [torch.zeros(max_n, device=device, dtype=torch.float32) for _ in range(world)]
    dist.all_gather(gathered, padded)
    parts = [g[: int(n_list[r].item())].cpu().numpy() for r, g in enumerate(gathered)]
    return np.concatenate(parts, axis=0)


@torch.inference_mode()
def evaluate_e7_adjacent_validation(
    policy: E7AdjacentPolicy,
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
    """Comprehensive distributed offline validation for E7.1 (identical protocol to E7 plus
    pair selection distribution, beta distribution, and selected position vs execution time)."""
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

    pool_all = np.flatnonzero(bank.val_valid_cross_mask)
    n_pool = len(pool_all)
    n_val = n_pool if max_val_samples is None else min(int(max_val_samples), n_pool)
    sel_local = pool_all[:n_val][rank::world]

    # Exact query execution time fraction t_q / (T_q - 1) for every local val sample
    q_eps_local = bank.val_episodes[sel_local]
    q_frames_local = bank.val_frames[sel_local].astype(np.float64)
    q_ep_lens_local = np.maximum(bank.ep_lengths_np[q_eps_local].astype(np.float64) - 1.0, 1.0)
    q_time_frac_local = np.clip(q_frames_local / q_ep_lens_local, 0.0, 1.0).astype(np.float32)

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

    acc = torch.zeros(37, device=device, dtype=torch.float64)

    # Collect per-sample pair selection & beta arrays for exact histograms, quantiles, and correlations
    sel_cols = (
        "norm_pair_pos",
        "norm_pair_idx",
        "beta_selected",
        "is_first_pair",
        "is_last_pair",
        "p_soft_entropy",
        "p_soft_uniform_entropy",
        "p_soft_norm_entropy",
        "p_soft_max_weight",
        "p_soft_uniform_max_weight",
        "p_soft_eff_pairs_exp",
        "p_soft_eff_pairs_ipr",
        "valid_pairs",
    )
    per_mode_buf: Dict[str, Dict[str, List[np.ndarray]]] = {
        m: {k: [] for k in sel_cols} for m in ("cross", "self", "wrong_demo")
    }

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
            z_c, _, info_c = policy.read_cross_latent(hm_c, bm_c, mpm_c, q_c, h_q=h_q)
            z_q, delta_c = policy.adapt_latent(z_c, h_q)

            z_s_ret, _, info_s = policy.read_cross_latent(hm_s, bm_s, mpm_s, q_s, h_q=h_q)
            z_s_q, delta_s_ret = policy.adapt_latent(z_s_ret, h_q)
            _, delta_s_teacher = policy.adapt_latent(z_self, h_q)

            z_w, _, info_w = policy.read_cross_latent(hm_w, bm_w, mpm_w, q_w, h_q=h_q)
            z_w_q, delta_w = policy.adapt_latent(z_w, h_q)

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

        gap_c = zs_f - zc_f
        gap_sq_i = gap_c.square().mean(dim=(1, 2))
        mse_zq_i = (zq_f - zs_f).square().mean(dim=(1, 2))
        g_rms_i = gap_sq_i.clamp_min(0.0).sqrt()
        d_rms_i = dc_f.square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()
        dir_cos_i = F.cosine_similarity(dc_f.reshape(B, -1), gap_c.reshape(B, -1), dim=-1)
        proj_frac_i = (dc_f * gap_c).sum(dim=(1, 2)) / gap_c.square().sum(dim=(1, 2)).clamp_min(1e-12)
        mag_ratio_i = d_rms_i / g_rms_i.clamp_min(1e-6)

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
        acc[36] += zq_f.square().mean(dim=(1, 2)).double().sum()

        for m_name, inf in (("cross", info_c), ("self", info_s), ("wrong_demo", info_w)):
            pst = compute_pair_selection_stats(
                inf["pair_idx"],
                inf["beta_selected"],
                inf["p_soft"],
                inf["pair_valid_mask"],
                inf["has_pair"],
            )
            for k in sel_cols:
                per_mode_buf[m_name][k].append(pst[k].float().cpu().numpy())

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)

    q_time_all = _gather_1d_np(q_time_frac_local, device)
    reader_pair_diag: Dict[str, Any] = {}
    for m_name in ("cross", "self", "wrong_demo"):
        gathered = {
            k: _gather_1d_np(np.concatenate(per_mode_buf[m_name][k], axis=0), device)
            for k in sel_cols
        }
        reader_pair_diag[m_name] = _summarize_selection_arrays(
            norm_pos=gathered["norm_pair_pos"],
            norm_idx=gathered["norm_pair_idx"],
            beta_sel=gathered["beta_selected"],
            is_first=gathered["is_first_pair"],
            is_last=gathered["is_last_pair"],
            query_time_frac=q_time_all,
            p_soft_ent=gathered["p_soft_entropy"],
            p_soft_unif_ent=gathered["p_soft_uniform_entropy"],
            p_soft_norm_ent=gathered["p_soft_norm_entropy"],
            p_soft_max_w=gathered["p_soft_max_weight"],
            p_soft_unif_max_w=gathered["p_soft_uniform_max_weight"],
            p_soft_eff_exp=gathered["p_soft_eff_pairs_exp"],
            p_soft_eff_ipr=gathered["p_soft_eff_pairs_ipr"],
            valid_pairs=gathered["valid_pairs"],
        )

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
    zq_rms = math.sqrt(float(acc[36].item() / total))
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
            "reader_gain_over_stage1": diff_s1_c - diff_c_nd,
            "delta_gain_over_cross_no_delta": diff_c_nd - diff_c_d,
            "total_gain_over_stage1": diff_s1_c - diff_c_d,
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
        "reader_pair_diagnostics": reader_pair_diag,
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
                z_c, _, _ = policy.read_cross_latent(hm_c, bm_c, mpm_c, q_c, h_q=h_q)
                z_q, _ = policy.adapt_latent(z_c, h_q)
                z_s_ret, _, _ = policy.read_cross_latent(hm_s, bm_s, mpm_s, q_s, h_q=h_q)
                z_s_q, _ = policy.adapt_latent(z_s_ret, h_q)
                z_w, _, _ = policy.read_cross_latent(hm_w, bm_w, mpm_w, q_w, h_q=h_q)
                z_w_q, _ = policy.adapt_latent(z_w, h_q)

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
        nd_mse10 = ddim_out["cross_no_delta"]["action10_mse"]
        cd_mse10 = ddim_out["cross_with_delta"]["action10_mse"]
        cd_mae7 = ddim_out["cross_with_delta"]["action7_mae"]
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
                    zc1, hq, _ = policy.read_cross_latent(hm1, bm1, pm1, q1)
                    zc2, _, _ = policy.read_cross_latent(hm2, bm2, pm2, q2, h_q=hq)
                    za1, _ = policy.adapt_latent(zc1, hq)
                    za2, _ = policy.adapt_latent(zc2, hq)
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


def _run_startup_audits(
    policy: E7AdjacentPolicy,
    bank: E7TransferBank,
    device: torch.device,
    per_rank_batch: int,
) -> Dict[str, Any]:
    """Verify all E7.1 invariants on real batch data before training:
      1. Training hard forward strictly uses ONLY the 2 adjacent latents `(b_{i*}, b_{i*+1})`.
      2. Training hard forward output `z_c` matches Inference deterministic output `z_c` identically.
      3. Pure `L_diff` through frozen UNet yields finite non-zero gradients on EVERY Reader parameter
         (Q/K, pair_k_proj, beta_q_proj, beta_k_proj, beta_bias) and `ResidualTransformer.out_proj`,
         while frozen Stage 1 Teacher receives ZERO gradients.
    """
    hm_g, bm_g, mpm_g, q_g, zs_g, a_g, is_s_g = next(
        bank.iter_e7_train_batches(1, min(8, per_rank_batch), max_samples=16)
    )
    h_q = E5Policy.format_query_tokens(q_g)

    # 1. Hard-forward vs Eval-forward exact consistency + strict 2-adjacent-latent invariance
    with torch.no_grad():
        policy.train(True)
        z_train, _, info_tr = policy.read_cross_latent(hm_g, bm_g, mpm_g, q_g, h_q=h_q)
        policy.eval()
        z_eval, _, info_ev = policy.read_cross_latent(hm_g, bm_g, mpm_g, q_g, h_q=h_q)

        train_eval_max_diff = float((z_train - z_eval).abs().max().item())
        idx_exact_match = bool(torch.equal(info_tr["pair_idx"], info_ev["pair_idx"]))

        # Corrupt ALL non-selected memory entries in b_mem with 1e5 -> z_train must not change at all!
        B, M = bm_g.shape[:2]
        bm_corrupt = bm_g.clone()
        pidx = info_tr["pair_idx"]
        keep_mask = torch.zeros((B, M), device=device, dtype=torch.bool)
        ar = torch.arange(B, device=device)
        keep_mask[ar, pidx] = True
        keep_mask[ar, pidx + 1] = True
        bm_corrupt[~keep_mask] = 1e5

        policy.train(True)
        z_corrupt, _, _ = policy.read_cross_latent(hm_g, bm_corrupt, mpm_g, q_g, h_q=h_q)
        non_adj_leak_max_diff = float((z_train - z_corrupt).abs().max().item())

    # 2. Pure L_diff STE gradient audit through frozen Stage 1 UNet
    policy.train(True)
    policy.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        out_g = policy(
            hm_g, bm_g, mpm_g, q_g, zs_g, a_g, is_s_g, lambda_a=1.0, lambda_z=0.0
        )
    out_g["loss"].backward()

    reader_gns = {
        name: float(p.grad.float().norm().item()) if p.grad is not None else 0.0
        for name, p in policy.reader.named_parameters()
    }
    res_out_gn = float(policy.residual_transformer.out_proj.weight.grad.float().norm().item())
    teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
    policy.zero_grad(set_to_none=True)

    audit = {
        "train_eval_max_abs_diff": train_eval_max_diff,
        "train_eval_pair_idx_exact_match": idx_exact_match,
        "non_adjacent_corruption_max_abs_diff": non_adj_leak_max_diff,
        "res_out_proj_grad_norm": res_out_gn,
        "reader_param_grad_norms": reader_gns,
        "teacher_any_grad": teacher_has_grad,
        "diff_loss": float(out_g["diff_loss"].item()),
    }
    print(
        f"[E7.1 Audit] train_vs_eval_diff={train_eval_max_diff:.2e} (idx_match={idx_exact_match}) | "
        f"non_adj_leak_diff={non_adj_leak_max_diff:.2e} | res.out_proj_gn={res_out_gn:.4f} | "
        f"reader_gn min={min(reader_gns.values()):.3e} max={max(reader_gns.values()):.3e} "
        f"(pair_k={reader_gns['pair_k_proj.weight']:.3e}, beta_k={reader_gns['beta_k_proj.weight']:.3e}, "
        f"beta_bias={reader_gns['beta_bias']:.3e}) | teacher_any_grad={teacher_has_grad}",
        flush=True,
    )
    if (
        not idx_exact_match
        or train_eval_max_diff > 1e-5
        or non_adj_leak_max_diff > 1e-5
        or res_out_gn <= 0.0
        or not math.isfinite(res_out_gn)
        or any((g <= 0.0 or not math.isfinite(g)) for g in reader_gns.values())
        or teacher_has_grad
    ):
        raise RuntimeError(f"E7.1 startup audit FAILED: {audit}")
    return audit


def _module_grad_norm(params: List[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach().float()
            sq += float(g.square().sum().item())
    return math.sqrt(sq)


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

        run_name = args.run_name if args.run_name else "e7_1_adjacent_pair_retrieval"
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

        # Load and freeze Stage 1 Teacher ONLY; freshly initialize Reader & ResidualTransformer
        teacher = load_frozen_stage1_teacher(args.stage1_ckpt, device=device)
        seed_everything(args.seed)
        policy = E7AdjacentPolicy(
            teacher,
            latent_scale_s_z=s_z,
            reader_attn_dim=args.reader_attn_dim,
            reader_temperature=args.temperature,
            residual_layers=args.residual_layers,
            residual_dropout=args.residual_dropout,
        ).to(device)

        startup_audit = None
        if rank == 0 and args.check_grads:
            startup_audit = _run_startup_audits(policy, bank, device, per_rank_batch)
        if use_ddp:
            dist.barrier()

        optim_groups = policy.get_optim_groups(
            reader_lr=args.reader_lr,
            residual_lr=args.residual_lr,
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
        reader_all_params = list(policy.reader.parameters())
        reader_pair_params = list(policy.reader.pair_k_proj.parameters())
        reader_beta_params = (
            list(policy.reader.beta_q_proj.parameters())
            + list(policy.reader.beta_k_proj.parameters())
            + [policy.reader.beta_bias]
        )
        reader_qk_params = (
            list(policy.reader.q_proj.parameters())
            + list(policy.reader.k_proj.parameters())
        )
        res_all_params = list(policy.residual_transformer.parameters())

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
            "stage": "e7_1_adjacent_pair_retrieval",
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
            "reader_type": "adjacent_pair_ste",
            "reader_attn_dim": args.reader_attn_dim,
            "reader_temperature": args.temperature,
            "residual_layers": args.residual_layers,
            "residual_heads": 8,
            "residual_ffn_dim": 3072,
            "residual_dropout": args.residual_dropout,
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
            "residual_lr": args.residual_lr,
            "residual_min_lr": args.residual_lr * args.min_lr_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_steps": args.warmup_steps,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "clip_grad_norm": args.clip_grad_norm,
            "mixed_precision": "bf16",
            "use_ema": bool(args.use_ema),
            "ema_decay": args.ema_decay if args.use_ema else None,
            "best_selection_metric": "ddim.cross_with_delta.action10_mse",
            "startup_audit": startup_audit,
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
            if ck.get("stage") != "e7_1_adjacent_pair_retrieval":
                raise RuntimeError(f"Refusing to resume non-E7.1 checkpoint: {ck.get('stage')}")
            policy.reader.load_state_dict(ck["reader"], strict=True)
            policy.residual_transformer.load_state_dict(ck["residual_transformer"], strict=True)
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
                    f"[E7.1] Resumed from epoch {ck['epoch']} "
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
                f"[E7.1] run={run_name} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={args.warmup_steps}) | "
                f"reader_lr={args.reader_lr:.2e}->{args.reader_lr * args.min_lr_ratio:.2e} (T={args.temperature}) | "
                f"res_lr={args.residual_lr:.2e}->{args.residual_lr * args.min_lr_ratio:.2e} | "
                f"weights=(diff:{args.lambda_a}, z:{args.lambda_z}) s_z={s_z:.6f} | "
                f"trainable={ps['total_trainable']:,} "
                f"(adjacent_reader={ps['adjacent_reader_trainable']:,}, "
                f"res={ps['residual_transformer_trainable']:,}, "
                f"teacher_frozen={ps['stage1_teacher_frozen']:,})",
                flush=True,
            )

        init_val = None
        if start_epoch == 1:
            init_val = evaluate_e7_adjacent_validation(
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
                pd = init_val["reader_pair_diagnostics"]["cross"]
                print(
                    f"[E7.1 Init Val] diff c_nd={dl['cross_no_delta']:.4f} c+d={dl['cross_with_delta']:.4f} | "
                    f"Lz={lt['cross_with_delta_L_z']:.4f} | "
                    f"p_soft H_n={pd['p_soft_normalized_entropy_mean']:.3f} max_w={pd['p_soft_max_weight_mean']:.3f} "
                    f"beta={pd['beta_distribution']['mean']:.3f}±{pd['beta_distribution']['std']:.3f} "
                    f"pos_time_r={pd['position_vs_execution_time']['pearson_r']:.3f} | "
                    f"DDIM mse10 c_nd={dm['cross_no_delta']['action10_mse']:.4f} "
                    f"c+d={dm['cross_with_delta']['action10_mse']:.4f}",
                    flush=True,
                )

        if start_epoch == 1:
            seed_everything(args.seed + 1000 + rank)

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            policy.teacher.eval()

            tr_acc = torch.zeros(26, device=device, dtype=torch.float64)

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

                # Record pre-clip module-level gradient norms (monitoring STE gradient health)
                r_gn = _module_grad_norm(reader_all_params)
                r_pair_gn = _module_grad_norm(reader_pair_params)
                r_beta_gn = _module_grad_norm(reader_beta_params)
                r_qk_gn = _module_grad_norm(reader_qk_params)
                res_gn = _module_grad_norm(res_all_params)

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
                tr_acc[14] += float(out["p_soft_entropy"].item()) * B
                tr_acc[15] += float(out["p_soft_norm_entropy"].item()) * B
                tr_acc[16] += float(out["p_soft_max_weight"].item()) * B
                tr_acc[17] += float(out["p_soft_eff_pairs"].item()) * B
                tr_acc[18] += float(out["beta_mean"].item()) * B
                tr_acc[19] += float(out["beta_std"].item()) * B
                tr_acc[20] += float(out["norm_pair_pos_mean"].item()) * B
                tr_acc[21] += r_gn
                tr_acc[22] += r_pair_gn
                tr_acc[23] += r_beta_gn
                tr_acc[24] += r_qk_gn
                tr_acc[25] += res_gn

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
                "cross_delta_rms": math.sqrt(float(tr_acc[8].item() / n_cross_tot)),
                "self_delta_rms": math.sqrt(float(tr_acc[10].item() / n_self_tot)),
                "p_soft_entropy": float(tr_acc[14].item() / n_tot),
                "p_soft_norm_entropy": float(tr_acc[15].item() / n_tot),
                "p_soft_max_weight": float(tr_acc[16].item() / n_tot),
                "p_soft_eff_pairs": float(tr_acc[17].item() / n_tot),
                "beta_mean": float(tr_acc[18].item() / n_tot),
                "beta_std": float(tr_acc[19].item() / n_tot),
                "norm_pair_pos_mean": float(tr_acc[20].item() / n_tot),
                "grad_norm_mean": float(tr_acc[12].item() / n_steps_tot),
                "reader_grad_norms": {
                    "reader_total_mean": float(tr_acc[21].item() / n_steps_tot),
                    "reader_pair_k_mean": float(tr_acc[22].item() / n_steps_tot),
                    "reader_beta_mean": float(tr_acc[23].item() / n_steps_tot),
                    "reader_qk_mean": float(tr_acc[24].item() / n_steps_tot),
                    "residual_total_mean": float(tr_acc[25].item() / n_steps_tot),
                },
                "reader_lr": float(last_lrs[0]),
                "residual_lr": float(last_lrs[2]),
                "ema_decay": float(ema.cur_decay_value or 0.0) if ema is not None else None,
                "train_seconds": round(t_train, 2),
                "steps_per_sec": round((n_steps_tot / world) / max(t_train, 1e-6), 2),
            }

            t_val0 = time.monotonic()
            if ema is not None:
                ema.store(trainable_params)
                ema.copy_to(trainable_params)

            val_metrics = evaluate_e7_adjacent_validation(
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
                        "stage": "e7_1_adjacent_pair_retrieval",
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_cross_ddim_mse10": best_val_mse,
                        "latent_scale_s_z": s_z,
                        "reader": {
                            k: v.detach().cpu() for k, v in policy.reader.state_dict().items()
                        },
                        "residual_transformer": {
                            k: v.detach().cpu()
                            for k, v in policy.residual_transformer.state_dict().items()
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
                            "stage": "e7_1_adjacent_pair_retrieval",
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_cross_ddim_mse10": best_val_mse,
                            "latent_scale_s_z": s_z,
                            "reader": {
                                k: v.detach().cpu()
                                for k, v in policy.reader.state_dict().items()
                            },
                            "residual_transformer": {
                                k: v.detach().cpu()
                                for k, v in policy.residual_transformer.state_dict().items()
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
                        "stage": "e7_1_adjacent_pair_retrieval",
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
                pd = val_metrics["reader_pair_diagnostics"]["cross"]
                rg = train_metrics["reader_grad_norms"]
                print(
                    f"[E7.1 Ep {epoch:02d}/{args.epochs}] "
                    f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, "
                    f"Lz={train_metrics['L_z']:.4f}, r_gn={rg['reader_total_mean']:.3f} "
                    f"[pair={rg['reader_pair_k_mean']:.3f},beta={rg['reader_beta_mean']:.3f}]) | "
                    f"val diff nd={dl['cross_no_delta']:.4f}->c+d={dl['cross_with_delta']:.4f} | "
                    f"Lz nd={lt['cross_no_delta_L_z']:.4f}->c+d={lt['cross_with_delta_L_z']:.4f} "
                    f"dz={lt['cross_delta_rms']:.3f} dir={lt['correction_direction_cosine_mean']:.3f} | "
                    f"pair H_n={pd['p_soft_normalized_entropy_mean']:.3f} "
                    f"max_w={pd['p_soft_max_weight_mean']:.3f} "
                    f"beta={pd['beta_distribution']['mean']:.2f}±{pd['beta_distribution']['std']:.2f} "
                    f"r_t={pd['position_vs_execution_time']['pearson_r']:.3f} | "
                    f"DDIM mse10 nd={dm['cross_no_delta']['action10_mse']:.4f}->"
                    f"c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"(mae7={dm['cross_with_delta']['action7_mae']:.4f}) | "
                    f"t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s){'*' if is_best else ''}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Reload best.pt and run full final offline evaluation
        reloaded, best_payload = load_e7_adjacent_policy(
            best_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        final_eval = evaluate_e7_adjacent_validation(
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
        if rank == 0:
            final_report = {
                "stage": "e7_1_adjacent_pair_retrieval",
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
                "final_eval": final_eval,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[E7.1 FINAL BEST EVAL (ep={best_payload['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--clip-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--z-self-cache-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument("--behavior-mem-dir", type=Path, default=DEFAULT_BEHAVIOR_MEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=ADJACENT_OUTPUT_ROOT / "train")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--reader-lr", type=float, default=1.0e-4)
    parser.add_argument("--residual-lr", type=float, default=1.0e-4)
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--residual-layers", type=int, default=3)
    parser.add_argument("--residual-dropout", type=float, default=0.1)
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
    parser.add_argument("--no-check-grads", dest="check_grads", action="store_false")
    parser.set_defaults(check_grads=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
