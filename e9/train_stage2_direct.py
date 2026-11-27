"""Training, smoke verification, and comprehensive offline validation for E9 Stage 2:
Local Behavior Space (`e9/outputs/stage1_local/best.pt`) + Direct Transfer (exact E7-Direct).

Specifications (100% matched to E7-Direct):
  - Frozen Stage 1 Local Teacher: `e9/outputs/stage1_local/best.pt`
  - Global Soft Reader (`SimpleBehaviorReader`, Q/K only): trained from scratch
  - Direct TransferTransformer (`ResidualTransformer`, prediction_mode='direct', NO skip):
    trained from scratch with `direct_latent_bias = mean(z_self_train)`,
    `output_scale = s_z = rms(z_self_train)`, `out_proj.weight ~ N(0, 0.002^2)`
  - Loss: `L = L_diff(A_q | z_q) + 0.5 * L_z` using the new behavior space's `s_z`
  - AdamW, lr=1e-4, betas=(0.9, 0.95), wd=0.01, grad_clip=1.0, BF16, EMA=0.999
  - Global batch=256, 8 full-pass / 42,912 steps, Cosine LR, warmup=500, min_lr=1e-5
  - Exact E7-Direct train/val split & pairing schedule (`seed=42`, `val_seed=12345`, 97.5% Cross / 2.5% Self)
  - Best checkpoint selected by Validation Cross DDIM 10D Action MSE (`ddim.cross_with_delta.action10_mse`)
  - Saved independently under `e9/outputs/stage2_direct/train/`.
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
from e7.model import E7Policy, compute_attention_stats
from e7.train import compute_train_z_self_mean, get_cosine_schedule_with_min_ratio

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    DEFAULT_STAGE1_LOCAL_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    DEFAULT_STAGE2_DIRECT_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
)
from .stage1_local_model import load_frozen_e9_stage1_teacher
from .stage2_data import E9Stage2TransferBank
from .stage2_direct_model import build_e9_stage2_direct_policy, load_e9_stage2_direct_policy


@torch.inference_mode()
def evaluate_e9_stage2_validation(
    policy: E7Policy,
    bank: E9Stage2TransferBank,
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
    """Comprehensive distributed offline validation matching `e7/train.py::evaluate_e7_validation`."""
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

    # 55 float64 accumulators for exact DDP reduction
    acc = torch.zeros(55, device=device, dtype=torch.float64)

    c_it = bank.iter_e7_val_batches(
        "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    s_it = bank.iter_e7_val_batches(
        "self", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    w_it = bank.iter_e7_val_batches(
        "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_val
    )

    cur = 0
    for (
        (hm_c, bm_c, mpm_c, q_c, z_self, a_c, _),
        (hm_s, bm_s, mpm_s, q_s, _, _, _),
        (hm_w, bm_w, mpm_w, q_w, _, a_w, _),
    ) in zip(c_it, s_it, w_it):
        B = len(hm_c)
        noise = local_noise[cur : cur + B].to(device, non_blocking=True)
        pertub = local_pertub[cur : cur + B].to(device, non_blocking=True)
        t = local_t[cur : cur + B].to(device, non_blocking=True)
        cur += B

        h_q = E5Policy.format_query_tokens(q_c)
        with amp_ctx:
            # Cross retrieval + Direct Transfer
            z_c, _, w_c = policy.read_cross_latent(hm_c, bm_c, mpm_c, q_c, h_q=h_q)
            z_q, delta_c = policy.adapt_latent(z_c, h_q)

            # Self sparse-memory retrieval + Direct Transfer
            z_s_ret, _, w_s = policy.read_cross_latent(hm_s, bm_s, mpm_s, q_s, h_q=h_q)
            z_s_q, delta_s_ret = policy.adapt_latent(z_s_ret, h_q)
            _, delta_s_teacher = policy.adapt_latent(z_self, h_q)

            # Wrong-task Demo retrieval + Direct Transfer
            z_w, _, w_w = policy.read_cross_latent(hm_w, bm_w, mpm_w, q_w, h_q=h_q)
            z_w_q, delta_w = policy.adapt_latent(z_w, h_q)

            l_self = policy.teacher.diffusion_loss_from_z_self(
                z_self, a_c, noise=noise, pertub_noise=pertub, timesteps=t
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
        dir_cos_i = F.cosine_similarity(
            dc_f.reshape(B, -1), gap_c.reshape(B, -1), dim=-1
        )
        proj_frac_i = (dc_f * gap_c).sum(dim=(1, 2)) / gap_c.square().sum(
            dim=(1, 2)
        ).clamp_min(1e-12)
        mag_ratio_i = d_rms_i / g_rms_i.clamp_min(1e-6)

        ast_c = compute_attention_stats(w_c, mpm_c)
        ast_s = compute_attention_stats(w_s, mpm_s)
        ast_w = compute_attention_stats(w_w, mpm_w)

        acc[0] += float(B)
        acc[1] += l_self.double().sum()
        acc[2] += l_c_nd.double().sum()
        acc[3] += l_c_d.double().sum()
        acc[4] += l_s_nd.double().sum()
        acc[5] += l_s_d.double().sum()
        acc[6] += l_w_nd.double().sum()
        acc[7] += l_w_d.double().sum()

        acc[8] += gap_sq_i.double().sum()
        acc[9] += mse_zq_i.double().sum()
        acc[10] += (zs_ret_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[11] += (zs_q_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[12] += (zw_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[13] += (zwq_f - zs_f).square().mean(dim=(1, 2)).double().sum()

        acc[14] += F.cosine_similarity(zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[15] += F.cosine_similarity(zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[16] += F.cosine_similarity(zs_ret_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[17] += F.cosine_similarity(zs_q_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()

        acc[18] += dc_f.square().mean(dim=(1, 2)).double().sum()
        acc[19] += zc_f.square().mean(dim=(1, 2)).double().sum()
        acc[20] += zs_f.square().mean(dim=(1, 2)).double().sum()
        acc[21] += delta_s_ret.float().square().mean(dim=(1, 2)).double().sum()
        acc[22] += delta_s_teacher.float().square().mean(dim=(1, 2)).double().sum()
        acc[23] += delta_w.float().square().mean(dim=(1, 2)).double().sum()

        acc[24] += dir_cos_i.double().sum()
        acc[25] += proj_frac_i.double().sum()
        acc[26] += mag_ratio_i.double().sum()
        acc[27] += g_rms_i.double().sum()
        acc[28] += d_rms_i.double().sum()
        acc[29] += (g_rms_i * g_rms_i).double().sum()
        acc[30] += (d_rms_i * d_rms_i).double().sum()
        acc[31] += (g_rms_i * d_rms_i).double().sum()
        acc[32] += (mse_zq_i < gap_sq_i).double().sum()

        acc[33] += ast_c["entropy"].double().sum()
        acc[34] += ast_c["uniform_entropy"].double().sum()
        acc[35] += ast_c["norm_entropy"].double().sum()
        acc[36] += ast_c["max_weight"].double().sum()
        acc[37] += ast_c["uniform_max_weight"].double().sum()
        acc[38] += ast_c["eff_pos_exp"].double().sum()
        acc[39] += ast_c["eff_pos_ipr"].double().sum()
        acc[40] += ast_c["eff_frac"].double().sum()
        acc[41] += ast_c["valid_pos"].double().sum()

        acc[42] += ast_s["entropy"].double().sum()
        acc[43] += ast_s["norm_entropy"].double().sum()
        acc[44] += ast_s["max_weight"].double().sum()
        acc[45] += ast_s["eff_pos_exp"].double().sum()
        acc[46] += ast_s["eff_pos_ipr"].double().sum()
        acc[47] += ast_s["valid_pos"].double().sum()

        acc[48] += ast_w["entropy"].double().sum()
        acc[49] += ast_w["norm_entropy"].double().sum()
        acc[50] += ast_w["max_weight"].double().sum()
        acc[51] += ast_w["eff_pos_exp"].double().sum()
        acc[52] += ast_w["eff_pos_ipr"].double().sum()
        acc[53] += ast_w["valid_pos"].double().sum()
        acc[54] += zq_f.square().mean(dim=(1, 2)).double().sum()

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
    diff_c_nd = float(acc[2].item() / total)
    diff_c_d = float(acc[3].item() / total)
    diff_s_nd = float(acc[4].item() / total)
    diff_s_d = float(acc[5].item() / total)
    diff_w_nd = float(acc[6].item() / total)
    diff_w_d = float(acc[7].item() / total)

    mse_c_nd = float(acc[8].item() / total)
    mse_c_d = float(acc[9].item() / total)
    mse_s_nd = float(acc[10].item() / total)
    mse_s_d = float(acc[11].item() / total)
    mse_w_nd = float(acc[12].item() / total)
    mse_w_d = float(acc[13].item() / total)

    cross_delta_rms = math.sqrt(float(acc[18].item() / total))
    cross_z_rms = math.sqrt(float(acc[19].item() / total))
    self_z_rms = math.sqrt(float(acc[20].item() / total))
    zq_rms = math.sqrt(float(acc[54].item() / total))
    self_ret_delta_rms = math.sqrt(float(acc[21].item() / total))
    self_teacher_delta_rms = math.sqrt(float(acc[22].item() / total))
    wrong_delta_rms = math.sqrt(float(acc[23].item() / total))
    gap_rms = math.sqrt(mse_c_nd)

    mag_gap_r = _pearson(
        float(acc[27].item()),
        float(acc[28].item()),
        float(acc[29].item()),
        float(acc[30].item()),
        float(acc[31].item()),
        total,
    )

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "diff_loss": {
            "self_teacher": diff_self,
            "cross_no_delta": diff_c_nd,
            "cross_with_delta": diff_c_d,
            "self_memory_no_delta": diff_s_nd,
            "self_memory_with_delta": diff_s_d,
            "wrong_demo_no_delta": diff_w_nd,
            "wrong_demo_with_delta": diff_w_d,
            "delta_gain_over_cross_no_delta": diff_c_nd - diff_c_d,
            "demo_sensitivity_no_delta": diff_w_nd - diff_c_nd,
            "demo_sensitivity_with_delta": diff_w_d - diff_c_d,
        },
        "latent": {
            "s_z": s_z,
            "cross_no_delta_mse_raw": mse_c_nd,
            "cross_with_delta_mse_raw": mse_c_d,
            "self_memory_no_delta_mse_raw": mse_s_nd,
            "self_memory_with_delta_mse_raw": mse_s_d,
            "wrong_demo_no_delta_mse_raw": mse_w_nd,
            "wrong_demo_with_delta_mse_raw": mse_w_d,
            "cross_no_delta_L_z": mse_c_nd / s_z2,
            "cross_with_delta_L_z": mse_c_d / s_z2,
            "self_memory_no_delta_L_z": mse_s_nd / s_z2,
            "self_memory_with_delta_L_z": mse_s_d / s_z2,
            "wrong_demo_no_delta_L_z": mse_w_nd / s_z2,
            "wrong_demo_with_delta_L_z": mse_w_d / s_z2,
            "delta_latent_mse_reduction_over_reader_pct": 100.0
            * (1.0 - mse_c_d / max(mse_c_nd, 1e-12)),
            "delta_improved_sample_frac": float(acc[32].item() / total),
            "cross_no_delta_cos": float(acc[14].item() / total),
            "cross_with_delta_cos": float(acc[15].item() / total),
            "self_memory_no_delta_cos": float(acc[16].item() / total),
            "self_memory_with_delta_cos": float(acc[17].item() / total),
            "cross_z_rms": cross_z_rms,
            "adapted_z_rms": zq_rms,
            "self_z_rms": self_z_rms,
            "cross_delta_rms": cross_delta_rms,
            "cross_delta_rel_rms_zc": cross_delta_rms / max(cross_z_rms, 1e-12),
            "cross_delta_rel_rms_sz": cross_delta_rms / max(s_z, 1e-12),
            "self_memory_delta_rms": self_ret_delta_rms,
            "self_teacher_delta_rms": self_teacher_delta_rms,
            "wrong_demo_delta_rms": wrong_delta_rms,
            "correction_direction_cosine_mean": float(acc[24].item() / total),
            "correction_projection_fraction_mean": float(acc[25].item() / total),
            "delta_to_gap_ratio_global": cross_delta_rms / max(gap_rms, 1e-12),
            "delta_to_gap_ratio_mean": float(acc[26].item() / total),
            "delta_mag_gap_pearson_r": mag_gap_r,
        },
        "reader_attention": {
            "cross": {
                "valid_positions_mean": float(acc[41].item() / total),
                "entropy_mean": float(acc[33].item() / total),
                "uniform_entropy_mean": float(acc[34].item() / total),
                "normalized_entropy_mean": float(acc[35].item() / total),
                "max_weight_mean": float(acc[36].item() / total),
                "uniform_max_weight_mean": float(acc[37].item() / total),
                "max_weight_over_uniform_ratio": float(acc[36].item())
                / max(float(acc[37].item()), 1e-12),
                "effective_positions_exp_entropy": float(acc[38].item() / total),
                "effective_positions_ipr": float(acc[39].item() / total),
                "effective_fraction_mean": float(acc[40].item() / total),
            },
            "self": {
                "valid_positions_mean": float(acc[47].item() / total),
                "entropy_mean": float(acc[42].item() / total),
                "normalized_entropy_mean": float(acc[43].item() / total),
                "max_weight_mean": float(acc[44].item() / total),
                "effective_positions_exp_entropy": float(acc[45].item() / total),
                "effective_positions_ipr": float(acc[46].item() / total),
            },
            "wrong_demo": {
                "valid_positions_mean": float(acc[53].item() / total),
                "entropy_mean": float(acc[48].item() / total),
                "normalized_entropy_mean": float(acc[49].item() / total),
                "max_weight_mean": float(acc[50].item() / total),
                "effective_positions_exp_entropy": float(acc[51].item() / total),
                "effective_positions_ipr": float(acc[52].item() / total),
            },
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

        cur = 0
        for (
            (hm_c, bm_c, mpm_c, q_c, z_self, a_c, _),
            (hm_s, bm_s, mpm_s, q_s, _, _, _),
            (hm_w, bm_w, mpm_w, q_w, _, a_w, _),
        ) in zip(c_it2, s_it2, w_it2):
            B = len(hm_c)
            z0 = local_z0[cur : cur + B].to(device, non_blocking=True)
            cur += B
            h_q = E5Policy.format_query_tokens(q_c)
            with amp_ctx:
                z_c, _, _ = policy.read_cross_latent(hm_c, bm_c, mpm_c, q_c, h_q=h_q)
                z_q, _ = policy.adapt_latent(z_c, h_q)
                z_s_ret, _, _ = policy.read_cross_latent(hm_s, bm_s, mpm_s, q_s, h_q=h_q)
                z_s_q, _ = policy.adapt_latent(z_s_ret, h_q)
                z_w, _, _ = policy.read_cross_latent(hm_w, bm_w, mpm_w, q_w, h_q=h_q)
                z_w_q, _ = policy.adapt_latent(z_w, h_q)

                pred_self = policy.teacher.sample_from_z_self(z_self, init_noise=z0)
                pred_c_nd = policy.teacher.sample_from_z_self(z_c, init_noise=z0)
                pred_c_d = policy.teacher.sample_from_z_self(z_q, init_noise=z0)
                pred_s_nd = policy.teacher.sample_from_z_self(z_s_ret, init_noise=z0)
                pred_s_d = policy.teacher.sample_from_z_self(z_s_q, init_noise=z0)
                pred_w_nd = policy.teacher.sample_from_z_self(z_w, init_noise=z0)
                pred_w_d = policy.teacher.sample_from_z_self(z_w_q, init_noise=z0)

            err["self_teacher"] += _action_error_sums(pred_self, a_c)
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
        nd_mae7 = ddim_out["cross_no_delta"]["action7_mae"]
        cd_mae7 = ddim_out["cross_with_delta"]["action7_mae"]
        ddim_out["delta_gain_action10_mse"] = nd_mse10 - cd_mse10
        ddim_out["delta_gain_action7_mae"] = nd_mae7 - cd_mae7
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
            if args.output_dir == DEFAULT_STAGE2_DIRECT_DIR / "train":
                args.output_dir = DEFAULT_STAGE2_DIRECT_DIR / "smoke"
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

        run_dir = Path(args.output_dir)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        seed_everything(args.seed)
        bank = E9Stage2TransferBank(
            clip_cache_dir=args.clip_cache_dir,
            stage2_cache_dir=args.stage2_cache_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            z_self_on_device=(device.type == "cuda" and not args.z_self_on_cpu),
            self_ratio=args.self_ratio,
            seed=args.seed,
        )

        # Recomputed statistics strictly from E9 Stage 1 Local training z_self
        s_z = float(bank.latent_scale_s_z)
        direct_latent_bias = compute_train_z_self_mean(bank)

        # Load and freeze E9 Stage 1 Local Teacher ONLY; Reader & Transfer strictly from scratch!
        teacher = load_frozen_e9_stage1_teacher(args.stage1_ckpt, device=device)
        seed_everything(args.seed)
        policy = build_e9_stage2_direct_policy(
            teacher,
            latent_scale_s_z=s_z,
            direct_latent_bias=direct_latent_bias,
            reader_attn_dim=args.reader_attn_dim,
            residual_layers=args.residual_layers,
            residual_dropout=args.residual_dropout,
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
                policy.residual_transformer.out_proj.weight.grad.float().norm().item()
            )
            q_gn = float(policy.reader.q_proj.weight.grad.float().norm().item())
            k_gn = float(policy.reader.k_proj.weight.grad.float().norm().item())
            teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
            policy.zero_grad(set_to_none=True)

            print(
                f"[E9 Stage2 Grad Check (pure L_diff via frozen UNet)] "
                f"res.out_proj_gn={out_proj_gn:.6f} | reader.q_gn={q_gn:.6e} | "
                f"reader.k_gn={k_gn:.6e} | teacher_any_grad={teacher_has_grad} | "
                f"diff_loss={float(out_g['diff_loss'].item()):.4f}",
                flush=True,
            )
            if (
                out_proj_gn <= 0.0
                or not math.isfinite(out_proj_gn)
                or q_gn <= 0.0
                or not math.isfinite(q_gn)
                or k_gn <= 0.0
                or not math.isfinite(k_gn)
                or teacher_has_grad
            ):
                raise RuntimeError("E9 Stage 2 gradient / freeze audit FAILED!")
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
            "stage": "e9_stage2_local_direct_transfer",
            "prediction_mode": "direct",
            "direct_bias_mode": "train_z_self_mean",
            "direct_out_proj_init_std": float(args.direct_out_proj_init_std),
            "direct_latent_bias_rms": float(
                policy.direct_latent_bias.float().square().mean().sqrt().item()
            ),
            "smoke": bool(args.smoke),
            "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
            "clip_cache_dir": str(Path(args.clip_cache_dir).resolve()),
            "stage2_cache_dir": str(Path(args.stage2_cache_dir).resolve()),
            "behavior_stride": int(bank.behavior_mem_meta["behavior_stride"]),
            "latent_scale_s_z": s_z,
            "lambda_a": args.lambda_a,
            "lambda_diff": args.lambda_a,
            "lambda_z": args.lambda_z,
            "self_ratio": args.self_ratio,
            "cross_ratio": round(1.0 - args.self_ratio, 6),
            "reader_attn_dim": args.reader_attn_dim,
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
            if ck.get("stage") != "e9_stage2_local_direct_transfer":
                raise RuntimeError(f"Refusing to resume unexpected stage: {ck.get('stage')}")
            if ck.get("direct_latent_bias") is not None:
                policy.set_direct_latent_bias(ck["direct_latent_bias"])
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
                    f"[E9 Stage2] Resumed from epoch {ck['epoch']} "
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
                f"[E9 Stage2 Direct] out={run_dir} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={args.warmup_steps}) | "
                f"lr={args.reader_lr:.2e}->{args.reader_lr * args.min_lr_ratio:.2e} | "
                f"L=L_diff+{args.lambda_z}*L_z | s_z={s_z:.6f} bias_rms={config['direct_latent_bias_rms']:.6f} | "
                f"trainable={ps['total_trainable']:,} (reader={ps['simple_reader_trainable']:,}, "
                f"transfer={ps['residual_transformer_trainable']:,}, teacher_frozen={ps['stage1_teacher_frozen']:,})",
                flush=True,
            )

        init_val = None
        if start_epoch == 1:
            init_val = evaluate_e9_stage2_validation(
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
                    f"[E9 Stage2 Init Val] zq_rms={lt['adapted_z_rms']:.4f} zc_rms={lt['cross_z_rms']:.4f} "
                    f"zs_rms={lt['self_z_rms']:.4f} | diff self={dl['self_teacher']:.4f} "
                    f"c_nd={dl['cross_no_delta']:.4f} c+d={dl['cross_with_delta']:.4f} | "
                    f"Lz={lt['cross_with_delta_L_z']:.4f} | attn H_n={at['normalized_entropy_mean']:.4f} "
                    f"max_w={at['max_weight_mean']:.4f} | DDIM mse10 self={dm['self_teacher']['action10_mse']:.4f} "
                    f"c_nd={dm['cross_no_delta']['action10_mse']:.4f} c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"(mae7={dm['cross_with_delta']['action7_mae']:.4f})",
                    flush=True,
                )

        if start_epoch == 1:
            seed_everything(args.seed + 1000 + rank)

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            policy.teacher.eval()

            tr_acc = torch.zeros(19, device=device, dtype=torch.float64)

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
                "attn_entropy": float(tr_acc[14].item() / n_tot),
                "attn_norm_entropy": float(tr_acc[15].item() / n_tot),
                "attn_max_weight": float(tr_acc[16].item() / n_tot),
                "attn_eff_pos_exp": float(tr_acc[17].item() / n_tot),
                "attn_eff_pos_ipr": float(tr_acc[18].item() / n_tot),
                "grad_norm_mean": float(tr_acc[12].item() / n_steps_tot),
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

            val_metrics = evaluate_e9_stage2_validation(
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
                        "stage": "e9_stage2_local_direct_transfer",
                        "prediction_mode": "direct",
                        "direct_latent_bias": policy.direct_latent_bias.detach().cpu(),
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
                            "stage": "e9_stage2_local_direct_transfer",
                            "prediction_mode": "direct",
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
                        "stage": "e9_stage2_local_direct_transfer",
                        "prediction_mode": "direct",
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
                print(
                    f"[E9 Stage2 Ep {epoch:02d}/{args.epochs}] "
                    f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, "
                    f"Lz={train_metrics['L_z']:.4f}, nd_Lz={train_metrics['nodelta_L_z']:.4f}) | "
                    f"val diff nd={dl['cross_no_delta']:.4f} -> c+d={dl['cross_with_delta']:.4f} | "
                    f"Lz nd={lt['cross_no_delta_L_z']:.4f} -> c+d={lt['cross_with_delta_L_z']:.4f} "
                    f"dir_cos={lt['correction_direction_cosine_mean']:.3f} | "
                    f"attn H_n={at['normalized_entropy_mean']:.3f} max_w={at['max_weight_mean']:.3f} "
                    f"eff={at['effective_positions_exp_entropy']:.2f}/{at['valid_positions_mean']:.1f} | "
                    f"DDIM mse10 nd={dm['cross_no_delta']['action10_mse']:.4f} -> "
                    f"c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"(mae7={dm['cross_with_delta']['action7_mae']:.4f}) | "
                    f"lr={train_metrics['reader_lr']:.1e} t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s){'*' if is_best else ''}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Reload best.pt and run full final offline evaluation across ALL 6,248 val starts
        reloaded, best_payload = load_e9_stage2_direct_policy(
            best_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        final_eval = evaluate_e9_stage2_validation(
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
                "stage": "e9_stage2_local_direct_transfer",
                "prediction_mode": "direct",
                "direct_bias_mode": config["direct_bias_mode"],
                "direct_out_proj_init_std": config["direct_out_proj_init_std"],
                "direct_latent_bias_rms": config["direct_latent_bias_rms"],
                "latent_scale_s_z": s_z,
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
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[E9 Stage2 FINAL BEST EVAL (ep={best_payload['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument("--clip-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--stage2-cache-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_STAGE2_DIRECT_DIR / "train"
    )
    # Exact E7-Direct hyperparameters
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
    parser.add_argument("--self-ratio", type=float, default=0.025)
    parser.add_argument("--reader-attn-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--residual-layers", type=int, default=3)
    parser.add_argument("--residual-dropout", type=float, default=0.1)
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
