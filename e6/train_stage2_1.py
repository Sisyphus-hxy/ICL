"""Training, validation, smoke test, and full offline evaluation for E6 Stage 2.1
(Joint Cross Reader Q/K & Latent Delta Training, CLIP, From Stage 1).

Initializes strictly from Stage 1 (`e6/outputs/stage1/stage1_self_latent/best.pt`):
  - Frozen Self Teacher (`PromptPooler`, Self Reader `R_T`, `ConditionalUnet1D`).
  - Independent Cross Reader `R_S` copied from Stage 1 Reader with ONLY Attention Q/K
    projections unfrozen (`lr=3e-6`).
  - Freshly initialized zero-out `ResidualTransformer` `G_psi` (`lr=1e-4`).
    NEVER loads existing Stage 2 ResidualTransformer weights, Optimizer, or EMA.

Loss:
    L = L_diff(A_q | z_c + delta_z) + 0.5 * L_z + 0.1 * L_anchor
    s_z = 7.822748 (fixed)

Selects Best checkpoint by Validation DDIM Cross 10D Action MSE (`cross_with_delta.action10_mse`).
All artifacts saved strictly under `e6/outputs/stage2_1/`.
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
from torch.optim.lr_scheduler import LambdaLR

from e5.diffusion import EMAModel
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

from . import ACTION_DIM_10D, ACTION_HORIZON, E5_CACHE_DIR, OUTPUT_ROOT
from .stage1_model import load_frozen_stage1_teacher
from .stage2_1_data import DEFAULT_STAGE2_CACHE_DIR, Stage2_1TransferBank
from .stage2_1_model import (
    DEFAULT_STAGE1_CKPT,
    FIXED_LATENT_SCALE_S_Z,
    E6Stage2_1Policy,
    load_stage2_1_policy,
)


def get_cosine_schedule_with_min_ratio(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup for `num_warmup_steps`, then cosine decay down to `min_lr_ratio * base_lr`
    independently for every parameter group (e.g., 3e-6->3e-7 and 1e-4->1e-5)."""
    min_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        denom = max(1, num_training_steps - num_warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - num_warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def get_scheduled_lambda_z(
    step: int,
    *,
    total_steps: int,
    schedule: str = "constant",
    lambda_z: float = 0.5,
    lambda_z_start: float = 0.5,
    lambda_z_end: float = 0.1,
    hold_frac: float = 0.2,
) -> float:
    """Per-step latent supervision weight lambda_z(t).

    - 'constant'   : fixed `lambda_z` for all steps (e.g. 0.5 for Stage 2.1, 0.0 for Action-only).
    - 'hold_linear': keep `lambda_z_start` for the first `hold_frac` of `total_steps`, then
                     linearly decay to `lambda_z_end` over the remaining steps.
    """
    if schedule == "constant":
        return float(lambda_z)
    if schedule != "hold_linear":
        raise ValueError(f"Unsupported lambda_z_schedule: {schedule}")
    total_steps = int(total_steps)
    if total_steps <= 1:
        return float(lambda_z_end)
    hold_steps = int(round(float(hold_frac) * total_steps))
    if step < hold_steps:
        return float(lambda_z_start)
    decay_denom = max(1, total_steps - 1 - hold_steps)
    progress = min(1.0, max(0.0, float(step - hold_steps) / float(decay_denom)))
    return float(lambda_z_start) + (float(lambda_z_end) - float(lambda_z_start)) * progress


@torch.inference_mode()
def evaluate_stage2_1_validation(
    policy: E6Stage2_1Policy,
    bank: Stage2_1TransferBank,
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
    """Comprehensive distributed validation for E6 Stage 2.1:
      1. Frozen Stage 1 Self Teacher (`z_s = R_T(M_Q, h_q)`)
      2. Frozen Stage 1 Reader on Cross Demo (`z_c0 = R_T(M_D, h_q)`) -> baseline before R_S
      3. Adapted Cross Reader Self Anchor (`z_s_stu = R_S(M_Q, h_q)`) -> L_anchor & Self retention
      4. Adapted Cross Reader Cross No-Delta (`z_c = R_S(M_D, h_q)`) -> Reader-only contribution
      5. Joint Cross + Delta (`z_q = z_c + G_psi(z_c, h_q)`) -> Full Stage 2.1
      6. Wrong-task Demo (No-Delta & +Delta) -> Wrong Demo Sensitivity
      7. Exact decomposition of Reader vs Delta contributions & Multi-Demo consistency
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

    # 49 float64 accumulators for exact DDP reduction:
    # 0: count
    # 1: diff_self_teacher
    # 2: diff_stage1_reader_cross
    # 3: diff_self_student_anchor
    # 4: diff_cross_nodelta
    # 5: diff_cross_delta (1-step)
    # 6: diff_wrong_nodelta
    # 7: diff_wrong_delta
    # 8: mse_raw_stage1_cross_vs_zs
    # 9: mse_raw_cross_nodelta_vs_zs
    # 10: mse_raw_cross_delta_vs_zs (1-step)
    # 11: mse_raw_anchor_vs_zs
    # 12: mse_raw_reader_shift (R_S(M_D) - R_T(M_D))
    # 13: cos_stage1_cross_vs_zs
    # 14: cos_cross_nodelta_vs_zs
    # 15: cos_cross_delta_vs_zs (1-step)
    # 16: cos_anchor_vs_zs
    # 17: cos_reader_shift (R_S(M_D) vs R_T(M_D))
    # 18: cross_delta_sq (1-step)
    # 19: cross_z_sq
    # 20: stage1_cross_z_sq
    # 21: self_delta_sq (on z_self_student)
    # 22: wrong_delta_sq
    # 23: mse_raw_wrong_nodelta_vs_zs
    # 24: mse_raw_wrong_delta_vs_zs
    # 25: self_teacher_delta_sq (G(z_s, h_q))
    # 26: zs_sq
    # --- Stage 2.3 operator stability & 2-step correction accumulators ---
    # 27: diff_cross_delta_2step
    # 28: mse_raw_cross_2step_vs_zs
    # 29: cos_cross_2step_vs_zs
    # 30: cross_2step_delta_sq (G(z_1, h_q))
    # 31: zq1_sq
    # 32: step2_improved_count (mse(z2, zs) < mse(z1, zs))
    # 33: step1_dir_cos_sum
    # 34: step1_proj_frac_sum
    # 35: step1_mag_ratio_sum
    # 36: step1_g_rms_sum
    # 37: step1_d_rms_sum
    # 38: step1_g_rms_sq_sum
    # 39: step1_d_rms_sq_sum
    # 40: step1_gd_rms_sum
    # 41: step2_dir_cos_sum
    # 42: step2_proj_frac_sum
    # 43: step2_mag_ratio_sum
    # 44: step2_g_rms_sum
    # 45: step2_d_rms_sum
    # 46: step2_g_rms_sq_sum
    # 47: step2_d_rms_sq_sum
    # 48: step2_gd_rms_sum
    acc = torch.zeros(49, device=device, dtype=torch.float64)

    c_it = bank.iter_val_batches_joint(
        "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    w_it = bank.iter_val_batches_joint(
        "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_val
    )

    cur = 0
    for (dm_c, pm_c, qm_c, qpm_c, q_c, z_self, a_c, _), (
        dm_w,
        pm_w,
        _,
        _,
        q_w,
        _,
        a_w,
        _,
    ) in zip(c_it, w_it):
        B = len(dm_c)
        noise = local_noise[cur : cur + B].to(device, non_blocking=True)
        pertub = local_pertub[cur : cur + B].to(device, non_blocking=True)
        t = local_t[cur : cur + B].to(device, non_blocking=True)
        cur += B

        with amp_ctx:
            z_c0, h_q = policy.read_teacher_latent_from_memory(dm_c, pm_c, q_c)
            z_c, _ = policy.read_cross_latent_from_memory(dm_c, pm_c, q_c, h_q=h_q)
            z_q, delta_c = policy.adapt_latent(z_c, h_q)
            z_q2, delta_c2 = policy.adapt_latent(z_q, h_q)

            z_s_stu, _ = policy.read_cross_latent_from_memory(qm_c, qpm_c, q_c, h_q=h_q)
            _, delta_s_stu = policy.adapt_latent(z_s_stu, h_q)
            _, delta_s_teacher = policy.adapt_latent(z_self, h_q)

            z_w, _ = policy.read_cross_latent_from_memory(dm_w, pm_w, q_w, h_q=h_q)
            z_w_q, delta_w = policy.adapt_latent(z_w, h_q)

            l_self = policy.teacher.diffusion_loss_from_z_self(
                z_self, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_s1_c = policy.teacher.diffusion_loss_from_z_self(
                z_c0, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_anc = policy.teacher.diffusion_loss_from_z_self(
                z_s_stu, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_nd = policy.teacher.diffusion_loss_from_z_self(
                z_c, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_d = policy.teacher.diffusion_loss_from_z_self(
                z_q, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_d2 = policy.teacher.diffusion_loss_from_z_self(
                z_q2, a_c, noise=noise, pertub_noise=pertub, timesteps=t
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
        zq2_f = z_q2.float()
        zanc_f = z_s_stu.float()
        zw_f = z_w.float()
        zwq_f = z_w_q.float()
        d1_f = delta_c.float()
        d2_f = delta_c2.float()

        mse_z1_i = (zq_f - zs_f).square().mean(dim=(1, 2))
        mse_z2_i = (zq2_f - zs_f).square().mean(dim=(1, 2))

        gap1 = zs_f - zc_f
        gap1_sq_i = gap1.square().mean(dim=(1, 2))
        g1_rms_i = gap1_sq_i.clamp_min(0.0).sqrt()
        d1_rms_i = d1_f.square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()
        cos_dir1_i = F.cosine_similarity(d1_f.reshape(B, -1), gap1.reshape(B, -1), dim=-1)
        proj_frac1_i = (d1_f * gap1).sum(dim=(1, 2)) / gap1.square().sum(dim=(1, 2)).clamp_min(1e-12)
        mag_ratio1_i = d1_rms_i / g1_rms_i.clamp_min(1e-6)

        gap2 = zs_f - zq_f
        gap2_sq_i = gap2.square().mean(dim=(1, 2))
        g2_rms_i = gap2_sq_i.clamp_min(0.0).sqrt()
        d2_rms_i = d2_f.square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()
        cos_dir2_i = F.cosine_similarity(d2_f.reshape(B, -1), gap2.reshape(B, -1), dim=-1)
        proj_frac2_i = (d2_f * gap2).sum(dim=(1, 2)) / gap2.square().sum(dim=(1, 2)).clamp_min(1e-12)
        mag_ratio2_i = d2_rms_i / g2_rms_i.clamp_min(1e-6)

        acc[0] += float(B)
        acc[1] += l_self.double().sum()
        acc[2] += l_s1_c.double().sum()
        acc[3] += l_anc.double().sum()
        acc[4] += l_c_nd.double().sum()
        acc[5] += l_c_d.double().sum()
        acc[6] += l_w_nd.double().sum()
        acc[7] += l_w_d.double().sum()

        acc[8] += (zc0_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[9] += gap1_sq_i.double().sum()
        acc[10] += mse_z1_i.double().sum()
        acc[11] += (zanc_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[12] += (zc_f - zc0_f).square().mean(dim=(1, 2)).double().sum()

        acc[13] += F.cosine_similarity(zc0_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[14] += F.cosine_similarity(zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[15] += F.cosine_similarity(zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[16] += F.cosine_similarity(zanc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[17] += F.cosine_similarity(zc_f.reshape(B, -1), zc0_f.reshape(B, -1), dim=-1).double().sum()

        acc[18] += d1_f.square().mean(dim=(1, 2)).double().sum()
        acc[19] += zc_f.square().mean(dim=(1, 2)).double().sum()
        acc[20] += zc0_f.square().mean(dim=(1, 2)).double().sum()
        acc[21] += delta_s_stu.float().square().mean(dim=(1, 2)).double().sum()
        acc[22] += delta_w.float().square().mean(dim=(1, 2)).double().sum()
        acc[23] += (zw_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[24] += (zwq_f - zs_f).square().mean(dim=(1, 2)).double().sum()
        acc[25] += delta_s_teacher.float().square().mean(dim=(1, 2)).double().sum()
        acc[26] += zs_f.square().mean(dim=(1, 2)).double().sum()

        acc[27] += l_c_d2.double().sum()
        acc[28] += mse_z2_i.double().sum()
        acc[29] += F.cosine_similarity(zq2_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).double().sum()
        acc[30] += d2_f.square().mean(dim=(1, 2)).double().sum()
        acc[31] += zq_f.square().mean(dim=(1, 2)).double().sum()
        acc[32] += (mse_z2_i < mse_z1_i).double().sum()

        acc[33] += cos_dir1_i.double().sum()
        acc[34] += proj_frac1_i.double().sum()
        acc[35] += mag_ratio1_i.double().sum()
        acc[36] += g1_rms_i.double().sum()
        acc[37] += d1_rms_i.double().sum()
        acc[38] += (g1_rms_i * g1_rms_i).double().sum()
        acc[39] += (d1_rms_i * d1_rms_i).double().sum()
        acc[40] += (g1_rms_i * d1_rms_i).double().sum()

        acc[41] += cos_dir2_i.double().sum()
        acc[42] += proj_frac2_i.double().sum()
        acc[43] += mag_ratio2_i.double().sum()
        acc[44] += g2_rms_i.double().sum()
        acc[45] += d2_rms_i.double().sum()
        acc[46] += (g2_rms_i * g2_rms_i).double().sum()
        acc[47] += (d2_rms_i * d2_rms_i).double().sum()
        acc[48] += (g2_rms_i * d2_rms_i).double().sum()

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)

    def _pearson(sx: float, sy: float, sxx: float, syy: float, sxy: float, n: float) -> float:
        mx = sx / n
        my = sy / n
        vx = max(sxx / n - mx * mx, 0.0)
        vy = max(syy / n - my * my, 0.0)
        denom = math.sqrt(vx * vy)
        if denom <= 1e-12:
            return 0.0
        return float((sxy / n - mx * my) / denom)

    diff_self = float(acc[1].item() / total)
    diff_s1_c = float(acc[2].item() / total)
    diff_anc = float(acc[3].item() / total)
    diff_c_nd = float(acc[4].item() / total)
    diff_c_d = float(acc[5].item() / total)
    diff_w_nd = float(acc[6].item() / total)
    diff_w_d = float(acc[7].item() / total)
    diff_c_d2 = float(acc[27].item() / total)

    mse_s1_c = float(acc[8].item() / total)
    mse_c_nd = float(acc[9].item() / total)
    mse_c_d = float(acc[10].item() / total)
    mse_anc = float(acc[11].item() / total)
    mse_shift = float(acc[12].item() / total)
    mse_c_d2 = float(acc[28].item() / total)

    cross_delta_rms = math.sqrt(float(acc[18].item() / total))
    cross_z_rms = math.sqrt(float(acc[19].item() / total))
    s1_cross_z_rms = math.sqrt(float(acc[20].item() / total))
    reader_shift_rms = math.sqrt(mse_shift)
    self_stu_delta_rms = math.sqrt(float(acc[21].item() / total))
    wrong_delta_rms = math.sqrt(float(acc[22].item() / total))
    self_teacher_delta_rms = math.sqrt(float(acc[25].item() / total))
    self_z_rms = math.sqrt(float(acc[26].item() / total))
    cross_2step_delta_rms = math.sqrt(float(acc[30].item() / total))
    zq1_rms = math.sqrt(float(acc[31].item() / total))
    step1_gap_rms = math.sqrt(mse_c_nd)
    step2_gap_rms = math.sqrt(mse_c_d)

    step1_pearson = _pearson(
        float(acc[36].item()),
        float(acc[37].item()),
        float(acc[38].item()),
        float(acc[39].item()),
        float(acc[40].item()),
        total,
    )
    step2_pearson = _pearson(
        float(acc[44].item()),
        float(acc[45].item()),
        float(acc[46].item()),
        float(acc[47].item()),
        float(acc[48].item()),
        total,
    )

    total_diff_gain = diff_s1_c - diff_c_d
    reader_diff_gain = diff_s1_c - diff_c_nd
    delta_diff_gain = diff_c_nd - diff_c_d

    total_latent_gain = mse_s1_c - mse_c_d
    reader_latent_gain = mse_s1_c - mse_c_nd
    delta_latent_gain = mse_c_nd - mse_c_d

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "diff_loss": {
            "self_teacher": diff_self,
            "stage1_reader_cross": diff_s1_c,
            "self_student_anchor": diff_anc,
            "cross_no_delta": diff_c_nd,
            "cross_with_delta": diff_c_d,
            "cross_with_delta_2step": diff_c_d2,
            "step2_gain_over_step1": diff_c_d - diff_c_d2,
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
            "cross_2step_mse_raw": mse_c_d2,
            "self_anchor_mse_raw": mse_anc,
            "stage1_reader_cross_L_z": mse_s1_c / s_z2,
            "cross_no_delta_L_z": mse_c_nd / s_z2,
            "cross_with_delta_L_z": mse_c_d / s_z2,
            "cross_2step_L_z": mse_c_d2 / s_z2,
            "self_anchor_L_anchor": mse_anc / s_z2,
            "self_teacher_L_id": float(acc[25].item() / total) / s_z2,
            "wrong_demo_no_delta_L_z": float(acc[23].item() / total) / s_z2,
            "wrong_demo_with_delta_L_z": float(acc[24].item() / total) / s_z2,
            "reader_latent_mse_reduction_pct": 100.0
            * (1.0 - mse_c_nd / max(mse_s1_c, 1e-12)),
            "total_latent_mse_reduction_pct": 100.0
            * (1.0 - mse_c_d / max(mse_s1_c, 1e-12)),
            "delta_latent_mse_reduction_over_reader_pct": 100.0
            * (1.0 - mse_c_d / max(mse_c_nd, 1e-12)),
            "cross_2step_vs_1step_mse_reduction_pct": 100.0
            * (1.0 - mse_c_d2 / max(mse_c_d, 1e-12)),
            "cross_2step_improved_frac": float(acc[32].item() / total),
            "stage1_reader_cross_cos": float(acc[13].item() / total),
            "cross_no_delta_cos": float(acc[14].item() / total),
            "cross_with_delta_cos": float(acc[15].item() / total),
            "cross_2step_cos": float(acc[29].item() / total),
            "self_anchor_cos": float(acc[16].item() / total),
            "reader_output_shift_rms": reader_shift_rms,
            "reader_output_shift_rel_rms": reader_shift_rms / max(s1_cross_z_rms, 1e-12),
            "reader_output_shift_cos": float(acc[17].item() / total),
            "cross_delta_rms": cross_delta_rms,
            "cross_2step_delta_rms": cross_2step_delta_rms,
            "cross_2step_delta_rel_rms": cross_2step_delta_rms / max(zq1_rms, 1e-12),
            "cross_z_rms": cross_z_rms,
            "self_z_rms": self_z_rms,
            "cross_delta_rel_rms": cross_delta_rms / max(cross_z_rms, 1e-12),
            "self_identity_delta_rms": self_stu_delta_rms,
            "self_identity_delta_rel_rms": self_stu_delta_rms / max(s_z, 1e-12),
            "self_teacher_delta_rms": self_teacher_delta_rms,
            "self_teacher_delta_rel_rms": self_teacher_delta_rms / max(s_z, 1e-12),
            "self_teacher_delta_rel_rms_zs": self_teacher_delta_rms / max(self_z_rms, 1e-12),
            "wrong_demo_delta_rms": wrong_delta_rms,
        },
        "operator_stability": {
            "self_teacher_L_id": float(acc[25].item() / total) / s_z2,
            "self_teacher_delta_rms": self_teacher_delta_rms,
            "self_teacher_delta_rel_rms_sz": self_teacher_delta_rms / max(s_z, 1e-12),
            "self_teacher_delta_rel_rms_zs": self_teacher_delta_rms / max(self_z_rms, 1e-12),
            "self_student_delta_rms": self_stu_delta_rms,
            "self_student_delta_rel_rms_sz": self_stu_delta_rms / max(s_z, 1e-12),
            "step1_gap_rms": step1_gap_rms,
            "step1_delta_rms": cross_delta_rms,
            "step1_delta_to_gap_ratio_global": cross_delta_rms / max(step1_gap_rms, 1e-12),
            "step1_delta_to_gap_ratio_mean": float(acc[35].item() / total),
            "step1_mag_gap_pearson_r": step1_pearson,
            "step1_direction_cosine_mean": float(acc[33].item() / total),
            "step1_projection_fraction_mean": float(acc[34].item() / total),
            "step2_gap_rms": step2_gap_rms,
            "step2_delta_rms": cross_2step_delta_rms,
            "step2_delta_to_gap_ratio_global": cross_2step_delta_rms / max(step2_gap_rms, 1e-12),
            "step2_delta_to_gap_ratio_mean": float(acc[43].item() / total),
            "step2_mag_gap_pearson_r": step2_pearson,
            "step2_direction_cosine_mean": float(acc[41].item() / total),
            "step2_projection_fraction_mean": float(acc[42].item() / total),
            "step1_mse_raw": mse_c_d,
            "step2_mse_raw": mse_c_d2,
            "step2_L_refine": mse_c_d2 / s_z2,
            "step2_vs_step1_mse_reduction_pct": 100.0
            * (1.0 - mse_c_d2 / max(mse_c_d, 1e-12)),
            "step2_improved_frac": float(acc[32].item() / total),
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
            "self_student_anchor",
            "cross_no_delta",
            "cross_with_delta",
            "cross_with_delta_2step",
            "wrong_demo_no_delta",
            "wrong_demo_with_delta",
        ]
        err = {m: torch.zeros(11, device=device, dtype=torch.float64) for m in modes}

        c_it2 = bank.iter_val_batches_joint(
            "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        w_it2 = bank.iter_val_batches_joint(
            "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        cur = 0
        for (dm_c, pm_c, qm_c, qpm_c, q_c, z_self, a_c, _), (
            dm_w,
            pm_w,
            _,
            _,
            q_w,
            _,
            a_w,
            _,
        ) in zip(c_it2, w_it2):
            B = len(dm_c)
            z0 = local_z0[cur : cur + B].to(device, non_blocking=True)
            cur += B
            with amp_ctx:
                z_c0, h_q = policy.read_teacher_latent_from_memory(dm_c, pm_c, q_c)
                z_s_stu, _ = policy.read_cross_latent_from_memory(qm_c, qpm_c, q_c, h_q=h_q)
                z_c, _ = policy.read_cross_latent_from_memory(dm_c, pm_c, q_c, h_q=h_q)
                z_q, _ = policy.adapt_latent(z_c, h_q)
                z_q2, _ = policy.adapt_latent(z_q, h_q)
                z_w, _ = policy.read_cross_latent_from_memory(dm_w, pm_w, q_w, h_q=h_q)
                z_w_q, _ = policy.adapt_latent(z_w, h_q)

                pred_self = policy.teacher.sample_from_z_self(z_self, init_noise=z0)
                pred_s1_c = policy.teacher.sample_from_z_self(z_c0, init_noise=z0)
                pred_anc = policy.teacher.sample_from_z_self(z_s_stu, init_noise=z0)
                pred_c_nd = policy.teacher.sample_from_z_self(z_c, init_noise=z0)
                pred_c_d = policy.teacher.sample_from_z_self(z_q, init_noise=z0)
                pred_c_d2 = policy.teacher.sample_from_z_self(z_q2, init_noise=z0)
                pred_w_nd = policy.teacher.sample_from_z_self(z_w, init_noise=z0)
                pred_w_d = policy.teacher.sample_from_z_self(z_w_q, init_noise=z0)

            err["self_teacher"] += _action_error_sums(pred_self, a_c)
            err["stage1_reader_cross"] += _action_error_sums(pred_s1_c, a_c)
            err["self_student_anchor"] += _action_error_sums(pred_anc, a_c)
            err["cross_no_delta"] += _action_error_sums(pred_c_nd, a_c)
            err["cross_with_delta"] += _action_error_sums(pred_c_d, a_c)
            err["cross_with_delta_2step"] += _action_error_sums(pred_c_d2, a_c)
            err["wrong_demo_no_delta"] += _action_error_sums(pred_w_nd, a_w)
            err["wrong_demo_with_delta"] += _action_error_sums(pred_w_d, a_w)

        ddim_out: Dict[str, Any] = {
            m: _unpack_action_errors(_all_reduce_sum(err[m])) for m in modes
        }
        s1_mse10 = ddim_out["stage1_reader_cross"]["action10_mse"]
        nd_mse10 = ddim_out["cross_no_delta"]["action10_mse"]
        cd_mse10 = ddim_out["cross_with_delta"]["action10_mse"]
        cd2_mse10 = ddim_out["cross_with_delta_2step"]["action10_mse"]
        reader_mse10_gain = s1_mse10 - nd_mse10
        delta_mse10_gain = nd_mse10 - cd_mse10
        total_mse10_gain = s1_mse10 - cd_mse10

        s1_mae7 = ddim_out["stage1_reader_cross"]["action7_mae"]
        nd_mae7 = ddim_out["cross_no_delta"]["action7_mae"]
        cd_mae7 = ddim_out["cross_with_delta"]["action7_mae"]
        cd2_mae7 = ddim_out["cross_with_delta_2step"]["action7_mae"]

        ddim_out["reader_gain_action10_mse"] = reader_mse10_gain
        ddim_out["delta_gain_action10_mse"] = delta_mse10_gain
        ddim_out["total_gain_action10_mse"] = total_mse10_gain
        ddim_out["step2_gain_action10_mse"] = cd_mse10 - cd2_mse10
        ddim_out["reader_gain_action7_mae"] = s1_mae7 - nd_mae7
        ddim_out["delta_gain_action7_mae"] = nd_mae7 - cd_mae7
        ddim_out["total_gain_action7_mae"] = s1_mae7 - cd_mae7
        ddim_out["step2_gain_action7_mae"] = cd_mae7 - cd2_mae7
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
                dm1, pm1, q1, zs1, a1, _ = bank.gather_batch(
                    split="val",
                    sample_indices=s_idx,
                    demo_episodes=bank.val_cross_a[s_idx],
                    is_self=np.zeros(B, dtype=bool),
                )
                dm2, pm2, q2, _, _, _ = bank.gather_batch(
                    split="val",
                    sample_indices=s_idx,
                    demo_episodes=bank.val_cross_b[s_idx],
                    is_self=np.zeros(B, dtype=bool),
                )
                with amp_ctx:
                    zc1, hq = policy.read_cross_latent_from_memory(dm1, pm1, q1)
                    zc2, _ = policy.read_cross_latent_from_memory(dm2, pm2, q2, h_q=hq)
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

        run_name = args.run_name if args.run_name else "stage2_1_joint_reader_delta"
        run_dir = args.output_dir / run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        seed_everything(args.seed)
        bank = Stage2_1TransferBank(
            clip_cache_dir=args.clip_cache_dir,
            z_self_cache_dir=args.z_self_cache_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            z_self_on_device=(device.type == "cuda" and not args.z_self_on_cpu),
            self_ratio=args.self_ratio,
            seed=args.seed,
        )

        # Fixed s_z = 7.822748 (verify consistency with z_self cache)
        s_z = float(args.latent_scale_s_z)
        if abs(s_z - bank.latent_scale_s_z) > 1e-4:
            raise RuntimeError(
                f"Configured s_z={s_z} diverges from cached train_z_self_rms={bank.latent_scale_s_z}"
            )

        # Initialize strictly from Stage 1 checkpoint (NEVER load Stage 2 weights/opt/EMA!)
        teacher = load_frozen_stage1_teacher(args.stage1_ckpt, device=device)
        seed_everything(args.seed)
        policy = E6Stage2_1Policy(
            teacher,
            latent_scale_s_z=s_z,
            reader_qk_scope=args.reader_qk_scope,
            residual_layers=args.residual_layers,
            residual_dropout=args.residual_dropout,
        ).to(device)

        # Explicit gradient & freezing audit on rank 0 before training
        if rank == 0 and args.check_grads:
            policy.train()
            dm_g, pm_g, qm_g, qpm_g, q_g, zs_g, a_g, is_s_g = next(
                bank.iter_train_batches_joint(1, min(8, per_rank_batch), max_samples=16)
            )
            audit_la = 1.0 if args.lambda_a != 0.0 else 0.0
            audit_lz = (
                0.0
                if args.lambda_a != 0.0
                else (
                    args.lambda_z_start
                    if args.lambda_z_schedule == "hold_linear"
                    else args.lambda_z
                )
            )
            audit_lanc = 0.0 if args.lambda_a != 0.0 else args.lambda_anchor
            unet_called = False
            orig_diff_fn = policy.teacher.diffusion_loss_from_z_self

            def _guarded_diff(*a, **kw):
                nonlocal unet_called
                unet_called = True
                return orig_diff_fn(*a, **kw)

            if audit_la == 0.0:
                policy.teacher.diffusion_loss_from_z_self = _guarded_diff
            try:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    out_g = policy(
                        dm_g,
                        pm_g,
                        qm_g,
                        qpm_g,
                        q_g,
                        zs_g,
                        a_g,
                        is_s_g,
                        lambda_a=audit_la,
                        lambda_z=audit_lz,
                        lambda_anchor=audit_lanc,
                        lambda_id=args.lambda_id,
                        lambda_refine=args.lambda_refine,
                    )
                out_g["loss"].backward()
            finally:
                policy.teacher.diffusion_loss_from_z_self = orig_diff_fn

            out_proj_gn = float(
                policy.residual_transformer.out_proj.weight.grad.float().norm().item()
            )
            reader_qk_gns = [
                float(p.grad.float().norm().item())
                for n, p in policy.cross_reader.named_parameters()
                if p.requires_grad and p.grad is not None
            ]
            reader_qk_min_gn = min(reader_qk_gns) if reader_qk_gns else 0.0
            reader_qk_max_gn = max(reader_qk_gns) if reader_qk_gns else 0.0
            teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
            frozen_reader_has_grad = any(
                p.grad is not None
                for p in policy.cross_reader.parameters()
                if not p.requires_grad
            )
            v_bias_max_grad = max(
                float(p.grad[2 * 768 :].abs().max().item())
                for n, p in policy.cross_reader.named_parameters()
                if p.requires_grad and n.endswith(".in_proj_bias") and p.grad is not None
            )
            audit_diff_val = float(out_g["diff_loss"].item())
            policy.zero_grad(set_to_none=True)

            # Stage 2.3 stop-gradient audit for L_id / L_refine:
            # verify neither L_id nor L_refine leaks gradient into cross_reader or teacher,
            # while L_refine DOES produce non-zero gradient on 2nd residual forward.
            s23_audit_ok = True
            s23_desc = ""
            if args.lambda_id != 0.0 or args.lambda_refine != 0.0:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    out_s23 = policy(
                        dm_g,
                        pm_g,
                        qm_g,
                        qpm_g,
                        q_g,
                        zs_g,
                        a_g,
                        is_s_g,
                        lambda_a=0.0,
                        lambda_z=0.0,
                        lambda_anchor=0.0,
                        lambda_id=args.lambda_id,
                        lambda_refine=args.lambda_refine,
                    )
                out_s23["loss"].backward()
                s23_res_gn = float(
                    policy.residual_transformer.out_proj.weight.grad.float().norm().item()
                )
                s23_reader_has_grad = any(
                    p.grad is not None for p in policy.cross_reader.parameters()
                )
                s23_teacher_has_grad = any(
                    p.grad is not None for p in policy.teacher.parameters()
                )
                policy.zero_grad(set_to_none=True)
                s23_desc = (
                    f" | s23(id={args.lambda_id},ref={args.lambda_refine}): "
                    f"res_out_gn={s23_res_gn:.6f}, reader_grad={s23_reader_has_grad}, "
                    f"teacher_grad={s23_teacher_has_grad}"
                )
                if s23_reader_has_grad or s23_teacher_has_grad or not math.isfinite(s23_res_gn):
                    s23_audit_ok = False
                if args.lambda_refine != 0.0 and s23_res_gn <= 0.0:
                    s23_audit_ok = False

            audit_label = (
                "pure L_diff via frozen UNet"
                if audit_la != 0.0
                else f"pure Latent Alignment (lz={audit_lz}, anc={audit_lanc}, unet_skipped={not unet_called})"
            )
            print(
                f"[E6 Stage2.1 Grad Check ({audit_label})] "
                f"res.out_proj_gn={out_proj_gn:.6f} | "
                f"reader_qk_gn=[{reader_qk_min_gn:.6e}, {reader_qk_max_gn:.6e}] "
                f"({len(reader_qk_gns)}/{len(policy.reader_qk_audit['trainable_param_names'])} tensors) | "
                f"teacher_any_grad={teacher_has_grad} | frozen_reader_any_grad={frozen_reader_has_grad} | "
                f"v_bias_max_grad={v_bias_max_grad:.1e} | diff_loss={audit_diff_val:.4f}{s23_desc}",
                flush=True,
            )
            if (
                out_proj_gn <= 0.0
                or not math.isfinite(out_proj_gn)
                or len(reader_qk_gns) != len(policy.reader_qk_audit["trainable_param_names"])
                or reader_qk_min_gn <= 0.0
                or teacher_has_grad
                or frozen_reader_has_grad
                or v_bias_max_grad != 0.0
                or (audit_la == 0.0 and (unet_called or audit_diff_val != 0.0))
                or not s23_audit_ok
            ):
                raise RuntimeError("Stage 2.1 gradient / freeze audit FAILED!")
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
            "stage": "e6_stage2_1_joint_reader_delta",
            "run_name": run_name,
            "smoke": bool(args.smoke),
            "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
            "clip_cache_dir": str(Path(args.clip_cache_dir).resolve()),
            "z_self_cache_dir": str(Path(args.z_self_cache_dir).resolve()),
            "latent_scale_s_z": s_z,
            "lambda_a": args.lambda_a,
            "lambda_diff": args.lambda_a,
            "lambda_z": args.lambda_z,
            "lambda_z_schedule": args.lambda_z_schedule,
            "lambda_z_start": args.lambda_z_start if args.lambda_z_schedule == "hold_linear" else args.lambda_z,
            "lambda_z_end": args.lambda_z_end if args.lambda_z_schedule == "hold_linear" else args.lambda_z,
            "lambda_z_hold_frac": args.lambda_z_hold_frac if args.lambda_z_schedule == "hold_linear" else 0.0,
            "lambda_z_hold_steps": (
                int(round(args.lambda_z_hold_frac * total_steps))
                if args.lambda_z_schedule == "hold_linear"
                else 0
            ),
            "lambda_anchor": args.lambda_anchor,
            "lambda_id": args.lambda_id,
            "lambda_refine": args.lambda_refine,
            "self_ratio": args.self_ratio,
            "cross_ratio": round(1.0 - args.self_ratio, 6),
            "reader_qk_scope": args.reader_qk_scope,
            "reader_qk_audit": policy.reader_qk_audit,
            "residual_layers": args.residual_layers,
            "residual_heads": 8,
            "residual_ffn_dim": 3072,
            "residual_dropout": args.residual_dropout,
            "samples_per_epoch": epoch_len,
            "total_cached_train_starts": bank.num_train_samples(),
            "total_cached_val_cross_starts": int(bank.val_valid_cross_mask.sum()),
            "val_multi_demo_starts": int(bank.val_has_two_demos.sum()),
            "seed": args.seed,
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
            if ck.get("stage") != "e6_stage2_1_joint_reader_delta":
                raise RuntimeError(
                    f"Refusing to resume non-Stage-2.1 checkpoint: stage={ck.get('stage')}"
                )
            policy.cross_reader.load_state_dict(ck["cross_reader"], strict=True)
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
                    f"[E6 Stage2.1] Resumed from epoch {ck['epoch']} "
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
            lz_desc = (
                f"{args.lambda_z_start}->{args.lambda_z_end}@hold{args.lambda_z_hold_frac:.0%}"
                if args.lambda_z_schedule == "hold_linear"
                else f"{args.lambda_z}"
            )
            print(
                f"[E6 Stage2.1] run={run_name} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={args.warmup_steps}) | "
                f"reader_lr={args.reader_lr:.2e}->{args.reader_lr * args.min_lr_ratio:.2e} | "
                f"res_lr={args.residual_lr:.2e}->{args.residual_lr * args.min_lr_ratio:.2e} | "
                f"weights=(diff:{args.lambda_a}, z:{lz_desc}, anc:{args.lambda_anchor}, "
                f"id:{args.lambda_id}, ref:{args.lambda_refine}) s_z={s_z:.6f} | "
                f"trainable_eff={ps['total_trainable_effective']:,} "
                f"(reader_qk={ps['cross_reader_trainable_effective_qk']:,} [{args.reader_qk_scope}], "
                f"res={ps['residual_transformer_trainable']:,}, teacher_frozen={ps['stage1_teacher_frozen']:,})",
                flush=True,
            )

        # Epoch-0 reference validation (exact Stage 1 baseline: R_S == R_T, delta_z == 0)
        init_val = None
        if start_epoch == 1:
            init_val = evaluate_stage2_1_validation(
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
                print(
                    f"[E6 Stage2.1 Init Val] diff self={dl['self_teacher']:.4f} "
                    f"s1_cross={dl['stage1_reader_cross']:.4f} c_nd={dl['cross_no_delta']:.4f} "
                    f"c+d={dl['cross_with_delta']:.4f} | "
                    f"L_z={lt['cross_with_delta_L_z']:.4f} L_anc={lt['self_anchor_L_anchor']:.6f} | "
                    f"DDIM mse10 self={dm['self_teacher']['action10_mse']:.4f} "
                    f"s1_cross={dm['stage1_reader_cross']['action10_mse']:.4f} "
                    f"c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"wrong={dm['wrong_demo_with_delta']['action10_mse']:.4f}",
                    flush=True,
                )

        if start_epoch == 1:
            # Align PyTorch/CUDA RNG right before training so runs with identical seed
            # receive identical diffusion noise/timesteps/dropout sequences from step 0.
            seed_everything(args.seed + 1000 + rank)

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            policy.teacher.eval()
            policy.cross_reader.eval()

            # [n_tot, loss_sum, diff_sum, lz_sum, lanc_sum, nd_lz_sum, cos_nd_sum, cos_d_sum,
            #  cross_dz_sq, n_cross, self_dz_sq, n_self, gnorm_sum, n_steps, lambda_z_sum,
            #  lid_sum, lref_sum, id_dz_sq, ref_dz_sq]
            tr_acc = torch.zeros(19, device=device, dtype=torch.float64)
            ep_lambda_z_first: Optional[float] = None
            ep_lambda_z_last: float = 0.0

            for (
                demo_mem,
                demo_pad_mask,
                query_mem,
                query_pad_mask,
                query_rgb,
                z_self,
                actions_10d,
                is_self,
            ) in bank.iter_train_batches_joint(
                epoch,
                per_rank_batch,
                rank=rank,
                world_size=world,
                max_samples=args.max_train_samples,
            ):
                cur_lambda_z = get_scheduled_lambda_z(
                    global_step,
                    total_steps=total_steps,
                    schedule=args.lambda_z_schedule,
                    lambda_z=args.lambda_z,
                    lambda_z_start=args.lambda_z_start,
                    lambda_z_end=args.lambda_z_end,
                    hold_frac=args.lambda_z_hold_frac,
                )
                if ep_lambda_z_first is None:
                    ep_lambda_z_first = cur_lambda_z
                ep_lambda_z_last = cur_lambda_z

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(
                        demo_mem,
                        demo_pad_mask,
                        query_mem,
                        query_pad_mask,
                        query_rgb,
                        z_self,
                        actions_10d,
                        is_self,
                        lambda_a=args.lambda_a,
                        lambda_z=cur_lambda_z,
                        lambda_anchor=args.lambda_anchor,
                        lambda_id=args.lambda_id,
                        lambda_refine=args.lambda_refine,
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

                B = float(len(demo_mem))
                ns = float(out["n_self"].item())
                nc = float(out["n_cross"].item())
                tr_acc[0] += B
                tr_acc[1] += float(out["loss"].item()) * B
                tr_acc[2] += float(out["diff_loss"].item()) * B
                tr_acc[3] += float(out["latent_loss_norm"].item()) * B
                tr_acc[4] += float(out["anchor_loss_norm"].item()) * B
                tr_acc[5] += float(out["nodelta_loss_norm"].item()) * B
                tr_acc[6] += float(out["cos_nodelta_self"].item()) * B
                tr_acc[7] += float(out["cos_adapted_self"].item()) * B
                tr_acc[8] += (float(out["cross_delta_rms"].item()) ** 2) * nc
                tr_acc[9] += nc
                tr_acc[10] += (float(out["self_delta_rms"].item()) ** 2) * ns
                tr_acc[11] += ns
                tr_acc[12] += grad_norm
                tr_acc[13] += 1.0
                tr_acc[14] += cur_lambda_z
                tr_acc[15] += float(out["id_loss_norm"].item()) * B
                tr_acc[16] += float(out["refine_loss_norm"].item()) * B
                tr_acc[17] += (float(out["id_delta_rms"].item()) ** 2) * B
                tr_acc[18] += (float(out["refine_delta_rms"].item()) ** 2) * B

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
                "L_anchor": float(tr_acc[4].item() / n_tot),
                "L_id": float(tr_acc[15].item() / n_tot),
                "L_refine": float(tr_acc[16].item() / n_tot),
                "id_delta_rms": math.sqrt(float(tr_acc[17].item() / n_tot)),
                "refine_delta_rms": math.sqrt(float(tr_acc[18].item() / n_tot)),
                "lambda_z_first": float(ep_lambda_z_first if ep_lambda_z_first is not None else 0.0),
                "lambda_z_last": float(ep_lambda_z_last),
                "lambda_z_mean": float(tr_acc[14].item() / n_steps_tot),
                "nodelta_L_z": float(tr_acc[5].item() / n_tot),
                "cos_nodelta_self": float(tr_acc[6].item() / n_tot),
                "cos_adapted_self": float(tr_acc[7].item() / n_tot),
                "cross_delta_rms": math.sqrt(float(tr_acc[8].item() / n_cross_tot)),
                "self_delta_rms": math.sqrt(float(tr_acc[10].item() / n_self_tot)),
                "grad_norm_mean": float(tr_acc[12].item() / n_steps_tot),
                "reader_lr": float(last_lrs[0]),
                "residual_lr": float(last_lrs[2]),
                "ema_decay": float(ema.cur_decay_value or 0.0) if ema is not None else None,
                "train_seconds": round(t_train, 2),
                "steps_per_sec": round((n_steps_tot / world) / max(t_train, 1e-6), 2),
            }

            # Evaluate EMA weights (or live weights if --no-ema)
            t_val0 = time.monotonic()
            if ema is not None:
                ema.store(trainable_params)
                ema.copy_to(trainable_params)

            val_metrics = evaluate_stage2_1_validation(
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

            # Save best checkpoint (EMA weights if enabled) BEFORE restoring live weights
            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "stage": "e6_stage2_1_joint_reader_delta",
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_cross_ddim_mse10": best_val_mse,
                        "latent_scale_s_z": s_z,
                        "cross_reader": {
                            k: v.detach().cpu()
                            for k, v in policy.cross_reader.state_dict().items()
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
                            "stage": "e6_stage2_1_joint_reader_delta",
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_cross_ddim_mse10": best_val_mse,
                            "latent_scale_s_z": s_z,
                            "cross_reader": {
                                k: v.detach().cpu()
                                for k, v in policy.cross_reader.state_dict().items()
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
                        "stage": "e6_stage2_1_joint_reader_delta",
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
                ct = val_metrics["contributions"]
                op = val_metrics["operator_stability"]
                print(
                    f"[E6 Stage2.1 Ep {epoch:02d}/{args.epochs}] "
                    f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, "
                    f"Lz={train_metrics['L_z']:.4f}, Lanc={train_metrics['L_anchor']:.5f}, "
                    f"Lid={train_metrics['L_id']:.5f}, Lref={train_metrics['L_refine']:.4f}) | "
                    f"val diff nd={dl['cross_no_delta']:.4f} -> c+d={dl['cross_with_delta']:.4f} "
                    f"-> 2st={dl['cross_with_delta_2step']:.4f} | "
                    f"Lz nd={lt['cross_no_delta_L_z']:.4f} 1st={lt['cross_with_delta_L_z']:.4f} "
                    f"2st={lt['cross_2step_L_z']:.4f} (imp={100.0 * op['step2_improved_frac']:.1f}%) "
                    f"self_id={100.0 * op['self_teacher_delta_rel_rms_sz']:.2f}% "
                    f"dir_cos={op['step1_direction_cosine_mean']:.3f} r={op['step1_mag_gap_pearson_r']:.3f} | "
                    f"DDIM mse10 nd={dm['cross_no_delta']['action10_mse']:.4f} -> "
                    f"1st={dm['cross_with_delta']['action10_mse']:.4f} -> "
                    f"2st={dm['cross_with_delta_2step']['action10_mse']:.4f} "
                    f"(R:{ct['ddim_action10_mse_reader_share_pct']:.1f}%/D:{ct['ddim_action10_mse_delta_share_pct']:.1f}%, "
                    f"mae7={dm['cross_with_delta']['action7_mae']:.4f}) | "
                    f"lr=({train_metrics['reader_lr']:.1e},{train_metrics['residual_lr']:.1e}) "
                    f"t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s){'*' if is_best else ''}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Reload best.pt and run FULL validation across all valid val cross & multi-demo samples
        reloaded, best_payload = load_stage2_1_policy(
            best_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        final_eval = evaluate_stage2_1_validation(
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
                "stage": "e6_stage2_1_joint_reader_delta",
                "run_name": run_name,
                "seed": args.seed,
                "effective_global_steps": global_step,
                "best_epoch": best_payload["epoch"],
                "best_global_step": best_payload.get("global_step"),
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_cross_ddim_mse10": best_payload["best_val_cross_ddim_mse10"],
                "lambda_diff": config["lambda_a"],
                "lambda_z_schedule": config["lambda_z_schedule"],
                "lambda_z_start": config["lambda_z_start"],
                "lambda_z_end": config["lambda_z_end"],
                "lambda_z_hold_frac": config["lambda_z_hold_frac"],
                "lambda_anchor": config["lambda_anchor"],
                "lambda_id": config["lambda_id"],
                "lambda_refine": config["lambda_refine"],
                "parameter_summary": config["parameter_summary"],
                "final_eval": final_eval,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[E6 Stage2.1 FINAL BEST EVAL (ep={best_payload['epoch']})] "
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "stage2_1")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--reader-lr", type=float, default=3.0e-6)
    parser.add_argument("--residual-lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-a", "--lambda-diff", dest="lambda_a", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=0.5)
    parser.add_argument(
        "--lambda-z-schedule",
        choices=("constant", "hold_linear"),
        default="constant",
    )
    parser.add_argument("--lambda-z-start", type=float, default=0.5)
    parser.add_argument("--lambda-z-end", type=float, default=0.1)
    parser.add_argument("--lambda-z-hold-frac", type=float, default=0.2)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)
    parser.add_argument("--lambda-id", type=float, default=0.0)
    parser.add_argument("--lambda-refine", type=float, default=0.0)
    parser.add_argument("--eval-only-ckpt", type=Path, default=None)
    parser.add_argument("--eval-out", type=Path, default=None)
    parser.add_argument("--latent-scale-s-z", type=float, default=FIXED_LATENT_SCALE_S_Z)
    parser.add_argument("--self-ratio", type=float, default=0.025)
    parser.add_argument(
        "--reader-qk-scope", choices=("both", "cross_only"), default="both"
    )
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


def run_eval_only(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    bank = Stage2_1TransferBank(
        clip_cache_dir=args.clip_cache_dir,
        z_self_cache_dir=args.z_self_cache_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        z_self_on_device=(device.type == "cuda" and not args.z_self_on_cpu),
        self_ratio=args.self_ratio,
        seed=args.seed,
    )
    policy, payload = load_stage2_1_policy(
        args.eval_only_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
    )
    res = evaluate_stage2_1_validation(
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
    report = {
        "checkpoint": str(Path(args.eval_only_ckpt).resolve()),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "val_seed": args.val_seed,
        "eval": res,
    }
    if args.eval_out is not None:
        args.eval_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report, args.eval_out)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.eval_only_ckpt is not None:
        run_eval_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
