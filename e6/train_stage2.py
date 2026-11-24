"""Training, validation, smoke test, and full offline evaluation for E6 Stage 2
(Latent Behavior Transfer, CLIP, RGB-only).

Trains ONLY `ResidualTransformer` (`delta_z = G(z_cross, h_q)`) on top of the frozen
Stage 1 EMA Teacher (`PromptPooler`, `BehaviorReferenceReader`, `ConditionalUnet1D`).
Selects Best checkpoint by Validation Cross DDIM Action MSE (`action10_mse`).
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
from .stage2_data import DEFAULT_STAGE2_CACHE_DIR, Stage2TransferBank
from .stage2_model import DEFAULT_STAGE1_CKPT, E6Stage2Policy, load_stage2_policy


def get_cosine_schedule_with_min_lr(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr: float,
    min_lr: float,
) -> LambdaLR:
    """Linear warmup for `num_warmup_steps`, then cosine decay from `lr` down to `min_lr`."""
    min_ratio = float(min_lr) / max(float(lr), 1e-12)
    min_ratio = max(0.0, min(1.0, min_ratio))

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        denom = max(1, num_training_steps - num_warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - num_warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate_stage2_validation(
    policy: E6Stage2Policy,
    bank: Stage2TransferBank,
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
    """Comprehensive distributed evaluation comparing:
      1. Self Teacher (`z_self`)
      2. Cross No-Delta (`z_cross`)
      3. Cross + Delta (`z_cross + delta_z`)
      4. Wrong-task Demo (No-Delta & +Delta) to verify Residual does not bypass Demo
      5. Multi-Demo consistency (two distinct Cross Demos D1 != D2 for the same Query)
    """
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    s_z = float(policy.latent_scale_s_z.item())
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

    # 16 accumulators (float64 for exact DDP reduction):
    # 0: count
    # 1: diff_self, 2: diff_cross_nodelta, 3: diff_cross_delta
    # 4: diff_wrong_nodelta, 5: diff_wrong_delta
    # 6: mse_raw_cross_nodelta, 7: mse_raw_cross_delta
    # 8: lz_norm_cross_nodelta, 9: lz_norm_cross_delta
    # 10: cos_cross_nodelta, 11: cos_cross_delta
    # 12: cross_delta_sq, 13: cross_z_sq, 14: self_delta_sq, 15: wrong_delta_sq
    acc = torch.zeros(16, device=device, dtype=torch.float64)

    c_it = bank.iter_val_batches("cross_a", batch_size, rank=rank, world_size=world, max_samples=n_val)
    w_it = bank.iter_val_batches("wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_val)

    cur = 0
    for (dm_c, pm_c, q_c, z_self, a_c, _), (dm_w, pm_w, q_w, _, a_w, _) in zip(c_it, w_it):
        B = len(dm_c)
        noise = local_noise[cur : cur + B].to(device, non_blocking=True)
        pertub = local_pertub[cur : cur + B].to(device, non_blocking=True)
        t = local_t[cur : cur + B].to(device, non_blocking=True)
        cur += B

        with amp_ctx:
            z_cross, h_q = policy.read_latent_from_memory(dm_c, pm_c, q_c)
            z_adapted, delta_c = policy.adapt_latent(z_cross, h_q)
            _, delta_s = policy.adapt_latent(z_self, h_q)

            z_wrong, _ = policy.read_latent_from_memory(dm_w, pm_w, q_w)
            z_wrong_adapted, delta_w = policy.adapt_latent(z_wrong, h_q)

            l_self = policy.teacher.diffusion_loss_from_z_self(
                z_self, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_nd = policy.teacher.diffusion_loss_from_z_self(
                z_cross, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_c_d = policy.teacher.diffusion_loss_from_z_self(
                z_adapted, a_c, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_w_nd = policy.teacher.diffusion_loss_from_z_self(
                z_wrong, a_w, noise=noise, pertub_noise=pertub, timesteps=t
            )
            l_w_d = policy.teacher.diffusion_loss_from_z_self(
                z_wrong_adapted, a_w, noise=noise, pertub_noise=pertub, timesteps=t
            )

        zs_f = z_self.float()
        zc_f = z_cross.float()
        za_f = z_adapted.float()
        mse_nd = (zc_f - zs_f).square().mean(dim=(1, 2))
        mse_d = (za_f - zs_f).square().mean(dim=(1, 2))
        cos_nd = F.cosine_similarity(zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1)
        cos_d = F.cosine_similarity(za_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1)

        acc[0] += float(B)
        acc[1] += l_self.double().sum()
        acc[2] += l_c_nd.double().sum()
        acc[3] += l_c_d.double().sum()
        acc[4] += l_w_nd.double().sum()
        acc[5] += l_w_d.double().sum()
        acc[6] += mse_nd.double().sum()
        acc[7] += mse_d.double().sum()
        acc[8] += (mse_nd / (s_z * s_z)).double().sum()
        acc[9] += (mse_d / (s_z * s_z)).double().sum()
        acc[10] += cos_nd.double().sum()
        acc[11] += cos_d.double().sum()
        acc[12] += delta_c.float().square().mean(dim=(1, 2)).double().sum()
        acc[13] += zc_f.square().mean(dim=(1, 2)).double().sum()
        acc[14] += delta_s.float().square().mean(dim=(1, 2)).double().sum()
        acc[15] += delta_w.float().square().mean(dim=(1, 2)).double().sum()

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)
    cross_delta_rms = math.sqrt(float(acc[12].item() / total))
    cross_z_rms = math.sqrt(float(acc[13].item() / total))
    self_delta_rms = math.sqrt(float(acc[14].item() / total))
    wrong_delta_rms = math.sqrt(float(acc[15].item() / total))

    diff_self = float(acc[1].item() / total)
    diff_c_nd = float(acc[2].item() / total)
    diff_c_d = float(acc[3].item() / total)
    diff_w_nd = float(acc[4].item() / total)
    diff_w_d = float(acc[5].item() / total)

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "diff_loss": {
            "self_teacher": diff_self,
            "cross_no_delta": diff_c_nd,
            "cross_with_delta": diff_c_d,
            "delta_gain_over_cross_no_delta": diff_c_nd - diff_c_d,
            "wrong_demo_no_delta": diff_w_nd,
            "wrong_demo_with_delta": diff_w_d,
            "demo_sensitivity_with_delta": diff_w_d - diff_c_d,
        },
        "latent": {
            "s_z": s_z,
            "cross_no_delta_mse_raw": float(acc[6].item() / total),
            "cross_with_delta_mse_raw": float(acc[7].item() / total),
            "cross_no_delta_L_z": float(acc[8].item() / total),
            "cross_with_delta_L_z": float(acc[9].item() / total),
            "latent_mse_reduction_pct": 100.0
            * (1.0 - float(acc[7].item()) / max(float(acc[6].item()), 1e-12)),
            "cross_no_delta_cos": float(acc[10].item() / total),
            "cross_with_delta_cos": float(acc[11].item() / total),
            "cross_delta_rms": cross_delta_rms,
            "cross_z_rms": cross_z_rms,
            "cross_delta_rel_rms": cross_delta_rms / max(cross_z_rms, 1e-12),
            "self_identity_delta_rms": self_delta_rms,
            "self_identity_delta_rel_rms": self_delta_rms / max(s_z, 1e-12),
            "wrong_demo_delta_rms": wrong_delta_rms,
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
            "wrong_demo_no_delta",
            "wrong_demo_with_delta",
        ]
        err = {m: torch.zeros(11, device=device, dtype=torch.float64) for m in modes}

        c_it2 = bank.iter_val_batches(
            "cross_a", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        w_it2 = bank.iter_val_batches(
            "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        cur = 0
        for (dm_c, pm_c, q_c, z_self, a_c, _), (dm_w, pm_w, q_w, _, a_w, _) in zip(c_it2, w_it2):
            B = len(dm_c)
            z0 = local_z0[cur : cur + B].to(device, non_blocking=True)
            cur += B
            with amp_ctx:
                pred_self = policy.teacher.sample_from_z_self(z_self, init_noise=z0)
                z_cross, h_q = policy.read_latent_from_memory(dm_c, pm_c, q_c)
                z_adapted, _ = policy.adapt_latent(z_cross, h_q)
                pred_c_nd = policy.teacher.sample_from_z_self(z_cross, init_noise=z0)
                pred_c_d = policy.teacher.sample_from_z_self(z_adapted, init_noise=z0)

                z_wrong, _ = policy.read_latent_from_memory(dm_w, pm_w, q_w)
                z_wrong_ad, _ = policy.adapt_latent(z_wrong, h_q)
                pred_w_nd = policy.teacher.sample_from_z_self(z_wrong, init_noise=z0)
                pred_w_d = policy.teacher.sample_from_z_self(z_wrong_ad, init_noise=z0)

            err["self_teacher"] += _action_error_sums(pred_self, a_c)
            err["cross_no_delta"] += _action_error_sums(pred_c_nd, a_c)
            err["cross_with_delta"] += _action_error_sums(pred_c_d, a_c)
            err["wrong_demo_no_delta"] += _action_error_sums(pred_w_nd, a_w)
            err["wrong_demo_with_delta"] += _action_error_sums(pred_w_d, a_w)

        ddim_out: Dict[str, Any] = {
            m: _unpack_action_errors(_all_reduce_sum(err[m])) for m in modes
        }
        ddim_out["delta_gain_action10_mse"] = (
            ddim_out["cross_no_delta"]["action10_mse"]
            - ddim_out["cross_with_delta"]["action10_mse"]
        )
        ddim_out["delta_gain_action7_mae"] = (
            ddim_out["cross_no_delta"]["action7_mae"]
            - ddim_out["cross_with_delta"]["action7_mae"]
        )
        ddim_out["demo_sensitivity_action10_mse"] = (
            ddim_out["wrong_demo_with_delta"]["action10_mse"]
            - ddim_out["cross_with_delta"]["action10_mse"]
        )
        ddim_out["demo_sensitivity_action7_mae"] = (
            ddim_out["wrong_demo_with_delta"]["action7_mae"]
            - ddim_out["cross_with_delta"]["action7_mae"]
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

            # [count, cos_d1_d2_nodelta, cos_d1_d2_adapted, mse_d1_d2_nodelta, mse_d1_d2_adapted,
            #  lz_d1, lz_d2]
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
                    zc1, hq = policy.read_latent_from_memory(dm1, pm1, q1)
                    zc2, _ = policy.read_latent_from_memory(dm2, pm2, q2)
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

        run_name = args.run_name if args.run_name else "stage2_latent_transfer"
        run_dir = args.output_dir / run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        seed_everything(args.seed)
        bank = Stage2TransferBank(
            clip_cache_dir=args.clip_cache_dir,
            z_self_cache_dir=args.z_self_cache_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            z_self_on_device=(device.type == "cuda" and not args.z_self_on_cpu),
            self_ratio=args.self_ratio,
            seed=args.seed,
        )

        teacher = load_frozen_stage1_teacher(args.stage1_ckpt, device=device)
        seed_everything(args.seed)
        policy = E6Stage2Policy(
            teacher,
            latent_scale_s_z=bank.latent_scale_s_z,
            residual_layers=args.residual_layers,
            residual_dropout=args.residual_dropout,
        ).to(device)

        # Verify frozen UNet -> ResidualTransformer gradient flow explicitly on rank 0 before training
        if rank == 0 and args.check_unet_grad:
            policy.train()
            dm_g, pm_g, q_g, zs_g, a_g, is_s_g = next(
                bank.iter_train_batches(1, min(8, per_rank_batch), max_samples=16)
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out_g = policy(
                    dm_g, pm_g, q_g, zs_g, a_g, is_s_g, lambda_z=0.0, lambda_a=1.0
                )
            out_g["loss"].backward()
            out_proj_gn = float(policy.residual_transformer.out_proj.weight.grad.float().norm().item())
            teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
            policy.zero_grad(set_to_none=True)
            print(
                f"[E6 Stage2 Grad Check] frozen-UNet->Residual out_proj grad_norm={out_proj_gn:.6f}, "
                f"teacher_any_grad={teacher_has_grad}",
                flush=True,
            )
            if out_proj_gn <= 0.0 or not math.isfinite(out_proj_gn) or teacher_has_grad:
                raise RuntimeError(
                    f"Frozen UNet gradient check failed: out_proj_gn={out_proj_gn}, teacher_has_grad={teacher_has_grad}"
                )
        if use_ddp:
            dist.barrier()

        optimizer = torch.optim.AdamW(
            policy.get_optim_groups(lr=args.lr, weight_decay=args.weight_decay),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

        epoch_len = (
            min(args.max_train_samples, bank.num_train_samples())
            if args.max_train_samples is not None
            else bank.num_train_samples()
        )
        steps_per_epoch = math.ceil(epoch_len / args.global_batch)
        total_steps = steps_per_epoch * args.epochs
        lr_scheduler = get_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            lr=args.lr,
            min_lr=args.min_lr,
        )
        ema = (
            EMAModel(
                policy.residual_transformer.parameters(),
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
            "stage": "e6_stage2_latent_transfer",
            "run_name": run_name,
            "smoke": bool(args.smoke),
            "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
            "clip_cache_dir": str(Path(args.clip_cache_dir).resolve()),
            "z_self_cache_dir": str(Path(args.z_self_cache_dir).resolve()),
            "latent_scale_s_z": bank.latent_scale_s_z,
            "lambda_a": args.lambda_a,
            "lambda_z": args.lambda_z,
            "self_ratio": args.self_ratio,
            "cross_ratio": round(1.0 - args.self_ratio, 6),
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
            "lr": args.lr,
            "min_lr": args.min_lr,
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
            policy.residual_transformer.load_state_dict(
                ck["residual_transformer"], strict=True
            )
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
                    f"[E6 Stage2] Resumed from epoch {ck['epoch']} "
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
                f"[E6 Stage2] run={run_name} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={args.warmup_steps}) | "
                f"lr={args.lr:.2e}->{args.min_lr:.2e} | lambda_a={args.lambda_a} lambda_z={args.lambda_z} "
                f"(s_z={bank.latent_scale_s_z:.4f}) | self_ratio={args.self_ratio:.3f} | "
                f"trainable={ps['total_trainable']:,} (teacher_frozen={ps['stage1_teacher_frozen']:,})",
                flush=True,
            )

        # Epoch-0 reference validation (exact Cross No-Delta since out_proj is zero-initialized)
        init_val = None
        if start_epoch == 1:
            init_val = evaluate_stage2_validation(
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
                    f"[E6 Stage2 Init Val] diff self={dl['self_teacher']:.4f} "
                    f"cross_nd={dl['cross_no_delta']:.4f} cross_d={dl['cross_with_delta']:.4f} | "
                    f"L_z={lt['cross_with_delta_L_z']:.4f} cos={lt['cross_with_delta_cos']:.4f} | "
                    f"DDIM mse10 self={dm['self_teacher']['action10_mse']:.4f} "
                    f"cross_nd={dm['cross_no_delta']['action10_mse']:.4f} "
                    f"cross_d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"wrong={dm['wrong_demo_with_delta']['action10_mse']:.4f}",
                    flush=True,
                )

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            # Keep teacher strictly in eval mode on every rank
            policy.teacher.eval()

            # [n_tot, loss_sum, diff_sum, lz_sum, mse_raw_sum, cos_sum,
            #  cross_dz_sq, n_cross, self_dz_sq, n_self, gnorm_sum, n_steps]
            tr_acc = torch.zeros(12, device=device, dtype=torch.float64)

            for (
                demo_mem,
                pad_mask,
                query_rgb,
                z_self,
                actions_10d,
                is_self,
            ) in bank.iter_train_batches(
                epoch,
                per_rank_batch,
                rank=rank,
                world_size=world,
                max_samples=args.max_train_samples,
            ):
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(
                        demo_mem,
                        pad_mask,
                        query_rgb,
                        z_self,
                        actions_10d,
                        is_self,
                        lambda_z=args.lambda_z,
                        lambda_a=args.lambda_a,
                    )
                out["loss"].backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        policy.residual_transformer.parameters(), args.clip_grad_norm
                    ).item()
                )
                optimizer.step()
                lr_scheduler.step()
                if ema is not None:
                    ema.step(policy.residual_transformer.parameters())
                global_step += 1

                B = float(len(demo_mem))
                ns = float(out["n_self"].item())
                nc = float(out["n_cross"].item())
                tr_acc[0] += B
                tr_acc[1] += float(out["loss"].item()) * B
                tr_acc[2] += float(out["diff_loss"].item()) * B
                tr_acc[3] += float(out["latent_loss_norm"].item()) * B
                tr_acc[4] += float(out["latent_mse_raw"].item()) * B
                tr_acc[5] += float(out["cos_adapted_self"].item()) * B
                tr_acc[6] += (float(out["cross_delta_rms"].item()) ** 2) * nc
                tr_acc[7] += nc
                tr_acc[8] += (float(out["self_delta_rms"].item()) ** 2) * ns
                tr_acc[9] += ns
                tr_acc[10] += grad_norm
                tr_acc[11] += 1.0

            tr_acc = _all_reduce_sum(tr_acc)
            t_train = time.monotonic() - t_ep

            n_tot = max(float(tr_acc[0].item()), 1.0)
            n_cross_tot = max(float(tr_acc[7].item()), 1.0)
            n_self_tot = max(float(tr_acc[9].item()), 1.0)
            n_steps_tot = max(float(tr_acc[11].item()), 1.0)
            train_metrics = {
                "samples": int(round(n_tot)),
                "self_fraction": float(tr_acc[9].item() / n_tot),
                "loss": float(tr_acc[1].item() / n_tot),
                "diff_loss": float(tr_acc[2].item() / n_tot),
                "L_z": float(tr_acc[3].item() / n_tot),
                "latent_mse_raw": float(tr_acc[4].item() / n_tot),
                "cos_adapted_self": float(tr_acc[5].item() / n_tot),
                "cross_delta_rms": math.sqrt(float(tr_acc[6].item() / n_cross_tot)),
                "self_delta_rms": math.sqrt(float(tr_acc[8].item() / n_self_tot)),
                "grad_norm_mean": float(tr_acc[10].item() / n_steps_tot),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "ema_decay": float(ema.cur_decay_value or 0.0) if ema is not None else None,
                "train_seconds": round(t_train, 2),
                "steps_per_sec": round((n_steps_tot / world) / max(t_train, 1e-6), 2),
            }

            # Evaluate EMA weights (or live weights if --no-ema)
            t_val0 = time.monotonic()
            if ema is not None:
                ema.store(policy.residual_transformer.parameters())
                ema.copy_to(policy.residual_transformer.parameters())

            val_metrics = evaluate_stage2_validation(
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

            # Best checkpoint selected strictly by Validation Cross DDIM Action MSE (action10_mse)
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

            # Save best checkpoint (with EMA weights if enabled) BEFORE restoring live weights
            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "stage": "e6_stage2_latent_transfer",
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_cross_ddim_mse10": best_val_mse,
                        "latent_scale_s_z": bank.latent_scale_s_z,
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
                ema.restore(policy.residual_transformer.parameters())

            if rank == 0:
                if epoch % args.save_last_every == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        {
                            "stage": "e6_stage2_latent_transfer",
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_cross_ddim_mse10": best_val_mse,
                            "latent_scale_s_z": bank.latent_scale_s_z,
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
                        "stage": "e6_stage2_latent_transfer",
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
                print(
                    f"[E6 Stage2 Ep {epoch:02d}/{args.epochs}] "
                    f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, "
                    f"Lz={train_metrics['L_z']:.4f}) | "
                    f"val diff c+d={dl['cross_with_delta']:.4f} (nd={dl['cross_no_delta']:.4f}, "
                    f"self={dl['self_teacher']:.4f}, wrong={dl['wrong_demo_with_delta']:.4f}) | "
                    f"Lz={lt['cross_with_delta_L_z']:.4f} cos={lt['cross_with_delta_cos']:.4f} "
                    f"dz_rms={lt['cross_delta_rms']:.3f} ({100*lt['cross_delta_rel_rms']:.1f}%) | "
                    f"DDIM mse10 c+d={dm['cross_with_delta']['action10_mse']:.4f} "
                    f"(nd={dm['cross_no_delta']['action10_mse']:.4f}, "
                    f"self={dm['self_teacher']['action10_mse']:.4f}, "
                    f"wrong={dm['wrong_demo_with_delta']['action10_mse']:.4f}) "
                    f"mae7={dm['cross_with_delta']['action7_mae']:.4f} | "
                    f"lr={train_metrics['lr']:.2e} t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s){'*' if is_best else ''}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Final reloaded best.pt FULL validation across ALL valid val cross samples & multi-demo pairs
        reloaded, best_payload = load_stage2_policy(
            best_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        final_eval = evaluate_stage2_validation(
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
                "stage": "e6_stage2_latent_transfer",
                "run_name": run_name,
                "best_epoch": best_payload["epoch"],
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_cross_ddim_mse10": best_payload["best_val_cross_ddim_mse10"],
                "parameter_summary": config["parameter_summary"],
                "final_eval": final_eval,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[E6 Stage2 FINAL BEST EVAL (ep={best_payload['epoch']})] "
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "stage2")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=0.5)
    parser.add_argument("--self-ratio", type=float, default=0.025)
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
    parser.add_argument("--no-check-unet-grad", dest="check_unet_grad", action="store_false")
    parser.set_defaults(check_unet_grad=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
