"""Training, validation, smoke test, and temporal-distinctiveness audit for E9 Stage 1
(Local Behavior Encoder, CLIP, RGB-only).

Strictly follows E6 Stage 1 data split, action window, training budget, hyperparameters,
and pure Action Diffusion Loss (`L = L_diff`), training both LocalBehaviorEncoder and
ConditionalUnet1D from scratch and selecting Best strictly on the original val split.
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
from torch.nn.parallel import DistributedDataParallel

from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
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

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    DEFAULT_DENSE_CACHE_DIR,
    DEFAULT_STAGE1_LOCAL_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_HEADS,
    OUTPUT_ROOT,
)
from .data import LocalClipFeatureBank
from .stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher


@torch.inference_mode()
def evaluate_timestep_distinctiveness(
    policy: E9Stage1LocalPolicy,
    bank: LocalClipFeatureBank,
    *,
    batch_size: int = 128,
    max_pairs: Optional[int] = 512,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Check whether distinct local clips `V_{t_a:t_a+16}` vs `V_{t_b:t_b+16}` on the SAME
    val Demo produce distinguishable behavior latents `b_t` and timestep-accurate actions."""
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    pairs = bank.val_timestep_pairs
    n_pairs = len(pairs) if max_pairs is None else min(int(max_pairs), len(pairs))
    if n_pairs == 0:
        return {"num_pairs": 0}

    sel_pairs = pairs[:n_pairs]
    local_pair_idx = np.arange(n_pairs)[rank::world]

    gen = torch.Generator(device="cpu").manual_seed(val_seed + 2026)
    all_z0 = torch.randn(
        (n_pairs, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32
    )
    local_z0 = all_z0[local_pair_idx]

    # [count, frame_gap_sum, z_cos_sum, z_l2_sum, z_rel_l2_sum]
    stat_acc = torch.zeros(5, device=device, dtype=torch.float64)
    pred_pair_err = torch.zeros(11, device=device, dtype=torch.float64)
    gt_pair_err = torch.zeros(11, device=device, dtype=torch.float64)
    matched_err = torch.zeros(11, device=device, dtype=torch.float64)
    mismatched_err = torch.zeros(11, device=device, dtype=torch.float64)

    for start in range(0, len(local_pair_idx), batch_size):
        end = min(start + batch_size, len(local_pair_idx))
        p_chunk = sel_pairs[local_pair_idx[start:end]]
        ia = p_chunk[:, 0]
        ib = p_chunk[:, 1]
        z0 = local_z0[start:end].to(device, non_blocking=True)

        eps_a = bank.val_episodes[ia]
        eps_b = bank.val_episodes[ib]
        if not np.array_equal(eps_a, eps_b):
            raise RuntimeError("Timestep distinctiveness pair spans different episodes")
        fa = bank.val_frames[ia]
        fb = bank.val_frames[ib]
        clip_a, mask_a, act_a = bank.gather_batch(eps_a, fa)
        clip_b, mask_b, act_b = bank.gather_batch(eps_b, fb)

        with amp_ctx:
            out_a = policy.predict_action(clip_a, mask_a, init_noise=z0)
            out_b = policy.predict_action(clip_b, mask_b, init_noise=z0)

        za = out_a["b_t"].float().reshape(len(ia), -1)
        zb = out_b["b_t"].float().reshape(len(ib), -1)
        z_cos = torch.nn.functional.cosine_similarity(za, zb, dim=-1)
        z_l2 = (za - zb).norm(dim=-1)
        z_rms_pair = 0.5 * (za.norm(dim=-1) + zb.norm(dim=-1)).clamp_min(1e-6)
        z_rel_l2 = z_l2 / z_rms_pair

        stat_acc[0] += float(len(ia))
        stat_acc[1] += float(np.abs(fb - fa).sum())
        stat_acc[2] += z_cos.double().sum()
        stat_acc[3] += z_l2.double().sum()
        stat_acc[4] += z_rel_l2.double().sum()

        pred_pair_err += _action_error_sums(out_a["action_10d"], out_b["action_10d"])
        gt_pair_err += _action_error_sums(act_a, act_b)
        matched_err += _action_error_sums(out_a["action_10d"], act_a)
        matched_err += _action_error_sums(out_b["action_10d"], act_b)
        mismatched_err += _action_error_sums(out_a["action_10d"], act_b)
        mismatched_err += _action_error_sums(out_b["action_10d"], act_a)

    stat_acc = _all_reduce_sum(stat_acc)
    pred_pair = _unpack_action_errors(_all_reduce_sum(pred_pair_err))
    gt_pair = _unpack_action_errors(_all_reduce_sum(gt_pair_err))
    matched = _unpack_action_errors(_all_reduce_sum(matched_err))
    mismatched = _unpack_action_errors(_all_reduce_sum(mismatched_err))

    n_tot = max(float(stat_acc[0].item()), 1.0)
    return {
        "num_pairs": int(round(n_tot)),
        "mean_frame_gap": float(stat_acc[1].item() / n_tot),
        "latent_cosine_sim": float(stat_acc[2].item() / n_tot),
        "latent_l2_dist": float(stat_acc[3].item() / n_tot),
        "latent_rel_l2_dist": float(stat_acc[4].item() / n_tot),
        "pred_pair_action10_mse": pred_pair["action10_mse"],
        "pred_pair_action7_mae": pred_pair["action7_mae"],
        "gt_pair_action10_mse": gt_pair["action10_mse"],
        "gt_pair_action7_mae": gt_pair["action7_mae"],
        "matched_action10_mse": matched["action10_mse"],
        "matched_action7_mae": matched["action7_mae"],
        "mismatched_same_demo_action10_mse": mismatched["action10_mse"],
        "mismatched_same_demo_action7_mae": mismatched["action7_mae"],
        "mismatch_to_match_mse10_ratio": (
            mismatched["action10_mse"] / max(matched["action10_mse"], 1e-12)
        ),
    }


@torch.inference_mode()
def evaluate_stage1_local_validation(
    policy: E9Stage1LocalPolicy,
    bank: LocalClipFeatureBank,
    *,
    batch_size: int = 128,
    max_val_samples: Optional[int] = None,
    run_ddim: bool = False,
    ddim_max_samples: Optional[int] = 512,
    run_timestep_check: bool = False,
    timestep_max_pairs: Optional[int] = 512,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Distributed validation on the exact original E6 Stage 1 validation split (6,248 starts)."""
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    n_val = (
        bank.num_val_samples()
        if max_val_samples is None
        else min(int(max_val_samples), bank.num_val_samples())
    )
    gen = torch.Generator(device="cpu").manual_seed(val_seed)
    all_noise = torch.randn(
        (n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32
    )
    all_pertub = torch.randn(
        (n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32
    )
    all_t = torch.randint(
        0,
        policy.noise_scheduler.config.num_train_timesteps,
        (n_val,),
        generator=gen,
        dtype=torch.long,
    )

    local_idx = np.arange(n_val)[rank::world]
    local_noise = all_noise[local_idx]
    local_pertub = all_pertub[local_idx]
    local_t = all_t[local_idx]

    # [count, self_diff_sum, wrong_diff_sum, self_b_sq_sum, wrong_b_sq_sum,
    #  tail_count, tail_diff_sum, full_count, full_diff_sum]
    acc = torch.zeros(9, device=device, dtype=torch.float64)

    self_iter = bank.iter_val_batches(
        "self", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    wrong_iter = bank.iter_val_batches(
        "wrong_clip", batch_size, rank=rank, world_size=world, max_samples=n_val
    )

    cursor = 0
    for (c_s, m_s, a_s), (c_w, m_w, a_w) in zip(self_iter, wrong_iter):
        B = len(c_s)
        noise = local_noise[cursor : cursor + B].to(device, non_blocking=True)
        pertub = local_pertub[cursor : cursor + B].to(device, non_blocking=True)
        t = local_t[cursor : cursor + B].to(device, non_blocking=True)
        cursor += B

        with amp_ctx:
            b_s = policy.encode_local_behavior(c_s, m_s)
            loss_s = policy.diffusion_loss_from_z_self(
                b_s, a_s, noise=noise, pertub_noise=pertub, timesteps=t
            )
            b_w = policy.encode_local_behavior(c_w, m_w)
            loss_w = policy.diffusion_loss_from_z_self(
                b_w, a_w, noise=noise, pertub_noise=pertub, timesteps=t
            )

        is_tail = m_s.any(dim=1)
        is_full = ~is_tail

        acc[0] += float(B)
        acc[1] += loss_s.double().sum()
        acc[2] += loss_w.double().sum()
        acc[3] += b_s.float().square().mean(dim=(1, 2)).double().sum()
        acc[4] += b_w.float().square().mean(dim=(1, 2)).double().sum()
        acc[5] += is_tail.double().sum()
        acc[6] += (loss_s.double() * is_tail.double()).sum()
        acc[7] += is_full.double().sum()
        acc[8] += (loss_s.double() * is_full.double()).sum()

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)
    self_diff = float(acc[1].item() / total)
    wrong_diff = float(acc[2].item() / total)
    n_tail = float(acc[5].item())
    n_full = float(acc[7].item())

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "val_tail_samples": int(round(n_tail)),
        "val_self_diff_loss": self_diff,
        "val_tail_diff_loss": float(acc[6].item() / max(n_tail, 1.0)) if n_tail > 0 else None,
        "val_full_window_diff_loss": float(acc[8].item() / max(n_full, 1.0)) if n_full > 0 else None,
        "val_wrong_clip_diff_loss": wrong_diff,
        "val_clip_sensitivity_diff_loss": wrong_diff - self_diff,
        "val_self_b_rms": math.sqrt(float(acc[3].item() / total)),
        "val_wrong_b_rms": math.sqrt(float(acc[4].item() / total)),
    }

    if run_ddim:
        n_ddim = n_val if ddim_max_samples is None else min(int(ddim_max_samples), n_val)
        gen_ddim = torch.Generator(device="cpu").manual_seed(val_seed + 777)
        init_noise_all = torch.randn(
            (n_ddim, ACTION_HORIZON, ACTION_DIM_10D),
            generator=gen_ddim,
            dtype=torch.float32,
        )
        local_ddim_noise = init_noise_all[np.arange(n_ddim)[rank::world]]

        self_err = torch.zeros(11, device=device, dtype=torch.float64)
        wrong_err = torch.zeros(11, device=device, dtype=torch.float64)

        s_it = bank.iter_val_batches(
            "self", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        w_it = bank.iter_val_batches(
            "wrong_clip", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        cur = 0
        for (c_s, m_s, a_s), (c_w, m_w, a_w) in zip(s_it, w_it):
            B = len(c_s)
            z0 = local_ddim_noise[cur : cur + B].to(device, non_blocking=True)
            cur += B
            with amp_ctx:
                out_s = policy.predict_action(c_s, m_s, init_noise=z0)
                out_w = policy.predict_action(c_w, m_w, init_noise=z0)
            self_err += _action_error_sums(out_s["action_10d"], a_s)
            wrong_err += _action_error_sums(out_w["action_10d"], a_w)

        self_ddim = _unpack_action_errors(_all_reduce_sum(self_err))
        wrong_ddim = _unpack_action_errors(_all_reduce_sum(wrong_err))
        result["ddim"] = {
            "self": self_ddim,
            "wrong_clip": wrong_ddim,
            "clip_sensitivity_action10_mse": (
                wrong_ddim["action10_mse"] - self_ddim["action10_mse"]
            ),
            "clip_sensitivity_action7_mae": (
                wrong_ddim["action7_mae"] - self_ddim["action7_mae"]
            ),
        }

    if run_timestep_check:
        result["timestep_distinctiveness"] = evaluate_timestep_distinctiveness(
            policy,
            bank,
            batch_size=batch_size,
            max_pairs=timestep_max_pairs,
            val_seed=val_seed,
            use_bf16=use_bf16,
        )

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

        # Smoke preset overrides for rapid end-to-end verification
        if args.smoke:
            if args.output_dir == DEFAULT_STAGE1_LOCAL_DIR:
                args.output_dir = OUTPUT_ROOT / "stage1_local_smoke"
            if args.epochs == 8:
                args.epochs = 2
            if args.max_train_samples is None:
                args.max_train_samples = args.global_batch * 100
            if args.max_val_samples is None:
                args.max_val_samples = 512
            if args.sample_count == 512:
                args.sample_count = 256
            if args.timestep_pairs == 512:
                args.timestep_pairs = 128
            args.final_ddim_samples = args.sample_count
            args.final_timestep_pairs = args.timestep_pairs
            args.sample_every = 1
            args.save_last_every = 1

        run_dir = Path(args.output_dir)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        seed_everything(args.seed)
        bank = LocalClipFeatureBank(
            e5_cache_dir=args.e5_cache_dir,
            dense_cache_dir=args.dense_cache_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            seed=args.seed,
        )

        # Strictly train BOTH LocalBehaviorEncoder and ConditionalUnet1D FROM SCRATCH
        seed_everything(args.seed)
        policy = E9Stage1LocalPolicy(
            embed_dim=EMBED_DIM,
            window_len=LOCAL_WINDOW_LEN,
            num_behavior_tokens=NUM_BEHAVIOR_TOKENS,
            encoder_layers=args.encoder_layers,
            encoder_heads=NUM_HEADS,
            encoder_ffn_dim=4 * EMBED_DIM,
            dropout=args.dropout,
            action_normalizer=bank.action_normalizer,
        ).to(device)

        lr = float(args.lr)
        optimizer = torch.optim.AdamW(
            policy.get_optim_groups(lr=lr, weight_decay=args.weight_decay),
            lr=lr,
            betas=(0.95, 0.999),
            weight_decay=args.weight_decay,
        )

        epoch_len = (
            min(args.max_train_samples, bank.num_train_samples())
            if args.max_train_samples is not None
            else bank.num_train_samples()
        )
        steps_per_epoch = math.ceil(epoch_len / args.global_batch)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        ema = EMAModel(
            policy.parameters(),
            decay=args.ema_decay,
            min_decay=0.0,
            update_after_step=0,
            use_ema_warmup=False,
            inv_gamma=1.0,
            power=0.75,
            foreach=True,
        ).to(device)

        config = {
            "stage": "e9_stage1_local_behavior_encoder",
            "smoke": bool(args.smoke),
            "init_mode": "from_scratch",
            "loss_type": "pure_action_diffusion_only",
            "embed_dim": EMBED_DIM,
            "window_len": LOCAL_WINDOW_LEN,
            "num_behavior_tokens": NUM_BEHAVIOR_TOKENS,
            "encoder_layers": args.encoder_layers,
            "encoder_heads": NUM_HEADS,
            "encoder_ffn_dim": 4 * EMBED_DIM,
            "dropout": args.dropout,
            "global_cond_dim": policy.global_cond_dim,
            "samples_per_epoch": epoch_len,
            "total_cached_train_starts": bank.num_train_samples(),
            "total_cached_val_starts": bank.num_val_samples(),
            "val_timestep_pairs_available": int(len(bank.val_timestep_pairs)),
            "seed": args.seed,
            "val_seed": args.val_seed,
            "epochs": args.epochs,
            "world_size": world,
            "global_batch": args.global_batch,
            "per_rank_batch": per_rank_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "betas": [0.95, 0.999],
            "lr_scheduler": "cosine",
            "lr_warmup_proportion": args.lr_warmup_proportion,
            "warmup_steps": warmup_steps,
            "clip_grad_norm": args.clip_grad_norm,
            "mixed_precision": "bf16",
            "use_ema": True,
            "ema_decay": args.ema_decay,
            "best_selection_criterion": "original_val_split_self_diff_loss_only",
            "e5_cache_dir": str(args.e5_cache_dir),
            "dense_cache_dir": str(args.dense_cache_dir),
            "dense_cache_meta": bank.dense_meta,
            "parameter_summary": policy.parameter_summary(),
        }

        last_ckpt = run_dir / "last.pt"
        best_ckpt = run_dir / "best.pt"
        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        start_epoch = 1
        global_step = 0

        if args.resume and last_ckpt.is_file():
            ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            policy.load_state_dict(ck["model"], strict=True)
            ema.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["optimizer"])
            lr_scheduler.load_state_dict(ck["lr_scheduler"])
            history = ck["history"]
            best_val_loss = ck["best_val_loss"]
            best_epoch = ck["best_epoch"]
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
            if rank == 0:
                print(
                    f"[E9 Stage1 Local] Resumed from epoch {ck['epoch']} "
                    f"(best_epoch={best_epoch}, best_val_self_diff={best_val_loss:.5f})",
                    flush=True,
                )

        ddp_model: torch.nn.Module = (
            DistributedDataParallel(policy, device_ids=[device.index])
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
                f"[E9 Stage1 Local] out={run_dir} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={warmup_steps}) | lr={lr:.2e} | "
                f"layers={args.encoder_layers} | global_cond_dim={policy.global_cond_dim} | "
                f"trainable={ps['policy_trainable']:,} (from scratch: "
                f"local_enc={ps['modules']['local_encoder']['trainable']:,}, "
                f"unet={ps['modules']['action_decoder']['trainable']:,})",
                flush=True,
            )

        # Epoch-0 reference validation before training updates
        init_val = None
        if start_epoch == 1:
            init_val = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=True,
                ddim_max_samples=args.sample_count,
                run_timestep_check=True,
                timestep_max_pairs=args.timestep_pairs,
                val_seed=args.val_seed,
            )
            if rank == 0:
                atomic_write_json(init_val, run_dir / "init_val.json")
                td = init_val["timestep_distinctiveness"]
                print(
                    f"[E9 Stage1 Local Init Val] self_diff={init_val['val_self_diff_loss']:.4f} | "
                    f"wrong_diff={init_val['val_wrong_clip_diff_loss']:.4f} | "
                    f"ddim_self_mse10={init_val['ddim']['self']['action10_mse']:.4f} | "
                    f"ddim_self_mae7={init_val['ddim']['self']['action7_mae']:.4f} | "
                    f"ts_mismatch_ratio={td['mismatch_to_match_mse10_ratio']:.2f}x",
                    flush=True,
                )

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            # [n_total, diff_sum, b_sq_sum, gnorm_sum, n_steps]
            tr_acc = torch.zeros(5, device=device, dtype=torch.float64)

            for clip_rgb, clip_pad_mask, actions_10d in bank.iter_train_batches(
                epoch,
                per_rank_batch,
                rank=rank,
                world_size=world,
                max_samples=args.max_train_samples,
            ):
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(clip_rgb, clip_pad_mask, actions_10d)
                out["loss"].backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(), args.clip_grad_norm
                    ).item()
                )
                optimizer.step()
                lr_scheduler.step()
                ema.step(policy.parameters())
                global_step += 1

                B = float(len(clip_rgb))
                tr_acc[0] += B
                tr_acc[1] += float(out["diff_loss"].item()) * B
                tr_acc[2] += (float(out["b_rms"].item()) ** 2) * B
                tr_acc[3] += grad_norm
                tr_acc[4] += 1.0

            tr_acc = _all_reduce_sum(tr_acc)
            t_train = time.monotonic() - t_ep

            n_tot = max(float(tr_acc[0].item()), 1.0)
            n_steps_tot = max(float(tr_acc[4].item()), 1.0)
            train_metrics = {
                "samples": int(round(n_tot)),
                "diff_loss": float(tr_acc[1].item() / n_tot),
                "b_rms": math.sqrt(float(tr_acc[2].item() / n_tot)),
                "grad_norm_mean": float(tr_acc[3].item() / n_steps_tot),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "ema_decay": float(ema.cur_decay_value or 0.0),
                "train_seconds": round(t_train, 2),
                "steps_per_sec": round((n_steps_tot / world) / max(t_train, 1e-6), 2),
            }

            # Evaluate EMA weights on original validation split (exact E6 Stage 1 protocol)
            t_val0 = time.monotonic()
            ema.store(policy.parameters())
            ema.copy_to(policy.parameters())
            do_ddim = (
                epoch == 1
                or epoch == args.epochs
                or (args.sample_every > 0 and epoch % args.sample_every == 0)
            )
            val_metrics = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=do_ddim,
                ddim_max_samples=args.sample_count,
                run_timestep_check=do_ddim,
                timestep_max_pairs=args.timestep_pairs,
                val_seed=args.val_seed,
            )
            val_metrics["val_seconds"] = round(time.monotonic() - t_val0, 2)

            cur_monitor = val_metrics["val_self_diff_loss"]
            is_best = cur_monitor < best_val_loss
            if is_best:
                best_val_loss = cur_monitor
                best_epoch = epoch

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_self_diff_loss": best_val_loss,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            # Save best EMA checkpoint BEFORE restoring live weights
            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "stage": "e9_stage1_local_behavior_encoder",
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_self_diff_loss": best_val_loss,
                        "model": {
                            k: v.detach().cpu() for k, v in policy.state_dict().items()
                        },
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    best_ckpt,
                )

            ema.restore(policy.parameters())

            if rank == 0:
                if epoch % args.save_last_every == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        {
                            "stage": "e9_stage1_local_behavior_encoder",
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "model": {
                                k: v.detach().cpu()
                                for k, v in policy.state_dict().items()
                            },
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "history": history,
                            "config": config,
                        },
                        last_ckpt,
                    )
                atomic_write_json(
                    {
                        "stage": "e9_stage1_local_behavior_encoder",
                        "best_epoch": best_epoch,
                        "best_val_self_diff_loss": best_val_loss,
                        "init_val": init_val,
                        "history": history,
                    },
                    run_dir / "metrics.json",
                )
                extra_str = ""
                if "ddim" in val_metrics:
                    dm = val_metrics["ddim"]
                    td = val_metrics["timestep_distinctiveness"]
                    extra_str = (
                        f" | DDIM self mse10={dm['self']['action10_mse']:.4f} "
                        f"mae7={dm['self']['action7_mae']:.4f} "
                        f"(wrong mse10={dm['wrong_clip']['action10_mse']:.4f}) | "
                        f"TS z_cos={td['latent_cosine_sim']:.3f} "
                        f"mismatch/match={td['mismatch_to_match_mse10_ratio']:.2f}x"
                    )
                print(
                    f"[E9 Stage1 Local Ep {epoch:02d}/{args.epochs}] "
                    f"tr_diff={train_metrics['diff_loss']:.4f} | "
                    f"val_self={val_metrics['val_self_diff_loss']:.4f} "
                    f"(wrong={val_metrics['val_wrong_clip_diff_loss']:.4f}, "
                    f"sens={val_metrics['val_clip_sensitivity_diff_loss']:+.4f}) | "
                    f"b_rms={val_metrics['val_self_b_rms']:.3f} | "
                    f"lr={train_metrics['lr']:.2e} "
                    f"t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s)"
                    f"{'*' if is_best else ''}{extra_str}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Final reloaded best.pt full validation + frozen Teacher reload sanity verification
        best_payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        teacher = load_frozen_e9_stage1_teacher(best_ckpt, device=device)
        final_eval = evaluate_stage1_local_validation(
            teacher,
            bank,
            batch_size=args.val_batch_size,
            max_val_samples=args.max_val_samples,
            run_ddim=True,
            ddim_max_samples=args.final_ddim_samples,
            run_timestep_check=True,
            timestep_max_pairs=args.final_timestep_pairs,
            val_seed=args.val_seed,
        )
        if rank == 0:
            final_report = {
                "stage": "e9_stage1_local_behavior_encoder",
                "best_epoch": best_payload["epoch"],
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_self_diff_loss": best_payload["best_val_self_diff_loss"],
                "frozen_teacher_trainable_params": sum(
                    p.numel() for p in teacher.parameters() if p.requires_grad
                ),
                "parameter_summary": config["parameter_summary"],
                "final_eval": final_eval,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[E9 Stage1 Local FINAL BEST EVAL (ep={best_payload['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE1_LOCAL_DIR)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Exact E6 Stage 1 training budget & hyperparameters
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--timestep-pairs", type=int, default=512)
    parser.add_argument("--final-ddim-samples", type=int, default=None)
    parser.add_argument("--final-timestep-pairs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--save-last-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
