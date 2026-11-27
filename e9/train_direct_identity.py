"""Experiment B: Training E9-Direct + Exact Self Identity from scratch.

Strictly preserves E9 Stage 1 Local Teacher (frozen), Global Soft Reader, Direct Transfer
architecture, original Cross/Self pairing schedule (`seed=42`, `val_seed=12345`, 97.5% Cross /
2.5% Self), AdamW (`lr=1e-4`, `betas=(0.9, 0.95)`, `wd=0.01`), Cosine LR (`warmup=500`,
`min_lr=1e-5`), `global_batch=256`, `BF16`, `EMA=0.999`, `grad_clip=1.0`, and `8 epochs / 42,912 steps`.

Total loss:
    L = L_diff + 0.5 * L_z + 0.1 * L_id
where:
    z_id = T_theta(z_self, h_q)           (NO Reader, NO UNet)
    L_id = MSE((z_id - z_self) / s_z)

Logs separate gradient contributions of Cross Loss (`L_diff + 0.5*L_z`) vs Identity Loss
(`0.1*L_id`) on Transfer parameters throughout training to verify Identity does not dominate.
Saves independently under `e9/outputs/direct_identity/train/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from e5.diffusion import EMAModel
from e5.train import (
    _all_reduce_sum,
    _rank,
    _world,
    atomic_torch_save,
    atomic_write_json,
    seed_everything,
)
from e7.train import compute_train_z_self_mean, get_cosine_schedule_with_min_ratio

from . import (
    DEFAULT_STAGE1_LOCAL_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    OUTPUT_ROOT,
)
from .direct_identity_model import E9DirectIdentityPolicy, build_e9_direct_identity_policy
from .stage1_local_model import load_frozen_e9_stage1_teacher
from .stage2_data import E9Stage2TransferBank
from .stage2_direct_model import load_e9_stage2_direct_policy
from .train_stage2_direct import evaluate_e9_stage2_validation

DEFAULT_DIRECT_IDENTITY_DIR = OUTPUT_ROOT / "direct_identity"


def _flatten_grads(
    grads: Tuple[Optional[torch.Tensor], ...],
    params: List[torch.nn.Parameter],
) -> torch.Tensor:
    flats = [
        (g.detach().float().reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        for g, p in zip(grads, params)
    ]
    return torch.cat(flats, dim=0)


def measure_gradient_contributions(
    policy: E9DirectIdentityPolicy,
    out: Dict[str, torch.Tensor],
    *,
    use_ddp: bool = False,
) -> Dict[str, float]:
    """Compute exact Transfer gradient norms & cosines for Cross (`L_diff + 0.5*L_z`),
    Diff (`L_diff`), Z (`0.5*L_z`), and Identity (`0.1*L_id`) terms, plus Reader grad norm.

    Uses `torch.autograd.grad(..., retain_graph=True)` before the actual optimizer `.backward()`,
    averaging gradient vectors across DDP ranks when `use_ddp=True`.
    """
    transfer_params = [p for p in policy.residual_transformer.parameters() if p.requires_grad]
    reader_params = [p for p in policy.reader.parameters() if p.requires_grad]

    g_cross_vec = _flatten_grads(
        torch.autograd.grad(
            out["cross_loss_tensor"], transfer_params, retain_graph=True, allow_unused=True
        ),
        transfer_params,
    )
    g_diff_vec = _flatten_grads(
        torch.autograd.grad(
            out["diff_term_tensor"], transfer_params, retain_graph=True, allow_unused=True
        ),
        transfer_params,
    )
    g_z_vec = _flatten_grads(
        torch.autograd.grad(
            out["z_term_tensor"], transfer_params, retain_graph=True, allow_unused=True
        ),
        transfer_params,
    )
    g_id_vec = _flatten_grads(
        torch.autograd.grad(
            out["id_term_tensor"], transfer_params, retain_graph=True, allow_unused=True
        ),
        transfer_params,
    )
    g_reader_cross_vec = _flatten_grads(
        torch.autograd.grad(
            out["cross_loss_tensor"], reader_params, retain_graph=True, allow_unused=True
        ),
        reader_params,
    )

    if use_ddp and dist.is_available() and dist.is_initialized():
        ws = float(dist.get_world_size())
        for vec in (g_cross_vec, g_diff_vec, g_z_vec, g_id_vec, g_reader_cross_vec):
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            vec.div_(ws)

    g_total_vec = g_cross_vec + g_id_vec
    eps = 1e-12

    gn_cross = float(g_cross_vec.norm().item())
    gn_diff = float(g_diff_vec.norm().item())
    gn_z = float(g_z_vec.norm().item())
    gn_id = float(g_id_vec.norm().item())
    gn_total = float(g_total_vec.norm().item())
    gn_reader = float(g_reader_cross_vec.norm().item())

    def _cos(a: torch.Tensor, b: torch.Tensor, na: float, nb: float) -> float:
        denom = na * nb
        if denom <= eps:
            return 0.0
        return float(torch.dot(a, b).item() / denom)

    return {
        "transfer_gn_cross": gn_cross,
        "transfer_gn_diff": gn_diff,
        "transfer_gn_z": gn_z,
        "transfer_gn_id": gn_id,
        "transfer_gn_total": gn_total,
        "reader_gn_cross": gn_reader,
        "grad_ratio_id_over_cross": gn_id / max(gn_cross, eps),
        "grad_share_id": gn_id / max(gn_cross + gn_id, eps),
        "grad_cos_cross_id": _cos(g_cross_vec, g_id_vec, gn_cross, gn_id),
        "grad_cos_diff_id": _cos(g_diff_vec, g_id_vec, gn_diff, gn_id),
        "grad_cos_z_id": _cos(g_z_vec, g_id_vec, gn_z, gn_id),
    }


def _enrich_val_with_identity(val_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Derive exact Self Identity validation `L_id = MSE((T(z_self, h_q) - z_self) / s_z)`
    directly from `latent.self_teacher_delta_rms` and `latent.s_z`."""
    s_z = float(val_metrics["latent"]["s_z"])
    delta_rms = float(val_metrics["latent"]["self_teacher_delta_rms"])
    val_metrics["latent"]["exact_self_identity_mse_raw"] = delta_rms * delta_rms
    val_metrics["latent"]["exact_self_identity_L_id"] = (delta_rms / max(s_z, 1e-12)) ** 2
    return val_metrics


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
            if args.output_dir == DEFAULT_DIRECT_IDENTITY_DIR / "train":
                args.output_dir = DEFAULT_DIRECT_IDENTITY_DIR / "smoke"
            if args.epochs == 8:
                args.epochs = 2
            if args.max_train_samples is None:
                args.max_train_samples = args.global_batch * 40
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
            if args.grad_log_every == 200:
                args.grad_log_every = 10

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

        s_z = float(bank.latent_scale_s_z)
        direct_latent_bias = compute_train_z_self_mean(bank)

        # Load frozen E9 Stage 1 Local Teacher ONLY; Reader & Transfer strictly from scratch!
        teacher = load_frozen_e9_stage1_teacher(args.stage1_ckpt, device=device)
        seed_everything(args.seed)
        policy = build_e9_direct_identity_policy(
            teacher,
            latent_scale_s_z=s_z,
            direct_latent_bias=direct_latent_bias,
            reader_attn_dim=args.reader_attn_dim,
            residual_layers=args.residual_layers,
            residual_dropout=args.residual_dropout,
            direct_out_proj_init_std=args.direct_out_proj_init_std,
        ).to(device)

        # Strict gradient, isolation & freeze audit on rank 0 before training
        if rank == 0 and args.check_grads:
            policy.train()
            hm_g, bm_g, mpm_g, q_g, zs_g, a_g, is_s_g = next(
                bank.iter_e7_train_batches(1, min(8, per_rank_batch), max_samples=16)
            )
            # Check 1: Pure L_diff flows to both Reader and Transfer, zero to Teacher
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                out_diff = policy(
                    hm_g, bm_g, mpm_g, q_g, zs_g, a_g, is_s_g,
                    lambda_a=1.0, lambda_z=0.0, lambda_id=0.0,
                )
            out_diff["loss"].backward()
            out_proj_gn = float(policy.residual_transformer.out_proj.weight.grad.float().norm().item())
            q_gn = float(policy.reader.q_proj.weight.grad.float().norm().item())
            k_gn = float(policy.reader.k_proj.weight.grad.float().norm().item())
            teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
            policy.zero_grad(set_to_none=True)

            # Check 2: Pure L_id flows ONLY to Transfer, STRICTLY ZERO to Reader & Teacher
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                out_id = policy(
                    hm_g, bm_g, mpm_g, q_g, zs_g, a_g, is_s_g,
                    lambda_a=0.0, lambda_z=0.0, lambda_id=args.lambda_id,
                )
            out_id["loss"].backward()
            id_out_proj_gn = float(policy.residual_transformer.out_proj.weight.grad.float().norm().item())
            id_reader_has_grad = any(p.grad is not None for p in policy.reader.parameters())
            id_teacher_has_grad = any(p.grad is not None for p in policy.teacher.parameters())
            policy.zero_grad(set_to_none=True)

            print(
                f"[ExpB Grad Audit] "
                f"pure L_diff: transfer_gn={out_proj_gn:.6f} reader_q_gn={q_gn:.4e} "
                f"reader_k_gn={k_gn:.4e} teacher_grad={teacher_has_grad} | "
                f"pure L_id: transfer_gn={id_out_proj_gn:.6f} "
                f"reader_grad={id_reader_has_grad} teacher_grad={id_teacher_has_grad}",
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
                or id_out_proj_gn <= 0.0
                or not math.isfinite(id_out_proj_gn)
                or id_reader_has_grad
                or id_teacher_has_grad
            ):
                raise RuntimeError("ExpB gradient / isolation / freeze audit FAILED!")
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
            "stage": "e9_exp_b_direct_exact_self_identity",
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
            "lambda_id": args.lambda_id,
            "grad_log_every": args.grad_log_every,
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
        grad_csv_path = run_dir / "grad_contributions.csv"
        history: List[Dict[str, Any]] = []
        grad_records: List[Dict[str, Any]] = []
        best_val_mse = float("inf")
        best_epoch = 0
        start_epoch = 1
        global_step = 0

        if args.resume and last_ckpt.is_file():
            ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            if ck.get("stage") != "e9_exp_b_direct_exact_self_identity":
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
            grad_records = ck.get("grad_records", [])
            best_val_mse = ck["best_val_cross_ddim_mse10"]
            best_epoch = ck["best_epoch"]
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
            if rank == 0:
                print(
                    f"[ExpB] Resumed from epoch {ck['epoch']} "
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
                f"[ExpB E9-Direct+Identity] out={run_dir} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={args.warmup_steps}) | "
                f"lr={args.reader_lr:.2e}->{args.reader_lr * args.min_lr_ratio:.2e} | "
                f"L=L_diff+{args.lambda_z}*L_z+{args.lambda_id}*L_id | "
                f"s_z={s_z:.6f} bias_rms={config['direct_latent_bias_rms']:.6f} | "
                f"trainable={ps['total_trainable']:,} (reader={ps['simple_reader_trainable']:,}, "
                f"transfer={ps['residual_transformer_trainable']:,}, teacher_frozen={ps['stage1_teacher_frozen']:,})",
                flush=True,
            )

        init_val = None
        if start_epoch == 1:
            init_val = _enrich_val_with_identity(
                evaluate_e9_stage2_validation(
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
            )
            if rank == 0:
                atomic_write_json(init_val, run_dir / "init_val.json")
                dl = init_val["diff_loss"]
                lt = init_val["latent"]
                dm = init_val["ddim"]
                at = init_val["reader_attention"]["cross"]
                print(
                    f"[ExpB Init Val] diff self={dl['self_teacher']:.4f} "
                    f"c_nd={dl['cross_no_delta']:.4f} c+d={dl['cross_with_delta']:.4f} | "
                    f"Lz={lt['cross_with_delta_L_z']:.4f} L_id={lt['exact_self_identity_L_id']:.4f} | "
                    f"attn H_n={at['normalized_entropy_mean']:.4f} max_w={at['max_weight_mean']:.4f} | "
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

            tr_acc = torch.zeros(23, device=device, dtype=torch.float64)
            ep_grad_probes: List[Dict[str, float]] = []

            for step_in_ep, (
                h_mem,
                b_mem,
                mem_pad_mask,
                query_rgb,
                z_self,
                actions_10d,
                is_self,
            ) in enumerate(
                bank.iter_e7_train_batches(
                    epoch,
                    per_rank_batch,
                    rank=rank,
                    world_size=world,
                    max_samples=args.max_train_samples,
                ),
                start=1,
            ):
                next_step = global_step + 1
                do_grad_probe = (
                    step_in_ep == 1
                    or step_in_ep == steps_per_epoch
                    or (args.grad_log_every > 0 and next_step % args.grad_log_every == 0)
                )

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
                        lambda_id=args.lambda_id,
                        return_loss_terms=do_grad_probe,
                    )

                if do_grad_probe:
                    gp = measure_gradient_contributions(policy, out, use_ddp=use_ddp)
                    gp["epoch"] = epoch
                    gp["step_in_epoch"] = step_in_ep
                    gp["global_step"] = next_step
                    gp["loss_total"] = float(out["loss"].item())
                    gp["loss_diff"] = float(out["diff_loss"].item())
                    gp["loss_z"] = float(out["latent_loss_norm"].item())
                    gp["loss_id"] = float(out["id_loss_norm"].item())
                    ep_grad_probes.append(gp)
                    if rank == 0:
                        grad_records.append(gp)

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
                tr_acc[19] += float(out["id_loss_norm"].item()) * B
                tr_acc[20] += float(out["cos_id_self"].item()) * B
                tr_acc[21] += (float(out["id_delta_rms"].item()) ** 2) * B
                tr_acc[22] += float(out["id_mse_raw"].item()) * B

            tr_acc = _all_reduce_sum(tr_acc)
            t_train = time.monotonic() - t_ep

            n_tot = max(float(tr_acc[0].item()), 1.0)
            n_cross_tot = max(float(tr_acc[9].item()), 1.0)
            n_self_tot = max(float(tr_acc[11].item()), 1.0)
            n_steps_tot = max(float(tr_acc[13].item()), 1.0)
            last_lrs = lr_scheduler.get_last_lr()

            def _mean_probe(key: str) -> Optional[float]:
                if not ep_grad_probes:
                    return None
                return float(sum(p[key] for p in ep_grad_probes) / len(ep_grad_probes))

            grad_contrib_summary = {
                "num_probes": len(ep_grad_probes),
                "transfer_gn_cross_mean": _mean_probe("transfer_gn_cross"),
                "transfer_gn_diff_mean": _mean_probe("transfer_gn_diff"),
                "transfer_gn_z_mean": _mean_probe("transfer_gn_z"),
                "transfer_gn_id_mean": _mean_probe("transfer_gn_id"),
                "transfer_gn_total_mean": _mean_probe("transfer_gn_total"),
                "reader_gn_cross_mean": _mean_probe("reader_gn_cross"),
                "grad_ratio_id_over_cross_mean": _mean_probe("grad_ratio_id_over_cross"),
                "grad_ratio_id_over_cross_max": (
                    max(p["grad_ratio_id_over_cross"] for p in ep_grad_probes)
                    if ep_grad_probes
                    else None
                ),
                "grad_share_id_mean": _mean_probe("grad_share_id"),
                "grad_cos_cross_id_mean": _mean_probe("grad_cos_cross_id"),
                "grad_cos_diff_id_mean": _mean_probe("grad_cos_diff_id"),
                "grad_cos_z_id_mean": _mean_probe("grad_cos_z_id"),
            }

            train_metrics = {
                "samples": int(round(n_tot)),
                "self_fraction": float(tr_acc[11].item() / n_tot),
                "loss": float(tr_acc[1].item() / n_tot),
                "diff_loss": float(tr_acc[2].item() / n_tot),
                "L_z": float(tr_acc[3].item() / n_tot),
                "L_id": float(tr_acc[19].item() / n_tot),
                "weighted_L_id": args.lambda_id * float(tr_acc[19].item() / n_tot),
                "id_mse_raw": float(tr_acc[22].item() / n_tot),
                "cos_id_self": float(tr_acc[20].item() / n_tot),
                "id_delta_rms": math.sqrt(float(tr_acc[21].item() / n_tot)),
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
                "grad_contributions": grad_contrib_summary,
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

            val_metrics = _enrich_val_with_identity(
                evaluate_e9_stage2_validation(
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
                        "stage": "e9_exp_b_direct_exact_self_identity",
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
                            "stage": "e9_exp_b_direct_exact_self_identity",
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
                            "grad_records": grad_records,
                            "config": config,
                        },
                        last_ckpt,
                    )
                if grad_records:
                    with grad_csv_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(grad_records[0].keys()))
                        writer.writeheader()
                        writer.writerows(grad_records)
                atomic_write_json(
                    {
                        "stage": "e9_exp_b_direct_exact_self_identity",
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
                gc = grad_contrib_summary
                print(
                    f"[ExpB Ep {epoch:02d}/{args.epochs}] "
                    f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, "
                    f"Lz={train_metrics['L_z']:.4f}, Lid={train_metrics['L_id']:.4f}) | "
                    f"grad ||g_id||/||g_cross||={gc['grad_ratio_id_over_cross_mean']:.3f} "
                    f"(share={100*gc['grad_share_id_mean']:.1f}%, cos={gc['grad_cos_cross_id_mean']:+.3f}) | "
                    f"val Lz nd={lt['cross_no_delta_L_z']:.4f}->c+d={lt['cross_with_delta_L_z']:.4f} "
                    f"val_Lid={lt['exact_self_identity_L_id']:.5f} | "
                    f"attn H_n={at['normalized_entropy_mean']:.3f} max_w={at['max_weight_mean']:.3f} | "
                    f"DDIM10 nd={dm['cross_no_delta']['action10_mse']:.4f}->"
                    f"c+d={dm['cross_with_delta']['action10_mse']:.5f} | "
                    f"t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s){'*' if is_best else ''}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Reload best.pt via standard E9 loader (verifying 100% compatibility!)
        reloaded, best_payload = load_e9_stage2_direct_policy(
            best_ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        final_eval = _enrich_val_with_identity(
            evaluate_e9_stage2_validation(
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
        )
        if rank == 0:
            final_report = {
                "stage": "e9_exp_b_direct_exact_self_identity",
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
                "lambda_id": config["lambda_id"],
                "parameter_summary": config["parameter_summary"],
                "final_eval": final_eval,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[ExpB FINAL BEST EVAL (ep={best_payload['epoch']})] "
                f"DDIM10 cross_full={final_eval['ddim']['cross_with_delta']['action10_mse']:.6f} "
                f"cross_nd={final_eval['ddim']['cross_no_delta']['action10_mse']:.6f} "
                f"val_L_id={final_eval['latent']['exact_self_identity_L_id']:.6f}",
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
        "--output-dir", type=Path, default=DEFAULT_DIRECT_IDENTITY_DIR / "train"
    )
    # Exact E9 Stage 2 Direct hyperparameters + lambda_id=0.1
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
    parser.add_argument("--lambda-id", type=float, default=0.1)
    parser.add_argument("--grad-log-every", type=int, default=200)
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
