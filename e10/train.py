"""Training runner for P1 (E10-A / E10-B / E10-C / Control) Learned Low-Rank Transfer.

Strictly preserves E9 Stage 2's data bank, pairing schedule, loss (L = L_diff + 0.5 * L_z),
optimizer, cosine LR schedule, EMA, batch size (256), and 8-epoch budget (42,912 steps),
while keeping E9 Encoder, E9 Reader, Action Decoder, and P0 SVD basis U_K completely frozen.
Best checkpoint is selected strictly on independent validation Cross DDIM 10D Action MSE.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from e5.diffusion import EMAModel
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e7.train import get_cosine_schedule_with_min_ratio
from e9.stage2_data import E9Stage2TransferBank
from e9.train_stage2_direct import evaluate_e9_stage2_validation

from . import (
    E5_CACHE_DIR,
    E9_STAGE1_CKPT,
    E9_STAGE2_CACHE_DIR,
    E9_STAGE2_CKPT,
    OUTPUT_ROOT,
    P0_BASES_PATH,
    VARIANTS,
)
from .model import E10LowRankPolicy, build_e10_policy, load_e10_policy


def _audit_freeze_and_gradients(
    policy: E10LowRankPolicy, bank: E9Stage2TransferBank, device: torch.device
) -> Dict[str, Any]:
    """Verify frozen modules get ZERO grad while trainable transfer gets non-zero grad."""
    policy.train()
    assert not policy.teacher.training, "Teacher must stay in eval()"
    assert not policy.reader.training, "Reader must stay in eval()"

    it = bank.iter_e7_train_batches(epoch=1, batch_size=16, max_samples=16)
    h_mem, b_mem, mpm, q_rgb, z_self, actions, is_self = next(it)

    for p in policy.parameters():
        p.grad = None

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = policy(h_mem, b_mem, mpm, q_rgb, z_self, actions, is_self, lambda_a=1.0, lambda_z=0.5)
    out["loss"].backward()

    teacher_any_grad = any(p.grad is not None for p in policy.teacher.parameters())
    reader_any_grad = any(p.grad is not None for p in policy.reader.parameters())
    basis_req_grad = bool(policy.basis_u_k.requires_grad)
    head_gn = float(policy.transfer.coeff_head.weight.grad.norm().item()) if policy.transfer.coeff_head.weight.grad is not None else 0.0
    transfer_grad_params = sum(1 for p in policy.transfer.parameters() if p.grad is not None and p.grad.norm().item() > 0)

    for p in policy.parameters():
        p.grad = None

    audit = {
        "teacher_any_grad": teacher_any_grad,
        "reader_any_grad": reader_any_grad,
        "basis_requires_grad": basis_req_grad,
        "coeff_head_grad_norm": head_gn,
        "transfer_nonzero_grad_tensors": transfer_grad_params,
        "step0_delta_rms": float(out["delta_rms"].item()),
    }
    if teacher_any_grad or reader_any_grad or basis_req_grad:
        raise RuntimeError(f"Freeze audit FAILED: {audit}")
    if head_gn <= 0.0:
        raise RuntimeError(f"Zero gradient into transfer coeff_head: {audit}")
    return audit


def train_e10(
    *,
    variant: str,
    run_dir: Path,
    rank_k: int,
    transfer_mode: str,
    basis_path: Path = P0_BASES_PATH,
    basis_key: str = "U_reader",
    stage1_ckpt: Path = E9_STAGE1_CKPT,
    e9_stage2_ckpt: Path = E9_STAGE2_CKPT,
    epochs: int = 8,
    batch_size: int = 256,
    val_batch_size: int = 256,
    lr: float = 1e-4,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    clip_grad_norm: float = 1.0,
    lambda_a: float = 1.0,
    lambda_z: float = 0.5,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    seed: int = 42,
    val_seed: int = 12345,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    ddim_val_samples: Optional[int] = None,
    multi_demo_samples: int = 1024,
    resume: bool = False,
    device: torch.device,
) -> Path:
    t0 = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    bank = E9Stage2TransferBank(
        clip_cache_dir=E5_CACHE_DIR,
        stage2_cache_dir=E9_STAGE2_CACHE_DIR,
        device=device,
        resident_on_device=True,
        z_self_on_device=True,
        self_ratio=0.025,
        seed=seed,
    )

    policy, build_meta = build_e10_policy(
        rank_k=rank_k,
        transfer_mode=transfer_mode,
        basis_path=basis_path,
        basis_key=basis_key,
        stage1_ckpt=stage1_ckpt,
        e9_stage2_ckpt=e9_stage2_ckpt,
        device=device,
    )
    s_z = float(policy.latent_scale_s_z.item())
    audit = _audit_freeze_and_gradients(policy, bank, device)
    print(f"[E10:{variant}] Freeze & grad audit PASSED: {json.dumps(audit)}", flush=True)

    trainable_params = policy.trainable_parameters()
    optim_groups = policy.get_optim_groups(lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(optim_groups, betas=(beta1, beta2))

    epoch_len = (
        bank.num_train_samples()
        if max_train_samples is None
        else min(int(max_train_samples), bank.num_train_samples())
    )
    steps_per_epoch = math.ceil(epoch_len / batch_size)
    total_steps = steps_per_epoch * epochs
    lr_scheduler = get_cosine_schedule_with_min_ratio(
        optimizer,
        num_warmup_steps=min(warmup_steps, max(total_steps // 10, 1)),
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    ema = (
        EMAModel(
            trainable_params,
            decay=ema_decay,
            min_decay=0.0,
            update_after_step=0,
            use_ema_warmup=False,
            inv_gamma=1.0,
            power=0.75,
            foreach=True,
        ).to(device)
        if use_ema
        else None
    )

    config: Dict[str, Any] = {
        "stage": "e10_p1_learned_lowrank_transfer",
        "variant": variant,
        "rank_k": int(rank_k),
        "transfer_mode": transfer_mode,
        **build_meta,
        "latent_scale_s_z": s_z,
        "lambda_a": float(lambda_a),
        "lambda_z": float(lambda_z),
        "self_ratio": 0.025,
        "cross_ratio": 0.975,
        "residual_layers": 3,
        "residual_heads": 8,
        "residual_ffn_dim": 3072,
        "residual_dropout": 0.1,
        "epochs": int(epochs),
        "global_batch": int(batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "lr": float(lr),
        "min_lr": float(lr * min_lr_ratio),
        "warmup_steps": int(warmup_steps),
        "betas": [beta1, beta2],
        "weight_decay": float(weight_decay),
        "clip_grad_norm": float(clip_grad_norm),
        "mixed_precision": "bf16",
        "use_ema": bool(use_ema),
        "ema_decay": ema_decay if use_ema else None,
        "seed": int(seed),
        "val_seed": int(val_seed),
        "best_selection_metric": "ddim.cross_with_delta.action10_mse",
        "freeze_audit": audit,
        "parameter_summary": policy.parameter_summary(),
    }
    atomic_write_json(config, run_dir / "config.json")

    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"
    history: List[Dict[str, Any]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    start_epoch = 1
    global_step = 0

    if resume and last_ckpt.is_file():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        policy.transfer.load_state_dict(ck["transfer"], strict=True)
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        lr_scheduler.load_state_dict(ck["lr_scheduler"])
        history = ck["history"]
        best_val_mse = ck["best_val_cross_ddim_mse10"]
        best_epoch = ck["best_epoch"]
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        print(f"[E10:{variant}] Resumed from epoch {ck['epoch']} (best={best_val_mse:.5f})", flush=True)

    ps = config["parameter_summary"]
    print(
        f"[E10:{variant}] mode={transfer_mode} K={rank_k} ({basis_key}) | "
        f"epochs={epochs} ({total_steps} steps, bs={batch_size}) | lr={lr:.1e}->{lr*min_lr_ratio:.1e} | "
        f"trainable={ps['total_trainable']:,} (reader_frozen={ps['reader_frozen']:,}, "
        f"teacher_frozen={ps['stage1_teacher_frozen']:,}) | out={run_dir}",
        flush=True,
    )

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    init_val = None
    if start_epoch == 1:
        init_val = evaluate_e9_stage2_validation(
            policy,
            bank,
            batch_size=val_batch_size,
            max_val_samples=max_val_samples,
            run_ddim=True,
            ddim_max_samples=ddim_val_samples,
            run_multi_demo=True,
            multi_demo_max_samples=multi_demo_samples,
            val_seed=val_seed,
        )
        atomic_write_json(init_val, run_dir / "init_val.json")
        dm = init_val["ddim"]
        lt = init_val["latent"]
        print(
            f"[E10:{variant} InitVal] zq_rms={lt['adapted_z_rms']:.4f} | "
            f"Lz={lt['cross_with_delta_L_z']:.4f} (nd={lt['cross_no_delta_L_z']:.4f}) | "
            f"DDIM mse10 c_nd={dm['cross_no_delta']['action10_mse']:.5f} "
            f"c+d={dm['cross_with_delta']['action10_mse']:.5f} "
            f"self={dm['self_teacher']['action10_mse']:.5f}",
            flush=True,
        )
        seed_everything(seed + 1000)

    for epoch in range(start_epoch, epochs + 1):
        t_ep = time.monotonic()
        policy.train()
        tr_acc = torch.zeros(14, device=device, dtype=torch.float64)

        for (
            h_mem,
            b_mem,
            mem_pad_mask,
            query_rgb,
            z_self,
            actions_10d,
            is_self,
        ) in bank.iter_e7_train_batches(
            epoch, batch_size, rank=0, world_size=1, max_samples=max_train_samples
        ):
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out = policy(
                    h_mem,
                    b_mem,
                    mem_pad_mask,
                    query_rgb,
                    z_self,
                    actions_10d,
                    is_self,
                    lambda_a=lambda_a,
                    lambda_z=lambda_z,
                )
            out["loss"].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad_norm).item())
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

        t_train = time.monotonic() - t_ep
        n_tot = max(float(tr_acc[0].item()), 1.0)
        n_c = max(float(tr_acc[9].item()), 1.0)
        n_s = max(float(tr_acc[11].item()), 1.0)
        n_st = max(float(tr_acc[13].item()), 1.0)

        train_metrics = {
            "samples": int(round(n_tot)),
            "loss": float(tr_acc[1].item() / n_tot),
            "diff_loss": float(tr_acc[2].item() / n_tot),
            "L_z": float(tr_acc[3].item() / n_tot),
            "nodelta_L_z": float(tr_acc[4].item() / n_tot),
            "cos_nodelta_self": float(tr_acc[5].item() / n_tot),
            "cos_adapted_self": float(tr_acc[6].item() / n_tot),
            "dir_cos": float(tr_acc[7].item() / n_tot),
            "cross_delta_rms": math.sqrt(float(tr_acc[8].item() / n_c)),
            "self_delta_rms": math.sqrt(float(tr_acc[10].item() / n_s)),
            "grad_norm_mean": float(tr_acc[12].item() / n_st),
            "lr": float(lr_scheduler.get_last_lr()[0]),
            "train_seconds": round(t_train, 2),
            "steps_per_sec": round(n_st / max(t_train, 1e-6), 2),
        }

        t_v0 = time.monotonic()
        if ema is not None:
            ema.store(trainable_params)
            ema.copy_to(trainable_params)

        val_metrics = evaluate_e9_stage2_validation(
            policy,
            bank,
            batch_size=val_batch_size,
            max_val_samples=max_val_samples,
            run_ddim=True,
            ddim_max_samples=ddim_val_samples,
            run_multi_demo=(epoch == 1 or epoch == epochs or epoch % 2 == 0),
            multi_demo_max_samples=multi_demo_samples,
            val_seed=val_seed,
        )
        val_metrics["val_seconds"] = round(time.monotonic() - t_v0, 2)

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

        if is_best:
            atomic_torch_save(
                {
                    "stage": "e10_p1_learned_lowrank_transfer",
                    "variant": variant,
                    "rank_k": int(rank_k),
                    "transfer_mode": transfer_mode,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_cross_ddim_mse10": best_val_mse,
                    "latent_scale_s_z": s_z,
                    "basis_u_k": policy.basis_u_k.detach().cpu(),
                    "z_mean": policy.z_mean.detach().cpu(),
                    "transfer": {k: v.detach().cpu() for k, v in policy.transfer.state_dict().items()},
                    "config": config,
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )

        if ema is not None:
            ema.restore(trainable_params)

        if epoch % 2 == 0 or epoch == epochs:
            atomic_torch_save(
                {
                    "stage": "e10_p1_learned_lowrank_transfer",
                    "variant": variant,
                    "rank_k": int(rank_k),
                    "transfer_mode": transfer_mode,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_cross_ddim_mse10": best_val_mse,
                    "latent_scale_s_z": s_z,
                    "basis_u_k": policy.basis_u_k.detach().cpu(),
                    "z_mean": policy.z_mean.detach().cpu(),
                    "transfer": {k: v.detach().cpu() for k, v in policy.transfer.state_dict().items()},
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
                "stage": "e10_p1_learned_lowrank_transfer",
                "variant": variant,
                "rank_k": int(rank_k),
                "transfer_mode": transfer_mode,
                "best_epoch": best_epoch,
                "best_val_cross_ddim_mse10": best_val_mse,
                "init_val": init_val,
                "history": history,
            },
            run_dir / "metrics.json",
        )

        dm = val_metrics["ddim"]
        lt = val_metrics["latent"]
        c_mse = dm["cross_with_delta"]["action10_mse"]
        w_mse = dm["wrong_demo_with_delta"]["action10_mse"]
        w_delta = dm["demo_sensitivity_action10_mse"]
        w_ratio = w_mse / max(c_mse, 1e-12)
        print(
            f"[E10:{variant} Ep {epoch:02d}/{epochs}] "
            f"tr L={train_metrics['loss']:.4f} (diff={train_metrics['diff_loss']:.4f}, Lz={train_metrics['L_z']:.4f}) | "
            f"val Lz={lt['cross_with_delta_L_z']:.4f} | "
            f"DDIM c+d={c_mse:.5f} "
            f"(mae7={dm['cross_with_delta']['action7_mae']:.5f}, grip={100*dm['cross_with_delta']['grip_sign_acc']:.2f}%) | "
            f"wrong_Δ={w_delta:+.5f} ({w_ratio:.2f}x) | "
            f"best={best_val_mse:.5f}@ep{best_epoch} | "
            f"t={train_metrics['train_seconds']:.0f}s+{val_metrics['val_seconds']:.0f}s",
            flush=True,
        )

    # Final full validation on best.pt (including full multi-demo consistency)
    best_policy, best_payload = load_e10_policy(best_ckpt, device=device)
    final_eval = evaluate_e9_stage2_validation(
        best_policy,
        bank,
        batch_size=val_batch_size,
        max_val_samples=max_val_samples,
        run_ddim=True,
        ddim_max_samples=ddim_val_samples,
        run_multi_demo=True,
        multi_demo_max_samples=None if max_val_samples is None else min(multi_demo_samples, max_val_samples),
        val_seed=val_seed,
    )
    final_eval["best_epoch"] = best_payload["epoch"]
    final_eval["total_elapsed_seconds"] = round(time.monotonic() - t0, 2)
    atomic_write_json(final_eval, run_dir / "final_eval.json")
    print(
        f"[E10:{variant} DONE] best_ep={best_epoch} best_val_mse10={best_val_mse:.5f} "
        f"total={final_eval['total_elapsed_seconds']}s -> {run_dir}",
        flush=True,
    )
    return best_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=str, choices=tuple(VARIANTS.keys()), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--basis-path", type=Path, default=P0_BASES_PATH)
    parser.add_argument("--basis-key", type=str, default="U_reader")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    run_dir = args.output_root / args.variant / ("smoke" if args.smoke else "train")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_e10(
        variant=args.variant,
        run_dir=run_dir,
        rank_k=spec["rank_k"],
        transfer_mode=spec["transfer_mode"],
        basis_path=args.basis_path,
        basis_key=args.basis_key,
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        seed=args.seed,
        val_seed=args.val_seed,
        max_train_samples=1024 if args.smoke else None,
        max_val_samples=256 if args.smoke else None,
        ddim_val_samples=128 if args.smoke else None,
        multi_demo_samples=128 if args.smoke else 1024,
        resume=args.resume,
        device=device,
    )


if __name__ == "__main__":
    main()
