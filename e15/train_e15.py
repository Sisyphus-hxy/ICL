"""Unified matched-budget trainer for E15: Robust Relational Behavior Representation Learning.

Strictly preserves E9/E12's exact architecture (`LocalBehaviorEncoder` + `ConditionalUnet1D`,
`4 x 768 = 3072D` latent, 149.58M params), optimizer (`AdamW lr=1e-4` cosine, `EMA=0.999`,
`clip_grad_norm=0.3`), and 1.0-epoch training budget (`total_steps=5364`, `batch_size=256`).

Supports:
  - E15-A (Frozen Encoder + Noise-Robust Decoder):
      `--freeze-encoder --rel-mode none --noise-type {real_cv,rand_matched}`
      Applied identically to E12-C (`e12_c/last_ema.pt`) and fair E9 baseline (`e9/best.pt`).
  - E15-B (Joint Encoder + Decoder with Relation Loss + Latent Robustness):
      `--rel-mode e12_c --noise-type {real_cv,rand_matched} --refresh-cv-every 1000`
      and fair E9 baseline (`--rel-mode none --noise-type {real_cv,rand_matched}`).
  - E15-C (Operational-Subspace-Restricted Relation + Residual Subspace Preservation):
      Splits each token `b_t[:, k, :]` (768D) into `z_op = b_t[:, :, :op_dim]` (default 384D,
      total 1536D) and `z_res = b_t[:, :, op_dim:]` (default 384D, total 1536D) with ZERO
      architecture or parameter change:
        * `op_e12_c`        : `L_diff(b_t) + lambda_rel * L_e12c(z_op)`
        * `op_e12_c_decorr` : `L_diff(b_t) + lambda_rel * L_e12c(z_op) + lambda_decorr * L_CKA(z_op, z_res)`
        * `op_bcfd`         : `L_diff(b_t) + lambda_rel * L_FactorDisp(z_op)` (+ optional decorr)
  - Step 5 Combinations: any combination of `rel-mode`, `noise-type`, `freeze-encoder`, `op-dim`,
    and `lambda-decorr`, decided strictly after evaluating E15-A / E15-B / E15-C!
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import (
    DEFAULT_DENSE_CACHE_DIR,
    DEFAULT_STAGE1_LOCAL_CKPT,
    E5_CACHE_DIR,
    EMBED_DIM,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_HEADS,
)
from e9.data import LocalClipFeatureBank
from e9.stage1_local_model import E9Stage1LocalPolicy
from e9.train_stage1_local import evaluate_stage1_local_validation
from e12.eval_suite import ValClipCache
from e12.losses import compute_e12_b_loss, compute_e12_c_loss
from e12.train_e12 import PairedTransitionBatchSampler
from e13.exp5_e12_bc import compute_bc_loss, compute_factorized_displacement_loss

from . import BASE_CHECKPOINTS
from .residual_noise import (
    LatentResidualNoiseInjector,
    build_train_only_cv_pairs,
    compute_cv_residual_bank_for_policy,
    load_residual_bank,
)


def compute_linear_cka_penalty(z_op: torch.Tensor, z_res: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant linear CKA in `[0, 1]` between `z_op` `(B, D_op)` and `z_res` `(B, D_res)`.

    Using batch Gram matrices `G_op = Zc_op @ Zc_op^T` and `G_res = Zc_res @ Zc_res^T`
    (`B x B`, typically `256 x 256`), computes exact:
        CKA = <G_op, G_res>_F / (||G_op||_F * ||G_res||_F)
    Minimizing CKA encourages `z_res` to encode genuinely complementary residual/contextual
    information rather than linearly duplicating `z_op`, with zero extra parameters.
    """
    B = z_op.shape[0]
    zo = z_op.float().reshape(B, -1)
    zr = z_res.float().reshape(B, -1)
    zo = zo - zo.mean(dim=0, keepdim=True)
    zr = zr - zr.mean(dim=0, keepdim=True)
    G_o = zo @ zo.T
    G_r = zr @ zr.T
    num = (G_o * G_r).sum()
    denom = (torch.linalg.norm(G_o, ord="fro") * torch.linalg.norm(G_r, ord="fro")).clamp_min(eps)
    return (num / denom).clamp(0.0, 1.0)


def compute_e15_relational_loss(
    rel_mode: str,
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
    *,
    op_dim: int = 384,
    disp_norm_weight: float = 0.25,
    lambda_decorr: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute E15 relational (+ optional subspace CKA decorrelation) loss on clean `b_t` `(B, 4, 768)`."""
    if rel_mode == "none":
        zero = b_t.sum() * 0.0
        return zero, {"rel_loss": zero.detach()}

    if rel_mode == "e12_c":
        return compute_e12_c_loss(b_t, actions_10d, actions_7d, norm_weight=disp_norm_weight)

    if rel_mode == "e12_bc_factordisp":
        return compute_bc_loss("e12_bc_factordisp", b_t, actions_10d, actions_7d)

    if rel_mode in ("op_e12_c", "op_e12_c_decorr", "op_bcfd", "op_bcfd_decorr"):
        if not (0 < op_dim < b_t.shape[-1]):
            raise ValueError(f"op_dim={op_dim} must lie strictly in (0, {b_t.shape[-1]})")
        z_op = b_t[:, :, :op_dim]   # (B, 4, op_dim)
        z_res = b_t[:, :, op_dim:]  # (B, 4, 768 - op_dim)

        if rel_mode.startswith("op_e12_c"):
            l_rel, stats = compute_e12_c_loss(
                z_op, actions_10d, actions_7d, norm_weight=disp_norm_weight
            )
        else:
            l_fd, s_fd = compute_factorized_displacement_loss(
                z_op, actions_10d, actions_7d, norm_weight=disp_norm_weight
            )
            l_b, _ = compute_e12_b_loss(z_op, actions_10d, actions_7d)
            l_rel = 0.75 * l_fd + 0.25 * l_b
            stats = {"rel_loss": l_rel.detach(), "loss_fdisp": l_fd.detach(), "loss_b": l_b.detach()}

        # Track subspace energy balance & CKA overlap
        cka = compute_linear_cka_penalty(z_op, z_res)
        with torch.no_grad():
            zo_c = z_op.float() - z_op.float().mean(dim=0, keepdim=True)
            zr_c = z_res.float() - z_res.float().mean(dim=0, keepdim=True)
            e_op = float(zo_c.square().sum(dim=(1, 2)).mean().item())
            e_res = float(zr_c.square().sum(dim=(1, 2)).mean().item())
            stats["cka_op_res"] = cka.detach()
            stats["res_energy_frac"] = b_t.new_tensor(e_res / max(e_op + e_res, 1e-12))

        eff_decorr = float(lambda_decorr)
        if rel_mode.endswith("_decorr") and eff_decorr <= 0.0:
            eff_decorr = 0.05
        if eff_decorr > 0.0:
            # Note: CKA penalty is returned scaled relative to lambda_rel inside total loss in caller,
            # OR added here cleanly with explicit weight ratio! Wait: better to return l_rel and
            # l_decorr separately in stats and fold eff_decorr directly into `l_rel` ONLY if caller
            # doesn't rescale it. Wait — caller multiplies by `cur_lam` (0.15)! So to keep `lambda_decorr`
            # independent of `lambda_rel`, we return `(l_rel, stats)` AND attach `l_decorr_raw`!
            stats["decorr_loss_raw"] = cka
        return l_rel, stats

    raise ValueError(f"Unknown rel_mode: {rel_mode}")


def _set_train_mode(policy: E9Stage1LocalPolicy, freeze_encoder: bool) -> None:
    """Keep frozen `local_encoder` strictly in `.eval()` mode so Dropout never perturbs frozen latents!"""
    if freeze_encoder:
        policy.local_encoder.eval()
        policy.action_decoder.train()
    else:
        policy.train()


def run_e15_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    t_load0 = time.monotonic()
    bank = LocalClipFeatureBank(
        e5_cache_dir=args.e5_cache_dir,
        dense_cache_dir=args.dense_cache_dir,
        device=device,
        resident_on_device=bool(args.gpu_resident_cache),
        seed=args.seed,
    )
    print(
        f"[E15:{args.run_name}] Loaded LocalClipFeatureBank "
        f"(gpu_resident={args.gpu_resident_cache}) in {time.monotonic()-t_load0:.1f}s",
        flush=True,
    )

    seed_everything(args.seed)
    policy = E9Stage1LocalPolicy(
        embed_dim=EMBED_DIM,
        window_len=LOCAL_WINDOW_LEN,
        num_behavior_tokens=NUM_BEHAVIOR_TOKENS,
        encoder_layers=6,
        encoder_heads=NUM_HEADS,
        encoder_ffn_dim=4 * EMBED_DIM,
        dropout=0.1,
        action_normalizer=bank.action_normalizer,
    ).to(device)

    init_ckpt = Path(args.init_ckpt)
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt}")
    payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["model"], strict=True)
    print(f"[E15:{args.run_name}] Warm-started from {init_ckpt}", flush=True)

    freeze_encoder = bool(args.freeze_encoder)
    if freeze_encoder:
        for p in policy.local_encoder.parameters():
            p.requires_grad_(False)
        policy.local_encoder.eval()

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        policy.get_optim_groups(lr=args.lr, weight_decay=args.weight_decay),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=args.weight_decay,
    )

    total_steps = int(args.total_steps)
    eval_every = int(args.eval_every)
    warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    ema = EMAModel(
        trainable_params,
        decay=args.ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    # Setup CV residual noise bank if noise_type != 'none'
    noise_injector: Optional[LatentResidualNoiseInjector] = None
    cv_pairs: Optional[Dict[str, Any]] = None
    cv_refresh_log: List[Dict[str, Any]] = []
    if args.noise_type != "none":
        need_live_refresh = (not freeze_encoder) and (int(args.refresh_cv_every) > 0)
        if need_live_refresh or args.noise_bank_path is None or not Path(args.noise_bank_path).is_file():
            val_cache = ValClipCache(e5_cache_dir=args.e5_cache_dir, dense_cache_dir=args.dense_cache_dir, seed=42)
            cv_pairs = build_train_only_cv_pairs(
                val_cache, n_folds=5, n_pairs_per_mode=5000, seed=2026
            )
        if args.noise_bank_path is not None and Path(args.noise_bank_path).is_file():
            nb = load_residual_bank(Path(args.noise_bank_path))
            print(
                f"[E15:{args.run_name}] Loaded cached CV noise bank {args.noise_bank_path} "
                f"(N={len(nb['residuals'])}, R_M={nb['R_M']:.2f}, rel_err={nb['rel_err']:.4f})",
                flush=True,
            )
        else:
            assert cv_pairs is not None
            nb = compute_cv_residual_bank_for_policy(policy, cv_pairs, device, compute_diagnostics=False)
            print(
                f"[E15:{args.run_name}] Computed initial CV noise bank "
                f"(N={len(nb['residuals'])}, R_M={nb['R_M']:.2f}, rel_err={nb['rel_err']:.4f})",
                flush=True,
            )
        noise_injector = LatentResidualNoiseInjector(
            nb["residuals"],
            nb["R_M"],
            device,
            noise_type=args.noise_type,
            p_noisy=args.p_noisy,
            scale_mult=args.noise_scale_mult,
            seed=args.seed,
        )
        cv_refresh_log.append(
            {"step": 0, "R_M": nb["R_M"], "rel_err": nb["rel_err"], "res_rms_l2": nb["res_rms_l2"]}
        )

    use_paired = bool(args.paired_transitions or args.rel_mode != "none")
    paired_sampler = (
        PairedTransitionBatchSampler(bank, within_ep_frac=0.5, seed=args.seed)
        if use_paired
        else None
    )

    config = {
        "stage": f"e15_{args.run_name}",
        "run_name": args.run_name,
        "init_ckpt": str(init_ckpt),
        "freeze_encoder": freeze_encoder,
        "rel_mode": args.rel_mode,
        "lambda_rel": float(args.lambda_rel),
        "rel_warmup_steps": int(args.rel_warmup_steps),
        "op_dim": int(args.op_dim),
        "lambda_decorr": float(args.lambda_decorr),
        "noise_type": args.noise_type,
        "p_noisy": float(args.p_noisy),
        "noise_scale_mult": float(args.noise_scale_mult),
        "refresh_cv_every": int(args.refresh_cv_every),
        "paired_transitions": use_paired,
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "eval_every": eval_every,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lr_warmup_proportion": float(args.lr_warmup_proportion),
        "clip_grad_norm": float(args.clip_grad_norm),
        "ema_decay": float(args.ema_decay),
        "embed_dim": EMBED_DIM,
        "window_len": LOCAL_WINDOW_LEN,
        "num_behavior_tokens": NUM_BEHAVIOR_TOKENS,
        "encoder_layers": 6,
        "encoder_heads": NUM_HEADS,
        "encoder_ffn_dim": 4 * EMBED_DIM,
        "dropout": 0.1,
        "seed": int(args.seed),
        "val_seed": int(args.val_seed),
        "parameter_summary": policy.parameter_summary(),
    }
    atomic_write_json(config, run_dir / "config.json")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    init_val = evaluate_stage1_local_validation(
        policy,
        bank,
        batch_size=256,
        run_ddim=True,
        ddim_max_samples=512,
        run_timestep_check=False,
        val_seed=args.val_seed,
    )
    atomic_write_json(init_val, run_dir / "init_val.json")
    print(
        f"[E15:{args.run_name} Init] val_self_diff={init_val['val_self_diff_loss']:.5f} | "
        f"ddim_10d_mse={init_val['ddim']['self']['action10_mse']:.5f} | "
        f"grip={init_val['ddim']['self']['grip_sign_acc']*100:.2f}% | "
        f"b_rms={init_val['val_self_b_rms']:.3f}",
        flush=True,
    )

    def _infinite_batches() -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ep = 1
        while True:
            if paired_sampler is not None:
                yield from paired_sampler.iter_batches(ep, args.batch_size)
            else:
                yield from bank.iter_train_batches(ep, args.batch_size)
            ep += 1

    batch_iter = _infinite_batches()
    global_step = 0
    best_val_loss = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []
    wt0 = time.monotonic()
    acc_diff = acc_rel = acc_dec = acc_tot = acc_eps = 0.0
    acc_steps = 0

    while global_step < total_steps:
        _set_train_mode(policy, freeze_encoder)
        clip_rgb, clip_pad_mask, actions_10d = next(batch_iter)
        global_step += 1

        # Refresh live CV residual bank if encoder is updating
        if (
            noise_injector is not None
            and (not freeze_encoder)
            and int(args.refresh_cv_every) > 0
            and global_step > 1
            and ((global_step - 1) % int(args.refresh_cv_every) == 0)
        ):
            assert cv_pairs is not None
            nb = compute_cv_residual_bank_for_policy(policy, cv_pairs, device, compute_diagnostics=False)
            noise_injector.update_bank(nb["residuals"], nb["R_M"])
            cv_refresh_log.append(
                {
                    "step": global_step - 1,
                    "R_M": nb["R_M"],
                    "rel_err": nb["rel_err"],
                    "res_rms_l2": nb["res_rms_l2"],
                }
            )
            print(
                f"[E15:{args.run_name}] Refreshed live CV noise bank @step {global_step-1}: "
                f"R_M={nb['R_M']:.2f}, rel_err={nb['rel_err']:.4f}",
                flush=True,
            )
            _set_train_mode(policy, freeze_encoder)

        if args.rel_warmup_steps > 0 and global_step <= args.rel_warmup_steps:
            cur_lam = float(args.lambda_rel) * (global_step / float(args.rel_warmup_steps))
            cur_dec = float(args.lambda_decorr) * (global_step / float(args.rel_warmup_steps))
        else:
            cur_lam = float(args.lambda_rel)
            cur_dec = float(args.lambda_decorr)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            if freeze_encoder:
                with torch.no_grad():
                    b_t = policy.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad_mask)
            else:
                b_t = policy.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad_mask)

            if noise_injector is not None:
                eps, n_stats = noise_injector.sample_noise(b_t)
                b_dec = b_t + eps
            else:
                n_stats = {"rel_eps_rms": 0.0}
                b_dec = b_t

            diff_loss = policy.diffusion_loss_from_z_self(b_dec, actions_10d).mean()

        rel_loss = diff_loss.new_zeros(())
        dec_loss = diff_loss.new_zeros(())
        rel_stats: Dict[str, torch.Tensor] = {}
        if args.rel_mode != "none" and (not freeze_encoder) and cur_lam > 0.0:
            a7 = action_10d_to_7d(actions_10d.float())
            rel_loss, rel_stats = compute_e15_relational_loss(
                args.rel_mode,
                b_t.float(),
                actions_10d.float(),
                a7,
                op_dim=args.op_dim,
                disp_norm_weight=args.disp_norm_weight,
                lambda_decorr=cur_dec,
            )
            total_loss = diff_loss + cur_lam * rel_loss
            if "decorr_loss_raw" in rel_stats and cur_dec > 0.0:
                dec_loss = rel_stats["decorr_loss_raw"]
                total_loss = total_loss + cur_dec * dec_loss
        else:
            total_loss = diff_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        ema.step(trainable_params)

        acc_diff += float(diff_loss.item())
        acc_rel += float(rel_loss.item())
        acc_dec += float(dec_loss.item())
        acc_tot += float(total_loss.item())
        acc_eps += float(n_stats["rel_eps_rms"])
        acc_steps += 1

        if (global_step % eval_every == 0) or (global_step == total_steps):
            dt = time.monotonic() - wt0
            ema.store(trainable_params)
            ema.copy_to(trainable_params)
            do_ddim = (global_step == total_steps) or (global_step % (eval_every * 2) == 0)
            val_metrics = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=256,
                run_ddim=do_ddim,
                ddim_max_samples=512,
                run_timestep_check=False,
                val_seed=args.val_seed,
            )
            cur_val = float(val_metrics["val_self_diff_loss"])
            eligible = global_step >= max(args.rel_warmup_steps, eval_every)
            is_best = eligible and (cur_val < best_val_loss)
            if is_best or best_step == 0:
                best_val_loss = cur_val
                best_step = global_step
                atomic_torch_save(
                    {
                        "stage": f"e15_{args.run_name}",
                        "step": global_step,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    run_dir / "best.pt",
                )
            if global_step == total_steps:
                atomic_torch_save(
                    {
                        "stage": f"e15_{args.run_name}",
                        "step": global_step,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    run_dir / "last_ema.pt",
                )
            ema.restore(trainable_params)

            tr_rec = {
                "diff_loss": acc_diff / max(acc_steps, 1),
                "rel_loss": acc_rel / max(acc_steps, 1),
                "decorr_loss": acc_dec / max(acc_steps, 1),
                "total_loss": acc_tot / max(acc_steps, 1),
                "rel_eps_rms": acc_eps / max(acc_steps, 1),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "steps_per_sec": round(acc_steps / max(dt, 1e-6), 2),
                "rel_stats_last": {
                    k: float(v.detach().item()) for k, v in rel_stats.items() if k != "decorr_loss_raw"
                },
            }
            history.append({"step": global_step, "train": tr_rec, "val": val_metrics})
            atomic_write_json(
                {
                    "config": config,
                    "best_step": best_step,
                    "best_val_self_diff_loss": best_val_loss,
                    "cv_refresh_log": cv_refresh_log,
                    "history": history,
                },
                run_dir / "metrics.json",
            )
            ddim_str = ""
            if "ddim" in val_metrics:
                dm = val_metrics["ddim"]["self"]
                ddim_str = (
                    f" | ddim_mse10={dm['action10_mse']:.5f} mae7={dm['action7_mae']:.5f} "
                    f"grip={dm['grip_sign_acc']*100:.2f}%"
                )
            print(
                f"[E15:{args.run_name} step {global_step:5d}/{total_steps}] "
                f"tr_diff={tr_rec['diff_loss']:.5f} rel={tr_rec['rel_loss']:.5f} "
                f"dec={tr_rec['decorr_loss']:.4f} eps={tr_rec['rel_eps_rms']:.3f} | "
                f"val_diff={cur_val:.5f} b_rms={val_metrics['val_self_b_rms']:.3f} | "
                f"{tr_rec['steps_per_sec']:.1f}it/s{ddim_str}",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_diff = acc_rel = acc_dec = acc_tot = acc_eps = 0.0
            acc_steps = 0

    print(f"[E15:{args.run_name}] Done. Saved to {run_dir / 'last_ema.pt'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument(
        "--rel-mode",
        type=str,
        default="none",
        choices=["none", "e12_c", "e12_bc_factordisp", "op_e12_c", "op_e12_c_decorr", "op_bcfd", "op_bcfd_decorr"],
    )
    p.add_argument("--lambda-rel", type=float, default=0.15)
    p.add_argument("--rel-warmup-steps", type=int, default=300)
    p.add_argument("--disp-norm-weight", type=float, default=0.25)
    p.add_argument("--op-dim", type=int, default=384)
    p.add_argument("--lambda-decorr", type=float, default=0.0)
    p.add_argument("--noise-type", type=str, default="none", choices=["none", "real_cv", "rand_matched"])
    p.add_argument("--noise-bank-path", type=Path, default=None)
    p.add_argument("--p-noisy", type=float, default=0.75)
    p.add_argument("--noise-scale-mult", type=float, default=1.0)
    p.add_argument("--refresh-cv-every", type=int, default=1000)
    p.add_argument("--paired-transitions", action="store_true")
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=5364)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-6)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.3)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=12345)
    return p.parse_args()


def main() -> None:
    run_e15_training(parse_args())


if __name__ == "__main__":
    main()
