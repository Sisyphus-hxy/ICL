"""Regime 1 (E16-A / E16-B / E16-C) Trainer: Video-only Relation Supervision.

Strictly preserves E9/E12/E15's exact `E9Stage1LocalPolicy` (`4x768=3072D`, 149.58M params),
warm-start (`e9/outputs/stage1_local/best.pt`), optimizer (`AdamW lr=1e-4` cosine, `EMA=0.999`,
`clip_grad_norm=0.3`), and full-data 1.0-epoch budget (`total_steps=5364`, `batch_size=256`).

Crucial scientific distinction:
  - Action diffusion loss `L_diff` trains the policy just like E9-Cont / E12 / E15.
  - The RELATIONAL objective uses **ZERO GT actions** (strictly replacing E12's GT action
    similarity / displacement supervision with purely video-derived or learnable video-reuse
    relations!).

Supported modes:
  - Control:
      * `random_rel`         : Random cross-demo displacement supervision (strict null control)
  - E16-A (Video-derived Relations, full 3072D space):
      * `vid_feat_delta`     : Scene-decorrelated temporal CLIP feature-change displacement
      * `vid_motion_flow`    : Pixel-space optical flow + grid point-track displacement
      * `vid_pretrained_lam` : Frozen 100% video-only pretrained LAM displacement
      * `vid_multiview`      : Combined pixel optical flow + pretrained LAM displacement
  - E16-B (Learnable Reuse-Driven Relations, full 3072D space):
      * `learnable_reuse`    : Cross-demo behavior substitution + conditional state-change
                               prediction + counterfactual cycle + canonical equiv + VICReg
      * `learnable_tok_reuse`: 4-token autonomous factorized cross-demo substitution
  - E16-C (Structured Self-Supervised Representation on `z_op`, preserving functional `z_res`):
      * `op_vid_motion`      : `vid_motion_flow` restricted strictly to `z_op` (1536D)
      * `op_vid_multiview`   : `vid_multiview` restricted strictly to `z_op` (1536D)
      * `op_learnable_reuse` : `learnable_reuse` restricted strictly to `z_op` (1536D)
      * `op_learnable_tok`   : `learnable_tok_reuse` restricted strictly to `z_op` (1536D)
    + optional E15-D calibrated mild real-CV noise (`--noise-type real_cv --noise-scale-mult 0.35 --p-noisy 0.50`)!
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

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
from e9.stage1_local_model import E9Stage1LocalPolicy
from e9.train_stage1_local import evaluate_stage1_local_validation
from e12.eval_suite import ValClipCache
from e15.residual_noise import (
    LatentResidualNoiseInjector,
    build_train_only_cv_pairs,
    compute_cv_residual_bank_for_policy,
    load_residual_bank,
)
from e15.train_e15 import compute_linear_cka_penalty

from . import MOTION_CACHE_DIR, OUTPUT_ROOT
from .pretrain_video_only import load_pretrained_video_lam
from .video_relations import (
    LearnableReuseRelationModule,
    RichPairedBatchSampler,
    VideoMotionLocalBank,
    compute_video_derived_displacement_loss,
    extract_video_relation_features,
)

VID_DERIVED_MODES = {
    "random_rel": ("random_rel", False),
    "vid_feat_delta": ("feat_delta_decorr", False),
    "vid_motion_flow": ("motion_flow", False),
    "vid_pretrained_lam": ("pretrained_lam", False),
    "vid_multiview": ("multiview_vid", False),
    "op_vid_motion": ("motion_flow", True),
    "op_vid_lam": ("pretrained_lam", True),
    "op_vid_multiview": ("multiview_vid", True),
}

LEARNABLE_MODES = {
    "learnable_reuse": (False, False),      # (op_only, token_wise)
    "learnable_tok_reuse": (False, True),
    "op_learnable_reuse": (True, False),
    "op_learnable_tok": (True, True),
}


def run_e16_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    is_learnable = args.rel_mode in LEARNABLE_MODES
    need_motion = (
        args.rel_mode in ("vid_motion_flow", "vid_multiview", "op_vid_motion", "op_vid_multiview")
        or (is_learnable and bool(args.predict_motion))
    )
    need_lam = args.rel_mode in ("vid_pretrained_lam", "vid_multiview", "op_vid_lam", "op_vid_multiview")

    t0 = time.monotonic()
    bank = VideoMotionLocalBank(
        e5_cache_dir=args.e5_cache_dir,
        dense_cache_dir=args.dense_cache_dir,
        motion_cache_dir=args.motion_cache_dir if need_motion else None,
        device=device,
        resident_on_device=bool(args.gpu_resident_cache),
        seed=args.seed,
    )
    print(
        f"[E16:{args.run_name}] Loaded bank (has_motion={bank.has_motion}, "
        f"gpu_resident={args.gpu_resident_cache}) in {time.monotonic()-t0:.1f}s",
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
    payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["model"], strict=True)
    print(f"[E16:{args.run_name}] Warm-started from {init_ckpt}", flush=True)

    lam_encoder = None
    if need_lam:
        if not args.lam_ckpt.is_file():
            raise FileNotFoundError(f"Pretrained video-only LAM checkpoint not found: {args.lam_ckpt}")
        lam_encoder = load_pretrained_video_lam(args.lam_ckpt, device=device)
        print(f"[E16:{args.run_name}] Loaded frozen video-only LAM from {args.lam_ckpt}", flush=True)

    reuse_module: Optional[LearnableReuseRelationModule] = None
    op_only = False
    if is_learnable:
        op_only, token_wise = LEARNABLE_MODES[args.rel_mode]
        z_rel_d = NUM_BEHAVIOR_TOKENS * (args.op_dim if op_only else EMBED_DIM)
        reuse_module = LearnableReuseRelationModule(
            z_rel_dim=z_rel_d,
            num_tokens=NUM_BEHAVIOR_TOKENS,
            token_wise=token_wise,
            proj_dim=256,
            hidden_dim=1024,
            predict_motion=bool(args.predict_motion and bank.has_motion),
            coverage_mode=args.coverage_mode,
            top_k_donors=args.top_k_donors,
        ).to(device)
    else:
        _, op_only = VID_DERIVED_MODES[args.rel_mode]

    # Optimizer: policy groups + optional learnable relation module params (higher LR 3e-4 for fresh heads)
    optim_groups = policy.get_optim_groups(lr=args.lr, weight_decay=args.weight_decay)
    if reuse_module is not None:
        optim_groups.append(
            {"params": list(reuse_module.parameters()), "lr": args.lr * 3.0, "weight_decay": 1e-5}
        )
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay)

    total_steps = int(args.total_steps)
    warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    policy_params = list(policy.parameters())
    ema = EMAModel(
        policy_params,
        decay=args.ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    # Optional calibrated mild real-CV residual noise (for E16-C + MildCV!)
    noise_injector: Optional[LatentResidualNoiseInjector] = None
    cv_pairs: Optional[Dict[str, Any]] = None
    if args.noise_type != "none":
        val_cache = ValClipCache(e5_cache_dir=args.e5_cache_dir, dense_cache_dir=args.dense_cache_dir, seed=42)
        cv_pairs = build_train_only_cv_pairs(val_cache, n_folds=5, n_pairs_per_mode=5000, seed=2026)
        if args.noise_bank_path is not None and Path(args.noise_bank_path).is_file():
            nb = load_residual_bank(Path(args.noise_bank_path))
        else:
            nb = compute_cv_residual_bank_for_policy(policy, cv_pairs, device, compute_diagnostics=False)
        noise_injector = LatentResidualNoiseInjector(
            nb["residuals"],
            nb["R_M"],
            device,
            noise_type=args.noise_type,
            p_noisy=args.p_noisy,
            scale_mult=args.noise_scale_mult,
            seed=args.seed,
        )
        print(
            f"[E16:{args.run_name}] Noise injector ready (type={args.noise_type}, "
            f"scale={args.noise_scale_mult}, p={args.p_noisy}, R_M={nb['R_M']:.2f})",
            flush=True,
        )

    sampler = RichPairedBatchSampler(bank, within_ep_frac=0.5, seed=args.seed, include_actions=True)

    config = {
        "regime": "video_only_relation_supervision_zero_action_relation",
        "run_name": args.run_name,
        "rel_mode": args.rel_mode,
        "op_only": op_only,
        "op_dim": int(args.op_dim),
        "init_ckpt": str(init_ckpt),
        "lam_ckpt": str(args.lam_ckpt) if need_lam else None,
        "lambda_rel": float(args.lambda_rel),
        "rel_warmup_steps": int(args.rel_warmup_steps),
        "disp_norm_weight": float(args.disp_norm_weight),
        "predict_motion": bool(args.predict_motion),
        "coverage_mode": args.coverage_mode,
        "top_k_donors": args.top_k_donors,
        "noise_type": args.noise_type,
        "noise_scale_mult": float(args.noise_scale_mult),
        "p_noisy": float(args.p_noisy),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "clip_grad_norm": float(args.clip_grad_norm),
        "ema_decay": float(args.ema_decay),
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

    def _inf_batches() -> Iterator[Dict[str, Optional[torch.Tensor]]]:
        ep = 1
        while True:
            yield from sampler.iter_batches(ep, args.batch_size)
            ep += 1

    batch_iter = _inf_batches()
    step = 0
    best_val = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []
    wt0 = time.monotonic()
    acc_diff = acc_rel = acc_tot = 0.0
    acc_n = 0

    all_params = policy_params + (list(reuse_module.parameters()) if reuse_module is not None else [])

    while step < total_steps:
        policy.train()
        if reuse_module is not None:
            reuse_module.train()
        batch = next(batch_iter)
        step += 1

        if (
            noise_injector is not None
            and int(args.refresh_cv_every) > 0
            and step > 1
            and ((step - 1) % int(args.refresh_cv_every) == 0)
        ):
            assert cv_pairs is not None
            nb = compute_cv_residual_bank_for_policy(policy, cv_pairs, device, compute_diagnostics=False)
            noise_injector.update_bank(nb["residuals"], nb["R_M"])
            policy.train()

        cur_lam = (
            float(args.lambda_rel) * (step / float(args.rel_warmup_steps))
            if (args.rel_warmup_steps > 0 and step <= args.rel_warmup_steps)
            else float(args.lambda_rel)
        )

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = policy.encode_local_behavior(batch["clip_rgb"], clip_pad_mask=batch["clip_pad_mask"])
            if noise_injector is not None:
                eps, _ = noise_injector.sample_noise(b_t)
                b_dec = b_t + eps
            else:
                b_dec = b_t
            diff_loss = policy.diffusion_loss_from_z_self(b_dec, batch["actions_10d"]).mean()

        # Relational loss uses ZERO GT actions!
        rel_loss = diff_loss.new_zeros(())
        rel_stats: Dict[str, torch.Tensor] = {}
        if cur_lam > 0.0:
            if is_learnable:
                assert reuse_module is not None
                do_cyc = step > 300
                do_eq = step > 300
                rel_loss, rel_stats = reuse_module.forward_loss(
                    b_t,
                    batch["clip_rgb"],
                    batch["clip_pad_mask"],
                    batch["ep_ids"],
                    task_ids=batch["task_ids"],
                    clip_motion=batch["clip_motion"] if (args.predict_motion and bank.has_motion) else None,
                    encoder_fn=policy.encode_local_behavior if do_cyc else None,
                    op_dim=args.op_dim if op_only else None,
                    use_cycle=do_cyc,
                    use_equiv=do_eq,
                )
            else:
                src_key, use_op = VID_DERIVED_MODES[args.rel_mode]
                u_vid = extract_video_relation_features(
                    src_key,
                    batch["clip_rgb"],
                    batch["clip_pad_mask"],
                    batch["clip_motion"],
                    lam_encoder=lam_encoder,
                    ep_ids=batch["ep_ids"],
                    frame_ids=batch["frame_ids"],
                )
                z_tgt = b_t[:, :, : args.op_dim] if use_op else b_t
                rel_loss, rel_stats = compute_video_derived_displacement_loss(
                    z_tgt, u_vid, norm_weight=args.disp_norm_weight
                )
            if op_only:
                with torch.no_grad():
                    rel_stats["cka_op_res"] = compute_linear_cka_penalty(
                        b_t[:, :, : args.op_dim], b_t[:, :, args.op_dim :]
                    )

        total_loss = diff_loss + cur_lam * rel_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        ema.step(policy_params)

        acc_diff += float(diff_loss.item())
        acc_rel += float(rel_loss.item())
        acc_tot += float(total_loss.item())
        acc_n += 1

        if (step % args.eval_every == 0) or (step == total_steps):
            dt = time.monotonic() - wt0
            ema.store(policy_params)
            ema.copy_to(policy_params)
            do_ddim = (step == total_steps) or (step % (args.eval_every * 2) == 0)
            val_m = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=256,
                run_ddim=do_ddim,
                ddim_max_samples=512,
                run_timestep_check=False,
                val_seed=args.val_seed,
            )
            cur_val = float(val_m["val_self_diff_loss"])
            eligible = step >= max(args.rel_warmup_steps, args.eval_every)
            if (eligible and cur_val < best_val) or best_step == 0:
                best_val = cur_val
                best_step = step
                save_obj: Dict[str, Any] = {
                    "stage": f"e16_{args.run_name}",
                    "step": step,
                    "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                    "config": config,
                    "val_metrics": val_m,
                }
                if reuse_module is not None:
                    save_obj["reuse_module"] = {k: v.detach().cpu() for k, v in reuse_module.state_dict().items()}
                atomic_torch_save(save_obj, run_dir / "best.pt")
            if step == total_steps:
                save_obj = {
                    "stage": f"e16_{args.run_name}",
                    "step": step,
                    "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                    "config": config,
                    "val_metrics": val_m,
                }
                if reuse_module is not None:
                    save_obj["reuse_module"] = {k: v.detach().cpu() for k, v in reuse_module.state_dict().items()}
                atomic_torch_save(save_obj, run_dir / "last_ema.pt")
            ema.restore(policy_params)

            rec = {
                "step": step,
                "train": {
                    "diff_loss": acc_diff / max(acc_n, 1),
                    "rel_loss": acc_rel / max(acc_n, 1),
                    "total_loss": acc_tot / max(acc_n, 1),
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "steps_per_sec": round(acc_n / max(dt, 1e-6), 2),
                    "rel_stats": {k: float(v.item()) for k, v in rel_stats.items()},
                },
                "val": val_m,
            }
            history.append(rec)
            atomic_write_json(
                {"config": config, "best_step": best_step, "best_val_self_diff_loss": best_val, "history": history},
                run_dir / "metrics.json",
            )
            ddim_str = ""
            if "ddim" in val_m:
                dm = val_m["ddim"]["self"]
                ddim_str = f" | ddim_mse10={dm['action10_mse']:.5f} grip={dm['grip_sign_acc']*100:.2f}%"
            print(
                f"[E16:{args.run_name} step {step:5d}/{total_steps}] "
                f"tr_diff={rec['train']['diff_loss']:.5f} rel={rec['train']['rel_loss']:.5f} | "
                f"val_diff={cur_val:.5f} b_rms={val_m['val_self_b_rms']:.3f} | "
                f"{rec['train']['steps_per_sec']:.1f}it/s{ddim_str}",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_diff = acc_rel = acc_tot = 0.0
            acc_n = 0

    print(f"[E16:{args.run_name}] Done. Saved to {run_dir / 'last_ema.pt'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--rel-mode",
        type=str,
        required=True,
        choices=sorted(list(VID_DERIVED_MODES.keys()) + list(LEARNABLE_MODES.keys())),
    )
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    p.add_argument("--lam-ckpt", type=Path, default=OUTPUT_ROOT / "e16_d1_lam" / "last_ema.pt")
    p.add_argument("--lambda-rel", type=float, default=0.15)
    p.add_argument("--rel-warmup-steps", type=int, default=300)
    p.add_argument("--disp-norm-weight", type=float, default=0.25)
    p.add_argument("--op-dim", type=int, default=384)
    p.add_argument("--predict-motion", action="store_true")
    p.add_argument(
        "--coverage-mode",
        type=str,
        default="cross_demo_anti_scene",
        choices=["same_task_only", "cross_demo_unmasked", "cross_demo_anti_scene", "strict_cross_task", "cross_demo_strict_anti_scene"],
    )
    p.add_argument("--top-k-donors", type=int, default=None)
    p.add_argument("--noise-type", type=str, default="none", choices=["none", "real_cv", "rand_matched"])
    p.add_argument("--noise-bank-path", type=Path, default=None)
    p.add_argument("--p-noisy", type=float, default=0.50)
    p.add_argument("--noise-scale-mult", type=float, default=0.35)
    p.add_argument("--refresh-cv-every", type=int, default=1000)
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--motion-cache-dir", type=Path, default=MOTION_CACHE_DIR)
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
    run_e16_training(parse_args())


if __name__ == "__main__":
    main()
