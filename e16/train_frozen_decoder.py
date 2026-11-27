"""Regime 2 (E16-D) Unified Frozen-Encoder Action Decoder Trainer & Label-Efficiency Evaluator.

Given ANY frozen `LocalBehaviorEncoder` checkpoint:
  - Pure video-only pretrained (`e16_d1_lam`, `e16_d2_rand_rel`, `e16_d3_vid_op`, `e16_d4_learn_op`,
    plus data-scale & coverage variants)
  - Untrained random-init frozen encoder (`--random-encoder`, lower bound)
  - Action-supervised E9 frozen encoder (`e9/outputs/stage1_local/best.pt`, upper bound)

Completely FREEZES `local_encoder` (`requires_grad=False`, `.eval()`), initializes a fresh
identical `ConditionalUnet1D` Action Decoder from scratch, and trains ONLY the Action Decoder
with pure DDPM-IP loss `L_diff` on `--action-fraction` (e.g. `0.01, 0.10, 1.00`) of training
episodes!

Output checkpoints (`last_ema.pt`, `best.pt`) are 100% `E9Stage1LocalPolicy`-compatible so
they drop straight into `e15/eval_e15.py` for full 6-axis evaluation (Fidelity, pure `z_op`
Editability, 5x5 Composability, Predictability & 3-component residual, Decoder Robustness,
and strictly Zero-Leakage Visual ICL)!
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch

from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import (
    DEFAULT_DENSE_CACHE_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_HEADS,
)
from e9.stage1_local_model import E9Stage1LocalPolicy
from e9.train_stage1_local import evaluate_stage1_local_validation

from .video_relations import VideoMotionLocalBank


def _load_encoder_weights(policy: E9Stage1LocalPolicy, ckpt_path: Path) -> str:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "local_encoder" in payload:
        policy.local_encoder.load_state_dict(payload["local_encoder"], strict=True)
        return f"pure_video_pretrain:{ckpt_path}"
    if "model" in payload:
        enc_sd = {
            k[len("local_encoder.") :]: v
            for k, v in payload["model"].items()
            if k.startswith("local_encoder.")
        }
        policy.local_encoder.load_state_dict(enc_sd, strict=True)
        return f"policy_encoder:{ckpt_path}"
    raise ValueError(f"Unrecognized encoder checkpoint format: {ckpt_path}")


def run_frozen_decoder_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    t0 = time.monotonic()
    bank = VideoMotionLocalBank(
        e5_cache_dir=args.e5_cache_dir,
        dense_cache_dir=args.dense_cache_dir,
        motion_cache_dir=None,
        device=device,
        resident_on_device=bool(args.gpu_resident_cache),
        seed=args.seed,
        train_episode_fraction=args.action_fraction,
        subset_seed=args.subset_seed,
    )
    print(
        f"[E16-D Decoder:{args.run_name}] Loaded bank (action_frac={args.action_fraction:.3f}, "
        f"train_eps={len(bank.kept_train_episode_ids):,}, train_starts={bank.num_train_samples():,}) "
        f"in {time.monotonic()-t0:.1f}s",
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

    if args.random_encoder:
        enc_src = "random_untrained_encoder"
    else:
        if args.encoder_ckpt is None or not Path(args.encoder_ckpt).is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {args.encoder_ckpt}")
        enc_src = _load_encoder_weights(policy, Path(args.encoder_ckpt))

    # Strictly freeze encoder!
    for p in policy.local_encoder.parameters():
        p.requires_grad_(False)
    policy.local_encoder.eval()

    dec_params = [p for p in policy.action_decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        policy.get_optim_groups(lr=args.lr, weight_decay=args.weight_decay),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=args.weight_decay,
    )

    total_steps = int(args.total_steps)
    warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMAModel(
        dec_params,
        decay=args.ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    config = {
        "stage": f"e16_d_frozen_decoder_{args.run_name}",
        "run_name": args.run_name,
        "encoder_source": enc_src,
        "freeze_encoder": True,
        "action_fraction": float(args.action_fraction),
        "num_labeled_episodes": len(bank.kept_train_episode_ids),
        "num_labeled_starts": bank.num_train_samples(),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "clip_grad_norm": float(args.clip_grad_norm),
        "ema_decay": float(args.ema_decay),
        "seed": int(args.seed),
        "subset_seed": int(args.subset_seed),
        "val_seed": int(args.val_seed),
        "parameter_summary": policy.parameter_summary(),
    }
    atomic_write_json(config, run_dir / "config.json")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def _inf_batches() -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ep = 1
        while True:
            yield from bank.iter_train_batches(ep, args.batch_size)
            ep += 1

    batch_iter = _inf_batches()
    step = 0
    best_val = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []
    wt0 = time.monotonic()
    acc_diff = 0.0
    acc_n = 0

    while step < total_steps:
        policy.local_encoder.eval()
        policy.action_decoder.train()
        clip_rgb, clip_pad_mask, actions_10d = next(batch_iter)
        step += 1

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            with torch.no_grad():
                b_t = policy.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad_mask)
            diff_loss = policy.diffusion_loss_from_z_self(b_t, actions_10d).mean()

        diff_loss.backward()
        torch.nn.utils.clip_grad_norm_(dec_params, args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        ema.step(dec_params)

        acc_diff += float(diff_loss.item())
        acc_n += 1

        if (step % args.eval_every == 0) or (step == total_steps):
            dt = time.monotonic() - wt0
            ema.store(dec_params)
            ema.copy_to(dec_params)
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
            if cur_val < best_val or best_step == 0:
                best_val = cur_val
                best_step = step
                atomic_torch_save(
                    {
                        "stage": f"e16_d_dec_{args.run_name}",
                        "step": step,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_m,
                    },
                    run_dir / "best.pt",
                )
            if step == total_steps:
                atomic_torch_save(
                    {
                        "stage": f"e16_d_dec_{args.run_name}",
                        "step": step,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_m,
                    },
                    run_dir / "last_ema.pt",
                )
            ema.restore(dec_params)

            rec = {
                "step": step,
                "train": {
                    "diff_loss": acc_diff / max(acc_n, 1),
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "steps_per_sec": round(acc_n / max(dt, 1e-6), 2),
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
                f"[E16-D Dec:{args.run_name} step {step:5d}/{total_steps}] "
                f"tr_diff={rec['train']['diff_loss']:.5f} | val_diff={cur_val:.5f} | "
                f"{rec['train']['steps_per_sec']:.1f}it/s{ddim_str}",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_diff = 0.0
            acc_n = 0

    print(
        f"[E16-D Dec:{args.run_name}] Done. best_val={best_val:.5f} @step {best_step}, saved to {run_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--encoder-ckpt", type=Path, default=None)
    p.add_argument("--random-encoder", action="store_true")
    p.add_argument("--action-fraction", type=float, default=1.0)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=10728)  # 2 full epochs for converged unified decoder!
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-6)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=12345)
    return p.parse_args()


def main() -> None:
    run_frozen_decoder_training(parse_args())


if __name__ == "__main__":
    main()
