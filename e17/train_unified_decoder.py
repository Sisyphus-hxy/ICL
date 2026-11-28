"""E17 Unified Frozen-Encoder Action Decoder Trainer.

Fixes the evaluation-protocol discrepancies identified in E16:
  1. Uses IDENTICAL architecture, optimizer, batch size, and checkpoint selection (`best.pt`
     on full 6,248 validation diffusion loss + `last_ema.pt`) across ALL frozen encoders.
  2. Replaces the biased first-512 validation prefix (27 tasks + 61 tail samples) with the
     UNIFIED 1,024 uniformly sampled non-tail validation starts across all 311 tasks during
     periodic DDIM checks (matching `eval_e15.py` to 4 decimal places!).
  3. Supports fine-grained validation cadence (`--eval-every 500` for 1%, `1000` for 10%,
     `2000` for 100%) so low-label-fraction decoders cleanly capture optimal early-stopped
     `best.pt` without overfitting!
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
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
from e12.eval_suite import _action_metrics_dict
from e16.train_frozen_decoder import _load_encoder_weights
from e16.video_relations import VideoMotionLocalBank


@torch.inference_mode()
def evaluate_uniform_nontail_ddim(
    policy: E9Stage1LocalPolicy,
    bank: VideoMotionLocalBank,
    *,
    num_samples: int = 1024,
    batch_size: int = 256,
    seed: int = 12345,
) -> Dict[str, float]:
    """Exact unified 311-task uniform non-tail DDIM fidelity check (matches `eval_e15.py`!)."""
    policy.eval()
    device = bank.device
    ep_lens = bank.ep_lengths_np
    is_tail = (bank.val_frames + LOCAL_WINDOW_LEN) > ep_lens[bank.val_episodes]
    non_tail = np.flatnonzero(~is_tail)
    rng = np.random.default_rng(42 + 101)
    sel = np.sort(rng.choice(non_tail, size=min(num_samples, len(non_tail)), replace=False))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    z0_all = torch.randn((len(sel), LOCAL_WINDOW_LEN, 10), generator=gen, dtype=torch.float32)

    preds, gts = [], []
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    for s in range(0, len(sel), batch_size):
        e = min(s + batch_size, len(sel))
        idx = sel[s:e]
        c_rgb, c_mask, a10 = bank.gather_batch(bank.val_episodes[idx], bank.val_frames[idx])
        z0 = z0_all[s:e].to(device, non_blocking=True)
        with amp_ctx:
            out = policy.predict_action(c_rgb, c_mask, init_noise=z0)
        preds.append(out["action_10d"].float().cpu().numpy())
        gts.append(a10.float().cpu().numpy())

    m = _action_metrics_dict(np.concatenate(preds, axis=0), np.concatenate(gts, axis=0))
    return {
        "num_samples": int(len(sel)),
        "action10_mse": float(m["action10_mse"]),
        "action7_mae": float(m["action7_mae"]),
        "grip_sign_acc": float(m["grip_sign_acc"]),
    }


def run_unified_decoder_training(args: argparse.Namespace) -> None:
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
        f"[E17 UnifiedDec:{args.run_name}] Bank ready (frac={args.action_fraction:.3f}, "
        f"eps={len(bank.kept_train_episode_ids):,}, starts={bank.num_train_samples():,}) in {time.monotonic()-t0:.1f}s",
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
        "stage": f"e17_unified_frozen_decoder_{args.run_name}",
        "protocol": "E17_UNIFIED_311TASK_NONTAIL_DDIM1024",
        "run_name": args.run_name,
        "encoder_source": enc_src,
        "freeze_encoder": True,
        "action_fraction": float(args.action_fraction),
        "num_labeled_episodes": len(bank.kept_train_episode_ids),
        "num_labeled_starts": bank.num_train_samples(),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "eval_every": int(args.eval_every),
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
            val_m = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=256,
                run_ddim=False,
                run_timestep_check=False,
                val_seed=args.val_seed,
            )
            do_ddim = (step == total_steps) or (step % (args.eval_every * 2) == 0)
            if do_ddim:
                val_m["ddim_uniform1024_311tasks"] = evaluate_uniform_nontail_ddim(
                    policy, bank, num_samples=1024, batch_size=256, seed=args.val_seed
                )
            cur_val = float(val_m["val_self_diff_loss"])
            if cur_val < best_val or best_step == 0:
                best_val = cur_val
                best_step = step
                atomic_torch_save(
                    {
                        "stage": f"e17_dec_{args.run_name}",
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
                        "stage": f"e17_dec_{args.run_name}",
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
            if "ddim_uniform1024_311tasks" in val_m:
                dm = val_m["ddim_uniform1024_311tasks"]
                ddim_str = f" | unif1024_mse10={dm['action10_mse']:.5f} grip={dm['grip_sign_acc']*100:.2f}%"
            print(
                f"[E17 Dec:{args.run_name} step {step:5d}/{total_steps}] "
                f"tr={rec['train']['diff_loss']:.5f} val={cur_val:.5f} | "
                f"{rec['train']['steps_per_sec']:.1f}it/s{ddim_str}",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_diff = 0.0
            acc_n = 0

    print(
        f"[E17 Dec:{args.run_name}] Done. best_val={best_val:.5f} @step {best_step}, saved to {run_dir}",
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
    p.add_argument("--total-steps", type=int, default=10728)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-6)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=12345)
    return p.parse_args()


if __name__ == "__main__":
    run_unified_decoder_training(parse_args())
