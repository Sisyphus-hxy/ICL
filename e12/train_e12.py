"""Training runner for E12: Relational Behavior Representation Learning.

Keeps E9 Stage 1's exact LocalBehaviorEncoder, ConditionalUnet1D, latent dimension
`(B, 4, 768)`, and original DDPM-IP action reconstruction objective `L_diff`,
adding ONE clean relational objective per variant:
  - `control` (E9-Cont): Pure `L_diff` (matched-budget control)
  - `e12_a`   (E12-A):   `L_diff + lambda_rel * L_global_action_sim` (arXiv:2609.19846)
  - `e12_b`   (E12-B):   `L_diff + lambda_rel * L_multi_attr_token_rel`
  - `e12_c`   (E12-C):   `L_diff + lambda_rel * L_behavior_displacement`

Memory-safe design for shared RTX 6000D GPUs (25-28 GB free):
  - Keeps dense CLIP `frame_features` in CPU RAM (`resident_on_device=False`),
    gathering batches via vectorized CPU indexing + non-blocking H2D transfer
    (<5 GB GPU VRAM per run!).
  - Supports paired transition batch sampling (`--paired-transitions`) for E12-C
    (50% within-episode temporal transitions + 50% cross-episode transitions) so
    every mini-batch contains diverse real behavioral changes from ~128 scenes.
  - Supports warm-start from `e9/outputs/stage1_local/best.pt` (default) or
    from-scratch training, with identical EMA, cosine LR, gradient clipping, and
    best-checkpoint selection on original val split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import action_10d_to_7d
from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
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

from . import OUTPUT_ROOT
from .losses import compute_relational_loss


class PairedTransitionBatchSampler:
    """Yields batches of size `B = 2*M` where consecutive pairs `(2m, 2m+1)` form
    behavioral transitions: `within_ep_frac` within-episode non-overlapping transitions
    (`|t_2 - t_1| >= 16`) and `1 - within_ep_frac` cross-episode transitions."""

    def __init__(
        self,
        bank: LocalClipFeatureBank,
        within_ep_frac: float = 0.5,
        seed: int = 42,
    ):
        self.bank = bank
        self.within_ep_frac = float(within_ep_frac)
        self.seed = int(seed)

        # Group train start indices by episode for O(1) within-episode partner sampling
        ep_groups: Dict[int, List[int]] = defaultdict(list)
        for idx, ep in enumerate(bank.train_episodes):
            ep_groups[int(ep)].append(idx)

        self.multi_eps: List[np.ndarray] = []
        self.multi_frames: List[np.ndarray] = []
        for ep, idxs in ep_groups.items():
            if len(idxs) >= 2:
                arr = np.array(idxs, dtype=np.int64)
                frs = bank.train_frames[arr]
                if int(frs.max() - frs.min()) >= 16:
                    self.multi_eps.append(arr)
                    self.multi_frames.append(frs)
        self.num_multi_eps = len(self.multi_eps)

    def iter_batches(
        self,
        epoch: int,
        batch_size: int,
        max_steps: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if batch_size % 2 != 0:
            raise ValueError(f"Paired batch_size must be even, got {batch_size}")
        M = batch_size // 2
        n_within = int(round(M * self.within_ep_frac))
        n_cross = M - n_within

        rng = np.random.default_rng([self.seed, int(epoch), 12012])
        n_train = self.bank.num_train_samples()
        total_steps = math.ceil(n_train / batch_size)
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        for _ in range(total_steps):
            idx_a = np.empty(M, dtype=np.int64)
            idx_b = np.empty(M, dtype=np.int64)

            # 1. Within-episode transitions (distinct scenes across pairs)
            if n_within > 0:
                ep_choices = rng.choice(
                    self.num_multi_eps, size=n_within, replace=(n_within > self.num_multi_eps)
                )
                for k, e_i in enumerate(ep_choices):
                    arr = self.multi_eps[e_i]
                    frs = self.multi_frames[e_i]
                    i0 = int(rng.integers(len(arr)))
                    f0 = int(frs[i0])
                    far = np.flatnonzero(np.abs(frs - f0) >= 16)
                    i1 = int(far[rng.integers(len(far))]) if len(far) > 0 else int(rng.integers(len(arr)))
                    idx_a[k] = arr[i0]
                    idx_b[k] = arr[i1]

            # 2. Cross-episode transitions
            if n_cross > 0:
                idx_a[n_within:] = rng.integers(0, n_train, size=n_cross)
                idx_b[n_within:] = rng.integers(0, n_train, size=n_cross)

            # Interleave [a_0, b_0, a_1, b_1, ...]
            batch_idx = np.empty(batch_size, dtype=np.int64)
            batch_idx[0::2] = idx_a
            batch_idx[1::2] = idx_b

            eps = self.bank.train_episodes[batch_idx]
            frs = self.bank.train_frames[batch_idx]
            yield self.bank.gather_batch(eps, frs)


def run_e12_training(args: argparse.Namespace) -> None:
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
        f"[E12:{args.variant}] Loaded LocalClipFeatureBank "
        f"(gpu_resident={args.gpu_resident_cache}) in {time.monotonic()-t_load0:.1f}s",
        flush=True,
    )

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

    if args.init_ckpt is not None and Path(args.init_ckpt).is_file():
        payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["model"], strict=True)
        init_mode = f"warm_start:{args.init_ckpt}"
        print(f"[E12:{args.variant}] Warm-started from {args.init_ckpt}", flush=True)
    else:
        init_mode = "from_scratch"
        print(f"[E12:{args.variant}] Initialized from scratch", flush=True)

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
        policy.parameters(),
        decay=args.ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    use_paired = bool(args.paired_transitions or args.variant == "e12_c")
    paired_sampler = (
        PairedTransitionBatchSampler(
            bank, within_ep_frac=args.within_ep_frac, seed=args.seed
        )
        if use_paired
        else None
    )

    config = {
        "stage": f"e12_{args.variant}",
        "variant": args.variant,
        "init_mode": init_mode,
        "lambda_rel": float(args.lambda_rel),
        "rel_warmup_steps": int(args.rel_warmup_steps),
        "disp_norm_weight": float(args.disp_norm_weight),
        "paired_transitions": use_paired,
        "within_ep_frac": float(args.within_ep_frac),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "eval_every": eval_every,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lr_warmup_proportion": float(args.lr_warmup_proportion),
        "warmup_steps": warmup_steps,
        "clip_grad_norm": float(args.clip_grad_norm),
        "ema_decay": float(args.ema_decay),
        "embed_dim": EMBED_DIM,
        "window_len": LOCAL_WINDOW_LEN,
        "num_behavior_tokens": NUM_BEHAVIOR_TOKENS,
        "encoder_layers": args.encoder_layers,
        "encoder_heads": NUM_HEADS,
        "encoder_ffn_dim": 4 * EMBED_DIM,
        "dropout": args.dropout,
        "seed": args.seed,
        "val_seed": args.val_seed,
        "parameter_summary": policy.parameter_summary(),
    }
    atomic_write_json(config, run_dir / "config.json")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Initial reference validation
    init_val = evaluate_stage1_local_validation(
        policy,
        bank,
        batch_size=args.val_batch_size,
        max_val_samples=args.max_val_samples,
        run_ddim=True,
        ddim_max_samples=args.sample_count,
        run_timestep_check=False,
        val_seed=args.val_seed,
    )
    atomic_write_json(init_val, run_dir / "init_val.json")
    print(
        f"[E12:{args.variant} Init] val_self_diff={init_val['val_self_diff_loss']:.5f} | "
        f"ddim_10d_mse={init_val['ddim']['self']['action10_mse']:.5f} | "
        f"7d_mae={init_val['ddim']['self']['action7_mae']:.5f} | "
        f"grip_acc={init_val['ddim']['self']['grip_sign_acc']*100:.2f}%",
        flush=True,
    )

    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_step = 0

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
    window_t0 = time.monotonic()
    acc_diff = 0.0
    acc_rel = 0.0
    acc_total = 0.0
    acc_gnorm = 0.0
    acc_steps = 0

    while global_step < total_steps:
        policy.train()
        clip_rgb, clip_pad_mask, actions_10d = next(batch_iter)
        global_step += 1

        # Optional linear ramp for lambda_rel
        if args.rel_warmup_steps > 0 and global_step <= args.rel_warmup_steps:
            cur_lam = float(args.lambda_rel) * (global_step / float(args.rel_warmup_steps))
        else:
            cur_lam = float(args.lambda_rel)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = policy.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad_mask)
            diff_loss = policy.diffusion_loss_from_z_self(b_t, actions_10d).mean()

        if args.variant != "control" and cur_lam > 0.0:
            # Compute relational loss in float32 for clean cosine/RBF numerics
            a7 = action_10d_to_7d(actions_10d.float())
            rel_loss, rel_stats = compute_relational_loss(
                args.variant,
                b_t.float(),
                actions_10d.float(),
                a7,
                disp_norm_weight=args.disp_norm_weight,
            )
            total_loss = diff_loss + cur_lam * rel_loss
        else:
            rel_loss = diff_loss.new_zeros(())
            rel_stats = {"rel_loss": rel_loss.detach()}
            total_loss = diff_loss

        total_loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.clip_grad_norm
            ).item()
        )
        optimizer.step()
        lr_scheduler.step()
        ema.step(policy.parameters())

        acc_diff += float(diff_loss.item())
        acc_rel += float(rel_loss.item())
        acc_total += float(total_loss.item())
        acc_gnorm += grad_norm
        acc_steps += 1

        is_eval_step = (global_step % eval_every == 0) or (global_step == total_steps)
        if is_eval_step:
            dt = time.monotonic() - window_t0
            tr_rec = {
                "diff_loss": acc_diff / max(acc_steps, 1),
                "rel_loss": acc_rel / max(acc_steps, 1),
                "total_loss": acc_total / max(acc_steps, 1),
                "lambda_rel": cur_lam,
                "grad_norm_mean": acc_gnorm / max(acc_steps, 1),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "steps_per_sec": round(acc_steps / max(dt, 1e-6), 2),
                "window_seconds": round(dt, 2),
                "rel_stats_last": {k: float(v.item()) for k, v in rel_stats.items()},
            }
            if device.type == "cuda":
                tr_rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)

            # Evaluate EMA weights
            ema.store(policy.parameters())
            ema.copy_to(policy.parameters())
            do_ddim = (global_step == total_steps) or (global_step % (eval_every * 2) == 0)
            val_metrics = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=do_ddim,
                ddim_max_samples=args.sample_count,
                run_timestep_check=False,
                val_seed=args.val_seed,
            )

            # Selection: for control use val_self_diff_loss; for E12 variants we also save last
            # AND best by val_self_diff_loss after relational warmup completes so best.pt reflects
            # the relational objective (not step 0 before rel_loss acted).
            eligible = global_step >= max(args.rel_warmup_steps, eval_every)
            cur_val = val_metrics["val_self_diff_loss"]
            is_best = eligible and (cur_val < best_val_loss)
            if is_best or best_step == 0:
                best_val_loss = cur_val
                best_step = global_step
                atomic_torch_save(
                    {
                        "stage": f"e12_{args.variant}",
                        "variant": args.variant,
                        "step": global_step,
                        "best_val_self_diff_loss": best_val_loss,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    best_ckpt,
                )

            # Always save latest EMA weights at final step as last_ema.pt too
            if global_step == total_steps:
                atomic_torch_save(
                    {
                        "stage": f"e12_{args.variant}",
                        "variant": args.variant,
                        "step": global_step,
                        "val_self_diff_loss": cur_val,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    run_dir / "last_ema.pt",
                )

            ema.restore(policy.parameters())

            rec = {
                "step": global_step,
                "best_step": best_step,
                "best_val_self_diff_loss": best_val_loss,
                "train": tr_rec,
                "val": val_metrics,
            }
            history.append(rec)
            atomic_write_json(
                {
                    "variant": args.variant,
                    "best_step": best_step,
                    "best_val_self_diff_loss": best_val_loss,
                    "init_val": init_val,
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
                f"[E12:{args.variant} step {global_step:5d}/{total_steps}] "
                f"tr_diff={tr_rec['diff_loss']:.5f} rel={tr_rec['rel_loss']:.5f} "
                f"(lam={cur_lam:.3f}) | val_diff={cur_val:.5f} "
                f"sens={val_metrics['val_clip_sensitivity_diff_loss']:+.4f} "
                f"b_rms={val_metrics['val_self_b_rms']:.3f} | "
                f"lr={tr_rec['lr']:.2e} {tr_rec['steps_per_sec']:.1f}it/s"
                f"{'*' if is_best else ''}{ddim_str}",
                flush=True,
            )

            window_t0 = time.monotonic()
            acc_diff = acc_rel = acc_total = acc_gnorm = 0.0
            acc_steps = 0

    atomic_torch_save(
        {
            "stage": f"e12_{args.variant}",
            "variant": args.variant,
            "step": global_step,
            "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "config": config,
        },
        last_ckpt,
    )
    print(
        f"[E12:{args.variant}] Training complete. best_step={best_step}, "
        f"best_val_self_diff={best_val_loss:.5f}, saved to {run_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["control", "e12_a", "e12_b", "e12_c"],
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--encoder-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=6000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--lambda-rel", type=float, default=0.05)
    p.add_argument("--rel-warmup-steps", type=int, default=300)
    p.add_argument("--disp-norm-weight", type=float, default=0.25)
    p.add_argument("--paired-transitions", action="store_true")
    p.add_argument("--within-ep-frac", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-6)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.3)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--sample-count", type=int, default=512)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=12345)
    args = p.parse_args()
    if args.from_scratch:
        args.init_ckpt = None
    return args


def main() -> None:
    args = parse_args()
    run_e12_training(args)


if __name__ == "__main__":
    main()
