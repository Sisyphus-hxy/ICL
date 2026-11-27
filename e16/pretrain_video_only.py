"""Regime 2 (E16-D): 100% Pure Video-Only Pretraining (ZERO actions, ZERO action-init).

Pretrains E9's exact `LocalBehaviorEncoder` (42.55M params, 6-layer Pre-LN Transformer,
4x768 = 3072D latent) COMPLETELY FROM SCRATCH on unlabeled video clips with ZERO action
supervision and ZERO action-supervised checkpoint initialization!

Supported pure video-only pretraining variants (all equal budget, default 6 full epochs =
32,184 steps on all 1,373,035 training clips, batch=256):
  - `lam_base`     (E16-D1): Pure LAM-style conditional state-change prediction
                             `F_psi(v_0, b_t) -> delta_v_{t+1:t+16}` + VICReg anti-collapse.
                             NO cross-demo relational supervision (equal-budget LAM baseline!).
  - `lam_rand_rel` (E16-D2): Pure LAM + Random cross-demo relation supervision on `z_op`.
  - `lam_vid_op`   (E16-D3): Pure LAM + Video-derived motion-flow displacement consistency
                             strictly on `z_op` (preserving functional `z_res` for full LAM
                             state-change prediction!).
  - `lam_learn_op` (E16-D4): Pure LAM + Learnable cross-demo behavior substitution,
                             reuse-value state-change prediction, counterfactual cycle
                             consistency & canonical displacement equivariance strictly on
                             `z_op` (preserving functional `z_res`!).
  - `lam_learn_tok`(E16-D4t): Token-wise learnable cross-demo substitution on `z_op`.

Also supports `--train-episode-fraction {0.1, 0.3, 1.0}` and `--coverage-mode` for
Requirement 5 (data-scale & cross-demo relation coverage studies).

Saves a clean checkpoint containing `local_encoder` state_dict (+ pretraining heads for
re-analysis of learned relation graphs `W_ij`!), with an explicit assertion that GT actions
are NEVER loaded or passed into any pretraining step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from e9.stage1_local_model import LocalBehaviorEncoder
from e12.eval_suite import _effective_rank_stats

from . import MOTION_CACHE_DIR, MOTION_FEAT_DIM
from .video_relations import (
    ConditionalStateChangePredictor,
    LearnableReuseRelationModule,
    RichPairedBatchSampler,
    VideoMotionLocalBank,
    compute_clip_state_change_targets,
    compute_vicreg_loss,
    compute_video_derived_displacement_loss,
    extract_video_relation_features,
)


class PureVideoLAMModel(nn.Module):
    """Purely video-only pretraining wrapper around E9's exact `LocalBehaviorEncoder`."""

    def __init__(
        self,
        *,
        variant: str = "lam_base",
        op_dim: int = 384,
        predict_motion: bool = False,
        coverage_mode: str = "cross_demo_anti_scene",
        top_k_donors: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = str(variant)
        self.op_dim = int(op_dim)
        self.predict_motion = bool(predict_motion)

        self.local_encoder = LocalBehaviorEncoder(
            embed_dim=EMBED_DIM,
            num_layers=6,
            num_heads=NUM_HEADS,
            ffn_dim=4 * EMBED_DIM,
            max_window_len=LOCAL_WINDOW_LEN,
            num_cameras=2,
            num_behavior_tokens=NUM_BEHAVIOR_TOKENS,
            dropout=dropout,
        )

        # Full-latent LAM conditional forward state-change predictor (uses full 3072D b_t)
        self.lam_predictor = ConditionalStateChangePredictor(
            z_dim=NUM_BEHAVIOR_TOKENS * EMBED_DIM,
            scene_dim=2 * EMBED_DIM,
            hidden_dim=1024,
            predict_motion=self.predict_motion,
        )
        # Compact dynamics bottleneck projection (used when exporting pretrained LAM relations for E16-A3!)
        self.lam_dyn_proj = nn.Sequential(
            nn.LayerNorm(NUM_BEHAVIOR_TOKENS * EMBED_DIM),
            nn.Linear(NUM_BEHAVIOR_TOKENS * EMBED_DIM, 256),
        )
        self.lam_dyn_pred = ConditionalStateChangePredictor(
            z_dim=256,
            scene_dim=2 * EMBED_DIM,
            hidden_dim=768,
            predict_motion=False,
        )

        # Optional learnable reuse-driven relation module (for E16-D4 / E16-D4t)
        self.reuse_module: Optional[LearnableReuseRelationModule] = None
        if variant in ("lam_learn_op", "lam_learn_tok", "lam_learn_full"):
            z_rel_d = (
                NUM_BEHAVIOR_TOKENS * EMBED_DIM
                if variant == "lam_learn_full"
                else NUM_BEHAVIOR_TOKENS * self.op_dim
            )
            self.reuse_module = LearnableReuseRelationModule(
                z_rel_dim=z_rel_d,
                num_tokens=NUM_BEHAVIOR_TOKENS,
                token_wise=(variant == "lam_learn_tok"),
                proj_dim=256,
                hidden_dim=1024,
                predict_motion=self.predict_motion,
                coverage_mode=coverage_mode,
                top_k_donors=top_k_donors,
            )

    def encode_local_behavior(
        self, clip_rgb: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.local_encoder(clip_rgb, clip_pad_mask=clip_pad_mask)

    @torch.no_grad()
    def encode_dynamics_latent(
        self, clip_rgb: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compact 256D video-only LAM dynamics latent (scene-invariant action bottleneck)."""
        b_t = self.local_encoder(clip_rgb, clip_pad_mask=clip_pad_mask)
        return self.lam_dyn_proj(b_t.float().reshape(b_t.shape[0], -1))

    def compute_lam_loss(
        self,
        b_t: torch.Tensor,
        clip_rgb: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
        clip_motion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = b_t.shape[0]
        tgts = compute_clip_state_change_targets(
            clip_rgb, clip_pad_mask, clip_motion if self.predict_motion else None
        )
        v0 = tgts["v0"]
        dv_tgt = tgts["delta_v_seg"]
        seg_valid = tgts["seg_valid"]
        mo_tgt = tgts.get("motion_seg")

        w = seg_valid[:, :, None, None]
        denom = w.sum().clamp_min(1.0)

        # Full 3072D state-change prediction
        pred_dv, pred_mo = self.lam_predictor(v0, b_t.float().reshape(B, -1))
        l_dv = ((pred_dv - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)
        l_mo = b_t.new_zeros(())
        if pred_mo is not None and mo_tgt is not None:
            l_mo = ((pred_mo - mo_tgt).square() * w).sum() / (denom * 2.0 * MOTION_FEAT_DIM)

        # Compact 256D bottleneck state-change prediction (gives crisp LAM action bottleneck!)
        u_dyn = self.lam_dyn_proj(b_t.float().reshape(B, -1))
        pred_dv_bot, _ = self.lam_dyn_pred(v0, u_dyn)
        l_bot = ((pred_dv_bot - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)

        # VICReg on full latent (and separate z_op/z_res variance protection)
        l_vic, l_var, l_cov = compute_vicreg_loss(b_t)
        total = l_dv + 0.5 * l_bot + 0.25 * l_mo + 0.05 * l_vic
        return total, {
            "lam_dv": l_dv.detach(),
            "lam_bot": l_bot.detach(),
            "lam_mo": l_mo.detach(),
            "lam_vic": l_vic.detach(),
            "lam_var": l_var.detach(),
        }


def load_pretrained_video_lam(ckpt_path: Path | str, device: torch.device | str = "cpu") -> PureVideoLAMModel:
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = PureVideoLAMModel(
        variant=cfg.get("variant", "lam_base"),
        op_dim=int(cfg.get("op_dim", 384)),
        predict_motion=bool(cfg.get("predict_motion", False)),
        coverage_mode=cfg.get("coverage_mode", "cross_demo_anti_scene"),
        top_k_donors=cfg.get("top_k_donors", None),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model.to(device)


@torch.inference_mode()
def evaluate_video_only_validation(
    model: PureVideoLAMModel,
    bank: VideoMotionLocalBank,
    device: torch.device,
    batch_size: int = 256,
    max_samples: int = 2048,
) -> Dict[str, Any]:
    """Pure video-only validation metrics (future visual delta R^2, erank, b_rms — ZERO actions!)."""
    model.eval()
    n = min(max_samples, bank.num_val_samples())
    latents = []
    dv_sse = 0.0
    dv_sst = 0.0
    dv_cnt = 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = bank.gather_rich_batch(
            bank.val_episodes[s:e], bank.val_frames[s:e], include_actions=False
        )
        assert batch["actions_10d"] is None
        b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])
        tgts = compute_clip_state_change_targets(batch["clip_rgb"], batch["clip_pad_mask"])
        pred_dv, _ = model.lam_predictor(tgts["v0"], b_t.float().reshape(len(b_t), -1))
        w = tgts["seg_valid"][:, :, None, None]
        dv_sse += float(((pred_dv - tgts["delta_v_seg"]).square() * w).sum().item())
        dv_sst += float((tgts["delta_v_seg"].square() * w).sum().item())
        dv_cnt += int(w.sum().item())
        latents.append(b_t.float().cpu().numpy())

    Z = np.concatenate(latents, axis=0)  # (N, 4, 768)
    er_full = _effective_rank_stats(Z.reshape(len(Z), -1), device=device)
    er_op = _effective_rank_stats(Z[:, :, : model.op_dim].reshape(len(Z), -1), device=device)
    er_res = _effective_rank_stats(Z[:, :, model.op_dim :].reshape(len(Z), -1), device=device)
    dv_mse = dv_sse / max(dv_cnt * 2 * 768, 1)
    dv_r2 = 1.0 - dv_sse / max(dv_sst, 1e-8)
    return {
        "val_samples": int(n),
        "val_dv_mse": float(dv_mse),
        "val_dv_r2": float(dv_r2),
        "val_b_rms": float(np.sqrt(np.mean(Z ** 2))),
        "erank_full": float(er_full["erank_energy"]),
        "erank_op": float(er_op["erank_energy"]),
        "erank_res": float(er_res["erank_energy"]),
    }


def run_pure_video_pretraining(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    use_motion = args.variant in ("lam_vid_op", "lam_learn_op", "lam_learn_tok", "lam_learn_full") or bool(args.predict_motion)
    t0 = time.monotonic()
    bank = VideoMotionLocalBank(
        e5_cache_dir=args.e5_cache_dir,
        dense_cache_dir=args.dense_cache_dir,
        motion_cache_dir=args.motion_cache_dir if use_motion else None,
        device=device,
        resident_on_device=bool(args.gpu_resident_cache),
        seed=args.seed,
        train_episode_fraction=args.train_episode_fraction,
        subset_seed=args.seed,
    )
    print(
        f"[E16-D Pretrain:{args.run_name}] ZERO-ACTION video bank loaded "
        f"(train_starts={bank.num_train_samples():,}, ep_frac={args.train_episode_fraction:.2f}, "
        f"has_motion={bank.has_motion}) in {time.monotonic()-t0:.1f}s",
        flush=True,
    )

    seed_everything(args.seed)
    model = PureVideoLAMModel(
        variant=args.variant,
        op_dim=args.op_dim,
        predict_motion=bool(args.predict_motion),
        coverage_mode=args.coverage_mode,
        top_k_donors=args.top_k_donors,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay
    )
    total_steps = int(args.total_steps)
    warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMAModel(
        model.parameters(),
        decay=args.ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    sampler = RichPairedBatchSampler(
        bank, within_ep_frac=0.5, seed=args.seed, include_actions=False
    )

    config = {
        "regime": "pure_video_only_pretraining_zero_actions",
        "run_name": args.run_name,
        "variant": args.variant,
        "init_mode": "from_scratch_zero_action_init",
        "action_supervision": False,
        "op_dim": int(args.op_dim),
        "lambda_rel": float(args.lambda_rel),
        "rel_warmup_steps": int(args.rel_warmup_steps),
        "predict_motion": bool(args.predict_motion),
        "coverage_mode": args.coverage_mode,
        "top_k_donors": args.top_k_donors,
        "train_episode_fraction": float(args.train_episode_fraction),
        "num_train_episodes": len(bank.kept_train_episode_ids),
        "num_train_starts": bank.num_train_samples(),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "clip_grad_norm": float(args.clip_grad_norm),
        "ema_decay": float(args.ema_decay),
        "seed": int(args.seed),
        "encoder_params": sum(p.numel() for p in model.local_encoder.parameters()),
        "total_pretrain_params": sum(p.numel() for p in model.parameters()),
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
    best_dv_mse = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []
    wt0 = time.monotonic()
    acc_lam = acc_rel = acc_tot = 0.0
    acc_n = 0

    while step < total_steps:
        model.train()
        batch = next(batch_iter)
        # Hard structural guarantee: ZERO actions ever touched in Regime 2 pretraining!
        assert batch["actions_10d"] is None, "Action tensor leaked into Pure Video-Only Pretraining!"
        step += 1

        cur_lam = (
            float(args.lambda_rel) * (step / float(args.rel_warmup_steps))
            if (args.rel_warmup_steps > 0 and step <= args.rel_warmup_steps)
            else float(args.lambda_rel)
        )

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])

        lam_loss, lam_stats = model.compute_lam_loss(
            b_t, batch["clip_rgb"], batch["clip_pad_mask"], batch["clip_motion"]
        )

        rel_loss = lam_loss.new_zeros(())
        rel_stats: Dict[str, torch.Tensor] = {}
        if cur_lam > 0.0 and args.variant != "lam_base":
            if args.variant == "lam_rand_rel":
                u_rand = extract_video_relation_features(
                    "random_rel",
                    batch["clip_rgb"],
                    batch["clip_pad_mask"],
                    ep_ids=batch["ep_ids"],
                    frame_ids=batch["frame_ids"],
                )
                rel_loss, rel_stats = compute_video_derived_displacement_loss(
                    b_t[:, :, : args.op_dim], u_rand
                )
            elif args.variant == "lam_vid_op":
                u_mot = extract_video_relation_features(
                    "motion_flow",
                    batch["clip_rgb"],
                    batch["clip_pad_mask"],
                    batch["clip_motion"],
                )
                rel_loss, rel_stats = compute_video_derived_displacement_loss(
                    b_t[:, :, : args.op_dim], u_mot
                )
            elif args.variant in ("lam_learn_op", "lam_learn_tok", "lam_learn_full"):
                assert model.reuse_module is not None
                op_d = None if args.variant == "lam_learn_full" else args.op_dim
                # Cycle consistency activated after initial forward-predictor warmup (step > 500)
                do_cyc = step > 500
                do_eq = step > 500
                rel_loss, rel_stats = model.reuse_module.forward_loss(
                    b_t,
                    batch["clip_rgb"],
                    batch["clip_pad_mask"],
                    batch["ep_ids"],
                    task_ids=batch["task_ids"],
                    clip_motion=batch["clip_motion"] if args.predict_motion else None,
                    encoder_fn=model.encode_local_behavior if do_cyc else None,
                    op_dim=op_d,
                    use_cycle=do_cyc,
                    use_equiv=do_eq,
                )

        total_loss = lam_loss + cur_lam * rel_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        ema.step(model.parameters())

        acc_lam += float(lam_loss.item())
        acc_rel += float(rel_loss.item())
        acc_tot += float(total_loss.item())
        acc_n += 1

        if (step % args.eval_every == 0) or (step == total_steps):
            dt = time.monotonic() - wt0
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            val_m = evaluate_video_only_validation(model, bank, device)
            is_best = val_m["val_dv_mse"] < best_dv_mse
            if is_best or best_step == 0:
                best_dv_mse = val_m["val_dv_mse"]
                best_step = step
                atomic_torch_save(
                    {
                        "stage": f"e16_d_{args.run_name}",
                        "step": step,
                        "local_encoder": {
                            k: v.detach().cpu() for k, v in model.local_encoder.state_dict().items()
                        },
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config,
                        "val_metrics": val_m,
                    },
                    run_dir / "best.pt",
                )
            if step == total_steps:
                atomic_torch_save(
                    {
                        "stage": f"e16_d_{args.run_name}",
                        "step": step,
                        "local_encoder": {
                            k: v.detach().cpu() for k, v in model.local_encoder.state_dict().items()
                        },
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config,
                        "val_metrics": val_m,
                    },
                    run_dir / "last_ema.pt",
                )
            ema.restore(model.parameters())

            rec = {
                "step": step,
                "train": {
                    "lam_loss": acc_lam / max(acc_n, 1),
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
                {"config": config, "best_step": best_step, "best_dv_mse": best_dv_mse, "history": history},
                run_dir / "metrics.json",
            )
            print(
                f"[E16-D:{args.run_name} step {step:5d}/{total_steps}] "
                f"lam={rec['train']['lam_loss']:.5f} rel={rec['train']['rel_loss']:.5f} | "
                f"val_dv_r2={val_m['val_dv_r2']:.4f} mse={val_m['val_dv_mse']:.5f} "
                f"erank(full/op/res)={val_m['erank_full']:.1f}/{val_m['erank_op']:.1f}/{val_m['erank_res']:.1f} | "
                f"{rec['train']['steps_per_sec']:.1f}it/s",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_lam = acc_rel = acc_tot = 0.0
            acc_n = 0

    print(f"[E16-D:{args.run_name}] Done. Saved pure video-only encoder to {run_dir / 'last_ema.pt'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["lam_base", "lam_rand_rel", "lam_vid_op", "lam_learn_op", "lam_learn_tok", "lam_learn_full"],
    )
    p.add_argument("--op-dim", type=int, default=384)
    p.add_argument("--lambda-rel", type=float, default=0.25)
    p.add_argument("--rel-warmup-steps", type=int, default=500)
    p.add_argument("--predict-motion", action="store_true")
    p.add_argument(
        "--coverage-mode",
        type=str,
        default="cross_demo_anti_scene",
        choices=["same_task_only", "cross_demo_unmasked", "cross_demo_anti_scene", "strict_cross_task", "cross_demo_strict_anti_scene"],
    )
    p.add_argument("--top-k-donors", type=int, default=None)
    p.add_argument("--train-episode-fraction", type=float, default=1.0)
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--motion-cache-dir", type=Path, default=MOTION_CACHE_DIR)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=32184)  # 6 full epochs of 1.373M clips!
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    run_pure_video_pretraining(parse_args())


if __name__ == "__main__":
    main()
