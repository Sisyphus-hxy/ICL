"""E17 Pure Video-Only Pretraining: Complementary D3/D4 Fusion, Sharp Authentic Reuse,
Compact Manifold Shaping, Equal-Param Controls, and Controlled Scaling / Coverage Matrix.

Hard scientific discipline:
  - 100% ZERO GT action supervision & ZERO action-supervised init (`LocalBehaviorEncoder`
    42.55M params initialized from scratch; `assert batch['actions_10d'] is None` every step!).
  - Strictly distinguishes Pure Video (`cross_demo_unmasked` / `cross_demo_anti_scene`, using
    only `ep_ids` + RGB) vs Task-Metadata-Assisted (`strict_cross_task` / `same_task_only`).
  - Supports intermediate checkpoint snapshots (`--save-steps 8046,16092,32184`) for exact
    training-step scaling analysis without extra runs.

Supported variants:
  - `lam_base`           : Pure LAM baseline (E16-D1 reproduction at any scale/steps)
  - `lam_vid_op`         : VidRel-Op (E16-D3 reproduction at any scale/steps)
  - `lam_vid_op_eqparam` : Equal-pretrain-parameter (~82.5M) VidRel-Op strict control!
  - `lam_learn_op`       : LearnReuse-Op (E16-D4 reproduction at any scale/steps/coverage)
  - `e17_m1_joint_op`    : Direct complementary fusion of VidRel + Sharp Authentic LearnReuse
                           (top-K=16 + straight-through Top-1 reuse) on shared `z_op` (384D/tok)
  - `e17_m2_compact_op`  : Compact-Rank complementary fusion (VidRel + Sharp LearnReuse on `z_op`
                           + compact 256D LAM bottleneck + gentle `z_res` energy regularization,
                           eliminating E16's 3072D isotropic VICReg rank explosion!)
  - `e17_m3_dual_orth`   : Orthogonal Dual-Factor `z_op` (`z_kin` [0:192] <- VidRel,
                           `z_reuse` [192:384] <- Sharp LearnReuse, CKA(kin, reuse) decorr,
                           full `z_op` fed into forward predictor + compact LAM bottleneck)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
from e15.train_e15 import compute_linear_cka_penalty
from e16 import MOTION_CACHE_DIR, MOTION_FEAT_DIM
from e16.video_relations import (
    ConditionalStateChangePredictor,
    LearnableReuseRelationModule,
    RichPairedBatchSampler,
    VideoMotionLocalBank,
    compute_clip_state_change_targets,
    compute_vicreg_loss,
    compute_video_derived_displacement_loss,
    extract_video_relation_features,
)


class SharpLearnableReuseModule(LearnableReuseRelationModule):
    """Extends `LearnableReuseRelationModule` with anti-averaging sharp top-K + Top-1 reuse loss.

    Addresses the convex-hull soft-averaging cheat discovered in E17 Exp 1 by:
      1. Restricting attention to `top_k_donors` (default 16) with sharper initial temperature (0.07).
      2. Adding a straight-through Top-1 hard donor substitution loss (`l_reuse_top1`) so every
         individual top-ranked cross-demo donor must genuinely predict target state change!
      3. Supporting softer variance-only VICReg (`cov_weight` configurable) to prevent rank bloat.
    """

    def __init__(
        self,
        z_rel_dim: int,
        *,
        num_tokens: int = 4,
        token_wise: bool = False,
        proj_dim: int = 256,
        hidden_dim: int = 1024,
        predict_motion: bool = True,
        coverage_mode: str = "cross_demo_unmasked",
        top_k_donors: Optional[int] = 16,
        init_temp: float = 0.07,
        top1_reuse_weight: float = 0.40,
        vic_std_target: float = 0.55,
        vic_cov_weight: float = 0.005,
        z_pred_dim: Optional[int] = None,
    ):
        # Allow forward_pred to take full z_op (z_pred_dim) even when W_ij routes on a sub-slice (z_rel_dim)!
        super().__init__(
            z_rel_dim=z_rel_dim,
            num_tokens=num_tokens,
            token_wise=token_wise,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            predict_motion=predict_motion,
            coverage_mode=coverage_mode,
            top_k_donors=top_k_donors,
            init_temp=init_temp,
        )
        self.top1_reuse_weight = float(top1_reuse_weight)
        self.vic_std_target = float(vic_std_target)
        self.vic_cov_weight = float(vic_cov_weight)
        self.z_pred_dim = int(z_pred_dim) if z_pred_dim is not None else int(z_rel_dim)
        if self.z_pred_dim != int(z_rel_dim):
            self.forward_pred = ConditionalStateChangePredictor(
                z_dim=self.z_pred_dim,
                scene_dim=2 * 768,
                hidden_dim=hidden_dim,
                predict_motion=predict_motion,
            )

    def forward_sharp_loss(
        self,
        z_route: torch.Tensor,
        z_pred: torch.Tensor,
        clip_rgb: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
        ep_ids: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor] = None,
        clip_motion: Optional[torch.Tensor] = None,
        encoder_fn: Optional[Any] = None,
        extract_pred_fn: Optional[Any] = None,
        use_cycle: bool = True,
        use_equiv: bool = True,
        cycle_weight: float = 0.15,
        equiv_weight: float = 0.35,
        bal_weight: float = 0.05,
        vic_weight: float = 0.03,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = z_route.shape[0]
        zr = z_route.float()
        zp_flat = z_pred.float().reshape(B, -1)

        tgts = compute_clip_state_change_targets(clip_rgb, clip_pad_mask, clip_motion)
        v0 = tgts["v0"]
        dv_tgt = tgts["delta_v_seg"]
        seg_valid = tgts["seg_valid"]
        mo_tgt = tgts.get("motion_seg")

        valid_mask, s_scene = self._build_donor_mask(v0, ep_ids, task_ids=task_ids)
        W, _, l_bal = self.compute_relation_weights(zr, valid_mask)
        W_eff = W.mean(dim=0) if W.ndim == 3 else W  # (B, B)

        # Soft substituted donor behavior in z_pred space
        z_tilde_soft = W_eff @ zp_flat  # (B, z_pred_dim)

        # Hard Top-1 straight-through substituted donor behavior (eliminates convex-hull averaging cheat!)
        top1_idx = torch.argmax(W_eff, dim=-1)
        one_hot = F.one_hot(top1_idx, num_classes=B).to(dtype=W_eff.dtype)
        W_st = one_hot + (W_eff - W_eff.detach())
        z_tilde_top1 = W_st @ zp_flat

        # 1. Self state-change prediction
        pred_dv_self, pred_mo_self = self.forward_pred(v0, zp_flat)
        l_self = self._masked_pred_loss(pred_dv_self, dv_tgt, seg_valid, pred_mo_self, mo_tgt)

        # 2a. Soft top-K cross-demo reuse loss
        pred_dv_soft, pred_mo_soft = self.forward_pred(v0, z_tilde_soft)
        l_reuse_soft = self._masked_pred_loss(pred_dv_soft, dv_tgt, seg_valid, pred_mo_soft, mo_tgt)

        # 2b. Hard Top-1 straight-through single-donor reuse loss
        pred_dv_top1, pred_mo_top1 = self.forward_pred(v0, z_tilde_top1)
        l_reuse_top1 = self._masked_pred_loss(pred_dv_top1, dv_tgt, seg_valid, pred_mo_top1, mo_tgt)

        # 3. Counterfactual cross-scene cycle consistency
        l_cycle = zp_flat.new_zeros(())
        if use_cycle and encoder_fn is not None and extract_pred_fn is not None:
            n_cyc = min(B, 64)
            with torch.no_grad():
                v_sub = valid_mask[:n_cyc]
                probs = v_sub.float() / v_sub.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                donor_j = torch.multinomial(probs, num_samples=1).squeeze(-1)
            v0_sub = v0[:n_cyc]
            z_donor = zp_flat[donor_j].detach()
            dv_cf, _ = self.forward_pred(v0_sub, z_donor)
            v_cf = v0_sub[:, None, :, :] + dv_cf.repeat_interleave(4, dim=1)
            b_cf = encoder_fn(v_cf.to(dtype=clip_rgb.dtype), None)
            z_cf = extract_pred_fn(b_cf).float().reshape(n_cyc, -1)
            l_cycle = (1.0 - F.cosine_similarity(z_cf, z_donor, dim=-1)).mean()

        # 4. Self-supervised displacement equivariance via scene-canonicalized forward effects
        l_equiv = zp_flat.new_zeros(())
        if use_equiv and B >= 8:
            with torch.no_grad():
                ref_idx = torch.linspace(0, B - 1, 4, device=zp_flat.device).long()
                eff_list = []
                for r in ref_idx:
                    v_ref = v0[r : r + 1].expand(B, -1, -1)
                    e_dv, e_mo = self.forward_pred(v_ref, zp_flat.detach())
                    e_vec = e_dv.reshape(B, -1)
                    if e_mo is not None:
                        e_dv_n = e_vec / e_vec.square().mean().sqrt().clamp_min(1e-5)
                        e_mo_n = e_mo.reshape(B, -1) / e_mo.square().mean().sqrt().clamp_min(1e-5)
                        e_vec = torch.cat([e_dv_n, e_mo_n], dim=-1)
                    eff_list.append(e_vec)
                e_canon = torch.stack(eff_list, dim=0).mean(dim=0)
            l_equiv, _ = compute_video_derived_displacement_loss(z_pred, e_canon, norm_weight=0.25)

        # 5. Anti-collapse (compact variance floor without isotropic high-rank bloat!)
        l_vic, _, _ = compute_vicreg_loss(z_pred, std_target=self.vic_std_target, cov_weight=self.vic_cov_weight)

        total = (
            0.5 * l_self
            + 0.8 * l_reuse_soft
            + self.top1_reuse_weight * l_reuse_top1
            + cycle_weight * l_cycle
            + equiv_weight * l_equiv
            + bal_weight * l_bal
            + vic_weight * l_vic
        )

        with torch.no_grad():
            w_ent = -(W_eff * W_eff.clamp_min(1e-8).log()).sum(dim=-1).mean()
            w_scene_mean = (W_eff * s_scene).sum(dim=-1).mean()
            w_max_mean = W_eff.max(dim=-1).values.mean()

        return total, {
            "rel_loss": total.detach(),
            "reuse_soft": l_reuse_soft.detach(),
            "reuse_top1": l_reuse_top1.detach(),
            "fwd_self": l_self.detach(),
            "cycle_loss": l_cycle.detach(),
            "equiv_loss": l_equiv.detach(),
            "bal_loss": l_bal.detach(),
            "w_entropy": w_ent.detach(),
            "w_scene_cos": w_scene_mean.detach(),
            "w_max": w_max_mean.detach(),
            "temp": self.log_temp.exp().detach(),
        }


class E17PureVideoModel(nn.Module):
    """Unified Pure-Video Pretraining Model for E17 (ZERO GT Actions)."""

    def __init__(
        self,
        *,
        variant: str,
        op_dim: int = 384,
        predict_motion: bool = True,
        coverage_mode: str = "cross_demo_unmasked",
        top_k_donors: Optional[int] = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = str(variant)
        self.op_dim = int(op_dim)
        self.predict_motion = bool(predict_motion)
        self.compact_mode = variant in ("e17_m2_compact_op", "e17_m3_dual_orth")

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

        # Match pretrain head params (~82.5M) for equal-parameter VidRel-Op control!
        lam_hidden = 1792 if variant == "lam_vid_op_eqparam" else 1024

        if self.compact_mode:
            # Compact 256D shared bottleneck inside LAM predictor prevents z_res from inflating to erank 340!
            self.lam_bottleneck = nn.Sequential(
                nn.LayerNorm(NUM_BEHAVIOR_TOKENS * EMBED_DIM),
                nn.Linear(NUM_BEHAVIOR_TOKENS * EMBED_DIM, 256),
                nn.GELU(),
                nn.LayerNorm(256),
            )
            self.lam_predictor = ConditionalStateChangePredictor(
                z_dim=256,
                scene_dim=2 * EMBED_DIM,
                hidden_dim=lam_hidden,
                predict_motion=self.predict_motion,
            )
            # Also keep standard full-dim auxiliary head only for comparable validation dv_r2 logging via bottleneck
            self.lam_dyn_proj = nn.Sequential(
                nn.LayerNorm(NUM_BEHAVIOR_TOKENS * self.op_dim),
                nn.Linear(NUM_BEHAVIOR_TOKENS * self.op_dim, 256),
            )
            self.lam_dyn_pred = ConditionalStateChangePredictor(
                z_dim=256,
                scene_dim=2 * EMBED_DIM,
                hidden_dim=768,
                predict_motion=False,
            )
        else:
            self.lam_bottleneck = None
            self.lam_predictor = ConditionalStateChangePredictor(
                z_dim=NUM_BEHAVIOR_TOKENS * EMBED_DIM,
                scene_dim=2 * EMBED_DIM,
                hidden_dim=lam_hidden,
                predict_motion=self.predict_motion,
            )
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

        self.reuse_module: Optional[nn.Module] = None
        if variant == "lam_learn_op":
            self.reuse_module = LearnableReuseRelationModule(
                z_rel_dim=NUM_BEHAVIOR_TOKENS * self.op_dim,
                num_tokens=NUM_BEHAVIOR_TOKENS,
                token_wise=False,
                proj_dim=256,
                hidden_dim=1024,
                predict_motion=self.predict_motion,
                coverage_mode=coverage_mode,
                top_k_donors=None,
            )
        elif variant in ("e17_m1_joint_op", "e17_m2_compact_op"):
            self.reuse_module = SharpLearnableReuseModule(
                z_rel_dim=NUM_BEHAVIOR_TOKENS * self.op_dim,
                num_tokens=NUM_BEHAVIOR_TOKENS,
                token_wise=False,
                proj_dim=256,
                hidden_dim=1024,
                predict_motion=self.predict_motion,
                coverage_mode=coverage_mode,
                top_k_donors=top_k_donors,
                init_temp=0.07,
                top1_reuse_weight=0.40,
                vic_std_target=0.50 if self.compact_mode else 0.70,
                vic_cov_weight=0.005 if self.compact_mode else 0.02,
            )
        elif variant == "e17_m3_dual_orth":
            # Route W_ij on z_reuse (half of z_op = 192D/tok = 768D), predict state-change from full z_op (1536D)!
            half_op = self.op_dim // 2
            self.reuse_module = SharpLearnableReuseModule(
                z_rel_dim=NUM_BEHAVIOR_TOKENS * half_op,
                z_pred_dim=NUM_BEHAVIOR_TOKENS * self.op_dim,
                num_tokens=NUM_BEHAVIOR_TOKENS,
                token_wise=False,
                proj_dim=256,
                hidden_dim=1024,
                predict_motion=self.predict_motion,
                coverage_mode=coverage_mode,
                top_k_donors=top_k_donors,
                init_temp=0.07,
                top1_reuse_weight=0.40,
                vic_std_target=0.50,
                vic_cov_weight=0.005,
            )

    def encode_local_behavior(
        self, clip_rgb: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.local_encoder(clip_rgb, clip_pad_mask=clip_pad_mask)

    def predict_state_change_from_b(self, v0: torch.Tensor, b_t: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = b_t.shape[0]
        z_flat = b_t.float().reshape(B, -1)
        if self.lam_bottleneck is not None:
            z_in = self.lam_bottleneck(z_flat)
        else:
            z_in = z_flat
        return self.lam_predictor(v0, z_in)

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

        pred_dv, pred_mo = self.predict_state_change_from_b(v0, b_t)
        l_dv = ((pred_dv - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)
        l_mo = b_t.new_zeros(())
        if pred_mo is not None and mo_tgt is not None:
            l_mo = ((pred_mo - mo_tgt).square() * w).sum() / (denom * 2.0 * MOTION_FEAT_DIM)

        if self.compact_mode:
            # Compact bottleneck on z_op drives op-dominance; mild L2/rank control on z_res prevents 340-rank noise!
            z_op = b_t[:, :, : self.op_dim].float()
            z_res = b_t[:, :, self.op_dim :].float()
            u_dyn = self.lam_dyn_proj(z_op.reshape(B, -1))
            pred_dv_bot, _ = self.lam_dyn_pred(v0, u_dyn)
            l_bot = ((pred_dv_bot - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)
            # Protect z_op from collapse with mild variance floor (NO isotropic 3072D cov explosion!)
            l_vic, l_var, _ = compute_vicreg_loss(z_op, std_target=0.50, cov_weight=0.005)
            # Gentle L2 shrinkage on z_res + batch nuclear-norm-like singular-value compression on z_res
            # so z_res stays low-rank and visually predictable instead of high-rank noise!
            l_res_reg = 0.015 * z_res.square().mean()
            # Mild rank shaping on centered z_op via soft L2 (keeps smooth geometry without collapse)
            l_op_l2 = 0.003 * z_op.square().mean()
            total = l_dv + 0.65 * l_bot + 0.25 * l_mo + 0.03 * l_vic + l_res_reg + l_op_l2
            return total, {
                "lam_dv": l_dv.detach(),
                "lam_bot": l_bot.detach(),
                "lam_mo": l_mo.detach(),
                "lam_vic": l_vic.detach(),
                "res_reg": l_res_reg.detach(),
            }
        else:
            u_dyn = self.lam_dyn_proj(b_t.float().reshape(B, -1))
            pred_dv_bot, _ = self.lam_dyn_pred(v0, u_dyn)
            l_bot = ((pred_dv_bot - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)
            l_vic, l_var, _ = compute_vicreg_loss(b_t)
            total = l_dv + 0.5 * l_bot + 0.25 * l_mo + 0.05 * l_vic
            return total, {
                "lam_dv": l_dv.detach(),
                "lam_bot": l_bot.detach(),
                "lam_mo": l_mo.detach(),
                "lam_vic": l_vic.detach(),
                "lam_var": l_var.detach(),
            }


@torch.inference_mode()
def evaluate_e17_video_validation(
    model: E17PureVideoModel,
    bank: VideoMotionLocalBank,
    device: torch.device,
    batch_size: int = 256,
    max_samples: int = 2048,
) -> Dict[str, Any]:
    model.eval()
    n = min(max_samples, bank.num_val_samples())
    latents = []
    dv_sse = dv_sst = 0.0
    dv_cnt = 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = bank.gather_rich_batch(bank.val_episodes[s:e], bank.val_frames[s:e], include_actions=False)
        assert batch["actions_10d"] is None
        b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])
        tgts = compute_clip_state_change_targets(batch["clip_rgb"], batch["clip_pad_mask"])
        pred_dv, _ = model.predict_state_change_from_b(tgts["v0"], b_t)
        w = tgts["seg_valid"][:, :, None, None]
        dv_sse += float(((pred_dv - tgts["delta_v_seg"]).square() * w).sum().item())
        dv_sst += float((tgts["delta_v_seg"].square() * w).sum().item())
        dv_cnt += int(w.sum().item())
        latents.append(b_t.float().cpu().numpy())

    Z = np.concatenate(latents, axis=0)
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
        "val_op_rms": float(np.sqrt(np.mean(Z[:, :, : model.op_dim] ** 2))),
        "val_res_rms": float(np.sqrt(np.mean(Z[:, :, model.op_dim :] ** 2))),
        "erank_full": float(er_full["erank_energy"]),
        "erank_op": float(er_op["erank_energy"]),
        "erank_res": float(er_res["erank_energy"]),
    }


def _save_snapshot(model: E17PureVideoModel, config: Dict[str, Any], step: int, val_m: Dict[str, Any], out_path: Path) -> None:
    payload: Dict[str, Any] = {
        "stage": f"e17_{config['run_name']}",
        "step": int(step),
        "local_encoder": {k: v.detach().cpu() for k, v in model.local_encoder.state_dict().items()},
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "val_metrics": val_m,
    }
    if model.reuse_module is not None:
        payload["reuse_module"] = {k: v.detach().cpu() for k, v in model.reuse_module.state_dict().items()}
    atomic_torch_save(payload, out_path)


def run_e17_pretraining(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    use_motion = args.variant != "lam_base" or bool(args.predict_motion)
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
        f"[E17 Pretrain:{args.run_name}] ZERO-ACTION bank ready (variant={args.variant}, "
        f"frac={args.train_episode_fraction:.2f}, starts={bank.num_train_samples():,}, cov={args.coverage_mode}) "
        f"in {time.monotonic()-t0:.1f}s",
        flush=True,
    )

    seed_everything(args.seed)
    model = E17PureVideoModel(
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

    sampler = RichPairedBatchSampler(bank, within_ep_frac=0.5, seed=args.seed, include_actions=False)

    uses_task_meta = args.coverage_mode in ("strict_cross_task", "same_task_only", "cross_demo_strict_anti_scene")
    save_steps = sorted({int(x) for x in args.save_steps.split(",") if x.strip()}) if args.save_steps else []

    config = {
        "regime": "pure_video_pretraining_task_meta_assisted" if uses_task_meta else "pure_video_only_pretraining_zero_actions_zero_task_meta",
        "uses_task_metadata": bool(uses_task_meta),
        "action_supervision": False,
        "init_mode": "from_scratch_zero_action_init",
        "run_name": args.run_name,
        "variant": args.variant,
        "op_dim": int(args.op_dim),
        "lambda_vid": float(args.lambda_vid),
        "lambda_reuse": float(args.lambda_reuse),
        "lambda_orth": float(args.lambda_orth),
        "rel_warmup_steps": int(args.rel_warmup_steps),
        "predict_motion": bool(args.predict_motion),
        "coverage_mode": args.coverage_mode,
        "top_k_donors": args.top_k_donors,
        "train_episode_fraction": float(args.train_episode_fraction),
        "num_train_episodes": len(bank.kept_train_episode_ids),
        "num_train_starts": bank.num_train_samples(),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "save_steps": save_steps,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "clip_grad_norm": float(args.clip_grad_norm),
        "ema_decay": float(args.ema_decay),
        "seed": int(args.seed),
        "encoder_params": sum(p.numel() for p in model.local_encoder.parameters()),
        "total_pretrain_params": sum(p.numel() for p in model.parameters()),
    }
    atomic_write_json(config, run_dir / "config.json")
    print(
        f"[E17 Pretrain:{args.run_name}] Params: encoder={config['encoder_params']/1e6:.2f}M, "
        f"total_pretrain={config['total_pretrain_params']/1e6:.2f}M | uses_task_meta={uses_task_meta}",
        flush=True,
    )

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
    acc_lam = acc_vid = acc_reu = acc_tot = 0.0
    acc_n = 0

    while step < total_steps:
        model.train()
        batch = next(batch_iter)
        assert batch["actions_10d"] is None, "Action tensor leaked into Pure Video-Only Pretraining!"
        step += 1

        warm = (step / float(args.rel_warmup_steps)) if (args.rel_warmup_steps > 0 and step <= args.rel_warmup_steps) else 1.0
        cur_lvid = float(args.lambda_vid) * warm
        cur_lreu = float(args.lambda_reuse) * warm

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])

        lam_loss, lam_stats = model.compute_lam_loss(
            b_t, batch["clip_rgb"], batch["clip_pad_mask"], batch["clip_motion"]
        )

        vid_loss = lam_loss.new_zeros(())
        reu_loss = lam_loss.new_zeros(())
        orth_loss = lam_loss.new_zeros(())
        rel_stats: Dict[str, torch.Tensor] = {}

        # Video-derived motion-flow displacement loss
        if args.variant in ("lam_vid_op", "lam_vid_op_eqparam", "e17_m1_joint_op", "e17_m2_compact_op", "e17_m3_dual_orth") and cur_lvid > 0.0:
            u_mot = extract_video_relation_features(
                "motion_flow", batch["clip_rgb"], batch["clip_pad_mask"], batch["clip_motion"]
            )
            z_vid_tgt = b_t[:, :, : args.op_dim // 2] if args.variant == "e17_m3_dual_orth" else b_t[:, :, : args.op_dim]
            vid_loss, vs = compute_video_derived_displacement_loss(z_vid_tgt, u_mot)
            rel_stats.update({f"vid_{k}": v for k, v in vs.items()})

        # Learnable reuse loss
        if args.variant == "lam_learn_op" and cur_lreu > 0.0:
            assert isinstance(model.reuse_module, LearnableReuseRelationModule)
            do_cyc = step > 500
            reu_loss, rs = model.reuse_module.forward_loss(
                b_t,
                batch["clip_rgb"],
                batch["clip_pad_mask"],
                batch["ep_ids"],
                task_ids=batch["task_ids"] if uses_task_meta else None,
                clip_motion=batch["clip_motion"] if args.predict_motion else None,
                encoder_fn=model.encode_local_behavior if do_cyc else None,
                op_dim=args.op_dim,
                use_cycle=do_cyc,
                use_equiv=do_cyc,
            )
            rel_stats.update(rs)
        elif args.variant in ("e17_m1_joint_op", "e17_m2_compact_op", "e17_m3_dual_orth") and cur_lreu > 0.0:
            assert isinstance(model.reuse_module, SharpLearnableReuseModule)
            do_cyc = step > 500
            z_op = b_t[:, :, : args.op_dim]
            z_route = b_t[:, :, args.op_dim // 2 : args.op_dim] if args.variant == "e17_m3_dual_orth" else z_op
            reu_loss, rs = model.reuse_module.forward_sharp_loss(
                z_route,
                z_op,
                batch["clip_rgb"],
                batch["clip_pad_mask"],
                batch["ep_ids"],
                task_ids=batch["task_ids"] if uses_task_meta else None,
                clip_motion=batch["clip_motion"] if args.predict_motion else None,
                encoder_fn=model.encode_local_behavior if do_cyc else None,
                extract_pred_fn=(lambda b: b[:, :, : args.op_dim]),
                use_cycle=do_cyc,
                use_equiv=do_cyc,
            )
            rel_stats.update(rs)

        if args.variant == "e17_m3_dual_orth" and args.lambda_orth > 0.0 and warm > 0.0:
            z_kin = b_t[:, :, : args.op_dim // 2].float().reshape(b_t.shape[0], -1)
            z_reu = b_t[:, :, args.op_dim // 2 : args.op_dim].float().reshape(b_t.shape[0], -1)
            orth_loss = compute_linear_cka_penalty(z_kin, z_reu)
            rel_stats["cka_kin_reuse"] = orth_loss.detach()

        total_loss = lam_loss + cur_lvid * vid_loss + cur_lreu * reu_loss + (float(args.lambda_orth) * warm) * orth_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        ema.step(model.parameters())

        acc_lam += float(lam_loss.item())
        acc_vid += float(vid_loss.item())
        acc_reu += float(reu_loss.item())
        acc_tot += float(total_loss.item())
        acc_n += 1

        need_eval = (step % args.eval_every == 0) or (step == total_steps) or (step in save_steps)
        if need_eval:
            dt = time.monotonic() - wt0
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            val_m = evaluate_e17_video_validation(model, bank, device)
            if val_m["val_dv_mse"] < best_dv_mse or best_step == 0:
                best_dv_mse = val_m["val_dv_mse"]
                best_step = step
                _save_snapshot(model, config, step, val_m, run_dir / "best.pt")
            if step in save_steps and step != total_steps:
                _save_snapshot(model, config, step, val_m, run_dir / f"step_{step:05d}_ema.pt")
            if step == total_steps:
                _save_snapshot(model, config, step, val_m, run_dir / "last_ema.pt")
            ema.restore(model.parameters())

            rec = {
                "step": step,
                "train": {
                    "lam_loss": acc_lam / max(acc_n, 1),
                    "vid_loss": acc_vid / max(acc_n, 1),
                    "reuse_loss": acc_reu / max(acc_n, 1),
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
                f"[E17:{args.run_name} step {step:5d}/{total_steps}] "
                f"lam={rec['train']['lam_loss']:.4f} vid={rec['train']['vid_loss']:.4f} reu={rec['train']['reuse_loss']:.4f} | "
                f"dv_r2={val_m['val_dv_r2']:.4f} erank(f/op/rs)={val_m['erank_full']:.1f}/{val_m['erank_op']:.1f}/{val_m['erank_res']:.1f} "
                f"rms(op/rs)={val_m['val_op_rms']:.2f}/{val_m['val_res_rms']:.2f} | {rec['train']['steps_per_sec']:.1f}it/s",
                flush=True,
            )
            wt0 = time.monotonic()
            acc_lam = acc_vid = acc_reu = acc_tot = 0.0
            acc_n = 0

    print(f"[E17:{args.run_name}] Done. Saved to {run_dir / 'last_ema.pt'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=[
            "lam_base",
            "lam_vid_op",
            "lam_vid_op_eqparam",
            "lam_learn_op",
            "e17_m1_joint_op",
            "e17_m2_compact_op",
            "e17_m3_dual_orth",
        ],
    )
    p.add_argument("--op-dim", type=int, default=384)
    p.add_argument("--lambda-vid", type=float, default=0.25)
    p.add_argument("--lambda-reuse", type=float, default=0.25)
    p.add_argument("--lambda-orth", type=float, default=0.05)
    p.add_argument("--rel-warmup-steps", type=int, default=500)
    p.add_argument("--predict-motion", action="store_true")
    p.add_argument(
        "--coverage-mode",
        type=str,
        default="cross_demo_unmasked",
        choices=["same_task_only", "cross_demo_unmasked", "cross_demo_anti_scene", "strict_cross_task", "cross_demo_strict_anti_scene"],
    )
    p.add_argument("--top-k-donors", type=int, default=16)
    p.add_argument("--train-episode-fraction", type=float, default=1.0)
    p.add_argument("--save-steps", type=str, default="8046,16092,32184")
    p.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    p.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    p.add_argument("--motion-cache-dir", type=Path, default=MOTION_CACHE_DIR)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=32184)
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_e17_pretraining(parse_args())
