"""E18 Unified Models & Fair External Baselines (100% Pure Video Pretraining, ZERO GT Actions).

Implements:
  1. `UnifiedReuseModule`: Clean, unified Cross-Demo Reuse module supporting:
       - Sharp Top-K / Straight-Through Top-1 single-donor substitution
       - Configurable ablation switches (`top_k_donors`, `top1_reuse_weight`, `use_cycle`, `use_equiv`)
       - Relation Scaling controls:
           * `donor_pool_size` (independent control of candidate donor count M_pool)
           * `memory_queue_size` (MoCo-style stop-grad FIFO cross-batch donor queue up to 1024+)
           * `exact_cross_task_frac` (exact fraction p_cov of cross-task vs same-task donors at fixed M_pool)
  2. `URBEModel`: Unified Relational Behavior Encoder supporting ALL Pillar 1 (ablations),
     Pillar 2 (geometry-aware motion relations & independent motion subspace), and
     Pillar 3 (relation scaling) configurations with matched parameter budget (~80M pretrain,
     42.55M identical `LocalBehaviorEncoder`).
  3. `LAPAVQModel`: Faithful external baseline implementing LAPA (Ye et al., ICLR 2025) —
     discrete EMA-Vector-Quantized latent action bottleneck (4 tokens, codebook K=256/tok,
     dead-code revival, commitment loss, FDM future prediction, ZERO cross-demo reuse graph).
  4. `CycleMimicModel`: Faithful external baseline implementing Cross-Scene Cycle Consistency —
     scene-invariance CKA regularization + bidirectional cross-scene behavior & visual cycle
     consistency on random cross-demo pairs (ZERO learned cross-demo reuse graph W_ij).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e9 import EMBED_DIM, LOCAL_WINDOW_LEN, NUM_BEHAVIOR_TOKENS, NUM_HEADS
from e9.stage1_local_model import LocalBehaviorEncoder
from e12.action_attributes import pairwise_cosine_sim
from e15.train_e15 import compute_linear_cka_penalty
from e16 import MOTION_FEAT_DIM
from e16.video_relations import (
    ConditionalStateChangePredictor,
    compute_clip_state_change_targets,
    compute_vicreg_loss,
    compute_video_derived_displacement_loss,
)


def _off_diag_mask(n: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(n, device=device, dtype=torch.bool)


class UnifiedReuseModule(nn.Module):
    """Unified Sharp Cross-Demo Reuse Module with full ablation & Relation Scaling controls."""

    def __init__(
        self,
        z_route_dim: int,
        *,
        z_pred_dim: Optional[int] = None,
        proj_dim: int = 256,
        hidden_dim: int = 1024,
        predict_motion: bool = False,
        coverage_mode: str = "cross_demo_unmasked",
        top_k_donors: Optional[int] = 16,
        init_temp: float = 0.07,
        top1_reuse_weight: float = 0.40,
        vic_std_target: float = 0.50,
        vic_cov_weight: float = 0.005,
        donor_pool_size: Optional[int] = None,
        memory_queue_size: int = 0,
        exact_cross_task_frac: Optional[float] = None,
    ):
        super().__init__()
        self.z_route_dim = int(z_route_dim)
        self.z_pred_dim = int(z_pred_dim) if z_pred_dim is not None else int(z_route_dim)
        self.coverage_mode = str(coverage_mode)
        self.top_k_donors = int(top_k_donors) if top_k_donors is not None else None
        self.top1_reuse_weight = float(top1_reuse_weight)
        self.vic_std_target = float(vic_std_target)
        self.vic_cov_weight = float(vic_cov_weight)
        self.donor_pool_size = int(donor_pool_size) if donor_pool_size is not None else None
        self.memory_queue_size = int(memory_queue_size)
        self.exact_cross_task_frac = (
            float(exact_cross_task_frac) if exact_cross_task_frac is not None else None
        )

        self.q_proj = nn.Sequential(nn.LayerNorm(self.z_route_dim), nn.Linear(self.z_route_dim, proj_dim))
        self.k_proj = nn.Sequential(nn.LayerNorm(self.z_route_dim), nn.Linear(self.z_route_dim, proj_dim))
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))
        self.forward_pred = ConditionalStateChangePredictor(
            z_dim=self.z_pred_dim,
            scene_dim=2 * 768,
            hidden_dim=hidden_dim,
            predict_motion=predict_motion,
        )

        # Optional MoCo-style FIFO donor memory queue for scaling donor pool beyond batch size!
        if self.memory_queue_size > 0:
            self.register_buffer("queue_k", torch.zeros(self.memory_queue_size, proj_dim))
            self.register_buffer("queue_z", torch.zeros(self.memory_queue_size, self.z_pred_dim))
            self.register_buffer("queue_ep", torch.full((self.memory_queue_size,), -1, dtype=torch.long))
            self.register_buffer("queue_task", torch.full((self.memory_queue_size,), -1, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(
        self, k_norm: torch.Tensor, z_pred: torch.Tensor, ep_ids: torch.Tensor, task_ids: Optional[torch.Tensor]
    ) -> None:
        if self.memory_queue_size <= 0:
            return
        Q = self.memory_queue_size
        B = k_norm.shape[0]
        ptr = int(self.queue_ptr.item())
        take = min(B, Q)
        k_in = k_norm[:take].detach()
        z_in = z_pred[:take].detach()
        ep_in = ep_ids[:take].detach()
        t_in = task_ids[:take].detach() if task_ids is not None else torch.full_like(ep_in, -1)

        end = ptr + take
        if end <= Q:
            self.queue_k[ptr:end] = k_in
            self.queue_z[ptr:end] = z_in
            self.queue_ep[ptr:end] = ep_in
            self.queue_task[ptr:end] = t_in
        else:
            first = Q - ptr
            self.queue_k[ptr:] = k_in[:first]
            self.queue_z[ptr:] = z_in[:first]
            self.queue_ep[ptr:] = ep_in[:first]
            self.queue_task[ptr:] = t_in[:first]
            rem = take - first
            self.queue_k[:rem] = k_in[first:]
            self.queue_z[:rem] = z_in[first:]
            self.queue_ep[:rem] = ep_in[first:]
            self.queue_task[:rem] = t_in[first:]

        self.queue_ptr[0] = (ptr + take) % Q
        self.queue_filled[0] = min(Q, int(self.queue_filled.item()) + take)

    @torch.no_grad()
    def _build_candidate_mask(
        self,
        v0: torch.Tensor,
        ep_ids: torch.Tensor,
        donor_ep_ids: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor] = None,
        donor_task_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build `(B, N_donors)` boolean validity mask respecting coverage, pool size & exact cross-task frac."""
        B = v0.shape[0]
        N = donor_ep_ids.shape[0]
        dev = v0.device
        s_scene_batch = pairwise_cosine_sim(v0.float().reshape(B, -1))
        diff_ep = (ep_ids[:, None] != donor_ep_ids[None, :]) & (donor_ep_ids[None, :] >= 0)

        # Case 1: Exact cross-task fraction at fixed donor_pool_size (Axis 3B — zero confounding!)
        if self.exact_cross_task_frac is not None and task_ids is not None and donor_task_ids is not None:
            pool_m = self.donor_pool_size if self.donor_pool_size is not None else 48
            n_ct = int(round(pool_m * self.exact_cross_task_frac))
            n_st = pool_m - n_ct
            same_t = diff_ep & (task_ids[:, None] == donor_task_ids[None, :])
            diff_t = diff_ep & (task_ids[:, None] != donor_task_ids[None, :])
            valid = torch.zeros((B, N), device=dev, dtype=torch.bool)
            # Vectorized random selection of exact n_st same-task + n_ct cross-task donors per row
            if n_st > 0:
                r_st = torch.rand((B, N), device=dev)
                r_st = r_st.masked_fill(~same_t, -1.0)
                k_st = min(n_st, N)
                idx_st = torch.topk(r_st, k=k_st, dim=-1).indices
                sel_st = torch.zeros_like(valid).scatter_(1, idx_st, True) & same_t
                valid = valid | sel_st
            if n_ct > 0:
                r_ct = torch.rand((B, N), device=dev)
                r_ct = r_ct.masked_fill(~diff_t, -1.0)
                k_ct = min(n_ct, N)
                idx_ct = torch.topk(r_ct, k=k_ct, dim=-1).indices
                sel_ct = torch.zeros_like(valid).scatter_(1, idx_ct, True) & diff_t
                valid = valid | sel_ct
            empty = ~valid.any(dim=1)
            if empty.any():
                valid = torch.where(empty[:, None], diff_ep, valid)
            return valid, s_scene_batch

        # Case 2: Standard coverage modes
        if self.coverage_mode == "same_task_only" and task_ids is not None and donor_task_ids is not None:
            valid = diff_ep & (task_ids[:, None] == donor_task_ids[None, :])
        elif self.coverage_mode == "strict_cross_task" and task_ids is not None and donor_task_ids is not None:
            valid = diff_ep & (task_ids[:, None] != donor_task_ids[None, :])
        elif self.coverage_mode == "same_demo_only":
            # Only different timestamps within the SAME episode!
            valid = (ep_ids[:, None] == donor_ep_ids[None, :])
            if N >= B:
                valid[:, :B] = valid[:, :B] & _off_diag_mask(B, dev)
        elif self.coverage_mode == "cross_demo_unmasked":
            valid = diff_ep
        elif self.coverage_mode == "cross_demo_anti_scene":
            # Anti-scene on in-batch donors; queue donors default allowed
            valid = diff_ep.clone()
            diff_b = diff_ep[:, :B]
            q75 = torch.quantile(s_scene_batch[diff_b], 0.75) if diff_b.any() else torch.tensor(1.0, device=dev)
            valid[:, :B] = diff_b & (s_scene_batch <= q75)
        else:
            raise ValueError(f"Unknown coverage_mode: {self.coverage_mode}")

        # Optional random donor pool size cap M_pool (Axis 3A!)
        if self.donor_pool_size is not None and self.donor_pool_size > 0:
            k_cap = min(self.donor_pool_size, N)
            r = torch.rand((B, N), device=dev).masked_fill(~valid, -1.0)
            idx = torch.topk(r, k=k_cap, dim=-1).indices
            capped = torch.zeros_like(valid).scatter_(1, idx, True) & valid
            valid = capped

        empty = ~valid.any(dim=1)
        if empty.any():
            fallback = diff_ep if diff_ep.any() else torch.ones((B, N), device=dev, dtype=torch.bool)
            if N >= B and not diff_ep.any():
                fallback[:, :B] = _off_diag_mask(B, dev)
            valid = torch.where(empty[:, None], fallback, valid)
        return valid, s_scene_batch

    def _masked_pred_loss(
        self,
        pred_dv: torch.Tensor,
        tgt_dv: torch.Tensor,
        seg_valid: torch.Tensor,
        pred_mo: Optional[torch.Tensor] = None,
        tgt_mo: Optional[torch.Tensor] = None,
        motion_weight: float = 0.25,
    ) -> torch.Tensor:
        w = seg_valid[:, :, None, None]
        denom = w.sum().clamp_min(1.0)
        dv_err = ((pred_dv - tgt_dv).square() * w).sum() / (denom * 2.0 * 768.0)
        if pred_mo is not None and tgt_mo is not None:
            mo_err = ((pred_mo - tgt_mo).square() * w).sum() / (denom * 2.0 * MOTION_FEAT_DIM)
            return dv_err + motion_weight * mo_err
        return dv_err

    def compute_relation_weights(
        self, z_route: torch.Tensor, valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute in-batch `(B, B)` relation weights (compatible with E17 Exp 1 authenticity audit)."""
        B = z_route.shape[0]
        zf = z_route.float().reshape(B, -1)
        qn = F.normalize(self.q_proj(zf), dim=-1)
        kn = F.normalize(self.k_proj(zf), dim=-1)
        temp = self.log_temp.exp().clamp(0.05, 0.5)
        logits = (qn @ kn.T) / temp
        logits = logits.masked_fill(~valid_mask, -1e4)
        if self.top_k_donors is not None and self.top_k_donors < B - 1:
            k_eff = min(int(self.top_k_donors), int(valid_mask.sum(dim=-1).min().clamp_min(1).item()))
            if k_eff < B - 1:
                topk_val, _ = torch.topk(logits, k=k_eff, dim=-1)
                logits = logits.masked_fill(logits < topk_val[:, -1:].detach(), -1e4)
        W = torch.softmax(logits, dim=-1)
        col_mean = W.mean(dim=0).clamp_min(1e-7)
        bal = (col_mean * torch.log(col_mean * float(B))).sum()
        z_tilde = (W @ zf).reshape_as(z_route)
        return W, z_tilde, bal

    def forward_loss(
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
        use_bf16_cycle: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = z_route.shape[0]
        zr = z_route.float().reshape(B, -1)
        zp = z_pred.float().reshape(B, -1)

        tgts = compute_clip_state_change_targets(
            clip_rgb, clip_pad_mask, clip_motion if self.forward_pred.predict_motion else None
        )
        v0 = tgts["v0"]
        dv_tgt = tgts["delta_v_seg"]
        seg_valid = tgts["seg_valid"]
        mo_tgt = tgts.get("motion_seg")

        qn = F.normalize(self.q_proj(zr), dim=-1)
        kn_batch = F.normalize(self.k_proj(zr), dim=-1)

        # Optionally extend donor bank with FIFO cross-batch memory queue!
        use_q = self.memory_queue_size > 0 and int(self.queue_filled.item()) > 0
        if use_q:
            nf = int(self.queue_filled.item())
            kn_all = torch.cat([kn_batch, self.queue_k[:nf].detach()], dim=0)
            zp_all = torch.cat([zp, self.queue_z[:nf].detach()], dim=0)
            ep_all = torch.cat([ep_ids, self.queue_ep[:nf]], dim=0)
            t_all = (
                torch.cat([task_ids, self.queue_task[:nf]], dim=0)
                if task_ids is not None
                else None
            )
        else:
            kn_all = kn_batch
            zp_all = zp
            ep_all = ep_ids
            t_all = task_ids

        N_don = kn_all.shape[0]
        valid_mask, s_scene_batch = self._build_candidate_mask(
            v0, ep_ids, ep_all, task_ids=task_ids, donor_task_ids=t_all
        )

        temp = self.log_temp.exp().clamp(0.05, 0.5)
        logits = (qn @ kn_all.T) / temp
        logits = logits.masked_fill(~valid_mask, -1e4)
        if self.top_k_donors is not None:
            min_valid = int(valid_mask.sum(dim=-1).min().clamp_min(1).item())
            k_eff = min(int(self.top_k_donors), min_valid)
            if k_eff < N_don - 1:
                topk_val, _ = torch.topk(logits, k=k_eff, dim=-1)
                logits = logits.masked_fill(logits < topk_val[:, -1:].detach(), -1e4)

        W = torch.softmax(logits, dim=-1)  # (B, N_don)
        # Load-balancing entropy over active donors (prevents hub collapse)
        col_mean = W.mean(dim=0).clamp_min(1e-7)
        l_bal = (col_mean * torch.log(col_mean * float(N_don))).sum()

        # Soft top-K cross-demo donor substitution
        z_tilde_soft = W @ zp_all

        # 1. Self state-change prediction
        pred_dv_self, pred_mo_self = self.forward_pred(v0, zp)
        l_self = self._masked_pred_loss(pred_dv_self, dv_tgt, seg_valid, pred_mo_self, mo_tgt)

        # 2a. Soft reuse loss
        pred_dv_soft, pred_mo_soft = self.forward_pred(v0, z_tilde_soft)
        l_reuse_soft = self._masked_pred_loss(pred_dv_soft, dv_tgt, seg_valid, pred_mo_soft, mo_tgt)

        # 2b. Hard Top-1 straight-through single-donor reuse loss
        l_reuse_top1 = zp.new_zeros(())
        if self.top1_reuse_weight > 0.0:
            top1_idx = torch.argmax(W, dim=-1)
            one_hot = F.one_hot(top1_idx, num_classes=N_don).to(dtype=W.dtype)
            W_st = one_hot + (W - W.detach())
            z_tilde_top1 = W_st @ zp_all
            pred_dv_t1, pred_mo_t1 = self.forward_pred(v0, z_tilde_top1)
            l_reuse_top1 = self._masked_pred_loss(pred_dv_t1, dv_tgt, seg_valid, pred_mo_t1, mo_tgt)

        # 3. Counterfactual cross-scene cycle consistency
        l_cycle = zp.new_zeros(())
        if use_cycle and encoder_fn is not None and extract_pred_fn is not None:
            n_cyc = min(B, 64)
            with torch.no_grad():
                v_sub = valid_mask[:n_cyc, :B]
                empty_c = ~v_sub.any(dim=1)
                if empty_c.any():
                    v_sub = torch.where(empty_c[:, None], _off_diag_mask(B, v0.device)[:n_cyc], v_sub)
                probs = v_sub.float() / v_sub.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                donor_j = torch.multinomial(probs, num_samples=1).squeeze(-1)
            v0_sub = v0[:n_cyc]
            z_donor = zp[donor_j].detach()
            dv_cf, _ = self.forward_pred(v0_sub, z_donor)
            v_cf = v0_sub[:, None, :, :] + dv_cf.repeat_interleave(4, dim=1)
            if use_bf16_cycle and v0.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    b_cf = encoder_fn(v_cf.to(dtype=clip_rgb.dtype), None)
            else:
                b_cf = encoder_fn(v_cf.to(dtype=clip_rgb.dtype), None)
            z_cf = extract_pred_fn(b_cf).float().reshape(n_cyc, -1)
            l_cycle = (1.0 - F.cosine_similarity(z_cf, z_donor, dim=-1)).mean()

        # 4. Self-supervised displacement equivariance via scene-canonicalized forward effects
        l_equiv = zp.new_zeros(())
        if use_equiv and B >= 8:
            with torch.no_grad():
                ref_idx = torch.linspace(0, B - 1, 4, device=zp.device).long()
                # Batched 4*B forward evaluation (faster than 4 sequential Python calls!)
                v_refs = v0[ref_idx][:, None, :, :].expand(4, B, 2, 768).reshape(4 * B, 2, 768)
                z_rep = zp.detach()[None, :, :].expand(4, B, -1).reshape(4 * B, -1)
                e_dv, e_mo = self.forward_pred(v_refs, z_rep)
                e_vec = e_dv.reshape(4, B, -1)
                if e_mo is not None:
                    e_dv_n = e_vec / e_vec.square().mean().sqrt().clamp_min(1e-5)
                    e_mo_r = e_mo.reshape(4, B, -1)
                    e_mo_n = e_mo_r / e_mo_r.square().mean().sqrt().clamp_min(1e-5)
                    e_vec = torch.cat([e_dv_n, e_mo_n], dim=-1)
                e_canon = e_vec.mean(dim=0)
            l_equiv, _ = compute_video_derived_displacement_loss(z_pred, e_canon, norm_weight=0.25)

        # 5. Anti-collapse variance floor (without isotropic 3072D rank bloat)
        l_vic, _, _ = compute_vicreg_loss(
            z_pred, std_target=self.vic_std_target, cov_weight=self.vic_cov_weight
        )

        # Update FIFO donor queue AFTER loss computation
        if self.memory_queue_size > 0 and self.training:
            self._enqueue(kn_batch, zp, ep_ids, task_ids)

        total = (
            0.5 * l_self
            + 0.8 * l_reuse_soft
            + self.top1_reuse_weight * l_reuse_top1
            + (cycle_weight * l_cycle if use_cycle else 0.0)
            + (equiv_weight * l_equiv if use_equiv else 0.0)
            + bal_weight * l_bal
            + vic_weight * l_vic
        )

        with torch.no_grad():
            w_ent = -(W * W.clamp_min(1e-8).log()).sum(dim=-1).mean()
            w_max_mean = W.max(dim=-1).values.mean()
            w_batch = W[:, :B]
            w_scene = (w_batch * s_scene_batch).sum(dim=-1).mean() / w_batch.sum(dim=-1).clamp_min(1e-5).mean()
            valid_mean = valid_mask.float().sum(dim=-1).mean()

        return total, {
            "rel_loss": total.detach(),
            "reuse_soft": l_reuse_soft.detach(),
            "reuse_top1": l_reuse_top1.detach(),
            "fwd_self": l_self.detach(),
            "cycle_loss": l_cycle.detach(),
            "equiv_loss": l_equiv.detach(),
            "bal_loss": l_bal.detach(),
            "w_entropy": w_ent.detach(),
            "w_scene_cos": w_scene.detach(),
            "w_max": w_max_mean.detach(),
            "n_valid_donors": valid_mean.detach(),
            "temp": self.log_temp.exp().detach(),
        }


class URBEModel(nn.Module):
    """Unified Relational Behavior Encoder (URBE) supporting all E18 Pillar 1, 2, 3 configurations."""

    def __init__(
        self,
        *,
        compact_manifold: bool = True,
        op_dim: int = 384,
        use_reuse: bool = True,
        top_k_donors: Optional[int] = 16,
        top1_reuse_weight: float = 0.40,
        init_temp: float = 0.07,
        coverage_mode: str = "cross_demo_unmasked",
        donor_pool_size: Optional[int] = None,
        memory_queue_size: int = 0,
        exact_cross_task_frac: Optional[float] = None,
        predict_motion: bool = False,
        indep_motion_subspace: bool = False,
        mot_subspace_dim: int = 128,
        match_params_no_reuse: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.compact_manifold = bool(compact_manifold)
        self.op_dim = int(op_dim)  # 384 = Op/Res split; 768 = No Op/Res split!
        self.use_reuse = bool(use_reuse)
        self.predict_motion = bool(predict_motion)
        self.indep_motion_subspace = bool(indep_motion_subspace)
        self.mot_subspace_dim = int(mot_subspace_dim) if indep_motion_subspace else 0
        self.reuse_dim = self.op_dim - self.mot_subspace_dim
        assert self.reuse_dim > 0

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

        # Match total pretrain parameter count (~80M) when use_reuse=False so capacity is strictly controlled!
        lam_hidden = 1920 if (not self.use_reuse and match_params_no_reuse) else 1024

        if self.compact_manifold:
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
            self.lam_dyn_proj = nn.Sequential(
                nn.LayerNorm(NUM_BEHAVIOR_TOKENS * self.op_dim),
                nn.Linear(NUM_BEHAVIOR_TOKENS * self.op_dim, 256),
            )
            self.lam_dyn_pred = ConditionalStateChangePredictor(
                z_dim=256,
                scene_dim=2 * EMBED_DIM,
                hidden_dim=1024 if (not self.use_reuse and match_params_no_reuse) else 768,
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

        self.reuse_module: Optional[UnifiedReuseModule] = None
        if self.use_reuse:
            z_reuse_total = NUM_BEHAVIOR_TOKENS * self.reuse_dim
            self.reuse_module = UnifiedReuseModule(
                z_route_dim=z_reuse_total,
                z_pred_dim=z_reuse_total,
                proj_dim=256,
                hidden_dim=1024,
                predict_motion=self.predict_motion and (not self.indep_motion_subspace),
                coverage_mode=coverage_mode,
                top_k_donors=top_k_donors,
                init_temp=init_temp,
                top1_reuse_weight=top1_reuse_weight,
                vic_std_target=0.50 if self.compact_manifold else 0.70,
                vic_cov_weight=0.005 if self.compact_manifold else 0.02,
                donor_pool_size=donor_pool_size,
                memory_queue_size=memory_queue_size,
                exact_cross_task_frac=exact_cross_task_frac,
            )

    def encode_local_behavior(
        self, clip_rgb: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.local_encoder(clip_rgb, clip_pad_mask=clip_pad_mask)

    def predict_state_change_from_b(
        self, v0: torch.Tensor, b_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = b_t.shape[0]
        z_flat = b_t.float().reshape(B, -1)
        z_in = self.lam_bottleneck(z_flat) if self.lam_bottleneck is not None else z_flat
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

        if self.compact_manifold:
            z_op = b_t[:, :, : self.op_dim].float()
            u_dyn = self.lam_dyn_proj(z_op.reshape(B, -1))
            pred_dv_bot, _ = self.lam_dyn_pred(v0, u_dyn)
            l_bot = ((pred_dv_bot - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)
            l_vic, _, _ = compute_vicreg_loss(z_op, std_target=0.50, cov_weight=0.005)
            l_res_reg = (
                0.015 * b_t[:, :, self.op_dim :].float().square().mean()
                if self.op_dim < EMBED_DIM
                else b_t.new_zeros(())
            )
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
            l_vic, _, _ = compute_vicreg_loss(b_t)
            total = l_dv + 0.5 * l_bot + 0.25 * l_mo + 0.05 * l_vic
            return total, {
                "lam_dv": l_dv.detach(),
                "lam_bot": l_bot.detach(),
                "lam_mo": l_mo.detach(),
                "lam_vic": l_vic.detach(),
            }


# ---------------------------------------------------------------------------
# External Baseline 1: LAPA (Discrete VQ-LAM, Ye et al. ICLR 2025)
# ---------------------------------------------------------------------------

class EMAVectorQuantizer(nn.Module):
    """Multi-token EMA Vector Quantizer with dead-code revival (LAPA / VQ-VAE style)."""

    def __init__(
        self,
        num_tokens: int = 4,
        codebook_size: int = 256,
        code_dim: int = 128,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.ema_decay = float(ema_decay)
        self.commitment_weight = float(commitment_weight)
        self.dead_threshold = float(dead_threshold)

        # Independent codebook per latent action token (4 x 256 -> 256^4 expressive discrete space!)
        embed = torch.randn(self.num_tokens, self.codebook_size, self.code_dim)
        embed = F.normalize(embed, dim=-1) * math.sqrt(self.code_dim) * 0.25
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(self.num_tokens, self.codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize `(B, K_tok, code_dim)` -> `(z_q_st, vq_loss, stats)`."""
        B, T, D = z_e.shape
        assert T == self.num_tokens and D == self.code_dim
        z_f = z_e.float()

        # Squared Euclidean distance per token: (T, B, K_code)
        zt = z_f.permute(1, 0, 2)  # (T, B, D)
        dist = (
            zt.square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.bmm(zt, self.embed.transpose(1, 2))
            + self.embed.square().sum(dim=-1).unsqueeze(1)
        )
        codes = torch.argmin(dist, dim=-1)  # (T, B)
        # Gather quantized codes
        z_q_t = torch.gather(
            self.embed, 1, codes.unsqueeze(-1).expand(-1, -1, D)
        )  # (T, B, D)

        if self.training:
            with torch.no_grad():
                oh = F.one_hot(codes, num_classes=self.codebook_size).float()  # (T, B, K)
                counts = oh.sum(dim=1)  # (T, K)
                dw = torch.bmm(oh.transpose(1, 2), zt)  # (T, K, D)
                self.cluster_size.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
                self.embed_avg.mul_(self.ema_decay).add_(dw, alpha=1.0 - self.ema_decay)

                # Laplace smoothing to prevent division by zero
                n_tot = self.cluster_size.sum(dim=-1, keepdim=True).clamp_min(1.0)
                cs = (self.cluster_size + 1e-5) / (n_tot + self.codebook_size * 1e-5) * n_tot
                self.embed.copy_(self.embed_avg / cs.unsqueeze(-1))

                # Dead-code revival: replace under-utilized codes with random batch latents + small noise
                dead = self.cluster_size < self.dead_threshold
                if dead.any():
                    for t_i in range(T):
                        d_idx = torch.nonzero(dead[t_i], as_tuple=False).squeeze(-1)
                        if d_idx.numel() > 0:
                            r_idx = torch.randint(0, B, (d_idx.numel(),), device=z_f.device)
                            rep = zt[t_i, r_idx] + 0.02 * torch.randn_like(zt[t_i, r_idx])
                            self.embed[t_i, d_idx] = rep
                            self.embed_avg[t_i, d_idx] = rep
                            self.cluster_size[t_i, d_idx] = 1.0

        z_q = z_q_t.permute(1, 0, 2)  # (B, T, D)
        commit_loss = F.mse_loss(z_f, z_q.detach())
        z_q_st = z_f + (z_q - z_f).detach()

        with torch.no_grad():
            oh_eval = F.one_hot(codes, num_classes=self.codebook_size).float()
            avg_p = oh_eval.mean(dim=1)  # (T, K)
            perp = torch.exp(-(avg_p * avg_p.clamp_min(1e-8).log()).sum(dim=-1)).mean()
            used_frac = (oh_eval.sum(dim=1) > 0).float().mean()

        return z_q_st, self.commitment_weight * commit_loss, {
            "vq_commit": commit_loss.detach(),
            "vq_perplexity": perp.detach(),
            "vq_used_frac": used_frac.detach(),
        }


class LAPAVQModel(nn.Module):
    """External Baseline: Faithful LAPA Discrete VQ-LAM (matched ~80M pretrain params, 42.55M encoder)."""

    def __init__(
        self,
        *,
        codebook_size: int = 256,
        code_dim: int = 128,
        fdm_hidden: int = 1920,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.op_dim = 384  # interface compatibility

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
        # Per-token VQ projection (768 -> code_dim) and back (code_dim -> 768)
        self.pre_vq = nn.Sequential(nn.LayerNorm(EMBED_DIM), nn.Linear(EMBED_DIM, self.code_dim))
        self.vq = EMAVectorQuantizer(
            num_tokens=NUM_BEHAVIOR_TOKENS,
            codebook_size=self.codebook_size,
            code_dim=self.code_dim,
        )
        self.post_vq = nn.Sequential(nn.Linear(self.code_dim, EMBED_DIM), nn.LayerNorm(EMBED_DIM))

        # Matched-capacity Forward Dynamics Model (FDM): (v0, z_q) -> delta_v_seg
        self.fdm = ConditionalStateChangePredictor(
            z_dim=NUM_BEHAVIOR_TOKENS * EMBED_DIM,
            scene_dim=2 * EMBED_DIM,
            hidden_dim=fdm_hidden,
            predict_motion=False,
        )

    def encode_local_behavior(
        self, clip_rgb: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.local_encoder(clip_rgb, clip_pad_mask=clip_pad_mask)

    def quantize_b(self, b_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z_e = self.pre_vq(b_t.float())  # (B, 4, code_dim)
        z_q_st, vq_loss, vq_stats = self.vq(z_e)
        b_q = self.post_vq(z_q_st)  # (B, 4, 768)
        return b_q, vq_loss, vq_stats

    def predict_state_change_from_b(
        self, v0: torch.Tensor, b_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b_q, _, _ = self.quantize_b(b_t)
        return self.fdm(v0, b_q.reshape(b_t.shape[0], -1))

    def compute_loss(
        self,
        b_t: torch.Tensor,
        clip_rgb: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = b_t.shape[0]
        tgts = compute_clip_state_change_targets(clip_rgb, clip_pad_mask, None)
        v0 = tgts["v0"]
        dv_tgt = tgts["delta_v_seg"]
        seg_valid = tgts["seg_valid"]
        w = seg_valid[:, :, None, None]
        denom = w.sum().clamp_min(1.0)

        b_q, vq_loss, vq_stats = self.quantize_b(b_t)
        pred_dv, _ = self.fdm(v0, b_q.reshape(B, -1))
        l_dv = ((pred_dv - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)

        # Mild latent scale control on pre-VQ b_t (prevents arbitrary norm drift without hurting VQ)
        l_reg = 0.002 * b_t.float().square().mean()
        total = l_dv + vq_loss + l_reg
        stats = {"lam_dv": l_dv.detach(), "l_reg": l_reg.detach()}
        stats.update(vq_stats)
        return total, stats


# ---------------------------------------------------------------------------
# External Baseline 2: CycleMimic (Cross-Scene Cycle Consistency, ZERO W_ij)
# ---------------------------------------------------------------------------

class CycleMimicModel(nn.Module):
    """External Baseline: Cross-Scene Cycle-Consistent Behavior Representation Learning.

    Enforces:
      1. Self forward state-change prediction F(v0_i, z_i) -> dv_i
      2. Scene-invariance decorrelation CKA(z_op, v0) (discourages static scene leakage)
      3. Bidirectional cross-scene cycle consistency on random cross-demo pairs (i, j):
           - Forward transfer: V_hat_{i<-j} = v0_i + F(v0_i, z_j)
           - Behavior cycle:   z_hat_j = E(V_hat_{i<-j}) ≈ z_j
           - Visual cycle:     F(v0_j, z_hat_j) ≈ dv_j (round-trip visual reconstruction!)
    Crucially has NO learned cross-demo reuse relation graph W_ij and NO cross-demo
    donor substitution loss L_reuse, isolating cycle consistency from relational reuse!
    """

    def __init__(
        self,
        *,
        op_dim: int = 384,
        fdm_hidden: int = 1920,
        compact_bottleneck: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.op_dim = int(op_dim)
        self.compact_bottleneck = bool(compact_bottleneck)

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
        if self.compact_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.LayerNorm(NUM_BEHAVIOR_TOKENS * EMBED_DIM),
                nn.Linear(NUM_BEHAVIOR_TOKENS * EMBED_DIM, 256),
                nn.GELU(),
                nn.LayerNorm(256),
            )
            fdm_in = 256
        else:
            self.bottleneck = None
            fdm_in = NUM_BEHAVIOR_TOKENS * EMBED_DIM

        self.fdm = ConditionalStateChangePredictor(
            z_dim=fdm_in,
            scene_dim=2 * EMBED_DIM,
            hidden_dim=fdm_hidden,
            predict_motion=False,
        )
        # Auxiliary operational predictor for clean counterfactual transfer
        self.op_pred = ConditionalStateChangePredictor(
            z_dim=NUM_BEHAVIOR_TOKENS * self.op_dim,
            scene_dim=2 * EMBED_DIM,
            hidden_dim=1024,
            predict_motion=False,
        )

    def encode_local_behavior(
        self, clip_rgb: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.local_encoder(clip_rgb, clip_pad_mask=clip_pad_mask)

    def predict_state_change_from_b(
        self, v0: torch.Tensor, b_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        zf = b_t.float().reshape(b_t.shape[0], -1)
        zin = self.bottleneck(zf) if self.bottleneck is not None else zf
        return self.fdm(v0, zin)

    def compute_loss(
        self,
        b_t: torch.Tensor,
        clip_rgb: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
        ep_ids: torch.Tensor,
        *,
        do_cycle: bool = True,
        cycle_beh_weight: float = 0.12,
        cycle_vis_weight: float = 0.40,
        scene_inv_weight: float = 0.015,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = b_t.shape[0]
        tgts = compute_clip_state_change_targets(clip_rgb, clip_pad_mask, None)
        v0 = tgts["v0"]
        dv_tgt = tgts["delta_v_seg"]
        seg_valid = tgts["seg_valid"]
        w = seg_valid[:, :, None, None]
        denom = w.sum().clamp_min(1.0)

        # 1. Self reconstruction (full + operational)
        pred_dv, _ = self.predict_state_change_from_b(v0, b_t)
        l_dv = ((pred_dv - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)

        z_op = b_t[:, :, : self.op_dim].float().reshape(B, -1)
        pred_dv_op, _ = self.op_pred(v0, z_op)
        l_dv_op = ((pred_dv_op - dv_tgt).square() * w).sum() / (denom * 2.0 * 768.0)

        # 2. Scene invariance penalty CKA(z_op, v0)
        l_sinv = compute_linear_cka_penalty(z_op, v0.float().reshape(B, -1))

        # 3. Bidirectional cross-scene cycle consistency (behavior cycle + visual round-trip cycle!)
        l_cyc_beh = b_t.new_zeros(())
        l_cyc_vis = b_t.new_zeros(())
        if do_cycle:
            n_cyc = min(B, 64)
            with torch.no_grad():
                diff_ep = ep_ids[:n_cyc, None] != ep_ids[None, :]
                empty = ~diff_ep.any(dim=1)
                if empty.any():
                    diff_ep = torch.where(empty[:, None], _off_diag_mask(B, v0.device)[:n_cyc], diff_ep)
                probs = diff_ep.float() / diff_ep.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                donor_j = torch.multinomial(probs, num_samples=1).squeeze(-1)

            v0_i = v0[:n_cyc]
            v0_j = v0[donor_j]
            dv_j = dv_tgt[donor_j]
            w_j = seg_valid[donor_j][:, :, None, None]
            denom_j = w_j.sum().clamp_min(1.0)
            z_j_det = z_op[donor_j].detach()

            # Forward cross-scene transfer: apply behavior j in scene i
            dv_ij, _ = self.op_pred(v0_i, z_j_det)
            v_ij = v0_i[:, None, :, :] + dv_ij.repeat_interleave(4, dim=1)
            if v0.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    b_ij = self.encode_local_behavior(v_ij.to(dtype=clip_rgb.dtype), None)
            else:
                b_ij = self.encode_local_behavior(v_ij.to(dtype=clip_rgb.dtype), None)
            z_hat_j = b_ij[:, :, : self.op_dim].float().reshape(n_cyc, -1)

            # Behavior cycle: re-encoded behavior matches donor j
            l_cyc_beh = (1.0 - F.cosine_similarity(z_hat_j, z_j_det, dim=-1)).mean()

            # Visual round-trip cycle: applying recovered z_hat_j back in scene j reconstructs dv_j!
            dv_j_back, _ = self.op_pred(v0_j, z_hat_j)
            l_cyc_vis = ((dv_j_back - dv_j).square() * w_j).sum() / (denom_j * 2.0 * 768.0)

        # Anti-collapse & compact residual regularization (prevents CKA/cycle rank collapse to erank~2!)
        l_vic, _, _ = compute_vicreg_loss(b_t[:, :, : self.op_dim], std_target=0.65, cov_weight=0.01)
        l_res = 0.015 * b_t[:, :, self.op_dim :].float().square().mean()
        l_op_l2 = 0.002 * z_op.square().mean()

        total = (
            l_dv
            + 0.65 * l_dv_op
            + cycle_beh_weight * l_cyc_beh
            + cycle_vis_weight * l_cyc_vis
            + scene_inv_weight * l_sinv
            + 0.12 * l_vic
            + l_res
            + l_op_l2
        )
        return total, {
            "lam_dv": l_dv.detach(),
            "dv_op": l_dv_op.detach(),
            "cyc_beh": l_cyc_beh.detach(),
            "cyc_vis": l_cyc_vis.detach(),
            "scene_inv": l_sinv.detach(),
        }
