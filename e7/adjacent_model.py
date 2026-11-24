"""E7.1 Model: Adjacent-Pair Behavior Retrieval + Latent Adaptation (CLIP).

Replaces E7's global soft attention over all Behavior Memory entries with a strict
local adjacent-pair retrieval mechanism:
  - Candidate intervals: all adjacent pairs `(b_i, b_{i+1})` for `i = 0 .. M - 2`.
  - Pair score `s_i` computed from `(h_q, h_i^D, h_{i+1}^D)`.
  - Continuous intra-pair fusion weight `beta_i in [0, 1]` computed from `(h_q, h_i^D, h_{i+1}^D)`:
        z_pair[i] = (1 - beta_i) * b_i + beta_i * b_{i+1}
  - Training: Straight-Through Softmax (`temperature = 1.0`, no Gumbel noise):
        p_soft = softmax(pair_scores / temperature)
        p_hard = one_hot(argmax(pair_scores))
        p_st   = p_hard + (p_soft - p_soft.detach())
        z_c    = sum_i p_st[i] * z_pair[i]   # forward uses strictly ONLY (b_{i*}, b_{i*+1})!
  - Inference (`eval()`): deterministic argmax `i* = argmax(pair_scores)`, NO STE:
        z_c    = (1 - beta_{i*}) * b_{i*} + beta_{i*} * b_{i*+1}
  - Edge case: if an episode has only 1 valid Behavior Memory entry, directly returns that latent.
  - Strictly NO trainable V projection, NO Output projection, NO Query residual.
  - Downstream `ResidualTransformer` (`z_q = z_c + G_theta(z_c, h_q)`) and frozen Stage 1 UNet
    are identical to E7.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, extract_exec_actions
from e5.model import E5Policy, ResidualTransformer
from e6.stage1_model import E6Stage1Policy, load_frozen_stage1_teacher

from . import (
    DEFAULT_STAGE1_CKPT,
    EMBED_DIM,
    EXEC_HORIZON,
    FIXED_LATENT_SCALE_S_Z,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
)


def compute_pair_selection_stats(
    pair_idx: torch.Tensor,
    beta_selected: torch.Tensor,
    p_soft: torch.Tensor,
    pair_valid_mask: torch.Tensor,
    has_pair: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample diagnostics for Adjacent-Pair Behavior Retrieval:
      - `norm_pair_pos`: continuous normalized position `(i* + beta_{i*}) / max(num_pairs, 1)` in `[0, 1]`
      - `norm_pair_idx`: discrete normalized pair index `i* / max(num_pairs - 1, 1)` in `[0, 1]`
      - `is_first_pair`: `i* == 0`
      - `is_last_pair`: `i* == num_pairs - 1`
      - `beta_selected`: chosen intra-pair interpolation weight in `[0, 1]`
      - `p_soft_entropy`, `p_soft_norm_entropy`, `p_soft_max_weight`, `p_soft_eff_pairs`, `valid_pairs`
    """
    dev = p_soft.device
    num_pairs = pair_valid_mask.float().sum(dim=-1).clamp_min(1.0)
    idx_f = pair_idx.detach().float()
    beta_f = beta_selected.detach().float()

    # Continuous position in [0, 1] along the Demo (0.0 = entry 0, 1.0 = entry M_valid - 1)
    norm_pos = torch.where(
        has_pair,
        (idx_f + beta_f) / num_pairs.clamp_min(1.0),
        torch.zeros_like(idx_f),
    ).clamp(0.0, 1.0)
    norm_idx = torch.where(
        num_pairs > 1.0,
        idx_f / (num_pairs - 1.0).clamp_min(1.0),
        torch.zeros_like(idx_f),
    ).clamp(0.0, 1.0)

    is_first = (pair_idx == 0) & has_pair
    is_last = (pair_idx == (num_pairs.long() - 1).clamp_min(0)) & has_pair

    w = p_soft.detach().float().masked_fill(~pair_valid_mask, 0.0)
    entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=-1)
    unif_entropy = num_pairs.log()
    norm_entropy = torch.where(
        num_pairs > 1.0,
        entropy / unif_entropy.clamp_min(1e-12),
        torch.zeros_like(entropy),
    )
    max_w = w.max(dim=-1).values
    unif_max_w = 1.0 / num_pairs
    eff_pairs_exp = entropy.exp()
    eff_pairs_ipr = 1.0 / w.square().sum(dim=-1).clamp_min(1e-12)

    return {
        "norm_pair_pos": norm_pos,
        "norm_pair_idx": norm_idx,
        "is_first_pair": is_first.float(),
        "is_last_pair": is_last.float(),
        "beta_selected": beta_f,
        "p_soft_entropy": entropy,
        "p_soft_uniform_entropy": unif_entropy,
        "p_soft_norm_entropy": norm_entropy,
        "p_soft_max_weight": max_w,
        "p_soft_uniform_max_weight": unif_max_w,
        "p_soft_eff_pairs_exp": eff_pairs_exp,
        "p_soft_eff_pairs_ipr": eff_pairs_ipr,
        "valid_pairs": num_pairs,
    }


class AdjacentPairBehaviorReader(nn.Module):
    """Adjacent-Pair Behavior Retrieval Reader over frozen Stage 1 Behavior Memory.

    For each candidate adjacent pair `(i, i + 1)` (`i = 0 .. M - 2`):
      1. Visual encoding (Q/K only, no V/out projection, no query residual):
            q      = q_proj(q_norm(flatten(h_q)))            # (B, d)
            k_i    = k_proj(k_norm(flatten(h_i^D)))          # (B, M, d)
            k_pair = pair_k_proj([k_i, k_{i+1}])             # (B, M-1, d)
            s_i    = (q · k_pair_i) / sqrt(d)                # (B, M-1)
      2. Continuous intra-pair fusion weight `beta_i in [0, 1]`:
            q_beta = beta_q_proj(q)                          # (B, d)
            k_beta = beta_k_proj([k_i, k_{i+1}])             # (B, M-1, d)
            beta_i = sigmoid((q_beta · k_beta_i) / sqrt(d) + beta_bias)
      3. Training (STE) vs Inference (deterministic argmax):
            z_pair[i] = (1 - beta_i) * b_i + beta_i * b_{i+1}
            Train: p_st = p_hard + (p_soft - p_soft.detach()),  z_c = sum_i p_st[i] * z_pair[i]
            Eval : z_c  = (1 - beta_{i*}) * b_{i*} + beta_{i*} * b_{i*+1}
    """

    def __init__(
        self,
        num_query_tokens: int = NUM_QUERY_TOKENS,
        embed_dim: int = EMBED_DIM,
        attn_dim: int = EMBED_DIM,
        temperature: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.num_query_tokens = int(num_query_tokens)
        self.embed_dim = int(embed_dim)
        self.in_dim = self.num_query_tokens * self.embed_dim  # 3072
        self.attn_dim = int(attn_dim)                         # 768
        self.temperature = float(temperature)
        self.scale = 1.0 / math.sqrt(float(self.attn_dim))

        # Shared query / entry visual projections (matching E7 SimpleBehaviorReader)
        self.q_norm = nn.LayerNorm(self.in_dim)
        self.k_norm = nn.LayerNorm(self.in_dim)
        self.q_proj = nn.Linear(self.in_dim, self.attn_dim)
        self.k_proj = nn.Linear(self.in_dim, self.attn_dim)

        # Pair-level key projection over concatenated adjacent pair [k_i, k_{i+1}]
        self.pair_k_proj = nn.Linear(2 * self.attn_dim, self.attn_dim)

        # Intra-pair continuous interpolation weight beta_i in [0, 1]
        self.beta_q_proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.beta_k_proj = nn.Linear(2 * self.attn_dim, self.attn_dim)
        self.beta_bias = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.ones_(self.q_norm.weight)
        nn.init.zeros_(self.q_norm.bias)
        nn.init.ones_(self.k_norm.weight)
        nn.init.zeros_(self.k_norm.bias)
        for lin in (
            self.q_proj,
            self.k_proj,
            self.pair_k_proj,
            self.beta_q_proj,
            self.beta_k_proj,
        ):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.beta_bias)

    def forward(
        self,
        h_q: torch.Tensor,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            h_q: (B, 4, 768) current Query visual tokens.
            h_mem: (B, M, 4, 768) Behavior Memory visual tokens h_i^D.
            b_mem: (B, M, 4, 768) frozen Stage 1 Behavior latents b_i.
            mem_pad_mask: optional (B, M) bool mask where True = padded / invalid entry.
        Returns:
            z_c: (B, 4, 768) strictly fused from adjacent pair `(b_{i*}, b_{i*+1})` (or single entry if M_valid==1).
            info: dict containing `pair_idx`, `beta_selected`, `beta_all`, `p_soft`, `pair_scores`,
                  `pair_valid_mask`, `has_pair`, `eff_entry_weights`.
        """
        if h_q.ndim != 3 or h_q.shape[1:] != (self.num_query_tokens, self.embed_dim):
            raise ValueError(
                f"Expected h_q (B, {self.num_query_tokens}, {self.embed_dim}), got {tuple(h_q.shape)}"
            )
        if h_mem.ndim != 4 or h_mem.shape[2:] != (self.num_query_tokens, self.embed_dim):
            raise ValueError(
                f"Expected h_mem (B, M, {self.num_query_tokens}, {self.embed_dim}), got {tuple(h_mem.shape)}"
            )
        if b_mem.shape != h_mem.shape:
            raise ValueError(
                f"Expected b_mem shape {tuple(h_mem.shape)}, got {tuple(b_mem.shape)}"
            )

        B, M, _, _ = h_mem.shape
        device = h_mem.device
        out_dtype = b_mem.dtype

        if mem_pad_mask is not None:
            if mem_pad_mask.shape != (B, M):
                raise ValueError(
                    f"Expected mem_pad_mask ({B}, {M}), got {tuple(mem_pad_mask.shape)}"
                )
            pad_b = mem_pad_mask.to(device=device, dtype=torch.bool)
            valid = ~pad_b
            if (~valid).all(dim=-1).any():
                raise ValueError("mem_pad_mask has a sample with zero valid Behavior Memory entries")
        else:
            valid = torch.ones((B, M), device=device, dtype=torch.bool)

        # Edge case 1: entire batch has M == 1 -> directly return the single latent
        if M == 1:
            z_c = b_mem[:, 0]
            zeros_idx = torch.zeros((B,), device=device, dtype=torch.long)
            zeros_beta = torch.zeros((B,), device=device, dtype=torch.float32)
            ones_p = torch.ones((B, 1), device=device, dtype=torch.float32)
            false_has = torch.zeros((B,), device=device, dtype=torch.bool)
            info = {
                "pair_idx": zeros_idx,
                "beta_selected": zeros_beta,
                "beta_all": zeros_beta.unsqueeze(-1),
                "p_soft": ones_p,
                "pair_scores": torch.zeros((B, 1), device=device, dtype=torch.float32),
                "pair_valid_mask": torch.zeros((B, 1), device=device, dtype=torch.bool),
                "has_pair": false_has,
                "eff_entry_weights": ones_p,
            }
            return z_c, info

        P = M - 1  # number of candidate adjacent pairs (0,1), (1,2), ..., (M-2, M-1)
        pair_valid = valid[:, :-1] & valid[:, 1:]  # (B, P)
        has_pair = pair_valid.any(dim=-1)          # (B,), False iff sample has only 1 valid entry

        # Encode query and memory entries
        q = self.q_proj(self.q_norm(h_q.reshape(B, self.in_dim)))            # (B, d)
        k = self.k_proj(self.k_norm(h_mem.reshape(B, M, self.in_dim)))       # (B, M, d)

        pair_cat = torch.cat([k[:, :-1, :], k[:, 1:, :]], dim=-1)            # (B, P, 2d)
        k_pair = self.pair_k_proj(pair_cat)                                  # (B, P, d)

        # Compute pair selection logits in float32 for exact masking & STE stability under BF16
        pair_scores_raw = (
            torch.einsum("bd,bpd->bp", q.float(), k_pair.float()) * self.scale
        )  # (B, P)

        # Guard rows with only 1 valid entry (has_pair == False) against all-(-inf) softmax NaNs
        safe_pair_valid = pair_valid.clone()
        if (~has_pair).any():
            safe_pair_valid[~has_pair, 0] = True

        pair_scores_masked = pair_scores_raw.masked_fill(~safe_pair_valid, float("-inf"))
        p_soft = torch.softmax(pair_scores_masked / self.temperature, dim=-1)  # (B, P) float32
        pair_idx = pair_scores_masked.argmax(dim=-1)                           # (B,) int64

        # Intra-pair continuous fusion weights beta_i in (0, 1)
        q_beta = self.beta_q_proj(q)                                           # (B, d)
        k_beta = self.beta_k_proj(pair_cat)                                    # (B, P, d)
        beta_logits = (
            torch.einsum("bd,bpd->bp", q_beta.float(), k_beta.float()) * self.scale
            + self.beta_bias.float()
        )  # (B, P) float32
        beta_all = torch.sigmoid(beta_logits)                                  # (B, P) float32

        batch_ar = torch.arange(B, device=device)
        beta_selected = beta_all[batch_ar, pair_idx]                           # (B,) float32

        b_mem_f = b_mem.float()
        if self.training:
            # Compute all candidate adjacent-pair fusions and aggregate via exact STE:
            # p_st = p_hard + (p_soft - p_soft.detach()) is BIT-EXACT p_hard in forward pass
            # (zero contribution from non-selected pairs) while passing exact proxy grad to p_soft!
            w_beta = beta_all[:, :, None, None]                                # (B, P, 1, 1)
            z_pairs = (1.0 - w_beta) * b_mem_f[:, :-1] + w_beta * b_mem_f[:, 1:]  # (B, P, 4, 768)
            p_hard = F.one_hot(pair_idx, num_classes=P).to(dtype=p_soft.dtype) # (B, P)
            p_st = p_hard + (p_soft - p_soft.detach())                         # (B, P)
            z_c_pair = torch.einsum("bp,bpkd->bkd", p_st, z_pairs)             # (B, 4, 768)
        else:
            # Inference: deterministic argmax selection + identical intra-pair fusion, NO STE!
            b_left = b_mem_f[batch_ar, pair_idx]                               # (B, 4, 768)
            b_right = b_mem_f[batch_ar, pair_idx + 1]                          # (B, 4, 768)
            bw = beta_selected[:, None, None]                                  # (B, 1, 1)
            z_c_pair = (1.0 - bw) * b_left + bw * b_right                      # (B, 4, 768)

        # Handle mixed-batch edge case where a sample has only 1 valid memory entry
        if (~has_pair).all():
            single_idx = valid.float().argmax(dim=-1)
            z_c_f = b_mem_f[batch_ar, single_idx]
        elif (~has_pair).any():
            single_idx = valid.float().argmax(dim=-1)
            z_single = b_mem_f[batch_ar, single_idx]
            z_c_f = torch.where(has_pair[:, None, None], z_c_pair, z_single)
        else:
            z_c_f = z_c_pair

        z_c = z_c_f.to(dtype=out_dtype)

        # Effective per-entry weights in R^M (strictly 2 adjacent non-zeros summing to 1, or 1 non-zero if M_valid==1)
        with torch.no_grad():
            eff_w = torch.zeros((B, M), device=device, dtype=torch.float32)
            if has_pair.any():
                hp_b = batch_ar[has_pair]
                hp_i = pair_idx[has_pair]
                hp_beta = beta_selected[has_pair]
                eff_w[hp_b, hp_i] = 1.0 - hp_beta
                eff_w[hp_b, hp_i + 1] = hp_beta
            if (~has_pair).any():
                sp_b = batch_ar[~has_pair]
                sp_i = valid[~has_pair].float().argmax(dim=-1)
                eff_w[sp_b, sp_i] = 1.0

        info = {
            "pair_idx": pair_idx,
            "beta_selected": beta_selected,
            "beta_all": beta_all,
            "p_soft": p_soft,
            "pair_scores": pair_scores_masked,
            "pair_valid_mask": pair_valid,
            "has_pair": has_pair,
            "eff_entry_weights": eff_w,
        }
        return z_c, info


class E7AdjacentPolicy(nn.Module):
    """E7.1 Policy: Frozen Stage 1 Teacher + AdjacentPairBehaviorReader + ResidualTransformer."""

    def __init__(
        self,
        teacher: E6Stage1Policy,
        *,
        latent_scale_s_z: float = FIXED_LATENT_SCALE_S_Z,
        reader_attn_dim: int = EMBED_DIM,
        reader_temperature: float = 1.0,
        residual_layers: int = 3,
        residual_heads: int = NUM_HEADS,
        residual_ffn_dim: int = 3072,
        residual_dropout: float = 0.1,
    ):
        super().__init__()
        self.teacher = teacher
        self._freeze_teacher()

        self.reader = AdjacentPairBehaviorReader(
            num_query_tokens=NUM_QUERY_TOKENS,
            embed_dim=EMBED_DIM,
            attn_dim=reader_attn_dim,
            temperature=reader_temperature,
        )
        self.residual_transformer = ResidualTransformer(
            embed_dim=EMBED_DIM,
            num_layers=residual_layers,
            num_heads=residual_heads,
            ffn_dim=residual_ffn_dim,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=residual_dropout,
        )
        self.register_buffer(
            "latent_scale_s_z",
            torch.tensor(float(latent_scale_s_z), dtype=torch.float32),
        )

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "E7AdjacentPolicy":
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for p in self.reader.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.residual_transformer.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    @torch.no_grad()
    def encode_full_demo_memory(
        self,
        demo_rgb_1hz: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M_d = self.teacher.prompt_pooler(demo_rgb_1hz, demo_pad_mask=demo_pad_mask)
        P = M_d.shape[1]
        rr = self.teacher.reference_reader
        if P > rr.max_prompt_tokens:
            raise ValueError(
                f"Demo 1Hz length {P} exceeds max_prompt_tokens {rr.max_prompt_tokens}"
            )
        mem = M_d + rr.prompt_pos_emb[:, :P, :]
        if demo_pad_mask is not None:
            mem = mem.masked_fill(demo_pad_mask.unsqueeze(-1), 0.0)
        return mem

    @torch.no_grad()
    def extract_behavior_memory_from_single_demo(
        self,
        demo_rgb_1hz: torch.Tensor,
        anchor_query_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct `(h_mem, b_mem)` for a single Demo purely from its video features."""
        if demo_rgb_1hz.ndim != 3 or demo_rgb_1hz.shape[1:] != (2, EMBED_DIM):
            raise ValueError(
                f"Expected demo_rgb_1hz (P, 2, {EMBED_DIM}), got {tuple(demo_rgb_1hz.shape)}"
            )
        rr = self.teacher.reference_reader
        P = demo_rgb_1hz.shape[0]
        if P > rr.max_prompt_tokens:
            demo_rgb_1hz = demo_rgb_1hz[: rr.max_prompt_tokens]
            P = rr.max_prompt_tokens

        full_mem = self.encode_full_demo_memory(demo_rgb_1hz.unsqueeze(0))
        h_mem = E5Policy.format_query_tokens(anchor_query_rgb)
        M = h_mem.shape[0]
        tgt = h_mem + rr.prompt_current_obs_pos_emb
        b_mem = rr.prompt_with_obs_decoder(
            tgt=tgt,
            memory=full_mem.expand(M, P, EMBED_DIM),
            memory_key_padding_mask=None,
            need_weights=False,
        )
        return h_mem, b_mem

    def read_cross_latent(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        h_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if h_q is None:
            h_q = E5Policy.format_query_tokens(query_rgb_feats)
        z_c, info = self.reader(h_q, h_mem, b_mem, mem_pad_mask=mem_pad_mask)
        return z_c, h_q, info

    def adapt_latent(
        self,
        z_cross: torch.Tensor,
        h_q: torch.Tensor,
        *,
        delta_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = float(delta_scale)
        if scale == 0.0:
            return z_cross, torch.zeros_like(z_cross)
        delta_z = self.residual_transformer(z_cross, h_q)
        if scale != 1.0:
            delta_z = delta_z * scale
        return z_cross + delta_z, delta_z

    def forward(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        z_self: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: Optional[torch.Tensor] = None,
        *,
        lambda_a: float = 1.0,
        lambda_z: float = 0.5,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_q = E5Policy.format_query_tokens(query_rgb_feats).detach()
        h_mem_det = h_mem.detach()
        b_mem_det = b_mem.detach()
        z_s = z_self.detach()
        s_z = self.latent_scale_s_z.to(device=h_q.device, dtype=torch.float32).clamp_min(1e-6)

        z_c, _, r_info = self.read_cross_latent(
            h_mem_det, b_mem_det, mem_pad_mask, query_rgb_feats, h_q=h_q
        )
        z_q, delta_z = self.adapt_latent(z_c, h_q)

        if lambda_z != 0.0:
            per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            latent_loss_norm = per_sample_lz.mean()
        else:
            with torch.no_grad():
                per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
                latent_loss_norm = per_sample_lz.mean()

        if lambda_a != 0.0:
            per_sample_diff = self.teacher.diffusion_loss_from_z_self(
                z_q,
                actions_10d,
                noise=noise,
                pertub_noise=pertub_noise,
                timesteps=timesteps,
            )
            diff_loss = per_sample_diff.mean()
        else:
            diff_loss = torch.zeros((), device=z_q.device, dtype=torch.float32)

        total_loss = (
            lambda_a * diff_loss
            if lambda_a != 0.0
            else torch.zeros((), device=z_q.device, dtype=torch.float32)
        )
        if lambda_z != 0.0:
            total_loss = total_loss + lambda_z * latent_loss_norm

        with torch.no_grad():
            B = z_q.shape[0]
            zc_f = z_c.detach().float()
            zq_f = z_q.detach().float()
            zs_f = z_s.float()
            dz_f = delta_z.detach().float()

            latent_mse_raw = (zq_f - zs_f).square().mean()
            nodelta_mse_raw = (zc_f - zs_f).square().mean()
            nodelta_lz = ((zc_f - zs_f) / s_z).square().mean()
            cos_nodelta = F.cosine_similarity(
                zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).mean()
            cos_adapted = F.cosine_similarity(
                zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).mean()
            gap = zs_f - zc_f
            dir_cos = F.cosine_similarity(
                dz_f.reshape(B, -1), gap.reshape(B, -1), dim=-1
            ).mean()

            delta_sq = dz_f.square().mean(dim=(1, 2))
            delta_rms = torch.sqrt(delta_sq.mean() + 1e-12)
            z_cross_rms = torch.sqrt(zc_f.square().mean() + 1e-12)

            if is_self is not None:
                is_self_b = is_self.to(device=z_q.device, dtype=torch.bool)
                s_m = is_self_b.float()
                c_m = (~is_self_b).float()
                n_s = s_m.sum()
                n_c = c_m.sum()
                self_delta_rms = torch.sqrt(
                    (delta_sq * s_m).sum() / n_s.clamp_min(1.0) + 1e-12
                )
                cross_delta_rms = torch.sqrt(
                    (delta_sq * c_m).sum() / n_c.clamp_min(1.0) + 1e-12
                )
            else:
                n_s = torch.tensor(0.0, device=z_q.device)
                n_c = torch.tensor(float(B), device=z_q.device)
                self_delta_rms = torch.tensor(0.0, device=z_q.device)
                cross_delta_rms = delta_rms

            pstats = compute_pair_selection_stats(
                r_info["pair_idx"],
                r_info["beta_selected"],
                r_info["p_soft"],
                r_info["pair_valid_mask"],
                r_info["has_pair"],
            )

        return {
            "loss": total_loss,
            "diff_loss": diff_loss.detach(),
            "latent_loss_norm": latent_loss_norm.detach(),
            "latent_mse_raw": latent_mse_raw,
            "nodelta_mse_raw": nodelta_mse_raw,
            "nodelta_loss_norm": nodelta_lz,
            "cos_nodelta_self": cos_nodelta,
            "cos_adapted_self": cos_adapted,
            "dir_cos": dir_cos,
            "delta_rms": delta_rms,
            "cross_delta_rms": cross_delta_rms,
            "self_delta_rms": self_delta_rms,
            "z_cross_rms": z_cross_rms,
            "p_soft_entropy": pstats["p_soft_entropy"].mean(),
            "p_soft_norm_entropy": pstats["p_soft_norm_entropy"].mean(),
            "p_soft_max_weight": pstats["p_soft_max_weight"].mean(),
            "p_soft_eff_pairs": pstats["p_soft_eff_pairs_exp"].mean(),
            "beta_mean": pstats["beta_selected"].mean(),
            "beta_std": pstats["beta_selected"].std(unbiased=False),
            "norm_pair_pos_mean": pstats["norm_pair_pos"].mean(),
            "n_self": n_s,
            "n_cross": n_c,
        }

    @torch.inference_mode()
    def predict_action_from_behavior_memory(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        apply_delta: bool = True,
        delta_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        z_cross, h_q, r_info = self.read_cross_latent(
            h_mem, b_mem, mem_pad_mask, query_rgb_feats
        )
        if apply_delta and delta_scale != 0.0:
            z_cond, delta_z = self.adapt_latent(z_cross, h_q, delta_scale=delta_scale)
        else:
            delta_z = torch.zeros_like(z_cross)
            z_cond = z_cross

        action_10d = self.teacher.sample_from_z_self(
            z_cond, generator=generator, init_noise=init_noise
        )
        return {
            "action_10d": action_10d,
            "action_7d": action_10d_to_7d(action_10d),
            "exec_action_7d": extract_exec_actions(action_10d, exec_horizon=EXEC_HORIZON),
            "z_cross": z_cross,
            "delta_z": delta_z,
            "z_adapted": z_cond,
            "reader_info": r_info,
            "apply_delta": bool(apply_delta and delta_scale != 0.0),
            "delta_scale": float(delta_scale),
        }

    def get_optim_groups(
        self,
        *,
        reader_lr: float = 1.0e-4,
        residual_lr: float = 1.0e-4,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        no_decay_keywords = ("bias", "norm", "pos_emb")

        reader_decay: List[nn.Parameter] = []
        reader_no_decay: List[nn.Parameter] = []
        for name, param in self.reader.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                reader_no_decay.append(param)
            else:
                reader_decay.append(param)

        res_decay: List[nn.Parameter] = []
        res_no_decay: List[nn.Parameter] = []
        for name, param in self.residual_transformer.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                res_no_decay.append(param)
            else:
                res_decay.append(param)

        teacher_trainable = sum(
            p.numel() for p in self.teacher.parameters() if p.requires_grad
        )
        if teacher_trainable != 0:
            raise RuntimeError(f"Frozen teacher has {teacher_trainable} trainable params!")

        all_trainable = [p for p in self.parameters() if p.requires_grad]
        grouped_count = (
            len(reader_decay) + len(reader_no_decay) + len(res_decay) + len(res_no_decay)
        )
        if grouped_count != len(all_trainable):
            raise RuntimeError(
                f"Optimizer parameter grouping mismatch: grouped={grouped_count} != trainable={len(all_trainable)}"
            )

        return [
            {
                "name": "adjacent_reader_decay",
                "params": reader_decay,
                "weight_decay": weight_decay,
                "lr": reader_lr,
            },
            {
                "name": "adjacent_reader_no_decay",
                "params": reader_no_decay,
                "weight_decay": 0.0,
                "lr": reader_lr,
            },
            {
                "name": "residual_decay",
                "params": res_decay,
                "weight_decay": weight_decay,
                "lr": residual_lr,
            },
            {
                "name": "residual_no_decay",
                "params": res_no_decay,
                "weight_decay": 0.0,
                "lr": residual_lr,
            },
        ]

    def parameter_summary(self, visual_encoder_params: int = 171_598_848) -> Dict[str, Any]:
        teacher_total = sum(p.numel() for p in self.teacher.parameters())
        teacher_trainable = sum(
            p.numel() for p in self.teacher.parameters() if p.requires_grad
        )
        reader_total = sum(p.numel() for p in self.reader.parameters())
        reader_trainable = sum(
            p.numel() for p in self.reader.parameters() if p.requires_grad
        )
        res_total = sum(p.numel() for p in self.residual_transformer.parameters())
        res_trainable = sum(
            p.numel() for p in self.residual_transformer.parameters() if p.requires_grad
        )
        return {
            "visual_encoder_frozen": visual_encoder_params,
            "stage1_teacher_frozen": teacher_total,
            "stage1_teacher_trainable": teacher_trainable,
            "adjacent_reader_total": reader_total,
            "adjacent_reader_trainable": reader_trainable,
            "adjacent_reader_in_dim": self.reader.in_dim,
            "adjacent_reader_attn_dim": self.reader.attn_dim,
            "adjacent_reader_temperature": self.reader.temperature,
            "residual_transformer_total": res_total,
            "residual_transformer_trainable": res_trainable,
            "total_trainable": reader_trainable + res_trainable,
            "full_system_total_with_frozen_clip": (
                visual_encoder_params + teacher_total + reader_total + res_total
            ),
            "latent_scale_s_z": float(self.latent_scale_s_z.item()),
        }


def load_e7_adjacent_policy(
    ckpt_path: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E7AdjacentPolicy, Dict[str, Any]]:
    """Load frozen Stage 1 Teacher + trained E7.1 AdjacentPairBehaviorReader & ResidualTransformer."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    s1_path = Path(stage1_ckpt) if stage1_ckpt is not None else Path(cfg["stage1_ckpt"])
    teacher = load_frozen_stage1_teacher(s1_path, device="cpu")
    policy = E7AdjacentPolicy(
        teacher,
        latent_scale_s_z=float(
            payload.get("latent_scale_s_z", cfg.get("latent_scale_s_z", FIXED_LATENT_SCALE_S_Z))
        ),
        reader_attn_dim=int(cfg.get("reader_attn_dim", EMBED_DIM)),
        reader_temperature=float(cfg.get("reader_temperature", 1.0)),
        residual_layers=int(cfg.get("residual_layers", 3)),
        residual_heads=int(cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
    )
    policy.reader.load_state_dict(payload["reader"], strict=True)
    policy.residual_transformer.load_state_dict(payload["residual_transformer"], strict=True)
    policy.eval()
    return policy.to(device), payload
