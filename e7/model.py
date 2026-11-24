"""E7 Model: Behavior Memory Retrieval (Simple Q/K Attention Reader) + Latent Adaptation.

Core architecture:
  1. Frozen Stage 1 Self Teacher (`e6/outputs/stage1/stage1_self_latent/best.pt`):
     - `PromptPooler`, `BehaviorReferenceReader` R_T, `ConditionalUnet1D`, `ActionNormalizer`.
     - For a Demo D of length T, every 16 control steps (t_i = 0, 16, 32, ... < T):
         b_i = R_T(M_D, h_i^D)   in R^(4 x 768)
       where R_T reads the COMPLETE Demo memory M_D (never a 16-frame slice).
  2. Simple Trainable Attention Reader (`SimpleBehaviorReader`):
     - Q: current Query visual feature h_q (4, 768) -> Q(h_q) in R^d
     - K: Behavior Memory visual features h_i^D (4, 768) -> K(h_i^D) in R^d
     - V: frozen Behavior latents b_i (4, 768)
         w_i = softmax_i( Q(h_q) K(h_i^D)^T / sqrt(d) )
         z_c = sum_i w_i * b_i
     - Entire (4, 768) latent shares ONE scalar reading weight distribution w_i.
     - Strictly NO trainable V projection, NO Output projection, NO Query residual.
  3. Freshly initialized zero-out `ResidualTransformer`:
         delta_z = G_theta(z_c, h_q),    z_q = z_c + delta_z
  4. Frozen Stage 1 UNet conditioned ONLY on z_q:
         L = L_diff(A_q | z_q) + 0.5 * MSE((z_q - sg(z_self)) / s_z, 0)
     with fixed s_z = 7.822748.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, extract_exec_actions
from e5.model import E5Policy, ResidualTransformer
from e6.stage1_model import E6Stage1Policy, load_frozen_stage1_teacher

from . import (
    BEHAVIOR_STRIDE,
    DEFAULT_STAGE1_CKPT,
    EMBED_DIM,
    EXEC_HORIZON,
    FIXED_LATENT_SCALE_S_Z,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
)


def compute_behavior_anchor_steps(
    num_frames: int, stride: int = BEHAVIOR_STRIDE
) -> np.ndarray:
    """Return stride-16 control step indices `0, 16, 32, ... < num_frames`.

    - Supports variable-length Demos and partial tail windows:
      * T = 192 -> 12 entries: [0, 16, ..., 176]
      * T = 195 -> 13 entries: [0, 16, ..., 176, 192] (tail window [192..194])
      * 1 <= T < 16 -> 1 entry: [0]
    """
    T = int(num_frames)
    if T < 1:
        raise ValueError(f"Demo num_frames must be >= 1, got {T}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return np.arange(0, T, stride, dtype=np.int32)


def compute_attention_stats(
    weights: torch.Tensor,
    mem_pad_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample Reader attention diagnostics from `weights` (B, M):
      - entropy: -sum_i w_i log(w_i)
      - uniform_entropy: log(M_valid)
      - norm_entropy: entropy / log(M_valid) in [0, 1] (1.0 = meaningless uniform average)
      - max_weight: max_i w_i
      - uniform_max_weight: 1 / M_valid
      - eff_pos_exp: exp(entropy) in [1, M_valid]
      - eff_pos_ipr: 1 / sum_i w_i^2 in [1, M_valid]
      - eff_frac: exp(entropy) / M_valid in (0, 1]
      - valid_pos: M_valid
    """
    w = weights.detach().float()
    if mem_pad_mask is not None:
        valid_mask = ~mem_pad_mask.to(device=w.device, dtype=torch.bool)
        w = w.masked_fill(~valid_mask, 0.0)
        valid_pos = valid_mask.float().sum(dim=-1).clamp_min(1.0)
    else:
        valid_pos = torch.full(
            (w.shape[0],), float(w.shape[1]), device=w.device, dtype=torch.float32
        )

    entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=-1)
    uniform_entropy = valid_pos.log()
    norm_entropy = torch.where(
        valid_pos > 1.0,
        entropy / uniform_entropy.clamp_min(1e-12),
        torch.zeros_like(entropy),
    )
    max_weight = w.max(dim=-1).values
    uniform_max_weight = 1.0 / valid_pos
    eff_pos_exp = entropy.exp()
    eff_pos_ipr = 1.0 / w.square().sum(dim=-1).clamp_min(1e-12)
    eff_frac = eff_pos_exp / valid_pos

    return {
        "entropy": entropy,
        "uniform_entropy": uniform_entropy,
        "norm_entropy": norm_entropy,
        "max_weight": max_weight,
        "uniform_max_weight": uniform_max_weight,
        "eff_pos_exp": eff_pos_exp,
        "eff_pos_ipr": eff_pos_ipr,
        "eff_frac": eff_frac,
        "valid_pos": valid_pos,
    }


class SimpleBehaviorReader(nn.Module):
    """Simple trainable Q/K Attention Reader over frozen Stage 1 Behavior Memory.

    Equations:
        w_i = softmax_i( Q(h_q) K(h_i^D)^T / sqrt(d) )
        z_c = sum_i w_i * b_i

    Invariants:
      - Entire `(4, 768)` behavior latent shares ONE scalar reading weight `w_i` per entry.
      - NO trainable Value projection, NO Output projection, NO Query residual.
      - Output `z_c` lies strictly in the convex hull of frozen Stage 1 behavior latents `{b_i}`.
    """

    def __init__(
        self,
        num_query_tokens: int = NUM_QUERY_TOKENS,
        embed_dim: int = EMBED_DIM,
        attn_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.embed_dim = int(embed_dim)
        self.in_dim = self.num_query_tokens * self.embed_dim  # 4 * 768 = 3072
        self.attn_dim = int(attn_dim)
        self.scale = 1.0 / math.sqrt(float(self.attn_dim))

        self.q_norm = nn.LayerNorm(self.in_dim)
        self.k_norm = nn.LayerNorm(self.in_dim)
        self.q_proj = nn.Linear(self.in_dim, self.attn_dim)
        self.k_proj = nn.Linear(self.in_dim, self.attn_dim)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.ones_(self.q_norm.weight)
        nn.init.zeros_(self.q_norm.bias)
        nn.init.ones_(self.k_norm.weight)
        nn.init.zeros_(self.k_norm.bias)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.k_proj.bias)

    def encode_query(self, h_q: torch.Tensor) -> torch.Tensor:
        """Map `h_q` (B, 4, 768) -> `Q(h_q)` (B, d)."""
        if h_q.ndim != 3 or h_q.shape[1:] != (self.num_query_tokens, self.embed_dim):
            raise ValueError(
                f"Expected h_q (B, {self.num_query_tokens}, {self.embed_dim}), got {tuple(h_q.shape)}"
            )
        B = h_q.shape[0]
        return self.q_proj(self.q_norm(h_q.reshape(B, self.in_dim)))

    def encode_keys(self, h_mem: torch.Tensor) -> torch.Tensor:
        """Map `h_mem` (B, M, 4, 768) -> `K(h_i^D)` (B, M, d)."""
        if h_mem.ndim != 4 or h_mem.shape[2:] != (self.num_query_tokens, self.embed_dim):
            raise ValueError(
                f"Expected h_mem (B, M, {self.num_query_tokens}, {self.embed_dim}), got {tuple(h_mem.shape)}"
            )
        B, M, _, _ = h_mem.shape
        return self.k_proj(self.k_norm(h_mem.reshape(B, M, self.in_dim)))

    def forward(
        self,
        h_q: torch.Tensor,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_q: (B, 4, 768) current Query visual tokens.
            h_mem: (B, M, 4, 768) Behavior Memory visual tokens h_i^D.
            b_mem: (B, M, 4, 768) frozen Stage 1 Behavior latents b_i = R_T(M_D, h_i^D).
            mem_pad_mask: optional (B, M) bool mask where True = padded / invalid entry.
        Returns:
            z_c: (B, 4, 768) convex combination sum_i w_i * b_i (NO V proj, NO out proj, NO skip).
            weights: (B, M) float32 reading weights summing to 1 over valid entries.
        """
        if b_mem.shape != h_mem.shape:
            raise ValueError(
                f"Expected b_mem shape {tuple(h_mem.shape)} matching h_mem, got {tuple(b_mem.shape)}"
            )
        B, M, _, _ = h_mem.shape
        q = self.encode_query(h_q)      # (B, d)
        k = self.encode_keys(h_mem)     # (B, M, d)

        # Compute scaled dot-product logits in float32 for exact masking & softmax under BF16
        logits = torch.einsum("bd,bmd->bm", q.float(), k.float()) * self.scale
        if mem_pad_mask is not None:
            if mem_pad_mask.shape != (B, M):
                raise ValueError(
                    f"Expected mem_pad_mask ({B}, {M}), got {tuple(mem_pad_mask.shape)}"
                )
            mask_b = mem_pad_mask.to(device=logits.device, dtype=torch.bool)
            if mask_b.all(dim=-1).any():
                raise ValueError("mem_pad_mask has an episode with zero valid Behavior Memory entries")
            logits = logits.masked_fill(mask_b, float("-inf"))

        weights = torch.softmax(logits, dim=-1)  # (B, M) float32
        z_c = torch.einsum("bm,bmkd->bkd", weights.to(dtype=b_mem.dtype), b_mem)
        return z_c, weights


class E7Policy(nn.Module):
    """E7 Behavior Memory Retrieval + Latent Adaptation Policy.

    - Frozen Stage 1 Teacher (`teacher`)
    - Randomly initialized `SimpleBehaviorReader` (`reader`, Q/K only)
    - Freshly initialized zero-out `ResidualTransformer` (`residual_transformer`)
    """

    def __init__(
        self,
        teacher: E6Stage1Policy,
        *,
        latent_scale_s_z: float = FIXED_LATENT_SCALE_S_Z,
        reader_attn_dim: int = EMBED_DIM,
        residual_layers: int = 3,
        residual_heads: int = NUM_HEADS,
        residual_ffn_dim: int = 3072,
        residual_dropout: float = 0.1,
        prediction_mode: str = "residual",
        direct_latent_bias: Optional[torch.Tensor] = None,
        direct_out_proj_init_std: float = 0.002,
    ):
        super().__init__()
        if prediction_mode not in ("residual", "direct"):
            raise ValueError(
                f"Unsupported prediction_mode={prediction_mode!r}; expected 'residual' or 'direct'"
            )
        self.prediction_mode = str(prediction_mode)
        self.direct_out_proj_init_std = float(direct_out_proj_init_std)
        self.teacher = teacher
        self._freeze_teacher()

        self.reader = SimpleBehaviorReader(
            num_query_tokens=NUM_QUERY_TOKENS,
            embed_dim=EMBED_DIM,
            attn_dim=reader_attn_dim,
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

        if self.prediction_mode == "direct":
            if self.direct_out_proj_init_std > 0.0:
                nn.init.normal_(
                    self.residual_transformer.out_proj.weight,
                    mean=0.0,
                    std=self.direct_out_proj_init_std,
                )
            else:
                nn.init.zeros_(self.residual_transformer.out_proj.weight)
            nn.init.zeros_(self.residual_transformer.out_proj.bias)

            bias_buf = torch.zeros((1, NUM_QUERY_TOKENS, EMBED_DIM), dtype=torch.float32)
            self.register_buffer("direct_latent_bias", bias_buf)
            if direct_latent_bias is not None:
                self.set_direct_latent_bias(direct_latent_bias)
        else:
            self.register_buffer("direct_latent_bias", None)

    def set_direct_latent_bias(self, mean_latent: torch.Tensor) -> None:
        """Set fixed (non-trainable) output bias buffer for `prediction_mode='direct'`."""
        if self.prediction_mode != "direct" or self.direct_latent_bias is None:
            raise RuntimeError("set_direct_latent_bias is only valid for prediction_mode='direct'")
        t = mean_latent.detach().to(device=self.direct_latent_bias.device, dtype=torch.float32)
        if t.shape == (NUM_QUERY_TOKENS, EMBED_DIM):
            t = t.unsqueeze(0)
        if t.shape != (1, NUM_QUERY_TOKENS, EMBED_DIM):
            raise ValueError(
                f"Expected direct_latent_bias shape ({NUM_QUERY_TOKENS}, {EMBED_DIM}) or "
                f"(1, {NUM_QUERY_TOKENS}, {EMBED_DIM}), got {tuple(t.shape)}"
            )
        with torch.no_grad():
            self.direct_latent_bias.copy_(t)

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "E7Policy":
        """Keep frozen Stage 1 Teacher permanently in eval() mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Deterministic list of all trainable parameters (SimpleReader + ResidualTransformer)."""
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
        """Encode complete 1Hz Demo CLIP features `(B, P, 2, 768)` -> `M_D + prompt_pos_emb` `(B, P, 768)`."""
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
        """Construct `(h_mem, b_mem)` for a single Demo purely from its video features:
        - `demo_rgb_1hz`: `(P, 2, 768)` full-Demo 1Hz CLIP features (complete Demo context)
        - `anchor_query_rgb`: `(M, 2, 2, 768)` stride-16 `[max(0, t_i-1), t_i]` CLIP features
        Returns:
            `h_mem`: `(M, 4, 768)` visual features h_i^D
            `b_mem`: `(M, 4, 768)` frozen Teacher behavior latents b_i = R_T(M_D, h_i^D)
        """
        if demo_rgb_1hz.ndim != 3 or demo_rgb_1hz.shape[1:] != (2, EMBED_DIM):
            raise ValueError(
                f"Expected demo_rgb_1hz (P, 2, {EMBED_DIM}), got {tuple(demo_rgb_1hz.shape)}"
            )
        rr = self.teacher.reference_reader
        P = demo_rgb_1hz.shape[0]
        if P > rr.max_prompt_tokens:
            demo_rgb_1hz = demo_rgb_1hz[: rr.max_prompt_tokens]
            P = rr.max_prompt_tokens

        full_mem = self.encode_full_demo_memory(demo_rgb_1hz.unsqueeze(0))  # (1, P, 768)
        h_mem = E5Policy.format_query_tokens(anchor_query_rgb)              # (M, 4, 768)
        M = h_mem.shape[0]
        tgt = h_mem + rr.prompt_current_obs_pos_emb                         # (M, 4, 768)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve `z_c = sum_i w_i * b_i` from Behavior Memory `(h_mem, b_mem)`."""
        if h_q is None:
            h_q = E5Policy.format_query_tokens(query_rgb_feats)
        z_c, weights = self.reader(h_q, h_mem, b_mem, mem_pad_mask=mem_pad_mask)
        return z_c, h_q, weights

    def adapt_latent(
        self,
        z_cross: torch.Tensor,
        h_q: torch.Tensor,
        *,
        delta_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute target latent `z_q`:
          - `prediction_mode == 'residual'`: `delta_z = G_theta(z_c, h_q)`, `z_q = z_c + delta_z`.
          - `prediction_mode == 'direct'`:   `z_q = bias + G_theta(z_c, h_q)` (NO skip from `z_c`),
            where `bias` is a fixed constant buffer (e.g. training `z_self` mean).
            Returns `(z_q, z_q - z_c)` so offline diagnostics remain directly comparable.
        """
        scale = float(delta_scale)
        if scale == 0.0:
            return z_cross, torch.zeros_like(z_cross)
        if self.prediction_mode == "direct":
            if scale != 1.0:
                raise ValueError(
                    f"delta_scale must be 1.0 or 0.0 in prediction_mode='direct', got {scale}"
                )
            g_out = self.residual_transformer(
                z_cross,
                h_q,
                output_scale=float(self.latent_scale_s_z.item()),
            )
            bias = self.direct_latent_bias.to(device=g_out.device, dtype=g_out.dtype)
            z_q = bias + g_out
            return z_q, z_q - z_cross

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
        """DDP-friendly E7 forward pass:
            w       = softmax(Q(h_q) K(h_mem)^T / sqrt(d))
            z_c     = sum_i w_i * b_i
            delta_z = G_theta(z_c, h_q)
            z_q     = z_c + delta_z
            L_z     = MSE((z_q - sg(z_self)) / s_z, 0)
            L       = lambda_a * L_diff(A_q | z_q) + lambda_z * L_z
        """
        h_q = E5Policy.format_query_tokens(query_rgb_feats).detach()
        h_mem_det = h_mem.detach()
        b_mem_det = b_mem.detach()
        z_s = z_self.detach()
        s_z = self.latent_scale_s_z.to(device=h_q.device, dtype=torch.float32).clamp_min(1e-6)

        # 1. Behavior Memory Retrieval: z_c = sum_i w_i * b_i
        z_c, _, attn_w = self.read_cross_latent(
            h_mem_det, b_mem_det, mem_pad_mask, query_rgb_feats, h_q=h_q
        )

        # 2. Single-step Latent Adaptation: z_q = z_c + G_theta(z_c, h_q)
        z_q, delta_z = self.adapt_latent(z_c, h_q)

        # 3. Normalized Latent Alignment Loss L_z = MSE((z_q - sg(z_self)) / s_z, 0)
        if lambda_z != 0.0:
            per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            latent_loss_norm = per_sample_lz.mean()
        else:
            with torch.no_grad():
                per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
                latent_loss_norm = per_sample_lz.mean()

        # 4. Diffusion Loss through FROZEN Stage 1 UNet conditioned ONLY on z_q
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

            astats = compute_attention_stats(attn_w, mem_pad_mask)

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
            "attn_entropy": astats["entropy"].mean(),
            "attn_norm_entropy": astats["norm_entropy"].mean(),
            "attn_max_weight": astats["max_weight"].mean(),
            "attn_eff_pos": astats["eff_pos_exp"].mean(),
            "attn_eff_ipr": astats["eff_pos_ipr"].mean(),
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
        """Closed-loop / offline inference from Behavior Memory `(h_mem, b_mem)`.
        Default: single-step Delta correction (`apply_delta=True`, `delta_scale=1.0`)."""
        z_cross, h_q, attn_weights = self.read_cross_latent(
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
            "attn_weights": attn_weights,
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
        """AdamW parameter groups over `self.reader` and `self.residual_transformer` ONLY."""
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
                "name": "simple_reader_decay",
                "params": reader_decay,
                "weight_decay": weight_decay,
                "lr": reader_lr,
            },
            {
                "name": "simple_reader_no_decay",
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
            "simple_reader_total": reader_total,
            "simple_reader_trainable": reader_trainable,
            "simple_reader_in_dim": self.reader.in_dim,
            "simple_reader_attn_dim": self.reader.attn_dim,
            "residual_transformer_total": res_total,
            "residual_transformer_trainable": res_trainable,
            "total_trainable": reader_trainable + res_trainable,
            "full_system_total_with_frozen_clip": (
                visual_encoder_params + teacher_total + reader_total + res_total
            ),
            "latent_scale_s_z": float(self.latent_scale_s_z.item()),
            "prediction_mode": self.prediction_mode,
            "direct_latent_bias_rms": (
                float(self.direct_latent_bias.float().square().mean().sqrt().item())
                if self.direct_latent_bias is not None
                else None
            ),
        }


def load_e7_policy(
    e7_ckpt: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E7Policy, Dict[str, Any]]:
    """Load frozen Stage 1 Teacher + trained E7 SimpleBehaviorReader & ResidualTransformer."""
    payload = torch.load(Path(e7_ckpt), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    s1_path = Path(stage1_ckpt) if stage1_ckpt is not None else Path(cfg["stage1_ckpt"])
    teacher = load_frozen_stage1_teacher(s1_path, device="cpu")
    prediction_mode = str(
        payload.get("prediction_mode", cfg.get("prediction_mode", "residual"))
    )
    direct_latent_bias = payload.get("direct_latent_bias", None)
    policy = E7Policy(
        teacher,
        latent_scale_s_z=float(
            payload.get("latent_scale_s_z", cfg.get("latent_scale_s_z", FIXED_LATENT_SCALE_S_Z))
        ),
        reader_attn_dim=int(cfg.get("reader_attn_dim", EMBED_DIM)),
        residual_layers=int(cfg.get("residual_layers", 3)),
        residual_heads=int(cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
        prediction_mode=prediction_mode,
        direct_latent_bias=direct_latent_bias,
        direct_out_proj_init_std=float(cfg.get("direct_out_proj_init_std", 0.002)),
    )
    policy.reader.load_state_dict(payload["reader"], strict=True)
    policy.residual_transformer.load_state_dict(payload["residual_transformer"], strict=True)
    policy.eval()
    return policy.to(device), payload
