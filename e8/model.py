"""E8 Model: Source-conditioned Latent Behavior Transfer (CLIP).

Core architecture (minimal extension of E7-Direct):
  1. Frozen Stage 1 Self Teacher (`e6/outputs/stage1/stage1_self_latent/best.pt`):
     - Reuses E7 Behavior Memory cache (stride=16): `(h_i^D, b_i)` where `b_i = R_T(M_D, h_i^D)`.
  2. Global Soft Reader (`SourceConditionedBehaviorReader`, subclass of `SimpleBehaviorReader`):
     - Exact same Q/K attention weights `w_i = softmax_i(Q(h_q) K(h_i^D)^T / sqrt(d))`.
     - Shared attention weights simultaneously read behavior latent and source visual state:
         z_c   = sum_i w_i * b_i
         h_ref = sum_i w_i * h_i^D
     - Strictly NO trainable V projection, NO Output projection, NO Query residual.
  3. `SourceConditionedTransferTransformer`:
     - Direct Prediction (NO residual skip connection from `z_c`):
         z_q = bias + G_theta(z_c, h_ref, h_q)
     - Exact same 3-layer TransformerDecoder backbone (hidden=768, 8 heads, FFN=3072, dropout=0.1)
       as E7-Direct, with `tgt` = Reference Behavior `z_c` (4 tokens) and
       `memory` = concat([Source State `h_ref` (4 tokens), Target State `h_q` (4 tokens)]) (8 tokens).
     - Independent type embeddings for all three input types (`z_c`, `h_ref`, `h_q`).
     - Direct output initialization strictly matches E7-Direct:
         * fixed non-trainable `direct_latent_bias = mean(z_self_train)`
         * fixed `output_scale = s_z = 7.822748`
         * `out_proj.weight ~ N(0, 0.002^2)`, `out_proj.bias = 0`
  4. Frozen Stage 1 UNet conditioned ONLY on `z_q`:
         L = L_diff(A_q | z_q) + 0.5 * MSE((z_q - sg(z_self)) / s_z, 0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, extract_exec_actions
from e5.model import E5Policy, bpp_init_weights
from e6.stage1_model import E6Stage1Policy, load_frozen_stage1_teacher
from e7.model import (
    SimpleBehaviorReader,
    compute_attention_stats,
    compute_behavior_anchor_steps,
)

from . import (
    EMBED_DIM,
    EXEC_HORIZON,
    FIXED_LATENT_SCALE_S_Z,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
)


class SourceConditionedBehaviorReader(SimpleBehaviorReader):
    """Global Soft Attention Reader reading BOTH `z_c` and `h_ref` with the EXACT SAME weights:
        w_i   = softmax_i( Q(h_q) K(h_i^D)^T / sqrt(d) )
        z_c   = sum_i w_i * b_i
        h_ref = sum_i w_i * h_i^D
    """

    def forward(
        self,
        h_q: torch.Tensor,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_q: (B, 4, 768) current Query visual tokens.
            h_mem: (B, M, 4, 768) Behavior Memory visual tokens h_i^D.
            b_mem: (B, M, 4, 768) frozen Stage 1 Behavior latents b_i.
            mem_pad_mask: optional (B, M) bool mask where True = padded / invalid entry.
        Returns:
            z_c: (B, 4, 768) sum_i w_i * b_i
            h_ref: (B, 4, 768) sum_i w_i * h_i^D (using IDENTICAL weights w_i)
            weights: (B, M) float32 reading weights summing to 1 over valid entries.
        """
        if b_mem.shape != h_mem.shape:
            raise ValueError(
                f"Expected b_mem shape {tuple(h_mem.shape)} matching h_mem, got {tuple(b_mem.shape)}"
            )
        B, M, _, _ = h_mem.shape
        q = self.encode_query(h_q)    # (B, d)
        k = self.encode_keys(h_mem)   # (B, M, d)

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
        h_ref = torch.einsum("bm,bmkd->bkd", weights.to(dtype=h_mem.dtype), h_mem)
        return z_c, h_ref, weights


class SourceConditionedTransferTransformer(nn.Module):
    """3-layer Transformer Decoder (hidden=768, 8 heads, FFN=3072) mapping
    `(z_c, h_ref, h_q) -> g_out` of shape `(B, 4, 768)`.

    Three input types have independent learned type embeddings:
      - Reference Behavior `z_c`   -> `tgt` (4 tokens)
      - Source State       `h_ref` -> first 4 tokens of `memory`
      - Target State       `h_q`   -> last 4 tokens of `memory`
    Source State and Target State share the 4-slot camera/timestep positional embedding
    `state_pos_emb` so corresponding views align while `source_type_emb` vs `target_type_emb`
    cleanly separate source vs target roles.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 3,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = 3072,
        num_query_tokens: int = NUM_QUERY_TOKENS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_query_tokens = int(num_query_tokens)

        layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ref_in_norm = nn.LayerNorm(self.embed_dim)
        self.source_in_norm = nn.LayerNorm(self.embed_dim)
        self.query_in_norm = nn.LayerNorm(self.embed_dim)

        self.decoder = nn.TransformerDecoder(decoder_layer=layer, num_layers=int(num_layers))

        # Slot positional embeddings for the 4 tokens [t-1:av, t-1:eh, t:av, t:eh]
        self.ref_pos_emb = nn.Parameter(torch.randn(1, self.num_query_tokens, self.embed_dim))
        self.state_pos_emb = nn.Parameter(torch.randn(1, self.num_query_tokens, self.embed_dim))

        # Three independent learned type embeddings (Reference Behavior, Source State, Target State)
        self.behavior_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.source_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.target_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self) -> None:
        self.decoder.apply(bpp_init_weights)
        for norm in (
            self.ref_in_norm,
            self.source_in_norm,
            self.query_in_norm,
            self.out_norm,
        ):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for emb in (
            self.ref_pos_emb,
            self.state_pos_emb,
            self.behavior_type_emb,
            self.source_type_emb,
            self.target_type_emb,
        ):
            nn.init.normal_(emb, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_c: torch.Tensor,
        h_ref: torch.Tensor,
        h_q: torch.Tensor,
        *,
        output_scale: float,
    ) -> torch.Tensor:
        if z_c.shape != h_q.shape:
            raise ValueError(f"z_c {tuple(z_c.shape)} and h_q {tuple(h_q.shape)} must match")
        if h_ref.shape != h_q.shape:
            raise ValueError(f"h_ref {tuple(h_ref.shape)} and h_q {tuple(h_q.shape)} must match")

        tgt = self.ref_in_norm(z_c) + self.ref_pos_emb + self.behavior_type_emb
        src_tok = self.source_in_norm(h_ref) + self.state_pos_emb + self.source_type_emb
        qry_tok = self.query_in_norm(h_q) + self.state_pos_emb + self.target_type_emb
        memory = torch.cat([src_tok, qry_tok], dim=1)  # (B, 8, 768)

        h = self.decoder(tgt=tgt, memory=memory)
        out = self.out_proj(self.out_norm(h))
        return out * float(output_scale)


class E8Policy(nn.Module):
    """E8 Source-conditioned Latent Behavior Transfer Policy (Direct Prediction).

    - Frozen Stage 1 Teacher (`teacher`)
    - From-scratch `SourceConditionedBehaviorReader` (`reader`, Q/K only, shared weights for z_c & h_ref)
    - From-scratch `SourceConditionedTransferTransformer` (`transfer_transformer`)
    - Direct Prediction strictly matching E7-Direct (`z_q = direct_latent_bias + G_theta(z_c, h_ref, h_q)`)
    - Frozen Stage 1 UNet conditioned ONLY on `z_q`.
    """

    def __init__(
        self,
        teacher: E6Stage1Policy,
        *,
        latent_scale_s_z: float = FIXED_LATENT_SCALE_S_Z,
        reader_attn_dim: int = EMBED_DIM,
        transformer_layers: int = 3,
        transformer_heads: int = NUM_HEADS,
        transformer_ffn_dim: int = 3072,
        transformer_dropout: float = 0.1,
        direct_latent_bias: Optional[torch.Tensor] = None,
        direct_out_proj_init_std: float = 0.002,
    ):
        super().__init__()
        self.prediction_mode = "direct"
        self.direct_out_proj_init_std = float(direct_out_proj_init_std)
        self.teacher = teacher
        self._freeze_teacher()

        self.reader = SourceConditionedBehaviorReader(
            num_query_tokens=NUM_QUERY_TOKENS,
            embed_dim=EMBED_DIM,
            attn_dim=reader_attn_dim,
        )
        self.transfer_transformer = SourceConditionedTransferTransformer(
            embed_dim=EMBED_DIM,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ffn_dim=transformer_ffn_dim,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=transformer_dropout,
        )

        self.register_buffer(
            "latent_scale_s_z",
            torch.tensor(float(latent_scale_s_z), dtype=torch.float32),
        )

        # Direct output initialization strictly matching E7-Direct
        if self.direct_out_proj_init_std > 0.0:
            nn.init.normal_(
                self.transfer_transformer.out_proj.weight,
                mean=0.0,
                std=self.direct_out_proj_init_std,
            )
        else:
            nn.init.zeros_(self.transfer_transformer.out_proj.weight)
        nn.init.zeros_(self.transfer_transformer.out_proj.bias)

        bias_buf = torch.zeros((1, NUM_QUERY_TOKENS, EMBED_DIM), dtype=torch.float32)
        self.register_buffer("direct_latent_bias", bias_buf)
        if direct_latent_bias is not None:
            self.set_direct_latent_bias(direct_latent_bias)

    def set_direct_latent_bias(self, mean_latent: torch.Tensor) -> None:
        """Set fixed (non-trainable) output bias buffer for Direct Prediction."""
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

    def train(self, mode: bool = True) -> "E8Policy":
        """Keep frozen Stage 1 Teacher permanently in eval() mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Deterministic list of all trainable parameters (Reader + TransferTransformer)."""
        params: List[nn.Parameter] = []
        for p in self.reader.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.transfer_transformer.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    @torch.no_grad()
    def encode_full_demo_memory(
        self,
        demo_rgb_1hz: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode complete 1Hz Demo CLIP features `(B, P, 2, 768)` -> `M_D + prompt_pos_emb`."""
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

    def read_cross_latent_and_source(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        h_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve `z_c = sum_i w_i b_i` and `h_ref = sum_i w_i h_i^D` using shared weights."""
        if h_q is None:
            h_q = E5Policy.format_query_tokens(query_rgb_feats)
        z_c, h_ref, weights = self.reader(h_q, h_mem, b_mem, mem_pad_mask=mem_pad_mask)
        return z_c, h_ref, h_q, weights

    def transfer_latent(
        self,
        z_cross: torch.Tensor,
        h_ref: torch.Tensor,
        h_q: torch.Tensor,
        *,
        apply_transfer: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Direct Prediction: `z_q = direct_latent_bias + G_theta(z_c, h_ref, h_q)` (NO skip).
        Returns `(z_q, z_q - z_c)` for direct comparability with E7 diagnostics.
        """
        if not apply_transfer:
            return z_cross, torch.zeros_like(z_cross)
        g_out = self.transfer_transformer(
            z_cross,
            h_ref,
            h_q,
            output_scale=float(self.latent_scale_s_z.item()),
        )
        bias = self.direct_latent_bias.to(device=g_out.device, dtype=g_out.dtype)
        z_q = bias + g_out
        return z_q, z_q - z_cross

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
        """DDP-friendly E8 forward pass:
            w     = softmax(Q(h_q) K(h_mem)^T / sqrt(d))
            z_c   = sum_i w_i * b_i
            h_ref = sum_i w_i * h_i^D
            z_q   = direct_latent_bias + G_theta(z_c, h_ref, h_q)
            L_z   = MSE((z_q - sg(z_self)) / s_z, 0)
            L     = lambda_a * L_diff(A_q | z_q) + lambda_z * L_z
        """
        h_q = E5Policy.format_query_tokens(query_rgb_feats).detach()
        h_mem_det = h_mem.detach()
        b_mem_det = b_mem.detach()
        z_s = z_self.detach()
        s_z = self.latent_scale_s_z.to(device=h_q.device, dtype=torch.float32).clamp_min(1e-6)

        # 1. Shared-weight readout of Reference Behavior z_c and Source State h_ref
        z_c, h_ref, _, attn_w = self.read_cross_latent_and_source(
            h_mem_det, b_mem_det, mem_pad_mask, query_rgb_feats, h_q=h_q
        )

        # 2. Direct source-conditioned transfer prediction (NO skip from z_c)
        z_q, delta_z = self.transfer_latent(z_c, h_ref, h_q, apply_transfer=True)

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
            href_f = h_ref.detach().float()
            hq_f = h_q.detach().float()

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
            source_target_cos = F.cosine_similarity(
                href_f.reshape(B, -1), hq_f.reshape(B, -1), dim=-1
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
            "source_target_cos": source_target_cos,
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
        apply_transfer: bool = True,
        h_ref_override: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Closed-loop / offline inference from Behavior Memory `(h_mem, b_mem)`."""
        z_cross, h_ref, h_q, attn_weights = self.read_cross_latent_and_source(
            h_mem, b_mem, mem_pad_mask, query_rgb_feats
        )
        h_ref_used = h_ref if h_ref_override is None else h_ref_override
        z_cond, delta_z = self.transfer_latent(
            z_cross, h_ref_used, h_q, apply_transfer=apply_transfer
        )

        action_10d = self.teacher.sample_from_z_self(
            z_cond, generator=generator, init_noise=init_noise
        )
        return {
            "action_10d": action_10d,
            "action_7d": action_10d_to_7d(action_10d),
            "exec_action_7d": extract_exec_actions(action_10d, exec_horizon=EXEC_HORIZON),
            "z_cross": z_cross,
            "h_ref": h_ref,
            "h_ref_used": h_ref_used,
            "delta_z": delta_z,
            "z_adapted": z_cond,
            "attn_weights": attn_weights,
            "apply_transfer": bool(apply_transfer),
        }

    def get_optim_groups(
        self,
        *,
        reader_lr: float = 1.0e-4,
        transformer_lr: float = 1.0e-4,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """AdamW parameter groups over `self.reader` and `self.transfer_transformer` ONLY."""
        no_decay_keywords = ("bias", "norm", "pos_emb", "type_emb")

        reader_decay: List[nn.Parameter] = []
        reader_no_decay: List[nn.Parameter] = []
        for name, param in self.reader.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                reader_no_decay.append(param)
            else:
                reader_decay.append(param)

        xf_decay: List[nn.Parameter] = []
        xf_no_decay: List[nn.Parameter] = []
        for name, param in self.transfer_transformer.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                xf_no_decay.append(param)
            else:
                xf_decay.append(param)

        teacher_trainable = sum(
            p.numel() for p in self.teacher.parameters() if p.requires_grad
        )
        if teacher_trainable != 0:
            raise RuntimeError(f"Frozen teacher has {teacher_trainable} trainable params!")

        all_trainable = [p for p in self.parameters() if p.requires_grad]
        grouped_count = (
            len(reader_decay) + len(reader_no_decay) + len(xf_decay) + len(xf_no_decay)
        )
        if grouped_count != len(all_trainable):
            raise RuntimeError(
                f"Optimizer parameter grouping mismatch: grouped={grouped_count} != trainable={len(all_trainable)}"
            )

        return [
            {
                "name": "reader_decay",
                "params": reader_decay,
                "weight_decay": weight_decay,
                "lr": reader_lr,
            },
            {
                "name": "reader_no_decay",
                "params": reader_no_decay,
                "weight_decay": 0.0,
                "lr": reader_lr,
            },
            {
                "name": "transfer_transformer_decay",
                "params": xf_decay,
                "weight_decay": weight_decay,
                "lr": transformer_lr,
            },
            {
                "name": "transfer_transformer_no_decay",
                "params": xf_no_decay,
                "weight_decay": 0.0,
                "lr": transformer_lr,
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
        xf_total = sum(p.numel() for p in self.transfer_transformer.parameters())
        xf_trainable = sum(
            p.numel() for p in self.transfer_transformer.parameters() if p.requires_grad
        )
        return {
            "visual_encoder_frozen": visual_encoder_params,
            "stage1_teacher_frozen": teacher_total,
            "stage1_teacher_trainable": teacher_trainable,
            "reader_total": reader_total,
            "reader_trainable": reader_trainable,
            "reader_in_dim": self.reader.in_dim,
            "reader_attn_dim": self.reader.attn_dim,
            "transfer_transformer_total": xf_total,
            "transfer_transformer_trainable": xf_trainable,
            "total_trainable": reader_trainable + xf_trainable,
            "full_system_total_with_frozen_clip": (
                visual_encoder_params + teacher_total + reader_total + xf_total
            ),
            "latent_scale_s_z": float(self.latent_scale_s_z.item()),
            "prediction_mode": self.prediction_mode,
            "source_conditioned": True,
            "direct_latent_bias_rms": float(
                self.direct_latent_bias.float().square().mean().sqrt().item()
            ),
        }


def load_e8_policy(
    e8_ckpt: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E8Policy, Dict[str, Any]]:
    """Load frozen Stage 1 Teacher + trained E8 Reader & SourceConditionedTransferTransformer."""
    payload = torch.load(Path(e8_ckpt), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    s1_path = Path(stage1_ckpt) if stage1_ckpt is not None else Path(cfg["stage1_ckpt"])
    teacher = load_frozen_stage1_teacher(s1_path, device="cpu")
    direct_latent_bias = payload.get("direct_latent_bias", None)
    policy = E8Policy(
        teacher,
        latent_scale_s_z=float(
            payload.get("latent_scale_s_z", cfg.get("latent_scale_s_z", FIXED_LATENT_SCALE_S_Z))
        ),
        reader_attn_dim=int(cfg.get("reader_attn_dim", EMBED_DIM)),
        transformer_layers=int(cfg.get("transformer_layers", 3)),
        transformer_heads=int(cfg.get("transformer_heads", NUM_HEADS)),
        transformer_ffn_dim=int(cfg.get("transformer_ffn_dim", 3072)),
        transformer_dropout=float(cfg.get("transformer_dropout", 0.1)),
        direct_latent_bias=direct_latent_bias,
        direct_out_proj_init_std=float(cfg.get("direct_out_proj_init_std", 0.002)),
    )
    policy.reader.load_state_dict(payload["reader"], strict=True)
    policy.transfer_transformer.load_state_dict(payload["transfer_transformer"], strict=True)
    policy.eval()
    return policy.to(device), payload


__all__ = [
    "E8Policy",
    "SourceConditionedBehaviorReader",
    "SourceConditionedTransferTransformer",
    "compute_attention_stats",
    "compute_behavior_anchor_steps",
    "load_e8_policy",
]
