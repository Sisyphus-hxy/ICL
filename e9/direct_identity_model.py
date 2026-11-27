"""Experiment B Model: E9-Direct + Exact Self Identity.

Strictly preserves E9 Stage 1 (frozen), Global Soft Reader (`SimpleBehaviorReader`), and
Direct Transfer (`ResidualTransformer`, `prediction_mode='direct'`, NO residual skip, NO
Source State Conditioning) without modifying `e7/model.py`.

Adds ONLY the Exact Self Identity branch on the Transfer module:
    z_id = T_theta(z_self, h_q)
    L_id = MSE((z_id - z_self) / s_z)
    L    = L_diff + 0.5 * L_z + 0.1 * L_id

Invariants:
  - `z_self` uses cached frozen E9 Stage 1 Local Encoder latents (detached).
  - The Identity branch does NOT pass through `Reader` and does NOT pass through `UNet`.
  - Consequently `d L_id / d theta_reader == 0` and `d L_id / d theta_teacher == 0`.
  - Checkpoints share exact state_dict keys with E9 Stage 2 Direct and load seamlessly via
    `load_e9_stage2_direct_policy`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from e5.model import E5Policy
from e7.model import E7Policy, compute_attention_stats

from . import EMBED_DIM, NUM_HEADS
from .stage1_local_model import E9Stage1LocalPolicy


class E9DirectIdentityPolicy(E7Policy):
    """Exact E9 Stage 2 Direct policy extended with the Exact Self Identity loss branch."""

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
        lambda_id: float = 0.1,
        return_loss_terms: bool = False,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing:
            Cross/Original branch:
                z_c, w = Reader(h_q, h_mem, b_mem)
                z_q    = T_theta(z_c, h_q)
                L_diff = diff_loss(A_q | z_q)   [via frozen Stage 1 UNet]
                L_z    = MSE((z_q - sg(z_self)) / s_z)
            Exact Self Identity branch (NO Reader, NO UNet):
                z_id   = T_theta(sg(z_self), h_q)
                L_id   = MSE((z_id - sg(z_self)) / s_z)
            Total:
                L      = lambda_a * L_diff + lambda_z * L_z + lambda_id * L_id
        """
        h_q = E5Policy.format_query_tokens(query_rgb_feats).detach()
        h_mem_det = h_mem.detach()
        b_mem_det = b_mem.detach()
        z_s = z_self.detach()
        s_z = self.latent_scale_s_z.to(device=h_q.device, dtype=torch.float32).clamp_min(1e-6)

        # 1. Original branch: Behavior Memory Retrieval -> Direct Transfer
        z_c, _, attn_w = self.read_cross_latent(
            h_mem_det, b_mem_det, mem_pad_mask, query_rgb_feats, h_q=h_q
        )
        z_q, delta_z = self.adapt_latent(z_c, h_q)

        # 2. Normalized Latent Alignment Loss L_z = MSE((z_q - sg(z_self)) / s_z)
        if lambda_z != 0.0:
            per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            latent_loss_norm = per_sample_lz.mean()
        else:
            with torch.no_grad():
                per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
                latent_loss_norm = per_sample_lz.mean()

        # 3. Diffusion Loss through FROZEN Stage 1 UNet conditioned ONLY on z_q
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

        # 4. Exact Self Identity branch: z_id = T_theta(z_self, h_q), NO Reader, NO UNet
        if lambda_id != 0.0:
            z_id, delta_id = self.adapt_latent(z_s, h_q)
            per_sample_lid = ((z_id.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            id_loss_norm = per_sample_lid.mean()
        else:
            with torch.no_grad():
                z_id, delta_id = self.adapt_latent(z_s, h_q)
                per_sample_lid = ((z_id.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
                id_loss_norm = per_sample_lid.mean()

        diff_term = (
            lambda_a * diff_loss
            if lambda_a != 0.0
            else torch.zeros((), device=z_q.device, dtype=torch.float32)
        )
        z_term = (
            lambda_z * latent_loss_norm
            if lambda_z != 0.0
            else torch.zeros((), device=z_q.device, dtype=torch.float32)
        )
        id_term = (
            lambda_id * id_loss_norm
            if lambda_id != 0.0
            else torch.zeros((), device=z_q.device, dtype=torch.float32)
        )
        cross_loss = diff_term + z_term
        total_loss = cross_loss + id_term

        with torch.no_grad():
            B = z_q.shape[0]
            zc_f = z_c.detach().float()
            zq_f = z_q.detach().float()
            zs_f = z_s.float()
            dz_f = delta_z.detach().float()
            zid_f = z_id.detach().float()
            did_f = delta_id.detach().float()

            latent_mse_raw = (zq_f - zs_f).square().mean()
            nodelta_mse_raw = (zc_f - zs_f).square().mean()
            nodelta_lz = ((zc_f - zs_f) / s_z).square().mean()
            id_mse_raw = (zid_f - zs_f).square().mean()
            id_delta_rms = torch.sqrt(did_f.square().mean() + 1e-12)

            cos_nodelta = F.cosine_similarity(
                zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).mean()
            cos_adapted = F.cosine_similarity(
                zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).mean()
            cos_id_self = F.cosine_similarity(
                zid_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
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

        out: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "diff_loss": diff_loss.detach(),
            "latent_loss_norm": latent_loss_norm.detach(),
            "id_loss_norm": id_loss_norm.detach(),
            "id_mse_raw": id_mse_raw,
            "id_delta_rms": id_delta_rms,
            "cos_id_self": cos_id_self,
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
        if return_loss_terms:
            out["cross_loss_tensor"] = cross_loss
            out["diff_term_tensor"] = diff_term
            out["z_term_tensor"] = z_term
            out["id_term_tensor"] = id_term
        return out


def build_e9_direct_identity_policy(
    teacher: E9Stage1LocalPolicy,
    *,
    latent_scale_s_z: float,
    direct_latent_bias: torch.Tensor,
    reader_attn_dim: int = EMBED_DIM,
    residual_layers: int = 3,
    residual_heads: int = NUM_HEADS,
    residual_ffn_dim: int = 3072,
    residual_dropout: float = 0.1,
    direct_out_proj_init_std: float = 0.002,
) -> E9DirectIdentityPolicy:
    """Construct from-scratch E9-Direct + Exact Self Identity policy over frozen E9 Stage 1."""
    return E9DirectIdentityPolicy(
        teacher,
        latent_scale_s_z=float(latent_scale_s_z),
        reader_attn_dim=int(reader_attn_dim),
        residual_layers=int(residual_layers),
        residual_heads=int(residual_heads),
        residual_ffn_dim=int(residual_ffn_dim),
        residual_dropout=float(residual_dropout),
        prediction_mode="direct",
        direct_latent_bias=direct_latent_bias,
        direct_out_proj_init_std=float(direct_out_proj_init_std),
    )
