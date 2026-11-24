"""E6 Stage 2.1: Joint Cross Reader (Q/K only) & Latent Delta Training (CLIP, From Stage 1).

Fixes the Stage 1 Behavior Latent Space while jointly learning:
  1. Cross Reader `R_S` initialized as an independent copy of Stage 1 `BehaviorReferenceReader`,
     with ONLY Attention Q/K projections unfrozen (`q_proj_weight`, `k_proj_weight`, and Q/K
     slices of `in_proj_bias`), while V projection (`v_proj_weight` + V slice of `in_proj_bias`),
     `out_proj`, FFN (`linear1`/`linear2`), LayerNorms, and positional embeddings remain strictly
     frozen.
  2. ResidualTransformer `G_psi(z_c, h_q) -> delta_z` newly created from scratch with
     zero-initialized `out_proj` (NEVER loaded from Stage 2).

Frozen Stage 1 Teacher (`PromptPooler`, Self Reader `R_T`, `ConditionalUnet1D`,
`ActionNormalizer`) stays completely frozen (`requires_grad=False`, `eval()` mode).
Diffusion Loss backpropagates through the frozen UNet into `z_q = z_c + delta_z`.

Equations:
    z_s     = sg(R_T(M_Q, h_q))            # cached Teacher self latent
    z_c     = R_S(M_D, h_q)                # online Cross Reader latent
    delta_z = G_psi(z_c, h_q)              # online Latent Delta
    z_q     = z_c + delta_z                # adapted latent fed to frozen UNet
    L_z     = MSE((z_q - z_s) / s_z)
    L_anc   = MSE((R_S(M_Q, h_q) - z_s) / s_z)
    L       = lambda_a * L_diff(A_q | z_q) + lambda_z * L_z + lambda_anchor * L_anc
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, extract_exec_actions
from e5.model import E5Policy, ResidualTransformer

from . import (
    EMBED_DIM,
    EXEC_HORIZON,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
    OUTPUT_ROOT,
)
from .stage1_model import BehaviorReferenceReader, E6Stage1Policy, load_frozen_stage1_teacher

DEFAULT_STAGE1_CKPT = OUTPUT_ROOT / "stage1/stage1_self_latent/best.pt"
FIXED_LATENT_SCALE_S_Z = 7.822748


def _make_zero_v_bias_hook(embed_dim: int):
    """Gradient hook that strictly zeroes the Value slice `[2*D:]` of `in_proj_bias`."""

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        out = grad.clone()
        out[2 * embed_dim :] = 0.0
        return out

    return _hook


def _split_mha_unfreeze_qk(mha: nn.MultiheadAttention) -> None:
    """Split packed `in_proj_weight (3D, D)` into PyTorch-native separate parameters
    `q_proj_weight` (trainable), `k_proj_weight` (trainable), and `v_proj_weight` (frozen),
    and mask `in_proj_bias[2*D:]` gradients to zero so Value bias stays bit-for-bit frozen.
    """
    D = mha.embed_dim
    if mha.in_proj_weight is not None:
        w = mha.in_proj_weight.detach()
        if w.shape != (3 * D, D):
            raise ValueError(f"Expected in_proj_weight {(3 * D, D)}, got {tuple(w.shape)}")
        mha.q_proj_weight = nn.Parameter(w[:D].clone(), requires_grad=True)
        mha.k_proj_weight = nn.Parameter(w[D : 2 * D].clone(), requires_grad=True)
        mha.v_proj_weight = nn.Parameter(w[2 * D :].clone(), requires_grad=False)
        mha.register_parameter("in_proj_weight", None)
        mha._qkv_same_embed_dim = False
    else:
        if (
            mha.q_proj_weight is None
            or mha.k_proj_weight is None
            or mha.v_proj_weight is None
        ):
            raise RuntimeError("MultiheadAttention missing Q/K/V projection weights")
        mha.q_proj_weight.requires_grad_(True)
        mha.k_proj_weight.requires_grad_(True)
        mha.v_proj_weight.requires_grad_(False)
        mha._qkv_same_embed_dim = False

    if mha.in_proj_bias is not None:
        if mha.in_proj_bias.shape != (3 * D,):
            raise ValueError(
                f"Expected in_proj_bias {(3 * D,)}, got {tuple(mha.in_proj_bias.shape)}"
            )
        mha.in_proj_bias.requires_grad_(True)
        mha.in_proj_bias.register_hook(_make_zero_v_bias_hook(D))


def configure_cross_reader_qk(
    reader: BehaviorReferenceReader,
    *,
    scope: str = "both",
) -> Dict[str, Any]:
    """Freeze all parameters of `reader` and unfreeze ONLY Attention Q/K projections.

    Args:
        reader: Independent `BehaviorReferenceReader` copy initialized from Stage 1.
        scope:
          - "both" (default): unfreeze Q/K projections of both `self_attn` and
            `multihead_attn` across all 6 decoder layers.
          - "cross_only": unfreeze Q/K projections of `multihead_attn` only across all 6 layers.
    Returns:
        Explicit audit report listing every unfrozen and frozen parameter name & count.
    """
    if scope not in ("both", "cross_only"):
        raise ValueError(f"Unsupported reader Q/K scope: {scope}")

    for param in reader.parameters():
        param.requires_grad_(False)

    attn_attr_names = ("self_attn", "multihead_attn") if scope == "both" else ("multihead_attn",)
    target_attn_modules: List[str] = []

    for layer_idx, layer in enumerate(reader.prompt_with_obs_decoder.layers):
        for attr in attn_attr_names:
            mha = getattr(layer, attr)
            if not isinstance(mha, nn.MultiheadAttention):
                raise TypeError(f"Expected nn.MultiheadAttention at layer {layer_idx}.{attr}")
            _split_mha_unfreeze_qk(mha)
            target_attn_modules.append(f"prompt_with_obs_decoder.layers.{layer_idx}.{attr}")

    # Strict audit of every single parameter name in `reader`
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    trainable_tensor_scalars = 0
    trainable_effective_scalars = 0
    frozen_effective_scalars = 0

    forbidden_substrings = (
        "v_proj_weight",
        "out_proj",
        "linear1",
        "linear2",
        "norm1",
        "norm2",
        "norm3",
        "prompt_pos_emb",
        "prompt_current_obs_pos_emb",
        "attention_sink_tokens",
    )

    for name, param in reader.named_parameters():
        numel = param.numel()
        if param.requires_grad:
            if any(bad in name for bad in forbidden_substrings):
                raise RuntimeError(f"Unexpected unfrozen Cross Reader parameter: {name}")
            if not (
                name.endswith(".q_proj_weight")
                or name.endswith(".k_proj_weight")
                or name.endswith(".in_proj_bias")
            ):
                raise RuntimeError(f"Unexpected trainable Cross Reader parameter: {name}")
            if scope == "cross_only" and ".self_attn." in name:
                raise RuntimeError(f"self_attn should be frozen in cross_only scope: {name}")
            trainable_names.append(name)
            trainable_tensor_scalars += numel
            if name.endswith(".in_proj_bias"):
                # Only Q/K slices (first 2/3) are active; V slice (last 1/3) is zero-grad frozen.
                active = (2 * numel) // 3
                trainable_effective_scalars += active
                frozen_effective_scalars += numel - active
            else:
                trainable_effective_scalars += numel
        else:
            frozen_names.append(name)
            frozen_effective_scalars += numel

    return {
        "scope": scope,
        "target_attention_modules": target_attn_modules,
        "num_target_attention_modules": len(target_attn_modules),
        "trainable_param_names": trainable_names,
        "frozen_param_names": frozen_names,
        "trainable_tensor_scalars": trainable_tensor_scalars,
        "trainable_effective_qk_scalars": trainable_effective_scalars,
        "frozen_effective_scalars": frozen_effective_scalars,
        "total_reader_scalars": trainable_effective_scalars + frozen_effective_scalars,
    }


class E6Stage2_1Policy(nn.Module):
    """E6 Stage 2.1: Frozen Stage 1 Self Teacher + Q/K-tunable Cross Reader +
    freshly initialized zero-out ResidualTransformer."""

    def __init__(
        self,
        teacher: E6Stage1Policy,
        *,
        latent_scale_s_z: float = FIXED_LATENT_SCALE_S_Z,
        reader_qk_scope: str = "both",
        residual_layers: int = 3,
        residual_heads: int = NUM_HEADS,
        residual_ffn_dim: int = 3072,
        residual_dropout: float = 0.1,
    ):
        super().__init__()
        # 1. Frozen Stage 1 Self Teacher (PromptPooler + Self Reader R_T + UNet + ActionNormalizer)
        self.teacher = teacher
        self._freeze_teacher()

        # 2. Independent Cross Reader R_S copied from Stage 1 Reader, ONLY Q/K unfrozen
        self.cross_reader: BehaviorReferenceReader = copy.deepcopy(teacher.reference_reader)
        self.reader_qk_scope = reader_qk_scope
        self.reader_qk_audit = configure_cross_reader_qk(
            self.cross_reader, scope=reader_qk_scope
        )
        # Keep frozen 6-layer Stage 1 backbone deterministic (no dropout noise on frozen V/FFN)
        self.cross_reader.eval()

        # Verify zero shared parameter storage between Teacher Reader and Cross Reader
        teacher_ptrs = {p.data_ptr() for p in self.teacher.parameters()}
        student_ptrs = {p.data_ptr() for p in self.cross_reader.parameters()}
        if teacher_ptrs & student_ptrs:
            raise RuntimeError("Teacher and Cross Reader share underlying tensor storage!")

        # 3. Freshly created ResidualTransformer from random init with zero-init out_proj
        self.residual_transformer = ResidualTransformer(
            embed_dim=EMBED_DIM,
            num_layers=residual_layers,
            num_heads=residual_heads,
            ffn_dim=residual_ffn_dim,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=residual_dropout,
        )

        # 4. Fixed training-set Self Latent RMS s_z (7.822748)
        self.register_buffer(
            "latent_scale_s_z",
            torch.tensor(float(latent_scale_s_z), dtype=torch.float32),
        )

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "E6Stage2_1Policy":
        """Keep frozen Teacher AND Cross Reader in eval() mode (zero dropout noise on the
        frozen 6-layer Stage 1 backbone; Q/K gradients still flow normally), while toggling
        `residual_transformer` train/eval mode."""
        super().train(mode)
        self.teacher.eval()
        self.cross_reader.eval()
        self.residual_transformer.train(mode)
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        """All trainable parameters in deterministic order (Cross Reader Q/K + ResidualTransformer)."""
        params: List[nn.Parameter] = []
        for p in self.cross_reader.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.residual_transformer.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    @torch.no_grad()
    def encode_demo_memory(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute frozen Demo memory tokens `M_D + prompt_pos_emb` of shape (B, P, 768).
        Identical for Teacher and Cross Reader because PromptPooler & prompt_pos_emb are frozen."""
        M_d = self.teacher.prompt_pooler(demo_rgb_feats, demo_pad_mask=demo_pad_mask)
        P = M_d.shape[1]
        rr = self.teacher.reference_reader
        mem = M_d + rr.prompt_pos_emb[:, :P, :]
        if demo_pad_mask is not None:
            mem = mem.masked_fill(demo_pad_mask.unsqueeze(-1), 0.0)
        return mem

    @torch.no_grad()
    def read_teacher_latent_from_memory(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read frozen Stage 1 Teacher latent `R_T(M, h_q)` from pre-pooled `demo_memory`."""
        h_q = E5Policy.format_query_tokens(query_rgb_feats)
        rr = self.teacher.reference_reader
        tgt = h_q + rr.prompt_current_obs_pos_emb
        z = rr.prompt_with_obs_decoder(
            tgt=tgt,
            memory=demo_memory,
            memory_key_padding_mask=demo_pad_mask,
            need_weights=False,
        )
        return z, h_q

    def read_cross_latent_from_memory(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        h_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read online Cross Reader latent `R_S(M, h_q)` WITH gradient flow into Q/K."""
        if h_q is None:
            h_q = E5Policy.format_query_tokens(query_rgb_feats)
        rr = self.cross_reader
        tgt = h_q + rr.prompt_current_obs_pos_emb
        z = rr.prompt_with_obs_decoder(
            tgt=tgt,
            memory=demo_memory,
            memory_key_padding_mask=demo_pad_mask,
            need_weights=False,
        )
        return z, h_q

    def adapt_latent(
        self, z_cross: torch.Tensor, h_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute `delta_z = G_psi(z_c, h_q)` and `z_q = z_c + delta_z` (preserves grad to `z_c`)."""
        delta_z = self.residual_transformer(z_cross, h_q)
        z_q = z_cross + delta_z
        return z_q, delta_z

    def adapt_latent_n_steps(
        self,
        z_cross: torch.Tensor,
        h_q: torch.Tensor,
        *,
        num_steps: int = 1,
        delta_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iteratively apply `T_alpha(z, h) = z + alpha * G_psi(z, h)` for `num_steps` steps.
        Returns `(z_final, z_final - z_cross)`."""
        steps = int(num_steps)
        scale = float(delta_scale)
        if steps < 0:
            raise ValueError(f"num_steps must be >= 0, got {steps}")
        if steps == 0 or scale == 0.0:
            return z_cross, torch.zeros_like(z_cross)
        if steps == 1:
            if scale == 1.0:
                return self.adapt_latent(z_cross, h_q)
            delta_z = self.residual_transformer(z_cross, h_q) * scale
            return z_cross + delta_z, delta_z
        z = z_cross
        for _ in range(steps):
            delta = self.residual_transformer(z, h_q)
            z = z + (delta if scale == 1.0 else delta * scale)
        return z, z - z_cross

    def forward(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_memory: torch.Tensor,
        query_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        z_self: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: Optional[torch.Tensor] = None,
        *,
        lambda_a: float = 1.0,
        lambda_z: float = 0.5,
        lambda_anchor: float = 0.1,
        lambda_id: float = 0.0,
        lambda_refine: float = 0.0,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """DDP-friendly joint Stage 2.1 / 2.3 forward:
            z_s      = sg(z_self)                  # cached Teacher R_T(M_Q, h_q)
            z_c      = R_S(M_D, h_q)               # trainable Cross Reader Q/K
            delta_z  = G_psi(z_c, h_q)             # trainable ResidualTransformer
            z_1      = z_c + delta_z               # 1-step adapted latent fed to frozen UNet
            z_anc    = R_S(M_Q, h_q)               # Self-reading anchor on Query's own demo M_Q
            delta_id = G_psi(z_s, h_q)             # Self fixed-point residual (Stage 2.3)
            z_2      = sg(z_1) + G_psi(sg(z_1), h_q) # 2nd-step refinement (Stage 2.3)
            L        = lambda_a * L_diff(A_q | z_1)
                     + lambda_z * MSE((z_1 - z_s) / s_z)
                     + lambda_anchor * MSE((z_anc - z_s) / s_z)
                     + lambda_id * MSE(delta_id / s_z, 0)
                     + lambda_refine * MSE((z_2 - z_s) / s_z, 0)
        """
        h_q = E5Policy.format_query_tokens(query_rgb_feats).detach()
        z_s = z_self.detach()
        s_z = self.latent_scale_s_z.to(device=h_q.device, dtype=torch.float32).clamp_min(1e-6)

        # 1. Online Cross Reader latent z_c = R_S(M_D, h_q) (with grad to R_S Q/K)
        z_c, _ = self.read_cross_latent_from_memory(
            demo_memory, demo_pad_mask, query_rgb_feats, h_q=h_q
        )

        # 2. Latent Delta delta_z = G_psi(z_c, h_q), z_q = z_1 = z_c + delta_z
        z_q, delta_z = self.adapt_latent(z_c, h_q)

        # 3. Normalized Latent Alignment Loss L_z = MSE((z_1 - z_s) / s_z)
        if lambda_z != 0.0:
            per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            latent_loss_norm = per_sample_lz.mean()
            latent_mse_raw = (z_q.detach().float() - z_s.float()).square().mean()
        else:
            with torch.no_grad():
                per_sample_lz = ((z_q.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
                latent_loss_norm = per_sample_lz.mean()
                latent_mse_raw = (z_q.float() - z_s.float()).square().mean()

        # 4. Self Anchor Loss L_anchor = MSE((R_S(M_Q, h_q) - z_s) / s_z)
        if lambda_anchor != 0.0:
            z_self_student, _ = self.read_cross_latent_from_memory(
                query_memory, query_pad_mask, query_rgb_feats, h_q=h_q
            )
            per_sample_anchor = (
                ((z_self_student.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            )
            anchor_loss_norm = per_sample_anchor.mean()
            anchor_mse_raw = (z_self_student.detach().float() - z_s.float()).square().mean()
        else:
            with torch.no_grad():
                z_self_student, _ = self.read_cross_latent_from_memory(
                    query_memory, query_pad_mask, query_rgb_feats, h_q=h_q
                )
                per_sample_anchor = (
                    ((z_self_student.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
                )
                anchor_loss_norm = per_sample_anchor.mean()
                anchor_mse_raw = (z_self_student.float() - z_s.float()).square().mean()

        # 5. Diffusion Loss through FROZEN UNet conditioned strictly on z_1 (grad flows to z_1).
        # When lambda_a == 0.0 (e.g. Pure Latent Alignment), skip UNet forward completely.
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

        # 6. Stage 2.3 Self Identity Fixed-Point Loss: L_id = MSE(G_psi(z_s, h_q) / s_z, 0)
        # z_s and h_q are detached -> gradients flow strictly into ResidualTransformer G_psi.
        if lambda_id != 0.0:
            delta_id = self.residual_transformer(z_s, h_q)
            per_sample_lid = (delta_id.float() / s_z).square().mean(dim=(1, 2))
            id_loss_norm = per_sample_lid.mean()
            id_delta_rms = torch.sqrt(delta_id.detach().float().square().mean() + 1e-12)
        else:
            id_loss_norm = torch.zeros((), device=z_q.device, dtype=torch.float32)
            id_delta_rms = torch.zeros((), device=z_q.device, dtype=torch.float32)

        # 7. Stage 2.3 2nd-Step Refinement Loss: z_2 = sg(z_1) + G_psi(sg(z_1), h_q),
        # L_refine = MSE((z_2 - z_s) / s_z, 0). Input z_1 is detached, 2nd G_psi forward has grad.
        if lambda_refine != 0.0:
            z1_sg = z_q.detach()
            z2, delta_2 = self.adapt_latent(z1_sg, h_q)
            per_sample_lref = ((z2.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
            refine_loss_norm = per_sample_lref.mean()
            refine_delta_rms = torch.sqrt(delta_2.detach().float().square().mean() + 1e-12)
        else:
            refine_loss_norm = torch.zeros((), device=z_q.device, dtype=torch.float32)
            refine_delta_rms = torch.zeros((), device=z_q.device, dtype=torch.float32)

        total_loss = (
            lambda_a * diff_loss
            if lambda_a != 0.0
            else torch.zeros((), device=z_q.device, dtype=torch.float32)
        )
        if lambda_z != 0.0:
            total_loss = total_loss + lambda_z * latent_loss_norm
        if lambda_anchor != 0.0:
            total_loss = total_loss + lambda_anchor * anchor_loss_norm
        if lambda_id != 0.0:
            total_loss = total_loss + lambda_id * id_loss_norm
        if lambda_refine != 0.0:
            total_loss = total_loss + lambda_refine * refine_loss_norm

        with torch.no_grad():
            B = z_q.shape[0]
            zc_f = z_c.detach().float()
            zq_f = z_q.detach().float()
            zs_f = z_s.float()
            cos_nodelta = F.cosine_similarity(
                zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).mean()
            cos_adapted = F.cosine_similarity(
                zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).mean()
            nodelta_lz = ((zc_f - zs_f) / s_z).square().mean()
            delta_sq = delta_z.detach().float().square().mean(dim=(1, 2))
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

        return {
            "loss": total_loss,
            "diff_loss": diff_loss.detach(),
            "latent_loss_norm": latent_loss_norm.detach(),
            "latent_mse_raw": latent_mse_raw.detach(),
            "anchor_loss_norm": anchor_loss_norm.detach(),
            "anchor_mse_raw": anchor_mse_raw.detach(),
            "id_loss_norm": id_loss_norm.detach(),
            "id_delta_rms": id_delta_rms.detach(),
            "refine_loss_norm": refine_loss_norm.detach(),
            "refine_delta_rms": refine_delta_rms.detach(),
            "nodelta_loss_norm": nodelta_lz,
            "cos_nodelta_self": cos_nodelta,
            "cos_adapted_self": cos_adapted,
            "delta_rms": delta_rms,
            "cross_delta_rms": cross_delta_rms,
            "self_delta_rms": self_delta_rms,
            "z_cross_rms": z_cross_rms,
            "n_self": n_s,
            "n_cross": n_c,
        }

    @torch.inference_mode()
    def predict_action_from_memory(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        apply_delta: bool = True,
        num_delta_steps: Optional[int] = None,
        delta_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Closed-loop / offline inference using the adapted Cross Reader `R_S` (+ 0/1/k-step Delta)."""
        z_cross, h_q = self.read_cross_latent_from_memory(
            demo_memory, demo_pad_mask, query_rgb_feats
        )
        steps = (1 if apply_delta else 0) if num_delta_steps is None else int(num_delta_steps)
        z_cond, delta_z = self.adapt_latent_n_steps(
            z_cross, h_q, num_steps=steps, delta_scale=delta_scale
        )
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
            "num_delta_steps": steps,
            "delta_scale": float(delta_scale),
        }

    def get_optim_groups(
        self,
        *,
        reader_lr: float = 3.0e-6,
        residual_lr: float = 1.0e-4,
        weight_decay: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Dual-LR AdamW parameter groups:
          - Group 0: Cross Reader Q/K weights (`lr=reader_lr`, `weight_decay=weight_decay`)
          - Group 1: Cross Reader Q/K biases  (`lr=reader_lr`, `weight_decay=0.0`)
          - Group 2: ResidualTransformer decay weights (`lr=residual_lr`, `weight_decay=weight_decay`)
          - Group 3: ResidualTransformer no-decay params (`lr=residual_lr`, `weight_decay=0.0`)
        """
        reader_decay: List[nn.Parameter] = []
        reader_no_decay: List[nn.Parameter] = []
        for name, param in self.cross_reader.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name:
                reader_no_decay.append(param)
            else:
                reader_decay.append(param)

        res_decay: List[nn.Parameter] = []
        res_no_decay: List[nn.Parameter] = []
        no_decay_keywords = ("bias", "norm", "pos_emb")
        for name, param in self.residual_transformer.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                res_no_decay.append(param)
            else:
                res_decay.append(param)

        teacher_trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
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
                "name": "cross_reader_qk_decay",
                "params": reader_decay,
                "weight_decay": weight_decay,
                "lr": reader_lr,
            },
            {
                "name": "cross_reader_qk_no_decay",
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
        teacher_trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        reader_total = sum(p.numel() for p in self.cross_reader.parameters())
        reader_trainable_tensors = sum(
            p.numel() for p in self.cross_reader.parameters() if p.requires_grad
        )
        reader_trainable_eff = self.reader_qk_audit["trainable_effective_qk_scalars"]
        res_total = sum(p.numel() for p in self.residual_transformer.parameters())
        res_trainable = sum(
            p.numel() for p in self.residual_transformer.parameters() if p.requires_grad
        )
        return {
            "visual_encoder_frozen": visual_encoder_params,
            "stage1_teacher_frozen": teacher_total,
            "stage1_teacher_trainable": teacher_trainable,
            "cross_reader_total": reader_total,
            "cross_reader_trainable_tensor_scalars": reader_trainable_tensors,
            "cross_reader_trainable_effective_qk": reader_trainable_eff,
            "cross_reader_frozen_effective": self.reader_qk_audit["frozen_effective_scalars"],
            "cross_reader_qk_scope": self.reader_qk_scope,
            "residual_transformer_total": res_total,
            "residual_transformer_trainable": res_trainable,
            "total_trainable_effective": reader_trainable_eff + res_trainable,
            "total_trainable_tensor_scalars": reader_trainable_tensors + res_trainable,
            "latent_scale_s_z": float(self.latent_scale_s_z.item()),
        }


def load_stage2_1_policy(
    stage2_1_ckpt: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E6Stage2_1Policy, Dict[str, Any]]:
    """Load frozen Stage 1 Teacher + trained Stage 2.1 Cross Reader & ResidualTransformer."""
    payload = torch.load(Path(stage2_1_ckpt), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    s1_path = Path(stage1_ckpt) if stage1_ckpt is not None else Path(cfg["stage1_ckpt"])
    teacher = load_frozen_stage1_teacher(s1_path, device="cpu")
    policy = E6Stage2_1Policy(
        teacher,
        latent_scale_s_z=float(
            payload.get("latent_scale_s_z", cfg.get("latent_scale_s_z", FIXED_LATENT_SCALE_S_Z))
        ),
        reader_qk_scope=str(cfg.get("reader_qk_scope", "both")),
        residual_layers=int(cfg.get("residual_layers", 3)),
        residual_heads=int(cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
    )
    policy.cross_reader.load_state_dict(payload["cross_reader"], strict=True)
    policy.residual_transformer.load_state_dict(payload["residual_transformer"], strict=True)
    policy.eval()
    return policy.to(device), payload
