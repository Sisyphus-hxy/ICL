"""E6 Stage 2: Latent Behavior Transfer (CLIP, RGB-only).

Freezes the entire Stage 1 EMA Teacher (`PromptPooler`, `BehaviorReferenceReader`,
`ConditionalUnet1D`, `ActionNormalizer`) and trains ONLY a zero-initialized
`ResidualTransformer` `G(z_cross, h_q) -> delta_z` such that:

    z_adapted = z_cross + delta_z
    L = lambda_a * L_diff(A_q | z_adapted) + lambda_z * mean(((z_adapted - sg(z_self)) / s_z)^2)

Teacher UNet parameters are frozen (`requires_grad=False`), while gradients from
`L_diff` backpropagate through the frozen UNet into `z_adapted -> delta_z -> G`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, extract_exec_actions
from e5.model import E5Policy, ResidualTransformer

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    EMBED_DIM,
    EXEC_HORIZON,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
    OUTPUT_ROOT,
)
from .stage1_model import E6Stage1Policy, load_frozen_stage1_teacher

DEFAULT_STAGE1_CKPT = OUTPUT_ROOT / "stage1/stage1_self_latent/best.pt"


class E6Stage2Policy(nn.Module):
    """Frozen Stage 1 Teacher + Trainable zero-initialized ResidualTransformer."""

    def __init__(
        self,
        teacher: E6Stage1Policy,
        *,
        latent_scale_s_z: float = 1.0,
        residual_layers: int = 3,
        residual_heads: int = NUM_HEADS,
        residual_ffn_dim: int = 3072,
        residual_dropout: float = 0.1,
    ):
        super().__init__()
        self.teacher = teacher
        self._freeze_teacher()

        # Trainable ResidualTransformer: (z_cross, h_q) -> delta_z, zero-init out_proj
        self.residual_transformer = ResidualTransformer(
            embed_dim=EMBED_DIM,
            num_layers=residual_layers,
            num_heads=residual_heads,
            ffn_dim=residual_ffn_dim,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=residual_dropout,
        )
        # Fixed training-set Self Latent RMS `s_z` stored as a non-trainable buffer
        self.register_buffer(
            "latent_scale_s_z",
            torch.tensor(float(latent_scale_s_z), dtype=torch.float32),
        )

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "E6Stage2Policy":
        """Keep the frozen Stage 1 Teacher permanently in eval() mode (no dropout)."""
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def encode_demo_memory(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute frozen Demo memory tokens `M_D + prompt_pos_emb` of shape (B, P, 768)."""
        M_d = self.teacher.prompt_pooler(demo_rgb_feats, demo_pad_mask=demo_pad_mask)
        P = M_d.shape[1]
        rr = self.teacher.reference_reader
        mem = M_d + rr.prompt_pos_emb[:, :P, :]
        if demo_pad_mask is not None:
            mem = mem.masked_fill(demo_pad_mask.unsqueeze(-1), 0.0)
        return mem

    @torch.no_grad()
    def read_latent_from_memory(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read frozen Teacher latent `z` (B, 4, 768) from pre-pooled `demo_memory`
        (M_D + prompt_pos_emb) and return `(z, h_q)`."""
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

    @torch.no_grad()
    def read_latent_from_raw(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read frozen Teacher latent `z` (B, 4, 768) directly from raw demo CLIP features."""
        h_q = E5Policy.format_query_tokens(query_rgb_feats)
        z, _ = self.teacher.encode_z_self(
            demo_rgb_feats, demo_pad_mask, query_rgb_feats, need_weights=False
        )
        return z, h_q

    def adapt_latent(
        self, z_cross: torch.Tensor, h_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute `delta_z = G(z_cross, h_q)` and `z_adapted = z_cross + delta_z`."""
        delta_z = self.residual_transformer(z_cross, h_q)
        z_adapted = z_cross + delta_z
        return z_adapted, delta_z

    def compute_loss_from_latents(
        self,
        z_cross: torch.Tensor,
        h_q: torch.Tensor,
        z_self: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: Optional[torch.Tensor] = None,
        *,
        lambda_z: float = 0.5,
        lambda_a: float = 1.0,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute E6 Stage 2 loss:
            L = lambda_a * L_diff(A_q | z_cross + delta_z)
              + lambda_z * mean(((z_cross + delta_z - sg(z_self)) / s_z)^2)

        Frozen UNet parameters have `requires_grad=False`, while `L_diff` backpropagates
        through the frozen UNet to `z_adapted -> delta_z -> ResidualTransformer`.
        """
        z_cross_det = z_cross.detach()
        h_q_det = h_q.detach()
        z_self_sg = z_self.detach()

        z_adapted, delta_z = self.adapt_latent(z_cross_det, h_q_det)

        # Normalized element-wise latent MSE: mean(((z_adapted - sg(z_self)) / s_z)^2)
        s_z = self.latent_scale_s_z.to(device=z_adapted.device, dtype=torch.float32).clamp_min(1e-6)
        diff_z = (z_adapted.float() - z_self_sg.float()) / s_z
        per_sample_lz = diff_z.square().mean(dim=(1, 2))
        latent_loss_norm = per_sample_lz.mean()

        raw_mse_per_sample = (z_adapted.float() - z_self_sg.float()).square().mean(dim=(1, 2))
        latent_mse_raw = raw_mse_per_sample.mean()

        # Diffusion loss through FROZEN UNet (conditioned ONLY on z_adapted, NO h_q!)
        if lambda_a != 0.0:
            per_sample_diff = self.teacher.diffusion_loss_from_z_self(
                z_adapted,
                actions_10d,
                noise=noise,
                pertub_noise=pertub_noise,
                timesteps=timesteps,
            )
            diff_loss = per_sample_diff.mean()
        else:
            with torch.no_grad():
                per_sample_diff = self.teacher.diffusion_loss_from_z_self(
                    z_adapted,
                    actions_10d,
                    noise=noise,
                    pertub_noise=pertub_noise,
                    timesteps=timesteps,
                )
            diff_loss = per_sample_diff.mean()

        total_loss = lambda_a * diff_loss + lambda_z * latent_loss_norm

        with torch.no_grad():
            B = z_adapted.shape[0]
            cos_adapted = F.cosine_similarity(
                z_adapted.float().reshape(B, -1),
                z_self_sg.float().reshape(B, -1),
                dim=-1,
            ).mean()
            z_cross_rms = torch.sqrt(z_cross_det.float().square().mean() + 1e-12)
            delta_sq = delta_z.float().square().mean(dim=(1, 2))
            delta_rms = torch.sqrt(delta_sq.mean() + 1e-12)

            if is_self is not None:
                is_self_b = is_self.to(device=z_adapted.device, dtype=torch.bool)
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
                cross_lz = (per_sample_lz * c_m).sum() / n_c.clamp_min(1.0)
                self_lz = (per_sample_lz * s_m).sum() / n_s.clamp_min(1.0)
                cross_diff = (per_sample_diff.detach() * c_m).sum() / n_c.clamp_min(1.0)
                self_diff = (per_sample_diff.detach() * s_m).sum() / n_s.clamp_min(1.0)
            else:
                n_s = torch.tensor(0.0, device=z_adapted.device)
                n_c = torch.tensor(float(B), device=z_adapted.device)
                self_delta_rms = torch.tensor(0.0, device=z_adapted.device)
                cross_delta_rms = delta_rms
                cross_lz = latent_loss_norm.detach()
                self_lz = torch.tensor(0.0, device=z_adapted.device)
                cross_diff = diff_loss.detach()
                self_diff = torch.tensor(0.0, device=z_adapted.device)

        return {
            "loss": total_loss,
            "diff_loss": diff_loss.detach(),
            "latent_loss_norm": latent_loss_norm.detach(),
            "latent_mse_raw": latent_mse_raw.detach(),
            "cos_adapted_self": cos_adapted,
            "delta_rms": delta_rms,
            "cross_delta_rms": cross_delta_rms,
            "self_delta_rms": self_delta_rms,
            "z_cross_rms": z_cross_rms,
            "cross_lz": cross_lz,
            "self_lz": self_lz,
            "cross_diff": cross_diff,
            "self_diff": self_diff,
            "n_self": n_s,
            "n_cross": n_c,
        }

    def forward(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        z_self: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: Optional[torch.Tensor] = None,
        *,
        lambda_z: float = 0.5,
        lambda_a: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """DDP-friendly forward: computes `z_cross` online from frozen `demo_memory`
        (or uses `z_self` directly on Self-pair rows), applies ResidualTransformer, and
        computes total Stage 2 loss."""
        z_cross, h_q = self.read_latent_from_memory(
            demo_memory, demo_pad_mask, query_rgb_feats
        )
        # On 2.5% Self-pair rows, `z_cross` computed from the same demo_memory matches `z_self`
        # (use `z_self` directly if `is_self` is True to avoid any fp16/bf16 rounding noise).
        if is_self is not None:
            is_self_b = is_self.to(device=z_cross.device, dtype=torch.bool)
            if is_self_b.any():
                z_cross = torch.where(
                    is_self_b[:, None, None],
                    z_self.to(dtype=z_cross.dtype),
                    z_cross,
                )
        return self.compute_loss_from_latents(
            z_cross,
            h_q,
            z_self,
            actions_10d,
            is_self=is_self,
            lambda_z=lambda_z,
            lambda_a=lambda_a,
        )

    @torch.inference_mode()
    def predict_action_from_memory(
        self,
        demo_memory: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        apply_delta: bool = True,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        z_cross, h_q = self.read_latent_from_memory(
            demo_memory, demo_pad_mask, query_rgb_feats
        )
        if apply_delta:
            z_cond, delta_z = self.adapt_latent(z_cross, h_q)
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
        }

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """AdamW parameter groups STRICTLY over `self.residual_transformer` only."""
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        no_decay_keywords = ("bias", "norm", "pos_emb")
        for name, param in self.residual_transformer.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Double-check zero teacher parameters are trainable
        teacher_trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        if teacher_trainable != 0:
            raise RuntimeError(f"Frozen teacher has {teacher_trainable} trainable params!")
        trainable = [p for p in self.parameters() if p.requires_grad]
        if len(decay_params) + len(no_decay_params) != len(trainable):
            raise RuntimeError("Optimizer parameter grouping mismatch in Stage 2")

        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
        ]

    def parameter_summary(self, visual_encoder_params: int = 171_598_848) -> Dict[str, Any]:
        teacher_total = sum(p.numel() for p in self.teacher.parameters())
        teacher_trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        res_total = sum(p.numel() for p in self.residual_transformer.parameters())
        res_trainable = sum(
            p.numel() for p in self.residual_transformer.parameters() if p.requires_grad
        )
        return {
            "visual_encoder_frozen": visual_encoder_params,
            "stage1_teacher_frozen": teacher_total,
            "stage1_teacher_trainable": teacher_trainable,
            "residual_transformer_total": res_total,
            "residual_transformer_trainable": res_trainable,
            "total_trainable": res_trainable,
            "full_system_total_with_frozen_clip": visual_encoder_params + teacher_total + res_total,
            "latent_scale_s_z": float(self.latent_scale_s_z.item()),
        }


def load_stage2_policy(
    stage2_ckpt: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E6Stage2Policy, Dict[str, Any]]:
    """Load frozen Stage 1 Teacher + trained Stage 2 ResidualTransformer checkpoint."""
    payload = torch.load(Path(stage2_ckpt), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    s1_path = Path(stage1_ckpt) if stage1_ckpt is not None else Path(cfg["stage1_ckpt"])
    teacher = load_frozen_stage1_teacher(s1_path, device="cpu")
    policy = E6Stage2Policy(
        teacher,
        latent_scale_s_z=float(payload.get("latent_scale_s_z", cfg.get("latent_scale_s_z", 1.0))),
        residual_layers=int(cfg.get("residual_layers", 3)),
        residual_heads=int(cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
    )
    policy.residual_transformer.load_state_dict(payload["residual_transformer"], strict=True)
    policy.eval()
    return policy.to(device), payload
