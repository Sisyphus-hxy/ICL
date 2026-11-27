"""E9 Stage 1: Local Behavior Encoder + Diffusion Action Decoder (CLIP, RGB-only).

Architecture:
    b_t = LocalBehaviorEncoder(V_{t:t+16}, clip_pad_mask)    # (B, 4, 768)
    a_hat_{t:t+16} ~ ConditionalUnet1D(global_cond=b_t)      # global_cond_dim = 4 * 768 = 3072

Key invariants:
  - Frozen CLIP ViT-B/16 backbone (offline dense cache / online BPPVisualEncoder).
  - Dual-camera input preserved: all 16 x 2 = 32 spatiotemporal tokens with learnable
    temporal positional embeddings and camera embeddings, compressed via 4 learnable
    behavior tokens inside a Pre-LN Transformer Encoder with strict tail-window padding mask.
  - UNet (`ConditionalUnet1D`) architecture is 100% identical to E6 Stage 1, conditioned
    ONLY on `b_t` (`global_cond_dim = 3072`).
  - Both `LocalBehaviorEncoder` and `ConditionalUnet1D` are trained from scratch with
    pure Action Diffusion Loss `L = L_diff`.
  - Behavior Memory `(h_mem, b_mem)` shape `(M, 4, 768)`, latent shape `(B, 4, 768)`,
    and Action Decoder interface (`diffusion_loss_from_z_self`, `sample_from_z_self`,
    `build_global_cond`, `action_normalizer`, `noise_scheduler`) remain 100% compatible
    with E7-Direct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import ActionNormalizer, action_10d_to_7d, extract_exec_actions
from e5.bpp_compat import import_bpp_components
from e5.diffusion import DDIMScheduler
from e5.model import E5Policy
from e7.model import compute_behavior_anchor_steps

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    BEHAVIOR_STRIDE,
    EMBED_DIM,
    EXEC_HORIZON,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_CAMERAS,
    NUM_HEADS,
    Z_SELF_COND_DIM,
)

_BPP = import_bpp_components()
ConditionalUnet1D = _BPP["ConditionalUnet1D"]
bpp_init_weights = _BPP["init_weights"]


class LocalBehaviorEncoder(nn.Module):
    """Transformer encoder with temporal + camera positional embeddings and 4 learnable
    tokens compressing dual-camera local clip `V_{t:t+16}` `(B, L, 2, 768)` into a
    `(B, 4, 768)` behavior latent `b_t`.

    Padding invariance:
      - `clip_pad_mask` `(B, L)` marks out-of-bounds tail frames as `True`.
      - Padded frames are zeroed before projection AND masked out via
        `src_key_padding_mask` in every Transformer layer so padded positions have
        mathematically zero effect on any valid token or the 4 learnable behavior tokens.
    """

    def __init__(
        self,
        *,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 6,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = 4 * EMBED_DIM,
        max_window_len: int = LOCAL_WINDOW_LEN,
        num_cameras: int = NUM_CAMERAS,
        num_behavior_tokens: int = NUM_BEHAVIOR_TOKENS,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_behavior_tokens < 1:
            raise ValueError(
                f"num_behavior_tokens must be >= 1, got {num_behavior_tokens}"
            )
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.max_window_len = int(max_window_len)
        self.num_cameras = int(num_cameras)
        self.num_behavior_tokens = int(num_behavior_tokens)

        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.temporal_pos_emb = nn.Parameter(
            torch.zeros(1, self.max_window_len, 1, self.embed_dim)
        )
        self.camera_emb = nn.Parameter(
            torch.zeros(1, 1, self.num_cameras, self.embed_dim)
        )
        self.behavior_tokens = nn.Parameter(
            torch.zeros(1, self.num_behavior_tokens, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self) -> None:
        self.in_norm.apply(bpp_init_weights)
        self.transformer.apply(bpp_init_weights)
        self.out_norm.apply(bpp_init_weights)
        nn.init.normal_(self.temporal_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.camera_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.behavior_tokens, mean=0.0, std=0.02)

    def forward(
        self,
        clip_rgb_feats: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            clip_rgb_feats: `(B, L, 2, D)` dual-camera frozen CLIP features for `V_{t:t+L}` (`L <= max_window_len`).
            clip_pad_mask: optional `(B, L)` bool mask where `True` = padded tail frame.
        Returns:
            b_t: `(B, 4, D)` compressed behavior latent.
        """
        if clip_rgb_feats.ndim != 4 or clip_rgb_feats.shape[2:] != (
            self.num_cameras,
            self.embed_dim,
        ):
            raise ValueError(
                f"Expected clip_rgb_feats (B, L, {self.num_cameras}, {self.embed_dim}), "
                f"got {tuple(clip_rgb_feats.shape)}"
            )
        B, L, C, D = clip_rgb_feats.shape
        if L < 1 or L > self.max_window_len:
            raise ValueError(
                f"Clip length L={L} out of valid range [1, {self.max_window_len}]"
            )

        x = self.in_norm(clip_rgb_feats)
        x = x + self.temporal_pos_emb[:, :L, :, :] + self.camera_emb

        full_pad_mask: Optional[torch.Tensor] = None
        if clip_pad_mask is not None:
            if clip_pad_mask.shape != (B, L):
                raise ValueError(
                    f"Expected clip_pad_mask ({B}, {L}), got {tuple(clip_pad_mask.shape)}"
                )
            pad_b = clip_pad_mask.to(device=x.device, dtype=torch.bool)
            if pad_b.all(dim=-1).any():
                raise ValueError("clip_pad_mask contains a clip with zero valid frames")
            x = x.masked_fill(pad_b[:, :, None, None], 0.0)
            video_pad_mask = pad_b.unsqueeze(-1).expand(B, L, C).reshape(B, L * C)
            lat_pad_mask = torch.zeros(
                (B, self.num_behavior_tokens), device=x.device, dtype=torch.bool
            )
            full_pad_mask = torch.cat([lat_pad_mask, video_pad_mask], dim=1)

        video_tokens = x.reshape(B, L * C, D)
        lat_tokens = self.behavior_tokens.expand(B, self.num_behavior_tokens, D).to(
            dtype=video_tokens.dtype
        )
        seq = torch.cat([lat_tokens, video_tokens], dim=1)  # (B, 4 + 2*L, D)

        encoded = self.transformer(seq, src_key_padding_mask=full_pad_mask)
        b_t = self.out_norm(encoded[:, : self.num_behavior_tokens, :])
        return b_t


class E9Stage1LocalPolicy(nn.Module):
    """E9 Stage 1 Local Behavior Encoder + Diffusion UNet policy (trained from scratch).

    Compatible with E7-Direct Teacher interface (`action_decoder`, `action_normalizer`,
    `noise_scheduler`, `global_cond_dim`, `build_global_cond`, `diffusion_loss_from_z_self`,
    `sample_from_z_self`, plus `extract_behavior_memory_from_demo_features`).
    """

    def __init__(
        self,
        *,
        embed_dim: int = EMBED_DIM,
        action_dim: int = ACTION_DIM_10D,
        action_horizon: int = ACTION_HORIZON,
        exec_horizon: int = EXEC_HORIZON,
        window_len: int = LOCAL_WINDOW_LEN,
        num_behavior_tokens: int = NUM_BEHAVIOR_TOKENS,
        encoder_layers: int = 6,
        encoder_heads: int = NUM_HEADS,
        encoder_ffn_dim: int = 4 * EMBED_DIM,
        dropout: float = 0.1,
        num_train_timesteps: int = 50,
        num_inference_steps: int = 16,
        input_pertub: float = 0.1,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        action_normalizer: Optional[ActionNormalizer] = None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.exec_horizon = int(exec_horizon)
        self.window_len = int(window_len)
        self.num_behavior_tokens = int(num_behavior_tokens)
        self.num_inference_steps = int(num_inference_steps)
        self.input_pertub = float(input_pertub)
        self.global_cond_dim = self.num_behavior_tokens * self.embed_dim  # 4 * 768 = 3072

        # 1. Local Behavior Encoder (from scratch)
        self.local_encoder = LocalBehaviorEncoder(
            embed_dim=self.embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ffn_dim=encoder_ffn_dim,
            max_window_len=self.window_len,
            num_cameras=NUM_CAMERAS,
            num_behavior_tokens=self.num_behavior_tokens,
            dropout=dropout,
        )

        # 2. Exact E6 Stage 1 / BPP ConditionalUnet1D conditioned ONLY on b_t (from scratch)
        self.action_decoder = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # 3. Exact BPP DDIM scheduler & ActionNormalizer
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
            timestep_spacing="leading",
        )
        self.action_normalizer = (
            action_normalizer if action_normalizer is not None else ActionNormalizer()
        )

    def encode_local_behavior(
        self,
        clip_rgb_feats: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode local video clip `V_{t:t+16}` -> `b_t` of shape `(B, 4, 768)`."""
        return self.local_encoder(clip_rgb_feats, clip_pad_mask=clip_pad_mask)

    @torch.no_grad()
    def extract_behavior_memory_from_demo_features(
        self,
        demo_all_frames_rgb: torch.Tensor,
        *,
        stride: int = BEHAVIOR_STRIDE,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Construct E7-Direct-compatible Behavior Memory `(h_mem, b_mem)` for a single
        already-provided Demo video of length `T` (`demo_all_frames_rgb`: `(T, 2, 768)`).

        Future frames are used STRICTLY inside the provided Demo video to encode each
        Demo segment `V_{t_k:t_k+16}` -> `b_k`; Query future frames are NEVER read.

        Returns:
            `h_mem`: `(M, 4, 768)` visual keys `[max(0, t_k-1):av, max(0, t_k-1):eh, t_k:av, t_k:eh]`
            `b_mem`: `(M, 4, 768)` local behavior latents `E_phi(V_{t_k:t_k+16})`
            `anchors`: `(M,)` int32 stride-16 anchor steps `0, 16, 32, ... < T`
        """
        if demo_all_frames_rgb.ndim != 3 or demo_all_frames_rgb.shape[1:] != (
            NUM_CAMERAS,
            self.embed_dim,
        ):
            raise ValueError(
                f"Expected demo_all_frames_rgb (T, {NUM_CAMERAS}, {self.embed_dim}), "
                f"got {tuple(demo_all_frames_rgb.shape)}"
            )
        T = int(demo_all_frames_rgb.shape[0])
        dev = demo_all_frames_rgb.device
        anchors = compute_behavior_anchor_steps(T, stride=stride)
        prev_anchors = np.maximum(anchors - 1, 0)

        # 1. Visual keys h_mem (M, 4, 768) matching E7-Direct exactly
        p0_t = torch.as_tensor(prev_anchors, dtype=torch.long, device=dev)
        p1_t = torch.as_tensor(anchors, dtype=torch.long, device=dev)
        anchor_query_rgb = torch.stack(
            [demo_all_frames_rgb[p0_t], demo_all_frames_rgb[p1_t]], dim=1
        )  # (M, 2, 2, 768)
        h_mem = E5Policy.format_query_tokens(anchor_query_rgb)  # (M, 4, 768)

        # 2. Local 16-frame clips V_{t_k:t_k+16} + tail padding mask -> b_mem (M, 4, 768)
        step_offsets = torch.arange(self.window_len, dtype=torch.long, device=dev)[None, :]
        raw_grid = p1_t[:, None] + step_offsets  # (M, 16)
        clip_pad_mask = raw_grid >= T
        clamped_grid = raw_grid.clamp_max(T - 1)
        clip_rgb = demo_all_frames_rgb[clamped_grid]  # (M, 16, 2, 768)
        clip_rgb = clip_rgb.masked_fill(clip_pad_mask[:, :, None, None], 0.0)
        b_mem = self.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad_mask)
        return h_mem, b_mem, anchors

    def build_global_cond(self, z_self: torch.Tensor) -> torch.Tensor:
        """Flatten behavior latent `(B, 4, 768)` -> `(B, 3072)` (E7-Direct compatible)."""
        if z_self.ndim != 3 or z_self.shape[1:] != (
            self.num_behavior_tokens,
            self.embed_dim,
        ):
            raise ValueError(
                f"Expected behavior latent (B, {self.num_behavior_tokens}, {self.embed_dim}), "
                f"got {tuple(z_self.shape)}"
            )
        return z_self.reshape(z_self.shape[0], self.global_cond_dim)

    def diffusion_loss_from_z_self(
        self,
        z_self: torch.Tensor,
        actions_10d: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample DDPM-IP epsilon MSE loss of shape `(B,)` conditioned ONLY on `b_t`."""
        global_cond = self.build_global_cond(z_self)
        nactions = self.action_normalizer.normalize(actions_10d)
        B = nactions.shape[0]
        if noise is None:
            noise = torch.randn(
                nactions.shape, device=nactions.device, dtype=torch.float32
            )
        if pertub_noise is None:
            pertub_noise = (
                torch.randn(nactions.shape, device=nactions.device, dtype=torch.float32)
                if self.input_pertub > 0.0
                else torch.zeros_like(noise)
            )
        noise_new = noise + self.input_pertub * pertub_noise
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=nactions.device,
                dtype=torch.long,
            )
        noisy_actions = self.noise_scheduler.add_noise(
            nactions.float(), noise_new, timesteps
        )
        pred = self.action_decoder(
            noisy_actions.to(dtype=global_cond.dtype),
            timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        target = (
            noise
            if self.noise_scheduler.config.prediction_type == "epsilon"
            else nactions
        )
        return F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2))

    # Alias for clarity alongside E7-Direct's `diffusion_loss_from_z_self`
    diffusion_loss_from_behavior_latent = diffusion_loss_from_z_self

    def compute_loss(
        self,
        clip_rgb_feats: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
        actions_10d: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pure Action Diffusion Loss `L = L_diff` with zero auxiliary terms."""
        b_t = self.encode_local_behavior(clip_rgb_feats, clip_pad_mask=clip_pad_mask)
        per_sample_diff = self.diffusion_loss_from_z_self(
            b_t,
            actions_10d,
            noise=noise,
            pertub_noise=pertub_noise,
            timesteps=timesteps,
        )
        diff_loss = per_sample_diff.mean()
        b_rms = torch.sqrt(b_t.detach().float().square().mean() + 1e-12)
        return {
            "loss": diff_loss,
            "diff_loss": diff_loss.detach(),
            "b_rms": b_rms,
        }

    def forward(
        self,
        clip_rgb_feats: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
        actions_10d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(clip_rgb_feats, clip_pad_mask, actions_10d)

    @torch.inference_mode()
    def sample_from_z_self(
        self,
        z_self: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """16-step DDIM denoising conditioned ONLY on behavior latent `b_t` -> `(B, 16, 10)`."""
        global_cond = self.build_global_cond(z_self)
        B = global_cond.shape[0]
        device = global_cond.device
        if init_noise is not None:
            trajectory = init_noise.to(device=device, dtype=torch.float32).clone()
        else:
            trajectory = torch.randn(
                (B, self.action_horizon, self.action_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)
        for t in self.noise_scheduler.timesteps:
            model_output = self.action_decoder(
                trajectory.to(dtype=global_cond.dtype),
                t,
                local_cond=None,
                global_cond=global_cond,
            ).float()
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, generator=generator
            ).prev_sample
        return self.action_normalizer.unnormalize(trajectory)

    sample_from_behavior_latent = sample_from_z_self

    @torch.inference_mode()
    def predict_action(
        self,
        clip_rgb_feats: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        b_t = self.encode_local_behavior(clip_rgb_feats, clip_pad_mask=clip_pad_mask)
        action_10d = self.sample_from_z_self(
            b_t, generator=generator, init_noise=init_noise
        )
        action_7d = action_10d_to_7d(action_10d)
        exec_action_7d = extract_exec_actions(
            action_10d, exec_horizon=self.exec_horizon
        )
        return {
            "action_10d": action_10d,
            "action_7d": action_7d,
            "exec_action_7d": exec_action_7d,
            "b_t": b_t,
            "z_self": b_t,  # E6/E7-compatible key alias
        }

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Exact BPP / E6 Stage 1 AdamW decay / no-decay parameter grouping."""
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        no_decay_keywords = (
            "bias",
            "norm",
            "pos_emb",
            "camera_emb",
            "behavior_tokens",
            "latent",
        )
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        trainable = [p for p in self.parameters() if p.requires_grad]
        if len(decay_params) + len(no_decay_params) != len(trainable):
            raise RuntimeError("Optimizer parameter grouping mismatch")

        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
        ]

    def parameter_summary(self, visual_encoder_params: int = 171_598_848) -> Dict[str, Any]:
        def _count(mod: Optional[nn.Module]) -> Dict[str, int]:
            if mod is None:
                return {"total": 0, "trainable": 0}
            return {
                "total": sum(p.numel() for p in mod.parameters()),
                "trainable": sum(
                    p.numel() for p in mod.parameters() if p.requires_grad
                ),
            }

        modules = {
            "visual_encoder_frozen": {"total": visual_encoder_params, "trainable": 0},
            "local_encoder": _count(self.local_encoder),
            "action_decoder": _count(self.action_decoder),
        }
        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        policy_total = sum(p.numel() for p in self.parameters())
        return {
            "modules": modules,
            "policy_total": policy_total,
            "policy_trainable": trainable_total,
            "global_cond_dim": self.global_cond_dim,
            "full_system_total_with_frozen_clip": policy_total + visual_encoder_params,
        }


def load_frozen_e9_stage1_teacher(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> E9Stage1LocalPolicy:
    """Load a saved E9 Stage 1 Local checkpoint and freeze all weights for E7-Direct / D1 use."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    policy = E9Stage1LocalPolicy(
        embed_dim=int(cfg.get("embed_dim", EMBED_DIM)),
        window_len=int(cfg.get("window_len", LOCAL_WINDOW_LEN)),
        num_behavior_tokens=int(cfg.get("num_behavior_tokens", NUM_BEHAVIOR_TOKENS)),
        encoder_layers=int(cfg.get("encoder_layers", 6)),
        encoder_heads=int(cfg.get("encoder_heads", NUM_HEADS)),
        encoder_ffn_dim=int(cfg.get("encoder_ffn_dim", 4 * EMBED_DIM)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    policy.load_state_dict(payload["model"], strict=True)
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.eval()
    return policy.to(device)
