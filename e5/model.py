"""E5 (Local Reference + Latent Residual Learning) and matched BPP RGB-only models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.attention_pool import AttentionPoolLatent

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    BPP_CKPT,
    EMBED_DIM,
    EXEC_HORIZON,
    MAX_DEMO_LEN,
    NUM_ATTENTION_SINKS,
    NUM_HEADS,
    QUERY_HISTORY,
)
from .actions import ActionNormalizer, action_10d_to_7d, extract_exec_actions
from .bpp_compat import (
    BPP_QUERY_RGB_POS_INDICES,
    import_bpp_components,
    load_bpp_pretrained_weights,
)
from .diffusion import DDIMScheduler

_BPP = import_bpp_components()
ConditionalUnet1D = _BPP["ConditionalUnet1D"]
TransformerDecoderLayerWithAttn = _BPP["TransformerDecoderLayerWithAttn"]
TransformerDecoderWithAttn = _BPP["TransformerDecoderWithAttn"]
bpp_init_weights = _BPP["init_weights"]

NUM_QUERY_TOKENS = QUERY_HISTORY * 2  # 2 steps x 2 cameras = 4 RGB tokens


class RGBPromptPooler(nn.Module):
    """BPP AttentionPoolLatent merging dual-camera RGB tokens per 1Hz Demo step -> 1 token.

    Retrained from scratch for RGB-only prompting (BPP architecture: AttentionPoolLatent
    with in_features=768, num_heads=8, mlp_ratio=4.0, pool_type='token').
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_attention_pool = AttentionPoolLatent(
            in_features=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pool_type="token",
        )

    def forward(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            demo_rgb_feats: (B, P, 2, D) dual-camera CLIP CLS features per 1Hz step.
            demo_pad_mask: (B, P) bool mask where True indicates padded positions.
        Returns:
            M_d: (B, P, D) pooled demonstration memory tokens.
        """
        if demo_rgb_feats.ndim != 4 or demo_rgb_feats.shape[2:] != (2, self.embed_dim):
            raise ValueError(
                f"Expected demo_rgb_feats (B, P, 2, {self.embed_dim}), got {tuple(demo_rgb_feats.shape)}"
            )
        B, P, C, D = demo_rgb_feats.shape
        pooled = self.prompt_attention_pool(demo_rgb_feats.reshape(B * P, C, D))
        pooled = pooled.reshape(B, P, D)
        if demo_pad_mask is not None:
            pooled = pooled.masked_fill(demo_pad_mask.unsqueeze(-1), 0.0)
        return pooled


class ReferenceReader(nn.Module):
    """BPP 6-layer Transformer Decoder (hidden=768, 8 heads, FFN=3072, ~56.8M)
    implementing z_ref = R(M_d, h_q) with BPP's prompt positional embeddings and
    learnable attention-sink tokens."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 6,
        num_heads: int = NUM_HEADS,
        max_prompt_tokens: int = MAX_DEMO_LEN,
        num_query_tokens: int = NUM_QUERY_TOKENS,
        dropout: float = 0.1,
        use_attention_sink: bool = True,
        num_attention_sink_tokens: int = NUM_ATTENTION_SINKS,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_prompt_tokens = max_prompt_tokens
        self.num_query_tokens = num_query_tokens
        self.use_attention_sink = use_attention_sink
        self.num_attention_sink_tokens = num_attention_sink_tokens if use_attention_sink else 0

        decoder_layer = TransformerDecoderLayerWithAttn(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_with_obs_decoder = TransformerDecoderWithAttn(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        self.prompt_pos_emb = nn.Parameter(torch.randn(1, max_prompt_tokens, embed_dim))
        self.prompt_current_obs_pos_emb = nn.Parameter(
            torch.randn(1, num_query_tokens, embed_dim)
        )
        if self.use_attention_sink:
            self.attention_sink_tokens = nn.Parameter(
                torch.zeros(1, self.num_attention_sink_tokens, embed_dim)
            )
        else:
            self.attention_sink_tokens = None

        self.init_weights()

    def init_weights(self) -> None:
        self.prompt_with_obs_decoder.apply(bpp_init_weights)
        nn.init.normal_(self.prompt_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.prompt_current_obs_pos_emb, mean=0.0, std=0.02)
        if self.attention_sink_tokens is not None:
            nn.init.normal_(self.attention_sink_tokens, mean=0.0, std=0.02)

    def load_from_bpp_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect compatibility and load pretrained BPP Reference Reader weights."""
        report: Dict[str, Any] = {
            "decoder_loaded": False,
            "prompt_pos_emb_loaded": False,
            "attention_sink_loaded": False,
            "current_obs_pos_emb_loaded": False,
            "current_obs_pos_emb_mode": "reinitialized",
        }

        # 1. 6-layer TransformerDecoder weights (exact key + shape compatibility check)
        target_dec_sd = self.prompt_with_obs_decoder.state_dict()
        src_dec_sd = bundle["reader_decoder"]
        dec_compatible = (
            set(target_dec_sd.keys()) == set(src_dec_sd.keys())
            and all(target_dec_sd[k].shape == src_dec_sd[k].shape for k in target_dec_sd)
        )
        if dec_compatible:
            self.prompt_with_obs_decoder.load_state_dict(src_dec_sd, strict=True)
            report["decoder_loaded"] = True
            report["decoder_tensors"] = len(src_dec_sd)
            report["decoder_params"] = sum(v.numel() for v in src_dec_sd.values())
        else:
            report["decoder_mismatch"] = True

        # 2. prompt_pos_emb (1, 50, 768)
        src_ppos = bundle["reader_prompt_pos_emb"]
        if src_ppos.shape == self.prompt_pos_emb.shape:
            with torch.no_grad():
                self.prompt_pos_emb.copy_(src_ppos)
            report["prompt_pos_emb_loaded"] = True

        # 3. attention_sink_tokens (1, 4, 768)
        src_sink = bundle["reader_attention_sink_tokens"]
        if (
            self.attention_sink_tokens is not None
            and src_sink.shape == self.attention_sink_tokens.shape
        ):
            with torch.no_grad():
                self.attention_sink_tokens.copy_(src_sink)
            report["attention_sink_loaded"] = True

        # 4. prompt_current_obs_pos_emb: slice RGB positions [0, 1, 5, 6] from BPP's (1, 10, 768)
        src_qpos_rgb = bundle["reader_current_obs_pos_emb_rgb"]
        if src_qpos_rgb.shape == self.prompt_current_obs_pos_emb.shape:
            with torch.no_grad():
                self.prompt_current_obs_pos_emb.copy_(src_qpos_rgb)
            report["current_obs_pos_emb_loaded"] = True
            report["current_obs_pos_emb_mode"] = (
                f"rgb_slice_{list(BPP_QUERY_RGB_POS_INDICES)}_from_10"
            )

        return report

    def forward(
        self,
        M_d: torch.Tensor,
        h_q: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            M_d: (B, P, D) pooled Demo memory tokens.
            h_q: (B, 4, D) Query RGB tokens [t-1:av, t-1:eh, t:av, t:eh].
            demo_pad_mask: (B, P) bool, True = padded / ignore.
        Returns:
            z_ref: (B, 4, D)
            cross_attn_weights: optional (L, B, [H], 4, P) with sinks stripped.
        """
        B, P, D = M_d.shape
        if P > self.max_prompt_tokens:
            raise ValueError(
                f"Demo length {P} exceeds max_prompt_tokens {self.max_prompt_tokens}"
            )
        if h_q.shape != (B, self.num_query_tokens, D):
            raise ValueError(
                f"Expected h_q ({B}, {self.num_query_tokens}, {D}), got {tuple(h_q.shape)}"
            )

        memory = M_d + self.prompt_pos_emb[:, :P, :]
        mask = demo_pad_mask

        if self.attention_sink_tokens is not None:
            sink = self.attention_sink_tokens.expand(B, self.num_attention_sink_tokens, D)
            memory = torch.cat([sink, memory], dim=1)
            if mask is not None:
                sink_mask = torch.zeros(
                    (B, self.num_attention_sink_tokens),
                    device=mask.device,
                    dtype=torch.bool,
                )
                mask = torch.cat([sink_mask, mask], dim=1)

        tgt = h_q + self.prompt_current_obs_pos_emb
        ret = self.prompt_with_obs_decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        if need_weights:
            z_ref, attn = ret
            if self.attention_sink_tokens is not None:
                attn = attn[..., self.num_attention_sink_tokens :]
            return z_ref, attn
        return ret, None


class ResidualTransformer(nn.Module):
    """3-layer Transformer Decoder (hidden=768, 8 heads, FFN=3072, ~28.4M + 0.6M proj)
    mapping (z_ref, h_q) -> delta_z of identical shape (B, 4, 768) with zero-initialized
    output projection so z_q = z_ref + delta_z == z_ref identically at initialization."""

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
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ref_in_norm = nn.LayerNorm(embed_dim)
        self.query_in_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer=layer, num_layers=num_layers)
        self.ref_pos_emb = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim))
        self.query_pos_emb = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim))
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.init_weights()

    def init_weights(self) -> None:
        self.decoder.apply(bpp_init_weights)
        for norm in (self.ref_in_norm, self.query_in_norm, self.out_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.normal_(self.ref_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.query_pos_emb, mean=0.0, std=0.02)
        # Zero-initialize output projection so delta_z == 0 at step 0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_ref: torch.Tensor,
        h_q: torch.Tensor,
        *,
        output_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if z_ref.shape != h_q.shape:
            raise ValueError(
                f"z_ref {tuple(z_ref.shape)} and h_q {tuple(h_q.shape)} must match"
            )
        h = self.decoder(
            tgt=self.ref_in_norm(z_ref) + self.ref_pos_emb,
            memory=self.query_in_norm(h_q) + self.query_pos_emb,
        )
        out = self.out_proj(self.out_norm(h))
        if output_scale is None:
            ref_scale = (
                z_ref.detach()
                .float()
                .square()
                .mean(dim=(-2, -1), keepdim=True)
                .clamp_min(1e-6)
                .sqrt()
            )
            # Scale dimensionless zero-init residual back to z_ref's natural RMS scale.
            return out * ref_scale.to(dtype=h.dtype)
        return out * float(output_scale)


class E5Policy(nn.Module):
    """E5 Local Reference + Latent Residual Learning policy (and matched BPP RGB-only
    baseline when `use_residual=False`).

    Equations:
        M_d   = PromptPooler(demo_rgb_feats)            # (B, P, 768)
        z_ref = ReferenceReader(M_d, h_q)               # (B, 4, 768)
        Δz    = ResidualTransformer(z_ref, h_q)         # (B, 4, 768), zero-init
        z_q   = z_ref + Δz                              # (B, 4, 768)
        A_q  ~ D_diffusion(z_q, h_q)                    # (B, 16, 10) -> exec 12x7D
    """

    def __init__(
        self,
        *,
        use_residual: bool = True,
        embed_dim: int = EMBED_DIM,
        action_dim: int = ACTION_DIM_10D,
        action_horizon: int = ACTION_HORIZON,
        exec_horizon: int = EXEC_HORIZON,
        num_train_timesteps: int = 50,
        num_inference_steps: int = 16,
        input_pertub: float = 0.1,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        dropout: float = 0.1,
        use_attention_sink: bool = True,
        num_attention_sink_tokens: int = NUM_ATTENTION_SINKS,
        action_normalizer: Optional[ActionNormalizer] = None,
        pretrained_bundle: Optional[Dict[str, Any]] = None,
        load_pretrained_reader: bool = True,
        ckpt_path: Path = BPP_CKPT,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        self.num_inference_steps = num_inference_steps
        self.input_pertub = input_pertub

        # 1. RGB-only Prompt Pooling (retrained from scratch)
        self.prompt_pooler = RGBPromptPooler(
            embed_dim=embed_dim, num_heads=NUM_HEADS, mlp_ratio=4.0
        )

        # 2. Reference Reader (BPP 6-layer Transformer, ~56.8M)
        self.reference_reader = ReferenceReader(
            embed_dim=embed_dim,
            num_layers=6,
            num_heads=NUM_HEADS,
            max_prompt_tokens=MAX_DEMO_LEN,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=dropout,
            use_attention_sink=use_attention_sink,
            num_attention_sink_tokens=num_attention_sink_tokens,
        )

        # 3. Action Decoder: official BPP ConditionalUnet1D (~151.08M)
        # Constructed before ResidualTransformer so E5 and matched BPP RGB-only baseline
        # receive bit-identical initialization for all shared modules under the same seed.
        # global_cond = flatten(cat([h_q (4x768), z_cond (4x768)], dim=1)) = 6144
        global_cond_dim = (NUM_QUERY_TOKENS * 2) * embed_dim
        self.action_decoder = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # 4. Residual Transformer (3-layer Transformer Decoder, ~28.95M, zero-init out_proj)
        self.residual_transformer: Optional[ResidualTransformer] = (
            ResidualTransformer(
                embed_dim=embed_dim,
                num_layers=3,
                num_heads=NUM_HEADS,
                ffn_dim=3072,
                num_query_tokens=NUM_QUERY_TOKENS,
                dropout=dropout,
            )
            if use_residual
            else None
        )

        # 5. BPP DDIM noise scheduler & Action Normalizer
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

        # Compatibility check & pretrained loading
        self.weight_compat_report: Dict[str, Any] = {
            "prompt_pooler": "retrained_from_scratch (RGB-only 2-cam vs BPP 6-modality)",
            "residual_transformer": (
                "initialized_from_scratch_zero_out_proj" if use_residual else "disabled (bpp_rgb_only)"
            ),
            "action_decoder": (
                f"reinitialized_from_scratch (global_cond_dim={global_cond_dim} vs BPP 15360)"
            ),
            "reference_reader": {"loaded": False},
        }
        if load_pretrained_reader:
            bundle = (
                pretrained_bundle
                if pretrained_bundle is not None
                else load_bpp_pretrained_weights(ckpt_path=ckpt_path)
            )
            self.weight_compat_report["reference_reader"] = (
                self.reference_reader.load_from_bpp_bundle(bundle)
            )
            self.weight_compat_report["bpp_unet_cond_encoder0_shape"] = list(
                bundle["bpp_unet_cond_encoder0_shape"]
            )
            self.weight_compat_report["e5_unet_cond_encoder0_shape"] = list(
                self.action_decoder.down_modules[0][0].cond_encoder[1].weight.shape
            )
            if action_normalizer is None:
                norm = bundle["action_normalizer"]
                self.action_normalizer.scale.copy_(norm["scale"])
                self.action_normalizer.offset.copy_(norm["offset"])

    @staticmethod
    def format_query_tokens(query_rgb_feats: torch.Tensor) -> torch.Tensor:
        """Convert query dual-camera history features (B, 2, 2, 768) -> h_q (B, 4, 768)
        in exact BPP ordering: [t-1:agentview, t-1:eye_in_hand, t:agentview, t:eye_in_hand]."""
        if query_rgb_feats.ndim == 3 and query_rgb_feats.shape[1:] == (
            NUM_QUERY_TOKENS,
            EMBED_DIM,
        ):
            return query_rgb_feats
        if query_rgb_feats.ndim != 4 or query_rgb_feats.shape[1:] != (
            QUERY_HISTORY,
            2,
            EMBED_DIM,
        ):
            raise ValueError(
                f"Expected query_rgb_feats (B, 2, 2, {EMBED_DIM}) or (B, 4, {EMBED_DIM}), "
                f"got {tuple(query_rgb_feats.shape)}"
            )
        return query_rgb_feats.reshape(query_rgb_feats.shape[0], NUM_QUERY_TOKENS, EMBED_DIM)

    def encode_conditioning(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute (h_q, z_ref, delta_z, z_q, cross_attn_weights)."""
        h_q = self.format_query_tokens(query_rgb_feats)
        M_d = self.prompt_pooler(demo_rgb_feats, demo_pad_mask=demo_pad_mask)
        z_ref, attn = self.reference_reader(
            M_d, h_q, demo_pad_mask=demo_pad_mask, need_weights=need_weights
        )
        if self.residual_transformer is not None:
            delta_z = self.residual_transformer(z_ref, h_q)
        else:
            delta_z = torch.zeros_like(z_ref)
        z_q = z_ref + delta_z
        return h_q, z_ref, delta_z, z_q, attn

    def _build_global_cond(self, h_q: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        # Matches BPP PairPromptObsEncoder: cat([receding_tokens, encoded_prompt_tokens], dim=1)
        return torch.cat([h_q, z_cond], dim=1).reshape(h_q.shape[0], -1)

    def diffusion_loss_from_cond(
        self,
        global_cond: torch.Tensor,
        actions_10d: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample DDPM-IP epsilon MSE loss of shape (B,)."""
        nactions = self.action_normalizer.normalize(actions_10d)
        B = nactions.shape[0]
        if noise is None:
            noise = torch.randn(nactions.shape, device=nactions.device, dtype=torch.float32)
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
        noisy_actions = self.noise_scheduler.add_noise(nactions.float(), noise_new, timesteps)
        pred = self.action_decoder(
            noisy_actions.to(dtype=global_cond.dtype),
            timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else nactions
        return F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2))

    def compute_loss(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: torch.Tensor,
        *,
        self_res_weight: float = 1.0,
        cross_res_weight: float = 0.05,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Direct Joint Training loss:
        - Self-pair: diffusion loss uses z_ref directly + delta_z ≈ 0 constraint.
        - Cross-pair: diffusion loss uses z_ref + delta_z + moderate residual reg.
        - BPP RGB-only baseline (`use_residual=False`): pure diffusion loss on z_ref.
        """
        h_q, z_ref, delta_z, z_q, _ = self.encode_conditioning(
            demo_rgb_feats, demo_pad_mask, query_rgb_feats
        )
        is_self = is_self.to(device=h_q.device, dtype=torch.bool)
        if self.use_residual:
            z_used = torch.where(is_self[:, None, None], z_ref, z_q)
        else:
            z_used = z_ref

        global_cond = self._build_global_cond(h_q, z_used)
        per_sample_diff = self.diffusion_loss_from_cond(
            global_cond,
            actions_10d,
            noise=noise,
            pertub_noise=pertub_noise,
            timesteps=timesteps,
        )
        diff_loss = per_sample_diff.mean()

        z_ref_sq = z_ref.float().square().mean(dim=(1, 2))
        delta_sq = delta_z.float().square().mean(dim=(1, 2))
        rel_sq = delta_sq / z_ref_sq.detach().clamp_min(1e-6)

        self_mask = is_self.float()
        cross_mask = (~is_self).float()
        n_self = self_mask.sum()
        n_cross = cross_mask.sum()

        diff_self = (per_sample_diff * self_mask).sum() / n_self.clamp_min(1.0)
        diff_cross = (per_sample_diff * cross_mask).sum() / n_cross.clamp_min(1.0)
        self_res_loss = (rel_sq * self_mask).sum() / n_self.clamp_min(1.0)
        cross_res_loss = (rel_sq * cross_mask).sum() / n_cross.clamp_min(1.0)

        if self.use_residual:
            total_loss = (
                diff_loss
                + self_res_weight * self_res_loss
                + cross_res_weight * cross_res_loss
            )
        else:
            total_loss = diff_loss

        self_delta_rms = torch.sqrt(
            (delta_sq * self_mask).sum() / n_self.clamp_min(1.0) + 1e-12
        )
        cross_delta_rms = torch.sqrt(
            (delta_sq * cross_mask).sum() / n_cross.clamp_min(1.0) + 1e-12
        )
        z_ref_rms = torch.sqrt(z_ref_sq.mean() + 1e-12)

        return {
            "loss": total_loss,
            "diff_loss": diff_loss.detach(),
            "diff_self": diff_self.detach(),
            "diff_cross": diff_cross.detach(),
            "self_res_loss": self_res_loss.detach(),
            "cross_res_loss": cross_res_loss.detach(),
            "self_delta_rms": self_delta_rms.detach(),
            "cross_delta_rms": cross_delta_rms.detach(),
            "z_ref_rms": z_ref_rms.detach(),
            "n_self": n_self.detach(),
            "n_cross": n_cross.detach(),
        }

    def forward(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: torch.Tensor,
        *,
        self_res_weight: float = 1.0,
        cross_res_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(
            demo_rgb_feats,
            demo_pad_mask,
            query_rgb_feats,
            actions_10d,
            is_self,
            self_res_weight=self_res_weight,
            cross_res_weight=cross_res_weight,
        )

    @torch.inference_mode()
    def sample_from_global_cond(
        self,
        global_cond: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run 16-step DDIM denoising -> unnormalized 16-step 10D actions (B, 16, 10)."""
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

    @torch.inference_mode()
    def predict_action(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        use_residual: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Dict[str, Any]:
        """Predict 16-step 10D actions + 16-step & 12-step 7D OSC controls."""
        h_q, z_ref, delta_z, z_q, attn = self.encode_conditioning(
            demo_rgb_feats, demo_pad_mask, query_rgb_feats, need_weights=need_weights
        )
        apply_res = self.use_residual if use_residual is None else (use_residual and self.use_residual)
        z_cond = z_q if apply_res else z_ref
        global_cond = self._build_global_cond(h_q, z_cond)
        action_10d = self.sample_from_global_cond(
            global_cond, generator=generator, init_noise=init_noise
        )
        action_7d = action_10d_to_7d(action_10d)
        exec_action_7d = extract_exec_actions(action_10d, exec_horizon=self.exec_horizon)
        return {
            "action_10d": action_10d,
            "action_7d": action_7d,
            "exec_action_7d": exec_action_7d,
            "z_ref": z_ref,
            "delta_z": delta_z,
            "z_q": z_q,
            "cross_attn_weights": attn,
        }

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Follow BPP's AdamW decay/no-decay separation cleanly across all modules."""
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        no_decay_keywords = (
            "bias",
            "norm",
            "pos_emb",
            "attention_sink_tokens",
            "latent",
        )
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Verify every trainable parameter is accounted for exactly once.
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
                "trainable": sum(p.numel() for p in mod.parameters() if p.requires_grad),
            }

        modules = {
            "visual_encoder_frozen": {"total": visual_encoder_params, "trainable": 0},
            "prompt_pooler": _count(self.prompt_pooler),
            "reference_reader": _count(self.reference_reader),
            "residual_transformer": _count(self.residual_transformer),
            "action_decoder": _count(self.action_decoder),
        }
        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        policy_total = sum(p.numel() for p in self.parameters())
        return {
            "modules": modules,
            "policy_total": policy_total,
            "policy_trainable": trainable_total,
            "full_system_total_with_frozen_clip": policy_total + visual_encoder_params,
        }
