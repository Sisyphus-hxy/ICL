"""E6 Stage 1: Self-conditioned Behavior Latent Learning (DINOv3 Spatial 4x4, RGB-only).

Preserves the exact E6 Stage 1 architecture & research hypothesis:
    M_D    = DINOv3SpatialPromptPooler(F_DINO(D))       # (B, P, K_m, 768), preserves temporal & spatial structure
    h_q    = DINOv3SpatialQueryPooler(F_DINO(Q))        # (B, 4, 768), used strictly as attention Query
    z_self = SpatialBehaviorReferenceReader(M_D, h_q)   # (B, 4, 768), NO h_q residual skip, NO attention sinks
    A_q   ~ ConditionalUnet1D(global_cond=z_self)       # global_cond_dim = 4 * 768 = 3072 (ONLY z_self)

No ResidualTransformer, no Auxiliary Head, no Progress Labels, pure Diffusion Loss only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.attention_pool import AttentionPoolLatent

from dinov3 import NUM_SPATIAL_TOKENS
from e5.actions import ActionNormalizer, action_10d_to_7d, extract_exec_actions
from e5.bpp_compat import import_bpp_components
from e5.diffusion import DDIMScheduler

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    EMBED_DIM,
    EXEC_HORIZON,
    MAX_DEMO_LEN,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
    OUTPUT_ROOT,
    Z_SELF_COND_DIM,
)
from .stage1_model import QueryReadLayer

_BPP = import_bpp_components()
ConditionalUnet1D = _BPP["ConditionalUnet1D"]
TransformerDecoderLayerWithAttn = _BPP["TransformerDecoderLayerWithAttn"]
TransformerDecoderWithAttn = _BPP["TransformerDecoderWithAttn"]
bpp_init_weights = _BPP["init_weights"]

DEFAULT_CLIP_STAGE1_CKPT = OUTPUT_ROOT / "stage1" / "stage1_self_latent" / "best.pt"
DEFAULT_DEMO_TOKENS_PER_STEP = 4
DINOV3_FROZEN_BACKBONE_PARAMS = 85_660_416  # facebook/dinov3-vitb16-pretrain-lvd1689m


class DINOv3SpatialPromptPooler(nn.Module):
    """Per-timestep spatiotemporal PromptPooler for Frozen DINOv3 features.

    Pools dual-camera spatial tokens `(2, S, D)` at each 1Hz Demo timestep independently
    into `K_m` spatial memory tokens `(K_m, D)`, strictly preserving temporal ordering `P`
    while avoiding memory explosion (`P * K_m <= 200` tokens instead of `P * 32 = 928`).

    Includes:
    - Input `LayerNorm(D)` to stabilize DINOv3's smaller feature magnitude (std ~0.35 vs CLIP ~1.05).
    - Learnable camera embedding `(2, D)` and 2D spatial patch positional embedding `(S, D)`.
    - Per-timestep `AttentionPoolLatent` with `latent_len=K_m` (`pool_type=""` returns `(B*P, K_m, D)`).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        mlp_ratio: float = 4.0,
        num_cameras: int = 2,
        num_spatial_tokens: int = NUM_SPATIAL_TOKENS,
        tokens_per_step: int = DEFAULT_DEMO_TOKENS_PER_STEP,
    ):
        super().__init__()
        if tokens_per_step < 1:
            raise ValueError(f"tokens_per_step must be >= 1, got {tokens_per_step}")
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_spatial_tokens = num_spatial_tokens
        self.tokens_per_step = tokens_per_step

        self.in_norm = nn.LayerNorm(embed_dim)
        self.camera_pos_emb = nn.Parameter(torch.zeros(1, 1, num_cameras, 1, embed_dim))
        self.spatial_pos_emb = nn.Parameter(
            torch.zeros(1, 1, 1, num_spatial_tokens, embed_dim)
        )
        self.prompt_attention_pool = AttentionPoolLatent(
            in_features=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            latent_len=tokens_per_step,
            pool_type="",
        )
        nn.init.normal_(self.camera_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_pos_emb, mean=0.0, std=0.02)

    def forward(
        self,
        demo_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            demo_feats: `(B, P, 2, S, D)` for spatial mode (or `(B, P, 2, D)` for global S=1).
            demo_pad_mask: `(B, P)` bool mask where True = padded timestep.
        Returns:
            M_d: `(B, P, K_m, D)` pooled Demo memory tokens preserving temporal axis `P`.
        """
        if demo_feats.ndim == 4:
            demo_feats = demo_feats.unsqueeze(3)  # (B, P, 2, 1, D)
        if demo_feats.ndim != 5 or demo_feats.shape[2:] != (
            self.num_cameras,
            self.num_spatial_tokens,
            self.embed_dim,
        ):
            raise ValueError(
                f"Expected demo_feats (B, P, {self.num_cameras}, {self.num_spatial_tokens}, {self.embed_dim}), "
                f"got {tuple(demo_feats.shape)}"
            )

        B, P, C, S, D = demo_feats.shape
        x = self.in_norm(demo_feats)
        x = x + self.camera_pos_emb + self.spatial_pos_emb
        # Pool strictly within each timestep p -> (B*P, C*S, D) -> (B*P, K_m, D)
        pooled = self.prompt_attention_pool(x.reshape(B * P, C * S, D))
        pooled = pooled.reshape(B, P, self.tokens_per_step, D)
        if demo_pad_mask is not None:
            pooled = pooled.masked_fill(demo_pad_mask[:, :, None, None], 0.0)
        return pooled


class DINOv3SpatialQueryPooler(nn.Module):
    """Pools Query spatial tokens `(B, 2, 2, S, D)` -> 4 Query tokens `h_q` `(B, 4, D)`.

    Each `(step in {t-1, t}, camera in {agentview, eye_in_hand})` frame is summarized
    via learnable 2D spatial positional encoding + `AttentionPoolLatent` into 1 query token.
    The resulting `h_q` is used STRICTLY as the attention Query in `QueryReadLayer` (no skip).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        mlp_ratio: float = 4.0,
        num_cameras: int = 2,
        num_spatial_tokens: int = NUM_SPATIAL_TOKENS,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_spatial_tokens = num_spatial_tokens

        self.in_norm = nn.LayerNorm(embed_dim)
        self.camera_pos_emb = nn.Parameter(torch.zeros(1, 1, num_cameras, 1, embed_dim))
        self.spatial_pos_emb = nn.Parameter(
            torch.zeros(1, 1, 1, num_spatial_tokens, embed_dim)
        )
        self.query_attention_pool = AttentionPoolLatent(
            in_features=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            latent_len=1,
            pool_type="token",
        )
        nn.init.normal_(self.camera_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_pos_emb, mean=0.0, std=0.02)

    def forward(self, query_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feats: `(B, 2, 2, S, D)` (or `(B, 2, 2, D)` for global S=1).
        Returns:
            h_q: `(B, 4, D)` ordered as `[t-1:av, t-1:eh, t:av, t:eh]`.
        """
        if query_feats.ndim == 4:
            query_feats = query_feats.unsqueeze(3)
        if query_feats.ndim != 5 or query_feats.shape[1:] != (
            2,
            self.num_cameras,
            self.num_spatial_tokens,
            self.embed_dim,
        ):
            raise ValueError(
                f"Expected query_feats (B, 2, {self.num_cameras}, {self.num_spatial_tokens}, {self.embed_dim}), "
                f"got {tuple(query_feats.shape)}"
            )
        B, T, C, S, D = query_feats.shape
        x = self.in_norm(query_feats)
        x = x + self.camera_pos_emb + self.spatial_pos_emb
        if S == 1:
            # Global mode: pass through query_attention_pool (or identity after norm+emb)
            pooled = self.query_attention_pool(x.reshape(B * T * C, 1, D))
        else:
            pooled = self.query_attention_pool(x.reshape(B * T * C, S, D))
        return pooled.reshape(B, T * C, D)


class SpatialBehaviorReferenceReader(nn.Module):
    """6-layer ReferenceReader over spatiotemporal Demo Memory `M_D` `(B, P, K_m, D)`.

    - Layer 0 (`QueryReadLayer`): `h_q` serves strictly as attention Query with NO residual
      skip path into `z_self`.
    - Demo Memory receives factorized **temporal positional encoding** (`prompt_pos_emb`:
      `(1, MAX_DEMO_LEN, D)`) + **spatial slot positional encoding** (`memory_spatial_pos_emb`:
      `(1, 1, K_m, D)`) before flattening `(P, K_m) -> P * K_m`.
    - Outputs `z_self` of shape `(B, 4, 768)` and optional cross-attention weights
      `(L, B, [H], 4, P * K_m)`.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 6,
        num_heads: int = NUM_HEADS,
        max_prompt_tokens: int = MAX_DEMO_LEN,
        tokens_per_step: int = DEFAULT_DEMO_TOKENS_PER_STEP,
        num_query_tokens: int = NUM_QUERY_TOKENS,
        dropout: float = 0.1,
        use_attention_sink: bool = False,
        num_attention_sink_tokens: int = 0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_prompt_tokens = max_prompt_tokens
        self.tokens_per_step = tokens_per_step
        self.num_query_tokens = num_query_tokens
        self.use_attention_sink = use_attention_sink
        self.num_attention_sink_tokens = (
            num_attention_sink_tokens if use_attention_sink else 0
        )

        refine_layer = TransformerDecoderLayerWithAttn(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_with_obs_decoder = TransformerDecoderWithAttn(
            decoder_layer=refine_layer,
            num_layers=num_layers,
        )
        self.prompt_with_obs_decoder.layers[0] = QueryReadLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Factorized temporal + spatial positional embeddings on Demo Memory
        self.prompt_pos_emb = nn.Parameter(torch.randn(1, max_prompt_tokens, embed_dim))
        self.memory_spatial_pos_emb = nn.Parameter(
            torch.randn(1, 1, tokens_per_step, embed_dim)
        )
        self.prompt_current_obs_pos_emb = nn.Parameter(
            torch.randn(1, num_query_tokens, embed_dim)
        )
        if self.use_attention_sink and self.num_attention_sink_tokens > 0:
            self.attention_sink_tokens = nn.Parameter(
                torch.zeros(1, self.num_attention_sink_tokens, embed_dim)
            )
        else:
            self.attention_sink_tokens = None

        self.init_weights()

    def init_weights(self) -> None:
        self.prompt_with_obs_decoder.apply(bpp_init_weights)
        nn.init.normal_(self.prompt_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_spatial_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.prompt_current_obs_pos_emb, mean=0.0, std=0.02)
        if self.attention_sink_tokens is not None:
            nn.init.normal_(self.attention_sink_tokens, mean=0.0, std=0.02)

    @staticmethod
    def aggregate_attn_to_temporal(
        cross_attn_weights: torch.Tensor,
        num_timesteps: int,
        tokens_per_step: int,
    ) -> torch.Tensor:
        """Aggregate spatial cross-attention `(..., P * K_m)` -> temporal `(..., P)` by
        summing attention probability mass over the `K_m` spatial tokens at each timestep.

        Since softmax normalizes over all `P * K_m` memory keys, summing over `K_m` yields
        the exact marginal temporal probability distribution summing to 1.0 over `0..P-1`,
        directly comparable to CLIP Stage 1's `(..., P)` temporal attention.
        """
        expected_kv = num_timesteps * tokens_per_step
        if cross_attn_weights.shape[-1] != expected_kv:
            raise ValueError(
                f"Expected attn last dim {expected_kv} (= {num_timesteps} * {tokens_per_step}), "
                f"got {cross_attn_weights.shape[-1]}"
            )
        new_shape = cross_attn_weights.shape[:-1] + (num_timesteps, tokens_per_step)
        return cross_attn_weights.reshape(new_shape).sum(dim=-1)

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
            M_d: `(B, P, K_m, D)` pooled spatiotemporal Demo memory tokens.
            h_q: `(B, 4, D)` Query tokens.
            demo_pad_mask: `(B, P)` bool mask where True = padded timestep.
        Returns:
            z_self: `(B, 4, D)`
            cross_attn_weights: optional `(L, B, [H], 4, P * K_m)`
        """
        if M_d.ndim == 3:
            M_d = M_d.unsqueeze(2)
        B, P, K_m, D = M_d.shape
        if P > self.max_prompt_tokens:
            raise ValueError(
                f"Demo length {P} exceeds max_prompt_tokens {self.max_prompt_tokens}"
            )
        if K_m != self.tokens_per_step:
            raise ValueError(
                f"Expected M_d tokens_per_step={self.tokens_per_step}, got {K_m}"
            )
        if h_q.shape != (B, self.num_query_tokens, D):
            raise ValueError(
                f"Expected h_q ({B}, {self.num_query_tokens}, {D}), got {tuple(h_q.shape)}"
            )

        # Add factorized temporal + spatial positional encodings, then flatten (P, K_m) -> P*K_m
        memory = (
            M_d
            + self.prompt_pos_emb[:, :P, None, :]
            + self.memory_spatial_pos_emb
        ).reshape(B, P * K_m, D)

        mask: Optional[torch.Tensor] = None
        if demo_pad_mask is not None:
            mask = (
                demo_pad_mask[:, :, None]
                .expand(B, P, K_m)
                .reshape(B, P * K_m)
            )

        if self.attention_sink_tokens is not None:
            sink = self.attention_sink_tokens.expand(
                B, self.num_attention_sink_tokens, D
            )
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
            z_self, attn = ret
            if self.attention_sink_tokens is not None:
                attn = attn[..., self.num_attention_sink_tokens :]
            return z_self, attn
        return ret, None


class E6Stage1DINOv3Policy(nn.Module):
    """E6 Stage 1 Self-conditioned Behavior Latent Learning policy on Frozen DINOv3 features."""

    def __init__(
        self,
        *,
        feature_mode: str = "spatial",
        num_spatial_tokens: Optional[int] = None,
        demo_tokens_per_step: int = DEFAULT_DEMO_TOKENS_PER_STEP,
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
        use_attention_sink: bool = False,
        action_normalizer: Optional[ActionNormalizer] = None,
    ):
        super().__init__()
        if feature_mode not in ("spatial", "global"):
            raise ValueError(f"Unsupported feature_mode={feature_mode!r}")
        self.feature_mode = feature_mode
        if num_spatial_tokens is None:
            num_spatial_tokens = NUM_SPATIAL_TOKENS if feature_mode == "spatial" else 1
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.demo_tokens_per_step = int(demo_tokens_per_step)
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        self.num_inference_steps = num_inference_steps
        self.input_pertub = input_pertub
        self.global_cond_dim = NUM_QUERY_TOKENS * embed_dim  # 4 * 768 = 3072 (ONLY z_self)

        # 1. Per-timestep spatiotemporal Demo PromptPooler
        self.prompt_pooler = DINOv3SpatialPromptPooler(
            embed_dim=embed_dim,
            num_heads=NUM_HEADS,
            mlp_ratio=4.0,
            num_cameras=2,
            num_spatial_tokens=self.num_spatial_tokens,
            tokens_per_step=self.demo_tokens_per_step,
        )

        # 2. Spatial Query Pooler (forms 4 Query tokens strictly for ReferenceReader attention Q)
        self.query_pooler = DINOv3SpatialQueryPooler(
            embed_dim=embed_dim,
            num_heads=NUM_HEADS,
            mlp_ratio=4.0,
            num_cameras=2,
            num_spatial_tokens=self.num_spatial_tokens,
        )

        # 3. Spatiotemporal ReferenceReader (no h_q skip, no attention sinks by default)
        self.reference_reader = SpatialBehaviorReferenceReader(
            embed_dim=embed_dim,
            num_layers=6,
            num_heads=NUM_HEADS,
            max_prompt_tokens=MAX_DEMO_LEN,
            tokens_per_step=self.demo_tokens_per_step,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=dropout,
            use_attention_sink=use_attention_sink,
        )

        # 4. Official BPP ConditionalUnet1D conditioned ONLY on z_self (global_cond_dim=3072)
        self.action_decoder = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=self.global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # 5. Exact BPP/CLIP Stage 1 DDIM scheduler & ActionNormalizer
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
        self.weight_compat_report: Dict[str, Any] = {}

    def load_compatible_from_checkpoint(
        self,
        ckpt_path: Path | str = DEFAULT_CLIP_STAGE1_CKPT,
        *,
        init_mode: str = "action_decoder_only",
    ) -> Dict[str, Any]:
        """Explicitly inspect & load shape-compatible weights from a CLIP Stage 1 checkpoint.

        Never silently loads incompatible tensors. Supports:
        - `init_mode="action_decoder_only"` (default): loads only modality-agnostic
          `action_decoder` (+ `action_normalizer`) since CLIP and DINOv3 visual embeddings
          have ~3x scale shift and distinct representational bases; re-initializes visual
          modules (`prompt_pooler`, `query_pooler`, `reference_reader`).
        - `init_mode="all_compatible"`: loads every tensor whose key & shape match exactly,
          explicitly reporting every shape-incompatible / missing / unused tensor.
        - `init_mode="none"`: pure from-scratch initialization.
        """
        if init_mode not in ("action_decoder_only", "all_compatible", "none"):
            raise ValueError(
                f"Unsupported init_mode={init_mode!r}; choose 'action_decoder_only', 'all_compatible', or 'none'"
            )
        if init_mode == "none":
            report = {"mode": "from_scratch", "init_mode": "none"}
            self.weight_compat_report = report
            return report

        ckpt_path = Path(ckpt_path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd: Dict[str, torch.Tensor] = payload["model"]
        tgt_sd = self.state_dict()

        allowed_modules = (
            {"action_decoder", "action_normalizer"}
            if init_mode == "action_decoder_only"
            else None
        )

        to_load: Dict[str, torch.Tensor] = {}
        module_stats: Dict[str, Dict[str, Any]] = {}
        incompatible: List[Dict[str, Any]] = []
        skipped_by_policy: List[Dict[str, Any]] = []
        missing_in_src: List[str] = []

        for key, tgt_tensor in tgt_sd.items():
            mod = key.split(".", 1)[0]
            st = module_stats.setdefault(
                mod,
                {
                    "loaded_tensors": 0,
                    "loaded_params": 0,
                    "skipped_by_policy_tensors": 0,
                    "skipped_by_policy_params": 0,
                    "reinit_tensors": 0,
                    "reinit_params": 0,
                },
            )
            if allowed_modules is not None and mod not in allowed_modules:
                st["skipped_by_policy_tensors"] += 1
                st["skipped_by_policy_params"] += tgt_tensor.numel()
                st["reinit_tensors"] += 1
                st["reinit_params"] += tgt_tensor.numel()
                skipped_by_policy.append(
                    {
                        "key": key,
                        "target_shape": list(tgt_tensor.shape),
                        "present_in_source": key in src_sd,
                        "source_shape": list(src_sd[key].shape) if key in src_sd else None,
                        "reason": f"init_mode={init_mode} skips CLIP-space visual module '{mod}'",
                    }
                )
                continue

            if key in src_sd and src_sd[key].shape == tgt_tensor.shape:
                to_load[key] = src_sd[key]
                st["loaded_tensors"] += 1
                st["loaded_params"] += tgt_tensor.numel()
            else:
                st["reinit_tensors"] += 1
                st["reinit_params"] += tgt_tensor.numel()
                if key in src_sd:
                    incompatible.append(
                        {
                            "key": key,
                            "target_shape": list(tgt_tensor.shape),
                            "source_shape": list(src_sd[key].shape),
                            "params": tgt_tensor.numel(),
                        }
                    )
                else:
                    missing_in_src.append(key)

        unused_in_src = [
            {
                "key": k,
                "source_shape": list(v.shape),
                "params": v.numel(),
            }
            for k, v in src_sd.items()
            if k not in to_load
        ]

        # Strict check on keys we decided to load
        missing_keys, unexpected_keys = self.load_state_dict(to_load, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys during checkpoint loading: {unexpected_keys}")
        expected_missing = set(tgt_sd.keys()) - set(to_load.keys())
        if set(missing_keys) != expected_missing:
            raise RuntimeError("Mismatch between planned and actually loaded tensors")

        total_params = sum(v.numel() for v in tgt_sd.values())
        loaded_params = sum(v.numel() for v in to_load.values())
        reinit_params = total_params - loaded_params

        report = {
            "init_mode": init_mode,
            "source_checkpoint": str(ckpt_path),
            "source_stage": payload.get("stage"),
            "source_epoch": payload.get("epoch"),
            "source_global_step": payload.get("global_step"),
            "total_target_params": total_params,
            "loaded_compatible_params": loaded_params,
            "loaded_fraction": round(loaded_params / max(total_params, 1), 6),
            "reinit_params": reinit_params,
            "per_module": module_stats,
            "shape_incompatible_tensors": incompatible,
            "skipped_by_policy_count": len(skipped_by_policy),
            "missing_in_source": missing_in_src,
            "unused_source_tensors_count": len(unused_in_src),
        }
        self.weight_compat_report = report
        return report

    def encode_demo_memory(
        self,
        demo_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode raw DINOv3 Demo features -> spatiotemporal memory `M_D` `(B, P, K_m, 768)`."""
        return self.prompt_pooler(demo_feats, demo_pad_mask=demo_pad_mask)

    def encode_query_tokens(self, query_feats: torch.Tensor) -> torch.Tensor:
        """Pool Query DINOv3 features -> `h_q` `(B, 4, 768)` strictly for ReferenceReader Query."""
        return self.query_pooler(query_feats)

    def encode_z_self(
        self,
        demo_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_feats: torch.Tensor,
        *,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        aggregate_attn_temporal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Read behavior latent `z_self` `(B, 4, 768)` from Demo Memory using Query `h_q`.

        When `need_weights=True` and `aggregate_attn_temporal=True`, cross-attention weights
        are summed over the `K_m` spatial memory tokens per timestep -> `(L, B, [H], 4, P)`.
        """
        M_d = self.encode_demo_memory(demo_feats, demo_pad_mask=demo_pad_mask)
        h_q = self.encode_query_tokens(query_feats)
        z_self, attn = self.reference_reader(
            M_d,
            h_q,
            demo_pad_mask=demo_pad_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        if need_weights and attn is not None and aggregate_attn_temporal:
            P = M_d.shape[1]
            attn = self.reference_reader.aggregate_attn_to_temporal(
                attn, num_timesteps=P, tokens_per_step=self.demo_tokens_per_step
            )
        return z_self, attn

    def build_global_cond(self, z_self: torch.Tensor) -> torch.Tensor:
        """Flatten `z_self` `(B, 4, 768)` -> `(B, 3072)`. Strictly excludes `h_q`."""
        if z_self.ndim != 3 or z_self.shape[1:] != (NUM_QUERY_TOKENS, self.embed_dim):
            raise ValueError(
                f"Expected z_self (B, {NUM_QUERY_TOKENS}, {self.embed_dim}), got {tuple(z_self.shape)}"
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
        """Per-sample DDPM-IP epsilon MSE loss of shape `(B,)` conditioned ONLY on `z_self`."""
        global_cond = self.build_global_cond(z_self)
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

    def compute_loss(
        self,
        demo_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_feats: torch.Tensor,
        actions_10d: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pure Diffusion Loss conditioned strictly on `z_self`."""
        z_self, _ = self.encode_z_self(demo_feats, demo_pad_mask, query_feats)
        per_sample_diff = self.diffusion_loss_from_z_self(
            z_self,
            actions_10d,
            noise=noise,
            pertub_noise=pertub_noise,
            timesteps=timesteps,
        )
        diff_loss = per_sample_diff.mean()
        z_self_rms = torch.sqrt(z_self.detach().float().square().mean() + 1e-12)
        return {
            "loss": diff_loss,
            "diff_loss": diff_loss.detach(),
            "z_self_rms": z_self_rms,
        }

    def forward(
        self,
        demo_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_feats: torch.Tensor,
        actions_10d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(demo_feats, demo_pad_mask, query_feats, actions_10d)

    @torch.inference_mode()
    def sample_from_z_self(
        self,
        z_self: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """16-step DDIM denoising conditioned ONLY on `z_self` -> `(B, 16, 10)` actions."""
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

    @torch.inference_mode()
    def predict_action(
        self,
        demo_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_feats: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        aggregate_attn_temporal: bool = False,
    ) -> Dict[str, Any]:
        z_self, attn = self.encode_z_self(
            demo_feats,
            demo_pad_mask,
            query_feats,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            aggregate_attn_temporal=aggregate_attn_temporal,
        )
        action_10d = self.sample_from_z_self(
            z_self, generator=generator, init_noise=init_noise
        )
        action_7d = action_10d_to_7d(action_10d)
        exec_action_7d = extract_exec_actions(
            action_10d, exec_horizon=self.exec_horizon
        )
        return {
            "action_10d": action_10d,
            "action_7d": action_7d,
            "exec_action_7d": exec_action_7d,
            "z_self": z_self,
            "cross_attn_weights": attn,
        }

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Exact BPP / CLIP Stage 1 AdamW decay / no-decay parameter grouping."""
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

        trainable = [p for p in self.parameters() if p.requires_grad]
        if len(decay_params) + len(no_decay_params) != len(trainable):
            raise RuntimeError("Optimizer parameter grouping mismatch")

        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
        ]

    def parameter_summary(
        self, visual_encoder_params: int = DINOV3_FROZEN_BACKBONE_PARAMS
    ) -> Dict[str, Any]:
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
            "visual_encoder_frozen_dinov3": {"total": visual_encoder_params, "trainable": 0},
            "prompt_pooler": _count(self.prompt_pooler),
            "query_pooler": _count(self.query_pooler),
            "reference_reader": _count(self.reference_reader),
            "action_decoder": _count(self.action_decoder),
        }
        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        policy_total = sum(p.numel() for p in self.parameters())
        return {
            "modules": modules,
            "policy_total": policy_total,
            "policy_trainable": trainable_total,
            "global_cond_dim": self.global_cond_dim,
            "feature_mode": self.feature_mode,
            "num_spatial_tokens": self.num_spatial_tokens,
            "demo_tokens_per_step": self.demo_tokens_per_step,
            "full_system_total_with_frozen_dinov3": policy_total + visual_encoder_params,
        }


def load_frozen_stage1_dinov3_teacher(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> E6Stage1DINOv3Policy:
    """Load a saved E6 Stage 1 (DINOv3) checkpoint and freeze all weights for Stage 2 Teacher use."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    policy = E6Stage1DINOv3Policy(
        feature_mode=str(cfg.get("feature_mode", "spatial")),
        num_spatial_tokens=cfg.get("num_spatial_tokens"),
        demo_tokens_per_step=int(
            cfg.get("demo_tokens_per_step", DEFAULT_DEMO_TOKENS_PER_STEP)
        ),
        use_attention_sink=bool(cfg.get("use_attention_sink", False)),
    )
    policy.load_state_dict(payload["model"], strict=True)
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.eval()
    return policy.to(device)
