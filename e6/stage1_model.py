"""E6 Stage 1: Self-conditioned Behavior Latent Learning (CLIP, RGB-only).

Architecture:
    M_D    = RGBPromptPooler(demo_rgb_feats)            # (B, P, 768), preserves temporal order
    z_self = BehaviorReferenceReader(M_D, h_q)          # (B, 4, 768), NO h_q residual skip
    A_q   ~ ConditionalUnet1D(global_cond=z_self)       # global_cond_dim = 4 * 768 = 3072 (NO h_q)

No ResidualTransformer, no Auxiliary Head, no Progress Alignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import ActionNormalizer, action_10d_to_7d, extract_exec_actions
from e5.bpp_compat import import_bpp_components
from e5.diffusion import DDIMScheduler
from e5.model import E5Policy, RGBPromptPooler

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    DEFAULT_INIT_CKPT,
    EMBED_DIM,
    EXEC_HORIZON,
    MAX_DEMO_LEN,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
    Z_SELF_COND_DIM,
)

_BPP = import_bpp_components()
ConditionalUnet1D = _BPP["ConditionalUnet1D"]
TransformerDecoderLayerWithAttn = _BPP["TransformerDecoderLayerWithAttn"]
TransformerDecoderWithAttn = _BPP["TransformerDecoderWithAttn"]
bpp_init_weights = _BPP["init_weights"]


class QueryReadLayer(TransformerDecoderLayerWithAttn):
    """First ReferenceReader layer with an explicit Demo-memory information bottleneck.

    `tgt` (derived from `h_q`) is used strictly to compute the attention Query (`Q`)
    for reading from Demo Memory (`K`, `V`). Unlike a standard TransformerDecoderLayer,
    there is NO residual/skip connection from `tgt` (`h_q`) across cross-attention, so
    Query hidden states cannot bypass Demo Memory into `z_self`.
    """

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ):
        if not self.norm_first:
            raise ValueError("QueryReadLayer requires norm_first=True")

        # 1. Refine the 4 query tokens among themselves (used strictly as attention Query).
        q = tgt + self._sa_block(
            self.norm1(tgt), tgt_mask, tgt_key_padding_mask, tgt_is_causal
        )
        # 2. Cross-attention reading from Demo Memory (K/V = memory).
        #    NO residual skip from `q` (`h_q`) -> output is strictly a convex combination
        #    of Value projections from Demo Memory.
        ret = self._mha_block(
            self.norm2(q),
            memory,
            memory_mask,
            memory_key_padding_mask,
            memory_is_causal,
            need_weights,
            average_attn_weights,
        )
        if need_weights:
            x, cross_attn_weights = ret
        else:
            x = ret
        # 3. FFN refines the Demo-read content.
        x = x + self._ff_block(self.norm3(x))

        if need_weights:
            return x, cross_attn_weights
        return x


class BehaviorReferenceReader(nn.Module):
    """6-layer ReferenceReader producing `z_self` (B, 4, 768) strictly from Demo Memory.

    - Layer 0 (`QueryReadLayer`): `h_q + prompt_current_obs_pos_emb` serves strictly as
      attention Query to read from temporal Demo Memory `M_D + prompt_pos_emb`, with NO
      residual/skip path from `h_q`.
    - Layers 1..5 (`TransformerDecoderLayerWithAttn`): iteratively refine the Demo-sourced
      representation `x` via self-attention, cross-attention on `M_D + prompt_pos_emb`,
      and FFN.
    - Attention sinks are disabled by default (`use_attention_sink=False`) so `h_q` cannot
      modulate constant sink tokens instead of reading Demo Memory.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 6,
        num_heads: int = NUM_HEADS,
        max_prompt_tokens: int = MAX_DEMO_LEN,
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
        # Replace Layer 0 with the no-skip QueryReadLayer (identical parameter names & shapes).
        self.prompt_with_obs_decoder.layers[0] = QueryReadLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.prompt_pos_emb = nn.Parameter(torch.randn(1, max_prompt_tokens, embed_dim))
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
        nn.init.normal_(self.prompt_current_obs_pos_emb, mean=0.0, std=0.02)
        if self.attention_sink_tokens is not None:
            nn.init.normal_(self.attention_sink_tokens, mean=0.0, std=0.02)

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
            M_d: (B, P, D) pooled Demo memory tokens in temporal order.
            h_q: (B, 4, D) Query RGB tokens [t-1:av, t-1:eh, t:av, t:eh].
            demo_pad_mask: (B, P) bool mask where True = padded / ignored.
        Returns:
            z_self: (B, 4, D)
            cross_attn_weights: optional (L, B, [H], 4, P)
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


class E6Stage1Policy(nn.Module):
    """E6 Stage 1 Self-conditioned Behavior Latent Learning policy.

    - Frozen CLIP (offline cached / online via `e5.vision.BPPVisualEncoder`)
    - `RGBPromptPooler` (reused from E5/BPP)
    - `BehaviorReferenceReader` (no `h_q` residual skip, no attention sinks)
    - `ConditionalUnet1D` conditioned ONLY on `z_self` (`global_cond_dim=3072`, NO `h_q`)
    - Trained purely with Diffusion Loss on 100% Self-pairs.
    """

    def __init__(
        self,
        *,
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
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        self.num_inference_steps = num_inference_steps
        self.input_pertub = input_pertub
        self.global_cond_dim = NUM_QUERY_TOKENS * embed_dim  # 4 * 768 = 3072

        # 1. Demo Prompt Pooler (exact E5 RGBPromptPooler)
        self.prompt_pooler = RGBPromptPooler(
            embed_dim=embed_dim, num_heads=NUM_HEADS, mlp_ratio=4.0
        )

        # 2. Modified ReferenceReader: h_q used strictly as Query, no residual bypass
        self.reference_reader = BehaviorReferenceReader(
            embed_dim=embed_dim,
            num_layers=6,
            num_heads=NUM_HEADS,
            max_prompt_tokens=MAX_DEMO_LEN,
            num_query_tokens=NUM_QUERY_TOKENS,
            dropout=dropout,
            use_attention_sink=use_attention_sink,
        )

        # 3. Official BPP ConditionalUnet1D conditioned ONLY on z_self (global_cond_dim=3072)
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

        # 4. Exact BPP DDIM scheduler & ActionNormalizer
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
        self, ckpt_path: Path = DEFAULT_INIT_CKPT
    ) -> Dict[str, Any]:
        """Load all shape-compatible weights from a trained BPP RGB-only / E5 checkpoint,
        and explicitly record every incompatible or unused tensor."""
        ckpt_path = Path(ckpt_path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd: Dict[str, torch.Tensor] = payload["model"]
        tgt_sd = self.state_dict()

        to_load: Dict[str, torch.Tensor] = {}
        module_stats: Dict[str, Dict[str, Any]] = {}
        incompatible: List[Dict[str, Any]] = []
        missing_in_src: List[str] = []

        for key, tgt_tensor in tgt_sd.items():
            mod = key.split(".", 1)[0]
            st = module_stats.setdefault(
                mod,
                {
                    "loaded_tensors": 0,
                    "loaded_params": 0,
                    "reinit_tensors": 0,
                    "reinit_params": 0,
                },
            )
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
            if k not in tgt_sd
        ]

        self.load_state_dict(to_load, strict=False)

        total_params = sum(v.numel() for v in tgt_sd.values())
        loaded_params = sum(v.numel() for v in to_load.values())
        reinit_params = total_params - loaded_params

        report = {
            "source_checkpoint": str(ckpt_path),
            "source_mode": payload.get("mode"),
            "source_epoch": payload.get("epoch"),
            "source_global_step": payload.get("global_step"),
            "total_target_params": total_params,
            "loaded_compatible_params": loaded_params,
            "loaded_fraction": round(loaded_params / max(total_params, 1), 6),
            "reinit_incompatible_params": reinit_params,
            "per_module": module_stats,
            "shape_incompatible_tensors": incompatible,
            "missing_in_source": missing_in_src,
            "unused_source_tensors": unused_in_src,
        }
        self.weight_compat_report = report
        return report

    def encode_z_self(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Read behavior latent `z_self` (B, 4, 768) from Demo Memory using Query `h_q`."""
        h_q = E5Policy.format_query_tokens(query_rgb_feats)
        M_d = self.prompt_pooler(demo_rgb_feats, demo_pad_mask=demo_pad_mask)
        z_self, attn = self.reference_reader(
            M_d,
            h_q,
            demo_pad_mask=demo_pad_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        return z_self, attn

    def build_global_cond(self, z_self: torch.Tensor) -> torch.Tensor:
        """Flatten `z_self` (B, 4, 768) -> (B, 3072). Strictly excludes `h_q`."""
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
        """Per-sample DDPM-IP epsilon MSE loss of shape (B,) conditioned ONLY on `z_self`."""
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
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        actions_10d: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pure Diffusion Loss conditioned strictly on `z_self`."""
        z_self, _ = self.encode_z_self(demo_rgb_feats, demo_pad_mask, query_rgb_feats)
        per_sample_diff = self.diffusion_loss_from_z_self(
            z_self,
            actions_10d,
            noise=noise,
            pertub_noise=pertub_noise,
            timesteps=timesteps,
        )
        diff_loss = per_sample_diff.mean()
        z_self_rms = torch.sqrt(
            z_self.detach().float().square().mean() + 1e-12
        )
        return {
            "loss": diff_loss,
            "diff_loss": diff_loss.detach(),
            "z_self_rms": z_self_rms,
        }

    def forward(
        self,
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        actions_10d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(
            demo_rgb_feats, demo_pad_mask, query_rgb_feats, actions_10d
        )

    @torch.inference_mode()
    def sample_from_z_self(
        self,
        z_self: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """16-step DDIM denoising conditioned ONLY on `z_self` -> (B, 16, 10) actions."""
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
        demo_rgb_feats: torch.Tensor,
        demo_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Dict[str, Any]:
        z_self, attn = self.encode_z_self(
            demo_rgb_feats,
            demo_pad_mask,
            query_rgb_feats,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
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
        """Exact BPP AdamW decay / no-decay parameter grouping."""
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
            "prompt_pooler": _count(self.prompt_pooler),
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
            "full_system_total_with_frozen_clip": policy_total + visual_encoder_params,
        }


def load_frozen_stage1_teacher(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> E6Stage1Policy:
    """Load a saved E6 Stage 1 checkpoint and freeze all weights for E6 Stage 2 Teacher use."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    policy = E6Stage1Policy(
        use_attention_sink=bool(cfg.get("use_attention_sink", False)),
    )
    policy.load_state_dict(payload["model"], strict=True)
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.eval()
    return policy.to(device)
