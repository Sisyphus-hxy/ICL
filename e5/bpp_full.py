"""Full BPP reproduction (RGB + proprio + 20-step prompt actions) under the exact same
frozen CLIP cache, 80-epoch unique starts manifest, and 2-GPU training budget as E5
and BPP RGB-only, for isolating architecture vs training-budget effects."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.attention_pool import AttentionPoolLatent
from timm.layers.mlp import Mlp
from torch.nn.parallel import DistributedDataParallel

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    BPP_CKPT,
    DEMO_STRIDE,
    EMBED_DIM,
    EXEC_HORIZON,
    MAX_DEMO_LEN,
    NUM_ATTENTION_SINKS,
    NUM_HEADS,
    OUTPUT_ROOT,
    QUERY_HISTORY,
)
from .actions import ActionNormalizer, action_10d_to_7d, extract_exec_actions
from .bpp_compat import (
    _load_raw_bpp_state_dict,
    import_bpp_components,
    load_bpp_pretrained_weights,
)
from .cache import DEFAULT_CACHE_DIR
from .data import E5FeatureBank
from .diffusion import DDIMScheduler, EMAModel, get_cosine_schedule_with_warmup
from .train import (
    _action_error_sums,
    _all_reduce_sum,
    _is_dist,
    _rank,
    _unpack_action_errors,
    _world,
    atomic_torch_save,
    atomic_write_json,
    seed_everything,
)

_BPP = import_bpp_components()
ConditionalUnet1D = _BPP["ConditionalUnet1D"]
TransformerDecoderLayerWithAttn = _BPP["TransformerDecoderLayerWithAttn"]
TransformerDecoderWithAttn = _BPP["TransformerDecoderWithAttn"]
bpp_init_weights = _BPP["init_weights"]

FULL_QUERY_TOKENS = QUERY_HISTORY * 5  # 2 steps x (av, eh, ee_ori, ee_pos, grip) = 10


class ProprioNormalizer(nn.Module):
    """Exact replica of BPP's proprioception normalizer:
    ee_pos (3D) -> [-1, 1], ee_ori (6D rot6d) -> identity, gripper_states (2D) -> [-1, 1]."""

    def __init__(self, state: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        if state is None:
            scale = torch.ones(11, dtype=torch.float32)
            offset = torch.zeros(11, dtype=torch.float32)
        else:
            scale = torch.cat(
                [state["ee_pos_scale"], state["ee_ori_scale"], state["gripper_states_scale"]]
            ).float()
            offset = torch.cat(
                [state["ee_pos_offset"], state["ee_ori_offset"], state["gripper_states_offset"]]
            ).float()
        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(device=x.device, dtype=x.dtype) + self.offset.to(
            device=x.device, dtype=x.dtype
        )


class BPPFullPolicy(nn.Module):
    """Exact official BPP PairPromptObsEncoder + ConditionalUnet1D (283M) operating on
    cached frozen CLIP features, 11D proprioception, and 20-step chunked prompt actions."""

    def __init__(
        self,
        *,
        embed_dim: int = EMBED_DIM,
        action_dim: int = ACTION_DIM_10D,
        action_horizon: int = ACTION_HORIZON,
        exec_horizon: int = EXEC_HORIZON,
        prompt_chunk_n_actions: int = DEMO_STRIDE,
        num_train_timesteps: int = 50,
        num_inference_steps: int = 16,
        input_pertub: float = 0.1,
        dropout: float = 0.1,
        action_normalizer: Optional[ActionNormalizer] = None,
        proprio_normalizer: Optional[ProprioNormalizer] = None,
        pretrained_bundle: Optional[Dict[str, Any]] = None,
        load_pretrained_reader: bool = True,
        ckpt_path: Path = BPP_CKPT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        self.prompt_chunk_n_actions = prompt_chunk_n_actions
        self.num_inference_steps = num_inference_steps
        self.input_pertub = input_pertub

        # Shared low-dim modality projections (exact BPP TransformerObsEncoder)
        self.key_projection_map = nn.ModuleDict(
            {
                "ee_ori": nn.Linear(6, embed_dim),
                "ee_pos": nn.Linear(3, embed_dim),
                "gripper_states": nn.Linear(2, embed_dim),
            }
        )
        # Exact BPP prompt 20-step action chunk MLP: 200 -> 200 -> 768
        self.action_proj = Mlp(
            in_features=action_dim * prompt_chunk_n_actions,
            out_features=embed_dim,
        )
        # Exact BPP 6-token (2 RGB + 3 proprio + 1 action_chunk) AttentionPoolLatent
        self.prompt_attention_pool = AttentionPoolLatent(
            in_features=embed_dim,
            num_heads=NUM_HEADS,
            mlp_ratio=4.0,
            pool_type="token",
        )

        # Exact BPP 6-layer Reference Reader (56.71M) + full 10-token query pos emb
        decoder_layer = TransformerDecoderLayerWithAttn(
            d_model=embed_dim,
            nhead=NUM_HEADS,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_with_obs_decoder = TransformerDecoderWithAttn(
            decoder_layer=decoder_layer, num_layers=6
        )
        self.prompt_pos_emb = nn.Parameter(torch.randn(1, MAX_DEMO_LEN, embed_dim))
        self.prompt_current_obs_pos_emb = nn.Parameter(
            torch.randn(1, FULL_QUERY_TOKENS, embed_dim)
        )
        self.attention_sink_tokens = nn.Parameter(
            torch.zeros(1, NUM_ATTENTION_SINKS, embed_dim)
        )

        self.prompt_with_obs_decoder.apply(bpp_init_weights)
        nn.init.normal_(self.prompt_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.prompt_current_obs_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.attention_sink_tokens, mean=0.0, std=0.02)

        # Exact BPP 283M ConditionalUnet1D (global_cond = 20 * 768 = 15360)
        global_cond_dim = (FULL_QUERY_TOKENS * 2) * embed_dim
        self.action_decoder = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

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
        self.proprio_normalizer = (
            proprio_normalizer if proprio_normalizer is not None else ProprioNormalizer()
        )

        self.weight_compat_report: Dict[str, Any] = {}
        if load_pretrained_reader:
            bundle = (
                pretrained_bundle
                if pretrained_bundle is not None
                else load_bpp_pretrained_weights(ckpt_path=ckpt_path)
            )
            self.prompt_with_obs_decoder.load_state_dict(bundle["reader_decoder"], strict=True)
            with torch.no_grad():
                self.prompt_pos_emb.copy_(bundle["reader_prompt_pos_emb"])
                self.attention_sink_tokens.copy_(bundle["reader_attention_sink_tokens"])
                self.prompt_current_obs_pos_emb.copy_(
                    bundle["reader_current_obs_pos_emb_full"]
                )
            self.weight_compat_report = {
                "reference_reader": "loaded_all_10_tokens_strict",
                "prompt_pooler_and_projs": "retrained_from_scratch (matched with e5 & bpp_rgb_only)",
                "action_decoder": "retrained_from_scratch (283M, matched with e5 & bpp_rgb_only)",
            }

    def load_full_official_bpp_checkpoint(self, ckpt_path: Path = BPP_CKPT) -> None:
        """Strictly load ALL non-CLIP weights + normalizers from official BPP checkpoint
        for zero-shot oracle reference evaluation."""
        sd = _load_raw_bpp_state_dict(ckpt_path)
        with torch.no_grad():
            for key in ("ee_ori", "ee_pos", "gripper_states"):
                p = f"obs_encoder.obs_encoder.prompt_obs_encoder.key_projection_map.{key}."
                self.key_projection_map[key].load_state_dict(
                    {k[len(p) :]: v for k, v in sd.items() if k.startswith(p)}, strict=True
                )
            ap = "obs_encoder.obs_encoder.action_proj."
            self.action_proj.load_state_dict(
                {k[len(ap) :]: v for k, v in sd.items() if k.startswith(ap)}, strict=True
            )
            pp = "obs_encoder.obs_encoder.prompt_attention_pool."
            self.prompt_attention_pool.load_state_dict(
                {k[len(pp) :]: v for k, v in sd.items() if k.startswith(pp)}, strict=True
            )
            dp = "obs_encoder.prompt_with_obs_decoder."
            self.prompt_with_obs_decoder.load_state_dict(
                {k[len(dp) :]: v for k, v in sd.items() if k.startswith(dp)}, strict=True
            )
            self.prompt_pos_emb.copy_(sd["obs_encoder.prompt_pos_emb"])
            self.prompt_current_obs_pos_emb.copy_(sd["obs_encoder.prompt_current_obs_pos_emb"])
            self.attention_sink_tokens.copy_(sd["obs_encoder.attention_sink_tokens"])
            mp = "model."
            self.action_decoder.load_state_dict(
                {k[len(mp) :]: v for k, v in sd.items() if k.startswith(mp)}, strict=True
            )
            self.action_normalizer.scale.copy_(sd["normalizer.params_dict.action.scale"])
            self.action_normalizer.offset.copy_(sd["normalizer.params_dict.action.offset"])
            self.proprio_normalizer.scale.copy_(
                torch.cat(
                    [
                        sd["normalizer.params_dict.ee_pos.scale"],
                        sd["normalizer.params_dict.ee_ori.scale"],
                        sd["normalizer.params_dict.gripper_states.scale"],
                    ]
                )
            )
            self.proprio_normalizer.offset.copy_(
                torch.cat(
                    [
                        sd["normalizer.params_dict.ee_pos.offset"],
                        sd["normalizer.params_dict.ee_ori.offset"],
                        sd["normalizer.params_dict.gripper_states.offset"],
                    ]
                )
            )

    def _project_proprio_tokens(self, proprio_11d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize 11D [ee_pos(3), ee_ori(6), grip(2)] and project in BPP's sorted order:
        (ee_ori, ee_pos, gripper_states)."""
        nprop = self.proprio_normalizer.normalize(proprio_11d)
        ee_pos = self.key_projection_map["ee_pos"](nprop[..., :3])
        ee_ori = self.key_projection_map["ee_ori"](nprop[..., 3:9])
        grip = self.key_projection_map["gripper_states"](nprop[..., 9:11])
        return ee_ori, ee_pos, grip

    def encode_global_cond(
        self,
        demo_rgb: torch.Tensor,
        demo_proprio: torch.Tensor,
        demo_actions_20: torch.Tensor,
        demo_pad_mask: torch.Tensor,
        query_rgb: torch.Tensor,
        query_proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Exact BPP PairPromptObsEncoder forward returning flattened (B, 20*768) global_cond."""
        B, P = demo_rgb.shape[:2]
        D = self.embed_dim

        # 1. Prompt step tokens: [av, eh, ee_ori, ee_pos, grip, action_20x10] -> pool -> (B, P, D)
        d_ori, d_pos, d_grip = self._project_proprio_tokens(demo_proprio)
        d_act_norm = self.action_normalizer.normalize(demo_actions_20)
        d_act_tok = self.action_proj(d_act_norm.reshape(B, P, -1))
        prompt_6 = torch.stack(
            [demo_rgb[:, :, 0], demo_rgb[:, :, 1], d_ori, d_pos, d_grip, d_act_tok], dim=2
        )  # (B, P, 6, D)
        M_d = self.prompt_attention_pool(prompt_6.reshape(B * P, 6, D)).reshape(B, P, D)
        M_d = M_d.masked_fill(demo_pad_mask.unsqueeze(-1), 0.0)

        # 2. Query receding obs tokens: 2 steps x [av, eh, ee_ori, ee_pos, grip] -> (B, 10, D)
        q_ori, q_pos, q_grip = self._project_proprio_tokens(query_proprio)
        h_q = torch.stack(
            [query_rgb[:, :, 0], query_rgb[:, :, 1], q_ori, q_pos, q_grip], dim=2
        ).reshape(B, FULL_QUERY_TOKENS, D)

        # 3. 6-layer Reference Reader with attention sinks
        memory = M_d + self.prompt_pos_emb[:, :P, :]
        sink = self.attention_sink_tokens.expand(B, NUM_ATTENTION_SINKS, D)
        memory = torch.cat([sink, memory], dim=1)
        sink_mask = torch.zeros((B, NUM_ATTENTION_SINKS), device=demo_pad_mask.device, dtype=torch.bool)
        mask = torch.cat([sink_mask, demo_pad_mask], dim=1)

        z_ref = self.prompt_with_obs_decoder(
            tgt=h_q + self.prompt_current_obs_pos_emb,
            memory=memory,
            memory_key_padding_mask=mask,
        )  # (B, 10, D)

        return torch.cat([h_q, z_ref], dim=1).reshape(B, -1)  # (B, 15360)

    def diffusion_loss_from_cond(
        self,
        global_cond: torch.Tensor,
        actions_10d: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        noisy = self.noise_scheduler.add_noise(nactions.float(), noise_new, timesteps)
        pred = self.action_decoder(
            noisy.to(dtype=global_cond.dtype), timesteps, local_cond=None, global_cond=global_cond
        )
        return F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2))

    def forward(
        self,
        demo_rgb: torch.Tensor,
        demo_proprio: torch.Tensor,
        demo_actions_20: torch.Tensor,
        demo_pad_mask: torch.Tensor,
        query_rgb: torch.Tensor,
        query_proprio: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gcond = self.encode_global_cond(
            demo_rgb, demo_proprio, demo_actions_20, demo_pad_mask, query_rgb, query_proprio
        )
        per_sample = self.diffusion_loss_from_cond(gcond, actions_10d)
        is_self = is_self.to(device=gcond.device, dtype=torch.bool)
        sm = is_self.float()
        cm = (~is_self).float()
        ns = sm.sum()
        nc = cm.sum()
        return {
            "loss": per_sample.mean(),
            "diff_loss": per_sample.mean().detach(),
            "diff_self": ((per_sample * sm).sum() / ns.clamp_min(1.0)).detach(),
            "diff_cross": ((per_sample * cm).sum() / nc.clamp_min(1.0)).detach(),
            "n_self": ns.detach(),
            "n_cross": nc.detach(),
        }

    @torch.inference_mode()
    def predict_action(
        self,
        demo_rgb: torch.Tensor,
        demo_proprio: torch.Tensor,
        demo_actions_20: torch.Tensor,
        demo_pad_mask: torch.Tensor,
        query_rgb: torch.Tensor,
        query_proprio: torch.Tensor,
        *,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gcond = self.encode_global_cond(
            demo_rgb, demo_proprio, demo_actions_20, demo_pad_mask, query_rgb, query_proprio
        )
        B = gcond.shape[0]
        device = gcond.device
        traj = (
            init_noise.to(device=device, dtype=torch.float32).clone()
            if init_noise is not None
            else torch.randn((B, self.action_horizon, self.action_dim), device=device)
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)
        for t in self.noise_scheduler.timesteps:
            out = self.action_decoder(
                traj.to(dtype=gcond.dtype), t, local_cond=None, global_cond=gcond
            ).float()
            traj = self.noise_scheduler.step(out, t, traj).prev_sample
        action_10d = self.action_normalizer.unnormalize(traj)
        return {
            "action_10d": action_10d,
            "action_7d": action_10d_to_7d(action_10d),
            "exec_action_7d": extract_exec_actions(action_10d, exec_horizon=self.exec_horizon),
        }

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        decay, no_decay = [], []
        no_decay_kw = ("bias", "norm", "pos_emb", "attention_sink_tokens", "latent")
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(kw in name for kw in no_decay_kw):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        ]

    def parameter_summary(self, visual_encoder_params: int = 171_598_848) -> Dict[str, Any]:
        def _c(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        reader_params = (
            _c(self.prompt_with_obs_decoder)
            + self.prompt_pos_emb.numel()
            + self.prompt_current_obs_pos_emb.numel()
            + self.attention_sink_tokens.numel()
        )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "modules": {
                "visual_encoder_frozen": {"total": visual_encoder_params, "trainable": 0},
                "lowdim_and_action_projs": {
                    "total": _c(self.key_projection_map) + _c(self.action_proj),
                    "trainable": _c(self.key_projection_map) + _c(self.action_proj),
                },
                "prompt_pooler": {
                    "total": _c(self.prompt_attention_pool),
                    "trainable": _c(self.prompt_attention_pool),
                },
                "reference_reader": {"total": reader_params, "trainable": reader_params},
                "action_decoder": {
                    "total": _c(self.action_decoder),
                    "trainable": _c(self.action_decoder),
                },
            },
            "policy_total": trainable,
            "policy_trainable": trainable,
            "full_system_total_with_frozen_clip": trainable + visual_encoder_params,
        }


class BPPFullBank(E5FeatureBank):
    """Extends E5FeatureBank with GPU-resident 11D proprioception & 20-step prompt actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prop_np = np.load(self.cache_dir / "proprio_11d.npy", allow_pickle=False)
        self.proprio_11d = torch.from_numpy(prop_np).to(self.storage_device)
        pnorm_state = torch.load(
            self.cache_dir / "proprio_normalizer.pt", map_location="cpu", weights_only=True
        )
        self.proprio_normalizer = ProprioNormalizer(pnorm_state)

    def gather_full_batch(
        self,
        demos: np.ndarray,
        queries: np.ndarray,
        frames: np.ndarray,
        q_rows: np.ndarray,
        is_self: np.ndarray,
    ):
        demo_rgb, pad_mask, query_rgb, actions_10d, is_self_t = self.gather_batch(
            demos, queries, frames, q_rows, is_self
        )
        dev = self.storage_device
        demos_t = torch.as_tensor(demos, dtype=torch.long, device=dev)
        queries_t = torch.as_tensor(queries, dtype=torch.long, device=dev)
        frames_t = torch.as_tensor(frames, dtype=torch.long, device=dev)

        B, max_p = demo_rgb.shape[:2]
        d_ep_len = self.ep_lengths[demos_t]  # (B,)
        d_act_off = self.action_offsets[demos_t]  # (B,)

        # 1Hz Demo proprio at steps 0, 20, 40, ...
        pos = torch.arange(max_p, device=dev, dtype=torch.long)[None, :]  # (1, max_p)
        step_1hz = torch.minimum(pos * DEMO_STRIDE, (d_ep_len - 1)[:, None])
        demo_proprio = self.proprio_11d[d_act_off[:, None] + step_1hz]  # (B, max_p, 11)
        demo_proprio = demo_proprio.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # 20-step chunked prompt actions with BPP pad_end_prompt_actions='zeros'
        sub = torch.arange(max_p * DEMO_STRIDE, device=dev, dtype=torch.long)[None, :]  # (1, max_p*20)
        valid_act = sub < d_ep_len[:, None]  # (B, max_p*20)
        safe_sub = torch.minimum(sub, (d_ep_len - 1)[:, None])
        demo_acts = self.actions_10d[d_act_off[:, None] + safe_sub]  # (B, max_p*20, 10)
        demo_acts = demo_acts.masked_fill(~valid_act.unsqueeze(-1), 0.0)
        demo_actions_20 = demo_acts.reshape(B, max_p, DEMO_STRIDE, ACTION_DIM_10D)

        # Query proprio at [max(0, t-1), t]
        q_act_off = self.action_offsets[queries_t]  # (B,)
        q_prev = (frames_t - 1).clamp_min(0)
        q_steps = torch.stack([q_prev, frames_t], dim=1)  # (B, 2)
        query_proprio = self.proprio_11d[q_act_off[:, None] + q_steps]  # (B, 2, 11)

        if dev != self.device:
            demo_proprio = demo_proprio.to(self.device, non_blocking=True)
            demo_actions_20 = demo_actions_20.to(self.device, non_blocking=True)
            query_proprio = query_proprio.to(self.device, non_blocking=True)

        return (
            demo_rgb,
            demo_proprio,
            demo_actions_20,
            pad_mask,
            query_rgb,
            query_proprio,
            actions_10d,
            is_self_t,
        )

    def iter_full_train_batches(
        self, epoch: int, batch_size: int, rank: int = 0, world_size: int = 1
    ):
        demos, queries, frames, q_rows, is_self = self.build_train_epoch_schedule(epoch)
        demos = demos[rank::world_size]
        queries = queries[rank::world_size]
        frames = frames[rank::world_size]
        q_rows = q_rows[rank::world_size]
        is_self = is_self[rank::world_size]
        for start in range(0, len(demos), batch_size):
            end = min(start + batch_size, len(demos))
            yield self.gather_full_batch(
                demos[start:end],
                queries[start:end],
                frames[start:end],
                q_rows[start:end],
                is_self[start:end],
            )

    def iter_full_val_batches(
        self,
        mode: str,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
    ):
        n = len(self.val_keys) if max_samples is None else min(max_samples, len(self.val_keys))
        idx = np.arange(n)[rank::world_size]
        queries = self.val_keys[idx, 1].astype(np.int64)
        frames = self.val_keys[idx, 2].astype(np.int64)
        q_rows = self.val_q_rows[idx]
        if mode == "cross":
            demos = self.val_keys[idx, 0].astype(np.int64)
            is_self = np.zeros(len(idx), dtype=bool)
        elif mode == "self":
            demos = queries.copy()
            is_self = np.ones(len(idx), dtype=bool)
        elif mode == "wrong_demo":
            demos = self.val_wrong_demos[idx].astype(np.int64)
            is_self = np.zeros(len(idx), dtype=bool)
        else:
            raise ValueError(mode)
        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            yield self.gather_full_batch(
                demos[start:end],
                queries[start:end],
                frames[start:end],
                q_rows[start:end],
                is_self[start:end],
            )


@torch.inference_mode()
def evaluate_bpp_full(
    policy: BPPFullPolicy,
    bank: BPPFullBank,
    *,
    batch_size: int = 128,
    run_ddim: bool = False,
    ddim_max_samples: Optional[int] = 512,
    val_seed: int = 12345,
) -> Dict[str, Any]:
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    n_val = len(bank.val_keys)
    gen = torch.Generator(device="cpu").manual_seed(val_seed)
    all_noise = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen)
    all_pertub = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen)
    all_t = torch.randint(
        0, policy.noise_scheduler.config.num_train_timesteps, (n_val,), generator=gen
    )
    local_idx = np.arange(n_val)[rank::world]
    local_noise = all_noise[local_idx]
    local_pertub = all_pertub[local_idx]
    local_t = all_t[local_idx]

    acc = torch.zeros(4, device=device, dtype=torch.float64)
    c_it = bank.iter_full_val_batches("cross", batch_size, rank=rank, world_size=world)
    s_it = bank.iter_full_val_batches("self", batch_size, rank=rank, world_size=world)
    w_it = bank.iter_full_val_batches("wrong_demo", batch_size, rank=rank, world_size=world)
    cur = 0
    for bc, bs, bw in zip(c_it, s_it, w_it):
        B = len(bc[0])
        noise = local_noise[cur : cur + B].to(device, non_blocking=True)
        pertub = local_pertub[cur : cur + B].to(device, non_blocking=True)
        t = local_t[cur : cur + B].to(device, non_blocking=True)
        cur += B
        with amp_ctx:
            lc = policy.diffusion_loss_from_cond(
                policy.encode_global_cond(*bc[:6]), bc[6], noise=noise, pertub_noise=pertub, timesteps=t
            )
            ls = policy.diffusion_loss_from_cond(
                policy.encode_global_cond(*bs[:6]), bs[6], noise=noise, pertub_noise=pertub, timesteps=t
            )
            lw = policy.diffusion_loss_from_cond(
                policy.encode_global_cond(*bw[:6]), bw[6], noise=noise, pertub_noise=pertub, timesteps=t
            )
        acc[0] += float(B)
        acc[1] += lc.double().sum()
        acc[2] += ls.double().sum()
        acc[3] += lw.double().sum()

    acc = _all_reduce_sum(acc)
    tot = max(float(acc[0].item()), 1.0)
    vc = float(acc[1].item() / tot)
    vs = float(acc[2].item() / tot)
    vw = float(acc[3].item() / tot)
    res: Dict[str, Any] = {
        "val_samples": int(round(tot)),
        "val_cross_diff_loss": vc,
        "val_self_diff_loss": vs,
        "val_wrong_demo_diff_loss": vw,
        "val_demo_sensitivity_diff_loss": vw - vc,
    }

    if run_ddim:
        n_ddim = n_val if ddim_max_samples is None else min(ddim_max_samples, n_val)
        gen_d = torch.Generator(device="cpu").manual_seed(val_seed + 777)
        init_n = torch.randn((n_ddim, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_d)
        loc_n = init_n[np.arange(n_ddim)[rank::world]]
        sums = {m: torch.zeros(11, device=device, dtype=torch.float64) for m in ("cross", "self", "wrong_demo")}
        c_it = bank.iter_full_val_batches("cross", batch_size, rank=rank, world_size=world, max_samples=n_ddim)
        s_it = bank.iter_full_val_batches("self", batch_size, rank=rank, world_size=world, max_samples=n_ddim)
        w_it = bank.iter_full_val_batches("wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_ddim)
        cur = 0
        for bc, bs, bw in zip(c_it, s_it, w_it):
            B = len(bc[0])
            z0 = loc_n[cur : cur + B].to(device, non_blocking=True)
            cur += B
            with amp_ctx:
                sums["cross"] += _action_error_sums(
                    policy.predict_action(*bc[:6], init_noise=z0)["action_10d"], bc[6]
                )
                sums["self"] += _action_error_sums(
                    policy.predict_action(*bs[:6], init_noise=z0)["action_10d"], bs[6]
                )
                sums["wrong_demo"] += _action_error_sums(
                    policy.predict_action(*bw[:6], init_noise=z0)["action_10d"], bw[6]
                )
        ddim = {m: _unpack_action_errors(_all_reduce_sum(sums[m])) for m in sums}
        ddim["demo_sensitivity_action10_mse"] = (
            ddim["wrong_demo"]["action10_mse"] - ddim["cross"]["action10_mse"]
        )
        ddim["demo_sensitivity_action7_mae"] = (
            ddim["wrong_demo"]["action7_mae"] - ddim["cross"]["action7_mae"]
        )
        res["ddim"] = ddim
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--base-lr-per-gpu", type=float, default=5.0e-5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=0.3)
    parser.add_argument("--self-ratio", type=float, default=0.5)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument(
        "--schedule-mode",
        choices=("manifest_unique", "all_cached_starts"),
        default="manifest_unique",
    )
    parser.add_argument("--run-name", type=str, default="bpp_full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-official-ckpt", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if use_ddp:
        dist.init_process_group(backend="nccl", device_id=device)

    try:
        rank, world = _rank(), _world()
        run_dir = args.output_dir / args.run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        bank = BPPFullBank(
            cache_dir=args.cache_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            self_ratio=args.self_ratio,
            seed=args.seed,
            schedule_mode=args.schedule_mode,
            num_epochs=args.epochs,
        )

        # Optional zero-shot oracle check of official BPP checkpoint on exact same val set!
        if args.eval_official_ckpt:
            oracle = BPPFullPolicy(load_pretrained_reader=False).to(device)
            oracle.load_full_official_bpp_checkpoint(BPP_CKPT)
            oracle_eval = evaluate_bpp_full(
                oracle, bank, batch_size=args.val_batch_size, run_ddim=True,
                ddim_max_samples=None, val_seed=args.val_seed,
            )
            if rank == 0:
                atomic_write_json(oracle_eval, run_dir / "official_bpp_ckpt_eval.json")
                print("[Official BPP Ckpt Eval]", json.dumps(oracle_eval, indent=2), flush=True)
            if use_ddp:
                dist.barrier()
            return

        bundle = load_bpp_pretrained_weights(ckpt_path=BPP_CKPT)
        seed_everything(args.seed)
        policy = BPPFullPolicy(
            action_normalizer=bank.action_normalizer,
            proprio_normalizer=bank.proprio_normalizer,
            pretrained_bundle=bundle,
            load_pretrained_reader=True,
        ).to(device)

        per_rank_batch = args.global_batch // world
        lr = args.lr if args.lr is not None else (args.base_lr_per_gpu * world)
        optimizer = torch.optim.AdamW(
            policy.get_optim_groups(lr=lr, weight_decay=args.weight_decay),
            lr=lr, betas=(0.95, 0.999), weight_decay=args.weight_decay,
        )
        epoch_len = bank.samples_per_epoch()
        steps_per_epoch = math.ceil(epoch_len / args.global_batch)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(args.lr_warmup_proportion * total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        ema = EMAModel(
            policy.parameters(), decay=args.ema_decay, min_decay=0.0,
            update_after_step=0, use_ema_warmup=False, inv_gamma=1.0, power=0.75, foreach=True,
        ).to(device)

        config = {
            "mode": "bpp_full",
            "run_name": args.run_name,
            "schedule_mode": args.schedule_mode,
            "samples_per_epoch": epoch_len,
            "ema_decay": args.ema_decay,
            "seed": args.seed,
            "epochs": args.epochs,
            "world_size": world,
            "global_batch": args.global_batch,
            "per_rank_batch": per_rank_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "betas": [0.95, 0.999],
            "clip_grad_norm": args.clip_grad_norm,
            "self_ratio": args.self_ratio,
            "parameter_summary": policy.parameter_summary(),
            "weight_compat_report": policy.weight_compat_report,
        }

        last_ckpt, best_ckpt = run_dir / "last.pt", run_dir / "best.pt"
        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        start_epoch = 1
        global_step = 0

        if args.resume and last_ckpt.is_file():
            ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            policy.load_state_dict(ck["model"], strict=True)
            ema.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["optimizer"])
            lr_scheduler.load_state_dict(ck["lr_scheduler"])
            history = ck["history"]
            best_val_loss = ck["best_val_loss"]
            best_epoch = ck["best_epoch"]
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]

        ddp_model: nn.Module = (
            DistributedDataParallel(policy, device_ids=[device.index]) if use_ddp else policy
        )
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type=device.type, enabled=False)
        )
        if rank == 0:
            atomic_write_json(config, run_dir / "config.json")
            ps = config["parameter_summary"]
            print(
                f"[BPP-Full Train] world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps) | lr={lr:.2e} | "
                f"trainable={ps['policy_trainable']:,} | full_with_clip={ps['full_system_total_with_frozen_clip']:,}",
                flush=True,
            )

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.monotonic()
            ddp_model.train()
            tr = torch.zeros(8, device=device, dtype=torch.float64)
            for batch in bank.iter_full_train_batches(epoch, per_rank_batch, rank=rank, world_size=world):
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(*batch)
                out["loss"].backward()
                gn = float(torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.clip_grad_norm).item())
                optimizer.step()
                lr_scheduler.step()
                ema.step(policy.parameters())
                global_step += 1
                B = float(len(batch[0]))
                ns, nc = float(out["n_self"].item()), float(out["n_cross"].item())
                tr[0] += B
                tr[1] += float(out["diff_loss"].item()) * B
                tr[2] += float(out["diff_self"].item()) * ns
                tr[3] += ns
                tr[4] += float(out["diff_cross"].item()) * nc
                tr[5] += nc
                tr[6] += gn
                tr[7] += 1.0
            tr = _all_reduce_sum(tr)
            t_train = time.monotonic() - t0

            n_tot = max(float(tr[0].item()), 1.0)
            train_m = {
                "diff_loss": float(tr[1].item() / n_tot),
                "diff_self": float(tr[2].item() / max(float(tr[3].item()), 1.0)),
                "diff_cross": float(tr[4].item() / max(float(tr[5].item()), 1.0)),
                "grad_norm_mean": float(tr[6].item() / max(float(tr[7].item()), 1.0)),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "train_seconds": round(t_train, 2),
            }

            t_v0 = time.monotonic()
            ema.store(policy.parameters())
            ema.copy_to(policy.parameters())
            do_ddim = epoch == 1 or epoch == args.epochs or (args.sample_every > 0 and epoch % args.sample_every == 0)
            val_m = evaluate_bpp_full(
                policy, bank, batch_size=args.val_batch_size, run_ddim=do_ddim,
                ddim_max_samples=args.sample_count, val_seed=args.val_seed,
            )
            val_m["val_seconds"] = round(time.monotonic() - t_v0, 2)
            is_best = val_m["val_cross_diff_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_m["val_cross_diff_loss"]
                best_epoch = epoch

            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_val_cross_diff_loss": best_val_loss,
                    "train": train_m,
                    "val": val_m,
                }
            )

            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "mode": "bpp_full",
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_cross_diff_loss": best_val_loss,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_m,
                    },
                    best_ckpt,
                )
            ema.restore(policy.parameters())

            if rank == 0:
                if epoch % args.save_last_every == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        {
                            "mode": "bpp_full",
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "history": history,
                            "config": config,
                        },
                        last_ckpt,
                    )
                atomic_write_json(
                    {
                        "mode": "bpp_full",
                        "best_epoch": best_epoch,
                        "best_val_cross_diff_loss": best_val_loss,
                        "history": history,
                    },
                    run_dir / "metrics.json",
                )
                dstr = ""
                if "ddim" in val_m:
                    dm = val_m["ddim"]
                    dstr = (
                        f" | DDIM mse10={dm['cross']['action10_mse']:.4f} "
                        f"(self={dm['self']['action10_mse']:.4f}, wrong={dm['wrong_demo']['action10_mse']:.4f}) "
                        f"mae7={dm['cross']['action7_mae']:.4f}"
                    )
                print(
                    f"[bpp_full Ep {epoch:03d}/{args.epochs}] "
                    f"tr_diff={train_m['diff_loss']:.4f} (s={train_m['diff_self']:.4f},c={train_m['diff_cross']:.4f}) | "
                    f"val_cross={val_m['val_cross_diff_loss']:.4f} "
                    f"(self={val_m['val_self_diff_loss']:.4f}, wrong={val_m['val_wrong_demo_diff_loss']:.4f}) | "
                    f"lr={train_m['lr']:.2e} t={t_train:.1f}+{val_m['val_seconds']:.1f}s{'*' if is_best else ''}{dstr}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Final full 6,180-sample DDIM evaluation on reloaded best.pt
        best_p = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(best_p["model"], strict=True)
        final_eval = evaluate_bpp_full(
            policy, bank, batch_size=args.val_batch_size, run_ddim=True,
            ddim_max_samples=None, val_seed=args.val_seed,
        )
        if rank == 0:
            report = {
                "mode": "bpp_full",
                "best_epoch": best_p["epoch"],
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_cross_diff_loss": best_p["best_val_cross_diff_loss"],
                "parameter_summary": config["parameter_summary"],
                "weight_compat_report": config["weight_compat_report"],
                "final_eval": final_eval,
            }
            atomic_write_json(report, run_dir / "final_eval.json")
            print(
                f"\n[bpp_full FINAL BEST EVAL (ep={best_p['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
