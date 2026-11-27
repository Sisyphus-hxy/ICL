"""E9 Stage 2 Model Builder & Loader: strictly reuses `e7/model.py::E7Policy`
(`prediction_mode="direct"`, Global Soft Attention Reader + Direct TransferTransformer,
NO Source State Conditioning, NO residual skip) with frozen E9 Stage 1 Local Teacher
(`e9/outputs/stage1_local/best.pt`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from e7.model import E7Policy

from . import DEFAULT_STAGE1_LOCAL_CKPT, EMBED_DIM, NUM_HEADS
from .stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher


def build_e9_stage2_direct_policy(
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
) -> E7Policy:
    """Construct exact E7-Direct policy over frozen E9 Stage 1 Local teacher (from scratch)."""
    return E7Policy(
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


def load_e9_stage2_direct_policy(
    e9_stage2_ckpt: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E7Policy, Dict[str, Any]]:
    """Load frozen E9 Stage 1 Local Teacher + trained E9 Stage 2 Direct Reader & TransferTransformer."""
    payload = torch.load(Path(e9_stage2_ckpt), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    s1_path = (
        Path(stage1_ckpt)
        if stage1_ckpt is not None
        else Path(cfg.get("stage1_ckpt", DEFAULT_STAGE1_LOCAL_CKPT))
    )
    teacher = load_frozen_e9_stage1_teacher(s1_path, device="cpu")
    direct_latent_bias = payload.get("direct_latent_bias", None)
    if direct_latent_bias is None:
        raise ValueError(f"Missing direct_latent_bias in checkpoint {e9_stage2_ckpt}")

    s_z = float(payload.get("latent_scale_s_z", cfg["latent_scale_s_z"]))
    policy = build_e9_stage2_direct_policy(
        teacher,
        latent_scale_s_z=s_z,
        direct_latent_bias=direct_latent_bias,
        reader_attn_dim=int(cfg.get("reader_attn_dim", EMBED_DIM)),
        residual_layers=int(cfg.get("residual_layers", 3)),
        residual_heads=int(cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
        direct_out_proj_init_std=float(cfg.get("direct_out_proj_init_std", 0.002)),
    )
    policy.reader.load_state_dict(payload["reader"], strict=True)
    policy.residual_transformer.load_state_dict(payload["residual_transformer"], strict=True)
    policy.eval()
    return policy.to(device), payload
