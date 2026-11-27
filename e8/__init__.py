"""E8: Source-conditioned Latent Behavior Transfer (CLIP, RGB-only).

Extends E7-Direct by reading BOTH behavior latent `z_c` and its corresponding
source visual state `h_ref` using the SAME Global Soft Reader attention weights:
    z_c   = sum_i w_i * b_i
    h_ref = sum_i w_i * h_i^D
    z_q   = G_theta(z_c, h_ref, h_q)   (Direct Prediction, no residual skip)
"""

from pathlib import Path

from e7 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    BEHAVIOR_STRIDE,
    BPP_CKPT,
    CAMERAS,
    DEFAULT_BEHAVIOR_MEM_DIR,
    DEFAULT_STAGE1_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    DEMO_STRIDE,
    E5_CACHE_DIR,
    EMBED_DIM,
    EXEC_HORIZON,
    FIXED_LATENT_SCALE_S_Z,
    MAX_DEMO_LEN,
    NUM_HEADS,
    NUM_QUERY_TOKENS,
    QUERY_HISTORY,
    RAW_LIBERO,
    ROOT,
    Z_SELF_COND_DIM,
)

E8_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E8_ROOT / "outputs" / "source_conditioned"

__all__ = [
    "ACTION_DIM_10D",
    "ACTION_DIM_7D",
    "ACTION_HORIZON",
    "BEHAVIOR_STRIDE",
    "BPP_CKPT",
    "CAMERAS",
    "DEFAULT_BEHAVIOR_MEM_DIR",
    "DEFAULT_STAGE1_CKPT",
    "DEFAULT_STAGE2_CACHE_DIR",
    "DEMO_STRIDE",
    "E5_CACHE_DIR",
    "E8_ROOT",
    "EMBED_DIM",
    "EXEC_HORIZON",
    "FIXED_LATENT_SCALE_S_Z",
    "MAX_DEMO_LEN",
    "NUM_HEADS",
    "NUM_QUERY_TOKENS",
    "OUTPUT_ROOT",
    "QUERY_HISTORY",
    "RAW_LIBERO",
    "ROOT",
    "Z_SELF_COND_DIM",
]
