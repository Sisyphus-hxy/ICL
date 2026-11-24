"""E6 Stage 1: Self-conditioned Behavior Latent Learning (CLIP, RGB-only)."""

from pathlib import Path

from e5 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    BPP_CKPT,
    CAMERAS,
    DEMO_STRIDE,
    EMBED_DIM,
    EXEC_HORIZON,
    MAX_DEMO_LEN,
    NUM_HEADS,
    QUERY_HISTORY,
    ROOT,
)
from e5.cache import DEFAULT_CACHE_DIR as E5_CACHE_DIR

E6_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E6_ROOT / "outputs"
DEFAULT_INIT_CKPT = ROOT / "e5/outputs/training/bpp_rgb_only_fullpass/best.pt"

NUM_QUERY_TOKENS = QUERY_HISTORY * 2  # 2 steps x 2 cameras = 4 RGB tokens
Z_SELF_COND_DIM = NUM_QUERY_TOKENS * EMBED_DIM  # 4 x 768 = 3072 (z_self only, no h_q)

__all__ = [
    "ACTION_DIM_10D",
    "ACTION_DIM_7D",
    "ACTION_HORIZON",
    "BPP_CKPT",
    "CAMERAS",
    "DEFAULT_INIT_CKPT",
    "DEMO_STRIDE",
    "E5_CACHE_DIR",
    "E6_ROOT",
    "EMBED_DIM",
    "EXEC_HORIZON",
    "MAX_DEMO_LEN",
    "NUM_HEADS",
    "NUM_QUERY_TOKENS",
    "OUTPUT_ROOT",
    "QUERY_HISTORY",
    "ROOT",
    "Z_SELF_COND_DIM",
]
