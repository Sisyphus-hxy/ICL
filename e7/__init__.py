"""E7: Behavior Memory Retrieval + Latent Adaptation (CLIP, RGB-only)."""

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
    RAW_LIBERO,
    ROOT,
)
from e5.cache import DEFAULT_CACHE_DIR as E5_CACHE_DIR
from e6 import NUM_QUERY_TOKENS, Z_SELF_COND_DIM
from e6.stage2_data import DEFAULT_STAGE2_CACHE_DIR
from e6.stage2_model import DEFAULT_STAGE1_CKPT

E7_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E7_ROOT / "outputs"
DEFAULT_BEHAVIOR_MEM_DIR = OUTPUT_ROOT / "behavior_memory_cache"

BEHAVIOR_STRIDE = 16  # Extract 1 behavior latent per 16 raw control steps
FIXED_LATENT_SCALE_S_Z = 7.822748

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
    "E7_ROOT",
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
