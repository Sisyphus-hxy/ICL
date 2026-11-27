"""E9 Stage 1: Local Behavior Encoder (CLIP, RGB-only).

Replaces E6 Stage 1's Full-Demo + Query-conditioned ReferenceReader with a direct
local video clip encoder:
    b_t = E_phi(V_{t:t+16})          in R^(4 x 768)
    a_hat_{t:t+16} = U_psi(b_t)      in R^(16 x 10)

Preserves frozen CLIP ViT-B/16 backbone, exact E6 Stage 1 data split / action window /
training budget / hyperparameters / pure Action Diffusion Loss, and full Behavior Memory,
latent shape, and Action Decoder compatibility with E7-Direct.
"""

from pathlib import Path

from e5 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    BPP_CKPT,
    CAMERAS,
    EMBED_DIM,
    EXEC_HORIZON,
    NUM_HEADS,
    QUERY_HISTORY,
    RAW_LIBERO,
    ROOT,
)
from e5.cache import DEFAULT_CACHE_DIR as E5_CACHE_DIR
from e6 import NUM_QUERY_TOKENS, Z_SELF_COND_DIM
from e7 import BEHAVIOR_STRIDE, DEFAULT_BEHAVIOR_MEM_DIR

E9_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E9_ROOT / "outputs"
DEFAULT_DENSE_CACHE_DIR = OUTPUT_ROOT / "dense_clip_cache"
DEFAULT_STAGE1_LOCAL_DIR = OUTPUT_ROOT / "stage1_local"
DEFAULT_STAGE1_LOCAL_CKPT = DEFAULT_STAGE1_LOCAL_DIR / "best.pt"
DEFAULT_STAGE2_DIRECT_DIR = OUTPUT_ROOT / "stage2_direct"
DEFAULT_STAGE2_CACHE_DIR = DEFAULT_STAGE2_DIRECT_DIR / "cache"

LOCAL_WINDOW_LEN = ACTION_HORIZON  # 16 consecutive control steps V_{t:t+16}
NUM_BEHAVIOR_TOKENS = NUM_QUERY_TOKENS  # 4 learnable tokens -> (4, 768) behavior latent
NUM_CAMERAS = len(CAMERAS)  # 2 (agentview_rgb, eye_in_hand_rgb)

__all__ = [
    "ACTION_DIM_10D",
    "ACTION_DIM_7D",
    "ACTION_HORIZON",
    "BEHAVIOR_STRIDE",
    "BPP_CKPT",
    "CAMERAS",
    "DEFAULT_BEHAVIOR_MEM_DIR",
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "DEFAULT_STAGE1_LOCAL_DIR",
    "DEFAULT_STAGE2_CACHE_DIR",
    "DEFAULT_STAGE2_DIRECT_DIR",
    "E5_CACHE_DIR",
    "E9_ROOT",
    "EMBED_DIM",
    "EXEC_HORIZON",
    "LOCAL_WINDOW_LEN",
    "NUM_BEHAVIOR_TOKENS",
    "NUM_CAMERAS",
    "NUM_HEADS",
    "NUM_QUERY_TOKENS",
    "OUTPUT_ROOT",
    "QUERY_HISTORY",
    "RAW_LIBERO",
    "ROOT",
    "Z_SELF_COND_DIM",
]
