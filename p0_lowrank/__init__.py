"""P0: Low-dimensional Behavior Transfer Diagnosis for frozen E7 and E9 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Physical GPU 0 (NVIDIA RTX A6000, 48 GB) per AGENTS.md
GPU0_UUID = "GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID = "GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

# Frozen checkpoints (NO retraining, NO architecture modification)
E7_STAGE1_CKPT = ROOT / "e6/outputs/stage1/stage1_self_latent/best.pt"
E7_STAGE2_CKPT = ROOT / "e7/outputs/direct_latent/train/e7_direct_latent/best.pt"
E7_BEHAVIOR_MEM_DIR = ROOT / "e7/outputs/behavior_memory_cache"

E9_STAGE1_CKPT = ROOT / "e9/outputs/stage1_local/best.pt"
E9_STAGE2_CKPT = ROOT / "e9/outputs/stage2_direct/train/best.pt"
E9_BEHAVIOR_MEM_DIR = ROOT / "e9/outputs/stage2_direct/cache"

E5_CACHE_DIR = ROOT / "e5/outputs/cache"

# Latent & action constants
LATENT_TOKENS = 4
EMBED_DIM = 768
LATENT_DIM = LATENT_TOKENS * EMBED_DIM  # 3072
BEHAVIOR_STRIDE = 16
ACTION_HORIZON = 16
EXEC_HORIZON = 12
ACTION_DIM_10D = 10

# P0 evaluation ranks (Full = 3072)
RANK_GRID: Tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256, LATENT_DIM)
NONTRIVIAL_RANKS: Tuple[int, ...] = (8, 16, 32, 64, 128, 256)

ACTION_10D_NAMES: Tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "pos_z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
)
