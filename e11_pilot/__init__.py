"""E11-Pilot: Action-Guided Minimal Behavior Adaptation over Frozen E9 & P0 Basis U_64.

Fixed frozen components (NEVER modified or trained):
  - E9 Stage 1 Local Behavior Encoder & Action Decoder (`e9/outputs/stage1_local/best.pt`)
  - E9 Stage 2 Direct Reader (`e9/outputs/stage2_direct/train/best.pt`)
  - P0 train-fitted orthonormal basis `U_64` (`p0_lowrank/outputs/e9_p0_bases.pt["U_reader"][:, :64]`)
  - E10-C reference learned transfer (`e10/outputs/e10_c/train/best.pt`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Physical GPU 4 & GPU 5 (2x NVIDIA RTX 6000D, 85 GB, sm_120) per AGENTS.md & user specification
GPU4_UUID = "GPU-129d1a29-e94e-1067-e547-41ef6ea6df69"
GPU5_UUID = "GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

# Frozen checkpoints, caches, and basis
E9_STAGE1_CKPT = ROOT / "e9/outputs/stage1_local/best.pt"
E9_STAGE2_CKPT = ROOT / "e9/outputs/stage2_direct/train/best.pt"
E9_STAGE2_CACHE_DIR = ROOT / "e9/outputs/stage2_direct/cache"
E5_CACHE_DIR = ROOT / "e5/outputs/cache"
P0_BASES_PATH = ROOT / "p0_lowrank/outputs/e9_p0_bases.pt"
P0_UNSEEN_CACHE_DIR = ROOT / "p0_lowrank/outputs/unseen_cache"
E10_C_CKPT = ROOT / "e10/outputs/e10_c/train/best.pt"

RANK_K = 64
LATENT_TOKENS = 4
EMBED_DIM = 768
LATENT_DIM = LATENT_TOKENS * EMBED_DIM  # 3072
BEHAVIOR_STRIDE = 16
ACTION_HORIZON = 16
EXEC_HORIZON = 12
ACTION_DIM_10D = 10

SEMANTIC_PHASES: Tuple[str, ...] = (
    "approach",
    "grasp_transition",
    "transport_carry",
    "release_transition",
    "post_release_transit",
)

DEFAULT_LAMBDA_GRID: Tuple[float, ...] = (
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
)
