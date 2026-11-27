"""E10 (P1): Learned Low-Rank Behavior Transfer over Frozen E9 Encoder, Reader & Action Decoder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# Physical GPU 0 & GPU 1 (2x NVIDIA RTX A6000, 48 GB each) per AGENTS.md
GPU0_UUID = "GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID = "GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

# Frozen E9 checkpoints & caches
E9_STAGE1_CKPT = ROOT / "e9/outputs/stage1_local/best.pt"
E9_STAGE2_CKPT = ROOT / "e9/outputs/stage2_direct/train/best.pt"
E9_STAGE2_CACHE_DIR = ROOT / "e9/outputs/stage2_direct/cache"
E5_CACHE_DIR = ROOT / "e5/outputs/cache"

# Fixed train-fitted P0 SVD bases (exported by p0_lowrank/p0_5_demo_preservation.py)
P0_BASES_PATH = ROOT / "p0_lowrank/outputs/e9_p0_bases.pt"

LATENT_TOKENS = 4
EMBED_DIM = 768
LATENT_DIM = LATENT_TOKENS * EMBED_DIM  # 3072
NUM_HEADS = 8
BEHAVIOR_STRIDE = 16
ACTION_HORIZON = 16
EXEC_HORIZON = 12
ACTION_DIM_10D = 10

VARIANTS = {
    "e10_a": {"rank_k": 16, "transfer_mode": "residual", "label": "E10-A (Fixed Basis, K=16)"},
    "e10_b": {"rank_k": 32, "transfer_mode": "residual", "label": "E10-B (Fixed Basis, K=32)"},
    "e10_c": {"rank_k": 64, "transfer_mode": "residual", "label": "E10-C (Fixed Basis, K=64)"},
    "control": {"rank_k": 32, "transfer_mode": "control", "label": "Control (No Skip z_mean, K=32)"},
}
