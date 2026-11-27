"""E12: Relational Behavior Representation Learning (based on E9 Stage 1).

Explores whether supervising relational structure across demonstrations (global
action similarity, multi-attribute relational similarity, and behavior
displacement consistency) produces a high-fidelity, structured, transfer-friendly
behavior latent space without modifying E9's Encoder, Decoder, latent dimension,
or core diffusion action reconstruction objective.
"""

from pathlib import Path

from e9 import (
    ACTION_DIM_10D,
    ACTION_DIM_7D,
    ACTION_HORIZON,
    DEFAULT_DENSE_CACHE_DIR,
    DEFAULT_STAGE1_LOCAL_CKPT,
    E5_CACHE_DIR,
    EMBED_DIM,
    EXEC_HORIZON,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_CAMERAS,
    NUM_HEADS,
    ROOT,
    Z_SELF_COND_DIM,
)

E12_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E12_ROOT / "outputs"
STAGE1_ANALYSIS_DIR = OUTPUT_ROOT / "stage1_e9_analysis"
SCENE_SENSITIVITY_LATENTS = (
    ROOT / "e9" / "outputs" / "scene_sensitivity" / "latents.npz"
)
P0_BASES_PATH = ROOT / "p0_lowrank" / "outputs" / "e9_p0_bases.pt"
UNSEEN_CACHE_DIR = ROOT / "p0_lowrank" / "outputs" / "unseen_cache"

# Physical GPU 4 & GPU 5 (2x NVIDIA RTX 6000D, 85 GB each) by UUID per AGENTS.md
GPU4_UUID = "GPU-129d1a29-e94e-1067-e547-41ef6ea6df69"
GPU5_UUID = "GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

ATTRIBUTE_NAMES = (
    "global_raw7d",
    "global_balanced7d",
    "global_10d",
    "translation",
    "translation_net_dir",
    "rotation",
    "magnitude",
    "temporal",
    "gripper",
)

__all__ = [
    "ACTION_DIM_10D",
    "ACTION_DIM_7D",
    "ACTION_HORIZON",
    "ATTRIBUTE_NAMES",
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E12_ROOT",
    "E5_CACHE_DIR",
    "EMBED_DIM",
    "EXEC_HORIZON",
    "GPU4_UUID",
    "GPU5_UUID",
    "LOCAL_WINDOW_LEN",
    "NUM_BEHAVIOR_TOKENS",
    "NUM_CAMERAS",
    "NUM_HEADS",
    "OUTPUT_ROOT",
    "P0_BASES_PATH",
    "ROOT",
    "SCENE_SENSITIVITY_LATENTS",
    "STAGE1_ANALYSIS_DIR",
    "UNSEEN_CACHE_DIR",
    "Z_SELF_COND_DIM",
]
