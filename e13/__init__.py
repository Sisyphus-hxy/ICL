"""E13: Transferability Validation for E12 Relational Behavior Representations.

Rigorous validation testing whether E12-B/C's transfer gains stem from genuine
structural improvement of the behavior representation rather than simple linear
reparameterization, privileged GT action leakage, or an action-space copy.
"""

from pathlib import Path

from e9 import DEFAULT_DENSE_CACHE_DIR, DEFAULT_STAGE1_LOCAL_CKPT, E5_CACHE_DIR, ROOT
from e12 import OUTPUT_ROOT as E12_OUTPUT_ROOT

E13_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E13_ROOT / "outputs"

# Physical GPU 2 & GPU 5 (2x empty NVIDIA RTX 6000D, 85 GB each) by UUID per AGENTS.md
GPU2_UUID = "GPU-925a799e-d7db-61b5-8d1e-2e97919fba76"
GPU5_UUID = "GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

CHECKPOINTS = {
    "e9": DEFAULT_STAGE1_LOCAL_CKPT,
    "control": E12_OUTPUT_ROOT / "control" / "last_ema.pt",
    "e12_a": E12_OUTPUT_ROOT / "e12_a" / "last_ema.pt",
    "e12_b": E12_OUTPUT_ROOT / "e12_b" / "last_ema.pt",
    "e12_c": E12_OUTPUT_ROOT / "e12_c" / "last_ema.pt",
}

SEEDS_DEFAULT = (42, 123, 777)

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E12_OUTPUT_ROOT",
    "E13_ROOT",
    "E5_CACHE_DIR",
    "GPU2_UUID",
    "GPU5_UUID",
    "OUTPUT_ROOT",
    "ROOT",
    "SEEDS_DEFAULT",
]
