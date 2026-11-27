"""E16: Self-supervised Relational Behavior Representation Learning.

Investigates whether shared cross-demo behavioral structure can be discovered purely
from large-scale video without human-defined action attributes or GT action relation
metrics, while achieving Editability, Composability, Predictability, and Robustness.

Strictly separates two experimental regimes:
  - Regime 1 (Video-only Relation Supervision: E16-A / E16-B / E16-C):
      Replaces E12/E15's GT action relation supervision with purely video-derived
      relations (E16-A), learnable reuse-driven cross-demo relations (E16-B), and
      operational-subspace-restricted self-supervised relations preserving functional
      z_res (E16-C), under identical 1.0-epoch (5,364 steps) matched budget alongside
      No-Relation (E9/E9-Cont), Random-Relation, and GT-Action-Relation baselines.
  - Regime 2 (Pure Video-only Pretraining: E16-D):
      Completely removes ALL action supervision and action-supervised initialization
      during pretraining (from-scratch LocalBehaviorEncoder on 1.373M video clips),
      then freezes the encoder and trains a unified Diffusion Action Decoder across
      1%, 10%, and 100% action labels, alongside data-scale and relation-coverage
      ablations.
"""

from pathlib import Path

from e9 import DEFAULT_DENSE_CACHE_DIR, DEFAULT_STAGE1_LOCAL_CKPT, E5_CACHE_DIR, RAW_LIBERO, ROOT
from e12 import OUTPUT_ROOT as E12_OUTPUT_ROOT
from e13 import OUTPUT_ROOT as E13_OUTPUT_ROOT
from e15 import OUTPUT_ROOT as E15_OUTPUT_ROOT

E16_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E16_ROOT / "outputs"
MOTION_CACHE_DIR = OUTPUT_ROOT / "video_motion_cache"

# Physical GPU 0 & GPU 1 (2x idle NVIDIA RTX A6000, 48 GB each) by UUID per user request & AGENTS.md
GPU0_UUID = "GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID = "GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

# Existing E9/E12/E15 checkpoints & evaluations reused for fair apples-to-apples comparison
REFERENCE_CHECKPOINTS = {
    "e9": DEFAULT_STAGE1_LOCAL_CKPT,
    "control": E12_OUTPUT_ROOT / "control" / "last_ema.pt",
    "e12_c": E12_OUTPUT_ROOT / "e12_c" / "last_ema.pt",
    "e12_bc_factordisp": E13_OUTPUT_ROOT / "e12_bc_factordisp" / "last_ema.pt",
    "e15_c_op_only": E15_OUTPUT_ROOT / "e15_c_op_only" / "last_ema.pt",
    "e15_d_op_mild_cv": E15_OUTPUT_ROOT / "e15_d_op_mild_cv" / "last_ema.pt",
    "e15_d_op_bcfd": E15_OUTPUT_ROOT / "e15_d_op_bcfd" / "last_ema.pt",
}

REFERENCE_EVALS = {
    k: E15_OUTPUT_ROOT / "eval" / k / "eval_e15.json"
    for k in REFERENCE_CHECKPOINTS
}

MOTION_FEAT_DIM = 64  # Per-camera pixel-space optical flow + grid point-track descriptor
SEEDS_DEFAULT = (42, 123, 777)
MATCH_MODES_DEFAULT = ("visual_knn", "rel_progress")

__all__ = [
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E12_OUTPUT_ROOT",
    "E13_OUTPUT_ROOT",
    "E15_OUTPUT_ROOT",
    "E16_ROOT",
    "E5_CACHE_DIR",
    "GPU0_UUID",
    "GPU1_UUID",
    "MATCH_MODES_DEFAULT",
    "MOTION_CACHE_DIR",
    "MOTION_FEAT_DIM",
    "OUTPUT_ROOT",
    "RAW_LIBERO",
    "REFERENCE_CHECKPOINTS",
    "REFERENCE_EVALS",
    "ROOT",
    "SEEDS_DEFAULT",
]
