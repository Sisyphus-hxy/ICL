"""E18: Unified Relational Behavior Representation Learning.

Converges E12–E17 findings into a clean, unified pure-video behavior representation
learning method (URBE) and rigorously tests:
  1. Core Method & Strict Component Ablations (Compact Manifold, Sharp Top-K/Top-1 Reuse,
     Operational/Residual factorization, Cycle & Equivariance necessity).
  2. Geometry-Aware Relations (diagnosing 2D optical flow perspective conflict vs
     Self-Supervised Cross-View Canonical Motion vs Independent Motion Subspace vs
     No-Motion, plus data-supported 3D RGB-D Scene Flow on RLBench).
  3. Relation Scaling (holding video count, training steps, and model capacity fixed while
     independently varying donor pool size up to 1024 via FIFO queue, exact cross-demo
     coverage fraction, and behavioral diversity at fixed video count).
  4. External Baselines & Multi-Level Generalization (fair LAPA VQ-LAM & CycleMimic
     comparisons + 4-level generalization strictly separating Encoder-Unseen vs
     Adapter-Unseen across LIBERO unseen suites and RLBench-18).
"""

from pathlib import Path

from e9 import DEFAULT_DENSE_CACHE_DIR, DEFAULT_STAGE1_LOCAL_CKPT, E5_CACHE_DIR, RAW_LIBERO, ROOT
from e16 import MOTION_CACHE_DIR, MOTION_FEAT_DIM, OUTPUT_ROOT as E16_OUTPUT_ROOT
from e17 import OUTPUT_ROOT as E17_OUTPUT_ROOT

E18_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E18_ROOT / "outputs"
GEN_CACHE_DIR = OUTPUT_ROOT / "generalization_cache"
CV_GEO_DIR = OUTPUT_ROOT / "cv_geo_encoder"

# Physical GPU 0 & GPU 1 (2x idle NVIDIA RTX A6000, 48 GB each) by UUID per user request & AGENTS.md
GPU0_UUID = "GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID = "GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

RLBENCH_ARCHIVE_ROOT = ROOT / "datasets" / "rlbench" / "peract" / "archives"

# Verified E16 & E17 full-budget (32,184 steps) reference checkpoints
REFERENCE_PRETRAIN_CKPTS = {
    "e16_d1_lam": E16_OUTPUT_ROOT / "e16_d1_lam" / "last_ema.pt",
    "e16_d3_vid_op": E16_OUTPUT_ROOT / "e16_d3_vid_op" / "last_ema.pt",
    "e16_d4_learn_op": E16_OUTPUT_ROOT / "e16_d4_learn_op" / "last_ema.pt",
    "e17_m2_compact_op": E17_OUTPUT_ROOT / "e17_m2_compact_op" / "last_ema.pt",
    "e17_m3_dual_orth": E17_OUTPUT_ROOT / "e17_m3_dual_orth" / "last_ema.pt",
    "e17_m4_compact_reuse": E17_OUTPUT_ROOT / "e17_m4_compact_reuse" / "last_ema.pt",
    "oracle_e9": DEFAULT_STAGE1_LOCAL_CKPT,
}

REFERENCE_DECODER_CKPTS = {
    "e16_d0_rand": E16_OUTPUT_ROOT / "e16_d0_rand_dec100" / "last_ema.pt",
    "e16_d1_lam": E16_OUTPUT_ROOT / "e16_d1_lam_dec100" / "last_ema.pt",
    "e16_d3_vid_op": E16_OUTPUT_ROOT / "e16_d3_vid_op_dec100" / "last_ema.pt",
    "e16_d4_learn_op": E16_OUTPUT_ROOT / "e16_d4_learn_op_dec100" / "last_ema.pt",
    "e17_m2_compact_op": E17_OUTPUT_ROOT / "e17_m2_compact_op_dec100" / "last_ema.pt",
    "e17_m3_dual_orth": E17_OUTPUT_ROOT / "e17_m3_dual_orth_dec100" / "last_ema.pt",
    "e17_m4_compact_reuse": E17_OUTPUT_ROOT / "e17_m4_compact_reuse_dec100" / "last_ema.pt",
    "oracle_e9": E16_OUTPUT_ROOT / "e16_d_oracle_e9_dec100" / "last_ema.pt",
}

SEEDS_DEFAULT = (42, 123, 777)

__all__ = [
    "CV_GEO_DIR",
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E16_OUTPUT_ROOT",
    "E17_OUTPUT_ROOT",
    "E18_ROOT",
    "E5_CACHE_DIR",
    "GEN_CACHE_DIR",
    "GPU0_UUID",
    "GPU1_UUID",
    "MOTION_CACHE_DIR",
    "MOTION_FEAT_DIM",
    "OUTPUT_ROOT",
    "RAW_LIBERO",
    "REFERENCE_DECODER_CKPTS",
    "REFERENCE_PRETRAIN_CKPTS",
    "RLBENCH_ARCHIVE_ROOT",
    "ROOT",
    "SEEDS_DEFAULT",
]
