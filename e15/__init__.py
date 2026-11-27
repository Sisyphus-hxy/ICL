"""E15: Robust Relational Behavior Representation Learning.

Investigates whether structured relational behavior geometry (Editability & Composability)
and generative/inference robustness (Decoder Robustness & Functional Residual Preservation)
are in fundamental conflict, or whether their trade-off can be broken without increasing
latent dimension, network capacity, or training budget.

Core modules:
  - `exp1_three_component.py`: Training-free strict 3-component decomposition
    (`z_c = z_lin + z_coax + z_perp`) + Ridge cross-term & SVD truncation error audit
    on E9, E12-C, and E12-BC-FD across Reconstruction, Motion Details, Editability,
    Decoder Robustness, and Visual ICL (strictly separating Oracle vs Zero-Leakage).
  - `residual_noise.py`: Strict train-only 5-fold task-grouped CV Adapter residual bank
    constructor + exact per-sample scale-matched isotropic random noise control.
  - `train_e15.py`: Matched-budget (`5,364` steps, `batch=256`, `lr=1e-4`, `EMA=0.999`,
    identical `4x768=3072D` architecture) trainer for:
      * E15-A: Frozen Encoder + Noise-Robust Decoder (`e12_c` & fair `e9` baseline, `real_cv` & `rand_matched`)
      * E15-B: Joint Encoder + Decoder with Relation Loss + Latent Robustness (`e12_c` & fair `e9`, `real_cv` & `rand_matched`)
      * E15-C: Operational-Subspace-Restricted Relation + Residual Subspace Preservation (`op_only` & `op_decorr`)
      * E15-Combo (Step 5): Evidence-driven combinations determined strictly after evaluating A/B/C.
  - `eval_e15.py`: Unified E12-E14 evaluation suite (Fidelity, Editability, Composability,
    Predictability, Decoder Robustness, plus Zero-Leakage Visual ICL downstream verification).
  - `report_e15.py`: Publication-grade figures, CSV summaries, and `e15/outputs/REPORT.md`.
"""

from pathlib import Path

from e9 import DEFAULT_DENSE_CACHE_DIR, DEFAULT_STAGE1_LOCAL_CKPT, E5_CACHE_DIR, ROOT
from e12 import OUTPUT_ROOT as E12_OUTPUT_ROOT
from e13 import OUTPUT_ROOT as E13_OUTPUT_ROOT
from e14 import OUTPUT_ROOT as E14_OUTPUT_ROOT

E15_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E15_ROOT / "outputs"

# Strictly physical GPU 0 & GPU 1 (2x idle NVIDIA RTX A6000, 48 GB each) per user request & AGENTS.md
GPU0_UUID = "GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"
GPU1_UUID = "GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272"

BASE_CHECKPOINTS = {
    "e9": DEFAULT_STAGE1_LOCAL_CKPT,
    "control": E12_OUTPUT_ROOT / "control" / "last_ema.pt",
    "e12_b": E12_OUTPUT_ROOT / "e12_b" / "last_ema.pt",
    "e12_c": E12_OUTPUT_ROOT / "e12_c" / "last_ema.pt",
    "e12_bc_factordisp": E13_OUTPUT_ROOT / "e12_bc_factordisp" / "last_ema.pt",
}

SEEDS_DEFAULT = (42, 123, 777)
MATCH_MODES_DEFAULT = ("visual_knn", "rel_progress")

__all__ = [
    "BASE_CHECKPOINTS",
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E12_OUTPUT_ROOT",
    "E13_OUTPUT_ROOT",
    "E14_OUTPUT_ROOT",
    "E15_ROOT",
    "E5_CACHE_DIR",
    "GPU0_UUID",
    "GPU1_UUID",
    "MATCH_MODES_DEFAULT",
    "OUTPUT_ROOT",
    "ROOT",
    "SEEDS_DEFAULT",
]
