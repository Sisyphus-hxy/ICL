"""E14: Disentangling Editability and Visual Transferability.

Investigates why E12 relational representations (E12-B, E12-C, E12-BC-FactorDisp)
substantially improve latent editability and composability yet underperform E9 in
strictly zero-leakage Visual ICL, cleanly disentangling:
  - Input-side demo information loss vs Output-side target/decoder bottleneck (Exp 1)
  - Target latent predictability, linear vs nonlinear adapter gap, decoder sensitivity
    to equal-scale perturbations, and E9 action-explainable vs action-unexplained
    subspace causal ablations (Exp 2)
  - Complementarity between E9 contextual information and E12-C/BC operational geometry
    under strict equal-dimension and equal-parameter controls (Exp 3)
"""

from pathlib import Path

from e9 import DEFAULT_DENSE_CACHE_DIR, DEFAULT_STAGE1_LOCAL_CKPT, E5_CACHE_DIR, ROOT
from e12 import OUTPUT_ROOT as E12_OUTPUT_ROOT
from e13 import OUTPUT_ROOT as E13_OUTPUT_ROOT

E14_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E14_ROOT / "outputs"

# Physical GPU 4 & GPU 5 (2x idle NVIDIA RTX 6000D, 85 GB each) by UUID per AGENTS.md
GPU4_UUID = "GPU-129d1a29-e94e-1067-e547-41ef6ea6df69"
GPU5_UUID = "GPU-fbb87876-0bf5-1011-4937-9ee93578f6f5"

CORE_MODELS = ("e9", "e12_b", "e12_c", "e12_bc_factordisp")

MODEL_LABELS = {
    "e9": "E9",
    "e12_b": "E12-B",
    "e12_c": "E12-C",
    "e12_bc_factordisp": "E12-BC-FD",
    "control": "E9-Cont",
    "none": "Vision-Only",
}

CHECKPOINTS = {
    "e9": DEFAULT_STAGE1_LOCAL_CKPT,
    "e12_b": E12_OUTPUT_ROOT / "e12_b" / "last_ema.pt",
    "e12_c": E12_OUTPUT_ROOT / "e12_c" / "last_ema.pt",
    "e12_bc_factordisp": E13_OUTPUT_ROOT / "e12_bc_factordisp" / "last_ema.pt",
    "control": E12_OUTPUT_ROOT / "control" / "last_ema.pt",
}

SEEDS_DEFAULT = (42, 123, 777)
BUDGETS_DEFAULT = (50, 100, 250, 500, 1000, 2000, 4000)
MATCH_MODES_DEFAULT = ("visual_knn", "rel_progress")

__all__ = [
    "BUDGETS_DEFAULT",
    "CHECKPOINTS",
    "CORE_MODELS",
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E12_OUTPUT_ROOT",
    "E13_OUTPUT_ROOT",
    "E14_ROOT",
    "E5_CACHE_DIR",
    "GPU4_UUID",
    "GPU5_UUID",
    "MATCH_MODES_DEFAULT",
    "MODEL_LABELS",
    "OUTPUT_ROOT",
    "ROOT",
    "SEEDS_DEFAULT",
]
