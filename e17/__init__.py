"""E17: Discovering and Validating Reusable Behavior Structure.

Deep investigation built on E16-D3/D4 examining:
  1. Relation Authenticity: Auditing LearnReuse's relation graph W_ij and conditional
     forward predictor F_psi against Vision-only, Zero-donor, Mean-donor, Random-donor,
     Wrong-donor, and Nearest-Neighbor controls + counterfactual interventions.
  2. Emergent Behavior Factors: Unsupervised discovery, optimal Hungarian attribute matching,
     all-pairs Token Swap, unsupervised PCA/ICA subspace interventions, z_op vs z_res
     causal disentanglement, and temporal autocorrelation / changepoint analysis.
  3. D3/D4 Complementarity: Geometric, spectral, information-distribution, and Decoder
     sensitivity analysis + parameter- and budget-matched pure-video complementary training.
  4. Controlled Scaling, Coverage & Unified Decoder Protocol: Reconciling E16 evaluation
     discrepancies, strictly controlling training steps vs unlabeled video scale vs relation
     coverage, separating Pure Video vs Task-Metadata-Assisted regimes and Oracle vs
     Zero-Leakage evaluations, with per-task paired statistics and 95% bootstrap CIs.
"""

from pathlib import Path

from e9 import DEFAULT_DENSE_CACHE_DIR, DEFAULT_STAGE1_LOCAL_CKPT, E5_CACHE_DIR, ROOT
from e16 import (
    GPU0_UUID,
    GPU1_UUID,
    MATCH_MODES_DEFAULT,
    MOTION_CACHE_DIR,
    MOTION_FEAT_DIM,
    OUTPUT_ROOT as E16_OUTPUT_ROOT,
    REFERENCE_CHECKPOINTS,
    REFERENCE_EVALS,
    SEEDS_DEFAULT,
)

E17_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = E17_ROOT / "outputs"

# Existing E16 checkpoints reused as baselines (strictly read-only)
E16_PRETRAIN_CKPTS = {
    "e16_d1_lam": E16_OUTPUT_ROOT / "e16_d1_lam" / "last_ema.pt",
    "e16_d2_rand_rel": E16_OUTPUT_ROOT / "e16_d2_rand_rel" / "last_ema.pt",
    "e16_d3_vid_op": E16_OUTPUT_ROOT / "e16_d3_vid_op" / "last_ema.pt",
    "e16_d4_learn_op": E16_OUTPUT_ROOT / "e16_d4_learn_op" / "last_ema.pt",
    "e16_d3_scale10": E16_OUTPUT_ROOT / "e16_d3_scale10" / "last_ema.pt",
    "e16_d3_scale30": E16_OUTPUT_ROOT / "e16_d3_scale30" / "last_ema.pt",
}

E16_DECODER_CKPTS = {
    "e16_d0_rand_dec01": E16_OUTPUT_ROOT / "e16_d0_rand_dec01",
    "e16_d0_rand_dec10": E16_OUTPUT_ROOT / "e16_d0_rand_dec10",
    "e16_d0_rand_dec100": E16_OUTPUT_ROOT / "e16_d0_rand_dec100",
    "e16_d1_lam_dec01": E16_OUTPUT_ROOT / "e16_d1_lam_dec01",
    "e16_d1_lam_dec10": E16_OUTPUT_ROOT / "e16_d1_lam_dec10",
    "e16_d1_lam_dec100": E16_OUTPUT_ROOT / "e16_d1_lam_dec100",
    "e16_d2_randrel_dec100": E16_OUTPUT_ROOT / "e16_d2_randrel_dec100",
    "e16_d3_vidop_dec01": E16_OUTPUT_ROOT / "e16_d3_vidop_dec01",
    "e16_d3_vidop_dec10": E16_OUTPUT_ROOT / "e16_d3_vidop_dec10",
    "e16_d3_vidop_dec100": E16_OUTPUT_ROOT / "e16_d3_vidop_dec100",
    "e16_d4_learnop_dec01": E16_OUTPUT_ROOT / "e16_d4_learnop_dec01",
    "e16_d4_learnop_dec10": E16_OUTPUT_ROOT / "e16_d4_learnop_dec10",
    "e16_d4_learnop_dec100": E16_OUTPUT_ROOT / "e16_d4_learnop_dec100",
    "e16_d_oracle_e9_dec01": E16_OUTPUT_ROOT / "e16_d_oracle_e9_dec01",
    "e16_d_oracle_e9_dec10": E16_OUTPUT_ROOT / "e16_d_oracle_e9_dec10",
    "e16_d_oracle_e9_dec100": E16_OUTPUT_ROOT / "e16_d_oracle_e9_dec100",
    "e16_d3_scale10_dec100": E16_OUTPUT_ROOT / "e16_d3_scale10_dec100",
    "e16_d3_scale30_dec100": E16_OUTPUT_ROOT / "e16_d3_scale30_dec100",
}

E16_REGIME1_LEARN_CKPTS = {
    "e16_b1_learnable_reuse": E16_OUTPUT_ROOT / "e16_b1_learnable_reuse" / "last_ema.pt",
    "e16_b2_learnable_tok": E16_OUTPUT_ROOT / "e16_b2_learnable_tok" / "last_ema.pt",
    "e16_c2_op_learnable": E16_OUTPUT_ROOT / "e16_c2_op_learnable" / "last_ema.pt",
    "e16_c3_op_learnable_tok": E16_OUTPUT_ROOT / "e16_c3_op_learnable_tok" / "last_ema.pt",
    "e16_cov_sametask": E16_OUTPUT_ROOT / "e16_cov_sametask" / "last_ema.pt",
    "e16_cov_unmasked": E16_OUTPUT_ROOT / "e16_cov_unmasked" / "last_ema.pt",
    "e16_cov_strict_ct": E16_OUTPUT_ROOT / "e16_cov_strict_ct" / "last_ema.pt",
}

__all__ = [
    "DEFAULT_DENSE_CACHE_DIR",
    "DEFAULT_STAGE1_LOCAL_CKPT",
    "E16_DECODER_CKPTS",
    "E16_OUTPUT_ROOT",
    "E16_PRETRAIN_CKPTS",
    "E16_REGIME1_LEARN_CKPTS",
    "E17_ROOT",
    "E5_CACHE_DIR",
    "GPU0_UUID",
    "GPU1_UUID",
    "MATCH_MODES_DEFAULT",
    "MOTION_CACHE_DIR",
    "MOTION_FEAT_DIM",
    "OUTPUT_ROOT",
    "REFERENCE_CHECKPOINTS",
    "REFERENCE_EVALS",
    "ROOT",
    "SEEDS_DEFAULT",
]
