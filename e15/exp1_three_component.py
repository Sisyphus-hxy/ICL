"""E15 Part 1 (Training-Free): Strict Three-Component Ablation + Ridge Cross-Term & SVD Truncation Audit.

For E9, E12-C, and E12-BC-FD, decomposes centered behavior latent `z_c = z - mu` into THREE
exact components fitted STRICTLY on the training split:
    z_c  =  z_lin  +  z_coax  +  z_perp

where:
  1. `z_lin = W_az @ a_c`         : Action-linear predicted component (from Ridge `a_10d -> z_c`).
  2. `z_coax = P_Q(z_c - z_lin)`  : Co-axial residual INSIDE the exact action subspace `span(Q_full)`
                                    (nonlinear action details, Ridge shrinkage residual, and scene
                                     context aligned with action subspace axes).
  3. `z_perp = (I - P_Q) z_c`     : Orthogonal residual strictly perpendicular to `span(Q_full)`
                                    (`<z_lin, z_perp> == 0` and `<z_coax, z_perp> == 0` to ~1e-12!).

Note the algebraic identity:
    ||z_c||^2 = ||z_lin||^2 + ||z_coax||^2 + 2<z_lin, z_coax> + ||z_perp||^2
where `2<z_lin, z_coax>` is the **Ridge cross-term** (zero on train ONLY for OLS alpha=0,
strictly non-zero under Ridge regularization alpha>0 and on held-out unseen tasks!), AND
truncating `Q_act` at 99.5% SVD variance (as in E14) introduces an explicit **SVD truncation
leakage** `||(I - P_{Q_0.995}) z_lin||^2` that we quantify directly across thresholds
`{0.990, 0.995, 0.999, 1.000}` and Ridge `alpha in {0(OLS), 1e-3, 1e-2, 1e-1, 1, 10, 100}`.

Evaluates each component's causal contribution across FOUR axes (strictly separating
Oracle GT Diagnostics vs Zero-Leakage Inference):
  A. Reconstruction & Fine Motion Details (vel/accel/jerk MSE, grip transition acc, high-motion decile)
  B. Editability (Oracle Tier-2 Latent Arithmetic selective component transport & ablation)
  C. Decoder Robustness (equal-scale perturbations along `z_lin`, `z_coax`, `z_perp` & actual ICL error split)
  D. Predictability & Visual ICL Role (probing content of each component + Zero-Leakage vs Oracle Visual ICL)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _effective_rank_stats,
    _ridge_fit_predict,
    _solve_ridge_gpu,
    encode_all_val_starts,
)
from e13.exp1_training_free import _build_fixed_analogy_quadruples
from e13.exp2_action_baselines import _gather_future_actions
from e14.common import (
    build_block_rms_features,
    build_seed_bundle,
    ddim_decode_fixed_noise,
    fit_predict_scaled_ridge,
)

from . import BASE_CHECKPOINTS, OUTPUT_ROOT, SEEDS_DEFAULT

PART1_MODELS = ("e9", "e12_c", "e12_bc_factordisp")


# ---------------------------------------------------------------------------
# Exact Three-Component Decomposition + Ridge Cross-Term & SVD Truncation Audit
# ---------------------------------------------------------------------------

def fit_three_component_model(
    a10_tr: np.ndarray,
    z_tr: np.ndarray,
    device: torch.device,
    *,
    alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    forced_alpha: float | None = None,
    svd_threshold: float = 1.0,
    sv_rtol: float = 1e-7,
) -> Dict[str, Any]:
    """Fit `a_10d -> z_c` Ridge/OLS weight matrix `W_az` and orthonormal basis `Q_act` on TRAIN ONLY.

    When `svd_threshold == 1.0`, `Q_act` spans the FULL numerical range of `Z_lin_tr = A_c @ W_az`
    (all singular values `s_i > sv_rtol * s_max`), guaranteeing `z_lin in span(Q_act)` to machine
    precision (`~1e-12`). When `svd_threshold < 1.0` (e.g. `0.995` as in E14), `Q_act` is
    truncated so we can directly measure SVD truncation leakage!
    """
    A_tr = torch.from_numpy(a10_tr.reshape(len(a10_tr), -1).astype(np.float64)).to(device)
    Z_tr = torch.from_numpy(z_tr.astype(np.float64)).to(device)
    a_mu = A_tr.mean(dim=0, keepdim=True)
    z_mu = Z_tr.mean(dim=0, keepdim=True)
    Ac = A_tr - a_mu
    Zc = Z_tr - z_mu

    n, d = Ac.shape
    if forced_alpha is not None:
        best_alpha = float(forced_alpha)
    else:
        # Internal 80/20 holdout strictly inside training set
        n_val = max(10, int(0.2 * n))
        A_a, Z_a = Ac[:-n_val], Zc[:-n_val]
        A_v, Z_v = Ac[-n_val:], Zc[-n_val:]
        G_base = A_a.T @ A_a
        AtZ = A_a.T @ Z_a
        best_alpha = alphas[0]
        best_err = float("inf")
        for al in alphas:
            G = G_base.clone()
            G.diagonal().add_(al)
            W_try = torch.linalg.solve(G, AtZ)
            err = float(((A_v @ W_try - Z_v) ** 2).mean().item())
            if err < best_err:
                best_err = err
                best_alpha = float(al)

    if best_alpha == 0.0:
        # Exact minimum-norm OLS via lstsq / SVD pseudoinverse
        W_az = torch.linalg.lstsq(Ac, Zc).solution
    else:
        G = Ac.T @ Ac
        G.diagonal().add_(best_alpha)
        W_az = torch.linalg.solve(G, Ac.T @ Zc)  # (160, 3072)

    # Fast exact rank-<=160 SVD of Z_lin_tr = Ac @ W_az using reduced QR of Ac (N, 160)!
    # Since Ac = Q_a @ R_a (N x d, d x d), Z_lin_tr = Q_a @ (R_a @ W_az) where (R_a @ W_az) is ONLY (160, 3072)!
    # SVD of (160, 3072) is 100% mathematically identical to SVD of (4000, 3072) and 50x faster!
    _, R_a = torch.linalg.qr(Ac, mode="reduced")  # (160, 160)
    RW = R_a @ W_az  # (160, 3072)
    _, S, Vh = torch.linalg.svd(RW, full_matrices=False)
    s_np = S.cpu().numpy()
    eig = s_np ** 2
    tot = float(eig.sum())
    cum = np.cumsum(eig) / max(tot, 1e-15)

    r_num = int(np.sum(s_np > sv_rtol * max(float(s_np[0]), 1e-15)))
    if svd_threshold >= 1.0:
        r_eff = max(1, min(r_num, Vh.shape[0]))
    else:
        r_eff = int(np.searchsorted(cum, svd_threshold) + 1)
        r_eff = max(1, min(r_eff, r_num, Vh.shape[0]))

    Q_act = Vh[:r_eff].T.cpu().numpy()  # (3072, r_eff) float64
    return {
        "a_mu": a_mu.cpu().numpy(),
        "z_mu": z_mu.cpu().numpy(),
        "W_az": W_az.cpu().numpy(),
        "Q_act": Q_act,
        "Vh_full": Vh.cpu().numpy(),
        "alpha": float(best_alpha),
        "rank": int(r_eff),
        "rank_full_numerical": int(r_num),
        "singular_values": s_np,
        "retained_lin_var_frac_train": float(eig[:r_eff].sum() / max(tot, 1e-15)),
    }


def decompose_three_components(
    a10: np.ndarray,
    z: np.ndarray,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    """Decompose `z - z_mu` into `(z_lin, z_coax, z_perp)` plus exact energy & orthogonality audit.

    Exact decomposition:
      - `z_c        = z - z_mu`
      - `z_lin_raw  = (a - a_mu) @ W_az`
      - `z_par      = (z_c @ Q) @ Q.T`
      - `z_perp     = z_c - z_par`
      - `z_lin_proj = (z_lin_raw @ Q) @ Q.T`
      - `z_lin_trunc = z_lin_raw - z_lin_proj`   (== 0 to ~1e-12 when Q is exact full-rank!)
      - `z_coax     = z_par - z_lin_proj`        (exact co-axial residual inside span(Q)!)

    Identity check (exact for ANY Q):
      `z_c == z_lin_proj + z_coax + z_perp`
      `||z_c||^2 == ||z_lin_proj||^2 + ||z_coax||^2 + 2<z_lin_proj, z_coax> + ||z_perp||^2`
    """
    A = a10.reshape(len(a10), -1).astype(np.float64)
    Z = z.astype(np.float64)
    a_mu = model["a_mu"]
    z_mu = model["z_mu"]
    W_az = model["W_az"]
    Q = model["Q_act"]

    z_c = Z - z_mu
    z_lin_raw = (A - a_mu) @ W_az
    z_par = (z_c @ Q) @ Q.T
    z_perp = z_c - z_par
    z_lin_proj = (z_lin_raw @ Q) @ Q.T
    z_lin_trunc = z_lin_raw - z_lin_proj
    z_coax = z_par - z_lin_proj

    # Exact Frobenius energy breakdown (mean squared L2 norm per sample)
    e_tot = float(np.mean(np.sum(z_c ** 2, axis=-1)))
    e_lin_raw = float(np.mean(np.sum(z_lin_raw ** 2, axis=-1)))
    e_lin = float(np.mean(np.sum(z_lin_proj ** 2, axis=-1)))
    e_lin_trunc = float(np.mean(np.sum(z_lin_trunc ** 2, axis=-1)))
    e_coax = float(np.mean(np.sum(z_coax ** 2, axis=-1)))
    e_par = float(np.mean(np.sum(z_par ** 2, axis=-1)))
    e_perp = float(np.mean(np.sum(z_perp ** 2, axis=-1)))
    cross_lin_coax = float(2.0 * np.mean(np.sum(z_lin_proj * z_coax, axis=-1)))
    cross_lin_perp = float(2.0 * np.mean(np.sum(z_lin_raw * z_perp, axis=-1)))
    cross_par_perp = float(2.0 * np.mean(np.sum(z_par * z_perp, axis=-1)))
    recon_err = float(np.max(np.abs(z_c - (z_lin_proj + z_coax + z_perp))))
    # Standard R^2(a->z) = 1 - ||z_c - z_lin_raw||^2 / ||z_c||^2
    mse_reg_res = float(np.mean(np.sum((z_c - z_lin_raw) ** 2, axis=-1)))
    r2_a_to_z = float(1.0 - mse_reg_res / max(e_tot, 1e-12))

    # Mean cosine between z_lin and z_coax
    n_l = np.linalg.norm(z_lin_proj, axis=-1)
    n_c = np.linalg.norm(z_coax, axis=-1)
    v_m = (n_l > 1e-8) & (n_c > 1e-8)
    cos_lin_coax = float(np.mean(np.sum(z_lin_proj[v_m] * z_coax[v_m], axis=-1) / (n_l[v_m] * n_c[v_m]))) if v_m.any() else 0.0

    stats = {
        "e_tot": e_tot,
        "frac_lin_raw": float(e_lin_raw / max(e_tot, 1e-12)),
        "frac_lin": float(e_lin / max(e_tot, 1e-12)),
        "frac_lin_trunc_svd_err": float(e_lin_trunc / max(e_tot, 1e-12)),
        "frac_lin_trunc_of_lin": float(e_lin_trunc / max(e_lin_raw, 1e-12)),
        "frac_coax": float(e_coax / max(e_tot, 1e-12)),
        "frac_cross_2_lin_coax": float(cross_lin_coax / max(e_tot, 1e-12)),
        "frac_par": float(e_par / max(e_tot, 1e-12)),
        "frac_perp": float(e_perp / max(e_tot, 1e-12)),
        "frac_sum_check": float((e_lin + e_coax + cross_lin_coax + e_perp) / max(e_tot, 1e-12)),
        "frac_cross_lin_raw_perp": float(cross_lin_perp / max(e_tot, 1e-12)),
        "frac_cross_par_perp": float(cross_par_perp / max(e_tot, 1e-12)),
        "cos_lin_coax": cos_lin_coax,
        "r2_a_to_z": r2_a_to_z,
        "reg_residual_frac": float(mse_reg_res / max(e_tot, 1e-12)),
        "max_abs_identity_err": recon_err,
    }
    return {
        "z_mu": z_mu.astype(np.float32),
        "z_c": z_c.astype(np.float32),
        "z_lin": z_lin_proj.astype(np.float32),
        "z_lin_raw": z_lin_raw.astype(np.float32),
        "z_lin_trunc": z_lin_trunc.astype(np.float32),
        "z_coax": z_coax.astype(np.float32),
        "z_par": z_par.astype(np.float32),
        "z_perp": z_perp.astype(np.float32),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Fine-Grained Motion Detail Metrics
# ---------------------------------------------------------------------------

def compute_motion_detail_metrics(pred_a10: np.ndarray, gt_a10: np.ndarray) -> Dict[str, float]:
    """Action accuracy + fine-grained kinematic/motion detail metrics (vel, accel, jerk, grip switch)."""
    base = _action_metrics_dict(pred_a10, gt_a10)
    p7 = action_10d_to_7d(pred_a10.astype(np.float32))
    g7 = action_10d_to_7d(gt_a10.astype(np.float32))

    # Temporal first/second differences on 3D delta-pos & 3D rot (kinematic smoothness & high-freq details!)
    p_pos, g_pos = p7[:, :, :3], g7[:, :, :3]
    p_rot, g_rot = p7[:, :, 3:6], g7[:, :, 3:6]
    # Note: p_pos itself is delta-pos (velocity); first diff is acceleration; second diff is jerk!
    accel_pos_mse = float(np.mean((np.diff(p_pos, n=1, axis=1) - np.diff(g_pos, n=1, axis=1)) ** 2))
    jerk_pos_mse = float(np.mean((np.diff(p_pos, n=2, axis=1) - np.diff(g_pos, n=2, axis=1)) ** 2))
    accel_rot_mse = float(np.mean((np.diff(p_rot, n=1, axis=1) - np.diff(g_rot, n=1, axis=1)) ** 2))

    # Path length error (magnitude detail)
    p_spd = np.linalg.norm(p_pos, axis=-1)  # (N, 16)
    g_spd = np.linalg.norm(g_pos, axis=-1)
    path_mae = float(np.mean(np.abs(p_spd.sum(axis=-1) - g_spd.sum(axis=-1))))
    spd_env_mse = float(np.mean((p_spd - g_spd) ** 2))

    # High-motion top-25% clips MSE10
    clip_motion = g_spd.sum(axis=-1)
    hi_mask = clip_motion >= np.quantile(clip_motion, 0.75)
    per_sample_mse10 = np.mean((pred_a10 - gt_a10) ** 2, axis=(1, 2))
    hi_motion_mse10 = float(np.mean(per_sample_mse10[hi_mask]))

    # Gripper steady vs switching transition breakdown
    g_grip = g7[:, :, 6]
    p_grip = p7[:, :, 6]
    is_switch = ~((g_grip < 0).all(axis=1) | (g_grip > 0).all(axis=1))
    grip_steady_acc = float(((p_grip[~is_switch] > 0) == (g_grip[~is_switch] > 0)).mean()) if (~is_switch).any() else 0.0
    grip_switch_acc = float(((p_grip[is_switch] > 0) == (g_grip[is_switch] > 0)).mean()) if is_switch.any() else 0.0
    grip_switch_mse = float(np.mean((p_grip[is_switch] - g_grip[is_switch]) ** 2)) if is_switch.any() else 0.0

    return {
        "mse10": float(base["action10_mse"]),
        "pos_mse": float(base["pos_mse"]),
        "rot6d_mse": float(base["rot6d_mse"]),
        "grip_mse": float(base["grip_mse"]),
        "mae7": float(base["action7_mae"]),
        "grip_acc": float(base["grip_sign_acc"]),
        "accel_pos_mse": accel_pos_mse,
        "jerk_pos_mse": jerk_pos_mse,
        "accel_rot_mse": accel_rot_mse,
        "path_mae": path_mae,
        "spd_env_mse": spd_env_mse,
        "hi_motion_mse10": hi_motion_mse10,
        "grip_steady_acc": grip_steady_acc,
        "grip_switch_acc": grip_switch_acc,
        "grip_switch_mse": grip_switch_mse,
    }


def _agg_dicts(dlist: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in dlist[0]:
        vals = [float(d[k]) for d in dlist]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ---------------------------------------------------------------------------
# Main Runner for Part 1 (Training-Free)
# ---------------------------------------------------------------------------

def run_exp1(
    out_dir: Path,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    policies = {m: load_frozen_e9_stage1_teacher(BASE_CHECKPOINTS[m], device=device) for m in PART1_MODELS}

    # Pre-encode full val starts ONCE for content probes & Tier-2 Editability
    encoded_val = {m: encode_all_val_starts(policies[m], cache, device) for m in PART1_MODELS}
    non_tail = np.flatnonzero(~encoded_val["e9"]["is_tail"])
    fut_a10, fut_valid = _gather_future_actions(cache, horizon_offset=16)
    fut_a7 = action_10d_to_7d(fut_a10)

    # Object labels matching E13 Exp 2
    procs = [cache.episodes[int(e)]["procedure"].split("|")[0].strip() for e in cache.val_episodes]
    obj_labels = np.array([(p.split()[1] if len(p.split()) >= 2 else "other") for p in procs], dtype=object)
    top_objs = sorted({o for o in obj_labels if (obj_labels == o).sum() >= 100})
    obj_to_id = {o: i for i, o in enumerate(top_objs)}
    obj_y = np.array([obj_to_id.get(o, -1) for o in obj_labels], dtype=np.int64)
    obj_oh = np.zeros((len(obj_y), len(top_objs)), dtype=np.float32)
    for i, oi in enumerate(obj_y):
        if oi >= 0:
            obj_oh[i, oi] = 1.0

    # Fixed Tier-2 analogy quadruples (shared across all models!)
    quads = _build_fixed_analogy_quadruples(cache, encoded_val["e9"], device, seed=42)
    A1_q, A2_q, B1_q, B2_q = quads["tier2"]

    # -----------------------------------------------------------------------
    # 1. Audit Ridge Cross-Term & SVD Truncation Error + 3-Component Probes +
    #    Reconstruction/Motion Details + Robustness + Visual ICL across 3 seeds
    # -----------------------------------------------------------------------
    audit_seeds: Dict[str, List[Dict[str, Any]]] = {m: [] for m in PART1_MODELS}
    recon_seeds: Dict[str, List[Dict[str, Dict[str, float]]]] = {m: [] for m in PART1_MODELS}
    robust_seeds: Dict[str, List[Dict[str, Any]]] = {m: [] for m in PART1_MODELS}
    probe_seeds: Dict[str, List[Dict[str, Dict[str, float]]]] = {m: [] for m in PART1_MODELS}
    icl_seeds: Dict[str, List[Dict[str, Dict[str, float]]]] = {m: [] for m in PART1_MODELS}
    edit_by_model: Dict[str, Dict[str, Any]] = {}

    for seed in seeds:
        ts = time.monotonic()
        print(f"[E15 Part 1] Running seed={seed}...", flush=True)
        # Reuse E14's exact zero-leakage bundle (rel_progress for clean visual-ICL diagnosis)
        bundle = build_seed_bundle(
            cache, policies, match_mode="rel_progress", seed=seed, device=device, n_train_pairs=4000
        )
        obs = bundle["obs"]
        acts = bundle["acts"]
        lats = bundle["latents"]
        gt_unseen = acts["test_unseen"]["q"]

        # Disjoint task split on val_starts for Fut_R2 / VisHQ_R2 / ObjAcc probes (matching E13 Exp 2)
        unique_t = np.unique(cache.val_task_ids[non_tail])
        rng_p = np.random.default_rng(seed + 222)
        perm_t = rng_p.permutation(unique_t)
        tr_t_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        valid_nt = non_tail[fut_valid[non_tail]]
        v_tasks = cache.val_task_ids[valid_nt]
        tr_vi = valid_nt[np.array([int(t) in tr_t_set for t in v_tasks])]
        te_vi = valid_nt[np.array([int(t) not in tr_t_set for t in v_tasks])]

        A10_val = encoded_val["e9"]["actions_10d"]
        A7_val = encoded_val["e9"]["actions_7d"].reshape(len(A10_val), -1)
        Fut7_val = fut_a7.reshape(len(fut_a7), -1)
        HQ_val = encoded_val["e9"]["h_queries"].reshape(len(A10_val), -1)

        for m in PART1_MODELS:
            pol = policies[m]
            zq_tr = lats[m]["train"]["q"]
            zd_tr = lats[m]["train"]["d"]
            zq_te = lats[m]["test_unseen"]["q"]
            zd_te = lats[m]["test_unseen"]["d"]
            aq_tr = acts["train"]["q"]
            aq_te = gt_unseen

            # (A) Exact full-rank 3-component fit vs SVD-truncated (E14 0.995) vs Ridge alpha sweep
            m_full = fit_three_component_model(aq_tr, zq_tr, device, svd_threshold=1.0)
            dec_tr = decompose_three_components(aq_tr, zq_tr, m_full)
            dec_te = decompose_three_components(aq_te, zq_te, m_full)
            dec_d_te = decompose_three_components(acts["test_unseen"]["d"], zd_te, m_full)

            # SVD truncation sweep (0.990, 0.995, 0.999, 1.000) using same best alpha
            svd_sweep = {}
            for thr in (0.990, 0.995, 0.999, 1.000):
                m_thr = fit_three_component_model(
                    aq_tr, zq_tr, device, forced_alpha=m_full["alpha"], svd_threshold=thr
                )
                d_te_t = decompose_three_components(aq_te, zq_te, m_thr)
                svd_sweep[f"{thr:.3f}"] = {
                    "rank": m_thr["rank"],
                    "retained_lin_train": m_thr["retained_lin_var_frac_train"],
                    "frac_lin_trunc_svd_err": d_te_t["stats"]["frac_lin_trunc_svd_err"],
                    "frac_lin_trunc_of_lin": d_te_t["stats"]["frac_lin_trunc_of_lin"],
                    "frac_par": d_te_t["stats"]["frac_par"],
                    "frac_perp": d_te_t["stats"]["frac_perp"],
                    "frac_cross_lin_raw_perp": d_te_t["stats"]["frac_cross_lin_raw_perp"],
                }

            # Ridge alpha sweep (OLS=0 vs 1e-3..100) showing exact cross-term origin!
            alpha_sweep = {}
            for al in (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
                m_al = fit_three_component_model(aq_tr, zq_tr, device, forced_alpha=al, svd_threshold=1.0)
                d_tr_a = decompose_three_components(aq_tr, zq_tr, m_al)
                d_te_a = decompose_three_components(aq_te, zq_te, m_al)
                alpha_sweep[f"{al:g}"] = {
                    "train_frac_lin": d_tr_a["stats"]["frac_lin"],
                    "train_frac_coax": d_tr_a["stats"]["frac_coax"],
                    "train_frac_cross": d_tr_a["stats"]["frac_cross_2_lin_coax"],
                    "train_frac_perp": d_tr_a["stats"]["frac_perp"],
                    "test_frac_lin": d_te_a["stats"]["frac_lin"],
                    "test_frac_coax": d_te_a["stats"]["frac_coax"],
                    "test_frac_cross": d_te_a["stats"]["frac_cross_2_lin_coax"],
                    "test_frac_perp": d_te_a["stats"]["frac_perp"],
                    "test_r2_a_to_z": d_te_a["stats"]["r2_a_to_z"],
                    "test_cos_lin_coax": d_te_a["stats"]["cos_lin_coax"],
                }

            audit_seeds[m].append(
                {
                    "cv_alpha": m_full["alpha"],
                    "rank_full": m_full["rank"],
                    "train_stats": dec_tr["stats"],
                    "test_stats": dec_te["stats"],
                    "svd_sweep": svd_sweep,
                    "alpha_sweep": alpha_sweep,
                }
            )

            # (B) Strict 3-Component Reconstruction & Motion Detail Ablation (Oracle GT)
            z_mu = dec_te["z_mu"]
            z_lin = dec_te["z_lin"]
            z_coax = dec_te["z_coax"]
            z_perp = dec_te["z_perp"]
            rng_s = np.random.default_rng(seed + 7001)
            perm1 = rng_s.permutation(len(zq_te))
            perm2 = rng_s.permutation(len(zq_te))

            # Also check E14's 0.995 truncated z_par_only vs exact full-rank z_par_only
            m_995 = fit_three_component_model(
                aq_tr, zq_tr, device, forced_alpha=m_full["alpha"], svd_threshold=0.995
            )
            dec_te_995 = decompose_three_components(aq_te, zq_te, m_995)

            recon_variants = [
                ("full", zq_te),
                ("lin_only", z_mu + z_lin),
                ("par_only_exact", z_mu + z_lin + z_coax),
                ("par_only_svd995", z_mu + dec_te_995["z_par"]),
                ("lin_plus_perp", z_mu + z_lin + z_perp),
                ("coax_only", z_mu + z_coax),
                ("perp_only", z_mu + z_perp),
                ("coax_plus_perp", z_mu + z_coax + z_perp),
                ("lin_plus_shuf_coax_plus_perp", z_mu + z_lin + z_coax[perm1] + z_perp),
                ("lin_plus_coax_plus_shuf_perp", z_mu + z_lin + z_coax + z_perp[perm2]),
                ("lin_plus_shuf_coax_plus_shuf_perp", z_mu + z_lin + z_coax[perm1] + z_perp[perm2]),
                ("lin_plus_demo_coax_plus_demo_perp", z_mu + z_lin + dec_d_te["z_coax"] + dec_d_te["z_perp"]),
            ]

            # (C) Equal-scale Decoder Perturbation Robustness by Component (z_lin, z_coax, z_perp)
            zc_tr = (zq_tr - z_mu).astype(np.float64)
            R_M = float(np.sqrt(np.mean(np.sum(zc_tr ** 2, axis=-1))))

            # Precompute component covariance bases ONCE per model (float32 SVD = ~0.2s each!)
            comp_cov_bases: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for cname in ("z_lin", "z_coax", "z_perp"):
                comp_tr = dec_tr[cname]
                ct = torch.from_numpy((comp_tr - comp_tr.mean(axis=0, keepdims=True)).astype(np.float32)).to(device)
                _, S_c, Vh_c = torch.linalg.svd(ct, full_matrices=False)
                sc = (S_c / math.sqrt(max(len(ct) - 1, 1))).cpu().numpy().astype(np.float64)
                vh = Vh_c.cpu().numpy().astype(np.float64)
                comp_cov_bases[cname] = (sc, vh)

            def _cov_sample(cname: str, n_samp: int, sig: float, r_seed: int) -> np.ndarray:
                sc, vh = comp_cov_bases[cname]
                g = np.random.default_rng(r_seed).standard_normal((n_samp, len(sc)))
                eps = (g * sc[None, :]) @ vh
                cur = float(np.sqrt(np.mean(np.sum(eps ** 2, axis=-1))))
                return (eps * (sig * R_M / max(cur, 1e-12))).astype(np.float32)

            # Visual ICL Ridge prediction (Zero-Leakage) for both ICL error split & part D
            X_tr_f, X_ev_f = build_block_rms_features(
                [obs["train"]["dh"], obs["train"]["h_q"], zd_tr],
                [[obs["test_unseen"]["dh"]], [obs["test_unseen"]["h_q"]], [zd_te]],
            )
            z_pred_icl = fit_predict_scaled_ridge(X_tr_f, zq_tr, X_ev_f, device)[0][0]
            e_icl = (z_pred_icl - zq_te).astype(np.float64)
            Q_full = m_full["Q_act"]
            e_icl_par = (e_icl @ Q_full) @ Q_full.T
            e_icl_perp = e_icl - e_icl_par
            # Predict z_perp from h_q only (Zero-Leakage!)
            X_hq_tr, X_hq_ev = build_block_rms_features(
                [obs["train"]["h_q"]], [[obs["test_unseen"]["h_q"]]]
            )
            z_perp_hq = fit_predict_scaled_ridge(X_hq_tr, dec_tr["z_perp"], X_hq_ev, device)[0][0]
            # Decompose z_pred_icl orthogonally (ZERO GT leakage at inference!)
            zp_c = (z_pred_icl - z_mu).astype(np.float64)
            zp_par = ((zp_c @ Q_full) @ Q_full.T).astype(np.float32)
            zp_perp = (zp_c - zp_par).astype(np.float32)
            # Oracle post-hoc breakdown of z_pred_icl error into (pred_par - z_lin) vs z_coax
            # Note: z_pred_par - (z_lin + z_coax) = e_icl_par
            # Also train zero-leakage visual predictors directly for z_lin, z_coax, z_perp (strictly train targets!)
            pred_z_lin = fit_predict_scaled_ridge(X_tr_f, dec_tr["z_lin"], X_ev_f, device)[0][0]
            pred_z_coax = fit_predict_scaled_ridge(X_tr_f, dec_tr["z_coax"], X_ev_f, device)[0][0]
            pred_z_perp = fit_predict_scaled_ridge(X_tr_f, dec_tr["z_perp"], X_ev_f, device)[0][0]

            # Check separate visual predictability R^2 of z_lin, z_coax, z_perp!
            def _r2_vec(yp: np.ndarray, yt: np.ndarray) -> float:
                v = float(np.mean((yt - yt.mean(axis=0, keepdims=True)) ** 2))
                return float(1.0 - np.mean((yp - yt) ** 2) / max(v, 1e-12))

            vis_pred_r2 = {
                "r2_vis_to_full": _r2_vec(z_pred_icl, zq_te),
                "r2_vis_to_lin": _r2_vec(pred_z_lin, z_lin),
                "r2_vis_to_coax": _r2_vec(pred_z_coax, z_coax),
                "r2_vis_to_perp": _r2_vec(pred_z_perp, z_perp),
                "r2_hq_to_perp": _r2_vec(z_perp_hq, z_perp),
                "icl_err_par_frac": float(np.mean(np.sum(e_icl_par ** 2, axis=-1)) / max(np.mean(np.sum(e_icl ** 2, axis=-1)), 1e-12)),
                "icl_err_perp_frac": float(np.mean(np.sum(e_icl_perp ** 2, axis=-1)) / max(np.mean(np.sum(e_icl ** 2, axis=-1)), 1e-12)),
            }

            # Perturbation latents at sigma in {0.10, 0.20, 0.35, 0.50}
            rob_variants: List[Tuple[str, np.ndarray]] = []
            for sig in (0.10, 0.20, 0.35, 0.50):
                rob_variants.append((f"rob|lin|{sig:.2f}", zq_te + _cov_sample("z_lin", len(zq_te), sig, seed + 8100 + int(sig * 100))))
                rob_variants.append((f"rob|coax|{sig:.2f}", zq_te + _cov_sample("z_coax", len(zq_te), sig, seed + 8200 + int(sig * 100))))
                rob_variants.append((f"rob|perp|{sig:.2f}", zq_te + _cov_sample("z_perp", len(zq_te), sig, seed + 8300 + int(sig * 100))))
                cur_e = float(np.sqrt(np.mean(np.sum(e_icl ** 2, axis=-1))))
                rob_variants.append((f"rob|icl_dir|{sig:.2f}", zq_te + (e_icl * (sig * R_M / max(cur_e, 1e-12))).astype(np.float32)))

            # Visual ICL Zero-Leakage & Oracle Diagnostic latents
            icl_variants = [
                # Strictly Zero-Leakage inference variants:
                ("zl|icl_full", z_pred_icl),
                ("zl|icl_par_only", z_mu + zp_par),
                ("zl|icl_par_plus_hq_perp", z_mu + zp_par + z_perp_hq),
                ("zl|icl_pred_lin_only", z_mu + pred_z_lin),
                ("zl|icl_pred_lin_plus_pred_perp", z_mu + pred_z_lin + pred_z_perp),
                ("zl|icl_pred_lin_plus_pred_coax", z_mu + pred_z_lin + pred_z_coax),
                # Explicitly labeled Oracle Diagnostic variants (uses GT component to localize error!):
                ("oracle|pred_lin_plus_gt_coax_plus_pred_perp", z_mu + pred_z_lin + z_coax + pred_z_perp),
                ("oracle|pred_lin_plus_pred_coax_plus_gt_perp", z_mu + pred_z_lin + pred_z_coax + z_perp),
                ("oracle|pred_lin_plus_gt_coax_plus_gt_perp", z_mu + pred_z_lin + z_coax + z_perp),
                ("oracle|gt_lin_plus_pred_coax_plus_pred_perp", z_mu + z_lin + pred_z_coax + pred_z_perp),
                ("oracle|gt_lin_plus_gt_coax_plus_pred_perp", z_mu + z_lin + z_coax + pred_z_perp),
                ("oracle|gt_lin_plus_pred_coax_plus_gt_perp", z_mu + z_lin + pred_z_coax + z_perp),
            ]

            # Batch decode ALL variants in ONE fixed-noise DDIM pass!
            all_tags = [f"rec|{k}" for k, _ in recon_variants] + [k for k, _ in rob_variants] + [f"icl|{k}" for k, _ in icl_variants]
            all_zs = [z_arr for _, z_arr in recon_variants] + [z_arr for _, z_arr in rob_variants] + [z_arr for _, z_arr in icl_variants]
            decoded = ddim_decode_fixed_noise(pol, all_zs, device, seed=12345, batch_size=1024)

            clean_act = decoded[0]  # "rec|full"
            r_rec: Dict[str, Dict[str, float]] = {}
            r_rob: Dict[str, Dict[str, Dict[str, float]]] = {}
            r_icl: Dict[str, Dict[str, float]] = {"vis_pred_r2": vis_pred_r2}

            for tag, pred_a in zip(all_tags, decoded):
                md_m = compute_motion_detail_metrics(pred_a, gt_unseen)
                shift_rms = float(np.sqrt(np.mean((pred_a - clean_act) ** 2)))
                md_m["shift_rms_vs_clean"] = shift_rms
                parts = tag.split("|")
                if parts[0] == "rec":
                    r_rec[parts[1]] = md_m
                elif parts[0] == "rob":
                    r_rob.setdefault(parts[1], {})[parts[2]] = md_m
                elif parts[0] == "icl":
                    r_icl[f"{parts[1]}|{parts[2]}"] = md_m

            recon_seeds[m].append(r_rec)
            robust_seeds[m].append(r_rob)
            icl_seeds[m].append(r_icl)

            # (D) Content Probes per Component (a7, Fut_R2, VisHQ_R2, ObjAcc) on disjoint val tasks
            Z_val = encoded_val[m]["latents_flat"]
            m_val = fit_three_component_model(A10_val[tr_vi], Z_val[tr_vi], device, svd_threshold=1.0)
            d_v_tr = decompose_three_components(A10_val[tr_vi], Z_val[tr_vi], m_val)
            d_v_te = decompose_three_components(A10_val[te_vi], Z_val[te_vi], m_val)

            var_a7 = float(((A7_val[te_vi] - A7_val[te_vi].mean(axis=0, keepdims=True)) ** 2).mean())
            var_fut = float(((Fut7_val[te_vi] - Fut7_val[te_vi].mean(axis=0, keepdims=True)) ** 2).mean())
            var_hq = float(((HQ_val[te_vi] - HQ_val[te_vi].mean(axis=0, keepdims=True)) ** 2).mean())
            obj_tr_m = obj_y[tr_vi] >= 0
            obj_te_m = obj_y[te_vi] >= 0

            comp_probes: Dict[str, Dict[str, float]] = {}
            for cname, Xtr_c, Xte_c in [
                ("full", d_v_tr["z_c"], d_v_te["z_c"]),
                ("z_lin", d_v_tr["z_lin"], d_v_te["z_lin"]),
                ("z_coax", d_v_tr["z_coax"], d_v_te["z_coax"]),
                ("z_perp", d_v_tr["z_perp"], d_v_te["z_perp"]),
                ("z_coax_plus_perp", d_v_tr["z_coax"] + d_v_tr["z_perp"], d_v_te["z_coax"] + d_v_te["z_perp"]),
            ]:
                p_a7, _ = _ridge_fit_predict(Xtr_c, A7_val[tr_vi], Xte_c, device=device)
                p_fut, _ = _ridge_fit_predict(Xtr_c, Fut7_val[tr_vi], Xte_c, device=device)
                p_hq, _ = _ridge_fit_predict(Xtr_c, HQ_val[tr_vi], Xte_c, device=device)
                p_obj, _ = _ridge_fit_predict(Xtr_c[obj_tr_m], obj_oh[tr_vi[obj_tr_m]], Xte_c[obj_te_m], device=device)
                er_c = _effective_rank_stats(Xte_c, device=device)
                comp_probes[cname] = {
                    "r2_to_a7": float(1.0 - ((p_a7 - A7_val[te_vi]) ** 2).mean() / max(var_a7, 1e-12)),
                    "r2_to_fut7": float(1.0 - ((p_fut - Fut7_val[te_vi]) ** 2).mean() / max(var_fut, 1e-12)),
                    "r2_to_hq": float(1.0 - ((p_hq - HQ_val[te_vi]) ** 2).mean() / max(var_hq, 1e-12)),
                    "obj_acc": float((np.argmax(p_obj, axis=-1) == obj_y[te_vi[obj_te_m]]).mean()),
                    "erank": float(er_c["erank_energy"]),
                }
            probe_seeds[m].append(comp_probes)

        print(f"[E15 Part 1] Finished seed={seed} in {time.monotonic() - ts:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # 2. Editability (Oracle Tier-2 Latent Arithmetic — Selective Component Transport)
    # -----------------------------------------------------------------------
    print("[E15 Part 1] Running Tier-2 Latent Arithmetic 3-component editability audit...", flush=True)
    # Fit 3-component decomposition on train-fold clips (strictly disjoint from val quadruples!)
    # Use seed=42 bundle's train clips to fit W_az and Q_act with zero leakage into val quadruples.
    bundle_ref = build_seed_bundle(
        cache, policies, match_mode="rel_progress", seed=42, device=device, n_train_pairs=4000
    )
    gt_B2 = encoded_val["e9"]["actions_10d"][B2_q]

    for m in PART1_MODELS:
        pol = policies[m]
        m_ref = fit_three_component_model(
            bundle_ref["acts"]["train"]["q"], bundle_ref["latents"][m]["train"]["q"], device, svd_threshold=1.0
        )
        zf = encoded_val[m]["latents_flat"]
        a10_v = encoded_val[m]["actions_10d"]
        dA1 = decompose_three_components(a10_v[A1_q], zf[A1_q], m_ref)
        dA2 = decompose_three_components(a10_v[A2_q], zf[A2_q], m_ref)
        dB1 = decompose_three_components(a10_v[B1_q], zf[B1_q], m_ref)
        dB2 = decompose_three_components(a10_v[B2_q], zf[B2_q], m_ref)

        z_mu = dB1["z_mu"]
        dz_full = zf[A2_q] - zf[A1_q]
        dz_lin = dA2["z_lin"] - dA1["z_lin"]
        dz_coax = dA2["z_coax"] - dA1["z_coax"]
        dz_par = dA2["z_par"] - dA1["z_par"]
        dz_perp = dA2["z_perp"] - dA1["z_perp"]

        # Exact parallel-transport edit ablations:
        #   - full_arith        : z_B1 + (z_A2 - z_A1)
        #   - transport_par_only: z_B1 + dz_par (preserves B1's z_perp!)
        #   - transport_lin_only: z_B1 + dz_lin (preserves B1's z_coax + z_perp!)
        #   - transport_coax_only: z_B1 + dz_coax
        #   - transport_perp_only: z_B1 + dz_perp
        #   - arith_zero_perp   : (z_B1_par + dz_par) with z_perp=0
        #   - arith_lin_only    : (z_B1_lin + dz_lin) with z_coax=0, z_perp=0
        edit_vars = [
            ("oracle_B2", zf[B2_q]),
            ("base_B1", zf[B1_q]),
            ("arith_full", zf[B1_q] + dz_full),
            ("transport_par_only", zf[B1_q] + dz_par),
            ("transport_lin_only", zf[B1_q] + dz_lin),
            ("transport_coax_only", zf[B1_q] + dz_coax),
            ("transport_perp_only", zf[B1_q] + dz_perp),
            ("arith_par_zero_perp", z_mu + dB1["z_par"] + dz_par),
            ("arith_lin_zero_coax_perp", z_mu + dB1["z_lin"] + dz_lin),
        ]
        dec_ed = ddim_decode_fixed_noise(pol, [z_arr for _, z_arr in edit_vars], device, seed=12345, batch_size=512)
        m_ora = compute_motion_detail_metrics(dec_ed[0], gt_B2)
        m_bas = compute_motion_detail_metrics(dec_ed[1], gt_B2)

        # Component-wise displacement cosine alignment between (A1->A2) and (B1->B2)
        def _mean_cos(u: np.ndarray, v: np.ndarray) -> float:
            nu = np.linalg.norm(u, axis=-1)
            nv = np.linalg.norm(v, axis=-1)
            ok = (nu > 1e-6) & (nv > 1e-6)
            return float(np.mean(np.sum(u[ok] * v[ok], axis=-1) / (nu[ok] * nv[ok]))) if ok.any() else 0.0

        edit_res: Dict[str, Any] = {
            "disp_cos": {
                "full": _mean_cos(dz_full, zf[B2_q] - zf[B1_q]),
                "lin": _mean_cos(dz_lin, dB2["z_lin"] - dB1["z_lin"]),
                "coax": _mean_cos(dz_coax, dB2["z_coax"] - dB1["z_coax"]),
                "par": _mean_cos(dz_par, dB2["z_par"] - dB1["z_par"]),
                "perp": _mean_cos(dz_perp, dB2["z_perp"] - dB1["z_perp"]),
            },
            "variants": {},
        }
        for (ename, _), pa in zip(edit_vars, dec_ed):
            mm = compute_motion_detail_metrics(pa, gt_B2)
            mm["arr_10d"] = float(
                (m_bas["mse10"] - mm["mse10"]) / max(m_bas["mse10"] - m_ora["mse10"], 1e-9)
            )
            edit_res["variants"][ename] = mm
        edit_by_model[m] = edit_res

    # -----------------------------------------------------------------------
    # Aggregate Multi-Seed Results
    # -----------------------------------------------------------------------
    agg_out: Dict[str, Any] = {}
    for m in PART1_MODELS:
        # Audit aggregation
        tr_stats = _agg_dicts([s["train_stats"] for s in audit_seeds[m]])
        te_stats = _agg_dicts([s["test_stats"] for s in audit_seeds[m]])
        svd_agg = {
            thr: _agg_dicts([s["svd_sweep"][thr] for s in audit_seeds[m]])
            for thr in audit_seeds[m][0]["svd_sweep"]
        }
        alpha_agg = {
            al: _agg_dicts([s["alpha_sweep"][al] for s in audit_seeds[m]])
            for al in audit_seeds[m][0]["alpha_sweep"]
        }
        # Recon aggregation
        rec_agg = {
            k: _agg_dicts([s[k] for s in recon_seeds[m]])
            for k in recon_seeds[m][0]
        }
        # Robust aggregation
        rob_agg = {
            comp: {
                sig: _agg_dicts([s[comp][sig] for s in robust_seeds[m]])
                for sig in robust_seeds[m][0][comp]
            }
            for comp in robust_seeds[m][0]
        }
        # Probes aggregation
        prb_agg = {
            cname: _agg_dicts([s[cname] for s in probe_seeds[m]])
            for cname in probe_seeds[m][0]
        }
        # ICL aggregation
        icl_agg = {
            k: _agg_dicts([s[k] for s in icl_seeds[m]])
            for k in icl_seeds[m][0]
        }
        agg_out[m] = {
            "cv_alpha_mean": float(np.mean([s["cv_alpha"] for s in audit_seeds[m]])),
            "rank_full_mean": float(np.mean([s["rank_full"] for s in audit_seeds[m]])),
            "energy_audit_train": tr_stats,
            "energy_audit_test_unseen": te_stats,
            "svd_truncation_audit": svd_agg,
            "ridge_alpha_crossterm_audit": alpha_agg,
            "reconstruction_motion_details": rec_agg,
            "decoder_robustness_by_component": rob_agg,
            "component_content_probes": prb_agg,
            "visual_icl_component_role": icl_agg,
            "tier2_editability_ablation": edit_by_model[m],
        }

    summary = {
        "models": list(PART1_MODELS),
        "seeds": list(seeds),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "results": agg_out,
    }
    atomic_write_json(summary, out_dir / "exp1_three_component_summary.json")

    # Print crisp executive diagnostic table
    print("\n=== [E15 Part 1] Strict Three-Component Decomposition & Audit (Unseen Tasks, 3 seeds) ===", flush=True)
    for m in PART1_MODELS:
        ts_m = agg_out[m]["energy_audit_test_unseen"]
        tr_m = agg_out[m]["energy_audit_train"]
        svd995 = agg_out[m]["svd_truncation_audit"]["0.995"]
        ols0 = agg_out[m]["ridge_alpha_crossterm_audit"]["0"]
        rec = agg_out[m]["reconstruction_motion_details"]
        ed = agg_out[m]["tier2_editability_ablation"]
        rob = agg_out[m]["decoder_robustness_by_component"]
        print(
            f"\n[{m.upper()}] alpha={agg_out[m]['cv_alpha_mean']:.3g} rank_full={agg_out[m]['rank_full_mean']:.1f} (r_995={svd995['rank_mean']:.1f}):\n"
            f"  Energy(Unseen): lin={ts_m['frac_lin_mean']*100:.2f}% | coax={ts_m['frac_coax_mean']*100:.2f}% | "
            f"2<lin,coax>={ts_m['frac_cross_2_lin_coax_mean']*100:+.2f}% (OLS train={ols0['train_frac_cross_mean']*100:+.4f}%, test={ols0['test_frac_cross_mean']*100:+.2f}%) | "
            f"perp={ts_m['frac_perp_mean']*100:.2f}% | sum={ts_m['frac_sum_check_mean']*100:.3f}%\n"
            f"  SVD-0.995 Trunc Err: {svd995['frac_lin_trunc_svd_err_mean']*100:.4f}% of ||z||^2 ({svd995['frac_lin_trunc_of_lin_mean']*100:.4f}% of ||z_lin||^2)\n"
            f"  Recon MSE10 (full / lin_only / par_exact / par_995 / lin+perp / shuf_coax / shuf_perp): "
            f"{rec['full']['mse10_mean']:.5f} / {rec['lin_only']['mse10_mean']:.5f} / {rec['par_only_exact']['mse10_mean']:.5f} / "
            f"{rec['par_only_svd995']['mse10_mean']:.5f} / {rec['lin_plus_perp']['mse10_mean']:.5f} / "
            f"{rec['lin_plus_shuf_coax_plus_perp']['mse10_mean']:.5f} / {rec['lin_plus_coax_plus_shuf_perp']['mse10_mean']:.5f}\n"
            f"  Tier2 Arith MSE10 (full / tr_par / tr_lin / par_0perp): "
            f"{ed['variants']['arith_full']['mse10']:.5f}({ed['variants']['arith_full']['arr_10d']*100:.1f}%) / "
            f"{ed['variants']['transport_par_only']['mse10']:.5f}({ed['variants']['transport_par_only']['arr_10d']*100:.1f}%) / "
            f"{ed['variants']['transport_lin_only']['mse10']:.5f}({ed['variants']['transport_lin_only']['arr_10d']*100:.1f}%) / "
            f"{ed['variants']['arith_par_zero_perp']['mse10']:.5f}({ed['variants']['arith_par_zero_perp']['arr_10d']*100:.1f}%)\n"
            f"  @sigma=0.35 ShiftRMS (lin / coax / perp / icl_dir): "
            f"{rob['lin']['0.35']['shift_rms_vs_clean_mean']:.4f} / {rob['coax']['0.35']['shift_rms_vs_clean_mean']:.4f} / "
            f"{rob['perp']['0.35']['shift_rms_vs_clean_mean']:.4f} / {rob['icl_dir']['0.35']['shift_rms_vs_clean_mean']:.4f}",
            flush=True,
        )
    print(f"\n[E15 Part 1] Saved summary to {out_dir / 'exp1_three_component_summary.json'} ({summary['elapsed_seconds']}s)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp1_three_component")
    args = p.parse_args()
    run_exp1(args.out_dir)


if __name__ == "__main__":
    main()
