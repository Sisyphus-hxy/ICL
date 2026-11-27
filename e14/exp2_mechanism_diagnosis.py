"""E14 Experiment 2: Deep Mechanism Diagnosis (3 Parts).

Disentangles THREE distinct candidate causes of why E12-B / E12-C / E12-BC-FactorDisp
underperform E9 in strictly zero-leakage Visual ICL:

  Part 2A — Target Latent Predictability & Linear vs Nonlinear Adapter Gap:
    - Scale-invariant latent predictability: total R^2(z_q), R^2(dz), cos(dz),
      action-subspace R^2(z_par) vs action-unexplained R^2(z_perp), per-token R^2(Tok0..3),
      effective rank of z & dz, Ridge shrinkage ratio ||z_hat-mu||/||z-mu||.
    - Properly-tuned Linear vs Nonlinear adapters across N in {100, 500, 1000, 4000}:
        1. Linear Ridge (full-dim block-RMS & PCA-320)
        2. Exact Closed-Form RBF Kernel Ridge (universal nonlinear baseline!)
        3. Early-Stopped Residual-on-Ridge MLP (guaranteed >= Ridge on val)
        4. Early-Stopped Standalone MLP (bottleneck-regularized)
    - Directly tests whether nonlinearity conservation holds (does nonlinear gain close
      the E12->E9 gap more on E12-C/BC than on E9?).

  Part 2B — Decoder Sensitivity to Equal-Scale Latent Perturbations & Shrinkage:
    - Evaluates all 4 frozen Diffusion Decoders on identical unseen-task val clips under
      strictly equal relative perturbation scales sigma in {0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0}
      (normalized by each model's own centered latent RMS):
        1. Isotropic Gaussian noise
        2. Covariance-matched on-manifold Gaussian noise N(0, Sigma_M)
        3. Action-explainable subspace (z_par) vs Action-unexplained subspace (z_perp) noise
        4. Radial Shrinkage toward mean: z_s = mu + (1 - s) * (z - mu) (mimics Ridge shrinkage!)
        5. Actual Visual-ICL error vectors (e_icl = z_hat - z_q), evaluated BOTH at actual
           scale AND rescaled to matched relative norm sigma in {0.20, 0.35, 0.50, 0.75}!
    - Computes exact multiplicative/log-additive factorization:
        sqrt(MSE_act(z_hat) - MSE_act(z_q))  ==  e_lat(M)  *  G_dec(M)
      cleanly quantifying % of Visual ICL gap from Latent Error vs Decoder Amplification!

  Part 2C — Action-Explainable (z_par) vs Action-Unexplained (z_perp) Subspace Causal Ablation:
    - Fits orthonormal basis Q_act of the action-explainable subspace STRICTLY on train split
      (zero test GT action leakage!).
    - Performs surgical causal ablations on E9 (and compares against E12-B, E12-C, E12-BC-FD):
        1. Decoder GT ablation: GT z_q vs z_par(GT) only vs z_par(GT)+z_perp(demo) vs
           z_par(GT)+z_perp(shuffled) vs z_perp(GT) only + matched-dim random subspace control.
        2. Input-side ablation: Full z_d vs z_par(demo) only vs z_perp(demo) only vs Vision-Only
           + matched-dim random subspace control.
        3. Predicted-side Visual ICL ablation:
           - Full pred = z_par_hat + z_perp_hat
           - Only z_par_hat (zero out predicted z_perp)
           - z_par_hat + z_perp(demo) (keep demo context, only adapt action subspace)
           - z_par_hat + z_perp_hat(h_q only) (predict z_perp purely from query image h_q!)
           - z_par_hat + z_perp_GT (Oracle z_perp upper bound)
           - z_par_GT + z_perp_hat (Oracle z_par upper bound)
           - Matched-dim random subspace zeroing control!
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from e5.train import atomic_write_json
from e12.eval_suite import ValClipCache, _action_metrics_dict, _effective_rank_stats

from . import CORE_MODELS, MATCH_MODES_DEFAULT, OUTPUT_ROOT, SEEDS_DEFAULT
from .common import (
    build_block_rms_features,
    build_pca_block_rms_features,
    build_seed_bundle,
    compute_latent_predictability_metrics,
    ddim_decode_fixed_noise,
    fit_action_explainable_subspace,
    fit_global_rms_standardizer,
    fit_predict_rbf_krr,
    fit_predict_scaled_ridge,
    fit_predict_tuned_mlp,
    load_policies,
    make_matched_energy_random_subspace,
    pack_action_metrics,
    project_subspace,
)


def _agg(blocks: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in blocks[0]:
        vals = [float(b[k]) for b in blocks]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ---------------------------------------------------------------------------
# Part 2A + 2B(actual-ICL) + 2C: Runs cleanly inside each (mode, seed) bundle
# ---------------------------------------------------------------------------

def _run_2a_and_2c_for_bundle(
    bundle: Dict[str, Any],
    policies: Dict[str, Any],
    device: torch.device,
    *,
    budgets_2a: Tuple[int, ...] = (100, 500, 1000, 4000),
    budgets_2c: Tuple[int, ...] = (100, 500, 1000, 4000),
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """Run Part 2A (predictability + linear/nonlinear gap) and Part 2C (subspace ablations).

    Also returns `icl_errors_unseen[model]` = actual Visual-ICL error vectors at N=4000
    for Part 2B's empirical decoder amplification analysis!
    """
    obs = bundle["obs"]
    acts = bundle["acts"]
    lats = bundle["latents"]
    seed = int(bundle["seed"])

    dh_tr, hq_tr = obs["train"]["dh"], obs["train"]["h_q"]
    dh_ev = [obs["test_seen"]["dh"], obs["test_unseen"]["dh"]]
    hq_ev = [obs["test_seen"]["h_q"], obs["test_unseen"]["h_q"]]
    gt_seen, gt_unseen = acts["test_seen"]["q"], acts["test_unseen"]["q"]
    gt_list = [gt_seen, gt_unseen]

    res_2a: Dict[str, Any] = {}
    res_2c: Dict[str, Any] = {}
    icl_err_bank: Dict[str, Dict[str, Any]] = {}

    for mname in CORE_MODELS:
        pol = policies[mname]
        zd_tr_all = lats[mname]["train"]["d"]
        zq_tr_all = lats[mname]["train"]["q"]
        zd_ev = [lats[mname]["test_seen"]["d"], lats[mname]["test_unseen"]["d"]]
        zq_ev = [lats[mname]["test_seen"]["q"], lats[mname]["test_unseen"]["q"]]

        # Fit action-explainable orthonormal subspace Q_act STRICTLY on training set!
        sub_info = fit_action_explainable_subspace(acts["train"]["q"], zq_tr_all, device, var_threshold=0.995)
        Q_act = sub_info["Q_act"]
        z_mu = sub_info["z_mu"]
        z_rms = sub_info["z_tr_rms"]
        r_act = int(sub_info["rank"])
        Q_rand = make_matched_energy_random_subspace(
            zq_tr_all, r_act, sub_info["geo_subspace_var_frac"], seed=seed + 77
        )

        # Effective rank stats of z_q and dz on train
        erank_z = _effective_rank_stats(zq_tr_all, device=device)
        erank_dz = _effective_rank_stats(zq_tr_all - zd_tr_all, device=device)

        # Direct copy & oracle bounds
        dc_s, ora_s = ddim_decode_fixed_noise(pol, [zd_ev[0], zq_ev[0]], device, seed=12345)
        dc_u, ora_u = ddim_decode_fixed_noise(pol, [zd_ev[1], zq_ev[1]], device, seed=12345)
        dc_m = {"seen": _action_metrics_dict(dc_s, gt_seen), "unseen": _action_metrics_dict(dc_u, gt_unseen)}
        ora_m = {"seen": _action_metrics_dict(ora_s, gt_seen), "unseen": _action_metrics_dict(ora_u, gt_unseen)}

        # Collect all predictions to decode in ONE batched DDIM call per split!
        pending: List[List[Tuple[str, np.ndarray, Dict[str, float]]]] = [[], []]

        # ---- Part 2A: Linear vs Nonlinear Adapters across budgets_2a ----
        for n_b in budgets_2a:
            n_eff = min(int(n_b), len(dh_tr))
            Y_tr = zq_tr_all[:n_eff]
            mu_n, rms_n = fit_global_rms_standardizer(Y_tr)

            # Full-dim block-RMS Linear Ridge
            X_tr_f, X_ev_f = build_block_rms_features(
                [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr_all[:n_eff]],
                [dh_ev, hq_ev, zd_ev],
            )
            pred_lin_f, al_f = fit_predict_scaled_ridge(X_tr_f, Y_tr, X_ev_f, device)

            # PCA-320 block-RMS features (for fair Linear vs RBF-KRR vs Residual-MLP vs Standalone-MLP)
            X_tr_p, X_ev_p = build_pca_block_rms_features(
                dh_tr[:n_eff], hq_tr[:n_eff], [zd_tr_all[:n_eff]],
                dh_ev, hq_ev, [zd_ev], device, k_obs=128, k_lat_list=(64,),
            )
            pred_lin_p, al_p = fit_predict_scaled_ridge(X_tr_p, Y_tr, X_ev_p, device)
            pred_krr, krr_info = fit_predict_rbf_krr(X_tr_p, Y_tr, X_ev_p, device)
            pred_rmlp, rmlp_info = fit_predict_tuned_mlp(
                X_tr_p, Y_tr, X_ev_p, device,
                hidden_dim=512, bottleneck_dim=256, max_steps=700,
                residual_on_ridge=True, seed=seed + n_eff,
            )
            pred_smlp, smlp_info = fit_predict_tuned_mlp(
                X_tr_p, Y_tr, X_ev_p, device,
                hidden_dim=512, bottleneck_dim=256, max_steps=700,
                residual_on_ridge=False, seed=seed + n_eff + 17,
            )
            # Also RBF-KRR on FULL-DIM block-RMS features (no PCA truncation!)
            pred_krr_f, krr_f_info = fit_predict_rbf_krr(X_tr_f, Y_tr, X_ev_f, device)

            adapter_preds = [
                (f"2a|linear_full|{n_eff}", pred_lin_f, {"alpha": float(al_f)}),
                (f"2a|rbf_krr_full|{n_eff}", pred_krr_f, {"gamma_mult": float(krr_f_info["gamma_mult"])}),
                (f"2a|linear_pca|{n_eff}", pred_lin_p, {"alpha": float(al_p)}),
                (f"2a|rbf_krr_pca|{n_eff}", pred_krr, {"gamma_mult": float(krr_info["gamma_mult"])}),
                (f"2a|res_mlp_pca|{n_eff}", pred_rmlp, {"best_step": float(rmlp_info["best_step"])}),
                (f"2a|standalone_mlp_pca|{n_eff}", pred_smlp, {"best_step": float(smlp_info["best_step"])}),
            ]
            for tag, p_list, extra in adapter_preds:
                for sp_i in (0, 1):
                    lm = compute_latent_predictability_metrics(
                        p_list[sp_i], zq_ev[sp_i], zd_ev[sp_i], mu_n, rms_n, Q_act=Q_act
                    )
                    lm.update(extra)
                    pending[sp_i].append((tag, p_list[sp_i], lm))

            if n_eff == max(budgets_2a):
                icl_err_bank[mname] = {
                    "z_true_unseen": zq_ev[1].copy(),
                    "z_pred_lin_full_unseen": pred_lin_f[1].copy(),
                    "z_pred_krr_full_unseen": pred_krr_f[1].copy(),
                    "z_mu": z_mu.copy(),
                    "z_rms": float(z_rms),
                    "Q_act": Q_act.copy(),
                    "gt_unseen": gt_unseen.copy(),
                    "ora_mse10_unseen": float(ora_m["unseen"]["action10_mse"]),
                }

        # ---- Part 2C: Action-Explainable (z_par) vs Action-Unexplained (z_perp) Subspace Causal Ablations ----
        # 1. Decoder-side GT z_q ablations (does frozen Decoder_M actually depend on z_perp?)
        rng_shuf = np.random.default_rng(seed + 901)
        for sp_i in (0, 1):
            zq_par, zq_perp = project_subspace(zq_ev[sp_i], z_mu, Q_act)
            zd_par, zd_perp = project_subspace(zd_ev[sp_i], z_mu, Q_act)
            perm = rng_shuf.permutation(len(zq_ev[sp_i]))
            zq_rand_par, zq_rand_perp = project_subspace(zq_ev[sp_i], z_mu, Q_rand)

            gt_ablations = [
                ("2c_gt|gt_full", zq_ev[sp_i]),
                ("2c_gt|gt_par_only", z_mu + zq_par),
                ("2c_gt|gt_par_plus_demo_perp", z_mu + zq_par + zd_perp),
                ("2c_gt|gt_par_plus_shuf_perp", z_mu + zq_par + zq_perp[perm]),
                ("2c_gt|gt_perp_only", z_mu + zq_perp),
                ("2c_gt|gt_rand_par_only", z_mu + zq_rand_par),
                ("2c_gt|gt_rand_perp_only", z_mu + zq_rand_perp),
            ]
            for tag, z_mod in gt_ablations:
                lm = compute_latent_predictability_metrics(
                    z_mod, zq_ev[sp_i], zd_ev[sp_i], z_mu, z_rms, Q_act=Q_act
                )
                pending[sp_i].append((tag, z_mod.astype(np.float32), lm))

        # 2. Input-side & Predicted-side Visual ICL subspace ablations across budgets_2c
        # (Run all 4 budgets on E9; run N=1000 & 4000 on E12-B/C/BC-FD for cross-model comparison!)
        b_list_2c = budgets_2c if mname == "e9" else (1000, 4000)
        for n_b in b_list_2c:
            n_eff = min(int(n_b), len(dh_tr))
            Y_tr = zq_tr_all[:n_eff]
            mu_n, rms_n = fit_global_rms_standardizer(Y_tr)

            # Decompose demo input strictly using train-fitted Q_act & Q_rand
            zd_tr_par, zd_tr_perp = project_subspace(zd_tr_all[:n_eff], z_mu, Q_act)
            zd_ev_par = [project_subspace(z, z_mu, Q_act)[0] for z in zd_ev]
            zd_ev_perp = [project_subspace(z, z_mu, Q_act)[1] for z in zd_ev]
            zd_tr_rpar, zd_tr_rperp = project_subspace(zd_tr_all[:n_eff], z_mu, Q_rand)
            zd_ev_rpar = [project_subspace(z, z_mu, Q_rand)[0] for z in zd_ev]
            zd_ev_rperp = [project_subspace(z, z_mu, Q_rand)[1] for z in zd_ev]

            # Input-side ablations: Full z_d vs z_par(demo) vs z_perp(demo) vs Vision-Only vs Random-Subspace
            in_variants = [
                ("in_full", [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr_all[:n_eff]], [dh_ev, hq_ev, zd_ev]),
                ("in_par_only", [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr_par], [dh_ev, hq_ev, zd_ev_par]),
                ("in_perp_only", [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr_perp], [dh_ev, hq_ev, zd_ev_perp]),
                ("in_vision_only", [dh_tr[:n_eff], hq_tr[:n_eff]], [dh_ev, hq_ev]),
                ("in_rand_par", [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr_rpar], [dh_ev, hq_ev, zd_ev_rpar]),
                ("in_rand_perp", [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr_rperp], [dh_ev, hq_ev, zd_ev_rperp]),
            ]
            full_pred_cache = None
            for vname, b_tr, b_ev in in_variants:
                X_tr_v, X_ev_v = build_block_rms_features(b_tr, b_ev)
                preds_v, _ = fit_predict_scaled_ridge(X_tr_v, Y_tr, X_ev_v, device)
                if vname == "in_full":
                    full_pred_cache = preds_v
                for sp_i in (0, 1):
                    lm = compute_latent_predictability_metrics(
                        preds_v[sp_i], zq_ev[sp_i], zd_ev[sp_i], mu_n, rms_n, Q_act=Q_act
                    )
                    pending[sp_i].append((f"2c_in|{vname}|{n_eff}", preds_v[sp_i], lm))

            # Output/Predicted-side ablations on `full_pred_cache`:
            # Also fit a pure query-image predictor `h_q -> z_perp(query)` (ZERO demo needed!)
            _, y_tr_perp = project_subspace(Y_tr, z_mu, Q_act)
            X_hq_tr, X_hq_ev = build_block_rms_features([hq_tr[:n_eff]], [hq_ev])
            z_perp_from_hq, _ = fit_predict_scaled_ridge(X_hq_tr, y_tr_perp, X_hq_ev, device)

            assert full_pred_cache is not None
            for sp_i in (0, 1):
                zp_par, zp_perp = project_subspace(full_pred_cache[sp_i], z_mu, Q_act)
                zq_par, zq_perp = project_subspace(zq_ev[sp_i], z_mu, Q_act)
                _, zd_p = project_subspace(zd_ev[sp_i], z_mu, Q_act)
                zp_rpar, zp_rperp = project_subspace(full_pred_cache[sp_i], z_mu, Q_rand)

                out_ablations = [
                    ("out_full", full_pred_cache[sp_i]),
                    ("out_par_only_zero_perp", z_mu + zp_par),
                    ("out_par_plus_demo_perp", z_mu + zp_par + zd_p),
                    ("out_par_plus_hq_perp", z_mu + zp_par + z_perp_from_hq[sp_i]),
                    ("out_par_plus_gt_perp", z_mu + zp_par + zq_perp),
                    ("out_gt_par_plus_pred_perp", z_mu + zq_par + zp_perp),
                    ("out_rand_par_only", z_mu + zp_rpar),
                    ("out_rand_perp_only", z_mu + zp_rperp),
                ]
                for vname, z_mod in out_ablations:
                    lm = compute_latent_predictability_metrics(
                        z_mod, zq_ev[sp_i], zd_ev[sp_i], mu_n, rms_n, Q_act=Q_act
                    )
                    pending[sp_i].append((f"2c_out|{vname}|{n_eff}", z_mod.astype(np.float32), lm))

        # Batched fixed-noise DDIM decoding for all pending predictions of this model!
        m_2a: Dict[str, Any] = {
            "subspace_stats": {
                "rank_995": float(r_act),
                "geo_subspace_var_frac": float(sub_info["geo_subspace_var_frac"]),
                "reg_r2_train": float(sub_info["reg_r2_train"]),
                "z_tr_rms": float(z_rms),
                "erank_z": float(erank_z["erank_energy"]),
                "d90_z": float(erank_z["d_90"]),
                "erank_dz": float(erank_dz["erank_energy"]),
                "d90_dz": float(erank_dz["d_90"]),
            },
            "direct_copy": {sp: {"mse10": float(dc_m[sp]["action10_mse"])} for sp in ("seen", "unseen")},
            "oracle": {sp: {"mse10": float(ora_m[sp]["action10_mse"])} for sp in ("seen", "unseen")},
            "adapters": {},
        }
        m_2c: Dict[str, Any] = {"gt_decoder_ablation": {}, "input_ablation": {}, "output_ablation": {}}

        sp_names = ("seen", "unseen")
        for sp_i, sp_name in enumerate(sp_names):
            items = pending[sp_i]
            decoded = ddim_decode_fixed_noise(
                pol, [it[1] for it in items], device, seed=12345, batch_size=1024
            )
            for (tag, _, lm), pred_a in zip(items, decoded):
                m_act = _action_metrics_dict(pred_a, gt_list[sp_i])
                packed = pack_action_metrics(m_act, dc_m[sp_name], ora_m[sp_name])
                packed.update(lm)
                parts = tag.split("|")
                if parts[0] == "2a":
                    adp, n_s = parts[1], parts[2]
                    m_2a["adapters"].setdefault(adp, {}).setdefault(n_s, {})[sp_name] = packed
                elif parts[0] == "2c_gt":
                    m_2c["gt_decoder_ablation"].setdefault(parts[1], {})[sp_name] = packed
                elif parts[0] == "2c_in":
                    m_2c["input_ablation"].setdefault(parts[2], {}).setdefault(parts[1], {})[sp_name] = packed
                elif parts[0] == "2c_out":
                    m_2c["output_ablation"].setdefault(parts[2], {}).setdefault(parts[1], {})[sp_name] = packed

        res_2a[mname] = m_2a
        res_2c[mname] = m_2c

    return res_2a, res_2c, icl_err_bank


# ---------------------------------------------------------------------------
# Part 2B: Decoder Sensitivity to Equal-Scale Latent Perturbations & Shrinkage
# ---------------------------------------------------------------------------

def _run_2b_decoder_sensitivity(
    icl_err_bank: Dict[str, Dict[str, Any]],
    z_train_bank: Dict[str, np.ndarray],
    policies: Dict[str, Any],
    device: torch.device,
    *,
    sigmas: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
    shrinkages: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50),
    n_noise_trials: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure exact Decoder sensitivity to equal-scale perturbations & shrinkage across all 4 models.

    Every perturbation type is strictly scaled by each model's own centered RMS norm
    `R_M = sqrt(mean(||z_tr - mu_M||_2^2))` so `sigma = ||eps||_2 / R_M` is 100% scale-free!
    """
    out: Dict[str, Any] = {}
    for mname in CORE_MODELS:
        pol = policies[mname]
        info = icl_err_bank[mname]
        z_q = info["z_true_unseen"]           # (N, 3072)
        z_pred_lin = info["z_pred_lin_full_unseen"]
        z_pred_krr = info["z_pred_krr_full_unseen"]
        z_mu = info["z_mu"]                   # (1, 3072)
        Q_act = info["Q_act"]                 # (3072, r)
        gt = info["gt_unseen"]                # (N, 16, 10)
        z_tr = z_train_bank[mname]            # (4000, 3072)
        N, D = z_q.shape

        # Reference RMS L2 norm of centered latent vector (exact scale normalizer!)
        zc_tr = (z_tr - z_mu).astype(np.float64)
        R_M = float(np.sqrt(np.mean(np.sum(zc_tr ** 2, axis=-1))))

        # Precompute SVD of zc_tr for exact covariance-matched sampling N(0, Sigma_M)
        # Sample eps = g @ diag(S / sqrt(n-1)) @ Vh
        zc_t = torch.from_numpy(zc_tr).to(device)
        _, S_tr, Vh_tr = torch.linalg.svd(zc_t, full_matrices=False)
        cov_scale = (S_tr / math.sqrt(max(len(zc_tr) - 1, 1))).cpu().numpy()
        Vh_np = Vh_tr.cpu().numpy()
        rank_cov = len(cov_scale)

        # Actual Visual-ICL error vectors
        e_lin = (z_pred_lin - z_q).astype(np.float64)
        e_krr = (z_pred_krr - z_q).astype(np.float64)
        e_lin_rel = float(np.sqrt(np.mean(np.sum(e_lin ** 2, axis=-1))) / max(R_M, 1e-6))
        e_krr_rel = float(np.sqrt(np.mean(np.sum(e_krr ** 2, axis=-1))) / max(R_M, 1e-6))

        # Helper to normalize perturbation batch to exact RMS L2 norm = sigma * R_M
        def _rescale_rms(eps: np.ndarray, sig: float) -> np.ndarray:
            cur = float(np.sqrt(np.mean(np.sum(eps.astype(np.float64) ** 2, axis=-1))))
            if cur < 1e-12 or sig == 0.0:
                return np.zeros_like(z_q, dtype=np.float32)
            return (eps * (sig * R_M / cur)).astype(np.float32)

        pending_tags: List[Tuple[str, float, int]] = []
        pending_z: List[np.ndarray] = []

        # 1. Clean baseline + actual ICL predictions
        pending_tags.append(("clean", 0.0, 0))
        pending_z.append(z_q.astype(np.float32))
        pending_tags.append(("actual_icl_lin", e_lin_rel, 0))
        pending_z.append(z_pred_lin.astype(np.float32))
        pending_tags.append(("actual_icl_krr", e_krr_rel, 0))
        pending_z.append(z_pred_krr.astype(np.float32))

        # 2. Synthetic equal-scale perturbations (multi-trial for smooth curves)
        for sig in sigmas:
            if sig == 0.0:
                continue
            for tr_i in range(n_noise_trials):
                rng = np.random.default_rng(seed + 10000 + tr_i * 100 + int(round(sig * 1000)))
                # (a) Isotropic Gaussian
                g_iso = rng.standard_normal((N, D))
                pending_tags.append(("isotropic", sig, tr_i))
                pending_z.append(z_q + _rescale_rms(g_iso, sig))

                # (b) Covariance-matched on-manifold Gaussian N(0, Sigma_M)
                g_cov = (rng.standard_normal((N, rank_cov)) * cov_scale[None, :]) @ Vh_np
                pending_tags.append(("cov_matched", sig, tr_i))
                pending_z.append(z_q + _rescale_rms(g_cov, sig))

                # (c) Action-explainable subspace (z_par) on-manifold perturbation
                g_par = (g_cov @ Q_act) @ Q_act.T
                pending_tags.append(("subspace_par", sig, tr_i))
                pending_z.append(z_q + _rescale_rms(g_par, sig))

                # (d) Action-unexplained subspace (z_perp) on-manifold perturbation
                g_perp = g_cov - g_par
                pending_tags.append(("subspace_perp", sig, tr_i))
                pending_z.append(z_q + _rescale_rms(g_perp, sig))

            # (e) Actual Visual-ICL error direction rescaled to equal sigma!
            pending_tags.append(("icl_err_rescaled", sig, 0))
            pending_z.append(z_q + _rescale_rms(e_lin, sig))

        # 3. Radial shrinkage sweep: z_s = z_mu + (1 - s) * (z_q - z_mu)
        for sh in shrinkages:
            if sh == 0.0:
                continue
            z_sh = z_mu + (1.0 - sh) * (z_q - z_mu)
            pending_tags.append(("shrinkage", sh, 0))
            pending_z.append(z_sh.astype(np.float32))

        # Batch decode all perturbed latents with IDENTICAL fixed DDIM noise z0!
        decoded_all = ddim_decode_fixed_noise(pol, pending_z, device, seed=12345, batch_size=1024)
        clean_act = decoded_all[0]
        clean_m = _action_metrics_dict(clean_act, gt)
        clean_mse10 = float(clean_m["action10_mse"])

        # Collect per-(ptype, level) metrics
        grouped: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        actual_lin_m = None
        actual_krr_m = None
        for (ptype, lvl, _), pred_a in zip(pending_tags, decoded_all):
            m_gt = _action_metrics_dict(pred_a, gt)
            # Direct action shift relative to clean decoded action (pure Jacobian sensitivity!)
            shift_mse10 = float(np.mean((pred_a - clean_act) ** 2))
            shift_rms = float(np.sqrt(max(shift_mse10, 0.0)))
            excess_mse10 = float(max(m_gt["action10_mse"] - clean_mse10, 0.0))
            gain_vs_clean = shift_rms / max(lvl, 1e-6) if lvl > 0 else 0.0
            rec = {
                "mse10_vs_gt": float(m_gt["action10_mse"]),
                "excess_mse10": excess_mse10,
                "shift_mse10_vs_clean": shift_mse10,
                "shift_rms_vs_clean": shift_rms,
                "jacobian_gain": float(gain_vs_clean),
                "grip_acc": float(m_gt["grip_sign_acc"]),
            }
            if ptype == "actual_icl_lin":
                actual_lin_m = rec
            elif ptype == "actual_icl_krr":
                actual_krr_m = rec
            elif ptype != "clean":
                grouped.setdefault(ptype, {}).setdefault(f"{lvl:.2f}", []).append(rec)

        curves: Dict[str, Dict[str, Dict[str, float]]] = {}
        for ptype, lvls in grouped.items():
            curves[ptype] = {"0.00": {
                "mse10_vs_gt_mean": clean_mse10, "mse10_vs_gt_std": 0.0,
                "excess_mse10_mean": 0.0, "excess_mse10_std": 0.0,
                "shift_mse10_vs_clean_mean": 0.0, "shift_mse10_vs_clean_std": 0.0,
                "shift_rms_vs_clean_mean": 0.0, "shift_rms_vs_clean_std": 0.0,
                "jacobian_gain_mean": 0.0, "jacobian_gain_std": 0.0,
                "grip_acc_mean": float(clean_m["grip_sign_acc"]), "grip_acc_std": 0.0,
            }}
            for l_str, recs in lvls.items():
                curves[ptype][l_str] = _agg(recs)

        # Exact multiplicative factorization on actual Visual ICL Linear & RBF-KRR errors:
        # sqrt(excess_mse10) == e_rel * G_dec_excess
        assert actual_lin_m is not None and actual_krr_m is not None
        excess_rms_lin = math.sqrt(max(actual_lin_m["excess_mse10"], 1e-12))
        excess_rms_krr = math.sqrt(max(actual_krr_m["excess_mse10"], 1e-12))

        out[mname] = {
            "rms_centered_l2_norm": R_M,
            "clean_mse10": clean_mse10,
            "clean_grip_acc": float(clean_m["grip_sign_acc"]),
            "factorization_linear": {
                "e_lat_rel": e_lin_rel,
                "excess_act_mse10": actual_lin_m["excess_mse10"],
                "excess_act_rms": excess_rms_lin,
                "G_dec_excess": float(excess_rms_lin / max(e_lin_rel, 1e-6)),
                "shift_rms_vs_clean": actual_lin_m["shift_rms_vs_clean"],
                "G_dec_shift": float(actual_lin_m["shift_rms_vs_clean"] / max(e_lin_rel, 1e-6)),
                "total_mse10": actual_lin_m["mse10_vs_gt"],
            },
            "factorization_rbf_krr": {
                "e_lat_rel": e_krr_rel,
                "excess_act_mse10": actual_krr_m["excess_mse10"],
                "excess_act_rms": excess_rms_krr,
                "G_dec_excess": float(excess_rms_krr / max(e_krr_rel, 1e-6)),
                "shift_rms_vs_clean": actual_krr_m["shift_rms_vs_clean"],
                "G_dec_shift": float(actual_krr_m["shift_rms_vs_clean"] / max(e_krr_rel, 1e-6)),
                "total_mse10": actual_krr_m["mse10_vs_gt"],
            },
            "curves": curves,
        }
    return out


# ---------------------------------------------------------------------------
# Main Runner & Multi-Seed Aggregation
# ---------------------------------------------------------------------------

def _agg_nested_2a(seed_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in CORE_MODELS:
        out[m] = {
            "subspace_stats": _agg([d[m]["subspace_stats"] for d in seed_list]),
            "direct_copy": {
                sp: _agg([d[m]["direct_copy"][sp] for d in seed_list]) for sp in ("seen", "unseen")
            },
            "oracle": {
                sp: _agg([d[m]["oracle"][sp] for d in seed_list]) for sp in ("seen", "unseen")
            },
            "adapters": {},
        }
        for adp in seed_list[0][m]["adapters"]:
            out[m]["adapters"][adp] = {}
            for n_s in seed_list[0][m]["adapters"][adp]:
                out[m]["adapters"][adp][n_s] = {
                    sp: _agg([d[m]["adapters"][adp][n_s][sp] for d in seed_list])
                    for sp in ("seen", "unseen")
                }
    return out


def _agg_nested_2c(seed_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in CORE_MODELS:
        out[m] = {"gt_decoder_ablation": {}, "input_ablation": {}, "output_ablation": {}}
        for v in seed_list[0][m]["gt_decoder_ablation"]:
            out[m]["gt_decoder_ablation"][v] = {
                sp: _agg([d[m]["gt_decoder_ablation"][v][sp] for d in seed_list])
                for sp in ("seen", "unseen")
            }
        for sec in ("input_ablation", "output_ablation"):
            for n_s in seed_list[0][m][sec]:
                out[m][sec][n_s] = {}
                for v in seed_list[0][m][sec][n_s]:
                    out[m][sec][n_s][v] = {
                        sp: _agg([d[m][sec][n_s][v][sp] for d in seed_list])
                        for sp in ("seen", "unseen")
                    }
    return out


def _agg_nested_2b(seed_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in CORE_MODELS:
        out[m] = {
            "rms_centered_l2_norm": float(np.mean([d[m]["rms_centered_l2_norm"] for d in seed_list])),
            "clean_mse10": float(np.mean([d[m]["clean_mse10"] for d in seed_list])),
            "clean_grip_acc": float(np.mean([d[m]["clean_grip_acc"] for d in seed_list])),
            "factorization_linear": _agg([d[m]["factorization_linear"] for d in seed_list]),
            "factorization_rbf_krr": _agg([d[m]["factorization_rbf_krr"] for d in seed_list]),
            "curves": {},
        }
        for ptype in seed_list[0][m]["curves"]:
            out[m]["curves"][ptype] = {}
            for lvl in seed_list[0][m]["curves"][ptype]:
                # Average across seeds of the per-seed means
                keys = [k[:-5] for k in seed_list[0][m]["curves"][ptype][lvl] if k.endswith("_mean")]
                lvl_dict = {}
                for k in keys:
                    vals = [d[m]["curves"][ptype][lvl][f"{k}_mean"] for d in seed_list]
                    lvl_dict[f"{k}_mean"] = float(np.mean(vals))
                    lvl_dict[f"{k}_std"] = float(np.std(vals))
                out[m]["curves"][ptype][lvl] = lvl_dict
    return out


def run_exp2(
    out_dir: Path,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    match_modes: Tuple[str, ...] = MATCH_MODES_DEFAULT,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    policies = load_policies(CORE_MODELS, device)

    results_by_mode: Dict[str, Any] = {}
    for match_mode in match_modes:
        s_2a, s_2b, s_2c = [], [], []
        for seed in seeds:
            ts = time.monotonic()
            print(f"[E14 Exp2] Running mode={match_mode} seed={seed}...", flush=True)
            bundle = build_seed_bundle(
                cache, policies, match_mode=match_mode, seed=seed, device=device, n_train_pairs=4000
            )
            r2a, r2c, err_bank = _run_2a_and_2c_for_bundle(bundle, policies, device)
            z_tr_bank = {m: bundle["latents"][m]["train"]["q"] for m in CORE_MODELS}
            r2b = _run_2b_decoder_sensitivity(err_bank, z_tr_bank, policies, device, seed=seed)

            s_2a.append(r2a)
            s_2b.append(r2b)
            s_2c.append(r2c)
            print(f"[E14 Exp2] Finished mode={match_mode} seed={seed} in {time.monotonic() - ts:.1f}s", flush=True)

        agg_2a = _agg_nested_2a(s_2a)
        agg_2b = _agg_nested_2b(s_2b)
        agg_2c = _agg_nested_2c(s_2c)
        results_by_mode[match_mode] = {"part_2a": agg_2a, "part_2b": agg_2b, "part_2c": agg_2c}

        # Print diagnostic highlights
        print(f"\n=== [E14 Exp2 | mode={match_mode}] Summary Highlights (Unseen, N=4000, 3 seeds) ===", flush=True)
        print("  [2A] Latent Predictability & Linear vs Nonlinear Adapter MSE10:")
        for m in CORE_MODELS:
            lf = agg_2a[m]["adapters"]["linear_full"]["4000"]["unseen"]
            kf = agg_2a[m]["adapters"]["rbf_krr_full"]["4000"]["unseen"]
            lp = agg_2a[m]["adapters"]["linear_pca"]["4000"]["unseen"]
            kp = agg_2a[m]["adapters"]["rbf_krr_pca"]["4000"]["unseen"]
            rm = agg_2a[m]["adapters"]["res_mlp_pca"]["4000"]["unseen"]
            print(
                f"    {m:18s} | R2(z)={lf['r2_z_mean']:.3f} R2(par)={lf['r2_z_par_mean']:.3f} R2(perp)={lf['r2_z_perp_mean']:.3f} "
                f"RelErr={lf['rel_rms_err_mean']:.3f} | "
                f"LinFull={lf['mse10_mean']:.5f} RBF-Full={kf['mse10_mean']:.5f} | "
                f"LinPCA={lp['mse10_mean']:.5f} RBF-PCA={kp['mse10_mean']:.5f} ResMLP={rm['mse10_mean']:.5f}",
                flush=True,
            )
        print("  [2B] Decoder Sensitivity & Exact Factorization (ExcessRMS = e_lat * G_dec):")
        for m in CORE_MODELS:
            fl = agg_2b[m]["factorization_linear"]
            c_iso = agg_2b[m]["curves"]["isotropic"]["0.35"]["shift_rms_vs_clean_mean"]
            c_cov = agg_2b[m]["curves"]["cov_matched"]["0.35"]["shift_rms_vs_clean_mean"]
            c_par = agg_2b[m]["curves"]["subspace_par"]["0.35"]["shift_rms_vs_clean_mean"]
            c_prp = agg_2b[m]["curves"]["subspace_perp"]["0.35"]["shift_rms_vs_clean_mean"]
            c_icl = agg_2b[m]["curves"]["icl_err_rescaled"]["0.35"]["shift_rms_vs_clean_mean"]
            c_sh = agg_2b[m]["curves"]["shrinkage"]["0.30"]["mse10_vs_gt_mean"]
            print(
                f"    {m:18s} | e_lat={fl['e_lat_rel_mean']:.4f} * G_dec={fl['G_dec_excess_mean']:.4f} "
                f"= ExcRMS={fl['excess_act_rms_mean']:.4f} | "
                f"@0.35 ShiftRMS(iso/cov/par/perp/icl)={c_iso:.4f}/{c_cov:.4f}/{c_par:.4f}/{c_prp:.4f}/{c_icl:.4f} | Shrink0.3={c_sh:.5f}",
                flush=True,
            )
        print("  [2C] E9 Subspace Ablations (Unseen, N=4000):")
        for k, v in agg_2c["e9"]["gt_decoder_ablation"].items():
            print(f"    GT-Dec  {k:28s}: MSE10={v['unseen']['mse10_mean']:.5f} Grip={v['unseen']['grip_acc_mean']*100:.1f}%", flush=True)
        for k, v in agg_2c["e9"]["input_ablation"]["4000"].items():
            print(f"    Input   {k:28s}: MSE10={v['unseen']['mse10_mean']:.5f} R2(z)={v['unseen']['r2_z_mean']:.3f}", flush=True)
        for k, v in agg_2c["e9"]["output_ablation"]["4000"].items():
            print(f"    Output  {k:28s}: MSE10={v['unseen']['mse10_mean']:.5f} Grip={v['unseen']['grip_acc_mean']*100:.1f}%", flush=True)

    summary = {
        "core_models": list(CORE_MODELS),
        "seeds": list(seeds),
        "match_modes": list(match_modes),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "results": results_by_mode,
    }
    atomic_write_json(summary, out_dir / "exp2_summary.json")
    print(f"[E14 Exp2] Saved summary to {out_dir / 'exp2_summary.json'} ({summary['elapsed_seconds']}s)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp2_mechanism_diagnosis")
    args = p.parse_args()
    run_exp2(args.out_dir)


if __name__ == "__main__":
    main()
