"""Unified comprehensive evaluation suite for E15: Robust Relational Behavior Representation Learning.

Reuses E12, E13, and E14 protocols with strict separation between Oracle Diagnostics and
Zero-Leakage Inference, saving a self-contained JSON per model so GPU 0 and GPU 1 can
evaluate models in parallel:

  1. Fidelity (Oracle self-reconstruction + fine-grained motion details: accel, jerk, grip switch)
  2. Editability (Oracle: Tier-2 Parallel Arithmetic, Op-Subspace-Only Arithmetic, Unseen
     Combination Ridge + Zero-Shot Compositional Arithmetic, Action-Change Linear Transfer
     @100/@1000, Midpoint Interpolation)
  3. Composability (Oracle: 5x5 Factor Token-Swap matrix + Unseen (dir x grip)/(dir x speed)
     zero-shot swaps + Op-Slice-Only Token-Swap for E15-C models)
  4. Predictability & Content (3-seed disjoint-task probes: R^2(a->z), R^2(z->a7), Fut_R2,
     VisHQ_R2, ObjAcc, erank(z/dz), 3-component energy audit, plus z_op vs z_res split probes
     & CKA for E15-C models)
  5. Decoder Robustness (3-seed E14 Exp 2B exact factorization ExcessRMS = e_lat * G_dec,
     equal-scale perturbation curves across iso/cov/par/perp/icl_dir at sigma=0..0.50,
     radial shrinkage 0.30, GT z_par_only & shuf_perp fragility)
  6. Downstream Visual ICL Verification (Strictly Zero-Leakage, 3 seeds, visual_knn &
     rel_progress, unseen tasks: Linear Full Ridge @100/1000/4000, RBF-KRR Full @4000,
     Vision-Only @4000, and out_par_plus_hq_perp @4000)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import action_10d_to_7d, action_7d_to_10d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _effective_rank_stats,
    _ridge_fit_predict,
    ddim_decode_batched,
    encode_all_val_starts,
    evaluate_suite3_displacement_and_arithmetic,
    evaluate_suite4_unseen_combinations,
    evaluate_suite5_transfer_efficiency,
)
from e13.exp1_training_free import _build_fixed_analogy_quadruples
from e13.exp2_action_baselines import _gather_future_actions
from e13.exp3_factor_composition import (
    ATTR_ORDER,
    TOKEN_FOR_ATTR,
    _select_swap_pairs,
    evaluate_model_factor_composition,
)
from e14.common import (
    build_block_rms_features,
    build_seed_bundle,
    compute_latent_predictability_metrics,
    ddim_decode_fixed_noise,
    fit_action_explainable_subspace,
    fit_global_rms_standardizer,
    fit_predict_rbf_krr,
    fit_predict_scaled_ridge,
    pack_action_metrics,
    project_subspace,
)

from . import OUTPUT_ROOT, SEEDS_DEFAULT
from .exp1_three_component import (
    compute_motion_detail_metrics,
    decompose_three_components,
    fit_three_component_model,
)
from .train_e15 import compute_linear_cka_penalty


def _agg(dlist: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in dlist[0]:
        vals = [float(d[k]) for d in dlist]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def _eval_fidelity_and_motion(
    policy, encoded: Dict[str, Any], device: torch.device, ddim_samples: int = 1024
) -> Dict[str, Any]:
    nt = np.flatnonzero(~encoded["is_tail"])
    rng = np.random.default_rng(42 + 101)
    sel = np.sort(rng.choice(nt, size=min(ddim_samples, len(nt)), replace=False))
    z_sel = encoded["latents_flat"][sel]
    gt_sel = encoded["actions_10d"][sel]
    pred = ddim_decode_fixed_noise(policy, [z_sel], device, seed=12345, batch_size=512)[0]
    md = compute_motion_detail_metrics(pred, gt_sel)
    return {"num_samples": int(len(sel)), "self_ddim": md}


def _eval_op_subspace_editability(
    policy,
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    op_dim: int = 384,
) -> Dict[str, Any]:
    """Evaluate Tier-2 Parallel Arithmetic restricting the edit strictly to `z_op` (leaving `z_res` of B1 untouched)."""
    quads = _build_fixed_analogy_quadruples(cache, encoded, device, seed=42)
    A1, A2, B1, B2 = quads["tier2"]
    z4 = encoded["latents"]  # (N, 4, 768)
    gt_B2 = encoded["actions_10d"][B2]

    z_full_arith = (z4[B1] + (z4[A2] - z4[A1])).reshape(len(B1), -1)
    z_op_arith = z4[B1].copy()
    z_op_arith[:, :, :op_dim] = z4[B1, :, :op_dim] + (z4[A2, :, :op_dim] - z4[A1, :, :op_dim])
    z_op_arith_f = z_op_arith.reshape(len(B1), -1)

    z_res_arith = z4[B1].copy()
    z_res_arith[:, :, op_dim:] = z4[B1, :, op_dim:] + (z4[A2, :, op_dim:] - z4[A1, :, op_dim:])
    z_res_arith_f = z_res_arith.reshape(len(B1), -1)

    p_ora, p_bas, p_full, p_op, p_res = ddim_decode_fixed_noise(
        policy,
        [z4[B2].reshape(len(B1), -1), z4[B1].reshape(len(B1), -1), z_full_arith, z_op_arith_f, z_res_arith_f],
        device,
        seed=12345,
        batch_size=512,
    )
    m_ora = _action_metrics_dict(p_ora, gt_B2)
    m_bas = _action_metrics_dict(p_bas, gt_B2)

    def _pack(p_a: np.ndarray) -> Dict[str, float]:
        m = _action_metrics_dict(p_a, gt_B2)
        arr = (m_bas["action10_mse"] - m["action10_mse"]) / max(m_bas["action10_mse"] - m_ora["action10_mse"], 1e-9)
        return {
            "mse10": float(m["action10_mse"]),
            "mae7": float(m["action7_mae"]),
            "grip_acc": float(m["grip_sign_acc"]),
            "arr_10d": float(arr),
        }

    # Displacement cosine specifically inside z_op and z_res
    dzA_op = (z4[A2, :, :op_dim] - z4[A1, :, :op_dim]).reshape(len(B1), -1)
    dzB_op = (z4[B2, :, :op_dim] - z4[B1, :, :op_dim]).reshape(len(B1), -1)
    dzA_rs = (z4[A2, :, op_dim:] - z4[A1, :, op_dim:]).reshape(len(B1), -1)
    dzB_rs = (z4[B2, :, op_dim:] - z4[B1, :, op_dim:]).reshape(len(B1), -1)

    def _cos(u, v):
        nu = np.linalg.norm(u, axis=-1)
        nv = np.linalg.norm(v, axis=-1)
        ok = (nu > 1e-6) & (nv > 1e-6)
        return float(np.mean(np.sum(u[ok] * v[ok], axis=-1) / (nu[ok] * nv[ok]))) if ok.any() else 0.0

    # Unseen-combination linear compositional Ridge using ONLY z_op (preserving z_res=mean? Or predicting full z)
    return {
        "op_dim": int(op_dim),
        "disp_cos_op": _cos(dzA_op, dzB_op),
        "disp_cos_res": _cos(dzA_rs, dzB_rs),
        "full_arith": _pack(p_full),
        "op_only_arith": _pack(p_op),
        "res_only_arith": _pack(p_res),
    }


def _eval_content_probes(
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    *,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    op_dim: int = 384,
) -> Dict[str, Any]:
    """3-seed disjoint-task behavior content probes (exact E13 Exp 2 + z_op vs z_res split)."""
    non_tail = np.flatnonzero(~encoded["is_tail"])
    fut_a10, fut_valid = _gather_future_actions(cache, horizon_offset=16)
    fut_a7 = action_10d_to_7d(fut_a10)

    procs = [cache.episodes[int(e)]["procedure"].split("|")[0].strip() for e in cache.val_episodes]
    obj_labels = np.array([(p.split()[1] if len(p.split()) >= 2 else "other") for p in procs], dtype=object)
    top_objs = sorted({o for o in obj_labels if (obj_labels == o).sum() >= 100})
    obj_to_id = {o: i for i, o in enumerate(top_objs)}
    obj_y = np.array([obj_to_id.get(o, -1) for o in obj_labels], dtype=np.int64)
    obj_oh = np.zeros((len(obj_y), len(top_objs)), dtype=np.float32)
    for i, oi in enumerate(obj_y):
        if oi >= 0:
            obj_oh[i, oi] = 1.0

    A10 = encoded["actions_10d"]
    A7_f = encoded["actions_7d"].reshape(len(A10), -1)
    Fut7_f = fut_a7.reshape(len(A10), -1)
    HQ_f = encoded["h_queries"].reshape(len(A10), -1)
    Z_full = encoded["latents_flat"]
    Z4 = encoded["latents"]
    Z_op = Z4[:, :, :op_dim].reshape(len(A10), -1)
    Z_res = Z4[:, :, op_dim:].reshape(len(A10), -1)

    unique_t = np.unique(cache.val_task_ids[non_tail])
    valid_nt = non_tail[fut_valid[non_tail]]
    v_tasks = cache.val_task_ids[valid_nt]

    full_seeds, op_seeds, res_seeds, three_seeds = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed + 222)
        perm_t = rng.permutation(unique_t)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        tr_i = valid_nt[np.array([int(t) in tr_set for t in v_tasks])]
        te_i = valid_nt[np.array([int(t) not in tr_set for t in v_tasks])]

        var_a7 = float(((A7_f[te_i] - A7_f[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        var_fut = float(((Fut7_f[te_i] - Fut7_f[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        var_hq = float(((HQ_f[te_i] - HQ_f[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        obj_tr_m = obj_y[tr_i] >= 0
        obj_te_m = obj_y[te_i] >= 0

        def _probe_rep(R: np.ndarray) -> Dict[str, float]:
            R_tr, R_te = R[tr_i], R[te_i]
            var_R_te = float(((R_te - R_te.mean(axis=0, keepdims=True)) ** 2).mean())
            R_from_a_tr, al_az = _ridge_fit_predict(A10[tr_i].reshape(len(tr_i), -1), R_tr, A10[tr_i].reshape(len(tr_i), -1), device=device)
            R_from_a_te, _ = _ridge_fit_predict(A10[tr_i].reshape(len(tr_i), -1), R_tr, A10[te_i].reshape(len(te_i), -1), alphas=(al_az,), device=device)
            r2_az = float(1.0 - ((R_from_a_te - R_te) ** 2).mean() / max(var_R_te, 1e-12))
            z_res_tr = R_tr - R_from_a_tr
            z_res_te = R_te - R_from_a_te

            p_a7, _ = _ridge_fit_predict(R_tr, A7_f[tr_i], R_te, device=device)
            p_fut, _ = _ridge_fit_predict(R_tr, Fut7_f[tr_i], R_te, device=device)
            p_fut_res, _ = _ridge_fit_predict(z_res_tr, Fut7_f[tr_i], z_res_te, device=device)
            p_hq, _ = _ridge_fit_predict(R_tr, HQ_f[tr_i], R_te, device=device)
            p_obj, _ = _ridge_fit_predict(R_tr[obj_tr_m], obj_oh[tr_i[obj_tr_m]], R_te[obj_te_m], device=device)
            er = _effective_rank_stats(R_te, device=device)
            return {
                "r2_a_to_z": r2_az,
                "non_action_var_frac": float(max(0.0, 1.0 - r2_az)),
                "r2_z_to_a7": float(1.0 - ((p_a7 - A7_f[te_i]) ** 2).mean() / max(var_a7, 1e-12)),
                "fut_r2": float(1.0 - ((p_fut - Fut7_f[te_i]) ** 2).mean() / max(var_fut, 1e-12)),
                "fut_r2_nonact": float(1.0 - ((p_fut_res - Fut7_f[te_i]) ** 2).mean() / max(var_fut, 1e-12)),
                "vis_hq_r2": float(1.0 - ((p_hq - HQ_f[te_i]) ** 2).mean() / max(var_hq, 1e-12)),
                "obj_acc": float((np.argmax(p_obj, axis=-1) == obj_y[te_i[obj_te_m]]).mean()),
                "erank": float(er["erank_energy"]),
                "d_90": float(er["d_90"]),
            }

        full_seeds.append(_probe_rep(Z_full))
        op_seeds.append(_probe_rep(Z_op))
        res_seeds.append(_probe_rep(Z_res))

        # Exact 3-component energy audit on this disjoint split
        m3 = fit_three_component_model(A10[tr_i], Z_full[tr_i], device, svd_threshold=1.0)
        d3 = decompose_three_components(A10[te_i], Z_full[te_i], m3)
        st3 = dict(d3["stats"])
        # Also CKA & energy ratio between z_op and z_res
        zo_t = torch.from_numpy(Z_op[te_i]).to(device)
        zr_t = torch.from_numpy(Z_res[te_i]).to(device)
        st3["cka_op_res"] = float(compute_linear_cka_penalty(zo_t, zr_t).item())
        eo = float(((Z_op[te_i] - Z_op[te_i].mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean())
        er_e = float(((Z_res[te_i] - Z_res[te_i].mean(axis=0, keepdims=True)) ** 2).sum(axis=-1).mean())
        st3["res_slice_energy_frac"] = float(er_e / max(eo + er_e, 1e-12))
        three_seeds.append(st3)

    return {
        "full": _agg(full_seeds),
        "op_slice": _agg(op_seeds),
        "res_slice": _agg(res_seeds),
        "three_component_energy": _agg(three_seeds),
    }


def _eval_robustness_and_visual_icl(
    model_key: str,
    policy,
    cache: ValClipCache,
    device: torch.device,
    *,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    match_modes: Tuple[str, ...] = ("visual_knn", "rel_progress"),
    budgets: Tuple[int, ...] = (100, 1000, 4000),
    sigmas: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50),
) -> Dict[str, Any]:
    """Exact E14 Exp 2A/2B/2C robustness + strictly zero-leakage Visual ICL downstream verification."""
    policies = {model_key: policy}
    out_modes: Dict[str, Any] = {}

    for match_mode in match_modes:
        icl_seed_list: List[Dict[str, Any]] = []
        rob_seed_list: List[Dict[str, Any]] = []
        for seed in seeds:
            bundle = build_seed_bundle(
                cache, policies, match_mode=match_mode, seed=seed, device=device, n_train_pairs=4000
            )
            obs = bundle["obs"]
            acts = bundle["acts"]
            lats = bundle["latents"][model_key]

            dh_tr, hq_tr = obs["train"]["dh"], obs["train"]["h_q"]
            dh_u, hq_u = obs["test_unseen"]["dh"], obs["test_unseen"]["h_q"]
            zd_tr, zq_tr = lats["train"]["d"], lats["train"]["q"]
            zd_u, zq_u = lats["test_unseen"]["d"], lats["test_unseen"]["q"]
            gt_u = acts["test_unseen"]["q"]

            # Strictly train-fitted Q_act (full-rank exact AND 0.995 matching E14!)
            sub_info = fit_action_explainable_subspace(acts["train"]["q"], zq_tr, device, var_threshold=0.995)
            Q_act = sub_info["Q_act"]
            z_mu = sub_info["z_mu"]
            z_rms = sub_info["z_tr_rms"]
            er_z = _effective_rank_stats(zq_tr, device=device)
            er_dz = _effective_rank_stats(zq_tr - zd_tr, device=device)

            zc_tr = (zq_tr - z_mu).astype(np.float64)
            R_M = float(np.sqrt(np.mean(np.sum(zc_tr ** 2, axis=-1))))

            # Precompute cov SVD on train for equal-scale robustness curves
            zc_t = torch.from_numpy(zc_tr).to(device)
            _, S_tr, Vh_tr = torch.linalg.svd(zc_t, full_matrices=False)
            cov_scale = (S_tr / math.sqrt(max(len(zc_tr) - 1, 1))).cpu().numpy()
            Vh_np = Vh_tr.cpu().numpy()
            rank_cov = len(cov_scale)

            # Build ALL pending predictions + perturbations for ONE batched fixed-noise DDIM decode!
            pending_tags: List[str] = []
            pending_z: List[np.ndarray] = []
            pending_lm: List[Dict[str, float]] = []

            pending_tags.extend(["base|direct_copy", "base|oracle"])
            pending_z.extend([zd_u, zq_u])
            pending_lm.extend([{}, {}])

            z_pred_lin_4k = None
            z_pred_krr_4k = None

            for n_b in budgets:
                n_eff = min(int(n_b), len(dh_tr))
                Y_tr = zq_tr[:n_eff]
                mu_n, rms_n = fit_global_rms_standardizer(Y_tr)
                X_tr_f, X_ev_f = build_block_rms_features(
                    [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr[:n_eff]],
                    [[dh_u], [hq_u], [zd_u]],
                )
                p_lin = fit_predict_scaled_ridge(X_tr_f, Y_tr, X_ev_f, device)[0][0]
                lm_lin = compute_latent_predictability_metrics(p_lin, zq_u, zd_u, mu_n, rms_n, Q_act=Q_act)
                pending_tags.append(f"icl|lin_full|{n_eff}")
                pending_z.append(p_lin)
                pending_lm.append(lm_lin)

                if n_eff == max(budgets):
                    z_pred_lin_4k = p_lin
                    # RBF-KRR full @4000
                    p_krr = fit_predict_rbf_krr(X_tr_f, Y_tr, X_ev_f, device)[0][0]
                    z_pred_krr_4k = p_krr
                    lm_krr = compute_latent_predictability_metrics(p_krr, zq_u, zd_u, mu_n, rms_n, Q_act=Q_act)
                    pending_tags.append(f"icl|rbf_krr_full|{n_eff}")
                    pending_z.append(p_krr)
                    pending_lm.append(lm_krr)

                    # Vision-Only @4000
                    X_vo_tr, X_vo_ev = build_block_rms_features([dh_tr, hq_tr], [[dh_u], [hq_u]])
                    p_vo = fit_predict_scaled_ridge(X_vo_tr, Y_tr, X_vo_ev, device)[0][0]
                    lm_vo = compute_latent_predictability_metrics(p_vo, zq_u, zd_u, mu_n, rms_n, Q_act=Q_act)
                    pending_tags.append(f"icl|vision_only|{n_eff}")
                    pending_z.append(p_vo)
                    pending_lm.append(lm_vo)

                    # Zero-Leakage out_par_plus_hq_perp & out_par_only @4000
                    _, y_tr_perp = project_subspace(Y_tr, z_mu, Q_act)
                    X_hq_tr, X_hq_ev = build_block_rms_features([hq_tr], [[hq_u]])
                    z_perp_hq = fit_predict_scaled_ridge(X_hq_tr, y_tr_perp, X_hq_ev, device)[0][0]
                    zp_par, _ = project_subspace(p_lin, z_mu, Q_act)
                    pending_tags.append("icl|out_par_only|4000")
                    pending_z.append((z_mu + zp_par).astype(np.float32))
                    pending_lm.append({})
                    pending_tags.append("icl|out_par_plus_hq_perp|4000")
                    pending_z.append((z_mu + zp_par + z_perp_hq).astype(np.float32))
                    pending_lm.append({})

            assert z_pred_lin_4k is not None and z_pred_krr_4k is not None
            e_lin = (z_pred_lin_4k - zq_u).astype(np.float64)
            e_krr = (z_pred_krr_4k - zq_u).astype(np.float64)
            e_lin_rel = float(np.sqrt(np.mean(np.sum(e_lin ** 2, axis=-1))) / max(R_M, 1e-6))
            e_krr_rel = float(np.sqrt(np.mean(np.sum(e_krr ** 2, axis=-1))) / max(R_M, 1e-6))

            def _rescale(eps: np.ndarray, sig: float) -> np.ndarray:
                cur = float(np.sqrt(np.mean(np.sum(eps.astype(np.float64) ** 2, axis=-1))))
                return (eps * (sig * R_M / max(cur, 1e-12))).astype(np.float32)

            # GT subspace fragility (par_only, shuf_perp) + radial shrinkage 0.30
            zq_par, zq_perp = project_subspace(zq_u, z_mu, Q_act)
            perm = np.random.default_rng(seed + 901).permutation(len(zq_u))
            pending_tags.extend(["rob|gt_par_only|0.00", "rob|gt_shuf_perp|0.00", "rob|shrinkage|0.30"])
            pending_z.extend([
                (z_mu + zq_par).astype(np.float32),
                (z_mu + zq_par + zq_perp[perm]).astype(np.float32),
                (z_mu + 0.70 * (zq_u - z_mu)).astype(np.float32),
            ])
            pending_lm.extend([{}, {}, {}])

            # Equal-scale synthetic + ICL-dir perturbation curves (2 trials per sigma for low variance & speed)
            N_u, D_u = zq_u.shape
            for sig in sigmas:
                for tr_i in range(2):
                    rng_n = np.random.default_rng(seed + 10000 + tr_i * 100 + int(round(sig * 1000)))
                    g_iso = rng_n.standard_normal((N_u, D_u))
                    g_cov = (rng_n.standard_normal((N_u, rank_cov)) * cov_scale[None, :]) @ Vh_np
                    g_par = (g_cov @ Q_act) @ Q_act.T
                    g_perp = g_cov - g_par
                    for ptype, vec in [
                        ("isotropic", g_iso),
                        ("cov_matched", g_cov),
                        ("subspace_par", g_par),
                        ("subspace_perp", g_perp),
                    ]:
                        pending_tags.append(f"rob|{ptype}|{sig:.2f}")
                        pending_z.append((zq_u + _rescale(vec, sig)).astype(np.float32))
                        pending_lm.append({})
                pending_tags.append(f"rob|icl_err_rescaled|{sig:.2f}")
                pending_z.append((zq_u + _rescale(e_lin, sig)).astype(np.float32))
                pending_lm.append({})

            decoded = ddim_decode_fixed_noise(policy, pending_z, device, seed=12345, batch_size=1024)
            dc_m = _action_metrics_dict(decoded[0], gt_u)
            ora_m = _action_metrics_dict(decoded[1], gt_u)
            clean_act = decoded[1]
            clean_mse10 = float(ora_m["action10_mse"])

            icl_rec: Dict[str, Any] = {
                "meta": {
                    "erank_z": float(er_z["erank_energy"]),
                    "erank_dz": float(er_dz["erank_energy"]),
                    "geo_subspace_var_frac": float(sub_info["geo_subspace_var_frac"]),
                    "z_perp_var_frac": float(1.0 - sub_info["geo_subspace_var_frac"]),
                    "R_M": R_M,
                    "direct_copy_mse10": float(dc_m["action10_mse"]),
                    "oracle_mse10": clean_mse10,
                },
                "adapters": {},
            }
            rob_grouped: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
            lin_4k_act = None
            krr_4k_act = None

            for tag, lm, pa in zip(pending_tags, pending_lm, decoded):
                parts = tag.split("|")
                if parts[0] == "base":
                    continue
                ma = _action_metrics_dict(pa, gt_u)
                shift_mse = float(np.mean((pa - clean_act) ** 2))
                shift_rms = float(math.sqrt(max(shift_mse, 0.0)))
                exc_mse = float(max(ma["action10_mse"] - clean_mse10, 0.0))
                if parts[0] == "icl":
                    packed = pack_action_metrics(ma, dc_m, ora_m)
                    packed.update(lm)
                    icl_rec["adapters"].setdefault(parts[1], {})[parts[2]] = packed
                    if parts[1] == "lin_full" and parts[2] == str(max(budgets)):
                        lin_4k_act = (ma, shift_rms, exc_mse)
                    elif parts[1] == "rbf_krr_full" and parts[2] == str(max(budgets)):
                        krr_4k_act = (ma, shift_rms, exc_mse)
                elif parts[0] == "rob":
                    rec = {
                        "mse10_vs_gt": float(ma["action10_mse"]),
                        "excess_mse10": exc_mse,
                        "shift_rms_vs_clean": shift_rms,
                        "grip_acc": float(ma["grip_sign_acc"]),
                    }
                    rob_grouped.setdefault(parts[1], {}).setdefault(parts[2], []).append(rec)

            assert lin_4k_act is not None and krr_4k_act is not None
            exc_rms_lin = math.sqrt(max(lin_4k_act[2], 1e-12))
            exc_rms_krr = math.sqrt(max(krr_4k_act[2], 1e-12))
            rob_rec: Dict[str, Any] = {
                "factorization_linear": {
                    "e_lat_rel": e_lin_rel,
                    "excess_act_mse10": lin_4k_act[2],
                    "excess_act_rms": exc_rms_lin,
                    "G_dec_excess": float(exc_rms_lin / max(e_lin_rel, 1e-6)),
                    "shift_rms_vs_clean": lin_4k_act[1],
                    "G_dec_shift": float(lin_4k_act[1] / max(e_lin_rel, 1e-6)),
                    "total_mse10": float(lin_4k_act[0]["action10_mse"]),
                },
                "factorization_rbf_krr": {
                    "e_lat_rel": e_krr_rel,
                    "excess_act_mse10": krr_4k_act[2],
                    "excess_act_rms": exc_rms_krr,
                    "G_dec_excess": float(exc_rms_krr / max(e_krr_rel, 1e-6)),
                    "shift_rms_vs_clean": krr_4k_act[1],
                    "G_dec_shift": float(krr_4k_act[1] / max(e_krr_rel, 1e-6)),
                    "total_mse10": float(krr_4k_act[0]["action10_mse"]),
                },
                "curves": {
                    pt: {lvl: _agg(recs) for lvl, recs in lvls.items()}
                    for pt, lvls in rob_grouped.items()
                },
            }
            icl_seed_list.append(icl_rec)
            rob_seed_list.append(rob_rec)

        # Aggregate across seeds
        agg_icl = {
            "meta": _agg([s["meta"] for s in icl_seed_list]),
            "adapters": {
                adp: {
                    nb: _agg([s["adapters"][adp][nb] for s in icl_seed_list])
                    for nb in icl_seed_list[0]["adapters"][adp]
                }
                for adp in icl_seed_list[0]["adapters"]
            },
        }
        agg_rob = {
            "factorization_linear": _agg([s["factorization_linear"] for s in rob_seed_list]),
            "factorization_rbf_krr": _agg([s["factorization_rbf_krr"] for s in rob_seed_list]),
            "curves": {},
        }
        for pt in rob_seed_list[0]["curves"]:
            agg_rob["curves"][pt] = {}
            for lvl in rob_seed_list[0]["curves"][pt]:
                keys = [k[:-5] for k in rob_seed_list[0]["curves"][pt][lvl] if k.endswith("_mean")]
                agg_rob["curves"][pt][lvl] = {
                    f"{k}_mean": float(np.mean([s["curves"][pt][lvl][f"{k}_mean"] for s in rob_seed_list]))
                    for k in keys
                }
        out_modes[match_mode] = {"visual_icl": agg_icl, "decoder_robustness": agg_rob}
    return out_modes


def evaluate_single_model(
    model_key: str,
    ckpt_path: Path,
    out_dir: Path,
    device: torch.device,
    *,
    op_dim: int = 384,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[E15 Eval:{model_key}] Loading {ckpt_path} on {device}...", flush=True)

    cache = ValClipCache(seed=42)
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=True)

    # 1. Fidelity & Motion Details
    fid = _eval_fidelity_and_motion(policy, encoded, device, ddim_samples=1024)
    print(
        f"[E15 Eval:{model_key}] 1. Fidelity: MSE10={fid['self_ddim']['mse10']:.5f} "
        f"MAE7={fid['self_ddim']['mae7']:.5f} Grip={fid['self_ddim']['grip_acc']*100:.2f}% "
        f"Accel={fid['self_ddim']['accel_pos_mse']:.5f}",
        flush=True,
    )

    # 2. Editability (Oracle: E12 Suite 3, Suite 4, Suite 5, plus Op-Subspace Arithmetic & Midpoint)
    s3 = evaluate_suite3_displacement_and_arithmetic(policy, cache, encoded, device, seed=42, use_bf16=True)
    s4 = evaluate_suite4_unseen_combinations(policy, cache, encoded, device, seed=42, use_bf16=True)
    s5 = evaluate_suite5_transfer_efficiency(
        policy, cache, encoded, device, sample_sizes=(100, 1000), test_pairs_count=256, seed=42, use_bf16=True
    )
    op_ed = _eval_op_subspace_editability(policy, cache, encoded, device, op_dim=op_dim)

    # Midpoint interpolation
    nt = np.flatnonzero(~encoded["is_tail"])
    mid_idx = np.random.default_rng(42 + 999).choice(nt, size=(256, 2), replace=False)
    ia_m, ib_m = mid_idx[:, 0], mid_idx[:, 1]
    a7_mid_gt = 0.5 * (encoded["actions_7d"][ia_m] + encoded["actions_7d"][ib_m])
    a10_mid_gt = action_7d_to_10d(a7_mid_gt)
    z_mid = 0.5 * (encoded["latents_flat"][ia_m] + encoded["latents_flat"][ib_m])
    pred_mid = ddim_decode_fixed_noise(policy, [z_mid], device, seed=12345, batch_size=256)[0]
    m_mid = _action_metrics_dict(pred_mid, a10_mid_gt)

    t2 = s3["latent_arithmetic_tier2_distinct_start"]
    u4 = s4["linear_compositional_ridge"]
    print(
        f"[E15 Eval:{model_key}] 2. Editability (Oracle): "
        f"Tier2={t2['ddim_arith']['action10_mse']:.5f}(ARR={t2['arithmetic_recovery_ratio_10d']*100:.1f}%, cos={t2['latent_disp_cos']:.3f}) | "
        f"OpOnlyArith={op_ed['op_only_arith']['mse10']:.5f}(ARR={op_ed['op_only_arith']['arr_10d']*100:.1f}%) | "
        f"UnseenRidge={u4['unseen_ddim']['action10_mse']:.5f} | "
        f"ActLin@100={s5['action_change_transfer_efficiency']['by_n']['100']['linear_ridge']['ddim_10d_mse']:.5f}",
        flush=True,
    )

    # 3. Composability (Oracle: 5x5 Factor Token Swap)
    pairs = _select_swap_pairs(
        encoded["actions_10d"][nt], encoded["actions_7d"][nt], cache.val_task_ids[nt], num_pairs=512, seed=42
    )
    comp = evaluate_model_factor_composition(model_key, ckpt_path, cache, pairs, device, seed=42)
    print(
        f"[E15 Eval:{model_key}] 3. Composability (Oracle): "
        f"Tok0={comp['matrix_5x5']['translation']['target_adoption']*100:.1f}% "
        f"Tok1={comp['matrix_5x5']['rotation']['target_adoption']*100:.1f}% "
        f"Tok2Mag={comp['matrix_5x5']['magnitude']['target_adoption']*100:.1f}% "
        f"Tok3Grip={comp['matrix_5x5']['gripper']['target_adoption']*100:.1f}% "
        f"UnseenDG_Tok3={comp['unseen_combo_swaps']['unseen_dir_x_grip_tok3_swap']['arr_10d']*100:.1f}%",
        flush=True,
    )

    # 4. Predictability & Content Probes (3 seeds)
    probes = _eval_content_probes(cache, encoded, device, seeds=seeds, op_dim=op_dim)
    pf = probes["full"]
    print(
        f"[E15 Eval:{model_key}] 4. Content Probes: "
        f"R2(a->z)={pf['r2_a_to_z_mean']*100:.1f}% R2(z->a7)={pf['r2_z_to_a7_mean']:.3f} "
        f"Fut_R2={pf['fut_r2_mean']:.3f} VisHQ_R2={pf['vis_hq_r2_mean']:.3f} "
        f"ObjAcc={pf['obj_acc_mean']*100:.1f}% erank={pf['erank_mean']:.1f} "
        f"(op/res CKA={probes['three_component_energy']['cka_op_res_mean']:.3f})",
        flush=True,
    )

    # 5 & 6. Decoder Robustness + Zero-Leakage Visual ICL Downstream Verification (3 seeds, 2 modes)
    icl_rob = _eval_robustness_and_visual_icl(model_key, policy, cache, device, seeds=seeds)
    for mm in ("visual_knn", "rel_progress"):
        fl = icl_rob[mm]["decoder_robustness"]["factorization_linear"]
        lf = icl_rob[mm]["visual_icl"]["adapters"]["lin_full"]["4000"]
        kf = icl_rob[mm]["visual_icl"]["adapters"]["rbf_krr_full"]["4000"]
        c_prp = icl_rob[mm]["decoder_robustness"]["curves"]["subspace_perp"]["0.35"]["shift_rms_vs_clean_mean"]
        c_icl = icl_rob[mm]["decoder_robustness"]["curves"]["icl_err_rescaled"]["0.35"]["shift_rms_vs_clean_mean"]
        print(
            f"[E15 Eval:{model_key}] 5&6. [{mm}] Zero-Leakage Visual ICL & Robustness: "
            f"Lin@4k={lf['mse10_mean']:.5f}(ARR={lf['arr_10d_mean']*100:.1f}%) "
            f"RBF@4k={kf['mse10_mean']:.5f} | R2(z)={lf['r2_z_mean']:.3f} cos(dz)={lf['cos_dz_mean']:.3f} | "
            f"e_lat={fl['e_lat_rel_mean']:.4f} * G_dec={fl['G_dec_excess_mean']:.4f} = ExcRMS={fl['excess_act_rms_mean']:.4f} | "
            f"@0.35 perp/icl={c_prp:.4f}/{c_icl:.4f}",
            flush=True,
        )

    summary = {
        "model_key": model_key,
        "checkpoint": str(ckpt_path),
        "op_dim": int(op_dim),
        "seeds": list(seeds),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "fidelity": fid,
        "editability": {
            "displacement_and_arithmetic": s3,
            "unseen_combinations": s4,
            "transfer_efficiency": s5,
            "op_subspace_editability": op_ed,
            "midpoint_interpolation": {
                "mse10": float(m_mid["action10_mse"]),
                "pos_mse": float(m_mid["pos_mse"]),
                "mae7": float(m_mid["action7_mae"]),
            },
        },
        "composability": comp,
        "content_probes": probes,
        "robustness_and_visual_icl": icl_rob,
    }
    out_file = out_dir / "eval_e15.json"
    atomic_write_json(summary, out_file)
    print(f"[E15 Eval:{model_key}] Saved full evaluation to {out_file} ({summary['elapsed_seconds']}s)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", type=str, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--op-dim", type=int, default=384)
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else (OUTPUT_ROOT / "eval" / args.model_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_single_model(args.model_key, args.ckpt, out_dir, device, op_dim=args.op_dim)


if __name__ == "__main__":
    main()
