"""E13 Experiment 1: Training-Free Post-Processing Baselines on Frozen E9.

Tests whether simple linear post-processing of frozen E9 latents can match E12-C's
transfer and editing performance:
  1. E9 (Raw)
  2. E9 + Centering (mean fitted strictly on train split)
  3. E9 + ZCA Whitening (reg eps=0.05 & 0.5)
  4. E9 + PCA-K Whitening / Projection (K=32, K=64)
  5. E9 + Linear Probe Pseudoinverse Editing (Euclidean right pseudoinverse W_p^+)
  6. E9 + Covariance-Aware (Mahalanobis) Pseudoinverse Editing (Sigma W_p^T (W_p Sigma W_p^T + g I)^-1)
Compared directly against E12-C (and E12-B) across 3 seeds on:
  A. Cross-task Relational Alignment & Top-1 Retrieval
  B. Tier-1 & Tier-2 Parallel Transport Latent Arithmetic (DDIM-decoded, ZERO GT Δa at test time!)
  C. Unseen (dir x grip) Behavior Combination Generalization
  D. Action-Change Linear Transfer Sample-Efficiency Curve (N = 50..2000, multi-seed, disjoint tasks)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from e5.train import atomic_write_json
from e9.stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher
from e12.action_attributes import (
    compute_attribute_similarity_matrix,
    discretize_behavior_factors,
    extract_action_attribute_vectors,
    extract_action_delta_features,
    pairwise_cosine_sim,
    pairwise_l2_dist,
)
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _ridge_fit_predict,
    _spearman_pearson_from_vectors,
    ddim_decode_batched,
    encode_all_val_starts,
)

from . import CHECKPOINTS, OUTPUT_ROOT, SEEDS_DEFAULT


def _fit_whitener_gpu(
    Z_tr: np.ndarray,
    device: torch.device,
    *,
    mode: str = "zca",
    eps: float = 0.05,
    pca_k: int = 64,
) -> Dict[str, torch.Tensor]:
    """Fit centering + ZCA or PCA-K whitening strictly on Z_tr."""
    Z = torch.from_numpy(np.asarray(Z_tr, dtype=np.float64)).to(device)
    mu = Z.mean(dim=0, keepdim=True)  # (1, D)
    Zc = Z - mu
    n, d = Zc.shape
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # Sort descending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx].clamp_min(0.0)
    eigvecs = eigvecs[:, idx]

    if mode == "zca":
        inv_sqrt = torch.rsqrt(eigvals + eps)
        sqrt = torch.sqrt(eigvals + eps)
        W = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T  # (D, D)
        W_inv = (eigvecs * sqrt[None, :]) @ eigvecs.T
        P = torch.eye(d, device=device, dtype=torch.float64)
    elif mode == "pca_whiten":
        U = eigvecs[:, :pca_k]  # (D, K)
        lam = eigvals[:pca_k]
        inv_sqrt = torch.rsqrt(lam + eps)
        sqrt = torch.sqrt(lam + eps)
        W = U * inv_sqrt[None, :]  # (D, K): Zc @ W -> (N, K)
        W_inv = (U * sqrt[None, :]).T  # (K, D): Z_w @ W_inv -> (N, D)
        P = U @ U.T
    else:
        raise ValueError(mode)
    return {"mu": mu, "W": W, "W_inv": W_inv, "P": P, "cov": cov}


def _fit_linear_probe_and_pinv_gpu(
    Z_tr: np.ndarray,
    A_tr: np.ndarray,
    device: torch.device,
    *,
    probe_alpha: float = 100.0,
    pinv_gammas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
) -> Dict[str, Any]:
    """Fit linear probe A_hat = (Z - mu_z) W_p + mu_a strictly on (Z_tr, A_tr),
    plus both Euclidean pseudoinverse W_p^+ and Covariance-aware (Mahalanobis) pseudoinverse."""
    Z = torch.from_numpy(np.asarray(Z_tr, dtype=np.float64)).to(device)
    A = torch.from_numpy(np.asarray(A_tr, dtype=np.float64)).to(device)
    n, dz = Z.shape
    da = A.shape[1]

    mu_z = Z.mean(dim=0, keepdim=True)
    mu_a = A.mean(dim=0, keepdim=True)
    Zc = Z - mu_z
    Ac = A - mu_a

    # Internal 80/20 split to select probe_alpha & pinv_gamma cleanly
    n_val = max(10, int(0.2 * n)) if n >= 50 else 0
    if n_val > 0:
        Za, Aa = Zc[:-n_val], Ac[:-n_val]
        Zv, Av = Zc[-n_val:], Ac[-n_val:]
        # Fit probe with best alpha from grid
        best_pa = probe_alpha
        best_perr = float("inf")
        use_dual = Za.shape[0] < dz
        K_base = (Za @ Za.T) if use_dual else (Za.T @ Za)
        for al in (1.0, 10.0, 100.0, 500.0):
            if use_dual:
                K = K_base.clone()
                K.diagonal().add_(al)
                Wp = Za.T @ torch.linalg.solve(K, Aa)
            else:
                G = K_base.clone()
                G.diagonal().add_(al)
                Wp = torch.linalg.solve(G, Za.T @ Aa)
            err = float(((Zv @ Wp - Av) ** 2).mean().item())
            if err < best_perr:
                best_perr = err
                best_pa = al
        probe_alpha = best_pa

    # Fit final probe W_p: (dz, da) so Zc @ W_p -> Ac
    if n < dz:
        K = Zc @ Zc.T
        K.diagonal().add_(probe_alpha)
        W_p = Zc.T @ torch.linalg.solve(K, Ac)  # (dz, da)
    else:
        G = Zc.T @ Zc
        G.diagonal().add_(probe_alpha)
        W_p = torch.linalg.solve(G, Zc.T @ Ac)  # (dz, da)

    # Covariance Sigma_z = (Zc^T Zc) / (n - 1)
    Sigma = (Zc.T @ Zc) / max(n - 1, 1)  # (dz, dz)

    # Select gamma on internal val split for Euclidean & Mahalanobis pseudoinverses (minimizing latent MSE)
    WtW = W_p.T @ W_p  # (da, da)
    WtSW = W_p.T @ Sigma @ W_p  # (da, da)
    SW = Sigma @ W_p  # (dz, da)

    def _pick_pinv(M_right: torch.Tensor, gram: torch.Tensor) -> Tuple[torch.Tensor, float]:
        best_g = pinv_gammas[1]
        best_err = float("inf")
        if n_val > 0:
            for g in pinv_gammas:
                G_g = gram.clone()
                G_g.diagonal().add_(g)
                # Pinv: (da, dz) so Ac @ Pinv -> Zc_hat
                pinv_cand = torch.linalg.solve(G_g, M_right.T)
                z_hat = Av @ pinv_cand
                err = float(((z_hat - Zv) ** 2).mean().item())
                if err < best_err:
                    best_err = err
                    best_g = g
        G_f = gram.clone()
        G_f.diagonal().add_(best_g)
        pinv_final = torch.linalg.solve(G_f, M_right.T)  # (da, dz)
        return pinv_final, float(best_g)

    pinv_euc, g_euc = _pick_pinv(W_p, WtW)  # (da, dz)
    pinv_cov, g_cov = _pick_pinv(SW, WtSW)  # (da, dz)

    return {
        "mu_z": mu_z,
        "mu_a": mu_a,
        "W_p": W_p,  # (dz, da): Zc @ W_p -> Ac
        "pinv_euc": pinv_euc,  # (da, dz): dA @ pinv_euc -> dZ
        "pinv_cov": pinv_cov,  # (da, dz): dA @ pinv_cov -> dZ
        "probe_alpha": float(probe_alpha),
        "gamma_euc": g_euc,
        "gamma_cov": g_cov,
    }


def _build_fixed_analogy_quadruples(
    cache: ValClipCache,
    encoded_ref: Dict[str, Any],
    device: torch.device,
    *,
    num_transitions: int = 1500,
    num_per_tier: int = 256,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build exact deterministic Tier-1 and Tier-2 quadruples strictly from GT actions & tasks."""
    ep_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, ep in enumerate(cache.val_episodes):
        if not encoded_ref["is_tail"][idx]:
            ep_to_indices[int(ep)].append(idx)

    cand_pairs: List[Tuple[int, int]] = []
    for ep, idxs in sorted(ep_to_indices.items()):
        if len(idxs) < 2:
            continue
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                i, j = idxs[a_i], idxs[b_i]
                if abs(int(cache.val_frames[j]) - int(cache.val_frames[i])) >= 16:
                    cand_pairs.append((i, j))

    rng = np.random.default_rng(seed + 303)
    cand_arr = np.array(cand_pairs, dtype=np.int64)
    if len(cand_arr) > num_transitions:
        trans = cand_arr[np.sort(rng.choice(len(cand_arr), size=num_transitions, replace=False))]
    else:
        trans = cand_arr
    T_n = len(trans)
    ia, ib = trans[:, 0], trans[:, 1]
    t_tasks = torch.from_numpy(cache.val_task_ids[ia]).to(device)

    a7_a = torch.from_numpy(encoded_ref["actions_7d"][ia]).to(device)
    a7_b = torch.from_numpy(encoded_ref["actions_7d"][ib]).to(device)
    d_attrs = extract_action_delta_features(a7_a, a7_b)
    da_bal = d_attrs["delta_balanced7d"]
    da_raw7 = d_attrs["delta_raw7d"]
    S_da_bal = pairwise_cosine_sim(da_bal)
    da_norms_b = da_bal.norm(dim=-1)
    da_mse = (pairwise_l2_dist(da_raw7) ** 2) / float(da_raw7.shape[1])

    a10_a = torch.from_numpy(encoded_ref["actions_10d"][ia].reshape(T_n, -1)).to(device)
    base_mse10 = (pairwise_l2_dist(a10_a) ** 2) / float(a10_a.shape[1])

    triu = torch.triu(torch.ones((T_n, T_n), device=device, dtype=torch.bool), diagonal=1)
    cross_task_t = triu & (t_tasks[:, None] != t_tasks[None, :])
    trans_mag = (da_raw7 ** 2).mean(dim=-1)
    nontrivial = (trans_mag[:, None] > 0.05) & (trans_mag[None, :] > 0.05)
    norm_ratio = da_norms_b[:, None] / da_norms_b[None, :].clamp_min(1e-6)
    good_delta = (
        cross_task_t
        & nontrivial
        & (S_da_bal > 0.78)
        & (norm_ratio > 0.72)
        & (norm_ratio < 1.38)
        & (da_mse < 0.035)
    )

    out = {}
    for name, mask, s_off in [
        ("tier1", good_delta & (base_mse10 < 0.035), 701),
        ("tier2", good_delta & (base_mse10 > 0.08), 702),
    ]:
        pairs_ij = torch.nonzero(mask, as_tuple=False).cpu().numpy()
        scores = da_mse[mask].cpu().numpy()
        order = np.argsort(scores)
        pool = pairs_ij[order[: min(len(order), num_per_tier * 4)]]
        rng_t = np.random.default_rng(seed + s_off)
        chosen = (
            pool[np.sort(rng_t.choice(len(pool), size=num_per_tier, replace=False))]
            if len(pool) > num_per_tier
            else pool
        )
        pA, pB = chosen[:, 0], chosen[:, 1]
        out[name] = (ia[pA], ib[pA], ia[pB], ib[pB])
    return out


def run_exp1(out_dir: Path, seeds: Tuple[int, ...] = SEEDS_DEFAULT) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    pol_e9 = load_frozen_e9_stage1_teacher(CHECKPOINTS["e9"], device=device)
    pol_c = load_frozen_e9_stage1_teacher(CHECKPOINTS["e12_c"], device=device)

    enc_e9 = encode_all_val_starts(pol_e9, cache, device)
    enc_c = encode_all_val_starts(pol_c, cache, device)

    non_tail = np.flatnonzero(~enc_e9["is_tail"])
    tasks_nt = cache.val_task_ids[non_tail]
    unique_tasks = np.unique(tasks_nt)

    # ------------------------------------------------------------------
    # Part A: Cross-Task Alignment & Top-1 Retrieval (Post-processed E9 vs E12-C)
    # Fit transforms strictly on 70% tasks, evaluate on 30% held-out tasks!
    # ------------------------------------------------------------------
    part_a_seeds = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 11)
        perm_t = rng.permutation(unique_tasks)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        tr_idx = non_tail[np.array([int(t) in tr_set for t in tasks_nt])]
        te_idx = non_tail[np.array([int(t) not in tr_set for t in tasks_nt])]

        Z_tr = enc_e9["latents_flat"][tr_idx]
        Z_te = enc_e9["latents_flat"][te_idx]
        A7_tr = enc_e9["actions_7d"][tr_idx].reshape(len(tr_idx), -1)

        zca005 = _fit_whitener_gpu(Z_tr, device, mode="zca", eps=0.05)
        zca05 = _fit_whitener_gpu(Z_tr, device, mode="zca", eps=0.5)
        pca64 = _fit_whitener_gpu(Z_tr, device, mode="pca_whiten", eps=0.05, pca_k=64)
        probe = _fit_linear_probe_and_pinv_gpu(Z_tr, A7_tr, device)

        Z_te_t = torch.from_numpy(Z_te.astype(np.float64)).to(device)
        Zc_te = Z_te_t - zca005["mu"]

        reps = {
            "e9_raw": Z_te_t.float(),
            "e9_centered": Zc_te.float(),
            "e9_zca_0p05": (Zc_te @ zca005["W"]).float(),
            "e9_zca_0p5": (Zc_te @ zca05["W"]).float(),
            "e9_pca64_whiten": (Zc_te @ pca64["W"]).float(),
            "e9_probe_proj": (Zc_te @ probe["W_p"]).float(),
            "e12_c_raw": torch.from_numpy(enc_c["latents_flat"][te_idx]).to(device).float(),
            "e12_c_centered": (
                torch.from_numpy(enc_c["latents_flat"][te_idx]).to(device)
                - torch.from_numpy(enc_c["latents_flat"][tr_idx].mean(axis=0, keepdims=True)).to(device)
            ).float(),
        }

        a10_te = torch.from_numpy(enc_e9["actions_10d"][te_idx]).to(device)
        a7_te = torch.from_numpy(enc_e9["actions_7d"][te_idx]).to(device)
        attrs = extract_action_attribute_vectors(a10_te, a7_te)
        S_bal = compute_attribute_similarity_matrix(attrs, "global_balanced7d")
        S_mag = compute_attribute_similarity_matrix(attrs, "magnitude")
        S_temp = compute_attribute_similarity_matrix(attrs, "temporal")

        t_te = torch.from_numpy(cache.val_task_ids[te_idx]).to(device)
        M = len(te_idx)
        triu = torch.triu(torch.ones((M, M), device=device, dtype=torch.bool), diagonal=1)
        ct_triu = triu & (t_te[:, None] != t_te[None, :])
        ct_valid = (t_te[:, None] != t_te[None, :]) & (~torch.eye(M, device=device, dtype=torch.bool))
        a10_te_np = enc_e9["actions_10d"][te_idx]

        seed_res = {}
        for rname, R in reps.items():
            S_r = pairwise_cosine_sim(R)
            rho_bal, _ = _spearman_pearson_from_vectors(S_r[ct_triu], S_bal[ct_triu])
            rho_mag, _ = _spearman_pearson_from_vectors(S_r[ct_triu], S_mag[ct_triu])
            rho_tmp, _ = _spearman_pearson_from_vectors(S_r[ct_triu], S_temp[ct_triu])
            top1 = torch.topk(S_r.masked_fill(~ct_valid, -1e9), k=1, dim=1).indices.squeeze(-1).cpu().numpy()
            top1_mse10 = float(((a10_te_np - a10_te_np[top1]) ** 2).mean())
            seed_res[rname] = {
                "ct_rho_bal7d": rho_bal,
                "ct_rho_mag": rho_mag,
                "ct_rho_temp": rho_tmp,
                "ct_top1_mse10": top1_mse10,
            }
        part_a_seeds.append(seed_res)

    # ------------------------------------------------------------------
    # Part B: Tier-1 & Tier-2 Parallel Transport Arithmetic (NO GT Δa at test!)
    # Fit post-processing transforms strictly on train tasks NOT appearing in B1/B2!
    # ------------------------------------------------------------------
    quads = _build_fixed_analogy_quadruples(cache, enc_e9, device, seed=42)
    # Use train-task split (fit transforms on 70% tasks, evaluate ONLY quadruples whose B is in held-out 30% tasks!)
    # Actually to keep full quadruple statistics AND zero leakage, use 2-fold task cross-fit for post-processing!
    part_b_res: Dict[str, Any] = {}
    rng_f = np.random.default_rng(42 + 88)
    perm_t_all = rng_f.permutation(unique_tasks)
    half = len(perm_t_all) // 2
    fold_tasks = [set(perm_t_all[:half].tolist()), set(perm_t_all[half:].tolist())]
    # Fit transforms on the OPPOSITE task fold for every task!
    fold_models = []
    for f_i in range(2):
        fit_set = fold_tasks[1 - f_i]
        f_idx = non_tail[np.array([int(t) in fit_set for t in tasks_nt])]
        Z_f = enc_e9["latents_flat"][f_idx]
        A7_f = enc_e9["actions_7d"][f_idx].reshape(len(f_idx), -1)
        fold_models.append(
            {
                "pca64": _fit_whitener_gpu(Z_f, device, mode="pca_whiten", eps=0.05, pca_k=64),
                "pca32": _fit_whitener_gpu(Z_f, device, mode="pca_whiten", eps=0.05, pca_k=32),
                "probe": _fit_linear_probe_and_pinv_gpu(Z_f, A7_f, device),
            }
        )

    for tier_name, (A1, A2, B1, B2) in quads.items():
        N_q = len(A1)
        gt_B2 = enc_e9["actions_10d"][B2]
        zA1_9, zA2_9 = enc_e9["latents_flat"][A1], enc_e9["latents_flat"][A2]
        zB1_9, zB2_9 = enc_e9["latents_flat"][B1], enc_e9["latents_flat"][B2]
        zA1_c, zA2_c = enc_c["latents_flat"][A1], enc_c["latents_flat"][A2]
        zB1_c, zB2_c = enc_c["latents_flat"][B1], enc_c["latents_flat"][B2]

        dzA_9 = zA2_9 - zA1_9
        # Apply strictly out-of-fold post-processing operators to dzA_9
        dz_pca64 = np.zeros_like(dzA_9)
        dz_pca32 = np.zeros_like(dzA_9)
        dz_pinv_euc = np.zeros_like(dzA_9)
        dz_pinv_cov = np.zeros_like(dzA_9)
        dz_oracle_cov = np.zeros_like(dzA_9)

        daA_gt = (enc_e9["actions_7d"][A2] - enc_e9["actions_7d"][A1]).reshape(N_q, -1)
        for q_i in range(N_q):
            b_task = int(cache.val_task_ids[B1[q_i]])
            f_i = 0 if b_task in fold_tasks[0] else 1
            fm = fold_models[f_i]
            dz_t = torch.from_numpy(dzA_9[q_i : q_i + 1].astype(np.float64)).to(device)
            da_t = torch.from_numpy(daA_gt[q_i : q_i + 1].astype(np.float64)).to(device)

            dz_pca64[q_i] = (dz_t @ fm["pca64"]["P"]).float().cpu().numpy()[0]
            dz_pca32[q_i] = (dz_t @ fm["pca32"]["P"]).float().cpu().numpy()[0]
            # Latent-only probe-projected arithmetic (ZERO GT Δa!)
            da_hat = dz_t @ fm["probe"]["W_p"]
            dz_pinv_euc[q_i] = (da_hat @ fm["probe"]["pinv_euc"]).float().cpu().numpy()[0]
            dz_pinv_cov[q_i] = (da_hat @ fm["probe"]["pinv_cov"]).float().cpu().numpy()[0]
            # Reference: Oracle GT Δa cov-pseudoinverse edit
            dz_oracle_cov[q_i] = (da_t @ fm["probe"]["pinv_cov"]).float().cpu().numpy()[0]

        m_base_9 = _action_metrics_dict(ddim_decode_batched(pol_e9, zB1_9, device), gt_B2)
        m_ora_9 = _action_metrics_dict(ddim_decode_batched(pol_e9, zB2_9, device), gt_B2)
        m_base_c = _action_metrics_dict(ddim_decode_batched(pol_c, zB1_c, device), gt_B2)
        m_ora_c = _action_metrics_dict(ddim_decode_batched(pol_c, zB2_c, device), gt_B2)

        tier_methods = [
            ("e9_raw_and_centered", pol_e9, zB1_9 + dzA_9, m_base_9, m_ora_9, zB2_9, zB1_9),
            ("e9_pca64_proj", pol_e9, zB1_9 + dz_pca64, m_base_9, m_ora_9, zB2_9, zB1_9),
            ("e9_pca32_proj", pol_e9, zB1_9 + dz_pca32, m_base_9, m_ora_9, zB2_9, zB1_9),
            ("e9_probe_pinv_euc_latent_only", pol_e9, zB1_9 + dz_pinv_euc, m_base_9, m_ora_9, zB2_9, zB1_9),
            ("e9_probe_pinv_cov_latent_only", pol_e9, zB1_9 + dz_pinv_cov, m_base_9, m_ora_9, zB2_9, zB1_9),
            ("e9_probe_pinv_cov_oracle_da", pol_e9, zB1_9 + dz_oracle_cov, m_base_9, m_ora_9, zB2_9, zB1_9),
            ("e12_c_raw", pol_c, zB1_c + (zA2_c - zA1_c), m_base_c, m_ora_c, zB2_c, zB1_c),
        ]
        part_b_res[tier_name] = {"num_quadruples": N_q, "methods": {}}
        for mname, pol, z_ed, mb, mo, zB2_ref, zB1_ref in tier_methods:
            pred = ddim_decode_batched(pol, z_ed, device)
            me = _action_metrics_dict(pred, gt_B2)
            arr = (mb["action10_mse"] - me["action10_mse"]) / max(
                mb["action10_mse"] - mo["action10_mse"], 1e-9
            )
            dz_ed = z_ed - zB1_ref
            dz_true = zB2_ref - zB1_ref
            dcos = float(
                (
                    (dz_ed * dz_true).sum(axis=-1)
                    / np.clip(np.linalg.norm(dz_ed, axis=-1) * np.linalg.norm(dz_true, axis=-1), 1e-6, None)
                ).mean()
            )
            part_b_res[tier_name]["methods"][mname] = {
                "ddim_10d_mse": me["action10_mse"],
                "ddim_pos_mse": me["pos_mse"],
                "ddim_grip_mse": me["grip_mse"],
                "ddim_7d_mae": me["action7_mae"],
                "grip_acc": me["grip_sign_acc"],
                "arr_10d": float(arr),
                "disp_cos": dcos,
            }

    # ------------------------------------------------------------------
    # Part C: Unseen Behavior Combination Generalization (Multi-Seed)
    # ------------------------------------------------------------------
    held_out = ["+x|closed", "-x|open", "+y|open", "-y|closed", "+z|closing", "-z|opening"]
    a7_nt = enc_e9["actions_7d"][non_tail]
    a10_nt = enc_e9["actions_10d"][non_tail]
    fac = discretize_behavior_factors(a7_nt)
    is_u = np.isin(fac["combo_dir_grip"], held_out)
    seen_all = np.flatnonzero(~is_u)
    unseen_all = np.flatnonzero(is_u)
    X_a7 = a7_nt.reshape(len(a7_nt), -1)

    part_c_seeds = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 404)
        perm_s = rng.permutation(seen_all)
        seen_tr = perm_s[512:]
        unseen_te = np.sort(rng.choice(unseen_all, size=min(512, len(unseen_all)), replace=False))
        gt_u = a10_nt[unseen_te]

        Z9_tr = enc_e9["latents_flat"][non_tail[seen_tr]]
        Zc_tr = enc_c["latents_flat"][non_tail[seen_tr]]

        # 1. E9 Ridge (a -> z)
        z9_ridge, _ = _ridge_fit_predict(X_a7[seen_tr], Z9_tr, X_a7[unseen_te])
        # 2. E9 Whitened Ridge (ZCA 0.05 -> Ridge -> inverse ZCA)
        w_zca = _fit_whitener_gpu(Z9_tr, device, mode="zca", eps=0.05)
        Z9_tr_w = ((torch.from_numpy(Z9_tr.astype(np.float64)).to(device) - w_zca["mu"]) @ w_zca["W"]).float().cpu().numpy()
        z9_w_pred, _ = _ridge_fit_predict(X_a7[seen_tr], Z9_tr_w, X_a7[unseen_te])
        z9_zca_ridge = (torch.from_numpy(z9_w_pred.astype(np.float64)).to(device) @ w_zca["W_inv"] + w_zca["mu"]).float().cpu().numpy()
        # 3. E9 PCA-64 Whitened Ridge
        w_p64 = _fit_whitener_gpu(Z9_tr, device, mode="pca_whiten", eps=0.05, pca_k=64)
        Z9_tr_p = ((torch.from_numpy(Z9_tr.astype(np.float64)).to(device) - w_p64["mu"]) @ w_p64["W"]).float().cpu().numpy()
        z9_p_pred, _ = _ridge_fit_predict(X_a7[seen_tr], Z9_tr_p, X_a7[unseen_te])
        z9_p64_ridge = (torch.from_numpy(z9_p_pred.astype(np.float64)).to(device) @ w_p64["W_inv"] + w_p64["mu"]).float().cpu().numpy()
        # 4. E9 Linear Probe Pseudoinverse (Euc & Cov)
        pr_s = _fit_linear_probe_and_pinv_gpu(Z9_tr, X_a7[seen_tr], device)
        dA_u = torch.from_numpy(X_a7[unseen_te].astype(np.float64)).to(device) - pr_s["mu_a"]
        z9_pinv_euc = (dA_u @ pr_s["pinv_euc"] + pr_s["mu_z"]).float().cpu().numpy()
        z9_pinv_cov = (dA_u @ pr_s["pinv_cov"] + pr_s["mu_z"]).float().cpu().numpy()
        # 5. E12-C Ridge
        zc_ridge, _ = _ridge_fit_predict(X_a7[seen_tr], Zc_tr, X_a7[unseen_te])

        c_res = {}
        for mname, pol, z_pred in [
            ("e9_ridge", pol_e9, z9_ridge),
            ("e9_zca_whiten_ridge", pol_e9, z9_zca_ridge),
            ("e9_pca64_whiten_ridge", pol_e9, z9_p64_ridge),
            ("e9_probe_pinv_euc", pol_e9, z9_pinv_euc),
            ("e9_probe_pinv_cov", pol_e9, z9_pinv_cov),
            ("e12_c_ridge", pol_c, zc_ridge),
        ]:
            m = _action_metrics_dict(ddim_decode_batched(pol, z_pred, device), gt_u)
            c_res[mname] = {
                "ddim_10d_mse": m["action10_mse"],
                "ddim_7d_mae": m["action7_mae"],
                "grip_acc": m["grip_sign_acc"],
            }
        part_c_seeds.append(c_res)

    # ------------------------------------------------------------------
    # Part D: Action-Change Transfer Sample-Efficiency Curve (Multi-Seed, N=50..2000)
    # Compares E9 Ridge, E9 PCA64-Whitened Ridge, E9 Probe Pinv (Euc & Cov, fit at budget N
    # AND fit on full train-task pool), vs E12-C Ridge!
    # ------------------------------------------------------------------
    sample_sizes = (50, 100, 250, 500, 1000, 2000)
    part_d_seeds = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 505)
        perm_t = rng.permutation(unique_tasks)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        tr_idx = non_tail[np.array([int(t) in tr_set for t in tasks_nt])]
        te_idx = non_tail[np.array([int(t) not in tr_set for t in tasks_nt])]

        def _ep_pairs(pool: np.ndarray, max_n: int, rng_l: np.random.Generator):
            by_ep: Dict[int, List[int]] = defaultdict(list)
            for idx in pool:
                by_ep[int(cache.val_episodes[idx])].append(int(idx))
            ps = []
            for ep, lst in sorted(by_ep.items()):
                for i in range(len(lst)):
                    for j in range(len(lst)):
                        if i != j and abs(int(cache.val_frames[lst[i]]) - int(cache.val_frames[lst[j]])) >= 16:
                            ps.append((lst[i], lst[j]))
            arr = np.array(ps, dtype=np.int64)
            arr = arr[rng_l.choice(len(arr), size=min(max_n, len(arr)), replace=False)]
            return arr[:, 0], arr[:, 1]

        tr_s, tr_t = _ep_pairs(tr_idx, max(sample_sizes), np.random.default_rng(seed + 601))
        te_s, te_t = _ep_pairs(te_idx, 256, np.random.default_rng(seed + 602))
        gt_te = enc_e9["actions_10d"][te_t]

        X_tr = (enc_e9["actions_7d"][tr_t] - enc_e9["actions_7d"][tr_s]).reshape(len(tr_s), -1)
        X_te = (enc_e9["actions_7d"][te_t] - enc_e9["actions_7d"][te_s]).reshape(len(te_s), -1)
        Y9_tr = enc_e9["latents_flat"][tr_t] - enc_e9["latents_flat"][tr_s]
        Yc_tr = enc_c["latents_flat"][tr_t] - enc_c["latents_flat"][tr_s]

        # Full-train-task E9 probe pseudoinverse upper bound (uses ALL train-task clips!)
        pr_full = _fit_linear_probe_and_pinv_gpu(
            enc_e9["latents_flat"][tr_idx],
            enc_e9["actions_7d"][tr_idx].reshape(len(tr_idx), -1),
            device,
        )
        X_te_t = torch.from_numpy(X_te.astype(np.float64)).to(device)
        z9_full_euc = enc_e9["latents_flat"][te_s] + (X_te_t @ pr_full["pinv_euc"]).float().cpu().numpy()
        z9_full_cov = enc_e9["latents_flat"][te_s] + (X_te_t @ pr_full["pinv_cov"]).float().cpu().numpy()

        seed_d: Dict[str, Any] = {
            "e9_full_train_pinv_euc": _action_metrics_dict(ddim_decode_batched(pol_e9, z9_full_euc, device), gt_te)["action10_mse"],
            "e9_full_train_pinv_cov": _action_metrics_dict(ddim_decode_batched(pol_e9, z9_full_cov, device), gt_te)["action10_mse"],
            "by_n": {},
        }
        for n_s in sample_sizes:
            n_eff = min(int(n_s), len(tr_s))
            # 1. E9 Ridge
            dz9_r, _ = _ridge_fit_predict(X_tr[:n_eff], Y9_tr[:n_eff], X_te)
            m_e9_r = _action_metrics_dict(ddim_decode_batched(pol_e9, enc_e9["latents_flat"][te_s] + dz9_r, device), gt_te)
            # 2. E9 PCA-64 Whitened Ridge (fit PCA on the same N pairs' endpoints!)
            ep_z = np.concatenate([enc_e9["latents_flat"][tr_s[:n_eff]], enc_e9["latents_flat"][tr_t[:n_eff]]], axis=0)
            w64 = _fit_whitener_gpu(ep_z, device, mode="pca_whiten", eps=0.05, pca_k=min(64, n_eff))
            Y9_w = (torch.from_numpy(Y9_tr[:n_eff].astype(np.float64)).to(device) @ w64["W"]).float().cpu().numpy()
            dz9_w, _ = _ridge_fit_predict(X_tr[:n_eff], Y9_w, X_te)
            dz9_p64 = (torch.from_numpy(dz9_w.astype(np.float64)).to(device) @ w64["W_inv"]).float().cpu().numpy()
            m_e9_p64 = _action_metrics_dict(ddim_decode_batched(pol_e9, enc_e9["latents_flat"][te_s] + dz9_p64, device), gt_te)
            # 3. E9 Probe Pinv (Cov) fit on N samples' endpoints
            ep_a = np.concatenate([enc_e9["actions_7d"][tr_s[:n_eff]], enc_e9["actions_7d"][tr_t[:n_eff]]], axis=0).reshape(2 * n_eff, -1)
            pr_n = _fit_linear_probe_and_pinv_gpu(ep_z, ep_a, device)
            dz9_pcov = (X_te_t @ pr_n["pinv_cov"]).float().cpu().numpy()
            m_e9_pcov = _action_metrics_dict(ddim_decode_batched(pol_e9, enc_e9["latents_flat"][te_s] + dz9_pcov, device), gt_te)
            # 4. E12-C Ridge
            dzc_r, _ = _ridge_fit_predict(X_tr[:n_eff], Yc_tr[:n_eff], X_te)
            m_c_r = _action_metrics_dict(ddim_decode_batched(pol_c, enc_c["latents_flat"][te_s] + dzc_r, device), gt_te)

            seed_d["by_n"][str(n_eff)] = {
                "e9_ridge": m_e9_r["action10_mse"],
                "e9_pca64_whiten_ridge": m_e9_p64["action10_mse"],
                "e9_probe_pinv_cov": m_e9_pcov["action10_mse"],
                "e12_c_ridge": m_c_r["action10_mse"],
            }
        part_d_seeds.append(seed_d)

    def _agg_dict_list(lst: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        out_agg: Dict[str, Dict[str, Dict[str, float]]] = {}
        for k1 in lst[0]:
            out_agg[k1] = {}
            for k2 in lst[0][k1]:
                vals = [d[k1][k2] for d in lst]
                out_agg[k1][k2] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out_agg

    part_d_agg = {
        "e9_full_train_pinv_euc": {
            "mean": float(np.mean([d["e9_full_train_pinv_euc"] for d in part_d_seeds])),
            "std": float(np.std([d["e9_full_train_pinv_euc"] for d in part_d_seeds])),
        },
        "e9_full_train_pinv_cov": {
            "mean": float(np.mean([d["e9_full_train_pinv_cov"] for d in part_d_seeds])),
            "std": float(np.std([d["e9_full_train_pinv_cov"] for d in part_d_seeds])),
        },
        "by_n": {},
    }
    for n_str in part_d_seeds[0]["by_n"]:
        part_d_agg["by_n"][n_str] = {}
        for m in part_d_seeds[0]["by_n"][n_str]:
            vals = [d["by_n"][n_str][m] for d in part_d_seeds]
            part_d_agg["by_n"][n_str][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "seeds": list(seeds),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "part_a_alignment_retrieval": _agg_dict_list(part_a_seeds),
        "part_b_latent_arithmetic": part_b_res,
        "part_c_unseen_combinations": _agg_dict_list(part_c_seeds),
        "part_d_transfer_efficiency": part_d_agg,
    }
    atomic_write_json(summary, out_dir / "exp1_summary.json")

    print("\n=== [E13 Exp 1] Training-Free Post-Processing on E9 vs E12-C ===", flush=True)
    print("Part A (Cross-Task Retrieval Top-1 MSE10 & Bal7D rho):")
    for k, v in summary["part_a_alignment_retrieval"].items():
        print(f"  {k:20s}: rho_bal={v['ct_rho_bal7d']['mean']:.3f} rho_mag={v['ct_rho_mag']['mean']:.3f} top1_mse10={v['ct_top1_mse10']['mean']:.4f}±{v['ct_top1_mse10']['std']:.4f}")
    print("Part B (Tier-2 Parallel Transport Arithmetic MSE10 & ARR):")
    for k, v in summary["part_b_latent_arithmetic"]["tier2"]["methods"].items():
        print(f"  {k:32s}: mse10={v['ddim_10d_mse']:.5f} ARR={v['arr_10d']*100:.1f}% grip={v['grip_acc']*100:.1f}% cos={v['disp_cos']:.3f}")
    print("Part C (Unseen Combo MSE10, 3 seeds):")
    for k, v in summary["part_c_unseen_combinations"].items():
        print(f"  {k:24s}: mse10={v['ddim_10d_mse']['mean']:.5f}±{v['ddim_10d_mse']['std']:.5f} grip={v['grip_acc']['mean']*100:.1f}%")
    print("Part D (Action-Change Transfer Efficiency MSE10, 3 seeds):")
    print(f"  e9_full_train_pinv_euc={part_d_agg['e9_full_train_pinv_euc']['mean']:.5f} | e9_full_train_pinv_cov={part_d_agg['e9_full_train_pinv_cov']['mean']:.5f}")
    for n_str, row in part_d_agg["by_n"].items():
        print(
            f"  N={n_str:4s}: e9_ridge={row['e9_ridge']['mean']:.5f} | "
            f"e9_pca64={row['e9_pca64_whiten_ridge']['mean']:.5f} | "
            f"e9_pinv_cov={row['e9_probe_pinv_cov']['mean']:.5f} | "
            f"e12_c={row['e12_c_ridge']['mean']:.5f}"
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp1_training_free")
    args = p.parse_args()
    run_exp1(args.out_dir)


if __name__ == "__main__":
    main()
