"""E17 Exp 2: Emergent Behavior Factors — Unsupervised Discovery, Optimal Matching, Intervention & Temporal Analysis.

Without presupposing that Token k corresponds to any human-defined E12 action attribute:
  1. Full 4-Token x 11-Property Probe Matrix (on z_full, z_op, z_res) + Inter-Token CKA.
  2. All-Pairs 4-Token x 5-Attribute Causal Token Swap (full token & z_op-only slice)
     + Optimal Hungarian Assignment (maximum alignment without fixed token labels!)
     + Selectivity Ratio (target adoption / mean off-target drift).
  3. Unsupervised Factor Discovery (train-only PCA & FastICA on z_op, ZERO GT actions):
     - Property probing per unsupervised axis -> DCI Disentanglement & Completeness!
     - Causal Subspace Intervention: swap top unsupervised axes aligned with each discovered
       factor and decode with frozen Action Decoder -> measure adoption vs interference!
  4. Multi-Factor Compositional Generalization: unseen (dir x grip) & (dir x speed) via
     optimal-matched token swap, unsupervised subspace swap, and z_op arithmetic.
  5. Temporal Dynamics Along Episodes: lag autocorrelation half-life (fast kinematic vs slow
     phase/intent factors) and latent-velocity changepoint alignment with physical grasp/release
     and direction-reversal subgoal boundaries!
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.action_attributes import discretize_behavior_factors, extract_action_attribute_vectors
from e12.eval_suite import ValClipCache, _action_metrics_dict, _ridge_fit_predict, encode_all_val_starts
from e13.exp3_factor_composition import ATTR_ORDER, _attr_pairwise_dist, _select_swap_pairs
from e14.common import ddim_decode_fixed_noise
from e15.train_e15 import compute_linear_cka_penalty
from e16 import REFERENCE_CHECKPOINTS
from e16.analyze_discovered_relations import _gather_beyond_e12_targets

from . import E16_DECODER_CKPTS, E16_REGIME1_LEARN_CKPTS, OUTPUT_ROOT


def _dci_disentanglement_completeness(R_mat: np.ndarray) -> Tuple[float, float]:
    """DCI-style Disentanglement & Completeness from nonnegative (K_factors x M_attrs) importance matrix."""
    R = np.maximum(np.asarray(R_mat, dtype=np.float64), 0.0)
    K, M = R.shape
    if R.sum() <= 1e-12 or K <= 1 or M <= 1:
        return 0.0, 0.0
    # Disentanglement: each factor concentrates on 1 attribute
    row_sums = R.sum(axis=1)
    d_scores = []
    for i in range(K):
        if row_sums[i] <= 1e-12:
            d_scores.append(0.0)
            continue
        p = R[i] / row_sums[i]
        p_pos = p[p > 1e-12]
        ent = -np.sum(p_pos * np.log(p_pos)) / math.log(M)
        d_scores.append(1.0 - ent)
    disent = float(np.sum(row_sums * np.array(d_scores)) / max(row_sums.sum(), 1e-12))

    # Completeness: each attribute concentrates on 1 factor
    col_sums = R.sum(axis=0)
    c_scores = []
    for j in range(M):
        if col_sums[j] <= 1e-12:
            c_scores.append(0.0)
            continue
        p = R[:, j] / col_sums[j]
        p_pos = p[p > 1e-12]
        ent = -np.sum(p_pos * np.log(p_pos)) / math.log(K)
        c_scores.append(1.0 - ent)
    comp = float(np.sum(col_sums * np.array(c_scores)) / max(col_sums.sum(), 1e-12))
    return disent, comp


def _roc_auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def analyze_emergent_factors(
    name: str,
    ckpt_path: Path,
    cache: ValClipCache,
    beyond: Dict[str, Any],
    swap_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    op_dim: int = 384,
    n_ica: int = 12,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=True)

    N_all = len(cache.val_episodes)
    Z4 = encoded["latents"]  # (N, 4, 768)
    Z_op4 = Z4[:, :, :op_dim]
    Z_op = Z_op4.reshape(N_all, -1)
    Z_full = Z4.reshape(N_all, -1)

    non_tail = np.flatnonzero(~encoded["is_tail"])
    valid_nt = non_tail[beyond["fut16_ok"][non_tail] & beyond["fut32_ok"][non_tail]]

    # Disjoint task split (70/30)
    v_tasks = cache.val_task_ids[valid_nt]
    unique_t = np.unique(v_tasks)
    rng = np.random.default_rng(1722)
    perm_t = rng.permutation(unique_t)
    tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
    tr_i = valid_nt[np.array([int(t) in tr_set for t in v_tasks])]
    te_i = valid_nt[np.array([int(t) not in tr_set for t in v_tasks])]

    # Build 11 behavioral property targets
    a10_all = encoded["actions_10d"]
    a7_all = encoded["actions_7d"]
    attrs_all = extract_action_attribute_vectors(torch.from_numpy(a10_all), torch.from_numpy(a7_all))
    trans_v = attrs_all["translation"].numpy()
    trans_dir = trans_v / np.clip(np.linalg.norm(trans_v, axis=-1, keepdims=True), 1e-6, None)
    props: Dict[str, np.ndarray] = {
        "trans_xyz": trans_v,
        "trans_dir": trans_dir,
        "rot": attrs_all["rotation"].numpy(),
        "mag": attrs_all["magnitude"].numpy(),
        "temporal": attrs_all["temporal"].numpy(),
        "grip_state": attrs_all["gripper"].numpy(),
        "grip_switch": np.eye(2, dtype=np.float32)[beyond["upcoming_grip_switch"]],
        "fut16": beyond["fut16_a7"],
        "fut32": beyond["fut32_a7"],
        "progress": beyond["progress"],
        "scene_v0": encoded["h_queries"][:, 2:4, :].reshape(N_all, -1),
    }
    prop_names = list(props.keys())

    # 1. Full 4-Token x 11-Property Probe Matrix (both full token 768D & op token 384D) + Inter-Token CKA
    tok_probe_full: Dict[str, Dict[str, float]] = {}
    tok_probe_op: Dict[str, Dict[str, float]] = {}
    tok_energy_full = []
    tok_energy_op = []
    for k in range(4):
        tk_f = Z4[:, k, :]
        tk_o = Z_op4[:, k, :]
        tok_energy_full.append(float(np.mean(np.sum((tk_f[te_i] - tk_f[te_i].mean(0, keepdims=True)) ** 2, axis=-1))))
        tok_energy_op.append(float(np.mean(np.sum((tk_o[te_i] - tk_o[te_i].mean(0, keepdims=True)) ** 2, axis=-1))))
        tok_probe_full[f"tok{k}"] = {}
        tok_probe_op[f"tok{k}"] = {}
        for pname in prop_names:
            Y = props[pname]
            v_y = float(np.mean((Y[te_i] - Y[te_i].mean(0, keepdims=True)) ** 2))
            pf, _ = _ridge_fit_predict(tk_f[tr_i], Y[tr_i], tk_f[te_i], device=device)
            po, _ = _ridge_fit_predict(tk_o[tr_i], Y[tr_i], tk_o[te_i], device=device)
            tok_probe_full[f"tok{k}"][pname] = float(1.0 - np.mean((pf - Y[te_i]) ** 2) / max(v_y, 1e-12))
            tok_probe_op[f"tok{k}"][pname] = float(1.0 - np.mean((po - Y[te_i]) ** 2) / max(v_y, 1e-12))

    # Inter-token CKA (4x4) on full & op
    cka_tok_full = np.ones((4, 4), dtype=np.float64)
    cka_tok_op = np.ones((4, 4), dtype=np.float64)
    for a in range(4):
        for b in range(a + 1, 4):
            cf = float(compute_linear_cka_penalty(torch.from_numpy(Z4[te_i, a]).to(device), torch.from_numpy(Z4[te_i, b]).to(device)).item())
            co = float(compute_linear_cka_penalty(torch.from_numpy(Z_op4[te_i, a]).to(device), torch.from_numpy(Z_op4[te_i, b]).to(device)).item())
            cka_tok_full[a, b] = cka_tok_full[b, a] = cf
            cka_tok_op[a, b] = cka_tok_op[b, a] = co
    off4 = ~np.eye(4, dtype=bool)

    # 2. All-Pairs 4-Token x 5-Attribute Causal Token Swap (Full token & z_op-only!) + Hungarian Matching!
    # For each attribute's swap pairs (base, donor in non_tail coords), try swapping EACH token k in {0,1,2,3}
    z4_nt = Z4[non_tail]
    a10_nt = a10_all[non_tail]
    all_pairs_swap: Dict[str, Any] = {"full_token": {}, "op_slice_token": {}}
    adopt_mat_full = np.zeros((4, len(ATTR_ORDER)), dtype=np.float64)
    drift_mat_full = np.zeros((4, len(ATTR_ORDER)), dtype=np.float64)
    adopt_mat_op = np.zeros((4, len(ATTR_ORDER)), dtype=np.float64)
    drift_mat_op = np.zeros((4, len(ATTR_ORDER)), dtype=np.float64)

    for a_idx, attr in enumerate(ATTR_ORDER):
        b_idx, d_idx = swap_pairs[attr]
        n_p = min(256, len(b_idx))
        b_idx, d_idx = b_idx[:n_p], d_idx[:n_p]
        zb = z4_nt[b_idx]
        zd = z4_nt[d_idx]
        lat_list = [zb.reshape(n_p, -1), zd.reshape(n_p, -1)]
        for k in range(4):
            zs_f = zb.copy()
            zs_f[:, k, :] = zd[:, k, :]
            lat_list.append(zs_f.reshape(n_p, -1))
        for k in range(4):
            zs_o = zb.copy()
            zs_o[:, k, :op_dim] = zd[:, k, :op_dim]
            lat_list.append(zs_o.reshape(n_p, -1))

        decs = ddim_decode_fixed_noise(policy, lat_list, device, seed=12345, batch_size=512)
        a_base_rec = decs[0]
        a_don_rec = decs[1]
        d_bd = _attr_pairwise_dist(a_base_rec, a_don_rec)

        for mode_i, mode_name in enumerate(["full_token", "op_slice_token"]):
            for k in range(4):
                a_sw = decs[2 + mode_i * 4 + k]
                d_sb = _attr_pairwise_dist(a_sw, a_base_rec)
                d_sd = _attr_pairwise_dist(a_sw, a_don_rec)
                # Target adoption: how much closer to donor than base on `attr`
                denom_t = np.maximum(d_bd[attr], 1e-5)
                adopt_t = float(np.mean(1.0 - d_sd[attr] / denom_t))
                # Off-target drift: normalized drift from base across OTHER 4 attributes
                other_drifts = []
                for o_attr in ATTR_ORDER:
                    if o_attr == attr:
                        continue
                    denom_o = float(np.mean(d_bd[o_attr])) + 1e-5
                    other_drifts.append(float(np.mean(d_sb[o_attr])) / denom_o)
                mean_drift = float(np.mean(other_drifts))
                all_pairs_swap[mode_name].setdefault(f"tok{k}", {})[attr] = {
                    "target_adoption": adopt_t,
                    "mean_other_drift": mean_drift,
                    "selectivity": float(adopt_t / max(mean_drift, 1e-4)),
                }
                if mode_name == "full_token":
                    adopt_mat_full[k, a_idx] = adopt_t
                    drift_mat_full[k, a_idx] = mean_drift
                else:
                    adopt_mat_op[k, a_idx] = adopt_t
                    drift_mat_op[k, a_idx] = mean_drift

    def _hungarian_summary(adopt_m: np.ndarray, drift_m: np.ndarray) -> Dict[str, Any]:
        # Optimal 1-to-1 assignment of 4 tokens to 4 best attributes (out of 5)
        r_ind, c_ind = linear_sum_assignment(-adopt_m)
        mapping = {f"tok{int(r)}": ATTR_ORDER[int(c)] for r, c in zip(r_ind, c_ind)}
        matched_adopts = [float(adopt_m[r, c]) for r, c in zip(r_ind, c_ind)]
        matched_drifts = [float(drift_m[r, c]) for r, c in zip(r_ind, c_ind)]
        # Also fixed E12 diagonal (tok0=trans, tok1=rot, tok2=mag, tok3=grip -> cols 0,1,2,4)
        fixed_cols = [0, 1, 2, 4]
        fixed_adopts = [float(adopt_m[k, fixed_cols[k]]) for k in range(4)]
        dci_d, dci_c = _dci_disentanglement_completeness(np.maximum(adopt_m, 0.0))
        return {
            "optimal_mapping": mapping,
            "hungarian_mean_adoption": float(np.mean(matched_adopts)),
            "hungarian_mean_drift": float(np.mean(matched_drifts)),
            "hungarian_selectivity": float(np.mean(matched_adopts) / max(np.mean(matched_drifts), 1e-4)),
            "hungarian_per_token": {f"tok{int(r)}": {"attr": ATTR_ORDER[int(c)], "adoption": float(adopt_m[r, c]), "drift": float(drift_m[r, c])} for r, c in zip(r_ind, c_ind)},
            "fixed_e12_mean_adoption": float(np.mean(fixed_adopts)),
            "causal_dci_disentanglement": dci_d,
            "causal_dci_completeness": dci_c,
            "adoption_matrix_4x5": adopt_m.tolist(),
            "drift_matrix_4x5": drift_m.tolist(),
        }

    hungarian_full = _hungarian_summary(adopt_mat_full, drift_mat_full)
    hungarian_op = _hungarian_summary(adopt_mat_op, drift_mat_op)

    # 3. Unsupervised Factor Discovery on z_op (Train-only PCA + FastICA, ZERO GT Actions!)
    mu_op = Z_op[tr_i].mean(axis=0, keepdims=True)
    X_tr_c = Z_op[tr_i] - mu_op
    X_te_c = Z_op[te_i] - mu_op
    # Top-12 PCA basis on GPU
    _, S_p, Vh_p = torch.linalg.svd(torch.from_numpy(X_tr_c.astype(np.float32)).to(device), full_matrices=False)
    V_pca = Vh_p[:n_ica].cpu().numpy()  # (n_ica, 1536) orthonormal rows
    pca_tr = X_tr_c @ V_pca.T
    pca_te = X_te_c @ V_pca.T

    # FastICA on top-n_ica whitened PCA subspace (strictly unsupervised, train-only!)
    ica = FastICA(n_components=n_ica, random_state=42, max_iter=500, tol=1e-3)
    ica_tr = ica.fit_transform(pca_tr)
    ica_te = ica.transform(pca_te)
    # Linear map from centered z_op -> ICA components & inverse reconstruction into z_op!
    # ica_x = (z_op - mu_op) @ V_pca.T @ W_ica^T  (using sklearn's components_ and mixing_)
    W_ica = ica.components_  # (n_ica, n_ica): ica_x = (pca - pca_mean) @ W_ica.T
    M_ica = ica.mixing_      # (n_ica, n_ica): pca - pca_mean = ica_x @ M_ica.T
    pca_mean = ica.mean_     # (n_ica,)

    # Probe each individual PCA & ICA component against core properties (unsupervised factor interpretation!)
    core_props = ["trans_xyz", "rot", "mag", "temporal", "grip_state", "fut16", "progress", "scene_v0"]

    def _probe_1d_factors(F_tr: np.ndarray, F_te: np.ndarray) -> Dict[str, Any]:
        # Matrix (n_ica x len(core_props)): R^2 of predicting 1D factor from each property OR predicting property from 1D factor!
        # Predicting 1D factor from property measures "what determines this emergent factor" cleanly across multi-dim props!
        R_prop_to_fac = np.zeros((F_tr.shape[1], len(core_props)), dtype=np.float64)
        for j, pname in enumerate(core_props):
            Y_tr, Y_te = props[pname][tr_i], props[pname][te_i]
            pred_f, _ = _ridge_fit_predict(Y_tr, F_tr, Y_te, device=device)
            for k in range(F_tr.shape[1]):
                v = float(np.var(F_te[:, k]))
                R_prop_to_fac[k, j] = max(0.0, float(1.0 - np.mean((pred_f[:, k] - F_te[:, k]) ** 2) / max(v, 1e-12)))
        # Disentanglement across behavioral props (excluding scene_v0)
        beh_mat = R_prop_to_fac[:, :7]
        d_beh, c_beh = _dci_disentanglement_completeness(beh_mat)
        dom_props = [core_props[int(np.argmax(R_prop_to_fac[k]))] for k in range(F_tr.shape[1])]
        dom_beh_props = [core_props[int(np.argmax(beh_mat[k]))] for k in range(F_tr.shape[1])]
        return {
            "props": core_props,
            "r2_prop_to_factor": R_prop_to_fac.tolist(),
            "dominant_prop_per_factor": dom_props,
            "dominant_beh_prop_per_factor": dom_beh_props,
            "dci_disentanglement_beh": d_beh,
            "dci_completeness_beh": c_beh,
            "mean_top_prop_r2": float(np.mean(np.max(R_prop_to_fac, axis=1))),
            "mean_top_beh_prop_r2": float(np.mean(np.max(beh_mat, axis=1))),
            "mean_scene_r2": float(np.mean(R_prop_to_fac[:, -1])),
        }

    pca_probe = _probe_1d_factors(pca_tr, pca_te)
    ica_probe = _probe_1d_factors(ica_tr, ica_te)

    # 3b. Causal Subspace Intervention using Unsupervised PCA & ICA Axes (swap top-3 axes aligned to each attr!)
    # Notice: axes themselves are 100% UNSUPERVISED; we only select which top-3 axes to swap per attr!
    subspace_interventions: Dict[str, Any] = {}
    attr_to_prop_col = {"translation": 0, "rotation": 1, "magnitude": 2, "temporal": 3, "gripper": 4}
    for basis_name in ("pca_top3", "ica_top3"):
        R_mat = np.array(pca_probe["r2_prop_to_factor"] if basis_name == "pca_top3" else ica_probe["r2_prop_to_factor"])
        by_attr: Dict[str, Any] = {}
        for attr in ATTR_ORDER:
            col = attr_to_prop_col[attr]
            top_axes = np.argsort(-R_mat[:, col])[:3]
            b_idx, d_idx = swap_pairs[attr]
            n_p = min(256, len(b_idx))
            b_idx, d_idx = b_idx[:n_p], d_idx[:n_p]
            zb = z4_nt[b_idx].copy()
            zd = z4_nt[d_idx].copy()
            zb_op = zb[:, :, :op_dim].reshape(n_p, -1) - mu_op
            zd_op = zd[:, :, :op_dim].reshape(n_p, -1) - mu_op

            if basis_name == "pca_top3":
                V_sub = V_pca[top_axes]  # (3, 1536)
                delta_op = ((zd_op - zb_op) @ V_sub.T) @ V_sub
            else:
                cb = (zb_op @ V_pca.T - pca_mean) @ W_ica.T
                cd = (zd_op @ V_pca.T - pca_mean) @ W_ica.T
                diff_c = np.zeros_like(cb)
                diff_c[:, top_axes] = cd[:, top_axes] - cb[:, top_axes]
                delta_op = (diff_c @ M_ica.T) @ V_pca

            z_sw = zb.copy()
            z_sw[:, :, :op_dim] = (zb_op + delta_op + mu_op).reshape(n_p, 4, op_dim)
            a_b, a_d, a_s = ddim_decode_fixed_noise(
                policy, [zb.reshape(n_p, -1), zd.reshape(n_p, -1), z_sw.reshape(n_p, -1)],
                device, seed=12345, batch_size=512
            )
            d_bd = _attr_pairwise_dist(a_b, a_d)
            d_sb = _attr_pairwise_dist(a_s, a_b)
            d_sd = _attr_pairwise_dist(a_s, a_d)
            adopt = float(np.mean(1.0 - d_sd[attr] / np.maximum(d_bd[attr], 1e-5)))
            drift = float(np.mean([float(np.mean(d_sb[o])) / (float(np.mean(d_bd[o])) + 1e-5) for o in ATTR_ORDER if o != attr]))
            by_attr[attr] = {
                "axes": [int(x) for x in top_axes],
                "target_adoption": adopt,
                "mean_other_drift": drift,
                "selectivity": float(adopt / max(drift, 1e-4)),
            }
        subspace_interventions[basis_name] = {
            "by_attr": by_attr,
            "mean_adoption": float(np.mean([v["target_adoption"] for v in by_attr.values()])),
            "mean_drift": float(np.mean([v["mean_other_drift"] for v in by_attr.values()])),
            "mean_selectivity": float(np.mean([v["selectivity"] for v in by_attr.values()])),
        }

    # 4. Unseen (Dir x Grip) Compositional Generalization via Optimal Token Swap vs ICA-Subspace Swap vs z_op Arith
    # Find best gripper token (from full_token adopt_mat) & best gripper ICA axes
    best_grip_tok = int(np.argmax(adopt_mat_full[:, 4]))
    best_grip_ica = np.argsort(-np.array(ica_probe["r2_prop_to_factor"])[:, 4])[:3]
    disc = discretize_behavior_factors(a7_all[non_tail])
    dir_lbl = disc["dir_label"]
    grip_lbl = disc["grip_label"]
    # Held-out combinations: (dir in {+x, -x}, closed) -> synthesize from ({+x,-x}, open) + ({+y,-y,+z,-z}, closed)!
    in_tgt_dir = np.isin(dir_lbl, ["+x", "-x"])
    in_oth_dir = np.isin(dir_lbl, ["+y", "-y", "+z", "-z"])
    tgt_mask = in_tgt_dir & (grip_lbl == "closed")
    src_base_pool = np.flatnonzero(in_tgt_dir & (grip_lbl == "open"))
    src_donor_pool = np.flatnonzero(in_oth_dir & (grip_lbl == "closed"))
    comp_gen: Dict[str, Any] = {}
    if tgt_mask.sum() >= 32 and len(src_base_pool) >= 32 and len(src_donor_pool) >= 32:
        rng_c = np.random.default_rng(1725)
        n_c = min(192, len(src_base_pool))
        bi = rng_c.choice(src_base_pool, size=n_c, replace=False)
        di = rng_c.choice(src_donor_pool, size=n_c, replace=True)
        # Also pick reference open clip with same dir group as donor for z_op arithmetic: base + (donor_closed - ref_open)
        ref_open_pool = np.flatnonzero(in_oth_dir & (grip_lbl == "open"))
        ri = rng_c.choice(ref_open_pool, size=n_c, replace=True)

        zb = z4_nt[bi].copy()
        zd = z4_nt[di].copy()
        zr = z4_nt[ri].copy()
        z_tok_sw = zb.copy()
        z_tok_sw[:, best_grip_tok, :] = zd[:, best_grip_tok, :]

        zb_op = zb[:, :, :op_dim].reshape(n_c, -1) - mu_op
        zd_op = zd[:, :, :op_dim].reshape(n_c, -1) - mu_op
        cb = (zb_op @ V_pca.T - pca_mean) @ W_ica.T
        cd = (zd_op @ V_pca.T - pca_mean) @ W_ica.T
        dc = np.zeros_like(cb)
        dc[:, best_grip_ica] = cd[:, best_grip_ica] - cb[:, best_grip_ica]
        z_ica_sw = zb.copy()
        z_ica_sw[:, :, :op_dim] = (zb_op + (dc @ M_ica.T) @ V_pca + mu_op).reshape(n_c, 4, op_dim)

        z_op_ar = zb.copy()
        z_op_ar[:, :, :op_dim] = zb[:, :, :op_dim] + (zd[:, :, :op_dim] - zr[:, :, :op_dim])

        ab, ad, atok, aica, aop = ddim_decode_fixed_noise(
            policy,
            [zb.reshape(n_c, -1), zd.reshape(n_c, -1), z_tok_sw.reshape(n_c, -1), z_ica_sw.reshape(n_c, -1), z_op_ar.reshape(n_c, -1)],
            device, seed=12345, batch_size=512
        )
        # Evaluate compositional success: (1) gripper flipped to closed (>0), AND (2) translation direction preserved (cos > 0.5 with base)!
        def _comp_metrics(pa: np.ndarray) -> Dict[str, float]:
            g_succ = (pa[:, :, 9].mean(axis=1) > 0)
            p_b = ab[:, :, :3].mean(axis=1)
            p_s = pa[:, :, :3].mean(axis=1)
            cos_d = (p_b * p_s).sum(axis=1) / np.maximum(np.linalg.norm(p_b, axis=1) * np.linalg.norm(p_s, axis=1), 1e-6)
            dir_pres = cos_d > 0.5
            return {
                "grip_flip_rate": float(g_succ.mean()),
                "dir_cos_vs_base": float(cos_d.mean()),
                "dir_preserve_rate": float(dir_pres.mean()),
                "joint_compositional_success": float((g_succ & dir_pres).mean()),
            }

        comp_gen = {
            "best_grip_tok": best_grip_tok,
            "best_grip_ica_axes": [int(x) for x in best_grip_ica],
            "optimal_token_swap": _comp_metrics(atok),
            "ica_subspace_swap": _comp_metrics(aica),
            "op_subspace_arith": _comp_metrics(aop),
        }

    # 5. Dense Temporal Factor Dynamics Along Continuous Validation Episodes (stride=2 frames!)
    # Encode 48 complete diverse val episodes densely from frame_feats_mmap -> true autocorrelation half-lives & changepoint AUC!
    val_ep_ids = sorted({int(e) for e in cache.val_episodes if int(cache.ep_lengths_np[int(e)]) >= 96})
    rng_t = np.random.default_rng(1729)
    sel_eps = np.sort(rng_t.choice(val_ep_ids, size=min(48, len(val_ep_ids)), replace=False))
    stride = 2
    lags = (2, 4, 8, 16, 32, 48, 64)
    lag_corrs_tok = {k: {lg: [] for lg in lags} for k in range(4)}
    lag_corrs_ica = {k: {lg: [] for lg in lags} for k in range(n_ica)}
    vel_scores, vel_op_scores, event_labels = [], [], []

    # Collect all dense (ep, frame) queries and encode in fast GPU batches
    ep_slices: List[Tuple[int, int, int]] = []
    q_eps_l, q_frs_l = [], []
    cur = 0
    for ep in sel_eps:
        T = int(cache.ep_lengths_np[int(ep)])
        frs = np.arange(0, max(T - 16, 1), stride, dtype=np.int64)
        if len(frs) < 20:
            continue
        q_eps_l.append(np.full(len(frs), int(ep), dtype=np.int64))
        q_frs_l.append(frs)
        ep_slices.append((int(ep), cur, cur + len(frs)))
        cur += len(frs)

    q_eps_all = np.concatenate(q_eps_l, axis=0)
    q_frs_all = np.concatenate(q_frs_l, axis=0)
    N_dense = len(q_eps_all)
    Z4_dense = np.zeros((N_dense, 4, 768), dtype=np.float32)
    A10_dense = np.zeros((N_dense, 16, 10), dtype=np.float32)
    for s in range(0, N_dense, 512):
        e = min(s + 512, N_dense)
        cr, cm, a10_d, _ = cache.gather(q_eps_all[s:e], q_frs_all[s:e])
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            b_d = policy.encode_local_behavior(cr.to(device), cm.to(device))
        Z4_dense[s:e] = b_d.float().cpu().numpy()
        A10_dense[s:e] = a10_d.numpy()

    Z_op_dense = Z4_dense[:, :, :op_dim].reshape(N_dense, -1)
    Z_full_dense = Z4_dense.reshape(N_dense, -1)
    ica_dense = ((Z_op_dense - mu_op) @ V_pca.T - pca_mean) @ W_ica.T
    ica_dense_n = (ica_dense - ica_dense.mean(0, keepdims=True)) / np.maximum(ica_dense.std(0, keepdims=True), 1e-6)

    for ep, s0, s1 in ep_slices:
        z4_e = Z4_dense[s0:s1]
        zf_e = Z_full_dense[s0:s1]
        zo_e = Z_op_dense[s0:s1]
        ic_e = ica_dense_n[s0:s1]
        a_e = A10_dense[s0:s1]
        L_e = s1 - s0

        for lg in lags:
            step_lg = lg // stride
            if step_lg <= 0 or step_lg >= L_e:
                continue
            for k in range(4):
                u = z4_e[:-step_lg, k]
                v = z4_e[step_lg:, k]
                c = np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-8)
                lag_corrs_tok[k][lg].append(float(np.mean(c)))
            for k in range(n_ica):
                lag_corrs_ica[k][lg].append(float(np.mean(ic_e[:-step_lg, k] * ic_e[step_lg:, k])))

        # Subgoal changepoint alignment at 4-frame step (step_d=2 -> 0.2s)
        g_e = a_e[:, 0, 9]
        p_e = a_e[:, :4, :3].mean(axis=1)
        for j in range(1, L_e - 1):
            dz_f = float(np.linalg.norm(zf_e[j + 1] - zf_e[j - 1]))
            dz_o = float(np.linalg.norm(zo_e[j + 1] - zo_e[j - 1]))
            # Physical event within +/-4 frames: gripper sign flip OR sharp velocity reversal
            lo_w, hi_w = max(0, j - 2), min(L_e - 1, j + 2)
            grip_chg = (g_e[lo_w] * g_e[hi_w]) < 0
            p0, p1 = p_e[lo_w], p_e[hi_w]
            n0, n1 = float(np.linalg.norm(p0)), float(np.linalg.norm(p1))
            dir_rev = (n0 > 0.15 and n1 > 0.15 and (float(np.dot(p0, p1)) / (n0 * n1)) < 0.0)
            vel_scores.append(dz_f)
            vel_op_scores.append(dz_o)
            event_labels.append(1 if (grip_chg or dir_rev) else 0)

    def _half_life(curve: Dict[int, float]) -> float:
        prev_lg, prev_v = 0.0, 1.0
        for lg in lags:
            v = curve[lg]
            if v < 0.5:
                # Linear interpolation for smooth half-life in frames
                frac = (prev_v - 0.5) / max(prev_v - v, 1e-6)
                return float(prev_lg + frac * (lg - prev_lg))
            prev_lg, prev_v = float(lg), v
        return float(lags[-1])

    tok_autocorr = {
        f"tok{k}": {str(lg): (float(np.mean(lag_corrs_tok[k][lg])) if lag_corrs_tok[k][lg] else float("nan")) for lg in lags}
        for k in range(4)
    }
    ica_autocorr = {}
    ica_half_lives = []
    for k in range(n_ica):
        c_k = {lg: (float(np.mean(lag_corrs_ica[k][lg])) if lag_corrs_ica[k][lg] else 0.0) for lg in lags}
        hl = _half_life(c_k)
        ica_autocorr[f"ica{k}"] = {
            "curve": {str(lg): c_k[lg] for lg in lags},
            "half_life_frames": hl,
            "dom_beh_prop": ica_probe["dominant_beh_prop_per_factor"][k],
        }
        ica_half_lives.append(hl)

    cp_auc_full = _roc_auc_binary(np.array(vel_scores), np.array(event_labels)) if vel_scores else 0.5
    cp_auc_op = _roc_auc_binary(np.array(vel_op_scores), np.array(event_labels)) if vel_op_scores else 0.5

    tot_ef = sum(tok_energy_full)
    tot_eo = sum(tok_energy_op)
    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "token_probes_full": tok_probe_full,
        "token_probes_op": tok_probe_op,
        "token_energy_frac_full": [float(e / max(tot_ef, 1e-12)) for e in tok_energy_full],
        "token_energy_frac_op": [float(e / max(tot_eo, 1e-12)) for e in tok_energy_op],
        "inter_token_cka_full_mean": float(cka_tok_full[off4].mean()),
        "inter_token_cka_op_mean": float(cka_tok_op[off4].mean()),
        "inter_token_cka_op_matrix": cka_tok_op.tolist(),
        "causal_token_swap_full": hungarian_full,
        "causal_token_swap_op": hungarian_op,
        "unsupervised_pca_factors": pca_probe,
        "unsupervised_ica_factors": ica_probe,
        "unsupervised_subspace_interventions": subspace_interventions,
        "compositional_generalization_unseen_dir_grip": comp_gen,
        "temporal_dynamics": {
            "tok_autocorr": tok_autocorr,
            "ica_autocorr": ica_autocorr,
            "ica_half_life_min": float(min(ica_half_lives)),
            "ica_half_life_max": float(max(ica_half_lives)),
            "ica_half_life_ratio_max_min": float(max(ica_half_lives) / max(min(ica_half_lives), 1.0)),
            "changepoint_subgoal_auc_full": cp_auc_full,
            "changepoint_subgoal_auc_op": cp_auc_op,
            "n_transitions_tested": int(len(vel_scores)),
            "event_rate": float(np.mean(event_labels)) if event_labels else 0.0,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp2_emergent_factors")
    p.add_argument("--extra-ckpts", type=str, nargs="*", default=[], help="Extra name=ckpt")
    p.add_argument("--only-extra", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=42)
    beyond = _gather_beyond_e12_targets(cache)

    non_tail = np.flatnonzero(~beyond["is_tail"])
    swap_pairs = _select_swap_pairs(
        beyond["val_a10"][non_tail], beyond["val_a7"][non_tail], cache.val_task_ids[non_tail], num_pairs=256, seed=42
    )

    targets: List[Tuple[str, Path]] = []
    if not args.only_extra:
        targets.extend([
            ("e16_d1_lam", E16_DECODER_CKPTS["e16_d1_lam_dec100"] / "last_ema.pt"),
            ("e16_d3_vid_op", E16_DECODER_CKPTS["e16_d3_vidop_dec100"] / "last_ema.pt"),
            ("e16_d4_learn_op", E16_DECODER_CKPTS["e16_d4_learnop_dec100"] / "last_ema.pt"),
            ("e16_d_oracle_e9", E16_DECODER_CKPTS["e16_d_oracle_e9_dec100"] / "last_ema.pt"),
            ("e16_c2_op_learnable", E16_REGIME1_LEARN_CKPTS["e16_c2_op_learnable"]),
            ("e16_c3_op_learnable_tok", E16_REGIME1_LEARN_CKPTS["e16_c3_op_learnable_tok"]),
            ("e12_bc_factordisp", REFERENCE_CHECKPOINTS["e12_bc_factordisp"]),
        ])
    for spec in args.extra_ckpts:
        name, ck = spec.split("=", 1)
        targets.append((name, Path(ck)))

    summary_path = args.out_dir / "emergent_factors_summary.json"
    results: Dict[str, Any] = json.loads(summary_path.read_text()) if (args.only_extra and summary_path.is_file()) else {}

    for name, ck in targets:
        if not ck.is_file():
            print(f"[Exp2] Skip missing {name}: {ck}", flush=True)
            continue
        res = analyze_emergent_factors(name, ck, cache, beyond, swap_pairs, device)
        results[name] = res
        hf = res["causal_token_swap_full"]
        ica_i = res["unsupervised_subspace_interventions"]["ica_top3"]
        cg = res["compositional_generalization_unseen_dir_grip"]
        td = res["temporal_dynamics"]
        print(
            f"[Exp2:{name:24s}] TokCKA_op={res['inter_token_cka_op_mean']:.3f} | "
            f"TokSwap Hung={hf['hungarian_mean_adoption']*100:.1f}% (Fixed={hf['fixed_e12_mean_adoption']*100:.1f}%, Sel={hf['hungarian_selectivity']:.2f}) | "
            f"ICA-SubSwap={ica_i['mean_adoption']*100:.1f}% (Drift={ica_i['mean_drift']*100:.1f}%, Sel={ica_i['mean_selectivity']:.2f}, DCI_D={res['unsupervised_ica_factors']['dci_disentanglement_beh']:.3f}) | "
            f"CompUnseen(Tok/ICA/OpAr)={cg.get('optimal_token_swap',{}).get('joint_compositional_success',0)*100:.1f}%/"
            f"{cg.get('ica_subspace_swap',{}).get('joint_compositional_success',0)*100:.1f}%/"
            f"{cg.get('op_subspace_arith',{}).get('joint_compositional_success',0)*100:.1f}% | "
            f"SubgoalAUC(op)={td['changepoint_subgoal_auc_op']:.3f}",
            flush=True,
        )
        atomic_write_json(results, summary_path)

    print(f"[Exp2] Saved emergent factor analysis to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
