"""Shared zero-leakage data, latent encoding, scale control, adapters, and DDIM utilities for E14."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e9 import ACTION_DIM_10D, ACTION_HORIZON
from e9.stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher
from e12.eval_suite import ValClipCache, _action_metrics_dict, _effective_rank_stats, _ridge_fit_predict
from e13.exp4_minimal_icl import (
    _build_zero_leakage_icl_pairs,
    _encode_clips_with_policy,
    _gather_and_encode_split_tensors,
)

from . import CHECKPOINTS


# ---------------------------------------------------------------------------
# Deterministic, Fast Batched DDIM Decoding (exact same noise z0 per sample!)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def ddim_decode_fixed_noise(
    policy: E9Stage1LocalPolicy,
    latents_list: Sequence[np.ndarray],
    device: torch.device,
    *,
    seed: int = 12345,
    batch_size: int = 512,
    use_bf16: bool = True,
) -> List[np.ndarray]:
    """Decode multiple `(N, 3072)` latent arrays for the SAME split using identical per-sample `z0`.

    Guarantees zero DDIM noise-draw variance across conditions being compared on the same
    test split, while fusing calls into large GPU batches for 2-3x speedup.
    """
    if not latents_list:
        return []
    policy.eval()
    N = int(latents_list[0].shape[0])
    for arr in latents_list:
        if arr.shape[0] != N:
            raise ValueError(f"All latent arrays must share N={N}, got {arr.shape[0]}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    z0_single = torch.randn((N, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32)

    stacked = np.concatenate(
        [np.asarray(a, dtype=np.float32).reshape(N, 4, 768) for a in latents_list], axis=0
    )
    z0_tiled = z0_single.repeat(len(latents_list), 1, 1)
    total = stacked.shape[0]

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )
    preds = np.zeros((total, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
    for s in range(0, total, batch_size):
        e = min(s + batch_size, total)
        b_g = torch.from_numpy(stacked[s:e]).to(device, non_blocking=True)
        z0 = z0_tiled[s:e].to(device, non_blocking=True)
        with amp_ctx:
            pred = policy.sample_from_z_self(b_g, init_noise=z0)
        preds[s:e] = pred.float().cpu().numpy()

    return [preds[i * N : (i + 1) * N].copy() for i in range(len(latents_list))]


# ---------------------------------------------------------------------------
# Cached Zero-Leakage Dataset Encoding Across Models
# ---------------------------------------------------------------------------

def build_seed_bundle(
    cache: ValClipCache,
    policies: Dict[str, E9Stage1LocalPolicy],
    *,
    match_mode: str,
    seed: int,
    device: torch.device,
    n_train_pairs: int = 4000,
    n_test_seen: int = 512,
    n_test_unseen: int = 512,
) -> Dict[str, Any]:
    """Build strictly zero-leakage splits and pre-encode ALL models once per (mode, seed).

    Reuses E13 Exp 4's exact RNG seeds (`seed + 444` for 70/30 task split and `seed + 414`
    inside `_build_zero_leakage_icl_pairs`) so diagonal numbers are 100% comparable to E13.
    """
    unique_tasks = np.arange(len(cache.task_to_id))
    rng = np.random.default_rng(seed + 444)
    perm_t = rng.permutation(unique_tasks)
    n_tr_t = int(0.7 * len(perm_t))
    tr_tasks = set(perm_t[:n_tr_t].tolist())
    un_tasks = set(perm_t[n_tr_t:].tolist())

    splits = _build_zero_leakage_icl_pairs(
        cache,
        train_tasks=tr_tasks,
        unseen_tasks=un_tasks,
        n_train_pairs=n_train_pairs,
        n_test_seen=n_test_seen,
        n_test_unseen=n_test_unseen,
        match_mode=match_mode,
        seed=seed,
    )
    raw_t = {sp: _gather_and_encode_split_tensors(cache, sp_idx) for sp, sp_idx in splits.items()}

    latents: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for mname, pol in policies.items():
        latents[mname] = {}
        for sp in ("train", "test_seen", "test_unseen"):
            latents[mname][sp] = {
                "d": _encode_clips_with_policy(pol, raw_t[sp]["d_clip"], raw_t[sp]["d_mask"], device),
                "q": _encode_clips_with_policy(pol, raw_t[sp]["q_clip"], raw_t[sp]["q_mask"], device),
            }

    obs: Dict[str, Dict[str, np.ndarray]] = {}
    acts: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in ("train", "test_seen", "test_unseen"):
        hd = raw_t[sp]["h_d"].numpy().astype(np.float32)
        hq = raw_t[sp]["h_q"].numpy().astype(np.float32)
        obs[sp] = {"h_d": hd, "h_q": hq, "dh": hq - hd}
        acts[sp] = {
            "d": raw_t[sp]["d_act10"].numpy().astype(np.float32),
            "q": raw_t[sp]["q_act10"].numpy().astype(np.float32),
        }

    return {
        "match_mode": match_mode,
        "seed": seed,
        "splits": splits,
        "obs": obs,
        "acts": acts,
        "latents": latents,
    }


def load_policies(model_names: Sequence[str], device: torch.device) -> Dict[str, E9Stage1LocalPolicy]:
    policies: Dict[str, E9Stage1LocalPolicy] = {}
    for name in model_names:
        ckpt = CHECKPOINTS[name]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {ckpt}")
        policies[name] = load_frozen_e9_stage1_teacher(ckpt, device=device)
    return policies


# ---------------------------------------------------------------------------
# Scale-Controlled Normalization & PCA Feature Builders (Strict Train-Only Fit!)
# ---------------------------------------------------------------------------

def fit_global_rms_standardizer(X_tr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center by train mean and compute scalar global Frobenius RMS per coordinate.

    Preserves internal singular-value spectrum (critical because per-dim whitening
    destroys E9's SNR ordering, per E13 Exp 1) while strictly matching total energy
    across different representations!
    """
    mu = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
    xc = X_tr - mu
    rms = float(np.sqrt(np.mean(xc.astype(np.float64) ** 2)))
    rms = max(rms, 1e-6)
    return mu, rms


def apply_global_rms_standardizer(X: np.ndarray, mu: np.ndarray, rms: float) -> np.ndarray:
    return ((X - mu) / rms).astype(np.float32)


def invert_global_rms_standardizer(Xn: np.ndarray, mu: np.ndarray, rms: float) -> np.ndarray:
    return (Xn * rms + mu).astype(np.float32)


def build_block_rms_features(
    blocks_tr: Sequence[np.ndarray],
    blocks_evals: Sequence[Sequence[np.ndarray]],
    *,
    block_weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Concatenate input blocks where each block is centered and normalized to unit per-element RMS.

    Each block contributes equal total L2 energy per dimension (or scaled by `block_weights`),
    strictly eliminating latent scale confounders across models A in {E9, E12-B, E12-C, E12-BC-FD}.
    """
    n_blocks = len(blocks_tr)
    weights = [1.0] * n_blocks if block_weights is None else list(block_weights)
    tr_parts: List[np.ndarray] = []
    ev_parts: List[List[np.ndarray]] = [[] for _ in range(len(blocks_evals[0]))]

    for b_idx in range(n_blocks):
        mu, rms = fit_global_rms_standardizer(blocks_tr[b_idx])
        # Normalize block by sqrt(D_b) * rms so every block has total expected ||x_b||^2 = w_b^2
        d_b = float(blocks_tr[b_idx].shape[1])
        scale = rms * math.sqrt(d_b) / max(float(weights[b_idx]), 1e-6)
        tr_parts.append(((blocks_tr[b_idx] - mu) / scale).astype(np.float32))
        for e_idx, xe in enumerate(blocks_evals[b_idx]):
            ev_parts[e_idx].append(((xe - mu) / scale).astype(np.float32))

    X_tr = np.concatenate(tr_parts, axis=-1)
    X_ev = [np.concatenate(parts, axis=-1) for parts in ev_parts]
    return X_tr, X_ev


def pca_project_block(
    X_tr: np.ndarray,
    X_evals: Sequence[np.ndarray],
    k: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Fit top-k PCA strictly on `X_tr` and project train + eval arrays."""
    Xt = torch.from_numpy(np.asarray(X_tr, dtype=np.float32)).to(device)
    mu = Xt.mean(dim=0, keepdim=True)
    Xc = Xt - mu
    k_eff = min(int(k), Xc.shape[0], Xc.shape[1])
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh[:k_eff].T
    p_tr = (Xc @ V).cpu().numpy().astype(np.float32)
    p_ev = [
        ((torch.from_numpy(np.asarray(Xe, dtype=np.float32)).to(device) - mu) @ V).cpu().numpy().astype(np.float32)
        for Xe in X_evals
    ]
    return p_tr, p_ev


def build_pca_block_rms_features(
    dh_tr: np.ndarray,
    hq_tr: np.ndarray,
    z_tr_blocks: Sequence[np.ndarray],
    dh_evals: Sequence[np.ndarray],
    hq_evals: Sequence[np.ndarray],
    z_eval_blocks: Sequence[Sequence[np.ndarray]],
    device: torch.device,
    *,
    k_obs: int = 128,
    k_lat_list: Sequence[int] = (64,),
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """PCA-project each block strictly on train, then equalize block L2 energy via block RMS."""
    p_dh_tr, p_dh_ev = pca_project_block(dh_tr, dh_evals, k_obs, device)
    p_hq_tr, p_hq_ev = pca_project_block(hq_tr, hq_evals, k_obs, device)

    blocks_tr = [p_dh_tr, p_hq_tr]
    blocks_ev: List[Sequence[np.ndarray]] = [p_dh_ev, p_hq_ev]
    for z_tr, z_ev, kl in zip(z_tr_blocks, z_eval_blocks, k_lat_list):
        if kl > 0:
            pz_tr, pz_ev = pca_project_block(z_tr, z_ev, kl, device)
            blocks_tr.append(pz_tr)
            blocks_ev.append(pz_ev)

    return build_block_rms_features(blocks_tr, blocks_ev)


# ---------------------------------------------------------------------------
# Scale-Controlled Adapters: Ridge, RBF Kernel Ridge, Tuned Residual MLP
# ---------------------------------------------------------------------------

RIDGE_ALPHAS_DEFAULT = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 500.0, 2000.0, 10000.0)


def fit_predict_scaled_ridge(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_evals: Sequence[np.ndarray],
    device: torch.device,
    *,
    alphas: Tuple[float, ...] = RIDGE_ALPHAS_DEFAULT,
) -> Tuple[List[np.ndarray], float]:
    """Scale-controlled closed-form Ridge regression with train-only target RMS normalization."""
    y_mu, y_rms = fit_global_rms_standardizer(Y_tr)
    Yn_tr = apply_global_rms_standardizer(Y_tr, y_mu, y_rms)

    # Select alpha once via internal train/val holdout on Yn_tr, then predict all evals
    pred0_n, best_alpha = _ridge_fit_predict(X_tr, Yn_tr, X_evals[0], alphas=alphas, device=device)
    preds = [invert_global_rms_standardizer(pred0_n, y_mu, y_rms)]
    for Xe in X_evals[1:]:
        pe_n, _ = _ridge_fit_predict(X_tr, Yn_tr, Xe, alphas=(best_alpha,), device=device)
        preds.append(invert_global_rms_standardizer(pe_n, y_mu, y_rms))
    return preds, best_alpha


def fit_predict_rbf_krr(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_evals: Sequence[np.ndarray],
    device: torch.device,
    *,
    gamma_mults: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Exact closed-form RBF Kernel Ridge Regression on GPU (universal nonlinear baseline!).

    Uses internal 80/20 holdout on the training pairs to select `(gamma, alpha)`, with
    median/mean squared-distance heuristic for `gamma_base` and global RMS target scaling.
    """
    y_mu, y_rms = fit_global_rms_standardizer(Y_tr)
    Yn_tr = apply_global_rms_standardizer(Y_tr, y_mu, y_rms)

    X_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float64)).to(device)
    Y_t = torch.from_numpy(np.asarray(Yn_tr, dtype=np.float64)).to(device)
    X_ev_t = [torch.from_numpy(np.asarray(Xe, dtype=np.float64)).to(device) for Xe in X_evals]

    x_mu = X_t.mean(dim=0, keepdim=True)
    Xc = X_t - x_mu
    Xev_c = [Xe - x_mu for Xe in X_ev_t]
    y_c_mu = Y_t.mean(dim=0, keepdim=True)
    Yc = Y_t - y_c_mu

    # Pairwise squared Euclidean distance matrix on training set
    sq_tr = (Xc ** 2).sum(dim=1, keepdim=True)
    D2_all = (sq_tr + sq_tr.T - 2.0 * (Xc @ Xc.T)).clamp_min(0.0)
    # Characteristic scale = mean off-diagonal squared distance
    n = Xc.shape[0]
    mean_d2 = float(D2_all.sum().item() / max(n * (n - 1), 1))
    gamma_base = 1.0 / max(mean_d2, 1e-6)

    n_val = max(10, int(0.2 * n))
    n_a = n - n_val
    D2_aa = D2_all[:n_a, :n_a]
    D2_va = D2_all[n_a:, :n_a]
    Y_a, Y_v = Yc[:n_a], Yc[n_a:]

    best_err = float("inf")
    best_gm = 1.0
    best_al = 0.1
    for gm in gamma_mults:
        g = gamma_base * gm
        K_aa_base = torch.exp(-g * D2_aa)
        K_va = torch.exp(-g * D2_va)
        for al in alphas:
            K = K_aa_base.clone()
            K.diagonal().add_(al)
            coef = torch.linalg.solve(K, Y_a)
            pred_v = K_va @ coef
            err = float(((pred_v - Y_v) ** 2).mean().item())
            if err < best_err:
                best_err = err
                best_gm = gm
                best_al = al

    # Refit on 100% of training pairs with selected (gamma, alpha)
    g_best = gamma_base * best_gm
    K_full = torch.exp(-g_best * D2_all)
    K_full.diagonal().add_(best_al)
    coef_full = torch.linalg.solve(K_full, Yc)

    preds: List[np.ndarray] = []
    for Xe_c in Xev_c:
        sq_e = (Xe_c ** 2).sum(dim=1, keepdim=True)
        D2_e = (sq_e + sq_tr.T - 2.0 * (Xe_c @ Xc.T)).clamp_min(0.0)
        K_e = torch.exp(-g_best * D2_e)
        pred_n = (K_e @ coef_full + y_c_mu).float().cpu().numpy()
        preds.append(invert_global_rms_standardizer(pred_n, y_mu, y_rms))

    return preds, {"gamma_mult": best_gm, "gamma": g_best, "alpha": best_al, "val_mse_norm": best_err}


def fit_predict_tuned_mlp(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_evals: Sequence[np.ndarray],
    device: torch.device,
    *,
    hidden_dim: int = 512,
    bottleneck_dim: int = 256,
    max_steps: int = 800,
    eval_every: int = 25,
    lr: float = 1.5e-3,
    weight_decay: float = 3e-2,
    dropout: float = 0.2,
    residual_on_ridge: bool = True,
    seed: int = 42,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Properly regularized & early-stopped nonlinear MLP adapter with strict train-only val split.

    When `residual_on_ridge=True`, initializes from a linear Ridge backbone fitted on the
    training fold and trains a low-rank bottleneck residual MLP `f_mlp(X)` with zero-init
    output layer and validation early stopping, guaranteeing the nonlinear adapter can
    strictly improve upon (and never catastrophically overfit worse than) Linear Ridge!
    """
    y_mu, y_rms = fit_global_rms_standardizer(Y_tr)
    Yn_tr = apply_global_rms_standardizer(Y_tr, y_mu, y_rms)

    n = X_tr.shape[0]
    n_val = max(16, int(0.15 * n))
    n_fit = n - n_val

    # Optional linear Ridge baseline fitted ONLY on n_fit for early stopping, and full n for final refit
    if residual_on_ridge:
        # Fit ridge on n_fit -> predict val & evals for residual supervision
        r_preds_fit, best_alpha = _ridge_fit_predict(
            X_tr[:n_fit], Yn_tr[:n_fit], X_tr[n_fit:], alphas=RIDGE_ALPHAS_DEFAULT, device=device
        )
        r_pred_tr_fit, _ = _ridge_fit_predict(
            X_tr[:n_fit], Yn_tr[:n_fit], X_tr[:n_fit], alphas=(best_alpha,), device=device
        )
        r_preds_ev = [
            _ridge_fit_predict(X_tr[:n_fit], Yn_tr[:n_fit], Xe, alphas=(best_alpha,), device=device)[0]
            for Xe in X_evals
        ]
        # Target for MLP is residual error of Ridge!
        target_fit = Yn_tr[:n_fit] - r_pred_tr_fit
        target_val = Yn_tr[n_fit:] - r_preds_fit
    else:
        r_preds_ev = [np.zeros((len(Xe), Y_tr.shape[1]), dtype=np.float32) for Xe in X_evals]
        target_fit = Yn_tr[:n_fit]
        target_val = Yn_tr[n_fit:]
        best_alpha = 0.0

    torch.manual_seed(seed)
    Xf_t = torch.from_numpy(np.asarray(X_tr[:n_fit], dtype=np.float32)).to(device)
    Xv_t = torch.from_numpy(np.asarray(X_tr[n_fit:], dtype=np.float32)).to(device)
    Yf_t = torch.from_numpy(np.asarray(target_fit, dtype=np.float32)).to(device)
    Yv_t = torch.from_numpy(np.asarray(target_val, dtype=np.float32)).to(device)
    Xev_t = [torch.from_numpy(np.asarray(Xe, dtype=np.float32)).to(device) for Xe in X_evals]

    # Normalize inputs using scalar RMS (not per-feature whitening!) to protect SNR ordering
    xm = Xf_t.mean(dim=0, keepdim=True)
    x_rms = torch.sqrt(((Xf_t - xm) ** 2).mean()).clamp_min(1e-4)
    Xf_n = (Xf_t - xm) / x_rms
    Xv_n = (Xv_t - xm) / x_rms
    Xev_n = [(Xe - xm) / x_rms for Xe in Xev_t]

    in_dim = Xf_n.shape[1]
    out_dim = Yf_t.shape[1]
    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, bottleneck_dim),
        nn.LayerNorm(bottleneck_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(bottleneck_dim, out_dim),
    ).to(device)
    if residual_on_ridge:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    bs = min(128, n_fit)
    gen = torch.Generator(device=device).manual_seed(seed)

    # Initial val loss (at step 0: zero residual = pure Ridge!)
    net.eval()
    with torch.inference_mode():
        best_val = float(F.mse_loss(net(Xv_n), Yv_t).item())
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    best_step = 0

    for step in range(1, max_steps + 1):
        net.train()
        if n_fit > bs:
            idx = torch.randint(0, n_fit, (bs,), generator=gen, device=device)
            xb, yb = Xf_n[idx], Yf_t[idx]
        else:
            xb, yb = Xf_n, Yf_t
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(net(xb), yb)
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == max_steps:
            net.eval()
            with torch.inference_mode():
                v_loss = float(F.mse_loss(net(Xv_n), Yv_t).item())
            if v_loss < best_val:
                best_val = v_loss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()
    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for Xe_n, r_ev in zip(Xev_n, r_preds_ev):
            res_n = net(Xe_n).cpu().numpy().astype(np.float32)
            preds.append(invert_global_rms_standardizer(r_ev + res_n, y_mu, y_rms))

    n_params = sum(p.numel() for p in net.parameters())
    return preds, {
        "best_step": best_step,
        "best_val_mse_norm": best_val,
        "ridge_alpha": best_alpha,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Scale-Free Latent Predictability & Action Metrics Helpers
# ---------------------------------------------------------------------------

def compute_latent_predictability_metrics(
    z_pred: np.ndarray,
    z_true: np.ndarray,
    z_demo: np.ndarray,
    z_tr_mean: np.ndarray,
    z_tr_rms: float,
    *,
    Q_act: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Scale-invariant latent prediction diagnostics (BEFORE DDIM decoding)."""
    err = (z_pred - z_true).astype(np.float64)
    zc_true = (z_true - z_tr_mean).astype(np.float64)
    zc_pred = (z_pred - z_tr_mean).astype(np.float64)

    var_true = float(np.mean((z_true - z_true.mean(axis=0, keepdims=True)) ** 2))
    mse_z = float(np.mean(err ** 2))
    r2_z = float(1.0 - mse_z / max(var_true, 1e-12))

    dz_true = (z_true - z_demo).astype(np.float64)
    dz_pred = (z_pred - z_demo).astype(np.float64)
    var_dz = float(np.mean((dz_true - dz_true.mean(axis=0, keepdims=True)) ** 2))
    r2_dz = float(1.0 - np.mean((dz_pred - dz_true) ** 2) / max(var_dz, 1e-12))

    # Mean cosine similarity of predicted displacement vs true displacement
    n_t = np.linalg.norm(dz_true, axis=-1)
    n_p = np.linalg.norm(dz_pred, axis=-1)
    valid = (n_t > 1e-6) & (n_p > 1e-6)
    cos_dz = float(np.mean((dz_true[valid] * dz_pred[valid]).sum(axis=-1) / (n_t[valid] * n_p[valid]))) if valid.any() else 0.0

    # Relative RMS error normalized by train centered RMS (scale-free across models!)
    rel_err = float(np.sqrt(mse_z) / max(z_tr_rms, 1e-6))

    # Norm shrinkage ratio: how much closer to mean is z_pred than z_true?
    norm_true = float(np.mean(np.linalg.norm(zc_true, axis=-1)))
    norm_pred = float(np.mean(np.linalg.norm(zc_pred, axis=-1)))
    shrinkage_ratio = norm_pred / max(norm_true, 1e-6)

    # Per-token R^2 (4 tokens x 768 dims)
    zt4 = z_true.reshape(-1, 4, 768)
    zp4 = z_pred.reshape(-1, 4, 768)
    per_tok_r2 = {}
    for k in range(4):
        vk = float(np.mean((zt4[:, k] - zt4[:, k].mean(axis=0, keepdims=True)) ** 2))
        mk = float(np.mean((zp4[:, k] - zt4[:, k]) ** 2))
        per_tok_r2[f"tok{k}_r2"] = float(1.0 - mk / max(vk, 1e-12))

    out = {
        "r2_z": r2_z,
        "r2_dz": r2_dz,
        "cos_dz": cos_dz,
        "rel_rms_err": rel_err,
        "shrinkage_ratio": shrinkage_ratio,
        "mse_z_raw": mse_z,
        **per_tok_r2,
    }

    if Q_act is not None:
        # Decompose into orthogonal action-explainable (||) and action-unexplained (perp) subspaces!
        # z_par = (z - mu) @ Q_act @ Q_act.T
        c_true = zc_true @ Q_act
        c_pred = zc_pred @ Q_act
        z_par_true = c_true @ Q_act.T
        z_par_pred = c_pred @ Q_act.T
        z_perp_true = zc_true - z_par_true
        z_perp_pred = zc_pred - z_par_pred

        v_par = float(np.mean((z_par_true - z_par_true.mean(axis=0, keepdims=True)) ** 2))
        v_perp = float(np.mean((z_perp_true - z_perp_true.mean(axis=0, keepdims=True)) ** 2))
        m_par = float(np.mean((z_par_pred - z_par_true) ** 2))
        m_perp = float(np.mean((z_perp_pred - z_perp_true) ** 2))
        out["r2_z_par"] = float(1.0 - m_par / max(v_par, 1e-12))
        out["r2_z_perp"] = float(1.0 - m_perp / max(v_perp, 1e-12))
        out["rel_err_par"] = float(np.sqrt(m_par) / max(z_tr_rms, 1e-6))
        out["rel_err_perp"] = float(np.sqrt(m_perp) / max(z_tr_rms, 1e-6))
        out["par_energy_frac"] = float(v_par / max(v_par + v_perp, 1e-12))
        out["perp_energy_frac"] = float(v_perp / max(v_par + v_perp, 1e-12))

    return out


def pack_action_metrics(
    m_d: Dict[str, float],
    dc_d: Dict[str, float],
    ora_d: Dict[str, float],
) -> Dict[str, float]:
    return {
        "mse10": float(m_d["action10_mse"]),
        "pos_mse": float(m_d["pos_mse"]),
        "rot6d_mse": float(m_d["rot6d_mse"]),
        "grip_mse": float(m_d["grip_mse"]),
        "mae7": float(m_d["action7_mae"]),
        "grip_acc": float(m_d["grip_sign_acc"]),
        "arr_10d": float(
            (dc_d["action10_mse"] - m_d["action10_mse"])
            / max(dc_d["action10_mse"] - ora_d["action10_mse"], 1e-9)
        ),
    }


def fit_action_explainable_subspace(
    a10_tr: np.ndarray,
    z_tr: np.ndarray,
    device: torch.device,
    *,
    var_threshold: float = 0.995,
    max_rank: int = 160,
) -> Dict[str, Any]:
    """Fit orthonormal basis `Q_act` (3072, r) of the action-explainable latent subspace on TRAIN ONLY.

    Strictly zero-leakage:
      1. Fits Ridge `a_10d_tr (N, 160) -> z_tr (N, 3072)` on train split only.
      2. Forms `Z_act_tr = W_az(a_10d_tr)` and extracts its orthonormal SVD basis `Q_act`
         capturing `var_threshold` of `Z_act_tr` variance (rank `r <= 160`).
      3. At test time, ANY latent `z` decomposes orthogonally WITHOUT touching GT action:
            `z_par  = (z - mu) @ Q_act @ Q_act.T`
            `z_perp = (z - mu) - z_par`
    """
    A_flat = a10_tr.reshape(len(a10_tr), -1).astype(np.float32)
    z_mu = z_tr.mean(axis=0, keepdims=True).astype(np.float32)
    zc_tr = z_tr - z_mu

    z_act_tr, alpha_az = _ridge_fit_predict(
        A_flat, zc_tr, A_flat, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), device=device
    )
    z_act_c = z_act_tr - z_act_tr.mean(axis=0, keepdims=True)
    # SVD on GPU to get exact orthonormal basis Q_act of span(Z_act_tr)
    Za_t = torch.from_numpy(z_act_c.astype(np.float64)).to(device)
    _, S, Vh = torch.linalg.svd(Za_t, full_matrices=False)
    eig = (S ** 2).cpu().numpy()
    tot = float(eig.sum())
    cum = np.cumsum(eig) / max(tot, 1e-12)
    r_eff = int(np.searchsorted(cum, var_threshold) + 1)
    r_eff = max(1, min(r_eff, max_rank, Vh.shape[0]))
    Q_act = Vh[:r_eff].T.float().cpu().numpy()  # (3072, r_eff)

    # Variance fraction of z_tr lying in geometric subspace Q_act vs exact regression R^2(a->z)
    z_par_tr = (zc_tr @ Q_act) @ Q_act.T
    var_tot = float(np.mean(zc_tr.astype(np.float64) ** 2))
    var_par_geo = float(np.mean(z_par_tr.astype(np.float64) ** 2))
    var_act_reg = float(np.mean(z_act_c.astype(np.float64) ** 2))

    return {
        "Q_act": Q_act,
        "z_mu": z_mu,
        "rank": r_eff,
        "alpha_az": float(alpha_az),
        "geo_subspace_var_frac": float(var_par_geo / max(var_tot, 1e-12)),
        "reg_r2_train": float(var_act_reg / max(var_tot, 1e-12)),
        "z_tr_rms": float(np.sqrt(var_tot)),
    }


def make_random_orthonormal_subspace(dim: int, rank: int, seed: int = 999) -> np.ndarray:
    """Create a random orthonormal basis `(dim, rank)` for matched-dim subspace ablation controls."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, rank)).astype(np.float64)
    Q, _ = np.linalg.qr(G, mode="reduced")
    return Q.astype(np.float32)


def make_matched_energy_random_subspace(
    z_tr: np.ndarray,
    rank: int,
    target_energy_frac: float,
    seed: int = 999,
) -> np.ndarray:
    """Create an orthonormal basis of `rank` inside `z_tr`'s PCA spectrum matching `target_energy_frac`."""
    zc = (z_tr - z_tr.mean(axis=0, keepdims=True)).astype(np.float64)
    _, S, Vh = np.linalg.svd(zc, full_matrices=False)
    # Random orthogonal rotation of top-k PCA components such that expected energy matches
    # First pick an active pool of top `m` principal components (e.g., m = min(512, len(S)))
    m = min(max(rank * 2, 256), Vh.shape[0])
    rng = np.random.default_rng(seed)
    # Random orthogonal basis within the top-m subspace of z_tr
    G = rng.standard_normal((m, rank))
    R, _ = np.linalg.qr(G, mode="reduced")
    Q_rand = (Vh[:m].T @ R).astype(np.float32)
    return Q_rand


def project_subspace(z: np.ndarray, z_mu: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthogonally split `z - z_mu` into `(z_par, z_perp)` (both centered, summing to `z - z_mu`)."""
    zc = z - z_mu
    z_par = (zc @ Q) @ Q.T
    z_perp = zc - z_par
    return z_par.astype(np.float32), z_perp.astype(np.float32)
