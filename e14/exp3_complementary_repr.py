"""E14 Experiment 3: Complementary Representation (E9 Context + E12-C/BC Operational Geometry).

Tests whether E9 (contextual information) and E12-C / E12-BC-FactorDisp (operational geometry)
are genuinely complementary under STRICT equal-dimension and equal-parameter controls:

  Part 3A — Zero-Leakage Visual ICL Fusion & Strict Capacity Controls:
    1. Matched-64 Latent Dim (`D_in = 128+128+64 = 320`, identical Ridge parameter count):
       - Single: E9(64), E12-C(64), E12-BC-FD(64), E9-Cont(64)
       - Concat (32+32=64): E9+E12-C, E9+E12-BC-FD, and same-family control E9+E9-Cont
       - Joint-PCA (64): JointPCA(E9, E12-C), JointPCA(E9, E12-BC-FD), control JointPCA(E9, E9-Cont)
       - Orthogonal Residual Supplementation (32+32=64):
           E9 + OrthRes(E12-C|E9), E9 + OrthRes(E12-BC-FD|E9), control E9 + OrthRes(E9-Cont|E9)
           E12-C + OrthRes(E9|E12-C), E12-BC-FD + OrthRes(E9|E12-BC-FD)
    2. Matched-128 Latent Dim (`D_in = 128+128+128 = 384`, identical Ridge parameter count):
       - Single (128): E9(128), E12-C(128), E12-BC-FD(128)
       - Concat (64+64=128): E9+E12-C, E9+E12-BC-FD, control E9+E9-Cont
    3. Full-Dim Block-RMS Concatenation vs Single & Same-Family `E9+E9-Cont` across all 7 budgets.
    4. Exact Parameter-Matched Lightweight Gated Fusion (`GatedFusionAdapter`):
       - `GatedFusion(E9, E12-C)` & `GatedFusion(E9, E12-BC-FD)`
       - vs exact-param `GatedSingle(E9[0:64], E9[64:128])`, `GatedSingle(E12C[0:64], E12C[64:128])`,
         and `GatedFusion(E9, E9-Cont)` (asserting exact identical `n_params`!).
    5. Dual-Decoder Action-Space Fusion (`w * a_hat^(E9) + (1-w) * a_hat^(E12)`, `w` selected
       strictly on internal train/val holdout) vs capacity-matched `Dual-Decoder(E9, E9-Cont)`,
       plus Per-Sample Complementarity / Win-Rate / Failure-Case Stratification!

  Part 3B — Dual-Regime Representation Audit (Context + Operational Geometry):
    Evaluates matched-dim (64D) and full-dim single vs fused representations on:
      - Future-Action Predictability `Fut_R2`, Visual Context `VisHQ_R2`, Object Linear Acc
      - Held-Out Unseen `(dir x grip)` Linear Action-Editability Transfer (`Unseen-Combo R^2 / MSE10`)
      - Zero-Leakage Visual ICL (`visual_knn` & `rel_progress`)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e12.eval_suite import ValClipCache, _action_metrics_dict, _ridge_fit_predict, encode_all_val_starts
from e13.exp2_action_baselines import _gather_future_actions

from . import BUDGETS_DEFAULT, MATCH_MODES_DEFAULT, OUTPUT_ROOT, SEEDS_DEFAULT
from .common import (
    RIDGE_ALPHAS_DEFAULT,
    apply_global_rms_standardizer,
    build_block_rms_features,
    build_pca_block_rms_features,
    build_seed_bundle,
    compute_latent_predictability_metrics,
    ddim_decode_fixed_noise,
    fit_global_rms_standardizer,
    fit_predict_scaled_ridge,
    invert_global_rms_standardizer,
    load_policies,
    pack_action_metrics,
    pca_project_block,
)


EXP3_MODELS = ("e9", "e12_c", "e12_bc_factordisp", "control")


# ---------------------------------------------------------------------------
# Exact Parameter-Matched Lightweight Gated Fusion Module
# ---------------------------------------------------------------------------

class GatedFusionAdapter(nn.Module):
    """Lightweight context-conditioned gated fusion adapter with fixed parameter count.

    Takes `(x_obs [256], z_a [64], z_b [64])`:
      - Projects each latent branch linearly into `d_fuse`
      - Computes sample-wise gate `g = sigmoid(MLP([x_obs, z_a, z_b]))`
      - Fuses `z_f = g * h_a + (1 - g) * h_b`
      - Predicts residual target correction via a compact bottleneck MLP
    Crucially, `GatedFusion(E9, E12-C)`, `GatedFusion(E9, E9-Cont)`, and
    `GatedSingle(E9[0:64], E9[64:128])` share the EXACT same architecture and parameter count!
    """

    def __init__(
        self,
        obs_dim: int = 256,
        branch_dim: int = 64,
        fuse_dim: int = 64,
        hidden_dim: int = 256,
        out_dim: int = 3072,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj_a = nn.Linear(branch_dim, fuse_dim)
        self.proj_b = nn.Linear(branch_dim, fuse_dim)
        self.gate = nn.Sequential(
            nn.Linear(obs_dim + 2 * branch_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, fuse_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(obs_dim + fuse_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # Zero-init output layer so residual starts cleanly from linear Ridge backbone
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_obs: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([x_obs, z_a, z_b], dim=-1)))
        z_f = g * self.proj_a(z_a) + (1.0 - g) * self.proj_b(z_b)
        out = self.head(torch.cat([x_obs, z_f], dim=-1))
        return out, g


def fit_predict_gated_fusion(
    x_obs_tr: np.ndarray,
    za_tr: np.ndarray,
    zb_tr: np.ndarray,
    Y_tr: np.ndarray,
    x_obs_ev: Sequence[np.ndarray],
    za_ev: Sequence[np.ndarray],
    zb_ev: Sequence[np.ndarray],
    device: torch.device,
    *,
    max_steps: int = 700,
    eval_every: int = 25,
    lr: float = 1.5e-3,
    weight_decay: float = 3e-2,
    seed: int = 42,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Train GatedFusionAdapter as a residual over an equal-dim linear Ridge backbone."""
    y_mu, y_rms = fit_global_rms_standardizer(Y_tr)
    Yn_tr = apply_global_rms_standardizer(Y_tr, y_mu, y_rms)

    # Combine inputs for the linear Ridge backbone (strictly same total dim = obs + 2*branch = 384!)
    X_lin_tr = np.concatenate([x_obs_tr, za_tr, zb_tr], axis=-1)
    X_lin_ev = [np.concatenate([o, a, b], axis=-1) for o, a, b in zip(x_obs_ev, za_ev, zb_ev)]

    n = len(X_lin_tr)
    n_val = max(16, int(0.15 * n))
    n_fit = n - n_val

    r_val, best_alpha = _ridge_fit_predict(
        X_lin_tr[:n_fit], Yn_tr[:n_fit], X_lin_tr[n_fit:], alphas=RIDGE_ALPHAS_DEFAULT, device=device
    )
    r_fit, _ = _ridge_fit_predict(
        X_lin_tr[:n_fit], Yn_tr[:n_fit], X_lin_tr[:n_fit], alphas=(best_alpha,), device=device
    )
    r_ev = [
        _ridge_fit_predict(X_lin_tr[:n_fit], Yn_tr[:n_fit], Xe, alphas=(best_alpha,), device=device)[0]
        for Xe in X_lin_ev
    ]

    res_fit = Yn_tr[:n_fit] - r_fit
    res_val = Yn_tr[n_fit:] - r_val

    def _norm_rms(tr_arr: np.ndarray, ev_list: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        mu, rms = fit_global_rms_standardizer(tr_arr[:n_fit])
        tf = torch.from_numpy(apply_global_rms_standardizer(tr_arr[:n_fit], mu, rms)).to(device)
        tv = torch.from_numpy(apply_global_rms_standardizer(tr_arr[n_fit:], mu, rms)).to(device)
        te = [torch.from_numpy(apply_global_rms_standardizer(e, mu, rms)).to(device) for e in ev_list]
        return tf, tv, te

    of, ov, oe = _norm_rms(x_obs_tr, x_obs_ev)
    af, av, ae = _norm_rms(za_tr, za_ev)
    bf, bv, be = _norm_rms(zb_tr, zb_ev)
    Yf = torch.from_numpy(res_fit.astype(np.float32)).to(device)
    Yv = torch.from_numpy(res_val.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    model = GatedFusionAdapter(
        obs_dim=of.shape[1], branch_dim=af.shape[1], fuse_dim=af.shape[1], hidden_dim=256, out_dim=Yf.shape[1]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    bs = min(128, n_fit)
    gen = torch.Generator(device=device).manual_seed(seed)

    model.eval()
    with torch.inference_mode():
        best_val = float(F.mse_loss(model(ov, av, bv)[0], Yv).item())
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step = 0

    for step in range(1, max_steps + 1):
        model.train()
        if n_fit > bs:
            idx = torch.randint(0, n_fit, (bs,), generator=gen, device=device)
            xo, xa, xb, yb = of[idx], af[idx], bf[idx], Yf[idx]
        else:
            xo, xa, xb, yb = of, af, bf, Yf
        opt.zero_grad(set_to_none=True)
        pred_r, _ = model(xo, xa, xb)
        loss = F.mse_loss(pred_r, yb)
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == max_steps:
            model.eval()
            with torch.inference_mode():
                v_loss = float(F.mse_loss(model(ov, av, bv)[0], Yv).item())
            if v_loss < best_val:
                best_val = v_loss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    preds: List[np.ndarray] = []
    gate_means: List[float] = []
    with torch.inference_mode():
        for xo, xa, xb, r_base in zip(oe, ae, be, r_ev):
            pr, g = model(xo, xa, xb)
            preds.append(invert_global_rms_standardizer(r_base + pr.cpu().numpy().astype(np.float32), y_mu, y_rms))
            gate_means.append(float(g.mean().item()))

    return preds, {
        "n_params": int(n_params),
        "best_step": int(best_step),
        "gate_mean_unseen": float(gate_means[-1]),
    }


def compute_orthogonal_residual(
    z_base_tr: np.ndarray,
    z_supp_tr: np.ndarray,
    z_base_ev: Sequence[np.ndarray],
    z_supp_ev: Sequence[np.ndarray],
    device: torch.device,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Extract the component of `z_supp` NOT linearly predictable from `z_base` (strict train-only fit!)."""
    preds, _ = fit_predict_scaled_ridge(
        z_base_tr, z_supp_tr, [z_base_tr] + list(z_base_ev), device
    )
    res_tr = (z_supp_tr - preds[0]).astype(np.float32)
    res_ev = [(zs - ps).astype(np.float32) for zs, ps in zip(z_supp_ev, preds[1:])]
    return res_tr, res_ev


def _agg(blocks: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in blocks[0]:
        vals = [float(b[k]) for b in blocks]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ---------------------------------------------------------------------------
# Part 3B: Dual-Regime Representation Audit (Context + Operational Geometry)
# ---------------------------------------------------------------------------

def run_part3b_dual_regime_audit(
    cache: ValClipCache,
    policies: Dict[str, Any],
    device: torch.device,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    """Compare single vs fused representations on Context (Fut_R2, VisHQ_R2, ObjAcc) + Action Geometry."""
    enc = {m: encode_all_val_starts(policies[m], cache, device) for m in EXP3_MODELS}
    a10_val = enc["e9"]["actions_10d"]
    a7_val = enc["e9"]["actions_7d"]
    non_tail = np.flatnonzero(~enc["e9"]["is_tail"])
    unique_tasks = np.unique(cache.val_task_ids[non_tail])

    fut_a10, fut_valid = _gather_future_actions(cache, horizon_offset=16)
    Fut7_flat = action_10d_to_7d(fut_a10).reshape(len(fut_a10), -1)
    A10_flat = a10_val.reshape(len(a10_val), -1)
    A7_flat = a7_val.reshape(len(a7_val), -1)
    HQ_flat = enc["e9"]["h_queries"].reshape(len(a7_val), -1)

    procs = [cache.episodes[int(e)]["procedure"].split("|")[0].strip() for e in cache.val_episodes]
    obj_labels = np.array([p.split()[1] if len(p.split()) >= 2 else "other" for p in procs], dtype=object)
    top_objs = sorted({o for o in obj_labels if (obj_labels == o).sum() >= 100})
    obj_to_id = {o: i for i, o in enumerate(top_objs)}
    obj_y = np.array([obj_to_id.get(o, -1) for o in obj_labels], dtype=np.int64)
    obj_oh = np.zeros((len(obj_y), len(top_objs)), dtype=np.float32)
    for i, oi in enumerate(obj_y):
        if oi >= 0:
            obj_oh[i, oi] = 1.0

    # Define matched-dim (k=64) AND full-dim representation candidates
    seed_rows: List[Dict[str, Dict[str, float]]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 222)
        perm_t = rng.permutation(unique_tasks)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        valid_nt = non_tail[fut_valid[non_tail]]
        v_tasks = cache.val_task_ids[valid_nt]
        tr_i = valid_nt[np.array([int(t) in tr_set for t in v_tasks])]
        te_i = valid_nt[np.array([int(t) not in tr_set for t in v_tasks])]

        var_fut = float(((Fut7_flat[te_i] - Fut7_flat[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        var_hq = float(((HQ_flat[te_i] - HQ_flat[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        var_a7 = float(((A7_flat[te_i] - A7_flat[te_i].mean(axis=0, keepdims=True)) ** 2).mean())

        def _make_pair_concat(m1: str, m2: str, k_each: int) -> Tuple[np.ndarray, np.ndarray]:
            z1_tr, z1_te = enc[m1]["latents_flat"][tr_i], enc[m1]["latents_flat"][te_i]
            z2_tr, z2_te = enc[m2]["latents_flat"][tr_i], enc[m2]["latents_flat"][te_i]
            if k_each > 0:
                p1_tr, [p1_te] = pca_project_block(z1_tr, [z1_te], k_each, device)
                p2_tr, [p2_te] = pca_project_block(z2_tr, [z2_te], k_each, device)
                X_tr, [X_te] = build_block_rms_features([p1_tr, p2_tr], [[p1_te], [p2_te]])
            else:
                X_tr, [X_te] = build_block_rms_features([z1_tr, z2_tr], [[z1_te], [z2_te]])
            return X_tr, X_te

        def _make_single(m1: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
            z1_tr, z1_te = enc[m1]["latents_flat"][tr_i], enc[m1]["latents_flat"][te_i]
            if k > 0:
                p1_tr, [p1_te] = pca_project_block(z1_tr, [z1_te], k, device)
                X_tr, [X_te] = build_block_rms_features([p1_tr], [[p1_te]])
            else:
                X_tr, [X_te] = build_block_rms_features([z1_tr], [[z1_te]])
            return X_tr, X_te

        rep_specs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "e9_k64": _make_single("e9", 64),
            "e12_c_k64": _make_single("e12_c", 64),
            "e12_bc_fd_k64": _make_single("e12_bc_factordisp", 64),
            "concat_e9_e9cont_32p32": _make_pair_concat("e9", "control", 32),
            "concat_e9_e12c_32p32": _make_pair_concat("e9", "e12_c", 32),
            "concat_e9_e12bcfd_32p32": _make_pair_concat("e9", "e12_bc_factordisp", 32),
            "e9_full": _make_single("e9", 0),
            "e12_c_full": _make_single("e12_c", 0),
            "e12_bc_fd_full": _make_single("e12_bc_factordisp", 0),
            "concat_e9_e9cont_full": _make_pair_concat("e9", "control", 0),
            "concat_e9_e12c_full": _make_pair_concat("e9", "e12_c", 0),
            "concat_e9_e12bcfd_full": _make_pair_concat("e9", "e12_bc_factordisp", 0),
        }

        s_res: Dict[str, Dict[str, float]] = {}
        for rname, (R_tr, R_te) in rep_specs.items():
            var_R_te = float(((R_te - R_te.mean(axis=0, keepdims=True)) ** 2).mean())
            R_from_a_te, _ = _ridge_fit_predict(A10_flat[tr_i], R_tr, A10_flat[te_i], device=device)
            r2_a_to_z = float(1.0 - ((R_from_a_te - R_te) ** 2).mean() / max(var_R_te, 1e-12))

            a7_hat, _ = _ridge_fit_predict(R_tr, A7_flat[tr_i], R_te, device=device)
            r2_z_to_a7 = float(1.0 - ((a7_hat - A7_flat[te_i]) ** 2).mean() / max(var_a7, 1e-12))

            fut_hat, _ = _ridge_fit_predict(R_tr, Fut7_flat[tr_i], R_te, device=device)
            r2_fut = float(1.0 - ((fut_hat - Fut7_flat[te_i]) ** 2).mean() / max(var_fut, 1e-12))

            hq_hat, _ = _ridge_fit_predict(R_tr, HQ_flat[tr_i], R_te, device=device)
            r2_hq = float(1.0 - ((hq_hat - HQ_flat[te_i]) ** 2).mean() / max(var_hq, 1e-12))

            obj_tr_m = obj_y[tr_i] >= 0
            obj_te_m = obj_y[te_i] >= 0
            obj_pred, _ = _ridge_fit_predict(R_tr[obj_tr_m], obj_oh[tr_i[obj_tr_m]], R_te[obj_te_m], device=device)
            obj_acc = float((np.argmax(obj_pred, axis=-1) == obj_y[te_i[obj_te_m]]).mean())

            s_res[rname] = {
                "dim": float(R_tr.shape[1]),
                "r2_action_to_z": r2_a_to_z,
                "r2_z_to_cur_action7d": r2_z_to_a7,
                "r2_z_to_future_action7d": r2_fut,
                "r2_z_to_visual_hq": r2_hq,
                "object_linear_acc": obj_acc,
            }
        seed_rows.append(s_res)

    return {rname: _agg([d[rname] for d in seed_rows]) for rname in seed_rows[0]}


# ---------------------------------------------------------------------------
# Part 3A: Visual ICL Fusion + Equal-Dim / Equal-Param Controls + Dual Decoder
# ---------------------------------------------------------------------------

def run_exp3(
    out_dir: Path,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    budgets: Tuple[int, ...] = BUDGETS_DEFAULT,
    match_modes: Tuple[str, ...] = MATCH_MODES_DEFAULT,
    key_budgets: Tuple[int, ...] = (100, 500, 1000, 4000),
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    policies = load_policies(EXP3_MODELS, device)

    print("[E14 Exp3] Running Part 3B dual-regime context & geometry audit...", flush=True)
    part3b = run_part3b_dual_regime_audit(cache, policies, device, seeds=seeds)

    all_mode_res: Dict[str, Any] = {}
    for match_mode in match_modes:
        per_seed_res: List[Dict[str, Any]] = []
        for seed in seeds:
            ts = time.monotonic()
            print(f"[E14 Exp3] Running mode={match_mode} seed={seed}...", flush=True)
            bundle = build_seed_bundle(
                cache, policies, match_mode=match_mode, seed=seed, device=device, n_train_pairs=max(budgets)
            )
            obs = bundle["obs"]
            acts = bundle["acts"]
            lats = bundle["latents"]
            dh_tr, hq_tr = obs["train"]["dh"], obs["train"]["h_q"]
            dh_ev = [obs["test_seen"]["dh"], obs["test_unseen"]["dh"]]
            hq_ev = [obs["test_seen"]["h_q"], obs["test_unseen"]["h_q"]]
            gt_seen, gt_unseen = acts["test_seen"]["q"], acts["test_unseen"]["q"]
            gt_list = [gt_seen, gt_unseen]

            # Direct copy & oracle bounds per decoder
            dc_m: Dict[str, Dict[str, Dict[str, float]]] = {}
            ora_m: Dict[str, Dict[str, Dict[str, float]]] = {}
            for dec_name in EXP3_MODELS:
                ds = ddim_decode_fixed_noise(policies[dec_name], [lats[dec_name]["test_seen"]["d"], lats[dec_name]["test_seen"]["q"]], device, seed=12345)
                du = ddim_decode_fixed_noise(policies[dec_name], [lats[dec_name]["test_unseen"]["d"], lats[dec_name]["test_unseen"]["q"]], device, seed=12345)
                dc_m[dec_name] = {"seen": _action_metrics_dict(ds[0], gt_seen), "unseen": _action_metrics_dict(du[0], gt_unseen)}
                ora_m[dec_name] = {"seen": _action_metrics_dict(ds[1], gt_seen), "unseen": _action_metrics_dict(du[1], gt_unseen)}

            # Queue all latent predictions per target decoder to batch DDIM calls!
            pending: Dict[str, List[List[Tuple[str, np.ndarray, Dict[str, float]]]]] = {
                m: [[], []] for m in EXP3_MODELS
            }

            # We also need validation predictions for internal selection of Dual-Decoder blend weight `w`!
            # Use last 512 of train pairs as internal holdout for `w` at N=4000 (ZERO test leakage!)
            dual_dec_val_preds: Dict[str, np.ndarray] = {}
            dual_dec_test_preds: Dict[str, List[np.ndarray]] = {}
            n_full = max(budgets)
            n_val_w = 512
            n_fit_w = n_full - n_val_w
            gt_val_w = acts["train"]["q"][n_fit_w:n_full]

            # ------------------------------------------------------------------
            # 1. Sample-efficiency curves across ALL 7 budgets (Matched-64 & Full-Dim)
            # ------------------------------------------------------------------
            for n_b in budgets:
                n_eff = min(int(n_b), len(dh_tr))

                # Helper to enqueue a variant targeting `dec_name`
                def _fit_and_queue(
                    tag: str, dec_name: str, X_tr_v: np.ndarray, X_ev_v: List[np.ndarray], dim_info: int
                ) -> List[np.ndarray]:
                    Y_tr = lats[dec_name]["train"]["q"][:n_eff]
                    mu_y, rms_y = fit_global_rms_standardizer(Y_tr)
                    z_true_ev = [lats[dec_name]["test_seen"]["q"], lats[dec_name]["test_unseen"]["q"]]
                    z_demo_ev = [lats[dec_name]["test_seen"]["d"], lats[dec_name]["test_unseen"]["d"]]
                    preds, al = fit_predict_scaled_ridge(X_tr_v, Y_tr, X_ev_v, device)
                    for sp_i in (0, 1):
                        lm = compute_latent_predictability_metrics(
                            preds[sp_i], z_true_ev[sp_i], z_demo_ev[sp_i], mu_y, rms_y
                        )
                        lm["input_dim"] = float(dim_info)
                        lm["alpha"] = float(al)
                        pending[dec_name][sp_i].append((tag, preds[sp_i], lm))
                    return preds

                # Full-dim single & concat curves across ALL 7 budgets (Target = E9, E12-C, E12-BC-FD)
                for m_single in ("e9", "e12_c", "e12_bc_factordisp", "control"):
                    Xtr, Xev = build_block_rms_features(
                        [dh_tr[:n_eff], hq_tr[:n_eff], lats[m_single]["train"]["d"][:n_eff]],
                        [dh_ev, hq_ev, [lats[m_single]["test_seen"]["d"], lats[m_single]["test_unseen"]["d"]]],
                    )
                    p_out = _fit_and_queue(f"full_single_{m_single}|{n_eff}", m_single, Xtr, Xev, Xtr.shape[1])
                    if n_eff == n_full:
                        dual_dec_test_preds[m_single] = p_out

                for partner in ("control", "e12_c", "e12_bc_factordisp"):
                    Xtr, Xev = build_block_rms_features(
                        [dh_tr[:n_eff], hq_tr[:n_eff], lats["e9"]["train"]["d"][:n_eff], lats[partner]["train"]["d"][:n_eff]],
                        [
                            dh_ev, hq_ev,
                            [lats["e9"]["test_seen"]["d"], lats["e9"]["test_unseen"]["d"]],
                            [lats[partner]["test_seen"]["d"], lats[partner]["test_unseen"]["d"]],
                        ],
                    )
                    # Decode with E9 decoder AND with partner decoder
                    _fit_and_queue(f"full_concat_e9+{partner}->e9|{n_eff}", "e9", Xtr, Xev, Xtr.shape[1])
                    if partner in ("e12_c", "e12_bc_factordisp"):
                        _fit_and_queue(f"full_concat_e9+{partner}->{partner}|{n_eff}", partner, Xtr, Xev, Xtr.shape[1])

                # Matched-64 PCA features (`k_obs=128, k_lat=64` -> EXACT D_in = 320 for ALL variants!)
                p_dh_tr, p_dh_ev = pca_project_block(dh_tr[:n_eff], dh_ev, 128, device)
                p_hq_tr, p_hq_ev = pca_project_block(hq_tr[:n_eff], hq_ev, 128, device)

                pca64 = {}
                pca32 = {}
                for m_k in EXP3_MODELS:
                    zd_t = lats[m_k]["train"]["d"][:n_eff]
                    zd_e = [lats[m_k]["test_seen"]["d"], lats[m_k]["test_unseen"]["d"]]
                    p64_tr, p64_ev = pca_project_block(zd_t, zd_e, 64, device)
                    pca64[m_k] = (p64_tr, p64_ev)
                    pca32[m_k] = (p64_tr[:, :32], [e[:, :32] for e in p64_ev])

                # Single k=64 baselines (D_in = 320)
                for m_k in ("e9", "e12_c", "e12_bc_factordisp", "control"):
                    Xtr, Xev = build_block_rms_features(
                        [p_dh_tr, p_hq_tr, pca64[m_k][0]],
                        [p_dh_ev, p_hq_ev, pca64[m_k][1]],
                    )
                    _fit_and_queue(f"m64_single_{m_k}|{n_eff}", m_k, Xtr, Xev, Xtr.shape[1])

                # Equal-dim 32+32=64 Concat (D_in = 320, IDENTICAL param count to Single k=64!)
                for partner in ("control", "e12_c", "e12_bc_factordisp"):
                    # First block-RMS normalize each 32D half so total latent block has equal energy to 64D single!
                    z_cat_tr, z_cat_ev = build_block_rms_features(
                        [pca32["e9"][0], pca32[partner][0]],
                        [pca32["e9"][1], pca32[partner][1]],
                    )
                    Xtr, Xev = build_block_rms_features(
                        [p_dh_tr, p_hq_tr, z_cat_tr],
                        [p_dh_ev, p_hq_ev, z_cat_ev],
                    )
                    _fit_and_queue(f"m64_concat_e9+{partner}->e9|{n_eff}", "e9", Xtr, Xev, Xtr.shape[1])
                    if partner in ("e12_c", "e12_bc_factordisp"):
                        _fit_and_queue(f"m64_concat_e9+{partner}->{partner}|{n_eff}", partner, Xtr, Xev, Xtr.shape[1])

                # Key budgets: Joint-PCA-64, Orthogonal Residual Supplementation, Matched-128, Gated Fusion!
                if n_eff in key_budgets:
                    # 1. Joint-PCA-64 (exact D_in = 320)
                    for partner in ("control", "e12_c", "e12_bc_factordisp"):
                        raw_cat_tr, raw_cat_ev = build_block_rms_features(
                            [lats["e9"]["train"]["d"][:n_eff], lats[partner]["train"]["d"][:n_eff]],
                            [
                                [lats["e9"]["test_seen"]["d"], lats["e9"]["test_unseen"]["d"]],
                                [lats[partner]["test_seen"]["d"], lats[partner]["test_unseen"]["d"]],
                            ],
                        )
                        jp_tr, jp_ev = pca_project_block(raw_cat_tr, raw_cat_ev, 64, device)
                        Xtr, Xev = build_block_rms_features([p_dh_tr, p_hq_tr, jp_tr], [p_dh_ev, p_hq_ev, jp_ev])
                        _fit_and_queue(f"m64_jointpca_e9+{partner}->e9|{n_eff}", "e9", Xtr, Xev, Xtr.shape[1])
                        if partner in ("e12_c", "e12_bc_factordisp"):
                            _fit_and_queue(f"m64_jointpca_e9+{partner}->{partner}|{n_eff}", partner, Xtr, Xev, Xtr.shape[1])

                    # 2. Orthogonal Residual Supplementation (exact matched 32+32=64, D_in = 320!)
                    # (a) E9 + OrthRes(partner | E9) -> E9 decoder
                    for partner in ("control", "e12_c", "e12_bc_factordisp"):
                        res_tr, res_ev = compute_orthogonal_residual(
                            lats["e9"]["train"]["d"][:n_eff],
                            lats[partner]["train"]["d"][:n_eff],
                            [lats["e9"]["test_seen"]["d"], lats["e9"]["test_unseen"]["d"]],
                            [lats[partner]["test_seen"]["d"], lats[partner]["test_unseen"]["d"]],
                            device,
                        )
                        pr_tr, pr_ev = pca_project_block(res_tr, res_ev, 32, device)
                        z_comb_tr, z_comb_ev = build_block_rms_features(
                            [pca32["e9"][0], pr_tr], [pca32["e9"][1], pr_ev]
                        )
                        Xtr, Xev = build_block_rms_features([p_dh_tr, p_hq_tr, z_comb_tr], [p_dh_ev, p_hq_ev, z_comb_ev])
                        _fit_and_queue(f"m64_orthres_e9+({partner}|e9)->e9|{n_eff}", "e9", Xtr, Xev, Xtr.shape[1])

                    # (b) E12 + OrthRes(E9 | E12) -> E12 decoder (does restoring E9 context rescue E12?)
                    for e12_m in ("e12_c", "e12_bc_factordisp"):
                        res_tr, res_ev = compute_orthogonal_residual(
                            lats[e12_m]["train"]["d"][:n_eff],
                            lats["e9"]["train"]["d"][:n_eff],
                            [lats[e12_m]["test_seen"]["d"], lats[e12_m]["test_unseen"]["d"]],
                            [lats["e9"]["test_seen"]["d"], lats["e9"]["test_unseen"]["d"]],
                            device,
                        )
                        pr_tr, pr_ev = pca_project_block(res_tr, res_ev, 32, device)
                        z_comb_tr, z_comb_ev = build_block_rms_features(
                            [pca32[e12_m][0], pr_tr], [pca32[e12_m][1], pr_ev]
                        )
                        Xtr, Xev = build_block_rms_features([p_dh_tr, p_hq_tr, z_comb_tr], [p_dh_ev, p_hq_ev, z_comb_ev])
                        _fit_and_queue(f"m64_orthres_{e12_m}+(e9|{e12_m})->{e12_m}|{n_eff}", e12_m, Xtr, Xev, Xtr.shape[1])

                    # 3. Matched-128 Latent Dim (`D_in = 128+128+128 = 384`): Single(128) vs Concat(64+64=128)
                    pca128 = {}
                    for m_k in ("e9", "e12_c", "e12_bc_factordisp"):
                        zd_t = lats[m_k]["train"]["d"][:n_eff]
                        zd_e = [lats[m_k]["test_seen"]["d"], lats[m_k]["test_unseen"]["d"]]
                        pca128[m_k] = pca_project_block(zd_t, zd_e, 128, device)
                        Xtr, Xev = build_block_rms_features(
                            [p_dh_tr, p_hq_tr, pca128[m_k][0]], [p_dh_ev, p_hq_ev, pca128[m_k][1]]
                        )
                        _fit_and_queue(f"m128_single_{m_k}|{n_eff}", m_k, Xtr, Xev, Xtr.shape[1])

                    for partner in ("control", "e12_c", "e12_bc_factordisp"):
                        z_cat_tr, z_cat_ev = build_block_rms_features(
                            [pca64["e9"][0], pca64[partner][0]], [pca64["e9"][1], pca64[partner][1]]
                        )
                        Xtr, Xev = build_block_rms_features([p_dh_tr, p_hq_tr, z_cat_tr], [p_dh_ev, p_hq_ev, z_cat_ev])
                        _fit_and_queue(f"m128_concat_e9+{partner}->e9|{n_eff}", "e9", Xtr, Xev, Xtr.shape[1])
                        if partner in ("e12_c", "e12_bc_factordisp"):
                            _fit_and_queue(f"m128_concat_e9+{partner}->{partner}|{n_eff}", partner, Xtr, Xev, Xtr.shape[1])

                    # 4. Exact Parameter-Matched Lightweight Gated Fusion at N in {1000, 4000}
                    if n_eff in (1000, 4000):
                        x_obs_tr, x_obs_ev = build_block_rms_features([p_dh_tr, p_hq_tr], [p_dh_ev, p_hq_ev])
                        gated_specs = [
                            # Single-model split controls (exact same parameter count!)
                            ("gated_single_e9_split->e9", "e9", pca128["e9"][0][:, :64], pca128["e9"][0][:, 64:128], [e[:, :64] for e in pca128["e9"][1]], [e[:, 64:128] for e in pca128["e9"][1]]),
                            ("gated_single_e12c_split->e12_c", "e12_c", pca128["e12_c"][0][:, :64], pca128["e12_c"][0][:, 64:128], [e[:, :64] for e in pca128["e12_c"][1]], [e[:, 64:128] for e in pca128["e12_c"][1]]),
                            # Same-family 2-checkpoint control
                            ("gated_e9+e9cont->e9", "e9", pca64["e9"][0], pca64["control"][0], pca64["e9"][1], pca64["control"][1]),
                            # Complementary fusion
                            ("gated_e9+e12c->e9", "e9", pca64["e9"][0], pca64["e12_c"][0], pca64["e9"][1], pca64["e12_c"][1]),
                            ("gated_e9+e12bcfd->e9", "e9", pca64["e9"][0], pca64["e12_bc_factordisp"][0], pca64["e9"][1], pca64["e12_bc_factordisp"][1]),
                            ("gated_e9+e12c->e12_c", "e12_c", pca64["e9"][0], pca64["e12_c"][0], pca64["e9"][1], pca64["e12_c"][1]),
                            ("gated_e9+e12bcfd->e12_bc_fd", "e12_bc_factordisp", pca64["e9"][0], pca64["e12_bc_factordisp"][0], pca64["e9"][1], pca64["e12_bc_factordisp"][1]),
                        ]
                        param_counts = set()
                        for gname, dec_n, za_t, zb_t, za_e, zb_e in gated_specs:
                            Y_tr = lats[dec_n]["train"]["q"][:n_eff]
                            mu_y, rms_y = fit_global_rms_standardizer(Y_tr)
                            z_true_ev = [lats[dec_n]["test_seen"]["q"], lats[dec_n]["test_unseen"]["q"]]
                            z_demo_ev = [lats[dec_n]["test_seen"]["d"], lats[dec_n]["test_unseen"]["d"]]
                            g_preds, g_info = fit_predict_gated_fusion(
                                x_obs_tr, za_t, zb_t, Y_tr, x_obs_ev, za_e, zb_e, device, seed=seed + n_eff
                            )
                            param_counts.add(g_info["n_params"])
                            for sp_i in (0, 1):
                                lm = compute_latent_predictability_metrics(
                                    g_preds[sp_i], z_true_ev[sp_i], z_demo_ev[sp_i], mu_y, rms_y
                                )
                                lm["n_params"] = float(g_info["n_params"])
                                lm["gate_mean_unseen"] = float(g_info["gate_mean_unseen"])
                                pending[dec_n][sp_i].append((f"{gname}|{n_eff}", g_preds[sp_i], lm))
                        assert len(param_counts) == 1, f"Gated fusion parameter mismatch: {param_counts}"

            # Internal train-holdout predictions for selecting Dual-Decoder convex weight `w` (ZERO test leakage!)
            for m_k in EXP3_MODELS:
                X_fit, [X_val] = build_block_rms_features(
                    [dh_tr[:n_fit_w], hq_tr[:n_fit_w], lats[m_k]["train"]["d"][:n_fit_w]],
                    [[dh_tr[n_fit_w:n_full]], [hq_tr[n_fit_w:n_full]], [lats[m_k]["train"]["d"][n_fit_w:n_full]]],
                )
                [z_val_pred], _ = fit_predict_scaled_ridge(
                    X_fit, lats[m_k]["train"]["q"][:n_fit_w], [X_val], device
                )
                dual_dec_val_preds[m_k] = ddim_decode_fixed_noise(policies[m_k], [z_val_pred], device, seed=12345)[0]

            # ------------------------------------------------------------------
            # Batched Fixed-Noise DDIM Decoding for ALL variants!
            # ------------------------------------------------------------------
            seed_variants: Dict[str, Any] = {}
            decoded_single_full: Dict[str, List[np.ndarray]] = {}
            sp_names = ("seen", "unseen")
            for dec_n in EXP3_MODELS:
                for sp_i, sp_name in enumerate(sp_names):
                    items = pending[dec_n][sp_i]
                    if not items:
                        continue
                    decoded = ddim_decode_fixed_noise(
                        policies[dec_n], [it[1] for it in items], device, seed=12345, batch_size=1024
                    )
                    for (tag, _, lm), pred_a in zip(items, decoded):
                        vname, n_s = tag.rsplit("|", 1)
                        m_act = _action_metrics_dict(pred_a, gt_list[sp_i])
                        packed = pack_action_metrics(m_act, dc_m[dec_n][sp_name], ora_m[dec_n][sp_name])
                        packed.update(lm)
                        seed_variants.setdefault(vname, {}).setdefault(n_s, {})[sp_name] = packed

                        if vname == f"full_single_{dec_n}" and int(n_s) == n_full:
                            decoded_single_full.setdefault(dec_n, [None, None])[sp_i] = pred_a

            # ------------------------------------------------------------------
            # Dual-Decoder Action-Space Fusion & Per-Sample Complementarity Audit
            # ------------------------------------------------------------------
            dual_dec_res: Dict[str, Any] = {}
            comp_audit: Dict[str, Any] = {}
            for partner in ("control", "e12_c", "e12_bc_factordisp"):
                # Select optimal convex weight w in [0, 1] STRICTLY on internal train holdout `gt_val_w`!
                a_e9_v = dual_dec_val_preds["e9"]
                a_p_v = dual_dec_val_preds[partner]
                w_grid = np.linspace(0.0, 1.0, 21)
                best_w = 0.5
                best_v_mse = float("inf")
                for w in w_grid:
                    blend_v = w * a_e9_v + (1.0 - w) * a_p_v
                    mse_v = float(np.mean((blend_v - gt_val_w) ** 2))
                    if mse_v < best_v_mse:
                        best_v_mse = mse_v
                        best_w = float(w)

                pair_res: Dict[str, Any] = {"selected_w_e9": best_w}
                for sp_i, sp_name in enumerate(sp_names):
                    a_e9 = decoded_single_full["e9"][sp_i]
                    a_p = decoded_single_full[partner][sp_i]
                    gt = gt_list[sp_i]

                    # Weighted blend (val-tuned w) & 50/50 blend
                    m_blend = _action_metrics_dict(best_w * a_e9 + (1.0 - best_w) * a_p, gt)
                    m_half = _action_metrics_dict(0.5 * a_e9 + 0.5 * a_p, gt)
                    pair_res[f"tuned_blend_{sp_name}"] = pack_action_metrics(m_blend, dc_m["e9"][sp_name], ora_m["e9"][sp_name])
                    pair_res[f"equal_blend_{sp_name}"] = pack_action_metrics(m_half, dc_m["e9"][sp_name], ora_m["e9"][sp_name])

                    if sp_name == "unseen":
                        err_e9 = ((a_e9 - gt) ** 2).mean(axis=(1, 2))
                        err_p = ((a_p - gt) ** 2).mean(axis=(1, 2))
                        oracle_best = np.minimum(err_e9, err_p)
                        corr = float(np.corrcoef(err_e9, err_p)[0, 1])
                        win_rate_p = float((err_p < err_e9).mean())

                        # Stratify win rate by visual shift ||dh||, retrieval cos, action change ||da||, grip transition
                        dh_norm = np.linalg.norm(obs["test_unseen"]["dh"], axis=-1)
                        da_mse = ((acts["test_unseen"]["q"] - acts["test_unseen"]["d"]) ** 2).mean(axis=(1, 2))
                        hd_u, hq_u = obs["test_unseen"]["h_d"], obs["test_unseen"]["h_q"]
                        cos_h = (hd_u * hq_u).sum(axis=-1) / (
                            np.clip(np.linalg.norm(hd_u, axis=-1), 1e-6, None)
                            * np.clip(np.linalg.norm(hq_u, axis=-1), 1e-6, None)
                        )
                        q_grip = acts["test_unseen"]["q"][:, :, 9]
                        g_trans = np.sign(q_grip[:, 0]) != np.sign(q_grip[:, -1])

                        dh_med = float(np.median(dh_norm))
                        da_med = float(np.median(da_mse))
                        cos_med = float(np.median(cos_h))

                        comp_audit[f"e9_vs_{partner}"] = {
                            "e9_mse10": float(err_e9.mean()),
                            "partner_mse10": float(err_p.mean()),
                            "tuned_blend_mse10": float(m_blend["action10_mse"]),
                            "oracle_best_of_two_mse10": float(oracle_best.mean()),
                            "oracle_gain_over_e9": float(err_e9.mean() - oracle_best.mean()),
                            "error_pearson_corr": corr,
                            "partner_win_rate": win_rate_p,
                            "partner_win_rate_high_dh": float((err_p[dh_norm >= dh_med] < err_e9[dh_norm >= dh_med]).mean()),
                            "partner_win_rate_low_dh": float((err_p[dh_norm < dh_med] < err_e9[dh_norm < dh_med]).mean()),
                            "partner_win_rate_high_da": float((err_p[da_mse >= da_med] < err_e9[da_mse >= da_med]).mean()),
                            "partner_win_rate_low_da": float((err_p[da_mse < da_med] < err_e9[da_mse < da_med]).mean()),
                            "partner_win_rate_low_cos": float((err_p[cos_h < cos_med] < err_e9[cos_h < cos_med]).mean()),
                            "partner_win_rate_high_cos": float((err_p[cos_h >= cos_med] < err_e9[cos_h >= cos_med]).mean()),
                            "partner_win_rate_grip_trans": float((err_p[g_trans] < err_e9[g_trans]).mean()) if g_trans.any() else 0.0,
                            "partner_win_rate_grip_steady": float((err_p[~g_trans] < err_e9[~g_trans]).mean()) if (~g_trans).any() else 0.0,
                            "e9_minus_partner_high_da": float((err_e9[da_mse >= da_med] - err_p[da_mse >= da_med]).mean()),
                            "e9_minus_partner_low_da": float((err_e9[da_mse < da_med] - err_p[da_mse < da_med]).mean()),
                            "e9_minus_partner_grip_trans": float((err_e9[g_trans] - err_p[g_trans]).mean()) if g_trans.any() else 0.0,
                        }
                dual_dec_res[f"e9+{partner}"] = pair_res

            per_seed_res.append(
                {"variants": seed_variants, "dual_decoder": dual_dec_res, "complementarity_audit": comp_audit}
            )
            print(f"[E14 Exp3] Finished mode={match_mode} seed={seed} in {time.monotonic() - ts:.1f}s", flush=True)

        # Aggregate across seeds
        agg_variants: Dict[str, Any] = {}
        for vname in per_seed_res[0]["variants"]:
            agg_variants[vname] = {}
            for n_s in per_seed_res[0]["variants"][vname]:
                agg_variants[vname][n_s] = {
                    sp: _agg([d["variants"][vname][n_s][sp] for d in per_seed_res])
                    for sp in ("seen", "unseen")
                }

        agg_dual: Dict[str, Any] = {}
        for pname in per_seed_res[0]["dual_decoder"]:
            agg_dual[pname] = {
                "selected_w_e9_mean": float(np.mean([d["dual_decoder"][pname]["selected_w_e9"] for d in per_seed_res])),
                "selected_w_e9_std": float(np.std([d["dual_decoder"][pname]["selected_w_e9"] for d in per_seed_res])),
            }
            for tag in ("tuned_blend_seen", "tuned_blend_unseen", "equal_blend_seen", "equal_blend_unseen"):
                agg_dual[pname][tag] = _agg([d["dual_decoder"][pname][tag] for d in per_seed_res])

        agg_audit: Dict[str, Any] = {}
        for pname in per_seed_res[0]["complementarity_audit"]:
            agg_audit[pname] = _agg([d["complementarity_audit"][pname] for d in per_seed_res])

        all_mode_res[match_mode] = {
            "variants": agg_variants,
            "dual_decoder": agg_dual,
            "complementarity_audit": agg_audit,
        }

        # Print key Exp 3 highlights
        print(f"\n=== [E14 Exp3 | mode={match_mode}] Complementary Representation Highlights (Unseen, N=4000, 3 seeds) ===", flush=True)
        for vname in sorted(agg_variants.keys()):
            if "4000" in agg_variants[vname]:
                u = agg_variants[vname]["4000"]["unseen"]
                print(
                    f"  {vname:42s} | MSE10={u['mse10_mean']:.5f}±{u['mse10_std']:.5f} "
                    f"ARR={u['arr_10d_mean']*100:5.1f}% R2={u['r2_z_mean']:.3f}",
                    flush=True,
                )
        print("  [Dual-Decoder Action-Space Fusion & Complementarity Audit]:", flush=True)
        for pname, dv in agg_dual.items():
            bu = dv["tuned_blend_unseen"]
            au = agg_audit[f"e9_vs_{pname.split('+')[1]}"]
            print(
                f"    {pname:24s} (w_e9={dv['selected_w_e9_mean']:.2f}) | "
                f"Blend={bu['mse10_mean']:.5f} (ARR={bu['arr_10d_mean']*100:.1f}%) | "
                f"Corr={au['error_pearson_corr_mean']:.3f} WinRate={au['partner_win_rate_mean']*100:.1f}% "
                f"OracleBest2={au['oracle_best_of_two_mse10_mean']:.5f}",
                flush=True,
            )

    summary = {
        "models": list(EXP3_MODELS),
        "seeds": list(seeds),
        "budgets": list(budgets),
        "key_budgets": list(key_budgets),
        "match_modes": list(match_modes),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "part_3b_dual_regime_audit": part3b,
        "results": all_mode_res,
    }
    atomic_write_json(summary, out_dir / "exp3_summary.json")
    print(f"[E14 Exp3] Saved summary to {out_dir / 'exp3_summary.json'} ({summary['elapsed_seconds']}s)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp3_complementary_repr")
    args = p.parse_args()
    run_exp3(args.out_dir)


if __name__ == "__main__":
    main()
