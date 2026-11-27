"""E13 Experiment 2: Action Space & Action Autoencoder Baselines.

Investigates whether E12-C is merely a linear copy of the 16-step action space or a
visually grounded behavior representation that preserves rich non-action context:
  - Builds two Action Space baselines:
      1. Raw Action (7D delta action trajectory `a_{t:t+16}`)
      2. Action Autoencoder (MLP AE `a_10d -> z_act (4x768, LayerNorm) -> a_10d`,
         trained strictly on E9's `train` fold with zero visual input)
  - Explicitly separates:
      * Category I (Oracle / Action-Guided Editing): Tier-1/2 Analogy & Δa-Transfer
      * Category II (Non-Action Behavioral Context & Predictive Information):
          - Variance of z linearly explained by current action `R^2(a_7d -> z)` vs
            action-orthogonal residual `z_perp` energy
          - Future-Action Predictability `z_t -> a_{t+16:t+32}` (and from `z_perp` alone!)
            compared against Raw Action momentum baseline and Action-AE
          - Visual Observation (`h_q`) & Task/Procedure Context Predictability from `z` & `z_perp`
          - Decoder Midpoint Interpolation Fidelity (on-manifold smoothness test)
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
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, action_7d_to_10d
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import ACTION_DIM_10D, ACTION_HORIZON, E5_CACHE_DIR
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _effective_rank_stats,
    _ridge_fit_predict,
    ddim_decode_batched,
    encode_all_val_starts,
)

from . import CHECKPOINTS, OUTPUT_ROOT, SEEDS_DEFAULT
from .exp1_training_free import _build_fixed_analogy_quadruples


class ActionAutoencoder(nn.Module):
    """Action-only MLP Autoencoder mapping `a_10d (B, 16, 10)` -> `z_act (B, 4, 768)` -> `a_10d`."""

    def __init__(self, hidden_dim: int = 1024, latent_tokens: int = 4, embed_dim: int = 768):
        super().__init__()
        self.latent_tokens = latent_tokens
        self.embed_dim = embed_dim
        in_dim = ACTION_HORIZON * ACTION_DIM_10D
        lat_dim = latent_tokens * embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lat_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, a10: torch.Tensor) -> torch.Tensor:
        B = a10.shape[0]
        h = self.encoder(a10.reshape(B, -1)).reshape(B, self.latent_tokens, self.embed_dim)
        return self.out_norm(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return self.decoder(z.reshape(B, -1)).reshape(B, ACTION_HORIZON, ACTION_DIM_10D)

    def forward(self, a10: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(a10)
        return self.decode(z), z


def train_or_load_action_ae(
    cache: ValClipCache,
    out_dir: Path,
    device: torch.device,
    *,
    steps: int = 4000,
    batch_size: int = 512,
    seed: int = 42,
) -> ActionAutoencoder:
    ckpt_path = out_dir / "action_ae.pt"
    model = ActionAutoencoder().to(device)
    if ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    seed_everything(seed)
    # Sample train starts directly from E5 cache (fold=train ONLY!)
    ep_off = cache.query_ep_offsets_np
    f_nums = cache.query_frame_nums_np
    tr_eps, tr_frs = [], []
    for ep_i, row in enumerate(cache.episodes):
        if row["fold"] != "train":
            continue
        lo, hi = int(ep_off[ep_i]), int(ep_off[ep_i + 1])
        ef = f_nums[lo:hi]
        if len(ef) == 0:
            continue
        tr_eps.append(np.full(len(ef), ep_i, dtype=np.int64))
        tr_frs.append(ef.astype(np.int64))
    tr_eps_arr = np.concatenate(tr_eps)
    tr_frs_arr = np.concatenate(tr_frs)

    # Load actions_10d into GPU (only 145 MB!)
    acts_all = torch.from_numpy(
        np.load(E5_CACHE_DIR / "actions_10d.npy", allow_pickle=False)
    ).to(device)
    f_off = torch.from_numpy(cache.frame_offsets_np).to(device)
    e_len = torch.from_numpy(cache.ep_lengths_np.astype(np.int64)).to(device)
    tr_eps_t = torch.from_numpy(tr_eps_arr).to(device)
    tr_frs_t = torch.from_numpy(tr_frs_arr).to(device)
    win = torch.arange(ACTION_HORIZON, device=device, dtype=torch.long)[None, :]

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    n_tr = len(tr_eps_t)

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_tr, (batch_size,), generator=gen, device=device)
        ep = tr_eps_t[idx]
        fr = tr_frs_t[idx]
        clamped = torch.minimum(fr[:, None] + win, (e_len[ep] - 1).clamp_min(0)[:, None])
        a10 = acts_all[f_off[ep][:, None] + clamped].float()
        opt.zero_grad(set_to_none=True)
        pred, _ = model(a10)
        loss = F.mse_loss(pred, a10)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    atomic_torch_save(model.state_dict(), ckpt_path)
    return model


def _gather_future_actions(cache: ValClipCache, horizon_offset: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Gather future 16-step action chunk `a_{t+16:t+32}` for each val start, plus valid mask."""
    N = len(cache.val_episodes)
    fut = np.zeros((N, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        ep = int(cache.val_episodes[i])
        fr = int(cache.val_frames[i])
        T = int(cache.ep_lengths_np[ep])
        off = int(cache.frame_offsets_np[ep])
        if fr + horizon_offset + ACTION_HORIZON <= T:
            fut[i] = np.asarray(
                cache.actions_10d_mmap[off + fr + horizon_offset : off + fr + horizon_offset + ACTION_HORIZON],
                dtype=np.float32,
            )
            valid[i] = True
    return fut, valid


def run_exp2(out_dir: Path, seeds: Tuple[int, ...] = SEEDS_DEFAULT) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    ae = train_or_load_action_ae(cache, out_dir, device)

    # Encode all models + Action-AE on val split
    models_to_eval = ["e9", "e12_a", "e12_b", "e12_c"]
    policies = {k: load_frozen_e9_stage1_teacher(CHECKPOINTS[k], device=device) for k in models_to_eval}
    encoded = {k: encode_all_val_starts(policies[k], cache, device) for k in models_to_eval}

    # Action-AE latents & self-reconstruction on val
    a10_val_np = encoded["e9"]["actions_10d"]
    a7_val_np = encoded["e9"]["actions_7d"]
    with torch.inference_mode():
        a10_t = torch.from_numpy(a10_val_np).to(device)
        z_ae_4 = ae.encode(a10_t).cpu().numpy()
        pred_ae_self = ae.decode(ae.encode(a10_t)).cpu().numpy()
    ae_self_m = _action_metrics_dict(pred_ae_self, a10_val_np)
    z_ae_flat = z_ae_4.reshape(len(a10_val_np), -1)

    non_tail = np.flatnonzero(~encoded["e9"]["is_tail"])
    tasks_nt = cache.val_task_ids[non_tail]
    unique_tasks = np.unique(tasks_nt)

    # ------------------------------------------------------------------
    # Test 2A: Category I (Oracle Action-Guided Editing) — Tier 1 & Tier 2 Arithmetic
    # ------------------------------------------------------------------
    quads = _build_fixed_analogy_quadruples(cache, encoded["e9"], device, seed=42)
    test_2a: Dict[str, Any] = {"action_ae_self_recon": ae_self_m, "tiers": {}}
    for tier_name, (A1, A2, B1, B2) in quads.items():
        gt_B2 = a10_val_np[B2]
        # 1. Raw Action 7D arithmetic (exact action-space ceiling!)
        a7_arith = a7_val_np[B1] + (a7_val_np[A2] - a7_val_np[A1])
        a7_arith[..., 6] = np.clip(a7_arith[..., 6], -1.0, 1.0)
        a10_raw_arith = action_7d_to_10d(a7_arith)
        m_raw = _action_metrics_dict(a10_raw_arith, gt_B2)
        m_raw_base = _action_metrics_dict(a10_val_np[B1], gt_B2)
        arr_raw = (m_raw_base["action10_mse"] - m_raw["action10_mse"]) / max(m_raw_base["action10_mse"], 1e-9)

        # 2. Action-AE latent arithmetic
        z_ae_arith = z_ae_flat[B1] + (z_ae_flat[A2] - z_ae_flat[A1])
        with torch.inference_mode():
            pred_ae_ar = ae.decode(torch.from_numpy(z_ae_arith).to(device)).cpu().numpy()
            pred_ae_b1 = ae.decode(torch.from_numpy(z_ae_flat[B1]).to(device)).cpu().numpy()
            pred_ae_b2 = ae.decode(torch.from_numpy(z_ae_flat[B2]).to(device)).cpu().numpy()
        m_ae = _action_metrics_dict(pred_ae_ar, gt_B2)
        m_ae_b1 = _action_metrics_dict(pred_ae_b1, gt_B2)
        m_ae_b2 = _action_metrics_dict(pred_ae_b2, gt_B2)
        arr_ae = (m_ae_b1["action10_mse"] - m_ae["action10_mse"]) / max(
            m_ae_b1["action10_mse"] - m_ae_b2["action10_mse"], 1e-9
        )

        tier_out = {
            "raw_action_7d_ceiling": {
                "mse10": m_raw["action10_mse"],
                "mae7": m_raw["action7_mae"],
                "grip_acc": m_raw["grip_sign_acc"],
                "arr_10d": float(arr_raw),
            },
            "action_ae_privileged": {
                "mse10": m_ae["action10_mse"],
                "mae7": m_ae["action7_mae"],
                "grip_acc": m_ae["grip_sign_acc"],
                "arr_10d": float(arr_ae),
            },
        }
        for k in models_to_eval:
            zf = encoded[k]["latents_flat"]
            z_ar = zf[B1] + (zf[A2] - zf[A1])
            mb = _action_metrics_dict(ddim_decode_batched(policies[k], zf[B1], device), gt_B2)
            mo = _action_metrics_dict(ddim_decode_batched(policies[k], zf[B2], device), gt_B2)
            ma = _action_metrics_dict(ddim_decode_batched(policies[k], z_ar, device), gt_B2)
            tier_out[k] = {
                "mse10": ma["action10_mse"],
                "mae7": ma["action7_mae"],
                "grip_acc": ma["grip_sign_acc"],
                "arr_10d": float(
                    (mb["action10_mse"] - ma["action10_mse"])
                    / max(mb["action10_mse"] - mo["action10_mse"], 1e-9)
                ),
            }
        test_2a["tiers"][tier_name] = tier_out

    # ------------------------------------------------------------------
    # Test 2B, 2C, 2D: Non-Action Residual Energy, Future-Action Predictability
    # (`z_t -> a_{t+16:t+32}`), and Visual-Context (`h_q`) Predictability (3 seeds, disjoint tasks!)
    # ------------------------------------------------------------------
    fut_a10, fut_valid = _gather_future_actions(cache, horizon_offset=16)
    fut_a7 = action_10d_to_7d(fut_a10)
    # Build procedure class labels for semantic task/object probing
    procs = [cache.episodes[int(e)]["procedure"].split("|")[0].strip() for e in cache.val_episodes]
    # Extract primary manipulated object from procedure string ("Grasp <obj> ...")
    def _parse_first_obj(p: str) -> str:
        toks = p.split()
        return toks[1] if len(toks) >= 2 else "other"

    obj_labels = np.array([_parse_first_obj(p) for p in procs], dtype=object)
    top_objs = sorted({o for o in obj_labels if (obj_labels == o).sum() >= 100})
    obj_to_id = {o: i for i, o in enumerate(top_objs)}
    obj_y = np.array([obj_to_id.get(o, -1) for o in obj_labels], dtype=np.int64)
    obj_oh = np.zeros((len(obj_y), len(top_objs)), dtype=np.float32)
    for i, oi in enumerate(obj_y):
        if oi >= 0:
            obj_oh[i, oi] = 1.0

    rep_bank = {
        "raw_action_7d": a7_val_np.reshape(len(a7_val_np), -1),
        "action_ae": z_ae_flat,
        "e9": encoded["e9"]["latents_flat"],
        "e12_a": encoded["e12_a"]["latents_flat"],
        "e12_b": encoded["e12_b"]["latents_flat"],
        "e12_c": encoded["e12_c"]["latents_flat"],
    }
    A7_flat = a7_val_np.reshape(len(a7_val_np), -1)
    A10_flat = a10_val_np.reshape(len(a10_val_np), -1)
    Fut7_flat = fut_a7.reshape(len(fut_a7), -1)
    HQ_flat = encoded["e9"]["h_queries"].reshape(len(a7_val_np), -1)

    non_action_seeds = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 222)
        perm_t = rng.permutation(unique_tasks)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        # Restrict to clips that have valid future chunk `a_{t+16:t+32}` and non-tail
        valid_nt = non_tail[fut_valid[non_tail]]
        v_tasks = cache.val_task_ids[valid_nt]
        tr_i = valid_nt[np.array([int(t) in tr_set for t in v_tasks])]
        te_i = valid_nt[np.array([int(t) not in tr_set for t in v_tasks])]

        var_fut = float(((Fut7_flat[te_i] - Fut7_flat[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        var_hq = float(((HQ_flat[te_i] - HQ_flat[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
        var_a7 = float(((A7_flat[te_i] - A7_flat[te_i].mean(axis=0, keepdims=True)) ** 2).mean())

        s_res = {}
        for rname, R in rep_bank.items():
            R_tr, R_te = R[tr_i], R[te_i]
            var_R_te = float(((R_te - R_te.mean(axis=0, keepdims=True)) ** 2).mean())

            # 1. How much of R is linearly explained by current action a_10d?
            R_from_a_tr, al_az = _ridge_fit_predict(A10_flat[tr_i], R_tr, A10_flat[tr_i])
            R_from_a_te, _ = _ridge_fit_predict(A10_flat[tr_i], R_tr, A10_flat[te_i], alphas=(al_az,))
            r2_a_to_z = float(1.0 - ((R_from_a_te - R_te) ** 2).mean() / max(var_R_te, 1e-12))
            z_perp_tr = R_tr - R_from_a_tr
            z_perp_te = R_te - R_from_a_te
            non_action_var_frac = float(max(0.0, 1.0 - r2_a_to_z))

            # 2. Current action linear probe R^2(z -> a_7d)
            a7_hat, _ = _ridge_fit_predict(R_tr, A7_flat[tr_i], R_te)
            r2_z_to_a7 = float(1.0 - ((a7_hat - A7_flat[te_i]) ** 2).mean() / max(var_a7, 1e-12))

            # 3. Future-action predictability: R -> a_{t+16:t+32} AND z_perp -> a_{t+16:t+32}
            fut_hat, _ = _ridge_fit_predict(R_tr, Fut7_flat[tr_i], R_te)
            r2_fut = float(1.0 - ((fut_hat - Fut7_flat[te_i]) ** 2).mean() / max(var_fut, 1e-12))
            fut_mse = float(((fut_hat - Fut7_flat[te_i]) ** 2).mean())

            fut_perp_hat, _ = _ridge_fit_predict(z_perp_tr, Fut7_flat[tr_i], z_perp_te)
            r2_fut_perp = float(1.0 - ((fut_perp_hat - Fut7_flat[te_i]) ** 2).mean() / max(var_fut, 1e-12))

            # 4. Visual context predictability: R -> h_q AND z_perp -> h_q
            hq_hat, _ = _ridge_fit_predict(R_tr, HQ_flat[tr_i], R_te)
            r2_hq = float(1.0 - ((hq_hat - HQ_flat[te_i]) ** 2).mean() / max(var_hq, 1e-12))

            hq_perp_hat, _ = _ridge_fit_predict(z_perp_tr, HQ_flat[tr_i], z_perp_te)
            r2_hq_perp = float(1.0 - ((hq_perp_hat - HQ_flat[te_i]) ** 2).mean() / max(var_hq, 1e-12))

            # 5. Primary object classification accuracy (linear probe on valid object subset)
            obj_tr_m = obj_y[tr_i] >= 0
            obj_te_m = obj_y[te_i] >= 0
            obj_pred, _ = _ridge_fit_predict(R_tr[obj_tr_m], obj_oh[tr_i[obj_tr_m]], R_te[obj_te_m])
            obj_acc = float((np.argmax(obj_pred, axis=-1) == obj_y[te_i[obj_te_m]]).mean())

            s_res[rname] = {
                "r2_action_to_z_linearity": r2_a_to_z,
                "non_action_var_frac": non_action_var_frac,
                "r2_z_to_cur_action7d": r2_z_to_a7,
                "r2_z_to_future_action7d": r2_fut,
                "future_action7d_mse": fut_mse,
                "r2_z_perp_to_future_action7d": r2_fut_perp,
                "r2_z_to_visual_hq": r2_hq,
                "r2_z_perp_to_visual_hq": r2_hq_perp,
                "object_linear_acc": obj_acc,
            }
        non_action_seeds.append(s_res)

    # ------------------------------------------------------------------
    # Test 2E: Midpoint Interpolation Smoothness (does 0.5*z_A + 0.5*z_B decode to 0.5*a_A + 0.5*a_B?)
    # ------------------------------------------------------------------
    rng_m = np.random.default_rng(42 + 999)
    mid_idx = rng_m.choice(non_tail, size=(256, 2), replace=False)
    ia_m, ib_m = mid_idx[:, 0], mid_idx[:, 1]
    a7_mid_gt = 0.5 * (a7_val_np[ia_m] + a7_val_np[ib_m])
    a10_mid_gt = action_7d_to_10d(a7_mid_gt)

    interp_res = {}
    for k in models_to_eval:
        z_mid = 0.5 * (encoded[k]["latents_flat"][ia_m] + encoded[k]["latents_flat"][ib_m])
        pred_mid = ddim_decode_batched(policies[k], z_mid, device)
        m_mid = _action_metrics_dict(pred_mid, a10_mid_gt)
        interp_res[k] = {
            "midpoint_10d_mse": m_mid["action10_mse"],
            "midpoint_pos_mse": m_mid["pos_mse"],
            "midpoint_7d_mae": m_mid["action7_mae"],
        }
    with torch.inference_mode():
        z_ae_mid = 0.5 * (z_ae_flat[ia_m] + z_ae_flat[ib_m])
        pred_ae_mid = ae.decode(torch.from_numpy(z_ae_mid).to(device)).cpu().numpy()
    m_ae_mid = _action_metrics_dict(pred_ae_mid, a10_mid_gt)
    interp_res["action_ae"] = {
        "midpoint_10d_mse": m_ae_mid["action10_mse"],
        "midpoint_pos_mse": m_ae_mid["pos_mse"],
        "midpoint_7d_mae": m_ae_mid["action7_mae"],
    }

    # Aggregate multi-seed non_action_seeds
    non_action_agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rname in non_action_seeds[0]:
        non_action_agg[rname] = {}
        for metric in non_action_seeds[0][rname]:
            vals = [d[rname][metric] for d in non_action_seeds]
            non_action_agg[rname][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "seeds": list(seeds),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "test_2a_oracle_editing": test_2a,
        "test_2bcd_representation_vs_action_copy": non_action_agg,
        "test_2e_midpoint_interpolation": interp_res,
    }
    atomic_write_json(summary, out_dir / "exp2_summary.json")

    print("\n=== [E13 Exp 2] Action Space / Action-AE vs Behavior Representations ===", flush=True)
    print(f"Action-AE self recon 10D MSE = {ae_self_m['action10_mse']:.6f}")
    print("Test 2A (Tier-2 Parallel Arithmetic — Oracle-Selected Pairs):")
    for k, v in test_2a["tiers"]["tier2"].items():
        print(f"  {k:24s}: mse10={v['mse10']:.5f} ARR={v['arr_10d']*100:.1f}% grip={v['grip_acc']*100:.1f}%")
    print("Test 2B/C/D (Action-Linearity, Future-Action R2, Visual Context R2, Object Acc — 3 seeds):")
    for k, v in non_action_agg.items():
        print(
            f"  {k:16s}: R2(a->z)={v['r2_action_to_z_linearity']['mean']:.3f} "
            f"(non_act={v['non_action_var_frac']['mean']*100:.1f}%) | "
            f"Fut_R2={v['r2_z_to_future_action7d']['mean']:.3f} "
            f"(perp={v['r2_z_perp_to_future_action7d']['mean']:.3f}) | "
            f"VisHQ_R2={v['r2_z_to_visual_hq']['mean']:.3f} "
            f"(perp={v['r2_z_perp_to_visual_hq']['mean']:.3f}) | "
            f"ObjAcc={v['object_linear_acc']['mean']*100:.1f}%"
        )
    print("Test 2E (Midpoint Interpolation MSE10 / Pos MSE):")
    for k, v in interp_res.items():
        print(f"  {k:14s}: mid_mse10={v['midpoint_10d_mse']:.5f} pos_mse={v['midpoint_pos_mse']:.5f} mae7={v['midpoint_7d_mae']:.5f}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp2_action_baselines")
    args = p.parse_args()
    run_exp2(args.out_dir)


if __name__ == "__main__":
    main()
