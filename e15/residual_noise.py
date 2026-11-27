"""Strict train-only K-fold task-grouped CV Adapter residual noise bank for E15.

Constructs realistic out-of-fold Visual-ICL Adapter residuals `e = z_hat - z_true`
STRICTLY from `fold == 'train'` episodes (ZERO access to `fold == 'val'` episodes!):
  1. Partitions all training tasks into `K = 5` disjoint task folds so every held-out
     fold mimics true unseen-task cross-layout Visual ICL.
  2. Samples cross-layout `(demo, query)` pairs strictly inside `fold == 'train'`,
     50% under `visual_knn` matching and 50% under `rel_progress` matching (zero GT
     action or query future used for matching!).
  3. For each fold `k`, fits the exact E14 scale-controlled Full-Dim Block-RMS Ridge
     Adapter `(dh, h_q, z_d) -> z_q` on the remaining `K - 1` folds (`N = 4,000` pairs
     per match mode, matching E14's exact adapter protocol) and predicts `z_hat` on fold `k`.
  4. Pools out-of-fold residuals `E_cv = z_hat - z_q` `(N_total, 3072)` plus centered
     latent reference RMS `R_M = sqrt(mean(||z_q - mu||^2))` so live training can inject
     scale-invariant realistic residual noise (or exact per-sample L2-norm-matched
     isotropic random Gaussian control noise!).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from e5.train import atomic_write_json
from e7.model import compute_behavior_anchor_steps
from e9 import LOCAL_WINDOW_LEN
from e9.stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher
from e12.eval_suite import ValClipCache, _effective_rank_stats
from e13.exp4_minimal_icl import _encode_clips_with_policy, _extract_h_obs, _gather_and_encode_split_tensors
from e14.common import (
    build_block_rms_features,
    fit_action_explainable_subspace,
    fit_predict_scaled_ridge,
    project_subspace,
)

from . import BASE_CHECKPOINTS, OUTPUT_ROOT


def build_train_only_cv_pairs(
    cache: ValClipCache,
    *,
    n_folds: int = 5,
    n_pairs_per_mode: int = 5000,
    match_modes: Sequence[str] = ("visual_knn", "rel_progress"),
    seed: int = 2026,
) -> Dict[str, Any]:
    """Build strictly train-fold-only cross-layout `(demo, query)` pairs grouped into K task folds.

    Hard zero-leakage assertion: EVERY single `d_ep` and `q_ep` has `episodes[ep]['fold'] == 'train'`.
    """
    rng = np.random.default_rng(seed)
    by_task_train: Dict[int, List[int]] = defaultdict(list)
    for ep_i, row in enumerate(cache.episodes):
        if row["fold"] != "train":
            continue
        tid = cache.task_to_id.get(row["task"])
        if tid is not None:
            by_task_train[tid].append(ep_i)

    valid_tasks = sorted(t for t, eps in by_task_train.items() if len(eps) >= 2)
    if len(valid_tasks) < n_folds:
        raise RuntimeError(f"Need at least {n_folds} multi-ep train tasks, got {len(valid_tasks)}")

    perm_tasks = rng.permutation(valid_tasks)
    task_to_fold: Dict[int, int] = {
        int(tid): int(i % n_folds) for i, tid in enumerate(perm_tasks)
    }
    fold_tasks: List[List[int]] = [[] for _ in range(n_folds)]
    for tid, fid in task_to_fold.items():
        fold_tasks[fid].append(tid)

    # Lazy stride-16 demo anchor visual key cache
    ep_anchor_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _get_demo_anchors(ep_i: int) -> Tuple[np.ndarray, np.ndarray]:
        if ep_i not in ep_anchor_cache:
            T = int(cache.ep_lengths_np[ep_i])
            off = int(cache.frame_offsets_np[ep_i])
            anchors = compute_behavior_anchor_steps(T, stride=16)
            h_mem = np.stack(
                [_extract_h_obs(cache.frame_feats_mmap, off, T, int(a)) for a in anchors],
                axis=0,
            ).reshape(len(anchors), -1)
            norms = np.clip(np.linalg.norm(h_mem, axis=-1, keepdims=True), 1e-6, None)
            ep_anchor_cache[ep_i] = (anchors, h_mem / norms)
        return ep_anchor_cache[ep_i]

    per_mode: Dict[str, Dict[str, Any]] = {}
    for m_idx, mode in enumerate(match_modes):
        rng_m = np.random.default_rng(seed + 100 * (m_idx + 1))
        n_per_fold = int(n_pairs_per_mode // n_folds)
        d_eps, d_frs, q_eps, q_frs, t_ids, f_ids = [], [], [], [], [], []
        for fid in range(n_folds):
            t_pool = fold_tasks[fid]
            for _ in range(n_per_fold):
                tid = int(t_pool[int(rng_m.integers(len(t_pool)))])
                ep_pool = by_task_train[tid]
                i_pair = rng_m.choice(len(ep_pool), size=2, replace=False)
                d_ep, q_ep = int(ep_pool[i_pair[0]]), int(ep_pool[i_pair[1]])
                T_q = int(cache.ep_lengths_np[q_ep])
                T_d = int(cache.ep_lengths_np[d_ep])
                max_q = max(1, T_q - LOCAL_WINDOW_LEN)
                q_fr = int(rng_m.integers(0, max_q))

                anchors_d, h_mem_u = _get_demo_anchors(d_ep)
                if mode == "visual_knn":
                    q_off = int(cache.frame_offsets_np[q_ep])
                    hq = _extract_h_obs(cache.frame_feats_mmap, q_off, T_q, q_fr).reshape(-1)
                    hq_u = hq / max(float(np.linalg.norm(hq)), 1e-6)
                    best_k = int(np.argmax(h_mem_u @ hq_u))
                    d_fr = int(anchors_d[best_k])
                elif mode == "rel_progress":
                    prog = float(q_fr) / max(T_q - 1, 1)
                    d_progs = anchors_d.astype(np.float32) / max(T_d - 1, 1)
                    best_k = int(np.argmin(np.abs(d_progs - prog)))
                    d_fr = int(anchors_d[best_k])
                else:
                    raise ValueError(f"Unknown match_mode: {mode}")

                d_eps.append(d_ep)
                d_frs.append(d_fr)
                q_eps.append(q_ep)
                q_frs.append(q_fr)
                t_ids.append(tid)
                f_ids.append(fid)

        split = {
            "d_ep": np.array(d_eps, dtype=np.int64),
            "d_fr": np.array(d_frs, dtype=np.int64),
            "q_ep": np.array(q_eps, dtype=np.int64),
            "q_fr": np.array(q_frs, dtype=np.int64),
            "task_id": np.array(t_ids, dtype=np.int64),
            "fold_id": np.array(f_ids, dtype=np.int64),
        }
        # STRICT ZERO-LEAKAGE ASSERTION: every episode must belong to fold == 'train'
        all_eps = set(split["d_ep"].tolist()) | set(split["q_ep"].tolist())
        bad_eps = [e for e in all_eps if cache.episodes[e]["fold"] != "train"]
        if bad_eps:
            raise AssertionError(f"LEAKAGE DETECTED: non-train episodes in CV bank: {bad_eps[:5]}")

        raw = _gather_and_encode_split_tensors(cache, split)
        hd = raw["h_d"].numpy().astype(np.float32)
        hq = raw["h_q"].numpy().astype(np.float32)
        per_mode[mode] = {
            "split": split,
            "d_clip": raw["d_clip"],
            "d_mask": raw["d_mask"],
            "q_clip": raw["q_clip"],
            "q_mask": raw["q_mask"],
            "q_act10": raw["q_act10"].numpy().astype(np.float32),
            "h_d": hd,
            "h_q": hq,
            "dh": (hq - hd).astype(np.float32),
        }

    return {
        "n_folds": n_folds,
        "match_modes": list(match_modes),
        "n_valid_train_tasks": len(valid_tasks),
        "seed": seed,
        "per_mode": per_mode,
    }


@torch.inference_mode()
def compute_cv_residual_bank_for_policy(
    policy: E9Stage1LocalPolicy,
    cv_pairs: Dict[str, Any],
    device: torch.device,
    *,
    compute_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Compute exact out-of-fold CV Adapter residuals `E_cv = z_hat - z_q` for `policy`.

    Can be called both offline (for E15-A) and live inside training (for E15-B/Combo refresh;
    takes ~4-6s on RTX A6000 for 10,000 pairs!).
    """
    was_training = policy.training
    policy.eval()
    t0 = time.monotonic()

    n_folds = int(cv_pairs["n_folds"])
    all_res: List[np.ndarray] = []
    all_zq: List[np.ndarray] = []
    all_aq: List[np.ndarray] = []
    mode_stats: Dict[str, Any] = {}

    for mode, md in cv_pairs["per_mode"].items():
        zd = _encode_clips_with_policy(policy, md["d_clip"], md["d_mask"], device, batch_size=512)
        zq = _encode_clips_with_policy(policy, md["q_clip"], md["q_mask"], device, batch_size=512)
        dh = md["dh"]
        hq = md["h_q"]
        f_ids = md["split"]["fold_id"]
        t_ids = md["split"]["task_id"]

        oof_pred = np.zeros_like(zq, dtype=np.float32)
        alphas: List[float] = []
        for fid in range(n_folds):
            val_m = f_ids == fid
            tr_m = ~val_m
            # Task disjointness assertion per fold
            if set(t_ids[tr_m].tolist()) & set(t_ids[val_m].tolist()):
                raise AssertionError(f"Task overlap in CV fold {fid}!")

            X_tr, X_ev = build_block_rms_features(
                [dh[tr_m], hq[tr_m], zd[tr_m]],
                [[dh[val_m]], [hq[val_m]], [zd[val_m]]],
            )
            preds, alpha = fit_predict_scaled_ridge(X_tr, zq[tr_m], X_ev, device)
            oof_pred[val_m] = preds[0]
            alphas.append(float(alpha))

        res = (oof_pred - zq).astype(np.float32)
        all_res.append(res)
        all_zq.append(zq)
        all_aq.append(md["q_act10"])

        # Basic per-mode diagnostics
        z_mu_m = zq.mean(axis=0, keepdims=True)
        zc_m = (zq - z_mu_m).astype(np.float64)
        R_m = float(np.sqrt(np.mean(np.sum(zc_m ** 2, axis=-1))))
        res_rms_l2 = float(np.sqrt(np.mean(np.sum(res.astype(np.float64) ** 2, axis=-1))))
        var_z = float(np.mean(zc_m ** 2))
        mse_res = float(np.mean(res.astype(np.float64) ** 2))
        mode_stats[mode] = {
            "n_pairs": int(len(zq)),
            "R_M": R_m,
            "res_rms_l2": res_rms_l2,
            "rel_err": float(res_rms_l2 / max(R_m, 1e-6)),
            "oof_r2_z": float(1.0 - mse_res / max(var_z, 1e-12)),
            "alphas": alphas,
        }

    residuals = np.concatenate(all_res, axis=0).astype(np.float32)  # (N_tot, 3072)
    zq_all = np.concatenate(all_zq, axis=0).astype(np.float32)
    aq_all = np.concatenate(all_aq, axis=0).astype(np.float32)

    z_mu = zq_all.mean(axis=0, keepdims=True)
    zc = (zq_all - z_mu).astype(np.float64)
    R_M = float(np.sqrt(np.mean(np.sum(zc ** 2, axis=-1))))
    z_coord_rms = float(np.sqrt(np.mean(zc ** 2)))
    res_l2 = np.linalg.norm(residuals.astype(np.float64), axis=-1)
    res_rms_l2 = float(np.sqrt(np.mean(res_l2 ** 2)))
    rel_err = float(res_rms_l2 / max(R_M, 1e-6))

    diag: Dict[str, Any] = {
        "n_residuals": int(len(residuals)),
        "R_M": R_M,
        "z_coord_rms": z_coord_rms,
        "res_rms_l2": res_rms_l2,
        "rel_err": rel_err,
        "res_l2_p10": float(np.quantile(res_l2, 0.10)),
        "res_l2_p50": float(np.quantile(res_l2, 0.50)),
        "res_l2_p90": float(np.quantile(res_l2, 0.90)),
        "res_mean_norm_over_rms": float(np.linalg.norm(residuals.mean(axis=0)) / max(res_rms_l2, 1e-6)),
        "per_mode": mode_stats,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }

    if compute_diagnostics:
        sub = fit_action_explainable_subspace(aq_all[:4000], zq_all[:4000], device, var_threshold=0.995)
        Q_act = sub["Q_act"]
        e_par = (residuals @ Q_act) @ Q_act.T
        e_perp = residuals - e_par
        e_par_e = float(np.mean(e_par.astype(np.float64) ** 2))
        e_perp_e = float(np.mean(e_perp.astype(np.float64) ** 2))
        er_res = _effective_rank_stats(residuals[:4000], device=device)
        er_z = _effective_rank_stats(zq_all[:4000], device=device)
        diag.update(
            {
                "err_par_energy_frac": float(e_par_e / max(e_par_e + e_perp_e, 1e-12)),
                "err_perp_energy_frac": float(e_perp_e / max(e_par_e + e_perp_e, 1e-12)),
                "z_par_energy_frac": float(sub["geo_subspace_var_frac"]),
                "z_erank": float(er_z["erank_energy"]),
                "res_erank": float(er_res["erank_energy"]),
                "res_d90": int(er_res["d_90"]),
            }
        )

    if was_training:
        policy.train()

    return {
        "residuals": residuals,  # (N_tot, 3072) float32
        "R_M": R_M,
        "z_coord_rms": z_coord_rms,
        "res_rms_l2": res_rms_l2,
        "rel_err": rel_err,
        "diagnostics": diag,
    }


class LatentResidualNoiseInjector:
    """Injects scale-invariant real CV Adapter residuals OR exact per-sample L2-norm-matched
    isotropic random Gaussian noise into `(B, 4, 768)` behavior latents during training.

    Scale-invariance guarantee (prevents Encoder norm-inflation cheating in joint training!):
      - Every noisy sample's perturbation is rescaled by `(cur_batch_R / bank_R)` so the
        relative perturbation ratio `||eps||_2 / R_M` cannot be gamed by inflating `||b_t||`.
      - When `noise_type == 'rand_matched'`, each noisy sample draws isotropic Gaussian
        `g ~ N(0, I_3072)` and rescales it to the EXACT L2 norm of the paired real CV
        residual `||e_i||_2`, guaranteeing 100% matched noise-scale distribution!
    """

    def __init__(
        self,
        residuals_np: np.ndarray,
        bank_R_M: float,
        device: torch.device,
        *,
        noise_type: str = "real_cv",
        p_noisy: float = 0.75,
        scale_mult: float = 1.0,
        zero_mean_residuals: bool = True,
        seed: int = 42,
    ):
        if noise_type not in ("none", "real_cv", "rand_matched"):
            raise ValueError(f"Unknown noise_type: {noise_type}")
        self.noise_type = noise_type
        self.p_noisy = float(p_noisy)
        self.scale_mult = float(scale_mult)
        self.zero_mean_residuals = bool(zero_mean_residuals)
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed + 15015)
        self.update_bank(residuals_np, bank_R_M)

    def update_bank(self, residuals_np: np.ndarray, bank_R_M: float) -> None:
        res = np.asarray(residuals_np, dtype=np.float32)
        if self.zero_mean_residuals:
            res = res - res.mean(axis=0, keepdims=True)
        self.bank = torch.from_numpy(res).to(self.device)  # (N_bank, 3072)
        self.bank_norms = torch.linalg.norm(self.bank, dim=-1, keepdim=True)  # (N_bank, 1)
        self.bank_R_M = max(float(bank_R_M), 1e-6)
        self.n_bank = int(self.bank.shape[0])

    @torch.no_grad()
    def sample_noise(self, b_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return detached additive noise `eps` of shape `(B, 4, 768)` (zeros on clean subset)."""
        B = b_t.shape[0]
        if self.noise_type == "none" or self.p_noisy <= 0.0:
            return torch.zeros_like(b_t), {"noisy_frac": 0.0, "rel_eps_rms": 0.0}

        b_flat = b_t.detach().float().reshape(B, -1)
        bc = b_flat - b_flat.mean(dim=0, keepdim=True)
        cur_R = float(torch.sqrt((bc ** 2).sum(dim=-1).mean().clamp_min(1e-12)).item())
        ratio = cur_R / self.bank_R_M

        idx = torch.randint(0, self.n_bank, (B,), generator=self.gen, device=self.device)
        if self.noise_type == "real_cv":
            eps_flat = self.bank[idx] * (ratio * self.scale_mult)
        elif self.noise_type == "rand_matched":
            # Exact per-sample L2-norm-matched isotropic Gaussian control!
            g = torch.randn(b_flat.shape, generator=self.gen, device=self.device, dtype=torch.float32)
            g_unit = g / torch.linalg.norm(g, dim=-1, keepdim=True).clamp_min(1e-6)
            target_norms = self.bank_norms[idx] * (ratio * self.scale_mult)
            eps_flat = g_unit * target_norms
        else:
            raise ValueError(self.noise_type)

        mask = (torch.rand((B, 1), generator=self.gen, device=self.device) < self.p_noisy).to(torch.float32)
        eps_flat = eps_flat * mask
        eps = eps_flat.reshape_as(b_t).to(dtype=b_t.dtype)

        noisy_n = max(float(mask.sum().item()), 1.0)
        rel_eps = float(
            torch.sqrt((eps_flat ** 2).sum(dim=-1).sum() / noisy_n).item() / max(cur_R, 1e-6)
        )
        return eps, {
            "noisy_frac": float(mask.mean().item()),
            "rel_eps_rms": rel_eps,
            "cur_R_M": cur_R,
            "scale_ratio": ratio,
        }


def save_residual_bank(bank_dict: Dict[str, Any], out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}_cv_residuals.npz"
    np.savez_compressed(
        npz_path,
        residuals=bank_dict["residuals"].astype(np.float16),
        R_M=np.array([bank_dict["R_M"]], dtype=np.float64),
        z_coord_rms=np.array([bank_dict["z_coord_rms"]], dtype=np.float64),
        res_rms_l2=np.array([bank_dict["res_rms_l2"]], dtype=np.float64),
        rel_err=np.array([bank_dict["rel_err"]], dtype=np.float64),
    )
    atomic_write_json(bank_dict["diagnostics"], out_dir / f"{name}_cv_residuals_diag.json")
    return npz_path


def load_residual_bank(npz_path: Path) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=False)
    return {
        "residuals": d["residuals"].astype(np.float32),
        "R_M": float(d["R_M"][0]),
        "z_coord_rms": float(d["z_coord_rms"][0]),
        "res_rms_l2": float(d["res_rms_l2"][0]),
        "rel_err": float(d["rel_err"][0]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["e9", "e12_c", "e12_bc_factordisp"])
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "cv_noise_banks")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-pairs-per-mode", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ValClipCache(seed=42)
    t0 = time.monotonic()
    print(f"[E15 NoiseBank] Building strict train-only {args.n_folds}-fold CV pairs...", flush=True)
    cv_pairs = build_train_only_cv_pairs(
        cache, n_folds=args.n_folds, n_pairs_per_mode=args.n_pairs_per_mode, seed=args.seed
    )
    print(
        f"[E15 NoiseBank] Built {args.n_pairs_per_mode}x2 train-only pairs across "
        f"{cv_pairs['n_valid_train_tasks']} tasks in {time.monotonic()-t0:.1f}s",
        flush=True,
    )

    all_diag: Dict[str, Any] = {}
    for mname in args.models:
        ckpt = BASE_CHECKPOINTS[mname]
        pol = load_frozen_e9_stage1_teacher(ckpt, device=device)
        bank = compute_cv_residual_bank_for_policy(pol, cv_pairs, device, compute_diagnostics=True)
        save_residual_bank(bank, args.out_dir, mname)
        dg = bank["diagnostics"]
        all_diag[mname] = dg
        print(
            f"  [{mname:18s}] N={dg['n_residuals']} R_M={dg['R_M']:.2f} "
            f"rel_err={dg['rel_err']:.4f} (vKNN={dg['per_mode']['visual_knn']['rel_err']:.4f}, "
            f"RelP={dg['per_mode']['rel_progress']['rel_err']:.4f}) | "
            f"err_par={dg['err_par_energy_frac']*100:.1f}% err_perp={dg['err_perp_energy_frac']*100:.1f}% | "
            f"erank(z/res)={dg['z_erank']:.1f}/{dg['res_erank']:.1f} ({dg['elapsed_seconds']}s)",
            flush=True,
        )
    atomic_write_json(all_diag, args.out_dir / "cv_noise_banks_summary.json")


if __name__ == "__main__":
    main()
