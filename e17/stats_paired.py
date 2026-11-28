"""E17 Paired Per-Task Statistical Testing & 95% Bootstrap Confidence Intervals.

Computes exact paired per-task comparisons across models on identical validation splits:
  1. Fidelity (Oracle self-reconstruction): all 5,799 non-tail val starts aggregated per
     task across all 311 LIBERO tasks (311 paired task MSE10 values).
  2. Strictly Zero-Leakage Visual ICL (`rel_progress` & `visual_knn`, Linear@4k & RBF@4k):
     3 seeds x ~94 held-out unseen tasks (~282 unseen task-seed pairs; also aggregated to
     unique unseen tasks) evaluated on identical splits & identical DDIM noise!
  3. Oracle Editability (Tier-2 Full & Pure `z_op` Arithmetic): 136 paired quadruples.

For each comparison `(B vs A)`, reports:
  - `mean_A`, `mean_B`, `delta_mean` (`B - A`, negative = B better for MSE10)
  - `ci95_low`, `ci95_high` (10,000 paired bootstrap resamples)
  - `win_rate` (% of paired tasks/quadruples where B beats A)
  - `wilcoxon_p`, `ttest_p`, `cohens_dz`, and `n_pairs`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import ValClipCache, encode_all_val_starts
from e13.exp1_training_free import _build_fixed_analogy_quadruples
from e14.common import (
    build_block_rms_features,
    build_seed_bundle,
    ddim_decode_fixed_noise,
    fit_predict_rbf_krr,
    fit_predict_scaled_ridge,
)

from . import OUTPUT_ROOT, SEEDS_DEFAULT


def paired_bootstrap_stats(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    *,
    lower_is_better: bool = True,
    n_boot: int = 10_000,
    seed: int = 1799,
) -> Dict[str, Any]:
    """Compute exact paired statistics for `B` vs `A` (`delta = B - A`)."""
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1
    n = len(a)
    diff = b - a  # negative = B better if lower_is_better

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    dz = float(mean_d / max(std_d, 1e-12))

    # Paired bootstrap 95% CI on mean(B - A) AND individual means
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_d = diff[idx].mean(axis=1)
    boot_a = a[idx].mean(axis=1)
    boot_b = b[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])
    ci_a_lo, ci_a_hi = np.percentile(boot_a, [2.5, 97.5])
    ci_b_lo, ci_b_hi = np.percentile(boot_b, [2.5, 97.5])

    win_rate = float(np.mean(b < a)) if lower_is_better else float(np.mean(b > a))

    # Paired t-test & Wilcoxon signed-rank test
    if n > 2 and std_d > 1e-12:
        t_stat, t_p = stats.ttest_rel(b, a)
        t_p = float(t_p)
        try:
            _, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            w_p = float(w_p)
        except Exception:
            w_p = 1.0
    else:
        t_p, w_p = 1.0, 1.0

    rel_pct = float(100.0 * (mean_b - mean_a) / max(abs(mean_a), 1e-12))
    return {
        "n_pairs": int(n),
        "mean_A": mean_a,
        "ci95_A": [float(ci_a_lo), float(ci_a_hi)],
        "mean_B": mean_b,
        "ci95_B": [float(ci_b_lo), float(ci_b_hi)],
        "delta_B_minus_A": mean_d,
        "rel_change_pct": rel_pct,
        "ci95_delta": [float(ci_lo), float(ci_hi)],
        "win_rate_B_over_A": win_rate,
        "cohens_dz": dz,
        "wilcoxon_p": w_p,
        "ttest_p": t_p,
        "significant_95ci": bool((ci_hi < 0.0) or (ci_lo > 0.0)),
    }


def collect_per_task_arrays(
    model_key: str,
    ckpt_path: Path,
    cache: ValClipCache,
    device: torch.device,
    *,
    op_dim: int = 384,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    """Extract per-task / per-quadruple MSE10 arrays for a single model on exact shared splits."""
    t0 = time.monotonic()
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=True)

    # 1. Fidelity per-task across all 5,799 non-tail starts (311 tasks)
    non_tail = np.flatnonzero(~encoded["is_tail"])
    z_nt = encoded["latents_flat"][non_tail]
    gt_nt = encoded["actions_10d"][non_tail]
    pred_nt = ddim_decode_fixed_noise(policy, [z_nt], device, seed=12345, batch_size=512)[0]
    per_samp_fid = np.mean((pred_nt - gt_nt) ** 2, axis=(1, 2))
    nt_tasks = cache.val_task_ids[non_tail]
    fid_by_task: Dict[str, float] = {}
    for tid in np.unique(nt_tasks):
        fid_by_task[str(int(tid))] = float(np.mean(per_samp_fid[nt_tasks == tid]))

    # 2. Oracle Editability per-quadruple (Tier-2 full & pure z_op)
    quads = _build_fixed_analogy_quadruples(cache, encoded, device, seed=42)
    A1, A2, B1, B2 = quads["tier2"]
    z4 = encoded["latents"]
    gt_B2 = encoded["actions_10d"][B2]
    z_full_ar = (z4[B1] + (z4[A2] - z4[A1])).reshape(len(B1), -1)
    z_op_ar = z4[B1].copy()
    z_op_ar[:, :, :op_dim] = z4[B1, :, :op_dim] + (z4[A2, :, :op_dim] - z4[A1, :, :op_dim])
    p_ora, p_bas, p_ful, p_op = ddim_decode_fixed_noise(
        policy,
        [z4[B2].reshape(len(B1), -1), z4[B1].reshape(len(B1), -1), z_full_ar, z_op_ar.reshape(len(B1), -1)],
        device,
        seed=12345,
        batch_size=512,
    )
    quad_base = np.mean((p_bas - gt_B2) ** 2, axis=(1, 2))
    quad_ora = np.mean((p_ora - gt_B2) ** 2, axis=(1, 2))
    quad_full = np.mean((p_ful - gt_B2) ** 2, axis=(1, 2))
    quad_op = np.mean((p_op - gt_B2) ** 2, axis=(1, 2))

    # 3. Zero-Leakage Visual ICL per unseen task-seed (rel_progress & visual_knn, Linear@4k & RBF@4k)
    policies = {model_key: policy}
    icl_per_task_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mm in ("rel_progress", "visual_knn"):
        lin_map: Dict[str, float] = {}
        rbf_map: Dict[str, float] = {}
        for seed in seeds:
            bundle = build_seed_bundle(cache, policies, match_mode=mm, seed=seed, device=device, n_train_pairs=4000)
            obs, acts, lats = bundle["obs"], bundle["acts"], bundle["latents"][model_key]
            X_tr, X_ev = build_block_rms_features(
                [obs["train"]["dh"], obs["train"]["h_q"], lats["train"]["d"]],
                [[obs["test_unseen"]["dh"]], [obs["test_unseen"]["h_q"]], [lats["test_unseen"]["d"]]],
            )
            Y_tr = lats["train"]["q"]
            gt_u = acts["test_unseen"]["q"]
            p_lin = fit_predict_scaled_ridge(X_tr, Y_tr, X_ev, device)[0][0]
            p_rbf = fit_predict_rbf_krr(X_tr, Y_tr, X_ev, device)[0][0]
            a_lin, a_rbf = ddim_decode_fixed_noise(policy, [p_lin, p_rbf], device, seed=12345, batch_size=512)
            s_lin = np.mean((a_lin - gt_u) ** 2, axis=(1, 2))
            s_rbf = np.mean((a_rbf - gt_u) ** 2, axis=(1, 2))

            # Task ID of each query in test_unseen!
            q_eps = bundle["splits"]["test_unseen"]["q_ep"]
            q_tasks = np.array([cache.task_to_id[cache.episodes[int(e)]["task"]] for e in q_eps], dtype=np.int64)
            for tid in np.unique(q_tasks):
                k_ts = f"s{seed}_t{int(tid)}"
                m = q_tasks == tid
                lin_map[k_ts] = float(np.mean(s_lin[m]))
                rbf_map[k_ts] = float(np.mean(s_rbf[m]))
        icl_per_task_seed[mm] = {"lin4k": lin_map, "rbf4k": rbf_map}

    return {
        "model_key": model_key,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "fidelity_per_task": fid_by_task,
        "fidelity_overall_mse10": float(np.mean(per_samp_fid)),
        "edit_quadruples": {
            "base_mse10": quad_base.tolist(),
            "oracle_mse10": quad_ora.tolist(),
            "full_arith_mse10": quad_full.tolist(),
            "op_arith_mse10": quad_op.tolist(),
        },
        "icl_per_task_seed": icl_per_task_seed,
    }


def compare_pair(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compute full paired statistical comparisons for B vs A across Fidelity, Editability, and Visual ICL."""
    out: Dict[str, Any] = {"A": rec_a["model_key"], "B": rec_b["model_key"]}

    # 1. Fidelity per task (311 tasks)
    t_keys = sorted(set(rec_a["fidelity_per_task"]) & set(rec_b["fidelity_per_task"]))
    out["fidelity_per_task"] = paired_bootstrap_stats(
        np.array([rec_a["fidelity_per_task"][k] for k in t_keys]),
        np.array([rec_b["fidelity_per_task"][k] for k in t_keys]),
        lower_is_better=True,
    )

    # 2. Oracle Editability per quadruple (136 quadruples)
    for k in ("full_arith_mse10", "op_arith_mse10"):
        out[f"edit_{k}"] = paired_bootstrap_stats(
            np.array(rec_a["edit_quadruples"][k]),
            np.array(rec_b["edit_quadruples"][k]),
            lower_is_better=True,
        )

    # 3. Zero-Leakage Visual ICL per unseen task-seed (and macro-averaged per unique unseen task)
    for mm in ("rel_progress", "visual_knn"):
        for adp in ("lin4k", "rbf4k"):
            ma = rec_a["icl_per_task_seed"][mm][adp]
            mb = rec_b["icl_per_task_seed"][mm][adp]
            ks = sorted(set(ma) & set(mb))
            out[f"icl_{mm}_{adp}_taskseed"] = paired_bootstrap_stats(
                np.array([ma[k] for k in ks]),
                np.array([mb[k] for k in ks]),
                lower_is_better=True,
            )
            # Also aggregate by unique task ID (strictly independent across tasks!)
            by_t_a: Dict[str, List[float]] = defaultdict(list)
            by_t_b: Dict[str, List[float]] = defaultdict(list)
            for k in ks:
                tid = k.split("_", 1)[1]
                by_t_a[tid].append(ma[k])
                by_t_b[tid].append(mb[k])
            ut = sorted(by_t_a)
            out[f"icl_{mm}_{adp}_per_task"] = paired_bootstrap_stats(
                np.array([np.mean(by_t_a[t]) for t in ut]),
                np.array([np.mean(by_t_b[t]) for t in ut]),
                lower_is_better=True,
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "paired_stats")
    p.add_argument("--models", type=str, nargs="+", required=True, help="List of key=ckpt_path")
    p.add_argument("--pairs", type=str, nargs="*", default=[], help="List of A:B comparison pairs (computes B vs A)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "per_model"
    cache_dir.mkdir(parents=True, exist_ok=True)

    val_cache: Optional[ValClipCache] = None
    model_recs: Dict[str, Dict[str, Any]] = {}
    for spec in args.models:
        k, ck = spec.split("=", 1)
        out_f = cache_dir / f"{k}.json"
        if out_f.is_file():
            model_recs[k] = json.loads(out_f.read_text())
            print(f"[PairedStats] Loaded cached {k} from {out_f}", flush=True)
            continue
        if val_cache is None:
            val_cache = ValClipCache(seed=42)
        rec = collect_per_task_arrays(k, Path(ck), val_cache, device)
        atomic_write_json(rec, out_f)
        model_recs[k] = rec
        print(f"[PairedStats] Extracted {k}: Fid={rec['fidelity_overall_mse10']:.5f} ({rec['elapsed_sec']}s)", flush=True)

    comparisons: Dict[str, Any] = {}
    for pspec in args.pairs:
        a_k, b_k = pspec.split(":", 1)
        if a_k not in model_recs or b_k not in model_recs:
            print(f"[PairedStats] Skip pair {pspec} (missing model)", flush=True)
            continue
        comp = compare_pair(model_recs[a_k], model_recs[b_k])
        comparisons[f"{b_k}_vs_{a_k}"] = comp
        fid = comp["fidelity_per_task"]
        rp = comp["icl_rel_progress_rbf4k_per_task"]
        op = comp["edit_op_arith_mse10"]
        print(
            f"[Pair {b_k} vs {a_k}] "
            f"Fid(311t): d={fid['delta_B_minus_A']:+.5f} CI=[{fid['ci95_delta'][0]:+.5f},{fid['ci95_delta'][1]:+.5f}] win={fid['win_rate_B_over_A']*100:.1f}% p={fid['wilcoxon_p']:.2e} | "
            f"RelP_RBF: d={rp['delta_B_minus_A']:+.5f} CI=[{rp['ci95_delta'][0]:+.5f},{rp['ci95_delta'][1]:+.5f}] win={rp['win_rate_B_over_A']*100:.1f}% p={rp['wilcoxon_p']:.2e} | "
            f"OpEdit: d={op['delta_B_minus_A']:+.5f} win={op['win_rate_B_over_A']*100:.1f}% p={op['wilcoxon_p']:.2e}",
            flush=True,
        )

    atomic_write_json(comparisons, args.out_dir / "paired_comparisons.json")
    print(f"[PairedStats] Saved comparisons to {args.out_dir / 'paired_comparisons.json'}", flush=True)


if __name__ == "__main__":
    main()
