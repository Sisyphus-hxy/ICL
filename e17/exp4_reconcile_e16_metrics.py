"""E17 Exp 4a: Reconcile E16 Frozen-Encoder Decoder Metrics & Unified Label-Efficiency Evaluation.

Directly audits the discrepancy in E16 between:
  - Protocol A (E16 Sec 4.1 `metrics.json`): First 512 validation starts in sequential order
    (covers only 27/311 tasks + includes 61 tail clips!), evaluated on `last_ema.pt`.
  - Protocol B (E16 Sec 3 `eval_e15.py`): 1,024 uniformly sampled non-tail starts across all 311 tasks.
  - Protocol C (E17 Unified Full Val): ALL 5,799 non-tail validation starts across all 311 tasks,
    saving per-task MSE10 arrays for paired statistical tests & 95% bootstrap CIs!
  - Protocol D (E17 Unified Zero-Leakage Visual ICL): 3-seed unseen-task `rel_progress` & `visual_knn`
    for ALL label fractions (1%, 10%, 100%) — previously omitted for 1% and 10% in E16!

Evaluates BOTH `best.pt` (early-stopped on full 6,248 val diffusion loss) AND `last_ema.pt`
for every frozen-decoder checkpoint to verify overfitting effects at low label fractions (1%, 10%).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import ValClipCache, _action_metrics_dict, encode_all_val_starts
from e14.common import (
    build_block_rms_features,
    build_seed_bundle,
    ddim_decode_fixed_noise,
    fit_predict_rbf_krr,
    fit_predict_scaled_ridge,
)

from . import E16_DECODER_CKPTS, OUTPUT_ROOT, SEEDS_DEFAULT


def _eval_ckpt_all_protocols(
    ckpt_path: Path,
    cache: ValClipCache,
    device: torch.device,
    *,
    run_icl: bool = True,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=True)

    N_all = len(cache.val_episodes)
    non_tail = np.flatnonzero(~encoded["is_tail"])
    z_all = encoded["latents_flat"]
    gt_all = encoded["actions_10d"]

    # Decode ALL 6,248 val starts ONCE with deterministic fixed noise -> covers Protocols A, B, and C!
    pred_all = ddim_decode_fixed_noise(policy, [z_all], device, seed=12345, batch_size=512)[0]

    # Protocol A: First 512 starts (exact E16 metrics.json prefix subset: 27 tasks, 61 tails!)
    m_pref512 = _action_metrics_dict(pred_all[:512], gt_all[:512])

    # Protocol B: Exact eval_e15.py 1,024 uniform non-tail starts (seed=42+101)
    rng_b = np.random.default_rng(42 + 101)
    sel_1024 = np.sort(rng_b.choice(non_tail, size=min(1024, len(non_tail)), replace=False))
    m_unif1024 = _action_metrics_dict(pred_all[sel_1024], gt_all[sel_1024])

    # Protocol C: ALL 5,799 non-tail starts across ALL 311 tasks + per-task MSE10!
    m_full_nt = _action_metrics_dict(pred_all[non_tail], gt_all[non_tail])
    m_full_all = _action_metrics_dict(pred_all, gt_all)
    per_sample_mse = np.mean((pred_all[non_tail] - gt_all[non_tail]) ** 2, axis=(1, 2))
    nt_tasks = cache.val_task_ids[non_tail]
    per_task_mse = {}
    for tid in np.unique(nt_tasks):
        per_task_mse[int(tid)] = float(np.mean(per_sample_mse[nt_tasks == tid]))

    out: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "protoA_prefix512_27tasks": {
            "mse10": float(m_pref512["action10_mse"]),
            "mae7": float(m_pref512["action7_mae"]),
            "grip_acc": float(m_pref512["grip_sign_acc"]),
        },
        "protoB_uniform1024_311tasks": {
            "mse10": float(m_unif1024["action10_mse"]),
            "mae7": float(m_unif1024["action7_mae"]),
            "grip_acc": float(m_unif1024["grip_sign_acc"]),
        },
        "protoC_full5799_nontail_311tasks": {
            "mse10": float(m_full_nt["action10_mse"]),
            "mae7": float(m_full_nt["action7_mae"]),
            "grip_acc": float(m_full_nt["grip_sign_acc"]),
            "task_macro_mse10": float(np.mean(list(per_task_mse.values()))),
            "n_tasks": int(len(per_task_mse)),
        },
        "protoC_full6248_all": {
            "mse10": float(m_full_all["action10_mse"]),
            "grip_acc": float(m_full_all["grip_sign_acc"]),
        },
        "per_task_mse10_nontail": per_task_mse,
    }

    if run_icl:
        icl_out: Dict[str, Any] = {}
        policies = {"m": policy}
        for mm in ("rel_progress", "visual_knn"):
            lin_mses, rbf_mses = [], []
            for seed in seeds:
                bundle = build_seed_bundle(cache, policies, match_mode=mm, seed=seed, device=device, n_train_pairs=4000)
                obs, acts, lats = bundle["obs"], bundle["acts"], bundle["latents"]["m"]
                X_tr, X_ev = build_block_rms_features(
                    [obs["train"]["dh"], obs["train"]["h_q"], lats["train"]["d"]],
                    [[obs["test_unseen"]["dh"]], [obs["test_unseen"]["h_q"]], [lats["test_unseen"]["d"]]],
                )
                Y_tr = lats["train"]["q"]
                gt_u = acts["test_unseen"]["q"]
                p_lin = fit_predict_scaled_ridge(X_tr, Y_tr, X_ev, device)[0][0]
                p_rbf = fit_predict_rbf_krr(X_tr, Y_tr, X_ev, device)[0][0]
                a_lin, a_rbf = ddim_decode_fixed_noise(policy, [p_lin, p_rbf], device, seed=12345, batch_size=512)
                lin_mses.append(float(np.mean((a_lin - gt_u) ** 2)))
                rbf_mses.append(float(np.mean((a_rbf - gt_u) ** 2)))
            icl_out[mm] = {
                "lin4k_mse10_mean": float(np.mean(lin_mses)),
                "lin4k_mse10_std": float(np.std(lin_mses)),
                "rbf4k_mse10_mean": float(np.mean(rbf_mses)),
                "rbf4k_mse10_std": float(np.std(rbf_mses)),
            }
        out["protoD_zero_leakage_icl"] = icl_out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp4_reconcile")
    p.add_argument("--extra-dirs", type=str, nargs="*", default=[], help="Extra name=run_dir")
    p.add_argument("--only-extra", action="store_true")
    p.add_argument("--skip-icl", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=42)

    targets: List[Tuple[str, Path]] = []
    if not args.only_extra:
        for k, d in E16_DECODER_CKPTS.items():
            targets.append((k, d))
    for spec in args.extra_dirs:
        name, d = spec.split("=", 1)
        targets.append((name, Path(d)))

    out_path = args.out_dir / "reconcile_summary.json"
    results: Dict[str, Any] = json.loads(out_path.read_text()) if (args.only_extra and out_path.is_file()) else {}

    for name, run_dir in targets:
        if not run_dir.is_dir():
            print(f"[Reconcile] Skip missing {name}: {run_dir}", flush=True)
            continue
        t0 = time.monotonic()
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").is_file() else {}
        met = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").is_file() else {}
        rec: Dict[str, Any] = {
            "run_name": name,
            "action_fraction": cfg.get("action_fraction"),
            "total_steps": cfg.get("total_steps"),
            "best_step": met.get("best_step"),
            "metrics_json_last_ddim512": met.get("history", [{}])[-1].get("val", {}).get("ddim", {}).get("self", {}).get("action10_mse"),
        }
        for tag in ("best", "last_ema"):
            ck = run_dir / f"{tag}.pt"
            if ck.is_file():
                # Run Visual ICL on best.pt (and on last_ema.pt if best_step == total_steps skip duplicate!)
                same_as_best = (tag == "last_ema" and met.get("best_step") == cfg.get("total_steps") and "best" in rec)
                if same_as_best:
                    rec[tag] = dict(rec["best"])
                    rec[tag]["checkpoint"] = str(ck)
                else:
                    do_icl = (not args.skip_icl) and (tag == "best" or name.endswith("dec100"))
                    rec[tag] = _eval_ckpt_all_protocols(ck, cache, device, run_icl=do_icl)
        rec["elapsed_sec"] = round(time.monotonic() - t0, 2)
        results[name] = rec
        b = rec.get("best", rec.get("last_ema", {}))
        l = rec.get("last_ema", b)
        icl_s = ""
        if "protoD_zero_leakage_icl" in b:
            rp = b["protoD_zero_leakage_icl"]["rel_progress"]
            icl_s = f" | RelP(Lin/RBF)={rp['lin4k_mse10_mean']:.4f}/{rp['rbf4k_mse10_mean']:.4f}"
        print(
            f"[Reconcile:{name:26s}] frac={str(rec['action_fraction']):4s} steps={str(rec['total_steps']):5s} best@{str(rec['best_step']):5s} | "
            f"Pref512={l['protoA_prefix512_27tasks']['mse10']:.5f} Unif1024={l['protoB_uniform1024_311tasks']['mse10']:.5f} "
            f"FullNT(last/best)={l['protoC_full5799_nontail_311tasks']['mse10']:.5f}/{b['protoC_full5799_nontail_311tasks']['mse10']:.5f}{icl_s}",
            flush=True,
        )
        atomic_write_json(results, out_path)

    print(f"[Reconcile] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
