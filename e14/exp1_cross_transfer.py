"""E14 Experiment 1: 4x4 Cross-Representation Transfer (Input A -> Target/Decoder B).

Freezes Encoder and Decoder of:
  - E9
  - E12-B
  - E12-C
  - E12-BC-FactorDisp

Runs a symmetric 4x4 (+ Vision-Only row `none -> B`) cross-representation transfer matrix
strictly reusing E13 Exp 4's zero-leakage retrieval pairs, visual features `(h_d, h_q)`,
70/30 task splits, and 3 seeds. Controls latent scale differences on BOTH input (block-RMS
energy equalization) and output (global scalar RMS standardization).

Evaluates 4 complementary adapter protocols to rule out formulation artifacts:
  1. `direct_full_ridge`  : Symmetric full-dim block-RMS Ridge `[dh, hq, z_d^(A)] -> z_q^(B)`
                            across all 7 data budgets N in {50..4000} (PRIMARY 4x4 matrix).
  2. `direct_pca_ridge`   : Matched-PCA (`k_obs=128, k_lat=64`) block-RMS Ridge at N in {100, 500, 1000, 4000}.
  3. `aligned_res_ridge`  : Cross-aligned residual Ridge `T_{A->B}(z_d^(A)) + f(dh, hq, z_d^(A))`
                            (on diagonal A==B, T=I recovers E13 Exp 4 exact residual skip).
  4. `rbf_krr` & `res_mlp`: Tuned nonlinear adapters (closed-form RBF Kernel Ridge + early-stopped
                            residual MLP on PCA-320) across the full 4x4 matrix at N in {1000, 4000}.

Quantifies Input vs Output bottleneck via Two-Way ANOVA variance decomposition of the 4x4 matrix.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from e5.train import atomic_write_json
from e12.eval_suite import ValClipCache, _action_metrics_dict, _ridge_fit_predict

from . import (
    BUDGETS_DEFAULT,
    CORE_MODELS,
    MATCH_MODES_DEFAULT,
    OUTPUT_ROOT,
    SEEDS_DEFAULT,
)
from .common import (
    build_block_rms_features,
    build_pca_block_rms_features,
    build_seed_bundle,
    compute_latent_predictability_metrics,
    ddim_decode_fixed_noise,
    fit_global_rms_standardizer,
    fit_predict_rbf_krr,
    fit_predict_scaled_ridge,
    fit_predict_tuned_mlp,
    load_policies,
    pack_action_metrics,
)


def two_way_anova_4x4(mat: np.ndarray) -> Dict[str, float]:
    """Decompose a 4x4 matrix `M[input_A, target_B]` into Input, Output, and Interaction SS%."""
    M = np.asarray(mat, dtype=np.float64)
    assert M.shape == (4, 4), f"Expected (4,4), got {M.shape}"
    grand = float(M.mean())
    row_means = M.mean(axis=1, keepdims=True)
    col_means = M.mean(axis=0, keepdims=True)

    ss_tot = float(((M - grand) ** 2).sum())
    ss_input = float(4.0 * ((row_means - grand) ** 2).sum())
    ss_output = float(4.0 * ((col_means - grand) ** 2).sum())
    resid = M - row_means - col_means + grand
    ss_inter = float((resid ** 2).sum())

    diag_vals = np.diag(M)
    off_mask = ~np.eye(4, dtype=bool)
    off_vals = M[off_mask]
    # Diagonal advantage vs same-row/col off-diagonals
    diag_adv = float(off_vals.mean() - diag_vals.mean())

    return {
        "grand_mean": grand,
        "row_means": [float(x) for x in row_means[:, 0]],
        "col_means": [float(x) for x in col_means[0, :]],
        "row_range": float(row_means.max() - row_means.min()),
        "col_range": float(col_means.max() - col_means.min()),
        "ss_tot": ss_tot,
        "ss_input_frac": float(ss_input / max(ss_tot, 1e-15)),
        "ss_output_frac": float(ss_output / max(ss_tot, 1e-15)),
        "ss_interaction_frac": float(ss_inter / max(ss_tot, 1e-15)),
        "diag_mean": float(diag_vals.mean()),
        "off_diag_mean": float(off_vals.mean()),
        "diag_advantage_mse": diag_adv,
    }


def _aggregate_metric_blocks(blocks: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(blocks[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(b[k]) for b in blocks]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def run_exp1(
    out_dir: Path,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    budgets: Tuple[int, ...] = BUDGETS_DEFAULT,
    match_modes: Tuple[str, ...] = MATCH_MODES_DEFAULT,
    key_budgets: Tuple[int, ...] = (100, 500, 1000, 4000),
    nonlin_budgets: Tuple[int, ...] = (1000, 4000),
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    policies = load_policies(CORE_MODELS, device)
    input_rows = list(CORE_MODELS) + ["none"]
    target_cols = list(CORE_MODELS)

    all_mode_results: Dict[str, Any] = {}
    per_sample_dump: Dict[str, np.ndarray] = {}

    for match_mode in match_modes:
        per_seed_data: List[Dict[str, Any]] = []
        for seed in seeds:
            ts = time.monotonic()
            print(f"[E14 Exp1] Building bundle mode={match_mode} seed={seed}...", flush=True)
            bundle = build_seed_bundle(
                cache,
                policies,
                match_mode=match_mode,
                seed=seed,
                device=device,
                n_train_pairs=max(budgets),
            )
            obs = bundle["obs"]
            acts = bundle["acts"]
            lats = bundle["latents"]

            dh_tr, hq_tr = obs["train"]["dh"], obs["train"]["h_q"]
            dh_ev = [obs["test_seen"]["dh"], obs["test_unseen"]["dh"]]
            hq_ev = [obs["test_seen"]["h_q"], obs["test_unseen"]["h_q"]]
            gt_seen, gt_unseen = acts["test_seen"]["q"], acts["test_unseen"]["q"]

            # Precompute Direct Copy (N=0) and Oracle upper bound for each target column B
            dc_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
            ora_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
            latent_stats: Dict[str, Dict[str, float]] = {}

            for B in target_cols:
                z_tr_q = lats[B]["train"]["q"]
                mu_b, rms_b = fit_global_rms_standardizer(z_tr_q)
                latent_stats[B] = {
                    "rms_centered": float(rms_b),
                    "mean_l2_norm": float(np.mean(np.linalg.norm(z_tr_q, axis=-1))),
                    "mean_centered_l2_norm": float(np.mean(np.linalg.norm(z_tr_q - mu_b, axis=-1))),
                }
                dec_s = ddim_decode_fixed_noise(
                    policies[B], [lats[B]["test_seen"]["d"], lats[B]["test_seen"]["q"]], device, seed=12345
                )
                dec_u = ddim_decode_fixed_noise(
                    policies[B], [lats[B]["test_unseen"]["d"], lats[B]["test_unseen"]["q"]], device, seed=12345
                )
                dc_metrics[B] = {
                    "seen": _action_metrics_dict(dec_s[0], gt_seen),
                    "unseen": _action_metrics_dict(dec_u[0], gt_unseen),
                }
                ora_metrics[B] = {
                    "seen": _action_metrics_dict(dec_s[1], gt_seen),
                    "unseen": _action_metrics_dict(dec_u[1], gt_unseen),
                }

            # Store latent predictions per target column B so we batch DDIM decoding per decoder!
            # Structure: pending_z[B][split_idx] -> list of (tag_tuple, z_pred_array, lat_metrics_dict)
            pending_z: Dict[str, List[List[Tuple[Tuple[str, str, int], np.ndarray, Dict[str, float]]]]] = {
                B: [[], []] for B in target_cols
            }

            # ------------------------------------------------------------------
            # 1. Direct Full-Dim Block-RMS Ridge across ALL 7 budgets (Primary 4x4 + Vision-Only)
            # ------------------------------------------------------------------
            for n_b in budgets:
                n_eff = min(int(n_b), len(dh_tr))
                # Pre-build input features per row A once, reuse across all 4 target columns B!
                X_full_by_A: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = {}
                for A in input_rows:
                    if A == "none":
                        X_full_by_A[A] = build_block_rms_features(
                            [dh_tr[:n_eff], hq_tr[:n_eff]],
                            [dh_ev, hq_ev],
                        )
                    else:
                        zd_tr = lats[A]["train"]["d"][:n_eff]
                        zd_ev = [lats[A]["test_seen"]["d"], lats[A]["test_unseen"]["d"]]
                        X_full_by_A[A] = build_block_rms_features(
                            [dh_tr[:n_eff], hq_tr[:n_eff], zd_tr],
                            [dh_ev, hq_ev, zd_ev],
                        )

                for B in target_cols:
                    Y_tr = lats[B]["train"]["q"][:n_eff]
                    mu_b, rms_b = fit_global_rms_standardizer(Y_tr)
                    z_true_ev = [lats[B]["test_seen"]["q"], lats[B]["test_unseen"]["q"]]
                    z_demo_ev = [lats[B]["test_seen"]["d"], lats[B]["test_unseen"]["d"]]

                    for A in input_rows:
                        X_tr, X_ev = X_full_by_A[A]
                        preds, alpha_sel = fit_predict_scaled_ridge(X_tr, Y_tr, X_ev, device)
                        for sp_i in (0, 1):
                            lm = compute_latent_predictability_metrics(
                                preds[sp_i], z_true_ev[sp_i], z_demo_ev[sp_i], mu_b, rms_b
                            )
                            lm["alpha"] = float(alpha_sel)
                            pending_z[B][sp_i].append((("direct_full_ridge", A, n_eff), preds[sp_i], lm))

            # ------------------------------------------------------------------
            # 2. Matched-PCA Direct Ridge, Cross-Aligned Residual Ridge, RBF-KRR, Tuned Res-MLP
            # ------------------------------------------------------------------
            for n_b in key_budgets:
                n_eff = min(int(n_b), len(dh_tr))
                X_pca_by_A: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = {}
                for A in input_rows:
                    if A == "none":
                        X_pca_by_A[A] = build_pca_block_rms_features(
                            dh_tr[:n_eff], hq_tr[:n_eff], [], dh_ev, hq_ev, [], device,
                            k_obs=128, k_lat_list=(),
                        )
                    else:
                        zd_tr = lats[A]["train"]["d"][:n_eff]
                        zd_ev = [lats[A]["test_seen"]["d"], lats[A]["test_unseen"]["d"]]
                        X_pca_by_A[A] = build_pca_block_rms_features(
                            dh_tr[:n_eff], hq_tr[:n_eff], [zd_tr], dh_ev, hq_ev, [zd_ev], device,
                            k_obs=128, k_lat_list=(64,),
                        )

                for B in target_cols:
                    Y_tr = lats[B]["train"]["q"][:n_eff]
                    mu_b, rms_b = fit_global_rms_standardizer(Y_tr)
                    z_true_ev = [lats[B]["test_seen"]["q"], lats[B]["test_unseen"]["q"]]
                    z_demo_ev = [lats[B]["test_seen"]["d"], lats[B]["test_unseen"]["d"]]

                    # 2a. Matched-PCA Direct Ridge (all 5 rows including vision-only)
                    for A in input_rows:
                        X_tr, X_ev = X_pca_by_A[A]
                        preds, alpha_sel = fit_predict_scaled_ridge(X_tr, Y_tr, X_ev, device)
                        for sp_i in (0, 1):
                            lm = compute_latent_predictability_metrics(
                                preds[sp_i], z_true_ev[sp_i], z_demo_ev[sp_i], mu_b, rms_b
                            )
                            lm["alpha"] = float(alpha_sel)
                            pending_z[B][sp_i].append((("direct_pca_ridge", A, n_eff), preds[sp_i], lm))

                    # 2b. Cross-Aligned Residual Ridge (4x4):
                    # On diagonal A==B: T_{B->B}(z_d) = z_d^(B) (exact E13 Exp 4 residual skip).
                    # Off-diagonal A!=B: T_{A->B}: z_d^(A) -> z_d^(B) fitted strictly on N train demos!
                    for A in CORE_MODELS:
                        if A == B:
                            base_tr = lats[B]["train"]["d"][:n_eff]
                            base_ev = [lats[B]["test_seen"]["d"], lats[B]["test_unseen"]["d"]]
                        else:
                            # Fit T_{A->B} strictly on train demos (ZERO query or test leakage!)
                            zd_A_tr = lats[A]["train"]["d"][:n_eff]
                            zd_B_tr = lats[B]["train"]["d"][:n_eff]
                            zd_A_ev = [lats[A]["test_seen"]["d"], lats[A]["test_unseen"]["d"]]
                            t_preds, _ = fit_predict_scaled_ridge(
                                zd_A_tr, zd_B_tr, [zd_A_tr, zd_A_ev[0], zd_A_ev[1]], device
                            )
                            base_tr = t_preds[0]
                            base_ev = [t_preds[1], t_preds[2]]

                        dY_tr = Y_tr - base_tr
                        X_tr, X_ev = X_pca_by_A[A]
                        dz_preds, alpha_sel = fit_predict_scaled_ridge(X_tr, dY_tr, X_ev, device)
                        for sp_i in (0, 1):
                            zp = base_ev[sp_i] + dz_preds[sp_i]
                            lm = compute_latent_predictability_metrics(
                                zp, z_true_ev[sp_i], z_demo_ev[sp_i], mu_b, rms_b
                            )
                            lm["alpha"] = float(alpha_sel)
                            pending_z[B][sp_i].append((("aligned_res_ridge", A, n_eff), zp, lm))

                    # 2c. Nonlinear 4x4 Adapters (RBF-KRR + Tuned Res-MLP) at nonlin_budgets
                    if n_eff in nonlin_budgets:
                        for A in CORE_MODELS:
                            X_tr, X_ev = X_pca_by_A[A]
                            krr_preds, krr_info = fit_predict_rbf_krr(X_tr, Y_tr, X_ev, device)
                            mlp_preds, mlp_info = fit_predict_tuned_mlp(
                                X_tr, Y_tr, X_ev, device,
                                hidden_dim=512, bottleneck_dim=256, max_steps=600,
                                residual_on_ridge=True, seed=seed + n_eff,
                            )
                            for sp_i in (0, 1):
                                lm_k = compute_latent_predictability_metrics(
                                    krr_preds[sp_i], z_true_ev[sp_i], z_demo_ev[sp_i], mu_b, rms_b
                                )
                                lm_k["gamma_mult"] = float(krr_info["gamma_mult"])
                                pending_z[B][sp_i].append((("rbf_krr", A, n_eff), krr_preds[sp_i], lm_k))

                                lm_m = compute_latent_predictability_metrics(
                                    mlp_preds[sp_i], z_true_ev[sp_i], z_demo_ev[sp_i], mu_b, rms_b
                                )
                                lm_m["best_step"] = float(mlp_info["best_step"])
                                pending_z[B][sp_i].append((("tuned_res_mlp", A, n_eff), mlp_preds[sp_i], lm_m))

            # ------------------------------------------------------------------
            # 3. Batched Fixed-Noise DDIM Decoding (once per target decoder B & split!)
            # ------------------------------------------------------------------
            seed_res: Dict[str, Any] = {
                "latent_stats": latent_stats,
                "direct_copy": {
                    B: {
                        sp: {
                            "mse10": float(dc_metrics[B][sp]["action10_mse"]),
                            "grip_acc": float(dc_metrics[B][sp]["grip_sign_acc"]),
                        }
                        for sp in ("seen", "unseen")
                    }
                    for B in target_cols
                },
                "oracle": {
                    B: {
                        sp: {
                            "mse10": float(ora_metrics[B][sp]["action10_mse"]),
                            "grip_acc": float(ora_metrics[B][sp]["grip_sign_acc"]),
                        }
                        for sp in ("seen", "unseen")
                    }
                    for B in target_cols
                },
                "protocols": {},
            }

            gt_by_sp = [gt_seen, gt_unseen]
            sp_names = ("seen", "unseen")
            for B in target_cols:
                for sp_i, sp_name in enumerate(sp_names):
                    items = pending_z[B][sp_i]
                    z_list = [it[1] for it in items]
                    decoded_list = ddim_decode_fixed_noise(
                        policies[B], z_list, device, seed=12345, batch_size=1024
                    )
                    for (proto, A, n_eff), pred_act, (_, _, lm) in zip(
                        [it[0] for it in items], decoded_list, items
                    ):
                        m_act = _action_metrics_dict(pred_act, gt_by_sp[sp_i])
                        packed = pack_action_metrics(m_act, dc_metrics[B][sp_name], ora_metrics[B][sp_name])
                        packed.update(lm)

                        p_dict = seed_res["protocols"].setdefault(proto, {})
                        n_dict = p_dict.setdefault(str(n_eff), {})
                        s_dict = n_dict.setdefault(sp_name, {})
                        a_dict = s_dict.setdefault(A, {})
                        a_dict[B] = packed

                        # Save per-sample unseen MSE10 at N=4000 for seed=42 (failure-case analysis)
                        if seed == seeds[0] and sp_name == "unseen" and n_eff == max(budgets) and proto == "direct_full_ridge":
                            per_samp_mse = ((pred_act - gt_unseen) ** 2).mean(axis=(1, 2)).astype(np.float32)
                            per_sample_dump[f"{match_mode}__{A}__to__{B}__mse10"] = per_samp_mse

            if seed == seeds[0]:
                # Also store test_unseen metadata for stratified failure-case analysis
                per_sample_dump[f"{match_mode}__dh_norm"] = np.linalg.norm(
                    obs["test_unseen"]["dh"], axis=-1
                ).astype(np.float32)
                hd_u = obs["test_unseen"]["h_d"]
                hq_u = obs["test_unseen"]["h_q"]
                cos_h = (hd_u * hq_u).sum(axis=-1) / (
                    np.clip(np.linalg.norm(hd_u, axis=-1), 1e-6, None)
                    * np.clip(np.linalg.norm(hq_u, axis=-1), 1e-6, None)
                )
                per_sample_dump[f"{match_mode}__cos_hd_hq"] = cos_h.astype(np.float32)
                da_u = acts["test_unseen"]["q"] - acts["test_unseen"]["d"]
                per_sample_dump[f"{match_mode}__da_mse10"] = (da_u ** 2).mean(axis=(1, 2)).astype(np.float32)
                q_grip = acts["test_unseen"]["q"][:, :, 9]
                per_sample_dump[f"{match_mode}__grip_transition"] = (
                    (np.sign(q_grip[:, 0]) != np.sign(q_grip[:, -1])).astype(np.int8)
                )

            per_seed_data.append(seed_res)
            print(
                f"[E14 Exp1] Finished mode={match_mode} seed={seed} in {time.monotonic() - ts:.1f}s",
                flush=True,
            )

        # ----------------------------------------------------------------------
        # Aggregate across seeds + compute 4x4 ANOVA for every (protocol, N, split)
        # ----------------------------------------------------------------------
        agg_mode: Dict[str, Any] = {
            "latent_stats": {
                B: _aggregate_metric_blocks([d["latent_stats"][B] for d in per_seed_data])
                for B in target_cols
            },
            "direct_copy": {
                B: {
                    sp: _aggregate_metric_blocks([d["direct_copy"][B][sp] for d in per_seed_data])
                    for sp in ("seen", "unseen")
                }
                for B in target_cols
            },
            "oracle": {
                B: {
                    sp: _aggregate_metric_blocks([d["oracle"][B][sp] for d in per_seed_data])
                    for sp in ("seen", "unseen")
                }
                for B in target_cols
            },
            "protocols": {},
            "anova_4x4": {},
        }

        proto_names = list(per_seed_data[0]["protocols"].keys())
        for proto in proto_names:
            agg_mode["protocols"][proto] = {}
            agg_mode["anova_4x4"][proto] = {}
            for n_str in per_seed_data[0]["protocols"][proto]:
                agg_mode["protocols"][proto][n_str] = {}
                agg_mode["anova_4x4"][proto][n_str] = {}
                for sp_name in ("seen", "unseen"):
                    agg_mode["protocols"][proto][n_str][sp_name] = {}
                    rows_here = list(per_seed_data[0]["protocols"][proto][n_str][sp_name].keys())
                    for A in rows_here:
                        agg_mode["protocols"][proto][n_str][sp_name][A] = {}
                        for B in target_cols:
                            blocks = [d["protocols"][proto][n_str][sp_name][A][B] for d in per_seed_data]
                            agg_mode["protocols"][proto][n_str][sp_name][A][B] = _aggregate_metric_blocks(blocks)

                    # Compute 4x4 ANOVA on MSE10, ARR, and R2(z_q) across CORE_MODELS x CORE_MODELS
                    mat_mse = np.array(
                        [
                            [agg_mode["protocols"][proto][n_str][sp_name][A][B]["mse10_mean"] for B in CORE_MODELS]
                            for A in CORE_MODELS
                        ],
                        dtype=np.float64,
                    )
                    mat_r2 = np.array(
                        [
                            [agg_mode["protocols"][proto][n_str][sp_name][A][B]["r2_z_mean"] for B in CORE_MODELS]
                            for A in CORE_MODELS
                        ],
                        dtype=np.float64,
                    )
                    agg_mode["anova_4x4"][proto][n_str][sp_name] = {
                        "mse10": two_way_anova_4x4(mat_mse),
                        "r2_z": two_way_anova_4x4(mat_r2),
                    }

        all_mode_results[match_mode] = agg_mode

        # Print crisp summary table for primary 4x4 at N=4000 unseen
        n_max_s = str(max(budgets))
        p_u = agg_mode["protocols"]["direct_full_ridge"][n_max_s]["unseen"]
        an_u = agg_mode["anova_4x4"]["direct_full_ridge"][n_max_s]["unseen"]["mse10"]
        print(
            f"\n=== [E14 Exp1 | mode={match_mode}] 4x4 Cross-Transfer MSE10 (Direct Full Ridge @ N={n_max_s}, Unseen, 3 seeds) ===",
            flush=True,
        )
        col_hdr = "Input A -> Target B"
        hdr = f"  {col_hdr:19s} | " + " | ".join(f"{B:>12s}" for B in CORE_MODELS) + " | RowMean"
        print(hdr, flush=True)
        for A in input_rows:
            vals = [p_u[A][B]["mse10_mean"] for B in CORE_MODELS]
            row_s = " | ".join(f"{v:12.5f}" for v in vals)
            print(f"  {A:18s} | {row_s} | {np.mean(vals):.5f}", flush=True)
        col_means_s = " | ".join(f"{v:12.5f}" for v in an_u["col_means"])
        print(f"  {'ColMean (4x4)':18s} | {col_means_s} | {an_u['grand_mean']:.5f}", flush=True)
        print(
            f"  ANOVA 4x4 MSE10 -> Input(Row)={an_u['ss_input_frac']*100:.1f}%, "
            f"Output/Decoder(Col)={an_u['ss_output_frac']*100:.1f}%, "
            f"Interaction={an_u['ss_interaction_frac']*100:.1f}% "
            f"(RowRange={an_u['row_range']:.5f}, ColRange={an_u['col_range']:.5f}, DiagAdv={an_u['diag_advantage_mse']:+.5f})",
            flush=True,
        )

    np.savez_compressed(out_dir / "exp1_per_sample_seed42.npz", **per_sample_dump)
    summary = {
        "core_models": list(CORE_MODELS),
        "input_rows": input_rows,
        "seeds": list(seeds),
        "budgets": list(budgets),
        "key_budgets": list(key_budgets),
        "nonlin_budgets": list(nonlin_budgets),
        "match_modes": list(match_modes),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "results": all_mode_results,
    }
    atomic_write_json(summary, out_dir / "exp1_summary.json")
    print(f"[E14 Exp1] Saved summary to {out_dir / 'exp1_summary.json'} ({summary['elapsed_seconds']}s)", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp1_cross_transfer")
    args = p.parse_args()
    run_exp1(args.out_dir)


if __name__ == "__main__":
    main()
