"""E17 Exp 3 (Diagnostic): D3 vs D4 Latent Geometry, Effective Rank, Information Distribution & Decoder Sensitivity.

Directly diagnoses WHY E16-D3 (VidRel-Op) and E16-D4 (LearnReuse-Op) exhibit complementary
strengths and weaknesses:
  1. Spectral & Geometric Analysis: Full singular-value spectrum of `z_full`, `z_op`, `z_res`,
     effective rank, participation ratio, d_50/80/90/95, isotropy, and cosine spread.
  2. Subspace Information Distribution (`z_op` vs `z_res` vs `z_full`):
     Probes 9 targets (`a7`, `pos`, `rot`, `grip`, `fut16`, `fut32`, `progress`, `v0`, `a->z`),
     plus `CKA(z_op, z_res)`, `R^2(z_op -> z_res)`, `R^2(z_res -> z_op)`, and energy fractions.
  3. Frozen Decoder Sensitivity & Attribution Audit:
     - Equal-scale perturbations strictly inside `z_op` vs `z_res` vs `z_full` (`sigma = 0.10, 0.25, 0.50`)
     - Mean-ablation of `z_op` (keeping `z_res`) vs mean-ablation of `z_res` (keeping `z_op`)
     - Cross-task full-slice swap of `z_op` vs `z_res` -> exact DDIM action adoption %!
     - Zero-Leakage Visual ICL predictability split (`R^2_z`, `e_lat` in `z_op` vs `z_res`)
       + Oracle diagnostic hybrid substitution (`pred z_op + gt z_res` vs `gt z_op + pred z_res`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _effective_rank_stats,
    _ridge_fit_predict,
    encode_all_val_starts,
)
from e14.common import (
    build_block_rms_features,
    build_seed_bundle,
    ddim_decode_fixed_noise,
    fit_predict_scaled_ridge,
)
from e15.train_e15 import compute_linear_cka_penalty
from e16.analyze_discovered_relations import _gather_beyond_e12_targets

from . import E16_DECODER_CKPTS, OUTPUT_ROOT, SEEDS_DEFAULT


def _svd_spectrum(mat: np.ndarray, device: torch.device, top_k: int = 128) -> Dict[str, Any]:
    x = torch.from_numpy(np.asarray(mat, dtype=np.float32)).to(device)
    xc = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(xc).double().cpu().numpy()
    e = s ** 2
    tot = max(float(e.sum()), 1e-12)
    p = e / tot
    er = _effective_rank_stats(mat, device=device)
    return {
        "erank": float(er["erank_energy"]),
        "pr": float(er["pr"]),
        "d_50": int(er["d_50"]),
        "d_80": int(er["d_80"]),
        "d_90": int(er["d_90"]),
        "d_95": int(er["d_95"]),
        "rms_centered": float(er["rms_centered"]),
        "top_energy_fracs": [float(v) for v in p[:top_k]],
        "cum_energy_fracs": [float(v) for v in np.cumsum(p)[:top_k]],
    }


def diagnose_model_geometry(
    name: str,
    ckpt_path: Path,
    cache: ValClipCache,
    beyond: Dict[str, Any],
    device: torch.device,
    *,
    op_dim: int = 384,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=True)

    N_all = len(cache.val_episodes)
    Z4 = encoded["latents"]  # (N, 4, 768)
    Z_full = Z4.reshape(N_all, -1)
    Z_op = Z4[:, :, :op_dim].reshape(N_all, -1)
    Z_res = Z4[:, :, op_dim:].reshape(N_all, -1)

    non_tail = np.flatnonzero(~encoded["is_tail"])
    valid_nt = non_tail[beyond["fut16_ok"][non_tail] & beyond["fut32_ok"][non_tail]]
    v_tasks = cache.val_task_ids[valid_nt]
    unique_t = np.unique(v_tasks)

    # 1. Spectral & Geometric Analysis on non-tail samples
    spec = {
        "full": _svd_spectrum(Z_full[non_tail], device),
        "op": _svd_spectrum(Z_op[non_tail], device),
        "res": _svd_spectrum(Z_res[non_tail], device),
    }

    # Energy fraction split (z_op vs z_res)
    e_op = float(np.mean(np.sum((Z_op[non_tail] - Z_op[non_tail].mean(0, keepdims=True)) ** 2, axis=-1)))
    e_res = float(np.mean(np.sum((Z_res[non_tail] - Z_res[non_tail].mean(0, keepdims=True)) ** 2, axis=-1)))
    energy_split = {
        "op_energy_frac": float(e_op / max(e_op + e_res, 1e-12)),
        "res_energy_frac": float(e_res / max(e_op + e_res, 1e-12)),
        "op_rms": float(np.sqrt(np.mean(Z_op[non_tail] ** 2))),
        "res_rms": float(np.sqrt(np.mean(Z_res[non_tail] ** 2))),
    }

    # 2. Subspace Information Distribution (3-seed disjoint-task Ridge probes)
    A10 = encoded["actions_10d"].reshape(N_all, -1)
    A7 = encoded["actions_7d"]  # (N, 16, 7)
    targets = {
        "a7_all": A7.reshape(N_all, -1),
        "pos_3d": A7[:, :, :3].reshape(N_all, -1),
        "rot_3d": A7[:, :, 3:6].reshape(N_all, -1),
        "grip_1d": A7[:, :, 6:7].reshape(N_all, -1),
        "fut16": beyond["fut16_a7"],
        "fut32": beyond["fut32_a7"],
        "progress": beyond["progress"],
        "vis_hq": encoded["h_queries"].reshape(N_all, -1),
    }

    info_by_space: Dict[str, List[Dict[str, float]]] = {"full": [], "op": [], "res": []}
    coupling_seeds: List[Dict[str, float]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed + 1733)
        perm_t = rng.permutation(unique_t)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        tr_i = valid_nt[np.array([int(t) in tr_set for t in v_tasks])]
        te_i = valid_nt[np.array([int(t) not in tr_set for t in v_tasks])]

        for sname, R in [("full", Z_full), ("op", Z_op), ("res", Z_res)]:
            rec: Dict[str, float] = {}
            for tname, Y in targets.items():
                pred, _ = _ridge_fit_predict(R[tr_i], Y[tr_i], R[te_i], device=device)
                v = float(np.mean((Y[te_i] - Y[te_i].mean(0, keepdims=True)) ** 2))
                rec[f"r2_{tname}"] = float(1.0 - np.mean((pred - Y[te_i]) ** 2) / max(v, 1e-12))
            # Reverse R^2(a10 -> R)
            p_az, _ = _ridge_fit_predict(A10[tr_i], R[tr_i], A10[te_i], device=device)
            v_r = float(np.mean((R[te_i] - R[te_i].mean(0, keepdims=True)) ** 2))
            rec["r2_a10_to_z"] = float(1.0 - np.mean((p_az - R[te_i]) ** 2) / max(v_r, 1e-12))
            info_by_space[sname].append(rec)

        # Cross-subspace coupling: CKA(z_op, z_res), R^2(z_op -> z_res), R^2(z_res -> z_op)
        cka = float(
            compute_linear_cka_penalty(
                torch.from_numpy(Z_op[te_i]).to(device),
                torch.from_numpy(Z_res[te_i]).to(device),
            ).item()
        )
        p_o2r, _ = _ridge_fit_predict(Z_op[tr_i], Z_res[tr_i], Z_op[te_i], device=device)
        v_res = float(np.mean((Z_res[te_i] - Z_res[te_i].mean(0, keepdims=True)) ** 2))
        r2_o2r = float(1.0 - np.mean((p_o2r - Z_res[te_i]) ** 2) / max(v_res, 1e-12))
        p_r2o, _ = _ridge_fit_predict(Z_res[tr_i], Z_op[tr_i], Z_res[te_i], device=device)
        v_op = float(np.mean((Z_op[te_i] - Z_op[te_i].mean(0, keepdims=True)) ** 2))
        r2_r2o = float(1.0 - np.mean((p_r2o - Z_op[te_i]) ** 2) / max(v_op, 1e-12))
        coupling_seeds.append({"cka_op_res": cka, "r2_op_to_res": r2_o2r, "r2_res_to_op": r2_r2o})

    def _agg(dl: List[Dict[str, float]]) -> Dict[str, float]:
        return {k: float(np.mean([d[k] for d in dl])) for k in dl[0]}

    info_summary = {s: _agg(dl) for s, dl in info_by_space.items()}
    coupling_summary = _agg(coupling_seeds)

    # 3. Frozen Decoder Sensitivity Audit (Perturbation, Mean-Ablation, and Cross-Task Slice Swap!)
    rng_d = np.random.default_rng(1734)
    sel = np.sort(rng_d.choice(non_tail, size=min(512, len(non_tail)), replace=False))
    Ns = len(sel)
    z4_s = Z4[sel].copy()  # (Ns, 4, 768)
    gt_s = encoded["actions_10d"][sel]
    t_s = cache.val_task_ids[sel]

    # Pick cross-task donor for each sample
    donors = np.empty(Ns, dtype=np.int64)
    for i in range(Ns):
        cands = np.flatnonzero(t_s != t_s[i])
        donors[i] = int(rng_d.choice(cands))
    gt_d = gt_s[donors]

    mu_op = Z4[non_tail, :, :op_dim].mean(axis=0, keepdims=True)
    mu_res = Z4[non_tail, :, op_dim:].mean(axis=0, keepdims=True)
    R_op = float(np.sqrt(np.mean(np.sum((Z4[non_tail, :, :op_dim] - mu_op).reshape(-1, 4 * op_dim) ** 2, axis=-1))))
    R_res = float(np.sqrt(np.mean(np.sum((Z4[non_tail, :, op_dim:] - mu_res).reshape(-1, 4 * (768 - op_dim)) ** 2, axis=-1))))
    R_full = float(np.sqrt(np.mean(np.sum((Z_full[non_tail] - Z_full[non_tail].mean(0, keepdims=True)) ** 2, axis=-1))))

    pending_tags: List[str] = ["clean", "donor_clean"]
    pending_z: List[np.ndarray] = [z4_s.reshape(Ns, -1), z4_s[donors].reshape(Ns, -1)]

    # Mean ablations
    z_ab_op = z4_s.copy()
    z_ab_op[:, :, :op_dim] = mu_op
    z_ab_res = z4_s.copy()
    z_ab_res[:, :, op_dim:] = mu_res
    pending_tags.extend(["ablate_op_to_mean", "ablate_res_to_mean"])
    pending_z.extend([z_ab_op.reshape(Ns, -1), z_ab_res.reshape(Ns, -1)])

    # Full slice swaps (cross-task donor z_op into base vs donor z_res into base)
    z_swap_op = z4_s.copy()
    z_swap_op[:, :, :op_dim] = z4_s[donors, :, :op_dim]
    z_swap_res = z4_s.copy()
    z_swap_res[:, :, op_dim:] = z4_s[donors, :, op_dim:]
    pending_tags.extend(["swap_op_from_donor", "swap_res_from_donor"])
    pending_z.extend([z_swap_op.reshape(Ns, -1), z_swap_res.reshape(Ns, -1)])

    # Equal-scale Gaussian perturbations strictly inside z_op vs z_res vs z_full
    # Both RELATIVE to own slice radius (sigma * R_slice) AND ABSOLUTE (matched L2 norm = sigma * R_full!)
    for sig in (0.10, 0.25, 0.50):
        g_op = rng_d.standard_normal((Ns, 4, op_dim))
        g_op_n = g_op / max(float(np.sqrt(np.mean(np.sum(g_op.reshape(Ns, -1) ** 2, axis=-1)))), 1e-12)
        g_rs = rng_d.standard_normal((Ns, 4, 768 - op_dim))
        g_rs_n = g_rs / max(float(np.sqrt(np.mean(np.sum(g_rs.reshape(Ns, -1) ** 2, axis=-1)))), 1e-12)

        # (a) Matched absolute L2 norm (sig * R_full) -> pure Decoder Jacobian sensitivity comparison!
        z_po_abs = z4_s.copy()
        z_po_abs[:, :, :op_dim] += (sig * R_full) * g_op_n
        z_pr_abs = z4_s.copy()
        z_pr_abs[:, :, op_dim:] += (sig * R_full) * g_rs_n

        # (b) Matched relative to own slice radius (sig * R_op vs sig * R_res)
        z_po_rel = z4_s.copy()
        z_po_rel[:, :, :op_dim] += (sig * R_op) * g_op_n
        z_pr_rel = z4_s.copy()
        z_pr_rel[:, :, op_dim:] += (sig * R_res) * g_rs_n

        pending_tags.extend([
            f"pert_abs_op_{sig:.2f}", f"pert_abs_res_{sig:.2f}",
            f"pert_rel_op_{sig:.2f}", f"pert_rel_res_{sig:.2f}",
        ])
        pending_z.extend([
            z_po_abs.reshape(Ns, -1), z_pr_abs.reshape(Ns, -1),
            z_po_rel.reshape(Ns, -1), z_pr_rel.reshape(Ns, -1),
        ])

    dec_acts = ddim_decode_fixed_noise(policy, pending_z, device, seed=12345, batch_size=512)
    act_map = dict(zip(pending_tags, dec_acts))
    clean_a = act_map["clean"]
    donor_a = act_map["donor_clean"]
    clean_mse = float(np.mean((clean_a - gt_s) ** 2))
    base_to_donor_mse = float(np.mean((clean_a - gt_d) ** 2))
    donor_oracle_mse = float(np.mean((donor_a - gt_d) ** 2))

    def _shift_rec(pa: np.ndarray) -> Dict[str, float]:
        mse_gt = float(np.mean((pa - gt_s) ** 2))
        shift_rms = float(math.sqrt(max(float(np.mean((pa - clean_a) ** 2)), 0.0)))
        mse_donor = float(np.mean((pa - gt_d) ** 2))
        adopt_donor = float((base_to_donor_mse - mse_donor) / max(base_to_donor_mse - donor_oracle_mse, 1e-8))
        return {
            "mse10_vs_base_gt": mse_gt,
            "excess_mse10": float(max(mse_gt - clean_mse, 0.0)),
            "shift_rms_vs_clean": shift_rms,
            "mse10_vs_donor_gt": mse_donor,
            "donor_adoption_ratio": adopt_donor,
        }

    dec_sens: Dict[str, Any] = {
        "clean_mse10": clean_mse,
        "R_op": R_op,
        "R_res": R_res,
        "R_full": R_full,
        "ablate_op_to_mean": _shift_rec(act_map["ablate_op_to_mean"]),
        "ablate_res_to_mean": _shift_rec(act_map["ablate_res_to_mean"]),
        "swap_op_from_donor": _shift_rec(act_map["swap_op_from_donor"]),
        "swap_res_from_donor": _shift_rec(act_map["swap_res_from_donor"]),
        "perturbations": {},
    }
    for sig in (0.10, 0.25, 0.50):
        dec_sens["perturbations"][f"{sig:.2f}"] = {
            "abs_op": _shift_rec(act_map[f"pert_abs_op_{sig:.2f}"]),
            "abs_res": _shift_rec(act_map[f"pert_abs_res_{sig:.2f}"]),
            "rel_op": _shift_rec(act_map[f"pert_rel_op_{sig:.2f}"]),
            "rel_res": _shift_rec(act_map[f"pert_rel_res_{sig:.2f}"]),
        }
    # Summary ratio: how strongly Decoder relies on z_op vs z_res
    op_sw = dec_sens["swap_op_from_donor"]["donor_adoption_ratio"]
    rs_sw = dec_sens["swap_res_from_donor"]["donor_adoption_ratio"]
    op_p = dec_sens["perturbations"]["0.25"]["abs_op"]["shift_rms_vs_clean"]
    rs_p = dec_sens["perturbations"]["0.25"]["abs_res"]["shift_rms_vs_clean"]
    dec_sens["op_vs_res_swap_dominance"] = float(op_sw / max(op_sw + rs_sw, 1e-6))
    dec_sens["op_vs_res_jacobian_ratio_025"] = float(op_p / max(rs_p, 1e-6))

    # 4. Zero-Leakage Visual ICL Subspace Error Attribution (z_op vs z_res, rel_progress & visual_knn)
    icl_sub_diag: Dict[str, Any] = {}
    policies = {name: policy}
    for mm in ("rel_progress", "visual_knn"):
        seed_recs = []
        for seed in seeds:
            bundle = build_seed_bundle(cache, policies, match_mode=mm, seed=seed, device=device, n_train_pairs=4000)
            obs = bundle["obs"]
            acts = bundle["acts"]
            lats = bundle["latents"][name]
            X_tr, X_ev = build_block_rms_features(
                [obs["train"]["dh"], obs["train"]["h_q"], lats["train"]["d"]],
                [[obs["test_unseen"]["dh"]], [obs["test_unseen"]["h_q"]], [lats["test_unseen"]["d"]]],
            )
            zq_tr = lats["train"]["q"]
            zq_u = lats["test_unseen"]["q"]
            gt_u = acts["test_unseen"]["q"]
            z_pred = fit_predict_scaled_ridge(X_tr, zq_tr, X_ev, device)[0][0]

            zq4 = zq_u.reshape(len(zq_u), 4, 768)
            zp4 = z_pred.reshape(len(zq_u), 4, 768)
            ztr4 = zq_tr.reshape(len(zq_tr), 4, 768)

            def _slice_r2_elat(y_tr_s, y_u_s, p_u_s):
                mu = y_tr_s.mean(0, keepdims=True)
                var_u = float(np.mean((y_u_s - y_u_s.mean(0, keepdims=True)) ** 2))
                mse_u = float(np.mean((p_u_s - y_u_s) ** 2))
                R_s = float(np.sqrt(np.mean(np.sum((y_tr_s - mu) ** 2, axis=-1))))
                e_s = float(np.sqrt(np.mean(np.sum((p_u_s - y_u_s) ** 2, axis=-1))) / max(R_s, 1e-6))
                return float(1.0 - mse_u / max(var_u, 1e-12)), e_s

            r2_op, elat_op = _slice_r2_elat(
                ztr4[:, :, :op_dim].reshape(len(zq_tr), -1),
                zq4[:, :, :op_dim].reshape(len(zq_u), -1),
                zp4[:, :, :op_dim].reshape(len(zq_u), -1),
            )
            r2_res, elat_res = _slice_r2_elat(
                ztr4[:, :, op_dim:].reshape(len(zq_tr), -1),
                zq4[:, :, op_dim:].reshape(len(zq_u), -1),
                zp4[:, :, op_dim:].reshape(len(zq_u), -1),
            )

            # Oracle hybrid diagnostic: (a) full pred, (b) pred z_op + GT z_res, (c) GT z_op + pred z_res, (d) full GT
            z_hyb_pred_op = zq4.copy()
            z_hyb_pred_op[:, :, :op_dim] = zp4[:, :, :op_dim]
            z_hyb_pred_res = zq4.copy()
            z_hyb_pred_res[:, :, op_dim:] = zp4[:, :, op_dim:]
            a_pred, a_hop, a_hres, a_gt = ddim_decode_fixed_noise(
                policy,
                [z_pred, z_hyb_pred_op.reshape(len(zq_u), -1), z_hyb_pred_res.reshape(len(zq_u), -1), zq_u],
                device,
                seed=12345,
                batch_size=512,
            )
            m_pred = float(np.mean((a_pred - gt_u) ** 2))
            m_hop = float(np.mean((a_hop - gt_u) ** 2))
            m_hres = float(np.mean((a_hres - gt_u) ** 2))
            m_gt = float(np.mean((a_gt - gt_u) ** 2))
            seed_recs.append({
                "r2_z_op": r2_op,
                "e_lat_op": elat_op,
                "r2_z_res": r2_res,
                "e_lat_res": elat_res,
                "mse10_full_pred": m_pred,
                "mse10_oracle_diag_pred_op_gt_res": m_hop,
                "mse10_oracle_diag_gt_op_pred_res": m_hres,
                "mse10_oracle_gt": m_gt,
                "excess_from_op_err": float(max(m_hop - m_gt, 0.0)),
                "excess_from_res_err": float(max(m_hres - m_gt, 0.0)),
            })
        icl_sub_diag[mm] = _agg(seed_recs)

    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "op_dim": int(op_dim),
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "spectrum": spec,
        "energy_split": energy_split,
        "info_distribution": info_summary,
        "subspace_coupling": coupling_summary,
        "decoder_sensitivity": dec_sens,
        "icl_subspace_attribution": icl_sub_diag,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp3_geometry")
    p.add_argument("--extra-ckpts", type=str, nargs="*", default=[], help="Extra name=ckpt")
    p.add_argument("--only-extra", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=42)
    beyond = _gather_beyond_e12_targets(cache)

    targets: List[Tuple[str, Path]] = []
    if not args.only_extra:
        targets.extend([
            ("e16_d1_lam", E16_DECODER_CKPTS["e16_d1_lam_dec100"] / "last_ema.pt"),
            ("e16_d2_randrel", E16_DECODER_CKPTS["e16_d2_randrel_dec100"] / "last_ema.pt"),
            ("e16_d3_vid_op", E16_DECODER_CKPTS["e16_d3_vidop_dec100"] / "last_ema.pt"),
            ("e16_d4_learn_op", E16_DECODER_CKPTS["e16_d4_learnop_dec100"] / "last_ema.pt"),
            ("e16_d_oracle_e9", E16_DECODER_CKPTS["e16_d_oracle_e9_dec100"] / "last_ema.pt"),
        ])
    for spec in args.extra_ckpts:
        name, ck = spec.split("=", 1)
        targets.append((name, Path(ck)))

    summary_path = args.out_dir / "geometry_summary.json"
    results: Dict[str, Any] = json.loads(summary_path.read_text()) if (args.only_extra and summary_path.is_file()) else {}

    for name, ck in targets:
        if not ck.is_file():
            print(f"[Exp3] Skip missing {name}: {ck}", flush=True)
            continue
        res = diagnose_model_geometry(name, ck, cache, beyond, device)
        results[name] = res
        sp = res["spectrum"]
        ds = res["decoder_sensitivity"]
        inf = res["info_distribution"]
        rp = res["icl_subspace_attribution"]["rel_progress"]
        print(
            f"[Exp3:{name:20s}] erank(f/op/rs)={sp['full']['erank']:.1f}/{sp['op']['erank']:.1f}/{sp['res']['erank']:.1f} "
            f"CKA={res['subspace_coupling']['cka_op_res']:.3f} | R2(op/rs->a7)={inf['op']['r2_a7_all']:.3f}/{inf['res']['r2_a7_all']:.3f} "
            f"R2(op/rs->grip)={inf['op']['r2_grip_1d']:.3f}/{inf['res']['r2_grip_1d']:.3f} | "
            f"SwapAdopt(op/rs)={ds['swap_op_from_donor']['donor_adoption_ratio']*100:.1f}%/{ds['swap_res_from_donor']['donor_adoption_ratio']*100:.1f}% "
            f"JacRatio(op/rs)={ds['op_vs_res_jacobian_ratio_025']:.2f} | "
            f"RelP_ICL: R2(op/rs)={rp['r2_z_op']:.3f}/{rp['r2_z_res']:.3f} Exc(op/rs)={rp['excess_from_op_err']:.5f}/{rp['excess_from_res_err']:.5f}",
            flush=True,
        )
        atomic_write_json(results, summary_path)

    print(f"[Exp3] Saved geometry diagnosis to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
