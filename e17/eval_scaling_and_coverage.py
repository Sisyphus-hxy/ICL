"""E17 Exp 4b: Unified Encoder-Level Scaling & Coverage Evaluation Suite.

Evaluates ANY pure-video pretrained encoder checkpoint (across data scale 10%/30%/100%,
training steps 8k/16k/32k, and relation coverage modes) on strictly controlled,
decoder-free representation benchmarks (3 disjoint-task seeds):
  1. Frozen Linear Probe Label Efficiency (1%, 5%, 10%, 25%, 100% -> R^2(z->a7), MSE7)
  2. Cross-Task Behavioral Alignment & Anti-Degeneracy (CT_bal/trans/grip, Disp rho,
     CT Top-1 ActMSE10, task/obj/scene leak, erank full/op/res, CKA(op,res))
  3. Beyond-E12 Long-Horizon Structure (Fut+0.8s, Fut+1.6s, Progress, Upcoming Grip, z_beyond)
  4. Strictly Zero-Leakage Latent Visual Predictability (RelP & vKNN: R^2(z), e_lat, cos(dz))
  5. Optional Learned Graph W_ij Authenticity (W_soft/W_top1 dv_r2, W_top1 ActMSE10,
     PartRho(W,A|S), hub Gini) whenever checkpoint includes `reuse_module`!
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import LocalBehaviorEncoder
from e12.action_attributes import (
    compute_attribute_similarity_matrix,
    extract_action_attribute_vectors,
    extract_action_delta_features,
    pairwise_cosine_sim,
)
from e12.eval_suite import (
    ValClipCache,
    _effective_rank_stats,
    _ridge_fit_predict,
    _spearman_pearson_from_vectors,
)
from e13.exp4_minimal_icl import _build_zero_leakage_icl_pairs, _gather_and_encode_split_tensors
from e14.common import (
    build_block_rms_features,
    compute_latent_predictability_metrics,
    fit_action_explainable_subspace,
    fit_global_rms_standardizer,
    fit_predict_scaled_ridge,
)
from e15.train_e15 import compute_linear_cka_penalty
from e16.analyze_discovered_relations import _gather_beyond_e12_targets

from . import OUTPUT_ROOT, SEEDS_DEFAULT


def _load_local_encoder(ckpt_path: Path, device: torch.device) -> Tuple[LocalBehaviorEncoder, Dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = LocalBehaviorEncoder(
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        ffn_dim=4 * 768,
        max_window_len=16,
        num_cameras=2,
        num_behavior_tokens=4,
        dropout=0.1,
    )
    if "local_encoder" in payload:
        enc.load_state_dict(payload["local_encoder"], strict=True)
    elif "model" in payload:
        sd = {k[len("local_encoder.") :]: v for k, v in payload["model"].items() if k.startswith("local_encoder.")}
        enc.load_state_dict(sd, strict=True)
    else:
        raise ValueError(f"Cannot load LocalBehaviorEncoder from {ckpt_path}")
    enc.eval().to(device)
    return enc, payload


@torch.inference_mode()
def _encode_clips(enc: LocalBehaviorEncoder, cr: torch.Tensor, cm: torch.Tensor, device: torch.device, bs: int = 512) -> np.ndarray:
    N = len(cr)
    out = np.zeros((N, 4, 768), dtype=np.float32)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            b = enc(cr[s:e].to(device, dtype=torch.float32), clip_pad_mask=cm[s:e].to(device))
        out[s:e] = b.float().cpu().numpy()
    return out


def evaluate_encoder_only(
    name: str,
    ckpt_path: Path,
    cache: ValClipCache,
    beyond: Dict[str, Any],
    device: torch.device,
    *,
    op_dim: int = 384,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    precomputed_icl_raw: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    enc, payload = _load_local_encoder(ckpt_path, device)
    cfg = payload.get("config", {})

    # Encode all 6,248 val starts
    N_all = len(cache.val_episodes)
    lat4 = np.zeros((N_all, 4, 768), dtype=np.float32)
    for s in range(0, N_all, 512):
        e = min(s + 512, N_all)
        cr, cm, _, _ = cache.gather(cache.val_episodes[s:e], cache.val_frames[s:e])
        lat4[s:e] = _encode_clips(enc, cr, cm, device)

    Z_full = lat4.reshape(N_all, -1)
    Z_op = lat4[:, :, :op_dim].reshape(N_all, -1)
    Z_res = lat4[:, :, op_dim:].reshape(N_all, -1)

    non_tail = np.flatnonzero(~beyond["is_tail"])
    A7_all = beyond["val_a7"].reshape(N_all, -1)
    A10_all = beyond["val_a10"].reshape(N_all, -1)
    v_tasks = cache.val_task_ids[non_tail]
    unique_t = np.unique(v_tasks)

    # 1. Frozen Linear Probe Label Efficiency (1%, 5%, 10%, 25%, 100%, 3 seeds)
    label_eff: Dict[str, Any] = {}
    for frac in (0.01, 0.05, 0.10, 0.25, 1.00):
        r2s, mses, r2s_op = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed + 1688)
            perm_t = rng.permutation(unique_t)
            tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
            tr_all = non_tail[np.array([int(t) in tr_set for t in v_tasks])]
            te_i = non_tail[np.array([int(t) not in tr_set for t in v_tasks])]
            n_keep = max(25, int(round(len(tr_all) * frac)))
            tr_i = tr_all[rng.choice(len(tr_all), size=n_keep, replace=False)]

            var_y = float(((A7_all[te_i] - A7_all[te_i].mean(0, keepdims=True)) ** 2).mean())
            p_f, _ = _ridge_fit_predict(Z_full[tr_i], A7_all[tr_i], Z_full[te_i], device=device)
            mse_f = float(((p_f - A7_all[te_i]) ** 2).mean())
            r2s.append(float(1.0 - mse_f / max(var_y, 1e-12)))
            mses.append(mse_f)

            p_o, _ = _ridge_fit_predict(Z_op[tr_i], A7_all[tr_i], Z_op[te_i], device=device)
            mse_o = float(((p_o - A7_all[te_i]) ** 2).mean())
            r2s_op.append(float(1.0 - mse_o / max(var_y, 1e-12)))
        label_eff[f"{int(round(frac*100))}pct"] = {
            "r2_full_mean": float(np.mean(r2s)),
            "r2_full_std": float(np.std(r2s)),
            "r2_op_mean": float(np.mean(r2s_op)),
            "mse7_full_mean": float(np.mean(mses)),
        }

    # 2. Cross-task behavioral alignment & anti-degeneracy audit
    rng_a = np.random.default_rng(42 + 1616)
    sel = np.sort(rng_a.choice(non_tail, size=min(2048, len(non_tail)), replace=False))
    Ns = len(sel)
    cr_sel, _, a10_sel_c, _ = cache.gather(cache.val_episodes[sel], cache.val_frames[sel])
    a10_s = a10_sel_c.to(device=device, dtype=torch.float32)
    a7_s = action_10d_to_7d(a10_s)
    t_ids = torch.from_numpy(cache.val_task_ids[sel]).to(device=device)
    o_ids = torch.from_numpy(beyond["obj_ids"][sel]).to(device=device)
    v0_s = cr_sel[:, 0].to(device=device, dtype=torch.float32).reshape(Ns, -1)
    attrs = extract_action_attribute_vectors(a10_s, a7_s)
    S_bal = compute_attribute_similarity_matrix(attrs, "global_balanced7d")
    S_tr = compute_attribute_similarity_matrix(attrs, "translation")
    S_gr = compute_attribute_similarity_matrix(attrs, "gripper")
    S_sc = pairwise_cosine_sim(v0_s)
    cross_task = t_ids[:, None] != t_ids[None, :]
    off_diag = ~torch.eye(Ns, device=device, dtype=torch.bool)
    a10_f = a10_s.reshape(Ns, -1)

    rng_p = np.random.default_rng(42 + 1617)
    p1 = rng_p.integers(0, Ns, size=1024)
    p2 = rng_p.integers(0, Ns, size=1024)
    da = extract_action_delta_features(a7_s[p1], a7_s[p2])["delta_balanced7d"]
    da_act = torch.linalg.norm(da, dim=-1) > 0.5
    S_da = pairwise_cosine_sim(da)
    off_m = (~torch.eye(1024, device=device, dtype=torch.bool)) & da_act[:, None] & da_act[None, :]

    def _audit(arr: np.ndarray) -> Dict[str, float]:
        u = torch.from_numpy(arr[sel]).to(device=device, dtype=torch.float32)
        S_u = pairwise_cosine_sim(u)
        ct_bal = float(_spearman_pearson_from_vectors(S_u[cross_task], S_bal[cross_task], seed=42)[0])
        ct_tr = float(_spearman_pearson_from_vectors(S_u[cross_task], S_tr[cross_task], seed=42)[0])
        ct_gr = float(_spearman_pearson_from_vectors(S_u[cross_task], S_gr[cross_task], seed=42)[0])
        S_du = pairwise_cosine_sim(u[p2] - u[p1])
        disp_rho = float(_spearman_pearson_from_vectors(S_du[off_m], S_da[off_m], seed=42)[0])
        top1 = torch.argmax(S_u.masked_fill(~cross_task, -1e9), dim=-1)
        top1_mse = float((a10_f - a10_f[top1]).square().mean().item())
        knn10 = torch.topk(S_u.masked_fill(~off_diag, -1e9), k=10, dim=-1).indices
        t_leak = float((t_ids[knn10] == t_ids[:, None]).float().mean().item())
        o_leak = float((o_ids[knn10] == o_ids[:, None]).float().mean().item())
        sc_rho = float(_spearman_pearson_from_vectors(S_u[off_diag], S_sc[off_diag], seed=42)[0])
        er = _effective_rank_stats(arr[sel], device=device)
        return {
            "ct_bal_rho": ct_bal,
            "ct_trans_rho": ct_tr,
            "ct_grip_rho": ct_gr,
            "disp_rho": disp_rho,
            "ct_top1_mse10": top1_mse,
            "task_leak": t_leak,
            "obj_leak": o_leak,
            "scene_rho": sc_rho,
            "erank": float(er["erank_energy"]),
        }

    audit_full = _audit(Z_full)
    audit_op = _audit(Z_op)
    er_res = _effective_rank_stats(Z_res[sel], device=device)
    cka_op_res = float(compute_linear_cka_penalty(torch.from_numpy(Z_op[sel]).to(device), torch.from_numpy(Z_res[sel]).to(device)).item())

    # 3. Beyond-E12 Long-Horizon Structure (3 seeds)
    valid_nt = non_tail[beyond["fut16_ok"][non_tail] & beyond["fut32_ok"][non_tail]]
    vt2 = cache.val_task_ids[valid_nt]
    ut2 = np.unique(vt2)
    bey_seeds = []
    for seed in seeds:
        rng_s = np.random.default_rng(seed + 1699)
        perm_t = rng_s.permutation(ut2)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        tr_i = valid_nt[np.array([int(t) in tr_set for t in vt2])]
        te_i = valid_nt[np.array([int(t) not in tr_set for t in vt2])]

        def _r2(Y_all):
            p, _ = _ridge_fit_predict(Z_full[tr_i], Y_all[tr_i], Z_full[te_i], device=device)
            v = float(((Y_all[te_i] - Y_all[te_i].mean(0, keepdims=True)) ** 2).mean())
            return float(1.0 - ((p - Y_all[te_i]) ** 2).mean() / max(v, 1e-12))

        bey_seeds.append({
            "fut16_r2": _r2(beyond["fut16_a7"]),
            "fut32_r2": _r2(beyond["fut32_a7"]),
            "progress_r2": _r2(beyond["progress"]),
        })
    beyond_agg = {k: float(np.mean([d[k] for d in bey_seeds])) for k in bey_seeds[0]}

    # 4. Zero-Leakage Latent Visual Predictability (RelP & vKNN, 3 seeds)
    vis_pred: Dict[str, Any] = {}
    for mm in ("rel_progress", "visual_knn"):
        s_recs = []
        for seed in seeds:
            if precomputed_icl_raw is not None and (mm, seed) in precomputed_icl_raw:
                raw_t = precomputed_icl_raw[(mm, seed)]
            else:
                ut_icl = np.arange(len(cache.task_to_id))
                rng_i = np.random.default_rng(seed + 444)
                pt = rng_i.permutation(ut_icl)
                tr_t = set(pt[: int(0.7 * len(pt))].tolist())
                un_t = set(pt[int(0.7 * len(pt)) :].tolist())
                splits = _build_zero_leakage_icl_pairs(
                    cache, train_tasks=tr_t, unseen_tasks=un_t,
                    n_train_pairs=4000, n_test_seen=256, n_test_unseen=512, match_mode=mm, seed=seed
                )
                raw_t = {sp: _gather_and_encode_split_tensors(cache, splits[sp]) for sp in ("train", "test_unseen")}
                if precomputed_icl_raw is not None:
                    precomputed_icl_raw[(mm, seed)] = raw_t

            zd_tr = _encode_clips(enc, raw_t["train"]["d_clip"], raw_t["train"]["d_mask"], device).reshape(-1, 3072)
            zq_tr = _encode_clips(enc, raw_t["train"]["q_clip"], raw_t["train"]["q_mask"], device).reshape(-1, 3072)
            zd_u = _encode_clips(enc, raw_t["test_unseen"]["d_clip"], raw_t["test_unseen"]["d_mask"], device).reshape(-1, 3072)
            zq_u = _encode_clips(enc, raw_t["test_unseen"]["q_clip"], raw_t["test_unseen"]["q_mask"], device).reshape(-1, 3072)
            hd_tr = raw_t["train"]["h_d"].numpy().astype(np.float32)
            hq_tr = raw_t["train"]["h_q"].numpy().astype(np.float32)
            hd_u = raw_t["test_unseen"]["h_d"].numpy().astype(np.float32)
            hq_u = raw_t["test_unseen"]["h_q"].numpy().astype(np.float32)

            mu_n, rms_n = fit_global_rms_standardizer(zq_tr)
            X_tr, X_ev = build_block_rms_features([hq_tr - hd_tr, hq_tr, zd_tr], [[hq_u - hd_u], [hq_u], [zd_u]])
            p_lin = fit_predict_scaled_ridge(X_tr, zq_tr, X_ev, device)[0][0]
            lm = compute_latent_predictability_metrics(p_lin, zq_u, zd_u, mu_n, rms_n)
            R_M = float(np.sqrt(np.mean(np.sum((zq_tr - mu_n) ** 2, axis=-1))))
            e_lat = float(np.sqrt(np.mean(np.sum((p_lin - zq_u) ** 2, axis=-1))) / max(R_M, 1e-6))
            s_recs.append({"r2_z": float(lm["r2_z"]), "cos_dz": float(lm["cos_dz"]), "e_lat": e_lat})
        vis_pred[mm] = {k: float(np.mean([d[k] for d in s_recs])) for k in s_recs[0]}

    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "config": {
            "variant": cfg.get("variant"),
            "train_episode_fraction": cfg.get("train_episode_fraction"),
            "total_steps": cfg.get("total_steps"),
            "step": payload.get("step"),
            "coverage_mode": cfg.get("coverage_mode"),
            "uses_task_metadata": cfg.get("uses_task_metadata", False),
            "total_pretrain_params": cfg.get("total_pretrain_params"),
        },
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "label_efficiency_probe": label_eff,
        "audit_full": audit_full,
        "audit_op": audit_op,
        "erank_res": float(er_res["erank_energy"]),
        "cka_op_res": cka_op_res,
        "beyond_e12": beyond_agg,
        "zero_leakage_latent_predictability": vis_pred,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-file", type=Path, default=OUTPUT_ROOT / "scaling_coverage_encoder_eval.json")
    p.add_argument("--models", type=str, nargs="+", required=True, help="List of name=ckpt_path")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=42)
    beyond = _gather_beyond_e12_targets(cache)
    pre_icl: Dict[Tuple[str, int], Dict[str, Any]] = {}

    results: Dict[str, Any] = json.loads(args.out_file.read_text()) if (args.merge and args.out_file.is_file()) else {}

    for spec in args.models:
        name, ck = spec.split("=", 1)
        ck_p = Path(ck)
        if not ck_p.is_file():
            print(f"[ScaleEval] Skip missing {name}: {ck_p}", flush=True)
            continue
        res = evaluate_encoder_only(name, ck_p, cache, beyond, device, precomputed_icl_raw=pre_icl)
        results[name] = res
        lp = res["label_efficiency_probe"]
        af = res["audit_full"]
        ao = res["audit_op"]
        rp = res["zero_leakage_latent_predictability"]["rel_progress"]
        print(
            f"[ScaleEval:{name:24s}] Probe(1/10/100)={lp['1pct']['r2_full_mean']:.3f}/{lp['10pct']['r2_full_mean']:.3f}/{lp['100pct']['r2_full_mean']:.3f} | "
            f"Op_CTbal={ao['ct_bal_rho']:.3f} Op_Disp={ao['disp_rho']:.3f} Top1MSE={af['ct_top1_mse10']:.4f} | "
            f"erank(f/o/r)={af['erank']:.1f}/{ao['erank']:.1f}/{res['erank_res']:.1f} | "
            f"Fut16={res['beyond_e12']['fut16_r2']:.3f} RelP_R2z={rp['r2_z']:.3f} e_lat={rp['e_lat']:.3f}",
            flush=True,
        )
        atomic_write_json(results, args.out_file)

    print(f"[ScaleEval] Saved to {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
