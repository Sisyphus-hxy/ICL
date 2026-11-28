"""E18 Pillar 4: Multi-Level Generalization Benchmark — Strictly Separating Encoder-Unseen vs Adapter-Unseen.

Evaluates frozen pure-video encoders across a 4-level generalization hierarchy:
  - Level 1b (Same-Scene, TRUE Encoder-Unseen Tasks): `libero_goal_chain_selected_view` (10 tasks)
  - Level 2  (Cross-Object / Cross-Spatial / Cross-Scene Unseen LIBERO):
        `libero_object` (10 tasks), `libero_spatial` (10 tasks), `libero_10` (10 unseen-room tasks)
        Measures:
          (a) Zero-Shot Frozen Decoder Transfer (Goal-Chain encoder + Goal-Chain decoder, 0 target data!)
          (b) Encoder-Unseen + Adapter-Seen   (split by episode within target tasks)
          (c) Encoder-Unseen + Adapter-Unseen (split by task: 70% train tasks -> 30% unseen tasks, 3 seeds)
              across 1%/10%/100% action probing + Zero-Leakage Visual ICL + Zero-Shot Cross-Task Reuse
              + Zero-Shot Tier-2 3D Editability!
  - Level 3  (Cross-Dataset / Cross-Simulator RLBench-18, CoppeliaSim):
        Strictly separates:
          (a) Encoder-Unseen + Adapter-Seen   (split by episode: train eps -> val/test eps, all 18 tasks)
          (b) Encoder-Unseen + Adapter-Unseen (split by task: 12 train tasks -> 6 unseen tasks, 3 seeds)
        Measures: 1%/10%/100% action probing R^2, Zero-Leakage Visual ICL (R^2_z & action MSE),
        Zero-Shot Cross-Task Reuse error reduction, and Subgoal Gripper-Boundary Detection AUC!
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from e5.train import atomic_write_json
from e9 import LOCAL_WINDOW_LEN
from e9.stage1_local_model import LocalBehaviorEncoder, load_frozen_e9_stage1_teacher
from e12.action_attributes import pairwise_cosine_sim
from e12.eval_suite import _action_metrics_dict, _effective_rank_stats, _ridge_fit_predict
from e14.common import build_block_rms_features, ddim_decode_fixed_noise, fit_predict_scaled_ridge

from . import GEN_CACHE_DIR, OUTPUT_ROOT, SEEDS_DEFAULT


def _load_encoder_only(ckpt_path: Path, device: torch.device) -> LocalBehaviorEncoder:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = LocalBehaviorEncoder(
        embed_dim=768, num_layers=6, num_heads=12, ffn_dim=4 * 768,
        max_window_len=16, num_cameras=2, num_behavior_tokens=4, dropout=0.1,
    )
    if "local_encoder" in payload:
        enc.load_state_dict(payload["local_encoder"], strict=True)
    else:
        sd = {k[len("local_encoder."):]: v for k, v in payload["model"].items() if k.startswith("local_encoder.")}
        enc.load_state_dict(sd, strict=True)
    return enc.eval().to(device)


@torch.inference_mode()
def _encode_clips(
    enc: LocalBehaviorEncoder,
    frame_features: np.ndarray,
    frame_offsets: np.ndarray,
    ep_lengths: np.ndarray,
    ep_arr: np.ndarray,
    fr_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode clips -> `(Z4 (N, 4, 768), V0 (N, 2, 768))`."""
    N = len(ep_arr)
    Z4 = np.empty((N, 4, 768), dtype=np.float32)
    V0 = np.empty((N, 2, 768), dtype=np.float32)
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type=device.type, enabled=False)
    win = np.arange(LOCAL_WINDOW_LEN, dtype=np.int64)[None, :]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        ep_s = ep_arr[s:e]
        fr_s = fr_arr[s:e]
        lens = ep_lengths[ep_s]
        offs = frame_offsets[ep_s]
        raw = fr_s[:, None] + win
        pad = raw >= lens[:, None]
        clamped = np.minimum(raw, np.maximum(lens - 1, 0)[:, None])
        g = offs[:, None] + clamped
        rgb = frame_features[g].astype(np.float32)
        rgb[pad] = 0.0
        V0[s:e] = rgb[:, 0]
        cr = torch.from_numpy(rgb).to(device, non_blocking=True)
        cm = torch.from_numpy(pad).to(device, non_blocking=True)
        with amp:
            b = enc(cr, clip_pad_mask=cm)
        Z4[s:e] = b.float().cpu().numpy()
    return Z4, V0


def _build_icl_pairs_rel_progress(
    ep_arr: np.ndarray,
    fr_arr: np.ndarray,
    ep_lengths: np.ndarray,
    task_ids: np.ndarray,
    indices: np.ndarray,
    *,
    max_pairs: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match each query clip in `indices` to a same-task, different-episode demo clip at closest relative progress."""
    rng = np.random.default_rng(seed)
    sub = indices if len(indices) <= max_pairs else np.sort(rng.choice(indices, size=max_pairs, replace=False))
    prog = fr_arr / np.maximum(ep_lengths[ep_arr] - 1, 1).astype(np.float32)

    q_list, d_list = [], []
    by_task: Dict[int, np.ndarray] = {}
    for tid in np.unique(task_ids[indices]):
        by_task[int(tid)] = indices[task_ids[indices] == tid]

    for qi in sub:
        tid = int(task_ids[qi])
        cands = by_task[tid]
        diff = cands[ep_arr[cands] != ep_arr[qi]]
        if len(diff) == 0:
            continue
        best = diff[int(np.argmin(np.abs(prog[diff] - prog[qi])))]
        q_list.append(int(qi))
        d_list.append(int(best))
    return np.array(q_list, dtype=np.int64), np.array(d_list, dtype=np.int64)


def _eval_split_probing_and_icl(
    Z_full: np.ndarray,
    Z_op: np.ndarray,
    V0_flat: np.ndarray,
    A_flat: np.ndarray,
    ep_arr: np.ndarray,
    fr_arr: np.ndarray,
    ep_lengths: np.ndarray,
    task_ids: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    device: torch.device,
    *,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate 1%/10%/100% action probing + Zero-Leakage Visual ICL on a given `(tr_idx, te_idx)` split."""
    rng = np.random.default_rng(seed + 1840)
    var_a = float(((A_flat[te_idx] - A_flat[te_idx].mean(0, keepdims=True)) ** 2).mean())

    def _r2(pred: np.ndarray, gt: np.ndarray, v: float) -> float:
        return float(1.0 - np.mean((pred - gt) ** 2) / max(v, 1e-12))

    # 100%, 10%, 1% action probing (full & op)
    p100, al100 = _ridge_fit_predict(Z_full[tr_idx], A_flat[tr_idx], Z_full[te_idx], device=device)
    pop100, _ = _ridge_fit_predict(Z_op[tr_idx], A_flat[tr_idx], Z_op[te_idx], device=device)

    n10 = max(16, int(round(0.10 * len(tr_idx))))
    i10 = tr_idx[rng.choice(len(tr_idx), size=min(n10, len(tr_idx)), replace=False)]
    p10, _ = _ridge_fit_predict(Z_full[i10], A_flat[i10], Z_full[te_idx], device=device)

    n01 = max(8, int(round(0.01 * len(tr_idx))))
    i01 = tr_idx[rng.choice(len(tr_idx), size=min(n01, len(tr_idx)), replace=False)]
    p01, _ = _ridge_fit_predict(Z_full[i01], A_flat[i01], Z_full[te_idx], device=device)

    # Zero-Leakage Visual ICL: (dh = h_q - h_d, h_q, z_d) -> z_hat_q -> Ridge action decode!
    tr_q, tr_d = _build_icl_pairs_rel_progress(ep_arr, fr_arr, ep_lengths, task_ids, tr_idx, max_pairs=2000, seed=seed)
    te_q, te_d = _build_icl_pairs_rel_progress(ep_arr, fr_arr, ep_lengths, task_ids, te_idx, max_pairs=1000, seed=seed + 1)
    if len(tr_q) >= 32 and len(te_q) >= 16:
        X_tr, X_ev = build_block_rms_features(
            [V0_flat[tr_q] - V0_flat[tr_d], V0_flat[tr_q], Z_full[tr_d]],
            [[V0_flat[te_q] - V0_flat[te_d]], [V0_flat[te_q]], [Z_full[te_d]]],
        )
        z_hat = fit_predict_scaled_ridge(X_tr, Z_full[tr_q], X_ev, device)[0][0]
        var_z = float(((Z_full[te_q] - Z_full[te_q].mean(0, keepdims=True)) ** 2).mean())
        icl_r2_z = _r2(z_hat, Z_full[te_q], var_z)
        a_hat, _ = _ridge_fit_predict(Z_full[tr_idx], A_flat[tr_idx], z_hat, alphas=(al100,), device=device)
        icl_act_mse = float(np.mean((a_hat - A_flat[te_q]) ** 2))
        var_a_q = float(((A_flat[te_q] - A_flat[te_q].mean(0, keepdims=True)) ** 2).mean())
        icl_act_r2 = _r2(a_hat, A_flat[te_q], var_a_q)
    else:
        icl_r2_z, icl_act_mse, icl_act_r2 = 0.0, float("nan"), 0.0

    return {
        "probe_r2_01": _r2(p01, A_flat[te_idx], var_a),
        "probe_r2_10": _r2(p10, A_flat[te_idx], var_a),
        "probe_r2_100": _r2(p100, A_flat[te_idx], var_a),
        "probe_op_r2_100": _r2(pop100, A_flat[te_idx], var_a),
        "icl_r2_z": float(icl_r2_z),
        "icl_act_mse": float(icl_act_mse),
        "icl_act_r2": float(icl_act_r2),
    }


def _eval_zero_shot_reuse_and_subgoals(
    Z_op: np.ndarray,
    V0_flat: np.ndarray,
    A_seq: np.ndarray,
    ep_arr: np.ndarray,
    fr_arr: np.ndarray,
    task_ids: np.ndarray,
    device: torch.device,
    *,
    seed: int = 1841,
) -> Dict[str, float]:
    """Zero-shot cross-task donor reuse error reduction (z_op-NN vs v0-NN vs random) + subgoal boundary AUC."""
    rng = np.random.default_rng(seed)
    N = len(Z_op)
    sub = np.sort(rng.choice(N, size=min(1024, N), replace=False))
    t_sub = task_ids[sub]
    ct = t_sub[:, None] != t_sub[None, :]
    S_op = pairwise_cosine_sim(torch.from_numpy(Z_op[sub]).to(device)).cpu().numpy()
    S_v0 = pairwise_cosine_sim(torch.from_numpy(V0_flat[sub]).to(device)).cpu().numpy()

    j_op = np.argmax(np.where(ct, S_op, -2.0), axis=1)
    j_v0 = np.argmax(np.where(ct, S_v0, -2.0), axis=1)
    j_rn = np.array([rng.choice(np.flatnonzero(ct[i])) for i in range(len(sub))])

    A_sub = A_seq[sub]
    mse_op = float(np.mean((A_sub - A_sub[j_op]) ** 2))
    mse_v0 = float(np.mean((A_sub - A_sub[j_v0]) ** 2))
    mse_rn = float(np.mean((A_sub - A_sub[j_rn]) ** 2))

    # Subgoal boundary detection ROC-AUC along episodes (gripper state switches!)
    scores_l, labels_l = [], []
    for ep in np.unique(ep_arr)[:80]:
        idx = np.flatnonzero(ep_arr == ep)
        if len(idx) < 8:
            continue
        order = idx[np.argsort(fr_arr[idx])]
        z_ep = Z_op[order]
        g_ep = A_seq[order, 0, -1]  # gripper state at start of clip
        vel = np.linalg.norm(z_ep[1:] - z_ep[:-1], axis=-1)
        g_chg = (np.abs(g_ep[1:] - g_ep[:-1]) > 0.5).astype(np.int32)
        if g_chg.sum() > 0 and g_chg.sum() < len(g_chg):
            scores_l.append(vel)
            labels_l.append(g_chg)
    if scores_l:
        subgoal_auc = float(roc_auc_score(np.concatenate(labels_l), np.concatenate(scores_l)))
    else:
        subgoal_auc = 0.5

    return {
        "ct_zop_nn_mse": mse_op,
        "ct_v0_nn_mse": mse_v0,
        "ct_rand_mse": mse_rn,
        "ct_reuse_err_red_pct": float(100.0 * (mse_rn - mse_op) / max(mse_rn, 1e-8)),
        "ct_reuse_vs_v0_pct": float(100.0 * (mse_v0 - mse_op) / max(mse_v0, 1e-8)),
        "subgoal_boundary_auc": subgoal_auc,
    }


def evaluate_model_on_libero_unseen(
    model_key: str,
    dec_ckpt: Optional[Path],
    enc_ckpt: Optional[Path],
    cache_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a model on all 4 completely Encoder-Unseen LIBERO suites."""
    episodes = json.loads((cache_dir / "episodes.json").read_text())
    feats = np.load(cache_dir / "frame_features.npy", mmap_mode="r")
    a10_all = np.load(cache_dir / "actions_10d.npy", mmap_mode="r")
    a7_all = np.load(cache_dir / "actions_7d.npy", mmap_mode="r")
    ep_lengths = np.load(cache_dir / "ep_lengths.npy", allow_pickle=False)
    frame_offsets = np.load(cache_dir / "frame_offsets.npy", allow_pickle=False)

    policy = load_frozen_e9_stage1_teacher(dec_ckpt, device=device) if dec_ckpt is not None and dec_ckpt.is_file() else None
    enc = policy.local_encoder if policy is not None else _load_encoder_only(enc_ckpt, device)

    # Build stride=8 non-tail clip starts across all 800 unseen episodes
    ep_l, fr_l, suite_l = [], [], []
    tasks_sorted = sorted({e["task"] for e in episodes})
    t2i = {t: i for i, t in enumerate(tasks_sorted)}
    ep_tids = np.array([t2i[e["task"]] for e in episodes], dtype=np.int64)

    for ep_i, row in enumerate(episodes):
        T = int(row["frames"])
        max_s = max(1, T - LOCAL_WINDOW_LEN + 1)
        fs = np.arange(0, max_s, 8, dtype=np.int64)
        ep_l.append(np.full(len(fs), ep_i, dtype=np.int64))
        fr_l.append(fs)
        suite_l.extend([row["suite"]] * len(fs))
    ep_arr = np.concatenate(ep_l)
    fr_arr = np.concatenate(fr_l)
    suites_arr = np.array(suite_l, dtype=object)
    tids_arr = ep_tids[ep_arr]

    Z4, V0 = _encode_clips(enc, feats, frame_offsets, ep_lengths, ep_arr, fr_arr, device)
    Z_full = Z4.reshape(len(ep_arr), -1)
    Z_op = Z4[:, :, :384].reshape(len(ep_arr), -1)
    V0_flat = V0.reshape(len(ep_arr), -1)

    # Gather 16-step actions
    win = np.arange(LOCAL_WINDOW_LEN, dtype=np.int64)[None, :]
    g = frame_offsets[ep_arr][:, None] + np.minimum(fr_arr[:, None] + win, (ep_lengths[ep_arr] - 1)[:, None])
    A10 = np.asarray(a10_all[g], dtype=np.float32)
    A7 = np.asarray(a7_all[g], dtype=np.float32)
    A7_f = A7.reshape(len(ep_arr), -1)

    suite_results: Dict[str, Any] = {}
    for suite in ("libero_goal_chain_selected_view", "libero_object", "libero_spatial", "libero_10"):
        s_idx = np.flatnonzero(suites_arr == suite)
        s_eps = np.unique(ep_arr[s_idx])
        s_tasks = np.unique(tids_arr[s_idx])

        # (a) Zero-Shot Frozen Decoder Transfer (Goal-Chain encoder + Goal-Chain decoder -> 0 target data!)
        zs_dec = {}
        if policy is not None:
            rng_z = np.random.default_rng(1842)
            sel = np.sort(rng_z.choice(s_idx, size=min(512, len(s_idx)), replace=False))
            pred_a10 = ddim_decode_fixed_noise(policy, [Z_full[sel]], device, seed=12345, batch_size=512)[0]
            m = _action_metrics_dict(pred_a10, A10[sel])
            zs_dec = {
                "zs_decoder_mse10": float(m["action10_mse"]),
                "zs_decoder_mae7": float(m["action7_mae"]),
                "zs_decoder_grip_acc": float(m["grip_sign_acc"]),
            }

        # (b) Encoder-Unseen + Adapter-Seen (episode split within each task: 70% eps -> 30% eps)
        seen_runs = []
        for sd in SEEDS_DEFAULT:
            r = np.random.default_rng(sd + 1843)
            tr_ep_set = set()
            for tid in s_tasks:
                teps = np.unique(ep_arr[s_idx[tids_arr[s_idx] == tid]])
                r.shuffle(teps)
                tr_ep_set.update(teps[: max(1, int(round(0.7 * len(teps))))].tolist())
            tr_m = np.array([int(e) in tr_ep_set for e in ep_arr[s_idx]])
            seen_runs.append(
                _eval_split_probing_and_icl(
                    Z_full, Z_op, V0_flat, A7_f, ep_arr, fr_arr, ep_lengths, tids_arr,
                    s_idx[tr_m], s_idx[~tr_m], device, seed=sd,
                )
            )

        # (c) Encoder-Unseen + Adapter-Unseen (strict task split: 7 tasks -> 3 completely unseen tasks!)
        unseen_runs = []
        for sd in SEEDS_DEFAULT:
            r = np.random.default_rng(sd + 1844)
            perm = r.permutation(s_tasks)
            tr_t_set = set(perm[:7].tolist())
            tr_m = np.array([int(t) in tr_t_set for t in tids_arr[s_idx]])
            unseen_runs.append(
                _eval_split_probing_and_icl(
                    Z_full, Z_op, V0_flat, A7_f, ep_arr, fr_arr, ep_lengths, tids_arr,
                    s_idx[tr_m], s_idx[~tr_m], device, seed=sd,
                )
            )

        # (d) Zero-Shot Cross-Task Reuse & Subgoal AUC
        zs_reuse = _eval_zero_shot_reuse_and_subgoals(
            Z_op[s_idx], V0_flat[s_idx], A7[s_idx], ep_arr[s_idx], fr_arr[s_idx], tids_arr[s_idx], device
        )

        def _mean_dict(dl: List[Dict[str, float]]) -> Dict[str, float]:
            return {k: float(np.nanmean([d[k] for d in dl])) for k in dl[0]}

        suite_results[suite] = {
            "num_clips": int(len(s_idx)),
            "zero_shot_decoder": zs_dec,
            "enc_unseen_adapter_seen": _mean_dict(seen_runs),
            "enc_unseen_adapter_unseen": _mean_dict(unseen_runs),
            "zero_shot_reuse": zs_reuse,
            "erank_full": float(_effective_rank_stats(Z_full[s_idx[:2048]], device=device)["erank_energy"]),
            "erank_op": float(_effective_rank_stats(Z_op[s_idx[:2048]], device=device)["erank_energy"]),
        }

    return {"model_key": model_key, "suites": suite_results}


def evaluate_model_on_rlbench(
    model_key: str,
    enc_ckpt: Path,
    cache_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Cross-Dataset / Cross-Simulator Generalization on RLBench-18 (strictly separating Adapter-Seen vs Adapter-Unseen)."""
    episodes = json.loads((cache_dir / "episodes.json").read_text())
    feats = np.load(cache_dir / "frame_features.npy", mmap_mode="r")
    a7_all = np.load(cache_dir / "actions_7d.npy", mmap_mode="r")
    ep_lengths = np.load(cache_dir / "ep_lengths.npy", allow_pickle=False)
    frame_offsets = np.load(cache_dir / "frame_offsets.npy", allow_pickle=False)

    enc = _load_encoder_only(enc_ckpt, device)

    tasks_sorted = sorted({e["task"] for e in episodes})
    t2i = {t: i for i, t in enumerate(tasks_sorted)}
    ep_tids = np.array([t2i[e["task"]] for e in episodes], dtype=np.int64)
    ep_splits = np.array([e["split"] for e in episodes], dtype=object)

    ep_l, fr_l = [], []
    for ep_i, row in enumerate(episodes):
        T = int(row["frames"])
        max_s = max(1, T - LOCAL_WINDOW_LEN + 1)
        fs = np.arange(0, max_s, 6, dtype=np.int64)
        ep_l.append(np.full(len(fs), ep_i, dtype=np.int64))
        fr_l.append(fs)
    ep_arr = np.concatenate(ep_l)
    fr_arr = np.concatenate(fr_l)
    tids_arr = ep_tids[ep_arr]
    splits_arr = ep_splits[ep_arr]

    Z4, V0 = _encode_clips(enc, feats, frame_offsets, ep_lengths, ep_arr, fr_arr, device)
    Z_full = Z4.reshape(len(ep_arr), -1)
    Z_op = Z4[:, :, :384].reshape(len(ep_arr), -1)
    V0_flat = V0.reshape(len(ep_arr), -1)

    win = np.arange(LOCAL_WINDOW_LEN, dtype=np.int64)[None, :]
    g = frame_offsets[ep_arr][:, None] + np.minimum(fr_arr[:, None] + win, (ep_lengths[ep_arr] - 1)[:, None])
    A7 = np.asarray(a7_all[g], dtype=np.float32)
    A7_f = A7.reshape(len(ep_arr), -1)

    # (A) Encoder-Unseen + Adapter-Seen: train on RLBench train split (all 18 tasks), test on RLBench val+test!
    tr_seen = np.flatnonzero(splits_arr == "train")
    te_seen = np.flatnonzero(splits_arr != "train")
    rng_sub = np.random.default_rng(1845)
    if len(tr_seen) > 6000:
        tr_seen = np.sort(rng_sub.choice(tr_seen, size=6000, replace=False))
    if len(te_seen) > 4000:
        te_seen = np.sort(rng_sub.choice(te_seen, size=4000, replace=False))
    seen_res = _eval_split_probing_and_icl(
        Z_full, Z_op, V0_flat, A7_f, ep_arr, fr_arr, ep_lengths, tids_arr, tr_seen, te_seen, device, seed=42
    )

    # (B) Encoder-Unseen + Adapter-Unseen: strict task holdout (12 train tasks -> 6 completely unseen RLBench tasks, 3 seeds!)
    all_idx = np.arange(len(ep_arr))
    if len(all_idx) > 9000:
        all_idx = np.sort(rng_sub.choice(all_idx, size=9000, replace=False))
    unique_t = np.unique(tids_arr)
    unseen_runs = []
    for sd in SEEDS_DEFAULT:
        r = np.random.default_rng(sd + 1846)
        perm = r.permutation(unique_t)
        tr_t = set(perm[:12].tolist())
        tr_m = np.array([int(t) in tr_t for t in tids_arr[all_idx]])
        unseen_runs.append(
            _eval_split_probing_and_icl(
                Z_full, Z_op, V0_flat, A7_f, ep_arr, fr_arr, ep_lengths, tids_arr,
                all_idx[tr_m], all_idx[~tr_m], device, seed=sd,
            )
        )
    unseen_res = {k: float(np.nanmean([d[k] for d in unseen_runs])) for k in unseen_runs[0]}
    unseen_std = {f"{k}_std": float(np.nanstd([d[k] for d in unseen_runs])) for k in unseen_runs[0]}

    # (C) Zero-Shot Cross-Task Reuse & Subgoal Detection on RLBench val+test
    zs_reuse = _eval_zero_shot_reuse_and_subgoals(
        Z_op[te_seen], V0_flat[te_seen], A7[te_seen], ep_arr[te_seen], fr_arr[te_seen], tids_arr[te_seen], device
    )

    return {
        "model_key": model_key,
        "num_clips": int(len(ep_arr)),
        "enc_unseen_adapter_seen": seen_res,
        "enc_unseen_adapter_unseen": {**unseen_res, **unseen_std},
        "zero_shot_reuse": zs_reuse,
        "erank_full": float(_effective_rank_stats(Z_full[te_seen[:2048]], device=device)["erank_energy"]),
        "erank_op": float(_effective_rank_stats(Z_op[te_seen[:2048]], device=device)["erank_energy"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="JSON mapping model_key -> {enc_ckpt, dec_ckpt}")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "generalization_eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())

    lib_cache = GEN_CACHE_DIR / "libero_unseen"
    rlb_cache = GEN_CACHE_DIR / "rlbench_18"
    out_all: Dict[str, Any] = {"libero_unseen": {}, "rlbench_18": {}}

    for key, info in manifest.items():
        t0 = time.monotonic()
        enc_p = Path(info["enc_ckpt"]) if info.get("enc_ckpt") else None
        dec_p = Path(info["dec_ckpt"]) if info.get("dec_ckpt") else None
        if enc_p is None and dec_p is not None:
            enc_p = dec_p
        print(f"[E18 Gen] Evaluating {key}...", flush=True)
        if lib_cache.is_dir():
            r_lib = evaluate_model_on_libero_unseen(key, dec_p, enc_p, lib_cache, device)
            out_all["libero_unseen"][key] = r_lib
        if rlb_cache.is_dir() and enc_p is not None:
            r_rlb = evaluate_model_on_rlbench(key, enc_p, rlb_cache, device)
            out_all["rlbench_18"][key] = r_rlb
        print(f"[E18 Gen] Done {key} in {time.monotonic()-t0:.1f}s", flush=True)
        atomic_write_json(out_all, args.output_dir / "generalization_summary.json")


if __name__ == "__main__":
    main()
