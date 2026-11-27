"""E13 Experiment 4: Strictly Zero-Leakage Minimal Offline ICL Transfer.

Freezes Encoder and Action Decoder completely. At inference time, adapters receive
ONLY:
  - `z_demo`  : Demo clip latent encoded from the provided Demo video
  - `h_demo`  : Demo initial observation CLIP features `[t_d-1, t_d]`
  - `h_query` : Query initial observation CLIP features `[t_q-1, t_q]`

STRICTLY FORBIDDEN at inference:
  - Query GT action `a_query` (used ONLY post-decoding for offline metric computation)
  - Query future frames `V_{t_q+1 : t_q+16}`
  - Query true latent `z_query` (used ONLY as adapter training target on strictly
    disjoint training episodes/layouts/tasks!)

Demo anchor selection at inference ALSO obeys zero-leakage:
  - Primary protocol (`visual_knn`): `t_d = argmax_k cos(h_query, h_demo_mem[k])`
    over stride-16 anchors of the Demo video (pure visual observation matching!).
  - Secondary protocol (`rel_progress`): `t_d` nearest in relative episode progress
    `t_q / T_q` (zero visual retrieval error, still ZERO GT action or query future!).

Compares across 3 seeds and 7 data budgets N in {50, 100, 250, 500, 1000, 2000, 4000}:
  - Adapters: Direct Copy (N=0), Linear Adapter (Residual Ridge), Lightweight 2-Layer MLP
  - Models: E9, E9-Cont, E12-A, E12-B, E12-C, Raw Action ICL, Action-AE ICL
  - Test Splits:
      1. Seen-Task, Unseen-Layout (fold=val episodes of the 217 train tasks; 0 layout/ep overlap)
      2. Unseen-Task, Unseen-Layout (fold=val episodes of the 94 held-out tasks; 0 task/ep overlap)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.train import atomic_write_json
from e7.model import compute_behavior_anchor_steps
from e9 import ACTION_DIM_10D, ACTION_HORIZON, DEFAULT_DENSE_CACHE_DIR, E5_CACHE_DIR, LOCAL_WINDOW_LEN
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _ridge_fit_predict,
    ddim_decode_batched,
)

from . import CHECKPOINTS, OUTPUT_ROOT, SEEDS_DEFAULT
from .exp2_action_baselines import ActionAutoencoder, train_or_load_action_ae


def _extract_h_obs(
    feats_mmap: np.ndarray, off: int, T: int, frame: int
) -> np.ndarray:
    """Extract `[max(0, t-1), t]` dual-camera CLIP features -> `(4, 768)` float32."""
    p0 = max(int(frame) - 1, 0)
    p1 = min(int(frame), T - 1)
    h = np.empty((4, 768), dtype=np.float32)
    h[0:2] = np.asarray(feats_mmap[off + p0], dtype=np.float32)
    h[2:4] = np.asarray(feats_mmap[off + p1], dtype=np.float32)
    return h


def _build_zero_leakage_icl_pairs(
    cache: ValClipCache,
    *,
    train_tasks: set,
    unseen_tasks: set,
    n_train_pairs: int = 4000,
    n_test_seen: int = 512,
    n_test_unseen: int = 512,
    match_mode: str = "visual_knn",
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build strictly isolated (train, test_seen_task, test_unseen_task) cross-layout ICL pairs.

    Hard isolation assertions:
      - Train pairs use ONLY `fold == 'train'` episodes from `train_tasks`.
      - Test Seen-Task pairs use ONLY `fold == 'val'` episodes from `train_tasks` (0 ep/layout overlap).
      - Test Unseen-Task pairs use ONLY `fold == 'val'` episodes from `unseen_tasks` (0 task/ep overlap).
      - Demo frame `t_d` is selected STRICTLY without Query GT action or Query future frames!
    """
    rng = np.random.default_rng(seed + 414)
    by_task_fold: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for ep_i, row in enumerate(cache.episodes):
        tid = cache.task_to_id.get(row["task"])
        if tid is not None:
            by_task_fold[(tid, row["fold"])].append(ep_i)

    # Cache stride-16 anchor visual keys h_mem per episode lazily
    ep_anchor_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _get_demo_anchors(ep_i: int) -> Tuple[np.ndarray, np.ndarray]:
        if ep_i not in ep_anchor_cache:
            T = int(cache.ep_lengths_np[ep_i])
            off = int(cache.frame_offsets_np[ep_i])
            anchors = compute_behavior_anchor_steps(T, stride=16)
            h_mem = np.stack(
                [_extract_h_obs(cache.frame_feats_mmap, off, T, int(a)) for a in anchors],
                axis=0,
            ).reshape(len(anchors), -1)  # (M, 3072)
            norms = np.clip(np.linalg.norm(h_mem, axis=-1, keepdims=True), 1e-6, None)
            ep_anchor_cache[ep_i] = (anchors, h_mem / norms)
        return ep_anchor_cache[ep_i]

    def _sample_split(
        task_set: set, fold: str, target_n: int
    ) -> Dict[str, np.ndarray]:
        valid_tasks = sorted(
            t for t in task_set if len(by_task_fold[(t, fold)]) >= 2
        )
        d_eps, d_frs, q_eps, q_frs, t_ids = [], [], [], [], []
        for _ in range(target_n):
            tid = int(valid_tasks[int(rng.integers(len(valid_tasks)))])
            ep_pool = by_task_fold[(tid, fold)]
            # Sample 2 distinct episodes (distinct layouts!)
            i_pair = rng.choice(len(ep_pool), size=2, replace=False)
            d_ep, q_ep = int(ep_pool[i_pair[0]]), int(ep_pool[i_pair[1]])
            T_q = int(cache.ep_lengths_np[q_ep])
            T_d = int(cache.ep_lengths_np[d_ep])
            # Sample non-tail query step t_q in [0, T_q - 16]
            max_q = max(1, T_q - LOCAL_WINDOW_LEN)
            q_fr = int(rng.integers(0, max_q))

            anchors_d, h_mem_u = _get_demo_anchors(d_ep)
            if match_mode == "visual_knn":
                # Pure visual observation key matching (ZERO GT action, ZERO query future!)
                q_off = int(cache.frame_offsets_np[q_ep])
                hq = _extract_h_obs(cache.frame_feats_mmap, q_off, T_q, q_fr).reshape(-1)
                hq_u = hq / max(float(np.linalg.norm(hq)), 1e-6)
                best_k = int(np.argmax(h_mem_u @ hq_u))
                d_fr = int(anchors_d[best_k])
            elif match_mode == "rel_progress":
                prog = float(q_fr) / max(T_q - 1, 1)
                d_progs = anchors_d.astype(np.float32) / max(T_d - 1, 1)
                best_k = int(np.argmin(np.abs(d_progs - prog)))
                d_fr = int(anchors_d[best_k])
            else:
                raise ValueError(match_mode)

            d_eps.append(d_ep)
            d_frs.append(d_fr)
            q_eps.append(q_ep)
            q_frs.append(q_fr)
            t_ids.append(tid)

        return {
            "d_ep": np.array(d_eps, dtype=np.int64),
            "d_fr": np.array(d_frs, dtype=np.int64),
            "q_ep": np.array(q_eps, dtype=np.int64),
            "q_fr": np.array(q_frs, dtype=np.int64),
            "task_id": np.array(t_ids, dtype=np.int64),
        }

    splits = {
        "train": _sample_split(train_tasks, "train", n_train_pairs),
        "test_seen": _sample_split(train_tasks, "val", n_test_seen),
        "test_unseen": _sample_split(unseen_tasks, "val", n_test_unseen),
    }

    # Strict zero-leakage assertions!
    tr_eps = set(splits["train"]["d_ep"].tolist()) | set(splits["train"]["q_ep"].tolist())
    ts_eps = set(splits["test_seen"]["d_ep"].tolist()) | set(splits["test_seen"]["q_ep"].tolist())
    tu_eps = set(splits["test_unseen"]["d_ep"].tolist()) | set(splits["test_unseen"]["q_ep"].tolist())
    assert len(tr_eps & ts_eps) == 0, "Episode leakage between train and test_seen!"
    assert len(tr_eps & tu_eps) == 0, "Episode leakage between train and test_unseen!"
    assert len(set(splits["train"]["task_id"].tolist()) & set(splits["test_unseen"]["task_id"].tolist())) == 0, "Task leakage!"
    return splits


@torch.inference_mode()
def _gather_and_encode_split_tensors(
    cache: ValClipCache,
    split: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    """Gather raw CLIP clips, masks, GT actions, and initial obs `h_d, h_q` once per split."""
    d_clip, d_mask, d_act10, h_d = cache.gather(split["d_ep"], split["d_fr"])
    q_clip, q_mask, q_act10, h_q = cache.gather(split["q_ep"], split["q_fr"])
    return {
        "d_clip": d_clip,
        "d_mask": d_mask,
        "d_act10": d_act10,
        "h_d": h_d.reshape(len(d_clip), -1),  # (N, 3072)
        "q_clip": q_clip,
        "q_mask": q_mask,
        "q_act10": q_act10,
        "h_q": h_q.reshape(len(q_clip), -1),  # (N, 3072)
    }


@torch.inference_mode()
def _encode_clips_with_policy(
    policy, clips: torch.Tensor, masks: torch.Tensor, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    N = len(clips)
    out = np.zeros((N, 3072), dtype=np.float32)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        with amp_ctx:
            b = policy.encode_local_behavior(
                clips[s:e].to(device, non_blocking=True),
                masks[s:e].to(device, non_blocking=True),
            )
        out[s:e] = b.float().reshape(e - s, -1).cpu().numpy()
    return out


def _build_pca_features(
    h_d_tr: np.ndarray,
    h_q_tr: np.ndarray,
    z_d_tr: np.ndarray,
    h_d_evals: List[np.ndarray],
    h_q_evals: List[np.ndarray],
    z_d_evals: List[np.ndarray],
    device: torch.device,
    *,
    k_obs: int = 128,
    k_lat: int = 64,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Project `(h_q - h_d, h_q, z_d)` with PCA fitted STRICTLY on the N training pairs."""
    def _pca_proj(X_tr: np.ndarray, X_list: List[np.ndarray], k: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
        mu = Xt.mean(dim=0, keepdim=True)
        Xc = Xt - mu
        k_eff = min(k, Xc.shape[0], Xc.shape[1])
        # Fast economy SVD
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        V = Vh[:k_eff].T  # (D, k_eff)
        proj_tr = (Xc @ V).cpu().numpy()
        proj_list = [
            ((torch.from_numpy(Xe.astype(np.float32)).to(device) - mu) @ V).cpu().numpy()
            for Xe in X_list
        ]
        return proj_tr, proj_list

    dh_tr = h_q_tr - h_d_tr
    dh_evals = [hq - hd for hq, hd in zip(h_q_evals, h_d_evals)]

    p_dh_tr, p_dh_ev = _pca_proj(dh_tr, dh_evals, k_obs)
    p_hq_tr, p_hq_ev = _pca_proj(h_q_tr, h_q_evals, k_obs)
    p_zd_tr, p_zd_ev = _pca_proj(z_d_tr, z_d_evals, k_lat)

    X_tr = np.concatenate([p_dh_tr, p_hq_tr, p_zd_tr], axis=-1)
    X_ev = [np.concatenate([a, b, c], axis=-1) for a, b, c in zip(p_dh_ev, p_hq_ev, p_zd_ev)]
    return X_tr, X_ev


def _train_residual_mlp(
    X_tr: np.ndarray,
    dY_tr: np.ndarray,
    X_evals: List[np.ndarray],
    device: torch.device,
    *,
    hidden_dim: int = 256,
    steps: int = 400,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    seed: int = 42,
) -> List[np.ndarray]:
    """Train a lightweight 2-layer residual MLP predicting `Δz = z_q - z_d` from obs+z_d."""
    torch.manual_seed(seed)
    X_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(dY_tr.astype(np.float32)).to(device)
    xm = X_t.mean(dim=0, keepdim=True)
    xs = X_t.std(dim=0, keepdim=True).clamp_min(1e-4)
    ym = Y_t.mean(dim=0, keepdim=True)
    Xn = (X_t - xm) / xs
    Yn = Y_t - ym

    net = nn.Sequential(
        nn.Linear(Xn.shape[1], hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(hidden_dim, Yn.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    n = Xn.shape[0]
    bs = min(128, n)
    gen = torch.Generator(device=device).manual_seed(seed)
    net.train()
    for _ in range(steps):
        if n > bs:
            idx = torch.randint(0, n, (bs,), generator=gen, device=device)
            xb, yb = Xn[idx], Yn[idx]
        else:
            xb, yb = Xn, Yn
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(net(xb), yb)
        loss.backward()
        opt.step()
        sched.step()

    net.eval()
    preds = []
    with torch.inference_mode():
        for Xe in X_evals:
            Xen = (torch.from_numpy(Xe.astype(np.float32)).to(device) - xm) / xs
            preds.append((net(Xen) + ym).cpu().numpy().astype(np.float32))
    return preds


def run_exp4(
    out_dir: Path,
    extra_models: Optional[Dict[str, Path]] = None,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
    budgets: Tuple[int, ...] = (50, 100, 250, 500, 1000, 2000, 4000),
    match_modes: Tuple[str, ...] = ("visual_knn", "rel_progress"),
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=42)
    ae = train_or_load_action_ae(cache, OUTPUT_ROOT / "exp2_action_baselines", device)

    model_map = dict(CHECKPOINTS)
    if extra_models:
        model_map.update(extra_models)
    model_keys = [k for k, p in model_map.items() if p.is_file()]
    policies = {k: load_frozen_e9_stage1_teacher(model_map[k], device=device) for k in model_keys}

    unique_tasks = np.arange(len(cache.task_to_id))
    all_mode_results: Dict[str, Any] = {}

    for match_mode in match_modes:
        per_seed_res: List[Dict[str, Any]] = []
        for seed in seeds:
            rng = np.random.default_rng(seed + 444)
            perm_t = rng.permutation(unique_tasks)
            n_tr_t = int(0.7 * len(perm_t))
            tr_tasks = set(perm_t[:n_tr_t].tolist())
            un_tasks = set(perm_t[n_tr_t:].tolist())

            splits = _build_zero_leakage_icl_pairs(
                cache,
                train_tasks=tr_tasks,
                unseen_tasks=un_tasks,
                n_train_pairs=max(budgets),
                n_test_seen=512,
                n_test_unseen=512,
                match_mode=match_mode,
                seed=seed,
            )
            raw_t = {s_name: _gather_and_encode_split_tensors(cache, sp) for s_name, sp in splits.items()}

            gt_seen = raw_t["test_seen"]["q_act10"].numpy()
            gt_unseen = raw_t["test_unseen"]["q_act10"].numpy()
            hd_tr = raw_t["train"]["h_d"].numpy()
            hq_tr = raw_t["train"]["h_q"].numpy()
            hd_ev = [raw_t["test_seen"]["h_d"].numpy(), raw_t["test_unseen"]["h_d"].numpy()]
            hq_ev = [raw_t["test_seen"]["h_q"].numpy(), raw_t["test_unseen"]["h_q"].numpy()]

            seed_out: Dict[str, Any] = {}

            # All representation candidates: neural video models + raw_action_10d + action_ae
            candidates = model_keys + ["raw_action_10d", "action_ae"]
            for cname in candidates:
                if cname == "raw_action_10d":
                    z_tr_d = raw_t["train"]["d_act10"].numpy().reshape(len(hd_tr), -1)
                    z_tr_q = raw_t["train"]["q_act10"].numpy().reshape(len(hd_tr), -1)
                    z_ev_d = [
                        raw_t["test_seen"]["d_act10"].numpy().reshape(len(gt_seen), -1),
                        raw_t["test_unseen"]["d_act10"].numpy().reshape(len(gt_unseen), -1),
                    ]
                    z_ev_q = [
                        gt_seen.reshape(len(gt_seen), -1),
                        gt_unseen.reshape(len(gt_unseen), -1),
                    ]

                    def _decode_fn(z_arr: np.ndarray) -> np.ndarray:
                        return z_arr.reshape(-1, ACTION_HORIZON, ACTION_DIM_10D)

                elif cname == "action_ae":
                    with torch.inference_mode():
                        z_tr_d = ae.encode(raw_t["train"]["d_act10"].to(device)).reshape(len(hd_tr), -1).cpu().numpy()
                        z_tr_q = ae.encode(raw_t["train"]["q_act10"].to(device)).reshape(len(hd_tr), -1).cpu().numpy()
                        z_ev_d = [
                            ae.encode(raw_t["test_seen"]["d_act10"].to(device)).reshape(len(gt_seen), -1).cpu().numpy(),
                            ae.encode(raw_t["test_unseen"]["d_act10"].to(device)).reshape(len(gt_unseen), -1).cpu().numpy(),
                        ]
                        z_ev_q = [
                            ae.encode(raw_t["test_seen"]["q_act10"].to(device)).reshape(len(gt_seen), -1).cpu().numpy(),
                            ae.encode(raw_t["test_unseen"]["q_act10"].to(device)).reshape(len(gt_unseen), -1).cpu().numpy(),
                        ]

                    def _decode_fn(z_arr: np.ndarray) -> np.ndarray:
                        with torch.inference_mode():
                            return ae.decode(torch.from_numpy(z_arr).to(device)).cpu().numpy()

                else:
                    pol = policies[cname]
                    z_tr_d = _encode_clips_with_policy(pol, raw_t["train"]["d_clip"], raw_t["train"]["d_mask"], device)
                    z_tr_q = _encode_clips_with_policy(pol, raw_t["train"]["q_clip"], raw_t["train"]["q_mask"], device)
                    z_ev_d = [
                        _encode_clips_with_policy(pol, raw_t["test_seen"]["d_clip"], raw_t["test_seen"]["d_mask"], device),
                        _encode_clips_with_policy(pol, raw_t["test_unseen"]["d_clip"], raw_t["test_unseen"]["d_mask"], device),
                    ]
                    z_ev_q = [
                        _encode_clips_with_policy(pol, raw_t["test_seen"]["q_clip"], raw_t["test_seen"]["q_mask"], device),
                        _encode_clips_with_policy(pol, raw_t["test_unseen"]["q_clip"], raw_t["test_unseen"]["q_mask"], device),
                    ]

                    def _decode_fn(z_arr: np.ndarray, _p=pol) -> np.ndarray:
                        return ddim_decode_batched(_p, z_arr, device, seed=12345)

                # Direct Copy (N=0) and Oracle upper bound on both test splits
                dc_seen = _action_metrics_dict(_decode_fn(z_ev_d[0]), gt_seen)
                dc_unseen = _action_metrics_dict(_decode_fn(z_ev_d[1]), gt_unseen)
                ora_seen = _action_metrics_dict(_decode_fn(z_ev_q[0]), gt_seen)
                ora_unseen = _action_metrics_dict(_decode_fn(z_ev_q[1]), gt_unseen)

                c_res: Dict[str, Any] = {
                    "direct_copy": {"seen": dc_seen, "unseen": dc_unseen},
                    "oracle": {"seen": ora_seen, "unseen": ora_unseen},
                    "by_n": {},
                }

                dz_tr_all = z_tr_q - z_tr_d
                for n_b in budgets:
                    n_eff = min(int(n_b), len(hd_tr))
                    X_tr_f, X_ev_f = _build_pca_features(
                        hd_tr[:n_eff], hq_tr[:n_eff], z_tr_d[:n_eff], hd_ev, hq_ev, z_ev_d, device
                    )
                    # 1. Linear Adapter (Residual Ridge)
                    dz_lin_seen, alpha_l = _ridge_fit_predict(
                        X_tr_f, dz_tr_all[:n_eff], X_ev_f[0], alphas=(1.0, 10.0, 100.0, 500.0, 2000.0, 10000.0)
                    )
                    dz_lin_unseen, _ = _ridge_fit_predict(
                        X_tr_f, dz_tr_all[:n_eff], X_ev_f[1], alphas=(alpha_l,)
                    )
                    m_lin_s = _action_metrics_dict(_decode_fn(z_ev_d[0] + dz_lin_seen), gt_seen)
                    m_lin_u = _action_metrics_dict(_decode_fn(z_ev_d[1] + dz_lin_unseen), gt_unseen)

                    # 2. Lightweight 2-Layer MLP Adapter
                    dz_mlp_s, dz_mlp_u = _train_residual_mlp(
                        X_tr_f, dz_tr_all[:n_eff], X_ev_f, device, seed=seed + n_eff
                    )
                    m_mlp_s = _action_metrics_dict(_decode_fn(z_ev_d[0] + dz_mlp_s), gt_seen)
                    m_mlp_u = _action_metrics_dict(_decode_fn(z_ev_d[1] + dz_mlp_u), gt_unseen)

                    def _pack(m_d: Dict[str, float], dc_d: Dict[str, float], ora_d: Dict[str, float]) -> Dict[str, float]:
                        return {
                            "mse10": m_d["action10_mse"],
                            "pos_mse": m_d["pos_mse"],
                            "rot6d_mse": m_d["rot6d_mse"],
                            "grip_mse": m_d["grip_mse"],
                            "mae7": m_d["action7_mae"],
                            "grip_acc": m_d["grip_sign_acc"],
                            "arr_10d": float(
                                (dc_d["action10_mse"] - m_d["action10_mse"])
                                / max(dc_d["action10_mse"] - ora_d["action10_mse"], 1e-9)
                            ),
                        }

                    c_res["by_n"][str(n_eff)] = {
                        "linear": {
                            "seen": _pack(m_lin_s, dc_seen, ora_seen),
                            "unseen": _pack(m_lin_u, dc_unseen, ora_unseen),
                        },
                        "mlp": {
                            "seen": _pack(m_mlp_s, dc_seen, ora_seen),
                            "unseen": _pack(m_mlp_u, dc_unseen, ora_unseen),
                        },
                    }
                seed_out[cname] = c_res
            per_seed_res.append(seed_out)

        # Aggregate across seeds
        agg: Dict[str, Any] = {}
        for cname in per_seed_res[0]:
            agg[cname] = {"direct_copy": {}, "oracle": {}, "by_n": {}}
            for sp_name in ("seen", "unseen"):
                for tag in ("direct_copy", "oracle"):
                    vals = [d[cname][tag][sp_name]["action10_mse"] for d in per_seed_res]
                    gvals = [d[cname][tag][sp_name]["grip_sign_acc"] for d in per_seed_res]
                    agg[cname][tag][sp_name] = {
                        "mse10_mean": float(np.mean(vals)),
                        "mse10_std": float(np.std(vals)),
                        "grip_acc_mean": float(np.mean(gvals)),
                    }
            for n_str in per_seed_res[0][cname]["by_n"]:
                agg[cname]["by_n"][n_str] = {}
                for adp in ("linear", "mlp"):
                    agg[cname]["by_n"][n_str][adp] = {}
                    for sp_name in ("seen", "unseen"):
                        block = [d[cname]["by_n"][n_str][adp][sp_name] for d in per_seed_res]
                        agg[cname]["by_n"][n_str][adp][sp_name] = {
                            f"{m}_mean": float(np.mean([b[m] for b in block]))
                            for m in ("mse10", "pos_mse", "grip_mse", "mae7", "grip_acc", "arr_10d")
                        }
                        agg[cname]["by_n"][n_str][adp][sp_name]["mse10_std"] = float(
                            np.std([b["mse10"] for b in block])
                        )
                        agg[cname]["by_n"][n_str][adp][sp_name]["arr_10d_std"] = float(
                            np.std([b["arr_10d"] for b in block])
                        )
        all_mode_results[match_mode] = agg

        print(f"\n=== [E13 Exp 4 | mode={match_mode}] Minimal Zero-Leakage ICL Transfer (Unseen Tasks, 3 seeds) ===", flush=True)
        for cname, cv in agg.items():
            dc = cv["direct_copy"]["unseen"]["mse10_mean"]
            ora = cv["oracle"]["unseen"]["mse10_mean"]
            l100 = cv["by_n"]["100"]["linear"]["unseen"]
            l1000 = cv["by_n"]["1000"]["linear"]["unseen"]
            l4000 = cv["by_n"]["4000"]["linear"]["unseen"]
            m1000 = cv["by_n"]["1000"]["mlp"]["unseen"]
            m4000 = cv["by_n"]["4000"]["mlp"]["unseen"]
            print(
                f"  {cname:14s} | DC={dc:.4f} Ora={ora:.4f} | "
                f"Lin@100={l100['mse10_mean']:.4f}({l100['arr_10d_mean']*100:.1f}%) "
                f"Lin@1k={l1000['mse10_mean']:.4f}({l1000['arr_10d_mean']*100:.1f}%) "
                f"Lin@4k={l4000['mse10_mean']:.4f}({l4000['arr_10d_mean']*100:.1f}%) | "
                f"MLP@1k={m1000['mse10_mean']:.4f}({m1000['arr_10d_mean']*100:.1f}%) "
                f"MLP@4k={m4000['mse10_mean']:.4f}({m4000['arr_10d_mean']*100:.1f}%)",
                flush=True,
            )

    summary = {
        "seeds": list(seeds),
        "budgets": list(budgets),
        "match_modes": list(match_modes),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "results": all_mode_results,
    }
    atomic_write_json(summary, out_dir / "exp4_summary.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp4_minimal_icl")
    args = p.parse_args()
    run_exp4(args.out_dir)


if __name__ == "__main__":
    main()
