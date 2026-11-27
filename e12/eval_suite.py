"""Unified 5-pillar evaluation suite for E9 and E12 behavior latent spaces.

Evaluates any `E9Stage1LocalPolicy` checkpoint (baseline E9 or E12-A/B/C) across:
  1. Action Reconstruction Fidelity (val diff loss, DDIM 10D/7D errors, timestep distinctiveness)
  2. Action-Latent Relational Alignment & Cross-Scene kNN Consistency (all 6 attributes,
     stratified by same-ep / same-task / cross-task, partial-attribute matching, scene sensitivity)
  3. Latent Displacement Consistency & DDIM-Decoded Latent Arithmetic Transfer (Tier 1 & Tier 2)
  4. Unseen Behavior Combination Generalization (held-out dir x grip arithmetic & linear transfer)
  5. Transfer Efficiency (cross-task linear probing R^2, Linear vs MLP sample-efficiency curves,
     cross-layout residual effective rank & visual-query transfer efficiency)

Memory-efficient: reads `frame_features.npy` via `mmap_mode='r'` so only validation frames
are loaded (<1s startup, <2GB RAM).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import ActionNormalizer, action_10d_to_7d
from e5.train import _action_error_sums, _unpack_action_errors, atomic_write_json
from e6.data import SelfPairFeatureBank
from e9 import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    DEFAULT_DENSE_CACHE_DIR,
    DEFAULT_STAGE1_LOCAL_CKPT,
    E5_CACHE_DIR,
    LOCAL_WINDOW_LEN,
)
from e9.stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher

from . import P0_BASES_PATH, SCENE_SENSITIVITY_LATENTS
from .action_attributes import (
    compute_attribute_similarity_matrix,
    discretize_behavior_factors,
    extract_action_attribute_vectors,
    extract_action_delta_features,
    pairwise_cosine_sim,
    pairwise_l2_dist,
)


def _spearman_pearson_from_vectors(
    x: torch.Tensor, y: torch.Tensor, max_pairs: int = 250_000, seed: int = 42
) -> Tuple[float, float]:
    """Compute (spearman_rho, pearson_r) between two 1D tensors, subsampling if huge."""
    n = int(x.numel())
    if n < 3:
        return float("nan"), float("nan")
    if n > max_pairs:
        gen = torch.Generator(device=x.device).manual_seed(seed)
        idx = torch.randperm(n, generator=gen, device=x.device)[:max_pairs]
        x = x[idx]
        y = y[idx]
    xf = x.float()
    yf = y.float()
    xm = xf - xf.mean()
    ym = yf - yf.mean()
    denom = (xm.norm() * ym.norm()).clamp_min(1e-12)
    pearson = float(((xm * ym).sum() / denom).item())

    # Spearman via rank transform
    rx = torch.empty_like(xf)
    ry = torch.empty_like(yf)
    rx[torch.argsort(xf)] = torch.arange(len(xf), device=xf.device, dtype=torch.float32)
    ry[torch.argsort(yf)] = torch.arange(len(yf), device=yf.device, dtype=torch.float32)
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    rdenom = (rxm.norm() * rym.norm()).clamp_min(1e-12)
    spearman = float(((rxm * rym).sum() / rdenom).item())
    return spearman, pearson


def _effective_rank_stats(mat: np.ndarray, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Compute effective rank (entropy-based), participation ratio, and d_50/80/90/95."""
    dev = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    x = torch.from_numpy(np.asarray(mat, dtype=np.float32)).to(dev)
    x_c = x - x.mean(dim=0, keepdim=True)
    rms_c = float(torch.sqrt((x_c ** 2).mean()).item())
    s = torch.linalg.svdvals(x_c).double().cpu().numpy()
    e = s ** 2
    tot = float(e.sum())
    if tot <= 1e-12:
        return {"erank_energy": 0.0, "pr": 0.0, "d_50": 0, "d_80": 0, "d_90": 0, "d_95": 0, "rms_centered": rms_c}
    p = e / tot
    p_pos = p[p > 1e-15]
    erank = float(np.exp(-(p_pos * np.log(p_pos)).sum()))
    pr = float((e.sum() ** 2) / max(float((e ** 2).sum()), 1e-15))
    cum = np.cumsum(p)
    return {
        "erank_energy": erank,
        "pr": pr,
        "d_50": int(np.searchsorted(cum, 0.50) + 1),
        "d_80": int(np.searchsorted(cum, 0.80) + 1),
        "d_90": int(np.searchsorted(cum, 0.90) + 1),
        "d_95": int(np.searchsorted(cum, 0.95) + 1),
        "rms_centered": rms_c,
    }


def _solve_ridge_gpu(
    Xc: torch.Tensor, Yc: torch.Tensor, Xte_c: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Fast primal/dual ridge solve in float64 on GPU/CPU."""
    n, d = Xc.shape
    if n >= d:
        G = Xc.T @ Xc
        G.diagonal().add_(alpha)
        W = torch.linalg.solve(G, Xc.T @ Yc)
        return Xte_c @ W
    else:
        K = Xc @ Xc.T
        K.diagonal().add_(alpha)
        A = torch.linalg.solve(K, Yc)
        return (Xte_c @ Xc.T) @ A


def _ridge_fit_predict(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, float]:
    """Fast GPU-accelerated closed-form Ridge regression with internal holdout alpha selection."""
    dev = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float64)).to(dev)
    Y_tr_t = torch.from_numpy(np.asarray(Y_tr, dtype=np.float64)).to(dev)
    X_te_t = torch.from_numpy(np.asarray(X_te, dtype=np.float64)).to(dev)

    x_mean = X_tr_t.mean(dim=0, keepdim=True)
    y_mean = Y_tr_t.mean(dim=0, keepdim=True)
    Xc = X_tr_t - x_mean
    Yc = Y_tr_t - y_mean
    Xte_c = X_te_t - x_mean

    n, d = Xc.shape
    if n >= 30 and len(alphas) > 1:
        n_val = max(5, int(0.2 * n))
        X_a, Y_a = Xc[:-n_val], Yc[:-n_val]
        X_v, Y_v = Xc[-n_val:], Yc[-n_val:]
        # Precompute Gram once for all alphas!
        use_primal = X_a.shape[0] >= d
        if use_primal:
            G_base = X_a.T @ X_a
            XtY = X_a.T @ Y_a
        else:
            K_base = X_a @ X_a.T
            K_v = X_v @ X_a.T
        best_alpha = alphas[0]
        best_err = float("inf")
        for alpha in alphas:
            if use_primal:
                G = G_base.clone()
                G.diagonal().add_(alpha)
                pred_v = X_v @ torch.linalg.solve(G, XtY)
            else:
                K = K_base.clone()
                K.diagonal().add_(alpha)
                pred_v = K_v @ torch.linalg.solve(K, Y_a)
            err = float(((pred_v - Y_v) ** 2).mean().item())
            if err < best_err:
                best_err = err
                best_alpha = alpha
    else:
        best_alpha = alphas[min(2, len(alphas) - 1)]

    pred = _solve_ridge_gpu(Xc, Yc, Xte_c, best_alpha) + y_mean
    return pred.float().cpu().numpy(), float(best_alpha)


def _train_small_mlp_predict(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    device: torch.device,
    hidden_dim: int = 256,
    steps: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """Train a lightweight 2-layer MLP adapter for transfer-efficiency comparison."""
    torch.manual_seed(seed)
    X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32)).to(device)
    Y_tr_t = torch.from_numpy(np.asarray(Y_tr, dtype=np.float32)).to(device)
    X_te_t = torch.from_numpy(np.asarray(X_te, dtype=np.float32)).to(device)

    x_mean = X_tr_t.mean(dim=0, keepdim=True)
    x_std = X_tr_t.std(dim=0, keepdim=True).clamp_min(1e-4)
    y_mean = Y_tr_t.mean(dim=0, keepdim=True)

    Xn_tr = (X_tr_t - x_mean) / x_std
    Yn_tr = Y_tr_t - y_mean
    Xn_te = (X_te_t - x_mean) / x_std

    in_dim = Xn_tr.shape[1]
    out_dim = Yn_tr.shape[1]
    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xn_tr.shape[0]
    bs = min(128, n)
    net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        if n > bs:
            idx = torch.randint(0, n, (bs,), generator=gen, device=device)
            xb, yb = Xn_tr[idx], Yn_tr[idx]
        else:
            xb, yb = Xn_tr, Yn_tr
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(net(xb), yb)
        loss.backward()
        opt.step()

    net.eval()
    with torch.inference_mode():
        pred = net(Xn_te) + y_mean
    return pred.cpu().numpy().astype(np.float32)


class ValClipCache:
    """Fast memory-mapped validation data cache for E9/E12 evaluation (<1s load)."""

    def __init__(
        self,
        e5_cache_dir: Path = E5_CACHE_DIR,
        dense_cache_dir: Path = DEFAULT_DENSE_CACHE_DIR,
        seed: int = 42,
    ):
        self.e5_cache_dir = Path(e5_cache_dir)
        self.dense_cache_dir = Path(dense_cache_dir)
        self.seed = int(seed)

        self.episodes = json.loads((self.e5_cache_dir / "episodes.json").read_text())
        self.frame_offsets_np = np.load(
            self.dense_cache_dir / "frame_offsets.npy", allow_pickle=False
        )
        self.ep_lengths_np = np.load(
            self.dense_cache_dir / "ep_lengths.npy", allow_pickle=False
        ).astype(np.int32)
        self.actions_10d_mmap = np.load(
            self.e5_cache_dir / "actions_10d.npy", mmap_mode="r", allow_pickle=False
        )
        self.frame_feats_mmap = np.load(
            self.dense_cache_dir / "frame_features.npy", mmap_mode="r", allow_pickle=False
        )
        self.query_ep_offsets_np = np.load(
            self.e5_cache_dir / "query_ep_offsets.npy", allow_pickle=False
        )
        self.query_frame_nums_np = np.load(
            self.e5_cache_dir / "query_frame_nums.npy", allow_pickle=False
        )
        norm_state = torch.load(
            self.e5_cache_dir / "action_normalizer.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.action_normalizer = ActionNormalizer(
            scale=norm_state["scale"], offset=norm_state["offset"]
        )

        # Reconstruct exact E6/E9 6,248 validation starts
        ep_list, fr_list = [], []
        for ep_idx, row in enumerate(self.episodes):
            if row["fold"] != "val":
                continue
            lo, hi = int(self.query_ep_offsets_np[ep_idx]), int(
                self.query_ep_offsets_np[ep_idx + 1]
            )
            ep_frames = self.query_frame_nums_np[lo:hi]
            if len(ep_frames) == 0:
                continue
            prev_f = np.maximum(ep_frames - 1, 0)
            p0 = np.searchsorted(ep_frames, prev_f)
            valid = (p0 < len(ep_frames)) & (
                ep_frames[np.minimum(p0, len(ep_frames) - 1)] == prev_f
            )
            if not np.any(valid):
                continue
            v_idx = np.flatnonzero(valid)
            ep_list.append(np.full(len(v_idx), ep_idx, dtype=np.int64))
            fr_list.append(ep_frames[v_idx].astype(np.int64))

        self.val_episodes = np.concatenate(ep_list, axis=0)
        self.val_frames = np.concatenate(fr_list, axis=0)
        self.val_wrong_episodes, self.val_wrong_frames = self._build_wrong_clips()
        self.val_timestep_pairs = SelfPairFeatureBank._build_same_episode_timestep_pairs(
            self.val_episodes, self.val_frames, min_frame_gap=20
        )

        # Task metadata per val start
        tasks_sorted = sorted({e["task"] for e in self.episodes if e["fold"] == "val"})
        self.task_to_id = {t: i for i, t in enumerate(tasks_sorted)}
        self.val_tasks = np.array(
            [self.episodes[int(e)]["task"] for e in self.val_episodes], dtype=object
        )
        self.val_task_ids = np.array(
            [self.task_to_id[t] for t in self.val_tasks], dtype=np.int64
        )
        self.val_layouts = np.array(
            [int(self.episodes[int(e)]["layout"]) for e in self.val_episodes],
            dtype=np.int64,
        )
        self.val_rel_progress = np.array(
            [
                float(f) / max(int(self.ep_lengths_np[int(e)]) - 1, 1)
                for e, f in zip(self.val_episodes, self.val_frames)
            ],
            dtype=np.float32,
        )

    def _build_wrong_clips(self) -> Tuple[np.ndarray, np.ndarray]:
        val_eps_by_task: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.episodes):
            if e["fold"] == "val":
                val_eps_by_task[e["task"]].append(i)
        tasks = sorted(val_eps_by_task.keys())
        task_to_idx = {t: i for i, t in enumerate(tasks)}
        rng = np.random.default_rng(self.seed + 99991)
        wrong_ep = np.empty(len(self.val_episodes), dtype=np.int64)
        wrong_fr = np.empty(len(self.val_episodes), dtype=np.int64)
        for idx, (ep, fr) in enumerate(zip(self.val_episodes, self.val_frames)):
            ep_int = int(ep)
            task = self.episodes[ep_int]["task"]
            t_i = task_to_idx[task]
            other_t = tasks[
                (t_i + 1 + int(rng.integers(len(tasks) - 1))) % len(tasks)
            ]
            cands = val_eps_by_task[other_t]
            w_ep = cands[int(rng.integers(len(cands)))]
            wrong_ep[idx] = w_ep
            w_len = int(self.ep_lengths_np[w_ep])
            wrong_fr[idx] = min(int(fr), max(w_len - 1, 0))
        return wrong_ep, wrong_fr

    def gather(
        self,
        episodes: np.ndarray,
        frames: np.ndarray,
        *,
        action_episodes: Optional[np.ndarray] = None,
        action_frames: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather `(clip_rgb, clip_pad_mask, actions_10d, h_query)` on CPU float32."""
        B = len(episodes)
        clips = np.zeros((B, LOCAL_WINDOW_LEN, 2, 768), dtype=np.float32)
        masks = np.zeros((B, LOCAL_WINDOW_LEN), dtype=bool)
        acts = np.zeros((B, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
        h_q = np.zeros((B, 4, 768), dtype=np.float32)

        a_eps = episodes if action_episodes is None else action_episodes
        a_frs = frames if action_frames is None else action_frames

        for i in range(B):
            ep = int(episodes[i])
            fr = int(frames[i])
            off = int(self.frame_offsets_np[ep])
            T = int(self.ep_lengths_np[ep])
            raw = fr + np.arange(LOCAL_WINDOW_LEN)
            pad = raw >= T
            clamped = np.minimum(raw, T - 1)
            f_slice = np.asarray(self.frame_feats_mmap[off + clamped], dtype=np.float32)
            if pad.any():
                f_slice[pad] = 0.0
            clips[i] = f_slice
            masks[i] = pad

            # h_query: [prev_frame_av, prev_frame_eh, cur_frame_av, cur_frame_eh]
            p0 = max(fr - 1, 0)
            f0 = np.asarray(self.frame_feats_mmap[off + p0], dtype=np.float32)
            f1 = np.asarray(self.frame_feats_mmap[off + min(fr, T - 1)], dtype=np.float32)
            h_q[i, 0:2] = f0
            h_q[i, 2:4] = f1

            a_ep = int(a_eps[i])
            a_fr = int(a_frs[i])
            a_off = int(self.frame_offsets_np[a_ep])
            a_T = int(self.ep_lengths_np[a_ep])
            a_clamped = np.minimum(a_fr + np.arange(ACTION_HORIZON), a_T - 1)
            acts[i] = np.asarray(self.actions_10d_mmap[a_off + a_clamped], dtype=np.float32)

        return (
            torch.from_numpy(clips),
            torch.from_numpy(masks),
            torch.from_numpy(acts),
            torch.from_numpy(h_q),
        )


@torch.inference_mode()
def encode_all_val_starts(
    policy: E9Stage1LocalPolicy,
    cache: ValClipCache,
    device: torch.device,
    *,
    batch_size: int = 256,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Encode all 6,248 val starts once, returning latents, CLIP baselines, actions, metadata."""
    policy.eval()
    N = len(cache.val_episodes)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    latents = np.zeros((N, 4, 768), dtype=np.float32)
    clip_mean = np.zeros((N, 2 * 768), dtype=np.float32)
    h_queries = np.zeros((N, 4, 768), dtype=np.float32)
    actions_10d = np.zeros((N, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
    masks_all = np.zeros((N, LOCAL_WINDOW_LEN), dtype=bool)

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        clip_cpu, mask_cpu, act_cpu, hq_cpu = cache.gather(
            cache.val_episodes[s:e], cache.val_frames[s:e]
        )
        clip_g = clip_cpu.to(device, non_blocking=True)
        mask_g = mask_cpu.to(device, non_blocking=True)
        with amp_ctx:
            b = policy.encode_local_behavior(clip_g, mask_g)
        latents[s:e] = b.float().cpu().numpy()

        # Valid-frame mean of dual-camera CLIP features as visual input baseline
        valid_w = (~mask_cpu).float().unsqueeze(-1).unsqueeze(-1)  # (B, 16, 1, 1)
        cm = (clip_cpu * valid_w).sum(dim=1) / valid_w.sum(dim=1).clamp_min(1.0)
        clip_mean[s:e] = cm.reshape(e - s, -1).numpy()
        h_queries[s:e] = hq_cpu.numpy()
        actions_10d[s:e] = act_cpu.numpy()
        masks_all[s:e] = mask_cpu.numpy()

    actions_7d = action_10d_to_7d(actions_10d)
    return {
        "latents": latents,  # (N, 4, 768)
        "latents_flat": latents.reshape(N, -1),  # (N, 3072)
        "clip_mean": clip_mean,  # (N, 1536)
        "h_queries": h_queries,  # (N, 4, 768)
        "actions_10d": actions_10d,  # (N, 16, 10)
        "actions_7d": actions_7d,  # (N, 16, 7)
        "masks": masks_all,  # (N, 16)
        "is_tail": masks_all.any(axis=1),
    }


@torch.inference_mode()
def ddim_decode_batched(
    policy: E9Stage1LocalPolicy,
    latents_3d_or_2d: np.ndarray | torch.Tensor,
    device: torch.device,
    *,
    seed: int = 12345,
    batch_size: int = 256,
    use_bf16: bool = True,
) -> np.ndarray:
    """Deterministic 16-step DDIM decoding from `(N, 4, 768)` or `(N, 3072)` latents."""
    policy.eval()
    arr = (
        latents_3d_or_2d.detach().cpu().numpy()
        if isinstance(latents_3d_or_2d, torch.Tensor)
        else np.asarray(latents_3d_or_2d, dtype=np.float32)
    )
    N = arr.shape[0]
    arr_3d = arr.reshape(N, 4, 768)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z0_all = torch.randn((N, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )
    preds = np.zeros((N, ACTION_HORIZON, ACTION_DIM_10D), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        b_g = torch.from_numpy(arr_3d[s:e]).to(device)
        z0 = z0_all[s:e].to(device)
        with amp_ctx:
            pred = policy.sample_from_z_self(b_g, init_noise=z0)
        preds[s:e] = pred.float().cpu().numpy()
    return preds


def _action_metrics_dict(pred_10d: np.ndarray, gt_10d: np.ndarray) -> Dict[str, float]:
    p_t = torch.from_numpy(np.asarray(pred_10d, dtype=np.float32))
    g_t = torch.from_numpy(np.asarray(gt_10d, dtype=np.float32))
    return _unpack_action_errors(_action_error_sums(p_t, g_t))


# ---------------------------------------------------------------------------
# Suite 1: Action Reconstruction Fidelity
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_suite1_reconstruction(
    policy: E9Stage1LocalPolicy,
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    *,
    ddim_samples: int = 2048,
    ts_pairs: int = 512,
    val_seed: int = 12345,
    batch_size: int = 256,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    policy.eval()
    N = len(cache.val_episodes)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    gen = torch.Generator(device="cpu").manual_seed(val_seed)
    all_noise = torch.randn((N, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32)
    all_pertub = torch.randn((N, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32)
    all_t = torch.randint(
        0, policy.noise_scheduler.config.num_train_timesteps, (N,), generator=gen, dtype=torch.long
    )

    self_diff_sum = 0.0
    wrong_diff_sum = 0.0
    b_sq_sum = 0.0

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        b_s = torch.from_numpy(encoded["latents"][s:e]).to(device)
        a_s = torch.from_numpy(encoded["actions_10d"][s:e]).to(device)
        c_w, m_w, _, _ = cache.gather(cache.val_wrong_episodes[s:e], cache.val_wrong_frames[s:e])
        c_w = c_w.to(device)
        m_w = m_w.to(device)
        noise = all_noise[s:e].to(device)
        pertub = all_pertub[s:e].to(device)
        t = all_t[s:e].to(device)
        with amp_ctx:
            ls = policy.diffusion_loss_from_z_self(b_s, a_s, noise=noise, pertub_noise=pertub, timesteps=t)
            b_w = policy.encode_local_behavior(c_w, m_w)
            lw = policy.diffusion_loss_from_z_self(b_w, a_s, noise=noise, pertub_noise=pertub, timesteps=t)
        self_diff_sum += float(ls.double().sum().item())
        wrong_diff_sum += float(lw.double().sum().item())
        b_sq_sum += float(b_s.float().square().mean(dim=(1, 2)).double().sum().item())

    # DDIM reconstruction on deterministic evenly spaced subset of val
    n_ddim = min(int(ddim_samples), N)
    ddim_idx = np.linspace(0, N - 1, n_ddim, dtype=np.int64)
    pred_self = ddim_decode_batched(
        policy, encoded["latents"][ddim_idx], device, seed=val_seed + 777, batch_size=batch_size, use_bf16=use_bf16
    )
    ddim_self = _action_metrics_dict(pred_self, encoded["actions_10d"][ddim_idx])

    # Timestep distinctiveness (same-episode distinct frames)
    pairs = cache.val_timestep_pairs[: min(int(ts_pairs), len(cache.val_timestep_pairs))]
    ia, ib = pairs[:, 0], pairs[:, 1]
    za = torch.from_numpy(encoded["latents_flat"][ia])
    zb = torch.from_numpy(encoded["latents_flat"][ib])
    z_cos = float(F.cosine_similarity(za, zb, dim=-1).mean().item())
    pred_a = ddim_decode_batched(
        policy, encoded["latents"][ia], device, seed=val_seed + 2026, batch_size=batch_size, use_bf16=use_bf16
    )
    pred_b = ddim_decode_batched(
        policy, encoded["latents"][ib], device, seed=val_seed + 2026, batch_size=batch_size, use_bf16=use_bf16
    )
    matched = _action_metrics_dict(
        np.concatenate([pred_a, pred_b], axis=0),
        np.concatenate([encoded["actions_10d"][ia], encoded["actions_10d"][ib]], axis=0),
    )
    mismatched = _action_metrics_dict(
        np.concatenate([pred_a, pred_b], axis=0),
        np.concatenate([encoded["actions_10d"][ib], encoded["actions_10d"][ia]], axis=0),
    )

    return {
        "val_samples": N,
        "val_self_diff_loss": self_diff_sum / N,
        "val_wrong_clip_diff_loss": wrong_diff_sum / N,
        "val_clip_sensitivity_diff_loss": (wrong_diff_sum - self_diff_sum) / N,
        "val_self_b_rms": math.sqrt(b_sq_sum / N),
        "ddim_samples": n_ddim,
        "ddim_self": ddim_self,
        "timestep_distinctiveness": {
            "num_pairs": int(len(pairs)),
            "latent_cosine_sim": z_cos,
            "matched_action10_mse": matched["action10_mse"],
            "mismatched_action10_mse": mismatched["action10_mse"],
            "mismatch_to_match_mse10_ratio": mismatched["action10_mse"]
            / max(matched["action10_mse"], 1e-12),
        },
    }


# ---------------------------------------------------------------------------
# Suite 2: Action-Latent Alignment, kNN Consistency, Partial-Attr & Scene Invariance
# ---------------------------------------------------------------------------
def _compute_knn_metrics(
    S_rep: torch.Tensor,
    S_act: torch.Tensor,
    valid_mask: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Compute kNN Recall@k (overlap between rep-kNN and act-kNN) under `valid_mask`."""
    neg_inf = -1e9
    Sr = S_rep.masked_fill(~valid_mask, neg_inf)
    Sa = S_act.masked_fill(~valid_mask, neg_inf)
    max_k = max(ks)
    _, top_r = torch.topk(Sr, k=max_k, dim=1)
    _, top_a = torch.topk(Sa, k=max_k, dim=1)

    res: Dict[str, float] = {}
    for k in ks:
        # Overlap between top_r[:, :k] and top_a[:, :k]
        # (N, k, 1) == (N, 1, k)
        matches = (top_r[:, :k, None] == top_a[:, None, :k]).any(dim=-1).float().mean(dim=-1)
        res[f"recall@{k}"] = float(matches.mean().item())
    return res


def evaluate_suite2_relational(
    policy: E9Stage1LocalPolicy,
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    *,
    max_rel_samples: int = 3500,
    seed: int = 42,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    # Select deterministic stratified non-tail subset covering all tasks
    non_tail_idx = np.flatnonzero(~encoded["is_tail"])
    rng = np.random.default_rng(seed + 101)
    if len(non_tail_idx) > max_rel_samples:
        sel = np.sort(rng.choice(non_tail_idx, size=max_rel_samples, replace=False))
    else:
        sel = non_tail_idx
    M = len(sel)

    z_4tok = torch.from_numpy(encoded["latents"][sel]).to(device)  # (M, 4, 768)
    z_flat = z_4tok.reshape(M, -1)  # (M, 3072)
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    clip_m = torch.from_numpy(encoded["clip_mean"][sel]).to(device)  # (M, 1536)
    a10 = torch.from_numpy(encoded["actions_10d"][sel]).to(device)
    a7 = torch.from_numpy(encoded["actions_7d"][sel]).to(device)

    attrs = extract_action_attribute_vectors(a10, a7)

    ep_t = torch.from_numpy(cache.val_episodes[sel]).to(device)
    fr_t = torch.from_numpy(cache.val_frames[sel]).to(device)
    task_t = torch.from_numpy(cache.val_task_ids[sel]).to(device)

    # Pair masks (upper triangle, i < j)
    triu = torch.triu(torch.ones((M, M), device=device, dtype=torch.bool), diagonal=1)
    same_ep = (ep_t[:, None] == ep_t[None, :]) & (torch.abs(fr_t[:, None] - fr_t[None, :]) >= 16)
    same_task_diff_ep = (task_t[:, None] == task_t[None, :]) & (ep_t[:, None] != ep_t[None, :])
    cross_task = task_t[:, None] != task_t[None, :]
    non_overlap = ~(
        (ep_t[:, None] == ep_t[None, :]) & (torch.abs(fr_t[:, None] - fr_t[None, :]) < 16)
    )

    pair_strata = {
        "all_pairs": triu & non_overlap,
        "same_episode": triu & same_ep,
        "same_task_diff_layout": triu & same_task_diff_ep,
        "cross_task": triu & cross_task,
    }

    # Representation similarity matrices
    S_z_cos = pairwise_cosine_sim(z_flat)
    S_z_ccos = pairwise_cosine_sim(z_centered)
    S_z_negl2 = -pairwise_l2_dist(z_flat)
    S_clip_cos = pairwise_cosine_sim(clip_m)

    rep_sims = {
        "latent_cos": S_z_cos,
        "latent_centered_cos": S_z_ccos,
        "latent_neg_l2": S_z_negl2,
        "clip_cos": S_clip_cos,
    }
    # Per-token similarity matrices
    for tok_i in range(4):
        rep_sims[f"token_{tok_i}_cos"] = pairwise_cosine_sim(z_4tok[:, tok_i, :])

    # Optional P(64D) vs Q(3008D) subspace decomposition (using P0 basis if present)
    if P0_BASES_PATH.is_file():
        bases = torch.load(P0_BASES_PATH, map_location="cpu", weights_only=False)
        U64 = None
        for k in ("U_64", "U64", "offline_U64", "reader_U64"):
            if k in bases:
                U64 = bases[k].float().to(device)
                break
        if U64 is None and "offline" in bases and isinstance(bases["offline"], dict):
            for k in ("U_64", "U64", "U"):
                if k in bases["offline"]:
                    U64 = bases["offline"][k][:, :64].float().to(device)
                    break
        if U64 is not None and U64.shape == (3072, 64):
            z_P = (z_flat @ U64) @ U64.T
            z_Q = z_flat - z_P
            rep_sims["subspace_P64_cos"] = pairwise_cosine_sim(z_P)
            rep_sims["subspace_Q3008_cos"] = pairwise_cosine_sim(z_Q)

    attr_list = [
        "global_raw7d",
        "global_balanced7d",
        "global_10d",
        "translation",
        "translation_net_dir",
        "rotation",
        "magnitude",
        "temporal",
        "gripper",
    ]
    S_attrs: Dict[str, torch.Tensor] = {
        name: compute_attribute_similarity_matrix(attrs, name) for name in attr_list
    }
    both_rot = attrs["is_rotating"][:, None] & attrs["is_rotating"][None, :]
    both_mov = attrs["is_moving"][:, None] & attrs["is_moving"][None, :]

    # 1. Pairwise correlations across strata and attributes
    correlations: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for rep_name, S_rep in rep_sims.items():
        correlations[rep_name] = {}
        for stratum_name, smask in pair_strata.items():
            # Main representations get all strata; token/subspace get cross_task & all_pairs
            if rep_name.startswith(("token_", "subspace_")) and stratum_name not in (
                "all_pairs",
                "cross_task",
            ):
                continue
            correlations[rep_name][stratum_name] = {}
            for attr_name in attr_list:
                mask = smask
                if attr_name == "translation_net_dir":
                    mask = smask & both_mov
                rho, r = _spearman_pearson_from_vectors(S_rep[mask], S_attrs[attr_name][mask])
                correlations[rep_name][stratum_name][attr_name] = {
                    "spearman": rho,
                    "pearson": r,
                }
            # Active-rotation subset
            rmask = smask & both_rot
            rho_r, r_r = _spearman_pearson_from_vectors(
                S_rep[rmask], S_attrs["rotation"][rmask]
            )
            correlations[rep_name][stratum_name]["rotation_active"] = {
                "spearman": rho_r,
                "pearson": r_r,
            }

    # 2. kNN Neighborhood Consistency (Unconstrained vs Cross-Task)
    eye = torch.eye(M, device=device, dtype=torch.bool)
    unconstrained_valid = non_overlap & (~eye)
    cross_task_valid = cross_task & (~eye)

    knn_results: Dict[str, Any] = {
        "unconstrained": {},
        "cross_task": {},
        "leakage": {},
        "cross_task_retrieval_quality": {},
    }
    for rep_name, S_rep in [("latent_cos", S_z_cos), ("latent_centered_cos", S_z_ccos), ("clip_cos", S_clip_cos)]:
        knn_results["unconstrained"][rep_name] = {
            a: _compute_knn_metrics(S_rep, S_attrs[a], unconstrained_valid) for a in attr_list
        }
        knn_results["cross_task"][rep_name] = {
            a: _compute_knn_metrics(S_rep, S_attrs[a], cross_task_valid) for a in attr_list
        }

    # Scene / task leakage in unconstrained top-10 neighbors
    for name, S_mat in [
        ("latent_cos", S_z_cos),
        ("latent_centered_cos", S_z_ccos),
        ("clip_cos", S_clip_cos),
        ("oracle_action_bal7d", S_attrs["global_balanced7d"]),
    ]:
        top10 = torch.topk(S_mat.masked_fill(~unconstrained_valid, -1e9), k=10, dim=1).indices
        same_ep_rate = (ep_t[top10] == ep_t[:, None]).float().mean().item()
        same_task_rate = (task_t[top10] == task_t[:, None]).float().mean().item()
        knn_results["leakage"][name] = {
            "top10_same_episode_rate": float(same_ep_rate),
            "top10_same_task_rate": float(same_task_rate),
        }

    # Cross-task retrieval action quality (Top-1 and Top-5 average)
    a10_np = encoded["actions_10d"][sel]
    a7_np = encoded["actions_7d"][sel]
    pos_np = a7_np[:, :, :3]
    rot_np = a7_np[:, :, 3:6]
    grip_np = a7_np[:, :, 6]
    mag_np = attrs["pos_path"].cpu().numpy()
    temp_np = attrs["temporal"].cpu().numpy()
    temp_c = temp_np - temp_np.mean(axis=0, keepdims=True)
    temp_u = temp_c / np.clip(np.linalg.norm(temp_c, axis=-1, keepdims=True), 1e-6, None)

    def _retrieval_quality(nn_idx: np.ndarray) -> Dict[str, float]:
        # nn_idx: (M, K)
        K = nn_idx.shape[1]
        q10 = np.repeat(a10_np[:, None, :, :], K, axis=1)
        n10 = a10_np[nn_idx]
        mse10 = float(((q10 - n10) ** 2).mean())
        q7 = np.repeat(a7_np[:, None, :, :], K, axis=1)
        n7 = a7_np[nn_idx]
        mae7 = float(np.abs(q7 - n7).mean())
        pos_mse = float(((q7[..., :3] - n7[..., :3]) ** 2).mean())
        rot_mse = float(((q7[..., 3:6] - n7[..., 3:6]) ** 2).mean())
        grip_acc = float(((q7[..., 6] > 0) == (n7[..., 6] > 0)).mean())
        # Translation direction cosine
        qp = q7[..., :3].reshape(M * K, -1)
        np_ = n7[..., :3].reshape(M * K, -1)
        pos_cos = float(
            (
                (qp * np_).sum(axis=-1)
                / np.clip(np.linalg.norm(qp, axis=-1) * np.linalg.norm(np_, axis=-1), 1e-6, None)
            ).mean()
        )
        mag_mae = float(np.abs(mag_np[:, None] - mag_np[nn_idx]).mean())
        temp_cos = float((temp_u[:, None, :] * temp_u[nn_idx]).sum(axis=-1).mean())
        return {
            "action10_mse": mse10,
            "action7_mae": mae7,
            "pos_mse": pos_mse,
            "pos_cos": pos_cos,
            "rot_mse": rot_mse,
            "mag_path_mae": mag_mae,
            "temporal_cos": temp_cos,
            "grip_sign_acc": grip_acc,
        }

    # Random cross-task baseline indices
    rand_nn = np.zeros((M, 5), dtype=np.int64)
    tasks_np = cache.val_task_ids[sel]
    for i in range(M):
        cands = np.flatnonzero(tasks_np != tasks_np[i])
        rand_nn[i] = rng.choice(cands, size=5, replace=False)

    for name, S_mat in [
        ("latent_cos", S_z_cos),
        ("latent_centered_cos", S_z_ccos),
        ("clip_cos", S_clip_cos),
        ("oracle_action_10d", S_attrs["global_10d"]),
        ("oracle_action_bal7d", S_attrs["global_balanced7d"]),
    ]:
        top5 = torch.topk(S_mat.masked_fill(~cross_task_valid, -1e9), k=5, dim=1).indices.cpu().numpy()
        knn_results["cross_task_retrieval_quality"][name] = {
            "top1": _retrieval_quality(top5[:, :1]),
            "top5": _retrieval_quality(top5),
        }
    knn_results["cross_task_retrieval_quality"]["random_cross_task"] = {
        "top1": _retrieval_quality(rand_nn[:, :1]),
        "top5": _retrieval_quality(rand_nn),
    }

    # 3. Partial-Attribute Cross-Scene Sensitivity (globally dissimilar, single-attribute similar!)
    partial_attr_results: Dict[str, Any] = {}
    S_glob = S_attrs["global_balanced7d"]
    glob_dissim = pair_strata["cross_task"] & (S_glob < 0.25)
    for attr_name in ("translation", "rotation", "magnitude", "temporal", "gripper"):
        amask = glob_dissim & both_rot if attr_name == "rotation" else glob_dissim
        s_a_vals = S_attrs[attr_name][amask]
        s_z_vals = S_z_cos[amask]
        s_zc_vals = S_z_ccos[amask]
        s_c_vals = S_clip_cos[amask]
        if s_a_vals.numel() > 1000:
            # Subsample for quantile speed
            if s_a_vals.numel() > 200_000:
                g = torch.Generator(device=device).manual_seed(seed + 55)
                sub = torch.randperm(s_a_vals.numel(), generator=g, device=device)[:200_000]
                s_a_vals, s_z_vals, s_zc_vals, s_c_vals = (
                    s_a_vals[sub],
                    s_z_vals[sub],
                    s_zc_vals[sub],
                    s_c_vals[sub],
                )
            hi_thr = torch.quantile(s_a_vals, 0.95)
            lo_thr = torch.quantile(s_a_vals, 0.50)
            hi = s_a_vals >= hi_thr
            lo = s_a_vals <= lo_thr
            rho_p, _ = _spearman_pearson_from_vectors(s_z_vals, s_a_vals)
            rho_pc, _ = _spearman_pearson_from_vectors(s_zc_vals, s_a_vals)
            partial_attr_results[attr_name] = {
                "high_attr_sim_threshold": float(hi_thr.item()),
                "latent_cos_partial_match": float(s_z_vals[hi].mean().item()),
                "latent_cos_non_match": float(s_z_vals[lo].mean().item()),
                "latent_cos_separation": float((s_z_vals[hi].mean() - s_z_vals[lo].mean()).item()),
                "centered_cos_separation": float((s_zc_vals[hi].mean() - s_zc_vals[lo].mean()).item()),
                "clip_cos_separation": float((s_c_vals[hi].mean() - s_c_vals[lo].mean()).item()),
                "partial_spearman_latent_cos": rho_p,
                "partial_spearman_centered_cos": rho_pc,
            }

    # 4. Controlled Scene-Sensitivity Invariance (re-encoding cached S0..S10 CLIP clips)
    scene_inv: Dict[str, Any] = {}
    if SCENE_SENSITIVITY_LATENTS.is_file():
        sd = np.load(SCENE_SENSITIVITY_LATENTS, allow_pickle=True)
        scene_ids = [str(x) for x in sd["scene_ids"]]
        clip_sc = torch.from_numpy(sd["clip_feats"].astype(np.float32)).to(device)  # (12, 16, 2, 768)
        gt10_sc = sd["gt_actions_10d"].astype(np.float32)  # (16, 10)
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if (use_bf16 and device.type == "cuda")
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with torch.inference_mode(), amp_ctx:
            b_sc = policy.encode_local_behavior(clip_sc, None).float()  # (12, 4, 768)
        b_sc_flat = b_sc.reshape(len(scene_ids), -1)
        pred_sc = ddim_decode_batched(policy, b_sc, device, seed=12345, batch_size=12, use_bf16=use_bf16)

        # Variants S1..S10 vs S0_base (index 0; index 1 is S0_repeat)
        z0 = b_sc_flat[0]
        per_scene = {}
        cos_list, nl2_list, mse_vs_base_list, mse_vs_gt_list = [], [], [], []
        for i, sid in enumerate(scene_ids):
            zi = b_sc_flat[i]
            cos_i = float(F.cosine_similarity(zi[None], z0[None]).item())
            denom = 0.5 * (zi.norm() + z0.norm()).clamp_min(1e-6)
            nl2_i = float(((zi - z0).norm() / denom).item())
            mse_b = float(((pred_sc[i] - pred_sc[0]) ** 2).mean())
            mse_g = float(((pred_sc[i] - gt10_sc) ** 2).mean())
            per_scene[sid] = {
                "cos_vs_base": cos_i,
                "norm_l2_vs_base": nl2_i,
                "decoder_mse_vs_base": mse_b,
                "decoder_mse_vs_gt": mse_g,
            }
            if sid not in ("S0_base", "S0_repeat"):
                cos_list.append(cos_i)
                nl2_list.append(nl2_i)
                mse_vs_base_list.append(mse_b)
                mse_vs_gt_list.append(mse_g)
        # Reference cross-action nL2 scale from val same_episode pairs
        same_ep_nl2 = float(
            (
                pairwise_l2_dist(z_flat)
                / (0.5 * (z_flat.norm(dim=-1)[:, None] + z_flat.norm(dim=-1)[None, :])).clamp_min(1e-6)
            )[pair_strata["same_episode"]].mean().item()
        )
        scene_inv = {
            "variants_vs_base_mean_cos": float(np.mean(cos_list)),
            "variants_vs_base_mean_nl2": float(np.mean(nl2_list)),
            "variants_vs_base_mean_decoder_mse": float(np.mean(mse_vs_base_list)),
            "variants_vs_gt_mean_decoder_mse": float(np.mean(mse_vs_gt_list)),
            "base_gt_decoder_mse": float(per_scene["S0_base"]["decoder_mse_vs_gt"]),
            "same_ep_diff_action_mean_nl2": same_ep_nl2,
            "scene_to_action_nl2_ratio": float(np.mean(nl2_list) / max(same_ep_nl2, 1e-6)),
            "per_scene": per_scene,
        }

    return {
        "num_samples": M,
        "correlations": correlations,
        "knn": knn_results,
        "partial_attribute": partial_attr_results,
        "scene_sensitivity": scene_inv,
    }


# ---------------------------------------------------------------------------
# Suite 3: Latent Displacement Consistency & Latent Arithmetic Transfer
# ---------------------------------------------------------------------------
def evaluate_suite3_displacement_and_arithmetic(
    policy: E9Stage1LocalPolicy,
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    *,
    num_transitions: int = 1500,
    num_analogies_per_tier: int = 256,
    seed: int = 42,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    # Build pool of within-episode non-overlapping transitions (i -> j)
    ep_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, ep in enumerate(cache.val_episodes):
        if not encoded["is_tail"][idx]:
            ep_to_indices[int(ep)].append(idx)

    cand_pairs: List[Tuple[int, int]] = []
    for ep, idxs in sorted(ep_to_indices.items()):
        if len(idxs) < 2:
            continue
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                i, j = idxs[a_i], idxs[b_i]
                if abs(int(cache.val_frames[j]) - int(cache.val_frames[i])) >= 16:
                    cand_pairs.append((i, j))

    rng = np.random.default_rng(seed + 303)
    cand_arr = np.array(cand_pairs, dtype=np.int64)
    if len(cand_arr) > num_transitions:
        sel_p = rng.choice(len(cand_arr), size=num_transitions, replace=False)
        trans = cand_arr[np.sort(sel_p)]
    else:
        trans = cand_arr
    T_n = len(trans)
    ia, ib = trans[:, 0], trans[:, 1]
    t_tasks = torch.from_numpy(cache.val_task_ids[ia]).to(device)

    a7_a = torch.from_numpy(encoded["actions_7d"][ia]).to(device)
    a7_b = torch.from_numpy(encoded["actions_7d"][ib]).to(device)
    d_attrs = extract_action_delta_features(a7_a, a7_b)

    dz = torch.from_numpy(encoded["latents_flat"][ib] - encoded["latents_flat"][ia]).to(device)
    dclip = torch.from_numpy(encoded["clip_mean"][ib] - encoded["clip_mean"][ia]).to(device)

    triu = torch.triu(torch.ones((T_n, T_n), device=device, dtype=torch.bool), diagonal=1)
    cross_task_t = triu & (t_tasks[:, None] != t_tasks[None, :])

    S_dz = pairwise_cosine_sim(dz)
    S_dclip = pairwise_cosine_sim(dclip)
    dz_norms = dz.norm(dim=-1)
    rel_dz_err = pairwise_l2_dist(dz) / (0.5 * (dz_norms[:, None] + dz_norms[None, :])).clamp_min(1e-6)

    disp_corr: Dict[str, Any] = {}
    for dname, dvec in d_attrs.items():
        S_da = pairwise_cosine_sim(dvec)
        da_norms = dvec.norm(dim=-1)
        active = cross_task_t & (da_norms[:, None] > 0.5) & (da_norms[None, :] > 0.5)
        rho_z, r_z = _spearman_pearson_from_vectors(S_dz[active], S_da[active])
        rho_c, r_c = _spearman_pearson_from_vectors(S_dclip[active], S_da[active])
        # Matched action-change analogies vs random cross-task transitions
        ratio = da_norms[:, None] / da_norms[None, :].clamp_min(1e-6)
        matched = active & (S_da > 0.80) & (ratio > 0.7) & (ratio < 1.43)
        n_matched = int(matched.sum().item())
        disp_corr[dname] = {
            "latent_disp_spearman": rho_z,
            "latent_disp_pearson": r_z,
            "clip_disp_spearman": rho_c,
            "clip_disp_pearson": r_c,
            "n_matched_cross_task_analogies": n_matched,
            "matched_latent_disp_cos": float(S_dz[matched].mean().item()) if n_matched > 0 else float("nan"),
            "random_latent_disp_cos": float(S_dz[active].mean().item()),
            "matched_latent_disp_rel_l2": float(rel_dz_err[matched].mean().item()) if n_matched > 0 else float("nan"),
            "random_latent_disp_rel_l2": float(rel_dz_err[active].mean().item()),
            "matched_clip_disp_cos": float(S_dclip[matched].mean().item()) if n_matched > 0 else float("nan"),
        }

    # --- Latent Arithmetic Transfer (Tier 1: Matched-Start; Tier 2: Distinct-Start Parallel Transport) ---
    # Construct deterministic quadruples (A1, A2, B1, B2) purely from ground-truth actions & tasks
    # (independent of the model being evaluated, so E9 vs E12-A/B/C share the EXACT SAME quadruples!)
    da_bal = d_attrs["delta_balanced7d"]  # (T_n, 112)
    S_da_bal = pairwise_cosine_sim(da_bal)
    da_raw7 = d_attrs["delta_raw7d"]
    da_norms_b = da_bal.norm(dim=-1)
    # Pairwise delta MSE in 7D
    da_mse = (pairwise_l2_dist(da_raw7) ** 2) / float(da_raw7.shape[1])

    # Base action MSE between A1 and B1 (in 10D)
    a10_a = torch.from_numpy(encoded["actions_10d"][ia].reshape(T_n, -1)).to(device)
    base_mse10 = (pairwise_l2_dist(a10_a) ** 2) / float(a10_a.shape[1])

    # Target action fidelity check: MSE(a_B1 + (a_A2 - a_A1), a_B2) in 7D == da_mse!
    # Also ensure non-trivial transitions: ||a_B2 - a_B1||^2 / 112 > 0.05
    trans_mag = (da_raw7 ** 2).mean(dim=-1)
    nontrivial = (trans_mag[:, None] > 0.05) & (trans_mag[None, :] > 0.05)
    norm_ratio = da_norms_b[:, None] / da_norms_b[None, :].clamp_min(1e-6)
    good_delta = cross_task_t & nontrivial & (S_da_bal > 0.78) & (norm_ratio > 0.72) & (norm_ratio < 1.38) & (da_mse < 0.035)

    tier1_mask = good_delta & (base_mse10 < 0.035)
    tier2_mask = good_delta & (base_mse10 > 0.08)

    def _eval_analogy_tier(mask: torch.Tensor, tier_name: str) -> Dict[str, Any]:
        pairs_ij = torch.nonzero(mask, as_tuple=False).cpu().numpy()
        if len(pairs_ij) == 0:
            return {"num_quadruples": 0}
        # Sort by smallest da_mse then sample deterministically from top pool for diversity
        scores = da_mse[mask].cpu().numpy()
        order = np.argsort(scores)
        pool_size = min(len(order), num_analogies_per_tier * 4)
        pool = pairs_ij[order[:pool_size]]
        rng_t = np.random.default_rng(seed + (701 if tier_name == "tier1" else 702))
        if len(pool) > num_analogies_per_tier:
            chosen = pool[np.sort(rng_t.choice(len(pool), size=num_analogies_per_tier, replace=False))]
        else:
            chosen = pool

        pA = chosen[:, 0]
        pB = chosen[:, 1]
        A1, A2 = ia[pA], ib[pA]
        B1, B2 = ia[pB], ib[pB]

        zA1 = encoded["latents_flat"][A1]
        zA2 = encoded["latents_flat"][A2]
        zB1 = encoded["latents_flat"][B1]
        zB2 = encoded["latents_flat"][B2]

        dzA = zA2 - zA1
        dzB = zB2 - zB1
        z_arith = zB1 + dzA

        # Latent geometry metrics
        disp_cos = float(
            (
                (dzA * dzB).sum(axis=-1)
                / np.clip(np.linalg.norm(dzA, axis=-1) * np.linalg.norm(dzB, axis=-1), 1e-6, None)
            ).mean()
        )
        lat_mse_base = float(((zB1 - zB2) ** 2).mean())
        lat_mse_arith = float(((z_arith - zB2) ** 2).mean())
        lat_mse_copy = float(((zA2 - zB2) ** 2).mean())

        # Decode via frozen DDIM Action Decoder with shared noise
        gt_B2 = encoded["actions_10d"][B2]
        pred_oracle = ddim_decode_batched(policy, zB2, device, seed=12345, use_bf16=use_bf16)
        pred_base = ddim_decode_batched(policy, zB1, device, seed=12345, use_bf16=use_bf16)
        pred_copy = ddim_decode_batched(policy, zA2, device, seed=12345, use_bf16=use_bf16)
        pred_arith = ddim_decode_batched(policy, z_arith, device, seed=12345, use_bf16=use_bf16)

        m_oracle = _action_metrics_dict(pred_oracle, gt_B2)
        m_base = _action_metrics_dict(pred_base, gt_B2)
        m_copy = _action_metrics_dict(pred_copy, gt_B2)
        m_arith = _action_metrics_dict(pred_arith, gt_B2)

        # Arithmetic Recovery Ratio (ARR): 1.0 = matches oracle, 0.0 = no improvement over z_B1
        arr_10d = (m_base["action10_mse"] - m_arith["action10_mse"]) / max(
            m_base["action10_mse"] - m_oracle["action10_mse"], 1e-9
        )
        arr_pos = (m_base["pos_mse"] - m_arith["pos_mse"]) / max(
            m_base["pos_mse"] - m_oracle["pos_mse"], 1e-9
        )
        arr_grip = (m_base["grip_mse"] - m_arith["grip_mse"]) / max(
            m_base["grip_mse"] - m_oracle["grip_mse"], 1e-9
        )
        # Ideal action-space arithmetic lower bound (in 7D)
        ideal_a7 = encoded["actions_7d"][B1] + (encoded["actions_7d"][A2] - encoded["actions_7d"][A1])
        ideal_a7_mae = float(np.abs(ideal_a7 - encoded["actions_7d"][B2]).mean())

        return {
            "num_quadruples": int(len(chosen)),
            "latent_disp_cos": disp_cos,
            "latent_mse_base": lat_mse_base,
            "latent_mse_arith": lat_mse_arith,
            "latent_mse_copy": lat_mse_copy,
            "latent_mse_reduction_pct": float(100.0 * (lat_mse_base - lat_mse_arith) / max(lat_mse_base, 1e-9)),
            "action_space_ideal_7d_mae": ideal_a7_mae,
            "ddim_oracle": m_oracle,
            "ddim_base_zB1": m_base,
            "ddim_copy_zA2": m_copy,
            "ddim_arith": m_arith,
            "arithmetic_recovery_ratio_10d": float(arr_10d),
            "arithmetic_recovery_ratio_pos": float(arr_pos),
            "arithmetic_recovery_ratio_grip": float(arr_grip),
        }

    return {
        "num_transitions": T_n,
        "displacement_consistency": disp_corr,
        "latent_arithmetic_tier1_matched_start": _eval_analogy_tier(tier1_mask, "tier1"),
        "latent_arithmetic_tier2_distinct_start": _eval_analogy_tier(tier2_mask, "tier2"),
    }


# ---------------------------------------------------------------------------
# Suite 4: Unseen Behavior Combination Generalization
# ---------------------------------------------------------------------------
def evaluate_suite4_unseen_combinations(
    policy: E9Stage1LocalPolicy,
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    *,
    max_eval_samples: int = 512,
    seed: int = 42,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    non_tail = np.flatnonzero(~encoded["is_tail"])
    a7 = encoded["actions_7d"][non_tail]
    a10 = encoded["actions_10d"][non_tail]
    z = encoded["latents_flat"][non_tail]
    factors = discretize_behavior_factors(a7)
    combos = factors["combo_dir_grip"]
    dirs = factors["dir_label"]
    grips = factors["grip_label"]

    # Choose 6 held-out (dir, grip) cells deterministically such that every dir and every grip
    # remains abundant (>= 40 samples) in the seen cells.
    unique_dirs = sorted(set(dirs.tolist()))
    unique_grips = sorted(set(grips.tolist()))
    combo_counts = {c: int((combos == c).sum()) for c in set(combos.tolist())}

    # Deterministic diverse held-out pairs covering different directions & gripper regimes
    preferred_holdouts = [
        "+x|closed",
        "-x|open",
        "+y|open",
        "-y|closed",
        "+z|closing",
        "-z|opening",
    ]
    held_out_combos = []
    for c in preferred_holdouts:
        if combo_counts.get(c, 0) >= 25:
            held_out_combos.append(c)
    # Fallback: add next largest combos not duplicating dir/grip if fewer than 6
    if len(held_out_combos) < 6:
        for c, cnt in sorted(combo_counts.items(), key=lambda kv: -kv[1]):
            if c in held_out_combos or cnt < 25:
                continue
            d_c, g_c = c.split("|")
            # Ensure dir and grip still have >= 40 samples outside held_out
            rem_d = sum(v for k, v in combo_counts.items() if k.startswith(d_c + "|") and k not in held_out_combos and k != c)
            rem_g = sum(v for k, v in combo_counts.items() if k.endswith("|" + g_c) and k not in held_out_combos and k != c)
            if rem_d >= 40 and rem_g >= 40:
                held_out_combos.append(c)
            if len(held_out_combos) == 6:
                break

    is_unseen = np.isin(combos, held_out_combos)
    seen_idx = np.flatnonzero(~is_unseen)
    unseen_idx = np.flatnonzero(is_unseen)

    # Split seen_idx into train (85%) and seen_test (15%) for direct seen-vs-unseen comparison
    rng = np.random.default_rng(seed + 404)
    perm_seen = rng.permutation(seen_idx)
    n_seen_te = min(max_eval_samples, max(100, int(0.15 * len(perm_seen))))
    seen_te = perm_seen[:n_seen_te]
    seen_tr = perm_seen[n_seen_te:]
    if len(unseen_idx) > max_eval_samples:
        unseen_te = np.sort(rng.choice(unseen_idx, size=max_eval_samples, replace=False))
    else:
        unseen_te = unseen_idx

    # 1. Linear Compositional Mapping: action_7d (112D) -> z (3072D) trained ONLY on seen_tr
    X_all = a7.reshape(len(a7), -1)
    z_hat_seen, alpha_lin = _ridge_fit_predict(X_all[seen_tr], z[seen_tr], X_all[seen_te])
    z_hat_unseen, _ = _ridge_fit_predict(X_all[seen_tr], z[seen_tr], X_all[unseen_te], alphas=(alpha_lin,))

    pred_seen_lin = ddim_decode_batched(policy, z_hat_seen, device, seed=12345, use_bf16=use_bf16)
    pred_seen_ora = ddim_decode_batched(policy, z[seen_te], device, seed=12345, use_bf16=use_bf16)
    pred_unseen_lin = ddim_decode_batched(policy, z_hat_unseen, device, seed=12345, use_bf16=use_bf16)
    pred_unseen_ora = ddim_decode_batched(policy, z[unseen_te], device, seed=12345, use_bf16=use_bf16)

    # 2. Zero-Shot Compositional Latent Arithmetic on unseen combos:
    #    For each unseen sample with (d*, g*), take a nearest seen sample with same dir (d*, g_base)
    #    and add the mean gripper-transition vector Δz(g_base -> g*) computed ONLY over OTHER dirs d != d*!
    seen_tr_dirs = dirs[seen_tr]
    seen_tr_grips = grips[seen_tr]
    seen_tr_pos = a7[seen_tr, :, :3].reshape(len(seen_tr), -1)

    # Precompute per-(d, g) mean latents in seen_tr
    cell_means: Dict[Tuple[str, str], np.ndarray] = {}
    for d_val in unique_dirs:
        for g_val in unique_grips:
            m = (seen_tr_dirs == d_val) & (seen_tr_grips == g_val)
            if m.sum() >= 5:
                cell_means[(d_val, g_val)] = z[seen_tr[m]].mean(axis=0)

    z_comp_arith = np.zeros((len(unseen_te), 3072), dtype=np.float32)
    z_comp_base = np.zeros((len(unseen_te), 3072), dtype=np.float32)
    valid_comp = np.zeros(len(unseen_te), dtype=bool)

    for k, u_i in enumerate(unseen_te):
        d_star, g_star = dirs[u_i], grips[u_i]
        # Find best base gripper g_base that has samples with d_star AND has other dirs with both g_base & g_star
        best_g_base = None
        best_dz_g = None
        for g_b in unique_grips:
            if g_b == g_star:
                continue
            if not ((seen_tr_dirs == d_star) & (seen_tr_grips == g_b)).any():
                continue
            deltas = []
            for d_o in unique_dirs:
                if d_o == d_star:
                    continue
                if (d_o, g_b) in cell_means and (d_o, g_star) in cell_means:
                    deltas.append(cell_means[(d_o, g_star)] - cell_means[(d_o, g_b)])
            if deltas:
                best_g_base = g_b
                best_dz_g = np.mean(deltas, axis=0)
                break
        if best_g_base is None:
            continue
        cands = np.flatnonzero((seen_tr_dirs == d_star) & (seen_tr_grips == best_g_base))
        # Match nearest translation trajectory inside (d_star, g_base)
        u_pos = a7[u_i, :, :3].reshape(1, -1)
        dists = np.linalg.norm(seen_tr_pos[cands] - u_pos, axis=-1)
        best_local = cands[int(np.argmin(dists))]
        z_base_k = z[seen_tr[best_local]]
        z_comp_base[k] = z_base_k
        z_comp_arith[k] = z_base_k + best_dz_g
        valid_comp[k] = True

    comp_res: Dict[str, Any] = {"num_valid": int(valid_comp.sum())}
    if valid_comp.sum() > 0:
        vi = unseen_te[valid_comp]
        pred_c_base = ddim_decode_batched(policy, z_comp_base[valid_comp], device, seed=12345, use_bf16=use_bf16)
        pred_c_arith = ddim_decode_batched(policy, z_comp_arith[valid_comp], device, seed=12345, use_bf16=use_bf16)
        pred_c_ora = ddim_decode_batched(policy, z[vi], device, seed=12345, use_bf16=use_bf16)
        mb = _action_metrics_dict(pred_c_base, a10[vi])
        ma = _action_metrics_dict(pred_c_arith, a10[vi])
        mo = _action_metrics_dict(pred_c_ora, a10[vi])
        comp_res.update(
            {
                "latent_mse_base": float(((z_comp_base[valid_comp] - z[vi]) ** 2).mean()),
                "latent_mse_arith": float(((z_comp_arith[valid_comp] - z[vi]) ** 2).mean()),
                "ddim_base": mb,
                "ddim_arith": ma,
                "ddim_oracle": mo,
                "recovery_ratio_10d": float(
                    (mb["action10_mse"] - ma["action10_mse"]) / max(mb["action10_mse"] - mo["action10_mse"], 1e-9)
                ),
                "grip_acc_gain": float(ma["grip_sign_acc"] - mb["grip_sign_acc"]),
            }
        )

    m_s_lin = _action_metrics_dict(pred_seen_lin, a10[seen_te])
    m_s_ora = _action_metrics_dict(pred_seen_ora, a10[seen_te])
    m_u_lin = _action_metrics_dict(pred_unseen_lin, a10[unseen_te])
    m_u_ora = _action_metrics_dict(pred_unseen_ora, a10[unseen_te])

    return {
        "held_out_combos": held_out_combos,
        "num_seen_train": int(len(seen_tr)),
        "num_seen_test": int(len(seen_te)),
        "num_unseen_test": int(len(unseen_te)),
        "linear_compositional_ridge": {
            "alpha": alpha_lin,
            "seen_latent_mse": float(((z_hat_seen - z[seen_te]) ** 2).mean()),
            "unseen_latent_mse": float(((z_hat_unseen - z[unseen_te]) ** 2).mean()),
            "seen_ddim": m_s_lin,
            "seen_oracle_ddim": m_s_ora,
            "unseen_ddim": m_u_lin,
            "unseen_oracle_ddim": m_u_ora,
            "unseen_to_seen_mse10_ratio": float(
                m_u_lin["action10_mse"] / max(m_s_lin["action10_mse"], 1e-9)
            ),
        },
        "zero_shot_compositional_arithmetic": comp_res,
    }


# ---------------------------------------------------------------------------
# Suite 5: Transfer Efficiency (Linear Probing, Data Efficiency, Effective Rank)
# ---------------------------------------------------------------------------
def evaluate_suite5_transfer_efficiency(
    policy: E9Stage1LocalPolicy,
    cache: ValClipCache,
    encoded: Dict[str, Any],
    device: torch.device,
    *,
    sample_sizes: Tuple[int, ...] = (50, 100, 250, 500, 1000, 2000),
    test_pairs_count: int = 256,
    seed: int = 42,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    non_tail = np.flatnonzero(~encoded["is_tail"])
    tasks_nt = cache.val_task_ids[non_tail]
    unique_tasks = np.unique(tasks_nt)
    rng = np.random.default_rng(seed + 505)
    perm_tasks = rng.permutation(unique_tasks)
    n_tr_tasks = int(0.7 * len(perm_tasks))
    tr_task_set = set(perm_tasks[:n_tr_tasks].tolist())

    tr_mask = np.array([int(t) in tr_task_set for t in tasks_nt], dtype=bool)
    tr_idx = non_tail[tr_mask]
    te_idx = non_tail[~tr_mask]

    # 5a. Cross-Task Linear Probing R^2 for all 6 behavioral attributes (Latent vs CLIP)
    a10_all = torch.from_numpy(encoded["actions_10d"])
    a7_all = torch.from_numpy(encoded["actions_7d"])
    attrs_all = extract_action_attribute_vectors(a10_all, a7_all)

    probe_attrs = ["global_10d", "global_balanced7d", "translation", "rotation", "magnitude", "temporal", "gripper"]
    linear_probes: Dict[str, Dict[str, Any]] = {"latent": {}, "clip": {}}

    Z_tr, Z_te = encoded["latents_flat"][tr_idx], encoded["latents_flat"][te_idx]
    C_tr, C_te = encoded["clip_mean"][tr_idx], encoded["clip_mean"][te_idx]

    for aname in probe_attrs:
        Y = attrs_all[aname].numpy()
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]
        var_te = float(((Y_te - Y_te.mean(axis=0, keepdims=True)) ** 2).mean())

        pred_z, a_z = _ridge_fit_predict(Z_tr, Y_tr, Z_te)
        mse_z = float(((pred_z - Y_te) ** 2).mean())
        r2_z = float(1.0 - mse_z / max(var_te, 1e-12))
        linear_probes["latent"][aname] = {"r2": r2_z, "mse": mse_z, "alpha": a_z}

        pred_c, a_c = _ridge_fit_predict(C_tr, Y_tr, C_te)
        mse_c = float(((pred_c - Y_te) ** 2).mean())
        r2_c = float(1.0 - mse_c / max(var_te, 1e-12))
        linear_probes["clip"][aname] = {"r2": r2_c, "mse": mse_c, "alpha": a_c}

    # 5b. Action-Change Transfer Efficiency Curve (Linear Ridge vs 2-Layer MLP)
    # Pairs (src -> tgt) from same episode; train tasks strictly disjoint from test tasks!
    def _sample_episode_pairs(idx_pool: np.ndarray, max_n: int, rng_local: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        by_ep: Dict[int, List[int]] = defaultdict(list)
        for idx in idx_pool:
            by_ep[int(cache.val_episodes[idx])].append(int(idx))
        pairs = []
        for ep, lst in sorted(by_ep.items()):
            for a_i in range(len(lst)):
                for b_i in range(len(lst)):
                    if a_i != b_i and abs(int(cache.val_frames[lst[a_i]]) - int(cache.val_frames[lst[b_i]])) >= 16:
                        pairs.append((lst[a_i], lst[b_i]))
        arr = np.array(pairs, dtype=np.int64)
        if len(arr) > max_n:
            arr = arr[rng_local.choice(len(arr), size=max_n, replace=False)]
        return arr[:, 0], arr[:, 1]

    tr_src, tr_tgt = _sample_episode_pairs(tr_idx, max(sample_sizes), np.random.default_rng(seed + 601))
    te_src, te_tgt = _sample_episode_pairs(te_idx, test_pairs_count, np.random.default_rng(seed + 602))

    # Input: Δa_7d (112D) -> predict Δz = z_tgt - z_src (3072D), then z_hat = z_src + Δz_hat!
    X_da_tr = (encoded["actions_7d"][tr_tgt] - encoded["actions_7d"][tr_src]).reshape(len(tr_src), -1)
    Y_dz_tr = encoded["latents_flat"][tr_tgt] - encoded["latents_flat"][tr_src]
    X_da_te = (encoded["actions_7d"][te_tgt] - encoded["actions_7d"][te_src]).reshape(len(te_src), -1)
    Y_dz_te = encoded["latents_flat"][te_tgt] - encoded["latents_flat"][te_src]

    gt_te_10d = encoded["actions_10d"][te_tgt]
    pred_ora_te = ddim_decode_batched(policy, encoded["latents_flat"][te_tgt], device, seed=12345, use_bf16=use_bf16)
    pred_id_te = ddim_decode_batched(policy, encoded["latents_flat"][te_src], device, seed=12345, use_bf16=use_bf16)
    ora_m = _action_metrics_dict(pred_ora_te, gt_te_10d)
    id_m = _action_metrics_dict(pred_id_te, gt_te_10d)

    action_transfer_curve: Dict[str, Any] = {
        "num_test_pairs": int(len(te_src)),
        "oracle_ddim": ora_m,
        "identity_no_transfer_ddim": id_m,
        "by_n": {},
    }
    for n_samp in sample_sizes:
        n_eff = min(int(n_samp), len(tr_src))
        # Linear Ridge: Δz_hat = W Δa + b
        dz_lin, alpha_l = _ridge_fit_predict(X_da_tr[:n_eff], Y_dz_tr[:n_eff], X_da_te)
        z_hat_lin = encoded["latents_flat"][te_src] + dz_lin
        pred_lin = ddim_decode_batched(policy, z_hat_lin, device, seed=12345, use_bf16=use_bf16)
        m_lin = _action_metrics_dict(pred_lin, gt_te_10d)

        # 2-Layer MLP: Δz_hat = MLP(Δa)
        dz_mlp = _train_small_mlp_predict(
            X_da_tr[:n_eff], Y_dz_tr[:n_eff], X_da_te, device=device, seed=seed + n_eff
        )
        z_hat_mlp = encoded["latents_flat"][te_src] + dz_mlp
        pred_mlp = ddim_decode_batched(policy, z_hat_mlp, device, seed=12345, use_bf16=use_bf16)
        m_mlp = _action_metrics_dict(pred_mlp, gt_te_10d)

        action_transfer_curve["by_n"][str(n_eff)] = {
            "linear_ridge": {
                "alpha": alpha_l,
                "latent_mse": float(((z_hat_lin - encoded["latents_flat"][te_tgt]) ** 2).mean()),
                "ddim_10d_mse": m_lin["action10_mse"],
                "ddim_7d_mae": m_lin["action7_mae"],
                "grip_acc": m_lin["grip_sign_acc"],
                "arr_10d": float(
                    (id_m["action10_mse"] - m_lin["action10_mse"])
                    / max(id_m["action10_mse"] - ora_m["action10_mse"], 1e-9)
                ),
            },
            "mlp_2layer": {
                "latent_mse": float(((z_hat_mlp - encoded["latents_flat"][te_tgt]) ** 2).mean()),
                "ddim_10d_mse": m_mlp["action10_mse"],
                "ddim_7d_mae": m_mlp["action7_mae"],
                "grip_acc": m_mlp["grip_sign_acc"],
                "arr_10d": float(
                    (id_m["action10_mse"] - m_mlp["action10_mse"])
                    / max(id_m["action10_mse"] - ora_m["action10_mse"], 1e-9)
                ),
            },
        }

    # 5c. Cross-Layout Same-Task Geometry (Effective Rank of Z and cross-layout Δz)
    # Match pairs in same task, different layout, close relative progress (|Δprog| < 0.08)
    by_task: Dict[int, List[int]] = defaultdict(list)
    for idx in non_tail:
        by_task[int(cache.val_task_ids[idx])].append(int(idx))

    cl_src, cl_tgt = [], []
    for tid, idxs in sorted(by_task.items()):
        for a_i in range(len(idxs)):
            i = idxs[a_i]
            best_j = None
            best_dp = 0.08
            for b_i in range(len(idxs)):
                j = idxs[b_i]
                if cache.val_episodes[i] == cache.val_episodes[j]:
                    continue
                dp = abs(float(cache.val_rel_progress[i] - cache.val_rel_progress[j]))
                if dp < best_dp:
                    best_dp = dp
                    best_j = j
            if best_j is not None:
                cl_src.append(i)
                cl_tgt.append(best_j)

    cl_src_arr = np.array(cl_src, dtype=np.int64)
    cl_tgt_arr = np.array(cl_tgt, dtype=np.int64)
    dz_cl = encoded["latents_flat"][cl_tgt_arr] - encoded["latents_flat"][cl_src_arr]
    z_rank_stats = _effective_rank_stats(encoded["latents_flat"][non_tail])
    dz_rank_stats = _effective_rank_stats(dz_cl)
    cl_stats = {
        "num_cross_layout_pairs": int(len(cl_src_arr)),
        "mean_action10_mse_between_matched_layouts": float(
            ((encoded["actions_10d"][cl_src_arr] - encoded["actions_10d"][cl_tgt_arr]) ** 2).mean()
        ),
        "latent_Z_rank": z_rank_stats,
        "cross_layout_delta_z_rank": dz_rank_stats,
        "delta_z_to_Z_rms_ratio": float(
            dz_rank_stats["rms_centered"] / max(z_rank_stats["rms_centered"], 1e-9)
        ),
    }

    return {
        "linear_probing_cross_task": linear_probes,
        "action_change_transfer_efficiency": action_transfer_curve,
        "cross_layout_geometry": cl_stats,
    }


# ---------------------------------------------------------------------------
# Full Unified Benchmark Runner
# ---------------------------------------------------------------------------
def run_full_eval(
    ckpt_path: Path,
    out_dir: Path,
    *,
    label: str = "model",
    device: torch.device,
    ddim_samples: int = 2048,
    max_rel_samples: int = 3500,
    use_bf16: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E12 Eval:{label}] Loading checkpoint {ckpt_path} on {device}...", flush=True)
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    cache = ValClipCache(seed=seed)

    t1 = time.monotonic()
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=use_bf16)
    print(f"[E12 Eval:{label}] Encoded {len(cache.val_episodes)} val starts in {time.monotonic()-t1:.1f}s", flush=True)

    s1 = evaluate_suite1_reconstruction(
        policy, cache, encoded, device, ddim_samples=ddim_samples, use_bf16=use_bf16
    )
    print(
        f"[E12 Eval:{label}] Suite 1 (Recon): self_diff={s1['val_self_diff_loss']:.5f} | "
        f"ddim_10d_mse={s1['ddim_self']['action10_mse']:.5f} | 7d_mae={s1['ddim_self']['action7_mae']:.5f} | "
        f"grip_acc={s1['ddim_self']['grip_sign_acc']*100:.2f}% | "
        f"ts_ratio={s1['timestep_distinctiveness']['mismatch_to_match_mse10_ratio']:.1f}x",
        flush=True,
    )

    s2 = evaluate_suite2_relational(
        policy, cache, encoded, device, max_rel_samples=max_rel_samples, seed=seed, use_bf16=use_bf16
    )
    ct_corr = s2["correlations"]["latent_cos"]["cross_task"]
    print(
        f"[E12 Eval:{label}] Suite 2 (Relational cross-task rho): "
        f"bal7d={ct_corr['global_balanced7d']['spearman']:.3f}, "
        f"trans={ct_corr['translation']['spearman']:.3f}, "
        f"rot_act={ct_corr['rotation_active']['spearman']:.3f}, "
        f"mag={ct_corr['magnitude']['spearman']:.3f}, "
        f"temp={ct_corr['temporal']['spearman']:.3f}, "
        f"grip={ct_corr['gripper']['spearman']:.3f} | "
        f"same_task_leak={s2['knn']['leakage']['latent_cos']['top10_same_task_rate']*100:.1f}%",
        flush=True,
    )

    s3 = evaluate_suite3_displacement_and_arithmetic(
        policy, cache, encoded, device, seed=seed, use_bf16=use_bf16
    )
    t1_ar = s3["latent_arithmetic_tier1_matched_start"]
    t2_ar = s3["latent_arithmetic_tier2_distinct_start"]
    print(
        f"[E12 Eval:{label}] Suite 3 (Disp/Arith): "
        f"disp_rho(bal7d)={s3['displacement_consistency']['delta_balanced7d']['latent_disp_spearman']:.3f} | "
        f"Tier1 ARR={t1_ar.get('arithmetic_recovery_ratio_10d', float('nan'))*100:.1f}% (cos={t1_ar.get('latent_disp_cos', float('nan')):.3f}) | "
        f"Tier2 ARR={t2_ar.get('arithmetic_recovery_ratio_10d', float('nan'))*100:.1f}% (cos={t2_ar.get('latent_disp_cos', float('nan')):.3f})",
        flush=True,
    )

    s4 = evaluate_suite4_unseen_combinations(
        policy, cache, encoded, device, seed=seed, use_bf16=use_bf16
    )
    print(
        f"[E12 Eval:{label}] Suite 4 (Unseen Combos): "
        f"ridge_unseen_mse10={s4['linear_compositional_ridge']['unseen_ddim']['action10_mse']:.5f} "
        f"(ratio={s4['linear_compositional_ridge']['unseen_to_seen_mse10_ratio']:.2f}x) | "
        f"zero_shot_arith_ARR={s4['zero_shot_compositional_arithmetic'].get('recovery_ratio_10d', float('nan'))*100:.1f}%",
        flush=True,
    )

    s5 = evaluate_suite5_transfer_efficiency(
        policy, cache, encoded, device, seed=seed, use_bf16=use_bf16
    )
    lp = s5["linear_probing_cross_task"]["latent"]
    tc100 = s5["action_change_transfer_efficiency"]["by_n"]["100"]
    tc1000 = s5["action_change_transfer_efficiency"]["by_n"]["1000"]
    print(
        f"[E12 Eval:{label}] Suite 5 (Transfer Eff): "
        f"probe R2(bal7d={lp['global_balanced7d']['r2']:.3f}, trans={lp['translation']['r2']:.3f}, "
        f"rot={lp['rotation']['r2']:.3f}, mag={lp['magnitude']['r2']:.3f}, temp={lp['temporal']['r2']:.3f}, "
        f"grip={lp['gripper']['r2']:.3f}) | "
        f"Lin@100 MSE10={tc100['linear_ridge']['ddim_10d_mse']:.5f} vs Lin@1000={tc1000['linear_ridge']['ddim_10d_mse']:.5f}",
        flush=True,
    )

    summary = {
        "label": label,
        "checkpoint": str(ckpt_path),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "suite1_reconstruction": s1,
        "suite2_relational": s2,
        "suite3_displacement_arithmetic": s3,
        "suite4_unseen_combinations": s4,
        "suite5_transfer_efficiency": s5,
    }
    atomic_write_json(summary, out_dir / "eval_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="e9_baseline")
    parser.add_argument("--ddim-samples", type=int, default=2048)
    parser.add_argument("--max-rel-samples", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_full_eval(
        args.ckpt,
        args.output_dir,
        label=args.label,
        device=device,
        ddim_samples=args.ddim_samples,
        max_rel_samples=args.max_rel_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
