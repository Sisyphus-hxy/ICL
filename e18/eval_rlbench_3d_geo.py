"""E18 Pillar 2 (RLBench RGB-D 3D Geometry Study): Does True 3D Metric Scene Flow Resolve the 2D Flow Conflict?

Operates strictly on the cached RLBench-18 dataset (`e18/outputs/generalization_cache/rlbench_18/`),
where metric RGB-D depth (`front_depth`, `wrist_depth`) and camera calibration are available:
  1. Quantitative Geometry Audit on RLBench val/test clips:
       Compares 2D raw optical flow, 2D location-invariant kinematics, self-supervised Cross-View
       2D canonical motion, and TRUE 3D METRIC RGB-D SCENE FLOW (camera-frame & world-frame) on
       cross-task 3D EEF displacement alignment & 3D parallel arithmetic quadruples.
  2. Controlled Pure-Video/RGB-D Pretraining (4 strictly matched `LocalBehaviorEncoder` models,
     ZERO GT actions, identical budget 4,000 steps, batch=256, same Sharp Reuse core):
       - `rlb_no_motion`      : Pure RGB CLIP Sharp Reuse (no explicit motion loss)
       - `rlb_raw_flow_2d`    : + Raw 2D optical flow displacement loss
       - `rlb_crossview_2d`   : + Self-supervised Cross-View 2D canonical flow loss
       - `rlb_sceneflow_3d`   : + True 3D metric RGB-D scene flow displacement loss (world + cam)
  3. Unseen-Task & Cross-Task Evaluation (3 seeds, disjoint RLBench tasks):
       Measures 3D action predictability (1%/10%/100%), Tier-2 3D arithmetic cosine & ARR,
       cross-task Top-1 reuse MSE, subgoal gripper-boundary AUC, and Zero-Leakage Visual ICL!
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import LOCAL_WINDOW_LEN
from e12.action_attributes import pairwise_cosine_sim
from e12.eval_suite import _effective_rank_stats, _ridge_fit_predict, _spearman_pearson_from_vectors
from e16.video_relations import compute_clip_state_change_targets, compute_video_derived_displacement_loss

from . import GEN_CACHE_DIR, OUTPUT_ROOT
from .geometry_relations import CVGeoMotionEncoder, compute_weighted_displacement_loss, extract_location_invariant_motion
from .models import URBEModel


class RLBenchClipBank:
    """Fast GPU/CPU clip bank for RLBench-18 cached CLIP + 2D flow + 3D RGB-D scene flow."""

    def __init__(self, cache_dir: Path, device: torch.device):
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.episodes: List[Dict[str, Any]] = json.loads((self.cache_dir / "episodes.json").read_text())
        self.tasks = sorted({e["task"] for e in self.episodes})
        self.task_to_id = {t: i for i, t in enumerate(self.tasks)}
        self.ep_task_ids = np.array([self.task_to_id[e["task"]] for e in self.episodes], dtype=np.int64)
        self.ep_lengths = np.load(self.cache_dir / "ep_lengths.npy", allow_pickle=False)
        self.frame_offsets = np.load(self.cache_dir / "frame_offsets.npy", allow_pickle=False)

        self.frame_features = torch.from_numpy(np.load(self.cache_dir / "frame_features.npy", allow_pickle=False)).to(device)
        self.motion_2d = torch.from_numpy(np.load(self.cache_dir / "motion_features.npy", allow_pickle=False)).to(device)
        self.flow_3d = torch.from_numpy(np.load(self.cache_dir / "flow3d_features.npy", allow_pickle=False)).to(device)
        self.m2_mean = torch.from_numpy(np.load(self.cache_dir / "motion_mean.npy", allow_pickle=False)).to(device=device, dtype=torch.float32)
        self.m2_std = torch.from_numpy(np.load(self.cache_dir / "motion_std.npy", allow_pickle=False)).to(device=device, dtype=torch.float32)
        self.f3_mean = torch.from_numpy(np.load(self.cache_dir / "flow3d_mean.npy", allow_pickle=False)).to(device=device, dtype=torch.float32)
        self.f3_std = torch.from_numpy(np.load(self.cache_dir / "flow3d_std.npy", allow_pickle=False)).to(device=device, dtype=torch.float32)
        self.actions_7d = np.load(self.cache_dir / "actions_7d.npy", allow_pickle=False)

        # Build clip start indices (stride=2 for train, stride=4 for val/test non-tail)
        self.split_starts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for split, stride in [("train", 2), ("val", 4), ("test", 4)]:
            eps_l, frs_l = [], []
            for ep_i, row in enumerate(self.episodes):
                if row["split"] != split:
                    continue
                T = int(row["frames"])
                max_s = max(1, T - LOCAL_WINDOW_LEN + 1)
                fs = np.arange(0, max_s, stride, dtype=np.int64)
                eps_l.append(np.full(len(fs), ep_i, dtype=np.int64))
                frs_l.append(fs)
            self.split_starts[split] = (np.concatenate(eps_l), np.concatenate(frs_l))

    def gather(
        self, ep_arr: np.ndarray, fr_arr: np.ndarray, *, include_actions: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        dev = self.device
        ep_np = np.asarray(ep_arr, dtype=np.int64)
        fr_np = np.asarray(fr_arr, dtype=np.int64)
        c_ep = torch.as_tensor(ep_np, dtype=torch.long, device=dev)
        c_fr = torch.as_tensor(fr_np, dtype=torch.long, device=dev)
        c_len = torch.as_tensor(self.ep_lengths[ep_np], dtype=torch.long, device=dev)
        c_off = torch.as_tensor(self.frame_offsets[ep_np], dtype=torch.long, device=dev)

        win = torch.arange(LOCAL_WINDOW_LEN, device=dev, dtype=torch.long)[None, :]
        raw = c_fr[:, None] + win
        pad = raw >= c_len[:, None]
        clamped = torch.minimum(raw, (c_len - 1).clamp_min(0)[:, None])
        g = c_off[:, None] + clamped

        rgb = self.frame_features[g].float().masked_fill(pad[:, :, None, None], 0.0)
        m2 = ((self.motion_2d[g].float() - self.m2_mean[None, None]) / self.m2_std[None, None]).clamp(-8.0, 8.0)
        m2 = m2.masked_fill(pad[:, :, None, None], 0.0)
        f3 = ((self.flow_3d[g].float() - self.f3_mean[None, None]) / self.f3_std[None, None]).clamp(-8.0, 8.0)
        f3 = f3.masked_fill(pad[:, :, None, None], 0.0)

        a7 = None
        if include_actions:
            g_cpu = g.cpu().numpy()
            a7 = torch.from_numpy(self.actions_7d[g_cpu]).to(dev, dtype=torch.float32)

        return {
            "clip_rgb": rgb,
            "clip_pad_mask": pad,
            "clip_motion": m2,
            "clip_flow3d": f3,
            "actions_7d": a7,
            "ep_ids": c_ep,
            "task_ids": torch.as_tensor(self.ep_task_ids[ep_np], dtype=torch.long, device=dev),
        }


def _extract_3d_clip_descriptor(clip_flow3d: torch.Tensor, clip_pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """4-segment 3D metric RGB-D scene flow + cumulative world-frame 3D displacement `(B, 432)`."""
    B, L, C, D = clip_flow3d.shape
    valid = (~clip_pad_mask.bool()) if clip_pad_mask is not None else torch.ones((B, L), device=clip_flow3d.device, dtype=torch.bool)
    f = clip_flow3d.float() * valid[:, :, None, None].float()
    cnt4 = valid.reshape(B, 4, 4).float().sum(dim=2).clamp_min(1.0)
    f4 = f.reshape(B, 4, 4, C, D).sum(dim=2) / cnt4[:, :, None, None]  # (B, 4, 2, 48) -> 384D
    # Cumulative active world 3D velocity [0:3] + centroid shift [15:18] -> 6D per cam x 2 x 4 = 48D
    vel_w = torch.cat([f[:, :, :, :3], f[:, :, :, 15:18]], dim=-1)
    cum_w = torch.cumsum(vel_w, dim=1)[:, [3, 7, 11, 15]] * 0.25  # (B, 4, 2, 6)
    return torch.cat([f4.reshape(B, -1), cum_w.reshape(B, -1)], dim=-1)


def _build_rlbench_tier2_quadruples(
    A7: np.ndarray, task_ids: np.ndarray, *, n_quads: int = 160, seed: int = 1831
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build cross-task parallel 3D action arithmetic quadruples (A1, A2, B1, B2) on RLBench."""
    rng = np.random.default_rng(seed)
    N = len(A7)
    # Use cumulative 16-step position delta + final gripper sign as 3D behavioral signature
    sig = np.concatenate([A7[:, :, :3].sum(axis=1), A7[:, :, 3:6].sum(axis=1) * 0.3, A7[:, -1:, 6] * 1.5], axis=-1)
    sig_n = sig / np.linalg.norm(sig, axis=-1, keepdims=True).clip(1e-6)

    # Sample candidate pairs (A1, A2) and match cross-task (B1, B2) with high parallel 3D displacement cosine!
    n_pool = min(N, 2000)
    pool = np.sort(rng.choice(N, size=n_pool, replace=False))
    p_a1 = pool[rng.integers(0, n_pool, size=1200)]
    p_a2 = pool[rng.integers(0, n_pool, size=1200)]
    ok = (p_a1 != p_a2) & (np.linalg.norm(sig[p_a2] - sig[p_a1], axis=-1) > 0.5)
    p_a1, p_a2 = p_a1[ok][:600], p_a2[ok][:600]
    da = sig[p_a2] - sig[p_a1]
    da_n = da / np.linalg.norm(da, axis=-1, keepdims=True).clip(1e-6)

    S_da = da_n @ da_n.T
    # Require strictly different tasks between A and B pairs
    diff_t = (task_ids[p_a1][:, None] != task_ids[p_a1][None, :]) & (task_ids[p_a2][:, None] != task_ids[p_a2][None, :])
    S_da = np.where(diff_t, S_da, -2.0)
    best_j = np.argmax(S_da, axis=1)
    best_s = S_da[np.arange(len(p_a1)), best_j]
    order = np.argsort(-best_s)[:n_quads]
    return p_a1[order], p_a2[order], p_a1[best_j[order]], p_a2[best_j[order]]


def train_rlb_cv_geo(bank: RLBenchClipBank, steps: int = 2500, seed: int = 42) -> CVGeoMotionEncoder:
    seed_everything(seed)
    enc = CVGeoMotionEncoder(in_dim=384, hidden_dim=384, out_dim=128).to(bank.device)
    opt = torch.optim.AdamW(enc.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    tr_ep, tr_fr = bank.split_starts["train"]
    rng = np.random.default_rng(seed + 901)
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(tr_ep), size=256)
        b = bank.gather(tr_ep[idx], tr_fr[idx], include_actions=False)
        assert b["actions_7d"] is None
        loss, _ = enc.forward_loss(b["clip_motion"], b["clip_pad_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.inference_mode()
def run_rlbench_geometry_audit(bank: RLBenchClipBank, cv_enc: CVGeoMotionEncoder) -> Dict[str, Any]:
    """Directly compare 2D flow vs Cross-View 2D vs TRUE 3D RGB-D Scene Flow on RLBench!"""
    vep, vfr = bank.split_starts["val"]
    rng = np.random.default_rng(1832)
    sel = np.sort(rng.choice(len(vep), size=min(2048, len(vep)), replace=False))

    u2d_l, uinv_l, ucv_l, u3d_l, u3d_w_l, v0_l, a7_l = [], [], [], [], [], [], []
    for s in range(0, len(sel), 256):
        e = min(s + 256, len(sel))
        b = bank.gather(vep[sel[s:e]], vfr[sel[s:e]], include_actions=True)
        tg = compute_clip_state_change_targets(b["clip_rgb"], b["clip_pad_mask"], b["clip_motion"])
        u2d = torch.cat([tg["motion_seg"].reshape(e - s, -1), tg["motion_cum"].reshape(e - s, -1)], dim=-1)
        uinv = extract_location_invariant_motion(b["clip_motion"], b["clip_pad_mask"])
        ucv, _ = cv_enc.encode_canonical(b["clip_motion"], b["clip_pad_mask"])
        u3d = _extract_3d_clip_descriptor(b["clip_flow3d"], b["clip_pad_mask"])
        # World-frame-only compact 3D velocity (channels [0:6] + [15:18] = 9D/cam)
        f3_w = torch.cat([b["clip_flow3d"][:, :, :, :6], b["clip_flow3d"][:, :, :, 15:18]], dim=-1)
        u3d_w = f3_w.reshape(e - s, 4, 4, 2, 9).mean(dim=2).reshape(e - s, -1)
        u2d_l.append(u2d.cpu().numpy())
        uinv_l.append(uinv.cpu().numpy())
        ucv_l.append(ucv.cpu().numpy())
        u3d_l.append(u3d.cpu().numpy())
        u3d_w_l.append(u3d_w.cpu().numpy())
        v0_l.append(tg["v0"].reshape(e - s, -1).cpu().numpy())
        a7_l.append(b["actions_7d"].cpu().numpy())

    A7 = np.concatenate(a7_l, axis=0)
    V0 = np.concatenate(v0_l, axis=0)
    t_ids = bank.ep_task_ids[vep[sel]]
    A1, A2, B1, B2 = _build_rlbench_tier2_quadruples(A7, t_ids, n_quads=160, seed=1831)

    def _pcos(x: np.ndarray, y: np.ndarray) -> float:
        nx = np.linalg.norm(x, axis=-1).clip(1e-8)
        ny = np.linalg.norm(y, axis=-1).clip(1e-8)
        return float(np.mean(np.sum(x * y, axis=-1) / (nx * ny)))

    # Cross-task action similarity
    a7_f = A7.reshape(len(A7), -1)
    S_act = pairwise_cosine_sim(torch.from_numpy(a7_f[:1024]).to(bank.device)).cpu().numpy()
    S_pos = pairwise_cosine_sim(torch.from_numpy(A7[:1024, :, :3].reshape(1024, -1)).to(bank.device)).cpu().numpy()
    S_v0 = pairwise_cosine_sim(torch.from_numpy(V0[:1024]).to(bank.device)).cpu().numpy()
    ct = t_ids[:1024, None] != t_ids[None, :1024]

    da_p = torch.from_numpy(a7_f[512:1024] - a7_f[:512]).to(bank.device)
    S_da = pairwise_cosine_sim(da_p).cpu().numpy()
    off = ~np.eye(512, dtype=bool)

    feats = {
        "raw_flow_2d": np.concatenate(u2d_l, axis=0),
        "loc_invariant_2d": np.concatenate(uinv_l, axis=0),
        "cv_geo_2d": np.concatenate(ucv_l, axis=0),
        "sceneflow_3d_full": np.concatenate(u3d_l, axis=0),
        "sceneflow_3d_world": np.concatenate(u3d_w_l, axis=0),
    }
    audit = {
        "gt_3d_quad_cos": _pcos((A7[A2] - A7[A1]).reshape(len(A1), -1), (A7[B2] - A7[B1]).reshape(len(A1), -1)),
        "descriptors": {},
    }
    for name, U in feats.items():
        q_cos = _pcos(U[A2] - U[A1], U[B2] - U[B1])
        S_u = pairwise_cosine_sim(torch.from_numpy(U[:1024]).to(bank.device)).cpu().numpy()
        def _sp(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
            return _spearman_pearson_from_vectors(
                torch.as_tensor(a, dtype=torch.float32),
                torch.as_tensor(b, dtype=torch.float32),
            )

        sp_act, _ = _sp(S_u[ct], S_act[ct])
        sp_pos, _ = _sp(S_u[ct], S_pos[ct])
        sp_v0, _ = _sp(S_u[ct], S_v0[ct])
        du = torch.from_numpy(U[512:1024] - U[:512]).to(bank.device)
        sp_disp, _ = _sp(pairwise_cosine_sim(du).cpu().numpy()[off], S_da[off])
        top1_j = np.argmax(np.where(ct, S_u, -2.0), axis=1)
        top1_mse = float(np.mean((A7[:1024] - A7[top1_j]) ** 2))
        audit["descriptors"][name] = {
            "tier2_3d_quad_cos": q_cos,
            "ct_spearman_act7d": float(sp_act),
            "ct_spearman_pos3d": float(sp_pos),
            "ct_spearman_v0_leak": float(sp_v0),
            "disp_spearman_3d_act": float(sp_disp),
            "ct_top1_act7_mse": top1_mse,
        }
    return audit


def train_and_eval_rlbench_model(
    variant: str,
    bank: RLBenchClipBank,
    cv_enc: CVGeoMotionEncoder,
    out_dir: Path,
    *,
    steps: int = 4000,
    batch_size: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train one pure-video/RGB-D encoder on RLBench (ZERO actions) and evaluate 3D editability/reuse/transfer."""
    seed_everything(seed)
    device = bank.device
    model = URBEModel(
        compact_manifold=True, op_dim=384, use_reuse=True, top_k_donors=16,
        top1_reuse_weight=0.40, init_temp=0.07, coverage_mode="cross_demo_unmasked",
        predict_motion=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.95, 0.999), weight_decay=1e-5)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    ema = EMAModel(model.parameters(), decay=0.995, min_decay=0.0, update_after_step=0, foreach=True).to(device)

    tr_ep, tr_fr = bank.split_starts["train"]
    rng = np.random.default_rng(seed + 1833)
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type=device.type, enabled=False)

    t0 = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        idx = rng.integers(0, len(tr_ep), size=batch_size)
        batch = bank.gather(tr_ep[idx], tr_fr[idx], include_actions=False)
        assert batch["actions_7d"] is None
        warm = min(1.0, step / 300.0)
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = model.encode_local_behavior(batch["clip_rgb"], batch["clip_pad_mask"])
        lam_loss, _ = model.compute_lam_loss(b_t, batch["clip_rgb"], batch["clip_pad_mask"])

        z_op = b_t[:, :, :384]
        reu_loss, _ = model.reuse_module.forward_loss(
            z_op, z_op, batch["clip_rgb"], batch["clip_pad_mask"], batch["ep_ids"],
            encoder_fn=model.encode_local_behavior if step > 300 else None,
            extract_pred_fn=(lambda b: b[:, :, :384]),
            use_cycle=(step > 300), use_equiv=(step > 300),
        )
        vid_loss = lam_loss.new_zeros(())
        if variant == "rlb_raw_flow_2d":
            tg = compute_clip_state_change_targets(batch["clip_rgb"], batch["clip_pad_mask"], batch["clip_motion"])
            u_m = torch.cat([tg["motion_seg"].reshape(batch_size, -1), tg["motion_cum"].reshape(batch_size, -1)], dim=-1)
            vid_loss, _ = compute_video_derived_displacement_loss(z_op, u_m)
        elif variant == "rlb_crossview_2d":
            with torch.no_grad():
                u_cv, ag = cv_enc.encode_canonical(batch["clip_motion"], batch["clip_pad_mask"])
                pw = torch.sigmoid((ag - 0.15) / 0.20)
            vid_loss, _ = compute_weighted_displacement_loss(z_op, u_cv, pair_weights=pw)
        elif variant == "rlb_sceneflow_3d":
            u_3d = _extract_3d_clip_descriptor(batch["clip_flow3d"], batch["clip_pad_mask"])
            vid_loss, _ = compute_video_derived_displacement_loss(z_op, u_3d)

        loss = lam_loss + (0.25 * warm) * reu_loss + (0.25 * warm) * vid_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()
        ema.step(model.parameters())

    ema.copy_to(model.parameters())
    model.eval()

    # Evaluate on held-out RLBench val + test splits!
    vep, vfr = bank.split_starts["val"]
    tep, tfr = bank.split_starts["test"]
    eval_ep = np.concatenate([vep, tep])
    eval_fr = np.concatenate([vfr, tfr])
    rng_e = np.random.default_rng(1834)
    sel = np.sort(rng_e.choice(len(eval_ep), size=min(3072, len(eval_ep)), replace=False))
    z_l, v0_l, a7_l = [], [], []
    with torch.inference_mode():
        for s in range(0, len(sel), 256):
            e = min(s + 256, len(sel))
            b = bank.gather(eval_ep[sel[s:e]], eval_fr[sel[s:e]], include_actions=True)
            with amp_ctx:
                bt = model.encode_local_behavior(b["clip_rgb"], b["clip_pad_mask"])
            z_l.append(bt.float().cpu().numpy())
            v0_l.append(b["clip_rgb"][:, 0].reshape(e - s, -1).cpu().numpy())
            a7_l.append(b["actions_7d"].cpu().numpy())
    Z4 = np.concatenate(z_l, axis=0)
    Z_full = Z4.reshape(len(sel), -1)
    Z_op = Z4[:, :, :384].reshape(len(sel), -1)
    V0 = np.concatenate(v0_l, axis=0)
    A7 = np.concatenate(a7_l, axis=0)
    A7_f = A7.reshape(len(sel), -1)
    t_ids = bank.ep_task_ids[eval_ep[sel]]

    # 3-seed task-disjoint unseen-task probing & Ridge-decoded 3D Tier-2 arithmetic + Zero-Leakage Visual ICL!
    unique_t = np.unique(t_ids)
    r2_100, r2_10, r2_op, icl_r2_z, icl_mse = [], [], [], [], []
    t2_cos_full, t2_cos_op, t2_arr_full, t2_arr_op = [], [], [], []

    for sd in (42, 123, 777):
        r_s = np.random.default_rng(sd + 1835)
        perm = r_s.permutation(unique_t)
        tr_t = set(perm[:12].tolist())
        tr_i = np.flatnonzero(np.array([int(t) in tr_t for t in t_ids]))
        te_i = np.flatnonzero(np.array([int(t) not in tr_t for t in t_ids]))
        var_a = float(((A7_f[te_i] - A7_f[te_i].mean(0, keepdims=True)) ** 2).mean())

        p100, al100 = _ridge_fit_predict(Z_full[tr_i], A7_f[tr_i], Z_full[te_i], device=device)
        r2_100.append(float(1.0 - np.mean((p100 - A7_f[te_i]) ** 2) / max(var_a, 1e-12)))

        sub10 = tr_i[r_s.choice(len(tr_i), size=max(16, int(0.1 * len(tr_i))), replace=False)]
        p10, _ = _ridge_fit_predict(Z_full[sub10], A7_f[sub10], Z_full[te_i], device=device)
        r2_10.append(float(1.0 - np.mean((p10 - A7_f[te_i]) ** 2) / max(var_a, 1e-12)))

        pop, _ = _ridge_fit_predict(Z_op[tr_i], A7_f[tr_i], Z_op[te_i], device=device)
        r2_op.append(float(1.0 - np.mean((pop - A7_f[te_i]) ** 2) / max(var_a, 1e-12)))

        # Unseen-task Tier-2 3D arithmetic decoded via training-fitted linear action decoder!
        A1, A2, B1, B2 = _build_rlbench_tier2_quadruples(A7[te_i], t_ids[te_i], n_quads=120, seed=sd)
        Z4_te = Z4[te_i]
        zf_te = Z_full[te_i]
        zo_te = Z_op[te_i]
        dzA_f, dzB_f = zf_te[A2] - zf_te[A1], zf_te[B2] - zf_te[B1]
        dzA_o, dzB_o = zo_te[A2] - zo_te[A1], zo_te[B2] - zo_te[B1]
        t2_cos_full.append(float(np.mean(np.sum(dzA_f * dzB_f, -1) / (np.linalg.norm(dzA_f, axis=-1) * np.linalg.norm(dzB_f, axis=-1)).clip(1e-8))))
        t2_cos_op.append(float(np.mean(np.sum(dzA_o * dzB_o, -1) / (np.linalg.norm(dzA_o, axis=-1) * np.linalg.norm(dzB_o, axis=-1)).clip(1e-8))))

        z_ed_f = zf_te[B1] + dzA_f
        z_ed_o = Z4_te[B1].copy()
        z_ed_o[:, :, :384] += (Z4_te[A2, :, :384] - Z4_te[A1, :, :384])
        z_ed_of = z_ed_o.reshape(len(B1), -1)
        gt_b2 = A7_f[te_i[B2]]
        # Decode oracle, base, full-arith, op-arith via shared Ridge action decoder trained on tr_i
        dec_in = np.concatenate([zf_te[B2], zf_te[B1], z_ed_f, z_ed_of], axis=0)
        dec_out, _ = _ridge_fit_predict(Z_full[tr_i], A7_f[tr_i], dec_in, alphas=(al100,), device=device)
        m_q = len(B1)
        mse_ora = float(np.mean((dec_out[:m_q] - gt_b2) ** 2))
        mse_bas = float(np.mean((dec_out[m_q : 2 * m_q] - gt_b2) ** 2))
        mse_ful = float(np.mean((dec_out[2 * m_q : 3 * m_q] - gt_b2) ** 2))
        mse_op = float(np.mean((dec_out[3 * m_q :] - gt_b2) ** 2))
        denom = max(mse_bas - mse_ora, 1e-8)
        t2_arr_full.append((mse_bas - mse_ful) / denom)
        t2_arr_op.append((mse_bas - mse_op) / denom)

        # Zero-Leakage Visual ICL (v0 -> z_op on unseen RLBench tasks -> Ridge action decode)
        z_hat, _ = _ridge_fit_predict(V0[tr_i], Z_full[tr_i], V0[te_i], device=device)
        var_z = float(((Z_full[te_i] - Z_full[te_i].mean(0, keepdims=True)) ** 2).mean())
        icl_r2_z.append(float(1.0 - np.mean((z_hat - Z_full[te_i]) ** 2) / max(var_z, 1e-12)))
        a_hat, _ = _ridge_fit_predict(Z_full[tr_i], A7_f[tr_i], z_hat, alphas=(al100,), device=device)
        icl_mse.append(float(np.mean((a_hat - A7_f[te_i]) ** 2)))

    # Cross-task Top-1 reuse action MSE (z_op-NN vs random)
    S_op = pairwise_cosine_sim(torch.from_numpy(Z_op[:1024]).to(device)).cpu().numpy()
    ct = t_ids[:1024, None] != t_ids[None, :1024]
    top1_j = np.argmax(np.where(ct, S_op, -2.0), axis=1)
    ct_top1_mse = float(np.mean((A7[:1024] - A7[top1_j]) ** 2))
    rand_j = np.array([rng_e.choice(np.flatnonzero(ct[i])) for i in range(1024)])
    ct_rand_mse = float(np.mean((A7[:1024] - A7[rand_j]) ** 2))

    er_f = _effective_rank_stats(Z_full, device=device)["erank_energy"]
    er_o = _effective_rank_stats(Z_op, device=device)["erank_energy"]

    res = {
        "variant": variant,
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "r2_a7_100": float(np.mean(r2_100)),
        "r2_a7_10": float(np.mean(r2_10)),
        "r2_op_a7": float(np.mean(r2_op)),
        "tier2_cos_full": float(np.mean(t2_cos_full)),
        "tier2_cos_op": float(np.mean(t2_cos_op)),
        "tier2_arr_full": float(np.mean(t2_arr_full)),
        "tier2_arr_op": float(np.mean(t2_arr_op)),
        "ct_top1_mse": ct_top1_mse,
        "ct_err_reduction_pct": float(100.0 * (ct_rand_mse - ct_top1_mse) / max(ct_rand_mse, 1e-8)),
        "vis_pred_r2_z": float(np.mean(icl_r2_z)),
        "vis_pred_a7_mse": float(np.mean(icl_mse)),
        "erank_full": float(er_f),
        "erank_op": float(er_o),
    }
    print(
        f"[RLB-3D:{variant:18s}] R2_a7={res['r2_a7_100']:.4f} | T2_ARR(full/op)={res['tier2_arr_full']*100:.1f}%/{res['tier2_arr_op']*100:.1f}% | "
        f"ReuseRed={res['ct_err_reduction_pct']:.1f}% | VisR2z={res['vis_pred_r2_z']:.4f} | erank={er_f:.1f}/{er_o:.1f}",
        flush=True,
    )
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=GEN_CACHE_DIR / "rlbench_18")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "pillar2_rlbench_3d")
    p.add_argument("--steps", type=int, default=4000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank = RLBenchClipBank(args.cache_dir, device)
    cv_enc = train_rlb_cv_geo(bank, steps=2500, seed=42)
    audit = run_rlbench_geometry_audit(bank, cv_enc)

    models_out = {}
    for var in ("rlb_no_motion", "rlb_raw_flow_2d", "rlb_crossview_2d", "rlb_sceneflow_3d"):
        models_out[var] = train_and_eval_rlbench_model(var, bank, cv_enc, args.output_dir, steps=args.steps, seed=42)

    atomic_write_json({"geometry_audit": audit, "models": models_out}, args.output_dir / "rlbench_3d_geo_summary.json")


if __name__ == "__main__":
    main()
