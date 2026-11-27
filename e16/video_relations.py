"""Self-supervised video-derived and learnable reuse-driven relational modules for E16.

Implements:
  1. `VideoMotionLocalBank`: Extends `LocalClipFeatureBank` with aligned 20Hz pixel-space
     optical flow & grid point-track descriptors `(3629339, 2, 64)`, episode/task IDs,
     and optional episode subsampling for data-scale & label-efficiency studies.
  2. `extract_video_relation_features`: Purely video-derived clip descriptors (ZERO actions):
       - `feat_delta`        : scene-subtracted dual-camera temporal CLIP deltas
       - `feat_delta_decorr` : static-scene-decorrelated temporal CLIP deltas
       - `motion_flow`       : pixel-space optical flow + integrated 4x4 grid point-tracks
       - `pretrained_lam`    : frozen 100% video-only pretrained LAM dynamics latent
       - `multiview_vid`     : standardized fusion of `motion_flow` + `pretrained_lam`
       - `random_rel`        : matched-dimension random per-clip hash (strict control)
  3. `compute_video_derived_displacement_loss`: E16-A displacement direction + scale
     consistency purely from video-derived descriptors (on full `b_t` or `z_op`).
  4. `LearnableReuseRelationModule`: E16-B/C/D learnable cross-demo relation module:
       - Strict cross-episode + anti-scene-matching relation graph `W_ij`
       - Column marginal load-balancing (anti-hub-collapse)
       - Conditional forward state-change predictor `F_psi(v_0, z_rel)`
       - Cross-demo behavior substitution reuse-value loss `L_reuse`
       - Counterfactual cross-scene cycle consistency `L_cycle`
       - Scene-canonicalized behavioral effect displacement equivariance `L_equiv`
       - VICReg variance + covariance anti-collapse regularization
       - Supports both global reuse (`token_wise=False`) and 4-token autonomous
         factorized reuse (`token_wise=True`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e9 import DEFAULT_DENSE_CACHE_DIR, E5_CACHE_DIR, LOCAL_WINDOW_LEN
from e9.data import LocalClipFeatureBank
from e12.action_attributes import pairwise_cosine_sim
from e15.train_e15 import compute_linear_cka_penalty

from . import MOTION_CACHE_DIR, MOTION_FEAT_DIM


def _off_diag_mask(n: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(n, device=device, dtype=torch.bool)


class VideoMotionLocalBank(LocalClipFeatureBank):
    """GPU/CPU feature bank yielding dense CLIP clips, pixel motion-flow trajectories,
    episode/task IDs, and (only when requested) GT actions."""

    def __init__(
        self,
        e5_cache_dir: Path = E5_CACHE_DIR,
        dense_cache_dir: Path = DEFAULT_DENSE_CACHE_DIR,
        motion_cache_dir: Optional[Path] = MOTION_CACHE_DIR,
        *,
        device: torch.device = torch.device("cpu"),
        resident_on_device: bool = False,
        seed: int = 42,
        train_episode_fraction: float = 1.0,
        subset_seed: int = 42,
    ):
        super().__init__(
            e5_cache_dir=e5_cache_dir,
            dense_cache_dir=dense_cache_dir,
            device=device,
            resident_on_device=resident_on_device,
            seed=seed,
        )
        # Task IDs per episode
        tasks_sorted = sorted({e["task"] for e in self.episodes})
        self.task_to_id = {t: i for i, t in enumerate(tasks_sorted)}
        self.ep_task_ids_np = np.array(
            [self.task_to_id[e["task"]] for e in self.episodes], dtype=np.int64
        )

        # Optional deterministic episode subsampling (for data-scale & label-efficiency studies!)
        self.train_episode_fraction = float(train_episode_fraction)
        if self.train_episode_fraction < 0.9999:
            rng = np.random.default_rng(subset_seed + 16001)
            # Stratified by task so every task keeps at least 1 episode when possible
            by_task: Dict[int, List[int]] = defaultdict(list)
            for ep_idx, row in enumerate(self.episodes):
                if row["fold"] == "train":
                    by_task[self.ep_task_ids_np[ep_idx]].append(ep_idx)
            kept_eps: set = set()
            all_tr_eps = sorted(ep for eps_l in by_task.values() for ep in eps_l)
            target_n = max(1, int(round(len(all_tr_eps) * self.train_episode_fraction)))
            if target_n >= len(by_task):
                # Stratified selection
                for tid, eps_l in sorted(by_task.items()):
                    arr = np.array(eps_l, dtype=np.int64)
                    rng.shuffle(arr)
                    k = max(1, int(round(len(arr) * self.train_episode_fraction)))
                    kept_eps.update(arr[:k].tolist())
            else:
                perm = rng.permutation(all_tr_eps)
                kept_eps.update(perm[:target_n].tolist())
            mask = np.array([int(ep) in kept_eps for ep in self.train_episodes], dtype=bool)
            self.train_episodes = self.train_episodes[mask]
            self.train_frames = self.train_frames[mask]
            self.kept_train_episode_ids = sorted(kept_eps)
        else:
            self.kept_train_episode_ids = [
                i for i, e in enumerate(self.episodes) if e["fold"] == "train"
            ]

        # Optional pixel-space motion cache (929 MB fp16)
        self.has_motion = False
        self.motion_features: Optional[torch.Tensor] = None
        self.motion_mean: Optional[torch.Tensor] = None
        self.motion_std: Optional[torch.Tensor] = None
        if motion_cache_dir is not None and (Path(motion_cache_dir) / "motion_features.npy").is_file():
            m_arr = np.load(Path(motion_cache_dir) / "motion_features.npy", allow_pickle=False)
            m_mean = np.load(Path(motion_cache_dir) / "motion_mean.npy", allow_pickle=False)
            m_std = np.load(Path(motion_cache_dir) / "motion_std.npy", allow_pickle=False)
            self.motion_features = torch.from_numpy(m_arr).to(self.storage_device)
            self.motion_mean = torch.from_numpy(m_mean).to(device=self.device, dtype=torch.float32)
            self.motion_std = torch.from_numpy(m_std).to(device=self.device, dtype=torch.float32)
            self.has_motion = True

    def gather_rich_batch(
        self,
        clip_episodes: np.ndarray | torch.Tensor,
        clip_frames: np.ndarray | torch.Tensor,
        *,
        include_actions: bool = True,
        out_dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Optional[torch.Tensor]]:
        dev = self.storage_device
        ep_np = np.asarray(clip_episodes, dtype=np.int64)
        c_ep = torch.as_tensor(ep_np, dtype=torch.long, device=dev)
        c_fr = torch.as_tensor(clip_frames, dtype=torch.long, device=dev)

        c_len = self.ep_lengths[c_ep]
        c_off = self.frame_offsets[c_ep]
        win_offsets = torch.arange(LOCAL_WINDOW_LEN, device=dev, dtype=torch.long)[None, :]
        raw_steps = c_fr[:, None] + win_offsets
        clip_pad_mask = raw_steps >= c_len[:, None]
        clamped_steps = torch.minimum(raw_steps, (c_len - 1).clamp_min(0)[:, None])
        g_idx = c_off[:, None] + clamped_steps

        clip_rgb = self.frame_features[g_idx].to(dtype=out_dtype)
        clip_rgb = clip_rgb.masked_fill(clip_pad_mask[:, :, None, None], 0.0)

        clip_motion: Optional[torch.Tensor] = None
        if self.has_motion and self.motion_features is not None:
            cm = self.motion_features[g_idx].to(dtype=out_dtype)
            cm = cm.masked_fill(clip_pad_mask[:, :, None, None], 0.0)
            clip_motion = cm

        actions: Optional[torch.Tensor] = None
        if include_actions:
            actions = self.actions_10d[g_idx]

        ep_ids = torch.as_tensor(ep_np, dtype=torch.long, device=self.device)
        task_ids = torch.as_tensor(self.ep_task_ids_np[ep_np], dtype=torch.long, device=self.device)
        frame_ids = torch.as_tensor(np.asarray(clip_frames, dtype=np.int64), dtype=torch.long, device=self.device)

        if dev != self.device:
            clip_rgb = clip_rgb.to(self.device, non_blocking=True)
            clip_pad_mask = clip_pad_mask.to(self.device, non_blocking=True)
            if clip_motion is not None:
                clip_motion = clip_motion.to(self.device, non_blocking=True)
            if actions is not None:
                actions = actions.to(self.device, non_blocking=True)

        if clip_motion is not None and self.motion_mean is not None and self.motion_std is not None:
            # Standardize motion descriptors cleanly (masked pads stay 0)
            clip_motion = ((clip_motion - self.motion_mean[None, None]) / self.motion_std[None, None]).clamp(-8.0, 8.0)
            clip_motion = clip_motion.masked_fill(clip_pad_mask[:, :, None, None], 0.0)

        return {
            "clip_rgb": clip_rgb,
            "clip_pad_mask": clip_pad_mask,
            "clip_motion": clip_motion,
            "actions_10d": actions,
            "ep_ids": ep_ids,
            "task_ids": task_ids,
            "frame_ids": frame_ids,
        }


class RichPairedBatchSampler:
    """Paired transition batch sampler returning rich dicts (with zero-action option)."""

    def __init__(
        self,
        bank: VideoMotionLocalBank,
        within_ep_frac: float = 0.5,
        seed: int = 42,
        include_actions: bool = True,
    ):
        self.bank = bank
        self.within_ep_frac = float(within_ep_frac)
        self.seed = int(seed)
        self.include_actions = bool(include_actions)

        ep_groups: Dict[int, List[int]] = defaultdict(list)
        for idx, ep in enumerate(bank.train_episodes):
            ep_groups[int(ep)].append(idx)

        self.multi_eps: List[np.ndarray] = []
        self.multi_frames: List[np.ndarray] = []
        for ep, idxs in ep_groups.items():
            if len(idxs) >= 2:
                arr = np.array(idxs, dtype=np.int64)
                frs = bank.train_frames[arr]
                if int(frs.max() - frs.min()) >= 16:
                    self.multi_eps.append(arr)
                    self.multi_frames.append(frs)
        self.num_multi_eps = len(self.multi_eps)

    def iter_batches(
        self,
        epoch: int,
        batch_size: int,
        max_steps: Optional[int] = None,
    ) -> Iterator[Dict[str, Optional[torch.Tensor]]]:
        if batch_size % 2 != 0:
            raise ValueError(f"Paired batch_size must be even, got {batch_size}")
        M = batch_size // 2
        n_within = int(round(M * self.within_ep_frac)) if self.num_multi_eps > 0 else 0
        n_cross = M - n_within

        rng = np.random.default_rng([self.seed, int(epoch), 16016])
        n_train = self.bank.num_train_samples()
        total_steps = math.ceil(n_train / batch_size)
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))

        for _ in range(total_steps):
            idx_a = np.empty(M, dtype=np.int64)
            idx_b = np.empty(M, dtype=np.int64)
            if n_within > 0:
                ep_choices = rng.choice(
                    self.num_multi_eps, size=n_within, replace=(n_within > self.num_multi_eps)
                )
                for k, e_i in enumerate(ep_choices):
                    arr = self.multi_eps[e_i]
                    frs = self.multi_frames[e_i]
                    i0 = int(rng.integers(len(arr)))
                    f0 = int(frs[i0])
                    far = np.flatnonzero(np.abs(frs - f0) >= 16)
                    i1 = int(far[rng.integers(len(far))]) if len(far) > 0 else int(rng.integers(len(arr)))
                    idx_a[k] = arr[i0]
                    idx_b[k] = arr[i1]
            if n_cross > 0:
                idx_a[n_within:] = rng.integers(0, n_train, size=n_cross)
                idx_b[n_within:] = rng.integers(0, n_train, size=n_cross)

            batch_idx = np.empty(batch_size, dtype=np.int64)
            batch_idx[0::2] = idx_a
            batch_idx[1::2] = idx_b
            eps = self.bank.train_episodes[batch_idx]
            frs = self.bank.train_frames[batch_idx]
            yield self.bank.gather_rich_batch(eps, frs, include_actions=self.include_actions)


# ---------------------------------------------------------------------------
# Purely Video-Derived State-Change Targets & Descriptors (ZERO GT Actions!)
# ---------------------------------------------------------------------------

def compute_clip_state_change_targets(
    clip_rgb: torch.Tensor,
    clip_pad_mask: Optional[torch.Tensor] = None,
    clip_motion: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute 4-segment temporal visual state-change targets purely from video clips.

    Returns:
      - `v0`: `(B, 2, 768)` initial dual-camera frame features
      - `delta_v_seg`: `(B, 4, 2, 768)` scene-subtracted visual state change in 4 segments
        (`t+0..3`, `t+4..7`, `t+8..11`, `t+12..15` minus `v0`)
      - `seg_valid`: `(B, 4)` float weight mask (0 if segment is completely padded)
      - `motion_seg`: optional `(B, 4, 2, 64)` segment-averaged normalized pixel flow
      - `motion_cum`: optional `(B, 4, 2, 32)` cumulative integrated 4x4 grid point-tracks
    """
    x = clip_rgb.float()  # (B, 16, 2, 768)
    B, L, C, D = x.shape
    v0 = x[:, 0]  # (B, 2, 768)
    valid = (
        (~clip_pad_mask.to(dtype=torch.bool))
        if clip_pad_mask is not None
        else torch.ones((B, L), device=x.device, dtype=torch.bool)
    )
    dv = (x - v0[:, None, :, :]) * valid[:, :, None, None].float()  # (B, 16, 2, 768)

    # Reshape into 4 temporal chunks of 4 steps each
    dv4 = dv.reshape(B, 4, 4, C, D)
    val4 = valid.reshape(B, 4, 4).float()
    cnt4 = val4.sum(dim=2).clamp_min(1.0)  # (B, 4)
    seg_valid = (val4.sum(dim=2) > 0).float()  # (B, 4)
    delta_v_seg = dv4.sum(dim=2) / cnt4[:, :, None, None]  # (B, 4, 2, 768)

    out: Dict[str, torch.Tensor] = {
        "v0": v0,
        "delta_v_seg": delta_v_seg,
        "seg_valid": seg_valid,
    }
    if clip_motion is not None:
        m = clip_motion.float() * valid[:, :, None, None].float()  # (B, 16, 2, 64)
        m4 = m.reshape(B, 4, 4, C, MOTION_FEAT_DIM)
        motion_seg = m4.sum(dim=2) / cnt4[:, :, None, None]  # (B, 4, 2, 64)
        # Integrated 4x4 spatial grid point-track cumulative displacement at t=3,7,11,15
        grid_flow = m[:, :, :, :32]  # (B, 16, 2, 32)
        cum_tracks = torch.cumsum(grid_flow, dim=1)[:, [3, 7, 11, 15]] * 0.25  # (B, 4, 2, 32)
        out["motion_seg"] = motion_seg
        out["motion_cum"] = cum_tracks
    return out


def _batch_decorrelate_from_scene(u: torch.Tensor, v0: torch.Tensor, ridge: float = 1e-1) -> torch.Tensor:
    """Project out linear dependence of video descriptor `u` on static initial scene `v0` within batch."""
    B = u.shape[0]
    X = v0.float().reshape(B, -1)
    X = X - X.mean(dim=0, keepdim=True)
    # Use dual ridge (B x B, fast for B=256)
    Xc = X / (X.norm(dim=-1, keepdim=True).mean().clamp_min(1e-4))
    K = Xc @ Xc.T
    K.diagonal().add_(ridge * B)
    U = u.float()
    Uc = U - U.mean(dim=0, keepdim=True)
    alpha = torch.linalg.solve(K, Uc)
    proj = (Xc @ Xc.T) @ alpha
    return Uc - proj


@torch.no_grad()
def extract_video_relation_features(
    source: str,
    clip_rgb: torch.Tensor,
    clip_pad_mask: Optional[torch.Tensor] = None,
    clip_motion: Optional[torch.Tensor] = None,
    *,
    lam_encoder: Optional[nn.Module] = None,
    ep_ids: Optional[torch.Tensor] = None,
    frame_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract purely video-derived (ZERO GT action) behavioral relation descriptor `(B, D_u)`.

    Sources:
      - `feat_delta`        : 4-segment temporal CLIP feature delta (scene-subtracted)
      - `feat_delta_decorr` : temporal CLIP delta with batch linear static-scene projection removed
      - `motion_flow`       : pixel-space optical flow + integrated 4x4 grid point-tracks
      - `pretrained_lam`    : frozen 100% video-only pretrained LAM dynamics latent
      - `multiview_vid`     : RMS-balanced concat of `motion_flow` + `pretrained_lam`
      - `random_rel`        : deterministic random Gaussian embedding per clip (strict control)
    """
    B = clip_rgb.shape[0]
    dev = clip_rgb.device

    if source == "random_rel":
        # Deterministic per-clip random Gaussian vector (zero behavioral meaning!)
        if ep_ids is not None and frame_ids is not None:
            keys = (ep_ids.long() * 1000003 + frame_ids.long() * 9176 + 1337).cpu().numpy()
            out = np.empty((B, 128), dtype=np.float32)
            for i, k in enumerate(keys):
                rng = np.random.default_rng(int(k) & 0xFFFFFFFF)
                out[i] = rng.standard_normal(128).astype(np.float32)
            return torch.from_numpy(out).to(dev)
        return torch.randn((B, 128), device=dev, dtype=torch.float32)

    tgts = compute_clip_state_change_targets(clip_rgb, clip_pad_mask, clip_motion)

    if source == "feat_delta":
        # 4-segment CLIP state changes (B, 4*2*768=6144) + frame-to-frame velocity summary
        u = tgts["delta_v_seg"].reshape(B, -1)
        return u

    if source == "feat_delta_decorr":
        u_raw = tgts["delta_v_seg"].reshape(B, -1)
        return _batch_decorrelate_from_scene(u_raw, tgts["v0"], ridge=0.05)

    if source == "motion_flow":
        if "motion_seg" not in tgts:
            raise RuntimeError("motion_flow source requires clip_motion cache!")
        # Combine 4-segment optical flow descriptors + cumulative 4x4 grid point-tracks
        ms = tgts["motion_seg"].reshape(B, -1)  # 512D
        mc = tgts["motion_cum"].reshape(B, -1)  # 256D
        return torch.cat([ms, mc], dim=-1)  # 768D

    if source == "pretrained_lam":
        if lam_encoder is None:
            raise RuntimeError("pretrained_lam source requires frozen lam_encoder!")
        lam_encoder.eval()
        z_lam = lam_encoder.encode_dynamics_latent(clip_rgb, clip_pad_mask)
        return z_lam.float().reshape(B, -1)

    if source == "multiview_vid":
        u_m = extract_video_relation_features(
            "motion_flow", clip_rgb, clip_pad_mask, clip_motion
        )
        u_l = extract_video_relation_features(
            "pretrained_lam", clip_rgb, clip_pad_mask, clip_motion, lam_encoder=lam_encoder
        )
        u_m = u_m / u_m.float().square().mean().sqrt().clamp_min(1e-5)
        u_l = u_l / u_l.float().square().mean().sqrt().clamp_min(1e-5)
        return torch.cat([u_m, u_l], dim=-1)

    raise ValueError(f"Unknown video relation source: {source}")


def compute_video_derived_displacement_loss(
    z_target: torch.Tensor,
    u_video: torch.Tensor,
    *,
    norm_weight: float = 0.25,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """E16-A / E16-C self-supervised displacement consistency loss (ZERO GT actions).

    Aligns pairwise latent displacements `Δz_m = z_{2m+1} - z_{2m}` (both direction cosine
    and relative norm scale) against purely video-derived behavioral displacements
    `Δu_m = u_{2m+1} - u_{2m}`.
    """
    B = z_target.shape[0]
    M = B // 2
    if M < 4:
        zero = z_target.sum() * 0.0
        return zero, {"rel_loss": zero.detach()}

    z = z_target[: 2 * M].float().reshape(M, 2, -1)
    u = u_video[: 2 * M].float().detach().reshape(M, 2, -1)

    dz = z[:, 1, :] - z[:, 0, :]
    du = u[:, 1, :] - u[:, 0, :]

    with torch.no_grad():
        du_norm = torch.linalg.norm(du, dim=-1)
        thresh = du_norm.median() * 0.25
        active = du_norm > thresh.clamp_min(1e-4)
        S_du = pairwise_cosine_sim(du)

    S_dz = pairwise_cosine_sim(dz)
    off = _off_diag_mask(M, z.device)
    valid = off & active[:, None] & active[None, :]
    if int(valid.sum().item()) > 0:
        dir_loss = F.mse_loss(S_dz[valid], S_du[valid])
    else:
        dir_loss = F.mse_loss(S_dz[off], S_du[off])

    dz_norm = torch.linalg.norm(dz, dim=-1)
    dz_rel = dz_norm / dz_norm.mean().clamp_min(eps)
    du_rel = du_norm / du_norm.mean().clamp_min(eps)
    scale_loss = F.mse_loss(dz_rel, du_rel)

    total = dir_loss + float(norm_weight) * scale_loss
    return total, {
        "rel_loss": total.detach(),
        "disp_dir_mse": dir_loss.detach(),
        "disp_scale_mse": scale_loss.detach(),
        "dz_norm_mean": dz_norm.detach().mean(),
        "active_pair_frac": valid.float().mean().detach(),
    }


# ---------------------------------------------------------------------------
# Anti-Collapse Regularization (VICReg Variance + Covariance)
# ---------------------------------------------------------------------------

def compute_vicreg_loss(
    z: torch.Tensor,
    *,
    std_target: float = 0.85,
    cov_weight: float = 0.04,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variance + off-diagonal covariance regularization to strictly prevent collapse."""
    B = z.shape[0]
    x = z.float().reshape(B, -1)
    # Subsample up to 512 channels if very high-dim so cov matrix is fast & exact
    D = x.shape[1]
    if D > 512:
        # Reshape into groups of 512 and average cov across groups (covers ALL channels!)
        n_grp = D // 512
        xg = x[:, : n_grp * 512].reshape(B, n_grp, 512)
        xg = xg - xg.mean(dim=0, keepdim=True)
        std = torch.sqrt(xg.var(dim=0, unbiased=False) + eps)
        var_loss = F.relu(std_target - std).mean()
        cov_losses = []
        eye = torch.eye(512, device=x.device, dtype=torch.bool)
        for g in range(n_grp):
            c = (xg[:, g].T @ xg[:, g]) / max(B - 1, 1)
            cov_losses.append(c[~eye].square().mean())
        cov_loss = torch.stack(cov_losses).mean()
    else:
        xc = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(xc.var(dim=0, unbiased=False) + eps)
        var_loss = F.relu(std_target - std).mean()
        c = (xc.T @ xc) / max(B - 1, 1)
        eye = torch.eye(D, device=x.device, dtype=torch.bool)
        cov_loss = c[~eye].square().mean()
    return var_loss + cov_weight * cov_loss, var_loss.detach(), cov_loss.detach()


# ---------------------------------------------------------------------------
# E16-B / E16-C / E16-D: Learnable Reuse-Driven Relational Behavior Module
# ---------------------------------------------------------------------------

class ConditionalStateChangePredictor(nn.Module):
    """LAM-style conditional forward dynamics model `F_psi(v_0, z) -> (delta_v_seg, motion_seg)`.

    Conditioned on initial dual-camera frame `v_0` `(B, 2, 768)` and behavior latent `z`
    `(B, D_z)`, predicts the 4-segment visual state change `delta_v_seg` `(B, 4, 2, 768)`
    and optional 4-segment motion summary `(B, 4, 2, 64)` via FiLM modulation so `v_0`
    supplies the scene context while `z` exclusively supplies the behavioral operator!
    """

    def __init__(
        self,
        z_dim: int,
        scene_dim: int = 2 * 768,
        hidden_dim: int = 1024,
        predict_motion: bool = True,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.scene_dim = int(scene_dim)
        self.hidden_dim = int(hidden_dim)
        self.predict_motion = bool(predict_motion)

        self.scene_enc = nn.Sequential(
            nn.LayerNorm(scene_dim),
            nn.Linear(scene_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.z_enc = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
        )
        # FiLM modulation of scene features by behavior latent z
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        # Output 4 temporal segments of dual-camera CLIP delta (4 * 2 * 768 = 6144)
        self.delta_v_head = nn.Linear(hidden_dim, 4 * 2 * 768)
        if self.predict_motion:
            self.motion_head = nn.Linear(hidden_dim, 4 * 2 * MOTION_FEAT_DIM)
        else:
            self.motion_head = None

    def forward(
        self, v0: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = v0.shape[0]
        hs = self.scene_enc(v0.float().reshape(B, -1))
        hz = self.z_enc(z.float().reshape(B, -1))
        gamma, beta = self.film(hz).chunk(2, dim=-1)
        h = self.trunk(hs * (1.0 + 0.5 * torch.tanh(gamma)) + beta + hz)
        dv = self.delta_v_head(h).reshape(B, 4, 2, 768)
        mo = (
            self.motion_head(h).reshape(B, 4, 2, MOTION_FEAT_DIM)
            if self.motion_head is not None
            else None
        )
        return dv, mo


class LearnableReuseRelationModule(nn.Module):
    """Learns cross-demo behavioral relations purely from reuse utility (E16-B / C / D).

    Given a mini-batch of clips from diverse episodes:
      1. Computes a learnable cross-demo relation graph `W_ij` using projected query/key
         embeddings of `z_rel` (either global or 4-token-wise), strictly masked to
         prevent same-episode and high-scene-similarity shortcuts (configurable via
         `coverage_mode` for Requirement 5 ablations!).
      2. Substitutes cross-demo behavior `z_tilde_i = sum_j W_ij z_j` into clip i's scene
         `v_{0,i}` and evaluates conditional state-change prediction accuracy (`L_reuse`).
         Thus `W_ij` is trained directly by the actual reuse value of donor `j` in scene `i`!
      3. Enforces counterfactual cross-scene cycle consistency (`L_cycle`): applying donor
         `j`'s behavior to scene `v_{0,i}` via `F_psi` and re-encoding must recover `z_j`.
      4. Enforces self-supervised displacement equivariance (`L_equiv`): matches `Δz_m` to
         scene-canonicalized forward-model behavioral effect differences `Δe_m`, giving
         flat Euclidean editability without ANY GT actions!
      5. Prevents hub collapse (column marginal entropy balancing `L_bal`) and
         representation collapse (VICReg `L_vic`).
    """

    def __init__(
        self,
        z_rel_dim: int,
        *,
        num_tokens: int = 4,
        token_wise: bool = False,
        proj_dim: int = 256,
        hidden_dim: int = 1024,
        predict_motion: bool = True,
        coverage_mode: str = "cross_demo_anti_scene",
        top_k_donors: Optional[int] = None,
        init_temp: float = 0.12,
    ):
        super().__init__()
        self.z_rel_dim = int(z_rel_dim)
        self.num_tokens = int(num_tokens)
        self.token_wise = bool(token_wise)
        self.coverage_mode = str(coverage_mode)
        self.top_k_donors = top_k_donors

        if self.token_wise:
            if z_rel_dim % num_tokens != 0:
                raise ValueError(f"z_rel_dim={z_rel_dim} must be divisible by num_tokens={num_tokens}")
            tok_d = z_rel_dim // num_tokens
            self.q_projs = nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(tok_d), nn.Linear(tok_d, proj_dim // 2)) for _ in range(num_tokens)]
            )
            self.k_projs = nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(tok_d), nn.Linear(tok_d, proj_dim // 2)) for _ in range(num_tokens)]
            )
        else:
            self.q_proj = nn.Sequential(nn.LayerNorm(z_rel_dim), nn.Linear(z_rel_dim, proj_dim))
            self.k_proj = nn.Sequential(nn.LayerNorm(z_rel_dim), nn.Linear(z_rel_dim, proj_dim))

        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))
        self.forward_pred = ConditionalStateChangePredictor(
            z_dim=z_rel_dim,
            scene_dim=2 * 768,
            hidden_dim=hidden_dim,
            predict_motion=predict_motion,
        )

    def _build_donor_mask(
        self,
        v0: torch.Tensor,
        ep_ids: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(valid_mask, scene_sim)` where `valid_mask[i, j]` is True iff donor `j` is allowed for `i`."""
        B = v0.shape[0]
        dev = v0.device
        with torch.no_grad():
            s_scene = pairwise_cosine_sim(v0.float().reshape(B, -1))  # (B, B)
            diff_ep = ep_ids[:, None] != ep_ids[None, :]

            if self.coverage_mode == "same_task_only" and task_ids is not None:
                valid = diff_ep & (task_ids[:, None] == task_ids[None, :])
            elif self.coverage_mode == "strict_cross_task" and task_ids is not None:
                valid = diff_ep & (task_ids[:, None] != task_ids[None, :])
            elif self.coverage_mode == "cross_demo_unmasked":
                valid = diff_ep
            elif self.coverage_mode == "cross_demo_anti_scene":
                # Mask out same episode AND top-25% most visually similar static scenes!
                q75 = torch.quantile(s_scene[diff_ep], 0.75) if diff_ep.any() else torch.tensor(1.0, device=dev)
                valid = diff_ep & (s_scene <= q75)
            elif self.coverage_mode == "cross_demo_strict_anti_scene":
                # Even stricter: only allow visually dissimilar bottom-50% scenes + cross-task if available
                q50 = torch.quantile(s_scene[diff_ep], 0.50) if diff_ep.any() else torch.tensor(1.0, device=dev)
                valid = diff_ep & (s_scene <= q50)
                if task_ids is not None:
                    valid = valid & (task_ids[:, None] != task_ids[None, :])
            else:
                raise ValueError(f"Unknown coverage_mode: {self.coverage_mode}")

            # Safety fallback for any row with 0 valid donors -> fall back to diff_ep or off-diag
            empty_row = ~valid.any(dim=1)
            if empty_row.any():
                fallback = diff_ep if diff_ep.any() else _off_diag_mask(B, dev)
                valid = torch.where(empty_row[:, None], fallback, valid)
            return valid, s_scene

    def compute_relation_weights(
        self,
        z_rel: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute cross-demo relation weights `W` (`(B, B)` or `(K, B, B)`), substituted `z_tilde`,
        and column-marginal load-balancing loss `L_bal`."""
        B = z_rel.shape[0]
        temp = self.log_temp.exp().clamp(0.05, 0.5)

        def _attn_from_qk(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            qn = F.normalize(q.float(), dim=-1)
            kn = F.normalize(k.float(), dim=-1)
            logits = (qn @ kn.T) / temp
            logits = logits.masked_fill(~valid_mask, -1e4)
            if self.top_k_donors is not None and self.top_k_donors < B - 1:
                topk_val, _ = torch.topk(logits, k=int(self.top_k_donors), dim=-1)
                thr = topk_val[:, -1:].detach()
                logits = logits.masked_fill(logits < thr, -1e4)
            W = torch.softmax(logits, dim=-1)
            # Column marginal load-balance KL(col_mean || Uniform) prevents hub collapse
            col_mean = W.mean(dim=0).clamp_min(1e-7)
            bal = (col_mean * torch.log(col_mean * float(B))).sum()
            return W, bal

        if not self.token_wise:
            zf = z_rel.float().reshape(B, -1)
            W, bal = _attn_from_qk(self.q_proj(zf), self.k_proj(zf))
            z_tilde = W @ zf  # (B, z_rel_dim)
            return W, z_tilde.reshape_as(z_rel), bal

        # Token-wise autonomous cross-demo substitution (each token picks its own donors!)
        zt = z_rel.float().reshape(B, self.num_tokens, -1)
        Ws, zts, bals = [], [], []
        for k in range(self.num_tokens):
            Wk, balk = _attn_from_qk(self.q_projs[k](zt[:, k]), self.k_projs[k](zt[:, k]))
            Ws.append(Wk)
            zts.append(Wk @ zt[:, k])
            bals.append(balk)
        W_stack = torch.stack(Ws, dim=0)  # (K, B, B)
        z_tilde = torch.stack(zts, dim=1).reshape_as(z_rel)
        bal = torch.stack(bals).mean()
        return W_stack, z_tilde, bal

    def _masked_pred_loss(
        self,
        pred_dv: torch.Tensor,
        tgt_dv: torch.Tensor,
        seg_valid: torch.Tensor,
        pred_mo: Optional[torch.Tensor] = None,
        tgt_mo: Optional[torch.Tensor] = None,
        motion_weight: float = 0.25,
    ) -> torch.Tensor:
        w = seg_valid[:, :, None, None]  # (B, 4, 1, 1)
        denom = w.sum().clamp_min(1.0)
        # Scale-normalized visual delta MSE (unit-scale across channels)
        dv_err = ((pred_dv - tgt_dv).square() * w).sum() / (denom * 2.0 * 768.0)
        if pred_mo is not None and tgt_mo is not None:
            mo_err = ((pred_mo - tgt_mo).square() * w).sum() / (denom * 2.0 * MOTION_FEAT_DIM)
            return dv_err + motion_weight * mo_err
        return dv_err

    def forward_loss(
        self,
        b_t: torch.Tensor,
        clip_rgb: torch.Tensor,
        clip_pad_mask: Optional[torch.Tensor],
        ep_ids: torch.Tensor,
        *,
        task_ids: Optional[torch.Tensor] = None,
        clip_motion: Optional[torch.Tensor] = None,
        encoder_fn: Optional[Any] = None,
        op_dim: Optional[int] = None,
        use_cycle: bool = True,
        use_equiv: bool = True,
        cycle_weight: float = 0.15,
        equiv_weight: float = 0.35,
        bal_weight: float = 0.05,
        vic_weight: float = 0.05,
        tok_decorr_weight: float = 0.05,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute full learnable reuse-driven relational loss (ZERO GT actions!)."""
        B = b_t.shape[0]
        if op_dim is not None:
            z_rel = b_t[:, :, :op_dim].float()
        else:
            z_rel = b_t.float()
        z_rel_flat = z_rel.reshape(B, -1)

        tgts = compute_clip_state_change_targets(clip_rgb, clip_pad_mask, clip_motion)
        v0 = tgts["v0"]
        dv_tgt = tgts["delta_v_seg"]
        seg_valid = tgts["seg_valid"]
        mo_tgt = tgts.get("motion_seg")

        valid_mask, s_scene = self._build_donor_mask(v0, ep_ids, task_ids=task_ids)
        W, z_tilde, l_bal = self.compute_relation_weights(z_rel, valid_mask)

        # 1. Self conditional state-change prediction (trains forward predictor F_psi)
        pred_dv_self, pred_mo_self = self.forward_pred(v0, z_rel_flat)
        l_self = self._masked_pred_loss(pred_dv_self, dv_tgt, seg_valid, pred_mo_self, mo_tgt)

        # 2. Cross-demo behavior substitution reuse-value loss!
        pred_dv_reuse, pred_mo_reuse = self.forward_pred(v0, z_tilde.reshape(B, -1))
        l_reuse = self._masked_pred_loss(pred_dv_reuse, dv_tgt, seg_valid, pred_mo_reuse, mo_tgt)

        # 3. Counterfactual cross-scene cycle consistency (re-encode imagined transition!)
        l_cycle = b_t.new_zeros(())
        if use_cycle and encoder_fn is not None:
            # Sample 1 valid cross-demo anti-scene donor j for each i (subset of 64 for speed)
            n_cyc = min(B, 64)
            with torch.no_grad():
                v_sub = valid_mask[:n_cyc]
                probs = v_sub.float() / v_sub.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                donor_j = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (n_cyc,)
            v0_sub = v0[:n_cyc]
            z_donor = z_rel_flat[donor_j].detach()
            # Generate counterfactual 4-segment state change of applying donor j's behavior in scene i
            dv_cf, _ = self.forward_pred(v0_sub, z_donor)  # (n_cyc, 4, 2, 768)
            # Upsample 4 segments back to 16 frames cleanly around v0_sub
            dv_16 = dv_cf.repeat_interleave(4, dim=1)  # (n_cyc, 16, 2, 768)
            v_cf = v0_sub[:, None, :, :] + dv_16  # (n_cyc, 16, 2, 768)
            b_cf = encoder_fn(v_cf.to(dtype=clip_rgb.dtype), None)
            z_cf = b_cf[:, :, :op_dim].float() if op_dim is not None else b_cf.float()
            l_cycle = (1.0 - F.cosine_similarity(z_cf.reshape(n_cyc, -1), z_donor, dim=-1)).mean()

        # 4. Self-supervised displacement equivariance via scene-canonicalized forward effects!
        l_equiv = b_t.new_zeros(())
        if use_equiv and B >= 8:
            # Evaluate every behavior latent at 4 shared reference scenes -> 100% scene-canonical effect!
            with torch.no_grad():
                ref_idx = torch.linspace(0, B - 1, 4, device=b_t.device).long()
                eff_list = []
                for r in ref_idx:
                    v_ref = v0[r : r + 1].expand(B, -1, -1)
                    e_dv, e_mo = self.forward_pred(v_ref, z_rel_flat.detach())
                    e_vec = e_dv.reshape(B, -1)
                    if e_mo is not None:
                        e_dv_n = e_vec / e_vec.square().mean().sqrt().clamp_min(1e-5)
                        e_mo_n = e_mo.reshape(B, -1) / e_mo.square().mean().sqrt().clamp_min(1e-5)
                        e_vec = torch.cat([e_dv_n, e_mo_n], dim=-1)
                    eff_list.append(e_vec)
                e_canon = torch.stack(eff_list, dim=0).mean(dim=0)  # (B, D_e)
            l_equiv, _ = compute_video_derived_displacement_loss(z_rel, e_canon, norm_weight=0.25)

        # 5. VICReg anti-collapse + optional inter-token decorrelation for token-wise reuse
        l_vic, l_var, l_cov = compute_vicreg_loss(z_rel)
        l_tdec = b_t.new_zeros(())
        if self.token_wise:
            zt = z_rel.reshape(B, self.num_tokens, -1)
            ckas = []
            for a in range(self.num_tokens):
                for b_idx in range(a + 1, self.num_tokens):
                    ckas.append(compute_linear_cka_penalty(zt[:, a], zt[:, b_idx]))
            l_tdec = torch.stack(ckas).mean()

        total = (
            0.5 * l_self
            + 1.0 * l_reuse
            + cycle_weight * l_cycle
            + equiv_weight * l_equiv
            + bal_weight * l_bal
            + vic_weight * l_vic
            + (tok_decorr_weight * l_tdec if self.token_wise else 0.0)
        )

        # Monitor anti-scene-matching & relation sharpness diagnostics
        with torch.no_grad():
            W_eff = W.mean(dim=0) if W.ndim == 3 else W
            w_ent = -(W_eff * W_eff.clamp_min(1e-8).log()).sum(dim=-1).mean()
            w_scene_mean = (W_eff * s_scene).sum(dim=-1).mean()
            w_max_mean = W_eff.max(dim=-1).values.mean()

        stats = {
            "rel_loss": total.detach(),
            "reuse_loss": l_reuse.detach(),
            "fwd_self_loss": l_self.detach(),
            "cycle_loss": l_cycle.detach(),
            "equiv_loss": l_equiv.detach(),
            "bal_loss": l_bal.detach(),
            "vic_loss": l_vic.detach(),
            "tok_decorr": l_tdec.detach(),
            "w_entropy": w_ent.detach(),
            "w_scene_cos": w_scene_mean.detach(),
            "w_max": w_max_mean.detach(),
            "temp": self.log_temp.exp().detach(),
        }
        return total, stats
