"""E18 Pillar 2: Geometry-Aware Motion Relations & Perspective Conflict Diagnostics.

1. Self-Supervised Cross-View Canonical Motion Encoder (`CVGeoMotionEncoder`):
     - Trained 100% purely from unlabeled dual-camera LIBERO optical flow
       (`agentview` cam0 vs `eye_in_hand` cam1, ZERO GT actions, ZERO task metadata!).
     - Uses symmetric Cross-View InfoNCE + VICReg anti-collapse across the two views
       of the same 16-frame clip so camera-specific 2D perspective artifacts (4x4 spatial
       grid location, perspective depth scaling, background parallax) are filtered out,
       retaining ONLY the cross-view consistent 3D physical kinematic motion!
2. Quantitative Perspective Distortion & 3D Arithmetic Conflict Audit:
     - Directly measures on LIBERO validation clips & Tier-2 parallel arithmetic quadruples:
       (a) Raw 2D optical flow (4x4 grid + kinematics, as used in E16-D3 / E17-M1..M3)
       (b) Location-invariant 2D kinematics only (dropping 4x4 spatial grid!)
       (c) Self-Supervised Cross-View Canonical Motion (`CV-Geo`)
       (d) Agentview-only vs Eye-in-Hand-only flow
     - Tests whether 2D spatial/perspective inconsistency explains E17's conflict between
       optical flow displacement loss and 3D action parallel arithmetic!
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import DEFAULT_DENSE_CACHE_DIR, E5_CACHE_DIR, LOCAL_WINDOW_LEN
from e12.action_attributes import extract_action_attribute_vectors, pairwise_cosine_sim
from e12.eval_suite import ValClipCache, _spearman_pearson_from_vectors
from e13.exp1_training_free import _build_fixed_analogy_quadruples
from e16 import MOTION_CACHE_DIR, MOTION_FEAT_DIM
from e16.video_relations import (
    VideoMotionLocalBank,
    compute_clip_state_change_targets,
    compute_vicreg_loss,
)

from . import CV_GEO_DIR, OUTPUT_ROOT


def extract_per_camera_motion_clips(
    clip_motion: torch.Tensor,
    clip_pad_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-camera 4-segment + cumulative grid clip motion vectors `(B, 384)` each.

    Camera 0 = `agentview` (fixed 3rd-person), Camera 1 = `eye_in_hand` (moving wrist).
    """
    B, L, C, D = clip_motion.shape
    assert C == 2 and D == MOTION_FEAT_DIM
    valid = (
        (~clip_pad_mask.to(dtype=torch.bool))
        if clip_pad_mask is not None
        else torch.ones((B, L), device=clip_motion.device, dtype=torch.bool)
    )
    m = clip_motion.float() * valid[:, :, None, None].float()
    val4 = valid.reshape(B, 4, 4).float()
    cnt4 = val4.sum(dim=2).clamp_min(1.0)
    m4 = m.reshape(B, 4, 4, C, D).sum(dim=2) / cnt4[:, :, None, None]  # (B, 4, 2, 64)
    cum = torch.cumsum(m[:, :, :, :32], dim=1)[:, [3, 7, 11, 15]] * 0.25  # (B, 4, 2, 32)

    m_cam0 = torch.cat([m4[:, :, 0, :].reshape(B, -1), cum[:, :, 0, :].reshape(B, -1)], dim=-1)  # 384D
    m_cam1 = torch.cat([m4[:, :, 1, :].reshape(B, -1), cum[:, :, 1, :].reshape(B, -1)], dim=-1)  # 384D
    return m_cam0, m_cam1


def extract_location_invariant_motion(
    clip_motion: torch.Tensor,
    clip_pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop the 4x4 spatial grid [0:32] (location-dependent!) and keep ONLY [32:64] kinematics+HOF."""
    B, L, C, D = clip_motion.shape
    valid = (
        (~clip_pad_mask.to(dtype=torch.bool))
        if clip_pad_mask is not None
        else torch.ones((B, L), device=clip_motion.device, dtype=torch.bool)
    )
    m = clip_motion.float() * valid[:, :, None, None].float()
    val4 = valid.reshape(B, 4, 4).float()
    cnt4 = val4.sum(dim=2).clamp_min(1.0)
    m4 = m.reshape(B, 4, 4, C, D).sum(dim=2) / cnt4[:, :, None, None]  # (B, 4, 2, 64)
    # Keep [32:64] only: active mean flow, divergence, curl, shear, lower convergence, HOF, quad speeds
    return m4[:, :, :, 32:].reshape(B, -1)  # (B, 256)


class CVGeoMotionEncoder(nn.Module):
    """Self-supervised Cross-View Canonical Motion Encoder (agentview <-> eye_in_hand).

    Maps each camera's 2D optical flow stream (`384D`) into a shared canonical 128D
    geometry-consistent motion embedding via symmetric Cross-View InfoNCE + VICReg.
    Trained with ZERO GT actions and ZERO task labels!
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        def _make_tower() -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.cam0_enc = _make_tower()
        self.cam1_enc = _make_tower()
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07), dtype=torch.float32))

    def encode_views(
        self, clip_motion: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m0, m1 = extract_per_camera_motion_clips(clip_motion, clip_pad_mask)
        u0 = self.cam0_enc(m0)
        u1 = self.cam1_enc(m1)
        return u0, u1

    def encode_canonical(
        self, clip_motion: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(u_cv, view_agreement)` where `u_cv` is the consensus motion vector `(B, 128)`."""
        u0, u1 = self.encode_views(clip_motion, clip_pad_mask)
        u0_n = F.normalize(u0.float(), dim=-1)
        u1_n = F.normalize(u1.float(), dim=-1)
        agree = (u0_n * u1_n).sum(dim=-1)  # cosine agreement in [-1, 1]
        # Scale-preserving average of the two canonical view embeddings
        u_cv = 0.5 * (u0 + u1)
        return u_cv, agree

    def forward_loss(
        self, clip_motion: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        u0, u1 = self.encode_views(clip_motion, clip_pad_mask)
        B = u0.shape[0]
        u0_n = F.normalize(u0.float(), dim=-1)
        u1_n = F.normalize(u1.float(), dim=-1)
        temp = self.log_temp.exp().clamp(0.03, 0.3)
        logits = (u0_n @ u1_n.T) / temp
        labels = torch.arange(B, device=u0.device)
        l_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        # Direct Euclidean invariance + VICReg prevents scale mismatch so displacements Δu are well-behaved!
        l_inv = F.mse_loss(u0.float(), u1.float())
        l_vic0, _, _ = compute_vicreg_loss(u0, std_target=0.80, cov_weight=0.04)
        l_vic1, _, _ = compute_vicreg_loss(u1, std_target=0.80, cov_weight=0.04)
        total = l_nce + 0.5 * l_inv + 0.25 * (l_vic0 + l_vic1)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            cos_pos = (u0_n * u1_n).sum(dim=-1).mean()
        return total, {
            "loss": total.detach(),
            "nce": l_nce.detach(),
            "inv": l_inv.detach(),
            "cv_acc": acc.detach(),
            "cv_cos": cos_pos.detach(),
        }


def compute_weighted_displacement_loss(
    z_target: torch.Tensor,
    u_video: torch.Tensor,
    pair_weights: Optional[torch.Tensor] = None,
    *,
    norm_weight: float = 0.25,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Geometry-aware displacement loss supporting optional cross-view consistency pair weighting."""
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
        if pair_weights is not None:
            pw = pair_weights[: 2 * M].float().reshape(M, 2).min(dim=1).values.clamp(0.1, 1.0)
            W_mat = pw[:, None] * pw[None, :]
        else:
            W_mat = torch.ones((M, M), device=z.device, dtype=torch.float32)

    S_dz = pairwise_cosine_sim(dz)
    off = ~torch.eye(M, device=z.device, dtype=torch.bool)
    valid = off & active[:, None] & active[None, :]
    if int(valid.sum().item()) > 0:
        w_v = W_mat[valid]
        dir_loss = ((S_dz[valid] - S_du[valid]).square() * w_v).sum() / w_v.sum().clamp_min(1e-6)
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
    }


class LinearCVCCAEncoder(nn.Module):
    """Closed-form Linear Cross-View Canonical Correlation Analysis (agentview <-> eye_in_hand).

    Unlike nonlinear InfoNCE (which warps Euclidean differences onto an isotropic sphere),
    Linear CV-CCA learns optimal linear projections `(W0, W1)` maximizing cross-view correlation,
    STRICTLY PRESERVING linear displacement arithmetic `Δ(W m) = W Δm` while filtering out
    camera-specific perspective distortion uncorrelated across the two cameras!
    Fitted 100% purely from unlabeled dual-camera optical flow (ZERO GT actions!).
    """

    def __init__(self, in_dim: int = 384, out_dim: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.register_buffer("mu0", torch.zeros(in_dim))
        self.register_buffer("mu1", torch.zeros(in_dim))
        self.register_buffer("W0", torch.eye(in_dim, out_dim))
        self.register_buffer("W1", torch.eye(in_dim, out_dim))
        self.register_buffer("cca_corrs", torch.ones(out_dim))

    @torch.no_grad()
    def fit_from_unlabeled(
        self, m0: torch.Tensor, m1: torch.Tensor, reg: float = 0.05
    ) -> Dict[str, float]:
        """Exact regularized SVD-CCA on `(N, 384)` dual-camera flow clips."""
        X = m0.double()
        Y = m1.double()
        N = X.shape[0]
        mu0 = X.mean(dim=0)
        mu1 = Y.mean(dim=0)
        Xc = X - mu0
        Yc = Y - mu1
        Cxx = (Xc.T @ Xc) / max(N - 1, 1) + reg * torch.eye(self.in_dim, device=X.device, dtype=torch.float64)
        Cyy = (Yc.T @ Yc) / max(N - 1, 1) + reg * torch.eye(self.in_dim, device=X.device, dtype=torch.float64)
        Cxy = (Xc.T @ Yc) / max(N - 1, 1)

        ex, Vx = torch.linalg.eigh(Cxx)
        ey, Vy = torch.linalg.eigh(Cyy)
        inv_sx = Vx @ torch.diag(ex.clamp_min(1e-6).rsqrt()) @ Vx.T
        inv_sy = Vy @ torch.diag(ey.clamp_min(1e-6).rsqrt()) @ Vy.T
        M = inv_sx @ Cxy @ inv_sy
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k = self.out_dim
        # Weight each canonical axis by its canonical correlation sqrt(r_k) so noisy low-corr axes fade!
        s_k = S[:k].clamp(0.0, 1.0)
        W0 = (inv_sx @ U[:, :k]) * s_k.sqrt().unsqueeze(0)
        W1 = (inv_sy @ Vh[:k, :].T) * s_k.sqrt().unsqueeze(0)

        self.mu0.copy_(mu0.float())
        self.mu1.copy_(mu1.float())
        self.W0.copy_(W0.float())
        self.W1.copy_(W1.float())
        self.cca_corrs.copy_(s_k.float())
        return {
            "top1_corr": float(s_k[0].item()),
            "top16_mean_corr": float(s_k[:16].mean().item()),
            "top64_mean_corr": float(s_k.mean().item()),
        }

    def encode_canonical(
        self, clip_motion: torch.Tensor, clip_pad_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m0, m1 = extract_per_camera_motion_clips(clip_motion, clip_pad_mask)
        u0 = (m0.float() - self.mu0) @ self.W0
        u1 = (m1.float() - self.mu1) @ self.W1
        agree = F.cosine_similarity(u0, u1, dim=-1)
        u_cv = 0.5 * (u0 + u1)
        return u_cv, agree


def fit_linear_cv_cca(
    *,
    out_dir: Path = CV_GEO_DIR,
    out_dim: int = 64,
    n_fit: int = 40_000,
    seed: int = 42,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = VideoMotionLocalBank(
        e5_cache_dir=E5_CACHE_DIR,
        dense_cache_dir=DEFAULT_DENSE_CACHE_DIR,
        motion_cache_dir=MOTION_CACHE_DIR,
        device=device,
        resident_on_device=False,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1805)
    idx = rng.choice(bank.num_train_samples(), size=min(n_fit, bank.num_train_samples()), replace=False)
    m0_l, m1_l = [], []
    with torch.inference_mode():
        for s in range(0, len(idx), 1024):
            e = min(s + 1024, len(idx))
            b = bank.gather_rich_batch(bank.train_episodes[idx[s:e]], bank.train_frames[idx[s:e]], include_actions=False)
            assert b["actions_10d"] is None
            m0, m1 = extract_per_camera_motion_clips(b["clip_motion"], b["clip_pad_mask"])
            m0_l.append(m0)
            m1_l.append(m1)
    cca = LinearCVCCAEncoder(in_dim=384, out_dim=out_dim).to(device)
    stats = cca.fit_from_unlabeled(torch.cat(m0_l, dim=0), torch.cat(m1_l, dim=0), reg=0.05)
    print(f"[E18 Linear CV-CCA] Fitted on {len(idx):,} unlabeled clips: {stats}", flush=True)
    ckpt = out_dir / "cv_geo_cca.pt"
    atomic_torch_save({"state_dict": {k: v.cpu() for k, v in cca.state_dict().items()}, "config": {"in_dim": 384, "out_dim": out_dim}, "stats": stats}, ckpt)
    return ckpt


def train_cv_geo_encoder(
    *,
    out_dir: Path = CV_GEO_DIR,
    steps: int = 8000,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 42,
    gpu_resident: bool = True,
) -> Path:
    """Train self-supervised Cross-View Canonical Motion Encoder on 100% unlabeled LIBERO flow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    bank = VideoMotionLocalBank(
        e5_cache_dir=E5_CACHE_DIR,
        dense_cache_dir=DEFAULT_DENSE_CACHE_DIR,
        motion_cache_dir=MOTION_CACHE_DIR,
        device=device,
        resident_on_device=gpu_resident,
        seed=seed,
    )
    model = CVGeoMotionEncoder(in_dim=384, hidden_dim=512, out_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    rng = np.random.default_rng(seed + 1802)
    n_train = bank.num_train_samples()
    t0 = time.monotonic()
    hist = []
    for step in range(1, steps + 1):
        model.train()
        idx = rng.integers(0, n_train, size=batch_size)
        batch = bank.gather_rich_batch(
            bank.train_episodes[idx], bank.train_frames[idx], include_actions=False
        )
        assert batch["actions_10d"] is None
        loss, st = model.forward_loss(batch["clip_motion"], batch["clip_pad_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == steps:
            model.eval()
            with torch.inference_mode():
                vn = min(2048, bank.num_val_samples())
                vb = bank.gather_rich_batch(
                    bank.val_episodes[:vn], bank.val_frames[:vn], include_actions=False
                )
                assert vb["actions_10d"] is None
                _, vst = model.forward_loss(vb["clip_motion"], vb["clip_pad_mask"])
            rec = {
                "step": step,
                "train_nce": float(st["nce"].item()),
                "train_acc": float(st["cv_acc"].item()),
                "val_nce": float(vst["nce"].item()),
                "val_acc": float(vst["cv_acc"].item()),
                "val_cos": float(vst["cv_cos"].item()),
            }
            hist.append(rec)
            print(
                f"[E18 CV-Geo step {step:4d}/{steps}] val_nce={rec['val_nce']:.4f} "
                f"val_acc={rec['val_acc']*100:.1f}% val_cos={rec['val_cos']:.4f} "
                f"({time.monotonic()-t0:.1f}s)",
                flush=True,
            )

    ckpt_path = out_dir / "cv_geo_encoder.pt"
    atomic_torch_save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": {"in_dim": 384, "hidden_dim": 512, "out_dim": 128, "steps": steps, "seed": seed},
            "history": hist,
        },
        ckpt_path,
    )
    atomic_write_json({"history": hist}, out_dir / "metrics.json")
    return ckpt_path


@torch.inference_mode()
def run_perspective_conflict_audit(
    cv_geo_ckpt: Path = CV_GEO_DIR / "cv_geo_encoder.pt",
    out_dir: Path = OUTPUT_ROOT / "pillar2_geometry_audit",
) -> Dict[str, Any]:
    """Quantitatively test whether 2D perspective inconsistency explains E17's flow vs arithmetic conflict!"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = VideoMotionLocalBank(
        e5_cache_dir=E5_CACHE_DIR,
        dense_cache_dir=DEFAULT_DENSE_CACHE_DIR,
        motion_cache_dir=MOTION_CACHE_DIR,
        device=device,
        resident_on_device=False,
        seed=42,
    )
    cache = ValClipCache(seed=42)
    N = len(cache.val_episodes)

    # Load trained self-supervised CV-Geo encoder & optional Linear CV-CCA
    cv_payload = torch.load(cv_geo_ckpt, map_location="cpu", weights_only=False)
    cv_enc = CVGeoMotionEncoder(**{k: cv_payload["config"][k] for k in ("in_dim", "hidden_dim", "out_dim")}).to(device).eval()
    cv_enc.load_state_dict(cv_payload["state_dict"], strict=True)

    cca_ckpt = Path(cv_geo_ckpt).parent / "cv_geo_cca.pt"
    cca_enc = None
    if cca_ckpt.is_file():
        cca_pl = torch.load(cca_ckpt, map_location="cpu", weights_only=False)
        cca_enc = LinearCVCCAEncoder(**cca_pl["config"]).to(device).eval()
        cca_enc.load_state_dict(cca_pl["state_dict"], strict=True)

    # Gather descriptors across all validation clips
    raw_flow_l, loc_inv_l, cam0_l, cam1_l, cv_geo_l, cca_l, agree_l, v0_l = [], [], [], [], [], [], [], []
    for s in range(0, N, 512):
        e = min(s + 512, N)
        b = bank.gather_rich_batch(cache.val_episodes[s:e], cache.val_frames[s:e], include_actions=False)
        tg = compute_clip_state_change_targets(b["clip_rgb"], b["clip_pad_mask"], b["clip_motion"])
        u_raw = torch.cat([tg["motion_seg"].reshape(e - s, -1), tg["motion_cum"].reshape(e - s, -1)], dim=-1)
        u_inv = extract_location_invariant_motion(b["clip_motion"], b["clip_pad_mask"])
        m0, m1 = extract_per_camera_motion_clips(b["clip_motion"], b["clip_pad_mask"])
        u_cv, ag = cv_enc.encode_canonical(b["clip_motion"], b["clip_pad_mask"])
        raw_flow_l.append(u_raw.cpu().numpy())
        loc_inv_l.append(u_inv.cpu().numpy())
        cam0_l.append(m0.cpu().numpy())
        cam1_l.append(m1.cpu().numpy())
        cv_geo_l.append(u_cv.cpu().numpy())
        if cca_enc is not None:
            u_cca, _ = cca_enc.encode_canonical(b["clip_motion"], b["clip_pad_mask"])
            cca_l.append(u_cca.cpu().numpy())
        agree_l.append(ag.cpu().numpy())
        v0_l.append(tg["v0"].reshape(e - s, -1).cpu().numpy())

    feats = {
        "raw_flow_2d": np.concatenate(raw_flow_l, axis=0),
        "loc_invariant_kin": np.concatenate(loc_inv_l, axis=0),
        "agentview_only_2d": np.concatenate(cam0_l, axis=0),
        "wrist_only_2d": np.concatenate(cam1_l, axis=0),
        "cv_geo_infonce": np.concatenate(cv_geo_l, axis=0),
    }
    if cca_l:
        feats["cv_geo_linear_cca"] = np.concatenate(cca_l, axis=0)
    V0 = np.concatenate(v0_l, axis=0)
    agree = np.concatenate(agree_l, axis=0)

    # Build exact Tier-2 parallel arithmetic quadruples (A1, A2, B1, B2) where Δa_A ≈ Δa_B in 3D!
    # Build dummy encoded dict using cache actions/is_tail
    is_tail = (cache.val_frames + LOCAL_WINDOW_LEN) > cache.ep_lengths_np[cache.val_episodes]
    a10_list, a7_list = [], []
    from e5.actions import action_10d_to_7d
    for s in range(0, N, 512):
        e = min(s + 512, N)
        _, _, a10, _ = cache.gather(cache.val_episodes[s:e], cache.val_frames[s:e])
        a10_np = a10.cpu().numpy()
        a10_list.append(a10_np)
        a7_list.append(action_10d_to_7d(a10_np))
    A10 = np.concatenate(a10_list, axis=0)
    A7 = np.concatenate(a7_list, axis=0)
    dummy_enc = {
        "actions_10d": A10,
        "actions_7d": A7,
        "is_tail": is_tail,
        "latents": np.zeros((N, 4, 1), dtype=np.float32),
        "latents_flat": np.zeros((N, 4), dtype=np.float32),
    }
    quads = _build_fixed_analogy_quadruples(cache, dummy_enc, device, seed=42)
    A1, A2, B1, B2 = quads["tier2"]
    # Also compute ground-truth 3D action displacement alignment on the exact same quadruples
    da_A = (A7[A2] - A7[A1]).reshape(len(A1), -1)
    da_B = (A7[B2] - A7[B1]).reshape(len(A1), -1)
    dpos_A = (A7[A2, :, :3] - A7[A1, :, :3]).reshape(len(A1), -1)
    dpos_B = (A7[B2, :, :3] - A7[B1, :, :3]).reshape(len(A1), -1)

    def _pair_cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        nu = np.linalg.norm(u, axis=-1).clip(1e-8)
        nv = np.linalg.norm(v, axis=-1).clip(1e-8)
        return np.sum(u * v, axis=-1) / (nu * nv)

    # Also evaluate cross-task pairwise displacement alignment (512 random cross-task pairs)
    nt = np.flatnonzero(~is_tail)
    rng = np.random.default_rng(1802)
    sub = np.sort(rng.choice(nt, size=1024, replace=False))
    t_sub = cache.val_task_ids[sub]
    a10_sub = torch.from_numpy(A10[sub]).to(device)
    attrs = extract_action_attribute_vectors(a10_sub)
    S_bal = pairwise_cosine_sim(attrs["global_balanced7d"]).cpu().numpy()
    S_trans = pairwise_cosine_sim(attrs["translation"]).cpu().numpy()
    S_grip = pairwise_cosine_sim(attrs["gripper"]).cpu().numpy()
    S_v0 = pairwise_cosine_sim(torch.from_numpy(V0[sub]).to(device)).cpu().numpy()
    ct_mask = t_sub[:, None] != t_sub[None, :]

    # Same-3D-action different-scene invariance test: for each sample i in sub, find cross-task j with
    # highest S_bal (nearly identical 3D action in a DIFFERENT scene!) and measure cosine similarity of u_i, u_j!
    S_bal_ct = np.where(ct_mask, S_bal, -2.0)
    best_act_j = np.argmax(S_bal_ct, axis=1)

    # Stratified by visual scene dissimilarity on Tier-2 quadruples (cos(v0_A1, v0_B1))
    quad_scene_cos = _pair_cos(V0[A1], V0[B1])
    med_sc = float(np.median(quad_scene_cos))
    far_scene = quad_scene_cos <= med_sc
    near_scene = quad_scene_cos > med_sc

    results: Dict[str, Any] = {
        "n_tier2_quads": int(len(A1)),
        "gt_action_quad_cos": float(np.mean(_pair_cos(da_A, da_B))),
        "gt_pos3d_quad_cos": float(np.mean(_pair_cos(dpos_A, dpos_B))),
        "cv_geo_mean_view_agreement": float(np.mean(agree)),
        "descriptors": {},
    }

    # Pairwise displacement correlation on 512 pairs from sub
    p1, p2 = sub[:512], sub[512:1024]
    da_pair = torch.from_numpy((A7[p2] - A7[p1]).reshape(512, -1)).to(device)
    S_da_pair = pairwise_cosine_sim(da_pair).cpu().numpy()
    off_p = ~np.eye(512, dtype=bool)

    for name, U in feats.items():
        du_A = U[A2] - U[A1]
        du_B = U[B2] - U[B1]
        q_cos = _pair_cos(du_A, du_B)

        U_t = torch.from_numpy(U[sub]).to(device)
        S_u = pairwise_cosine_sim(U_t).cpu().numpy()
        def _sp(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
            return _spearman_pearson_from_vectors(
                torch.as_tensor(a, dtype=torch.float32),
                torch.as_tensor(b, dtype=torch.float32),
            )

        sp_bal, _ = _sp(S_u[ct_mask], S_bal[ct_mask])
        sp_tr, _ = _sp(S_u[ct_mask], S_trans[ct_mask])
        sp_gr, _ = _sp(S_u[ct_mask], S_grip[ct_mask])
        sp_v0, _ = _sp(S_u[ct_mask], S_v0[ct_mask])

        # Displacement consistency rho(cos(du_m, du_n), cos(da_m, da_n))
        du_p = torch.from_numpy(U[p2] - U[p1]).to(device)
        S_du_p = pairwise_cosine_sim(du_p).cpu().numpy()
        sp_disp, pe_disp = _sp(S_du_p[off_p], S_da_pair[off_p])

        # Same-3D-action cross-scene invariance: cosine(u_i, u_{best_act_j}) vs random
        same_act_cos = float(np.mean(S_u[np.arange(len(sub)), best_act_j]))
        # Cross-task Top-1 action MSE10
        S_u_ct = np.where(ct_mask, S_u, -2.0)
        top1_j = np.argmax(S_u_ct, axis=1)
        top1_mse = float(np.mean((A10[sub] - A10[sub[top1_j]]) ** 2))

        results["descriptors"][name] = {
            "dim": int(U.shape[1]),
            "tier2_quad_disp_cos_all": float(np.mean(q_cos)),
            "tier2_quad_disp_cos_near_scene": float(np.mean(q_cos[near_scene])),
            "tier2_quad_disp_cos_far_scene": float(np.mean(q_cos[far_scene])),
            "perspective_drop_near_to_far": float(np.mean(q_cos[near_scene]) - np.mean(q_cos[far_scene])),
            "ct_spearman_bal7d": float(sp_bal),
            "ct_spearman_trans": float(sp_tr),
            "ct_spearman_grip": float(sp_gr),
            "ct_spearman_v0_leak": float(sp_v0),
            "disp_spearman_3d_act": float(sp_disp),
            "disp_pearson_3d_act": float(pe_disp),
            "same_3d_act_cross_scene_cos": same_act_cos,
            "ct_top1_act_mse10": top1_mse,
        }

    # Add cross-view agreement vs perspective error correlation
    c0_du = _pair_cos(feats["agentview_only_2d"][A2] - feats["agentview_only_2d"][A1],
                      feats["agentview_only_2d"][B2] - feats["agentview_only_2d"][B1])
    c1_du = _pair_cos(feats["wrist_only_2d"][A2] - feats["wrist_only_2d"][A1],
                      feats["wrist_only_2d"][B2] - feats["wrist_only_2d"][B1])
    results["cam0_vs_cam1_quad_disagreement_l1"] = float(np.mean(np.abs(c0_du - c1_du)))

    atomic_write_json(results, out_dir / "perspective_conflict_audit.json")
    print("[E18 Pillar 2 Perspective Conflict Audit]", flush=True)
    for k, d in results["descriptors"].items():
        print(
            f"  {k:20s}: Tier2_cos={d['tier2_quad_disp_cos_all']:.4f} "
            f"(near={d['tier2_quad_disp_cos_near_scene']:.4f}, far={d['tier2_quad_disp_cos_far_scene']:.4f}) | "
            f"disp_rho={d['disp_spearman_3d_act']:.4f} | bal7d={d['ct_spearman_bal7d']:.4f} | "
            f"v0_leak={d['ct_spearman_v0_leak']:.4f} | top1_mse={d['ct_top1_act_mse10']:.4f}",
            flush=True,
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cv-geo", action="store_true")
    p.add_argument("--fit-cca", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gpu-resident", action="store_true")
    args = p.parse_args()

    ckpt = CV_GEO_DIR / "cv_geo_encoder.pt"
    if args.train_cv_geo or not ckpt.is_file():
        ckpt = train_cv_geo_encoder(steps=args.steps, batch_size=args.batch_size, gpu_resident=args.gpu_resident)
    if args.fit_cca:
        fit_linear_cv_cca()
    if args.audit:
        run_perspective_conflict_audit(cv_geo_ckpt=ckpt)


if __name__ == "__main__":
    main()
