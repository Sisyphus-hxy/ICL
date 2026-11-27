"""Decomposed behavioral action attributes and pairwise similarity utilities for E12.

Given a batch of 16-step 10D/7D LIBERO OSC delta action chunks:
    a_7d: (N, 16, 7) = [dx, dy, dz, ax, ay, az, gripper]
    a_10d: (N, 16, 10) = [dx, dy, dz, rot6d(6), gripper]

We extract six distinct behavioral attribute views plus global views:
  1. global_raw7d      : raw flattened 7D delta trajectory (112D) -> arXiv:2609.19846 baseline
  2. global_balanced7d : scale-balanced 7D delta trajectory (pos/rot/grip balanced by empirical RMS)
  3. global_10d        : normalized 10D trajectory with centered rot6d (rot6d - I_6d)
  4. translation       : 3D delta position trajectory (48D) + net displacement direction (3D)
  5. rotation          : 3D axis-angle delta trajectory (48D) + active-rotation mask
  6. magnitude         : speed profile ||pos_t||, rot-speed profile ||rot_t||, path lengths, peak speed
  7. temporal          : direction- & scale-invariant temporal envelope (normalized speed profile,
                         early/mid/late energy, curvature/turning profile, peak timing, grip transition timing)
  8. gripper           : 16-step gripper trajectory + discrete semantic regime (open/closed/closing/opening)

All core feature extractors run in pure PyTorch on CPU or GPU so they can be shared
identically between Stage 1 training-free diagnostics and Stage 2 E12 training losses.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from e5.actions import action_10d_to_7d

# Empirical per-group RMS scales from E9 val/train split (pos ~0.26, rot ~0.083, grip ~1.0)
EMPIRICAL_POS_SCALE = 0.2587
EMPIRICAL_ROT_SCALE = 0.0831
EMPIRICAL_GRIP_SCALE = 1.0000

# Identity 6D rotation [1, 0, 0, 0, 1, 0] (first two columns of I_3 in BPP convention)
_ROT6D_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def extract_action_attribute_vectors(
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Extract decomposed behavioral attribute vectors from a batch of action chunks.

    Args:
        actions_10d: `(N, H, 10)` float tensor `[dx, dy, dz, rot6d(6), grip]`.
        actions_7d: optional `(N, H, 7)` float tensor `[dx, dy, dz, ax, ay, az, grip]`.
            Computed via exact `action_10d_to_7d` if omitted.

    Returns:
        Dictionary mapping attribute names to `(N, D_attr)` float32 tensors plus
        auxiliary masks / scalars (`pos_path`, `rot_path`, `is_moving`, `is_rotating`,
        `gripper_regime`).
    """
    a10 = actions_10d.float()
    if actions_7d is None:
        a7 = action_10d_to_7d(a10).float()
    else:
        a7 = actions_7d.float()

    N, H, _ = a7.shape
    dev = a7.device

    pos = a7[:, :, 0:3]  # (N, H, 3)
    rot = a7[:, :, 3:6]  # (N, H, 3)
    grip = a7[:, :, 6:7]  # (N, H, 1)

    # 1. Global views
    global_raw7d = a7.reshape(N, H * 7)

    # Balanced 7D: equalize pos(3), rot(3), grip(1) contributions
    # Normalize each group by empirical RMS, then weight so pos:rot:grip variance ~ 3:2:1
    bal_pos = (pos / EMPIRICAL_POS_SCALE) * (1.0 / (3.0 ** 0.5))
    bal_rot = (rot / EMPIRICAL_ROT_SCALE) * ((2.0 / 3.0) ** 0.5 / (3.0 ** 0.5))
    bal_grip = (grip / EMPIRICAL_GRIP_SCALE) * (1.0 / (3.0 ** 0.5))
    global_balanced7d = torch.cat([bal_pos, bal_rot, bal_grip], dim=-1).reshape(N, H * 7)

    # Centered 10D (subtract identity rot6d so near-zero rotations stay near 0)
    rot6d_id = torch.tensor(_ROT6D_IDENTITY, device=dev, dtype=torch.float32).view(1, 1, 6)
    a10_centered = torch.cat(
        [a10[:, :, 0:3], a10[:, :, 3:9] - rot6d_id, a10[:, :, 9:10]], dim=-1
    )
    global_10d = a10_centered.reshape(N, H * 10)

    # 2. Translation
    translation = pos.reshape(N, H * 3)
    net_pos = pos.sum(dim=1)  # (N, 3)
    translation_net_dir = net_pos

    # 3. Rotation (axis-angle 3D trajectory)
    rotation = rot.reshape(N, H * 3)
    net_rot = rot.sum(dim=1)  # (N, 3)

    # 4. Motion Magnitude / Amplitude (direction-invariant!)
    pos_speed = torch.linalg.norm(pos, dim=-1)  # (N, H)
    rot_speed = torch.linalg.norm(rot, dim=-1)  # (N, H)
    pos_path = pos_speed.sum(dim=-1, keepdim=True)  # (N, 1)
    net_disp_norm = torch.linalg.norm(net_pos, dim=-1, keepdim=True)  # (N, 1)
    rot_path = rot_speed.sum(dim=-1, keepdim=True)  # (N, 1)
    pos_peak = pos_speed.max(dim=-1, keepdim=True).values  # (N, 1)
    rot_peak = rot_speed.max(dim=-1, keepdim=True).values  # (N, 1)
    # Standardized magnitude summary vector (H pos_speed + H rot_speed + summary scalars)
    magnitude = torch.cat(
        [
            pos_speed / EMPIRICAL_POS_SCALE,
            rot_speed / EMPIRICAL_ROT_SCALE * 0.5,
            pos_path / (H * EMPIRICAL_POS_SCALE),
            net_disp_norm / (H * EMPIRICAL_POS_SCALE),
            rot_path / (H * EMPIRICAL_ROT_SCALE) * 0.5,
            pos_peak / EMPIRICAL_POS_SCALE,
            rot_peak / EMPIRICAL_ROT_SCALE * 0.5,
        ],
        dim=-1,
    )  # (N, 2*H + 5)

    # 5. Temporal Structure (direction-invariant AND scale-invariant!)
    # 5a. L2-normalized speed envelope (shape of acceleration/deceleration/pause)
    speed_norm = torch.linalg.norm(pos_speed, dim=-1, keepdim=True).clamp_min(eps)
    speed_env = pos_speed / speed_norm  # (N, H), unit norm when moving
    # 5b. Step-to-step direction persistence / turning cosine
    if H > 1:
        p_unit = pos / pos_speed.unsqueeze(-1).clamp_min(eps)
        turn_cos = (p_unit[:, :-1] * p_unit[:, 1:]).sum(dim=-1)  # (N, H-1)
        both_moving = (pos_speed[:, :-1] > 0.05) & (pos_speed[:, 1:] > 0.05)
        turn_cos = torch.where(both_moving, turn_cos, torch.zeros_like(turn_cos))
        # Normalized speed derivative (accel vs decel shape)
        speed_diff = (pos_speed[:, 1:] - pos_speed[:, :-1]) / speed_norm  # (N, H-1)
        # Gripper change timing profile |Δgrip_t|
        grip_change = (grip[:, 1:, 0] - grip[:, :-1, 0]).abs() * 0.5  # (N, H-1) in [0, 1]
    else:
        turn_cos = torch.zeros((N, 0), device=dev)
        speed_diff = torch.zeros((N, 0), device=dev)
        grip_change = torch.zeros((N, 0), device=dev)
    # 5c. Temporal center-of-mass (early vs late activity, in [0, 1])
    t_grid = torch.linspace(0.0, 1.0, H, device=dev, dtype=torch.float32).view(1, H)
    speed_sum = pos_speed.sum(dim=-1, keepdim=True).clamp_min(eps)
    t_com = (pos_speed * t_grid).sum(dim=-1, keepdim=True) / speed_sum
    early_frac = pos_speed[:, : H // 2].sum(dim=-1, keepdim=True) / speed_sum
    # Straightness = net_disp / path_len (1.0 = straight line, <1 = curved/back-and-forth)
    straightness = (net_disp_norm / speed_sum).clamp(0.0, 1.0)
    temporal = torch.cat(
        [
            speed_env,
            speed_diff,
            turn_cos * 0.35,
            grip_change * 0.5,
            t_com,
            early_frac,
            straightness,
        ],
        dim=-1,
    )

    # 6. Gripper State (16-step trajectory + onset/offset delta)
    grip_seq = grip.squeeze(-1)  # (N, H) in [-1, 1]
    gripper = grip_seq

    # Discrete gripper regime: 0=open, 1=closed, 2=closing (grasp), 3=opening (release)
    all_open = (grip_seq < 0.0).all(dim=-1)
    all_closed = (grip_seq > 0.0).all(dim=-1)
    closing = (~all_open) & (~all_closed) & (grip_seq[:, -1] > grip_seq[:, 0])
    opening = (~all_open) & (~all_closed) & (grip_seq[:, -1] < grip_seq[:, 0])
    grip_regime = torch.full((N,), 0, device=dev, dtype=torch.long)
    grip_regime = torch.where(all_closed, torch.ones_like(grip_regime), grip_regime)
    grip_regime = torch.where(closing, torch.full_like(grip_regime, 2), grip_regime)
    grip_regime = torch.where(opening, torch.full_like(grip_regime, 3), grip_regime)

    # Activity masks
    is_moving = pos_path.squeeze(-1) > 0.5
    is_rotating = rot_path.squeeze(-1) > 0.5

    return {
        "global_raw7d": global_raw7d,
        "global_balanced7d": global_balanced7d,
        "global_10d": global_10d,
        "translation": translation,
        "translation_net_dir": translation_net_dir,
        "rotation": rotation,
        "rotation_net": net_rot,
        "magnitude": magnitude,
        "temporal": temporal,
        "gripper": gripper,
        "pos_path": pos_path.squeeze(-1),
        "rot_path": rot_path.squeeze(-1),
        "is_moving": is_moving,
        "is_rotating": is_rotating,
        "gripper_regime": grip_regime,
    }


def pairwise_cosine_sim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise cosine similarity `(N, N)` for feature matrix `x` `(N, D)`."""
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    x_unit = x / norms.clamp_min(eps)
    sim = x_unit @ x_unit.T
    # Zero vectors get similarity 0 with non-zero vectors, 1 with themselves
    is_zero = (norms.squeeze(-1) <= eps)
    if is_zero.any():
        both_zero = is_zero[:, None] & is_zero[None, :]
        either_zero = is_zero[:, None] | is_zero[None, :]
        sim = torch.where(either_zero, torch.zeros_like(sim), sim)
        sim = torch.where(both_zero, torch.ones_like(sim), sim)
    return sim.clamp(-1.0, 1.0)


def pairwise_l2_dist(x: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance `(N, N)` for feature matrix `x` `(N, D)`."""
    return torch.cdist(x.float(), x.float(), p=2)


def pairwise_rbf_sim(x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Pairwise Gaussian RBF similarity `exp(-||x_i - x_j||^2 / (2 sigma^2))` in `(0, 1]`.

    If `sigma` is None, uses the median off-diagonal pairwise L2 distance (adaptive scale).
    """
    dist = pairwise_l2_dist(x)
    if sigma is None:
        n = dist.shape[0]
        if n > 1:
            off = dist[~torch.eye(n, device=dist.device, dtype=torch.bool)]
            sigma_val = float(off.median().item())
        else:
            sigma_val = 1.0
        sigma = max(sigma_val, 1e-4)
    return torch.exp(-0.5 * (dist / float(sigma)).square())


def compute_attribute_similarity_matrix(
    attrs: Dict[str, torch.Tensor],
    attr_name: str,
    *,
    metric: str = "auto",
) -> torch.Tensor:
    """Compute `(N, N)` similarity matrix for a named behavioral attribute.

    Metric defaults (`metric="auto"`):
      - `global_raw7d`, `global_balanced7d`, `translation`, `translation_net_dir`,
        `rotation`, `gripper`: cosine similarity in `[-1, 1]`.
      - `global_10d`: RBF similarity on centered 10D trajectory (captures scale + direction).
      - `magnitude`: RBF similarity (because magnitude is positive-valued & scale-sensitive;
        cosine on positive speed scalars loses overall amplitude!).
      - `temporal`: cosine similarity of centered temporal shape features (captures profile
        shape & timing alignment).
    """
    feat = attrs[attr_name].float()
    if metric == "cosine":
        return pairwise_cosine_sim(feat)
    if metric == "centered_cosine":
        return pairwise_cosine_sim(feat - feat.mean(dim=0, keepdim=True))
    if metric == "neg_l2":
        return -pairwise_l2_dist(feat)
    if metric == "rbf":
        return pairwise_rbf_sim(feat)

    # Auto metric per attribute semantics
    if attr_name in (
        "global_raw7d",
        "global_balanced7d",
        "translation",
        "translation_net_dir",
        "rotation",
        "gripper",
    ):
        return pairwise_cosine_sim(feat)
    if attr_name == "global_10d":
        return pairwise_rbf_sim(feat)
    if attr_name == "magnitude":
        return pairwise_rbf_sim(feat)
    if attr_name == "temporal":
        return pairwise_cosine_sim(feat - feat.mean(dim=0, keepdim=True))
    raise ValueError(f"Unknown attribute name: {attr_name}")


def discretize_behavior_factors(actions_7d: np.ndarray) -> Dict[str, np.ndarray]:
    """Assign interpretable discrete behavioral factor labels to each `(N, 16, 7)` chunk.

    Used for Unseen Behavior Combination generalization tests:
      - `dir_label`: 7 classes (`still`, `+x`, `-x`, `+y`, `-y`, `+z`, `-z`) from dominant
        net translation axis.
      - `grip_label`: 4 classes (`open`, `closed`, `closing`, `opening`).
      - `speed_label`: 3 classes (`slow`, `medium`, `fast`) via 33%/67% quantiles of pos_path.
      - `combo_dir_grip`: string `"{dir_label}|{grip_label}"`.
      - `combo_dir_speed`: string `"{dir_label}|{speed_label}"`.
    """
    pos = actions_7d[:, :, 0:3]
    grip = actions_7d[:, :, 6]
    net_pos = pos.sum(axis=1)  # (N, 3)
    pos_path = np.linalg.norm(pos, axis=-1).sum(axis=-1)
    net_norm = np.linalg.norm(net_pos, axis=-1)

    axis_names = [("+x", "-x"), ("+y", "-y"), ("+z", "-z")]
    dir_labels = []
    for i in range(len(actions_7d)):
        if net_norm[i] < 1.0:
            dir_labels.append("still")
        else:
            ax = int(np.argmax(np.abs(net_pos[i])))
            dir_labels.append(axis_names[ax][0] if net_pos[i, ax] >= 0 else axis_names[ax][1])
    dir_arr = np.array(dir_labels, dtype=object)

    grip_labels = []
    for i in range(len(actions_7d)):
        g = grip[i]
        if (g < 0).all():
            grip_labels.append("open")
        elif (g > 0).all():
            grip_labels.append("closed")
        elif g[-1] > g[0]:
            grip_labels.append("closing")
        else:
            grip_labels.append("opening")
    grip_arr = np.array(grip_labels, dtype=object)

    q33, q67 = np.quantile(pos_path, [1.0 / 3.0, 2.0 / 3.0])
    speed_arr = np.where(
        pos_path <= q33, "slow", np.where(pos_path <= q67, "medium", "fast")
    ).astype(object)

    combo_dg = np.array([f"{d}|{g}" for d, g in zip(dir_arr, grip_arr)], dtype=object)
    combo_ds = np.array([f"{d}|{s}" for d, s in zip(dir_arr, speed_arr)], dtype=object)

    return {
        "dir_label": dir_arr,
        "grip_label": grip_arr,
        "speed_label": speed_arr,
        "combo_dir_grip": combo_dg,
        "combo_dir_speed": combo_ds,
    }


def extract_action_delta_features(
    actions_7d_a: torch.Tensor,
    actions_7d_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute structured behavioral change `Δa = a_b - a_a` vectors for displacement analysis."""
    da = (actions_7d_b - actions_7d_a).float()  # (M, H, 7)
    M, H, _ = da.shape
    d_pos = da[:, :, 0:3]
    d_rot = da[:, :, 3:6]
    d_grip = da[:, :, 6:7]

    bal = torch.cat(
        [
            (d_pos / EMPIRICAL_POS_SCALE) * (1.0 / (3.0 ** 0.5)),
            (d_rot / EMPIRICAL_ROT_SCALE) * ((2.0 / 3.0) ** 0.5 / (3.0 ** 0.5)),
            (d_grip / EMPIRICAL_GRIP_SCALE) * (1.0 / (3.0 ** 0.5)),
        ],
        dim=-1,
    ).reshape(M, H * 7)

    # Magnitude change
    spd_a = torch.linalg.norm(actions_7d_a[:, :, 0:3].float(), dim=-1)
    spd_b = torch.linalg.norm(actions_7d_b[:, :, 0:3].float(), dim=-1)
    d_mag = (spd_b - spd_a) / EMPIRICAL_POS_SCALE

    return {
        "delta_raw7d": da.reshape(M, H * 7),
        "delta_balanced7d": bal,
        "delta_translation": d_pos.reshape(M, H * 3),
        "delta_rotation": d_rot.reshape(M, H * 3),
        "delta_gripper": d_grip.reshape(M, H),
        "delta_magnitude": d_mag,
    }
