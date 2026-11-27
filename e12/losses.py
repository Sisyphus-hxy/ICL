"""Relational representation learning losses for E12-A, E12-B, and E12-C.

All losses operate purely on E9 Stage 1's `(B, 4, 768)` behavior latent `b_t`
and ground-truth 16-step 10D/7D actions, keeping E9's Encoder, Decoder, latent
dimension, and original action diffusion reconstruction objective untouched.

Variants:
  - `control` (E9-Cont): Pure `L_diff` only (matched-step control).
  - `e12_a`   (E12-A):   Global Action Similarity Supervision (arXiv:2609.19846).
                         Aligns off-diagonal pairwise latent cosine similarity with
                         scale-balanced 7D action cosine similarity.
  - `e12_b`   (E12-B):   Multi-Attribute Relational Supervision.
                         Factorizes E9's 4 behavior tokens across 4 behavioral
                         attributes (Token 0: Translation, Token 1: Rotation,
                         Token 2: Magnitude+Temporal, Token 3: Gripper) so globally
                         distinct samples sharing a behavioral attribute connect
                         cleanly in that attribute's token subspace.
  - `e12_c`   (E12-C):   Behavior Displacement Consistency.
                         Supervises pairwise directional and relative-magnitude
                         consistency of latent displacements `Δz = z(y) - z(x)`
                         against action changes `Δa = a(y) - a(x)` across transitions
                         from different scenes/episodes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .action_attributes import (
    compute_attribute_similarity_matrix,
    extract_action_attribute_vectors,
    extract_action_delta_features,
    pairwise_cosine_sim,
    pairwise_rbf_sim,
)


def _off_diag_mask(n: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(n, device=device, dtype=torch.bool)


def compute_e12_a_loss(
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """E12-A: Global Action Similarity Supervision (arXiv:2609.19846).

    Matches pairwise cosine similarity of flattened behavior latents `z_i in R^3072`
    to pairwise cosine similarity of scale-balanced 7D delta action sequences.
    """
    B = b_t.shape[0]
    z = b_t.float().reshape(B, -1)
    attrs = extract_action_attribute_vectors(actions_10d, actions_7d)
    S_z = pairwise_cosine_sim(z)
    with torch.no_grad():
        S_a = compute_attribute_similarity_matrix(attrs, "global_balanced7d")
    off = _off_diag_mask(B, z.device)
    loss = F.mse_loss(S_z[off], S_a[off])
    return loss, {
        "rel_loss": loss.detach(),
        "rel_global_mse": loss.detach(),
        "s_z_off_mean": S_z[off].detach().mean(),
        "s_a_off_mean": S_a[off].detach().mean(),
    }


def compute_e12_b_loss(
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """E12-B: Multi-Attribute Relational Supervision across the 4 behavior tokens.

    Token assignment (zero architecture change to E9's `(B, 4, 768)` latent):
      - Token 0 -> Translation (3D delta-pos trajectory similarity)
      - Token 1 -> Rotation (3D axis-angle similarity combining active cosine + RBF)
      - Token 2 -> Motion Magnitude & Temporal Structure (50% mapped RBF mag + 50% centered temporal cos)
      - Token 3 -> Gripper State (16-step gripper trajectory cosine)
    """
    if b_t.ndim != 3 or b_t.shape[1] != 4:
        raise ValueError(f"Expected b_t (B, 4, D), got {tuple(b_t.shape)}")
    B = b_t.shape[0]
    dev = b_t.device
    off = _off_diag_mask(B, dev)

    attrs = extract_action_attribute_vectors(actions_10d, actions_7d)

    # Target similarity matrices in [-1, 1] (computed with no_grad from GT actions)
    with torch.no_grad():
        # Token 0: Translation (cosine on 3D delta-pos trajectory; RBF blend stabilizes near-still pairs)
        S_trans_cos = compute_attribute_similarity_matrix(attrs, "translation", metric="cosine")
        S_trans_rbf = 2.0 * pairwise_rbf_sim(attrs["translation"]) - 1.0
        S_t0 = 0.75 * S_trans_cos + 0.25 * S_trans_rbf

        # Token 1: Rotation (for rotating pairs use cosine+RBF; for non-rotating pairs rely on RBF so zero-rot doesn't add noise)
        S_rot_cos = compute_attribute_similarity_matrix(attrs, "rotation", metric="cosine")
        S_rot_rbf = 2.0 * pairwise_rbf_sim(attrs["rotation"]) - 1.0
        both_rot = attrs["is_rotating"][:, None] & attrs["is_rotating"][None, :]
        S_t1 = torch.where(both_rot, 0.65 * S_rot_cos + 0.35 * S_rot_rbf, S_rot_rbf)

        # Token 2: Magnitude (mapped to [-1, 1]) + Temporal Structure (centered cosine in [-1, 1])
        S_mag = 2.0 * compute_attribute_similarity_matrix(attrs, "magnitude", metric="rbf") - 1.0
        S_temp = compute_attribute_similarity_matrix(attrs, "temporal", metric="centered_cosine")
        S_t2 = 0.50 * S_mag + 0.50 * S_temp

        # Token 3: Gripper State (16-step cosine in [-1, 1])
        S_t3 = compute_attribute_similarity_matrix(attrs, "gripper", metric="cosine")

    targets = (S_t0, S_t1, S_t2, S_t3)
    names = ("tok0_trans", "tok1_rot", "tok2_mag_temp", "tok3_grip")

    losses = []
    stats: Dict[str, torch.Tensor] = {}
    for k, (S_tgt, name) in enumerate(zip(targets, names)):
        S_zk = pairwise_cosine_sim(b_t[:, k, :].float())
        lk = F.mse_loss(S_zk[off], S_tgt[off])
        losses.append(lk)
        stats[f"rel_{name}_mse"] = lk.detach()

    total_rel = torch.stack(losses).mean()
    stats["rel_loss"] = total_rel.detach()
    return total_rel, stats


def compute_e12_c_loss(
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
    *,
    norm_weight: float = 0.25,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """E12-C: Behavior Displacement Consistency.

    Splits the batch `(B, ...)` into `M = B // 2` ordered transition pairs
    `(2m -> 2m+1)` and supervises:
      1. Directional displacement consistency:
         `cos(Δz_m, Δz_n)` matches `cos(Δa_m, Δa_n)` across all pairs `m != n`
         with non-trivial action change (`||Δa|| > 0.5`).
      2. Relative displacement scale consistency:
         `||Δz_m|| / mean(||Δz||)` matches `||Δa_m|| / mean(||Δa||)` so latent
         displacements transfer directly via `z_B1 + (z_A2 - z_A1)` without
         requiring context-specific rescaling.
    """
    B = b_t.shape[0]
    M = B // 2
    if M < 4:
        zero = b_t.sum() * 0.0
        return zero, {"rel_loss": zero.detach()}

    z = b_t[: 2 * M].float().reshape(M, 2, -1)
    dz = z[:, 1, :] - z[:, 0, :]  # (M, 3072)

    if actions_7d is None:
        from e5.actions import action_10d_to_7d

        a7 = action_10d_to_7d(actions_10d[: 2 * M].float())
    else:
        a7 = actions_7d[: 2 * M].float()
    a7_pair = a7.reshape(M, 2, a7.shape[1], a7.shape[2])

    with torch.no_grad():
        da_dict = extract_action_delta_features(a7_pair[:, 0], a7_pair[:, 1])
        da = da_dict["delta_balanced7d"]  # (M, 112)
        da_norm = torch.linalg.norm(da, dim=-1)  # (M,)
        active = da_norm > 0.5
        S_da = pairwise_cosine_sim(da)

    S_dz = pairwise_cosine_sim(dz)
    off = _off_diag_mask(M, z.device)
    valid_pairs = off & active[:, None] & active[None, :]
    if int(valid_pairs.sum().item()) > 0:
        dir_loss = F.mse_loss(S_dz[valid_pairs], S_da[valid_pairs])
    else:
        dir_loss = F.mse_loss(S_dz[off], S_da[off])

    # Relative scale consistency (normalized displacement norms)
    dz_norm = torch.linalg.norm(dz, dim=-1)  # (M,)
    dz_rel = dz_norm / dz_norm.mean().clamp_min(eps)
    da_rel = da_norm / da_norm.mean().clamp_min(eps)
    scale_loss = F.mse_loss(dz_rel, da_rel)

    total = dir_loss + float(norm_weight) * scale_loss
    return total, {
        "rel_loss": total.detach(),
        "disp_dir_mse": dir_loss.detach(),
        "disp_scale_mse": scale_loss.detach(),
        "dz_norm_mean": dz_norm.detach().mean(),
        "active_pair_frac": valid_pairs.float().mean().detach(),
    }


def compute_relational_loss(
    variant: str,
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
    *,
    disp_norm_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Dispatch relational loss for `control | e12_a | e12_b | e12_c`."""
    if variant == "control":
        zero = b_t.sum() * 0.0
        return zero, {"rel_loss": zero.detach()}
    if variant == "e12_a":
        return compute_e12_a_loss(b_t, actions_10d, actions_7d)
    if variant == "e12_b":
        return compute_e12_b_loss(b_t, actions_10d, actions_7d)
    if variant == "e12_c":
        return compute_e12_c_loss(
            b_t, actions_10d, actions_7d, norm_weight=disp_norm_weight
        )
    raise ValueError(f"Unknown E12 variant: {variant}")
