"""Audit video-derived relation signals vs GT action relations & anti-degeneracy checks.

Directly addresses E16-A and Requirement 4 (preventing degeneracy into scene matching,
object matching, task classification, or representation collapse) by quantitatively
comparing ALL candidate video-only relation sources on the held-out validation split:
  - `static_scene_v0`   (degenerate lower-bound: raw static frame matching!)
  - `random_rel`        (null control)
  - `feat_delta`        (raw scene-subtracted temporal CLIP delta)
  - `feat_delta_decorr` (scene-decorrelated temporal CLIP delta)
  - `motion_flow`       (pixel-space optical flow + integrated 4x4 grid point-tracks)
  - `pretrained_lam`    (100% video-only pretrained LAM dynamics bottleneck)
  - `multiview_vid`     (motion_flow + pretrained_lam)
  - `gt_action_oracle`  (E12 GT balanced 7D action oracle upper bound)

Measures for EVERY source:
  1. Cross-task Behavioral Alignment: Spearman rho vs GT `bal7d`, `trans`, `rot`, `mag`,
     `temp`, `grip`, AND pairwise displacement alignment `rho(cos(Δu), cos(Δa))`!
  2. Cross-task Retrieval Quality: Top-1 / Top-5 GT action 10D MSE of cross-task kNN!
  3. Anti-Degeneracy / Leakage Audit:
     - Same-Task kNN@10 Leakage Rate (lower = less task-classification shortcut!)
     - Same-Object kNN@10 Leakage Rate (lower = less object-matching shortcut!)
     - Scene-Similarity Correlation `rho(S_rel, S_v0)` (lower = less scene matching!)
     - Effective Rank `erank(u)` (higher = no representation collapse!)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from e5.train import atomic_write_json
from e12.action_attributes import (
    compute_attribute_similarity_matrix,
    extract_action_attribute_vectors,
    extract_action_delta_features,
    pairwise_cosine_sim,
)
from e12.eval_suite import ValClipCache, _effective_rank_stats, _spearman_pearson_from_vectors

from . import MOTION_CACHE_DIR, OUTPUT_ROOT
from .pretrain_video_only import load_pretrained_video_lam
from .video_relations import extract_video_relation_features


def _gather_val_motion(
    cache: ValClipCache, episodes: np.ndarray, frames: np.ndarray, motion_dir: Path
) -> np.ndarray:
    m_all = np.load(motion_dir / "motion_features.npy", mmap_mode="r")
    m_mean = np.load(motion_dir / "motion_mean.npy", allow_pickle=False).astype(np.float32)
    m_std = np.load(motion_dir / "motion_std.npy", allow_pickle=False).astype(np.float32)

    N = len(episodes)
    out = np.zeros((N, 16, 2, 64), dtype=np.float32)
    win = np.arange(16, dtype=np.int64)
    for i in range(N):
        ep = int(episodes[i])
        fr = int(frames[i])
        ep_len = int(cache.ep_lengths_np[ep])
        off = int(cache.frame_offsets_np[ep])
        steps = fr + win
        pad = steps >= ep_len
        clamped = np.minimum(steps, max(ep_len - 1, 0))
        arr = m_all[off + clamped].astype(np.float32)
        arr = np.clip((arr - m_mean[None]) / m_std[None], -8.0, 8.0)
        arr[pad] = 0.0
        out[i] = arr
    return out


def audit_all_video_relations(
    *,
    lam_ckpt: Optional[Path] = None,
    out_dir: Path = OUTPUT_ROOT / "relation_audit",
    max_samples: int = 2048,
    seed: int = 42,
) -> Dict[str, Any]:
    from e5.actions import action_10d_to_7d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=seed)

    # Subsample non-tail validation clips deterministically
    is_tail = np.array(
        [int(fr) + 16 > int(cache.ep_lengths_np[int(ep)]) for ep, fr in zip(cache.val_episodes, cache.val_frames)],
        dtype=bool,
    )
    non_tail = np.flatnonzero(~is_tail)
    rng = np.random.default_rng(seed + 1601)
    sel = np.sort(rng.choice(non_tail, size=min(max_samples, len(non_tail)), replace=False))
    N = len(sel)

    sel_eps = cache.val_episodes[sel]
    sel_frs = cache.val_frames[sel]
    cr_cpu, cm_cpu, a10_cpu, _ = cache.gather(sel_eps, sel_frs)
    clip_rgb = cr_cpu.to(device=device, dtype=torch.float32)
    clip_mask = cm_cpu.to(device=device)
    a10 = a10_cpu.to(device=device, dtype=torch.float32)
    a7 = action_10d_to_7d(a10)
    ep_ids = torch.from_numpy(sel_eps).to(device=device)
    fr_ids = torch.from_numpy(sel_frs).to(device=device)
    task_ids = torch.from_numpy(cache.val_task_ids[sel]).to(device=device)

    # Object labels from procedure string (same definition as E13/E15!)
    procs = [cache.episodes[int(e)]["procedure"].split("|")[0].strip() for e in sel_eps]
    obj_names = [(p.split()[1] if len(p.split()) >= 2 else "other") for p in procs]
    obj_vocab = {o: i for i, o in enumerate(sorted(set(obj_names)))}
    obj_ids = torch.tensor([obj_vocab[o] for o in obj_names], device=device, dtype=torch.long)

    clip_motion = None
    if (MOTION_CACHE_DIR / "motion_features.npy").is_file():
        m_val = _gather_val_motion(cache, sel_eps, sel_frs, MOTION_CACHE_DIR)
        clip_motion = torch.from_numpy(m_val).to(device=device, dtype=torch.float32)

    lam_enc = load_pretrained_video_lam(lam_ckpt, device=device) if (lam_ckpt and lam_ckpt.is_file()) else None

    # GT action similarities and static scene similarity
    attrs = extract_action_attribute_vectors(a10, a7)
    S_gt = {
        "bal7d": compute_attribute_similarity_matrix(attrs, "global_balanced7d"),
        "trans": compute_attribute_similarity_matrix(attrs, "translation"),
        "rot": compute_attribute_similarity_matrix(attrs, "rotation"),
        "mag": compute_attribute_similarity_matrix(attrs, "magnitude"),
        "temp": compute_attribute_similarity_matrix(attrs, "temporal"),
        "grip": compute_attribute_similarity_matrix(attrs, "gripper"),
    }
    v0_flat = clip_rgb[:, 0].reshape(N, -1)
    S_scene = pairwise_cosine_sim(v0_flat)

    cross_task = task_ids[:, None] != task_ids[None, :]
    same_task_off = (task_ids[:, None] == task_ids[None, :]) & (~torch.eye(N, device=device, dtype=torch.bool))
    same_obj_off = (obj_ids[:, None] == obj_ids[None, :]) & (~torch.eye(N, device=device, dtype=torch.bool))
    off_diag = ~torch.eye(N, device=device, dtype=torch.bool)

    # Fixed 1024 cross-task transition pairs for displacement consistency audit
    rng_p = np.random.default_rng(seed + 1602)
    M = 1024
    p1 = rng_p.integers(0, N, size=M)
    p2 = rng_p.integers(0, N, size=M)
    da = extract_action_delta_features(a7[p1], a7[p2])["delta_balanced7d"]
    da_act = torch.linalg.norm(da, dim=-1) > 0.5
    S_da = pairwise_cosine_sim(da)
    off_m = (~torch.eye(M, device=device, dtype=torch.bool)) & da_act[:, None] & da_act[None, :]

    sources: List[str] = ["static_scene_v0", "random_rel", "feat_delta", "feat_delta_decorr"]
    if clip_motion is not None:
        sources.append("motion_flow")
    if lam_enc is not None:
        sources.append("pretrained_lam")
    if clip_motion is not None and lam_enc is not None:
        sources.append("multiview_vid")
    sources.append("gt_action_oracle")

    results: Dict[str, Any] = {}
    a10_flat = a10.reshape(N, -1)

    for src in sources:
        if src == "static_scene_v0":
            u = v0_flat
        elif src == "gt_action_oracle":
            u = attrs["global_balanced7d"]
        else:
            u = extract_video_relation_features(
                src,
                clip_rgb,
                clip_mask,
                clip_motion,
                lam_encoder=lam_enc,
                ep_ids=ep_ids,
                frame_ids=fr_ids,
            )

        S_u = pairwise_cosine_sim(u)
        er = _effective_rank_stats(u.detach().cpu().numpy(), device=device)

        # 1. Cross-task behavioral alignment against GT attributes
        ct_rho = {}
        for k, S_a in S_gt.items():
            rho, _ = _spearman_pearson_from_vectors(S_u[cross_task], S_a[cross_task], seed=seed)
            ct_rho[k] = float(rho)

        # 2. Displacement consistency rho(cos(Δu), cos(Δa))
        du = u[p2] - u[p1]
        S_du = pairwise_cosine_sim(du)
        disp_rho, disp_r = _spearman_pearson_from_vectors(S_du[off_m], S_da[off_m], seed=seed)

        # 3. Cross-task Top-1 & Top-5 action retrieval MSE10
        S_ct = S_u.masked_fill(~cross_task, -1e9)
        top5 = torch.topk(S_ct, k=5, dim=-1).indices  # (N, 5)
        top1_mse = float((a10_flat - a10_flat[top5[:, 0]]).square().mean().item())
        top5_mse = float(
            (a10_flat[:, None, :] - a10_flat[top5]).square().mean().item()
        )

        # 4. Anti-degeneracy leakage audit (all off-diag pairs!)
        S_all = S_u.masked_fill(~off_diag, -1e9)
        knn10 = torch.topk(S_all, k=10, dim=-1).indices  # (N, 10)
        same_task_rate = float((task_ids[knn10] == task_ids[:, None]).float().mean().item())
        same_obj_rate = float((obj_ids[knn10] == obj_ids[:, None]).float().mean().item())
        scene_rho, _ = _spearman_pearson_from_vectors(S_u[off_diag], S_scene[off_diag], seed=seed)

        results[src] = {
            "dim": int(u.shape[1]),
            "erank": float(er["erank_energy"]),
            "ct_rho": ct_rho,
            "disp_rho_bal7d": float(disp_rho),
            "disp_pearson_bal7d": float(disp_r),
            "ct_top1_mse10": top1_mse,
            "ct_top5_mse10": top5_mse,
            "same_task_knn10_leak": same_task_rate,
            "same_obj_knn10_leak": same_obj_rate,
            "scene_sim_rho": float(scene_rho),
        }
        print(
            f"[Audit:{src:18s}] CT_rho(bal/tr/mg/gr)="
            f"{ct_rho['bal7d']:.3f}/{ct_rho['trans']:.3f}/{ct_rho['mag']:.3f}/{ct_rho['grip']:.3f} | "
            f"DispRho={disp_rho:.3f} | Top1MSE={top1_mse:.4f} | "
            f"Leak(Task/Obj/Scene)={same_task_rate*100:.1f}%/{same_obj_rate*100:.1f}%/{scene_rho:.3f} | "
            f"erank={er['erank_energy']:.1f}",
            flush=True,
        )

    atomic_write_json({"num_samples": N, "sources": results}, out_dir / "video_relation_audit.json")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lam-ckpt", type=Path, default=OUTPUT_ROOT / "e16_d1_lam" / "last_ema.pt")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "relation_audit")
    p.add_argument("--max-samples", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    audit_all_video_relations(
        lam_ckpt=args.lam_ckpt, out_dir=args.out_dir, max_samples=args.max_samples, seed=args.seed
    )


if __name__ == "__main__":
    main()
