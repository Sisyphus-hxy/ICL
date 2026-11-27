"""Extract full-dataset pixel-space dense optical flow & multi-grid point-track motion features.

Operates strictly on raw dual-camera RGB videos (`agentview_rgb`, `eye_in_hand_rgb`,
128x128 uint8) across ALL 15,550 LIBERO episodes (3,629,339 frames) with ZERO action
supervision, producing a compact contiguous `(3629339, 2, 64)` float16 array (~929 MB)
aligned 1-to-1 with `e9/outputs/dense_clip_cache/frame_features.npy`.

Per frame transition `t -> t+1` and per camera (64D):
  - [0:32]  4x4 spatial-grid tracked flow `(dx, dy)` (32D): regional motion trajectories
            capturing localized arm/object motion in agentview and gripper/ego-motion
            in eye-in-hand.
  - [32:48] Kinematic & differential flow invariants (16D):
            * active-weighted mean flow `(ux, uy)` (2D)
            * global mean flow `(gux, guy)` (2D)
            * mean speed, p95 speed, active-motion fraction, frame-diff L1 (4D)
            * affine differential invariants: divergence (`du/dx + dv/dy`, Z-depth / zoom),
              curl (`dv/dx - du/dy`, wrist roll rotation), shear1 (`du/dx - dv/dy`),
              shear2 (`du/dy + dv/dx`) (4D)
            * lower-half horizontal convergence `mean(u_left - u_right)` & vertical flow,
              plus upper-half horizontal/vertical flow (4D) — directly sensitive to
              symmetric gripper finger opening/closing in wrist camera!
  - [48:64] Appearance-invariant motion distribution (16D):
            * 8-bin Histogram of Oriented Optical Flow (HOF) weighted by speed (8D)
            * 4-quadrant mean speed (4D)
            * 4-quadrant foreground temporal-diff energy (4D)

Uses 2-level coarse-to-fine GPU Lucas-Kanade optical flow (fast, exact, zero external
weights) with multi-threaded HDF5 readers feeding GPU 0 & GPU 1 in parallel.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from e5.train import atomic_write_json

from . import (
    DEFAULT_DENSE_CACHE_DIR,
    E5_CACHE_DIR,
    MOTION_CACHE_DIR,
    MOTION_FEAT_DIM,
    RAW_LIBERO,
)


class GPUPyramidalFlowExtractor:
    """Fast GPU 2-level coarse-to-fine Lucas-Kanade optical flow & motion descriptor extractor."""

    def __init__(self, device: torch.device, win_size: int = 9):
        self.device = device
        self.win_size = int(win_size)
        self.pad = self.win_size // 2

        # Gaussian spatial smoothing kernel (5x5) and LK integration window (Gaussian-weighted)
        g1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=torch.float32)
        g1d = g1d / g1d.sum()
        self.smooth_k = (g1d[:, None] * g1d[None, :]).view(1, 1, 5, 5)

        coords = torch.linspace(-1.5, 1.5, self.win_size, device=device, dtype=torch.float32)
        w1d = torch.exp(-0.5 * coords * coords)
        w1d = w1d / w1d.sum()
        self.lk_k = (w1d[:, None] * w1d[None, :]).view(1, 1, self.win_size, self.win_size)

        # Sobel-like derivative kernels
        self.kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.ky = self.kx.transpose(-1, -2).contiguous()

        # Precompute base normalized sampling grid for 128x128 warping
        ys = torch.linspace(-1.0, 1.0, 128, device=device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, 128, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.base_grid_128 = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, 128, 128, 2)

        # 8 orientation bin centers for soft HOF
        ang = torch.linspace(-math.pi, math.pi, 9, device=device, dtype=torch.float32)[:-1] + (math.pi / 8.0)
        self.hof_cos = torch.cos(ang).view(1, 8, 1, 1)
        self.hof_sin = torch.sin(ang).view(1, 8, 1, 1)

    def _lk_step(self, i0: torch.Tensor, i1: torch.Tensor, eps: float = 2e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-level Lucas-Kanade flow `(u, v)` in pixels for `(N, 1, H, W)` grayscale inputs."""
        ix = 0.5 * (F.conv2d(i0, self.kx, padding=1) + F.conv2d(i1, self.kx, padding=1))
        iy = 0.5 * (F.conv2d(i0, self.ky, padding=1) + F.conv2d(i1, self.ky, padding=1))
        it = i1 - i0

        sxx = F.conv2d(ix * ix, self.lk_k, padding=self.pad) + eps
        syy = F.conv2d(iy * iy, self.lk_k, padding=self.pad) + eps
        sxy = F.conv2d(ix * iy, self.lk_k, padding=self.pad)
        sxt = F.conv2d(ix * it, self.lk_k, padding=self.pad)
        syt = F.conv2d(iy * it, self.lk_k, padding=self.pad)

        det = (sxx * syy - sxy * sxy).clamp_min(1e-6)
        u = -(syy * sxt - sxy * syt) / det
        v = -(sxx * syt - sxy * sxt) / det
        # Suppress flow on nearly flat regions to avoid amplification of camera noise
        grad_e = sxx + syy - 2.0 * eps
        conf = grad_e / (grad_e + 1e-3)
        u = (u * conf).clamp(-6.0, 6.0)
        v = (v * conf).clamp(-6.0, 6.0)
        return u, v

    @torch.inference_mode()
    def extract_pair_descriptors(
        self,
        rgb0_u8: np.ndarray,
        rgb1_u8: np.ndarray,
        *,
        convention: str = "opengl",
    ) -> np.ndarray:
        """Compute 64D motion descriptor for batch of consecutive RGB frame pairs `(N, 128, 128, 3)`."""
        t0 = torch.from_numpy(rgb0_u8).to(self.device, non_blocking=True).float() / 255.0
        t1 = torch.from_numpy(rgb1_u8).to(self.device, non_blocking=True).float() / 255.0
        if convention == "opengl":
            t0 = torch.flip(t0, dims=[1])
            t1 = torch.flip(t1, dims=[1])

        # Luminance grayscale (N, 1, 128, 128)
        g0 = (0.299 * t0[..., 0] + 0.587 * t0[..., 1] + 0.114 * t0[..., 2]).unsqueeze(1)
        g1 = (0.299 * t1[..., 0] + 0.587 * t1[..., 1] + 0.114 * t1[..., 2]).unsqueeze(1)
        frame_diff = (t1 - t0).abs().mean(dim=-1, keepdim=False).unsqueeze(1)  # (N, 1, 128, 128)

        # Smooth full-res and build coarse 64x64 level
        g0_s = F.conv2d(g0, self.smooth_k, padding=2)
        g1_s = F.conv2d(g1, self.smooth_k, padding=2)
        g0_c = F.avg_pool2d(g0_s, 2)  # (N, 1, 64, 64)
        g1_c = F.avg_pool2d(g1_s, 2)

        # Level 1: coarse flow (at 64x64, 1 px = 2 px at full-res)
        u_c, v_c = self._lk_step(g0_c, g1_c)
        u_up = F.interpolate(u_c * 2.0, size=(128, 128), mode="bilinear", align_corners=True)
        v_up = F.interpolate(v_c * 2.0, size=(128, 128), mode="bilinear", align_corners=True)

        # Warp g1_s backward by (u_up, v_up) for fine residual LK refinement
        N = g0.shape[0]
        grid = self.base_grid_128.expand(N, -1, -1, -1).clone()
        grid[..., 0] = grid[..., 0] + u_up[:, 0] * (2.0 / 127.0)
        grid[..., 1] = grid[..., 1] + v_up[:, 0] * (2.0 / 127.0)
        g1_warp = F.grid_sample(g1_s, grid, mode="bilinear", padding_mode="border", align_corners=True)

        du, dv = self._lk_step(g0_s, g1_warp)
        u = (u_up + du).clamp(-10.0, 10.0)  # (N, 1, 128, 128)
        v = (v_up + dv).clamp(-10.0, 10.0)
        spd = torch.sqrt(u * u + v * v + 1e-8)

        # 1. [0:32] 4x4 spatial grid tracked flow (dx, dy)
        u_g4 = F.adaptive_avg_pool2d(u, (4, 4)).reshape(N, 16)
        v_g4 = F.adaptive_avg_pool2d(v, (4, 4)).reshape(N, 16)
        grid_flow = torch.cat([u_g4, v_g4], dim=-1)  # (N, 32)

        # 2. [32:48] Kinematic & affine differential invariants (16D)
        # Active foreground motion weights (emphasizes moving arm/object, ignores static bg)
        act_w = (spd - 0.08).clamp_min(0.0)
        w_sum = act_w.sum(dim=(2, 3)).clamp_min(1e-5)  # (N, 1)
        act_ux = (u * act_w).sum(dim=(2, 3)) / w_sum
        act_uy = (v * act_w).sum(dim=(2, 3)) / w_sum
        glob_ux = u.mean(dim=(2, 3))
        glob_uy = v.mean(dim=(2, 3))

        mean_spd = spd.mean(dim=(2, 3))
        max_spd = spd.amax(dim=(2, 3))
        act_frac = (spd > 0.15).float().mean(dim=(2, 3))
        mean_diff = frame_diff.mean(dim=(2, 3))

        # Spatial derivatives of flow field -> divergence, curl, shear1, shear2
        dudx = u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]
        dudy = u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]
        dvdx = v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]
        dvdy = v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]
        div = (dudx + dvdy).mean(dim=(2, 3)) * 5.0
        curl = (dvdx - dudy).mean(dim=(2, 3)) * 5.0
        shear1 = (dudx - dvdy).mean(dim=(2, 3)) * 5.0
        shear2 = (dudy + dvdx).mean(dim=(2, 3)) * 5.0

        # Lower-half horizontal convergence (gripper fingers closing/opening!) + regional means
        u_low_l = u[:, :, 64:, :64].mean(dim=(2, 3))
        u_low_r = u[:, :, 64:, 64:].mean(dim=(2, 3))
        low_conv = u_low_l - u_low_r  # >0: inward closing, <0: outward opening!
        v_low = v[:, :, 64:, :].mean(dim=(2, 3))
        u_up_m = u[:, :, :64, :].mean(dim=(2, 3))
        v_up_m = v[:, :, :64, :].mean(dim=(2, 3))

        kinematics = torch.cat(
            [
                act_ux, act_uy, glob_ux, glob_uy,
                mean_spd, max_spd, act_frac, mean_diff,
                div, curl, shear1, shear2,
                low_conv, v_low, u_up_m, v_up_m,
            ],
            dim=-1,
        )  # (N, 16)

        # 3. [48:64] 8-bin soft HOF + 4-quadrant speed + 4-quadrant diff energy (16D)
        inv_spd = 1.0 / spd.clamp_min(1e-4)
        u_dir = u * inv_spd
        v_dir = v * inv_spd
        align = (u_dir * self.hof_cos + v_dir * self.hof_sin).clamp_min(0.0).pow(4)  # (N, 8, 128, 128)
        hof = (align * spd).mean(dim=(2, 3))  # (N, 8)
        q_spd = F.adaptive_avg_pool2d(spd, (2, 2)).reshape(N, 4)
        q_diff = F.adaptive_avg_pool2d(frame_diff, (2, 2)).reshape(N, 4)

        desc = torch.cat([grid_flow, kinematics, hof, q_spd, q_diff], dim=-1)  # (N, 64)
        return desc.to(torch.float16).cpu().numpy()


def _process_task_shard(
    gpu_idx: int,
    tasks_shard: List[str],
    episodes: List[Dict[str, Any]],
    task_to_eps: Dict[str, List[int]],
    frame_offsets: np.ndarray,
    tmp_memmap_path: str,
    total_frames: int,
    batch_size: int,
) -> Dict[str, int]:
    """Process a disjoint shard of tasks in an isolated process (zero HDF5/GIL lock contention)."""
    dev = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")
    ext = GPUPyramidalFlowExtractor(dev)
    mm = np.memmap(tmp_memmap_path, dtype=np.float16, mode="r+", shape=(total_frames, 2, MOTION_FEAT_DIM))
    t0 = time.monotonic()
    eps_done = 0
    frames_done = 0
    n_tasks = len(tasks_shard)

    for ti, task in enumerate(tasks_shard):
        h5_path = RAW_LIBERO / f"{task}_demo.hdf5"
        ep_list = sorted(
            task_to_eps[task],
            key=lambda idx: int(episodes[idx]["id"].rsplit("_", 1)[1]),
        )
        # Batch all episodes of one task (~11k frames) on GPU in large chunks for max throughput
        with h5py.File(h5_path, "r") as f:
            dg = f["data"]
            for ep_idx in ep_list:
                row = episodes[ep_idx]
                demo = row["id"].rsplit("/", 1)[1]
                obs = dg[demo]["obs"]
                av = np.ascontiguousarray(obs["agentview_rgb"][:])
                eh = np.ascontiguousarray(obs["eye_in_hand_rgb"][:])
                T = len(av)
                g_lo = int(frame_offsets[ep_idx])
                if T >= 2:
                    conv = row["convention"]
                    out_av = np.zeros((T, MOTION_FEAT_DIM), dtype=np.float16)
                    out_eh = np.zeros((T, MOTION_FEAT_DIM), dtype=np.float16)
                    for s in range(0, T - 1, batch_size):
                        e = min(s + batch_size, T - 1)
                        out_av[s:e] = ext.extract_pair_descriptors(av[s:e], av[s + 1 : e + 1], convention=conv)
                        out_eh[s:e] = ext.extract_pair_descriptors(eh[s:e], eh[s + 1 : e + 1], convention=conv)
                    mm[g_lo : g_lo + T, 0, :] = out_av
                    mm[g_lo : g_lo + T, 1, :] = out_eh
                eps_done += 1
                frames_done += T
        if (ti + 1) % 15 == 0 or (ti + 1) == n_tasks:
            dt = max(time.monotonic() - t0, 1e-6)
            rate = frames_done / dt
            print(
                f"[E16 MotionCache gpu:{gpu_idx}] tasks={ti+1}/{n_tasks} | "
                f"eps={eps_done:,} | frames={frames_done:,} | {rate:.0f} f/s | elapsed={dt:.0f}s",
                flush=True,
            )
    mm.flush()
    return {"eps_done": eps_done, "frames_done": frames_done}


def extract_full_motion_cache(
    *,
    e5_cache_dir: Path = E5_CACHE_DIR,
    dense_cache_dir: Path = DEFAULT_DENSE_CACHE_DIR,
    out_dir: Path = MOTION_CACHE_DIR,
    batch_size: int = 512,
    num_readers: int = 4,
    max_tasks: Optional[int] = None,
) -> Path:
    import multiprocessing as mp

    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = json.loads((e5_cache_dir / "episodes.json").read_text())
    frame_offsets = np.load(dense_cache_dir / "frame_offsets.npy", allow_pickle=False)
    total_frames = int(frame_offsets[-1])
    n_eps = len(episodes)

    task_to_eps: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(episodes):
        task_to_eps[e["task"]].append(i)
    tasks_sorted = sorted(task_to_eps.keys())
    if max_tasks is not None:
        tasks_sorted = tasks_sorted[: int(max_tasks)]

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # Use 2 parallel processes per GPU (4 total across 2 GPUs) to overlap HDF5 I/O & GPU compute
    n_procs = max(n_gpus * 2, 1) if max_tasks is None else min(2, len(tasks_sorted))
    shards = [tasks_sorted[i::n_procs] for i in range(n_procs)]

    print(
        f"[E16 MotionCache] Extracting pixel-space optical flow + grid point-tracks for "
        f"{len(tasks_sorted)} tasks ({n_eps:,} eps, {total_frames:,} frames, "
        f"{total_frames * 2 * MOTION_FEAT_DIM * 2 / 1e9:.2f} GB fp16) using {n_procs} procs on {n_gpus} GPUs...",
        flush=True,
    )

    tmp_mm_path = out_dir / "_tmp_motion_features.dat"
    mm_init = np.memmap(tmp_mm_path, dtype=np.float16, mode="w+", shape=(total_frames, 2, MOTION_FEAT_DIM))
    mm_init[:] = 0
    mm_init.flush()
    del mm_init

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        async_res = [
            pool.apply_async(
                _process_task_shard,
                (
                    i % max(n_gpus, 1),
                    shards[i],
                    episodes,
                    dict(task_to_eps),
                    frame_offsets,
                    str(tmp_mm_path),
                    total_frames,
                    batch_size,
                ),
            )
            for i in range(n_procs)
        ]
        shard_stats = [r.get() for r in async_res]

    total_eps_done = sum(s["eps_done"] for s in shard_stats)
    total_frames_done = sum(s["frames_done"] for s in shard_stats)
    if max_tasks is None and total_eps_done != n_eps:
        raise RuntimeError(f"Incomplete motion cache: {total_eps_done}/{n_eps} episodes!")

    # Load memmap into contiguous .npy & compute strictly train-only mean/std
    motion_all = np.array(
        np.memmap(tmp_mm_path, dtype=np.float16, mode="r", shape=(total_frames, 2, MOTION_FEAT_DIM))
    )
    tmp_mm_path.unlink(missing_ok=True)

    train_mask = np.zeros(total_frames, dtype=bool)
    kept_tasks = set(tasks_sorted)
    for i, e in enumerate(episodes):
        if e["fold"] == "train" and e["task"] in kept_tasks:
            train_mask[int(frame_offsets[i]) : int(frame_offsets[i + 1])] = True
    rng = np.random.default_rng(42)
    tr_idx = np.flatnonzero(train_mask)
    sub = tr_idx[rng.choice(len(tr_idx), size=min(250_000, len(tr_idx)), replace=False)]
    sub_arr = motion_all[sub].astype(np.float64)
    feat_mean = sub_arr.mean(axis=0).astype(np.float32)
    feat_std = np.maximum(sub_arr.std(axis=0), 1e-4).astype(np.float32)

    with TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        np.save(td / "motion_features.npy", motion_all)
        np.save(td / "motion_mean.npy", feat_mean)
        np.save(td / "motion_std.npy", feat_std)
        for fn in ("motion_features.npy", "motion_mean.npy", "motion_std.npy"):
            (td / fn).replace(out_dir / fn)

    elapsed = round(time.monotonic() - t0, 2)
    meta = {
        "version": "e16_pyramidal_lk_motion_v1",
        "num_episodes": int(total_eps_done),
        "total_frames": int(total_frames_done),
        "shape": list(motion_all.shape),
        "dtype": "float16",
        "storage_gb": round(motion_all.nbytes / 1e9, 4),
        "elapsed_seconds": elapsed,
        "feature_layout": {
            "0:32": "4x4_grid_tracked_flow_dx_dy",
            "32:48": "kinematics_active_global_speed_div_curl_shear_gripper_conv",
            "48:64": "8bin_hof_4quad_speed_4quad_diff",
        },
        "zero_action_supervision": True,
    }
    atomic_write_json(meta, out_dir / "motion_meta.json")
    print(f"[E16 MotionCache DONE] Saved to {out_dir} ({meta['storage_gb']:.2f} GB) in {elapsed:.1f}s", flush=True)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=MOTION_CACHE_DIR)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-readers", type=int, default=4)
    p.add_argument("--max-tasks", type=int, default=None)
    args = p.parse_args()
    extract_full_motion_cache(
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_readers=args.num_readers,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
