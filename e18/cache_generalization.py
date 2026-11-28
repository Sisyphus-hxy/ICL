"""Fast Dual-GPU Feature & 3D RGB-D Motion Cache Builder for E18 Generalization & 3D Geometry.

Builds two clean, compact caches under `e18/outputs/generalization_cache/`:
  1. `libero_unseen/` — 4 completely Encoder-Unseen LIBERO suites (0 task overlap with 311 training tasks!):
       * `libero_goal_chain_selected_view` (10 tasks, same scene, unseen task compositions)
       * `libero_object`                   (10 tasks, unseen objects & layouts)
       * `libero_spatial`                  (10 tasks, unseen spatial arrangements)
       * `libero_10`                       (10 tasks, unseen rooms: Living Room, Study, Kitchen 3/4/6)
     Extracts 20 episodes/task (800 eps, ~170k frames): exact BPP CLIP (N, 2, 768) fp16,
     2D optical flow (N, 2, 64) fp16, and exact LIBERO actions_10d (N, 10) fp32.

  2. `rlbench_18/` — Cross-Dataset / Cross-Simulator RLBench PerAct (all 18 tasks!):
       * `train` (30 eps/task = 540 eps), `val` (25 eps/task = 450 eps), `test` (25 eps/task = 450 eps)
     Extracts `front_rgb` + `wrist_rgb` exact BPP CLIP (N, 2, 768) fp16, 2D optical flow (N, 2, 64) fp16,
     TRUE 3D METRIC RGB-D SCENE FLOW (N, 2, 48) fp16 computed purely from RGB + 24-bit depth + camera
     intrinsics/extrinsics (ZERO GT actions!), and 7D EEF relative pose delta + gripper targets (N, 7) fp32!
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation

from datasets.rlbench.read_episode import _RecordReader
from e5.actions import action_7d_to_10d
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder
from e9 import RAW_LIBERO
from e16.extract_video_motion import GPUPyramidalFlowExtractor

from . import GEN_CACHE_DIR, RLBENCH_ARCHIVE_ROOT


UNSEEN_LIBERO_SUITES = (
    "libero_goal_chain_selected_view",
    "libero_object",
    "libero_spatial",
    "libero_10",
)


def decode_rlbench_depth_rgb(depth_rgb: np.ndarray, near: float, far: float) -> np.ndarray:
    """Decode RLBench 24-bit RGB-encoded depth PNG `(N, H, W, 3)` uint8 -> metric depth meters `(N, H, W)` fp32."""
    d = depth_rgb.astype(np.float32)
    packed = d[..., 0] + d[..., 1] * 256.0 + d[..., 2] * 65536.0
    norm = packed / float(256 * 256 * 256 - 1)
    return (float(near) + norm * (float(far) - float(near))).astype(np.float32)


def compute_rlbench_7d_actions(gripper_pose: np.ndarray, gripper_open: np.ndarray) -> np.ndarray:
    """Convert RLBench `(T, 7)` world XYZ + quat XYZW and `(T,)` gripper_open [0,1] -> `(T, 7)` delta actions."""
    T = len(gripper_pose)
    act = np.zeros((T, 7), dtype=np.float32)
    if T >= 2:
        # Position delta scaled cleanly to match cm-scale control units (x10 -> ~[-1, 1] range)
        dpos = (gripper_pose[1:, :3] - gripper_pose[:-1, :3]) * 10.0
        q0 = gripper_pose[:-1, 3:7]
        q1 = gripper_pose[1:, 3:7]
        # Safe unit quaternion normalization
        q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True).clip(1e-8)
        q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True).clip(1e-8)
        r0 = Rotation.from_quat(q0)
        r1 = Rotation.from_quat(q1)
        drot = (r1 * r0.inv()).as_rotvec() * 2.0
        act[:-1, :3] = dpos
        act[:-1, 3:6] = drot
        act[-1, :6] = act[-2, :6]
    # Signed gripper state: +1 open, -1 closed (matches LIBERO convention)
    act[:, 6] = np.where(gripper_open.reshape(-1) > 0.5, 1.0, -1.0).astype(np.float32)
    return act


class GPU3DSceneFlowExtractor:
    """Computes BOTH 2D optical flow (64D) AND true 3D metric RGB-D scene flow (48D) per camera on GPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self.flow2d = GPUPyramidalFlowExtractor(device)
        ys = torch.arange(128, device=device, dtype=torch.float32)
        xs = torch.arange(128, device=device, dtype=torch.float32)
        py, px = torch.meshgrid(ys, xs, indexing="ij")
        self.px = px.view(1, 1, 128, 128)
        self.py = py.view(1, 1, 128, 128)

    @torch.inference_mode()
    def extract_2d_and_3d(
        self,
        rgb_u8: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute `(T, 64)` 2D flow AND `(T, 48)` 3D metric scene flow for an episode `(T, 128, 128, 3)`.

        3D Scene Flow Descriptor (48D per camera per frame):
          - [0:3]   active-weighted mean 3D velocity in WORLD frame (dXw, dYw, dZw)
          - [3:6]   global mean 3D velocity in WORLD frame
          - [6:9]   active-weighted mean 3D velocity in CAMERA frame (dXc, dYc, dZc)
          - [9:12]  3D velocity std in WORLD frame (spread/rotation/articulation)
          - [12:15] 3D speed mean, p95, active 3D motion fraction
          - [15:18] active-weighted mean 3D centroid displacement (moving object/arm centroid Δp_world)
          - [18:24] 3D world velocity percentiles (p10, p90 for Xw, Yw, Zw)
          - [24:36] 2x2 spatial-quadrant mean 3D velocity in WORLD frame (4 * 3 = 12D)
          - [36:48] 2x2 spatial-quadrant mean 3D velocity in CAMERA frame (4 * 3 = 12D)
        """
        T = len(rgb_u8)
        if T <= 1:
            return np.zeros((T, 64), dtype=np.float16), np.zeros((T, 48), dtype=np.float16)

        # Pair indices: (0->1, ..., T-2->T-1, T-1->T-1)
        rgb0 = rgb_u8
        rgb1 = np.concatenate([rgb_u8[1:], rgb_u8[-1:]], axis=0)
        d0_np = depth_m
        d1_np = np.concatenate([depth_m[1:], depth_m[-1:]], axis=0)

        # 1. Standard 64D 2D optical flow (opencv upright convention!)
        f2d = self.flow2d.extract_pair_descriptors(rgb0, rgb1, convention="opencv")  # (T, 64) fp16

        # 2. Compute dense 2D LK flow `(u, v)` on GPU to lift pixels (x, y, d0) -> (x+u, y+v, d1) into 3D!
        t0 = torch.from_numpy(rgb0).to(self.device, non_blocking=True).float() / 255.0
        t1 = torch.from_numpy(rgb1).to(self.device, non_blocking=True).float() / 255.0
        g0 = (0.299 * t0[..., 0] + 0.587 * t0[..., 1] + 0.114 * t0[..., 2]).unsqueeze(1)
        g1 = (0.299 * t1[..., 0] + 0.587 * t1[..., 1] + 0.114 * t1[..., 2]).unsqueeze(1)
        g0_s = F.conv2d(g0, self.flow2d.smooth_k, padding=2)
        g1_s = F.conv2d(g1, self.flow2d.smooth_k, padding=2)
        u_c, v_c = self.flow2d._lk_step(F.avg_pool2d(g0_s, 2), F.avg_pool2d(g1_s, 2))
        u_up = F.interpolate(u_c * 2.0, size=(128, 128), mode="bilinear", align_corners=True)
        v_up = F.interpolate(v_c * 2.0, size=(128, 128), mode="bilinear", align_corners=True)
        grid = self.flow2d.base_grid_128.expand(T, -1, -1, -1).clone()
        grid[..., 0] = grid[..., 0] + u_up[:, 0] * (2.0 / 127.0)
        grid[..., 1] = grid[..., 1] + v_up[:, 0] * (2.0 / 127.0)
        g1_warp = F.grid_sample(g1_s, grid, mode="bilinear", padding_mode="border", align_corners=True)
        du, dv = self.flow2d._lk_step(g0_s, g1_warp)
        u = (u_up + du).clamp(-10.0, 10.0)  # (T, 1, 128, 128)
        v = (v_up + dv).clamp(-10.0, 10.0)

        # Sample depth d1 at warped position (x + u, y + v)
        d0 = torch.from_numpy(d0_np).to(self.device, non_blocking=True).unsqueeze(1)  # (T, 1, 128, 128)
        d1 = torch.from_numpy(d1_np).to(self.device, non_blocking=True).unsqueeze(1)
        grid_f = self.flow2d.base_grid_128.expand(T, -1, -1, -1).clone()
        grid_f[..., 0] = (grid_f[..., 0] + u[:, 0] * (2.0 / 127.0)).clamp(-1.0, 1.0)
        grid_f[..., 1] = (grid_f[..., 1] + v[:, 0] * (2.0 / 127.0)).clamp(-1.0, 1.0)
        d1_w = F.grid_sample(d1, grid_f, mode="bilinear", padding_mode="border", align_corners=True)

        # Back-project to 3D camera coordinates using exact intrinsics K
        # Note RLBench uses CoppeliaSim intrinsics where fx, fy can be negative; use exact K_inv!
        K = torch.from_numpy(intrinsics.astype(np.float32)).to(self.device)  # (T, 3, 3) or (3, 3)
        E = torch.from_numpy(extrinsics.astype(np.float32)).to(self.device)  # (T, 4, 4) or (4, 4)
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(T, -1, -1)
        if E.ndim == 2:
            E = E.unsqueeze(0).expand(T, -1, -1)
        E1 = torch.cat([E[1:], E[-1:]], dim=0)

        fx = K[:, 0, 0].view(T, 1, 1, 1)
        fy = K[:, 1, 1].view(T, 1, 1, 1)
        cx = K[:, 0, 2].view(T, 1, 1, 1)
        cy = K[:, 1, 2].view(T, 1, 1, 1)

        # Camera-frame 3D points at t and t+1
        x0_c = (self.px - cx) * d0 / fx
        y0_c = (self.py - cy) * d0 / fy
        z0_c = d0
        p0_c = torch.cat([x0_c, y0_c, z0_c], dim=1)  # (T, 3, 128, 128)

        x1_c = (self.px + u - cx) * d1_w / fx
        y1_c = (self.py + v - cy) * d1_w / fy
        z1_c = d1_w
        p1_c = torch.cat([x1_c, y1_c, z1_c], dim=1)

        dp_c = (p1_c - p0_c).clamp(-0.25, 0.25)  # metric camera-frame 3D scene flow (m/step)

        # Transform p0_c (by E_t) and p1_c (by E_{t+1}) into WORLD frame so even moving wrist camera is world-canonical!
        R0 = E[:, :3, :3]  # (T, 3, 3)
        t0_e = E[:, :3, 3:4]  # (T, 3, 1)
        R1 = E1[:, :3, :3]
        t1_e = E1[:, :3, 3:4]

        p0_w = (torch.bmm(R0, p0_c.reshape(T, 3, -1)) + t0_e).reshape(T, 3, 128, 128)
        p1_w = (torch.bmm(R1, p1_c.reshape(T, 3, -1)) + t1_e).reshape(T, 3, 128, 128)
        dp_w = (p1_w - p0_w).clamp(-0.25, 0.25)  # metric WORLD-frame 3D scene flow!

        # Mask out depth discontinuities (|d1_w - d0| > 0.15m indicates occlusion jump, not physical motion)
        valid_3d = ((d1_w - d0).abs() < 0.15).float()
        dp_c = dp_c * valid_3d
        dp_w = dp_w * valid_3d

        spd_3d = torch.sqrt(dp_w.square().sum(dim=1, keepdim=True) + 1e-10)  # (T, 1, 128, 128)
        act_w = (spd_3d - 0.003).clamp_min(0.0) * valid_3d
        w_sum = act_w.sum(dim=(2, 3)).clamp_min(1e-6)  # (T, 1)

        act_dp_w = (dp_w * act_w).sum(dim=(2, 3)) / w_sum  # (T, 3)
        glob_dp_w = dp_w.mean(dim=(2, 3))  # (T, 3)
        act_dp_c = (dp_c * act_w).sum(dim=(2, 3)) / w_sum  # (T, 3)
        std_dp_w = dp_w.std(dim=(2, 3))  # (T, 3)

        spd_flat = spd_3d.reshape(T, -1)
        mean_spd = spd_flat.mean(dim=1, keepdim=True)
        p95_spd = torch.quantile(spd_flat, 0.95, dim=1, keepdim=True)
        act_frac = (spd_3d > 0.005).float().mean(dim=(2, 3))  # (T, 1)

        # Active moving centroid 3D position change in world frame
        cent0 = (p0_w * act_w).sum(dim=(2, 3)) / w_sum
        cent1 = (p1_w * act_w).sum(dim=(2, 3)) / w_sum
        d_cent = (cent1 - cent0).clamp(-0.25, 0.25)  # (T, 3)

        # Percentiles of world 3D flow
        dp_w_flat = dp_w.reshape(T, 3, -1)
        q10 = torch.quantile(dp_w_flat, 0.10, dim=-1)  # (T, 3)
        q90 = torch.quantile(dp_w_flat, 0.90, dim=-1)  # (T, 3)

        # 2x2 spatial quadrant 3D velocities
        quad_w = F.adaptive_avg_pool2d(dp_w, (2, 2)).reshape(T, 12)
        quad_c = F.adaptive_avg_pool2d(dp_c, (2, 2)).reshape(T, 12)

        desc3d = torch.cat(
            [act_dp_w, glob_dp_w, act_dp_c, std_dp_w, mean_spd, p95_spd, act_frac, d_cent, q10, q90, quad_w, quad_c],
            dim=-1,
        )  # (T, 48)
        # Scale up meters -> decimeters so fp16 precision is crisp
        desc3d = (desc3d * 10.0).clamp(-30.0, 30.0)
        return f2d, desc3d.to(torch.float16).cpu().numpy()


def build_libero_unseen_cache(
    *,
    out_dir: Path = GEN_CACHE_DIR / "libero_unseen",
    eps_per_task: int = 20,
    batch_size: int = 512,
) -> Path:
    """Cache CLIP + 2D flow + 10D actions for all 4 completely Encoder-Unseen LIBERO suites."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "frame_features.npy").is_file() and (out_dir / "meta.json").is_file():
        print(f"[E18 Cache] Unseen LIBERO cache already exists at {out_dir}", flush=True)
        return out_dir

    t0 = time.monotonic()
    vis = BPPVisualEncoder().to(device).eval()
    flow_ext = GPUPyramidalFlowExtractor(device)

    episodes_meta: List[Dict[str, Any]] = []
    feats_list, mot_list, act10_list, act7_list = [], [], [], []
    offset = 0

    for suite in UNSEEN_LIBERO_SUITES:
        sdir = RAW_LIBERO / suite
        h5_files = sorted(sdir.glob("*.hdf5"))
        for h5_path in h5_files:
            task_short = h5_path.stem.replace("_demo", "")
            task_full = f"{suite}/{task_short}"
            with h5py.File(h5_path, "r") as f:
                dg = f["data"]
                demo_keys = sorted(dg.keys(), key=lambda s: int(s.split("_")[1]))[:eps_per_task]
                for dk in demo_keys:
                    ep = dg[dk]
                    av = ep["obs"]["agentview_rgb"][:]
                    eh = ep["obs"]["eye_in_hand_rgb"][:]
                    a7 = ep["actions"][:].astype(np.float32)
                    a10 = action_7d_to_10d(a7)
                    T = len(a7)

                    # CLIP features
                    cf = vis.encode_dual_camera_uint8(
                        av, eh, convention="opengl", batch_size=batch_size, autocast_dtype=torch.float16
                    ).to(torch.float16).cpu().numpy()

                    # 2D optical flow
                    av1 = np.concatenate([av[1:], av[-1:]], axis=0)
                    eh1 = np.concatenate([eh[1:], eh[-1:]], axis=0)
                    mf = np.empty((T, 2, 64), dtype=np.float16)
                    for s in range(0, T, batch_size):
                        e = min(s + batch_size, T)
                        mf[s:e, 0] = flow_ext.extract_pair_descriptors(av[s:e], av1[s:e], convention="opengl")
                        mf[s:e, 1] = flow_ext.extract_pair_descriptors(eh[s:e], eh1[s:e], convention="opengl")

                    feats_list.append(cf)
                    mot_list.append(mf)
                    act10_list.append(a10)
                    act7_list.append(a7)
                    episodes_meta.append({
                        "id": f"{task_full}/{dk}",
                        "suite": suite,
                        "task": task_full,
                        "frames": int(T),
                        "offset": int(offset),
                    })
                    offset += T
        print(f"[E18 Cache] LIBERO unseen suite {suite} done ({len(episodes_meta)} eps, {offset:,} frames, {time.monotonic()-t0:.1f}s)", flush=True)

    feats = np.concatenate(feats_list, axis=0)
    mots = np.concatenate(mot_list, axis=0)
    a10s = np.concatenate(act10_list, axis=0)
    a7s = np.concatenate(act7_list, axis=0)
    ep_lengths = np.array([e["frames"] for e in episodes_meta], dtype=np.int32)
    frame_offsets = np.zeros(len(episodes_meta) + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(ep_lengths)

    np.save(out_dir / "frame_features.npy", feats)
    np.save(out_dir / "motion_features.npy", mots)
    np.save(out_dir / "actions_10d.npy", a10s)
    np.save(out_dir / "actions_7d.npy", a7s)
    np.save(out_dir / "ep_lengths.npy", ep_lengths)
    np.save(out_dir / "frame_offsets.npy", frame_offsets)
    atomic_write_json(episodes_meta, out_dir / "episodes.json")
    atomic_write_json(
        {
            "suites": list(UNSEEN_LIBERO_SUITES),
            "num_episodes": len(episodes_meta),
            "total_frames": int(offset),
            "elapsed_sec": round(time.monotonic() - t0, 1),
        },
        out_dir / "meta.json",
    )
    print(f"[E18 Cache] Saved LIBERO unseen cache: {len(episodes_meta)} eps, {offset:,} frames in {time.monotonic()-t0:.1f}s", flush=True)
    return out_dir


def _load_rlbench_task_zip(args_tuple: Tuple[Path, str, str, int]) -> List[Dict[str, Any]]:
    """Worker process: opens a task ZIP ONCE and decodes `n_eps` episodes (zero GIL contention!)."""
    zip_path, split, task, n_eps = args_tuple
    out_eps: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for ep_idx in range(n_eps):
            prefix = f"{task}/all_variations/episodes/episode{ep_idx}/"
            with zf.open(prefix + "low_dim_obs.pkl") as stream:
                obs_list = _RecordReader(stream).load()._observations
            T = len(obs_list)

            front_rgb = np.empty((T, 128, 128, 3), dtype=np.uint8)
            wrist_rgb = np.empty((T, 128, 128, 3), dtype=np.uint8)
            front_dep_raw = np.empty((T, 128, 128, 3), dtype=np.uint8)
            wrist_dep_raw = np.empty((T, 128, 128, 3), dtype=np.uint8)

            f_intr = np.empty((T, 3, 3), dtype=np.float32)
            f_extr = np.empty((T, 4, 4), dtype=np.float32)
            w_intr = np.empty((T, 3, 3), dtype=np.float32)
            w_extr = np.empty((T, 4, 4), dtype=np.float32)

            for t in range(T):
                with zf.open(f"{prefix}front_rgb/{t}.png") as s:
                    front_rgb[t] = np.asarray(Image.open(s), dtype=np.uint8)
                with zf.open(f"{prefix}wrist_rgb/{t}.png") as s:
                    wrist_rgb[t] = np.asarray(Image.open(s), dtype=np.uint8)
                with zf.open(f"{prefix}front_depth/{t}.png") as s:
                    front_dep_raw[t] = np.asarray(Image.open(s), dtype=np.uint8)
                with zf.open(f"{prefix}wrist_depth/{t}.png") as s:
                    wrist_dep_raw[t] = np.asarray(Image.open(s), dtype=np.uint8)
                m = obs_list[t].misc
                f_intr[t] = m["front_camera_intrinsics"]
                f_extr[t] = m["front_camera_extrinsics"]
                w_intr[t] = m["wrist_camera_intrinsics"]
                w_extr[t] = m["wrist_camera_extrinsics"]

            m0 = obs_list[0].misc
            front_dep = decode_rlbench_depth_rgb(front_dep_raw, m0["front_camera_near"], m0["front_camera_far"])
            wrist_dep = decode_rlbench_depth_rgb(wrist_dep_raw, m0["wrist_camera_near"], m0["wrist_camera_far"])

            g_pose = np.stack([o.gripper_pose for o in obs_list]).astype(np.float32)
            g_open = np.array([float(o.gripper_open) for o in obs_list], dtype=np.float32)
            a7 = compute_rlbench_7d_actions(g_pose, g_open)

            out_eps.append({
                "split": split,
                "task": task,
                "ep_idx": ep_idx,
                "frames": T,
                "front_rgb": front_rgb,
                "wrist_rgb": wrist_rgb,
                "front_dep": front_dep,
                "wrist_dep": wrist_dep,
                "f_intr": f_intr,
                "f_extr": f_extr,
                "w_intr": w_intr,
                "w_extr": w_extr,
                "actions_7d": a7,
                "gripper_pose": g_pose,
                "gripper_open": g_open,
            })
    return out_eps


def build_rlbench_cache(
    *,
    out_dir: Path = GEN_CACHE_DIR / "rlbench_18",
    train_eps_per_task: int = 15,
    val_eps_per_task: int = 15,
    test_eps_per_task: int = 15,
    batch_size: int = 512,
    num_workers: int = 9,
) -> Path:
    """Cache RLBench-18 CLIP + 2D flow + TRUE 3D metric RGB-D scene flow + 7D EEF actions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "frame_features.npy").is_file() and (out_dir / "flow3d_features.npy").is_file() and (out_dir / "meta.json").is_file():
        print(f"[E18 Cache] RLBench-18 cache already exists at {out_dir}", flush=True)
        return out_dir

    t0 = time.monotonic()
    vis = BPPVisualEncoder().to(device).eval()
    flow3d_ext = GPU3DSceneFlowExtractor(device)

    tasks = sorted([p.stem for p in (RLBENCH_ARCHIVE_ROOT / "val").glob("*.zip")])
    split_counts = [("train", train_eps_per_task), ("val", val_eps_per_task), ("test", test_eps_per_task)]

    episodes_meta: List[Dict[str, Any]] = []
    feats_l, mot2d_l, mot3d_l, a7_l, gpose_l = [], [], [], [], []
    offset = 0

    for split, n_eps in split_counts:
        task_jobs = [(RLBENCH_ARCHIVE_ROOT / split / f"{task}.zip", split, task, n_eps) for task in tasks]
        # Process 6 tasks at a time in parallel worker processes to bound RAM while maximizing CPU throughput
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for ti, ep_list in enumerate(pool.map(_load_rlbench_task_zip, task_jobs)):
                for item in ep_list:
                    T = item["frames"]
                    cf = vis.encode_dual_camera_uint8(
                        item["front_rgb"],
                        item["wrist_rgb"],
                        convention="opencv",
                        batch_size=batch_size,
                        autocast_dtype=torch.float16,
                    ).to(torch.float16).cpu().numpy()

                    f2d_0, f3d_0 = flow3d_ext.extract_2d_and_3d(
                        item["front_rgb"], item["front_dep"], item["f_intr"], item["f_extr"]
                    )
                    f2d_1, f3d_1 = flow3d_ext.extract_2d_and_3d(
                        item["wrist_rgb"], item["wrist_dep"], item["w_intr"], item["w_extr"]
                    )
                    m2d = np.stack([f2d_0, f2d_1], axis=1)  # (T, 2, 64)
                    m3d = np.stack([f3d_0, f3d_1], axis=1)  # (T, 2, 48)

                    feats_l.append(cf)
                    mot2d_l.append(m2d)
                    mot3d_l.append(m3d)
                    a7_l.append(item["actions_7d"])
                    gpose_l.append(item["gripper_pose"])
                    episodes_meta.append({
                        "id": f"rlbench/{item['split']}/{item['task']}/ep_{item['ep_idx']}",
                        "split": item["split"],
                        "task": item["task"],
                        "frames": int(T),
                        "offset": int(offset),
                    })
                    offset += T
                if (ti + 1) % 6 == 0 or (ti + 1) == len(tasks):
                    print(
                        f"[E18 Cache] RLBench split={split} {ti+1}/{len(tasks)} tasks "
                        f"({len(episodes_meta)} eps, {offset:,} frames, {time.monotonic()-t0:.1f}s)",
                        flush=True,
                    )

    feats = np.concatenate(feats_l, axis=0)
    m2d_all = np.concatenate(mot2d_l, axis=0)
    m3d_all = np.concatenate(mot3d_l, axis=0)
    a7_all = np.concatenate(a7_l, axis=0)
    gp_all = np.concatenate(gpose_l, axis=0)
    ep_lengths = np.array([e["frames"] for e in episodes_meta], dtype=np.int32)
    frame_offsets = np.zeros(len(episodes_meta) + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(ep_lengths)

    # Save standardization stats computed strictly on train split
    tr_end = int(frame_offsets[sum(1 for e in episodes_meta if e["split"] == "train")])
    m2_tr = m2d_all[:tr_end].astype(np.float32)
    m3_tr = m3d_all[:tr_end].astype(np.float32)

    np.save(out_dir / "frame_features.npy", feats)
    np.save(out_dir / "motion_features.npy", m2d_all)
    np.save(out_dir / "flow3d_features.npy", m3d_all)
    np.save(out_dir / "motion_mean.npy", m2_tr.mean(axis=0))
    np.save(out_dir / "motion_std.npy", m2_tr.std(axis=0).clip(1e-4))
    np.save(out_dir / "flow3d_mean.npy", m3_tr.mean(axis=0))
    np.save(out_dir / "flow3d_std.npy", m3_tr.std(axis=0).clip(1e-4))
    np.save(out_dir / "actions_7d.npy", a7_all)
    np.save(out_dir / "gripper_pose.npy", gp_all)
    np.save(out_dir / "ep_lengths.npy", ep_lengths)
    np.save(out_dir / "frame_offsets.npy", frame_offsets)
    atomic_write_json(episodes_meta, out_dir / "episodes.json")
    atomic_write_json(
        {
            "tasks": tasks,
            "num_tasks": len(tasks),
            "num_episodes": len(episodes_meta),
            "total_frames": int(offset),
            "elapsed_sec": round(time.monotonic() - t0, 1),
        },
        out_dir / "meta.json",
    )
    print(f"[E18 Cache] Saved RLBench-18 cache: {len(episodes_meta)} eps, {offset:,} frames in {time.monotonic()-t0:.1f}s", flush=True)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libero-unseen", action="store_true")
    p.add_argument("--rlbench", action="store_true")
    p.add_argument("--eps-per-task", type=int, default=20)
    p.add_argument("--rlb-train-eps", type=int, default=35)
    args = p.parse_args()
    if args.libero_unseen:
        build_libero_unseen_cache(eps_per_task=args.eps_per_task)
    if args.rlbench:
        build_rlbench_cache(train_eps_per_task=args.rlb_train_eps)


if __name__ == "__main__":
    main()
