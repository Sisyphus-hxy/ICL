"""Build compact 11D proprioception cache (ee_pos[3], ee_ori_rot6d[6], gripper[2]) for BPP-Full."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import BPP_CKPT, E4_OUTPUTS
from .bpp_compat import _load_raw_bpp_state_dict
from .cache import DEFAULT_CACHE_DIR


def _fit_range_norm(arr: np.ndarray, range_eps: float = 1e-7):
    mn = arr.min(axis=0).astype(np.float32)
    mx = arr.max(axis=0).astype(np.float32)
    rng = mx - mn
    ign = rng < range_eps
    rng = np.where(ign, 2.0, rng)
    scale = (2.0 / rng).astype(np.float32)
    offset = np.where(ign, -mn, -1.0 - scale * mn).astype(np.float32)
    return scale, offset


def build_proprio_cache(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    tasks_root: Path = E4_OUTPUTS / "tasks",
) -> Path:
    t0 = time.monotonic()
    cache_dir = Path(cache_dir)
    episodes = json.loads((cache_dir / "episodes.json").read_text())
    action_offsets = np.load(cache_dir / "action_offsets.npy")
    total_steps = int(action_offsets[-1])

    proprio_11d = np.empty((total_steps, 11), dtype=np.float32)
    by_task = defaultdict(list)
    for idx, row in enumerate(episodes):
        by_task[row["task"]].append(idx)

    for t_i, task in enumerate(sorted(by_task.keys()), 1):
        with h5py.File(tasks_root / task / "episodes.h5", "r") as f:
            for ep_idx in by_task[task]:
                row = episodes[ep_idx]
                demo = row["id"].rsplit("/", 1)[1]
                raw = f[f"{demo}/proprio"][:].astype(np.float32, copy=False)  # (T, 8)
                a_s, a_e = int(action_offsets[ep_idx]), int(action_offsets[ep_idx + 1])
                if len(raw) != a_e - a_s:
                    raise ValueError(
                        f"Proprio length mismatch for {row['id']}: {len(raw)} vs {a_e - a_s}"
                    )
                rot6d = (
                    Rotation.from_rotvec(raw[:, 3:6])
                    .as_matrix()[:, :2, :]
                    .reshape(-1, 6)
                    .astype(np.float32)
                )
                # Store as [ee_pos(3), ee_ori_rot6d(6), gripper_states(2)]
                proprio_11d[a_s:a_e, :3] = raw[:, :3]
                proprio_11d[a_s:a_e, 3:9] = rot6d
                proprio_11d[a_s:a_e, 9:11] = raw[:, 6:8]
        if t_i % 100 == 0 or t_i == len(by_task):
            print(
                f"[E5 Proprio Cache] Tasks {t_i}/{len(by_task)} ({time.monotonic()-t0:.1f}s)",
                flush=True,
            )

    if not np.isfinite(proprio_11d).all():
        raise RuntimeError("Non-finite values in proprio_11d")

    train_slices = [
        proprio_11d[int(action_offsets[i]) : int(action_offsets[i + 1])]
        for i, e in enumerate(episodes)
        if e["fold"] == "train"
    ]
    train_p = np.concatenate(train_slices, axis=0)
    pos_scale, pos_offset = _fit_range_norm(train_p[:, :3])
    ori_scale = np.ones(6, dtype=np.float32)
    ori_offset = np.zeros(6, dtype=np.float32)
    grip_scale, grip_offset = _fit_range_norm(train_p[:, 9:11])

    norm_state = {
        "ee_pos_scale": torch.from_numpy(pos_scale),
        "ee_pos_offset": torch.from_numpy(pos_offset),
        "ee_ori_scale": torch.from_numpy(ori_scale),
        "ee_ori_offset": torch.from_numpy(ori_offset),
        "gripper_states_scale": torch.from_numpy(grip_scale),
        "gripper_states_offset": torch.from_numpy(grip_offset),
    }

    with TemporaryDirectory(dir=cache_dir) as tmp:
        staging = Path(tmp)
        np.save(staging / "proprio_11d.npy", proprio_11d)
        torch.save(norm_state, staging / "proprio_normalizer.pt")
        meta = {
            "total_steps": total_steps,
            "shape": list(proprio_11d.shape),
            "bytes": int((staging / "proprio_11d.npy").stat().st_size),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "ee_pos_scale": pos_scale.tolist(),
            "ee_pos_offset": pos_offset.tolist(),
            "gripper_states_scale": grip_scale.tolist(),
            "gripper_states_offset": grip_offset.tolist(),
        }
        (staging / "proprio_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        for p in staging.iterdir():
            p.replace(cache_dir / p.name)

    print(
        f"[E5 Proprio Cache] Done in {time.monotonic()-t0:.1f}s -> "
        f"{cache_dir / 'proprio_11d.npy'} ({meta['bytes']/1e6:.1f} MB)",
        flush=True,
    )
    return cache_dir / "proprio_meta.json"


if __name__ == "__main__":
    build_proprio_cache()
