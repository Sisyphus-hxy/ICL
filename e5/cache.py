"""Single-pass sequential HDF5 pre-encoding and feature caching for E5.

Reads the 311 raw LIBERO-Gen Chain HDF5 archives sequentially in file/episode/frame order,
encodes the deduplicated union of:
  1. Full Demo 1Hz dual-camera RGB frames (stride 20, aligning with BPP's 20Hz / 20 steps = 1Hz)
  2. All Query dual-camera RGB frames (t-1, t) needed by the existing 80-epoch unique starts
     manifest + fixed validation manifest
using BPP's frozen CLIP ViT-B/16 weights (`agentview_rgb`, `eye_in_hand_rgb`), and saves compact
packed arrays + exact temporal index & mapping metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from . import (
    BPP_CKPT,
    CAMERAS,
    CENTER_CROP_RATIO,
    DEMO_STRIDE,
    E4_OUTPUTS,
    EMBED_DIM,
    IMAGE_SIZE,
    MANIFEST_80_DIR,
    OUTPUT_ROOT,
    RAW_LIBERO,
)
from .actions import ActionNormalizer, action_7d_to_10d
from .bpp_compat import load_bpp_pretrained_weights
from .vision import BPPVisualEncoder

CACHE_VERSION = "e5_bpp_clip_1hz_v1"
DEFAULT_CACHE_DIR = OUTPUT_ROOT / "cache"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episodes_and_manifest(
    episodes_path: Path = E4_OUTPUTS / "episodes.json",
    manifest_dir: Path = MANIFEST_80_DIR,
    verify_manifest_sha: bool = True,
) -> Tuple[List[dict], dict, List[np.ndarray], np.ndarray]:
    """Load episodes index and the existing 80-epoch + val query starts manifest."""
    episodes = json.loads(episodes_path.read_text())
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    manifest_ep_ids = json.loads((manifest_dir / manifest["episode_ids"]).read_text())
    expected_ids = [e["id"] for e in episodes]
    if manifest_ep_ids != expected_ids:
        raise ValueError("Episode ordering mismatch between episodes.json and 80-epoch manifest")

    if verify_manifest_sha:
        if sha256_file(manifest_dir / manifest["episode_ids"]) != manifest["episode_ids_sha256"]:
            raise ValueError("episodes.json checksum mismatch inside manifest dir")

    epoch_keys: List[np.ndarray] = []
    num_epochs = manifest["recipe"]["epochs"]
    for ep in range(1, num_epochs + 1):
        info = manifest["splits"][f"epoch_{ep:03d}"]
        p = manifest_dir / info["file"]
        if verify_manifest_sha and sha256_file(p) != info["sha256"]:
            raise ValueError(f"Manifest checksum mismatch at {p}")
        arr = np.asarray(json.loads(p.read_text()), dtype=np.int32)
        if arr.shape != (info["count"], 3):
            raise ValueError(f"Unexpected shape {arr.shape} at {p}")
        epoch_keys.append(arr)

    val_info = manifest["splits"]["val"]
    val_path = manifest_dir / val_info["file"]
    if verify_manifest_sha and sha256_file(val_path) != val_info["sha256"]:
        raise ValueError(f"Validation manifest checksum mismatch at {val_path}")
    val_keys = np.asarray(json.loads(val_path.read_text()), dtype=np.int32)
    if val_keys.shape != (val_info["count"], 3):
        raise ValueError(f"Unexpected val shape {val_keys.shape}")

    return episodes, manifest, epoch_keys, val_keys


def plan_deduplicated_indices(
    episodes: List[dict],
    epoch_keys: List[np.ndarray],
    val_keys: np.ndarray,
    demo_stride: int = DEMO_STRIDE,
) -> Dict[str, np.ndarray]:
    """Compute exact per-episode Demo 1Hz frame indices and deduplicated Query frame indices."""
    n_eps = len(episodes)
    ep_lengths = np.array([e["frames"] for e in episodes], dtype=np.int32)
    global_offsets = np.zeros(n_eps + 1, dtype=np.int64)
    global_offsets[1:] = np.cumsum(ep_lengths, dtype=np.int64)
    total_raw_frames = int(global_offsets[-1])

    # 1. Demo 1Hz indices: 0, 20, 40, ... for every episode (aligns with BPP PromptActionChunker)
    demo_lengths = (ep_lengths + demo_stride - 1) // demo_stride
    demo_offsets = np.zeros(n_eps + 1, dtype=np.int64)
    demo_offsets[1:] = np.cumsum(demo_lengths, dtype=np.int64)
    demo_frame_indices = np.empty(int(demo_offsets[-1]), dtype=np.int32)
    for i in range(n_eps):
        demo_frame_indices[demo_offsets[i] : demo_offsets[i + 1]] = np.arange(
            0, ep_lengths[i], demo_stride, dtype=np.int32
        )

    # 2. Deduplicated Query frame indices across all 80 train epochs + val (including t-1, t)
    need_query = np.zeros(total_raw_frames, dtype=bool)
    for keys in epoch_keys:
        q = keys[:, 1]
        f = keys[:, 2]
        need_query[global_offsets[q] + f] = True
        need_query[global_offsets[q] + np.maximum(f - 1, 0)] = True
    vq = val_keys[:, 1]
    vf = val_keys[:, 2]
    need_query[global_offsets[vq] + vf] = True
    need_query[global_offsets[vq] + np.maximum(vf - 1, 0)] = True

    query_ep_offsets = np.zeros(n_eps + 1, dtype=np.int64)
    query_frame_chunks: List[np.ndarray] = []
    for i in range(n_eps):
        local_mask = need_query[global_offsets[i] : global_offsets[i + 1]]
        frames = np.flatnonzero(local_mask).astype(np.int32)
        query_frame_chunks.append(frames)
        query_ep_offsets[i + 1] = query_ep_offsets[i] + len(frames)
    query_frame_nums = (
        np.concatenate(query_frame_chunks) if query_frame_chunks else np.empty(0, dtype=np.int32)
    )

    return {
        "ep_lengths": ep_lengths,
        "action_offsets": global_offsets,
        "demo_lengths": demo_lengths,
        "demo_offsets": demo_offsets,
        "demo_frame_indices": demo_frame_indices,
        "query_ep_offsets": query_ep_offsets,
        "query_frame_nums": query_frame_nums,
    }


def build_feature_cache(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    max_tasks: Optional[int] = None,
    batch_size: int = 512,
    precision: str = "fp16",
    num_gpus: Optional[int] = None,
    ckpt_path: Path = BPP_CKPT,
    manifest_dir: Path = MANIFEST_80_DIR,
    episodes_path: Path = E4_OUTPUTS / "episodes.json",
) -> Path:
    """Sequentially scan raw HDF5 files once and cache frozen BPP CLIP features + 10D actions."""
    precision_map = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in precision_map:
        raise ValueError(f"Unsupported precision {precision}; choose fp32, fp16, or bf16")
    autocast_dtype = precision_map[precision]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"

    episodes, manifest, epoch_keys, val_keys = load_episodes_and_manifest(
        episodes_path=episodes_path, manifest_dir=manifest_dir, verify_manifest_sha=True
    )

    # Group episodes by task HDF5 file in deterministic sorted order.
    task_to_ep_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(episodes):
        task_to_ep_indices[row["task"]].append(idx)
    all_tasks = sorted(task_to_ep_indices.keys())
    if max_tasks is not None:
        selected_tasks = all_tasks[:max_tasks]
        selected_ep_set = {i for t in selected_tasks for i in task_to_ep_indices[t]}
        # Remap episodes and filter epoch/val keys to selected subset for smoke mode.
        old_to_new = {}
        sub_episodes = []
        for old_i in sorted(selected_ep_set):
            old_to_new[old_i] = len(sub_episodes)
            sub_episodes.append(episodes[old_i])

        def _remap_keys(arr: np.ndarray) -> np.ndarray:
            mask = np.isin(arr[:, 0], list(selected_ep_set)) & np.isin(
                arr[:, 1], list(selected_ep_set)
            )
            sub = arr[mask].copy()
            for r in range(len(sub)):
                sub[r, 0] = old_to_new[int(sub[r, 0])]
                sub[r, 1] = old_to_new[int(sub[r, 1])]
            return sub

        episodes = sub_episodes
        epoch_keys = [_remap_keys(k) for k in epoch_keys]
        val_keys = _remap_keys(val_keys)
        task_to_ep_indices = defaultdict(list)
        for idx, row in enumerate(episodes):
            task_to_ep_indices[row["task"]].append(idx)
        all_tasks = sorted(task_to_ep_indices.keys())

    plan = plan_deduplicated_indices(episodes, epoch_keys, val_keys, demo_stride=DEMO_STRIDE)
    n_eps = len(episodes)
    n_demo_total = int(plan["demo_offsets"][-1])
    n_query_total = int(plan["query_ep_offsets"][-1])
    n_action_total = int(plan["action_offsets"][-1])

    # Load extracted BPP pretrained weights once on CPU.
    bundle = load_bpp_pretrained_weights(ckpt_path=ckpt_path)

    if torch.cuda.is_available():
        avail_gpus = torch.cuda.device_count()
        n_workers = min(avail_gpus, num_gpus) if num_gpus is not None else avail_gpus
        devices = [torch.device(f"cuda:{i}") for i in range(n_workers)]
    else:
        devices = [torch.device("cpu")]

    print(
        f"[E5 Cache] Tasks={len(all_tasks)}, Episodes={n_eps:,}, "
        f"Demo 1Hz steps={n_demo_total:,} ({n_demo_total * 2 * EMBED_DIM * 2 / 1e9:.2f} GB), "
        f"Unique Query frames={n_query_total:,} ({n_query_total * 2 * EMBED_DIM * 2 / 1e9:.2f} GB), "
        f"Devices={[str(d) for d in devices]}, precision={precision}->fp16",
        flush=True,
    )

    # Preallocate contiguous host arrays (total ~6.4 GB for full dataset).
    demo_features = np.empty((n_demo_total, 2, EMBED_DIM), dtype=np.float16)
    query_features = np.empty((n_query_total, 2, EMBED_DIM), dtype=np.float16)
    actions_10d = np.empty((n_action_total, 10), dtype=np.float32)

    # Create one frozen visual encoder per GPU device.
    encoders = [
        BPPVisualEncoder(pretrained_bundle=bundle, load_weights=True).to(dev)
        for dev in devices
    ]

    # Producer-consumer: 1 sequential disk reader thread -> bounded queue -> N GPU encoder threads.
    work_q: "queue.Queue[Optional[Tuple]]" = queue.Queue(maxsize=max(8, len(devices) * 4))
    errors: List[BaseException] = []
    progress_lock = threading.Lock()
    stats = {
        "tasks_done": 0,
        "episodes_done": 0,
        "frames_encoded": 0,
        "bytes_read": 0,
    }
    t0 = time.monotonic()

    def reader_thread() -> None:
        try:
            for task in all_tasks:
                if errors:
                    break
                ep_indices = task_to_ep_indices[task]
                h5_path = RAW_LIBERO / (task + "_demo.hdf5")
                # Sort by episode demo index for pure sequential disk layout inside the HDF5 file.
                ep_indices_sorted = sorted(
                    ep_indices, key=lambda i: int(episodes[i]["id"].rsplit("_", 1)[1])
                )
                with h5py.File(h5_path, "r") as f:
                    data_grp = f["data"]
                    for ep_idx in ep_indices_sorted:
                        if errors:
                            break
                        row = episodes[ep_idx]
                        demo_name = row["id"].rsplit("/", 1)[1]
                        demo_grp = data_grp[demo_name]
                        obs_grp = demo_grp["obs"]

                        d_start = int(plan["demo_offsets"][ep_idx])
                        d_end = int(plan["demo_offsets"][ep_idx + 1])
                        d_frames = plan["demo_frame_indices"][d_start:d_end]

                        q_start = int(plan["query_ep_offsets"][ep_idx])
                        q_end = int(plan["query_ep_offsets"][ep_idx + 1])
                        q_frames = plan["query_frame_nums"][q_start:q_end]

                        union_frames = np.union1d(d_frames, q_frames)
                        av_full = obs_grp["agentview_rgb"][:]
                        eh_full = obs_grp["eye_in_hand_rgb"][:]
                        act_raw = demo_grp["actions"][:].astype(np.float32, copy=False)

                        if len(av_full) != row["frames"] or len(eh_full) != row["frames"]:
                            raise ValueError(
                                f"Frame count mismatch for {row['id']}: "
                                f"obs={len(av_full)}/{len(eh_full)} vs index={row['frames']}"
                            )

                        # Slice needed union frames immediately to keep RAM queue small.
                        av_sub = np.ascontiguousarray(av_full[union_frames])
                        eh_sub = np.ascontiguousarray(eh_full[union_frames])
                        bytes_ep = av_full.nbytes + eh_full.nbytes + act_raw.nbytes
                        del av_full, eh_full

                        # Convert 7D actions -> 10D BPP format and write directly to actions_10d.
                        a_start = int(plan["action_offsets"][ep_idx])
                        a_end = int(plan["action_offsets"][ep_idx + 1])
                        actions_10d[a_start:a_end] = action_7d_to_10d(act_raw)

                        work_q.put(
                            (
                                ep_idx,
                                row["convention"],
                                union_frames,
                                d_start,
                                d_end,
                                d_frames,
                                q_start,
                                q_end,
                                q_frames,
                                av_sub,
                                eh_sub,
                                bytes_ep,
                            )
                        )
                with progress_lock:
                    stats["tasks_done"] += 1
                    if stats["tasks_done"] % 10 == 0 or stats["tasks_done"] == len(all_tasks):
                        dt = max(time.monotonic() - t0, 1e-6)
                        gb = stats["bytes_read"] / 1e9
                        print(
                            f"[E5 Cache] Tasks {stats['tasks_done']}/{len(all_tasks)} | "
                            f"Eps {stats['episodes_done']:,}/{n_eps:,} | "
                            f"Encoded {stats['frames_encoded']:,} frames ({stats['frames_encoded']/dt:.0f} f/s) | "
                            f"Read {gb:.1f} GB ({gb*1000/dt:.0f} MB/s) | Elapsed {dt:.1f}s",
                            flush=True,
                        )
        except BaseException as exc:
            errors.append(exc)
        finally:
            for _ in devices:
                work_q.put(None)

    def gpu_worker(worker_idx: int) -> None:
        enc = encoders[worker_idx]
        if enc.device.type == "cuda":
            torch.cuda.set_device(enc.device)
        try:
            while True:
                item = work_q.get()
                if item is None:
                    break
                (
                    ep_idx,
                    convention,
                    union_frames,
                    d_start,
                    d_end,
                    d_frames,
                    q_start,
                    q_end,
                    q_frames,
                    av_sub,
                    eh_sub,
                    bytes_ep,
                ) = item
                if len(union_frames) > 0:
                    feats = enc.encode_dual_camera_uint8(
                        av_sub,
                        eh_sub,
                        convention=convention,
                        batch_size=batch_size,
                        autocast_dtype=autocast_dtype,
                    )
                    feats_fp16 = feats.to(dtype=torch.float16).cpu().numpy()

                    d_pos = np.searchsorted(union_frames, d_frames)
                    demo_features[d_start:d_end] = feats_fp16[d_pos]

                    if q_end > q_start:
                        q_pos = np.searchsorted(union_frames, q_frames)
                        query_features[q_start:q_end] = feats_fp16[q_pos]

                with progress_lock:
                    stats["episodes_done"] += 1
                    stats["frames_encoded"] += len(union_frames)
                    stats["bytes_read"] += bytes_ep
        except BaseException as exc:
            errors.append(exc)

    reader = threading.Thread(target=reader_thread, name="hdf5-reader", daemon=True)
    workers = [
        threading.Thread(target=gpu_worker, args=(i,), name=f"clip-gpu-{i}", daemon=True)
        for i in range(len(devices))
    ]
    reader.start()
    for w in workers:
        w.start()
    reader.join()
    for w in workers:
        w.join()

    if errors:
        raise RuntimeError(f"Cache extraction failed with {len(errors)} error(s)") from errors[0]

    # Fit train-fold BPP action normalizer and verify against BPP checkpoint normalizer.
    train_action_slices = [
        actions_10d[int(plan["action_offsets"][i]) : int(plan["action_offsets"][i + 1])]
        for i, e in enumerate(episodes)
        if e["fold"] == "train"
    ]
    train_actions_concat = np.concatenate(train_action_slices, axis=0)
    action_norm = ActionNormalizer.fit_from_train_actions_10d(train_actions_concat)
    bpp_norm_scale = bundle["action_normalizer"]["scale"]
    bpp_norm_offset = bundle["action_normalizer"]["offset"]
    norm_scale_diff = float((action_norm.scale - bpp_norm_scale).abs().max().item())
    norm_offset_diff = float((action_norm.offset - bpp_norm_offset).abs().max().item())

    # Sanity check finite values before saving.
    if not np.isfinite(demo_features).all():
        raise RuntimeError("Non-finite values detected in demo_features")
    if not np.isfinite(query_features).all():
        raise RuntimeError("Non-finite values detected in query_features")
    if not np.isfinite(actions_10d).all():
        raise RuntimeError("Non-finite values detected in actions_10d")

    # Atomically save all arrays & metadata.
    with TemporaryDirectory(dir=cache_dir) as tmp:
        staging = Path(tmp)
        np.save(staging / "demo_features.npy", demo_features)
        np.save(staging / "demo_offsets.npy", plan["demo_offsets"])
        np.save(staging / "demo_lengths.npy", plan["demo_lengths"])
        np.save(staging / "demo_frame_indices.npy", plan["demo_frame_indices"])

        np.save(staging / "query_features.npy", query_features)
        np.save(staging / "query_ep_offsets.npy", plan["query_ep_offsets"])
        np.save(staging / "query_frame_nums.npy", plan["query_frame_nums"])

        np.save(staging / "actions_10d.npy", actions_10d)
        np.save(staging / "action_offsets.npy", plan["action_offsets"])
        torch.save(
            {
                "scale": action_norm.scale.cpu(),
                "offset": action_norm.offset.cpu(),
                "bpp_ckpt_scale": bpp_norm_scale.cpu(),
                "bpp_ckpt_offset": bpp_norm_offset.cpu(),
            },
            staging / "action_normalizer.pt",
        )
        (staging / "episodes.json").write_text(json.dumps(episodes))

        elapsed = time.monotonic() - t0
        file_sizes = {
            p.name: p.stat().st_size for p in sorted(staging.iterdir()) if p.is_file()
        }
        meta = {
            "version": CACHE_VERSION,
            "max_tasks": max_tasks,
            "num_tasks": len(all_tasks),
            "num_episodes": n_eps,
            "num_train_episodes": sum(1 for e in episodes if e["fold"] == "train"),
            "num_val_episodes": sum(1 for e in episodes if e["fold"] == "val"),
            "demo_stride": DEMO_STRIDE,
            "image_size": IMAGE_SIZE,
            "center_crop_ratio": CENTER_CROP_RATIO,
            "cameras": list(CAMERAS),
            "embed_dim": EMBED_DIM,
            "encode_precision": f"{precision}_to_fp16",
            "storage_dtype": "float16",
            "demo_total_steps": n_demo_total,
            "demo_max_len": int(plan["demo_lengths"].max()),
            "demo_min_len": int(plan["demo_lengths"].min()),
            "demo_mean_len": float(plan["demo_lengths"].mean()),
            "query_unique_frames": n_query_total,
            "union_encoded_frames": int(stats["frames_encoded"]),
            "total_action_steps": n_action_total,
            "manifest_dir": str(manifest_dir),
            "manifest_sha256": sha256_file(manifest_dir / "manifest.json"),
            "bpp_clip_sha256": bundle["clip_sha256"],
            "action_normalizer_scale": action_norm.scale.tolist(),
            "action_normalizer_offset": action_norm.offset.tolist(),
            "action_normalizer_diff_vs_bpp_ckpt": {
                "max_scale_diff": norm_scale_diff,
                "max_offset_diff": norm_offset_diff,
            },
            "elapsed_seconds": round(elapsed, 2),
            "file_sizes_bytes": file_sizes,
            "total_cache_bytes": sum(file_sizes.values()),
            "total_cache_gb": round(sum(file_sizes.values()) / 1e9, 4),
        }
        (staging / "cache_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        for p in staging.iterdir():
            p.replace(cache_dir / p.name)

    print(
        f"[E5 Cache] Complete in {elapsed:.1f}s -> {cache_dir} "
        f"(total {meta['total_cache_gb']:.3f} GB, action_norm diff vs BPP={norm_scale_diff:.2e})",
        flush=True,
    )
    return meta_path


@torch.inference_mode()
def verify_cache_consistency(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    num_episodes: int = 8,
    seed: int = 42,
    device: Optional[str] = None,
    ckpt_path: Path = BPP_CKPT,
) -> Dict[str, float]:
    """Verify online frozen CLIP features match offline cached Demo & Query features,
    plus verify temporal positions, valid lengths, and 10D<->7D action consistency."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    episodes = json.loads((cache_dir / "episodes.json").read_text())
    demo_feats = np.load(cache_dir / "demo_features.npy", mmap_mode="r")
    demo_offsets = np.load(cache_dir / "demo_offsets.npy")
    demo_lengths = np.load(cache_dir / "demo_lengths.npy")
    demo_frame_indices = np.load(cache_dir / "demo_frame_indices.npy")
    query_feats = np.load(cache_dir / "query_features.npy", mmap_mode="r")
    query_ep_offsets = np.load(cache_dir / "query_ep_offsets.npy")
    query_frame_nums = np.load(cache_dir / "query_frame_nums.npy")
    actions_10d = np.load(cache_dir / "actions_10d.npy", mmap_mode="r")
    action_offsets = np.load(cache_dir / "action_offsets.npy")

    dev = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    encoder = BPPVisualEncoder(ckpt_path=ckpt_path, load_weights=True).to(dev)
    # Always verify against pure-FP32 online reference encoding.
    autocast_dtype = None

    rng = np.random.default_rng(seed)
    n_check = min(num_episodes, len(episodes))
    chosen = rng.choice(len(episodes), size=n_check, replace=False)

    demo_max_abs = 0.0
    demo_max_rel_l2 = 0.0
    demo_min_cos = 1.0
    query_max_abs = 0.0
    query_max_rel_l2 = 0.0
    query_min_cos = 1.0
    action_max_abs = 0.0
    checked_demo_frames = 0
    checked_query_frames = 0

    for ep_idx in chosen:
        row = episodes[int(ep_idx)]
        task = row["task"]
        demo_name = row["id"].rsplit("/", 1)[1]
        h5_path = RAW_LIBERO / (task + "_demo.hdf5")

        d_s, d_e = int(demo_offsets[ep_idx]), int(demo_offsets[ep_idx + 1])
        d_len = int(demo_lengths[ep_idx])
        assert d_e - d_s == d_len == (row["frames"] + DEMO_STRIDE - 1) // DEMO_STRIDE
        d_frames = demo_frame_indices[d_s:d_e]
        assert np.array_equal(d_frames, np.arange(0, row["frames"], DEMO_STRIDE, dtype=np.int32))

        q_s, q_e = int(query_ep_offsets[ep_idx]), int(query_ep_offsets[ep_idx + 1])
        q_frames_all = query_frame_nums[q_s:q_e]

        with h5py.File(h5_path, "r") as f:
            obs = f[f"data/{demo_name}/obs"]
            av_raw = obs["agentview_rgb"][:]
            eh_raw = obs["eye_in_hand_rgb"][:]
            act_raw = f[f"data/{demo_name}/actions"][:].astype(np.float32)

        # Verify actions_10d exact match
        a_s, a_e = int(action_offsets[ep_idx]), int(action_offsets[ep_idx + 1])
        expected_10d = action_7d_to_10d(act_raw)
        action_max_abs = max(
            action_max_abs, float(np.abs(actions_10d[a_s:a_e] - expected_10d).max())
        )

        # Verify Demo 1Hz features online (always FP32 reference) vs offline cached FP16
        online_demo = (
            encoder.encode_dual_camera_uint8(
                av_raw[d_frames],
                eh_raw[d_frames],
                convention=row["convention"],
                autocast_dtype=autocast_dtype,
            )
            .cpu()
            .numpy()
        )
        cached_demo = demo_feats[d_s:d_e].astype(np.float32)
        demo_max_abs = max(demo_max_abs, float(np.abs(online_demo - cached_demo).max()))
        rel_l2 = np.linalg.norm(online_demo - cached_demo, axis=-1) / (
            np.linalg.norm(online_demo, axis=-1) + 1e-12
        )
        demo_max_rel_l2 = max(demo_max_rel_l2, float(rel_l2.max()))
        cos = (online_demo * cached_demo).sum(-1) / (
            np.linalg.norm(online_demo, axis=-1) * np.linalg.norm(cached_demo, axis=-1) + 1e-12
        )
        demo_min_cos = min(demo_min_cos, float(cos.min()))
        checked_demo_frames += len(d_frames)

        # Verify sampled Query features online vs offline
        if len(q_frames_all) > 0:
            sub_idx = rng.choice(len(q_frames_all), size=min(16, len(q_frames_all)), replace=False)
            q_f = q_frames_all[sub_idx]
            online_q = (
                encoder.encode_dual_camera_uint8(
                    av_raw[q_f],
                    eh_raw[q_f],
                    convention=row["convention"],
                    autocast_dtype=autocast_dtype,
                )
                .cpu()
                .numpy()
            )
            cached_q = query_feats[q_s + sub_idx].astype(np.float32)
            query_max_abs = max(query_max_abs, float(np.abs(online_q - cached_q).max()))
            rel_l2_q = np.linalg.norm(online_q - cached_q, axis=-1) / (
                np.linalg.norm(online_q, axis=-1) + 1e-12
            )
            query_max_rel_l2 = max(query_max_rel_l2, float(rel_l2_q.max()))
            cos_q = (online_q * cached_q).sum(-1) / (
                np.linalg.norm(online_q, axis=-1) * np.linalg.norm(cached_q, axis=-1) + 1e-12
            )
            query_min_cos = min(query_min_cos, float(cos_q.min()))
            checked_query_frames += len(sub_idx)

    report = {
        "episodes_checked": int(n_check),
        "demo_frames_checked": int(checked_demo_frames),
        "query_frames_checked": int(checked_query_frames),
        "demo_max_abs_diff": demo_max_abs,
        "demo_max_rel_l2": demo_max_rel_l2,
        "demo_min_cosine_sim": demo_min_cos,
        "query_max_abs_diff": query_max_abs,
        "query_max_rel_l2": query_max_rel_l2,
        "query_min_cosine_sim": query_min_cos,
        "action_10d_max_abs_diff": action_max_abs,
    }
    (cache_dir / "consistency_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[E5 Cache Audit]", json.dumps(report, indent=2), flush=True)
    if demo_min_cos < 0.9999 or query_min_cos < 0.9999:
        raise RuntimeError(f"Online/offline feature consistency check failed: {report}")
    if demo_max_rel_l2 > 2e-3 or query_max_rel_l2 > 2e-3:
        raise RuntimeError(f"Online/offline feature rel L2 error too large: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-episodes", type=int, default=12)
    args = parser.parse_args()

    if not args.verify_only:
        build_feature_cache(
            cache_dir=args.cache_dir,
            max_tasks=args.max_tasks,
            batch_size=args.batch_size,
            precision=args.precision,
            num_gpus=args.num_gpus,
        )
    verify_cache_consistency(
        cache_dir=args.cache_dir, num_episodes=args.verify_episodes
    )


if __name__ == "__main__":
    main()
