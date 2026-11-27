"""Build a complete dense (20Hz, stride=1) dual-camera frozen CLIP feature cache for E9 Stage 1.

Temporal Sampling Density Audit of existing caches (15,550 episodes, 3,629,339 total frames):
  - `e5/outputs/cache/demo_features.npy`: 1Hz (stride=20), 188,840 frames (0-1 frame per 16-step clip).
  - `e5/outputs/cache/query_features.npy`: deduplicated (t-1, t) manifest frames, 1,825,431 frames
    (train: 55.5% of frames / 42.1% full 16-frame windows; val: 3.35% of frames / 0.8% full 16-frame windows).
  - `e7/outputs/behavior_memory_cache/mem_h.npy`: stride-16 anchor frames, adds 198,974 unique frames.
  - Combined reusable frames: 2,117,809 / 3,629,339 (58.35%), verified bit-exact (max_abs_diff = 0.0).

This builder reuses ALL 2,117,809 cached dual-camera CLIP features bit-identically and encodes ONLY
the remaining 1,511,530 missing frames from `bpp/datasets/libero/*.hdf5` using the exact same frozen
`BPPVisualEncoder` (fp16 -> fp16), producing a single contiguous `(3629339, 2, 768)` float16 array
(~11.15 GB, GPU-resident on RTX A6000) so every 16-frame local window `V_{t:t+16}` on both train and
val has 100% exact 20Hz frame coverage without train/val sampling density mismatch.
"""

from __future__ import annotations

import argparse
import json
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

from e5.bpp_compat import load_bpp_pretrained_weights
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder

from . import (
    BPP_CKPT,
    CAMERAS,
    DEFAULT_BEHAVIOR_MEM_DIR,
    DEFAULT_DENSE_CACHE_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    NUM_CAMERAS,
    RAW_LIBERO,
)

DENSE_CACHE_VERSION = "e9_bpp_clip_dense_20hz_v1"


def audit_existing_cache_density(
    e5_cache_dir: Path = E5_CACHE_DIR,
    e7_mem_dir: Optional[Path] = DEFAULT_BEHAVIOR_MEM_DIR,
) -> Dict[str, Any]:
    """Compute exact temporal coverage statistics of existing E5 (+ optional E7) CLIP caches."""
    e5_cache_dir = Path(e5_cache_dir)
    episodes = json.loads((e5_cache_dir / "episodes.json").read_text())
    q_off = np.load(e5_cache_dir / "query_ep_offsets.npy", allow_pickle=False)
    q_f = np.load(e5_cache_dir / "query_frame_nums.npy", allow_pickle=False)
    d_off = np.load(e5_cache_dir / "demo_offsets.npy", allow_pickle=False)
    d_len = np.load(e5_cache_dir / "demo_lengths.npy", allow_pickle=False)
    d_f = np.load(e5_cache_dir / "demo_frame_indices.npy", allow_pickle=False)

    use_e7 = (
        e7_mem_dir is not None
        and (Path(e7_mem_dir) / "mem_h.npy").is_file()
        and (Path(e7_mem_dir) / "mem_offsets.npy").is_file()
        and (Path(e7_mem_dir) / "mem_anchor_steps.npy").is_file()
    )
    if use_e7:
        m_off = np.load(Path(e7_mem_dir) / "mem_offsets.npy", allow_pickle=False)
        m_anc = np.load(Path(e7_mem_dir) / "mem_anchor_steps.npy", allow_pickle=False)

    total_frames = {"train": 0, "val": 0}
    e5_only_frames = {"train": 0, "val": 0}
    reusable_frames = {"train": 0, "val": 0}
    missing_frames = {"train": 0, "val": 0}

    for ep_idx, ep in enumerate(episodes):
        fold = ep["fold"]
        T = int(ep["frames"])
        total_frames[fold] += T
        present = np.zeros(T, dtype=bool)
        qf = q_f[int(q_off[ep_idx]) : int(q_off[ep_idx + 1])]
        df = d_f[int(d_off[ep_idx]) : int(d_off[ep_idx]) + int(d_len[ep_idx])]
        present[qf] = True
        present[df] = True
        e5_only_frames[fold] += int(present.sum())
        if use_e7:
            ancs = m_anc[int(m_off[ep_idx]) : int(m_off[ep_idx + 1])]
            present[ancs] = True
            present[np.maximum(ancs - 1, 0)] = True
        n_have = int(present.sum())
        reusable_frames[fold] += n_have
        missing_frames[fold] += T - n_have

    tot_all = total_frames["train"] + total_frames["val"]
    reus_all = reusable_frames["train"] + reusable_frames["val"]
    miss_all = missing_frames["train"] + missing_frames["val"]
    return {
        "num_episodes": len(episodes),
        "total_frames": tot_all,
        "total_frames_by_fold": total_frames,
        "e5_only_cached_frames_by_fold": e5_only_frames,
        "reusable_frames_total": reus_all,
        "reusable_frames_by_fold": reusable_frames,
        "reusable_fraction": reus_all / max(tot_all, 1),
        "missing_frames_total": miss_all,
        "missing_frames_by_fold": missing_frames,
        "used_e7_behavior_memory_cache": use_e7,
    }


@torch.inference_mode()
def build_dense_clip_cache(
    *,
    e5_cache_dir: Path = E5_CACHE_DIR,
    e7_mem_dir: Optional[Path] = DEFAULT_BEHAVIOR_MEM_DIR,
    out_dir: Path = DEFAULT_DENSE_CACHE_DIR,
    batch_size: int = 512,
    num_gpus: Optional[int] = None,
    audit_episodes: int = 16,
    seed: int = 42,
) -> Path:
    """Build dense 20Hz dual-camera CLIP feature cache reusing E5 + E7 cached frames."""
    t0 = time.monotonic()
    e5_cache_dir = Path(e5_cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e5_meta = json.loads((e5_cache_dir / "cache_meta.json").read_text())
    episodes = json.loads((e5_cache_dir / "episodes.json").read_text())
    n_eps = len(episodes)

    ep_lengths = np.array([int(e["frames"]) for e in episodes], dtype=np.int32)
    frame_offsets = np.zeros(n_eps + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(ep_lengths, dtype=np.int64)
    total_frames = int(frame_offsets[-1])

    q_off = np.load(e5_cache_dir / "query_ep_offsets.npy", allow_pickle=False)
    q_f = np.load(e5_cache_dir / "query_frame_nums.npy", allow_pickle=False)
    q_feats = np.load(e5_cache_dir / "query_features.npy", mmap_mode="r")

    d_off = np.load(e5_cache_dir / "demo_offsets.npy", allow_pickle=False)
    d_len = np.load(e5_cache_dir / "demo_lengths.npy", allow_pickle=False)
    d_f = np.load(e5_cache_dir / "demo_frame_indices.npy", allow_pickle=False)
    d_feats = np.load(e5_cache_dir / "demo_features.npy", mmap_mode="r")

    use_e7 = (
        e7_mem_dir is not None
        and (Path(e7_mem_dir) / "mem_h.npy").is_file()
        and (Path(e7_mem_dir) / "mem_offsets.npy").is_file()
        and (Path(e7_mem_dir) / "mem_anchor_steps.npy").is_file()
    )
    if use_e7:
        m_off = np.load(Path(e7_mem_dir) / "mem_offsets.npy", allow_pickle=False)
        m_anc = np.load(Path(e7_mem_dir) / "mem_anchor_steps.npy", allow_pickle=False)
        m_h = np.load(Path(e7_mem_dir) / "mem_h.npy", mmap_mode="r")

    # Allocate contiguous dense feature array on host RAM (~11.15 GB for 3.63M frames)
    all_features = np.empty((total_frames, NUM_CAMERAS, EMBED_DIM), dtype=np.float16)
    filled = np.zeros(total_frames, dtype=bool)

    reused_query = 0
    reused_demo = 0
    reused_e7 = 0
    missing_by_ep: Dict[int, np.ndarray] = {}

    t_copy = time.monotonic()
    for ep_idx in range(n_eps):
        g_lo = int(frame_offsets[ep_idx])
        g_hi = int(frame_offsets[ep_idx + 1])
        T = g_hi - g_lo

        # 1. Copy E5 query_features
        ql, qh = int(q_off[ep_idx]), int(q_off[ep_idx + 1])
        if qh > ql:
            qf = q_f[ql:qh]
            all_features[g_lo + qf] = q_feats[ql:qh]
            filled[g_lo + qf] = True
            reused_query += len(qf)

        # 2. Copy E5 demo_features (only unfilled frames)
        dl = int(d_off[ep_idx])
        dh = dl + int(d_len[ep_idx])
        if dh > dl:
            df = d_f[dl:dh]
            need_d = ~filled[g_lo + df]
            if np.any(need_d):
                sel_df = df[need_d]
                all_features[g_lo + sel_df] = d_feats[dl:dh][need_d]
                filled[g_lo + sel_df] = True
                reused_demo += int(need_d.sum())

        # 3. Copy E7 mem_h frames (prev_anchor = tokens 0..1, anchor = tokens 2..3)
        if use_e7:
            ml, mh = int(m_off[ep_idx]), int(m_off[ep_idx + 1])
            if mh > ml:
                ancs = m_anc[ml:mh]
                prevs = np.maximum(ancs - 1, 0)
                need_p = ~filled[g_lo + prevs]
                need_c = ~filled[g_lo + ancs]
                if np.any(need_p) or np.any(need_c):
                    mh_ep = np.asarray(m_h[ml:mh])  # (M, 4, 768)
                    if np.any(need_p):
                        sel_p = prevs[need_p]
                        all_features[g_lo + sel_p] = mh_ep[need_p, 0:2, :]
                        filled[g_lo + sel_p] = True
                        reused_e7 += int(need_p.sum())
                    # Re-check need_c in case anchor 0 == prev 0
                    need_c = ~filled[g_lo + ancs]
                    if np.any(need_c):
                        sel_c = ancs[need_c]
                        all_features[g_lo + sel_c] = mh_ep[need_c, 2:4, :]
                        filled[g_lo + sel_c] = True
                        reused_e7 += int(need_c.sum())

        ep_missing = np.flatnonzero(~filled[g_lo:g_hi]).astype(np.int32)
        if len(ep_missing) > 0:
            missing_by_ep[ep_idx] = ep_missing

    total_reused = int(filled.sum())
    total_missing = total_frames - total_reused
    copy_sec = time.monotonic() - t_copy

    if torch.cuda.is_available():
        avail_gpus = torch.cuda.device_count()
        n_workers = min(avail_gpus, num_gpus) if num_gpus is not None else avail_gpus
        devices = [torch.device(f"cuda:{i}") for i in range(n_workers)]
    else:
        devices = [torch.device("cpu")]

    print(
        f"[E9 DenseCache] Episodes={n_eps:,} | Total frames={total_frames:,} "
        f"({total_frames * NUM_CAMERAS * EMBED_DIM * 2 / 1e9:.2f} GB fp16) | "
        f"Reused={total_reused:,} ({100.0 * total_reused / total_frames:.2f}%: "
        f"E5_query={reused_query:,}, E5_demo={reused_demo:,}, E7_mem={reused_e7:,}) in {copy_sec:.1f}s | "
        f"Missing to encode={total_missing:,} across {len(missing_by_ep):,} eps | "
        f"Devices={[str(d) for d in devices]}",
        flush=True,
    )

    bundle = load_bpp_pretrained_weights(ckpt_path=BPP_CKPT)
    encoders = [
        BPPVisualEncoder(pretrained_bundle=bundle, load_weights=True).to(dev).eval()
        for dev in devices
    ]
    autocast_dtype = torch.float16 if devices[0].type == "cuda" else None

    if total_missing > 0:
        t_enc = time.monotonic()
        task_to_eps: Dict[str, List[int]] = defaultdict(list)
        for ep_idx in missing_by_ep:
            task_to_eps[episodes[ep_idx]["task"]].append(ep_idx)
        tasks_sorted = sorted(task_to_eps.keys())

        # Bounded work queue carrying multi-episode batches (~batch_size frames) to keep GPUs saturated
        work_q: "queue.Queue[Optional[List[Tuple[int, str, np.ndarray, np.ndarray, np.ndarray]]]]" = (
            queue.Queue(maxsize=max(12, len(devices) * 6))
        )
        errors: List[BaseException] = []
        prog_lock = threading.Lock()
        stats = {"tasks_done": 0, "frames_encoded": 0}

        def _reader_thread() -> None:
            pending: List[Tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
            pending_frames = 0
            try:
                for task in tasks_sorted:
                    if errors:
                        break
                    h5_path = RAW_LIBERO / f"{task}_demo.hdf5"
                    ep_list = sorted(
                        task_to_eps[task],
                        key=lambda i: int(episodes[i]["id"].rsplit("_", 1)[1]),
                    )
                    with h5py.File(h5_path, "r") as f:
                        data_grp = f["data"]
                        for ep_idx in ep_list:
                            if errors:
                                break
                            row = episodes[ep_idx]
                            demo_name = row["id"].rsplit("/", 1)[1]
                            obs_grp = data_grp[demo_name]["obs"]
                            f_idx = missing_by_ep[ep_idx]
                            # When >50% of frames are missing (e.g. val episodes), sequential
                            # full-episode read + numpy slice is faster than HDF5 fancy indexing.
                            T_ep = int(row["frames"])
                            if len(f_idx) * 2 >= T_ep:
                                av_sub = np.ascontiguousarray(obs_grp["agentview_rgb"][:][f_idx])
                                eh_sub = np.ascontiguousarray(obs_grp["eye_in_hand_rgb"][:][f_idx])
                            else:
                                av_sub = np.ascontiguousarray(obs_grp["agentview_rgb"][f_idx])
                                eh_sub = np.ascontiguousarray(obs_grp["eye_in_hand_rgb"][f_idx])
                            pending.append((ep_idx, row["convention"], f_idx, av_sub, eh_sub))
                            pending_frames += len(f_idx)
                            if pending_frames >= batch_size:
                                work_q.put(pending)
                                pending = []
                                pending_frames = 0
                    with prog_lock:
                        stats["tasks_done"] += 1
                if pending and not errors:
                    work_q.put(pending)
            except BaseException as exc:
                errors.append(exc)
            finally:
                for _ in devices:
                    work_q.put(None)

        def _gpu_worker(worker_id: int) -> None:
            enc = encoders[worker_id]
            try:
                while True:
                    item = work_q.get()
                    if item is None:
                        break
                    if errors:
                        continue
                    # Group by camera convention inside the batch (typically all 'opengl')
                    by_conv: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = (
                        defaultdict(list)
                    )
                    for ep_idx, conv, f_idx, av_sub, eh_sub in item:
                        by_conv[conv].append((ep_idx, f_idx, av_sub, eh_sub))

                    n_done = 0
                    for conv, group in by_conv.items():
                        av_cat = np.concatenate([g[2] for g in group], axis=0)
                        eh_cat = np.concatenate([g[3] for g in group], axis=0)
                        feats = (
                            enc.encode_dual_camera_uint8(
                                av_cat,
                                eh_cat,
                                convention=conv,
                                batch_size=batch_size,
                                autocast_dtype=autocast_dtype,
                            )
                            .to(dtype=torch.float16)
                            .cpu()
                            .numpy()
                        )
                        cursor = 0
                        for ep_idx, f_idx, av_sub, _ in group:
                            cnt = len(f_idx)
                            g_lo = int(frame_offsets[ep_idx])
                            all_features[g_lo + f_idx] = feats[cursor : cursor + cnt]
                            filled[g_lo + f_idx] = True
                            cursor += cnt
                            n_done += cnt

                    with prog_lock:
                        prev_b = stats["frames_encoded"] // 100_000
                        stats["frames_encoded"] += n_done
                        cur_b = stats["frames_encoded"] // 100_000
                        if cur_b > prev_b or stats["frames_encoded"] >= total_missing:
                            dt = max(time.monotonic() - t_enc, 1e-6)
                            rate = stats["frames_encoded"] / dt
                            rem = max(total_missing - stats["frames_encoded"], 0) / max(rate, 1e-6)
                            print(
                                f"[E9 DenseCache] Encoded {stats['frames_encoded']:,}/{total_missing:,} "
                                f"({100.0 * stats['frames_encoded'] / total_missing:.1f}%) | "
                                f"tasks={stats['tasks_done']}/{len(tasks_sorted)} | "
                                f"{rate:.0f} f/s | elapsed={dt:.0f}s | eta={rem:.0f}s",
                                flush=True,
                            )
            except BaseException as exc:
                errors.append(exc)

        rt = threading.Thread(target=_reader_thread, daemon=True)
        wts = [threading.Thread(target=_gpu_worker, args=(i,), daemon=True) for i in range(len(devices))]
        rt.start()
        for wt in wts:
            wt.start()
        rt.join()
        for wt in wts:
            wt.join()

        if errors:
            raise RuntimeError(f"Dense CLIP cache build failed: {errors[0]}") from errors[0]

    if not bool(filled.all()):
        n_unfilled = int((~filled).sum())
        raise RuntimeError(f"Dense CLIP cache incomplete: {n_unfilled} unfilled frames remain!")

    # Consistency audit against fresh raw HDF5 CLIP encoding across train + val episodes
    rng = np.random.default_rng(seed + 909)
    train_ids = [i for i, e in enumerate(episodes) if e["fold"] == "train"]
    val_ids = [i for i, e in enumerate(episodes) if e["fold"] == "val"]
    n_tr = min(audit_episodes // 2, len(train_ids))
    n_va = min(audit_episodes - n_tr, len(val_ids))
    audit_ids = sorted(
        rng.choice(train_ids, size=n_tr, replace=False).tolist()
        + rng.choice(val_ids, size=n_va, replace=False).tolist()
    )
    enc0 = encoders[0]
    max_abs_diff = 0.0
    min_cos = 1.0
    frames_audited = 0
    for ep_idx in audit_ids:
        row = episodes[ep_idx]
        h5_path = RAW_LIBERO / f"{row['task']}_demo.hdf5"
        demo_name = row["id"].rsplit("/", 1)[1]
        T_ep = int(row["frames"])
        sample_f = np.unique(np.linspace(0, T_ep - 1, min(16, T_ep), dtype=np.int32))
        with h5py.File(h5_path, "r") as f:
            obs = f["data"][demo_name]["obs"]
            av = np.ascontiguousarray(obs["agentview_rgb"][sample_f])
            eh = np.ascontiguousarray(obs["eye_in_hand_rgb"][sample_f])
        ref = (
            enc0.encode_dual_camera_uint8(
                av, eh, convention=row["convention"], batch_size=64, autocast_dtype=autocast_dtype
            )
            .float()
            .cpu()
            .numpy()
        )
        g_lo = int(frame_offsets[ep_idx])
        cached = all_features[g_lo + sample_f].astype(np.float32)
        diff = np.max(np.abs(cached - ref))
        max_abs_diff = max(max_abs_diff, float(diff))
        # Per-(frame, cam) cosine similarity
        c_flat = cached.reshape(-1, EMBED_DIM)
        r_flat = ref.reshape(-1, EMBED_DIM)
        cos = np.sum(c_flat * r_flat, axis=-1) / (
            np.linalg.norm(c_flat, axis=-1) * np.linalg.norm(r_flat, axis=-1) + 1e-12
        )
        min_cos = min(min_cos, float(cos.min()))
        frames_audited += len(sample_f)

    audit_report = {
        "episodes_checked": len(audit_ids),
        "frames_checked": frames_audited,
        "max_abs_diff": max_abs_diff,
        "min_cosine_sim": min_cos,
    }
    if min_cos < 0.9999 or max_abs_diff > 0.05:
        raise RuntimeError(f"Dense CLIP cache consistency audit failed: {audit_report}")

    # Atomic save of numpy arrays
    t_save = time.monotonic()
    with TemporaryDirectory(dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        np.save(tmp_dir / "frame_features.npy", all_features)
        np.save(tmp_dir / "frame_offsets.npy", frame_offsets)
        np.save(tmp_dir / "ep_lengths.npy", ep_lengths)
        for fname in ("frame_features.npy", "frame_offsets.npy", "ep_lengths.npy"):
            (tmp_dir / fname).replace(out_dir / fname)

    elapsed = round(time.monotonic() - t0, 2)
    meta = {
        "version": DENSE_CACHE_VERSION,
        "e5_cache_dir": str(e5_cache_dir.resolve()),
        "e7_mem_dir": str(Path(e7_mem_dir).resolve()) if use_e7 else None,
        "num_episodes": n_eps,
        "num_train_episodes": len(train_ids),
        "num_val_episodes": len(val_ids),
        "total_frames": total_frames,
        "frame_features_shape": list(all_features.shape),
        "storage_dtype": "float16",
        "storage_bytes": int(all_features.nbytes),
        "storage_gb": round(all_features.nbytes / 1e9, 4),
        "cameras": list(CAMERAS),
        "embed_dim": EMBED_DIM,
        "bpp_clip_sha256": e5_meta.get("bpp_clip_sha256"),
        "provenance": {
            "reused_total": total_reused,
            "reused_fraction": round(total_reused / max(total_frames, 1), 6),
            "reused_from_e5_query": reused_query,
            "reused_from_e5_demo": reused_demo,
            "reused_from_e7_mem": reused_e7,
            "newly_encoded_missing": total_missing,
        },
        "consistency_audit": audit_report,
        "save_seconds": round(time.monotonic() - t_save, 2),
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(meta, out_dir / "dense_cache_meta.json")
    print(
        f"[E9 DenseCache DONE] Saved {out_dir} ({meta['storage_gb']:.2f} GB) in {elapsed:.1f}s | "
        f"audit min_cos={min_cos:.7f}, max_abs_diff={max_abs_diff:.5f}",
        flush=True,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--e7-mem-dir", type=Path, default=DEFAULT_BEHAVIOR_MEM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--audit-episodes", type=int, default=16)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.audit_only:
        report = audit_existing_cache_density(args.e5_cache_dir, args.e7_mem_dir)
        print(json.dumps(report, indent=2))
        return

    build_dense_clip_cache(
        e5_cache_dir=args.e5_cache_dir,
        e7_mem_dir=args.e7_mem_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        audit_episodes=args.audit_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
