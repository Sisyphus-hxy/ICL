"""Build and verify the E7 Behavior Memory Cache (`e7/outputs/behavior_memory_cache/`).

For every Demo D (15,550 episodes total), extracts Behavior Memory entries at stride 16
(`t_i = 0, 16, 32, ... < T`, supporting variable-length Demos and tail windows):
  - `h_i^D`: `(4, 768)` visual feature `[max(0, t_i - 1):av, max(0, t_i - 1):eh, t_i:av, t_i:eh]`
  - `b_i`:   `(4, 768)` frozen Stage 1 Teacher latent `R_T(M_D, h_i^D)` where `R_T` reads
             the COMPLETE Demo memory `M_D` (never a 16-frame slice).

Reuse priority (zero modification to existing data or E5/E6 caches):
  1. Reuses existing `z_self_train.npy` / `z_self_val.npy` (`e6/outputs/stage2/z_self_cache/`)
     directly wherever `(ep_idx, t_i)` is already cached.
  2. Reuses existing CLIP features from `query_features.npy` and `demo_features.npy`
     (`e5/outputs/cache/`) wherever raw frame `f` is already cached.
  3. Encodes ONLY missing raw frames from `bpp/datasets/libero/*.hdf5` via frozen `BPPVisualEncoder`
     and computes ONLY missing `b_i` entries via frozen Stage 1 Teacher (`best.pt`).
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

from e5.bpp_compat import load_bpp_pretrained_weights
from e5.model import E5Policy
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder
from e6.data import SelfPairFeatureBank
from e6.stage1_model import load_frozen_stage1_teacher

from . import (
    BEHAVIOR_STRIDE,
    BPP_CKPT,
    DEFAULT_BEHAVIOR_MEM_DIR,
    DEFAULT_STAGE1_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    NUM_QUERY_TOKENS,
    RAW_LIBERO,
)
from .model import compute_behavior_anchor_steps


@torch.inference_mode()
def build_behavior_memory_cache(
    *,
    stage1_ckpt: Path = DEFAULT_STAGE1_CKPT,
    clip_cache_dir: Path = E5_CACHE_DIR,
    z_self_cache_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
    out_dir: Path = DEFAULT_BEHAVIOR_MEM_DIR,
    stride: int = BEHAVIOR_STRIDE,
    clip_batch_size: int = 512,
    teacher_batch_size: int = 1024,
    num_gpus: Optional[int] = None,
    use_bf16: bool = True,
    seed: int = 42,
) -> Path:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata & index arrays on CPU from E5 & E6 caches (read-only)
    s1_bank = SelfPairFeatureBank(
        cache_dir=clip_cache_dir,
        device=torch.device("cpu"),
        resident_on_device=False,
        seed=seed,
    )
    episodes = s1_bank.episodes
    n_eps = s1_bank.num_episodes

    demo_feats_mmap = np.load(clip_cache_dir / "demo_features.npy", mmap_mode="r")
    query_feats_mmap = np.load(clip_cache_dir / "query_features.npy", mmap_mode="r")
    demo_offsets = s1_bank.demo_offsets_np
    demo_lengths = s1_bank.demo_lengths_np
    demo_frame_indices = s1_bank.demo_frame_indices_np
    query_ep_offsets = s1_bank.query_ep_offsets_np
    query_frame_nums = s1_bank.query_frame_nums_np

    demo_memories_np = np.load(z_self_cache_dir / "demo_memories.npy", allow_pickle=False)
    z_self_train_mmap = np.load(z_self_cache_dir / "z_self_train.npy", mmap_mode="r")
    z_self_val_mmap = np.load(z_self_cache_dir / "z_self_val.npy", mmap_mode="r")

    # Build fast lookup `(ep_idx, frame_t) -> row_idx` for existing z_self cache
    train_z_lookup: Dict[Tuple[int, int], int] = {
        (int(ep), int(fr)): idx
        for idx, (ep, fr) in enumerate(zip(s1_bank.train_episodes, s1_bank.train_frames))
    }
    val_z_lookup: Dict[Tuple[int, int], int] = {
        (int(ep), int(fr)): idx
        for idx, (ep, fr) in enumerate(zip(s1_bank.val_episodes, s1_bank.val_frames))
    }

    # 2. Plan all stride-16 Behavior Memory entries across all 15,550 episodes
    mem_lengths = np.empty(n_eps, dtype=np.int32)
    mem_offsets = np.zeros(n_eps + 1, dtype=np.int64)
    anchor_chunks: List[np.ndarray] = []
    for ep_idx, ep in enumerate(episodes):
        anchors = compute_behavior_anchor_steps(int(ep["frames"]), stride=stride)
        mem_lengths[ep_idx] = len(anchors)
        mem_offsets[ep_idx + 1] = mem_offsets[ep_idx] + len(anchors)
        anchor_chunks.append(anchors)
    mem_anchor_steps = np.concatenate(anchor_chunks, axis=0)
    total_entries = int(mem_offsets[-1])

    mem_h = np.empty((total_entries, NUM_QUERY_TOKENS, EMBED_DIM), dtype=np.float16)
    mem_b = np.empty((total_entries, NUM_QUERY_TOKENS, EMBED_DIM), dtype=np.float16)
    b_filled = np.zeros(total_entries, dtype=bool)

    # 3. Populate cached CLIP frames & cached z_self entries, and record missing raw frames per episode
    missing_frames_by_ep: Dict[int, np.ndarray] = {}
    missing_slots_by_ep: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    reused_z_train = 0
    reused_z_val = 0
    reused_clip_slots = 0
    missing_clip_slots = 0

    for ep_idx in range(n_eps):
        m_s = int(mem_offsets[ep_idx])
        m_e = int(mem_offsets[ep_idx + 1])
        anchors = mem_anchor_steps[m_s:m_e]
        fold = episodes[ep_idx]["fold"]

        q_lo, q_hi = int(query_ep_offsets[ep_idx]), int(query_ep_offsets[ep_idx + 1])
        qf = query_frame_nums[q_lo:q_hi]
        d_lo, d_hi = int(demo_offsets[ep_idx]), int(demo_offsets[ep_idx + 1])
        df = demo_frame_indices[d_lo:d_hi]

        need_raw_set = set()
        for local_k, t_val in enumerate(anchors):
            g_idx = m_s + local_k
            t_int = int(t_val)
            prev_int = max(0, t_int - 1)

            # Reuse z_self if present
            if fold == "train":
                zr = train_z_lookup.get((ep_idx, t_int))
                if zr is not None:
                    mem_b[g_idx] = z_self_train_mmap[zr]
                    b_filled[g_idx] = True
                    reused_z_train += 1
            else:
                zr = val_z_lookup.get((ep_idx, t_int))
                if zr is not None:
                    mem_b[g_idx] = z_self_val_mmap[zr]
                    b_filled[g_idx] = True
                    reused_z_val += 1

            # Reuse CLIP visual features for prev_int (tokens 0..1) and t_int (tokens 2..3)
            for slot_pair, f_id in ((0, prev_int), (1, t_int)):
                tok_lo = slot_pair * 2
                tok_hi = tok_lo + 2
                pos_q = int(np.searchsorted(qf, f_id))
                if pos_q < len(qf) and int(qf[pos_q]) == f_id:
                    mem_h[g_idx, tok_lo:tok_hi] = query_feats_mmap[q_lo + pos_q]
                    reused_clip_slots += 1
                    continue
                pos_d = int(np.searchsorted(df, f_id))
                if pos_d < len(df) and int(df[pos_d]) == f_id:
                    mem_h[g_idx, tok_lo:tok_hi] = demo_feats_mmap[d_lo + pos_d]
                    reused_clip_slots += 1
                    continue
                need_raw_set.add(f_id)
                missing_slots_by_ep[ep_idx].append((g_idx, slot_pair, f_id))
                missing_clip_slots += 1

        if need_raw_set:
            missing_frames_by_ep[ep_idx] = np.array(sorted(need_raw_set), dtype=np.int32)

    total_unique_missing_frames = sum(len(v) for v in missing_frames_by_ep.values())
    missing_b_indices = np.flatnonzero(~b_filled)

    if torch.cuda.is_available():
        avail_gpus = torch.cuda.device_count()
        n_workers = min(avail_gpus, num_gpus) if num_gpus is not None else avail_gpus
        devices = [torch.device(f"cuda:{i}") for i in range(n_workers)]
    else:
        devices = [torch.device("cpu")]

    print(
        f"[E7 BehaviorMem] Episodes={n_eps:,}, Stride={stride}, Total entries={total_entries:,} "
        f"(len min={int(mem_lengths.min())}, mean={mem_lengths.mean():.2f}, max={int(mem_lengths.max())}) | "
        f"Reused z_self={reused_z_train + reused_z_val:,}/{total_entries:,} "
        f"({100.0 * (reused_z_train + reused_z_val) / total_entries:.2f}%), "
        f"Missing b_i={len(missing_b_indices):,} | "
        f"Missing unique CLIP frames={total_unique_missing_frames:,} across {len(missing_frames_by_ep):,} eps | "
        f"Devices={[str(d) for d in devices]}",
        flush=True,
    )

    # 4. Encode missing raw frames sequentially by task HDF5 across all available GPUs
    if total_unique_missing_frames > 0:
        t_clip = time.monotonic()
        bundle = load_bpp_pretrained_weights(ckpt_path=BPP_CKPT)
        encoders = [
            BPPVisualEncoder(pretrained_bundle=bundle, load_weights=True).to(dev).eval()
            for dev in devices
        ]
        autocast_dtype = torch.float16 if devices[0].type == "cuda" else None

        task_to_eps: Dict[str, List[int]] = defaultdict(list)
        for ep_idx in missing_frames_by_ep:
            task_to_eps[episodes[ep_idx]["task"]].append(ep_idx)
        tasks_sorted = sorted(task_to_eps.keys())

        # Accumulate episodes into multi-episode batches (~512 frames) so CLIP saturates GPU
        work_q: "queue.Queue[Optional[List[Tuple[int, str, np.ndarray, np.ndarray, np.ndarray]]]]" = (
            queue.Queue(maxsize=max(8, len(devices) * 4))
        )
        errors: List[BaseException] = []
        prog_lock = threading.Lock()
        clip_stats = {"tasks_done": 0, "frames_encoded": 0}

        def _hdf5_reader() -> None:
            pending: List[Tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
            pending_frames = 0
            try:
                for t_idx, task in enumerate(tasks_sorted, start=1):
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
                            f_indices = missing_frames_by_ep[ep_idx]
                            # Direct fancy indexing on contiguous uncompressed HDF5 (avoids full video read)
                            av_sub = np.ascontiguousarray(obs_grp["agentview_rgb"][f_indices])
                            eh_sub = np.ascontiguousarray(obs_grp["eye_in_hand_rgb"][f_indices])
                            pending.append((ep_idx, row["convention"], f_indices, av_sub, eh_sub))
                            pending_frames += len(f_indices)
                            if pending_frames >= clip_batch_size:
                                work_q.put(pending)
                                pending = []
                                pending_frames = 0
                    with prog_lock:
                        clip_stats["tasks_done"] = t_idx
                        if t_idx % 40 == 0 or t_idx == len(tasks_sorted):
                            dt = max(time.monotonic() - t_clip, 1e-6)
                            print(
                                f"[E7 BehaviorMem CLIP] Tasks {t_idx}/{len(tasks_sorted)} | "
                                f"Frames {clip_stats['frames_encoded']:,}/{total_unique_missing_frames:,} "
                                f"({clip_stats['frames_encoded'] / dt:,.0f} f/s)",
                                flush=True,
                            )
                if pending and not errors:
                    work_q.put(pending)
            except BaseException as exc:
                errors.append(exc)
            finally:
                for _ in devices:
                    work_q.put(None)

        def _clip_worker(w_id: int) -> None:
            enc = encoders[w_id]
            if enc.device.type == "cuda":
                torch.cuda.set_device(enc.device)
            try:
                while True:
                    batch_items = work_q.get()
                    if batch_items is None:
                        break
                    # Group by camera convention (almost all are 'opengl') for large batched CLIP calls
                    by_conv: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = (
                        defaultdict(list)
                    )
                    for ep_idx, conv, f_indices, av_sub, eh_sub in batch_items:
                        by_conv[conv].append((ep_idx, f_indices, av_sub, eh_sub))
                    total_n = 0
                    for conv, group in by_conv.items():
                        av_cat = np.concatenate([g[2] for g in group], axis=0)
                        eh_cat = np.concatenate([g[3] for g in group], axis=0)
                        feats_cat = (
                            enc.encode_dual_camera_uint8(
                                av_cat,
                                eh_cat,
                                convention=conv,
                                batch_size=clip_batch_size,
                                autocast_dtype=autocast_dtype,
                            )
                            .to(dtype=torch.float16)
                            .cpu()
                            .numpy()
                        )
                        offset = 0
                        for ep_idx, f_indices, av_sub, _ in group:
                            cnt = len(f_indices)
                            feats = feats_cat[offset : offset + cnt]
                            offset += cnt
                            for g_idx, slot_pair, f_id in missing_slots_by_ep[ep_idx]:
                                pos = int(np.searchsorted(f_indices, f_id))
                                tok_lo = slot_pair * 2
                                mem_h[g_idx, tok_lo : tok_lo + 2] = feats[pos]
                        total_n += len(av_cat)
                    with prog_lock:
                        clip_stats["frames_encoded"] += total_n
            except BaseException as exc:
                errors.append(exc)

        r_thread = threading.Thread(target=_hdf5_reader, daemon=True)
        w_threads = [
            threading.Thread(target=_clip_worker, args=(i,), daemon=True)
            for i in range(len(devices))
        ]
        r_thread.start()
        for wt in w_threads:
            wt.start()
        r_thread.join()
        for wt in w_threads:
            wt.join()
        if errors:
            raise RuntimeError(f"CLIP feature completion failed: {errors[0]}") from errors[0]

        del encoders, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. Compute missing b_i = R_T(M_D, h_i^D) using frozen Stage 1 Teacher on FULL Demo memory M_D
    primary_dev = devices[0]
    if primary_dev.type == "cuda":
        torch.cuda.set_device(primary_dev)
    teacher = load_frozen_stage1_teacher(stage1_ckpt, device=primary_dev)
    rr = teacher.reference_reader
    amp_ctx = (
        torch.autocast(device_type=primary_dev.type, dtype=torch.bfloat16)
        if (use_bf16 and primary_dev.type == "cuda")
        else torch.autocast(device_type=primary_dev.type, enabled=False)
    )

    demo_mem_gpu = torch.from_numpy(demo_memories_np).to(primary_dev)
    demo_off_gpu = torch.from_numpy(demo_offsets).to(primary_dev)
    demo_len_gpu = torch.from_numpy(demo_lengths.astype(np.int64)).to(primary_dev)

    # Map every global Behavior Memory entry index -> its parent episode index
    entry_ep_ids = np.repeat(np.arange(n_eps, dtype=np.int64), mem_lengths)

    if len(missing_b_indices) > 0:
        t_teacher = time.monotonic()
        n_miss = len(missing_b_indices)
        for start in range(0, n_miss, teacher_batch_size):
            end = min(start + teacher_batch_size, n_miss)
            g_idxs = missing_b_indices[start:end]
            ep_t = torch.as_tensor(entry_ep_ids[g_idxs], dtype=torch.long, device=primary_dev)
            h_batch = torch.from_numpy(mem_h[g_idxs]).to(
                device=primary_dev, dtype=torch.float32
            )

            lengths = demo_len_gpu[ep_t]
            max_p = int(lengths.max().item())
            starts = demo_off_gpu[ep_t]
            pos = torch.arange(max_p, device=primary_dev, dtype=torch.long)[None, :]
            pad_mask = pos >= lengths[:, None]
            valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
            d_mem = demo_mem_gpu[starts[:, None] + valid_pos].float()
            d_mem = d_mem.masked_fill(pad_mask[:, :, None], 0.0)

            with amp_ctx:
                tgt = h_batch + rr.prompt_current_obs_pos_emb
                b_out = rr.prompt_with_obs_decoder(
                    tgt=tgt,
                    memory=d_mem,
                    memory_key_padding_mask=pad_mask,
                    need_weights=False,
                )
            mem_b[g_idxs] = b_out.float().to(dtype=torch.float16).cpu().numpy()
            b_filled[g_idxs] = True

            if (start // teacher_batch_size) % 40 == 0 or end == n_miss:
                dt = max(time.monotonic() - t_teacher, 1e-6)
                print(
                    f"[E7 BehaviorMem Teacher] {end:,}/{n_miss:,} ({end / dt:,.0f} entries/s)",
                    flush=True,
                )

    if not b_filled.all():
        raise RuntimeError("Some Behavior Memory b_i entries were not filled!")
    if not np.isfinite(mem_h).all():
        raise RuntimeError("Non-finite values detected in mem_h")
    if not np.isfinite(mem_b).all():
        raise RuntimeError("Non-finite values detected in mem_b")

    # 6. Consistency audit: verify reused z_self entries match fresh Teacher forward R_T(M_D, h_i^D)
    reused_indices = np.flatnonzero(
        np.isin(np.arange(total_entries), missing_b_indices, invert=True)
    )
    rng = np.random.default_rng(seed)
    check_sel = rng.choice(
        reused_indices, size=min(256, len(reused_indices)), replace=False
    )
    ep_chk = torch.as_tensor(entry_ep_ids[check_sel], dtype=torch.long, device=primary_dev)
    h_chk = torch.from_numpy(mem_h[check_sel]).to(device=primary_dev, dtype=torch.float32)
    lens_chk = demo_len_gpu[ep_chk]
    max_p_chk = int(lens_chk.max().item())
    pos_chk = torch.arange(max_p_chk, device=primary_dev, dtype=torch.long)[None, :]
    pm_chk = pos_chk >= lens_chk[:, None]
    vp_chk = torch.minimum(pos_chk, (lens_chk - 1).clamp_min(0)[:, None])
    dm_chk = demo_mem_gpu[demo_off_gpu[ep_chk][:, None] + vp_chk].float()
    dm_chk = dm_chk.masked_fill(pm_chk[:, :, None], 0.0)
    with amp_ctx:
        b_fresh = rr.prompt_with_obs_decoder(
            tgt=h_chk + rr.prompt_current_obs_pos_emb,
            memory=dm_chk,
            memory_key_padding_mask=pm_chk,
            need_weights=False,
        ).float()
    b_cached = torch.from_numpy(mem_b[check_sel]).to(device=primary_dev, dtype=torch.float32)
    cos_chk = torch.nn.functional.cosine_similarity(
        b_fresh.reshape(len(check_sel), -1),
        b_cached.reshape(len(check_sel), -1),
        dim=-1,
    )
    min_cos_chk = float(cos_chk.min().item())
    max_abs_chk = float((b_fresh - b_cached).abs().max().item())
    if min_cos_chk < 0.9999:
        raise RuntimeError(
            f"Consistency check failed between reused z_self and fresh R_T: min_cos={min_cos_chk}"
        )

    # Compute RMS statistics per fold
    train_ep_mask = np.array([e["fold"] == "train" for e in episodes], dtype=bool)
    entry_is_train = train_ep_mask[entry_ep_ids]
    train_b_rms = float(
        math.sqrt(
            float(np.mean(mem_b[entry_is_train].astype(np.float32) ** 2, dtype=np.float64))
        )
    )
    val_b_rms = float(
        math.sqrt(
            float(np.mean(mem_b[~entry_is_train].astype(np.float32) ** 2, dtype=np.float64))
        )
    )

    elapsed = round(time.monotonic() - t0, 2)
    meta: Dict[str, Any] = {
        "version": "e7_behavior_memory_stride16_v1",
        "stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "clip_cache_dir": str(Path(clip_cache_dir).resolve()),
        "z_self_cache_dir": str(Path(z_self_cache_dir).resolve()),
        "behavior_stride": int(stride),
        "num_episodes": int(n_eps),
        "total_entries": int(total_entries),
        "mem_h_shape": list(mem_h.shape),
        "mem_b_shape": list(mem_b.shape),
        "storage_dtype": "float16",
        "entries_per_demo": {
            "min": int(mem_lengths.min()),
            "median": float(np.median(mem_lengths)),
            "mean": float(mem_lengths.mean()),
            "max": int(mem_lengths.max()),
        },
        "provenance": {
            "reused_z_self_train": int(reused_z_train),
            "reused_z_self_val": int(reused_z_val),
            "reused_z_self_total": int(reused_z_train + reused_z_val),
            "reused_z_self_fraction": round(
                (reused_z_train + reused_z_val) / max(total_entries, 1), 6
            ),
            "computed_by_frozen_teacher": int(len(missing_b_indices)),
            "reused_clip_frame_slots": int(reused_clip_slots),
            "newly_encoded_clip_frame_slots": int(missing_clip_slots),
            "newly_encoded_unique_frames": int(total_unique_missing_frames),
        },
        "train_b_rms": train_b_rms,
        "val_b_rms": val_b_rms,
        "consistency_audit": {
            "checked_reused_entries": int(len(check_sel)),
            "min_cosine_sim": min_cos_chk,
            "max_abs_diff": max_abs_chk,
        },
        "elapsed_seconds": elapsed,
    }

    with TemporaryDirectory(dir=out_dir) as tmp:
        staging = Path(tmp)
        np.save(staging / "mem_h.npy", mem_h)
        np.save(staging / "mem_b.npy", mem_b)
        np.save(staging / "mem_offsets.npy", mem_offsets)
        np.save(staging / "mem_lengths.npy", mem_lengths)
        np.save(staging / "mem_anchor_steps.npy", mem_anchor_steps)
        (staging / "behavior_memory_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        for p in staging.iterdir():
            p.replace(out_dir / p.name)

    atomic_write_json(meta["consistency_audit"], out_dir / "consistency_audit.json")
    print(
        f"[E7 BehaviorMem DONE] {total_entries:,} entries in {elapsed}s -> {out_dir} | "
        f"train_b_rms={train_b_rms:.4f}, val_b_rms={val_b_rms:.4f}, "
        f"audit_min_cos={min_cos_chk:.6f}",
        flush=True,
    )
    return out_dir / "behavior_memory_meta.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--clip-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--z-self-cache-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_BEHAVIOR_MEM_DIR)
    parser.add_argument("--stride", type=int, default=BEHAVIOR_STRIDE)
    parser.add_argument("--clip-batch-size", type=int, default=512)
    parser.add_argument("--teacher-batch-size", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_behavior_memory_cache(
        stage1_ckpt=args.stage1_ckpt,
        clip_cache_dir=args.clip_cache_dir,
        z_self_cache_dir=args.z_self_cache_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        clip_batch_size=args.clip_batch_size,
        teacher_batch_size=args.teacher_batch_size,
        num_gpus=args.num_gpus,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
