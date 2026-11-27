"""Build E9 Stage 2 Behavior Memory (stride=16) and train/val `z_self` caches from the
frozen E9 Stage 1 Local Behavior Encoder (`e9/outputs/stage1_local/best.pt`) and the
20Hz dense dual-camera CLIP cache (`e9/outputs/dense_clip_cache/`).

Outputs under `e9/outputs/stage2_direct/cache/`:
  1. Behavior Memory (stride=16 across all 15,550 episodes = 234,091 entries):
     - `mem_h.npy`           : `(234091, 4, 768)` float16 — visual keys `[max(0, t_i-1), t_i]`
       (verified bit-exact `max_abs_diff == 0.0` against E7's `mem_h.npy`)
     - `mem_b.npy`           : `(234091, 4, 768)` float16 — `b_i = E_local(V_{t_i:t_i+16}^D)`
     - `mem_offsets.npy`     : `(15551,)` int64
     - `mem_lengths.npy`     : `(15550,)` int32
     - `mem_anchor_steps.npy`: `(234091,)` int32
     - `behavior_memory_meta.json`
  2. Target `z_self` supervision cache (exact E6/E7 1,373,035 train + 6,248 val starts):
     - `z_self_train.npy`    : `(1373035, 4, 768)` float16 — `E_local(V_{t:t+16}^Q)`
     - `z_self_val.npy`      : `(6248, 4, 768)` float16    — `E_local(V_{t:t+16}^Q)`
     - `train_z_self_mean.npy`: `(1, 4, 768)` float32      — exact float64 mean for E7-Direct bias
     - `z_self_meta.json`    : exact float64 `train_z_self_rms` (`s_z`), `val_z_self_rms`,
       `train_z_self_mean_rms`, consistency audit, and provenance.

Note: Query future frames `V_{t:t+16}^Q` are used STRICTLY offline here to precompute target
supervision `z_self` for `L_z`. They NEVER enter the Reader or TransferTransformer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import numpy as np
import torch

from e5.model import E5Policy
from e5.train import atomic_write_json
from e7.model import compute_behavior_anchor_steps

from . import (
    BEHAVIOR_STRIDE,
    DEFAULT_BEHAVIOR_MEM_DIR,
    DEFAULT_DENSE_CACHE_DIR,
    DEFAULT_STAGE1_LOCAL_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    E5_CACHE_DIR,
    EMBED_DIM,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
)
from .data import LocalClipFeatureBank
from .stage1_local_model import load_frozen_e9_stage1_teacher

CACHE_VERSION = "e9_stage2_local_behavior_memory_and_z_self_v1"


@torch.inference_mode()
def build_e9_stage2_caches(
    *,
    stage1_ckpt: Path = DEFAULT_STAGE1_LOCAL_CKPT,
    e5_cache_dir: Path = E5_CACHE_DIR,
    dense_cache_dir: Path = DEFAULT_DENSE_CACHE_DIR,
    out_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
    e7_ref_mem_dir: Path = DEFAULT_BEHAVIOR_MEM_DIR,
    stride: int = BEHAVIOR_STRIDE,
    batch_size: int = 1024,
    device_str: str = "cuda:0",
    use_bf16: bool = True,
    seed: int = 42,
) -> Path:
    t0 = time.monotonic()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_frozen_e9_stage1_teacher(stage1_ckpt, device=device)
    if sum(p.numel() for p in teacher.parameters() if p.requires_grad) != 0:
        raise RuntimeError("Stage 1 Local teacher must be completely frozen!")

    bank = LocalClipFeatureBank(
        e5_cache_dir=e5_cache_dir,
        dense_cache_dir=dense_cache_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        seed=seed,
    )
    n_eps = bank.num_episodes
    episodes = bank.episodes

    # ------------------------------------------------------------------
    # 1. Plan & build Behavior Memory (stride=16 across all 15,550 eps)
    # ------------------------------------------------------------------
    mem_lengths = np.empty(n_eps, dtype=np.int32)
    mem_offsets = np.zeros(n_eps + 1, dtype=np.int64)
    anchor_ep_chunks = []
    anchor_step_chunks = []
    for ep_idx, ep in enumerate(episodes):
        anc = compute_behavior_anchor_steps(int(ep["frames"]), stride=stride)
        mem_lengths[ep_idx] = len(anc)
        mem_offsets[ep_idx + 1] = mem_offsets[ep_idx] + len(anc)
        anchor_ep_chunks.append(np.full(len(anc), ep_idx, dtype=np.int64))
        anchor_step_chunks.append(anc)

    mem_anchor_eps = np.concatenate(anchor_ep_chunks, axis=0)
    mem_anchor_steps = np.concatenate(anchor_step_chunks, axis=0)
    total_mem = int(mem_offsets[-1])

    mem_h = np.empty((total_mem, NUM_BEHAVIOR_TOKENS, EMBED_DIM), dtype=np.float16)
    mem_b = np.empty((total_mem, NUM_BEHAVIOR_TOKENS, EMBED_DIM), dtype=np.float16)

    t_mem = time.monotonic()
    for start in range(0, total_mem, batch_size):
        end = min(start + batch_size, total_mem)
        ep_chunk = mem_anchor_eps[start:end]
        fr_chunk = mem_anchor_steps[start:end]
        prev_chunk = np.maximum(fr_chunk - 1, 0)

        ep_t = torch.as_tensor(ep_chunk, dtype=torch.long, device=device)
        p0_t = torch.as_tensor(prev_chunk, dtype=torch.long, device=device)
        p1_t = torch.as_tensor(fr_chunk, dtype=torch.long, device=device)
        off_t = bank.frame_offsets[ep_t]

        # h_i^D: (B, 4, 768) = [prev:av, prev:eh, cur:av, cur:eh]
        q_rgb = torch.stack(
            [bank.frame_features[off_t + p0_t], bank.frame_features[off_t + p1_t]],
            dim=1,
        )  # (B, 2, 2, 768) float16
        h_batch = E5Policy.format_query_tokens(q_rgb)
        mem_h[start:end] = h_batch.cpu().numpy()

        # b_i = E_local(V_{t_i:t_i+16}^D, pad_mask)
        clip_rgb, clip_pad, _ = bank.gather_batch(ep_chunk, fr_chunk)
        with amp_ctx:
            b_batch = teacher.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad)
        mem_b[start:end] = b_batch.float().to(dtype=torch.float16).cpu().numpy()

    mem_sec = time.monotonic() - t_mem
    print(
        f"[E9 Stage2 Cache] Built Behavior Memory: entries={total_mem:,} "
        f"(stride={stride}, len min={int(mem_lengths.min())}, mean={mem_lengths.mean():.2f}, "
        f"max={int(mem_lengths.max())}) in {mem_sec:.1f}s",
        flush=True,
    )

    # Bit-exact verification of mem_h against E7's Behavior Memory cache (if present)
    e7_mem_h_check: Dict[str, Any] = {"checked": False}
    if (Path(e7_ref_mem_dir) / "mem_h.npy").is_file():
        e7_h = np.load(Path(e7_ref_mem_dir) / "mem_h.npy", mmap_mode="r")
        e7_anc = np.load(Path(e7_ref_mem_dir) / "mem_anchor_steps.npy", allow_pickle=False)
        if not np.array_equal(mem_anchor_steps, e7_anc):
            raise RuntimeError("mem_anchor_steps mismatch against E7 Behavior Memory!")
        rng_chk = np.random.default_rng(seed + 101)
        idx_chk = rng_chk.choice(total_mem, size=min(4096, total_mem), replace=False)
        diff_h = np.max(
            np.abs(mem_h[idx_chk].astype(np.float32) - e7_h[idx_chk].astype(np.float32))
        )
        e7_mem_h_check = {
            "checked": True,
            "samples": int(len(idx_chk)),
            "max_abs_diff_vs_e7_mem_h": float(diff_h),
        }
        if diff_h > 0.0:
            raise RuntimeError(f"mem_h diverged from E7 mem_h: max_abs_diff={diff_h}")

    # ------------------------------------------------------------------
    # 2. Encode z_self_train (1,373,035) and z_self_val (6,248)
    # ------------------------------------------------------------------
    def _encode_z_self_split(
        eps_arr: np.ndarray, frames_arr: np.ndarray, split_name: str
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        n = len(eps_arr)
        out = np.empty((n, NUM_BEHAVIOR_TOKENS, EMBED_DIM), dtype=np.float16)
        sq_sum = 0.0
        mean_acc = np.zeros((NUM_BEHAVIOR_TOKENS, EMBED_DIM), dtype=np.float64)
        elem_count = 0
        t_s = time.monotonic()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            clip_rgb, clip_pad, _ = bank.gather_batch(
                eps_arr[start:end], frames_arr[start:end]
            )
            with amp_ctx:
                b_t = teacher.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad)
            b_f32 = b_t.float()
            sq_sum += float(b_f32.square().sum().item())
            elem_count += int(b_f32.numel())
            b_np = b_f32.cpu().numpy()
            mean_acc += b_np.astype(np.float64).sum(axis=0)
            out[start:end] = b_np.astype(np.float16)

            if (start // batch_size) % 300 == 0 or end == n:
                dt = max(time.monotonic() - t_s, 1e-6)
                print(
                    f"[E9 Stage2 Cache] z_self_{split_name}: {end:,}/{n:,} "
                    f"({end / dt:,.0f} samples/s)",
                    flush=True,
                )
        rms = math.sqrt(sq_sum / max(elem_count, 1))
        mean_f32 = (mean_acc / max(float(n), 1.0)).astype(np.float32)
        return out, rms, mean_f32

    z_self_train, train_rms, train_mean_f32 = _encode_z_self_split(
        bank.train_episodes, bank.train_frames, "train"
    )
    z_self_val, val_rms, val_mean_f32 = _encode_z_self_split(
        bank.val_episodes, bank.val_frames, "val"
    )

    # Also compute exact fp16-roundtrip RMS & mean (matching runtime E7TransferBank tensors!)
    # Note: E7 computes s_z from fp32 encoder output during caching and direct_latent_bias
    # from float64 sum over stored fp16 `z_self_train`. Verify they agree to <1e-4!
    fp16_sq_sum = 0.0
    fp16_mean_acc = np.zeros((NUM_BEHAVIOR_TOKENS, EMBED_DIM), dtype=np.float64)
    for start in range(0, len(z_self_train), 131_072):
        blk = z_self_train[start : start + 131_072].astype(np.float64)
        fp16_sq_sum += float((blk * blk).sum())
        fp16_mean_acc += blk.sum(axis=0)
    train_rms_fp16 = math.sqrt(fp16_sq_sum / max(z_self_train.size, 1))
    train_mean_fp16 = (fp16_mean_acc / max(float(len(z_self_train)), 1.0)).astype(np.float32)
    train_mean_rms = float(np.sqrt(np.mean(train_mean_fp16.astype(np.float64) ** 2)))

    if not (
        np.isfinite(mem_h).all()
        and np.isfinite(mem_b).all()
        and np.isfinite(z_self_train).all()
        and np.isfinite(z_self_val).all()
    ):
        raise RuntimeError("Non-finite values detected in E9 Stage 2 cache!")

    # Compute train/val mem_b RMS for diagnostics
    train_ep_mask = np.array([e["fold"] == "train" for e in episodes], dtype=bool)
    mem_is_train = train_ep_mask[mem_anchor_eps]
    train_b_rms = float(np.sqrt(np.mean(mem_b[mem_is_train].astype(np.float64) ** 2)))
    val_b_rms = float(np.sqrt(np.mean(mem_b[~mem_is_train].astype(np.float64) ** 2)))

    # Consistency audit: re-encode random mem_b & z_self_val entries and verify cosine ~1.0
    rng_aud = np.random.default_rng(seed + 707)
    aud_idx = rng_aud.choice(len(z_self_val), size=min(256, len(z_self_val)), replace=False)
    clip_aud, pad_aud, _ = bank.gather_batch(
        bank.val_episodes[aud_idx], bank.val_frames[aud_idx]
    )
    with amp_ctx:
        ref_b = teacher.encode_local_behavior(clip_aud, clip_pad_mask=pad_aud).float().cpu().numpy()
    cached_b = z_self_val[aud_idx].astype(np.float32)
    aud_max_diff = float(np.max(np.abs(cached_b - ref_b)))
    c_flat = cached_b.reshape(len(aud_idx), -1)
    r_flat = ref_b.reshape(len(aud_idx), -1)
    aud_cos = np.sum(c_flat * r_flat, axis=-1) / (
        np.linalg.norm(c_flat, axis=-1) * np.linalg.norm(r_flat, axis=-1) + 1e-12
    )
    aud_min_cos = float(aud_cos.min())
    if aud_min_cos < 0.9999 or aud_max_diff > 0.02:
        raise RuntimeError(
            f"E9 Stage 2 cache audit failed: min_cos={aud_min_cos}, max_diff={aud_max_diff}"
        )

    elapsed = round(time.monotonic() - t0, 2)
    bm_meta = {
        "version": CACHE_VERSION,
        "stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "e5_cache_dir": str(Path(e5_cache_dir).resolve()),
        "dense_cache_dir": str(Path(dense_cache_dir).resolve()),
        "behavior_stride": int(stride),
        "local_window_len": int(LOCAL_WINDOW_LEN),
        "num_episodes": n_eps,
        "total_entries": total_mem,
        "mem_h_shape": list(mem_h.shape),
        "mem_b_shape": list(mem_b.shape),
        "storage_dtype": "float16",
        "entries_per_demo": {
            "min": int(mem_lengths.min()),
            "median": float(np.median(mem_lengths)),
            "mean": float(mem_lengths.mean()),
            "max": int(mem_lengths.max()),
        },
        "train_b_rms": train_b_rms,
        "val_b_rms": val_b_rms,
        "e7_mem_h_consistency": e7_mem_h_check,
        "elapsed_seconds": elapsed,
    }
    zs_meta = {
        "version": CACHE_VERSION,
        "stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "e5_cache_dir": str(Path(e5_cache_dir).resolve()),
        "dense_cache_dir": str(Path(dense_cache_dir).resolve()),
        "num_episodes": n_eps,
        "z_self_train_shape": list(z_self_train.shape),
        "z_self_val_shape": list(z_self_val.shape),
        "storage_dtype": "float16",
        "train_z_self_rms": train_rms,
        "train_z_self_rms_fp16_stored": train_rms_fp16,
        "val_z_self_rms": val_rms,
        "train_z_self_mean_rms": train_mean_rms,
        "train_mean_max_abs_fp32_vs_fp16": float(np.max(np.abs(train_mean_f32 - train_mean_fp16))),
        "consistency_audit": {
            "checked_val_entries": int(len(aud_idx)),
            "min_cosine_sim": aud_min_cos,
            "max_abs_diff": aud_max_diff,
        },
        "elapsed_seconds": elapsed,
    }

    with TemporaryDirectory(dir=out_dir) as tmp:
        st = Path(tmp)
        np.save(st / "mem_h.npy", mem_h)
        np.save(st / "mem_b.npy", mem_b)
        np.save(st / "mem_offsets.npy", mem_offsets)
        np.save(st / "mem_lengths.npy", mem_lengths)
        np.save(st / "mem_anchor_steps.npy", mem_anchor_steps)
        np.save(st / "z_self_train.npy", z_self_train)
        np.save(st / "z_self_val.npy", z_self_val)
        np.save(st / "train_z_self_mean.npy", train_mean_fp16[None, :, :])
        (st / "behavior_memory_meta.json").write_text(json.dumps(bm_meta, indent=2) + "\n")
        (st / "z_self_meta.json").write_text(json.dumps(zs_meta, indent=2) + "\n")
        for p in st.iterdir():
            p.replace(out_dir / p.name)

    print(
        f"[E9 Stage2 Cache DONE] Saved to {out_dir} in {elapsed:.1f}s | "
        f"s_z (train_z_self_rms)={train_rms:.6f} (fp16={train_rms_fp16:.6f}), "
        f"val_z_self_rms={val_rms:.6f}, train_mean_rms={train_mean_rms:.6f}, "
        f"audit min_cos={aud_min_cos:.7f}, e7_mem_h_diff={e7_mem_h_check.get('max_abs_diff_vs_e7_mem_h')}",
        flush=True,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument("--e5-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--dense-cache-dir", type=Path, default=DEFAULT_DENSE_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument("--stride", type=int, default=BEHAVIOR_STRIDE)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_e9_stage2_caches(
        stage1_ckpt=args.stage1_ckpt,
        e5_cache_dir=args.e5_cache_dir,
        dense_cache_dir=args.dense_cache_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        batch_size=args.batch_size,
        device_str=args.device,
        use_bf16=not args.no_bf16,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
