"""Step 1-3 for E6 Stage 2:
1. Freeze Stage 1 Teacher (`e6/outputs/stage1/stage1_self_latent/best.pt`).
2. Pre-cache frozen Demo memories (`M_D + prompt_pos_emb`) for all 15,550 episodes
   and `z_self` for all 1,373,035 train starts + 6,248 val starts.
3. Compute exact fixed training-set Self Latent RMS `s_z`, and audit Self vs Cross
   No-Delta latent statistics (RMS, Cosine, raw/normalized Latent Gap, `L_z`, `0.5*L_z`,
   `L_diff`, and DDIM Action MSE/MAE) to validate loss scaling before Stage 2 training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from e5.train import _action_error_sums, _unpack_action_errors, atomic_write_json

from . import ACTION_DIM_10D, ACTION_HORIZON, E5_CACHE_DIR, EMBED_DIM, NUM_QUERY_TOKENS
from .data import SelfPairFeatureBank
from .stage1_model import load_frozen_stage1_teacher
from .stage2_data import DEFAULT_STAGE2_CACHE_DIR, Stage2TransferBank
from .stage2_model import DEFAULT_STAGE1_CKPT, E6Stage2Policy


@torch.inference_mode()
def build_stage2_z_self_cache(
    *,
    stage1_ckpt: Path = DEFAULT_STAGE1_CKPT,
    clip_cache_dir: Path = E5_CACHE_DIR,
    out_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
    batch_size: int = 1024,
    device_str: str = "cuda:0",
    use_bf16: bool = True,
) -> Path:
    """Pre-encode `demo_memories.npy`, `z_self_train.npy`, `z_self_val.npy`, and `z_self_meta.json`."""
    t0 = time.monotonic()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_frozen_stage1_teacher(stage1_ckpt, device=device)
    s2 = E6Stage2Policy(teacher).to(device).eval()

    bank = SelfPairFeatureBank(
        cache_dir=clip_cache_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        seed=42,
    )

    # 1. Pre-encode all 15,550 episodes' Demo memories: M_D + prompt_pos_emb -> (188840, 768) fp16
    n_eps = bank.num_episodes
    total_demo_steps = int(bank.demo_offsets_np[-1])
    demo_memories = np.empty((total_demo_steps, EMBED_DIM), dtype=np.float16)
    ep_batch = 512
    for start in range(0, n_eps, ep_batch):
        end = min(start + ep_batch, n_eps)
        ep_ids = torch.arange(start, end, device=device, dtype=torch.long)
        lengths = bank.demo_lengths[ep_ids]
        max_p = int(lengths.max().item())
        starts = bank.demo_offsets[ep_ids]
        pos = torch.arange(max_p, device=device, dtype=torch.long)[None, :]
        pad_mask = pos >= lengths[:, None]
        valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
        demo_rgb = bank.demo_features[starts[:, None] + valid_pos].float()
        demo_rgb = demo_rgb.masked_fill(pad_mask[:, :, None, None], 0.0)
        with amp_ctx:
            mem = s2.encode_demo_memory(demo_rgb, demo_pad_mask=pad_mask)
        mem_fp16 = mem.to(dtype=torch.float16).cpu().numpy()
        for local_i, ep_idx in enumerate(range(start, end)):
            d_s = int(bank.demo_offsets_np[ep_idx])
            d_e = int(bank.demo_offsets_np[ep_idx + 1])
            demo_memories[d_s:d_e] = mem_fp16[local_i, : d_e - d_s]

    demo_mem_gpu = torch.from_numpy(demo_memories).to(device)
    print(
        f"[Stage2 Cache] Encoded demo_memories {demo_memories.shape} fp16 "
        f"({demo_memories.nbytes / 1e6:.1f} MB) in {time.monotonic() - t0:.1f}s",
        flush=True,
    )

    def _encode_split_z_self(
        episodes: np.ndarray, q_rows: np.ndarray, split_name: str
    ) -> tuple[np.ndarray, float]:
        n = len(episodes)
        out = np.empty((n, NUM_QUERY_TOKENS, EMBED_DIM), dtype=np.float16)
        sq_sum = 0.0
        elem_count = 0
        t_s = time.monotonic()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ep_t = torch.as_tensor(episodes[start:end], dtype=torch.long, device=device)
            qr_t = torch.as_tensor(q_rows[start:end], dtype=torch.long, device=device)

            lengths = bank.demo_lengths[ep_t]
            max_p = int(lengths.max().item())
            starts = bank.demo_offsets[ep_t]
            pos = torch.arange(max_p, device=device, dtype=torch.long)[None, :]
            pad_mask = pos >= lengths[:, None]
            valid_pos = torch.minimum(pos, (lengths - 1).clamp_min(0)[:, None])
            d_mem = demo_mem_gpu[starts[:, None] + valid_pos].float()
            d_mem = d_mem.masked_fill(pad_mask[:, :, None], 0.0)
            q_rgb = bank.query_features[qr_t].float()

            with amp_ctx:
                z_self, _ = s2.read_latent_from_memory(d_mem, pad_mask, q_rgb)

            z_f32 = z_self.float()
            sq_sum += float(z_f32.square().sum().item())
            elem_count += int(z_f32.numel())
            out[start:end] = z_f32.to(dtype=torch.float16).cpu().numpy()

            if (start // batch_size) % 300 == 0 or end == n:
                dt = max(time.monotonic() - t_s, 1e-6)
                print(
                    f"[Stage2 Cache] {split_name}: {end:,}/{n:,} ({end / dt:,.0f} samples/s)",
                    flush=True,
                )
        rms = math.sqrt(sq_sum / max(elem_count, 1))
        return out, rms

    z_self_train, train_rms = _encode_split_z_self(
        bank.train_episodes, bank.train_q_rows, "train"
    )
    z_self_val, val_rms = _encode_split_z_self(
        bank.val_episodes, bank.val_q_rows, "val"
    )

    if not np.isfinite(demo_memories).all():
        raise RuntimeError("Non-finite values in demo_memories")
    if not np.isfinite(z_self_train).all() or not np.isfinite(z_self_val).all():
        raise RuntimeError("Non-finite values in z_self cache")

    # Atomically persist cache files
    meta = {
        "stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "clip_cache_dir": str(Path(clip_cache_dir).resolve()),
        "num_episodes": n_eps,
        "demo_memory_shape": list(demo_memories.shape),
        "z_self_train_shape": list(z_self_train.shape),
        "z_self_val_shape": list(z_self_val.shape),
        "storage_dtype": "float16",
        "train_z_self_rms": train_rms,
        "val_z_self_rms": val_rms,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }

    with TemporaryDirectory(dir=out_dir) as tmp:
        staging = Path(tmp)
        np.save(staging / "demo_memories.npy", demo_memories)
        np.save(staging / "z_self_train.npy", z_self_train)
        np.save(staging / "z_self_val.npy", z_self_val)
        (staging / "z_self_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        for p in staging.iterdir():
            p.replace(out_dir / p.name)

    print(
        f"[Stage2 Cache] Done in {meta['elapsed_seconds']}s -> {out_dir} | "
        f"s_z (train_z_self_rms)={train_rms:.6f}, val_z_self_rms={val_rms:.6f}",
        flush=True,
    )
    return out_dir / "z_self_meta.json"


@torch.inference_mode()
def audit_self_cross_baseline_stats(
    *,
    stage1_ckpt: Path = DEFAULT_STAGE1_CKPT,
    clip_cache_dir: Path = E5_CACHE_DIR,
    z_self_cache_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
    train_sample_count: int = 8192,
    ddim_val_samples: int = 1024,
    batch_size: int = 256,
    device_str: str = "cuda:0",
    seed: int = 42,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Measure Step 1 & Step 2 statistics on the frozen Stage 1 Teacher BEFORE Stage 2 training:
    - `z_self` and `z_cross` RMS, Cosine Similarity, raw & normalized Latent Gap (`L_z`)
    - `L_diff(Self)` vs `L_diff(Cross No-Delta)` vs `0.5 * L_z` scale check
    - DDIM Action MSE/MAE of `Self Teacher` vs `Cross No-Delta` vs `Wrong Demo`
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    teacher = load_frozen_stage1_teacher(stage1_ckpt, device=device)
    bank = Stage2TransferBank(
        clip_cache_dir=clip_cache_dir,
        z_self_cache_dir=z_self_cache_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        z_self_on_device=(device.type == "cuda"),
        self_ratio=0.0,  # pure cross for baseline gap audit
        seed=seed,
    )
    s_z = bank.latent_scale_s_z
    s2 = E6Stage2Policy(teacher, latent_scale_s_z=s_z).to(device).eval()

    def _measure_split(split: str, max_n: int) -> Dict[str, float]:
        gen = torch.Generator(device="cpu").manual_seed(seed + (111 if split == "train" else 222))
        acc = torch.zeros(9, device=device, dtype=torch.float64)
        if split == "train":
            it = bank.iter_train_batches(1, batch_size, max_samples=max_n)
        else:
            it = bank.iter_val_batches("cross_a", batch_size, max_samples=max_n)

        for d_mem, pad_m, q_rgb, z_self, act10, is_self in it:
            keep = ~is_self
            if not keep.any():
                continue
            d_mem, pad_m, q_rgb, z_self, act10 = (
                d_mem[keep],
                pad_m[keep],
                q_rgb[keep],
                z_self[keep],
                act10[keep],
            )
            B = len(d_mem)
            noise = torch.randn((B, ACTION_HORIZON, ACTION_DIM_10D), generator=gen).to(device)
            pertub = torch.randn((B, ACTION_HORIZON, ACTION_DIM_10D), generator=gen).to(device)
            t = torch.randint(
                0,
                teacher.noise_scheduler.config.num_train_timesteps,
                (B,),
                generator=gen,
                dtype=torch.long,
            ).to(device)

            with amp_ctx:
                z_cross, _ = s2.read_latent_from_memory(d_mem, pad_m, q_rgb)
                l_diff_self = teacher.diffusion_loss_from_z_self(
                    z_self, act10, noise=noise, pertub_noise=pertub, timesteps=t
                )
                l_diff_cross = teacher.diffusion_loss_from_z_self(
                    z_cross, act10, noise=noise, pertub_noise=pertub, timesteps=t
                )

            zc = z_cross.float()
            zs = z_self.float()
            gap_sq = (zc - zs).square().mean(dim=(1, 2))
            lz_norm = ((zc - zs) / s_z).square().mean(dim=(1, 2))
            cos = F.cosine_similarity(zc.reshape(B, -1), zs.reshape(B, -1), dim=-1)

            acc[0] += float(B)
            acc[1] += zs.square().mean(dim=(1, 2)).double().sum()
            acc[2] += zc.square().mean(dim=(1, 2)).double().sum()
            acc[3] += gap_sq.double().sum()
            acc[4] += lz_norm.double().sum()
            acc[5] += cos.double().sum()
            acc[6] += l_diff_self.double().sum()
            acc[7] += l_diff_cross.double().sum()
            acc[8] += (zc - zs).norm(dim=(1, 2)).double().sum()

        n = max(float(acc[0].item()), 1.0)
        lz_mean = float(acc[4].item() / n)
        l_diff_cross_mean = float(acc[7].item() / n)
        l_diff_self_mean = float(acc[6].item() / n)
        return {
            "count": int(round(n)),
            "z_self_rms": math.sqrt(float(acc[1].item() / n)),
            "z_cross_rms": math.sqrt(float(acc[2].item() / n)),
            "latent_mse_raw": float(acc[3].item() / n),
            "latent_rmse_raw": math.sqrt(float(acc[3].item() / n)),
            "latent_l2_dist": float(acc[8].item() / n),
            "L_z_normalized": lz_mean,
            "half_L_z": 0.5 * lz_mean,
            "cosine_sim_cross_self": float(acc[5].item() / n),
            "L_diff_self": l_diff_self_mean,
            "L_diff_cross_no_delta": l_diff_cross_mean,
            "L_diff_gap_cross_minus_self": l_diff_cross_mean - l_diff_self_mean,
            "ratio_half_Lz_to_Ldiff_cross": (0.5 * lz_mean) / max(l_diff_cross_mean, 1e-12),
            "ratio_Lz_to_Ldiff_cross": lz_mean / max(l_diff_cross_mean, 1e-12),
        }

    train_stats = _measure_split("train", train_sample_count)
    val_stats = _measure_split("val", bank.num_val_samples())

    # Val DDIM comparison: Self Teacher vs Cross No-Delta vs Wrong Demo
    gen_ddim = torch.Generator(device="cpu").manual_seed(seed + 777)
    n_ddim = min(ddim_val_samples, int(bank.val_valid_cross_mask.sum()))
    init_noise_all = torch.randn((n_ddim, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_ddim)

    err_self = torch.zeros(11, device=device, dtype=torch.float64)
    err_cross = torch.zeros(11, device=device, dtype=torch.float64)
    err_wrong = torch.zeros(11, device=device, dtype=torch.float64)

    cur = 0
    for (dm_c, pm_c, q_c, zs_c, a_c, _), (dm_w, pm_w, q_w, _, a_w, _) in zip(
        bank.iter_val_batches("cross_a", batch_size, max_samples=n_ddim),
        bank.iter_val_batches("wrong_demo", batch_size, max_samples=n_ddim),
    ):
        B = len(dm_c)
        z0 = init_noise_all[cur : cur + B].to(device)
        cur += B
        with amp_ctx:
            pred_self = teacher.sample_from_z_self(zs_c, init_noise=z0)
            out_cross = s2.predict_action_from_memory(
                dm_c, pm_c, q_c, apply_delta=False, init_noise=z0
            )
            out_wrong = s2.predict_action_from_memory(
                dm_w, pm_w, q_w, apply_delta=False, init_noise=z0
            )
        err_self += _action_error_sums(pred_self, a_c)
        err_cross += _action_error_sums(out_cross["action_10d"], a_c)
        err_wrong += _action_error_sums(out_wrong["action_10d"], a_w)

    report = {
        "stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "fixed_s_z_train_rms": s_z,
        "train_self_cross_stats": train_stats,
        "val_self_cross_stats": val_stats,
        "val_ddim_baseline": {
            "self_teacher": _unpack_action_errors(err_self),
            "cross_no_delta": _unpack_action_errors(err_cross),
            "wrong_demo_no_delta": _unpack_action_errors(err_wrong),
        },
    }
    out_path = Path(z_self_cache_dir) / "baseline_self_cross_stats.json"
    atomic_write_json(report, out_path)
    print("\n[Stage2 Step 1-2 Self/Cross Baseline Stats]", json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--clip-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.audit_only:
        build_stage2_z_self_cache(
            stage1_ckpt=args.stage1_ckpt,
            clip_cache_dir=args.clip_cache_dir,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            device_str=args.device,
        )
    audit_self_cross_baseline_stats(
        stage1_ckpt=args.stage1_ckpt,
        clip_cache_dir=args.clip_cache_dir,
        z_self_cache_dir=args.out_dir,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
