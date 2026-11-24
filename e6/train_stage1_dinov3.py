"""Training, validation, smoke test, and CLIP-comparison audit for E6 Stage 1 (DINOv3).

Strictly follows the CLIP Stage 1 `config.json` hyperparameters & evaluation protocol:
- 100% Self-pair, 8 full-pass epochs (~42,912 optimizer steps at global_batch=256)
- AdamW (lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6), cosine schedule (5% warmup),
  clip_grad_norm=0.3, bf16 mixed precision, EMA decay=0.9998
- All 6 validation metrics evaluated on full 6,248 val samples & 2,557 val timestep pairs,
  with spatial cross-attention summed over `K_m` spatial memory tokens per timestep before
  peak-shift computation for direct 1-to-1 comparability with E6-CLIP Stage 1.
- Saves independently to `e6/outputs/stage1/stage1_self_latent_dinov3/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dinov3 import BACKBONE_ID as DINOV3_BACKBONE_ID
from dinov3 import DEFAULT_CACHE_DIR as DINOV3_CACHE_DIR
from dinov3 import DEFAULT_CLIP_CACHE_DIR
from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import (
    _action_error_sums,
    _all_reduce_sum,
    _rank,
    _unpack_action_errors,
    _world,
    atomic_torch_save,
    atomic_write_json,
    seed_everything,
)

from . import ACTION_DIM_10D, ACTION_HORIZON, OUTPUT_ROOT
from .dinov3_data import DINOv3SelfPairFeatureBank, verify_dinov3_cache_compatibility
from .stage1_dinov3_model import (
    DEFAULT_CLIP_STAGE1_CKPT,
    DEFAULT_DEMO_TOKENS_PER_STEP,
    E6Stage1DINOv3Policy,
    load_frozen_stage1_dinov3_teacher,
)

DEFAULT_CLIP_FINAL_EVAL = OUTPUT_ROOT / "stage1" / "stage1_self_latent" / "final_eval.json"


@torch.inference_mode()
def evaluate_timestep_distinctiveness_dinov3(
    policy: E6Stage1DINOv3Policy,
    bank: DINOv3SelfPairFeatureBank,
    *,
    batch_size: int = 128,
    max_pairs: Optional[int] = 512,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Same-Demo distinct-timestep latent/action distinctiveness & temporal cross-attention shift.

    Spatial cross-attention weights `(L, B, 4, P * K_m)` are first summed across the `K_m`
    spatial tokens per timestep -> `(L, B, 4, P)` before computing `layer0_attn_peak_shift_steps`,
    making the metric strictly comparable to E6-CLIP Stage 1.
    """
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    pairs = bank.val_timestep_pairs
    n_pairs = len(pairs) if max_pairs is None else min(int(max_pairs), len(pairs))
    if n_pairs == 0:
        return {"num_pairs": 0}

    sel_pairs = pairs[:n_pairs]
    local_pair_idx = np.arange(n_pairs)[rank::world]

    gen = torch.Generator(device="cpu").manual_seed(val_seed + 2026)
    all_z0 = torch.randn(
        (n_pairs, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32
    )
    local_z0 = all_z0[local_pair_idx]

    # [count, frame_gap_sum, z_cos_sum, z_l2_sum, z_rel_l2_sum, attn_peak_shift_sum]
    stat_acc = torch.zeros(6, device=device, dtype=torch.float64)
    pred_pair_err = torch.zeros(11, device=device, dtype=torch.float64)
    gt_pair_err = torch.zeros(11, device=device, dtype=torch.float64)
    matched_err = torch.zeros(11, device=device, dtype=torch.float64)
    mismatched_err = torch.zeros(11, device=device, dtype=torch.float64)

    for start in range(0, len(local_pair_idx), batch_size):
        end = min(start + batch_size, len(local_pair_idx))
        p_chunk = sel_pairs[local_pair_idx[start:end]]
        ia = p_chunk[:, 0]
        ib = p_chunk[:, 1]
        z0 = local_z0[start:end].to(device, non_blocking=True)

        eps_a = bank.val_episodes[ia]
        eps_b = bank.val_episodes[ib]
        if not np.array_equal(eps_a, eps_b):
            raise RuntimeError("Timestep distinctiveness pair spans different episodes")
        fa = bank.val_frames[ia]
        fb = bank.val_frames[ib]
        d_a, m_a, q_a, act_a = bank.gather_val_batch_by_indices(ia, mode="self")
        d_b, m_b, q_b, act_b = bank.gather_val_batch_by_indices(ib, mode="self")

        with amp_ctx:
            out_a = policy.predict_action(
                d_a,
                m_a,
                q_a,
                init_noise=z0,
                need_weights=True,
                average_attn_weights=True,
                aggregate_attn_temporal=True,
            )
            out_b = policy.predict_action(
                d_b,
                m_b,
                q_b,
                init_noise=z0,
                need_weights=True,
                average_attn_weights=True,
                aggregate_attn_temporal=True,
            )

        za = out_a["z_self"].float().reshape(len(ia), -1)
        zb = out_b["z_self"].float().reshape(len(ib), -1)
        z_cos = torch.nn.functional.cosine_similarity(za, zb, dim=-1)
        z_l2 = (za - zb).norm(dim=-1)
        z_rms_pair = 0.5 * (za.norm(dim=-1) + zb.norm(dim=-1)).clamp_min(1e-6)
        z_rel_l2 = z_l2 / z_rms_pair

        # Temporally aggregated Layer-0 cross-attention: (L, B, 4, P) -> mean over 4 queries -> (B, P)
        attn_a_l0 = out_a["cross_attn_weights"][0].float().mean(dim=1)
        attn_b_l0 = out_b["cross_attn_weights"][0].float().mean(dim=1)
        peak_a = attn_a_l0.argmax(dim=-1).float()
        peak_b = attn_b_l0.argmax(dim=-1).float()
        peak_shift = (peak_b - peak_a).abs()

        stat_acc[0] += float(len(ia))
        stat_acc[1] += float(np.abs(fb - fa).sum())
        stat_acc[2] += z_cos.double().sum()
        stat_acc[3] += z_l2.double().sum()
        stat_acc[4] += z_rel_l2.double().sum()
        stat_acc[5] += peak_shift.double().sum()

        pred_pair_err += _action_error_sums(out_a["action_10d"], out_b["action_10d"])
        gt_pair_err += _action_error_sums(act_a, act_b)
        matched_err += _action_error_sums(out_a["action_10d"], act_a)
        matched_err += _action_error_sums(out_b["action_10d"], act_b)
        mismatched_err += _action_error_sums(out_a["action_10d"], act_b)
        mismatched_err += _action_error_sums(out_b["action_10d"], act_a)

    stat_acc = _all_reduce_sum(stat_acc)
    pred_pair = _unpack_action_errors(_all_reduce_sum(pred_pair_err))
    gt_pair = _unpack_action_errors(_all_reduce_sum(gt_pair_err))
    matched = _unpack_action_errors(_all_reduce_sum(matched_err))
    mismatched = _unpack_action_errors(_all_reduce_sum(mismatched_err))

    n_tot = max(float(stat_acc[0].item()), 1.0)
    return {
        "num_pairs": int(round(n_tot)),
        "mean_frame_gap": float(stat_acc[1].item() / n_tot),
        "latent_cosine_sim": float(stat_acc[2].item() / n_tot),
        "latent_l2_dist": float(stat_acc[3].item() / n_tot),
        "latent_rel_l2_dist": float(stat_acc[4].item() / n_tot),
        "layer0_attn_peak_shift_steps": float(stat_acc[5].item() / n_tot),
        "pred_pair_action10_mse": pred_pair["action10_mse"],
        "pred_pair_action7_mae": pred_pair["action7_mae"],
        "gt_pair_action10_mse": gt_pair["action10_mse"],
        "gt_pair_action7_mae": gt_pair["action7_mae"],
        "matched_action10_mse": matched["action10_mse"],
        "matched_action7_mae": matched["action7_mae"],
        "mismatched_same_demo_action10_mse": mismatched["action10_mse"],
        "mismatched_same_demo_action7_mae": mismatched["action7_mae"],
        "mismatch_to_match_mse10_ratio": (
            mismatched["action10_mse"] / max(matched["action10_mse"], 1e-12)
        ),
    }


@torch.inference_mode()
def evaluate_stage1_dinov3_validation(
    policy: E6Stage1DINOv3Policy,
    bank: DINOv3SelfPairFeatureBank,
    *,
    batch_size: int = 128,
    max_val_samples: Optional[int] = None,
    run_ddim: bool = False,
    ddim_max_samples: Optional[int] = 512,
    run_timestep_check: bool = False,
    timestep_max_pairs: Optional[int] = 512,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Exact mirror of CLIP Stage 1 validation protocol."""
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    n_val = (
        bank.num_val_samples()
        if max_val_samples is None
        else min(int(max_val_samples), bank.num_val_samples())
    )
    gen = torch.Generator(device="cpu").manual_seed(val_seed)
    all_noise = torch.randn(
        (n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32
    )
    all_pertub = torch.randn(
        (n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen, dtype=torch.float32
    )
    all_t = torch.randint(
        0,
        policy.noise_scheduler.config.num_train_timesteps,
        (n_val,),
        generator=gen,
        dtype=torch.long,
    )

    local_idx = np.arange(n_val)[rank::world]
    local_noise = all_noise[local_idx]
    local_pertub = all_pertub[local_idx]
    local_t = all_t[local_idx]

    acc = torch.zeros(5, device=device, dtype=torch.float64)
    self_iter = bank.iter_val_batches(
        "self", batch_size, rank=rank, world_size=world, max_samples=n_val
    )
    wrong_iter = bank.iter_val_batches(
        "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_val
    )

    cursor = 0
    for (d_s, m_s, q_s, a_s), (d_w, m_w, q_w, a_w) in zip(self_iter, wrong_iter):
        B = len(d_s)
        noise = local_noise[cursor : cursor + B].to(device, non_blocking=True)
        pertub = local_pertub[cursor : cursor + B].to(device, non_blocking=True)
        t = local_t[cursor : cursor + B].to(device, non_blocking=True)
        cursor += B

        with amp_ctx:
            z_s, _ = policy.encode_z_self(d_s, m_s, q_s)
            loss_s = policy.diffusion_loss_from_z_self(
                z_s, a_s, noise=noise, pertub_noise=pertub, timesteps=t
            )
            z_w, _ = policy.encode_z_self(d_w, m_w, q_w)
            loss_w = policy.diffusion_loss_from_z_self(
                z_w, a_w, noise=noise, pertub_noise=pertub, timesteps=t
            )

        acc[0] += float(B)
        acc[1] += loss_s.double().sum()
        acc[2] += loss_w.double().sum()
        acc[3] += z_s.float().square().mean(dim=(1, 2)).double().sum()
        acc[4] += z_w.float().square().mean(dim=(1, 2)).double().sum()

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)
    self_diff = float(acc[1].item() / total)
    wrong_diff = float(acc[2].item() / total)

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "val_self_diff_loss": self_diff,
        "val_wrong_demo_diff_loss": wrong_diff,
        "val_demo_sensitivity_diff_loss": wrong_diff - self_diff,
        "val_self_z_self_rms": math.sqrt(float(acc[3].item() / total)),
        "val_wrong_z_self_rms": math.sqrt(float(acc[4].item() / total)),
    }

    if run_ddim:
        n_ddim = n_val if ddim_max_samples is None else min(int(ddim_max_samples), n_val)
        gen_ddim = torch.Generator(device="cpu").manual_seed(val_seed + 777)
        init_noise_all = torch.randn(
            (n_ddim, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_ddim, dtype=torch.float32
        )
        local_ddim_noise = init_noise_all[np.arange(n_ddim)[rank::world]]

        self_err = torch.zeros(11, device=device, dtype=torch.float64)
        wrong_err = torch.zeros(11, device=device, dtype=torch.float64)

        s_it = bank.iter_val_batches(
            "self", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        w_it = bank.iter_val_batches(
            "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        cur = 0
        for (d_s, m_s, q_s, a_s), (d_w, m_w, q_w, a_w) in zip(s_it, w_it):
            B = len(d_s)
            z0 = local_ddim_noise[cur : cur + B].to(device, non_blocking=True)
            cur += B
            with amp_ctx:
                out_s = policy.predict_action(d_s, m_s, q_s, init_noise=z0)
                out_w = policy.predict_action(d_w, m_w, q_w, init_noise=z0)
            self_err += _action_error_sums(out_s["action_10d"], a_s)
            wrong_err += _action_error_sums(out_w["action_10d"], a_w)

        self_ddim = _unpack_action_errors(_all_reduce_sum(self_err))
        wrong_ddim = _unpack_action_errors(_all_reduce_sum(wrong_err))
        result["ddim"] = {
            "self": self_ddim,
            "wrong_demo": wrong_ddim,
            "demo_sensitivity_action10_mse": (
                wrong_ddim["action10_mse"] - self_ddim["action10_mse"]
            ),
            "demo_sensitivity_action7_mae": (
                wrong_ddim["action7_mae"] - self_ddim["action7_mae"]
            ),
        }

    if run_timestep_check:
        result["timestep_distinctiveness"] = evaluate_timestep_distinctiveness_dinov3(
            policy,
            bank,
            batch_size=batch_size,
            max_pairs=timestep_max_pairs,
            val_seed=val_seed,
            use_bf16=use_bf16,
        )

    return result


def build_clip_vs_dinov3_comparison(
    dinov3_final_eval: Dict[str, Any],
    clip_final_eval_path: Path = DEFAULT_CLIP_FINAL_EVAL,
) -> Optional[Dict[str, Any]]:
    """Build structured & Markdown side-by-side comparison against E6-CLIP Stage 1."""
    clip_final_eval_path = Path(clip_final_eval_path)
    if not clip_final_eval_path.is_file():
        return None

    clip_payload = json.loads(clip_final_eval_path.read_text())
    clip_ev = clip_payload["final_eval"]
    dino_ev = dinov3_final_eval

    def _get(d: Dict[str, Any], path: List[str]) -> Optional[float]:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return float(cur) if cur is not None else None

    metric_specs = [
        ("Val Self Diffusion Loss", ["val_self_diff_loss"], "lower"),
        ("Val Wrong-Demo Diffusion Loss", ["val_wrong_demo_diff_loss"], "higher"),
        ("Demo Sensitivity (Diff Loss)", ["val_demo_sensitivity_diff_loss"], "higher"),
        ("DDIM Self Action10 MSE", ["ddim", "self", "action10_mse"], "lower"),
        ("DDIM Self Action10 MAE", ["ddim", "self", "action10_mae"], "lower"),
        ("DDIM Self Action7 MSE", ["ddim", "self", "action7_mse"], "lower"),
        ("DDIM Self Action7 MAE", ["ddim", "self", "action7_mae"], "lower"),
        ("DDIM Self Exec12 Action7 MSE", ["ddim", "self", "exec12_action7_mse"], "lower"),
        ("DDIM Self Exec12 Action7 MAE", ["ddim", "self", "exec12_action7_mae"], "lower"),
        ("DDIM Self Gripper Sign Acc", ["ddim", "self", "grip_sign_acc"], "higher"),
        ("DDIM Wrong-Demo Action10 MSE", ["ddim", "wrong_demo", "action10_mse"], "higher"),
        ("DDIM Demo Sensitivity Action10 MSE", ["ddim", "demo_sensitivity_action10_mse"], "higher"),
        ("DDIM Demo Sensitivity Action7 MAE", ["ddim", "demo_sensitivity_action7_mae"], "higher"),
        ("Timestep Latent Cosine Sim", ["timestep_distinctiveness", "latent_cosine_sim"], "lower"),
        ("Timestep Latent Rel-L2 Dist", ["timestep_distinctiveness", "latent_rel_l2_dist"], "higher"),
        ("Matched Same-Demo Action10 MSE", ["timestep_distinctiveness", "matched_action10_mse"], "lower"),
        ("Mismatched Same-Demo Action10 MSE", ["timestep_distinctiveness", "mismatched_same_demo_action10_mse"], "higher"),
        ("Mismatch / Match MSE10 Ratio", ["timestep_distinctiveness", "mismatch_to_match_mse10_ratio"], "higher"),
        ("Layer-0 Attn Peak Shift (1Hz steps)", ["timestep_distinctiveness", "layer0_attn_peak_shift_steps"], "higher"),
    ]

    rows = []
    md_lines = [
        "# E6 Stage 1 Comparison: CLIP vs DINOv3 (Spatial 4x4)",
        "",
        f"- CLIP baseline: `{clip_final_eval_path}` (val_samples={clip_ev.get('val_samples')})",
        f"- DINOv3 run: val_samples={dino_ev.get('val_samples')}",
        "",
        "| Metric | Better | E6-CLIP Stage 1 | E6-DINOv3 Stage 1 | Delta (DINOv3 - CLIP) |",
        "| --- | :---: | ---: | ---: | ---: |",
    ]
    for label, key_path, direction in metric_specs:
        c_val = _get(clip_ev, key_path)
        d_val = _get(dino_ev, key_path)
        delta = (d_val - c_val) if (c_val is not None and d_val is not None) else None
        rows.append(
            {
                "metric": label,
                "better": direction,
                "clip": c_val,
                "dinov3": d_val,
                "delta": delta,
            }
        )
        c_s = f"{c_val:.6f}" if c_val is not None else "N/A"
        d_s = f"{d_val:.6f}" if d_val is not None else "N/A"
        dl_s = f"{delta:+.6f}" if delta is not None else "N/A"
        md_lines.append(f"| {label} | {direction} | {c_s} | {d_s} | {dl_s} |")

    return {
        "clip_reference_path": str(clip_final_eval_path),
        "rows": rows,
        "markdown": "\n".join(md_lines) + "\n",
    }


def run_training(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if use_ddp:
        dist.init_process_group(backend="nccl", device_id=device)

    try:
        rank = _rank()
        world = _world()
        if args.global_batch % world != 0:
            raise ValueError(
                f"global_batch ({args.global_batch}) must be divisible by world_size ({world})"
            )
        per_rank_batch = args.global_batch // world

        # Smoke preset overrides for rapid end-to-end verification
        if args.smoke:
            if args.run_name is None:
                args.run_name = "smoke_dinov3"
            if args.epochs == 8:
                args.epochs = 2
            if args.max_train_samples is None:
                args.max_train_samples = args.global_batch * 8
            if args.max_val_samples is None:
                args.max_val_samples = 256
            if args.sample_count == 512:
                args.sample_count = 128
            if args.timestep_pairs == 512:
                args.timestep_pairs = 64
            args.final_ddim_samples = args.sample_count
            args.final_timestep_pairs = args.timestep_pairs
            args.sample_every = 1
            args.save_last_every = 1

        run_name = args.run_name if args.run_name else "stage1_self_latent_dinov3"
        run_dir = args.output_dir / run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        # 1. Pre-flight DINOv3 cache check & CLIP vs DINOv3 feature distribution audit
        cache_audit = verify_dinov3_cache_compatibility(
            dinov3_cache_dir=args.cache_dir,
            clip_cache_dir=args.clip_cache_dir,
            feature_mode=args.feature_mode,
        )
        if args.check_only:
            if rank == 0:
                print(json.dumps(cache_audit, indent=2), flush=True)
            return

        seed_everything(args.seed)
        bank = DINOv3SelfPairFeatureBank(
            cache_dir=args.cache_dir,
            feature_mode=args.feature_mode,
            device=device,
            demo_in_ram=not args.demo_mmap,
            demo_on_device=args.demo_on_device,
            preload_val_query=not args.no_preload_val_query,
            superchunk_episodes=args.superchunk_episodes,
            seed=args.seed,
        )

        # Eval-only mode: reload best.pt and run full 6,248-sample evaluation + CLIP comparison
        if args.eval_only:
            ckpt_path = Path(args.eval_ckpt) if args.eval_ckpt else (run_dir / "best.pt")
            teacher = load_frozen_stage1_dinov3_teacher(ckpt_path, device=device)
            final_eval = evaluate_stage1_dinov3_validation(
                teacher,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=True,
                ddim_max_samples=args.final_ddim_samples,
                run_timestep_check=True,
                timestep_max_pairs=args.final_timestep_pairs,
                val_seed=args.val_seed,
                use_bf16=(device.type == "cuda"),
            )
            if rank == 0:
                comp = build_clip_vs_dinov3_comparison(final_eval, args.clip_final_eval)
                if comp is not None:
                    atomic_write_json(comp, run_dir / "clip_vs_dinov3_comparison.json")
                    (run_dir / "clip_vs_dinov3_comparison.md").write_text(comp["markdown"])
                    print("\n" + comp["markdown"], flush=True)
                print(json.dumps(final_eval, indent=2), flush=True)
            return

        seed_everything(args.seed)
        policy = E6Stage1DINOv3Policy(
            feature_mode=args.feature_mode,
            demo_tokens_per_step=args.demo_tokens_per_step,
            action_normalizer=bank.action_normalizer,
            use_attention_sink=args.use_attention_sink,
        )
        compat_report: Dict[str, Any] = {"mode": "from_scratch", "init_mode": "none"}
        if (
            args.init_mode != "none"
            and args.init_ckpt is not None
            and str(args.init_ckpt).lower() != "none"
        ):
            compat_report = policy.load_compatible_from_checkpoint(
                Path(args.init_ckpt), init_mode=args.init_mode
            )
        policy = policy.to(device)

        lr = args.lr if args.lr is not None else (args.base_lr_per_gpu * world)
        optimizer = torch.optim.AdamW(
            policy.get_optim_groups(lr=lr, weight_decay=args.weight_decay),
            lr=lr,
            betas=(0.95, 0.999),
            weight_decay=args.weight_decay,
        )

        epoch_len = (
            min(args.max_train_samples, bank.num_train_samples())
            if args.max_train_samples is not None
            else bank.num_train_samples()
        )
        steps_per_epoch = math.ceil(epoch_len / args.global_batch)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = max(1, int(args.lr_warmup_proportion * total_steps))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        ema = EMAModel(
            policy.parameters(),
            decay=args.ema_decay,
            min_decay=0.0,
            update_after_step=0,
            use_ema_warmup=False,
            inv_gamma=1.0,
            power=0.75,
            foreach=True,
        ).to(device)

        config = {
            "stage": "e6_stage1_self_latent_dinov3",
            "backbone_id": DINOV3_BACKBONE_ID,
            "feature_mode": args.feature_mode,
            "num_spatial_tokens": policy.num_spatial_tokens,
            "demo_tokens_per_step": args.demo_tokens_per_step,
            "run_name": run_name,
            "smoke": bool(args.smoke),
            "self_pair_only": True,
            "rgb_only": True,
            "use_residual": False,
            "use_aux_head": False,
            "use_attention_sink": bool(args.use_attention_sink),
            "global_cond_dim": policy.global_cond_dim,
            "samples_per_epoch": epoch_len,
            "total_cached_train_starts": bank.num_train_samples(),
            "total_cached_val_starts": bank.num_val_samples(),
            "val_timestep_pairs_available": int(len(bank.val_timestep_pairs)),
            "superchunk_episodes": args.superchunk_episodes,
            "seed": args.seed,
            "epochs": args.epochs,
            "world_size": world,
            "global_batch": args.global_batch,
            "per_rank_batch": per_rank_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "betas": [0.95, 0.999],
            "lr_scheduler": "cosine",
            "lr_warmup_proportion": args.lr_warmup_proportion,
            "warmup_steps": warmup_steps,
            "clip_grad_norm": args.clip_grad_norm,
            "mixed_precision": "bf16" if device.type == "cuda" else "fp32",
            "use_ema": True,
            "ema_decay": args.ema_decay,
            "init_mode": args.init_mode,
            "init_ckpt": str(args.init_ckpt),
            "cache_dir": str(args.cache_dir),
            "cache_audit": cache_audit,
            "cache_meta": bank.meta,
            "parameter_summary": policy.parameter_summary(),
            "weight_compat_report": compat_report,
        }

        last_ckpt = run_dir / "last.pt"
        best_ckpt = run_dir / "best.pt"
        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        start_epoch = 1
        global_step = 0

        if args.resume and last_ckpt.is_file():
            ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            policy.load_state_dict(ck["model"], strict=True)
            ema.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["optimizer"])
            lr_scheduler.load_state_dict(ck["lr_scheduler"])
            history = ck["history"]
            best_val_loss = ck["best_val_loss"]
            best_epoch = ck["best_epoch"]
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
            if rank == 0:
                print(
                    f"[E6-DINOv3 Stage1] Resumed from epoch {ck['epoch']} "
                    f"(best_epoch={best_epoch}, best_val_self_diff={best_val_loss:.5f})",
                    flush=True,
                )

        ddp_model: torch.nn.Module = (
            DistributedDataParallel(policy, device_ids=[device.index])
            if use_ddp
            else policy
        )
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type=device.type, enabled=False)
        )

        if rank == 0:
            atomic_write_json(config, run_dir / "config.json")
            ps = config["parameter_summary"]
            print(
                f"[E6-DINOv3 Stage1] run={run_name} | mode={args.feature_mode} "
                f"(K_m={args.demo_tokens_per_step}) | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={warmup_steps}) | lr={lr:.2e} | "
                f"global_cond_dim={policy.global_cond_dim} | trainable={ps['policy_trainable']:,} | "
                f"init_mode={args.init_mode} loaded={compat_report.get('loaded_compatible_params', 0):,} "
                f"({100 * compat_report.get('loaded_fraction', 0.0):.1f}%)",
                flush=True,
            )

        # Epoch-0 reference validation before training updates
        init_val = None
        if start_epoch == 1 and not args.skip_init_val:
            init_val = evaluate_stage1_dinov3_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=True,
                ddim_max_samples=args.sample_count,
                run_timestep_check=True,
                timestep_max_pairs=args.timestep_pairs,
                val_seed=args.val_seed,
                use_bf16=(device.type == "cuda"),
            )
            if rank == 0:
                atomic_write_json(init_val, run_dir / "init_val.json")
                td = init_val["timestep_distinctiveness"]
                print(
                    f"[E6-DINOv3 Stage1 Init Val] self_diff={init_val['val_self_diff_loss']:.4f} | "
                    f"wrong_diff={init_val['val_wrong_demo_diff_loss']:.4f} | "
                    f"ddim_self_mse10={init_val['ddim']['self']['action10_mse']:.4f} | "
                    f"ddim_self_mae7={init_val['ddim']['self']['action7_mae']:.4f} | "
                    f"ts_mismatch_ratio={td['mismatch_to_match_mse10_ratio']:.2f}x",
                    flush=True,
                )

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            tr_acc = torch.zeros(5, device=device, dtype=torch.float64)

            batch_feat_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            for demo_feats, pad_mask, query_feats, actions_10d in bank.iter_train_batches(
                epoch,
                per_rank_batch,
                rank=rank,
                world_size=world,
                max_samples=args.max_train_samples,
                out_dtype=batch_feat_dtype,
            ):
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(demo_feats, pad_mask, query_feats, actions_10d)
                out["loss"].backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(), args.clip_grad_norm
                    ).item()
                )
                optimizer.step()
                lr_scheduler.step()
                ema.step(policy.parameters())
                global_step += 1

                B = float(len(demo_feats))
                tr_acc[0] += B
                tr_acc[1] += float(out["diff_loss"].item()) * B
                tr_acc[2] += (float(out["z_self_rms"].item()) ** 2) * B
                tr_acc[3] += grad_norm
                tr_acc[4] += 1.0

            tr_acc = _all_reduce_sum(tr_acc)
            t_train = time.monotonic() - t_ep

            n_tot = max(float(tr_acc[0].item()), 1.0)
            n_steps_tot = max(float(tr_acc[4].item()), 1.0)
            train_metrics = {
                "samples": int(round(n_tot)),
                "diff_loss": float(tr_acc[1].item() / n_tot),
                "z_self_rms": math.sqrt(float(tr_acc[2].item() / n_tot)),
                "grad_norm_mean": float(tr_acc[3].item() / n_steps_tot),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "ema_decay": float(ema.cur_decay_value or 0.0),
                "train_seconds": round(t_train, 2),
                "steps_per_sec": round((n_steps_tot / world) / max(t_train, 1e-6), 2),
            }

            # Evaluate EMA weights
            t_val0 = time.monotonic()
            ema.store(policy.parameters())
            ema.copy_to(policy.parameters())
            do_ddim = (
                epoch == 1
                or epoch == args.epochs
                or (args.sample_every > 0 and epoch % args.sample_every == 0)
            )
            val_metrics = evaluate_stage1_dinov3_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                run_ddim=do_ddim,
                ddim_max_samples=args.sample_count,
                run_timestep_check=do_ddim,
                timestep_max_pairs=args.timestep_pairs,
                val_seed=args.val_seed,
                use_bf16=(device.type == "cuda"),
            )
            val_metrics["val_seconds"] = round(time.monotonic() - t_val0, 2)

            cur_monitor = val_metrics["val_self_diff_loss"]
            is_best = cur_monitor < best_val_loss
            if is_best:
                best_val_loss = cur_monitor
                best_epoch = epoch

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_self_diff_loss": best_val_loss,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "stage": "e6_stage1_self_latent_dinov3",
                        "backbone_id": DINOV3_BACKBONE_ID,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_self_diff_loss": best_val_loss,
                        "model": {
                            k: v.detach().cpu() for k, v in policy.state_dict().items()
                        },
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    best_ckpt,
                )

            ema.restore(policy.parameters())

            if rank == 0:
                if epoch % args.save_last_every == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        {
                            "stage": "e6_stage1_self_latent_dinov3",
                            "backbone_id": DINOV3_BACKBONE_ID,
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "model": {
                                k: v.detach().cpu()
                                for k, v in policy.state_dict().items()
                            },
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "history": history,
                            "config": config,
                        },
                        last_ckpt,
                    )
                atomic_write_json(
                    {
                        "stage": "e6_stage1_self_latent_dinov3",
                        "run_name": run_name,
                        "best_epoch": best_epoch,
                        "best_val_self_diff_loss": best_val_loss,
                        "init_val": init_val,
                        "history": history,
                    },
                    run_dir / "metrics.json",
                )
                extra_str = ""
                if "ddim" in val_metrics:
                    dm = val_metrics["ddim"]
                    td = val_metrics["timestep_distinctiveness"]
                    extra_str = (
                        f" | DDIM self mse10={dm['self']['action10_mse']:.4f} "
                        f"mae7={dm['self']['action7_mae']:.4f} "
                        f"grip={dm['self']['grip_sign_acc']:.4f} "
                        f"(wrong mse10={dm['wrong_demo']['action10_mse']:.4f}) | "
                        f"TS z_cos={td['latent_cosine_sim']:.3f} "
                        f"peak_shift={td['layer0_attn_peak_shift_steps']:.2f} "
                        f"mismatch/match={td['mismatch_to_match_mse10_ratio']:.2f}x"
                    )
                print(
                    f"[E6-DINOv3 Stage1 Ep {epoch:02d}/{args.epochs}] "
                    f"tr_diff={train_metrics['diff_loss']:.4f} | "
                    f"val_self={val_metrics['val_self_diff_loss']:.4f} "
                    f"(wrong={val_metrics['val_wrong_demo_diff_loss']:.4f}, "
                    f"sens={val_metrics['val_demo_sensitivity_diff_loss']:+.4f}) | "
                    f"z_rms={val_metrics['val_self_z_self_rms']:.3f} | "
                    f"lr={train_metrics['lr']:.2e} "
                    f"t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s "
                    f"({train_metrics['steps_per_sec']:.1f} it/s)"
                    f"{'*' if is_best else ''}{extra_str}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Final reloaded best.pt full validation (all 6,248 val samples) + frozen Teacher check
        best_payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        teacher = load_frozen_stage1_dinov3_teacher(best_ckpt, device=device)
        final_eval = evaluate_stage1_dinov3_validation(
            teacher,
            bank,
            batch_size=args.val_batch_size,
            max_val_samples=args.max_val_samples,
            run_ddim=True,
            ddim_max_samples=args.final_ddim_samples,
            run_timestep_check=True,
            timestep_max_pairs=args.final_timestep_pairs,
            val_seed=args.val_seed,
            use_bf16=(device.type == "cuda"),
        )
        if rank == 0:
            comparison = build_clip_vs_dinov3_comparison(final_eval, args.clip_final_eval)
            final_report = {
                "stage": "e6_stage1_self_latent_dinov3",
                "backbone_id": DINOV3_BACKBONE_ID,
                "feature_mode": args.feature_mode,
                "demo_tokens_per_step": args.demo_tokens_per_step,
                "run_name": run_name,
                "best_epoch": best_payload["epoch"],
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_self_diff_loss": best_payload["best_val_self_diff_loss"],
                "frozen_teacher_trainable_params": sum(
                    p.numel() for p in teacher.parameters() if p.requires_grad
                ),
                "parameter_summary": config["parameter_summary"],
                "weight_compat_report": compat_report,
                "cache_audit": cache_audit,
                "final_eval": final_eval,
                "clip_vs_dinov3_comparison": comparison["rows"] if comparison else None,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            if comparison is not None:
                atomic_write_json(comparison, run_dir / "clip_vs_dinov3_comparison.json")
                (run_dir / "clip_vs_dinov3_comparison.md").write_text(comparison["markdown"])
                print("\n" + comparison["markdown"], flush=True)
            print(
                f"\n[E6-DINOv3 Stage1 FINAL BEST EVAL (ep={best_payload['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DINOV3_CACHE_DIR)
    parser.add_argument("--clip-cache-dir", type=Path, default=DEFAULT_CLIP_CACHE_DIR)
    parser.add_argument("--clip-final-eval", type=Path, default=DEFAULT_CLIP_FINAL_EVAL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "stage1")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--feature-mode", choices=("spatial", "global"), default="spatial")
    parser.add_argument(
        "--demo-tokens-per-step",
        type=int,
        default=DEFAULT_DEMO_TOKENS_PER_STEP,
        help="Spatial memory tokens retained per 1Hz Demo step by PromptPooler.",
    )
    parser.add_argument(
        "--init-mode",
        choices=("action_decoder_only", "all_compatible", "none"),
        default="action_decoder_only",
        help="Weight reuse strategy from CLIP Stage 1 checkpoint (checks distribution shift).",
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default=str(DEFAULT_CLIP_STAGE1_CKPT),
        help="Path to CLIP Stage 1 best.pt for compatible initialization (or 'none').",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--superchunk-episodes", type=int, default=512)
    parser.add_argument("--demo-mmap", action="store_true", help="Keep demo_spatial_features mmap'd instead of RAM.")
    parser.add_argument("--demo-on-device", action="store_true", help="Load demo_spatial_features directly onto GPU.")
    parser.add_argument("--no-preload-val-query", action="store_true", help="Skip preloading val query frames into RAM.")
    parser.add_argument("--base-lr-per-gpu", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--use-attention-sink", action="store_true")
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--timestep-pairs", type=int, default=512)
    parser.add_argument("--final-ddim-samples", type=int, default=None)
    parser.add_argument("--final-timestep-pairs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--save-last-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-ckpt", type=str, default=None)
    parser.add_argument("--skip-init-val", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
