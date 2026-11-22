"""Direct Joint Training, small-sample overfit check, and full evaluation for E5
and the matched BPP RGB-only baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from . import ACTION_DIM_10D, ACTION_HORIZON, BPP_CKPT, EXEC_HORIZON, OUTPUT_ROOT
from .actions import action_10d_to_7d
from .bpp_compat import load_bpp_pretrained_weights
from .cache import DEFAULT_CACHE_DIR
from .data import E5FeatureBank
from .diffusion import EMAModel, get_cosine_schedule_with_warmup
from .model import E5Policy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, delete=False, mode="w", suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(obj, tmp, indent=2)
        tmp.write("\n")
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _all_reduce_sum(vec: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    return vec


def _action_error_sums(
    pred_10d: torch.Tensor, gt_10d: torch.Tensor
) -> torch.Tensor:
    """Compute unweighted sum statistics across batch for exact DDP reduction."""
    pred_10d = pred_10d.float()
    gt_10d = gt_10d.float()
    B = float(pred_10d.shape[0])

    diff10 = pred_10d - gt_10d
    mse10 = diff10.square().mean(dim=(1, 2)).sum()
    mae10 = diff10.abs().mean(dim=(1, 2)).sum()
    pos_mse = diff10[..., :3].square().mean(dim=(1, 2)).sum()
    rot6d_mse = diff10[..., 3:9].square().mean(dim=(1, 2)).sum()
    grip_mse = diff10[..., 9:10].square().mean(dim=(1, 2)).sum()

    pred_7d = action_10d_to_7d(pred_10d)
    gt_7d = action_10d_to_7d(gt_10d)
    diff7 = pred_7d - gt_7d
    mse7 = diff7.square().mean(dim=(1, 2)).sum()
    mae7 = diff7.abs().mean(dim=(1, 2)).sum()

    diff7_exec = diff7[:, :EXEC_HORIZON, :]
    exec_mse7 = diff7_exec.square().mean(dim=(1, 2)).sum()
    exec_mae7 = diff7_exec.abs().mean(dim=(1, 2)).sum()

    grip_sign = ((pred_7d[..., 6] >= 0) == (gt_7d[..., 6] >= 0)).float().mean(dim=1).sum()

    return torch.stack(
        [
            torch.tensor(B, device=pred_10d.device, dtype=torch.float64),
            mse10.double(),
            mae10.double(),
            pos_mse.double(),
            rot6d_mse.double(),
            grip_mse.double(),
            mse7.double(),
            mae7.double(),
            exec_mse7.double(),
            exec_mae7.double(),
            grip_sign.double(),
        ]
    )


def _unpack_action_errors(sums: torch.Tensor) -> Dict[str, float]:
    n = max(float(sums[0].item()), 1.0)
    return {
        "count": int(round(float(sums[0].item()))),
        "action10_mse": float(sums[1].item() / n),
        "action10_mae": float(sums[2].item() / n),
        "pos_mse": float(sums[3].item() / n),
        "rot6d_mse": float(sums[4].item() / n),
        "grip_mse": float(sums[5].item() / n),
        "action7_mse": float(sums[6].item() / n),
        "action7_mae": float(sums[7].item() / n),
        "exec12_action7_mse": float(sums[8].item() / n),
        "exec12_action7_mae": float(sums[9].item() / n),
        "grip_sign_acc": float(sums[10].item() / n),
    }


@torch.inference_mode()
def evaluate_validation(
    policy: E5Policy,
    bank: E5FeatureBank,
    *,
    batch_size: int = 128,
    run_ddim: bool = False,
    ddim_max_samples: Optional[int] = 512,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Comprehensive distributed validation:
    1. Fixed-noise Diffusion Loss + Residual RMS on ALL 6,180 val pairs for:
       - cross (with residual & ref-only)
       - self (ref-only & with residual)
       - wrong_demo (correct vs wrong demo sensitivity)
    2. Optional 16-step DDIM Action Denoising MSE/MAE (10D & 7D & exec-12) for all modes.
    """
    policy.eval()
    device = bank.device
    rank, world = _rank(), _world()
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Pre-generate deterministic diffusion noise & timesteps per global val sample
    # so cross, ref-only, self, and wrong_demo are compared on IDENTICAL noise!
    n_val = len(bank.val_keys)
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

    # Accumulators (float64 for exact reduction):
    # [count, cross_res_loss, cross_ref_loss, self_ref_loss, self_res_loss, wrong_loss,
    #  cross_delta_sq, cross_zref_sq, self_delta_sq, self_zref_sq, wrong_delta_sq]
    acc = torch.zeros(11, device=device, dtype=torch.float64)

    cross_iter = bank.iter_val_batches("cross", batch_size, rank=rank, world_size=world)
    self_iter = bank.iter_val_batches("self", batch_size, rank=rank, world_size=world)
    wrong_iter = bank.iter_val_batches("wrong_demo", batch_size, rank=rank, world_size=world)

    cursor = 0
    for (
        (d_c, m_c, q_c, a_c, _),
        (d_s, m_s, q_s, a_s, _),
        (d_w, m_w, q_w, a_w, _),
    ) in zip(cross_iter, self_iter, wrong_iter):
        B = len(d_c)
        noise = local_noise[cursor : cursor + B].to(device, non_blocking=True)
        pertub = local_pertub[cursor : cursor + B].to(device, non_blocking=True)
        t = local_t[cursor : cursor + B].to(device, non_blocking=True)
        cursor += B

        with amp_ctx:
            hq_c, zref_c, dz_c, zq_c, _ = policy.encode_conditioning(d_c, m_c, q_c)
            loss_cross_res = policy.diffusion_loss_from_cond(
                policy._build_global_cond(hq_c, zq_c),
                a_c,
                noise=noise,
                pertub_noise=pertub,
                timesteps=t,
            )
            loss_cross_ref = policy.diffusion_loss_from_cond(
                policy._build_global_cond(hq_c, zref_c),
                a_c,
                noise=noise,
                pertub_noise=pertub,
                timesteps=t,
            )

            hq_s, zref_s, dz_s, zq_s, _ = policy.encode_conditioning(d_s, m_s, q_s)
            loss_self_ref = policy.diffusion_loss_from_cond(
                policy._build_global_cond(hq_s, zref_s),
                a_s,
                noise=noise,
                pertub_noise=pertub,
                timesteps=t,
            )
            loss_self_res = policy.diffusion_loss_from_cond(
                policy._build_global_cond(hq_s, zq_s),
                a_s,
                noise=noise,
                pertub_noise=pertub,
                timesteps=t,
            )

            hq_w, zref_w, dz_w, zq_w, _ = policy.encode_conditioning(d_w, m_w, q_w)
            z_w_eval = zq_w if policy.use_residual else zref_w
            loss_wrong = policy.diffusion_loss_from_cond(
                policy._build_global_cond(hq_w, z_w_eval),
                a_w,
                noise=noise,
                pertub_noise=pertub,
                timesteps=t,
            )

        acc[0] += float(B)
        acc[1] += loss_cross_res.double().sum()
        acc[2] += loss_cross_ref.double().sum()
        acc[3] += loss_self_ref.double().sum()
        acc[4] += loss_self_res.double().sum()
        acc[5] += loss_wrong.double().sum()
        acc[6] += dz_c.float().square().mean(dim=(1, 2)).double().sum()
        acc[7] += zref_c.float().square().mean(dim=(1, 2)).double().sum()
        acc[8] += dz_s.float().square().mean(dim=(1, 2)).double().sum()
        acc[9] += zref_s.float().square().mean(dim=(1, 2)).double().sum()
        acc[10] += dz_w.float().square().mean(dim=(1, 2)).double().sum()

    acc = _all_reduce_sum(acc)
    total = max(float(acc[0].item()), 1.0)
    cross_delta_rms = math.sqrt(float(acc[6].item() / total))
    cross_zref_rms = math.sqrt(float(acc[7].item() / total))
    self_delta_rms = math.sqrt(float(acc[8].item() / total))
    self_zref_rms = math.sqrt(float(acc[9].item() / total))
    wrong_delta_rms = math.sqrt(float(acc[10].item() / total))

    cross_diff = float(acc[1].item() / total)
    cross_ref_diff = float(acc[2].item() / total)
    self_ref_diff = float(acc[3].item() / total)
    self_res_diff = float(acc[4].item() / total)
    wrong_diff = float(acc[5].item() / total)

    result: Dict[str, Any] = {
        "val_samples": int(round(total)),
        "val_cross_diff_loss": cross_diff,
        "val_cross_ref_only_diff_loss": cross_ref_diff,
        "val_cross_res_diff_gain": cross_ref_diff - cross_diff,
        "val_self_ref_only_diff_loss": self_ref_diff,
        "val_self_with_res_diff_loss": self_res_diff,
        "val_wrong_demo_diff_loss": wrong_diff,
        "val_demo_sensitivity_diff_loss": wrong_diff - cross_diff,
        "val_cross_delta_rms": cross_delta_rms,
        "val_cross_z_ref_rms": cross_zref_rms,
        "val_cross_delta_rel_rms": cross_delta_rms / max(cross_zref_rms, 1e-12),
        "val_self_delta_rms": self_delta_rms,
        "val_self_z_ref_rms": self_zref_rms,
        "val_self_delta_rel_rms": self_delta_rms / max(self_zref_rms, 1e-12),
        "val_wrong_delta_rms": wrong_delta_rms,
        "val_cross_to_self_rms_ratio": cross_delta_rms / max(self_delta_rms, 1e-12),
    }

    if run_ddim:
        n_ddim = n_val if ddim_max_samples is None else min(ddim_max_samples, n_val)
        gen_ddim = torch.Generator(device="cpu").manual_seed(val_seed + 777)
        init_noise_all = torch.randn(
            (n_ddim, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_ddim, dtype=torch.float32
        )
        local_ddim_noise = init_noise_all[np.arange(n_ddim)[rank::world]]

        ddim_modes = ["cross_with_res", "cross_ref_only", "self_ref_only", "wrong_demo"]
        if policy.use_residual:
            ddim_modes.append("self_with_res")
        err_sums = {
            m: torch.zeros(11, device=device, dtype=torch.float64) for m in ddim_modes
        }

        c_it = bank.iter_val_batches(
            "cross", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        s_it = bank.iter_val_batches(
            "self", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        w_it = bank.iter_val_batches(
            "wrong_demo", batch_size, rank=rank, world_size=world, max_samples=n_ddim
        )
        cur = 0
        for (
            (d_c, m_c, q_c, a_c, _),
            (d_s, m_s, q_s, a_s, _),
            (d_w, m_w, q_w, a_w, _),
        ) in zip(c_it, s_it, w_it):
            B = len(d_c)
            z0 = local_ddim_noise[cur : cur + B].to(device, non_blocking=True)
            cur += B
            with amp_ctx:
                out_c_res = policy.predict_action(
                    d_c, m_c, q_c, use_residual=True, init_noise=z0
                )
                err_sums["cross_with_res"] += _action_error_sums(
                    out_c_res["action_10d"], a_c
                )
                if policy.use_residual:
                    out_c_ref = policy.predict_action(
                        d_c, m_c, q_c, use_residual=False, init_noise=z0
                    )
                    err_sums["cross_ref_only"] += _action_error_sums(
                        out_c_ref["action_10d"], a_c
                    )
                else:
                    err_sums["cross_ref_only"] += _action_error_sums(
                        out_c_res["action_10d"], a_c
                    )

                out_s_ref = policy.predict_action(
                    d_s, m_s, q_s, use_residual=False, init_noise=z0
                )
                err_sums["self_ref_only"] += _action_error_sums(
                    out_s_ref["action_10d"], a_s
                )
                if policy.use_residual:
                    out_s_res = policy.predict_action(
                        d_s, m_s, q_s, use_residual=True, init_noise=z0
                    )
                    err_sums["self_with_res"] += _action_error_sums(
                        out_s_res["action_10d"], a_s
                    )

                out_w = policy.predict_action(
                    d_w, m_w, q_w, use_residual=policy.use_residual, init_noise=z0
                )
                err_sums["wrong_demo"] += _action_error_sums(out_w["action_10d"], a_w)

        ddim_metrics: Dict[str, Any] = {}
        for m in ddim_modes:
            ddim_metrics[m] = _unpack_action_errors(_all_reduce_sum(err_sums[m]))
        ddim_metrics["residual_gain_action10_mse"] = (
            ddim_metrics["cross_ref_only"]["action10_mse"]
            - ddim_metrics["cross_with_res"]["action10_mse"]
        )
        ddim_metrics["residual_gain_action7_mae"] = (
            ddim_metrics["cross_ref_only"]["action7_mae"]
            - ddim_metrics["cross_with_res"]["action7_mae"]
        )
        ddim_metrics["demo_sensitivity_action10_mse"] = (
            ddim_metrics["wrong_demo"]["action10_mse"]
            - ddim_metrics["cross_with_res"]["action10_mse"]
        )
        ddim_metrics["demo_sensitivity_action7_mae"] = (
            ddim_metrics["wrong_demo"]["action7_mae"]
            - ddim_metrics["cross_with_res"]["action7_mae"]
        )
        result["ddim"] = ddim_metrics

    return result


def build_policy_for_mode(
    mode: str,
    bank: E5FeatureBank,
    bundle: Dict[str, Any],
    seed: int,
    load_pretrained_reader: bool = True,
) -> E5Policy:
    """Build E5 or matched BPP RGB-only policy with IDENTICAL base initialization."""
    seed_everything(seed)
    # Build shared modules in identical seed order across modes.
    policy = E5Policy(
        use_residual=(mode == "e5"),
        action_normalizer=bank.action_normalizer,
        pretrained_bundle=bundle,
        load_pretrained_reader=load_pretrained_reader,
    )
    return policy


def run_overfit_test(args: argparse.Namespace) -> None:
    """Small-sample overfit sanity check on 64 real cached samples (32 self + 32 cross)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    out_dir = args.output_dir / "overfit_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = E5FeatureBank(
        cache_dir=args.cache_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        self_ratio=0.5,
        seed=args.seed,
    )
    bundle = load_bpp_pretrained_weights(ckpt_path=BPP_CKPT)
    policy = build_policy_for_mode(
        args.mode, bank, bundle, args.seed, load_pretrained_reader=not args.no_pretrained_reader
    ).to(device)

    # Grab one fixed 64-sample batch (32 self + 32 cross) from epoch 1.
    demos, queries, frames, q_rows, is_self = bank.build_train_epoch_schedule(1)
    self_idx = np.flatnonzero(is_self)[:32]
    cross_idx = np.flatnonzero(~is_self)[:32]
    sel = np.concatenate([self_idx, cross_idx])
    demo_rgb, pad_mask, query_rgb, actions_10d, is_self_t = bank.gather_batch(
        demos[sel], queries[sel], frames[sel], q_rows[sel], is_self[sel]
    )

    lr = args.lr if args.lr is not None else 2.0e-4
    optimizer = torch.optim.AdamW(
        policy.get_optim_groups(lr=lr, weight_decay=args.weight_decay),
        lr=lr,
        betas=(0.95, 0.999),
        weight_decay=args.weight_decay,
    )
    steps = args.overfit_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, steps // 20), num_training_steps=steps
    )
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Evaluate initial DDIM action error (step 0)
    policy.eval()
    gen0 = torch.Generator(device="cpu").manual_seed(999)
    z0 = torch.randn((len(sel), ACTION_HORIZON, ACTION_DIM_10D), generator=gen0).to(device)
    with amp_ctx:
        init_pred = policy.predict_action(demo_rgb, pad_mask, query_rgb, init_noise=z0)
        init_err = _unpack_action_errors(_action_error_sums(init_pred["action_10d"], actions_10d))

    print(
        f"[Overfit Check] mode={args.mode}, n={len(sel)} (self={int(is_self_t.sum())}, "
        f"cross={int((~is_self_t).sum())}), steps={steps}, lr={lr}",
        flush=True,
    )
    print(f"[Overfit Step 0] init_action10_mse={init_err['action10_mse']:.4f}, "
          f"init_action7_mae={init_err['action7_mae']:.4f}", flush=True)

    log_rows = []
    t0 = time.monotonic()
    policy.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            out = policy.compute_loss(
                demo_rgb,
                pad_mask,
                query_rgb,
                actions_10d,
                is_self_t,
                self_res_weight=args.self_res_weight,
                cross_res_weight=args.cross_res_weight,
            )
        out["loss"].backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad_norm).item()
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(out["loss"].item()),
                "diff_loss": float(out["diff_loss"].item()),
                "diff_self": float(out["diff_self"].item()),
                "diff_cross": float(out["diff_cross"].item()),
                "self_delta_rms": float(out["self_delta_rms"].item()),
                "cross_delta_rms": float(out["cross_delta_rms"].item()),
                "z_ref_rms": float(out["z_ref_rms"].item()),
                "grad_norm": grad_norm,
            }
            log_rows.append(row)
            print(
                f"[Overfit step {step:4d}/{steps}] loss={row['loss']:.4f} "
                f"(self={row['diff_self']:.4f}, cross={row['diff_cross']:.4f}) | "
                f"delta_rms self={row['self_delta_rms']:.5f} cross={row['cross_delta_rms']:.5f} | "
                f"zref={row['z_ref_rms']:.3f} gnorm={grad_norm:.3f}",
                flush=True,
            )

    policy.eval()
    with amp_ctx:
        fin_res = policy.predict_action(
            demo_rgb, pad_mask, query_rgb, use_residual=True, init_noise=z0
        )
        fin_ref = policy.predict_action(
            demo_rgb, pad_mask, query_rgb, use_residual=False, init_noise=z0
        )
    cross_m = ~is_self_t
    self_m = is_self_t
    fin_all = _unpack_action_errors(_action_error_sums(fin_res["action_10d"], actions_10d))
    fin_cross_res = _unpack_action_errors(
        _action_error_sums(fin_res["action_10d"][cross_m], actions_10d[cross_m])
    )
    fin_cross_ref = _unpack_action_errors(
        _action_error_sums(fin_ref["action_10d"][cross_m], actions_10d[cross_m])
    )
    fin_self_ref = _unpack_action_errors(
        _action_error_sums(fin_ref["action_10d"][self_m], actions_10d[self_m])
    )

    summary = {
        "mode": args.mode,
        "steps": steps,
        "lr": lr,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "init_ddim": init_err,
        "final_ddim_all": fin_all,
        "final_ddim_cross_with_res": fin_cross_res,
        "final_ddim_cross_ref_only": fin_cross_ref,
        "final_ddim_self_ref_only": fin_self_ref,
        "parameter_summary": policy.parameter_summary(),
        "weight_compat_report": policy.weight_compat_report,
        "curve": log_rows,
    }
    atomic_write_json(summary, out_dir / f"overfit_{args.mode}.json")
    print(
        f"[Overfit Done] {summary['elapsed_seconds']}s | "
        f"final_all_mse10={fin_all['action10_mse']:.5f} (init {init_err['action10_mse']:.4f}) | "
        f"final_all_mae7={fin_all['action7_mae']:.5f} (init {init_err['action7_mae']:.4f}) | "
        f"cross_res_mae7={fin_cross_res['action7_mae']:.5f} vs ref_only={fin_cross_ref['action7_mae']:.5f}",
        flush=True,
    )
    if fin_all["action10_mse"] >= 0.25 * init_err["action10_mse"]:
        raise RuntimeError(
            f"Overfit check did not converge sufficiently: {fin_all['action10_mse']} vs {init_err['action10_mse']}"
        )


def run_training(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if torch.cuda.is_available():
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

        run_name = args.run_name if args.run_name else args.mode
        run_dir = args.output_dir / run_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()

        # Preload full ~6.4 GB feature cache directly on each GPU (zero disk I/O during training!).
        bank = E5FeatureBank(
            cache_dir=args.cache_dir,
            device=device,
            resident_on_device=(device.type == "cuda"),
            self_ratio=args.self_ratio,
            seed=args.seed,
            schedule_mode=args.schedule_mode,
            num_epochs=args.epochs,
        )
        bundle = load_bpp_pretrained_weights(ckpt_path=BPP_CKPT)
        policy = build_policy_for_mode(
            args.mode,
            bank,
            bundle,
            args.seed,
            load_pretrained_reader=not args.no_pretrained_reader,
        ).to(device)

        # Exact BPP LR scaling: base_lr * world_size (5e-5 * 4 = 2e-4 on 4 GPUs) unless --lr given.
        lr = args.lr if args.lr is not None else (args.base_lr_per_gpu * world)
        optimizer = torch.optim.AdamW(
            policy.get_optim_groups(lr=lr, weight_decay=args.weight_decay),
            lr=lr,
            betas=(0.95, 0.999),
            weight_decay=args.weight_decay,
        )

        epoch_len = bank.samples_per_epoch()
        steps_per_epoch = math.ceil(epoch_len / args.global_batch)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(args.lr_warmup_proportion * total_steps)
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
            "mode": args.mode,
            "run_name": run_name,
            "schedule_mode": args.schedule_mode,
            "samples_per_epoch": epoch_len,
            "use_residual": args.mode == "e5",
            "seed": args.seed,
            "epochs": args.epochs,
            "world_size": world,
            "global_batch": args.global_batch,
            "per_rank_batch": per_rank_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "lr": lr,
            "base_lr_per_gpu": args.base_lr_per_gpu,
            "weight_decay": args.weight_decay,
            "betas": [0.95, 0.999],
            "lr_scheduler": "cosine",
            "lr_warmup_proportion": args.lr_warmup_proportion,
            "warmup_steps": warmup_steps,
            "clip_grad_norm": args.clip_grad_norm,
            "mixed_precision": "bf16",
            "use_ema": True,
            "ema_decay": args.ema_decay,
            "self_ratio": args.self_ratio,
            "self_res_weight": args.self_res_weight if args.mode == "e5" else 0.0,
            "cross_res_weight": args.cross_res_weight if args.mode == "e5" else 0.0,
            "load_pretrained_reader": not args.no_pretrained_reader,
            "cache_dir": str(args.cache_dir),
            "cache_meta": bank.meta,
            "parameter_summary": policy.parameter_summary(),
            "weight_compat_report": policy.weight_compat_report,
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
                    f"[E5 Train] Resumed {args.mode} from epoch {ck['epoch']} "
                    f"(best_epoch={best_epoch}, best_val={best_val_loss:.5f})",
                    flush=True,
                )

        ddp_model: torch.nn.Module = (
            DistributedDataParallel(policy, device_ids=[device.index]) if use_ddp else policy
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
                f"[E5 Train] mode={args.mode} | world={world} | global_batch={args.global_batch} | "
                f"epochs={args.epochs} ({total_steps} steps, warmup={warmup_steps}) | lr={lr:.2e} | "
                f"trainable={ps['policy_trainable']:,} | full_with_clip={ps['full_system_total_with_frozen_clip']:,}",
                flush=True,
            )
            print(
                "[E5 Train] Weight compatibility:",
                json.dumps(config["weight_compat_report"]),
                flush=True,
            )

        # Initial epoch-0 validation (before any weight updates) for clean baseline reference.
        init_val = None
        if start_epoch == 1:
            init_val = evaluate_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                run_ddim=True,
                ddim_max_samples=args.sample_count,
                val_seed=args.val_seed,
            )
            if rank == 0:
                atomic_write_json(init_val, run_dir / "init_val.json")
                print(
                    f"[E5 Init Val] cross_diff={init_val['val_cross_diff_loss']:.4f} | "
                    f"self_diff={init_val['val_self_ref_only_diff_loss']:.4f} | "
                    f"ddim_cross_mse10={init_val['ddim']['cross_with_res']['action10_mse']:.4f} | "
                    f"delta_rms cross={init_val['val_cross_delta_rms']:.6f} self={init_val['val_self_delta_rms']:.6f}",
                    flush=True,
                )

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.monotonic()
            ddp_model.train()
            # Accumulate train metrics on device:
            # [n_total, loss_sum, diff_sum, diff_self_sum, n_self, diff_cross_sum, n_cross,
            #  self_res_sum, cross_res_sum, self_dz_sq, cross_dz_sq, zref_sq, gnorm_sum, n_steps]
            tr_acc = torch.zeros(14, device=device, dtype=torch.float64)

            for demo_rgb, pad_mask, query_rgb, actions_10d, is_self in bank.iter_train_batches(
                epoch, per_rank_batch, rank=rank, world_size=world
            ):
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = ddp_model(
                        demo_rgb,
                        pad_mask,
                        query_rgb,
                        actions_10d,
                        is_self,
                        self_res_weight=args.self_res_weight,
                        cross_res_weight=args.cross_res_weight,
                    )
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

                B = float(len(demo_rgb))
                ns = float(out["n_self"].item())
                nc = float(out["n_cross"].item())
                tr_acc[0] += B
                tr_acc[1] += float(out["loss"].item()) * B
                tr_acc[2] += float(out["diff_loss"].item()) * B
                tr_acc[3] += float(out["diff_self"].item()) * ns
                tr_acc[4] += ns
                tr_acc[5] += float(out["diff_cross"].item()) * nc
                tr_acc[6] += nc
                tr_acc[7] += float(out["self_res_loss"].item()) * ns
                tr_acc[8] += float(out["cross_res_loss"].item()) * nc
                tr_acc[9] += (float(out["self_delta_rms"].item()) ** 2) * ns
                tr_acc[10] += (float(out["cross_delta_rms"].item()) ** 2) * nc
                tr_acc[11] += (float(out["z_ref_rms"].item()) ** 2) * B
                tr_acc[12] += grad_norm
                tr_acc[13] += 1.0

            tr_acc = _all_reduce_sum(tr_acc)
            t_train = time.monotonic() - t_ep

            n_tot = max(float(tr_acc[0].item()), 1.0)
            n_self_tot = max(float(tr_acc[4].item()), 1.0)
            n_cross_tot = max(float(tr_acc[6].item()), 1.0)
            n_steps_tot = max(float(tr_acc[13].item()), 1.0)
            train_metrics = {
                "samples": int(round(n_tot)),
                "self_fraction": float(tr_acc[4].item() / n_tot),
                "loss": float(tr_acc[1].item() / n_tot),
                "diff_loss": float(tr_acc[2].item() / n_tot),
                "diff_self": float(tr_acc[3].item() / n_self_tot),
                "diff_cross": float(tr_acc[5].item() / n_cross_tot),
                "self_res_loss": float(tr_acc[7].item() / n_self_tot),
                "cross_res_loss": float(tr_acc[8].item() / n_cross_tot),
                "self_delta_rms": math.sqrt(float(tr_acc[9].item() / n_self_tot)),
                "cross_delta_rms": math.sqrt(float(tr_acc[10].item() / n_cross_tot)),
                "z_ref_rms": math.sqrt(float(tr_acc[11].item() / n_tot)),
                "grad_norm_mean": float(tr_acc[12].item() / n_steps_tot),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "ema_decay": float(ema.cur_decay_value or 0.0),
                "train_seconds": round(t_train, 2),
            }

            # Evaluate using EMA weights (matching official BPP TrainPolicyWorkspace!).
            t_val0 = time.monotonic()
            ema.store(policy.parameters())
            ema.copy_to(policy.parameters())
            do_ddim = (
                epoch == 1
                or epoch == args.epochs
                or (args.sample_every > 0 and epoch % args.sample_every == 0)
            )
            val_metrics = evaluate_validation(
                policy,
                bank,
                batch_size=args.val_batch_size,
                run_ddim=do_ddim,
                ddim_max_samples=args.sample_count,
                val_seed=args.val_seed,
            )
            val_metrics["val_seconds"] = round(time.monotonic() - t_val0, 2)

            # Monitor EMA val cross diffusion loss for best checkpoint selection.
            cur_monitor = val_metrics["val_cross_diff_loss"]
            is_best = cur_monitor < best_val_loss
            if is_best:
                best_val_loss = cur_monitor
                best_epoch = epoch

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_cross_diff_loss": best_val_loss,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            # Save best EMA checkpoint BEFORE restoring raw training weights!
            if rank == 0 and is_best:
                atomic_torch_save(
                    {
                        "mode": args.mode,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_cross_diff_loss": best_val_loss,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": val_metrics,
                    },
                    best_ckpt,
                )

            # Restore live training weights before saving last.pt / continuing next epoch.
            ema.restore(policy.parameters())

            if rank == 0:
                if epoch % args.save_last_every == 0 or epoch == args.epochs:
                    atomic_torch_save(
                        {
                            "mode": args.mode,
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
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
                        "mode": args.mode,
                        "best_epoch": best_epoch,
                        "best_val_cross_diff_loss": best_val_loss,
                        "init_val": init_val,
                        "history": history,
                    },
                    run_dir / "metrics.json",
                )
                ddim_str = ""
                if "ddim" in val_metrics:
                    dm = val_metrics["ddim"]
                    ddim_str = (
                        f" | DDIM mse10={dm['cross_with_res']['action10_mse']:.4f} "
                        f"(ref={dm['cross_ref_only']['action10_mse']:.4f}, "
                        f"wrong={dm['wrong_demo']['action10_mse']:.4f}) "
                        f"mae7={dm['cross_with_res']['action7_mae']:.4f}"
                    )
                print(
                    f"[{args.mode} Ep {epoch:03d}/{args.epochs}] "
                    f"tr_diff={train_metrics['diff_loss']:.4f} "
                    f"(s={train_metrics['diff_self']:.4f},c={train_metrics['diff_cross']:.4f}) | "
                    f"val_cross={val_metrics['val_cross_diff_loss']:.4f} "
                    f"(ref={val_metrics['val_cross_ref_only_diff_loss']:.4f}, "
                    f"self={val_metrics['val_self_ref_only_diff_loss']:.4f}, "
                    f"wrong={val_metrics['val_wrong_demo_diff_loss']:.4f}) | "
                    f"dz_rms c={val_metrics['val_cross_delta_rms']:.4f}/s={val_metrics['val_self_delta_rms']:.4f} | "
                    f"lr={train_metrics['lr']:.2e} t={t_train:.1f}+{val_metrics['val_seconds']:.1f}s"
                    f"{'*' if is_best else ''}{ddim_str}",
                    flush=True,
                )
            if use_ddp:
                dist.barrier()

        # Final reloaded best.pt full-dataset (all 6,180 val pairs) DDIM & sensitivity evaluation!
        best_payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(best_payload["model"], strict=True)
        final_eval = evaluate_validation(
            policy,
            bank,
            batch_size=args.val_batch_size,
            run_ddim=True,
            ddim_max_samples=args.final_ddim_samples,
            val_seed=args.val_seed,
        )
        if rank == 0:
            final_report = {
                "mode": args.mode,
                "best_epoch": best_payload["epoch"],
                "best_checkpoint": str(best_ckpt),
                "last_checkpoint": str(last_ckpt),
                "best_val_cross_diff_loss": best_payload["best_val_cross_diff_loss"],
                "parameter_summary": config["parameter_summary"],
                "weight_compat_report": config["weight_compat_report"],
                "final_eval": final_eval,
            }
            atomic_write_json(final_report, run_dir / "final_eval.json")
            print(
                f"\n[{args.mode} FINAL BEST EVAL (ep={best_payload['epoch']})] "
                + json.dumps(final_eval, indent=2),
                flush=True,
            )
    finally:
        if use_ddp:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("e5", "bpp_rgb_only"), default="e5")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--base-lr-per-gpu", type=float, default=5.0e-5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-proportion", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=0.3)
    parser.add_argument("--self-ratio", type=float, default=0.5,
                        help="Target self-pair ratio; set < 0 (e.g. -1) for natural uniform sampling (~2.2% self)")
    parser.add_argument("--schedule-mode", choices=("manifest_unique", "all_cached_starts"),
                        default="manifest_unique")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--self-res-weight", type=float, default=1.0)
    parser.add_argument("--cross-res-weight", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--final-ddim-samples", type=int, default=None)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--no-pretrained-reader", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overfit:
        run_overfit_test(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
