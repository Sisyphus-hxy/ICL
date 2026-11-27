"""E13 Experiment 5: Exploring Compatibility of Modular Composition (E12-B) & Displacement Consistency (E12-C).

Keeps E9's EXACT architecture (`LocalBehaviorEncoder` + `ConditionalUnet1D`, `(B, 4, 768)`
latent), diffusion objective `L_diff`, initialization (`e9/outputs/stage1_local/best.pt`),
and 1.0-epoch budget (`5,364` steps, `batch=256`, `lr=1e-4` cosine, `EMA=0.999`, total
`lambda_rel = 0.15`).

Tests two principled B+C formulations:
  1. `e12_bc_joint`      : `L_diff + lambda_rel * (0.5 * L_B + 0.5 * L_C)`
                           Direct budget-matched joint combination of E12-B (4-token
                           multi-attribute similarity) and E12-C (global displacement).
  2. `e12_bc_factordisp` : `L_diff + lambda_rel * (0.75 * L_FactorDisp + 0.25 * L_B)`
                           Per-token Factorized Displacement Consistency: each token k
                           enforces Euclidean displacement direction + relative scale
                           consistency strictly with attribute k's change `Δa^(k)`
                           (Tok 0: Δtrans, Tok 1: Δrot, Tok 2: Δmag+Δtemp, Tok 3: Δgrip),
                           plus a light `0.25 * L_B` static anchor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.diffusion import EMAModel, get_cosine_schedule_with_warmup
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e9 import (
    DEFAULT_DENSE_CACHE_DIR,
    DEFAULT_STAGE1_LOCAL_CKPT,
    E5_CACHE_DIR,
    EMBED_DIM,
    LOCAL_WINDOW_LEN,
    NUM_BEHAVIOR_TOKENS,
    NUM_HEADS,
)
from e9.data import LocalClipFeatureBank
from e9.stage1_local_model import E9Stage1LocalPolicy
from e9.train_stage1_local import evaluate_stage1_local_validation
from e12.action_attributes import (
    extract_action_attribute_vectors,
    pairwise_cosine_sim,
)
from e12.losses import _off_diag_mask, compute_e12_b_loss, compute_e12_c_loss
from e12.train_e12 import PairedTransitionBatchSampler

from . import OUTPUT_ROOT


def compute_factorized_displacement_loss(
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
    *,
    norm_weight: float = 0.25,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-token Euclidean displacement direction & relative-scale consistency.

    Pairs `(2m -> 2m+1)` produce per-token latent displacements `Δb^(k)` `(M, 768)`
    aligned strictly to per-attribute action changes `Δa^(k)`:
      - Token 0: `Δtranslation` (`a_trans[1] - a_trans[0]`)
      - Token 1: `Δrotation` (`a_rot[1] - a_rot[0]`)
      - Token 2: `Δmagnitude_and_temporal` (standardized `[Δmag, 0.5*Δtemp]`)
      - Token 3: `Δgripper` (`a_grip[1] - a_grip[0]`)
    """
    B = b_t.shape[0]
    M = B // 2
    if M < 4:
        zero = b_t.sum() * 0.0
        return zero, {"factor_disp_loss": zero.detach()}

    b_pair = b_t[: 2 * M].float().reshape(M, 2, 4, -1)
    db = b_pair[:, 1] - b_pair[:, 0]  # (M, 4, D)

    with torch.no_grad():
        attrs = extract_action_attribute_vectors(actions_10d[: 2 * M].float(), actions_7d[: 2 * M].float() if actions_7d is not None else None)
        da_trans = attrs["translation"].reshape(M, 2, -1)
        da_trans = da_trans[:, 1] - da_trans[:, 0]

        da_rot = attrs["rotation"].reshape(M, 2, -1)
        da_rot = da_rot[:, 1] - da_rot[:, 0]

        mag_p = attrs["magnitude"].reshape(M, 2, -1)
        tmp_p = attrs["temporal"].reshape(M, 2, -1)
        da_mt = torch.cat([mag_p[:, 1] - mag_p[:, 0], 0.5 * (tmp_p[:, 1] - tmp_p[:, 0])], dim=-1)

        grp_p = attrs["gripper"].reshape(M, 2, -1)
        da_grip = grp_p[:, 1] - grp_p[:, 0]

    da_list = (da_trans, da_rot, da_mt, da_grip)
    min_norms = (0.3, 0.15, 0.3, 0.25)
    names = ("tok0_dtrans", "tok1_drot", "tok2_dmagtmp", "tok3_dgrip")
    off = _off_diag_mask(M, b_t.device)

    losses = []
    stats: Dict[str, torch.Tensor] = {}
    for k, (da_k, thr, name) in enumerate(zip(da_list, min_norms, names)):
        dz_k = db[:, k, :]  # (M, D)
        da_norm = torch.linalg.norm(da_k, dim=-1)
        dz_norm = torch.linalg.norm(dz_k, dim=-1)

        with torch.no_grad():
            S_da = pairwise_cosine_sim(da_k)
            active = da_norm > thr
            valid = off & active[:, None] & active[None, :]
        S_dz = pairwise_cosine_sim(dz_k)
        if int(valid.sum().item()) > 8:
            l_dir = F.mse_loss(S_dz[valid], S_da[valid])
        else:
            l_dir = F.mse_loss(S_dz[off], S_da[off])

        # Relative norm matching (crucially forces ||Δb^(k)|| -> 0 when Δa^(k) == 0!)
        da_rel = da_norm / da_norm.mean().clamp_min(eps)
        dz_rel = dz_norm / dz_norm.mean().clamp_min(eps)
        l_scale = F.mse_loss(dz_rel, da_rel)

        lk = l_dir + float(norm_weight) * l_scale
        losses.append(lk)
        stats[f"{name}_dir"] = l_dir.detach()
        stats[f"{name}_scale"] = l_scale.detach()

    total = torch.stack(losses).mean()
    stats["factor_disp_loss"] = total.detach()
    return total, stats


def compute_bc_loss(
    mode: str,
    b_t: torch.Tensor,
    actions_10d: torch.Tensor,
    actions_7d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if mode == "e12_bc_joint":
        lb, sb = compute_e12_b_loss(b_t, actions_10d, actions_7d)
        lc, sc = compute_e12_c_loss(b_t, actions_10d, actions_7d, norm_weight=0.25)
        tot = 0.5 * lb + 0.5 * lc
        stats = {"rel_loss": tot.detach(), "loss_b": lb.detach(), "loss_c": lc.detach()}
        return tot, stats
    if mode == "e12_bc_factordisp":
        lfd, sfd = compute_factorized_displacement_loss(b_t, actions_10d, actions_7d, norm_weight=0.25)
        lb, sb = compute_e12_b_loss(b_t, actions_10d, actions_7d)
        tot = 0.75 * lfd + 0.25 * lb
        stats = {"rel_loss": tot.detach(), "loss_fdisp": lfd.detach(), "loss_b": lb.detach()}
        return tot, stats
    raise ValueError(f"Unknown BC mode: {mode}")


def train_e12_bc(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    t0 = time.monotonic()
    bank = LocalClipFeatureBank(
        e5_cache_dir=E5_CACHE_DIR,
        dense_cache_dir=DEFAULT_DENSE_CACHE_DIR,
        device=device,
        resident_on_device=bool(args.gpu_resident_cache),
        seed=args.seed,
    )
    print(f"[E13 Exp5:{args.mode}] Loaded bank in {time.monotonic()-t0:.1f}s", flush=True)

    seed_everything(args.seed)
    policy = E9Stage1LocalPolicy(
        embed_dim=EMBED_DIM,
        window_len=LOCAL_WINDOW_LEN,
        num_behavior_tokens=NUM_BEHAVIOR_TOKENS,
        encoder_layers=6,
        encoder_heads=NUM_HEADS,
        encoder_ffn_dim=4 * EMBED_DIM,
        dropout=0.1,
        action_normalizer=bank.action_normalizer,
    ).to(device)
    payload = torch.load(DEFAULT_STAGE1_LOCAL_CKPT, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["model"], strict=True)

    optimizer = torch.optim.AdamW(
        policy.get_optim_groups(lr=args.lr, weight_decay=1e-6),
        lr=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
    total_steps = int(args.total_steps)
    warmup_steps = max(1, int(0.05 * total_steps))
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

    paired_sampler = PairedTransitionBatchSampler(bank, within_ep_frac=0.5, seed=args.seed)

    config = {
        "stage": f"e13_{args.mode}",
        "mode": args.mode,
        "init_mode": f"warm_start:{DEFAULT_STAGE1_LOCAL_CKPT}",
        "lambda_rel": float(args.lambda_rel),
        "rel_warmup_steps": int(args.rel_warmup_steps),
        "batch_size": int(args.batch_size),
        "total_steps": total_steps,
        "lr": float(args.lr),
        "ema_decay": float(args.ema_decay),
        "seed": int(args.seed),
    }
    atomic_write_json(config, run_dir / "config.json")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def _inf_batches():
        ep = 1
        while True:
            yield from paired_sampler.iter_batches(ep, args.batch_size)
            ep += 1

    b_iter = _inf_batches()
    step = 0
    best_val = float("inf")
    best_step = 0
    history = []
    wt0 = time.monotonic()
    ad = ar = 0.0
    as_n = 0

    while step < total_steps:
        policy.train()
        clip_rgb, clip_pad_mask, actions_10d = next(b_iter)
        step += 1
        cur_lam = (
            float(args.lambda_rel) * (step / float(args.rel_warmup_steps))
            if args.rel_warmup_steps > 0 and step <= args.rel_warmup_steps
            else float(args.lambda_rel)
        )

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            b_t = policy.encode_local_behavior(clip_rgb, clip_pad_mask=clip_pad_mask)
            diff_loss = policy.diffusion_loss_from_z_self(b_t, actions_10d).mean()
        a7 = action_10d_to_7d(actions_10d.float())
        rel_loss, rel_stats = compute_bc_loss(args.mode, b_t.float(), actions_10d.float(), a7)
        (diff_loss + cur_lam * rel_loss).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.3)
        optimizer.step()
        lr_scheduler.step()
        ema.step(policy.parameters())

        ad += float(diff_loss.item())
        ar += float(rel_loss.item())
        as_n += 1

        if (step % args.eval_every == 0) or (step == total_steps):
            dt = time.monotonic() - wt0
            ema.store(policy.parameters())
            ema.copy_to(policy.parameters())
            do_ddim = (step == total_steps) or (step % (args.eval_every * 2) == 0)
            vm = evaluate_stage1_local_validation(
                policy,
                bank,
                batch_size=256,
                run_ddim=do_ddim,
                ddim_max_samples=512,
                run_timestep_check=False,
                val_seed=12345,
            )
            cur_v = vm["val_self_diff_loss"]
            if cur_v < best_val:
                best_val = cur_v
                best_step = step
                atomic_torch_save(
                    {
                        "stage": f"e13_{args.mode}",
                        "step": step,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": vm,
                    },
                    run_dir / "best.pt",
                )
            if step == total_steps:
                atomic_torch_save(
                    {
                        "stage": f"e13_{args.mode}",
                        "step": step,
                        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                        "config": config,
                        "val_metrics": vm,
                    },
                    run_dir / "last_ema.pt",
                )
            ema.restore(policy.parameters())
            history.append(
                {
                    "step": step,
                    "tr_diff": ad / max(as_n, 1),
                    "tr_rel": ar / max(as_n, 1),
                    "val_diff": cur_v,
                    "rel_stats": {k: float(v.item()) for k, v in rel_stats.items()},
                }
            )
            atomic_write_json({"config": config, "best_step": best_step, "history": history}, run_dir / "metrics.json")
            ddim_s = ""
            if "ddim" in vm:
                dm = vm["ddim"]["self"]
                ddim_s = f" | ddim_mse10={dm['action10_mse']:.5f} mae7={dm['action7_mae']:.5f} grip={dm['grip_sign_acc']*100:.2f}%"
            print(
                f"[E13:{args.mode} step {step:5d}/{total_steps}] tr_diff={ad/max(as_n,1):.5f} "
                f"rel={ar/max(as_n,1):.5f} | val_diff={cur_v:.5f} b_rms={vm['val_self_b_rms']:.3f} "
                f"{as_n/max(dt,1e-6):.1f}it/s{ddim_s}",
                flush=True,
            )
            wt0 = time.monotonic()
            ad = ar = 0.0
            as_n = 0

    print(f"[E13:{args.mode}] Done. Saved to {run_dir / 'last_ema.pt'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", type=str, required=True, choices=["e12_bc_joint", "e12_bc_factordisp"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-resident-cache", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=5364)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--lambda-rel", type=float, default=0.15)
    p.add_argument("--rel-warmup-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_e12_bc(args)


if __name__ == "__main__":
    main()
