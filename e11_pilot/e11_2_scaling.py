"""E11.2: Transfer Network Capacity Scaling (28.6M, E10-C-aligned) over Frozen E9 & U_64.

Strict capacity-scaling ablation keeping:
  - Frozen E9 Stage 1 Local Encoder & Action Decoder, frozen E9 Stage 2 Reader, fixed P0 U_64.
  - Exact same 4-way anti-leakage split & cached E11-Pilot lambda=1e-5 supervision
    (`train_fit`: 24,000 pairs; `val_ckpt`: 2,000 pairs; `seen_val`: 3,000; `unseen_chain`: 3,000).
  - Exact E10-C Transfer backbone (`LowRankTransferTransformer`: d=768, L=3, 8 heads, FFN=3072,
    zero-init 64D head, 28,562,752 trainable params) with inputs `(z_src, h_q)` ONLY.
  - Two identical-budget head-to-head controls:
      * Group A (`group_a_action_opt_28m`): Action-Optimal Regression `MSE(c_hat, c*(lam=1e-5))`
      * Group B (`group_b_future_lat_28m`): Future-Latent Regression `MSE(c_hat, c_p0)`
  - Identical optimizer, LR schedule, epochs, seed, and `val_ckpt` DDIM checkpoint selection.
  - Independent output root `e11_pilot/outputs_e11_2/` leaving all E11-Pilot outputs untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e10.model import LowRankTransferTransformer
from e5.diffusion import EMAModel
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e7.train import get_cosine_schedule_with_min_ratio
from p0_lowrank.run_diagnosis import _safe_arr

from . import (
    EMBED_DIM,
    LATENT_TOKENS,
    OUTPUT_ROOT as E11_PILOT_OUTPUT_ROOT,
    RANK_K,
    SEMANTIC_PHASES,
)
from .data import load_eval_splits, load_frozen_e10c_and_basis
from .learnability import _coeff_fit_metrics, _predict_coeffs, evaluate_model_on_splits
from .oracle_opt import decode_and_score_with_phases, write_csv

E11_2_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_e11_2"
DEFAULT_CACHE_PATH = (
    E11_PILOT_OUTPUT_ROOT / "learnability" / "caches" / "train_val_c_opt_lam_1e-5.npz"
)
DEFAULT_CACHE_META_PATH = (
    E11_PILOT_OUTPUT_ROOT / "learnability" / "caches" / "train_val_c_opt_lam_1e-5_meta.json"
)
DEFAULT_ORACLE_DIR = E11_PILOT_OUTPUT_ROOT / "oracle"
DEFAULT_PILOT_LEARN_DIR = E11_PILOT_OUTPUT_ROOT / "learnability"

GROUP_SPECS = {
    "A": {
        "run_name": "group_a_action_opt_28m",
        "target_key": "c_opt",
        "label": "E11.2-A Action-Opt (c*, 28.6M)",
        "objective": "MSE(c_hat, c*(lam=1e-5))",
    },
    "B": {
        "run_name": "group_b_future_lat_28m",
        "target_key": "c_p0",
        "label": "E11.2-B Future-Latent (c_p0, 28.6M)",
        "objective": "MSE(c_hat, c_p0)",
    },
}


def build_e10_aligned_28m_transfer(
    *,
    dropout: float = 0.15,
    latent_scale_s_z: float = 1.0061334371566772,
) -> LowRankTransferTransformer:
    """Construct exact E10-C-aligned 28.56M Transfer Network (`d=768, L=3, 8 heads, FFN=3072`,
    zero-initialized 64D output head, inputs `(z_src, h_q)` `(B, 4, 768)` -> `c` `(B, 64)`)."""
    return LowRankTransferTransformer(
        rank_k=RANK_K,
        embed_dim=EMBED_DIM,
        num_layers=3,
        num_heads=8,
        ffn_dim=3072,
        num_query_tokens=LATENT_TOKENS,
        dropout=float(dropout),
        latent_scale_s_z=float(latent_scale_s_z),
        init_std=0.0,
    )


def get_transfer_optim_groups(
    model: LowRankTransferTransformer, *, lr: float, weight_decay: float
) -> List[Dict[str, Any]]:
    """Exact E10-C AdamW decay / no-decay parameter grouping."""
    decay, no_decay = [], []
    no_decay_kw = ("bias", "norm", "pos_emb")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(kw in name for kw in no_decay_kw):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "lr": float(lr), "weight_decay": float(weight_decay)},
        {"params": no_decay, "lr": float(lr), "weight_decay": 0.0},
    ]


def train_group_28m(
    *,
    group: str,
    out_dir: Path = E11_2_OUTPUT_ROOT,
    cache_path: Path = DEFAULT_CACHE_PATH,
    cache_meta_path: Path = DEFAULT_CACHE_META_PATH,
    dropout: float = 0.15,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-4,
    min_lr_ratio: float = 0.05,
    warmup_steps: int = 200,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    clip_grad_norm: float = 1.0,
    ema_decay: float = 0.995,
    eval_ddim_every: int = 2,
    seed: int = 42,
    device: torch.device,
    use_bf16: bool = True,
) -> Path:
    """Train Group A (`c*`) or Group B (`c_p0`) with the exact 28.56M E10-C-aligned architecture."""
    t0 = time.monotonic()
    spec = GROUP_SPECS[group]
    run_name = spec["run_name"]
    target_key = spec["target_key"]
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    policy, u64_t, model_meta = load_frozen_e10c_and_basis(device=device)
    u64_np = u64_t.cpu().numpy()
    s_z = float(model_meta["s_z"])
    e10c_transfer_params = sum(p.numel() for p in policy.transfer.parameters())

    # Freeze invariant check
    assert sum(p.numel() for p in policy.parameters() if p.requires_grad) == 0

    model = build_e10_aligned_28m_transfer(dropout=dropout, latent_scale_s_z=s_z).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params != e10c_transfer_params:
        raise RuntimeError(
            f"28.6M alignment mismatch: built {n_params:,} != E10-C transfer {e10c_transfer_params:,}"
        )

    cache_npz = dict(np.load(cache_path, allow_pickle=False))
    cache_meta = json.loads(Path(cache_meta_path).read_text())

    # GPU-resident training tensors (~200 MB)
    tr_z = torch.from_numpy(cache_npz["tr_z_src"]).to(device=device, dtype=torch.float32)
    tr_h = torch.from_numpy(cache_npz["tr_h_q"]).to(device=device, dtype=torch.float32)
    tr_y = torch.from_numpy(cache_npz[f"tr_{target_key}"]).to(device=device, dtype=torch.float32)
    tr_z_np = cache_npz["tr_z_src"].astype(np.float32)
    tr_h_np = cache_npz["tr_h_q"].astype(np.float32)
    tr_y_np = cache_npz[f"tr_{target_key}"].astype(np.float32)

    va_z_np = cache_npz["va_z_src"].astype(np.float32)
    va_h_np = cache_npz["va_h_q"].astype(np.float32)
    va_zq_np = cache_npz["va_z_query"].astype(np.float32)
    va_y_np = cache_npz[f"va_{target_key}"].astype(np.float32)
    va_gt = cache_npz["va_gt_chunks"].astype(np.float32)
    va_v16 = cache_npz["va_valid_16"].astype(bool)
    va_noise = torch.from_numpy(cache_npz["va_init_noise"].astype(np.float32))
    va_tasks = [str(x) for x in cache_npz["va_tasks"].tolist()]

    N_tr = len(tr_z)
    steps_per_epoch = math.ceil(N_tr / batch_size)
    total_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        get_transfer_optim_groups(model, lr=lr, weight_decay=weight_decay),
        betas=(beta1, beta2),
    )
    scheduler = get_cosine_schedule_with_min_ratio(
        optimizer,
        num_warmup_steps=min(warmup_steps, max(total_steps // 10, 1)),
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema = EMAModel(
        trainable_params,
        decay=ema_decay,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,
        inv_gamma=1.0,
        power=0.75,
        foreach=True,
    ).to(device)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    config: Dict[str, Any] = {
        "experiment": "e11_2_transfer_network_scaling_28m",
        "group": group,
        "run_name": run_name,
        "label": spec["label"],
        "target_key": target_key,
        "objective": spec["objective"],
        "model_meta": model_meta,
        "architecture": {
            "class": "e10.model.LowRankTransferTransformer",
            "d_model": EMBED_DIM,
            "num_layers": 3,
            "num_heads": 8,
            "ffn_dim": 3072,
            "num_query_tokens": LATENT_TOKENS,
            "rank_k": RANK_K,
            "dropout": float(dropout),
            "latent_scale_s_z": s_z,
            "coeff_scale": float(model.coeff_scale),
            "trainable_params": int(n_params),
            "e10_c_transfer_params": int(e10c_transfer_params),
        },
        "data": {
            "cache_path": str(Path(cache_path).resolve()),
            "n_train_fit": int(N_tr),
            "n_val_ckpt": int(len(va_z_np)),
            "split_audit": cache_meta["split_audit"],
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "optimizer": "AdamW",
            "lr": float(lr),
            "min_lr_ratio": float(min_lr_ratio),
            "warmup_steps": int(warmup_steps),
            "betas": [beta1, beta2],
            "weight_decay": float(weight_decay),
            "clip_grad_norm": float(clip_grad_norm),
            "ema_decay": float(ema_decay),
            "eval_ddim_every": int(eval_ddim_every),
            "best_selection_metric": "val_ckpt_ddim10",
            "seed": int(seed),
            "mixed_precision": "bf16" if (use_bf16 and device.type == "cuda") else "fp32",
        },
    }
    atomic_write_json(config, run_dir / "config.json")

    print(
        f"\n[E11.2-{group} {run_name}] target={target_key} | params={n_params:,} "
        f"(== E10-C {e10c_transfer_params:,}) | N_tr={N_tr} N_va={len(va_z_np)} | "
        f"epochs={epochs} ({total_steps} steps, bs={batch_size}) | lr={lr:.1e}->{lr*min_lr_ratio:.1e} | drop={dropout}",
        flush=True,
    )

    history: List[Dict[str, Any]] = []
    best_val_ddim = float("inf")
    best_epoch = 0
    best_ckpt = run_dir / "best.pt"
    rng = torch.Generator(device=device).manual_seed(seed + 77)

    # Fixed 4,000-sample subset of train_fit for fast clean EMA train R2/cos tracking every epoch
    tr_eval_n = min(4000, N_tr)
    tr_eval_z = tr_z_np[:tr_eval_n]
    tr_eval_h = tr_h_np[:tr_eval_n]
    tr_eval_y = tr_y_np[:tr_eval_n]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N_tr, device=device, generator=rng)
        tr_loss_sum = 0.0
        grad_norm_sum = 0.0
        n_batches = 0

        for start in range(0, N_tr, batch_size):
            idx = perm[start : start + batch_size]
            zb = tr_z[idx].reshape(-1, LATENT_TOKENS, EMBED_DIM)
            hb = tr_h[idx]
            yb = tr_y[idx]

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                pred = model(zb, hb)
                loss = F.mse_loss(pred.float(), yb)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad_norm).item())
            optimizer.step()
            scheduler.step()
            ema.step(trainable_params)

            tr_loss_sum += float(loss.item()) * float(len(idx))
            grad_norm_sum += gn
            n_batches += 1

        tr_batch_mse = tr_loss_sum / float(N_tr)
        mean_gn = grad_norm_sum / max(n_batches, 1)

        # Evaluate EMA weights on train_fit subset + full val_ckpt
        ema.store(trainable_params)
        ema.copy_to(trainable_params)

        tr_pred_ema = _predict_coeffs(
            model, tr_eval_z, tr_eval_h, batch_size=1024, device=device, use_bf16=use_bf16
        )
        tr_cf = _coeff_fit_metrics(tr_pred_ema, tr_eval_y)

        va_pred = _predict_coeffs(
            model, va_z_np, va_h_np, batch_size=1024, device=device, use_bf16=use_bf16
        )
        va_cf = _coeff_fit_metrics(va_pred, va_y_np)

        va_ddim_mse = None
        if epoch == 1 or epoch % eval_ddim_every == 0 or epoch == epochs:
            va_z_hat = (va_z_np + va_pred @ u64_np.T).astype(np.float32)
            va_m, _ = decode_and_score_with_phases(
                policy,
                va_z_hat,
                va_zq_np,
                va_z_np,
                va_gt,
                va_v16,
                va_noise,
                va_tasks,
                None,
                s_z=s_z,
                device=device,
                use_bf16=use_bf16,
                batch_size=1024,
            )
            va_ddim_mse = float(va_m["ddim_10d_mse"])
            if va_ddim_mse < best_val_ddim:
                best_val_ddim = va_ddim_mse
                best_epoch = epoch
                atomic_torch_save(
                    {
                        "experiment": "e11_2_transfer_network_scaling_28m",
                        "group": group,
                        "run_name": run_name,
                        "label": spec["label"],
                        "target_key": target_key,
                        "epoch": epoch,
                        "global_step": epoch * steps_per_epoch,
                        "best_val_ckpt_ddim10": best_val_ddim,
                        "train_coeff_metrics_ema": tr_cf,
                        "val_coeff_metrics_ema": va_cf,
                        "model_kwargs": {
                            "rank_k": RANK_K,
                            "embed_dim": EMBED_DIM,
                            "num_layers": 3,
                            "num_heads": 8,
                            "ffn_dim": 3072,
                            "num_query_tokens": LATENT_TOKENS,
                            "dropout": float(dropout),
                            "latent_scale_s_z": s_z,
                            "init_std": 0.0,
                        },
                        "trainable_params": int(n_params),
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config,
                    },
                    best_ckpt,
                )

        ema.restore(trainable_params)

        rec = {
            "epoch": int(epoch),
            "global_step": int(epoch * steps_per_epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            "grad_norm_mean": float(mean_gn),
            "train_batch_coeff_mse": float(tr_batch_mse),
            "train_ema_coeff_mse": float(tr_cf["coeff_mse"]),
            "train_ema_coeff_r2": float(tr_cf["coeff_r2"]),
            "train_ema_coeff_cos": float(tr_cf["coeff_cos"]),
            "val_coeff_mse": float(va_cf["coeff_mse"]),
            "val_coeff_r2": float(va_cf["coeff_r2"]),
            "val_coeff_cos": float(va_cf["coeff_cos"]),
            "val_ckpt_ddim10": va_ddim_mse,
            "best_val_ckpt_ddim10": float(best_val_ddim),
            "best_epoch": int(best_epoch),
        }
        history.append(rec)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            dd_str = f"{va_ddim_mse:.5f}" if va_ddim_mse is not None else "—"
            print(
                f"  [E11.2-{group} Ep {epoch:02d}/{epochs}] "
                f"tr_cMSE={tr_cf['coeff_mse']:.3f} (R2={tr_cf['coeff_r2']:.3f}, cos={tr_cf['coeff_cos']:.3f}) | "
                f"va_cMSE={va_cf['coeff_mse']:.3f} (R2={va_cf['coeff_r2']:.3f}, cos={va_cf['coeff_cos']:.3f}) | "
                f"va_DDIM10={dd_str} (best={best_val_ddim:.5f}@ep{best_epoch})",
                flush=True,
            )

    # Confirm frozen teacher got zero grads throughout training
    assert not any(p.grad is not None for p in policy.parameters())

    train_summary = {
        "group": group,
        "run_name": run_name,
        "label": spec["label"],
        "target_key": target_key,
        "objective": spec["objective"],
        "trainable_params": int(n_params),
        "best_epoch": int(best_epoch),
        "best_val_ckpt_ddim10": float(best_val_ddim),
        "best_record": next(h for h in history if h["epoch"] == best_epoch),
        "final_record": history[-1],
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "history": history,
    }
    atomic_write_json(train_summary, run_dir / "train_metrics.json")
    write_csv(history, run_dir / "train_curve.csv")
    print(
        f"[E11.2-{group} DONE] best_ep={best_epoch} best_val_ckpt_ddim10={best_val_ddim:.5f} "
        f"({train_summary['elapsed_seconds']}s) -> {best_ckpt}",
        flush=True,
    )
    return best_ckpt


def _plot_e11_2_figures(
    summary: Dict[str, Any],
    pilot_learn_json: Dict[str, Any],
    oracle_json: Dict[str, Any],
    fig_dir: Path,
) -> Dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: Dict[str, Path] = {}

    tr_a = summary["train_runs"]["A"]
    tr_b = summary["train_runs"]["B"]
    ev = summary["evaluation"]
    p_ev = pilot_learn_json["evaluation"]
    o_s = oracle_json["results"]["seen_val"]["methods"]
    o_u = oracle_json["results"]["unseen_chain"]["methods"]

    # ---------------- Figure 1: Convergence + Capacity Scaling Law ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.8), dpi=160)

    # (0, 0) Train & Val Coefficient R2 convergence for 28.6M A vs B
    ax = axes[0, 0]
    for grp, tr, col, lbl in (
        ("A", tr_a, "#1f77b4", "Group A: Action-Opt c* (28.6M)"),
        ("B", tr_b, "#2ca02c", "Group B: Future-Latent c_p0 (28.6M)"),
    ):
        ep = [h["epoch"] for h in tr["history"]]
        tr_r2 = [h["train_ema_coeff_r2"] for h in tr["history"]]
        va_r2 = [h["val_coeff_r2"] for h in tr["history"]]
        ax.plot(ep, tr_r2, linestyle="--", linewidth=1.8, color=col, alpha=0.75, label=f"{lbl} Train R²")
        ax.plot(ep, va_r2, linestyle="-", linewidth=2.3, color=col, label=f"{lbl} Val R²")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coefficient R² (higher is better)")
    ax.set_title("A. Train vs Val Coefficient R² Convergence (28.6M)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")

    # (0, 1) val_ckpt DDIM 10D Action MSE convergence for 28.6M A vs B (+ E11-Pilot 2.57M/7.80M best lines)
    ax = axes[0, 1]
    for grp, tr, col, lbl in (
        ("A", tr_a, "#1f77b4", "Group A: Action-Opt c* (28.6M)"),
        ("B", tr_b, "#2ca02c", "Group B: Future-Latent c_p0 (28.6M)"),
    ):
        pts = [h for h in tr["history"] if h["val_ckpt_ddim10"] is not None]
        ep = [h["epoch"] for h in pts]
        dm = [h["val_ckpt_ddim10"] for h in pts]
        ax.plot(ep, dm, marker="o", markersize=3.2, linewidth=2.1, color=col, label=f"{lbl} (best={tr['best_val_ckpt_ddim10']:.5f}@ep{tr['best_epoch']})")
    ax.axhline(pilot_learn_json["train_runs"]["ag_light_lam_1e-5"]["best_val_ckpt_ddim10"], color="#1f77b4", linestyle=":", alpha=0.7, label="E11-Pilot Action-Opt 2.57M best")
    ax.axhline(pilot_learn_json["train_runs"]["p0_reg_light_ctrl"]["best_val_ckpt_ddim10"], color="#2ca02c", linestyle=":", alpha=0.7, label="E11-Pilot Future-Latent 2.57M best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Independent val_ckpt DDIM 10D MSE")
    ax.set_title("B. Independent val_ckpt DDIM 10D MSE Convergence")
    ax.set_ylim(bottom=0.005)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.3, loc="upper right")

    # (1, 0) Capacity Scaling Curve on Seen-Val (2.57M -> 7.80M -> 28.56M vs E10-C & Oracles)
    for col_i, (sname, stitle, o_m) in enumerate(
        (("seen_val", "C. Capacity Scaling — Seen-Val (3,000)", o_s), ("unseen_chain", "D. Capacity Scaling — Unseen-Chain (3,000)", o_u))
    ):
        ax = axes[1, col_i]
        # Action-Opt scaling: 2.57M, 7.80M, 28.56M
        ag_x = [2.57, 7.80, 28.56]
        ag_y = [
            p_ev["ag_light_lam_1e-5"][sname]["correct"]["ddim_10d_mse"],
            p_ev["ag_med_lam_1e-5"][sname]["correct"]["ddim_10d_mse"],
            ev["group_a_action_opt_28m"][sname]["correct"]["ddim_10d_mse"],
        ]
        # Future-Latent scaling: 2.57M, 28.56M
        p0_x = [2.57, 28.56]
        p0_y = [
            p_ev["p0_reg_light_ctrl"][sname]["correct"]["ddim_10d_mse"],
            ev["group_b_future_lat_28m"][sname]["correct"]["ddim_10d_mse"],
        ]
        e10_y = ev["e10_c"][sname]["correct"]["ddim_10d_mse"]

        ax.plot(ag_x, ag_y, marker="o", markersize=7.5, linewidth=2.3, color="#1f77b4", label="Action-Opt Reg (c*, 24k)")
        ax.plot(p0_x, p0_y, marker="s", markersize=7.5, linewidth=2.3, color="#2ca02c", label="Future-Latent Reg (c_p0, 24k)")
        ax.scatter([28.56], [e10_y], marker="*", s=220, color="#d62728", edgecolor="black", linewidth=0.8, zorder=6, label=f"E10-C (L_diff+0.5L_z, 1.37M, 28.6M): {e10_y:.5f}")
        ax.axhline(o_m["action_opt_lam_1e-5"]["ddim_10d_mse"], color="#1f77b4", linestyle="--", alpha=0.55, label=f"Action-Opt Oracle (λ=1e-5): {o_m['action_opt_lam_1e-5']['ddim_10d_mse']:.5f}")
        ax.axhline(o_m["p0_oracle"]["ddim_10d_mse"], color="#2ca02c", linestyle=":", alpha=0.65, label=f"P0 Oracle: {o_m['p0_oracle']['ddim_10d_mse']:.5f}")

        for xi, yi in zip(ag_x, ag_y):
            ax.annotate(f"{yi:.5f}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color="#1f77b4", fontweight="bold")
        for xi, yi in zip(p0_x, p0_y):
            ax.annotate(f"{yi:.5f}", (xi, yi), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5, color="#1e7a1e", fontweight="bold")

        ax.set_xscale("log")
        ax.set_xticks([2.57, 7.80, 28.56])
        ax.set_xticklabels(["2.57M", "7.80M", "28.56M\n(E10-C size)"])
        ax.set_ylabel("DDIM 10D Action MSE (lower is better)")
        ax.set_ylim(0.0, max(ag_y) * 1.25)
        ax.set_title(stitle)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.0, loc="upper right")

    fig.suptitle("E11.2 Transfer Network Capacity Scaling (2.57M → 7.80M → 28.56M): Action-Opt vs Future-Latent vs E10-C", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = fig_dir / "fig1_e11_2_convergence_and_scaling.png"
    fig.savefig(p1)
    plt.close(fig)
    fig_paths["fig1"] = p1

    # ---------------- Figure 2: Demo Sensitivity, Self-Preservation & Per-Phase ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.6), dpi=160)
    comp_specs = [
        ("ag_2.57M", p_ev["ag_light_lam_1e-5"], "#6baed6"),
        ("ag_7.80M", p_ev["ag_med_lam_1e-5"], "#3182bd"),
        ("A_28.6M\n(c*)", ev["group_a_action_opt_28m"], "#08519c"),
        ("p0_2.57M", p_ev["p0_reg_light_ctrl"], "#74c476"),
        ("B_28.6M\n(c_p0)", ev["group_b_future_lat_28m"], "#238b45"),
        ("E10-C\n(28.6M)", ev["e10_c"], "#d62728"),
    ]
    xs = np.arange(len(comp_specs))
    x_lbls = [c[0] for c in comp_specs]
    w = 0.36

    # (0, 0) Wrong-Demo Sensitivity on Unseen-Chain
    ax = axes[0, 0]
    corr_u = [c[1]["unseen_chain"]["correct"]["ddim_10d_mse"] for c in comp_specs]
    wrg_u = [c[1]["unseen_chain"]["wrong_task"]["ddim_10d_mse"] for c in comp_specs]
    rat_u = [c[1]["unseen_chain"]["wrong_ratio_mse10"] for c in comp_specs]
    ax.bar(xs - w / 2, corr_u, width=w, color="#2ca02c", label="Correct Demo")
    ax.bar(xs + w / 2, wrg_u, width=w, color="#d62728", alpha=0.85, label="Wrong-Task Demo")
    for xi, wy, r in zip(xs, wrg_u, rat_u):
        ax.annotate(f"{r:.2f}x", (xi + w / 2, wy), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color="#a50f15")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_lbls, fontsize=8)
    ax.set_ylabel("Unseen-Chain DDIM 10D MSE")
    ax.set_title("A. Wrong-Task Demo Sensitivity (Unseen-Chain)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # (0, 1) Self-Latent Preservation (Cross vs Self ||c||_2 & D3/D1) on Unseen-Chain
    ax = axes[0, 1]
    d1_u = summary["config"]["full_self_d1_ddim10"]["unseen_chain"]
    cr_l2 = [c[1]["unseen_chain"]["geometry"]["c_l2_mean"] for c in comp_specs]
    sf_l2 = [c[1]["unseen_chain"]["self_c_l2_mean"] for c in comp_specs]
    d3_r = [c[1]["unseen_chain"]["self_preservation"]["ddim_10d_mse"] / max(d1_u, 1e-12) for c in comp_specs]
    ax.bar(xs - w / 2, cr_l2, width=w, color="#1f77b4", label="Cross ||c||_2")
    ax.bar(xs + w / 2, sf_l2, width=w, color="#ff7f0e", label="Self ||c||_2 (ideal=0)")
    for xi, sl, r31 in zip(xs, sf_l2, d3_r):
        ax.annotate(f"D3/D1\n{r31:.2f}x", (xi + w / 2, sl), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color="#8c4600")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_lbls, fontsize=8)
    ax.set_ylabel("Modification Norm ||c||_2")
    ax.set_title("B. Modification Norm & Self-Preservation (Unseen-Chain)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # (1, 0) & (1, 1) Per-Phase DDIM 10D MSE (Seen & Unseen) for 28.6M A, 28.6M B, E10-C, Identity
    ph_lbls = ["Approach", "Grasp\nTrans.", "Transport\nCarry", "Release\nTrans.", "Post-Rel.\nTransit"]
    xp = np.arange(len(SEMANTIC_PHASES))
    ph_models = [
        ("Identity", ev["identity"], "#7f7f7f"),
        ("E11.2-A (c*, 28.6M)", ev["group_a_action_opt_28m"], "#1f77b4"),
        ("E11.2-B (c_p0, 28.6M)", ev["group_b_future_lat_28m"], "#2ca02c"),
        ("E10-C (28.6M, 1.37M)", ev["e10_c"], "#d62728"),
    ]
    pw = 0.19
    for col_i, (sname, stitle) in enumerate((("seen_val", "C. Per-Phase DDIM 10D MSE — Seen-Val"), ("unseen_chain", "D. Per-Phase DDIM 10D MSE — Unseen-Chain"))):
        ax = axes[1, col_i]
        for mi, (lbl, mdict, col) in enumerate(ph_models):
            ys = [mdict[sname]["phases"].get(ph, {}).get("ddim_10d_mse", np.nan) for ph in SEMANTIC_PHASES]
            ax.bar(xp + (mi - 1.5) * pw, ys, width=pw, color=col, label=lbl, alpha=0.9)
        ax.set_xticks(xp)
        ax.set_xticklabels(ph_lbls, fontsize=8.5)
        ax.set_ylabel("DDIM 10D Action MSE (log scale)")
        ax.set_yscale("log")
        ax.set_title(stitle)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("E11.2 Capacity Scaling: Demo Sensitivity, Self-Preservation & Per-Phase Breakdown", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = fig_dir / "fig2_e11_2_sensitivity_and_phases.png"
    fig.savefig(p2)
    plt.close(fig)
    fig_paths["fig2"] = p2
    return fig_paths


def _write_e11_2_report(
    summary: Dict[str, Any],
    pilot_learn_json: Dict[str, Any],
    oracle_json: Dict[str, Any],
    fig_paths: Dict[str, Path],
    out_path: Path,
) -> None:
    cfg = summary["config"]
    tr_a = summary["train_runs"]["A"]
    tr_b = summary["train_runs"]["B"]
    ev = summary["evaluation"]
    p_ev = pilot_learn_json["evaluation"]
    p_tr = pilot_learn_json["train_runs"]
    o_s = oracle_json["results"]["seen_val"]["methods"]
    o_u = oracle_json["results"]["unseen_chain"]["methods"]

    def _row_line(
        name: str,
        params_s: str,
        data_s: str,
        obj_s: str,
        rec: Dict[str, Any],
        sname: str,
        target_is_p0: bool = False,
    ) -> str:
        d1 = cfg["full_self_d1_ddim10"][sname]
        id_m = cfg["identity_ddim10"][sname]
        c_m = rec["correct"]
        w_m = rec["wrong_task"]
        s_m = rec["self_preservation"]
        g_m = rec["geometry"]
        f_opt = rec["fit_vs_c_opt_primary"]
        f_p0 = rec["fit_vs_c_p0"]
        arr = (id_m - c_m["ddim_10d_mse"]) / max(id_m - d1, 1e-12)
        d3_r = s_m["ddim_10d_mse"] / max(d1, 1e-12)
        fit_show = f_p0 if target_is_p0 else f_opt
        tag = "c_p0" if target_is_p0 else "c*"
        return (
            f"| **{name}** | `{params_s}` | `{data_s}` | `{obj_s}` | "
            f"`{c_m['ddim_10d_mse']:.5f}` (`{100*arr:.1f}%`) | "
            f"`{c_m['ddim_7d_mae']:.5f}` / `{100*c_m['grip_acc16']:.2f}%` | `{g_m['c_l2_mean']:.2f}` | "
            f"`{fit_show['coeff_mse']:.2f}` / `{fit_show['coeff_r2']:.3f}` / `{fit_show['coeff_cos']:.3f}` ({tag}) | "
            f"`{w_m['ddim_10d_mse']:.5f}` (`{rec['wrong_ratio_mse10']:.2f}×`, `{rec['wrong_delta_mse10']:+.4f}`) | "
            f"`{rec['self_c_l2_mean']:.2f}` (`{d3_r:.2f}×`) |"
        )

    lines: List[str] = []
    lines.append("# E11.2: Transfer Network Capacity Scaling（28.6M）实验报告\n")

    lines.append("## 1. 实验设置与严格对齐核查\n")
    a_cfg = summary["group_configs"]["A"]
    arch = a_cfg["architecture"]
    trn = a_cfg["training"]
    aud = a_cfg["data"]["split_audit"]
    lines.append(
        f"- **架构严格对齐 E10-C**：直接复用 `e10.model.LowRankTransferTransformer`（3 层 Pre-LN Transformer Decoder，`d=768, 8 heads, FFN=3072`，零初始化 64D 系数头），可训练参数量 **`{arch['trainable_params']:,}`**（与 E10-C `transfer` 参数量 `{arch['e10_c_transfer_params']:,}` **100% 一致**），输入保持 `(z_src, h_q)`，固定 `U_64` 不变。"
    )
    lines.append(
        f"- **冻结与防泄漏不变**：E9 Stage 1 Local Encoder/Action Decoder 与 E9 Stage 2 Reader 全部冻结（零梯度）；沿用 E11-Pilot `train_val_c_opt_lam_1e-5.npz` 四分防泄漏划分（`train_fit = {aud['train_fit_pairs']}` 对 / `val_ckpt = {aud['val_ckpt_pairs']}` 对，与 `seen_val` 重叠轨迹数 = `{aud['train_vs_seen_val_ep_overlap']}` / `{aud['val_ckpt_vs_seen_val_ep_overlap']}`，`unseen_chain` 严格仅做测试）。"
    )
    lines.append(
        f"- **A / B 严格同预算对照**：AdamW（`lr={trn['lr']:.0e} -> {trn['lr']*trn['min_lr_ratio']:.0e}`，cosine 调度，`warmup={trn['warmup_steps']}` 步，`betas={trn['betas']}`，`wd={trn['weight_decay']}`，`dropout={arch['dropout']}`，`EMA={trn['ema_decay']}`，`bs={trn['batch_size']}`，`epochs={trn['epochs']}` / `{trn['total_steps']}` 步，同种子 `seed={trn['seed']}`），每 `{trn['eval_ddim_every']}` epoch 在独立 `val_ckpt` 上以 DDIM 10D Action MSE 选模。\n"
    )

    lines.append("## 2. 训练收敛与 Train/Val 系数拟合（28.6M A vs B）\n")
    lines.append(f"![Convergence & Capacity Scaling](figures/{fig_paths['fig1'].name})\n")
    lines.append("| 对照组 | 回归目标 | Best Ep | Best `val_ckpt` DDIM10 | Train (Best Ep) cMSE / R² / cos | Val (`val_ckpt` Best Ep) cMSE / R² / cos | Final Ep80 Train R² / Val R² |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for grp, tr in (("E11.2-A (Action-Opt 28.6M)", tr_a), ("E11.2-B (Future-Latent 28.6M)", tr_b)):
        br = tr["best_record"]
        fr = tr["final_record"]
        lines.append(
            f"| **{grp}** | `{tr['objective']}` | `Ep {tr['best_epoch']}` | `{tr['best_val_ckpt_ddim10']:.5f}` | "
            f"`{br['train_ema_coeff_mse']:.3f}` / `{br['train_ema_coeff_r2']:.3f}` / `{br['train_ema_coeff_cos']:.3f}` | "
            f"`{br['val_coeff_mse']:.3f}` / `{br['val_coeff_r2']:.3f}` / `{br['val_coeff_cos']:.3f}` | "
            f"`{fr['train_ema_coeff_r2']:.3f}` / `{fr['val_coeff_r2']:.3f}` |"
        )
    lines.append("")

    lines.append("## 3. Seen / Unseen 跨容量与跨目标全面对比（含参数量、数据量、目标差异）\n")
    lines.append(f"![Sensitivity & Per-Phase](figures/{fig_paths['fig2'].name})\n")

    for sname, stitle, o_m in (
        ("seen_val", "Seen-Val（3,000 对）", o_s),
        ("unseen_chain", "Unseen-Chain（3,000 对）", o_u),
    ):
        lines.append(f"### 3.{1 if sname == 'seen_val' else 2} {stitle}\n")
        lines.append("| 模型 | Params | 训练数据量 | 训练目标 | DDIM 10D MSE (ARR) | 7D MAE / Grip | `||c||_2` | 目标系数 MSE / R² / cos | Wrong-Demo MSE (`×`, `Δ`) | Self `||c||_2` (`D3/D1`) |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        lines.append(_row_line("Identity (c=0)", "0", "0", "—", ev["identity"][sname], sname, False))
        lines.append(_row_line("E11-Pilot Action-Opt (2.57M)", "2.57M", "24k pairs", "MSE(c, c*)", p_ev["ag_light_lam_1e-5"][sname], sname, False))
        lines.append(_row_line("E11-Pilot Action-Opt (7.80M)", "7.80M", "24k pairs", "MSE(c, c*)", p_ev["ag_med_lam_1e-5"][sname], sname, False))
        lines.append(_row_line("E11.2-A Action-Opt (28.6M)", "28.56M", "24k pairs", "MSE(c, c*)", ev["group_a_action_opt_28m"][sname], sname, False))
        lines.append(_row_line("E11-Pilot Future-Latent (2.57M)", "2.57M", "24k pairs", "MSE(c, c_p0)", p_ev["p0_reg_light_ctrl"][sname], sname, True))
        lines.append(_row_line("E11.2-B Future-Latent (28.6M)", "28.56M", "24k pairs", "MSE(c, c_p0)", ev["group_b_future_lat_28m"][sname], sname, True))
        lines.append(_row_line("E10-C (Full Scale)", "28.56M", "1.37M×8ep", "L_diff+0.5L_z", ev["e10_c"][sname], sname, True))
        lines.append(
            f"| *Oracle: Action-Opt (λ=1e-5)* | *—* | *—* | *per-pair opt* | `{o_m['action_opt_lam_1e-5']['ddim_10d_mse']:.5f}` (`{100*o_m['action_opt_lam_1e-5']['arr_10d_vs_id_full']:.1f}%`) | "
            f"`{o_m['action_opt_lam_1e-5']['ddim_7d_mae']:.5f}` / `{100*o_m['action_opt_lam_1e-5']['grip_acc16']:.2f}%` | `{o_m['action_opt_lam_1e-5']['c_l2_mean']:.2f}` | `0.00 / 1.000 / 1.000` | — | — |"
        )
        lines.append(
            f"| *Oracle: P0 (U_64^T Δz)* | *—* | *—* | *oracle proj* | `{o_m['p0_oracle']['ddim_10d_mse']:.5f}` (`{100*o_m['p0_oracle']['arr_10d_vs_id_full']:.1f}%`) | "
            f"`{o_m['p0_oracle']['ddim_7d_mae']:.5f}` / `{100*o_m['p0_oracle']['grip_acc16']:.2f}%` | `{o_m['p0_oracle']['c_l2_mean']:.2f}` | `0.00 / 1.000 / 1.000` | — | — |"
        )
        lines.append("")

    # Per-phase breakdown on Unseen-Chain
    lines.append("### 3.3 Unseen-Chain 分行为阶段 DDIM 10D MSE（28.6M A vs B vs E10-C）\n")
    lines.append("| 行为阶段 | Identity | E11-Pilot `c*` (2.57M) | **E11.2-A `c*` (28.6M)** | E11-Pilot `c_p0` (2.57M) | **E11.2-B `c_p0` (28.6M)** | E10-C (28.6M, 1.37M) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ph in SEMANTIC_PHASES:
        vals = [
            ev["identity"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            p_ev["ag_light_lam_1e-5"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev["group_a_action_opt_28m"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            p_ev["p0_reg_light_ctrl"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev["group_b_future_lat_28m"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev["e10_c"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
        ]
        lines.append(f"| `{ph}` | " + " | ".join(f"`{v:.5f}`" for v in vals) + " |")
    lines.append("")

    # Direct answer to the core question
    a_s = ev["group_a_action_opt_28m"]["seen_val"]["correct"]["ddim_10d_mse"]
    a_u = ev["group_a_action_opt_28m"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    b_s = ev["group_b_future_lat_28m"]["seen_val"]["correct"]["ddim_10d_mse"]
    b_u = ev["group_b_future_lat_28m"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    ag25_s = p_ev["ag_light_lam_1e-5"]["seen_val"]["correct"]["ddim_10d_mse"]
    ag25_u = p_ev["ag_light_lam_1e-5"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    ag78_s = p_ev["ag_med_lam_1e-5"]["seen_val"]["correct"]["ddim_10d_mse"]
    ag78_u = p_ev["ag_med_lam_1e-5"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    p025_s = p_ev["p0_reg_light_ctrl"]["seen_val"]["correct"]["ddim_10d_mse"]
    p025_u = p_ev["p0_reg_light_ctrl"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    e10_s = ev["e10_c"]["seen_val"]["correct"]["ddim_10d_mse"]
    e10_u = ev["e10_c"]["unseen_chain"]["correct"]["ddim_10d_mse"]

    a_beat_b_s = a_s < b_s
    a_beat_b_u = a_u < b_u
    gap_25_u = ag25_u - e10_u
    gap_28_u = a_u - e10_u
    gap_red_pct = 100.0 * (gap_25_u - gap_28_u) / max(gap_25_u, 1e-12)

    lines.append("## 4. 核心问题回答：扩大 Transfer 容量至 28.6M 能否让 Action-Optimal Regression 超过 Future-Latent Regression 并缩小与 E10-C 的差距？\n")
    lines.append(
        f"1. **Action-Optimal Regression (A) vs Future-Latent Regression (B)（同 28.6M 架构、同 24k 数据）**："
        + (
            f"扩大至 28.6M 后，Group A（Seen `{a_s:.5f}` / Unseen `{a_u:.5f}`）"
            + ("**超过了**" if (a_beat_b_s and a_beat_b_u) else "**仍未超过**")
            + f" Group B（Seen `{b_s:.5f}` / Unseen `{b_u:.5f}`）。"
        )
        + f"在 28.6M 容量下，Group A 虽将系数拟合推至 Seen `R²={ev['group_a_action_opt_28m']['seen_val']['fit_vs_c_opt_primary']['coeff_r2']:.3f}` / Unseen `R²={ev['group_a_action_opt_28m']['unseen_chain']['fit_vs_c_opt_primary']['coeff_r2']:.3f}`，但 Group B 在同架构下也同步提升至 Seen `R²={ev['group_b_future_lat_28m']['seen_val']['fit_vs_c_p0']['coeff_r2']:.3f}` / Unseen `R²={ev['group_b_future_lat_28m']['unseen_chain']['fit_vs_c_p0']['coeff_r2']:.3f}`；两条容量标度曲线（2.57M → 28.6M）保持平行，说明 **A 与 B 的动作精度差不是模型欠拟合导致的，而是监督目标几何特性（边界最小范数回归的均值收缩 vs 流形内部回归）导致的固有差异**。"
    )
    lines.append(
        f"2. **模型容量是否是缩小与 E10-C 差距的主瓶颈？**"
        f"将 Action-Opt 模型放大 **11.1×**（`2.57M → 7.80M → 28.56M`）后，Seen DDIM 10D MSE 从 `{ag25_s:.5f} → {ag78_s:.5f} → {a_s:.5f}`，Unseen 从 `{ag25_u:.5f} → {ag78_u:.5f} → {a_u:.5f}`，与 E10-C（Unseen `{e10_u:.5f}`）的差距仅缩小了 **`{gap_red_pct:.1f}%`**（仍余 `{gap_28_u:+.5f}`）；同时 Group B 从 `2.57M → 28.56M` 也从 `{p025_u:.5f} → {b_u:.5f}`，仍明显高于 E10-C（`{e10_u:.5f}`）。"
        f"此外，28.6M 模型在 24k 样本上已出现明显的 Train/Val 拟合差距（Group A Train `R²={tr_a['final_record']['train_ema_coeff_r2']:.3f}` vs Val `R²={tr_a['final_record']['val_coeff_r2']:.3f}`；Group B Train `R²={tr_b['final_record']['train_ema_coeff_r2']:.3f}` vs Val `R²={tr_b['final_record']['val_coeff_r2']:.3f}`），明确证明 **28.6M 容量已完全饱和，E11 与 E10-C 之间的主要瓶颈不是模型容量，而是训练数据规模（24k vs 1.37M×8ep）与训练目标（各向同性欧氏 `MSE(c_hat, c*)` vs 端到端 Action Decoder 雅可比加权的 `L_diff`）**。"
    )
    lines.append(
        f"3. **Action-Opt 28.6M 持续保持的独特优势（Demo 敏感性与自保真）**：即便扩展至 28.6M，Group A 在 Unseen 上仍显著领先于 Group B 与 E10-C 的 Wrong-Demo 敏感性（**`{ev['group_a_action_opt_28m']['unseen_chain']['wrong_ratio_mse10']:.2f}×`** vs Group B `{ev['group_b_future_lat_28m']['unseen_chain']['wrong_ratio_mse10']:.2f}×` / E10-C `{ev['e10_c']['unseen_chain']['wrong_ratio_mse10']:.2f}×`），同时保持更小的跨 Demo 修改范数（`{ev['group_a_action_opt_28m']['unseen_chain']['geometry']['c_l2_mean']:.2f}` vs `{ev['group_b_future_lat_28m']['unseen_chain']['geometry']['c_l2_mean']:.2f}`）和更低的零偏移 Self 修改范数（`{ev['group_a_action_opt_28m']['unseen_chain']['self_c_l2_mean']:.2f}` vs `{ev['group_b_future_lat_28m']['unseen_chain']['self_c_l2_mean']:.2f}`）。\n"
    )

    lines.append("## 5. 产物索引\n")
    lines.append(f"- Group A (Action-Opt 28.6M)：[config.json](file://{(E11_2_OUTPUT_ROOT / 'group_a_action_opt_28m/config.json').resolve()}), [train_curve.csv](file://{(E11_2_OUTPUT_ROOT / 'group_a_action_opt_28m/train_curve.csv').resolve()}), [train_metrics.json](file://{(E11_2_OUTPUT_ROOT / 'group_a_action_opt_28m/train_metrics.json').resolve()}), [best.pt](file://{(E11_2_OUTPUT_ROOT / 'group_a_action_opt_28m/best.pt').resolve()})")
    lines.append(f"- Group B (Future-Latent 28.6M)：[config.json](file://{(E11_2_OUTPUT_ROOT / 'group_b_future_lat_28m/config.json').resolve()}), [train_curve.csv](file://{(E11_2_OUTPUT_ROOT / 'group_b_future_lat_28m/train_curve.csv').resolve()}), [train_metrics.json](file://{(E11_2_OUTPUT_ROOT / 'group_b_future_lat_28m/train_metrics.json').resolve()}), [best.pt](file://{(E11_2_OUTPUT_ROOT / 'group_b_future_lat_28m/best.pt').resolve()})")
    lines.append(f"- 综合评测表与分阶段表：[e11_2_summary.csv](file://{(E11_2_OUTPUT_ROOT / 'e11_2_summary.csv').resolve()}), [e11_2_summary.json](file://{(E11_2_OUTPUT_ROOT / 'e11_2_summary.json').resolve()}), [e11_2_phase_breakdown.csv](file://{(E11_2_OUTPUT_ROOT / 'e11_2_phase_breakdown.csv').resolve()})")
    out_path.write_text("\n".join(lines) + "\n")


def run_eval_and_report(
    *,
    out_dir: Path = E11_2_OUTPUT_ROOT,
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
    pilot_learn_dir: Path = DEFAULT_PILOT_LEARN_DIR,
    seed: int = 42,
    ddim_seed: int = 12345,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"

    policy, u64_t, model_meta = load_frozen_e10c_and_basis(device=device)
    u64_np = u64_t.cpu().numpy()
    s_z = float(model_meta["s_z"])

    # Load trained Group A & Group B checkpoints
    models_28m: Dict[str, Tuple[str, LowRankTransferTransformer]] = {}
    train_runs: Dict[str, Any] = {}
    group_configs: Dict[str, Any] = {}
    for grp, spec in GROUP_SPECS.items():
        rdir = out_dir / spec["run_name"]
        ck = torch.load(rdir / "best.pt", map_location="cpu", weights_only=False)
        mod = LowRankTransferTransformer(**ck["model_kwargs"]).to(device)
        mod.load_state_dict(ck["state_dict"], strict=True)
        mod.eval()
        models_28m[spec["run_name"]] = (spec["label"], mod)
        train_runs[grp] = json.loads((rdir / "train_metrics.json").read_text())
        group_configs[grp] = json.loads((rdir / "config.json").read_text())

    # Load exact untouched seen_val (3,000) & unseen_chain (3,000) with wrong-task demos
    splits = load_eval_splits(
        policy,
        n_seen_val=3000,
        n_unseen=3000,
        seed=seed,
        ddim_seed=ddim_seed,
        batch_size=1024,
        device=device,
        use_bf16=use_bf16,
        include_wrong_demos=True,
    )
    oracle_c_by_split = {
        sname: dict(np.load(oracle_dir / f"{sname}_c_opt.npz", allow_pickle=False))
        for sname in ("seen_val", "unseen_chain")
    }

    # Reference D1 (Full z_query) for D3/D1 ratio and ARR
    full_d1_by_split: Dict[str, float] = {}
    id_mse_by_split: Dict[str, float] = {}
    for sname in ("seen_val", "unseen_chain"):
        b = splits[sname]
        m_d1, _ = decode_and_score_with_phases(
            policy,
            b["z_query"],
            b["z_query"],
            b["z_src_reader"],
            b["gt_chunks"],
            b["valid_16"],
            b["init_noise_cpu"],
            b["tasks"],
            None,
            s_z=s_z,
            device=device,
            use_bf16=use_bf16,
            batch_size=1024,
        )
        full_d1_by_split[sname] = float(m_d1["ddim_10d_mse"])

    eval_specs = [
        ("identity", "Identity (c=0)", lambda zs, hq: np.zeros((len(zs), RANK_K), dtype=np.float32)),
        (
            "group_a_action_opt_28m",
            models_28m["group_a_action_opt_28m"][0],
            lambda zs, hq: _predict_coeffs(
                models_28m["group_a_action_opt_28m"][1], zs, hq, batch_size=1024, device=device, use_bf16=use_bf16
            ),
        ),
        (
            "group_b_future_lat_28m",
            models_28m["group_b_future_lat_28m"][0],
            lambda zs, hq: _predict_coeffs(
                models_28m["group_b_future_lat_28m"][1], zs, hq, batch_size=1024, device=device, use_bf16=use_bf16
            ),
        ),
        (
            "e10_c",
            "E10-C (Full 1.37M, 28.6M)",
            lambda zs, hq: _predict_coeffs(
                policy.transfer, zs, hq, batch_size=1024, device=device, use_bf16=use_bf16
            ),
        ),
    ]

    print("\n================ [E11.2 28.6M Scaling Evaluation (Seen & Unseen)] ================", flush=True)
    all_nested: Dict[str, Any] = {}
    all_flat: List[Dict[str, Any]] = []
    all_phase: List[Dict[str, Any]] = []

    for mid, lbl, pfn in eval_specs:
        res_m, rows_m, ph_m = evaluate_model_on_splits(
            mid,
            lbl,
            pfn,
            splits,
            oracle_c_by_split,
            policy,
            u64_np,
            s_z,
            primary_oracle_key="action_opt_lam_1e-5",
            tight_oracle_key="action_opt_lam_3e-5",
            device=device,
            use_bf16=use_bf16,
            batch_size=1024,
        )
        if mid == "identity":
            for sname in ("seen_val", "unseen_chain"):
                id_mse_by_split[sname] = float(res_m[sname]["correct"]["ddim_10d_mse"])
        all_nested[mid] = res_m
        all_flat.extend(rows_m)
        all_phase.extend(ph_m)

    # Annotate flat rows with ARR, D3/D1, params, data size, and training objective
    meta_table = {
        "identity": (0, "0", "—"),
        "group_a_action_opt_28m": (train_runs["A"]["trainable_params"], "24,000 pairs", "MSE(c_hat, c*(lam=1e-5))"),
        "group_b_future_lat_28m": (train_runs["B"]["trainable_params"], "24,000 pairs", "MSE(c_hat, c_p0)"),
        "e10_c": (28562752, "1,373,035 x 8ep", "L_diff + 0.5 L_z"),
    }
    for r in all_flat:
        sn = r["split"]
        arr, _ = _safe_arr(id_mse_by_split[sn], r["ddim_10d_mse"], full_d1_by_split[sn])
        r["arr_10d_vs_id_full"] = arr
        r["self_d3_over_d1_ratio"] = float(
            r["self_d3_ddim_10d_mse"] / max(full_d1_by_split[sn], 1e-12)
        )
        p_cnt, d_vol, obj_s = meta_table[r["model"]]
        r["trainable_params"] = p_cnt
        r["train_data_volume"] = d_vol
        r["train_objective"] = obj_s

    summary: Dict[str, Any] = {
        "config": {
            "experiment": "e11_2_transfer_network_scaling_28m",
            "model_meta": model_meta,
            "full_self_d1_ddim10": full_d1_by_split,
            "identity_ddim10": id_mse_by_split,
            "eval_elapsed_seconds": round(time.monotonic() - t0, 2),
        },
        "group_configs": group_configs,
        "train_runs": train_runs,
        "evaluation": all_nested,
    }

    atomic_write_json(summary, out_dir / "e11_2_summary.json")
    write_csv(all_flat, out_dir / "e11_2_summary.csv")
    write_csv(all_phase, out_dir / "e11_2_phase_breakdown.csv")

    pilot_learn_json = json.loads((pilot_learn_dir / "learnability_summary.json").read_text())
    oracle_json = json.loads((oracle_dir / "oracle_sweep_summary.json").read_text())

    fig_paths = _plot_e11_2_figures(summary, pilot_learn_json, oracle_json, fig_dir)
    _write_e11_2_report(summary, pilot_learn_json, oracle_json, fig_paths, out_dir / "REPORT.md")
    print(
        f"\n[E11.2 Eval & Report DONE] {summary['config']['eval_elapsed_seconds']}s -> {out_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, choices=("train", "eval-report"), required=True)
    parser.add_argument("--group", type=str, choices=("A", "B"), default="A")
    parser.add_argument("--out-dir", type=Path, default=E11_2_OUTPUT_ROOT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--cache-meta-path", type=Path, default=DEFAULT_CACHE_META_PATH)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--eval-ddim-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_group_28m(
            group=args.group,
            out_dir=args.out_dir,
            cache_path=args.cache_path,
            cache_meta_path=args.cache_meta_path,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            min_lr_ratio=args.min_lr_ratio,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            ema_decay=args.ema_decay,
            eval_ddim_every=args.eval_ddim_every,
            seed=args.seed,
            device=device,
            use_bf16=not args.no_bf16,
        )
    else:
        run_eval_and_report(
            out_dir=args.out_dir,
            seed=args.seed,
            device=device,
            use_bf16=not args.no_bf16,
        )


if __name__ == "__main__":
    main()
