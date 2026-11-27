"""Stage 4 of E11-Pilot: Learnability Verification for Action-Guided Minimal Adaptation.

Two-stage pipeline (supports multi-GPU parallel c* cache generation on GPU 4 & GPU 5):
  1. `gen-cache`: Build strictly episode-disjoint `train_fit` (from ~90% of fold='train' layouts)
     and `val_ckpt` (from ~10% of fold='train' layouts) pairs, optimize `c*(lam)` using ONLY
     frozen E9 Action Decoder diffusion loss, and save cached `(z_src, h_q, c_opt, c_p0, ...)`.
     ZERO overlap with `seen_val` (fold='val') or `unseen_chain`!
  2. `train-eval`: Freeze E9 Encoder, Reader, Action Decoder, and `U_64`. Train lightweight
     Transfer Network `c_hat = f_theta(z_src, h_q)` ONLY on `train_fit` `c*` (plus matched-data
     `c_p0` Future-Latent Regression control). Select best checkpoint strictly on independent
     `val_ckpt` DDIM 10D Action MSE. Test on untouched `seen_val` (3,000) and `unseen_chain` (3,000),
     reporting coefficient MSE/R2/cos, DDIM Action MSE, Correct/Wrong Demo sensitivity,
     Self-Preservation, and per-behavioral-phase results.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.diffusion import EMAModel
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e7.train import get_cosine_schedule_with_min_ratio
from p0_lowrank.run_diagnosis import _safe_arr

from . import (
    EMBED_DIM,
    LATENT_DIM,
    LATENT_TOKENS,
    OUTPUT_ROOT,
    RANK_K,
    SEMANTIC_PHASES,
)
from .data import (
    build_learnability_train_val_bundles,
    load_eval_splits,
    load_frozen_e10c_and_basis,
)
from .oracle_opt import (
    compute_coeff_geometry,
    decode_and_score_with_phases,
    optimize_coefficients_batch,
    write_csv,
)


class LightweightLowRankTransfer(nn.Module):
    """Lightweight Cross-Attention Transfer Network predicting 64-dim coefficient `c`
    exclusively from existing inputs `(z_src, h_q)` of shape `(B, 4, 768)` each.

    Zero-initialized output head guarantees exact Identity (`c = 0`) at initialization.
    Default (`d_model=256, num_layers=2, num_heads=4, ffn_dim=1024`) has ~2.5M params
    (~11.4x lighter than E10-C's 28.6M Transformer).
    """

    def __init__(
        self,
        *,
        rank_k: int = RANK_K,
        in_dim: int = EMBED_DIM,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_tokens: int = LATENT_TOKENS,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.rank_k = int(rank_k)
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.num_tokens = int(num_tokens)

        self.z_in_norm = nn.LayerNorm(self.in_dim)
        self.h_in_norm = nn.LayerNorm(self.in_dim)
        self.z_in_proj = nn.Linear(self.in_dim, self.d_model)
        self.h_in_proj = nn.Linear(self.in_dim, self.d_model)

        self.z_pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, self.d_model))
        self.h_pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, self.d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.d_model)
        self.coeff_head = nn.Linear(self.num_tokens * self.d_model, self.rank_k)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.z_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.h_pos_emb, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init output head -> exact Identity c = 0 at step 0
        nn.init.zeros_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)

    def forward(self, z_src: torch.Tensor, h_q: torch.Tensor) -> torch.Tensor:
        """Map `(z_src, h_q)` `(B, 4, 768)` -> predicted coefficients `c_hat` `(B, 64)`."""
        if z_src.ndim == 2:
            z_src = z_src.reshape(-1, self.num_tokens, self.in_dim)
        if h_q.ndim == 2:
            h_q = h_q.reshape(-1, self.num_tokens, self.in_dim)
        zt = self.z_in_proj(self.z_in_norm(z_src)) + self.z_pos_emb
        ht = self.h_in_proj(self.h_in_norm(h_q)) + self.h_pos_emb
        out = self.decoder(tgt=zt, memory=ht)
        flat = self.out_norm(out).reshape(out.shape[0], self.num_tokens * self.d_model)
        return self.coeff_head(flat)

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        decay, no_decay = [], []
        no_decay_kw = ("bias", "norm", "pos_emb")
        for name, p in self.named_parameters():
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


def lam_to_tag(lam: float) -> str:
    if lam == 0.0:
        return "lam_0"
    return f"lam_{lam:.0e}".replace("+", "").replace("-0", "-")


def generate_train_val_c_cache(
    *,
    lam: float,
    out_dir: Path = OUTPUT_ROOT / "learnability" / "caches",
    n_train_fit: int = 32000,
    n_val_ckpt: int = 2000,
    steps: int = 250,
    lr: float = 1.0,
    mc_timesteps: int = 4,
    batch_size: int = 1024,
    seed: int = 1101,
    ddim_seed: int = 12345,
    device: torch.device,
    use_bf16: bool = True,
) -> Path:
    """Generate and save `c*(lam)` strictly on `train_fit` and `val_ckpt` (both from `fold='train'`)."""
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = lam_to_tag(lam)
    cache_path = out_dir / f"train_val_c_opt_{tag}.npz"
    meta_path = out_dir / f"train_val_c_opt_{tag}_meta.json"

    policy, u64_t, model_meta = load_frozen_e10c_and_basis(device=device)
    u64_np = u64_t.cpu().numpy()

    tr_b, va_b, audit = build_learnability_train_val_bundles(
        policy,
        n_train_fit=n_train_fit,
        n_val_ckpt=n_val_ckpt,
        seed=seed,
        ddim_seed=ddim_seed,
        batch_size=batch_size,
        device=device,
        use_bf16=use_bf16,
    )
    print(
        f"[E11-Cache {tag}] Built anti-leakage splits: {json.dumps(audit)}",
        flush=True,
    )

    # Optimize c* on train_fit
    t_tr = time.monotonic()
    c_tr_opt, tr_curves = optimize_coefficients_batch(
        policy.teacher,
        tr_b["z_src_reader"],
        tr_b["gt_chunks"],
        u64_t,
        lam=float(lam),
        steps=steps,
        lr=lr,
        mc_timesteps=mc_timesteps,
        batch_size=batch_size,
        seed=seed + 50000 + int(round(lam * 1e7)),
        device=device,
        use_bf16=use_bf16,
        log_steps=(0, 50, 150, steps),
    )
    dt_tr = time.monotonic() - t_tr
    print(
        f"[E11-Cache {tag}] train_fit ({len(c_tr_opt)}) done in {dt_tr:.1f}s | "
        f"diff={tr_curves[-1]['diff_loss']:.5f} | ||c*||_2={tr_curves[-1]['c_l2_mean']:.3f}",
        flush=True,
    )

    # Optimize c* on val_ckpt
    t_va = time.monotonic()
    c_va_opt, va_curves = optimize_coefficients_batch(
        policy.teacher,
        va_b["z_src_reader"],
        va_b["gt_chunks"],
        u64_t,
        lam=float(lam),
        steps=steps,
        lr=lr,
        mc_timesteps=mc_timesteps,
        batch_size=batch_size,
        seed=seed + 60000 + int(round(lam * 1e7)),
        device=device,
        use_bf16=use_bf16,
        log_steps=(0, 50, 150, steps),
    )
    dt_va = time.monotonic() - t_va
    print(
        f"[E11-Cache {tag}] val_ckpt ({len(c_va_opt)}) done in {dt_va:.1f}s | "
        f"diff={va_curves[-1]['diff_loss']:.5f} | ||c*||_2={va_curves[-1]['c_l2_mean']:.3f}",
        flush=True,
    )

    c_tr_p0 = ((tr_b["z_query"] - tr_b["z_src_reader"]) @ u64_np).astype(np.float32)
    c_va_p0 = ((va_b["z_query"] - va_b["z_src_reader"]) @ u64_np).astype(np.float32)

    # Quick sanity check on val_ckpt DDIM MSE for Identity, P0 Oracle, and c*
    s_z = float(model_meta["s_z"])
    va_id, _ = decode_and_score_with_phases(
        policy,
        va_b["z_src_reader"],
        va_b["z_query"],
        va_b["z_src_reader"],
        va_b["gt_chunks"],
        va_b["valid_16"],
        va_b["init_noise_cpu"],
        va_b["tasks"],
        None,
        s_z=s_z,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
    )
    va_opt_m, _ = decode_and_score_with_phases(
        policy,
        (va_b["z_src_reader"] + c_va_opt @ u64_np.T).astype(np.float32),
        va_b["z_query"],
        va_b["z_src_reader"],
        va_b["gt_chunks"],
        va_b["valid_16"],
        va_b["init_noise_cpu"],
        va_b["tasks"],
        None,
        s_z=s_z,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
    )
    print(
        f"[E11-Cache {tag}] val_ckpt sanity: Identity DDIM10={va_id['ddim_10d_mse']:.5f} -> "
        f"Oracle c* DDIM10={va_opt_m['ddim_10d_mse']:.5f}",
        flush=True,
    )

    # Save float16 for large (z_src, h_q) arrays to keep disk compact & fast, float32 for targets
    np.savez(
        cache_path,
        tr_z_src=tr_b["z_src_reader"].astype(np.float16),
        tr_h_q=tr_b["h_q"].astype(np.float16),
        tr_c_opt=c_tr_opt.astype(np.float32),
        tr_c_p0=c_tr_p0.astype(np.float32),
        va_z_src=va_b["z_src_reader"].astype(np.float16),
        va_h_q=va_b["h_q"].astype(np.float16),
        va_z_query=va_b["z_query"].astype(np.float16),
        va_c_opt=c_va_opt.astype(np.float32),
        va_c_p0=c_va_p0.astype(np.float32),
        va_gt_chunks=va_b["gt_chunks"].astype(np.float32),
        va_valid_16=va_b["valid_16"].astype(np.bool_),
        va_init_noise=va_b["init_noise_cpu"].numpy().astype(np.float32),
        va_tasks=np.asarray(va_b["tasks"]),
    )
    meta = {
        "lambda": float(lam),
        "tag": tag,
        "cache_path": str(cache_path.resolve()),
        "n_train_fit": len(c_tr_opt),
        "n_val_ckpt": len(c_va_opt),
        "steps": int(steps),
        "lr": float(lr),
        "mc_timesteps": int(mc_timesteps),
        "seed": int(seed),
        "ddim_seed": int(ddim_seed),
        "split_audit": audit,
        "train_opt_curves": tr_curves,
        "val_opt_curves": va_curves,
        "val_ckpt_identity_ddim10": va_id["ddim_10d_mse"],
        "val_ckpt_oracle_c_opt_ddim10": va_opt_m["ddim_10d_mse"],
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    atomic_write_json(meta, meta_path)
    print(f"[E11-Cache {tag} DONE] Saved -> {cache_path} ({meta['elapsed_seconds']}s)", flush=True)
    return cache_path


def _predict_coeffs(
    model: nn.Module,
    z_src_np: np.ndarray,
    h_q_np: np.ndarray,
    *,
    batch_size: int = 1024,
    device: torch.device,
    use_bf16: bool = True,
) -> np.ndarray:
    model.eval()
    N = len(z_src_np)
    out = np.empty((N, RANK_K), dtype=np.float32)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with torch.inference_mode():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start
            zs_t = torch.from_numpy(
                z_src_np[start:end].astype(np.float32).reshape(B, LATENT_TOKENS, EMBED_DIM)
            ).to(device=device)
            hq_t = torch.from_numpy(h_q_np[start:end].astype(np.float32)).to(device=device)
            with amp_ctx:
                pred = model(zs_t, hq_t)
            out[start:end] = pred.float().cpu().numpy()
    return out


def _coeff_fit_metrics(c_pred: np.ndarray, c_target: np.ndarray) -> Dict[str, float]:
    cp = np.asarray(c_pred, dtype=np.float64)
    ct = np.asarray(c_target, dtype=np.float64)
    mse = float(np.mean((cp - ct) ** 2))
    l2_err = float(np.mean(np.linalg.norm(cp - ct, axis=-1)))
    var_t = float(np.mean((ct - ct.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - mse / max(var_t, 1e-18))
    # Uncentered variance explained: 1 - ||cp - ct||_F^2 / ||ct||_F^2
    ev_uncentered = float(1.0 - np.sum((cp - ct) ** 2) / max(float(np.sum(ct * ct)), 1e-18))
    pn = np.linalg.norm(cp, axis=-1)
    tn = np.linalg.norm(ct, axis=-1)
    denom = pn * tn
    ok = denom > 1e-12
    cos = float(np.mean(np.sum(cp[ok] * ct[ok], axis=-1) / denom[ok])) if np.any(ok) else 0.0
    return {
        "coeff_mse": mse,
        "coeff_l2_err": l2_err,
        "coeff_r2": r2,
        "coeff_ev_uncentered": ev_uncentered,
        "coeff_cos": cos,
    }


def train_lightweight_transfer(
    *,
    run_name: str,
    run_dir: Path,
    target_key: str,  # "c_opt" or "c_p0"
    cache_npz: Dict[str, np.ndarray],
    policy: torch.nn.Module,
    u64_np: np.ndarray,
    s_z: float,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_dim: int = 1024,
    dropout: float = 0.15,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 5e-4,
    min_lr_ratio: float = 0.05,
    weight_decay: float = 1e-2,
    ema_decay: float = 0.995,
    eval_ddim_every: int = 5,
    seed: int = 42,
    device: torch.device,
    use_bf16: bool = True,
) -> Tuple[LightweightLowRankTransfer, Dict[str, Any]]:
    """Train LightweightLowRankTransfer purely on `train_fit` supervision (`target_key`),
    selecting best checkpoint strictly on independent `val_ckpt` DDIM 10D Action MSE."""
    t0 = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    # Keep frozen teacher strictly frozen and verify
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    assert sum(p.numel() for p in policy.parameters() if p.requires_grad) == 0

    model = LightweightLowRankTransfer(
        rank_k=RANK_K,
        in_dim=EMBED_DIM,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_tokens=LATENT_TOKENS,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GPU-resident tensors (32k * 3072 * 2 bytes = ~196 MB -> trivial on 85GB VRAM!)
    tr_z = torch.from_numpy(cache_npz["tr_z_src"]).to(device=device, dtype=torch.float32)
    tr_h = torch.from_numpy(cache_npz["tr_h_q"]).to(device=device, dtype=torch.float32)
    tr_y = torch.from_numpy(cache_npz[f"tr_{target_key}"]).to(device=device, dtype=torch.float32)

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
        model.get_optim_groups(lr=lr, weight_decay=weight_decay),
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_min_ratio(
        optimizer,
        num_warmup_steps=min(200, max(total_steps // 15, 1)),
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    ema = EMAModel(
        list(model.parameters()),
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

    history: List[Dict[str, Any]] = []
    best_val_ddim = float("inf")
    best_epoch = 0
    best_ckpt = run_dir / "best.pt"
    rng = torch.Generator(device=device).manual_seed(seed + 77)

    print(
        f"\n[E11-Train {run_name}] target={target_key} | params={n_params:,} "
        f"(d={d_model}, L={num_layers}) | N_tr={N_tr} N_va={len(va_z_np)} | epochs={epochs}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N_tr, device=device, generator=rng)
        tr_loss_sum = 0.0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.step(list(model.parameters()))
            tr_loss_sum += float(loss.item()) * float(len(idx))

        tr_mse = tr_loss_sum / float(N_tr)

        # Evaluate EMA weights on val_ckpt
        ema.store(list(model.parameters()))
        ema.copy_to(list(model.parameters()))

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
                        "run_name": run_name,
                        "target_key": target_key,
                        "epoch": epoch,
                        "best_val_ckpt_ddim10": best_val_ddim,
                        "val_coeff_metrics": va_cf,
                        "model_kwargs": {
                            "rank_k": RANK_K,
                            "in_dim": EMBED_DIM,
                            "d_model": d_model,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "ffn_dim": ffn_dim,
                            "num_tokens": LATENT_TOKENS,
                            "dropout": dropout,
                        },
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    },
                    best_ckpt,
                )

        ema.restore(list(model.parameters()))

        rec = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_coeff_mse": tr_mse,
            "val_coeff_mse": va_cf["coeff_mse"],
            "val_coeff_r2": va_cf["coeff_r2"],
            "val_coeff_cos": va_cf["coeff_cos"],
            "val_ckpt_ddim10": va_ddim_mse,
            "best_val_ckpt_ddim10": best_val_ddim,
            "best_epoch": best_epoch,
        }
        history.append(rec)
        if va_ddim_mse is not None and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"  [{run_name} Ep {epoch:02d}/{epochs}] tr_cMSE={tr_mse:.3f} | "
                f"va_cMSE={va_cf['coeff_mse']:.3f} (R2={va_cf['coeff_r2']:.3f}, cos={va_cf['coeff_cos']:.3f}) | "
                f"va_DDIM10={va_ddim_mse:.5f} (best={best_val_ddim:.5f}@ep{best_epoch})",
                flush=True,
            )

    # Reload best EMA checkpoint
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    train_meta = {
        "run_name": run_name,
        "target_key": target_key,
        "trainable_params": n_params,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "ema_decay": ema_decay,
        "best_epoch": best_epoch,
        "best_val_ckpt_ddim10": best_val_ddim,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "history": history,
    }
    atomic_write_json(train_meta, run_dir / "train_metrics.json")
    return model, train_meta


def evaluate_model_on_splits(
    model_id: str,
    label: str,
    predict_fn,
    splits: Dict[str, Dict[str, Any]],
    oracle_c_by_split: Dict[str, Dict[str, np.ndarray]],
    policy: torch.nn.Module,
    u64_np: np.ndarray,
    s_z: float,
    *,
    primary_oracle_key: str,
    tight_oracle_key: Optional[str],
    device: torch.device,
    use_bf16: bool = True,
    batch_size: int = 1024,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate a transfer model (or baseline) on `seen_val` and `unseen_chain`:
      - Correct Demo DDIM 10D Action MSE & per-phase metrics
      - Wrong-Task Demo DDIM 10D Action MSE & Correct/Wrong sensitivity
      - Self-Preservation (feeding z_src = z_query)
      - Coefficient MSE / R2 / cosine vs c*(primary), c*(tight), and c_p0
    """
    res_by_split: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []

    for sname in ("seen_val", "unseen_chain"):
        b = splits[sname]
        oc = oracle_c_by_split[sname]
        zq = b["z_query"]
        zsrc = b["z_src_reader"]
        zwrg = b["z_wrong_reader"]
        hq = b["h_q"]
        c_p0 = oc["c_p0"]
        c_e10c = b["c_e10c"]

        c_corr = predict_fn(zsrc, hq)
        c_wrg = predict_fn(zwrg, hq)
        c_self = predict_fn(zq, hq)

        z_corr_hat = (zsrc + c_corr @ u64_np.T).astype(np.float32)
        z_wrg_hat = (zwrg + c_wrg @ u64_np.T).astype(np.float32)
        z_self_hat = (zq + c_self @ u64_np.T).astype(np.float32)

        m_corr, ph_corr = decode_and_score_with_phases(
            policy,
            z_corr_hat,
            zq,
            zsrc,
            b["gt_chunks"],
            b["valid_16"],
            b["init_noise_cpu"],
            b["tasks"],
            b["phases"],
            c_mat=c_corr,
            c_p0=c_p0,
            s_z=s_z,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
        )
        m_wrg, _ = decode_and_score_with_phases(
            policy,
            z_wrg_hat,
            zq,
            zwrg,
            b["gt_chunks"],
            b["valid_16"],
            b["init_noise_cpu"],
            b["tasks"],
            None,
            s_z=s_z,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
        )
        m_self, _ = decode_and_score_with_phases(
            policy,
            z_self_hat,
            zq,
            zq,
            b["gt_chunks"],
            b["valid_16"],
            b["init_noise_cpu"],
            b["tasks"],
            None,
            s_z=s_z,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
        )

        geom = compute_coeff_geometry(c_corr, c_p0, c_e10c, s_z=s_z)
        fit_p0 = _coeff_fit_metrics(c_corr, c_p0)
        fit_opt_pri = _coeff_fit_metrics(c_corr, oc[primary_oracle_key])
        fit_opt_tight = (
            _coeff_fit_metrics(c_corr, oc[tight_oracle_key])
            if (tight_oracle_key is not None and tight_oracle_key in oc)
            else None
        )

        wrong_delta = float(m_wrg["ddim_10d_mse"] - m_corr["ddim_10d_mse"])
        wrong_ratio = float(m_wrg["ddim_10d_mse"] / max(m_corr["ddim_10d_mse"], 1e-12))
        self_c_l2 = float(np.mean(np.linalg.norm(c_self.astype(np.float64), axis=-1)))
        wrg_c_l2 = float(np.mean(np.linalg.norm(c_wrg.astype(np.float64), axis=-1)))

        rec = {
            "split": sname,
            "model": model_id,
            "label": label,
            "correct": m_corr,
            "wrong_task": m_wrg,
            "self_preservation": m_self,
            "wrong_delta_mse10": wrong_delta,
            "wrong_ratio_mse10": wrong_ratio,
            "wrong_c_l2_mean": wrg_c_l2,
            "self_c_l2_mean": self_c_l2,
            "geometry": geom,
            "fit_vs_c_p0": fit_p0,
            "fit_vs_c_opt_primary": fit_opt_pri,
            "fit_vs_c_opt_tight": fit_opt_tight,
            "phases": ph_corr,
        }
        res_by_split[sname] = rec

        flat_row: Dict[str, Any] = {
            "split": sname,
            "model": model_id,
            "label": label,
            "ddim_10d_mse": m_corr["ddim_10d_mse"],
            "ddim_10d_mse_macro": m_corr["ddim_10d_mse_macro"],
            "ddim_pos_mse": m_corr["ddim_pos_mse"],
            "ddim_rot6d_mse": m_corr["ddim_rot6d_mse"],
            "ddim_grip_mse": m_corr["ddim_grip_mse"],
            "ddim_7d_mae": m_corr["ddim_7d_mae"],
            "exec12_7d_mae": m_corr["exec12_7d_mae"],
            "grip_acc16": m_corr["grip_acc16"],
            "latent_lz": m_corr["latent_lz"],
            "c_l2_mean": geom["c_l2_mean"],
            "delta_z_rms": geom["delta_z_rms"],
            "c_norm_ratio_vs_p0": geom["c_norm_ratio_vs_p0"],
            "c_erank_energy": geom["c_erank_energy"],
            "coeff_mse_vs_c_opt_primary": fit_opt_pri["coeff_mse"],
            "coeff_r2_vs_c_opt_primary": fit_opt_pri["coeff_r2"],
            "coeff_cos_vs_c_opt_primary": fit_opt_pri["coeff_cos"],
            "coeff_mse_vs_c_opt_tight": fit_opt_tight["coeff_mse"] if fit_opt_tight else None,
            "coeff_r2_vs_c_opt_tight": fit_opt_tight["coeff_r2"] if fit_opt_tight else None,
            "coeff_cos_vs_c_opt_tight": fit_opt_tight["coeff_cos"] if fit_opt_tight else None,
            "coeff_mse_vs_c_p0": fit_p0["coeff_mse"],
            "coeff_r2_vs_c_p0": fit_p0["coeff_r2"],
            "coeff_cos_vs_c_p0": fit_p0["coeff_cos"],
            "wrong_ddim_10d_mse": m_wrg["ddim_10d_mse"],
            "wrong_delta_mse10": wrong_delta,
            "wrong_ratio_mse10": wrong_ratio,
            "wrong_c_l2_mean": wrg_c_l2,
            "self_d3_ddim_10d_mse": m_self["ddim_10d_mse"],
            "self_d3_latent_lz": m_self["latent_lz"],
            "self_c_l2_mean": self_c_l2,
        }
        flat_rows.append(flat_row)

        for ph_name, ph_m in ph_corr.items():
            phase_rows.append(
                {
                    "split": sname,
                    "model": model_id,
                    "label": label,
                    "phase": ph_name,
                    "count": ph_m["count"],
                    "ddim_10d_mse": ph_m["ddim_10d_mse"],
                    "ddim_pos_mse": ph_m["ddim_pos_mse"],
                    "ddim_rot6d_mse": ph_m["ddim_rot6d_mse"],
                    "ddim_grip_mse": ph_m["ddim_grip_mse"],
                    "ddim_7d_mae": ph_m["ddim_7d_mae"],
                    "grip_acc16": ph_m["grip_acc16"],
                    "latent_lz": ph_m["latent_lz"],
                    "c_l2_mean": ph_m.get("c_l2_mean"),
                    "cos_vs_c_p0": ph_m.get("cos_vs_c_p0"),
                }
            )

        print(
            f"  [{sname:12s} | {model_id:20s}] "
            f"DDIM10={m_corr['ddim_10d_mse']:.5f} | MAE7={m_corr['ddim_7d_mae']:.5f} | "
            f"Grip={100*m_corr['grip_acc16']:.2f}% | ||c||_2={geom['c_l2_mean']:5.2f} | "
            f"cMSE(opt)={fit_opt_pri['coeff_mse']:5.2f} (R2={fit_opt_pri['coeff_r2']:.3f}, cos={fit_opt_pri['coeff_cos']:.3f}) | "
            f"cMSE(p0)={fit_p0['coeff_mse']:5.2f} | "
            f"Wrong={m_wrg['ddim_10d_mse']:.5f} ({wrong_ratio:.2f}x, Δ={wrong_delta:+.5f}) | "
            f"Self_cL2={self_c_l2:5.2f} (D3={m_self['ddim_10d_mse']:.5f})",
            flush=True,
        )

    return res_by_split, flat_rows, phase_rows


def run_train_and_eval(
    *,
    out_dir: Path = OUTPUT_ROOT / "learnability",
    oracle_dir: Path = OUTPUT_ROOT / "oracle",
    lam_primary: float = 1e-5,
    lam_tight: Optional[float] = 3e-5,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 5e-4,
    seed: int = 42,
    ddim_seed: int = 12345,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "caches"

    tag_pri = lam_to_tag(lam_primary)
    cache_pri_path = cache_dir / f"train_val_c_opt_{tag_pri}.npz"
    if not cache_pri_path.is_file():
        raise FileNotFoundError(f"Missing primary c* cache: {cache_pri_path}")
    cache_pri = dict(np.load(cache_pri_path, allow_pickle=False))
    meta_pri = json.loads((cache_dir / f"train_val_c_opt_{tag_pri}_meta.json").read_text())

    cache_tight = None
    tag_tight = None
    meta_tight = None
    if lam_tight is not None:
        tag_tight = lam_to_tag(lam_tight)
        cache_tight_path = cache_dir / f"train_val_c_opt_{tag_tight}.npz"
        if cache_tight_path.is_file():
            cache_tight = dict(np.load(cache_tight_path, allow_pickle=False))
            meta_tight = json.loads(
                (cache_dir / f"train_val_c_opt_{tag_tight}_meta.json").read_text()
            )

    policy, u64_t, model_meta = load_frozen_e10c_and_basis(device=device)
    u64_np = u64_t.cpu().numpy()
    s_z = float(model_meta["s_z"])

    # Load exact untouched seen_val (3,000) & unseen_chain (3,000) WITH wrong-task demos
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
    pri_oracle_key = f"action_opt_{tag_pri}"
    tight_oracle_key = f"action_opt_{tag_tight}" if tag_tight is not None else None

    # Train lightweight models strictly on train_fit, selecting ckpt on val_ckpt
    trained_models: Dict[str, Tuple[str, LightweightLowRankTransfer, Dict[str, Any]]] = {}

    m_ag_pri, meta_ag_pri = train_lightweight_transfer(
        run_name=f"ag_light_{tag_pri}",
        run_dir=out_dir / f"ag_light_{tag_pri}",
        target_key="c_opt",
        cache_npz=cache_pri,
        policy=policy,
        u64_np=u64_np,
        s_z=s_z,
        d_model=256,
        num_layers=2,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.15,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        use_bf16=use_bf16,
    )
    trained_models[f"ag_light_{tag_pri}"] = (
        f"Action-Guided Light (λ={lam_primary:.0e}, 2.5M)",
        m_ag_pri,
        meta_ag_pri,
    )

    # Medium-capacity check (d=384, L=3, ~7.0M) on same primary c* to separate capacity vs signal
    m_ag_med, meta_ag_med = train_lightweight_transfer(
        run_name=f"ag_med_{tag_pri}",
        run_dir=out_dir / f"ag_med_{tag_pri}",
        target_key="c_opt",
        cache_npz=cache_pri,
        policy=policy,
        u64_np=u64_np,
        s_z=s_z,
        d_model=384,
        num_layers=3,
        num_heads=6,
        ffn_dim=1536,
        dropout=0.20,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr * 0.7,
        seed=seed + 1,
        device=device,
        use_bf16=use_bf16,
    )
    trained_models[f"ag_med_{tag_pri}"] = (
        f"Action-Guided Med (λ={lam_primary:.0e}, 7.0M)",
        m_ag_med,
        meta_ag_med,
    )

    if cache_tight is not None and lam_tight is not None and tag_tight is not None:
        m_ag_tight, meta_ag_tight = train_lightweight_transfer(
            run_name=f"ag_light_{tag_tight}",
            run_dir=out_dir / f"ag_light_{tag_tight}",
            target_key="c_opt",
            cache_npz=cache_tight,
            policy=policy,
            u64_np=u64_np,
            s_z=s_z,
            d_model=256,
            num_layers=2,
            num_heads=4,
            ffn_dim=1024,
            dropout=0.15,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + 2,
            device=device,
            use_bf16=use_bf16,
        )
        trained_models[f"ag_light_{tag_tight}"] = (
            f"Action-Guided Light (λ={lam_tight:.0e}, 2.5M)",
            m_ag_tight,
            meta_ag_tight,
        )

    # Matched-data Future-Latent Regression control (same 32k pairs, same 2.5M arch, predicting c_p0)
    m_p0_ctrl, meta_p0_ctrl = train_lightweight_transfer(
        run_name="p0_reg_light_ctrl",
        run_dir=out_dir / "p0_reg_light_ctrl",
        target_key="c_p0",
        cache_npz=cache_pri,
        policy=policy,
        u64_np=u64_np,
        s_z=s_z,
        d_model=256,
        num_layers=2,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.15,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed + 3,
        device=device,
        use_bf16=use_bf16,
    )
    trained_models["p0_reg_light_ctrl"] = (
        "Future-Latent Reg Ctrl (c_p0, 2.5M)",
        m_p0_ctrl,
        meta_p0_ctrl,
    )

    # Evaluate baselines + all trained models on untouched seen_val & unseen_chain
    print("\n================ [E11-Learnability Evaluation (Seen & Unseen)] ================", flush=True)
    eval_specs: List[Tuple[str, str, Any]] = [
        ("identity", "Identity (c=0)", lambda zs, hq: np.zeros((len(zs), RANK_K), dtype=np.float32)),
        (
            "e10_c",
            "E10-C (Full 1.37M, 28.6M)",
            lambda zs, hq: _predict_coeffs(
                policy.transfer, zs, hq, batch_size=1024, device=device, use_bf16=use_bf16
            ),
        ),
    ]
    for mid, (lbl, mod, _) in trained_models.items():
        eval_specs.append(
            (
                mid,
                lbl,
                lambda zs, hq, _m=mod: _predict_coeffs(
                    _m, zs, hq, batch_size=1024, device=device, use_bf16=use_bf16
                ),
            )
        )

    all_nested: Dict[str, Any] = {}
    all_flat: List[Dict[str, Any]] = []
    all_phase: List[Dict[str, Any]] = []

    # Also compute Self D1 (Full z_query) reference for D3/D1 ratio and ARR
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
            primary_oracle_key=pri_oracle_key,
            tight_oracle_key=tight_oracle_key,
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

    # Attach ARR and D3/D1 self-preservation ratio
    for r in all_flat:
        sn = r["split"]
        arr, _ = _safe_arr(id_mse_by_split[sn], r["ddim_10d_mse"], full_d1_by_split[sn])
        r["arr_10d_vs_id_full"] = arr
        r["self_d3_over_d1_ratio"] = float(
            r["self_d3_ddim_10d_mse"] / max(full_d1_by_split[sn], 1e-12)
        )

    config = {
        "experiment": "e11_pilot_learnability_verification",
        "model_meta": model_meta,
        "lam_primary": float(lam_primary),
        "lam_tight": float(lam_tight) if lam_tight is not None else None,
        "primary_cache_meta": meta_pri,
        "tight_cache_meta": meta_tight,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "ddim_seed": int(ddim_seed),
        "full_self_d1_ddim10": full_d1_by_split,
        "identity_ddim10": id_mse_by_split,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }

    summary = {
        "config": config,
        "train_runs": {k: v[2] for k, v in trained_models.items()},
        "evaluation": all_nested,
    }
    atomic_write_json(config, out_dir / "config.json")
    atomic_write_json(summary, out_dir / "learnability_summary.json")
    write_csv(all_flat, out_dir / "learnability_summary.csv")
    write_csv(all_phase, out_dir / "learnability_phase_breakdown.csv")

    print(
        f"\n[E11-Learnability DONE] Total elapsed: {config['elapsed_seconds']}s -> {out_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("gen-cache", "train-eval"),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "learnability")
    parser.add_argument("--oracle-dir", type=Path, default=OUTPUT_ROOT / "oracle")
    parser.add_argument("--lam", type=float, default=1e-5, help="Lambda for gen-cache")
    parser.add_argument("--lam-primary", type=float, default=1e-5)
    parser.add_argument("--lam-tight", type=float, default=3e-5)
    parser.add_argument("--n-train-fit", type=int, default=32000)
    parser.add_argument("--n-val-ckpt", type=int, default=2000)
    parser.add_argument("--opt-steps", type=int, default=250)
    parser.add_argument("--opt-lr", type=float, default=1.0)
    parser.add_argument("--mc-timesteps", type=int, default=4)
    parser.add_argument("--opt-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "gen-cache":
        generate_train_val_c_cache(
            lam=args.lam,
            out_dir=args.out_dir / "caches",
            n_train_fit=args.n_train_fit,
            n_val_ckpt=args.n_val_ckpt,
            steps=args.opt_steps,
            lr=args.opt_lr,
            mc_timesteps=args.mc_timesteps,
            batch_size=args.opt_batch_size,
            seed=1101,
            device=device,
            use_bf16=not args.no_bf16,
        )
    else:
        run_train_and_eval(
            out_dir=args.out_dir,
            oracle_dir=args.oracle_dir,
            lam_primary=args.lam_primary,
            lam_tight=args.lam_tight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
            use_bf16=not args.no_bf16,
        )


if __name__ == "__main__":
    main()
