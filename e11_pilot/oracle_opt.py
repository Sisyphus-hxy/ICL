"""Stage 1-3 of E11-Pilot: Oracle Adaptation Optimization & Multi-Baseline Comparison.

Optimizes ONLY 64-dim coefficients `c` per Demo-Query pair:
    c* = argmin_c [ L_diff(G(z_src + U_64 c), a_q*) + lambda * ||c||_2^2 ]
using fresh stratified multi-timestep diffusion training loss (never backpropagating
through DDIM sampling), initialized at c = 0.

Independently evaluates via fixed-noise 16-step DDIM Action MSE (primary P0/E10 seed +
independent secondary DDIM seed) and compares:
  - Identity (`c = 0`)
  - E10-C (frozen learned coefficient)
  - P0 Oracle (`c = U_64^T (z_query - z_src)`)
  - Full Self-Latent (`z_query`)
  - Action-Optimized Oracle (`c*(lambda)` across lambda sweep)
overall and across all 5 behavioral phases on `seen_val` (3,000) and `unseen_chain` (3,000).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from p0_lowrank.run_diagnosis import _safe_arr

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    DEFAULT_LAMBDA_GRID,
    EMBED_DIM,
    EXEC_HORIZON,
    LATENT_DIM,
    LATENT_TOKENS,
    OUTPUT_ROOT,
    RANK_K,
    SEMANTIC_PHASES,
)
from .data import load_eval_splits, load_frozen_e10c_and_basis

ACTION_10D_NAMES = (
    "dx",
    "dy",
    "dz",
    "rot0",
    "rot1",
    "rot2",
    "rot3",
    "rot4",
    "rot5",
    "grip",
)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_coeff_effective_rank(c_mat: np.ndarray) -> Dict[str, float]:
    """Compute effective rank (energy entropy & participation ratio) of coefficient matrix (N, K)."""
    X = np.asarray(c_mat, dtype=np.float64)
    tot = float(np.sum(X * X))
    if tot <= 1e-18:
        return {"erank_energy": 0.0, "participation_ratio": 0.0, "d_80": 0.0, "d_90": 0.0}
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    eig = S * S
    p = eig / max(float(np.sum(eig)), 1e-18)
    p_pos = p[p > 1e-18]
    erank = float(math.exp(-float(np.sum(p_pos * np.log(p_pos)))))
    pr = float((float(np.sum(eig)) ** 2) / max(float(np.sum(eig * eig)), 1e-30))
    cum = np.cumsum(p)
    d80 = int(min(np.searchsorted(cum, 0.80) + 1, len(S)))
    d90 = int(min(np.searchsorted(cum, 0.90) + 1, len(S)))
    return {
        "erank_energy": erank,
        "participation_ratio": pr,
        "d_80": float(d80),
        "d_90": float(d90),
    }


def compute_coeff_geometry(
    c_mat: np.ndarray,
    c_p0: np.ndarray,
    c_e10c: Optional[np.ndarray],
    *,
    s_z: float,
) -> Dict[str, float]:
    """Compute latent modification magnitude & directional alignment metrics for coefficients `c`."""
    c64 = np.asarray(c_mat, dtype=np.float64)
    p064 = np.asarray(c_p0, dtype=np.float64)
    norms = np.linalg.norm(c64, axis=-1)  # == ||U_64 c||_2 since U_64 is orthonormal
    sq_norms = np.sum(c64 * c64, axis=-1)
    p0_norms = np.linalg.norm(p064, axis=-1)

    dz_rms_per = norms / math.sqrt(float(LATENT_DIM))
    dz_rms_global = math.sqrt(float(np.mean(sq_norms)) / float(LATENT_DIM))

    # Cosine similarity vs c_p0 (0 when c == 0)
    dot_p0 = np.sum(c64 * p064, axis=-1)
    denom_p0 = norms * p0_norms
    valid_cos = denom_p0 > 1e-12
    cos_p0 = float(np.mean(dot_p0[valid_cos] / denom_p0[valid_cos])) if np.any(valid_cos) else 0.0
    dist_p0 = float(np.mean(np.linalg.norm(c64 - p064, axis=-1)))
    mse_vs_p0 = float(np.mean((c64 - p064) ** 2))

    out = {
        "c_l2_mean": float(np.mean(norms)),
        "c_l2_median": float(np.median(norms)),
        "c_l2_p90": float(np.percentile(norms, 90)),
        "c_l2_sq_mean": float(np.mean(sq_norms)),
        "delta_z_rms": float(dz_rms_global),
        "delta_z_rms_per_sample_mean": float(np.mean(dz_rms_per)),
        "delta_z_rms_over_sz": float(dz_rms_global / max(s_z, 1e-12)),
        "c_norm_ratio_vs_p0": float(np.mean(norms) / max(float(np.mean(p0_norms)), 1e-12)),
        "c_energy_ratio_vs_p0": float(
            np.mean(sq_norms) / max(float(np.mean(np.sum(p064 * p064, axis=-1))), 1e-12)
        ),
        "cos_vs_c_p0": cos_p0,
        "l2_dist_vs_c_p0": dist_p0,
        "coeff_mse_vs_c_p0": mse_vs_p0,
    }
    if c_e10c is not None:
        e64 = np.asarray(c_e10c, dtype=np.float64)
        e_norms = np.linalg.norm(e64, axis=-1)
        dot_e = np.sum(c64 * e64, axis=-1)
        denom_e = norms * e_norms
        ve = denom_e > 1e-12
        out["cos_vs_c_e10c"] = float(np.mean(dot_e[ve] / denom_e[ve])) if np.any(ve) else 0.0
        out["coeff_mse_vs_c_e10c"] = float(np.mean((c64 - e64) ** 2))

    er = compute_coeff_effective_rank(c64)
    out["c_erank_energy"] = er["erank_energy"]
    out["c_participation_ratio"] = er["participation_ratio"]
    out["c_d80"] = er["d_80"]
    out["c_d90"] = er["d_90"]
    return out


@torch.inference_mode()
def decode_and_score_with_phases(
    policy: torch.nn.Module,
    z_hat: np.ndarray,
    z_query: np.ndarray,
    z_source: np.ndarray,
    gt_chunks: np.ndarray,
    valid_16: np.ndarray,
    init_noise_cpu: torch.Tensor,
    tasks: Sequence[str],
    phases: Optional[Sequence[str]] = None,
    *,
    c_mat: Optional[np.ndarray] = None,
    c_p0: Optional[np.ndarray] = None,
    s_z: float,
    device: torch.device,
    use_bf16: bool = True,
    batch_size: int = 1024,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Decode `z_hat` via frozen Action Decoder with deterministic DDIM init_noise,
    returning both overall metrics and per-behavioral-phase breakdown."""
    N = len(z_hat)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    zq64 = z_query.astype(np.float64)
    zs64 = z_source.astype(np.float64)
    zh64 = z_hat.astype(np.float64)
    err_z = zh64 - zq64
    res_z = zq64 - zs64
    dz_mod = zh64 - zs64

    per_sample_lat_mse = np.mean(err_z * err_z, axis=-1)
    per_sample_lz = per_sample_lat_mse / (s_z * s_z)
    per_sample_mod_l2 = np.linalg.norm(dz_mod, axis=-1)
    zh_norm = np.linalg.norm(zh64, axis=-1)
    zq_norm = np.linalg.norm(zq64, axis=-1)
    per_sample_cos = np.sum(zh64 * zq64, axis=-1) / np.maximum(zh_norm * zq_norm, 1e-12)

    per_sample_v16 = np.empty(N, dtype=np.float64)
    per_sample_v12 = np.empty(N, dtype=np.float64)
    per_sample_sq_dim = np.empty((N, ACTION_DIM_10D), dtype=np.float64)
    per_sample_abs7 = np.empty(N, dtype=np.float64)
    per_sample_exec12_abs7 = np.empty(N, dtype=np.float64)
    per_sample_grip16_ok = np.empty(N, dtype=np.float64)
    per_sample_grip12_ok = np.empty(N, dtype=np.float64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        z_t = torch.from_numpy(
            z_hat[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)
        ).to(device=device, dtype=torch.float32)
        noise_t = init_noise_cpu[start:end].to(device=device, dtype=torch.float32)

        with amp_ctx:
            pred_10d = policy.teacher.sample_from_z_self(z_t, init_noise=noise_t).float()

        gt_t = torch.from_numpy(gt_chunks[start:end]).to(device=device, dtype=torch.float32)
        v16_t = torch.from_numpy(valid_16[start:end]).to(device=device, dtype=torch.float32)
        v12_t = v16_t[:, :EXEC_HORIZON]

        sq10 = (pred_10d - gt_t).square()  # (B, 16, 10)
        per_sample_sq_dim[start:end] = (sq10 * v16_t.unsqueeze(-1)).sum(dim=1).cpu().numpy()
        per_sample_v16[start:end] = v16_t.sum(dim=1).cpu().numpy()
        per_sample_v12[start:end] = v12_t.sum(dim=1).cpu().numpy()

        pred_7d = action_10d_to_7d(pred_10d)
        gt_7d = action_10d_to_7d(gt_t)
        abs7_step = (pred_7d - gt_7d).abs().mean(dim=-1)
        grip_ok = ((pred_7d[..., 6] >= 0.0) == (gt_7d[..., 6] >= 0.0)).float()

        per_sample_abs7[start:end] = (abs7_step * v16_t).sum(dim=1).cpu().numpy()
        per_sample_exec12_abs7[start:end] = (abs7_step[:, :EXEC_HORIZON] * v12_t).sum(dim=1).cpu().numpy()
        per_sample_grip16_ok[start:end] = (grip_ok * v16_t).sum(dim=1).cpu().numpy()
        per_sample_grip12_ok[start:end] = (grip_ok[:, :EXEC_HORIZON] * v12_t).sum(dim=1).cpu().numpy()

    def _agg(idxs: np.ndarray) -> Dict[str, Any]:
        tot_v16 = max(float(np.sum(per_sample_v16[idxs])), 1.0)
        tot_v12 = max(float(np.sum(per_sample_v12[idxs])), 1.0)
        per_dim = np.sum(per_sample_sq_dim[idxs], axis=0) / tot_v16

        task_v: Dict[str, float] = defaultdict(float)
        task_sq: Dict[str, float] = defaultdict(float)
        for ii in idxs:
            tk = tasks[int(ii)]
            task_v[tk] += float(per_sample_v16[ii])
            task_sq[tk] += float(np.mean(per_sample_sq_dim[ii]))
        macro_mse = float(
            np.mean([task_sq[t] / max(task_v[t], 1.0) for t in sorted(task_v.keys())])
        )

        res_sq = float(np.sum(res_z[idxs] ** 2))
        err_sq = float(np.sum(err_z[idxs] ** 2))
        block: Dict[str, Any] = {
            "count": int(len(idxs)),
            "ddim_10d_mse": float(np.mean(per_dim)),
            "ddim_10d_mse_macro": macro_mse,
            "ddim_pos_mse": float(np.mean(per_dim[:3])),
            "ddim_rot6d_mse": float(np.mean(per_dim[3:9])),
            "ddim_grip_mse": float(per_dim[9]),
            "ddim_7d_mae": float(np.sum(per_sample_abs7[idxs]) / tot_v16),
            "exec12_7d_mae": float(np.sum(per_sample_exec12_abs7[idxs]) / tot_v12),
            "grip_acc16": float(np.sum(per_sample_grip16_ok[idxs]) / tot_v16),
            "grip_acc12": float(np.sum(per_sample_grip12_ok[idxs]) / tot_v12),
            "latent_mse_raw": float(np.mean(per_sample_lat_mse[idxs])),
            "latent_lz": float(np.mean(per_sample_lz[idxs])),
            "latent_cos": float(np.mean(per_sample_cos[idxs])),
            "residual_energy_recovery": float(1.0 - err_sq / max(res_sq, 1e-18)),
            "delta_z_l2_mean": float(np.mean(per_sample_mod_l2[idxs])),
            "delta_z_rms": float(
                math.sqrt(float(np.mean(per_sample_mod_l2[idxs] ** 2)) / float(LATENT_DIM))
            ),
            "per_dim_mse": {
                ACTION_10D_NAMES[d]: float(per_dim[d]) for d in range(ACTION_DIM_10D)
            },
        }
        if c_mat is not None and c_p0 is not None:
            cm = c_mat[idxs].astype(np.float64)
            cp = c_p0[idxs].astype(np.float64)
            cn = np.linalg.norm(cm, axis=-1)
            pn = np.linalg.norm(cp, axis=-1)
            denom = cn * pn
            ok = denom > 1e-12
            block["c_l2_mean"] = float(np.mean(cn))
            block["cos_vs_c_p0"] = (
                float(np.mean(np.sum(cm[ok] * cp[ok], axis=-1) / denom[ok]))
                if np.any(ok)
                else 0.0
            )
        return block

    overall = _agg(np.arange(N, dtype=np.int64))
    phase_dict: Dict[str, Dict[str, Any]] = {}
    if phases is not None:
        phases_arr = np.asarray(phases)
        for ph in SEMANTIC_PHASES:
            idxs = np.flatnonzero(phases_arr == ph)
            if len(idxs) > 0:
                phase_dict[ph] = _agg(idxs)
    return overall, phase_dict


def optimize_coefficients_batch(
    teacher: torch.nn.Module,
    z_src_np: np.ndarray,
    gt_actions_np: np.ndarray,
    u64_device: torch.Tensor,
    *,
    lam: float,
    steps: int = 250,
    lr: float = 1.0,
    eta_min_ratio: float = 0.05,
    mc_timesteps: int = 4,
    batch_size: int = 1024,
    seed: int = 1100,
    device: torch.device,
    use_bf16: bool = True,
    log_steps: Sequence[int] = (0, 10, 25, 50, 75, 100, 150, 200, 250),
    ddim_track_steps: Sequence[int] = (),
    ddim_track_n: int = 512,
    ddim_init_noise_cpu: Optional[torch.Tensor] = None,
    ddim_valid_16: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Optimize 64-dim coefficients `c` (initialized at 0) for every pair in `z_src_np`
    by minimizing:
        L_diff(G(z_src + U_64 c), a_q*) + lam * ||c||_2^2
    with fresh stratified multi-timestep noise at every step (no single-noise overfitting,
    strictly no gradient through DDIM sampling, frozen teacher).
    """
    N = len(z_src_np)
    c_out = np.empty((N, RANK_K), dtype=np.float32)

    teacher.eval()
    for p in teacher.parameters():
        if p.requires_grad:
            raise RuntimeError("Teacher parameters must remain strictly frozen during Oracle opt!")

    num_train_t = int(teacher.noise_scheduler.config.num_train_timesteps)
    t_edges = np.linspace(0, num_train_t, mc_timesteps + 1, dtype=int)
    log_set = set(int(s) for s in log_steps)
    track_set = set(int(s) for s in ddim_track_steps)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Accumulate weighted averages across batches for each logged step
    curve_acc: Dict[int, Dict[str, float]] = {
        s: {
            "n": 0.0,
            "diff_loss": 0.0,
            "reg_loss": 0.0,
            "total_obj": 0.0,
            "c_l2": 0.0,
            "c_l2_sq": 0.0,
            "grad_norm": 0.0,
        }
        for s in sorted(log_set)
    }
    ddim_track_acc: Dict[int, Tuple[float, float]] = {s: (0.0, 0.0) for s in sorted(track_set)}

    gen = torch.Generator(device=device).manual_seed(int(seed))
    M = int(mc_timesteps)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        zsrc_b = torch.from_numpy(z_src_np[start:end]).to(device=device, dtype=torch.float32)
        act_b = torch.from_numpy(gt_actions_np[start:end]).to(device=device, dtype=torch.float32)
        act_rep = act_b.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, ACTION_HORIZON, ACTION_DIM_10D)

        c = torch.zeros((B, RANK_K), device=device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([c], lr=float(lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(steps), eta_min=float(lr) * float(eta_min_ratio)
        )

        # Helper to record step metrics (step 0 before any update, or after update)
        def _record_ddim_track(step_idx: int) -> None:
            if (
                step_idx not in track_set
                or ddim_init_noise_cpu is None
                or ddim_valid_16 is None
                or start >= ddim_track_n
            ):
                return
            take = min(B, ddim_track_n - start)
            with torch.inference_mode(), amp_ctx:
                z_cur = (zsrc_b[:take] + c.detach()[:take] @ u64_device.T).reshape(
                    take, LATENT_TOKENS, EMBED_DIM
                )
                n_cur = ddim_init_noise_cpu[start : start + take].to(
                    device=device, dtype=torch.float32
                )
                pred_10d = teacher.sample_from_z_self(z_cur, init_noise=n_cur).float()
            v16 = torch.from_numpy(ddim_valid_16[start : start + take]).to(
                device=device, dtype=torch.float32
            )
            sq10 = ((pred_10d - act_b[:take]).square().mean(dim=-1) * v16).sum().item()
            v_sum = v16.sum().item()
            prev_sq, prev_v = ddim_track_acc[step_idx]
            ddim_track_acc[step_idx] = (prev_sq + sq10, prev_v + v_sum)

        # Log step 0 (c = 0)
        if 0 in log_set:
            with torch.no_grad():
                ts0 = torch.stack(
                    [
                        torch.randint(
                            int(t_edges[m]),
                            int(t_edges[m + 1]),
                            (B,),
                            device=device,
                            generator=gen,
                        )
                        for m in range(M)
                    ],
                    dim=1,
                ).reshape(-1)
                eps0 = torch.randn(act_rep.shape, device=device, dtype=torch.float32, generator=gen)
                p_eps0 = torch.zeros_like(eps0)
                with amp_ctx:
                    z0_rep = zsrc_b.unsqueeze(1).expand(-1, M, -1).reshape(B * M, LATENT_TOKENS, EMBED_DIM)
                    dl0 = (
                        teacher.diffusion_loss_from_z_self(
                            z0_rep, act_rep, noise=eps0, pertub_noise=p_eps0, timesteps=ts0
                        )
                        .reshape(B, M)
                        .mean(dim=1)
                    )
                acc = curve_acc[0]
                acc["n"] += float(B)
                acc["diff_loss"] += float(dl0.sum().item())
                acc["reg_loss"] += 0.0
                acc["total_obj"] += float(dl0.sum().item())
                acc["c_l2"] += 0.0
                acc["c_l2_sq"] += 0.0
                acc["grad_norm"] += 0.0
        _record_ddim_track(0)

        for step in range(1, int(steps) + 1):
            opt.zero_grad(set_to_none=True)
            ts = torch.stack(
                [
                    torch.randint(
                        int(t_edges[m]),
                        int(t_edges[m + 1]),
                        (B,),
                        device=device,
                        generator=gen,
                    )
                    for m in range(M)
                ],
                dim=1,
            ).reshape(-1)
            eps = torch.randn(act_rep.shape, device=device, dtype=torch.float32, generator=gen)
            p_eps = torch.zeros_like(eps)

            with amp_ctx:
                z_hat = (zsrc_b + c @ u64_device.T).unsqueeze(1).expand(-1, M, -1).reshape(
                    B * M, LATENT_TOKENS, EMBED_DIM
                )
                diff_l = (
                    teacher.diffusion_loss_from_z_self(
                        z_hat, act_rep, noise=eps, pertub_noise=p_eps, timesteps=ts
                    )
                    .reshape(B, M)
                    .mean(dim=1)
                )
            c_sq = c.square().sum(dim=-1)
            reg_l = float(lam) * c_sq
            total_per = diff_l + reg_l
            total_per.sum().backward()

            g_norm = float(c.grad.detach().norm(dim=-1).sum().item()) if c.grad is not None else 0.0
            opt.step()
            sched.step()

            if step in log_set:
                with torch.no_grad():
                    c_det = c.detach()
                    c_sq_det = c_det.square().sum(dim=-1)
                    c_l2_det = torch.sqrt(c_sq_det + 1e-12)
                    reg_det = float(lam) * c_sq_det
                    acc = curve_acc[step]
                    acc["n"] += float(B)
                    acc["diff_loss"] += float(diff_l.detach().sum().item())
                    acc["reg_loss"] += float(reg_det.sum().item())
                    acc["total_obj"] += float((diff_l.detach() + reg_det).sum().item())
                    acc["c_l2"] += float(c_l2_det.sum().item())
                    acc["c_l2_sq"] += float(c_sq_det.sum().item())
                    acc["grad_norm"] += g_norm
            _record_ddim_track(step)

        c_out[start:end] = c.detach().cpu().numpy()

    curve_rows: List[Dict[str, Any]] = []
    for step in sorted(log_set):
        acc = curve_acc[step]
        n = max(acc["n"], 1.0)
        c_l2_sq_m = acc["c_l2_sq"] / n
        row: Dict[str, Any] = {
            "lambda": float(lam),
            "step": int(step),
            "diff_loss": float(acc["diff_loss"] / n),
            "reg_loss": float(acc["reg_loss"] / n),
            "total_obj": float(acc["total_obj"] / n),
            "c_l2_mean": float(acc["c_l2"] / n),
            "c_l2_sq_mean": float(c_l2_sq_m),
            "delta_z_rms": float(math.sqrt(c_l2_sq_m / float(LATENT_DIM))),
            "grad_norm_mean": float(acc["grad_norm"] / n),
        }
        if step in track_set and ddim_track_acc[step][1] > 0:
            row["ddim_track_10d_mse"] = float(ddim_track_acc[step][0] / ddim_track_acc[step][1])
        else:
            row["ddim_track_10d_mse"] = None
        curve_rows.append(row)

    return c_out, curve_rows


def run_oracle_experiment(
    *,
    out_dir: Path = OUTPUT_ROOT / "oracle",
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    n_seen_val: int = 3000,
    n_unseen: int = 3000,
    steps: int = 250,
    lr: float = 1.0,
    mc_timesteps: int = 4,
    batch_size: int = 1024,
    seed: int = 42,
    ddim_seed: int = 12345,
    ddim_seed_alt: int = 54321,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, u64_t, model_meta = load_frozen_e10c_and_basis(device=device)
    u64_np = u64_t.cpu().numpy()
    s_z = float(model_meta["s_z"])

    print(
        f"[E11-Oracle] Loaded frozen E9+E10-C & P0 U_64 "
        f"(ortho_err={model_meta['basis_ortho_err']:.2e}, diff_vs_e10c={model_meta['basis_vs_e10c_max_abs_diff']:.2e}, s_z={s_z:.6f})",
        flush=True,
    )

    splits = load_eval_splits(
        policy,
        n_seen_val=n_seen_val,
        n_unseen=n_unseen,
        seed=seed,
        ddim_seed=ddim_seed,
        ddim_seed_alt=ddim_seed_alt,
        batch_size=batch_size,
        device=device,
        use_bf16=use_bf16,
        include_wrong_demos=False,
    )

    summary_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []
    curve_rows_all: List[Dict[str, Any]] = []
    nested_results: Dict[str, Any] = {}

    for split_idx, split_name in enumerate(("seen_val", "unseen_chain")):
        b = splits[split_name]
        zq = b["z_query"]
        zsrc = b["z_src_reader"]
        c_e10c = b["c_e10c"]
        c_zero = np.zeros_like(c_e10c)
        c_p0 = ((zq - zsrc) @ u64_np).astype(np.float32)

        print(f"\n================ [E11-Oracle: {split_name} (N={len(zq)})] ================", flush=True)
        # Phase distribution
        ph_counts = {ph: int(sum(1 for x in b["phases"] if x == ph)) for ph in SEMANTIC_PHASES}
        print(f"  Phase counts: {ph_counts}", flush=True)

        # 1. Fixed baselines: Identity, E10-C, P0 Oracle, Full z_query
        baselines_spec = [
            ("identity", "Identity (c=0)", None, c_zero, zsrc),
            ("e10_c", "E10-C (Learned K=64)", None, c_e10c, zsrc + c_e10c @ u64_np.T),
            ("p0_oracle", "P0 Oracle (U_64^T (z_q - z_src))", None, c_p0, zsrc + c_p0 @ u64_np.T),
            ("full_self", "Full Self-Latent (z_query)", None, None, zq),
        ]

        split_method_res: Dict[str, Any] = {}
        saved_c: Dict[str, np.ndarray] = {
            "c_p0": c_p0,
            "c_e10c": c_e10c,
        }

        for method_id, label, lam_val, c_mat, z_hat in baselines_spec:
            ov, ph_d = decode_and_score_with_phases(
                policy,
                z_hat,
                zq,
                zsrc,
                b["gt_chunks"],
                b["valid_16"],
                b["init_noise_cpu"],
                b["tasks"],
                b["phases"],
                c_mat=c_mat,
                c_p0=c_p0,
                s_z=s_z,
                device=device,
                use_bf16=use_bf16,
                batch_size=batch_size,
            )
            ov_alt, _ = decode_and_score_with_phases(
                policy,
                z_hat,
                zq,
                zsrc,
                b["gt_chunks"],
                b["valid_16"],
                b["init_noise_alt_cpu"],
                b["tasks"],
                None,
                s_z=s_z,
                device=device,
                use_bf16=use_bf16,
                batch_size=batch_size,
            )
            geom = (
                compute_coeff_geometry(c_mat, c_p0, c_e10c, s_z=s_z)
                if c_mat is not None
                else {
                    "c_l2_mean": float(np.mean(np.linalg.norm(zq - zsrc, axis=-1))),
                    "c_l2_median": float(np.median(np.linalg.norm(zq - zsrc, axis=-1))),
                    "c_l2_p90": float(np.percentile(np.linalg.norm(zq - zsrc, axis=-1), 90)),
                    "c_l2_sq_mean": float(np.mean(np.sum((zq - zsrc) ** 2, axis=-1))),
                    "delta_z_rms": float(math.sqrt(float(np.mean((zq - zsrc) ** 2)))),
                    "delta_z_rms_per_sample_mean": float(
                        np.mean(np.linalg.norm(zq - zsrc, axis=-1)) / math.sqrt(float(LATENT_DIM))
                    ),
                    "delta_z_rms_over_sz": float(
                        math.sqrt(float(np.mean((zq - zsrc) ** 2))) / s_z
                    ),
                    "c_norm_ratio_vs_p0": None,
                    "c_energy_ratio_vs_p0": None,
                    "cos_vs_c_p0": None,
                    "l2_dist_vs_c_p0": None,
                    "coeff_mse_vs_c_p0": None,
                    "cos_vs_c_e10c": None,
                    "coeff_mse_vs_c_e10c": None,
                    "c_erank_energy": None,
                    "c_participation_ratio": None,
                    "c_d80": None,
                    "c_d90": None,
                }
            )
            rec = {
                "split": split_name,
                "method": method_id,
                "label": label,
                "lambda": lam_val,
                **ov,
                "ddim_10d_mse_alt_seed": ov_alt["ddim_10d_mse"],
                **geom,
                "phases": ph_d,
            }
            split_method_res[method_id] = rec
            print(
                f"  [{method_id:18s}] DDIM10={ov['ddim_10d_mse']:.5f} (alt={ov_alt['ddim_10d_mse']:.5f}) | "
                f"MAE7={ov['ddim_7d_mae']:.5f} | Grip={100*ov['grip_acc16']:.2f}% | "
                f"||c||_2={geom['c_l2_mean']:.3f} | Δz_rms={geom['delta_z_rms']:.4f} | Lz_vs_zq={ov['latent_lz']:.4f}",
                flush=True,
            )

        mse_id = split_method_res["identity"]["ddim_10d_mse"]
        mse_full = split_method_res["full_self"]["ddim_10d_mse"]

        # 2. Action-Optimized Oracle across lambda sweep
        for lam_idx, lam in enumerate(lambdas):
            t_lam = time.monotonic()
            lam_tag = f"action_opt_lam_{lam:.0e}".replace("+", "").replace("-0", "-")
            if lam == 0.0:
                lam_tag = "action_opt_lam_0"

            c_opt, curves = optimize_coefficients_batch(
                policy.teacher,
                zsrc,
                b["gt_chunks"],
                u64_t,
                lam=float(lam),
                steps=steps,
                lr=lr,
                mc_timesteps=mc_timesteps,
                batch_size=batch_size,
                seed=seed + 10000 * (split_idx + 1) + 100 * lam_idx,
                device=device,
                use_bf16=use_bf16,
                log_steps=(0, 10, 25, 50, 75, 100, 150, 200, steps),
                ddim_track_steps=(0, 25, 50, 100, 150, steps),
                ddim_track_n=512,
                ddim_init_noise_cpu=b["init_noise_cpu"],
                ddim_valid_16=b["valid_16"],
            )
            saved_c[lam_tag] = c_opt
            for cr in curves:
                cr["split"] = split_name
                cr["method"] = lam_tag
                curve_rows_all.append(cr)

            z_opt = (zsrc + c_opt @ u64_np.T).astype(np.float32)
            ov, ph_d = decode_and_score_with_phases(
                policy,
                z_opt,
                zq,
                zsrc,
                b["gt_chunks"],
                b["valid_16"],
                b["init_noise_cpu"],
                b["tasks"],
                b["phases"],
                c_mat=c_opt,
                c_p0=c_p0,
                s_z=s_z,
                device=device,
                use_bf16=use_bf16,
                batch_size=batch_size,
            )
            ov_alt, _ = decode_and_score_with_phases(
                policy,
                z_opt,
                zq,
                zsrc,
                b["gt_chunks"],
                b["valid_16"],
                b["init_noise_alt_cpu"],
                b["tasks"],
                None,
                s_z=s_z,
                device=device,
                use_bf16=use_bf16,
                batch_size=batch_size,
            )
            geom = compute_coeff_geometry(c_opt, c_p0, c_e10c, s_z=s_z)
            dt_lam = time.monotonic() - t_lam
            rec = {
                "split": split_name,
                "method": lam_tag,
                "label": f"Action-Opt Oracle (λ={lam:.0e})" if lam > 0 else "Action-Opt Oracle (λ=0)",
                "lambda": float(lam),
                **ov,
                "ddim_10d_mse_alt_seed": ov_alt["ddim_10d_mse"],
                "final_diff_loss": curves[-1]["diff_loss"],
                "final_total_obj": curves[-1]["total_obj"],
                "final_grad_norm": curves[-1]["grad_norm_mean"],
                "opt_seconds": round(dt_lam, 2),
                **geom,
                "phases": ph_d,
            }
            split_method_res[lam_tag] = rec
            arr, _ = _safe_arr(mse_id, ov["ddim_10d_mse"], mse_full)
            print(
                f"  [{lam_tag:18s}] DDIM10={ov['ddim_10d_mse']:.5f} (alt={ov_alt['ddim_10d_mse']:.5f}, ARR={100*(arr or 0):5.1f}%) | "
                f"MAE7={ov['ddim_7d_mae']:.5f} | Grip={100*ov['grip_acc16']:.2f}% | "
                f"||c||_2={geom['c_l2_mean']:6.3f} ({100*geom['c_norm_ratio_vs_p0']:4.1f}% P0) | "
                f"Lz_vs_zq={ov['latent_lz']:.4f} | cos_p0={geom['cos_vs_c_p0']:.3f} | "
                f"eRank={geom['c_erank_energy']:.1f} | {dt_lam:.1f}s",
                flush=True,
            )

        # Attach ARR and flatten summary + phase CSV rows
        for method_id, rec in split_method_res.items():
            arr_10d, _ = _safe_arr(mse_id, rec["ddim_10d_mse"], mse_full)
            rec["arr_10d_vs_id_full"] = arr_10d
            flat = {
                k: v
                for k, v in rec.items()
                if k not in ("phases", "per_dim_mse")
            }
            summary_rows.append(flat)

            for ph_name, ph_m in rec["phases"].items():
                phase_rows.append(
                    {
                        "split": split_name,
                        "method": method_id,
                        "label": rec["label"],
                        "lambda": rec["lambda"],
                        "phase": ph_name,
                        "count": ph_m["count"],
                        "ddim_10d_mse": ph_m["ddim_10d_mse"],
                        "ddim_pos_mse": ph_m["ddim_pos_mse"],
                        "ddim_rot6d_mse": ph_m["ddim_rot6d_mse"],
                        "ddim_grip_mse": ph_m["ddim_grip_mse"],
                        "ddim_7d_mae": ph_m["ddim_7d_mae"],
                        "grip_acc16": ph_m["grip_acc16"],
                        "latent_lz": ph_m["latent_lz"],
                        "delta_z_l2_mean": ph_m["delta_z_l2_mean"],
                        "delta_z_rms": ph_m["delta_z_rms"],
                        "c_l2_mean": ph_m.get("c_l2_mean"),
                        "cos_vs_c_p0": ph_m.get("cos_vs_c_p0"),
                    }
                )

        # Save cached coefficients for downstream analysis / learnability evaluation
        np.savez_compressed(out_dir / f"{split_name}_c_opt.npz", **saved_c)
        nested_results[split_name] = {
            "n_pairs": len(zq),
            "phase_counts": ph_counts,
            "reader_audit": b["reader_audit"],
            "methods": split_method_res,
        }

    config = {
        "experiment": "e11_pilot_oracle_action_guided_minimal_adaptation",
        "model_meta": model_meta,
        "n_seen_val": int(n_seen_val),
        "n_unseen": int(n_unseen),
        "lambdas": [float(x) for x in lambdas],
        "steps": int(steps),
        "lr": float(lr),
        "eta_min_ratio": 0.05,
        "mc_timesteps": int(mc_timesteps),
        "pertub_noise": 0.0,
        "batch_size": int(batch_size),
        "seed": int(seed),
        "ddim_seed": int(ddim_seed),
        "ddim_seed_alt": int(ddim_seed_alt),
        "use_bf16": bool(use_bf16 and device.type == "cuda"),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }

    atomic_write_json(config, out_dir / "config.json")
    atomic_write_json(
        {"config": config, "results": nested_results},
        out_dir / "oracle_sweep_summary.json",
    )
    write_csv(summary_rows, out_dir / "oracle_sweep_summary.csv")
    write_csv(phase_rows, out_dir / "phase_breakdown.csv")
    write_csv(curve_rows_all, out_dir / "convergence_curves.csv")

    print(
        f"\n[E11-Oracle DONE] Total elapsed: {config['elapsed_seconds']}s -> {out_dir}",
        flush=True,
    )
    return {"config": config, "results": nested_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "oracle")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_GRID),
    )
    parser.add_argument("--n-seen-val", type=int, default=3000)
    parser.add_argument("--n-unseen", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--mc-timesteps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-seed", type=int, default=12345)
    parser.add_argument("--ddim-seed-alt", type=int, default=54321)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_oracle_experiment(
        out_dir=args.out_dir,
        lambdas=args.lambdas,
        n_seen_val=args.n_seen_val,
        n_unseen=args.n_unseen,
        steps=args.steps,
        lr=args.lr,
        mc_timesteps=args.mc_timesteps,
        batch_size=args.batch_size,
        seed=args.seed,
        ddim_seed=args.ddim_seed,
        ddim_seed_alt=args.ddim_seed_alt,
        device=device,
        use_bf16=not args.no_bf16,
    )


if __name__ == "__main__":
    main()
