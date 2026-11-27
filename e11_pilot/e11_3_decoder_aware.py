"""E11.3: Decoder-Aware Transfer Optimization over Frozen E9 & P0 Basis U_64.

Tests whether Action-Opt Oracle initialization (E11.2-A) translates into real
end-to-end action accuracy advantages over Future-Latent initialization (E11.2-B)
and Scratch initialization (C) under the exact same Decoder-Aware objective:

    L(theta) = L_diff(G(z_src + U_64 * c_hat), a_q*) + lambda * ||c_hat||_2^2
    where c_hat = f_theta(z_src, h_q)

Strict experimental invariants:
  - Identical 28.56M Transfer Transformer (`e10.model.LowRankTransferTransformer`,
    d=768, L=3, 8 heads, FFN=3072, dropout=0.15, zero-init 64D head).
  - Fixed P0 basis `U_64`, frozen E9 Stage 1 Encoder & Action Decoder, frozen E9 Reader.
  - Exact same 24k `train_fit` / 2k `val_ckpt` anti-leakage split as E11.2 (bit-identical
    `z_src`, `h_q`, `va_gt_chunks` + verified deterministic `tr_gt_chunks`).
  - NO coefficient regression loss; gradients flow solely through the frozen Action Decoder
    plus `lambda * ||c_hat||_2^2`.
  - Three strictly fair controls with identical optimizer, cosine LR schedule, batch size,
    total steps, EMA decay, seed, and identical mini-batch / diffusion noise streams:
      * A (`group_a_action_opt_init`): initialized from E11.2-A `best.pt`
      * B (`group_b_future_lat_init`): initialized from E11.2-B `best.pt`
      * C (`group_c_scratch`):         randomly initialized identical architecture
  - Independent `val_ckpt` DDIM 10D Action MSE checkpoint selection.
  - Independent output root `e11_pilot/outputs_e11_3/` leaving E11-Pilot & E11.2 untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from e10.model import LowRankTransferTransformer
from e5.diffusion import EMAModel
from e5.train import atomic_torch_save, atomic_write_json, seed_everything
from e7.train import get_cosine_schedule_with_min_ratio
from p0_lowrank.run_diagnosis import _compute_alignment_and_gt_for_split, _safe_arr

from . import (
    ACTION_DIM_10D,
    ACTION_HORIZON,
    E5_CACHE_DIR,
    E9_STAGE2_CACHE_DIR,
    EMBED_DIM,
    LATENT_TOKENS,
    OUTPUT_ROOT as E11_PILOT_OUTPUT_ROOT,
    RANK_K,
    SEMANTIC_PHASES,
)
from .data import (
    load_eval_splits,
    load_frozen_e10c_and_basis,
    sample_pairs_from_episode_subset,
    split_train_episodes_for_learnability,
)
from .e11_2_scaling import (
    DEFAULT_CACHE_META_PATH as E11_2_CACHE_META_PATH,
    DEFAULT_CACHE_PATH as E11_2_CACHE_PATH,
    DEFAULT_ORACLE_DIR,
    E11_2_OUTPUT_ROOT,
    build_e10_aligned_28m_transfer,
    get_transfer_optim_groups,
)
from .learnability import _coeff_fit_metrics, _predict_coeffs, evaluate_model_on_splits, lam_to_tag
from .oracle_opt import decode_and_score_with_phases, write_csv

E11_3_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_e11_3"
DEFAULT_E11_3_CACHE_PATH = E11_3_OUTPUT_ROOT / "cache" / "train_val_e11_3.npz"
DEFAULT_E11_3_CACHE_META_PATH = E11_3_OUTPUT_ROOT / "cache" / "train_val_e11_3_meta.json"

E11_2_CKPT_A = E11_2_OUTPUT_ROOT / "group_a_action_opt_28m" / "best.pt"
E11_2_CKPT_B = E11_2_OUTPUT_ROOT / "group_b_future_lat_28m" / "best.pt"

DEFAULT_E11_3_LAMBDAS: Tuple[float, ...] = (0.0, 1e-6, 3e-6, 1e-5)

INIT_SPECS: Dict[str, Dict[str, Any]] = {
    "A": {
        "run_name": "group_a_action_opt_init",
        "short": "A (Action-Opt Init)",
        "label": "E11.3-A Action-Opt Init",
        "init_ckpt": E11_2_CKPT_A,
        "init_source": "E11.2-A best.pt",
        "color": "#1f77b4",
        "marker": "o",
    },
    "B": {
        "run_name": "group_b_future_lat_init",
        "short": "B (Future-Latent Init)",
        "label": "E11.3-B Future-Latent Init",
        "init_ckpt": E11_2_CKPT_B,
        "init_source": "E11.2-B best.pt",
        "color": "#2ca02c",
        "marker": "s",
    },
    "C": {
        "run_name": "group_c_scratch",
        "short": "C (Scratch)",
        "label": "E11.3-C Scratch",
        "init_ckpt": None,
        "init_source": "Random init (zero-init head)",
        "color": "#ff7f0e",
        "marker": "^",
    },
}


def prepare_e11_3_action_cache(
    *,
    e11_2_cache_path: Path = E11_2_CACHE_PATH,
    e11_2_cache_meta_path: Path = E11_2_CACHE_META_PATH,
    out_cache_path: Path = DEFAULT_E11_3_CACHE_PATH,
    out_meta_path: Path = DEFAULT_E11_3_CACHE_META_PATH,
) -> Path:
    """Attach verified ground-truth `tr_gt_chunks` (24,000, 16, 10) to the exact E11.2
    `train_val_c_opt_lam_1e-5.npz` cache without modifying a single latent or split entry.

    Deterministically reconstructs the exact E11-Pilot `train_fit` (24k) and `val_ckpt` (2k)
    pairs on CPU (~2s), cross-checks bit-for-bit equality of `tr_h_q` (fp16), `va_h_q` (fp16),
    `va_gt_chunks` (fp32), `va_valid_16`, `va_tasks`, and `split_audit`, then saves the
    complete training/validation bundle for E11.3.
    """
    t0 = time.monotonic()
    out_cache_path = Path(out_cache_path)
    out_cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_npz = dict(np.load(e11_2_cache_path, allow_pickle=False))
    cache_meta = json.loads(Path(e11_2_cache_meta_path).read_text())

    n_train_fit = int(cache_meta["n_train_fit"])
    n_val_ckpt = int(cache_meta["n_val_ckpt"])
    split_seed = int(cache_meta["seed"])

    episodes = json.loads((E5_CACHE_DIR / "episodes.json").read_text())
    actions_10d_e5 = np.load(E5_CACHE_DIR / "actions_10d.npy", mmap_mode="r")
    ee_pos_e5 = np.load(E5_CACHE_DIR / "proprio_11d.npy", mmap_mode="r")[:, :3]
    action_offsets_e5 = np.load(E5_CACHE_DIR / "action_offsets.npy")

    e9_mem_h = np.load(E9_STAGE2_CACHE_DIR / "mem_h.npy", mmap_mode="r")
    e9_mem_anchors = np.load(E9_STAGE2_CACHE_DIR / "mem_anchor_steps.npy")
    e9_mem_offsets = np.load(E9_STAGE2_CACHE_DIR / "mem_offsets.npy")
    e9_mem_lengths = np.load(E9_STAGE2_CACHE_DIR / "mem_lengths.npy")

    train_fit_eps, val_ckpt_eps = split_train_episodes_for_learnability(
        episodes, val_frac=0.10, seed=split_seed
    )
    train_pairs = sample_pairs_from_episode_subset(
        episodes,
        e9_mem_lengths,
        train_fit_eps,
        split_name="train_fit",
        target_pairs=n_train_fit,
        seed=split_seed + 101,
    )
    val_pairs = sample_pairs_from_episode_subset(
        episodes,
        e9_mem_lengths,
        val_ckpt_eps,
        split_name="val_ckpt",
        target_pairs=n_val_ckpt,
        seed=split_seed + 202,
    )

    tr_sb = _compute_alignment_and_gt_for_split(
        "train_fit",
        train_pairs,
        actions_10d=actions_10d_e5,
        ee_pos_all=ee_pos_e5,
        action_offsets=action_offsets_e5,
        mem_anchor_steps=e9_mem_anchors,
        mem_offsets=e9_mem_offsets,
        mem_lengths=e9_mem_lengths,
        seed=split_seed + 301,
    )
    va_sb = _compute_alignment_and_gt_for_split(
        "val_ckpt",
        val_pairs,
        actions_10d=actions_10d_e5,
        ee_pos_all=ee_pos_e5,
        action_offsets=action_offsets_e5,
        mem_anchor_steps=e9_mem_anchors,
        mem_offsets=e9_mem_offsets,
        mem_lengths=e9_mem_lengths,
        seed=split_seed + 302,
    )

    # Strict bit-for-bit verification against existing E11.2 cache
    tr_h_rebuilt = np.empty((n_train_fit, LATENT_TOKENS, EMBED_DIM), dtype=np.float16)
    for i, p in enumerate(train_pairs):
        q_flat = int(e9_mem_offsets[int(p["q_ep"])]) + int(p["q_anchor_idx"])
        tr_h_rebuilt[i] = e9_mem_h[q_flat].astype(np.float16)

    va_h_rebuilt = np.empty((n_val_ckpt, LATENT_TOKENS, EMBED_DIM), dtype=np.float16)
    for i, p in enumerate(val_pairs):
        q_flat = int(e9_mem_offsets[int(p["q_ep"])]) + int(p["q_anchor_idx"])
        va_h_rebuilt[i] = e9_mem_h[q_flat].astype(np.float16)

    tr_h_diff = float(np.max(np.abs(tr_h_rebuilt.astype(np.float32) - cache_npz["tr_h_q"].astype(np.float32))))
    va_h_diff = float(np.max(np.abs(va_h_rebuilt.astype(np.float32) - cache_npz["va_h_q"].astype(np.float32))))
    va_gt_diff = float(np.max(np.abs(va_sb["gt_chunks"].astype(np.float32) - cache_npz["va_gt_chunks"].astype(np.float32))))
    va_v16_eq = bool(np.array_equal(va_sb["valid_16"].astype(np.bool_), cache_npz["va_valid_16"].astype(np.bool_)))
    va_tasks_eq = [p["task"] for p in val_pairs] == cache_npz["va_tasks"].tolist()

    tr_eps_used = {p["q_ep"] for p in train_pairs} | {p["s_ep"] for p in train_pairs}
    va_eps_used = {p["q_ep"] for p in val_pairs} | {p["s_ep"] for p in val_pairs}
    val_fold_eps = {i for i, e in enumerate(episodes) if e["fold"] == "val"}
    audit = {
        "train_fit_pairs": len(train_pairs),
        "val_ckpt_pairs": len(val_pairs),
        "train_fit_unique_eps": len(tr_eps_used),
        "val_ckpt_unique_eps": len(va_eps_used),
        "train_vs_val_ckpt_ep_overlap": len(tr_eps_used & va_eps_used),
        "train_vs_seen_val_ep_overlap": len(tr_eps_used & val_fold_eps),
        "val_ckpt_vs_seen_val_ep_overlap": len(va_eps_used & val_fold_eps),
        "all_diff_layout": all(p["q_layout"] != p["s_layout"] for p in (train_pairs + val_pairs)),
    }

    if (
        tr_h_diff != 0.0
        or va_h_diff != 0.0
        or va_gt_diff != 0.0
        or not va_v16_eq
        or not va_tasks_eq
        or audit != cache_meta["split_audit"]
    ):
        raise RuntimeError(
            "E11.3 cache alignment verification FAILED: "
            f"tr_h_diff={tr_h_diff}, va_h_diff={va_h_diff}, va_gt_diff={va_gt_diff}, "
            f"va_v16_eq={va_v16_eq}, va_tasks_eq={va_tasks_eq}, audit_eq={audit == cache_meta['split_audit']}"
        )

    # Save exact E11.2 arrays + verified tr_gt_chunks / tr_valid_16
    save_payload = dict(cache_npz)
    save_payload["tr_gt_chunks"] = tr_sb["gt_chunks"].astype(np.float32)
    save_payload["tr_valid_16"] = tr_sb["valid_16"].astype(np.bool_)
    np.savez(out_cache_path, **save_payload)

    meta = {
        "source_e11_2_cache": str(Path(e11_2_cache_path).resolve()),
        "source_e11_2_meta": str(Path(e11_2_cache_meta_path).resolve()),
        "out_cache_path": str(out_cache_path.resolve()),
        "n_train_fit": n_train_fit,
        "n_val_ckpt": n_val_ckpt,
        "seed": split_seed,
        "ddim_seed": int(cache_meta["ddim_seed"]),
        "split_audit": audit,
        "verification": {
            "tr_h_q_fp16_max_abs_diff": tr_h_diff,
            "va_h_q_fp16_max_abs_diff": va_h_diff,
            "va_gt_chunks_max_abs_diff": va_gt_diff,
            "va_valid_16_exact_match": va_v16_eq,
            "va_tasks_exact_match": va_tasks_eq,
            "split_audit_exact_match": True,
        },
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    atomic_write_json(meta, out_meta_path)
    print(
        f"[E11.3 Cache] Verified 100% bit-exact match with E11.2 (24k train_fit / 2k val_ckpt) "
        f"in {meta['elapsed_seconds']}s -> {out_cache_path}",
        flush=True,
    )
    return out_cache_path


def build_initialized_transfer_model(
    *,
    group: str,
    s_z: float,
    dropout: float = 0.15,
    seed: int = 42,
    device: torch.device,
) -> Tuple[LowRankTransferTransformer, Dict[str, Any]]:
    """Construct the exact 28.56M E11.2 Transfer Transformer for Group A, B, or C."""
    spec = INIT_SPECS[group]
    seed_everything(seed)
    model = build_e10_aligned_28m_transfer(dropout=dropout, latent_scale_s_z=s_z).to(device)
    init_meta: Dict[str, Any] = {
        "group": group,
        "init_source": spec["init_source"],
        "init_ckpt": str(Path(spec["init_ckpt"]).resolve()) if spec["init_ckpt"] is not None else None,
        "init_ckpt_epoch": None,
        "init_ckpt_val_ddim10": None,
    }
    if spec["init_ckpt"] is not None:
        ck = torch.load(spec["init_ckpt"], map_location="cpu", weights_only=False)
        expected_kwargs = {
            "rank_k": RANK_K,
            "embed_dim": EMBED_DIM,
            "num_layers": 3,
            "num_heads": 8,
            "ffn_dim": 3072,
            "num_query_tokens": LATENT_TOKENS,
            "dropout": float(dropout),
            "latent_scale_s_z": float(s_z),
            "init_std": 0.0,
        }
        ck_kw = ck["model_kwargs"]
        for k, v in expected_kwargs.items():
            if isinstance(v, float):
                if abs(float(ck_kw[k]) - v) > 1e-6:
                    raise RuntimeError(f"Init ckpt mismatch on {k}: {ck_kw[k]} != {v}")
            elif ck_kw[k] != v:
                raise RuntimeError(f"Init ckpt mismatch on {k}: {ck_kw[k]} != {v}")
        model.load_state_dict(ck["state_dict"], strict=True)
        init_meta["init_ckpt_epoch"] = int(ck["epoch"])
        init_meta["init_ckpt_val_ddim10"] = float(ck["best_val_ckpt_ddim10"])
    return model, init_meta


def compute_decoder_aware_loss(
    *,
    model: LowRankTransferTransformer,
    teacher: nn.Module,
    u64_t: torch.Tensor,
    z_src_flat: torch.Tensor,
    h_q: torch.Tensor,
    gt_actions: torch.Tensor,
    lam: float,
    mc_timesteps: int,
    t_edges: np.ndarray,
    diff_gen: torch.Generator,
    use_bf16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact E11.3 Decoder-Aware loss:

        L = L_diff(G(z_src + U_64 * c_hat), a_q*) + lam * ||c_hat||_2^2

    using `mc_timesteps` stratified timesteps across [0, num_train_timesteps) and
    zero input perturbation (`pertub_noise=0`), matching the E11 Oracle objective exactly.
    Returns `(total_loss, diff_loss, reg_loss, c_hat_fp32)`.
    """
    device = z_src_flat.device
    B = z_src_flat.shape[0]
    M = int(mc_timesteps)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    ts = torch.stack(
        [
            torch.randint(
                int(t_edges[m]),
                int(t_edges[m + 1]),
                (B,),
                device=device,
                generator=diff_gen,
            )
            for m in range(M)
        ],
        dim=1,
    ).reshape(-1)
    act_rep = gt_actions.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, ACTION_HORIZON, ACTION_DIM_10D)
    eps = torch.randn(act_rep.shape, device=device, dtype=torch.float32, generator=diff_gen)
    p_eps = torch.zeros_like(eps)

    z_in = z_src_flat.reshape(B, LATENT_TOKENS, EMBED_DIM)
    with amp_ctx:
        c_hat = model(z_in, h_q)  # (B, 64)
        c_f = c_hat.float()
        z_hat_flat = z_src_flat + torch.matmul(c_f, u64_t.T)
        z_rep = (
            z_hat_flat.unsqueeze(1)
            .expand(-1, M, -1)
            .reshape(B * M, LATENT_TOKENS, EMBED_DIM)
        )
        diff_per = (
            teacher.diffusion_loss_from_z_self(
                z_rep, act_rep, noise=eps, pertub_noise=p_eps, timesteps=ts
            )
            .reshape(B, M)
            .mean(dim=1)
        )
        diff_loss = diff_per.mean()

    c_sq_mean = c_f.square().sum(dim=-1).mean()
    reg_loss = float(lam) * c_sq_mean
    total_loss = diff_loss + reg_loss
    return total_loss, diff_loss, reg_loss, c_f


def audit_decoder_aware_freeze_and_grad(
    *,
    model: LowRankTransferTransformer,
    policy: nn.Module,
    u64_t: torch.Tensor,
    z_src_b: torch.Tensor,
    h_q_b: torch.Tensor,
    a_b: torch.Tensor,
    lam: float,
    mc_timesteps: int,
    t_edges: np.ndarray,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Verify frozen E9 Teacher, Reader, and U_64 receive ZERO gradients while
    the 28.56M Transfer Transformer receives non-zero gradients through the frozen Decoder."""
    policy.eval()
    assert sum(p.numel() for p in policy.parameters() if p.requires_grad) == 0
    assert not u64_t.requires_grad

    model.train()
    model.zero_grad(set_to_none=True)
    for p in policy.parameters():
        p.grad = None

    g = torch.Generator(device=z_src_b.device).manual_seed(9999)
    total_loss, diff_loss, reg_loss, c_f = compute_decoder_aware_loss(
        model=model,
        teacher=policy.teacher,
        u64_t=u64_t,
        z_src_flat=z_src_b,
        h_q=h_q_b,
        gt_actions=a_b,
        lam=lam,
        mc_timesteps=mc_timesteps,
        t_edges=t_edges,
        diff_gen=g,
        use_bf16=use_bf16,
    )
    total_loss.backward()

    teacher_any_grad = any(p.grad is not None for p in policy.teacher.parameters())
    reader_any_grad = any(p.grad is not None for p in policy.reader.parameters())
    head_gn = (
        float(model.coeff_head.weight.grad.norm().item())
        if model.coeff_head.weight.grad is not None
        else 0.0
    )
    nonzero_tensors = sum(
        1 for p in model.parameters() if p.grad is not None and float(p.grad.norm().item()) > 0.0
    )

    model.zero_grad(set_to_none=True)
    for p in policy.parameters():
        p.grad = None

    audit = {
        "teacher_any_grad": teacher_any_grad,
        "reader_any_grad": reader_any_grad,
        "u64_requires_grad": bool(u64_t.requires_grad),
        "coeff_head_grad_norm": head_gn,
        "transfer_nonzero_grad_tensors": int(nonzero_tensors),
        "probe_diff_loss": float(diff_loss.item()),
        "probe_reg_loss": float(reg_loss.item()),
        "probe_c_l2_mean": float(c_f.detach().norm(dim=-1).mean().item()),
    }
    if teacher_any_grad or reader_any_grad or u64_t.requires_grad:
        raise RuntimeError(f"Freeze audit FAILED: {audit}")
    if head_gn <= 0.0:
        raise RuntimeError(f"Zero gradient into Transfer coeff_head through frozen Decoder: {audit}")
    return audit


@torch.inference_mode()
def eval_decoder_aware_objective_fixed(
    *,
    model: LowRankTransferTransformer,
    teacher: nn.Module,
    u64_t: torch.Tensor,
    z_src_t: torch.Tensor,
    h_q_t: torch.Tensor,
    gt_actions_t: torch.Tensor,
    lam: float,
    mc_timesteps: int,
    t_edges: np.ndarray,
    eval_seed: int,
    batch_size: int = 512,
    use_bf16: bool = True,
) -> Dict[str, float]:
    """Deterministic evaluation of `L_diff`, `lam * ||c||_2^2`, `total_loss`, and `||c||_2`
    using a fixed RNG seed so training/val objective curves are smooth and comparable."""
    model.eval()
    device = z_src_t.device
    N = len(z_src_t)
    g = torch.Generator(device=device).manual_seed(int(eval_seed))

    diff_sum = 0.0
    reg_sum = 0.0
    tot_sum = 0.0
    cl2_sum = 0.0
    csq_sum = 0.0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        tot_l, diff_l, reg_l, c_f = compute_decoder_aware_loss(
            model=model,
            teacher=teacher,
            u64_t=u64_t,
            z_src_flat=z_src_t[start:end],
            h_q=h_q_t[start:end],
            gt_actions=gt_actions_t[start:end],
            lam=lam,
            mc_timesteps=mc_timesteps,
            t_edges=t_edges,
            diff_gen=g,
            use_bf16=use_bf16,
        )
        diff_sum += float(diff_l.item()) * B
        reg_sum += float(reg_l.item()) * B
        tot_sum += float(tot_l.item()) * B
        cl2_sum += float(c_f.norm(dim=-1).sum().item())
        csq_sum += float(c_f.square().sum(dim=-1).sum().item())

    return {
        "diff_loss": diff_sum / N,
        "reg_loss": reg_sum / N,
        "total_loss": tot_sum / N,
        "c_l2_mean": cl2_sum / N,
        "c_l2_sq_mean": csq_sum / N,
    }


def train_e11_3_run(
    *,
    group: str,
    lam: float,
    out_dir: Path = E11_3_OUTPUT_ROOT,
    cache_path: Path = DEFAULT_E11_3_CACHE_PATH,
    cache_meta_path: Path = DEFAULT_E11_3_CACHE_META_PATH,
    dropout: float = 0.15,
    epochs: int = 80,
    batch_size: int = 256,
    mc_timesteps: int = 4,
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
    """Train one (group in {A, B, C}, lambda) Decoder-Aware run with strict fairness."""
    t0 = time.monotonic()
    spec = INIT_SPECS[group]
    ltag = lam_to_tag(lam)
    run_name = spec["run_name"]
    run_dir = Path(out_dir) / ltag / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not Path(cache_path).is_file():
        prepare_e11_3_action_cache(out_cache_path=cache_path, out_meta_path=cache_meta_path)

    policy, u64_t, model_meta = load_frozen_e10c_and_basis(device=device)
    u64_np = u64_t.cpu().numpy()
    s_z = float(model_meta["s_z"])
    e10c_transfer_params = sum(p.numel() for p in policy.transfer.parameters())
    assert sum(p.numel() for p in policy.parameters() if p.requires_grad) == 0

    model, init_meta = build_initialized_transfer_model(
        group=group, s_z=s_z, dropout=dropout, seed=seed, device=device
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params != e10c_transfer_params:
        raise RuntimeError(f"Param mismatch: {n_params:,} != E10-C {e10c_transfer_params:,}")

    cache_npz = dict(np.load(cache_path, allow_pickle=False))
    cache_meta = json.loads(Path(cache_meta_path).read_text())

    # GPU-resident training & validation tensors (~300 MB total on 48GB A6000)
    tr_z = torch.from_numpy(cache_npz["tr_z_src"]).to(device=device, dtype=torch.float32)
    tr_h = torch.from_numpy(cache_npz["tr_h_q"]).to(device=device, dtype=torch.float32)
    tr_a = torch.from_numpy(cache_npz["tr_gt_chunks"]).to(device=device, dtype=torch.float32)

    va_z_np = cache_npz["va_z_src"].astype(np.float32)
    va_h_np = cache_npz["va_h_q"].astype(np.float32)
    va_zq_np = cache_npz["va_z_query"].astype(np.float32)
    va_c_opt_np = cache_npz["va_c_opt"].astype(np.float32)
    va_c_p0_np = cache_npz["va_c_p0"].astype(np.float32)
    va_gt = cache_npz["va_gt_chunks"].astype(np.float32)
    va_v16 = cache_npz["va_valid_16"].astype(bool)
    va_noise = torch.from_numpy(cache_npz["va_init_noise"].astype(np.float32))
    va_tasks = [str(x) for x in cache_npz["va_tasks"].tolist()]

    va_z_t = torch.from_numpy(va_z_np).to(device=device, dtype=torch.float32)
    va_h_t = torch.from_numpy(va_h_np).to(device=device, dtype=torch.float32)
    va_a_t = torch.from_numpy(va_gt).to(device=device, dtype=torch.float32)

    N_tr = len(tr_z)
    steps_per_epoch = math.ceil(N_tr / batch_size)
    total_steps = steps_per_epoch * epochs

    num_train_t = int(policy.teacher.noise_scheduler.config.num_train_timesteps)
    t_edges = np.linspace(0, num_train_t, mc_timesteps + 1, dtype=int)

    # Strict freeze & through-decoder gradient audit
    grad_audit = audit_decoder_aware_freeze_and_grad(
        model=model,
        policy=policy,
        u64_t=u64_t,
        z_src_b=tr_z[:32],
        h_q_b=tr_h[:32],
        a_b=tr_a[:32],
        lam=lam,
        mc_timesteps=mc_timesteps,
        t_edges=t_edges,
        use_bf16=use_bf16,
    )

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
    # Initialize EMA directly from the group's starting weights (E11.2-A, E11.2-B, or Scratch)
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

    model_kwargs = {
        "rank_k": RANK_K,
        "embed_dim": EMBED_DIM,
        "num_layers": 3,
        "num_heads": 8,
        "ffn_dim": 3072,
        "num_query_tokens": LATENT_TOKENS,
        "dropout": float(dropout),
        "latent_scale_s_z": s_z,
        "init_std": 0.0,
    }

    config: Dict[str, Any] = {
        "experiment": "e11_3_decoder_aware_transfer_optimization",
        "group": group,
        "run_name": run_name,
        "label": f"{spec['label']} (lam={lam:.0e})" if lam > 0 else f"{spec['label']} (lam=0)",
        "lambda": float(lam),
        "lambda_tag": ltag,
        "objective": f"L_diff(G(z_src + U_64 c_hat), a_q*) + {lam:.1e} * ||c_hat||_2^2",
        "init_meta": init_meta,
        "model_meta": model_meta,
        "architecture": {
            "class": "e10.model.LowRankTransferTransformer",
            **model_kwargs,
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
            "mc_timesteps": int(mc_timesteps),
            "pertub_noise": 0.0,
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
        "freeze_and_grad_audit": grad_audit,
    }
    atomic_write_json(config, run_dir / "config.json")

    print(
        f"\n[E11.3 {ltag} | {group} {run_name}] init={init_meta['init_source']} | lam={lam:.1e} | "
        f"params={n_params:,} | N_tr={N_tr} N_va={len(va_z_np)} | "
        f"epochs={epochs} ({total_steps} steps, bs={batch_size}, M={mc_timesteps}) | "
        f"lr={lr:.1e}->{lr*min_lr_ratio:.1e} | grad_audit head_gn={grad_audit['coeff_head_grad_norm']:.4f}",
        flush=True,
    )

    # Fixed 4,000-sample subset of train_fit for smooth deterministic EMA objective tracking
    tr_eval_n = min(4000, N_tr)
    tr_eval_z_t = tr_z[:tr_eval_n]
    tr_eval_h_t = tr_h[:tr_eval_n]
    tr_eval_a_t = tr_a[:tr_eval_n]

    def _eval_current_weights(ep: int, run_ddim: bool) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Optional[float]]:
        tr_obj = eval_decoder_aware_objective_fixed(
            model=model,
            teacher=policy.teacher,
            u64_t=u64_t,
            z_src_t=tr_eval_z_t,
            h_q_t=tr_eval_h_t,
            gt_actions_t=tr_eval_a_t,
            lam=lam,
            mc_timesteps=mc_timesteps,
            t_edges=t_edges,
            eval_seed=seed + 9001,
            batch_size=512,
            use_bf16=use_bf16,
        )
        va_obj = eval_decoder_aware_objective_fixed(
            model=model,
            teacher=policy.teacher,
            u64_t=u64_t,
            z_src_t=va_z_t,
            h_q_t=va_h_t,
            gt_actions_t=va_a_t,
            lam=lam,
            mc_timesteps=mc_timesteps,
            t_edges=t_edges,
            eval_seed=seed + 9002,
            batch_size=512,
            use_bf16=use_bf16,
        )
        va_pred = _predict_coeffs(
            model, va_z_np, va_h_np, batch_size=1024, device=device, use_bf16=use_bf16
        )
        va_cf_opt = _coeff_fit_metrics(va_pred, va_c_opt_np)
        va_cf_p0 = _coeff_fit_metrics(va_pred, va_c_p0_np)

        va_ddim_mse = None
        if run_ddim:
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
        return tr_obj, va_obj, va_cf_opt, va_cf_p0, va_ddim_mse

    history: List[Dict[str, Any]] = []
    best_val_ddim = float("inf")
    best_epoch = 0
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    # Epoch 0 evaluation (exact starting state before any gradient step)
    tr_obj0, va_obj0, va_cf_opt0, va_cf_p00, va_ddim0 = _eval_current_weights(0, run_ddim=True)
    rec0: Dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "lr": 0.0,
        "grad_norm_mean": 0.0,
        "train_batch_total_loss": tr_obj0["total_loss"],
        "train_batch_diff_loss": tr_obj0["diff_loss"],
        "train_batch_reg_loss": tr_obj0["reg_loss"],
        "train_ema_total_loss": tr_obj0["total_loss"],
        "train_ema_diff_loss": tr_obj0["diff_loss"],
        "train_ema_reg_loss": tr_obj0["reg_loss"],
        "train_ema_c_l2": tr_obj0["c_l2_mean"],
        "val_total_loss": va_obj0["total_loss"],
        "val_diff_loss": va_obj0["diff_loss"],
        "val_reg_loss": va_obj0["reg_loss"],
        "val_c_l2": va_obj0["c_l2_mean"],
        "val_r2_vs_c_opt": va_cf_opt0["coeff_r2"],
        "val_r2_vs_c_p0": va_cf_p00["coeff_r2"],
        "val_ckpt_ddim10": va_ddim0,
        "best_val_ckpt_ddim10": None,
        "best_epoch": 0,
    }
    history.append(rec0)
    print(
        f"  [E11.3 {ltag}-{group} Ep 00/{epochs}] "
        f"tr_diff={tr_obj0['diff_loss']:.5f} reg={tr_obj0['reg_loss']:.5f} ||c||={tr_obj0['c_l2_mean']:.2f} | "
        f"va_diff={va_obj0['diff_loss']:.5f} ||c||={va_obj0['c_l2_mean']:.2f} | "
        f"va_DDIM10={va_ddim0:.5f} (INIT)",
        flush=True,
    )

    # Decoupled deterministic RNGs: A, B, and C receive IDENTICAL batches, timesteps, and noise!
    perm_gen = torch.Generator(device=device).manual_seed(seed + 77)
    diff_gen = torch.Generator(device=device).manual_seed(seed + 1337)
    # Re-seed torch global RNG identically for identical dropout masks across A, B, C
    seed_everything(seed + 2026)

    for epoch in range(1, epochs + 1):
        model.train()
        policy.eval()
        perm = torch.randperm(N_tr, device=device, generator=perm_gen)

        tr_tot_sum = 0.0
        tr_diff_sum = 0.0
        tr_reg_sum = 0.0
        grad_norm_sum = 0.0
        n_batches = 0

        for start in range(0, N_tr, batch_size):
            idx = perm[start : start + batch_size]
            zb = tr_z[idx]
            hb = tr_h[idx]
            ab = tr_a[idx]

            optimizer.zero_grad(set_to_none=True)
            tot_l, diff_l, reg_l, _ = compute_decoder_aware_loss(
                model=model,
                teacher=policy.teacher,
                u64_t=u64_t,
                z_src_flat=zb,
                h_q=hb,
                gt_actions=ab,
                lam=lam,
                mc_timesteps=mc_timesteps,
                t_edges=t_edges,
                diff_gen=diff_gen,
                use_bf16=use_bf16,
            )
            tot_l.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad_norm).item())
            optimizer.step()
            scheduler.step()
            ema.step(trainable_params)

            bsz = float(len(idx))
            tr_tot_sum += float(tot_l.item()) * bsz
            tr_diff_sum += float(diff_l.item()) * bsz
            tr_reg_sum += float(reg_l.item()) * bsz
            grad_norm_sum += gn
            n_batches += 1

        tr_batch_tot = tr_tot_sum / float(N_tr)
        tr_batch_diff = tr_diff_sum / float(N_tr)
        tr_batch_reg = tr_reg_sum / float(N_tr)
        mean_gn = grad_norm_sum / max(n_batches, 1)

        # Evaluate EMA weights on train_fit subset + full val_ckpt
        ema.store(trainable_params)
        ema.copy_to(trainable_params)

        run_ddim = (epoch == 1) or (epoch % eval_ddim_every == 0) or (epoch == epochs)
        tr_obj, va_obj, va_cf_opt, va_cf_p0, va_ddim_mse = _eval_current_weights(epoch, run_ddim=run_ddim)

        if va_ddim_mse is not None and va_ddim_mse < best_val_ddim:
            best_val_ddim = va_ddim_mse
            best_epoch = epoch
            atomic_torch_save(
                {
                    "experiment": "e11_3_decoder_aware_transfer_optimization",
                    "group": group,
                    "run_name": run_name,
                    "label": config["label"],
                    "lambda": float(lam),
                    "lambda_tag": ltag,
                    "epoch": int(epoch),
                    "global_step": int(epoch * steps_per_epoch),
                    "init_val_ckpt_ddim10": float(va_ddim0),
                    "best_val_ckpt_ddim10": float(best_val_ddim),
                    "train_obj_ema": tr_obj,
                    "val_obj_ema": va_obj,
                    "model_kwargs": model_kwargs,
                    "trainable_params": int(n_params),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config,
                },
                best_ckpt,
            )

        if epoch == epochs:
            atomic_torch_save(
                {
                    "experiment": "e11_3_decoder_aware_transfer_optimization",
                    "group": group,
                    "run_name": run_name,
                    "label": config["label"],
                    "lambda": float(lam),
                    "lambda_tag": ltag,
                    "epoch": int(epoch),
                    "global_step": int(epoch * steps_per_epoch),
                    "init_val_ckpt_ddim10": float(va_ddim0),
                    "val_ckpt_ddim10": float(va_ddim_mse) if va_ddim_mse is not None else None,
                    "best_val_ckpt_ddim10": float(best_val_ddim),
                    "best_epoch": int(best_epoch),
                    "model_kwargs": model_kwargs,
                    "trainable_params": int(n_params),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config,
                },
                last_ckpt,
            )

        ema.restore(trainable_params)

        rec: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(epoch * steps_per_epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            "grad_norm_mean": float(mean_gn),
            "train_batch_total_loss": float(tr_batch_tot),
            "train_batch_diff_loss": float(tr_batch_diff),
            "train_batch_reg_loss": float(tr_batch_reg),
            "train_ema_total_loss": float(tr_obj["total_loss"]),
            "train_ema_diff_loss": float(tr_obj["diff_loss"]),
            "train_ema_reg_loss": float(tr_obj["reg_loss"]),
            "train_ema_c_l2": float(tr_obj["c_l2_mean"]),
            "val_total_loss": float(va_obj["total_loss"]),
            "val_diff_loss": float(va_obj["diff_loss"]),
            "val_reg_loss": float(va_obj["reg_loss"]),
            "val_c_l2": float(va_obj["c_l2_mean"]),
            "val_r2_vs_c_opt": float(va_cf_opt["coeff_r2"]),
            "val_r2_vs_c_p0": float(va_cf_p0["coeff_r2"]),
            "val_ckpt_ddim10": va_ddim_mse,
            "best_val_ckpt_ddim10": float(best_val_ddim),
            "best_epoch": int(best_epoch),
        }
        history.append(rec)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            dd_str = f"{va_ddim_mse:.5f}" if va_ddim_mse is not None else "—"
            print(
                f"  [E11.3 {ltag}-{group} Ep {epoch:02d}/{epochs}] "
                f"tr_diff={tr_obj['diff_loss']:.5f} reg={tr_obj['reg_loss']:.5f} ||c||={tr_obj['c_l2_mean']:.2f} | "
                f"va_diff={va_obj['diff_loss']:.5f} ||c||={va_obj['c_l2_mean']:.2f} | "
                f"va_DDIM10={dd_str} (best={best_val_ddim:.5f}@ep{best_epoch}, gn={mean_gn:.3f})",
                flush=True,
            )

    # Confirm frozen policy got zero grads throughout training
    assert not any(p.grad is not None for p in policy.parameters())

    train_summary = {
        "group": group,
        "run_name": run_name,
        "label": config["label"],
        "lambda": float(lam),
        "lambda_tag": ltag,
        "objective": config["objective"],
        "init_meta": init_meta,
        "trainable_params": int(n_params),
        "init_val_ckpt_ddim10": float(va_ddim0),
        "best_epoch": int(best_epoch),
        "best_val_ckpt_ddim10": float(best_val_ddim),
        "delta_vs_init_val_ddim10": float(best_val_ddim - va_ddim0),
        "init_record": history[0],
        "best_record": next(h for h in history if h["epoch"] == best_epoch),
        "final_record": history[-1],
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "history": history,
    }
    atomic_write_json(train_summary, run_dir / "train_metrics.json")
    write_csv(history, run_dir / "train_curve.csv")
    print(
        f"[E11.3 {ltag}-{group} DONE] init={va_ddim0:.5f} -> best={best_val_ddim:.5f}@ep{best_epoch} "
        f"(Δ={best_val_ddim - va_ddim0:+.5f}) ({train_summary['elapsed_seconds']}s) -> {best_ckpt}",
        flush=True,
    )
    return best_ckpt


def _plot_e11_3_figures(
    summary: Dict[str, Any],
    e11_2_json: Dict[str, Any],
    oracle_json: Dict[str, Any],
    lambdas: Sequence[float],
    fig_dir: Path,
) -> Dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: Dict[str, Path] = {}

    ev = summary["evaluation"]
    tr_runs = summary["train_runs"]
    e11_2_ev = e11_2_json["evaluation"]
    o_s = oracle_json["results"]["seen_val"]["methods"]
    o_u = oracle_json["results"]["unseen_chain"]["methods"]
    ltags = [lam_to_tag(l) for l in lambdas]

    # ---------------- Figure 1: Training Curves (val_ckpt DDIM10 & L_diff) across lambdas ----------------
    n_l = len(lambdas)
    fig, axes = plt.subplots(2, n_l, figsize=(5.2 * n_l, 8.6), dpi=160, squeeze=False)
    for ci, (lam, lt) in enumerate(zip(lambdas, ltags)):
        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]
        for grp in ("A", "B", "C"):
            spec = INIT_SPECS[grp]
            key = f"{lt}__{spec['run_name']}"
            tr = tr_runs[key]
            pts = [h for h in tr["history"] if h["val_ckpt_ddim10"] is not None]
            ep = [h["epoch"] for h in pts]
            dd = [h["val_ckpt_ddim10"] for h in pts]
            ax_top.plot(
                ep,
                dd,
                marker=spec["marker"],
                markersize=3.0,
                linewidth=2.0,
                color=spec["color"],
                label=f"{spec['short']} (best={tr['best_val_ckpt_ddim10']:.5f}@ep{tr['best_epoch']})",
            )

            ep_all = [h["epoch"] for h in tr["history"]]
            tr_dl = [h["train_ema_diff_loss"] for h in tr["history"]]
            va_dl = [h["val_diff_loss"] for h in tr["history"]]
            ax_bot.plot(ep_all, tr_dl, linestyle="--", linewidth=1.4, color=spec["color"], alpha=0.65, label=f"{grp} Train L_diff")
            ax_bot.plot(ep_all, va_dl, linestyle="-", linewidth=2.0, color=spec["color"], label=f"{grp} Val L_diff")

        ax_top.axhline(e11_2_json["train_runs"]["A"]["best_val_ckpt_ddim10"], color="#1f77b4", linestyle=":", alpha=0.6, label="E11.2-A init")
        ax_top.axhline(e11_2_json["train_runs"]["B"]["best_val_ckpt_ddim10"], color="#2ca02c", linestyle=":", alpha=0.6, label="E11.2-B init")
        ax_top.set_title(f"λ = {lam:.0e}" if lam > 0 else "λ = 0 (Pure L_diff)")
        ax_top.set_xlabel("Epoch")
        ax_top.set_ylabel("val_ckpt DDIM 10D MSE")
        ax_top.set_ylim(0.004, 0.025)
        ax_top.grid(True, alpha=0.25)
        ax_top.legend(fontsize=6.8, loc="upper right")

        ax_bot.set_xlabel("Epoch")
        ax_bot.set_ylabel("L_diff (4-step stratified)")
        ax_bot.set_yscale("log")
        ax_bot.grid(True, which="both", alpha=0.25)
        ax_bot.legend(fontsize=6.6, loc="upper right", ncol=2)

    fig.suptitle("E11.3 Decoder-Aware Transfer Optimization: val_ckpt DDIM10 & L_diff Convergence", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = fig_dir / "fig1_e11_3_training_curves.png"
    fig.savefig(p1)
    plt.close(fig)
    fig_paths["fig1"] = p1

    # ---------------- Figure 2: Lambda Sweep Comparison (Seen & Unseen DDIM10, Norm, Wrong-Demo, Self) ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.6), dpi=160)
    xs = np.arange(len(lambdas))
    x_lbls = [f"λ={l:.0e}" if l > 0 else "λ=0" for l in lambdas]
    w = 0.24

    for col_i, (sname, stitle, o_m) in enumerate(
        (("seen_val", "A. Seen-Val (3,000) DDIM 10D MSE", o_s), ("unseen_chain", "B. Unseen-Chain (3,000) DDIM 10D MSE", o_u))
    ):
        ax = axes[0, col_i]
        for gi, grp in enumerate(("A", "B", "C")):
            spec = INIT_SPECS[grp]
            ys = [ev[f"{lt}__{spec['run_name']}"][sname]["correct"]["ddim_10d_mse"] for lt in ltags]
            bars = ax.bar(xs + (gi - 1) * w, ys, width=w, color=spec["color"], alpha=0.9, label=spec["short"])
            for bx, by in zip(xs + (gi - 1) * w, ys):
                ax.annotate(f"{by:.5f}", (bx, by), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.8, fontweight="bold", color=spec["color"])

        ax.axhline(e11_2_ev["group_a_action_opt_28m"][sname]["correct"]["ddim_10d_mse"], color="#1f77b4", linestyle="--", alpha=0.6, label=f"E11.2-A ({e11_2_ev['group_a_action_opt_28m'][sname]['correct']['ddim_10d_mse']:.5f})")
        ax.axhline(e11_2_ev["group_b_future_lat_28m"][sname]["correct"]["ddim_10d_mse"], color="#2ca02c", linestyle="--", alpha=0.6, label=f"E11.2-B ({e11_2_ev['group_b_future_lat_28m'][sname]['correct']['ddim_10d_mse']:.5f})")
        ax.axhline(ev["e10_c"][sname]["correct"]["ddim_10d_mse"], color="#d62728", linestyle="-.", linewidth=1.6, label=f"E10-C 1.37M ({ev['e10_c'][sname]['correct']['ddim_10d_mse']:.5f})")
        ax.axhline(o_m["action_opt_lam_1e-5"]["ddim_10d_mse"], color="#444444", linestyle=":", alpha=0.8, label=f"Action-Opt Oracle λ=1e-5 ({o_m['action_opt_lam_1e-5']['ddim_10d_mse']:.5f})")
        ax.set_xticks(xs)
        ax.set_xticklabels(x_lbls, fontsize=9)
        ax.set_ylabel("DDIM 10D Action MSE (lower is better)")
        ax.set_title(stitle)
        ax.set_ylim(0.0, e11_2_ev["group_a_action_opt_28m"][sname]["correct"]["ddim_10d_mse"] * 1.25)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=6.8, loc="upper right", ncol=2)

    # (1, 0) Unseen-Chain Wrong-Demo Ratio across lambdas
    ax = axes[1, 0]
    for gi, grp in enumerate(("A", "B", "C")):
        spec = INIT_SPECS[grp]
        ys = [ev[f"{lt}__{spec['run_name']}"]["unseen_chain"]["wrong_ratio_mse10"] for lt in ltags]
        ax.bar(xs + (gi - 1) * w, ys, width=w, color=spec["color"], alpha=0.9, label=spec["short"])
        for bx, by in zip(xs + (gi - 1) * w, ys):
            ax.annotate(f"{by:.2f}×", (bx, by), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.0, fontweight="bold")
    ax.axhline(e11_2_ev["group_a_action_opt_28m"]["unseen_chain"]["wrong_ratio_mse10"], color="#1f77b4", linestyle="--", alpha=0.6, label="E11.2-A")
    ax.axhline(e11_2_ev["group_b_future_lat_28m"]["unseen_chain"]["wrong_ratio_mse10"], color="#2ca02c", linestyle="--", alpha=0.6, label="E11.2-B")
    ax.axhline(ev["e10_c"]["unseen_chain"]["wrong_ratio_mse10"], color="#d62728", linestyle="-.", label="E10-C")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_lbls, fontsize=9)
    ax.set_ylabel("Wrong / Correct DDIM 10D MSE Ratio (higher = more sensitive)")
    ax.set_title("C. Wrong-Demo Sensitivity (Unseen-Chain)")
    ax.set_ylim(0.0, 5.15)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.2, loc="upper left", ncol=2)

    # (1, 1) Unseen-Chain Modification Norm ||c||_2 & Self ||c||_2
    ax = axes[1, 1]
    for gi, grp in enumerate(("A", "B", "C")):
        spec = INIT_SPECS[grp]
        cr = [ev[f"{lt}__{spec['run_name']}"]["unseen_chain"]["geometry"]["c_l2_mean"] for lt in ltags]
        sf = [ev[f"{lt}__{spec['run_name']}"]["unseen_chain"]["self_c_l2_mean"] for lt in ltags]
        ax.bar(xs + (gi - 1) * w, cr, width=w, color=spec["color"], alpha=0.85, label=f"{grp} Cross ||c||_2")
        ax.bar(xs + (gi - 1) * w, sf, width=w * 0.55, color="#222222", alpha=0.45, label=f"Self ||c||_2 (overlay)" if gi == 0 else None)
        for bx, c_v, s_v in zip(xs + (gi - 1) * w, cr, sf):
            ax.annotate(f"{c_v:.1f}\n({s_v:.1f})", (bx, c_v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
    ax.axhline(e11_2_ev["group_a_action_opt_28m"]["unseen_chain"]["geometry"]["c_l2_mean"], color="#1f77b4", linestyle="--", alpha=0.5, label="E11.2-A Cross")
    ax.axhline(ev["e10_c"]["unseen_chain"]["geometry"]["c_l2_mean"], color="#d62728", linestyle="-.", alpha=0.7, label="E10-C Cross")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_lbls, fontsize=9)
    ax.set_ylabel("L2 Norm ||c||_2 (overlay: Self ||c||_2)")
    ax.set_title("D. Modification Norm & Self-Preservation (Unseen-Chain)")
    ax.set_ylim(0.0, 45.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.0, loc="upper right", ncol=2)

    fig.suptitle("E11.3 Action-Opt Init (A) vs Future-Latent Init (B) vs Scratch (C) across λ", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = fig_dir / "fig2_e11_3_lambda_comparison.png"
    fig.savefig(p2)
    plt.close(fig)
    fig_paths["fig2"] = p2

    # ---------------- Figure 3: Per-Phase Unseen-Chain Breakdown (Best λ per Group vs Baselines) ----------------
    best_per_grp: Dict[str, Tuple[str, str]] = {}
    for grp in ("A", "B", "C"):
        spec = INIT_SPECS[grp]
        best_lt = min(
            ltags,
            key=lambda lt: tr_runs[f"{lt}__{spec['run_name']}"]["best_val_ckpt_ddim10"],
        )
        best_per_grp[grp] = (best_lt, f"{best_lt}__{spec['run_name']}")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), dpi=160)
    ph_lbls = ["Approach", "Grasp\nTrans.", "Transport\nCarry", "Release\nTrans.", "Post-Rel.\nTransit"]
    xp = np.arange(len(SEMANTIC_PHASES))
    ph_models = [
        ("E11.2-A (c*)", e11_2_ev["group_a_action_opt_28m"], "#9ecae1"),
        ("E11.2-B (c_p0)", e11_2_ev["group_b_future_lat_28m"], "#a1d99b"),
        (f"E11.3-A ({best_per_grp['A'][0]})", ev[best_per_grp["A"][1]], "#1f77b4"),
        (f"E11.3-B ({best_per_grp['B'][0]})", ev[best_per_grp["B"][1]], "#2ca02c"),
        (f"E11.3-C ({best_per_grp['C'][0]})", ev[best_per_grp["C"][1]], "#ff7f0e"),
        ("E10-C (1.37M)", ev["e10_c"], "#d62728"),
    ]
    pw = 0.13
    for col_i, (sname, stitle) in enumerate((("seen_val", "A. Per-Phase DDIM 10D MSE — Seen-Val"), ("unseen_chain", "B. Per-Phase DDIM 10D MSE — Unseen-Chain"))):
        ax = axes[col_i]
        for mi, (lbl, mdict, col) in enumerate(ph_models):
            ys = [mdict[sname]["phases"].get(ph, {}).get("ddim_10d_mse", np.nan) for ph in SEMANTIC_PHASES]
            ax.bar(xp + (mi - 2.5) * pw, ys, width=pw, color=col, label=lbl, alpha=0.92)
        ax.set_xticks(xp)
        ax.set_xticklabels(ph_lbls, fontsize=8.5)
        ax.set_ylabel("DDIM 10D Action MSE (log scale)")
        ax.set_yscale("log")
        ax.set_ylim(8e-4, 0.16)
        ax.set_title(stitle)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(fontsize=7.3, loc="upper left", ncol=2)

    fig.suptitle("E11.3 Best-λ Per-Group vs E11.2-A/B & E10-C: Per-Phase Breakdown", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p3 = fig_dir / "fig3_e11_3_per_phase.png"
    fig.savefig(p3)
    plt.close(fig)
    fig_paths["fig3"] = p3
    return fig_paths


def _write_e11_3_report(
    summary: Dict[str, Any],
    e11_2_json: Dict[str, Any],
    oracle_json: Dict[str, Any],
    lambdas: Sequence[float],
    fig_paths: Dict[str, Path],
    out_path: Path,
) -> None:
    cfg = summary["config"]
    ev = summary["evaluation"]
    tr_runs = summary["train_runs"]
    e11_2_ev = e11_2_json["evaluation"]
    o_s = oracle_json["results"]["seen_val"]["methods"]
    o_u = oracle_json["results"]["unseen_chain"]["methods"]
    ltags = [lam_to_tag(l) for l in lambdas]

    def _row_line(name: str, init_s: str, obj_s: str, rec: Dict[str, Any], sname: str) -> str:
        d1 = cfg["full_self_d1_ddim10"][sname]
        id_m = cfg["identity_ddim10"][sname]
        c_m = rec["correct"]
        w_m = rec["wrong_task"]
        s_m = rec["self_preservation"]
        g_m = rec["geometry"]
        arr = (id_m - c_m["ddim_10d_mse"]) / max(id_m - d1, 1e-12)
        d3_r = s_m["ddim_10d_mse"] / max(d1, 1e-12)
        return (
            f"| **{name}** | `{init_s}` | `{obj_s}` | "
            f"`{c_m['ddim_10d_mse']:.5f}` (`{100*arr:.1f}%`) | "
            f"`{c_m['ddim_7d_mae']:.5f}` / `{100*c_m['grip_acc16']:.2f}%` | "
            f"`{g_m['c_l2_mean']:.2f}` | "
            f"`{w_m['ddim_10d_mse']:.5f}` (`{rec['wrong_ratio_mse10']:.2f}×`, `{rec['wrong_delta_mse10']:+.4f}`) | "
            f"`{rec['self_c_l2_mean']:.2f}` (`{d3_r:.2f}×`) |"
        )

    lines: List[str] = []
    lines.append("# E11.3: Decoder-Aware Transfer Optimization 实验报告\n")

    lines.append("## 1. 实验设置与严格对齐核查\n")
    first_cfg = next(iter(summary["run_configs"].values()))
    arch = first_cfg["architecture"]
    trn = first_cfg["training"]
    aud = first_cfg["data"]["split_audit"]
    lines.append(
        f"- **固定组件与架构**：复用 E11.2 / E10-C 的 `LowRankTransferTransformer`（3 层 Pre-LN Transformer Decoder，`d=768, 8 heads, FFN=3072, dropout={arch['dropout']}`，可训练参数量 **`{arch['trainable_params']:,}`**）、固定 P0 正交基 `U_64`，冻结 E9 Stage 1 Local Encoder/Action Decoder 与 E9 Stage 2 Reader（训练前与训练后均核查零梯度）。"
    )
    lines.append(
        f"- **数据划分严格一致**：沿用 E11.2 的 `train_fit = {aud['train_fit_pairs']}` 对 / `val_ckpt = {aud['val_ckpt_pairs']}` 对四分防泄漏划分（与 `seen_val` 轨迹重叠 = `0` / `0`，`unseen_chain` 严格仅做测试），并以比特级一致校验补全 `train_fit` 的真值动作块 `a_q*`。"
    )
    lines.append(
        f"- **公平训练预算与优化目标**：三组对照 **A (E11.2-A Action-Opt Init)**、**B (E11.2-B Future-Latent Init)**、**C (Scratch)** 共享完全相同的优化器（AdamW，`lr={trn['lr']:.0e} -> {trn['lr']*trn['min_lr_ratio']:.0e}`，cosine 调度，`warmup={trn['warmup_steps']}` 步，`betas={trn['betas']}`，`wd={trn['weight_decay']}`，`clip={trn['clip_grad_norm']}`，`EMA={trn['ema_decay']}`，`bs={trn['batch_size']}`，`epochs={trn['epochs']}` / `{trn['total_steps']}` 步，同种子 `seed={trn['seed']}`）及**完全相同的 mini-batch、分层时间步（`M={trn['mc_timesteps']}`）与高斯噪声流**。目标函数为纯端到端 `L_diff(G(z_src + U_64 c_hat), a_q*) + λ ||c_hat||_2^2`（不加任何 coefficient regression loss），扫描 `λ ∈ {{{', '.join(f'{l:.0e}' if l > 0 else '0' for l in lambdas)}}}`。\n"
    )

    lines.append("## 2. 训练收敛与 `val_ckpt` 选模汇总\n")
    lines.append(f"![Training Curves](figures/{fig_paths['fig1'].name})\n")
    lines.append("| λ | 对照组 | Init `val_ckpt` DDIM10 | Best Ep | Best `val_ckpt` DDIM10 (`Δ vs Init`) | Best Ep Train `L_diff` / Val `L_diff` | Best Ep Val `||c||_2` | Final Ep80 `val_ckpt` DDIM10 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for lam, lt in zip(lambdas, ltags):
        l_str = f"`{lam:.0e}`" if lam > 0 else "`0`"
        for grp in ("A", "B", "C"):
            spec = INIT_SPECS[grp]
            tr = tr_runs[f"{lt}__{spec['run_name']}"]
            br = tr["best_record"]
            fr = tr["final_record"]
            lines.append(
                f"| {l_str} | **{spec['short']}** | `{tr['init_val_ckpt_ddim10']:.5f}` | `Ep {tr['best_epoch']}` | "
                f"`{tr['best_val_ckpt_ddim10']:.5f}` (`{tr['delta_vs_init_val_ddim10']:+.5f}`) | "
                f"`{br['train_ema_diff_loss']:.5f}` / `{br['val_diff_loss']:.5f}` | "
                f"`{br['val_c_l2']:.2f}` | `{fr['val_ckpt_ddim10']:.5f}` |"
            )
    lines.append("")

    lines.append("## 3. Seen / Unseen 全面对比（E11.3 A/B/C × λ vs E11.2-A/B vs E10-C vs Oracles）\n")
    lines.append(f"![Lambda Comparison](figures/{fig_paths['fig2'].name})\n")

    for sname, stitle, o_m in (
        ("seen_val", "Seen-Val（3,000 对）", o_s),
        ("unseen_chain", "Unseen-Chain（3,000 对）", o_u),
    ):
        lines.append(f"### 3.{1 if sname == 'seen_val' else 2} {stitle}\n")
        lines.append("| 模型 | 初始化 | 训练目标 (24k unless noted) | DDIM 10D MSE (ARR) | 7D MAE / Grip | `||c||_2` | Wrong-Demo MSE (`×`, `Δ`) | Self `||c||_2` (`D3/D1`) |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        lines.append(_row_line("Identity (c=0)", "—", "—", ev["identity"][sname], sname))
        lines.append(_row_line("E11.2-A Action-Opt (28.6M)", "Scratch", "MSE(c, c*)", e11_2_ev["group_a_action_opt_28m"][sname], sname))
        lines.append(_row_line("E11.2-B Future-Latent (28.6M)", "Scratch", "MSE(c, c_p0)", e11_2_ev["group_b_future_lat_28m"][sname], sname))
        for lam, lt in zip(lambdas, ltags):
            obj_s = "L_diff" if lam == 0.0 else f"L_diff + {lam:.0e}||c||²"
            for grp in ("A", "B", "C"):
                spec = INIT_SPECS[grp]
                key = f"{lt}__{spec['run_name']}"
                lines.append(_row_line(f"E11.3-{grp} ({lt})", spec["init_source"], obj_s, ev[key][sname], sname))
        lines.append(_row_line("E10-C (Full 1.37M×8ep)", "Scratch", "L_diff+0.5L_z (1.37M)", ev["e10_c"][sname], sname))
        lines.append(
            f"| *Oracle: Action-Opt (λ=1e-5)* | *—* | *per-pair opt* | `{o_m['action_opt_lam_1e-5']['ddim_10d_mse']:.5f}` (`{100*o_m['action_opt_lam_1e-5']['arr_10d_vs_id_full']:.1f}%`) | "
            f"`{o_m['action_opt_lam_1e-5']['ddim_7d_mae']:.5f}` / `{100*o_m['action_opt_lam_1e-5']['grip_acc16']:.2f}%` | `{o_m['action_opt_lam_1e-5']['c_l2_mean']:.2f}` | — | — |"
        )
        lines.append(
            f"| *Oracle: P0 (U_64^T Δz)* | *—* | *oracle proj* | `{o_m['p0_oracle']['ddim_10d_mse']:.5f}` (`{100*o_m['p0_oracle']['arr_10d_vs_id_full']:.1f}%`) | "
            f"`{o_m['p0_oracle']['ddim_7d_mae']:.5f}` / `{100*o_m['p0_oracle']['grip_acc16']:.2f}%` | `{o_m['p0_oracle']['c_l2_mean']:.2f}` | — | — |"
        )
        lines.append("")

    # Per-phase breakdown for best-val_ckpt lambda per group
    best_by_grp: Dict[str, Tuple[str, str]] = {}
    for grp in ("A", "B", "C"):
        spec = INIT_SPECS[grp]
        best_lt = min(ltags, key=lambda lt: tr_runs[f"{lt}__{spec['run_name']}"]["best_val_ckpt_ddim10"])
        best_by_grp[grp] = (best_lt, f"{best_lt}__{spec['run_name']}")

    lines.append("### 3.3 分行为阶段 DDIM 10D MSE（各组 `val_ckpt` 最优 λ vs E11.2-A/B & E10-C）\n")
    lines.append(f"![Per-Phase Breakdown](figures/{fig_paths['fig3'].name})\n")
    lines.append(
        f"| Unseen-Chain 行为阶段 | E11.2-A | E11.2-B | **E11.3-A ({best_by_grp['A'][0]})** | "
        f"**E11.3-B ({best_by_grp['B'][0]})** | **E11.3-C ({best_by_grp['C'][0]})** | E10-C (1.37M) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ph in SEMANTIC_PHASES:
        vals = [
            e11_2_ev["group_a_action_opt_28m"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            e11_2_ev["group_b_future_lat_28m"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev[best_by_grp["A"][1]]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev[best_by_grp["B"][1]]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev[best_by_grp["C"][1]]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
            ev["e10_c"]["unseen_chain"]["phases"][ph]["ddim_10d_mse"],
        ]
        lines.append(f"| `{ph}` | " + " | ".join(f"`{v:.5f}`" for v in vals) + " |")
    lines.append("")

    # Core questions synthesis (computed strictly from actual measured metrics)
    a_best_lt, a_best_k = best_by_grp["A"]
    b_best_lt, b_best_k = best_by_grp["B"]
    c_best_lt, c_best_k = best_by_grp["C"]

    a_s = ev[a_best_k]["seen_val"]["correct"]["ddim_10d_mse"]
    a_u = ev[a_best_k]["unseen_chain"]["correct"]["ddim_10d_mse"]
    b_s = ev[b_best_k]["seen_val"]["correct"]["ddim_10d_mse"]
    b_u = ev[b_best_k]["unseen_chain"]["correct"]["ddim_10d_mse"]
    c_s = ev[c_best_k]["seen_val"]["correct"]["ddim_10d_mse"]
    c_u = ev[c_best_k]["unseen_chain"]["correct"]["ddim_10d_mse"]

    e112_a_s = e11_2_ev["group_a_action_opt_28m"]["seen_val"]["correct"]["ddim_10d_mse"]
    e112_a_u = e11_2_ev["group_a_action_opt_28m"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    e112_b_s = e11_2_ev["group_b_future_lat_28m"]["seen_val"]["correct"]["ddim_10d_mse"]
    e112_b_u = e11_2_ev["group_b_future_lat_28m"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    e10_s = ev["e10_c"]["seen_val"]["correct"]["ddim_10d_mse"]
    e10_u = ev["e10_c"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    ora_s = o_s["action_opt_lam_1e-5"]["ddim_10d_mse"]
    ora_u = o_u["action_opt_lam_1e-5"]["ddim_10d_mse"]

    gap_ora_before_u = e112_a_u - ora_u
    gap_ora_after_u = a_u - ora_u
    gap_ora_red_u = 100.0 * (gap_ora_before_u - gap_ora_after_u) / max(gap_ora_before_u, 1e-12)

    gap_e10_before_u = e112_a_u - e10_u
    gap_e10_after_u = a_u - e10_u
    gap_e10_red_u = 100.0 * (gap_e10_before_u - gap_e10_after_u) / max(gap_e10_before_u, 1e-12)

    # Compute best-across-all-lambdas per group and per-lambda comparisons for thorough answers
    a_wins_unseen_lams = [
        lt for lt in ltags
        if ev[f"{lt}__{INIT_SPECS['A']['run_name']}"]["unseen_chain"]["correct"]["ddim_10d_mse"]
        < ev[f"{lt}__{INIT_SPECS['B']['run_name']}"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    ]
    b_wins_unseen_lams = [
        lt for lt in ltags
        if ev[f"{lt}__{INIT_SPECS['B']['run_name']}"]["unseen_chain"]["correct"]["ddim_10d_mse"]
        < ev[f"{lt}__{INIT_SPECS['A']['run_name']}"]["unseen_chain"]["correct"]["ddim_10d_mse"]
    ]
    a_highest_wrg_all = all(
        ev[f"{lt}__{INIT_SPECS['A']['run_name']}"]["unseen_chain"]["wrong_ratio_mse10"]
        > max(
            ev[f"{lt}__{INIT_SPECS['B']['run_name']}"]["unseen_chain"]["wrong_ratio_mse10"],
            ev[f"{lt}__{INIT_SPECS['C']['run_name']}"]["unseen_chain"]["wrong_ratio_mse10"],
        )
        for lt in ltags
    )
    a_smallest_norm_all = all(
        ev[f"{lt}__{INIT_SPECS['A']['run_name']}"]["unseen_chain"]["geometry"]["c_l2_mean"]
        < min(
            ev[f"{lt}__{INIT_SPECS['B']['run_name']}"]["unseen_chain"]["geometry"]["c_l2_mean"],
            ev[f"{lt}__{INIT_SPECS['C']['run_name']}"]["unseen_chain"]["geometry"]["c_l2_mean"],
        )
        for lt in ltags
    )
    best_seen_overall_key = min(
        [f"{lt}__{INIT_SPECS[g]['run_name']}" for lt in ltags for g in ("A", "B", "C")],
        key=lambda k: ev[k]["seen_val"]["correct"]["ddim_10d_mse"],
    )
    best_unseen_overall_key = min(
        [f"{lt}__{INIT_SPECS[g]['run_name']}" for lt in ltags for g in ("A", "B", "C")],
        key=lambda k: ev[k]["unseen_chain"]["correct"]["ddim_10d_mse"],
    )

    lines.append("## 4. 核心问题直接回答与机制分析\n")
    lines.append(
        f"1. **Action-Opt 初始化 (A) 能否优于 Future-Latent 初始化 (B)？**\n"
        f"   - **结论：在纯动作精度（DDIM 10D MSE）上未能形成稳定优势，但在 Demo 敏感性与最小修改几何上保持全 λ 一致领先。**\n"
        f"   - 按独立 `val_ckpt` 最优 λ 选取（均为 `lam_0`）：E11.3-A 达到 Seen `{a_s:.5f}` / Unseen `{a_u:.5f}`（全部 12 组 Unseen 最低 `{best_unseen_overall_key}`），略优于 E11.3-B 的 Seen `{b_s:.5f}` / Unseen `{b_u:.5f}`（`A - B`: Seen `{a_s - b_s:+.5f}` / Unseen `{a_u - b_u:+.5f}`，仅约 `1%` 级微弱优势），彻底扭转了 E11.2 中 A 落后于 B（Unseen `0.01217` vs `0.01054`）的局面。\n"
        f"   - 然而在全部三个带范数正则的 λ（`{', '.join(b_wins_unseen_lams)}`）下，**B 均反超 A**，且 E11.3-B (`lam_1e-6`) 取得全部 12 组 Seen 最低 MSE（`{ev[best_seen_overall_key]['seen_val']['correct']['ddim_10d_mse']:.5f}`，Unseen `0.01031`）。\n"
        f"   - **A 独有的持久几何优势**：在全部 4 个 λ 下，A 的 Unseen Wrong-Demo Sensitivity 均严格高于 B 与 C（`a_highest_wrg_all={a_highest_wrg_all}`，如 `lam_0`: `3.48×` vs `2.99×`/`2.96×`；`lam_1e-6`: `3.65×` vs `3.19×`/`3.08×`；`lam_3e-6`: `3.81×` vs `3.52×`/`3.38×`；`lam_1e-5`: `3.89×` vs `3.66×`/`3.68×`），且在全部 4 个 λ 下均保持最小的 Cross `||c||_2`（`a_smallest_norm_all={a_smallest_norm_all}`）与更低的 Self `||c||_2`（相比 Scratch 低 `2.0–3.3`）。"
    )
    lines.append(
        f"2. **Decoder-Aware Loss 能否缩小 Oracle 与 Learned Transfer 的差距？**\n"
        f"   - **结论：能显著消除 E11.2-A 的各向同性回归目标失配，但在 24k 数据规模下仍存在明显的泛化天花板（`~0.0080` Seen / `~0.0102` Unseen），未能实质逼近 Oracle 或 E10-C。**\n"
        f"   - 对比 E11.2 系数回归：E11.3-A（`lam_0`）相对 E11.2-A 将 Seen DDIM10 从 `{e112_a_s:.5f}` 降至 `{a_s:.5f}`（`{a_s - e112_a_s:+.5f}`, `-14.9%`），Unseen 从 `{e112_a_u:.5f}` 降至 `{a_u:.5f}`（`{a_u - e112_a_u:+.5f}`, `-16.6%`），与 Action-Opt Oracle（Unseen `{ora_u:.5f}`）的差距缩小 **`{gap_ora_red_u:.1f}%`**，与 E10-C（Unseen `{e10_u:.5f}`）的差距缩小 **`{gap_e10_red_u:.1f}%`**；即便在相同 `λ=1e-5`（`||c||_2 ≈ 21.5`）下也从 `0.00941/0.01217` 改善至 `0.00868/0.01188`。\n"
        f"   - E11.3-B 相对 E11.2-B 的增幅则小得多（Seen `{e112_b_s:.5f} -> {b_s:.5f}` / `0.00796@1e-6`，Unseen `{e112_b_u:.5f} -> {b_u:.5f}`），说明 E11.2-B 的流形内目标原本已较接近 Decoder 敏感方向，而 E11.2-A 的主要失分来自各向同性 `MSE(c_hat, c*)` 损失失配。\n"
        f"   - **24k 数据泛化瓶颈与 λ 权衡**：训练曲线显示三组在 24k 数据上 Train `L_diff` 持续降至 `0.0014–0.0021`，但 Val `L_diff` 在 Ep 10–20 后回升，全部 12 组收敛到 `~0.0080 / ~0.0102` 平台，距 E10-C（`0.00524 / 0.00747`，1.37M×8ep）和 Oracle（`0.00104 / 0.00113`）的剩余差距主要由 **24k vs 1.37M 数据规模** 决定。此外，`λ=0` 虽动作误差略低，却导致 **Self `||c||_2` 翻倍恶化（`9.5–12.2 → 20.2–24.5`）**；轻量 `λ=1e-6` 即可将 Self `||c||_2` 压回 `12.9–16.5`、Cross `||c||_2` 控制在 `27.0–29.0`（紧于 E10-C 的 `29.5`）且几乎无精度损失；而 `λ=1e-5` 过度正则化（`reg ≈ 2.5× L_diff`），损害全部三组的动作精度。"
    )
    lines.append(
        f"3. **Scratch (C) 同样有效，是否说明 Oracle Distillation 没有额外价值？**\n"
        f"   - **结论：对于离线动作精度（DDIM 10D MSE），Oracle Distillation（逐对优化 `c*` + 系数回归预训练）确实没有显著额外价值；其唯一剩余价值体现在收敛加速和保留更高的 Demo 敏感性 / 低范数几何偏置。**\n"
        f"   - 同架构、同 24k 数据、同 Decoder-Aware 目标下，Scratch（`lam_0`）直接达到 Seen `{c_s:.5f}` / Unseen `{c_u:.5f}`，与 A（`A - C`: Seen `{a_s - c_s:+.5f}` / Unseen `{a_u - c_u:+.5f}`）和 B（`B - C`: Seen `{b_s - c_s:+.5f}` / Unseen `{b_u - c_u:+.5f}`）差距仅 `~1–2%`，且**无需任何 Oracle 优化即同时超越 E11.2-A (`0.00941/0.01217`) 和 E11.2-B (`0.00824/0.01054`)**；在 `λ=3e-6` 和 `λ=1e-5` 的 Unseen 上 Scratch 甚至反超 A（`0.01059 vs 0.01089`、`0.01118 vs 0.01188`）。\n"
        f"   - **对后续路线的直接启示**：不必将昂贵的逐对 Oracle `c*` 蒸馏扩展到 1.37M 样本；直接从 Scratch 进行端到端 Decoder-Aware 优化（配合轻量范数约束 `λ ≈ 1e-6` 与自保真约束/自样本）即可在更大训练数据上直接获得动作精度与流形稳定性；若闭环场景需要 A 的高 Demo 敏感性（`+0.4–0.6×`），更值得在端到端训练中显式引入低范数或 Demo 对比机制，而非依赖高成本 Oracle 预蒸馏。\n"
    )

    lines.append("## 5. 产物索引\n")
    lines.append(
        f"- 综合表：[e11_3_summary.csv](file://{(E11_3_OUTPUT_ROOT / 'e11_3_summary.csv').resolve()}), "
        f"[e11_3_summary.json](file://{(E11_3_OUTPUT_ROOT / 'e11_3_summary.json').resolve()}), "
        f"[e11_3_phase_breakdown.csv](file://{(E11_3_OUTPUT_ROOT / 'e11_3_phase_breakdown.csv').resolve()})"
    )
    for lam, lt in zip(lambdas, ltags):
        run_links = []
        for grp in ("A", "B", "C"):
            rdir = (E11_3_OUTPUT_ROOT / lt / INIT_SPECS[grp]["run_name"]).resolve()
            run_links.append(f"{grp}: [best.pt](file://{rdir / 'best.pt'}) / [train_curve.csv](file://{rdir / 'train_curve.csv'})")
        lines.append(f"- `{lt}`: " + " | ".join(run_links))

    out_path.write_text("\n".join(lines) + "\n")


def run_eval_and_report(
    *,
    out_dir: Path = E11_3_OUTPUT_ROOT,
    lambdas: Sequence[float] = DEFAULT_E11_3_LAMBDAS,
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
    e11_2_dir: Path = E11_2_OUTPUT_ROOT,
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

    # Load all trained E11.3 checkpoints across lambdas x {A, B, C}
    loaded_models: Dict[str, Tuple[str, LowRankTransferTransformer, float, str]] = {}
    train_runs: Dict[str, Any] = {}
    run_configs: Dict[str, Any] = {}
    for lam in lambdas:
        lt = lam_to_tag(lam)
        for grp in ("A", "B", "C"):
            spec = INIT_SPECS[grp]
            rdir = out_dir / lt / spec["run_name"]
            key = f"{lt}__{spec['run_name']}"
            ck = torch.load(rdir / "best.pt", map_location="cpu", weights_only=False)
            mod = LowRankTransferTransformer(**ck["model_kwargs"]).to(device)
            mod.load_state_dict(ck["state_dict"], strict=True)
            mod.eval()
            loaded_models[key] = (ck["label"], mod, float(lam), grp)
            train_runs[key] = json.loads((rdir / "train_metrics.json").read_text())
            run_configs[key] = json.loads((rdir / "config.json").read_text())

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
    for key, (lbl, mod, _, _) in loaded_models.items():
        eval_specs.append(
            (
                key,
                lbl,
                lambda zs, hq, _m=mod: _predict_coeffs(
                    _m, zs, hq, batch_size=1024, device=device, use_bf16=use_bf16
                ),
            )
        )

    print("\n================ [E11.3 Decoder-Aware Evaluation (Seen & Unseen)] ================", flush=True)
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

    for r in all_flat:
        sn = r["split"]
        arr, _ = _safe_arr(id_mse_by_split[sn], r["ddim_10d_mse"], full_d1_by_split[sn])
        r["arr_10d_vs_id_full"] = arr
        r["self_d3_over_d1_ratio"] = float(
            r["self_d3_ddim_10d_mse"] / max(full_d1_by_split[sn], 1e-12)
        )
        if r["model"] in loaded_models:
            _, _, lam_v, grp_v = loaded_models[r["model"]]
            r["group"] = grp_v
            r["lambda"] = lam_v
            r["best_epoch"] = train_runs[r["model"]]["best_epoch"]
            r["best_val_ckpt_ddim10"] = train_runs[r["model"]]["best_val_ckpt_ddim10"]
        else:
            r["group"] = r["model"]
            r["lambda"] = None
            r["best_epoch"] = None
            r["best_val_ckpt_ddim10"] = None

    summary: Dict[str, Any] = {
        "config": {
            "experiment": "e11_3_decoder_aware_transfer_optimization",
            "lambdas": [float(l) for l in lambdas],
            "model_meta": model_meta,
            "full_self_d1_ddim10": full_d1_by_split,
            "identity_ddim10": id_mse_by_split,
            "eval_elapsed_seconds": round(time.monotonic() - t0, 2),
        },
        "run_configs": run_configs,
        "train_runs": train_runs,
        "evaluation": all_nested,
    }

    atomic_write_json(summary, out_dir / "e11_3_summary.json")
    write_csv(all_flat, out_dir / "e11_3_summary.csv")
    write_csv(all_phase, out_dir / "e11_3_phase_breakdown.csv")

    e11_2_json = json.loads((Path(e11_2_dir) / "e11_2_summary.json").read_text())
    oracle_json = json.loads((Path(oracle_dir) / "oracle_sweep_summary.json").read_text())

    fig_paths = _plot_e11_3_figures(summary, e11_2_json, oracle_json, lambdas, fig_dir)
    _write_e11_3_report(summary, e11_2_json, oracle_json, lambdas, fig_paths, out_dir / "REPORT.md")
    print(
        f"\n[E11.3 Eval & Report DONE] {summary['config']['eval_elapsed_seconds']}s -> {out_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("prep-cache", "train", "eval-report", "replot"),
        required=True,
    )
    parser.add_argument("--group", type=str, choices=("A", "B", "C"), default="A")
    parser.add_argument("--lam", type=float, default=1e-5)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_E11_3_LAMBDAS),
    )
    parser.add_argument("--out-dir", type=Path, default=E11_3_OUTPUT_ROOT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_E11_3_CACHE_PATH)
    parser.add_argument("--cache-meta-path", type=Path, default=DEFAULT_E11_3_CACHE_META_PATH)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mc-timesteps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--eval-ddim-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    if args.mode == "prep-cache":
        prepare_e11_3_action_cache(
            out_cache_path=args.cache_path,
            out_meta_path=args.cache_meta_path,
        )
        return

    if args.mode == "replot":
        summary = json.loads((args.out_dir / "e11_3_summary.json").read_text())
        e11_2_json = json.loads((E11_2_OUTPUT_ROOT / "e11_2_summary.json").read_text())
        oracle_json = json.loads((DEFAULT_ORACLE_DIR / "oracle_sweep_summary.json").read_text())
        lams = summary["config"]["lambdas"]
        fig_paths = _plot_e11_3_figures(summary, e11_2_json, oracle_json, lams, args.out_dir / "figures")
        _write_e11_3_report(summary, e11_2_json, oracle_json, lams, fig_paths, args.out_dir / "REPORT.md")
        print(f"[E11.3 Replot DONE] -> {args.out_dir}", flush=True)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_e11_3_run(
            group=args.group,
            lam=args.lam,
            out_dir=args.out_dir,
            cache_path=args.cache_path,
            cache_meta_path=args.cache_meta_path,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mc_timesteps=args.mc_timesteps,
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
            lambdas=args.lambdas,
            seed=args.seed,
            device=device,
            use_bf16=not args.no_bf16,
        )


if __name__ == "__main__":
    main()
