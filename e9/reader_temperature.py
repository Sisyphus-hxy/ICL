"""Experiment A: Training-Free Reader Temperature Sweep on E9 Stage 2 Direct.

Loads frozen `e9/outputs/stage2_direct/train/best.pt` (all weights frozen), adjusts ONLY
Reader Attention softmax temperature:
    w_i = softmax(s_i / tau),   where s_i = Q(h_q) K(h_i^D)^T / sqrt(d)
across tau in [0.25, 0.5, 0.75, 1.0, 1.5] (tau = 1.0 is the original E9 baseline).

Evaluates:
  1. Full 6,248-sample offline Cross Validation (`evaluate_e9_stage2_validation`, DDIM 10D MSE,
     latent L_z, attention entropy/max_w/eff_pos, multi-demo consistency).
  2. Full 531-demo D1-D4 offline decomposition (Default 100, Shifted 100, Oracle-GT 331)
     in a SINGLE shared pass per demo (CLIP encoding, Behavior Memory extraction, D1, D3, and
     Reader Q/K logits computed ONCE per demo; D2 and D4 swept across all temperatures).
  3. Verifies tau = 1.0 reproducibility against saved `e9/outputs/stage2_direct/` baselines.
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5 import ACTION_DIM_10D, ACTION_HORIZON, BPP_CKPT, EXEC_HORIZON
from e5.actions import action_10d_to_7d, action_7d_to_10d
from e5.policy_server import get_chain_tasks
from e5.train import atomic_write_json
from e5.vision import BPPVisualEncoder
from e7.diagnose_behavior_space import (
    MODES,
    _aggregate_rows,
    _collect_demo_specs,
    _compute_per_anchor_action_metrics,
    _macro_average_tasks,
    _stable_seed,
)
from e7.model import E7Policy, SimpleBehaviorReader, compute_attention_stats

from . import (
    BEHAVIOR_STRIDE,
    DEFAULT_STAGE1_LOCAL_CKPT,
    DEFAULT_STAGE2_CACHE_DIR,
    DEFAULT_STAGE2_DIRECT_DIR,
    E5_CACHE_DIR,
    OUTPUT_ROOT,
)
from .stage2_data import E9Stage2TransferBank
from .stage2_direct_model import load_e9_stage2_direct_policy
from .train_stage2_direct import evaluate_e9_stage2_validation

DEFAULT_TEMP_OUT_DIR = OUTPUT_ROOT / "reader_temperature"
DEFAULT_TAUS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5)


def tau_tag(tau: float) -> str:
    """Filesystem-friendly tag for a temperature value, e.g. 0.25 -> 'tau_0p25'."""
    s = f"{float(tau):.4g}".replace(".", "p")
    return f"tau_{s}"


class TemperatureScaledReader(nn.Module):
    """Non-invasive wrapper around `SimpleBehaviorReader` that applies exact temperature scaling:

        s_i = Q(h_q) K(h_i^D)^T / sqrt(d)
        w_i = softmax_i(s_i / tau)
        z_c = sum_i w_i * b_i

    Preserves all underlying weights untouched. At `temperature == 1.0`, delegates directly to
    exact `s_i / 1.0` (bit-for-bit identical to `SimpleBehaviorReader.forward`).
    """

    def __init__(self, base_reader: SimpleBehaviorReader, temperature: float = 1.0):
        super().__init__()
        if isinstance(base_reader, TemperatureScaledReader):
            base_reader = base_reader.base_reader
        if float(temperature) <= 0.0:
            raise ValueError(f"Reader temperature must be > 0, got {temperature}")
        self.base_reader = base_reader
        self.temperature = float(temperature)

    @property
    def num_query_tokens(self) -> int:
        return self.base_reader.num_query_tokens

    @property
    def embed_dim(self) -> int:
        return self.base_reader.embed_dim

    @property
    def in_dim(self) -> int:
        return self.base_reader.in_dim

    @property
    def attn_dim(self) -> int:
        return self.base_reader.attn_dim

    @property
    def scale(self) -> float:
        return self.base_reader.scale

    def encode_query(self, h_q: torch.Tensor) -> torch.Tensor:
        return self.base_reader.encode_query(h_q)

    def encode_keys(self, h_mem: torch.Tensor) -> torch.Tensor:
        return self.base_reader.encode_keys(h_mem)

    def compute_logits(
        self,
        h_q: torch.Tensor,
        h_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, M, _, _ = h_mem.shape
        q = self.base_reader.encode_query(h_q)
        k = self.base_reader.encode_keys(h_mem)
        logits = torch.einsum("bd,bmd->bm", q.float(), k.float()) * self.base_reader.scale
        if mem_pad_mask is not None:
            if mem_pad_mask.shape != (B, M):
                raise ValueError(
                    f"Expected mem_pad_mask ({B}, {M}), got {tuple(mem_pad_mask.shape)}"
                )
            mask_b = mem_pad_mask.to(device=logits.device, dtype=torch.bool)
            if mask_b.all(dim=-1).any():
                raise ValueError("mem_pad_mask has an episode with zero valid Behavior Memory entries")
            logits = logits.masked_fill(mask_b, float("-inf"))
        return logits

    def forward(
        self,
        h_q: torch.Tensor,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if b_mem.shape != h_mem.shape:
            raise ValueError(
                f"Expected b_mem shape {tuple(h_mem.shape)} matching h_mem, got {tuple(b_mem.shape)}"
            )
        logits = self.compute_logits(h_q, h_mem, mem_pad_mask=mem_pad_mask)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        weights = torch.softmax(logits, dim=-1)
        z_c = torch.einsum("bm,bmkd->bkd", weights.to(dtype=b_mem.dtype), b_mem)
        return z_c, weights


def set_policy_reader_temperature(policy: E7Policy, temperature: float) -> E7Policy:
    """Wrap or update `policy.reader` with exact `TemperatureScaledReader(tau)`."""
    tau = float(temperature)
    if tau <= 0.0:
        raise ValueError(f"temperature must be > 0, got {tau}")
    if isinstance(policy.reader, TemperatureScaledReader):
        policy.reader.temperature = tau
    else:
        policy.reader = TemperatureScaledReader(policy.reader, temperature=tau)
    return policy


@torch.inference_mode()
def run_val_temperature_sweep(
    *,
    policy: E7Policy,
    taus: Sequence[float],
    out_dir: Path,
    device: torch.device,
    clip_cache_dir: Path = E5_CACHE_DIR,
    stage2_cache_dir: Path = DEFAULT_STAGE2_CACHE_DIR,
    val_batch_size: int = 128,
    max_val_samples: Optional[int] = None,
    ddim_max_samples: Optional[int] = None,
    multi_demo_max_samples: Optional[int] = None,
    val_seed: int = 12345,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Run offline Cross Validation across all temperatures using frozen E9 Stage 2 policy."""
    t0 = time.monotonic()
    bank = E9Stage2TransferBank(
        clip_cache_dir=clip_cache_dir,
        stage2_cache_dir=stage2_cache_dir,
        device=device,
        resident_on_device=(device.type == "cuda"),
        z_self_on_device=False,  # val-only: keep train z_self on CPU to save VRAM
        seed=42,
    )

    val_root = out_dir / "val_cross"
    val_root.mkdir(parents=True, exist_ok=True)
    results_by_tau: Dict[str, Any] = {}

    for tau in taus:
        t_tau = time.monotonic()
        set_policy_reader_temperature(policy, tau)
        ev = evaluate_e9_stage2_validation(
            policy,
            bank,
            batch_size=val_batch_size,
            max_val_samples=max_val_samples,
            run_ddim=True,
            ddim_max_samples=ddim_max_samples,
            run_multi_demo=True,
            multi_demo_max_samples=multi_demo_max_samples,
            val_seed=val_seed,
            use_bf16=use_bf16,
        )
        ev["tau"] = float(tau)
        ev["elapsed_seconds"] = round(time.monotonic() - t_tau, 2)

        tag = tau_tag(tau)
        t_dir = val_root / tag
        t_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(ev, t_dir / "val_summary.json")
        results_by_tau[tag] = ev

        dm = ev["ddim"]
        lt = ev["latent"]
        at = ev["reader_attention"]["cross"]
        print(
            f"[ExpA Val tau={tau:<4g}] "
            f"DDIM10 nd={dm['cross_no_delta']['action10_mse']:.6f} -> "
            f"full={dm['cross_with_delta']['action10_mse']:.6f} | "
            f"MAE7={dm['cross_with_delta']['action7_mae']:.5f} | "
            f"Lz nd={lt['cross_no_delta_L_z']:.4f} full={lt['cross_with_delta_L_z']:.4f} | "
            f"attn H={at['entropy_mean']:.4f} H_n={at['normalized_entropy_mean']:.4f} "
            f"max_w={at['max_weight_mean']:.4f} eff={at['effective_positions_exp_entropy']:.2f} | "
            f"wrong_sens={dm['demo_sensitivity_action10_mse']:+.5f} ({ev['elapsed_seconds']}s)",
            flush=True,
        )

    del bank
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "taus": [float(t) for t in taus],
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "by_tau": results_by_tau,
    }


@torch.inference_mode()
def run_d1_d4_temperature_sweep(
    *,
    policy: E7Policy,
    payload: Dict[str, Any],
    e9_stage2_ckpt: Path,
    stage1_ckpt: Path,
    taus: Sequence[float],
    out_dir: Path,
    device: torch.device,
    use_bf16: bool = True,
    base_seed: int = 12345,
    behavior_stride: int = BEHAVIOR_STRIDE,
    demo_types: Tuple[str, ...] = ("default", "shifted", "oracle_gt"),
    max_demos_per_type: Optional[int] = None,
) -> Dict[str, Any]:
    """Single-pass D1-D4 offline decomposition across all temperatures in `taus`.

    Shares frozen CLIP encoding, Behavior Memory extraction, D1/D3 DDIM decoding, and Reader Q/K
    logits across all `tau` values so 5 temperatures take ~1.3x a single D1-D4 run instead of 5x.
    """
    t0 = time.monotonic()
    d1_d4_root = out_dir / "d1_d4"
    d1_d4_root.mkdir(parents=True, exist_ok=True)

    tasks = get_chain_tasks()
    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    base_reader = (
        policy.reader.base_reader
        if isinstance(policy.reader, TemperatureScaledReader)
        else policy.reader
    )

    s_z = float(policy.latent_scale_s_z.item())
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    specs = _collect_demo_specs(tasks, demo_types=demo_types)
    if max_demos_per_type is not None:
        filtered: List[Dict[str, Any]] = []
        counts: Dict[str, int] = defaultdict(int)
        for sp in specs:
            dt = sp["demo_type"]
            if counts[dt] < max_demos_per_type:
                filtered.append(sp)
                counts[dt] += 1
        specs = filtered

    tau_list = [float(t) for t in taus]
    rows_by_tau: Dict[float, List[Dict[str, Any]]] = {t: [] for t in tau_list}
    max_rot_roundtrip_err = 0.0

    print(
        f"[ExpA D1-D4 Sweep] {len(specs)} demos x {len(tau_list)} temps {tau_list} "
        f"(types={demo_types}, stride={behavior_stride}, s_z={s_z:.6f}, bf16={use_bf16}) on {device}",
        flush=True,
    )

    for idx, sp in enumerate(specs, start=1):
        demo_type = sp["demo_type"]
        task = sp["task"]
        task_idx = sp["task_idx"]
        demo_label = sp["demo_label"]

        with h5py.File(sp["h5_path"], "r") as f:
            conv = str(f["data"].attrs.get("macros_image_convention", "opengl"))
            d = f[sp["group_key"]]
            av_all = d["obs/agentview_rgb"][:]
            eh_all = d["obs/eye_in_hand_rgb"][:]
            actions_7d = d["actions"][:].astype(np.float32)

        T = len(av_all)
        if len(eh_all) != T or len(actions_7d) != T:
            raise ValueError(f"Length mismatch in {sp['h5_path']}:{sp['group_key']}")

        # 1. Shared CLIP encoding (once per demo)
        demo_all_feats = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if device.type == "cuda" else None,
        )

        # 2. Shared Behavior Memory, D1, D3, and Reader Q/K logits (once per demo)
        with amp_ctx:
            h_mem, b_mem, anchors = policy.teacher.extract_behavior_memory_from_demo_features(
                demo_all_feats, stride=behavior_stride
            )
            h_mem_f = h_mem.float().contiguous()
            b_mem_f = b_mem.float().contiguous()
            M = len(anchors)
            h_q = h_mem_f

            z_d1 = b_mem_f
            z_d3, _ = policy.adapt_latent(z_d1, h_q, delta_scale=1.0)

            h_mem_exp = h_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            b_mem_exp = b_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            q_vec = base_reader.encode_query(h_q)         # (M, d)
            k_vec = base_reader.encode_keys(h_mem_exp)    # (M, M, d)
            base_logits = torch.einsum("bd,bmd->bm", q_vec.float(), k_vec.float()) * base_reader.scale

            noises = []
            for t_k in anchors:
                s_k = _stable_seed(
                    f"{demo_type}/{task}/{demo_label}/t{int(t_k)}", base_seed=base_seed
                )
                gen = torch.Generator(device="cpu").manual_seed(s_k)
                noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
            init_noise = torch.stack(noises, dim=0).to(device=device, dtype=torch.float32)

            pred_d1 = policy.teacher.sample_from_z_self(z_d1, init_noise=init_noise)
            pred_d3 = policy.teacher.sample_from_z_self(z_d3, init_noise=init_noise)

        actions_10d_np = action_7d_to_10d(actions_7d)
        rt_7d_np = action_10d_to_7d(actions_10d_np)
        rt_err = float(np.max(np.abs(rt_7d_np - actions_7d)))
        if rt_err > max_rot_roundtrip_err:
            max_rot_roundtrip_err = rt_err

        step_grid = anchors[:, None] + np.arange(ACTION_HORIZON, dtype=np.int32)[None, :]
        valid_16_np = step_grid < T
        clipped_grid = np.minimum(step_grid, T - 1)
        gt_10d = torch.from_numpy(actions_10d_np[clipped_grid]).to(
            device=device, dtype=torch.float32
        )
        valid_16 = torch.from_numpy(valid_16_np).to(device=device, dtype=torch.bool)

        # Shared D1 & D3 metrics (tau-invariant)
        act_d1 = _compute_per_anchor_action_metrics(pred_d1, gt_10d, valid_16)
        act_d3 = _compute_per_anchor_action_metrics(pred_d3, gt_10d, valid_16)
        z_ref = z_d1.float()
        z_ref_flat = z_ref.reshape(M, -1)
        z_d3_f = z_d3.float()
        lat_d1 = {
            "mse": np.zeros(M, dtype=np.float32),
            "cos": np.ones(M, dtype=np.float32),
            "sq_mean": z_ref.square().mean(dim=(1, 2)).cpu().numpy(),
        }
        lat_d3 = {
            "mse": (z_d3_f - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
            "cos": F.cosine_similarity(z_d3_f.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
            "sq_mean": z_d3_f.square().mean(dim=(1, 2)).cpu().numpy(),
        }

        idx_grid = np.arange(M, dtype=np.float64)
        anchors_f = anchors.astype(np.float64)
        gt_chunks_7d = actions_7d[clipped_grid]
        gt_pos_l2_arr = [
            float(np.mean(np.linalg.norm(gt_chunks_7d[k, valid_16_np[k], :3], axis=-1)))
            for k in range(M)
        ]
        gt_grip_pos_arr = [
            float(np.mean(gt_chunks_7d[k, valid_16_np[k], 6] >= 0.0)) for k in range(M)
        ]

        # 3. Sweep D2 and D4 across all temperatures
        for tau in tau_list:
            with amp_ctx:
                logits_tau = base_logits if tau == 1.0 else (base_logits / tau)
                attn_w = torch.softmax(logits_tau, dim=-1)
                z_d2 = torch.einsum("bm,bmkd->bkd", attn_w.to(dtype=b_mem_exp.dtype), b_mem_exp)
                z_d4, _ = policy.adapt_latent(z_d2, h_q, delta_scale=1.0)
                pred_d2 = policy.teacher.sample_from_z_self(z_d2, init_noise=init_noise)
                pred_d4 = policy.teacher.sample_from_z_self(z_d4, init_noise=init_noise)

            act_metrics = {
                "D1_teacher": act_d1,
                "D2_retrieval": _compute_per_anchor_action_metrics(pred_d2, gt_10d, valid_16),
                "D3_transfer": act_d3,
                "D4_full": _compute_per_anchor_action_metrics(pred_d4, gt_10d, valid_16),
            }
            z_d2_f = z_d2.float()
            z_d4_f = z_d4.float()
            lat_metrics = {
                "D1_teacher": lat_d1,
                "D2_retrieval": {
                    "mse": (z_d2_f - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
                    "cos": F.cosine_similarity(z_d2_f.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
                    "sq_mean": z_d2_f.square().mean(dim=(1, 2)).cpu().numpy(),
                },
                "D3_transfer": lat_d3,
                "D4_full": {
                    "mse": (z_d4_f - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
                    "cos": F.cosine_similarity(z_d4_f.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
                    "sq_mean": z_d4_f.square().mean(dim=(1, 2)).cpu().numpy(),
                },
            }
            lat_mse_d4_d3 = (z_d4_f - z_d3_f).square().mean(dim=(1, 2)).cpu().numpy()
            lat_cos_d4_d3 = (
                F.cosine_similarity(z_d4_f.reshape(M, -1), z_d3_f.reshape(M, -1), dim=-1)
                .cpu()
                .numpy()
            )

            w_np = attn_w.float().cpu().numpy()
            ast = compute_attention_stats(attn_w, mem_pad_mask=None)
            ent_np = ast["entropy"].cpu().numpy()
            nent_np = ast["norm_entropy"].cpu().numpy()
            eff_np = ast["eff_pos_exp"].cpu().numpy()
            wmax_np = ast["max_weight"].cpu().numpy()

            tau_rows = rows_by_tau[tau]
            for k in range(M):
                wk = w_np[k]
                argmax_j = int(np.argmax(wk))
                signed_slots = argmax_j - k
                err_slots = abs(signed_slots)
                err_steps = abs(int(anchors[argmax_j]) - int(anchors[k]))
                err_norm = float(err_slots) / float(max(M - 1, 1))
                exp_j = float(np.sum(wk * idx_grid))
                exp_step = float(np.sum(wk * anchors_f))
                exp_err_slots = abs(exp_j - float(k))
                exp_err_steps = abs(exp_step - float(anchors[k]))
                exp_abs_slots = float(np.sum(wk * np.abs(idx_grid - float(k))))
                exp_abs_steps = float(np.sum(wk * np.abs(anchors_f - float(anchors[k]))))

                row: Dict[str, Any] = {
                    "demo_type": demo_type,
                    "task_idx": task_idx,
                    "task": task,
                    "demo_label": demo_label,
                    "demo_len": T,
                    "mem_len": M,
                    "anchor_idx": k,
                    "anchor_step": int(anchors[k]),
                    "is_tail_window": bool(not valid_16_np[k].all()),
                    "n_valid_16": int(act_d1["n_valid_16"][k]),
                    "n_valid_12": int(act_d1["n_valid_12"][k]),
                    "reader_max_weight": float(wmax_np[k]),
                    "reader_self_weight": float(wk[k]),
                    "reader_argmax_idx": argmax_j,
                    "reader_exact_hit": int(argmax_j == k),
                    "reader_within1_hit": int(err_slots <= 1),
                    "reader_pos_err_argmax_slots": int(err_slots),
                    "reader_pos_err_argmax_steps": int(err_steps),
                    "reader_pos_err_argmax_norm": err_norm,
                    "reader_signed_err_argmax_slots": int(signed_slots),
                    "reader_pos_err_expected_slots": exp_err_slots,
                    "reader_pos_err_expected_steps": exp_err_steps,
                    "reader_expected_abs_dist_slots": exp_abs_slots,
                    "reader_expected_abs_dist_steps": exp_abs_steps,
                    "reader_entropy": float(ent_np[k]),
                    "reader_norm_entropy": float(nent_np[k]),
                    "reader_eff_pos_exp": float(eff_np[k]),
                    "latent_mse_d4_vs_d3": float(lat_mse_d4_d3[k]),
                    "latent_cos_d4_vs_d3": float(lat_cos_d4_d3[k]),
                    "gt_pos_l2": gt_pos_l2_arr[k],
                    "gt_grip_pos_frac": gt_grip_pos_arr[k],
                }
                for m in MODES:
                    am = act_metrics[m]
                    lm = lat_metrics[m]
                    row[f"{m}_mse10_masked"] = float(am["mse10_masked"][k])
                    row[f"{m}_mae7_masked"] = float(am["mae7_masked"][k])
                    row[f"{m}_exec12_mae7_masked"] = float(am["exec12_mae7_masked"][k])
                    row[f"{m}_grip_acc16_masked"] = float(am["grip_acc16_masked"][k])
                    row[f"{m}_grip_acc12_masked"] = float(am["grip_acc12_masked"][k])
                    row[f"{m}_mse10_valid_sum"] = float(am["mse10_valid_sum"][k])
                    row[f"{m}_pos_mse_valid_sum"] = float(am["pos_mse_valid_sum"][k])
                    row[f"{m}_rot6d_mse_valid_sum"] = float(am["rot6d_mse_valid_sum"][k])
                    row[f"{m}_grip_mse_valid_sum"] = float(am["grip_mse_valid_sum"][k])
                    row[f"{m}_mae7_valid_sum"] = float(am["mae7_valid_sum"][k])
                    row[f"{m}_exec12_mae7_valid_sum"] = float(am["exec12_mae7_valid_sum"][k])
                    row[f"{m}_grip_acc16_valid_sum"] = float(am["grip_acc16_valid_sum"][k])
                    row[f"{m}_grip_acc12_valid_sum"] = float(am["grip_acc12_valid_sum"][k])
                    row[f"{m}_mse10_unmasked"] = float(am["mse10_unmasked"][k])
                    row[f"{m}_latent_mse_vs_self"] = float(lm["mse"][k])
                    row[f"{m}_latent_cos_vs_self"] = float(lm["cos"][k])
                    row[f"{m}_latent_sq_mean"] = float(lm["sq_mean"][k])
                tau_rows.append(row)

        if idx % 50 == 0 or idx == len(specs):
            dt = time.monotonic() - t0
            print(
                f"[ExpA D1-D4 Sweep] {idx}/{len(specs)} demos "
                f"({len(rows_by_tau[tau_list[0]]):,} anchors x {len(tau_list)} temps) "
                f"in {dt:.1f}s ({idx / max(dt, 1e-6):.2f} demos/s)",
                flush=True,
            )

    # Aggregate and save per-temperature summaries
    summaries_by_tau: Dict[str, Any] = {}
    elapsed_total = round(time.monotonic() - t0, 2)
    for tau in tau_list:
        tag = tau_tag(tau)
        t_dir = d1_d4_root / tag
        t_dir.mkdir(parents=True, exist_ok=True)
        all_rows = rows_by_tau[tau]

        by_demo_type: Dict[str, Any] = {}
        per_task_by_type: Dict[str, Dict[str, Any]] = {}
        for dt in demo_types:
            dt_rows = [r for r in all_rows if r["demo_type"] == dt]
            by_demo_type[dt] = _aggregate_rows(dt_rows, s_z=s_z)
            per_task_by_type[dt] = {}
            for task in tasks:
                t_rows = [r for r in dt_rows if r["task"] == task]
                if t_rows:
                    per_task_by_type[dt][task] = _aggregate_rows(t_rows, s_z=s_z)

        shared_9_tasks = [t for t in tasks if t in per_task_by_type.get("oracle_gt", {})]
        ogt_demo_weights = {
            t: float(per_task_by_type["oracle_gt"][t]["num_demos"]) for t in shared_9_tasks
        }
        ogt_anchor_weights = {
            t: float(per_task_by_type["oracle_gt"][t]["num_anchors"]) for t in shared_9_tasks
        }

        macro_all_10: Dict[str, Any] = {}
        macro_shared_9: Dict[str, Any] = {}
        oracle_demo_weighted_9: Dict[str, Any] = {}
        oracle_anchor_weighted_9: Dict[str, Any] = {}
        for dt in demo_types:
            macro_all_10[dt] = _macro_average_tasks(per_task_by_type[dt], tasks)
            if shared_9_tasks:
                macro_shared_9[dt] = _macro_average_tasks(per_task_by_type[dt], shared_9_tasks)
                oracle_demo_weighted_9[dt] = _macro_average_tasks(
                    per_task_by_type[dt], shared_9_tasks, weights=ogt_demo_weights
                )
                oracle_anchor_weighted_9[dt] = _macro_average_tasks(
                    per_task_by_type[dt], shared_9_tasks, weights=ogt_anchor_weights
                )

        if all_rows:
            with (t_dir / "per_anchor_records.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)

        summary_rows: List[Dict[str, Any]] = []
        for agg_kind, container in (
            ("micro_all", by_demo_type),
            ("macro_all_10", macro_all_10),
            ("macro_shared_9", macro_shared_9),
            ("oracle_demo_weighted_9", oracle_demo_weighted_9),
        ):
            for dt in demo_types:
                blk = container.get(dt)
                if not blk or "modes" not in blk:
                    continue
                rd = blk["reader"]
                for m in MODES:
                    ms = blk["modes"][m]
                    dl = blk["deltas_vs_D1"].get(m, {})
                    summary_rows.append(
                        {
                            "tau": tau,
                            "aggregation": agg_kind,
                            "demo_type": dt,
                            "mode": m,
                            "ddim_10d_mse": ms["ddim_10d_mse"],
                            "ddim_7d_mae": ms["ddim_7d_mae"],
                            "exec12_7d_mae": ms["exec12_7d_mae"],
                            "grip_acc": ms["grip_acc"],
                            "delta_vs_D1_10d_mse_abs": dl.get("ddim_10d_mse_abs", 0.0),
                            "delta_vs_D1_10d_mse_rel_pct": dl.get("ddim_10d_mse_rel_pct", 0.0),
                            "latent_mse_vs_self": ms["latent_mse_vs_self"],
                            "latent_lz_vs_self": ms["latent_lz_vs_self"],
                            "latent_cos_vs_self": ms["latent_cos_vs_self"],
                            "reader_max_weight": rd["max_weight_mean"],
                            "reader_self_weight": rd["self_weight_mean"],
                            "reader_exact_hit_rate": rd["exact_hit_rate"],
                            "reader_entropy": rd.get("attn_entropy_mean"),
                            "reader_norm_entropy": rd.get("attn_norm_entropy_mean"),
                            "reader_eff_pos_exp": rd.get("attn_eff_pos_exp_mean"),
                            "reader_pos_err_argmax_slots": rd["pos_err_argmax_slots_mean"],
                        }
                    )
        if summary_rows:
            with (t_dir / "summary_by_demo_type.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        summary: Dict[str, Any] = {
            "experiment": "e9_reader_temperature_sweep_d1_d4",
            "tau": tau,
            "checkpoints": {
                "clip_ckpt": str(BPP_CKPT),
                "stage1_local_ckpt": str(Path(stage1_ckpt).resolve()),
                "e9_stage2_ckpt": str(Path(e9_stage2_ckpt).resolve()),
                "e9_stage2_epoch": payload.get("epoch"),
                "prediction_mode": policy.prediction_mode,
            },
            "protocol": {
                "behavior_stride": int(behavior_stride),
                "action_horizon": int(ACTION_HORIZON),
                "exec_horizon": int(EXEC_HORIZON),
                "base_seed": int(base_seed),
                "use_bf16": bool(use_bf16 and device.type == "cuda"),
                "s_z": s_z,
                "max_rot_roundtrip_err": max_rot_roundtrip_err,
                "all_modules_frozen": True,
            },
            "elapsed_seconds_all_taus": elapsed_total,
            "shared_9_tasks": shared_9_tasks,
            "by_demo_type_micro": by_demo_type,
            "by_demo_type_macro_all_10": macro_all_10,
            "by_demo_type_macro_shared_9": macro_shared_9,
            "by_demo_type_oracle_demo_weighted_9": oracle_demo_weighted_9,
            "by_demo_type_oracle_anchor_weighted_9": oracle_anchor_weighted_9,
            "per_task_by_demo_type": per_task_by_type,
        }
        atomic_write_json(summary, t_dir / "offline_summary.json")
        summaries_by_tau[tag] = summary

    del visual
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "taus": tau_list,
        "elapsed_seconds": elapsed_total,
        "by_tau": summaries_by_tau,
    }


def _plot_temperature_sweep(
    taus: Sequence[float],
    val_res: Optional[Dict[str, Any]],
    d1_d4_res: Optional[Dict[str, Any]],
    out_dir: Path,
) -> None:
    tau_arr = [float(t) for t in taus]
    tags = [tau_tag(t) for t in tau_arr]

    if d1_d4_res is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=160)
        dts = ["default", "shifted", "oracle_gt"]
        dt_titles = ["Default (9 shared)", "Shifted (9 shared)", "Oracle-GT (9 shared)"]
        for col, (dt, title) in enumerate(zip(dts, dt_titles)):
            ax_act = axes[0, col]
            ax_rd = axes[1, col]
            d1 = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["modes"]["D1_teacher"]["ddim_10d_mse"] for tg in tags]
            d2 = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["modes"]["D2_retrieval"]["ddim_10d_mse"] for tg in tags]
            d3 = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["modes"]["D3_transfer"]["ddim_10d_mse"] for tg in tags]
            d4 = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["modes"]["D4_full"]["ddim_10d_mse"] for tg in tags]

            ax_act.plot(tau_arr, d2, "o-", color="#e67e22", lw=2.0, label="D2 Reader Only")
            ax_act.plot(tau_arr, d4, "s-", color="#1f77b4", lw=2.2, label="D4 Full")
            ax_act.axhline(d3[0], color="#8e44ad", ls="--", lw=1.4, label=f"D3 Transfer ({d3[0]:.4f})")
            ax_act.axhline(d1[0], color="#2c3e50", ls=":", lw=1.4, label=f"D1 Teacher ({d1[0]:.4f})")
            ax_act.axvline(1.0, color="#95a5a6", ls=":", lw=1.0)
            ax_act.set_title(f"{title}: Action 10D MSE", fontweight="bold")
            ax_act.set_xlabel("Temperature τ")
            ax_act.set_ylabel("DDIM 10D MSE (↓)")
            ax_act.set_xticks(tau_arr)
            ax_act.grid(alpha=0.3, linestyle="--")
            if col == 0:
                ax_act.legend(fontsize=8)

            w_self = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["reader"]["self_weight_mean"] for tg in tags]
            ent_n = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["reader"]["attn_norm_entropy_mean"] for tg in tags]
            d2_lz = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["modes"]["D2_retrieval"]["latent_lz_vs_self"] for tg in tags]
            d4_lz = [d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"][dt]["modes"]["D4_full"]["latent_lz_vs_self"] for tg in tags]

            ax_rd.plot(tau_arr, w_self, "o-", color="#27ae60", lw=2.0, label="w_self")
            ax_rd.plot(tau_arr, ent_n, "^--", color="#c0392b", lw=1.8, label="Norm Entropy H_n")
            ax_rd.plot(tau_arr, d2_lz, "d-", color="#e67e22", lw=1.8, label="D2 Latent L_z")
            ax_rd.plot(tau_arr, d4_lz, "s-", color="#1f77b4", lw=1.8, label="D4 Latent L_z")
            ax_rd.axvline(1.0, color="#95a5a6", ls=":", lw=1.0)
            ax_rd.set_title(f"{title}: Reader Sharpness & Latent L_z", fontweight="bold")
            ax_rd.set_xlabel("Temperature τ")
            ax_rd.set_xticks(tau_arr)
            ax_rd.grid(alpha=0.3, linestyle="--")
            if col == 0:
                ax_rd.legend(fontsize=8)

        fig.suptitle("Experiment A: E9 Stage 2 Direct Reader Temperature Sweep (D1–D4)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "temperature_sweep_d1_d4.png", bbox_inches="tight")
        plt.close(fig)

    if val_res is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=160)
        nd_mse = [val_res["by_tau"][tg]["ddim"]["cross_no_delta"]["action10_mse"] for tg in tags]
        fd_mse = [val_res["by_tau"][tg]["ddim"]["cross_with_delta"]["action10_mse"] for tg in tags]
        nd_lz = [val_res["by_tau"][tg]["latent"]["cross_no_delta_L_z"] for tg in tags]
        fd_lz = [val_res["by_tau"][tg]["latent"]["cross_with_delta_L_z"] for tg in tags]
        max_w = [val_res["by_tau"][tg]["reader_attention"]["cross"]["max_weight_mean"] for tg in tags]
        ent_n = [val_res["by_tau"][tg]["reader_attention"]["cross"]["normalized_entropy_mean"] for tg in tags]
        eff_p = [val_res["by_tau"][tg]["reader_attention"]["cross"]["effective_positions_exp_entropy"] for tg in tags]

        axes[0].plot(tau_arr, nd_mse, "o-", color="#e67e22", lw=2.0, label="Cross Reader Only")
        axes[0].plot(tau_arr, fd_mse, "s-", color="#1f77b4", lw=2.2, label="Cross Full (With Transfer)")
        axes[0].axvline(1.0, color="#95a5a6", ls=":")
        axes[0].set_title("Val Cross DDIM 10D MSE (n=6,248)", fontweight="bold")
        axes[0].set_xlabel("Temperature τ")
        axes[0].set_ylabel("DDIM 10D MSE (↓)")
        axes[0].set_xticks(tau_arr)
        axes[0].grid(alpha=0.3, ls="--")
        axes[0].legend(fontsize=8.5)

        axes[1].plot(tau_arr, nd_lz, "o-", color="#e67e22", lw=2.0, label="Cross Reader Only L_z")
        axes[1].plot(tau_arr, fd_lz, "s-", color="#1f77b4", lw=2.2, label="Cross Full L_z")
        axes[1].axvline(1.0, color="#95a5a6", ls=":")
        axes[1].set_title("Val Cross Latent L_z", fontweight="bold")
        axes[1].set_xlabel("Temperature τ")
        axes[1].set_ylabel("Normalized Latent MSE L_z (↓)")
        axes[1].set_xticks(tau_arr)
        axes[1].grid(alpha=0.3, ls="--")
        axes[1].legend(fontsize=8.5)

        axes[2].plot(tau_arr, max_w, "o-", color="#27ae60", lw=2.0, label="max_w")
        axes[2].plot(tau_arr, ent_n, "^--", color="#c0392b", lw=2.0, label="Norm Entropy H_n")
        axes[2].axvline(1.0, color="#95a5a6", ls=":")
        ax2_r = axes[2].twinx()
        ax2_r.plot(tau_arr, eff_p, "d:", color="#8e44ad", lw=1.8, label="eff_pos (right)")
        ax2_r.set_ylabel("Effective Positions exp(H)", color="#8e44ad")
        axes[2].set_title("Val Cross Reader Attention Sharpness", fontweight="bold")
        axes[2].set_xlabel("Temperature τ")
        axes[2].set_xticks(tau_arr)
        axes[2].grid(alpha=0.3, ls="--")
        axes[2].legend(loc="center left", fontsize=8)
        ax2_r.legend(loc="center right", fontsize=8)

        fig.suptitle("Experiment A: Offline Cross Validation Temperature Sweep", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "temperature_sweep_val.png", bbox_inches="tight")
        plt.close(fig)


def run_reader_temperature_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    taus = [float(t) for t in args.taus]
    val_res = None
    d1_d4_res = None

    if args.aggregate_only:
        payload = {"epoch": None}
        val_by_tau = {}
        d1_d4_by_tau = {}
        for tau in taus:
            tg = tau_tag(tau)
            vp = out_dir / "val_cross" / tg / "val_summary.json"
            if vp.is_file():
                val_by_tau[tg] = json.loads(vp.read_text())
            dp = out_dir / "d1_d4" / tg / "offline_summary.json"
            if dp.is_file():
                d1_d4_by_tau[tg] = json.loads(dp.read_text())
                if payload["epoch"] is None:
                    payload["epoch"] = d1_d4_by_tau[tg].get("checkpoints", {}).get("e9_stage2_epoch")
        if len(val_by_tau) == len(taus):
            val_res = {"taus": taus, "by_tau": val_by_tau}
        if len(d1_d4_by_tau) == len(taus):
            d1_d4_res = {"taus": taus, "by_tau": d1_d4_by_tau}
    else:
        policy, payload = load_e9_stage2_direct_policy(
            args.ckpt, stage1_ckpt=args.stage1_ckpt, device=device
        )
        for p in policy.parameters():
            p.requires_grad_(False)
        policy.eval()

        trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        if trainable_cnt != 0:
            raise RuntimeError(f"Expected all weights frozen, found {trainable_cnt} trainable params")

        if not args.skip_val:
            val_res = run_val_temperature_sweep(
                policy=policy,
                taus=taus,
                out_dir=out_dir,
                device=device,
                clip_cache_dir=args.clip_cache_dir,
                stage2_cache_dir=args.stage2_cache_dir,
                val_batch_size=args.val_batch_size,
                max_val_samples=args.max_val_samples,
                ddim_max_samples=args.ddim_val_samples,
                multi_demo_max_samples=args.multi_demo_samples,
                val_seed=args.val_seed,
                use_bf16=not args.no_bf16,
            )

        if not args.skip_d1_d4:
            d1_d4_res = run_d1_d4_temperature_sweep(
                policy=policy,
                payload=payload,
                e9_stage2_ckpt=args.ckpt,
                stage1_ckpt=args.stage1_ckpt,
                taus=taus,
                out_dir=out_dir,
                device=device,
                use_bf16=not args.no_bf16,
                base_seed=args.val_seed,
                behavior_stride=args.stride,
                demo_types=tuple(args.demo_types),
                max_demos_per_type=args.max_demos_per_type,
            )

    # Build compact comparison table & sanity check tau=1.0 against original E9 Stage 2
    compact_rows: List[Dict[str, Any]] = []
    for tau in taus:
        tg = tau_tag(tau)
        row: Dict[str, Any] = {"tau": tau}
        if val_res is not None:
            v = val_res["by_tau"][tg]
            row.update(
                {
                    "val_cross_nd_ddim10_mse": v["ddim"]["cross_no_delta"]["action10_mse"],
                    "val_cross_full_ddim10_mse": v["ddim"]["cross_with_delta"]["action10_mse"],
                    "val_cross_full_ddim7_mae": v["ddim"]["cross_with_delta"]["action7_mae"],
                    "val_cross_nd_L_z": v["latent"]["cross_no_delta_L_z"],
                    "val_cross_full_L_z": v["latent"]["cross_with_delta_L_z"],
                    "val_cross_attn_entropy": v["reader_attention"]["cross"]["entropy_mean"],
                    "val_cross_attn_norm_entropy": v["reader_attention"]["cross"]["normalized_entropy_mean"],
                    "val_cross_attn_max_w": v["reader_attention"]["cross"]["max_weight_mean"],
                    "val_cross_attn_eff_pos": v["reader_attention"]["cross"]["effective_positions_exp_entropy"],
                    "val_wrong_demo_sens_10d": v["ddim"]["demo_sensitivity_action10_mse"],
                }
            )
        if d1_d4_res is not None:
            ds = d1_d4_res["by_tau"][tg]["by_demo_type_macro_shared_9"]
            dm_micro = d1_d4_res["by_tau"][tg]["by_demo_type_micro"]
            for dt in ("default", "shifted", "oracle_gt"):
                if dt not in ds:
                    continue
                rd = ds[dt]["reader"]
                m = ds[dt]["modes"]
                st9 = dm_micro[dt]["d3_stratification"].get("w_self_ge_0.9", {})
                row.update(
                    {
                        f"{dt}_exact_hit": rd["exact_hit_rate"],
                        f"{dt}_w_self": rd["self_weight_mean"],
                        f"{dt}_max_w": rd["max_weight_mean"],
                        f"{dt}_entropy": rd["attn_entropy_mean"],
                        f"{dt}_norm_entropy": rd["attn_norm_entropy_mean"],
                        f"{dt}_eff_pos": rd["attn_eff_pos_exp_mean"],
                        f"{dt}_D1_mse10": m["D1_teacher"]["ddim_10d_mse"],
                        f"{dt}_D2_mse10": m["D2_retrieval"]["ddim_10d_mse"],
                        f"{dt}_D3_mse10": m["D3_transfer"]["ddim_10d_mse"],
                        f"{dt}_D4_mse10": m["D4_full"]["ddim_10d_mse"],
                        f"{dt}_D2_lat_mse": m["D2_retrieval"]["latent_mse_vs_self"],
                        f"{dt}_D2_lat_lz": m["D2_retrieval"]["latent_lz_vs_self"],
                        f"{dt}_D3_lat_mse": m["D3_transfer"]["latent_mse_vs_self"],
                        f"{dt}_D4_lat_mse": m["D4_full"]["latent_mse_vs_self"],
                        f"{dt}_D4_lat_lz": m["D4_full"]["latent_lz_vs_self"],
                        f"{dt}_w_self_ge_0p9_frac": st9.get("frac", 0.0),
                    }
                )
        compact_rows.append(row)

    if compact_rows:
        with (out_dir / "sweep_table.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(compact_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compact_rows)

    baseline_check: Dict[str, Any] = {}
    tg1 = tau_tag(1.0)
    if val_res is not None and tg1 in val_res["by_tau"]:
        ref_val_path = DEFAULT_STAGE2_DIRECT_DIR / "train/final_eval.json"
        if ref_val_path.is_file():
            ref_v = json.loads(ref_val_path.read_text())["final_eval"]
            cur_v = val_res["by_tau"][tg1]
            baseline_check["val_cross_full_ddim10_diff"] = abs(
                cur_v["ddim"]["cross_with_delta"]["action10_mse"]
                - ref_v["ddim"]["cross_with_delta"]["action10_mse"]
            )
            baseline_check["val_cross_nd_ddim10_diff"] = abs(
                cur_v["ddim"]["cross_no_delta"]["action10_mse"]
                - ref_v["ddim"]["cross_no_delta"]["action10_mse"]
            )
    if d1_d4_res is not None and tg1 in d1_d4_res["by_tau"]:
        ref_d_path = DEFAULT_STAGE2_DIRECT_DIR / "d1_d4_diagnosis/offline_summary.json"
        if ref_d_path.is_file():
            ref_d = json.loads(ref_d_path.read_text())["by_demo_type_macro_shared_9"]
            cur_d = d1_d4_res["by_tau"][tg1]["by_demo_type_macro_shared_9"]
            baseline_check["d1_d4_oracle_D4_diff"] = abs(
                cur_d["oracle_gt"]["modes"]["D4_full"]["ddim_10d_mse"]
                - ref_d["oracle_gt"]["modes"]["D4_full"]["ddim_10d_mse"]
            )
            baseline_check["d1_d4_default_D2_diff"] = abs(
                cur_d["default"]["modes"]["D2_retrieval"]["ddim_10d_mse"]
                - ref_d["default"]["modes"]["D2_retrieval"]["ddim_10d_mse"]
            )

    _plot_temperature_sweep(taus, val_res, d1_d4_res, out_dir)

    sweep_summary = {
        "experiment": "e9_exp_a_reader_temperature_sweep",
        "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_epoch": payload.get("epoch"),
        "stage1_ckpt": str(Path(args.stage1_ckpt).resolve()),
        "taus": taus,
        "all_weights_frozen": True,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "tau_1p0_baseline_check": baseline_check,
        "compact_table": compact_rows,
    }
    atomic_write_json(sweep_summary, out_dir / "sweep_summary.json")

    print("\n=== Experiment A: Reader Temperature Sweep Summary ===", flush=True)
    for r in compact_rows:
        parts = [f"tau={r['tau']:<4g}"]
        if "val_cross_full_ddim10_mse" in r:
            parts.append(
                f"ValND={r['val_cross_nd_ddim10_mse']:.5f} ValFull={r['val_cross_full_ddim10_mse']:.5f}"
            )
        if "default_D2_mse10" in r:
            parts.append(
                f"Def(hit={100*r['default_exact_hit']:.1f}% wS={r['default_w_self']:.3f} "
                f"H={r['default_entropy']:.3f} D2={r['default_D2_mse10']:.5f} D4={r['default_D4_mse10']:.5f})"
            )
        if "oracle_gt_D2_mse10" in r:
            parts.append(
                f"OGT(wS={r['oracle_gt_w_self']:.3f} H={r['oracle_gt_entropy']:.3f} "
                f"D2={r['oracle_gt_D2_mse10']:.5f} D4={r['oracle_gt_D4_mse10']:.5f})"
            )
        print(" | ".join(parts), flush=True)
    if baseline_check:
        print("tau=1.0 baseline check:", json.dumps(baseline_check), flush=True)
    print(f"[ExpA DONE] Saved to {out_dir} in {sweep_summary['elapsed_seconds']}s", flush=True)
    return sweep_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, default=DEFAULT_STAGE2_DIRECT_DIR / "train/best.pt"
    )
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument("--clip-cache-dir", type=Path, default=E5_CACHE_DIR)
    parser.add_argument("--stage2-cache-dir", type=Path, default=DEFAULT_STAGE2_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEMP_OUT_DIR)
    parser.add_argument(
        "--taus", type=float, nargs="+", default=list(DEFAULT_TAUS)
    )
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--ddim-val-samples", type=int, default=None)
    parser.add_argument("--multi-demo-samples", type=int, default=None)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--stride", type=int, default=BEHAVIOR_STRIDE)
    parser.add_argument(
        "--demo-types",
        nargs="+",
        default=["default", "shifted", "oracle_gt"],
        choices=("default", "shifted", "oracle_gt"),
    )
    parser.add_argument("--max-demos-per-type", type=int, default=None)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-d1-d4", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_reader_temperature_experiment(args)


if __name__ == "__main__":
    main()
