"""Comprehensive Offline Evaluation for P1 (E10-A / E10-B / E10-C / Control + E9 Direct baseline).

Evaluates:
  1. Unseen-Chain (3,000 pairs from P0/P0.5 with identical DDIM noise):
     - Correct Demo DDIM 10D Action MSE (overall, Pos, Rot6D, Grip, 7D MAE, Grip Acc, Latent L_z)
     - Wrong-Task Demo DDIM 10D Action MSE & Correct/Wrong sensitivity (ΔMSE, ratio)
     - Comparison with P0 Oracle rank-K upper bound and K=0 frozen Reader!
  2. Full D1-D4 Offline Decomposition & Self-Latent Preservation (`w_self >= 0.9` zero-shift subset)
     on Default (100 demos), Shifted (100 demos), and Oracle-GT (331 demos), matching
     `e9/diagnose_stage2_d1_d4.py` bit-for-bit!
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
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
from e7.model import compute_attention_stats
from e9.stage2_direct_model import load_e9_stage2_direct_policy
from p0_lowrank.alignment import sample_unseen_chain_pairs
from p0_lowrank.cache_unseen import DEFAULT_UNSEEN_CACHE_DIR
from p0_lowrank.p0_5_demo_preservation import _extract_wrong_latents, _sample_wrong_task_episodes
from p0_lowrank.run_diagnosis import (
    _build_init_noises,
    _compute_alignment_and_gt_for_split,
    _decode_and_score_latents,
    _extract_model_latents_for_split,
    _safe_arr,
    _write_csv,
)

from . import (
    BEHAVIOR_STRIDE,
    E9_STAGE1_CKPT,
    E9_STAGE2_CKPT,
    EMBED_DIM,
    LATENT_DIM,
    LATENT_TOKENS,
    OUTPUT_ROOT,
    VARIANTS,
)
from .model import load_e10_policy


@torch.inference_mode()
def evaluate_unseen_chain_all(
    models: Dict[str, Tuple[torch.nn.Module, Dict[str, Any]]],
    *,
    unseen_cache_dir: Path = DEFAULT_UNSEEN_CACHE_DIR,
    n_unseen: int = 3000,
    seed: int = 42,
    ddim_seed: int = 12345,
    batch_size: int = 1024,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Evaluate all models on the exact P0/P0.5 Unseen-Chain 3,000 pairs (Correct & Wrong-task)."""
    unseen_meta = json.loads((unseen_cache_dir / "unseen_meta.json").read_text())
    unseen_demos = unseen_meta["demos"]
    u_mem_h = np.load(unseen_cache_dir / "mem_h.npy", mmap_mode="r")
    u_mem_b = np.load(unseen_cache_dir / "mem_b_e9.npy", mmap_mode="r")
    u_mem_anchors = np.load(unseen_cache_dir / "mem_anchor_steps.npy")
    u_mem_offsets = np.load(unseen_cache_dir / "mem_offsets.npy")
    u_mem_lengths = np.load(unseen_cache_dir / "mem_lengths.npy")
    u_actions_10d = np.load(unseen_cache_dir / "actions_10d.npy", mmap_mode="r")
    u_ee_pos = np.load(unseen_cache_dir / "ee_pos.npy", mmap_mode="r")
    u_action_offsets = np.load(unseen_cache_dir / "action_offsets.npy")

    pairs = sample_unseen_chain_pairs(unseen_demos, target_pairs=n_unseen, seed=seed + 2002)
    sb = _compute_alignment_and_gt_for_split(
        "unseen_chain",
        pairs,
        actions_10d=u_actions_10d,
        ee_pos_all=u_ee_pos,
        action_offsets=u_action_offsets,
        mem_anchor_steps=u_mem_anchors,
        mem_offsets=u_mem_offsets,
        mem_lengths=u_mem_lengths,
        seed=seed + 30,
    )
    sb["pairs"] = pairs
    sb["tasks"] = [p["task"] for p in pairs]
    sb["init_noise_cpu"] = _build_init_noises(pairs, sb["q_steps"], base_seed=ddim_seed)

    # All models share the EXACT same frozen E9 Reader & Teacher!
    ref_policy = next(iter(models.values()))[0]
    s_z = float(ref_policy.latent_scale_s_z.item())
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    lat_base = _extract_model_latents_for_split(
        ref_policy,
        pairs,
        sb,
        mem_h=u_mem_h,
        mem_b=u_mem_b,
        mem_anchor_steps=u_mem_anchors,
        mem_offsets=u_mem_offsets,
        mem_lengths=u_mem_lengths,
        actions_10d=u_actions_10d,
        action_offsets=u_action_offsets,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
        compute_d4=False,
    )
    zq = lat_base["z_query"]
    z_src_rdr = lat_base["z_src_reader"]

    w_eps = _sample_wrong_task_episodes(pairs, unseen_demos, fold_filter=None, seed=seed + 7002)
    _, z_wrg_rdr = _extract_wrong_latents(
        ref_policy,
        pairs,
        w_eps,
        mem_h=u_mem_h,
        mem_b=u_mem_b,
        mem_offsets=u_mem_offsets,
        mem_lengths=u_mem_lengths,
        device=device,
        use_bf16=use_bf16,
        batch_size=batch_size,
    )

    # Pre-gather h_q (N, 4, 768)
    N = len(pairs)
    h_q_all = np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
    for i, p in enumerate(pairs):
        q_flat = int(u_mem_offsets[int(p["q_ep"])]) + int(p["q_anchor_idx"])
        h_q_all[i] = u_mem_h[q_flat].astype(np.float32)

    # Shared baselines: Full self-latent & K=0 Reader-only
    m_full = _decode_and_score_latents(
        ref_policy, zq, zq, z_src_rdr, sb["gt_chunks"], sb["valid_16"],
        sb["init_noise_cpu"], sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
    )
    m_k0 = _decode_and_score_latents(
        ref_policy, z_src_rdr, zq, z_src_rdr, sb["gt_chunks"], sb["valid_16"],
        sb["init_noise_cpu"], sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
    )
    m_k0_wrg = _decode_and_score_latents(
        ref_policy, z_wrg_rdr, zq, z_wrg_rdr, sb["gt_chunks"], sb["valid_16"],
        sb["init_noise_cpu"], sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
    )

    out: Dict[str, Any] = {
        "n_pairs": N,
        "s_z": s_z,
        "baselines": {
            "self_full": m_full,
            "reader_only_correct": m_k0,
            "reader_only_wrong": m_k0_wrg,
        },
        "models": {},
    }

    for name, (pol, payload) in models.items():
        z_corr_hat = np.empty((N, LATENT_DIM), dtype=np.float32)
        z_wrg_hat = np.empty((N, LATENT_DIM), dtype=np.float32)
        z_self_hat = np.empty((N, LATENT_DIM), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start
            hq_t = torch.from_numpy(h_q_all[start:end]).to(device=device, dtype=torch.float32)
            zc_t = torch.from_numpy(z_src_rdr[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)).to(device=device, dtype=torch.float32)
            zw_t = torch.from_numpy(z_wrg_rdr[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)).to(device=device, dtype=torch.float32)
            zs_t = torch.from_numpy(zq[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)).to(device=device, dtype=torch.float32)

            with amp_ctx:
                zh_c, _ = pol.adapt_latent(zc_t, hq_t, delta_scale=1.0)
                zh_w, _ = pol.adapt_latent(zw_t, hq_t, delta_scale=1.0)
                zh_s, _ = pol.adapt_latent(zs_t, hq_t, delta_scale=1.0)

            z_corr_hat[start:end] = zh_c.float().cpu().numpy().reshape(B, LATENT_DIM)
            z_wrg_hat[start:end] = zh_w.float().cpu().numpy().reshape(B, LATENT_DIM)
            z_self_hat[start:end] = zh_s.float().cpu().numpy().reshape(B, LATENT_DIM)

        m_corr = _decode_and_score_latents(
            pol, z_corr_hat, zq, z_src_rdr, sb["gt_chunks"], sb["valid_16"],
            sb["init_noise_cpu"], sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
        )
        m_wrg = _decode_and_score_latents(
            pol, z_wrg_hat, zq, z_wrg_rdr, sb["gt_chunks"], sb["valid_16"],
            sb["init_noise_cpu"], sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
        )
        m_self_pres = _decode_and_score_latents(
            pol, z_self_hat, zq, zq, sb["gt_chunks"], sb["valid_16"],
            sb["init_noise_cpu"], sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
        )

        arr_corr, _ = _safe_arr(m_k0["ddim_10d_mse"], m_corr["ddim_10d_mse"], m_full["ddim_10d_mse"])
        wrong_delta = m_wrg["ddim_10d_mse"] - m_corr["ddim_10d_mse"]
        wrong_ratio = m_wrg["ddim_10d_mse"] / max(m_corr["ddim_10d_mse"], 1e-12)
        self_deg_ratio = m_self_pres["ddim_10d_mse"] / max(m_full["ddim_10d_mse"], 1e-12)

        out["models"][name] = {
            "epoch": payload.get("epoch"),
            "correct": m_corr,
            "wrong_task": m_wrg,
            "self_d3_preservation": m_self_pres,
            "arr_10d_vs_reader0": arr_corr,
            "wrong_delta_mse10": float(wrong_delta),
            "wrong_ratio_mse10": float(wrong_ratio),
            "self_d3_over_d1_ratio": float(self_deg_ratio),
        }
        print(
            f"  [Unseen-Chain {name:12s}] Correct MSE10={m_corr['ddim_10d_mse']:.5f} (ARR={100*(arr_corr or 0):.1f}%, Lz={m_corr['latent_lz']:.4f}) | "
            f"Wrong MSE10={m_wrg['ddim_10d_mse']:.5f} (Δ={wrong_delta:+.5f}, {wrong_ratio:.2f}x) | "
            f"D3 Self-Pres MSE10={m_self_pres['ddim_10d_mse']:.5f} ({self_deg_ratio:.2f}x D1, Lz={m_self_pres['latent_lz']:.4f})",
            flush=True,
        )
    return out


@torch.inference_mode()
def run_d1_d4_for_all_models(
    models: Dict[str, Tuple[torch.nn.Module, Dict[str, Any]]],
    *,
    out_dir: Path,
    device: torch.device,
    use_bf16: bool = True,
    base_seed: int = 12345,
    demo_types: Tuple[str, ...] = ("default", "shifted", "oracle_gt"),
) -> Dict[str, Any]:
    """Run exact D1-D4 offline decomposition (Default 100, Shifted 100, Oracle-GT 331)
    for ALL models in a SINGLE shared CLIP + E9 Stage 1 + Frozen Reader pass per demo!"""
    t0 = time.monotonic()
    tasks = get_chain_tasks()
    specs = _collect_demo_specs(tasks, demo_types=demo_types)

    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    ref_policy = next(iter(models.values()))[0]
    s_z = float(ref_policy.latent_scale_s_z.item())
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    rows_by_model: Dict[str, List[Dict[str, Any]]] = {name: [] for name in models}

    print(
        f"[E10 D1-D4 SharedPass] Evaluating {len(models)} models on {len(specs)} demos "
        f"({demo_types}) in ONE shared CLIP/Teacher/Reader pass...",
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
        demo_all_feats = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if device.type == "cuda" else None,
        )

        with amp_ctx:
            h_mem, b_mem, anchors = ref_policy.teacher.extract_behavior_memory_from_demo_features(
                demo_all_feats, stride=BEHAVIOR_STRIDE
            )
            h_mem_f = h_mem.float().contiguous()
            b_mem_f = b_mem.float().contiguous()
            M = len(anchors)
            h_q = h_mem_f
            z_d1 = b_mem_f

            h_mem_exp = h_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            b_mem_exp = b_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            z_d2, attn_w = ref_policy.reader(h_q, h_mem_exp, b_mem_exp, mem_pad_mask=None)

            noises = []
            for t_k in anchors:
                s_k = _stable_seed(f"{demo_type}/{task}/{demo_label}/t{int(t_k)}", base_seed=base_seed)
                gen = torch.Generator(device="cpu").manual_seed(s_k)
                noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
            init_noise = torch.stack(noises, dim=0).to(device=device, dtype=torch.float32)

            # Shared D1 & D2 DDIM predictions (identical across all models!)
            pred_d1 = ref_policy.teacher.sample_from_z_self(z_d1, init_noise=init_noise)
            pred_d2 = ref_policy.teacher.sample_from_z_self(z_d2, init_noise=init_noise)

        actions_10d_np = action_7d_to_10d(actions_7d)
        step_grid = anchors[:, None] + np.arange(ACTION_HORIZON, dtype=np.int32)[None, :]
        valid_16_np = step_grid < T
        clipped_grid = np.minimum(step_grid, T - 1)
        gt_10d = torch.from_numpy(actions_10d_np[clipped_grid]).to(device=device, dtype=torch.float32)
        valid_16 = torch.from_numpy(valid_16_np).to(device=device, dtype=torch.bool)
        gt_chunks_7d = actions_7d[clipped_grid]

        am_d1 = _compute_per_anchor_action_metrics(pred_d1, gt_10d, valid_16)
        am_d2 = _compute_per_anchor_action_metrics(pred_d2, gt_10d, valid_16)

        z_ref = z_d1.float()
        z_ref_flat = z_ref.reshape(M, -1)
        z_d2_f = z_d2.float()
        lm_d1 = {
            "mse": np.zeros(M, dtype=np.float64),
            "cos": np.ones(M, dtype=np.float64),
            "sq_mean": z_ref.square().mean(dim=(1, 2)).cpu().numpy(),
        }
        lm_d2 = {
            "mse": (z_d2_f - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
            "cos": F.cosine_similarity(z_d2_f.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
            "sq_mean": z_d2_f.square().mean(dim=(1, 2)).cpu().numpy(),
        }

        w_np = attn_w.float().cpu().numpy()
        ast = compute_attention_stats(attn_w, mem_pad_mask=None)
        ent_np = ast["entropy"].cpu().numpy()
        nent_np = ast["norm_entropy"].cpu().numpy()
        eff_np = ast["eff_pos_exp"].cpu().numpy()
        wmax_np = ast["max_weight"].cpu().numpy()
        idx_grid = np.arange(M, dtype=np.float64)
        anchors_f = anchors.astype(np.float64)

        # Per-model D3 and D4
        for name, (pol, _) in models.items():
            with amp_ctx:
                z_d3, _ = pol.adapt_latent(z_d1, h_q, delta_scale=1.0)
                z_d4, _ = pol.adapt_latent(z_d2, h_q, delta_scale=1.0)
                pred_d3 = pol.teacher.sample_from_z_self(z_d3, init_noise=init_noise)
                pred_d4 = pol.teacher.sample_from_z_self(z_d4, init_noise=init_noise)

            am_d3 = _compute_per_anchor_action_metrics(pred_d3, gt_10d, valid_16)
            am_d4 = _compute_per_anchor_action_metrics(pred_d4, gt_10d, valid_16)
            z_d3_f = z_d3.float()
            z_d4_f = z_d4.float()
            act_metrics = {
                "D1_teacher": am_d1,
                "D2_retrieval": am_d2,
                "D3_transfer": am_d3,
                "D4_full": am_d4,
            }
            lat_metrics = {
                "D1_teacher": lm_d1,
                "D2_retrieval": lm_d2,
                "D3_transfer": {
                    "mse": (z_d3_f - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
                    "cos": F.cosine_similarity(z_d3_f.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
                    "sq_mean": z_d3_f.square().mean(dim=(1, 2)).cpu().numpy(),
                },
                "D4_full": {
                    "mse": (z_d4_f - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
                    "cos": F.cosine_similarity(z_d4_f.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
                    "sq_mean": z_d4_f.square().mean(dim=(1, 2)).cpu().numpy(),
                },
            }
            lat_mse_d4_d3 = (z_d4_f - z_d3_f).square().mean(dim=(1, 2)).cpu().numpy()
            lat_cos_d4_d3 = F.cosine_similarity(z_d4_f.reshape(M, -1), z_d3_f.reshape(M, -1), dim=-1).cpu().numpy()

            m_rows = rows_by_model[name]
            for k in range(M):
                wk = w_np[k]
                argmax_j = int(np.argmax(wk))
                signed_slots = argmax_j - k
                err_slots = abs(signed_slots)
                err_steps = abs(int(anchors[argmax_j]) - int(anchors[k]))
                err_norm = float(err_slots) / float(max(M - 1, 1))
                exp_j = float(np.sum(wk * idx_grid))
                exp_step = float(np.sum(wk * anchors_f))
                v_mask = valid_16_np[k]
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
                    "n_valid_16": int(am_d1["n_valid_16"][k]),
                    "n_valid_12": int(am_d1["n_valid_12"][k]),
                    "reader_max_weight": float(wmax_np[k]),
                    "reader_self_weight": float(wk[k]),
                    "reader_argmax_idx": argmax_j,
                    "reader_exact_hit": int(argmax_j == k),
                    "reader_within1_hit": int(err_slots <= 1),
                    "reader_pos_err_argmax_slots": int(err_slots),
                    "reader_pos_err_argmax_steps": int(err_steps),
                    "reader_pos_err_argmax_norm": err_norm,
                    "reader_signed_err_argmax_slots": int(signed_slots),
                    "reader_pos_err_expected_slots": abs(exp_j - float(k)),
                    "reader_pos_err_expected_steps": abs(exp_step - float(anchors[k])),
                    "reader_expected_abs_dist_slots": float(np.sum(wk * np.abs(idx_grid - float(k)))),
                    "reader_expected_abs_dist_steps": float(np.sum(wk * np.abs(anchors_f - float(anchors[k])))),
                    "reader_entropy": float(ent_np[k]),
                    "reader_norm_entropy": float(nent_np[k]),
                    "reader_eff_pos_exp": float(eff_np[k]),
                    "latent_mse_d4_vs_d3": float(lat_mse_d4_d3[k]),
                    "latent_cos_d4_vs_d3": float(lat_cos_d4_d3[k]),
                    "gt_pos_l2": float(np.mean(np.linalg.norm(gt_chunks_7d[k, v_mask, :3], axis=-1))),
                    "gt_grip_pos_frac": float(np.mean(gt_chunks_7d[k, v_mask, 6] >= 0.0)),
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
                m_rows.append(row)

        if idx % 100 == 0 or idx == len(specs):
            dt = time.monotonic() - t0
            print(f"[E10 D1-D4 SharedPass] {idx}/{len(specs)} demos in {dt:.1f}s", flush=True)

    d1_d4_summaries: Dict[str, Any] = {}
    for name, all_rows in rows_by_model.items():
        by_dt: Dict[str, Any] = {}
        per_task_by_dt: Dict[str, Dict[str, Any]] = {}
        for dt in demo_types:
            dt_rows = [r for r in all_rows if r["demo_type"] == dt]
            by_dt[dt] = _aggregate_rows(dt_rows, s_z=s_z)
            per_task_by_dt[dt] = {}
            for task in tasks:
                t_rows = [r for r in dt_rows if r["task"] == task]
                if t_rows:
                    per_task_by_dt[dt][task] = _aggregate_rows(t_rows, s_z=s_z)
        shared_9 = [t for t in tasks if t in per_task_by_dt.get("oracle_gt", {})]
        macro_9 = {dt: _macro_average_tasks(per_task_by_dt[dt], shared_9) for dt in demo_types}
        macro_10 = {dt: _macro_average_tasks(per_task_by_dt[dt], tasks) for dt in demo_types}
        res_m = {
            "model": name,
            "by_demo_type_micro": by_dt,
            "by_demo_type_macro_all_10": macro_10,
            "by_demo_type_macro_shared_9": macro_9,
        }
        d1_d4_summaries[name] = res_m
        m_dir = out_dir / "d1_d4" / name
        m_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(res_m, m_dir / "d1_d4_summary.json")

        print(f"\n  === D1-D4 9-Task Macro ({name}) ===", flush=True)
        for dt in demo_types:
            blk = macro_9[dt]["modes"]
            st9 = by_dt[dt]["d3_stratification"].get("w_self_ge_0.9", {})
            d3_over_d1_st9 = st9["D3_mse10"] / max(st9["D1_mse10"], 1e-12) if st9.get("num_anchors", 0) > 0 else float("nan")
            print(
                f"    [{dt:9s}] D1={blk['D1_teacher']['ddim_10d_mse']:.5f} | "
                f"D2={blk['D2_retrieval']['ddim_10d_mse']:.5f} | "
                f"D3={blk['D3_transfer']['ddim_10d_mse']:.5f} (Lz={blk['D3_transfer']['latent_lz_vs_self']:.4f}) | "
                f"D4={blk['D4_full']['ddim_10d_mse']:.5f} (Lz={blk['D4_full']['latent_lz_vs_self']:.4f}) | "
                f"[w>=0.9] D3/D1={d3_over_d1_st9:.2f}x (D3_Lz={st9.get('D3_lat_mse', 0)/(s_z*s_z):.4f})",
                flush=True,
            )

    return d1_d4_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--include-e9-baseline", action="store_true", default=True)
    parser.add_argument("--skip-d1-d4", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_dir = args.output_root / "offline_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Tuple[torch.nn.Module, Dict[str, Any]]] = {}
    if args.include_e9_baseline:
        pol_e9, pay_e9 = load_e9_stage2_direct_policy(
            E9_STAGE2_CKPT, stage1_ckpt=E9_STAGE1_CKPT, device=device
        )
        models["e9_direct"] = (pol_e9, pay_e9)
    for v in args.variants:
        ckpt = args.output_root / v / "train/best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {v}: {ckpt}")
        models[v] = load_e10_policy(ckpt, device=device)

    # 1. Seen-val summary directly from each variant's train/final_eval.json + e9 final_eval.json
    seen_val_summary: Dict[str, Any] = {}
    e9_fe = E9_STAGE2_CKPT.parent / "final_eval.json"
    if e9_fe.is_file():
        seen_val_summary["e9_direct"] = json.loads(e9_fe.read_text())
    for v in args.variants:
        fe = args.output_root / v / "train/final_eval.json"
        if fe.is_file():
            seen_val_summary[v] = json.loads(fe.read_text())

    # 2. Unseen-Chain 3,000-pair Correct / Wrong / Self-Preservation evaluation
    unseen_res = evaluate_unseen_chain_all(models, batch_size=args.batch_size, device=device)

    # 3. Full D1-D4 offline decomposition & self-latent preservation
    d1_d4_res = {}
    if not args.skip_d1_d4:
        d1_d4_res = run_d1_d4_for_all_models(models, out_dir=eval_dir, device=device)

    combined = {
        "seen_val_final_eval": seen_val_summary,
        "unseen_chain_eval": unseen_res,
        "d1_d4_eval": d1_d4_res,
    }
    atomic_write_json(combined, eval_dir / "offline_eval_summary.json")

    # Compact flat CSV comparison table
    flat_rows = []
    for name in models:
        sv = seen_val_summary.get(name, {})
        sv_dm = sv.get("ddim", {})
        sv_lt = sv.get("latent", {})
        sv_c_mse = sv_dm.get("cross_with_delta", {}).get("action10_mse")
        sv_w_mse = sv_dm.get("wrong_demo_with_delta", {}).get("action10_mse")
        sv_w_delta = sv_dm.get("demo_sensitivity_action10_mse")
        sv_w_ratio = (sv_w_mse / max(sv_c_mse, 1e-12)) if (sv_c_mse and sv_w_mse) else None
        un = unseen_res["models"][name]
        d4_def = d1_d4_res.get(name, {}).get("by_demo_type_macro_shared_9", {}).get("default", {}).get("modes", {})
        d4_ogt = d1_d4_res.get(name, {}).get("by_demo_type_macro_shared_9", {}).get("oracle_gt", {}).get("modes", {})
        st9_def = d1_d4_res.get(name, {}).get("by_demo_type_micro", {}).get("default", {}).get("d3_stratification", {}).get("w_self_ge_0.9", {})
        st9_ogt = d1_d4_res.get(name, {}).get("by_demo_type_micro", {}).get("oracle_gt", {}).get("d3_stratification", {}).get("w_self_ge_0.9", {})

        flat_rows.append(
            {
                "model": name,
                "best_epoch": un["epoch"],
                "seen_cross_mse10": sv_c_mse,
                "seen_cross_mae7": sv_dm.get("cross_with_delta", {}).get("action7_mae"),
                "seen_cross_grip_acc": sv_dm.get("cross_with_delta", {}).get("grip_sign_acc"),
                "seen_cross_lz": sv_lt.get("cross_with_delta_L_z"),
                "seen_wrong_delta_mse10": sv_w_delta,
                "seen_wrong_ratio_mse10": sv_w_ratio,
                "unseen_correct_mse10": un["correct"]["ddim_10d_mse"],
                "unseen_correct_mae7": un["correct"]["ddim_7d_mae"],
                "unseen_correct_grip_acc": un["correct"]["grip_acc16"],
                "unseen_correct_lz": un["correct"]["latent_lz"],
                "unseen_arr_vs_reader0": un["arr_10d_vs_reader0"],
                "unseen_wrong_mse10": un["wrong_task"]["ddim_10d_mse"],
                "unseen_wrong_delta_mse10": un["wrong_delta_mse10"],
                "unseen_wrong_ratio_mse10": un["wrong_ratio_mse10"],
                "unseen_self_d3_mse10": un["self_d3_preservation"]["ddim_10d_mse"],
                "unseen_self_d3_over_d1": un["self_d3_over_d1_ratio"],
                "d1d4_def_d3_mse10": d4_def.get("D3_transfer", {}).get("ddim_10d_mse"),
                "d1d4_def_d4_mse10": d4_def.get("D4_full", {}).get("ddim_10d_mse"),
                "d1d4_ogt_d3_mse10": d4_ogt.get("D3_transfer", {}).get("ddim_10d_mse"),
                "d1d4_ogt_d4_mse10": d4_ogt.get("D4_full", {}).get("ddim_10d_mse"),
                "d1d4_ogt_d4_lz": d4_ogt.get("D4_full", {}).get("latent_lz_vs_self"),
                "w90_def_d3_over_d1": (
                    st9_def["D3_mse10"] / max(st9_def["D1_mse10"], 1e-12)
                    if st9_def.get("num_anchors", 0) > 0
                    else None
                ),
                "w90_ogt_d3_over_d1": (
                    st9_ogt["D3_mse10"] / max(st9_ogt["D1_mse10"], 1e-12)
                    if st9_ogt.get("num_anchors", 0) > 0
                    else None
                ),
            }
        )
    _write_csv(flat_rows, eval_dir / "offline_comparison.csv")
    print(f"\n[E10 OfflineEval DONE] Saved to {eval_dir}", flush=True)


if __name__ == "__main__":
    main()
