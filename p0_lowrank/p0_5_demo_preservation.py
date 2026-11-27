"""P0.5: Demo Preservation Diagnosis on frozen E9 checkpoint & P0 train-fitted SVD basis.

Tests whether `(I - P) @ z_source` preserves genuine demo-specific action value beyond
the low-rank adaptation subspace `P = U_K @ U_K.T` (K in {8, 16, 32, 64}), comparing:
  - Correct      : (I - P) @ z_source + P @ z_query
  - Wrong-task   : (I - P) @ z_wrong  + P @ z_query
  - Mean         : (I - P) @ z_mean   + P @ z_query
  - Behavior PCA : z_mean + V_K @ V_K.T @ (z_query - z_mean)   [train-fitted PCA of Z_query]

Also exports the exact P0 train-fitted SVD bases (`e9_p0_bases.pt`) so P1 (E10-A/B/C/Control)
uses bit-for-bit identical fixed bases without refitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from e5.train import atomic_write_json
from e9.stage2_direct_model import load_e9_stage2_direct_policy

from . import (
    ACTION_10D_NAMES,
    E5_CACHE_DIR,
    E9_BEHAVIOR_MEM_DIR,
    E9_STAGE1_CKPT,
    E9_STAGE2_CKPT,
    EMBED_DIM,
    LATENT_DIM,
    LATENT_TOKENS,
    OUTPUT_ROOT,
)
from .alignment import sample_cached_split_pairs, sample_unseen_chain_pairs
from .cache_unseen import DEFAULT_UNSEEN_CACHE_DIR, build_unseen_chain_cache
from .run_diagnosis import (
    _build_init_noises,
    _compute_alignment_and_gt_for_split,
    _compute_svd_stats,
    _decode_and_score_latents,
    _extract_model_latents_for_split,
    _safe_arr,
    _write_csv,
)

P05_RANKS: Tuple[int, ...] = (8, 16, 32, 64)
P05_OUT_DIR = OUTPUT_ROOT / "p0_5_demo_preservation"
BASES_EXPORT_PATH = OUTPUT_ROOT / "e9_p0_bases.pt"


def _sample_wrong_task_episodes(
    pairs: List[Dict[str, Any]],
    all_demos_meta: List[Dict[str, Any]],
    *,
    fold_filter: Optional[str],
    seed: int,
) -> np.ndarray:
    """For each pair, deterministically pick a demo episode from a DIFFERENT task
    in the same split (`fold_filter`), with zero leakage."""
    by_task: Dict[str, List[int]] = defaultdict(list)
    for ep_idx, row in enumerate(all_demos_meta):
        if fold_filter is not None and row.get("fold") != fold_filter:
            continue
        by_task[row["task"]].append(ep_idx)

    tasks = sorted(by_task.keys())
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    rng = np.random.default_rng(seed)
    wrong_eps = np.empty(len(pairs), dtype=np.int64)
    for i, p in enumerate(pairs):
        t_i = task_to_idx[p["task"]]
        other_t = tasks[(t_i + 1 + int(rng.integers(len(tasks) - 1))) % len(tasks)]
        cands = by_task[other_t]
        w_ep = int(cands[int(rng.integers(len(cands)))])
        assert all_demos_meta[w_ep]["task"] != p["task"]
        wrong_eps[i] = w_ep
    return wrong_eps


@torch.inference_mode()
def _extract_wrong_latents(
    policy: torch.nn.Module,
    pairs: List[Dict[str, Any]],
    wrong_eps: np.ndarray,
    *,
    mem_h: np.ndarray,
    mem_b: np.ndarray,
    mem_offsets: np.ndarray,
    mem_lengths: np.ndarray,
    device: torch.device,
    use_bf16: bool,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract wrong-task source latents under both:
      - offline_wrong: same relative progress anchor in wrong-task demo
      - reader_wrong : frozen E9 Reader attending to wrong-task demo with h_q
    """
    N = len(pairs)
    z_wrong_off = np.empty((N, LATENT_DIM), dtype=np.float32)
    z_wrong_rdr = np.empty((N, LATENT_DIM), dtype=np.float32)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        batch_pairs = pairs[start:end]
        w_eps = wrong_eps[start:end]
        lens = [int(mem_lengths[int(ep)]) for ep in w_eps]
        max_m = max(lens)

        h_q_np = np.empty((B, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        h_mem_np = np.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        b_mem_np = np.zeros((B, max_m, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        pad_mask_np = np.ones((B, max_m), dtype=bool)

        for bi, (p, w_ep_i) in enumerate(zip(batch_pairs, w_eps)):
            q_ep = int(p["q_ep"])
            k = int(p["q_anchor_idx"])
            w_ep = int(w_ep_i)
            q_len = int(mem_lengths[q_ep])
            w_len = int(mem_lengths[w_ep])
            rel = float(k) / float(max(q_len - 1, 1))
            j_w = int(round(rel * float(max(w_len - 1, 0))))
            j_w = min(max(j_w, 0), w_len - 1)

            q_flat = int(mem_offsets[q_ep]) + k
            w_m0 = int(mem_offsets[w_ep])
            z_wrong_off[start + bi] = mem_b[w_m0 + j_w].astype(np.float32).reshape(LATENT_DIM)
            h_q_np[bi] = mem_h[q_flat].astype(np.float32)
            h_mem_np[bi, :w_len] = mem_h[w_m0 : w_m0 + w_len].astype(np.float32)
            b_mem_np[bi, :w_len] = mem_b[w_m0 : w_m0 + w_len].astype(np.float32)
            pad_mask_np[bi, :w_len] = False

        h_q_t = torch.from_numpy(h_q_np).to(device=device, dtype=torch.float32)
        h_mem_t = torch.from_numpy(h_mem_np).to(device=device, dtype=torch.float32)
        b_mem_t = torch.from_numpy(b_mem_np).to(device=device, dtype=torch.float32)
        pad_t = torch.from_numpy(pad_mask_np).to(device=device, dtype=torch.bool)

        with amp_ctx:
            z_c, _ = policy.reader(h_q_t, h_mem_t, b_mem_t, mem_pad_mask=pad_t)
        z_wrong_rdr[start:end] = z_c.float().cpu().numpy().reshape(B, LATENT_DIM)

    return z_wrong_off, z_wrong_rdr


def _plot_p05(summary: Dict[str, Any], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=160)
    xs = np.arange(len(P05_RANKS), dtype=np.float64)
    xlabels = [str(k) for k in P05_RANKS]

    cond_styles = [
        ("correct", "Correct: (I-P)z_src + P z_q", "#1f77b4", "o", "-"),
        ("wrong_task", "Wrong-task: (I-P)z_wrong + P z_q", "#d62728", "s", "-"),
        ("mean", "Mean: (I-P)z_mean + P z_q", "#ff7f0e", "^", "--"),
        ("behavior_pca", "Behavior PCA: z_mean + V_K V_K^T (z_q - z_mean)", "#2ca02c", "D", "-."),
    ]

    for row_i, split in enumerate(("seen_val", "unseen_chain")):
        for col_i, align in enumerate(("reader_align", "offline_align")):
            ax = axes[row_i, col_i]
            rows = summary["results"][align][split]
            by_ck = {(r["condition"], r["rank_k"]): r for r in rows}
            full_mse = by_ck[("full_self", LATENT_DIM)]["ddim_10d_mse"]

            for cond, label, color, marker, ls in cond_styles:
                ys = [by_ck[(cond, k)]["ddim_10d_mse"] for k in P05_RANKS]
                ax.plot(xs, ys, marker=marker, color=color, linestyle=ls, linewidth=2.2, markersize=6, label=label)
                for xi, yi in zip(xs, ys):
                    ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5, color=color)

            ax.axhline(full_mse, color="black", linestyle=":", linewidth=1.2, label=f"Full Self-Latent ({full_mse:.4f})")
            s_lbl = "Seen-Val (3,000)" if split == "seen_val" else "Unseen-Chain (3,000)"
            a_lbl = "Frozen Reader Align" if align == "reader_align" else "Offline Phase Align"
            ax.set_title(f"E9 P0.5 — {s_lbl} — {a_lbl}")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"K={k}" for k in xlabels])
            ax.set_ylabel("DDIM 10D Action MSE (lower is better)")
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("P0.5 Demo Preservation Diagnosis: Does (I - P) z_source Carry Real Action Value?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p_out = out_dir / "p0_5_demo_preservation.png"
    fig.savefig(p_out)
    plt.close(fig)
    return p_out


def run_p05(
    *,
    out_dir: Path = P05_OUT_DIR,
    bases_path: Path = BASES_EXPORT_PATH,
    unseen_cache_dir: Path = DEFAULT_UNSEEN_CACHE_DIR,
    n_train: int = 8000,
    n_seen_val: int = 3000,
    n_unseen: int = 3000,
    seed: int = 42,
    ddim_seed: int = 12345,
    batch_size: int = 1024,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_unseen_chain_cache(out_dir=unseen_cache_dir, demos_per_task=20, device=device, use_bf16=use_bf16)

    # Load frozen E9 policy (Encoder + Reader + Action Decoder ALL frozen)
    policy, payload = load_e9_stage2_direct_policy(
        E9_STAGE2_CKPT, stage1_ckpt=E9_STAGE1_CKPT, device=device
    )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    assert sum(p.numel() for p in policy.parameters() if p.requires_grad) == 0
    s_z = float(policy.latent_scale_s_z.item())
    # Exact E9 training z_self mean (matching E9 Stage 2 direct_latent_bias)
    z_mean_e9_train = (
        np.load(E9_BEHAVIOR_MEM_DIR / "train_z_self_mean.npy").astype(np.float32).reshape(LATENT_DIM)
    )

    episodes = json.loads((E5_CACHE_DIR / "episodes.json").read_text())
    actions_10d_e5 = np.load(E5_CACHE_DIR / "actions_10d.npy", mmap_mode="r")
    ee_pos_e5 = np.load(E5_CACHE_DIR / "proprio_11d.npy", mmap_mode="r")[:, :3]
    action_offsets_e5 = np.load(E5_CACHE_DIR / "action_offsets.npy")
    e9_mem_h = np.load(E9_BEHAVIOR_MEM_DIR / "mem_h.npy", mmap_mode="r")
    e9_mem_b = np.load(E9_BEHAVIOR_MEM_DIR / "mem_b.npy", mmap_mode="r")
    e9_mem_anchors = np.load(E9_BEHAVIOR_MEM_DIR / "mem_anchor_steps.npy")
    e9_mem_offsets = np.load(E9_BEHAVIOR_MEM_DIR / "mem_offsets.npy")
    e9_mem_lengths = np.load(E9_BEHAVIOR_MEM_DIR / "mem_lengths.npy")

    unseen_meta = json.loads((unseen_cache_dir / "unseen_meta.json").read_text())
    unseen_demos = unseen_meta["demos"]
    unseen_mem_h = np.load(unseen_cache_dir / "mem_h.npy", mmap_mode="r")
    unseen_mem_b = np.load(unseen_cache_dir / "mem_b_e9.npy", mmap_mode="r")
    unseen_mem_anchors = np.load(unseen_cache_dir / "mem_anchor_steps.npy")
    unseen_mem_offsets = np.load(unseen_cache_dir / "mem_offsets.npy")
    unseen_mem_lengths = np.load(unseen_cache_dir / "mem_lengths.npy")
    unseen_actions_10d = np.load(unseen_cache_dir / "actions_10d.npy", mmap_mode="r")
    unseen_ee_pos = np.load(unseen_cache_dir / "ee_pos.npy", mmap_mode="r")
    unseen_action_offsets = np.load(unseen_cache_dir / "action_offsets.npy")

    # Bit-exact same pair sampling & DDIM noise as P0
    train_pairs = sample_cached_split_pairs(episodes, e9_mem_lengths, fold="train", target_pairs=n_train, seed=seed)
    seen_val_pairs = sample_cached_split_pairs(episodes, e9_mem_lengths, fold="val", target_pairs=n_seen_val, seed=seed + 1001)
    unseen_pairs = sample_unseen_chain_pairs(unseen_demos, target_pairs=n_unseen, seed=seed + 2002)

    split_bundles: Dict[str, Dict[str, Any]] = {}
    for sname, spairs, a10, eep, aoff, ma, mo, ml, s_off in (
        ("train", train_pairs, actions_10d_e5, ee_pos_e5, action_offsets_e5, e9_mem_anchors, e9_mem_offsets, e9_mem_lengths, 10),
        ("seen_val", seen_val_pairs, actions_10d_e5, ee_pos_e5, action_offsets_e5, e9_mem_anchors, e9_mem_offsets, e9_mem_lengths, 20),
        ("unseen_chain", unseen_pairs, unseen_actions_10d, unseen_ee_pos, unseen_action_offsets, unseen_mem_anchors, unseen_mem_offsets, unseen_mem_lengths, 30),
    ):
        sb = _compute_alignment_and_gt_for_split(
            sname, spairs,
            actions_10d=a10, ee_pos_all=eep, action_offsets=aoff,
            mem_anchor_steps=ma, mem_offsets=mo, mem_lengths=ml,
            seed=seed + s_off,
        )
        sb["pairs"] = spairs
        sb["tasks"] = [p["task"] for p in spairs]
        if sname != "train":
            sb["init_noise_cpu"] = _build_init_noises(spairs, sb["q_steps"], base_seed=ddim_seed)
        split_bundles[sname] = sb

    # Extract E9 latents for train / seen_val / unseen_chain
    lat_by_split: Dict[str, Dict[str, Any]] = {}
    for sname in ("train", "seen_val", "unseen_chain"):
        is_u = sname == "unseen_chain"
        lat_by_split[sname] = _extract_model_latents_for_split(
            policy,
            split_bundles[sname]["pairs"],
            split_bundles[sname],
            mem_h=unseen_mem_h if is_u else e9_mem_h,
            mem_b=unseen_mem_b if is_u else e9_mem_b,
            mem_anchor_steps=unseen_mem_anchors if is_u else e9_mem_anchors,
            mem_offsets=unseen_mem_offsets if is_u else e9_mem_offsets,
            mem_lengths=unseen_mem_lengths if is_u else e9_mem_lengths,
            actions_10d=unseen_actions_10d if is_u else actions_10d_e5,
            action_offsets=unseen_action_offsets if is_u else action_offsets_e5,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
            compute_d4=False,
        )

    # Extract Wrong-task source latents for seen_val and unseen_chain
    wrong_by_split: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sname, meta_list, fold_f, s_seed in (
        ("seen_val", episodes, "val", seed + 7001),
        ("unseen_chain", unseen_demos, None, seed + 7002),
    ):
        is_u = sname == "unseen_chain"
        w_eps = _sample_wrong_task_episodes(
            split_bundles[sname]["pairs"], meta_list, fold_filter=fold_f, seed=s_seed
        )
        wrong_by_split[sname] = _extract_wrong_latents(
            policy,
            split_bundles[sname]["pairs"],
            w_eps,
            mem_h=unseen_mem_h if is_u else e9_mem_h,
            mem_b=unseen_mem_b if is_u else e9_mem_b,
            mem_offsets=unseen_mem_offsets if is_u else e9_mem_offsets,
            mem_lengths=unseen_mem_lengths if is_u else e9_mem_lengths,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
        )

    # Fit exact P0 train-only SVD bases & Behavior PCA on train 8k
    zq_tr = lat_by_split["train"]["z_query"].astype(np.float64)
    z_mean_tr8k = zq_tr.mean(axis=0).astype(np.float32)
    dz_tr_off = zq_tr - lat_by_split["train"]["z_src_offline"].astype(np.float64)
    dz_tr_rdr = zq_tr - lat_by_split["train"]["z_src_reader"].astype(np.float64)

    svd_off = _compute_svd_stats(dz_tr_off, center=False)
    svd_rdr = _compute_svd_stats(dz_tr_rdr, center=False)
    svd_beh_pca = _compute_svd_stats(zq_tr, center=True)

    U_off = svd_off["_Vt"].T.astype(np.float32)       # (3072, 3072)
    U_rdr = svd_rdr["_Vt"].T.astype(np.float32)       # (3072, 3072)
    V_pca = svd_beh_pca["_Vt"].T.astype(np.float32)   # (3072, 3072)

    # Export fixed P0 train-fitted SVD bases for P1 (E10-A/B/C/Control)
    bases_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": "e9_p0_train_svd_bases_v1",
            "n_train": len(train_pairs),
            "seed": seed,
            "s_z": s_z,
            "U_reader": torch.from_numpy(U_rdr[:, :256].copy()),
            "S_reader": torch.from_numpy(svd_rdr["_S"][:256].astype(np.float32).copy()),
            "U_offline": torch.from_numpy(U_off[:, :256].copy()),
            "S_offline": torch.from_numpy(svd_off["_S"][:256].astype(np.float32).copy()),
            "V_behavior_pca": torch.from_numpy(V_pca[:, :256].copy()),
            "S_behavior_pca": torch.from_numpy(svd_beh_pca["_S"][:256].astype(np.float32).copy()),
            "z_mean_train_8k": torch.from_numpy(z_mean_tr8k.copy()),
            "z_mean_e9_full_train": torch.from_numpy(z_mean_e9_train.copy()),
        },
        bases_path,
    )
    mean_cos = float(
        np.dot(z_mean_tr8k, z_mean_e9_train)
        / max(np.linalg.norm(z_mean_tr8k) * np.linalg.norm(z_mean_e9_train), 1e-12)
    )
    print(
        f"[P0.5] Exported P0 train-only SVD bases -> {bases_path} "
        f"(cos(z_mean_8k, z_mean_full_train)={mean_cos:.6f})",
        flush=True,
    )

    # Evaluate P0.5 conditions on seen_val & unseen_chain for both reader_align and offline_align
    all_rows: List[Dict[str, Any]] = []
    results_nested: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"reader_align": {}, "offline_align": {}}

    # Pre-decode shared full_self (z_query) and z_mean_0 once per split!
    shared_full: Dict[str, Dict[str, Any]] = {}
    shared_mean0: Dict[str, Dict[str, Any]] = {}
    shared_pca: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for split in ("seen_val", "unseen_chain"):
        sb = split_bundles[split]
        zq = lat_by_split[split]["z_query"]
        zm_mat = np.broadcast_to(z_mean_tr8k[None, :], zq.shape).copy()

        shared_full[split] = _decode_and_score_latents(
            policy, zq, zq, zq, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
            sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
        )
        shared_mean0[split] = _decode_and_score_latents(
            policy, zm_mat, zq, zm_mat, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
            sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
        )
        for k in P05_RANKS:
            Vk = V_pca[:, :k]
            z_pca = z_mean_tr8k[None, :] + ((zq - z_mean_tr8k[None, :]) @ Vk) @ Vk.T
            shared_pca[(split, k)] = _decode_and_score_latents(
                policy, z_pca, zq, zm_mat, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
                sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
            )

    for align_name, src_key, w_idx, U_mat in (
        ("reader_align", "z_src_reader", 1, U_rdr),
        ("offline_align", "z_src_offline", 0, U_off),
    ):
        for split in ("seen_val", "unseen_chain"):
            sb = split_bundles[split]
            zq = lat_by_split[split]["z_query"]
            zs = lat_by_split[split][src_key]
            zw = wrong_by_split[split][w_idx]
            zm = np.broadcast_to(z_mean_tr8k[None, :], zq.shape).copy()

            split_rows: List[Dict[str, Any]] = []

            # K=0 baselines
            m_src0 = _decode_and_score_latents(
                policy, zs, zq, zs, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
                sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
            )
            m_wrg0 = _decode_and_score_latents(
                policy, zw, zq, zw, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
                sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
            )
            m_mn0 = shared_mean0[split]
            m_full = shared_full[split]

            base_mse0 = m_src0["ddim_10d_mse"]
            full_mse = m_full["ddim_10d_mse"]

            def _make_row(cond: str, k: int, metrics: Dict[str, Any], self_mse0: float) -> Dict[str, Any]:
                arr_vs_src, ok_src = _safe_arr(base_mse0, metrics["ddim_10d_mse"], full_mse)
                arr_self, ok_self = _safe_arr(self_mse0, metrics["ddim_10d_mse"], full_mse)
                arr_pos, _ = _safe_arr(m_src0["ddim_pos_mse"], metrics["ddim_pos_mse"], m_full["ddim_pos_mse"])
                arr_rot, _ = _safe_arr(m_src0["ddim_rot6d_mse"], metrics["ddim_rot6d_mse"], m_full["ddim_rot6d_mse"])
                arr_grip, _ = _safe_arr(m_src0["ddim_grip_mse"], metrics["ddim_grip_mse"], m_full["ddim_grip_mse"])
                return {
                    "alignment": align_name,
                    "split": split,
                    "condition": cond,
                    "rank_k": int(k),
                    "latent_mse_raw": metrics["latent_mse_raw"],
                    "latent_lz": metrics["latent_lz"],
                    "latent_cos": metrics["latent_cos"],
                    "ddim_10d_mse": metrics["ddim_10d_mse"],
                    "ddim_10d_mse_macro": metrics["ddim_10d_mse_macro"],
                    "ddim_pos_mse": metrics["ddim_pos_mse"],
                    "ddim_rot6d_mse": metrics["ddim_rot6d_mse"],
                    "ddim_grip_mse": metrics["ddim_grip_mse"],
                    "ddim_7d_mae": metrics["ddim_7d_mae"],
                    "exec12_7d_mae": metrics["exec12_7d_mae"],
                    "grip_acc16": metrics["grip_acc16"],
                    "arr_vs_source0": arr_vs_src,
                    "arr_vs_source0_valid": ok_src,
                    "arr_vs_self0": arr_self,
                    "arr_pos_vs_source0": arr_pos,
                    "arr_rot6d_vs_source0": arr_rot,
                    "arr_grip_vs_source0": arr_grip,
                }

            split_rows.append(_make_row("correct", 0, m_src0, m_src0["ddim_10d_mse"]))
            split_rows.append(_make_row("wrong_task", 0, m_wrg0, m_wrg0["ddim_10d_mse"]))
            split_rows.append(_make_row("mean", 0, m_mn0, m_mn0["ddim_10d_mse"]))
            split_rows.append(_make_row("behavior_pca", 0, m_mn0, m_mn0["ddim_10d_mse"]))
            split_rows.append(_make_row("full_self", LATENT_DIM, m_full, base_mse0))

            for k in P05_RANKS:
                Uk = U_mat[:, :k]
                # P @ z_query is fixed across Correct / Wrong-task / Mean!
                pz_q = (zq @ Uk) @ Uk.T
                z_corr = (zs - (zs @ Uk) @ Uk.T) + pz_q
                z_wrg = (zw - (zw @ Uk) @ Uk.T) + pz_q
                z_mn = (zm - (zm @ Uk) @ Uk.T) + pz_q

                m_c = _decode_and_score_latents(
                    policy, z_corr, zq, zs, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
                    sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
                )
                m_w = _decode_and_score_latents(
                    policy, z_wrg, zq, zw, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
                    sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
                )
                m_m = _decode_and_score_latents(
                    policy, z_mn, zq, zm, sb["gt_chunks"], sb["valid_16"], sb["init_noise_cpu"],
                    sb["tasks"], s_z=s_z, device=device, use_bf16=use_bf16, batch_size=batch_size,
                )
                m_p = shared_pca[(split, k)]

                split_rows.append(_make_row("correct", k, m_c, m_src0["ddim_10d_mse"]))
                split_rows.append(_make_row("wrong_task", k, m_w, m_wrg0["ddim_10d_mse"]))
                split_rows.append(_make_row("mean", k, m_m, m_mn0["ddim_10d_mse"]))
                split_rows.append(_make_row("behavior_pca", k, m_p, m_mn0["ddim_10d_mse"]))

                print(
                    f"  [P0.5 {align_name:13s} {split:12s} K={k:2d}] "
                    f"Correct={m_c['ddim_10d_mse']:.5f} (ARR={100*_safe_arr(base_mse0, m_c['ddim_10d_mse'], full_mse)[0]:.1f}%) | "
                    f"Wrong={m_w['ddim_10d_mse']:.5f} ({m_w['ddim_10d_mse']/m_c['ddim_10d_mse']:.2f}x) | "
                    f"Mean={m_m['ddim_10d_mse']:.5f} ({m_m['ddim_10d_mse']/m_c['ddim_10d_mse']:.2f}x) | "
                    f"BehPCA={m_p['ddim_10d_mse']:.5f} ({m_p['ddim_10d_mse']/m_c['ddim_10d_mse']:.2f}x) | "
                    f"Full={full_mse:.5f}",
                    flush=True,
                )

            results_nested[align_name][split] = split_rows
            all_rows.extend(split_rows)

    summary = {
        "experiment": "p0_5_demo_preservation_diagnosis",
        "model": "e9",
        "s_z": s_z,
        "ranks": list(P05_RANKS),
        "bases_export_path": str(bases_path),
        "n_train_basis": len(train_pairs),
        "n_seen_val": len(seen_val_pairs),
        "n_unseen_chain": len(unseen_pairs),
        "ddim_seed": ddim_seed,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "results": results_nested,
    }
    atomic_write_json(summary, out_dir / "p0_5_summary.json")
    _write_csv(all_rows, out_dir / "p0_5_results.csv")
    fig_path = _plot_p05(summary, out_dir)
    print(f"[P0.5 DONE] {summary['elapsed_seconds']}s -> {out_dir} (fig: {fig_path})", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=P05_OUT_DIR)
    parser.add_argument("--bases-out", type=Path, default=BASES_EXPORT_PATH)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_p05(out_dir=args.output, bases_path=args.bases_out, batch_size=args.batch_size, device=device)


if __name__ == "__main__":
    main()
