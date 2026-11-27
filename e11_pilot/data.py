"""Dataset loading, phase labeling, frozen feature/latent extraction, and strict anti-leakage splits for E11-Pilot.

Reuses the exact P0/P0.5/E10 evaluation pairs (`seen_val` 3,000, `unseen_chain` 3,000),
exact DDIM initial noise (`base_seed=12345`), exact wrong-task demo sampling, and
exact E9 behavioral phase labeling (`approach`, `grasp_transition`, `transport_carry`,
`release_transition`, `post_release_transit`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from e10.model import E10LowRankPolicy, load_e10_policy
from e9.reader_diagnosis import label_episode_behavioral_phases
from p0_lowrank.alignment import sample_cached_split_pairs, sample_unseen_chain_pairs
from p0_lowrank.p0_5_demo_preservation import _extract_wrong_latents, _sample_wrong_task_episodes
from p0_lowrank.run_diagnosis import (
    _build_init_noises,
    _compute_alignment_and_gt_for_split,
    _extract_model_latents_for_split,
)

from . import (
    ACTION_HORIZON,
    E10_C_CKPT,
    E5_CACHE_DIR,
    E9_STAGE1_CKPT,
    E9_STAGE2_CACHE_DIR,
    E9_STAGE2_CKPT,
    EMBED_DIM,
    LATENT_DIM,
    LATENT_TOKENS,
    P0_BASES_PATH,
    P0_UNSEEN_CACHE_DIR,
    RANK_K,
    SEMANTIC_PHASES,
)


def load_frozen_e10c_and_basis(
    *,
    device: torch.device,
    e10c_ckpt: Path = E10_C_CKPT,
    basis_path: Path = P0_BASES_PATH,
    stage1_ckpt: Path = E9_STAGE1_CKPT,
    e9_stage2_ckpt: Path = E9_STAGE2_CKPT,
) -> Tuple[E10LowRankPolicy, torch.Tensor, Dict[str, Any]]:
    """Load frozen E10-C policy (containing frozen E9 Encoder/Decoder + frozen E9 Reader + E10-C Transfer)
    and verify exact match with P0 train-fitted `U_64`."""
    policy, payload = load_e10_policy(
        e10c_ckpt,
        stage1_ckpt=stage1_ckpt,
        e9_stage2_ckpt=e9_stage2_ckpt,
        device=device,
    )
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if trainable_cnt != 0:
        raise RuntimeError(f"Expected 0 trainable params in frozen policy, got {trainable_cnt}")

    bases = torch.load(Path(basis_path), map_location="cpu", weights_only=False)
    u64 = bases["U_reader"][:, :RANK_K].detach().float().contiguous()
    gram = u64.T @ u64
    ortho_err = float((gram - torch.eye(RANK_K, dtype=torch.float32)).abs().max().item())
    if ortho_err > 1e-4:
        raise RuntimeError(f"P0 U_64 is not orthonormal: max |U^T U - I| = {ortho_err:.3e}")

    basis_diff = float((u64 - policy.basis_u_k.detach().cpu().float()).abs().max().item())
    if basis_diff > 1e-6:
        raise RuntimeError(f"P0 U_64 differs from E10-C basis_u_k by {basis_diff:.3e}")

    meta = {
        "e9_stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "e9_stage2_ckpt": str(Path(e9_stage2_ckpt).resolve()),
        "e10_c_ckpt": str(Path(e10c_ckpt).resolve()),
        "e10_c_epoch": payload.get("epoch"),
        "basis_path": str(Path(basis_path).resolve()),
        "basis_key": "U_reader[:, :64]",
        "basis_ortho_err": ortho_err,
        "basis_vs_e10c_max_abs_diff": basis_diff,
        "s_z": float(policy.latent_scale_s_z.item()),
    }
    return policy, u64.to(device), meta


def compute_pair_semantic_phases(
    pairs: Sequence[Dict[str, Any]],
    *,
    actions_10d: np.ndarray,
    action_offsets: np.ndarray,
    mem_anchor_steps: np.ndarray,
    mem_offsets: np.ndarray,
    mem_lengths: np.ndarray,
) -> List[str]:
    """Assign each pair's query anchor to one of E9's 5 universal semantic behavioral phases."""
    ep_phase_cache: Dict[int, List[str]] = {}
    out: List[str] = []
    for p in pairs:
        q_ep = int(p["q_ep"])
        k = int(p["q_anchor_idx"])
        if q_ep not in ep_phase_cache:
            a0, a1 = int(action_offsets[q_ep]), int(action_offsets[q_ep + 1])
            m0 = int(mem_offsets[q_ep])
            mlen = int(mem_lengths[q_ep])
            grip_cmd = np.asarray(actions_10d[a0:a1, 9], dtype=np.float32)
            anchors = np.asarray(mem_anchor_steps[m0 : m0 + mlen], dtype=np.int32)
            ph = label_episode_behavioral_phases(
                grip_cmd, anchors, window_len=ACTION_HORIZON, min_run_len=6
            )
            ep_phase_cache[q_ep] = ph["semantic_phases"]
        out.append(ep_phase_cache[q_ep][k])
    return out


def load_eval_splits(
    policy: E10LowRankPolicy,
    *,
    n_seen_val: int = 3000,
    n_unseen: int = 3000,
    seed: int = 42,
    ddim_seed: int = 12345,
    ddim_seed_alt: int = 54321,
    batch_size: int = 1024,
    device: torch.device,
    use_bf16: bool = True,
    include_wrong_demos: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Build exact P0/P0.5/E10 `seen_val` and `unseen_chain` evaluation bundles with:
      - `pairs`, `tasks`, `phases`
      - `z_query`, `z_src_reader`, `h_q`, `c_e10c`
      - `gt_chunks`, `valid_16`, `q_steps`
      - `init_noise_cpu` (ddim_seed=12345) and `init_noise_alt_cpu` (ddim_seed_alt=54321)
      - Optional `z_wrong_reader` for Correct/Wrong Demo sensitivity evaluation
    """
    episodes = json.loads((E5_CACHE_DIR / "episodes.json").read_text())
    actions_10d_e5 = np.load(E5_CACHE_DIR / "actions_10d.npy", mmap_mode="r")
    ee_pos_e5 = np.load(E5_CACHE_DIR / "proprio_11d.npy", mmap_mode="r")[:, :3]
    action_offsets_e5 = np.load(E5_CACHE_DIR / "action_offsets.npy")

    e9_mem_h = np.load(E9_STAGE2_CACHE_DIR / "mem_h.npy", mmap_mode="r")
    e9_mem_b = np.load(E9_STAGE2_CACHE_DIR / "mem_b.npy", mmap_mode="r")
    e9_mem_anchors = np.load(E9_STAGE2_CACHE_DIR / "mem_anchor_steps.npy")
    e9_mem_offsets = np.load(E9_STAGE2_CACHE_DIR / "mem_offsets.npy")
    e9_mem_lengths = np.load(E9_STAGE2_CACHE_DIR / "mem_lengths.npy")

    unseen_meta = json.loads((P0_UNSEEN_CACHE_DIR / "unseen_meta.json").read_text())
    unseen_demos = unseen_meta["demos"]
    u_mem_h = np.load(P0_UNSEEN_CACHE_DIR / "mem_h.npy", mmap_mode="r")
    u_mem_b = np.load(P0_UNSEEN_CACHE_DIR / "mem_b_e9.npy", mmap_mode="r")
    u_mem_anchors = np.load(P0_UNSEEN_CACHE_DIR / "mem_anchor_steps.npy")
    u_mem_offsets = np.load(P0_UNSEEN_CACHE_DIR / "mem_offsets.npy")
    u_mem_lengths = np.load(P0_UNSEEN_CACHE_DIR / "mem_lengths.npy")
    u_actions_10d = np.load(P0_UNSEEN_CACHE_DIR / "actions_10d.npy", mmap_mode="r")
    u_ee_pos = np.load(P0_UNSEEN_CACHE_DIR / "ee_pos.npy", mmap_mode="r")
    u_action_offsets = np.load(P0_UNSEEN_CACHE_DIR / "action_offsets.npy")

    seen_pairs = sample_cached_split_pairs(
        episodes, e9_mem_lengths, fold="val", target_pairs=n_seen_val, seed=seed + 1001
    )
    unseen_pairs = sample_unseen_chain_pairs(
        unseen_demos, target_pairs=n_unseen, seed=seed + 2002
    )

    specs = [
        (
            "seen_val",
            seen_pairs,
            episodes,
            "val",
            actions_10d_e5,
            ee_pos_e5,
            action_offsets_e5,
            e9_mem_h,
            e9_mem_b,
            e9_mem_anchors,
            e9_mem_offsets,
            e9_mem_lengths,
            seed + 20,
            seed + 7001,
        ),
        (
            "unseen_chain",
            unseen_pairs,
            unseen_demos,
            None,
            u_actions_10d,
            u_ee_pos,
            u_action_offsets,
            u_mem_h,
            u_mem_b,
            u_mem_anchors,
            u_mem_offsets,
            u_mem_lengths,
            seed + 30,
            seed + 7002,
        ),
    ]

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    bundles: Dict[str, Dict[str, Any]] = {}
    for (
        sname,
        pairs,
        demos_meta,
        fold_filter,
        a10,
        eep,
        aoff,
        mh,
        mb,
        ma,
        mo,
        ml,
        align_seed,
        wrong_seed,
    ) in specs:
        sb = _compute_alignment_and_gt_for_split(
            sname,
            pairs,
            actions_10d=a10,
            ee_pos_all=eep,
            action_offsets=aoff,
            mem_anchor_steps=ma,
            mem_offsets=mo,
            mem_lengths=ml,
            seed=align_seed,
        )
        lat = _extract_model_latents_for_split(
            policy,
            pairs,
            sb,
            mem_h=mh,
            mem_b=mb,
            mem_anchor_steps=ma,
            mem_offsets=mo,
            mem_lengths=ml,
            actions_10d=a10,
            action_offsets=aoff,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
            compute_d4=False,
        )

        N = len(pairs)
        h_q = np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        for i, p in enumerate(pairs):
            q_flat = int(mo[int(p["q_ep"])]) + int(p["q_anchor_idx"])
            h_q[i] = mh[q_flat].astype(np.float32)

        # Compute frozen E10-C coefficients c_e10c (N, 64)
        c_e10c = np.empty((N, RANK_K), dtype=np.float32)
        z_src = lat["z_src_reader"]
        with torch.inference_mode():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                B = end - start
                zc_t = torch.from_numpy(
                    z_src[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)
                ).to(device=device, dtype=torch.float32)
                hq_t = torch.from_numpy(h_q[start:end]).to(device=device, dtype=torch.float32)
                with amp_ctx:
                    c_b = policy.transfer(zc_t, hq_t)
                c_e10c[start:end] = c_b.float().cpu().numpy()

        phases = compute_pair_semantic_phases(
            pairs,
            actions_10d=a10,
            action_offsets=aoff,
            mem_anchor_steps=ma,
            mem_offsets=mo,
            mem_lengths=ml,
        )

        bundle: Dict[str, Any] = {
            "split": sname,
            "pairs": pairs,
            "tasks": [p["task"] for p in pairs],
            "phases": phases,
            "q_steps": sb["q_steps"],
            "gt_chunks": sb["gt_chunks"],
            "valid_16": sb["valid_16"],
            "init_noise_cpu": _build_init_noises(pairs, sb["q_steps"], base_seed=ddim_seed),
            "init_noise_alt_cpu": _build_init_noises(pairs, sb["q_steps"], base_seed=ddim_seed_alt),
            "z_query": lat["z_query"],
            "z_src_reader": z_src,
            "h_q": h_q,
            "c_e10c": c_e10c,
            "reader_audit": lat["reader_audit"],
        }

        if include_wrong_demos:
            w_eps = _sample_wrong_task_episodes(
                pairs, demos_meta, fold_filter=fold_filter, seed=wrong_seed
            )
            _, z_wrg_rdr = _extract_wrong_latents(
                policy,
                pairs,
                w_eps,
                mem_h=mh,
                mem_b=mb,
                mem_offsets=mo,
                mem_lengths=ml,
                device=device,
                use_bf16=use_bf16,
                batch_size=batch_size,
            )
            c_e10c_wrg = np.empty((N, RANK_K), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    B = end - start
                    zw_t = torch.from_numpy(
                        z_wrg_rdr[start:end].reshape(B, LATENT_TOKENS, EMBED_DIM)
                    ).to(device=device, dtype=torch.float32)
                    hq_t = torch.from_numpy(h_q[start:end]).to(device=device, dtype=torch.float32)
                    with amp_ctx:
                        c_bw = policy.transfer(zw_t, hq_t)
                    c_e10c_wrg[start:end] = c_bw.float().cpu().numpy()
            bundle["z_wrong_reader"] = z_wrg_rdr
            bundle["c_e10c_wrong"] = c_e10c_wrg

        bundles[sname] = bundle

    return bundles


def split_train_episodes_for_learnability(
    episodes: Sequence[Dict[str, Any]],
    *,
    val_frac: float = 0.10,
    seed: int = 1101,
) -> Tuple[set[int], set[int]]:
    """Split `fold == 'train'` episodes into strictly episode-disjoint `train_fit` and
    `val_ckpt` sets, stratified per `(task, procedure)` group while keeping >= 2 distinct
    layouts in both subsets wherever possible.

    Guarantees ZERO overlap with `fold == 'val'` (`seen_val`) and `unseen_chain`.
    """
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ep_idx, row in enumerate(episodes):
        if row["fold"] == "train":
            groups[(row["task"], row["procedure"])].append(ep_idx)

    train_fit_eps: set[int] = set()
    val_ckpt_eps: set[int] = set()

    for key in sorted(groups.keys()):
        ids = list(groups[key])
        by_layout: Dict[int, List[int]] = defaultdict(list)
        for ep_i in ids:
            by_layout[int(episodes[ep_i]["layout"])].append(ep_i)

        if len(by_layout) < 4 or len(ids) < 6:
            train_fit_eps.update(ids)
            continue

        # Hold out ~val_frac of distinct layouts (at least 2 layouts for cross-layout val pairs,
        # while guaranteeing train_fit retains at least 2 distinct layouts).
        layouts = sorted(by_layout.keys())
        rng.shuffle(layouts)
        n_val_lay = max(2, int(round(len(layouts) * val_frac)))
        n_val_lay = min(n_val_lay, len(layouts) - 2)
        val_layouts = set(layouts[:n_val_lay])
        for lay, eps_l in by_layout.items():
            if lay in val_layouts:
                val_ckpt_eps.update(eps_l)
            else:
                train_fit_eps.update(eps_l)

    assert train_fit_eps.isdisjoint(val_ckpt_eps), "Episode overlap between train_fit and val_ckpt!"
    val_fold_eps = {i for i, e in enumerate(episodes) if e["fold"] == "val"}
    assert train_fit_eps.isdisjoint(val_fold_eps), "train_fit leaked into fold=val!"
    assert val_ckpt_eps.isdisjoint(val_fold_eps), "val_ckpt leaked into fold=val!"
    return train_fit_eps, val_ckpt_eps


def sample_pairs_from_episode_subset(
    episodes: Sequence[Dict[str, Any]],
    mem_lengths: np.ndarray,
    allowed_episodes: set[int],
    *,
    split_name: str,
    target_pairs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample stratified cross-layout Demo-Query anchor pairs strictly within `allowed_episodes`."""
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ep_idx in sorted(allowed_episodes):
        row = episodes[ep_idx]
        groups[(row["task"], row["procedure"])].append(ep_idx)

    eligible_groups: List[Tuple[Tuple[str, str], List[int], Dict[int, List[int]]]] = []
    for key in sorted(groups.keys()):
        ids = groups[key]
        by_layout: Dict[int, List[int]] = defaultdict(list)
        for ep_i in ids:
            by_layout[int(episodes[ep_i]["layout"])].append(ep_i)
        if len(by_layout) >= 2:
            eligible_groups.append((key, ids, by_layout))

    if not eligible_groups:
        raise RuntimeError(f"No eligible multi-layout groups in episode subset {split_name!r}")

    rng = np.random.default_rng(seed)
    group_pools: List[List[Tuple[int, int]]] = []
    for _, ids, _ in eligible_groups:
        pool: List[Tuple[int, int]] = []
        for ep_i in ids:
            m = int(mem_lengths[ep_i])
            for k in range(m):
                pool.append((ep_i, k))
        order = rng.permutation(len(pool))
        group_pools.append([pool[int(i)] for i in order])

    cursors = [0] * len(eligible_groups)
    selected: List[Dict[str, Any]] = []
    group_order = list(range(len(eligible_groups)))
    while len(selected) < target_pairs:
        rng.shuffle(group_order)
        progressed = False
        for gi in group_order:
            if len(selected) >= target_pairs:
                break
            pool = group_pools[gi]
            if cursors[gi] >= len(pool):
                continue
            ep_q, anchor_k = pool[cursors[gi]]
            cursors[gi] += 1
            progressed = True

            _, _, by_layout = eligible_groups[gi]
            q_layout = int(episodes[ep_q]["layout"])
            diff_layout_cands = [
                d for lay, eps_l in by_layout.items() if lay != q_layout for d in eps_l
            ]
            ep_s = int(diff_layout_cands[int(rng.integers(len(diff_layout_cands)))])
            assert ep_q in allowed_episodes and ep_s in allowed_episodes
            assert int(episodes[ep_q]["layout"]) != int(episodes[ep_s]["layout"])
            selected.append(
                {
                    "split": split_name,
                    "task": episodes[ep_q]["task"],
                    "procedure": episodes[ep_q]["procedure"],
                    "q_ep": int(ep_q),
                    "s_ep": int(ep_s),
                    "q_layout": q_layout,
                    "s_layout": int(episodes[ep_s]["layout"]),
                    "q_anchor_idx": int(anchor_k),
                }
            )
        if not progressed:
            break
    return selected


def build_learnability_train_val_bundles(
    policy: E10LowRankPolicy,
    *,
    n_train_fit: int = 24000,
    n_val_ckpt: int = 2000,
    seed: int = 1101,
    ddim_seed: int = 12345,
    batch_size: int = 1024,
    device: torch.device,
    use_bf16: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build strictly episode-disjoint `train_fit` and `val_ckpt` bundles from `fold == 'train'`."""
    episodes = json.loads((E5_CACHE_DIR / "episodes.json").read_text())
    actions_10d_e5 = np.load(E5_CACHE_DIR / "actions_10d.npy", mmap_mode="r")
    ee_pos_e5 = np.load(E5_CACHE_DIR / "proprio_11d.npy", mmap_mode="r")[:, :3]
    action_offsets_e5 = np.load(E5_CACHE_DIR / "action_offsets.npy")

    e9_mem_h = np.load(E9_STAGE2_CACHE_DIR / "mem_h.npy", mmap_mode="r")
    e9_mem_b = np.load(E9_STAGE2_CACHE_DIR / "mem_b.npy", mmap_mode="r")
    e9_mem_anchors = np.load(E9_STAGE2_CACHE_DIR / "mem_anchor_steps.npy")
    e9_mem_offsets = np.load(E9_STAGE2_CACHE_DIR / "mem_offsets.npy")
    e9_mem_lengths = np.load(E9_STAGE2_CACHE_DIR / "mem_lengths.npy")

    train_fit_eps, val_ckpt_eps = split_train_episodes_for_learnability(
        episodes, val_frac=0.10, seed=seed
    )
    train_pairs = sample_pairs_from_episode_subset(
        episodes,
        e9_mem_lengths,
        train_fit_eps,
        split_name="train_fit",
        target_pairs=n_train_fit,
        seed=seed + 101,
    )
    val_pairs = sample_pairs_from_episode_subset(
        episodes,
        e9_mem_lengths,
        val_ckpt_eps,
        split_name="val_ckpt",
        target_pairs=n_val_ckpt,
        seed=seed + 202,
    )

    out_bundles = {}
    for sname, pairs, align_seed in (
        ("train_fit", train_pairs, seed + 301),
        ("val_ckpt", val_pairs, seed + 302),
    ):
        sb = _compute_alignment_and_gt_for_split(
            sname,
            pairs,
            actions_10d=actions_10d_e5,
            ee_pos_all=ee_pos_e5,
            action_offsets=action_offsets_e5,
            mem_anchor_steps=e9_mem_anchors,
            mem_offsets=e9_mem_offsets,
            mem_lengths=e9_mem_lengths,
            seed=align_seed,
        )
        lat = _extract_model_latents_for_split(
            policy,
            pairs,
            sb,
            mem_h=e9_mem_h,
            mem_b=e9_mem_b,
            mem_anchor_steps=e9_mem_anchors,
            mem_offsets=e9_mem_offsets,
            mem_lengths=e9_mem_lengths,
            actions_10d=actions_10d_e5,
            action_offsets=action_offsets_e5,
            device=device,
            use_bf16=use_bf16,
            batch_size=batch_size,
            compute_d4=False,
        )
        N = len(pairs)
        h_q = np.empty((N, LATENT_TOKENS, EMBED_DIM), dtype=np.float32)
        for i, p in enumerate(pairs):
            q_flat = int(e9_mem_offsets[int(p["q_ep"])]) + int(p["q_anchor_idx"])
            h_q[i] = e9_mem_h[q_flat].astype(np.float32)

        bundle: Dict[str, Any] = {
            "split": sname,
            "pairs": pairs,
            "tasks": [p["task"] for p in pairs],
            "q_steps": sb["q_steps"],
            "gt_chunks": sb["gt_chunks"],
            "valid_16": sb["valid_16"],
            "z_query": lat["z_query"],
            "z_src_reader": lat["z_src_reader"],
            "h_q": h_q,
        }
        if sname == "val_ckpt":
            bundle["init_noise_cpu"] = _build_init_noises(pairs, sb["q_steps"], base_seed=ddim_seed)
        out_bundles[sname] = bundle

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
    assert audit["train_vs_val_ckpt_ep_overlap"] == 0
    assert audit["train_vs_seen_val_ep_overlap"] == 0
    assert audit["val_ckpt_vs_seen_val_ep_overlap"] == 0
    return out_bundles["train_fit"], out_bundles["val_ckpt"], audit
