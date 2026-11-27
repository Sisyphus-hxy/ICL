"""Offline phase alignment, pair sampling, and alignment quality auditing for P0.

All alignment logic operates strictly on existing trajectory kinematics (EE position,
action velocity, gripper commands/states) and NEVER reads visual latents or models.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def detect_debounced_gripper_flips(
    grip_cmd: np.ndarray, *, min_run_len: int = 6
) -> Tuple[List[int], int]:
    """Detect debounced gripper open/close transition step indices from binary grip_cmd >= 0.

    Returns:
        flips: sorted list of step indices where the debounced gripper state flips.
        init_state: debounced initial binary state (0 = open/neg, 1 = closed/non-neg).
    """
    state = (np.asarray(grip_cmd, dtype=np.float32) >= 0.0).astype(np.int32)
    T = len(state)
    if T == 0:
        return [], 0
    runs: List[List[int]] = []
    cur = int(state[0])
    start = 0
    for t in range(1, T):
        s = int(state[t])
        if s != cur:
            runs.append([start, t, cur])
            start = t
            cur = s
    runs.append([start, T, cur])

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if r[1] - r[0] < min_run_len:
                if i == 0:
                    runs[1][0] = r[0]
                elif i == len(runs) - 1:
                    runs[i - 1][1] = r[1]
                else:
                    runs[i - 1][1] = runs[i + 1][1]
                    runs.pop(i + 1)
                runs.pop(i)
                changed = True
                break

    flips = [int(r[0]) for r in runs[1:]]
    return flips, int(runs[0][2])


def _cumulative_segment_progress(
    ee_pos_seg: np.ndarray, *, arc_weight: float = 0.85
) -> np.ndarray:
    """Compute strictly monotone [0, 1] progress inside a phase segment combining
    EE 3D arc length (translation-invariant across layouts) and normalized time."""
    L = len(ee_pos_seg)
    if L <= 1:
        return np.zeros((max(L, 1),), dtype=np.float64)
    time_p = np.linspace(0.0, 1.0, L, dtype=np.float64)
    diffs = np.diff(ee_pos_seg.astype(np.float64), axis=0)
    step_d = np.linalg.norm(diffs, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(step_d)])
    total = float(cum[-1])
    if total > 1e-6:
        arc_p = cum / total
        prog = arc_weight * arc_p + (1.0 - arc_weight) * time_p
    else:
        prog = time_p
    return prog


def _build_dtw_features(ee_pos: np.ndarray, grip_cmd: np.ndarray) -> np.ndarray:
    """Translation-invariant per-step kinematic features for DTW fallback:
    [arc_progress, time_progress, norm_vel_z, speed_norm, grip_binary]."""
    T = len(ee_pos)
    pos = ee_pos.astype(np.float64)
    vel = np.zeros_like(pos)
    if T > 1:
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]
    speed = np.linalg.norm(vel, axis=-1)
    med_speed = float(np.median(speed[speed > 1e-6])) if np.any(speed > 1e-6) else 1.0
    med_speed = max(med_speed, 1e-5)
    speed_norm = np.clip(speed / med_speed, 0.0, 3.0) / 3.0
    vel_z = np.clip(vel[:, 2] / med_speed, -3.0, 3.0) / 3.0
    arc_p = _cumulative_segment_progress(pos, arc_weight=1.0)
    time_p = np.linspace(0.0, 1.0, T, dtype=np.float64) if T > 1 else np.zeros(1)
    grip_bin = (np.asarray(grip_cmd, dtype=np.float64) >= 0.0).astype(np.float64)
    # Emphasize progress & gripper state so DTW respects global phase and grasp events
    feats = np.stack(
        [
            2.0 * arc_p,
            1.0 * time_p,
            0.6 * vel_z,
            0.4 * speed_norm,
            1.5 * grip_bin,
        ],
        axis=-1,
    )
    return feats


def _dtw_align_map(feats_q: np.ndarray, feats_s: np.ndarray) -> np.ndarray:
    """Standard monotone DTW returning for each query step t in [0, T_q) the
    mean matched source step t_s* in [0, T_s)."""
    Tq = len(feats_q)
    Ts = len(feats_s)
    # Pairwise squared Euclidean cost (Tq, Ts)
    diff = feats_q[:, None, :] - feats_s[None, :, :]
    cost = np.sum(diff * diff, axis=-1)

    dp = np.full((Tq, Ts), np.inf, dtype=np.float64)
    ptr = np.zeros((Tq, Ts), dtype=np.int8)  # 0=diag, 1=up (i-1,j), 2=left (i,j-1)
    dp[0, 0] = cost[0, 0]
    for i in range(1, Tq):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
        ptr[i, 0] = 1
    for j in range(1, Ts):
        dp[0, j] = dp[0, j - 1] + cost[0, j]
        ptr[0, j] = 2
    for i in range(1, Tq):
        ci = cost[i]
        dpi = dp[i]
        dpi_1 = dp[i - 1]
        ptri = ptr[i]
        for j in range(1, Ts):
            a = dpi_1[j - 1]
            b = dpi_1[j]
            c = dpi[j - 1]
            if a <= b and a <= c:
                dpi[j] = ci[j] + a
                ptri[j] = 0
            elif b <= c:
                dpi[j] = ci[j] + b
                ptri[j] = 1
            else:
                dpi[j] = ci[j] + c
                ptri[j] = 2

    # Backtrack warping path
    i, j = Tq - 1, Ts - 1
    mapped_sum = np.zeros(Tq, dtype=np.float64)
    mapped_cnt = np.zeros(Tq, dtype=np.float64)
    while True:
        mapped_sum[i] += j
        mapped_cnt[i] += 1.0
        if i == 0 and j == 0:
            break
        p = ptr[i, j]
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    return mapped_sum / np.maximum(mapped_cnt, 1.0)


def align_demo_pair_offline(
    ee_pos_q: np.ndarray,
    grip_q: np.ndarray,
    ee_pos_s: np.ndarray,
    grip_s: np.ndarray,
    anchors_q: np.ndarray,
    anchors_s: np.ndarray,
    *,
    min_run_len: int = 6,
) -> Dict[str, Any]:
    """Compute offline phase correspondence from query anchors to source anchors.

    Method:
      1. Detect debounced gripper close/open events in both demos.
      2. If both demos share the exact same event count and initial gripper state,
         partition into matched manipulation phases [0, e_1, ..., e_K, T] and match
         each query anchor step t_k by within-phase EE arc-length progress.
      3. Otherwise, fall back to monotone DTW on translation-invariant trajectory
         kinematics (arc-length progress, time progress, vertical velocity, speed,
         binary gripper state).
      4. Snap continuous matched source step t_s* to the nearest stride-16 anchor
         j* in `anchors_s`, and record quantization error |t_s* - anchors_s[j*]|.
    """
    Tq = len(ee_pos_q)
    Ts = len(ee_pos_s)
    flips_q, init_q = detect_debounced_gripper_flips(grip_q, min_run_len=min_run_len)
    flips_s, init_s = detect_debounced_gripper_flips(grip_s, min_run_len=min_run_len)

    exact_phase_match = (len(flips_q) == len(flips_s)) and (init_q == init_s)
    n_flips_q = len(flips_q)
    n_flips_s = len(flips_s)

    if exact_phase_match:
        bounds_q = [0] + flips_q + [Tq]
        bounds_s = [0] + flips_s + [Ts]
        n_phases = len(bounds_q) - 1
        prog_q_segs = [
            _cumulative_segment_progress(ee_pos_q[bounds_q[m] : bounds_q[m + 1]])
            for m in range(n_phases)
        ]
        prog_s_segs = [
            _cumulative_segment_progress(ee_pos_s[bounds_s[m] : bounds_s[m + 1]])
            for m in range(n_phases)
        ]

        t_s_cont = np.empty(len(anchors_q), dtype=np.float64)
        phase_ids = np.empty(len(anchors_q), dtype=np.int32)
        for idx_k, tk in enumerate(anchors_q):
            tk_i = int(tk)
            # Find phase m such that bounds_q[m] <= tk_i < bounds_q[m+1]
            m = n_phases - 1
            for cand in range(n_phases):
                if bounds_q[cand] <= tk_i < bounds_q[cand + 1]:
                    m = cand
                    break
            phase_ids[idx_k] = m
            rel_q = tk_i - bounds_q[m]
            pq = float(prog_q_segs[m][min(rel_q, len(prog_q_segs[m]) - 1)])
            ps = prog_s_segs[m]
            # Interpolate continuous step in source segment m
            if len(ps) <= 1:
                rel_s = 0.0
            else:
                rel_s = float(np.interp(pq, ps, np.arange(len(ps), dtype=np.float64)))
            t_s_cont[idx_k] = float(bounds_s[m]) + rel_s
        method = "piecewise_gripper_arclen" if n_flips_q > 0 else "single_phase_arclen"
    else:
        fq = _build_dtw_features(ee_pos_q, grip_q)
        fs = _build_dtw_features(ee_pos_s, grip_s)
        dtw_map = _dtw_align_map(fq, fs)
        t_s_cont = dtw_map[np.asarray(anchors_q, dtype=np.int64)]
        phase_ids = np.full(len(anchors_q), -1, dtype=np.int32)
        method = "dtw_fallback"

    anchors_s_f = np.asarray(anchors_s, dtype=np.float64)
    diffs = np.abs(t_s_cont[:, None] - anchors_s_f[None, :])
    matched_j = np.argmin(diffs, axis=1).astype(np.int32)
    quant_err_steps = diffs[np.arange(len(anchors_q)), matched_j].astype(np.float64)

    return {
        "method": method,
        "exact_phase_match": bool(exact_phase_match),
        "n_flips_q": int(n_flips_q),
        "n_flips_s": int(n_flips_s),
        "t_s_cont": t_s_cont,
        "matched_j": matched_j,
        "quant_err_steps": quant_err_steps,
        "phase_ids": phase_ids,
    }


def sample_cached_split_pairs(
    episodes: Sequence[Dict[str, Any]],
    mem_lengths: np.ndarray,
    *,
    fold: str,
    target_pairs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample `target_pairs` anchor-level (query_ep, anchor_idx, source_ep) pairs
    strictly within `fold` ('train' or 'val'), requiring same (task, procedure) and
    DIFFERENT layout (`layout_q != layout_s`), stratified across all eligible groups."""
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ep_idx, row in enumerate(episodes):
        if row["fold"] == fold:
            groups[(row["task"], row["procedure"])].append(ep_idx)

    # Filter to groups that have >= 2 distinct layouts
    eligible_groups: List[Tuple[Tuple[str, str], List[int], Dict[int, List[int]]]] = []
    for key in sorted(groups.keys()):
        ids = groups[key]
        by_layout: Dict[int, List[int]] = defaultdict(list)
        for ep_i in ids:
            by_layout[int(episodes[ep_i]["layout"])].append(ep_i)
        if len(by_layout) >= 2:
            eligible_groups.append((key, ids, by_layout))

    if not eligible_groups:
        raise RuntimeError(f"No eligible multi-layout (task, procedure) groups in fold={fold!r}")

    rng = np.random.default_rng(seed)

    # Build per-group pool of all valid query anchors (ep_q, anchor_k)
    group_pools: List[List[Tuple[int, int]]] = []
    for _, ids, _ in eligible_groups:
        pool: List[Tuple[int, int]] = []
        for ep_i in ids:
            m = int(mem_lengths[ep_i])
            for k in range(m):
                pool.append((ep_i, k))
        order = rng.permutation(len(pool))
        group_pools.append([pool[int(i)] for i in order])

    # Round-robin stratified sampling across all groups until target_pairs reached
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

            _, ids, by_layout = eligible_groups[gi]
            q_layout = int(episodes[ep_q]["layout"])
            diff_layout_cands = [
                d for lay, eps_l in by_layout.items() if lay != q_layout for d in eps_l
            ]
            ep_s = int(diff_layout_cands[int(rng.integers(len(diff_layout_cands)))])
            assert episodes[ep_q]["fold"] == fold and episodes[ep_s]["fold"] == fold
            assert episodes[ep_q]["task"] == episodes[ep_s]["task"]
            assert episodes[ep_q]["procedure"] == episodes[ep_s]["procedure"]
            assert int(episodes[ep_q]["layout"]) != int(episodes[ep_s]["layout"])

            selected.append(
                {
                    "split": fold,
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


def sample_unseen_chain_pairs(
    unseen_demos: Sequence[Dict[str, Any]],
    *,
    target_pairs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample `target_pairs` anchor-level pairs across the 10 unseen LIBERO-Gen Chain tasks.
    Each query demo `demo_i` is paired with a different-layout demo `demo_j` (default: shifted
    `j = (i + half) % N_demos`) of the same task, stratified evenly across all 10 tasks."""
    by_task: Dict[str, List[int]] = defaultdict(list)
    for idx, d in enumerate(unseen_demos):
        by_task[d["task"]].append(idx)

    rng = np.random.default_rng(seed)
    tasks = sorted(by_task.keys())
    task_pools: List[List[Tuple[int, int, int]]] = []  # (q_idx, anchor_k, s_idx)

    for task in tasks:
        d_idxs = by_task[task]
        n_d = len(d_idxs)
        if n_d < 2:
            raise RuntimeError(f"Unseen task {task} has < 2 demos")
        half = max(1, n_d // 2)
        pool: List[Tuple[int, int, int]] = []
        for local_i, q_idx in enumerate(d_idxs):
            s_idx = d_idxs[(local_i + half) % n_d]
            assert q_idx != s_idx
            m = int(unseen_demos[q_idx]["mem_len"])
            for k in range(m):
                pool.append((q_idx, k, s_idx))
        order = rng.permutation(len(pool))
        task_pools.append([pool[int(i)] for i in order])

    cursors = [0] * len(tasks)
    selected: List[Dict[str, Any]] = []
    while len(selected) < target_pairs:
        progressed = False
        for ti, task in enumerate(tasks):
            if len(selected) >= target_pairs:
                break
            pool = task_pools[ti]
            if cursors[ti] >= len(pool):
                continue
            q_idx, anchor_k, s_idx = pool[cursors[ti]]
            cursors[ti] += 1
            progressed = True
            selected.append(
                {
                    "split": "unseen_chain",
                    "task": task,
                    "procedure": task,
                    "q_ep": int(q_idx),
                    "s_ep": int(s_idx),
                    "q_layout": int(unseen_demos[q_idx]["demo_idx"]),
                    "s_layout": int(unseen_demos[s_idx]["demo_idx"]),
                    "q_anchor_idx": int(anchor_k),
                }
            )
        if not progressed:
            break

    return selected
