"""E13 Experiment 3: Comprehensive 5-Attribute Factor Composition & Interference Audit.

Extends E12-B's Token-Swap validation across ALL FIVE behavioral attributes:
  - Translation  (Token 0 in E12-B)
  - Rotation     (Token 1 in E12-B)
  - Magnitude    (Token 2 in E12-B)
  - Temporal     (Token 2 in E12-B — explicitly tests Magnitude vs Temporal collision!)
  - Gripper      (Token 3 in E12-B — plus steady vs transition breakdown!)

Produces:
  1. 5 x 5 Causal Editability (on-diagonal Adoption %) & Cross-Attribute Interference
     (off-diagonal Drift %) Matrix for E9, E12-A, E12-B, E12-C (and any extra ckpt).
  2. Multi-Factor Unseen Combination Zero-Shot Token Swaps:
     - Unseen (dir x grip) via Token-3 swap AND via Token-0 swap
     - Unseen (dir x speed) via Token-2 swap
  3. Detailed Failure-Case Audit on E12-B (worst decile cases & phase breakdown).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.action_attributes import (
    discretize_behavior_factors,
    extract_action_attribute_vectors,
)
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    ddim_decode_batched,
    encode_all_val_starts,
)

from . import CHECKPOINTS, OUTPUT_ROOT


ATTR_ORDER = ("translation", "rotation", "magnitude", "temporal", "gripper")
TOKEN_FOR_ATTR = {
    "translation": 0,
    "rotation": 1,
    "magnitude": 2,
    "temporal": 2,
    "gripper": 3,
}


def _attr_pairwise_dist(a10_x: np.ndarray, a10_y: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-sample scalar behavioral distances for all 5 attributes between `(N,16,10)` chunks."""
    ax = extract_action_attribute_vectors(torch.from_numpy(a10_x.astype(np.float32)))
    ay = extract_action_attribute_vectors(torch.from_numpy(a10_y.astype(np.float32)))

    # 1. Translation: 3D delta-pos trajectory RMSE (captures direction + trajectory)
    #    AND direction-only cosine distance (1 - cos)
    t_x, t_y = ax["translation"].numpy(), ay["translation"].numpy()
    d_trans = np.sqrt(((t_x - t_y) ** 2).mean(axis=-1))
    # Direction-only cosine distance (scale-invariant translation direction!)
    t_nx = np.clip(np.linalg.norm(t_x, axis=-1), 1e-6, None)
    t_ny = np.clip(np.linalg.norm(t_y, axis=-1), 1e-6, None)
    d_trans_dir = 1.0 - (t_x * t_y).sum(axis=-1) / (t_nx * t_ny)

    # 2. Rotation: 3D axis-angle trajectory RMSE
    r_x, r_y = ax["rotation"].numpy(), ay["rotation"].numpy()
    d_rot = np.sqrt(((r_x - r_y) ** 2).mean(axis=-1))

    # 3. Magnitude: L1 distance on path length + speed profile RMSE (direction-invariant!)
    m_x, m_y = ax["magnitude"].numpy(), ay["magnitude"].numpy()
    d_mag = np.sqrt(((m_x - m_y) ** 2).mean(axis=-1))

    # 4. Temporal: L2 distance on scale- and direction-invariant temporal envelope
    #    (speed_env [0:16] + speed_diff [16:31] + t_com/early_frac/straightness)
    tmp_x, tmp_y = ax["temporal"].numpy(), ay["temporal"].numpy()
    d_temp = np.sqrt(((tmp_x - tmp_y) ** 2).mean(axis=-1))

    # 5. Gripper: MAE on 16-step gripper sequence (0 = identical, 2 = completely flipped)
    g_x, g_y = ax["gripper"].numpy(), ay["gripper"].numpy()
    d_grip = np.abs(g_x - g_y).mean(axis=-1)

    return {
        "translation": d_trans,
        "translation_dir": d_trans_dir,
        "rotation": d_rot,
        "magnitude": d_mag,
        "temporal": d_temp,
        "gripper": d_grip,
    }


def _select_swap_pairs(
    a10_nt: np.ndarray,
    a7_nt: np.ndarray,
    tasks_nt: np.ndarray,
    *,
    num_pairs: int = 512,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build deterministic strictly cross-task `(base, donor)` pair indices (in non-tail coords)
    where base and donor differ strongly on each target attribute."""
    N = len(a10_nt)
    attrs = extract_action_attribute_vectors(torch.from_numpy(a10_nt), torch.from_numpy(a7_nt))
    pos_path = attrs["pos_path"].numpy()
    rot_path = attrs["rot_path"].numpy()
    grip_reg = attrs["gripper_regime"].numpy()
    trans_v = attrs["translation"].numpy()
    rot_v = attrs["rotation"].numpy()
    tmp_v = attrs["temporal"].numpy()

    # Precompute unit vectors
    t_u = trans_v / np.clip(np.linalg.norm(trans_v, axis=-1, keepdims=True), 1e-6, None)
    r_u = rot_v / np.clip(np.linalg.norm(rot_v, axis=-1, keepdims=True), 1e-6, None)
    tmp_c = tmp_v - tmp_v.mean(axis=0, keepdims=True)
    tmp_u = tmp_c / np.clip(np.linalg.norm(tmp_c, axis=-1, keepdims=True), 1e-6, None)

    rng = np.random.default_rng(seed + 313)
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for attr in ATTR_ORDER:
        bases, donors = [], []
        if attr == "translation":
            # Moving clips with opposite/orthogonal direction (cos < -0.2), similar speed
            pool = np.flatnonzero(pos_path > 3.0)
            for b in rng.permutation(pool):
                cands = pool[(tasks_nt[pool] != tasks_nt[b]) & (np.abs(pos_path[pool] - pos_path[b]) < 2.0)]
                if len(cands) == 0:
                    continue
                cos_b = t_u[cands] @ t_u[b]
                opp = cands[cos_b < -0.2]
                if len(opp) > 0:
                    bases.append(int(b))
                    donors.append(int(rng.choice(opp)))
                if len(bases) >= num_pairs:
                    break

        elif attr == "rotation":
            # Active rotation clips paired with distinct active rotation or opposite rotation
            pool_rot = np.flatnonzero(rot_path > 0.8)
            for b in rng.permutation(pool_rot):
                cands = pool_rot[tasks_nt[pool_rot] != tasks_nt[b]]
                cos_r = r_u[cands] @ r_u[b]
                opp = cands[cos_r < 0.0]
                if len(opp) > 0:
                    bases.append(int(b))
                    donors.append(int(rng.choice(opp)))
                if len(bases) >= num_pairs:
                    break

        elif attr == "magnitude":
            # Fast vs slow clips (high path diff > 5.0), both moving so direction is defined
            slow = np.flatnonzero((pos_path > 0.5) & (pos_path < 3.0))
            fast = np.flatnonzero(pos_path > 8.0)
            for k in range(num_pairs * 2):
                if k % 2 == 0:
                    b = int(rng.choice(slow))
                    cands = fast[tasks_nt[fast] != tasks_nt[b]]
                else:
                    b = int(rng.choice(fast))
                    cands = slow[tasks_nt[slow] != tasks_nt[b]]
                if len(cands) > 0:
                    bases.append(b)
                    donors.append(int(rng.choice(cands)))
                if len(bases) >= num_pairs:
                    break

        elif attr == "temporal":
            # Clips with SIMILAR total magnitude (|Δpath| < 1.2) but VERY DIFFERENT temporal envelope (cos < 0.0)
            pool = np.flatnonzero(pos_path > 2.0)
            for b in rng.permutation(pool):
                cands = pool[(tasks_nt[pool] != tasks_nt[b]) & (np.abs(pos_path[pool] - pos_path[b]) < 1.2)]
                if len(cands) == 0:
                    continue
                cos_t = tmp_u[cands] @ tmp_u[b]
                diff_t = cands[cos_t < -0.1]
                if len(diff_t) > 0:
                    bases.append(int(b))
                    donors.append(int(rng.choice(diff_t)))
                if len(bases) >= num_pairs:
                    break

        elif attr == "gripper":
            # Different gripper regimes (open/closed/closing/opening)
            pool = np.arange(N)
            for b in rng.permutation(pool):
                cands = pool[(tasks_nt[pool] != tasks_nt[b]) & (grip_reg[pool] != grip_reg[b])]
                if len(cands) > 0:
                    bases.append(int(b))
                    donors.append(int(rng.choice(cands)))
                if len(bases) >= num_pairs:
                    break

        pairs[attr] = (np.array(bases, dtype=np.int64), np.array(donors, dtype=np.int64))
    return pairs


def evaluate_model_factor_composition(
    key: str,
    ckpt_path: Path,
    cache: ValClipCache,
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    pol = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    enc = encode_all_val_starts(pol, cache, device)
    nt = np.flatnonzero(~enc["is_tail"])
    a10_nt = enc["actions_10d"][nt]
    a7_nt = enc["actions_7d"][nt]
    z4_nt = enc["latents"][nt]

    matrix: Dict[str, Dict[str, Any]] = {}
    failure_audit: Dict[str, Any] = {}

    for target_attr in ATTR_ORDER:
        tok_k = TOKEN_FOR_ATTR[target_attr]
        ib, idn = pairs[target_attr]
        z_base = z4_nt[ib]
        z_donor = z4_nt[idn]
        z_edit = z_base.copy()
        z_edit[:, tok_k, :] = z_donor[:, tok_k, :]

        # Shared DDIM noise across base, donor, edit for exact causal delta measurement
        a_base = ddim_decode_batched(pol, z_base, device, seed=12345)
        a_donor = ddim_decode_batched(pol, z_donor, device, seed=12345)
        a_edit = ddim_decode_batched(pol, z_edit, device, seed=12345)

        d_bd = _attr_pairwise_dist(a_base, a_donor)
        d_ed = _attr_pairwise_dist(a_edit, a_donor)
        d_eb = _attr_pairwise_dist(a_edit, a_base)

        row_metrics: Dict[str, Any] = {"token_swapped": tok_k, "num_pairs": int(len(ib))}
        off_drifts = []
        for meas_attr in ("translation", "translation_dir", "rotation", "magnitude", "temporal", "gripper"):
            valid = d_bd[meas_attr] > 1e-4
            if not np.any(valid):
                adopt = 0.0
                drift = 0.0
            else:
                adopt_arr = (d_bd[meas_attr][valid] - d_ed[meas_attr][valid]) / d_bd[meas_attr][valid]
                drift_arr = d_eb[meas_attr][valid] / d_bd[meas_attr][valid]
                adopt = float(np.clip(adopt_arr, -1.0, 1.0).mean())
                drift = float(np.clip(drift_arr, 0.0, 2.0).mean())
            row_metrics[meas_attr] = {
                "adoption_ratio": adopt,
                "drift_ratio": drift,
                "raw_dist_base_donor": float(d_bd[meas_attr].mean()),
                "raw_dist_edit_base": float(d_eb[meas_attr].mean()),
                "raw_dist_edit_donor": float(d_ed[meas_attr].mean()),
            }
            if meas_attr in ATTR_ORDER and meas_attr != target_attr:
                off_drifts.append(drift)

        row_metrics["target_adoption"] = row_metrics[target_attr]["adoption_ratio"]
        row_metrics["mean_off_target_drift"] = float(np.mean(off_drifts))
        row_metrics["selectivity_score"] = float(
            row_metrics["target_adoption"] - row_metrics["mean_off_target_drift"]
        )

        # Extra gripper sign flip accuracy when editing gripper
        if target_attr == "gripper":
            g_ed = action_10d_to_7d(a_edit)[:, :, 6] > 0
            g_dn = action_10d_to_7d(a_donor)[:, :, 6] > 0
            g_bs = action_10d_to_7d(a_base)[:, :, 6] > 0
            row_metrics["donor_grip_sign_acc"] = float((g_ed == g_dn).mean())
            row_metrics["base_grip_sign_acc"] = float((g_ed == g_bs).mean())
            # Breakdown by steady-to-steady vs transition donors
            g_dn_raw = a7_nt[idn, :, 6]
            is_trans_donor = ~((g_dn_raw < 0).all(axis=1) | (g_dn_raw > 0).all(axis=1))
            row_metrics["steady_donor_grip_acc"] = float((g_ed[~is_trans_donor] == g_dn[~is_trans_donor]).mean())
            row_metrics["transition_donor_grip_acc"] = float((g_ed[is_trans_donor] == g_dn[is_trans_donor]).mean())

        matrix[target_attr] = row_metrics

        # Failure-case audit (worst 10% adoption samples)
        valid_t = d_bd[target_attr] > 1e-4
        adopt_per = np.where(valid_t, (d_bd[target_attr] - d_ed[target_attr]) / np.maximum(d_bd[target_attr], 1e-6), 0.0)
        q10 = float(np.quantile(adopt_per, 0.10))
        q50 = float(np.quantile(adopt_per, 0.50))
        q90 = float(np.quantile(adopt_per, 0.90))
        fail_frac = float((adopt_per < 0.25).mean())
        failure_audit[target_attr] = {
            "adopt_p10": q10,
            "adopt_median": q50,
            "adopt_p90": q90,
            "failure_rate_adopt_lt_25pct": fail_frac,
        }

    # Unseen Combination Swaps: (dir x grip via Tok3), (dir x grip via Tok0), (dir x speed via Tok2)
    fac = discretize_behavior_factors(a7_nt)
    dirs, grips, speeds = fac["dir_label"], fac["grip_label"], fac["speed_label"]
    combo_dg, combo_ds = fac["combo_dir_grip"], fac["combo_dir_speed"]

    unseen_swaps: Dict[str, Any] = {}
    # 1. Unseen (dir x grip): Tok3 swap (keep dir base, graft grip) & Tok0 swap (keep grip base, graft dir)
    held_dg = ["+x|closed", "-x|open", "+y|open", "-y|closed", "+z|closing", "-z|opening"]
    is_u_dg = np.isin(combo_dg, held_dg)
    seen_dg = np.flatnonzero(~is_u_dg)
    unseen_dg = np.flatnonzero(is_u_dg)
    rng = np.random.default_rng(seed + 404)
    u_dg_te = np.sort(rng.choice(unseen_dg, size=min(512, len(unseen_dg)), replace=False))
    seen_pos = a7_nt[seen_dg, :, :3].reshape(len(seen_dg), -1)
    seen_grip = a7_nt[seen_dg, :, 6]

    z_b_dir = np.zeros((len(u_dg_te), 4, 768), dtype=np.float32)
    z_sw_tok3 = np.zeros_like(z_b_dir)
    z_b_grp = np.zeros_like(z_b_dir)
    z_sw_tok0 = np.zeros_like(z_b_dir)
    for k, ui in enumerate(u_dg_te):
        d_s, g_s = dirs[ui], grips[ui]
        c_dir = np.flatnonzero((dirs[seen_dg] == d_s) & (grips[seen_dg] != g_s))
        c_grp = np.flatnonzero((grips[seen_dg] == g_s) & (dirs[seen_dg] != d_s))
        i_d = seen_dg[c_dir[int(np.argmin(np.linalg.norm(seen_pos[c_dir] - a7_nt[ui, :, :3].reshape(1, -1), axis=-1)))]]
        i_g = seen_dg[c_grp[int(np.argmin(np.linalg.norm(seen_grip[c_grp] - a7_nt[ui, :, 6:7].T, axis=-1)))]]
        z_b_dir[k] = z4_nt[i_d]
        z_sw_tok3[k] = z4_nt[i_d].copy()
        z_sw_tok3[k, 3] = z4_nt[i_g, 3]
        z_b_grp[k] = z4_nt[i_g]
        z_sw_tok0[k] = z4_nt[i_g].copy()
        z_sw_tok0[k, 0] = z4_nt[i_d, 0]

    gt_dg = a10_nt[u_dg_te]
    m_ora_dg = _action_metrics_dict(ddim_decode_batched(pol, z4_nt[u_dg_te], device), gt_dg)
    m_bd = _action_metrics_dict(ddim_decode_batched(pol, z_b_dir, device), gt_dg)
    m_t3 = _action_metrics_dict(ddim_decode_batched(pol, z_sw_tok3, device), gt_dg)
    m_bg = _action_metrics_dict(ddim_decode_batched(pol, z_b_grp, device), gt_dg)
    m_t0 = _action_metrics_dict(ddim_decode_batched(pol, z_sw_tok0, device), gt_dg)

    unseen_swaps["unseen_dir_x_grip_tok3_swap"] = {
        "base_mse10": m_bd["action10_mse"],
        "swap_mse10": m_t3["action10_mse"],
        "base_pos_mse": m_bd["pos_mse"],
        "swap_pos_mse": m_t3["pos_mse"],
        "base_grip_acc": m_bd["grip_sign_acc"],
        "swap_grip_acc": m_t3["grip_sign_acc"],
        "arr_10d": float((m_bd["action10_mse"] - m_t3["action10_mse"]) / max(m_bd["action10_mse"] - m_ora_dg["action10_mse"], 1e-9)),
    }
    unseen_swaps["unseen_dir_x_grip_tok0_swap"] = {
        "base_mse10": m_bg["action10_mse"],
        "swap_mse10": m_t0["action10_mse"],
        "base_pos_mse": m_bg["pos_mse"],
        "swap_pos_mse": m_t0["pos_mse"],
        "base_grip_acc": m_bg["grip_sign_acc"],
        "swap_grip_acc": m_t0["grip_sign_acc"],
        "arr_pos": float((m_bg["pos_mse"] - m_t0["pos_mse"]) / max(m_bg["pos_mse"] - m_ora_dg["pos_mse"], 1e-9)),
    }

    # 2. Unseen (dir x speed) via Token-2 swap
    held_ds = ["+x|fast", "-x|slow", "+y|fast", "-y|slow", "+z|fast", "-z|slow"]
    is_u_ds = np.isin(combo_ds, held_ds)
    seen_ds = np.flatnonzero(~is_u_ds)
    unseen_ds = np.flatnonzero(is_u_ds)
    u_ds_te = np.sort(rng.choice(unseen_ds, size=min(512, len(unseen_ds)), replace=False))
    pos_path_all = np.linalg.norm(a7_nt[:, :, :3], axis=-1).sum(axis=-1)
    seen_pos_u = a7_nt[seen_ds, :, :3].reshape(len(seen_ds), -1)
    seen_pos_u = seen_pos_u / np.clip(np.linalg.norm(seen_pos_u, axis=-1, keepdims=True), 1e-6, None)

    z_b_ds = np.zeros((len(u_ds_te), 4, 768), dtype=np.float32)
    z_sw_tok2 = np.zeros_like(z_b_ds)
    for k, ui in enumerate(u_ds_te):
        d_s, s_s = dirs[ui], speeds[ui]
        c_dir = np.flatnonzero((dirs[seen_ds] == d_s) & (speeds[seen_ds] != s_s))
        c_spd = np.flatnonzero((speeds[seen_ds] == s_s) & (dirs[seen_ds] != d_s))
        u_dir_v = a7_nt[ui, :, :3].reshape(-1)
        u_dir_v = u_dir_v / max(float(np.linalg.norm(u_dir_v)), 1e-6)
        i_d = seen_ds[c_dir[int(np.argmax(seen_pos_u[c_dir] @ u_dir_v))]]
        i_s = seen_ds[c_spd[int(np.argmin(np.abs(pos_path_all[seen_ds[c_spd]] - pos_path_all[ui])))]]
        z_b_ds[k] = z4_nt[i_d]
        z_sw_tok2[k] = z4_nt[i_d].copy()
        z_sw_tok2[k, 2] = z4_nt[i_s, 2]

    gt_ds = a10_nt[u_ds_te]
    p_base_ds = ddim_decode_batched(pol, z_b_ds, device)
    p_sw_ds = ddim_decode_batched(pol, z_sw_tok2, device)
    p_ora_ds = ddim_decode_batched(pol, z4_nt[u_ds_te], device)
    gt_path = np.linalg.norm(action_10d_to_7d(gt_ds)[:, :, :3], axis=-1).sum(axis=-1)
    bs_path = np.linalg.norm(action_10d_to_7d(p_base_ds)[:, :, :3], axis=-1).sum(axis=-1)
    sw_path = np.linalg.norm(action_10d_to_7d(p_sw_ds)[:, :, :3], axis=-1).sum(axis=-1)
    or_path = np.linalg.norm(action_10d_to_7d(p_ora_ds)[:, :, :3], axis=-1).sum(axis=-1)
    base_mag_mae = float(np.abs(bs_path - gt_path).mean())
    swap_mag_mae = float(np.abs(sw_path - gt_path).mean())
    ora_mag_mae = float(np.abs(or_path - gt_path).mean())

    unseen_swaps["unseen_dir_x_speed_tok2_swap"] = {
        "base_path_mae": base_mag_mae,
        "swap_path_mae": swap_mag_mae,
        "oracle_path_mae": ora_mag_mae,
        "path_recovery_ratio": float((base_mag_mae - swap_mag_mae) / max(base_mag_mae - ora_mag_mae, 1e-9)),
        "base_pos_mse": _action_metrics_dict(p_base_ds, gt_ds)["pos_mse"],
        "swap_pos_mse": _action_metrics_dict(p_sw_ds, gt_ds)["pos_mse"],
    }

    return {
        "model": key,
        "checkpoint": str(ckpt_path),
        "matrix_5x5": matrix,
        "failure_audit": failure_audit,
        "unseen_combo_swaps": unseen_swaps,
    }


def run_exp3(
    out_dir: Path,
    extra_models: Optional[Dict[str, Path]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = ValClipCache(seed=seed)
    # Build deterministic pairs strictly from GT val actions (identical across all models!)
    pol_ref = load_frozen_e9_stage1_teacher(CHECKPOINTS["e9"], device=device)
    enc_ref = encode_all_val_starts(pol_ref, cache, device)
    nt = np.flatnonzero(~enc_ref["is_tail"])
    pairs = _select_swap_pairs(
        enc_ref["actions_10d"][nt], enc_ref["actions_7d"][nt], cache.val_task_ids[nt], num_pairs=512, seed=seed
    )

    model_map = {k: CHECKPOINTS[k] for k in ("e9", "e12_a", "e12_b", "e12_c")}
    if extra_models:
        model_map.update(extra_models)

    results = {}
    for k, ckpt in model_map.items():
        if not ckpt.is_file():
            continue
        results[k] = evaluate_model_factor_composition(k, ckpt, cache, pairs, device, seed=seed)

    summary = {
        "seed": seed,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "models": results,
    }
    atomic_write_json(summary, out_dir / "exp3_summary.json")

    print("\n=== [E13 Exp 3] 5-Attribute Factor Editability & Interference ===", flush=True)
    for k, res in results.items():
        print(f"--- {k} ---")
        for attr in ATTR_ORDER:
            r = res["matrix_5x5"][attr]
            drifts = " ".join(f"{a[:3]}:{r[a]['drift_ratio']*100:.0f}%" for a in ATTR_ORDER if a != attr)
            extra = f" | grip_acc={r['donor_grip_sign_acc']*100:.1f}% (steady={r['steady_donor_grip_acc']*100:.1f}%, trans={r['transition_donor_grip_acc']*100:.1f}%)" if attr == "gripper" else ""
            print(
                f"  Edit {attr:11s} (tok{r['token_swapped']}): Adopt={r['target_adoption']*100:5.1f}% | "
                f"MeanOffDrift={r['mean_off_target_drift']*100:4.1f}% | Sel={r['selectivity_score']*100:+5.1f}% | "
                f"Off[{drifts}]{extra}"
            )
        us = res["unseen_combo_swaps"]
        print(
            f"  Unseen Swaps: (d*g)Tok3 ARR={us['unseen_dir_x_grip_tok3_swap']['arr_10d']*100:.1f}% | "
            f"(d*g)Tok0 PosARR={us['unseen_dir_x_grip_tok0_swap']['arr_pos']*100:.1f}% | "
            f"(d*s)Tok2 PathRec={us['unseen_dir_x_speed_tok2_swap']['path_recovery_ratio']*100:.1f}%"
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp3_factor_composition")
    args = p.parse_args()
    run_exp3(args.out_dir)


if __name__ == "__main__":
    main()
