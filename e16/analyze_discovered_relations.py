"""Deep scientific analysis of E16 discovered relations, reusability, and beyond-E12 structures.

Answers the three core scientific questions of E16 without presupposing any method:
  1. What behavioral relations did each self-supervised method discover?
     - Cross-task alignment with GT attributes (`bal7d/trans/rot/mag/temp/grip`),
       displacement alignment, autonomous token specialization, and anti-degeneracy
       audit (same-task leak, same-object leak, static-scene correlation, erank).
  2. Are the discovered relations genuinely reusable?
     - Directly inspects the learned cross-demo relation graph `W_ij` (for E16-B/C/D4)
       on held-out strictly cross-task clips: measures cross-task donor transfer MSE10,
       cross-task behavioral alignment, and forward visual state-change reuse R^2.
  3. Can self-supervised relations discover behavioral structures BEYOND E12's hand-defined
     16-step action attributes?
     - Projects out E12's full immediate 16-step action space (160D `a10`) to isolate
       `z_beyond`, and probes both `z` and `z_beyond` on:
         * Multi-horizon future actions: `a_{t+16:t+32}` (+0.8s) and `a_{t+32:t+48}` (+1.6s)
         * Normalized episode phase / progress `t / T`
         * Upcoming subgoal / gripper transition anticipation (whether a grasp/release
           happens in `[t+16, t+48]`, invisible in constant-gripper `a_{t:t+16}`!)
         * Manipulation phase classification (Pre-Grasp / Holding-Transport / Post-Release)
  4. Frozen-Encoder Action Label Efficiency (1%, 5%, 10%, 25%, 100% disjoint-task probes)
     for E16-D pure video-only pretrained encoders vs random & E9 oracle!
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.action_attributes import (
    compute_attribute_similarity_matrix,
    extract_action_attribute_vectors,
    extract_action_delta_features,
    pairwise_cosine_sim,
)
from e12.eval_suite import (
    ValClipCache,
    _effective_rank_stats,
    _ridge_fit_predict,
    _spearman_pearson_from_vectors,
    encode_all_val_starts,
)
from e13.exp2_action_baselines import _gather_future_actions

from . import OUTPUT_ROOT, REFERENCE_CHECKPOINTS, SEEDS_DEFAULT
from .pretrain_video_only import load_pretrained_video_lam
from .video_relations import LearnableReuseRelationModule, compute_clip_state_change_targets


def _gather_beyond_e12_targets(cache: ValClipCache) -> Dict[str, Any]:
    """Compute strictly non-immediate behavioral targets beyond E12's 16-step action attributes."""
    N = len(cache.val_episodes)
    # 1. Multi-horizon future actions (+16 steps = +0.8s, +32 steps = +1.6s)
    fut16_a10, fut16_ok = _gather_future_actions(cache, horizon_offset=16)
    fut32_a10, fut32_ok = _gather_future_actions(cache, horizon_offset=32)
    fut16_a7 = action_10d_to_7d(fut16_a10).reshape(N, -1)
    fut32_a7 = action_10d_to_7d(fut32_a10).reshape(N, -1)

    # 2. Normalized episode progress t / T
    progress = np.zeros((N, 1), dtype=np.float32)
    # 3. Upcoming gripper transition in [t+16, t+48] (pre-grasp / pre-release anticipation!)
    upcoming_grip_switch = np.zeros(N, dtype=np.int64)
    # 4. 3-class manipulation phase (0=pre-first-grasp, 1=holding/manipulating, 2=post-release/late)
    manip_phase = np.zeros(N, dtype=np.int64)
    # 5. Object labels for anti-degeneracy audit
    procs = [cache.episodes[int(e)]["procedure"].split("|")[0].strip() for e in cache.val_episodes]
    obj_names = [(p.split()[1] if len(p.split()) >= 2 else "other") for p in procs]
    obj_vocab = {o: i for i, o in enumerate(sorted(set(obj_names)))}
    obj_ids = np.array([obj_vocab[o] for o in obj_names], dtype=np.int64)

    all_a10 = cache.actions_10d_mmap
    is_tail = np.zeros(N, dtype=bool)
    val_a10 = np.zeros((N, 16, 10), dtype=np.float32)
    for i in range(N):
        ep = int(cache.val_episodes[i])
        fr = int(cache.val_frames[i])
        T = max(int(cache.ep_lengths_np[ep]), 1)
        off = int(cache.frame_offsets_np[ep])
        is_tail[i] = (fr + 16) > T
        clamped = np.minimum(fr + np.arange(16), T - 1)
        val_a10[i] = np.asarray(all_a10[off + clamped], dtype=np.float32)
        progress[i, 0] = float(fr) / float(max(T - 1, 1))

        ep_grip = np.asarray(all_a10[off : off + T, 9], dtype=np.float32)
        # Upcoming gripper sign flip in [fr+16, min(fr+48, T))
        lo = min(fr + 16, T - 1)
        hi = min(fr + 48, T)
        cur_sign = 1.0 if ep_grip[min(fr + 15, T - 1)] > 0 else -1.0
        fut_seg = ep_grip[lo:hi]
        if len(fut_seg) > 0 and np.any((fut_seg > 0) != (cur_sign > 0)):
            upcoming_grip_switch[i] = 1

        # First grasp (open -> closed) and last release (closed -> open) in episode
        closed_idx = np.flatnonzero(ep_grip > 0)
        if len(closed_idx) == 0:
            manip_phase[i] = 0 if progress[i, 0] < 0.5 else 2
        else:
            first_c, last_c = int(closed_idx[0]), int(closed_idx[-1])
            if fr < first_c:
                manip_phase[i] = 0  # Pre-grasp approach
            elif fr <= last_c:
                manip_phase[i] = 1  # Holding / transport / interaction
            else:
                manip_phase[i] = 2  # Post-release / retract

    val_a7 = action_10d_to_7d(val_a10)
    return {
        "fut16_a7": fut16_a7,
        "fut16_ok": fut16_ok,
        "fut32_a7": fut32_a7,
        "fut32_ok": fut32_ok,
        "progress": progress,
        "upcoming_grip_switch": upcoming_grip_switch,
        "manip_phase": manip_phase,
        "obj_ids": obj_ids,
        "is_tail": is_tail,
        "val_a10": val_a10,
        "val_a7": val_a7,
    }


def _encode_policy_or_pretrain(
    ckpt_path: Path, cache: ValClipCache, device: torch.device
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Encode all val starts from either a full Policy checkpoint or a PureVideoLAM checkpoint."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "local_encoder" in payload and "action_decoder.final_conv.0.weight" not in payload.get("model", {}):
        # Pure video-only pretrained checkpoint (Regime 2)
        vmodel = load_pretrained_video_lam(ckpt_path, device=device)
        N = len(cache.val_episodes)
        out = np.zeros((N, 4, 768), dtype=np.float32)
        for s in range(0, N, 512):
            e = min(s + 512, N)
            cr, cm, _, _ = cache.gather(cache.val_episodes[s:e], cache.val_frames[s:e])
            cr = cr.to(device=device, dtype=torch.float32)
            cm = cm.to(device=device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                b = vmodel.encode_local_behavior(cr, cm)
            out[s:e] = b.float().cpu().numpy()
        return out, payload
    else:
        pol = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
        enc = encode_all_val_starts(pol, cache, device, use_bf16=True)
        return enc["latents"], payload


def analyze_single_representation(
    name: str,
    ckpt_path: Path,
    cache: ValClipCache,
    beyond: Dict[str, Any],
    device: torch.device,
    *,
    op_dim: int = 384,
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    lat4, payload = _encode_policy_or_pretrain(ckpt_path, cache, device)
    N_all = len(lat4)
    Z_full = lat4.reshape(N_all, -1)
    Z_op = lat4[:, :, :op_dim].reshape(N_all, -1)

    non_tail = np.flatnonzero(~beyond["is_tail"])
    rng = np.random.default_rng(42 + 1616)
    sel = np.sort(rng.choice(non_tail, size=min(2048, len(non_tail)), replace=False))
    Ns = len(sel)

    # 1. Discovered Relations & Anti-Degeneracy Audit (both full and z_op, plus per-token!)
    cr_sel, cm_sel, a10_sel_c, _ = cache.gather(cache.val_episodes[sel], cache.val_frames[sel])
    a10_s = a10_sel_c.to(device=device, dtype=torch.float32)
    a7_s = action_10d_to_7d(a10_s)
    t_ids = torch.from_numpy(cache.val_task_ids[sel]).to(device=device)
    o_ids = torch.from_numpy(beyond["obj_ids"][sel]).to(device=device)
    v0_s = cr_sel[:, 0].to(device=device, dtype=torch.float32).reshape(Ns, -1)

    attrs = extract_action_attribute_vectors(a10_s, a7_s)
    S_gt = {
        k: compute_attribute_similarity_matrix(attrs, attr_key)
        for k, attr_key in [
            ("bal7d", "global_balanced7d"),
            ("trans", "translation"),
            ("rot", "rotation"),
            ("mag", "magnitude"),
            ("temp", "temporal"),
            ("grip", "gripper"),
        ]
    }
    S_scene = pairwise_cosine_sim(v0_s)
    cross_task = t_ids[:, None] != t_ids[None, :]
    off_diag = ~torch.eye(Ns, device=device, dtype=torch.bool)
    a10_f = a10_s.reshape(Ns, -1)

    rng_p = np.random.default_rng(42 + 1617)
    p1 = rng_p.integers(0, Ns, size=1024)
    p2 = rng_p.integers(0, Ns, size=1024)
    da = extract_action_delta_features(a7_s[p1], a7_s[p2])["delta_balanced7d"]
    da_act = torch.linalg.norm(da, dim=-1) > 0.5
    S_da = pairwise_cosine_sim(da)
    off_m = (~torch.eye(1024, device=device, dtype=torch.bool)) & da_act[:, None] & da_act[None, :]

    def _audit_space(arr: np.ndarray) -> Dict[str, Any]:
        u = torch.from_numpy(arr[sel]).to(device=device, dtype=torch.float32)
        S_u = pairwise_cosine_sim(u)
        ct_rho = {
            k: float(_spearman_pearson_from_vectors(S_u[cross_task], S_a[cross_task], seed=42)[0])
            for k, S_a in S_gt.items()
        }
        du = u[p2] - u[p1]
        S_du = pairwise_cosine_sim(du)
        disp_rho = float(_spearman_pearson_from_vectors(S_du[off_m], S_da[off_m], seed=42)[0])

        S_ct = S_u.masked_fill(~cross_task, -1e9)
        top5 = torch.topk(S_ct, k=5, dim=-1).indices
        top1_mse = float((a10_f - a10_f[top5[:, 0]]).square().mean().item())
        top5_mse = float((a10_f[:, None, :] - a10_f[top5]).square().mean().item())

        S_all = S_u.masked_fill(~off_diag, -1e9)
        knn10 = torch.topk(S_all, k=10, dim=-1).indices
        task_leak = float((t_ids[knn10] == t_ids[:, None]).float().mean().item())
        obj_leak = float((o_ids[knn10] == o_ids[:, None]).float().mean().item())
        scene_rho = float(_spearman_pearson_from_vectors(S_u[off_diag], S_scene[off_diag], seed=42)[0])
        er = _effective_rank_stats(arr[sel], device=device)
        return {
            "ct_rho": ct_rho,
            "disp_rho": disp_rho,
            "ct_top1_mse10": top1_mse,
            "ct_top5_mse10": top5_mse,
            "same_task_knn10_leak": task_leak,
            "same_obj_knn10_leak": obj_leak,
            "scene_sim_rho": scene_rho,
            "erank": float(er["erank_energy"]),
        }

    audit_full = _audit_space(Z_full)
    audit_op = _audit_space(Z_op)

    # Per-token specialization matrix (4 tokens x 5 attributes)
    tok_spec: Dict[str, Dict[str, float]] = {}
    for k in range(4):
        uk = torch.from_numpy(lat4[sel, k, :]).to(device=device, dtype=torch.float32)
        S_uk = pairwise_cosine_sim(uk)
        tok_spec[f"tok{k}"] = {
            ak: float(_spearman_pearson_from_vectors(S_uk[cross_task], S_gt[ak][cross_task], seed=42)[0])
            for ak in ("trans", "rot", "mag", "temp", "grip")
        }

    # 2. Beyond-E12 Behavioral Structure Probing (3 seeds, strictly disjoint tasks!)
    A10_all = beyond["val_a10"].reshape(N_all, -1)
    valid_nt = non_tail[beyond["fut16_ok"][non_tail] & beyond["fut32_ok"][non_tail]]
    v_tasks = cache.val_task_ids[valid_nt]
    unique_t = np.unique(v_tasks)

    # One-hot for 3-class manipulation phase & binary upcoming grip switch
    phase_oh = np.eye(3, dtype=np.float32)[beyond["manip_phase"]]
    up_oh = np.eye(2, dtype=np.float32)[beyond["upcoming_grip_switch"]]

    bey_full_seeds, bey_resid_seeds = [], []
    for seed in seeds:
        rng_s = np.random.default_rng(seed + 1699)
        perm_t = rng_s.permutation(unique_t)
        tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
        tr_i = valid_nt[np.array([int(t) in tr_set for t in v_tasks])]
        te_i = valid_nt[np.array([int(t) not in tr_set for t in v_tasks])]

        # Project out immediate 16-step GT action A10 -> strictly beyond-E12 residual z_beyond!
        z_from_a_tr, al_az = _ridge_fit_predict(A10_all[tr_i], Z_full[tr_i], A10_all[tr_i], device=device)
        z_from_a_te, _ = _ridge_fit_predict(A10_all[tr_i], Z_full[tr_i], A10_all[te_i], alphas=(al_az,), device=device)
        z_bey_tr = Z_full[tr_i] - z_from_a_tr
        z_bey_te = Z_full[te_i] - z_from_a_te

        def _probe_beyond(R_tr: np.ndarray, R_te: np.ndarray) -> Dict[str, float]:
            def _r2(Y_all: np.ndarray) -> float:
                p, _ = _ridge_fit_predict(R_tr, Y_all[tr_i], R_te, device=device)
                v = float(((Y_all[te_i] - Y_all[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
                return float(1.0 - ((p - Y_all[te_i]) ** 2).mean() / max(v, 1e-12))

            p_up, _ = _ridge_fit_predict(R_tr, up_oh[tr_i], R_te, device=device)
            p_ph, _ = _ridge_fit_predict(R_tr, phase_oh[tr_i], R_te, device=device)
            # Balanced accuracy for upcoming grip transition
            pred_up = np.argmax(p_up, axis=-1)
            gt_up = beyond["upcoming_grip_switch"][te_i]
            tpr = float((pred_up[gt_up == 1] == 1).mean()) if (gt_up == 1).any() else 0.0
            tnr = float((pred_up[gt_up == 0] == 0).mean()) if (gt_up == 0).any() else 0.0
            return {
                "fut16_r2": _r2(beyond["fut16_a7"]),
                "fut32_r2": _r2(beyond["fut32_a7"]),
                "progress_r2": _r2(beyond["progress"]),
                "upcoming_grip_bal_acc": 0.5 * (tpr + tnr),
                "manip_phase_acc": float((np.argmax(p_ph, axis=-1) == beyond["manip_phase"][te_i]).mean()),
            }

        bey_full_seeds.append(_probe_beyond(Z_full[tr_i], Z_full[te_i]))
        bey_resid_seeds.append(_probe_beyond(z_bey_tr, z_bey_te))

    def _agg(dlist: List[Dict[str, float]]) -> Dict[str, float]:
        return {
            f"{k}_mean": float(np.mean([d[k] for d in dlist])) for k in dlist[0]
        }

    # 3. Direct Inspection of Learned Cross-Demo Relation Graph W_ij (if checkpoint contains reuse_module!)
    reuse_graph_audit: Optional[Dict[str, Any]] = None
    reuse_sd = None
    cfg = payload.get("config", {}) if payload else {}
    if payload and "reuse_module" in payload:
        reuse_sd = payload["reuse_module"]
    elif payload and "model" in payload and any(k.startswith("reuse_module.") for k in payload["model"]):
        reuse_sd = {
            k[len("reuse_module.") :]: v
            for k, v in payload["model"].items()
            if k.startswith("reuse_module.")
        }
    if reuse_sd is not None:
        is_tok = any(k.startswith("q_projs.") for k in reuse_sd)
        has_mo = "forward_pred.motion_head.weight" in reuse_sd
        z_in_d = int(reuse_sd["forward_pred.z_enc.0.weight"].shape[0])
        rm = LearnableReuseRelationModule(
            z_rel_dim=z_in_d,
            num_tokens=4,
            token_wise=is_tok,
            predict_motion=has_mo,
            coverage_mode="strict_cross_task",
        ).to(device)
        rm.load_state_dict(reuse_sd, strict=True)
        rm.eval()

        # Evaluate strictly cross-task learned graph W_ij on 512 unseen clips
        n_g = min(512, Ns)
        sub_idx = sel[:n_g]
        z_in = (
            torch.from_numpy(lat4[sub_idx, :, :op_dim]).to(device=device, dtype=torch.float32)
            if z_in_d == 4 * op_dim
            else torch.from_numpy(lat4[sub_idx]).to(device=device, dtype=torch.float32)
        )
        cr_g = cr_sel[:n_g].to(device=device, dtype=torch.float32)
        cm_g = cm_sel[:n_g].to(device=device)
        ep_g = torch.from_numpy(cache.val_episodes[sub_idx]).to(device=device)
        tk_g = torch.from_numpy(cache.val_task_ids[sub_idx]).to(device=device)
        a10_g = a10_sel_c[:n_g].to(device=device, dtype=torch.float32).reshape(n_g, -1)

        with torch.inference_mode():
            tgts = compute_clip_state_change_targets(cr_g, cm_g)
            vmask, s_sc = rm._build_donor_mask(tgts["v0"], ep_g, task_ids=tk_g)
            W, z_tilde, _ = rm.compute_relation_weights(z_in, vmask)
            W_eff = W.mean(dim=0) if W.ndim == 3 else W  # (n_g, n_g)
            # Soft-transferred & Top-1 transferred cross-task action MSE10 vs random cross-task donor
            top1_j = torch.argmax(W_eff, dim=-1)
            w_top1_mse = float((a10_g - a10_g[top1_j]).square().mean().item())
            w_soft_act = W_eff @ a10_g
            w_soft_mse = float((a10_g - w_soft_act).square().mean().item())
            # Random cross-task donor baseline
            rand_p = vmask.float() / vmask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            rand_j = torch.multinomial(rand_p, 1).squeeze(-1)
            rand_mse = float((a10_g - a10_g[rand_j]).square().mean().item())
            # Forward state-change reuse R^2 using cross-task substituted z_tilde!
            pred_self, _ = rm.forward_pred(tgts["v0"], z_in.reshape(n_g, -1))
            pred_reuse, _ = rm.forward_pred(tgts["v0"], z_tilde.reshape(n_g, -1))
            dv_var = tgts["delta_v_seg"].square().mean().clamp_min(1e-8)
            r2_self = float(1.0 - (pred_self - tgts["delta_v_seg"]).square().mean() / dv_var)
            r2_reuse = float(1.0 - (pred_reuse - tgts["delta_v_seg"]).square().mean() / dv_var)
            w_ent = float(-(W_eff * W_eff.clamp_min(1e-8).log()).sum(dim=-1).mean().item())
            w_scene = float((W_eff * s_sc).sum(dim=-1).mean().item())
            reuse_graph_audit = {
                "cross_task_top1_donor_mse10": w_top1_mse,
                "cross_task_soft_donor_mse10": w_soft_mse,
                "random_cross_task_donor_mse10": rand_mse,
                "relative_error_reduction_vs_rand": float((rand_mse - w_top1_mse) / max(rand_mse, 1e-8)),
                "fwd_dv_r2_self": r2_self,
                "fwd_dv_r2_cross_task_reuse": r2_reuse,
                "w_entropy": w_ent,
                "w_scene_cos": w_scene,
            }

    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "audit_full": audit_full,
        "audit_op": audit_op,
        "token_specialization": tok_spec,
        "beyond_e12_full": _agg(bey_full_seeds),
        "beyond_e12_residual": _agg(bey_resid_seeds),
        "reuse_graph_audit": reuse_graph_audit,
    }


def run_label_efficiency_probe_sweep(
    encoder_ckpts: Dict[str, Optional[Path]],
    cache: ValClipCache,
    beyond: Dict[str, Any],
    device: torch.device,
    *,
    fractions: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 1.00),
    seeds: Tuple[int, ...] = SEEDS_DEFAULT,
) -> Dict[str, Any]:
    """3-seed disjoint-task frozen-encoder linear action-probe label efficiency across 1%..100%."""
    from e9.stage1_local_model import LocalBehaviorEncoder

    non_tail = np.flatnonzero(~beyond["is_tail"])
    A7_all = beyond["val_a7"].reshape(len(cache.val_episodes), -1)
    v_tasks = cache.val_task_ids[non_tail]
    unique_t = np.unique(v_tasks)

    out: Dict[str, Any] = {}
    for name, ckpt in encoder_ckpts.items():
        if ckpt is None:
            # Untrained random encoder lower bound
            torch.manual_seed(42)
            enc = LocalBehaviorEncoder().to(device).eval()
            N = len(cache.val_episodes)
            Z = np.zeros((N, 3072), dtype=np.float32)
            for s in range(0, N, 512):
                e = min(s + 512, N)
                cr, cm, _, _ = cache.gather(cache.val_episodes[s:e], cache.val_frames[s:e])
                cr = cr.to(device=device, dtype=torch.float32)
                cm = cm.to(device=device)
                with torch.inference_mode():
                    Z[s:e] = enc(cr, cm).reshape(e - s, -1).cpu().numpy()
        else:
            lat4, _ = _encode_policy_or_pretrain(ckpt, cache, device)
            Z = lat4.reshape(len(lat4), -1)

        by_frac: Dict[str, Any] = {}
        for frac in fractions:
            r2s, mses = [], []
            for seed in seeds:
                rng = np.random.default_rng(seed + 1688)
                perm_t = rng.permutation(unique_t)
                tr_set = set(perm_t[: int(0.7 * len(perm_t))].tolist())
                tr_all = non_tail[np.array([int(t) in tr_set for t in v_tasks])]
                te_i = non_tail[np.array([int(t) not in tr_set for t in v_tasks])]
                n_keep = max(25, int(round(len(tr_all) * frac)))
                tr_i = tr_all[rng.choice(len(tr_all), size=n_keep, replace=False)]

                pred, _ = _ridge_fit_predict(Z[tr_i], A7_all[tr_i], Z[te_i], device=device)
                mse = float(((pred - A7_all[te_i]) ** 2).mean())
                var = float(((A7_all[te_i] - A7_all[te_i].mean(axis=0, keepdims=True)) ** 2).mean())
                r2s.append(float(1.0 - mse / max(var, 1e-12)))
                mses.append(mse)
            by_frac[f"{int(round(frac*100))}pct"] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
                "mse7_mean": float(np.mean(mses)),
                "mse7_std": float(np.std(mses)),
            }
        out[name] = by_frac
        print(
            f"[LabelEffProbe:{name:20s}] "
            + " | ".join(f"{k}: R2={v['r2_mean']:.3f}" for k, v in by_frac.items()),
            flush=True,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "discovered_analysis")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = ValClipCache(seed=42)
    beyond = _gather_beyond_e12_targets(cache)

    candidates: Dict[str, Path] = {
        "e9": REFERENCE_CHECKPOINTS["e9"],
        "e12_c": REFERENCE_CHECKPOINTS["e12_c"],
        "e15_c_op_only": REFERENCE_CHECKPOINTS["e15_c_op_only"],
        "e15_d_op_mild_cv": REFERENCE_CHECKPOINTS["e15_d_op_mild_cv"],
    }
    # Auto-discover all trained E16 models present under OUTPUT_ROOT
    for sub in sorted(OUTPUT_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        ck = sub / "last_ema.pt"
        if ck.is_file():
            candidates[sub.name] = ck

    analyses: Dict[str, Any] = {}
    for name, ck in candidates.items():
        if not ck.is_file():
            continue
        res = analyze_single_representation(name, ck, cache, beyond, device)
        analyses[name] = res
        af = res["audit_full"]
        bf = res["beyond_e12_full"]
        br = res["beyond_e12_residual"]
        print(
            f"[Discover:{name:24s}] CT_bal={af['ct_rho']['bal7d']:.3f} Disp={af['disp_rho']:.3f} "
            f"Leak(T/O/S)={af['same_task_knn10_leak']*100:.1f}%/{af['same_obj_knn10_leak']*100:.1f}%/{af['scene_sim_rho']:.3f} "
            f"erank={af['erank']:.1f} | BeyFull(f16/f32/prog/upG)={bf['fut16_r2_mean']:.3f}/{bf['fut32_r2_mean']:.3f}/"
            f"{bf['progress_r2_mean']:.3f}/{bf['upcoming_grip_bal_acc_mean']*100:.1f}% | "
            f"BeyRes(f16/prog)={br['fut16_r2_mean']:.3f}/{br['progress_r2_mean']:.3f}",
            flush=True,
        )

    # Probe label-efficiency sweep on E16-D frozen encoders
    probe_encs: Dict[str, Optional[Path]] = {"e16_d0_rand_enc": None}
    for k in ("e16_d1_lam", "e16_d2_rand_rel", "e16_d3_vid_op", "e16_d4_learn_op", "e16_d4_learn_tok"):
        ck = OUTPUT_ROOT / k / "last_ema.pt"
        if ck.is_file():
            probe_encs[k] = ck
    probe_encs["oracle_e9"] = REFERENCE_CHECKPOINTS["e9"]
    label_eff = run_label_efficiency_probe_sweep(probe_encs, cache, beyond, device)

    atomic_write_json(
        {"analyses": analyses, "label_efficiency_probes": label_eff},
        args.out_dir / "discovered_relations_summary.json",
    )
    print(f"[E16 Analyze] Saved summary to {args.out_dir / 'discovered_relations_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
