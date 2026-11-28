"""E17 Exp 1: Relation Authenticity & Counterfactual Intervention Audit.

Audits LearnReuse's learned relation graph `W_ij` and conditional forward predictor
`F_psi(v_0, z)` with strict controls to distinguish genuine cross-task donor reuse
from static scene (`v_0`) shortcuts:
  - Vision-Only Ridge & Vision-Only MLP baselines `g(v_0) -> delta_v` (ZERO z!)
  - Zero-Donor `F_psi(v_0, 0)` and Mean-Donor `F_psi(v_0, z_bar)`
  - Random-Donor (5-draw average) & Wrong-Donor (adversarial opposite-action donor)
  - Nearest-Neighbor controls: `v0-NN` (scene shortcut), `MotionFlow-NN` (pure video),
    `z-Cosine-NN` (raw latent without W_ij heads), `Action-Oracle-NN` (GT action upper bound)
  - Learned `W_ij` Soft & Top-1 cross-task donors (both all cross-task AND anti-scene!)
  - Counterfactual Intervention: inject cross-task anti-scene donor `z_j` into scene `v_{0,i}`,
    generate imagined clip `V_cf`, re-encode `z_cf = E(V_cf)`, test latent cycle selectivity,
    cross-scene invariance, AND frozen-decoder action adoption!
  - Graph Topology & Anti-Shortcut Audit: entropy, top-1/top-5 mass, column hubness (Gini),
    partial correlation `rho(W_ij, S_act | S_v0)`, and scene-similarity quartile stratification!
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d
from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.action_attributes import (
    compute_attribute_similarity_matrix,
    extract_action_attribute_vectors,
    pairwise_cosine_sim,
)
from e12.eval_suite import (
    ValClipCache,
    _action_metrics_dict,
    _ridge_fit_predict,
    _spearman_pearson_from_vectors,
    encode_all_val_starts,
)
from e14.common import ddim_decode_fixed_noise
from e16 import MOTION_CACHE_DIR
from e16.video_relations import LearnableReuseRelationModule, compute_clip_state_change_targets


def _encode_any_ckpt(ckpt_path: Path, cache: ValClipCache, device: torch.device) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Encode all val starts from ANY E9/E16/E17 checkpoint cleanly via `LocalBehaviorEncoder`."""
    from e9.stage1_local_model import LocalBehaviorEncoder
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = LocalBehaviorEncoder(
        embed_dim=768, num_layers=6, num_heads=12, ffn_dim=4 * 768,
        max_window_len=16, num_cameras=2, num_behavior_tokens=4, dropout=0.1,
    )
    if "local_encoder" in payload:
        enc.load_state_dict(payload["local_encoder"], strict=True)
    else:
        sd = {k[len("local_encoder.") :]: v for k, v in payload["model"].items() if k.startswith("local_encoder.")}
        enc.load_state_dict(sd, strict=True)
    enc.eval().to(device)
    N = len(cache.val_episodes)
    out = np.zeros((N, 4, 768), dtype=np.float32)
    for s in range(0, N, 512):
        e = min(s + 512, N)
        cr, cm, _, _ = cache.gather(cache.val_episodes[s:e], cache.val_frames[s:e])
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            b = enc(cr.to(device, dtype=torch.float32), clip_pad_mask=cm.to(device))
        out[s:e] = b.float().cpu().numpy()
    return out, payload

from . import E16_DECODER_CKPTS, E16_PRETRAIN_CKPTS, E16_REGIME1_LEARN_CKPTS, OUTPUT_ROOT


def _gini(x: np.ndarray) -> float:
    arr = np.sort(np.maximum(np.asarray(x, dtype=np.float64), 0.0))
    n = len(arr)
    if n == 0 or arr.sum() <= 1e-12:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


def _fit_vision_only_mlp(
    v0_tr: torch.Tensor,
    dv_tr: torch.Tensor,
    v0_te: torch.Tensor,
    dv_te: torch.Tensor,
    device: torch.device,
    *,
    hidden_dim: int = 1024,
    steps: int = 400,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Train a vision-only MLP `g(v_0) -> delta_v` matched in capacity to `scene_enc + trunk`."""
    torch.manual_seed(seed)
    B_tr, D_in = v0_tr.shape
    D_out = dv_tr.shape[1]
    mlp = nn.Sequential(
        nn.LayerNorm(D_in),
        nn.Linear(D_in, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, D_out),
    ).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    mlp.train()
    for _ in range(steps):
        idx = torch.randint(0, B_tr, (min(batch_size, B_tr),), generator=gen, device=device)
        pred = mlp(v0_tr[idx])
        loss = F.mse_loss(pred, dv_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    mlp.eval()
    with torch.inference_mode():
        pred_te = mlp(v0_te)
        dv_var = dv_te.square().mean().clamp_min(1e-8)
        r2 = float(1.0 - (pred_te - dv_te).square().mean().item() / float(dv_var.item()))
    return pred_te.cpu().numpy(), r2


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray, max_pairs: int = 200_000, seed: int = 42) -> float:
    """Partial Spearman correlation rho(x, y | z) via rank residuals."""
    n = len(x)
    if n > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pairs, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        return (r - r.mean()) / max(r.std(), 1e-12)

    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    rx_res = rx - np.dot(rx, rz) / max(np.dot(rz, rz), 1e-12) * rz
    ry_res = ry - np.dot(ry, rz) / max(np.dot(rz, rz), 1e-12) * rz
    denom = math.sqrt(max(float(np.dot(rx_res, rx_res) * np.dot(ry_res, ry_res)), 1e-18))
    return float(np.dot(rx_res, ry_res) / denom)


def _load_reuse_module(payload: Dict[str, Any], device: torch.device) -> Tuple[Optional[LearnableReuseRelationModule], int, int, bool]:
    reuse_sd = None
    if "reuse_module" in payload:
        reuse_sd = payload["reuse_module"]
    elif "model" in payload and any(k.startswith("reuse_module.") for k in payload["model"]):
        reuse_sd = {
            k[len("reuse_module.") :]: v
            for k, v in payload["model"].items()
            if k.startswith("reuse_module.")
        }
    if reuse_sd is None:
        return None, 0, 0, False
    is_tok = any(k.startswith("q_projs.") for k in reuse_sd)
    has_mo = "forward_pred.motion_head.weight" in reuse_sd
    z_pred_d = int(reuse_sd["forward_pred.z_enc.0.weight"].shape[0])
    z_route_d = (
        int(reuse_sd["q_projs.0.0.weight"].shape[0]) * 4
        if is_tok
        else int(reuse_sd["q_proj.0.weight"].shape[0])
    )
    proj_dim = (
        int(reuse_sd["q_projs.0.1.weight"].shape[0]) * 2
        if is_tok
        else int(reuse_sd["q_proj.1.weight"].shape[0])
    )
    hidden_dim = int(reuse_sd["forward_pred.scene_enc.1.weight"].shape[0])
    cfg = payload.get("config", {})
    top_k = cfg.get("top_k_donors", None)
    if z_route_d != z_pred_d:
        from .pretrain_e17 import SharpLearnableReuseModule
        rm = SharpLearnableReuseModule(
            z_rel_dim=z_route_d,
            z_pred_dim=z_pred_d,
            num_tokens=4,
            token_wise=is_tok,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            predict_motion=has_mo,
            coverage_mode="strict_cross_task",
            top_k_donors=top_k,
        ).to(device)
    else:
        rm = LearnableReuseRelationModule(
            z_rel_dim=z_route_d,
            num_tokens=4,
            token_wise=is_tok,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            predict_motion=has_mo,
            coverage_mode="strict_cross_task",
            top_k_donors=top_k,
        ).to(device)
    rm.load_state_dict(reuse_sd, strict=True)
    rm.eval()
    return rm, z_route_d, z_pred_d, has_mo


def _load_motion_for_val(cache: ValClipCache, sel: np.ndarray) -> Optional[np.ndarray]:
    m_path = MOTION_CACHE_DIR / "motion_features.npy"
    if not m_path.is_file():
        return None
    m_mmap = np.load(m_path, mmap_mode="r", allow_pickle=False)
    m_mean = np.load(MOTION_CACHE_DIR / "motion_mean.npy", allow_pickle=False).astype(np.float32)
    m_std = np.load(MOTION_CACHE_DIR / "motion_std.npy", allow_pickle=False).astype(np.float32)
    B = len(sel)
    out = np.zeros((B, 16, 2, 64), dtype=np.float32)
    for i, idx in enumerate(sel):
        ep = int(cache.val_episodes[idx])
        fr = int(cache.val_frames[idx])
        off = int(cache.frame_offsets_np[ep])
        T = int(cache.ep_lengths_np[ep])
        raw = fr + np.arange(16)
        pad = raw >= T
        clamped = np.minimum(raw, T - 1)
        arr = np.asarray(m_mmap[off + clamped], dtype=np.float32)
        arr = np.clip((arr - m_mean[None, :, :]) / np.maximum(m_std[None, :, :], 1e-6), -8.0, 8.0)
        if pad.any():
            arr[pad] = 0.0
        out[i] = arr
    return out


def audit_single_model_authenticity(
    name: str,
    ckpt_path: Path,
    regime_tag: str,
    cache: ValClipCache,
    device: torch.device,
    *,
    decoder_ckpt: Optional[Path] = None,
    op_dim: int = 384,
    n_eval: int = 1024,
    save_viz_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    lat4, payload = _encode_any_ckpt(ckpt_path, cache, device)
    assert payload is not None
    rm, z_route_d, z_in_d, has_mo = _load_reuse_module(payload, device)
    if rm is None:
        raise ValueError(f"Checkpoint {ckpt_path} has no reuse_module!")

    # Load encoder callable for counterfactual re-encoding + optional decoder for action-space verification
    is_pretrain_only = "local_encoder" in payload and "action_decoder.final_conv.0.weight" not in payload.get("model", {})
    policy_for_decode = None
    if not is_pretrain_only:
        policy_for_decode = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
        encoder_fn = policy_for_decode.encode_local_behavior
    else:
        from e9.stage1_local_model import LocalBehaviorEncoder
        enc_only = LocalBehaviorEncoder(
            embed_dim=768, num_layers=6, num_heads=12, ffn_dim=4 * 768,
            max_window_len=16, num_cameras=2, num_behavior_tokens=4, dropout=0.1,
        ).to(device)
        enc_only.load_state_dict(payload["local_encoder"], strict=True)
        enc_only.eval()
        encoder_fn = enc_only
        if decoder_ckpt is not None and decoder_ckpt.is_file():
            policy_for_decode = load_frozen_e9_stage1_teacher(decoder_ckpt, device=device)

    # Select non-tail validation samples: disjoint train split (for Vision-Only baseline) & eval pool (n_eval)
    N_all = len(cache.val_episodes)
    is_tail = np.array(
        [(int(cache.val_frames[i]) + 16) > int(cache.ep_lengths_np[int(cache.val_episodes[i])]) for i in range(N_all)],
        dtype=bool,
    )
    non_tail = np.flatnonzero(~is_tail)
    unique_t = np.unique(cache.val_task_ids[non_tail])
    rng = np.random.default_rng(1701)
    perm_t = rng.permutation(unique_t)
    tr_tasks = set(perm_t[: int(0.65 * len(perm_t))].tolist())
    vo_tr_idx = non_tail[np.array([int(cache.val_task_ids[i]) in tr_tasks for i in non_tail])]
    vo_te_all = non_tail[np.array([int(cache.val_task_ids[i]) not in tr_tasks for i in non_tail])]

    # Use up to n_eval unseen-task samples for the authenticity pool (strictly disjoint from Vision-Only train!)
    sel = np.sort(rng.choice(vo_te_all, size=min(n_eval, len(vo_te_all)), replace=False))
    Ns = len(sel)

    # Gather clips, targets, actions, motion
    cr_sel, cm_sel, a10_sel, _ = cache.gather(cache.val_episodes[sel], cache.val_frames[sel])
    cr_g = cr_sel.to(device=device, dtype=torch.float32)
    cm_g = cm_sel.to(device=device)
    a10_g = a10_sel.to(device=device, dtype=torch.float32)
    a7_g = action_10d_to_7d(a10_g)
    a10_f = a10_g.reshape(Ns, -1)
    ep_g = torch.from_numpy(cache.val_episodes[sel]).to(device=device)
    tk_g = torch.from_numpy(cache.val_task_ids[sel]).to(device=device)

    mo_np = _load_motion_for_val(cache, sel)
    mo_g = torch.from_numpy(mo_np).to(device=device) if mo_np is not None else None
    tgts = compute_clip_state_change_targets(cr_g, cm_g, mo_g)
    v0_g = tgts["v0"]                  # (Ns, 2, 768)
    dv_g = tgts["delta_v_seg"]         # (Ns, 4, 2, 768)
    mo_seg_g = tgts.get("motion_seg")  # (Ns, 4, 2, 64) or None

    # Gather Vision-Only train split (v0, dv) for fair Vision-Only Ridge & MLP baselines
    vo_tr_sub = np.sort(rng.choice(vo_tr_idx, size=min(3000, len(vo_tr_idx)), replace=False))
    v0_tr_list, dv_tr_list = [], []
    for s in range(0, len(vo_tr_sub), 512):
        e = min(s + 512, len(vo_tr_sub))
        cr_b, cm_b, _, _ = cache.gather(cache.val_episodes[vo_tr_sub[s:e]], cache.val_frames[vo_tr_sub[s:e]])
        t_b = compute_clip_state_change_targets(cr_b.to(device), cm_b.to(device))
        v0_tr_list.append(t_b["v0"].reshape(e - s, -1).cpu().numpy())
        dv_tr_list.append(t_b["delta_v_seg"].reshape(e - s, -1).cpu().numpy())
    v0_tr_np = np.concatenate(v0_tr_list, axis=0)
    dv_tr_np = np.concatenate(dv_tr_list, axis=0)
    v0_te_np = v0_g.reshape(Ns, -1).cpu().numpy()
    dv_te_np = dv_g.reshape(Ns, -1).cpu().numpy()

    # 1. Vision-Only Baselines (ZERO z!)
    dv_pred_vo_ridge, _ = _ridge_fit_predict(v0_tr_np, dv_tr_np, v0_te_np, device=device)
    dv_var_np = float(np.mean(dv_te_np ** 2))
    r2_vo_ridge = float(1.0 - np.mean((dv_pred_vo_ridge - dv_te_np) ** 2) / max(dv_var_np, 1e-8))
    _, r2_vo_mlp = _fit_vision_only_mlp(
        torch.from_numpy(v0_tr_np).to(device),
        torch.from_numpy(dv_tr_np).to(device),
        v0_g.reshape(Ns, -1),
        dv_g.reshape(Ns, -1),
        device=device,
    )

    # Prepare z_in (for forward_pred) and z_route (for W_ij routing)
    if z_in_d == 4 * op_dim:
        z_in = torch.from_numpy(lat4[sel, :, :op_dim]).to(device=device, dtype=torch.float32)
        z_slice_mode = f"op_{op_dim}"
    elif z_in_d == 4 * 768:
        z_in = torch.from_numpy(lat4[sel]).to(device=device, dtype=torch.float32)
        z_slice_mode = "full_768"
    else:
        raise ValueError(f"Unexpected z_in_d={z_in_d}")
    if z_route_d == z_in_d:
        z_route = z_in
    elif z_route_d == 4 * (op_dim // 2):
        z_route = torch.from_numpy(lat4[sel, :, op_dim // 2 : op_dim]).to(device=device, dtype=torch.float32)
        z_slice_mode = f"route_halfop_pred_op_{op_dim}"
    else:
        raise ValueError(f"Unexpected z_route_d={z_route_d}")
    z_flat = z_in.reshape(Ns, -1)

    # Precompute similarity matrices on eval pool
    S_scene = pairwise_cosine_sim(v0_g.reshape(Ns, -1))
    S_z = pairwise_cosine_sim(z_flat)
    attrs = extract_action_attribute_vectors(a10_g, a7_g)
    S_act_bal = compute_attribute_similarity_matrix(attrs, "global_balanced7d")
    S_act_trans = compute_attribute_similarity_matrix(attrs, "translation")
    S_act_grip = compute_attribute_similarity_matrix(attrs, "gripper")
    # Pairwise 10D action MSE matrix (Ns, Ns)
    a10_sq = (a10_f ** 2).sum(dim=-1, keepdim=True)
    D_act_mse = (a10_sq + a10_sq.T - 2.0 * (a10_f @ a10_f.T)).clamp_min(0.0) / float(a10_f.shape[1])

    if mo_seg_g is not None:
        u_mo = torch.cat([mo_seg_g.reshape(Ns, -1), tgts["motion_cum"].reshape(Ns, -1)], dim=-1)
        S_motion = pairwise_cosine_sim(u_mo)
    else:
        S_motion = None

    cross_task = tk_g[:, None] != tk_g[None, :]
    med_scene = torch.quantile(S_scene[cross_task], 0.50)
    anti_scene_ct = cross_task & (S_scene <= med_scene)

    def _eval_z_condition(z_cond: torch.Tensor, donor_idx: Optional[torch.Tensor] = None) -> Dict[str, float]:
        with torch.inference_mode():
            pred_dv, pred_mo = rm.forward_pred(v0_g, z_cond.reshape(Ns, -1))
            dv_var = dv_g.square().mean().clamp_min(1e-8)
            dv_mse = float((pred_dv - dv_g).square().mean().item())
            dv_r2 = float(1.0 - dv_mse / float(dv_var.item()))
            out = {"dv_mse": dv_mse, "dv_r2": dv_r2}
            if pred_mo is not None and mo_seg_g is not None:
                mo_var = mo_seg_g.square().mean().clamp_min(1e-8)
                mo_mse = float((pred_mo - mo_seg_g).square().mean().item())
                out["mo_r2"] = float(1.0 - mo_mse / float(mo_var.item()))
            if donor_idx is not None:
                out["donor_act_mse10"] = float((a10_f - a10_f[donor_idx]).square().mean().item())
                out["donor_scene_cos"] = float(S_scene[torch.arange(Ns, device=device), donor_idx].mean().item())
                out["donor_act_bal_cos"] = float(S_act_bal[torch.arange(Ns, device=device), donor_idx].mean().item())
            return out

    # Evaluate across two donor candidate pools: (A) strict_cross_task, (B) anti_scene_cross_task (bottom 50% scene sim!)
    controls_by_pool: Dict[str, Dict[str, Any]] = {}
    saved_W_ct = None
    for pool_name, pool_mask in [("strict_cross_task", cross_task), ("anti_scene_cross_task", anti_scene_ct)]:
        with torch.inference_mode():
            W, _, _ = rm.compute_relation_weights(z_route, pool_mask)
            W_eff = W.mean(dim=0) if W.ndim == 3 else W  # (Ns, Ns)
            z_tilde = W_eff @ z_flat
            if pool_name == "strict_cross_task":
                saved_W_ct = W_eff.cpu().numpy()

            # Learned Top-1 & Soft
            w_top1_idx = torch.argmax(W_eff, dim=-1)
            w_soft_act_mse = float((a10_f - (W_eff @ a10_f)).square().mean().item())

            # Zero-donor & Mean-donor
            z_zero = torch.zeros_like(z_flat)
            z_mean = z_flat.mean(dim=0, keepdim=True).expand_as(z_flat)

            # Random-donor (average over 5 deterministic draws)
            rand_stats = []
            rand_p = pool_mask.float() / pool_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            gen_r = torch.Generator(device=device).manual_seed(1702)
            for _ in range(5):
                rj = torch.multinomial(rand_p, 1, generator=gen_r).squeeze(-1)
                rand_stats.append(_eval_z_condition(z_flat[rj], rj))
            rand_agg = {k: float(np.mean([d[k] for d in rand_stats])) for k in rand_stats[0]}

            # Wrong-donor (adversarial: cross-task clip with LARGEST 10D action MSE!)
            wrong_idx = torch.argmax(D_act_mse.masked_fill(~pool_mask, -1e9), dim=-1)

            # Nearest-Neighbor controls
            v0_nn_idx = torch.argmax(S_scene.masked_fill(~pool_mask, -1e9), dim=-1)
            z_nn_idx = torch.argmax(S_z.masked_fill(~pool_mask, -1e9), dim=-1)
            act_oracle_nn_idx = torch.argmin(D_act_mse.masked_fill(~pool_mask, 1e9), dim=-1)
            flow_nn_res = None
            if S_motion is not None:
                flow_nn_idx = torch.argmax(S_motion.masked_fill(~pool_mask, -1e9), dim=-1)
                flow_nn_res = _eval_z_condition(z_flat[flow_nn_idx], flow_nn_idx)

            res_self = _eval_z_condition(z_flat)
            res_zero = _eval_z_condition(z_zero)
            res_mean = _eval_z_condition(z_mean)
            res_wrong = _eval_z_condition(z_flat[wrong_idx], wrong_idx)
            res_v0_nn = _eval_z_condition(z_flat[v0_nn_idx], v0_nn_idx)
            res_z_nn = _eval_z_condition(z_flat[z_nn_idx], z_nn_idx)
            res_act_nn = _eval_z_condition(z_flat[act_oracle_nn_idx], act_oracle_nn_idx)
            res_w_soft = _eval_z_condition(z_tilde.reshape(Ns, -1))
            res_w_soft["donor_act_mse10"] = w_soft_act_mse
            res_w_soft["donor_scene_cos"] = float((W_eff * S_scene).sum(dim=-1).mean().item())
            res_w_top1 = _eval_z_condition(z_flat[w_top1_idx], w_top1_idx)

            # Compute Donor Utilization Index (DUI): excess R^2 above mean_donor normalized by (self - mean_donor)
            base_r2 = res_mean["dv_r2"]
            span_r2 = max(res_self["dv_r2"] - base_r2, 1e-6)
            for d_entry in (res_w_soft, res_w_top1, res_z_nn, res_v0_nn, res_act_nn, rand_agg, res_wrong):
                d_entry["dv_r2_above_mean_donor"] = float(d_entry["dv_r2"] - base_r2)
                d_entry["donor_utilization_index"] = float((d_entry["dv_r2"] - base_r2) / span_r2)
                d_entry["dv_r2_above_vision_mlp"] = float(d_entry["dv_r2"] - r2_vo_mlp)
            if flow_nn_res is not None:
                flow_nn_res["dv_r2_above_mean_donor"] = float(flow_nn_res["dv_r2"] - base_r2)
                flow_nn_res["donor_utilization_index"] = float((flow_nn_res["dv_r2"] - base_r2) / span_r2)
                flow_nn_res["dv_r2_above_vision_mlp"] = float(flow_nn_res["dv_r2"] - r2_vo_mlp)

            controls_by_pool[pool_name] = {
                "self_oracle": res_self,
                "vision_only_ridge": {"dv_r2": r2_vo_ridge},
                "vision_only_mlp": {"dv_r2": r2_vo_mlp},
                "zero_donor": res_zero,
                "mean_donor": res_mean,
                "random_donor": rand_agg,
                "wrong_donor_opp": res_wrong,
                "nn_scene_v0": res_v0_nn,
                "nn_motion_flow": flow_nn_res,
                "nn_z_cosine": res_z_nn,
                "nn_action_oracle": res_act_nn,
                "learned_W_soft": res_w_soft,
                "learned_W_top1": res_w_top1,
            }

    # 2. Predictor Input Sensitivity Ratio: varying z (fixed v0) vs varying v0 (fixed z)
    with torch.inference_mode():
        perm1 = torch.roll(torch.arange(Ns, device=device), shifts=1)
        perm2 = torch.roll(torch.arange(Ns, device=device), shifts=17)
        p_00, _ = rm.forward_pred(v0_g, z_flat)
        p_01, _ = rm.forward_pred(v0_g, z_flat[perm1])
        p_10, _ = rm.forward_pred(v0_g[perm1], z_flat)
        p_02, _ = rm.forward_pred(v0_g, z_flat[perm2])
        dz_effect = float((p_00 - p_01).reshape(Ns, -1).norm(dim=-1).mean().item())
        dv0_effect = float((p_00 - p_10).reshape(Ns, -1).norm(dim=-1).mean().item())
        base_norm = float(p_00.reshape(Ns, -1).norm(dim=-1).mean().clamp_min(1e-6).item())
        # Cross-scene consistency of predicted delta_v for SAME z across two different scenes!
        dv_same_z_cos = float(F.cosine_similarity(p_00.reshape(Ns, -1), p_10.reshape(Ns, -1), dim=-1).mean().item())
        dv_same_v0_diff_z_cos = float(F.cosine_similarity(p_00.reshape(Ns, -1), p_02.reshape(Ns, -1), dim=-1).mean().item())

    predictor_sensitivity = {
        "z_variation_l2_rel": float(dz_effect / base_norm),
        "v0_variation_l2_rel": float(dv0_effect / base_norm),
        "z_to_v0_sensitivity_ratio": float(dz_effect / max(dv0_effect, 1e-6)),
        "pred_dv_cos_same_z_diff_scene": dv_same_z_cos,
        "pred_dv_cos_same_scene_diff_z": dv_same_v0_diff_z_cos,
    }

    # 3. Counterfactual Intervention & Cycle Recovery (strictly cross-task & anti-scene!)
    n_cf = min(512, Ns)
    with torch.inference_mode():
        # Pick 2 distinct cross-task anti-scene donors: (a) random anti-scene donor, (b) opposite-action anti-scene donor
        cf_results: Dict[str, Any] = {}
        for cf_tag, donor_type in [("anti_scene_random", "rand"), ("anti_scene_opposite_act", "opp")]:
            sub_mask = anti_scene_ct[:n_cf]
            if donor_type == "rand":
                p_cf = sub_mask.float() / sub_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                gen_c = torch.Generator(device=device).manual_seed(1703)
                dj = torch.multinomial(p_cf, 1, generator=gen_c).squeeze(-1)
            else:
                dj = torch.argmax(D_act_mse[:n_cf].masked_fill(~sub_mask, -1e9), dim=-1)

            v0_i = v0_g[:n_cf]
            z_i = z_flat[:n_cf]
            z_j = z_flat[dj]
            dv_cf, _ = rm.forward_pred(v0_i, z_j)
            v_cf = v0_i[:, None, :, :] + dv_cf.repeat_interleave(4, dim=1)  # (n_cf, 16, 2, 768)
            b_cf = encoder_fn(v_cf, None)  # (n_cf, 4, 768)

            # Extract matching z_rel slice from b_cf
            if z_in_d == 4 * op_dim:
                z_cf = b_cf[:, :, :op_dim].float().reshape(n_cf, -1)
            elif z_in_d == 4 * 768:
                z_cf = b_cf.float().reshape(n_cf, -1)
            else:
                d_tok = z_in_d // 4
                z_cf = b_cf[:, :, op_dim - d_tok : op_dim].float().reshape(n_cf, -1)

            cos_to_donor = F.cosine_similarity(z_cf, z_j, dim=-1)
            cos_to_orig = F.cosine_similarity(z_cf, z_i, dim=-1)
            margin = cos_to_donor - cos_to_orig
            win_rate = float((cos_to_donor > cos_to_orig).float().mean().item())

            # Cross-scene invariance: apply SAME donor z_j to a SECOND different scene v0_k!
            k_alt = (torch.arange(n_cf, device=device) + 73) % Ns
            v0_k = v0_g[k_alt]
            dv_cf_k, _ = rm.forward_pred(v0_k, z_j)
            v_cf_k = v0_k[:, None, :, :] + dv_cf_k.repeat_interleave(4, dim=1)
            b_cf_k = encoder_fn(v_cf_k, None)
            if z_in_d == 4 * op_dim:
                z_cf_k = b_cf_k[:, :, :op_dim].float().reshape(n_cf, -1)
            elif z_in_d == 4 * 768:
                z_cf_k = b_cf_k.float().reshape(n_cf, -1)
            else:
                d_tok = z_in_d // 4
                z_cf_k = b_cf_k[:, :, op_dim - d_tok : op_dim].float().reshape(n_cf, -1)
            cross_scene_cf_cos = float(F.cosine_similarity(z_cf, z_cf_k, dim=-1).mean().item())

            entry: Dict[str, Any] = {
                "n_samples": int(n_cf),
                "cos_cf_to_donor": float(cos_to_donor.mean().item()),
                "cos_cf_to_orig_scene": float(cos_to_orig.mean().item()),
                "latent_selectivity_margin": float(margin.mean().item()),
                "latent_donor_win_rate": win_rate,
                "cross_scene_cf_consistency_cos": cross_scene_cf_cos,
            }

            # Action-space counterfactual verification via frozen/policy Decoder!
            if policy_for_decode is not None:
                b_orig_np = lat4[sel[:n_cf]].reshape(n_cf, -1)
                b_donor_np = lat4[sel[dj.cpu().numpy()]].reshape(n_cf, -1)
                b_cf_np = b_cf.float().cpu().numpy().reshape(n_cf, -1)
                a_orig_gt = a10_sel[:n_cf].numpy()
                a_donor_gt = a10_sel[dj.cpu().numpy()].numpy()
                p_orig, p_donor, p_cf = ddim_decode_fixed_noise(
                    policy_for_decode, [b_orig_np, b_donor_np, b_cf_np], device, seed=12345, batch_size=512
                )
                mse_cf_donor = float(np.mean((p_cf - a_donor_gt) ** 2))
                mse_cf_orig = float(np.mean((p_cf - a_orig_gt) ** 2))
                mse_orig_donor = float(np.mean((p_orig - a_donor_gt) ** 2))
                mse_donor_oracle = float(np.mean((p_donor - a_donor_gt) ** 2))
                adopt = (mse_orig_donor - mse_cf_donor) / max(mse_orig_donor - mse_donor_oracle, 1e-8)
                per_cf_d = np.mean((p_cf - a_donor_gt) ** 2, axis=(1, 2))
                per_cf_o = np.mean((p_cf - a_orig_gt) ** 2, axis=(1, 2))
                act_win = float(np.mean(per_cf_d < per_cf_o))
                # Gripper flip adoption on pairs with opposite gripper sign
                g_o = np.sign(a_orig_gt[:, :, 9].mean(axis=1))
                g_d = np.sign(a_donor_gt[:, :, 9].mean(axis=1))
                g_cf = np.sign(p_cf[:, :, 9].mean(axis=1))
                opp_g = (g_o * g_d) < 0
                grip_flip_acc = float(np.mean(g_cf[opp_g] == g_d[opp_g])) if opp_g.any() else float("nan")
                entry["action_counterfactual"] = {
                    "mse10_cf_vs_donor": mse_cf_donor,
                    "mse10_cf_vs_orig": mse_cf_orig,
                    "mse10_orig_vs_donor": mse_orig_donor,
                    "mse10_donor_oracle": mse_donor_oracle,
                    "action_adoption_ratio": float(adopt),
                    "action_donor_win_rate": act_win,
                    "grip_flip_adoption_acc": grip_flip_acc,
                    "n_opp_grip_pairs": int(opp_g.sum()),
                }
            cf_results[cf_tag] = entry

    # 4. Graph Topology, Anti-Hubness, Partial Correlation & Scene-Similarity Quartile Stratification
    assert saved_W_ct is not None
    W_t = torch.from_numpy(saved_W_ct).to(device)
    ct_mask_np = cross_task.cpu().numpy()
    w_ent = float(-(W_t * W_t.clamp_min(1e-8).log()).sum(dim=-1).mean().item())
    w_max = float(W_t.max(dim=-1).values.mean().item())
    w_top5 = float(torch.topk(W_t, k=5, dim=-1).values.sum(dim=-1).mean().item())
    col_sums = saved_W_ct.sum(axis=0)
    col_gini = _gini(col_sums)
    col_eff_n = float(np.exp(-np.sum((col_sums / col_sums.sum()) * np.log(np.maximum(col_sums / col_sums.sum(), 1e-12)))))

    # Partial correlation rho(W_ij, S_act_bal | S_scene) on cross-task entries!
    w_vec = saved_W_ct[ct_mask_np]
    sa_vec = S_act_bal.cpu().numpy()[ct_mask_np]
    st_vec = S_act_trans.cpu().numpy()[ct_mask_np]
    sg_vec = S_act_grip.cpu().numpy()[ct_mask_np]
    sc_vec = S_scene.cpu().numpy()[ct_mask_np]
    raw_rho_w_act = float(_spearman_pearson_from_vectors(W_t[cross_task], S_act_bal[cross_task], seed=42)[0])
    raw_rho_w_scene = float(_spearman_pearson_from_vectors(W_t[cross_task], S_scene[cross_task], seed=42)[0])
    part_rho_act_given_scene = _partial_spearman(w_vec, sa_vec, sc_vec, seed=42)
    part_rho_trans_given_scene = _partial_spearman(w_vec, st_vec, sc_vec, seed=42)
    part_rho_grip_given_scene = _partial_spearman(w_vec, sg_vec, sc_vec, seed=42)

    # Scene-similarity quartile stratification (Q1=most dissimilar 25% scenes ... Q4=most similar 25% scenes!)
    q_edges = torch.quantile(S_scene[cross_task], torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], device=device))
    quartiles: Dict[str, Any] = {}
    for qi in range(4):
        lo, hi = q_edges[qi], q_edges[qi + 1]
        qmask = cross_task & (S_scene >= lo) & (S_scene <= hi if qi == 3 else S_scene < hi)
        empty = ~qmask.any(dim=1)
        if empty.any():
            qmask = torch.where(empty[:, None], cross_task, qmask)
        with torch.inference_mode():
            Wq, _, _ = rm.compute_relation_weights(z_route, qmask)
            Wq_eff = Wq.mean(dim=0) if Wq.ndim == 3 else Wq
            ztq = Wq_eff @ z_flat
            t1_j = torch.argmax(Wq_eff, dim=-1)
            r_top1 = _eval_z_condition(z_flat[t1_j], t1_j)
            r_soft = _eval_z_condition(ztq.reshape(Ns, -1))
            r_soft["donor_act_mse10"] = float((a10_f - (Wq_eff @ a10_f)).square().mean().item())
            # Random within same quartile
            rp = qmask.float() / qmask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            rj = torch.multinomial(rp, 1).squeeze(-1)
            r_rnd = _eval_z_condition(z_flat[rj], rj)
        quartiles[f"Q{qi+1}"] = {
            "scene_cos_range": [float(lo.item()), float(hi.item())],
            "learned_top1_act_mse10": r_top1["donor_act_mse10"],
            "learned_soft_act_mse10": r_soft["donor_act_mse10"],
            "random_act_mse10": r_rnd["donor_act_mse10"],
            "rel_act_err_reduction": float((r_rnd["donor_act_mse10"] - r_top1["donor_act_mse10"]) / max(r_rnd["donor_act_mse10"], 1e-8)),
            "learned_soft_dv_r2": r_soft["dv_r2"],
            "learned_top1_dv_r2": r_top1["dv_r2"],
            "random_dv_r2": r_rnd["dv_r2"],
        }

    # Optional: save compact W_ij visualization artifact (128x128 sorted by action cluster!)
    if save_viz_dir is not None:
        save_viz_dir.mkdir(parents=True, exist_ok=True)
        n_v = min(128, Ns)
        # Order samples by gripper sign then primary translation angle for crisp block-structure visualization
        g_m = a10_sel[:n_v, :, 9].numpy().mean(axis=1)
        pos_m = a10_sel[:n_v, :, :3].numpy().mean(axis=1)
        ang = np.arctan2(pos_m[:, 1], pos_m[:, 0])
        order = np.lexsort((ang, (g_m > 0).astype(np.int32)))
        np.savez_compressed(
            save_viz_dir / f"graph_{name}.npz",
            W=saved_W_ct[:n_v, :n_v][np.ix_(order, order)].astype(np.float32),
            S_act=S_act_bal[:n_v, :n_v].cpu().numpy()[np.ix_(order, order)].astype(np.float32),
            S_scene=S_scene[:n_v, :n_v].cpu().numpy()[np.ix_(order, order)].astype(np.float32),
            col_sums=col_sums.astype(np.float32),
            grip_sign=(g_m[order] > 0).astype(np.int8),
        )

    return {
        "name": name,
        "regime_tag": regime_tag,
        "checkpoint": str(ckpt_path),
        "decoder_ckpt": str(decoder_ckpt) if decoder_ckpt else None,
        "z_slice_mode": z_slice_mode,
        "token_wise": bool(rm.token_wise),
        "predict_motion": bool(has_mo),
        "n_eval": int(Ns),
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "controls_by_pool": controls_by_pool,
        "predictor_sensitivity": predictor_sensitivity,
        "counterfactual_interventions": cf_results,
        "graph_topology": {
            "row_entropy": w_ent,
            "uniform_entropy": float(math.log(max(Ns - 1, 1))),
            "top1_weight_mean": w_max,
            "top5_mass_mean": w_top5,
            "col_hub_gini": col_gini,
            "col_effective_donors": col_eff_n,
            "raw_spearman_W_vs_act_bal": raw_rho_w_act,
            "raw_spearman_W_vs_scene": raw_rho_w_scene,
            "partial_spearman_W_vs_act_given_scene": part_rho_act_given_scene,
            "partial_spearman_W_vs_trans_given_scene": part_rho_trans_given_scene,
            "partial_spearman_W_vs_grip_given_scene": part_rho_grip_given_scene,
        },
        "scene_quartile_stratification": quartiles,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "exp1_authenticity")
    p.add_argument("--extra-ckpts", type=str, nargs="*", default=[], help="Extra name=ckpt[:decoder_ckpt][:regime_tag]")
    p.add_argument("--only-extra", action="store_true")
    p.add_argument("--n-eval", type=int, default=1024)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.out_dir / "viz_data"
    cache = ValClipCache(seed=42)

    targets: List[Tuple[str, Path, Optional[Path], str]] = []
    if not args.only_extra:
        targets.extend([
            ("e16_d4_learn_op", E16_PRETRAIN_CKPTS["e16_d4_learn_op"], E16_DECODER_CKPTS["e16_d4_learnop_dec100"] / "last_ema.pt", "pure_video_pretrain"),
            ("e16_b1_learnable_reuse", E16_REGIME1_LEARN_CKPTS["e16_b1_learnable_reuse"], None, "regime1_pure_video_rel"),
            ("e16_b2_learnable_tok", E16_REGIME1_LEARN_CKPTS["e16_b2_learnable_tok"], None, "regime1_pure_video_rel"),
            ("e16_c2_op_learnable", E16_REGIME1_LEARN_CKPTS["e16_c2_op_learnable"], None, "regime1_pure_video_rel"),
            ("e16_c3_op_learnable_tok", E16_REGIME1_LEARN_CKPTS["e16_c3_op_learnable_tok"], None, "regime1_pure_video_rel"),
            ("e16_cov_unmasked", E16_REGIME1_LEARN_CKPTS["e16_cov_unmasked"], None, "regime1_pure_video_rel"),
            ("e16_cov_strict_ct", E16_REGIME1_LEARN_CKPTS["e16_cov_strict_ct"], None, "regime1_task_meta_assisted"),
            ("e16_cov_sametask", E16_REGIME1_LEARN_CKPTS["e16_cov_sametask"], None, "regime1_task_meta_restricted"),
        ])
    for spec in args.extra_ckpts:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        ck = Path(parts[0])
        dec = Path(parts[1]) if len(parts) > 1 and parts[1] else None
        reg = parts[2] if len(parts) > 2 and parts[2] else "e17_pure_video_pretrain"
        targets.append((name, ck, dec, reg))

    summary_path = args.out_dir / "authenticity_summary.json"
    results: Dict[str, Any] = json.loads(summary_path.read_text()) if (args.only_extra and summary_path.is_file()) else {}

    for name, ck, dec, reg in targets:
        if not ck.is_file():
            print(f"[Exp1] Skip missing {name}: {ck}", flush=True)
            continue
        res = audit_single_model_authenticity(
            name, ck, reg, cache, device, decoder_ckpt=dec, n_eval=args.n_eval, save_viz_dir=viz_dir
        )
        results[name] = res
        ct = res["controls_by_pool"]["strict_cross_task"]
        ac = res["controls_by_pool"]["anti_scene_cross_task"]
        cf = res["counterfactual_interventions"]["anti_scene_opposite_act"]
        gt = res["graph_topology"]
        act_cf_str = ""
        if "action_counterfactual" in cf:
            acf = cf["action_counterfactual"]
            act_cf_str = f" | ActCF adopt={acf['action_adoption_ratio']*100:.1f}% win={acf['action_donor_win_rate']*100:.1f}% gripFlip={acf['grip_flip_adoption_acc']*100:.1f}%"
        print(
            f"[Exp1:{name:24s}] dvR2: Self={ct['self_oracle']['dv_r2']:.3f} VO_MLP={ct['vision_only_mlp']['dv_r2']:.3f} "
            f"Zero={ct['zero_donor']['dv_r2']:.3f} Mean={ct['mean_donor']['dv_r2']:.3f} Rand={ct['random_donor']['dv_r2']:.3f} "
            f"Wrong={ct['wrong_donor_opp']['dv_r2']:.3f} | W_soft={ct['learned_W_soft']['dv_r2']:.3f}(DUI={ct['learned_W_soft']['donor_utilization_index']*100:.1f}%) "
            f"AntiSc_W={ac['learned_W_soft']['dv_r2']:.3f} | PartRho(W,A|S)={gt['partial_spearman_W_vs_act_given_scene']:.3f} "
            f"Gini={gt['col_hub_gini']:.3f} | CF_margin={cf['latent_selectivity_margin']:+.3f}(win={cf['latent_donor_win_rate']*100:.1f}%){act_cf_str}",
            flush=True,
        )
        atomic_write_json(results, summary_path)

    print(f"[Exp1] Saved authenticity audit to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
