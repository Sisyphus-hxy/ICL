"""Unified Label Efficiency (1% / 10% / 100% Full-5799 + Zero-Leakage Visual ICL) & LAPA-VQ Fairness Check."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from e5.train import atomic_write_json
from e9.stage1_local_model import load_frozen_e9_stage1_teacher
from e12.eval_suite import ValClipCache, _action_metrics_dict, _ridge_fit_predict, encode_all_val_starts
from e14.common import build_block_rms_features, build_seed_bundle, ddim_decode_fixed_noise, fit_predict_rbf_krr, fit_predict_scaled_ridge

from . import OUTPUT_ROOT, SEEDS_DEFAULT
from .models import LAPAVQModel


@torch.inference_mode()
def eval_decoder_full5799_and_icl(
    model_key: str, ckpt_path: Path, cache: ValClipCache, device: torch.device
) -> Dict[str, Any]:
    policy = load_frozen_e9_stage1_teacher(ckpt_path, device=device)
    encoded = encode_all_val_starts(policy, cache, device, use_bf16=True)
    nt = np.flatnonzero(~encoded["is_tail"])
    pred = ddim_decode_fixed_noise(policy, [encoded["latents_flat"][nt]], device, seed=12345, batch_size=512)[0]
    m_full = _action_metrics_dict(pred, encoded["actions_10d"][nt])

    # 3-seed Zero-Leakage Visual ICL (rel_progress Lin@4k & RBF@4k)
    policies = {model_key: policy}
    lin_l, rbf_l = [], []
    for sd in SEEDS_DEFAULT:
        bundle = build_seed_bundle(cache, policies, match_mode="rel_progress", seed=sd, device=device, n_train_pairs=4000)
        obs, acts, lats = bundle["obs"], bundle["acts"], bundle["latents"][model_key]
        X_tr, X_ev = build_block_rms_features(
            [obs["train"]["dh"], obs["train"]["h_q"], lats["train"]["d"]],
            [[obs["test_unseen"]["dh"]], [obs["test_unseen"]["h_q"]], [lats["test_unseen"]["d"]]],
        )
        Y_tr = lats["train"]["q"]
        gt_u = acts["test_unseen"]["q"]
        p_lin = fit_predict_scaled_ridge(X_tr, Y_tr, X_ev, device)[0][0]
        p_rbf = fit_predict_rbf_krr(X_tr, Y_tr, X_ev, device)[0][0]
        a_lin, a_rbf = ddim_decode_fixed_noise(policy, [p_lin, p_rbf], device, seed=12345, batch_size=512)
        lin_l.append(float(_action_metrics_dict(a_lin, gt_u)["action10_mse"]))
        rbf_l.append(float(_action_metrics_dict(a_rbf, gt_u)["action10_mse"]))

    return {
        "ckpt": str(ckpt_path),
        "full5799_mse10": float(m_full["action10_mse"]),
        "full5799_mae7": float(m_full["action7_mae"]),
        "full5799_grip_acc": float(m_full["grip_sign_acc"]),
        "relp_lin4k_mse10": float(np.mean(lin_l)),
        "relp_rbf4k_mse10": float(np.mean(rbf_l)),
    }


@torch.inference_mode()
def eval_linear_probe_label_eff(
    enc_ckpt: Path, cache: ValClipCache, device: torch.device, *, is_lapa_vq: bool = False
) -> Dict[str, Any]:
    """Frozen linear probe R^2(z -> a7) at 1%, 5%, 10%, 25%, 100% labels (3 seeds, disjoint tasks)."""
    from e9.stage1_local_model import LocalBehaviorEncoder
    payload = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
    enc = LocalBehaviorEncoder(
        embed_dim=768, num_layers=6, num_heads=12, ffn_dim=4 * 768,
        max_window_len=16, num_cameras=2, num_behavior_tokens=4, dropout=0.1,
    )
    if "local_encoder" in payload:
        enc.load_state_dict(payload["local_encoder"], strict=True)
    else:
        sd = {k[len("local_encoder."):]: v for k, v in payload["model"].items() if k.startswith("local_encoder.")}
        enc.load_state_dict(sd, strict=True)
    enc.eval().to(device)

    lapa_model = None
    if is_lapa_vq and "model" in payload:
        lapa_model = LAPAVQModel(codebook_size=256, code_dim=128, fdm_hidden=1920).to(device).eval()
        lapa_model.load_state_dict(payload["model"], strict=True)

    N = len(cache.val_episodes)
    Z_pre = np.empty((N, 4 * 768), dtype=np.float32)
    Z_vq = np.empty((N, 4 * 768), dtype=np.float32) if lapa_model is not None else None
    A7 = np.empty((N, 16 * 7), dtype=np.float32)
    from e5.actions import action_10d_to_7d
    for s in range(0, N, 512):
        e = min(s + 512, N)
        cr, cm, a10, _ = cache.gather(cache.val_episodes[s:e], cache.val_frames[s:e])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bt = enc(cr.to(device, dtype=torch.float32), clip_pad_mask=cm.to(device))
        Z_pre[s:e] = bt.float().reshape(e - s, -1).cpu().numpy()
        if lapa_model is not None and Z_vq is not None:
            bq, _, _ = lapa_model.quantize_b(bt.float())
            Z_vq[s:e] = bq.reshape(e - s, -1).cpu().numpy()
        A7[s:e] = action_10d_to_7d(a10.cpu().numpy()).reshape(e - s, -1)

    is_tail = (cache.val_frames + 16) > cache.ep_lengths_np[cache.val_episodes]
    nt = np.flatnonzero(~is_tail)
    unique_t = np.unique(cache.val_task_ids[nt])

    def _probe(Z: np.ndarray) -> Dict[str, float]:
        out = {}
        for frac in (0.01, 0.05, 0.10, 0.25, 1.0):
            r2s = []
            for sd in SEEDS_DEFAULT:
                rng = np.random.default_rng(sd + 1888)
                perm = rng.permutation(unique_t)
                tr_t = set(perm[: int(0.7 * len(perm))].tolist())
                tr_i = nt[np.array([int(t) in tr_t for t in cache.val_task_ids[nt]])]
                te_i = nt[np.array([int(t) not in tr_t for t in cache.val_task_ids[nt]])]
                if frac < 0.999:
                    k = max(16, int(round(frac * len(tr_i))))
                    tr_i = tr_i[rng.choice(len(tr_i), size=min(k, len(tr_i)), replace=False)]
                pred, _ = _ridge_fit_predict(Z[tr_i], A7[tr_i], Z[te_i], device=device)
                v = float(((A7[te_i] - A7[te_i].mean(0, keepdims=True)) ** 2).mean())
                r2s.append(float(1.0 - np.mean((pred - A7[te_i]) ** 2) / max(v, 1e-12)))
            out[f"probe_{int(round(frac*100)):03d}"] = float(np.mean(r2s))
        return out

    res = {"pre_vq_or_z": _probe(Z_pre)}
    if Z_vq is not None:
        res["post_vq_zq"] = _probe(Z_vq)
    return res


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ValClipCache(seed=42)
    models = [
        "p1_a0_compact_lam_s42",
        "p1_a2_urbe_s42",
        "p2_g2_crossview_geo",
        "p2_g3_indep_mot_subspace",
        "p4_ext_lapa_vq_s42",
        "p4_ext_cyclemimic_s42",
    ]
    out: Dict[str, Any] = {}
    for m in models:
        t0 = time.monotonic()
        rec: Dict[str, Any] = {}
        enc_p = OUTPUT_ROOT / m / "last_ema.pt"
        rec["linear_probes"] = eval_linear_probe_label_eff(enc_p, cache, device, is_lapa_vq=("lapa_vq" in m))
        for tag in ("dec01", "dec10", "dec100"):
            ck = OUTPUT_ROOT / f"{m}_{tag}" / "last_ema.pt"
            if ck.is_file():
                rec[tag] = eval_decoder_full5799_and_icl(f"{m}_{tag}", ck, cache, device)
        out[m] = rec
        print(
            f"[E18 LabelEff:{m:25s}] "
            f"1%={rec.get('dec01',{}).get('full5799_mse10',float('nan')):.5f}({rec.get('dec01',{}).get('relp_rbf4k_mse10',float('nan')):.4f}) | "
            f"10%={rec.get('dec10',{}).get('full5799_mse10',float('nan')):.5f}({rec.get('dec10',{}).get('relp_rbf4k_mse10',float('nan')):.4f}) | "
            f"100%={rec.get('dec100',{}).get('full5799_mse10',float('nan')):.5f}({rec.get('dec100',{}).get('relp_rbf4k_mse10',float('nan')):.4f}) "
            f"({time.monotonic()-t0:.1f}s)",
            flush=True,
        )
        atomic_write_json(out, OUTPUT_ROOT / "label_efficiency_summary.json")


if __name__ == "__main__":
    main()
