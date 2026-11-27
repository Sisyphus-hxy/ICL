"""D1 / D2 / D3 / D4 Offline Decomposition Diagnosis for E9 Stage 2 Direct.

Reuses the EXACT protocol from `e7/diagnose_behavior_space.py`:
  - Default (100 demos / 2,025 anchors), Shifted (100 demos / 2,042 anchors),
    Oracle-GT (331 demos / 7,053 anchors across 9 shared tasks)
  - D1 (Teacher Only)   : exact self latent b_k = E_local(V_{t_k:t_k+16}) -> Frozen UNet
  - D2 (Retrieval Only) : Behavior Memory -> Reader -> z_c                 -> Frozen UNet
  - D3 (Transfer Only)  : exact self latent b_k -> Direct Transfer         -> Frozen UNet
  - D4 (Full Model)     : Behavior Memory -> Reader -> Direct Transfer     -> Frozen UNet
  - Includes `w_self >= 0.9` zero-shift stratification to explicitly test whether
    Direct Transfer still overwrites already-correct behavior latents in the new
    Local Behavior Space!
"""

from __future__ import annotations

import argparse
import csv
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

from . import BEHAVIOR_STRIDE, DEFAULT_STAGE1_LOCAL_CKPT, DEFAULT_STAGE2_DIRECT_DIR
from .stage2_direct_model import load_e9_stage2_direct_policy


@torch.inference_mode()
def run_e9_d1_d4_diagnosis(
    *,
    e9_stage2_ckpt: Path = DEFAULT_STAGE2_DIRECT_DIR / "train/best.pt",
    stage1_ckpt: Path = DEFAULT_STAGE1_LOCAL_CKPT,
    out_dir: Path = DEFAULT_STAGE2_DIRECT_DIR / "d1_d4_diagnosis",
    device: torch.device,
    use_bf16: bool = True,
    base_seed: int = 12345,
    behavior_stride: int = BEHAVIOR_STRIDE,
    demo_types: Tuple[str, ...] = ("default", "shifted", "oracle_gt"),
    max_demos_per_type: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = get_chain_tasks()

    visual = BPPVisualEncoder(ckpt_path=BPP_CKPT, load_weights=True).to(device).eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    policy, payload = load_e9_stage2_direct_policy(
        e9_stage2_ckpt, stage1_ckpt=stage1_ckpt, device=device
    )
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()

    trainable_cnt = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    trainable_cnt += sum(p.numel() for p in visual.parameters() if p.requires_grad)
    if trainable_cnt != 0:
        raise RuntimeError(f"Expected all modules frozen, found {trainable_cnt} trainable params")

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

    print(
        f"[E9 D1-D4 Diag] Starting diagnosis: {len(specs)} demos "
        f"(types={demo_types}, stride={behavior_stride}, s_z={s_z:.6f}, bf16={use_bf16}, "
        f"seed={base_seed}, ckpt={e9_stage2_ckpt} ep={payload.get('epoch')}) on {device}",
        flush=True,
    )

    all_rows: List[Dict[str, Any]] = []
    max_rot_roundtrip_err = 0.0

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

        # Encode all T Demo frames with frozen CLIP (fp16 autocast)
        demo_all_feats = visual.encode_dual_camera_uint8(
            np.ascontiguousarray(av_all),
            np.ascontiguousarray(eh_all),
            convention=conv,
            batch_size=256,
            autocast_dtype=torch.float16 if device.type == "cuda" else None,
        )  # (T, 2, 768)

        with amp_ctx:
            h_mem, b_mem, anchors = policy.teacher.extract_behavior_memory_from_demo_features(
                demo_all_feats, stride=behavior_stride
            )
            h_mem_f = h_mem.float().contiguous()
            b_mem_f = b_mem.float().contiguous()
            M = len(anchors)

            h_q = h_mem_f  # Query = Demo's own [t_k-1, t_k] visual obs (NO future frames)

            # D1: exact local self latent b_k
            z_d1 = b_mem_f

            # D2: Reader retrieval only
            h_mem_exp = h_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            b_mem_exp = b_mem_f.unsqueeze(0).expand(M, M, -1, -1)
            z_d2, attn_w = policy.reader(h_q, h_mem_exp, b_mem_exp, mem_pad_mask=None)

            # D3: Transfer only on exact self latent b_k
            z_d3, _ = policy.adapt_latent(z_d1, h_q, delta_scale=1.0)

            # D4: Full Model (Reader -> Direct Transfer)
            z_d4, _ = policy.adapt_latent(z_d2, h_q, delta_scale=1.0)

            noises = []
            for t_k in anchors:
                s_k = _stable_seed(
                    f"{demo_type}/{task}/{demo_label}/t{int(t_k)}", base_seed=base_seed
                )
                gen = torch.Generator(device="cpu").manual_seed(s_k)
                noises.append(torch.randn((ACTION_HORIZON, ACTION_DIM_10D), generator=gen))
            init_noise = torch.stack(noises, dim=0).to(device=device, dtype=torch.float32)

            pred_d1 = policy.teacher.sample_from_z_self(z_d1, init_noise=init_noise)
            pred_d2 = policy.teacher.sample_from_z_self(z_d2, init_noise=init_noise)
            pred_d3 = policy.teacher.sample_from_z_self(z_d3, init_noise=init_noise)
            pred_d4 = policy.teacher.sample_from_z_self(z_d4, init_noise=init_noise)

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

        preds = {
            "D1_teacher": pred_d1,
            "D2_retrieval": pred_d2,
            "D3_transfer": pred_d3,
            "D4_full": pred_d4,
        }
        latents = {
            "D1_teacher": z_d1.float(),
            "D2_retrieval": z_d2.float(),
            "D3_transfer": z_d3.float(),
            "D4_full": z_d4.float(),
        }
        act_metrics = {
            m: _compute_per_anchor_action_metrics(preds[m], gt_10d, valid_16)
            for m in MODES
        }

        z_ref = latents["D1_teacher"]
        z_ref_flat = z_ref.reshape(M, -1)
        lat_metrics: Dict[str, Dict[str, np.ndarray]] = {}
        for m in MODES:
            zm = latents[m]
            lat_metrics[m] = {
                "mse": (zm - z_ref).square().mean(dim=(1, 2)).cpu().numpy(),
                "cos": F.cosine_similarity(zm.reshape(M, -1), z_ref_flat, dim=-1).cpu().numpy(),
                "sq_mean": zm.square().mean(dim=(1, 2)).cpu().numpy(),
            }
        lat_mse_d4_d3 = (
            (latents["D4_full"] - latents["D3_transfer"]).square().mean(dim=(1, 2)).cpu().numpy()
        )
        lat_cos_d4_d3 = (
            F.cosine_similarity(
                latents["D4_full"].reshape(M, -1),
                latents["D3_transfer"].reshape(M, -1),
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        w_np = attn_w.float().cpu().numpy()
        ast = compute_attention_stats(attn_w, mem_pad_mask=None)
        ent_np = ast["entropy"].cpu().numpy()
        nent_np = ast["norm_entropy"].cpu().numpy()
        eff_np = ast["eff_pos_exp"].cpu().numpy()
        wmax_np = ast["max_weight"].cpu().numpy()

        idx_grid = np.arange(M, dtype=np.float64)
        anchors_f = anchors.astype(np.float64)
        gt_chunks_7d = actions_7d[clipped_grid]

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

            v_mask = valid_16_np[k]
            gt_pos_l2 = float(
                np.mean(np.linalg.norm(gt_chunks_7d[k, v_mask, :3], axis=-1))
            )
            gt_grip_pos = float(np.mean(gt_chunks_7d[k, v_mask, 6] >= 0.0))

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
                "n_valid_16": int(act_metrics["D1_teacher"]["n_valid_16"][k]),
                "n_valid_12": int(act_metrics["D1_teacher"]["n_valid_12"][k]),
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
                "gt_pos_l2": gt_pos_l2,
                "gt_grip_pos_frac": gt_grip_pos,
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
            all_rows.append(row)

        if idx % 50 == 0 or idx == len(specs):
            dt = time.monotonic() - t0
            print(
                f"[E9 D1-D4 Diag] {idx}/{len(specs)} demos ({len(all_rows):,} anchors) "
                f"in {dt:.1f}s ({idx / max(dt, 1e-6):.1f} demos/s)",
                flush=True,
            )

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
        with (out_dir / "per_anchor_records.csv").open("w", newline="") as f:
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
                        "reader_pos_err_argmax_slots": rd["pos_err_argmax_slots_mean"],
                    }
                )
    if summary_rows:
        with (out_dir / "summary_by_demo_type.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    elapsed = round(time.monotonic() - t0, 2)
    summary: Dict[str, Any] = {
        "experiment": "e9_stage2_direct_d1_d4_offline_decomposition",
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
            "tail_window_mask": "valid_step = (t_k + s < T); primary metrics step-weighted over valid steps",
            "max_rot_roundtrip_err": max_rot_roundtrip_err,
            "all_modules_frozen": True,
        },
        "elapsed_seconds": elapsed,
        "shared_9_tasks": shared_9_tasks,
        "by_demo_type_micro": by_demo_type,
        "by_demo_type_macro_all_10": macro_all_10,
        "by_demo_type_macro_shared_9": macro_shared_9,
        "by_demo_type_oracle_demo_weighted_9": oracle_demo_weighted_9,
        "by_demo_type_oracle_anchor_weighted_9": oracle_anchor_weighted_9,
        "per_task_by_demo_type": per_task_by_type,
    }
    atomic_write_json(summary, out_dir / "offline_summary.json")

    print("\n=== E9 Stage 2 Direct D1-D4 Summary (9-Task Equal-Weight Macro) ===", flush=True)
    for dt in demo_types:
        blk = macro_shared_9.get(dt, by_demo_type[dt])
        rd = blk["reader"]
        print(
            f"\n[{dt.upper()}] Reader: max_w={rd['max_weight_mean']:.4f}, "
            f"self_w={rd['self_weight_mean']:.4f}, exact_hit={100*rd['exact_hit_rate']:.2f}%, "
            f"within1={100*rd['within1_hit_rate']:.2f}%, "
            f"argmax_err={rd['pos_err_argmax_slots_mean']:.3f} slots ({rd['pos_err_argmax_steps_mean']:.1f} steps)",
            flush=True,
        )
        for m in MODES:
            ms = blk["modes"][m]
            dl = blk["deltas_vs_D1"].get(m)
            dstr = (
                f" (Δ10D={dl['ddim_10d_mse_abs']:+.6f} / {dl['ddim_10d_mse_rel_pct']:+.1f}%)"
                if dl
                else ""
            )
            print(
                f"  {m:13s}: 10D_MSE={ms['ddim_10d_mse']:.6f}{dstr} | "
                f"7D_MAE={ms['ddim_7d_mae']:.5f} | Exec12={ms['exec12_7d_mae']:.5f} | "
                f"Grip={100*ms['grip_acc']:.2f}% | "
                f"LatMSE={ms['latent_mse_vs_self']:.4f} (Lz={ms['latent_lz_vs_self']:.4f}) | "
                f"Cos={ms['latent_cos_vs_self']:.5f}",
                flush=True,
            )
        # Print w_self >= 0.9 stratification from micro block
        st9 = by_demo_type[dt]["d3_stratification"].get("w_self_ge_0.9", {})
        if st9.get("num_anchors", 0) > 0:
            print(
                f"  [w_self>=0.9 ({st9['num_anchors']} / {100*st9['frac']:.1f}%)] "
                f"D1={st9['D1_mse10']:.5f} -> D2={st9['D2_mse10']:.5f} (LatMSE={st9['D2_lat_mse']:.4f}) -> "
                f"D3={st9['D3_mse10']:.5f} (LatMSE={st9['D3_lat_mse']:.4f}) / "
                f"D4={st9['D4_mse10']:.5f} (LatMSE={st9['D4_lat_mse']:.4f})",
                flush=True,
            )

    print(f"\n[E9 D1-D4 Diag DONE] Saved to {out_dir} in {elapsed}s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, default=DEFAULT_STAGE2_DIRECT_DIR / "train/best.pt"
    )
    parser.add_argument("--stage1-ckpt", type=Path, default=DEFAULT_STAGE1_LOCAL_CKPT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_STAGE2_DIRECT_DIR / "d1_d4_diagnosis"
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--stride", type=int, default=BEHAVIOR_STRIDE)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--demo-types",
        nargs="+",
        default=["default", "shifted", "oracle_gt"],
        choices=("default", "shifted", "oracle_gt"),
    )
    parser.add_argument("--max-demos-per-type", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_e9_d1_d4_diagnosis(
        e9_stage2_ckpt=args.ckpt,
        stage1_ckpt=args.stage1_ckpt,
        out_dir=args.output,
        device=device,
        use_bf16=not args.no_bf16,
        base_seed=args.seed,
        behavior_stride=args.stride,
        demo_types=tuple(args.demo_types),
        max_demos_per_type=args.max_demos_per_type,
    )


if __name__ == "__main__":
    main()
