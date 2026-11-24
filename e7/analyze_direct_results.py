"""Aggregate E7-Direct (Direct Latent Prediction Ablation) offline & closed-loop results
and compare head-to-head against E7 Residual (Global Soft Attention) across Default (500),
Oracle-GT (331 paired), and Shifted (500 paired).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from e5.train import atomic_write_json

from . import OUTPUT_ROOT


def _load_rollout_csv(path: Path) -> Dict[Tuple[str, int], bool]:
    out: Dict[Tuple[str, int], bool] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["task"], int(row["layout"]))] = str(row["success"]).lower() == "true"
    return out


def _paired_stats(
    map_a: Dict[Tuple[str, int], bool],
    map_b: Dict[Tuple[str, int], bool],
) -> Dict[str, Any]:
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    n = len(keys)
    if n == 0:
        return {"n": 0}
    a_ok = sum(1 for k in keys if map_a[k])
    b_ok = sum(1 for k in keys if map_b[k])
    w = sum(1 for k in keys if map_a[k] and not map_b[k])
    l = sum(1 for k in keys if not map_a[k] and map_b[k])
    return {
        "n": n,
        "a_ok": a_ok,
        "b_ok": b_ok,
        "a_sr": a_ok / n,
        "b_sr": b_ok / n,
        "delta_pp": 100.0 * (a_ok - b_ok) / n,
        "wins": w,
        "losses": l,
        "ties": n - w - l,
    }


def _extract_offline_block(final_full: Dict[str, Any]) -> Dict[str, Any]:
    ev = final_full["final_eval"]
    return {
        "best_epoch": final_full["best_epoch"],
        "best_global_step": final_full.get("best_global_step"),
        "prediction_mode": final_full.get("prediction_mode", "residual"),
        "init_val_summary": final_full.get("init_val_summary"),
        "cross_no_delta_latent_mse_raw": ev["latent"]["cross_no_delta_mse_raw"],
        "cross_no_delta_L_z": ev["latent"]["cross_no_delta_L_z"],
        "cross_with_delta_latent_mse_raw": ev["latent"]["cross_with_delta_mse_raw"],
        "cross_with_delta_L_z": ev["latent"]["cross_with_delta_L_z"],
        "cross_no_delta_cos": ev["latent"]["cross_no_delta_cos"],
        "cross_with_delta_cos": ev["latent"]["cross_with_delta_cos"],
        "cross_z_rms": ev["latent"]["cross_z_rms"],
        "adapted_z_rms": ev["latent"]["adapted_z_rms"],
        "self_z_rms": ev["latent"]["self_z_rms"],
        "cross_delta_rms": ev["latent"]["cross_delta_rms"],
        "correction_direction_cosine": ev["latent"]["correction_direction_cosine_mean"],
        "delta_mag_gap_pearson_r": ev["latent"]["delta_mag_gap_pearson_r"],
        "cross_no_delta_diff_loss": ev["diff_loss"]["cross_no_delta"],
        "cross_with_delta_diff_loss": ev["diff_loss"]["cross_with_delta"],
        "cross_no_delta_ddim_mse10": ev["ddim"]["cross_no_delta"]["action10_mse"],
        "cross_no_delta_ddim_mae10": ev["ddim"]["cross_no_delta"]["action10_mae"],
        "cross_no_delta_ddim_mae7": ev["ddim"]["cross_no_delta"]["action7_mae"],
        "cross_with_delta_ddim_mse10": ev["ddim"]["cross_with_delta"]["action10_mse"],
        "cross_with_delta_ddim_mae10": ev["ddim"]["cross_with_delta"]["action10_mae"],
        "cross_with_delta_ddim_mae7": ev["ddim"]["cross_with_delta"]["action7_mae"],
        "cross_with_delta_exec12_mae7": ev["ddim"]["cross_with_delta"]["exec12_action7_mae"],
        "cross_with_delta_grip_sign_acc": ev["ddim"]["cross_with_delta"]["grip_sign_acc"],
        "wrong_demo_sens_diff_no_delta": ev["diff_loss"]["demo_sensitivity_no_delta"],
        "wrong_demo_sens_diff_with_delta": ev["diff_loss"]["demo_sensitivity_with_delta"],
        "wrong_demo_sens_ddim_mse10_no_delta": ev["ddim"][
            "demo_sensitivity_no_delta_action10_mse"
        ],
        "wrong_demo_sens_ddim_mse10_with_delta": ev["ddim"]["demo_sensitivity_action10_mse"],
        "wrong_demo_sens_ddim_mae7_with_delta": ev["ddim"]["demo_sensitivity_action7_mae"],
        "multi_demo_consistency": ev["multi_demo_consistency"],
        "reader_attention": ev["reader_attention"],
    }


def build_direct_comparison(
    *,
    direct_root: Path = OUTPUT_ROOT / "direct_latent",
    run_name: str = "e7_direct_latent",
) -> Dict[str, Any]:
    direct_run = direct_root / "train" / run_name
    e7_run = OUTPUT_ROOT / "train/e7_behavior_memory_retrieval"

    e7_final_full = json.loads((e7_run / "final_eval.json").read_text())
    e7_metrics = json.loads((e7_run / "metrics.json").read_text())
    dir_final_full = json.loads((direct_run / "final_eval.json").read_text())
    dir_metrics = json.loads((direct_run / "metrics.json").read_text())

    # Closed-loop rollout CSVs
    e7_def = _load_rollout_csv(OUTPUT_ROOT / "eval_chain/e7/rollout_results.csv")
    e7_ogt = _load_rollout_csv(OUTPUT_ROOT / "demoswap/oracle_gt/e7/rollout_results.csv")
    e7_shf = _load_rollout_csv(OUTPUT_ROOT / "demoswap/shifted/e7/rollout_results.csv")

    d_def = _load_rollout_csv(direct_root / "eval_chain/e7_direct/rollout_results.csv")
    d_ogt = _load_rollout_csv(direct_root / "demoswap/oracle_gt/e7_direct/rollout_results.csv")
    d_shf = _load_rollout_csv(direct_root / "demoswap/shifted/e7_direct/rollout_results.csv")

    tasks = sorted({k[0] for k in d_def.keys()})
    per_task: Dict[str, Any] = {}
    for task in tasks:
        def _tsr(m: Optional[Dict[Tuple[str, int], bool]]) -> Optional[Dict[str, Any]]:
            if m is None:
                return None
            sub = [v for (t, _), v in m.items() if t == task]
            if not sub:
                return None
            ok = sum(1 for v in sub if v)
            return {"ok": ok, "n": len(sub), "sr": ok / len(sub)}

        per_task[task] = {
            "e7_residual_default": _tsr(e7_def),
            "e7_direct_default": _tsr(d_def),
            "e7_residual_oracle_gt": _tsr(e7_ogt),
            "e7_direct_oracle_gt": _tsr(d_ogt),
            "e7_residual_shifted": _tsr(e7_shf),
            "e7_direct_shifted": _tsr(d_shf),
        }

    def_vs_e7 = _paired_stats(d_def, e7_def)
    ogt_vs_e7 = _paired_stats(d_ogt, e7_ogt)
    shf_vs_e7 = _paired_stats(d_shf, e7_shf)

    dir_ogt_swap = _paired_stats(d_ogt, d_def)
    dir_shf_swap = _paired_stats(d_shf, d_def)
    e7_ogt_swap = _paired_stats(e7_ogt, e7_def)
    e7_shf_swap = _paired_stats(e7_shf, e7_def)

    dir_mean3 = (def_vs_e7["a_sr"] + ogt_vs_e7["a_sr"] + shf_vs_e7["a_sr"]) / 3.0
    e7_mean3 = (def_vs_e7["b_sr"] + ogt_vs_e7["b_sr"] + shf_vs_e7["b_sr"]) / 3.0

    def _traj(metrics: Dict[str, Any]) -> list[Dict[str, Any]]:
        rows = []
        iv = metrics.get("init_val")
        if iv is not None:
            rows.append(
                {
                    "epoch": 0,
                    "global_step": 0,
                    "zq_rms": iv["latent"]["adapted_z_rms"],
                    "cross_no_delta_L_z": iv["latent"]["cross_no_delta_L_z"],
                    "cross_with_delta_L_z": iv["latent"]["cross_with_delta_L_z"],
                    "cross_with_delta_diff": iv["diff_loss"]["cross_with_delta"],
                    "cross_no_delta_ddim10": iv["ddim"]["cross_no_delta"]["action10_mse"],
                    "cross_with_delta_ddim10": iv["ddim"]["cross_with_delta"]["action10_mse"],
                    "cross_with_delta_mae7": iv["ddim"]["cross_with_delta"]["action7_mae"],
                    "attn_entropy": iv["reader_attention"]["cross"]["entropy_mean"],
                    "attn_norm_entropy": iv["reader_attention"]["cross"]["normalized_entropy_mean"],
                    "attn_max_weight": iv["reader_attention"]["cross"]["max_weight_mean"],
                    "attn_eff_pos_exp": iv["reader_attention"]["cross"][
                        "effective_positions_exp_entropy"
                    ],
                    "attn_eff_pos_ipr": iv["reader_attention"]["cross"][
                        "effective_positions_ipr"
                    ],
                }
            )
        for rec in metrics["history"]:
            v = rec["val"]
            rows.append(
                {
                    "epoch": rec["epoch"],
                    "global_step": rec["global_step"],
                    "zq_rms": v["latent"]["adapted_z_rms"],
                    "cross_no_delta_L_z": v["latent"]["cross_no_delta_L_z"],
                    "cross_with_delta_L_z": v["latent"]["cross_with_delta_L_z"],
                    "cross_with_delta_diff": v["diff_loss"]["cross_with_delta"],
                    "cross_no_delta_ddim10": v["ddim"]["cross_no_delta"]["action10_mse"],
                    "cross_with_delta_ddim10": v["ddim"]["cross_with_delta"]["action10_mse"],
                    "cross_with_delta_mae7": v["ddim"]["cross_with_delta"]["action7_mae"],
                    "attn_entropy": v["reader_attention"]["cross"]["entropy_mean"],
                    "attn_norm_entropy": v["reader_attention"]["cross"]["normalized_entropy_mean"],
                    "attn_max_weight": v["reader_attention"]["cross"]["max_weight_mean"],
                    "attn_eff_pos_exp": v["reader_attention"]["cross"][
                        "effective_positions_exp_entropy"
                    ],
                    "attn_eff_pos_ipr": v["reader_attention"]["cross"][
                        "effective_positions_ipr"
                    ],
                }
            )
        return rows

    summary: Dict[str, Any] = {
        "offline_val_6248": {
            "e7_residual": _extract_offline_block(e7_final_full),
            "e7_direct": _extract_offline_block(dir_final_full),
        },
        "epoch_trajectory": {
            "e7_residual": _traj(e7_metrics),
            "e7_direct": _traj(dir_metrics),
        },
        "closed_loop": {
            "direct_vs_residual_default_500": def_vs_e7,
            "direct_vs_residual_oracle_gt_331": ogt_vs_e7,
            "direct_vs_residual_shifted_500": shf_vs_e7,
            "three_group_mean": {
                "e7_direct_sr": dir_mean3,
                "e7_residual_sr": e7_mean3,
                "delta_pp": 100.0 * (dir_mean3 - e7_mean3),
            },
            "direct_oracle_gt_vs_default_331": dir_ogt_swap,
            "direct_shifted_vs_default_500": dir_shf_swap,
            "residual_oracle_gt_vs_default_331": e7_ogt_swap,
            "residual_shifted_vs_default_500": e7_shf_swap,
            "per_task": per_task,
        },
    }

    out_path = direct_root / "comparison_vs_e7_residual.json"
    atomic_write_json(summary, out_path)
    print(json.dumps(summary["closed_loop"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-root", type=Path, default=OUTPUT_ROOT / "direct_latent")
    parser.add_argument("--run-name", type=str, default="e7_direct_latent")
    args = parser.parse_args()
    build_direct_comparison(direct_root=args.direct_root, run_name=args.run_name)


if __name__ == "__main__":
    main()
