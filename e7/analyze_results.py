"""Aggregate E7 offline & closed-loop evaluation results and compare head-to-head
against E6 Stage 2.1 (B) across Default (500), Oracle-GT (331 paired), and Shifted (500 paired),
including both Cross+Delta and No-Delta variants.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from e5.train import atomic_write_json

from . import OUTPUT_ROOT, ROOT


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
    t = n - w - l
    return {
        "n": n,
        "a_ok": a_ok,
        "b_ok": b_ok,
        "a_sr": a_ok / n,
        "b_sr": b_ok / n,
        "delta_pp": 100.0 * (a_ok - b_ok) / n,
        "wins": w,
        "losses": l,
        "ties": t,
    }


def build_comparison_summary(
    *,
    e7_run_dir: Path = OUTPUT_ROOT / "train/e7_behavior_memory_retrieval",
    e7_eval_root: Path = OUTPUT_ROOT,
    out_path: Path = OUTPUT_ROOT / "comparison_vs_e6_stage2_1.json",
) -> Dict[str, Any]:
    b_final = json.loads(
        (
            ROOT
            / "e6/outputs/stage2_1/stage2_1_joint_reader_delta/final_eval.json"
        ).read_text()
    )["final_eval"]
    b_op = json.loads(
        (ROOT / "e6/outputs/stage2_3/baseline_b_eval.json").read_text()
    )["eval"]
    e7_final_full = json.loads((e7_run_dir / "final_eval.json").read_text())
    e7_final = e7_final_full["final_eval"]
    e7_metrics = json.loads((e7_run_dir / "metrics.json").read_text())

    # Load rollout CSVs
    b_def = _load_rollout_csv(
        ROOT / "e6/outputs/stage2_1/eval_chain/e6_stage2_1/rollout_results.csv"
    )
    b_nd_def = _load_rollout_csv(
        ROOT / "e6/outputs/stage2_1/eval_chain/e6_stage2_1_no_delta/rollout_results.csv"
    )
    b_ogt = _load_rollout_csv(
        ROOT / "e6/outputs/stage2_1/demoswap/oracle_gt/e6_stage2_1/rollout_results.csv"
    )
    b_shf = _load_rollout_csv(
        ROOT / "e6/outputs/stage2_1/demoswap/shifted/e6_stage2_1/rollout_results.csv"
    )

    e7_def = _load_rollout_csv(e7_eval_root / "eval_chain/e7/rollout_results.csv")
    e7_nd_def = _load_rollout_csv(
        e7_eval_root / "eval_chain/e7_no_delta/rollout_results.csv"
    )
    e7_ogt = _load_rollout_csv(
        e7_eval_root / "demoswap/oracle_gt/e7/rollout_results.csv"
    )
    e7_shf = _load_rollout_csv(
        e7_eval_root / "demoswap/shifted/e7/rollout_results.csv"
    )

    e7_nd_ogt_path = e7_eval_root / "demoswap/oracle_gt/e7_no_delta/rollout_results.csv"
    e7_nd_shf_path = e7_eval_root / "demoswap/shifted/e7_no_delta/rollout_results.csv"
    e7_nd_ogt = _load_rollout_csv(e7_nd_ogt_path) if e7_nd_ogt_path.is_file() else None
    e7_nd_shf = _load_rollout_csv(e7_nd_shf_path) if e7_nd_shf_path.is_file() else None

    tasks = sorted({k[0] for k in e7_def.keys()})
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
            "b_default": _tsr(b_def),
            "b_no_delta_default": _tsr(b_nd_def),
            "e7_default": _tsr(e7_def),
            "e7_no_delta_default": _tsr(e7_nd_def),
            "b_oracle_gt": _tsr(b_ogt),
            "e7_oracle_gt": _tsr(e7_ogt),
            "e7_no_delta_oracle_gt": _tsr(e7_nd_ogt),
            "b_shifted": _tsr(b_shf),
            "e7_shifted": _tsr(e7_shf),
            "e7_no_delta_shifted": _tsr(e7_nd_shf),
        }

    summary: Dict[str, Any] = {
        "offline_val_6248": {
            "e6_stage2_1_B": {
                "cross_no_delta_latent_mse_raw": b_final["latent"]["cross_no_delta_mse_raw"],
                "cross_no_delta_L_z": b_final["latent"]["cross_no_delta_L_z"],
                "cross_with_delta_latent_mse_raw": b_final["latent"]["cross_with_delta_mse_raw"],
                "cross_with_delta_L_z": b_final["latent"]["cross_with_delta_L_z"],
                "cross_no_delta_diff_loss": b_final["diff_loss"]["cross_no_delta"],
                "cross_with_delta_diff_loss": b_final["diff_loss"]["cross_with_delta"],
                "cross_no_delta_ddim_mse10": b_final["ddim"]["cross_no_delta"]["action10_mse"],
                "cross_no_delta_ddim_mae7": b_final["ddim"]["cross_no_delta"]["action7_mae"],
                "cross_with_delta_ddim_mse10": b_final["ddim"]["cross_with_delta"]["action10_mse"],
                "cross_with_delta_ddim_mae7": b_final["ddim"]["cross_with_delta"]["action7_mae"],
                "cross_delta_rms": b_final["latent"]["cross_delta_rms"],
                "correction_direction_cosine": b_op["operator_stability"][
                    "step1_direction_cosine_mean"
                ],
                "wrong_demo_sens_diff_no_delta": b_final["diff_loss"][
                    "demo_sensitivity_no_delta"
                ],
                "wrong_demo_sens_diff_with_delta": b_final["diff_loss"][
                    "demo_sensitivity_with_delta"
                ],
                "wrong_demo_sens_ddim_mse10_no_delta": b_final["ddim"][
                    "demo_sensitivity_no_delta_action10_mse"
                ],
                "wrong_demo_sens_ddim_mse10_with_delta": b_final["ddim"][
                    "demo_sensitivity_action10_mse"
                ],
                "multi_demo_mse10_no_delta": b_final["multi_demo_consistency"][
                    "pred_d1_vs_d2_action10_mse_no_delta"
                ],
                "multi_demo_mse10_with_delta": b_final["multi_demo_consistency"][
                    "pred_d1_vs_d2_action10_mse_with_delta"
                ],
            },
            "e7": {
                "best_epoch": e7_final_full["best_epoch"],
                "cross_no_delta_latent_mse_raw": e7_final["latent"]["cross_no_delta_mse_raw"],
                "cross_no_delta_L_z": e7_final["latent"]["cross_no_delta_L_z"],
                "cross_with_delta_latent_mse_raw": e7_final["latent"][
                    "cross_with_delta_mse_raw"
                ],
                "cross_with_delta_L_z": e7_final["latent"]["cross_with_delta_L_z"],
                "cross_no_delta_diff_loss": e7_final["diff_loss"]["cross_no_delta"],
                "cross_with_delta_diff_loss": e7_final["diff_loss"]["cross_with_delta"],
                "cross_no_delta_ddim_mse10": e7_final["ddim"]["cross_no_delta"]["action10_mse"],
                "cross_no_delta_ddim_mae7": e7_final["ddim"]["cross_no_delta"]["action7_mae"],
                "cross_with_delta_ddim_mse10": e7_final["ddim"]["cross_with_delta"][
                    "action10_mse"
                ],
                "cross_with_delta_ddim_mae7": e7_final["ddim"]["cross_with_delta"][
                    "action7_mae"
                ],
                "cross_delta_rms": e7_final["latent"]["cross_delta_rms"],
                "correction_direction_cosine": e7_final["latent"][
                    "correction_direction_cosine_mean"
                ],
                "delta_mag_gap_pearson_r": e7_final["latent"]["delta_mag_gap_pearson_r"],
                "wrong_demo_sens_diff_no_delta": e7_final["diff_loss"][
                    "demo_sensitivity_no_delta"
                ],
                "wrong_demo_sens_diff_with_delta": e7_final["diff_loss"][
                    "demo_sensitivity_with_delta"
                ],
                "wrong_demo_sens_ddim_mse10_no_delta": e7_final["ddim"][
                    "demo_sensitivity_no_delta_action10_mse"
                ],
                "wrong_demo_sens_ddim_mse10_with_delta": e7_final["ddim"][
                    "demo_sensitivity_action10_mse"
                ],
                "multi_demo_mse10_no_delta": e7_final["multi_demo_consistency"][
                    "pred_d1_vs_d2_action10_mse_no_delta"
                ],
                "multi_demo_mse10_with_delta": e7_final["multi_demo_consistency"][
                    "pred_d1_vs_d2_action10_mse_with_delta"
                ],
                "reader_attention": e7_final["reader_attention"],
                "contributions": e7_final["contributions"],
            },
        },
        "attention_trajectory": [
            {
                "epoch": 0,
                **e7_metrics["init_val"]["reader_attention"]["cross"],
            }
        ]
        + [
            {
                "epoch": rec["epoch"],
                **rec["val"]["reader_attention"]["cross"],
            }
            for rec in e7_metrics["history"]
        ],
        "closed_loop": {
            "e7_vs_B_default_500": _paired_stats(e7_def, b_def),
            "e7_no_delta_vs_B_no_delta_default_500": _paired_stats(e7_nd_def, b_nd_def),
            "e7_vs_e7_no_delta_default_500": _paired_stats(e7_def, e7_nd_def),
            "e7_vs_B_oracle_gt_331": _paired_stats(e7_ogt, b_ogt),
            "e7_vs_B_shifted_500": _paired_stats(e7_shf, b_shf),
            "e7_oracle_gt_vs_default_331": _paired_stats(e7_ogt, e7_def),
            "e7_shifted_vs_default_500": _paired_stats(e7_shf, e7_def),
            "B_oracle_gt_vs_default_331": _paired_stats(b_ogt, b_def),
            "B_shifted_vs_default_500": _paired_stats(b_shf, b_def),
            "e7_no_delta_oracle_gt_vs_default_331": (
                _paired_stats(e7_nd_ogt, e7_nd_def) if e7_nd_ogt is not None else None
            ),
            "e7_no_delta_shifted_vs_default_500": (
                _paired_stats(e7_nd_shf, e7_nd_def) if e7_nd_shf is not None else None
            ),
            "e7_vs_e7_no_delta_oracle_gt_331": (
                _paired_stats(e7_ogt, e7_nd_ogt) if e7_nd_ogt is not None else None
            ),
            "e7_vs_e7_no_delta_shifted_500": (
                _paired_stats(e7_shf, e7_nd_shf) if e7_nd_shf is not None else None
            ),
            "per_task": per_task,
        },
    }
    atomic_write_json(summary, out_path)
    print(json.dumps(summary["closed_loop"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e7-run-dir",
        type=Path,
        default=OUTPUT_ROOT / "train/e7_behavior_memory_retrieval",
    )
    parser.add_argument("--e7-eval-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_ROOT / "comparison_vs_e6_stage2_1.json",
    )
    args = parser.parse_args()
    build_comparison_summary(
        e7_run_dir=args.e7_run_dir, e7_eval_root=args.e7_eval_root, out_path=args.out
    )


if __name__ == "__main__":
    main()
