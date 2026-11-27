"""Aggregate E8 Source-conditioned Latent Behavior Transfer offline, source-shuffle,
and closed-loop results and compare head-to-head against E7-Direct (and E7 Residual)
across Default (500), Oracle-GT (331 paired), and Shifted (500 paired).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from e5.train import atomic_write_json
from e7 import OUTPUT_ROOT as E7_OUTPUT_ROOT
from e7.analyze_direct_results import _extract_offline_block, _load_rollout_csv, _paired_stats

from . import OUTPUT_ROOT


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
                "cos_href_hq": (
                    iv["source_state_alignment"]["cross_cos_href_hq"]
                    if "source_state_alignment" in iv
                    else None
                ),
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
                "cos_href_hq": (
                    v["source_state_alignment"]["cross_cos_href_hq"]
                    if "source_state_alignment" in v
                    else None
                ),
            }
        )
    return rows


def build_e8_comparison(
    *,
    e8_root: Path = OUTPUT_ROOT,
    run_name: str = "e8_source_conditioned",
) -> Dict[str, Any]:
    e8_run = e8_root / "train" / run_name
    e7_direct_root = E7_OUTPUT_ROOT / "direct_latent"
    e7_direct_run = e7_direct_root / "train" / "e7_direct_latent"
    e7_res_run = E7_OUTPUT_ROOT / "train" / "e7_behavior_memory_retrieval"

    e8_final = json.loads((e8_run / "final_eval.json").read_text())
    e8_metrics = json.loads((e8_run / "metrics.json").read_text())
    e8_shuffle = (
        json.loads((e8_run / "source_shuffle_diag.json").read_text())
        if (e8_run / "source_shuffle_diag.json").is_file()
        else e8_final.get("source_state_shuffle_diag")
    )

    dir_final = json.loads((e7_direct_run / "final_eval.json").read_text())
    dir_metrics = json.loads((e7_direct_run / "metrics.json").read_text())
    res_final = json.loads((e7_res_run / "final_eval.json").read_text())

    # Closed-loop rollout CSVs
    d_def = _load_rollout_csv(e7_direct_root / "eval_chain/e7_direct/rollout_results.csv")
    d_ogt = _load_rollout_csv(e7_direct_root / "demoswap/oracle_gt/e7_direct/rollout_results.csv")
    d_shf = _load_rollout_csv(e7_direct_root / "demoswap/shifted/e7_direct/rollout_results.csv")

    r_def = _load_rollout_csv(E7_OUTPUT_ROOT / "eval_chain/e7/rollout_results.csv")
    r_ogt = _load_rollout_csv(E7_OUTPUT_ROOT / "demoswap/oracle_gt/e7/rollout_results.csv")
    r_shf = _load_rollout_csv(E7_OUTPUT_ROOT / "demoswap/shifted/e7/rollout_results.csv")

    e8_def = _load_rollout_csv(e8_root / "eval_chain/e8_source_conditioned/rollout_results.csv")
    e8_ogt = _load_rollout_csv(e8_root / "demoswap/oracle_gt/e8_source_conditioned/rollout_results.csv")
    e8_shf = _load_rollout_csv(e8_root / "demoswap/shifted/e8_source_conditioned/rollout_results.csv")

    tasks = sorted({k[0] for k in e8_def.keys()})
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
            "e7_direct_default": _tsr(d_def),
            "e8_default": _tsr(e8_def),
            "e7_direct_oracle_gt": _tsr(d_ogt),
            "e8_oracle_gt": _tsr(e8_ogt),
            "e7_direct_shifted": _tsr(d_shf),
            "e8_shifted": _tsr(e8_shf),
        }

    def_vs_dir = _paired_stats(e8_def, d_def)
    ogt_vs_dir = _paired_stats(e8_ogt, d_ogt)
    shf_vs_dir = _paired_stats(e8_shf, d_shf)

    def_vs_res = _paired_stats(e8_def, r_def)
    ogt_vs_res = _paired_stats(e8_ogt, r_ogt)
    shf_vs_res = _paired_stats(e8_shf, r_shf)

    e8_ogt_swap = _paired_stats(e8_ogt, e8_def)
    e8_shf_swap = _paired_stats(e8_shf, e8_def)
    dir_ogt_swap = _paired_stats(d_ogt, d_def)
    dir_shf_swap = _paired_stats(d_shf, d_def)

    e8_mean3 = (def_vs_dir["a_sr"] + ogt_vs_dir["a_sr"] + shf_vs_dir["a_sr"]) / 3.0
    dir_mean3 = (def_vs_dir["b_sr"] + ogt_vs_dir["b_sr"] + shf_vs_dir["b_sr"]) / 3.0
    res_mean3 = (def_vs_res["b_sr"] + ogt_vs_res["b_sr"] + shf_vs_res["b_sr"]) / 3.0

    e8_off = _extract_offline_block(e8_final)
    e8_off["source_state_alignment"] = e8_final["final_eval"].get("source_state_alignment")

    summary: Dict[str, Any] = {
        "offline_val_6248": {
            "e7_residual": _extract_offline_block(res_final),
            "e7_direct": _extract_offline_block(dir_final),
            "e8_source_conditioned": e8_off,
        },
        "source_state_shuffle_diag": e8_shuffle,
        "epoch_trajectory": {
            "e7_direct": _traj(dir_metrics),
            "e8_source_conditioned": _traj(e8_metrics),
        },
        "closed_loop": {
            "e8_vs_e7_direct_default_500": def_vs_dir,
            "e8_vs_e7_direct_oracle_gt_331": ogt_vs_dir,
            "e8_vs_e7_direct_shifted_500": shf_vs_dir,
            "e8_vs_e7_residual_default_500": def_vs_res,
            "e8_vs_e7_residual_oracle_gt_331": ogt_vs_res,
            "e8_vs_e7_residual_shifted_500": shf_vs_res,
            "three_group_mean": {
                "e8_sr": e8_mean3,
                "e7_direct_sr": dir_mean3,
                "e7_residual_sr": res_mean3,
                "delta_pp_vs_e7_direct": 100.0 * (e8_mean3 - dir_mean3),
                "delta_pp_vs_e7_residual": 100.0 * (e8_mean3 - res_mean3),
            },
            "e8_oracle_gt_vs_default_331": e8_ogt_swap,
            "e8_shifted_vs_default_500": e8_shf_swap,
            "e7_direct_oracle_gt_vs_default_331": dir_ogt_swap,
            "e7_direct_shifted_vs_default_500": dir_shf_swap,
            "per_task": per_task,
        },
    }

    out_path = e8_root / "comparison_vs_e7_direct.json"
    atomic_write_json(summary, out_path)
    print(json.dumps(summary["closed_loop"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e8-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="e8_source_conditioned")
    args = parser.parse_args()
    build_e8_comparison(e8_root=args.e8_root, run_name=args.run_name)


if __name__ == "__main__":
    main()
