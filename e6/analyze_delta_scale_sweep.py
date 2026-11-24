"""Analyze E6 Stage 2.1 Delta Scaling Sweep (CLIP) and run optional offline validation.

Reuses existing alpha=0 (No-Delta) and alpha=1.0 (original Stage 2.1) results without
modifying or overwriting any original files, materializing exact 331-trial copies under
`e6/outputs/delta_scale_sweep/alpha_0/` and `e6/outputs/delta_scale_sweep/alpha_1.0/`.

Outputs under `e6/outputs/delta_scale_sweep/`:
  - summary_table.csv       : unified comparison table across all 7 alphas
  - summary.json            : full structured metrics (overall, paired, vs-alpha1, per-task, offline)
  - paired_trials.csv       : 331 paired trials across all 7 alphas (Default & Oracle-GT)
  - per_task_sr.csv         : per-task SR and mean success steps across all 7 alphas
  - offline_metrics.csv     : offline DDIM Action MSE & corrected latent RMS on 6,248 val Cross samples
  - offline_metrics.json    : detailed offline metrics
  - REPORT.md               : comprehensive experiment report and scientific diagnosis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from e5 import ACTION_DIM_10D, ACTION_HORIZON
from e5.actions import action_10d_to_7d
from e6.stage2_1_data import Stage2_1TransferBank
from e6.stage2_1_model import load_stage2_1_policy
from e6.stage2_model import DEFAULT_STAGE1_CKPT

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = REPO_ROOT / "e6/outputs/stage2_1/stage2_1_joint_reader_delta/best.pt"
OUT_ROOT = REPO_ROOT / "e6/outputs/delta_scale_sweep"

ALPHAS: List[Tuple[str, float]] = [
    ("0", 0.0),
    ("0.5", 0.5),
    ("0.75", 0.75),
    ("1.0", 1.0),
    ("1.25", 1.25),
    ("1.5", 1.5),
    ("2.0", 2.0),
]

EXISTING_SOURCES = {
    "0": {
        "default": REPO_ROOT / "e6/outputs/stage2_1/eval_chain/e6_stage2_1_no_delta/rollout_results.csv",
        "oracle_gt": REPO_ROOT / "e6/outputs/demoswap/oracle_gt/no_delta_diagnostic/rollout_results.csv",
    },
    "1.0": {
        "default": REPO_ROOT / "e6/outputs/stage2_1/eval_chain/e6_stage2_1/rollout_results.csv",
        "oracle_gt": REPO_ROOT / "e6/outputs/stage2_1/demoswap/oracle_gt/e6_stage2_1/rollout_results.csv",
    },
}

TASK_SHORT_NAMES = {
    "open_the_middle_drawer_of_the_cabinet_and_then_put_the_cream_cheese_on_the_stove": "open_middle_drawer -> cream_cheese_on_stove",
    "open_the_top_drawer_and_then_put_the_wine_bottle_on_top_of_the_cabinet": "open_top_drawer -> wine_bottle_on_cabinet",
    "put_the_bowl_on_the_plate_and_then_put_the_cream_cheese_in_the_bowl": "bowl_on_plate -> cream_cheese_in_bowl",
    "put_the_bowl_on_top_of_the_cabinet_and_then_put_the_cream_cheese_on_the_plate": "bowl_on_cabinet -> cream_cheese_on_plate",
    "put_the_cream_cheese_on_the_plate_and_then_put_the_bowl_on_the_cream_cheese": "cream_cheese_on_plate -> bowl_on_cream_cheese",
    "put_the_cream_cheese_on_top_of_the_cabinet_and_then_put_the_wine_bottle_in_the_bowl": "cream_cheese_on_cabinet -> wine_bottle_in_bowl",
    "put_the_wine_bottle_in_front_of_the_stove_and_then_put_the_bowl_on_the_stove": "wine_bottle_front_stove -> bowl_on_stove",
    "put_the_wine_bottle_on_the_rack_and_then_put_the_bowl_on_top_of_the_cabinet": "wine_bottle_on_rack -> bowl_on_cabinet",
    "turn_on_the_stove_and_then_put_the_bowl_on_the_stove": "turn_on_stove -> bowl_on_stove",
}


def _load_rollout_csv(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["task"], int(r["layout"]))
            rows[key] = {
                "name": r["name"],
                "task": r["task"],
                "layout": int(r["layout"]),
                "donor": r["donor"],
                "prompt_mode": r.get("prompt_mode", "default"),
                "success": str(r["success"]).strip().lower() == "true",
                "steps": int(float(r["steps"])),
                "replans": int(float(r["replans"])),
                "wall_seconds": float(r["wall_seconds"]),
            }
    return rows


def _write_subset_csv_and_summary(
    rows_map: Dict[Tuple[str, int], Dict[str, Any]],
    keys: List[Tuple[str, int]],
    out_dir: Path,
    *,
    alpha_val: float,
    prompt_mode: str,
    source_csv: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = [rows_map[k] for k in keys]
    csv_path = out_dir / "rollout_results.csv"
    fieldnames = [
        "name",
        "task",
        "layout",
        "donor",
        "prompt_mode",
        "success",
        "steps",
        "replans",
        "wall_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in subset:
            row = dict(r)
            row["prompt_mode"] = prompt_mode
            w.writerow(row)

    tasks = sorted({k[0] for k in keys})
    by_task: Dict[str, Any] = {}
    for task in tasks:
        tr = [r for r in subset if r["task"] == task]
        ok = sum(1 for r in tr if r["success"])
        avg_steps = float(np.mean([r["steps"] for r in tr if r["success"]])) if ok > 0 else None
        by_task[task] = {
            "successes": ok,
            "trials": len(tr),
            "success_rate": ok / len(tr),
            "mean_success_steps": round(avg_steps, 1) if avg_steps is not None else None,
        }
    total_ok = sum(1 for r in subset if r["success"])
    avg_steps_all = (
        float(np.mean([r["steps"] for r in subset if r["success"]])) if total_ok > 0 else None
    )
    summary = {
        "mode": "e6_stage2_1_no_delta" if alpha_val == 0.0 else "e6_stage2_1",
        "prompt_mode": prompt_mode,
        "delta_scale": float(alpha_val),
        "checkpoint": str(CKPT_PATH.relative_to(REPO_ROOT)),
        "reused_from": str(source_csv.relative_to(REPO_ROOT)),
        "tasks": len(tasks),
        "trials": len(subset),
        "successes": total_ok,
        "success_rate": total_ok / len(subset),
        "mean_success_steps": round(avg_steps_all, 1) if avg_steps_all is not None else None,
        "by_task": by_task,
    }
    (out_dir / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _wlt(
    a_map: Dict[Tuple[str, int], Dict[str, Any]],
    b_map: Dict[Tuple[str, int], Dict[str, Any]],
    keys: List[Tuple[str, int]],
) -> Dict[str, Any]:
    """Compare `a` against baseline `b` (wins = a succeeded & b failed)."""
    wins = 0
    losses = 0
    ties_both_ok = 0
    ties_both_fail = 0
    a_ok = 0
    b_ok = 0
    for k in keys:
        sa = bool(a_map[k]["success"])
        sb = bool(b_map[k]["success"])
        a_ok += int(sa)
        b_ok += int(sb)
        if sa and not sb:
            wins += 1
        elif sb and not sa:
            losses += 1
        elif sa and sb:
            ties_both_ok += 1
        else:
            ties_both_fail += 1
    n = len(keys)
    return {
        "n": n,
        "a_successes": a_ok,
        "b_successes": b_ok,
        "delta_successes": a_ok - b_ok,
        "delta_sr": (a_ok - b_ok) / n,
        "delta_sr_pp": 100.0 * (a_ok - b_ok) / n,
        "wins": wins,
        "losses": losses,
        "ties": ties_both_ok + ties_both_fail,
        "ties_both_success": ties_both_ok,
        "ties_both_fail": ties_both_fail,
    }


def _outcome_tag(new_ok: bool, base_ok: bool) -> str:
    if new_ok and not base_ok:
        return "W"
    if base_ok and not new_ok:
        return "L"
    return "T1" if new_ok else "T0"


@torch.inference_mode()
def run_offline_eval(
    *,
    device: torch.device,
    batch_size: int = 128,
    val_seed: int = 12345,
) -> Dict[str, Any]:
    """Evaluate DDIM Action MSE, corrected latent RMS, Latent MSE, and cosine vs z_s
    for all 7 alphas on the exact same 6,248 val Cross samples (`val_seed=12345`)."""
    t0 = time.monotonic()
    policy, payload = load_stage2_1_policy(
        CKPT_PATH, stage1_ckpt=DEFAULT_STAGE1_CKPT, device=device
    )
    policy.eval()
    bank = Stage2_1TransferBank(device=device, resident_on_device=True, z_self_on_device=True)
    n_val = int(bank.val_valid_cross_mask.sum())
    s_z = float(policy.latent_scale_s_z.item())
    s_z2 = s_z * s_z

    gen_ddim = torch.Generator(device="cpu").manual_seed(val_seed + 777)
    init_noise_all = torch.randn((n_val, ACTION_HORIZON, ACTION_DIM_10D), generator=gen_ddim)

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    # Accumulators per alpha:
    # [count, mse10_sum, mae7_sum, mse7_sum, z_q_sq_sum, delta_scaled_sq_sum,
    #  latent_mse_raw_sum, cos_sum]
    alpha_keys = [k for k, _ in ALPHAS]
    acc = {k: torch.zeros(8, device=device, dtype=torch.float64) for k in alpha_keys}
    zs_sq_sum = torch.zeros(1, device=device, dtype=torch.float64)
    zc_sq_sum = torch.zeros(1, device=device, dtype=torch.float64)
    raw_delta_sq_sum = torch.zeros(1, device=device, dtype=torch.float64)
    total_count = 0

    it = bank.iter_val_batches_joint("cross_a", batch_size, rank=0, world_size=1, max_samples=n_val)
    cur = 0
    for dm_c, pm_c, _qm_c, _qpm_c, q_c, z_self, a_c, _ in it:
        B = len(dm_c)
        z0 = init_noise_all[cur : cur + B].to(device, non_blocking=True)
        cur += B
        total_count += B

        with amp_ctx:
            z_c, h_q = policy.read_cross_latent_from_memory(dm_c, pm_c, q_c)
            delta_raw = policy.residual_transformer(z_c, h_q)

        zc_f = z_c.float()
        dr_f = delta_raw.float()
        zs_f = z_self.float()
        gt10 = a_c.float()
        gt7 = action_10d_to_7d(gt10)

        zs_sq_sum += zs_f.square().mean(dim=(1, 2)).sum().double()
        zc_sq_sum += zc_f.square().mean(dim=(1, 2)).sum().double()
        raw_delta_sq_sum += dr_f.square().mean(dim=(1, 2)).sum().double()

        for a_str, a_val in ALPHAS:
            if a_val == 0.0:
                z_q = z_c
                d_scaled_f = torch.zeros_like(dr_f)
            elif a_val == 1.0:
                z_q = z_c + delta_raw
                d_scaled_f = dr_f
            else:
                z_q = z_c + (delta_raw * a_val)
                d_scaled_f = dr_f * a_val

            with amp_ctx:
                pred10 = policy.teacher.sample_from_z_self(z_q, init_noise=z0)

            p10 = pred10.float()
            p7 = action_10d_to_7d(p10)
            zq_f = z_q.float()

            mse10 = (p10 - gt10).square().mean(dim=(1, 2)).sum().double()
            mae7 = (p7 - gt7).abs().mean(dim=(1, 2)).sum().double()
            mse7 = (p7 - gt7).square().mean(dim=(1, 2)).sum().double()
            zq_sq = zq_f.square().mean(dim=(1, 2)).sum().double()
            ds_sq = d_scaled_f.square().mean(dim=(1, 2)).sum().double()
            lat_mse = (zq_f - zs_f).square().mean(dim=(1, 2)).sum().double()
            cos = F.cosine_similarity(
                zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1
            ).sum().double()

            acc[a_str][0] += B
            acc[a_str][1] += mse10
            acc[a_str][2] += mae7
            acc[a_str][3] += mse7
            acc[a_str][4] += zq_sq
            acc[a_str][5] += ds_sq
            acc[a_str][6] += lat_mse
            acc[a_str][7] += cos

    results: Dict[str, Any] = {
        "checkpoint": str(CKPT_PATH.relative_to(REPO_ROOT)),
        "epoch": payload.get("epoch"),
        "val_samples": int(total_count),
        "val_seed": int(val_seed),
        "latent_scale_s_z": s_z,
        "z_self_rms": float(math.sqrt(zs_sq_sum.item() / total_count)),
        "z_cross_rms": float(math.sqrt(zc_sq_sum.item() / total_count)),
        "raw_delta_rms": float(math.sqrt(raw_delta_sq_sum.item() / total_count)),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "by_alpha": {},
    }
    for a_str, a_val in ALPHAS:
        v = acc[a_str].cpu().tolist()
        n = v[0]
        lat_mse_raw = v[6] / n
        results["by_alpha"][a_str] = {
            "alpha": a_val,
            "ddim_action_10d_mse": v[1] / n,
            "ddim_action_7d_mae": v[2] / n,
            "ddim_action_7d_mse": v[3] / n,
            "corrected_latent_rms": math.sqrt(v[4] / n),
            "scaled_delta_rms": math.sqrt(v[5] / n),
            "latent_mse_raw": lat_mse_raw,
            "latent_loss_norm": lat_mse_raw / s_z2,
            "cos_vs_z_self": v[7] / n,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--offline-batch-size", type=int, default=128)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Determine canonical 331 paired keys from alpha=1.0 oracle_gt
    a1_ora_raw = _load_rollout_csv(EXISTING_SOURCES["1.0"]["oracle_gt"])
    canonical_keys = sorted(a1_ora_raw.keys())
    tasks = sorted({t for t, _ in canonical_keys})
    assert len(canonical_keys) == 331, f"Expected 331 paired trials, got {len(canonical_keys)}"
    assert len(tasks) == 9, f"Expected 9 tasks, got {len(tasks)}"

    # 2. Materialize reused alpha=0 and alpha=1.0 into alpha_<value>/{default,oracle_gt}
    for a_str in ("0", "1.0"):
        a_val = float(a_str)
        def_raw = _load_rollout_csv(EXISTING_SOURCES[a_str]["default"])
        ora_raw = _load_rollout_csv(EXISTING_SOURCES[a_str]["oracle_gt"])
        assert set(canonical_keys).issubset(def_raw.keys()), f"Missing keys in alpha={a_str} default"
        assert set(canonical_keys) == set(ora_raw.keys()), f"Key mismatch in alpha={a_str} oracle_gt"
        _write_subset_csv_and_summary(
            def_raw,
            canonical_keys,
            OUT_ROOT / f"alpha_{a_str}" / "default",
            alpha_val=a_val,
            prompt_mode="default_paired",
            source_csv=EXISTING_SOURCES[a_str]["default"],
        )
        _write_subset_csv_and_summary(
            ora_raw,
            canonical_keys,
            OUT_ROOT / f"alpha_{a_str}" / "oracle_gt",
            alpha_val=a_val,
            prompt_mode="oracle_gt",
            source_csv=EXISTING_SOURCES[a_str]["oracle_gt"],
        )

    # 3. Load all 7 alphas (Default & Oracle-GT) and validate exact 331-trial alignment
    def_maps: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    ora_maps: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    for a_str, _ in ALPHAS:
        def_csv = OUT_ROOT / f"alpha_{a_str}" / "default" / "rollout_results.csv"
        ora_csv = OUT_ROOT / f"alpha_{a_str}" / "oracle_gt" / "rollout_results.csv"
        if not def_csv.is_file() or not ora_csv.is_file():
            raise FileNotFoundError(
                f"Missing closed-loop CSV for alpha={a_str}: {def_csv} / {ora_csv}"
            )
        dm = _load_rollout_csv(def_csv)
        om = _load_rollout_csv(ora_csv)
        assert set(canonical_keys).issubset(dm.keys()), f"alpha={a_str} default missing trials"
        assert set(canonical_keys) == set(om.keys()), f"alpha={a_str} oracle_gt trial mismatch"
        # Verify identical donor assignments across all alphas
        for k in canonical_keys:
            expected_def_donor = f"demo_{k[1] % 10}"
            expected_ora_donor = f"oracle_gt/query_{k[1]:02d}"
            assert dm[k]["donor"] == expected_def_donor, f"Donor mismatch def {a_str} {k}: {dm[k]['donor']}"
            assert om[k]["donor"] == expected_ora_donor, f"Donor mismatch ora {a_str} {k}: {om[k]['donor']}"
        def_maps[a_str] = {k: dm[k] for k in canonical_keys}
        ora_maps[a_str] = {k: om[k] for k in canonical_keys}

    # 4. Optional offline evaluation
    offline: Optional[Dict[str, Any]] = None
    offline_json_path = OUT_ROOT / "offline_metrics.json"
    if not args.skip_offline:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[OfflineEval] Running on {device} across {len(ALPHAS)} alphas...", flush=True)
        offline = run_offline_eval(device=device, batch_size=args.offline_batch_size)
        offline_json_path.write_text(json.dumps(offline, indent=2) + "\n")
        with (OUT_ROOT / "offline_metrics.csv").open("w", newline="") as f:
            fields = [
                "alpha",
                "ddim_action_10d_mse",
                "ddim_action_7d_mae",
                "ddim_action_7d_mse",
                "corrected_latent_rms",
                "scaled_delta_rms",
                "latent_mse_raw",
                "latent_loss_norm",
                "cos_vs_z_self",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for a_str, _ in ALPHAS:
                w.writerow(offline["by_alpha"][a_str])
        print(
            f"[OfflineEval DONE] {offline['val_samples']} val samples in {offline['elapsed_seconds']}s",
            flush=True,
        )
    elif offline_json_path.is_file():
        offline = json.loads(offline_json_path.read_text())

    # 5. Compute comprehensive closed-loop summary across all 7 alphas
    summary_rows: List[Dict[str, Any]] = []
    by_alpha_summary: Dict[str, Any] = {}
    per_task_rows: List[Dict[str, Any]] = []

    base_def_a1 = def_maps["1.0"]
    base_ora_a1 = ora_maps["1.0"]

    for a_str, a_val in ALPHAS:
        dm = def_maps[a_str]
        om = ora_maps[a_str]

        def_ok = sum(1 for k in canonical_keys if dm[k]["success"])
        ora_ok = sum(1 for k in canonical_keys if om[k]["success"])
        def_sr = def_ok / len(canonical_keys)
        ora_sr = ora_ok / len(canonical_keys)
        def_steps = (
            float(np.mean([dm[k]["steps"] for k in canonical_keys if dm[k]["success"]]))
            if def_ok > 0
            else None
        )
        ora_steps = (
            float(np.mean([om[k]["steps"] for k in canonical_keys if om[k]["success"]]))
            if ora_ok > 0
            else None
        )

        ora_vs_def = _wlt(om, dm, canonical_keys)
        def_vs_a1 = _wlt(dm, base_def_a1, canonical_keys)
        ora_vs_a1 = _wlt(om, base_ora_a1, canonical_keys)

        per_task: Dict[str, Any] = {}
        for task in tasks:
            tkeys = [k for k in canonical_keys if k[0] == task]
            td_ok = sum(1 for k in tkeys if dm[k]["success"])
            to_ok = sum(1 for k in tkeys if om[k]["success"])
            td_steps = (
                float(np.mean([dm[k]["steps"] for k in tkeys if dm[k]["success"]]))
                if td_ok > 0
                else None
            )
            to_steps = (
                float(np.mean([om[k]["steps"] for k in tkeys if om[k]["success"]]))
                if to_ok > 0
                else None
            )
            t_ora_vs_def = _wlt(om, dm, tkeys)
            t_def_vs_a1 = _wlt(dm, base_def_a1, tkeys)
            t_ora_vs_a1 = _wlt(om, base_ora_a1, tkeys)
            per_task[task] = {
                "trials": len(tkeys),
                "default_successes": td_ok,
                "default_sr": td_ok / len(tkeys),
                "default_mean_success_steps": round(td_steps, 1) if td_steps is not None else None,
                "oracle_gt_successes": to_ok,
                "oracle_gt_sr": to_ok / len(tkeys),
                "oracle_gt_mean_success_steps": round(to_steps, 1) if to_steps is not None else None,
                "oracle_vs_default": t_ora_vs_def,
                "default_vs_alpha1": t_def_vs_a1,
                "oracle_vs_alpha1": t_ora_vs_a1,
            }
            per_task_rows.append(
                {
                    "alpha": a_str,
                    "task": task,
                    "task_short": TASK_SHORT_NAMES.get(task, task),
                    "trials": len(tkeys),
                    "default_successes": td_ok,
                    "default_sr_pct": round(100.0 * td_ok / len(tkeys), 2),
                    "default_mean_success_steps": round(td_steps, 1) if td_steps is not None else "",
                    "oracle_gt_successes": to_ok,
                    "oracle_gt_sr_pct": round(100.0 * to_ok / len(tkeys), 2),
                    "oracle_gt_mean_success_steps": round(to_steps, 1) if to_steps is not None else "",
                    "oracle_vs_default_dsr_pp": round(t_ora_vs_def["delta_sr_pp"], 2),
                    "oracle_vs_default_wlt": f"W{t_ora_vs_def['wins']}/L{t_ora_vs_def['losses']}/T{t_ora_vs_def['ties']}",
                }
            )

        off_entry = offline["by_alpha"][a_str] if offline is not None else None
        by_alpha_summary[a_str] = {
            "alpha": a_val,
            "reused": a_str in ("0", "1.0"),
            "trials": len(canonical_keys),
            "default_successes": def_ok,
            "default_sr": def_sr,
            "default_mean_success_steps": round(def_steps, 1) if def_steps is not None else None,
            "oracle_gt_successes": ora_ok,
            "oracle_gt_sr": ora_sr,
            "oracle_gt_mean_success_steps": round(ora_steps, 1) if ora_steps is not None else None,
            "oracle_vs_default": ora_vs_def,
            "default_vs_alpha1": def_vs_a1,
            "oracle_vs_alpha1": ora_vs_a1,
            "offline": off_entry,
            "by_task": per_task,
        }

        row = {
            "alpha": a_str,
            "reused": a_str in ("0", "1.0"),
            "trials": len(canonical_keys),
            "default_successes": def_ok,
            "default_sr_pct": round(100.0 * def_sr, 2),
            "default_mean_success_steps": round(def_steps, 1) if def_steps is not None else "",
            "oracle_gt_successes": ora_ok,
            "oracle_gt_sr_pct": round(100.0 * ora_sr, 2),
            "oracle_gt_mean_success_steps": round(ora_steps, 1) if ora_steps is not None else "",
            "oracle_vs_default_dsr_pp": round(ora_vs_def["delta_sr_pp"], 2),
            "oracle_vs_default_W": ora_vs_def["wins"],
            "oracle_vs_default_L": ora_vs_def["losses"],
            "oracle_vs_default_T": ora_vs_def["ties"],
            "default_vs_a1_dsr_pp": round(def_vs_a1["delta_sr_pp"], 2),
            "default_vs_a1_W": def_vs_a1["wins"],
            "default_vs_a1_L": def_vs_a1["losses"],
            "default_vs_a1_T": def_vs_a1["ties"],
            "oracle_vs_a1_dsr_pp": round(ora_vs_a1["delta_sr_pp"], 2),
            "oracle_vs_a1_W": ora_vs_a1["wins"],
            "oracle_vs_a1_L": ora_vs_a1["losses"],
            "oracle_vs_a1_T": ora_vs_a1["ties"],
            "offline_ddim_10d_mse": round(off_entry["ddim_action_10d_mse"], 6) if off_entry else "",
            "offline_ddim_7d_mae": round(off_entry["ddim_action_7d_mae"], 6) if off_entry else "",
            "offline_corrected_latent_rms": round(off_entry["corrected_latent_rms"], 4) if off_entry else "",
            "offline_latent_mse_raw": round(off_entry["latent_mse_raw"], 4) if off_entry else "",
            "offline_cos_vs_z_self": round(off_entry["cos_vs_z_self"], 5) if off_entry else "",
        }
        summary_rows.append(row)

    # 6. Write summary_table.csv, per_task_sr.csv, paired_trials.csv, summary.json
    with (OUT_ROOT / "summary_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with (OUT_ROOT / "per_task_sr.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()))
        w.writeheader()
        w.writerows(per_task_rows)

    paired_fields = ["task", "layout", "default_donor", "oracle_gt_donor"]
    for a_str, _ in ALPHAS:
        tag = a_str.replace(".", "_")
        paired_fields.extend(
            [
                f"def_a{tag}_ok",
                f"def_a{tag}_steps",
                f"ora_a{tag}_ok",
                f"ora_a{tag}_steps",
                f"ora_vs_def_a{tag}",
                f"def_a{tag}_vs_a1",
                f"ora_a{tag}_vs_a1",
            ]
        )
    with (OUT_ROOT / "paired_trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=paired_fields)
        w.writeheader()
        for k in canonical_keys:
            row: Dict[str, Any] = {
                "task": k[0],
                "layout": k[1],
                "default_donor": def_maps["1.0"][k]["donor"],
                "oracle_gt_donor": ora_maps["1.0"][k]["donor"],
            }
            d1_ok = bool(base_def_a1[k]["success"])
            o1_ok = bool(base_ora_a1[k]["success"])
            for a_str, _ in ALPHAS:
                tag = a_str.replace(".", "_")
                d_r = def_maps[a_str][k]
                o_r = ora_maps[a_str][k]
                d_ok = bool(d_r["success"])
                o_ok = bool(o_r["success"])
                row[f"def_a{tag}_ok"] = int(d_ok)
                row[f"def_a{tag}_steps"] = d_r["steps"]
                row[f"ora_a{tag}_ok"] = int(o_ok)
                row[f"ora_a{tag}_steps"] = o_r["steps"]
                row[f"ora_vs_def_a{tag}"] = _outcome_tag(o_ok, d_ok)
                row[f"def_a{tag}_vs_a1"] = _outcome_tag(d_ok, d1_ok)
                row[f"ora_a{tag}_vs_a1"] = _outcome_tag(o_ok, o1_ok)
            w.writerow(row)

    full_summary = {
        "experiment": "E6 Stage 2.1 Delta Scaling Sweep (CLIP)",
        "checkpoint": str(CKPT_PATH.relative_to(REPO_ROOT)),
        "alphas": [a for _, a in ALPHAS],
        "n_tasks": len(tasks),
        "n_paired_trials": len(canonical_keys),
        "exec_horizon": 12,
        "max_steps": 550,
        "warmup_steps": 12,
        "by_alpha": by_alpha_summary,
        "offline": offline,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(full_summary, indent=2) + "\n")

    # 7. Write REPORT.md
    lines: List[str] = []
    lines.append("# E6 Stage 2.1 — Delta Scaling Sweep (CLIP) 实验报告\n")
    lines.append("## 1. 实验设置\n")
    lines.append(f"- **模型 Checkpoint**：`{CKPT_PATH.relative_to(REPO_ROOT)}`（E6 Stage 2.1 B, Fixed Joint Best, Epoch 8 EMA），Cross Reader、ResidualTransformer 与 UNet 全部冻结。")
    lines.append("- **推理公式**：$z_q = z_c + \\alpha \\Delta z$（单步 correction，仅对已预测的 $\\Delta z$ 进行标量缩放，不使用多步 refinement）。")
    lines.append("- **扫描范围**：`alpha = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`（`alpha=0` 与 `alpha=1.0` 复用已有闭环结果，`0.5 / 0.75 / 1.25 / 1.5 / 2.0` 在物理 GPU 0 & 1 运行完整 331 paired trials 闭环评测）。")
    lines.append("- **严格配对设置**：Oracle-GT 对应的 **331 paired trials / 9 tasks**，相同的初态、Demo 映射（Default `demo_{layout%10}` vs Oracle-GT `oracle_gt/query_{layout:02d}`）、DDIM 噪声种子、12-step execution、12-step warmup 与 max 550 steps。\n")

    lines.append("## 2. 闭环总览对照表（331 Paired Trials，9 Tasks）\n")
    lines.append("| $\\alpha$ | Default SR (n/331) | Default 均值步数 | Oracle-GT SR (n/331) | Oracle-GT 均值步数 | Oracle vs Default ΔSR (W/L/T) | Default vs $\\alpha=1$ ΔSR (W/L/T) | Oracle vs $\\alpha=1$ ΔSR (W/L/T) |")
    lines.append("|---:|---:|---:|---:|---:|---|---|---|")
    for a_str, _ in ALPHAS:
        s = by_alpha_summary[a_str]
        ovd = s["oracle_vs_default"]
        dv1 = s["default_vs_alpha1"]
        ov1 = s["oracle_vs_alpha1"]
        d_st = f"{s['default_mean_success_steps']:.1f}" if s["default_mean_success_steps"] is not None else "—"
        o_st = f"{s['oracle_gt_mean_success_steps']:.1f}" if s["oracle_gt_mean_success_steps"] is not None else "—"
        lines.append(
            f"| **{a_str}** | {100*s['default_sr']:.2f}% ({s['default_successes']}/331) | {d_st} | "
            f"{100*s['oracle_gt_sr']:.2f}% ({s['oracle_gt_successes']}/331) | {o_st} | "
            f"{ovd['delta_sr_pp']:+.2f} pp (W{ovd['wins']}/L{ovd['losses']}/T{ovd['ties']}) | "
            f"{dv1['delta_sr_pp']:+.2f} pp (W{dv1['wins']}/L{dv1['losses']}/T{dv1['ties']}) | "
            f"{ov1['delta_sr_pp']:+.2f} pp (W{ov1['wins']}/L{ov1['losses']}/T{ov1['ties']}) |"
        )
    lines.append("")

    if offline is not None:
        lines.append("## 3. 离线指标（6,248 Val Cross 样本，val_seed=12345）\n")
        lines.append(
            f"> 参考：$z_s$ RMS = `{offline['z_self_rms']:.4f}`，$z_c$ RMS = `{offline['z_cross_rms']:.4f}`，"
            f"未缩放 $\\Delta z$ RMS = `{offline['raw_delta_rms']:.4f}`，固定 $s_z = {offline['latent_scale_s_z']:.6f}$。\n"
        )
        lines.append("| $\\alpha$ | DDIM 10D Action MSE | DDIM 7D Action MAE | Corrected Latent $z_q$ RMS | Scaled $\\alpha\\Delta z$ RMS | Raw Latent MSE vs $z_s$ | Norm $L_z$ | $\\cos(z_q, z_s)$ |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a_str, _ in ALPHAS:
            o = offline["by_alpha"][a_str]
            lines.append(
                f"| **{a_str}** | {o['ddim_action_10d_mse']:.6f} | {o['ddim_action_7d_mae']:.6f} | "
                f"{o['corrected_latent_rms']:.4f} | {o['scaled_delta_rms']:.4f} | "
                f"{o['latent_mse_raw']:.4f} | {o['latent_loss_norm']:.6f} | {o['cos_vs_z_self']:.5f} |"
            )
        lines.append("")

    lines.append("## 4. 逐任务 SR 对照（Default / Oracle-GT）\n")
    alpha_hdr = " / ".join(a for a, _ in ALPHAS)
    lines.append(f"### 4.1 Default Demo 逐任务 SR（%）（$\\alpha$: {alpha_hdr}）\n")
    lines.append("| 任务 | n | " + " | ".join(f"$\\alpha={a}$" for a, _ in ALPHAS) + " |")
    lines.append("|---|---:|" + "---:|" * len(ALPHAS))
    for task in tasks:
        n_t = by_alpha_summary["1.0"]["by_task"][task]["trials"]
        vals = " | ".join(f"{100*by_alpha_summary[a]['by_task'][task]['default_sr']:.1f}%" for a, _ in ALPHAS)
        lines.append(f"| {TASK_SHORT_NAMES.get(task, task)} | {n_t} | {vals} |")
    lines.append(
        "| **Total** | **331** | "
        + " | ".join(f"**{100*by_alpha_summary[a]['default_sr']:.2f}%**" for a, _ in ALPHAS)
        + " |\n"
    )

    lines.append(f"### 4.2 Oracle-GT Demo 逐任务 SR（%）（$\\alpha$: {alpha_hdr}）\n")
    lines.append("| 任务 | n | " + " | ".join(f"$\\alpha={a}$" for a, _ in ALPHAS) + " |")
    lines.append("|---|---:|" + "---:|" * len(ALPHAS))
    for task in tasks:
        n_t = by_alpha_summary["1.0"]["by_task"][task]["trials"]
        vals = " | ".join(f"{100*by_alpha_summary[a]['by_task'][task]['oracle_gt_sr']:.1f}%" for a, _ in ALPHAS)
        lines.append(f"| {TASK_SHORT_NAMES.get(task, task)} | {n_t} | {vals} |")
    lines.append(
        "| **Total** | **331** | "
        + " | ".join(f"**{100*by_alpha_summary[a]['oracle_gt_sr']:.2f}%**" for a, _ in ALPHAS)
        + " |\n"
    )

    # Diagnostic conclusion section
    ora_a1 = by_alpha_summary["1.0"]["oracle_gt_sr"]
    def_a1 = by_alpha_summary["1.0"]["default_sr"]
    best_lt1_a = max(["0", "0.5", "0.75"], key=lambda a: by_alpha_summary[a]["oracle_gt_sr"])
    best_gt1_a = max(["1.25", "1.5", "2.0"], key=lambda a: by_alpha_summary[a]["oracle_gt_sr"])
    best_all_ora_a = max([a for a, _ in ALPHAS], key=lambda a: by_alpha_summary[a]["oracle_gt_sr"])
    best_all_def_a = max([a for a, _ in ALPHAS], key=lambda a: by_alpha_summary[a]["default_sr"])

    lines.append("## 5. 核心诊断结论\n")
    lines.append(
        f"- **Default Demo 最优 $\\alpha$**：`alpha={best_all_def_a}` -> `{100*by_alpha_summary[best_all_def_a]['default_sr']:.2f}%` "
        f"({by_alpha_summary[best_all_def_a]['default_successes']}/331；对比 `alpha=1.0` 的 `{100*def_a1:.2f}%`, "
        f"{by_alpha_summary[best_all_def_a]['default_vs_alpha1']['delta_sr_pp']:+.2f} pp)。"
    )
    lines.append(
        f"- **Oracle-GT Demo 最优 $\\alpha$**：`alpha={best_all_ora_a}` -> `{100*by_alpha_summary[best_all_ora_a]['oracle_gt_sr']:.2f}%` "
        f"({by_alpha_summary[best_all_ora_a]['oracle_gt_successes']}/331；对比 `alpha=1.0` 的 `{100*ora_a1:.2f}%`, "
        f"{by_alpha_summary[best_all_ora_a]['oracle_vs_alpha1']['delta_sr_pp']:+.2f} pp)。"
    )
    lines.append(
        f"- **$\\alpha < 1$ 侧最优（检验修正过度）**：`alpha={best_lt1_a}` -> Oracle-GT `{100*by_alpha_summary[best_lt1_a]['oracle_gt_sr']:.2f}%` "
        f"({by_alpha_summary[best_lt1_a]['oracle_vs_alpha1']['delta_sr_pp']:+.2f} pp vs $\\alpha=1$, "
        f"W{by_alpha_summary[best_lt1_a]['oracle_vs_alpha1']['wins']}/L{by_alpha_summary[best_lt1_a]['oracle_vs_alpha1']['losses']}/T{by_alpha_summary[best_lt1_a]['oracle_vs_alpha1']['ties']})。"
    )
    lines.append(
        f"- **$\\alpha > 1$ 侧最优（检验修正不足）**：`alpha={best_gt1_a}` -> Oracle-GT `{100*by_alpha_summary[best_gt1_a]['oracle_gt_sr']:.2f}%` "
        f"({by_alpha_summary[best_gt1_a]['oracle_vs_alpha1']['delta_sr_pp']:+.2f} pp vs $\\alpha=1$, "
        f"W{by_alpha_summary[best_gt1_a]['oracle_vs_alpha1']['wins']}/L{by_alpha_summary[best_gt1_a]['oracle_vs_alpha1']['losses']}/T{by_alpha_summary[best_gt1_a]['oracle_vs_alpha1']['ties']})。"
    )

    lt1_gain_pp = by_alpha_summary[best_lt1_a]["oracle_vs_alpha1"]["delta_sr_pp"]
    gt1_gain_pp = by_alpha_summary[best_gt1_a]["oracle_vs_alpha1"]["delta_sr_pp"]
    if gt1_gain_pp > 1.5 and gt1_gain_pp > lt1_gain_pp:
        verdict = "放大 $\\Delta z$（$\\alpha > 1$）显著改善了 Oracle-GT，说明原始推理存在一定程度的修正不足（under-correction）。"
    elif lt1_gain_pp > 1.5 and lt1_gain_pp > gt1_gain_pp:
        verdict = "缩小 $\\Delta z$（$\\alpha < 1$）显著改善了 Oracle-GT，说明原始推理在 Oracle-GT 上存在修正过度（over-correction）。"
    else:
        verdict = (
            "在 $\\alpha < 1$ 和 $\\alpha > 1$ 两侧均未带来显著的 Oracle-GT 改善（峰值位于 $\\alpha \\approx 1$ 附近或微弱波动），"
            "说明 Oracle-GT 相对 Default 的落差**不是简单的标量修正幅度（过大或过小）问题**，而应指向 **correction direction（修正方向失配）或 reference representation（冻结 PromptPooler / Stage 1 参考表征的域偏移）问题**。"
        )
    lines.append(f"- **综合判断**：{verdict}\n")

    lines.append("## 6. 产物清单\n")
    lines.append("- 统一对照表：`e6/outputs/delta_scale_sweep/summary_table.csv`、`summary.json`")
    lines.append("- 逐局配对 CSV：`e6/outputs/delta_scale_sweep/paired_trials.csv`（331 trials × 7 alphas × Default/Oracle-GT）")
    lines.append("- 逐任务对照 CSV：`e6/outputs/delta_scale_sweep/per_task_sr.csv`")
    if offline is not None:
        lines.append("- 离线指标：`e6/outputs/delta_scale_sweep/offline_metrics.csv`、`offline_metrics.json`")
    lines.append("- 各 $\\alpha$ 独立目录：`e6/outputs/delta_scale_sweep/alpha_<value>/{default,oracle_gt}/`")

    (OUT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[Analyze DONE] Wrote summary_table.csv, paired_trials.csv, per_task_sr.csv, summary.json, REPORT.md -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
