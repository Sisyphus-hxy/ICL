"""Parallel LIBERO-Gen Goal Chain closed-loop evaluator for non-default demo swaps.

Matches the exact simulation setup, initial states, warmup, and per-(task, layout, replan)
DDIM noise seeds of `e5/eval_libero_chain.py`, while supporting:
  - shifted   : same-task disjoint demos demo_10..demo_19 (500 trials)
  - oracle_gt : same-layout GT demos from `bpp/experiments/oracle_gt` (331 trials across 9 tasks)
  - shuffle   : wrong-task demos from task (ti + 1) % 10 (500 trials)

Optionally compares against a default-demo `rollout_results.csv` to compute exact paired deltas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from e5.eval_libero_chain import (
    CHAIN_BDDL_DIR,
    CHAIN_INIT_DIR,
    CHAIN_SPLIT,
    DEFAULT_PROMPT_SLOTS,
    EXEC_HORIZON,
    MAX_STEPS,
    N_QUERIES,
    OPEN_GRIPPER,
    REPO_ROOT,
    WARMUP_STEPS,
    _atomic_write_json,
    _query_server,
    get_chain_tasks,
)

ORACLE_GT_EXPERTS_DIR = (
    REPO_ROOT / "bpp/experiments/oracle_gt/outputs/liberogen_goal_chain/experts"
)


def _run_shard(
    job: Tuple[int, str, List[int], str, str, str, int, int, Optional[int]]
) -> List[Dict[str, Any]]:
    task_idx, task, layouts, prompt_mode, sock_path, out_dir_s, max_steps, seed_base, egl_id = job
    if egl_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_id)
    torch.set_num_threads(1)

    from libero.libero.envs.env_wrapper import ControlEnv

    out_dir = Path(out_dir_s)
    bddl = CHAIN_BDDL_DIR / f"{task}.bddl"
    init_file = CHAIN_INIT_DIR / f"{task}.pruned_init"
    states = torch.load(str(init_file), weights_only=False, map_location="cpu")

    rows: List[Dict[str, Any]] = []
    todo: List[int] = []
    for qi in layouts:
        rpath = out_dir / task / f"layout_{qi:02d}" / "result.json"
        if rpath.is_file():
            rows.append(json.loads(rpath.read_text()))
        else:
            todo.append(qi)
    if not todo:
        return rows

    env = ControlEnv(
        str(bddl),
        camera_heights=128,
        camera_widths=128,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        hard_reset=False,
        ignore_done=True,
        control_freq=20,
    )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    try:
        if not env.robots[0].controller.use_delta:
            raise RuntimeError("Expected delta OSC_POSE controller")
        if hasattr(env.env, "visualize"):
            env.env.visualize({"robots": True, "env": False, "grippers": False})

        for qi in todo:
            t0 = time.monotonic()
            if prompt_mode == "oracle_gt":
                donor_key = qi
                donor_label = f"oracle_gt/query_{qi:02d}"
            elif prompt_mode == "default_paired":
                donor_key = qi % DEFAULT_PROMPT_SLOTS
                donor_label = f"demo_{donor_key}"
            elif prompt_mode == "shifted":
                donor_key = qi % DEFAULT_PROMPT_SLOTS
                donor_label = f"demo_{10 + donor_key}"
            else:  # shuffle
                donor_key = qi % DEFAULT_PROMPT_SLOTS
                donor_label = f"shuffle_demo_{donor_key}"

            state = np.asarray(states[qi])
            env.seed(7)
            env.env.reset()
            env.set_init_state(state)
            ctrl = env.robots[0].controller
            ctrl.update(force=True)
            ctrl.reset_goal()
            env.robots[0].gripper.current_action = np.zeros(2)
            obs = env.env._get_observations(force_update=True)
            history = [obs, obs]

            for _ in range(WARMUP_STEPS):
                obs, _, _, _ = env.step(OPEN_GRIPPER)
                history = [history[1], obs]

            steps = 0
            replan_idx = 0
            success = bool(env.check_success())
            while not success and steps < max_steps:
                step_seed = (
                    (seed_base * 10_000_000)
                    + (task_idx * 100_000)
                    + (qi * 100)
                    + replan_idx
                )
                chunk = _query_server(sock, task_idx, donor_key, step_seed, history)
                replan_idx += 1
                for a in chunk:
                    obs, _, _, _ = env.step(a)
                    history = [history[1], obs]
                    steps += 1
                    success = bool(env.check_success())
                    if success or steps >= max_steps:
                        break

            rec = {
                "name": f"{task}/layout_{qi:02d}/result",
                "task": task,
                "layout": qi,
                "donor": donor_label,
                "prompt_mode": prompt_mode,
                "success": bool(success),
                "steps": int(steps),
                "replans": int(replan_idx),
                "wall_seconds": round(time.monotonic() - t0, 2),
                "init_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            }
            _atomic_write_json(rec, out_dir / task / f"layout_{qi:02d}" / "result.json")
            rows.append(rec)

        n_ok = sum(1 for r in rows if r["success"])
        print(
            f"[DemoSwapEval:{prompt_mode}] {task} ({len(layouts)} layouts, done={len(todo)} new): "
            f"{n_ok}/{len(rows)} ({100.0 * n_ok / len(rows):.1f}%)",
            flush=True,
        )
    finally:
        sock.close()
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("default_paired", "shifted", "oracle_gt", "shuffle"),
        required=True,
    )
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-queries", type=int, default=N_QUERIES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--egl-device", type=int, default=None)
    args = parser.parse_args()

    all_tasks = get_chain_tasks()
    selected_tasks = args.tasks if args.tasks else all_tasks
    for t in selected_tasks:
        if t not in all_tasks:
            raise ValueError(f"Unknown Chain task: {t}")

    args.output.mkdir(parents=True, exist_ok=True)

    task_layouts: Dict[str, List[int]] = {}
    for task in selected_tasks:
        if args.prompt_mode in ("oracle_gt", "default_paired"):
            tdir = ORACLE_GT_EXPERTS_DIR / task
            ls = [
                qi
                for qi in range(args.max_queries)
                if (tdir / f"query_{qi:02d}" / "demo.hdf5").is_file()
            ]
            if ls:
                task_layouts[task] = ls
        else:
            task_layouts[task] = list(range(args.max_queries))

    active_tasks = [t for t in selected_tasks if t in task_layouts]
    splits_per_task = max(1, args.workers // max(1, len(active_tasks)))
    jobs = []
    for task in active_tasks:
        ti = all_tasks.index(task)
        ls = task_layouts[task]
        for s in range(splits_per_task):
            sub = ls[s::splits_per_task]
            if sub:
                jobs.append(
                    (
                        ti,
                        task,
                        sub,
                        args.prompt_mode,
                        str(args.socket),
                        str(args.output),
                        args.max_steps,
                        args.seed_base,
                        args.egl_device,
                    )
                )

    total_expected = sum(len(v) for v in task_layouts.values())
    t0 = time.monotonic()
    print(
        f"[DemoSwapEval] Starting mode={args.mode} prompt_mode={args.prompt_mode}: "
        f"{len(active_tasks)} tasks, {total_expected} trials | workers={min(args.workers, len(jobs))} "
        f"| egl={args.egl_device} -> {args.output}",
        flush=True,
    )

    ctx = multiprocessing.get_context("spawn")
    all_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=ctx) as pool:
        for shard_rows in pool.map(_run_shard, jobs):
            all_rows.extend(shard_rows)

    all_rows.sort(key=lambda r: (r["task"], r["layout"]))
    if len(all_rows) != total_expected:
        raise RuntimeError(f"Expected {total_expected} rows, got {len(all_rows)}")

    csv_path = args.output / "rollout_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "task",
                "layout",
                "donor",
                "prompt_mode",
                "success",
                "steps",
                "replans",
                "wall_seconds",
            ],
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # Optional paired comparison against default-demo baseline CSV
    base_map: Dict[Tuple[str, int], bool] = {}
    if args.baseline_csv is not None and args.baseline_csv.is_file():
        with args.baseline_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                base_map[(row["task"], int(row["layout"]))] = (
                    str(row["success"]).lower() == "true"
                )

    by_task: Dict[str, Any] = {}
    paired_base_ok = 0
    paired_new_ok = 0
    paired_n = 0
    wins = 0
    losses = 0
    for task in active_tasks:
        tr = [r for r in all_rows if r["task"] == task]
        ok = sum(1 for r in tr if r["success"])
        avg_steps_ok = (
            float(np.mean([r["steps"] for r in tr if r["success"]])) if ok > 0 else None
        )
        entry: Dict[str, Any] = {
            "successes": ok,
            "trials": len(tr),
            "success_rate": ok / len(tr),
            "mean_success_steps": round(avg_steps_ok, 1) if avg_steps_ok is not None else None,
        }
        if base_map:
            t_paired = [r for r in tr if (r["task"], int(r["layout"])) in base_map]
            if t_paired:
                b_ok = sum(1 for r in t_paired if base_map[(r["task"], int(r["layout"]))])
                n_ok = sum(1 for r in t_paired if r["success"])
                entry["default_sr_paired"] = b_ok / len(t_paired)
                entry["delta_sr_paired"] = (n_ok - b_ok) / len(t_paired)
                paired_base_ok += b_ok
                paired_new_ok += n_ok
                paired_n += len(t_paired)
                for r in t_paired:
                    b = base_map[(r["task"], int(r["layout"]))]
                    n = bool(r["success"])
                    if n and not b:
                        wins += 1
                    elif b and not n:
                        losses += 1
        by_task[task] = entry

    total_ok = sum(1 for r in all_rows if r["success"])
    elapsed = round(time.monotonic() - t0, 1)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "prompt_mode": args.prompt_mode,
        "delta_scale": float(args.delta_scale),
        "checkpoint": args.checkpoint,
        "split": CHAIN_SPLIT,
        "tasks": len(active_tasks),
        "trials": len(all_rows),
        "successes": total_ok,
        "success_rate": total_ok / len(all_rows),
        "max_steps": args.max_steps,
        "warmup_steps": WARMUP_STEPS,
        "exec_horizon": EXEC_HORIZON,
        "elapsed_seconds": elapsed,
        "by_task": by_task,
    }
    if paired_n > 0:
        summary["paired_vs_default"] = {
            "baseline_csv": str(args.baseline_csv),
            "n_paired": paired_n,
            "default_successes": paired_base_ok,
            "default_sr": paired_base_ok / paired_n,
            "new_successes": paired_new_ok,
            "new_sr": paired_new_ok / paired_n,
            "delta_sr": (paired_new_ok - paired_base_ok) / paired_n,
            "wins": wins,
            "losses": losses,
            "ties": paired_n - wins - losses,
        }

    _atomic_write_json(summary, args.output / "test_summary.json")
    print(
        f"\n[DemoSwapEval DONE] mode={args.mode} prompt={args.prompt_mode} | "
        f"{total_ok}/{len(all_rows)} ({100.0 * total_ok / len(all_rows):.2f}%) in {elapsed:.1f}s "
        f"-> {args.output / 'test_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
