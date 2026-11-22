"""Parallel LIBERO-Gen Goal Chain (10 tasks x 50 layouts = 500 trials) closed-loop evaluator.

Runs inside `env/libero/.conda` (MuJoCo + robosuite + LIBERO + EGL) and queries
`e5/policy_server.py` (GPU BF16 batched inference) over a local Unix domain socket.

Protocol matches official BPP `bpp/experiments/oracle_gt/evaluate.py` & `e4_chain/eval_chain.py`:
  - 10 unseen tasks in `libero_goal_chain_selected_view`, 50 `.pruned_init` layouts each = 500
  - Official default prompt assignment: layout `qi` -> `demo_{qi % 10}`
  - 128x128 dual cameras (`agentview`, `robot0_eye_in_hand`), `control_freq=20`, delta OSC
  - Reset: `seed(7)` -> `env.reset()` -> `set_init_state(states[qi])` -> controller/gripper reset
  - Warmup: 12 steps of `[0, 0, 0, 0, 0, 0, -1.0]` (open gripper)
  - Closed-loop: 12-step execution horizon, max 550 steps, immediate stop on `check_success()`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import socket
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAIN_SPLIT = "libero_goal_chain_selected_view"
CHAIN_BDDL_DIR = (
    REPO_ROOT
    / "bpp/src/behavior_prompting/train_network/env/libero/bddl_files"
    / CHAIN_SPLIT
)
CHAIN_INIT_DIR = (
    REPO_ROOT
    / "bpp/src/behavior_prompting/train_network/env/libero/init_files"
    / CHAIN_SPLIT
)

N_QUERIES = 50
DEFAULT_PROMPT_SLOTS = 10
WARMUP_STEPS = 12
EXEC_HORIZON = 12
MAX_STEPS = 550
OPEN_GRIPPER = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

REQ_HDR = struct.Struct("<iiii")
RESP_BYTES = EXEC_HORIZON * 7 * 4


def get_chain_tasks() -> List[str]:
    tasks = sorted(p.stem for p in CHAIN_BDDL_DIR.glob("*.bddl"))
    if len(tasks) != 10:
        raise RuntimeError(f"Expected 10 Chain tasks under {CHAIN_BDDL_DIR}, found {len(tasks)}")
    return tasks


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(mv[got:], n - got)
        if k == 0:
            raise EOFError(f"Policy server closed socket ({got}/{n} bytes)")
        got += k
    return bytes(buf)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False, suffix=".tmp") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def _obs_to_proprio_11d(obs: Dict[str, np.ndarray]) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    rot6d = (
        Rotation.from_quat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
        .as_matrix()[:2, :]
        .reshape(6)
        .astype(np.float32)
    )
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate([pos, rot6d, grip], axis=0)


def _query_server(
    sock: socket.socket,
    task_idx: int,
    donor_idx: int,
    seed: int,
    history: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    h0, h1 = history[0], history[1]
    rgb = np.stack(
        [
            np.stack([h0["agentview_image"], h0["robot0_eye_in_hand_image"]], axis=0),
            np.stack([h1["agentview_image"], h1["robot0_eye_in_hand_image"]], axis=0),
        ],
        axis=0,
    )
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    prop = np.ascontiguousarray(
        np.stack([_obs_to_proprio_11d(h0), _obs_to_proprio_11d(h1)], axis=0),
        dtype=np.float32,
    )
    payload = REQ_HDR.pack(task_idx, donor_idx, seed, 0) + rgb.tobytes() + prop.tobytes()
    sock.sendall(payload)
    resp = _recv_exact(sock, RESP_BYTES)
    return np.frombuffer(resp, dtype=np.float32).reshape(EXEC_HORIZON, 7).copy()


def _run_shard(job: Tuple[int, str, List[int], str, str, int, int, Optional[int]]) -> List[Dict[str, Any]]:
    task_idx, task, layouts, sock_path, out_dir_s, max_steps, seed_base, egl_id = job
    if egl_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_id)
    torch.set_num_threads(1)

    from libero.libero.envs.env_wrapper import ControlEnv

    out_dir = Path(out_dir_s)
    bddl = CHAIN_BDDL_DIR / f"{task}.bddl"
    init_file = CHAIN_INIT_DIR / f"{task}.pruned_init"
    states = torch.load(str(init_file), weights_only=False, map_location="cpu")
    if len(states) != N_QUERIES:
        raise RuntimeError(f"Expected {N_QUERIES} init states for {task}, got {len(states)}")

    # Load already-completed layouts first so resumed runs skip MuJoCo if shard is done
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
            donor = qi % DEFAULT_PROMPT_SLOTS
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
                # Deterministic seed per (task, layout, replan) independent of worker scheduling
                step_seed = (seed_base * 10_000_000) + (task_idx * 100_000) + (qi * 100) + replan_idx
                chunk = _query_server(sock, task_idx, donor, step_seed, history)
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
                "donor": f"demo_{donor}",
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
            f"[ChainEval] {task} ({len(layouts)} layouts, done={len(todo)} new): "
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
    parser.add_argument("--checkpoint", type=str, default="")
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
    layouts_all = list(range(args.max_queries))

    # Split each task across sub-shards when workers > len(selected_tasks)
    splits_per_task = max(1, args.workers // len(selected_tasks))
    jobs = []
    for task in selected_tasks:
        ti = all_tasks.index(task)
        for s in range(splits_per_task):
            sub = layouts_all[s::splits_per_task]
            if sub:
                jobs.append(
                    (
                        ti,
                        task,
                        sub,
                        str(args.socket),
                        str(args.output),
                        args.max_steps,
                        args.seed_base,
                        args.egl_device,
                    )
                )

    t0 = time.monotonic()
    print(
        f"[ChainEval] Starting {args.mode}: {len(selected_tasks)} tasks x {len(layouts_all)} layouts = "
        f"{len(selected_tasks) * len(layouts_all)} trials | workers={min(args.workers, len(jobs))} "
        f"| egl={args.egl_device} -> {args.output}",
        flush=True,
    )
    ctx = multiprocessing.get_context("spawn")
    all_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=ctx) as pool:
        for shard_rows in pool.map(_run_shard, jobs):
            all_rows.extend(shard_rows)

    all_rows.sort(key=lambda r: (r["task"], r["layout"]))
    expected = len(selected_tasks) * len(layouts_all)
    if len(all_rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(all_rows)}")

    # Write CSV & summary JSON
    csv_path = args.output / "rollout_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "task", "layout", "donor", "success", "steps", "replans", "wall_seconds"],
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    by_task: Dict[str, Any] = {}
    for task in selected_tasks:
        tr = [r for r in all_rows if r["task"] == task]
        ok = sum(1 for r in tr if r["success"])
        avg_steps_ok = (
            float(np.mean([r["steps"] for r in tr if r["success"]])) if ok > 0 else None
        )
        by_task[task] = {
            "successes": ok,
            "trials": len(tr),
            "success_rate": ok / len(tr),
            "mean_success_steps": round(avg_steps_ok, 1) if avg_steps_ok is not None else None,
        }

    total_ok = sum(1 for r in all_rows if r["success"])
    elapsed = round(time.monotonic() - t0, 1)
    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "split": CHAIN_SPLIT,
        "tasks": len(selected_tasks),
        "queries_per_task": len(layouts_all),
        "trials": len(all_rows),
        "successes": total_ok,
        "success_rate": total_ok / len(all_rows),
        "max_steps": args.max_steps,
        "warmup_steps": WARMUP_STEPS,
        "exec_horizon": EXEC_HORIZON,
        "prompt_assignment": "layout % 10",
        "elapsed_seconds": elapsed,
        "by_task": by_task,
    }
    _atomic_write_json(summary, args.output / "test_summary.json")
    print(
        f"\n[ChainEval DONE] mode={args.mode} | {total_ok}/{len(all_rows)} "
        f"({100.0 * total_ok / len(all_rows):.2f}%) in {elapsed:.1f}s -> {args.output / 'test_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
