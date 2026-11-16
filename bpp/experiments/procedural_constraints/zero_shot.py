#!/usr/bin/env python3
"""Zero-shot procedure prompt swaps on the already generated demonstrations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BPP_ROOT = PROJECT / "bpp"
DATA = HERE / "data" / "minimal"
CLEARANCE = HERE.parent / "clearance"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["rollouts", "teacher", "analyze", "all"])
    p.add_argument("--output", type=Path, default=HERE / "data" / "zero_shot")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--checkpoint", type=Path,
                   default=BPP_ROOT / "checkpoints" / "liberogen_spatial_combination_behavior_prompting.ckpt")
    p.add_argument("--gpu", default="5", help="CUDA_VISIBLE_DEVICES value or GPU UUID")
    p.add_argument("--egl-device", type=int, default=0)
    p.add_argument("--query-seeds", default="0,1")
    p.add_argument("--layouts", default="0,1,2,3,4")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--skip-teacher", action="store_true")
    return p.parse_args()


def seed_all(seed: int, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_environment(args):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["PYNPUT_BACKEND"] = "dummy"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def import_runtime():
    sys.path.insert(0, str(CLEARANCE))
    from inference import load_policy, load_prompt, query_observations
    from recording import Recorder
    from scene import Scene
    import scene as clearance_scene
    sys.path.insert(0, str(HERE))
    from generate import (GENTLE_LAYOUT_POSITIONS, OBSTACLE_POSITIONS,
                          set_layout)
    from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
    import h5py
    import torch
    from scipy.spatial.transform import Rotation
    return {
        "load_policy": load_policy, "load_prompt": load_prompt,
        "query_observations": query_observations, "Recorder": Recorder,
        "Scene": Scene, "clearance_scene": clearance_scene,
        "GENTLE_LAYOUT_POSITIONS": GENTLE_LAYOUT_POSITIONS,
        "OBSTACLE_POSITIONS": OBSTACLE_POSITIONS, "set_layout": set_layout,
        "RotationTransformer": RotationTransformer, "h5py": h5py,
        "torch": torch, "Rotation": Rotation,
    }


def checkpoint_info(path: Path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def task_specs(runtime):
    return {
        "obstacle_clearance": {
            "procedures": ("narrow", "wide"),
            "positions": runtime["OBSTACLE_POSITIONS"],
            "prompt_trace": "narrow",
        },
        "gentle_transport": {
            "procedures": ("normal", "gentle"),
            "positions": runtime["GENTLE_LAYOUT_POSITIONS"],
            "prompt_trace": "normal",
        },
    }


def hdf5_name(data: Path, task: str, layout: int, procedure: str):
    return data / task / f"layout_{layout:02d}" / f"{procedure}.hdf5"


def annotate_rollout_hdf5(path: Path, metadata: dict, h5py):
    with h5py.File(path, "r+") as handle:
        demo = handle["data/demo_0"]
        for key, value in metadata.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            demo.attrs[key] = value


def stage_metrics(trajectory: Path, query_initial_height: float,
                  goal_xy: np.ndarray, obstacle: np.ndarray):
    data = np.genfromtxt(trajectory, delimiter=",", names=True, dtype=None, encoding="utf8")
    phases = np.asarray(data["phase"], dtype=str)
    eef = np.stack([data[key] for key in ("eef_x", "eef_y", "eef_z")], axis=1)
    obj = np.stack([data[key] for key in ("object_x", "object_y", "object_z")], axis=1)
    dt = 1.0 / 20.0
    lifted = obj[:, 2] > query_initial_height + .035
    xy_error = np.linalg.norm(obj[:, :2] - goal_xy[None, :], axis=1)
    grasp_success = bool(lifted.any())
    transport_mask = phases == "transport"
    transport_success = bool(np.any(transport_mask & lifted & (xy_error < .045)))
    goal_success = bool(data["goal_success"][-1])
    collision = bool(np.any(data["collision"] > 0))
    if goal_success:
        failure_stage = "none"
    elif not grasp_success:
        failure_stage = "grasp"
    elif not transport_success:
        failure_stage = "transport"
    else:
        failure_stage = "place"
    if transport_mask.any():
        tp = eef[transport_mask]
        travel = np.linalg.norm(np.diff(np.vstack([eef[0], tp]), axis=0), axis=1)
        # The first segment in this construction can include approach to the
        # transport label; use adjacent transport positions for dynamics.
        velocity = np.diff(tp, axis=0) / dt if len(tp) > 1 else np.zeros((0, 3))
        speed = np.linalg.norm(velocity, axis=1) if len(velocity) else np.zeros(1)
        acceleration = np.diff(velocity, axis=0) / dt if len(velocity) > 1 else np.zeros((0, 3))
        acc_norm = np.linalg.norm(acceleration, axis=1) if len(acceleration) else np.zeros(1)
        jerk = np.diff(acceleration, axis=0) / dt if len(acceleration) > 1 else np.zeros((0, 3))
        jerk_norm = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros(1)
        transport_path = float(np.linalg.norm(np.diff(tp, axis=0), axis=1).sum())
        transport_duration = float(len(tp) * dt)
        clearance = float(np.min(data["object_clearance_m"][transport_mask]))
        max_tilt = float(np.max(data["object_tilt_deg"][transport_mask])) if "object_tilt_deg" in data.dtype.names else float("nan")
    else:
        speed = acc_norm = jerk_norm = np.zeros(1)
        transport_path = transport_duration = clearance = max_tilt = float("nan")
    return {
        "goal_success": goal_success,
        "grasp_success": grasp_success,
        "transport_success": transport_success,
        "place_success": goal_success,
        "failure_stage": failure_stage,
        "collision": collision,
        "transport_min_clearance_m": clearance,
        "transport_path_length_m": transport_path,
        "transport_duration_s": transport_duration,
        "eef_speed_mean_m_s": float(speed.mean()),
        "eef_speed_peak_m_s": float(speed.max()),
        "eef_acceleration_peak_m_s2": float(acc_norm.max()),
        "eef_jerk_peak_m_s3": float(jerk_norm.max()),
        "max_object_tilt_deg": max_tilt,
        "control_steps": int(len(data)),
        "duration_s": float(len(data) * dt),
        "object_goal_final_xy_error_m": float(xy_error[-1]),
        "obstacle_position": obstacle.tolist(),
    }


def rollout_one(model, cfg, runtime, scene, output: Path, task: str,
                query_layout: int, source_layout: int, procedure: str,
                query_seed: int, prompt_path: Path, obstacle: np.ndarray,
                max_steps: int, device: str):
    torch = runtime["torch"]
    RotationTransformer = runtime["RotationTransformer"]
    Recorder = runtime["Recorder"]
    set_layout = runtime["set_layout"]
    set_layout(scene, f"layout_{query_layout:02d}", obstacle)
    query_initial_height = float(scene.object_pos()[2])
    goal_xy = scene.goal_pos()[:2].copy()
    model.reset(action_exec_horizon=12)
    prompt = runtime["load_prompt"](prompt_path, cfg.task.shape_meta, device)
    with torch.inference_mode():
        model.prompt(prompt)
    seed_all(query_seed, torch)
    condition_dir = output / task / f"layout_{query_layout:02d}" / f"seed{query_seed}"
    name = f"{task}/layout_{query_layout:02d}/seed{query_seed}/{procedure}"
    rec = Recorder(condition_dir, procedure, save_images=True)
    rec.start(scene)
    initial_identity = rec.identity
    history = [rec.latest_obs]
    inverse = RotationTransformer("axis_angle", "rotation_6d")
    started = time.monotonic()
    while len(rec.actions) < max_steps:
        query = runtime["query_observations"](history, cfg.task.shape_meta, device)
        with torch.inference_mode():
            prediction = model.predict_action(query)["action"][0, :12].detach().cpu().numpy()
        actions = np.concatenate([prediction[:, :3], inverse.inverse(prediction[:, 3:9]), prediction[:, 9:]], axis=-1)
        for action in actions:
            obj = scene.object_pos()
            transporting = bool(obj[2] > query_initial_height + .035 and
                                np.linalg.norm(obj[:2] - goal_xy) > .045)
            if obj[2] <= query_initial_height + .035:
                rec.phase = "grasp"
            elif transporting:
                rec.phase = "transport"
            else:
                rec.phase = "place"
            rec.step(action)
            # Add object orientation to the existing rollout recorder's CSV
            # row so Gentle tilt is measured from the real simulation state.
            q_mj = scene.d.xquat[scene.object_body]
            q_xyzw = np.r_[q_mj[1:], q_mj[0]]
            tilt = np.degrees(np.arccos(np.clip(runtime["Rotation"].from_quat(q_xyzw).as_matrix()[2, 2], -1., 1.)))
            rec.rows[-1]["object_tilt_deg"] = float(tilt)
            history.append(rec.latest_obs)
            history = history[-2:]
            if scene.env._check_success() or len(rec.actions) >= max_steps:
                break
        if len(rec.actions) % 120 == 0:
            print(name, "step", len(rec.actions), "elapsed", round(time.monotonic() - started, 1), flush=True)
        if scene.env._check_success():
            break
    summary = rec.finish(scene)
    metrics = stage_metrics(condition_dir / f"{procedure}_trajectory.csv", query_initial_height, goal_xy, obstacle)
    metadata = {
        "experiment": "zero_shot_procedure_prompt_swap",
        "task_id": task,
        "procedure_id": procedure,
        "prompt_id": str(prompt_path.relative_to(prompt_path.parents[2]).with_suffix("")),
        "query_layout_id": f"layout_{query_layout:02d}",
        "prompt_layout_id": f"layout_{source_layout:02d}",
        "query_seed": int(query_seed),
        "diffusion_seed": int(query_seed),
        "query_obstacle_position": obstacle.tolist(),
        "prompt_is_cross_layout": True,
        "procedure_label_provided_to_model": False,
        "checkpoint_task": str(cfg.task.name),
        "observation_alignment": "post_action",
        "wall_seconds": time.monotonic() - started,
    }
    annotate_rollout_hdf5(condition_dir / f"{procedure}.hdf5", metadata, runtime["h5py"])
    result = {**metadata, **summary, **metrics,
              "name": name, "initial_identity": initial_identity}
    return result


def write_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def run_rollouts(args, runtime, model, cfg, device, output: Path):
    import h5py
    torch = runtime["torch"]
    output.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    layouts = [int(x) for x in args.layouts.split(",") if x.strip()]
    seeds = [int(x) for x in args.query_seeds.split(",") if x.strip()]
    specs = task_specs(runtime)
    scene = runtime["Scene"](record_images=True)
    scene.canonical_state = np.load(data / "initial_state.npz")["state"]
    rows = []
    pair_identity = {}
    try:
        for task, spec in specs.items():
            for query_layout in layouts:
                obstacle = np.asarray(spec["positions"][query_layout], dtype=np.float64)
                source_layout = (query_layout + 1) % 5
                for query_seed in seeds:
                    pair_key = f"{task}/layout_{query_layout:02d}/seed_{query_seed}"
                    pair_identity[pair_key] = None
                    for procedure in spec["procedures"]:
                        prompt_path = hdf5_name(data, task, source_layout, procedure)
                        if not prompt_path.is_file():
                            raise FileNotFoundError(prompt_path)
                        result = rollout_one(model, cfg, runtime, scene, output, task,
                                             query_layout, source_layout, procedure,
                                             query_seed, prompt_path, obstacle,
                                             args.max_steps, device)
                        identity = result["initial_identity"]
                        if pair_identity[pair_key] is None:
                            pair_identity[pair_key] = identity
                        elif identity != pair_identity[pair_key]:
                            raise RuntimeError(f"query initial state changed within pair: {pair_key}")
                        rows.append(result)
                        write_rows(output / "rollout_results.csv", rows)
                        write_rows(output / "stage_results.csv", rows)
                        print(json.dumps({k: result.get(k) for k in (
                            "name", "goal_success", "grasp_success", "transport_success",
                            "place_success", "transport_min_clearance_m",
                            "transport_duration_s", "eef_speed_peak_m_s",
                            "eef_acceleration_peak_m_s2", "eef_jerk_peak_m_s3",
                            "max_object_tilt_deg")}, indent=2), flush=True)
    finally:
        scene.close()
    return rows


def teacher_obs(path: Path, runtime, h5py):
    Rotation = runtime["Rotation"]
    with h5py.File(path, "r") as handle:
        d = handle["data/demo_0"]
        obs = d["obs"]
        n = len(obs["ee_pos"])
        records = []
        for index in range(n):
            records.append({
                "robot0_eef_pos": obs["ee_pos"][index].copy(),
                "robot0_eef_quat": Rotation.from_rotvec(obs["ee_ori"][index]).as_quat(),
                "robot0_gripper_qpos": obs["gripper_states"][index].copy(),
                "agentview_image": obs["agentview_rgb"][index].copy(),
                "robot0_eye_in_hand_image": obs["eye_in_hand_rgb"][index].copy(),
            })
        phases = np.asarray(d["phases"][:])
    return records, phases


def teacher_trace(model, cfg, runtime, prompt_path: Path, query_path: Path,
                  query_indices: np.ndarray, device: str, seed: int):
    torch = runtime["torch"]
    inverse = runtime["RotationTransformer"]("axis_angle", "rotation_6d")
    records, _ = teacher_obs(query_path, runtime, runtime["h5py"])
    prompt = runtime["load_prompt"](prompt_path, cfg.task.shape_meta, device)
    model.reset(action_exec_horizon=12)
    with torch.inference_mode():
        model.prompt(prompt)
    seed_all(seed, torch)
    chunks = []
    for index in query_indices:
        history = [records[max(0, int(index) - 1)], records[int(index)]]
        query = runtime["query_observations"](history, cfg.task.shape_meta, device)
        with torch.inference_mode():
            native = model.predict_action(query)["action"][0, :12].detach().cpu().numpy()
        converted = np.concatenate([native[:, :3], inverse.inverse(native[:, 3:9]), native[:, 9:]], axis=-1)
        chunks.append({"query_step": int(index), "native": native, "action7": converted})
    return chunks


def teacher_forced(args, runtime, model, cfg, device, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    layouts = [int(x) for x in args.layouts.split(",") if x.strip()]
    seeds = [int(x) for x in args.query_seeds.split(",") if x.strip()]
    specs = task_specs(runtime)
    summary_rows = []
    for task, spec in specs.items():
        for query_layout in layouts:
            source_layout = (query_layout + 1) % 5
            query_path = hdf5_name(args.data.resolve(), task, query_layout, spec["prompt_trace"])
            with runtime["h5py"].File(query_path, "r") as handle:
                phases = np.asarray(handle["data/demo_0/phases"][:])
            transport_indices = np.where(phases == b"transport")[0]
            # Include a transport sample every action chunk and the last one.
            indices = np.unique(np.r_[transport_indices[::12], transport_indices[-1]])
            for seed in seeds:
                p_a = hdf5_name(args.data.resolve(), task, source_layout, spec["procedures"][0])
                p_b = hdf5_name(args.data.resolve(), task, source_layout, spec["procedures"][1])
                a1 = teacher_trace(model, cfg, runtime, p_a, query_path, indices, device, seed)
                a2 = teacher_trace(model, cfg, runtime, p_b, query_path, indices, device, seed)
                repeat = teacher_trace(model, cfg, runtime, p_a, query_path, indices, device, seed)
                repeat_b = teacher_trace(model, cfg, runtime, p_b, query_path, indices, device, seed)
                native_diffs = np.asarray([np.linalg.norm(x["native"] - y["native"], axis=1).mean() for x, y in zip(a1, a2)])
                action7_diffs = np.asarray([np.linalg.norm(x["action7"] - y["action7"], axis=1).mean() for x, y in zip(a1, a2)])
                repeat_a_error = max(float(np.max(np.abs(x["native"] - y["native"]))) for x, y in zip(a1, repeat))
                repeat_b_error = max(float(np.max(np.abs(x["native"] - y["native"]))) for x, y in zip(a2, repeat_b))
                rows = []
                for index, x, y, nd, d7 in zip(indices, a1, a2, native_diffs, action7_diffs):
                    rows.append({"task_id": task, "query_layout_id": f"layout_{query_layout:02d}",
                                 "prompt_layout_id": f"layout_{source_layout:02d}",
                                 "query_seed": seed, "diffusion_seed": seed,
                                 "query_step": int(index), "query_phase": "transport",
                                 "native_chunk_mean_l2": float(nd), "action7_chunk_mean_l2": float(d7),
                                 "prompt_a": spec["procedures"][0], "prompt_b": spec["procedures"][1]})
                seed_dir = output / task / f"layout_{query_layout:02d}" / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                write_rows(seed_dir / "teacher_forced.csv", rows)
                np.savez_compressed(seed_dir / "teacher_forced.npz",
                                    query_steps=indices,
                                    prompt_a=np.asarray([x["native"] for x in a1]),
                                    prompt_b=np.asarray([x["native"] for x in a2]),
                                    repeat_a=np.asarray([x["native"] for x in repeat]),
                                    repeat_b=np.asarray([x["native"] for x in repeat_b]))
                summary_rows.append({"task_id": task, "query_layout_id": f"layout_{query_layout:02d}",
                                     "prompt_layout_id": f"layout_{source_layout:02d}",
                                     "query_seed": seed, "diffusion_seed": seed,
                                     "transport_query_chunks": len(indices),
                                     "native_diff_mean": float(native_diffs.mean()),
                                     "native_diff_peak": float(native_diffs.max()),
                                     "action7_diff_mean": float(action7_diffs.mean()),
                                     "action7_diff_peak": float(action7_diffs.max()),
                                     "same_prompt_repeat_a_max_abs": repeat_a_error,
                                     "same_prompt_repeat_b_max_abs": repeat_b_error,
                                     "prompt_sensitive": bool(native_diffs.max() > 1e-5),
                                     "deterministic_repeat": bool(repeat_a_error < 1e-5 and repeat_b_error < 1e-5),
                                     "prompt_a": spec["procedures"][0], "prompt_b": spec["procedures"][1]})
                write_rows(output / "teacher_forced_summary.csv", summary_rows)
                print(json.dumps(summary_rows[-1], indent=2), flush=True)
    make_teacher_plots(output, summary_rows)
    return summary_rows


def make_teacher_plots(output: Path, rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in sorted({row["task_id"] for row in rows}):
        subset = [row for row in rows if row["task_id"] == task]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for row in subset:
            label = f"q{row['query_layout_id'][-2:]} s{row['query_seed']}"
            axes[0].scatter(row["query_layout_id"], row["native_diff_mean"], label=label)
            axes[1].scatter(row["query_layout_id"], row["action7_diff_mean"], label=label)
        axes[0].set(title=f"{task}: teacher-forced native action difference", ylabel="mean chunk L2")
        axes[1].set(title=f"{task}: teacher-forced 7D action difference", ylabel="mean chunk L2")
        for ax in axes:
            ax.set_xlabel("query layout")
            ax.tick_params(axis="x", rotation=45)
        axes[0].legend(fontsize=6, ncol=2)
        figure.tight_layout()
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(task_dir / "teacher_forced_diff.png", dpi=160)
        plt.close(figure)


def analyze_rollouts(args, output: Path):
    path = output / "rollout_results.csv"
    if not path.is_file():
        return
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ("goal_success", "grasp_success", "transport_success", "place_success", "collision"):
            row[key] = row[key] == "True"
        for key in ("transport_min_clearance_m", "transport_path_length_m", "transport_duration_s",
                    "eef_speed_peak_m_s", "eef_acceleration_peak_m_s2", "eef_jerk_peak_m_s3",
                    "max_object_tilt_deg", "object_goal_final_xy_error_m"):
            try: row[key] = float(row[key])
            except (ValueError, TypeError): row[key] = float("nan")
        row["query_layout_id"] = str(row["query_layout_id"])
        row["query_seed"] = int(row["query_seed"])
    write_rows(path, rows)
    write_rows(output / "stage_results.csv", [
        {key: row.get(key) for key in (
            "name", "task_id", "procedure_id", "query_layout_id", "prompt_layout_id",
            "query_seed", "diffusion_seed", "grasp_success", "transport_success",
            "place_success", "failure_stage", "goal_success", "collision")}
        for row in rows
    ])
    pairs = []
    for task in ("obstacle_clearance", "gentle_transport"):
        procedures = ("narrow", "wide") if task == "obstacle_clearance" else ("normal", "gentle")
        keys = sorted({(row["query_layout_id"], row["query_seed"]) for row in rows if row["task_id"] == task})
        for layout, seed in keys:
            pair = {row["procedure_id"]: row for row in rows if row["task_id"] == task and row["query_layout_id"] == layout and row["query_seed"] == seed}
            if set(pair) != set(procedures):
                continue
            a, b = pair[procedures[0]], pair[procedures[1]]
            both_closed_loop_valid = all(x["goal_success"] and x["transport_success"] and not x["collision"] for x in (a, b))
            if task == "obstacle_clearance":
                delta = b["transport_min_clearance_m"] - a["transport_min_clearance_m"]
                pairs.append({"task_id": task, "query_layout_id": layout, "query_seed": seed,
                              "prompt_a": procedures[0], "prompt_b": procedures[1],
                              "closed_loop_valid_pair": both_closed_loop_valid,
                              "paired_clearance_wide_minus_narrow_m": delta,
                              "goal_success_a": a["goal_success"], "goal_success_b": b["goal_success"],
                              "transport_success_a": a["transport_success"], "transport_success_b": b["transport_success"],
                              "collision_a": a["collision"], "collision_b": b["collision"]})
            else:
                pairs.append({"task_id": task, "query_layout_id": layout, "query_seed": seed,
                              "prompt_a": procedures[0], "prompt_b": procedures[1],
                              "closed_loop_valid_pair": both_closed_loop_valid,
                              "gentle_minus_normal_duration_s": b["transport_duration_s"] - a["transport_duration_s"],
                              "gentle_minus_normal_speed_peak_m_s": b["eef_speed_peak_m_s"] - a["eef_speed_peak_m_s"],
                              "gentle_minus_normal_acceleration_peak_m_s2": b["eef_acceleration_peak_m_s2"] - a["eef_acceleration_peak_m_s2"],
                              "gentle_minus_normal_jerk_peak_m_s3": b["eef_jerk_peak_m_s3"] - a["eef_jerk_peak_m_s3"],
                              "gentle_minus_normal_max_tilt_deg": b["max_object_tilt_deg"] - a["max_object_tilt_deg"],
                              "goal_success_a": a["goal_success"], "goal_success_b": b["goal_success"],
                              "transport_success_a": a["transport_success"], "transport_success_b": b["transport_success"],
                              "collision_a": a["collision"], "collision_b": b["collision"]})
    write_rows(output / "paired_rollout_summary.csv", pairs)
    failures = []
    for row in rows:
        if not row["goal_success"] or not row["transport_success"] or row["collision"]:
            failures.append({"name": row["name"], "task_id": row["task_id"],
                             "query_layout_id": row["query_layout_id"], "query_seed": row["query_seed"],
                             "failure_stage": row["failure_stage"],
                             "goal_success": row["goal_success"], "transport_success": row["transport_success"],
                             "collision": row["collision"]})
    write_rows(output / "closed_loop_failures.csv", failures)
    make_rollout_plots(output, rows, pairs)
    summary = {
        "experiment": "zero_shot_procedure_prompt_swap",
        "checkpoint": checkpoint_info(args.checkpoint),
        "device": "cuda:0",
        "rollouts": len(rows),
        "paired_rollout_conditions": len(pairs),
        "goal_successes": sum(row["goal_success"] for row in rows),
        "transport_successes": sum(row["transport_success"] for row in rows),
        "collision_free": sum(not row["collision"] for row in rows),
        "closed_loop_valid_pairs": sum(row["closed_loop_valid_pair"] for row in pairs),
        "interpretation": "Prompt sensitivity and procedure adherence are reported separately; action differences alone are not treated as procedure understanding.",
    }
    (output / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def make_rollout_plots(output: Path, rows: list[dict], pairs: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for pair in pairs:
        task = pair["task_id"]
        layout = pair["query_layout_id"]
        seed = pair["query_seed"]
        subset = [row for row in rows if row["task_id"] == task and row["query_layout_id"] == layout and row["query_seed"] == seed]
        if len(subset) != 2:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for row in subset:
            data = np.genfromtxt(output / f"{row['name']}_trajectory.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
            mask = data["phase"] == "transport"
            axes[0].plot(data["eef_x"][mask], data["eef_y"][mask], label=row["procedure_id"], linewidth=1.8)
            axes[1].plot(data["time_s"], data["eef_x"], label=f"{row['procedure_id']} x")
            axes[1].plot(data["time_s"], data["eef_z"], linestyle="--", label=f"{row['procedure_id']} z")
        axes[0].set(title=f"{task} {layout} seed {seed}", xlabel="EEF x (m)", ylabel="EEF y (m)", aspect="equal")
        axes[1].set(title="EEF time trace", xlabel="time (s)", ylabel="position (m)")
        axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
        figure.tight_layout()
        seed_dir = output / task / layout / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(seed_dir / "pair.png", dpi=160)
        plt.close(figure)


def main():
    args = parse_args()
    configure_environment(args)
    runtime = import_runtime()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    invocation = {"command": sys.argv, "checkpoint": checkpoint_info(args.checkpoint),
                  "data": str(args.data.resolve()), "cuda_visible_devices": args.gpu,
                  "egl_device": args.egl_device, "query_seeds": args.query_seeds,
                  "layouts": args.layouts, "max_steps": args.max_steps,
                  "procedure_label_provided_to_model": False,
                  "evaluation_regime": "zero-shot ID diagnostic"}
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2))
    subprocess.run([sys.executable, str(BPP_ROOT / "scripts" / "check_cuda.py")], check=True)
    device = "cuda:0"
    model, cfg, load_metadata = runtime["load_policy"](args.checkpoint, device, args.threads)
    (output / "model_loading.json").write_text(json.dumps(load_metadata, indent=2, default=str))
    rows = []
    if args.stage in ("rollouts", "all"):
        rows = run_rollouts(args, runtime, model, cfg, device, output)
    if args.stage in ("teacher", "all") or (args.stage == "rollouts" and not args.skip_teacher):
        teacher_forced(args, runtime, model, cfg, device, output)
    if args.stage in ("analyze", "all", "rollouts"):
        analyze_rollouts(args, output)
    print("Zero-shot experiment complete:", output, flush=True)


if __name__ == "__main__":
    main()
