#!/usr/bin/env python3
"""Run the C0 original-demonstration baseline on the C1 paired query states."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BPP_ROOT = PROJECT / "bpp"
CLEARANCE = HERE.parent / "clearance"
SOURCE_PROMPT = BPP_ROOT / "datasets/libero/libero_spatial_selected_combinations_view" / (
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo.hdf5"
)


def configure(args):
    os.environ.update({
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "PYNPUT_BACKEND": "dummy", "MUJOCO_EGL_DEVICE_ID": str(args.egl_device),
        "CUDA_VISIBLE_DEVICES": args.gpu, "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def seed_all(seed, torch):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_rows(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=BPP_ROOT / "checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt")
    p.add_argument("--gpu", default="5")
    p.add_argument("--egl-device", type=int, default=0)
    p.add_argument("--layouts", default="0,1,2,3,4")
    p.add_argument("--query-seeds", default="0,1")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    configure(args)
    sys.path.insert(0, str(CLEARANCE)); sys.path.insert(0, str(HERE))
    from inference import load_policy, load_prompt, query_observations
    from recording import Recorder
    from scene import Scene
    from generate import GENTLE_LAYOUT_POSITIONS, OBSTACLE_POSITIONS, set_layout
    from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
    import torch
    import h5py

    args.output.mkdir(parents=True, exist_ok=True)
    subprocess = __import__("subprocess")
    subprocess.run([sys.executable, str(BPP_ROOT / "scripts/check_cuda.py")], check=True)
    model, cfg, load_metadata = load_policy(args.checkpoint, "cuda:0", args.threads)
    (args.output / "model_loading.json").write_text(json.dumps(load_metadata, indent=2, default=str))
    invocation = {"checkpoint": str(args.checkpoint.resolve()), "data": str(args.data.resolve()),
                  "prompt": str(SOURCE_PROMPT.resolve()), "gpu": args.gpu,
                  "egl_device": args.egl_device, "layouts": args.layouts,
                  "query_seeds": args.query_seeds, "max_steps": args.max_steps,
                  "procedure_label_provided_to_model": False,
                  "paired_with": "procedural_constraints/data/zero_shot"}
    (args.output / "invocation.json").write_text(json.dumps(invocation, indent=2))

    specs = {
        "obstacle_clearance": OBSTACLE_POSITIONS,
        "gentle_transport": GENTLE_LAYOUT_POSITIONS,
    }
    layouts = [int(x) for x in args.layouts.split(",") if x.strip()]
    seeds = [int(x) for x in args.query_seeds.split(",") if x.strip()]
    data = args.data.resolve()
    scene = Scene(record_images=True)
    scene.canonical_state = np.load(data / "initial_state.npz")["state"]
    inverse = RotationTransformer("axis_angle", "rotation_6d")
    rows = []
    identities = {}
    try:
        for task, positions in specs.items():
            for layout in layouts:
                obstacle = np.asarray(positions[layout], dtype=np.float64)
                for seed in seeds:
                    key = f"{task}/layout_{layout:02d}/seed_{seed}"
                    set_layout(scene, f"layout_{layout:02d}", obstacle)
                    identity = scene.initial_identity()
                    if key in identities and identity != identities[key]:
                        raise RuntimeError(f"query identity changed: {key}")
                    identities[key] = identity
                    model.reset(action_exec_horizon=12)
                    prompt = load_prompt(SOURCE_PROMPT, cfg.task.shape_meta, "cuda:0")
                    with torch.inference_mode(): model.prompt(prompt)
                    seed_all(seed, torch)
                    name = f"{task}/layout_{layout:02d}/seed{seed}/original"
                    rec = Recorder(args.output / task / f"layout_{layout:02d}" / f"seed{seed}",
                                   "original", save_images=True)
                    rec.start(scene); history = [rec.latest_obs]
                    initial_height = float(scene.object_pos()[2]); goal_xy = scene.goal_pos()[:2].copy()
                    started = time.monotonic()
                    while len(rec.actions) < args.max_steps:
                        query = query_observations(history, cfg.task.shape_meta, "cuda:0")
                        with torch.inference_mode():
                            pred = model.predict_action(query)["action"][0, :12].detach().cpu().numpy()
                        actions = np.concatenate([pred[:, :3], inverse.inverse(pred[:, 3:9]), pred[:, 9:]], axis=-1)
                        for action in actions:
                            obj = scene.object_pos()
                            transporting = bool(obj[2] > initial_height + .035 and
                                                np.linalg.norm(obj[:2] - goal_xy) > .045)
                            rec.phase = "grasp" if obj[2] <= initial_height + .035 else ("transport" if transporting else "place")
                            rec.step(action); history.append(rec.latest_obs); history = history[-2:]
                            if scene.env._check_success() or len(rec.actions) >= args.max_steps:
                                break
                        if scene.env._check_success():
                            break
                    summary = rec.finish(scene)
                    trajectory = np.genfromtxt(args.output / f"{name}_trajectory.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
                    phases = np.asarray(trajectory["phase"], dtype=str)
                    obj = np.stack([trajectory[k] for k in ("object_x", "object_y", "object_z")], axis=1)
                    lifted = bool((obj[:, 2] > initial_height + .035).any())
                    transport = phases == "transport"
                    xyerr = np.linalg.norm(obj[:, :2] - goal_xy[None, :], axis=1)
                    transported = bool(np.any(transport & lifted & (xyerr < .045)))
                    if summary["goal_success"]: failure = "none"
                    elif not lifted: failure = "grasp"
                    elif not transported: failure = "transport"
                    else: failure = "place"
                    result = {"name": name, "task_id": task, "procedure_id": "original",
                              "query_layout_id": f"layout_{layout:02d}", "prompt_layout_id": "source_demo_0",
                              "query_seed": seed, "diffusion_seed": seed,
                              "goal_success": bool(summary["goal_success"]), "grasp_success": lifted,
                              "transport_success": transported, "place_success": bool(summary["goal_success"]),
                              "failure_stage": failure, "collision": bool(summary["collision"]),
                              "control_steps": int(summary["control_steps"]),
                              "transport_path_length_m": summary.get("transport_path_length_m"),
                              "transport_duration_s": summary.get("transport_duration_s"),
                              "initial_identity": identity,
                              "source_prompt": str(SOURCE_PROMPT.resolve()),
                              "wall_seconds": time.monotonic() - started}
                    rows.append(result); write_rows(args.output / "rollout_results.csv", rows)
                    print(json.dumps({k: result[k] for k in ("name", "goal_success", "grasp_success", "transport_success", "failure_stage", "collision")}, indent=2), flush=True)
    finally:
        scene.close()
    summary = {"experiment": "C0_original_prompt_baseline", "rollouts": len(rows),
               "goal_successes": sum(r["goal_success"] for r in rows),
               "transport_successes": sum(r["transport_success"] for r in rows),
               "grasp_successes": sum(r["grasp_success"] for r in rows),
               "collision_free": sum(not r["collision"] for r in rows),
               "prompt": str(SOURCE_PROMPT.resolve()), "same_query_states_as": str((args.data / "../zero_shot").resolve())}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (args.output / "query_identities.json").write_text(json.dumps(identities, indent=2))


if __name__ == "__main__":
    main()
