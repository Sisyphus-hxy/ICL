#!/usr/bin/env python3
"""Lightweight, read-only audit for LIBERO/LIBERO-Gen demonstration HDF5 files.

The script discovers canonical robomimic-style HDF5 files, keeps multi-episode
datasets for the complementarity analysis, and records the single-episode
canonical files as an inventory. RGB frames are read one at a time while
creating montages; numeric trajectories are small enough to inspect directly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import statistics
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEMO_RE = re.compile(r"^demo_(\d+)$")
EXCLUDE_PARTS = {"libero_audit", "env", "site-packages"}


def scalar(value: Any) -> Any:
    """Convert h5py/numpy values to JSON/CSV-friendly Python values."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return scalar(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def attr_dict(attrs: h5py.AttributeManager) -> dict[str, Any]:
    return {str(k): scalar(v) for k, v in attrs.items()}


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_jsonish(value: Any) -> Any:
    value = scalar(value)
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def slug(text: str, limit: int = 100) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "task")[:limit]


def num(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    return float(np.percentile(vals, q))


def stats(values: Iterable[float]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(min(vals)),
        "p25": percentile(vals, 25),
        "median": float(np.median(vals)),
        "p75": percentile(vals, 75),
        "max": float(max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def natural_demo_names(group: h5py.Group) -> list[str]:
    names = []
    for key in group.keys():
        m = DEMO_RE.match(str(key))
        if m and isinstance(group[key], h5py.Group):
            names.append((int(m.group(1)), str(key)))
    return [name for _, name in sorted(names)]


def infer_family(path: Path, data_attrs: dict[str, Any]) -> tuple[str, str]:
    """Return family and evidence; the family for generated files is explicit about inference."""
    low = str(path).lower()
    tag = str(data_attrs.get("tag", "")).lower()
    if tag == "libero-v1":
        return "LIBERO", "observed data.attrs[tag]=libero-v1"
    if "selected_combinations" in low or "liberogen" in low or "seed" in low:
        return "LIBERO-Gen/custom", "inferred from path/episode seed; no explicit LIBERO-Gen tag"
    if "/datasets/libero/" in low:
        return "LIBERO-like", "inferred from dataset path"
    return "other canonical HDF5", "canonical data/demo_* schema, outside LIBERO dataset root"


def discover(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover canonical HDF5 files and return (all_inventory, primary_multi_episode)."""
    inventory: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".hdf5", ".h5"}:
            continue
        parts = set(path.parts)
        if parts & EXCLUDE_PARTS:
            continue
        try:
            with h5py.File(path, "r") as f:
                if "data" not in f or not isinstance(f["data"], h5py.Group):
                    continue
                demos = natural_demo_names(f["data"])
                if not demos:
                    continue
                data_attrs = attr_dict(f["data"].attrs)
                family, family_evidence = infer_family(path, data_attrs)
                problem_info = parse_jsonish(data_attrs.get("problem_info", {}))
                instruction = problem_info.get("language_instruction") if isinstance(problem_info, dict) else None
                if not instruction:
                    instruction = data_attrs.get("bddl_file_name") or path.stem
                row = {
                    "path": path,
                    "relative_path": str(path.relative_to(root)),
                    "num_episodes": len(demos),
                    "attr_num_demos": data_attrs.get("num_demos"),
                    "family": family,
                    "family_evidence": family_evidence,
                    "task_instruction": str(instruction),
                    "data_attrs": data_attrs,
                    "demo_names": demos,
                }
                inventory.append(row)
                # Multi-demonstration analysis needs enough independent episodes for N=1/2.
                # Keep all files with at least 5 demos and task metadata; this is robust to
                # future additional LIBERO/Gen files outside the current dataset directory.
                if len(demos) >= 5 and (
                    "problem_info" in data_attrs
                    or "bddl_file_name" in data_attrs
                    or "libero" in str(path).lower()
                ):
                    primary.append(row)
        except Exception as exc:
            inventory.append({
                "path": path,
                "relative_path": str(path.relative_to(root)),
                "num_episodes": 0,
                "attr_num_demos": None,
                "family": "unreadable HDF5",
                "family_evidence": repr(exc),
                "task_instruction": "",
                "data_attrs": {},
                "demo_names": [],
            })
    return inventory, primary


def camera_info(obs: h5py.Group) -> tuple[list[str], dict[str, Any]]:
    cams = []
    shapes: dict[str, Any] = {}
    for key in sorted(obs.keys()):
        ds = obs[key]
        shape = tuple(int(x) for x in ds.shape)
        if "rgb" in key.lower() or "depth" in key.lower() or (len(shape) >= 3 and shape[-1] in (1, 3, 4)):
            cams.append(str(key))
            shapes[str(key)] = list(shape[1:])
    return cams, shapes


def field_dims(obs: h5py.Group) -> dict[str, Any]:
    return {str(k): list(obs[k].shape[1:]) for k in sorted(obs.keys())}


def proprio_fields(obs: h5py.Group) -> dict[str, Any]:
    out = {}
    for key in sorted(obs.keys()):
        low = key.lower()
        if "rgb" not in low and "depth" not in low and "seg" not in low:
            out[str(key)] = list(obs[key].shape[1:])
    return out


def parse_model_joints(xml: str) -> tuple[list[dict[str, Any]], int, int]:
    """Parse MuJoCo joint addresses without compiling meshes/assets."""
    free: list[dict[str, Any]] = []
    qpos = 0
    dof = 0
    try:
        root = ET.fromstring(xml)
    except Exception:
        return free, 0, 0
    qpos_width = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
    dof_width = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}
    for joint in root.iter("joint"):
        typ = joint.attrib.get("type", "hinge")
        qw = qpos_width.get(typ, 1)
        dw = dof_width.get(typ, 1)
        name = joint.attrib.get("name", f"joint_{qpos}")
        if typ == "free":
            clean = re.sub(r"_joint\d*$", "", name)
            free.append({"name": clean, "joint_name": name, "qpos_addr": qpos, "dof_addr": dof})
        qpos += qw
        dof += dw
    return free, qpos, dof


def state_layout(state_dim: int, qpos_total: int, dof_total: int) -> tuple[str, int] :
    if state_dim == 1 + qpos_total + dof_total:
        return "time+qpos+qvel", 1
    if state_dim == qpos_total + dof_total:
        return "qpos+qvel", 0
    return "unresolved", 0


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 < 1e-9 or n1 < 1e-9:
        return float("nan")
    dot = abs(float(np.dot(q0 / n0, q1 / n1)))
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def object_pose_changes(states: np.ndarray, free_joints: list[dict[str, Any]], offset: int) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    if states.ndim != 2 or states.shape[0] < 1 or not free_joints:
        return result
    first, last = states[0], states[-1]
    for obj in free_joints:
        start = offset + int(obj["qpos_addr"])
        if start + 7 > states.shape[1]:
            continue
        p0, p1 = first[start : start + 3], last[start : start + 3]
        q0, q1 = first[start + 3 : start + 7], last[start + 3 : start + 7]
        result[str(obj["name"])] = {
            "position_delta": float(np.linalg.norm(p1 - p0)),
            "orientation_angle_rad": quat_angle(q0, q1),
            "initial_x": float(p0[0]),
            "initial_y": float(p0[1]),
            "initial_z": float(p0[2]),
            "final_x": float(p1[0]),
            "final_y": float(p1[1]),
            "final_z": float(p1[2]),
        }
    return result


def resample(arr: np.ndarray, n: int = 32) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((n, arr.shape[1] if arr.ndim == 2 else 0), dtype=float)
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    old_x = np.linspace(0.0, 1.0, arr.shape[0])
    new_x = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(new_x, old_x, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def pairwise_mean_distance(arrays: list[np.ndarray]) -> list[float]:
    vals = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            if arrays[i].shape != arrays[j].shape:
                continue
            vals.append(float(np.mean(np.linalg.norm(arrays[i] - arrays[j], axis=1))))
    return vals


def safe_array(group: h5py.Group, key: str) -> np.ndarray | None:
    if key not in group:
        return None
    try:
        return np.asarray(group[key][...], dtype=float)
    except Exception:
        return None


def episode_record(
    root: Path,
    source: dict[str, Any],
    demo_name: str,
    group: h5py.Group,
    free_joints: list[dict[str, Any]],
    qpos_total: int,
    dof_total: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actions = safe_array(group, "actions")
    states = safe_array(group, "states")
    robot_states = safe_array(group, "robot_states")
    obs = group["obs"]
    ee_pos = safe_array(obs, "ee_pos")
    joint_states = safe_array(obs, "joint_states")
    gripper_states = safe_array(obs, "gripper_states")
    cameras, camera_shapes = camera_info(obs)
    state_dim = int(states.shape[1]) if states is not None and states.ndim == 2 else None
    layout, offset = state_layout(state_dim or -1, qpos_total, dof_total)
    obj = object_pose_changes(states, free_joints, offset) if states is not None else {}
    length = int(actions.shape[0]) if actions is not None and actions.ndim >= 1 else int(group.attrs.get("num_samples", 0))
    seed = scalar(group.attrs.get("seed")) if "seed" in group.attrs else None
    reward = safe_array(group, "rewards")
    done = safe_array(group, "dones")

    def delta_norm(arr: np.ndarray | None) -> float | None:
        if arr is None or arr.ndim != 2 or arr.shape[0] < 2:
            return None
        return float(np.linalg.norm(arr[-1] - arr[0]))

    rec: dict[str, Any] = {
        "source_file": source["relative_path"],
        "dataset_family": source["family"],
        "task_id": source["task_id"],
        "task_instruction": source["task_instruction"],
        "demo_id": demo_name,
        "demo_index": int(demo_name.split("_")[1]),
        "seed": seed,
        "length": length,
        "action_dim": int(actions.shape[1]) if actions is not None and actions.ndim == 2 else None,
        "abs_action_dim": int(group["abs_actions"].shape[1]) if "abs_actions" in group and len(group["abs_actions"].shape) == 2 else None,
        "camera_names": json_compact(cameras),
        "camera_shapes": json_compact(camera_shapes),
        "obs_fields": json_compact(field_dims(obs)),
        "proprio_fields": json_compact(proprio_fields(obs)),
        "state_dim": state_dim,
        "state_layout": layout,
        "action_l2_total": float(np.sum(np.linalg.norm(actions, axis=1))) if actions is not None else None,
        "action_l2_mean": float(np.mean(np.linalg.norm(actions, axis=1))) if actions is not None else None,
        "action_abs_max": float(np.max(np.abs(actions))) if actions is not None and actions.size else None,
        "state_delta_l2": delta_norm(states),
        "state_delta_max_abs": float(np.max(np.abs(states[-1] - states[0]))) if states is not None and states.ndim == 2 and states.shape[0] > 1 else None,
        "robot_state_delta_l2": delta_norm(robot_states),
        "joint_state_delta_l2": delta_norm(joint_states),
        "ee_pos_delta_l2": delta_norm(ee_pos),
        "gripper_state_delta_l2": delta_norm(gripper_states),
        "initial_ee_pos": json_compact(ee_pos[0].tolist()) if ee_pos is not None and len(ee_pos) else "",
        "final_ee_pos": json_compact(ee_pos[-1].tolist()) if ee_pos is not None and len(ee_pos) else "",
        "object_pose_changes": json_compact(obj),
        "object_names": json_compact(sorted(obj)),
        "reward_sum": float(np.sum(reward)) if reward is not None else None,
        "reward_max": float(np.max(reward)) if reward is not None and reward.size else None,
        "done_terminal": float(done[-1]) if done is not None and done.size else None,
    }
    numeric = {
        "length": length,
        "states": states,
        "actions_resampled": resample(actions, 32) if actions is not None else np.zeros((32, 0)),
        "ee_pos_resampled": resample(ee_pos, 32) if ee_pos is not None else np.zeros((32, 0)),
        "initial_state": states[0] if states is not None and len(states) else np.zeros(0),
        "final_state": states[-1] if states is not None and len(states) else np.zeros(0),
        "initial_ee_pos": ee_pos[0] if ee_pos is not None and len(ee_pos) else np.zeros(0),
        "final_ee_pos": ee_pos[-1] if ee_pos is not None and len(ee_pos) else np.zeros(0),
        "object_pose": obj,
    }
    return rec, numeric


def read_task(source: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = source["path"]
    episode_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        demos = source["demo_names"]
        first = data[demos[0]]
        states0 = safe_array(first, "states")
        state_dim = int(states0.shape[1]) if states0 is not None and states0.ndim == 2 else -1
        xml = scalar(first.attrs.get("model_file", ""))
        free_joints, qpos_total, dof_total = parse_model_joints(xml if isinstance(xml, str) else "")
        for demo in demos:
            rec, numeric = episode_record(root, source, demo, data[demo], free_joints, qpos_total, dof_total)
            episode_rows.append(rec)
            numeric_rows.append(numeric)
        attrs = source["data_attrs"]
        env_args = parse_jsonish(attrs.get("env_args", {}))
        bddl = attrs.get("bddl_file_name", "")
        if isinstance(env_args, dict) and env_args.get("bddl_file"):
            bddl = env_args.get("bddl_file")
        cams = json.loads(episode_rows[0]["camera_names"]) if episode_rows else []
        cam_shapes = json.loads(episode_rows[0]["camera_shapes"]) if episode_rows else {}
        pfields = json.loads(episode_rows[0]["proprio_fields"]) if episode_rows else {}
        lengths = [r["length"] for r in episode_rows]
        action_dims = sorted({r["action_dim"] for r in episode_rows if r["action_dim"] is not None})
        abs_action_dims = sorted({r["abs_action_dim"] for r in episode_rows if r["abs_action_dim"] is not None})
        pair_init_state = pairwise_mean_distance([n["initial_state"].reshape(1, -1) for n in numeric_rows if n["initial_state"].size])
        pair_final_state = pairwise_mean_distance([n["final_state"].reshape(1, -1) for n in numeric_rows if n["final_state"].size])
        pair_init_ee = pairwise_mean_distance([n["initial_ee_pos"].reshape(1, -1) for n in numeric_rows if n["initial_ee_pos"].size])
        pair_final_ee = pairwise_mean_distance([n["final_ee_pos"].reshape(1, -1) for n in numeric_rows if n["final_ee_pos"].size])
        pair_action = pairwise_mean_distance([n["actions_resampled"] for n in numeric_rows if n["actions_resampled"].shape[1]])
        pair_ee_traj = pairwise_mean_distance([n["ee_pos_resampled"] for n in numeric_rows if n["ee_pos_resampled"].shape[1]])
        length_diffs = [abs(numeric_rows[i]["length"] - numeric_rows[j]["length"]) for i in range(len(numeric_rows)) for j in range(i + 1, len(numeric_rows))]
        object_names = sorted({name for r in episode_rows for name in json.loads(r["object_names"])})
        object_summary: dict[str, Any] = {}
        for name in object_names:
            pos = []
            ang = []
            for r in episode_rows:
                o = json.loads(r["object_pose_changes"]).get(name, {})
                if o.get("position_delta") is not None:
                    pos.append(o["position_delta"])
                if o.get("orientation_angle_rad") is not None and math.isfinite(float(o["orientation_angle_rad"])):
                    ang.append(o["orientation_angle_rad"])
            object_summary[name] = {"position_delta": stats(pos), "orientation_angle_rad": stats(ang)}
        pair = {
            "task_id": source["task_id"],
            "source_file": source["relative_path"],
            "dataset_family": source["family"],
            "num_episodes": len(episode_rows),
            "initial_state_pair_l2_mean": float(np.mean(pair_init_state)) if pair_init_state else None,
            "initial_state_pair_l2_p95": percentile(pair_init_state, 95),
            "final_state_pair_l2_mean": float(np.mean(pair_final_state)) if pair_final_state else None,
            "final_state_pair_l2_p95": percentile(pair_final_state, 95),
            "initial_ee_pair_l2_mean": float(np.mean(pair_init_ee)) if pair_init_ee else None,
            "final_ee_pair_l2_mean": float(np.mean(pair_final_ee)) if pair_final_ee else None,
            "action_trajectory_pair_l2_mean": float(np.mean(pair_action)) if pair_action else None,
            "action_trajectory_pair_l2_p95": percentile(pair_action, 95),
            "ee_trajectory_pair_l2_mean": float(np.mean(pair_ee_traj)) if pair_ee_traj else None,
            "length_abs_diff_mean": float(np.mean(length_diffs)) if length_diffs else None,
            "length_abs_diff_p95": percentile(length_diffs, 95),
            "length_abs_diff_max": float(max(length_diffs)) if length_diffs else None,
            "object_pose_summary": json_compact(object_summary),
        }
        summary = {
            "task_id": source["task_id"],
            "dataset_family": source["family"],
            "family_evidence": source["family_evidence"],
            "source_file": source["relative_path"],
            "task_instruction": source["task_instruction"],
            "bddl_file": bddl,
            "num_episodes": len(episode_rows),
            "attr_num_demos": attrs.get("num_demos"),
            "total_steps_attr": attrs.get("total"),
            "length_min": stats(lengths)["min"],
            "length_p25": stats(lengths)["p25"],
            "length_median": stats(lengths)["median"],
            "length_p75": stats(lengths)["p75"],
            "length_max": stats(lengths)["max"],
            "length_mean": stats(lengths)["mean"],
            "length_std": stats(lengths)["std"],
            "camera_names": json_compact(cams),
            "camera_shapes": json_compact(cam_shapes),
            "action_dims": json_compact(action_dims),
            "abs_action_dims": json_compact(abs_action_dims),
            "proprio_fields": json_compact(pfields),
            "state_dim": state_dim if state_dim >= 0 else None,
            "state_layout": episode_rows[0]["state_layout"] if episode_rows else "",
            "object_pose_annotation": bool(object_names),
            "object_names": json_compact(object_names),
            "object_pose_summary": json_compact(object_summary),
            "pair_initial_state_l2_mean": pair["initial_state_pair_l2_mean"],
            "pair_final_state_l2_mean": pair["final_state_pair_l2_mean"],
            "pair_action_traj_l2_mean": pair["action_trajectory_pair_l2_mean"],
            "pair_ee_traj_l2_mean": pair["ee_trajectory_pair_l2_mean"],
            "pair_length_abs_diff_mean": pair["length_abs_diff_mean"],
            "language_instruction_available": bool(source["task_instruction"]),
            "initial_rgb_available": bool(cams),
            "raw_object_state_available": bool(object_names),
            "support_query_N1": len(episode_rows) >= 2,
            "support_query_N2": len(episode_rows) >= 3,
            "support_query_N4": len(episode_rows) >= 5,
            "support_query_N8": len(episode_rows) >= 9,
        }
    return summary, episode_rows, [pair], {"numeric": numeric_rows, "free_joints": free_joints, "camera_names": cams}


def default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.truetype("DejaVuSans.ttf", 12)


def make_montage(source: dict[str, Any], out_dir: Path, selected: list[str], frame_count: int = 8) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(source["path"], "r") as f:
        data = f["data"]
        if not selected:
            return records
        convention = str(scalar(data.attrs.get("macros_image_convention", ""))).lower()
        flip = convention == "opengl"
        cameras, _ = camera_info(data[selected[0]]["obs"])
        font = default_font()
        for camera in cameras:
            thumb_w, thumb_h = 160, 160
            label_h = 22
            canvas = Image.new("RGB", (frame_count * thumb_w, len(selected) * (thumb_h + label_h)), "white")
            draw = ImageDraw.Draw(canvas)
            frame_indices: dict[str, list[int]] = {}
            for row, demo in enumerate(selected):
                ds = data[demo]["obs"][camera]
                length = int(ds.shape[0])
                idxs = np.linspace(0, max(0, length - 1), frame_count).round().astype(int).tolist()
                frame_indices[demo] = idxs
                y = row * (thumb_h + label_h)
                draw.text((2, y + 2), f"{demo}  T={length}", fill="black", font=font)
                for col, idx in enumerate(idxs):
                    arr = np.asarray(ds[idx])
                    if flip:
                        arr = np.flipud(arr)
                    if arr.ndim == 2:
                        arr = np.repeat(arr[..., None], 3, axis=-1)
                    if arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    image = Image.fromarray(arr.astype(np.uint8)).resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
                    canvas.paste(image, (col * thumb_w, y + label_h))
                    draw.text((col * thumb_w + 2, y + label_h + 2), str(idx), fill="yellow", font=font)
            file_name = f"{source['task_id']}__{slug(camera)}.png"
            canvas.save(out_dir / file_name, optimize=True)
            records.append({
                "task_id": source["task_id"],
                "camera": camera,
                "file": str((out_dir / file_name).relative_to(out_dir.parent)),
                "episodes": json_compact(selected),
                "frame_indices": json_compact(frame_indices),
                "image_convention": convention or "unspecified",
                "flipud_applied": flip,
            })
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: scalar(row.get(k, "")) for k in keys})


def write_report(
    out: Path,
    root: Path,
    inventory: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    montage_rows: list[dict[str, Any]],
) -> None:
    canonical = [r for r in inventory if r.get("num_episodes", 0) > 0]
    single = [r for r in canonical if r.get("num_episodes", 0) < 5]
    lines: list[str] = []
    lines.append("# LIBERO / LIBERO-Gen HDF5 demonstration 数据审计")
    lines.append("")
    lines.append("本报告由 `libero_audit/audit.py` 生成。原始 HDF5 以只读方式打开；RGB montage 逐帧读取，没有把整段图像一次性载入内存。")
    lines.append("")
    lines.append("## 数据定位与范围")
    lines.append("")
    lines.append(f"- 扫描根目录：`{root}`。发现 {len(canonical)} 个可识别的 robomimic 风格 `data/demo_*` HDF5（另有 {len(inventory)-len(canonical)} 个无法按该 schema 读取）。")
    lines.append(f"- 其中 {len(summaries)} 个文件有至少 5 条 episode，作为多示范互补性主分析对象；{len(single)} 个仅有单条 episode 的 canonical HDF5 记录在 inventory 中，但不能构造同 task 的 support/query 对。当前主分析对应两份 `bpp/datasets/libero/` 数据，各 50 episodes。")
    lines.append("- `.cache/rollout/` 下还存在 Zarr rollout，但本次任务限定 HDF5，因此未混入 HDF5 统计。")
    lines.append("")
    lines.append("### 主分析文件")
    lines.append("")
    lines.append("| family | task | episodes | instruction |")
    lines.append("|---|---|---:|---|")
    for s in summaries:
        lines.append(f"| {s['dataset_family']} | `{s['task_id']}` | {s['num_episodes']} | {s['task_instruction']} |")
    lines.append("")
    lines.append("`LIBERO-Gen/custom` 是基于路径包含 `selected_combinations_view` 与 episode-level `seed` 的推断；文件没有显式 `LIBERO-Gen` tag，因而报告不把这一点当作观测事实。`LIBERO` 则有 `data.attrs[tag]=libero-v1` 的直接证据。")
    lines.append("")

    lines.append("## Schema 与真实统计")
    lines.append("")
    lines.append("两份主数据均为 HDF5 → `data` → `demo_i`；每条 demo 有 `actions`、`dones`、`rewards`、`states`、`robot_states` 和 `obs`。`obs` 的真实字段包括 `agentview_rgb`、`eye_in_hand_rgb`、`ee_pos`、`ee_ori`、`ee_states`、`gripper_states`、`joint_states`。动作均为 7 维；生成变体另外有 7 维 `abs_actions`。")
    lines.append("")
    lines.append("| task | length min/p25/median/p75/max (mean±std) | cameras | state | object pose |")
    lines.append("|---|---|---|---:|---|")
    for s in summaries:
        lengths = "/".join(fmt(s[k], 1) for k in ["length_min", "length_p25", "length_median", "length_p75", "length_max"])
        meanstd = f"{fmt(s['length_mean'], 1)}±{fmt(s['length_std'], 1)}"
        cams = ", ".join(json.loads(s["camera_names"]))
        objects = ", ".join(json.loads(s["object_names"])) or "无可解析 free-joint"
        lines.append(f"| `{s['task_id']}` | {lengths} ({meanstd}) | {cams} | {s['state_dim']} ({s['state_layout']}) | {objects} |")
    lines.append("")
    lines.append("`states` 的 92 维在 XML joint 计数下可解析为 `time + qpos(48) + qvel(43)`；free joint 的物体位姿使用 qpos 中的 xyz+quaternion 计算。这个映射是由数据中的 model XML 和维数一致性得到的真实结构检查，不需要视觉模型。")
    lines.append("")
    lines.append("RGB 帧按 `data.attrs['macros_image_convention']='opengl'`（OpenGL 底左原点）存储，直接可视化是上下颠倒的；本报告的 montage 已做 `flipud` 还原为自然视角。任何把 RGB 喂给模型的管线都必须同样处理这个翻转，否则等于在颠倒画面上训练。")
    lines.append("")

    lines.append("## 同 task demonstrations 的差异")
    lines.append("")
    lines.append("下表为所有 demo pair 的数值统计：state/EEF 是首帧或末帧的 L2 距离；动作和 EEF trajectory 先各自重采样到 32 个相对时间点，再计算每时间点 L2 的均值；长度差为绝对 step 数。")
    lines.append("")
    lines.append("| task | init state pair mean | final state pair mean | action trajectory pair mean | EEF trajectory pair mean | length diff mean/p95/max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for p in pairs:
        lines.append(f"| `{p['task_id']}` | {fmt(p['initial_state_pair_l2_mean'])} | {fmt(p['final_state_pair_l2_mean'])} | {fmt(p['action_trajectory_pair_l2_mean'])} | {fmt(p['ee_trajectory_pair_l2_mean'])} | {fmt(p['length_abs_diff_mean'],1)}/{fmt(p['length_abs_diff_p95'],1)}/{fmt(p['length_abs_diff_max'],1)} |")
    lines.append("")
    lines.append("物体位姿变化的均值/分位数已写入 `summary.csv` 和 `pairwise_stats.csv` 的 JSON 字段；`episode_stats.csv` 保存每条 episode 的初末位置、旋转角、EEF/关节/robot state 变化。数值是状态和动作数据的直接统计，未使用大型视觉模型。")
    lines.append("")

    lines.append("## Montage")
    lines.append("")
    lines.append("每个主任务随机（固定 seed=0）抽取最多 5 条 episode，每条从头到尾均匀取 8 帧。每个可用 RGB 相机各生成一张 montage，文件位于 `montages/`；抽样 episode、精确 frame index 和是否应用 `flipud` 记录在 `montages/index.csv`。")
    lines.append("")
    for row in montage_rows:
        lines.append(f"- `{row['camera']}`: `{row['file']}`")
    lines.append("")

    lines.append("## Support-query pairs")
    lines.append("")
    lines.append("对每个 task，N 条 support 与另一条 query 独立抽取、且 query 不进入 support。50 条 episode 使 N=1/2/4/8 全部可行；若固定 query，剩余可选 support 组合数分别为 C(49,N)。")
    lines.append("")
    lines.append("| task | N=1 | N=2 | N=4 | N=8 | query count (按 N) |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        n = int(s["num_episodes"])
        lines.append(f"| `{s['task_id']}` | {'yes' if n>=2 else 'no'} | {'yes' if n>=3 else 'no'} | {'yes' if n>=5 else 'no'} | {'yes' if n>=9 else 'no'} | {n-1}/{n-1}/{n-1}/{n-1} |")
    lines.append("")
    lines.append("这表示数据层面可以做 support-query pair；它不保证 support 一定包含 query 所需的互补信息，互补性需要在严格 episode-level split 上评测。")
    lines.append("")

    lines.append("## Task 可推断性与数据捷径风险")
    lines.append("")
    lines.append("以下是基于字段和场景结构的审计判断，属于推测/风险分析，不是训练分类器得到的准确率：")
    lines.append("")
    lines.append("- 语言指令直接存于 `data.attrs['problem_info']['language_instruction']`，并且 task 文件名、BDDL 名称也携带目标物体和动作。若 predictive study 输入 instruction，模型可能只靠语言完成 task identity 判断而忽略 support。")
    lines.append("- `agentview_rgb` 和 `eye_in_hand_rgb` 同时存在；初始帧可见物体颜色、几何和相对布局。若 query 目标/场景在视觉上足够显著，模型可能只依赖当前 observation。")
    lines.append("- 原始 `states` 含可解析的物体 free-joint 位姿，`joint_states`/`ee_states` 含机器人状态。若把这些状态直接提供给模型，任务和进度可能被状态捷径编码。")
    lines.append("- 两个主文件各自只对应一个 task；因此当前仓库不能凭这两份数据估计跨 task 的 instruction/observation-only baseline，也不能证明 support 对 query 的增益。应显式加入 no-support、language-only、initial-observation-only 和 state-only 对照。")
    lines.append("")

    lines.append("## 离线 predictive study 适用性与划分建议")
    lines.append("")
    lines.append("判断：适合做小规模、同 task、episode-level 的离线 predictive study 原型，尤其适合检验动作/轨迹预测中 support 数量和示范差异的影响；不适合作为广泛 LIBERO/多 task 泛化结论的数据集。当前只有 2 个 task、每 task 50 条 demo，且不同文件的生成协议不同，需在论文中明确数据来源。")
    lines.append("")
    lines.append("建议划分：对每个 task 按 episode ID 做 30/10/10 的 train/val/test（或 35/7/8），所有帧和由同一 episode 生成的窗口必须留在同一 split；support 与 query 只能从同一 split 内抽取，禁止跨 split 配对。生成变体可按 seed 分层，LIBERO-v1 没有 seed 字段则按 demo ID 固定随机划分并保存 manifest。需要跨 task 泛化时只能做 leave-one-task-out 探索，当前两 task 规模过小，不能作为稳定泛化结论。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `summary.csv`: 每个主 task 的 schema、轨迹分布、相机、动作维度、物体位姿和 support-query 可行性。")
    lines.append("- `episode_stats.csv`: 每条 demo 的长度、状态/动作/EEF/物体变化和奖励字段。")
    lines.append("- `pairwise_stats.csv`: 同 task demo pair 的差异汇总。")
    lines.append("- `inventory.csv`: 扫描到的所有 canonical HDF5 及其 episode 数量，便于复核排除的单 episode rollout。")
    lines.append("- `montages/`: 相机对比图及 `montages/index.csv`。")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("libero_audit"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)

    inventory, primary = discover(root)
    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    montage_rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for source in primary:
        source["task_id"] = slug(f"{source['family']}__{Path(source['relative_path']).stem}")
        summary, episodes, pairs, extra = read_task(source, root)
        summaries.append(summary)
        episode_rows.extend(episodes)
        pair_rows.extend(pairs)
        selected = list(source["demo_names"])
        if len(selected) > 5:
            selected = sorted(rng.sample(selected, 5), key=lambda x: int(x.split("_")[1]))
        montage_rows.extend(make_montage(source, out / "montages", selected))

    inventory_rows = []
    for item in inventory:
        inventory_rows.append({
            "source_file": item.get("relative_path", ""),
            "dataset_family": item.get("family", ""),
            "family_evidence": item.get("family_evidence", ""),
            "num_episodes": item.get("num_episodes", 0),
            "attr_num_demos": item.get("attr_num_demos", ""),
            "task_instruction": item.get("task_instruction", ""),
            "included_in_primary_audit": item in primary,
        })
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "episode_stats.csv", episode_rows)
    write_csv(out / "pairwise_stats.csv", pair_rows)
    write_csv(out / "inventory.csv", inventory_rows)
    write_csv(out / "montages" / "index.csv", montage_rows)
    write_report(out, root, inventory, summaries, pair_rows, montage_rows)
    print(json.dumps({
        "root": str(root),
        "out": str(out),
        "canonical_files": len([x for x in inventory if x.get("num_episodes", 0) > 0]),
        "primary_files": len(primary),
        "primary_episodes": sum(int(x["num_episodes"]) for x in summaries),
        "montages": len(montage_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
