#!/usr/bin/env python3
"""Generate and validate the minimal procedural-constraint dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
CLEARANCE = HERE.parent / "clearance"
sys.path.insert(0, str(CLEARANCE))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from expert import Servo  # noqa: E402
from scene import Scene  # noqa: E402
import scene as clearance_scene  # noqa: E402
from recorder import Recorder  # noqa: E402


GOAL_ID = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box"
OBJECT_ID = "akita_black_bowl_1"
LAYOUT_COUNT = 5
OBSTACLE_POSITIONS = [
    np.array([-0.020, 0.005, 0.990]),
    np.array([-0.020, 0.010, 0.990]),
    np.array([-0.020, 0.015, 0.990]),
    np.array([-0.020, 0.020, 0.990]),
    np.array([-0.020, 0.025, 0.990]),
]
GENTLE_LAYOUT_POSITIONS = [
    np.array([-0.020, 0.260, 0.990]),
    np.array([-0.020, 0.280, 0.990]),
    np.array([-0.020, 0.300, 0.990]),
    np.array([-0.020, 0.320, 0.990]),
    np.array([-0.020, 0.340, 0.990]),
]


def arc_resample(points: np.ndarray, count: int) -> np.ndarray:
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.stack(
        [np.interp(np.linspace(0, distance[-1], count), distance, points[:, axis]) for axis in range(3)],
        axis=-1,
    )


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def obstacle_paths(start: np.ndarray, end: np.ndarray, obstacle: np.ndarray) -> dict[str, np.ndarray]:
    """Reuse the validated smooth LIBERO route shape around a relocated post."""
    t = np.linspace(0.0, 1.0, 3001)
    base = start[None] * (1.0 - t[:, None]) + end[None] * t[:, None]
    midpoint_y = (start[1] + end[1]) / 2.0

    def curve(peak, compensate=0.0):
        result = base.copy()
        result[:, 1] += (peak - midpoint_y) * np.sin(np.pi * t) ** 2
        u = np.clip(t / .40, 0.0, 1.0)
        result[:, 0] -= compensate * np.sin(np.pi * u) ** 2
        return result

    wide = curve(obstacle[1] + .225)
    # The extra early x excursion is the same path-length compensation used by
    # the already validated clearance experiment and keeps both procedures'
    # control durations equal.
    from scipy.optimize import brentq
    target = path_length(wide)
    amount = brentq(lambda value: path_length(curve(obstacle[1] + .160, value)) - target, 0.0, .30)
    narrow = curve(obstacle[1] + .160, amount)
    return {"narrow": arc_resample(narrow, 150), "wide": arc_resample(wide, 150)}


def common_transport_path(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 3001)
    path = start[None] * (1.0 - t[:, None]) + end[None] * t[:, None]
    path[:, 1] += 0.030 * np.sin(np.pi * t) ** 2
    return path


def set_layout(scene: Scene, layout_id: str, position: np.ndarray):
    """Reset the common query state and place a stationary constraint object."""
    # Scene.initial_identity and the existing recorder use this map. Adding
    # entries here is process-local and leaves the installed BPP code intact.
    clearance_scene.LAYOUTS[layout_id] = np.asarray(position, dtype=np.float64)
    scene.reset("original")
    scene.d.mocap_pos[scene.mocap_id] = np.asarray(position, dtype=np.float64)
    scene.d.mocap_quat[scene.mocap_id] = [1, 0, 0, 0]
    scene.env.sim.forward()
    scene.env.robots[0].controller.update(force=True)
    scene.env.robots[0].controller.reset_goal()
    scene.env._update_observables(force=True)
    scene.layout = layout_id
    scene.layout_seed = int(layout_id.rsplit("_", 1)[-1])
    scene.obstacle_layout_position = np.asarray(position, dtype=np.float64)


def generate_one(scene: Scene, output: Path, task_id: str, procedure: str,
                 layout_id: str, obstacle: np.ndarray, scene_seed: int,
                 images: bool = False) -> dict:
    set_layout(scene, layout_id, obstacle)
    name = f"{task_id}/{layout_id}/{procedure}"
    metadata = {
        "task_id": task_id,
        "procedure_id": procedure,
        "layout_id": layout_id,
        "goal_id": GOAL_ID,
        "object_id": OBJECT_ID,
        "scene_seed": int(scene_seed),
        "obstacle_position": obstacle.tolist(),
        "constraint_object_id": "procedure_obstacle",
        "constraint_object_radius_m": float(clearance_scene.OBSTACLE_RADIUS),
        "constraint_object_height_m": float(clearance_scene.OBSTACLE_HEIGHT),
        "model_input_excludes_procedure_label": True,
    }
    rec = Recorder(output, name, metadata)
    rec.start(scene)
    servo = Servo(rec)
    rec.phase = "grasp"
    grasp_pose = scene.reference_grasp_pose.copy()
    grasp_pose[:3] += scene.object_pos() - scene.reference_object_pos

    def fixed_move(pose, steps, closed=True):
        for _ in range(steps // 2):
            servo.move_to_pose(
                pose, close_gripper=closed, max_steps=2,
                pos_threshold=0.0, rot_threshold=0.0,
            )

    above = grasp_pose.copy()
    above[2] += 0.10
    fixed_move(above, 80, closed=False)
    fixed_move(grasp_pose, 80, closed=False)
    for _ in range(20):
        rec.step(np.array([0., 0., 0., 0., 0., 0., 1.]))

    rec.phase = "lift"
    start = grasp_pose[:3].copy()
    start[2] = 1.075
    lift = np.r_[start, grasp_pose[3:]]
    fixed_move(lift, 110)
    if np.linalg.norm(scene.obs()["robot0_eef_pos"] - start) >= .006:
        raise RuntimeError(f"lift failed for {name}")
    if scene.object_pos()[2] <= .97:
        raise RuntimeError(f"grasp did not lift bowl for {name}")

    end = start.copy()
    end[:2] += scene.goal_pos()[:2] - scene.object_pos()[:2]
    place = end.copy()
    place[2] = scene.goal_pos()[2] + .010 + (start[2] - scene.object_pos()[2])

    if task_id == "obstacle_clearance":
        path_pair = obstacle_paths(start, end, obstacle)
        points = path_pair[procedure]
    else:
        common = common_transport_path(start, end)
        # Both procedures use the identical geometric curve. Gentle uses a
        # denser time parameterization and therefore more nonzero control steps.
        points = arc_resample(common, 120 if procedure == "normal" else 240)

    rec.phase = "transport"
    for point in points:
        kwargs = {}
        if task_id == "gentle_transport" and procedure == "gentle":
            kwargs.update(kp_pos=4.0, kp_rot=1.5, max_pos_speed=0.25, max_rot_speed=0.30)
        servo.move_to_pose(
            np.r_[point, grasp_pose[3:]], close_gripper=True,
            max_steps=2 if task_id == "obstacle_clearance" else 1,
            pos_threshold=0.0, rot_threshold=0.0, **kwargs,
        )

    rec.phase = "align"
    fixed_move(np.r_[end, grasp_pose[3:]], 50)
    if np.linalg.norm(scene.obs()["robot0_eef_pos"] - end) >= .006:
        raise RuntimeError(f"alignment failed for {name}")
    rec.phase = "place"
    fixed_move(np.r_[place, grasp_pose[3:]], 100)
    rec.phase = "release"
    for _ in range(16):
        rec.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    summary = rec.finish(scene)
    summary["transport_start"] = start.tolist()
    summary["transport_end"] = end.tolist()
    summary["place_target"] = place.tolist()
    summary["commanded_transport_length_m"] = path_length(points)
    summary["transport_sample_count"] = len(points)
    return summary


def write_stats(output: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    with (output / "statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def generate_dataset(output: Path, images: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    config_path = HERE / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if not np.allclose(np.asarray(config["obstacle_clearance"]["obstacle_positions_m"]), OBSTACLE_POSITIONS):
            raise ValueError("config.json obstacle layouts do not match generator constants")
        if not np.allclose(np.asarray(config["gentle_transport"]["obstacle_positions_m"]), GENTLE_LAYOUT_POSITIONS):
            raise ValueError("config.json gentle layouts do not match generator constants")
    manifest = {
        "dataset_id": "in_context_procedural_constraint_transfer_minimal_v1",
        "goal_id": GOAL_ID,
        "object_id": OBJECT_ID,
        "layout_count_per_task": LAYOUT_COUNT,
        "tasks": {
            "obstacle_clearance": {"procedures": ["narrow", "wide"]},
            "gentle_transport": {"procedures": ["normal", "gentle"]},
        },
        "procedure_labels_in_model_input": False,
        "generator_command": sys.argv,
        "source_scene": "bpp/experiments/clearance/scene.py",
        "config_file": str(HERE / "config.json"),
    }
    if config_path.is_file():
        (output / "config.json").write_text(config_path.read_text())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    scene = Scene(record_images=False)
    # Freeze the same generated LIBERO initial state for every paired layout.
    scene.canonical_state = scene.env.sim.get_state().flatten().copy()
    (output / "scene.xml").write_text(scene.model_xml)
    np.savez(output / "initial_state.npz", state=scene.canonical_state)
    rows = []
    try:
        for task_id, procedures, layout_positions in (
            ("obstacle_clearance", ("narrow", "wide"), OBSTACLE_POSITIONS),
            ("gentle_transport", ("normal", "gentle"), GENTLE_LAYOUT_POSITIONS),
        ):
            for layout_index, obstacle in enumerate(layout_positions):
                layout_id = f"layout_{layout_index:02d}"
                for procedure in procedures:
                    print(f"Generating {task_id}/{layout_id}/{procedure}", flush=True)
                    row = generate_one(
                        scene, output, task_id, procedure, layout_id, obstacle,
                        scene_seed=1000 + layout_index,
                        images=images,
                    )
                    rows.append(row)
                    write_stats(output, rows)
                    print(json.dumps({k: row[k] for k in (
                        "name", "goal_success", "collision", "control_steps",
                        "object_min_clearance_m", "transport_duration_s",
                        "eef_speed_mean_m_s", "eef_acceleration_rms_m_s2",
                        "eef_jerk_rms_m_s3", "max_object_tilt_deg")}, indent=2), flush=True)
    finally:
        scene.close()
    (output / "generation_complete.json").write_text(json.dumps({
        "valid": len(rows) == 20,
        "demonstrations": len(rows),
        "successes": sum(bool(r["goal_success"]) for r in rows),
        "collision_free": sum(not bool(r["collision"]) for r in rows),
    }, indent=2))
    return rows


def _load_row(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        result = {key: demo.attrs[key] for key in demo.attrs.keys()}
        result["name"] = path.stem
        result["goal_success"] = bool(np.asarray(demo["phases"][-1]).size and demo.attrs.get("goal_success", False))
        result["steps"] = len(demo["actions"])
        result["obs_steps"] = len(demo["obs/ee_pos"])
    return result


def replay_validate(output: Path):
    """Replay saved states/actions and verify post-action proprioception alignment."""
    scene = Scene(record_images=False)
    reports = []
    try:
        for path in sorted(output.rglob("*.hdf5")):
            print(f"Replaying {path.stem}", flush=True)
            with h5py.File(path, "r") as handle:
                demo = handle["data/demo_0"]
                obstacle = np.asarray(json.loads(demo.attrs["obstacle_position"]))
                layout_id = demo.attrs["layout_id"]
                set_layout(scene, layout_id, obstacle)
                states = demo["states"][:]
                actions = demo["actions"][:]
                stored_ee = demo["obs/ee_pos"][:]
                gripper_state = demo["gripper_command_state"][:]
                scene.env.sim.set_state_from_flattened(states[0])
                scene.env.sim.forward()
                scene.env.robots[0].controller.update(force=True)
                scene.env.robots[0].controller.reset_goal()
                sampled = []
                replay_collision = False
                for step, action in enumerate(actions):
                    scene.env.robots[0].gripper.current_action = gripper_state[step].copy()
                    obs, _, _, _ = scene.env.step(action)
                    if step in (0, len(actions) // 2, len(actions) - 1):
                        sampled.append(float(np.max(np.abs(obs["robot0_eef_pos"] - stored_ee[step]))))
                    # Contact checks are sampled during replay; the original
                    # recorder already checked every MuJoCo substep.
                    if step % 20 == 0 and scene.contacts(transport=bool(demo["phases"][step] == b"transport")):
                        replay_collision = True
                terminal_error = float(np.max(np.abs(scene.env.sim.get_state().flatten() - demo["terminal_state"][:])))
                stored_phases = np.asarray(demo["phases"][:])
                transport = stored_phases == b"transport"
                reports.append({
                    "name": str(path.relative_to(output).with_suffix("")),
                    "steps": len(actions),
                    "obs_steps": len(stored_ee),
                    "post_action_alignment_max_m": max(sampled),
                    "terminal_state_max_abs_error": terminal_error,
                    "goal_success": bool(demo.attrs.get("goal_success", False)),
                    "recorded_collision": bool(demo.attrs.get("collision", False)),
                    "replay_collision": replay_collision,
                    "transport_samples": int(transport.sum()),
                    "valid": len(actions) == len(stored_ee) and max(sampled) < 2e-4 and terminal_error < 2e-3 and not replay_collision,
                })
                (output / "replay_validation.partial.json").write_text(json.dumps({"reports": reports}, indent=2))
    finally:
        scene.close()
    (output / "replay_validation.json").write_text(json.dumps({
        "valid": all(item["valid"] for item in reports),
        "reports": reports,
    }, indent=2))
    return reports


def validate_pairs(output: Path, rows: list[dict] | None = None):
    if rows is None:
        rows = []
        for path in sorted(output.rglob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                demo = handle["data/demo_0"]
                attrs = dict(demo.attrs)
                row = {"name": str(path.relative_to(output).with_suffix("")), **attrs}
                for key in ("goal_success", "collision", "object_min_clearance_m", "transport_path_length_m",
                            "transport_duration_s", "eef_speed_mean_m_s", "eef_acceleration_rms_m_s2",
                            "eef_jerk_rms_m_s3", "action_translation_zero_fraction"):
                    if key in attrs:
                        row[key] = attrs[key]
                rows.append(row)
    checks = {}
    for task_id, procedures in (("obstacle_clearance", ("narrow", "wide")),
                                ("gentle_transport", ("normal", "gentle"))):
        for index in range(LAYOUT_COUNT):
            layout = f"layout_{index:02d}"
            pair = {row["procedure_id"]: row for row in rows
                    if row.get("task_id") == task_id and row.get("layout_id") == layout}
            key = f"{task_id}/{layout}"
            if set(pair) != set(procedures):
                checks[key] = {"valid": False, "reason": "missing_procedure"}
                continue
            a, b = pair[procedures[0]], pair[procedures[1]]
            valid = bool(a.get("goal_success", False)) and bool(b.get("goal_success", False)) \
                and not bool(a.get("collision", True)) and not bool(b.get("collision", True))
            with h5py.File(output / task_id / layout / f"{procedures[0]}.hdf5", "r") as fa, \
                    h5py.File(output / task_id / layout / f"{procedures[1]}.hdf5", "r") as fb:
                da, db = fa["data/demo_0"], fb["data/demo_0"]
                first_a = int(np.where(da["phases"][:] == b"transport")[0][0])
                first_b = int(np.where(db["phases"][:] == b"transport")[0][0])
                pretransport_action_delta = float(np.max(np.abs(da["actions"][:first_a] - db["actions"][:first_b])))
                same_initial_identity = da.attrs["initial_identity"] == db.attrs["initial_identity"]
                same_object_initial = da.attrs["object_initial_pos"] == db.attrs["object_initial_pos"]
                same_goal_initial = da.attrs["goal_initial_pos"] == db.attrs["goal_initial_pos"]
            valid = valid and first_a == first_b and pretransport_action_delta < 1e-7 \
                and same_initial_identity and same_object_initial and same_goal_initial
            if task_id == "obstacle_clearance":
                gap = float(b["object_min_clearance_m"]) - float(a["object_min_clearance_m"])
                # The common grasp/place segments must remain paired. The
                # transport route itself is the only intended difference.
                def transport_positions(procedure):
                    data = np.genfromtxt(output / task_id / layout / f"{procedure}_trajectory.csv",
                                         delimiter=",", names=True, dtype=None, encoding="utf8")
                    mask = data["phase"] == "transport"
                    return np.stack([data[key][mask] for key in ("eef_x", "eef_y", "eef_z")], axis=1), data
                narrow_pos, narrow_data = transport_positions("narrow")
                wide_pos, wide_data = transport_positions("wide")
                endpoint_delta = float(np.linalg.norm(narrow_pos[-1] - wide_pos[-1]))
                place_n = np.stack([narrow_data[key][narrow_data["phase"] == "place"] for key in ("eef_x", "eef_y", "eef_z")], axis=1)[-1]
                place_w = np.stack([wide_data[key][wide_data["phase"] == "place"] for key in ("eef_x", "eef_y", "eef_z")], axis=1)[-1]
                valid = valid and gap >= .010 and float(a["object_min_clearance_m"]) > .005 and endpoint_delta < .010
                checks[key] = {"valid": valid, "wide_minus_narrow_clearance_m": gap,
                               "narrow_min_clearance_m": float(a["object_min_clearance_m"]),
                               "wide_min_clearance_m": float(b["object_min_clearance_m"]),
                               "transport_endpoint_delta_m": endpoint_delta,
                               "place_endpoint_delta_m": float(np.linalg.norm(place_n - place_w)),
                               "pretransport_action_max_delta": pretransport_action_delta,
                               "same_initial_identity": same_initial_identity,
                               "same_object_initial": same_object_initial,
                               "same_goal_initial": same_goal_initial,
                               "same_goal_id": a["goal_id"] == b["goal_id"],
                               "same_scene_seed": int(a["scene_seed"]) == int(b["scene_seed"])}
            else:
                speed_ratio = float(b["eef_speed_mean_m_s"]) / max(float(a["eef_speed_mean_m_s"]), 1e-8)
                acc_ratio = float(b["eef_acceleration_rms_m_s2"]) / max(float(a["eef_acceleration_rms_m_s2"]), 1e-8)
                jerk_ratio = float(b["eef_jerk_rms_m_s3"]) / max(float(a["eef_jerk_rms_m_s3"]), 1e-8)
                duration_ratio = float(b["transport_duration_s"]) / max(float(a["transport_duration_s"]), 1e-8)
                def transport_positions(procedure):
                    data = np.genfromtxt(output / task_id / layout / f"{procedure}_trajectory.csv",
                                         delimiter=",", names=True, dtype=None, encoding="utf8")
                    mask = data["phase"] == "transport"
                    return np.stack([data[key][mask] for key in ("eef_x", "eef_y", "eef_z")], axis=1), data
                normal_pos, normal_data = transport_positions("normal")
                gentle_pos, gentle_data = transport_positions("gentle")
                def normalized(points, count=100):
                    grid = np.linspace(0.0, 1.0, len(points))
                    target = np.linspace(0.0, 1.0, count)
                    return np.stack([np.interp(target, grid, points[:, axis]) for axis in range(3)], axis=1)
                spatial_delta = float(np.linalg.norm(normalized(normal_pos) - normalized(gentle_pos), axis=1).max())
                place_n = np.stack([normal_data[key][normal_data["phase"] == "place"] for key in ("eef_x", "eef_y", "eef_z")], axis=1)[-1]
                place_g = np.stack([gentle_data[key][gentle_data["phase"] == "place"] for key in ("eef_x", "eef_y", "eef_z")], axis=1)[-1]
                gentle_motion = np.linalg.norm(np.diff(gentle_pos, axis=0), axis=1)
                valid = valid and duration_ratio > 1.5 and speed_ratio < .8 and acc_ratio < .8 and jerk_ratio < .8 \
                    and spatial_delta < .015 and np.mean(gentle_motion > 1e-6) > .95
                checks[key] = {"valid": valid, "gentle_to_normal_speed_ratio": speed_ratio,
                               "gentle_to_normal_acceleration_ratio": acc_ratio,
                               "gentle_to_normal_jerk_ratio": jerk_ratio,
                               "gentle_to_normal_duration_ratio": duration_ratio,
                               "normalized_transport_path_max_delta_m": spatial_delta,
                               "gentle_transport_nonzero_motion_fraction": float(np.mean(gentle_motion > 1e-6)),
                               "place_endpoint_delta_m": float(np.linalg.norm(place_n - place_g)),
                               "pretransport_action_max_delta": pretransport_action_delta,
                               "same_initial_identity": same_initial_identity,
                               "same_object_initial": same_object_initial,
                               "same_goal_initial": same_goal_initial,
                               "same_goal_id": a["goal_id"] == b["goal_id"],
                               "same_scene_seed": int(a["scene_seed"]) == int(b["scene_seed"]),
                               "no_zero_action_padding": float(b["action_translation_zero_fraction"]) < .25}
    result = {"valid": all(item.get("valid", False) for item in checks.values()), "checks": checks}
    (output / "pair_validation.json").write_text(json.dumps(
        result, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)
    ))
    # Exercise the unmodified BPP prompt loader on a copied HDF5 whose
    # procedure metadata is changed. Its tensors must be identical because the
    # loader reads only obs/action and deliberately ignores metadata labels.
    input_contract = {"procedure_label_provided_to_model": False, "valid": False}
    try:
        import shutil
        import tempfile
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        sys.path.insert(0, str(CLEARANCE))
        from inference import load_prompt
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "src/behavior_prompting/train_network/config"), version_base=None):
            cfg = compose(config_name="libero_policy_dunetp", overrides=["task=liberogen_spatial_combination"])
        source = output / "obstacle_clearance/layout_02/narrow.hdf5"
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "procedure_renamed.hdf5"
            shutil.copy2(source, copied)
            first = load_prompt(source, cfg.task.shape_meta, "cpu")
            with h5py.File(copied, "r+") as handle:
                handle["data/demo_0"].attrs["procedure_id"] = "metadata_only_renamed"
            second = load_prompt(copied, cfg.task.shape_meta, "cpu")
            same = torch_equal_tree(first, second)
        input_contract["valid"] = bool(same)
    except Exception as exc:
        input_contract["error"] = repr(exc)
    (output / "input_contract.json").write_text(json.dumps(input_contract, indent=2))
    result["input_contract"] = input_contract
    (output / "pair_validation.json").write_text(json.dumps(
        result, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)
    ))
    failures = [{"sample": key, "reason": json.dumps(value, sort_keys=True)}
                for key, value in checks.items() if not bool(value.get("valid", False))]
    replay_path = output / "replay_validation.json"
    replay = json.loads(replay_path.read_text()) if replay_path.is_file() else {"valid": False, "reports": []}
    failures.extend({"sample": item["name"], "reason": "replay_validation_failed"}
                    for item in replay.get("reports", []) if not item.get("valid", False))
    with (output / "failed_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    (output / "validation_summary.json").write_text(json.dumps({
        "valid": bool(result["valid"] and replay.get("valid", False) and input_contract.get("valid", False)),
        "demonstrations": len(list(output.rglob("*.hdf5"))),
        "goal_successes": sum(bool(row.get("goal_success", False)) for row in rows),
        "collision_free": sum(not bool(row.get("collision", True)) for row in rows),
        "failed_samples": failures,
        "replay_valid": replay.get("valid", False),
        "pair_valid": result["valid"],
        "input_contract_valid": input_contract.get("valid", False),
    }, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    return result


def torch_equal_tree(first, second):
    """Compare the tensor leaves returned by BPP's prompt loader."""
    import torch
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(torch_equal_tree(first[key], second[key]) for key in first)
    if torch.is_tensor(first):
        return torch.equal(first, second)
    return first == second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "render", "validate", "pairs", "all"])
    parser.add_argument("--output", type=Path, default=HERE / "data" / "minimal")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in ("generate", "all"):
        generate_dataset(output)
    if args.stage in ("render", "all"):
        from render import render_dataset
        render_dataset(output)
    if args.stage in ("validate", "all"):
        replay_validate(output)
        validate_pairs(output)
    if args.stage == "pairs":
        validate_pairs(output)
    if args.stage == "validate":
        print((output / "replay_validation.json").read_text())
        print((output / "pair_validation.json").read_text())


if __name__ == "__main__":
    main()
