"""Physics-step recorder for the procedural-constraint demonstrations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


DT = 1.0 / 20.0


def _finite_metrics(positions: np.ndarray, dt: float = DT) -> dict:
    if len(positions) < 2:
        return {
            "eef_speed_mean_m_s": 0.0,
            "eef_speed_max_m_s": 0.0,
            "eef_acceleration_rms_m_s2": 0.0,
            "eef_acceleration_max_m_s2": 0.0,
            "eef_jerk_rms_m_s3": 0.0,
            "eef_jerk_max_m_s3": 0.0,
        }
    velocity = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / dt if len(velocity) > 1 else np.zeros((0, 3))
    jerk = np.diff(acceleration, axis=0) / dt if len(acceleration) > 1 else np.zeros((0, 3))
    acc_norm = np.linalg.norm(acceleration, axis=1) if len(acceleration) else np.zeros(1)
    jerk_norm = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros(1)
    return {
        "eef_speed_mean_m_s": float(speed.mean()),
        "eef_speed_max_m_s": float(speed.max()),
        "eef_acceleration_rms_m_s2": float(np.sqrt(np.mean(acc_norm ** 2))),
        "eef_acceleration_max_m_s2": float(acc_norm.max()),
        "eef_jerk_rms_m_s3": float(np.sqrt(np.mean(jerk_norm ** 2))),
        "eef_jerk_max_m_s3": float(jerk_norm.max()),
    }


class Recorder:
    """Record actions and observations while the real MuJoCo env is stepping."""

    def __init__(self, output: Path, name: str, metadata: dict):
        self.output = Path(output)
        self.name = name
        self.metadata = dict(metadata)
        self.phase = "unknown"
        self.rows = []
        self.actions = []
        self.states = []
        self.observations = []
        self.gripper_commands = []
        self.collision_events = []
        self.substep_min = {"object": 2.0, "gripper": 2.0}
        self.phase_collision = False
        self.video = None

    def __getattr__(self, name):
        return getattr(self.scene.env, name)

    def start(self, scene):
        self.scene = scene
        self.identity = scene.initial_identity()
        self.latest_obs = scene.obs()
        self.identity["object_initial_pos"] = scene.object_pos().tolist()
        self.identity["goal_initial_pos"] = scene.goal_pos().tolist()
        self._physics_step = scene.env.sim.step
        scene.env.sim.step = self._monitored_step
        (self.output / self.name).parent.mkdir(parents=True, exist_ok=True)

    def _monitored_step(self):
        self._physics_step()
        s = self.scene
        for key, geoms in (("object", s.object_geoms), ("gripper", s.gripper_geoms)):
            self.substep_min[key] = min(self.substep_min[key], s.surface_clearance(geoms))
        collisions = s.contacts(transport=self.phase == "transport")
        if collisions:
            self.phase_collision = True
            self.collision_events.append({
                "control_step": len(self.actions),
                "phase": self.phase,
                "sim_time": float(s.d.time),
                "contacts": collisions,
            })

    def _capture_obs(self, obs):
        keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        return {key: obs[key].copy() for key in keys if key in obs}

    def step(self, action):
        s = self.scene
        obs = self.latest_obs
        self.observations.append(self._capture_obs(obs))
        self.states.append(s.env.sim.get_state().flatten().copy())
        self.gripper_commands.append(s.env.robots[0].gripper.current_action.copy())
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.substep_min = {"object": 2.0, "gripper": 2.0}
        self.phase_collision = False
        result = s.env.step(np.asarray(action, dtype=np.float64))
        self.latest_obs = result[0]
        p = self.latest_obs["robot0_eef_pos"]
        q = self.latest_obs["robot0_eef_quat"]
        obj_pos = s.object_pos()
        # MuJoCo stores xquat as wxyz; scipy/robosuite observations use xyzw.
        obj_quat_mj = s.d.xquat[s.object_body].copy()
        obj_quat = np.r_[obj_quat_mj[1:], obj_quat_mj[0]]
        obstacle = np.asarray(self.metadata["obstacle_position"], dtype=np.float64)
        tilt = float(np.arccos(np.clip(Rotation.from_quat(obj_quat).as_matrix()[2, 2], -1.0, 1.0)))
        self.rows.append({
            "step": len(self.actions) - 1,
            "time_s": len(self.actions) * DT,
            "phase": self.phase,
            "eef_x": float(p[0]), "eef_y": float(p[1]), "eef_z": float(p[2]),
            "eef_qx": float(q[0]), "eef_qy": float(q[1]),
            "eef_qz": float(q[2]), "eef_qw": float(q[3]),
            "object_x": float(obj_pos[0]), "object_y": float(obj_pos[1]),
            "object_z": float(obj_pos[2]),
            "object_qx": float(obj_quat[0]), "object_qy": float(obj_quat[1]),
            "object_qz": float(obj_quat[2]), "object_qw": float(obj_quat[3]),
            "object_rel_x": float(obj_pos[0] - obstacle[0]),
            "object_rel_y": float(obj_pos[1] - obstacle[1]),
            "object_rel_z": float(obj_pos[2] - obstacle[2]),
            "object_clearance_m": float(self.substep_min["object"]),
            "gripper_clearance_m": float(self.substep_min["gripper"]),
            "collision": int(self.phase_collision),
            "goal_success": int(s.env._check_success()),
        })
        return result

    def _write_trajectory(self):
        path = self.output / f"{self.name}_trajectory.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self, scene):
        scene.env.sim.step = self._physics_step
        self._write_trajectory()
        (self.output / f"{self.name}_contacts.json").write_text(
            json.dumps(self.collision_events, indent=2)
        )
        summary = self.summary()
        with h5py.File(self.output / f"{self.name}.hdf5", "w") as handle:
            data = handle.create_group("data")
            demo = data.create_group("demo_0")
            for key, value in self.metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                demo.attrs[key] = value
            for key, value in summary.items():
                if key == "initial_identity":
                    continue
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                if isinstance(value, (bool, np.bool_)):
                    value = bool(value)
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    demo.attrs[key] = value
            demo.attrs["model_file"] = scene.model_xml
            demo.attrs["seed"] = int(self.metadata["scene_seed"])
            demo.attrs["initial_identity"] = json.dumps(self.identity)
            demo.attrs["num_samples"] = len(self.actions)
            demo.attrs["observation_alignment"] = "post_action"
            demo.attrs["state_alignment"] = "pre_action"
            demo.create_dataset("states", data=np.asarray(self.states), compression="gzip", compression_opts=1)
            demo.create_dataset("terminal_state", data=scene.env.sim.get_state().flatten())
            demo.create_dataset("initial_ee_pos", data=self.observations[0]["robot0_eef_pos"])
            demo.create_dataset("actions", data=np.asarray(self.actions), compression="gzip", compression_opts=1)
            demo.create_dataset("gripper_command_state", data=np.asarray(self.gripper_commands))
            obs_group = demo.create_group("obs")
            post = self.observations[1:] + [self.latest_obs]
            obs_group.create_dataset("ee_pos", data=np.asarray([o["robot0_eef_pos"] for o in post]))
            obs_group.create_dataset(
                "ee_ori", data=Rotation.from_quat(np.asarray([o["robot0_eef_quat"] for o in post])).as_rotvec()
            )
            obs_group.create_dataset("gripper_states", data=np.asarray([o["robot0_gripper_qpos"] for o in post]))
            obs_group.create_dataset("object_pos", data=np.asarray([[r["object_x"], r["object_y"], r["object_z"]] for r in self.rows]))
            obs_group.create_dataset(
                "object_ori",
                data=Rotation.from_quat(np.asarray([[r["object_qx"], r["object_qy"], r["object_qz"], r["object_qw"]] for r in self.rows])).as_rotvec(),
            )
            demo.create_dataset("phases", data=np.asarray([r["phase"].encode() for r in self.rows]))
        (self.output / f"{self.name}_statistics.csv").open("w").write(
            "metric,value\n" + "\n".join(f"{k},{v}" for k, v in summary.items()) + "\n"
        )
        return summary

    def summary(self):
        positions = np.asarray([[r["eef_x"], r["eef_y"], r["eef_z"]] for r in self.rows])
        transport = np.asarray([r["phase"] == "transport" for r in self.rows])
        idx = np.where(transport)[0]
        transport_pos = positions[idx]
        travel = np.linalg.norm(np.diff(np.vstack([self.observations[0]["robot0_eef_pos"], positions]), axis=0), axis=1)
        transport_travel = travel[transport]
        object_quat = np.asarray([[r["object_qx"], r["object_qy"], r["object_qz"], r["object_qw"]] for r in self.rows])
        tilt = np.arccos(np.clip(Rotation.from_quat(object_quat).as_matrix()[:, 2, 2], -1.0, 1.0))
        motion = np.linalg.norm(np.asarray(self.actions)[:, :3], axis=1)
        metrics = _finite_metrics(transport_pos)
        metrics.update({
            "name": self.name,
            "task_id": self.metadata["task_id"],
            "procedure_id": self.metadata["procedure_id"],
            "layout_id": self.metadata["layout_id"],
            "goal_id": self.metadata["goal_id"],
            "scene_seed": int(self.metadata["scene_seed"]),
            "goal_success": bool(self.rows[-1]["goal_success"]),
            "collision": bool(any(r["collision"] for r in self.rows)),
            "collision_control_steps": int(sum(r["collision"] for r in self.rows)),
            "object_min_clearance_m": float(min(r["object_clearance_m"] for r in self.rows if r["phase"] == "transport")),
            "gripper_min_clearance_m": float(min(r["gripper_clearance_m"] for r in self.rows if r["phase"] == "transport")),
            "max_object_tilt_deg": float(np.degrees(tilt[transport].max())),
            "path_length_m": float(travel.sum()),
            "transport_path_length_m": float(transport_travel.sum()),
            "transport_duration_s": float(len(transport_pos) * DT),
            "control_steps": len(self.rows),
            "duration_s": len(self.rows) * DT,
            "action_translation_nonzero_fraction": float(np.mean(motion > 1e-7)),
            "action_translation_zero_fraction": float(np.mean(motion <= 1e-7)),
            "initial_identity": self.identity,
            "object_initial_pos": self.identity["object_initial_pos"],
            "goal_initial_pos": self.identity["goal_initial_pos"],
            "obstacle_position": self.metadata["obstacle_position"],
        })
        return metrics
