"""Replay saved states to add aligned RGB observations and videos."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import h5py
import numpy as np

from behavior_prompting.train_network.utils.video_recorder import VideoRecorder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clearance"))
from scene import Scene  # noqa: E402
import scene as clearance_scene  # noqa: E402
from generate import set_layout  # noqa: E402


def render_dataset(output: Path, stride: int = 4):
    output = Path(output)
    scene = Scene(resolution=128, record_images=True)
    reports = []
    try:
        for path in sorted(output.rglob("*.hdf5")):
            with h5py.File(path, "r+") as handle:
                demo = handle["data/demo_0"]
                obstacle = np.asarray(json.loads(demo.attrs["obstacle_position"]))
                set_layout(scene, demo.attrs["layout_id"], obstacle)
                states = demo["states"][:]
                terminal = demo["terminal_state"][:]
                post_states = np.concatenate([states[1:], terminal[None]], axis=0)
                video = VideoRecorder.create_h264(fps=20 // stride, thread_count=1)
                video_path = path.with_suffix(".mp4")
                video.start(str(video_path))
                agent = []
                wrist = []
                try:
                    for index in range(0, len(post_states), stride):
                        scene.env.sim.set_state_from_flattened(post_states[index])
                        scene.d.mocap_pos[scene.mocap_id] = obstacle
                        scene.d.mocap_quat[scene.mocap_id] = [1, 0, 0, 0]
                        scene.env.sim.forward()
                        obs = scene.obs()
                        a = obs["agentview_image"].copy()
                        w = obs["robot0_eye_in_hand_image"].copy()
                        agent.append(a)
                        wrist.append(w)
                        video.write_frame(np.concatenate([a[::-1], w[::-1]], axis=1).copy())
                finally:
                    video.stop()
                for key, value in (("agentview_rgb", agent), ("eye_in_hand_rgb", wrist)):
                    if key in demo["obs"]:
                        del demo["obs"][key]
                    dense = np.repeat(np.asarray(value), stride, axis=0)[: len(states)]
                    demo["obs"].create_dataset(key, data=dense, compression="gzip", compression_opts=1)
                demo.attrs["rgb_capture_stride_control_steps"] = int(stride)
                demo.attrs["rgb_storage"] = "post-action 5Hz zero-order hold"
                reports.append({"name": str(path.relative_to(output).with_suffix("")),
                                "video": video_path.name,
                                "frames": len(agent), "video_fps": 20 // stride})
    finally:
        scene.close()
    (output / "render_validation.json").write_text(json.dumps({"valid": True, "reports": reports}, indent=2))
    make_pair_plots(output)
    validate_rgb(output)


def validate_rgb(output: Path):
    """Check that rendered RGB has one post-action frame per action."""
    import csv
    result = {"valid": True, "reports": [], "same_layout_initial_rgb": {}}
    hashes = {}
    for path in sorted(Path(output).rglob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            demo = handle["data/demo_0"]
            n = len(demo["actions"])
            ok = all(key in demo["obs"] and demo["obs"][key].shape[0] == n and demo["obs"][key].dtype == np.uint8
                     for key in ("agentview_rgb", "eye_in_hand_rgb"))
            digest = hashlib.sha256(demo["obs/agentview_rgb"][0].tobytes() + demo["obs/eye_in_hand_rgb"][0].tobytes()).hexdigest()
            hashes[str(path.relative_to(output).with_suffix(""))] = digest
            result["reports"].append({"name": str(path.relative_to(output).with_suffix("")),
                                      "frames": n, "valid": ok})
            result["valid"] = result["valid"] and ok and path.with_suffix(".mp4").is_file()
    for task, procedures in (("obstacle_clearance", ("narrow", "wide")),
                             ("gentle_transport", ("normal", "gentle"))):
        for index in range(5):
            layout = f"layout_{index:02d}"
            a = f"{task}/{layout}/{procedures[0]}"
            b = f"{task}/{layout}/{procedures[1]}"
            same = hashes.get(a) == hashes.get(b)
            result["same_layout_initial_rgb"][f"{task}/{layout}"] = same
            result["valid"] = result["valid"] and same
    (Path(output) / "rgb_alignment_validation.json").write_text(json.dumps(result, indent=2))
    return result


def make_pair_plots(output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    rows = list(csv.DictReader((output / "statistics.csv").open()))
    for task, procedures in (("obstacle_clearance", ("narrow", "wide")),
                             ("gentle_transport", ("normal", "gentle"))):
        for index in range(5):
            layout = f"layout_{index:02d}"
            pair = [r for r in rows if r["task_id"] == task and r["layout_id"] == layout]
            if len(pair) != 2:
                continue
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for row in pair:
                data = np.genfromtxt(output / f"{row['name']}_trajectory.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
                transport = data["phase"] == "transport"
                axes[0].plot(data["eef_x"][transport], data["eef_y"][transport], label=row["procedure_id"], linewidth=2)
                axes[1].plot(data["time_s"], data["eef_z"], label=row["procedure_id"], linewidth=1.5)
            obstacle = np.asarray(json.loads(pair[0]["obstacle_position"]))
            axes[0].scatter([obstacle[0]], [obstacle[1]], c="tab:red", marker="x", s=80, label="constraint object")
            axes[0].set(xlabel="EEF x (m)", ylabel="EEF y (m)", title=f"{task} / {layout}", aspect="equal")
            axes[1].set(xlabel="time (s)", ylabel="EEF z (m)", title="transport timing")
            axes[0].legend()
            axes[1].legend()
            figure.tight_layout()
            layout_dir = output / task / layout
            layout_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(layout_dir / "pair.png", dpi=160)
            plt.close(figure)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_dataset(args.output)
