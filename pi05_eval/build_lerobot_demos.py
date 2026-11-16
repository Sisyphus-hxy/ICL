"""Convert pi0.5 successful rollouts into a LeRobot v2.0 dataset for ContextFlow.

Reads ``rollout.npz`` files produced by ``rollout.py --mode contextflow`` and
writes episodes into a new dataset ``pi05/libero_demos`` under
``$LEROBOT_HOME`` with the same feature schema (image/wrist_image 256x256
embedded PNGs, state f32[8], actions f32[7], fps=10) and the same task_index
mapping as ``physical-intelligence/libero``, so
``CustomLeRobotDataset.load_incontext_demonstration`` resolves demos by the
task_index the eval client sends.

Frame convention matches (obs_t, action_t): frame i pairs the post-action
image/state of step i-1 with the action executed at step i; the first frame
uses the settled obs after the num_steps_wait dummy prefix. The final
post-done frame has no following action and is dropped.

Run with the ContextFlow repo env:
    ContextFlow/.venv/bin/python pi05_eval/build_lerobot_demos.py \
        --rollouts <rollout_root> [--repo-id pi05/libero_demos]
"""

import dataclasses
import json
import math
import os
import pathlib

import numpy as np
import tyro

LEROBOT_HOME = pathlib.Path(
    os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
SRC_DATASET = LEROBOT_HOME / "physical-intelligence" / "libero"


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclasses.dataclass
class Args:
    rollouts: str  # root containing <suite>/<task>/init_XX/rollout.npz
    repo_id: str = "pi05/libero_demos"
    src_tasks: str = str(SRC_DATASET / "meta" / "tasks.jsonl")
    only_success: bool = True


def _episode_frames(npz_path):
    """Reconstruct (image, wrist, state8, action7) frames from a rollout.npz."""
    d = np.load(npz_path)
    meta = json.loads(str(d["meta_json"]))
    wait = int(meta.get("num_steps_wait", 10))
    actions = d["actions"][wait:]                      # policy actions (m,7)
    agentview = d["agentview_image"]                   # raw (upside-down) obs
    wrist = d["eye_in_hand_image"]
    eef_pos = d["eef_pos"]
    eef_quat = d["eef_quat"]
    gripper = d["gripper_qpos"]
    m = len(actions)
    # post-action obs frames o_0..o_{m-1}: obs indices wait-1 .. wait+m-2
    lo, hi = wait - 1, wait + m - 1                    # exclusive hi
    frames = []
    for j in range(lo, hi):
        img = np.ascontiguousarray(agentview[j][::-1, ::-1])
        wr = np.ascontiguousarray(wrist[j][::-1, ::-1])
        state = np.concatenate(
            [eef_pos[j], _quat2axisangle(eef_quat[j]), gripper[j]]).astype(np.float32)
        frames.append({
            "image": img, "wrist_image": wr, "state": state,
            "actions": actions[j - lo].astype(np.float32),
            "timestamp": np.float32((j - lo) / 10.0),
        })
    return frames, meta


def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import append_jsonlines

    root = LEROBOT_HOME / args.repo_id
    npzs = sorted(pathlib.Path(args.rollouts).rglob("rollout.npz"))
    episodes = []
    for p in npzs:
        result = p.parent / "result.json"
        ok = result.exists() and json.loads(result.read_text()).get("success")
        if args.only_success and not ok:
            continue
        episodes.append(p)
    print(f"found {len(episodes)} successful rollouts under {args.rollouts}")

    features = {
        "image": {"dtype": "image", "shape": (256, 256, 3),
                  "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=10, root=root, robot_type="panda",
        features=features, use_videos=False,
    )

    # Pre-register every task of the source dataset so task_index matches what
    # the eval client (physical-intelligence/libero tasks.jsonl) sends.
    src_tasks = []
    for line in pathlib.Path(args.src_tasks).read_text().splitlines():
        line = line.strip()
        if line:
            src_tasks.append(json.loads(line))
    for t in src_tasks:
        append_jsonlines(t, root / "meta" / "tasks.jsonl")
    ds.meta.tasks.update({int(t["task_index"]): t["task"] for t in src_tasks})
    ds.meta.task_to_task_index.update({t["task"]: int(t["task_index"]) for t in src_tasks})

    written = 0
    for p in episodes:
        frames, meta = _episode_frames(p)
        if not frames:
            print(f"  skip (0 frames): {p}")
            continue
        task = meta["task"]
        assert task in ds.meta.task_to_task_index, f"task not in src tasks.jsonl: {task}"
        for fr in frames:
            ds.add_frame(fr)
        ds.save_episode(task=task, encode_videos=False)
        written += 1
        print(f"  episode {written}: {task} ({len(frames)} frames) <- {p.parent}")

    # info.json bookkeeping: total_tasks reflects the copied task table.
    info = ds.meta.info
    info["total_tasks"] = len(ds.meta.tasks)
    from lerobot.common.datasets.utils import write_json
    write_json(info, root / "meta" / "info.json")
    print(f"wrote {written} episodes -> {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
