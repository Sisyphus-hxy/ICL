"""Extract GT demo data (actions + first-frame image) from the LeRobot parquet dataset.

Runs with the ContextFlow .venv python (3.11, has pyarrow/PIL). The libero sim
env (python 3.8) has no parquet reader, so replay consumes the extracted .npz
files instead.

Usage:
    .venv/bin/python scripts/extract_gt_demo_data.py \
        --dataset-root /data2/huangxiangyun/.cache/huggingface/lerobot/physical-intelligence/libero \
        --out-dir data/libero_incontext/gt_demo_cache
"""

import io
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tyro
from PIL import Image

# Same list as openpi.training.config_libero.LIBERO_UNSEEN_TASKS, duplicated here
# so this script runs without importing the training deps.
LIBERO_UNSEEN_TASKS = [
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "pick up the milk and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
]


def main(
    dataset_root: str = "/data2/huangxiangyun/.cache/huggingface/lerobot/physical-intelligence/libero",
    out_dir: str = "data/libero_incontext/gt_demo_cache",
    unseen_only: bool = True,
) -> None:
    root = pathlib.Path(dataset_root)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes = []
    for line in (root / "meta" / "episodes.jsonl").open():
        line = line.strip()
        if line:
            episodes.append(json.loads(line))

    written = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        tasks = ep["tasks"]
        assert len(tasks) == 1
        task = tasks[0]
        if unseen_only and task not in LIBERO_UNSEEN_TASKS:
            continue

        chunk = ep_idx // 1000  # chunks_size = 1000 per info.json
        pq_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        table = pq.read_table(pq_path, columns=["image", "state", "actions", "task_index", "episode_index"])
        n = table.num_rows
        assert n == int(ep["length"]), f"{pq_path}: {n} rows != length {ep['length']}"

        actions = np.asarray(table.column("actions").combine_chunks().to_pylist(), dtype=np.float32)
        first = table.slice(0, 1).to_pylist()[0]
        frame0 = np.asarray(Image.open(io.BytesIO(first["image"]["bytes"])).convert("RGB"), dtype=np.uint8)
        state0 = np.asarray(first["state"], dtype=np.float32)
        task_index = int(first["task_index"])

        np.savez_compressed(
            out / f"ep_{ep_idx:06d}.npz",
            episode_index=np.int64(ep_idx),
            task_index=np.int64(task_index),
            task=np.str_(task),
            n_frames=np.int64(n),
            actions=actions,
            frame0=frame0,
            state0=state0,
        )
        written.append({"episode_index": ep_idx, "task_index": task_index, "task": task, "n_frames": n})
        print(f"[{len(written):3d}] ep={ep_idx:5d} task_index={task_index:3d} frames={n:4d} {task[:70]}")

    (out / "index.json").write_text(json.dumps(written, indent=2))
    print(f"Wrote {len(written)} episodes to {out}")


if __name__ == "__main__":
    tyro.cli(main)
