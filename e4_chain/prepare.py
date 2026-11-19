"""Index successful RGB demonstrations from the three official Chain training views."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import hashlib
import os
import time

import h5py
import numpy as np
import yaml

from behavior_transfer.build import BP_DEF, bpp_episode
from behavior_transfer.manifest import ANGLE_TOL, XYZ_TOL, distances
from contextflow_probe.e0.ocbt_core import segment_donor
from e2.common import read, write, table
from e2.data import relative_labels
from .archives import CATALOG, RAW, OUT, TRAIN, DOUBLE, TEST, verified

RECIPE = 1


def official_jobs():
    config = yaml.safe_load(
        (BP_DEF.parents[2] / "config/task/liberogen_goal_chain.yaml").read_text()
    )
    if tuple(config["dataset_splits"]) != TRAIN or config["unseen_dataset_splits"] != [
        TEST
    ]:
        raise ValueError("Official Chain views changed; review the pinned release")
    tasks = {
        v: yaml.safe_load((BP_DEF / v / "task_metadata.yaml").read_text())["tasks"]
        for v in (*TRAIN, TEST)
    }
    jobs = []
    for rel, (size, checksum) in sorted(CATALOG["files"].items()):
        view, filename = rel.split("/")
        name = filename.removesuffix("_demo.hdf5")
        if name not in tasks[view]:
            raise ValueError(rel)
        if view != TEST:
            if view not in TRAIN or name in tasks[TEST]:
                raise ValueError(f"Held-out/unknown task: {rel}")
            jobs.append((view, name, tasks[view][name], size, checksum))
    return jobs, sorted(tasks[TEST])


def pack(job):
    view, task, metadata, size, checksum = job
    raw = RAW / view / (task + "_demo.hdf5")
    dest = OUT / "tasks" / view / task
    path = dest / "episodes.h5"
    if not verified(raw, size, checksum):
        return False
    summary = dest / "index.json"
    raw_link = os.path.relpath(raw, dest)
    if path.exists() and summary.is_file():
        old = read(summary)
        if (
            old["sha256"] == checksum
            and old["recipe"] == RECIPE
            and old.get("raw_link") == raw_link
        ):
            return True
    dest.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    steps = metadata["execution_steps"]
    params = dict(
        raw=str(raw),
        raw_link=raw_link,
        goal=(BP_DEF / view / (task + ".bddl")).read_text().split("(:goal")[-1],
        procedure_metadata=metadata,
    )
    episodes, excluded = [], {}
    try:
        with h5py.File(raw) as src, h5py.File(tmp, "w") as out:
            for demo in sorted(src["data"], key=lambda k: int(k.split("_")[-1])):
                g = src["data"][demo]
                if not (g["dones"][-1] and g["rewards"][-1] > 0):
                    excluded[demo] = "unsuccessful"
                    continue
                if "obs/agentview_rgb" not in g or "actions" not in g:
                    excluded[demo] = "missing RGB/actions"
                    continue
                if g["obs/agentview_rgb"].dtype != np.uint8:
                    raise ValueError(f"Non-uint8 RGB in {raw}/{demo}")
                dst = out.create_group(demo)
                row = dict(id=f"{view}/{task}/{demo}")
                bpp_episode(params, row, dst, g, None)
                pose, action, fingers = (
                    dst[k][:] for k in ("eef_pose", "actions", "gripper")
                )
                lag = row["action_offset"]
                dst["target"] = relative_labels(pose, action, fingers, lag)
                # Post-action EEF/body trajectories. The final next-state is absent
                # in post-action RGB recordings and cannot support a body motion label.
                n = len(action) - 1
                obs = slice(1 - lag, 1 - lag + n)
                objects = dst["object_poses"][obs]
                names = row["names"]
                movable = [
                    k
                    for k in names
                    if k not in ("flat_stove_1", "wooden_cabinet_1", "wine_rack_1")
                ]
                seg = segment_donor(pose[obs], objects, names, movable, action[:n, 6])
                episodes.append(
                    dict(
                        demo=demo,
                        frames=row["frames"],
                        action_offset=lag,
                        action_sha=row["action_sha256"],
                        image_convention=row["image_convention"],
                        names=names,
                        initial=row["initial"],
                        segments=[
                            [int(a), int(b), anchor] for a, b, anchor in seg["segments"]
                        ],
                        gripper_windows=[
                            [int(i) for i in w["window"]] for w in seg["windows"]
                        ],
                    )
                )
        tmp.replace(path)
        write(
            summary,
            dict(
                recipe=RECIPE,
                view=view,
                task=task,
                steps=1 + int(view == DOUBLE),
                procedure=steps,
                metadata=metadata,
                sha256=checksum,
                raw_link=raw_link,
                episodes=episodes,
                excluded=excluded,
            ),
        )
        print("indexed", view, task, len(episodes), flush=True)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def split_layouts(rows, task):
    """Connected components prevent nearly identical layouts leaking across folds."""
    connected = list(range(len(rows)))

    def root(i):
        while connected[i] != i:
            i = connected[i]
        return i

    for i, r in enumerate(rows):
        for j, other in enumerate(rows[:i]):
            if r["names"] != other["names"]:
                continue
            xyz, angle = distances(np.array(r["initial"]), np.array(other["initial"]))
            if (xyz <= XYZ_TOL and angle <= ANGLE_TOL) or r["action_sha"] == other[
                "action_sha"
            ]:
                connected[root(i)] = root(j)
    groups = sorted(
        {root(i) for i in range(len(rows))},
        key=lambda i: hashlib.sha256(f"e4/{task}/{i}".encode()).digest(),
    )
    nval = (
        min(len(groups) - 2, max(2, round(len(groups) * 0.1)))
        if len(groups) >= 4
        else 0
    )
    val = set(groups[:nval])
    return [(root(i), "val" if root(i) in val else "train") for i in range(len(rows))]


def assemble(jobs, held, index_root=OUT):
    rows, pairs, counts, absent = [], defaultdict(list), [], []
    for view, task, _, _, checksum in jobs:
        name = f"{view}/{task}"
        path = OUT / "tasks" / name / "index.json"
        if not path.is_file():
            absent.append(name)
            continue
        info = read(path)
        eps = info["episodes"]
        start = len(rows)
        folds = split_layouts(eps, name)
        procedures = []
        for r, (layout, fold) in zip(eps, folds):
            procedure = " > ".join(info["procedure"])
            if info["metadata"]["is_existing_task"]:
                # Human recordings may include re-grasps or a different interaction order.
                anchors = [a for a, _ in groupby(seg[2] for seg in r["segments"])]
                procedure += (
                    " | "
                    + ">".join(anchors)
                    + f" | closures={len(r['gripper_windows'])}"
                )
            procedures.append(procedure)
            rows.append(
                dict(
                    id=name + "/" + r["demo"],
                    task=name,
                    layout=layout,
                    procedure=procedure,
                    fold=fold,
                    steps=info["steps"],
                    frames=r["frames"],
                    convention=r["image_convention"],
                    archive_sha=checksum,
                )
            )
        cnt = Counter()
        for i, (li, fi) in enumerate(folds):
            for j, (lj, fj) in enumerate(folds):
                if (
                    li == lj
                    or procedures[i] != procedures[j]
                    or eps[i]["names"] != eps[j]["names"]
                ):
                    continue
                cnt["all"] += 1
                if fi == fj:
                    pairs[fi].append((start + i, start + j))
                    cnt[fi] += 1
        counts.append(
            dict(
                task=name,
                steps=info["steps"],
                episodes=len(eps),
                train=sum(f == "train" for _, f in folds),
                layouts=len(set(folds)),
                pairs=cnt["all"],
                train_pairs=cnt["train"],
                val_pairs=cnt["val"],
                excluded=len(info["excluded"]),
            )
        )
    from behavior_transfer.dataset import save_feature

    for fold in ("train", "val"):
        save_feature(
            index_root / f"{fold}_pairs.npy",
            np.asarray(pairs[fold], dtype=np.int32).reshape(-1, 2),
        )
    write(index_root / "episodes.json", rows)
    table(index_root / "task_summary.csv", counts)
    by_view = {}
    for view in TRAIN:
        part = [c for c in counts if c["task"].split("/")[0] == view]
        by_view[view] = dict(
            tasks=sum(c["train"] > 0 for c in part),
            transfer_train_tasks=sum(c["train_pairs"] > 0 for c in part),
            rgb_episodes=sum(c["episodes"] for c in part),
            train_episodes=sum(c["train"] for c in part),
            all_directed_pairs=sum(c["pairs"] for c in part),
            train_pairs=sum(c["train_pairs"] for c in part),
            val_pairs=sum(c["val_pairs"] for c in part),
        )
    write(
        index_root / "data_summary.json",
        dict(
            revision=CATALOG["revision"],
            recipe=RECIPE,
            missing=absent,
            official_test_tasks=held,
            tasks=sum(c["train"] > 0 for c in counts),
            transfer_train_tasks=sum(c["train_pairs"] > 0 for c in counts),
            by_view=by_view,
            single_tasks=sum(c["steps"] == 1 and c["episodes"] > 0 for c in counts),
            double_tasks=sum(c["steps"] == 2 and c["episodes"] > 0 for c in counts),
            rgb_episodes=len(rows),
            train_episodes=sum(c["train"] for c in counts),
            single_episodes=sum(c["episodes"] for c in counts if c["steps"] == 1),
            double_episodes=sum(c["episodes"] for c in counts if c["steps"] == 2),
            excluded=sum(c["excluded"] for c in counts),
            all_directed_pairs=sum(c["pairs"] for c in counts),
            train_pairs=len(pairs["train"]),
            val_pairs=len(pairs["val"]),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--follow", action="store_true", help="index new archives as downloads finish"
    )
    args = p.parse_args()
    jobs, held = official_jobs()
    pending = jobs.copy()
    with ProcessPoolExecutor(args.workers) as pool:
        while pending:
            pending = [j for j, ok in zip(pending, pool.map(pack, pending)) if not ok]
            if pending and not args.follow:
                raise FileNotFoundError(
                    f"{len(pending)} training archives pending; run archives or prepare --follow"
                )
            if pending:
                time.sleep(15)
    assemble(jobs, held)


if __name__ == "__main__":
    main()
