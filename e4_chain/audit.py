"""Verify split membership, pair indexes and modality coverage before training."""

import argparse
from collections import Counter
from math import prod
import os

import h5py
import numpy as np
import yaml

from behavior_transfer.build import BP_DEF

from behavior_transfer.sampling import frame_indices
from e2.common import read, write
from .archives import CATALOG, OUT, RAW, TEST, TRAIN, verified
from .cache import SHAPE, SIGNATURE, bank_file, checked
from .model import Chain, MODULES


def verify_data():
    rows = read(OUT / "episodes.json")
    summary = read(OUT / "data_summary.json")
    tasks = {r["task"] for r in rows}
    expected = {
        p.removesuffix("_demo.hdf5")
        for p in CATALOG["files"]
        if p.split("/")[0] in TRAIN
    }
    excluded = {
        p.split("/")[1].removesuffix("_demo.hdf5")
        for p in CATALOG["files"]
        if p.startswith(TEST + "/")
    }
    if summary["missing"] or len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("Missing archives or duplicate episode IDs")
    if set(summary["official_test_tasks"]) != excluded or not tasks <= expected:
        raise ValueError("Official split mismatch")
    heldout_archives_verified = 0
    for rel, (size, checksum) in CATALOG["files"].items():
        heldout = rel.startswith(TEST + "/")
        if heldout and not (RAW / rel).exists():
            continue
        if not verified(RAW / rel, size, checksum):
            raise FileNotFoundError(rel)
        heldout_archives_verified += int(heldout)
    annotations = {}
    slots = set()
    sequences = {1: set(), 2: set()}
    for task in sorted(expected):
        folder = OUT / "tasks" / task
        index = read(folder / "index.json")
        if index["episodes"]:
            sequences[index["steps"]].add(tuple(index["procedure"]))
        if index["task"] in excluded:
            raise ValueError(task)
        raw_link = os.path.relpath(RAW / (task + "_demo.hdf5"), folder)
        with h5py.File(folder / "episodes.h5") as f:
            for ep in index["episodes"]:
                g = f[ep["demo"]]
                n = ep["frames"]
                link = g.get("raw", getlink=True)
                if (
                    not isinstance(link, h5py.ExternalLink)
                    or link.filename != raw_link
                    or link.path != "/data/" + ep["demo"]
                    or g["raw/obs/agentview_rgb"].shape[0] != n
                    or g["actions"].shape != (n, 7)
                    or g["eef_pose"].shape != (n, 7)
                    or g["gripper"].shape != (n, 2)
                    or g["object_poses"].shape != (n, len(ep["names"]), 7)
                    or g["target"].shape != (192, 16)
                    or not ep["segments"]
                    or not np.isfinite(g["target"][:]).all()
                ):
                    raise ValueError(g.name)
                annotations[task + "/" + ep["demo"]] = tuple(ep["names"])
                slots.update(ep["names"])
    test_meta = yaml.safe_load((BP_DEF / TEST / "task_metadata.yaml").read_text())[
        "tasks"
    ]
    held_sequences = {tuple(test_meta[t]["execution_steps"]) for t in excluded}
    if held_sequences & sequences[2]:
        raise ValueError("A full held-out procedure appears among the training tasks")
    if set(annotations) != {r["id"] for r in rows}:
        raise ValueError("Data index coverage mismatch")
    fields = [
        np.asarray([r[name] for r in rows]) for name in ("task", "procedure", "layout")
    ]
    # Factorize long strings before large pair-array gathers.
    task_id, procedure, layout = (np.unique(v, return_inverse=True)[1] for v in fields)
    comparable = np.unique(
        ["/".join(annotations[r["id"]]) for r in rows], return_inverse=True
    )[1]
    populations = [Counter(), Counter()]
    for i in range(len(rows)):
        key = (task_id[i], procedure[i], comparable[i])
        populations[0][key] += 1
        populations[1][(*key, layout[i])] += 1
    possible = sum(n * n for n in populations[0].values()) - sum(
        n * n for n in populations[1].values()
    )
    if possible != summary["all_directed_pairs"]:
        raise ValueError("All-pair count mismatch")
    for split in ("train", "val"):
        pairs = np.load(OUT / f"{split}_pairs.npy")
        i, j = pairs.T
        subset = np.array([r["fold"] == split for r in rows])
        if not (
            len(pairs) == summary[split + "_pairs"] == len(np.unique(pairs, axis=0))
            and (i >= 0).all()
            and (j >= 0).all()
            and (i < len(rows)).all()
            and (j < len(rows)).all()
            and (task_id[i] == task_id[j]).all()
            and (procedure[i] == procedure[j]).all()
            and (comparable[i] == comparable[j]).all()
            and (layout[i] != layout[j]).all()
            and (subset[i] & subset[j]).all()
        ):
            raise ValueError(split + " pair validation failed")
    parameters = {
        n: sum(p.numel() for p in m.parameters())
        for n, m in Chain(len(slots)).named_children()
    }
    report = dict(
        **summary,
        parameters=parameters,
        e4_parameters=sum(parameters.values()),
        trainable_by_stage={
            s: sum(parameters[n] for n in names) for s, names in MODULES.items()
        },
        geometry_slots=sorted(slots),
        heldout_archives_verified=heldout_archives_verified,
        unique_official_step_sequences={str(n): len(v) for n, v in sequences.items()},
        checks=[
            "pinned archive SHA-256",
            "official test task and full execution-sequence exclusion",
            "episode modality shapes and finite targets",
            "relative RGB archive links",
            "pair membership, uniqueness and count",
        ],
    )
    write(OUT / "data_audit.json", report)
    return report


def verify_cache():
    from .cache import feature_key

    rows = read(OUT / "episodes.json")
    counts = Counter()
    for row in rows:
        demo = row["id"].split("/")[-1]
        key = feature_key(row["archive_sha"], demo, frame_indices(row["frames"]))
        path = bank_file(demo, row["archive_sha"])
        source = checked(path, key)
        if source not in ("E2", "JEPA"):
            raise ValueError(path)
        counts[source] += 1
    write(
        OUT / "cache_audit.json",
        dict(
            recipe=SIGNATURE,
            shape=SHAPE,
            rgb_episodes=len(rows),
            reused_e2=counts["E2"],
            encoded=counts["JEPA"],
            values_per_episode=prod(SHAPE),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="store_true")
    if parser.parse_args().cache:
        verify_cache()
    else:
        verify_data()
