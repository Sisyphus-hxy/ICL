"""Load the Chain-only training index and sample uniformly among valid partners per donor."""

from collections import defaultdict
import h5py
import numpy as np
import torch

from e2.common import read, sha
from e2.data import world_pose
from contextflow_probe.e0.ocbt_core import PoseTracker
from .archives import OUT
from .cache import bank_file, SHAPE, SIGNATURE


def scene_geometry(slots, names, poses):
    """Pack named reset poses into the fixed body order stored in a checkpoint."""
    geom = np.zeros((len(slots), 8), dtype=np.float32)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate body in geometry")
    for name, pose in zip(names, poses, strict=True):
        q = np.array(pose[3:], dtype=np.float64)
        if q[np.argmax(np.abs(q))] < 0:
            q *= -1
        geom[slots.index(name)] = [*pose[:3], *q, 1]
    if not np.isfinite(geom).all():
        raise ValueError("Invalid reset geometry")
    return geom


class Bank:
    def __init__(self, stage, device, index_root=OUT):
        self.rows = read(index_root / "episodes.json")
        self.fingerprint = tuple(
            sha(index_root / f)
            for f in (
                "episodes.json",
                "train_pairs.npy",
                "val_pairs.npy",
                "data_summary.json",
            )
        )
        groups = defaultdict(list)
        for i, r in enumerate(self.rows):
            groups[r["task"]].append(i)
        infos = {k: read(OUT / "tasks" / k / "index.json") for k in groups}
        self.names = sorted(
            {
                n
                for info in infos.values()
                for ep in info["episodes"]
                for n in ep["names"]
            }
        )
        geom = np.zeros((len(self.rows), len(self.names), 8), np.float32)
        labels = np.empty((len(self.rows), 192, 16), np.float32)
        self.x = (
            None
            if stage == "A"
            else torch.empty(
                (len(self.rows), *SHAPE), dtype=torch.float16, device=device
            )
        )
        for task, ids in groups.items():
            folder = OUT / "tasks" / task
            info = infos[task]
            if self.x is not None:
                m = read(folder / "features.json")
                if m["signature"] != SIGNATURE or m["archive_sha"] != info["sha256"]:
                    raise ValueError(f"Stale cache for {task}")
            with h5py.File(folder / "episodes.h5") as f:
                features = []
                for i, ep in zip(ids, info["episodes"], strict=True):
                    demo = ep["demo"]
                    if self.rows[i]["id"] != task + "/" + demo:
                        raise ValueError(f"Episode index mismatch: {task}/{demo}")
                    labels[i] = f[demo + "/target"][:]
                    geom[i] = scene_geometry(self.names, ep["names"], ep["initial"])
                    if self.x is not None:
                        features.append(
                            np.load(bank_file(demo, info["sha256"]), allow_pickle=False)
                        )
                if features:
                    self.x[ids] = torch.from_numpy(np.stack(features)).to(device)
        self.ids = {
            fold: np.array([i for i, r in enumerate(self.rows) if r["fold"] == fold])
            for fold in ("train", "val")
        }
        self.partners = {}
        if stage == "C":
            for fold in self.ids:
                pairs = np.load(index_root / f"{fold}_pairs.npy")
                groups = defaultdict(list)
                for i, j in pairs:
                    groups[int(i)].append(int(j))
                self.partners[fold] = groups
                self.ids[fold] = np.array(sorted(groups))
        self.mean = torch.as_tensor(
            labels[[i for i, r in enumerate(self.rows) if r["fold"] == "train"]].mean(
                (0, 1), dtype=np.float64
            ),
            dtype=torch.float32,
            device=device,
        )
        self.std = torch.as_tensor(
            labels[[i for i, r in enumerate(self.rows) if r["fold"] == "train"]]
            .std((0, 1), dtype=np.float64)
            .clip(min=0.01),
            dtype=torch.float32,
            device=device,
        )
        self.mean[3:8], self.std[3:8] = 0, 1
        self.y = (torch.from_numpy(labels).to(device) - self.mean) / self.std
        self.geometry = torch.from_numpy(geom).to(device)
        self.device = device

    def batch(self, ids, fold, rng):
        ix = torch.as_tensor(ids, device=self.device)
        if self.partners:
            partners = self.partners[fold]
            j = torch.tensor(
                [rng.choice(partners[int(i)]) for i in ids], device=self.device
            )
            return (self.y[j], self.x[ix], self.geometry[ix], self.geometry[j])
        return (self.y[ix], None if self.x is None else self.x[ix])


def tracker(prediction, initial):
    """Build the existing E0 pose tracker from a single denormalized decoder output."""
    plan = np.asarray(prediction)
    quat = plan[:, 3:7]
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    unit = np.tile([0.0, 0.0, 0.0, 1.0], (len(plan), 1))
    unit = np.divide(quat, norm, out=unit, where=norm > 1e-7)
    poses = world_pose(np.c_[plan[:, :3], unit], initial)
    return PoseTracker(poses, np.where(plan[:, 7] > 0, 1.0, -1.0))
