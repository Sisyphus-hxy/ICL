import csv
from collections import defaultdict
import sys
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from .common import DATA, E1, OUT, read, sha

sys.path.insert(0, str(E1))
from e1_common import resample_trajectory


def relative_labels(pose, act, fingers, offset):
    """Post-action pose paired with its applied OSC command, in the first observed EEF frame."""
    initial = pose[0]
    if offset == 0:
        pose, act, fingers = pose[1:], act[:-1], fingers[1:]
    r0 = Rotation.from_quat(initial[3:]).inv()
    local = np.c_[
        r0.apply(pose[:, :3] - initial[:3]),
        (r0 * Rotation.from_quat(pose[:, 3:])).as_quat(),
    ]
    target, fq = resample_trajectory(local, act[:, 6], fingers)
    raw = np.stack(
        [
            np.interp(np.linspace(0, len(act) - 1, 192), np.arange(len(act)), c)
            for c in act[:, :6].T
        ],
        axis=1,
    )
    return np.c_[target[:, :8], raw, fq].astype("float32")


def world_pose(relative, initial):
    r = Rotation.from_quat(initial[3:])
    return np.c_[
        r.apply(relative[:, :3]) + initial[:3],
        (r * Rotation.from_quat(relative[:, 3:7])).as_quat(),
    ]


def geometry(names, poses):
    if len(names) != len(set(names)) or len(names) != len(poses) or len(names) > 16:
        raise ValueError(names)
    order = np.argsort(names)
    return tuple(np.asarray(names)[order]), np.asarray(poses)[order]


def difference(gd, gq):
    dn, d = gd
    qn, q = gq
    if dn != qn:
        raise ValueError((dn, qn))
    a = np.zeros((16, 6), "f")
    a[: len(d), :3] = (q[:, :3] - d[:, :3]) / 0.03
    a[: len(d), 3:] = (
        Rotation.from_quat(q[:, 3:]) * Rotation.from_quat(d[:, 3:]).inv()
    ).as_rotvec() / 0.15
    return a.flatten()


class Data:
    def __init__(self, features=True):
        cache = OUT / "cache"
        verify = read(cache / "verified.json")
        assert verify["inventory_sha"] == sha(cache / "inventory.json")
        jobs = read(cache / "inventory.json")["jobs"]
        split = read(OUT / "split.json")
        assert split["train_role_sha"] == sha(DATA / "roles/train_episodes.json")
        self.ids = split["train"] + split["val"]
        self.index = {k: i for i, k in enumerate(self.ids)}
        self.split = split
        self.nt = len(split["train"])
        rows = {r["id"]: r for r in read(DATA / "roles/train_episodes.json")}
        self.groups = defaultdict(list)
        self.geom = {}
        labels = []
        if features:
            self.x = torch.empty(
                (len(self.ids), 32, 8, 8, 768), device="cuda", dtype=torch.float16
            )
        for i, eid in enumerate(self.ids):
            row = rows[eid]
            with h5py.File(DATA / row["path"]) as f:
                labels.append(
                    relative_labels(
                        f["eef_pose"][:],
                        f["actions"][:],
                        f["gripper"][:],
                        row["action_offset"],
                    )
                )
                if "initial_poses" in f:
                    self.geom[i] = geometry(
                        f["body_names"].asstr()[:].tolist(), f["initial_poses"][:]
                    )
            self.groups[eid.rsplit("/", 1)[0]].append(i)
            if features:
                self.x[i].copy_(torch.from_numpy(np.load(cache / jobs[eid]["path"])))
        y = torch.tensor(np.stack(labels), device="cuda")
        self.mean = y[: self.nt].mean((0, 1))
        self.std = y[: self.nt].std((0, 1)).clamp_min(0.015)
        self.mean[3:8] = 0
        self.std[3:8] = 1
        self.y = (y - self.mean) / self.std
        self.train = {k: v for k, v in self.groups.items() if v[0] < self.nt}
        self.val = {k: v for k, v in self.groups.items() if v[0] >= self.nt}

    def pairs(self):
        assert self.split["pair_sha"] == sha(DATA / "roles/train_pairs.csv")
        groups = defaultdict(list)
        with (DATA / "roles/train_pairs.csv").open() as f:
            for d, q in csv.reader(f):
                if d not in self.index or q not in self.index:
                    continue
                i, j = self.index[d], self.index[q]
                assert (i < self.nt) == (j < self.nt)
                groups[d.rsplit("/", 1)[0]].append((i, j))
        return groups


def balanced(groups, batch, rng):
    """Equal benchmark probability, then uniform tasks, then uniform episodes/pairs."""
    by = {
        b: [v for k, v in groups.items() if k.startswith(b + "/")]
        for b in ("cf", "bpp")
    }
    result = []
    for i in range(batch):
        tasks = by[("cf", "bpp")[i % 2]]
        samples = tasks[rng.integers(len(tasks))]
        result.append(samples[rng.integers(len(samples))])
    return result
