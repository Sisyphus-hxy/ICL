"""CPU-only RGB clips and causal query observations for offline ICL."""

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from .sampling import frame_indices

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "datasets/behavior_transfer"


def read_rgb(f, row, indices, camera):
    if camera not in ("agentview", "wrist"):
        raise ValueError(camera)
    raw = None
    if row["benchmark"] == "bpp" or row.get("storage") == "native_hdf5":
        raw = f[
            "raw/obs/"
            + {"agentview": "agentview_rgb", "wrist": "eye_in_hand_rgb"}[camera]
        ]
    else:
        edges = f[camera + "_edges"]
        blob = f[camera + "_bytes"]
    frames = []
    for i in indices:
        if raw is None:
            start, end = edges[i : i + 2]
            with Image.open(io.BytesIO(blob[start:end].tobytes())) as im:
                frame = np.asarray(im.convert("RGB"))
        else:
            frame = raw[i]
            if row.get("rgb_transform") == "rotate180":
                frame = frame[::-1, ::-1]
            elif row["image_convention"] == "opengl":
                frame = frame[::-1]
        frames.append(frame)
    return np.stack(frames)


def resized_rgb(frames, size):
    if frames.shape[1:3] != (size, size):
        frames = np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).resize(
                        (size, size), Image.Resampling.BILINEAR
                    )
                )
                for frame in frames
            ]
        )
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous()


class EpisodeClips(Dataset):
    """Enumerate every available, split-eligible RGB episode once for feature extraction."""

    def __init__(
        self,
        root=DEFAULT_OUT,
        split="train",
        camera="agentview",
        size=384,
        prefer_lerobot=False,
    ):
        if (
            split not in ("train", "test")
            or camera not in ("agentview", "wrist")
            or size < 1
        ):
            raise ValueError((split, camera, size))
        self.root = Path(root)
        self.split, self.camera, self.size = split, camera, size
        role_index = "train_episodes.json" if split == "train" else "test_donors.json"
        self.rows = {
            r["id"]: r
            for r in json.loads((self.root / "roles" / role_index).read_text())
        }
        if prefer_lerobot:
            for key, r in self.rows.items():
                if "lerobot_view" in r:
                    self.rows[key] = dict(r, **r["lerobot_view"], storage="parquet_rgb")
        self.ids = list(self.rows)

    def __len__(self):
        return len(self.ids)

    def read(self, eid, phase=0.5):
        row = self.rows[eid]
        idx = frame_indices(row["frames"], phase)
        with h5py.File(self.root / row["path"], "r") as f:
            frames = read_rgb(f, row, idx, self.camera)
            times = f["timestamps"][:][idx]  # demonstration only
        return dict(
            rgb=resized_rgb(frames, self.size),
            indices=torch.from_numpy(idx),
            timestamps=torch.from_numpy(times.copy()),
        )

    def __getitem__(self, index):
        return dict(episode=self.ids[index], **self.read(self.ids[index]))

    def cache_path(self, directory, eid, indices, encoder_signature):
        """Include checkpoint SHA, layer, normalization and pooling in encoder_signature."""
        if not encoder_signature:
            raise ValueError("encoder_signature is required")
        row = self.rows[eid]
        st = (self.root / row["path"]).stat()
        key = [
            eid,
            row["visual_sha256"],
            st.st_mtime_ns,
            st.st_size,
            self.camera,
            self.size,
            "PIL-BILINEAR-uint8-v1",
            np.asarray(indices).tolist(),
            encoder_signature,
        ]
        hash_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return Path(directory) / self.split / eid / (hash_id + ".npy")


def save_feature(path, features):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        np.save(f, np.asarray(features), allow_pickle=False)
    os.replace(f.name, path)


class BehaviorPairs(Dataset):
    """Training defaults to positives. Test enumerates the complete E1 donor/query list."""

    def __init__(
        self,
        root=DEFAULT_OUT,
        split="train",
        camera="agentview",
        size=384,
        seed=7,
        initial_geometry=False,
        oracle_supervision=False,
        feature_cache=None,
        encoder_signature=None,
        sampling_phases=None,
    ):
        self.clips = EpisodeClips(root, split, camera, size)
        if split == "test" and oracle_supervision:
            raise ValueError("oracle_supervision requires train split")
        self.training = split == "train"
        self.initial_geometry, self.oracle_supervision = (
            initial_geometry,
            oracle_supervision,
        )
        self.feature_cache, self.encoder_signature = feature_cache, encoder_signature
        if feature_cache is not None and not encoder_signature:
            raise ValueError("cached features require encoder_signature")
        # A finite clip bank is reusable across pairs and epochs during cached training.
        if feature_cache is not None and sampling_phases is None:
            sampling_phases = (0.5,)
        self.sampling_phases = (
            None if sampling_phases is None else tuple(sampling_phases)
        )
        if self.sampling_phases is not None and (
            not self.sampling_phases
            or any(not 0 <= p < 1 for p in self.sampling_phases)
        ):
            raise ValueError("sampling_phases must be a nonempty subset of [0,1)")
        self.seed = seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        roles = self.clips.root / "roles"
        self.queries = (
            self.clips.rows
            if self.training
            else {
                r["id"]: r
                for r in json.loads((roles / "test_queries.json").read_text())
            }
        )
        with (roles / (split + "_pairs.csv")).open() as f:
            self.pairs = [tuple(r) for r in csv.reader(f)]
        if any(
            d not in self.clips.rows or q not in self.queries for d, q in self.pairs
        ):
            raise ValueError("Split/inventory mismatch")

    def __len__(self):
        return len(self.pairs)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def at(self, index, t, phase=0.5):
        """Read exactly one query frame. t is the index of the current recorded observation."""
        d, q = self.pairs[index]
        qr = self.queries[q]
        if t < 0 or (self.training and t >= qr["frames"] - qr["action_offset"]):
            raise IndexError(t)
        idx = frame_indices(self.clips.rows[d]["frames"], phase)
        sample = {}
        if self.feature_cache is None:
            clip = self.clips.read(d, phase)
            sample["demo_rgb"] = clip["rgb"]
            sample["demo_timestamps"] = clip["timestamps"]
        else:
            path = self.clips.cache_path(
                self.feature_cache, d, idx, self.encoder_signature
            )
            sample["demo_features"] = torch.from_numpy(
                np.load(path, allow_pickle=False)
            )
            with h5py.File(self.clips.root / self.clips.rows[d]["path"], "r") as f:
                sample["demo_timestamps"] = torch.from_numpy(f["timestamps"][:][idx])
        with h5py.File(self.clips.root / qr["path"], "r") as f:
            sample["query_rgb"] = resized_rgb(
                read_rgb(f, qr, [t], self.clips.camera), self.clips.size
            )[0]
            sample["query_proprio"] = torch.tensor(f["proprio"][t], dtype=torch.float32)
            if self.initial_geometry:
                if "initial_poses" not in f:
                    raise ValueError("Missing agreed initial geometry")
                sample["initial_geometry"] = self._geometry(f, f["initial_poses"][:])
            if self.training:
                labels = dict(
                    action=torch.tensor(
                        f["actions"][t + qr["action_offset"]], dtype=torch.float32
                    )
                )
                if self.oracle_supervision:
                    arr = f["object_poses"][t]
                    labels["oracle_geometry"] = self._geometry(f, arr)
                    if "replay" in (qr["geometry_source"] or ""):
                        # E1 saved a matched replay. It is not synchronized source ground truth.
                        labels["oracle_geometry"]["valid"].zero_()
                        labels["oracle_geometry"]["poses"].zero_()
                return dict(inputs=sample, targets=labels)
        return dict(inputs=sample)

    @staticmethod
    def _geometry(f, arr):
        valid = np.isfinite(arr).all(axis=1)
        return dict(
            names=f["body_names"].asstr()[:].tolist(),
            poses=torch.tensor(np.nan_to_num(arr), dtype=torch.float32),
            valid=torch.tensor(valid),
        )

    def __getitem__(self, i):
        if not self.training:
            return self.at(i, 0)
        rng = np.random.default_rng([self.seed, self.epoch.item(), i])
        qr = self.clips.rows[self.pairs[i][1]]
        t = int(rng.integers(qr["frames"] - qr["action_offset"]))
        phase = (
            rng.random()
            if self.sampling_phases is None
            else rng.choice(self.sampling_phases)
        )
        return self.at(i, t, phase)


def collate(items):
    first = items[0]
    if isinstance(first, dict):
        if set(first) == {"names", "poses", "valid"}:
            return dict(
                names=[r["names"] for r in items],
                poses=torch.nn.utils.rnn.pad_sequence(
                    [r["poses"] for r in items], batch_first=True
                ),
                valid=torch.nn.utils.rnn.pad_sequence(
                    [r["valid"] for r in items], batch_first=True
                ),
            )
        return {k: collate([r[k] for r in items]) for k in first}
    return default_collate(items)


def dataloader(dataset, batch_size=2, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, **kwargs)


def vjepa_input(batch_rgb):
    """(B,64,3,384,384) uint8 to ImageNet-normalized (B,3,64,384,384)."""
    if (
        batch_rgb.ndim != 5
        or tuple(batch_rgb.shape[1:]) != (64, 3, 384, 384)
        or batch_rgb.dtype != torch.uint8
    ):
        raise ValueError("Expected B x 64 x 3 x 384 x 384 uint8 RGB")
    clip = batch_rgb.transpose(1, 2).float().div_(255)
    mean = clip.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
    std = clip.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
    return clip.sub_(mean).div_(std)
