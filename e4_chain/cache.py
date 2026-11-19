"""Cache 64-frame Chain videos, reusing matching E2 patch features before encoding RGB."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from behavior_transfer.dataset import resized_rgb, save_feature
from behavior_transfer.sampling import frame_indices
from e2.cache import CACHE as E2_CACHE, encode, valid
from e2.common import DATA, read, sha, write
from vjepa_probe.vjepa_common import CKPT, load_encoder
from .archives import OUT, RAW, TRAIN, CATALOG, DOUBLE

SHAPE = (32, 4, 768)
WEIGHTS = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
SIGNATURE = f"{WEIGHTS}/agentview:canonical64:PIL384:imagenet:bf16:E2-THWD:2x2-fp16:v1"


def compress(patches):
    return (
        patches.reshape(32, 2, 4, 2, 4, 768)
        .mean((2, 4), dtype=np.float32)
        .reshape(SHAPE)
        .astype(np.float16)
    )


def bank_file(name, checksum):
    return OUT / "features" / checksum / (name + ".npy")


def feature_key(checksum, demo, indices):
    return hashlib.sha256(
        f"{SIGNATURE}/{checksum}/{demo}/{indices.tolist()}".encode()
    ).hexdigest()


def checked(path, key):
    try:
        m = read(path.with_suffix(".json"))
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            a.shape == SHAPE
            and a.dtype == np.float16
            and m["key"] == key
            and sha(path) == m["sha256"]
        ):
            return m["source"]
    except (KeyError, ValueError, OSError, EOFError):
        pass
    return None


def old_inventory():
    path = E2_CACHE / "inventory.json"
    verified_index = E2_CACHE / "verified.json"
    if (
        not path.exists()
        or not verified_index.exists()
        or read(verified_index).get("inventory_sha") != sha(path)
    ):
        return {}, {}
    inventory = read(path)
    expected = dict(
        encoder="official ViT-B/16 V-JEPA 2.1 ema_encoder",
        sha256=WEIGHTS,
        pooling="final 24x24 tokens -> spatial avg 3x3 stride 3; THWD",
        precision="autocast-bf16/output-f16",
        preprocessing="agentview, PIL bilinear 384, ImageNet, canonical 64 frames",
    )
    if inventory["signature"] != expected:
        return {}, {}
    jobs = inventory["jobs"]
    rows = {
        r["id"]: r
        for r in read(DATA / "roles/train_episodes.json")
        if r["id"].startswith("bpp/liberogen_goal_chain_seen/")
    }
    return jobs, rows


def load_input(info, ep, jobs, rows):
    demo = ep["demo"]
    idx = frame_indices(ep["frames"])
    path = bank_file(demo, info["sha256"])
    key = feature_key(info["sha256"], demo, idx)
    source = checked(path, key)
    if source:
        return path, key, source, None
    raw = RAW / info["view"] / (info["task"] + "_demo.hdf5")
    eid = f"bpp/liberogen_goal_chain_seen/{info['task']}/{demo}"
    if info["view"] == DOUBLE and eid in jobs and eid in rows:
        j, r = jobs[eid], rows[eid]
        with h5py.File(DATA / r["path"]) as f:
            link = f.get("raw", getlink=True)
        st = raw.stat()
        source_id = f"{link.filename}:{link.path}:{st.st_size}:{st.st_mtime_ns}"
        if (
            Path(link.filename).is_file()
            and raw.samefile(link.filename)
            and hashlib.sha256(source_id.encode()).hexdigest() == r["visual_sha256"]
            and link.path == "/data/" + demo
            and idx.tolist() == j["indices"]
            and r["image_convention"] == ep["image_convention"]
            and not r.get("rgb_transform")
            and valid(E2_CACHE / j["path"], j)
        ):
            return path, key, "E2", compress(np.load(E2_CACHE / j["path"]))
    # The official RGB arrays are contiguous HDF5 datasets. One sequential
    # read avoids 64 seeks per video on the shared data disk.
    with h5py.File(raw) as f:
        rgb = f[f"data/{demo}/obs/agentview_rgb"][:][idx]
    if ep["image_convention"] == "opengl":
        rgb = rgb[:, ::-1]
    return path, key, "JEPA", resized_rgb(rgb, 384)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--follow", action="store_true")
    args = p.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    x = torch.ones(8, 8, device="cuda", dtype=torch.bfloat16)
    assert (x @ x).mean().item() == 8 and sha(CKPT) == WEIGHTS
    jobs, rows = old_inventory()
    enc = None
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
    done = set()
    all_files = sorted(Path(f) for f in CATALOG["files"] if f.split("/")[0] in TRAIN)
    # Rotate the queue so that workers can claim disjoint ready tasks on startup.
    offset = len(all_files) * rank // int(os.environ.get("WORLD_SIZE", 1))
    all_files = all_files[offset:] + all_files[:offset]
    with torch.inference_mode(), ThreadPoolExecutor(1) as pool:
        while len(done) < len(all_files):
            available = False
            for rel in all_files:
                if rel in done:
                    continue
                name = rel.name.removesuffix("_demo.hdf5")
                folder = OUT / "tasks" / rel.parent / name
                index = folder / "index.json"
                if not index.is_file():
                    continue
                with (folder / "features.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    info = read(index)
                    marker = folder / "features.json"
                    # Verify each artifact once per invocation, also recovering lost/corrupt files.
                    current = read(marker) if marker.is_file() else {}
                    if (
                        current.get("signature") == SIGNATURE
                        and current.get("archive_sha") == info["sha256"]
                        and current.get("run") == run_id
                    ):
                        done.add(rel)
                        continue
                    available = True
                    counts = {"JEPA": 0, "E2": 0}
                    iterator = iter(info["episodes"])

                    def submit():
                        ep = next(iterator, None)
                        return (
                            pool.submit(load_input, info, ep, jobs, rows)
                            if ep
                            else None
                        )

                    future = submit()
                    while future is not None:
                        path, key, source, value = future.result()
                        future = submit()
                        if value is not None:
                            if source == "JEPA":
                                if enc is None:
                                    enc = load_encoder("cuda")
                                    assert not enc.training and not any(
                                        p.requires_grad for p in enc.parameters()
                                    )
                                    print(
                                        "frozen JEPA parameters",
                                        sum(p.numel() for p in enc.parameters()),
                                        flush=True,
                                    )
                                value = compress(encode(enc, value))
                            if not np.isfinite(value).all():
                                raise ValueError(path)
                            save_feature(path, value)
                            write(
                                path.with_suffix(".json"),
                                dict(key=key, source=source, sha256=sha(path)),
                            )
                        counts[source] += 1
                    write(
                        marker,
                        dict(
                            signature=SIGNATURE,
                            counts=counts,
                            archive_sha=info["sha256"],
                            run=run_id,
                        ),
                    )
                    done.add(rel)
                    print(rank, "cached", name, counts, flush=True)
            if not available and len(done) < len(all_files):
                if not args.follow and any(
                    not (
                        OUT
                        / "tasks"
                        / r.parent
                        / r.name.removesuffix("_demo.hdf5")
                        / "index.json"
                    ).exists()
                    for r in all_files
                ):
                    raise FileNotFoundError("Run prepare before cache or use --follow")
                time.sleep(5)


if __name__ == "__main__":
    main()
