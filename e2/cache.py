"""Offline frozen JEPA patches. One CUDA process per explicitly selected UUID."""

import argparse
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from behavior_transfer.dataset import (
    EpisodeClips,
    resized_rgb,
    save_feature,
    vjepa_input,
)
from behavior_transfer.sampling import frame_indices
from vjepa_probe.vjepa_common import CKPT, load_encoder
from .common import B1, DATA, OFFICIAL, OUT, ROOT, read, sha, write

CACHE = OUT / "cache"
SHAPE = (32, 8, 8, 768)


def inventory():
    signature = dict(
        encoder="official ViT-B/16 V-JEPA 2.1 ema_encoder",
        sha256=sha(CKPT),
        pooling="final 24x24 tokens -> spatial avg 3x3 stride 3; THWD",
        precision="autocast-bf16/output-f16",
        preprocessing="agentview, PIL bilinear 384, ImageNet, canonical 64 frames",
    )
    jobs = {}
    for split in ("train", "test"):
        clips = EpisodeClips(DATA, split)
        for eid, row in clips.rows.items():
            idx = frame_indices(row["frames"])
            path = clips.cache_path(CACHE / "features", eid, idx, signature)
            jobs[eid] = dict(
                split=split, path=str(path.relative_to(CACHE)), indices=idx.tolist()
            )
    for key in (
        "procedural_constraints/obstacle_clearance",
        "procedural_demos/obstacle",
    ):
        src = B1 / "outputs" / key
        donors = read(OFFICIAL["bpp"] / "procedure_tests" / key / "splits.json")[
            "donors"
        ]
        for row in read(src / "manifest.json")["episodes"]:
            if row["name"] not in donors:
                continue
            eid = "external/" + key + "/" + row["name"]
            source = Path(row["source"])
            with h5py.File(source) as f:
                idx = frame_indices(len(f["data/demo_0/obs/agentview_rgb"]))
            token = sha(source) + signature["sha256"][:12]
            jobs[eid] = dict(
                split="external",
                path=str(Path("features") / eid / (token + ".npy")),
                indices=idx.tolist(),
                source=str(source),
            )
    write(CACHE / "inventory.json", dict(signature=signature, jobs=jobs))
    return jobs


def valid(path, job):
    marker = path.with_suffix(".json")
    if not (path.exists() and marker.exists()):
        return False
    try:
        m = read(marker)
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            a.shape == SHAPE
            and a.dtype == np.float16
            and m["indices"] == job["indices"]
            and m["sha256"] == sha(path)
            and np.isfinite(a).all()
        )
    except (EOFError, OSError, ValueError, KeyError):
        return False


def encode(enc, rgb):
    x = vjepa_input(rgb[None].cuda())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = enc(x)
    assert z.shape == (1, 32 * 24 * 24, 768)
    z = z.float().reshape(32, 24, 24, 768).permute(0, 3, 1, 2)
    return F.avg_pool2d(z, 3, 3).permute(0, 2, 3, 1).half().contiguous().cpu().numpy()


def worker(rank, world):
    torch.set_num_threads(1)
    assert (
        torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")
    ).mean().item() == 8
    clips = {s: EpisodeClips(DATA, s) for s in ("train", "test")}
    jobs = list(read(CACHE / "inventory.json")["jobs"].items())[rank::world]

    def frames(eid, j):
        idx = np.array(j["indices"])
        if j["split"] == "external":
            with h5py.File(j["source"]) as f:
                g = f["data/demo_0/obs/agentview_rgb"]
                rgb = resized_rgb(np.stack([g[i][::-1] for i in idx]), 384)
        else:
            c = clips[j["split"]].read(eid)
            assert np.array_equal(c["indices"], idx) and np.all(
                np.diff(c["timestamps"]) >= 0
            )
            rgb = c["rgb"]
        return rgb

    def pending():
        for eid, j in jobs:
            path = CACHE / j["path"]
            if not valid(path, j):
                yield eid, j, path

    with ThreadPoolExecutor(1) as pool, torch.inference_mode():
        it = iter(pending())
        item = next(it, None)
        if item is None:
            print("verified unchanged shard", rank, len(jobs), flush=True)
            return
        enc = load_encoder("cuda")
        assert not enc.training and not any(p.requires_grad for p in enc.parameters())
        future = pool.submit(frames, *item[:2])
        count = 0
        while item:
            eid, j, path = item
            rgb = future.result()
            item = next(it, None)
            future = pool.submit(frames, *item[:2]) if item else None
            a = encode(enc, rgb)
            assert np.isfinite(a).all() and a.std(dtype=np.float32) > 0
            if count == 0:
                # Check reproducibility and sensitivity to time order before processing shard.
                again = encode(enc, rgb)
                reverse = encode(enc, rgb.flip(0))
                assert np.allclose(a, again, rtol=0.005, atol=0.01)
                diff = float(np.mean(np.abs(a.astype("f") - reverse)))
                assert diff > 1e-5
                write(
                    CACHE / f"preflight_{rank}.json",
                    dict(
                        episode=eid,
                        device=torch.cuda.get_device_name(),
                        uuid=os.environ["CUDA_VISIBLE_DEVICES"],
                        shape=list(a.shape),
                        reversed_mae=diff,
                        repeat_max=float(np.abs(again - a).max()),
                        rgb_range=[int(rgb.min()), int(rgb.max())],
                    ),
                )
            save_feature(path, a)
            write(
                path.with_suffix(".json"), dict(indices=j["indices"], sha256=sha(path))
            )
            count += 1
            if count % 40 == 0:
                print(rank, count, eid, flush=True)
    print("finished", rank, len(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+")
    p.add_argument("--worker", nargs=2, type=int)
    args = p.parse_args()
    if args.worker:
        return worker(*args.worker)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        p.error("supply distinct GPU UUIDs")
    jobs = inventory()
    procs = []
    try:
        for i, uid in enumerate(args.gpus):
            with (CACHE / f"worker_{i}.log").open("a") as log:
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "e2.cache",
                            "--worker",
                            str(i),
                            str(len(args.gpus)),
                        ],
                        env=dict(
                            os.environ,
                            CUDA_VISIBLE_DEVICES=uid,
                            OPENBLAS_NUM_THREADS="1",
                            OMP_NUM_THREADS="1",
                        ),
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
        if any(p.wait() != 0 for p in procs):
            raise RuntimeError("Cache worker failed; inspect worker logs")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    for eid, j in jobs.items():
        if not valid(CACHE / j["path"], j):
            raise ValueError(eid)
    write(
        CACHE / "verified.json",
        dict(
            inventory_sha=sha(CACHE / "inventory.json"),
            episodes=len(jobs),
            shape=SHAPE,
            counts={
                s: sum(j["split"] == s for j in jobs.values())
                for s in ("train", "test", "external")
            },
        ),
    )


if __name__ == "__main__":
    main()
