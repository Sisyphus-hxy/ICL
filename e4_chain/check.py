"""Reload the three best checkpoints, check frozen components and decode a held-out layout."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from e2.common import read, sha, write
from vjepa_probe.vjepa_common import load_encoder
from .archives import OUT
from .cache import bank_file
from .data import scene_geometry, tracker
from .model import Chain, MODULES, osc


@torch.no_grad()
def verify(run, index_root):
    rows = read(index_root / "episodes.json")
    pairs = np.load(index_root / "val_pairs.npy")
    pair = next(p for p in pairs if rows[p[0]]["steps"] == 2)
    d, q = (rows[i] for i in pair)
    assert d["task"] == q["task"] and d["layout"] != q["layout"]
    assert d["procedure"] == q["procedure"] and d["fold"] == q["fold"] == "val"
    source = OUT / "tasks" / d["task"]
    meta = read(source / "index.json")
    donor, query = (
        next(ep for ep in meta["episodes"] if ep["demo"] == r["id"].split("/")[-1])
        for r in (d, q)
    )
    scores = {}
    prev, previous_path = None, None
    for stage in "ABC":
        path = run / stage / "best.pt"
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if prev is not None:
            assert ck["config"]["parent_sha"] == sha(previous_path)
            for k, v in ck["model"].items():
                if k.split(".")[0] not in MODULES[stage]:
                    assert torch.equal(
                        v, prev["model"][k]
                    ), f"Frozen {k} changed in {stage}"
        prev, previous_path = ck, path
        scores[stage] = ck["history"][-1]
    device = "cpu"
    slots = ck["lineage"]["geometry"]
    net = Chain(len(slots)).to(device).eval()
    net.load_state_dict(ck["model"])
    x = torch.from_numpy(np.load(bank_file(donor["demo"], meta["sha256"]))).float()[
        None
    ]
    gd, gq = (
        torch.from_numpy(scene_geometry(slots, e["names"], e["initial"]))[None]
        for e in (donor, query)
    )
    y, z, delta = net.predict(x, gd, gq)
    assert torch.equal(y, net.decoder(z + delta)) and torch.isfinite(y).all()
    plan = (y * ck["std"] + ck["mean"])[0]
    with h5py.File(source / "episodes.h5") as f:
        initial = f[query["demo"] + "/eef_pose"][0]
    action, _ = tracker(plan.numpy(), initial).step(initial)
    assert (
        action.shape == (7,)
        and np.isfinite(action).all()
        and osc(plan).shape == (192, 7)
    )
    jepa = load_encoder("cpu")
    assert not any(p.requires_grad for p in jepa.parameters())
    parameters = {
        k: sum(p.numel() for p in m.parameters()) for k, m in net.named_children()
    }
    summary = dict(
        data=read(index_root / "data_summary.json"),
        parameters=parameters,
        e4_parameters=sum(parameters.values()),
        frozen_jepa=sum(p.numel() for p in jepa.parameters()),
        stage_parameters={
            s: sum(parameters[k] for k in names) for s, names in MODULES.items()
        },
        best_validation=scores,
        reload_pair=[d["id"], q["id"]],
        raw_action_shape=list(osc(plan).shape),
        e0_first_action=action.tolist(),
        frozen_components_equal=True,
    )
    write(run / "checks.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=OUT / "training")
    p.add_argument("--index-dir", type=Path, default=OUT)
    args = p.parse_args()
    verify(args.run, args.index_dir)
