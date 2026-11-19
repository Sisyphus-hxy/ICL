"""Join locally available stock LIBERO recordings by their full OpenVLA action sequence."""

import copy
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .build import dump, read_json, digest, bpp_episode


def no_op_indices(actions):
    # OpenVLA regenerate_libero_dataset.is_noop uses the *previous kept* command.
    previous = None
    keep = []
    for t, a in enumerate(actions):
        moving = np.linalg.norm(a[:-1]) >= 1e-4
        if moving or (previous is not None and a[-1] != previous):
            keep.append(t)
            previous = a[-1]
    return np.array(keep, dtype=int)


def augment(args):
    j, out, bpp_root = args
    bddl = Path(j["bddl"])
    path = bpp_root / j["suite"] / (bddl.stem + "_demo.hdf5")
    base = "/".join((j["benchmark"], j["suite"], j["task"]))
    dest = out / "episodes" / base
    original = read_json(dest / "index.json")
    rows = {r["id"]: r for r in original}
    if not path.exists():
        dump(dest / "views.json", original)
        return
    full = {r["action_sha256"]: r for r in original}
    visited = set()
    # For held-out tasks also recognize non-E1 source episodes so they cannot be added to train.
    with h5py.File(path, "r") as src:
        for eid, g in src["data"].items():
            if not (g["dones"][-1] and g["rewards"][-1] > 0):
                continue
            actions = g["actions"][:]
            keep = no_op_indices(actions)
            seq = digest(actions[keep])
            match = full.get(seq) or full.get(digest(actions))
            if match is None and j["split"] == "test":
                continue
            if seq in visited:
                raise ValueError(f"Duplicate stock sequence {path} {eid}")
            visited.add(seq)
            rid = match["id"] if match else base + "/stock_" + eid
            row = (
                copy.deepcopy(match)
                if match
                else dict(
                    id=rid,
                    benchmark="cf",
                    suite=j["suite"],
                    task=j["task"],
                    split=j["split"],
                    e1_name="",
                    e1_excluded=None,
                )
            )
            row.update(procedure=None, procedure_evidence=None)
            alt = dest / "stock" / (eid + ".h5")
            alt.parent.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=alt.parent, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with h5py.File(tmp_path, "w") as f:
                    bpp_episode(
                        dict(
                            j,
                            raw=str(path),
                            procedure_metadata={"is_existing_task": True},
                        ),
                        row,
                        f,
                        g,
                        None,
                    )
                    f["kept_action_indices"] = keep
                    f.attrs["episode_id"] = rid
                    f.attrs["action_offset"] = row["action_offset"]
                os.replace(tmp_path, alt)
            finally:
                tmp_path.unlink(missing_ok=True)
            row.update(
                path=str(alt.relative_to(out)),
                storage="native_hdf5",
                rgb_transform="rotate180",
                stock_filtered_action_sha256=seq,
            )
            if match:
                # Both views reference the same underlying human demonstration. The full native
                # observation stream supplies measured geometry. Keep the complete 256px stream.
                alt_meta = {
                    k: match[k]
                    for k in (
                        "path",
                        "frames",
                        "source",
                        "camera",
                        "action_sha256",
                        "procedure",
                        "geometry_source",
                        "initial",
                        "names",
                        "visual_sha256",
                        "action_offset",
                    )
                }
                row["lerobot_view"] = alt_meta
                if j["split"] == "test":
                    match.update(
                        stock_view={
                            k: row[k]
                            for k in (
                                "path",
                                "frames",
                                "source",
                                "source_group",
                                "action_sha256",
                                "initial",
                                "names",
                                "action_offset",
                            )
                        },
                        stock_filtered_action_sha256=seq,
                    )
                    continue
            rows[rid] = row
    dump(dest / "views.json", list(rows.values()))
