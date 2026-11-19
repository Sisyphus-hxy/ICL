"""Index the official shared E1 task split and export episode-level observations."""

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .sampling import frame_indices

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "datasets/behavior_transfer"
CF_RAW = Path(
    "/data2/huangxiangyun/.cache/huggingface/lerobot/physical-intelligence/libero"
)
BPP_DATA = ROOT / "bpp/datasets/libero"
CF_LIB = ROOT / "ContextFlow/third_party/libero/libero/libero"
BP_DEF = ROOT / "bpp/src/behavior_prompting/train_network/env/libero/bddl_files"
OFFICIAL = {
    "cf": ROOT / "contextflow_probe/e1/outputs/shared/cf/official",
    "bpp": ROOT / "bpp/experiments/e1/outputs/shared/bpp/official",
}
DIRS = {
    "liberogen_goal_chain": "libero_goal_chain_selected_view",
    "liberogen_spatial_combination": "libero_spatial_selected_combinations_view",
}
DIRS.update(
    {k + "_seen": v.replace("_view", "_inverse_view") for k, v in list(DIRS.items())}
)


def read_json(path):
    return json.loads(path.read_text())


def digest(array):
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f4").tobytes()
    ).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(f.name, path)


@lru_cache
def procedures(directory):
    return yaml.safe_load((BP_DEF / directory / "task_metadata.yaml").read_text())[
        "tasks"
    ]


def cf_camera(bddl):
    problem = re.search(r"\(problem\s+(\w+)\)", bddl.read_text(), re.I)[1].lower()
    source = CF_LIB / f"envs/problems/{problem}.py"
    for call in ast.walk(ast.parse(source.read_text())):
        if (
            not isinstance(call, ast.Call)
            or getattr(call.func, "attr", "") != "set_camera"
        ):
            continue
        kw = {x.arg: x.value for x in call.keywords}
        if getattr(kw.get("camera_name"), "value", "") == "agentview":
            return dict(
                pos=ast.literal_eval(kw["pos"]),
                quat_wxyz=ast.literal_eval(kw["quat"]),
                fovy_deg=45,
                source=str(source.relative_to(ROOT)),
                provenance="simulator default, no per-episode calibration in parquet",
                image_transform="original LeRobot RGB (upstream 180 degree rotation)",
            )
    raise ValueError(source)


@lru_cache
def local_archives(root):
    """Resolve official inverse-view files also installed under a stock LIBERO name."""
    by_size = defaultdict(list)
    for path in root.glob("*/*.hdf5"):
        by_size[path.stat().st_size].append(path)
    lookup = {}
    for suite in DIRS:
        if suite.endswith("_seen"):
            continue
        catalog = read_json(ROOT / "bpp/logs" / ("data-" + suite + "-manifest.json"))
        for item in catalog["artifacts"]:
            name = Path(item["rfilename"])
            desired = root / name.parent.name / name.name
            if desired.is_file():
                if desired.stat().st_size != item["size"]:
                    raise ValueError(f"Incomplete archive: {desired}")
                continue
            if name.parent.name not in DIRS.values():
                continue
            for other in by_size[item["size"]]:
                with other.open("rb") as f:
                    checksum = hashlib.file_digest(f, "sha256").hexdigest()
                if checksum == item["lfs"]["sha256"]:
                    lookup[desired] = other
                    break
    return lookup


def inventory(cf_raw, bpp_raw):
    groups = defaultdict(list)
    for line in (cf_raw / "meta/episodes.jsonl").read_text().splitlines():
        ep = json.loads(line)
        groups[ep["tasks"][0]].append(ep)
    jobs = []
    for bench, official in OFFICIAL.items():
        manifest = read_json(official / "manifest.json")
        splits = {
            (r["suite"], r["task"]): r
            for r in csv.DictReader((official / "task_split.csv").open())
        }
        if len(splits) != len(manifest["tasks"]):
            raise ValueError("E1 CSV and manifest disagree")
        for key, t in manifest["tasks"].items():
            suite, task = key.split("/")
            assert t["role"] == splits[suite, task]["split"]
            src = ROOT / t["source"]
            e1 = read_json(src / "manifest.json")
            recs = {r["episode"]: r for r in e1["episodes"]}
            test_pairs = []
            if t["role"] == "test":
                test_pairs = read_json(official / "tasks" / key / "splits.json")[
                    "test_pairs"
                ]
            if bench == "cf":
                (bddl,) = (CF_LIB / "bddl_files" / suite).glob("*" + task + ".bddl")
                candidates = groups[t["instruction"]]
                raw = cf_raw
                camera = cf_camera(bddl)
                proc = {}
            else:
                directory = DIRS[suite]
                bddl = BP_DEF / directory / (task + ".bddl")
                raw = bpp_raw / directory / (task + "_demo.hdf5")
                raw = local_archives(bpp_raw).get(raw, raw)
                candidates = []
                proc = procedures(directory)[task]
                camera = {}
            text = bddl.read_text()
            jobs.append(
                dict(
                    benchmark=bench,
                    suite=suite,
                    task=task,
                    split=t["role"],
                    instruction=t["instruction"],
                    bddl=str(bddl.relative_to(ROOT)),
                    bddl_language=re.search(r"\(:language\s+([^)]*)", text, re.I)[
                        1
                    ].strip(),
                    goal=text[text.lower().rindex("(:goal") :],
                    procedure_metadata=proc,
                    raw=str(raw),
                    camera=camera,
                    e1=t,
                    e1_records=recs,
                    candidates=candidates,
                    test_pairs=test_pairs,
                )
            )
    return jobs


def single_pick(goal, names, poses, eef, action):
    """A coarse observed manipulation sequence, requiring one grasp and one lifted goal body."""
    predicates = re.findall(r"\((\w+)\s+([\w]+)\s+([\w]+)\)", goal.lower())
    if len(predicates) != 1 or predicates[0][0] not in ("on", "in"):
        return None
    # Do not infer a procedure for goals with extra unary predicates.
    if len(re.findall(r"\([^()]+\)", goal)) != 1:
        return None
    _, target, dest = predicates[0]
    if target not in names:
        return None
    closed = action[:, 6] > 0.5
    starts = np.flatnonzero(closed & ~np.r_[False, closed[:-1]])
    if len(starts) != 1:
        return None
    p = poses[:, names.index(target), :3]
    lifted = (p[:, 2] - p[0, 2] > 0.025) & closed
    near = np.linalg.norm(p - eef[:, :3], axis=1) < 0.13
    if (lifted & near).sum() < 3:
        return None
    return f"grasp {target} > lift {target} > {predicates[0][0]} {dest}"


def poses_from_states(xml, states):
    """World-level free joints and fixed object roots in recorded MJCF; xyzw convention."""
    world = ET.fromstring(xml).find("worldbody")
    offsets, nq, nv = {}, 0, 0
    for joint in world.iter():
        if joint.tag not in ("joint", "freejoint"):
            continue
        typ = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
        offsets[joint.get("name")] = nq + 1
        nq += {"free": 7, "ball": 4}.get(typ, 1)
        nv += {"free": 6, "ball": 3}.get(typ, 1)
    if states.shape[1] != 1 + nq + nv:
        raise ValueError(
            f"Unknown flattened state schema {states.shape[1]}, {nq}, {nv}"
        )
    poses = {}
    for body in world.findall("body"):
        name = body.get("name", "")
        if not name.endswith("_main"):
            continue
        joint = body.find("joint")
        if joint is None:
            if any(k in body.attrib for k in ("euler", "axisangle", "xyaxes", "zaxis")):
                raise ValueError("Unsupported orientation " + name)
            row = np.fromstring(
                body.get("pos", "0 0 0") + " " + body.get("quat", "1 0 0 0"), sep=" "
            )
            arr = np.broadcast_to(row, (len(states), 7))
        elif joint.get("type") == "free":
            start = offsets[joint.get("name")]
            arr = states[:, start : start + 7]
        else:
            raise ValueError("Unsupported root joint " + name)
        poses[name[:-5]] = arr[:, [0, 1, 2, 4, 5, 6, 3]]
    names = sorted(poses)
    return names, np.stack([poses[n] for n in names], axis=1)


def common(row, f, state, actions, timestamps, offset, geom, names, initial):
    n = len(actions)
    if state.shape != (n, 8) or actions.shape != (n, 7) or len(timestamps) != n:
        raise ValueError(row["id"])
    if (
        not all(np.isfinite(a).all() for a in (state, actions, timestamps))
        or not (np.diff(timestamps) > 0).all()
    ):
        raise ValueError("non-finite/non-monotonic trajectory " + row["id"])
    idx = frame_indices(n)
    for k, v in dict(
        proprio=state,
        actions=actions,
        timestamps=timestamps,
        eef_pose=np.c_[state[:, :3], Rotation.from_rotvec(state[:, 3:6]).as_quat()],
        gripper=state[:, 6:],
        frame_indices64=idx,
        timestamps64=timestamps[idx],
    ).items():
        f[k] = v
    if initial is not None:
        f["initial_poses"] = initial
        f["body_names"] = np.asarray(names, dtype=h5py.string_dtype())
    if geom is not None:
        f["object_poses"] = geom
        f["object_pose_valid"] = np.isfinite(geom).all(axis=(1, 2))
    row.update(
        frames=n,
        action_offset=offset,
        action_sha256=digest(actions),
        names=names,
        initial=None if initial is None else initial.tolist(),
    )


def store_images(f, name, column):
    # Preserve encoded pixels. An index lookup reads only one frame's bytes.
    images = [x["bytes"] for x in column.to_pylist()]
    edges = np.cumsum([0] + list(map(len, images)))
    f[name + "_bytes"] = np.frombuffer(b"".join(images), dtype=np.uint8)
    f[name + "_edges"] = edges
    return hashlib.sha256(f[name + "_bytes"][:].tobytes()).hexdigest()


def cf_episode(j, row, f, candidate, record):
    import pyarrow.parquet as pq

    ep = candidate["episode_index"]
    path = Path(j["raw"]) / f"data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet"
    table = pq.read_table(path)
    assert set(table["episode_index"].to_pylist()) == {ep}
    n = len(table)
    assert n == candidate["length"] and table["frame_index"].to_pylist() == list(
        range(n)
    )
    state, actions = (
        np.asarray(table[k].to_pylist(), dtype=np.float32) for k in ("state", "actions")
    )
    row.update(
        source=str(path),
        camera=j["camera"],
        control_hz=20,
        timestamp_source="LeRobot 10 fps",
        success_source="published OpenVLA successful-demo subset",
        rgb=True,
    )
    row["visual_sha256"] = store_images(f, "agentview", table["image"])
    store_images(f, "wrist", table["wrist_image"])
    names, initial, geom = [], None, None
    if record:
        rp = ROOT / j["e1"]["source"] / (record["name"] + ".npz")
        with np.load(rp) as r:
            a = r["actions"]
            assert bool(r["success"]) and np.array_equal(a, actions[: len(a)])
            names = r["body_names"].tolist()
            initial = r["init_obj_poses"]
            geom = np.full((n, len(names), 7), np.nan)
            geom[0] = initial
            steps = min(n - 1, len(a))
            geom[1 : steps + 1] = r["obj_poses"][:steps]
            row["geometry_source"] = "E1 matched layout and successful action replay"
            row["replay_path"] = str(rp.relative_to(ROOT))
            row["replay_action_sha256"] = digest(a)
            row["replay_xyz_p95_m"] = float(
                np.percentile(
                    np.linalg.norm(
                        state[1 : steps + 1, :3] - r["eef_poses"][:steps, :3], axis=1
                    ),
                    95,
                )
            )
            row["init_image_mse"] = record.get("init_image_mse")
            # Require agreement with the recorded motion as well as observed grasp/lift.
            if row["replay_xyz_p95_m"] <= 0.025:
                row["procedure"] = single_pick(j["goal"], names, geom, state, actions)
                if row["procedure"]:
                    row["procedure_evidence"] = (
                        "single goal; single closure; target lift near EEF in matched replay"
                    )
    common(
        row, f, state, actions, np.asarray(table["timestamp"]), 0, geom, names, initial
    )


def bpp_episode(j, row, f, group, record):
    if group is None:
        p = ROOT / j["e1"]["source"] / (record["name"] + ".npz")
        s = read_json(p.parents[1] / "source.json")
        with np.load(p) as r:
            assert bool(r["success"])
            state = np.c_[
                r["eef_poses"][:, :3],
                Rotation.from_quat(r["eef_poses"][:, 3:]).as_rotvec(),
                r["gripper"],
            ]
            actions = r["actions"]
            names, initial = r["body_names"].tolist(), r["init_obj_poses"]
        row.update(
            rgb=False,
            source=str(p.relative_to(ROOT)),
            original_url=s["url"],
            control_hz=20,
            timestamp_source="derived nominal seconds at 20 Hz",
            geometry_source="E1 initial geometry",
            camera=None,
        )
        common(
            row,
            f,
            state,
            actions,
            (np.arange(len(actions)) + 1) / 20,
            1,
            None,
            names,
            initial,
        )
    else:
        d = group
        xml = d.attrs["model_file"]
        xmlroot = ET.fromstring(xml)
        env = json.loads(d.parent.attrs["env_args"])["env_kwargs"]
        assert (
            env["controller_configs"]["type"] == "OSC_POSE"
            and env["controller_configs"]["control_delta"]
        )
        hz = env["control_freq"]
        states = d["states"][:]
        names, poses = poses_from_states(xml, states)
        joints = d["obs/joint_states"][:]
        # Determine producer ordering by matching independent joint telemetry.
        errs = [
            np.max(np.abs(joints[1:-1] - states[1 + lag : len(states) - 1 + lag, 1:8]))
            for lag in (0, 1)
        ]
        lag = int(np.argmin(errs))
        if errs[lag] > 0.002 or errs[1 - lag] <= errs[lag] + 1e-6:
            raise ValueError(f"ambiguous observation timing {errs} at {d.name}")
        actions = d["actions"][:]
        state = np.c_[
            d["obs/ee_pos"][:], d["obs/ee_ori"][:], d["obs/gripper_states"][:]
        ]
        convention = d.parent.attrs["macros_image_convention"]
        assert convention in ("opencv", "opengl")
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            assert d["obs/" + key].shape == (
                len(actions),
                env["camera_heights"],
                env["camera_widths"],
                3,
            )
        raw_link = j.get("raw_link") or str(Path(j["raw"]).resolve())
        f["raw"] = h5py.ExternalLink(raw_link, d.name)
        initial = poses[0].copy()
        geom = (
            poses
            if not lag
            else np.concatenate([poses[1:], np.full_like(poses[:1], np.nan)])
        )
        camera = {c.get("name"): c.attrib for c in xmlroot.iter("camera")}
        row.update(
            rgb=True,
            source=j["raw"],
            source_group=d.name,
            control_hz=hz,
            geometry_source="named world roots in recorded MJCF and flattened states",
            timestamp_source="sim time at observation",
            camera=camera,
            image_convention=convention,
        )
        # Path, group and file fingerprint distinguish external image sequences.
        st = Path(j["raw"]).stat()
        row["visual_sha256"] = hashlib.sha256(
            f'{j["raw"]}:{d.name}:{st.st_size}:{st.st_mtime_ns}'.encode()
        ).hexdigest()
        if j["procedure_metadata"].get("is_existing_task"):
            row["procedure"] = single_pick(j["goal"], names, geom, state, actions)
            if row["procedure"]:
                row["procedure_evidence"] = (
                    "single goal; single closure; recorded target lift near EEF"
                )
        common(
            row, f, state, actions, states[:, 0] + lag / hz, lag, geom, names, initial
        )
    meta = j["procedure_metadata"]
    if meta.get("is_existing_task") is False and meta.get("execution_steps"):
        row["procedure"] = " > ".join(meta["execution_steps"])
        row["procedure_evidence"] = (
            "LIBERO-Gen generator execution_steps + successful source"
        )
    row["success_source"] = (
        "original terminal done and positive reward"
        if group is not None
        else "E1 verified original terminal done and positive reward"
    )


def extract(args):
    j, out = args
    base = "/".join(j[k] for k in ("benchmark", "suite", "task"))
    dest = out / "episodes" / base
    dest.mkdir(parents=True, exist_ok=True)
    bpp = j["benchmark"] == "bpp"
    raw = Path(j["raw"])
    source = h5py.File(raw, "r") if bpp and raw.exists() else None
    recs = j["e1_records"]
    if source:
        ids = list(source["data"])
    elif bpp:
        ids = list(recs)
    else:
        ids = [str(c["episode_index"]) for c in j["candidates"]]
    candidates = {str(c["episode_index"]): c for c in j["candidates"]}
    rows = []
    try:
        for eid in ids:
            record = recs.get(eid)
            key = record["name"] if record else ""
            if j["split"] == "test" and key not in j["e1"]["test"]:
                continue
            g = source["data/" + eid] if source else None
            if g is not None and not (g["dones"][-1] and g["rewards"][-1] > 0):
                continue
            rid = base + "/" + eid
            row = dict(
                id=rid,
                benchmark=j["benchmark"],
                suite=j["suite"],
                task=j["task"],
                split=j["split"],
                procedure=None,
                procedure_evidence=None,
                geometry_source=None,
                e1_name=key,
                e1_excluded=j["e1"]["excluded"].get(key),
                path="episodes/" + rid + ".h5",
            )
            target = dest / (eid + ".h5")
            with tempfile.NamedTemporaryFile(dir=dest, delete=False) as tmp:
                with h5py.File(tmp.name, "w") as f:
                    if bpp:
                        bpp_episode(j, row, f, g, record)
                    else:
                        cf_episode(j, row, f, candidates[eid], record)
                    f.attrs["episode_id"] = rid
                    f.attrs["action_offset"] = row["action_offset"]
                os.replace(tmp.name, target)
            rows.append(row)
    finally:
        if source:
            source.close()
    dump(dest / "index.json", rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cf-raw", type=Path, default=CF_RAW)
    p.add_argument("--bpp-raw", type=Path, default=BPP_DATA)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--manifest-only", action="store_true")
    args = p.parse_args()
    jobs = inventory(args.cf_raw.resolve(), args.bpp_raw.resolve())
    if not args.manifest_only:
        with ProcessPoolExecutor(args.workers) as pool:
            total = sum(pool.map(extract, ((j, args.out) for j in jobs)))
        print("Indexed episodes", total, flush=True)
    from .stock import augment

    with ProcessPoolExecutor(args.workers) as pool:
        list(
            pool.map(
                augment,
                (
                    (j, args.out, args.bpp_raw.resolve())
                    for j in jobs
                    if j["benchmark"] == "cf"
                ),
            )
        )
    from .manifest import assemble

    assemble(jobs, args.out)


if __name__ == "__main__":
    main()
