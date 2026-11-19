"""Apply shared holdouts before enumerating ordered, within-task pairs."""

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

from .procedure import two_placements, fixture_procedure
from .build import ROOT, OFFICIAL, dump, read_json

XYZ_TOL = 1e-4
ANGLE_TOL = 0.001


def language_id(text):
    text = re.sub(r"_\d+$", "", text.lower()).replace("_", " ")
    return (
        " ".join(re.findall("[a-z0-9]+", text))
        .replace("pick the ", "pick up the ")
        .replace("akita black bowl", "black bowl")
        .replace("cookies box", "cookie box")
    )


def geom_key(names, poses):
    order = np.argsort(names)
    return tuple(np.array(names)[order]), np.asarray(poses)[order]


def distances(a, b):
    xyz = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    q1, q2 = a[..., 3:], b[..., 3:]
    dot = np.sum(q1 * q2, axis=-1) / (
        np.linalg.norm(q1, axis=-1) * np.linalg.norm(q2, axis=-1)
    )
    angles = 2 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    return xyz.max(axis=-1), angles.max(axis=-1)


def task_aliases(job):
    return {language_id(job[k]) for k in ("task", "bddl_language", "instruction")}


def percentile(values):
    if not values:
        return None
    return dict(
        zip(
            ("min", "p25", "p50", "p75", "p95", "max"),
            map(float, np.percentile(values, [0, 25, 50, 75, 95, 100])),
        )
    )


def simple_task(task):
    description = re.sub(r"_\d+$", "", task).replace("_and_place_it_", "_")
    return (
        description.startswith(("put_the_", "pick_up_"))
        and "_and_" not in description
        and "drawer" not in description
        and "microwave" not in description
    )


def assemble(jobs, out):
    groups = {}
    held_names, held_hashes = set(), set()
    held_geometry = defaultdict(list)
    task_info = []
    for j in jobs:
        key = "/".join(j[k] for k in ("benchmark", "suite", "task"))
        folder = out / "episodes" / key
        rows = read_json(
            folder / ("views.json" if j["benchmark"] == "cf" else "index.json")
        )
        for r in rows:
            if (
                not r["procedure"]
                and j["benchmark"] == "cf"
                and j["suite"] == "libero_10"
            ):
                r["procedure"] = two_placements(r, j["goal"], out / r["path"])
                if r["procedure"]:
                    r["procedure_evidence"] = (
                        "two goal objects; two observed closures and uniquely attributed lifts in execution order"
                    )
            if r["procedure_evidence"] and "single" in r["procedure_evidence"]:
                if not simple_task(j["task"]):
                    r["procedure"] = r["procedure_evidence"] = None
            if not r["procedure"] and r["rgb"]:
                r["procedure"] = fixture_procedure(r, out / r["path"])
                if r["procedure"]:
                    r["procedure_evidence"] = (
                        "recorded fixture joint crosses LIBERO predicate threshold; observed lift order for open/place"
                    )
        groups[key] = rows
        task_info.append(
            {
                k: j[k]
                for k in (
                    "benchmark",
                    "suite",
                    "task",
                    "split",
                    "bddl",
                    "instruction",
                    "bddl_language",
                    "goal",
                    "procedure_metadata",
                    "camera",
                    "test_pairs",
                )
            }
        )
        if j["split"] == "test":
            held_names.update(task_aliases(j))
            assert {r["e1_name"] for r in rows} == set(j["e1"]["test"])
            for r in rows:
                held_hashes.add(r["action_sha256"])
                if "stock_view" in r:
                    stock = r["stock_view"]
                    held_hashes.update(
                        [stock["action_sha256"], r["stock_filtered_action_sha256"]]
                    )
                    ns, p = geom_key(stock["names"], stock["initial"])
                    held_geometry[ns].append(p)
                if r.get("replay_action_sha256"):
                    held_hashes.add(r["replay_action_sha256"])
                ns, p = geom_key(r["names"], r["initial"])
                held_geometry[ns].append(p)
                with np.load(ROOT / j["e1"]["source"] / (r["e1_name"] + ".npz")) as e:
                    ns, p = geom_key(e["body_names"], e["init_obj_poses"])
                    held_geometry[ns].append(p)
    held_geometry = {k: np.stack(v) for k, v in held_geometry.items()}
    all_rows, by_id, layouts = [], {}, {}
    for j, rows in zip(jobs, groups.values()):
        # A replayed held-out source stage can also appear inside a generated composite.
        meta = j["procedure_metadata"]
        parents = [meta.get("actions_from", "")]
        parents.extend(
            x.get("task", "") for x in (meta.get("grasps_from") or {}).values()
        )
        for r in rows:
            veto = []
            if r["initial"] is not None:
                layouts[r["id"]] = geom_key(r["names"], r["initial"])
            if j["split"] == "train":
                if r["e1_excluded"]:
                    veto.append("E1 exclusion")
                if task_aliases(j) & held_names:
                    veto.append("shared held-out task")
                if any(language_id(x) in held_names for x in parents if x):
                    veto.append("generated from held-out task reference")
                if {
                    r["action_sha256"],
                    r.get("replay_action_sha256"),
                    r.get("stock_filtered_action_sha256"),
                } & held_hashes:
                    veto.append("held-out actions")
                available_geometry = [layouts[r["id"]]] if r["id"] in layouts else []
                alt = r.get("lerobot_view", {})
                if alt.get("initial") is not None:
                    available_geometry.append(geom_key(alt["names"], alt["initial"]))
                for ns, pose in available_geometry:
                    if ns in held_geometry:
                        xyz, angle = distances(held_geometry[ns], pose)
                        if ((xyz <= XYZ_TOL) & (angle <= ANGLE_TOL)).any():
                            veto.append("held-out initial geometry")
            r["veto"] = sorted(set(veto))
            all_rows.append(r)
            assert r["id"] not in by_id
            by_id[r["id"]] = r
    dump(out / "tasks.json", task_info)
    dump(out / "episodes.json", all_rows)
    pair_stats = defaultdict(Counter)
    metrics = defaultdict(lambda: defaultdict(list))
    task_counts = defaultdict(Counter)
    proc_episodes = defaultdict(Counter)
    proc_tasks = defaultdict(set)
    paired = set()
    temp = out / "pairs.csv.tmp"
    with temp.open("w") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "benchmark",
                "split",
                "demo",
                "query",
                "status",
                "layout",
                "procedure_match",
                "max_translation_m",
                "max_rotation_rad",
            ]
        )
        for j, rows in zip(jobs, groups.values()):
            b, s = j["benchmark"], j["split"]
            key = (b, s)
            tc = task_counts[key]
            tc["official_tasks"] += 1
            tc["tasks_with_rgb"] += int(any(r["rgb"] for r in rows))
            tc["indexed_successes"] += len(rows)
            tc["rgb_episodes"] += sum(r["rgb"] for r in rows)
            tc["no_rgb"] += sum(not r["rgb"] for r in rows)
            tc["vetoed_episodes"] += sum(bool(r["veto"]) for r in rows)
            kept = [r for r in rows if not r["veto"]]
            rgb = [r for r in kept if r["rgb"]]
            tc["eligible_rgb_episodes"] += len(rgb)
            tc["eligible_rgb_tasks"] += bool(rgb)
            for r in rgb:
                proc_episodes[key][r["procedure"] or "unconfirmed"] += 1
                if r["procedure"]:
                    proc_tasks[key].add((j["suite"], j["task"]))
            if s == "test":
                lookup = {r["e1_name"]: r for r in rows}
                pairs = [(lookup[d], lookup[q]) for d, q in j["test_pairs"]]
            else:
                pairs = permutations(rgb, 2)
            task_positive = 0
            for d, q in pairs:
                di, qi = d["id"], q["id"]
                assert di != qi and not d["veto"] and not q["veto"]
                x, a = None, None
                layout = "unknown"
                if di in layouts and qi in layouts:
                    ns, pg = layouts[di]
                    qs, qg = layouts[qi]
                    if ns == qs:
                        x, a = map(float, distances(pg, qg))
                        layout = "different" if x > XYZ_TOL or a > ANGLE_TOL else "same"
                match = (
                    "unknown"
                    if not d["procedure"] or not q["procedure"]
                    else "same"
                    if d["procedure"] == q["procedure"]
                    else "different"
                )
                if layout != "different":
                    status = "layout_" + layout
                elif d["action_sha256"] == q["action_sha256"]:
                    status = "duplicate_actions"
                elif match == "same":
                    status = "positive"
                else:
                    status = "procedure_" + match
                if status == "positive":
                    task_positive += 1
                    paired.update([di, qi])
                    metrics[key]["max_translation_m"].append(x)
                    metrics[key]["max_rotation_rad"].append(a)
                w.writerow([b, s, di, qi, status, layout, match, x, a])
                pair_stats[key][status] += 1
            tc["tasks_with_positives"] += int(task_positive > 0)
    os.replace(temp, out / "pairs.csv")
    # The runtime reads only its own role index. Query records carry no labels,
    # future-derived procedure metadata, trajectory lengths or action hashes.
    roles = out / "roles"
    roles.mkdir(exist_ok=True)
    permitted = (
        "id",
        "benchmark",
        "path",
        "storage",
        "rgb_transform",
        "image_convention",
    )
    clip_fields = permitted + ("frames", "visual_sha256")

    def project(r, keys):
        return {k: r[k] for k in keys if k in r}

    train = [
        project(r, clip_fields + ("action_offset", "geometry_source"))
        for r in all_rows
        if r["split"] == "train" and not r["veto"] and r["rgb"]
    ]
    # Alternate RGB streams are for episode-level use and have their own timebase.
    by_train_id = {r["id"]: r for r in train}
    for r in all_rows:
        if r["id"] in by_train_id and "lerobot_view" in r:
            by_train_id[r["id"]]["lerobot_view"] = project(
                r["lerobot_view"], ("frames", "path", "visual_sha256")
            )
    dump(roles / "train_episodes.json", train)
    with (
        (out / "pairs.csv").open() as f,
        (roles / "train_pairs.csv.tmp").open("w") as tr,
        (roles / "test_pairs.csv.tmp").open("w") as te,
    ):
        train_writer, test_writer = csv.writer(tr), csv.writer(te)
        donors, queries = set(), set()
        for r in csv.DictReader(f):
            pair = r["demo"], r["query"]
            if r["split"] == "test":
                test_writer.writerow(pair)
                donors.add(pair[0])
                queries.add(pair[1])
            elif r["status"] == "positive":
                train_writer.writerow(pair)
    assert not donors & queries
    dump(
        roles / "test_donors.json",
        [project(by_id[d], clip_fields) for d in sorted(donors)],
    )
    dump(
        roles / "test_queries.json",
        [project(by_id[q], permitted) for q in sorted(queries)],
    )
    for split in ("train", "test"):
        os.replace(roles / (split + "_pairs.csv.tmp"), roles / (split + "_pairs.csv"))
    stats = {}
    for key, counts in task_counts.items():
        b, s = key
        stats.setdefault(b, {})[s] = dict(
            **counts,
            pair_status=pair_stats[key],
            episodes_with_positive=sum(
                r["id"] in paired
                for r in all_rows
                if (r["benchmark"], r["split"]) == key
            ),
            procedure_known_tasks=len(proc_tasks[key]),
            procedure_episodes=proc_episodes[key],
            positive_layout_distribution={
                k: percentile(v) for k, v in metrics[key].items()
            },
        )
    for b in OFFICIAL:
        stats[b]["e1_manifest_sha256"] = hashlib.sha256(
            (OFFICIAL[b] / "manifest.json").read_bytes()
        ).hexdigest()
    dump(out / "statistics.json", stats)
    missing = []
    for j in jobs:
        if j["benchmark"] == "bpp" and not Path(j["raw"]).exists():
            src = read_json(ROOT / j["e1"]["source"] / "source.json")
            missing.append(
                dict(
                    task=j["task"],
                    suite=j["suite"],
                    path=j["raw"],
                    url=src["url"],
                    bytes=src["size"],
                    revision=src["revision"],
                    local_state_episodes=src["episodes"],
                )
            )
    dump(out / "missing_rgb.json", missing)
    print(
        json.dumps(
            {
                b: {
                    s: [stats[b][s]["indexed_successes"], stats[b][s]["pair_status"]]
                    for s in ("train", "test")
                }
                for b in OFFICIAL
            }
        )
    )
