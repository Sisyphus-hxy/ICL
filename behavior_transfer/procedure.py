"""Conservative order labels for conjunctive two-object LIBERO placement tasks."""

import re
import h5py
import numpy as np


def two_placements(row, goal, path):
    atoms = re.findall(r"\(([^()]+)\)", goal.lower())
    predicates = [a.split() for a in atoms]
    if len(predicates) != 2 or any(
        len(p) != 3 or p[0] not in ("on", "in") for p in predicates
    ):
        return None
    targets = {p[1]: (p[0], p[2]) for p in predicates}
    if len(targets) != 2 or not targets.keys() <= set(row["names"]):
        return None
    if (
        row["geometry_source"]
        and "replay" in row["geometry_source"]
        and row["replay_xyz_p95_m"] > 0.025
    ):
        return None
    with h5py.File(path, "r") as f:
        if "object_poses" not in f:
            return None
        xyz = f["object_poses"][:, :, :3]
        ee = f["eef_pose"][:, :3]
        closed = f["actions"][:, 6] > 0.5
    edges = np.diff(np.r_[False, closed, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) != 2:
        return None
    sequence = []
    for a, b in zip(starts, ends):
        matches = []
        for name in targets:
            pos = xyz[a:b, row["names"].index(name)]
            lift = pos[:, 2] - pos[0, 2] > 0.025
            nearby = np.linalg.norm(pos - ee[a:b], axis=1) < 0.13
            if (lift & nearby).sum() >= 3:
                matches.append(name)
        if len(matches) != 1:
            return None
        sequence.extend(matches)
    if len(set(sequence)) != 2:
        return None
    return " > ".join(
        f"grasp {n} > lift {n} > {targets[n][0]} {targets[n][1]}" for n in sequence
    )


def fixture_procedure(row, path):
    """Verify cabinet/stove events in the recorded joint time series for these stock tasks."""
    import xml.etree.ElementTree as ET

    cases = {
        "open_the_middle_drawer_of_the_cabinet": (
            "wooden_cabinet_1_middle_level",
            "open",
        ),
        "open_the_top_drawer_and_put_the_bowl_inside": (
            "wooden_cabinet_1_top_level",
            "open_place",
        ),
        "turn_on_the_stove": ("flat_stove_1_button", "turn_on"),
    }
    if row["task"] not in cases:
        return None
    name, kind = cases[row["task"]]
    with h5py.File(path, "r") as f:
        if "raw" not in f:
            return None
        g = f["raw"]
        world = ET.fromstring(g.attrs["model_file"]).find("worldbody")
        offset = 1
        for joint in world.iter("joint"):
            if joint.get("name") == name:
                break
            offset += {"free": 7, "ball": 4}.get(joint.get("type"), 1)
        else:
            return None
        value = g["states"][:, offset]
        active = value >= 0.5 if kind == "turn_on" else value < -0.14
        if active[0] or not active[-1] or active.sum() < 3:
            return None
        if kind == "turn_on":
            return "turn on flat_stove_1"
        if kind == "open":
            return "open wooden_cabinet_1_middle_level"
        if "akita_black_bowl_1" not in row["names"]:
            return None
        i = row["names"].index("akita_black_bowl_1")
        target = f["object_poses"][:, i, :3]
        lift = target[:, 2] - target[0, 2] > 0.025
        contact = np.linalg.norm(target - f["eef_pose"][:, :3], axis=1) < 0.13
        lifted = lift & contact & (f["actions"][:, 6] > 0.5)
        if (
            lifted.sum() >= 3
            and np.flatnonzero(active)[0]
            < np.flatnonzero(lifted)[0] + row["action_offset"]
        ):
            return "open wooden_cabinet_1_top_level > grasp akita_black_bowl_1 > lift > place in top drawer"
    return None
