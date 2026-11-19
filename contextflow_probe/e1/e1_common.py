"""Shared trajectory resampling, oracle geometry and evaluation metrics (xyzw)."""

import csv
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np

N_STEPS = 192  # fixed decode grid (uniform in episode time)


# ------------------------------------------------------------- resampling


def _nlerp_quats(quats, t_new, t_old):
    """Piecewise-linear quaternion resample on the same hemisphere."""
    q = np.array(quats, dtype=float, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    idx = np.searchsorted(t_old, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(t_old) - 2)
    a, b = q[idx], q[idx + 1]
    ta, tb = t_old[idx], t_old[idx + 1]
    w = np.divide(t_new - ta, tb - ta, out=np.zeros_like(t_new), where=tb > ta)[:, None]
    out = a * (1 - w) + b * w
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _linear(arr, t_new, t_old):
    idx = np.searchsorted(t_old, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(t_old) - 2)
    a, b = arr[idx], arr[idx + 1]
    ta, tb = t_old[idx], t_old[idx + 1]
    w = np.divide(t_new - ta, tb - ta, out=np.zeros_like(t_new), where=tb > ta)[:, None]
    return a * (1 - w) + b * w


def _nearest(arr, t_new, t_old):
    idx = np.searchsorted(t_old, t_new, side="left")
    idx = np.clip(idx, 0, len(t_old) - 1)
    prev = np.clip(idx - 1, 0, len(t_old) - 1)
    pick = np.where(np.abs(t_new - t_old[idx]) < np.abs(t_new - t_old[prev]), idx, prev)
    return arr[pick]


def resample_trajectory(eef_poses, grip_cmd, gripper=None, n=N_STEPS):
    """Uniform-in-time resample onto the decode grid.

    Returns x (n,9): pos3, quat4 (nlerp), grip_cmd1 (nearest, kept in
    {-1,+1}), phase1 in [0,1].  ``gripper`` (T,2) is appended separately as
    encoder-only proprioception -> returns gq (n,2) or None.
    """
    eef = np.asarray(eef_poses, dtype=float)
    T = len(eef)
    if T < 2 or eef.shape != (T, 7) or len(grip_cmd) != T:
        raise ValueError("Expected at least two aligned EEF and gripper samples")
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, n)
    pos = _linear(eef[:, :3], t_new, t_old)
    quat = _nlerp_quats(eef[:, 3:7], t_new, t_old)
    grip = _nearest(np.asarray(grip_cmd, dtype=float).reshape(-1, 1), t_new, t_old)
    # Canonicalize the first quaternion across demonstrations.
    if quat[0, np.argmax(np.abs(quat[0]))] < 0:
        quat *= -1
    phase = t_new[:, None]
    x = np.concatenate([pos, quat, grip, phase], axis=-1)
    gq = None
    if gripper is not None:
        gq = _linear(np.asarray(gripper, dtype=float).reshape(len(eef), -1), t_new, t_old)
    return x.astype(np.float32), (
        gq.astype(np.float32) if gq is not None else np.zeros((n, 2), np.float32)
    )


# ------------------------------------------------------------- geometry


def distinct_native_layouts(root, manifest):
    """Keep the first successful episode per physical oracle layout before splitting."""
    kept, repeated = [], {}
    representatives = {}
    for row in manifest["episodes"]:
        with np.load(root / (row["name"] + ".npz")) as rec:
            order = np.argsort(rec["body_names"])
            pose = rec["init_obj_poses"][order]
            names = tuple(rec["body_names"][order])
        matched = None
        for name, (other_names, other) in representatives.items():
            if names != other_names:
                continue
            position = np.linalg.norm(pose[:, :3] - other[:, :3], axis=1)
            # Distance is invariant to quaternion sign.
            dots = np.abs(np.sum(pose[:, 3:] * other[:, 3:], axis=1))
            if position.max() <= 1e-4 and dots.min() >= np.cos(0.001 / 2):
                matched = name
                break
        if matched is None:
            representatives[row["name"]] = names, pose
            kept.append(row)
        else:
            repeated[row["name"]] = matched
    return dict(manifest, episodes=kept), repeated


def geom_vector(init_obj_poses):
    """(K,7) -> flat [pos(3K), quat(4K)] geometry vector."""
    p = np.asarray(init_obj_poses, dtype=float)
    quat = p[:, 3:7].copy()
    sign = np.sign(quat[np.arange(len(p)), np.abs(quat).argmax(axis=1)])
    return np.r_[p[:, :3].ravel(), (quat * sign[:, None]).ravel()]


def shared_geometry(names, poses, slots):
    """Align donor/query by body name within a scene and pad to a fixed benchmark-wide width."""
    if len(names) != len(poses) or len(set(names)) != len(names) or len(names) > slots:
        raise ValueError("Invalid named oracle geometry for the shared model")
    aligned = np.zeros((slots, 7), dtype=float)
    aligned[: len(names)] = np.asarray(poses)[np.argsort(names)]
    return geom_vector(aligned)


def pose_metrics(pred, gt):
    """Offline decode metrics in resampled space.

    pred/gt: (..., n, >=8) pos3, quat4, grip1 columns.
    Returns dict of means over leading dims.
    """
    pos_err = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)
    dp = np.abs(np.sum(pred[..., 3:7] * gt[..., 3:7], axis=-1))
    rot_err = 2.0 * np.arccos(np.clip(dp, -1.0, 1.0))
    grip_pred = pred[..., 7] > 0
    grip_gt = gt[..., 7] > 0
    grip_acc = (grip_pred == grip_gt).astype(float)
    return {
        "pos_m": float(np.mean(pos_err)),
        "pos_max_m": float(np.max(pos_err)),
        "rot_rad": float(np.mean(rot_err)),
        "end_pos_m": float(np.mean(pos_err[..., -1])),
        "grip_acc": float(np.mean(grip_acc)),
    }


def save_csv(path, rows):
    # A stage summary also acts as the completion marker for batch resumption.
    with NamedTemporaryFile(mode="w", dir=path.parent, newline="", delete=False) as f:
        tmp = Path(f.name)
        try:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def cases(split):
    # Trackability controls use one run per distinct evaluation donor.
    for donor in split["donors"]:
        yield "same_track", donor, donor
    for d, q in split["test_pairs"]:
        yield ("heldout_combo" if q in split["heldout_cells"] else "heldout_layout"), d, q


def leaf(root, kind, donor, query):
    return root / "runs" / kind / query / "from" / donor
