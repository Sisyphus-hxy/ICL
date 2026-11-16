"""Analyze natural execution-procedure variation in raw LIBERO demos.

For each downloaded task hdf5 (50 demos):
  1. Segment every demo into approach / grasp / transport / place using the
     gripper command flips and the target-object z profile.
  2. Extract per-demo metrics: object-frame grasp offset, approach direction,
     transport height / lateral deviation / path length, speed / accel / jerk.
  3. Canonical alignment (translate object start to origin, rotate
     object->goal direction to +x) + per-phase arc-length resampling, so
     scene layout and duration do not masquerade as procedure differences.
  4. Distribution plots + aligned-trajectory overlays; multimodality checks
     (GMM BIC, bimodality coefficient, KDE mode count, regression on scene
     covariates).  If a stable multimodal structure appears, representative
     demos per mode get mp4 videos rendered from the stored rgb obs.

No data generation, no scene changes, no rollouts — everything comes from
the recorded states/obs/actions in the source hdf5 files.
"""
import json
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde, kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

BPP = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
DT = 1.0 / 20.0          # control_freq = 20 Hz
RESAMPLE = 60            # points per phase for aligned plotting

# flat-state layout (identical in both files, verified against model_file):
#   [0]=time, [1:10] robot qpos, then 5 free joints (pos3+quat4 each),
#   then 4 fixture joints, then qvel.  Pos slice start = 1 + 9 + 7*i.
OBJ_SLICE = {"bowl1": 10, "bowl2": 17, "cookies": 24, "ramekin": 31, "plate": 38}

TASKS = {
    "libero_spatial_bowl_between_to_plate": {
        "path": BPP / "datasets/libero/libero_spatial"
                / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        "target": "bowl1", "goal": "plate",
        "label": "LIBERO-v1 official: bowl (between plate & ramekin) -> plate",
    },
    "selected_comb_bowl_to_cookies": {
        "path": BPP / "datasets/libero/libero_spatial_selected_combinations_view"
                / "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo.hdf5",
        "target": "bowl1", "goal": "cookies",
        "label": "generated (libero_spatial_extra): bowl (table center) -> cookies box",
    },
}


# ---------------------------------------------------------------- loading

def load_demo(g):
    st = g["states"][:]
    return {
        "ee": g["obs/ee_pos"][:],
        "ee_ori": g["obs/ee_ori"][:],
        "finger": np.abs(g["obs/gripper_states"][:]).mean(1),
        "gcmd": g["actions"][:, 6],
        "actions": g["actions"][:, :6],
        "obj_pos": {k: st[:, s:s + 3] for k, s in OBJ_SLICE.items()},
        "obj_quat": {k: st[:, s + 3:s + 7] for k, s in OBJ_SLICE.items()},
        "n": st.shape[0],
        "rgb_agent": g["obs/agentview_rgb"][:],
        "rgb_hand": g["obs/eye_in_hand_rgb"][:],
    }


def sustained_flip(cmd, want):
    """First index i where cmd switches to `want` and holds it >=3 steps."""
    flips = np.nonzero(np.diff(cmd) != 0)[0]
    for f in flips:
        if cmd[f + 1] == want and np.all(cmd[f + 1:f + 4] == want):
            return int(f + 1)
    return None


def segment(d):
    """Phase boundary indices via gripper-command flips + object z profile."""
    z = d["obj_pos_z"]
    n = d["n"]
    t_close = sustained_flip(d["gcmd"], 1.0)      # -1 open / +1 close
    t_open = sustained_flip(d["gcmd"][t_close:], -1.0)
    t_open = None if t_open is None else t_open + t_close
    z_rest = float(np.median(z[:8]))
    z_fin = float(np.median(z[-8:]))
    above = np.nonzero(z > z_rest + 0.010)[0]
    above = above[above > (t_close or 0)]
    t_lift = int(above[0]) if len(above) else int(np.argmax(z))
    hi_end = t_open if (t_open and t_open > t_lift) else n
    z_peak = float(z[t_lift:hi_end].max())
    # last index still within 1.2 cm of the peak -> descent starts right after
    near = np.nonzero(z[:hi_end] >= z_peak - 0.012)[0]
    t_desc = int(near[-1]) + 1 if len(near) else t_lift + 1
    low = np.nonzero(z[t_desc:hi_end] <= z_fin + 0.012)[0]
    t_down = t_desc + int(low[0]) if len(low) else hi_end - 1
    return dict(t_close=t_close, t_lift=t_lift, t_desc=t_desc,
                t_down=t_down, t_open=t_open, z_rest=z_rest,
                z_fin=z_fin, z_peak=z_peak)


# ---------------------------------------------------------------- metrics

def canonical(pts, origin_xy, goal_xy):
    """Translate object start to origin, rotate object->goal onto +x."""
    th = -math.atan2(goal_xy[1] - origin_xy[1], goal_xy[0] - origin_xy[0])
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    out = pts.copy()
    out[..., :2] = (pts[..., :2] - origin_xy) @ R.T
    return out, th


def resample(path, m=RESAMPLE):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(path[:1], m, axis=0)
    t = np.linspace(0, s[-1], m)
    return np.stack([np.interp(t, s, path[:, k]) for k in range(3)], 1)


def deriv(x):
    v = np.gradient(x, DT, axis=0)
    a = np.gradient(v, DT, axis=0)
    j = np.gradient(a, DT, axis=0)
    return np.linalg.norm(v, axis=1), np.linalg.norm(a, axis=1), np.linalg.norm(j, axis=1)


def metrics(d, seg, task):
    n = d["n"]
    ee = d["ee"]
    obj = d["obj_pos"][task["target"]]
    goal0 = d["obj_pos"][task["goal"]][0].copy()
    quat = d["obj_quat"][task["target"]]
    tc, tl, td, to = seg["t_close"], seg["t_lift"], seg["t_desc"], seg["t_open"]
    td = min(td, n)
    speed, accel, jerk = deriv(ee)

    # contact = closest eef-object approach inside [t_close, t_lift]
    lo = slice(max(tc - 2, 0), min(tl + 3, n))
    t_con = lo.start + int(np.argmin(np.linalg.norm(ee[lo] - obj[lo], axis=1)))
    off_w = ee[t_con] - obj[t_con]
    off_l = Rotation.from_quat(quat[t_con], scalar_first=True).inv().apply(off_w)

    # canonical frame for this demo
    R = np.eye(3)
    origin, goal = obj[0, :2].copy(), goal0[:2]
    ee_c, th = canonical(ee, origin, goal)
    obj_c, _ = canonical(obj, origin, goal)

    # approach direction: last 15 steps before contact
    k = max(t_con - 15, 0)
    v = ee_c[t_con] - ee_c[k]
    v_app_world = ee[t_con] - ee[k]
    az_can = math.degrees(math.atan2(v[1], v[0])) if np.linalg.norm(v[:2]) > 1e-6 else np.nan
    elev = math.degrees(math.atan2(v_app_world[2], np.linalg.norm(v_app_world[:2])))
    src = ee_c[k] - obj_c[t_con]         # where the gripper came from (obj frame)
    src_az = math.degrees(math.atan2(src[1], src[0]))
    pre = np.linalg.norm(ee[:t_con] - obj[:t_con], axis=1)
    near_idx = np.nonzero(pre < 0.05)[0]
    hover = 0
    if len(near_idx):                    # consecutive steps within 5 cm pre-contact
        run = np.split(near_idx, np.nonzero(np.diff(near_idx) > 1)[0] + 1)
        hover = len(run[-1]) if run[-1][-1] >= t_con - 2 else 0

    tr = slice(tl, max(td, tl + 1))
    tr_z_obj = d["obj_pos"][task["target"]][tr, 2]
    tr_y = ee_c[tr, 1]
    pl = float(np.sum(np.linalg.norm(np.diff(ee[tr], axis=0), axis=1)))
    direct = float(np.linalg.norm(ee_c[min(td, n - 1), :2] - ee_c[tl, :2]))
    ap = slice(0, tc)

    return dict(
        n_steps=n, duration_s=n * DT,
        t_close=tc, t_lift=tl, t_desc=td, t_down=seg["t_down"], t_open=to,
        n_flips=int((np.diff(d["gcmd"]) != 0).sum()),
        n_regrasp=max(int((np.diff(d["gcmd"]) != 0).sum()) - 2, 0) // 2,
        released=bool(to is not None),
        obj0_x=obj[0, 0], obj0_y=obj[0, 1], obj0_z=obj[0, 2],
        goal_x=goal0[0], goal_y=goal0[1], goal_z=goal0[2],
        obj_goal_dist=float(np.linalg.norm(goal0[:2] - obj[0, :2])),
        grasp_off_x=off_l[0], grasp_off_y=off_l[1], grasp_off_z=off_l[2],
        grasp_r_xy=float(np.linalg.norm(off_l[:2])),
        approach_azimuth_deg=az_can, approach_source_azimuth_deg=src_az,
        approach_elev_deg=elev, pregrasp_hover_s=hover * DT,
        approach_len=float(np.sum(np.linalg.norm(np.diff(ee[ap], axis=0), axis=1))),
        approach_speed_mean=float(speed[ap].mean()) if tc > 1 else np.nan,
        transport_len=pl, transport_direct=direct,
        transport_curvature=pl / max(direct, 1e-6),
        transport_obj_z_mean=float(tr_z_obj.mean()),
        transport_obj_z_max=float(tr_z_obj.max()),
        transport_lift_mean=float((tr_z_obj - seg["z_rest"]).mean()),
        transport_lift_max=float((tr_z_obj - seg["z_rest"]).max()),
        transport_eef_z_mean=float(ee[tr, 2].mean()),
        transport_lat_abs_max=float(np.abs(tr_y).max()),
        transport_lat_min=float(tr_y.min()), transport_lat_max=float(tr_y.max()),
        transport_lat_mean=float(np.abs(tr_y).mean()),
        transport_speed_p50=float(np.percentile(speed[tr], 50)),
        transport_speed_p95=float(np.percentile(speed[tr], 95)),
        transport_speed_max=float(speed[tr].max()),
        transport_accel_p95=float(np.percentile(accel[tr], 95)),
        transport_jerk_p95=float(np.percentile(jerk[tr], 95)),
        transport_jerk_rms=float(np.sqrt((jerk[tr] ** 2).mean())),
        transport_hover_frac=float((speed[tr] < 0.02).mean()),
        grasp_dur_s=(tl - tc) * DT, transport_dur_s=(td - tl) * DT,
        place_dur_s=(n - td) * DT, approach_dur_s=tc * DT,
    )


# ---------------------------------------------------------------- stats

def bimodality_coef(x):
    x = np.asarray(x, float)
    return float((skew(x) ** 2 + 1) / max(kurtosis(x, fisher=False), 1e-9))


def kde_modes(x, grid=200):
    x = np.asarray(x, float)
    xs = np.linspace(x.min() - 0.05 * x.ptp(), x.max() + 0.05 * x.ptp(), grid)
    try:
        k = gaussian_kde(x)
        d = np.diff(np.sign(np.diff(k(xs))))
        return int((d < 0).sum())
    except np.linalg.LinAlgError:
        return -1


def gmm_scan(X, kmax=4, seed=0):
    """BIC for k=1..kmax; best k>1 gets silhouette + sizes."""
    X = np.asarray(X, float)
    res = []
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-5,
                            random_state=seed, n_init=8).fit(X)
        res.append((k, g.bic(X), g))
    best = min(res, key=lambda r: r[1])
    out = {"bic": {k: b for k, b, _ in res}, "best_k": best[0]}
    if best[0] > 1:
        lab = best[2].predict(X)
        out["silhouette"] = float(silhouette_score(X, lab)) if len(set(lab)) > 1 else np.nan
        out["sizes"] = np.bincount(lab).tolist()
        out["labels"] = lab
        out["model"] = best[2]
    return out


def lin_r2(y, X):
    return float(LinearRegression().fit(np.asarray(X, float), np.asarray(y, float)).score(
        np.asarray(X, float), np.asarray(y, float)))


# ---------------------------------------------------------------- videos

def write_video(d, path, fps=15):
    a, h = d["rgb_agent"], d["rgb_hand"]
    frames = [np.concatenate([a[t], h[t]], 1) for t in range(d["n"])]
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


# ---------------------------------------------------------------- figures

def fig_distributions(m, out):
    keys = ["grasp_r_xy", "grasp_off_z", "approach_azimuth_deg",
            "approach_elev_deg", "pregrasp_hover_s",
            "transport_lift_mean", "transport_lift_max",
            "transport_lat_abs_max", "transport_curvature",
            "transport_speed_p50", "transport_speed_p95",
            "transport_jerk_rms", "grasp_dur_s",
            "obj_goal_dist", "n_steps"]
    fig, ax = plt.subplots(4, 4, figsize=(17, 13))
    for a, k in zip(ax.ravel(), keys):
        a.hist(m[k], bins=16, color="#4878cf", alpha=.8, edgecolor="w")
        a.set_title(f"{k}\nBC={bimodality_coef(m[k]):.2f} modes={kde_modes(m[k])}",
                    fontsize=8)
        a.tick_params(labelsize=7)
    for a in ax.ravel()[len(keys):]:
        a.axis("off")
    fig.suptitle("Per-demo metric distributions (BC>0.555 suggests multimodality)")
    fig.tight_layout()
    fig.savefig(out / "fig_distributions.png", dpi=130)
    plt.close(fig)


def fig_aligned(trajs, phases, m, out, tag=""):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    names = ["approach", "grasp", "transport", "place"]
    colors = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"]
    for i, P in enumerate(trajs):
        col = plt.cm.tab10(i % 10)
        for ph, c in zip(names, colors):
            p = P[ph]
            ax[0].plot(p[:, 0], p[:, 1], color=c, lw=.6, alpha=.35)
            ax[1].plot(np.linspace(0, 1, len(p)), p[:, 2], color=c, lw=.6, alpha=.35)
        ax[0].plot(trajs[i]["transport"][:, 0], trajs[i]["transport"][:, 1],
                   color=col, lw=1.0, alpha=.8)
        ax[1].plot(np.linspace(0, 1, RESAMPLE), trajs[i]["transport"][:, 2],
                   color=col, lw=1.0, alpha=.8)
    for ph, c in zip(names, colors):
        ax[0].plot([], [], color=c, label=ph)
        ax[1].plot([], [], color=c, label=ph)
    ax[0].set_title("canonical XY (obj0=origin, obj->goal=+x)")
    ax[0].set_xlabel("x' (m)"); ax[0].set_ylabel("y' (m)"); ax[0].legend(fontsize=7)
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].set_title("z vs phase-normalized progress")
    ax[1].set_xlabel("progress in phase"); ax[1].set_ylabel("z (m)"); ax[1].legend(fontsize=7)
    sc = ax[2].scatter(m["transport_lat_abs_max"], m["transport_lift_mean"],
                       c=m["approach_azimuth_deg"], cmap="coolwarm", s=28, edgecolor="k",
                       linewidth=.3)
    ax[2].set_xlabel("transport |lateral| max (m)"); ax[2].set_ylabel("lift mean (m)")
    ax[2].set_title("transport height vs lateral dev\ncolor=approach azimuth")
    fig.colorbar(sc, ax=ax[2], label="deg")
    fig.suptitle("50 demos aligned: " + tag)
    fig.tight_layout()
    fig.savefig(out / "fig_aligned.png", dpi=130)
    plt.close(fig)


def fig_grasp(m, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    sc = ax[0].scatter(m.grasp_off_x, m.grasp_off_y, c=m.grasp_off_z,
                       cmap="viridis", s=30, edgecolor="k", linewidth=.3)
    ax[0].set_xlabel("grasp offset x (obj frame)"); ax[0].set_ylabel("y")
    ax[0].set_title("grasp offset XY (color=z)"); fig.colorbar(sc, ax=ax[0])
    ax[1].hist(m.approach_azimuth_deg, bins=18, color="#4878cf", edgecolor="w")
    ax[1].set_xlabel("approach azimuth deg (canonical)"); ax[1].set_title("approach direction")
    sc2 = ax[2].scatter(m.approach_azimuth_deg, m.approach_elev_deg,
                        c=m.transport_lift_mean, cmap="viridis", s=30,
                        edgecolor="k", linewidth=.3)
    ax[2].set_xlabel("approach azimuth"); ax[2].set_ylabel("approach elevation deg")
    ax[2].set_title("approach dir (color=lift)"); fig.colorbar(sc2, ax=ax[2])
    fig.tight_layout(); fig.savefig(out / "fig_grasp.png", dpi=130); plt.close(fig)


def fig_approach_clusters(trajs, m, labels, centers_deg, out, task):
    """Approach/grasp paths colored by circular-kmeans cluster + polar hist."""
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.2))
    cmap = plt.cm.tab10
    for P, lab in zip(trajs, labels):
        for ph in ("approach", "grasp"):
            p = P[ph]
            ax[0].plot(p[:, 0], p[:, 1], color=cmap(lab), lw=.8, alpha=.55)
    for lab in sorted(set(labels)):
        ax[0].plot([], [], color=cmap(lab), lw=2,
                   label=f"cluster {lab} (n={(labels == lab).sum()}, dir~{centers_deg[lab]:.0f}°)")
    ax[0].scatter([0], [0], marker="o", s=60, facecolor="w", edgecolor="k", zorder=5)
    ax[0].set_title(f"approach paths by cluster\n{task['label']}")
    ax[0].set_xlabel("x' (m)"); ax[0].set_ylabel("y' (m)")
    ax[0].legend(fontsize=7); ax[0].set_aspect("equal", adjustable="datalim")
    # polar histogram of travel direction at contact
    axp = fig.add_subplot(1, 3, 2, polar=True)
    az = np.deg2rad(m.approach_azimuth_deg)
    bins = np.linspace(-np.pi, np.pi, 25)
    for lab in sorted(set(labels)):
        h, _ = np.histogram(az[labels == lab], bins)
        axp.bar(bins[:-1], h, width=bins[1] - bins[0], alpha=.45,
                color=cmap(lab), label=f"c{lab}")
    axp.set_title("travel dir at contact (polar)")
    axp.legend(fontsize=7)
    sc = ax[2].scatter(m.approach_source_azimuth_deg, m.approach_elev_deg,
                       c=labels, cmap="tab10", s=30, edgecolor="k", linewidth=.3)
    ax[2].set_xlabel("source azimuth deg (obj frame)"); ax[2].set_ylabel("elevation deg")
    ax[2].set_title("where the gripper came from vs steepness")
    fig.tight_layout(); fig.savefig(out / "fig_approach_clusters.png", dpi=130)
    plt.close(fig)


def circular_kmeans(deg, k=3, seed=0):
    a = np.deg2rad(deg)
    X = np.stack([np.cos(a), np.sin(a)], 1)
    lab = KMeans(k, n_init=20, random_state=seed).fit_predict(X)
    centers = {i: float(np.degrees(np.arctan2(np.sin(a[lab == i]).mean(),
                                              np.cos(a[lab == i]).mean())))
               for i in range(k)}
    return lab, centers


def fig_regrip(d, seg, out, dn):
    t = np.arange(d["n"]) * DT
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(t, d["gcmd"], lw=1.2); ax[0].set_ylabel("gripper cmd")
    ax[0].plot(t, (d["finger"] - d["finger"].min())
               / max(d["finger"].ptp(), 1e-9) * 2 - 1, lw=.8, alpha=.7,
               label="finger qpos (norm)")
    ax[0].legend(fontsize=7); ax[0].set_title(f"{dn}: gripper events")
    ax[1].plot(t, d["obj_pos_z"], lw=1.2); ax[1].set_ylabel("obj z (m)")
    for b, nm in [(seg["t_close"], "close"), (seg["t_lift"], "lift"),
                  (seg["t_desc"], "desc"), (seg["t_open"], "open")]:
        if b is None:
            continue
        for a in ax:
            a.axvline(b * DT, ls="--", lw=.8, color="k", alpha=.5)
    ax[1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out / f"fig_regrip_{dn}.png", dpi=110); plt.close(fig)


def fig_traj_pca(trajs, m, out, labels=None):
    X = np.stack([np.concatenate([t["transport"][:, 0], t["transport"][:, 1],
                                  t["transport"][:, 2]]) for t in trajs])
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    P = PCA(3).fit_transform(Xz)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    c = labels if labels is not None else m["transport_lift_mean"]
    sc = ax[0].scatter(P[:, 0], P[:, 1], c=c, cmap="tab10" if labels is not None else "viridis",
                       s=30, edgecolor="k", linewidth=.3)
    for i, p in enumerate(P):
        ax[0].annotate(str(i), p[:2], fontsize=5, alpha=.6)
    ax[0].set_title("transport-shape PCA (canonical, resampled)")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); fig.colorbar(sc, ax=ax[0])
    sc = ax[1].scatter(m.transport_speed_p50, m.transport_jerk_rms,
                       c=m.transport_lift_mean, cmap="viridis", s=30,
                       edgecolor="k", linewidth=.3)
    ax[1].set_xlabel("transport speed p50"); ax[1].set_ylabel("jerk RMS")
    ax[1].set_title("dynamics (color=lift)"); fig.colorbar(sc, ax=ax[1])
    fig.tight_layout(); fig.savefig(out / "fig_traj_pca.png", dpi=130); plt.close(fig)
    return Xz


# ---------------------------------------------------------------- main

def run_task(key, task):
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    rows, trajs, demos, segs = [], [], {}, {}
    with h5py.File(task["path"], "r") as f:
        demo_ids = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for dn in demo_ids:
            d = load_demo(f["data"][dn])
            d["obj_pos_z"] = d["obj_pos"][task["target"]][:, 2]
            demos[dn] = d
    for dn in demo_ids:
        d = demos[dn]
        seg = segment(d)
        segs[dn] = seg
        row = metrics(d, seg, task)
        row["demo"] = dn
        rows.append(row)
        # aligned per-phase resampled eef path
        origin = d["obj_pos"][task["target"]][0, :2]
        goal = d["obj_pos"][task["goal"]][0, :2]
        ee_c, _ = canonical(d["ee"], origin, goal)
        bounds = dict(approach=(0, seg["t_close"]),
                      grasp=(seg["t_close"], seg["t_lift"]),
                      transport=(seg["t_lift"], min(seg["t_desc"], d["n"])),
                      place=(min(seg["t_desc"], d["n"]), d["n"]))
        trajs.append({ph: resample(ee_c[a:b]) for ph, (a, b) in bounds.items()})

    m = pd.DataFrame(rows).set_index("demo")
    m.to_csv(out / "metrics.csv")
    ph = pd.DataFrame({dn: segs[dn] for dn in demo_ids}).T
    ph.index.name = "demo"
    ph.to_csv(out / "phases.csv")
    # initial scene table (all object positions, for comparability checks)
    init = {dn: {f"{k}_{ax}": demos[dn]["obj_pos"][k][0, i]
                 for k in OBJ_SLICE for i, ax in enumerate("xyz")}
            for dn in demo_ids}
    pd.DataFrame(init).T.rename_axis("demo").to_csv(out / "initial_scene.csv")

    fig_distributions(m, out)
    fig_aligned(trajs, None, m, out, tag=task["label"])
    fig_grasp(m, out)

    # ---- multimodality analysis -----------------------------------------
    az_rad = np.deg2rad(m.approach_azimuth_deg)
    m["az_cos"], m["az_sin"] = np.cos(az_rad), np.sin(az_rad)
    feat = ["grasp_r_xy", "grasp_off_z", "az_cos", "az_sin",
            "approach_elev_deg", "pregrasp_hover_s",
            "transport_lift_mean", "transport_lat_abs_max",
            "transport_curvature", "transport_speed_p50",
            "transport_jerk_rms", "transport_hover_frac",
            "grasp_dur_s", "approach_dur_s"]
    X = m[feat].to_numpy(float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    gmm = gmm_scan(Xz)
    labels = gmm.get("labels")
    if labels is not None:
        m["cluster"] = labels
    # circular kmeans on contact travel direction — robust to the ±180 wrap
    az_lab, az_ctr = circular_kmeans(m.approach_azimuth_deg, k=3)
    m["approach_cluster"] = az_lab
    m.to_csv(out / "metrics.csv")
    fig_approach_clusters(trajs, m, az_lab, az_ctr, out, task)
    Xtraj = fig_traj_pca(trajs, m, out, labels)

    # re-grip / unreleased demos — discrete procedure events, plot + flag
    flagged = m[(m.n_flips > 2) | (~m.released)].index.tolist()
    for dn in flagged:
        fig_regrip(demos[dn], segs[dn], out, dn)

    # approach-cluster vs scene covariates (is the mode scene-forced?)
    from scipy.stats import f_oneway, kruskal
    scene_cols = ["obj0_x", "obj0_y", "goal_x", "goal_y", "obj_goal_dist"]
    az_scene = {}
    for c in scene_cols:
        try:
            az_scene[c] = float(kruskal(
                *[m.loc[m.approach_cluster == i, c]
                  for i in sorted(set(az_lab))]).pvalue)
        except ValueError:
            az_scene[c] = float("nan")

    # scene-covariate regression: how much of each metric is explained by layout
    r2 = {k: lin_r2(m[k], m[scene_cols]) for k in feat}

    # 1-D structure on the two priority metrics
    uni = {}
    for k in ["transport_lift_mean", "transport_lift_max", "approach_azimuth_deg",
              "transport_lat_abs_max", "transport_speed_p50"]:
        x = m[k].to_numpy()
        g1 = gmm_scan(x.reshape(-1, 1), kmax=3)
        uni[k] = {"BC": bimodality_coef(x), "kde_modes": kde_modes(x),
                  "bic": g1["bic"], "best_k": g1["best_k"],
                  **({"silhouette": g1.get("silhouette"),
                      "sizes": g1.get("sizes")} if g1["best_k"] > 1 else {})}

    summary = {
        "task": task["label"], "n_demos": len(demo_ids),
        "metric_BC": {k: bimodality_coef(m[k]) for k in feat},
        "metric_scene_R2": r2,
        "approach_clusters": {
            "centers_deg": az_ctr,
            "sizes": np.bincount(az_lab).tolist(),
            "members": {int(i): m.index[m.approach_cluster == i].tolist()
                        for i in sorted(set(az_lab))},
            "kruskal_scene_pvalue": az_scene,
        },
        "regrip_demos": flagged,
        "unreleased_demos": m.index[~m.released].tolist(),
        "univariate_modes": uni,
        "multivariate_gmm": {k: v for k, v in gmm.items()
                             if k not in ("labels", "model")},
        "phase_bounds_mean": {c: float(m[c].mean())
                              for c in ["t_close", "t_lift", "t_desc", "t_down", "t_open"]},
        "key_metrics": {k: {"mean": float(m[k].mean()), "std": float(m[k].std()),
                            "min": float(m[k].min()), "max": float(m[k].max())}
                        for k in ["transport_lift_mean", "transport_lat_abs_max",
                                  "approach_azimuth_deg", "transport_speed_p50",
                                  "transport_jerk_rms", "grasp_r_xy"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- representative videos -------------------------------------------
    # For each approach cluster, the 2 demos closest to its (cos,sin) centre;
    # plus flagged re-grip/unreleased demos; plus comb's slow/wide subgroup.
    reps = {"approach_cluster": {}, "regrip_or_unreleased": flagged}
    vid_dir = out / "representatives"
    vid_dir.mkdir(exist_ok=True)
    cs = np.stack([m.az_cos, m.az_sin], 1)
    for cl in sorted(set(az_lab)):
        idx = np.nonzero(az_lab == cl)[0]
        ctr = cs[idx].mean(0)
        order = np.argsort(np.linalg.norm(cs[idx] - ctr, axis=1))[:2]
        picked = [m.index[idx[o]] for o in order]
        reps["approach_cluster"][int(cl)] = picked
        for dn in picked:
            write_video(demos[dn], vid_dir / f"approach_c{cl}_{dn}.mp4")
    for dn in flagged[:4]:
        write_video(demos[dn], vid_dir / f"regrip_{dn}.mp4")
    # task-specific secondary structure: slowest / widest transports
    extra = m.nsmallest(2, "transport_speed_p50").index.tolist() + \
        m.nlargest(2, "transport_lat_abs_max").index.tolist()
    reps["extremes"] = extra
    for dn in extra:
        write_video(demos[dn], vid_dir / f"extreme_{dn}.mp4")
    (vid_dir / "README.json").write_text(json.dumps(
        {"representatives": reps,
         "initial_scene": {dn: {f"{o}_{a}": round(float(demos[dn]["obj_pos"][o][0, i]), 4)
                                for o in OBJ_SLICE for i, a in enumerate("xyz")}
                           for dn in set(sum(reps["approach_cluster"].values(), [])
                                         + flagged[:4] + extra)},
         "note": "agentview | eye_in_hand side by side, 15 fps"}, indent=2))
    summary["representatives"] = reps
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return m, summary


if __name__ == "__main__":
    results = {}
    for key, task in TASKS.items():
        print(f"=== {key}")
        m, s = run_task(key, task)
        results[key] = s
        print(json.dumps(s["univariate_modes"], indent=2, default=float))
        print("multivariate_gmm:", s["multivariate_gmm"])
        print("scene R2:", {k: round(v, 3) for k, v in s["metric_scene_R2"].items()})
    print("done")
