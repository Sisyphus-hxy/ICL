"""Generate semantically-labelled procedural-constraint demonstrations.

Task A — Filled Cup: Empty vessel -> Normal transport; visually Filled vessel
(light-blue level water surface, mocap visual only) -> Gentle transport
(same spatial path, reduced speed/acceleration/jerk/tilt via a bounded-rate
pose servo — continuous motion, not static frames).

Task B — Fragile Obstacle: identical post geometry; Plastic material ->
Narrow clearance path, Glass material -> Wide clearance path. Same grasp,
place and goal; only the transport segment differs.

Five layouts = source demos 0..4 (different bowl/box poses and paths). All
labels are stored as HDF5/manifest metadata and are never part of the model
input. No training, no BPP changes, no policy evaluation.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['cup', 'obstacle', 'all'])
parser.add_argument('--output', required=True)
parser.add_argument('--egl-device', default='0')
parser.add_argument('--layouts', type=int, default=5)
parser.add_argument('--cup-only', choices=['all', 'gentle'], default='all',
                    help='gentle: regenerate only the filled_gentle variant')
parser.add_argument('--gentle-speed', type=float, default=0.03,
                    help='target m/s for gentle transport (path_len/DT/step budget)')
parser.add_argument('--obstacle-xy', default='auto')
parser.add_argument('--narrow-clearance-m', type=float, default=0.005)
parser.add_argument('--wide-clearance-m', type=float, default=0.018)
parser.add_argument('--wide-end-shift-m', type=float, default=0.025)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--n-drops', type=int, default=16,
                    help='free spill spheres inside the Filled bowl '
                         '(real dynamics; parked outside for Empty)')
args = parser.parse_args()

import os
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', args.egl_device)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'clearance'))
sys.path.insert(0, str(HERE))

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from scene import LAYOUTS
from recording import Recorder
from expert import Servo, resample
from scene_lib import SemanticScene, SOURCE, MATERIALS

LAYOUTS['none'] = np.array([np.nan, np.nan, np.nan])
DT = 0.05
out_dir = Path(args.output).expanduser().resolve()


# ---------------------------------------------------------------- data utils

def demo_info(i):
    with h5py.File(SOURCE) as f:
        d = f[f'data/demo_{i}']
        return dict(state=d['states'][0].copy(), states=d['states'][:],
                    actions=d['actions'][:], ee_pos=d['obs/ee_pos'][:],
                    ee_ori=d['obs/ee_ori'][:],
                    seed=int(d.attrs.get('seed', 2)))


def find_windows(scene, info):
    """Transport window [t0, t1) and release index from the bowl-z profile.
    Parks any mocap post under the table first so it cannot perturb the bowl."""
    n = len(info['actions'])
    parked = None
    if scene.obstacle_id is not None:
        parked = scene.d.mocap_pos[scene.mocap_id].copy()
        scene.d.mocap_pos[scene.mocap_id] = [0., 0., 0.5]
    bowlz = np.empty(n)
    for k in range(n):
        scene.set_state(info['states'][k])
        scene.env.sim.forward()
        bowlz[k] = scene.object_pos()[2]
    if parked is not None:
        scene.d.mocap_pos[scene.mocap_id] = parked
    rel_idx = np.nonzero(np.diff(info['actions'][:, 6]) < -0.5)[0]
    rel = int(rel_idx[0]) + 1 if len(rel_idx) else n - 12
    peak_z = float(bowlz.max())
    high = np.nonzero(bowlz > peak_z - 0.02)[0]
    return dict(t0=int(high[0]), t1=int(high[-1]) + 1, rel=rel,
                rest_z=float(bowlz[0]), peak_z=peak_z, n=n)


def phase_of(i, w):
    if i < w['t0'] - 12:
        return 'grasp'
    if i < w['t0']:
        return 'lift'
    if i < w['t1']:
        return 'transport'
    if i < w['rel']:
        return 'place'
    return 'release'


class ProcRecorder(Recorder):
    """Recorder that keeps the mocap water surface glued under the bowl and
    logs per-step spill status of the free drop contents."""

    def _monitored_step(self):
        super()._monitored_step()
        self.scene.update_water()

    def step(self, action):
        out = super().step(action)
        if self.rows:
            inside, _ = self.scene.spill_status()
            self.rows[-1]['n_drops_out'] = int((~inside).sum()) \
                if len(inside) else 0
        return out


def quat_at(info, i):
    return Rotation.from_rotvec(info['ee_ori'][i]).as_quat()


def eef_pose_at(info, i):
    return np.r_[info['ee_pos'][i], quat_at(info, i)]


def level_quat(quat):
    """Same yaw as `quat` but with the eef z-axis pointing straight down
    (vessel held level)."""
    r = Rotation.from_quat(quat).as_matrix()
    z = np.array([0., 0., -1.])
    x = np.array([r[0, 0], r[1, 0], 0.])
    if np.linalg.norm(x) < 1e-3:
        x = np.array([r[0, 1], r[1, 1], 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()


def servo_waypoints(servo, rec, poses, phase, max_pos_speed=0.5,
                    max_rot_speed=0.5, steps_per=2):
    rec.phase = phase
    for pose in poses:
        servo.move_to_pose(pose, close_gripper=True, max_steps=steps_per,
                           pos_threshold=0., rot_threshold=0.,
                           max_pos_speed=max_pos_speed,
                           max_rot_speed=max_rot_speed)


def chase_path(servo, rec, path_pts, quats, n_steps, phase,
               lookahead=4, max_pos_speed=0.08, max_rot_speed=0.12):
    """Pure-pursuit tracking: each control step advances a carrot point along
    the arc-length-parametrized path. Actual transport speed ~= path length /
    (n_steps * dt); motion is continuous (no static frames)."""
    rec.phase = phase
    dists = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))]
    total = dists[-1]
    rot = Slerp(np.linspace(0., 1., len(quats)), Rotation.from_quat(quats))
    for k in range(n_steps):
        s = min(total, ((k + lookahead) / n_steps) * total)
        pos = np.array([np.interp(s, dists, path_pts[:, d]) for d in range(3)])
        quat = rot(s / total).as_quat()
        servo.move_to_pose(np.r_[pos, quat], close_gripper=True, max_steps=1,
                           pos_threshold=0., rot_threshold=0.,
                           max_pos_speed=max_pos_speed,
                           max_rot_speed=max_rot_speed)


def tilt_deg(quats):
    """Vessel tilt from level: angle between bowl-up (-eef z) and world +z."""
    z = Rotation.from_quat(np.asarray(quats)).as_matrix()[:, :, 2]
    return np.degrees(np.arccos(np.clip(-z[:, 2], -1., 1.)))


def replay_segment(rec, actions, w, i0, i1):
    for i in range(i0, i1):
        rec.phase = phase_of(i, w)
        rec.step(actions[i])


def demo_metrics(rec, w):
    traj = np.genfromtxt(rec.directory / (rec.name + '_trajectory.csv'),
                         delimiter=',', names=True, dtype=None, encoding='utf8')
    contacts = json.loads((rec.directory / (rec.name + '_contacts.json')).read_text())
    mask = traj['phase'] == 'transport'
    seg = traj[mask]
    eef = np.stack([seg[k] for k in ('eef_x', 'eef_y', 'eef_z')], axis=1) \
        if mask.any() else np.zeros((1, 3))
    v = np.linalg.norm(np.diff(eef, axis=0), axis=1) / DT if len(eef) > 2 else np.zeros(1)
    a = np.diff(v) / DT if len(v) > 1 else np.zeros(1)
    j = np.diff(a) / DT if len(a) > 1 else np.zeros(1)
    quats = [o['robot0_eef_quat'] for o in rec.observations
             if 'robot0_eef_quat' in o]
    idx = np.nonzero(mask)[0]
    tilts = tilt_deg([quats[i] for i in idx if i < len(quats)]) \
        if len(quats) and len(idx) else None
    post_contacts = [ev for ev in contacts
                     if any('procedure_obstacle' in c[0] or 'procedure_obstacle' in c[1]
                            for c in ev['contacts'])]
    post_tr = [ev for ev in post_contacts if ev['phase'] == 'transport']
    post_pl = [ev for ev in post_contacts if ev['phase'] != 'transport']
    return dict(steps=int(len(traj)),
                transport_steps=int(mask.sum()),
                transport_duration_s=float(mask.sum()) * DT,
                transport_path_length_m=float(
                    np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()) if len(eef) > 1 else 0.,
                eef_speed_mean_m_s=float(v.mean()),
                eef_speed_p95_m_s=float(np.percentile(v, 95)),
                eef_acc_p95_m_s2=float(np.percentile(np.abs(a), 95)) if len(a) else 0.,
                eef_jerk_p95_m_s3=float(np.percentile(np.abs(j), 95)) if len(j) else 0.,
                tilt_mean_deg=float(tilts.mean()) if tilts is not None else None,
                tilt_max_deg=float(tilts.max()) if tilts is not None else None,
                object_min_clearance_m=float(seg['object_clearance_m'].min())
                if mask.any() else None,
                n_contact_events=len(contacts),
                post_contact_events=len(post_contacts),
                post_contacts_transport=len(post_tr),
                post_contacts_other=len(post_pl),
                n_drops_out_max=int(traj['n_drops_out'].max())
                if 'n_drops_out' in traj.dtype.names else 0,
                n_drops_out_end=int(traj['n_drops_out'][-1])
                if 'n_drops_out' in traj.dtype.names else 0,
                first_spill_step=int(np.nonzero(traj['n_drops_out'] > 0)[0][0])
                if 'n_drops_out' in traj.dtype.names
                and (traj['n_drops_out'] > 0).any() else -1)


def save_attrs(rec, extra_attrs):
    path = rec.directory / (rec.name + '.hdf5')
    with h5py.File(path, 'a') as f:
        demo = f['data/demo_0']
        for k, v in extra_attrs.items():
            demo.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v


# ---------------------------------------------------------------- task A

def gen_cup(layout, scene, stats):
    info = demo_info(layout)
    scene.canonical_state = info['state']
    w = find_windows(scene, info)
    print('  windows', w, flush=True)
    base = dict(task='filled_cup', layout=layout, source_demo=layout, window=w)
    # --- Empty + Normal: plain replay of the native demonstration.
    leaf = out_dir / 'cup' / f'layout_{layout:02d}'
    if args.cup_only == 'all':
        scene.water_enabled = False
        scene.drops_in_bowl = False
        scene.reset()
        rec = ProcRecorder(leaf, 'normal', save_images=True)
        rec.start(scene)
        replay_segment(rec, info['actions'], w, 0, w['n'])
        summary = rec.finish(scene)
        r = dict(success=summary['goal_success'], **demo_metrics(rec, w))
        save_attrs(rec, dict(**base, name=f'cup/layout_{layout:02d}/normal',
                             variant='empty', procedure='normal',
                             water='none', n_drops=0,
                             drops='parked outside workspace'))
        stats.append(dict(name=f'cup/layout_{layout:02d}/normal',
                          task='cup', layout=layout,
                          variant='empty_normal', **r))
    # --- Filled + Gentle: replay grasp/lift, then a bounded-rate servo along
    # the demo's own eef path with levelled orientation, then replayed
    # place/release. Continuous motion — no static frames. Filled = a bowl
    # heaped with free blue spheres (real spillable contents; no water disk).
    scene.water_enabled = False
    scene.drops_in_bowl = True
    scene.reset()
    rec = ProcRecorder(leaf, 'gentle', save_images=True)
    rec.start(scene)
    servo = Servo(rec)
    replay_segment(rec, info['actions'], w, 0, w['t0'])
    # Dense arc-length resample of the demo's own transport eef path.
    path = resample(info['ee_pos'][w['t0'] - 1:w['t1']], 60)
    dists = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    path_len = dists[-1]
    # Level for the first 65% of the path, then blend back to the demo's
    # place-entry orientation so the replayed place segment stays identical.
    q_level = level_quat(rec.latest_obs['robot0_eef_quat'].copy())
    q_end = quat_at(info, w['t1'] - 1)
    rot = Slerp([0., 1.], Rotation.from_quat([q_level, q_end]))
    blend = np.clip((np.linspace(0., 1., 60) - 0.65) / 0.35, 0., 1.)
    quats = [rot(b).as_quat() for b in blend]
    n_steps = int(round(path_len / (args.gentle_speed * DT)))
    chase_path(servo, rec, path, quats, n_steps, 'transport')
    servo.move_to_pose(np.r_[info['ee_pos'][w['t1'] - 1], q_end],
                       close_gripper=True, max_steps=50,
                       pos_threshold=0.002, rot_threshold=0.015,
                       max_pos_speed=args.gentle_speed * 3,
                       max_rot_speed=0.3)
    # Track the demo's own place path (arc over the box, then vertical drop)
    # with the same servo — reproduces the shared place instead of a blind
    # straight-line descent that clips the box approach.
    place_pts = resample(info['ee_pos'][w['t1'] - 1:w['rel']], 40)
    place_quats = [quat_at(info, i) for i in range(w['t1'] - 1, w['rel'])]
    chase_path(servo, rec, place_pts, place_quats,
               max(30, w['rel'] - w['t1']), 'place',
               max_pos_speed=0.15, max_rot_speed=0.3)
    servo.move_to_pose(eef_pose_at(info, w['rel'] - 1), close_gripper=True,
                       max_steps=50, pos_threshold=0.002, rot_threshold=0.02,
                       max_pos_speed=0.1, max_rot_speed=0.2)
    replay_segment(rec, info['actions'], w, w['rel'], w['n'])
    summary = rec.finish(scene)
    r = dict(success=summary['goal_success'], **demo_metrics(rec, w))
    save_attrs(rec, dict(**base, name=f'cup/layout_{layout:02d}/gentle',
                         variant='filled', procedure='gentle',
                         water='none',
                         drops=(f'{scene.n_drops} free blue spheres heaped in '
                                'bowl - real dynamics, spillable'),
                         n_drops=scene.n_drops,
                         gentle_max_speed_m_s=args.gentle_speed))
    stats.append(dict(name=f'cup/layout_{layout:02d}/gentle',
                      task='cup', layout=layout, variant='filled_gentle', **r))


# ---------------------------------------------------------------- task B

def auto_place(scene, info, w):
    """Grid-search a post position on the corridor that stays out of the
    demonstration's pre-transport swept volume and grazes the direct path."""
    robot, bowl, states = scene.robot_geoms, scene.object_geoms, info['states']

    def gd(pos, i, geoms, cutoff=None):
        scene.set_state(states[i])
        scene.d.mocap_pos[scene.mocap_id] = pos
        scene.env.sim.forward()
        out6 = np.empty(6)
        best = np.inf
        for g in geoms:
            best = min(best, float(mujoco.mj_geomDistance(
                scene.m, scene.d, g, scene.obstacle_id, 2., out6)))
            if cutoff is not None and best < cutoff:
                return best
        return best

    L = w['t1'] - w['t0']
    sp = list(range(0, w['t0'], 2))                              # pre-transport
    s_edge = list(range(w['t0'], w['t0'] + max(3, L // 3)))      # path start
    sm = list(range(w['t0'] + max(3, L // 2), w['t1']))          # late graze
    spl = list(range(w['t1'], w['n']))                           # place/release
    TH = dict(pre=0.005, start=0.015, place=0.005)
    veto = dict(pre=0, start=0, place=0, graze=0, ok=0)
    best = None
    for z in (0.95, 0.99, 1.03, 0.91):
        for x in np.arange(-0.06, 0.121, 0.01):
            for y in np.arange(-0.08, 0.121, 0.01):
                pos = np.array([x, y, z])
                pre = np.inf
                for i in sp:
                    pre = min(pre, gd(pos, i, robot + bowl, TH['pre']))
                    if pre < TH['pre']:
                        break
                if pre < TH['pre']:
                    veto['pre'] += 1
                    continue
                # The post must graze the LATE part of the direct transport
                # path (3-8 mm), stay out of the approach swept volume and
                # clear of the shared start pose and the place descent.
                start_c = min(gd(pos, i, bowl) for i in s_edge)
                if start_c < TH['start']:
                    veto['start'] += 1
                    continue
                place = min(gd(pos, i, robot + bowl) for i in spl)
                if place < TH['place']:
                    veto['place'] += 1
                    continue
                graze = min(gd(pos, i, bowl) for i in sm)
                if not (0.003 <= graze <= 0.008):
                    veto['graze'] += 1
                    continue
                veto['ok'] += 1
                score = (abs(graze - 0.005), -min(pre, start_c, place))
                if best is None or score < best[0]:
                    best = (score, pos, pre, graze, start_c, place)
    if best is None:
        print('  vetoes:', veto, flush=True)
        return None
    print('  post', np.round(best[1], 3).tolist(), veto,
          'pre %.4f graze %.4f start %.4f place %.4f'
          % (best[2], best[3], best[4], best[5]), flush=True)
    return best[1]


def curved_path(start, end, peak, n=40):
    t = np.linspace(0., 1., 4001)
    p = start[None] * (1 - t[:, None]) + end[None] * t[:, None]
    p[:, 1] += (peak - (start[1] + end[1]) / 2) * np.sin(np.pi * t) ** 2
    return resample(p, n)


def gen_obstacle_variant(scene, info, w, name, peak, y_shift, direct=False,
                         directory=None):
    """Replay grasp/lift, transport (direct replay or servoed detour), a
    shared servo place descent, then replayed release."""
    scene.reset()
    rec = ProcRecorder(directory or out_dir, name,
                       save_images=not name.startswith('_tuning'))
    rec.start(scene)
    servo = Servo(rec)
    replay_segment(rec, info['actions'], w, 0, w['t0'])
    if direct:
        replay_segment(rec, info['actions'], w, w['t0'], w['t1'])
    else:
        start = rec.latest_obs['robot0_eef_pos'].copy()
        end_pos = info['ee_pos'][w['t1'] - 1] + np.array([0., y_shift, 0.])
        q0 = Rotation.from_quat(rec.latest_obs['robot0_eef_quat'].copy())
        q1 = Rotation.from_quat(quat_at(info, w['t1'] - 1))
        slerp = Slerp([0., 1.], Rotation.from_quat([q0.as_quat(), q1.as_quat()]))
        path = curved_path(start, end_pos, peak)
        poses = [np.r_[p, slerp((k + 1) / len(path)).as_quat()]
                 for k, p in enumerate(path)]
        servo_waypoints(servo, rec, poses, 'transport', steps_per=2)
        servo.move_to_pose(np.r_[end_pos, q1.as_quat()], close_gripper=True,
                           max_steps=40, pos_threshold=0.002,
                           rot_threshold=0.01)
        # Track the demo's own place path (shared arc-over-then-drop) rather
        # than a straight diagonal that can clip the box approach.
        place_pts = resample(info['ee_pos'][w['t1'] - 1:w['rel']], 40)
        place_quats = [quat_at(info, i) for i in range(w['t1'] - 1, w['rel'])]
        chase_path(servo, rec, place_pts, place_quats,
                   max(30, w['rel'] - w['t1']), 'place',
                   max_pos_speed=0.15, max_rot_speed=0.3)
        servo.move_to_pose(eef_pose_at(info, w['rel'] - 1), close_gripper=True,
                           max_steps=50, pos_threshold=0.002,
                           rot_threshold=0.02, max_pos_speed=0.1,
                           max_rot_speed=0.2)
        replay_segment(rec, info['actions'], w, w['rel'], w['n'])
        summary = rec.finish(scene)
        m = demo_metrics(rec, w)
        m.update(peak=peak, y_shift=y_shift, success=summary['goal_success'])
        return m, rec
    replay_segment(rec, info['actions'], w, w['t1'], w['n'])
    summary = rec.finish(scene)
    m = demo_metrics(rec, w)
    m.update(peak=0., y_shift=0., success=summary['goal_success'])
    return m, rec


def tune(scene, info, w, target, lo, hi, y_shift, iters=8):
    """Bisect the detour apex: smallest tested peak whose transport stays
    collision-free and reaches `target` clearance. Contacts outside the
    transport segment are reported but do not steer the apex."""
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        r, _ = gen_obstacle_variant(scene, info, w, '_tuning', mid, y_shift)
        print('  tune peak=%.4f -> clr %s succ %s tr_post %d other_post %d' %
              (mid, r['object_min_clearance_m'], r['success'],
               r['post_contacts_transport'], r['post_contacts_other']), flush=True)
        clr = r['object_min_clearance_m']
        if r['post_contacts_transport'] > 0 or clr is None or clr < target:
            lo = mid
        elif r['success']:
            hi, best = mid, mid
        else:
            hi = mid
    return hi if best is None else best


def gen_obstacle(layout, stats):
    info = demo_info(layout)
    # One shared placement and window set for both material variants.
    scene = SemanticScene(record_images=True, post_rgba=MATERIALS['plastic'])
    try:
        scene.canonical_state = info['state']
        w = find_windows(scene, info)
        post = auto_place(scene, info, w)
    finally:
        scene.close()
    if post is None:
        print('  layout%d: no feasible post position — skipped' % layout, flush=True)
        return False
    print('  post', np.round(post, 3).tolist(), 'windows', w, flush=True)
    LAYOUTS['post'] = post
    LAYOUTS['plastic'] = post
    LAYOUTS['glass'] = post
    base = dict(task='fragile_obstacle', layout=layout, source_demo=layout,
                window=w, post_xyz=post.tolist())
    mid_y = float(np.mean(info['ee_pos'][w['t0']:w['t1'], 1]))
    for material, proc, shift in (('plastic', 'narrow', 0.),
                                  ('glass', 'wide', args.wide_end_shift_m)):
        scene = SemanticScene(record_images=True,
                              post_rgba=MATERIALS[material])
        try:
            scene.canonical_state = info['state']
            scene.post_pos = post
            scene.material = material
            leaf = out_dir / 'obstacle' / f'layout_{layout:02d}'
            rel = f'obstacle/layout_{layout:02d}/{proc}'
            if proc == 'narrow':
                # Plastic takes the native direct route — no extra margin.
                r, rec = gen_obstacle_variant(scene, info, w, proc,
                                              peak=0., y_shift=0., direct=True,
                                              directory=leaf)
            else:
                target = max(args.wide_clearance_m,
                             stats[-1]['object_min_clearance_m'] + 0.012)
                peak = tune(scene, info, w, target, mid_y, mid_y + 0.30, shift)
                r, rec = gen_obstacle_variant(scene, info, w, proc, peak, shift,
                                              directory=leaf)
            save_attrs(rec, dict(**base, name=rel,
                                 variant=material, procedure=proc,
                                 material=material,
                                 fragility=('expressed via clearance margin only'
                                            if material == 'glass' else 'n/a'),
                                 peak=r['peak'], y_shift=shift))
            stats.append(dict(name=rel, task='obstacle', layout=layout,
                              variant=f'{material}_{proc}',
                              post_xyz=post.tolist(), **r))
        finally:
            scene.close()
    return True


# ---------------------------------------------------------------- driver

def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    if args.stage in ('cup', 'all'):
        scene = SemanticScene(record_images=True, water=True,
                              n_drops=args.n_drops)
        try:
            for layout in range(args.layouts):
                print(f'=== cup layout {layout} ===', flush=True)
                gen_cup(layout, scene, stats)
        finally:
            scene.close()
    if args.stage in ('obstacle', 'all'):
        made = 0
        layout = 0
        while made < args.layouts and layout < 25:
            print(f'=== obstacle layout {layout} ===', flush=True)
            if gen_obstacle(layout, stats):
                made += 1
            layout += 1
        print(f'obstacle: {made}/{args.layouts} layout pairs generated', flush=True)
    path = out_dir / 'demo_stats.csv'
    rows = {}
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):
                rows[(r['task'], r['layout'], r['variant'])] = r
    for r in stats:
        key = (r['task'], str(r['layout']), r['variant'])
        rows[key] = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                     for k, v in r.items()}
    keys = sorted({k for r in rows.values() for k in r})
    with path.open('w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w_.writeheader()
        for r in rows.values():
            w_.writerow(r)
    manifest = [dict(task=r['task'], layout=r['layout'], variant=r['variant'],
                     source_demo=r.get('source_demo', r['layout']),
                     success=r['success'],
                     semantics=('filled vessel -> gentle handling'
                                if r['task'] == 'cup' else
                                'fragile glass -> wider clearance'))
                for r in rows.values()]
    keys = sorted({k for r in manifest for k in r})
    with (out_dir / 'manifest.csv').open('w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=keys)
        w_.writeheader()
        w_.writerows(manifest)
    # per-leaf <proc>_statistics.csv (same rows, one per leaf)
    for r in rows.values():
        if not r.get('name'):
            continue
        leaf_stats = out_dir / f"{r['name']}_statistics.csv"
        leaf_stats.parent.mkdir(parents=True, exist_ok=True)
        with leaf_stats.open('w', newline='') as f:
            w_ = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w_.writeheader()
            w_.writerow(r)
    for junk in out_dir.rglob('_tuning_*'):
        junk.unlink()
    print('done', flush=True)


if __name__ == '__main__':
    main()
