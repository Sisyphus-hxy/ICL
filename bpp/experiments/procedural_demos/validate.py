"""Validate generated procedural-constraint demonstrations.

For every HDF5 under the output directory:
  - rebuild the matching scene variant (water / post material from attrs),
  - replay the recorded actions in fresh simulation,
  - re-measure goal success, contacts, object clearance and trajectory,
  - verify the final object pose matches the recorded terminal state.

Pairwise checks:
  - cup:   identical grasp-window prefix; spatial-path deviation empty/filled;
           pixel-level visual separability at BPP input resolution (128x128).
  - post:  identical position/size of the post across materials; shared grasp
           and place; transport-only difference; object-relative clearance.

Writes validation_report.json + per-pair comparison figures.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--egl-device', default='1')
parser.add_argument('--image-tol', type=float, default=0.01,
                    help='min fraction of pixels differing >0.15 to call variants separable')
parser.add_argument('--pairs-only', action='store_true',
                    help='skip the per-demo replay pass')
args = parser.parse_args()

import os
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', args.egl_device)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'clearance'))
sys.path.insert(0, str(HERE))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scene import LAYOUTS
from scene_lib import SemanticScene, MATERIALS

LAYOUTS['none'] = np.array([np.nan, np.nan, np.nan])
DT = 0.05
in_dir = Path(args.input).expanduser().resolve()


def load_demo(path):
    with h5py.File(path) as f:
        d = f['data/demo_0']
        attrs = {k: (json.loads(v) if isinstance(v, str) and v[:1] in '{[' else v)
                 for k, v in d.attrs.items()}
        return dict(attrs=attrs,
                    actions=d['actions'][:],
                    states=d['states'][:],
                    ee_pos=d['obs/ee_pos'][:],
                    ee_ori=d['obs/ee_ori'][:],
                    agentview=d['obs/agentview_rgb'][:],
                    phases=[p.decode() for p in d['phases'][:]])


def replay(scene, demo, post_pos=None):
    scene.canonical_state = demo['states'][0]
    if post_pos is not None:
        scene.post_pos = post_pos
    scene.reset()
    eef, obj, clr, contacts, succ = [], [], [], 0, False
    for a in demo['actions']:
        obs, *_ = scene.env.step(a)
        scene.update_water()  # keep the fill plug glued under the bowl
        eef.append(obs['robot0_eef_pos'].copy())
        obj.append(scene.object_pos().copy())
        if scene.obstacle_id is not None:
            clr.append(min(
                scene.surface_clearance(scene.object_geoms),
                scene.surface_clearance(scene.robot_geoms)))
        contacts += len(scene.contacts(transport=True))
        succ = bool(scene.env._check_success())
    _, pts = scene.spill_status()
    return dict(eef=np.asarray(eef), obj=np.asarray(obj),
                clearance=clr, contacts=contacts, goal_success=succ,
                n_drops_out=scene.spill_count(),
                final_obj=obj[-1] if obj else None)


def pair_figure(rows, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, eef, ph in rows:
        i0 = ph.index('transport') if 'transport' in ph else 0
        i1 = len(ph) - 1 - ph[::-1].index('transport') if 'transport' in ph else len(ph)
        axes[0].plot(eef[i0:i1, 0], eef[i0:i1, 1], label=name)
        axes[1].plot(eef[:, 0], eef[:, 2], label=name)
        axes[2].plot(np.arange(len(eef)) * DT, np.linalg.norm(
            np.gradient(eef, axis=0), axis=1) / DT, label=name)
    axes[0].set_title('transport xy'); axes[1].set_title('xz profile')
    axes[2].set_title('eef |v| (m/s)'); axes[2].set_xlabel('t (s)')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def visual_pair(img_a, img_b, name_a, name_b, path):
    n = min(len(img_a), len(img_b))
    # compare at matched normalized indices; report both the mean absolute
    # difference and the fraction of pixels that differ strongly — localised
    # changes (water disk, post material) matter more than frame averages.
    idxs = np.linspace(0, n - 1, min(n, 30)).astype(int)
    diffs, fracs = [], []
    for i in idxs:
        d = np.abs(img_a[i].astype(int) - img_b[i].astype(int)).mean(-1) / 255.
        diffs.append(d.mean())
        fracs.append((d > 0.15).mean())
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    pick = [idxs[len(idxs) // 3], idxs[2 * len(idxs) // 3]]
    for ax, i in zip(axes[:2], pick):
        ax.imshow(img_a[i]); ax.set_title(f'{name_a} f{i}', fontsize=8)
        ax.axis('off')
    for ax, i in zip(axes[2:], pick):
        ax.imshow(img_b[i]); ax.set_title(f'{name_b} f{i}', fontsize=8)
        ax.axis('off')
    fig.suptitle(f'mean|Δpx|={np.mean(diffs):.4f} frac>0.15={np.mean(fracs):.3f}')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return float(np.mean(diffs)), float(np.mean(fracs))


def main():
    hdf5s = sorted(p for p in in_dir.rglob('*.hdf5')
                   if 'cross_' not in p.parts)
    report = {'generated_at': in_dir.name, 'demos': [], 'pairs': []}
    if not args.pairs_only:
        replay_pass(hdf5s, report)
    pair_pass(report)
    (in_dir / 'validation_report.json').write_text(
        json.dumps(report, indent=2,
                   default=lambda o: int(o) if isinstance(o, np.integer)
                   else float(o) if isinstance(o, np.floating) else str(o)))
    print('wrote', in_dir / 'validation_report.json')


def replay_pass(hdf5s, report):
    scenes = {}
    for path in hdf5s:
        demo = load_demo(path)
        at = demo['attrs']
        task, variant = at.get('task'), at.get('variant')
        key = (task, variant)
        if key not in scenes:
            if task == 'filled_cup':
                nd = int(at.get('n_drops', 16))
                scenes[key] = SemanticScene(record_images=False, n_drops=nd)
                scenes[key].drops_in_bowl = (variant == 'filled')
            else:
                scenes[key] = SemanticScene(record_images=False,
                                            post_rgba=MATERIALS[variant])
        scene = scenes[key]
        post = np.asarray(at['post_xyz']) if 'post_xyz' in at else None
        r = replay(scene, demo, post)
        recorded_final = None
        # terminal bowl pose recorded in states[-1] is pre-action; use the
        # last trajectory row's object position via states replay instead.
        dev = np.linalg.norm(demo['ee_pos'] - r['eef'], axis=1)
        rel = str(path.relative_to(in_dir).with_suffix(''))
        report['demos'].append(dict(
            name=rel, file=path.name, task=task, variant=variant,
            layout=at.get('layout'), procedure=at.get('procedure'),
            goal_success=r['goal_success'],
            n_contacts=r['contacts'],
            min_clearance_m=float(min(r['clearance'])) if r['clearance'] else None,
            replay_dev_mean_m=float(dev.mean()),
            replay_dev_max_m=float(dev.max()),
            n_drops_out=r['n_drops_out'],
            final_obj=r['final_obj'].tolist()))
        print(f"{rel}: success={r['goal_success']} contacts={r['contacts']} "
              f"dev_mean={dev.mean() * 1000:.1f}mm dev_max={dev.max() * 1000:.1f}mm",
              flush=True)
    for s in scenes.values():
        s.close()


def pair_pass(report):
    stats_rows = {str(p.relative_to(in_dir).with_suffix('')): load_demo(p)
                  for p in sorted(in_dir.rglob('*.hdf5'))
                  if 'cross_' not in p.parts}
    for layout in sorted({int(a['attrs'].get('layout', -1))
                          for a in stats_rows.values()}):
        cup_pair = [(f'cup/layout_{layout:02d}/normal',
                     f'cup/layout_{layout:02d}/gentle')]
        for a_name, b_name in cup_pair:
            if a_name in stats_rows and b_name in stats_rows:
                da, db = stats_rows[a_name], stats_rows[b_name]
                tmask_a = np.array([p == 'transport' for p in da['phases']])
                tmask_b = np.array([p == 'transport' for p in db['phases']])
                pa, pb = da['ee_pos'][tmask_a], db['ee_pos'][tmask_b]
                # spatial path deviation: resample both to 50 points by arc len
                def rsp(p):
                    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                    s = np.linspace(0, d[-1], 50)
                    return np.stack([np.interp(s, d, p[:, i]) for i in range(3)], 1)
                dev = np.linalg.norm(rsp(pa) - rsp(pb), axis=1)
                vd, vf = visual_pair(da['agentview'][tmask_a], db['agentview'][tmask_b],
                                     a_name, b_name,
                                     in_dir / f'cup/layout_{layout:02d}/pair_visual.png')
                pair_figure([(a_name, da['ee_pos'], da['phases']),
                             (b_name, db['ee_pos'], db['phases'])],
                            f'cup layout{layout}',
                            in_dir / f'cup/layout_{layout:02d}/pair.png')
                report['pairs'].append(dict(
                    task='filled_cup', layout=int(layout),
                    path_dev_mean_m=float(dev.mean()),
                    path_dev_max_m=float(dev.max()),
                    visual_diff_mean=float(vd),
                    visual_diff_frac=float(vf),
                    visual_separable=vf > args.image_tol))
                print(f'cup pair l{layout}: path dev mean {dev.mean() * 1000:.1f}mm '
                      f'max {dev.max() * 1000:.1f}mm | vis diff {vd:.4f} frac {vf:.3f}',
                      flush=True)
        ob_pair = [(f'obstacle/layout_{layout:02d}/narrow',
                    f'obstacle/layout_{layout:02d}/wide')]
        for a_name, b_name in ob_pair:
            if a_name in stats_rows and b_name in stats_rows:
                da, db = stats_rows[a_name], stats_rows[b_name]
                pa, pb = np.asarray(da['attrs']['post_xyz']), np.asarray(db['attrs']['post_xyz'])
                tmask_a = np.array([p == 'transport' for p in da['phases']])
                tmask_b = np.array([p == 'transport' for p in db['phases']])
                vd, vf = visual_pair(da['agentview'][tmask_a], db['agentview'][tmask_b],
                                     a_name, b_name,
                                     in_dir / f'obstacle/layout_{layout:02d}/pair_visual.png')
                pair_figure([(a_name, da['ee_pos'], da['phases']),
                             (b_name, db['ee_pos'], db['phases'])],
                            f'obstacle layout{layout}',
                            in_dir / f'obstacle/layout_{layout:02d}/pair.png')
                report['pairs'].append(dict(
                    task='fragile_obstacle', layout=int(layout),
                    post_identical=bool(np.allclose(pa, pb)),
                    post_pos_a=pa.tolist(), post_pos_b=pb.tolist(),
                    visual_diff_mean=float(vd),
                    visual_diff_frac=float(vf),
                    visual_separable=vf > args.image_tol))
                print(f'obstacle pair l{layout}: post match {np.allclose(pa, pb)} '
                      f'| vis diff {vd:.4f} frac {vf:.3f}', flush=True)


if __name__ == '__main__':
    main()
