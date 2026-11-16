"""Migrate outputs/ to the AGENTS.md output-directory convention.

Flat names
    cup_layout{N}_{visual}_{proc}.*
    obstacle_layout{N}_{mat}_{proc}.*
    cross_*/{task}_l{N}_{visual}_s{S}_{proc}.*
become
    <task>/layout_{NN}/<proc>.*
    cross_*/layout_{NN}/{visual}/seed{S}/<proc>.*

Also: adds `name` (extensionless relative path) to HDF5 attrs and to
manifest/demo_stats/rollout_results rows, splits demo_stats into per-leaf
<proc>_statistics.csv, updates evaluation_config prompt paths, and runs a
path-hit self-check. Idempotent.
"""
import csv
import json
import re
import shutil
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

TASK_OF = {'cup': {'empty_normal': 'normal', 'filled_gentle': 'gentle'},
           'obstacle': {'plastic_narrow': 'narrow', 'glass_wide': 'wide'}}


def move(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dst))


def set_name_attr(hdf5, name):
    with h5py.File(hdf5, 'a') as f:
        f['data/demo_0'].attrs['name'] = name


def main():
    moved = []
    # ---------------- generated demonstrations ----------------
    for task, variants in TASK_OF.items():
        for variant, proc in variants.items():
            for f in OUT.glob(f'{task}_layout*_{variant}*'):
                m = re.match(
                    rf'{task}_layout(\d+)_{variant}'
                    rf'(_trajectory\.csv|_contacts\.json|\.hdf5|\.mp4)$', f.name)
                if not m:
                    continue
                leaf = OUT / task / f'layout_{int(m.group(1)):02d}'
                dst = leaf / f'{proc}{m.group(2)}'
                move(f, dst)
                moved.append(dst)
                if dst.suffix == '.hdf5':
                    set_name_attr(dst, f'{task}/layout_{int(m.group(1)):02d}/{proc}')
        # pair figures: traj_* -> pair.png, vis_* -> pair_visual.png
        for f in OUT.glob(f'traj_{task}_layout*.png'):
            n = int(re.match(rf'traj_{task}_layout(\d+)', f.name).group(1))
            move(f, OUT / task / f'layout_{n:02d}' / 'pair.png')
        for f in OUT.glob(f'vis_{task}_layout*.png'):
            n = int(re.match(rf'vis_{task}_layout(\d+)', f.name).group(1))
            move(f, OUT / task / f'layout_{n:02d}' / 'pair_visual.png')

    # ---------------- cross-eval rollouts ----------------
    for exp in ('cross_cup', 'cross_obstacle', 'cross_obstacle_grasped'):
        d = OUT / exp
        if not d.is_dir():
            continue
        task = 'obstacle' if 'obstacle' in exp else 'cup'
        for f in sorted(d.glob(f'{task}_l*_*_s*_*')):
            m = re.match(
                rf'{task}_l(\d+)_([a-z]+)_s(\d+)_([a-z]+)'
                rf'(_trajectory\.csv|_contacts\.json|\.mp4)$', f.name)
            if not m:
                continue
            li, vis, seed, proc, ext = m.groups()
            leaf = d / f'layout_{int(li):02d}' / vis / f'seed{seed}'
            move(f, leaf / f'{proc}{ext}')
            moved.append(leaf / f'{proc}{ext}')
        # rollout_results: add `name` = extensionless relative path
        for res_file in ('rollout_results.json',):
            p = d / res_file
            if p.exists():
                rows = json.loads(p.read_text())
                for r in rows:
                    if 'name' not in r:
                        r['name'] = (f'{exp}/layout_{int(r["layout"]):02d}/'
                                     f'{r["visual"]}/seed{r["diff_seed"]}/'
                                     f'{r["prompt"]}')
                p.write_text(json.dumps(rows, indent=2, default=str) + '\n')
        p = d / 'rollout_results.csv'
        if p.exists():
            rows = list(csv.DictReader(p.open()))
            for r in rows:
                if not r.get('name'):
                    r['name'] = (f'{exp}/layout_{int(r["layout"]):02d}/'
                                 f'{r["visual"]}/seed{r["diff_seed"]}/'
                                 f'{r["prompt"]}')
            keys = sorted({k for r in rows for k in r})
            with p.open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        # evaluation_config prompt paths
        p = d / 'evaluation_config.json'
        if p.exists():
            cfg = json.loads(p.read_text())
            fixed = {}
            for k, v in cfg.get('prompt_paths', {}).items():
                vm = re.match(r'(cup|obstacle)_layout(\d+)_([a-z]+)_([a-z]+)\.hdf5',
                              Path(v).name)
                if vm:
                    t, li, _vis, proc = vm.groups()
                    fixed[k] = str(OUT / t / f'layout_{int(li):02d}' / f'{proc}.hdf5')
                else:
                    fixed[k] = v
            if fixed:
                cfg['prompt_paths'] = fixed
                p.write_text(json.dumps(cfg, indent=2) + '\n')

    # ---------------- top-level tables ----------------
    # demo_stats.csv: add name col, split per-leaf <proc>_statistics.csv
    stats = OUT / 'demo_stats.csv'
    if stats.exists():
        rows = list(csv.DictReader(stats.open()))
        for r in rows:
            task, li, variant = r['task'], int(r['layout']), r['variant']
            proc = TASK_OF[task][variant]
            r['name'] = f'{task}/layout_{li:02d}/{proc}'
            leaf_stats = OUT / task / f'layout_{li:02d}' / f'{proc}_statistics.csv'
            leaf_stats.parent.mkdir(parents=True, exist_ok=True)
            with leaf_stats.open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(r))
                w.writeheader()
                w.writerow(r)
        keys = sorted({k for r in rows for k in r})
        with stats.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    mani = OUT / 'manifest.csv'
    if mani.exists():
        rows = list(csv.DictReader(mani.open()))
        for r in rows:
            task, li, variant = r['task'], int(r['layout']), r['variant']
            r['name'] = f'{task}/layout_{li:02d}/{TASK_OF[task][variant]}'
        keys = sorted({k for r in rows for k in r})
        with mani.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    # validation_report.json: file -> relative name
    rep = OUT / 'validation_report.json'
    if rep.exists():
        r = json.loads(rep.read_text())
        for d in r.get('demos', []):
            m = re.match(r'(cup|obstacle)_layout(\d+)_([a-z]+_[a-z]+)\.hdf5',
                         d.get('file', ''))
            if m:
                t, li, variant = m.groups()
                proc = TASK_OF[t][variant]
                d['name'] = f'{t}/layout_{int(li):02d}/{proc}'
                d['file'] = f'{proc}.hdf5'
        rep.write_text(json.dumps(r, indent=2) + '\n')
    # generation logs are runtime logs, not experiment artifacts
    for f in OUT.glob('gen_*.log'):
        move(f, ROOT / 'logs' / f.name)
        moved.append(ROOT / 'logs' / f.name)

    print(f'moved {len(moved)} files')

    # ---------------- self-check: every name resolves to files ----------------
    missing = []
    def check(name, suffixes):
        for s in suffixes:
            if not (OUT / f'{name}{s}').exists():
                missing.append(f'{name}{s}')
    for task in ('cup', 'obstacle'):
        for leaf in (OUT / task).glob('layout_*'):
            for proc in ('normal', 'gentle') if task == 'cup' else ('narrow', 'wide'):
                check(f'{task}/{leaf.name}/{proc}',
                      ['.hdf5', '.mp4', '_trajectory.csv', '_contacts.json',
                       '_statistics.csv'])
    for exp in ('cross_cup', 'cross_obstacle', 'cross_obstacle_grasped'):
        d = OUT / exp
        if not d.is_dir():
            continue
        for p in d.rglob('*.mp4'):
            rel = p.relative_to(OUT).with_suffix('')
            check(str(rel), ['.mp4', '_trajectory.csv', '_contacts.json'])
    if missing:
        print('MISSING:', *missing, sep='\n  ')
        raise SystemExit(1)
    print('self-check OK: every name resolves to its artifact set')


if __name__ == '__main__':
    main()
