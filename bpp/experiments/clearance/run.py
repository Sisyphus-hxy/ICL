#!/usr/bin/env python3
"""Small, isolated same-goal/different-procedures experiment on installed LIBERO."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=['inspect', 'experts', 'render', 'validate', 'rollouts', 'all', 'analyze', 'audit'])
parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs/latest')
parser.add_argument('--software-render', action='store_true')
parser.add_argument('--no-images', action='store_true', help='Physics tuning only; not valid BPP prompts')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--gpu', default='5', help='CUDA_VISIBLE_DEVICES value or GPU UUID')
parser.add_argument('--egl-device',type=int,help='Verified EGL renderer device index (separate from CUDA visibility)')
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--query-seeds', default='0')
parser.add_argument('--query-start',choices=['initial','transport'],default='initial',
                    help='Optional additional diagnostic: start from the common expert lifted/grasped state')
parser.add_argument('--max-steps', type=int, default=800)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--audit-mode',choices=['official','direct','parity'],default='official')
parser.add_argument('--audit-prompt',type=Path,help='Override only the demonstration in the original-scene direct control')
parser.add_argument('--trials',type=int,default=3)
parser.add_argument('--only', choices=['narrow', 'wide'], help='Expert tuning only; pair is not certified')
args = parser.parse_args()
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')  # upstream generator imports teleoperation helpers
if args.device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.egl_device is not None:
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.egl_device)
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ['MKL_NUM_THREADS'] = str(args.threads)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if args.software_render:
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
    os.environ.setdefault('LP_NUM_THREADS','2')

import numpy as np
import h5py
from scene import Scene, BPP_ROOT, LAYOUTS, OBSTACLE_RADIUS
from expert import generate
from recording import Recorder


def write_results(output, rows):
    with (output / 'results.csv').open('w', newline='') as f:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def experts(output):
    if any(output.glob('expert_*.hdf5')) and any(p.is_symlink() for p in output.glob('expert_*.hdf5')):
        raise RuntimeError('This output directory reuses expert files. Run rollouts or choose a fresh expert directory.')
    scene = Scene(record_images=False)
    scene.canonical_state = scene.env.sim.get_state().flatten().copy()
    np.savez(output / 'initial_state.npz', state=scene.canonical_state, **LAYOUTS)
    (output / 'scene.xml').write_text(scene.model_xml)
    rows, targets = [], {}
    try:
        for layout in ['original', 'relocated']:
            for procedure in ([args.only] if args.only else ['narrow', 'wide']):
                name = f'expert_{layout}_{procedure}'
                print('Generating', name, flush=True)
                rec = Recorder(output, name, save_images=False)
                try:
                    target = generate(scene, rec, procedure, layout)
                except Exception:
                    if rec.actions:
                        rec.finish(scene)
                    raise
                targets[name] = target
                row = rec.summary()
                row.update(kind='expert', procedure=procedure, layout=layout)
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_results(output, rows)
        if not args.no_images and not args.only:
            from rendering import render_experts
            identities=render_experts(output)
            for row in rows:
                row['initial_identity']=identities[row['name']]
            write_results(output,rows)
        relocation = relocation_check(scene, output) if not args.only else {}
        validation = validate_experts(rows, targets, output)
        validation['checks']['old_routes_blocked_after_relocation'] = bool(relocation) and all(
            item['object_min_clearance_m'] < -.001 for item in relocation.values())
        validation['valid'] = all(validation['checks'].values()) and not args.no_images
        validation['relocation_geometry'] = relocation
        (output / 'expert_validation.json').write_text(json.dumps(validation, indent=2))
        print('Expert validation:', json.dumps(validation), flush=True)
        return rows
    finally:
        scene.close()


def relocation_check(scene, output):
    result = {}
    for procedure in ['narrow', 'wide']:
        values = []
        with h5py.File(output/f'expert_original_{procedure}.hdf5','r') as f:
            demo=f['data/demo_0']
            for state,phase in zip(demo['states'],demo['phases']):
                if phase != b'transport':
                    continue
                scene.env.sim.set_state_from_flattened(state)
                scene.d.mocap_pos[scene.mocap_id] = LAYOUTS['relocated']
                scene.env.sim.forward()
                values.append(scene.surface_clearance(scene.object_geoms))
        result[procedure]={'object_min_clearance_m': min(values),
                           'old_path_intersects_relocated_obstacle': min(values)<0}
    (output/'relocation_validation.json').write_text(json.dumps(result,indent=2))
    return result


def validate_experts(rows, targets, output):
    pair = [r for r in rows if r['layout'] == 'original']
    valid = len(rows) == 4 and all(r['goal_success'] and not r['collision'] for r in rows)
    checks = {'four_successful_collision_free_experts': valid}
    if len(pair) == 2:
        narrow, wide = pair
        checks['identical_query_initial_state'] = narrow['initial_identity'] == wide['initial_identity']
        checks['clearance_gap_at_least_1cm'] = wide['object_min_clearance_m']-narrow['object_min_clearance_m'] >= .01
        checks['positive_narrow_clearance'] = narrow['object_min_clearance_m'] > .005
        checks['same_duration'] = narrow['control_steps'] == wide['control_steps']
        checks['transport_path_length_within_10pct'] = abs(narrow['transport_path_length_m']/wide['transport_path_length_m']-1) < .10
        for key in ['grasp_target', 'transport_start', 'transport_end', 'place_target']:
            checks['matched_' + key] = bool(np.allclose(targets[narrow['name']][key], targets[wide['name']][key], atol=.003))
        with h5py.File(output/(narrow['name']+'.hdf5'),'r') as fn, h5py.File(output/(wide['name']+'.hdf5'),'r') as fw:
            dn, dw = fn['data/demo_0'], fw['data/demo_0']
            first_transport = np.where(dn['phases'][:]==b'transport')[0][0]
            checks['matched_pretransport_actions'] = bool(np.allclose(dn['actions'][:first_transport],dw['actions'][:first_transport],atol=1e-4,rtol=0))
            indices = [np.where(d['phases'][:]==b'place')[0][-1] for d in [dn,dw]]
            checks['matched_actual_place_position'] = bool(np.allclose(dn['obs/ee_pos'][indices[0]],dw['obs/ee_pos'][indices[1]],atol=.003))
            from scipy.spatial.transform import Rotation
            rn=Rotation.from_rotvec(dn['obs/ee_ori'][indices[0]])
            rw=Rotation.from_rotvec(dw['obs/ee_ori'][indices[1]])
            checks['matched_actual_place_orientation'] = float((rn.inv()*rw).magnitude()) < .02
            if not args.no_images:
                checks['identical_initial_rgb'] = all(np.array_equal(dn['obs'][key][0],dw['obs'][key][0])
                    for key in ['agentview_rgb','eye_in_hand_rgb'])
    return {'valid': all(checks.values()) and not args.no_images, 'checks': checks,
            'targets': targets, 'images_present': not args.no_images,
            'matching_tolerances': {'pretransport_action_absolute':1e-4,
                                    'actual_place_position_m':.003,'actual_place_rotation_rad':.02},
            'evaluation_regime': 'zero-shot diagnostic'}


def revalidate(output):
    previous=json.loads((output/'expert_validation.json').read_text())
    rows=list(csv.DictReader((output/'results.csv').open()))
    for row in rows:
        for key in ['goal_success','collision']:
            row[key]=row[key]=='True'
        row['control_steps']=int(row['control_steps'])
        for key in ['object_min_clearance_m','transport_path_length_m']:
            row[key]=float(row[key])
        row['initial_identity']=json.loads(row['initial_identity'])
    validation=validate_experts(rows,previous['targets'],output)
    scene=Scene(record_images=False)
    try:
        relocation=relocation_check(scene,output)
    finally:
        scene.close()
    validation['checks']['old_routes_blocked_after_relocation']=all(
        value['object_min_clearance_m']<-.001 for value in relocation.values())
    validation['relocation_geometry']=relocation
    validation['valid']=all(validation['checks'].values()) and not args.no_images
    (output/'expert_validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2),flush=True)


def analyze(output):
    # Recompute from adjacent control samples, including the initial point.
    # Disjoint lifting intervals must never be joined by a fictitious segment.
    from recording import path_metrics
    for filename in ['results.csv','rollout_results.csv']:
        path=output/filename
        if not path.is_file():
            continue
        rows=list(csv.DictReader(path.open()))
        for row in rows:
            data=np.genfromtxt(output/(row['name']+'_trajectory.csv'),delimiter=',',names=True,dtype=None,encoding='utf8')
            with h5py.File(output/(row['name']+'.hdf5'),'r') as f:
                demo=f['data/demo_0']
                initial=demo['initial_ee_pos'][:] if 'initial_ee_pos' in demo else demo['obs/ee_pos'][0]
            pos=np.stack([data[key] for key in ['eef_x','eef_y','eef_z']],axis=-1)
            row.update(path_metrics(pos,data['phase']=='transport',initial))
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, layout in zip(axes, ['original', 'relocated']):
        ax.add_patch(plt.Circle(LAYOUTS[layout][:2], OBSTACLE_RADIUS, color='tomato', alpha=.7))
        for path in sorted(output.glob('*_trajectory.csv')):
            if layout not in path.name:
                continue
            data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf8')
            transport = data['phase'] == 'transport'
            if not transport.any():
                transport = np.ones(len(data), dtype=bool)
            ax.plot(np.where(transport,data['eef_x'],np.nan),np.where(transport,data['eef_y'],np.nan),
                    label=path.stem.removesuffix('_trajectory'))
        ax.set(title=layout, xlabel='EEF world x (m)', ylabel='EEF world y (m)', aspect='equal')
        ax.legend(fontsize=6)
    figure.tight_layout(); figure.savefig(output/'trajectories.png', dpi=160); plt.close(figure)
    figure=plt.figure(figsize=(12,5))
    for i,layout in enumerate(['original','relocated']):
        ax=figure.add_subplot(1,2,i+1,projection='3d')
        for path in sorted(output.glob('*_trajectory.csv')):
            if layout not in path.name:
                continue
            data=np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8')
            ax.plot(data['eef_x'],data['eef_y'],data['eef_z'],label=path.stem.removesuffix('_trajectory'))
        ax.set(title=layout,xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.legend(fontsize=5)
    figure.tight_layout();figure.savefig(output/'trajectories_3d.png',dpi=160);plt.close(figure)
    result_file=output/'rollout_results.csv'
    if result_file.is_file():
        rows=list(csv.DictReader(result_file.open()))
        comparisons=[]
        for layout,seed in sorted({(r['layout'],r['query_seed']) for r in rows}):
            pair={r['procedure']:r for r in rows if r['layout']==layout and r['query_seed']==seed}
            if len(pair)!=2:
                continue
            n,w=pair['narrow'],pair['wide']
            valid=all(r['goal_success']=='True' and r['collision']=='False' and r['object_min_clearance_m']
                      for r in [n,w])
            difference=(float(w['object_min_clearance_m'])-float(n['object_min_clearance_m'])) \
                if n['object_min_clearance_m'] and w['object_min_clearance_m'] else None
            comparisons.append(dict(layout=layout,query_seed=seed,valid_successful_comparison=bool(valid),
                                    wide_minus_narrow_clearance_m=difference,
                                    evaluation_regime='zero-shot diagnostic'))
        if comparisons:
            with (output/'comparisons.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
        summary={'evaluation_regime':'zero-shot diagnostic','episodes':len(rows),
                 'goal_successes':sum(r['goal_success']=='True' for r in rows),
                 'collision_free_episodes':sum(r['collision']=='False' for r in rows),
                 'valid_successful_prompt_comparisons':sum(c['valid_successful_comparison'] for c in comparisons),
                 'preference_conclusion':'inconclusive unless successful collision-free paired transport is observed',
                 'capacity_claim':'No conclusion about the capacity of BPP to learn these preferences.'}
        (output/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))


def main():
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation = {'command':sys.argv, 'utc':datetime.now(timezone.utc).isoformat(),
                  'device':args.device, 'software_render':args.software_render,
                  'seed_for_scene':2, 'new_downloads':0,
                  'evaluation_regime':'zero-shot diagnostic'}
    (output/f'invocation_{args.stage}.json').write_text(json.dumps(invocation,indent=2))
    if args.stage == 'audit':
        from audit import run_audit
        run_audit(args,output)
        return
    if args.stage == 'inspect':
        s=Scene(record_images=not args.no_images)
        print(json.dumps({'source':str(s.source), 'initial':s.initial_identity(),
                          'object_pos':s.object_pos().tolist(), 'goal_pos':s.goal_pos().tolist(),
                          'native_mujoco':True, 'new_downloads':0}, indent=2));s.close();return
    if args.stage in ('experts', 'all'):
        experts(output)
        analyze(output)
    if args.stage in ('rollouts', 'all'):
        from inference import run_rollouts
        run_rollouts(args, output)
        analyze(output)
    if args.stage == 'analyze':
        analyze(output)
    if args.stage == 'validate':
        revalidate(output)
        analyze(output)
    if args.stage == 'render':
        from rendering import render_experts
        render_experts(output)
        revalidate(output)
        analyze(output)


if __name__ == '__main__':
    main()
