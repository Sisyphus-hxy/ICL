"""Fixed-observation action prediction and path sensitivity for native LIBERO demos."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.distance import cdist

from scene import BPP_ROOT
from inference import load_policy,load_prompt,query_observations

NAMES=('1p5cm','3cm','5cm')


def csv_rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paths(args):
    return dict(zip(NAMES,[p.expanduser().resolve() for p in args.variant_outputs]))


def teacher_forced(args,out):
    """Every prompt receives the identical saved original-rollout sensor history."""
    import subprocess
    subprocess.run([sys.executable,str(BPP_ROOT/'scripts/check_cuda.py')],check=True)
    baseline=args.baseline_output.expanduser().resolve()
    variants=paths(args)
    checkpoint=BPP_ROOT/'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt'
    model,cfg,metadata=load_policy(checkpoint,'cuda:0',args.threads)
    prompt_paths={'original':baseline/'original_replayed_demo.hdf5',
                  **{name:path/'curved_demo.hdf5' for name,path in variants.items()}}
    prompts={name:load_prompt(path,cfg.task.shape_meta,'cuda:0') for name,path in prompt_paths.items()}
    assert len({tuple(p['action'].shape) for p in prompts.values()})==1
    (out/'teacher_forced_config.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint),model_loading=metadata,
        baseline_output=str(baseline),variant_outputs={k:str(v) for k,v in variants.items()},
        prompt_paths={k:str(v) for k,v in prompt_paths.items()},
        fixed_input='original-prompt rollout raw two-frame RGB/EEF/gripper history at transport query steps',
        noise_seed_rule='12345+100*query_seed+query_chunk_index; reset after prompt encoding',
        repeat_control='original A/A and each variant B/B with same inputs and noise'),indent=2)+'\n')
    rows=[];prediction_rows=[]
    def predict(prompt,obs,noise_seed):
        model.reset(action_exec_horizon=12)
        with torch.inference_mode():model.prompt(prompt)
        torch.manual_seed(noise_seed);torch.cuda.manual_seed_all(noise_seed)
        with torch.inference_mode():return model.predict_action(obs)['action'][0].cpu().numpy()
    for seed in range(10):
        pred_npz=np.load(baseline/f'rollout_seed{seed}_original_action_chunks.npz')
        histories=np.load(baseline/f'rollout_seed{seed}_original_query_histories.npz')
        selected=np.flatnonzero(pred_npz['query_transport'])
        query_phase='transport'
        if len(selected)==0:
            # Some successful transfers last only 3–6 control steps, shorter
            # than the 12-step action execution horizon. Use the query boundary
            # immediately before that observed transport interval, and label it.
            trajectory=csv_rows(baseline/f'rollout_seed{seed}_original_trajectory.csv')
            transport_steps=[int(r['step']) for r in trajectory if r['transport']=='True']
            assert transport_steps,f'No transported object in original rollout seed {seed}'
            start=transport_steps[0]
            chunk_starts=pred_npz['chunk_start_steps']
            selected=np.asarray([int(np.searchsorted(chunk_starts,start,side='right')-1)])
            query_phase='transport_entry_boundary'
        if len(selected)>5:
            selected=selected[np.unique(np.linspace(0,len(selected)-1,5).round().astype(int))]
        assert len(selected)>0,f'No original-rollout transport query for seed {seed}'
        for chunk_index in selected:
            # The same two raw frames, proprioception and history are passed to
            # all prompts. query_observations performs the normal upstream input transform.
            history=[{key:histories[key][chunk_index,past].copy() for key in histories.files}
                     for past in range(2)]
            obs=query_observations(history,cfg.task.shape_meta,'cuda:0')
            query_hash=hashlib.sha256(b''.join(np.ascontiguousarray(history[past][key]).tobytes()
                for past in range(2) for key in sorted(history[past]))).hexdigest()
            noise_seed=12345+100*seed+int(chunk_index)
            a=predict(prompts['original'],obs,noise_seed)
            a_repeat=predict(prompts['original'],obs,noise_seed)
            assert a.shape==(16,10)
            stored={'original':a,'original_repeat':a_repeat}
            repeat_noise=float(np.max(np.abs(a-a_repeat)))
            step=int(pred_npz['chunk_start_steps'][chunk_index])
            for name in NAMES:
                b=predict(prompts[name],obs,noise_seed)
                b_repeat=predict(prompts[name],obs,noise_seed)
                stored[name]=b;stored[name+'_repeat']=b_repeat
                diff=b-a
                rows.append(dict(seed=seed,query_step=step,query_chunk_index=int(chunk_index),
                    query_phase=query_phase,
                    variant=name,query_sha256=query_hash,diffusion_seed=noise_seed,
                    mean_action_l2=float(np.linalg.norm(diff,axis=1).mean()),
                    mean_position_action_l2=float(np.linalg.norm(diff[:12,:3],axis=1).mean()),
                    mean_signed_y_action_delta=float(diff[:12,1].mean()),
                    mean_gripper_action_abs_diff=float(np.abs(diff[:12,-1]).mean()),
                    max_abs_action_diff=float(np.abs(diff).max()),
                    original_repeat_max_abs_diff=repeat_noise,
                    variant_repeat_max_abs_diff=float(np.max(np.abs(b-b_repeat)))))
            prediction_rows.append((seed,step,query_hash,stored))
        print('teacher forced seed',seed,'transport query chunks',len(selected),flush=True)
        save_csv(out/'teacher_forced_action_differences.csv',rows)
    payload={'seeds':np.asarray([x[0] for x in prediction_rows]),
             'query_steps':np.asarray([x[1] for x in prediction_rows]),
             'query_sha256':np.asarray([x[2] for x in prediction_rows])}
    for name in ['original','original_repeat']+[x for variant in NAMES for x in [variant,variant+'_repeat']]:
        payload[name]=np.stack([x[3][name] for x in prediction_rows])
    np.savez_compressed(out/'teacher_forced_action_chunks.npz',**payload)
    summary={name:{'queries':sum(r['variant']==name for r in rows),
        'mean_action_l2_median':float(np.median([r['mean_action_l2'] for r in rows if r['variant']==name])),
        'mean_position_action_l2_median':float(np.median([r['mean_position_action_l2'] for r in rows if r['variant']==name])),
        'same_prompt_noise_max':float(max(max(r['original_repeat_max_abs_diff'],r['variant_repeat_max_abs_diff'])
            for r in rows if r['variant']==name))} for name in NAMES}
    (out/'teacher_forced_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def demo_xyz(path):
    r=csv_rows(path)
    return np.asarray([[float(x[k]) for k in ['eef_x','eef_y','eef_z']] for x in r])


def transport_xyz(path):
    r=csv_rows(path)
    return np.asarray([[float(x[k]) for k in ['eef_x','eef_y','eef_z']]
                       for x in r if x['transport']=='True'])


def transport_gripper(path):
    r=csv_rows(path)
    return np.asarray([[float(x['gripper_qpos_0']),float(x['gripper_qpos_1'])]
                       for x in r if x['transport']=='True'])


def gripper_difference(a,b):
    if len(a)<2 or len(b)<2:return None
    grid=np.linspace(0,1,50)
    aa=np.column_stack([np.interp(grid,np.linspace(0,1,len(a)),a[:,i]) for i in range(2)])
    bb=np.column_stack([np.interp(grid,np.linspace(0,1,len(b)),b[:,i]) for i in range(2)])
    return float(np.abs(bb-aa).mean())


def progress_resample(points,n=50):
    distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    distinct=np.r_[True,np.diff(distances)>1e-9]
    if distinct.sum()<2:return None
    return np.column_stack([np.interp(np.linspace(0,distances[-1],n),distances[distinct],points[distinct,i])
                            for i in range(3)])


def path_measures(a,b,normal):
    if len(a)<2 or len(b)<2:return None
    aa=progress_resample(a);bb=progress_resample(b)
    if aa is None or bb is None:return None
    signed=np.einsum('ij,j->i',bb[:,:2]-aa[:,:2],normal)
    d=cdist(a[:,:2],b[:,:2])
    return dict(signed_lateral_mean_m=float(signed.mean()),
        signed_lateral_peak_m=float(signed.max()),
        xy_progress_distance_mean_m=float(np.linalg.norm(bb[:,:2]-aa[:,:2],axis=1).mean()),
        xz_progress_distance_mean_m=float(np.linalg.norm(bb[:,[0,2]]-aa[:,[0,2]],axis=1).mean()),
        xy_symmetric_chamfer_m=float((d.min(axis=0).mean()+d.min(axis=1).mean())/2))


def sensitivity_report(args,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline=args.baseline_output.expanduser().resolve();variants=paths(args)
    original=json.loads((baseline/'rollout_results.json').read_text())
    aresults={r['seed']:r for r in original if r['prompt']=='original'}
    assert len(aresults)==10
    teacher=json.loads((out/'teacher_forced_summary.json').read_text())
    detail=[];summary={}
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    for name,path in variants.items():
        others=json.loads((path/'rollout_results.json').read_text())
        bresults={r['seed']:r for r in others if r['prompt']=='curved'}
        assert len(bresults)==10
        source_demo=demo_xyz(path/'source_demo_trajectory.csv')[82:113]
        curved_demo=demo_xyz(path/'curved_demo_trajectory.csv')[82:113]
        tangent=source_demo[-1,:2]-source_demo[0,:2]
        normal=np.r_[-tangent[1],tangent[0]]/np.linalg.norm(tangent)
        demo=path_measures(source_demo,curved_demo,normal)
        assert demo is not None
        crosses={'both_success':0,'original_only':0,'curved_only':0,'both_failed':0}
        valid_lateral=[]
        for seed in range(10):
            a,b=aresults[seed],bresults[seed]
            assert a['initial_identity']==b['initial_identity']
            crosses['both_success' if a['success'] and b['success'] else
                    'original_only' if a['success'] else
                    'curved_only' if b['success'] else 'both_failed']+=1
            at=transport_xyz(baseline/f'rollout_seed{seed}_original_trajectory.csv')
            bt=transport_xyz(path/f'rollout_seed{seed}_curved_trajectory.csv')
            grip_a=transport_gripper(baseline/f'rollout_seed{seed}_original_trajectory.csv')
            grip_b=transport_gripper(path/f'rollout_seed{seed}_curved_trajectory.csv')
            rollout=path_measures(at,bt,normal)
            agree=(bool(np.sign(rollout['signed_lateral_mean_m'])==
                        np.sign(demo['signed_lateral_mean_m'])) if rollout else None)
            if rollout:valid_lateral.append(rollout['signed_lateral_mean_m'])
            detail.append(dict(variant=name,seed=seed,original_success=a['success'],curved_success=b['success'],
                original_transport_points=len(at),curved_transport_points=len(bt),
                demo_lateral_mean_m=demo['signed_lateral_mean_m'],
                demo_lateral_peak_m=demo['signed_lateral_peak_m'],
                rollout_lateral_mean_m=rollout['signed_lateral_mean_m'] if rollout else None,
                rollout_lateral_peak_m=rollout['signed_lateral_peak_m'] if rollout else None,
                rollout_xy_progress_distance_mean_m=rollout['xy_progress_distance_mean_m'] if rollout else None,
                rollout_xz_progress_distance_mean_m=rollout['xz_progress_distance_mean_m'] if rollout else None,
                rollout_xy_symmetric_chamfer_m=rollout['xy_symmetric_chamfer_m'] if rollout else None,
                rollout_gripper_qpos_mean_abs_diff=gripper_difference(grip_a,grip_b),
                lateral_direction_agrees_with_demo=agree))
            fig,axes=plt.subplots(2,2,figsize=(10,8))
            for row,(ta,tb,title) in enumerate([(source_demo,curved_demo,'Demo transport'),
                (at if len(at)>=2 else demo_xyz(baseline/f'rollout_seed{seed}_original_trajectory.csv'),
                 bt if len(bt)>=2 else demo_xyz(path/f'rollout_seed{seed}_curved_trajectory.csv'),'Rollout transport')]):
                for col,projection in enumerate([[0,1],[0,2]]):
                    for points,color,label in [(ta,'steelblue','Original'),(tb,'darkorange',name)]:
                        axes[row,col].plot(points[:,projection[0]],points[:,projection[1]],color=color,label=label,lw=1.5)
                    axes[row,col].set(xlabel='EEF X (m)',ylabel='EEF Y (m)' if col==0 else 'EEF Z (m)',
                        title=f'{title} {"XY" if col==0 else "XZ"}',aspect='equal')
            axes[0,0].legend();fig.suptitle(f'{name} / query {seed} / success {int(a["success"])}/{int(b["success"])}')
            fig.tight_layout();fig.savefig(figdir/f'{name}_seed{seed}_xy_xz.png',dpi=130);plt.close(fig)
        summary[name]={**crosses,'demo_lateral_mean_m':demo['signed_lateral_mean_m'],
            'demo_lateral_peak_m':demo['signed_lateral_peak_m'],
            'rollout_lateral_mean_median_m':float(np.median(valid_lateral)) if valid_lateral else None,
            'rollout_xy_progress_distance_median_m':float(np.median([
                r['rollout_xy_progress_distance_mean_m'] for r in detail if r['variant']==name
                and r['rollout_xy_progress_distance_mean_m'] is not None])),
            'rollout_gripper_qpos_difference_median':float(np.median([
                r['rollout_gripper_qpos_mean_abs_diff'] for r in detail if r['variant']==name
                and r['rollout_gripper_qpos_mean_abs_diff'] is not None])),
            'transport_pairs_measurable':len(valid_lateral),
            'direction_agreement_count':sum(r['lateral_direction_agrees_with_demo'] is True
                for r in detail if r['variant']==name),
            **teacher[name]}
    save_csv(out/'rollout_path_sensitivity.csv',detail)
    (out/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
