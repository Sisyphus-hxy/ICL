#!/usr/bin/env python3
"""Paired original/curved demonstration test in the unmodified LIBERO task."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['analyze','generate','validate','evaluate','report','teacher_forced','sensitivity_report','all'])
parser.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs/latest')
parser.add_argument('--gpu',default='GPU-e0914423-1b44-589b-fdb3-0360e8f79b43')
parser.add_argument('--egl-device',default='0')
parser.add_argument('--max-steps',type=int,default=550)
parser.add_argument('--trials',type=int,default=10)
parser.add_argument('--threads',type=int,default=8)
parser.add_argument('--target-offset-cm',type=float,default=1.5,
                    help='Desired maximum transport EEF XY displacement from source replay')
parser.add_argument('--baseline-output',type=Path,
                    help='Reuse original-prompt rollouts and run only the curved prompt, asserting paired initial states')
parser.add_argument('--capture-chunks',action='store_true',
                    help='Save each full 16-step predicted action chunk and original-query observation histories')
parser.add_argument('--variant-outputs',type=Path,nargs=3,
                    help='1.5 cm, 3 cm, and 5 cm variant output directories for teacher forcing and comparison')
args=parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ['MUJOCO_EGL_DEVICE_ID']=args.egl_device
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['PYNPUT_BACKEND']='dummy'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=str(args.threads)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'clearance'))

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
from scene import BPP_ROOT,SPLIT,TASK,OBJECT,GOAL
from inference import load_policy,load_prompt,query_observations

SOURCE=BPP_ROOT/'datasets/libero'/SPLIT/(TASK+'_demo.hdf5')
BDDL=BPP_ROOT/'src/behavior_prompting/train_network/env/libero/bddl_files'/SPLIT/(TASK+'.bddl')
CHECKPOINT=BPP_ROOT/'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt'
TRANSPORT=(82,113)  # original demo_0; EEF lifted by 82, vertical placement begins at 113
PHASES=((0,50,'approach'),(50,60,'grasp'),(60,82,'lift'),(82,113,'transport'),
        (113,122,'place'),(122,135,'release'))


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def source_demo(index=0):
    with h5py.File(SOURCE,'r') as f:
        d=f[f'data/demo_{index}']
        return d['states'][0].copy(),d['actions'][:].copy(),int(d.attrs.get('seed',2))


def wrapper(images):
    return ControlEnvForRunner(str(BDDL),camera_heights=128,camera_widths=128,
        control_delta=True,use_camera_obs=images,has_offscreen_renderer=images,
        ignore_done=True,hard_reset=False)


def reset(e,state,seed):
    np.random.seed(seed);e.seed(seed);e.reset()
    e.sim.set_state_from_flattened(state);e.sim.forward()
    e.robots[0].controller.update(force=True);e.robots[0].controller.reset_goal()
    e.robots[0].gripper.current_action=np.zeros(2)
    return e._get_observations(force_update=True)


def curve(actions,amplitude):
    out=actions.copy()
    start,end=TRANSPORT
    # A smooth sideward control pulse integrates to a shallow curved transport
    # route. The pulse sums to zero; action count, grasp, and place controls stay.
    t=(np.arange(end-start)+.5)/(end-start)
    out[start:end,1]+=amplitude*np.sin(2*np.pi*t)
    assert np.array_equal(out[:start],actions[:start])
    assert np.array_equal(out[end:],actions[end:])
    return out


def replay(e,state,seed,actions,video_path=None,record=False):
    obs=reset(e,state,seed)
    initial_state=e.sim.get_state().flatten().copy()
    initial_obs={k:obs[k].copy() for k in ['robot0_eef_pos','agentview_image','robot0_eye_in_hand_image'] if k in obs}
    rows=[];observations=[];states=[]
    video=None
    if video_path:
        video=VideoRecorder.create_h264(fps=20,thread_count=1);video.start(str(video_path))
    try:
        for i,a in enumerate(actions):
            states.append(e.sim.get_state().flatten().copy())
            obs,_,_,_=e.step(a)
            if record:
                observations.append({k:obs[k].copy() for k in ['robot0_eef_pos','robot0_eef_quat',
                    'robot0_gripper_qpos','agentview_image','robot0_eye_in_hand_image']})
            if video:
                video.write_frame(np.concatenate([obs['agentview_image'][::-1],
                    obs['robot0_eye_in_hand_image'][::-1]],axis=1).copy())
            bowl=e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
            goal=e.sim.data.body_xpos[e.obj_body_id[GOAL]].copy()
            rows.append(dict(step=i,eef_x=float(obs['robot0_eef_pos'][0]),
                eef_y=float(obs['robot0_eef_pos'][1]),eef_z=float(obs['robot0_eef_pos'][2]),
                bowl_x=float(bowl[0]),bowl_y=float(bowl[1]),bowl_z=float(bowl[2]),
                goal_distance_xy=float(np.linalg.norm(bowl[:2]-goal[:2])),
                goal_success=int(e._check_success())))
    finally:
        if video:video.stop()
    return dict(rows=rows,obs=observations,states=states,
                terminal_state=e.sim.get_state().flatten().copy(),initial_state=initial_state,
                initial_obs=initial_obs,success=bool(e._check_success()))


def write_demo(path,replayed,actions,seed,amplitude,origin):
    with h5py.File(SOURCE,'r') as src,h5py.File(path,'w') as f:
        original=src['data/demo_0']
        demo=f.create_group('data/demo_0')
        demo.attrs['model_file']=original.attrs['model_file']
        demo.attrs['seed']=seed
        demo.attrs['num_samples']=len(actions)
        demo.attrs['observation_alignment']='post_action'
        demo.attrs['state_alignment']='pre_action'
        demo.attrs['origin']=origin
        demo.attrs['transport_start']=TRANSPORT[0]
        demo.attrs['transport_end']=TRANSPORT[1]
        demo.attrs['pulse_amplitude']=amplitude
        demo.create_dataset('states',data=np.asarray(replayed['states']))
        demo.create_dataset('actions',data=actions)
        demo.create_dataset('initial_ee_pos',data=replayed['initial_obs']['robot0_eef_pos'])
        demo.create_dataset('terminal_state',data=replayed['terminal_state'])
        o=demo.create_group('obs')
        mapping={'robot0_eef_pos':'ee_pos','robot0_gripper_qpos':'gripper_states',
                 'agentview_image':'agentview_rgb','robot0_eye_in_hand_image':'eye_in_hand_rgb'}
        for native,key in mapping.items():
            o.create_dataset(key,data=np.asarray([v[native] for v in replayed['obs']]),
                compression='gzip',compression_opts=1)
        ori=Rotation.from_quat(np.asarray([v['robot0_eef_quat'] for v in replayed['obs']])).as_rotvec()
        o.create_dataset('ee_ori',data=ori)


def stage_metrics(actions,positions,phases):
    speed=np.linalg.norm(np.diff(positions,axis=0,prepend=positions[:1]),axis=1)*20
    norm=np.linalg.norm(actions[:,:6],axis=1)
    changes=np.r_[False,actions[1:,6]!=actions[:-1,6]]
    repeat=np.r_[False,np.all(np.isclose(actions[1:],actions[:-1],atol=1e-9,rtol=0),axis=1)]
    rows=[]
    for name,idx in phases:
        a=actions[idx];n=norm[idx];s=speed[idx]
        rows.append(dict(stage=name,steps=len(a),duration_s=len(a)/20,
            action_norm_mean=float(n.mean()),action_norm_p95=float(np.percentile(n,95)),
            action_norm_max=float(n.max()),max_abs_action_component=float(np.abs(a[:,:6]).max()),
            eef_speed_mean_m_s=float(s.mean()),eef_speed_p95_m_s=float(np.percentile(s,95)),
            gripper_switches=int(changes[idx].sum()),
            gripper_close_steps=int((a[:,6]>0).sum()),
            eef_pause_steps=int((s<.005).sum()),
            low_action_steps=int((n<.02).sum()),
            repeated_action_steps=int(repeat[idx].sum())))
    return rows


def analyze(out):
    files=[('source_demo_0',SOURCE),('legacy_narrow',BPP_ROOT/'experiments/clearance/outputs/gpu_aligned/expert_original_narrow.hdf5'),
           ('legacy_wide',BPP_ROOT/'experiments/clearance/outputs/gpu_aligned/expert_original_wide.hdf5')]
    if (out/'original_replayed_demo.hdf5').exists():files.append(('original_replayed',out/'original_replayed_demo.hdf5'))
    if (out/'curved_demo.hdf5').exists():files.append(('curved_demo',out/'curved_demo.hdf5'))
    allrows=[];summary={}
    reference_overall=[]
    with h5py.File(SOURCE,'r') as f:
        lengths=[]
        for i in range(50):
            d=f[f'data/demo_{i}'];a=d['actions'][:];positions=d['obs/ee_pos'][:]
            lengths.append(len(a))
            row=stage_metrics(a,positions,[('all',np.arange(len(a)))])[0]
            row['demo_index']=i
            reference_overall.append(row)
    save_csv(out/'source_50_overall_statistics.csv',reference_overall)
    summary['source_50_lengths']={'min':min(lengths),'median':float(np.median(lengths)),
        'max':max(lengths),'demo_0':lengths[0]}
    summary['source_50_overall_medians']={key:float(np.median([r[key] for r in reference_overall]))
        for key in ['steps','action_norm_mean','action_norm_max','eef_speed_mean_m_s',
                    'eef_pause_steps','repeated_action_steps','gripper_switches']}
    for label,path in files:
        with h5py.File(path,'r') as f:
            d=f['data/demo_0'];actions=d['actions'][:];pos=d['obs/ee_pos'][:]
            if 'phases' in d: phases=[(x.decode(),np.where(d['phases'][:]==x)[0]) for x in dict.fromkeys(d['phases'][:])]
            else:phases=[(name,np.arange(start,end)) for start,end,name in PHASES]
            records=stage_metrics(actions,pos,phases)
            records+=stage_metrics(actions,pos,[('all',np.arange(len(actions)))])
            for record in records:record['demo']=label;allrows.append(record)
            summary[label]={'steps':len(actions),'prompt_chunks':int(np.ceil(len(actions)/20)),
                'prompt_padding_actions':int((-len(actions))%20),
                'gripper_switch_indices':list(map(int,np.where(np.diff(actions[:,6])!=0)[0]+1)),
                'stage_boundaries_inferred_for_source':label=='source_demo_0',
                'summary_by_stage':records}
    save_csv(out/'stage_statistics.csv',allrows)
    (out/'stage_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def generate(out):
    state,original,seed=source_demo(0)
    w=wrapper(False)
    try:
        baseline=replay(w.env,state,seed,original)
        assert baseline['success'],'Original source actions must replay successfully.'
        bpos=np.array([[r[k] for k in ['eef_x','eef_y','eef_z']] for r in baseline['rows']])
        candidates=[];chosen=None
        target=args.target_offset_cm/100
        amplitude_guesses=([.12,.20,.30,.40,.55] if target<.02 else
                           [target*8.,target*8.*.9,target*8.*1.1,target*8.*.8,target*8.*1.2])
        for amplitude in amplitude_guesses:
            actions=curve(original,amplitude)
            candidate=replay(w.env,state,seed,actions)
            cpos=np.array([[r[k] for k in ['eef_x','eef_y','eef_z']] for r in candidate['rows']])
            start,end=TRANSPORT
            offset=np.linalg.norm(cpos[start:end,:2]-bpos[start:end,:2],axis=1)
            result=dict(amplitude=amplitude,success=candidate['success'],
                transport_max_eef_xy_difference_m=float(offset.max()),
                final_eef_distance_m=float(np.linalg.norm(cpos[-1]-bpos[-1])))
            candidates.append(result)
            tolerance=.008 if target>=.04 else .006
            if result['success'] and abs(result['transport_max_eef_xy_difference_m']-target)<=tolerance and result['final_eef_distance_m']<.015:
                chosen=(actions,amplitude,result);break
    finally:w.close()
    (out/'candidate_trials.json').write_text(json.dumps(candidates,indent=2)+'\n')
    if chosen is None:raise RuntimeError('No shallow successful transport curve found; inspect candidate_trials.json.')
    actions,amplitude,metrics=chosen
    w=wrapper(True)
    try:
        source_replay=replay(w.env,state,seed,original,out/'source_demo.mp4',record=True)
        modified=replay(w.env,state,seed,actions,out/'curved_demo.mp4',record=True)
        check=replay(w.env,state,seed,actions,record=True)
    finally:w.close()
    assert source_replay['success'] and modified['success'] and check['success']
    assert np.array_equal(modified['initial_state'],source_replay['initial_state'])
    for a,b in zip(modified['obs'],check['obs']):
        assert np.array_equal(a['robot0_eef_pos'],b['robot0_eef_pos'])
        assert np.array_equal(a['agentview_image'],b['agentview_image'])
    assert len(actions)==len(original)==135
    for a,b in zip(source_replay['obs'][:TRANSPORT[0]],modified['obs'][:TRANSPORT[0]]):
        for key in a:assert np.array_equal(a[key],b[key]),('pretransport observation differs',key)
    write_demo(out/'original_replayed_demo.hdf5',source_replay,original,seed,0.,
               'original demo_0 actions replayed with native observations')
    write_demo(out/'curved_demo.hdf5',modified,actions,seed,amplitude,
               'original demo_0 actions with transport-only sideward pulse, natively replayed')
    save_csv(out/'source_demo_trajectory.csv',source_replay['rows'])
    save_csv(out/'curved_demo_trajectory.csv',modified['rows'])
    (out/'generation_validation.json').write_text(json.dumps(dict(
        original_replay_success=source_replay['success'],curved_replay_success=modified['success'],
        second_curved_replay_success=check['success'],exact_observation_replay=True,
        identical_initial_sim_state=True,same_action_count=True,
        unchanged_nontransport_actions=True,exact_pretransport_observations=True,
        transport_window=list(TRANSPORT),
        selected_curve=metrics),indent=2)+'\n')
    analyze(out)


def validate_one_hdf5(path):
    with h5py.File(path,'r') as f:
        d=f['data/demo_0'];actions=d['actions'][:];states=d['states'][:]
        assert d.attrs['observation_alignment']=='post_action'
        assert d.attrs['state_alignment']=='pre_action'
        seed=int(d.attrs['seed'])
        expected={key:d['obs'][key][:] for key in ['ee_pos','ee_ori','gripper_states',
                                                   'agentview_rgb','eye_in_hand_rgb']}
        terminal=d['terminal_state'][:]
    w=wrapper(True)
    try:
        rec=replay(w.env,states[0],seed,actions,record=True)
    finally:w.close()
    observed={
        'ee_pos':np.asarray([o['robot0_eef_pos'] for o in rec['obs']]),
        'ee_ori':Rotation.from_quat(np.asarray([o['robot0_eef_quat'] for o in rec['obs']])).as_rotvec(),
        'gripper_states':np.asarray([o['robot0_gripper_qpos'] for o in rec['obs']]),
        'agentview_rgb':np.asarray([o['agentview_image'] for o in rec['obs']]),
        'eye_in_hand_rgb':np.asarray([o['robot0_eye_in_hand_image'] for o in rec['obs']])}
    difference={key:float(np.max(np.abs(observed[key].astype(np.float64)-value.astype(np.float64))))
                for key,value in expected.items()}
    difference['states']=float(np.max(np.abs(np.asarray(rec['states'])-states)))
    difference['terminal_state']=float(np.max(np.abs(rec['terminal_state']-terminal)))
    result={'replay_success':rec['success'],'max_difference':difference,
            'all_exact':all(x==0 for x in difference.values())}
    assert rec['success'] and result['all_exact'],result
    return result


def validate_hdf5(out):
    results={name:validate_one_hdf5(out/(name+'.hdf5'))
             for name in ['original_replayed_demo','curved_demo']}
    with h5py.File(out/'original_replayed_demo.hdf5') as fa,h5py.File(out/'curved_demo.hdf5') as fb:
        a,b=fa['data/demo_0'],fb['data/demo_0']
        assert np.array_equal(a['actions'][:TRANSPORT[0]],b['actions'][:TRANSPORT[0]])
        assert np.array_equal(a['actions'][TRANSPORT[1]:],b['actions'][TRANSPORT[1]:])
        for key in ['ee_pos','ee_ori','gripper_states','agentview_rgb','eye_in_hand_rgb']:
            assert np.array_equal(a['obs'][key][:TRANSPORT[0]],b['obs'][key][:TRANSPORT[0]]),(key,'pretransport differs')
    results['exact_pretransport_pairing']=True
    (out/'hdf5_replay_validation.json').write_text(json.dumps(results,indent=2)+'\n')


def evaluate(out):
    assert json.loads((out/'generation_validation.json').read_text())['curved_replay_success']
    assert args.trials==10,'Paired evaluation requires the requested 10 query states.'
    import subprocess
    subprocess.run([sys.executable,str(BPP_ROOT/'scripts/check_cuda.py')],check=True)
    device='cuda:0'
    model,cfg,metadata=load_policy(CHECKPOINT,device,args.threads)
    prompts={'original':load_prompt(out/'original_replayed_demo.hdf5',cfg.task.shape_meta,device),
             'curved':load_prompt(out/'curved_demo.hdf5',cfg.task.shape_meta,device)}
    pn=model.normalizer.get_prompt_normalizer()
    prompt_diagnostics={}
    for label,prompt in prompts.items():
        normalized=pn['action'].normalize(prompt['action'])
        prompt_diagnostics[label]={
            'action_shape':list(prompt['action'].shape),
            'prompt_chunks':int(prompt['action'].shape[1]),
            'input_action_abs_max':float(prompt['action'].abs().max()),
            'normalized_action_abs_max':float(normalized.abs().max()),
            'normalized_action_abs_p95':float(torch.quantile(normalized.abs().flatten(),.95)),
            'all_finite':bool(torch.isfinite(normalized).all()),
            'mask_count':int(prompt['metadata']['mask'].sum())}
    assert prompt_diagnostics['original']['action_shape']==prompt_diagnostics['curved']['action_shape']
    assert all(v['all_finite'] for v in prompt_diagnostics.values())
    (out/'prompt_diagnostics.json').write_text(json.dumps(prompt_diagnostics,indent=2)+'\n')
    inverse=RotationTransformer('axis_angle','rotation_6d')
    configs={'checkpoint':str(CHECKPOINT),'checkpoint_load':metadata,
             'prompt_paths':{'original':str(out/'original_replayed_demo.hdf5'),'curved':str(out/'curved_demo.hdf5')},
             'max_steps':args.max_steps,'paired_query_state_indices':list(range(args.trials)),
             'diffusion_seeds':list(range(args.trials)),'warmup_open_steps':12,
             'prompt_chunk_n_actions':int(cfg.task.shape_meta.prompt_chunk_n_actions),
             'prompt_sequence_length':int(cfg.task.shape_meta.prompt_sequence_length),
             'max_sequence_length':int(cfg.task.shape_meta.max_sequence_length),
             'pad_end_prompt_actions':str(cfg.task.shape_meta.pad_end_prompt_actions)}
    configs['max_prompt_chunks_from_max_sequence_length']=int(math.ceil(
        configs['max_sequence_length']/configs['prompt_chunk_n_actions']))
    configs['only_prompt_sampler_uses_full_task_without_prompt_sequence_length_trim']=True
    configs['prompt_source']='both demonstrations natively replayed through identical HDF5 capture; original task cache is not rebuilt or overwritten'
    (out/'evaluation_config.json').write_text(json.dumps(configs,indent=2)+'\n')
    baseline={}
    if args.baseline_output:
        original_rows=json.loads((args.baseline_output/'rollout_results.json').read_text())
        baseline={r['seed']:r for r in original_rows if r['prompt']=='original'}
        assert len(baseline)==args.trials
    w=wrapper(True);e=w.env;results=[]
    try:
        for seed_index in range(args.trials):
            state,_,reset_seed=source_demo(seed_index)
            paired_identity=None
            for label in (['curved'] if baseline else ['original','curved']):
                obs=reset(e,state,reset_seed);history=[obs]
                for _ in range(12):
                    obs,_,_,_=e.step(np.r_[np.zeros(6),-1.]);history.append(obs);history=history[-2:]
                identity={'sim_state_sha256':hashlib.sha256(e.sim.get_state().flatten().tobytes()).hexdigest(),
                          'agentview_rgb_sha256':hashlib.sha256(obs['agentview_image'].tobytes()).hexdigest(),
                          'eef_pos':obs['robot0_eef_pos'].tolist(),
                          'gripper_command':e.robots[0].gripper.current_action.tolist(),
                          'controller_goal_pos':e.robots[0].controller.goal_pos.tolist()}
                if paired_identity is None:paired_identity=identity
                assert identity==paired_identity,'Paired query initial state drifted.'
                if baseline:assert identity==baseline[seed_index]['initial_identity'], 'Reused baseline query state drifted.'
                model.reset(action_exec_horizon=12)
                with torch.inference_mode():model.prompt(prompts[label])
                torch.manual_seed(seed_index);torch.cuda.manual_seed_all(seed_index)
                name=f'rollout_seed{seed_index}_{label}'
                video=VideoRecorder.create_h264(fps=20,thread_count=1)
                video.start(str(out/(name+'.mp4')))
                rows=[];initial_bowl=e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
                goal=e.sim.data.body_xpos[e.obj_body_id[GOAL]].copy()
                min_goal=float(np.linalg.norm(initial_bowl[:2]-goal[:2]))
                chunk_predictions=[];chunk_starts=[];chunk_transport=[];chunk_histories=[]
                try:
                    while len(rows)<args.max_steps:
                        obs_dict=query_observations(history,cfg.task.shape_meta,device)
                        bowl_before=e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
                        goal_before=e.sim.data.body_xpos[e.obj_body_id[GOAL]].copy()
                        transport_now=bool(bowl_before[2]>initial_bowl[2]+.045
                            and np.linalg.norm(bowl_before[:2]-initial_bowl[:2])>.02
                            and np.linalg.norm(bowl_before[:2]-goal_before[:2])>.055)
                        if args.capture_chunks and label=='original':
                            past=history[-2:]
                            chunk_histories.append({k:np.stack([o[k].copy() for o in past]) for k in [
                                'agentview_image','robot0_eye_in_hand_image','robot0_eef_pos',
                                'robot0_eef_quat','robot0_gripper_qpos']})
                        with torch.inference_mode():prediction=model.predict_action(obs_dict)['action'][0].cpu().numpy()
                        if args.capture_chunks:
                            chunk_predictions.append(prediction.copy());chunk_starts.append(len(rows))
                            chunk_transport.append(transport_now)
                        pred=prediction[:12]
                        actions=np.concatenate([pred[:,:3],inverse.inverse(pred[:,3:9]),pred[:,9:]],axis=-1)
                        for a in actions:
                            video.write_frame(np.concatenate([history[-1]['agentview_image'][::-1],
                                history[-1]['robot0_eye_in_hand_image'][::-1]],axis=1).copy())
                            obs,_,_,_=e.step(a);history.append(obs);history=history[-2:]
                            bowl=e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
                            goal_dist=float(np.linalg.norm(bowl[:2]-goal[:2]))
                            min_goal=min(min_goal,goal_dist)
                            rows.append(dict(step=len(rows),eef_x=float(obs['robot0_eef_pos'][0]),
                                eef_y=float(obs['robot0_eef_pos'][1]),eef_z=float(obs['robot0_eef_pos'][2]),
                                gripper_qpos_0=float(obs['robot0_gripper_qpos'][0]),
                                gripper_qpos_1=float(obs['robot0_gripper_qpos'][1]),
                                predicted_gripper_command=float(a[-1]),
                                transport=bool(bowl[2]>initial_bowl[2]+.045
                                    and np.linalg.norm(bowl[:2]-initial_bowl[:2])>.02
                                    and goal_dist>.055),
                                bowl_x=float(bowl[0]),bowl_y=float(bowl[1]),bowl_z=float(bowl[2]),
                                goal_distance_xy=goal_dist,goal_success=int(e._check_success())))
                            if rows[-1]['goal_success'] or len(rows)>=args.max_steps:break
                        if rows[-1]['goal_success']:break
                finally:video.stop()
                if args.capture_chunks:
                    np.savez_compressed(out/(name+'_action_chunks.npz'),
                        predicted_action_chunks=np.asarray(chunk_predictions),
                        chunk_start_steps=np.asarray(chunk_starts,dtype=int),
                        query_transport=np.asarray(chunk_transport,dtype=bool))
                    if label=='original':
                        np.savez_compressed(out/(name+'_query_histories.npz'),
                            **{key:np.stack([h[key] for h in chunk_histories]) for key in chunk_histories[0]})
                save_csv(out/(name+'_trajectory.csv'),rows)
                bowl=np.array([[r[k] for k in ['bowl_x','bowl_y','bowl_z']] for r in rows])
                lifted=bool((bowl[:,2]>initial_bowl[2]+.045).any())
                transported=bool((np.linalg.norm(bowl[:,:2]-initial_bowl[:2],axis=1)>.035).any())
                near_goal=bool(min_goal<.055)
                result=dict(seed=seed_index,source_initial_state_index=seed_index,source_reset_seed=reset_seed,
                    prompt=label,success=bool(rows[-1]['goal_success']),steps=len(rows),
                    lifted=lifted,transported=transported,near_goal=near_goal,
                    min_goal_distance_xy_m=min_goal,initial_identity=identity)
                results.append(result);print(json.dumps(result),flush=True)
                (out/'rollout_results.json').write_text(json.dumps(results,indent=2)+'\n')
                save_csv(out/'rollout_results.csv',results)
    finally:w.close()
    assert len(results)==(10 if baseline else 20)
    combined=list(baseline.values())+results
    (out/'paired_summary.json').write_text(json.dumps({label:{k:sum(r[k] for r in combined if r['prompt']==label)
        for k in ['success','lifted','transported','near_goal']} for label in ['original','curved']},indent=2)+'\n')


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((out/'rollout_results.json').read_text())
    assert len(results)==20
    bypair={(r['seed'],r['prompt']):r for r in results}
    comparisons=[]
    def failed_stage(row):
        for stage in ['lifted','transported','near_goal']:
            if not row[stage]:return stage
        return 'none' if row['success'] else 'goal_placement'
    fig,axes=plt.subplots(2,5,figsize=(19,8))
    for seed,ax in enumerate(axes.flat):
        left=bypair[(seed,'original')]
        right=bypair[(seed,'curved')]
        assert left['initial_identity']==right['initial_identity']
        comparisons.append(dict(seed=seed,original_success=left['success'],curved_success=right['success'],
            original_lifted=left['lifted'],curved_lifted=right['lifted'],
            original_transported=left['transported'],curved_transported=right['transported'],
            original_near_goal=left['near_goal'],curved_near_goal=right['near_goal'],
            original_first_failed_stage=failed_stage(left),curved_first_failed_stage=failed_stage(right),
            same_query_initial_state=True))
        for label,color in [('original','steelblue'),('curved','darkorange')]:
            rows=list(csv.DictReader((out/f'rollout_seed{seed}_{label}_trajectory.csv').open()))
            ax.plot([float(r['eef_x']) for r in rows],[float(r['eef_y']) for r in rows],
                color=color,label=label,lw=1,alpha=.85)
        ax.set(title=f'seed {seed}: {int(left["success"])}/{int(right["success"])}',
            xlabel='EEF x (m)',ylabel='EEF y (m)',aspect='equal')
    axes.flat[0].legend()
    fig.tight_layout();fig.savefig(out/'paired_eef_trajectories.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for label,color in [('source_demo','steelblue'),('curved_demo','darkorange')]:
        rows=list(csv.DictReader((out/f'{label}_trajectory.csv').open()))
        transport=rows[TRANSPORT[0]:TRANSPORT[1]]
        ax.plot([float(r['eef_x']) for r in transport],
                [float(r['eef_y']) for r in transport],label=label,color=color,lw=2)
    ax.set(xlabel='EEF x (m)',ylabel='EEF y (m)',aspect='equal',
           title='Native replay: transport window')
    ax.legend();fig.tight_layout();fig.savefig(out/'demo_eef_transport.png',dpi=180);plt.close(fig)
    save_csv(out/'paired_results.csv',comparisons)
    (out/'paired_results.json').write_text(json.dumps(comparisons,indent=2)+'\n')


def main():
    out=args.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if args.stage in ['analyze','all']:analyze(out)
    if args.stage in ['generate','all']:generate(out)
    if args.stage in ['validate','all']:validate_hdf5(out)
    if args.stage in ['evaluate','all']:evaluate(out)
    if args.stage in ['report','all']:report(out)
    if args.stage in ['teacher_forced','sensitivity_report']:
        assert args.baseline_output and args.variant_outputs and len(args.variant_outputs)==3
        from sensitivity import teacher_forced, sensitivity_report
        if args.stage=='teacher_forced':teacher_forced(args,out)
        else:sensitivity_report(args,out)

if __name__=='__main__':main()
