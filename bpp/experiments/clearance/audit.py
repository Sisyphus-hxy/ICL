"""Original-scene positive controls and parity with the official prompt loader."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scene import BPP_ROOT, SPLIT, TASK
from inference import load_policy, load_prompt, query_observations


def config():
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    with initialize_config_dir(config_dir=str(BPP_ROOT/'src/behavior_prompting/train_network/config'),version_base=None):
        return compose(config_name='libero_policy_dunetp',overrides=[
            'task=liberogen_spatial_combination','+modifiers=libero/liberogen_spatial_combination'])


def source_path():
    return BPP_ROOT/'datasets/libero'/SPLIT/(TASK+'_demo.hdf5')


def official(args,output):
    workdir=BPP_ROOT/'src/behavior_prompting/train_network'
    checkpoint=args.checkpoint or BPP_ROOT/'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt'
    command=[sys.executable,'train.py','--config-name=libero_policy_dunetp',
             'task=liberogen_spatial_combination','+modifiers=libero/liberogen_spatial_combination',
             '+rollout=libero',f'rollout.checkpoint_path={checkpoint}',
             'logging.mode=offline','training.seed=0','exp_name=original_task_baseline',
             f'hydra.run.dir={output / "official_run"}',
             f'task.env_runner.cache_dir={BPP_ROOT / ".cache/rollout/liberogen_spatial_combination"}',
             'task.env_runner.eval_seen=false','task.env_runner.eval_unseen=true',
             'task.unseen_rollout_max_tasks=1',f'task.unseen_include_file_filters=[{TASK}_demo]',
             'task.env_runner.n_envs=1','task.env_runner.n_train=0',
             f'task.env_runner.n_test={args.trials}',f'task.env_runner.unseen_max_n={args.trials}',
             'task.env_runner.n_train_vis=0','task.env_runner.n_test_vis=1',
             f'task.env_runner.max_steps={args.max_steps}','rollout.enable_expensive_vis=false']
    (output/'official_command.json').write_text(json.dumps({'command':command,'cwd':str(workdir),
        'scene':'unmodified original LIBERO task','prompt':'official dataset sampler',
        'checkpoint':str(checkpoint),'trials':args.trials},indent=2))
    subprocess.run([sys.executable,str(BPP_ROOT/'scripts/check_cuda.py')],check=True)
    with (output/'official.log').open('w') as log:
        result=subprocess.run(command,cwd=workdir,stdout=log,stderr=subprocess.STDOUT)
    print('Official evaluator exit code:',result.returncode,flush=True)
    if result.returncode:
        print((output/'official.log').read_text()[-7000:],flush=True)
        raise RuntimeError('Official baseline execution failed; inspect official.log')


def parity(output):
    from behavior_prompting.train_network.dataset.libero_replay_image_dataset import LiberoReplayImageDataset
    from behavior_prompting.train_network.utils.prompt_util import collate_prompts
    cfg=config()
    dataset=LiberoReplayImageDataset(shape_meta=cfg.task.shape_meta,dataset_splits=[SPLIT],
        dataset_name=SPLIT,use_cache=True,cache_dir=str(BPP_ROOT/'.cache/rollout/liberogen_spatial_combination'),
        seed=0,val_ratio=0,sample_type='task',only_prompt=True,max_segments=-1,
        name_suffix=TASK+'_demo',include_file_filters=[TASK+'_demo'])
    official_prompt=collate_prompts([dataset[0]])['obs']['prompt']
    assert int(official_prompt['metadata']['episode_indices'][0,0])==0, 'First official prompt is not demo_0'
    explicit=load_prompt(source_path(),cfg.task.shape_meta,'cpu')
    differences={}
    for key in official_prompt['obs']:
        a=official_prompt['obs'][key];b=explicit['obs'][key]
        assert a.shape==b.shape,(key,a.shape,b.shape)
        differences[key]=float((a-b).abs().max())
    differences['action']=float((official_prompt['action']-explicit['action']).abs().max())
    differences['mask']=int(torch.logical_xor(official_prompt['metadata']['mask'],explicit['metadata']['mask']).sum())
    result={'matches':all(value<1e-6 for value in differences.values()),'maximum_differences':differences,
            'official_prompt_shape':list(official_prompt['action'].shape)}
    (output/'prompt_parity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    assert result['matches'],'Explicit prompt differs from official loader'


def direct(args,output):
    """Original BDDL and original states: no extra obstacle or object translations."""
    from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
    from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
    from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
    device='cuda:0' if args.device=='cuda' else 'cpu'
    checkpoint=args.checkpoint or BPP_ROOT/'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt'
    subprocess.run([sys.executable,str(BPP_ROOT/'scripts/check_cuda.py')],check=True)
    model,cfg,metadata=load_policy(checkpoint,device,args.threads)
    prompt_path=args.audit_prompt or source_path()
    prompt=load_prompt(prompt_path,cfg.task.shape_meta,device)
    (output/'direct_config.json').write_text(json.dumps({'scene':'unmodified original task',
        'prompt':str(prompt_path),'checkpoint':str(checkpoint),'warmup_open_steps':12,
        'trials':args.trials,'max_steps':args.max_steps,'model_loading':metadata},indent=2))
    inverse=RotationTransformer('axis_angle','rotation_6d')
    bddl=BPP_ROOT/'src/behavior_prompting/train_network/env/libero/bddl_files'/SPLIT/(TASK+'.bddl')
    wrapper=ControlEnvForRunner(str(bddl),camera_heights=128,camera_widths=128,
                               control_delta=True,ignore_done=True,hard_reset=False)
    env=wrapper.env
    results=[]
    try:
        for trial in range(args.trials):
            with h5py.File(source_path(),'r') as f:
                demo=f[f'data/demo_{trial}'];state=demo['states'][0].copy();seed=int(demo.attrs.get('seed',2))
            np.random.seed(seed);env.seed(seed);env.reset()
            env.sim.set_state_from_flattened(state);env.sim.forward()
            env.robots[0].controller.update(force=True);env.robots[0].controller.reset_goal()
            env.robots[0].gripper.current_action=np.zeros(2)
            history=[env._get_observations(force_update=True)]
            # Same initial open-gripper warmup as the official runner.
            for _ in range(12):
                obs,_,_,_=env.step(np.r_[np.zeros(6),-1.]);history.append(obs);history=history[-2:]
            model.reset(action_exec_horizon=12)
            with torch.inference_mode():model.prompt(prompt)
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            video=VideoRecorder.create_h264(fps=20,thread_count=1)
            video.start(str(output/f'direct_trial{trial}.mp4'))
            rows=[]
            try:
                while len(rows)<args.max_steps:
                    obs_dict=query_observations(history,cfg.task.shape_meta,device)
                    with torch.inference_mode():pred=model.predict_action(obs_dict)['action'][0,:12].cpu().numpy()
                    action=np.concatenate([pred[:,:3],inverse.inverse(pred[:,3:9]),pred[:,9:]],axis=-1)
                    for a in action:
                        before=history[-1]
                        video.write_frame(np.concatenate([before['agentview_image'][::-1],before['robot0_eye_in_hand_image'][::-1]],axis=1).copy())
                        obs,_,_,_=env.step(a);history.append(obs);history=history[-2:]
                        rows.append({'step':len(rows),'success':bool(env._check_success()),
                                     **{key:float(value) for key,value in zip(['eef_x','eef_y','eef_z'],obs['robot0_eef_pos'])}})
                        if rows[-1]['success'] or len(rows)>=args.max_steps:break
                    if rows[-1]['success']:break
            finally:video.stop()
            with (output/f'direct_trial{trial}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            results.append({'trial':trial,'query_demo_initial_state':trial,'prompt_demo':0,'success':rows[-1]['success'],'steps':len(rows)})
            print(json.dumps(results[-1]),flush=True)
            (output/'direct_results.json').write_text(json.dumps(results,indent=2))
    finally:wrapper.close()


def run_audit(args,output):
    if args.audit_mode=='official':official(args,output)
    elif args.audit_mode=='parity':parity(output)
    else:direct(args,output)
