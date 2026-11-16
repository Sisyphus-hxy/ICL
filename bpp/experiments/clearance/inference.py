"""Direct explicit behavior prompts through the unmodified BPP policy interface."""
import csv
import hashlib
import json
import random
import time

import dill
import h5py
import hydra
from hydra import compose, initialize_config_dir
import numpy as np
import torch
from omegaconf import OmegaConf

from behavior_prompting.train_network.dataset.libero_replay_image_dataset import (
    _convert_actions, _receding_rgb_numpy_thwc_to_float_chw,
)
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker
from recording import Recorder
from scene import BPP_ROOT, Scene


def load_policy(checkpoint, device, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    payload = torch.load(str(checkpoint), map_location='cpu', mmap=True,
                         pickle_module=dill, weights_only=False)
    saved_cfg = payload['cfg']
    task = str(saved_cfg.task.name)
    if task not in ['libero_all','liberogen_spatial_combination']:
        raise ValueError(f'Unsupported BPP checkpoint task: {task}')
    # Use the same released configuration as the existing run_eval.sh launcher.
    # This release's serialized cfg omits a flag whose constructor default enables
    # a positional embedding absent from its state dict. The official config
    # disables that embedding; strict loading verifies every remaining weight.
    with initialize_config_dir(config_dir=str(BPP_ROOT/'src/behavior_prompting/train_network/config'),version_base=None):
        cfg = compose(config_name='libero_policy_dunetp',overrides=[
            f'task={task}',f'+modifiers=libero/{task}'])
    def flatten(d,prefix=''):
        result={}
        for key,value in d.items():
            if isinstance(value,dict):
                result.update(flatten(value,prefix+key+'.'))
            else:
                result[prefix+key]=value
        return result
    old=flatten(OmegaConf.to_container(saved_cfg.model,resolve=True))
    new=flatten(OmegaConf.to_container(cfg.model,resolve=True))
    differences=[key for key in set(old)|set(new) if old.get(key)!=new.get(key)
                 and key!='obs_encoder.obs_encoder.use_pool_modality_pos_embed']
    if differences:
        raise ValueError(f'Checkpoint has unverified model configuration differences: {sorted(differences)}')
    # Keep the original architecture/initialization path. The upstream custom
    # non-pretrained initializer does not support this ViT's Conv2d patch stem.
    # run.sh enables HF offline mode, so the already cached CLIP initialization
    # is reused and no network requests or downloads can occur.
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(payload['state_dicts']['model'], strict=True)
    model.eval().to(device)
    metadata={'configuration_source':'pinned official libero_policy_dunetp + task/modifier (same as run_eval.sh)',
              'saved_pool_modality_pos_embed':saved_cfg.model.obs_encoder.obs_encoder.get('use_pool_modality_pos_embed',None),
              'constructor_default_for_missing_saved_flag':True,
              'effective_pool_modality_pos_embed':bool(cfg.model.obs_encoder.obs_encoder.use_pool_modality_pos_embed),
              'other_resolved_model_configuration_differences':differences,
              'state_dict_load':'strict; no missing or unexpected keys'}
    return model, cfg, metadata


def load_prompt(path, shape_meta, device):
    """No filename, label, HDF5 attribute, task text or procedure ID enters BPP."""
    rot = RotationTransformer('axis_angle', 'rotation_6d')
    chunk = int(shape_meta['prompt_chunk_n_actions'])
    with h5py.File(path, 'r') as f:
        d = f['data/demo_0']
        if 'initial_identity' in d.attrs and d.attrs.get('observation_alignment') != 'post_action':
            raise ValueError('Legacy experiment prompt has pre-action observations; regenerate experts before inference.')
        obs = {key: d['obs'][key][::chunk].astype(np.float32)
               for key in ['ee_pos', 'ee_ori', 'gripper_states']}
        obs['ee_ori'] = rot.forward(obs['ee_ori'])
        for key in ['agentview_rgb', 'eye_in_hand_rgb']:
            obs[key] = _receding_rgb_numpy_thwc_to_float_chw(d['obs'][key][::chunk])
        action = _convert_actions(d['actions'][:].astype(np.float32), rot)
    raw = {'obs': {key: value[None] for key, value in obs.items()},
           'action': action[None], 'metadata': {}}
    prompt = PromptActionChunker(shape_meta).chunk_prompt(raw, obs_predownsampled=True)
    prompt['metadata']['mask'] = np.zeros(prompt['action'].shape[:2], dtype=bool)
    def transfer(value):
        if isinstance(value, dict):
            return {key: transfer(item) for key, item in value.items()}
        return torch.from_numpy(np.ascontiguousarray(value)).to(device)
    return transfer(prompt)


def query_observations(history, shape_meta, device):
    rot = RotationTransformer('quaternion', 'rotation_6d')
    mapping = {'ee_pos':'robot0_eef_pos', 'ee_ori':'robot0_eef_quat',
               'gripper_states':'robot0_gripper_qpos', 'agentview_rgb':'agentview_image',
               'eye_in_hand_rgb':'robot0_eye_in_hand_image'}
    result = {}
    for key, sim_key in mapping.items():
        n = int(shape_meta['obs'][key]['horizon'])
        values = [history[max(len(history)-n+i, 0)][sim_key] for i in range(n)]
        value = np.asarray(values)
        if key.endswith('_rgb'):
            value = _receding_rgb_numpy_thwc_to_float_chw(value)
        elif key == 'ee_ori':
            value = rot.forward(value[..., [3,0,1,2]].astype(np.float32))
        result[key] = torch.as_tensor(value[None].copy(), dtype=torch.float32, device=device)
    return result


def run_rollouts(args, output):
    validation = json.loads((output/'expert_validation.json').read_text())
    if not validation['valid']:
        raise RuntimeError('Expert pair has not passed validation; do not interpret prompt swaps.')
    if args.no_images:
        raise ValueError('BPP requires the expert RGB observations; omit --no-images.')
    device = 'cuda:0' if args.device == 'cuda' else 'cpu'
    if args.device == 'cuda':
        import subprocess
        subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv'], check=True)
        subprocess.run([__import__('sys').executable, str(BPP_ROOT/'scripts/check_cuda.py')], check=True)
    checkpoint = (args.checkpoint or BPP_ROOT/'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt').resolve()
    print('Loading BPP:', checkpoint, 'device:', device, flush=True)
    model, cfg, checkpoint_metadata = load_policy(checkpoint, device, args.threads)
    print('BPP loaded; diffusion steps:', model.num_inference_steps, flush=True)
    config = {'checkpoint':str(checkpoint), 'checkpoint_task':cfg.task.name, 'state_dict':'model',
              'checkpoint_loading':checkpoint_metadata,
              'device':device, 'diffusion_steps':model.num_inference_steps,
              'prompting':'policy.reset(); policy.prompt(explicit_demo); policy.predict_action(obs)',
              'procedure_label_provided_to_model':False, 'evaluation_regime':'zero-shot diagnostic',
              'max_steps':args.max_steps, 'query_seeds':args.query_seeds,
              'query_start':args.query_start,
              'run_scope':'interface_smoke' if args.max_steps<(756 if args.query_start=='initial' else 466) else 'full_horizon_diagnostic'}
    (output/'rollout_config.json').write_text(json.dumps(config, indent=2))
    scene = Scene()
    scene.canonical_state = np.load(output/'initial_state.npz')['state']
    if args.query_start=='transport':
        with h5py.File(output/'expert_original_narrow.hdf5','r') as f:
            demo=f['data/demo_0']
            first=np.where(demo['phases'][:]==b'transport')[0][0]
            scene.canonical_state=demo['states'][first].copy()
            if 'gripper_command_state' in demo:
                scene.restore_gripper_action=demo['gripper_command_state'][first].copy()
            else:
                gripper=scene.env.robots[0].gripper
                gripper.current_action=np.zeros(2)
                # robosuite updates its gripper integrator at every physics
                # substep, not just once per 20 Hz policy action.
                substeps=int(scene.env.control_timestep/scene.env.model_timestep)
                for action in demo['actions'][:first]:
                    for _ in range(substeps):
                        gripper.format_action(np.array([action[-1]]))
                scene.restore_gripper_action=gripper.current_action.copy()
        np.savez(output/'query_initial_state.npz',state=scene.canonical_state,
                 gripper_current_action=scene.restore_gripper_action)
    inverse_rot = RotationTransformer('axis_angle', 'rotation_6d')
    rows = []
    layout_image_hashes = {}
    try:
        for layout in ['original', 'relocated']:
            for seed in [int(x) for x in args.query_seeds.split(',')]:
                paired_identity = None
                paired_rgb_hash = None
                for procedure in ['narrow', 'wide']:
                    scene.reset(layout)
                    identity = scene.initial_identity()
                    if paired_identity is None:
                        paired_identity = identity
                    assert identity == paired_identity, 'Query state changed between prompts'
                    model.reset(action_exec_horizon=12)
                    path = output/f'expert_original_{procedure}.hdf5'
                    prompt = load_prompt(path, cfg.task.shape_meta, device)
                    with torch.inference_mode():
                        model.prompt(prompt)
                    # Identical diffusion random-number sequence for the paired prompts.
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    if args.device == 'cuda':
                        torch.cuda.manual_seed_all(seed)
                    name=f'rollout_{layout}_{procedure}_seed{seed}'
                    rec=Recorder(output,name); rec.start(scene)
                    if paired_rgb_hash is None:
                        paired_rgb_hash = rec.identity['initial_rgb_sha256']
                    assert rec.identity['initial_rgb_sha256'] == paired_rgb_hash, 'Query images changed between prompts'
                    layout_image_hashes[layout] = paired_rgb_hash
                    if len(layout_image_hashes)==2:
                        assert layout_image_hashes['original'] != layout_image_hashes['relocated'], 'Relocated obstacle is invisible in query RGB'
                    history=[rec.latest_obs]
                    initial_height=scene.object_pos()[2]
                    started=time.monotonic()
                    while len(rec.actions)<args.max_steps:
                        query=query_observations(history,cfg.task.shape_meta,device)
                        with torch.inference_mode():
                            prediction=model.predict_action(query)['action'][0,:12].cpu().numpy()
                        actions=np.concatenate([prediction[:,:3],inverse_rot.inverse(prediction[:,3:9]),prediction[:,9:]],axis=-1)
                        for action in actions:
                            obj=scene.object_pos()
                            transporting=(np.linalg.norm(obj[:2]-scene.goal_pos()[:2])>.045) \
                                if args.query_start=='transport' else (obj[2]>initial_height+.035)
                            rec.phase='transport' if transporting else 'approach_or_place'
                            rec.step(action);history.append(rec.latest_obs);history=history[-2:]
                            if scene.env._check_success() or len(rec.actions)>=args.max_steps:
                                break
                        if len(rec.actions)%60==0:
                            print(name,'step',len(rec.actions),'object_z',round(scene.object_pos()[2],3),
                                  'elapsed',round(time.monotonic()-started,1),flush=True)
                        if scene.env._check_success():
                            break
                    row=rec.finish(scene)
                    row.update(kind='bpp_rollout',procedure=procedure,layout=layout,query_seed=seed,
                               wall_seconds=time.monotonic()-started,
                               explicit_prompt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                    fields=sorted(set().union(*(r.keys() for r in rows)))
                    with (output/'rollout_results.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                        for r in rows:
                            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    finally:
        scene.close()
    return rows
