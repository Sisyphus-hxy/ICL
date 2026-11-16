"""BPP policy/prompt helpers for the paired Default-vs-Oracle-GT evaluation.

Adapted from bpp/experiments/clearance/inference.py (same official loading
path); ``load_prompt`` gains a ``demo_idx`` argument so the Default arm can
reproduce the official prompt assignment (env/query i -> demo_(i % 10)).
"""
import dill
import h5py
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from behavior_prompting.train_network.dataset.libero_replay_image_dataset import (
    _convert_actions, _receding_rgb_numpy_thwc_to_float_chw,
)
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker

BPP_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_TASKS = ['libero_all', 'liberogen_spatial_combination',
                   'liberogen_goal_chain']


def load_policy(checkpoint, device, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    payload = torch.load(str(checkpoint), map_location='cpu', mmap=True,
                         pickle_module=dill, weights_only=False)
    saved_cfg = payload['cfg']
    task = str(saved_cfg.task.name)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f'Unsupported BPP checkpoint task: {task}')
    with initialize_config_dir(
            config_dir=str(BPP_ROOT / 'src/behavior_prompting/train_network/config'),
            version_base=None):
        cfg = compose(config_name='libero_policy_dunetp',
                      overrides=[f'task={task}', f'+modifiers=libero/{task}'])
    def flatten(d, prefix=''):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result.update(flatten(value, prefix + key + '.'))
            else:
                result[prefix + key] = value
        return result
    old = flatten(OmegaConf.to_container(saved_cfg.model, resolve=True))
    new = flatten(OmegaConf.to_container(cfg.model, resolve=True))
    differences = [key for key in set(old) | set(new)
                   if old.get(key) != new.get(key)
                   and key != 'obs_encoder.obs_encoder.use_pool_modality_pos_embed']
    if differences:
        raise ValueError(
            f'Checkpoint has unverified model configuration differences: {sorted(differences)}')
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(payload['state_dicts']['model'], strict=True)
    model.eval().to(device)
    metadata = {'configuration_source':
                'pinned official libero_policy_dunetp + task/modifier (same as run_eval.sh)',
                'saved_pool_modality_pos_embed':
                saved_cfg.model.obs_encoder.obs_encoder.get('use_pool_modality_pos_embed', None),
                'effective_pool_modality_pos_embed':
                bool(cfg.model.obs_encoder.obs_encoder.use_pool_modality_pos_embed),
                'other_resolved_model_configuration_differences': differences,
                'state_dict_load': 'strict; no missing or unexpected keys'}
    return model, cfg, metadata


def load_prompt(path, shape_meta, device, demo_idx=0):
    """Load one HDF5 demonstration as a BPP pair-prompt tensor dict."""
    rot = RotationTransformer('axis_angle', 'rotation_6d')
    chunk = int(shape_meta['prompt_chunk_n_actions'])
    with h5py.File(path, 'r') as f:
        d = f[f'data/demo_{demo_idx}']
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
    mapping = {'ee_pos': 'robot0_eef_pos', 'ee_ori': 'robot0_eef_quat',
               'gripper_states': 'robot0_gripper_qpos',
               'agentview_rgb': 'agentview_image',
               'eye_in_hand_rgb': 'robot0_eye_in_hand_image'}
    result = {}
    for key, sim_key in mapping.items():
        n = int(shape_meta['obs'][key]['horizon'])
        values = [history[max(len(history) - n + i, 0)][sim_key] for i in range(n)]
        value = np.asarray(values)
        if key.endswith('_rgb'):
            value = _receding_rgb_numpy_thwc_to_float_chw(value)
        elif key == 'ee_ori':
            value = rot.forward(value[..., [3, 0, 1, 2]].astype(np.float32))
        result[key] = torch.as_tensor(value[None].copy(), dtype=torch.float32,
                                      device=device)
    return result
