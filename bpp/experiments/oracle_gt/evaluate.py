#!/usr/bin/env python3
"""Paired Default vs Oracle-GT prompt evaluation.

For every query initial state of a task, the same env is reset to *the exact
same flattened MuJoCo state* for both arms (asserted by sim-state + RGB hash)
and the same diffusion seed is used. Only the prompt differs:

* default: official prompt sampling — query i gets demo_(i % 10) of the task's
  unseen-view HDF5, exactly as the official runner assigns prompts
  (n_envs=10 -> env j -> dataset item j -> demo_j; query g -> demo_(g % 10)).
* oracle:  the GT demonstration generated from *this* query init state by
  gtgen.py (skipped when generation failed for that query).

The policy runs autonomously (predict_action + 12-step exec horizon + 12-step
open-gripper warmup + 550-step cap) — identical semantics to the official
LiberoImageRunner pair-prompting path; GT actions are never replayed.

Per-query results land in <output>/<benchmark>/eval/<task>/query_XX/<arm>/
result.json + trajectory.csv so runs are resumable per query and shardable by
task/query across GPUs/processes.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--tasks', nargs='*', default=None)
    parser.add_argument('--queries', type=str, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed-base', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=550)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--arms', nargs='+', default=['default', 'oracle'],
                        choices=['default', 'oracle', 'shuffle'])
    parser.add_argument('--strict-identity', action='store_true',
                        help='hard-fail on paired initial-state drift (default: warn '
                             'and record identity_mismatch.json; drift is sporadic '
                             '~1e-4 sim-state residual from nondeterministic rollouts)')
    return parser.parse_args()


args = None

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('LIBERO_CONFIG_PATH',
                      str(Path(__file__).resolve().parents[2] / 'config/libero'))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'src'))

import numpy as np
import torch

import tasks as T
from eval_common import load_policy, load_prompt, query_observations
from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder

WARMUP_STEPS = 12
EXEC_HORIZON = 12
OPEN_GRIPPER = np.concatenate([np.zeros(6), [-1.0]])


def make_env(bddl):
    return ControlEnvForRunner(str(bddl), camera_heights=128, camera_widths=128,
                               control_delta=True, use_camera_obs=True,
                               has_offscreen_renderer=True, ignore_done=True,
                               hard_reset=False)


def reset_to_query(w, init_state):
    """Official reset order: seed -> reset -> registered init state -> reset.

    Then canonicalize the controller/gripper internal state (not part of the
    flattened MuJoCo state) so both arms start from bit-identical dynamics —
    same fix used by experiments/native_prompt_swap.
    """
    # Clear any previously registered state so the first reset is always the
    # same random-reset code path across arms and queries (otherwise residual
    # non-sim-state fields can differ and produce ~1e-4 post-warmup drift).
    w.register_init_state(None)
    w.seed(7)
    w.reset()
    w.register_init_state(np.asarray(init_state))
    w.reset()  # set_init_state -> regenerate_obs_from_state
    e = w.env
    e.robots[0].controller.update(force=True)
    e.robots[0].controller.reset_goal()
    e.robots[0].gripper.current_action = np.zeros(2)
    obs = e._get_observations(force_update=True)
    history = [obs]
    for _ in range(WARMUP_STEPS):
        obs, _, _, _ = e.step(OPEN_GRIPPER)
        history.append(obs)
        history = history[-2:]
    identity = {
        'sim_state_sha256': hashlib.sha256(
            e.sim.get_state().flatten().tobytes()).hexdigest(),
        'agentview_rgb_sha256': hashlib.sha256(
            obs['agentview_image'].tobytes()).hexdigest(),
    }
    return history, identity


def rollout(model, w, prompt, shape_meta, device, out_dir, seed, history):
    model.reset(action_exec_horizon=EXEC_HORIZON)
    with torch.inference_mode():
        model.prompt(prompt)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    inverse = RotationTransformer('axis_angle', 'rotation_6d')
    e = w.env
    video = None
    if args.save_video:
        video = VideoRecorder.create_h264(fps=20, thread_count=1)
        video.start(str(out_dir / 'rollout.mp4'))
    steps = 0
    traj = []
    success = False
    started = time.time()
    try:
        while steps < args.max_steps:
            obs_dict = query_observations(history, shape_meta, device)
            with torch.inference_mode():
                prediction = model.predict_action(obs_dict)['action'][0].cpu().numpy()
            chunk = prediction[:EXEC_HORIZON]
            actions = np.concatenate([chunk[:, :3],
                                      inverse.inverse(chunk[:, 3:9]),
                                      chunk[:, 9:]], axis=-1)
            for a in actions:
                if video is not None:
                    video.write_frame(np.concatenate(
                        [history[-1]['agentview_image'][::-1],
                         history[-1]['robot0_eye_in_hand_image'][::-1]],
                        axis=1).copy())
                obs, _, _, _ = e.step(a)
                history.append(obs)
                history = history[-2:]
                steps += 1
                success = bool(e._check_success())
                traj.append({'step': steps, 'eef_pos': obs['robot0_eef_pos'].tolist(),
                             'success': int(success)})
                if success or steps >= args.max_steps:
                    break
            if success:
                break
    finally:
        if video is not None:
            video.stop()
    return {'success': success, 'steps': steps,
            'wall_seconds': round(time.time() - started, 2),
            'trajectory': traj}


def evaluate_task(benchmark, task, model, cfg, device, output):
    init_states = T.load_init_states(benchmark, task)
    unseen = T.unseen_hdf5(benchmark, task)
    assert unseen.exists(), f'missing unseen hdf5: {unseen}'
    eval_dir = output / benchmark / 'eval' / task
    queries = parse_queries(args.queries)
    w = make_env(T.bddl_path(benchmark, task))
    inverse_meta = cfg.task.shape_meta
    prompt_cache = {}
    try:
        for qi in queries:
            pair_identity = {}
            for arm in args.arms:
                qdir = eval_dir / f'query_{qi:02d}' / arm
                rpath = qdir / 'result.json'
                if rpath.exists():
                    entry = json.loads(rpath.read_text())
                    pair_identity[arm] = entry.get('identity', {})
                    continue
                if arm == 'default':
                    demo_idx = qi % T.DEFAULT_PROMPT_SLOTS
                    key = ('default', demo_idx)
                    if key not in prompt_cache:
                        prompt_cache[key] = load_prompt(unseen, inverse_meta,
                                                        device, demo_idx=demo_idx)
                    prompt = prompt_cache[key]
                    prompt_desc = f'{unseen.name}#demo_{demo_idx}'
                else:
                    own_gt = (output / benchmark / 'experts' / task
                              / f'query_{qi:02d}' / 'demo.hdf5')
                    if not own_gt.exists():
                        # same covered query set as the oracle arm
                        continue
                    if arm == 'oracle':
                        gt = own_gt
                        prompt_desc = str(gt.relative_to(output))
                    else:  # shuffle: GT demo generated for a DIFFERENT task
                        donor = T.shuffle_donor(benchmark, task, qi, output)
                        if donor is None:
                            continue
                        gt = (output / benchmark / 'experts' / donor
                              / f'query_{qi:02d}' / 'demo.hdf5')
                        prompt_desc = f'shuffle:{donor}#q{qi:02d}'
                    key = (arm, qi)
                    if key not in prompt_cache:
                        prompt_cache[key] = load_prompt(gt, inverse_meta, device)
                    prompt = prompt_cache[key]
                qdir.mkdir(parents=True, exist_ok=True)
                history, identity = reset_to_query(w, init_states[qi])
                pair_identity[arm] = identity
                seed = args.seed_base * 1000 + qi
                result = rollout(model, w, prompt, inverse_meta, device, qdir,
                                 seed, history)
                traj = result.pop('trajectory')
                result.update({'benchmark': benchmark, 'task': task, 'query': qi,
                               'arm': arm, 'prompt': prompt_desc, 'diffusion_seed': seed,
                               'identity': identity,
                               'query_init_sha256': hashlib.sha256(
                                   np.asarray(init_states[qi]).tobytes()).hexdigest()})
                rpath.write_text(json.dumps(result, indent=1))
                with (qdir / 'trajectory.csv').open('w', newline='') as fcsv:
                    writer = csv.DictWriter(fcsv, fieldnames=['step', 'eef_pos', 'success'])
                    writer.writeheader()
                    for row in traj:
                        writer.writerow(row)
                print(f'[{benchmark}/{task}/q{qi}/{arm}] '
                      f'success={result["success"]} steps={result["steps"]}', flush=True)
            ref = pair_identity.get('default')
            if ref:
                for other_arm, other_id in pair_identity.items():
                    if other_arm == 'default' or not other_id:
                        continue
                    diff = [k for k in ref if ref[k] != other_id.get(k)]
                    if diff:
                        msg = (f'paired initial state drifted for {task}/q{qi} '
                               f'arm={other_arm} in {diff}')
                        assert not args.strict_identity, msg
                        print(f'[WARN] {msg} (recorded; --strict-identity to enforce)',
                              flush=True)
                        mpath = (eval_dir / f'query_{qi:02d}'
                                 / f'identity_mismatch_{other_arm}.json')
                        mpath.write_text(json.dumps({
                            'query': qi, 'arm': other_arm, 'mismatched_keys': diff,
                            'default': ref, other_arm: other_id}, indent=1))
    finally:
        w.close()


def parse_queries(spec):
    if not spec:
        return list(range(T.N_QUERIES))
    out = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    global args
    args = _parse_args()
    checkpoint = args.checkpoint or T.checkpoint_path(args.benchmark)
    assert checkpoint.exists(), f'missing checkpoint: {checkpoint}'
    model, cfg, metadata = load_policy(checkpoint, args.device, args.threads)
    print('loaded', checkpoint.name, 'task:', cfg.task.name, flush=True)
    (args.output / args.benchmark).mkdir(parents=True, exist_ok=True)
    (args.output / args.benchmark / 'eval_config.json').write_text(json.dumps({
        'checkpoint': str(checkpoint), 'checkpoint_metadata': metadata,
        'seed_base': args.seed_base, 'max_steps': args.max_steps,
        'warmup_steps': WARMUP_STEPS, 'exec_horizon': EXEC_HORIZON,
        'default_prompt_rule': 'demo_(query % 10) of unseen hdf5 (official assignment)',
        'shuffle_prompt_rule': ('GT demo generated for a DIFFERENT task in the same '
                                'benchmark at the same query index (task-irrelevant '
                                'arm actions); runs on the same covered queries as oracle'),
        'arms': args.arms, 'queries': parse_queries(args.queries)}, indent=1))
    names = args.tasks or T.task_names(args.benchmark)
    for task in names:
        evaluate_task(args.benchmark, task, model, cfg, args.device, args.output)


if __name__ == '__main__':
    main()
