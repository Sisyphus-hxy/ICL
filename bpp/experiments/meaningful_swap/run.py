#!/usr/bin/env python3
"""Meaningful procedure prompt swap in the *native* LIBERO scene.

Two experiments, both paired on identical query initial states and diffusion
seeds, with the native 12-step open-gripper warmup and 550-step cap restored:

* gen_gentle / eval_gentle: query scene is the unmodified original scene.
  `normal` prompt = natively replayed source demo_0 (identical content to the
  validated 7/10 baseline prompt). `gentle` prompt = the same action sequence
  with the transport window time-dilated x2 (each 6D delta split into halves).
  Same spatial path, same goal; only transport speed/acceleration/jerk change.

* gen_obstacle / eval_obstacle: query scene = native scene + one relocatable
  mocap post placed just beside the transport corridor (outside the grasp
  swept volume; direct path grazes it). `narrow` / `wide` prompts are demo_0
  with the same one-sided lateral transport bump of different amplitude, both
  verified collision-free and successful in the same scene.

No training, no checkpoint or model changes. Helpers are copied/reused from
native_prompt_swap/run.py and clearance/{inference,scene,recording}.py.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['gen_gentle', 'eval_gentle',
                                      'gen_obstacle', 'eval_obstacle'])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--gpu', default='GPU-e0914423-1b44-589b-fdb3-0360e8f79b43')
parser.add_argument('--egl-device', default='0')
parser.add_argument('--max-steps', type=int, default=550)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--slow-factor', type=int, default=2)
parser.add_argument('--obstacle-xy', default='auto',
                    help='"x,y" or "auto" to grid-search a grazing post position')
parser.add_argument('--narrow-clearance-m', type=float, default=0.007)
parser.add_argument('--wide-clearance-m', type=float, default=0.030)
parser.add_argument('--wide-end-shift-m', type=float, default=0.025,
                    help='+y shift of the wide demo transport end, lifting the '
                         'converged-end clearance cap imposed by the short corridor')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['MUJOCO_EGL_DEVICE_ID'] = args.egl_device
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYNPUT_BACKEND'] = 'dummy'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'clearance'))

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
from scene import BPP_ROOT, SPLIT, TASK, OBJECT, GOAL
from inference import load_policy, load_prompt, query_observations

SOURCE = BPP_ROOT / 'datasets/libero' / SPLIT / (TASK + '_demo.hdf5')
BDDL = BPP_ROOT / 'src/behavior_prompting/train_network/env/libero/bddl_files' / SPLIT / (TASK + '.bddl')
CHECKPOINT = BPP_ROOT / 'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt'
BASELINE_PROMPT = (HERE.parent / 'native_prompt_swap' / 'outputs' /
                   'paired10_controlled' / 'original_replayed_demo.hdf5')
TRANSPORT = (82, 113)
WARMUP = 12
DT = 1.0 / 20.0


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_demo(index=0):
    with h5py.File(SOURCE, 'r') as f:
        d = f[f'data/demo_{index}']
        return d['states'][0].copy(), d['actions'][:].copy(), int(d.attrs.get('seed', 2))


def wrapper(images):
    return ControlEnvForRunner(str(BDDL), camera_heights=128, camera_widths=128,
                               control_delta=True, use_camera_obs=images,
                               has_offscreen_renderer=images,
                               ignore_done=True, hard_reset=False)


def reset(e, state, seed):
    np.random.seed(seed)
    e.seed(seed)
    e.reset()
    e.sim.set_state_from_flattened(state)
    e.sim.forward()
    e.robots[0].controller.update(force=True)
    e.robots[0].controller.reset_goal()
    e.robots[0].gripper.current_action = np.zeros(2)
    return e._get_observations(force_update=True)


def replay(e, state, seed, actions, video_path=None, record=False, monitor=None):
    obs = reset(e, state, seed)
    initial_obs = {k: obs[k].copy() for k in
                   ['robot0_eef_pos', 'agentview_image', 'robot0_eye_in_hand_image'] if k in obs}
    rows = []
    observations = []
    states = []
    video = None
    if video_path:
        video = VideoRecorder.create_h264(fps=20, thread_count=1)
        video.start(str(video_path))
    try:
        for i, a in enumerate(actions):
            if monitor:
                monitor['step'] = i
            states.append(e.sim.get_state().flatten().copy())
            obs, _, _, _ = e.step(a)
            if record:
                observations.append({k: obs[k].copy() for k in [
                    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                    'agentview_image', 'robot0_eye_in_hand_image']})
            if video:
                video.write_frame(np.concatenate([obs['agentview_image'][::-1],
                                                  obs['robot0_eye_in_hand_image'][::-1]], axis=1).copy())
            bowl = e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
            goal = e.sim.data.body_xpos[e.obj_body_id[GOAL]].copy()
            rows.append(dict(step=i, eef_x=float(obs['robot0_eef_pos'][0]),
                             eef_y=float(obs['robot0_eef_pos'][1]),
                             eef_z=float(obs['robot0_eef_pos'][2]),
                             bowl_x=float(bowl[0]), bowl_y=float(bowl[1]), bowl_z=float(bowl[2]),
                             goal_distance_xy=float(np.linalg.norm(bowl[:2] - goal[:2])),
                             goal_success=int(e._check_success())))
    finally:
        if video:
            video.stop()
    return dict(rows=rows, obs=observations, states=states, initial_obs=initial_obs,
                terminal_state=e.sim.get_state().flatten().copy(),
                success=bool(e._check_success()))


def write_demo(path, replayed, actions, seed, origin):
    with h5py.File(SOURCE, 'r') as src, h5py.File(path, 'w') as f:
        original = src['data/demo_0']
        demo = f.create_group('data/demo_0')
        demo.attrs['model_file'] = original.attrs['model_file']
        demo.attrs['seed'] = seed
        demo.attrs['num_samples'] = len(actions)
        demo.attrs['observation_alignment'] = 'post_action'
        demo.attrs['state_alignment'] = 'pre_action'
        demo.attrs['origin'] = origin
        demo.create_dataset('states', data=np.asarray(replayed['states']))
        demo.create_dataset('actions', data=actions)
        demo.create_dataset('initial_ee_pos', data=replayed['initial_obs']['robot0_eef_pos'])
        demo.create_dataset('terminal_state', data=replayed['terminal_state'])
        o = demo.create_group('obs')
        mapping = {'robot0_eef_pos': 'ee_pos', 'robot0_gripper_qpos': 'gripper_states',
                   'agentview_image': 'agentview_rgb', 'robot0_eye_in_hand_image': 'eye_in_hand_rgb'}
        for native, key in mapping.items():
            o.create_dataset(key, data=np.asarray([v[native] for v in replayed['obs']]),
                             compression='gzip', compression_opts=1)
        ori = Rotation.from_quat(np.asarray([v['robot0_eef_quat'] for v in replayed['obs']])).as_rotvec()
        o.create_dataset('ee_ori', data=ori)


def dilate(actions, window, factor):
    """Time-dilate `window` by `factor`: each 6D delta becomes `factor` smaller
    steps along the identical piecewise-linear path. Gripper channel unchanged."""
    out = list(actions[:window[0]])
    for a in actions[window[0]:window[1]]:
        half = a.copy()
        half[:6] /= factor
        for _ in range(factor):
            out.append(half.copy())
    out.extend(actions[window[1]:])
    return np.asarray(out)


def bump(actions, window, amplitude):
    """One-sided lateral bump inside `window`: path deviates by `amplitude` at
    the midpoint and returns to the original route (net-zero displacement)."""
    out = actions.copy()
    n = window[1] - window[0]
    t = (np.arange(n) + 0.5) / n
    offset = amplitude * np.sin(np.pi * t)
    out[window[0]:window[1], 1] += np.diff(np.r_[0.0, offset])
    return out


def arc_interp(points, count):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if d[-1] == 0:
        return np.repeat(points[:1], count, axis=0)
    t = np.linspace(0.0, d[-1], count)
    return np.stack([np.interp(t, d, points[:, k]) for k in range(points.shape[1])], axis=1)


def path_stats(rows):
    eef = np.array([[r['eef_x'], r['eef_y'], r['eef_z']] for r in rows])
    bowl = np.array([[r['bowl_x'], r['bowl_y'], r['bowl_z']] for r in rows])
    return eef, bowl


def dynamics(positions):
    """EEF speed/acc/jerk magnitudes at 20 Hz over the given positions."""
    if len(positions) < 4:
        z = np.zeros(1)
        return z, z, z
    vel = np.diff(positions, axis=0) / DT
    acc = np.diff(vel, axis=0) / DT
    jerk = np.diff(acc, axis=0) / DT
    return (np.linalg.norm(vel, axis=1), np.linalg.norm(acc, axis=1),
            np.linalg.norm(jerk, axis=1))


def milestones(rows, initial_bowl, goal):
    bowl = np.array([[r['bowl_x'], r['bowl_y'], r['bowl_z']] for r in rows])
    min_goal = min(r['goal_distance_xy'] for r in rows)
    return {
        'lifted': bool((bowl[:, 2] > initial_bowl[2] + .045).any()),
        'transported': bool((np.linalg.norm(bowl[:, :2] - initial_bowl[:2], axis=1) > .035).any()),
        'near_goal': bool(min_goal < .055),
        'min_goal_distance_xy_m': float(min_goal),
    }


def failure_stage(row):
    if row['success']:
        return 'none'
    for stage in ['lifted', 'transported', 'near_goal']:
        if not row[stage]:
            return stage
    return 'goal_placement'


# ---------------- contact monitoring (plain env, same rules as Scene.contacts)

def make_contact_monitor(e):
    m = e.sim.model._model
    d = e.sim.data._data
    bowl_body = e.obj_body_id[OBJECT]
    bowl_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == bowl_body
                  and (m.geom_contype[i] or m.geom_conaffinity[i])}
    robot_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] > 0
                   and (e.sim.model.geom_id2name(i) or '').startswith(('robot0_', 'gripper0_'))
                   and (m.geom_contype[i] or m.geom_conaffinity[i])}
    gripper_geoms = {i for i in robot_geoms
                     if (e.sim.model.geom_id2name(i) or '').startswith('gripper0_')}
    state = {'step': 0, 'phase': 'approach'}
    events = []

    def scan():
        for c in d.contact[:d.ncon]:
            if c.dist > 0:
                continue
            a, b = int(c.geom1), int(c.geom2)
            pair = {a, b}
            if (a in gripper_geoms and b in bowl_geoms) or (b in gripper_geoms and a in bowl_geoms):
                continue
            hit = None
            if a in robot_geoms and b in robot_geoms:
                if not (a in gripper_geoms and b in gripper_geoms):
                    hit = (a, b)
            elif (a in robot_geoms) != (b in robot_geoms):
                other = b if a in robot_geoms else a
                if (e.sim.model.geom_id2name(other) or '') != 'floor':
                    hit = (a, b)
            elif state['phase'] == 'transport' and ((a in bowl_geoms) != (b in bowl_geoms)) \
                    and not pair & gripper_geoms:
                hit = (a, b)
            if hit:
                events.append({'control_step': state['step'], 'phase': state['phase'],
                               'geom1': e.sim.model.geom_id2name(hit[0]) or '',
                               'geom2': e.sim.model.geom_id2name(hit[1]) or '',
                               'dist': float(c.dist)})
    return scan, events, state


def contact_category(geom1, geom2, phase, step):
    names = geom1 + ' ' + geom2
    if 'procedure_obstacle' in names:
        return 'obstacle'
    if 'plate_1' in names or 'flat_stove' in names:
        return 'distractor'
    if 'cookies_1' in names:
        return 'goal_box'
    if 'table_collision' in names and 'akita_black_bowl' in names:
        return 'bowl_table_transport' if phase == 'transport' else 'bowl_table_rest'
    if 'table_collision' in names:
        return 'finger_table_grasp' if (phase == 'grasp' and step < 120) else 'robot_table_other'
    if ('robot0' in names or 'gripper0' in names):
        a_r = geom1.startswith(('robot0', 'gripper0'))
        b_r = geom2.startswith(('robot0', 'gripper0'))
        return 'robot_self' if (a_r and b_r) else 'robot_other'
    return 'other'


UNINTENDED = ('obstacle', 'distractor', 'goal_box', 'robot_table_other',
              'bowl_table_transport', 'robot_self', 'robot_other', 'other')


def summarize_contacts(events):
    import collections
    cats = collections.Counter()
    first = None
    first_unintended = None
    for ev in events:
        cat = contact_category(ev['geom1'], ev['geom2'], ev['phase'], ev['control_step'])
        cats[cat] += 1
        if first is None:
            first = {'control_step': ev['control_step'], 'phase': ev['phase'],
                     'geom1': ev['geom1'], 'geom2': ev['geom2'], 'category': cat}
        if first_unintended is None and cat in UNINTENDED:
            first_unintended = {'control_step': ev['control_step'], 'phase': ev['phase'],
                                'geom1': ev['geom1'], 'geom2': ev['geom2'], 'category': cat}
    return {
        'any_contact': bool(events),
        'n_contact_events': len(events),
        'first_contact': first,
        'first_unintended_contact': first_unintended,
        'unintended_collision': first_unintended is not None,
        'benign_only_contacts': bool(events) and first_unintended is None,
        'contact_counts': dict(cats),
    }


def phase_label(bowl, initial_bowl, goal_xy):
    if bowl[2] <= initial_bowl[2] + .035:
        return 'grasp'
    if np.linalg.norm(bowl[:2] - goal_xy) > .045:
        return 'transport'
    return 'place'


# ---------------- generic paired evaluator (native env)

def paired_rollout(e, model, cfg, prompt, state, seed, query_seed, name, out, device):
    obs = reset(e, state, seed)
    history = [obs]
    for _ in range(WARMUP):
        obs, _, _, _ = e.step(np.r_[np.zeros(6), -1.])
        history.append(obs)
        history = history[-2:]
    identity = {'sim_state_sha256': hashlib.sha256(e.sim.get_state().flatten().tobytes()).hexdigest(),
                'agentview_rgb_sha256': hashlib.sha256(obs['agentview_image'].tobytes()).hexdigest(),
                'eef_pos': obs['robot0_eef_pos'].tolist(),
                'gripper_command': e.robots[0].gripper.current_action.tolist(),
                'controller_goal_pos': e.robots[0].controller.goal_pos.tolist()}
    model.reset(action_exec_horizon=12)
    with torch.inference_mode():
        model.prompt(prompt)
    torch.manual_seed(query_seed)
    torch.cuda.manual_seed_all(query_seed)
    scan, events, mon = make_contact_monitor(e)
    original_step = e.sim.step
    e.sim.step = lambda: (original_step(), scan())
    inverse = RotationTransformer('axis_angle', 'rotation_6d')
    video = VideoRecorder.create_h264(fps=20, thread_count=1)
    video.start(str(out / (name + '.mp4')))
    rows = []
    initial_bowl = e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
    goal = e.sim.data.body_xpos[e.obj_body_id[GOAL]].copy()
    started = time.monotonic()
    try:
        while len(rows) < args.max_steps:
            query = query_observations(history, cfg.task.shape_meta, device)
            with torch.inference_mode():
                pred = model.predict_action(query)['action'][0, :12].cpu().numpy()
            actions = np.concatenate([pred[:, :3], inverse.inverse(pred[:, 3:9]), pred[:, 9:]], axis=-1)
            for a in actions:
                video.write_frame(np.concatenate([history[-1]['agentview_image'][::-1],
                                                  history[-1]['robot0_eye_in_hand_image'][::-1]], axis=1).copy())
                mon['step'] = len(rows)
                obs, _, _, _ = e.step(a)
                history.append(obs)
                history = history[-2:]
                bowl = e.sim.data.body_xpos[e.obj_body_id[OBJECT]].copy()
                goal_dist = float(np.linalg.norm(bowl[:2] - goal[:2]))
                mon['phase'] = phase_label(bowl, initial_bowl, goal[:2])
                rows.append(dict(step=len(rows), eef_x=float(obs['robot0_eef_pos'][0]),
                                 eef_y=float(obs['robot0_eef_pos'][1]), eef_z=float(obs['robot0_eef_pos'][2]),
                                 phase=mon['phase'], bowl_x=float(bowl[0]), bowl_y=float(bowl[1]),
                                 bowl_z=float(bowl[2]), goal_distance_xy=goal_dist,
                                 goal_success=int(e._check_success())))
                if rows[-1]['goal_success'] or len(rows) >= args.max_steps:
                    break
            if rows[-1]['goal_success']:
                break
    finally:
        video.stop()
        e.sim.step = original_step
    save_csv(out / (name + '_trajectory.csv'), rows)
    contacts_path = out / (name + '_contacts.json')
    contacts_path.write_text(json.dumps(events, indent=2))
    contact_summary = summarize_contacts(events)
    eef, bowl = path_stats(rows)
    transport = np.asarray([r['phase'] == 'transport' for r in rows])
    speed, acc, jerk = dynamics(eef[transport]) if transport.sum() > 3 else (np.zeros(1),) * 3
    result = dict(seed=query_seed, source_initial_state_index=query_seed,
                  source_reset_seed=seed, prompt=name.rsplit('_', 1)[-1],
                  name=name, success=bool(rows[-1]['goal_success']), steps=len(rows),
                  control_seconds=len(rows) * DT,
                  transport_steps=int(transport.sum()),
                  transport_duration_s=float(transport.sum()) * DT,
                  transport_path_length_m=float(np.linalg.norm(np.diff(eef[transport], axis=0), axis=1).sum()) if transport.sum() > 1 else None,
                  eef_speed_mean_m_s=float(speed.mean()), eef_speed_peak_m_s=float(speed.max()),
                  eef_acc_peak_m_s2=float(acc.max()), eef_jerk_peak_m_s3=float(jerk.max()),
                  initial_identity=identity, wall_seconds=time.monotonic() - started,
                  **contact_summary)
    result.update(milestones(rows, initial_bowl, goal))
    result['failure_stage'] = failure_stage(result)
    return result


def evaluate(out, prompts, tag):
    assert args.trials == 10
    import subprocess
    subprocess.run([sys.executable, str(BPP_ROOT / 'scripts/check_cuda.py')], check=True)
    device = 'cuda:0'
    model, cfg, metadata = load_policy(CHECKPOINT, device, args.threads)
    labels = list(prompts)
    loaded = {label: load_prompt(path, cfg.task.shape_meta, device) for label, path in prompts.items()}
    states = []
    for i in range(args.trials):
        state, _, reset_seed = source_demo(i)
        states.append((state, reset_seed))
    unique = {hashlib.sha256(s.tobytes()).hexdigest() for s, _ in states}
    assert len(unique) == args.trials, 'query initial states are not distinct'
    config = {'checkpoint': str(CHECKPOINT), 'checkpoint_load': metadata,
              'prompt_paths': {k: str(v) for k, v in prompts.items()},
              'max_steps': args.max_steps, 'warmup_open_steps': WARMUP,
              'paired_query_state_indices': list(range(args.trials)),
              'diffusion_seeds': list(range(args.trials)),
              'query_state_source': 'source demo_0..9 flattened initial states; 10 distinct sha256',
              'experiment': tag}
    (out / 'evaluation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    w = wrapper(True)
    e = w.env
    results = []
    try:
        for i in range(args.trials):
            state, reset_seed = states[i]
            pair_identity = None
            for label in labels:
                name = f'rollout_seed{i}_{label}'
                result = paired_rollout(e, model, cfg, loaded[label], state, reset_seed,
                                        i, name, out, device)
                if pair_identity is None:
                    pair_identity = result['initial_identity']
                assert result['initial_identity'] == pair_identity, 'paired query state drifted'
                results.append(result)
                print(json.dumps({k: result[k] for k in
                                  ('seed', 'prompt', 'success', 'steps', 'failure_stage',
                                   'unintended_collision')}, default=str), flush=True)
                (out / 'rollout_results.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
    finally:
        w.close()
    save_csv(out / 'rollout_results.csv',
             [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
              for r in results])
    paired = []
    for i in range(args.trials):
        pair = {r['prompt']: r for r in results if r['seed'] == i}
        row = {'seed': i}
        for label in labels:
            r = pair[label]
            for k in ('success', 'lifted', 'transported', 'near_goal', 'steps',
                      'failure_stage', 'transport_duration_s', 'transport_path_length_m',
                      'eef_speed_mean_m_s', 'eef_acc_peak_m_s2', 'eef_jerk_peak_m_s3',
                      'unintended_collision', 'any_contact'):
                row[f'{label}_{k}'] = r[k]
        paired.append(row)
    save_csv(out / 'paired_results.csv', paired)
    (out / 'paired_results.json').write_text(json.dumps(paired, indent=2, default=str) + '\n')
    summary = {label: {k: sum(bool(r[k]) for r in results if r['prompt'] == label)
                       for k in ('success', 'lifted', 'transported', 'near_goal',
                                 'unintended_collision')} for label in labels}
    (out / 'paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return results


# ---------------- experiment 1: gentle

def gen_gentle(out):
    state, actions, seed = source_demo(0)
    factor = args.slow_factor
    gentle_actions = dilate(actions, TRANSPORT, factor)
    w = wrapper(True)
    try:
        normal = replay(w.env, state, seed, actions, out / 'normal_demo.mp4', record=True)
        gentle = replay(w.env, state, seed, gentle_actions, out / 'gentle_demo.mp4', record=True)
    finally:
        w.close()
    assert normal['success'], 'normal replay must succeed'
    assert gentle['success'], 'gentle replay must succeed'
    # identical commanded transport path: compare realized EEF paths on a
    # common arc-length grid inside the dilated window
    n_eef, n_bowl = path_stats(normal['rows'])
    g_eef, g_bowl = path_stats(gentle['rows'])
    n_seg = n_eef[TRANSPORT[0]:TRANSPORT[1]]
    g_seg = g_eef[TRANSPORT[0]:TRANSPORT[0] + (TRANSPORT[1] - TRANSPORT[0]) * factor]
    n_res = arc_interp(n_seg, 200)
    g_res = arc_interp(g_seg, 200)
    deviation = float(np.linalg.norm(n_res - g_res, axis=1).max())
    n_speed, n_acc, n_jerk = dynamics(n_seg)
    g_speed, g_acc, g_jerk = dynamics(g_seg)
    write_demo(out / 'normal_demo.hdf5', normal, actions, seed,
               'native replay of source demo_0 actions')
    write_demo(out / 'gentle_demo.hdf5', gentle, gentle_actions, seed,
               f'demo_0 transport x{factor} time-dilated; identical commanded path')
    # verify regenerated normal prompt equals the validated 7/10 prompt content
    parity = {}
    if BASELINE_PROMPT.is_file():
        with h5py.File(BASELINE_PROMPT) as a, h5py.File(out / 'normal_demo.hdf5') as b:
            da, db = a['data/demo_0'], b['data/demo_0']
            for key in ('ee_pos', 'ee_ori', 'gripper_states', 'agentview_rgb', 'eye_in_hand_rgb'):
                parity[key] = bool(np.array_equal(da['obs/' + key][:], db['obs/' + key][:]))
            parity['actions'] = bool(np.array_equal(da['actions'][:], db['actions'][:]))
    save_csv(out / 'normal_demo_trajectory.csv', normal['rows'])
    save_csv(out / 'gentle_demo_trajectory.csv', gentle['rows'])
    validation = {
        'factor': factor,
        'normal_replay_success': normal['success'],
        'gentle_replay_success': gentle['success'],
        'normal_steps': len(actions), 'gentle_steps': len(gentle_actions),
        'transport_window': list(TRANSPORT),
        'transport_eef_max_path_deviation_m': deviation,
        'transport_bowl_max_path_deviation_m': float(np.linalg.norm(
            arc_interp(n_bowl[TRANSPORT[0]:TRANSPORT[1]], 200) -
            arc_interp(g_bowl[TRANSPORT[0]:TRANSPORT[0] + (TRANSPORT[1] - TRANSPORT[0]) * factor], 200),
            axis=1).max()),
        'normal_transport_eef_speed_mean': float(n_speed.mean()),
        'gentle_transport_eef_speed_mean': float(g_speed.mean()),
        'speed_ratio': float(n_speed.mean() / max(g_speed.mean(), 1e-9)),
        'normal_acc_p95': float(np.percentile(n_acc, 95)),
        'gentle_acc_p95': float(np.percentile(g_acc, 95)),
        'normal_jerk_p95': float(np.percentile(n_jerk, 95)),
        'gentle_jerk_p95': float(np.percentile(g_jerk, 95)),
        'actions_identical_outside_transport': bool(
            np.array_equal(actions[:TRANSPORT[0]], gentle_actions[:TRANSPORT[0]]) and
            np.array_equal(actions[TRANSPORT[1]:], gentle_actions[TRANSPORT[0] + (TRANSPORT[1] - TRANSPORT[0]) * factor:])),
        'same_goal': True,
        'normal_prompt_equals_validated_baseline': parity,
    }
    (out / 'demo_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2), flush=True)


def eval_gentle(out):
    assert json.loads((out / 'demo_validation.json').read_text())['gentle_replay_success']
    prompts = {'normal': out / 'normal_demo.hdf5', 'gentle': out / 'gentle_demo.hdf5'}
    evaluate(out, prompts, 'gentle_transport_native_scene')


# ---------------- experiment 2: obstacle

def gen_obstacle(out):
    import scene as clearance_scene
    from recording import Recorder
    from scene import Scene
    state, actions, seed = source_demo(0)
    with h5py.File(SOURCE) as f:
        demo_states = f['data/demo_0/states'][:]
    scene = Scene(record_images=True)
    scene.canonical_state = state

    def place_obstacle(pos):
        clearance_scene.LAYOUTS['post'] = np.asarray(pos, dtype=np.float64)
        scene.d.mocap_pos[scene.mocap_id] = np.asarray(pos, dtype=np.float64)
        scene.d.mocap_quat[scene.mocap_id] = [1, 0, 0, 0]

    def geom_distance(pos, state_vec, geoms, cutoff=None):
        """Set post to `pos`, restore `state_vec`, return min surface distance
        between `geoms` and the post (early exit once below `cutoff`)."""
        scene.env.sim.set_state_from_flattened(state_vec)
        place_obstacle(pos)
        scene.env.sim.forward()
        import mujoco
        out6 = np.empty(6)
        best = np.inf
        for g in geoms:
            best = min(best, float(mujoco.mj_geomDistance(
                scene.m, scene.d, g, scene.obstacle_id, 2., out6)))
            if cutoff is not None and best < cutoff:
                return best
        return best

    def auto_place():
        """Grid-search a post position that blocks/grazes the transport corridor
        while staying outside the grasp/approach/lift swept volume and clear of
        the place segment (state-level check against demo_0 snapshots, no action
        replay). The 9 cm native corridor plus the ~8 cm bowl geom spread means
        any grazing post sits within a few mm of the converged end pose, so the
        narrow demo is endpoint-capped; the wide demo shifts its transport-end
        +y to lift the cap and lets the mid-path detour set the clearance."""
        robot = scene.robot_geoms
        bowl = scene.object_geoms
        samples_pre = list(range(0, TRANSPORT[0], 2))
        samples_tr = list(range(TRANSPORT[0], TRANSPORT[1]))
        samples_place = list(range(TRANSPORT[1], len(demo_states)))
        best = None
        for z in (0.99, 0.95, 0.91):
            for x in np.arange(-0.06, 0.121, 0.01):
                for y in np.arange(-0.08, 0.121, 0.01):
                    pos = np.array([x, y, z])
                    pre = np.inf
                    for i in samples_pre:
                        pre = min(pre, geom_distance(pos, demo_states[i],
                                                     robot + bowl, cutoff=0.005))
                        if pre < 0.005:
                            break
                    if pre < 0.005:
                        continue
                    place = min(geom_distance(pos, demo_states[i], robot + bowl)
                                for i in samples_place)
                    if place < 0.003:
                        continue
                    tr = min(geom_distance(pos, demo_states[i], bowl)
                             for i in samples_tr)
                    if tr > 0.008:
                        continue
                    score = (abs(tr - (-0.002)), -pre)
                    if best is None or score < best[0]:
                        best = (score, pos, pre, tr, place)
        assert best is not None, 'no obstacle position satisfies the swept-volume constraints'
        return best

    if args.obstacle_xy == 'auto':
        score, obstacle, pre_c, tr_c, place_c = auto_place()
        print('auto-placed obstacle', obstacle.tolist(),
              'pre-transport clearance %.4f' % pre_c,
              'direct transport bowl clearance %.4f' % tr_c,
              'place clearance %.4f' % place_c, flush=True)
    else:
        obstacle = np.array([float(x) for x in args.obstacle_xy.split(',')] + [0.99])
    clearance_scene.LAYOUTS['post'] = obstacle
    phases = [(0, 50, 'approach'), (50, 60, 'grasp'), (60, 82, 'lift'),
              (82, 113, 'transport'), (113, 122, 'place'), (122, 135, 'release')]

    def phase_of(i):
        for s, e_, name in phases:
            if s <= i < e_:
                return name
        return 'release'

    from expert import Servo, resample

    def replay_with_recorder(act, name):
        scene.reset('post')
        rec = Recorder(out, name, save_images=not name.startswith('_tuning'))
        rec.start(scene)
        try:
            for i, a in enumerate(act):
                rec.phase = phase_of(i)
                rec.step(a)
        finally:
            summary = rec.finish(scene)
        traj = np.genfromtxt(out / (name + '_trajectory.csv'), delimiter=',',
                             names=True, dtype=None, encoding='utf8')
        contacts = json.loads((out / (name + '_contacts.json')).read_text())
        post_contacts = [ev for ev in contacts
                         if any('procedure_obstacle' in c[0] or 'procedure_obstacle' in c[1]
                                for c in ev['contacts'])]
        transport_mask = traj['phase'] == 'transport'
        min_clear = float(traj['object_clearance_m'][transport_mask].min()) \
            if transport_mask.any() else float('nan')
        return summary, traj, contacts, post_contacts, min_clear

    # direct replay: verify the post grazes the transport path but never
    # enters the grasp swept volume (no post contact before step 82)
    direct, direct_traj, direct_contacts, direct_post, direct_clear = \
        replay_with_recorder(actions, 'direct_demo')
    pre_transport_post = [ev for ev in direct_contacts
                          if ev['control_step'] < TRANSPORT[0] and
                          any('procedure_obstacle' in ' '.join(c[:2]) for c in ev['contacts'])]
    validation = {'obstacle_xy': obstacle.tolist(),
                  'direct_replay_success': direct['goal_success'],
                  'direct_min_transport_clearance_m': direct_clear,
                  'direct_post_contact_events': len(direct_post),
                  'post_contacts_before_transport': len(pre_transport_post)}
    assert len(pre_transport_post) == 0, 'obstacle enters the grasp swept volume'

    # Transport-end and place poses from the source demo's post-action
    # observations: the servo segment lands where the demo's transport ended,
    # then a common servo descent recreates the demo's place approach, and the
    # demo's release actions finish the task identically for both procedures.
    with h5py.File(SOURCE) as f:
        demo = f['data/demo_0']
        end_pos = demo['obs/ee_pos'][TRANSPORT[1] - 1][:]
        end_quat = Rotation.from_rotvec(demo['obs/ee_ori'][TRANSPORT[1] - 1][:]).as_quat()
        place_pos = demo['obs/ee_pos'][121][:]
        place_quat = Rotation.from_rotvec(demo['obs/ee_ori'][121][:]).as_quat()
    end_pose = np.r_[end_pos, end_quat]
    place_pose = np.r_[place_pos, place_quat]

    def curved_path(start, end, peak, n=40):
        """Line from start to end plus a one-sided sin^2 lateral excursion whose
        apex sits at absolute height `peak` in +y (away from the post)."""
        t = np.linspace(0., 1., 4001)
        p = start[None] * (1 - t[:, None]) + end[None] * t[:, None]
        p[:, 1] += (peak - (start[1] + end[1]) / 2) * np.sin(np.pi * t) ** 2
        return resample(p, n)

    def servo_variant(peak, name, y_shift=0.):
        """Replay demo approach/grasp/lift and release; servo the transport
        segment along `curved_path(peak)` to a transport end shifted by
        `y_shift` in +y, then servo down to the demo's place pose. Returns
        metrics for the tuning loop."""
        scene.reset('post')
        rec = Recorder(out, name, save_images=not name.startswith('_tuning'))
        rec.start(scene)
        servo = Servo(rec)
        try:
            for i, a in enumerate(actions[:TRANSPORT[0]]):
                rec.phase = phase_of(i)
                rec.step(a)
            start = rec.latest_obs['robot0_eef_pos'].copy()
            q0 = Rotation.from_quat(rec.latest_obs['robot0_eef_quat'].copy())
            q1 = Rotation.from_quat(end_quat)
            slerp = Slerp([0., 1.], Rotation.from_quat([q0.as_quat(), q1.as_quat()]))
            target = end_pos + np.array([0., y_shift, 0.])
            path = curved_path(start, target, peak)
            rec.phase = 'transport'
            for k, pt in enumerate(path):
                quat = slerp((k + 1) / len(path)).as_quat()
                servo.move_to_pose(np.r_[pt, quat], close_gripper=True,
                                   max_steps=2, pos_threshold=0., rot_threshold=0.)
            servo.move_to_pose(np.r_[target, end_quat], close_gripper=True,
                               max_steps=40, pos_threshold=0.002, rot_threshold=0.01)
            rec.phase = 'place'
            servo.move_to_pose(place_pose, close_gripper=True, max_steps=80,
                               pos_threshold=0.003, rot_threshold=0.02)
            for j, a in enumerate(actions[122:], start=122):
                rec.phase = phase_of(j)
                rec.step(a)
        finally:
            summary = rec.finish(scene)
        traj = np.genfromtxt(out / (name + '_trajectory.csv'), delimiter=',',
                             names=True, dtype=None, encoding='utf8')
        contacts = json.loads((out / (name + '_contacts.json')).read_text())
        post_contacts = [ev for ev in contacts
                         if any('procedure_obstacle' in c[0] or 'procedure_obstacle' in c[1]
                                for c in ev['contacts'])]
        pre = [ev for ev in contacts if ev['control_step'] < TRANSPORT[0] and
               any('procedure_obstacle' in ' '.join(c[:2]) for c in ev['contacts'])]
        mask = traj['phase'] == 'transport'
        min_clear = float(traj['object_clearance_m'][mask].min()) if mask.any() else float('nan')
        return dict(name=name, peak=peak, y_shift=y_shift,
                    success=summary['goal_success'],
                    min_transport_clearance_m=min_clear,
                    post_contact_events=len(post_contacts),
                    post_contacts_before_transport=len(pre),
                    n_contact_events=len(contacts),
                    transport_steps=int(mask.sum()))

    def tune(target, lo, hi, y_shift=0., iters=8):
        """Bisect `peak` for the smallest apex giving min bowl clearance >= target.
        Post contact or insufficient clearance means the apex is too low;
        failure without post contact means the excursion is too aggressive."""
        best = None
        for _ in range(iters):
            mid = (lo + hi) / 2
            r = servo_variant(mid, '_tuning', y_shift)
            print('  tune peak=%.4f shift=%.3f -> %s' % (mid, y_shift, {k: r[k] for k in
                  ('success', 'min_transport_clearance_m', 'post_contact_events',
                   'n_contact_events')}), flush=True)
            if r['post_contact_events'] > 0 or r['min_transport_clearance_m'] < target \
                    or np.isnan(r['min_transport_clearance_m']):
                lo = mid
            elif r['success'] and r['n_contact_events'] == 0 \
                    and r['post_contacts_before_transport'] == 0:
                hi, best = mid, r
            else:
                hi = mid
        return best or servo_variant(hi, '_tuning', y_shift)

    mid_y = 0.04  # corridor bowl-y; +y excursion apex is parameterized above it
    narrow = tune(args.narrow_clearance_m, mid_y, mid_y + 0.15)
    target_w = max(args.wide_clearance_m, narrow['min_transport_clearance_m'] + .020)
    wide = tune(target_w, narrow['peak'], mid_y + 0.30,
                y_shift=args.wide_end_shift_m)
    results = {'narrow': servo_variant(narrow['peak'], 'narrow_demo'),
               'wide': servo_variant(wide['peak'], 'wide_demo',
                                     args.wide_end_shift_m)}
    validation.update(results)
    valid = (results['narrow']['success'] and results['wide']['success']
             and results['narrow']['post_contact_events'] == 0
             and results['wide']['post_contact_events'] == 0
             and results['narrow']['min_transport_clearance_m'] >= .003
             and results['wide']['min_transport_clearance_m']
             >= results['narrow']['min_transport_clearance_m'] + .020)
    validation['valid'] = valid
    validation['note'] = 'both demos must be successful and collision-free; wide clearance must exceed narrow by >=20mm'
    (out / 'demo_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2), flush=True)
    for junk in out.glob('_tuning_*'):
        junk.unlink()
    scene.close()


def eval_obstacle(out):
    assert json.loads((out / 'demo_validation.json').read_text())['valid']
    import scene as clearance_scene
    from recording import Recorder
    from scene import Scene
    obstacle = np.array(json.loads((out / 'demo_validation.json').read_text())['obstacle_xy'])
    clearance_scene.LAYOUTS['post'] = obstacle
    import subprocess
    subprocess.run([sys.executable, str(BPP_ROOT / 'scripts/check_cuda.py')], check=True)
    device = 'cuda:0'
    model, cfg, metadata = load_policy(CHECKPOINT, device, args.threads)
    prompts = {'narrow': load_prompt(out / 'narrow_demo.hdf5', cfg.task.shape_meta, device),
               'wide': load_prompt(out / 'wide_demo.hdf5', cfg.task.shape_meta, device)}
    states = [source_demo(i) for i in range(args.trials)]
    unique = {hashlib.sha256(s.tobytes()).hexdigest() for s, _, _ in states}
    assert len(unique) == args.trials
    config = {'checkpoint': str(CHECKPOINT), 'checkpoint_load': metadata,
              'prompt_paths': {'narrow': str(out / 'narrow_demo.hdf5'),
                               'wide': str(out / 'wide_demo.hdf5')},
              'obstacle_xy': obstacle.tolist(), 'max_steps': args.max_steps,
              'warmup_open_steps': WARMUP, 'diffusion_seeds': list(range(args.trials)),
              'query_state_source': 'source demo_0..9 flattened initial states; 10 distinct sha256',
              'experiment': 'obstacle_clearance_native_scene'}
    (out / 'evaluation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    scene = Scene(record_images=True)
    inverse = RotationTransformer('axis_angle', 'rotation_6d')
    results = []
    try:
        for i in range(args.trials):
            state, _, reset_seed = source_demo(i)
            pair_identity = None
            for label, prompt in prompts.items():
                scene.canonical_state = state
                obs = scene.reset('post')
                history = [obs]
                for _ in range(WARMUP):
                    obs, _, _, _ = scene.env.step(np.r_[np.zeros(6), -1.])
                    history.append(obs)
                    history = history[-2:]
                identity = {'sim_state_sha256': hashlib.sha256(
                                scene.env.sim.get_state().flatten().tobytes()).hexdigest(),
                            'agentview_rgb_sha256': hashlib.sha256(
                                obs['agentview_image'].tobytes()).hexdigest(),
                            'eef_pos': obs['robot0_eef_pos'].tolist(),
                            'obstacle_xy': obstacle[:2].tolist()}
                if pair_identity is None:
                    pair_identity = identity
                assert identity == pair_identity, 'paired query state drifted'
                model.reset(action_exec_horizon=12)
                with torch.inference_mode():
                    model.prompt(prompt)
                torch.manual_seed(i)
                torch.cuda.manual_seed_all(i)
                name = f'rollout_seed{i}_{label}'
                rec = Recorder(out, name, save_images=True)
                rec.start(scene)
                initial_bowl = scene.object_pos()
                goal = scene.goal_pos()
                started = time.monotonic()
                try:
                    while len(rec.actions) < args.max_steps:
                        query = query_observations(history, cfg.task.shape_meta, device)
                        with torch.inference_mode():
                            pred = model.predict_action(query)['action'][0, :12].cpu().numpy()
                        actions = np.concatenate([pred[:, :3], inverse.inverse(pred[:, 3:9]),
                                                  pred[:, 9:]], axis=-1)
                        for a in actions:
                            bowl = scene.object_pos()
                            rec.phase = phase_label(bowl, initial_bowl, goal[:2])
                            rec.step(a)
                            history.append(rec.latest_obs)
                            history = history[-2:]
                            if scene.env._check_success() or len(rec.actions) >= args.max_steps:
                                break
                        if scene.env._check_success():
                            break
                finally:
                    summary = rec.finish(scene)
                traj = np.genfromtxt(out / (name + '_trajectory.csv'), delimiter=',',
                                     names=True, dtype=None, encoding='utf8')
                events = json.loads((out / (name + '_contacts.json')).read_text())
                flat = [{'control_step': ev['control_step'], 'phase': ev['phase'],
                         'geom1': c[0], 'geom2': c[1], 'dist': c[2]}
                        for ev in events for c in ev['contacts']]
                contact_summary = summarize_contacts(flat)
                transport = traj['phase'] == 'transport'
                eef = np.stack([traj[k] for k in ('eef_x', 'eef_y', 'eef_z')], axis=1)
                bowl_arr = np.stack([traj[k] for k in ('object_x', 'object_y', 'object_z')], axis=1)
                speed, acc, jerk = dynamics(eef[transport]) if transport.sum() > 3 else (np.zeros(1),) * 3
                rows_like = [{'bowl_x': r[0], 'bowl_y': r[1], 'bowl_z': r[2],
                              'goal_distance_xy': float(np.linalg.norm(r[:2] - goal[:2])),
                              'goal_success': 0} for r in bowl_arr]
                result = dict(seed=i, prompt=label, name=name,
                              success=bool(summary['goal_success']),
                              steps=int(summary['control_steps']),
                              min_transport_clearance_m=float(
                                  traj['object_clearance_m'][transport].min()) if transport.any() else None,
                              min_gripper_clearance_m=float(
                                  traj['gripper_clearance_m'][transport].min()) if transport.any() else None,
                              transport_steps=int(transport.sum()),
                              transport_duration_s=float(transport.sum()) * DT,
                              eef_speed_mean_m_s=float(speed.mean()),
                              eef_acc_peak_m_s2=float(acc.max()),
                              eef_jerk_peak_m_s3=float(jerk.max()),
                              initial_identity=identity,
                              wall_seconds=time.monotonic() - started,
                              **contact_summary)
                result.update(milestones(rows_like, initial_bowl, goal))
                result['failure_stage'] = failure_stage(result)
                results.append(result)
                print(json.dumps({k: result[k] for k in
                                  ('seed', 'prompt', 'success', 'steps', 'failure_stage',
                                   'min_transport_clearance_m', 'unintended_collision')},
                                 default=str), flush=True)
                (out / 'rollout_results.json').write_text(
                    json.dumps(results, indent=2, default=str) + '\n')
    finally:
        scene.close()
    save_csv(out / 'rollout_results.csv',
             [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
               for k, v in r.items()} for r in results])
    labels = list(prompts)
    paired = []
    for i in range(args.trials):
        pair = {r['prompt']: r for r in results if r['seed'] == i}
        row = {'seed': i}
        for label in labels:
            r = pair[label]
            for k in ('success', 'lifted', 'transported', 'near_goal', 'steps',
                      'failure_stage', 'min_transport_clearance_m',
                      'min_gripper_clearance_m', 'transport_duration_s',
                      'unintended_collision', 'any_contact'):
                row[f'{label}_{k}'] = r[k]
        paired.append(row)
    save_csv(out / 'paired_results.csv', paired)
    (out / 'paired_results.json').write_text(json.dumps(paired, indent=2, default=str) + '\n')
    summary = {label: {k: sum(bool(r[k]) for r in results if r['prompt'] == label)
                       for k in ('success', 'lifted', 'transported', 'near_goal',
                                 'unintended_collision')} for label in labels}
    (out / 'paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'gen_gentle':
        gen_gentle(out)
    elif args.stage == 'eval_gentle':
        eval_gentle(out)
    elif args.stage == 'gen_obstacle':
        gen_obstacle(out)
    elif args.stage == 'eval_obstacle':
        eval_obstacle(out)


if __name__ == '__main__':
    main()
