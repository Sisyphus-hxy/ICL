"""2x2 crossed prompt-query diagnostic over the procedural-constraint demos.

Task A (cup):    query visual {empty, filled} x prompt {normal, gentle}
Task B (post):   query visual {plastic, glass} x prompt {narrow, wide}

Per task class: 5 layouts x 2 diffusion seeds x 2 query visuals x 2 prompts
= 40 rollouts (20 same-query paired prompt comparisons). Identical post-warmup
sim state, history and seed inside each (layout, seed, visual) prompt pair.
Cup Filled adds real free-sphere contents to the query bowl — the visual axis
is therefore a genuine physical difference (recorded via n_drops_out);
cross-visual equality is checked on the native (non-drop) state portion.

Prompts are demos of a DIFFERENT layout than the query (cross-layout pairing).
No training, no checkpoint changes. Reuses inference.py policy loading and
the paired_rollout convention from meaningful_swap/run.py.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['eval_cup', 'eval_obstacle',
                                      'eval_obstacle_grasped',
                                      'teacher_cup', 'teacher_obstacle'])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--gpu', default='GPU-e0914423-1b44-589b-fdb3-0360e8f79b43')
parser.add_argument('--egl-device', default='0')
parser.add_argument('--max-steps', type=int, default=550)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--seeds', default='1,2')
parser.add_argument('--n-drops', type=int, default=16,
                    help='spill spheres inside the Filled cup query bowl')
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
# cache paths from scripts/common.sh — the CLIP/HF caches live under the repo
_BPP_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('BPP_ROOT', str(_BPP_ROOT))
os.environ.setdefault('HF_HOME', str(_BPP_ROOT / '.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(_BPP_ROOT / '.cache/torch'))
os.environ.setdefault('XDG_CACHE_HOME', str(_BPP_ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(_BPP_ROOT / '.cache/matplotlib'))
os.environ.setdefault('TMPDIR', str(_BPP_ROOT / '.cache/tmp'))
os.environ.setdefault('LIBERO_CONFIG_PATH', str(_BPP_ROOT / 'config/libero'))
os.environ.setdefault('WANDB_MODE', 'offline')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'clearance'))
sys.path.insert(0, str(HERE))

import h5py
import numpy as np
import torch

from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
from scene import BPP_ROOT
from inference import load_policy, load_prompt, query_observations
from scene_lib import SemanticScene, MATERIALS, SOURCE


# --- helpers copied from generate.py (its module-level argparse makes it
# unimportable); keep implementations identical ---

def demo_info(i):
    with h5py.File(SOURCE) as f:
        d = f[f'data/demo_{i}']
        return dict(state=d['states'][0].copy(), states=d['states'][:],
                    actions=d['actions'][:], ee_pos=d['obs/ee_pos'][:],
                    ee_ori=d['obs/ee_ori'][:],
                    seed=int(d.attrs.get('seed', 2)))


def find_windows(scene, info):
    """Transport window [t0, t1) and release index from the bowl-z profile.
    Parks any mocap post under the table first so it cannot perturb the bowl."""
    n = len(info['actions'])
    parked = None
    if scene.obstacle_id is not None:
        parked = scene.d.mocap_pos[scene.mocap_id].copy()
        scene.d.mocap_pos[scene.mocap_id] = [0., 0., 0.5]
    bowlz = np.empty(n)
    for k in range(n):
        scene.set_state(info['states'][k])
        scene.env.sim.forward()
        bowlz[k] = scene.object_pos()[2]
    if parked is not None:
        scene.d.mocap_pos[scene.mocap_id] = parked
    rel_idx = np.nonzero(np.diff(info['actions'][:, 6]) < -0.5)[0]
    rel = int(rel_idx[0]) + 1 if len(rel_idx) else n - 12
    peak_z = float(bowlz.max())
    high = np.nonzero(bowlz > peak_z - 0.02)[0]
    return dict(t0=int(high[0]), t1=int(high[-1]) + 1, rel=rel,
                rest_z=float(bowlz[0]), peak_z=peak_z, n=n)


def tilt_deg(quats):
    """Vessel tilt from level: angle between bowl-up (-eef z) and world +z."""
    from scipy.spatial.transform import Rotation
    z = Rotation.from_quat(np.asarray(quats)).as_matrix()[:, :, 2]
    return np.degrees(np.arccos(np.clip(-z[:, 2], -1., 1.)))

CHECKPOINT = BPP_ROOT / 'checkpoints/liberogen_spatial_combination_behavior_prompting.ckpt'
DEMOS = HERE / 'outputs'
WARMUP = 12
DT = 1.0 / 20.0

TASKS = {
    'cup': dict(layouts=[0, 1, 2, 3, 4],
                visuals=['empty', 'filled'],
                prompts=['normal', 'gentle'],
                # leaf dir of the generated dataset: procedure name = file
                # (normal<->empty bowl demo, gentle<->filled bowl demo)
                prompt_file='cup/layout_{j:02d}/{p}.hdf5',
                prompt_cycle=[1, 2, 3, 4, 0]),
    'obstacle': dict(layouts=[0, 3, 5, 8, 11],
                     visuals=['plastic', 'glass'],
                     prompts=['narrow', 'wide'],
                     prompt_file='obstacle/layout_{j:02d}/{p}.hdf5',
                     prompt_cycle=[3, 5, 8, 11, 0]),
}


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_scene(task, visual):
    if task == 'cup':
        # Filled = a bowl heaped with free blue spill spheres (real dynamics,
        # parked outside for Empty). No water disk.
        s = SemanticScene(record_images=True, n_drops=args.n_drops)
        s.drops_in_bowl = (visual == 'filled')
        return s
    return SemanticScene(record_images=True, post_rgba=MATERIALS[visual])


def reset_scene(scene, state, reset_seed):
    scene.reference_seed = reset_seed
    scene.canonical_state = state
    return scene.reset()


def rollout(scene, model, cfg, prompt, state, reset_seed, diff_seed,
            name, out, device, grasped=False, rest_z=None):
    """Reset + 12-step warmup, then closed-loop policy control. Records
    kinematics, eef tilt, post clearance and contacts. With grasped=True the
    start state already holds the bowl (post-grasp probe): no open-gripper
    warmup; history is the two identical post-reset frames."""
    e = scene.env
    reset_scene(scene, state, reset_seed)
    obs = scene.obs()
    history = [obs]
    if grasped:
        history = [obs.copy(), obs.copy()]
    else:
        for _ in range(WARMUP):
            obs, _, _, _ = e.step(np.r_[np.zeros(6), -1.])
            scene.update_water()
            history.append(obs)
            history = history[-2:]
    identity = {
        'sim_state_sha256': hashlib.sha256(
            scene.native_flat().tobytes()).hexdigest(),
        'agentview_rgb_sha256': hashlib.sha256(
            obs['agentview_image'].tobytes()).hexdigest(),
        'eef_pos': obs['robot0_eef_pos'].tolist(),
        'gripper_command': e.robots[0].gripper.current_action.tolist()}
    model.reset(action_exec_horizon=12)
    with torch.inference_mode():
        model.prompt(prompt)
    torch.manual_seed(diff_seed)
    torch.cuda.manual_seed_all(diff_seed)
    events, substep_min = [], {'object': 2., 'gripper': 2.}
    mon = {'step': 0, 'phase': 'approach'}
    original_step = e.sim.step

    def monitored():
        original_step()
        scene.update_water()
        for key, geoms in (('object', scene.object_geoms),
                           ('gripper', scene.gripper_geoms)):
            substep_min[key] = min(substep_min[key], scene.surface_clearance(geoms))
        for c in scene.contacts(transport=mon['phase'] == 'transport'):
            events.append({'control_step': mon['step'], 'phase': mon['phase'],
                           'contact': c})

    e.sim.step = monitored
    inverse = RotationTransformer('axis_angle', 'rotation_6d')
    leaf = out / name
    leaf.parent.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder.create_h264(fps=20, thread_count=1)
    video.start(str(leaf.parent / (leaf.name + '.mp4')))
    rows = []
    initial_bowl = scene.object_pos()
    base_z = rest_z if rest_z is not None else initial_bowl[2]
    goal = scene.goal_pos()
    started = time.monotonic()

    def phase(bowl):
        if bowl[2] <= base_z + .035:
            return 'grasp'
        return 'transport' if np.linalg.norm(bowl[:2] - goal[:2]) > .045 else 'place'

    try:
        while len(rows) < args.max_steps:
            query = query_observations(history, cfg.task.shape_meta, device)
            with torch.inference_mode():
                pred = model.predict_action(query)['action'][0, :12].cpu().numpy()
            actions = np.concatenate([pred[:, :3], inverse.inverse(pred[:, 3:9]),
                                      pred[:, 9:]], axis=-1)
            for a in actions:
                video.write_frame(np.concatenate(
                    [history[-1]['agentview_image'][::-1],
                     history[-1]['robot0_eye_in_hand_image'][::-1]], axis=1).copy())
                substep_min['object'] = 2.
                substep_min['gripper'] = 2.
                mon['step'] = len(rows)
                obs, _, _, _ = e.step(a)
                scene.update_water()
                history.append(obs)
                history = history[-2:]
                bowl = scene.object_pos()
                mon['phase'] = phase(bowl)
                rows.append(dict(
                    step=len(rows), phase=mon['phase'],
                    eef_x=float(obs['robot0_eef_pos'][0]),
                    eef_y=float(obs['robot0_eef_pos'][1]),
                    eef_z=float(obs['robot0_eef_pos'][2]),
                    eef_qx=float(obs['robot0_eef_quat'][0]),
                    eef_qy=float(obs['robot0_eef_quat'][1]),
                    eef_qz=float(obs['robot0_eef_quat'][2]),
                    eef_qw=float(obs['robot0_eef_quat'][3]),
                    bowl_x=float(bowl[0]), bowl_y=float(bowl[1]),
                    bowl_z=float(bowl[2]),
                    object_clearance_m=float(substep_min['object']),
                    gripper_clearance_m=float(substep_min['gripper']),
                    goal_distance_xy=float(np.linalg.norm(bowl[:2] - goal[:2])),
                    n_drops_out=int(scene.spill_count()),
                    goal_success=int(e._check_success())))
                if rows[-1]['goal_success']:
                    break
            if rows[-1]['goal_success']:
                break
    finally:
        video.stop()
        e.sim.step = original_step
    save_csv(leaf.parent / (leaf.name + '_trajectory.csv'), rows)
    (leaf.parent / (leaf.name + '_contacts.json')).write_text(
        json.dumps(events, indent=2))
    return dict(rows=rows, events=events, identity=identity,
                grasped=grasped, wall_seconds=time.monotonic() - started)


def metrics(r):
    rows = r['rows']
    eef = np.array([[row['eef_x'], row['eef_y'], row['eef_z']] for row in rows])
    bowl = np.array([[row['bowl_x'], row['bowl_y'], row['bowl_z']] for row in rows])
    tr = np.asarray([row['phase'] == 'transport' for row in rows])
    seg = eef[tr]
    v = np.linalg.norm(np.diff(seg, axis=0), axis=1) / DT if len(seg) > 2 else np.zeros(1)
    a = np.diff(v) / DT if len(v) > 1 else np.zeros(1)
    j = np.diff(a) / DT if len(a) > 1 else np.zeros(1)
    quats = np.array([[row['eef_qx'], row['eef_qy'], row['eef_qz'], row['eef_qw']]
                      for row in rows])
    tilts = tilt_deg(quats[tr]) if tr.any() else None
    clr = [row['object_clearance_m'] for row in rows
           if row['phase'] == 'transport' and row['object_clearance_m'] < 2.]
    post_ev = [ev for ev in r['events']
               if 'procedure_obstacle' in str(ev['contact'])]
    m = dict(success=bool(rows[-1]['goal_success']), steps=len(rows),
             lifted=True if r.get('grasped') else
             bool((bowl[:, 2] > bowl[0, 2] + .045).any()),
             transported=bool((np.linalg.norm(bowl[:, :2] - bowl[0, :2],
                                              axis=1) > .035).any()),
             near_goal=bool(min(row['goal_distance_xy'] for row in rows) < .055),
             transport_steps=int(tr.sum()),
             transport_duration_s=float(tr.sum()) * DT,
             transport_path_length_m=float(
                 np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else None,
             eef_speed_mean_m_s=float(v.mean()),
             eef_speed_p95_m_s=float(np.percentile(v, 95)),
             eef_acc_p95_m_s2=float(np.percentile(np.abs(a), 95)) if len(a) else None,
             eef_jerk_p95_m_s3=float(np.percentile(np.abs(j), 95)) if len(j) else None,
             tilt_mean_deg=float(tilts.mean()) if tilts is not None else None,
             tilt_max_deg=float(tilts.max()) if tilts is not None else None,
             transport_min_clearance_m=float(min(clr)) if clr else None,
             n_contact_events=len(r['events']),
             post_contact_events=len(post_ev),
             n_drops_out_max=int(max(row['n_drops_out'] for row in rows)),
             n_drops_out_end=int(rows[-1]['n_drops_out']),
             first_spill_step=(min((row['step'] for row in rows
                                    if row['n_drops_out'] > 0), default=None)),
             first_post_contact=(min((e['control_step'] for e in post_ev),
                                     default=None)))
    for stage in ('lifted', 'transported', 'near_goal'):
        if not m[stage] and not m['success']:
            m['failure_stage'] = stage
            break
    else:
        m['failure_stage'] = 'none' if m['success'] else 'goal_placement'
    return m


def layout_posts(task, layouts):
    if task != 'obstacle':
        return {}
    posts = {}
    for layout in layouts:
        with h5py.File(DEMOS / TASKS[task]['prompt_file'].format(
                j=layout, p='narrow')) as f:
            v = f['data/demo_0'].attrs['post_xyz']
            posts[layout] = np.asarray(json.loads(v) if isinstance(v, str) else v,
                                       dtype=float)
    return posts


def evaluate(task, grasped=False):
    import subprocess
    subprocess.run([sys.executable, str(BPP_ROOT / 'scripts/check_cuda.py')], check=True)
    device = 'cuda:0'
    model, cfg, metadata = load_policy(CHECKPOINT, device, args.threads)
    T = TASKS[task]
    seeds = [int(s) for s in args.seeds.split(',')]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    layouts = T['layouts']
    loaded = {}
    prompt_paths = {}
    for li, layout in enumerate(layouts):
        j = T['prompt_cycle'][li]
        for pname in T['prompts']:
            path = DEMOS / T['prompt_file'].format(j=j, p=pname)
            assert path.exists(), f'missing prompt demo: {path}'
            prompt_paths[(layout, pname)] = str(path)
            loaded[(layout, pname)] = load_prompt(path, cfg.task.shape_meta, device)
    scenes = {v: make_scene(task, v) for v in T['visuals']}
    posts = layout_posts(task, layouts)
    hash_rows = []
    config = dict(task=task, checkpoint=str(CHECKPOINT),
                  checkpoint_load=metadata, diffusion_seeds=seeds, layouts=layouts,
                  prompt_cycle=T['prompt_cycle'], prompt_paths={str(k): v for k, v in prompt_paths.items()},
                  max_steps=args.max_steps, warmup=WARMUP,
                  start_state='post_grasp_t0' if grasped else 'initial',
                  note='seeds 1-2 are native-baseline successes (maximize '
                       'transport coverage); failure rates still reported')
    (out / 'evaluation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    results = []
    try:
        for layout in layouts:
            info = demo_info(layout)
            start_state, rest_z, t0 = info['state'], None, None
            if grasped:
                w_ = find_windows(scenes[T['visuals'][0]], info)
                start_state = info['states'][w_['t0']]
                rest_z, t0 = w_['rest_z'], int(w_['t0'])
            layout_hashes = set()
            for visual in T['visuals']:
                scene = scenes[visual]
                if task == 'obstacle':
                    scene.post_pos = posts[layout]
                    scene.material = visual
                for diff_seed in seeds:
                    group_identity = None
                    for pname in T['prompts']:
                        name = (f'{out.name}/layout_{layout:02d}/{visual}/'
                                f'seed{diff_seed}/{pname}')
                        r = rollout(scene, model, cfg, loaded[(layout, pname)],
                                    start_state, info['seed'], diff_seed,
                                    name, out, device, grasped=grasped,
                                    rest_z=rest_z)
                        m = metrics(r)
                        key = r['identity']['sim_state_sha256']
                        if group_identity is None:
                            group_identity = key
                        assert key == group_identity, 'paired query state drifted'
                        layout_hashes.add(key)
                        hash_rows.append(dict(
                            layout=layout, visual=visual, diff_seed=diff_seed,
                            sim_state_sha256=key,
                            agentview_rgb_sha256=r['identity']['agentview_rgb_sha256']))
                        results.append(dict(name=name, task=task, layout=layout,
                                            visual=visual, diff_seed=diff_seed,
                                            prompt=pname,
                                            prompt_layout=T['prompt_cycle'][layouts.index(layout)],
                                            start_t0=t0,
                                            sim_state_sha256=key, **m))
                        print(json.dumps(dict(name=name, success=m['success'],
                                              steps=m['steps'],
                                              failure_stage=m['failure_stage'],
                                              speed=m['eef_speed_mean_m_s'],
                                              tilt=m['tilt_mean_deg'],
                                              clr=m['transport_min_clearance_m']),
                                         default=str), flush=True)
                        (out / 'rollout_results.json').write_text(
                            json.dumps(results, indent=2, default=str) + '\n')
            # the physics state must be identical across visual variants of a
            # layout (mocap visuals are not part of the flattened sim state)
            assert len(layout_hashes) == 1, \
                f'layout {layout}: visual variants drifted apart physically'
    finally:
        for s in scenes.values():
            s.close()
    save_csv(out / 'rollout_results.csv',
             [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
               for k, v in r.items()} for r in results])
    save_csv(out / 'state_hashes.csv', hash_rows)
    distinct = {r['sim_state_sha256'] for r in hash_rows}
    print(json.dumps({'distinct_query_states': len(distinct),
                      'expected_layouts': len(layouts)}, indent=2))


# ---------------------------------------------------------------- teacher-forced

def teacher(task):
    """Fixed query obs/history/diffusion-noise; swap only the prompt.
    Probes at post-warmup initial state and at the demo's transport-entry
    state (bowl in hand), per layout and query visual."""
    import subprocess
    subprocess.run([sys.executable, str(BPP_ROOT / 'scripts/check_cuda.py')], check=True)
    device = 'cuda:0'
    model, cfg, metadata = load_policy(CHECKPOINT, device, args.threads)
    T = TASKS[task]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    layouts = T['layouts']
    loaded = {}
    for li, layout in enumerate(layouts):
        j = T['prompt_cycle'][li]
        for pname in T['prompts']:
            loaded[(layout, pname)] = load_prompt(
                DEMOS / T['prompt_file'].format(j=j, p=pname),
                cfg.task.shape_meta, device)
    scenes = {v: make_scene(task, v) for v in T['visuals']}
    posts = layout_posts(task, layouts)
    rows = []
    try:
        for layout in layouts:
            info = demo_info(layout)
            w_ = None
            for visual in T['visuals']:
                scene = scenes[visual]
                if task == 'obstacle':
                    scene.post_pos = posts[layout]
                    scene.material = visual
                e = scene.env
                reset_scene(scene, info['state'], info['seed'])
                obs = scene.obs()
                history = [obs]
                for _ in range(WARMUP):
                    obs, _, _, _ = e.step(np.r_[np.zeros(6), -1.])
                    scene.update_water()
                    history.append(obs)
                    history = history[-2:]
                probes = {'post_warmup': [o.copy() for o in history]}
                if w_ is None:
                    w_ = find_windows(scene, info)
                scene.set_state(info['states'][w_['t0']])
                scene.place_drops()  # re-heap drops at the lifted bowl pose
                e.sim.forward()
                scene.update_water()
                obs2 = scene.obs()
                probes['transport_entry'] = [obs2.copy(), obs2.copy()]
                for probe_name, hist in probes.items():
                    query = query_observations(hist, cfg.task.shape_meta, device)
                    chunks = {}
                    for pname in T['prompts']:
                        model.reset(action_exec_horizon=12)
                        with torch.inference_mode():
                            model.prompt(loaded[(layout, pname)])
                        torch.manual_seed(0)
                        torch.cuda.manual_seed_all(0)
                        with torch.inference_mode():
                            chunks[pname] = model.predict_action(
                                query)['action'][0, :12].cpu().numpy()
                    pa, pb = (chunks[p] for p in T['prompts'])
                    d = pa - pb
                    fa, fb = pa[0], pb[0]
                    rows.append(dict(
                        task=task, layout=layout, visual=visual,
                        probe=probe_name,
                        prompt_a=T['prompts'][0], prompt_b=T['prompts'][1],
                        l2=float(np.linalg.norm(d)),
                        l2_mean=float(np.linalg.norm(d, axis=1).mean()),
                        cosine=float((pa.ravel() * pb.ravel()).sum() /
                                     (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-12)),
                        first_a=fa.tolist(), first_b=fb.tolist(),
                        first_delta=(fb - fa).tolist(),
                        chunk_a=pa.tolist(), chunk_b=pb.tolist()))
                    print(json.dumps(dict(task=task, layout=layout, visual=visual,
                                          probe=probe_name, l2=rows[-1]['l2'],
                                          cosine=rows[-1]['cosine']),
                                     default=str), flush=True)
                    (out / 'teacher_forced.json').write_text(
                        json.dumps(rows, indent=2, default=str) + '\n')
    finally:
        for s in scenes.values():
            s.close()
    save_csv(out / 'teacher_forced.csv',
             [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
               for k, v in r.items()} for r in rows])


if __name__ == '__main__':
    if args.stage == 'eval_obstacle_grasped':
        evaluate('obstacle', grasped=True)
    else:
        task = args.stage.rsplit('_', 1)[-1]
        teacher(task) if args.stage.startswith('teacher') else evaluate(task)
