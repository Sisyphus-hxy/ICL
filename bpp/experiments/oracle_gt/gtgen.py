#!/usr/bin/env python3
"""Generate Oracle-GT demonstrations starting from fixed query initial states.

Upstream ``TaskDemonstrationGenerator`` randomises the initial state per demo
(``env.seed(demo_idx); env.reset()``) and its ``actions_from`` replay resets the
simulator to the *reference* demonstration's initial state. Neither accepts an
externally specified query state. This driver subclasses it to:

* start every trajectory at a given flattened MuJoCo state (the query init),
* script every ``Grasp``/``Place`` stage that is covered by ``actions_from``
  (so movable-object pick-and-places survive arbitrary query layouts), while
  still replaying ``Open``/``Turnon``/``Touch`` fixture stages that have no
  scripted planner — replay now starts from the query state and retries over
  reference demos ranked by initial-state similarity,
* record states/actions/RGB observations during generation and write one
  prompt-loadable HDF5 per query in the exact schema BPP's prompt loader reads
  (``data/demo_0/{obs/{ee_pos,ee_ori,gripper_states,agentview_rgb,
  eye_in_hand_rgb},actions,states,...}``), matching ``create_dataset``
  conventions including the 5-step ``cap_index`` trim.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'src'))
sys.path.insert(0, str(HERE.parent.parent / 'src/behavior_prompting/train_network/scripts/libero'))

import tasks as T

os.environ.setdefault('LIBERO_CONFIG_PATH', str(T.CONFIG_DIR))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')

import h5py
import numpy as np

import generate_demonstrations as gd
from generate_demonstrations import (TaskDemonstrationGenerator,
    set_sim_state_from_flattened_with_update, stage_to_predicate,
    task_name_to_hdf5_path, get_base_object_name)
from robosuite.utils import transform_utils as RT

SCRIPTABLE = {'grasp', 'place'}
CAP_INDEX = 5  # create_dataset.py trims the first 5 recorded steps
SETTLE_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


# The official generator never encounters scripted stages following a Touch
# (it only replays them), so get_end_location_of_stage raises for touch/push.
# A touch ends at the touched object — same convention as grasp/open/turnon.
_orig_end_location = gd.get_end_location_of_stage
def _end_location_of_stage(stage):
    predicate = stage_to_predicate(stage)
    if predicate[0] in ('touch', 'push'):
        return predicate[1]
    return _orig_end_location(stage)
gd.get_end_location_of_stage = _end_location_of_stage


def _find_grasp_ref(task_metadata, object_name):
    """Reference task containing 'Grasp <obj>' whose HDF5 exists locally.

    Prefers the actions_from task, then any other reference task; returns an
    existing-local task before falling back to a (missing) candidate.
    """
    all_metadata = task_metadata.get('all_metadata', {})
    candidates = []
    actions_from = task_metadata.get('actions_from')
    if actions_from is not None:
        candidates.append(actions_from)
    candidates += sorted(all_metadata.keys())
    needle = f'Grasp {object_name}'
    existing, missing = None, None
    for task in candidates:
        steps = all_metadata.get(task, {}).get('execution_steps', [])
        if needle not in steps:
            continue
        if os.path.exists(task_name_to_hdf5_path(task_metadata, task)):
            existing = task
            break
        if missing is None:
            missing = task
    return existing or missing


def _patch_grasps_from(task_metadata):
    """Synthesize grasps_from for scripted Grasp stages missing it.

    The official planner (and its __init__ pose pre-computation) requires
    grasps_from[object] for every Grasp stage it executes — including
    natively scripted ones where the task metadata omits the entry. Replay
    (actions_from_steps) stages never reach the planner and are skipped.
    """
    grasps_from = task_metadata.setdefault('grasps_from', {})
    replayed = set(task_metadata.get('actions_from_steps') or [])
    for stage in task_metadata.get('execution_steps', []):
        predicate = stage_to_predicate(stage)
        if predicate[0] != 'grasp' or predicate[1] in grasps_from:
            continue
        if stage in replayed:
            continue
        ref_task = _find_grasp_ref(task_metadata, predicate[1])
        if ref_task is not None:
            grasps_from[predicate[1]] = {'task': ref_task, 'object': predicate[1]}


class QueryDemoGenerator(TaskDemonstrationGenerator):
    """TaskDemonstrationGenerator that starts from an explicit query state."""

    SCRIPTABLE_VERBS = SCRIPTABLE

    def __init__(self, bddl_file_path, run_dir, resolution=128,
                 script_prefix=True, max_replay_candidates=None,
                 suppress_print=True, **kwargs):
        self.script_prefix = script_prefix
        self.max_replay_candidates = max_replay_candidates or 25
        self.query_state = None
        self.record = {'states': [], 'actions': [], 'obs': []}
        # Force camera obs so the prompt HDF5 gets agentview/eye_in_hand RGB.
        orig_get_env = gd.get_environment
        gd.get_environment = lambda *a, **k: orig_get_env(
            *a, **{**k, 'use_camera_obs': True, 'resolution': resolution})
        # The official __init__ computes relative grasp poses for every
        # scripted stage and dies when grasps_from lacks the object. Patch the
        # metadata loader so synthesized references are present up front.
        orig_load_md = gd.load_metadata_for_task
        def _load_md(*a, **k):
            md = orig_load_md(*a, **k)
            _patch_grasps_from(md)
            return md
        gd.load_metadata_for_task = _load_md
        try:
            super().__init__(bddl_file_path, run_dir=run_dir,
                             resolution=resolution, suppress_print=suppress_print,
                             **kwargs)
        finally:
            gd.get_environment = orig_get_env
            gd.load_metadata_for_task = orig_load_md
        self._install_step_recorder()
        self.newly_scripted_stages = []
        if self.script_prefix and self.actions_from is not None:
            self._rescript_prefix()
        self._ensure_grasps_from()

    def _ensure_grasps_from(self):
        """Re-run the grasps_from patch after _rescript_prefix.

        _rescript_prefix may move stages out of actions_from_steps into the
        scripted set; those need references even though the loader-time patch
        skipped them as replayed.
        """
        self.task_metadata['actions_from_steps'] = self.actions_from_steps
        _patch_grasps_from(self.task_metadata)

    # ------------------------------------------------------------------ setup
    def _install_step_recorder(self):
        """Capture (pre-state, action, post-obs) for every simulator step."""
        env = self.env
        orig_step = env.step
        rec = self.record

        def recording_step(action):
            rec['states'].append(env.sim.get_state().flatten().copy())
            result = orig_step(action)
            rec['actions'].append(np.asarray(action, dtype=np.float32).copy())
            obs = result[0]
            rec['obs'].append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                               for k, v in obs.items()} if isinstance(obs, dict) else obs)
            return result
        env.step = recording_step

    def _reset_recorder(self):
        for key in self.record:
            self.record[key].clear()

    def _grasp_reference_task(self, object_name):
        """Find a reference task whose execution_steps contain 'Grasp <obj>'

        and whose demonstration HDF5 exists locally. Prefers the original
        actions_from task, then any other reference task with the stage.
        """
        return _find_grasp_ref(self.task_metadata, object_name)

    def _stage_scriptable(self, stage):
        predicate = stage_to_predicate(stage)
        if predicate[0] not in SCRIPTABLE:
            return False
        if predicate[0] == 'place':
            return True  # place poses are computed analytically
        object_name = predicate[1]
        grasps_from = self.task_metadata.get('grasps_from', {})
        if object_name in grasps_from:
            return True
        return self._grasp_reference_task(object_name) is not None

    def _rescript_prefix(self):
        """Move scriptable actions_from steps into the scripted stage set.

        ``actions_from_steps`` must stay a prefix of ``execution_steps``; we
        therefore only re-script the suffix after the last non-scriptable step.
        """
        af_steps = list(self.actions_from_steps)
        last_forced = -1
        for i, stage in enumerate(af_steps):
            if not self._stage_scriptable(stage):
                last_forced = i
        keep = af_steps[:last_forced + 1]
        rescripted = af_steps[len(keep):]
        if not rescripted:
            return
        grasps_from = self.task_metadata.setdefault('grasps_from', {})
        for stage in rescripted:
            predicate = stage_to_predicate(stage)
            if predicate[0] == 'grasp' and predicate[1] not in grasps_from:
                ref_task = self._grasp_reference_task(predicate[1])
                assert ref_task is not None, f'no grasp reference for {stage}'
                grasps_from[predicate[1]] = {'task': ref_task, 'object': predicate[1]}
                self._print(f' - synthesized grasps_from[{predicate[1]}] -> {ref_task}')
        self.actions_from_steps = keep
        if not keep:
            self._print(f' - all actions_from steps scriptable; disabling replay '
                        f'(was {self.actions_from})')
            self.actions_from = None
            self.actions_from_hdf5_path = None
        for stage in rescripted:
            self.relative_grasp_locations[stage] = \
                self.get_rel_robot_pose_from_teleop_data(stage)
            self.newly_scripted_stages.append(stage)

    # ---------------------------------------------------------------- replay
    def _rank_reference_demos(self, f):
        """Rank reference demos by initial-state similarity to the query."""
        keys = sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1]))
        try:
            q = self.query_state
            scored = []
            for k in keys:
                s = f['data'][k]['states'][0]
                if s.shape == q.shape:
                    scored.append((float(np.linalg.norm(s - q)), k))
                else:
                    scored.append((float('inf'), k))
            scored.sort()
            order = [k for _, k in scored]
        except Exception:
            order = keys
        return order[:self.max_replay_candidates]

    def copy_actions_for_query(self):
        """Replay actions_from stages starting from the query state.

        Unlike ``copy_actions_from_hdf5`` this does *not* reset to the
        reference demo's initial state; it retries over reference demos ranked
        by how close their initial state is to the query.
        """
        assert self.actions_from_hdf5_path is not None
        zero = SETTLE_ACTION
        with h5py.File(self.actions_from_hdf5_path, 'r') as f:
            candidates = self._rank_reference_demos(f)
            actions_by_demo = {k: f['data'][k]['actions'][:] for k in candidates}
        for demo_key in candidates:
            actions = actions_by_demo[demo_key]
            set_sim_state_from_flattened_with_update(self.env, self.query_state)
            self.env._start_new_episode()
            for _ in range(5):
                self.env.step(zero)
            current_stage_i = 0
            step = 0
            ok = False
            while step < len(actions):
                self.env.step(actions[step])
                current_stage = self.actions_from_steps[current_stage_i]
                predicate = stage_to_predicate(current_stage)
                if predicate[0] == 'touch':
                    current_stage_i += 1
                    step += 1
                    if current_stage_i == len(self.actions_from_steps):
                        # Official code assumes a stage follows Touch; chain
                        # tasks can end with one. Touch/push predicates are
                        # not evaluable (eval_predicate_fn asserts), so finish
                        # the remaining actions then accept — downstream
                        # scripted stages and the final goal still gate
                        # success.
                        while step < len(actions):
                            self.env.step(actions[step])
                            step += 1
                        ok = True
                        break
                elif self.check_predicate_success(self.env, current_stage):
                    current_stage_i += 1
                    if current_stage_i == len(self.actions_from_steps):
                        ok = True
                        break
                    step += 1
                else:
                    step += 1
            if ok:
                self._print(f' - replay of {len(self.actions_from_steps)} stage(s) '
                            f'succeeded from query state using {demo_key}')
                return True
            self._print(f' - replay via {demo_key} failed; trying next candidate')
            self._mark_current_demo_as_failed()
            self._reset_recorder()
        return False

    # ------------------------------------------------------------- generation
    def generate_demo_from_state(self, query_state, seed=0, pose_idx=0):
        """Generate one demo starting exactly at ``query_state``.

        ``pose_idx`` selects among collected reference grasp poses — vary it
        across retries to get genuinely different attempts.
        """
        self.query_state = np.asarray(query_state)
        self._reset_recorder()
        seed_used = seed
        for _ in range(10):
            try:
                np.random.seed(seed_used)
                self.env.seed(seed_used)
                self.env.reset()
                break
            except Exception:
                seed_used += 1
        else:
            return False, 'reset_failed'
        set_sim_state_from_flattened_with_update(self.env, self.query_state)

        final_step = self.execution_steps[-1]
        if self.check_predicate_success(self.env, final_step):
            return False, 'already_satisfied'

        if self.actions_from is not None:
            assert self.execution_steps[:len(self.actions_from_steps)] == \
                self.actions_from_steps
            if not self.copy_actions_for_query():
                return False, 'replay_failed'
            remaining = self.execution_steps[len(self.actions_from_steps):]
            last_stage = self.execution_steps[len(self.actions_from_steps) - 1]
        else:
            self.env._start_new_episode()
            remaining = self.execution_steps
            last_stage = None

        for i, cur_stage in enumerate(remaining):
            prev = last_stage if i == 0 else remaining[i - 1]
            self._print(f'Stage {i+1}/{len(remaining)}: {cur_stage}')
            if not self.complete_stage(prev, cur_stage, demo_idx=pose_idx):
                return False, f'stage_failed:{cur_stage}'

        found_done = False
        for _ in range(10):
            _, _, done, _ = self.env.step(SETTLE_ACTION)
            if done:
                found_done = True
                break
        if not found_done:
            return False, 'settle_failed'
        for _ in range(20):
            _, _, done, _ = self.env.step(SETTLE_ACTION)
        if not done:
            return False, 'padding_failed'

        for step in self.execution_steps:
            predicate = stage_to_predicate(step)
            if predicate[0] != 'place':
                continue
            placed_object_name, place_location_name = predicate[1], predicate[2]
            placed_base = get_base_object_name(placed_object_name)
            loc_base = get_base_object_name(place_location_name)
            if loc_base in self.skip_fall_check_when_placed_on.get(placed_base, []):
                continue
            check_name = (place_location_name if loc_base in
                          self.check_place_location_fall_instead_of_placed_object
                          else placed_object_name)
            try:
                pose = gd.get_object_pose(self.env, check_name)
            except (KeyError, AttributeError):
                continue
            z_axis = RT.quat2mat(pose[3:7])[:, 2]
            cos_t = np.clip(np.dot(z_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
            if np.arccos(cos_t) > self.object_fall_angle_threshold_rad:
                return False, f'fall_check:{check_name}'
        return True, 'ok'

    # ---------------------------------------------------------------- output
    def write_prompt_hdf5(self, path):
        """Write the recorded trajectory in BPP prompt HDF5 format."""
        rec = self.record
        actions = np.asarray(rec['actions'])[CAP_INDEX:]
        obs_seq = rec['obs'][CAP_INDEX:]
        states = np.asarray(rec['states'])[CAP_INDEX:]
        assert len(actions) == len(obs_seq) == len(states) and len(actions) > 0
        ee_ori = np.stack([RT.quat2axisangle(o['robot0_eef_quat']) for o in obs_seq])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        unwrapped = self.env.env if hasattr(self.env, 'env') else self.env
        env_info = getattr(self, 'env_info', None) or {}
        with h5py.File(path, 'w') as f:
            data = f.create_group('data')
            data.attrs['env_info'] = json.dumps(env_info, default=str)
            data.attrs['problem_info'] = json.dumps(
                getattr(self, 'problem_info', {}), default=str)
            g = data.create_group('demo_0')
            g.attrs['model_file'] = unwrapped.sim.model.get_xml()
            g.attrs['seed'] = -1
            g.attrs['num_samples'] = len(actions)
            g.attrs['observation_alignment'] = 'post_action'
            g.attrs['state_alignment'] = 'pre_action'
            g.create_dataset('actions', data=actions)
            g.create_dataset('states', data=states)
            g.create_dataset('terminal_state',
                             data=self.env.sim.get_state().flatten())
            g.create_dataset('init_state', data=states[0])
            rewards = np.zeros(len(actions), dtype=np.uint8); rewards[-1] = 1
            g.create_dataset('rewards', data=rewards)
            g.create_dataset('dones', data=rewards.copy())
            obs = g.create_group('obs')
            obs.create_dataset('ee_pos',
                               data=np.stack([o['robot0_eef_pos'] for o in obs_seq]))
            obs.create_dataset('ee_ori', data=ee_ori)
            obs.create_dataset('ee_states',
                               data=np.concatenate([
                                   np.stack([o['robot0_eef_pos'] for o in obs_seq]),
                                   ee_ori], axis=-1))
            obs.create_dataset('gripper_states',
                               data=np.stack([o['robot0_gripper_qpos'] for o in obs_seq]))
            if 'robot0_joint_pos' in obs_seq[0]:
                obs.create_dataset('joint_states',
                                   data=np.stack([o['robot0_joint_pos'] for o in obs_seq]))
            obs.create_dataset('agentview_rgb',
                               data=np.stack([o['agentview_image'] for o in obs_seq]),
                               compression='gzip', compression_opts=1)
            obs.create_dataset('eye_in_hand_rgb',
                               data=np.stack([o['robot0_eye_in_hand_image'] for o in obs_seq]),
                               compression='gzip', compression_opts=1)
        return path


def generate_task(benchmark, task, output_root, queries=None, force=False,
                  resolution=128, max_attempts_per_query=3):
    """Generate one GT prompt HDF5 per query initial state for ``task``."""
    init_states = T.load_init_states(benchmark, task)
    queries = queries if queries is not None else range(T.N_QUERIES)
    task_dir = Path(output_root) / benchmark / 'experts' / task
    task_dir.mkdir(parents=True, exist_ok=True)
    results_path = task_dir / 'generation_results.json'
    results = {}
    if results_path.exists() and not force:
        results = {int(k): v for k, v in
                   json.loads(results_path.read_text()).items()}
    bddl = str(T.bddl_path(benchmark, task))
    run_dir = str(task_dir / 'gen_tmp')
    generator = None
    hdf5_checked = False
    for qi in queries:
        qdir = task_dir / f'query_{qi:02d}'
        hdf5_path = qdir / 'demo.hdf5'
        if hdf5_path.exists() and results.get(qi, {}).get('success') and not force:
            continue
        if generator is None:
            generator = QueryDemoGenerator(bddl, run_dir=run_dir,
                                           resolution=resolution,
                                           suppress_print=True)
            if generator.actions_from is not None and not hdf5_checked:
                hdf5_checked = True
                if not Path(generator.actions_from_hdf5_path).exists():
                    raise FileNotFoundError(
                        f'reference HDF5 missing: {generator.actions_from_hdf5_path}')
        started = time.time()
        ok, reason = False, 'not_attempted'
        for attempt in range(max_attempts_per_query):
            ok, reason = generator.generate_demo_from_state(
                init_states[qi], seed=qi, pose_idx=attempt)
            if ok:
                break
        entry = {'success': bool(ok), 'reason': reason,
                 'attempts': attempt + 1,
                 'wall_seconds': round(time.time() - started, 2),
                 'scripted_replay_stages': list(generator.newly_scripted_stages),
                 'replayed_stages': list(generator.actions_from_steps)}
        if ok:
            qdir.mkdir(parents=True, exist_ok=True)
            generator.write_prompt_hdf5(hdf5_path)
            entry['hdf5'] = str(hdf5_path.relative_to(output_root))
            entry['init_state_sha256'] = hashlib.sha256(
                np.asarray(init_states[qi]).tobytes()).hexdigest()
        results[qi] = entry
        results_path.write_text(json.dumps(results, indent=1))
        print(f'[{benchmark}/{task}] query {qi}: {reason}', flush=True)
    if generator is not None:
        try:
            generator.env.close()
        except Exception:
            pass
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', choices=sorted(T.BENCHMARKS), required=True)
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--queries', type=str, default=None,
                   help='e.g. "0-49" or "0,3,7"')
    p.add_argument('--output', type=Path, default=T.OUTPUTS)
    p.add_argument('--force', action='store_true')
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--max-attempts', type=int, default=3)
    args = p.parse_args()
    if args.queries:
        queries = set()
        for part in args.queries.split(','):
            if '-' in part:
                a, b = part.split('-')
                queries.update(range(int(a), int(b) + 1))
            else:
                queries.add(int(part))
        queries = sorted(queries)
    else:
        queries = None
    names = args.tasks or T.task_names(args.benchmark)
    for task in names:
        generate_task(args.benchmark, task, args.output, queries=queries,
                      force=args.force, resolution=args.resolution,
                      max_attempts_per_query=args.max_attempts)


if __name__ == '__main__':
    main()
