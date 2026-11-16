#!/usr/bin/env python3
"""GT action replay — are the generated expert actions executable open-loop?

Replays every expert HDF5's recorded action sequence in a fresh env and checks
the task success predicate. Two start conditions per query:

* ``state0``: reset to the demo's recorded pre-step sim state — pure dynamics
  determinism check (should reproduce ~100%).
* ``query``: reset to the benchmark query init state — eval-equivalent
  condition (the state the paired arms actually start from).

No model inference; camera obs disabled so it runs CPU-only.
Results: <output>/<benchmark>/replay/<task>/query_XX/replay_result.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LIBERO_CONFIG_PATH',
                      str(Path(__file__).resolve().parents[2] / 'config/libero'))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'src'))

import h5py
import numpy as np

import tasks as T
from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner


def make_env(bddl):
    # use_camera_obs=False hangs in regenerate_obs_from_state -> _get_observations;
    # camera obs is required by this env's observable pipeline (EGL render).
    return ControlEnvForRunner(str(bddl), camera_heights=128, camera_widths=128,
                               control_delta=True, use_camera_obs=True,
                               has_offscreen_renderer=True, ignore_done=True,
                               hard_reset=False)


def reset_to_state(w, state):
    """Same canonicalization as evaluate.reset_to_query (no warmup)."""
    w.register_init_state(None)
    w.seed(7)
    w.reset()
    w.register_init_state(np.asarray(state))
    w.reset()
    e = w.env
    e.robots[0].controller.update(force=True)
    e.robots[0].controller.reset_goal()
    e.robots[0].gripper.current_action = np.zeros(2)


def replay(w, start_state, actions):
    reset_to_state(w, start_state)
    e = w.env
    first_success = None
    for i, a in enumerate(actions):
        e.step(np.asarray(a, dtype=np.float64))
        if first_success is None and bool(e._check_success()):
            first_success = i
    final = e.sim.get_state().flatten().copy()
    return {'success': first_success is not None,
            'first_success_step': first_success,
            'n_actions': int(len(actions)),
            'final_state': final}


def run_task(benchmark, task, queries, output):
    init_states = T.load_init_states(benchmark, task)
    w = make_env(T.bddl_path(benchmark, task))
    rdir = output / benchmark / 'replay' / task
    try:
        for qi in queries:
            demo = output / benchmark / 'experts' / task / f'query_{qi:02d}' / 'demo.hdf5'
            if not demo.exists():
                continue
            rpath = rdir / f'query_{qi:02d}' / 'replay_result.json'
            if rpath.exists():
                continue
            with h5py.File(demo, 'r') as f:
                actions = f['data/demo_0/actions'][:]
                state0 = f['data/demo_0/states'][0]
                last_state = f['data/demo_0/states'][-1]
            out = {'benchmark': benchmark, 'task': task, 'query': qi}
            for tag, start in (('state0', state0),
                               ('query', init_states[qi])):
                res = replay(w, start, actions)
                final = res.pop('final_state')
                same = final.shape == last_state.shape
                res['final_state_l2_vs_recorded'] = float(
                    np.linalg.norm(final - last_state) if same else -1.0)
                res['final_state_maxdiff_vs_recorded'] = float(
                    np.abs(final - last_state).max() if same else -1.0)
                out[tag] = res
            rpath.parent.mkdir(parents=True, exist_ok=True)
            rpath.write_text(json.dumps(out, indent=1))
            print(f'[{benchmark}/{task}/q{qi}] '
                  f'state0={out["state0"]["success"]} '
                  f'query={out["query"]["success"]} '
                  f'len={out["query"]["n_actions"]}', flush=True)
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
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--queries', type=str, default=None)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for task in (a.tasks or T.task_names(a.benchmark)):
        run_task(a.benchmark, task, parse_queries(a.queries), a.output)


if __name__ == '__main__':
    main()
