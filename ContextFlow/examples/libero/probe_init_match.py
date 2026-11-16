"""Probe episode->init-state matching for specific episodes.

For each requested episode, computes MSE of the dataset first frame against
(a) all pruned_init states and (b) all raw .init states, each rendered both
immediately after set_init_state (pre-settle) and after the eval settle window.
Reports the ranked candidates to diagnose why some episodes fail replay.
"""

import logging
import pathlib
from typing import List

import numpy as np
import torch
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _agentview(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _mse(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))


def main(
    episode_indices: List[int],
    cache_dir: str = "data/libero_incontext/gt_demo_cache",
    num_steps_wait: int = 10,
    seed: int = 7,
) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    index = None
    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        task_suite = benchmark_dict[suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.language not in LIBERO_UNSEEN_TASKS:
                continue
            # episodes of this task present in the cache
            import json

            if index is None:
                index = json.loads(
                    (pathlib.Path(cache_dir) / "index.json").read_text()
                )
                global task_to_eps
                task_to_eps = {}
                for rec in index:
                    task_to_eps.setdefault(rec["task"], []).append(rec["episode_index"])

            eps_here = [e for e in task_to_eps.get(task.language, []) if e in episode_indices]
            if not eps_here:
                continue

            task_bddl_file = (
                pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            )
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256
            )
            env.seed(seed)
            env.reset()

            pruned = np.asarray(task_suite.get_task_init_states(task_id))
            init_dir = pathlib.Path(get_libero_path("init_states")) / task.problem_folder
            raw_init = np.asarray(
                torch.load(init_dir / task.init_states_file.replace(".pruned_init", ".init"))
            )

            refs = {"pruned": {"pre": [], "post": []}, "init100": {"pre": [], "post": []}}
            for name, states in [("pruned", pruned), ("init100", raw_init)]:
                for st in states:
                    obs = env.set_init_state(st)
                    refs[name]["pre"].append(_agentview(obs))
                    try:
                        for _ in range(num_steps_wait):
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        refs[name]["post"].append(_agentview(obs))
                    except ValueError:
                        # init state already terminal -> cannot settle; reuse pre image
                        refs[name]["post"].append(refs[name]["pre"][-1])
                        env.reset()
            env.close()

            for ep in eps_here:
                frame0 = np.load(pathlib.Path(cache_dir) / f"ep_{ep:06d}.npz")["frame0"]
                print(f"\n=== ep {ep} ({suite_name}/{task_id}: {task.language[:60]})")
                for name in ["pruned", "init100"]:
                    for when in ["post", "pre"]:
                        mses = np.array([_mse(r, frame0) for r in refs[name][when]])
                        order = np.argsort(mses)
                        top = ", ".join(f"j{j}:{mses[j]:.0f}" for j in order[:6])
                        print(f"  {name:7s} {when:5s}: {top}   (n={len(mses)})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    tyro.cli(main)
