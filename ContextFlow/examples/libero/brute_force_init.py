"""Replay specific episodes from EVERY pruned init state of their task.

Decisive check for failed episodes: if any init state replays to success, the
demo annotation is fine and the earlier failure was a wrong init match; if all
50 fail, the recorded trajectory does not reproduce the goal open-loop.
"""

import logging
import pathlib
from typing import List

import imageio
import numpy as np
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def _agentview(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def main(
    episode_indices: List[int],
    cache_dir: str = "data/libero_incontext/gt_demo_cache",
    video_out_path: str = "data/libero_incontext/videos/gt_replay",
    num_steps_wait: int = 10,
    seed: int = 7,
) -> None:
    import json

    index = json.loads((pathlib.Path(cache_dir) / "index.json").read_text())
    ep_to_task = {rec["episode_index"]: rec["task"] for rec in index}

    benchmark_dict = benchmark.get_benchmark_dict()
    for ep_idx in episode_indices:
        task_desc = ep_to_task[ep_idx]
        found = False
        for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
            task_suite = benchmark_dict[suite_name]()
            for task_id in range(task_suite.n_tasks):
                task = task_suite.get_task(task_id)
                if task.language != task_desc:
                    continue
                found = True
                init_states = np.asarray(task_suite.get_task_init_states(task_id))
                task_bddl_file = (
                    pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256
                )
                env.seed(seed)

                data = np.load(pathlib.Path(cache_dir) / f"ep_{ep_idx:06d}.npz")
                actions = data["actions"]
                max_steps = MAX_STEPS[suite_name]
                results = []
                best = (-1, None, None)  # steps, init, frames
                for j in range(len(init_states)):
                    env.reset()
                    obs = env.set_init_state(init_states[j])
                    for _ in range(num_steps_wait):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    frames = [_agentview(obs)]
                    success = False
                    for t, a in enumerate(actions):
                        obs, _, done, _ = env.step(a.tolist())
                        frames.append(_agentview(obs))
                        if done:
                            success = True
                            break
                        if t >= max_steps:
                            break
                    if not success:
                        success = bool(env.check_success())
                    steps = len(frames) - 1
                    results.append((j, success, steps))
                    if steps > best[0]:
                        best = (steps, j, frames)
                    print(f"  ep {ep_idx} init {j:2d}: success={success} steps={steps}", flush=True)
                    if success:
                        best = (steps, j, frames)
                        break
                env.close()
                ok = [r for r in results if r[1]]
                print(f"ep {ep_idx}: {len(ok)} successful inits {ok}")
                slug = task_desc.replace(" ", "_")
                vdir = pathlib.Path(video_out_path) / suite_name / slug
                vdir.mkdir(parents=True, exist_ok=True)
                suffix = "success" if ok else "failure"
                imageio.mimwrite(vdir / f"ep_{ep_idx:06d}_bruteforce_{suffix}.mp4", best[2], fps=10)
                break
            if found:
                break
        if not found:
            print(f"ep {ep_idx}: task '{task_desc}' not found in any suite")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    tyro.cli(main)
