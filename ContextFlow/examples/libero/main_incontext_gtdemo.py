"""LIBERO eval with same-layout GT demos.

Variant of main_incontext.py: for every rollout, the injected in-context
demonstration is a dataset episode whose initial layout was verified (by GT
replay) to match the rollout's own init state — instead of the default
first-episode-of-task demo.

The mapping init_index -> episode_index is produced by GT replay
(logs/eval_results/gt_replay/init_to_episode.json). Only init indices that have
a verified demo are evaluated, so per-init outcomes compare 1:1 against the
default eval on the same init_states[k].
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
from collections import Counter

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

LEROBOT_HOME = pathlib.Path(os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
LIBERO_TASKS_JSONL = LEROBOT_HOME / "physical-intelligence" / "libero" / "meta" / "tasks.jsonl"


def get_task_to_index_mapping(file_path: pathlib.Path) -> dict:
    mapping = {}
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_description = record.get("task")
            task_index = record.get("task_index")
            if task_description is not None and task_index is not None:
                mapping[task_description] = task_index
    return mapping


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    init_demo_map: str = "logs/eval_results/gt_replay/init_to_episode.json"

    # Utils
    video_out_path: str = "data/libero_incontext/videos/gt_demo_eval"
    results_out_path: str = ""
    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    if not args.results_out_path:
        args.results_out_path = str(
            pathlib.Path("logs") / "eval_results" / f"gt_demo_eval_{args.task_suite_name}.json"
        )

    init_map = json.loads(pathlib.Path(args.init_demo_map).read_text())

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")

    task_description2index = get_task_to_index_mapping(LIBERO_TASKS_JSONL)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_ids = [
        task_id
        for task_id in range(task_suite.n_tasks)
        if task_suite.get_task(task_id).language in LIBERO_UNSEEN_TASKS
    ]
    logging.info(f"Found {len(task_ids)} unseen tasks in suite: {task_ids}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    per_task_episodes = Counter()
    per_task_successes = Counter()
    total_episodes, total_successes = 0, 0
    per_task_results = []
    per_episode_results = []

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        task_description = task.language

        initial_states = task_suite.get_task_init_states(task_id)
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # init indices that have a verified same-layout demo
        demo_map = init_map.get(args.task_suite_name, {}).get(task_description, {})
        trial_inits = sorted(int(j) for j in demo_map.keys())
        if not trial_inits:
            logging.warning(f"No GT demos for task '{task_description}'")
            continue

        task_episodes, task_successes = 0, 0
        for init_idx in tqdm.tqdm(trial_inits):
            demo_episode = demo_map[str(init_idx)]
            logging.info(f"\nTask: {task_description} | init={init_idx} demo_ep={demo_episode}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[init_idx])

            t = 0
            replay_images = []
            done = False
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "task_index": task_description2index[task_description],
                            "split": "test",
                            "demo_episode": int(demo_episode),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_init{init_idx:02d}_demo{demo_episode:05d}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            per_episode_results.append(
                {
                    "suite": args.task_suite_name,
                    "task_id": task_id,
                    "task": task_description,
                    "init_index": init_idx,
                    "demo_episode": demo_episode,
                    "success": bool(done),
                }
            )
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        per_task_episodes[task_description] = task_episodes
        per_task_successes[task_description] = task_successes
        per_task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": float(task_successes) / float(task_episodes),
            }
        )
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    logging.info(f"Total success rate: {total_success_rate} over {total_episodes} episodes")

    results = {
        "config": {
            "task_suite_name": args.task_suite_name,
            "seed": args.seed,
            "init_demo_map": args.init_demo_map,
            "mode": "same_layout_gt_demo",
        },
        "per_task_results": per_task_results,
        "per_episode_results": per_episode_results,
        "summary": {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_success_rate": total_success_rate,
            "num_tasks_evaluated": len(per_task_results),
        },
    }
    results_path = pathlib.Path(args.results_out_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
