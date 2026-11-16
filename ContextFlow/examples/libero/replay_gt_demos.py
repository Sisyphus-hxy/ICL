"""Replay GT demo episodes (recorded actions) in the LIBERO simulator.

Verifies that the demonstrations injected as in-context prompts (or any other
episode of the unseen tasks) actually complete the task when their recorded
actions are replayed open-loop, and saves replay videos.

The lerobot `physical-intelligence/libero` parquet data is pre-extracted to
.npz by scripts/extract_gt_demo_data.py (the 3.8 sim env has no parquet reader).

Episode -> init-state correspondence is established empirically: for each task
we render every entry of task_suite.get_task_init_states() after the standard
10-step settle and pick the argmin-MSE match against the episode's recorded
first frame. The chosen index is reported next to the episode's ordinal rank in
the dataset so the "ordinal == init index" hypothesis is checked, not assumed.

Run with the libero sim env (python 3.8):
    PYTHONPATH=src:third_party/libero:packages/openpi-client/src \
    /data2/huangxiangyun/project/xvla/env/libero/bin/python \
        examples/libero/replay_gt_demos.py --task-suite-name all --episodes all
"""

import dataclasses
import json
import logging
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training.config_libero import LIBERO_UNSEEN_TASKS

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Same per-suite horizon as eval (only a backstop; demos are far shorter).
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


@dataclasses.dataclass
class Args:
    task_suite_name: str = "all"  # "all" or one of SUITES
    episodes: str = "all"  # "all" | "injected" (first episode of each task, as eval sees it)
    num_steps_wait: int = 10
    seed: int = 7  # same seed as eval
    cache_dir: str = "data/libero_incontext/gt_demo_cache"
    video_out_path: str = "data/libero_incontext/videos/gt_replay"
    results_out_path: str = "logs/eval_results/gt_replay/results.json"
    match_init_states: bool = True  # verify episode->init_state via first-frame image
    save_match_png: bool = True  # side-by-side of dataset frame0 vs matched render
    retry_failed_from: str = ""  # path to a previous results.json; retry its failed episodes only
    retry_topk: int = 10  # candidate init states (ranked by match MSE) tried per failed episode


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _agentview(obs):
    # rotate 180 degrees, same preprocessing as eval / training data
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _settle(env, obs, num_steps_wait):
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def _mse(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def _replay_episode(env, init_state, actions, num_steps_wait, max_steps):
    """Replay recorded actions from an init state. Returns (frames, success, done_step)."""
    env.reset()
    obs = env.set_init_state(init_state)
    obs = _settle(env, obs, num_steps_wait)

    frames = [_agentview(obs)]
    success, done_step = False, -1
    for t, action in enumerate(actions):
        obs, _, done, _ = env.step(action.tolist())
        frames.append(_agentview(obs))
        if done:
            success = True
            done_step = t
            break
        if t >= max_steps:
            break
    if not success:
        success = bool(env.check_success())
    return frames, success, done_step


def replay_suite(args: Args) -> None:
    np.random.seed(args.seed)

    index = json.loads((pathlib.Path(args.cache_dir) / "index.json").read_text())
    task_to_eps = {}
    for rec in index:
        task_to_eps.setdefault(rec["task"], []).append(rec["episode_index"])
    for eps in task_to_eps.values():
        eps.sort()

    retry_eps = None
    if args.retry_failed_from:
        prev = json.loads(pathlib.Path(args.retry_failed_from).read_text())
        retry_eps = {}
        for r in prev["episodes"]:
            if not r["success"]:
                retry_eps.setdefault((r["suite"], r["task_id"]), []).append(r["episode_index"])
        for eps in retry_eps.values():
            eps.sort()
        logging.info(
            f"Retry mode: {sum(len(v) for v in retry_eps.values())} failed episodes from {args.retry_failed_from}"
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_names = SUITES if args.task_suite_name == "all" else [args.task_suite_name]

    all_results = []
    for suite_name in suite_names:
        task_suite = benchmark_dict[suite_name]()
        task_ids = [
            i for i in range(task_suite.n_tasks) if task_suite.get_task(i).language in LIBERO_UNSEEN_TASKS
        ]
        logging.info(f"{suite_name}: {len(task_ids)} unseen tasks {task_ids}")

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            task_description = task.language
            init_states = task_suite.get_task_init_states(task_id)
            init_states = np.asarray(init_states)
            env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            env.reset()

            ep_ids = task_to_eps.get(task_description, [])
            if args.episodes == "injected":
                ep_ids = ep_ids[:1]
            if retry_eps is not None:
                ep_ids = retry_eps.get((suite_name, task_id), [])
            if not ep_ids:
                logging.warning(f"No extracted episodes for task '{task_description}'")
                continue

            # Reference render for every init state (post-settle), used to map
            # each dataset episode back to its simulator initial state.
            ref_images = None
            if args.match_init_states:
                ref_images = []
                for j in range(len(init_states)):
                    obs = env.set_init_state(init_states[j])
                    obs = _settle(env, obs, args.num_steps_wait)
                    ref_images.append(_agentview(obs))

            task_slug = task_description.replace(" ", "_")
            video_dir = pathlib.Path(args.video_out_path) / suite_name / task_slug
            video_dir.mkdir(parents=True, exist_ok=True)

            for ep_idx in tqdm.tqdm(ep_ids, desc=f"{suite_name}/{task_id}"):
                ordinal = task_to_eps[task_description].index(ep_idx)
                data = np.load(pathlib.Path(args.cache_dir) / f"ep_{ep_idx:06d}.npz", allow_pickle=True)
                actions = data["actions"]
                frame0 = data["frame0"]

                init_idx = ordinal
                match_mse = None
                match_margin = None
                order = None
                if ref_images is not None:
                    mses = [_mse(ref, frame0) for ref in ref_images]
                    order = np.argsort(mses)
                    init_idx = int(order[0])
                    match_mse = mses[init_idx]
                    match_margin = mses[order[1]] - match_mse if len(order) > 1 else None
                    if match_mse > 1000.0 or (match_margin is not None and match_margin < 30.0):
                        logging.warning(
                            f"ep {ep_idx}: weak init match j={init_idx} mse={match_mse:.1f} "
                            f"margin={match_margin:.1f} (ordinal={ordinal})"
                        )
                    if args.save_match_png:
                        pair = np.concatenate([ref_images[init_idx], frame0], axis=1)
                        imageio.imwrite(video_dir / f"ep_{ep_idx:06d}_match.png", pair)

                # Replay exactly like eval: reset -> set_init_state -> settle -> act.
                # In retry mode, walk the top-K candidate init states (ranked by
                # match MSE) until one replays to success; this separates
                # "matched the wrong init" from "demo genuinely diverges".
                max_steps = MAX_STEPS[suite_name]
                candidates = (
                    [int(j) for j in order[: args.retry_topk]]
                    if retry_eps is not None and order is not None
                    else [init_idx]
                )
                attempts = []
                best_frames, best_steps, success, done_step = [], -1, False, -1
                for j in candidates:
                    frames, ok, ds = _replay_episode(
                        env, init_states[j], actions, args.num_steps_wait, max_steps
                    )
                    steps = len(frames) - 1
                    attempts.append({"init_index": j, "success": bool(ok), "steps": steps})
                    if ok:
                        best_frames, best_steps, success, done_step = frames, steps, True, ds
                        init_idx = j
                        break
                    if steps > best_steps:
                        best_frames, best_steps = frames, steps
                frames = best_frames

                suffix = "success" if success else "failure"
                video_path = video_dir / f"ep_{ep_idx:06d}_init{init_idx:02d}_{suffix}.mp4"
                imageio.mimwrite(video_path, frames, fps=10)

                result = {
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": task_description,
                    "episode_index": ep_idx,
                    "ordinal_in_task": ordinal,
                    "init_index": init_idx,
                    "ordinal_equals_matched": bool(init_idx == ordinal),
                    "match_mse": match_mse,
                    "match_margin": match_margin,
                    "n_actions": int(len(actions)),
                    "steps_executed": int(len(frames) - 1),
                    "done_step": done_step,
                    "success": bool(success),
                    "video": str(video_path),
                }
                if len(attempts) > 1:
                    result["retry_attempts"] = attempts
                all_results.append(result)
                logging.info(
                    f"ep {ep_idx} ord={ordinal} init={init_idx} "
                    f"mse={f'{match_mse:.1f}' if match_mse is not None else 'n/a'} "
                    f"steps={len(frames)-1}/{len(actions)} success={success} "
                    f"attempts={len(attempts)}"
                )

            env.close()

    out_path = pathlib.Path(args.results_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(r["success"] for r in all_results)
    summary = {
        "episodes": len(all_results),
        "successes": n_ok,
        "success_rate": n_ok / len(all_results) if all_results else 0.0,
        "ordinal_mismatch": sum(1 for r in all_results if not r["ordinal_equals_matched"]),
        "per_task": {},
    }
    for r in all_results:
        key = f"{r['suite']}/{r['task_id']}"
        agg = summary["per_task"].setdefault(
            key, {"task": r["task"], "episodes": 0, "successes": 0, "mismatches": 0}
        )
        agg["episodes"] += 1
        agg["successes"] += int(r["success"])
        agg["mismatches"] += int(not r["ordinal_equals_matched"])
    for agg in summary["per_task"].values():
        agg["success_rate"] = agg["successes"] / agg["episodes"]

    out_path.write_text(json.dumps({"summary": summary, "episodes": all_results}, indent=2))
    logging.info(f"GT replay: {n_ok}/{len(all_results)} success ({summary['success_rate']:.1%})")
    logging.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    replay_suite(tyro.cli(Args))
