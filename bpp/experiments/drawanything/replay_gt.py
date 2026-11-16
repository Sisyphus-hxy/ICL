#!/usr/bin/env python3
"""Replay ground-truth DrawAnything demos inside DrawEnv to establish the
metric ceiling. Each eval_handmade episode's recorded actions are executed in
the same DrawEnv used by rollouts, with the env boundary angle fixed to the
angle the demo was drawn at and the agent reset to the demo's start position.
The per-episode score is max(-chamfer_distance) over steps, matching
DrawRunner's np.max(all_rewards[i]) convention.

Run with the BPP env:
  bpp/env/bpp/bin/python bpp/experiments/drawanything/replay_gt.py \
      --dataset bpp/datasets/drawanything/eval_handmade.zarr \
      --output bpp/experiments/drawanything/outputs/replay_gt
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

BPP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BPP_ROOT / 'src'))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import cv2  # noqa: E402
from behavior_prompting.common.replay_buffer import ReplayBuffer  # noqa: E402
from behavior_prompting.train_network.env.draw.draw_env import DrawEnv  # noqa: E402
from behavior_prompting.train_network.utils.draw_util import (  # noqa: E402
    get_target_drawing_image_for_task_idx)


def vis_task_name(task_name: str) -> str:
    """Same sanitization DrawRunner uses for media filenames."""
    name = task_name.replace(' ', '_')
    end = name[len('draw_'):]
    if end.upper() == end:
        return name + '_upper'
    if end.lower() == end:
        return name + '_lower'
    raise ValueError(f'invalid task name: {task_name}')


def replay_episode(replay_buffer, episode_idx, out_dir, seed):
    task_name = str(replay_buffer.task_names[episode_idx])
    start = 0 if episode_idx == 0 else int(replay_buffer.episode_ends[episode_idx - 1])
    end = int(replay_buffer.episode_ends[episode_idx])
    actions = replay_buffer['action'][start:end]
    target_drawing, boundary_angle = get_target_drawing_image_for_task_idx(
        replay_buffer, episode_idx)

    env = DrawEnv(
        boundary_angle=float(boundary_angle),
        reset_to_state=np.array([*replay_buffer['agent_pos'][start], 0]),
        target_drawing=target_drawing,
        target_boundary_angle=float(boundary_angle),
        overlay_reward=False,
        overlay_action_cross=False,
        overlay_target_drawing=True,
        render_mode='rgb_array')
    env.seed(seed)
    env.reset()

    rewards = []
    for action in actions:
        _, reward, _, _ = env.step(np.asarray(action, dtype=np.float64))
        rewards.append(reward)

    result = {
        'task_name': task_name,
        'demo_idx': None,
        'seed': seed,
        'boundary_angle': float(boundary_angle),
        'n_steps': len(rewards),
        'max_reward': float(np.max(rewards)),
        'final_reward': float(rewards[-1]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / 'final_canvas.png'),
                cv2.cvtColor(env.get_drawing_image(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / 'frame.png'),
                cv2.cvtColor(env.render('rgb_array'), cv2.COLOR_RGB2BGR))
    env.close()
    return task_name, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-start-seed', type=int, default=10000)
    args = parser.parse_args()

    replay_buffer = ReplayBuffer.create_from_path(args.dataset)
    dataset_name = args.dataset.stem
    out_root = args.output / dataset_name

    # demo_idx is the per-task ordinal of each episode (0..4 in eval_handmade)
    seen = {}
    rows = []
    for episode_idx in range(replay_buffer.n_episodes):
        task_name = str(replay_buffer.task_names[episode_idx])
        demo_idx = seen.get(task_name, 0)
        seen[task_name] = demo_idx + 1
        out_dir = out_root / vis_task_name(task_name) / f'demo{demo_idx}'
        _, result = replay_episode(replay_buffer, episode_idx, out_dir,
                                   seed=args.test_start_seed + demo_idx)
        result['demo_idx'] = demo_idx
        result['name'] = f'{dataset_name}/{vis_task_name(task_name)}/demo{demo_idx}'
        (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        rows.append(result)
        print(f"{result['name']:60s} steps={result['n_steps']:4d} "
              f"max={result['max_reward']:8.3f} final={result['final_reward']:8.3f}",
              flush=True)

    csv_path = args.output / 'rollout_results.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    maxima = [r['max_reward'] for r in rows]
    finals = [r['final_reward'] for r in rows]
    per_task = {}
    for r in rows:
        per_task.setdefault(r['task_name'], []).append(r['max_reward'])
    summary = {
        'episodes': len(rows),
        'tasks': len(per_task),
        'mean_max_reward': float(np.mean(maxima)),
        'min_max_reward': float(np.min(maxima)),
        'mean_final_reward': float(np.mean(finals)),
        'worst_tasks': sorted(
            ((name, float(np.mean(vals))) for name, vals in per_task.items()),
            key=lambda item: item[1])[:5],
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
