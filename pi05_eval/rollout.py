"""pi0.5 rollout driver for generating demonstrations.

Runs closed-loop rollouts of a served pi0.5 policy (obs + prompt only, no
in-context demo) on either

  * ``contextflow`` mode: the 8 held-out LIBERO tasks (LIBERO_UNSEEN_TASKS)
    with the standard suite init states, recording at 256x256; or

  * ``liberogen`` mode: BPP's LIBERO-Gen tasks (BDDL + .pruned_init query
    states), recording at 128x128 and writing one BPP-prompt ``demo.hdf5``
    per successful query.

Every episode records the full trajectory (pre-action flattened sim state,
executed action, post-action raw obs) to ``rollout.npz`` so downstream
converters can rebuild any convention (LeRobot dataset, BPP prompt HDF5,
replay).

Client env: /data2/huangxiangyun/project/xvla/env/libero/bin/python (py3.8).
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import re
import sys

import imageio
import numpy as np
import tqdm
import tyro

import h5py

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

ICL_ROOT = pathlib.Path("/data2/huangxiangyun/project/icl")
BPP_ROOT = ICL_ROOT / "bpp"
BPP_ENV_DIR = BPP_ROOT / "src/behavior_prompting/train_network/env/libero"
CONTEXTFLOW_SRC = ICL_ROOT / "ContextFlow/src"

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

# BPP LIBERO-Gen benchmark -> (bddl split dir, max rollout steps)
LIBEROGEN = {
    "liberogen_spatial_combination": ("libero_spatial_selected_combinations_view", 550),
    "liberogen_goal_chain": ("libero_goal_chain_selected_view", 550),
}


@dataclasses.dataclass
class Args:
    # Policy server
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    mode: str = "contextflow"  # contextflow | liberogen

    # contextflow mode
    task_suite_name: str = "libero_spatial"  # or "all"
    num_trials_per_task: int = 50
    unseen_task_index: int = -1  # -1 = both unseen tasks in the suite

    # liberogen mode
    benchmark: str = "liberogen_spatial_combination"
    tasks: str = ""  # comma-separated task subset; empty = all
    queries: str = "0-49"  # e.g. "0-49", "0,3,7"
    instruction: str = ""  # if set, override the bddl language_instruction for the prompt

    # Common rollout settings (defaults follow each mode's official protocol)
    resolution: int = 0  # 0 = auto (256 contextflow / 128 liberogen)
    num_steps_wait: int = 0  # 0 = auto (10 contextflow / 12 liberogen)
    max_steps: int = 0  # 0 = auto (per-suite / 550)
    seed: int = 7

    # Arm-only trajectory perturbation (initial layout untouched). Comma list:
    # "prefix" = scripted detour right after settle; "noise" = mid-episode bias window
    perturb: str = ""
    perturb_prefix_steps: int = 18  # scripted detour length (~13cm at default amp)
    perturb_prefix_amp: float = 0.35  # action units/step (~7mm eef motion)
    perturb_noise_steps: int = 14  # mid-episode bias window length
    perturb_noise_amp: float = 0.15  # action units/step peak bias

    # Output
    output: str = ""  # required
    save_video: bool = True
    write_npz: bool = True


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _sim(env):
    if hasattr(env, "sim"):
        return env.sim
    return env.env.sim


def _flatten_sim_state(env):
    return _sim(env).get_state().flatten().astype(np.float64)


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _parse_index_set(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


# ---------------------------------------------------------------------------
# Trajectory recording -> artifacts
# ---------------------------------------------------------------------------

def _save_npz(path, rec, meta):
    obs = rec["obs"]
    payload = {
        "actions": np.asarray(rec["actions"], dtype=np.float32),
        "sim_states": np.asarray(rec["states"], dtype=np.float64),
        "agentview_image": np.stack([o["agentview_image"] for o in obs]),
        "eye_in_hand_image": np.stack([o["robot0_eye_in_hand_image"] for o in obs]),
        "eef_pos": np.stack([o["robot0_eef_pos"] for o in obs]).astype(np.float32),
        "eef_quat": np.stack([o["robot0_eef_quat"] for o in obs]).astype(np.float32),
        "gripper_qpos": np.stack([o["robot0_gripper_qpos"] for o in obs]).astype(np.float32),
        "dones": np.asarray(rec["dones"], dtype=np.uint8),
        "meta_json": np.asarray(json.dumps(meta)),
    }
    if "robot0_joint_pos" in obs[0]:
        payload["joint_pos"] = np.stack([o["robot0_joint_pos"] for o in obs]).astype(np.float32)
    np.savez_compressed(path, **payload)


def _write_bpp_hdf5(path, rec, env, trim, meta):
    """Write recorded trajectory in BPP's prompt HDF5 schema.

    Mirrors experiments/oracle_gt/gtgen.py write_prompt_hdf5 +
    create_dataset.py conventions: post-action obs, pre-action flattened sim
    states, axis-angle ee_ori, raw (opengl / upside-down) RGB, dones/rewards
    with last=1.
    """
    actions = np.asarray(rec["actions"])[trim:]
    states = np.asarray(rec["states"])[trim:]
    obs_seq = rec["obs"][trim:]
    assert len(actions) == len(obs_seq) == len(states) and len(actions) > 0

    ee_pos = np.stack([o["robot0_eef_pos"] for o in obs_seq])
    ee_ori = np.stack([_quat2axisangle(o["robot0_eef_quat"]) for o in obs_seq])
    dones = np.zeros(len(actions), dtype=np.uint8)
    dones[-1] = 1

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped = env.env if hasattr(env, "env") else env
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_name"] = meta.get("problem_name", "libero_tabletop_manipulation")
        data.attrs["problem_info"] = json.dumps(meta.get("problem_info", {}), default=str)
        try:
            import robosuite.macros as macros
            data.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
        except Exception:
            data.attrs["macros_image_convention"] = "opengl"
        data.attrs["bddl_file_name"] = meta.get("bddl_file_name", "")
        data.attrs["env_info"] = json.dumps(meta.get("env_info", {}), default=str)
        if meta.get("bddl_file_name"):
            try:
                data.attrs["bddl_file_content"] = pathlib.Path(meta["bddl_file_name"]).read_text()
            except OSError:
                pass

        g = data.create_group("demo_0")
        g.attrs["model_file"] = unwrapped.sim.model.get_xml()
        g.attrs["seed"] = int(meta.get("seed", -1))
        g.attrs["num_samples"] = len(actions)
        g.attrs["observation_alignment"] = "post_action"
        g.attrs["state_alignment"] = "pre_action"
        g.attrs["init_state"] = states[0]
        g.create_dataset("terminal_state", data=_flatten_sim_state(env))
        g.create_dataset("actions", data=actions)
        g.create_dataset("states", data=states)
        g.create_dataset("rewards", data=dones.copy())
        g.create_dataset("dones", data=dones)
        g.create_dataset("robot_states", data=np.concatenate(
            [np.stack([o["robot0_gripper_qpos"] for o in obs_seq]),
             ee_pos,
             np.stack([o["robot0_eef_quat"] for o in obs_seq])], axis=-1))
        obs_g = g.create_group("obs")
        obs_g.create_dataset("ee_pos", data=ee_pos)
        obs_g.create_dataset("ee_ori", data=ee_ori)
        obs_g.create_dataset("ee_states", data=np.concatenate([ee_pos, ee_ori], axis=-1))
        obs_g.create_dataset("gripper_states",
                             data=np.stack([o["robot0_gripper_qpos"] for o in obs_seq]))
        if "robot0_joint_pos" in obs_seq[0]:
            obs_g.create_dataset("joint_states",
                                 data=np.stack([o["robot0_joint_pos"] for o in obs_seq]))
        obs_g.create_dataset("agentview_rgb",
                             data=np.stack([o["agentview_image"] for o in obs_seq]),
                             compression="gzip", compression_opts=1)
        obs_g.create_dataset("eye_in_hand_rgb",
                             data=np.stack([o["robot0_eye_in_hand_image"] for o in obs_seq]),
                             compression="gzip", compression_opts=1)
    return path


# ---------------------------------------------------------------------------
# Core rollout loop
# ---------------------------------------------------------------------------

def _run_episode(env, client, instruction, max_steps, args, resolution, ep_seed=0):
    """One closed-loop episode; env is already at the query init state."""
    from openpi_client import image_tools

    rng = np.random.default_rng(ep_seed)
    modes = {m.strip() for m in str(args.perturb).split(",") if m.strip()}
    prefix_steps = int(args.perturb_prefix_steps) if "prefix" in modes else 0
    noise_steps = int(args.perturb_noise_steps) if "noise" in modes else 0

    detour_dir = np.zeros(3)
    if prefix_steps:
        ang = rng.uniform(0, 2 * np.pi)
        detour_dir = np.array([np.cos(ang), np.sin(ang), rng.uniform(-0.25, 0.1)])
        detour_dir /= np.linalg.norm(detour_dir)
    noise_start, noise_dir = -1, np.zeros(3)
    if noise_steps:
        lo = prefix_steps + 15
        noise_start = int(rng.integers(lo, lo + 40))
        ang2 = rng.uniform(0, 2 * np.pi)
        noise_dir = np.array([np.cos(ang2), np.sin(ang2), rng.uniform(-0.2, 0.2)])
        noise_dir /= np.linalg.norm(noise_dir)

    action_plan = collections.deque()
    rec = {"actions": [], "states": [], "obs": [], "dones": []}
    replay_images = []
    t, done = 0, False
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            pre = _flatten_sim_state(env)
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            rec["states"].append(pre)
            rec["actions"].append(np.asarray(LIBERO_DUMMY_ACTION, dtype=np.float32))
            rec["obs"].append(obs)
            rec["dones"].append(bool(done))
            t += 1
            continue

        j = t - args.num_steps_wait
        if j < prefix_steps:
            # Scripted detour: smooth push in a random direction, gripper held.
            ramp = min(1.0, (j + 1) / 4.0) * min(1.0, (prefix_steps - j) / 4.0)
            action = np.zeros(7, dtype=np.float32)
            action[:3] = args.perturb_prefix_amp * ramp * detour_dir
            action[6] = -1.0
            pre = _flatten_sim_state(env)
            obs, _, done, _ = env.step(action.tolist())
            rec["states"].append(pre)
            rec["actions"].append(action)
            rec["obs"].append(obs)
            rec["dones"].append(bool(done))
            t += 1
            if done:
                break
            continue

        raw_img = obs["agentview_image"]
        raw_wrist = obs["robot0_eye_in_hand_image"]
        img = np.ascontiguousarray(raw_img[::-1, ::-1])
        wrist_img = np.ascontiguousarray(raw_wrist[::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
        replay_images.append(np.ascontiguousarray(raw_img[::-1, ::-1]))

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"],
                     _quat2axisangle(obs["robot0_eef_quat"]),
                     obs["robot0_gripper_qpos"])).astype(np.float32),
                "prompt": str(instruction),
            }
            action_chunk = client.infer(element)["actions"]
            assert len(action_chunk) >= args.replan_steps, (
                f"policy returned {len(action_chunk)} actions < replan_steps {args.replan_steps}")
            action_plan.extend(action_chunk[: args.replan_steps])

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        if noise_steps and noise_start <= j < noise_start + noise_steps:
            k = j - noise_start
            tri = 1.0 - abs(2.0 * k / max(noise_steps - 1, 1) - 1.0)
            action[:3] += args.perturb_noise_amp * tri * noise_dir
        pre = _flatten_sim_state(env)
        obs, _, done, _ = env.step(action.tolist())
        rec["states"].append(pre)
        rec["actions"].append(action)
        rec["obs"].append(obs)
        rec["dones"].append(bool(done))
        if done:
            break
        t += 1
    return rec, done, replay_images


def _finalize_episode(ep_dir, rec, done, replay_images, env, args, meta):
    ep_dir.mkdir(parents=True, exist_ok=True)
    n_policy_steps = len(rec["actions"]) - args.num_steps_wait
    result = {
        "success": bool(done),
        "policy_steps": int(n_policy_steps),
        "recorded_steps": int(len(rec["actions"])),
        **{k: meta[k] for k in ("mode", "task", "index") if k in meta},
    }
    if args.write_npz:
        _save_npz(ep_dir / "rollout.npz", rec, {**meta, "success": bool(done),
                                              "seed": args.seed,
                                              "num_steps_wait": args.num_steps_wait})
    if args.save_video and replay_images:
        imageio.mimwrite(ep_dir / "rollout.mp4", [np.asarray(x) for x in replay_images], fps=10)
    if done and args.mode == "liberogen":
        _write_bpp_hdf5(ep_dir / "demo.hdf5", rec, env, trim=args.num_steps_wait, meta=meta)
        result["demo_hdf5"] = "demo.hdf5"
    (ep_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------------------
# contextflow mode: 8 held-out tasks, standard suite init states
# ---------------------------------------------------------------------------

def run_contextflow(args):
    sys.path.insert(0, str(CONTEXTFLOW_SRC))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training.config_libero import LIBERO_UNSEEN_TASKS
    from openpi_client import websocket_client_policy as _ws

    suites = list(SUITE_MAX_STEPS) if args.task_suite_name == "all" else [args.task_suite_name]
    client = _ws.WebsocketClientPolicy(args.host, args.port)
    out_root = pathlib.Path(args.output)

    for suite_name in suites:
        max_steps = args.max_steps or SUITE_MAX_STEPS[suite_name]
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
        task_ids = [i for i in range(task_suite.n_tasks)
                    if task_suite.get_task(i).language in LIBERO_UNSEEN_TASKS]
        if args.unseen_task_index >= 0:
            task_ids = [task_ids[args.unseen_task_index]]
        logging.info("[%s] unseen tasks: %s", suite_name, task_ids)

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            instruction = task.language
            task_slug = _slug(instruction)
            initial_states = task_suite.get_task_init_states(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                     camera_heights=args.resolution,
                                     camera_widths=args.resolution)
            env.seed(args.seed)

            n_done, n_eps = 0, 0
            for init_idx in tqdm.tqdm(range(min(args.num_trials_per_task, len(initial_states))),
                                      desc=f"{suite_name}/{task_slug}"):
                ep_dir = out_root / suite_name / task_slug / f"init_{init_idx:02d}"
                if (ep_dir / "result.json").exists():
                    prev = json.loads((ep_dir / "result.json").read_text())
                    n_eps += 1
                    n_done += bool(prev.get("success"))
                    continue
                meta = {"mode": "contextflow", "suite": suite_name,
                        "task": instruction, "task_id": int(task_id),
                        "index": int(init_idx), "bddl_file_name": str(bddl)}
                if args.perturb:
                    meta["perturb"] = args.perturb
                env.reset()
                obs = env.set_init_state(initial_states[init_idx])
                try:
                    rec, done, replay = _run_episode(env, client, instruction,
                                                   max_steps, args, args.resolution,
                                                   ep_seed=args.seed * 100003 + init_idx)
                except Exception as e:  # keep batch going; record the failure
                    logging.error("episode failed with exception: %s", e)
                    rec = {"actions": [], "states": [], "obs": [], "dones": []}
                    done, replay = False, []
                res = _finalize_episode(ep_dir, rec, done, replay, env, args, meta)
                n_eps += 1
                n_done += bool(res["success"])
                logging.info("[%s/%s init=%d] success=%s | %d/%d",
                             suite_name, task_slug, init_idx, res["success"], n_done, n_eps)
            env.close()


# ---------------------------------------------------------------------------
# liberogen mode: BPP LIBERO-Gen bddl + .pruned_init query states
# ---------------------------------------------------------------------------

def run_liberogen(args):
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info
    from openpi_client import websocket_client_policy as _ws

    split, default_max = LIBEROGEN[args.benchmark]
    max_steps = args.max_steps or default_max
    bddl_dir = BPP_ENV_DIR / "bddl_files" / split
    init_dir = BPP_ENV_DIR / "init_files" / split
    names = sorted(p.stem for p in bddl_dir.glob("*.bddl"))
    if args.tasks:
        wanted = set(args.tasks.split(","))
        names = [n for n in names if n in wanted]
    query_indices = _parse_index_set(args.queries)
    client = _ws.WebsocketClientPolicy(args.host, args.port)
    out_root = pathlib.Path(args.output)

    for task in names:
        bddl = bddl_dir / f"{task}.bddl"
        init_path = init_dir / f"{task}.pruned_init"
        init_states = torch.load(str(init_path), map_location="cpu")
        problem_info = get_problem_info(str(bddl))
        instruction = (args.instruction or problem_info.get("language_instruction")
                       or task.replace("_", " "))
        logging.info("[%s] task=%s instruction='%s' inits=%d",
                     args.benchmark, task, instruction, len(init_states))

        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_heights=args.resolution,
                                 camera_widths=args.resolution)
        env.seed(args.seed)
        n_done, n_eps = 0, 0
        for qi in tqdm.tqdm(query_indices, desc=f"{args.benchmark}/{task}"):
            if qi >= len(init_states):
                continue
            ep_dir = out_root / args.benchmark / "experts" / task / f"query_{qi:02d}"
            if (ep_dir / "result.json").exists():
                prev = json.loads((ep_dir / "result.json").read_text())
                n_eps += 1
                n_done += bool(prev.get("success"))
                continue
            meta = {"mode": "liberogen", "benchmark": args.benchmark, "split": split,
                    "task": task, "instruction": instruction, "index": int(qi),
                    "bddl_file_name": str(bddl),
                    "problem_name": problem_info.get("problem_name", ""),
                    "problem_info": {k: str(v) for k, v in problem_info.items()},
                    "seed": args.seed}
            env.reset()
            env.set_init_state(np.asarray(init_states[qi], dtype=np.float64))
            # BPP reset_to_query canonicalization: controller/gripper internal
            # state is not part of the flattened MuJoCo state.
            _e = env.env if hasattr(env, "env") else env
            _e.robots[0].controller.update(force=True)
            _e.robots[0].controller.reset_goal()
            _e.robots[0].gripper.current_action = np.zeros(2)
            _e._get_observations(force_update=True)
            try:
                rec, done, replay = _run_episode(env, client, instruction,
                                               max_steps, args, args.resolution)
            except Exception as e:
                logging.error("episode failed with exception: %s", e)
                rec = {"actions": [], "states": [], "obs": [], "dones": []}
                done, replay = False, []
            res = _finalize_episode(ep_dir, rec, done, replay, env, args, meta)
            n_eps += 1
            n_done += bool(res["success"])
            logging.info("[%s/%s q=%d] success=%s | %d/%d",
                         args.benchmark, task, qi, res["success"], n_done, n_eps)
        env.close()


def main(args: Args) -> None:
    if not args.output:
        raise ValueError("--output is required")
    if args.mode == "contextflow":
        args.resolution = args.resolution or 256
        args.num_steps_wait = args.num_steps_wait or 10
        run_contextflow(args)
    elif args.mode == "liberogen":
        if args.benchmark not in LIBEROGEN:
            raise ValueError(f"unknown benchmark {args.benchmark}")
        args.resolution = args.resolution or 128
        args.num_steps_wait = args.num_steps_wait or 12
        run_liberogen(args)
    else:
        raise ValueError(f"unknown mode {args.mode}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
