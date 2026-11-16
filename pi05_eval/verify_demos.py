"""Verify generated demos are loadable by their consumers.

* BPP: ``bpp/env/bpp/bin/python pi05_eval/verify_demos.py bpp <root>`` —
  runs eval_common.load_prompt on every demo.hdf5.
* ContextFlow: ``ContextFlow/.venv/bin/python pi05_eval/verify_demos.py cf``
  — loads CustomLeRobotDataset on pi05/libero_demos and checks every
  task_index resolves to at least one episode.
"""
import json
import pathlib
import sys


def verify_bpp(root: str):
    sys.path.insert(0, "/data2/huangxiangyun/project/icl/bpp/experiments/oracle_gt")
    sys.path.insert(0, "/data2/huangxiangyun/project/icl/bpp/src")
    import eval_common  # noqa
    from hydra import compose, initialize_config_dir
    cfg_dir = "/data2/huangxiangyun/project/icl/bpp/src/behavior_prompting/train_network/config"
    bad = []
    n = 0
    for h in sorted(pathlib.Path(root).rglob("demo.hdf5")):
        bench = "liberogen_spatial_combination" if "spatial" in str(h) else "liberogen_goal_chain"
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(config_name="libero_policy_dunetp",
                          overrides=[f"task={bench}", f"+modifiers=libero/{bench}"])
        try:
            p = eval_common.load_prompt(str(h), cfg.shape_meta, "cpu", demo_idx=0)
            assert p["action"].numel() > 0
            n += 1
        except Exception as e:
            bad.append((str(h), str(e)))
    print(f"bpp: {n} demo.hdf5 load_prompt OK, {len(bad)} failed")
    for h, e in bad[:10]:
        print("  FAIL", h, e)


def verify_cf(repo_id="pi05/libero_demos"):
    import os
    from openpi.training.custom_dataset import CustomLeRobotDataset
    from openpi.training.config_libero import LIBERO_UNSEEN_TASKS
    import json as _json
    root = pathlib.Path(os.environ["LEROBOT_HOME"]) / repo_id
    tasks = {t["task_index"]: t["task"]
             for t in map(_json.loads,
                          (pathlib.Path(os.environ["LEROBOT_HOME"])
                           / "physical-intelligence/libero/meta/tasks.jsonl")
                          .read_text().splitlines())}
    ds = CustomLeRobotDataset(
        repo_id=repo_id, local_files_only=True,
        delta_timestamps={"actions": [t / 10.0 for t in range(10)]},
        num_sample_frames=8, num_sample_actions=128, random_select=False)
    unseen_idx = [i for i, t in tasks.items() if t in LIBERO_UNSEEN_TASKS]
    print("unseen task_indices:", unseen_idx)
    for ti in unseen_idx:
        eps = ds.task_to_episode.get(ti, ())
        print(f"  task_index {ti} '{tasks[ti][:50]}': {len(eps)} demo episode(s)")
        if eps:
            demo = ds.load_incontext_demonstration(current_ep_idx=-1, task_index=ti)
            assert demo["dem_prompt_images"]["image"].shape[0] == 8
            assert demo["dem_prompt_actions"].shape[0] == 128


if __name__ == "__main__":
    if sys.argv[1] == "bpp":
        verify_bpp(sys.argv[2])
    elif sys.argv[1] == "cf":
        verify_cf(*sys.argv[2:])
