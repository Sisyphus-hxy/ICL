# Default Prompt vs Oracle-GT Prompt (LIBERO-Gen)

Paired per-query comparison on the 20 unseen LIBERO-Gen tasks
(10 Spatial Combination + 10 Goal Chain), 50 fixed query initializations each.

* **Default arm**: official prompt sampling — query `i` gets `demo_(i % 10)` of
  the task's unseen-view HDF5 (identical assignment to the official
  `LiberoImageRunner` with `n_envs=10`, `only_prompt` dataset ordered by demo
  index).
* **Oracle arm**: a successful expert trajectory generated from *the same*
  query initial state (`gtgen.py`), supplied as the complete RGB/action prompt
  in BPP's original input schema (`eval_common.load_prompt`, verified
  bit-exact vs the official dataset sampler in
  `native_prompt_swap/outputs/audit_parity_final`).

Both arms share one checkpoint, the exact same flattened MuJoCo init state
(asserted via SHA-256 of sim state + agentview RGB after the 12-step
open-gripper warmup) and the same diffusion seed (`seed_base*1000 + query`).
The policy predicts actions autonomously; GT actions are never replayed.

## Expert generation from a specified query init state

Upstream `TaskDemonstrationGenerator` does not support this natively: demos
start from `env.seed(demo_idx); env.reset()`, and `copy_actions_from_hdf5`
hard-resets to the *reference* demo's initial state. `gtgen.QueryDemoGenerator`
subclasses it:

* after `env.reset()`, injects the query state with
  `set_sim_state_from_flattened_with_update` + `env._start_new_episode()`;
* re-scripts every scriptable `actions_from_steps` prefix stage
  (`Grasp`/`Place`; `Place` rel poses are analytic, `Grasp` poses come from
  `grasps_from`, auto-synthesized from the reference task when missing), so
  movable-object pick-places survive arbitrary query layouts;
* replays only truly non-scriptable stages (`Open`/`Turnon`/`Touch` — no
  scripted planner exists) from the query state, retrying over reference demos
  ranked by init-state similarity (fixtures are identical across layouts, so
  this is reliable; `push_the_plate`'s movable plate is the residual risk).

Recorded demos match `create_dataset` conventions: post-action obs, pre-action
states, axis-angle `ee_ori`, upside-down RGB, first-5-step `cap_index` trim,
final-step success marking. Each query's prompt lands in
`experts/<task>/query_XX/demo.hdf5` (`data/demo_0` schema that
`load_prompt`/`LiberoReplayImageDataset` consume).

## Layout

```
outputs/<benchmark>/
  eval_config.json              # arm rules, seeds, checkpoint metadata
  experts/<task>/
    generation_results.json     # per-query expert success + stage provenance
    query_XX/demo.hdf5          # GT prompt (data/demo_0)
  eval/<task>/query_XX/
    default/{result.json,trajectory.csv,rollout.mp4?}
    oracle/{result.json,trajectory.csv,rollout.mp4?}
  summary.csv per_query.csv report.json
```

## Running

GPU values follow `AGENTS.md` (`--gpu 5`/`6` = RTX A6000, `1..4` = RTX 6000D,
`0` = Blackwell needing `env/bpp-sm120`). Check data/GPU status first:

```bash
bash bpp/experiments/oracle_gt/run.sh status
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
```

GT generation is CPU-only (MuJoCo EGL renders offscreen on CPU):

```bash
bash bpp/experiments/oracle_gt/run.sh generate \
  --benchmark liberogen_goal_chain --queries 0-49          # all tasks
# shard across workers for speed:
bash bpp/experiments/oracle_gt/run.sh generate \
  --benchmark liberogen_goal_chain --tasks <task> --queries 0-24
```

Paired evaluation (one process per GPU, shard by task or query):

```bash
CUDA_VISIBLE_DEVICES=5 bash bpp/experiments/oracle_gt/run.sh eval \
  --benchmark liberogen_goal_chain --tasks <task>
CUDA_VISIBLE_DEVICES=6 bash bpp/experiments/oracle_gt/run.sh eval \
  --benchmark liberogen_goal_chain --tasks <task2> --save-video
```

Resume: existing `result.json`/`demo.hdf5` are skipped; rerun the same command.

Aggregate:

```bash
bash bpp/experiments/oracle_gt/run.sh report --benchmark liberogen_goal_chain
```

## Prerequisites

* checkpoint per benchmark (`checkpoints/liberogen_*_behavior_prompting.ckpt`)
* unseen-view HDF5 per task (`datasets/libero/<split>/<task>_demo.hdf5`) for the
  default prompts — the user is downloading these
* base-split reference HDF5s (`datasets/libero/libero_goal/`,
  `datasets/libero/libero_spatial/`) for grasp poses + replay stages
* `init_files/<split>/<task>.pruned_init` (already present, 50 states each)

## Verification built in

* `reset_to_query` asserts identical sim-state+RGB hash between the two arms
  of every query pair.
* Expert yield per task is in `summary.csv` (`expert_yield`); queries without a
  successful GT are marked `no_gt` in `per_query.csv` and excluded from the
  paired delta.
* `prompt_diagnostics`-style checks can be repeated via
  `eval_common.load_prompt` on any `experts/**/demo.hdf5`.
