# pi0.5 demo generation (no in-context demo, prompt only)

Runs the official `pi05_libero` checkpoint (upstream openpi, JAX) on:

- **ContextFlow's unseen test set**: the 8 `LIBERO_UNSEEN_TASKS` held-out
  tasks, 50 trials each (standard suite init states, 256px) — successful
  rollouts are converted to a LeRobot v2 dataset `pi05/libero_demos` that
  `CustomLeRobotDataset.load_incontext_demonstration` can serve as
  in-context demos (same task_index space as `physical-intelligence/libero`).
- **BPP LIBERO-Gen** (paused): both unseen benchmarks
  (`liberogen_spatial_combination`, `liberogen_goal_chain`), 50 query inits
  per task (`.pruned_init`, 128px) — each success lands as
  `demo.hdf5` in BPP's prompt schema under
  `bpp/experiments/pi05_demos/outputs/<benchmark>/experts/<task>/query_XX/`,
  loadable via `experiments/oracle_gt/eval_common.load_prompt`.

## Status (2026-11-15)

- **ContextFlow rollouts DONE: 391/400 = 97.8% success** — matches the
  official pi05_libero benchmark (~96.85%). Per-task:
  spatial 97/100, goal 100/100, object 100/100, libero_10 94/100.
  Report: `ContextFlow/data/pi05_demos/rollouts/REPORT.md`.
- **LIBERO-Gen paused: π0.5_libero ≈ 0% success** (0/63 across both
  benchmarks). Ablated resolution (128/256), instruction phrasing
  (official LIBERO-Gen text vs canonical), replan_steps (1/5) — the arm
  approaches but never commits to novel-combination goals. This is a
  capability limit of the 90-task LIBERO finetune, not a pipeline bug.
  BPP demos would need `experiments/oracle_gt/gtgen.py` instead.
- **ContextFlow eval with pi05 demos DONE**: `pi05/libero_demos`
  (391 eps) injected via `CONTEXTFLOW_DEMO_REPO_ID`, official
  `main_incontext.py` protocol (random same-task demo, 50 trials/task):

  | suite | pi05 demos | GT demos (repo's own runs) | GT same-layout |
  |---|---|---|---|
  | libero_spatial | 59% | 63% (`unseen/`) | 51% (`gt_demo_eval/`) |
  | libero_object | 93% | 89% | 91% |
  | libero_goal | 0% | 0% | 0% |
  | libero_10 | ~0%* | 0% | 0% |

  **pi0.5 trajectories are a drop-in equivalent for GT demos.** The 0%
  suites are hard zeros of this ContextFlow checkpoint, NOT a data
  problem: repo's own GT-demo results
  (`ContextFlow/logs/eval_results/{unseen,gt_demo_eval}/`) are also 0%
  on all 4 goal/l10 tasks, and an independent 25-ep GT baseline on
  "bowl on stove" confirmed 0/25. Video evidence: the policy executes
  a *seen* task instead (bowl → cabinet top = task_index 18) — a
  training prior the ICL conditioning cannot override on held-out
  goal semantics. Even a same-init π0.5 demo (`demo_episode` pinning,
  `main_incontext_gtdemo.py`, map at
  `logs/eval_results/init_to_episode_pi05.json`) fails identically.
  Per-task pattern also matches GT exactly (spatial cookie-box 88%
  vs GT 88%; next-to-plate ~30% vs GT 38%).

- **Perturbed-demo control (libero_spatial, 100 eps)**: a second
  dataset `pi05/libero_demos_perturbed` (96 eps) was generated with
  `--perturb prefix,noise` — an 18-step scripted arm detour after
  settle plus a ~14-step mid-episode bias window (layout untouched,
  π0.5 still succeeds 96/100). Trajectory stats shift ONTO the GT
  distribution (steps/path/directness/speed match GT), yet the demo
  *shape* is off-manifold (scripted wander prefix). ContextFlow with
  perturbed demos: **42/100** vs 59/100 clean pi05 demos vs 63/100 GT
  (cookie-box 34/50 vs 44/50; next-to-plate 8/50 vs 15/50). ICL is
  moderately sensitive to demo trajectory *shape*, not just the
  endpoint task — off-style demos still work ~2/3 as well.

## GPU mapping gotchas (this machine)

- `CUDA_VISIBLE_DEVICES` values differ from `nvidia-smi` physical indices
  (see root AGENTS.md): phys0/1 A6000 = CVD 5/6, phys2-5 6000D = CVD 1-4,
  phys6 Blackwell = CVD 0.
- **Policy server** (`serve.sh`) takes the **CVD** value.
- **Sim clients** (`run_*.sh`) take the **PHYSICAL nvidia-smi index** —
  they only do EGL rendering (no CUDA). mujoco/robosuite select the EGL
  device via `MUJOCO_EGL_DEVICE_ID` in physical order; passing a CVD
  value there lands the ~700MB GL context on the WRONG card (possibly
  someone else's).

## Setup (done once)

```bash
source pi05_eval/env.sh        # proxy + caches (all on /data2)
git clone https://github.com/Physical-Intelligence/openpi $ICL/openpi  # pinned 215abfb
cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync
gsutil -m cp -r "gs://openpi-assets/checkpoints/pi05_libero/*" \
    "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero.partial" && \
    mv "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero.partial" \
       "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
```

Note: the openpi server env uses **new lerobot** — it needs
`HF_LEROBOT_HOME` (setting `LEROBOT_HOME` makes it crash). Old lerobot
(ContextFlow venv, for build/verify scripts) still reads `LEROBOT_HOME`;
set it per-invocation.

## Run

One policy server + sim client(s) per GPU. The websocket server
serializes inference; 2 clients per server ≈ 1.7x throughput since sim
time dominates.

```bash
# server (CVD value):
bash pi05_eval/serve.sh 5 8000

# clients (PHYSICAL index):
bash pi05_eval/run_contextflow.sh 0 8000 all            # 8 unseen tasks x 50
bash pi05_eval/run_liberogen.sh 3 8002 liberogen_spatial_combination "" 0-49
bash pi05_eval/run_liberogen.sh 3 8002 liberogen_goal_chain "" 0-24
```

Resume: existing `result.json` episode dirs are skipped.

## Convert + verify

```bash
LEROBOT_HOME=/data2/huangxiangyun/.cache/huggingface/lerobot \
ContextFlow/.venv/bin/python pi05_eval/build_lerobot_demos.py \
    --rollouts ContextFlow/data/pi05_demos/rollouts --repo-id pi05/libero_demos
LEROBOT_HOME=/data2/huangxiangyun/.cache/huggingface/lerobot \
ContextFlow/.venv/bin/python pi05_eval/verify_demos.py cf
bpp/env/bpp/bin/python pi05_eval/verify_demos.py bpp bpp/experiments/pi05_demos/outputs
python3 pi05_eval/summarize.py ContextFlow/data/pi05_demos/rollouts bpp/experiments/pi05_demos/outputs
```

## Output layout

```
ContextFlow/data/pi05_demos/rollouts/<suite>/<task>/init_XX/
    {rollout.npz, rollout.mp4, result.json}
bpp/experiments/pi05_demos/outputs/<benchmark>/experts/<task>/query_XX/
    {demo.hdf5 (success only), rollout.npz, rollout.mp4, result.json}
$LEROBOT_HOME/pi05/libero_demos/            # LeRobot v2 dataset
```

`rollout.npz` keeps the raw trajectory (post-action obs 128/256px, pre-action
flattened sim states, executed actions) for any future re-conversion.
