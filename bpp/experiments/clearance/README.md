# Same Goal, Different Procedures: zero-shot diagnostic

This directory extends the installed LIBERO bowl-on-cookie-box task. It leaves the
upstream training and evaluation code unchanged and downloads no resources.
Use the existing BPP Python environment and dedicated Spatial Combination checkpoint.

Original-task positive controls now succeed: the official evaluator scored 2/3,
and this directory's inference adapter scored 2/3 on three original demonstration
initial states. These are small positive controls, not a full benchmark reproduction.
The explicit original prompt agrees exactly with the official dataset loader for
all RGB, proprioception, action and mask tensors. See `AUDIT.md` for the isolation
checks and the generated-demonstration alignment defect found during this audit.
With the same original query states and an 800-step budget, replacing only the
original prompt with the corrected Narrow or Wide demonstration scored 0/3 each.
The alignment fix alone did not resolve this new-demonstration failure.

The original pick/place distance is only about 15 cm. Moving the box farther away
alone made the reference grasp orientation unreachable. The validated alternative
keeps the same task, Panda, bowl, box, assets and success predicate, and translates
the bowl and box to a reachable 30 cm corridor. Distractors are retained. The new
obstacle is one MuJoCo cylinder, radius 2.5 cm and height 18 cm. Its mocap body is
stationary during each episode and relocatable between episodes.

`scene.py` restores the local generated `demo_0` initial state (reference seed 2)
and transfers its successful grasp relative to the bowl. `expert.py` reuses
`TaskDemonstrationGenerator.move_to_pose` from the upstream generation script.
The original generator imports keyboard helpers; the isolated runner selects
pynput's dummy backend so those unused helpers work without an X display.

Narrow and Wide have identical initial states, grasp and place targets, orientations,
gripper schedules and control-step budgets. Only transport waypoints change.
The narrow route includes an extra excursion away from the obstacle to match
transport path length; both paths are sampled uniformly by arc length. The
validation gate additionally checks actual simulated path lengths within 10%,
equal durations, a positive narrow gap of at least 5 mm, and at least 10 mm of
separation between the procedures' measured minimum object clearances.
The gate does not interpret different global route shapes as proof of abstract preference.

For the counterfactual, the cylinder moves from `(-.020, .025, .990)` to
`(-.020, .145, .990)`. The original object/robot poses are replayed geometrically
against the relocated obstacle to verify that both old routes penetrate it.
Two additional successful, collision-free experts certify that alternative routes
exist. These relocation experts are feasibility witnesses only: the model always
receives the original Narrow or Wide demonstration in both query layouts.

## Run

From the project root:

```bash
# No network requests; simulation and rendering can run without GPU access.
bash bpp/experiments/clearance/run.sh inspect --device cpu --software-render
bash bpp/experiments/clearance/run.sh experts --device cpu --software-render

# Actual BPP, same checkpoint and diffusion RNG seed for each paired prompt.
# Check available A6000 memory before starting. --gpu is CUDA visibility, not
# the nvidia-smi physical index. GPU UUIDs require selecting the EGL device
# separately; using software rendering keeps the renderer on the CPU.
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
bash bpp/experiments/clearance/run.sh rollouts --device cuda --gpu 5 --software-render

# CPU inference uses the same full checkpoint and 16 diffusion steps.
bash bpp/experiments/clearance/run.sh rollouts --device cpu --software-render

# Optional: two diffusion RNG seeds; each seed is paired across both prompts.
bash bpp/experiments/clearance/run.sh all --device cuda --gpu 5 \
  --software-render --query-seeds 0,1 --output bpp/experiments/clearance/outputs/two_seeds

# Offline contracts, including absence of procedure labels from the model inputs.
source bpp/scripts/common.sh
bpp/env/bpp/bin/python -m unittest discover -s bpp/experiments/clearance -p test_experiment.py -v
```

Default output is `outputs/latest/`. Use the same output directory for experts and
rollouts. For a new seed set, first generate/copy the validated expert artifacts
to that directory, or use `all` to generate experts and rollouts together.
`--no-images` is for physics tuning only and deliberately cannot pass the
behavior-prompt validation gate. `BPP_PYTHON` can select an existing alternate
environment, including `bpp/env/bpp-sm120`; environments are never installed here.

`inference.py` instantiates the same pinned official configuration as `run_eval.sh`, restores
the exact `model` state dict strictly, and calls the existing `reset`, `prompt`
and `predict_action` APIs. HF offline mode reuses the existing CLIP initialization
cache and prevents additional downloads. It uses
the upstream image preprocessing, rotation conversion and `PromptActionChunker`.
Only RGB, EEF position/orientation, gripper state and actions enter the prompt.
Procedure names, task text and file metadata never enter BPP.

The released Spatial Combination checkpoint's serialized configuration omits
`use_pool_modality_pos_embed`; the class default enables an embedding absent from
its state dict. The official
`libero_policy_dunetp` configuration disables that flag. The experiment uses this
official configuration and requires a strict state-dict load; it never invents a
missing learned parameter or ignores unmatched keys. Both values are recorded.

## Measurements and outputs

`results.csv` contains expert metrics; `rollout_results.csv` contains actual BPP
metrics. Each episode also has an HDF5 file, an H.264 video with both camera views,
a full XYZ EEF/object trajectory CSV, and a detailed contact JSON. `trajectories.png`
compares paths. `initial_state.npz`, `scene.xml`, invocation JSON and checkpoint
configuration preserve reproduction information.

Expert physics is recorded at 20 Hz. To avoid repeating expensive software renders,
the `render` stage renders saved states at 5 Hz and stores a zero-order hold between
RGB keyframes. Every frame used by BPP's 1 Hz prompt sampler is rendered exactly;
actions and proprioception remain at 20 Hz. Expert videos play at the correct
physical time. Rollout videos and query observations retain the full 20 Hz rate.
`render_validation.json` checks that moving the obstacle changes RGB and that the
same-layout initial images match exactly; the new post is in visible geometry group 1.
Rendering can be repeated without regenerating or changing the verified physics.

As in upstream `scripts/libero/create_dataset.py`, `obs[t]` is recorded **after**
executing `actions[t]`, while `states[t]` is the pre-action state used for replay.
The terminal state and initial EEF position are stored separately. Sparse RGB
renders use `states[t+1]` (or the terminal state), so prompt images and proprioception
refer to the same time. Legacy experimental HDF5s used pre-action observations;
inference now rejects them until experts are regenerated.

Clearance is signed MuJoCo `mj_geomDistance` between the obstacle collision
geometry and every bowl/gripper collision geometry, measured after every physics
substep. The primary object-relative minimum is the **bowl surface to obstacle
surface distance during transport**, in metres, not an EEF-centre proxy. CSVs
also include obstacle-relative bowl coordinates and separate gripper clearance.
Both full-episode and transport EEF path lengths are recorded. Collision records
include obstacle/robot/object contact, other unintended robot contact, and carried
object/environment contact during transport. Intended grasp contacts, gripper
closure, initial tabletop support and placement support are excluded. Contacts
with negative or zero distance count; no collision-avoidance filter changes BPP actions.

The initial simulation-state hash, obstacle pose and camera-image hash are recorded
per episode, along with the OSC goal and gripper accumulated command. Reset fixes
all of these, not only MuJoCo qpos/qvel. Same-layout paired rollouts must have identical initial state and
images; the RNG seed is reset after prompting. The default 800-step limit exceeds
the experts' 756 steps. A truncated or unsuccessful rollout is not evidence of
retained preference; comparisons require successful, collision-free trajectories.

This is **zero-shot diagnostic**. The checkpoint was not trained on this newly
constructed pair or obstacle. One seed provides no statistical estimate. Failure
here cannot establish that BPP lacks the capacity to learn execution preferences.
The new obstacle, translated object positions and slower expert demonstrations
also introduce distribution shifts; matching Narrow/Wide timing does not remove
those shifts relative to the checkpoint's original training demonstrations.

`--query-start transport` is an additional isolation check. It restores the first
transport state from the original Narrow expert (common lifted/grasped state)
for both prompts and both layouts, keeps the same full demonstrations, restores
the Panda gripper's accumulated command (which is not in a MuJoCo state vector),
and records both in `query_initial_state.npz`. This helps separate acquisition
of the grasp from transport behavior. It is reported separately from the requested
initial-state experiment; a 520-step limit covers the experts' remaining 466 steps.

**Use `outputs/gpu_aligned` for current primary results.**
`gpu_verified`, `gpu_transport_verified`, `audit_custom_narrow` and
`audit_custom_wide` contain legacy prompts with the observation/action alignment
defect. Their `PROMPT_ALIGNMENT_DEFECT.json` excludes those model runs from
current interpretation; the successful expert physics remains valid.
The legacy held-state model diagnostic has not been rerun with aligned prompts.
Regeneration refuses to overwrite reused HDF5 symlinks.
Older `outputs/pilot` results are exploratory only:
they matched physical state and RGB but did not reset all controller internals.
The old `gpu_pilot` and `gpu_transport_start` directories are superseded.
The directory
`outputs/obsolete_invisible_obstacle_smoke` contains invalid early interface
checks made before the obstacle's rendering-group correction; exclude it from
all results and interpretation.
`outputs/obsolete_unrestored_gripper_diagnostic` is also invalid and excluded:
its held-state reset omitted the gripper's internal accumulated command. Saved
expert suffix replay exposed that reset defect. Do not count its four model runs
as evidence about BPP.

## Measured pilot and current limitations

No new datasets, checkpoints or environments were downloaded. Only this independent
experiment directory was added: `scene.py`, `expert.py`, `recording.py`,
`rendering.py`, `inference.py`, `audit.py`, `run.py`, `run.sh`, and `test_experiment.py`.

| Expert | Goal success | Unintended collision | Minimum bowl/post gap | Transport EEF path |
|---|---|---|---|---|
| Original Narrow | yes | no | 6.69 mm | 0.474 m |
| Original Wide | yes | no | 21.13 mm | 0.489 m |
| Relocated Narrow witness | yes | no | 27.45 mm | 0.316 m |
| Relocated Wide witness | yes | no | 92.15 mm | 0.319 m |

All four experts have 756 control steps (37.8 seconds). The two original prompts
have exactly matching initial RGB. Their actual placement positions differ by
about 17 micrometres; their command targets agree to numerical precision. The
original paths penetrate the relocated obstacle by 36.97 and 32.34 mm, and both
alternative routes remain feasible. Fifteen expert checks and the visibility
checks pass. Five automated contracts pass, including the post-action observation
convention. Stored Narrow actions have also been
replayed from the original query state successfully without unintended collisions.

The current primary GPU experiment uses regenerated post-action prompts, seed 0,
800 steps per condition, the actual Spatial Combination checkpoint, strict
parameter loading, and the original 16 diffusion steps. Full-state pairing checks
include physics, mocap, OSC goals, gripper command and RGB. See
`outputs/gpu_aligned/rollout_results.csv` and `diagnostic_summary.json` for the
rerun results. Clearance differences from failed episodes must not be interpreted
as retained preference. Earlier 0/8 figures involved legacy prompts and are
superseded; do not pool them with current results.

The aligned primary rerun scored **0/4 goal successes**, with unintended contact
in three episodes and one collision-free failed episode. Both successful,
collision-free prompt comparisons remain unavailable. The alignment defect was
real, but fixing it alone did not solve the new experiment.

The held-state reset was validated separately: replaying the saved expert suffix
now succeeds without unintended collisions. Reconstruction of a legacy HDF5's
gripper command must apply `format_action` at **every physics substep** (25 per
policy step here). New HDF5s store that command state explicitly. Validation is
saved in `gpu_verified/transport_restore_validation_exact.json`.

Both BPP inference and EGL rendering ran on physical GPU 0, RTX A6000, UUID
`GPU-e0914423-1b44-589b-fdb3-0360e8f79b43`. Real CUDA kernels passed under PyTorch
2.7.1+cu118. EGL device 0 was matched to this UUID using EGL's device UUID and
CUDA ordinal query; its CUDA default ordinal is 5. The BPP process was observed
using 3748 MiB on that UUID. The earlier `gpu_verified/gpu_selection.json` records
the renderer/device mapping; the current rollout log records fresh CUDA preflight.
Commands
ran outside the Codex workspace sandbox after approval; a missing driver inside
the sandbox is not a model outcome.

Reproduce the complete primary experiment with no new downloads:

```bash
bash bpp/experiments/clearance/run.sh all --device cuda \
  --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --output bpp/experiments/clearance/outputs/new_run
```

`--egl-device 0` is the verified renderer index on this server; use a separately
verified index on another server. For Codex, request command-scoped execution
outside the sandbox when NVIDIA device nodes are hidden; permanent Full access
is unnecessary. GPU execution is now verified. Still unresolved: establishing
successful preference retention/adaptation, multi-seed statistics, and any
preference-specific training. This pilot makes no claim of absent model capacity.
