# In-Context Procedural Constraint Transfer: minimal data

This directory contains the smallest reproducible dataset used for the two
transport-only procedures requested in the experiment:

* `obstacle_clearance`: five obstacle layouts, each with `narrow` and `wide`.
* `gentle_transport`: five layouts, each with `normal` and `gentle`.  The two
  demonstrations follow the same sampled spatial curve; only its time
  parameterization differs.

The generator reuses the installed LIBERO-Gen bowl-on-cookie-box BDDL, scene,
Panda OSC controller, and demonstration state. It does not change BPP source,
checkpoints, training data, or model inputs. Procedure names are HDF5/CSV
metadata only.

## Reproduce

From the repository root:

```bash
bash bpp/experiments/procedural_constraints/run.sh generate \
  --output bpp/experiments/procedural_constraints/data/minimal
bash bpp/experiments/procedural_constraints/run.sh render \
  --output bpp/experiments/procedural_constraints/data/minimal
bash bpp/experiments/procedural_constraints/run.sh validate \
  --output bpp/experiments/procedural_constraints/data/minimal
```

The default run uses CPU MuJoCo and software EGL rendering. `generate` records
real physics-step observations, proprioception, actions, pre-action states and
terminal states. `render` replays those saved states and adds camera RGB and
H.264 videos without changing actions or proprioception. `validate` replays the
HDF5 actions, checks post-action observation alignment, success and collision
status, and verifies each paired constraint.

The generated directory keeps each demo under `<task>/layout_XX/`: one
`<procedure>.hdf5`, `_trajectory.csv`, `_statistics.csv`, `_contacts.json`,
`.mp4` per procedure plus a `pair.png` comparison per layout. `statistics.csv`,
`manifest.json`, `initial_state.npz`, `scene.xml` and validation reports stay
at the top level.
`config.json` records the five obstacle layouts, controller/time-parameterization
values, camera stride, and source task used by the run.
`failed_samples.csv` is empty for the checked-in minimal run; any future
failed replay or pair gate is listed there instead of being silently dropped.

The executed minimal run contains 20/20 successful, collision-free demos. HDF5
replay passed for all 20 (maximum sampled post-action EEF error 0.151 mm;
maximum terminal-state error 2.65e-4). Narrow clearance is 6.69–12.66 mm and
Wide exceeds it by 14.44–15.98 mm. Gentle transport takes 2× as long as
Normal, with speed, acceleration, and jerk ratios of 0.51, 0.28, and 0.37;
normalized path deviation is below 7 mm and every transport sample moves.

The zero-shot prompt-swap implementation is [zero_shot.py](./zero_shot.py).
Its completed GPU run and concise interpretation are in
`data/zero_shot/REPORT.md`; the raw 40 rollouts and 20 teacher-forced traces
are organized under `<task>/layout_XX/seedY/` folders beside that report.

## Measurements

Obstacle clearance is the minimum signed MuJoCo geometry distance from the
carried bowl collision geometry to the added obstacle during transport, sampled
at every physics substep. Gentle transport reports EEF speed, acceleration,
jerk, and maximum carried-object tilt (angle of the bowl's local z axis from
world z). The recorder also stores the object pose, EEF pose, gripper command,
phase, collision contacts and goal-success flag at 20 Hz.

The gentle trajectory uses twice as many points on the same arc-length curve;
each point receives one nonzero proportional-control step. It therefore takes
longer through the curve instead of using repeated frames, waits, or zero-action
padding.
