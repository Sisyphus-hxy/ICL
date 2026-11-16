# Zero-shot Procedure Prompt Swap report

## Protocol

This run uses the existing `liberogen_spatial_combination_behavior_prompting.ckpt`
without training or changing BPP. Query layout `i` uses prompts from layout
`(i + 1) mod 5`, so a query's own demonstrations are never used as prompts.
Each task has five query layouts, two query/diffusion seeds (`0, 1`), and two
prompt conditions: 40 closed-loop rollouts / 20 paired conditions in total.
The two A6000s were used independently: closed-loop on
`GPU-e0914423-1b44-589b-fdb3-0360e8f79b43`, teacher-forced on
`GPU-5e94d3ee-5bff-5f9f-4d70-c0db8b0df272`. Both passed the real CUDA preflight.

Teacher-forced traces use the same saved query observation history and the same
diffusion seed for each prompt pair. Obstacle traces contain 26 transport action
chunks per pair; Gentle traces contain 11. Same-prompt repeated predictions
have maximum absolute native-action error `0` for all 20 pairs.

## Results

| metric | Obstacle Clearance | Gentle Transport |
|---|---:|---:|
| closed-loop rollouts | 20 | 20 |
| grasp success | 20/20 | 20/20 |
| transport success | 0/20 | 0/20 |
| place / goal success | 0/20 | 0/20 |
| collision-free rollouts | 2/20 | 1/20 |
| valid closed-loop prompt pairs | 0/10 | 0/10 |
| teacher-forced mean native chunk L2 | 0.0801 | 0.0316 |
| teacher-forced peak native chunk L2 | 0.5635 | 0.5320 |

The closed-loop rollouts commonly grasped the bowl but failed during transport;
therefore their clearance and dynamics differences are failure-conditioned
measurements. They are not evidence that the model selected Narrow/Wide or
Normal/Gentle. The paired rollout CSV retains the raw values and stage labels.

## Interpretation

1. **Does swapping the demonstration change BPP action prediction?** Yes.
   Teacher-forced action chunks changed for all 20 pairs, while same-prompt
   same-seed repeats were deterministic.
2. **Does the change follow the specified procedure?** Not established. No
   pair completed both transport and place successfully, and the closed-loop
   clearance/dynamic differences are not consistently aligned with the prompt.
3. **Is the change stable in closed loop?** No successful paired closed-loop
   comparison was observed under this protocol (0/20 valid pairs).

This is an ID diagnostic: the result separates prompt sensitivity from correct
procedure adherence and makes no claim about model capacity or training.

Artifacts include `rollout_results.csv`, `paired_rollout_summary.csv`,
`closed_loop_failures.csv`, `teacher_forced_summary.csv`, and per-condition
folders `<task>/layout_XX/seedY/` containing one `<procedure>.hdf5`/`.mp4`/
`_trajectory.csv`/`_contacts.json` per rollout plus `pair.png`,
`teacher_forced.csv`/`.npz`, and per-task `teacher_forced_diff.png` plots.
