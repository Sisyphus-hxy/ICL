# Original-scene demonstration audit and paired prompt swap

This experiment uses the installed LIBERO `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box` task and the released Spatial Combination BPP checkpoint. It adds no obstacle, moves no object or goal, and changes no model code or weights. It reads the original 712 MB HDF5; the existing Zarr cache remains untouched. No resources were downloaded.

`run.py analyze` compares original `demo_0` (135 steps), both earlier 756-step experts, and the new curved demo. `stage_statistics.csv` reports, per stage, duration, six-dimensional action norm, largest action component, EEF speed, gripper transitions, near-stationary steps (`EEF speed < 0.005 m/s`), small-action steps (`six-dimensional norm < 0.02`) and exact repeated control steps. The original HDF5 has no phase tags; its `demo_0` stage boundaries are inferred from the gripper transition at step 50, observed lift/transport at 82, descent at 113 and release at 122. These approximate labels apply only to the comparative statistics. All 50 source demonstration lengths are checked: 114–190 steps, median 142.

Across the 50 original episodes, the median episode mean six-dimensional action norm is 0.411 and median EEF speed is 0.103 m/s. The earlier Narrow expert's corresponding values are 0.108 and 0.026 m/s. It has 258/756 near-stationary steps, compared with 11/135 for source `demo_0`; both figures include the intended release wait. `source_50_overall_statistics.csv` and `stage_statistics.csv` provide the full measurements, including maximum action size and gripper switches.

The old Narrow transport took 300 steps at mean EEF speed 0.032 m/s; the original demo took 31 steps at 0.067 m/s. The old expert spent 89/180 approach-and-grasp, 64/110 lift, 22/50 align and 59/100 place steps near stationary. These are measured slowdowns and waits; the classification of a particular wait as *unnecessary* cannot be proven from velocity alone. The old expert also has 38 prompt chunks compared with 7 for the original.

The new candidate changes only actions 82–112 of source `demo_0`. A smooth sideward pulse produces a shallow bend while preserving every other action, the 135-step count and the original scene. It changes the native replayed transport EEF XY path by at most 14.8 mm and the final EEF position by 0.32 mm. Source and curved action sequences both replay to task success. **Both** original and curved prompts are recaptured through the same MuJoCo pipeline. The saved HDF5 files contain actual post-action observations, not image/observation arrays pasted from different rollouts. `hdf5_replay_validation.json` replays both saved files and verifies exact equality of both RGB cameras, EEF pose, gripper state, pre-action simulation states and terminal state. Every pre-transport action and observation agrees exactly across the two recaptured demonstrations.

The official configuration has `prompt_sequence_length=768`, `max_sequence_length=1000`, `prompt_chunk_n_actions=20` and `pad_end_prompt_actions=zeros`. This checkpoint uses **pair prompting**. Its sampler loads a whole single demonstration for both training pairs and evaluation prompts. `prompt_sequence_length` is used by the separate *sequence prompting* branch and does not truncate this pair-prompt run. The chunker keeps an incomplete last chunk by padding actions with zeros. Original and curved demos therefore each produce 7 chunks with 5 padded actions. The old 756-step demos produced 38 chunks with 4 padded actions, including their release actions. `max_sequence_length` sizes learned prompt positional embeddings to at most 50 chunks; it is not an automatic truncation to 1000 control steps. The 756-step prompt fits that capacity, but is much longer than the 114–190-step examples of this local task. We have not established the length distribution over all checkpoint training tasks.

The policy normalizes prompt observations and actions through its saved `Normalizer` in `normalize_obs_with_optional_prompt`; the explicit loader applies upstream RGB conversion, rotation conversion and chunking. `prompt_diagnostics.json` records actual tensor shapes, finite checks and action ranges for both prompts. Earlier `audit_parity_final/prompt_parity.json` verified bit-exact equality between the explicit downloaded source-demo prompt and the official dataset sampler for RGB, proprioception, actions and masks. The official runner reads a named Zarr cache; changing the source HDF5 in place could leave that cache stale. This experiment leaves the source and cache intact, saves both recaptured HDF5 files separately, and loads them directly through the same explicit prompt interface.

The 10 paired query states are the initial states of source demonstrations 0–9. The diffusion seed also equals 0–9, and is reset after each prompt. Both prompts in each pair receive identical simulation, camera, controller and gripper state after the same 12-step open-gripper warmup; the evaluator asserts this identity. The released checkpoint, 16 diffusion steps and 550-step rollout limit are shared. `rollout_results.csv` and `paired_results.csv` contain goal success plus four measured milestones: bowl lifted by 4.5 cm, bowl translated by 3.5 cm, bowl within 5.5 cm XY of the goal, and LIBERO task predicate success. These are operational milestones, not claims that the model deliberately executed named stages. Every rollout has a 20 Hz two-camera video and EEF/object trajectory CSV. `paired_eef_trajectories.png` and `demo_eef_transport.png` visualize the paths.

Run from the repository root (GPU commands require command-scoped sandbox escalation in Codex):

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
bash bpp/experiments/native_prompt_swap/run.sh generate --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 --output bpp/experiments/native_prompt_swap/outputs/paired10_controlled
bash bpp/experiments/native_prompt_swap/run.sh validate --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 --output bpp/experiments/native_prompt_swap/outputs/paired10_controlled
bash bpp/experiments/native_prompt_swap/run.sh evaluate --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 --trials 10 --max-steps 550 --output bpp/experiments/native_prompt_swap/outputs/paired10_controlled
bash bpp/experiments/native_prompt_swap/run.sh report --output bpp/experiments/native_prompt_swap/outputs/paired10_controlled
```

The older `outputs/paired10` result is exploratory because it paired the downloaded official HDF5 observations with a native recapture; pre-transport EEF observations differed by up to 1.3 mm. `EXPLORATORY_CONTROL_MISMATCH.json` explains this. Use `outputs/paired10_controlled` for the strictly paired result.

## Controlled GPU result

| Prompt | Lifted | Transported | Near goal | Goal success |
|---|---:|---:|---:|---:|
| Original actions, recaptured | 10/10 | 10/10 | 10/10 | 7/10 |
| Curved transport, recaptured | 9/10 | 9/10 | 9/10 | 5/10 |

Both HDF5 files replay exactly, and all 10 prompt pairs have exactly matching query simulation state, camera image, controller goal and gripper command. Seed 0 failed at lift under the curved prompt; the original prompt reached the goal area but failed final placement. Seeds 7 and 8 succeeded under the original prompt and reached the goal area under the curved prompt, but did not satisfy final placement there. Seeds 5 and 9 failed final placement under both prompts. The other five pairs succeeded under both. Detailed per-seed milestones and EEF paths are in `paired_results.csv`, `rollout_results.csv`, and `paired_eef_trajectories.png`.

The generated 135-step expert differs by at most 14.8 mm in transport XY and succeeds in native replay. The BPP paired outcome is a 2/10 difference in goal success on this small sample. It does not establish the direction or strength of a general procedure preference. No model training or checkpoint changes were made.

The prompt swap tests goal completion and path differences for a small perturbation. Successful or failed pairs alone do not establish whether BPP represents a general procedure preference.

## Transport sensitivity diagnostic

`sensitivity.py` reuses the same model, prompt loader, initial states and native
LIBERO scene. It generates 3 cm and 5 cm curved expert demonstrations with the
same 135 control steps. Their measured peak XY EEF displacements from the
original expert are **2.985 cm** and **4.991 cm**, respectively. Both experts
reach the original goal; their saved HDF5 observations and states replay exactly.
Before transport, all actions and observations match the recaptured original
demonstration exactly. The verified 1.5 cm HDF5 pair is reused through symlinks.

The 1.5 cm evaluator was rerun to save every full 16-step predicted action
chunk, the two-frame raw RGB/EEF/gripper query history at each model call, and
both observed gripper joint positions plus the executed gripper command at each
control step. The numeric EEF/bowl trajectories of its ten original-prompt
rollouts agree exactly with the earlier controlled run. The 3 cm and 5 cm
evaluators reuse those original-prompt rollouts and assert the same physical,
camera, controller and gripper initial state for every variant query. Their
curved-prompt rollouts save the same chunk and gripper data. All runs use the
same 550-step cap and diffusion seed per query.

Teacher forcing takes a fixed two-frame observation/history from each original
rollout at a transport query, then calls the same policy with the original,
1.5 cm, 3 cm and 5 cm prompts. It resets the diffusion seed **after** each
prompt is encoded. It also repeats prediction for each same prompt using the
same input and seed. Across 10 query states there are **29** fixed observations.
For seeds 6 and 9, transport lasts only 3 and 6 control steps, respectively,
shorter than the model's 12-step execution chunk. Their sample is the model
query immediately preceding transport and is marked `transport_entry_boundary`
in the CSV. The other eight seeds have actual in-transport model queries.

| Demonstrated peak lateral displacement | 1.5 cm | 3 cm | 5 cm |
|---|---:|---:|---:|
| BPP goal success (same 10 states) | 5/10 | 7/10 | 8/10 |
| Original-prompt success | 7/10 | 7/10 | 7/10 |
| Teacher-forced median full-action L2 difference | 0.01227 | 0.01214 | 0.00409 |
| Teacher-forced same-prompt maximum numerical difference | 0 | 0 | 0 |
| Median rollout transport XY path distance | 5.03 mm | 5.99 mm | 4.48 mm |
| Median signed rollout lateral offset | −3.60 mm | −3.49 mm | −1.98 mm |
| Rollouts moving in demo's positive lateral direction | 0/9 | 1/10 | 3/10 |

The goal-success cross tables (both succeed / original only / curved only /
both fail) are **5/2/0/3**, **6/1/1/2**, and **6/1/2/1** for the three
amplitudes. A path metric is undefined when a rollout never transports, so the
1.5 cm direction denominator is nine. `rollout_path_sensitivity.csv` includes
per-seed XY and XZ progress-aligned distances, a symmetric nearest-path distance,
lateral offsets and gripper-state differences. The plots under `figures/` show
Demo A/B and Rollout A/B in XY and XZ for each seed and amplitude. These metrics
compare carried-object transport segments, not all approach and release motion.

**Interpretation:** The action prediction changes above the exact-repeat
numerical control when only the demonstration changes, so BPP is sensitive to
the demonstrated intermediate motion under fixed query observations. This
sensitivity is not monotonic with demonstrated offset: the 5 cm variant yields
a smaller teacher-forced action difference and a smaller median rollout path
distance than 1.5 or 3 cm. The demonstrated bend is toward positive lateral Y,
while most measured rollout offsets are negative. This ten-state experiment
does not establish stable copying of the procedure direction or a general
preference representation. Success rate also does not measure that property.

Run the additional diagnostic after the verified 1.5 cm experiment:

```bash
bash bpp/experiments/native_prompt_swap/run.sh generate --target-offset-cm 3 --output bpp/experiments/native_prompt_swap/outputs/offset3cm --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0
bash bpp/experiments/native_prompt_swap/run.sh validate --output bpp/experiments/native_prompt_swap/outputs/offset3cm --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0
bash bpp/experiments/native_prompt_swap/run.sh generate --target-offset-cm 5 --output bpp/experiments/native_prompt_swap/outputs/offset5cm --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0
bash bpp/experiments/native_prompt_swap/run.sh validate --output bpp/experiments/native_prompt_swap/outputs/offset5cm --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0

# sensitivity_1p5 reuses the verified paired10_controlled HDF5 files by symlink.
bash bpp/experiments/native_prompt_swap/run.sh evaluate --capture-chunks --trials 10 --max-steps 550 --output bpp/experiments/native_prompt_swap/outputs/sensitivity_1p5 --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0
bash bpp/experiments/native_prompt_swap/run.sh evaluate --capture-chunks --trials 10 --max-steps 550 --baseline-output bpp/experiments/native_prompt_swap/outputs/paired10_controlled --output bpp/experiments/native_prompt_swap/outputs/offset3cm --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0
bash bpp/experiments/native_prompt_swap/run.sh evaluate --capture-chunks --trials 10 --max-steps 550 --baseline-output bpp/experiments/native_prompt_swap/outputs/paired10_controlled --output bpp/experiments/native_prompt_swap/outputs/offset5cm --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0

bash bpp/experiments/native_prompt_swap/run.sh teacher_forced --baseline-output bpp/experiments/native_prompt_swap/outputs/sensitivity_1p5 --variant-outputs bpp/experiments/native_prompt_swap/outputs/sensitivity_1p5 bpp/experiments/native_prompt_swap/outputs/offset3cm bpp/experiments/native_prompt_swap/outputs/offset5cm --output bpp/experiments/native_prompt_swap/outputs/sensitivity_analysis --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0
bash bpp/experiments/native_prompt_swap/run.sh sensitivity_report --baseline-output bpp/experiments/native_prompt_swap/outputs/sensitivity_1p5 --variant-outputs bpp/experiments/native_prompt_swap/outputs/sensitivity_1p5 bpp/experiments/native_prompt_swap/outputs/offset3cm bpp/experiments/native_prompt_swap/outputs/offset5cm --output bpp/experiments/native_prompt_swap/outputs/sensitivity_analysis
```

The exact controlled files are in `outputs/sensitivity_analysis/`: teacher-forced
per-query CSV and full action chunk NPZ, path/success CSV, 30 comparison figures,
config, summary and verification JSON. Rollout action chunks and gripper traces
remain in their respective variant output directories. No additional data,
model weights, training environment or simulator assets were downloaded.
