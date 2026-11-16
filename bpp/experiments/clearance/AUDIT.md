# Reproduction and experiment audit

This audit separates an executable original-task positive control from the new
clearance experiment. All runs use the existing Spatial Combination checkpoint,
PyTorch 2.7.1+cu118, the RTX A6000 UUID below, strict weight loading and the
original 16 diffusion steps. No datasets, weights or environments were downloaded.

## Confirmed implementation defect and correction

Upstream `scripts/libero/create_dataset.py` calls `env.step(action)` before
recording the corresponding RGB/proprioception row. Our generated demonstrations
originally stored pre-action observations. This introduced a one-step convention
mismatch in the new prompts; original downloaded demonstrations were unaffected.

`recording.py` now writes post-action observations and preserves pre-action states,
the terminal state and the true initial EEF position for replay/metrics.
`rendering.py` renders the matching post-action state for every sampled prompt
frame. `inference.py` rejects legacy experimental prompts and casts actions to
float32 before upstream rotation conversion, matching the official loader exactly.
The float32 correction removed a separate maximum numerical difference of 1.49e-8.

Five regression checks pass, including a native simulation check that every saved
EEF observation is the result returned by the matching action. The four regenerated
experts pass all 15 expert gates, and their saved EEF observations match the
post-action trajectory CSV exactly. `outputs/gpu_aligned/observation_alignment_validation.json`
records that check. RGB visibility/pairing checks also pass.

Exclude legacy model runs in `gpu_verified`, `gpu_transport_verified`,
`audit_custom_narrow` and `audit_custom_wide` from current interpretation.
`PROMPT_ALIGNMENT_DEFECT.json` marks each directory; successful expert physics
remains valid. Earlier 0/8 model figures are superseded. The legacy held-state
model diagnostic has not been rerun with the corrected prompt convention.

## Positive controls and controlled substitution

The official evaluator on the unchanged LIBERO task scored **2/3** with a
550-step budget (`outputs/audit_official/official.log`). It uses the official
initial-state sampler, so this is not episode-for-episode parity with the direct
controls, which use the first three original demonstration initial states.

The original demonstration processed through `load_prompt` matches the official
dataset/collation tensors exactly: both RGB cameras, all proprioception, actions
and masks have maximum difference zero
(`outputs/audit_parity_final/prompt_parity.json`).

The direct original-scene controls hold the source initial state, seed, 12-step
open-gripper warmup, checkpoint, diffusion seed and execution code fixed, replacing
only the prompt. Their final budget is **800 steps** for all prompts, exceeding
the generated experts' 756 steps. The earlier 550-step generated-prompt controls
are preliminary and must not be used to conclude full-horizon failure.

Final direct-control results are in `outputs/audit_original_800`,
`outputs/audit_aligned_narrow_800` and `outputs/audit_aligned_wide_800`, each with
`direct_results.json`, command/config metadata, EEF CSVs and videos. Aggregate
results are recorded in `outputs/reproduction_audit.json`.

| Original query scene; 800-step budget | Goal successes |
|---|---:|
| Original prompt | 2/3 |
| Corrected Narrow prompt | 0/3 |
| Corrected Wide prompt | 0/3 |

Thus failure persists with the new demonstrations even without modifying the
query scene. This isolates a problem at the new-demonstration adaptation/generalization
stage; it does not identify a unique defect or show absence of model capacity.

The corrected primary clearance experiment (`outputs/gpu_aligned`) ran four
800-step prompt/layout conditions: **0/4 goal successes**, three episodes with
unintended contact, one collision-free failed episode. There are no successful,
collision-free prompt comparisons. This remains a **zero-shot diagnostic** and
cannot establish whether BPP can learn clearance preferences.

The one-step mismatch was real, and correcting it alone did not solve the new
experiment. Potential remaining causes include the translated object layouts,
new obstacle, route shapes and demonstration timing. In this task's 50 local
original demos, length is 114–190 steps (median 142), versus 756 for our experts.
This is a measured difference for this task, not a claim about all checkpoint
training data. No unique remaining root cause has been established.

The original-task controls establish an executable checkpoint and inference
path. Three episodes do not reproduce the full benchmark or its reported score.
A successful goal baseline with self-generated demonstrations is still needed
before interpreting differences as retained execution preferences.

## Commands

Run from the repository root. Inside Codex, GPU commands require command-scoped
execution outside the workspace sandbox, as instructed by the root `AGENTS.md`.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv

# Official original-task positive control; reuses the existing rollout cache.
bash bpp/experiments/clearance/run.sh audit --audit-mode official \
  --device cuda --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --trials 3 --max-steps 550 --output bpp/experiments/clearance/outputs/audit_official

# Exact original-prompt tensor parity; requires no GPU or download.
bash bpp/experiments/clearance/run.sh audit --audit-mode parity \
  --device cpu --software-render \
  --output bpp/experiments/clearance/outputs/audit_parity_final

# Matched-budget original scene and original prompt.
bash bpp/experiments/clearance/run.sh audit --audit-mode direct \
  --device cuda --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --trials 3 --max-steps 800 --output bpp/experiments/clearance/outputs/audit_original_800

# Regenerate aligned expert prompts; validate physics, visibility and pairing.
bash bpp/experiments/clearance/run.sh experts \
  --device cuda --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --output bpp/experiments/clearance/outputs/gpu_aligned

# Repeat for wide by changing both "narrow" names to "wide".
bash bpp/experiments/clearance/run.sh audit --audit-mode direct \
  --audit-prompt "$PWD/bpp/experiments/clearance/outputs/gpu_aligned/expert_original_narrow.hdf5" \
  --device cuda --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --trials 3 --max-steps 800 \
  --output bpp/experiments/clearance/outputs/audit_aligned_narrow_800

# Full primary rerun; videos, trajectory plots and result CSVs are saved.
bash bpp/experiments/clearance/run.sh rollouts \
  --device cuda --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --max-steps 800 --output bpp/experiments/clearance/outputs/gpu_aligned

source bpp/scripts/common.sh
bpp/env/bpp/bin/python -m unittest discover \
  -s bpp/experiments/clearance -p test_experiment.py -v
```

Changed implementation files: `audit.py`, `run.py`, `inference.py`, `recording.py`,
`rendering.py`, `test_experiment.py`. Documentation: this file and `README.md`.
All changes remain inside this independent experiment directory.
