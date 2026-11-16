# Meaningful Procedure Prompt Swap — Native-Scene Report

BPP checkpoint `liberogen_spatial_combination_behavior_prompting.ckpt`, task
`pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box`
(native LIBERO geometry, native demo_0..9 query states, 12-step open-gripper
warmup, 550-step closed-loop limit, strict checkpoint load, matched diffusion
seeds 0–9 per prompt pair).

## Experiment 1: Gentle Transport

### Setup validity

- Query scene = native LIBERO (bowl −0.075,0.007; box 0.086,0.018); verified
  10 distinct post-warmup `sim_state_sha256` — no canonical-state reuse — and
  each normal/gentle pair shares an identical query identity.
- `normal` prompt = recaptured native demo_0, bit-identical to the validated
  7/10 `paired10_controlled` prompt (eef pos/ori, gripper, both RGB streams,
  actions).
- `gentle` prompt = same demo with the transport segment (steps 82–112) time-
  stretched ×2: identical actions outside transport, same spatial path
  (max EEF deviation 4.7 mm, bowl 4.5 mm), same goal; transport speed
  0.033 vs 0.068 m/s (2.04×), acc p95 4.2× lower, jerk p95 6.0× lower.
- Both demos replay successfully under physics (`demo_validation.json`).

### Results (10 paired rollouts)

| prompt | success | lifted | transported | near-goal | unintended collisions |
|---|---|---|---|---|---|
| normal | 7/10 | 10 | 10 | 10 | 1 |
| gentle | 8/10 | 9 | 9 | 9 | 2 |

Per-seed (`rollout_results.csv`):

- Normal arm reproduces the historical baseline **exactly**: successes at
  steps 143/126/140/134/115/125/196 identical to `paired10_controlled`,
  failures on seeds 0/5/9 (all reach the goal region, fail final placement).
- Gentle arm: same successes plus seed 5 rescued (442 steps), but a **new**
  failure on seed 0 — the policy wandered to the goal box without ever
  grasping (bowl unmoved; 940 finger↔goal-box contact events from step 252).
- Success delta +1 for gentle is within small-sample noise (one seed flip in
  each direction: normal loses seed 0→gentle-new-failure, gains seed 5).

### Procedure transfer (the key measurement)

7 seeds succeed under both prompts (1,2,3,4,6,7,8). Transport-phase metrics:

| metric | demo normal | demo gentle | rollout normal | rollout gentle | rollout ratio |
|---|---|---|---|---|---|
| mean EEF speed (m/s) | 0.068 | 0.033 | 0.1095 | 0.1081 | **0.987** |
| transport duration (s) | 1.55 | 3.10 | 2.12 | 1.98 | 0.93 |
| peak acc (m/s²) | — | — | 1.41 | 1.36 | 0.96 |
| peak jerk (m/s³) | — | — | 26.3 | 24.9 | 0.95 |

A **2× demonstrated speed difference transfers as ~1% in closed loop**. The
policy's own transport speed (~0.11 m/s) is ~60% faster than even the normal
demonstration — execution speed is governed by the policy/scene prior, not
copied from the prompt procedure. Spatial paths also match within ~mm-level
differences (see `paired_eef_trajectories.png`,
`paired_transport_speed.png`).

### Failure localization

- seed 0 normal: bowl placed 3.3 cm from goal beside the box; 8031 contact
  events with the `plate_1` distractor from step 227 (identical to history).
- seed 0 gentle: policy drifted toward the goal box at ~step 250 without
  grasping — a qualitative behavior change induced by the prompt swap (new
  failure mode, not a speed effect).
- seed 5 gentle: recovered a struggling attempt and placed on the box with 16
  goal-box contacts (succeeded despite contact).
- seed 9 both: bowl ends 3.2–3.5 cm off the box — same placement failure as
  the historical baseline.
- Contacts are classified per event (`*_contacts.json`); `any_contact` and
  `unintended_collision` are reported separately — no gripper-bowl grasp
  contacts are counted as failures.

### Gentle conclusion

Under fully de-confounded conditions (native scene, validated baseline
prompt, paired states/seeds, native warmup/horizon): **BPP preserves task
success under a meaningful procedure swap, but does not transfer the
demonstrated transport dynamics into closed-loop execution.** The 2× speed
contrast visible in the prompt is not reproduced (0.987 ratio); behavior
differences manifest only as stochastic outcome flips (8/10 vs 7/10, within
noise) and one qualitative failure-mode change (seed 0 grasp skip).
Procedure prompting at this checkpoint is task/goal conditioning, not
dynamics conditioning.

## Experiment 2: Obstacle Clearance

### Setup validity

- Native bowl/target/robot/query states preserved (`canonical_state` from
  demo_0..9 — the `Scene` C0 geometry override is bypassed, verified by 10
  distinct post-warmup `sim_state_sha256` and identical identities inside
  each narrow/wide pair).
- One mocap cylinder (r 2.5 cm, h 18 cm) added at (0.05, −0.04, 0.95), near
  the transport corridor. Chosen by grid search over demo_0 state snapshots:
  ≥17 mm from the **demonstration** grasp/approach/lift swept volume, ≥5 mm
  from the place segment, and intersecting the direct transport path
  (direct replay: 170 post contact events, bowl clearance −38 mm).
- Demos share the identical replayed approach/grasp/lift and release
  segments and the identical servo-generated place descent; only the
  transport segment differs:
  - `narrow`: servo transport with a +y apex 0.103 → bowl min clearance
    **5.0 mm**, 0 contacts, success (the converged end pose caps clearance
    at ~6 mm — the 9 cm corridor is too short for anything tighter to be
    the bottleneck).
  - `wide`: +y apex 0.34 plus a +25 mm transport-end shift → bowl min
    clearance **26.9 mm**, 0 contacts, success.
  - Clearance contrast 5.4× (>20 mm), both demos collision-free
    (`demo_validation.json`, `*_demo.mp4`).

### Results (10 paired rollouts)

| prompt | success | lifted | transported | near-goal | post contact during rollout |
|---|---|---|---|---|---|
| narrow | 4/10 | 9 | 9 | 7 | 10/10 |
| wide | 6/10 | 9 | 9 | 7 | 10/10 |

- The post is a real hazard: 7/10 baseline success drops to 4–6/10.
- **Every rollout contacts the post** (all 20 `unintended_collision=true`).
  All 18 rollouts reaching transport show negative min transport clearance
  (bowl-geom penetration during contact; the mocap post cannot be pushed).
- First post contact: transport phase at steps ~60–92 in 18 rollouts; in
  seed 0 (both prompts) the **policy's own** grasp approach hits the post
  at step 38 → never lifts. The post was verified outside the *demo's*
  swept volume but the policy's approach trajectory differs from the
  demo's — the same mechanism that failed C0's grasp.
- Successes occur *despite* contact — the bowl slides around the smooth
  post and still reaches the box (seed1 both, seed4 both, seed5 both,
  seed6 both, seed8/9 wide).

### Procedure transfer

| metric | narrow demo | wide demo | narrow rollouts | wide rollouts |
|---|---|---|---|---|
| min transport clearance | +5.0 mm | +26.9 mm | ≤−0.6 mm (contact in all) | ≤−0.65 mm (contact in all) |
| mean transport clearance | — | — | +2.4 mm | +2.4 mm |
| max EEF y in transport | ~0.10 | ~0.34 | 0.047 | 0.052 |

- The wide demo's 23 cm-larger lateral excursion transfers as ~5 mm of max
  EEF-y difference in closed loop — essentially zero spatial transfer.
  `paired_eef_trajectories.png` shows every rollout hugging the direct
  line through the post regardless of prompt.
- wide 6/10 vs narrow 4/10 success is within small-sample noise and driven
  by contact-dynamics luck, not by reproduced clearance (identical contact
  rates and identical mean clearance).

### Obstacle conclusion

Under native, non-confounded conditions the BPP checkpoint does **not**
condition closed-loop avoidance geometry on the demonstrated procedure:
a clearly-validated narrow-vs-wide clearance contrast (5 mm vs 27 mm,
collision-free demos, same goal) produces identical rollout geometry —
straight through the obstacle. The obstacle instead acts as a passive
physical hazard that reduces success from 7/10 to 4–6/10.

## Combined answer

**Does a physically-meaningful demonstration procedure change BPP's
closed-loop behavior?** No — not on either axis tested.

- Transport **dynamics** (gentle): 2× demo speed contrast → 0.987× rollout
  speed ratio.
- Transport **geometry** (obstacle clearance): 5.4× clearance contrast →
  identical paths, identical mean clearance, identical contact rates.

The prompt still determines *what* the policy does (task, goal, grasp) —
the same checkpoint reaches 7/10 with the baseline prompt — and prompt
swaps perturb outcomes stochastically (8/10 vs 7/10; 6/10 vs 4/10) and
occasionally change failure modes qualitatively (gentle seed 0). But the
executed trajectory's speed and spatial clearance are governed by the
policy's own prior and the scene, not by the procedure shown in the
prompt. Caveats: 10 seeds per arm is small for success-rate differences;
the servo-generated transport segment has a different action style than
human demonstrations, which may limit how strongly the model can bind to
it; results are specific to this checkpoint and scene.

