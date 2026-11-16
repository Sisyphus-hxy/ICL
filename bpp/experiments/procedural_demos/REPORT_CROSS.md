# 2×2 Crossed Prompt-Query Diagnostic

**Question:** is BPP's execution style driven by the demonstration's
procedure, the current query's visual attributes, or their interaction?

**Setup.** Same checkpoint (`liberogen_spatial_combination_behavior_prompting.ckpt`,
strict load), no training. Prompts are the 20 procedural demos generated in
this package; each query layout is paired with prompts from a *different*
layout (`prompt_cycle`). Query visual variants share identical physics
(mocap-only differences), verified by identical post-warmup
`sim_state_sha256` inside each pair and 5 distinct hashes per task.

| | Prompt level 1 | Prompt level 2 |
|---|---|---|
| **Task A (cup)** | `normal` = empty-bowl demo | `gentle` = filled-bowl demo (3-5× slower, lower jerk) |
| **Task B (post)** | `narrow` = plastic-post demo (~5-7 mm graze) | `wide` = glass-post demo (~18-26 mm detour) |

Query visuals: A `{empty, filled}` (water mocap disk), B `{plastic, glass}`
(post `rgba` only). Per task: 5 layouts × 2 diffusion seeds (1, 2 —
native-baseline successes, maximising transport coverage) × 2 visuals × 2
prompts = **40 rollouts**, 12-step open-gripper warmup, 550-step cap.

## Task A — Filled Cup / Gentle

Success: 37/40 (A1 empty×normal 9/10, A2 empty×gentle 9/10,
A3 filled×normal 10/10, A4 filled×gentle 9/10). Failures: layout0+seed2
never lifts the bowl (both prompts, empty query), and one filled×gentle
`goal_placement` miss — prompt-independent flakiness, not a semantic
effect.

Paired differences (`gentle − normal`, n=20 pairs unless noted):

| Metric | Empty query | Filled query | All |
|---|---|---|---|
| transport speed (m/s) | +0.0002 | +0.0020 | **+0.0011** |
| acc p95 | +0.10 | +0.35 | +0.23 |
| jerk p95 | +2.02 | +5.43¹ | +3.73 |
| tilt mean (deg, n=19) | +0.04 | +0.38 | +0.22 |
| transport duration (s) | −0.07 | +0.25 | +0.09 |

¹ dominated by the single failed rollout's erratic transport.

Visual effect (`filled − empty`, fixed prompt): speed **+0.006/+0.008 m/s
(faster)**, tilt **+0.93°/+0.76° (more tilt)** — the direction is *less*
careful, not more; consistent with mild OOD visual perturbation, not
physics reasoning. Interaction ≈ 0.

**Teacher-forced swap** (fixed obs/history/noise, n=10 pairs/probe):
post-warmup L2 = 0.011 (cosine 0.999998); transport-entry L2 = 0.064
(cosine 0.99992). Predicted action magnitudes identical across prompts
(0.444 vs 0.443) — the model does respond to the prompt but not by
reducing speed.

## Task B — Fragile Obstacle / Clearance

**Grasp-phase interference (independent of both factors).** Layout0's post
(0.04, −0.05) sits in the policy's approach-overshoot swept volume: all 8
rollouts fail at `lifted` — gripper hand/fingers push the post from
control step ~39 (8,101 monitored contacts), the bowl is never grasped.
Same mechanism as the earlier C0/meaningful-swap obstacle failures. The
other four layouts clear the approach and proceed.

**Transport clearance (from initial state, n=8 valid/cell).** Every
transport-reaching rollout — all four cells — *penetrates* the post
(min clearance −8 to −21 mm; thousands of sustained post contacts). The
policy takes the direct line and shoves through.

Paired differences on valid transports:

| Metric | prompt Δ wide−narrow | visual Δ glass−plastic |
|---|---|---|
| min clearance | +0.004 / −0.003 m | −0.006 / −0.013 m |
| post contacts | −1,525 / +644 | −976 / +1,192 |
| path length | −0.013 / +0.016 m | +0.005 / +0.034 m |
| success | +0.1 / −0.1 | +0.1 / −0.1 |

All within noise; nothing directionally consistent with the prompt's
23 mm detour. If anything glass queries show *more* penetration.

**Post-grasp probe** (contingency in the spec): same 2×2 started from the
shared `demo.states[t0]` bowl-in-hand state (history = duplicated frame,
no warmup — an open gripper would drop the bowl). 25/40 rollouts transport
meaningfully (l0 8/8, l5 8/8, l11 6/8, l8 3/8; **l3 8/8 froze** — bowl
held but never moved, a start-state artefact without motion history, not a
prompt/visual effect). On valid transports the mean min-clearance is
*positive* (+6 to +18 mm — the post is grazed/pushed less than in the
full run) and the prompt diff is +6 mm in the expected direction on both
visuals (sd 11–23 mm) — a weak signal, far from the demos' 20 mm gap and
statistically inconclusive.

**Teacher-forced swap**: transport-entry L2 = 0.107 (max 0.237, cosine
0.99976); the wide prompt produces a *directionally consistent* shift —
commanded Σdy +0.93 vs +0.67 over the 12-step chunk (+38% toward the
demo's +y detour side), identical across plastic/glass queries. So the
prompt does modulate open-loop predictions in the semantically correct
direction — but the closed loop never expresses it.

## Answers

1. **Prompt effect (procedure):** essentially zero in closed loop.
   A 3–5× demo speed gap yields +0.001 m/s execution change; a 20 mm demo
   clearance gap yields ±4 mm noise. Teacher-forced predictions show a
   small but real, semantically-ordered response (wider prompt → wider
   commanded excursion; gentle prompt → *no* speed reduction) — the
   conditioning exists at the prediction level but is weak and does not
   survive closed-loop execution.

2. **Visual effect:** small but nonzero in Task A — the filled query runs
   *faster* with *more* tilt (+0.007 m/s, +0.8°), i.e. opposite to the
   physical common-sense direction; this is OOD-visual noise, not
   "carefulness". In Task B, glass vs plastic changes nothing systematic
   (if anything, slightly worse clearance on glass).

3. **Interaction:** ~0 on every metric.

**BPP's execution style in this task is dominated by the policy's own
prior over the task geometry — the demonstration's procedure and the
query's visual semantics both modulate open-loop predictions at most
weakly and neither reaches closed-loop behaviour.** Per the prompt, this
cannot be read as absence of procedure understanding in general: the
prompt-conditioned prediction deltas are real (nonzero, directionally
consistent for clearance), just far too weak to matter online.

## Caveats

- Filled/glass are *visual proxies* — no liquid dynamics or fragility is
  simulated; any "careful" response would have to be learned association
  only.
- The post blocks the *policy's* approach on layout0 (8 rollouts lost);
  the post-grasp probe recovers transport coverage but introduces its own
  artefact (l3 freeze, frozen-history start).
- 10 pairs per cell; success/failure differences of ±1 rollout are noise.
- Clearance "min" is measured at physics substeps and can go negative
  under interpenetration — consistent with contact counts.
- Servo-generated demo segments (gentle transport, wide detour) have a
  different action style than human demos; may weaken prompt binding.

## Artifacts

Layout follows the AGENTS.md output-directory convention. Each rollout row
carries `name` = extensionless relative path, e.g.
`cross_obstacle/layout_00/plastic/seed1/narrow` —
`outputs / f"{name}_trajectory.csv"` resolves directly.

- `outputs/cross_cup/`, `outputs/cross_obstacle/`,
  `outputs/cross_obstacle_grasped/`:
  `layout_XX/<visual>/seed<Y>/<prompt>.{mp4,_trajectory.csv,_contacts.json}`;
  top-level `rollout_results.{csv,json}`, `state_hashes.csv`, `tables.json`,
  `teacher_forced.{json,csv}`, `cross_trajectories.png`,
  `evaluation_config.json`.
- Prompt demos: `outputs/cup/layout_XX/{normal,gentle}.hdf5`,
  `outputs/obstacle/layout_XX/{narrow,wide}.hdf5`.
- `eval_cross.py` (runner), `analyze_cross.py` (2×2 tables),
  `migrate_outputs.py` (one-off flat→nested migration, self-checking).
