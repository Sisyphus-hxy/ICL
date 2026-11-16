# Natural procedure variation in the downloaded LIBERO demonstrations

Question: do the 50 raw demonstrations per downloaded task contain stable,
multi-modal *execution procedures* — or is the spread just scene placement
jitter plus random motion noise?

**Short answer:** the variation is real and largely *not* scene-driven, but it
is mostly **continuous**, not cleanly multi-modal. The strongest discrete
structures are (a) a dominant-vs-minority spread of *approach arcs* and
(b) re-grasp / truncated-release events in the official task (9/50 demos).
Transport height, lateral deviation and speed form unimodal continua —
no evidence of "two transport styles".

## Data analysed

| dataset | task | demos | steps @ 20 Hz | provenance |
| --- | --- | --- | --- | --- |
| `libero_spatial` | bowl between plate & ramekin → plate | 50 | 75–181 | official LIBERO-v1 (`tag: libero-v1`, bddl `libero/libero/...`) |
| `libero_spatial_selected_combinations_view` | bowl at table center → cookies box | 50 | 115–181 | generated set (`umi_day` run, `libero_spatial_extra` BDDL) |

Everything comes from the recorded `states` / `obs` / `actions` in the hdf5 —
no replay, no scene edits. Object poses are decoded from the flattened MuJoCo
state (`[t, qpos(48), qvel(43)]`; free-joint slices verified against the
embedded `model_file`): `bowl1@10`, `bowl2@17`, `cookies@24`, `ramekin@31`,
`plate@38`. EEF = `obs/ee_pos`, gripper command = `actions[:,6]` (−1 open /
+1 close), finger travel = `obs/gripper_states`.

## Method

- **Phase split** per demo: `approach` → close-command; `grasp` → object
  liftoff (z > rest+1 cm); `transport` → descent start (last point within
  1.2 cm of peak carry z); `place` → end.
- **Canonical alignment** removes scene placement: translate obj₀ to origin,
  rotate obj₀→goal onto +x′. **Progress normalisation** removes duration:
  each phase is arc-length-resampled to 60 points.
- **Metrics**: object-local grasp offset (quat⁻¹·(eef−obj) at contact),
  approach travel azimuth / source azimuth / elevation, pre-grasp hover,
  transport lift (obj_z − rest), lateral deviation in the canonical frame,
  path length/curvature, speed/accel/jerk (5th/50th/95th pct), phase
  durations, gripper-flip count.
- **Mode checks**: 1-D GMM BIC + bimodality coefficient + KDE mode count per
  metric; multivariate GMM on the 14-dim z-scored feature set (azimuth as
  cos/sin); circular k-means (k=3) on contact azimuth; Kruskal tests of
  cluster-vs-scene covariates; Spearman vs demo index; linear R² of every
  metric on scene covariates (obj₀, goal, distance).

## Result 1 — official `libero_spatial` task

### Approach — structured, multi-family, but continuous

Contact-direction azimuth spans **76° → ±180° → −166° contiguously**; the
polar histogram (fig_approach_clusters) has no gaps. Circular k-means
separates three overlapping families — interpretable, but they are lobes of
one distribution, not isolated modes:

| family | n | travel dir | entry | elevation | notes |
| --- | --- | --- | --- | --- | --- |
| right-hook | 28 | ~175° (−x′) | front-right, hooks left toward bowl | −56° | deepest grasp (−x off. 46 mm, z off. 31 mm) |
| diagonal | 16 | ~141° | front, up-left arc | −64° | intermediate |
| steep | 6 | ~100° | near-vertical descent | −65° | shallowest grip (z off. 22 mm) |

- All 50 approaches enter from the front half (−y′); none from behind.
- Not scene-forced: Kruskal p = 0.19–0.93 across obj₀/goal/dist; χ² vs
  session-third p = 0.14 (not a demonstrator-drift artefact).
- Verified on frames (fig_approach_frames.png): hook demos visibly come in
  from the bowl's goal side; steep demos descend vertically.
- Scene covariates explain only R² = 0.08–0.14 of azimuth components.

### Grasp — continuum

Radial offset 45 ± 7 mm, z-offset 15–50 mm, weak coupling to approach
family. No modes (BC = 0.28).

### Transport — continuum, no modes

| metric | mean ± std | range | structure |
| --- | --- | --- | --- |
| lift (obj_z − rest) | 71 ± 9 mm | 40–88 mm | unimodal (GMM k=1, BC 0.41) |
| \|lateral\| max | 50 ± 13 mm | 25–80 mm | unimodal |
| curvature (len/direct) | ~1.4 | 1.2–1.95 | heavy tail, 1 KDE mode |
| speed p50 | 0.21 ± 0.02 m/s | 0.12–0.24 | unimodal (the k=2 "mode" is a 1-demo outlier) |
| jerk RMS | 4.4 ± 0.9 m/s³ | | unimodal |

Speed and jerk drift slowly down over the session (Spearman r ≈ −0.35/−0.33,
p < 0.02) — mild demonstrator effect, not a mode.

### Discrete events — the clearest "modes" in the data

- **Re-grasp: 7/50 demos** (14 %) — gripper closes, re-opens and closes again
  before/while transporting (demo_11, 14, 17, 30, 31, 34, 37). fig_regrip_*
  shows first-grip failure → retry, e.g. demo_11 closes at 1.95 s, object
  never lifts, re-opens at 2.9 s, succeeds on the second grip.
- **Unreleased: 2/50** (demo_25, demo_35) — bowl ends on the plate (≤2.6 cm
  off-centre) but the release segment was never recorded; demo truncated
  while still holding.

### Global structure

14-dim feature GMM picks k=4 but silhouette = 0.14 — no coherent
multivariate "procedures"; the variation is structured per-dimension.

## Result 2 — generated `selected_combinations` task

Same overall shape, wider tails:

- **Approach**: dominant hook-left family 36/50 (dir ~−175°, sources in the
  +y′ far half — the gripper overshoots past the bowl then curls back), a
  10-demo minority (~112°) and a 4-demo right-entry (~22°). Kruskal vs scene
  p = 0.37–0.67; session χ² p = 0.52.
- **Transport**: heavy tails rather than modes — 9/50 slow (p50 ≤ 0.085 m/s
  vs 0.09–0.13 for the rest; 1-D GMM k=2, silhouette 0.74, skewed sizes) and
  11–14/50 wide-lateral (|y′| max > 0.06 m vs < 0.056); half the demos have
  loopy transports (curvature > 1.9). Slow ∩ wide barely overlap ({demo_5,
  demo_30}) — different axes.
- **Lift** mean 81 ± 25 mm, a few carries up to 125 mm; still 1 GMM mode.
- **Zero re-grips; every demo releases** — the generated collection is
  cleaner than the official one.

## Verdict

- Approach geometry varies far beyond noise (≈100° of entry spread, three
  interpretable families) and is independent of scene placement — but it is
  a **continuum with a dominant lobe plus shoulders**, not separable
  procedures.
- The only cleanly *discrete* procedure variation is the official task's
  ~18 % event rate: **failed-grip → re-grasp** (7) and **truncated release**
  (2). A "clean vs re-grasp" contrast is naturally available for paired
  study.
- Transport height/path/speed are unimodal continua in the official task;
  the generated task adds slow/wide tails but no stable secondary mode.
- **No stable two-or-more procedure modes were found** — we did not force a
  split. If a natural contrast pair is needed, the best candidates are
  (i) right-hook (n=28) vs steep approach (n=6) in the official task and
  (ii) clean vs re-grasp executions (43 vs 7), both verified scene-matched.

## Outputs

```
outputs/
  REPORT.md
  libero_spatial_bowl_between_to_plate/
    metrics.csv            # per-demo metrics + cluster labels
    phases.csv             # phase-boundary indices per demo
    initial_scene.csv      # all object init poses (comparability)
    summary.json           # BC / GMM / kruskal / representatives
    fig_distributions.png  fig_aligned.png  fig_grasp.png
    fig_approach_clusters.png  fig_traj_pca.png  fig_approach_frames.png
    fig_regrip_demo_*.png
    representatives/*.mp4  # approach_c{0,1,2}_*, regrip_*, extreme_*
  selected_comb_bowl_to_cookies/   (same structure)
```

Videos are rendered straight from stored `agentview_rgb | eye_in_hand_rgb`
at 15 fps; `representatives/README.json` maps each video to its cluster and
the demo's initial scene, confirming representatives are scene-comparable.
