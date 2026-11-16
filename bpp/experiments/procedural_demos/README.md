# Procedural Constraint Demonstrations (scene/data construction only)

Semantically-labelled demonstration pairs where the *movement difference*
carries an explicit object-property meaning. Scope: scene construction and
demonstration generation/validation only — **no training, no BPP changes, no
policy evaluation** were performed.

Native scene: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box`
(`libero_spatial_selected_combinations_view`), demos 0–4 provide five distinct
layouts (different bowl/box poses and transport paths).

## Task A — Filled Cup → Gentle Transport

| variant | vessel contents | procedure |
| --- | --- | --- |
| `empty_normal`  | empty bowl | native transport replay |
| `filled_gentle` | 16 free blue spheres (r=8 mm) filling the bowl | slowed/levelled servo along the same spatial path |

- **Asset limitation (documented, not hidden):** the selected BDDL task has no
  cup asset; the manipulated object is `akita_black_bowl_1`. The bowl is used
  as a graspable *vessel proxy*. If a real cup asset/task is required, a
  different BDDL or asset set must be introduced — this dataset does not claim
  otherwise.
- **Contents are real dynamics, not a visual proxy:** free-joint blue spheres
  rest on an invisible mocap plug (`cup_fill`, collides only with the drops)
  glued inside the bowl at rim height. The bowl interior is ~70 mm deep, so
  floor-level contents could never spill — the plug puts the heap at the rim
  where moderate tilt genuinely rolls balls out onto the table. Measured
  behaviour: 0 spills through the full native replay (grasp/lift/transport/
  place), spill onset ≈25–30° tilt, 6+ drops out at 45° tip-over.
- Empty vs Filled therefore differ **physically** (not only visually): Filled
  carries loose contents that can be lost by careless handling; Empty parks
  the spheres outside the workspace. `n_drops_out` is recorded per control
  step in the trajectory CSVs.
- Drop bodies extend qpos/qvel (+7/+6 each): demo `states` are recorded in
  the augmented layout; `scene.set_state` accepts both native- and
  augmented-length states.
- Gentle is produced by a pure-pursuit servo (`chase_path`): a carrot point
  advances along the arc-length-resampled native eef path at a prescribed
  step budget → continuous real motion, ~3–5× slower than normal, lower
  acceleration/jerk; orientation is held level for the first 65% of the path
  then blended back to the demo's place-entry orientation (placing needs the
  native wrist tilt). No static frames or zero-action padding.
- Same grasp/lift prefix (identical replayed actions), same place/release
  (identical replayed actions), same start/end poses.

## Task B — Fragile Obstacle → Clearance

| variant | post appearance | procedure |
| --- | --- | --- |
| `plastic_narrow` | opaque orange cylinder | native direct transport (no extra margin) |
| `glass_wide`    | translucent light-blue cylinder | servoed lateral detour + 25 mm endpoint shift |

- **Identical physics across materials:** same mocap body, same cylinder
  (`r=25 mm, h=180 mm`), same position per layout, `contype/conaffinity=1`,
  friction/density unchanged. Only `rgba` differs. `post_xyz` is recorded per
  demo; the pair check asserts `allclose`.
- Glass "fragility" is expressed **only** through the clearance-margin
  proxy — no breakage/fracture simulation.
- Post placement is searched per layout (`auto_place`): it must graze the
  *late* segment of the direct transport path (~3–8 mm), stay ≥5 mm out of
  the whole approach/grasp swept volume, ≥15 mm off the shared transport-start
  pose, and ≥5 mm off the place descent. Layouts where no post satisfies all
  constraints are skipped and reported.
- Generated layouts: **0, 3, 5, 8, 11** (demos 1, 2, 4, 6, 7, 9, 10 had no
  feasible post position — the dominant veto was `place`, i.e. the descent
  corridor is too crowded to keep the post ≥5 mm clear).
- Both demos share the same grasp/lift prefix, the same place path (tracked
  to the demo's own place trajectory) and release tail.

## Outputs

Layout follows the AGENTS.md output-directory convention — parameters live
in the path, leaf filenames keep only the procedure role:

```
outputs/
  cup/layout_{00..04}/           # Task A
    normal.{hdf5,mp4}  normal_trajectory.csv  normal_contacts.json
    normal_statistics.csv
    gentle.{hdf5,mp4}  gentle_trajectory.csv  gentle_contacts.json
    gentle_statistics.csv
    pair.png                     # paired trajectory comparison
    pair_visual.png              # visual-separability comparison
  obstacle/layout_{00,03,05,08,11}/   # Task B, same shape with narrow/wide
  demo_stats.csv  manifest.csv  validation_report.json
```

`name` fields (HDF5 attrs, `demo_stats.csv`, `manifest.csv`,
`validation_report.json`) are extensionless relative paths, e.g.
`cup/layout_00/normal` — `outputs / f"{name}_trajectory.csv"` resolves
directly.

- `*.hdf5` — LIBERO-format demonstrations (post-action obs alignment,
  pre-action states), plus semantic attrs: `name`, `task`, `variant`,
  `procedure`, `layout`, `source_demo`, `window`, `drops`/`n_drops`,
  `material`/`post_xyz`, `peak`, `y_shift`. These labels are **not** part
  of the observations.
- `*.mp4` — paired videos (agentview + eye-in-hand).
- `*_trajectory.csv`, `*_contacts.json` — per-step phase-labelled kinematics,
  sub-step minimum clearance, and contact events (normal grasp contacts
  excluded by construction).
- `demo_stats.csv`, `manifest.csv` — per-demo metrics and dataset manifest
  (top-level summaries; per-leaf `<proc>_statistics.csv` carries each row).
- `pair.png` / `pair_visual.png` — pairwise trajectory and
  visual-separability comparisons (written by `validate.py`).
- `validation_report.json` — replay validation (goal success, contacts,
  clearance, replay deviation, pair checks).
- `cross_{cup,obstacle,obstacle_grasped}/` — 2×2 prompt-query diagnostic
  rollouts (`layout_XX/<visual>/seed<Y>/<prompt>.*`), see
  `REPORT_CROSS.md`.

## Reproduce

```bash
# Task A (5 layouts x Empty/Normal + Filled/Gentle)
./run.sh cup      --output outputs --layouts 5 --egl-device 0
# Task B (5 layouts x Plastic/Narrow + Glass/Wide)
./run.sh obstacle --output outputs --layouts 5 --egl-device 0
# Replay validation + pair checks + figures
../../env/bpp/bin/python validate.py --input outputs --egl-device 1
```

## Validation summary (see `outputs/validation_report.json`)

- **Replay:** 20/20 HDF5s re-executed in fresh simulation → all
  `goal_success=True`, replay trajectory deviation mean ≤0.2 mm, max ≤0.5 mm.
- **Cup contrast** (per `demo_stats.csv`): gentle transport speed p95
  0.022–0.035 m/s vs normal 0.104–0.134 m/s (≈3–5× slower), jerk p95 reduced
  5–20×, mean tilt reduced (e.g. 7.5°→6.4°, 7.9°→6.0°); same spatial path
  (paired path deviation mean 3.5–5.3 mm over ~7–9 cm).
- **Obstacle contrast:** plastic/narrow min clearance 5.0–6.6 mm (direct
  native path) vs glass/wide 18.0–26.4 mm (≈3–5×); zero obstacle contacts in
  every demo (transport and place), all successful.
- **Visual separability** at BPP input resolution (128×128): every pair
  differs in 2.4–7.2% of pixels (>0.15 intensity); water surface and post
  material are directly visible in `pair_visual.png`.
- **Pair invariants verified:** identical post position/geometry across
  materials (`post_identical=True` ×5), shared grasp/place/goal.

## Known limitations

- No real cup asset in this BDDL task — the black bowl is a vessel proxy.
- The "contents" are discrete spheres on an invisible support plug, not a
  liquid: no sloshing waves, and the heap is the same arrangement every
  reset (plus settle noise). Spilling is real but discrete.
- Gentle transport is servo-generated; its action stream is controller-style,
  not human-demo-style (same caveat as earlier obstacle experiments).
- Obstacle clearance is bounded by shared start/end poses: the corridor is
  ~9 cm long, so the achievable narrow/wide gap is ~5 mm vs ~18–27 mm.
- `akita_black_bowl_1` geoms have a wide spread; post positions are limited
  to a few feasible sites per layout and some layouts may be infeasible.
