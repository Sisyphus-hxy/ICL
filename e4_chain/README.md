# E4-Chain · Latent Residual Behavior Transfer

E4 learns an offline action representation on **BPP-LIBERO-Gen Chain**. Stage A teaches a latent to reconstruct expert motion. Stage B learns to predict that latent from 64 frames of donor RGB. Stage C changes the donor latent using the two scenes' initial object poses. The action decoder accepts one tensor, `Z_demo + delta`. The decoder and the RGB encoder have their own E4 weights.

The [pre-training audit](REPORT.md) records task coverage and measured reconstruction and transfer checks.

## Data

The pinned [official HDF5 release](https://huggingface.co/datasets/austinpatel/libero_gen_goal_chain_hdf5/tree/3f5d3ae4d8c9b9270afe0b73974f67e80fa4401d/demonstration_data) contains 21 first-step, 146 second-step and 144 full-chain training archives. The 10 selected-view tasks are held out. Counts here refer to official task IDs, including numbered second-step variants. The local `catalog.json` records the size and LFS SHA-256 of each published archive. The loader also checks the upstream Chain training YAML and BDDL metadata.

The archive command downloads training files, resumes partial transfers and verifies SHA-256. It recognizes byte-identical stock archives. From the repository root on this server, use the shell's setcy proxy for downloads.

```bash
bash -ic 'setcy && python -u -m e4_chain.archives'
bash e4_chain/run.sh prepare --follow
bash e4_chain/run.sh cache --follow
```

Preparation and cache extraction can run in separate terminals during downloads. Source archives stay under `bpp/datasets/libero/`. Derived data and weights live under `e4_chain/outputs/`.

The episode HDF5 files link to the official training views using paths relative to each episode file. When copying to a different project root, keep the repository-relative directory layout and dereference filesystem symlinks in the raw views (`rsync -aLR`). Stages A/B/C can run directly from the prepared tasks, indexes and feature bank.

To extract features from RGB or run the checkpoint reload check in a new checkout, install the frozen encoder source at `vjepa_probe/third_party/vjepa2` using commit `204698b45b3712590f06245fbfba32d3be539812` of [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2). Place the [official ViT-B checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt) in `vjepa_probe/checkpoints/`. The cache command verifies its SHA-256 before encoding.

* Every successful demonstration with agent-view RGB and OSC actions has a row in `episodes.json`. The per-task HDF5 keeps native actions, EEF, finger state, object pose tracks and an external link to the original demo. The task index retains ordered execution steps, E0 motion segments, gripper windows and observation/action lag. Segment boundaries come from recorded motion and contact proxies.
* Pairs are ordered and confined to one official task, procedure and fold. Generated procedures use the generator's ordered steps. Human procedures additionally match observed anchor order and closure count. A different layout requires a maximum named-body translation above 0.1 mm or a rotation above 0.001 rad. Repeated layouts and actions stay together. About 10% of each task's layouts form internal validation where sufficient layouts exist.
* `task_summary.csv` and `data_summary.json` distinguish the full within-task pair population from the training and validation subsets. Each Stage C epoch visits each eligible training donor once and samples one of its eligible query layouts. The validation partner draw is repeatable.
* JEPA uses the frozen V-JEPA 2.1 ViT-B/16 EMA encoder, 64 canonical agent-view frames at 384 px and BF16. Matching E2 float16 patch arrays are validated against their sidecars and source identity. New features use the same E2 extraction. A further fixed 2×2 spatial pooling produces a compact 32×4×768 bank. Per-task `features.json` records the reused and newly encoded episode counts.

Audit the prepared data and features before starting training. These commands write `data_audit.json` and `cache_audit.json` with measured coverage, pair and parameter counts.

```bash
bpp/env/bpp-sm120/bin/python -m e4_chain.audit
bpp/env/bpp-sm120/bin/python -m e4_chain.audit --cache
```

## Training and checks

```bash
bash e4_chain/run.sh A
bash e4_chain/run.sh B
bash e4_chain/run.sh C
OMP_NUM_THREADS=2 bpp/env/bpp-sm120/bin/python -m e4_chain.check
```

The launcher selects 2–4 idle RTX 6000D GPUs and executes a BF16 matrix multiply on each. For an explicit allocation set `E4_GPU_UUIDS` to space-separated physical UUIDs. Batch size is per GPU. The three stages use BF16 autocast, NCCL DDP and FP32 AdamW weights.

| Stage | Updated weights | Objective |
| --- | --- | --- |
| A | Action Encoder, Decoder | Reconstruct the 192-step, 7D OSC action sequence. Auxiliary outputs reconstruct relative EEF pose and two measured finger positions. |
| B | RGB Behavior Encoder | Match the frozen action teacher's tokens and decode the donor GT actions and auxiliary trajectory. |
| C | Two-layer residual Transformer | Match the frozen query action teacher and decode query GT actions and auxiliary trajectory. Squared residual magnitude has weight 0.02. |

Each latent has 24 ordered tokens of width 32. The action encoder embeds eight-step blocks and applies three temporal Transformer layers. The behavior encoder applies four temporal layers to the frozen features. The decoder uses four temporal layers of width 192 followed by an eight-step output head. The residual Transformer reads donor tokens and corresponding named initial geometry, with a zero-initialized output projection. Query actions enter only the teacher and training loss. Train-fold statistics normalize continuous signals.

`outputs/training/{A,B,C}/` contains atomically written `best.pt`, `last.pt` and `metrics.json`. Each successor loads its predecessor's best validation checkpoint. Use `bash e4_chain/run.sh B --resume` to continue Stage B at the next epoch or pass `--run PATH` consistently to train a separate run. Resume validates the data and pair indexes, geometry slots, feature recipe, parent checkpoint hash, world size and optimizer settings. Epoch-derived shuffles and partner draws are reproducible.

The checks include validation action error (also the latter half of the trajectory), gripper sign accuracy, EEF error, latent alignment, residual magnitude, gradients and frozen parameter hashes. The Stage C metrics also record the uncorrected donor error on the same pairs. The reload check decodes one held-out double-step pair, verifies frozen weights against the parent checkpoint and feeds the predicted pose plan into the original E0 tracker for one control calculation.

For offline inference, load `C/best.pt` into `Chain(len(ck["lineage"]["geometry"]))`, set `eval()`, and use `scene_geometry` to put both reset layouts in the checkpoint's body order. With float32 tensors, the complete action path is

```python
with torch.no_grad():
    prediction, donor_latent, residual = model.predict(donor_features, donor_geometry, query_geometry)
    trajectory = prediction * ck["std"] + ck["mean"]
    query_actions = osc(trajectory)  # (batch, 192, 7), native OSC units on a uniform time grid
```
