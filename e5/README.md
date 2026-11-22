# E5 · Local Reference + Latent Residual Learning

Independent implementation built on official BPP components (`bpp/src/behavior_prompting/`) without modifying `bpp/` or `e4_chain/`.

## Architecture

| Module | Implementation | Params (Total) | Trainable | Initialization |
| --- | --- | ---: | ---: | --- |
| Visual Encoder | BPP CLIP ViT-B/16 (`agentview_rgb`, `eye_in_hand_rgb`, `share_rgb_model=False`) | 171,598,848 | 0 (Frozen) | Strictly loaded from `liberogen_goal_chain_behavior_prompting.ckpt` |
| RGB-only Prompt Pooling | BPP `AttentionPoolLatent(768, 8 heads, mlp_ratio=4.0)` | 7,085,568 | 7,085,568 | Fresh (retrained for RGB-only) |
| Reference Reader | BPP 6-layer `TransformerDecoderWithAttn(768, 8 heads, FFN 3072)` + pos emb + 4 attention sinks | 56,755,200 | 56,755,200 | Compatible BPP checkpoint weights loaded & finetuned |
| Residual Transformer | 3-layer `TransformerDecoder(768, 8 heads, FFN 3072)` + input/output `LayerNorm` + scale-matched zero-init `Linear(768, 768)` | 28,956,672 | 28,956,672 | Fresh, output projection zero-initialized ($\Delta z \equiv 0$ at step 0) |
| Action Decoder | BPP `ConditionalUnet1D(input=10, global_cond=6144, down_dims=[256,512,1024], kernel=5)` | 151,075,722 | 151,075,722 | Fresh (RGB-only `global_cond=8*768=6144` vs BPP 20-token 15360) |
| **E5 Total** | | **415,472,010** | **243,873,162** | |
| **BPP RGB-only Baseline** | Identical without Residual Transformer (`use_residual=False`) | **386,515,338** | **214,916,490** | Unified initialization & training budget |
| **BPP Full Reproduction** | Official BPP (`RGB + 11D proprio + 20-step prompt actions`, 283.2M U-Net) | **518,845,842** | **347,246,994** | Same frozen CLIP cache, 80-epoch manifest & budget |

## Data & Feature Cache

- **Demo sampling**: Aligned with BPP's `PromptActionChunker` (`prompt_chunk_n_actions=20`, `pad_end_prompt_actions=zeros`): 20Hz data sampled every 20 steps (~1Hz, `0, 20, 40, ...`), full demonstration length (up to 29 steps, no 64-frame truncation), dual-camera 224×224 RGB (`CenterCrop(212) -> Resize(224) -> CLIP norm`).
- **Query sampling**: Reuses the existing 80-epoch unique query starts manifest (`seed42/277e1c5c8129e89c`, 14,080 windows/epoch × 80 epochs = 1,126,400 unique starts; 6,180 fixed val windows) with deduplicated `(episode, frame)` visual feature caching for `[max(0, t-1), t]`.
- **Cache layout** (`e5/outputs/cache/`): `demo_features.npy` (~0.58 GB), `query_features.npy` (~5.61 GB), `actions_10d.npy` (~0.15 GB), temporal indices/offsets, train-fold BPP action normalizer, and online/offline consistency audit.

## Slurm Workflow

Submit from repository root:

```bash
mkdir -p e5/outputs/logs
sbatch e5/test.sbatch                                                    # Unit tests (1 GPU)
sbatch e5/cache.sbatch                                                   # Full pre-encoding & audit (2 GPUs)
sbatch e5/overfit.sbatch                                                 # 64-sample overfit check (1 GPU)
sbatch --job-name=e5-joint e5/train.sbatch --mode e5                     # E5 80-epoch Joint Training (2 GPUs)
sbatch --job-name=e5-bpp-rgb e5/train.sbatch --mode bpp_rgb_only         # Matched BPP RGB-only baseline (2 GPUs)
env/train/e4_chain/bin/python -u -m e5.cache_proprio                     # Build 11D proprio cache (~160 MB, CPU)
sbatch e5/bpp_full.sbatch                                                # Full BPP reproduction (RGB+proprio+action20, 2 GPUs)
sbatch e5/eval_chain.sbatch e5 e5/outputs/training/e5/best.pt e5/outputs/evaluation/e5/liberogen_goal_chain
sbatch e5/eval_chain.sbatch bpp_rgb_only e5/outputs/training/bpp_rgb_only/best.pt e5/outputs/evaluation/bpp_rgb_only/liberogen_goal_chain
sbatch e5/eval_chain.sbatch bpp_full e5/outputs/training/bpp_full/best.pt e5/outputs/evaluation/bpp_full/liberogen_goal_chain
```
