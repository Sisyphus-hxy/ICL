#!/bin/bash
# Terminal 1 — ContextFlow policy server (JAX, Python 3.11 venv).
# GPU: RTX 6000D is Blackwell sm_120 — jax[cuda12]==0.5.0 ptxas (12.4) can't
# compile for it. Use RTX A6000 (sm_86): CUDA_VISIBLE_DEVICES=5 or 6.
# Do NOT pick by nvidia-smi index — see icl/AGENTS.md GPU mapping table.
cd "$(dirname "$0")"
source run_contextflow_env.sh

GPU="${1:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
exec uv run scripts/serve_policy.py policy:checkpoint \
    --policy.inference_dtype=float32 \
    --policy.config=ContextFlow \
    --policy.dir=checkpoints/ContextFlow/ContextFlow_run1/19999
