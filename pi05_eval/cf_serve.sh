#!/bin/bash
# Start the ContextFlow in-context policy server with a selectable demo dataset.
# Usage: bash pi05_eval/cf_serve.sh <CVD gpu> [port] [demo_repo_id]
#   e.g.  bash pi05_eval/cf_serve.sh 5 8010 pi05/libero_demos
#         bash pi05_eval/cf_serve.sh 5 8010 physical-intelligence/libero   # GT baseline
# NOTE: <CVD gpu> is a CUDA_VISIBLE_DEVICES value (AGENTS.md mapping).
cd "$(dirname "$0")/../ContextFlow"
source run_contextflow_env.sh

GPU="${1:?CUDA_VISIBLE_DEVICES value required}"
PORT="${2:-8010}"
export CONTEXTFLOW_DEMO_REPO_ID="${3:-pi05/libero_demos}"

export CUDA_VISIBLE_DEVICES="$GPU"
exec uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config=ContextFlow \
    --policy.dir=checkpoints/ContextFlow/ContextFlow_run1/19999 \
    --policy.inference_dtype=float32
