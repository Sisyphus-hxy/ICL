#!/bin/bash
# Start a pi0.5 policy server.
# Usage: bash pi05_eval/serve.sh <CUDA_VISIBLE_DEVICES> [port]
#   e.g.  bash pi05_eval/serve.sh 5 8000   # physical GPU 0 (RTX A6000)
cd "$(dirname "$0")/.."
source pi05_eval/env.sh

GPU="${1:?CUDA_VISIBLE_DEVICES value required (5/6=A6000, 1-4=6000D, 0=Blackwell)}"
PORT="${2:-8000}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$OPENPI"
exec uv run scripts/serve_policy.py --env LIBERO --port "$PORT"
