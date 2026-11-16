#!/bin/bash
# Terminal 2 — LIBERO unseen eval client (xvla/env/libero, Python 3.8).
# Usage: bash run_eval.sh [suite] [trials] [unseen_task_index]
#   suite default libero_spatial; use --unseen-task-index N for a single-task smoke test.
cd "$(dirname "$0")"
source run_contextflow_env.sh

SUITE="${1:-libero_spatial}"
TRIALS="${2:-50}"
IDX="${3:--1}"

LIBERO_PY=/data2/huangxiangyun/project/xvla/env/libero/bin/python
export PYTHONPATH="$PWD/src:$PWD/third_party/libero:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=()
SUFFIX=""
if [ "$IDX" -ge 0 ]; then
    EXTRA+=(--unseen-task-index "$IDX")
    SUFFIX="_idx$IDX"
fi

exec "$LIBERO_PY" examples/libero/main_incontext.py \
    --host 127.0.0.1 --port 8000 \
    --task-suite-name "$SUITE" \
    --num-trials-per-task "$TRIALS" \
    "${EXTRA[@]}" \
    --results-out-path "logs/eval_results/unseen/${SUITE}${SUFFIX}.json" \
    --video-out-path "data/libero_incontext/videos/unseen/${SUITE}${SUFFIX}"
