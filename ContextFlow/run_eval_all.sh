#!/bin/bash
# Full LIBERO unseen evaluation: all 4 suites, 2 unseen tasks each, 50 trials/task.
# Logs to logs/eval_full_<suite>.log; results in logs/eval_results/unseen/<suite>.json
cd "$(dirname "$0")"
mkdir -p logs
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "=== $(date +%T) suite: $suite ==="
    bash run_eval.sh "$suite" "${1:-50}" > "logs/eval_full_${suite}.log" 2>&1
    echo "=== $(date +%T) $suite done (exit $?) ==="
done
echo "=== all suites finished ==="
