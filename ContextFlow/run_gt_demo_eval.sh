#!/bin/bash
# Same-layout GT demo eval on all 4 suites (only init indices with a verified demo).
cd "$(dirname "$0")"
source run_contextflow_env.sh
export PYTHONPATH="$PWD/src:$PWD/third_party/libero:$PWD/packages/openpi-client/src"
PY=/data2/huangxiangyun/project/xvla/env/libero/bin/python
mkdir -p logs logs/eval_results/gt_demo_eval
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "=== $(date +%T) suite: $suite ==="
    $PY examples/libero/main_incontext_gtdemo.py \
        --task-suite-name "$suite" \
        --video-out-path "data/libero_incontext/videos/gt_demo_eval/$suite" \
        --results-out-path "logs/eval_results/gt_demo_eval/$suite.json" \
        > "logs/gt_demo_eval_$suite.log" 2>&1
    echo "=== $(date +%T) $suite done (exit $?) ==="
done
echo "=== all suites finished ==="
