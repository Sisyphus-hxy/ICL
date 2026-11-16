#!/bin/bash
# Wrong-task demo ablation: same 191 covered inits, but the injected demo
# episode comes from a task in a DIFFERENT suite (arm actions task-irrelevant).
cd "$(dirname "$0")"
source run_contextflow_env.sh
export PYTHONPATH="$PWD/src:$PWD/third_party/libero:$PWD/packages/openpi-client/src"
PY=/data2/huangxiangyun/project/xvla/env/libero/bin/python
MAP=logs/eval_results/gt_replay/init_to_episode_wrongtask.json
mkdir -p logs logs/eval_results/gt_demo_wrongtask
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "=== $(date +%T) suite: $suite ==="
    $PY examples/libero/main_incontext_gtdemo.py \
        --task-suite-name "$suite" \
        --init-demo-map "$MAP" \
        --video-out-path "data/libero_incontext/videos/gt_demo_wrongtask/$suite" \
        --results-out-path "logs/eval_results/gt_demo_wrongtask/$suite.json" \
        > "logs/gt_demo_wrongtask_$suite.log" 2>&1
    echo "=== $(date +%T) $suite done (exit $?) ==="
done
echo "=== all suites finished ==="
