#!/usr/bin/env bash
# Default Prompt vs Oracle-GT Prompt experiment launcher.
#
# Stages:
#   status    print dataset/checkpoint/init-state availability per benchmark
#   generate  run gtgen.py (GT demos from query init states)
#   eval      run evaluate.py (paired default/oracle rollouts)
#   report    run report.py (SR aggregation)
#
# Usage (GPU values follow the AGENTS.md CUDA_VISIBLE_DEVICES mapping):
#   bash bpp/experiments/oracle_gt/run.sh status --benchmark liberogen_goal_chain
#   bash bpp/experiments/oracle_gt/run.sh generate --benchmark liberogen_goal_chain --tasks put_the_bowl_on_the_plate_and_then_put_the_cream_cheese_in_the_bowl
#   CUDA_VISIBLE_DEVICES=5 bash bpp/experiments/oracle_gt/run.sh eval --benchmark liberogen_goal_chain --tasks <task>
#   bash bpp/experiments/oracle_gt/run.sh report --benchmark liberogen_goal_chain
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BPP_ROOT="$(cd "$HERE/../.." && pwd)"
source "$BPP_ROOT/scripts/common.sh"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM=egl
export PYNPUT_BACKEND=dummy
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# robosuite asserts MUJOCO_EGL_DEVICE_ID is one of the ids listed in
# CUDA_VISIBLE_DEVICES (its EGL device index mirrors the CUDA mask), so the
# only valid choice is the same value — do NOT use the nvidia-smi index here.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
    export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi
PY="$BPP_ROOT/env/bpp/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "missing python env: $PY (try env/bpp-sm120 for Blackwell)" >&2
    exit 1
fi
STAGE="$1"; shift
case "$STAGE" in
    status)
        export ORACLE_GT_DIR="$HERE"
        exec "$PY" - "$@" <<'EOF'
import argparse, json, os, sys
sys.path.insert(0, os.environ['ORACLE_GT_DIR'])
import tasks as T
p = argparse.ArgumentParser(); p.add_argument('--benchmark', default=None)
a = p.parse_args()
for b in ([a.benchmark] if a.benchmark else sorted(T.BENCHMARKS)):
    print(json.dumps(T.data_status(b), indent=1))
EOF
        ;;
    generate) exec "$PY" "$HERE/gtgen.py" "$@" ;;
    eval)     exec "$PY" "$HERE/evaluate.py" --output "$HERE/outputs" "$@" ;;
    report)   exec "$PY" "$HERE/report.py" --output "$HERE/outputs" "$@" ;;
    *) echo "unknown stage: $STAGE (status|generate|eval|report)" >&2; exit 1 ;;
esac
