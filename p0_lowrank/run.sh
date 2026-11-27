#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${P0_PYTHON:-$ROOT/bpp/env/bpp-sm120/bin/python}"

# Physical GPU 0 (NVIDIA RTX A6000, 48 GB) by UUID per AGENTS.md
GPU0_UUID="GPU-e0914423-1b44-589b-fdb3-0360e8f79b43"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU0_UUID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "$ROOT"
cmd="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  cache-unseen)
    exec "$PYTHON" -u -m p0_lowrank.cache_unseen "$@"
    ;;
  diagnose)
    exec "$PYTHON" -u -m p0_lowrank.run_diagnosis "$@"
    ;;
  plot)
    exec "$PYTHON" -u -m p0_lowrank.plot_and_report "$@"
    ;;
  all)
    "$PYTHON" -u -m p0_lowrank.cache_unseen
    "$PYTHON" -u -m p0_lowrank.run_diagnosis "$@"
    "$PYTHON" -u -m p0_lowrank.plot_and_report
    ;;
  *)
    echo "Usage: bash p0_lowrank/run.sh {cache-unseen|diagnose|plot|all} [args...]" >&2
    exit 2
    ;;
esac
