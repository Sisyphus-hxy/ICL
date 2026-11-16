#!/usr/bin/env bash
BPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BPP_ROOT
export PIP_CACHE_DIR="$BPP_ROOT/.cache/pip"
export HF_HOME="$BPP_ROOT/.cache/huggingface"
export TORCH_HOME="$BPP_ROOT/.cache/torch"
export CONDA_PKGS_DIRS="$BPP_ROOT/.cache/conda"
export CONDA_ENVS_PATH="$BPP_ROOT/env"
export XDG_CACHE_HOME="$BPP_ROOT/.cache"
export MPLCONFIGDIR="$BPP_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$BPP_ROOT/.cache/numba"
export TMPDIR="$BPP_ROOT/.cache/tmp"
export LIBERO_CONFIG_PATH="$BPP_ROOT/config/libero"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export WANDB_DIR="$BPP_ROOT/logs"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$CONDA_PKGS_DIRS" "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$WANDB_DIR"
