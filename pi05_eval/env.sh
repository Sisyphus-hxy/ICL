# Shared environment for pi0.5 server + LIBERO eval client.
# Usage: source pi05_eval/env.sh
# Mirrors ContextFlow/run_contextflow_env.sh (caches + proxies on /data2,
# root partition is full) minus ContextFlow-specific knobs.

# Newer lerobot (openpi server env) rejects LEROBOT_HOME; it wants
# HF_LEROBOT_HOME. Older lerobot (ContextFlow env, used by
# build_lerobot_demos.py/verify_demos.py) reads LEROBOT_HOME — set it
# per-invocation there, not here.
export HF_LEROBOT_HOME=/data2/huangxiangyun/.cache/huggingface/lerobot
export HF_HOME=/data2/huangxiangyun/.cache/huggingface
export OPENPI_DATA_HOME=/data2/huangxiangyun/.cache/openpi
export UV_CACHE_DIR=/data2/huangxiangyun/.cache/uv

# HF / GCS / S3 access goes through the local proxy (see `setproxy` alias).
export https_proxy=http://127.0.0.1:50311
export http_proxy=http://127.0.0.1:50311
export all_proxy=socks5://127.0.0.1:50311
export no_proxy=localhost,127.0.0.1,::1

# LIBERO offscreen rendering on NVIDIA GPUs.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# xet transfer is ~170KB/s through the proxy; plain HTTP is much faster.
export HF_HUB_DISABLE_XET=1

# Leave headroom on shared 48GB cards (pi0.5 inference needs ~10GB).
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}

ICL=/data2/huangxiangyun/project/icl
OPENPI=$ICL/openpi
LIBERO_PY=/data2/huangxiangyun/project/xvla/env/libero/bin/python

# Client PYTHONPATH (set it only around the py3.8 sim client via client.sh):
# upstream openpi-client (matches server), ContextFlow's openpi fork only for
# LIBERO_UNSEEN_TASKS, and this directory. Server-side commands must NOT see
# this (ContextFlow/src would shadow the installed upstream openpi).
export PI05_CLIENT_PYTHONPATH="$OPENPI/packages/openpi-client/src:$ICL/ContextFlow/src:$ICL/pi05_eval"
