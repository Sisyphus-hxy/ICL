# Shared environment for ContextFlow server + LIBERO eval client.
# Usage: source run_contextflow_env.sh
# All caches and the LeRobot dataset live on /data2 (root partition is full).

export LEROBOT_HOME=/data2/huangxiangyun/.cache/huggingface/lerobot
export HF_HOME=/data2/huangxiangyun/.cache/huggingface
export OPENPI_DATA_HOME=/data2/huangxiangyun/.cache/openpi
export UV_CACHE_DIR=/data2/huangxiangyun/.cache/uv

# HF + Google Drive access goes through the local proxy.
export https_proxy=http://127.0.0.1:50311
export http_proxy=http://127.0.0.1:50311
export all_proxy=socks5://127.0.0.1:50311
export no_proxy=localhost,127.0.0.1,::1

# ContextFlow in-context inference needs float32 (see README troubleshooting).
export JAX_DEFAULT_MATMUL_PRECISION=float32

# LIBERO offscreen rendering on NVIDIA GPUs.
export MUJOCO_GL=egl

# xet transfer is ~170KB/s through the proxy; plain HTTP is much faster.
export HF_HUB_DISABLE_XET=1

# Limit the policy demo dataset to the held-out (unseen) tasks so the server
# can start with only those episodes downloaded. Unset for seen-task eval.
export CONTEXTFLOW_POLICY_EPISODES=remove_list
