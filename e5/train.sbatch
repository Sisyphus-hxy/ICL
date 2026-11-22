#!/usr/bin/env bash
#SBATCH --job-name=e5-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=e5/outputs/logs/train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source env/train/common.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -z "${NCCL_RAS_ADDR:-}" ]]; then
    port=$("$E4_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    export NCCL_RAS_ADDR="127.0.0.1:$port"
fi
ranks=$("$E4_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
echo "job=$SLURM_JOB_ID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ranks=$ranks args=$*"
exec "$E4_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$ranks" \
    -m -- e5.train --resume "$@"
