#!/usr/bin/env bash
#SBATCH --job-name=e5-cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=e5/outputs/logs/cache-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source env/train/common.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job=$SLURM_JOB_ID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec "$E4_PYTHON" -u -m e5.cache --precision fp16 --batch-size 512 --verify-episodes 16 "$@"
