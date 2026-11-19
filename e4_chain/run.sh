#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gpu=bpp/env/bpp-sm120/bin/python
stage=${1:?use prepare, cache, A, B or C}
shift
if [[ $stage == prepare ]]; then
    exec "$gpu" -u -m e4_chain.prepare "$@"
fi
case "$stage" in cache|A|B|C) ;; *) exit 2 ;; esac
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
if [[ -n ${E4_GPU_UUIDS:-} ]]; then
    read -ra devices <<< "$E4_GPU_UUIDS"
else
    mapfile -t devices < <(nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader,nounits |
        awk -F ', ' '$2=="NVIDIA RTX 6000D" && $3<1024 {print $1}' | head -4)
fi
if (( ${#devices[@]} < 2 || ${#devices[@]} > 4 )); then
    echo "Select 2–4 available 6000D UUIDs with E4_GPU_UUIDS." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${devices[*]}")
"$gpu" -c 'import torch,os
ids=os.environ["CUDA_VISIBLE_DEVICES"].split(",")
assert len(ids)==len(set(ids))==torch.cuda.device_count()
for i in range(len(ids)):
    assert "6000D" in torch.cuda.get_device_name(i)
    x=torch.ones(16,16,device=f"cuda:{i}",dtype=torch.bfloat16)
    assert (x@x).mean().item()==16'
if [[ $stage == cache ]]; then
    exec "$gpu" -m torch.distributed.run --standalone --nproc-per-node="${#devices[@]}" -m -- e4_chain.cache "$@"
fi
exec "$gpu" -m torch.distributed.run --standalone --nproc-per-node="${#devices[@]}" -m -- e4_chain.train --stage "$stage" "$@"
