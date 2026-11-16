#!/usr/bin/env python3
"""Check the selected GPU with real CUDA kernels before loading the BPP model."""
import os

import torch


def main():
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '<default>')
    print(f'CUDA preflight: CUDA_VISIBLE_DEVICES={selected}; '
          f'PyTorch={torch.__version__}; CUDA={torch.version.cuda}', flush=True)
    if not torch.cuda.is_available():
        raise SystemExit(f'CUDA is unavailable for selected GPU {selected}. '
                         'Check the GPU ID/UUID and driver access.')
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Selected GPU: {name} (sm_{major}{minor}); '
          f'PyTorch architectures: {", ".join(torch.cuda.get_arch_list())}', flush=True)
    try:
        # Allocation alone can pass even when no compatible CUDA kernels exist.
        tensor = torch.ones((16, 16), device='cuda:0')
        normalized = tensor * 2 + 1
        product = normalized @ normalized
        torch.cuda.synchronize()
        if product[0, 0].item() != 144:
            raise RuntimeError('Unexpected CUDA arithmetic result')
    except RuntimeError as error:
        raise SystemExit(
            f'CUDA kernel check failed on {name} (sm_{major}{minor}): {error}\n'
            'Select another compatible GPU with --gpu CUDA_INDEX or --gpu GPU-UUID.\n'
            'CUDA indices may differ from nvidia-smi indices; prefer a GPU UUID.\n'
            'List candidates: nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv'
        ) from error
    print('CUDA kernel check passed.', flush=True)


if __name__ == '__main__':
    main()
