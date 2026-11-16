#!/usr/bin/env python3
"""Inspect a trusted BPP checkpoint on CPU and reject benchmark mismatches."""
import argparse
import json
import sys


def inspect_checkpoint(path, expected_task):
    import dill
    import torch
    from omegaconf import OmegaConf

    payload = torch.load(str(path), map_location='cpu', mmap=True, pickle_module=dill,
                         weights_only=False)
    cfg = payload['cfg']
    info = {key: OmegaConf.to_container(value) if OmegaConf.is_config(value) else value
            for key in ['task.name', 'task.dataset_splits', 'task.unseen_dataset_splits']
            for value in [OmegaConf.select(cfg, key)]}
    if info['task.name'] != expected_task:
        raise ValueError(f'Checkpoint task is {info["task.name"]!r}; '
                         f'selected benchmark requires {expected_task!r}. '
                         'Select the matching checkpoint with --checkpoint PATH.')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.path, args.task)))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
