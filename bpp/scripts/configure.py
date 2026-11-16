#!/usr/bin/env python3
"""Translate the pinned upstream environment and keep all mutable paths inside bpp/."""
import json
from pathlib import Path

from benchmarks import BENCHMARKS
from environment import ENVIRONMENTS, torch_requirements

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'


def main():
    snapshot = (ROOT / 'upstream/environment.yml').read_text()
    conda, pip = snapshot.split('  - pip:\n', 1)
    # Install torch separately: inline --index-url in a requirement line is ambiguous.
    requirements = []
    for line in pip.splitlines():
        if not line.startswith('    - '):
            continue
        requirement = line[6:].split(' #', 1)[0].strip()
        if requirement.startswith(('torch==', 'torchvision==')):
            continue
        if requirement == '-e .':
            requirement = '-e ' + str(SOURCE)
        requirement = requirement.replace('-e deps/', '-e ' + str(SOURCE / 'deps') + '/')
        requirements.append(requirement)
    (ROOT / 'config/environment.conda.yml').write_text(conda)
    (ROOT / 'config/requirements.upstream.txt').write_text('\n'.join(requirements) + '\n')
    for profile in ENVIRONMENTS:
        (ROOT / f'config/constraints.{profile}.txt').write_text('\n'.join(torch_requirements(profile)) + '\n')
    # Reuse asset files, but retain BPP's task definitions and initial states.
    xvla = ROOT.parent.parent / 'xvla'
    assets = xvla / 'external/LIBERO/libero/libero/assets'
    if not assets.is_dir():
        assets = SOURCE / 'deps/LIBERO/libero/libero/assets'
    paths = {
        'assets': str(assets),
        'benchmark_root': str(SOURCE / 'deps/LIBERO/libero/libero'),
        'bddl_files': str(SOURCE / 'behavior_prompting/train_network/env/libero/bddl_files'),
        'init_states': str(SOURCE / 'behavior_prompting/train_network/env/libero/init_files'),
        'datasets': str(ROOT / 'datasets/libero'),
    }
    directory = ROOT / 'config/libero'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.yaml').write_text(json.dumps(paths, indent=2) + '\n')
    # Existing demonstration files can be read through individual links. Never write to XVLA.
    existing = xvla / 'external/libero_datasets/datasets'
    reused = 0
    for split in dict.fromkeys(split for spec in BENCHMARKS.values() for split in spec['splits']):
        target_dir = ROOT / 'datasets/libero' / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in sorted((existing / split).glob('*.hdf5')):
            target = target_dir / file.name
            if not target.exists() and not target.is_symlink():
                target.symlink_to(file)
                reused += 1
    print(f'Wrote local environment/config; reused {reused} existing HDF5 files. Assets: {assets}')


if __name__ == '__main__':
    main()
