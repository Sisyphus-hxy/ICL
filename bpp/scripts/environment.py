#!/usr/bin/env python3
"""Install and select isolated BPP CUDA environments."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    'default': {'directory': 'bpp', 'torch': '2.7.1+cu118',
                'torchvision': '0.22.1+cu118', 'cuda': '11.8', 'wheel_index': 'cu118'},
    'sm120': {'directory': 'bpp-sm120', 'torch': '2.8.0+cu128',
              'torchvision': '0.23.0+cu128', 'cuda': '12.8', 'wheel_index': 'cu128'},
}


def environment_prefix(profile, root=ROOT):
    return root / 'env' / ENVIRONMENTS[profile]['directory']


def torch_requirements(profile):
    spec = ENVIRONMENTS[profile]
    return [f'{package}=={spec[package]}' for package in ('torch', 'torchvision')]


def install_commands(profile, root=ROOT):
    prefix = environment_prefix(profile, root)
    python = str(prefix / 'bin/python')
    spec = ENVIRONMENTS[profile]
    commands = []
    if not (prefix / 'conda-meta').is_dir():
        commands.append(['conda', 'env', 'create', '--prefix', str(prefix),
                         '--file', str(root / 'config/environment.conda.yml')])
    commands.extend([
        [python, '-m', 'pip', 'install', *torch_requirements(profile),
         '--index-url', f'https://download.pytorch.org/whl/{spec["wheel_index"]}'],
        [python, '-m', 'pip', 'install', '-r', str(root / 'config/requirements.upstream.txt'),
         '--constraint', str(root / f'config/constraints.{profile}.txt')],
        [python, '-m', 'pip', 'check'],
    ])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=ENVIRONMENTS, default='default')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    prefix = environment_prefix(args.env)
    if prefix.is_symlink() or (prefix.exists() and not (prefix / 'conda-meta').is_dir()):
        raise RuntimeError(f'Refusing to install into an existing non-conda directory or symlink: {prefix}')
    print(f'BPP environment: {args.env} -> {prefix}', flush=True)
    commands = install_commands(args.env)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    if not all((ROOT / path).is_file() for path in ('src/setup.py', 'src/deps/LIBERO/setup.py')):
        raise RuntimeError('Source/submodules missing. Run bash bpp/setup.sh source first.')
    env = dict(os.environ, PATH=str(prefix / 'bin') + os.pathsep + os.environ['PATH'])
    for command in commands:
        subprocess.run(command, env=env, check=True)
    python = str(prefix / 'bin/python')
    inspect = ('import json, torch, torchvision; '
               'print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__, '
               '"cuda": torch.version.cuda, "cuda_architectures": torch.cuda.get_arch_list()}))')
    versions = json.loads(subprocess.check_output([python, '-c', inspect], env=env, text=True))
    spec = ENVIRONMENTS[args.env]
    if any(versions[key] != spec[key] for key in ('torch', 'torchvision', 'cuda')):
        raise RuntimeError(f'Installed PyTorch build differs from environment profile: {versions}')
    if args.env == 'sm120' and versions['cuda_architectures'] and 'sm_120' not in versions['cuda_architectures']:
        raise RuntimeError(f'Installed PyTorch build does not list sm_120: {versions}')
    log_dir = ROOT / 'logs' if args.env == 'default' else ROOT / 'logs/env-sm120'
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, command in (
        ('requirements-installed.txt', [python, '-m', 'pip', 'freeze']),
        ('conda-explicit.txt', ['conda', 'list', '--prefix', str(prefix), '--explicit']),
    ):
        (log_dir / name).write_text(subprocess.check_output(command, env=env, text=True))
    (log_dir / 'environment.json').write_text(json.dumps({
        'profile': args.env, 'prefix': str(prefix), 'versions': versions,
        'installed_at': datetime.now(timezone.utc).isoformat(),
    }, indent=2) + '\n')
    print(f'Environment installed and package checks passed: {prefix}', flush=True)
    print('Run run_eval.sh with the matching --env and --gpu to verify CUDA kernels.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Environment setup failed: {error}', file=sys.stderr)
        sys.exit(1)
