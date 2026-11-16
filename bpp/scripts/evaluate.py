#!/usr/bin/env python3
"""Run the original BPP LIBERO or LIBERO-Gen RolloutPolicyWorkspace."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from benchmarks import BENCHMARKS, ROOT, SOURCES, data_root as default_data_root, local_dataset_path, manifest_path
from environment import ENVIRONMENTS, environment_prefix


def smoke_task(benchmark, data_root):
    spec = BENCHMARKS[benchmark]
    if benchmark == 'libero':
        return spec['smoke_task']
    files = sorted((data_root / spec['smoke_split']).glob('*.hdf5'))
    return files[0].stem if files else None


def command(mode, seed, output, save_video=False, benchmark='libero', checkpoint=None, data_root=None,
            env_profile='default'):
    spec = BENCHMARKS[benchmark]
    python = environment_prefix(env_profile, ROOT) / 'bin/python'
    checkpoint = checkpoint or ROOT / 'checkpoints' / spec['checkpoint']
    data_root = data_root or default_data_root(spec, ROOT)
    if spec['task'] == 'draw':
        result = [str(python), 'train.py', f'--config-name={spec["config_name"]}',
                  f'+rollout={spec["rollout"]}', f'rollout.checkpoint_path={checkpoint}',
                  'logging.mode=offline', f'training.seed={seed}',
                  f'exp_name=bpp_{benchmark}_{mode}', 'group_tag=local_reproduction',
                  f'hydra.run.dir={output}',
                  f'task.eval_dataset_path={data_root / spec["eval_dataset"]}']
        train_zarr = data_root / spec['train_dataset']
        if mode == 'full' and train_zarr.is_dir():
            result.extend([f'task.dataset.dataset_path={train_zarr}',
                           f'task.dataset.num_training_demos_per_task='
                           f'{spec["num_training_demos_per_task"]}'])
        if mode == 'smoke':
            result.extend(['task.eval_rollout_max_tasks=1', 'task.env_runner.n_train=0',
                           'task.env_runner.n_test=4',
                           f'task.env_runner.n_test_vis={int(save_video)}',
                           'task.env_runner.n_train_vis=0', 'task.env_runner.max_steps=120',
                           'rollout.enable_expensive_vis=false'])
        elif save_video:
            result.extend(['task.env_runner.n_train_vis=1', 'task.env_runner.n_test_vis=1'])
        return result
    cache_dir = ROOT / '.cache/rollout'
    if benchmark != 'libero':
        cache_dir /= benchmark
    result = [str(python), 'train.py', '--config-name=libero_policy_dunetp',
              f'task={spec["task"]}', f'+modifiers={spec["modifier"]}', '+rollout=libero',
              f'rollout.checkpoint_path={checkpoint}', 'logging.mode=offline',
              f'training.seed={seed}', f'exp_name=bpp_{benchmark}_{mode}', 'group_tag=local_reproduction',
              f'hydra.run.dir={output}', f'task.env_runner.cache_dir={cache_dir}']
    if mode == 'smoke':
        result.extend(['task.dataset.name_suffix=bpp_smoke', 'task.env_runner.n_envs=1',
                       'task.env_runner.n_test=1',
                       f'task.env_runner.n_test_vis={int(save_video)}',
                       'task.env_runner.max_steps=24', 'rollout.enable_expensive_vis=false'])
        task = smoke_task(benchmark, data_root)
        if benchmark == 'libero':
            result.extend(['task.dataset_splits=[libero_spatial]',
                           f'task.include_file_filters=[{task}]',
                           'task.train_rollout_max_tasks=1', 'task.env_runner.eval_unseen=false',
                           'task.env_runner.n_train=1', f'task.env_runner.n_train_vis={int(save_video)}'])
        else:
            result.extend(['task.env_runner.eval_seen=false', 'task.env_runner.eval_unseen=true',
                           'task.unseen_rollout_max_tasks=1', 'task.env_runner.unseen_max_n=1',
                           'task.env_runner.n_train=0', 'task.env_runner.n_train_vis=0'])
            if task:
                result.append(f'task.unseen_include_file_filters=[{task}]')
    elif save_video:
        result.extend(['task.env_runner.n_train_vis=1', 'task.env_runner.n_test_vis=1'])
    return result


def missing_inputs(mode, benchmark='libero', checkpoint=None, data_root=None, env_profile='default'):
    spec = BENCHMARKS[benchmark]
    checkpoint = checkpoint or ROOT / 'checkpoints' / spec['checkpoint']
    data = data_root or default_data_root(spec, ROOT)
    files = [ROOT / 'src/behavior_prompting/train_network/train.py',
             environment_prefix(env_profile, ROOT) / 'bin/python', checkpoint,
             ROOT / 'src/deps/LIBERO/libero/libero/__init__.py']
    if spec['task'] == 'draw':
        missing = [str(file) for file in files if not file.exists()]
        eval_zarr = data / spec['eval_dataset']
        if not (eval_zarr.is_dir() and any(eval_zarr.iterdir())):
            missing.append(f'{eval_zarr}: run bash bpp/setup.sh data '
                           '--benchmark drawanything_sim first')
        train_zarr = data / spec['train_dataset']
        if mode == 'full' and not (train_zarr.is_dir() and any(train_zarr.iterdir())):
            print(f'Note: {train_zarr} not found; full mode will run the handmade '
                  'eval tasks only (no seen-task rollouts). Prepare it with '
                  '`bash bpp/setup.sh data-full --benchmark drawanything_sim`.')
        return missing
    env_root = ROOT / 'src/behavior_prompting/train_network/env/libero'
    for split in spec['splits']:
        for kind in ['bddl_files', 'init_files']:
            files.append(env_root / kind / split)
    missing = [str(file) for file in files if not file.exists()]
    selected = []
    if mode == 'smoke':
        split = 'libero_spatial' if benchmark == 'libero' else spec['smoke_split']
        task = smoke_task(benchmark, data)
        if task is None:
            missing.append(f'{data / split}: need at least one official smoke HDF5 task')
        else:
            selected = [data / split / (task + '.hdf5')]
            if not selected[0].is_file() or selected[0].stat().st_size == 0:
                missing.append(str(selected[0]))
    elif benchmark != 'libero':
        catalog_file = manifest_path('data', benchmark)
        catalog = json.loads(catalog_file.read_text()) if catalog_file.is_file() else {}
        expected = catalog.get('dataset_files', [])
        if (catalog.get('repo') != spec['dataset_repo'] or not catalog.get('catalog_full')
                or not expected or any(not any(Path(name).parent.name == split for name in expected)
                                       for split in spec['splits'])):
            missing.append(f'Official full dataset catalog missing: {catalog_file}. '
                           'Prepare data --full or record --metadata-only --full first.')
        else:
            expected_sizes = {str(local_dataset_path(item['rfilename'], spec)):
                              item.get('size') or (item.get('lfs') or {}).get('size')
                              for item in catalog['artifacts']}
            for name in expected:
                file = data / name
                selected.append(file)
                if (not file.is_file() or file.stat().st_size == 0
                        or (expected_sizes.get(name) and file.stat().st_size != expected_sizes[name])):
                    missing.append(str(file))
            for split in spec['splits']:
                actual = {str(path.relative_to(data)) for path in (data / split).glob('*.hdf5')}
                wanted = {name for name in expected if Path(name).parent.name == split}
                if actual - wanted:
                    missing.append(f'{split}: unexpected task files outside the official catalog: {sorted(actual - wanted)}')
    else:
        for split, expected in spec['splits'].items():
            count = sum(p.stat().st_size > 0 for p in (data / split).glob('*.hdf5'))
            if count != expected:
                missing.append(f'{split}: expected {expected} nonempty HDF5 task files, found {count}')
    for file in selected:
        task = file.name.removesuffix('_demo.hdf5')
        for kind, suffix in [('bddl_files', '.bddl'), ('init_files', '.pruned_init')]:
            resource = env_root / kind / file.parent.name / (task + suffix)
            if not resource.is_file():
                missing.append(str(resource))
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=BENCHMARKS, default='libero')
    parser.add_argument('--env', choices=ENVIRONMENTS, default=os.environ.get('BPP_ENV', 'default'),
                        help='default: PyTorch cu118 for A6000; sm120: PyTorch cu128 for Blackwell')
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                        help='One CUDA_VISIBLE_DEVICES index or GPU UUID; indices may differ from nvidia-smi')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-video', action='store_true',
                        help='Save one train and one test rollout video per task under the run media/ directory')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Check local inputs and checkpoint task without GPU or downloads')
    parser.add_argument('--checkpoint', type=Path, help='Reuse a trusted local checkpoint in place; never copied or downloaded')
    parser.add_argument('--data-root', type=Path, help='Existing directory containing benchmark split folders')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if ',' in args.gpu:
        parser.error('This launcher uses one GPU; select one CUDA_VISIBLE_DEVICES index or GPU UUID')
    spec = BENCHMARKS[args.benchmark]
    checkpoint = (args.checkpoint or ROOT / 'checkpoints' / spec['checkpoint']).expanduser().resolve()
    data_root = (args.data_root or default_data_root(spec, ROOT)).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    prefix = args.mode if args.benchmark == 'libero' else f'{args.benchmark}_{args.mode}'
    if args.env != 'default':
        prefix += '_' + args.env
    output = (args.output or ROOT / 'logs' / f'{prefix}_{stamp}').resolve()
    argv = command(args.mode, args.seed, output, save_video=args.save_video,
                   benchmark=args.benchmark, checkpoint=checkpoint, data_root=data_root,
                   env_profile=args.env)
    workdir = ROOT / 'src/behavior_prompting/train_network'
    print(f'cwd: {workdir}\ncommand: {shlex.join(argv)}', flush=True)
    if args.dry_run:
        print('Preview only: no checkpoint loaded, no model or simulator executed.')
        return 0
    missing = missing_inputs(args.mode, args.benchmark, checkpoint, data_root, env_profile=args.env)
    checkpoint_info = None
    if checkpoint.is_file() and Path(argv[0]).is_file():
        checked = subprocess.run([argv[0], str(ROOT / 'scripts/check_checkpoint.py'),
                                  str(checkpoint), '--task', spec['task']],
                                 capture_output=True, text=True)
        if checked.returncode:
            missing.append('Checkpoint validation failed: ' + checked.stderr.strip())
        else:
            checkpoint_info = json.loads(checked.stdout)
            print(f'Checkpoint config: {json.dumps(checkpoint_info)}', flush=True)
    if missing:
        print('Setup incomplete:\n- ' + '\n- '.join(missing), file=sys.stderr)
        return 1
    from fetch_source import apply_source_patches, revision_of
    actual_revision = revision_of(ROOT / 'src')
    if actual_revision != SOURCES['revision']:
        raise RuntimeError('Source revision differs from config/sources.json')
    if args.check_only:
        print('Local inputs and checkpoint task verified; no GPU used and nothing downloaded.')
        return 0
    source_patches = apply_source_patches(ROOT / 'src')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    env.setdefault('MUJOCO_GL', 'egl')
    env['LIBERO_CONFIG_PATH'] = str(ROOT / 'config/libero')
    env['PATH'] = str(Path(argv[0]).parent) + os.pathsep + env['PATH']
    subprocess.run([argv[0], str(ROOT / 'scripts/check_cuda.py')], env=env, check=True)
    output.mkdir(parents=True, exist_ok=False)
    if spec['task'] != 'draw':
        config = json.loads((ROOT / 'config/libero/config.yaml').read_text())
        config['datasets'] = str(data_root)
        config_dir = output / 'libero_config'
        config_dir.mkdir()
        (config_dir / 'config.yaml').write_text(json.dumps(config, indent=2) + '\n')
        env['LIBERO_CONFIG_PATH'] = str(config_dir)
    record = {'mode': args.mode, 'benchmark': args.benchmark, 'official_full_benchmark': args.mode == 'full',
              'status': 'running', 'seed': args.seed, 'gpu': args.gpu, 'command': argv,
              'checkpoint': str(checkpoint), 'checkpoint_config': checkpoint_info,
              'data_root': str(data_root),
              'environment': args.env, 'python_prefix': str(environment_prefix(args.env, ROOT)),
              'cwd': str(workdir), 'source_revision': actual_revision,
              'source_patches': source_patches, 'save_video_requested': args.save_video}
    run_file = output / 'invocation.json'
    run_file.write_text(json.dumps(record, indent=2) + '\n')
    with (output / 'rollout.log').open('w') as log:
        process = subprocess.Popen(argv, cwd=workdir, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    record.update(status='completed' if code == 0 else 'failed', exit_code=code)
    run_file.write_text(json.dumps(record, indent=2) + '\n')
    if code == 0 and args.save_video:
        videos = sorted((output / 'media').glob('*.mp4'))
        print(f'Saved {len(videos)} rollout videos under {output / "media"}', flush=True)
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
