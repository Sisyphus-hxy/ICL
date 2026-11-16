"""Offline checks for isolated CUDA profile installation and evaluation routing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import environment
import evaluate


class EnvironmentTests(unittest.TestCase):
    def test_sm120_plan_uses_separate_prefix_and_exact_cuda_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = environment.install_commands('sm120', root)
            prefix = root / 'env/bpp-sm120'
            self.assertEqual(commands[0][commands[0].index('--prefix') + 1], str(prefix))
            self.assertIn('torch==2.8.0+cu128', commands[1])
            self.assertIn('torchvision==0.23.0+cu128', commands[1])
            self.assertIn('https://download.pytorch.org/whl/cu128', commands[1])
            self.assertIn(str(root / 'config/constraints.sm120.txt'), commands[2])
            for command in commands[1:]:
                self.assertEqual(command[0], str(prefix / 'bin/python'))
            self.assertFalse(prefix.exists())

    def test_resume_installs_into_existing_sm120_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'env/bpp-sm120/conda-meta').mkdir(parents=True)
            commands = environment.install_commands('sm120', root)
            self.assertFalse(any(command[0] == 'conda' for command in commands))
            self.assertEqual(len(commands), 3)

    def test_evaluation_profile_switch_preserves_benchmark_and_rollout_options(self):
        for benchmark in ('libero', 'liberogen_spatial_combination'):
            with self.subTest(benchmark=benchmark):
                default = evaluate.command('smoke', 3, Path('/tmp/bpp-profile-check'),
                                           save_video=True, benchmark=benchmark)
                sm120 = evaluate.command('smoke', 3, Path('/tmp/bpp-profile-check'),
                                         save_video=True, benchmark=benchmark, env_profile='sm120')
                self.assertEqual(default[1:], sm120[1:])
                self.assertEqual(Path(default[0]).parent.parent.name, 'bpp')
                self.assertEqual(Path(sm120[0]).parent.parent.name, 'bpp-sm120')

    def test_missing_profile_never_falls_back_to_default_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default_python = root / 'env/bpp/bin/python'
            default_python.parent.mkdir(parents=True)
            default_python.touch()
            with patch.object(evaluate, 'ROOT', root):
                missing = evaluate.missing_inputs('smoke', env_profile='sm120')
            self.assertIn(str(root / 'env/bpp-sm120/bin/python'), missing)
            self.assertNotIn(str(default_python), missing)

    def test_setup_routes_profile_and_preserves_asset_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'python3'
            log = root / 'commands.jsonl'
            executable.write_text('#!/usr/bin/python3\nimport json, os, sys\n'
                                  'with open(os.environ["BPP_TEST_COMMAND_LOG"], "a") as stream:\n'
                                  '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            executable.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       BPP_TEST_COMMAND_LOG=str(log))
            subprocess.run(['bash', str(environment.ROOT / 'setup.sh'), 'env', '--env', 'sm120'],
                           env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(calls[-1], ['scripts/environment.py', '--env', 'sm120'])
            log.write_text('')
            subprocess.run(['bash', str(environment.ROOT / 'setup.sh'), 'data', '--env=sm120',
                            '--benchmark', 'liberogen_spatial_combination', '--local-only'],
                           env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(calls[-1], ['scripts/fetch_assets.py', 'data', '--benchmark',
                                        'liberogen_spatial_combination', '--local-only'])


if __name__ == '__main__':
    unittest.main()
