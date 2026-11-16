"""Offline regression checks for benchmark selection and artifact reuse."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmarks
import evaluate
import fetch_assets
from check_checkpoint import inspect_checkpoint

SPATIAL = 'liberogen_spatial_combination'


class ArtifactReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b'official fixture artifact'
        self.sha = hashlib.sha256(self.content).hexdigest()

    def prepare(self, target, roots=()):
        with patch.object(fetch_assets.subprocess, 'run', side_effect=AssertionError('Network called')):
            return fetch_assets.download('test/repo', 'revision', 'demonstration_data/split/task_demo.hdf5',
                                         target, len(self.content), self.sha, True, roots, local_only=True)

    def test_existing_verified_file_never_downloads(self):
        target = self.root / 'task_demo.hdf5'
        target.write_bytes(self.content)
        self.assertTrue(self.prepare(target))
        self.assertEqual(target.read_bytes(), self.content)

    def test_existing_dataset_is_linked_without_copy_or_network(self):
        source = self.root / 'existing/split/task_demo.hdf5'
        source.parent.mkdir(parents=True)
        source.write_bytes(self.content)
        target = self.root / 'managed/split/task_demo.hdf5'
        self.assertTrue(self.prepare(target, [self.root / 'existing']))
        self.assertTrue(target.is_symlink())
        self.assertEqual(target.resolve(), source)

    def test_conflicting_existing_file_is_preserved(self):
        target = self.root / 'task_demo.hdf5'
        target.write_bytes(b'keep this local file')
        with self.assertRaisesRegex(RuntimeError, 'preserved'):
            self.prepare(target)
        self.assertEqual(target.read_bytes(), b'keep this local file')

    def test_missing_local_file_never_downloads(self):
        target = self.root / 'missing/task_demo.hdf5'
        self.assertFalse(self.prepare(target))
        self.assertFalse(target.exists())

    def test_verified_partial_is_reused_without_network(self):
        target = self.root / 'task_demo.hdf5'
        target.with_name(target.name + '.part').write_bytes(self.content)
        self.assertTrue(self.prepare(target))
        self.assertEqual(target.read_bytes(), self.content)

    def test_legacy_checkpoint_manifest_works_offline(self):
        spec = benchmarks.BENCHMARKS['libero']
        previous = {'repo': spec['checkpoint_repo'], 'revision': 'pinned', 'files': [
            {'path': 'checkpoints/' + spec['checkpoint'], 'bytes': 24, 'sha256': self.sha}]}
        with patch.object(fetch_assets.subprocess, 'run', side_effect=AssertionError('Network called')):
            result = fetch_assets.get_metadata(spec, 'checkpoint', previous, False, True)
        self.assertEqual(result['sha'], 'pinned')
        self.assertEqual(result['siblings'][0]['rfilename'], spec['checkpoint'])

    def test_partial_catalog_cannot_be_used_for_full_offline_preparation(self):
        spec = benchmarks.BENCHMARKS['libero']
        previous = {'repo': spec['dataset_repo'], 'revision': 'pinned', 'catalog_full': False,
                    'artifacts': [{'rfilename': 'libero_spatial/task_demo.hdf5'}]}
        with self.assertRaisesRegex(RuntimeError, 'No cached official'):
            fetch_assets.get_metadata(spec, 'data', previous, True, True)

    def test_spatial_full_preparation_strips_remote_prefix_and_reuses_files(self):
        spec = benchmarks.BENCHMARKS[SPATIAL]
        artifacts = []
        for split in spec['splits']:
            name = split + '/task_demo.hdf5'
            source = self.root / 'existing' / name
            source.parent.mkdir(parents=True)
            source.write_bytes(self.content)
            artifacts.append({'rfilename': spec['dataset_prefix'] + name,
                              'size': len(self.content), 'lfs': {'sha256': self.sha}})
        manifest = self.root / 'catalog.json'
        manifest.write_text(json.dumps({'repo': spec['dataset_repo'], 'revision': 'pinned',
                                       'catalog_full': True, 'artifacts': artifacts}))
        args = ['fetch_assets.py', 'data', '--benchmark', SPATIAL, '--full',
                '--reuse-from', str(self.root / 'existing'), '--local-only']
        with patch.object(fetch_assets, 'ROOT', self.root), \
                patch.object(fetch_assets, 'manifest_path', return_value=manifest), \
                patch.object(sys, 'argv', args), \
                patch.object(fetch_assets.subprocess, 'run', side_effect=AssertionError('Network called')):
            fetch_assets.main()
            # Repeating setup must reuse the same files and pinned catalog offline.
            fetch_assets.main()
        result = json.loads(manifest.read_text())
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['files']), len(spec['splits']))
        for split in spec['splits']:
            self.assertTrue((self.root / 'datasets/libero' / split / 'task_demo.hdf5').is_symlink())
        self.assertFalse((self.root / 'datasets/libero/demonstration_data').exists())


class EvaluationTests(unittest.TestCase):
    def test_hydra_composes_standard_and_spatial_smoke_full_and_video(self):
        from hydra import compose, initialize_config_dir
        config_dir = benchmarks.ROOT / 'src/behavior_prompting/train_network/config'
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            for benchmark in ['libero', SPATIAL]:
                for mode in ['smoke', 'full']:
                    for video in [False, True]:
                        with self.subTest(benchmark=benchmark, mode=mode, video=video):
                            cmd = evaluate.command(mode, 0, Path('/tmp/bpp-hydra-check'), video,
                                                   benchmark=benchmark)
                            cfg = compose(config_name='libero_policy_dunetp', overrides=cmd[3:])
                            self.assertEqual(cfg.task.name, benchmarks.BENCHMARKS[benchmark]['task'])
                            self.assertEqual(cfg._target_, 'behavior_prompting.train_network.workspace.rollout_policy_workspace.RolloutPolicyWorkspace')
                            self.assertTrue(cfg.task.shape_meta.use_prompting)
                            if benchmark == SPATIAL:
                                self.assertEqual(list(cfg.task.unseen_dataset_splits), benchmarks.BENCHMARKS[SPATIAL]['unseen_splits'])
                                if mode == 'smoke':
                                    self.assertFalse(cfg.task.env_runner.eval_seen)
                                    self.assertTrue(cfg.task.env_runner.eval_unseen)
                                    self.assertEqual(cfg.task.unseen_rollout_max_tasks, 1)
                                    self.assertEqual(cfg.task.env_runner.unseen_max_n, 1)
                                    self.assertEqual(cfg.task.env_runner.n_train_vis, 0)
                                else:
                                    self.assertEqual(cfg.task.env_runner.max_steps, 550)

    def test_wrong_benchmark_checkpoint_is_rejected_on_cpu(self):
        import torch
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint.ckpt'
            torch.save({'cfg': OmegaConf.create({'task': {'name': 'libero_all',
                        'dataset_splits': ['libero_spatial'], 'unseen_dataset_splits': []}})}, path)
            self.assertEqual(inspect_checkpoint(path, 'libero_all')['task.name'], 'libero_all')
            with self.assertRaisesRegex(ValueError, 'selected benchmark requires'):
                inspect_checkpoint(path, SPATIAL)

    def test_spatial_full_rejects_missing_catalog_tasks(self):
        spec = benchmarks.BENCHMARKS[SPATIAL]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env_root = root / 'src/behavior_prompting/train_network/env/libero'
            base_files = ['src/behavior_prompting/train_network/train.py', 'env/bpp/bin/python',
                          'checkpoints/' + spec['checkpoint'], 'src/deps/LIBERO/libero/libero/__init__.py']
            for name in base_files:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            names = []
            artifacts = []
            for split in spec['splits']:
                names.append(split + '/task_demo.hdf5')
                artifacts.append({'rfilename': spec['dataset_prefix'] + names[-1], 'size': 4})
                data_file = root / 'datasets/libero' / names[-1]
                data_file.parent.mkdir(parents=True)
                data_file.write_bytes(b'data')
                for kind, suffix in [('bddl_files', '.bddl'), ('init_files', '.pruned_init')]:
                    file = env_root / kind / split / ('task' + suffix)
                    file.parent.mkdir(parents=True)
                    file.touch()
            manifest = root / 'catalog.json'
            manifest.write_text(json.dumps({'repo': spec['dataset_repo'], 'catalog_full': True,
                                           'dataset_files': names, 'artifacts': artifacts}))
            with patch.object(evaluate, 'ROOT', root), patch.object(evaluate, 'manifest_path', return_value=manifest):
                self.assertEqual(evaluate.missing_inputs('full', SPATIAL), [])
                missing = root / 'datasets/libero' / names[-1]
                missing.unlink()
                self.assertIn(str(missing), evaluate.missing_inputs('full', SPATIAL))


if __name__ == '__main__':
    unittest.main()
