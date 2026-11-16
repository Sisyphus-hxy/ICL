"""Shared task inventory for the Default-vs-Oracle-GT prompt experiment.

Each of the 20 unseen LIBERO-Gen tasks provides a BDDL scene file, a
``.pruned_init`` archive of exactly 50 flattened MuJoCo initial states
(query initializations), and an unseen-view demonstration HDF5 used by the
official BPP evaluator for default prompt sampling.
"""
import json
import os
from pathlib import Path

BPP_ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = BPP_ROOT / 'src/behavior_prompting/train_network/env'
SCRIPTS_DIR = BPP_ROOT / 'src/behavior_prompting/train_network/scripts/libero'
CONFIG_DIR = BPP_ROOT / 'config/libero'
OUTPUTS = Path(__file__).resolve().parent / 'outputs'

BENCHMARKS = {
    'liberogen_spatial_combination': {
        'split': 'libero_spatial_selected_combinations_view',
        'checkpoint': 'liberogen_spatial_combination_behavior_prompting.ckpt',
        'base_splits': ['libero_spatial'],
    },
    'liberogen_goal_chain': {
        'split': 'libero_goal_chain_selected_view',
        'checkpoint': 'liberogen_goal_chain_behavior_prompting.ckpt',
        'base_splits': ['libero_goal'],
    },
}

N_QUERIES = 50
# Official runner: n_envs=10, each env i gets prompt demo_i, query g -> demo g%10.
DEFAULT_PROMPT_SLOTS = 10


def _cfg():
    path = os.environ.get('LIBERO_CONFIG_PATH', str(CONFIG_DIR))
    for name in ('config.yaml', 'config.json'):
        candidate = Path(path) / name
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f'no config.yaml/config.json under {path}')


def bddl_root():
    return Path(_cfg()['bddl_files'])


def init_root():
    return Path(_cfg()['init_states'])


def dataset_root():
    return Path(_cfg()['datasets'])


def split_of(benchmark):
    return BENCHMARKS[benchmark]['split']


def task_names(benchmark):
    split = split_of(benchmark)
    return sorted(p.stem for p in bddl_root().joinpath(split).glob('*.bddl'))


def bddl_path(benchmark, task):
    return bddl_root() / split_of(benchmark) / f'{task}.bddl'


def init_states_path(benchmark, task):
    return init_root() / split_of(benchmark) / f'{task}.pruned_init'


def unseen_hdf5(benchmark, task):
    return dataset_root() / split_of(benchmark) / f'{task}_demo.hdf5'


def checkpoint_path(benchmark):
    return BPP_ROOT / 'checkpoints' / BENCHMARKS[benchmark]['checkpoint']


def load_init_states(benchmark, task):
    import torch
    states = torch.load(str(init_states_path(benchmark, task)), weights_only=False)
    assert len(states) == N_QUERIES, (init_states_path(benchmark, task), len(states))
    return states


def shuffle_donor(benchmark, task, query, output=OUTPUTS):
    """Wrong-task prompt ablation donor.

    Returns a *different* task in the same benchmark whose generated GT demo
    exists for this query index. Tasks are scanned in sorted order starting at
    ``task``'s successor, so donors rotate deterministically across tasks and
    the donor's arm actions are guaranteed to be task-irrelevant.
    """
    names = task_names(benchmark)
    idx = names.index(task)
    for step in range(1, len(names)):
        donor = names[(idx + step) % len(names)]
        demo = (output / benchmark / 'experts' / donor
                / f'query_{query:02d}' / 'demo.hdf5')
        if demo.exists():
            return donor
    return None


def required_reference_hdf5s(benchmark):
    """Reference task HDF5 files needed for grasp-pose extraction / replay."""
    return [dataset_root() / s for s in BENCHMARKS[benchmark]['base_splits']]


def data_status(benchmark):
    """Which inputs exist locally; used by run.sh status + README."""
    split = split_of(benchmark)
    tasks = task_names(benchmark)
    status = {'benchmark': benchmark, 'split': split, 'n_tasks': len(tasks),
              'checkpoint': checkpoint_path(benchmark).exists(), 'tasks': {}}
    for task in tasks:
        hdf5 = unseen_hdf5(benchmark, task)
        inits = init_states_path(benchmark, task)
        status['tasks'][task] = {
            'bddl': bddl_path(benchmark, task).exists(),
            'init_states': inits.exists(),
            'unseen_hdf5': hdf5.exists() and not str(hdf5).endswith('.part'),
            'unseen_hdf5_bytes': hdf5.stat().st_size if hdf5.exists() else 0}
    status['reference_hdf5'] = {
        str(d.name): (sorted(p.name for p in d.glob('*.hdf5')) if d.exists() else 'MISSING_DIR')
        for d in required_reference_hdf5s(benchmark)}
    return status
