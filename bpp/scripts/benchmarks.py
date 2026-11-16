"""Benchmark-specific artifacts and paths shared by setup and evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / 'config/sources.json').read_text())
BENCHMARKS = {
    'libero': dict(SOURCES, task='libero_all', modifier='libero/libero_all',
                   dataset_prefix='', unseen_splits=[]),
    **SOURCES.get('benchmarks', {}),
}


def manifest_path(kind, benchmark):
    suffix = '' if benchmark == 'libero' else '-' + benchmark
    return ROOT / 'logs' / f'{kind}{suffix}-manifest.json'


def data_root(spec, root=ROOT):
    return root / spec.get('data_root', 'datasets/libero')


def local_dataset_path(filename, spec):
    if spec.get('dataset_kind') == 'zarr':
        path = Path(filename)
        if (len(path.parts) != 1 or path.is_absolute()
                or not filename.endswith('.zarr.zip')):
            return None
        return path
    prefix = spec['dataset_prefix']
    if not filename.startswith(prefix):
        return None
    path = Path(filename[len(prefix):])
    if (len(path.parts) != 2 or '..' in path.parts or path.is_absolute()
            or path.parts[0] not in spec['splits'] or path.suffix != '.hdf5'):
        return None
    return path
