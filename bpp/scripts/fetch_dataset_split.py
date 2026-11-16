#!/usr/bin/env python3
"""Download only the files of one dataset split, reusing fetch_assets verification.

Example:
    HF_ENDPOINT=https://hf-mirror.com BPP_INSECURE_TLS=1 \
        python3 scripts/fetch_dataset_split.py liberogen_spatial_combination \
        libero_spatial_selected_combinations_view
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarks import BENCHMARKS, ROOT, local_dataset_path, manifest_path
from fetch_assets import download


def main():
    benchmark, split = sys.argv[1], sys.argv[2]
    spec = BENCHMARKS[benchmark]
    if split not in spec['splits']:
        raise SystemExit(f'{split} not in {benchmark} splits: {sorted(spec["splits"])}')
    manifest = json.loads(manifest_path('data', benchmark).read_text())
    if not manifest.get('artifacts'):
        raise SystemExit('Manifest has no cached artifact metadata; run fetch_assets.py --metadata-only first')
    repo, revision = manifest['repo'], manifest['revision']
    pending = []
    for item in manifest['artifacts']:
        relative = local_dataset_path(item['rfilename'], spec)
        if relative is None or relative.parts[0] != split:
            continue
        lfs = item.get('lfs') or {}
        size = item.get('size') or lfs.get('size')
        sha = lfs.get('sha256')
        if not size or not sha:
            raise SystemExit(f'Missing size/SHA256 metadata: {item["rfilename"]}')
        pending.append((relative, item['rfilename'], size, sha))
    print(f'{len(pending)} files in split {split}', flush=True)
    done = 0
    for relative, rfilename, size, sha in sorted(pending):
        target = ROOT / 'datasets/libero' / relative
        if download(repo, revision, rfilename, target, size, sha, dataset=True):
            done += 1
    print(f'{done}/{len(pending)} files ready in {split}', flush=True)


if __name__ == '__main__':
    main()
