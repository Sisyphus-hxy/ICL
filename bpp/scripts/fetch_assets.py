#!/usr/bin/env python3
"""Prepare official BPP artifacts, reusing verified local files before downloading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
# Opt-in TLS relaxation for networks whose egress intercepts CDN redirects with a
# self-signed certificate (e.g. hf-mirror -> cas-bridge.xethub.hf.co). File
# integrity is still enforced by the SHA256 check after every download.
CURL_TLS_ARGS = ['--insecure'] if os.environ.get('BPP_INSECURE_TLS') == '1' else []

from benchmarks import BENCHMARKS, ROOT, data_root, local_dataset_path, manifest_path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(repo, revision, filename, target, expected_size, expected_sha, dataset,
             reuse_roots=(), local_only=False):
    if target.is_file() and target.stat().st_size == expected_size and digest(target) == expected_sha:
        print(f'Verified existing: {target}', flush=True)
        return True
    if target.exists() or target.is_symlink():
        raise RuntimeError(f'Existing file does not match the official release; preserved: {target}')
    relative = Path(*Path(filename).parts[1:]) if filename.startswith('demonstration_data/') else Path(filename)
    candidates = []
    for root in reuse_roots:
        candidates.extend([root / filename, root / relative, root / Path(filename).name])
        candidates.extend(sorted((root / 'snapshots').glob(f'*/{filename}')))
    checked = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if (candidate.is_file() and candidate.stat().st_size == expected_size
                and digest(candidate) == expected_sha):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(resolved)
            print(f'Reused: {target} -> {resolved}', flush=True)
            return True
    partial = target.with_name(target.name + '.part')
    if partial.is_file() and partial.stat().st_size == expected_size and digest(partial) == expected_sha:
        partial.replace(target)
        print(f'Verified completed partial: {target}', flush=True)
        return True
    if local_only:
        print(f'Missing locally: {target}', flush=True)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'datasets/' if dataset else ''
    url = f'{HF_ENDPOINT}/{prefix}{repo}/resolve/{revision}/{quote(filename)}'
    print(f'Downloading {filename} ({expected_size:,} bytes)', flush=True)
    subprocess.run(['curl', '--fail', '--location', '--continue-at', '-',
                    '--connect-timeout', '15', '--speed-time', '60', '--speed-limit', '1024',
                    '--retry', '2', *CURL_TLS_ARGS, '--output', str(partial), url], check=True)
    if partial.stat().st_size != expected_size or digest(partial) != expected_sha:
        raise RuntimeError(f'Integrity check failed: {partial}; inspect it before retrying')
    partial.replace(target)
    return True


def get_metadata(spec, kind, previous, full, local_only):
    dataset = kind == 'data'
    repo = spec['dataset_repo' if dataset else 'checkpoint_repo']
    revision = previous.get('revision') or spec.get('dataset_revision' if dataset else 'checkpoint_revision', 'main')
    if previous.get('repo') == repo:
        if previous.get('artifacts') and (not dataset or not full or previous.get('catalog_full')):
            return {'sha': revision, 'siblings': previous['artifacts'],
                    'catalog_full': previous.get('catalog_full', False)}
        # Older manifests already contain enough metadata for the requested files.
        files = previous.get('files', [])
        if files and (not dataset or not full or all(
                count is not None and sum(Path(item['path']).parent.name == split for item in files) == count
                for split, count in spec['splits'].items())):
            prefix = 'datasets/libero/' if dataset else 'checkpoints/'
            return {'sha': revision, 'catalog_full': not dataset or all(
                count is not None and sum(Path(item['path']).parent.name == split for item in files) == count
                for split, count in spec['splits'].items()), 'siblings': [
                {'rfilename': spec['dataset_prefix'] + item['path'].removeprefix(prefix)
                 if dataset else item['path'].removeprefix(prefix),
                 'size': item['bytes'], 'lfs': {'sha256': item['sha256']}}
                for item in files]}
    if local_only:
        raise RuntimeError('No cached official artifact metadata. Run with --metadata-only once '
                           'to record size/SHA256 without downloading weights or data.')
    api_type = 'datasets' if dataset else 'models'
    url = f'{HF_ENDPOINT}/api/{api_type}/{repo}/revision/{revision}?blobs=true'
    response = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                               '--connect-timeout', '10', '--max-time', '30', url],
                              check=True, capture_output=True, text=True)
    return json.loads(response.stdout)


def extract_zarr_archives(root_dir):
    """Unpack downloaded .zarr.zip stores next to the archives using the BPP env."""
    if not list(root_dir.glob('*.zarr.zip')):
        return
    from environment import environment_prefix
    python = environment_prefix('default') / 'bin/python'
    script = ROOT / 'src/behavior_prompting/scripts/unzip_zarr.py'
    if not python.is_file() or not script.is_file():
        raise RuntimeError(f'Zarr extraction needs the BPP environment and source: {python}, {script}')
    subprocess.run([str(python), str(script), '-i', str(root_dir)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['checkpoint', 'data'])
    parser.add_argument('--benchmark', choices=BENCHMARKS, default='libero')
    parser.add_argument('--full', action='store_true', help='All tasks in the selected benchmark; default is one smoke task')
    parser.add_argument('--reuse-from', type=Path, action='append', default=[],
                        help='Existing checkpoint/data directory or HF repo cache; may be repeated')
    parser.add_argument('--local-only', action='store_true', help='Reuse local artifacts only; never access the network')
    parser.add_argument('--metadata-only', action='store_true', help='Record official metadata without downloading artifacts')
    args = parser.parse_args()
    spec = BENCHMARKS[args.benchmark]
    dataset = args.kind == 'data'
    repo = spec['dataset_repo' if dataset else 'checkpoint_repo']
    manifest_file = manifest_path(args.kind, args.benchmark)
    previous = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    if previous.get('repo') != repo:
        previous = {}
    metadata = get_metadata(spec, args.kind, previous, args.full, args.local_only)
    revision = metadata['sha']
    if dataset:
        available = [item for item in metadata['siblings'] if local_dataset_path(item['rfilename'], spec)]
        if spec.get('dataset_kind') == 'zarr':
            wanted = {spec['eval_dataset'] + '.zip'}
            if args.full:
                wanted.add(spec['train_dataset'] + '.zip')
            files = [item for item in available if item['rfilename'] in wanted]
            missing_names = sorted(wanted - {item['rfilename'] for item in files})
            if missing_names:
                raise RuntimeError(f'Requested official artifacts not found in {repo}@{revision}: '
                                   f'{missing_names}')
        elif args.full:
            for split, count in spec['splits'].items():
                found = sum(local_dataset_path(item['rfilename'], spec).parts[0] == split for item in available)
                if found == 0 or (count is not None and found != count):
                    raise RuntimeError(f'Incomplete upstream dataset listing for {split}: {found}')
            files = available
        elif args.benchmark == 'libero':
            files = [item for item in available if item['rfilename'] == f'libero_spatial/{spec["smoke_task"]}.hdf5']
        else:
            files = sorted([item for item in available if local_dataset_path(item['rfilename'], spec).parts[0]
                            == spec['smoke_split']], key=lambda item: item['rfilename'])[:1]
    else:
        available = files = [item for item in metadata['siblings'] if item['rfilename'] == spec['checkpoint']]
    if not files:
        raise RuntimeError(f'Requested official artifacts not found in {repo}@{revision}')
    manifest = {'repo': repo, 'revision': revision, 'benchmark': args.benchmark,
                'complete': False, 'full': args.full, 'files': previous.get('files', []),
                'catalog_full': metadata.get('catalog_full', True), 'artifacts': available}
    if dataset:
        manifest['dataset_files'] = sorted(str(local_dataset_path(item['rfilename'], spec)) for item in available)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.metadata_only:
        print(f'Recorded metadata for {len(available)} artifacts; no artifacts downloaded.')
        return
    cache_name = ('datasets--' if dataset else 'models--') + repo.replace('/', '--')
    reuse_roots = [path.expanduser().resolve() for path in args.reuse_from]
    for cache in [ROOT / '.cache/huggingface/hub', Path.home() / '.cache/huggingface/hub']:
        reuse_roots.append(cache / cache_name)
    missing = []
    for item in sorted(files, key=lambda entry: entry['rfilename']):
        lfs = item.get('lfs') or {}
        size = item.get('size') or lfs.get('size')
        sha = lfs.get('sha256')
        if not size or not sha:
            raise RuntimeError(f'Missing size/SHA256 metadata: {item["rfilename"]}')
        relative = local_dataset_path(item['rfilename'], spec) if dataset else Path(item['rfilename'])
        target = (data_root(spec, ROOT) if dataset else ROOT / 'checkpoints') / relative
        if not download(repo, revision, item['rfilename'], target, size, sha, dataset,
                        reuse_roots, args.local_only):
            missing.append(str(relative))
            continue
        entry = {'path': str(target.relative_to(ROOT)), 'bytes': size, 'sha256': sha}
        manifest['files'] = [old for old in manifest['files'] if old['path'] != entry['path']] + [entry]
        manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    if missing:
        raise RuntimeError(f'{len(missing)} requested artifacts missing locally; nothing downloaded.')
    manifest['complete'] = True
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    if dataset and spec.get('dataset_kind') == 'zarr':
        extract_zarr_archives(data_root(spec, ROOT))
    print(f'All {len(files)} requested files ready and verified.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Download failed: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
