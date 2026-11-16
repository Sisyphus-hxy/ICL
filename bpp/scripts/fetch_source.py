#!/usr/bin/env python3
"""Fetch pinned GitHub sources, falling back from Git/GnuTLS to curl/OpenSSL archives."""
import argparse
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
GIT_OPTIONS = ['git', '-c', 'http.version=HTTP/1.1',
               '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=30']


def revision_of(path):
    if (path / '.git').exists():
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    manifest = path / '.source-revision.json'
    return json.loads(manifest.read_text())['revision'] if manifest.is_file() else None


def extract_archive(archive, directory, expected_root):
    """Validate root and members before extracting; never allow traversal or special files."""
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        if not members or any(member.name.split('/')[0] != expected_root for member in members):
            raise RuntimeError('Archive root does not match the pinned source revision')
        for member in members:
            tarfile.data_filter(member, str(directory))
        bundle.extractall(directory, filter='data')
    return directory / expected_root


def acquire(repo, revision, target, cache, method='auto', local=None):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError(f'Expected full commit SHA, got {revision}')
    if target.is_symlink():
        raise RuntimeError(f'Refusing to replace a source symlink: {target}')
    if target.exists() and any(target.iterdir()):
        if revision_of(target) != revision:
            raise RuntimeError(f'Existing source has a different or unknown revision: {target}')
        print(f'Using existing {target} @ {revision}', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    name = repo.split('/')[-1]
    git_source = f'https://github.com/{repo}.git'
    use_local = False
    if local and (local / '.git').exists():
        present = subprocess.run(['git', '-C', str(local), 'cat-file', '-e', revision + '^{commit}'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if present.returncode == 0:
            git_source = str(local)
            use_local = True
    if method != 'archive' or use_local:
        stage = cache / f'{name}-{revision}.git-work'
        stage.mkdir(exist_ok=True)
        subprocess.run(['git', 'init', '-q', str(stage)], check=True)
        for attempt in range(1, 4):
            print(f'Fetching {repo}@{revision} (Git HTTP/1.1, attempt {attempt}/3)', flush=True)
            try:
                subprocess.run(GIT_OPTIONS + ['-C', str(stage), 'fetch', '--depth=1', '--no-tags', git_source, revision],
                               check=True, timeout=45)
                subprocess.run(['git', '-C', str(stage), 'checkout', '--detach', revision], check=True)
                if target.exists():
                    target.rmdir()  # only an empty directory can reach here
                stage.rename(target)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt < 3:
                    time.sleep(2)
        if method == 'git':
            raise RuntimeError(f'Git fetch failed for {repo}')
        print('Git transfer failed; trying the official GitHub commit archive with curl.', flush=True)
    url = f'https://codeload.github.com/{repo}/tar.gz/{revision}'
    archive = cache / f'{name}-{revision}.tar.gz'
    partial = archive.with_suffix(archive.suffix + '.part')
    if not archive.is_file():
        subprocess.run(['curl', '--http1.1', '--fail', '--location', '--continue-at', '-',
                        '--connect-timeout', '15', '--max-time', '1800', '--retry', '3',
                        '--retry-all-errors', '--retry-delay', '2', '--output', str(partial), url], check=True)
        partial.replace(archive)
    with tempfile.TemporaryDirectory(prefix='extract-', dir=cache) as temporary:
        extracted = extract_archive(archive, Path(temporary), f'{name}-{revision}')
        checksum = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(block)
        (extracted / '.source-revision.json').write_text(json.dumps({
            'repository': repo, 'revision': revision, 'transport': 'github-archive',
            'url': url, 'archive_sha256': checksum.hexdigest(),
        }, indent=2) + '\n')
        if target.exists():
            target.rmdir()
        extracted.rename(target)


def parse_submodule_revisions(page, repositories):
    """Read full gitlink targets from GitHub's HTML or embedded JSON links."""
    page = unescape(page).replace(r'\/', '/')
    versions = {}
    for name, repo in repositories.items():
        pattern = (r'''["'](?:https://github\.com)?/''' + re.escape(repo)
                   + r'''/tree/([0-9a-f]{40})/?["']''')
        commits = {sha.lower() for sha in re.findall(pattern, page, re.IGNORECASE)}
        if len(commits) != 1:
            raise RuntimeError(
                f'Expected one full submodule revision for deps/{name} on the pinned '
                'GitHub page. Its layout may have changed or a proxy returned another page; '
                'refusing to substitute the default branch.')
        versions[name] = commits.pop()
    return versions


def submodule_revisions(source, revision, repositories, cache):
    """Resolve gitlinks without the rate-limited GitHub REST API."""
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError(f'Expected full commit SHA, got {revision}')
    if (source / '.git').exists():
        versions = {}
        for name in repositories:
            relative = 'deps/' + name
            entry = subprocess.check_output(
                ['git', '-C', str(source), 'ls-tree', revision, '--', relative], text=True)
            parts = entry.split()
            if (len(parts) != 4 or parts[:2] != ['160000', 'commit']
                    or not re.fullmatch(r'[0-9a-f]{40}', parts[2]) or parts[3] != relative):
                raise RuntimeError(f'Expected a pinned git submodule at {relative}')
            versions[name] = parts[2]
        return versions

    # Archives omit gitlinks. Use the exact commit's directory page, never main.
    url = f'https://github.com/real-stanford/behavior_prompting/tree/{revision}/deps'
    manifest = cache / f'submodules-{revision}.json'
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if (not isinstance(saved, dict) or saved.get('revision') != revision or saved.get('url') != url
                or saved.get('repositories') != repositories):
            raise RuntimeError(f'Submodule cache does not match the requested source: {manifest}')
        versions = saved.get('submodules')
        if (not isinstance(versions, dict) or versions.keys() != repositories.keys()
                or any(not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha)
                       for sha in versions.values())):
            raise RuntimeError(f'Invalid pinned submodule revisions in {manifest}')
        print(f'Using cached submodule revisions for {revision}', flush=True)
        return versions

    print('Reading pinned submodule revisions from github.com (no REST API).', flush=True)
    result = subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
                             '--connect-timeout', '10', '--max-time', '45', '--retry', '3',
                             url], capture_output=True, text=True, check=True)
    versions = parse_submodule_revisions(result.stdout, repositories)
    cache.mkdir(parents=True, exist_ok=True)
    partial = manifest.with_suffix('.json.part')
    partial.write_text(json.dumps({
        'revision': revision, 'url': url, 'repositories': repositories,
        'submodules': versions,
    }, indent=2) + '\n')
    partial.replace(manifest)
    return versions


def apply_source_patches(source):
    """Apply recorded local fixes once, refusing conflicting source changes."""
    records = []
    for patch in sorted((ROOT / 'patches').glob('*.patch')):
        command = ['git', '-C', str(source), 'apply']
        already_applied = subprocess.run(command + ['--reverse', '--check', str(patch)],
                                         capture_output=True, text=True)
        if already_applied.returncode != 0:
            ready = subprocess.run(command + ['--check', str(patch)], capture_output=True, text=True)
            if ready.returncode != 0:
                raise RuntimeError(f'Local source patch conflicts with existing files: {patch.name}\n'
                                   + ready.stderr.strip())
            subprocess.run(command + [str(patch)], check=True)
            print(f'Applied local source patch: {patch.name}', flush=True)
        records.append({'file': str(patch.relative_to(ROOT)),
                        'sha256': hashlib.sha256(patch.read_bytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['auto', 'git', 'archive'], default='auto')
    args = parser.parse_args()
    settings = json.loads((ROOT / 'config/sources.json').read_text())
    revision = settings['revision']
    source = ROOT / 'src'
    cache = ROOT / '.cache/source'
    acquire('real-stanford/behavior_prompting', revision, source, cache, args.method)
    dependencies = {
        'icrt': ('Max-Fu/icrt', None),
        'LIBERO': ('Lifelong-Robot-Learning/LIBERO', ROOT.parent.parent / 'xvla/external/LIBERO'),
    }
    commits = submodule_revisions(source, revision,
                                  {name: repo for name, (repo, _) in dependencies.items()}, cache)
    versions = {'behavior_prompting': revision}
    for name, (repo, local) in dependencies.items():
        commit = commits[name]
        acquire(repo, commit, source / 'deps' / name, cache, args.method, local)
        versions[name] = commit
    if (source / 'environment.yml').read_bytes() != (ROOT / 'upstream/environment.yml').read_bytes():
        raise RuntimeError('Downloaded environment.yml differs from the saved pinned upstream snapshot')
    patches = apply_source_patches(source)
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'logs/source-revisions.json').write_text(json.dumps(versions, indent=2) + '\n')
    (ROOT / 'logs/source-patches.json').write_text(json.dumps(patches, indent=2) + '\n')
    print('Pinned BPP source and required submodules are ready.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print(f'Source setup failed: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
