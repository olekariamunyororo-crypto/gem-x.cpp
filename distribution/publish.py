#!/usr/bin/env python3
"""Publish the explicit GEM-X GGUF bundle; verify-only unless --upload is set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = 'LocalAI-io/GEM-X-GGUF'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'distribution/manifest.json').read_text())
    files = {}
    for name, expected in manifest.items():
        path = args.models / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Invalid file or size: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Hash mismatch: {name}')
        files[name] = path
        print(f'Verified {name}', flush=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain']).strip():
        raise ValueError('Commit source changes before publishing')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    documents = {
        'manifest.json': (ROOT / 'distribution/manifest.json').read_bytes(),
        'NOTICE': (ROOT / 'distribution/NOTICE').read_bytes(),
        'LICENSE.pdf': (ROOT / 'LICENSES/NVIDIA-Open-Model-2025-10-24.pdf').read_bytes(),
    }
    for name in ['DINOv3.md', 'MHR-Apache-2.0.txt', 'Momentum.txt', 'YOLOX-Apache-2.0.txt', 'NVIDIA-Open-Model-2025-10-24.pdf']:
        documents['LICENSES/' + name] = (ROOT / 'LICENSES' / name).read_bytes()
    documents['LICENSES/Apache-2.0.txt'] = (ROOT / 'LICENSE').read_bytes()
    for name in ['reference/sources.json', 'include/gemx.h', 'docs/LIVE-OFFLINE-PARITY.md', 'docs/LIVE-PIPELINING.md', 'docs/LICENSING.md']:
        documents[name] = (ROOT / name).read_bytes()
    card = (ROOT / 'distribution/README.md').read_text()
    card = card.replace('(../README.md)', '(USAGE.md)').replace('(../demo/README.md)', '(USAGE.md)')
    card = card.replace('(../', '(')
    documents['README.md'] = card.encode()
    documents['USAGE.md'] = (ROOT / 'distribution/USAGE.md').read_bytes()
    documents['SOURCE.json'] = (json.dumps({'runtime': 'gem-x.cpp', 'commit': source, 'models': manifest}, indent=2) + '\n').encode()
    sums = {name: row['sha256'] for name, row in manifest.items()}
    sums.update({name: hashlib.sha256(data).hexdigest() for name, data in documents.items()})
    documents['SHA256SUMS'] = ''.join(f'{value}  {name}\n' for name, value in sorted(sums.items())).encode()
    print(f'{REPO}: {len(files)} models, {len(documents)} documents; source {source}', flush=True)
    if not args.upload:
        print('Verified locally; no network calls or uploads.')
        return
    from huggingface_hub import HfApi, CommitOperationAdd, ModelCard
    from huggingface_hub.errors import RepositoryNotFoundError
    ModelCard(card).validate()
    api = HfApi()
    try:
        info = api.model_info(REPO, files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(REPO, repo_type='model', private=False, exist_ok=False)
        info = api.model_info(REPO, files_metadata=True)
    existing = {f.rfilename: f for f in info.siblings}
    operations = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=d) for n, d in documents.items()]
    for name, path in files.items():
        remote = existing.get(name)
        if remote:
            if not remote.lfs or remote.lfs.sha256 != manifest[name]['sha256']:
                raise ValueError(f'Refusing to replace a different existing weight: {name}')
            print(f'Already present: {name}', flush=True)
        else:
            operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
    result = api.create_commit(repo_id=REPO, operations=operations, parent_commit=info.sha,
                              commit_message='Publish verified GEM-X F32 GGUF bundle', num_threads=4)
    print(result.commit_url, flush=True)
    info = api.model_info(REPO, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    for name, row in manifest.items():
        assert remote[name].size == row['bytes'] and remote[name].lfs.sha256 == row['sha256'], name
    print('All remote weight sizes and SHA-256 values verified.', flush=True)


if __name__ == '__main__':
    main()
