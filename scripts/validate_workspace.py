#!/usr/bin/env python3
"""Offline source/config validation using only the Python standard library."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    errors = []
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    manifest = json.loads((ROOT / 'third_party/SHA256SUMS.json').read_text())
    for relative, expected in {**lock['downloaded_files_sha256'], **manifest['files']}.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'Missing: {relative}')
        elif sha256(path) != expected:
            errors.append(f'Hash mismatch: {relative}')
    pdf = ROOT / 'papers/MiSS_ICLR2026.pdf'
    if pdf.exists() and not pdf.read_bytes().startswith(b'%PDF-'):
        errors.append('Paper is not a PDF')
    configs = sorted((ROOT / 'configs').glob('*.json'))
    for path in configs:
        conf = json.loads(path.read_text())
        if conf['status'] != 'plan_only_not_gpu_validated':
            errors.append(f'Unexpected validation status: {path.name}')
        if conf['model']['revision'] != lock['huggingface'][conf['model']['id']]['revision']:
            errors.append(f'Model revision mismatch: {path.name}')
        if conf['data']['revision'] != lock['huggingface']['openai/gsm8k']['revision']:
            errors.append(f'Dataset revision mismatch: {path.name}')
        if not re.fullmatch(r'[0-9a-f]{40}', conf['model']['revision']):
            errors.append(f'Unpinned model: {path.name}')
        if conf['data']['split_seed'] != 2026 or conf['data']['use_official_test']:
            errors.append(f'Exploration split protocol mismatch: {path.name}')
        if conf['adapter']['method'] in ('miss', 'weighted_miss') and conf['adapter']['init_weights'] is not True:
            errors.append(f'Expected standard MiSS initialization: {path.name}')
        if conf['training']['max_steps'] <= 0:
            errors.append(f'Invalid optimizer steps: {path.name}')
    summary = {
        'source_files_checked': len(lock['downloaded_files_sha256']) + len(manifest['files']),
        'configs_checked': len(configs), 'errors': errors,
        'scope': 'Offline integrity and config checks only; no GPU/runtime/performance validation.',
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
