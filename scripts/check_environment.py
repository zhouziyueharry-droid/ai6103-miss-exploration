#!/usr/bin/env python3
"""Report environment only. Never load a model, download assets, or train."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    packages = {}
    for name in ('torch', 'peft', 'transformers', 'datasets', 'accelerate', 'safetensors'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        'python': sys.version.split()[0], 'platform': platform.system(),
        'machine': platform.machine(), 'packages': packages,
        'cuda_checked': False, 'training_started': False,
        'note': 'Metadata only unless --require-cuda is supplied; no model is loaded.',
    }
    errors = []
    if args.require_cuda:
        if sys.version_info < (3, 11):
            errors.append('Pinned PEFT requires Python >=3.11')
        for name, version in packages.items():
            if version is None:
                errors.append(f'Missing dependency: {name}')
        if packages['torch']:
            try:
                import torch
                report['cuda_checked'] = True
                report['torch_cuda_build'] = torch.version.cuda
                report['cuda_available'] = torch.cuda.is_available()
                if not report['cuda_available']:
                    errors.append('CUDA is unavailable; stop before training')
                else:
                    report['gpus'] = [
                        {'name': torch.cuda.get_device_name(i),
                         'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
                         'compute_capability': list(torch.cuda.get_device_capability(i))}
                        for i in range(torch.cuda.device_count())
                    ]
                    report['bf16_supported'] = torch.cuda.is_bf16_supported()
                    if not report['bf16_supported']:
                        errors.append('Proposed BF16 config requires adjustment')
                from peft import MissConfig, LoraConfig
                import peft
                from transformers import AutoModelForCausalLM, AutoTokenizer
                report['peft_import_path'] = peft.__file__
                report['adapter_import_ok'] = bool(MissConfig and LoraConfig and AutoModelForCausalLM and AutoTokenizer)
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
    report['errors'] = errors
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + '\n', encoding='utf-8')
    print(payload)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
