"""Serial combined-tree checks. Run after building the current maturin extension."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def select_python(root, platform, fallback):
    relative = 'Scripts/python.exe' if platform == 'nt' else 'bin/python'
    candidate = root / 'crates/ferro-py/.venv' / relative
    return str(candidate) if candidate.is_file() else fallback


PY = select_python(ROOT, os.name, sys.executable)
COMMANDS = [
    ('core', ['cargo', 'test', '-j2', '-p', 'ferro-core']),
    ('cpu-tokenizer', ['cargo', 'test', '-j2', '-p', 'ferro-fastcpu', '-p', 'ferro-tokenizer']),
    ('cuda', ['cargo', 'test', '-j2', '-p', 'ferro-cuda']),
    ('cuda-check', ['cargo', 'check', '-j2', '-p', 'ferro-cuda', '--all-targets']),
    ('python-capture', [PY, '-m', 'unittest', 'discover', '-s', 'crates/ferro-py/tests', '-v']),
    ('python-compiled', [PY, 'crates/ferro-py/examples/py_compiled_fusion_regression.py']),
    ('python-fuse', [PY, 'crates/ferro-py/examples/py_fuse_regression.py']),
    ('python-general', [PY, 'examples/py_regression.py']),
    ('python-ops', [PY, 'examples/ops_vs_torch.py']),
    ('python-safetensors', [PY, 'examples/safetensors_vs_python.py']),
    ('python-fuzz', [PY, 'examples/fuzz_vs_torch.py', '--trials', '200', '--seed', '0']),
    ('diff-check', ['git', 'diff', '--check']),
]


def main():
    results = []
    for name, command in COMMANDS:
        log = ROOT / 'verification' / ('integration-' + name + '.log')
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(name=name, command=command, exit_code=result.returncode, log=str(log.relative_to(ROOT)))
        results.append(row)
        print(json.dumps(row), flush=True)
        (ROOT / 'verification/integration-results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return int(any(row['exit_code'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
