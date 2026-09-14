"""Reproduce scoped checkpoint verification; keep raw logs local only."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FILES = ['crates/ferro-core/src/tensor.rs', 'crates/ferro-core/src/checkpoint.rs',
         'crates/ferro-core/tests/checkpoint_dtype.rs', 'crates/ferro-cuda/tests/checkpoint_dtype.rs']

def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}

before = hashes()
env = os.environ.copy()
torch_lib = ROOT / 'crates/ferro-py/.venv/Lib/site-packages/torch/lib'
if torch_lib.is_dir():
    env['PATH'] = str(torch_lib) + os.pathsep + env['PATH']
env['FERRO_REQUIRE_CUDA'] = '1'
commands = [
    ('cpu', ['cargo', 'test', '-p', 'ferro-core', '-j2', '--test', 'checkpoint_dtype']),
    ('core', ['cargo', 'test', '-p', 'ferro-core', '-j2']),
    ('cuda-check', ['cargo', 'check', '-p', 'ferro-cuda', '-j2']),
    ('cuda', ['cargo', 'test', '-p', 'ferro-cuda', '-j2', '--test', 'checkpoint_dtype']),
    ('diff', ['git', 'diff', '--check']),
]
results = {'base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'environment': {'FERRO_REQUIRE_CUDA': '1', 'PATH_prefix': str(torch_lib)}, 'runs': {}}
for name, command in commands:
    run = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / f'independent-{name}.log').write_bytes(run.stdout)
    text = run.stdout.decode(errors='replace')
    rows = []
    for block in re.split(r'(?m)^\s*(?:Running |Doc-tests )', text):
        matches = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', block)
        if matches:
            state, passed, failed, ignored = matches[-1]
            rows.append([int(passed), int(failed), int(ignored)])
    results['runs'][name] = {'command': command, 'exit_code': run.returncode,
        'passed': sum(r[0] for r in rows), 'failed': sum(r[1] for r in rows),
        'ignored': sum(r[2] for r in rows)}
    print(name, results['runs'][name], flush=True)
    assert run.returncode == 0, text
assert results['runs']['cpu']['passed'] == 8
assert results['runs']['cuda']['passed'] == 1
assert 'skipping CUDA' not in (OUT / 'independent-cuda.log').read_text(errors='replace')
results['sha256'] = hashes()
assert before == results['sha256'], 'sources changed during verification'
(OUT / 'independent-results.json').write_text(json.dumps(results, indent=2) + '\n')
