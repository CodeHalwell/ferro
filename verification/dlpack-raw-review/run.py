"""Scoped raw DLPack review: no test failures are filtered out."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Run current-source checks, not archived RED reproduction')
parser.add_argument('phase', choices=('all', 'probe', 'current-test', 'unit-existing', 'unit', 'build', 'eager', 'snapshot', 'integration'))
parser.add_argument('--output-dir', type=Path, required=True, help='fresh output directory outside historical evidence')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
if any(OUT.iterdir()):
    parser.error('output directory must be empty')
VENV = ROOT / 'crates/ferro-py/.venv'
PY = str(VENV / 'Scripts/python.exe')
env = dict(os.environ)
runtime = Path(os.environ['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia'
prefix = [runtime / 'cuda_nvrtc/bin', runtime / 'cublas/bin', VENV / 'Scripts', Path(sys.executable).parent]
assert all(p.is_dir() for p in prefix)
env['PATH'] = os.pathsep.join(map(str, prefix)) + os.pathsep + env['PATH']
env['VIRTUAL_ENV'] = str(VENV)
env['PYO3_PYTHON'] = PY
env['FERRO_REQUIRE_CUDA'] = '1'
env.pop('RUST_TEST_THREADS', None)
phase = args.phase
if phase == 'all':
    rows = []
    for name in ('unit', 'build', 'eager', 'snapshot', 'integration'):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), name, '--output-dir', str(OUT / name)], cwd=ROOT)
        row = json.loads((OUT / name / (name + '.json')).read_text())
        assert row['exit_code'] == result.returncode
        rows.append(row)
    (OUT / 'all-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    raise SystemExit(int(any(row['exit_code'] for row in rows)))
commands = {
    'probe': ['cargo', 'run', '-j2', '--manifest-path', 'verification/dlpack-raw-review/probe/Cargo.toml'],
    'current-test': ['cargo', 'test', '-j2', '--manifest-path', 'crates/ferro-py/Cargo.toml', '--lib', 'dlpack::tests::gpu_export_borrows_and_import_htod', '--', '--nocapture'],
    'unit-existing': ['cargo', 'test', '-j2', '--manifest-path', 'crates/ferro-py/Cargo.toml', '--lib', 'dlpack::tests::gpu_export_borrows_and_import_htod', '--', '--exact', '--nocapture'],
    'unit': ['cargo', 'test', '-j2', '--manifest-path', 'crates/ferro-py/Cargo.toml', '--lib', '--', '--nocapture'],
    'build': [PY, '-m', 'maturin', 'develop', '--release', '-j2', '--manifest-path', 'crates/ferro-py/Cargo.toml'],
    'eager': [PY, 'crates/ferro-py/tests/test_eager_dlpack.py', '-v'],
    'snapshot': [PY, 'crates/ferro-py/tests/test_static_snapshot_dlpack.py', '-v'],
    'integration': [sys.executable, 'verification/static-perf-wave/run_checks.py', '--output-dir', str(OUT / 'integration-checks')],
}
command = commands[phase]
if phase in ('current-test', 'unit', 'unit-existing'):
    env['PYTHONHOME'] = subprocess.check_output([PY, '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()
with (OUT / (phase + '.log')).open('x', encoding='utf-8') as log:
    log.write('COMMAND: ' + subprocess.list2cmdline(command) + '\nCUDA runtime prefix: ' + repr(list(map(str, prefix))) + '\nFERRO_REQUIRE_CUDA=1; RUST_TEST_THREADS unset (continuation runner sets 1)\n')
    log.flush()
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
row = dict(phase=phase, command=command, exit_code=result.returncode)
(OUT / (phase + '.json')).write_text(json.dumps(row, indent=2) + '\n')
if phase == 'integration':
    source = OUT / 'integration-checks'
    rows = json.loads((source / 'integration-results.json').read_text())
    assert len(rows) == 13
    for r in rows:
        shutil.copy2(ROOT / r['log'], OUT / ('integration-' + r['name'] + '.log'))
    shutil.copy2(source / 'integration-results.json', OUT / 'integration-results.json')
print(json.dumps(row), flush=True)
print((OUT / (phase + '.log')).read_text(encoding='utf-8')[-7000:])
sys.exit(result.returncode)
