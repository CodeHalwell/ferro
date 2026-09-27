"""Append-only CI acceptance evidence in the isolated PR worktree."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PY = ROOT / '.venv/Scripts/python.exe'
env = dict(os.environ)
for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE', 'FERRO_REQUIRE_CUDA', 'CUDA'):
    env.pop(key, None)
env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
           CARGO_TARGET_DIR=str(ROOT / 'target-stabilisation-final'),
           PYO3_PYTHON=str(PY), VIRTUAL_ENV=str(PY.parents[1]))
runtime = Path(env['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia'
libs = [PY.parent, runtime / 'cuda_nvrtc/bin', runtime / 'cublas/bin']
env['PATH'] = os.pathsep.join(str(p) for p in libs if p.is_dir()) + os.pathsep + env['PATH']
name, *command = sys.argv[1:]
if command[0] == 'python':
    command[0] = str(PY)
with (OUT / (name + '.log')).open('x', encoding='utf8') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
with (OUT / (name + '.json')).open('x', encoding='utf8') as output:
    json.dump({'command': command, 'cwd': str(ROOT), 'exit_code': result.returncode,
               'environment': {k: env.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CARGO_TARGET_DIR')}}, output, indent=2)
print(name, 'exit', result.returncode)
print((OUT / (name + '.log')).read_text(encoding='utf8')[-6000:])
sys.exit(result.returncode)
