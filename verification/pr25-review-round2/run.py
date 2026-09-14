import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PY = ROOT / 'crates/ferro-py/.venv/Scripts/python.exe'
env = os.environ.copy()
env['FERRO_REQUIRE_CUDA'] = '1'
lib = subprocess.check_output([str(PY), '-c', 'import torch,pathlib;print(pathlib.Path(torch.__file__).parent / "lib")'], text=True).strip()
env['PATH'] = lib + os.pathsep + env['PATH']
label, *args = sys.argv[1:]
args = [str(PY) if a == '@python' else a for a in args]
p = subprocess.run(args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(OUT / (label + '.log')).write_bytes(p.stdout)
print(p.stdout.decode(errors='replace'))
print('exit_code:', p.returncode)
sys.exit(p.returncode)
