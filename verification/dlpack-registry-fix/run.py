"""Bounded required-CUDA correctness runner; preserves per-run evidence."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
env = os.environ.copy()
env['PATH'] = os.pathsep.join([str(Path(env['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia/cuda_nvrtc/bin'), str(Path(env['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia/cublas/bin'), env['PATH']])
env['FERRO_REQUIRE_CUDA'] = '1'
env['VIRTUAL_ENV'] = str(ROOT / 'crates/ferro-py/.venv')
env.pop('RUST_TEST_THREADS', None)
name, *command = sys.argv[1:]
with (Path(__file__).parent / (name + '.log')).open('x', encoding='utf-8') as log:
    log.write('COMMAND: ' + subprocess.list2cmdline(command) + '\nFERRO_REQUIRE_CUDA=1; CUDA DLL PATH; RUST_TEST_THREADS unset\n')
    log.flush()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=480)
        code = result.returncode
    except subprocess.TimeoutExpired:
        log.write('\nTIMEOUT: child killed and reaped\n')
        code = 124
    log.write('\nEXIT_CODE: ' + str(code) + '\n')
print(name, 'exit', code)
sys.exit(code)
