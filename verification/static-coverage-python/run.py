"""Run acceptance against installed bindings only; never rebuild or install."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PYTHON = ROOT / "crates/ferro-py/.venv/Scripts/python.exe"
name = sys.argv[1] if len(sys.argv) > 1 else "acceptance"
if Path(name).name != name or name in (".", ".."):
    raise SystemExit("log name must be a plain filename stem")
env = os.environ.copy()
prefix = [Path(env["LOCALAPPDATA"]) / "Temp/cuda-rt/nvidia" / part / "bin"
          for part in ("cuda_nvrtc", "cublas")]
env["PATH"] = os.pathsep.join([*(str(path) for path in prefix), env["PATH"]])
env["FERRO_REQUIRE_CUDA"] = "1"
command = [str(PYTHON), "crates/ferro-py/tests/test_static_coverage.py", "-v", *sys.argv[2:]]
logpath = OUT / (name + ".log")
if logpath.exists():
    raise SystemExit(f"refusing to overwrite evidence: {logpath}")
with logpath.open("w", encoding="utf-8") as log:
    log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
    log.write("FERRO_REQUIRE_CUDA=1; CUDA PATH=" + repr(list(map(str, prefix))) + "\n")
    log.write("TEST_SHA256=" + hashlib.sha256((ROOT / command[1]).read_bytes()).hexdigest() + "\n")
    log.flush()
    probe = subprocess.run([str(PYTHON), "-c", "import ferro,torch; print('ferro:',ferro.__file__); print('torch:',torch.__version__); print('CUDA:',torch.cuda.is_available(),ferro.cuda_is_available()); print('GPU:',torch.cuda.get_device_name(0))"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nPROBE_EXIT_CODE: {probe.returncode}\nEXIT_CODE: {result.returncode}\n")
print(json.dumps(dict(command=command, exit_code=result.returncode, probe_exit_code=probe.returncode, log=str(logpath))))
raise SystemExit(result.returncode or probe.returncode)
