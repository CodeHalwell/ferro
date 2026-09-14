"""Frozen-source final verification. No production edits; preserve every exit."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import zipfile
import argparse
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'final', help='Fresh directory; existing evidence is never overwritten')
OUT = parser.parse_args().output_dir.resolve()
PY = ROOT / 'crates/ferro-py/.venv/Scripts/python.exe'
OUT.mkdir(exist_ok=False)
env = dict(os.environ)
runtime = Path(env['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia'
prefix = [runtime/'cuda_nvrtc/bin', runtime/'cublas/bin', PY.parent]
assert all(p.is_dir() for p in prefix)
env['PATH'] = os.pathsep.join(map(str, prefix)) + os.pathsep + env['PATH']
env['FERRO_REQUIRE_CUDA'] = '1'
env['VIRTUAL_ENV'] = str(PY.parents[1])
env['PYO3_PYTHON'] = str(PY)
env.pop('RUST_TEST_THREADS', None)
env.pop('PYTHONHOME', None)
rows = []
def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    paths = subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'], cwd=ROOT).decode().split('\0')
    # Include every repository source/config/script, tracked or untracked, including
    # verification scripts. Exclude only generated output/binaries and archived source snapshots.
    selected = {}
    excluded = []
    for s in sorted(set(filter(None, paths))):
        p = ROOT/s
        if not p.is_file():
            continue
        if p.suffix.lower() in {'.rs','.py','.toml','.lock','.yaml','.yml','.sh','.ps1','.json'} or p.name in {'.gitignore','Cargo.lock'}:
            if s.startswith('verification/static-empty-fix/sources/'):
                excluded.append(dict(path=s, reason='historical source snapshot, not executable current tree'))
            elif p.suffix == '.json':
                excluded.append(dict(path=s, reason='verification/diagnostic result data' if s.startswith(('verification/','evidencefoundation-wave/')) else 'review required JSON'))
                if not s.startswith(('verification/','evidencefoundation-wave/')):
                    selected[s] = digest(p)
            else:
                selected[s] = digest(p)
    return selected, excluded
def run(name, command, cwd=ROOT):
    command = list(map(str, command))
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (OUT/(name+'.log')).open('x', encoding='utf-8') as log:
        log.write('COMMAND: '+subprocess.list2cmdline(command)+'\nFERRO_REQUIRE_CUDA=1; RUST_TEST_THREADS unset\n'); log.flush()
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = dict(name=name,command=command,exit_code=proc.returncode,utc=start,log=str((OUT/(name+'.log')).relative_to(ROOT)))
    rows.append(row); save('results.json',rows); print(json.dumps(row),flush=True)
    return proc.returncode
def installed(name):
    return run(name,[PY,'-c','import ferro,ferro._native as n,pathlib,hashlib,json; p=pathlib.Path(ferro.__file__).parent; print(json.dumps({"package":str(p),"files":{str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(p.rglob("*")) if x.is_file() and x.suffix in (".py",".pyd")}},indent=2))'])
before, exclusions = sources(); save('sources-before.json',before); save('source-exclusions.json',exclusions)
installed('initial-installed')
run('required-unchanged',[sys.executable,'verification/dlpack-raw-review/run.py','all','--output-dir',OUT/'required-unchanged'])
run('wheel-build',[PY,'-m','maturin','build','--release','-j2','--manifest-path','crates/ferro-py/Cargo.toml','--out',OUT/'wheels','-i',PY])
wheels = list((OUT/'wheels').glob('*.whl'))
if len(wheels) == 1:
    wheel = wheels[0]
    run('wheel-install',[PY,'-m','pip','install','--force-reinstall','--no-deps',wheel])
    installed('wheel-installed-before')
    package = PY.parents[1]/'Lib/site-packages/ferro'
    with zipfile.ZipFile(wheel) as z:
        packaging = []
        for p in sorted((ROOT/'crates/ferro-py/python/ferro').rglob('*.py')):
            entry = 'ferro/'+p.relative_to(ROOT/'crates/ferro-py/python/ferro').as_posix()
            packaging.append(dict(path=entry,source=digest(p),wheel=hashlib.sha256(z.read(entry)).hexdigest(),installed=digest(package/p.relative_to(ROOT/'crates/ferro-py/python/ferro'))))
        native = [n for n in z.namelist() if n.endswith('.pyd')]
        for n in native:
            packaging.append(dict(path=n,wheel=hashlib.sha256(z.read(n)).hexdigest(),installed=digest(package/Path(n).name)))
    save('packaging.json',dict(wheel_sha256=digest(wheel),files=packaging,matching=all(len(set(v for k,v in r.items() if k!='path'))==1 for r in packaging)))
    run('wheel-integration',[sys.executable,'verification/static-perf-wave/run_checks.py','--output-dir',OUT/'wheel-integration'])
    run('cuda-default-parallel',['cargo','test','-j2','-p','ferro-cuda'])
    run('python-discovery',[PY,'-m','unittest','discover','-s','crates/ferro-py/tests','-v'])
    run('contracts17',[PY,'verification/python-api-contract/contract.py'])
    run('architecture8',[PY,'verification/python-architecture-api/test_architecture.py'])
    run('training-examples',[PY,'verification/python-architecture-api/training_examples.py'])
    installed('wheel-installed-after')
run('diff-check',['git','diff','--check'])
after, _ = sources(); save('sources-after.json',after)
save('source-stability.json',dict(stable=before==after,added=sorted(after.keys()-before.keys()),removed=sorted(before.keys()-after.keys()),changed=[p for p in before.keys() & after.keys() if before[p]!=after[p]]))
sys.exit(int(any(r['exit_code'] for r in rows) or before!=after))
