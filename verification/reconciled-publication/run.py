"""Append-only serial final verification; no performance claims."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if os.environ.get('FERRO_FINAL_EVIDENCE_RUN'):
    OUT = OUT / os.environ['FERRO_FINAL_EVIDENCE_RUN']
    OUT.mkdir(exist_ok=False)
PY = ROOT / '.venv/Scripts/python.exe'
env = dict(os.environ)
rt = Path(env['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia'
libs = [rt/'cuda_nvrtc/bin', rt/'cublas/bin', PY.parent]
if not all(p.is_dir() for p in libs): raise RuntimeError('missing DLL directory')
env['PATH'] = os.pathsep.join(map(str, libs)) + os.pathsep + env['PATH']
env.update(FERRO_REQUIRE_CUDA='1', CUDA='1', PYO3_PYTHON=str(PY), VIRTUAL_ENV=str(PY.parents[1]), PYTHONDONTWRITEBYTECODE='1', CARGO_TARGET_DIR=str(ROOT/'target-stabilisation-final'))
for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONOPTIMIZE', 'RUST_TEST_THREADS'): env.pop(key, None)
def save(name, obj):
    with (OUT/name).open('x', encoding='utf8') as f: json.dump(obj, f, indent=2)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():
    paths = subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'], cwd=ROOT).decode().split('\0')
    selected = set(s for s in paths if s and (s.startswith(('crates/','examples/','bench/','scripts/','.cargo/')) or s in ('Cargo.toml','Cargo.lock','rust-toolchain.toml','verification/reconciled-publication/run.py')))
    selected.update(s for s in ('Cargo.toml','Cargo.lock','rust-toolchain.toml','rust-toolchain') if (ROOT/s).is_file())
    selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'.cargo').rglob('*') if p.is_file())
    return {s: sha(ROOT/s) for s in sorted(selected) if (ROOT/s).is_file() and Path(s).suffix not in ('.pdb','.pyd','.pyc')}
def installed():
    pkg = PY.parents[1]/'Lib/site-packages/ferro'
    return {p.relative_to(pkg).as_posix():sha(p) for p in sorted(pkg.rglob('*')) if p.suffix in ('.py','.pyd')}
def run(name, args, extra=None):
    args = list(map(str,args)); e = {**env, **(extra or {})}
    with (OUT/(name+'.log')).open('x',encoding='utf8') as f:
        f.write('COMMAND: '+subprocess.list2cmdline(args)+'\n'); f.flush()
        try: code = subprocess.run(args,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=3600).returncode
        except subprocess.TimeoutExpired: code = 124
    save(name+'.json',dict(command=args,cwd=str(ROOT),exit_code=code,environment={k:e.get(k) for k in ('PATH','CUDA','FERRO_REQUIRE_CUDA','RUST_TEST_THREADS','PYO3_PYTHON','PYTHONHOME','PYTHONOPTIMIZE','CARGO_TARGET_DIR','FERRO_GPT2_DIR')}))
    print(name, code, flush=True)
    if code: raise SystemExit(code)
def freeze_end():
    after=manifest(); save('sources-after.json',after)
    stability={'sources_stable':json.loads((OUT/'sources-before.json').read_text())==after}
    if (OUT/'installed-before.json').exists():
        i=installed(); save('installed-after.json',i)
        stability['installed_stable']=json.loads((OUT/'installed-before.json').read_text())==i
    save('stability.json',stability)
    if not all(stability.values()): raise RuntimeError('frozen input drift')
if __name__ == '__main__':
    phase=sys.argv[1]
    if phase=='command': run(sys.argv[2],sys.argv[3:])
    elif phase=='final':
        save('sources-before.json',manifest())
        try:
            run('gpu',['nvidia-smi','--query-gpu=name,driver_version','--format=csv'])
            run('wheel-build',[PY,'-m','maturin','build','--release','-j2','--manifest-path','crates/ferro-py/Cargo.toml','--out',OUT/'wheels','-i',PY])
            wheels=list((OUT/'wheels').glob('*.whl'))
            if len(wheels)!=1: raise RuntimeError('expected one wheel')
            run('wheel-install',[PY,'-m','pip','install','--force-reinstall','--no-deps',wheels[0]])
            pkg=PY.parents[1]/'Lib/site-packages/ferro'
            entries=[]
            with zipfile.ZipFile(wheels[0]) as z:
                for p in sorted((ROOT/'crates/ferro-py/python/ferro').rglob('*.py')):
                    rel=p.relative_to(ROOT/'crates/ferro-py/python/ferro'); name='ferro/'+rel.as_posix()
                    entries.append(dict(path=name,source=sha(p),wheel=hashlib.sha256(z.read(name)).hexdigest(),installed=sha(pkg/rel)))
                for name in z.namelist():
                    if name.endswith('.pyd'): entries.append(dict(path=name,wheel=hashlib.sha256(z.read(name)).hexdigest(),installed=sha(pkg/Path(name).name)))
            save('packaging.json',dict(wheel=str(wheels[0]),sha256=sha(wheels[0]),files=entries))
            if not all(len(set(v for k,v in r.items() if k!='path'))==1 for r in entries): raise RuntimeError('package mismatch')
            save('installed-before.json',installed())
            run('identity',[PY,'-c','import ferro,ferro._native as n,sys, pathlib, importlib.metadata as m; p=pathlib.Path(sys.prefix)/"Lib/site-packages/ferro"; print(ferro.__file__); print(n.__file__); print(m.distribution("ferro").read_text("direct_url.json"));\nif pathlib.Path(ferro.__file__).parent!=p or pathlib.Path(n.__file__).parent!=p: raise RuntimeError("unexpected import origin")\nferro.cuda_init(); print("CUDA initialized")'])
            for name,file in [('tolist','test_tolist_typed.py'),('registry','test_cuda_registry_copy.py'),('restart','test_training_restart_gates.py')]:
                for suffix,flags in [('normal',[]),('optimized',['-O'])]: run(name+'-'+suffix,[PY,'-B',*flags,'crates/ferro-py/tests/'+file,'-v'])
            run('python-discovery',[PY,'-B','-m','unittest','discover','-s','crates/ferro-py/tests','-v'])
            for name in ('core','cuda','fastcpu','tokenizer'): run(name,['cargo','test','-j2','-p','ferro-'+name])
            base=subprocess.check_output([str(PY),'-c','import sys; print(sys.base_prefix)'],text=True).strip()
            run('bindings',['cargo','test','-j2','--manifest-path','crates/ferro-py/Cargo.toml','--lib'],{'PYTHONHOME':base})
        finally: freeze_end()
    else: raise SystemExit('unknown phase')
