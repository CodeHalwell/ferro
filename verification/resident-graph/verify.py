"""Build and exercise the resident graph and CPU restart gates in a fresh evidence directory."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
                                     'crates', 'examples', 'verification/resident-graph/verify.py', '.github/workflows/correctness.yml'], cwd=ROOT, text=True)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sorted(set(paths.splitlines()))
            if (ROOT / name).is_file() and Path(name).suffix in ('.rs', '.cu', '.py', '.toml', '.yml', '.lock')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = sources()
    (output / 'sources-before.json').write_text(json.dumps(before, indent=2))
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '0',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    env['FERRO_REQUIRE_CUDA'] = '1'
    if os.name == 'nt':
        torch_lib = Path(importlib.metadata.distribution('torch').locate_file('torch/lib'))
        if torch_lib.is_dir():
            env['PATH'] = str(torch_lib) + os.pathsep + env.get('PATH', '')
    results = []

    def run(label, arguments):
        with (output / (label + '.log')).open('w') as log:
            result = subprocess.run(arguments, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append({'name': label, 'command': arguments, 'exit': result.returncode})
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        print(label, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit('Failed: ' + label + '; see ' + str(output))

    run('build', [sys.executable, '-m', 'maturin', 'build', '--release', '-j2', '--locked',
                  '--manifest-path', 'crates/ferro-py/Cargo.toml', '--out', str(output / 'wheels')])
    wheels = list((output / 'wheels').glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Expected one wheel')
    run('install', [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', str(wheels[0])])
    import ferro
    import ferro._native as native
    package = Path(ferro.__file__).parent
    wheel_hashes = {}
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in wheel.namelist():
            if name.startswith('ferro/') and (name.endswith('.py') or '.pyd' in name or name.endswith('.so')):
                data = wheel.read(name)
                installed = package / name.removeprefix('ferro/')
                if installed.read_bytes() != data:
                    raise AssertionError('Wheel/install mismatch: ' + name)
                if name.endswith('.py'):
                    source = ROOT / 'crates/ferro-py/python' / name
                    if source.read_bytes() != data:
                        raise AssertionError('Source/wheel mismatch: ' + name)
                wheel_hashes[name] = hashlib.sha256(data).hexdigest()
    provenance = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'python': sys.version, 'platform': platform.platform(),
                  'torch': importlib.metadata.version('torch'), 'native_path': native.__file__,
                  'wheel_sha256': hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
                  'package_hashes': wheel_hashes, 'proof_device': 'cpu (training/restart), cuda:0 (primitives)', 'cuda_required': True}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    for task in ('regression', 'classifier', 'gnn'):
        run(task, [sys.executable, '-B', 'examples/train_restart_' + task + '.py'])
    run('python', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'crates/ferro-py/tests', '-v'])
    run('core', ['cargo', 'test', '-j2', '-p', 'ferro-core'])
    run('cuda-check', ['cargo', 'check', '-j2', '-p', 'ferro-cuda', '--all-targets'])
    run('cuda', ['cargo', 'test', '-j2', '-p', 'ferro-cuda'])
    run('cpu-tokenizer', ['cargo', 'test', '-j2', '-p', 'ferro-fastcpu', '-p', 'ferro-tokenizer'])
    for name, path in (
            ('bindings', 'examples/py_regression.py'),
            ('ops', 'examples/ops_vs_torch.py'),
            ('safetensors', 'examples/safetensors_vs_python.py'),
            ('compiled', 'crates/ferro-py/examples/py_compiled_fusion_regression.py'),
            ('fusion', 'crates/ferro-py/examples/py_fuse_regression.py')):
        run(name, [sys.executable, '-B', path])
    run('fuzz', [sys.executable, '-B', 'examples/fuzz_vs_torch.py', '--trials', '200', '--seed', '0'])
    # Binding Rust tests embed Python. Match its DLL and standard library exactly.
    if os.name == 'nt':
        env['PYO3_PYTHON'] = sys.executable
        env['PYTHONHOME'] = sys.base_prefix
        env['PATH'] = sys.base_prefix + os.pathsep + env['PATH']
    run('binding-rust', ['cargo', 'test', '-j2', '--manifest-path', 'crates/ferro-py/Cargo.toml', '--lib'])
    after = sources()
    (output / 'sources-after.json').write_text(json.dumps(after, indent=2))
    if before != after:
        raise AssertionError('Sources changed during verification')
    for name, expected in wheel_hashes.items():
        if hashlib.sha256((package / name.removeprefix('ferro/')).read_bytes()).hexdigest() != expected:
            raise AssertionError('Installed package changed: ' + name)
    (output / 'stable.json').write_text(json.dumps({'sources': len(before), 'unchanged': True}))
    print('All selected CPU/CUDA gates passed; source and installed identities unchanged.', flush=True)


if __name__ == '__main__':
    main()
