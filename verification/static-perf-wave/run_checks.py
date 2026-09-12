"""Current-tree serial integration; requires a rebuilt binding and CUDA runtime."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory, never an archived evidence directory')
    parser.add_argument('--runtime-root', type=Path, help='nvidia directory containing cuda_nvrtc/bin and cublas/bin')
    parser.add_argument('--plan', action='store_true', help='validate directory prerequisites and print commands; NOT runtime verification')
    args = parser.parse_args()
    env = {**os.environ, 'RUST_TEST_THREADS': '1', 'FERRO_REQUIRE_CUDA': '1'}
    runtime = args.runtime_root
    if runtime is None and os.name == 'nt':
        if not env.get('LOCALAPPDATA'):
            parser.error('LOCALAPPDATA missing; supply --runtime-root')
        runtime = Path(env['LOCALAPPDATA']) / 'Temp/cuda-rt/nvidia'
    prefix = [] if runtime is None else [runtime.resolve() / 'cuda_nvrtc/bin', runtime.resolve() / 'cublas/bin']
    for directory in prefix:
        if not directory.is_dir():
            parser.error(f'CUDA runtime directory missing: {directory}')
    env['PATH'] = os.pathsep.join(map(str, prefix + [env.get('PATH', '')]))
    spec = importlib.util.spec_from_file_location('integration', ROOT / 'verification/run_integration.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    commands = runner.COMMANDS + [('capture-layout-gpu', ['cargo', 'test', '-j2', '--manifest-path', 'verification/capture-layout-gpu/Cargo.toml', '--', '--test-threads=1'])]
    if args.plan:
        print(json.dumps(dict(cwd=str(ROOT), runtime_prefix=list(map(str, prefix)), required_cuda=env['FERRO_REQUIRE_CUDA'], commands=commands, runtime_verified=False)))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for name, command in commands:
        log = args.output_dir.resolve() / (name + '.log')
        with log.open('x', encoding='utf-8') as stream:
            stream.write('COMMAND: ' + subprocess.list2cmdline(command) + '\nFERRO_REQUIRE_CUDA=1; RUST_TEST_THREADS=1\nCUDA runtime prefix: ' + repr(list(map(str, prefix))) + '\n')
            stream.flush()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(name=name, command=command, exit_code=result.returncode, log=os.path.relpath(log, ROOT))
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / 'integration-results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return int(any(row['exit_code'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
