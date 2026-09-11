"""Execute the unchanged integration runner's command matrix into scoped logs."""
import importlib.util
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
spec = importlib.util.spec_from_file_location('integration', ROOT / 'verification/run_integration.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
results = []
for name, command in runner.COMMANDS + [('capture-layout-gpu', ['cargo', 'test', '-j2', '--manifest-path', 'verification/capture-layout-gpu/Cargo.toml', '--', '--test-threads=1'])]:
    log = OUT / (name + '.log')
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=name, command=command, exit_code=result.returncode, log=str(log.relative_to(ROOT)))
    results.append(row)
    print(json.dumps(row), flush=True)
    (OUT / 'integration-results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
assert len(runner.COMMANDS) == 12
raise SystemExit(int(any(row['exit_code'] for row in results)))
