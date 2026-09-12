"""Archive exact combined-wave sources and unfiltered verification evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
rows = []
for label, command in [
    ('empty-core', ['cargo', 'test', '-j2', '-p', 'ferro-core', '--test', 'empty_softmax']),
    ('coverage-final', [sys.executable, 'verification/static-coverage-python/run.py', 'empty-fix-final']),
]:
    with (OUT / (label + '.log')).open('w', encoding='utf-8') as log:
        log.write('COMMAND: ' + subprocess.list2cmdline(command) + '\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    rows.append(dict(name=label, command=command, exit_code=result.returncode))

raw = ROOT / 'verification/dlpack-raw-review'
phases = json.loads((raw / 'all-results.json').read_text())
integration = json.loads((raw / 'integration-results.json').read_text())
assert len(phases) == 5 and len(integration) == 13
for p in raw.iterdir():
    if p.suffix in ('.log', '.json') and (p.stem in ('unit', 'build', 'eager', 'snapshot', 'integration', 'all-results', 'integration-results') or p.name.startswith('integration-')):
        shutil.copy2(p, OUT / p.name)
for label in ('empty-fix-red', 'empty-fix-green', 'empty-fix-final'):
    shutil.copy2(ROOT / 'verification/static-coverage-python' / (label + '.log'), OUT / (label + '.log'))

changed = subprocess.check_output(['git', 'diff', '--name-only', '--', 'crates'], cwd=ROOT, text=True).splitlines()
extra = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'crates/ferro-core/tests', 'crates/ferro-py/tests'], cwd=ROOT, text=True).splitlines()
paths = sorted(set(changed + extra + ['crates/ferro-py/tests/test_static_snapshot_dlpack.py']))
hashes = {}
for name in paths:
    source = ROOT / name
    target = OUT / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
(OUT / 'sources-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
(OUT / 'combined-wave.patch').write_bytes(subprocess.check_output(['git', 'diff', '--', 'crates'], cwd=ROOT))
(OUT / 'results.json').write_text(json.dumps(dict(targeted=rows, phases=phases, integration=integration), indent=2) + '\n')
assert all(row['exit_code'] == 0 for row in rows + phases + integration)
print(json.dumps(dict(targeted=rows, phase_count=len(phases), integration_count=len(integration), source_count=len(hashes), all_passed=True)))
