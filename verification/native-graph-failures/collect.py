"""Archive and summarize actual final verification; no synthetic test results."""
import json
from pathlib import Path
import re
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'verification/dlpack-raw-review'
phases = json.loads((SOURCE / 'all-results.json').read_text())
integration = json.loads((SOURCE / 'integration-results.json').read_text())
assert len(phases) == 5 and len(integration) == 13
assert all(row['exit_code'] == 0 for row in phases + integration)
for row in phases:
    shutil.copy2(SOURCE / (row['phase'] + '.log'), OUT / ('final-' + row['phase'] + '.log'))
for row in integration:
    shutil.copy2(ROOT / row['log'], OUT / ('final-integration-' + row['name'] + '.log'))
shutil.copy2(SOURCE / 'all-results.json', OUT / 'all-results.json')
shutil.copy2(SOURCE / 'integration-results.json', OUT / 'integration-results.json')
summaries = {}
for name in ['final-cuda-default-parallel', 'final-unit', 'final-eager', 'final-snapshot', 'final-integration-python-capture', 'final-integration-core', 'final-integration-cpu-tokenizer', 'final-integration-capture-layout-gpu']:
    text = (OUT / (name + '.log')).read_text()
    rust = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    python = re.findall(r'Ran (\d+) tests? in [^\n]+\n\n(OK[^\n]*|FAILED[^\n]*)', text)
    summaries[name] = dict(rust_results=rust, python_results=python)
# A bounded child runs the concurrency test too. Count each harness once,
# retaining the last result within each Running/Doc-tests block.
text = (OUT / 'final-cuda-default-parallel.log').read_text()
blocks = re.split(r'(?m)^\s*(?:Running |Doc-tests )', text)[1:]
counts = [re.findall(r'test result: ok\. (\d+) passed;', block)[-1] for block in blocks]
summaries['cuda_harness_pass_counts'] = list(map(int, counts))
summaries['cuda_harness_pass_total'] = sum(map(int, counts))
(OUT / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
print(json.dumps(summaries, indent=2))
