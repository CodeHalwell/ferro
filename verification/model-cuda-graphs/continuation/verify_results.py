"""Verify final evidence counts, freshness gates and median calculations."""
import json
import re
import statistics
from pathlib import Path
P=Path(__file__).resolve().parent
integration=json.loads((P/'integration-results.json').read_text())
assert len(integration)==13 and all(r['exit_code']==0 for r in integration)
passes={name:sum(map(int,re.findall(r'test result: ok\. (\d+) passed',(P/(name+'.log')).read_text()))) for name in ['core','cpu-tokenizer','cuda','capture-layout-gpu']}
reports=[json.loads((P/f'run{i}.json').read_text()) for i in [3,4]]
assert reports[0]['modes']==reports[1]['modes'][::-1]
summary=dict(integration_commands=len(integration),rust_passes=passes,runs=[])
for number,report in zip([3,4],reports):
    assert len(report['rows'])==21
    keys={(r['case'],r['mode']) for r in report['rows']}
    assert len(keys)==21
    passed=[]
    blocked=[]
    for row in report['rows']:
        if row['status']=='passed':
            assert len(row['samples_us'])==40
            assert row['median_us']==statistics.median(row['samples_us'])
            assert len(row['freshness_max_abs_errors'])==4
            passed.append({k:row[k] for k in ['case','mode','median_us','prepare_ms','freshness_max_abs_errors']})
        else:
            assert row['status']=='blocked' and 'samples_us' not in row
            assert 'Cannot find a working triton installation' in row['error']
            blocked.append(dict(case=row['case'],mode=row['mode'],reason='Triton unavailable'))
    assert len(passed)==15 and len(blocked)==6
    summary['runs'].append(dict(run=number,passed=passed,blocked=blocked))
(P/'verified-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(integration_commands=13,rust_passes=passes,benchmark_runs=2,passed_per_run=15,blocked_per_run=6,samples_per_passed_row=40)))
