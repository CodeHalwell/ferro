import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runs = {name: json.loads((ROOT / f'{name}.json').read_text()) for name in ['baseline1', 'baseline2', 'post1', 'post2']}
summary = {}
for name, run in runs.items():
    summary[name] = []
    for row in run['rows']:
        if row['scope'] == 'full_model' or row.get('component') in ('ln1', 'ln2'):
            summary[name].append({k: row[k] for k in ['scope', 'component', 'backend', 'status', 'structure', 'max_abs_errors', 'error'] if k in row} | ({'median_us': row['timing']['total']['median_us']} if 'timing' in row else {}))
    assert all(row['status'] == 'passed' for row in run['rows'] if row['scope'] == 'full_model' and row['backend'] != 'torch_compile')
comparisons = {}
for backend in ['ferro_eager', 'ferro_compiled', 'torch_eager']:
    values = {name: next(r['timing']['total']['median_us'] for r in run['rows'] if r['scope'] == 'full_model' and r['backend'] == backend) for name, run in runs.items()}
    comparisons[backend] = values | {'paired_speedups': [values[f'baseline{i}']/values[f'post{i}'] for i in (1,2)], 'paired_reduction_percent': [100*(1-values[f'post{i}']/values[f'baseline{i}']) for i in (1,2)]}
result = {'runs': summary, 'full_model_comparison': comparisons}
(ROOT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
