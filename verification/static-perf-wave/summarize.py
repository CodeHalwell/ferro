"""Summarize saved real measurements; no generated performance fixtures."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
names=['baseline1','baseline2','layout1','guards1','final1','final2']
reports={name:json.loads((OUT/(name+'.json')).read_text()) for name in names}
rows={name:{(r['case'],r['mode']):r for r in report['rows']} for name,report in reports.items()}
comparisons=[]
for model in ['mlp','residual_mlp','transformer']:
    for mode in ['ferro_static','ferro_static_snapshot']:
        for order in [1,2]:
            before=rows[f'baseline{order}'][model,mode]['median_us']
            after=rows[f'final{order}'][model,mode]['median_us']
            comparisons.append(dict(model=model,mode=mode,order=order,baseline_us=before,final_us=after,reduction_percent=100*(before-after)/before))
(OUT/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
files=[ROOT/'crates/ferro-cuda/src/static_graph.rs',ROOT/'crates/ferro-cuda/src/static_graph/model.rs',ROOT/'verification/model-cuda-graphs/continuation/benchmark.py',ROOT/'verification/model-cuda-graphs/continuation/verify_results.py']
files+=list((ROOT/'crates/ferro-py/.venv/Lib/site-packages/ferro').glob('*.pyd'))
metadata=dict(platform=platform.platform(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_and_binding_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},evidence_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for name in names for p in [OUT/(name+'.json')]})
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
