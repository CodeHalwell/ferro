from pathlib import Path
import subprocess,json,re,hashlib
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent; FINAL=OUT/'final'
def load(p): return json.loads(p.read_text())
def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
rows=[]
for phase,manifest in [('unchanged',FINAL/'required-unchanged/integration/integration-checks/integration-results.json'),('wheel',FINAL/'wheel-integration/integration-results.json')]:
    batch=load(manifest); assert len(batch)==13
    rows += [dict(phase=phase,**r) for r in batch]
for r in load(FINAL/'required-unchanged/all-results.json'):
    if r['phase']!='integration': rows.append(dict(name=r['phase'],**r,log=str((FINAL/'required-unchanged'/r['phase']/(r['phase']+'.log')).relative_to(ROOT))))
rows += [r for r in load(FINAL/'results.json') if r['name'] not in ('required-unchanged','wheel-integration')]
seen=set()
for r in rows:
    p=(ROOT/r['log']).resolve(); assert p not in seen;seen.add(p)
    text=p.read_text(encoding='utf-8',errors='replace');harnesses=[];nested=[]
    for section in re.split(r'(?m)^\s*(?:Running |Doc-tests )',text):
        found=re.findall(r'test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out',section)
        if found: harnesses.append(found[-1]);nested.extend(found[:-1])
    r['rust_counts']=dict(zip(('passed','failed','ignored','measured','filtered'),[sum(int(m[i]) for m in harnesses) for i in range(5)]))
    r['harnesses']=len(harnesses);r['nested_excluded']=nested
    r['python_tests_run']=sum(map(int,re.findall(r'Ran (\d+) tests? in',text)))
    r['python_skipped']=re.findall(r'OK \(skipped=(\d+)\)',text)
    r['log_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
summary=dict(leaf_commands=len(rows),nonzero=[r['name'] for r in rows if r['exit_code']],rust_test_executions=sum(r['rust_counts']['passed']+r['rust_counts']['failed'] for r in rows),python_test_executions=sum(r['python_tests_run'] for r in rows),source_stability=load(FINAL/'source-stability.json'),packaging_matches=load(FINAL/'packaging.json')['matching'],note='Executions, not unique tests. Wrapper and copied logs excluded; final Rust summary per harness used; nested child summaries retained separately.')
def installed(p):
    text=p.read_text(); return json.loads('\n'.join(text.splitlines()[2:]))
summary['installed_stable']=installed(FINAL/'wheel-installed-before.log')==installed(FINAL/'wheel-installed-after.log')
save('leaf-results-counts.json',rows);save('summary.json',summary)
modified=subprocess.check_output(['git','diff','--name-only','HEAD','-z'],cwd=ROOT).decode().split('\0')
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
selected=[]; excluded=[]
exact_docs={'docs/FOUNDATIONS_AND_ARCHITECTURE_ROADMAP.md','docs/PYTHON_API_DESIGN.md'}
evidence={'verification/python-api-contract/contract.py','verification/python-architecture-api/test_architecture.py','verification/python-architecture-api/training_examples.py','verification/python-architecture-api/README.md','verification/foundation-wave/second-review/STATUS.md','verification/foundation-wave/fastcpu-second-review/README.md','verification/foundation-wave/review-fixes/fastcpu/README.md'}
for p in sorted(set(filter(None,modified+untracked))):
    path=ROOT/p
    if not path.is_file(): continue
    if p in modified: reason='changed production/config/test relative to HEAD'; include=True
    elif p.startswith(('crates/ferro-core/','crates/ferro-fastcpu/','crates/ferro-py/')) and path.suffix in {'.rs','.py','.toml'}: reason='new foundation production/test';include=True
    elif p in exact_docs|evidence: reason='scope/contracts/architecture evidence or unresolved incident disclosure';include=True
    elif p.startswith('verification/foundation-wave/pr-readiness/') and (path.name in {'README.md','PR_BODY.md','verify.py','collect.py','tested-source-provenance.json','security-scan.json','summary.json','leaf-results-counts.json','proposed-files.json','inventory.json'} or path.name in {'packaging.json','sources-before.json','sources-after.json','source-stability.json','results.json','source-exclusions.json'}): reason='fresh portable verification manifest/report (raw logs intentionally not proposed)';include=True
    else: reason='archived/generated evidence, binary, unrelated prior-wave artifact, or not accepted frozen deliverable';include=False
    entry=dict(path=p,status='modified' if p in modified else 'untracked',reason=reason,sha256=None if path.name in {'inventory.json','proposed-files.json'} else hashlib.sha256(path.read_bytes()).hexdigest())
    (selected if include else excluded).append(entry)
# Never stage automatically. Inventory is a review proposal, not security certification.
save('proposed-files.json',selected);save('inventory.json',dict(proposed=selected,excluded=excluded,all_untracked_count=len(set(filter(None,untracked))),warning='No git add executed. Parent must inspect this exact allowlist and sensitive content before staging. No archived logs, wheels, executables, caches, or credentials are proposed.'))
print(json.dumps(summary,indent=2))
for r in rows: print(r.get('phase','supplement'),r['name'],r['exit_code'],r['rust_counts'],r['python_tests_run'])
print('proposed',len(selected),'excluded',len(excluded))
