"""Collect current crate sources without executing checks or changing archived evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True, help='new directory; existing paths are refused')
parser.add_argument('--evidence-dir', type=Path, help='optional read-only log/JSON inputs, copied separately without asserting source/run identity')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
changed = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--', 'crates'], cwd=ROOT).decode().strip('\0').split('\0')
extra = subprocess.check_output(['git', 'ls-files', '-z', '--others', '--exclude-standard', '--', 'crates'], cwd=ROOT).decode().strip('\0').split('\0')
paths = sorted(set(changed + extra) - {''})
hashes = {}
deleted = []
for name in paths:
    source = ROOT / name
    if not source.is_file():
        deleted.append(name)
        continue
    target = OUT / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', 'crates'], cwd=ROOT)
for name in sorted(set(extra) - {''}):
    if not (ROOT / name).is_file():
        continue
    try:
        addition = subprocess.check_output(['git', 'diff', '--no-index', '--binary', '--', os.devnull, name], cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        if error.returncode != 1:
            raise
        addition = error.output
    patch += addition
(OUT / 'combined-wave.patch').write_bytes(patch)
(OUT / 'sources-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
evidence = {}
if args.evidence_dir:
    destination = OUT / 'copied-evidence'
    destination.mkdir()
    for source in sorted(args.evidence_dir.resolve().iterdir()):
        if source.is_file() and source.suffix in ('.json', '.log'):
            target = destination / source.name
            shutil.copy2(source, target)
            evidence[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
metadata = dict(scope='current source snapshot; copied evidence is historical/unattested against this source, not a new test run', head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_count=len(hashes), deleted_paths=deleted, copied_evidence_sha256=evidence, checks_executed=False)
(OUT / 'collection.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata))
