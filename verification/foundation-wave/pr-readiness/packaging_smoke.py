"""Test the packaged Python facade against the verified installed native binary."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import ferro
import ferro._native as native

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'crates/ferro-py/python/ferro'
NATIVE = Path(native.__file__).resolve()
INSTALLED = Path(ferro.__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_hashes():
    return {str(p.relative_to(SOURCE)): digest(p) for p in SOURCE.rglob('*.py')}

before = source_hashes()
native_hash = digest(NATIVE)
assert native_hash == '8db7f4c8fa6fb60e9b38a987a756d54a2f5138815b20e0409c1ec873ab431358', 'Install the verified native wheel first'
source_text = (SOURCE / 'nn/functional.py').read_text()
installed_text = (INSTALLED / 'nn/functional.py').read_text()
assert source_text.rstrip() == installed_text.rstrip(), 'Only EOF whitespace may differ'
assert ast.dump(ast.parse(source_text)) == ast.dump(ast.parse(installed_text))
print(json.dumps({'installed_package': str(INSTALLED), 'native': str(NATIVE), 'native_sha256': native_hash, 'facade_files': len(before), 'functional_ast_unchanged': True}), flush=True)
with tempfile.TemporaryDirectory(prefix='ferro-publication-') as directory:
    package = Path(directory) / 'ferro'
    shutil.copytree(SOURCE, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyd', '*.so'))
    shutil.copy2(NATIVE, package / NATIVE.name)
    env = dict(os.environ, PYTHONPATH=directory)
    check = "import ferro,ferro._native as n,pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); assert pathlib.Path(ferro.__file__).resolve().parent==root; assert pathlib.Path(n.__file__).resolve().parent==root; print('imports verified:',ferro.__file__,n.__file__)"
    commands = [
        [sys.executable, '-c', check, str(package)],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'crates/ferro-py/tests', '-v'],
        [sys.executable, 'verification/python-api-contract/contract.py'],
        [sys.executable, 'verification/python-architecture-api/test_architecture.py'],
    ]
    for command in commands:
        print('COMMAND:', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
assert source_hashes() == before
assert digest(NATIVE) == native_hash
print('Source facade and native identities stable; no shared environment install/rebuild')
