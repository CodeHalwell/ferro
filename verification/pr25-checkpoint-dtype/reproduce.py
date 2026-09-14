"""Run current typed regressions against the exact reviewed core in isolation."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HEAD = 'f7ed528db68c8dfab9236d40badf0779b277439e'
archive = subprocess.check_output(['git', 'archive', '--format=zip', HEAD, 'crates/ferro-core'], cwd=ROOT)
with tempfile.TemporaryDirectory(prefix='ferro-pr25-dtype-') as tmp:
    root = Path(tmp)
    zipfile.ZipFile(io.BytesIO(archive)).extractall(root)
    (root / 'Cargo.toml').write_text('[workspace]\nmembers = ["crates/ferro-core"]\nresolver = "2"\n[workspace.package]\nversion = "0.0.1"\nedition = "2021"\nlicense = "MIT"\n')
    test = Path('crates/ferro-core/tests/checkpoint_dtype.rs')
    (root / test).write_bytes((ROOT / test).read_bytes())
    cmd = ['cargo', 'test', '-p', 'ferro-core', '-j2', '--test', 'checkpoint_dtype']
    run = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'reviewed-head-red.log').write_bytes(run.stdout)
    result = {'reviewed_head': HEAD, 'command': cmd, 'exit_code': run.returncode,
              'source': 'git archive of reviewed head, plus current regression file'}
    (OUT / 'reviewed-head-red.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    assert run.returncode != 0, 'reviewed source unexpectedly passed regressions'
    text = run.stdout.decode(errors='replace')
    for name in ['f64_checkpoint_roundtrip', 'i64_checkpoint_roundtrip', 'f16_checkpoint_roundtrip', 'bf16_checkpoint_roundtrip', 'named_snapshot_returns_device_copy_errors']:
        assert f'test {name} ... FAILED' in text, text
