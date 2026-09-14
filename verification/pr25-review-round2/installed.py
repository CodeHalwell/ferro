import hashlib
import json
from pathlib import Path
import zipfile
import ferro._native as native
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
wheel = next((out / 'wheels').glob('*.whl'))
installed = Path(native.__file__)
with zipfile.ZipFile(wheel) as z:
    entry = next(n for n in z.namelist() if n.endswith('.pyd'))
    wheel_native = hashlib.sha256(z.read(entry)).hexdigest()
record = {'installed': str(installed), 'native_sha256': hashlib.sha256(installed.read_bytes()).hexdigest(),
          'wheel_native_sha256': wheel_native, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
paths = ['crates/ferro-py/src/lib.rs', 'crates/ferro-py/src/convolution.rs',
         'crates/ferro-py/tests/test_pr25_copy.py', 'crates/ferro-py/tests/test_pr25_convolution.py']
paths += ['crates/ferro-core/src/ops_ext/' + n + '.rs' for n in
          ['bce_with_logits_loss', 'huber_loss', 'smooth_l1_loss', 'index_select', 'cat']]
record['source_sha256'] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
assert record['native_sha256'] == wheel_native
(out / 'installed.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
