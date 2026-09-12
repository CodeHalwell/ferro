"""Compare explicit saved runs; current provenance is not historical build proof."""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('baseline1', 'baseline2', 'layout1', 'guards1', 'final1', 'final2')


def resolve_binding(explicit=None):
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise ValueError(f'binding does not exist: {path}')
        candidates = [path]
    else:
        spec = importlib.util.find_spec('ferro')
        if spec is None:
            raise ValueError('active ferro binding does not exist; use its interpreter or --binding')
        candidates = [Path(spec.origin)] if spec.origin else []
        for directory in spec.submodule_search_locations or []:
            candidates.extend(Path(directory).rglob('*'))
    extensions = sorted({p.resolve() for p in candidates if p.is_file() and any(p.name.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)})
    if len(extensions) != 1:
        raise ValueError(f'expected exactly one native ferro binding, found {len(extensions)}; use --binding')
    return extensions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='directory containing the six named stage reports')
    parser.add_argument('--output-dir', type=Path, required=True, help='new directory; existing paths are never overwritten')
    parser.add_argument('--binding', type=Path, help='explicit native extension; otherwise discover in this interpreter')
    args = parser.parse_args()
    try:
        binding = resolve_binding(args.binding)
    except ValueError as error:
        parser.error(str(error))
    reports = {name: json.loads((args.input_dir / (name + '.json')).read_text()) for name in NAMES}
    rows = {name: {(r['case'], r['mode']): r for r in report['rows']} for name, report in reports.items()}
    comparisons = []
    for model in ('mlp', 'residual_mlp', 'transformer'):
        for mode in ('ferro_static', 'ferro_static_snapshot'):
            for order in (1, 2):
                before = rows[f'baseline{order}'][model, mode]['median_us']
                after = rows[f'final{order}'][model, mode]['median_us']
                comparisons.append(dict(model=model, mode=mode, order=order, baseline_us=before, final_us=after, reduction_percent=100*(before-after)/before))
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', 'crates', 'bench', 'verification', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain', 'rust-toolchain.toml', '.cargo'], cwd=ROOT).decode().split('\0')
    files = sorted({ROOT / name for name in tracked if name and (ROOT / name).is_file() and (not name.startswith('verification/') or name.endswith(('.py', '.toml', '.lock', '.rs')))})
    if ROOT / 'crates/ferro-core/src/graph.rs' not in files:
        raise ValueError('required core lowering source is missing')
    metadata = dict(provenance_scope='current source and selected binding at summarization; NOT proof of the source/binary used by saved runs', platform=platform.platform(), python=sys.executable, head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, binding=dict(path=str(binding), selection='explicit' if args.binding else 'active interpreter discovery', sha256=hashlib.sha256(binding.read_bytes()).hexdigest()), evidence_sha256={name + '.json': hashlib.sha256((args.input_dir / (name + '.json')).read_bytes()).hexdigest() for name in NAMES})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
