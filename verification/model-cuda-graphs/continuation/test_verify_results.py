"""CPU-only regression tests; no framework imports or GPU initialization."""
import importlib.util
import json
import subprocess
import tempfile
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

P = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, P / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    # Framework imports are forbidden in this CPU-only test process.
    with patch.dict(sys.modules, {'torch': None, 'ferro': None}):
        spec.loader.exec_module(module)
    return module


class ClassificationTests(unittest.TestCase):
    def test_only_explicit_missing_triton_is_blocked(self):
        benchmark = load('benchmark')
        for message in ['Cannot find a working triton installation. Either the package is not installed or it is too old.', "No module named 'triton'", 'No module named "triton"']:
            for mode in ['torch_inductor_no_graphs', 'torch_reduce_overhead']:
                self.assertEqual(benchmark.error_status(mode, RuntimeError(message)), 'blocked')
            self.assertEqual(benchmark.error_status('torch_eager', RuntimeError(message)), 'failed')
            self.assertEqual(benchmark.error_status('ferro_compiled', RuntimeError(message)), 'failed')
        for message in ['BackendCompilerFailed: name models is not defined', 'triton compiler internal error', 'CUDA out of memory', 'illegal memory access', "No module named 'triton.ops'", 'unknown compiler error']:
            self.assertEqual(benchmark.error_status('torch_reduce_overhead', RuntimeError(message)), 'failed')


def fixture(reverse=False):
    modes = load('benchmark').MODES
    return dict(modes=modes[::-1] if reverse else modes, samples=40, warmup=10,
                parity_rtol=2e-4, parity_atol=2e-5, rows=[
        dict(case=case, mode=mode, status='passed', samples_us=[1.0]*40,
             median_us=1.0, min_us=1.0, max_us=1.0, prepare_ms=0.0,
             freshness_max_abs_errors=[1e-6]*4,
             freshness_reference_max_abs=[1.0]*4,
             freshness_max_tolerance_ratios=[0.05]*4)
        for case in ['mlp', 'residual_mlp', 'transformer']
        for mode in (modes[::-1] if reverse else modes)])


class ValidationTests(unittest.TestCase):
    def test_freshness_rejects_invalid_values_at_every_gate(self):
        verifier = load('verify_results')
        for field in ['freshness_max_abs_errors', 'freshness_reference_max_abs', 'freshness_max_tolerance_ratios']:
            for index in range(4):
                values = [-1.0, float('nan'), float('inf')]
                if field != 'freshness_reference_max_abs':
                    values += [1e9]
                for value in values:
                    with self.subTest(field=field, index=index, value=value):
                        report = fixture()
                        report['rows'][0][field][index] = value
                        with self.assertRaises(ValueError):
                            verifier.validate_report(report)

    def test_rejects_incomplete_rows_and_nonfinite_timings(self):
        verifier = load('verify_results')
        mutations = [
            lambda r: r['rows'].__setitem__(0, copy.deepcopy(r['rows'][1])),
            lambda r: r['rows'][0].update(case='unknown'),
            lambda r: r['rows'][0].update(mode='unknown'),
            lambda r: r.update(modes=['unknown']*7),
            lambda r: r.update(samples=39),
            lambda r: r.update(warmup=0),
            lambda r: r['rows'][0].update(samples_us=[1.0]*39),
            lambda r: r['rows'][0].update(samples_us=[1.0]*39+[float('nan')]),
            lambda r: r['rows'][0].update(samples_us=[1.0]*39+[float('inf')]),
            lambda r: r['rows'][0].update(samples_us=[1.0]*39+[-1.0]),
            lambda r: r['rows'][0].update(median_us=float('inf'), samples_us=[float('inf')]*40),
            lambda r: r['rows'][0].update(min_us=-1),
            lambda r: r['rows'][0].update(max_us=float('inf')),
            lambda r: r['rows'][0].update(prepare_ms=float('nan')),
        ]
        for index, mutate in enumerate(mutations):
            report = fixture()
            mutate(report)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                verifier.validate_report(report)

    def test_failed_unknown_and_spurious_blocked_statuses_rejected(self):
        verifier = load('verify_results')
        for status, mode, error in [('failed', 'torch_reduce_overhead', 'CUDA error'), ('unknown', 'torch_reduce_overhead', ''), ('blocked', 'torch_eager', 'Cannot find a working triton installation'), ('blocked', 'torch_reduce_overhead', 'triton compilation internal failure')]:
            report = fixture()
            row = next(r for r in report['rows'] if r['mode'] == mode)
            row.update(status=status, error=error)
            row.pop('samples_us')
            with self.subTest(status=status, mode=mode), self.assertRaises(ValueError):
                verifier.validate_report(report)

    def test_missing_triton_rows_and_all_compilers_working_are_valid(self):
        verifier = load('verify_results')
        report = fixture()
        self.assertEqual(len(verifier.validate_report(report)['passed']), 21)
        for row in report['rows']:
            if row['mode'] in ['torch_inductor_no_graphs', 'torch_reduce_overhead']:
                case, mode = row['case'], row['mode']
                row.clear()
                row.update(case=case, mode=mode, status='blocked', error='Cannot find a working triton installation')
        result = verifier.validate_report(report)
        self.assertEqual((len(result['passed']), len(result['blocked'])), (15, 6))

    def test_declared_tolerance_cannot_be_weakened(self):
        verifier = load('verify_results')
        for field, value in [('parity_rtol', 1.0), ('parity_atol', 1.0), ('parity_rtol', float('nan'))]:
            report = fixture()
            report[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                verifier.validate_report(report)


class FreshPathTests(unittest.TestCase):
    def test_explicit_paths_not_archived_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Copy only the verifier and its CPU-only dependency: no archive exists.
            for name in ['verify_results.py', 'benchmark.py']:
                (root/name).write_bytes((P/name).read_bytes())
            for number in [1, 2]:
                (root/f'fresh{number}.json').write_text(json.dumps(fixture(number == 2)))
            command = [sys.executable, str(root/'verify_results.py'), '--runs', 'fresh1.json', 'fresh2.json', '--summary', 'fresh-summary.json']
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'fresh-summary.json').read_text())
            self.assertEqual([r['input'] for r in summary['runs']], [str((root/f'fresh{i}.json').resolve()) for i in [1, 2]])
            (root/'fresh-summary.json').unlink()
            (root/'fresh1.json').write_text('{}')
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, 'must validate requested fresh file, not cached evidence')


if __name__ == '__main__':
    unittest.main()
