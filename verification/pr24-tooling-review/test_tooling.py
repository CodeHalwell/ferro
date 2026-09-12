"""CPU-only fixture checks. No binding import, CUDA calls, or builds."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ToolingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        self.cwd = self.base / 'unrelated'
        self.cwd.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        for name in ('static-perf-wave/summarize.py', 'static-perf-wave/run_checks.py', 'static-empty-fix/collect.py', 'run_integration.py'):
            dest = self.root / 'verification' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'verification' / name, dest)
        self.inputs = self.base / 'inputs'
        self.inputs.mkdir()
        rows = [dict(case=model, mode=mode, median_us=10) for model in ('mlp', 'residual_mlp', 'transformer') for mode in ('ferro_static', 'ferro_static_snapshot')]
        for name in ('baseline1', 'baseline2', 'layout1', 'guards1', 'final1', 'final2'):
            (self.inputs / (name + '.json')).write_text(json.dumps(dict(rows=rows)))
            shutil.copy2(self.inputs / (name + '.json'), self.root / 'verification/static-perf-wave' / (name + '.json'))
        for name in ('Cargo.toml', 'Cargo.lock', 'crates/ferro-core/src/graph.rs', 'crates/ferro-cuda/src/lib.rs', 'crates/ferro-cuda/src/static_graph.rs', 'crates/ferro-cuda/src/static_graph/model.rs', 'verification/model-cuda-graphs/continuation/benchmark.py', 'verification/model-cuda-graphs/continuation/verify_results.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture source\n')
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], cwd=self.root, check=True, capture_output=True)

    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(self.root / 'verification' / name), *map(str, args)], cwd=self.cwd, capture_output=True, text=True)

    def test_missing_binding_fails_without_emitting_metadata(self):
        out = self.base / 'output'
        result = self.run_script('static-perf-wave/summarize.py', '--input-dir', self.inputs, '--output-dir', out, '--binding', self.base / 'missing.pyd')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('binding does not exist', result.stderr.lower())
        self.assertFalse((out / 'metadata.json').exists())
        self.assertFalse((self.root / 'verification/static-perf-wave/metadata.json').exists())

    def test_summary_from_unrelated_cwd_hashes_untracked_production_and_core(self):
        source = self.root / 'crates/ferro-cuda/src/static_graph/native.rs'
        source.write_text('new production dependency\n')
        binding = self.base / ('ferro' + importlib.machinery.EXTENSION_SUFFIXES[0])
        binding.write_bytes(b'CPU fixture only; never loaded')
        out = self.base / 'output'
        result = self.run_script('static-perf-wave/summarize.py', '--input-dir', self.inputs, '--output-dir', out, '--binding', binding)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((out / 'metadata.json').read_text())
        for name in ('crates/ferro-core/src/graph.rs', 'crates/ferro-cuda/src/static_graph/native.rs', 'Cargo.toml', 'Cargo.lock'):
            self.assertIn(name, metadata['source_sha256'])
        self.assertIn('NOT proof', metadata['provenance_scope'])
        self.assertEqual(metadata['binding']['path'], str(binding.resolve()))
        self.assertFalse((self.root / 'verification/static-perf-wave/metadata.json').exists())
        before = (out / 'metadata.json').read_bytes()
        again = self.run_script('static-perf-wave/summarize.py', '--input-dir', self.inputs, '--output-dir', out, '--binding', binding)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(before, (out / 'metadata.json').read_bytes())

    def test_collector_includes_untracked_production(self):
        import ast
        source = self.root / 'crates/ferro-cuda/src/static_graph/native.rs'
        source.write_text('untracked owner\n')
        tree = ast.parse((self.root / 'verification/static-empty-fix/collect.py').read_text())
        assignments = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('changed', 'extra', 'paths') for t in n.targets)]
        namespace = dict(ROOT=self.root, subprocess=subprocess)
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<collector-path-selection>', 'exec'), namespace)
        self.assertIn(source.relative_to(self.root).as_posix(), namespace['paths'])

    def test_runtime_prerequisites_fail_before_commands(self):
        # --plan validates paths without loading DLLs or launching tests.
        import contextlib
        import io
        import runpy
        from unittest.mock import patch
        errors = io.StringIO()
        with patch.object(sys, 'argv', ['run_checks.py', '--plan', '--runtime-root', str(self.base / 'missing-runtime'), '--output-dir', str(self.base / 'checks')]), patch.object(subprocess, 'run', side_effect=RuntimeError('attempted command before prerequisite validation')), contextlib.redirect_stderr(errors):
            try:
                runpy.run_path(str(self.root / 'verification/static-perf-wave/run_checks.py'), run_name='__main__')
            except (SystemExit, RuntimeError):
                pass
        self.assertIn('CUDA runtime directory missing', errors.getvalue())
        self.assertFalse((self.root / 'verification/static-perf-wave/core.log').exists())

    def test_collector_archives_untracked_file_and_patch_from_other_cwd(self):
        # Block subprocess.run until the collector is collection-only.
        import ast
        tree = ast.parse((self.root / 'verification/static-empty-fix/collect.py').read_text())
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'subprocess' and n.func.attr == 'run' for n in ast.walk(tree)), 'collection must not run tests or builds')
        source = self.root / 'crates/ferro-cuda/src/static_graph/new owner.rs'
        source.write_text('untracked owner\n')
        out = self.base / 'archive'
        result = self.run_script('static-empty-fix/collect.py', '--output-dir', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(source.read_bytes(), (out / 'sources' / source.relative_to(self.root)).read_bytes())
        self.assertIn('crates/ferro-cuda/src/static_graph/new owner.rs', json.loads((out / 'sources-sha256.json').read_text()))
        self.assertIn('untracked owner', (out / 'combined-wave.patch').read_text())
        self.assertFalse((self.root / 'verification/static-empty-fix/results.json').exists())

    def test_raw_runner_exposes_current_phase_not_historical_red(self):
        import ast
        tree = ast.parse((ROOT / 'verification/dlpack-raw-review/run.py').read_text())
        keys = {k.value for node in ast.walk(tree) if isinstance(node, ast.Dict) for k in node.keys if isinstance(k, ast.Constant)}
        self.assertNotIn('red', keys)
        self.assertNotIn('green', keys)
        self.assertIn('current-test', keys)

    def test_active_binding_missing_and_ambiguous_fail_closed(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location('summary_fixture', self.root / 'verification/static-perf-wave/summarize.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module.importlib.util, 'find_spec', return_value=None):
            with self.assertRaisesRegex(ValueError, 'active ferro binding does not exist'):
                module.resolve_binding()
        package = self.base / 'ferro'
        package.mkdir()
        extension = package / ('ferro' + importlib.machinery.EXTENSION_SUFFIXES[0])
        extension.write_bytes(b'fixture, not executable')
        active = SimpleNamespace(origin=str(package / '__init__.py'), submodule_search_locations=[str(package)])
        with patch.object(module.importlib.util, 'find_spec', return_value=active):
            self.assertEqual(module.resolve_binding(), extension.resolve())
            (package / ('other' + importlib.machinery.EXTENSION_SUFFIXES[0])).write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'exactly one native ferro binding'):
                module.resolve_binding()

    def test_runtime_plan_from_unrelated_cwd_is_not_execution(self):
        runtime = self.base / 'runtime'
        for part in ('cuda_nvrtc/bin', 'cublas/bin'):
            (runtime / part).mkdir(parents=True)
        out = self.base / 'checks'
        result = self.run_script('static-perf-wave/run_checks.py', '--plan', '--runtime-root', runtime, '--output-dir', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['cwd'], str(self.root.resolve()))
        self.assertEqual(plan['required_cuda'], '1')
        self.assertFalse(plan['runtime_verified'])
        self.assertEqual(plan['runtime_prefix'], [str(runtime.resolve() / part) for part in ('cuda_nvrtc/bin', 'cublas/bin')])
        self.assertEqual(len(plan['commands']), 13)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
