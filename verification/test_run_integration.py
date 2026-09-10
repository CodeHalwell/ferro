"""Interpreter selection uses real temporary venv layouts, without launching jobs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("run_integration", Path(__file__).with_name("run_integration.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(runner, "select_python", None)),
                        "runner needs a platform-aware repo-venv selector")

    def test_prefers_existing_posix_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "crates/ferro-py/.venv/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.touch()
            self.assertEqual(runner.select_python(root, "posix", "fallback"), str(candidate))

    def test_prefers_existing_windows_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "crates/ferro-py/.venv/Scripts/python.exe"
            candidate.parent.mkdir(parents=True)
            candidate.touch()
            self.assertEqual(runner.select_python(root, "nt", "fallback"), str(candidate))

    def test_missing_venv_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            for platform in ("posix", "nt"):
                self.assertEqual(runner.select_python(Path(tmp), platform, "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
