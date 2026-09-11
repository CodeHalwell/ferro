"""CPU-only CLI regressions; torch and ferro imports are explicitly blocked."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from validate import cases


class EpsilonTests(unittest.TestCase):
    def run_cli(self, epsilons, output):
        # Blocking imports makes a validation regression safe even before the fix.
        script = Path(__file__).with_name("validate.py")
        bootstrap = ("import runpy, sys; "
                     "sys.modules['torch'] = None; sys.modules['ferro'] = None; "
                     "sys.path.insert(0, sys.argv[1]); "
                     "sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')")
        return subprocess.run([sys.executable, "-W", "error", "-c", bootstrap, str(script.parent), str(script),
                               "--device", "cpu", "--eps", *epsilons, "--output", str(output)],
                              capture_output=True, text=True, timeout=30)

    def test_cli_rejects_eps_that_rounds_to_zero_or_infinity(self):
        for eps in ("1e-50", "1e40"):
            with self.subTest(eps=eps), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results.jsonl"
                output.write_text("preserve existing report\n", encoding="utf-8")
                result = self.run_cli(["1e-5", eps], output)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("eps", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(output.read_text(encoding="utf-8"), "preserve existing report\n")

    def test_cli_preserves_rejection_of_nonpositive_or_nonfinite_eps(self):
        for eps in ("0", "-1", "nan", "inf"):
            with self.subTest(eps=eps), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results.jsonl"
                result = self.run_cli([eps], output)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_valid_f32_eps_reaches_import_boundary_and_rounds_in_cases(self):
        values = [1e-5, 1e-3, float(np.finfo(np.float32).smallest_subnormal),
                  float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            result = self.run_cli([str(e) for e in values], output)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("import of torch halted; None in sys.modules", result.stderr)
            self.assertFalse(output.exists())
        generated = list(cases([17], [1], values, 1))
        self.assertEqual({meta["eps"] for meta, _ in generated}, {float(np.float32(e)) for e in values})
        self.assertTrue(all(np.isfinite(meta["eps"]) and meta["eps"] > 0 for meta, _ in generated))


if __name__ == "__main__":
    unittest.main()
