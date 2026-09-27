"""GPU-free checks of the real prepared-segment setup and discovery contract."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).with_name("test_prepared_segments.py")


def load_isolated():
    # Only import dependencies are replaced; setup and discovery use real source.
    ferro = types.ModuleType("ferro")
    native = types.ModuleType("ferro._native")
    ferro._native = native
    spec = importlib.util.spec_from_file_location("isolated_prepared_segments", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"ferro": ferro, "ferro._native": native}):
        spec.loader.exec_module(module)
    return module, ferro


class PreparedSegmentsSetup(unittest.TestCase):
    def test_setup_calls_initialization(self):
        module, ferro = load_isolated()
        ferro.cuda_init = Mock(return_value=True)
        module.PreparedSegmentsCUDA.setUpClass()
        ferro.cuda_init.assert_called_once_with(0)

    def test_unavailable_is_optional_skip_or_required_failure(self):
        for required in [False, True]:
            for unavailable in [False, RuntimeError("CUDA unavailable")]:
                with self.subTest(required=required, unavailable=unavailable):
                    module, ferro = load_isolated()
                    ferro.cuda_init = Mock(return_value=unavailable)
                    if isinstance(unavailable, Exception):
                        ferro.cuda_init.side_effect = unavailable
                    env = {"FERRO_REQUIRE_CUDA": "1"} if required else {}
                    expected = AssertionError if required else unittest.SkipTest
                    with patch.dict(os.environ, env, clear=True):
                        with self.assertRaises(expected) as caught:
                            module.PreparedSegmentsCUDA.setUpClass()
                    if required:
                        self.assertIn("CUDA required:", str(caught.exception))
                    ferro.cuda_init.assert_called_once_with(0)

    def test_cpu_discovery_excludes_cuda_only_methods(self):
        module, _ = load_isolated()
        loader = unittest.TestLoader()
        cpu = loader.getTestCaseNames(module.PreparedSegmentsCPU)
        cuda = loader.getTestCaseNames(module.PreparedSegmentsCUDA)
        for name in ["test_device_mismatch_and_cpu_only_ops",
                     "test_registry_replacement_retains_owner_and_rejects_foreign"]:
            self.assertNotIn(name, cpu)
            self.assertIn(name, cuda)
        self.assertEqual(len(cpu), 4)
        self.assertEqual(len(cuda), 6)

    def test_unavailable_skips_cuda_bodies_during_discovery(self):
        module, ferro = load_isolated()
        ferro.cuda_init = Mock(return_value=False)
        with patch.dict(os.environ, {}, clear=True):
            suite = unittest.TestLoader().loadTestsFromTestCase(module.PreparedSegmentsCUDA)
            result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertEqual(result.testsRun, 0)
        self.assertEqual(len(result.skipped), 1)
        ferro.cuda_init.assert_called_once_with(0)

    def test_contract_in_normal_and_optimized_children(self):
        for flags in [[], ["-O"]]:
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [sys.executable, "-B", *flags, str(Path(__file__).resolve()),
                     "PreparedSegmentsSetup.test_setup_calls_initialization",
                     "PreparedSegmentsSetup.test_unavailable_is_optional_skip_or_required_failure",
                     "PreparedSegmentsSetup.test_cpu_discovery_excludes_cuda_only_methods",
                     "PreparedSegmentsSetup.test_unavailable_skips_cuda_bodies_during_discovery", "-v"],
                    text=True, capture_output=True,
                    env={key: value for key, value in os.environ.items() if key != "PYTHONOPTIMIZE"},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
