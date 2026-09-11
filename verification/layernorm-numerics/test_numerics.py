"""CPU-only tests; does not import ferro or initialize CUDA."""
import importlib.util
import unittest

import numpy as np


class MetricTests(unittest.TestCase):
    def test_ordered_ulp_edges(self):
        self.assertIsNotNone(importlib.util.find_spec("numerics"), "ULP helper is not implemented")
        from numerics import ulp_distance
        tiny = np.nextafter(np.float32(0), np.float32(1))
        a = np.array([0., -0., 1., -1., -tiny, np.nan, np.nan, np.inf], dtype=np.float32)
        b = np.array([-0., 0., np.nextafter(np.float32(1), np.float32(2)),
                      np.nextafter(np.float32(-1), np.float32(-2)), tiny, np.nan, 1., np.inf], dtype=np.float32)
        np.testing.assert_array_equal(ulp_distance(a, b), [0, 0, 1, 1, 2, 0, 2**32-1, 0])
        self.assertGreater(ulp_distance(np.array([1e-7], np.float32), np.array([9e-7], np.float32))[0], 1000000)

    def test_comparison_keeps_near_zero_tail_and_rejects_nan(self):
        import numerics
        self.assertTrue(hasattr(numerics, "compare"), "comparison reporter is not implemented")
        a = np.array([0., 9e-7, 1.], np.float32)
        b = np.array([0., 1e-7, 1.], np.float32)
        report = numerics.compare(a, b, atol=1e-6, rtol=0.)
        self.assertTrue(report["passed"])
        self.assertEqual(report["relative_reference_zero_count"], 1)
        self.assertGreater(report["ulp"]["p100"], 1000000)
        self.assertEqual(report["ulp"]["p99"], report["ulp"]["p100"])
        self.assertFalse(numerics.compare(np.array([np.nan], np.float32), np.array([np.nan], np.float32), 1., 1.)["passed"])
        self.assertFalse(numerics.compare(a, b, 0., 0.)["passed"])

    def test_reproducible_cases_cover_independent_affine(self):
        self.assertIsNotNone(importlib.util.find_spec("validate"), "validation driver is not implemented")
        from validate import cases
        first = list(cases([17], [3, 33], [1e-5], 1))
        second = list(cases([17], [3, 33], [1e-5], 1))
        self.assertEqual(len(first), 40)
        self.assertEqual({c[0]["affine"] for c in first}, {"none", "weight", "bias", "both"})
        self.assertEqual({c[0]["distribution"] for c in first}, {"normal", "wide", "constant", "near_constant", "large_offset"})
        for left, right in zip(first, second):
            self.assertEqual(left[0], right[0])
            for key in left[1]:
                np.testing.assert_array_equal(left[1][key], right[1][key])
            for key in ("weight", "bias"):
                if key in left[1]:
                    self.assertGreater(np.ptp(left[1][key]), 0)

    def test_cpu_torch_oracle_and_gate_contract(self):
        import torch
        from validate import torch_outputs, gate_budget
        arrays = dict(x=np.full((2, 3), .5, np.float32),
                      weight=np.array([-2., .25, 1.5], np.float32),
                      bias=np.array([.3, -.4, .8], np.float32),
                      upstream=np.array([[.2, -.7, 1.1], [-.3, .9, -.8]], np.float32))
        eps = float(np.float32(1e-5))
        out = torch_outputs(torch, arrays, eps, "cpu", torch.float64, True)
        np.testing.assert_array_equal(out["output"], np.broadcast_to(arrays["bias"], (2, 3)))
        np.testing.assert_array_equal(out["grad_weight"], np.zeros(3))
        np.testing.assert_allclose(out["grad_bias"], arrays["upstream"].astype(np.float64).sum(0), atol=1e-12, rtol=0)
        dxh = arrays["upstream"].astype(np.float64) * arrays["weight"]
        np.testing.assert_allclose(out["grad_x"], (dxh - dxh.mean(-1, keepdims=True)) / np.sqrt(eps), rtol=1e-12, atol=1e-12)
        meta = dict(distribution="constant", eps=eps)
        atol, rtol, condition = gate_budget(meta, arrays, out["output"], "output")
        self.assertEqual((atol, rtol, condition), (5e-5, 5e-5, 1.))
        self.assertFalse(__import__("numerics").compare(out["output"] + .01, out["output"], atol, rtol)["passed"])


if __name__ == "__main__":
    unittest.main()
