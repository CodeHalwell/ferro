"""Public reusable topology contracts; CUDA is mandatory with FERRO_REQUIRE_CUDA."""
import gc
import math
import os
import unittest

import ferro as fr
import ferro._native as native


class PreparedSegmentsCPU(unittest.TestCase):
    device = "cpu"

    def tensor(self, values, shape, grad=False):
        return fr.Tensor(values, shape).to(self.device).requires_grad_(grad)

    def test_reuse_fresh_inputs_and_gradients(self):
        self.assertTrue(hasattr(fr.graph, "PreparedSegments"), "missing reusable Python topology")
        self.assertIs(fr.graph.PreparedSegments, native.PreparedSegments)
        ids = [2, 0, 2]
        plan = fr.graph.PreparedSegments(ids, num_segments=4, device=self.device)
        ids[:] = [0, 0, 0]
        del ids
        gc.collect()
        for shift in [0., 2.]:
            x = self.tensor([1.+shift, 2., 3.], [3], True)
            y = plan.sum(x)
            self.assertEqual(y.device, self.device)
            self.assertEqual(y.tolist(), [2., 0., 4.+shift, 0.])
            y.backward_with(self.tensor([.1, .2, .3, .4], [4]))
            for actual, expected in zip(x.grad.tolist(), [.3, .1, .3]):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(x.grad.device, self.device)
            z = self.tensor([1.+shift, 2., 3.], [3], True)
            out = plan.softmax(z)
            p = 1. / (1. + math.exp(2.-shift))
            for actual, expected in zip(out.tolist(), [p, 1., 1.-p]):
                self.assertAlmostEqual(actual, expected, places=6)
            out.backward_with(self.tensor([.2, .4, .7], [3]))
            for actual, expected in zip(z.grad.tolist(), [-.5*p*(1.-p), 0., .5*p*(1.-p)]):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(z.grad.device, self.device)

    def test_validation_and_nonfinite_errors(self):
        for ids, groups in [([-1], 2), ([2], 2), ([2**63], 2), ([0.5], 2), ([0], -1)]:
            with self.subTest(ids=ids, groups=groups):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    fr.graph.PreparedSegments(ids, groups, self.device)
        for device in ["bogus", "cuda:-1", "cuda:4294967295"]:
            with self.assertRaises(ValueError):
                fr.graph.PreparedSegments([0], 1, device)
        plan = fr.graph.PreparedSegments([0], 1, self.device)
        with self.assertRaises(ValueError):
            plan.sum(self.tensor([1., 2.], [2]))
        with self.assertRaises(ValueError):
            plan.sum(fr.Tensor.from_i64([1], [1]).to(self.device))
        with self.assertRaisesRegex(ValueError, "capture/replay"):
            fr.capture(lambda: plan.sum(self.tensor([1.], [1])))
        with self.assertRaisesRegex(ValueError, "capture/replay"):
            fr.capture(lambda: plan.softmax(self.tensor([1.], [1])))
        for value in [float("inf"), float("-inf"), float("nan")]:
            with self.assertRaisesRegex(ValueError, "finite"):
                plan.softmax(self.tensor([value], [1]))
        self.assertEqual(plan.softmax(self.tensor([2.], [1])).tolist(), [1.])

    def test_backward_owns_dropped_plan(self):
        for method in ["sum", "softmax"]:
            plan = fr.graph.PreparedSegments([1, 0, 1], 3, self.device)
            x = self.tensor([1., 2., 3.], [3], True)
            y = getattr(plan, method)(x)
            del plan
            gc.collect()
            y.backward_with(self.tensor([.2, .4, .7], [3]))
            p = 1. / (1. + math.exp(2.))
            expected = [.4, .2, .4] if method == "sum" else [-.5*p*(1.-p), 0., .5*p*(1.-p)]
            for a, b in zip(x.grad.tolist(), expected):
                self.assertAlmostEqual(a, b, places=6)
            self.assertEqual(x.grad.device, self.device)

    def test_empty_and_convenience_contracts(self):
        for shape, ids, groups in [([0, 2], [], 4), ([0, 2], [], 0), ([3, 0], [2, 0, 2], 4)]:
            plan = fr.graph.PreparedSegments(ids, groups, self.device)
            x = self.tensor([], shape, True)
            self.assertEqual(plan.softmax(x).shape, shape)
            y = plan.sum(x)
            self.assertEqual(y.shape, [groups, shape[1]])
            y.backward_with(fr.Tensor.ones(y.shape, device=self.device))
            self.assertEqual(x.grad.device, self.device)
        x = self.tensor([1., 2., 3.], [3])
        plan = fr.graph.PreparedSegments([1, 0, 1], 3, self.device)
        for name in ["sum", "softmax"]:
            actual = getattr(fr.graph, "segment_" + name)(x, [1, 0, 1], 3)
            self.assertEqual(actual.device, self.device)
            self.assertEqual(actual.tolist(), getattr(plan, name)(x).tolist())


class PreparedSegmentsCUDA(PreparedSegmentsCPU):
    device = "cuda:0"

    @classmethod
    def setUpClass(cls):
        try:
            assert fr.cuda_init(0)
        except Exception as exc:
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                raise AssertionError(f"CUDA required: {exc}") from exc
            raise unittest.SkipTest(str(exc))

    def test_device_mismatch_and_cpu_only_ops(self):
        plan = fr.graph.PreparedSegments([0], 1, self.device)
        cpu = fr.Tensor([1.], [1])
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            plan.sum(cpu)
        with self.assertRaisesRegex(ValueError, "device mismatch"):
            fr.graph.PreparedSegments([0], 1).softmax(cpu.cuda())
        for name in ["segment_mean", "segment_max"]:
            with self.assertRaisesRegex(ValueError, "CPU"):
                getattr(fr.graph, name)(cpu.cuda(), [0], 1)

    def test_registry_replacement_retains_owner_and_rejects_foreign(self):
        # All old-context tensors, including backward seeds, precede replacement.
        plan = fr.graph.PreparedSegments([1, 0, 1], 3, self.device)
        xs = [self.tensor([1.+shift, 2., 3.], [3], True) for shift in [0., 2.]]
        sum_x = self.tensor([1., 2., 3.], [3], True)
        seed = self.tensor([.2, .4, .7], [3])
        fr.cuda_init(0)  # install actually replaces the registry backend/stream
        with self.assertRaises(ValueError):
            plan.sum(self.tensor([1., 2., 3.], [3]))
        outputs = [plan.softmax(x) for x in xs]
        sum_y = plan.sum(sum_x)
        del plan
        gc.collect()
        # Generic tolist uses the replacement registry, so inspect through the
        # allocation-owned DLPack producer fence instead (no production workaround).
        import torch
        sum_y.backward_with(seed)
        torch.testing.assert_close(torch.from_dlpack(sum_y).cpu(), torch.tensor([2., 4., 0.]))
        torch.testing.assert_close(torch.from_dlpack(sum_x.grad).cpu(), torch.tensor([.4, .2, .4]))
        self.assertEqual(sum_x.grad.device, self.device)
        for x, y, p in zip(xs, outputs, [1./(1.+math.exp(2.)), .5]):
            y.backward_with(seed)
            torch.testing.assert_close(torch.from_dlpack(y).cpu(), torch.tensor([p, 1., 1.-p]))
            actual = torch.from_dlpack(x.grad).cpu().tolist()
            for a, b in zip(actual, [-.5*p*(1.-p), 0., .5*p*(1.-p)]):
                self.assertAlmostEqual(a, b, places=6)
            self.assertEqual(x.grad.device, self.device)


if __name__ == "__main__":
    unittest.main(verbosity=2)
