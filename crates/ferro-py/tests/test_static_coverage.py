"""Public static replay shape acceptance, with independent Torch oracles."""
import gc
import os
import unittest

import ferro
import torch


def tensor(value):
    return ferro.Tensor(value.flatten().tolist(), list(value.shape)).to("cuda")


class StaticCoverageTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not ferro.cuda_is_available() or not ferro.cuda_init(0):
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                self.fail("CUDA required for both Torch and ferro")
            self.skipTest("CUDA unavailable")

    def assert_values(self, actual, expected, max_ulp):
        actual = actual.detach().cpu()
        expected = expected.detach().cpu()
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.isfinite(expected).all().item())
        if not actual.numel():
            return
        def ordered(value):
            value = torch.where(value.abs() < torch.finfo(torch.float32).tiny, 0., value).contiguous()
            bits = value.view(torch.int32).to(torch.int64) & 0xffffffff
            return bits ^ torch.where((bits & 0x80000000) != 0, 0xffffffff, 0x80000000)
        distance = (ordered(actual) - ordered(expected)).abs()
        self.assertLessEqual(distance.max().item(), max_ulp, f"maximum f32 ULP distance: {distance.max().item()}")

    def eager_values(self, values, leaves, forward, shape):
        # Legacy eager CUDA has empty-softmax/zero-column-GEMM failures.
        # CPU eager is the independent Ferro oracle for degenerate inputs.
        oracle_leaves = [ferro.Tensor(value.flatten().tolist(), list(value.shape)) for value in values] if any(value.numel() == 0 for value in values) else leaves
        result = forward(*oracle_leaves)
        self.assertEqual(tuple(result.shape), tuple(shape))
        return torch.tensor(result.tolist(), dtype=torch.float32).reshape(shape)

    def exercise(self, inputs, forward, reference, max_ulp=32, mutate=True):
        leaves = [tensor(value) for value in inputs]
        expected = reference(*inputs)
        eager_values = self.eager_values(inputs, leaves, forward, expected.shape)
        self.assert_values(eager_values, expected, max_ulp)
        root = ferro.capture(lambda: forward(*leaves))
        compiled = root.compile_fused()
        graph = compiled.prepare_static()
        retained = []
        stream = torch.cuda.Stream()
        for iteration in range(3):
            values = [value + iteration * (index + 1) * 0.125 if mutate else value
                      for index, value in enumerate(inputs)]
            if iteration and mutate:
                for leaf, value in zip(leaves, values):
                    leaf.copy_(tensor(value))
            expected = reference(*values)
            self.assertIsNone(graph.replay())
            snapshot = graph.snapshot()
            self.assertFalse(snapshot.requires_grad)
            self.assertEqual(tuple(snapshot.shape), tuple(expected.shape))
            with torch.cuda.stream(stream):
                # Consume directly: no producer tolist(), cpu(), or fence first.
                consumed = torch.from_dlpack(snapshot)
                self.assertEqual(consumed.device.type, "cuda")
                copied = consumed.clone()
                self.assert_values(copied, expected, max_ulp)
            eager_values = self.eager_values(values, leaves, forward, expected.shape)
            self.assert_values(copied, eager_values, max_ulp)
            retained.append((snapshot, consumed, expected.clone()))
        self.assertEqual(graph.replay_count, 3)
        if mutate and expected.numel():
            self.assertFalse(torch.equal(retained[0][2], retained[-1][2]), "mutation must change oracle output")
            self.assertNotEqual(retained[0][1].data_ptr(), retained[-1][1].data_ptr())
        del graph, compiled, root, leaves, snapshot
        gc.collect()
        for snapshot, consumed, expected in retained:
            self.assert_values(consumed, expected, max_ulp)
            snapshot_values = torch.tensor(snapshot.tolist(), dtype=torch.float32).reshape(expected.shape)
            self.assert_values(snapshot_values, expected, max_ulp)
        # The DLPack consumer owns its buffer after all Ferro snapshots die.
        consumers = [(consumed, expected) for _, consumed, expected in retained]
        del retained, snapshot
        gc.collect()
        for consumed, expected in consumers:
            self.assert_values(consumed, expected, max_ulp)

    def test_softmax_axes(self):
        value = (torch.arange(24, dtype=torch.float32).sin() * 2).reshape(2, 3, 4)
        for dim in (0, 1, 2, -1, -2, -3):
            with self.subTest(dim=dim):
                # Nonuniform mutation avoids softmax's constant-shift invariance.
                self.exercise([value, torch.ones_like(value)],
                              lambda x, scale: (x * scale).softmax(dim),
                              lambda x, scale: torch.softmax(x * scale, dim))

    def test_supported_scalar_and_same_shape_controls(self):
        for shape in ((), (2, 3)):
            with self.subTest(shape=shape):
                def forward(a, b):
                    shared = a / b
                    return (shared - b) / (shared + a)
                self.exercise([torch.full(shape, 2.), torch.full(shape, 5.)],
                              forward, forward, max_ulp=8)

    def test_expanding_first_pointwise_seed(self):
        shapes = [((1, 3), (2, 3)), ((2, 1), (2, 3)),
                  ((3,), (2, 1, 3)), ((), (2, 3)), ((1, 3, 1), (2, 1, 4))]
        for small, large in shapes:
            for operation in ("subtract", "divide", "shared"):
                with self.subTest(small=small, large=large, operation=operation):
                    x = torch.arange(torch.empty(small).numel(), dtype=torch.float32).reshape(small) * 0.25 + 2
                    y = torch.arange(torch.empty(large).numel(), dtype=torch.float32).reshape(large) * 0.125 + 5
                    if operation == "subtract":
                        forward = lambda a, b: a - b
                    elif operation == "divide":
                        forward = lambda a, b: a / b
                    else:
                        # Both a shared leaf and a shared intermediate; order matters.
                        def forward(a, b):
                            shared = a / b
                            return (shared - b) / (shared + a)
                    self.exercise([x, y], forward, forward, max_ulp=8)

    def test_empty_pointwise_outputs_preserve_rank(self):
        for shape in ((0,), (0, 3), (2, 0), (2, 0, 4), (1, 2, 0, 3)):
            with self.subTest(shape=shape):
                self.exercise([torch.empty(shape)], lambda x: -x.relu(),
                              lambda x: -torch.relu(x), max_ulp=0, mutate=False)

    def test_empty_broadcast_outputs(self):
        for left, right in (((1,), (0,)), ((1, 3), (0, 3)), ((2, 1, 4), (1, 0, 1))):
            with self.subTest(left=left, right=right):
                self.exercise([torch.ones(left), torch.ones(right)],
                              lambda a, b: (a - b) / a, lambda a, b: (a - b) / a,
                              max_ulp=0, mutate=False)

    def test_empty_softmax_outputs(self):
        for shape in ((0,), (0, 3), (2, 0), (2, 0, 4), (0, 3, 4)):
            for dim in range(len(shape)):
                with self.subTest(shape=shape, dim=dim):
                    self.exercise([torch.empty(shape)], lambda x: x.softmax(dim),
                                  lambda x: torch.softmax(x, dim), max_ulp=0, mutate=False)

    def test_empty_layout_outputs(self):
        self.exercise([torch.empty(2, 0, 4)],
                      lambda x: x.transpose(0, 2).reshape([0, 8]).relu(),
                      lambda x: x.transpose(0, 2).reshape(0, 8).relu(),
                      max_ulp=0, mutate=False)

    def test_empty_matmul_outputs(self):
        for m, k, n in ((0, 3, 4), (2, 3, 0), (0, 0, 4), (2, 0, 0)):
            with self.subTest(m=m, k=k, n=n):
                self.exercise([torch.ones(m, k), torch.ones(k, n)],
                              lambda a, b: a.matmul(b), lambda a, b: a @ b,
                              max_ulp=0, mutate=False)

    def test_zero_inner_dimension_matmul(self):
        for m, n in ((1, 1), (2, 3), (3, 5)):
            with self.subTest(m=m, n=n, bias=False):
                self.exercise([torch.empty(m, 0), torch.empty(0, n)],
                              lambda a, b: a.matmul(b), lambda a, b: a @ b,
                              max_ulp=0, mutate=False)
            with self.subTest(m=m, n=n, bias=True):
                self.exercise([torch.empty(m, 0), torch.empty(0, n), torch.arange(n, dtype=torch.float32) + 1],
                              lambda a, b, bias: a.matmul(b) + bias,
                              lambda a, b, bias: a @ b + bias, max_ulp=0)

    def test_invalid_dimensions_and_shapes_still_rejected(self):
        x = tensor(torch.ones(2, 3, 4))
        for dim in (3, -4):
            with self.subTest(dim=dim):
                with self.assertRaisesRegex(ValueError, "dim|axis"):
                    ferro.capture(lambda: x.softmax(dim)).compile_fused().prepare_static()
        for a, b in ((torch.ones(2, 3), torch.ones(4, 2)),
                     (torch.empty(2, 0), torch.ones(1, 3))):
            with self.subTest(a=tuple(a.shape), b=tuple(b.shape)):
                left, right = tensor(a), tensor(b)
                with self.assertRaises(ValueError):
                    ferro.capture(lambda: left.matmul(right)).compile_fused().prepare_static()
        left, right = tensor(torch.ones(2, 3)), tensor(torch.ones(4, 3))
        with self.assertRaises(ValueError):
            ferro.capture(lambda: left - right).compile_fused().prepare_static()
        # A new softmax axis is supported; log_softmax remains a distinct barrier.
        with self.assertRaises(ValueError):
            ferro.capture(lambda: x.log_softmax(-1)).compile_fused().prepare_static()


if __name__ == "__main__":
    unittest.main()
