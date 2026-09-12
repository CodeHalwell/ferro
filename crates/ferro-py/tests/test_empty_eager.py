"""Eager empty CUDA regressions independent of DLPack import/static replay."""
import os
import unittest

import ferro
import torch


def tensor(value):
    return ferro.Tensor(value.flatten().tolist(), list(value.shape)).to("cuda")


class EmptyEagerTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not ferro.cuda_is_available() or not ferro.cuda_init(0):
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                self.fail("CUDA required")
            self.skipTest("CUDA unavailable")

    def check(self, actual, expected):
        self.assertEqual(actual.__dlpack_device__(), (2, 0))
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.tolist(), expected.cpu().tolist())

    def test_empty_softmax_every_axis(self):
        for shape in [(0,), (0, 3), (2, 0), (2, 0, 4), (0, 3, 4)]:
            expected_input = torch.empty(shape, device="cuda")
            actual_input = tensor(expected_input)
            for axis in range(len(shape)):
                with self.subTest(shape=shape, axis=axis):
                    self.check(actual_input.softmax(axis), expected_input.softmax(axis))

    def test_empty_and_zero_reduction_gemm(self):
        for m, k, n in [(0, 3, 4), (2, 3, 0), (0, 0, 4), (2, 0, 0), (2, 0, 3)]:
            with self.subTest(m=m, k=k, n=n):
                a, b = torch.ones((m, k), device="cuda"), torch.ones((k, n), device="cuda")
                self.check(tensor(a).matmul(tensor(b)), a @ b)


if __name__ == "__main__":
    unittest.main()
