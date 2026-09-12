"""General eager exports: immediate foreign consumption, no producer fence."""
import gc
import os
import unittest

import ferro
import torch


class CpuDLPackTests(unittest.TestCase):
    def test_owned_cpu_capsule_lifetime_and_stream_contract(self):
        tensor = ferro.Tensor([1.0, 2.0], [2])
        with self.assertRaises(ValueError):
            tensor.__dlpack__(stream=1)
        capsule = tensor.__dlpack__()
        abandoned = tensor.__dlpack__()
        del abandoned, tensor
        gc.collect()
        imported = torch.from_dlpack(capsule)
        torch.testing.assert_close(imported, torch.tensor([1.0, 2.0]))
        with self.assertRaises(RuntimeError):
            torch.from_dlpack(capsule)


class EagerDLPackTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not ferro.cuda_is_available() or not ferro.cuda_init(0):
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                self.fail("CUDA required")
            self.skipTest("CUDA unavailable")

    def check_consumer(self, stream):
        size = 2048
        weight = ferro.Tensor([1.0 / size] * (size * size), [size, size]).to("cuda")
        for iteration in range(12):
            value = float(2 + iteration % 3)
            source = ferro.Tensor([value] * (size * size), [size, size]).to("cuda")
            # Queue substantial real eager GPU production, not a host sleep.
            output = source
            for _ in range(8):
                output = output.matmul(weight)
            output = output.relu()
            with torch.cuda.stream(stream):
                exported = torch.from_dlpack(output)
                consumed = exported.clone()
                torch.testing.assert_close(consumed.cpu(), torch.full((size, size), value), rtol=0, atol=0)
            del output, source
            gc.collect()
            torch.testing.assert_close(exported.cpu(), torch.full((size, size), value), rtol=0, atol=0)

    def test_export_keeps_original_allocation_after_backend_reinstall(self):
        tensor = ferro.Tensor([3.0] * 1024, [1024]).to("cuda").relu()
        self.assertTrue(ferro.cuda_init(0))
        first = torch.from_dlpack(tensor)
        capsule = tensor.__dlpack__()
        second = torch.from_dlpack(capsule)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        with self.assertRaises(RuntimeError):
            torch.from_dlpack(capsule)
        del tensor
        gc.collect()
        torch.testing.assert_close(first.cpu(), torch.full((1024,), 3.0), rtol=0, atol=0)
        torch.testing.assert_close(second.cpu(), torch.full((1024,), 3.0), rtol=0, atol=0)

    def test_stream_token_validation(self):
        tensor = ferro.Tensor([2.0], [1]).to("cuda")
        for token in (0, -1, -2, True, False, 1.5, "1", object(), 1 << 128):
            with self.subTest(token=repr(token)), self.assertRaises((TypeError, ValueError, OverflowError)):
                tensor.__dlpack__(stream=token)
        consumer = torch.cuda.Stream()
        for token in (None, 1, 2, consumer.cuda_stream):
            capsule = tensor.__dlpack__(stream=token)
            torch.testing.assert_close(torch.from_dlpack(capsule).cpu(), torch.tensor([2.0]))

    def test_default_consumer(self):
        self.check_consumer(torch.cuda.default_stream())

    def test_nondefault_consumer(self):
        self.check_consumer(torch.cuda.Stream())


if __name__ == "__main__":
    unittest.main()
