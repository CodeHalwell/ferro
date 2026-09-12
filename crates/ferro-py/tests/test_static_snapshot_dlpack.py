"""Snapshots must be ready for DLPack without a producer-side fence."""
import gc
import os
import unittest

import ferro
import torch


class StaticSnapshotDLPackTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not ferro.cuda_is_available() or not ferro.cuda_init(0):
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                self.fail("CUDA required")
            self.skipTest("CUDA unavailable")

    def test_direct_consumption_on_default_and_nondefault_streams(self):
        size = 2048
        x = ferro.Tensor([1.0] * (size * size), [size, size]).to("cuda")
        w = ferro.Tensor([1.0 / size] * (size * size), [size, size]).to("cuda")
        root = ferro.capture(lambda: x.matmul(w).relu())
        compiled = root.compile_fused()
        graph = compiled.prepare_static()
        replacements = [ferro.Tensor([v] * (size * size), [size, size]).to("cuda") for v in (2.0, 3.0)]
        retained = []
        for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
            for iteration in range(50):
                x.copy_(replacements[iteration % 2])
                graph.replay()
                with torch.cuda.stream(stream):
                    # No tolist(), device synchronize, or producer fence before use.
                    actual = torch.from_dlpack(graph.snapshot())
                    consumed = actual.clone()
                    torch.testing.assert_close(consumed.cpu(), torch.full((size, size), 2.0 + iteration % 2), rtol=0, atol=0)
                if iteration == 0:
                    retained.append(actual)
        del compiled, root, x, w, replacements
        gc.collect()
        graph.replay()
        with torch.cuda.stream(stream):
            final = torch.from_dlpack(graph.snapshot())
            torch.testing.assert_close(final.cpu(), torch.full((size, size), 3.0), rtol=0, atol=0)
        del graph
        gc.collect()
        for old in retained:
            torch.testing.assert_close(old.cpu(), torch.full((size, size), 2.0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
