"""Dependency-free inference capture regression tests: python -m unittest discover -s tests."""
import unittest
import ferro


class CaptureTests(unittest.TestCase):
    def test_changed_leaf_values(self):
        x = ferro.Tensor([1., 2.], [2])
        graph = ferro.capture(lambda: -(x.exp())).compile_fused()
        before = graph.replay().tolist()
        self.assertIs(x.copy_(ferro.Tensor([3., 4.], [2])), x)
        after = graph.replay().tolist()
        self.assertNotEqual(before, after)
        self.assertEqual(after, (-x.exp()).tolist())

    def test_nested_exception_cleanup_and_threads(self):
        import concurrent.futures
        x = ferro.Tensor([1., 2.], [2])
        def outer():
            with self.assertRaisesRegex(RuntimeError, "boom"):
                ferro.capture(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                self.assertEqual(pool.submit(lambda: x.exp().fusion_launches()).result(), (0, 0))
            return -ferro.capture(lambda: x.exp())
        y = ferro.capture(outer)
        self.assertEqual(y.compile_fused().num_steps, 2)
        with self.assertRaises(RuntimeError):
            ferro.capture(lambda: (_ for _ in ()).throw(RuntimeError("outer")))
        self.assertEqual(x.exp().fusion_launches(), (0, 0))
        with self.assertRaises(TypeError):
            ferro.capture(lambda: None)
        self.assertEqual(x.exp().fusion_launches(), (0, 0))

    def test_cuda_capture_changed_leaves(self):
        if not ferro.cuda_is_available() or not ferro.cuda_init(0):
            self.skipTest('CUDA runtime unavailable')
        x = ferro.Tensor([1., 2.], [2]).to('cuda')
        y = ferro.capture(lambda: -x.exp())
        graph = y.compile_fused()
        x.copy_(ferro.Tensor([3., 4.], [2]))
        result = graph.replay()
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.fusion_launches(), (0, 0))
        for actual, expected in zip(result.tolist(), (-x.exp()).tolist()):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_unsupported_barrier_and_checked_copy(self):
        x = ferro.Tensor([1., 2.], [2])
        with self.assertRaises(ValueError):
            ferro.capture(lambda: -x.log_softmax(0)).compile_fused()
        with self.assertRaises(ValueError):
            x.copy_(ferro.Tensor([1.], [1]))
        self.assertEqual(x.tolist(), [1., 2.])

    def test_branched_graph_replays_shared_nodes(self):
        x = ferro.Tensor([1., 2.], [2])
        def build():
            shared = x.exp()
            return shared * shared + x.sigmoid()
        y = ferro.capture(build)
        graph = y.compile_fused()
        self.assertEqual(graph.num_steps, 4)
        self.assertEqual(graph.num_operands, 1)
        self.assertGreater(graph.num_runs, 1)
        x.copy_(ferro.Tensor([2., 3.], [2]))
        self.assertEqual(graph.replay().tolist(), build().tolist())

    def test_capture_compile_replay(self):
        x = ferro.Tensor([1., 2.], [2])
        y = ferro.capture(lambda: -(x.exp()))
        self.assertFalse(y.requires_grad)
        graph = y.compile_fused()
        self.assertEqual(graph.num_steps, 2)
        self.assertEqual(graph.replay().tolist(), y.tolist())
        self.assertFalse(graph.replay().requires_grad)
        self.assertEqual(graph.replay().fusion_launches(), (0, 0))
        with self.assertRaises(ValueError):
            x.exp().compile_fused()


if __name__ == '__main__':
    unittest.main()
