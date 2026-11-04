"""Functional derivatives reject inference capture before freezing a gradient."""
import unittest

import ferro
from ferro.autograd import grad, vjp


class FunctionalAutogradCaptureTests(unittest.TestCase):
    def test_grad_and_vjp_reject_capture_without_changing_state(self):
        for create_graph in (False, True):
            for derivative in ('grad', 'vjp'):
                with self.subTest(create_graph=create_graph, derivative=derivative):
                    x = ferro.Tensor([2.0], [1]).requires_grad_(True)
                    seed = ferro.Tensor([3.0], [1]).requires_grad_(True)
                    (x * x).sum().backward()
                    before = x.grad.tolist()

                    def build():
                        y = x * x
                        if derivative == 'grad':
                            dx, = grad(y.sum(), [x], create_graph=create_graph)
                        else:
                            dx, = vjp(y, [x], seed, create_graph=create_graph)
                        return dx * x

                    for enabled in (False, True):
                        with ferro.enable_grad() if enabled else ferro.no_grad():
                            with self.assertRaisesRegex(ValueError, 'capture'):
                                ferro.capture(build)
                            self.assertEqual(ferro.is_grad_enabled(), enabled)
                            self.assertEqual(x.tolist(), [2.0])
                            self.assertEqual(x.grad.tolist(), before)
                            self.assertIsNone(seed.grad)
                    self.assertTrue(ferro.is_grad_enabled())
                    self.assertEqual(build().tolist(), [8.0 if derivative == 'grad' else 24.0])

    def test_nested_rejection_restores_outer_capture(self):
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        inference = ferro.Tensor([1.0], [1])

        def build():
            with self.assertRaisesRegex(ValueError, 'capture'):
                ferro.capture(lambda: grad((x * x).sum(), [x])[0] * x)
            return -inference.exp()

        graph = ferro.capture(build).compile_fused()
        self.assertEqual(graph.num_steps, 2)
        inference.copy_(ferro.Tensor([3.0], [1]))
        self.assertEqual(graph.replay().tolist(), (-inference.exp()).tolist())
        self.assertEqual(inference.exp().fusion_launches(), (0, 0))


if __name__ == '__main__':
    unittest.main()
