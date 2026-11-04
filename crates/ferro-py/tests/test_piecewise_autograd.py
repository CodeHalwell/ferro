import unittest
import ferro as fr
import ferro.autograd as ad
import torch


class PiecewiseAutograd(unittest.TestCase):
    def same(self, actual, expected):
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), rtol=1e-6, atol=1e-6, equal_nan=True)
        self.assertEqual(actual.requires_grad, expected.requires_grad)

    def test_nonfinite_inputs_and_cotangents_match_first_order(self):
        for dtype in ['float32', 'float64']:
            for name in ['relu', 'abs', 'sign']:
                for seed in [1., float('inf'), float('-inf'), float('nan')]:
                    with self.subTest(dtype=dtype, name=name, seed=seed):
                        values = [float('nan'), -0., 0., -.2, .2]
                        a = fr.tensor(values, dtype=dtype, requires_grad=True)
                        x = torch.tensor(values, dtype=getattr(torch, dtype), requires_grad=True)
                        actual = ad.vjp(getattr(a, name)(), [a], fr.full((5,), seed, dtype=dtype))[0]
                        expected, = torch.autograd.grad(getattr(x, name)(), x, torch.full((5,), seed, dtype=x.dtype))
                        self.same(actual, expected)

    def test_piecewise_create_graph_keeps_input_and_live_seed_dependencies(self):
        for name in ['relu', 'abs']:
            for seed in [1., float('inf'), float('nan')]:
                with self.subTest(name=name, seed=seed):
                    values = [float('nan'), 0., -.2, .2]
                    a = fr.tensor(values, requires_grad=True); x = torch.tensor(values, requires_grad=True)
                    v = fr.full((4,), seed, requires_grad=True); tv = torch.full((4,), seed, requires_grad=True)
                    actual = ad.vjp(getattr(a, name)(), [a], v, create_graph=True)[0]
                    expected, = torch.autograd.grad(getattr(x, name)(), x, tv, create_graph=True)
                    self.same(actual, expected)
                    self.assertTrue(actual._depends_on(a))
                    self.assertTrue(actual._depends_on(v))
                    got = ad.grad(actual.sum(), [a, v])
                    reference = torch.autograd.grad(expected.sum(), [x, tv])
                    for g, e in zip(got, reference): self.same(g, e)

    def test_relu_strict_hessian_and_higher_zero_derivatives_keep_result_history(self):
        from ferro.autograd import functional as af
        from torch.autograd import functional as tf
        a = fr.tensor([-.4, .3], requires_grad=True)
        x = torch.tensor([-.4, .3], requires_grad=True)
        actual = af.hessian(lambda value: value.relu().sum(), a, create_graph=True, strict=True)
        expected = tf.hessian(lambda value: value.relu().sum(), x, create_graph=True, strict=True)
        self.same(actual, expected)
        for _order in range(3, 6):
            actual = ad.grad(actual.sum(), [a], create_graph=True)[0]
            expected, = torch.autograd.grad(expected.sum(), x, create_graph=True)
            self.same(actual, expected)
            self.assertTrue(actual._depends_on(a))


if __name__ == '__main__':
    unittest.main()
