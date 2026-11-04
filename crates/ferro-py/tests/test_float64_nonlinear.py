import unittest
import ferro as fr
import ferro.autograd as ad
import torch
import torch.nn.functional as tf


class Float64Nonlinear(unittest.TestCase):
    def same(self, actual, expected, *, atol=1e-12):
        self.assertEqual(actual.dtype, 'float64')
        self.assertEqual(list(actual.shape), list(expected.shape))
        got = torch.tensor(actual.tolist(), dtype=torch.float64).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), rtol=1e-12, atol=atol, equal_nan=True)

    def test_elementwise_values_and_live_seed_first_derivatives(self):
        methods = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'exp2', 'expm1', 'log1p', 'log2', 'log10', 'reciprocal', 'silu', 'gelu']
        for name in methods:
            with self.subTest(name=name):
                values = [[.125, .2500000000001, .625], [.75, 1.125, 1.25]]
                a = fr.tensor(values, dtype='float64', requires_grad=True)
                x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                view, tv = a.transpose(0, 1), x.transpose(0, 1)
                actual = getattr(view, name)()
                expected = tf.silu(tv) if name == 'silu' else tf.gelu(tv, approximate='tanh') if name == 'gelu' else getattr(tv, name)()
                self.same(actual, expected)
                seed_values = [[.2, -.4], [.6, .8], [-1., 1.2]]
                seed = fr.tensor(seed_values, dtype='float64', requires_grad=True)
                ts = torch.tensor(seed_values, dtype=torch.float64, requires_grad=True)
                derivative = ad.vjp(actual, [a], seed)[0]
                reference = torch.autograd.grad(expected, x, ts, retain_graph=True)[0]
                self.same(derivative, reference)
                self.assertFalse(derivative.requires_grad)
                self.assertIsNone(a.grad); self.assertIsNone(seed.grad)
                actual.backward_with(seed); expected.backward(ts)
                self.same(a.grad, x.grad)

    def test_small_values_large_values_and_ieee_derivatives(self):
        cases = [('expm1', [1e-300, -1e-300, 1e-30]), ('log1p', [1e-300, -1e-300, 1e-30]),
                 ('sin', [1e-300, 1e100]), ('cos', [1e-300, 1e100]),
                 ('log2', [1e308, 1e-320]), ('log10', [1e308, 1e-320]),
                 ('reciprocal', [1e-200, 1e200])]
        for name, values in cases:
            with self.subTest(name=name):
                a = fr.tensor(values, dtype='float64', requires_grad=True)
                x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                actual, expected = getattr(a, name)(), getattr(x, name)()
                self.same(actual, expected, atol=0)
                actual.sum().backward(); expected.sum().backward()
                self.same(a.grad, x.grad, atol=0)

    def test_softmax_and_log_softmax_all_axes_and_strides(self):
        for name in ['softmax', 'log_softmax']:
            for dim in [0, 1, 2, -1]:
                with self.subTest(name=name, dim=dim):
                    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).div(7).sub(1.3).requires_grad_()
                    a = fr.tensor(x.tolist(), dtype='float64', requires_grad=True)
                    av, xv = a.transpose(0, 2), x.transpose(0, 2)
                    actual, expected = getattr(av, name)(dim), getattr(xv, name)(dim)
                    self.same(actual, expected)
                    seed = torch.arange(24, dtype=torch.float64).mul(.7).sin().reshape(expected.shape)
                    actual.backward_with(fr.tensor(seed.tolist(), dtype='float64')); expected.backward(seed)
                    self.same(a.grad, x.grad)
        for values in [[1e100, 1e100, 1e100], [1000., 999., -1000.], [float('-inf'), 2., 1.],
                       [float('inf'), 2., 1.], [float('nan'), 2., 1.], [float('-inf')] * 3]:
            for name in ['softmax', 'log_softmax']:
                self.same(getattr(fr.tensor(values, dtype='float64'), name)(0),
                          getattr(torch.tensor(values, dtype=torch.float64), name)(0))

    def test_empty_scalar_and_rejected_capture_or_higher_order(self):
        for shape in [(2, 0, 3), (0, 2, 3), (2, 3, 0), ()]:
            for name in ['softmax', 'log_softmax']:
                for dim in range(max(len(shape), 1)):
                    a = fr.zeros(shape, dtype='float64', requires_grad=True)
                    x = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
                    actual, expected = getattr(a, name)(dim), getattr(x, name)(dim)
                    self.same(actual, expected)
                    actual.sum().backward(); expected.sum().backward()
                    self.same(a.grad, x.grad)
        for name in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'exp2', 'expm1', 'log1p', 'log2', 'log10', 'reciprocal', 'silu', 'gelu']:
            self.same(getattr(fr.zeros((2, 0), dtype='float64'), name)(), getattr(torch.zeros((2, 0), dtype=torch.float64), name)() if name not in ['silu', 'gelu'] else (tf.silu(torch.zeros((2, 0), dtype=torch.float64)) if name == 'silu' else tf.gelu(torch.zeros((2, 0), dtype=torch.float64), approximate='tanh')))
        a = fr.tensor([.2, .4], dtype='float64', requires_grad=True)
        for operation in [a.sin, a.silu, a.gelu, lambda: a.softmax(0), lambda: a.log_softmax(0)]:
            with self.assertRaises(ValueError): fr.capture(operation)
            with self.assertRaises(ValueError): ad.grad(operation().sum(), [a], create_graph=True)
        with self.assertRaises(ValueError): a.softmax(1)
        old = a.softmax(0)
        with fr.no_grad(): a.copy_(fr.ones(2, dtype='float64'))
        with self.assertRaises(ValueError): old.sum().backward()


if __name__ == '__main__':
    unittest.main()
