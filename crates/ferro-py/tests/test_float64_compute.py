import math
import unittest

import ferro as fr
import ferro.autograd as ad
import torch


class Float64Compute(unittest.TestCase):
    def tensor(self, value, grad=True):
        if isinstance(value, torch.Tensor):
            result = fr.tensor(value.tolist(), dtype='float64', requires_grad=grad)
            return result.reshape(list(value.shape)) if list(result.shape) != list(value.shape) else result
        return fr.tensor(value, dtype='float64', requires_grad=grad)

    def same(self, actual, expected, atol=1e-12):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(list(actual.shape), list(expected.shape))
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), rtol=1e-12, atol=atol, equal_nan=True)

    def gradients(self, actual_inputs, expected_inputs, actual, expected, seed=None):
        self.same(actual, expected)
        if seed is None:
            actual.sum().backward()
            expected.sum().backward()
        else:
            actual.backward_with(self.tensor(seed, False))
            expected.backward(seed)
        for a, b in zip(actual_inputs, expected_inputs):
            self.assertIsNotNone(a.grad)
            self.same(a.grad, b.grad)

    def test_broadcast_arithmetic_all_input_gradients(self):
        torch.manual_seed(8321)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        b = torch.tensor([0.7, 1.1, 1.5], dtype=torch.float64, requires_grad=True)
        a, c = self.tensor(x), self.tensor(b)
        def expression(x, b):
            return ((x * b + x / b - 0.1234567890123456) * 1.0000000000001).tanh() + 0.2 / b
        self.gradients([a, c], [x, b], expression(a, c), expression(x, b), torch.randn(2, 3, dtype=torch.float64))

    def test_unary_values_and_gradients(self):
        methods = [lambda x: -x, lambda x: x.exp(), lambda x: x.log(), lambda x: x.sigmoid(),
                   lambda x: x.tanh(), lambda x: x.sqrt(), lambda x: x.abs(), lambda x: x.relu(),
                   lambda x: x.square(), lambda x: x ** 1.23456789012345,
                   lambda x: x.clamp(0.7123456789012345, 1.3123456789012345)]
        for i, operation in enumerate(methods):
            with self.subTest(operation=i):
                expected = torch.tensor([0.4, 0.8, 1.2, 1.6], dtype=torch.float64, requires_grad=True)
                actual = self.tensor(expected)
                self.gradients([actual], [expected], operation(actual), operation(expected))
        expected = torch.tensor([-1.1, 0.0, 1.3], dtype=torch.float64, requires_grad=True)
        actual = self.tensor(expected)
        self.gradients([actual], [expected], actual.relu() + actual.abs(), expected.relu() + expected.abs())

    def test_precision_range_subnormal_and_accumulation(self):
        values = [1.0000000000000002, 1e100, 1e-300, float.fromhex('0x0.0000000000007p-1022')]
        expected = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        actual = self.tensor(expected)
        self.same(actual * 2.0, expected * 2.0, atol=0)
        a, b = actual * 2.0, expected * 2.0
        a.sum().backward(); b.sum().backward(retain_graph=True)
        held, reference = actual.grad, expected.grad
        a.sum().backward(); b.sum().backward()
        self.same(actual.grad, expected.grad, atol=0)
        self.assertEqual(actual.grad._id(), held._id())
        self.assertIs(expected.grad, reference)
        self.same(held, reference, atol=0)
        for magnitude in [1e-300, 1e100]:
            with self.subTest(magnitude=magnitude):
                x = torch.tensor(magnitude, dtype=torch.float64, requires_grad=True)
                y = torch.tensor(magnitude, dtype=torch.float64, requires_grad=True)
                a, b = self.tensor(x), self.tensor(y)
                self.gradients([a, b], [x, y], a / b, x / y)
                self.same(a.grad, x.grad, atol=0)
                self.same(b.grad, y.grad, atol=0)
        for dtype in ['float32', 'float64']:
            for p in [0., 1.]:
                with self.subTest(dtype=dtype, p=p):
                    x = torch.tensor([0., -0., float('inf'), float('nan')], dtype=getattr(torch, dtype), requires_grad=True)
                    a = fr.tensor(x.tolist(), dtype=dtype, requires_grad=True)
                    self.gradients([a], [x], a.pow(p), x.pow(p))

    def test_matmul_broadcast_vectors_strides_and_empty(self):
        torch.manual_seed(713)
        for sa, sb in [((2, 3), (3, 4)), ((3,), (3,)), ((3,), (3, 2)), ((2, 3), (3,)),
                       ((2, 1, 2, 3), (1, 4, 3, 2)), ((2, 3, 4), (4, 2)),
                       ((2, 0), (0, 3)), ((0, 2, 3), (3, 4))]:
            with self.subTest(sa=sa, sb=sb):
                x = torch.randn(sa, dtype=torch.float64, requires_grad=True)
                y = torch.randn(sb, dtype=torch.float64, requires_grad=True)
                a, b = self.tensor(x), self.tensor(y)
                self.gradients([a, b], [x, y], a @ b, x @ y)
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        y = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        a, b = self.tensor(x), self.tensor(y)
        self.gradients([a, b], [x, y], a.transpose(1, 2).bmm(b), x.transpose(1, 2).bmm(y))

    def test_empty_broadcast_capacity(self):
        # Empty outputs must not materialize an enormous broadcast operand.
        a = self.tensor([[[1., 2.]]])
        b = fr.Tensor.from_f64([], [1 << 61, 2, 0]).requires_grad_(True)
        out = a @ b
        self.assertEqual(list(out.shape), [1 << 61, 1, 0])
        out.sum().backward()
        self.assertEqual(a.grad.tolist(), [[[0., 0.]]])
        self.assertEqual(b.grad.numel(), 0)

    def test_reductions_views_masks_and_typed_selectors(self):
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float64, requires_grad=True)
        a = self.tensor(x)
        self.gradients([a], [x], a[:, 1::2].transpose(0, 1).reshape([4]).mean(), x[:, 1::2].T.reshape(4).mean())
        x = torch.tensor([[0.2], [0.5]], dtype=torch.float64, requires_grad=True)
        a = self.tensor(x)
        mask = [[True, False, True], [False, True, True]]
        self.gradients([a], [x], a.masked_select(fr.tensor(mask)).sum(), x.masked_select(torch.tensor(mask)).sum())
        for shape in [(2, 3), (2, 0, 3), (0,), ()]:
            for axis in range(len(shape)):
                for keepdim in [False, True]:
                    with self.subTest(shape=shape, axis=axis, keepdim=keepdim):
                        x = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape).requires_grad_()
                        a = self.tensor(x)
                        self.gradients([a], [x], a.sum_dim(axis, keepdim), x.sum(axis, keepdim=keepdim))
                        self.same(a.mean_dim(axis, keepdim), x.mean(axis, keepdim=keepdim))
        x = torch.tensor([2., 2., 1.], dtype=torch.float64, requires_grad=True)
        a = self.tensor(x)
        self.gradients([a], [x], a.max(), x.max())

    def test_functional_first_order_typed_seeds_zeros_and_mode(self):
        x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        a = self.tensor(x)
        unused = self.tensor(0.2)
        y, b = x.square(), a.square()
        seed = torch.tensor([0.2, 0.4], dtype=torch.float64)
        y.sum().backward(); b.sum().backward()
        before = a.grad.tolist()
        with fr.no_grad():
            gradients = ad.vjp(b, [a, unused], self.tensor(seed, False))
            self.assertFalse((a + 1.).requires_grad)
        expected = torch.autograd.grad(x.square(), [x], seed)[0]
        self.same(gradients[0], expected)
        self.same(gradients[1], torch.tensor(0., dtype=torch.float64))
        self.assertEqual(a.grad.tolist(), before)
        self.assertFalse(gradients[0].requires_grad)
        with self.assertRaises((ValueError, TypeError)):
            ad.grad(b.sum(), [a], create_graph=True)
        with self.assertRaises((ValueError, TypeError)):
            b.backward_with(fr.ones(2))
        self.assertEqual(a.grad.tolist(), before)

    def test_integer_arithmetic_exact_and_errors_explicit(self):
        values = [(1 << 63) - 1, -(1 << 63), (1 << 53) + 1]
        actual, expected = fr.tensor(values), torch.tensor(values)
        for a, b in [(actual + 1, expected + 1), (actual - 1, expected - 1),
                     (actual * actual, expected * expected), (-actual, -expected),
                     (actual.sum(), expected.sum()), (actual.max(), expected.max())]:
            self.same(a, b, atol=0)
        double = self.tensor([1., 2.])
        for operation in [lambda: double.bmm(double), lambda: double @ self.tensor(1.),
                          lambda: double @ fr.ones(2)]:
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                operation()
        with self.assertRaises((ValueError, TypeError)):
            fr.capture(lambda: double + double)
        self.assertIsNone(double.grad)
        parameter = fr.nn.Parameter(double)
        reference = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
        self.same(parameter, reference, atol=0)
        self.assertTrue(parameter.is_leaf)
        self.assertEqual(parameter._id(), parameter.tensor()._id())
        self.assertFalse(parameter._shares_storage(double))
        parameter.square().sum().backward(); reference.square().sum().backward()
        self.same(parameter.grad, reference.grad, atol=0)
        self.assertIsNone(double.grad)
        self.assertEqual(double.tolist(), [1., 2.])

    def test_float64_random_full_precision_generator_and_leaf_gradients(self):
        for factory in [fr.rand, fr.randn]:
            with self.subTest(factory=factory.__name__):
                generator = fr.Generator(17)
                first = factory((3, 4), generator=generator, dtype='float64', requires_grad=True)
                second = factory((3, 4), generator=generator, dtype='float64')
                self.assertEqual(first.dtype, 'float64')
                self.assertNotEqual(first.tolist(), second.tolist())
                values = torch.tensor(first.tolist(), dtype=torch.float64)
                self.assertTrue(bool((values != values.float().double()).any()))
                generator.manual_seed(17)
                self.assertEqual(first.tolist(), factory(3, 4, generator=generator, dtype='float64').tolist())
                self.assertEqual(first.tolist(), factory(3, 4, seed=17, dtype='float64').tolist())
                first.sum().backward()
                self.same(first.grad, torch.ones(3, 4, dtype=torch.float64), atol=0)
                generator.manual_seed(17)
                empty = factory(2, 0, 3, generator=generator, dtype='float64')
                self.assertEqual(list(empty.shape), [2, 0, 3])
                self.assertEqual(first.tolist(), factory(3, 4, generator=generator, dtype='float64').tolist())
                with self.assertRaises(ValueError):
                    factory(1, dtype='float64', device='cuda')

    def test_materialization_errors_precede_reads(self):
        size = ((1 << 63) - 1) // 8 + 1
        huge = self.tensor(1.).expand(size)
        for operation in [lambda: huge.index_select(0, []), lambda: huge.gather(0, fr.tensor([], dtype='int64')),
                          lambda: fr.cat([huge])]:
            with self.assertRaises(ValueError):
                operation()
        identity = huge.reshape([size])
        self.assertEqual(identity.stride(), huge.stride())
        self.assertTrue(identity._shares_storage(huge))
        oversized = self.tensor(1.).expand((1 << 63) - 1)
        for operation in [oversized.exp, oversized.log, lambda: oversized.pow(2.)]:
            with self.assertRaises((ValueError, TypeError)):
                operation()


if __name__ == '__main__':
    unittest.main()
