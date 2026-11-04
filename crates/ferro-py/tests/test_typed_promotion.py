import itertools
import operator
import unittest

import ferro as fr
import ferro.autograd as ad
import torch


class TypedPromotion(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(list(actual.shape), list(expected.shape))
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 2e-6 if expected.dtype == torch.float32 else 1e-12
        torch.testing.assert_close(got, expected.detach(), rtol=tolerance, atol=tolerance, equal_nan=True)

    def test_tensor_promotion_grid_including_scalar_dimensions(self):
        types = ['bool', 'int64', 'float32', 'float64']
        operations = [operator.add, operator.sub, operator.mul, operator.truediv]
        for da, db, sa, sb, operation in itertools.product(types, types, [(), (2,)], [(), (2,)], operations):
            with self.subTest(da=da, db=db, sa=sa, sb=sb, op=operation.__name__):
                av, bv = (True if da == 'bool' else 3), (True if db == 'bool' else 2)
                a = fr.full(sa, av, dtype=da)
                b = fr.full(sb, bv, dtype=db)
                x = torch.full(sa, av, dtype=getattr(torch, da))
                y = torch.full(sb, bv, dtype=getattr(torch, db))
                if operation is operator.sub and 'bool' in (da, db):
                    with self.assertRaises((TypeError, ValueError)):
                        operation(a, b)
                    continue
                self.same(operation(a, b), operation(x, y))

    def test_mixed_comparison_grid_and_scalar_precision(self):
        operations = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
        for da, db, sa, sb, operation in itertools.product(
                ['bool', 'int64', 'float32', 'float64'], ['bool', 'int64', 'float32', 'float64'],
                [(), (2,)], [(), (2,)], operations):
            with self.subTest(da=da, db=db, sa=sa, sb=sb, op=operation.__name__):
                a, b = fr.full(sa, 1, dtype=da), fr.full(sb, 1.0000000000001, dtype=db)
                x = torch.full(sa, 1, dtype=getattr(torch, da))
                y = torch.full(sb, 1.0000000000001, dtype=getattr(torch, db))
                self.same(operation(a, b), operation(x, y))
        a = fr.tensor([(1 << 24) + 1], dtype='int64')
        x = torch.tensor([(1 << 24) + 1], dtype=torch.int64)
        for scalar, reference in [(float(1 << 24), float(1 << 24)),
                                  (fr.tensor(float(1 << 24), dtype='float64'), torch.tensor(float(1 << 24), dtype=torch.float64))]:
            for operation in operations:
                self.same(operation(a, scalar), operation(x, reference))
        for dtype in ['bool', 'int64', 'float32', 'float64']:
            for scalar in [True, 2, 0.5]:
                for operation in operations:
                    self.same(operation(fr.ones(2, dtype=dtype), scalar), operation(torch.ones(2, dtype=getattr(torch, dtype)), scalar))
        source = fr.tensor([float('nan'), float('inf'), 1e-300], dtype='float64', requires_grad=True)
        reference = torch.tensor([float('nan'), float('inf'), 1e-300], dtype=torch.float64)
        for operation in operations:
            result = operation(source, fr.tensor(0., dtype='float32'))
            self.assertFalse(result.requires_grad)
            self.same(result, operation(reference, torch.tensor(0., dtype=torch.float32)))
        self.same(fr.zeros((0, 2), dtype='float64') < fr.ones(2, dtype='int64'),
                  torch.zeros((0, 2), dtype=torch.float64) < torch.ones(2, dtype=torch.int64))

    def test_typed_batched_triangles_and_gradients(self):
        for dtype, values in [('float64', [1e100, 1.0000000000000002, 1e-200, -3., 4., 5., 6., 7.]),
                              ('int64', [-(1 << 63), (1 << 63) - 1, (1 << 53) + 1, -7, 8, 9, 10, 11]),
                              ('bool', [True, False, True, True, False, True, True, False]),
                              ('float32', [1., 2., 3., 4., 5., 6., 7., 8.])]:
            for method, diagonal in itertools.product(['triu', 'tril'], [-3, -1, 0, 1, 3]):
                with self.subTest(dtype=dtype, method=method, diagonal=diagonal):
                    a = fr.tensor(values, dtype=dtype).reshape([2, 2, 2])
                    x = torch.tensor(values, dtype=getattr(torch, dtype)).reshape([2, 2, 2])
                    if dtype in ('float32', 'float64'):
                        a = a.requires_grad_(True); x.requires_grad_(True)
                    actual, expected = getattr(a, method)(diagonal), getattr(x, method)(diagonal)
                    self.same(actual, expected)
                    if a.requires_grad:
                        seed = fr.tensor([.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.], dtype=dtype).reshape([2, 2, 2])
                        (actual * seed).sum().backward()
                        expected.backward(torch.tensor(seed.tolist(), dtype=x.dtype))
                        self.same(a.grad, x.grad)
        for shape in [(2, 0, 3), (0, 2, 3), (2, 3, 0)]:
            for method in ['triu', 'tril']:
                self.same(getattr(fr.zeros(shape, dtype='float64'), method)(),
                          getattr(torch.zeros(shape, dtype=torch.float64), method)())

    def test_python_scalar_promotion_and_reverse_operations(self):
        for dtype, scalar, operation in itertools.product(['bool', 'int64', 'float32', 'float64'],
                                                          [True, 2, 0.1234567890123456],
                                                          [operator.add, operator.sub, operator.mul, operator.truediv]):
            with self.subTest(dtype=dtype, scalar=scalar, op=operation.__name__):
                a = fr.full((2,), 3, dtype=dtype)
                x = torch.full((2,), 3, dtype=getattr(torch, dtype))
                if operation is operator.sub and (dtype == 'bool' or type(scalar) is bool):
                    with self.assertRaises((TypeError, ValueError)):
                        operation(a, scalar)
                    continue
                self.same(operation(a, scalar), operation(x, scalar))
                self.same(operation(scalar, a), operation(scalar, x))

    def test_mixed_gradients_and_functional_vjp_keep_input_dtypes(self):
        for xs, ys in [((2, 1), (1, 2)), ((2,), ()), ((), (2,))]:
            for xd, yd in [('float32', 'float64'), ('float64', 'float32')]:
                with self.subTest(xs=xs, ys=ys, xd=xd, yd=yd):
                    x = torch.full(xs, 0.25, dtype=getattr(torch, xd), requires_grad=True)
                    y = torch.full(ys, 1.23456789012345, dtype=getattr(torch, yd), requires_grad=True)
                    a = fr.full(xs, 0.25, dtype=xd, requires_grad=True)
                    b = fr.full(ys, 1.23456789012345, dtype=yd, requires_grad=True)
                    actual = (a * b + a / b).sum()
                    expected = (x * y + x / y).sum()
                    self.same(actual, expected)
                    native_grads = ad.grad(actual, [a, b])
                    torch_grads = torch.autograd.grad(expected, [x, y], retain_graph=True)
                    self.assertIsNone(a.grad)
                    self.assertIsNone(b.grad)
                    for got, reference in zip(native_grads, torch_grads):
                        self.same(got, reference)
                    actual.backward(); expected.backward()
                    self.same(a.grad, x.grad)
                    self.same(b.grad, y.grad)

    def test_canonical_casts_identity_copy_and_as_tensor_history(self):
        x = torch.tensor([0.2, 0.4], dtype=torch.float32, requires_grad=True)
        a = fr.tensor([0.2, 0.4], requires_grad=True)
        self.assertIs(a.to(dtype='float32'), a)
        self.assertIs(a.to('cpu'), a)
        self.assertIs(a.float(), a)
        b = a.double()
        self.assertTrue(b.requires_grad)
        self.assertTrue(b._depends_on(a))
        self.assertIs(b.double(), b)
        c = fr.as_tensor(b, dtype='float32')
        self.assertTrue(c.requires_grad)
        loss = c.double().square().sum()
        expected = x.double().float().double().square().sum()
        loss.backward(); expected.backward()
        self.same(a.grad, x.grad)
        self.assertFalse(a.long().requires_grad)
        self.same(a.long(), x.long())
        self.same(a.to(fr.ones(2, dtype='float64')), x.to(torch.ones(2, dtype=torch.float64)))
        original = fr.tensor([2., 3.])
        copy = original.to(copy=True)
        self.assertIsNot(copy, original)
        original.copy_(fr.zeros(2))
        self.assertEqual(copy.tolist(), [2., 3.])
        self.assertIs(fr.as_tensor(b), b)
        with self.assertRaises((TypeError, ValueError)):
            a.to('float64', dtype='float32')
        for transfer in [lambda: a.to('cuda'), a.cuda]:
            with self.assertRaisesRegex(ValueError, 'differentiable device transfers'):
                transfer()

    def test_casts_need_no_input_values_and_integer_cast_limits_are_explicit(self):
        source = fr.tensor([0.2, 0.4], requires_grad=True)
        reference = torch.tensor([0.2, 0.4], requires_grad=True)
        converted = source.double()
        expected = reference.double()
        with fr.no_grad():
            source.copy_(fr.ones(2))
        with torch.no_grad():
            reference.fill_(1.)
        self.assertEqual(converted.tolist(), [torch.tensor(0.2).item(), torch.tensor(0.4).item()])
        converted.sum().backward();expected.sum().backward()
        self.same(source.grad, reference.grad)
        nonlinear = (converted * converted).sum()
        oracle = (expected * expected).sum()
        converted.detach().copy_(fr.zeros(2, dtype='float64'))
        expected.detach().zero_()
        with self.assertRaises(ValueError):
            nonlinear.backward()
        with self.assertRaises(RuntimeError):
            oracle.backward()
        for value in [float('nan'), float('inf'), float(1 << 63)]:
            with self.assertRaises(ValueError):
                fr.tensor(value, dtype='float64').long()
        for dtype in ['float32', 'float64']:
            self.same(fr.tensor([-1.9, 2.9], dtype=dtype).long(), torch.tensor([-1.9, 2.9], dtype=getattr(torch, dtype)).long())

    def test_double_dropout_exact_scale_and_gradients(self):
        p = 0.1234567890123456
        x = fr.ones(256, dtype='float64', requires_grad=True)
        y = fr.nn.functional.dropout(x, p=p, training=True, seed=19, offset=7)
        scale = 1.0 / (1.0 - p)
        values = y.tolist()
        self.assertIn(0., values)
        self.assertIn(scale, values)
        self.assertTrue(all(value in (0., scale) for value in values))
        expected_input = torch.ones(256, dtype=torch.float64, requires_grad=True)
        expected = expected_input * torch.tensor(values, dtype=torch.float64)
        y.sum().backward(); expected.sum().backward()
        self.same(x.grad, expected_input.grad)
        self.assertEqual(y.tolist(), x.dropout(p, True, 19, 7).tolist())
        self.assertNotEqual(y.tolist(), x.dropout(p, True, 19, 8).tolist())
        self.assertIs(fr.nn.functional.dropout(x, p=p, training=False), x)
        self.same(fr.nn.functional.dropout(x, p=1.), torch.zeros(256, dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
