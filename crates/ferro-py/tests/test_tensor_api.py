"""Public factory/metadata contracts compared with real PyTorch."""
import math
import sys
import unittest

import torch
import ferro as fr


class TensorApi(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.ndim, expected.ndim)
        self.assertEqual(actual.dim(), expected.dim())
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(actual.size(), tuple(expected.size()))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=0, atol=0, equal_nan=True)

    def test_inference_scalars_nested_empty_and_legacy_constructor(self):
        cases = [3, 1.25, -0., [1, 2], (1, 2.5), [[1, 2], [3, 4]],
                 [], [[], []], [[[], []], [[], []]]]
        for data in cases:
            with self.subTest(data=data):
                self.same(fr.tensor(data), torch.tensor(data))
        self.same(fr.Tensor([1, 2], [2]), torch.tensor([1., 2.]))
        self.assertEqual(fr.Tensor([1.], []).item(), 1.)
        self.assertIs(type(fr.Tensor([1.], []).item()), float)

    def test_exact_typed_item_values_and_views(self):
        integers = [-(1 << 63), -(1 << 63) + 1, (1 << 53) + 1, (1 << 63) - 1]
        doubles = [1.0000000000000002, 1e300, 1e-300, -0.]
        for dtype, values, scalar_type in [('int64', integers, int), ('float64', doubles, float)]:
            tensor = fr.tensor([values[:2], values[2:]], dtype=dtype)
            expected = torch.tensor([values[:2], values[2:]], dtype=getattr(torch, dtype))
            self.same(tensor.transpose(0, 1), expected.T)
            for i, value in enumerate(values):
                scalar = tensor.narrow(0, i // 2, 1).narrow(1, i % 2, 1)
                self.assertIs(type(scalar.item()), scalar_type)
                self.assertEqual(scalar.item(), value)
                self.assertIn(str(value) if dtype == 'int64' else 'float64', repr(scalar))
            self.same(fr.tensor(tensor.transpose(0, 1)), expected.T)
        self.assertEqual(math.copysign(1., fr.tensor(-0., dtype='float64').item()), -1.)
        for value in [float('inf'), float('-inf'), float('nan')]:
            self.same(fr.tensor(value, dtype='float64'), torch.tensor(value, dtype=torch.float64))
        with self.assertRaises(ValueError):
            fr.tensor([]).item()
        with self.assertRaises(ValueError):
            fr.tensor([1, 2]).item()

    def test_size_len_negative_dimensions_and_empty_metadata(self):
        for shape in [(), (2, 3), (2, 0, 3), (0, 4)]:
            for dtype in ('float32', 'float64', 'int64'):
                actual = fr.zeros(shape, dtype=dtype)
                expected = torch.zeros(shape, dtype=getattr(torch, dtype))
                self.same(actual, expected)
                for dim in range(-len(shape), len(shape)):
                    self.assertEqual(actual.size(dim), expected.size(dim))
                for dim in (-len(shape) - 1, len(shape)):
                    with self.assertRaises(ValueError):
                        actual.size(dim)
                if shape:
                    self.assertEqual(len(actual), len(expected))
                else:
                    with self.assertRaises(TypeError):
                        len(actual)

    def test_truth_values_are_typed_and_reject_ambiguous_tensors(self):
        cases = [('float32', [0., -0., 2., float('nan')]),
                 ('float64', [0., -0., 1e-300, -1e-300, float('nan')]),
                 ('int64', [0, -(1 << 63), (1 << 63) - 1])]
        for dtype, values in cases:
            for value in values:
                for data in (value, [value], [[value]]):
                    with self.subTest(dtype=dtype, data=data):
                        self.assertEqual(bool(fr.tensor(data, dtype=dtype)),
                                         bool(torch.tensor(data, dtype=getattr(torch, dtype))))
            for shape in [(0,), (2,), (1, 0)]:
                with self.assertRaises(ValueError):
                    bool(fr.zeros(shape, dtype=dtype))

    def test_normalize_zero_tiny_boundary_and_overflow_gradients(self):
        with self.assertRaisesRegex(ValueError, 'eps must not be NaN'):
            fr.zeros(2, 2).normalize(1, float('nan'))
        cases = [([[0., 0.], [3., 4.]], 1e-12),
                 ([[0., 0.]], 1e-30), ([[1e-22, 1e-22]], 1e-30),
                 ([[.5, 0.]], .5), ([[1e20, -1e20], [3e38, 3e38]], 1e-12)]
        for data, eps in cases:
            with self.subTest(data=data, eps=eps):
                actual = fr.tensor(data, requires_grad=True)
                expected = torch.tensor(data, requires_grad=True)
                output = actual.normalize(1, eps)
                reference = torch.nn.functional.normalize(expected, dim=1, eps=eps)
                seed = torch.arange(1, expected.numel() + 1, dtype=torch.float32).reshape(expected.shape)
                output.backward_with(fr.tensor(seed.tolist()))
                reference.backward(seed)
                torch.testing.assert_close(torch.tensor(output.tolist()), reference.detach(), rtol=2e-5, atol=1e-6)
                torch.testing.assert_close(torch.tensor(actual.grad.tolist()), expected.grad, rtol=2e-5, atol=1e-6)
                self.assertTrue(bool(torch.isfinite(torch.tensor(actual.grad.tolist())).all()))

    def test_constant_and_eye_factories(self):
        for dtype in ('float32', 'float64', 'int64'):
            options = {'dtype': dtype}
            torch_options = {'dtype': getattr(torch, dtype)}
            for shape in [(), (2, 3), (0, 3), (2, 0, 3)]:
                allocated = fr.empty(shape, **options)
                self.same(allocated.fill_(3), torch.empty(shape, **torch_options).fill_(3))
                self.same(fr.zeros(shape, **options), torch.zeros(shape, **torch_options))
                self.same(fr.ones(shape, **options), torch.ones(shape, **torch_options))
                self.same(fr.full(shape, 3, **options), torch.full(shape, 3, **torch_options))
            for n, m in [(3, None), (2, 4), (0, 3), (3, 0)]:
                args = (n,) if m is None else (n, m)
                self.same(fr.eye(*args, **options), torch.eye(*args, **torch_options))
        self.same(fr.full((2,), (1 << 53) + 1), torch.full((2,), (1 << 53) + 1))
        self.same(fr.full((2,), 1.0000000000000002, dtype='float64'),
                  torch.full((2,), 1.0000000000000002, dtype=torch.float64))
        self.same(fr.zeros(2, 3), torch.zeros(2, 3))
        self.same(fr.ones(2, 3), torch.ones(2, 3))

    def test_arange_inference_precision_direction_and_integer_casts(self):
        for args in [(5,), (3, 3), (5, 0, -2), (-3, 4, 2),
                     (1.5, 5.5, 1), (1.9, 5.1, 1.1), (5.9, 1.1, -1.1)]:
            for dtype in (None, 'int64', 'float64'):
                with self.subTest(args=args, dtype=dtype):
                    actual = fr.arange(*args, dtype=dtype)
                    expected = torch.arange(*args, dtype=None if dtype is None else getattr(torch, dtype))
                    if actual.dtype == 'int64':
                        self.same(actual, expected)
                    else:
                        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
                        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype), expected)
        start = (1 << 53) + 1
        self.same(fr.arange(start, start + 4), torch.arange(start, start + 4))
        for args in [(0, 2, 0), (0, 2, -1), (2, 0, 1), (0, 2, .5)]:
            with self.assertRaises(ValueError):
                fr.arange(*args, dtype='int64')

    def test_tensor_copy_and_as_tensor_identity_preserve_gradient_contracts(self):
        x = fr.tensor([2., 3.], requires_grad=True)
        y = x * x
        self.assertIs(fr.as_tensor(y), y)
        self.assertIs(fr.as_tensor(y, dtype='float32', device='cpu'), y)
        copied = fr.tensor(y, requires_grad=True)
        copied.sum().backward()
        self.assertEqual(copied.grad.tolist(), [1., 1.])
        self.assertIsNone(x.grad)
        fr.as_tensor(y).sum().backward()
        self.assertEqual(x.grad.tolist(), [4., 6.])
        with fr.no_grad():
            copied.copy_(fr.zeros(2))
        self.assertEqual(y.tolist(), [4., 9.])
        cast = fr.as_tensor(y, dtype='float64')
        self.assertTrue(cast.requires_grad)
        self.assertEqual(cast.dtype, 'float64')
        cast.sum().backward()
        self.assertEqual(x.grad.tolist(), [8., 12.])
        tx = torch.tensor([2., 3.], requires_grad=True)
        ty = tx * tx
        torch.as_tensor(ty, dtype=torch.float64).sum().backward()
        self.assertEqual(cast.grad.tolist(), [1., 1.])
        self.assertEqual([v / 2 for v in x.grad.tolist()], tx.grad.tolist())
        detached = fr.tensor(y)
        self.assertFalse(detached.requires_grad)
        converted = fr.as_tensor(detached, dtype='float64')
        self.assertEqual(converted.tolist(), [4., 9.])
        self.assertEqual(converted.dtype, 'float64')
        empty = fr.zeros(2, 0, 3, dtype='int64').transpose(0, 2)
        self.same(fr.tensor(empty), torch.zeros(2, 0, 3, dtype=torch.int64).transpose(0, 2))

    def test_factory_gradient_leaf_ownership(self):
        factories = [lambda: fr.zeros((2,), requires_grad=True),
                     lambda: fr.ones(2, requires_grad=True),
                     lambda: fr.full((2,), 2., requires_grad=True),
                     lambda: fr.arange(2., requires_grad=True),
                     lambda: fr.eye(2, requires_grad=True),
                     lambda: fr.rand(2, seed=7, requires_grad=True),
                     lambda: fr.randn(2, seed=7, requires_grad=True)]
        for factory in factories:
            tensor = factory()
            (tensor * 3).sum().backward()
            self.assertTrue(tensor.requires_grad)
            self.assertEqual(torch.tensor(tensor.grad.tolist()).flatten().tolist(), [3.] * tensor.numel())

    def test_random_shapes_stream_reproducibility_and_validation(self):
        for factory in (fr.rand, fr.randn):
            generator = fr.Generator(17)
            first = factory(2, 3, generator=generator)
            second = factory((2, 3), generator=generator)
            self.assertEqual(first.size(), (2, 3))
            self.assertEqual(first.dtype, 'float32')
            self.assertNotEqual(first.tolist(), second.tolist())
            generator.manual_seed(17)
            self.assertEqual(first.tolist(), factory(2, 3, generator=generator).tolist())
            self.assertEqual(first.tolist(), factory(2, 3, seed=17).tolist())
            self.assertEqual(factory(2, 0).size(), (2, 0))
            self.assertEqual(factory(()).size(), ())
            with self.assertRaises(ValueError):
                factory(1, seed=1, generator=generator)
            with self.assertRaises(TypeError):
                factory(1, dtype='int64')
        draws = torch.tensor(fr.rand(1024, seed=5).tolist())
        self.assertTrue(bool(((draws >= 0) & (draws < 1)).all()))

    def test_invalid_data_options_and_capacity_fail_before_allocation(self):
        cycle = []
        cycle.append(cycle)
        for data in ([[1], [2, 3]], [[], [1]], cycle):
            with self.assertRaises(ValueError):
                fr.tensor(data)
        for data in ([object()], {'x': 1}, '1'):
            with self.assertRaises(TypeError):
                fr.tensor(data)
        for dtype in ('float16', 'bfloat16', 'f32', torch.float32):
            with self.assertRaises(TypeError):
                fr.tensor([], dtype=dtype)
        for device in ('metal', 'cuda:-1', 'cuda:4294967296', 0, object()):
            with self.assertRaises(ValueError):
                fr.tensor([], device=device)
        for dtype in ('int64', 'bool'):
            with self.assertRaises(ValueError):
                fr.tensor([], dtype=dtype, requires_grad=True)
        with self.assertRaises(ValueError):
            fr.zeros(0, dtype='float64', device='cuda')
        for shape in [(-1,), (True,), (1.5,), (sys.maxsize, 2, 0), (0, sys.maxsize, 2), (sys.maxsize,)]:
            with self.assertRaises((TypeError, ValueError)):
                fr.zeros(shape)
        for factory in (fr.zeros, fr.ones, fr.rand, fr.randn):
            with self.assertRaises(TypeError):
                factory()
        for value in [-(1 << 63) - 1, 1 << 63]:
            with self.assertRaises(OverflowError):
                fr.tensor(value)
            with self.assertRaises(OverflowError):
                fr.full((0,), value)
        with self.assertRaises(TypeError):
            fr.zeros(1, layout='sparse')
        with self.assertRaises(TypeError):
            fr.tensor([1.], requires_grad=1)
        for shape in [(0,), (1,)]:
            with self.assertRaises(OverflowError):
                fr.full(shape, 1e300, dtype='float32')


if __name__ == '__main__':
    unittest.main()
