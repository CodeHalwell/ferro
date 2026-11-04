"""Functional tensor shapes, reductions and matrix products against PyTorch."""
import math
import unittest

import torch
import ferro as fr


class FunctionalTensorOps(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 1e-11 if expected.dtype == torch.float64 else 3e-5
        torch.testing.assert_close(values, expected.detach(), rtol=tolerance, atol=tolerance / 10, equal_nan=True)

    def leaf(self, shape, offset=0):
        data = [math.sin(i + offset) * .7 for i in range(math.prod(shape))]
        return (fr.Tensor(data, list(shape)).requires_grad_(True),
                torch.tensor(data).reshape(shape).requires_grad_())

    def backward(self, output, reference, actual_inputs, expected_inputs):
        self.same(output, reference)
        seed = torch.arange(1, reference.numel() + 1, dtype=reference.dtype).reshape(reference.shape) / 13
        output.backward_with(fr.tensor(seed.tolist(), dtype=str(seed.dtype).removeprefix('torch.')).reshape(list(seed.shape)))
        reference.backward(seed)
        for actual, expected in zip(actual_inputs, expected_inputs):
            self.same(actual.grad, expected.grad)

    def test_matmul_vectors_matrices_batches_broadcast_and_empty_gradients(self):
        cases = [((4,), (4,)), ((4,), (4, 3)), ((2, 4), (4,)), ((2, 4), (4, 3)),
                 ((2, 3, 4), (2, 4, 5)), ((3, 4), (2, 4, 5)), ((2, 3, 4), (4, 5)),
                 ((4,), (2, 4, 3)), ((2, 3, 4), (4,)),
                 ((2, 1, 3, 4), (1, 5, 4, 2)), ((1, 3, 4), (2, 5, 4, 2)),
                 ((0,), (0,)), ((2, 0), (0, 3)), ((0, 4), (4, 3)),
                 ((0, 3, 4), (1, 4, 2)), ((2, 3, 0), (1, 0, 5))]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                a, ta = self.leaf(left)
                b, tb = self.leaf(right, 5)
                self.backward(fr.matmul(a, b), torch.matmul(ta, tb), [a, b], [ta, tb])

    def test_mm_bmm_and_strided_broadcast_matmul(self):
        for name, left, right in [('mm', (3, 4), (4, 2)), ('bmm', (2, 3, 4), (2, 4, 2))]:
            a, ta = self.leaf(left)
            b, tb = self.leaf(right, 2)
            self.backward(getattr(fr, name)(a, b), getattr(torch, name)(ta, tb), [a, b], [ta, tb])
        a, ta = self.leaf((2, 4, 3))
        b, tb = self.leaf((2, 5, 4))
        self.backward(fr.matmul(a.transpose(1, 2), b.transpose(1, 2)),
                      ta.transpose(1, 2) @ tb.transpose(1, 2), [a, b], [ta, tb])

    def test_tensor_matmul_operator_methods_and_reverse_match_functional(self):
        cases = [((4,), (4,)), ((4,), (2, 4, 3)), ((2, 3, 4), (4,)),
                 ((2, 1, 3, 4), (1, 5, 4, 2)), ((2, 4), (4, 3)),
                 ((0,), (0,)), ((0, 3, 4), (1, 4, 2)), ((2, 3, 0), (1, 0, 5))]
        for dtype in ('float32', 'float64'):
            for method in ('operator', 'method', 'reverse'):
                for left, right in cases:
                    with self.subTest(dtype=dtype, method=method, left=left, right=right):
                        a, ta = self.leaf(left)
                        b, tb = self.leaf(right, 5)
                        av, bv = a.to(dtype=dtype), b.to(dtype=dtype)
                        ar, br = ta.to(getattr(torch, dtype)), tb.to(getattr(torch, dtype))
                        actual = av @ bv if method == 'operator' else av.matmul(bv) if method == 'method' else bv.__rmatmul__(av)
                        self.backward(actual, ar @ br, [a, b], [ta, tb])
        with self.assertRaises(TypeError):
            fr.ones(2, 2) @ 2

    def test_multi_axis_reductions_scalars_empty_axes_and_gradients(self):
        for shape in [(2, 3, 4), (2, 0, 3), ()]:
            dims = [None, (), 0, -1] + ([(0, -1), (2, 1)] if len(shape) == 3 else [])
            for name in ('sum', 'mean'):
                for dim in dims:
                    for keepdim in (False, True):
                        with self.subTest(shape=shape, name=name, dim=dim, keepdim=keepdim):
                            a, ta = self.leaf(shape)
                            self.backward(getattr(fr, name)(a, dim, keepdim),
                                          getattr(torch, name)(ta, dim, keepdim), [a], [ta])
        a, ta = self.leaf((2, 3, 4))
        self.backward(fr.mean(a.transpose(0, 2), (0, 2)), ta.transpose(0, 2).mean((0, 2)), [a], [ta])

    def test_shape_composition_and_inferred_reshape_gradients(self):
        a, ta = self.leaf((2, 3, 4))
        actual = fr.reshape(fr.permute(a, (2, 0, 1)), (4, -1))
        expected = torch.reshape(torch.permute(ta, (2, 0, 1)), (4, -1))
        actual = fr.squeeze(fr.unsqueeze(actual, 1), (0, 1))
        expected = torch.squeeze(torch.unsqueeze(expected, 1), (0, 1))
        self.backward(fr.flatten(fr.transpose(actual, 0, 1)),
                      torch.flatten(torch.transpose(expected, 0, 1)), [a], [ta])
        for shape, target in [((0, 3), (-1, 3)), ((), (1,)), ((1,), ())]:
            a, ta = self.leaf(shape)
            self.backward(fr.reshape(a, target), torch.reshape(ta, target), [a], [ta])
        scalar = fr.tensor(2.)
        self.same(fr.transpose(scalar, -1, 0), torch.tensor(2.).transpose(-1, 0))
        self.same(fr.squeeze(scalar, (0,)), torch.tensor(2.).squeeze((0,)))

    def test_partition_and_concat_gradients(self):
        for name, option in [('split', 2), ('split', [1, 0, 4]), ('chunk', 3), ('chunk', 8), ('unbind', None)]:
            with self.subTest(name=name, option=option):
                a, ta = self.leaf((2, 5))
                args = () if option is None else (option,)
                parts = getattr(fr, name)(a, *args, dim=-1)
                expected = getattr(torch, name)(ta, *args, dim=-1)
                self.assertIsInstance(parts, tuple)
                self.assertEqual(len(parts), len(expected))
                for part, ref in zip(parts, expected):
                    self.same(part, ref)
                loss = fr.sum(parts[0])
                reference = expected[0].sum()
                for i, (part, ref) in enumerate(zip(parts[1:], expected[1:]), 2):
                    loss = loss + fr.sum(part) * i
                    reference = reference + ref.sum() * i
                self.backward(loss, reference, [a], [ta])
        a, ta = self.leaf((2, 3))
        self.backward(fr.concat([a, a], dim=0), torch.concat([ta, ta]), [a], [ta])
        for name, option in [('split', 0), ('split', 3), ('chunk', 3), ('unbind', None)]:
            args = () if option is None else (option,)
            self.assertEqual(len(getattr(fr, name)(fr.zeros(2, 0), *args, dim=1)),
                             len(getattr(torch, name)(torch.zeros(2, 0), *args, dim=1)))

    def test_typed_shapes_and_like_factories(self):
        for dtype, data in [('int64', [[(1 << 53) + 1, 2], [3, 4]]),
                            ('float64', [[1.0000000000000002, 2.], [3., 1e200]])]:
            a = fr.tensor(data, dtype=dtype)
            ta = torch.tensor(data, dtype=getattr(torch, dtype))
            self.same(fr.reshape(fr.transpose(a, 0, 1), (-1,)), ta.T.reshape(-1))
            self.same(fr.cat([a, a], -1), torch.cat([ta, ta], -1))
            for name in ('zeros_like', 'ones_like'):
                self.same(getattr(fr, name)(a), getattr(torch, name)(ta))
            self.same(fr.full_like(a, 7), torch.full_like(ta, 7))
        a = fr.ones(2, 3, requires_grad=True)
        for name in ('zeros_like', 'ones_like', 'rand_like', 'randn_like'):
            output = getattr(fr, name)(a)
            self.assertEqual(output.shape, a.shape)
            self.assertFalse(output.requires_grad)
        output = fr.ones_like(a, requires_grad=True)
        output.sum().backward()
        self.assertIsNone(a.grad)
        self.assertEqual(output.grad.tolist(), [[1., 1., 1.], [1., 1., 1.]])
        self.assertTrue(fr.is_tensor(a))
        self.assertFalse(fr.is_tensor([1.]))
        self.assertEqual(fr.numel(a), 6)

    def test_unbind_preserves_views_and_strided_gradients(self):
        a = fr.tensor([[1., 2.], [3., 4.]])
        ta = torch.tensor([[1., 2.], [3., 4.]])
        parts, refs = fr.unbind(a), torch.unbind(ta)
        a.copy_(fr.tensor([[5., 6.], [7., 8.]]))
        ta.copy_(torch.tensor([[5., 6.], [7., 8.]]))
        for part, ref in zip(parts, refs):
            self.same(part, ref)
        a, ta = self.leaf((2, 3))
        parts, refs = fr.unbind(a.transpose(0, 1)), torch.unbind(ta.T)
        self.backward(parts[0] * 2 + parts[2] * 3, refs[0] * 2 + refs[2] * 3, [a], [ta])

    def test_invalid_shapes_axes_arithmetic_dtypes_and_broadcasts(self):
        a = fr.ones(2, 3)
        for shape in [(-1, -1), (-2, 3), (4,), (True, 6)]:
            with self.assertRaises((ValueError, TypeError)):
                fr.reshape(a, shape)
        with self.assertRaises(ValueError):
            fr.reshape(fr.zeros(0, 3), (0, -1))
        for dim in [(0, 0), 2, -3]:
            with self.assertRaises(ValueError):
                fr.sum(a, dim)
        self.same(fr.sum(fr.ones(3, dtype='int64')), torch.ones(3, dtype=torch.int64).sum())
        with self.assertRaises(NotImplementedError):
            fr.mean(fr.ones(3, dtype='int64'))
        for fn, ref in ((fr.sum, torch.sum), (fr.mean, torch.mean)):
            self.same(fn(a, dtype='float64'), ref(torch.ones(2, 3), dtype=torch.float64))
            with self.assertRaises(NotImplementedError):
                fn(a, dtype='complex128')
        for left, right in [((), (2,)), ((2, 3), (2, 4)), ((2, 3, 4), (3, 4, 2))]:
            with self.assertRaises(ValueError):
                fr.matmul(fr.ones(left), fr.ones(right))
        with self.assertRaises(ValueError):
            fr.mm(fr.ones(3), fr.ones(3))
        with self.assertRaises(ValueError):
            fr.bmm(a, a)

    def test_reduction_dtype_conversions_and_original_gradient_dtype(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            for target in ('float32', 'float64', 'int64', 'bool'):
                for name in ('sum', 'mean'):
                    if name == 'mean' and target not in ('float32', 'float64'):
                        continue
                    with self.subTest(dtype=dtype, target=target, name=name):
                        raw = [[1, 0, 2], [0, 1, 0]]
                        reference = torch.tensor(raw, dtype=getattr(torch, dtype))
                        actual = fr.tensor(raw, dtype=dtype)
                        self.same(getattr(fr, name)(actual, dim=0, keepdim=True, dtype=target),
                                  getattr(torch, name)(reference, dim=0, keepdim=True, dtype=getattr(torch, target)))
        for source, target in (('float32', 'float64'), ('float64', 'float32')):
            actual = fr.tensor([[.2, .8], [-.7, 1.3]], dtype=source, requires_grad=True)
            reference = torch.tensor(actual.tolist(), dtype=getattr(torch, source), requires_grad=True)
            result = fr.mean(actual, dim=1, dtype=target).sum()
            expected = reference.mean(dim=1, dtype=getattr(torch, target)).sum()
            result.backward()
            expected.backward()
            self.same(result, expected)
            self.same(actual.grad, reference.grad)


if __name__ == '__main__':
    unittest.main()
