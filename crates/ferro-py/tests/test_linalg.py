"""Native CPU linear algebra versus actual Torch, including all input gradients."""
import sys
import math
import unittest
from itertools import permutations

import torch
import ferro as fr
from ferro import linalg


def native(value, grad=False):
    return fr.tensor(value.detach().flatten().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(grad)


class LinalgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = (3e-5, 2e-4) if expected.dtype == torch.float32 else (2e-12, 2e-11)
        torch.testing.assert_close(value, expected.detach(), atol=tolerance[0], rtol=tolerance[1], equal_nan=True)

    def gradients(self, actual, expected, inputs):
        self.close(actual, expected)
        probe = torch.randn(expected.shape, dtype=expected.dtype, generator=torch.Generator().manual_seed(971))
        actual.backward_with(native(probe))
        expected.backward(probe)
        for name, got, wanted in inputs:
            with self.subTest(gradient=name):
                self.assertIsNotNone(got.grad)
                self.assertIsNotNone(wanted.grad)
                self.close(got.grad, wanted.grad)

    def matrix(self, shape, dtype, *, symmetric=False):
        values = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(967))
        size = shape[-1]
        if symmetric:
            values = values @ values.transpose(-1, -2)
        return (values + torch.eye(size, dtype=dtype) * (size + 1)).requires_grad_(True)

    def test_solve_left_right_vector_matrix_and_broadcast_gradients(self):
        cases = (((3, 3), (3,), True), ((2, 3, 3), (2, 3), True),
                 ((2, 1, 3, 3), (1, 4, 3, 2), True), ((3, 3), (4, 3, 2), True),
                 ((2, 1, 3, 3), (1, 4, 2, 3), False), ((3, 3), (2, 3), False),
                 ((1, 3, 3), (2, 3, 2), True))
        for dtype in (torch.float32, torch.float64):
            for ashape, bshape, left in cases:
                with self.subTest(dtype=dtype, ashape=ashape, bshape=bshape, left=left):
                    ta = self.matrix(ashape, dtype)
                    tb = torch.randn(bshape, dtype=dtype, generator=torch.Generator().manual_seed(973), requires_grad=True)
                    sa, sb = native(ta, True), native(tb, True)
                    self.gradients(linalg.solve(sa, sb, left=left), torch.linalg.solve(ta, tb, left=left), [('A', sa, ta), ('B', sb, tb)])

    def test_inverse_batched_and_pivoting_values_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((3, 3), (2, 2, 2), (2, 1, 3, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    ta = self.matrix(shape, dtype)
                    sa = native(ta, True)
                    self.gradients(linalg.inv(sa), torch.linalg.inv(ta), [('A', sa, ta)])
            ta = torch.tensor([[0., 2., 1.], [3., 1., 2.], [1., 0., 4.]], dtype=dtype, requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.inv(sa), torch.linalg.inv(ta), [('pivoted A', sa, ta)])

    def test_triangular_options_broadcasts_and_all_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                for left in (False, True):
                    for unit in (False, True):
                        with self.subTest(dtype=dtype, upper=upper, left=left, unit=unit):
                            ta = self.matrix((2, 1, 3, 3), dtype)
                            shape = (1, 4, 3, 2) if left else (1, 4, 2, 3)
                            tb = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(977), requires_grad=True)
                            sa, sb = native(ta, True), native(tb, True)
                            options = dict(upper=upper, left=left, unitriangular=unit)
                            self.gradients(linalg.solve_triangular(sa, sb, **options), torch.linalg.solve_triangular(ta, tb, **options), [('A', sa, ta), ('B', sb, tb)])

    def test_triangular_ignored_entries_and_unit_diagonal_are_not_read(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.tensor([[float('nan'), .2], [float('nan'), float('nan')]], dtype=dtype, requires_grad=True)
            tb = torch.tensor([[1.], [2.]], dtype=dtype, requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            kwargs = dict(upper=True, unitriangular=True)
            self.gradients(linalg.solve_triangular(sa, sb, **kwargs), torch.linalg.solve_triangular(ta, tb, **kwargs), [('A', sa, ta), ('B', sb, tb)])
            self.assertEqual(sa.grad.tolist()[1], [0., 0.])
            self.assertEqual(sa.grad.tolist()[0][0], 0.)

    def test_cholesky_upper_lower_batches_and_full_symmetric_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((3, 3), (2, 3, 3), (2, 1, 2, 2)):
                for upper in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, upper=upper):
                        ta = self.matrix(shape, dtype, symmetric=True)
                        sa = native(ta, True)
                        self.gradients(linalg.cholesky(sa, upper=upper), torch.linalg.cholesky(ta, upper=upper), [('A', sa, ta)])
                        actual = linalg.cholesky_ex(native(ta), upper=upper)
                        expected = torch.linalg.cholesky_ex(ta.detach(), upper=upper)
                        self.close(actual.L, expected.L)
                        self.assertEqual(actual.info.dtype, 'int64')
                        self.assertEqual(actual.info.tolist(), expected.info.tolist())
                        self.assertFalse(actual.info.requires_grad)

    def test_cholesky_reads_selected_triangle_and_reports_failure_minor(self):
        for upper in (False, True):
            ta = torch.tensor([[4., 2.], [2., 3.]], dtype=torch.float64)
            ta[1 if upper else 0, 0 if upper else 1] = float('nan')
            self.close(linalg.cholesky(native(ta), upper=upper), torch.linalg.cholesky(ta, upper=upper))
        ta = torch.tensor([[[4., 2.], [2., 3.]], [[1., 2.], [2., 1.]], [[-1., 0.], [0., 1.]]], dtype=torch.float64)
        actual, expected = linalg.cholesky_ex(native(ta)), torch.linalg.cholesky_ex(ta)
        self.assertEqual(actual.info.tolist(), expected.info.tolist())
        self.close(actual.L[0], expected.L[0])
        for apply in (linalg.cholesky, lambda a: linalg.cholesky_ex(a, check_errors=True)):
            with self.assertRaisesRegex(ValueError, 'positive definite'):
                apply(native(ta))

    def test_strided_inputs_and_f64_values_beyond_f32_precision(self):
        ta = self.matrix((2, 3, 3), torch.float64)
        tb = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
        sa, sb = native(ta, True), native(tb, True)
        self.gradients(linalg.solve(sa.transpose(-1, -2), sb.transpose(-1, -2)),
                       torch.linalg.solve(ta.transpose(-1, -2), tb.transpose(-1, -2)), [('A', sa, ta), ('B', sb, tb)])
        ta = torch.tensor([[1. + 2. ** -40]], dtype=torch.float64, requires_grad=True)
        sa = native(ta, True)
        self.gradients(linalg.inv(sa), torch.linalg.inv(ta), [('precise A', sa, ta)])
        self.assertNotEqual(linalg.inv(native(ta)).tolist(), [[1.]])
        self.close(linalg.solve(fr.tensor([[1e-300]], dtype='float64'), fr.tensor([2e-300], dtype='float64')),
                   torch.tensor([2.], dtype=torch.float64))

    def test_empty_batches_zero_matrices_and_zero_rhs_preserve_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((0, 0), (3, 0, 0), (0, 2, 2)):
                for method in ('inv', 'cholesky'):
                    with self.subTest(dtype=dtype, shape=shape, method=method):
                        ta = torch.empty(shape, dtype=dtype, requires_grad=True)
                        sa = native(ta, True)
                        self.gradients(getattr(linalg, method)(sa), getattr(torch.linalg, method)(ta), [('A', sa, ta)])
            for ashape, bshape in (((3, 3), (3, 0)), ((0, 3, 3), (1, 3, 2)), ((2, 1, 0, 0), (1, 3, 0, 2))):
                with self.subTest(dtype=dtype, ashape=ashape, bshape=bshape):
                    ta = self.matrix(ashape, dtype)
                    tb = torch.empty(bshape, dtype=dtype, requires_grad=True)
                    sa, sb = native(ta, True), native(tb, True)
                    self.gradients(linalg.solve(sa, sb), torch.linalg.solve(ta, tb), [('A', sa, ta), ('B', sb, tb)])

        coefficient = fr.tensor([[2., 0.], [0., 3.]], dtype='float64', requires_grad=True)
        huge_empty = fr.tensor([], dtype='float64').reshape([1 << 60, 2, 0]).requires_grad_(True)
        empty_solution = linalg.solve(coefficient, huge_empty)
        self.assertEqual(empty_solution.shape, huge_empty.shape)
        empty_solution.sum().backward()
        self.assertEqual(coefficient.grad.tolist(), [[0., 0.], [0., 0.]])
        self.assertEqual(huge_empty.grad.shape, huge_empty.shape)

    def test_invalid_shapes_singular_matrices_dtype_flags_and_capacity(self):
        singular = fr.tensor([[1., 2.], [2., 4.]])
        for apply in (linalg.inv, lambda a: linalg.solve(a, fr.ones(2)), lambda a: linalg.solve(a, fr.ones(2, 0))):
            with self.assertRaisesRegex(ValueError, 'singular'):
                apply(singular)
        for a, b in ((fr.ones(2, 3), fr.ones(2)), (fr.ones(2, 2), fr.ones(3)),
                      (fr.ones(2, 2), fr.ones(2, dtype='float64')),
                      (fr.ones(2, 2, dtype='int64'), fr.ones(2, dtype='int64'))):
            with self.assertRaises((TypeError, ValueError)):
                linalg.solve(a, b)
        with self.assertRaises(ValueError):
            linalg.solve(fr.ones(2, 2), fr.ones(2), left=False)
        with self.assertRaises(ValueError):
            linalg.solve_triangular(fr.ones(2, 2), fr.ones(2), upper=False)
        with self.assertRaises(TypeError):
            linalg.cholesky(fr.eye(2), upper=1)
        with self.assertRaises(NotImplementedError):
            linalg.inv(fr.eye(2), out=fr.zeros(2, 2))
        with self.assertRaisesRegex(ValueError, 'finite'):
            linalg.inv(fr.tensor([[float('nan')]]))
        huge = fr.tensor(1., dtype='float64').expand(sys.maxsize // 8 + 1, 1, 1)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            linalg.inv(huge)

    def test_capture_higher_order_and_stale_parent_guards(self):
        from ferro.autograd import grad
        for dtype in ('float32', 'float64'):
            a = fr.tensor([[2., .1], [.2, 3.]], dtype=dtype, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'capture'):
                fr.capture(lambda: linalg.inv(a))
            with self.assertRaises(ValueError):
                grad(linalg.inv(a).sum(), [a], create_graph=True)
            result = linalg.inv(a).sum()
            with fr.no_grad():
                a.copy_(fr.tensor([[3., .1], [.2, 4.]], dtype=dtype))
            with self.assertRaises((ValueError, RuntimeError)):
                result.backward()
            self.assertIsNone(a.grad)

    def test_determinant_and_slogdet_batched_strided_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((1, 1), (3, 3), (2, 1, 3, 3), (0, 2, 2), (3, 0, 0)):
                for logarithmic in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, logarithmic=logarithmic):
                        ta = self.matrix(shape, dtype)
                        sa = native(ta, True)
                        if logarithmic:
                            actual, expected = linalg.slogdet(sa.transpose(-1, -2)), torch.linalg.slogdet(ta.transpose(-1, -2))
                            self.close(actual.sign, expected.sign)
                            self.gradients(actual.logabsdet, expected.logabsdet, [('A', sa, ta)])
                        else:
                            self.gradients(linalg.det(sa.transpose(-1, -2)), torch.linalg.det(ta.transpose(-1, -2)), [('A', sa, ta)])
            ta = torch.tensor([[0., 2., 1.], [3., 1., 2.], [1., 0., 4.]], dtype=dtype, requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.det(sa), torch.linalg.det(ta), [('pivoted A', sa, ta)])

    def test_singular_determinant_uses_polynomial_cofactors_and_scaled_extremes(self):
        def polynomial(a):
            n = a.shape[-1]
            result = a.sum() * 0
            for permutation in permutations(range(n)):
                term = a.new_tensor((-1) ** sum(permutation[i] > permutation[j] for i in range(n) for j in range(i + 1, n)))
                for row, column in enumerate(permutation):
                    term = term * a[row, column]
                result = result + term
            return result

        cases = ([[1., 2.], [2., 4.]], [[1., 2., 3.], [0., 2., 4.], [1., 4., 7.]],
                 [[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]], [[0.]])
        for dtype in (torch.float32, torch.float64):
            for values in cases:
                ta = torch.tensor(values, dtype=dtype, requires_grad=True)
                sa = native(ta, True)
                self.gradients(linalg.det(sa), polynomial(ta), [('polynomial A', sa, ta)])
        for scale in (1e-300, 1e300):
            a = fr.tensor([[scale, 0.], [0., scale]], dtype='float64', requires_grad=True)
            linalg.det(a).backward()
            self.assertEqual(a.grad.tolist(), [[scale, 0.], [0., scale]])
            expected = torch.linalg.slogdet(torch.tensor([[scale, 0.], [0., scale]], dtype=torch.float64))
            self.close(linalg.slogdet(a).logabsdet, expected.logabsdet)
        singular = fr.tensor([[1., 2.], [2., 4.]], dtype='float64', requires_grad=True)
        result = linalg.slogdet(singular)
        self.assertEqual(result.sign.item(), 0.)
        self.assertEqual(result.logabsdet.item(), -float('inf'))
        result.logabsdet.backward()
        self.assertTrue(torch.tensor(singular.grad.tolist()).isnan().all())

    def test_vector_norm_orders_dimensions_keepdim_and_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for order in (.5, 1, 2, 3, -1, -2, float('inf'), -float('inf')):
                for dims in (None, (), -1, (0, 2)):
                    for keepdim in (False, True):
                        with self.subTest(dtype=dtype, order=order, dims=dims, keepdim=keepdim):
                            ta = torch.randn(2, 3, 4, dtype=dtype, generator=torch.Generator().manual_seed(983), requires_grad=True)
                            sa = native(ta, True)
                            self.gradients(linalg.vector_norm(sa, order, dims, keepdim), torch.linalg.vector_norm(ta, order, dims, keepdim), [('input', sa, ta)])
        ta = torch.tensor([2., -2., 0.], dtype=torch.float64, requires_grad=True)
        sa = native(ta, True)
        self.gradients(linalg.vector_norm(sa, float('inf')), torch.linalg.vector_norm(ta, float('inf')), [('ties', sa, ta)])

    def test_vector_norm_empty_scalar_nonfinite_and_dtype_widening(self):
        for order in (.5, 1, 2, 3, -1, -2, float('inf'), -float('inf')):
            for values in ([0., 2.], [float('inf'), 2.], [float('nan'), 2.], [0., float('inf')], [0., float('nan')]):
                ta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                sa = native(ta, True)
                with self.subTest(order=order, values=values):
                    self.gradients(linalg.vector_norm(sa, order), torch.linalg.vector_norm(ta, order), [('nonfinite/zero', sa, ta)])
        for shape, dims in (((2, 0), -1), ((0, 2), -1), ((), 0), ((), ())):
            ta = torch.ones(shape, dtype=torch.float64, requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.vector_norm(sa, dim=dims, keepdim=True), torch.linalg.vector_norm(ta, dim=dims, keepdim=True), [('empty/scalar', sa, ta)])
        ta = torch.tensor([.2, -.6, 1.3], dtype=torch.float32, requires_grad=True)
        sa = native(ta, True)
        self.gradients(linalg.vector_norm(sa, dtype='float64'), torch.linalg.vector_norm(ta, dtype=torch.float64), [('widened input', sa, ta)])
        for scale in (1e-300, 1e300):
            result = linalg.vector_norm(fr.tensor([scale, scale], dtype='float64')).tolist()
            self.assertAlmostEqual(result / scale, 2 ** .5, places=14)

    def test_matrix_norms_axes_keepdim_and_norm_routing_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for order in ('fro', 1, -1, float('inf'), -float('inf')):
                for dims in ((-2, -1), (0, 2), (2, 0)):
                    for keepdim in (False, True):
                        ta = torch.randn(2, 3, 4, dtype=dtype, generator=torch.Generator().manual_seed(991), requires_grad=True)
                        sa = native(ta, True)
                        with self.subTest(dtype=dtype, order=order, dims=dims, keepdim=keepdim):
                            self.gradients(linalg.matrix_norm(sa, order, dims, keepdim), torch.linalg.matrix_norm(ta, order, dims, keepdim), [('matrix', sa, ta)])
        for order in (1, -1):
            ta = torch.tensor([[1.1, .3], [.4, 1.2]], dtype=torch.float64, requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.matrix_norm(sa, order), torch.linalg.matrix_norm(ta, order), [('equal column sums', sa, ta)])
        for shape, order, dims in (((2, 3, 4), None, None), ((2, 3), 1, None), ((3,), 3, None),
                                   ((2, 3, 4), None, (0, 2)), ((2, 3, 4), 2, -1), ((2, 3), 'fro', None)):
            ta = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(997), requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.norm(sa, order, dims), torch.linalg.norm(ta, order, dims), [('routed', sa, ta)])

    def test_determinant_norm_validation_and_graph_guards(self):
        for method in (linalg.det, linalg.slogdet):
            for value in (fr.ones(2, 3), fr.ones(2, 2, dtype='int64'), fr.tensor([[float('nan')]])):
                with self.assertRaises((ValueError, TypeError)):
                    method(value)
        for order in (0, 3, 'unsupported'):
            with self.assertRaises(ValueError):
                linalg.matrix_norm(fr.ones(2, 3), order)
        for dims in ((0, -1), 1):
            with self.assertRaises(ValueError):
                linalg.vector_norm(fr.ones(2), dim=dims)
        for dims in ((), (0, 1, 2)):
            with self.assertRaises(ValueError):
                linalg.norm(fr.ones(2, 3, 4), dim=dims)
        for order in (-1, float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                linalg.vector_norm(fr.zeros(2, 0), order, -1)
        with self.assertRaises(ValueError):
            linalg.vector_norm(fr.ones(2, dtype='float64'), dtype='float32')
        from ferro.autograd import grad
        for method in (linalg.det, lambda a: linalg.slogdet(a).logabsdet, linalg.vector_norm, linalg.matrix_norm):
            for dtype in ('float32', 'float64'):
                a = fr.tensor([[2., .1], [.2, 3.]], dtype=dtype, requires_grad=True)
                with self.assertRaisesRegex(ValueError, 'capture'):
                    fr.capture(lambda: method(a))
                with self.assertRaises(ValueError):
                    grad(method(a).sum(), [a], create_graph=True)
                result = method(a).sum()
                with fr.no_grad():
                    a.copy_(fr.eye(2, dtype=dtype))
                with self.assertRaises((ValueError, RuntimeError)):
                    result.backward()
                self.assertIsNone(a.grad)

    def test_diagonal_views_axes_offsets_strides_and_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape, dims in (((3, 4), (0, 1)), ((2, 3, 4), (-2, -1)), ((2, 3, 4), (2, 0)), ((2, 0, 3), (1, 2))):
                for offset in (-5, -1, 0, 2, 5):
                    with self.subTest(dtype=dtype, shape=shape, dims=dims, offset=offset):
                        ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1009), requires_grad=True)
                        sa = native(ta, True)
                        self.gradients(linalg.diagonal(sa, offset=offset, dim1=dims[0], dim2=dims[1]),
                                       torch.linalg.diagonal(ta, offset=offset, dim1=dims[0], dim2=dims[1]), [('diagonal', sa, ta)])
            ta = torch.randn(3, 4, 5, dtype=dtype, generator=torch.Generator().manual_seed(1013), requires_grad=True)
            sa = native(ta, True)
            actual = linalg.diagonal(sa.narrow(0, 1, 2).transpose(0, 2), offset=-1, dim1=0, dim2=1)
            expected = torch.linalg.diagonal(ta.narrow(0, 1, 2).transpose(0, 2), offset=-1, dim1=0, dim2=1)
            self.gradients(actual, expected, [('offset strided', sa, ta)])
            ta = torch.tensor([[.3, -.4, .7]], dtype=dtype, requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.diagonal(sa.expand(4, 3)), torch.linalg.diagonal(ta.expand(4, 3)), [('expanded alias', sa, ta)])

    def test_diagonal_alias_trace_public_methods_and_boundaries(self):
        for dtype in ('float32', 'float64'):
            original = fr.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype)
            view = linalg.diagonal(original, offset=1)
            original.copy_(fr.tensor([[7., 8., 9.], [10., 11., 12.]], dtype=dtype))
            self.assertEqual(view.tolist(), [8., 12.])
            self.assertEqual(original.diagonal(1).tolist(), view.tolist())
            self.assertEqual(fr.diagonal(original, 1).tolist(), view.tolist())
            ta = torch.tensor([[.2, -.4, .7], [1.2, .3, -.2]], dtype=getattr(torch, dtype), requires_grad=True)
            sa = native(ta, True)
            self.gradients(sa.trace(), ta.trace(), [('trace', sa, ta)])
            self.close(fr.trace(native(ta)), torch.trace(ta))
            leaf = fr.tensor([[1., 2.], [3., 4.]], dtype=dtype, requires_grad=True)
            saved = linalg.diagonal(leaf).sum()
            with fr.no_grad():
                leaf.copy_(fr.eye(2, dtype=dtype))
            with self.assertRaises((ValueError, RuntimeError)):
                saved.backward()
            self.assertIsNone(leaf.grad)
        huge_empty = fr.tensor([], dtype='float64').reshape([1 << 60, 0, 2]).requires_grad_(True)
        empty = linalg.diagonal(huge_empty, offset=-sys.maxsize-1)
        self.assertEqual(empty.shape, [1 << 60, 0])
        empty.sum().backward()
        self.assertEqual(huge_empty.grad.shape, huge_empty.shape)
        for options in (dict(dim1=0, dim2=-2), dict(dim2=2), dict(offset=True)):
            with self.assertRaises((ValueError, TypeError)):
                linalg.diagonal(fr.ones(2, 3), **options)
        with self.assertRaises(ValueError):
            linalg.diagonal(fr.ones(3))
        with self.assertRaises(ValueError):
            fr.ones(2, 3, 4).trace()
        with self.assertRaisesRegex(ValueError, 'capture'):
            fr.capture(lambda: linalg.diagonal(fr.ones(2, 3)))

    def test_sign_and_zero_norm_ordinary_unused_slots_and_functional_zeros(self):
        from ferro.autograd import grad
        for dtype in (torch.float32, torch.float64):
            for method in ('sign', 'count'):
                for shape in ((2, 3), (0, 3)):
                    if method == 'sign' and shape[0] == 0:
                        continue
                    ta = torch.tensor([[2., 0.], [0., -3.]], dtype=dtype, requires_grad=True) if method == 'sign' else torch.zeros(shape, dtype=dtype, requires_grad=True)
                    sa = native(ta, True)
                    actual = linalg.slogdet(sa).sign if method == 'sign' else linalg.vector_norm(sa, 0, dim=-1)
                    expected = torch.linalg.slogdet(ta).sign if method == 'sign' else torch.linalg.vector_norm(ta, 0, dim=-1)
                    self.close(actual, expected)
                    self.assertTrue(actual.requires_grad)
                    actual.sum().backward(); expected.sum().backward(retain_graph=True)
                    self.assertIsNone(sa.grad); self.assertIsNone(ta.grad)
                    self.assertEqual(grad(actual.sum(), [sa], allow_unused=True), (None,))
                    self.assertEqual(torch.autograd.grad(expected.sum(), [ta], allow_unused=True), (None,))
                    self.close(actual.sum().grad_wrt([sa])[0], torch.zeros_like(ta))
                    if dtype == torch.float32:
                        self.assertEqual(grad(actual.sum(), [sa], create_graph=True, allow_unused=True), (None,))
                        self.close(actual.sum().grad_wrt([sa], create_graph=True)[0], torch.zeros_like(ta))
                    self.assertIsNone(sa.grad)
            ta = torch.tensor([[float('nan'), 0., 2.], [0., -3., float('inf')]], dtype=dtype, requires_grad=True)
            sa = native(ta, True)
            self.close(linalg.vector_norm(sa, 0, dim=0, keepdim=True), torch.linalg.vector_norm(ta, 0, dim=0, keepdim=True))

    def test_reduced_qr_all_factor_values_and_separate_combined_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((5, 3), (3, 3), (4, 1), (2, 1, 5, 3), (1, 3), (2, 3), (3, 7), (2, 1, 3, 5), (3, 0), (0, 4, 2), (2, 0, 0), (0, 3)):
                for selected in ('Q', 'R', 'both'):
                    with self.subTest(dtype=dtype, shape=shape, selected=selected):
                        ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1021))
                        ta = (ta + torch.eye(shape[-2], shape[-1], dtype=dtype) * (shape[-1] + 1)).requires_grad_(True)
                        sa = native(ta, True)
                        actual, expected = linalg.qr(sa), torch.linalg.qr(ta)
                        self.close(actual.Q, expected.Q); self.close(actual.R, expected.R)
                        actual_loss, expected_loss = None, None
                        for name in ('Q', 'R') if selected == 'both' else (selected,):
                            value, target = getattr(actual, name), getattr(expected, name)
                            probe = torch.randn(target.shape, dtype=dtype, generator=torch.Generator().manual_seed(1031))
                            term, wanted = (value * native(probe)).sum(), (target * probe).sum()
                            actual_loss = term if actual_loss is None else actual_loss + term
                            expected_loss = wanted if expected_loss is None else expected_loss + wanted
                        actual_loss.backward(); expected_loss.backward()
                        self.close(sa.grad, ta.grad)
                        if min(shape[-2:]) and torch.tensor(shape[:-2]).prod() != 0:
                            self.close(actual.Q @ actual.R, ta)

    def test_qr_strides_full_rank_boundary_and_graph_guards(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.randn(2, 3, 5, dtype=dtype, generator=torch.Generator().manual_seed(1033), requires_grad=True)
            sa = native(ta, True)
            actual, expected = linalg.qr(sa.transpose(-1, -2)), torch.linalg.qr(ta.transpose(-1, -2))
            self.gradients(actual.Q.sum() + actual.R.sum(), expected.Q.sum() + expected.R.sum(), [('strided QR', sa, ta)])
        for a in (fr.zeros(2, 3), fr.ones(3), fr.zeros(3, 2), fr.tensor([[float('nan')], [1.]]), fr.ones(3, 2, dtype='int64')):
            with self.assertRaises((ValueError, TypeError)):
                linalg.qr(a)
        for mode in ('complete', 'r', 'bad'):
            with self.assertRaises((ValueError, NotImplementedError)):
                linalg.qr(fr.ones(3, 2), mode=mode)
        with self.assertRaises(NotImplementedError):
            linalg.qr(fr.ones(3, 2), out=(fr.ones(3, 2), fr.ones(2, 2)))
        from ferro.autograd import grad
        for dtype in ('float32', 'float64'):
            a = fr.tensor([[2., .1], [.2, 3.], [-.4, .3]], dtype=dtype, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'capture'):
                fr.capture(lambda: linalg.qr(a).Q)
            with self.assertRaises(ValueError):
                grad(linalg.qr(a).R.sum(), [a], create_graph=True)
            result = linalg.qr(a).Q.sum()
            with fr.no_grad():
                a.copy_(fr.tensor([[3., .1], [.2, 4.], [-.4, .3]], dtype=dtype))
            with self.assertRaises((ValueError, RuntimeError)):
                result.backward()
            self.assertIsNone(a.grad)

    def test_wide_complete_qr_strided_tail_gradients_and_empty_batches(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((2, 5, 3), (2, 3, 3), (2, 3, 0)):
                for mode in ('reduced', 'complete'):
                    ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1049)).requires_grad_(True)
                    sa = native(ta, True)
                    actual, expected = linalg.qr(sa.transpose(-1, -2), mode), torch.linalg.qr(ta.transpose(-1, -2), mode)
                    self.close(actual.Q, expected.Q); self.close(actual.R, expected.R)
                    self.gradients(actual.Q.sum() + actual.R.sum(), expected.Q.sum() + expected.R.sum(), [('wide strided QR', sa, ta)])
            empty = fr.tensor([], dtype=str(dtype).split('.')[-1]).reshape([1 << 59, 0, 3]).requires_grad_(True)
            factors = linalg.qr(empty)
            self.assertEqual(list(factors.Q.shape), [1 << 59, 0, 0])
            self.assertEqual(list(factors.R.shape), [1 << 59, 0, 3])
            (factors.Q.sum() + factors.R.sum()).backward()
            self.assertEqual(list(empty.grad.shape), [1 << 59, 0, 3])

    def test_detached_saved_outputs_and_qr_siblings_reject_stale_backward(self):
        from ferro.autograd import grad
        cases = ('inv', 'solve', 'triangular', 'cholesky', 'det', 'norm0', 'norm1', 'norm2', 'norminf',
                 'qrQ', 'qrR', 'qrQ_sibling', 'qrR_sibling', 'slogsign', 'slogsign_sibling')
        for dtype in ('float32', 'float64'):
            for name in cases:
                for functional in (False, True):
                    with self.subTest(dtype=dtype, name=name, functional=functional):
                        a = fr.tensor([[3., .2], [.2, 2.]], dtype=dtype, requires_grad=True)
                        ta = torch.tensor([[3., .2], [.2, 2.]], dtype=getattr(torch, dtype), requires_grad=True)
                        if name in ('solve', 'triangular'):
                            b, tb = fr.ones(2, 1, dtype=dtype), torch.ones(2, 1, dtype=getattr(torch, dtype))
                            output = linalg.solve(a, b) if name == 'solve' else linalg.solve_triangular(a, b, upper=True)
                            wanted = torch.linalg.solve(ta, tb) if name == 'solve' else torch.linalg.solve_triangular(ta, tb, upper=True)
                            modified, ref_modified = output, wanted
                        elif name.startswith('norm'):
                            order = {'norm0': 0, 'norm1': 1, 'norm2': 2, 'norminf': float('inf')}[name]
                            output, wanted = linalg.vector_norm(a, order), torch.linalg.vector_norm(ta, order)
                            modified, ref_modified = output, wanted
                        elif name.startswith('qr'):
                            output_pair, wanted_pair = linalg.qr(a), torch.linalg.qr(ta)
                            selected = name[2]
                            output, wanted = getattr(output_pair, selected), getattr(wanted_pair, selected)
                            other = ('R' if selected == 'Q' else 'Q') if name.endswith('sibling') else selected
                            modified, ref_modified = getattr(output_pair, other), getattr(wanted_pair, other)
                        elif name.startswith('slog'):
                            output_pair, wanted_pair = linalg.slogdet(a), torch.linalg.slogdet(ta)
                            selected = 'logabsdet' if name.endswith('sibling') else 'sign'
                            output, wanted = getattr(output_pair, selected), getattr(wanted_pair, selected)
                            modified, ref_modified = output_pair.sign, wanted_pair.sign
                        else:
                            output, wanted = getattr(linalg, name)(a), getattr(torch.linalg, name)(ta)
                            modified, ref_modified = output, wanted
                        modified.detach().copy_(fr.ones(modified.shape, dtype=dtype))
                        ref_modified.detach().copy_(torch.ones_like(ref_modified))
                        with self.assertRaises((ValueError, RuntimeError)):
                            grad(output.sum(), [a]) if functional else output.sum().backward()
                        with self.assertRaises(RuntimeError):
                            torch.autograd.grad(wanted.sum(), [ta]) if functional else wanted.sum().backward()
                        self.assertIsNone(a.grad)
                        self.assertIsNone(ta.grad)
            ta = torch.tensor([[3., .2], [.2, 2.]], dtype=getattr(torch, dtype), requires_grad=True)
            a = native(ta, True)
            output, expected = linalg.slogdet(a).logabsdet, torch.linalg.slogdet(ta).logabsdet
            output.detach().copy_(fr.tensor(2., dtype=dtype)); expected.detach().fill_(2.)
            self.gradients(output, expected, [('unsaved logarithm', a, ta)])

    def test_eigh_and_eigvalsh_torch_values_all_input_gradients_and_triangles(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((1, 1), (3, 3), (2, 1, 4, 4), (0, 0), (0, 3, 3)):
                for upper in ('L', 'U'):
                    for selected in ('values', 'vectors', 'both'):
                        with self.subTest(dtype=dtype, shape=shape, upper=upper, selected=selected):
                            ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1087)).requires_grad_(True)
                            sa = native(ta, True)
                            actual, expected = linalg.eigh(sa, upper), torch.linalg.eigh(ta, upper)
                            self.close(actual.eigenvalues, expected.eigenvalues)
                            numeric = torch.tensor(actual.eigenvectors.tolist(), dtype=dtype).reshape(shape)
                            signs = (numeric * expected.eigenvectors.detach()).sum(-2).sign()
                            aligned = expected.eigenvectors * signs.unsqueeze(-2)
                            self.close(actual.eigenvectors, aligned)
                            actual_loss = fr.tensor(0., dtype=str(dtype).split('.')[-1])
                            expected_loss = torch.tensor(0., dtype=dtype)
                            for name, target in (('values', expected.eigenvalues), ('vectors', aligned)):
                                if selected not in (name, 'both'):
                                    continue
                                value = actual.eigenvalues if name == 'values' else actual.eigenvectors
                                probe = torch.randn(target.shape, dtype=dtype, generator=torch.Generator().manual_seed(1091))
                                actual_loss = actual_loss + (value * native(probe)).sum()
                                expected_loss = expected_loss + (target * probe).sum()
                            self.gradients(actual_loss, expected_loss, [('eigh input', sa, ta)])
                            self.close(linalg.eigvalsh(sa, upper.lower()), torch.linalg.eigvalsh(ta, upper))

    def test_eigh_strided_symmetric_manifold_finite_differences_and_degeneracy(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.tensor([[[2., .7, -.4], [.3, 4., .9], [-.2, .1, 6.]]], dtype=dtype, requires_grad=True)
            sa = native(ta, True)
            actual, expected = linalg.eigh(sa.transpose(-1, -2)), torch.linalg.eigh(ta.transpose(-1, -2))
            probe = torch.tensor([[.2, .5, .7], [-.3, .1, .6], [.9, -.8, .4]], dtype=dtype)
            self.gradients((actual.eigenvectors.pow(2.) * native(probe)).sum() + actual.eigenvalues.sum(),
                           (expected.eigenvectors.square() * probe).sum() + expected.eigenvalues.sum(), [('strided eigen', sa, ta)])
            repeated = fr.tensor([[2., 0.], [0., 2.]], dtype=str(dtype).split('.')[-1], requires_grad=True)
            linalg.eigvalsh(repeated).sum().backward()
            self.close(repeated.grad, torch.eye(2, dtype=dtype))
            repeated.zero_grad()
            linalg.eigh(repeated).eigenvectors.sum().backward()
            self.assertTrue(all(math.isnan(x) for x in repeated.grad.reshape([-1]).tolist()))
        base = torch.tensor([[2., .2], [.2, 4.]], dtype=torch.float64)
        source = native(base, True)
        # Squared entries give a sign-invariant eigenvector objective.
        probe = fr.tensor([[.2, -.3], [.5, .7]], dtype='float64')
        (linalg.eigh(source).eigenvectors.pow(2.) * probe).sum().backward()
        analytic = torch.tensor(source.grad.tolist(), dtype=torch.float64)
        for row, column in ((0, 0), (0, 1), (1, 1)):
            high, low = base.clone(), base.clone()
            high[row, column] += 1e-6; low[row, column] -= 1e-6
            if row != column:
                high[column, row] += 1e-6; low[column, row] -= 1e-6
            def scalar(matrix):
                vectors = linalg.eigh(native(matrix)).eigenvectors
                return (vectors.pow(2.) * probe).sum().item()
            numeric = (scalar(high) - scalar(low)) / 2e-6
            wanted = analytic[row, column] * (2 if row != column else 1)
            self.assertAlmostEqual(numeric, wanted.item(), delta=3e-8)

    def test_eigh_guards_ignored_triangle_empty_and_saved_outputs(self):
        from ferro.autograd import grad
        for dtype in ('float32', 'float64'):
            ignored = fr.tensor([[2., float('nan')], [.3, 4.]], dtype=dtype, requires_grad=True)
            linalg.eigvalsh(ignored, 'L').sum().backward()
            self.close(ignored.grad, torch.eye(2, dtype=getattr(torch, dtype)))
            with self.assertRaises(ValueError):
                linalg.eigh(ignored, 'U')
            empty = fr.tensor([], dtype=dtype).reshape([1 << 60, 0, 0]).requires_grad_(True)
            result = linalg.eigh(empty)
            (result.eigenvalues.sum() + result.eigenvectors.sum()).backward()
            self.assertEqual(list(empty.grad.shape), [1 << 60, 0, 0])
            for selected in ('eigenvalues', 'eigenvectors'):
                for modified in ('eigenvalues', 'eigenvectors'):
                    a = fr.tensor([[2., .3], [.3, 4.]], dtype=dtype, requires_grad=True)
                    result = linalg.eigh(a)
                    value = getattr(result, modified)
                    value.detach().copy_(fr.ones(value.shape, dtype=dtype))
                    with self.assertRaises((ValueError, RuntimeError)):
                        getattr(result, selected).sum().backward()
                    self.assertIsNone(a.grad)
            a = fr.tensor([[2., .3], [.3, 4.]], dtype=dtype, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'capture'):
                fr.capture(lambda: linalg.eigh(a).eigenvalues)
            with self.assertRaises(ValueError):
                grad(linalg.eigh(a).eigenvalues.sum(), [a], create_graph=True)
        for a in (fr.ones(3), fr.ones(2, 3), fr.ones(2, 2, dtype='int64')):
            with self.assertRaises((ValueError, TypeError)):
                linalg.eigh(a)
        for upper in ('bad', '', None):
            with self.assertRaises((ValueError, TypeError)):
                linalg.eigh(fr.ones(2, 2), upper)
        with self.assertRaises(NotImplementedError):
            linalg.eigh(fr.ones(2, 2), out=(fr.ones(2), fr.ones(2, 2)))


    def test_svd_full_reduced_factors_and_separate_combined_torch_gradients(self):
        shapes = ((5, 3), (3, 3), (2, 5), (1, 4), (2, 1, 4, 2), (2, 3, 5), (0, 0), (0, 3), (3, 0), (0, 4, 2))
        for dtype in (torch.float32, torch.float64):
            for shape in shapes:
                width = min(shape[-2:])
                for full in (False, True):
                    for selected in ('U', 'S', 'Vh', 'all'):
                        with self.subTest(dtype=dtype, shape=shape, full=full, selected=selected):
                            ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1129)).requires_grad_(True)
                            sa = native(ta, True)
                            actual, expected = linalg.svd(sa, full), torch.linalg.svd(ta, full_matrices=full)
                            self.assertEqual(actual.U.shape, list(expected.U.shape))
                            self.assertEqual(actual.Vh.shape, list(expected.Vh.shape))
                            au = actual.U.narrow(len(shape)-1, 0, width)
                            av = actual.Vh.narrow(len(shape)-2, 0, width)
                            tu = expected.U.narrow(-1, 0, width)
                            tv = expected.Vh.narrow(-2, 0, width)
                            numeric = torch.tensor(au.tolist(), dtype=dtype).reshape(tu.shape)
                            signs = (numeric * tu.detach()).sum(-2).sign()
                            components = {'U': (au, tu * signs.unsqueeze(-2)), 'S': (actual.S, expected.S), 'Vh': (av, tv * signs.unsqueeze(-1))}
                            for value, target in components.values():
                                self.close(value, target)
                            actual_loss, expected_loss = None, None
                            for name in ('U', 'S', 'Vh') if selected == 'all' else (selected,):
                                value, target = components[name]
                                probe = torch.randn(target.shape, dtype=dtype, generator=torch.Generator().manual_seed(1151))
                                term, wanted = (value * native(probe)).sum(), (target * probe).sum()
                                actual_loss = term if actual_loss is None else actual_loss + term
                                expected_loss = wanted if expected_loss is None else expected_loss + wanted
                            self.gradients(actual_loss, expected_loss, [('SVD input', sa, ta)])
                            if selected == 'all':
                                self.close(linalg.svdvals(sa), torch.linalg.svdvals(ta))
                                if width and math.prod(shape[:-2]):
                                    self.close((au * actual.S.unsqueeze(len(shape)-2)) @ av, ta)
                                self.close(actual.U.transpose(-1, -2) @ actual.U, expected.U.transpose(-1, -2) @ expected.U)
                                self.close(actual.Vh @ actual.Vh.transpose(-1, -2), expected.Vh @ expected.Vh.transpose(-1, -2))

    def test_svd_strided_inputs_and_ignored_full_complementary_adjoints(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((2, 3, 5), (2, 5, 3)):
                ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1153)).requires_grad_(True)
                sa = native(ta, True)
                actual, expected = linalg.svd(sa.transpose(-1, -2), False), torch.linalg.svd(ta.transpose(-1, -2), full_matrices=False)
                up = torch.randn(actual.U.shape, dtype=dtype, generator=torch.Generator().manual_seed(1163))
                vp = torch.randn(actual.Vh.shape, dtype=dtype, generator=torch.Generator().manual_seed(1171))
                self.gradients((actual.U.pow(2.) * native(up)).sum() + (actual.Vh.pow(2.) * native(vp)).sum() + actual.S.sum(),
                               (expected.U.square() * up).sum() + (expected.Vh.square() * vp).sum() + expected.S.sum(), [('strided SVD', sa, ta)])
                ta = ta.detach().requires_grad_(); sa = native(ta, True)
                actual, expected = linalg.svd(sa), torch.linalg.svd(ta)
                width = min(shape[-2:])
                if shape[-2] > width:
                    loss = actual.U.narrow(len(shape)-1, width, shape[-2]-width).sum()
                    target = expected.U[..., width:].sum()
                else:
                    loss = actual.Vh.narrow(len(shape)-2, width, shape[-1]-width).sum()
                    target = expected.Vh[..., width:, :].sum()
                loss.backward(); target.backward()
                self.close(sa.grad, ta.grad)
                self.close(sa.grad, torch.zeros_like(ta))

    def test_svd_zero_rank_repeated_values_empty_capacities_and_guards(self):
        for dtype in (torch.float32, torch.float64):
            for values in ([[0., 0.], [0., 0.], [0., 0.]], [[1., 2.], [2., 4.], [0., 0.]]):
                a = torch.tensor(values, dtype=dtype)
                for transposed in (False, True):
                    ta = a.T if transposed else a
                    actual, expected = linalg.svd(native(ta)), torch.linalg.svd(ta)
                    self.close(actual.S, expected.S)
                    width = min(ta.shape)
                    self.close((actual.U.narrow(1, 0, width) * actual.S.unsqueeze(0)) @ actual.Vh.narrow(0, 0, width), ta)
                    self.close(actual.U.transpose(0, 1) @ actual.U, torch.eye(ta.shape[0], dtype=dtype))
                    self.close(actual.Vh @ actual.Vh.transpose(0, 1), torch.eye(ta.shape[1], dtype=dtype))
            ta = (torch.eye(3, dtype=dtype) * 2).requires_grad_()
            sa = native(ta, True)
            self.gradients(linalg.svdvals(sa).sum(), torch.linalg.svdvals(ta).sum(), [('repeated singular values', sa, ta)])
            for full in (False, True):
                empty = fr.tensor([], dtype=str(dtype).split('.')[-1]).reshape([1 << 60, 0, 0]).requires_grad_(True)
                u, s, vh = linalg.svd(empty, full)
                (u.sum() + s.sum() + vh.sum()).backward()
                self.assertEqual(empty.grad.shape, [1 << 60, 0, 0])
            huge = fr.tensor([], dtype=str(dtype).split('.')[-1]).reshape([1 << 30, 0])
            with self.assertRaisesRegex(ValueError, 'capacity|large'):
                linalg.svd(huge)
        for a in (fr.ones(3), fr.ones(2, 2, dtype='int64'), fr.tensor([[float('nan')]])):
            with self.assertRaises((ValueError, TypeError)):
                linalg.svd(a)
        with self.assertRaises(TypeError): linalg.svd(fr.ones(2, 2), full_matrices=1)
        with self.assertRaises(NotImplementedError): linalg.svd(fr.ones(2, 2), driver='gesvd')
        with self.assertRaises(NotImplementedError): linalg.svdvals(fr.ones(2, 2), out=fr.ones(2))
        from ferro.autograd import grad
        for dtype in ('float32', 'float64'):
            a = fr.tensor([[2., .2], [-.1, 1.7], [.4, -.3]], dtype=dtype, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.svdvals(a))
            with self.assertRaises(ValueError): grad(linalg.svdvals(a).sum(), [a], create_graph=True)

    def test_spectral_and_nuclear_norm_axes_and_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for dims in ((-2, -1), (0, 2), (2, 0)):
                for order in ('nuc', 2, -2):
                    for keepdim in (False, True):
                        ta = torch.randn(2, 3, 4, dtype=dtype, generator=torch.Generator().manual_seed(61), requires_grad=True)
                        sa = native(ta, True)
                        with self.subTest(dtype=dtype, dims=dims, order=order, keepdim=keepdim):
                            self.gradients(linalg.matrix_norm(sa, order, dims, keepdim), torch.linalg.matrix_norm(ta, order, dims, keepdim), [('input', sa, ta)])
        for order in ('nuc', 2, -2):
            ta = torch.tensor([[.3, -.7], [1.2, .4], [.6, -.3]], requires_grad=True)
            sa = native(ta, True)
            self.gradients(linalg.norm(sa, order, dtype='float64'), torch.linalg.norm(ta, order, dtype=torch.float64), [('widened input', sa, ta)])

    def test_spectral_nuclear_zero_ties_empty_and_guards(self):
        for dtype in (torch.float32, torch.float64):
            for values in (torch.zeros(3, 2, dtype=dtype), torch.eye(3, dtype=dtype), torch.diag(torch.tensor([2., 2., 0.], dtype=dtype))):
                for order in ('nuc', 2, -2):
                    ta = values.clone().requires_grad_(); sa = native(ta, True)
                    self.gradients(linalg.matrix_norm(sa, order), torch.linalg.matrix_norm(ta, order), [('zero or tied singular value', sa, ta)])
            for shape in ((0, 3, 2), (2, 0, 3), (2, 3, 0)):
                for order in ('nuc', 2, -2):
                    ta = torch.empty(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                    if order == -2 and min(shape[-2:]) == 0:
                        with self.assertRaises(ValueError): linalg.matrix_norm(sa, order)
                        with self.assertRaises(IndexError): torch.linalg.matrix_norm(ta, order)
                    elif order == 2 and min(shape[-2:]) == 0:
                        actual, expected = linalg.matrix_norm(sa, order), torch.linalg.matrix_norm(ta, order)
                        self.close(actual, expected)
                        self.assertFalse(actual.requires_grad); self.assertFalse(expected.requires_grad)
                    else:
                        self.gradients(linalg.matrix_norm(sa, order), torch.linalg.matrix_norm(ta, order), [('empty input', sa, ta)])
        from ferro.autograd import grad
        for order in ('nuc', 2, -2):
            a = fr.tensor([[2., .1], [.2, 3.]], requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.matrix_norm(a, order))
            with self.assertRaises(ValueError): grad(linalg.matrix_norm(a, order), [a], create_graph=True)
            result = linalg.matrix_norm(a, order)
            with fr.no_grad(): a.copy_(fr.eye(2))
            with self.assertRaises((ValueError, RuntimeError)): result.backward()
            self.assertIsNone(a.grad)

    def test_pinv_values_all_gradients_full_rank_and_singular(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((3, 2), (2, 3), (3, 3), (2, 4, 2)):
                for singular in (False, True):
                    values = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1307))
                    if singular: values[..., 1] = values[..., 0]
                    ta = values.requires_grad_(); sa = native(ta, True)
                    with self.subTest(dtype=dtype, shape=shape, singular=singular):
                        self.gradients(linalg.pinv(sa), torch.linalg.pinv(ta), [('input', sa, ta)])
            for values in (torch.zeros(3, 2, dtype=dtype), torch.eye(3, dtype=dtype)):
                ta = values.requires_grad_(); sa = native(ta, True)
                self.gradients(linalg.pinv(sa), torch.linalg.pinv(ta), [('zero or repeated spectrum', sa, ta)])
            parent = torch.randn(2, 3, 4, dtype=dtype, generator=torch.Generator().manual_seed(1313), requires_grad=True)
            actual_parent = native(parent, True)
            self.gradients(linalg.pinv(actual_parent.transpose(-2, -1)), torch.linalg.pinv(parent.transpose(-2, -1)), [('strided parent', actual_parent, parent)])

    def test_pinv_tolerance_batch_broadcasts_and_legacy_aliases(self):
        for dtype in (torch.float32, torch.float64):
            for option in ({'atol': .1}, {'rtol': .1}, {'atol': torch.tensor([0., .4, 2.], dtype=torch.float64)},
                           {'atol': torch.tensor([[0.], [.4]], dtype=torch.float32), 'rtol': torch.tensor([0., .2, .9], dtype=torch.float64)}):
                ta = torch.tensor([[2., 0.], [0., .5], [0., 0.]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                options = {key: native(value) if isinstance(value, torch.Tensor) else value for key, value in option.items()}
                self.gradients(linalg.pinv(sa, **options), torch.linalg.pinv(ta, **option), [('broadcast input', sa, ta)])
            ta = torch.tensor([[[2., 0.], [0., .5]], [[1., 0.], [0., .2]]], dtype=dtype, requires_grad=True); sa = native(ta, True)
            tolerance = torch.tensor([[.1], [.3], [2.]], dtype=torch.float64)
            self.gradients(linalg.pinv(sa, atol=native(tolerance)), torch.linalg.pinv(ta, atol=tolerance), [('existing input batch', sa, ta)])
        value = torch.tensor([[2., 0.], [0., .1]], dtype=torch.float64)
        self.close(linalg.pinv(native(value), .2), torch.linalg.pinv(value, .2))
        self.close(linalg.matrix_rank(native(value), .2), torch.linalg.matrix_rank(value, .2))

    def test_rank_tolerances_dtype_cutoffs_and_no_gradient(self):
        for dtype in (torch.float32, torch.float64):
            a = torch.tensor([[[1., 0.], [2., 0.], [0., 0.]], [[2., 0.], [0., .5], [0., 0.]]], dtype=dtype, requires_grad=True)
            for absolute in (None, 0., -.1, .2, float('nan'), float('inf'), torch.tensor([[0.], [.6], [2.]], dtype=torch.float64)):
                for relative in (None, 0., .1):
                    options = {'atol': absolute, 'rtol': relative}
                    if any(isinstance(value, torch.Tensor) for value in options.values()):
                        options = {key: torch.tensor(value, dtype=torch.float64) if value is not None and not isinstance(value, torch.Tensor) else value for key, value in options.items()}
                    actual_options = {key: native(value) if isinstance(value, torch.Tensor) else value for key, value in options.items()}
                    result = linalg.matrix_rank(native(a, True), **actual_options)
                    self.close(result, torch.linalg.matrix_rank(a, **options))
                    self.assertFalse(result.requires_grad)
        a = torch.diag(torch.tensor([1., .1]))
        for dtype in (torch.float32, torch.float64):
            tolerance = torch.tensor(.1000000009, dtype=dtype)
            self.close(linalg.matrix_rank(native(a), atol=native(tolerance)), torch.linalg.matrix_rank(a, atol=tolerance))
        small = torch.tensor(torch.tensor(1.1).item() * 3 * torch.finfo(torch.float32).eps).item()
        a = torch.diag(torch.tensor([1.1, .1, small]))
        for relative in (None, 3 * torch.finfo(torch.float32).eps, torch.tensor(3 * torch.finfo(torch.float32).eps), torch.tensor(3 * torch.finfo(torch.float32).eps, dtype=torch.float64)):
            self.close(linalg.matrix_rank(native(a), rtol=native(relative) if isinstance(relative, torch.Tensor) else relative), torch.linalg.matrix_rank(a, rtol=relative))
        a = torch.tensor([[33554432.]])
        for ad in (torch.float32, torch.float64, torch.int64):
            for rd in (torch.float32, torch.float64, torch.int64):
                absolute, relative = torch.tensor(33554431, dtype=ad), torch.tensor(0, dtype=rd)
                for actual, expected in ((linalg.matrix_rank, torch.linalg.matrix_rank), (linalg.pinv, torch.linalg.pinv)):
                    self.close(actual(native(a), atol=native(absolute), rtol=native(relative)), expected(a, atol=absolute, rtol=relative))
        a = torch.tensor([[float((1 << 62) + (1 << 39))]])
        for dtype in (torch.int64, torch.float64):
            absolute = torch.tensor((1 << 62) + (1 << 38) + 1, dtype=dtype)
            self.close(linalg.matrix_rank(native(a), atol=native(absolute), rtol=fr.tensor(0.)),
                       torch.linalg.matrix_rank(a, atol=absolute, rtol=torch.tensor(0.)))

    def test_zero_cutoff_rank_keeps_exact_column_dependence(self):
        # A zero cutoff exposes SVD algorithm rounding: LAPACK can retain a
        # tiny positive singular value for these exactly dependent columns.
        for dtype in (torch.float32, torch.float64):
            a = torch.tensor([[1., 2.], [2., 4.], [0., 0.]], dtype=dtype)
            self.assertEqual(linalg.matrix_rank(native(a), rtol=0.).tolist(), 1)
            self.close(linalg.matrix_rank(native(a)), torch.linalg.matrix_rank(a))

    def test_hermitian_pinv_lower_triangle_and_dense_adjoint(self):
        for dtype in (torch.float32, torch.float64):
            for values in ([[2., .4], [.4, 3.]], [[2., .4], [.4, -3.]], [[2., 7.], [.4, 3.]], [[1., 9., 4.], [1., 1., 7.], [1., 1., 1.]]):
                ta = torch.tensor(values, dtype=dtype, requires_grad=True); sa = native(ta, True)
                self.gradients(linalg.pinv(sa, hermitian=True), torch.linalg.pinv(ta, hermitian=True), [('original dense input', sa, ta)])
                self.close(linalg.matrix_rank(sa, hermitian=True), torch.linalg.matrix_rank(ta, hermitian=True))
            ta = torch.tensor([[2., float('nan')], [.4, 3.]], dtype=dtype)
            self.close(linalg.pinv(native(ta), hermitian=True), torch.linalg.pinv(ta, hermitian=True))

    def test_pinv_rank_empty_capacity_and_invalid_options(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((0, 3, 2), (2, 0, 3), (2, 3, 0), (0, 0)):
                ta = torch.empty(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                self.gradients(linalg.pinv(sa), torch.linalg.pinv(ta), [('empty input', sa, ta)])
                self.close(linalg.matrix_rank(sa), torch.linalg.matrix_rank(ta))
        empty = fr.zeros(1 << 60, 0, 0, requires_grad=True)
        linalg.pinv(empty).sum().backward(); self.assertEqual(empty.grad.shape, empty.shape)
        with self.assertRaisesRegex(ValueError, 'capacity|large'): linalg.matrix_rank(empty)
        for method in (linalg.pinv, linalg.matrix_rank):
            with self.assertRaisesRegex(ValueError, 'capacity|large'): method(fr.zeros(0, 1 << 30, 1 << 30))
            for value in (fr.ones(3), fr.ones(2, 2, dtype='int64')):
                with self.assertRaises((ValueError, TypeError)): method(value)
            with self.assertRaises(ValueError): method(fr.ones(2, 3), hermitian=True)
            with self.assertRaises(ValueError): method(fr.ones(2, 2, 2), atol=fr.ones(3))
            with self.assertRaises(TypeError): method(fr.ones(2, 2), hermitian=1)
            with self.assertRaises(TypeError): method(fr.ones(2, 2), .1, atol=.1)
            with self.assertRaises(NotImplementedError): method(fr.ones(2, 2), out=fr.zeros(2, 2))
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: method(fr.eye(2)))

    def test_pinv_saved_input_output_and_unsaved_tolerance_versions(self):
        from ferro.autograd import grad
        for dtype in ('float32', 'float64'):
            for selected in ('input', 'output'):
                for functional in (False, True):
                    a = fr.tensor([[2., .2], [-.1, 1.7]], dtype=dtype, requires_grad=True)
                    absolute, relative = fr.tensor(.0, dtype='float64'), fr.tensor(1e-7, dtype='float64')
                    output = linalg.pinv(a, atol=absolute, rtol=relative)
                    value = dict(input=a, atol=absolute, rtol=relative, output=output)[selected]
                    with fr.no_grad(): value.detach().copy_(fr.ones(value.shape, dtype=value.dtype))
                    with self.assertRaises((ValueError, RuntimeError)):
                        grad(output.sum(), [a]) if functional else output.sum().backward()
                    self.assertIsNone(a.grad)
            for selected in ('atol', 'rtol'):
                ta = torch.tensor([[2., .2], [-.1, 1.7]], dtype=getattr(torch, dtype), requires_grad=True); sa = native(ta, True)
                atol, rtol = torch.tensor(.0, dtype=torch.float64), torch.tensor(1e-7, dtype=torch.float64)
                absolute, relative = native(atol), native(rtol)
                actual, expected = linalg.pinv(sa, atol=absolute, rtol=relative), torch.linalg.pinv(ta, atol=atol, rtol=rtol)
                dict(atol=absolute, rtol=relative)[selected].copy_(fr.tensor(1., dtype='float64'))
                dict(atol=atol, rtol=rtol)[selected].fill_(1.)
                self.gradients(actual, expected, [('tolerance mutation', sa, ta)])
        tolerance = fr.tensor(.1, requires_grad=True)
        with self.assertRaises(ValueError): linalg.pinv(fr.eye(2), atol=tolerance)
        self.assertFalse(linalg.matrix_rank(fr.eye(2), atol=tolerance).requires_grad)
        with fr.no_grad(): self.assertFalse(linalg.pinv(fr.eye(2), atol=tolerance).requires_grad)
        a = fr.eye(2).requires_grad_(True)
        with self.assertRaises(ValueError): grad(linalg.pinv(a).sum(), [a], create_graph=True)

    def test_lstsq_default_solution_rank_and_all_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((4, 2), (2, 4), (3, 3), (2, 4, 2)):
                for vector in (False, True):
                    for singular in (False, True):
                        generator = torch.Generator().manual_seed(1351)
                        values = torch.randn(shape, dtype=dtype, generator=generator)
                        if singular:
                            if shape[-2] <= shape[-1]: values[..., -1, :] = 0.
                            else: values[..., :, -1] = 0.
                        ta = values.requires_grad_(); sa = native(ta, True)
                        rhs_shape = shape[:-1] if vector else (*shape[:-1], 2)
                        tb = torch.randn(rhs_shape, dtype=dtype, generator=generator, requires_grad=True); sb = native(tb, True)
                        with self.subTest(dtype=dtype, shape=shape, vector=vector, singular=singular):
                            actual, expected = linalg.lstsq(sa, sb), torch.linalg.lstsq(ta, tb)
                            self.gradients(actual.solution, expected.solution, [('A', sa, ta), ('B', sb, tb)])
                            for name in ('rank', 'residuals', 'singular_values'):
                                self.close(getattr(actual, name), getattr(expected, name))
                            self.assertFalse(actual.rank.requires_grad); self.assertFalse(actual.singular_values.requires_grad)

    def test_lstsq_broadcast_and_strided_original_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(1361)
            ta = torch.randn(2, 1, 3, 2, dtype=dtype, generator=generator, requires_grad=True)
            tb = torch.randn(1, 4, 3, 2, dtype=dtype, generator=generator, requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            actual, expected = linalg.lstsq(sa, sb), torch.linalg.lstsq(ta, tb)
            self.gradients(actual.solution, expected.solution, [('broadcast A', sa, ta), ('broadcast B', sb, tb)])
            self.close(actual.rank, expected.rank); self.assertEqual(actual.rank.shape, [2, 1])
            ta = torch.randn(2, 2, 4, dtype=dtype, generator=generator, requires_grad=True)
            tb = torch.randn(2, 3, 4, dtype=dtype, generator=generator, requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            self.gradients(linalg.lstsq(sa.transpose(-2, -1), sb.transpose(-2, -1)).solution,
                           torch.linalg.lstsq(ta.transpose(-2, -1), tb.transpose(-2, -1)).solution,
                           [('strided A parent', sa, ta), ('strided B parent', sb, tb)])

    def test_lstsq_explicit_cutoff_torch_backward_convention(self):
        for dtype in (torch.float32, torch.float64):
            for values in ([[2., 0.], [0., .1]], [[2., 0.], [0., .1], [0., 0.]], [[2., 0., 0.], [0., .1, 0.]]):
                for rcond in (1e-5, .2):
                    ta = torch.tensor(values, dtype=dtype, requires_grad=True)
                    tb = torch.arange(1., 1. + ta.shape[0], dtype=dtype, requires_grad=True)
                    sa, sb = native(ta, True), native(tb, True)
                    with self.subTest(dtype=dtype, shape=tuple(ta.shape), rcond=rcond):
                        # The default gelsy driver intermittently misranks this wide zero-column input.
                        actual, expected = linalg.lstsq(sa, sb, rcond), torch.linalg.lstsq(ta, tb, rcond=rcond, driver='gelsd')
                        self.gradients(actual.solution, expected.solution, [('A', sa, ta), ('B', sb, tb)])
                        self.close(actual.rank, expected.rank)
            ta = torch.tensor([[2., .5], [.3, .2], [.2, .1]], dtype=dtype, requires_grad=True)
            tb = torch.tensor([1., 2., 3.], dtype=dtype, requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            actual, expected = linalg.lstsq(sa, sb, .2), torch.linalg.lstsq(ta, tb, rcond=.2, driver='gelsd')
            self.gradients(actual.solution, expected.solution, [('SVD-truncated A', sa, ta), ('SVD-truncated B', sb, tb)])
            self.close(actual.rank, expected.rank)
        a = fr.tensor([[2., 0.], [0., .1]], dtype='float64', requires_grad=True)
        b = fr.tensor([1., 2.], dtype='float64', requires_grad=True)
        linalg.lstsq(a, b, .2).solution.sum().backward()
        self.assertEqual(b.grad.tolist(), [.5, 10.])
        high = linalg.lstsq(a, fr.tensor([1., 2.00001], dtype='float64'), .2).solution
        low = linalg.lstsq(a, fr.tensor([1., 1.99999], dtype='float64'), .2).solution
        self.assertEqual(high.tolist(), low.tolist())

    def test_lstsq_saved_solution_inputs_and_unused_residual_slots(self):
        from ferro.autograd import grad
        for dtype in ('float32', 'float64'):
            for selected in ('A', 'B', 'solution'):
                for differentiated in ('solution', 'residuals'):
                    a = fr.tensor([[2., .3], [.1, 1.7]], dtype=dtype, requires_grad=True)
                    b = fr.tensor([.2, .4], dtype=dtype, requires_grad=True)
                    result = linalg.lstsq(a, b)
                    value = dict(A=a, B=b, solution=result.solution)[selected]
                    with fr.no_grad(): value.detach().copy_(fr.ones(value.shape, dtype=dtype))
                    with self.assertRaises((ValueError, RuntimeError)): getattr(result, differentiated).sum().backward()
                    self.assertIsNone(a.grad); self.assertIsNone(b.grad)
            ta = torch.eye(2, dtype=getattr(torch, dtype), requires_grad=True)
            tb = torch.ones(2, dtype=getattr(torch, dtype), requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            actual, expected = linalg.lstsq(sa, sb), torch.linalg.lstsq(ta, tb)
            self.assertTrue(actual.residuals.requires_grad); self.assertTrue(expected.residuals.requires_grad)
            actual.residuals.sum().backward(); expected.residuals.sum().backward(retain_graph=True)
            self.assertIsNone(sa.grad); self.assertIsNone(sb.grad); self.assertIsNone(ta.grad); self.assertIsNone(tb.grad)
            self.assertEqual(grad(actual.residuals.sum(), [sa, sb], allow_unused=True), (None, None))
            self.assertEqual(torch.autograd.grad(expected.residuals.sum(), [ta, tb], allow_unused=True), (None, None))
            for value, source in zip(actual.residuals.sum().grad_wrt([sa, sb]), (sa, sb)):
                self.assertEqual(value.tolist(), fr.zeros(source.shape, dtype=dtype).tolist())

    def test_lstsq_empties_capacity_and_explicit_scope(self):
        for dtype in (torch.float32, torch.float64):
            for a_shape, b_shape in (((0, 2), (0, 1)), ((2, 0), (2, 1)), ((2, 2), (2, 0)), ((0, 3, 2), (0, 3, 1))):
                ta = torch.zeros(a_shape, dtype=dtype, requires_grad=True); tb = torch.zeros(b_shape, dtype=dtype, requires_grad=True)
                sa, sb = native(ta, True), native(tb, True)
                actual, expected = linalg.lstsq(sa, sb), torch.linalg.lstsq(ta, tb)
                self.gradients(actual.solution, expected.solution, [('empty A', sa, ta), ('empty B', sb, tb)])
                self.close(actual.rank, expected.rank)
        a = fr.zeros(1, 0, 0, requires_grad=True); b = fr.zeros(1 << 60, 0, 0, requires_grad=True)
        result = linalg.lstsq(a, b); result.solution.sum().backward()
        self.assertEqual(b.grad.shape, b.shape); self.assertEqual(result.rank.shape, [1])
        with self.assertRaisesRegex(ValueError, 'capacity|large'): linalg.lstsq(b, b)
        a, b = fr.eye(2).requires_grad_(True), fr.ones(2, requires_grad=True)
        for rcond in (0., -1., 1., 2., float('inf'), float('nan')):
            with self.assertRaises(ValueError): linalg.lstsq(a, b, rcond)
        with self.assertRaises(TypeError): linalg.lstsq(a, b, fr.tensor(.1))
        for driver in ('gels', 'gelsy', 'gelsd', 'gelss'):
            with self.assertRaises(NotImplementedError): linalg.lstsq(a, b, driver=driver)
        with self.assertRaises(NotImplementedError): linalg.lstsq(a, b, out=())
        with self.assertRaises(ValueError): linalg.lstsq(a, b.to(dtype='float64'))
        with self.assertRaises(ValueError): linalg.lstsq(fr.ones(2, 3, 2), fr.ones(1, 3))
        with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.lstsq(a, b).solution)
        from ferro.autograd import grad
        with self.assertRaises(ValueError): grad(linalg.lstsq(a, b).solution.sum(), [a, b], create_graph=True)

    def test_solve_ex_broadcast_vector_right_and_all_input_gradients(self):
        cases = (((3, 3), (3,), True), ((2, 3, 3), (2, 3), True),
                 ((2, 1, 3, 3), (1, 4, 3, 2), True), ((3, 3), (4, 3, 2), True),
                 ((2, 1, 3, 3), (1, 4, 2, 3), False), ((3, 3), (2, 3), False))
        for dtype in (torch.float32, torch.float64):
            for shape, rhs_shape, left in cases:
                for check_errors in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, rhs=rhs_shape, left=left, checked=check_errors):
                        ta = self.matrix(shape, dtype); sa = native(ta, True)
                        tb = torch.randn(rhs_shape, dtype=dtype, generator=torch.Generator().manual_seed(1451), requires_grad=True); sb = native(tb, True)
                        actual, expected = linalg.solve_ex(sa, sb, left=left, check_errors=check_errors), torch.linalg.solve_ex(ta, tb, left=left, check_errors=check_errors)
                        self.gradients(actual.result, expected.result, [('A', sa, ta), ('B', sb, tb)])
                        self.close(actual.info, expected.info.to(torch.int64))
                        self.assertEqual(expected.info.dtype, torch.int32)
                        self.assertEqual(actual.info.shape, list(ta.shape[:-2]))
                        self.assertFalse(actual.info.requires_grad)
            ta = self.matrix((2, 3, 3), dtype); sa = native(ta, True)
            tb = torch.randn(2, 2, 3, dtype=dtype, generator=torch.Generator().manual_seed(1453), requires_grad=True); sb = native(tb, True)
            self.gradients(linalg.solve_ex(sa.transpose(-1, -2), sb.transpose(-1, -2)).result,
                           torch.linalg.solve_ex(ta.transpose(-1, -2), tb.transpose(-1, -2)).result,
                           [('strided A', sa, ta), ('strided B', sb, tb)])

    def test_inv_ex_all_input_gradients_and_status_dtype(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((1, 1), (3, 3), (2, 3, 3)):
                for checked in (False, True):
                    ta = self.matrix(shape, dtype); sa = native(ta, True)
                    actual, expected = linalg.inv_ex(sa.transpose(-1, -2), check_errors=checked), torch.linalg.inv_ex(ta.transpose(-1, -2), check_errors=checked)
                    self.gradients(actual.inverse, expected.inverse, [('A', sa, ta)])
                    self.close(actual.info, expected.info.to(torch.int64))
                    self.assertEqual(expected.info.dtype, torch.int32)
                    self.assertFalse(actual.info.requires_grad)

    def test_ex_status_original_pivots_singular_batches_and_checked_errors(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.tensor([[[[2., .3], [.1, 1.7]]], [[[0., 1.], [0., 2.]]], [[[1., 2.], [2., 4.]]]], dtype=dtype)
            tb = torch.full((1, 3, 2, 2), .2, dtype=dtype)
            sa, sb = native(ta), native(tb)
            for left in (False, True):
                actual, expected = linalg.solve_ex(sa, sb, left=left), torch.linalg.solve_ex(ta, tb, left=left)
                self.assertEqual(actual.result.shape, [3, 3, 2, 2])
                self.close(actual.info, expected.info.to(torch.int64)); self.assertEqual(actual.info.tolist(), [[0], [1], [2]])
                self.close(actual.result.narrow(0, 0, 1), expected.result[:1])
                self.assertTrue(any(not math.isfinite(value) for value in actual.result.narrow(0, 1, 2).reshape(-1).tolist()))
                with self.assertRaises(ValueError): linalg.solve_ex(sa, sb, left=left, check_errors=True)
                with self.assertRaises(RuntimeError): torch.linalg.solve_ex(ta, tb, left=left, check_errors=True)
            actual, expected = linalg.inv_ex(sa), torch.linalg.inv_ex(ta)
            self.close(actual.info, expected.info.to(torch.int64)); self.close(actual.inverse.narrow(0, 0, 1), expected.inverse[:1])
            with self.assertRaises(ValueError): linalg.inv_ex(sa, check_errors=True)
            with self.assertRaises(RuntimeError): torch.linalg.inv_ex(ta, check_errors=True)
            # Failed factors have undefined gradients, but remain ordinary AD nodes.
            bad = fr.zeros(2, 2, dtype=str(dtype).split('.')[-1], requires_grad=True)
            linalg.inv_ex(bad).inverse.sum().backward()
            self.assertIsNotNone(bad.grad)

    def test_ex_saved_value_versions_match_actual_torch(self):
        from ferro.autograd import grad
        for dtype in (torch.float32, torch.float64):
            for inverse in (False, True):
                for selected in ('A', 'B', 'output', 'info'):
                    if inverse and selected == 'B': continue
                    for functional in (False, True):
                        ta = torch.tensor([[2., .3], [.1, 1.7]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                        tb = torch.tensor([.2, .4], dtype=dtype, requires_grad=True); sb = native(tb, True)
                        actual = linalg.inv_ex(sa) if inverse else linalg.solve_ex(sa, sb)
                        expected = torch.linalg.inv_ex(ta) if inverse else torch.linalg.solve_ex(ta, tb)
                        got = dict(A=sa, B=sb, output=actual[0], info=actual.info)[selected]
                        wanted = dict(A=ta, B=tb, output=expected[0], info=expected.info)[selected]
                        with fr.no_grad(): got.detach().copy_(fr.ones(got.shape, dtype=got.dtype))
                        with torch.no_grad(): wanted.fill_(1.)
                        reject = selected == 'output' or (selected == 'A' and not inverse)
                        if reject:
                            with self.assertRaises((ValueError, RuntimeError)):
                                grad(actual[0].sum(), [sa]) if functional else actual[0].sum().backward()
                            with self.assertRaises(RuntimeError):
                                torch.autograd.grad(expected[0].sum(), [ta]) if functional else expected[0].sum().backward()
                            self.assertIsNone(sa.grad)
                        elif functional:
                            inputs, targets = ([sa], [ta]) if inverse else ([sa, sb], [ta, tb])
                            for got, wanted in zip(grad(actual[0].sum(), inputs), torch.autograd.grad(expected[0].sum(), targets)):
                                self.close(got, wanted)
                        else:
                            inputs = [('A', sa, ta)] if inverse else [('A', sa, ta), ('B', sb, tb)]
                            self.gradients(actual[0], expected[0], inputs)

    def test_ex_empty_rhs_status_capacity_and_declared_scope(self):
        for dtype in (torch.float32, torch.float64):
            for shape, rhs in (((0, 0), (0,)), ((2, 0, 0), (2, 0, 1)), ((2, 2), (2, 0)), ((0, 2, 2), (0, 2, 1))):
                ta = torch.eye(shape[-1], dtype=dtype).expand(shape).clone().requires_grad_(); sa = native(ta, True)
                tb = torch.zeros(rhs, dtype=dtype, requires_grad=True); sb = native(tb, True)
                actual, expected = linalg.solve_ex(sa, sb), torch.linalg.solve_ex(ta, tb)
                self.gradients(actual.result, expected.result, [('empty A', sa, ta), ('empty B', sb, tb)])
                self.close(actual.info, expected.info.to(torch.int64))
            for shape in ((0, 0), (2, 0, 0), (0, 2, 2)):
                ta = torch.zeros(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                actual, expected = linalg.inv_ex(sa), torch.linalg.inv_ex(ta)
                self.gradients(actual.inverse, expected.inverse, [('empty inverse A', sa, ta)])
                self.close(actual.info, expected.info.to(torch.int64))
        a = fr.eye(2).reshape(1, 2, 2).requires_grad_(True); b = fr.zeros(1 << 60, 2, 0, requires_grad=True)
        result = linalg.solve_ex(a, b); result.result.sum().backward()
        self.assertEqual(result.info.tolist(), [0]); self.assertEqual(b.grad.shape, b.shape)
        huge = fr.zeros(1 << 60, 0, 0)
        with self.assertRaisesRegex(ValueError, 'capacity|large'): linalg.solve_ex(huge, huge)
        with self.assertRaisesRegex(ValueError, 'capacity|large'): linalg.inv_ex(huge)
        a = fr.eye(2).requires_grad_(True)
        for function in (lambda value, **kw: linalg.inv_ex(value, **kw), lambda value, **kw: linalg.solve_ex(value, fr.ones(2, dtype=value.dtype), **kw)):
            with self.assertRaises(TypeError): function(a, check_errors=1)
            with self.assertRaises(NotImplementedError): function(a, out=())
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: function(a)[0])
            from ferro.autograd import grad
            with self.assertRaises(ValueError): grad(function(a)[0].sum(), [a], create_graph=True)
            with self.assertRaises(ValueError): function(fr.tensor([[float('nan'), 0.], [0., 1.]]))
            with self.assertRaises(ValueError): function(fr.ones(2, 2, dtype='int64'))
        with self.assertRaises(ValueError): linalg.solve_ex(a, fr.ones(2, dtype='float64'))

    def test_lu_factor_rectangular_batched_pivots_and_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((3, 3), (5, 2), (2, 5), (2, 4, 3), (2, 2, 4)):
                for extended in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, extended=extended):
                        ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1553), requires_grad=True); sa = native(ta, True)
                        actual = linalg.lu_factor_ex(sa, check_errors=True) if extended else linalg.lu_factor(sa)
                        expected = torch.linalg.lu_factor_ex(ta, check_errors=True) if extended else torch.linalg.lu_factor(ta)
                        self.gradients(actual.LU, expected.LU, [('A', sa, ta)])
                        self.close(actual.pivots, expected.pivots.to(torch.int64))
                        self.assertEqual(expected.pivots.dtype, torch.int32); self.assertFalse(actual.pivots.requires_grad)
                        if extended: self.close(actual.info, expected.info.to(torch.int64))
            ta = torch.randn(2, 3, 4, dtype=dtype, generator=torch.Generator().manual_seed(1559), requires_grad=True); sa = native(ta, True)
            self.gradients(linalg.lu_factor(sa.transpose(-1, -2)).LU, torch.linalg.lu_factor(ta.transpose(-1, -2)).LU, [('strided parent', sa, ta)])

    def test_lu_unpacked_reconstruction_values_and_both_factor_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((3, 3), (5, 2), (2, 5), (2, 4, 3)):
                ta = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1567), requires_grad=True); sa = native(ta, True)
                actual, expected = linalg.lu(sa), torch.linalg.lu(ta)
                for got, wanted in zip(actual, expected): self.close(got, wanted)
                self.assertFalse(actual.P.requires_grad)
                self.close(actual.P @ actual.L @ actual.U, ta)
                self.gradients(actual.L.square().sum()*.4 + actual.U.square().sum()*.7,
                               expected.L.square().sum()*.4 + expected.U.square().sum()*.7, [('A', sa, ta)])

    def test_lu_solve_independent_packed_and_rhs_gradients_all_directions(self):
        for dtype in (torch.float32, torch.float64):
            for left in (False, True):
                for adjoint in (False, True):
                    for batch in (False, True):
                        generator = torch.Generator().manual_seed(1571)
                        shape = (2, 1, 3, 3) if batch else (3, 3)
                        source = torch.randn(shape, dtype=dtype, generator=generator)+torch.eye(3, dtype=dtype)*.7
                        packed, pivots = torch.linalg.lu_factor(source)
                        ta = packed.detach().transpose(-1, -2).contiguous().requires_grad_(); sa = native(ta, True)
                        rhs_shape = ((1, 4) if batch else ()) + ((3, 2) if left else (2, 3))
                        tb = torch.randn(rhs_shape, dtype=dtype, generator=generator, requires_grad=True); sb = native(tb, True)
                        sp = native(pivots.to(torch.int64))
                        actual = linalg.lu_solve(sa.transpose(-1, -2), sp, sb, left=left, adjoint=adjoint)
                        expected = torch.linalg.lu_solve(ta.transpose(-1, -2), pivots, tb, left=left, adjoint=adjoint)
                        self.gradients(actual, expected, [('independent strided LU', sa, ta), ('B', sb, tb)])
                    ta = source.detach().requires_grad_(); sa = native(ta, True)
                    tb = tb.detach().requires_grad_(); sb = native(tb, True)
                    actual_lu, expected_lu = linalg.lu_factor(sa), torch.linalg.lu_factor(ta)
                    self.gradients(linalg.lu_solve(*actual_lu, sb, left=left, adjoint=adjoint),
                                   torch.linalg.lu_solve(*expected_lu, tb, left=left, adjoint=adjoint), [('factor-to-solve A', sa, ta), ('B', sb, tb)])

    def test_lu_status_singular_factors_and_explicit_pivot_scope(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.tensor([[[0., 1.], [0., 2.]], [[1., 2.], [2., 4.]], [[0., 0.], [0., 0.]]], dtype=dtype)
            sa = native(ta); actual, expected = linalg.lu_factor_ex(sa), torch.linalg.lu_factor_ex(ta)
            self.close(actual.LU, expected.LU)
            self.close(actual.pivots, expected.pivots.to(torch.int64)); self.close(actual.info, expected.info.to(torch.int64))
            for got, wanted in zip(linalg.lu(sa), torch.linalg.lu(ta)): self.close(got, wanted)
            with self.assertRaises(ValueError): linalg.lu_factor(sa)
            with self.assertRaises(ValueError): linalg.lu_factor_ex(sa, check_errors=True)
        for method in (linalg.lu_factor, linalg.lu_factor_ex, linalg.lu):
            with self.assertRaises(NotImplementedError): method(fr.eye(2), pivot=False)
            with self.assertRaises(TypeError): method(fr.eye(2), pivot=1)
            with self.assertRaises(NotImplementedError): method(fr.eye(2), out=())
        packed, pivots = linalg.lu_factor(fr.eye(2))
        for values in ([0, 2], [1, 3], [-(1 << 63), 2]):
            with self.assertRaises(ValueError): linalg.lu_solve(packed, fr.tensor(values, dtype='int64'), fr.ones(2, 1))
        for bad in (pivots.to(dtype='float32'), pivots.reshape(1, 2)):
            with self.assertRaises(ValueError): linalg.lu_solve(packed, bad, fr.ones(2, 1))
        with self.assertRaises(ValueError): linalg.lu_solve(packed, pivots, fr.ones(2))
        with self.assertRaises(ValueError): linalg.lu_solve(packed, pivots, fr.ones(2, 1, dtype='float64'))
        with self.assertRaises(NotImplementedError): linalg.lu_solve(packed, pivots, fr.ones(2, 1), out=fr.ones(2, 1))
        tp = torch.tensor([2, 1], dtype=torch.int32)
        self.close(linalg.lu_solve(packed, native(tp.to(torch.int64)), fr.ones(2, 1)), torch.linalg.lu_solve(torch.eye(2), tp, torch.ones(2, 1)))

    def test_lu_saved_factor_pivots_and_unpacked_sibling_values(self):
        for dtype in (torch.float32, torch.float64):
            for selected in ('A', 'LU', 'pivots', 'info'):
                ta = torch.tensor([[.2, 2.1], [1.1, .3]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                actual, expected = linalg.lu_factor_ex(sa), torch.linalg.lu_factor_ex(ta)
                got = sa if selected == 'A' else getattr(actual, selected)
                wanted = ta if selected == 'A' else getattr(expected, selected)
                with fr.no_grad(): got.detach().copy_(fr.ones(got.shape, dtype=got.dtype))
                with torch.no_grad(): wanted.fill_(1.)
                if selected in ('LU', 'pivots'):
                    with self.assertRaises((ValueError, RuntimeError)): actual.LU.sum().backward()
                    with self.assertRaises(RuntimeError): expected.LU.sum().backward()
                else: self.gradients(actual.LU, expected.LU, [('A', sa, ta)])
            for selected in ('P', 'L', 'U'):
                for differentiated in ('L', 'U'):
                    ta = torch.tensor([[.2, 2.1], [1.1, .3]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                    actual, expected = linalg.lu(sa), torch.linalg.lu(ta)
                    got, wanted = getattr(actual, selected), getattr(expected, selected)
                    with fr.no_grad(): got.detach().copy_(fr.ones(got.shape, dtype=got.dtype))
                    with torch.no_grad(): wanted.fill_(1.)
                    with self.assertRaises((ValueError, RuntimeError)): getattr(actual, differentiated).sum().backward()
                    with self.assertRaises(RuntimeError): getattr(expected, differentiated).sum().backward()
            for selected in ('LU', 'pivots', 'B', 'output'):
                ta, tp = torch.linalg.lu_factor(torch.tensor([[.2, 2.1], [1.1, .3]], dtype=dtype)); ta.requires_grad_(True)
                tb = torch.tensor([[.2], [.4]], dtype=dtype, requires_grad=True); sa, sp, sb = native(ta, True), native(tp.to(torch.int64)), native(tb, True)
                actual, expected = linalg.lu_solve(sa, sp, sb), torch.linalg.lu_solve(ta, tp, tb)
                got = dict(LU=sa, pivots=sp, B=sb, output=actual)[selected]; wanted = dict(LU=ta, pivots=tp, B=tb, output=expected)[selected]
                with fr.no_grad(): got.detach().copy_(fr.ones(got.shape, dtype=got.dtype))
                with torch.no_grad(): wanted.fill_(1.)
                if selected != 'B':
                    with self.assertRaises((ValueError, RuntimeError)): actual.sum().backward()
                    with self.assertRaises(RuntimeError): expected.sum().backward()
                else: self.gradients(actual, expected, [('LU', sa, ta), ('B', sb, tb)])

    def test_lu_empty_geometry_capacity_and_graph_guards(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((0, 0), (2, 0), (0, 2), (0, 2, 2)):
                ta = torch.zeros(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                actual, expected = linalg.lu_factor_ex(sa), torch.linalg.lu_factor_ex(ta)
                self.gradients(actual.LU, expected.LU, [('empty A', sa, ta)])
                self.close(actual.pivots, expected.pivots.to(torch.int64)); self.close(actual.info, expected.info.to(torch.int64))
                ta.grad = None; sa.zero_grad()
                actual, expected = linalg.lu(sa), torch.linalg.lu(ta)
                for got, wanted in zip(actual, expected): self.close(got, wanted)
                self.gradients(actual.L.sum()+actual.U.sum(), expected.L.sum()+expected.U.sum(), [('empty unpacked A', sa, ta)])
        huge = fr.zeros(1 << 60, 0, 0, requires_grad=True)
        factor, pivots = linalg.lu_factor(huge); factor.sum().backward()
        self.assertEqual(huge.grad.shape, huge.shape); self.assertEqual(pivots.shape, [1 << 60, 0])
        self.assertEqual(linalg.lu(huge).P.shape, huge.shape)
        self.assertEqual(linalg.lu_solve(factor, pivots, huge).shape, huge.shape)
        with self.assertRaisesRegex(ValueError, 'capacity|large'): linalg.lu_factor_ex(huge)
        with self.assertRaisesRegex(ValueError, 'capacity|large'): linalg.lu(fr.zeros(1 << 30, 0))
        a = fr.eye(2).reshape(1, 2, 2).requires_grad_(True); factor, pivots = linalg.lu_factor(a)
        b = fr.zeros(1 << 60, 2, 0, requires_grad=True)
        result = linalg.lu_solve(factor, pivots, b); result.sum().backward(); self.assertEqual(b.grad.shape, b.shape)
        from ferro.autograd import grad
        for method in (lambda x: linalg.lu_factor(x).LU, lambda x: linalg.lu_factor_ex(x).LU, lambda x: linalg.lu(x).L):
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: method(a))
            with self.assertRaises(ValueError): grad(method(a).sum(), [a], create_graph=True)
            with self.assertRaises(ValueError): method(fr.ones(2, 2, dtype='int64'))
        with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.lu_solve(factor, pivots, fr.ones(2, 1)))

    def test_matrix_power_signed_exponents_batched_strided_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((1, 1), (3, 3), (2, 3, 3)):
                for exponent in (-5, -3, -2, -1, 0, 1, 2, 3, 4, 7, 16):
                    with self.subTest(dtype=dtype, shape=shape, exponent=exponent):
                        ta = (torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1597))*.05 + torch.eye(shape[-1], dtype=dtype)).requires_grad_()
                        sa = native(ta, True)
                        self.gradients(linalg.matrix_power(sa.transpose(-1, -2), exponent), torch.linalg.matrix_power(ta.transpose(-1, -2), exponent), [('A', sa, ta)])
        for exponent in ((1 << 63) - 1, -((1 << 63) - 1)):
            self.assertEqual(linalg.matrix_power(fr.ones(1, 1), exponent).tolist(), [[1.]])

    def test_matrix_power_zero_nonfinite_copies_and_saved_versions(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.tensor([[float('nan'), 0.], [0., float('inf')]], dtype=dtype, requires_grad=True); sa = native(ta, True)
            self.gradients(linalg.matrix_power(sa, 0), torch.linalg.matrix_power(ta, 0), [('nonfinite A zero gradient', sa, ta)])
            for exponent in (0, 1, 2, 3, 4, -1, -2):
                for selected in ('input', 'output'):
                    ta = torch.tensor([[1., .2], [.1, 1.2]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                    actual, expected = linalg.matrix_power(sa, exponent), torch.linalg.matrix_power(ta, exponent)
                    self.assertFalse(actual._shares_storage(sa))
                    if exponent in (0, 1): self.assertEqual(actual.stride(), expected.stride())
                    got, wanted = (sa, ta) if selected == 'input' else (actual, expected)
                    with fr.no_grad(): got.detach().copy_(fr.ones(got.shape, dtype=got.dtype))
                    with torch.no_grad(): wanted.fill_(1.)
                    reject = (selected == 'input' and exponent > 1) or (selected == 'output' and exponent == -1)
                    if reject:
                        with self.assertRaises((ValueError, RuntimeError)): actual.sum().backward()
                        with self.assertRaises(RuntimeError): expected.sum().backward()
                    else: self.gradients(actual, expected, [('A', sa, ta)])

    def test_multi_dot_matrix_vector_endpoints_and_all_input_gradients(self):
        cases = (((3,), (3,)), ((3,), (3, 2), (2,)), ((2, 3), (3, 4), (4, 2)),
                 ((3,), (3, 2), (2, 4), (4,)), ((2, 3), (3, 4), (4, 2), (2, 5), (5, 1)))
        for dtype in (torch.float32, torch.float64):
            for shapes in cases:
                expected_inputs = [torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1601+i), requires_grad=True) for i, shape in enumerate(shapes)]
                actual_inputs = [native(value, True) for value in expected_inputs]
                self.gradients(linalg.multi_dot(actual_inputs), torch.linalg.multi_dot(expected_inputs),
                               [(f'input{i}', a, b) for i, (a, b) in enumerate(zip(actual_inputs, expected_inputs))])
            ta = torch.randn(3, 2, dtype=dtype, generator=torch.Generator().manual_seed(1613), requires_grad=True)
            tb = torch.randn(3, 4, dtype=dtype, generator=torch.Generator().manual_seed(1619), requires_grad=True)
            tc = torch.randn(4, 2, dtype=dtype, generator=torch.Generator().manual_seed(1621), requires_grad=True)
            sa, sb, sc = native(ta, True), native(tb, True), native(tc, True)
            self.gradients(linalg.multi_dot((sa.narrow(1, 0, 1).squeeze(1), sb, sc.narrow(1, 1, 1).squeeze(1))),
                           torch.linalg.multi_dot((ta[:, 0], tb, tc[:, 1])), [('strided left vector', sa, ta), ('middle', sb, tb), ('strided right vector', sc, tc)])

    def test_multi_dot_three_operand_ties_preserve_torch_overflow_and_underflow(self):
        for values in ((1e30, 1e30, 1e-30), (1e30, 1e-30, 1e-30), (1e-30, 1e30, 1e30)):
            expected = [torch.tensor([[value]], dtype=torch.float32, requires_grad=True) for value in values]
            actual = [native(value, True) for value in expected]
            output, target = linalg.multi_dot(actual), torch.linalg.multi_dot(expected)
            self.assertEqual(output.item(), target.item())
            output.sum().backward(); target.sum().backward()
            for a, b in zip(actual, expected): self.assertEqual(a.grad.tolist(), b.grad.tolist())

    def test_matrix_composition_empty_shapes_and_addressable_chain_plans(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((0, 0), (2, 0, 0), (0, 3, 3)):
                for exponent in (-2, -1, 0, 1, 3):
                    ta = torch.empty(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                    self.gradients(linalg.matrix_power(sa, exponent), torch.linalg.matrix_power(ta, exponent), [('empty A', sa, ta)])
            for shapes in (((2, 0), (0, 3), (3, 1)), ((0,), (0, 2), (2, 0), (0,))):
                expected = [torch.zeros(shape, dtype=dtype, requires_grad=True) for shape in shapes]
                actual = [native(value, True) for value in expected]
                self.gradients(linalg.multi_dot(actual), torch.linalg.multi_dot(expected), [(str(i), a, b) for i, (a, b) in enumerate(zip(actual, expected))])
        huge = 1 << 30
        for shapes in (((huge, 0), (0, huge), (huge, 0)), ((0, huge), (huge, 0), (0, huge))):
            inputs = [fr.zeros(shape, requires_grad=True) for shape in shapes]
            output = linalg.multi_dot(inputs); output.sum().backward()
            self.assertEqual(output.shape, [shapes[0][0], shapes[-1][1]])
            for value in inputs: self.assertEqual(value.grad.shape, value.shape)
        with self.assertRaisesRegex(ValueError, 'allocation plan'):
            linalg.multi_dot((fr.zeros(huge, 0), fr.zeros(0, huge)))
        empty = fr.zeros(1 << 60, 0, 0, requires_grad=True)
        for exponent in (-1, 0, 1, 7): self.assertEqual(linalg.matrix_power(empty, exponent).shape, empty.shape)

    def test_matrix_composition_validation_and_multi_dot_saved_values(self):
        for n in (True, 1., None):
            with self.assertRaises(TypeError): linalg.matrix_power(fr.eye(2), n)
        for n in (1 << 63, -(1 << 63)-1):
            with self.assertRaises(ValueError): linalg.matrix_power(fr.eye(2), n)
        for value in (fr.ones(2), fr.ones(2, 3), fr.ones(2, 2, dtype='int64')):
            with self.assertRaises(ValueError): linalg.matrix_power(value, 2)
        with self.assertRaises(ValueError): linalg.matrix_power(fr.ones(2, 2), -1)
        with self.assertRaises(NotImplementedError): linalg.matrix_power(fr.eye(2), 1, out=fr.eye(2))
        for tensors in ([], [fr.eye(2)], [fr.ones(2, 3), fr.ones(2, 2)], [fr.eye(2), fr.ones(2), fr.eye(2)], [fr.ones(1, 2, 2), fr.eye(2)]):
            with self.assertRaises(ValueError): linalg.multi_dot(tensors)
        with self.assertRaises(TypeError): linalg.multi_dot(iter((fr.eye(2), fr.eye(2))))
        with self.assertRaises(ValueError): linalg.multi_dot((fr.eye(2), fr.eye(2, dtype='float64')))
        with self.assertRaises(NotImplementedError): linalg.multi_dot((fr.eye(2), fr.eye(2)), out=fr.eye(2))
        with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.matrix_power(fr.eye(2), 0))
        with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.multi_dot((fr.eye(2), fr.eye(2))))
        for selected in (0, 1, 2, 'output'):
            expected = [torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1627+i), requires_grad=True) for i, shape in enumerate(((3,), (3, 2), (2,)))]
            actual = [native(value, True) for value in expected]
            output, target = linalg.multi_dot(actual), torch.linalg.multi_dot(expected)
            got, wanted = (output, target) if selected == 'output' else (actual[selected], expected[selected])
            with fr.no_grad(): got.detach().copy_(fr.ones(got.shape, dtype=got.dtype))
            with torch.no_grad(): wanted.fill_(1.)
            if selected == 'output': self.gradients(output, target, [(str(i), a, b) for i, (a, b) in enumerate(zip(actual, expected))])
            else:
                with self.assertRaises((ValueError, RuntimeError)): output.backward()
                with self.assertRaises(RuntimeError): target.backward()

    def test_cholesky_factor_solves_strided_broadcast_inputs_and_full_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                for ashape, bshape in (((3, 3), (3, 2)), ((1, 3, 3), (4, 3, 2)), ((2, 1, 3, 3), (1, 4, 3, 2)), ((2, 3, 3), (3, 1))):
                    for inverse in (False, True):
                        with self.subTest(dtype=dtype, upper=upper, ashape=ashape, bshape=bshape, inverse=inverse):
                            generator = torch.Generator().manual_seed(1693)
                            ta = (torch.randn(ashape, generator=generator, dtype=dtype) * .2 + torch.eye(3, dtype=dtype) * 2).requires_grad_()
                            tb = torch.randn(bshape[:-2] + bshape[-2:][::-1], generator=generator, dtype=dtype, requires_grad=True)
                            sa, sb = native(ta, True), native(tb, True)
                            actual = linalg.cholesky_inverse(sa.transpose(-2, -1), upper) if inverse else linalg.cholesky_solve(sb.transpose(-2, -1), sa.transpose(-2, -1), upper)
                            expected = torch.cholesky_inverse(ta.transpose(-2, -1), upper=upper) if inverse else torch.cholesky_solve(tb.transpose(-2, -1), ta.transpose(-2, -1), upper=upper)
                            inputs = [('full strided factor', sa, ta)] + ([] if inverse else [('strided RHS', sb, tb)])
                            self.gradients(actual, expected, inputs)
                            if inverse:
                                value = torch.tensor(actual.tolist(), dtype=dtype)
                                torch.testing.assert_close(value, value.transpose(-2, -1), atol=0, rtol=0)

    def test_cholesky_factor_triangular_tangents_finite_differences_and_unmasked_boundary(self):
        values = torch.tensor([[2., .3], [-.2, 1.1]], dtype=torch.float64)
        rhs = native(torch.tensor([[.3, .5], [.7, -.4]], dtype=torch.float64))
        for upper in (False, True):
            for inverse in (False, True):
                def evaluate(a):
                    selected = a.triu() if upper else a.tril()
                    return linalg.cholesky_inverse(selected, upper) if inverse else linalg.cholesky_solve(rhs, selected, upper)
                a = native(values, True)
                evaluate(a).sum().backward()
                for row in range(2):
                    for column in range(2):
                        high, low = values.clone(), values.clone()
                        high[row, column] += 1e-6; low[row, column] -= 1e-6
                        difference = (float(evaluate(native(high)).sum().tolist()) - float(evaluate(native(low)).sum().tolist())) / 2e-6
                        self.assertAlmostEqual(a.grad.tolist()[row][column], difference, delta=3e-9)
        ta = torch.tensor([[2., .3], [.5, 1.]], dtype=torch.float64, requires_grad=True); sa = native(ta, True)
        tb = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64); sb = native(tb)
        linalg.cholesky_solve(sb, sa).sum().backward()
        torch.cholesky_solve(tb, ta).sum().backward()
        self.close(sa.grad, ta.grad)
        self.assertEqual(sa.grad.tolist()[0][1], .24921875)
        plus, minus = ta.detach().clone(), ta.detach().clone()
        plus[0, 1] += 1e-5; minus[0, 1] -= 1e-5
        difference = (linalg.cholesky_solve(sb, native(plus)).sum() - linalg.cholesky_solve(sb, native(minus)).sum()).tolist()
        self.assertEqual(difference, 0.)

    def test_cholesky_factor_empty_geometry_and_huge_zero_output_paths(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                for ashape, bshape in (((0, 0), (0, 2)), ((2, 0, 0), (0, 1)), ((0, 2, 2), (2, 1)), ((2, 2), (2, 0))):
                    for inverse in (False, True):
                        ta = torch.eye(ashape[-1], dtype=dtype).expand(ashape).clone().requires_grad_()
                        tb = torch.ones(bshape, dtype=dtype, requires_grad=True)
                        sa, sb = native(ta, True), native(tb, True)
                        actual = linalg.cholesky_inverse(sa, upper) if inverse else linalg.cholesky_solve(sb, sa, upper)
                        expected = torch.cholesky_inverse(ta, upper=upper) if inverse else torch.cholesky_solve(tb, ta, upper=upper)
                        self.gradients(actual, expected, [('factor', sa, ta)] + ([] if inverse else [('RHS', sb, tb)]))
            factor = fr.zeros(1 << 60, 0, 0, dtype=str(dtype).split('.')[-1], requires_grad=True)
            value = linalg.cholesky_inverse(factor)
            value.sum().backward()
            self.assertEqual(value.shape, factor.shape)
            self.assertEqual(factor.grad.shape, factor.shape)
            factor = fr.eye(2, dtype=str(dtype).split('.')[-1], requires_grad=True)
            rhs = fr.zeros(1 << 60, 2, 0, dtype=factor.dtype, requires_grad=True)
            value = linalg.cholesky_solve(rhs, factor)
            value.sum().backward()
            self.assertEqual(value.shape, rhs.shape)
            self.assertEqual(rhs.grad.shape, rhs.shape)
            self.assertEqual(factor.grad.tolist(), [[0., 0.], [0., 0.]])

    def test_cholesky_factor_ignored_nonfinite_entries_and_singular_contract(self):
        for dtype in (torch.float32, torch.float64):
            for upper in (False, True):
                for inverse in (False, True):
                    ta = torch.tensor([[2., float('nan')], [.5, 1.]], dtype=dtype)
                    if upper: ta = ta.transpose(0, 1).contiguous()
                    ta.requires_grad_(); sa = native(ta, True)
                    tb = torch.ones(2, 1, dtype=dtype); sb = native(tb)
                    actual = linalg.cholesky_inverse(sa, upper) if inverse else linalg.cholesky_solve(sb, sa, upper)
                    expected = torch.cholesky_inverse(ta, upper=upper) if inverse else torch.cholesky_solve(tb, ta, upper=upper)
                    self.gradients(actual, expected, [('ignored NaN factor', sa, ta)])
                singular = fr.tensor([[0., 0.], [0., 1.]], dtype=str(dtype).split('.')[-1])
                with self.assertRaises(ValueError): linalg.cholesky_inverse(singular, upper)
                result = linalg.cholesky_solve(fr.ones(2, 1, dtype=singular.dtype), singular, upper)
                self.assertTrue(any(not math.isfinite(value) for value in result.reshape(-1).tolist()))
                for bshape in ((2, 0), (0, 2, 1)):
                    ta = torch.tensor([[2., float('nan')], [.5, 1.]], dtype=dtype)
                    if upper: ta = ta.transpose(0, 1).contiguous()
                    ta.requires_grad_(); sa = native(ta, True)
                    tb = torch.empty(bshape, dtype=dtype, requires_grad=True); sb = native(tb, True)
                    self.gradients(linalg.cholesky_solve(sb, sa, upper), torch.cholesky_solve(tb, ta, upper=upper), [('empty ignored NaN factor', sa, ta), ('empty RHS', sb, tb)])

    def test_cholesky_factor_saved_operands_and_output_versions(self):
        from ferro.autograd import grad
        for dtype in (torch.float32, torch.float64):
            for inverse in (False, True):
                for modified in (('factor', 'output') if inverse else ('factor', 'rhs', 'output')):
                    for functional in (False, True):
                        ta = torch.tensor([[2., .3], [.5, 1.]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                        tb = torch.ones(2, 1, dtype=dtype, requires_grad=True); sb = native(tb, True)
                        actual = linalg.cholesky_inverse(sa) if inverse else linalg.cholesky_solve(sb, sa)
                        expected = torch.cholesky_inverse(ta) if inverse else torch.cholesky_solve(tb, ta)
                        value = {'factor': sa, 'rhs': sb, 'output': actual}[modified]
                        wanted = {'factor': ta, 'rhs': tb, 'output': expected}[modified]
                        value.detach().copy_(fr.full(value.shape, 3., dtype=value.dtype)); wanted.detach().fill_(3.)
                        with self.assertRaises((ValueError, RuntimeError)):
                            grad(actual.sum(), [sa]) if functional else actual.sum().backward()
                        with self.assertRaises(RuntimeError):
                            torch.autograd.grad(expected.sum(), [ta]) if functional else expected.sum().backward()
                        self.assertIsNone(sa.grad); self.assertIsNone(sb.grad)

    def test_cholesky_factor_validation_capacity_capture_and_public_exports(self):
        from ferro.autograd import grad
        for function in (linalg.cholesky_inverse, lambda a, **kw: linalg.cholesky_solve(fr.ones(2, 1, dtype=a.dtype), a, **kw)):
            for shape in ((2,), (2, 3)):
                with self.assertRaises(ValueError): function(fr.ones(shape))
            for dtype in ('int64', 'bool'):
                with self.assertRaises(ValueError): function(fr.ones(2, 2, dtype=dtype))
            with self.assertRaises(TypeError): function(fr.eye(2), upper=1)
            with self.assertRaises(NotImplementedError): function(fr.eye(2), out=fr.eye(2))
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: function(fr.eye(2)))
            with self.assertRaises(ValueError): function(fr.ones(1).expand(1 << 30, 1 << 30))
            with self.assertRaises(ValueError): function(fr.zeros(0, 1 << 30, 1 << 30))
            a = fr.eye(2).requires_grad_(True)
            with self.assertRaises(ValueError): grad(function(a).sum(), [a], create_graph=True)
        with self.assertRaises(ValueError): linalg.cholesky_solve(fr.ones(2), fr.eye(2))
        with self.assertRaises(ValueError): linalg.cholesky_solve(fr.ones(3, 1), fr.eye(2))
        with self.assertRaises(ValueError): linalg.cholesky_solve(fr.ones(2, 1, dtype='float64'), fr.eye(2))
        self.close(fr.cholesky_inverse(input=fr.eye(2)), torch.eye(2))
        self.close(fr.cholesky_solve(input=fr.ones(2, 1), input2=fr.eye(2)), torch.ones(2, 1))

    def test_cross_broadcast_dimensions_strides_and_exact_integer_overflow(self):
        cases = (((3,), (3,), -1), ((2, 3), (1, 3), -1), ((3, 2), (3, 1), 0), ((2, 1, 3), (1, 4, 3), -1))
        for dtype in (torch.float32, torch.float64):
            for ashape, bshape, dim in cases:
                ta = torch.randn(ashape, dtype=dtype, generator=torch.Generator().manual_seed(1721), requires_grad=True)
                tb = torch.randn(bshape, dtype=dtype, generator=torch.Generator().manual_seed(1723), requires_grad=True)
                sa, sb = native(ta, True), native(tb, True)
                self.gradients(linalg.cross(sa, sb, dim=dim), torch.linalg.cross(ta, tb, dim=dim), [('input', sa, ta), ('other', sb, tb)])
            ta = torch.randn(2, 3, dtype=dtype, generator=torch.Generator().manual_seed(1733), requires_grad=True)
            tb = torch.randn(1, 3, dtype=dtype, generator=torch.Generator().manual_seed(1741), requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            self.gradients(linalg.cross(sa.T, sb.T, dim=0), torch.linalg.cross(ta.T, tb.T, dim=0), [('strided input', sa, ta), ('strided other', sb, tb)])
        for av, bv in (([2**62, -3, 7], [-9, 2**62, 11]), ([0, 1, -2], [4, 5, 6])):
            ta, tb = torch.tensor(av, dtype=torch.int64), torch.tensor(bv, dtype=torch.int64)
            self.close(linalg.cross(native(ta), native(tb)), torch.linalg.cross(ta, tb))

    def test_vecdot_broadcast_contraction_scalar_and_all_input_gradients(self):
        cases = (((2, 3), (3,), -1), ((2, 3), (3,), 0), ((2, 3), (2, 1), -1), ((2, 1, 3), (4, 3), 1),
                 ((3, 2), (3, 1), 0), ((), (), -1), ((), (), 0), ((), (3,), -1), ((3,), (), 0))
        for dtype in (torch.float32, torch.float64):
            for ashape, bshape, dim in cases:
                ta = torch.randn(ashape, dtype=dtype, generator=torch.Generator().manual_seed(1747), requires_grad=True)
                tb = torch.randn(bshape, dtype=dtype, generator=torch.Generator().manual_seed(1753), requires_grad=True)
                sa, sb = native(ta, True), native(tb, True)
                self.gradients(linalg.vecdot(sa, sb, dim=dim), torch.linalg.vecdot(ta, tb, dim=dim), [('input', sa, ta), ('other', sb, tb)])
            ta = torch.randn(2, 3, dtype=dtype, generator=torch.Generator().manual_seed(1759), requires_grad=True)
            tb = torch.randn(1, 3, dtype=dtype, generator=torch.Generator().manual_seed(1777), requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            self.gradients(linalg.vecdot(sa.T, sb.T, dim=0), torch.linalg.vecdot(ta.T, tb.T, dim=0), [('strided input', sa, ta), ('strided other', sb, tb)])

    def test_vector_products_empty_huge_metadata_and_reduction_capacity(self):
        for dtype in (torch.float32, torch.float64):
            for name, cases in (('cross', (((0, 3), (1, 3), -1), ((3, 0), (3, 1), 0))),
                                ('vecdot', (((0, 3), (1, 3), -1), ((2, 0), (1, 0), -1), ((0, 2), (0, 1), 0)))):
                for ashape, bshape, dim in cases:
                    ta = torch.empty(ashape, dtype=dtype, requires_grad=True)
                    tb = torch.ones(bshape, dtype=dtype, requires_grad=True)
                    sa, sb = native(ta, True), native(tb, True)
                    self.gradients(getattr(linalg, name)(sa, sb, dim=dim), getattr(torch.linalg, name)(ta, tb, dim=dim), [('empty input', sa, ta), ('other', sb, tb)])
            dtype_name = str(dtype).split('.')[-1]
            huge = fr.zeros(1 << 60, 0, 3, dtype=dtype_name, requires_grad=True)
            other = fr.ones(1, 1, 3, dtype=dtype_name, requires_grad=True)
            out = linalg.cross(huge, other)
            out.sum().backward()
            self.assertEqual(out.shape, huge.shape)
            self.assertEqual(huge.grad.shape, huge.shape)
            self.assertEqual(other.grad.tolist(), [[[0., 0., 0.]]])
            a = fr.zeros(0, 1 << 60, dtype=dtype_name, requires_grad=True)
            b = fr.ones(1, 1, dtype=dtype_name, requires_grad=True)
            out = linalg.vecdot(a, b)
            out.sum().backward()
            self.assertEqual(out.shape, [0])
            self.assertEqual(a.grad.shape, a.shape)
            self.assertEqual(b.grad.tolist(), [[0.]])
            with self.assertRaisesRegex(ValueError, 'capacity'):
                linalg.vecdot(fr.zeros(1 << 61, 0, dtype=dtype_name), fr.zeros(1, 0, dtype=dtype_name))

    def test_vector_products_mixed_higher_gradients_and_saved_input_versions(self):
        from ferro.autograd import grad
        for name in ('cross', 'vecdot'):
            ta = torch.tensor([.2, .5, -.7], requires_grad=True); tb = torch.tensor([.3, -.4, .9], requires_grad=True)
            sa, sb = native(ta, True), native(tb, True)
            actual, expected = getattr(linalg, name)(sa, sb), getattr(torch.linalg, name)(ta, tb)
            weight = torch.tensor([.7, -.1, .4]) if name == 'cross' else torch.tensor(.7)
            first = grad((actual * native(weight)).sum(), [sa], create_graph=True)[0]
            reference = torch.autograd.grad((expected * weight).sum(), ta, create_graph=True)[0]
            self.close(first, reference)
            probe = torch.tensor([.1, .6, .2])
            self.close(grad((first * native(probe)).sum(), [sb])[0], torch.autograd.grad((reference * probe).sum(), tb)[0])
            self.assertIsNone(sa.grad); self.assertIsNone(sb.grad)
            for dtype in (torch.float32, torch.float64):
                for modified in ('input', 'output'):
                    ta = torch.tensor([.2, .5, -.7], dtype=dtype, requires_grad=True)
                    tb = torch.tensor([.3, -.4, .9], dtype=dtype, requires_grad=True)
                    sa, sb = native(ta, True), native(tb, True)
                    actual, expected = getattr(linalg, name)(sa, sb), getattr(torch.linalg, name)(ta, tb)
                    value, wanted = (sa, ta) if modified == 'input' else (actual, expected)
                    value.detach().copy_(fr.full(value.shape, 9., dtype=value.dtype)); wanted.detach().fill_(9.)
                    if modified == 'input':
                        with self.assertRaises((ValueError, RuntimeError)): actual.sum().backward()
                        with self.assertRaises(RuntimeError): expected.sum().backward()
                        self.assertIsNone(sa.grad); self.assertIsNone(sb.grad)
                    else:
                        actual.sum().backward(); expected.sum().backward()
                        self.close(sa.grad, ta.grad); self.close(sb.grad, tb.grad)

    def test_vector_products_validation_and_capture_boundaries(self):
        for name in ('cross', 'vecdot'):
            function = getattr(linalg, name)
            for dim in (True, None, 1.):
                with self.assertRaises(TypeError): function(fr.ones(3), fr.ones(3), dim=dim)
            for dim in (-2, 1):
                with self.assertRaises(ValueError): function(fr.ones(3), fr.ones(3), dim=dim)
            with self.assertRaises(ValueError): function(fr.ones(3), fr.ones(3, dtype='float64'))
            with self.assertRaises(ValueError): function(fr.ones(3, dtype='bool'), fr.ones(3, dtype='bool'))
            with self.assertRaises(ValueError): function(fr.ones(2, 3), fr.ones(4, 3))
            with self.assertRaises(TypeError): function([1., 2., 3.], fr.ones(3))
            with self.assertRaises(NotImplementedError): function(fr.ones(3), fr.ones(3), out=fr.ones(3))
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: function(fr.ones(3), fr.ones(3)))
            with self.assertRaisesRegex(ValueError, 'capacity'):
                function(fr.ones(1).expand(1 << 30, 1 << 30, 3), fr.ones(1, 1, 3))
        with self.assertRaises(ValueError): linalg.cross(fr.ones(2, 3), fr.ones(3))
        with self.assertRaises(ValueError): linalg.cross(fr.ones(2, 3), fr.ones(2, 1))
        with self.assertRaises(ValueError): linalg.cross(fr.tensor(1.), fr.tensor(2.))
        with self.assertRaises(ValueError): linalg.vecdot(fr.ones(3, dtype='int64'), fr.ones(3, dtype='int64'))

    def test_condition_number_orders_strided_batches_and_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((3, 3), (2, 3, 3), (4, 2), (2, 4)):
                for order in (None, 2, -2, 'nuc', 'fro', 1, -1, math.inf, -math.inf):
                    if shape[-1] != shape[-2] and order not in (None, 2, -2):
                        continue
                    with self.subTest(dtype=dtype, shape=shape, order=order):
                        values = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(1657)) * .2
                        ta = (values + torch.eye(shape[-2], shape[-1], dtype=dtype) * 1.5).requires_grad_()
                        sa = native(ta, True)
                        actual = linalg.cond(sa.transpose(-2, -1), order)
                        matrix = ta.transpose(-2, -1)
                        expected = torch.linalg.cond(matrix, order)
                        self.close(actual, expected)
                        # Torch 2.13 cond backward mutates its saved inverse for these orders.
                        oracle = expected if order in (None, 2, -2, 'nuc') else torch.linalg.matrix_norm(matrix, order) * torch.linalg.matrix_norm(torch.linalg.inv(matrix), order)
                        self.gradients(actual, oracle, [('strided A parent', sa, ta)])

    def test_condition_number_inverse_norm_gradients_finite_differences(self):
        values = torch.tensor([[2., .3], [-.2, 1.1]], dtype=torch.float64)
        for order in (None, 2, -2, 'nuc', 'fro', 1, -1, math.inf, -math.inf):
            a = native(values, True)
            linalg.cond(a, order).backward()
            gradient = a.grad.tolist()
            for row in range(2):
                for column in range(2):
                    plus, minus = values.clone(), values.clone()
                    plus[row, column] += 1e-6
                    minus[row, column] -= 1e-6
                    difference = (float(linalg.cond(native(plus), order).tolist()) - float(linalg.cond(native(minus), order).tolist())) / 2e-6
                    with self.subTest(order=order, row=row, column=column):
                        self.assertAlmostEqual(gradient[row][column], difference, delta=2e-8)

    def test_condition_number_singular_zero_and_selected_spectral_ties(self):
        for dtype in (torch.float32, torch.float64):
            for values in ([[1., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]]):
                for order in (None, 2, -2, 'nuc', 'fro', 1, -1, math.inf, -math.inf):
                    with self.subTest(dtype=dtype, values=values, order=order):
                        ta = torch.tensor(values, dtype=dtype)
                        self.close(linalg.cond(native(ta), order), torch.linalg.cond(ta, order))
            for order in (None, 2, -2, 'nuc'):
                ta = torch.eye(2, dtype=dtype).requires_grad_(); sa = native(ta, True)
                self.gradients(linalg.cond(sa, order), torch.linalg.cond(ta, order), [('tied selected spectrum', sa, ta)])

    def test_condition_number_empty_outputs_are_disconnected_and_capacity_checked(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((0, 0), (2, 0, 0), (0, 2, 2), (0, 2, 3), (2, 0), (0, 2)):
                for order in (None, 2, -2, 'nuc', 'fro', 1, -1, math.inf, -math.inf, True):
                    with self.subTest(dtype=dtype, shape=shape, order=order):
                        ta = torch.empty(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                        if isinstance(order, str) and shape[-2] != shape[-1]:
                            with self.assertRaises(ValueError): linalg.cond(sa, order)
                            with self.assertRaises(RuntimeError): torch.linalg.cond(ta, order)
                        else:
                            actual, expected = linalg.cond(sa, order), torch.linalg.cond(ta, order)
                            self.close(actual, expected)
                            self.assertFalse(actual.requires_grad)
                            self.assertFalse(expected.requires_grad)
                            self.assertIsNone(sa.grad)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            linalg.cond(fr.zeros(1 << 60, 0, 0))

    def test_condition_number_validation_capture_and_higher_order_boundary(self):
        from ferro.autograd import grad
        for order in (False, 0, 3, float('nan'), 'bad'):
            for shape in ((2, 2), (0, 0)):
                with self.assertRaises(ValueError): linalg.cond(fr.zeros(shape), order)
        for order in ([], 1j, fr.tensor(2.)):
            with self.assertRaises(TypeError): linalg.cond(fr.eye(2), order)
        for shape in ((), (2,)):
            with self.assertRaises(ValueError): linalg.cond(fr.zeros(shape))
        for order in ('nuc', 'fro', 1, -1, math.inf, -math.inf):
            with self.assertRaises(ValueError): linalg.cond(fr.ones(2, 3), order)
        for dtype in ('int64', 'bool'):
            with self.assertRaises(ValueError): linalg.cond(fr.ones(2, 2, dtype=dtype))
        with self.assertRaises(TypeError): linalg.cond([[1., 0.], [0., 1.]])
        with self.assertRaises(NotImplementedError): linalg.cond(fr.eye(2), out=fr.tensor(0.))
        with self.assertRaisesRegex(ValueError, 'capacity'): linalg.cond(fr.ones(1).expand(1 << 30, 1 << 30))
        for order in (None, 'nuc', 1):
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: linalg.cond(fr.eye(2), order))
            a = fr.tensor([[2., .3], [-.2, 1.1]], requires_grad=True)
            with self.assertRaises(ValueError): grad(linalg.cond(a, order), [a], create_graph=True)
            with self.assertRaises(ValueError): linalg.cond(fr.tensor([[float('nan'), 0.], [0., 1.]]), order)
        self.close(linalg.cond(fr.eye(2), True), torch.linalg.cond(torch.eye(2), True))

    def test_condition_number_inverse_norm_saved_input_and_owning_output(self):
        for dtype in (torch.float32, torch.float64):
            for modified in ('input', 'output'):
                ta = torch.tensor([[2., .3], [-.2, 1.1]], dtype=dtype, requires_grad=True); sa = native(ta, True)
                actual = linalg.cond(sa, 1)
                expected = torch.linalg.matrix_norm(ta, 1) * torch.linalg.matrix_norm(torch.linalg.inv(ta), 1)
                wanted = ta if modified == 'input' else expected
                value = sa if modified == 'input' else actual
                value.detach().copy_(fr.full(value.shape, 3., dtype=value.dtype))
                wanted.detach().fill_(3.)
                if modified == 'input':
                    with self.assertRaises((ValueError, RuntimeError)): actual.backward()
                    with self.assertRaises(RuntimeError): expected.backward()
                    self.assertIsNone(sa.grad)
                else:
                    actual.backward(); expected.backward()
                    self.close(sa.grad, ta.grad)

    def test_tensor_inverse_split_permutation_and_all_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape, ind in (((2, 2), 1), ((2, 1, 2), 1), ((2, 1, 2), 2), ((2, 3, 2, 3), 2), ((1,), 1), ((1, 1), 2)):
                with self.subTest(dtype=dtype, shape=shape, ind=ind):
                    size = math.prod(shape[:ind])
                    ta = self.matrix((size, size), dtype).detach().reshape(shape).requires_grad_(); sa = native(ta, True)
                    self.gradients(linalg.tensorinv(sa, ind), torch.linalg.tensorinv(ta, ind), [('A', sa, ta)])
            ta = self.matrix((6, 6), dtype).detach().reshape(2, 3, 2, 3).requires_grad_(); sa = native(ta, True)
            self.gradients(linalg.tensorinv(sa.permute(1, 0, 3, 2)), torch.linalg.tensorinv(ta.permute(1, 0, 3, 2)), [('strided A parent', sa, ta)])

    def test_tensor_solve_reordered_dimensions_and_all_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape, rhs, dims in (((2, 3, 2, 3), (2, 3), None), ((2, 3, 2, 3), (3, 2), None),
                                     ((2, 3, 3, 2), (2, 3), (1, 3)), ((2, 3, 3, 2), (2, 3), (-3, -1)),
                                     ((2, 3, 2, 3), (2, 3), ())):
                with self.subTest(dtype=dtype, shape=shape, rhs=rhs, dims=dims):
                    values = self.matrix((6, 6), dtype).detach().reshape(shape)
                    if dims: values = values.permute(0, 2, 1, 3)
                    ta = values.contiguous().reshape(shape).requires_grad_(); sa = native(ta, True)
                    tb = torch.randn(rhs, dtype=dtype, generator=torch.Generator().manual_seed(1411), requires_grad=True); sb = native(tb, True)
                    self.gradients(linalg.tensorsolve(sa, sb, dims), torch.linalg.tensorsolve(ta, tb, dims=dims), [('A', sa, ta), ('B', sb, tb)])
            ta = self.matrix((6, 6), dtype).detach().reshape(2, 3, 2, 3).requires_grad_(); sa = native(ta, True)
            tb = torch.randn(2, 3, dtype=dtype, generator=torch.Generator().manual_seed(1423), requires_grad=True); sb = native(tb, True)
            self.gradients(linalg.tensorsolve(sa.permute(1, 0, 3, 2), sb.transpose(0, 1)),
                           torch.linalg.tensorsolve(ta.permute(1, 0, 3, 2), tb.transpose(0, 1)), [('strided A', sa, ta), ('strided B', sb, tb)])

    def test_tensor_algebra_scalars_and_empty_geometry_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape, rhs in (((), ()), ((1,), ()), ((1,), (1,)), ((1, 1), ()), ((0, 0), (0,)),
                               ((2, 0), (0,)), ((2, 0, 3), (0,)), ((2, 0, 3, 0), (2, 0))):
                with self.subTest(dtype=dtype, shape=shape, rhs=rhs):
                    ta = torch.full(shape, 2., dtype=dtype, requires_grad=True); tb = torch.full(rhs, .7, dtype=dtype, requires_grad=True)
                    sa, sb = native(ta, True), native(tb, True)
                    self.gradients(linalg.tensorsolve(sa, sb), torch.linalg.tensorsolve(ta, tb), [('empty/scalar A', sa, ta), ('empty/scalar B', sb, tb)])
            for shape, ind in (((0, 0), 1), ((0, 1, 0), 2), ((2, 0, 3, 0), 2)):
                ta = torch.empty(shape, dtype=dtype, requires_grad=True); sa = native(ta, True)
                self.gradients(linalg.tensorinv(sa, ind), torch.linalg.tensorinv(ta, ind), [('empty inverse A', sa, ta)])

    def test_tensor_algebra_validation_capacity_and_saved_outputs(self):
        from ferro.autograd import grad
        for ind in (0, -1, 3):
            with self.assertRaises(ValueError): linalg.tensorinv(fr.eye(2), ind)
        for ind in (True, 1., None):
            with self.assertRaises(TypeError): linalg.tensorinv(fr.eye(2), ind)
        with self.assertRaises(ValueError): linalg.tensorinv(fr.ones(2, 3), 1)
        with self.assertRaises(ValueError): linalg.tensorinv(fr.ones(2, 2), 1)
        for dims in ((0, 0), (-1, 1), (2,), (-3,)):
            with self.assertRaises(ValueError): linalg.tensorsolve(fr.eye(2), fr.ones(2), dims)
        for dims in (True, 1, (1.,), [True]):
            with self.assertRaises(TypeError): linalg.tensorsolve(fr.eye(2), fr.ones(2), dims)
        for a, b in ((fr.zeros(0, 2), fr.zeros(0)), (fr.ones(2, 2), fr.ones(3)), (fr.tensor(1.), fr.ones(1))):
            with self.assertRaises(ValueError): linalg.tensorsolve(a, b)
        with self.assertRaises(ValueError): linalg.tensorsolve(fr.eye(2), fr.ones(2, dtype='float64'))
        with self.assertRaises(TypeError): linalg.tensorinv([[1., 0.], [0., 1.]], 1)
        with self.assertRaises(TypeError): linalg.tensorsolve([[1., 0.], [0., 1.]], fr.ones(2))
        huge = fr.ones(1).expand(1 << 30, 1 << 30)
        with self.assertRaisesRegex(ValueError, 'capacity'): linalg.tensorinv(huge, 1)
        with self.assertRaisesRegex(ValueError, 'capacity'): linalg.tensorsolve(huge, fr.ones(1).expand(1 << 30))
        for function in (lambda a: linalg.tensorinv(a, 1), lambda a: linalg.tensorsolve(a, fr.ones(2, dtype=a.dtype))):
            for dtype in ('int64', 'bool'):
                with self.assertRaises(ValueError): function(fr.ones(2, 2, dtype=dtype))
            with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: function(fr.eye(2)))
            a = fr.eye(2).requires_grad_(True)
            with self.assertRaises(ValueError): grad(function(a).sum(), [a], create_graph=True)
        with self.assertRaises(NotImplementedError): linalg.tensorinv(fr.eye(2), 1, out=fr.eye(2))
        with self.assertRaises(NotImplementedError): linalg.tensorsolve(fr.eye(2), fr.ones(2), out=fr.ones(2))
        for dtype in ('float32', 'float64'):
            for inverse in (False, True):
                a = fr.tensor([[2., .3], [.1, 1.7]], dtype=dtype, requires_grad=True)
                output = linalg.tensorinv(a, 1) if inverse else linalg.tensorsolve(a, fr.ones(2, dtype=dtype))
                output.detach().copy_(fr.zeros(output.shape, dtype=dtype))
                with self.assertRaises((ValueError, RuntimeError)): output.sum().backward()
                self.assertIsNone(a.grad)

    def test_stalled_svd_null_column_reconstructs_and_preserves_orthogonal_subnormal(self):
        for dtype in (torch.float32, torch.float64):
            source = torch.randn(3, 3, generator=torch.Generator().manual_seed(1351)); source[-1] = 0.
            ta = source.to(dtype).requires_grad_(); sa = native(ta, True)
            actual, expected = linalg.svd(sa, False), torch.linalg.svd(ta, full_matrices=False)
            self.close(actual.S, expected.S)
            self.close((actual.U * actual.S) @ actual.Vh, ta)
            self.close(actual.U.transpose(0, 1) @ actual.U, torch.eye(3, dtype=dtype))
            self.close(actual.Vh @ actual.Vh.transpose(0, 1), torch.eye(3, dtype=dtype))
            self.gradients(linalg.pinv(sa), torch.linalg.pinv(ta), [('rank2 A', sa, ta)])
        tiny = math.ulp(0.)
        actual = linalg.svdvals(fr.tensor([[1., 0.], [0., tiny]], dtype='float64'))
        self.assertEqual(actual.tolist(), [1., tiny])

    def test_svd_saved_outputs_and_all_siblings_reject_mutation(self):
        from ferro.autograd import grad
        for dtype in (torch.float32, torch.float64):
            for selected in ('U', 'S', 'Vh'):
                for modified in ('U', 'S', 'Vh'):
                    for functional in (False, True):
                        ta = torch.tensor([[2., .2], [-.1, 1.7], [.4, -.3]], dtype=dtype, requires_grad=True)
                        sa = native(ta, True)
                        actual, expected = linalg.svd(sa, False), torch.linalg.svd(ta, full_matrices=False)
                        value, wanted = getattr(actual, modified), getattr(expected, modified)
                        value.detach().copy_(fr.ones(value.shape, dtype=str(dtype).split('.')[-1]))
                        wanted.detach().fill_(1.)
                        with self.assertRaises((ValueError, RuntimeError)):
                            grad(getattr(actual, selected).sum(), [sa]) if functional else getattr(actual, selected).sum().backward()
                        with self.assertRaises(RuntimeError):
                            torch.autograd.grad(getattr(expected, selected).sum(), [ta]) if functional else getattr(expected, selected).sum().backward()
                        self.assertIsNone(sa.grad); self.assertIsNone(ta.grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
