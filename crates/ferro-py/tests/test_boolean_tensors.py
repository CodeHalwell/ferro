"""Native Bool dtype, predicates, selection, and gradients against PyTorch."""
import operator
import sys
import unittest

import ferro as fr
import torch


class BooleanTensors(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.requires_grad, expected.requires_grad)

    def test_construction_factories_scalar_types_and_explicit_cast(self):
        for values in (True, False, [True, False], [[True], [False]], [], [[], []]):
            self.same(fr.tensor(values, dtype='bool'), torch.tensor(values, dtype=torch.bool))
        self.same(fr.tensor([True, False]), torch.tensor([True, False]))
        self.same(fr.tensor([True, 2]), torch.tensor([True, 2]))
        self.same(fr.tensor([True, 2.5]), torch.tensor([True, 2.5]))
        for shape in ((), (0,), (2, 0, 3), (2, 3)):
            for name in ('zeros', 'ones'):
                self.same(getattr(fr, name)(shape, dtype='bool'), getattr(torch, name)(shape, dtype=torch.bool))
            self.same(fr.full(shape, True), torch.full(shape, True))
        self.same(fr.eye(3, 4, dtype='bool'), torch.eye(3, 4, dtype=torch.bool))
        for value in (True, False):
            scalar = fr.tensor(value)
            self.assertIs(type(scalar.item()), bool)
            self.assertIs(type(scalar.tolist()), bool)
            self.assertEqual(bool(scalar), value)
        data = [0., -0., 1e-300, float('nan'), float('inf')]
        self.same(fr.tensor(data, dtype='float64').bool(), torch.tensor(data, dtype=torch.float64).bool())
        x = fr.tensor([0., 2.], requires_grad=True)
        self.assertFalse(x.bool().requires_grad)
        self.assertIsNone(x.grad)

    def test_exact_comparisons_broadcast_and_nondifferentiability(self):
        for dtype, data, rhs in [('int64', [-(1 << 63), (1 << 53) + 1, (1 << 63) - 1], [(1 << 53), 0, (1 << 63) - 1]),
                                 ('float64', [1e-300, float('nan'), -0.], [0., 1., 0.]),
                                 ('bool', [True, False, True], [False, False, True])]:
            actual = fr.tensor(data, dtype=dtype)
            other = fr.tensor([rhs], dtype=dtype)
            expected = torch.tensor(data, dtype=getattr(torch, dtype))
            reference = torch.tensor([rhs], dtype=getattr(torch, dtype))
            for name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                self.same(getattr(actual, name)(other), getattr(expected, name)(reference))
                self.same(getattr(operator, name)(actual, other), getattr(operator, name)(expected, reference))
        x = fr.tensor([-1., 0., 2.], requires_grad=True)
        self.same(x > 0., torch.tensor([-1., 0., 2.], requires_grad=True) > 0.)
        self.assertIsNone(x.grad)
        huge = (1 << 53) + 1
        self.same(fr.tensor([huge, huge + 1]) == huge, torch.tensor([huge, huge + 1]) == huge)
        data = [float('nan'), float('inf'), -float('inf'), 0.]
        for name in ('isnan', 'isfinite'):
            self.same(getattr(fr.tensor(data), name)(), getattr(torch.tensor(data), name)())

    def test_logical_operations_reductions_and_nonzero(self):
        data = [[True, False, False], [False, True, True]]
        other = [[True, False, True]]
        x, y = fr.tensor(data), fr.tensor(other)
        tx, ty = torch.tensor(data), torch.tensor(other)
        for name in ('logical_and', 'logical_or', 'logical_xor'):
            self.same(getattr(x, name)(y), getattr(tx, name)(ty))
        for op in (operator.and_, operator.or_, operator.xor):
            self.same(op(x, y), op(tx, ty))
        self.same(~x, ~tx)
        self.same(x.logical_not(), tx.logical_not())
        for shape in [(2, 3), (0,), (2, 0, 3)]:
            x = fr.tensor(data) if shape == (2, 3) else fr.zeros(shape, dtype='bool')
            tx = torch.tensor(data) if shape == (2, 3) else torch.zeros(shape, dtype=torch.bool)
            for name in ('any', 'all'):
                self.same(getattr(x, name)(), getattr(tx, name)())
                for dim in range(len(shape)):
                    for keep in (False, True):
                        self.same(getattr(x, name)(dim, keep), getattr(tx, name)(dim, keep))
            self.same(x.nonzero(), tx.nonzero())
            for a, b in zip(x.nonzero(as_tuple=True), tx.nonzero(as_tuple=True)):
                self.same(a, b)
        for value in (True, False):
            x, tx = fr.tensor(value), torch.tensor(value)
            self.same(x.nonzero(), tx.nonzero())
            self.same(x.nonzero(as_tuple=True)[0], tx.nonzero(as_tuple=True)[0])
            for dim in (0, -1):
                self.same(x.any(dim, True), tx.any(dim, True))
                self.same(x.all(dim, True), tx.all(dim, True))

    def test_typed_views_basic_indexing_and_scalar_index_gradients(self):
        for dtype, values in [('bool', [[True, False, True], [False, True, False]]),
                              ('int64', [[(1 << 53) + 1, 2, 3], [4, 5, (1 << 63) - 1]]),
                              ('float64', [[1e-300, 1., 2.], [3., 4., 1e300]])]:
            x, tx = fr.tensor(values, dtype=dtype), torch.tensor(values, dtype=getattr(torch, dtype))
            for key in (0, (0, 1), (slice(None), 1), slice(0, 0), (1, slice(0, 0)), Ellipsis):
                self.same(x[key], tx[key])
            self.same(x.transpose(0, 1).reshape([6]), tx.T.reshape(6))
            self.same(fr.cat([x, x], 0), torch.cat([tx, tx], 0))
            index, tindex = fr.tensor([[2, 0], [1, 2]]), torch.tensor([[2, 0], [1, 2]])
            self.same(x.gather(1, index), tx.gather(1, tindex))
            self.same(x[fr.tensor([-1, 0, 0])], tx[torch.tensor([-1, 0, 0])])
            self.same(x[fr.tensor(1)], tx[torch.tensor(1)])
            self.assertEqual(x[0, -2::-1].tolist(), list(reversed(values[0][:2])))
        x = fr.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        x[1, 0].backward()
        self.assertEqual(x.grad.tolist(), [[0., 0.], [1., 0.]])
        scalar = fr.tensor(True)
        self.same(scalar[()], torch.tensor(True)[()])
        with self.assertRaises(ValueError):
            scalar[0]
        cube = fr.tensor([[[True, False], [False, True]]])
        self.same(cube[0, ..., 1], torch.tensor(cube.tolist())[0, ..., 1])

    def test_boolean_mask_indexing_and_masked_select_gradients(self):
        data = [[1., 2., 3.], [4., 5., 6.]]
        for mask_data in ([True, False], [[True, False, True], [False, True, False]], True, False):
            x, tx = fr.tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
            mask, tmask = fr.tensor(mask_data), torch.tensor(mask_data)
            selected, reference = x[mask], tx[tmask]
            self.same(selected, reference)
            selected.sum().backward()
            reference.sum().backward()
            self.same(x.grad, tx.grad)
        x, tx = fr.tensor([[1., 2., 3.]], requires_grad=True), torch.tensor([[1., 2., 3.]], requires_grad=True)
        mask_data = [[True, True, False], [True, False, True]]
        mask, tmask = fr.tensor(mask_data), torch.tensor(mask_data)
        selected, reference = x.masked_select(mask), tx.masked_select(tmask)
        self.same(selected, reference)
        selected.sum().backward()
        reference.sum().backward()
        self.same(x.grad, tx.grad)
        self.same(mask[mask], tmask[tmask])
        self.same(mask[(fr.tensor([True, False]),)], tmask[(torch.tensor([True, False]),)])

    def test_where_and_masked_fill_keep_types_and_gradients(self):
        condition = [[True], [False]]
        c, tc = fr.tensor(condition), torch.tensor(condition)
        for dtype, left, right in [('bool', [[True, False]], [[False, True]]),
                                   ('int64', [[(1 << 63) - 1, -(1 << 63)]], [[2, 3]]),
                                   ('float64', [[1e-300, 1e300]], [[2., 3.]])]:
            self.same(fr.where(c, fr.tensor(left, dtype=dtype), fr.tensor(right, dtype=dtype)),
                      torch.where(tc, torch.tensor(left, dtype=getattr(torch, dtype)), torch.tensor(right, dtype=getattr(torch, dtype))))
        x, tx = fr.tensor([[1., 2.]], requires_grad=True), torch.tensor([[1., 2.]], requires_grad=True)
        y, ty = fr.tensor([[3., 4.], [5., 6.]], requires_grad=True), torch.tensor([[3., 4.], [5., 6.]], requires_grad=True)
        fr.where(c, x, y).sum().backward()
        torch.where(tc, tx, ty).sum().backward()
        self.same(x.grad, tx.grad)
        self.same(y.grad, ty.grad)
        mask = fr.tensor([[True, False], [False, True]])
        self.same(mask.masked_fill(c, False), torch.tensor(mask.tolist()).masked_fill(tc, False))
        x, tx = fr.tensor([[1., 2.], [3., 4.]], requires_grad=True), torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        x.masked_fill(mask, 9.).sum().backward()
        tx.masked_fill(torch.tensor(mask.tolist()), 9.).sum().backward()
        self.same(x.grad, tx.grad)

    def test_bool_counts_triangles_and_empty_max(self):
        for shape in [(2, 3), (2, 0, 3), ()]:
            x, tx = fr.ones(shape, dtype='bool'), torch.ones(shape, dtype=torch.bool)
            self.same(x.sum(), tx.sum())
            for dim in range(len(shape)):
                self.same(x.sum_dim(dim, True), tx.sum(dim, keepdim=True))
        x, tx = fr.tensor([[True, False, True], [False, True, True]]), torch.tensor([[True, False, True], [False, True, True]])
        self.same(x.max(), tx.max())
        for diagonal in (-1, 0, 1, 5):
            self.same(x.triu(diagonal), tx.triu(diagonal))
            self.same(x.tril(diagonal), tx.tril(diagonal))
        with self.assertRaises(ValueError):
            fr.tensor([], dtype='bool').max()

    def test_hash_identity_and_ambiguous_truth(self):
        x, y = fr.tensor([True, False]), fr.tensor([True, False])
        self.assertEqual(hash(x), id(x))
        self.assertEqual(hash(y), id(y))
        mapping = {x: 'x', y: 'y'}
        self.assertEqual(mapping[x], 'x')
        self.assertEqual(mapping[y], 'y')
        self.assertEqual(len({x, y}), 2)
        with self.assertRaises(ValueError):
            bool(x == y)
        with self.assertRaises(ValueError):
            bool(fr.tensor([], dtype='bool'))

    def test_unsupported_dtype_device_masks_and_capture_are_explicit(self):
        mask = fr.tensor([True, False])
        with self.assertRaises(ValueError):
            mask.requires_grad_(True)
        for factory in (lambda: mask.cuda(), lambda: fr.zeros(0, dtype='bool', device='cuda'),
                        lambda: fr.ones(1, dtype='bool', requires_grad=True)):
            with self.assertRaises(ValueError):
                factory()
        for operation in (lambda: mask - mask, lambda: -mask, lambda: mask.mean(), lambda: mask.relu(),
                          lambda: mask.softmax(0), lambda: mask.argmax(0), lambda: mask & fr.tensor([1, 0]),
                          lambda: fr.arange(3, dtype='bool'), lambda: fr.randn(2, dtype='bool')):
            with self.assertRaises((TypeError, ValueError)):
                operation()
        with self.assertRaises(ValueError):
            fr.zeros(2, 3)[fr.tensor([True, False, True])]
        with self.assertRaises(ValueError):
            fr.zeros(2).masked_select(fr.ones(2))
        for operation in (lambda: mask.logical_not(), lambda: mask.bool(), lambda: fr.ones(2) > 0.,
                          lambda: fr.ones(2)[mask], lambda: mask.sum()):
            with self.assertRaises(ValueError):
                fr.capture(operation)
        self.same(mask.logical_not(), torch.tensor([False, True]))

    def test_unsupported_numeric_paths_reject_bool_without_cast_or_panic(self):
        mask = fr.tensor([[True, False], [False, True]])
        matrix = mask.reshape([1, 2, 2])
        operations = [lambda: matrix.bmm(matrix), lambda: fr.ones(1, 2, 2).bmm(matrix),
                      lambda: mask.prod_dim(1), lambda: mask.logsumexp(1), lambda: mask.softmin(1),
                      lambda: mask.silu(), lambda: mask.gelu_erf(), lambda: mask.dropout(0., False, 7),
                      lambda: mask.var_dim(1), lambda: mask.std_dim(1)]
        for operation in operations:
            with self.assertRaises((TypeError, ValueError)):
                operation()

    def test_empty_boolean_reducers_reject_unaddressable_outputs(self):
        for shape in [(2 * sys.maxsize + 1, 0), (0, 2 * sys.maxsize + 1)]:
            x = fr.Tensor.from_bool([], list(shape))
            zero_dim = shape.index(0)
            for name in ('all', 'any', 'sum_dim'):
                with self.assertRaisesRegex(ValueError, 'capacity'):
                    getattr(x, name)(zero_dim)
            self.assertEqual(x.all(1 - zero_dim).shape, [0])
        x = fr.Tensor.from_bool([], [sys.maxsize // 8 + 1, 0])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            x.sum_dim(1)


if __name__ == '__main__':
    unittest.main()
