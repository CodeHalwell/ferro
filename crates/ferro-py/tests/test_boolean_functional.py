"""Public boolean wrappers and exact reductions against PyTorch."""
import unittest

import torch
import ferro as fr


class BooleanFunctional(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=0, atol=0, equal_nan=True)

    def test_boolean_sum_any_all_multi_axis_and_empty_reductions(self):
        for shape in ((2, 3, 2), (2, 0, 3), ()):
            count = torch.empty(shape).numel()
            value = torch.tensor([i % 3 == 0 for i in range(count)]).reshape(shape).bool()
            actual = fr.tensor(value.tolist(), dtype=fr.bool).reshape(list(value.shape))
            dims = [None, (), 0, -1] + ([(0, -1), (2, 1)] if len(shape) == 3 else [])
            for name in ('sum', 'any', 'all'):
                for dim in dims:
                    for keepdim in (False, True):
                        with self.subTest(shape=shape, name=name, dim=dim, keepdim=keepdim):
                            self.same(getattr(fr, name)(actual, dim=dim, keepdim=keepdim),
                                      getattr(torch, name)(value, dim=dim, keepdim=keepdim))

    def test_comparisons_and_predicates_preserve_typed_values(self):
        for dtype, values in [('int64', [(1 << 53) + 1, (1 << 53) + 2, -(1 << 63)]),
                              ('float64', [1e-300, -1e-300, 0.]), ('float32', [-.2, 0., .8])]:
            actual = fr.tensor(values, dtype=dtype)
            expected = torch.tensor(values, dtype=getattr(torch, dtype))
            other = values[0]
            for name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                self.same(getattr(fr, name)(actual, other), getattr(torch, name)(expected, other))
            for name in ('any', 'all', 'isnan', 'isfinite'):
                self.same(getattr(fr, name)(actual), getattr(torch, name)(expected))
        actual = fr.tensor([float('nan'), float('inf'), -float('inf'), -0., 2.])
        expected = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 2.])
        for name in ('isnan', 'isfinite'):
            self.same(getattr(fr, name)(actual), getattr(torch, name)(expected))
        left, right = fr.tensor([[True], [False]]), fr.tensor([[True, False]])
        tl, tr = torch.tensor([[True], [False]]), torch.tensor([[True, False]])
        for name in ('logical_and', 'logical_or', 'logical_xor'):
            self.same(getattr(fr, name)(left, right), getattr(torch, name)(tl, tr))
        self.same(fr.logical_not(left), torch.logical_not(tl))

    def test_typed_where_nonzero_masked_select_and_gradients(self):
        mask = fr.tensor([[True], [False]])
        tm = torch.tensor([[True], [False]])
        for dtype, values in [('int64', [[(1 << 53) + 1, 2], [3, 4]]),
                              ('float64', [[1e-300, 1e200], [1.0000000000000002, -0.]])]:
            actual = fr.tensor(values, dtype=dtype)
            expected = torch.tensor(values, dtype=getattr(torch, dtype))
            self.same(fr.where(mask, actual, 7), torch.where(tm, expected, 7))
            self.same(fr.where(mask, -2, actual), torch.where(tm, -2, expected))
        x = fr.tensor([[.2, -.3], [.7, .9]], requires_grad=True)
        y = fr.tensor([.5, .8], requires_grad=True)
        tx = torch.tensor(x.tolist(), requires_grad=True)
        ty = torch.tensor(y.tolist(), requires_grad=True)
        output, reference = fr.where(mask, x, y), torch.where(tm, tx, ty)
        output.pow(2).sum().backward()
        reference.square().sum().backward()
        self.same(output, reference)
        self.same(x.grad, tx.grad)
        self.same(y.grad, ty.grad)
        for value in (fr.tensor(True), fr.tensor(False), mask, fr.zeros(0, 2, dtype='bool')):
            target = torch.tensor(value.tolist(), dtype=torch.bool).reshape(value.shape)
            self.same(fr.nonzero(value), torch.nonzero(target))
            for left, right in zip(fr.where(value), torch.where(target)):
                self.same(left, right)
        a = fr.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        expected = torch.tensor(a.tolist(), requires_grad=True)
        selected, target = fr.masked_select(a, mask), torch.masked_select(expected, tm)
        selected.sum().backward()
        target.sum().backward()
        self.same(selected, target)
        self.same(a.grad, expected.grad)


if __name__ == '__main__':
    unittest.main()
