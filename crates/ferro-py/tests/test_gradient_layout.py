import unittest
import warnings

import ferro as fr
import torch


class GradientLayout(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected, rtol=0, atol=0)

    def test_dense_gapped_broadcast_and_empty_leaf_slots(self):
        for dtype in ('float32', 'float64'):
            base = fr.arange(1, 25).to(dtype=dtype).reshape(4, 6)
            reference = torch.arange(1, 25).to(dtype=getattr(torch, dtype)).reshape(4, 6)
            cases = [(base, reference), (base.T, reference.T),
                     (base[:, ::2], reference[:, ::2]), (base[:1].expand(3, 6), reference[:1].expand(3, 6)),
                     (base[1:3], reference[1:3]), (base[:0, 3:5], reference[:0, 3:5])]
            for view, expected in cases:
                with self.subTest(dtype=dtype, shape=view.shape, stride=view.stride()):
                    a = view.detach().requires_grad_(True)
                    x = expected.detach().requires_grad_()
                    before = a.tolist()
                    for _ in range(2):
                        a.T.square().sum().backward()
                        x.T.square().sum().backward()
                        self.same(a.grad, x.grad)
                        self.assertFalse(a.grad._shares_storage(a))
                    self.assertEqual(a.tolist(), before)

    def test_assigned_dense_gradient_keeps_layout_and_exact_double_values(self):
        for dtype in ('float32', 'float64'):
            values = [[1., 2., 3.], [4., 5., 6.]]
            a = fr.tensor(values, dtype=dtype, requires_grad=True)
            x = torch.tensor(values, dtype=getattr(torch, dtype), requires_grad=True)
            gradient = fr.arange(6).to(dtype=dtype).reshape(3, 2).T
            reference = torch.arange(6).to(dtype=x.dtype).reshape(3, 2).T
            a._set_leaf_grad(gradient)
            x.grad = reference
            self.assertEqual(a.grad._id(), gradient._id())
            for _ in range(2):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    a.square().sum().backward()
                    x.square().sum().backward()
                self.same(a.grad, x.grad)
        a = fr.tensor([[1e100, 1.0000000000000002], [-2e100, 1e-200]], dtype='float64', requires_grad=True)
        x = torch.tensor(a.tolist(), dtype=torch.float64, requires_grad=True)
        a.T.square().sum().backward(); x.T.square().sum().backward()
        self.same(a.grad, x.grad)


if __name__ == '__main__':
    unittest.main()
