import unittest
import ferro as fr
import torch


class TypedCopy(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(list(actual.shape), list(expected.shape))
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), rtol=0, atol=0, equal_nan=True)

    def test_exact_values_strided_sources_and_storage_aliases(self):
        for dtype, values in [('float64', [[1.0000000000000002, 1e100], [1e-300, -0.0]]),
                              ('int64', [[-(1 << 63), (1 << 63) - 1], [(1 << 53) + 1, 7]]),
                              ('bool', [[True, False], [True, True]])]:
            with self.subTest(dtype=dtype):
                dst = fr.zeros((2, 2), dtype=dtype)
                reference = torch.zeros((2, 2), dtype=getattr(torch, dtype))
                view, tv = dst.transpose(0, 1), reference.transpose(0, 1)
                src = fr.tensor(values, dtype=dtype).transpose(0, 1)
                ts = torch.tensor(values, dtype=getattr(torch, dtype)).transpose(0, 1)
                self.assertIs(dst.copy_(src), dst)
                reference.copy_(ts)
                self.same(dst, reference)
                self.same(view, tv)
                # Source and destination share storage; values are read before writes.
                expected = reference.transpose(0, 1).clone()
                dst.copy_(view)
                reference.copy_(expected)
                self.same(dst, reference)
                self.same(view, tv)
                self.same(fr.zeros((2, 0), dtype=dtype).copy_(fr.zeros((2, 0), dtype=dtype)),
                          torch.zeros((2, 0), dtype=getattr(torch, dtype)))

    def test_no_grad_leaf_copy_preserves_grad_slot_and_invalidates_old_graphs(self):
        x = fr.tensor([2., 3.], dtype='float64', requires_grad=True)
        tx = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
        x.square().sum().backward(); tx.square().sum().backward()
        saved, tsaved = x.square().sum(), tx.square().sum()
        with fr.no_grad(), torch.no_grad():
            view, tv = x.reshape([1, 2]), tx.reshape([1, 2])
            self.assertIs(x.copy_(fr.tensor([5., 7.], dtype='float64')), x)
            tx.copy_(torch.tensor([5., 7.], dtype=torch.float64))
        self.same(x.grad, tx.grad)
        self.same(view, tv)
        with self.assertRaises(ValueError): saved.backward()
        with self.assertRaises(RuntimeError): tsaved.backward()
        x.zero_grad(); tx.grad = None
        x.square().sum().backward(); tx.square().sum().backward()
        self.same(x.grad, tx.grad)

    def test_rejected_copy_is_atomic_and_self_copy_preserves_saved_graph(self):
        x = fr.tensor([[2., 3.], [4., 5.]], dtype='float64', requires_grad=True)
        saved = x.square().sum()
        with self.assertRaises(ValueError): x.copy_(fr.ones((2, 2), dtype='float64'))
        with fr.no_grad():
            for source in [fr.ones((2, 2)), fr.ones(1, dtype='float64')]:
                with self.assertRaises(ValueError): x.copy_(source)
            with self.assertRaises(ValueError): x.transpose(0, 1).copy_(fr.ones((2, 2), dtype='float64'))
            self.assertIs(x.copy_(x), x)
        saved.backward()
        self.assertEqual(x.grad.tolist(), [[4., 6.], [8., 10.]])
        self.assertEqual(x.tolist(), [[2., 3.], [4., 5.]])

    def test_differentiable_source_requires_explicit_no_grad(self):
        source = fr.tensor([2., 3.], dtype='float64', requires_grad=True)
        dst = fr.zeros(2, dtype='float64')
        with self.assertRaisesRegex(ValueError, 'disabled grad recording'): dst.copy_(source)
        self.assertEqual(dst.tolist(), [0., 0.])
        with fr.no_grad(): dst.copy_(source)
        self.assertFalse(dst.requires_grad)
        self.assertEqual(dst.tolist(), [2., 3.])


if __name__ == '__main__':
    unittest.main()
