import sys
import unittest

import ferro as fr
import torch


class SameShapeReshape(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=0, atol=0)

    def test_typed_identity_reshape_preserves_strides_offsets_and_aliases(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            base = fr.arange(24).to(dtype=dtype).reshape(4, 6)
            x = torch.arange(24).to(dtype=getattr(torch, dtype)).reshape(4, 6)
            pairs = [(base.T, x.T), (base[:, 1::2], x[:, 1::2]),
                     (base[:1].expand(3, 6), x[:1].expand(3, 6)), (base[:0, 3:5], x[:0, 3:5])]
            aliases = []
            for view, expected in pairs:
                reshaped, reference = view.reshape(view.shape), expected.reshape(expected.shape)
                self.same(reshaped, reference)
                self.assertTrue(reshaped._shares_storage(base))
                self.assertIsNot(reshaped, view)
                aliases.append((reshaped, reference))
            base.copy_(fr.zeros(base.shape, dtype=dtype))
            x.copy_(torch.zeros_like(x))
            for alias, reference in aliases:
                self.same(alias, reference)
                self.assertEqual(alias.tolist(), reference.tolist())

    def test_identity_reshape_gradient_flags_versions_and_capture(self):
        for dtype in ('float32', 'float64'):
            base = fr.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, requires_grad=True)
            x = torch.tensor(base.tolist(), dtype=getattr(torch, dtype), requires_grad=True)
            view, expected = base.T, x.T
            result, reference = view.reshape(view.shape), expected.reshape(expected.shape)
            self.same(result, reference)
            result.square().sum().backward(); reference.square().sum().backward()
            self.same(base.grad, x.grad)
            with fr.no_grad(), torch.no_grad():
                frozen, tfrozen = view.reshape(view.shape), expected.reshape(expected.shape)
            self.same(frozen, tfrozen)
            loss, tloss = result.square().sum(), reference.square().sum()
            with fr.no_grad(), torch.no_grad():
                base.copy_(fr.zeros(base.shape, dtype=dtype)); x.zero_()
            with self.assertRaises(ValueError): loss.backward()
            with self.assertRaises(RuntimeError): tloss.backward()
            source = fr.ones((2, 3), dtype=dtype)
            view = source.T
            replay = fr.capture(lambda: view.reshape(view.shape)).compile_fused()
            source.copy_(fr.zeros(source.shape, dtype=dtype))
            self.same(replay.replay(), torch.zeros((2, 3), dtype=x.dtype).T)

    def test_huge_broadcast_identity_reshape_is_metadata_only(self):
        extent = sys.maxsize // 8 + 1
        view = fr.tensor([1.], dtype='float64').expand(extent)
        reference = torch.tensor([1.], dtype=torch.float64).expand(extent)
        result, expected = view.reshape(view.shape), reference.reshape(reference.shape)
        self.assertEqual(result.shape, list(expected.shape))
        self.assertEqual(result.stride(), expected.stride())
        self.assertTrue(result._shares_storage(view))


if __name__ == '__main__':
    unittest.main()
