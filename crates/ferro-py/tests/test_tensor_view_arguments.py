"""Sequence/variadic view arguments retain typed values and real alias layouts."""
import unittest

import ferro as fr
import torch


class TensorViewArguments(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), atol=0, rtol=0)

    def test_sequence_and_variadic_forms_all_public_dtypes(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            x = fr.arange(6).to(dtype=dtype).reshape(1, 2, 3)
            reference = torch.arange(6).to(dtype=getattr(torch, dtype)).reshape(1, 2, 3)
            for args in (([2, 0, 1],), ((2, 0, 1),), (2, 0, 1), ([-1, 0, -2],)):
                self.same(x.permute(*args), reference.permute(*args))
            for args in (([4, -1, -1],), ((4, 2, 3),), (4, 2, 3)):
                self.same(x.expand(*args), reference.expand(*args))
            scalar, expected = fr.tensor(1, dtype=dtype), torch.tensor(1, dtype=getattr(torch, dtype))
            self.same(scalar.permute([]), expected.permute([]))
            self.same(scalar.expand(()), expected.expand(()))

    def test_prepend_expansion_preserves_strided_and_empty_storage_aliases(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            x = fr.arange(6).to(dtype=dtype).reshape(2, 3)
            reference = torch.arange(6).to(dtype=getattr(torch, dtype)).reshape(2, 3)
            aliases = []
            for shape in ([4, 3, 2], [1, 3, 2], [2, 1, 3, 2]):
                alias, expected = x.T.expand(shape), reference.T.expand(shape)
                self.same(alias, expected)
                self.assertTrue(alias._shares_storage(x))
                aliases.append((alias, expected))
            x.copy_(fr.ones(x.shape, dtype=dtype)); reference.copy_(torch.ones_like(reference))
            for alias, expected in aliases: self.same(alias, expected)
            empty, expected_empty = fr.zeros((0, 6), dtype=dtype), torch.zeros((0, 6), dtype=getattr(torch, dtype))
            view, expected_view = empty.narrow(1, 3, 2), expected_empty.narrow(1, 3, 2)
            self.same(view.expand([4, 0, 2]), expected_view.expand([4, 0, 2]))

    def test_view_gradients_no_grad_flags_and_replay(self):
        for dtype in ('float32', 'float64'):
            x = fr.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, requires_grad=True)
            reference = torch.tensor(x.tolist(), dtype=getattr(torch, dtype), requires_grad=True)
            actual = x.T.expand([4, 3, 2]).permute([1, 0, 2])
            expected = reference.T.expand([4, 3, 2]).permute([1, 0, 2])
            self.same(actual, expected)
            actual.square().sum().backward(); expected.square().sum().backward()
            self.same(x.grad, reference.grad)
            with fr.no_grad(), torch.no_grad():
                self.same(x.T.expand([1, 3, 2]), reference.T.expand([1, 3, 2]))
                self.same(x.permute([1, 0]), reference.permute([1, 0]))
            source = fr.ones((2, 3), dtype=dtype)
            replay = fr.capture(lambda: source.permute([1, 0]).expand([2, 3, 2])).compile_fused()
            source.copy_(fr.zeros(source.shape, dtype=dtype))
            self.same(replay.replay(), torch.zeros((2, 3), dtype=reference.dtype).T.expand(2, 3, 2))

    def test_invalid_dimensions_are_rejected_before_reading_values(self):
        x, reference = fr.ones(1, 2, 3), torch.ones(1, 2, 3)
        cases = {'permute': (([0, 0, 1],), ([0, 1],), ([0, 1, 3],), ([True, 1, 2],), ([0., 1, 2],), ([0, 1, 2], 0)),
                 'expand': (([-1, 1, 2, 3],), ([2, 3],), ([2, -2, 3],), ([True, 2, 3],), ([2., 2, 3],), ([2, 2, 3], 0))}
        for name, configurations in cases.items():
            for args in configurations:
                with self.subTest(method=name, args=args):
                    with self.assertRaises((TypeError, ValueError, OverflowError)): getattr(x, name)(*args)
                    with self.assertRaises((TypeError, RuntimeError, IndexError)): getattr(reference, name)(*args)
        self.assertEqual(x.tolist(), [[[1., 1., 1.], [1., 1., 1.]]])


if __name__ == '__main__':
    unittest.main()
