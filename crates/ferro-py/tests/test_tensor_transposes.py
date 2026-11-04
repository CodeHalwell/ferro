"""Real-valued transpose conveniences preserve typed values, views, and AD."""
import unittest
import warnings

import ferro as fr
import torch


def apply(value, name):
    return value.t() if name == 't' else getattr(value, name)


class TensorTransposes(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=0, atol=0)

    def test_typed_strided_empty_and_scalar_layouts(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            tdtype = getattr(torch, dtype)
            a = fr.arange(24).to(dtype=dtype).reshape(2, 3, 4)
            x = torch.arange(24).to(dtype=tdtype).reshape(2, 3, 4)
            pairs = [(a, x), (a[:, :, ::2], x[:, :, ::2]),
                     (a[1], x[1]), (a[0, 1], x[0, 1]), (a[1, 1, 2], x[1, 1, 2]),
                     (a[:0, :, 2:], x[:0, :, 2:])]
            for av, xv in pairs:
                for name in ('T', 'mT', 't', 'H', 'mH'):
                    with self.subTest(dtype=dtype, shape=av.shape, name=name), warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        invalid = (av.ndim == 1 and name in ('mT', 'H', 'mH')) or (av.ndim > 2 and name in ('t', 'H'))
                        if invalid:
                            with self.assertRaises(ValueError): apply(av, name)
                            with self.assertRaises(RuntimeError): apply(xv, name)
                        else:
                            result, expected = apply(av, name), apply(xv, name)
                            self.same(result, expected)
                            self.assertTrue(result._shares_storage(av))
                            self.assertEqual(result is av, expected is xv)
                        self.same(av, xv)

    def test_views_observe_source_writes_without_changing_source_layout(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            for name in ('T', 'mT', 't', 'H', 'mH'):
                a = fr.ones((2, 3), dtype=dtype)
                x = torch.ones((2, 3), dtype=getattr(torch, dtype))
                view, reference = apply(a, name), apply(x, name)
                a.copy_(fr.zeros((2, 3), dtype=dtype))
                x.copy_(torch.zeros((2, 3), dtype=x.dtype))
                self.same(view, reference)
                self.same(a, x)

    def test_first_order_and_no_grad_view_semantics(self):
        for dtype in ('float32', 'float64'):
            for shape in ((), (3,), (2, 3), (2, 3, 4)):
                for name in ('T', 'mT', 't', 'H', 'mH'):
                    if (len(shape) == 1 and name in ('mT', 'H', 'mH')) or (len(shape) > 2 and name in ('t', 'H')):
                        continue
                    with self.subTest(dtype=dtype, shape=shape, name=name), warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        a = fr.ones(shape, dtype=dtype, requires_grad=True)
                        x = torch.ones(shape, dtype=getattr(torch, dtype), requires_grad=True)
                        out, reference = apply(a, name), apply(x, name)
                        self.same(out, reference)
                        (out.square() * 3).sum().backward()
                        (reference.square() * 3).sum().backward()
                        self.same(a.grad, x.grad)
                        with fr.no_grad(), torch.no_grad():
                            out, reference = apply(a, name), apply(x, name)
                        self.same(out, reference)
                        self.assertEqual(out is a, reference is x)

    def test_replay_preserves_axis_order_after_source_update(self):
        for name, shape in (('T', (2, 3, 4)), ('mT', (2, 3, 4)), ('mH', (2, 3, 4)),
                            ('t', (2, 3)), ('H', (2, 3))):
            a = fr.ones(shape)
            captured = fr.capture(lambda: apply(a, name)).compile_fused()
            values = fr.arange(a.numel(), dtype='float32').reshape(shape)
            a.copy_(values)
            expected = torch.arange(a.numel(), dtype=torch.float32).reshape(shape)
            self.same(captured.replay(), apply(expected, name))


if __name__ == '__main__':
    unittest.main()
