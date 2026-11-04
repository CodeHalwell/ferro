"""Exact scalar fills over typed whole storage and existing mutation guards."""
import math
import unittest

import torch
import ferro as fr


class TypedFillTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected, atol=0, rtol=0, equal_nan=True)
        if expected.dtype.is_floating_point:
            self.assertEqual(torch.signbit(values).tolist(), torch.signbit(expected).tolist())

    def test_real_scalar_values_all_storage_types_and_empty_shapes(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            values = (False, True, 0, 2**62+1, -2**63, 1.75, -1.75)
            if dtype != 'int64': values += (float('inf'), -float('inf'), float('nan'), -0.)
            for shape in ((), (2, 3), (2, 0, 3)):
                for value in values:
                    with self.subTest(dtype=dtype, shape=shape, value=value):
                        actual, expected = fr.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=getattr(torch, dtype))
                        identity = actual._id()
                        self.assertIs(actual.fill_(value), actual); expected.fill_(value)
                        self.same(actual, expected)
                        self.assertEqual(actual._id(), identity)
                        self.assertIs(actual.zero_(), actual); expected.zero_()
                        self.same(actual, expected)

    def test_integer_precision_direct_rounding_and_reference_float_boundary(self):
        for value in (2**62+2**38+1, -(2**62+2**38+1), 2**64-1):
            for dtype in ('float32', 'float64', 'bool'):
                actual = fr.zeros(2, dtype=dtype).fill_(value)
                expected = torch.zeros(2, dtype=getattr(torch, dtype)).fill_(value)
                self.same(actual, expected)
        value = 2**62+1
        actual = fr.zeros(2, dtype='int64').fill_(value)
        self.assertEqual(actual.tolist(), [value, value])
        for value in (math.nextafter(float(2**63), 0.), float(2**63), -float(2**63)):
            self.same(fr.zeros(2, dtype='int64').fill_(value), torch.zeros(2, dtype=torch.int64).fill_(value))

    def test_scalar_tensors_and_explicit_differentiable_source_boundary(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            for source_dtype in ('float32', 'float64', 'int64', 'bool'):
                value = 2**62+1 if source_dtype == 'int64' else 1.75
                source = fr.tensor(value, dtype=source_dtype)
                reference = torch.tensor(value, dtype=getattr(torch, source_dtype))
                self.same(fr.zeros(2, dtype=dtype).fill_(source), torch.zeros(2, dtype=getattr(torch, dtype)).fill_(reference))
            for value in (1e100, float('inf'), -float('inf'), float('nan'), math.nextafter(float(2**63), float('inf'))):
                source = fr.tensor(value, dtype='float64')
                reference = torch.tensor(value, dtype=torch.float64)
                self.same(fr.zeros(2, dtype=dtype).fill_(source), torch.zeros(2, dtype=getattr(torch, dtype)).fill_(reference))
        source = fr.tensor(2., requires_grad=True)
        actual, expected = fr.zeros(2, dtype='float64'), torch.zeros(2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'disabled grad recording'): actual.fill_(source)
        self.assertEqual(actual.tolist(), [0., 0.])
        with fr.no_grad(), torch.no_grad():
            actual.fill_(source); expected.fill_(torch.tensor(2., requires_grad=True))
        self.same(actual, expected)
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(source.grad)
        with self.assertRaises(ValueError): actual.fill_(fr.ones(1))

    def test_overflow_and_invalid_arguments_leave_values_and_saved_versions_unchanged(self):
        cases = {'float32': (1e100, 2**64, -2**63-1),
                 'float64': (2**64, -2**63-1),
                 'int64': (2**63, 2**64, float('nan'), float('inf'), math.nextafter(float(2**63), float('inf'))),
                 'bool': (2**64, -2**63-1)}
        for dtype, values in cases.items():
            for value in values:
                actual, expected = fr.ones(2, dtype=dtype), torch.ones(2, dtype=getattr(torch, dtype))
                with self.assertRaises((ValueError, OverflowError)): actual.fill_(value)
                with self.assertRaises((RuntimeError, OverflowError)): expected.fill_(value)
                self.same(actual, expected)
        for dtype in ('float32', 'float64'):
            actual = fr.ones(2, dtype=dtype, requires_grad=True)
            saved = actual.square().sum()
            with self.assertRaises(TypeError): actual.detach().fill_('3')
            saved.backward()
            self.assertEqual(actual.grad.tolist(), [2., 2.])

    def test_no_grad_leaf_aliases_keep_ids_slots_and_guard_saved_values(self):
        for dtype in ('float32', 'float64'):
            actual = fr.ones(2, dtype=dtype, requires_grad=True)
            expected = torch.ones(2, dtype=getattr(torch, dtype), requires_grad=True)
            actual.sum().backward(); expected.sum().backward()
            identity, gradient = actual._id(), actual.grad._id()
            stale, reference = actual.square().sum(), expected.square().sum()
            with self.assertRaises(ValueError): actual.zero_()
            with fr.no_grad(), torch.no_grad():
                self.assertIs(actual.fill_(3.), actual); expected.fill_(3.)
            self.same(actual, expected)
            self.assertEqual(actual._id(), identity)
            self.assertEqual(actual.grad._id(), gradient)
            with self.assertRaises(ValueError): stale.backward()
            with self.assertRaises(RuntimeError): reference.backward()
            with fr.no_grad(), torch.no_grad(): actual.zero_(); expected.zero_()
            self.same(actual, expected)

    def test_layout_capture_and_structural_zero_rejections_are_prewrite(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            actual = fr.ones(2, 2, dtype=dtype)
            for view in (actual.T, actual.narrow(0, 0, 1)):
                with self.assertRaises(ValueError): view.fill_(3)
                with self.assertRaises(ValueError): view.zero_()
            with self.assertRaises((ValueError, NotImplementedError)):
                fr.capture(lambda: (actual.fill_(3), actual)[1])
            self.assertEqual(actual.tolist(), [[True, True], [True, True]] if dtype == 'bool' else [[1., 1.], [1., 1.]])
        x = fr.ones(2, requires_grad=True)
        _, tangent = fr.func.jvp(lambda value: value*2., (x,), (fr.ones(2),))
        zero = fr.autograd.grad(tangent.sum(), [x])[0]
        for value in (zero, zero*fr.ones(2, dtype='float64')):
            self.assertTrue(value._is_structural_zero())
            with self.assertRaises(ValueError): value.fill_(1.)
            with self.assertRaises(ValueError): value.zero_()
            self.assertEqual(value.tolist(), [0., 0.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
