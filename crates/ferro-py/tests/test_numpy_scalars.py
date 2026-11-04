"""NumPy scalar interoperability consults loaded types and computes natively."""
from fractions import Fraction
import unittest
import warnings

import numpy as np
import torch
import ferro as fr


def native(value):
    return fr.tensor(value.tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))


class NumpyScalarTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertIsInstance(actual, fr.Tensor)
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=0, rtol=0, equal_nan=True)
        if expected.dtype.is_floating_point:
            zero = expected.detach() == 0
            self.assertEqual(torch.signbit(value[zero]).tolist(), torch.signbit(expected.detach()[zero]).tolist())

    def test_tensor_arithmetic_comparisons_and_scalar_categories(self):
        scalars = (np.bool_(True), np.bool_(False), np.int32(2), np.int64(-2), np.uint64(2),
                   np.float16(.5), np.float32(.5), np.float64(.5), np.longdouble('.5'))
        for dtype in (torch.bool, torch.int64, torch.float32, torch.float64):
            x = torch.tensor([0, 1, 2], dtype=dtype); value = native(x)
            for scalar in scalars:
                for name in ('add', 'sub', 'mul', 'true_divide', 'eq', 'lt', 'ge'):
                    with self.subTest(dtype=dtype, scalar=type(scalar).__name__, name=name):
                        try:
                            expected = getattr(torch, name)(x, scalar)
                        except (RuntimeError, TypeError):
                            with self.assertRaises((TypeError, ValueError)):
                                getattr(fr, name)(value, scalar)
                        else:
                            self.same(getattr(fr, name)(value, scalar), expected)
                            self.same(getattr(value, name)(scalar), expected)
        value = fr.ones(2)
        with self.assertRaises(ValueError):
            fr.add(value, value, alpha=True)
        self.same(fr.add(value, value, alpha=np.bool_(True)), torch.add(torch.ones(2), torch.ones(2), alpha=np.bool_(True)))

    def test_number_pairs_normalize_before_unsigned_promotion_and_integer_rounding(self):
        for scalar in (np.bool_(True), np.float16(.5), np.float32(.5), np.float64(.5), np.longdouble('.5')):
            for left, right in ((2**63, scalar), (scalar, 2**63)):
                for name in ('add', 'sub', 'mul', 'true_divide'):
                    with self.subTest(scalar=type(scalar).__name__, name=name, reversed=left is scalar):
                        self.same(getattr(fr, name)(left, right), getattr(torch, name)(left, right))
        integer = np.int64(2**62 + 2**38 + 1)
        x = torch.tensor([.5, 1.5]); value = native(x)
        self.same(value * integer, x * integer)
        self.same(integer * value, integer * x)
        self.same(fr.add(value, integer), torch.add(x, integer))

    def test_alpha_numpy_bool_is_floating_and_unsigned_overflow_remains_distinct(self):
        for dtype in (torch.bool, torch.int64, torch.float32, torch.float64):
            x = torch.tensor([1, 2], dtype=dtype); value = native(x)
            for alpha in (np.bool_(False), np.bool_(True), np.int64(2), np.uint64(2), np.float32(.25), np.longdouble('.25')):
                for name in ('add', 'sub'):
                    with self.subTest(dtype=dtype, alpha=type(alpha).__name__, name=name):
                        try:
                            expected = getattr(torch, name)(x, x, alpha=alpha)
                        except (RuntimeError, TypeError):
                            with self.assertRaises((TypeError, ValueError)):
                                getattr(fr, name)(value, value, alpha=alpha)
                        else:
                            self.same(getattr(fr, name)(value, value, alpha=alpha), expected)
            for scalar in (np.uint64(2**63), np.uint64(2**64-1)):
                for name in ('add', 'mul', 'eq', 'true_divide'):
                    with self.assertRaises(TypeError): getattr(torch, name)(x, scalar)
                    with self.assertRaises(TypeError): getattr(fr, name)(value, scalar)
                with self.assertRaises(TypeError): torch.add(x, x, alpha=scalar)
                with self.assertRaises(TypeError): fr.add(value, value, alpha=scalar)

    def test_tolerances_use_double_parser_with_full_unsigned_range(self):
        x = torch.tensor([0., .1, float('inf'), float('nan')]); y = torch.tensor([.01, .2, 3., float('nan')])
        a, b = native(x), native(y)
        for scalar in (np.bool_(True), np.int64(2), np.uint64(2**63), np.uint64(2**64-1),
                       np.float16(.1), np.float32(.1), np.float64(.1), np.longdouble('1e4000')):
            for key in ('rtol', 'atol'):
                kwargs = {key:scalar, 'equal_nan':True}
                with self.subTest(key=key, scalar=type(scalar).__name__):
                    self.same(fr.isclose(a, b, **kwargs), torch.isclose(x, y, **kwargs))
                    self.assertEqual(fr.allclose(a, b, **kwargs), torch.allclose(x, y, **kwargs))
                    self.same(a.isclose(b, **kwargs), torch.isclose(x, y, **kwargs))
        for scalar in (np.float32(-.1), np.float64(float('nan'))):
            with self.assertRaises(RuntimeError): torch.isclose(x, y, rtol=scalar)
            with self.assertRaises(RuntimeError): fr.isclose(a, b, rtol=scalar)

    def test_softplus_options_and_primitive_checkpoint_configuration(self):
        for scalar in (np.bool_(True), np.int64(2), np.uint64(2), np.float16(.5), np.float32(.5), np.float64(.5), np.longdouble('.5')):
            for dtype in (torch.float32, torch.float64):
                x = torch.tensor([-.3, .7], dtype=dtype); value = native(x)
                for key in ('beta', 'threshold'):
                    with self.subTest(key=key, scalar=type(scalar).__name__, dtype=dtype):
                        actual = fr.nn.functional.softplus(value, **{key:scalar})
                        expected = torch.nn.functional.softplus(x, **{key:scalar})
                        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=dtype), expected, atol=2e-15 if dtype==torch.float64 else 3e-6, rtol=2e-15 if dtype==torch.float64 else 3e-6)
        model = fr.nn.Softplus(np.float32(.5), np.int64(20))
        self.assertIs(type(model.beta), float); self.assertIs(type(model.threshold), float)
        snapshot = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(17))
        restored = fr.nn.Softplus(.5, 20.)
        snapshot.restore(restored, optimizers={}, generator=fr.Generator(99))
        self.assertEqual((restored.beta, restored.threshold), (model.beta, model.threshold))
        for scalar in (np.uint64(2**63), np.uint64(2**64-1)):
            with self.assertRaises(TypeError): torch.nn.functional.softplus(torch.ones(2), beta=scalar)
            with self.assertRaises(TypeError): fr.nn.functional.softplus(fr.ones(2), beta=scalar)

    def test_arrays_complex_and_generic_protocols_remain_explicit_boundaries(self):
        class FloatProtocol:
            calls = 0
            def __float__(self):
                type(self).calls += 1
                return .5
        class IndexProtocol:
            calls = 0
            def __index__(self):
                type(self).calls += 1
                return 2
        x = torch.ones(2); value = fr.ones(2)
        for scalar in (np.array(.5), np.array([.5]), 1j, Fraction(1, 2), FloatProtocol(), IndexProtocol()):
            with self.subTest(scalar=type(scalar).__name__):
                with self.assertRaises((TypeError, RuntimeError)): torch.isclose(x, x, rtol=scalar)
                with self.assertRaises((TypeError, ValueError)): fr.isclose(value, value, rtol=scalar)
                with self.assertRaises((TypeError, RuntimeError)): torch.add(x, x, alpha=scalar)
                with self.assertRaises((TypeError, ValueError)): fr.add(value, value, alpha=scalar)
        self.assertEqual(FloatProtocol.calls, 0); self.assertEqual(IndexProtocol.calls, 0)

    def test_numpy_complex_scalars_follow_real_conversion_with_warning(self):
        x = torch.tensor([.2, 1.]); y = torch.tensor([.25, .9]); a, b = native(x), native(y)
        for scalar in (np.complex64(.5+1j), np.complex128(.5+1j), np.clongdouble(.5+1j)):
            calls = [(lambda:fr.isclose(a, b, rtol=scalar), lambda:torch.isclose(x, y, rtol=scalar))]
            if isinstance(scalar, complex):
                self.assertTrue(torch.mul(x, scalar).dtype.is_complex)
                with self.assertRaises(TypeError): fr.mul(a, scalar)
            else:
                calls.extend(((lambda:fr.add(a, b, alpha=scalar), lambda:torch.add(x, y, alpha=scalar)),
                              (lambda:fr.mul(a, scalar), lambda:torch.mul(x, scalar))))
            for actual_call, reference_call in calls:
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter('always')
                    expected = reference_call()
                with warnings.catch_warnings(record=True) as actual_warnings:
                    warnings.simplefilter('always')
                    actual = actual_call()
                self.same(actual, expected)
                self.assertTrue(any(issubclass(value.category, np.exceptions.ComplexWarning) for value in expected_warnings))
                self.assertTrue(any(issubclass(value.category, np.exceptions.ComplexWarning) for value in actual_warnings))
            if not isinstance(scalar, complex):
                for key in ('beta', 'threshold'):
                    with warnings.catch_warnings(record=True) as expected_warnings:
                        warnings.simplefilter('always')
                        expected = torch.nn.functional.softplus(x, **{key:scalar})
                    with warnings.catch_warnings(record=True) as actual_warnings:
                        warnings.simplefilter('always')
                        actual = fr.nn.functional.softplus(a, **{key:scalar})
                    torch.testing.assert_close(torch.tensor(actual.tolist()), expected, atol=3e-6, rtol=3e-6)
                    self.assertTrue(any(issubclass(value.category, np.exceptions.ComplexWarning) for value in expected_warnings))
                    self.assertTrue(any(issubclass(value.category, np.exceptions.ComplexWarning) for value in actual_warnings))


if __name__ == '__main__':
    unittest.main(verbosity=2)
