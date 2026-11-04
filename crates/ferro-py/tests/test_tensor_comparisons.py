"""Typed equality and closeness values/validation against actual Torch."""
import decimal
import fractions
import itertools
import math
import unittest

import ferro as fr
import torch


def tensor(value):
    return fr.tensor(value.tolist(), dtype=str(value.dtype).removeprefix('torch.'))


def same(test, value, expected):
    test.assertEqual(value.shape, list(expected.shape))
    test.assertEqual(value.dtype, 'bool')
    test.assertEqual(value.tolist(), expected.tolist())
    test.assertFalse(value.requires_grad)


class TensorComparisonTests(unittest.TestCase):
    def test_broadcast_nonfinite_and_tolerance_matrix_448_configurations(self):
        count = 0
        for dtype in (torch.bool, torch.int64, torch.float32, torch.float64):
            if dtype == torch.bool:
                values = [False, True]
            elif dtype == torch.int64:
                values = [-2**63, -2**63 + 1, -2**24 - 1, -1, 0, 1, 2**24, 2**24 + 1, 2**63 - 2, 2**63 - 1]
            else:
                tiny = torch.finfo(dtype).tiny
                values = [-math.inf, -1e30, -1., -tiny, -0., 0., tiny, 1., 1. + torch.finfo(dtype).eps, 1e30, math.inf, math.nan]
            left = torch.tensor(values, dtype=dtype).reshape(-1, 1)
            right = torch.tensor(values, dtype=dtype).reshape(1, -1)
            a, b = tensor(left), tensor(right)
            for rtol, atol, equal_nan in itertools.product(
                    (0., 1e-8, 1e-5, .1, 1., math.inf, 1e40, 2**-150),
                    (0., 1e-8, .5, 1., math.inf, 1e40, 2**-150), (False, True)):
                with self.subTest(dtype=dtype, rtol=rtol, atol=atol, equal_nan=equal_nan):
                    options = dict(rtol=rtol, atol=atol, equal_nan=equal_nan)
                    expected = torch.isclose(left, right, **options)
                    same(self, fr.isclose(a, b, **options), expected)
                    same(self, a.isclose(b, **options), expected)
                    self.assertEqual(fr.allclose(a, b, **options), torch.allclose(left, right, **options))
                    self.assertEqual(a.allclose(b, **options), torch.allclose(left, right, **options))
                    count += 1
        self.assertEqual(count, 448)

    def test_equal_ignores_dtype_but_requires_shape_and_compares_nan_values(self):
        count = 0
        for da, db in itertools.product((torch.bool, torch.int64, torch.float32, torch.float64), repeat=2):
            for shape in ((), (2,), (0, 2)):
                left = torch.tensor(1, dtype=da) if not shape else torch.tensor([0, 1], dtype=da).expand(shape)
                right = torch.tensor(1, dtype=db) if not shape else torch.tensor([0, 1], dtype=db).expand(shape)
                a, b = tensor(left), tensor(right)
                self.assertEqual(fr.equal(a, b), torch.equal(left, right))
                self.assertEqual(a.equal(b), torch.equal(left, right))
                count += 1
        self.assertEqual(count, 48)
        pairs = [
            (torch.tensor([2**24 + 1], dtype=torch.int64), torch.tensor([2**24], dtype=torch.float32)),
            (torch.tensor([2**63 - 1], dtype=torch.int64), torch.tensor([2**63], dtype=torch.float64)),
            (torch.tensor(math.nan), torch.tensor(math.nan)),
            (torch.tensor(math.inf), torch.tensor(math.inf)),
            (torch.tensor(1.), torch.tensor([1.])),
            (torch.empty(0, 2), torch.empty(2, 0)),
        ]
        for left, right in pairs:
            a, b = tensor(left), tensor(right)
            self.assertEqual(fr.equal(a, b), torch.equal(left, right))
            self.assertEqual(a.equal(b), torch.equal(left, right))
        value = fr.tensor([math.nan])
        self.assertFalse(value.equal(value))

    def test_integer_exactness_and_dtype_rounded_tolerance_thresholds(self):
        cases = [(torch.tensor([2**63 - 1, 2**24 + 1, -2**63]),
                  torch.tensor([2**63 - 2, 2**24, -2**63 + 1]))]
        for dtype in (torch.float32, torch.float64):
            eps = torch.finfo(dtype).eps
            cases.append((torch.tensor([1 + eps, 1 - eps, 0., -eps, 1e30, -1e30], dtype=dtype),
                          torch.tensor([1., 1., eps, 0., -1e30, 1e30], dtype=dtype)))
        for left, right in cases:
            a, b = tensor(left), tensor(right)
            for rtol, atol in itertools.product((0., -0., 5e-324, 2**-150, 1 - 2**-24, 1 + 2**-24, math.inf),
                                               (0., 5e-324, 2**-150, 2**-24, 1 + 2**-24, math.inf)):
                with self.subTest(dtype=left.dtype, rtol=rtol, atol=atol):
                    expected = torch.isclose(left, right, rtol=rtol, atol=atol)
                    same(self, fr.isclose(a, b, rtol=rtol, atol=atol), expected)
                    same(self, a.isclose(b, rtol, atol), expected)

    def test_tolerance_argument_errors_and_input_dtype_checks(self):
        left, right = torch.tensor([1., 2.]), torch.tensor([1.1, 2.])
        a, b = tensor(left), tensor(right)
        values = [True, False, 0, 1, .1, -1., math.nan, math.inf, None, '0.1',
                  fractions.Fraction(1, 10), decimal.Decimal('.1'), torch.tensor(.1),
                  torch.tensor([.1]), torch.tensor(0), torch.tensor(.1, requires_grad=True), 10**1000]
        for name, key, value in itertools.product(('isclose', 'allclose'), ('rtol', 'atol', 'equal_nan'), values):
            actual_value = tensor(value).requires_grad_(value.requires_grad) if isinstance(value, torch.Tensor) else value
            with self.subTest(name=name, key=key, value=repr(value)):
                try:
                    expected = getattr(torch, name)(left, right, **{key: value})
                except Exception as error:
                    with self.assertRaises(type(error)): getattr(fr, name)(a, b, **{key: actual_value})
                    with self.assertRaises(type(error)): getattr(a, name)(b, **{key: actual_value})
                else:
                    actual = getattr(fr, name)(a, b, **{key: actual_value})
                    method = getattr(a, name)(b, **{key: actual_value})
                    if isinstance(expected, torch.Tensor):
                        same(self, actual, expected)
                        same(self, method, expected)
                    else:
                        self.assertEqual(actual, expected)
                        self.assertEqual(method, expected)
        for da, db in itertools.product(('bool', 'int64', 'float32', 'float64'), repeat=2):
            if da != db:
                a, b = fr.ones(2, dtype=da), fr.ones(2, dtype=db)
                for name in ('isclose', 'allclose'):
                    with self.assertRaises(RuntimeError): getattr(fr, name)(a, b)
                    with self.assertRaises(RuntimeError): getattr(a, name)(b)

    def test_comparisons_preserve_grad_mode_and_existing_slots(self):
        left = torch.tensor([1., 2.], requires_grad=True)
        right = torch.tensor([1.1, 2.], requires_grad=True)
        a, b = tensor(left).requires_grad_(True), tensor(right).requires_grad_(True)
        a.sum().backward()
        b.sum().backward()
        before = [value.grad.tolist() for value in (a, b)]
        for enabled in (False, True):
            with fr.enable_grad() if enabled else fr.no_grad():
                same(self, fr.isclose(a, b), torch.isclose(left, right))
                same(self, a.isclose(b), torch.isclose(left, right))
                self.assertEqual(fr.is_grad_enabled(), enabled)
                fr.equal(a, b)
                a.equal(b)
                fr.allclose(a, b)
                a.allclose(b)
            self.assertTrue(fr.is_grad_enabled())
            self.assertEqual([value.grad.tolist() for value in (a, b)], before)

    def test_empty_broadcasts_strides_and_shape_validation(self):
        left = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
        a, b = tensor(left).T, tensor(left.T)
        same(self, fr.isclose(a, b), torch.isclose(left.T, left.T))
        same(self, a.isclose(b), torch.isclose(left.T, left.T))
        for dtype in ('bool', 'int64', 'float32', 'float64'):
            a, b = fr.zeros(2**62, 0, dtype=dtype), fr.zeros(0, dtype=dtype)
            self.assertTrue(a.equal(a))
            self.assertFalse(a.equal(b))
            self.assertEqual(a.isclose(b).shape, [2**62, 0])
            self.assertTrue(a.allclose(b))
            expanded = fr.ones(1, 1, dtype=dtype).expand([2**60, 1])
            empty = fr.zeros(1, 0, dtype=dtype)
            # No value materialization is needed for an empty result. Torch's
            # integer path may allocate its unused cast; this path stays bounded.
            self.assertEqual(fr.isclose(expanded, empty, equal_nan=True).shape, [2**60, 0])
            self.assertTrue(expanded.allclose(empty))
        with self.assertRaises(ValueError): fr.isclose(fr.zeros(2), fr.zeros(3))
        with self.assertRaises(TypeError): fr.equal(fr.ones(2), [1., 1.])
        with self.assertRaises(TypeError): fr.ones(2).equal([1., 1.])

    def test_method_positional_keyword_forwarding_and_nan_equality(self):
        left, right = torch.tensor([1., 2.]), torch.tensor([1.1, 2.])
        a, b = tensor(left), tensor(right)
        same(self, a.isclose(b, 0., .2, False), left.isclose(right, 0., .2, False))
        self.assertEqual(a.allclose(b, 0., .2, False), left.allclose(right, 0., .2, False))
        self.assertFalse(a.allclose(b))
        value = fr.tensor([math.nan, math.inf])
        reference = torch.tensor([math.nan, math.inf])
        for equal_nan in (False, True):
            same(self, value.isclose(value, equal_nan=equal_nan), reference.isclose(reference, equal_nan=equal_nan))
            self.assertEqual(value.allclose(value, equal_nan=equal_nan), reference.allclose(reference, equal_nan=equal_nan))
        for name in ('isclose', 'allclose'):
            with self.assertRaises(TypeError): getattr(a, name)(b, unknown=True)
            with self.assertRaises(TypeError): getattr(a, name)(b, 0., rtol=0.)
            with self.assertRaises(TypeError): getattr(a, name)(b, 0., 0., False, 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
