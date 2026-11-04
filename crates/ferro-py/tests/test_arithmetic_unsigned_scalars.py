"""Unsigned Python scalars wrap for integer arithmetic, retain magnitude for division."""
import operator
import unittest

import torch
import ferro as fr


def same(test, value, expected):
    test.assertEqual(value.shape, list(expected.shape))
    test.assertEqual(value.dtype, str(expected.dtype).removeprefix('torch.'))
    actual = torch.tensor(value.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


class UnsignedScalarArithmeticTests(unittest.TestCase):
    def test_integer_scalar_wrapping_methods_functions_dunders_and_comparisons(self):
        for shape in ((), (5,), (0, 5)):
            x = torch.tensor(-2, dtype=torch.int64) if not shape else torch.tensor([0, 1, -2, 2**63-1, -(2**63)]).expand(shape)
            a = fr.tensor(x.tolist(), dtype='int64').reshape(list(shape))
            for number in (2**63, 2**63+1, 2**64-1):
                for name in ('add', 'sub', 'mul'):
                    with self.subTest(shape=shape, number=number, name=name):
                        same(self, getattr(fr, name)(a, number), getattr(torch, name)(x, number))
                        same(self, getattr(a, name)(number), getattr(x, name)(number))
                        same(self, getattr(fr, name)(number, a), getattr(torch, name)(number, x))
                        same(self, getattr(operator, name)(a, number), getattr(operator, name)(x, number))
                        same(self, getattr(operator, name)(number, a), getattr(operator, name)(number, x))
                for name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                    same(self, getattr(operator, name)(a, number), getattr(operator, name)(x, number))

    def test_true_division_preserves_positive_unsigned_magnitude(self):
        x = torch.tensor([[0, 1, -2], [7, -(2**63), 2**63-1]], dtype=torch.int64).T
        a = fr.tensor(x.T.tolist(), dtype='int64').T
        for number in (2**63, 2**63+2**39+1, 2**64-1):
            same(self, a / number, x / number)
            same(self, number / a, number / x)
            same(self, a.true_divide(number), x.true_divide(number))
            same(self, fr.true_divide(a, number), torch.true_divide(x, number))
            same(self, fr.true_divide(number, a), torch.true_divide(number, x))

    def test_unsigned_promotion_failures_and_floating_scalar_context(self):
        for number in (2**63, 2**64-1):
            same(self, fr.true_divide(number, 2**64-1), torch.true_divide(number, 2**64-1))
            self.assertEqual(torch.mul(number, number).dtype, torch.uint64)
            with self.assertRaisesRegex(TypeError, 'unsigned Number/Number'):
                fr.mul(number, number)
            for name in ('add', 'sub', 'mul', 'true_divide'):
                for left, right in ((1, number), (number, 1)):
                    with self.assertRaises((RuntimeError, NotImplementedError)):
                        getattr(torch, name)(left, right)
                    with self.assertRaises((TypeError, ValueError, NotImplementedError)):
                        getattr(fr, name)(left, right)
                for dtype in ('float32', 'float64'):
                    a = fr.tensor([.5, 1.5], dtype=dtype)
                    x = torch.tensor([.5, 1.5], dtype=getattr(torch, dtype))
                    same(self, getattr(fr, name)(a, number), getattr(torch, name)(x, number))
                for left, right in ((1.5, number), (number, 1.5)):
                    same(self, getattr(fr, name)(left, right), getattr(torch, name)(left, right))
                with self.assertRaises((RuntimeError, TypeError)):
                    getattr(torch, name)(torch.tensor([True, False]), number)
                with self.assertRaises((TypeError, ValueError)):
                    getattr(fr, name)(fr.tensor([True, False]), number)

    def test_out_of_scalar_range_rejects_without_mutating_tensor(self):
        a = fr.tensor([0, 2, -3], dtype='int64')
        x = torch.tensor([0, 2, -3], dtype=torch.int64)
        for number in (2**64, -(2**63)-1):
            for name in ('add', 'sub', 'mul', 'true_divide'):
                with self.assertRaises(OverflowError):
                    getattr(torch, name)(x, number)
                with self.assertRaises(OverflowError):
                    getattr(fr, name)(a, number)
        self.assertEqual(a.tolist(), [0, 2, -3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
