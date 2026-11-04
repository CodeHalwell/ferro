"""Structural-zero immutability and safe materializing CPU interchange."""
import unittest

import torch
import ferro as fr


class SymbolicZeroInterop(unittest.TestCase):
    def zero_adjoint(self, dtype):
        x = fr.tensor([1., 2.], requires_grad=True)
        _, tangent = fr.func.jvp(lambda value: value * 2., (x,), (fr.ones(2),))
        value = fr.autograd.grad(tangent.sum(), [x])[0]
        if dtype == 'float64':
            value = value * fr.ones(2, dtype=dtype)
        self.assertEqual(value.dtype, dtype)
        self.assertTrue(value._is_structural_zero())
        return value

    def test_aliases_remain_immutable_and_clone_materializes(self):
        for dtype in ('float32', 'float64'):
            value = self.zero_adjoint(dtype)
            self.assertEqual(value.tolist(), [0., 0.])
            for alias in (value, value.detach(), value.reshape([1, 2])):
                self.assertTrue(alias._is_structural_zero())
                with self.assertRaises((ValueError, RuntimeError)):
                    alias.copy_(fr.ones(*alias.shape, dtype=dtype))
            owned = value.clone()
            self.assertFalse(owned._is_structural_zero())
            owned.copy_(fr.full([2], 3., dtype=dtype))
            self.assertEqual(owned.tolist(), [3., 3.])
            self.assertEqual(value.tolist(), [0., 0.])

    def test_cpu_dlpack_export_is_a_safe_writable_materialization(self):
        for dtype in ('float32', 'float64'):
            value = self.zero_adjoint(dtype)
            copied = fr.from_dlpack(value)
            copied.copy_(fr.ones(2, dtype=dtype))
            self.assertEqual(copied.tolist(), [1., 1.])
            self.assertEqual(value.tolist(), [0., 0.])
            foreign = torch.from_dlpack(value)
            foreign.fill_(7.)
            self.assertEqual(foreign.tolist(), [7., 7.])
            self.assertEqual(value.tolist(), [0., 0.])
