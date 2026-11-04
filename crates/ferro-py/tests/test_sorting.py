"""Typed native sorting values, stable permutations and real input adjoints."""
import math
import subprocess
import sys
import unittest

import torch
import ferro as fr


class SortingTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=0, rtol=0, equal_nan=True)
        if expected.dtype.is_floating_point:
            self.assertEqual(torch.signbit(value).tolist(), torch.signbit(expected).tolist())

    def test_all_dtypes_axes_stable_ties_strided_values_and_named_results(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            data = [[3, 1, 2, 1], [0, 5, 4, 5]]
            if dtype == 'int64': data = [[value+2**62 for value in row] for row in data]
            actual, expected = fr.tensor(data, dtype=dtype), torch.tensor(data, dtype=getattr(torch, dtype))
            for a, b in ((actual, expected), (actual.T, expected.T)):
                for dim in range(-2, 2):
                    for descending in (False, True):
                        with self.subTest(dtype=dtype, dim=dim, descending=descending, stride=a.stride()):
                            value, reference = fr.sort(a, dim, descending, stable=True), torch.sort(b, dim=dim, descending=descending, stable=True)
                            self.same(value.values, reference.values); self.same(value.indices, reference.indices)
                            self.assertIs(value.values, value[0]); self.assertIs(value.indices, value[1])
                            self.same(a.argsort(dim, descending, stable=True), b.argsort(dim=dim, descending=descending, stable=True))
                            self.same(fr.argsort(a, dim, descending, stable=True), torch.argsort(b, dim=dim, descending=descending, stable=True))
                            self.assertFalse(value.indices.requires_grad)
                            unstable = a.sort(dim, descending)
                            self.same(unstable.values, reference.values)

    def test_nan_signed_zero_and_infinite_values_preserve_stable_encounter_order(self):
        for dtype in ('float32', 'float64'):
            values = [float('nan'), -0., 0., float('nan'), 2., -2., float('inf'), -float('inf')]
            for descending in (False, True):
                actual = fr.sort(fr.tensor(values, dtype=dtype), descending=descending, stable=True)
                expected = torch.sort(torch.tensor(values, dtype=getattr(torch, dtype)), descending=descending, stable=True)
                self.same(actual.values, expected.values); self.same(actual.indices, expected.indices)

    def test_float_input_and_signed_cotangent_gradients_match_torch(self):
        for dtype in ('float32', 'float64'):
            for scalar in (False, True):
                for descending in (False, True):
                    data = 3. if scalar else [[3., 1., 2.], [.5, 4., -1.]]
                    actual = fr.tensor(data, dtype=dtype, requires_grad=True)
                    expected = torch.tensor(data, dtype=getattr(torch, dtype), requires_grad=True)
                    a, b = actual.sort(descending=descending, stable=True).values, expected.sort(descending=descending, stable=True).values
                    seed = torch.full_like(b, -0.) if scalar else torch.tensor([[-0., 2., 4.], [7., -0., 9.]], dtype=b.dtype)
                    a.backward_with(fr.tensor(seed.tolist(), dtype=dtype)); b.backward(seed)
                    self.same(actual.grad, expected.grad)

    def test_mutating_indices_is_guarded_while_input_and_values_are_not_saved(self):
        for dtype in ('float32', 'float64'):
            actual, expected = fr.tensor([3., 1., 2.], dtype=dtype, requires_grad=True), torch.tensor([3., 1., 2.], dtype=getattr(torch, dtype), requires_grad=True)
            a, b = actual.sort(stable=True), expected.sort(stable=True)
            with fr.no_grad(), torch.no_grad():
                actual.fill_(8.); expected.fill_(8.)
                a.values.fill_(9.) if a.values.is_leaf else a.values.detach().fill_(9.)
                b.values.detach().fill_(9.)
            a.values.sum().backward(); b.values.sum().backward(); self.same(actual.grad, expected.grad)
            actual.zero_grad(); expected.grad = None
            a, b = actual.sort(stable=True), expected.sort(stable=True)
            a.indices.zero_(); b.indices.zero_()
            with self.assertRaises(ValueError): a.values.sum().backward()
            with self.assertRaises(RuntimeError): b.values.sum().backward()
            self.assertIsNone(actual.grad)

    def test_scalar_empty_and_huge_zero_output_paths_are_bounded(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            for shape in ((), (0,), (2, 0, 3)):
                actual = fr.zeros(shape, dtype=dtype)
                expected = torch.zeros(shape, dtype=getattr(torch, dtype))
                for dim in range(-max(1, len(shape)), max(1, len(shape))):
                    a, b = actual.sort(dim, stable=True), expected.sort(dim=dim, stable=True)
                    self.same(a.values, b.values); self.same(a.indices, b.indices)
        program = '''import ferro as f
for shape in ([2**60,0],[0,2**60]):
 x=f.Tensor([],shape).requires_grad_(True)
 for dim in (0,1):
  v,i=x.sort(dim,stable=True)
  assert v.shape==shape and i.numel()==0
  v.sum().backward()
  assert x.grad.numel()==0
  x.zero_grad()
x=f.ones(1).expand(2**61)
try:x.sort()
except ValueError:pass
else:raise AssertionError('capacity must reject before materialization')
'''
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_explicit_out_capture_forward_and_higher_order_boundaries(self):
        actual = fr.tensor([3., 1., 2.], requires_grad=True)
        for dim in (-2, 1):
            with self.assertRaises(ValueError): actual.sort(dim)
        with self.assertRaises(NotImplementedError): fr.sort(actual, out=(fr.zeros(3), fr.zeros(3, dtype='int64')))
        with self.assertRaises(TypeError): fr.sort([3., 1., 2.])
        with self.assertRaises(ValueError): fr.capture(lambda: actual.sort().values)
        with self.assertRaises(ValueError): fr.func.jvp(lambda value: value.sort().values, (actual,), (fr.ones(3),))
        with self.assertRaises(ValueError):
            fr.autograd.grad(actual.sort().values.square().sum(), [actual], create_graph=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
