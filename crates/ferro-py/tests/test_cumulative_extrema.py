"""Typed cumulative extrema: selected values, indices and live derivative routes."""
import math
import subprocess
import sys
import unittest

import torch
import ferro as fr


def native(x, grad=False):
    return fr.tensor(x.detach().reshape(-1).tolist(), dtype=str(x.dtype).removeprefix('torch.')).reshape(list(x.shape)).requires_grad_(grad)


class CumulativeExtremaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), rtol=0, atol=0, equal_nan=True)
        if expected.dtype.is_floating_point:
            self.assertTrue(torch.equal(torch.signbit(value[value == 0]), torch.signbit(expected.detach()[expected == 0])))

    def test_typed_values_indices_ties_nan_and_strided_axes(self):
        for dtype in (torch.bool, torch.int64, torch.float32, torch.float64):
            for shape in ((), (6,), (2, 3), (2, 2, 3), (2, 0)):
                raw = torch.arange(math.prod(shape)).reshape(shape).remainder(4).to(dtype)
                if shape == (6,):
                    if dtype.is_floating_point: raw = torch.tensor([0., -0., math.nan, 3., math.nan, -1.], dtype=dtype)
                    elif dtype == torch.int64: raw = torch.tensor([2**62+1, 2**62+2, 2**62+2, -2**63, -2**63, 2**63-1])
                a = native(raw)
                for transposed in (False, True) if raw.ndim > 1 else (False,):
                    x, source = (raw.transpose(0, -1), a.transpose(0, -1)) if transposed else (raw, a)
                    for dim in range(max(x.ndim, 1)):
                        for name in ('cummax', 'cummin'):
                            with self.subTest(dtype=dtype, shape=shape, dim=dim, name=name, transpose=transposed):
                                expected = getattr(x, name)(dim)
                                for actual in (getattr(source, name)(dim), getattr(fr, name)(source, dim)):
                                    self.assertEqual(actual._fields, ('values', 'indices'))
                                    self.close(actual.values, expected.values); self.close(actual.indices, expected.indices)
                                    self.assertFalse(actual.indices.requires_grad)

    def test_all_input_gradients_f64_strides_duplicates_and_nonfinite_seeds(self):
        for dtype in (torch.float32, torch.float64):
            for name in ('cummax', 'cummin'):
                for dim in (0, 1):
                    for weights in ([[1., 3., 5.], [2., 4., 6.]], [[math.inf, 1., math.nan], [1., 2., 3.]]):
                        x = torch.tensor([[2., 1., 2.], [0., math.nan, math.nan]], dtype=dtype, requires_grad=True); a = native(x, True)
                        expected = getattr(x.T, name)(dim); actual = getattr(a.T, name)(dim)
                        seed = torch.tensor(weights, dtype=dtype).T
                        self.close(actual.values, expected.values); self.close(actual.indices, expected.indices)
                        self.close(fr.autograd.grad(actual.values, a, native(seed))[0], torch.autograd.grad(expected.values, x, seed)[0])
                        self.assertIsNone(a.grad)

    def test_higher_live_seed_is_transposed_routing_without_input_anchor(self):
        for name in ('cummax', 'cummin'):
            x = torch.tensor([2., 1., 2., 0.], requires_grad=True); a = native(x, True)
            seed = torch.tensor([1., 3., 5., 7.], requires_grad=True); v = native(seed, True)
            first = fr.autograd.grad(getattr(a, name)(0).values, a, v, create_graph=True)[0]
            expected = torch.autograd.grad(getattr(x, name)(0).values, x, seed, create_graph=True)[0]
            self.close(first, expected)
            query = torch.tensor([11., 13., 17., 19.], requires_grad=True); q = native(query, True)
            second = fr.autograd.grad(first, (a, v), q, create_graph=True, allow_unused=True)
            reference = torch.autograd.grad(expected, (x, seed), query, create_graph=True, allow_unused=True)
            self.assertIsNone(second[0]); self.assertIsNone(reference[0]); self.close(second[1], reference[1])
            self.close(fr.autograd.grad(second[1].sum(), q)[0], torch.autograd.grad(reference[1].sum(), query)[0])

    def test_ordinary_saved_input_indices_and_mutable_values(self):
        for shape in ((), (0,), (1,), (4,)):
            for graph in (False, True):
                for name in ('cummax', 'cummin'):
                    for changed in ('input', 'values', 'indices'):
                        x = torch.ones(shape, requires_grad=True); a = native(x, True)
                        expected = getattr(x, name)(-1); actual = getattr(a, name)(-1)
                        with torch.no_grad(), fr.no_grad():
                            (x if changed == 'input' else getattr(expected, changed).detach()).fill_(0)
                            (a if changed == 'input' else getattr(actual, changed).detach()).fill_(0)
                        if changed != 'values':
                            with self.assertRaises(RuntimeError): torch.autograd.grad(expected.values.sum(), x, create_graph=graph)
                            with self.assertRaises(ValueError): fr.autograd.grad(actual.values.sum(), a, create_graph=graph)
                        else: self.close(fr.autograd.grad(actual.values.sum(), a, create_graph=graph)[0], torch.autograd.grad(expected.values.sum(), x, create_graph=graph)[0])

    def test_jvp_dtype_and_direction_index_version_dependencies(self):
        for name in ('cummax', 'cummin'):
            for changed in ('input', 'values', 'indices', 'direction', 'tangent'):
                for dtype in (torch.float32, torch.float64):
                    x = torch.tensor([2., 1., 2., 0.], requires_grad=True); a = native(x, True)
                    seed = torch.tensor([.1, .2, .3, .4], dtype=dtype, requires_grad=True); v = native(seed, True)
                    tref, native_result = [], []
                    def tf(z):
                        result = getattr(z, name)(0); tref.append(result); return result.values
                    def ff(z):
                        result = getattr(z, name)(0); native_result.append(result); return result.values
                    expected = torch.func.jvp(tf, (x,), (seed,)); actual = fr.func.jvp(ff, (a,), (v,))
                    for got, wanted in zip(actual, expected): self.close(got, wanted)
                    targets_t = {'input':x, 'values':expected[0], 'indices':tref[0].indices, 'direction':seed, 'tangent':expected[1]}
                    targets_f = {'input':a, 'values':actual[0], 'indices':native_result[0].indices, 'direction':v, 'tangent':actual[1]}
                    with torch.no_grad(), fr.no_grad(): targets_t[changed].detach().fill_(0); targets_f[changed].detach().fill_(0)
                    if changed in ('direction', 'indices'):
                        with self.assertRaises(RuntimeError): torch.autograd.grad(expected[1].sum(), seed)
                        with self.assertRaises(ValueError): fr.autograd.grad(actual[1].sum(), v)
                    else:
                        reference = torch.autograd.grad(expected[1].sum(), (x, seed), allow_unused=True)
                        gradients = fr.autograd.grad(actual[1].sum(), (a, v), allow_unused=True)
                        self.assertIsNone(reference[0]); self.assertIsNone(gradients[0]); self.close(gradients[1], reference[1])

    def test_empty_adjoint_returns_original_input_identity_and_graph(self):
        for name in ('cummax', 'cummin'):
            for shape in ((0,), (2, 0)):
                for graph in (False, True):
                    x = torch.empty(shape, requires_grad=True); a = native(x, True)
                    seed = torch.empty(shape, requires_grad=True); v = native(seed, True)
                    expected = torch.autograd.grad(getattr(x, name)(-1).values, x, seed, create_graph=graph)[0]
                    actual = fr.autograd.grad(getattr(a, name)(-1).values, a, v, create_graph=graph)[0]
                    self.assertIs(expected, x); self.assertIs(actual, a)
                    reference = torch.autograd.grad(expected.sum(), (x, seed), allow_unused=True)
                    gradients = fr.autograd.grad(actual.sum(), (a, v), allow_unused=True)
                    self.close(gradients[0], reference[0]); self.assertIsNone(gradients[1]); self.assertIsNone(reference[1])

        for name in ('cummax', 'cummin'):
            for dtype in (torch.float32, torch.float64):
                x = torch.empty((2, 0), dtype=dtype, requires_grad=True); a = native(x, True)
                getattr(x, name)(1).values.sum().backward(); getattr(a, name)(1).values.sum().backward()
                self.close(a.grad, x.grad)
                self.assertFalse(x.grad.requires_grad); self.assertFalse(a.grad.requires_grad)
                self.assertIsNot(x.grad, x); self.assertNotEqual(a.grad._id(), a._id())

    def test_validation_capacity_capture_and_no_grad(self):
        a = fr.tensor([2., 3.], requires_grad=True)
        for name in ('cummax', 'cummin'):
            for dim in (True, 1., None, 1, -2):
                with self.assertRaises((ValueError, TypeError)): getattr(a, name)(dim)
            with self.assertRaises(NotImplementedError): getattr(fr, name)(a, 0, out=(fr.zeros(2), fr.zeros(2, dtype='int64')))
            with self.assertRaises((ValueError, NotImplementedError)): fr.capture(lambda:getattr(a, name)(0).values)
            with self.assertRaises(ValueError): getattr(fr.tensor(1.).expand([1 << 60]), name)(0)
            with fr.no_grad(): self.assertFalse(getattr(a, name)(0).values.requires_grad)
        self.assertEqual(a.tolist(), [2., 3.]); self.assertIsNone(a.grad)
        result = subprocess.run([sys.executable, __file__, '--huge-worker'], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('HUGE_EXTREMA_OK', result.stdout)


def huge_worker():
    for dtype in ('float32', 'float64'):
        for shape in ((0, 1 << 60), (1 << 60, 0)):
            x = fr.zeros(shape, dtype=dtype, requires_grad=True)
            for dim in (0, 1):
                for name in ('cummax', 'cummin'):
                    result = getattr(x, name)(dim)
                    if result.values.shape != list(shape) or result.indices.shape != list(shape): raise RuntimeError('bad huge empty shape')
                    gradient = fr.autograd.grad(result.values.sum(), x)[0]
                    if gradient is not x: raise RuntimeError('empty extrema lost input identity')
    print('HUGE_EXTREMA_OK')


if __name__ == '__main__':
    if '--huge-worker' in sys.argv: huge_worker()
    else: unittest.main(verbosity=2)
