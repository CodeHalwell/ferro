"""Native cumulative arithmetic and live derivative contracts versus Torch."""
import itertools
import math
import subprocess
import sys
import unittest

import torch
import ferro as fr


def native(value, grad=False):
    return fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(grad)


class CumulativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def close(self, actual, expected, *, exact=False):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = (0, 0) if exact or not expected.dtype.is_floating_point else ((2e-6, 2e-5) if expected.dtype == torch.float32 else (2e-12, 2e-11))
        torch.testing.assert_close(value, expected.detach(), atol=tolerance[0], rtol=tolerance[1], equal_nan=True)

    def test_typed_strided_axes_scalar_and_wrapping_values(self):
        for dtype in (torch.bool, torch.int64, torch.float32, torch.float64):
            raw = torch.tensor([[1, 2, -3], [4, 0, 2]], dtype=dtype)
            base = native(raw)
            for x, a in ((raw, base), (raw.T, base.T)):
                for dim in (-1, 0):
                    for name in ('cumsum', 'cumprod'):
                        with self.subTest(dtype=dtype, shape=tuple(x.shape), dim=dim, name=name):
                            expected = getattr(x, name)(dim)
                            self.close(getattr(a, name)(dim), expected)
                            self.close(getattr(fr, name)(a, dim), expected)
            scalar = torch.tensor(2, dtype=dtype)
            for name in ('cumsum', 'cumprod'):
                for dim in (0, -1): self.close(getattr(fr, name)(native(scalar), dim), getattr(torch, name)(scalar, dim))
        x = torch.tensor([2**63-1, 2, 3], dtype=torch.int64)
        self.close(fr.cumprod(native(x), 0), x.cumprod(0), exact=True)
        self.close(fr.cumsum(native(x), 0), x.cumsum(0), exact=True)
        for values in ([1e30, 1e30, 1e-30], [1e-30, 1e-30, 1e30], [16777216., 1., -16777216.]):
            x = torch.tensor(values)
            for name in ('cumsum', 'cumprod'): self.close(getattr(fr, name)(native(x), 0), getattr(x, name)(0), exact=True)

    def test_dtype_options_and_original_input_gradients(self):
        for source in (torch.bool, torch.int64, torch.float32, torch.float64):
            for target in (torch.int64, torch.float32, torch.float64):
                for name in ('cumsum', 'cumprod'):
                    x = torch.tensor([2, 3, 4], dtype=source, requires_grad=source.is_floating_point)
                    a = native(x, x.requires_grad)
                    option = str(target).removeprefix('torch.')
                    if x.requires_grad and target == torch.int64:
                        with self.assertRaises(RuntimeError): getattr(x, name)(0, dtype=target)
                        with self.assertRaises((ValueError, NotImplementedError)): getattr(a, name)(0, dtype=option)
                        with torch.no_grad(), fr.no_grad(): self.close(getattr(a, name)(0, dtype=option), getattr(x, name)(0, dtype=target))
                    else:
                        y, expected = getattr(a, name)(0, dtype=option), getattr(x, name)(0, dtype=target)
                        self.close(y, expected)
                        if x.requires_grad:
                            y.sum().backward(); expected.sum().backward()
                            self.close(a.grad, x.grad)

    def test_nonfinite_zero_cotangent_ordinary_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for values in itertools.product((0., 2., math.inf, math.nan), repeat=3):
                for weights in ((1., 2., 3.), (math.inf, 1., 1.), (1., math.nan, 1.)):
                    x = torch.tensor(values, dtype=dtype, requires_grad=True); a = native(x, True)
                    seed = torch.tensor(weights, dtype=dtype); v = native(seed)
                    expected = x.cumprod(0); actual = a.cumprod(0)
                    with self.subTest(dtype=dtype, values=values, seed=weights):
                        self.close(actual, expected)
                        self.close(fr.autograd.grad(actual, a, v)[0], torch.autograd.grad(expected, x, seed)[0])
                        self.assertIsNone(a.grad)
            x = torch.tensor([0., 2., 0., 3.], dtype=dtype, requires_grad=True); a = native(x, True)
            seed = torch.tensor([1., 1., math.inf, 1.], dtype=dtype)
            self.close(fr.autograd.grad(a.cumprod(0), a, native(seed))[0], torch.autograd.grad(x.cumprod(0), x, seed)[0])

    def test_explicit_dtype_preserves_original_adjoint_staging(self):
        for source, target in ((torch.float64, torch.float32), (torch.float32, torch.float64)):
            for values in ([1e-50, 2.], [1e-30, 1e30], [1.00000003, 2.00000003, 0.], [1.00000003, 0., 2.00000003], [0., 1.00000003, 1.00000003], [1e40, 0., 3.], [0., 1e40, 0.], [1., 2., 3.]):
                for name in ('cumsum', 'cumprod'):
                    with self.subTest(source=source, target=target, values=values, name=name):
                        x = torch.tensor(values, dtype=source, requires_grad=True); a = native(x, True)
                        expected = getattr(x, name)(0, dtype=target)
                        actual = getattr(a, name)(0, dtype=str(target).removeprefix('torch.'))
                        seed = torch.tensor([1e8, 1., -1e8][:len(values)], dtype=target)
                        self.close(actual, expected)
                        self.close(fr.autograd.grad(actual, a, native(seed))[0], torch.autograd.grad(expected, x, seed)[0])

    def test_strided_batched_all_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for zeros in (False, True):
                for dim in (0, 1):
                    raw = torch.tensor([[.5, -1.2, .8], [1.4, -.3, .9]], dtype=dtype)
                    if zeros: raw[0, 1] = 0
                    x = raw.requires_grad_(); a = native(x, True)
                    for name in ('cumsum', 'cumprod'):
                        expected = getattr(x.T, name)(dim); actual = getattr(a.T, name)(dim)
                        seed = torch.linspace(-.7, .9, expected.numel(), dtype=dtype).reshape(expected.shape)
                        self.close(actual, expected)
                        self.close(fr.autograd.grad(actual, a, native(seed))[0], torch.autograd.grad(expected, x, seed, retain_graph=True)[0])

    def test_live_second_third_and_cotangent_derivatives_at_zeros(self):
        for values in ([2., 3., 4.], [2., 0., 4.], [0., 3., 0.], [0., 0., 0.]):
            x = torch.tensor(values, requires_grad=True); a = native(x, True)
            seed = torch.tensor([1., 2., 3.], requires_grad=True); v = native(seed, True)
            first = fr.autograd.grad(a.cumprod(0), a, v, create_graph=True)[0]
            expected = torch.autograd.grad(x.cumprod(0), x, seed, create_graph=True)[0]
            self.close(first, expected)
            second = fr.autograd.grad(first.sum(), (a, v), create_graph=True)
            reference = torch.autograd.grad(expected.sum(), (x, seed), create_graph=True)
            for actual, wanted in zip(second, reference): self.close(actual, wanted)
            self.close(fr.autograd.grad(second[0].sum(), a)[0], torch.autograd.grad(reference[0].sum(), x)[0])
            self.assertIsNone(a.grad); self.assertIsNone(v.grad)
        x = fr.zeros(2048, requires_grad=True)
        with self.assertRaisesRegex((ValueError, NotImplementedError), 'graph-work limit'):
            fr.autograd.grad(x.cumprod(0).sum(), x, create_graph=True)
        self.assertIsNone(x.grad)

    def test_higher_nonfinite_global_zero_branch_preserves_reference_staging(self):
        for values in ([[math.inf, 2., 3.], [2., 0., 4.]], [[math.inf, 0., 0.], [0., math.inf, 0.]], [[2., math.inf, 0.], [math.nan, 0., 3.]], [[1e30, 1e30, 1e-30, 0.], [1e-30, 1e-30, 1e30, 0.]]):
            for graph in (False, True):
                x = torch.tensor(values, requires_grad=True); a = native(x, True)
                seed = torch.arange(1., x.shape[-1] + 1).expand_as(x)
                actual = fr.autograd.grad(a.cumprod(1), a, native(seed), create_graph=graph)[0]
                expected = torch.autograd.grad(x.cumprod(1), x, seed, create_graph=graph)[0]
                self.close(actual, expected)

    def test_jvp_nonfinite_zeros_and_reverse_over_tangent(self):
        for dtype in (torch.float32,):
            for values in ([2., 3., 4.], [2., 0., 4.], [0., 3., 0.], [0., 0., math.inf], [math.inf, 0., 0.], [math.nan, 2., 0.]):
                for weights in ([.1, .2, .3], [math.inf, 1., 1.]):
                    x = torch.tensor(values, dtype=dtype, requires_grad=True); a = native(x, True)
                    seed = torch.tensor(weights, dtype=dtype, requires_grad=True); v = native(seed, True)
                    expected = torch.func.jvp(lambda z:z.cumprod(0), (x,), (seed,))
                    actual = fr.func.jvp(lambda z:z.cumprod(0), (a,), (v,))
                    for got, wanted in zip(actual, expected): self.close(got, wanted)
                    gradients = fr.autograd.grad(actual[1].sum(), (a, v), allow_unused=True)
                    reference = torch.autograd.grad(expected[1].sum(), (x, seed), allow_unused=True)
                    for got, wanted in zip(gradients, reference):
                        if wanted is None: self.assertIsNone(got)
                        else: self.close(got, wanted)
        for shape in ((), (1,), (2, 1)):
            for value in (0., math.inf, math.nan, 2.):
                x = torch.full(shape, value); a = native(x)
                seed = torch.ones_like(x); v = native(seed)
                actual = fr.func.jvp(lambda z:z.cumprod(-1), (a,), (v,))
                expected = torch.func.jvp(lambda z:z.cumprod(-1), (x,), (seed,))
                for got, wanted in zip(actual, expected): self.close(got, wanted)

    def test_mixed_tangent_precision_and_explicit_cumsum_dtype(self):
        for values, weights in (([1e-30, 1e30], [1e-50, 0.]), ([1., 0., 1e30], [0., 1e-50, 0.]), ([1., 2., 3.], [1.+2**-30, -2., 0.])):
            x = torch.tensor(values); a = native(x)
            seed = torch.tensor(weights, dtype=torch.float64, requires_grad=True); v = native(seed, True)
            for name in ('cumsum', 'cumprod'):
                for dtype in (None, torch.float32, torch.float64):
                    option = None if dtype is None else str(dtype).removeprefix('torch.')
                    expected = torch.func.jvp(lambda z:getattr(z, name)(0, dtype=dtype), (x,), (seed,))
                    actual = fr.func.jvp(lambda z:getattr(z, name)(0, dtype=option), (a,), (v,))
                    for got, wanted in zip(actual, expected): self.close(got, wanted)
                    self.close(fr.autograd.grad(actual[1].sum(), v)[0], torch.autograd.grad(expected[1].sum(), seed)[0])

    def test_saved_versions_linear_metadata_and_tangent_guards(self):
        for dtype in (torch.float32, torch.float64):
            for name in ('cumsum', 'cumprod'):
                for changed in ('input', 'output'):
                    x = torch.tensor([2., 3.], dtype=dtype, requires_grad=True); a = native(x, True)
                    expected = getattr(x, name)(0); actual = getattr(a, name)(0)
                    with torch.no_grad(), fr.no_grad():
                        (x if changed == 'input' else expected.detach()).fill_(5.)
                        (a if changed == 'input' else actual.detach()).fill_(5.)
                    if name == 'cumprod':
                        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'): torch.autograd.grad(expected.sum(), x)
                        with self.assertRaises(ValueError): fr.autograd.grad(actual.sum(), a)
                    else: self.close(fr.autograd.grad(actual.sum(), a)[0], torch.autograd.grad(expected.sum(), x)[0])
                    self.assertIsNone(a.grad)
                    if dtype == torch.float64:
                        # The shared forward-AD engine accepts F32 primals;
                        # F64 remains covered above for ordinary derivatives.
                        with self.assertRaises((ValueError, NotImplementedError)):
                            fr.func.jvp(lambda z:getattr(z, name)(0), (a,), (fr.ones(2, dtype='float64'),))
                        continue
                    for size in (1, 3):
                        x = torch.ones(size, dtype=dtype); a = native(x)
                        seed = torch.ones_like(x, requires_grad=True); v = native(seed, True)
                        expected = torch.func.jvp(lambda z:getattr(z, name)(0), (x,), (seed,))
                        actual = fr.func.jvp(lambda z:getattr(z, name)(0), (a,), (v,))
                        with torch.no_grad(), fr.no_grad():
                            (x if changed == 'input' else expected[0]).fill_(5.)
                            (a if changed == 'input' else actual[0]).fill_(5.)
                        if name == 'cumprod':
                            with self.assertRaises(RuntimeError): torch.autograd.grad(expected[1].sum(), seed)
                            with self.assertRaises(ValueError): fr.autograd.grad(actual[1].sum(), v)
                        else: self.close(fr.autograd.grad(actual[1].sum(), v)[0], torch.autograd.grad(expected[1].sum(), seed)[0])

    def test_empty_scalar_seed_identity_and_bounded_huge_process(self):
        for shape in ((), (1,), (3, 1), (0,), (2, 0)):
            for graph in (False, True):
                x = torch.ones(shape, requires_grad=True); a = native(x, True)
                seed = torch.ones_like(x, requires_grad=True); v = native(seed, True)
                for name in ('cumsum', 'cumprod'):
                    actual = fr.autograd.grad(getattr(a, name)(-1), a, v, create_graph=graph)[0]
                    expected = torch.autograd.grad(getattr(x, name)(-1), x, seed, create_graph=graph)[0]
                    self.assertIs(expected, seed); self.assertIs(actual, v)
        result = subprocess.run([sys.executable, __file__, '--huge-worker'], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('HUGE_CUMULATIVE_OK', result.stdout)

    def test_vander_cumulative_powers_gradients_and_integer_overflow(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((), (3,), (2, 3)):
                x = torch.linspace(-.5, 1., math.prod(shape), dtype=dtype).reshape(shape).requires_grad_(); a = native(x, True)
                for count in (2, 3, 6):
                    expected = torch.linalg.vander(x, N=count); actual = fr.linalg.vander(a, N=count)
                    self.close(actual, expected)
                    self.close(fr.autograd.grad(actual.sum(), a)[0], torch.autograd.grad(expected.sum(), x)[0])
        x = torch.tensor([2**62+1, -3, 0], dtype=torch.int64)
        self.close(fr.linalg.vander(native(x), N=5), torch.linalg.vander(x, N=5), exact=True)
        for value in (0., math.inf, math.nan, 1e20):
            for count in (2, 3, 5):
                x = torch.tensor([value], requires_grad=True); a = native(x, True)
                expected = torch.linalg.vander(x, N=count); actual = fr.linalg.vander(a, N=count)
                self.close(actual, expected)
                self.close(fr.autograd.grad(actual.sum(), a)[0], torch.autograd.grad(expected.sum(), x)[0])

    def test_validation_capacity_and_capture_leave_input_unchanged(self):
        a = fr.tensor([2., 3.], requires_grad=True)
        for name in ('cumsum', 'cumprod'):
            for dim in (True, 1.0, None, 1, -2):
                with self.assertRaises((ValueError, TypeError)): getattr(a, name)(dim)
            for dtype in ('bool', 'complex64', 'invalid'):
                with self.assertRaises((ValueError, TypeError)): getattr(a, name)(0, dtype=dtype)
            with self.assertRaises(NotImplementedError): getattr(fr, name)(a, 0, out=fr.zeros(2))
            huge = fr.tensor(1.).expand([1 << 60])
            with self.assertRaises(ValueError): getattr(huge, name)(0)
        with self.assertRaises((ValueError, NotImplementedError)): fr.capture(lambda:a.cumprod(0))
        with self.assertRaises((ValueError, NotImplementedError)): fr.capture(lambda:a.cumsum(0, dtype='float64'))
        for count in (0, 1, True, 2., -2):
            with self.assertRaises((ValueError, TypeError)): fr.linalg.vander(a, N=count)
        with self.assertRaises(ValueError): fr.linalg.vander(a, N=sys.maxsize)
        with self.assertRaises(ValueError): fr.linalg.vander(fr.tensor([True]))
        self.assertEqual(a.tolist(), [2., 3.]); self.assertIsNone(a.grad)


def huge_worker():
    for dtype in ('float32', 'float64'):
        for shape in ((0, 1 << 60), (1 << 60, 0)):
            x = fr.zeros(shape, dtype=dtype, requires_grad=True)
            for dim in (0, 1):
                for name in ('cumsum', 'cumprod'):
                    output = getattr(x, name)(dim)
                    gradient = fr.autograd.grad(output.sum(), x)[0]
                    if gradient.shape != list(shape): raise RuntimeError('wrong huge empty gradient shape')
        empty = fr.zeros(0, dtype=dtype, requires_grad=True)
        output = fr.linalg.vander(empty, N=1 << 60)
        if output.shape != [0, 1 << 60]: raise RuntimeError('wrong huge empty Vander shape')
        if fr.autograd.grad(output.sum(), empty)[0].shape != [0]: raise RuntimeError('wrong empty Vander gradient')
    print('HUGE_CUMULATIVE_OK')


if __name__ == '__main__':
    if '--huge-worker' in sys.argv: huge_worker()
    else: unittest.main(verbosity=2)
