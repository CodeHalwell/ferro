"""Typed rounding and rsqrt values, derivative provenance and saved data."""
import unittest
import torch
import ferro as fr


def native(value, grad=False):
    result = fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


class TypedRoundingTests(unittest.TestCase):
    def same(self, actual, expected, exact=True):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=0 if exact else 3e-6, rtol=0 if exact else 3e-6, equal_nan=True)
        zero = expected.detach() == 0
        self.assertEqual(torch.signbit(value[zero]).tolist(), torch.signbit(expected.detach()[zero]).tolist())

    def test_typed_values_ties_nonfinite_and_exact_integer_copies(self):
        for dtype in (torch.float32, torch.float64):
            values = [-float('inf'), -2.5, -1.5, -0.5, -0., 0., .5, 1.5, 2.5, float('inf'), float('nan')]
            if dtype == torch.float64:
                values += [1e300, -1e300, 1e-300, -1e-300, 1.0000000000000002]
            x = torch.tensor(values, dtype=dtype)
            a = native(x)
            for operation in ('floor', 'ceil', 'trunc', 'round', 'frac', 'rsqrt'):
                with self.subTest(dtype=dtype, operation=operation):
                    expected = getattr(torch, operation)(x)
                    self.same(getattr(fr, operation)(a), expected, operation != 'rsqrt')
                    self.same(getattr(a, operation)(), expected, operation != 'rsqrt')
        x = torch.tensor([-(2**63), 2**63-1, 2**54+1], dtype=torch.int64)
        for operation in ('floor', 'ceil', 'trunc', 'round'):
            a = native(x)
            out = getattr(fr, operation)(a)
            self.same(out, getattr(torch, operation)(x))
            self.assertFalse(out._shares_storage(a))
            a.zero_()
            self.same(out, x)
        for dtype in (torch.bool, torch.int64):
            x = torch.tensor([0, 1, 4], dtype=dtype)
            self.same(fr.rsqrt(native(x)), torch.rsqrt(x), False)

    def test_decimal_rounding_keeps_dtype_stages_signed_zero_and_omitted_argument(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([-float('inf'), -123.45, -1.25, -.25, -0., .25, 1.25, 123.45, float('inf'), float('nan')], dtype=dtype)
            for decimals in (-1000, -40, -3, -1, 0, 1, 3, 40, 1000, -(2**63), 2**63-1):
                with self.subTest(dtype=dtype, decimals=decimals):
                    self.same(fr.round(native(x), decimals=decimals), torch.round(x, decimals=decimals))
                    self.same(native(x).round(decimals=decimals), torch.round(x, decimals=decimals))
            self.same(native(x).round(decimals=fr.tensor(1)), x.round(decimals=torch.tensor(1)))
        self.same(fr.round(fr.tensor([2**54+1])), torch.round(torch.tensor([2**54+1])))
        with self.assertRaises((ValueError, TypeError)):
            fr.round(fr.tensor([1]), decimals=0)
        with self.assertRaises((RuntimeError, NotImplementedError)):
            torch.round(torch.tensor([1]), decimals=0)

    def test_strided_empty_scalar_and_virtual_capacity(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.arange(24, dtype=dtype).reshape(4, 6)
            a = native(x)
            cases = [(a.T, x.T), (a[::2, ::2].T, x[::2, ::2].T),
                     (fr.ones(1, 3, dtype=str(dtype).removeprefix('torch.')).expand(2, 3).T, torch.ones(1, 3, dtype=dtype).expand(2, 3).T),
                     (fr.zeros(0, 8, dtype=str(dtype).removeprefix('torch.'))[:, 4:].T, torch.zeros(0, 8, dtype=dtype)[:, 4:].T),
                     (fr.tensor(1.5, dtype=str(dtype).removeprefix('torch.')), torch.tensor(1.5, dtype=dtype))]
            for left, right in cases:
                for operation in ('floor', 'ceil', 'trunc', 'round', 'frac', 'rsqrt'):
                    actual, expected = getattr(left, operation)(), getattr(right, operation)()
                    self.same(actual, expected, operation != 'rsqrt')
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.storage_offset(), 0)
            huge = fr.ones(1, dtype=str(dtype).removeprefix('torch.')).expand(2**62)
            with self.assertRaises(ValueError):
                huge.floor()
            with self.assertRaises(ValueError):
                huge.rsqrt()

    def test_zero_and_identity_reverse_rules_and_grad_slots(self):
        for dtype in (torch.float32, torch.float64):
            for operation in ('floor', 'ceil', 'trunc', 'round', 'frac'):
                x = torch.tensor([.5, 1.5, float('inf'), float('nan')], dtype=dtype, requires_grad=True)
                seed = torch.tensor([1., -0., float('inf'), float('nan')], dtype=dtype, requires_grad=True)
                a, v = native(x, True), native(seed, True)
                y, reference = getattr(a, operation)(), getattr(x, operation)()
                g = fr.autograd.vjp(y, [a], v)[0]
                expected, = torch.autograd.grad(reference, [x], seed)
                self.same(g, expected)
                self.assertFalse(g.requires_grad)
                self.assertIsNone(a.grad)
                self.assertIsNone(v.grad)
        x = fr.ones(2, requires_grad=True)
        seed = fr.ones(2, requires_grad=True)
        _, pullback = fr.func.vjp(lambda value: value.frac(), x)
        self.assertIs(pullback(seed, create_graph=False)[0], seed)
        gradient = fr.autograd.vjp(x.frac(), [x], seed, create_graph=True)[0]
        self.assertEqual(gradient._id(), seed._id())
        self.assertEqual(fr.autograd.grad(gradient.sum(), [seed])[0].tolist(), [1., 1.])

    def test_rsqrt_signed_zero_extremes_first_and_second_derivatives(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([0., -0., .5, 1.5, float('inf'), float('nan')], dtype=dtype, requires_grad=True)
            a = native(x, True)
            self.same(fr.autograd.grad(a.rsqrt().sum(), [a])[0], torch.autograd.grad(x.rsqrt().sum(), [x])[0], False)
            tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype))
            x = torch.tensor([-0., 1e-20, 2.], dtype=dtype, requires_grad=True)
            seed = torch.tensor([tiny, 5e8, 1.], dtype=dtype)
            a = native(x, True)
            self.same(fr.autograd.vjp(a.rsqrt(), [a], native(seed))[0], torch.autograd.grad(x.rsqrt(), [x], seed)[0], False)
        x = torch.tensor([.5, 1.5, 4.], requires_grad=True)
        a = native(x, True)
        expected, = torch.autograd.grad(x.rsqrt().sum(), [x], create_graph=True)
        actual, = fr.autograd.grad(a.rsqrt().sum(), [a], create_graph=True)
        self.same(actual, expected, False)
        self.same(fr.autograd.grad(actual.sum(), [a])[0], torch.autograd.grad(expected.sum(), [x])[0], False)

    def test_saved_output_versions_allow_only_unused_value_mutations(self):
        for dtype in ('float32', 'float64'):
            for operation in ('floor', 'ceil', 'trunc', 'round', 'frac', 'rsqrt'):
                a = fr.tensor([.5, 1.5], dtype=dtype, requires_grad=True)
                output = getattr(a, operation)()
                with fr.no_grad():
                    a.fill_(2.)
                self.assertIsNotNone(fr.autograd.grad(output.sum(), [a])[0])
                output.detach().fill_(3.)
                if operation == 'rsqrt':
                    with self.assertRaises(ValueError):
                        fr.autograd.grad(output.sum(), [a])
                else:
                    self.assertIsNotNone(fr.autograd.grad(output.sum(), [a])[0])
                self.assertIsNone(a.grad)
        a = fr.tensor([.5, 1.5], requires_grad=True)
        x = torch.tensor([.5, 1.5], requires_grad=True)
        out, reference = a.rsqrt(), x.rsqrt()
        with fr.no_grad(), torch.no_grad():
            a.fill_(4.)
            x.fill_(4.)
        gradient = fr.autograd.grad(out.sum(), [a], create_graph=True)[0]
        expected = torch.autograd.grad(reference.sum(), [x], create_graph=True)[0]
        self.same(gradient, expected, False)
        self.same(fr.autograd.grad(gradient.sum(), [a])[0], torch.autograd.grad(expected.sum(), [x])[0], False)
        a = fr.tensor([.5, 1.5], requires_grad=True)
        out = a.frac()
        loss = out * out
        out.detach().zero_()
        with self.assertRaises(ValueError):
            fr.autograd.grad(loss.sum(), [a])

    def test_direct_jvp_float64_directions_and_no_grad_controls(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([.5, 1.5], requires_grad=True)
            v = torch.tensor([1.0000000000000002, 2.], dtype=dtype, requires_grad=True)
            for operation in ('floor', 'ceil', 'trunc', 'round', 'frac', 'rsqrt'):
                for enabled in (False, True):
                    with self.subTest(dtype=dtype, operation=operation, enabled=enabled):
                        with torch.set_grad_enabled(enabled), fr.enable_grad() if enabled else fr.no_grad():
                            expected = torch.func.jvp(lambda value: getattr(torch, operation)(value), (x,), (v,))
                            actual = fr.func.jvp(lambda value: getattr(fr, operation)(value), (native(x, True),), (native(v, True),))
                        for result, reference in zip(actual, expected):
                            self.same(result, reference, operation != 'rsqrt')
                            self.assertEqual(result.requires_grad, reference.requires_grad)

    def test_invalid_arguments_capture_and_replay(self):
        x = fr.ones(2)
        for value in (True, None, 1., '1', fr.tensor(1.), fr.tensor([1]), 2**64):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                x.round(decimals=value)
        with self.assertRaises(TypeError):
            x.round(1)
        with self.assertRaises(TypeError):
            fr.floor([1.])
        with self.assertRaises(NotImplementedError):
            fr.floor(x, out=fr.zeros(2))
        with self.assertRaises(ValueError):
            fr.capture(lambda: fr.ones(2, dtype='float64').round())
        value = fr.tensor([.25, 1.75])
        captured = fr.capture(lambda: value.round(decimals=1))
        chain = captured.compile_fused()
        value.copy_(fr.tensor([.35, 1.85]))
        self.same(chain.replay(), torch.tensor([.35, 1.85]).round(decimals=1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
