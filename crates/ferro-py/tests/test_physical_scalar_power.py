"""Original scalar power forward AD, including ordinary absent-zero layout."""
import unittest
import torch
import ferro as fr


class PhysicalScalarPowerTests(unittest.TestCase):
    def compare(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual._is_structural_zero(), torch._is_zerotensor(expected))
        if not torch._is_zerotensor(expected):
            values = torch.tensor(actual.reshape([-1]).tolist(), dtype=expected.dtype)
            torch.testing.assert_close(values, expected.detach().reshape(-1), rtol=4e-6, atol=4e-6, equal_nan=True)
            finite_zero = (expected.detach().reshape(-1) == 0) & (values == 0)
            self.assertTrue(torch.equal(torch.signbit(values[finite_zero]), torch.signbit(expected.detach().reshape(-1)[finite_zero])))

    def make(self, lib, kind, dtype='float32', live=False):
        dtype = getattr(torch, dtype) if lib is torch else dtype
        if kind == 'scalar':
            value = lib.ones((), dtype=dtype)
        elif kind == 'vector':
            value = lib.ones(3, dtype=dtype)
        elif kind == 'transpose':
            value = lib.ones(2, 3, dtype=dtype).T
        elif kind == 'offset':
            value = lib.ones(4, 3, dtype=dtype)[1:3].T
        elif kind == 'gap':
            value = lib.ones(2, 6, dtype=dtype)[:, ::2].T
        else:
            value = lib.ones(2, 3, dtype=dtype).T[:0]
        return value.requires_grad_(live)

    def test_zero_power_direct_jvp_layout_dtype_modes_and_ordinary_vjp(self):
        for kind in ('scalar', 'vector', 'transpose', 'offset', 'gap', 'empty'):
            for dtype in ('float32', 'float64'):
                for live in (False, True):
                    for enabled in (False, True):
                        with self.subTest(kind=kind, dtype=dtype, live=live, enabled=enabled):
                            x, tx = self.make(fr, kind, live=live), self.make(torch, kind, live=live)
                            v, tv = self.make(fr, kind, dtype, live), self.make(torch, kind, dtype, live)
                            with fr.enable_grad() if enabled else fr.no_grad():
                                primal, tangent = fr.func.jvp(lambda a: a ** 0., (x,), (v,))
                            with torch.set_grad_enabled(enabled):
                                expected_primal, expected_tangent = torch.func.jvp(lambda a: a ** 0., (tx,), (tv,))
                            self.compare(tangent, expected_tangent)
                            self.assertFalse(tangent._shares_storage(primal))
                            self.assertFalse(tangent._shares_storage(v))
                            if live and enabled:
                                actual = fr.autograd.grad(primal.sum(), x)[0]
                                expected = torch.autograd.grad(expected_primal.sum(), tx)[0]
                                self.compare(actual, expected)

    def test_jacfwd_original_exponent_keeps_live_power_history(self):
        for kind in ('scalar', 'transpose', 'empty'):
            for power in (0., 1., 2., 3., -1., -2., .5, -.5, .5000000000000001, .49999999999999994, 1.0000000000000002, 2.0000000000000004):
                with self.subTest(kind=kind, power=power):
                    x, tx = self.make(fr, kind, live=True), self.make(torch, kind, live=True)
                    actual = fr.func.jacfwd(lambda a: a ** power)(x)
                    expected = torch.func.jacfwd(lambda a: a ** power)(tx)
                    self.compare(actual, expected)
                    if expected.requires_grad:
                        got = fr.autograd.grad(actual.sum(), x, allow_unused=True)[0]
                        want = torch.autograd.grad(expected.sum(), tx, allow_unused=True)[0]
                        self.assertEqual(got is None, want is None)
                        if want is not None:
                            self.compare(got, want)
                    self.assertIsNone(x.grad)

    def test_power_one_tangent_is_computed_and_nonfinite_controls_are_retained(self):
        for power in (0., 1., 2., 2.0000000000000004, .5000000000000001):
            with self.subTest(power=power):
                values = [float('-inf'), -2., -0., 0., 2., float('inf'), float('nan')]
                x, tx = fr.Tensor(values, [7]).requires_grad_(), torch.tensor(values, requires_grad=True)
                v, tv = fr.ones(7).requires_grad_(), torch.ones(7, requires_grad=True)
                _, tangent = fr.func.jvp(lambda a: a ** power, (x,), (v,))
                _, expected = torch.func.jvp(lambda a: a ** power, (tx,), (tv,))
                self.compare(tangent, expected)
                self.assertFalse(tangent._shares_storage(v))
                actual = fr.func.jacfwd(lambda a: a ** power)(x)
                oracle = torch.func.jacfwd(lambda a: a ** power)(tx)
                self.compare(actual, oracle)


if __name__ == '__main__':
    unittest.main()
