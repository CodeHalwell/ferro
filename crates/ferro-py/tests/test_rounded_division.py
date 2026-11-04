"""Native rounded division: exact values/layouts, zero AD and floor API errors."""
import math
import unittest
import torch
import ferro as fr


def native(value, grad=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(grad)


def same(test, actual, expected, *, layout=False):
    test.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
    test.assertEqual(actual.shape, list(expected.shape))
    result = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(result, expected.detach(), rtol=0, atol=0, equal_nan=True)
    if expected.dtype.is_floating_point:
        zero = expected.detach() == 0
        test.assertEqual(torch.signbit(result[zero]).tolist(), torch.signbit(expected.detach()[zero]).tolist())
    if layout:
        test.assertEqual(actual.stride(), expected.stride())
        test.assertEqual(actual.storage_offset(), 0)


class RoundedDivisionTests(unittest.TestCase):
    def test_public_aliases_scalars_modes_and_uint64_conversion(self):
        for dtype in (torch.float32, torch.float64, torch.int64):
            x = torch.tensor([-7, -1, 1, 7], dtype=dtype); a = native(x)
            for value in (2, -3, .5, True):
                for mode in (None, 'floor', 'trunc'):
                    expected = torch.div(x, value, rounding_mode=mode)
                    for call in (lambda: fr.div(a, value, rounding_mode=mode), lambda: fr.divide(a, value, rounding_mode=mode), lambda: a.div(value, rounding_mode=mode), lambda: a.divide(value, rounding_mode=mode)):
                        same(self, call(), expected)
                same(self, fr.floor_divide(a, value), torch.floor_divide(x, value))
                same(self, a.floor_divide(value), torch.floor_divide(x, value))
                same(self, a // value, x // value)
                same(self, value // a, value // x)
                same(self, fr.div(value, a, rounding_mode='floor'), torch.div(value, x, rounding_mode='floor'))
            for value in (2**63, 2**64 - 1):
                for mode in (None, 'floor', 'trunc'):
                    same(self, a.div(value, rounding_mode=mode), torch.div(x, value, rounding_mode=mode))
                    reverse_x = torch.tensor([2, 3, 7], dtype=dtype)
                    same(self, fr.div(value, native(reverse_x), rounding_mode=mode), torch.div(value, reverse_x, rounding_mode=mode))
        for left, right in ((1, 2), (True, 2), (2.5, -3), (2**63, 2**64-1)):
            same(self, fr.div(left, right), torch.div(left, right))

    def test_adversarial_floats_match_corrected_floor_and_trunc_exactly(self):
        for dtype in (torch.float32, torch.float64):
            tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)).item()
            values = [-math.inf, -torch.finfo(dtype).max, -3.25, -1., -tiny, -0., 0., tiny, 1., 3.25, torch.finfo(dtype).max, math.inf, math.nan]
            left = torch.tensor(values, dtype=dtype).reshape(-1, 1)
            right = torch.tensor(values, dtype=dtype).reshape(1, -1)
            a, b = native(left), native(right)
            for mode in ('floor', 'trunc'):
                same(self, fr.div(a, b, rounding_mode=mode), torch.div(left, right, rounding_mode=mode))
            same(self, fr.floor_divide(a, b), torch.floor_divide(left, right))
            same(self, fr.div(fr.tensor([1.], dtype=str(dtype).removeprefix('torch.')), .1, rounding_mode='floor'), torch.div(torch.tensor([1.], dtype=dtype), .1, rounding_mode='floor'))

    def test_strided_promoted_broadcast_and_empty_layouts(self):
        for da, db in ((torch.float32, torch.float32), (torch.float64, torch.float64), (torch.float32, torch.float64), (torch.bool, torch.int64), (torch.int64, torch.float32)):
            x = torch.arange(1, 25).reshape(2, 3, 4).to(da); y = torch.arange(1, 25).reshape(2, 3, 4).to(db)
            a, b = native(x), native(y)
            for left, right, actual_left, actual_right in ((x.permute(2, 0, 1), y.permute(2, 0, 1), a.permute(2, 0, 1), b.permute(2, 0, 1)), (x[:, :, ::2].transpose(0, 1), y[:, :, ::2].transpose(0, 1), a[:, :, ::2].transpose(0, 1), b[:, :, ::2].transpose(0, 1)), (x[:1].expand(3, 3, 4), y[0], a[:1].expand(3, 3, 4), b[0]), (x[:, :, :0], y[:1, :1, :1], a[:, :, :0], b[:1, :1, :1])):
                original = (a.tolist(), b.tolist())
                for mode in ('floor', 'trunc'):
                    same(self, fr.div(actual_left, actual_right, rounding_mode=mode), torch.div(left, right, rounding_mode=mode), layout=True)
                self.assertEqual((a.tolist(), b.tolist()), original)
        for da, db in ((torch.float32, torch.float64), (torch.float64, torch.float32)):
            x = torch.tensor([1., 3.], dtype=da); y = torch.tensor(2., dtype=db)
            same(self, fr.div(native(x), native(y), rounding_mode='floor'), torch.div(x, y, rounding_mode='floor'))

    def test_rounded_vjp_is_real_detached_zero_including_nonfinite_seeds(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ('floor', 'trunc'):
                for graph in ((False, True) if dtype == torch.float32 else (False,)):
                    x = torch.tensor([[3., math.nan], [0., math.inf]], dtype=dtype, requires_grad=True)
                    y = torch.tensor([[0., 2.]], dtype=dtype, requires_grad=True)
                    seed = torch.full_like(x, math.nan, requires_grad=True)
                    a, b, v = native(x, True), native(y, True), native(seed, True)
                    output, expected = fr.div(a, b, rounding_mode=mode), torch.div(x, y, rounding_mode=mode)
                    actual_grads = fr.autograd.vjp(output, [a, b], v, create_graph=graph)
                    wanted = torch.autograd.grad(expected, (x, y), seed, create_graph=graph, retain_graph=True)
                    for actual, reference in zip(actual_grads, wanted):
                        same(self, actual, reference)
                        self.assertFalse(actual.requires_grad)
                        self.assertFalse(actual._is_structural_zero())
                    self.assertIsNone(a.grad); self.assertIsNone(b.grad); self.assertIsNone(v.grad)
                    output.sum().backward(); expected.sum().backward()
                    same(self, a.grad, x.grad); same(self, b.grad, y.grad)

    def test_direct_jvp_zero_uses_primal_dtype_layout_and_reverse_queries_stay_unused(self):
        for mode in ('floor', 'trunc'):
            for dtype in (torch.float32, torch.float64):
                x = torch.tensor([[3., math.nan], [0., math.inf]]).T.requires_grad_()
                y = torch.ones_like(x).requires_grad_()
                v = torch.full_like(x, math.nan, dtype=dtype, requires_grad=True)
                a = native(x.T, True).T; b = native(y.T, True).T; direction = native(v.T, True).T
                for enabled in (False, True):
                    with torch.set_grad_enabled(enabled):
                        expected = torch.func.jvp(lambda u, w: torch.div(u, w, rounding_mode=mode), (x, y), (v, v))
                    with fr.enable_grad() if enabled else fr.no_grad():
                        actual = fr.func.jvp(lambda u, w: fr.div(u, w, rounding_mode=mode), (a, b), (direction, direction))
                    for result, reference in zip(actual, expected): same(self, result, reference, layout=True)
                    self.assertFalse(actual[1].requires_grad)
        x = torch.ones(3, 2, requires_grad=True); v = torch.ones(2, 3).T.requires_grad_()
        a, seed = native(x, True), native(v.T, True).T
        _, expected = torch.func.vjp(lambda z: torch.div(z, 2., rounding_mode='floor'), x)
        _, actual = fr.func.vjp(lambda z: fr.div(z, 2., rounding_mode='floor'), a)
        same(self, actual(seed)[0], expected(v)[0], layout=True)
        self.assertFalse(actual(seed)[0].requires_grad)

    def test_saved_input_versions_follow_original_active_edges(self):
        for dtype in (torch.float32, torch.float64):
            for right_live in (False, True):
                for changed in ('left', 'right', 'output'):
                    for mode in ('floor', 'trunc'):
                        x = torch.full((2,), 3., dtype=dtype, requires_grad=True)
                        y = torch.full((2,), 2., dtype=dtype, requires_grad=right_live)
                        a, b = native(x, True), native(y, right_live)
                        output, expected = fr.div(a, b, rounding_mode=mode), torch.div(x, y, rounding_mode=mode)
                        with fr.no_grad(), torch.no_grad():
                            {'left': a, 'right': b, 'output': output.detach()}[changed].fill_(4.)
                            {'left': x, 'right': y, 'output': expected.detach()}[changed].fill_(4.)
                        if changed == 'right' or changed == 'left' and right_live:
                            with self.assertRaises(ValueError): output.sum().grad_wrt([a])
                            with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'): torch.autograd.grad(expected.sum(), x)
                        else:
                            same(self, output.sum().grad_wrt([a])[0], torch.autograd.grad(expected.sum(), x)[0])
                        self.assertIsNone(a.grad); self.assertIsNone(b.grad)

        for later_enable in (False, True):
            x = torch.ones(2, requires_grad=True); base = torch.ones(2, requires_grad=True)
            a, real_base = native(x, True), native(base, True)
            with torch.no_grad(), fr.no_grad():
                view, actual_view = base.reshape(2), real_base.reshape(2)
            expected = torch.div(x, view, rounding_mode='floor')
            output = fr.div(a, actual_view, rounding_mode='floor')
            if later_enable:
                view.requires_grad_(True); actual_view.requires_grad_(True)
            with torch.no_grad(), fr.no_grad():
                x.fill_(4.); a.fill_(4.)
            same(self, output.sum().grad_wrt([a])[0], torch.autograd.grad(expected.sum(), x)[0])
            self.assertIsNone(actual_view.grad); self.assertIsNone(real_base.grad)

    def test_floor_divide_ad_errors_are_distinct_and_leave_slots_unchanged(self):
        x = torch.tensor([3., 4.], requires_grad=True); a = native(x, True)
        for call, reference in ((lambda: a // 2, lambda: x // 2), (lambda: a.floor_divide(2), lambda: x.floor_divide(2)), (lambda: fr.floor_divide(a, 2), lambda: torch.floor_divide(x, 2))):
            output, expected = call(), reference()
            self.assertTrue(output.requires_grad)
            with self.assertRaisesRegex(ValueError, 'not implemented'): output.sum().grad_wrt([a])
            with self.assertRaisesRegex(RuntimeError, 'not implemented'): torch.autograd.grad(expected.sum(), x)
            with self.assertRaisesRegex(ValueError, 'not implemented'): output.sum().backward()
            self.assertIsNone(a.grad)
        with self.assertRaisesRegex((ValueError, NotImplementedError), 'not implemented'):
            fr.func.jvp(lambda z: z // 2, (a,), (fr.ones(2),))
        with self.assertRaises(NotImplementedError):
            torch.func.jvp(lambda z: z // 2, (x,), (torch.ones(2),))
        self.assertIsNone(a.grad)
        same(self, fr.div(a, 2), torch.div(x, 2))

    def test_integer_overflow_zero_divisors_and_huge_empty_inputs_are_safe(self):
        minimum = fr.tensor([-2**63], dtype='int64')
        self.assertEqual(fr.div(minimum, -1, rounding_mode='floor').tolist(), [-2**63])
        self.assertEqual((minimum // -1).tolist(), [-2**63])
        with self.assertRaisesRegex(ValueError, 'overflow'): fr.div(minimum, -1, rounding_mode='trunc')
        for mode in ('floor', 'trunc'):
            with self.assertRaisesRegex(ValueError, 'zero'): fr.div(fr.tensor([1], dtype='int64'), 0, rounding_mode=mode)
            a = fr.zeros(0, 1 << 60); b = fr.tensor(2.).expand(1, 1 << 60)
            self.assertEqual(fr.div(a, b, rounding_mode=mode).shape, [0, 1 << 60])
            self.assertEqual(fr.div(fr.zeros(0, 1 << 60, dtype='int64'), 0, rounding_mode=mode).shape, [0, 1 << 60])

    def test_invalid_modes_out_and_capture_fail_before_mutation(self):
        a = fr.tensor([3., 4.], requires_grad=True); saved_loss = (a * a).sum()
        for mode in ('nearest', '', 1, True, ['floor']):
            with self.assertRaises((TypeError, ValueError)): fr.div(a, 2, rounding_mode=mode)
        with self.assertRaises(NotImplementedError): fr.div(a, 2, out=fr.zeros(2))
        with self.assertRaises((ValueError, NotImplementedError)): fr.capture(lambda: fr.div(a, 2, rounding_mode='floor'))
        with self.assertRaises((ValueError, NotImplementedError)): fr.floor_divide(fr.tensor([True]), fr.tensor([True]))
        self.assertEqual(a.tolist(), [3., 4.]); self.assertIsNone(a.grad)
        self.assertEqual(saved_loss.grad_wrt([a])[0].tolist(), [6., 8.])
        graph = fr.capture(lambda: a * 2).compile_fused()
        same(self, graph.replay(), torch.tensor([6., 8.]))


if __name__ == '__main__': unittest.main(verbosity=2)
