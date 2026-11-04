"""PReLU JVP arithmetic keeps logical axes before the final Where layout."""
import itertools
import math
import unittest
import ferro as fr
import torch


def inputs(lib, transpose, direction_transpose, wide):
    dtype = ('float64' if wide else 'float32') if lib is fr else (torch.float64 if wide else torch.float32)
    x = lib.tensor([-2., 0., 3., 1., -3., 2., 4., -1., -0., -2., 1., 3.]).reshape((2, 2, 3) if transpose else (3, 2, 2))
    if transpose: x = x.transpose(0, 2)
    x = x.detach().requires_grad_(True)
    w = lib.tensor([.2, .7], requires_grad=True)
    dx = lib.tensor([.3, -.7, 0., 1.1] * 3, dtype=dtype).reshape((2, 2, 3) if direction_transpose else (3, 2, 2))
    if direction_transpose: dx = dx.transpose(0, 2)
    dx = dx.detach().requires_grad_(True)
    dw = lib.tensor([.4, -.2], dtype=dtype, requires_grad=True)
    return x, w, dx, dw


def forward(lib, values, mask, enabled):
    x, w, dx, dw = values
    args, directions = ((x,), (dx,)) if mask == 0 else ((w,), (dw,)) if mask == 1 else ((x, w), (dx, dw))
    fn = (lambda a: lib.nn.functional.prelu(a, w)) if mask == 0 else (lambda b: lib.nn.functional.prelu(x, b)) if mask == 1 else lib.nn.functional.prelu
    with lib.enable_grad() if enabled else lib.no_grad():
        return lib.func.jvp(fn, args, directions)


class PReLUForwardLayoutTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual is None, expected is None)
        if actual is None: return
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual._is_structural_zero(), torch._is_zerotensor(expected))
        av = actual.detach().clone().reshape([-1]).tolist()
        bv = expected.detach().clone().reshape(-1).tolist()
        self.assertEqual(len(av), len(bv))
        for a, b in zip(av, bv):
            if math.isnan(b): self.assertTrue(math.isnan(a))
            elif math.isinf(b): self.assertEqual(a, b)
            else: self.assertTrue(math.isclose(a, b, rel_tol=2e-7, abs_tol=2e-7), (a, b))
            if a == b == 0.: self.assertEqual(math.copysign(1., a), math.copysign(1., b))

    def test_48_layout_direction_dtype_and_mode_controls_keep_further_gradient_metadata(self):
        for xt, vt, mask, wide, enabled in itertools.product((False, True), (False, True), range(3), (False, True), (False, True)):
            with self.subTest(input_transposed=xt, direction_transposed=vt, mask=mask, wide=wide, enabled=enabled):
                a = inputs(fr, xt, vt, wide)
                b = inputs(torch, xt, vt, wide)
                out, tangent = forward(fr, a, mask, enabled)
                ref, desired = forward(torch, b, mask, enabled)
                self.same(out, ref)
                self.same(tangent, desired)
                for value in (out, tangent):
                    for original in a: self.assertFalse(value._shares_storage(original))
                for value in (ref, desired):
                    for original in b: self.assertFalse(torch._C._is_alias_of(value, original))
                if enabled:
                    ga = fr.autograd.grad(tangent, a, fr.ones(tangent.shape, dtype=tangent.dtype), allow_unused=True, create_graph=not wide, retain_graph=True)
                    gb = torch.autograd.grad(desired, b, torch.ones(desired.shape, dtype=desired.dtype), allow_unused=True, create_graph=not wide, retain_graph=True)
                    for actual, expected in zip(ga, gb): self.same(actual, expected)
                for value in (*a, *b): self.assertIsNone(value.grad)

    def test_original_transposed_structural_zero_regression_and_nonfinite_branches(self):
        a, b = inputs(fr, True, False, False), inputs(torch, True, False, False)
        a = a[:2] + (fr.full(a[0].shape, .3, requires_grad=True), a[3])
        b = b[:2] + (torch.full(b[0].shape, .3, requires_grad=True), b[3])
        _, tangent = forward(fr, a, 0, True)
        _, desired = forward(torch, b, 0, True)
        ga = fr.autograd.grad(tangent.sum(), a[0])[0]
        gb = torch.autograd.grad(desired.sum(), b[0])[0]
        self.same(ga, gb)
        self.assertTrue(ga._is_structural_zero())
        self.assertEqual(ga.stride(), (4, 2, 1))
        self.assertFalse(ga.requires_grad)
        for mask, wide in itertools.product(range(3), (False, True)):
            av, bv = [], []
            for lib, result in ((fr, av), (torch, bv)):
                dtype = ('float64' if wide else 'float32') if lib is fr else (torch.float64 if wide else torch.float32)
                x = lib.tensor([-math.inf, -2., -0., 0., 2., math.inf, math.nan] * 4).reshape(2, 2, 7).transpose(0, 2).detach().requires_grad_(True)
                w = lib.tensor([math.inf, .3], requires_grad=True)
                dx = lib.full(x.shape, .7, dtype=dtype, requires_grad=True)
                dw = lib.tensor([0., .2], dtype=dtype, requires_grad=True)
                result.extend((x, w, dx, dw))
            _, tangent = forward(fr, av, mask, True)
            _, desired = forward(torch, bv, mask, True)
            self.same(tangent, desired)
            ga = fr.autograd.grad(tangent, av, fr.ones(tangent.shape, dtype=tangent.dtype), allow_unused=True, create_graph=not wide, retain_graph=True)
            gb = torch.autograd.grad(desired, bv, torch.ones(desired.shape, dtype=desired.dtype), allow_unused=True, create_graph=not wide, retain_graph=True)
            for actual, expected in zip(ga, gb): self.same(actual, expected)


if __name__ == '__main__': unittest.main()
