"""Scalar Clamp bounds, non-finite adjoints, and signed zeros against Torch."""
import itertools
import unittest

import torch
import ferro as fr


BOUNDS = ((-1., 1.), (2., -2.), (-float('inf'), float('inf')),
          (float('inf'), float('inf')), (-float('inf'), -float('inf')),
          (float('nan'), 1.), (-1., float('nan')), (float('nan'), float('nan')),
          (-0., -0.), (-0., 0.), (0., -0.), (0., 0.))
VALUES = [-float('inf'), -2., -1., -0., 0., 1., 2., float('inf'), float('nan')]


class ClampBoundsTests(unittest.TestCase):
    def same(self, actual, expected, layout=True):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        if layout: self.assertEqual(actual.stride(), expected.stride())
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), atol=0, rtol=0, equal_nan=True)
        zeros = expected.detach() == 0
        self.assertEqual(torch.signbit(got[zeros]).tolist(), torch.signbit(expected.detach()[zeros]).tolist())

    def test_nan_infinite_reversed_and_signed_zero_bounds_typed_values(self):
        for dtype, (lower, upper), shape in itertools.product(('float32', 'float64'), BOUNDS, ((9,), (3, 3))):
            with self.subTest(dtype=dtype, lower=lower, upper=upper, shape=shape):
                x = fr.tensor(VALUES, dtype=dtype).reshape(shape).requires_grad_(True)
                a = torch.tensor(VALUES, dtype=getattr(torch, dtype)).reshape(shape).requires_grad_(True)
                self.same(x.clamp(lower, upper), a.clamp(lower, upper))
        # Float64 bounds are passed at their original precision.
        x = fr.tensor([-1e300, 0., 1e300], dtype='float64')
        a = torch.tensor([-1e300, 0., 1e300], dtype=torch.float64)
        self.same(x.clamp(-1e200, 1e200), a.clamp(-1e200, 1e200))

    def test_ordinary_and_live_seed_derivatives_have_conditional_values_and_real_edges(self):
        for dtype, graph, (lower, upper), amount in itertools.product(('float32', 'float64'), (False, True), BOUNDS, (.7, float('inf'), float('nan'), -0.)):
            if dtype == 'float64' and graph: continue
            with self.subTest(dtype=dtype, graph=graph, lower=lower, upper=upper, seed=amount):
                x = fr.tensor(VALUES, dtype=dtype, requires_grad=True)
                a = torch.tensor(VALUES, dtype=getattr(torch, dtype), requires_grad=True)
                g = fr.full((9,), amount, dtype=dtype, requires_grad=True)
                seed = torch.full((9,), amount, dtype=getattr(torch, dtype), requires_grad=True)
                actual, = fr.autograd.grad(x.clamp(lower, upper), x, g, create_graph=graph)
                expected, = torch.autograd.grad(a.clamp(lower, upper), a, seed, create_graph=graph)
                self.same(actual, expected)
                if graph:
                    got = fr.autograd.grad(actual.sum(), (x, g), allow_unused=True)
                    wanted = torch.autograd.grad(expected.sum(), (a, seed), allow_unused=True)
                    for result, reference in zip(got, wanted): self.same(result, reference)
                self.assertIsNone(x.grad); self.assertIsNone(g.grad)

    def test_jvp_and_higher_masks_preserve_transposed_layout_and_wide_seed_precision(self):
        for wide, enabled, (lower, upper), amount in itertools.product((False, True), (False, True), BOUNDS, (float('inf'), float('nan'), -0.)):
            with self.subTest(wide=wide, mode=enabled, lower=lower, upper=upper, seed=amount):
                x = fr.tensor(VALUES).reshape(3, 3).T.detach().requires_grad_(True)
                a = torch.tensor(VALUES).reshape(3, 3).T.detach().requires_grad_(True)
                dtype = 'float64' if wide else 'float32'
                g = fr.full((3, 3), amount, dtype=dtype).T.requires_grad_(True)
                seed = torch.full((3, 3), amount, dtype=getattr(torch, dtype)).T.requires_grad_(True)
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = fr.func.jvp(lambda p: p.clamp(lower, upper), (x,), (g,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected = torch.func.jvp(lambda p: p.clamp(lower, upper), (a,), (seed,))
                for result, reference in zip(actual, expected): self.same(result, reference)
                if enabled:
                    got = fr.autograd.grad(actual[1].sum(), (x, g), allow_unused=True)
                    wanted = torch.autograd.grad(expected[1].sum(), (a, seed), allow_unused=True)
                    for result, reference in zip(got, wanted): self.same(result, reference)
        # The raw and graph-building F32 adjoints both use predicate-first layout.
        for graph in (False, True):
            x = fr.tensor(VALUES).reshape(3, 3).T.detach().requires_grad_(True)
            a = torch.tensor(VALUES).reshape(3, 3).T.detach().requires_grad_(True)
            g = fr.ones((3, 3), requires_grad=True); seed = torch.ones((3, 3), requires_grad=True)
            got, = fr.autograd.grad(x.clamp(-1., 1.), x, g, create_graph=graph)
            wanted, = torch.autograd.grad(a.clamp(-1., 1.), a, seed, create_graph=graph)
            self.same(got, wanted)

    def test_empty_scalar_saved_input_and_output_mutations_and_declared_dtype_limits(self):
        for dtype, shape, (lower, upper) in itertools.product(('float32', 'float64'), ((), (0,), (2, 0, 3)), BOUNDS):
            x = fr.zeros(shape, dtype=dtype, requires_grad=True)
            a = torch.zeros(shape, dtype=getattr(torch, dtype), requires_grad=True)
            y, b = x.clamp(lower, upper), a.clamp(lower, upper)
            self.same(y, b)
            got, = fr.autograd.grad(y, x, fr.ones(shape, dtype=dtype))
            wanted, = torch.autograd.grad(b, a, torch.ones(shape, dtype=getattr(torch, dtype)))
            self.same(got, wanted)
        for dtype, changed in itertools.product(('float32', 'float64'), ('input', 'output')):
            x = fr.tensor([-.3, .2], dtype=dtype, requires_grad=True)
            a = torch.tensor([-.3, .2], dtype=getattr(torch, dtype), requires_grad=True)
            y, b = x.clamp(float('nan'), 1.), a.clamp(float('nan'), 1.)
            with fr.no_grad(): (x if changed == 'input' else y.detach()).fill_(0.)
            with torch.no_grad(): (a if changed == 'input' else b.detach()).fill_(0.)
            if changed == 'input':
                with self.assertRaises((ValueError, RuntimeError)): fr.autograd.grad(y.sum(), x)
                with self.assertRaises(RuntimeError): torch.autograd.grad(b.sum(), a)
            else:
                got, = fr.autograd.grad(y.sum(), x); wanted, = torch.autograd.grad(b.sum(), a)
                self.same(got, wanted)
        for dtype in ('bool', 'int64'):
            # Torch promotes these inputs; this increment explicitly supports
            # floating Clamp only, including its existing typed rejection.
            reference = torch.tensor([1, 0], dtype=getattr(torch, dtype)).clamp(-1., 1.)
            self.assertEqual(reference.dtype, torch.float32)
            self.assertEqual(reference.tolist(), [1., 0.])
            with self.assertRaises((TypeError, ValueError, NotImplementedError)): fr.tensor([1, 0], dtype=dtype).clamp(-1., 1.)


if __name__ == '__main__': unittest.main()
