"""Actual Torch conditional iteration layouts and non-finite Clamp directions."""
import itertools
import math
import os
import subprocess
import sys
import unittest

import torch
import ferro as fr


def make(library, shape, kind, dtype, role=1):
    shape = list(shape)
    reverse = kind in ('T', 'gapT')
    base_shape = shape[::-1] if reverse else shape[:]
    if kind == 'expand':
        value = library.tensor(role != 0 if dtype == 'bool' else role, dtype=getattr(library, dtype))
        return value.expand(shape)
    if kind in ('gap', 'gapT') and base_shape:
        base_shape[-1] *= 2
    count = math.prod(base_shape)
    values = [(i % 3 == 0) if role == 0 else (i % 7 - 3 if role == 1 else 8 - i % 5) for i in range(count)]
    value = library.tensor(values, dtype=getattr(library, dtype)).reshape(base_shape)
    if kind in ('gap', 'gapT') and base_shape:
        value = value[(slice(None),) * (len(base_shape) - 1) + (slice(None, None, 2),)]
    if reverse:
        value = value.permute(tuple(reversed(range(len(shape)))))
    return value


class WhereLayoutTests(unittest.TestCase):
    def same(self, actual, expected, *, flags=True):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.stride(), expected.stride())
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=3e-6, rtol=3e-5, equal_nan=True)
        if expected.is_floating_point():
            zero = expected.detach() == 0
            self.assertEqual(torch.signbit(value[zero]).tolist(), torch.signbit(expected.detach()[zero]).tolist())
        if flags:
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)

    def test_typed_raw_condition_order_broadcast_empty_and_singleton_layouts(self):
        shapes = [(), (3, 2), (2, 1, 3), (1, 2, 1, 3), (3, 0), (2, 0, 3)]
        kinds = ('C', 'T', 'gap', 'gapT', 'expand')
        for shape, dtype in itertools.product(shapes, ('bool', 'int64', 'float32', 'float64')):
            for layouts in itertools.product(kinds, repeat=3):
                with self.subTest(shape=shape, dtype=dtype, layouts=layouts):
                    actual = [make(fr, shape, k, 'bool' if i == 0 else dtype, i) for i, k in enumerate(layouts)]
                    expected = [make(torch, shape, k, 'bool' if i == 0 else dtype, i) for i, k in enumerate(layouts)]
                    self.same(fr.where(*actual), torch.where(*expected))
        for shapes in (((3, 2), (3, 1), (1, 2)), ((1, 2), (), (3, 2)), ((), (3, 2), (3, 1))):
            actual = [make(fr, shape, 'T', 'bool' if i == 0 else 'float32', i) for i, shape in enumerate(shapes)]
            expected = [make(torch, shape, 'T', 'bool' if i == 0 else 'float32', i) for i, shape in enumerate(shapes)]
            self.same(fr.where(*actual), torch.where(*expected))

    def test_branch_promotion_uses_cast_layout_and_preserves_original_input_gradients(self):
        for left_dtype, right_dtype, left_shape, layout in itertools.product(
                ('bool', 'int64', 'float32', 'float64'), ('bool', 'int64', 'float32', 'float64'), ((), (3, 2)), ('T', 'gapT')):
            with self.subTest(left=left_dtype, right=right_dtype, shape=left_shape, layout=layout):
                c = make(fr, (3, 2), 'T', 'bool', 0)
                tc = make(torch, (3, 2), 'T', 'bool', 0)
                x = make(fr, left_shape, layout, left_dtype).detach().requires_grad_(left_dtype.startswith('float'))
                y = make(fr, (3, 2), 'C', right_dtype, 2).requires_grad_(right_dtype.startswith('float'))
                a = make(torch, left_shape, layout, left_dtype).detach().requires_grad_(left_dtype.startswith('float'))
                b = make(torch, (3, 2), 'C', right_dtype, 2).requires_grad_(right_dtype.startswith('float'))
                output, expected = fr.where(c, x, y), torch.where(tc, a, b)
                self.same(output, expected)
                inputs = [v for v in (x, y) if v.requires_grad]
                refs = [v for v in (a, b) if v.requires_grad]
                if refs:
                    actual_grad = fr.autograd.grad(output, inputs, fr.ones(output.shape, dtype=output.dtype))
                    expected_grad = torch.autograd.grad(expected, refs, torch.ones_like(expected))
                    for got, want in zip(actual_grad, expected_grad): self.same(got, want)

    def test_comparison_and_logical_masks_are_computed_in_operand_order(self):
        comparisons = ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
        logical = ('logical_and', 'logical_or', 'logical_xor')
        for dtype, layout in itertools.product(('bool', 'int64', 'float32', 'float64'), ('T', 'gapT', 'expand')):
            x, a = make(fr, (3, 2), layout, dtype), make(torch, (3, 2), layout, dtype)
            y, b = make(fr, (3, 2), 'C', dtype, 2), make(torch, (3, 2), 'C', dtype, 2)
            for name in comparisons + logical:
                self.same(getattr(fr, name)(x, y), getattr(torch, name)(a, b))
            for name in comparisons:
                self.same(getattr(fr, name)(x, fr.tensor(0, dtype=dtype)), getattr(torch, name)(a, torch.tensor(0, dtype=getattr(torch, dtype))))

    def test_canonical_first_and_live_seed_higher_adjoints_match_torch_layout(self):
        for dtype, graph, condition_layout, seed_layout in itertools.product(('float32', 'float64'), (False, True), ('C', 'T', 'gapT'), ('C', 'T')):
            if dtype == 'float64' and graph: continue
            with self.subTest(dtype=dtype, graph=graph, mask=condition_layout, seed=seed_layout):
                c, tc = (make(lib, (3, 2), condition_layout, 'bool', 0) for lib in (fr, torch))
                x, a = (make(lib, (3, 2), 'T', dtype).detach().requires_grad_(True) for lib in (fr, torch))
                y, b = (make(lib, (3, 2), 'C', dtype, 2).requires_grad_(True) for lib in (fr, torch))
                g, tg = (make(lib, (3, 2), seed_layout, dtype).detach().requires_grad_(True) for lib in (fr, torch))
                output, expected = fr.where(c, x, y), torch.where(tc, a, b)
                actual = fr.autograd.grad(output, (x, y), g, create_graph=graph)
                wanted = torch.autograd.grad(expected, (a, b), tg, create_graph=graph)
                for got, want in zip(actual, wanted): self.same(got, want)
                if graph:
                    actual_mixed = fr.autograd.grad(actual[0].sum(), (x, y, g), allow_unused=True)
                    wanted_mixed = torch.autograd.grad(wanted[0].sum(), (a, b, tg), allow_unused=True)
                    for got, want in zip(actual_mixed, wanted_mixed): self.same(got, want)
                for value in (x, y, g): self.assertIsNone(value.grad)

    def test_clamp_direct_jvp_nonfinite_directions_dtype_layout_and_provenance(self):
        for layout, wide, seed_layout, value in itertools.product(('C', 'T', 'gapT'), (False, True), ('C', 'T'), (.7, float('inf'), float('nan'), -0.)):
            with self.subTest(layout=layout, wide=wide, seed=seed_layout, value=value):
                x, a = (make(lib, (3, 2), layout, 'float32').detach().requires_grad_(True) for lib in (fr, torch))
                dtype = 'float64' if wide else 'float32'
                # A real multiply preserves the chosen seed layout and live history.
                v, tv = (make(lib, (3, 2), seed_layout, dtype).detach().requires_grad_(True) for lib in (fr, torch))
                v, tv = v * 0 + value, tv * 0 + value
                out, dy = fr.func.jvp(lambda q: q.clamp(-1., 1.), (x,), (v,))
                ref, tdy = torch.func.jvp(lambda q: q.clamp(-1., 1.), (a,), (tv,))
                self.same(out, ref); self.same(dy, tdy)
                actual = fr.autograd.grad(dy.sum(), (x, v), allow_unused=True)
                wanted = torch.autograd.grad(tdy.sum(), (a, tv), allow_unused=True)
                for got, want in zip(actual, wanted): self.same(got, want)
        for lower, upper in ((-1., 1.), (2., -2.)):
            values = [[float('nan'), float('inf'), -float('inf')], [-1., 0., 1.]]
            x, a = (lib.tensor(values).T.detach().requires_grad_(True) for lib in (fr, torch))
            v, tv = (lib.full((3, 2), float('inf'), requires_grad=True) for lib in (fr, torch))
            actual = fr.func.jvp(lambda q: q.clamp(lower, upper), (x,), (v,))
            expected = torch.func.jvp(lambda q: q.clamp(lower, upper), (a,), (tv,))
            for got, want in zip(actual, expected): self.same(got, want)

    def test_where_jvp_tracks_only_live_directions_and_masks_nonfinite_values(self):
        for layout, value in itertools.product(('C', 'T', 'gapT'), (float('inf'), float('nan'), -0.)):
            c, tc = (make(lib, (3, 2), layout, 'bool', 0) for lib in (fr, torch))
            x, a = (make(lib, (3, 2), 'T', 'float32').detach().requires_grad_(True) for lib in (fr, torch))
            y, b = (make(lib, (3, 2), 'C', 'float32', 2).requires_grad_(True) for lib in (fr, torch))
            v = fr.full((3, 2), value, requires_grad=True)
            tv = torch.full((3, 2), value, requires_grad=True)
            out, dy = fr.func.jvp(lambda p: fr.where(c, p, y), (x,), (v,))
            ref, tdy = torch.func.jvp(lambda p: torch.where(tc, p, b), (a,), (tv,))
            self.same(out, ref); self.same(dy, tdy)
            actual = fr.autograd.grad(dy.sum(), (x, y, v), allow_unused=True)
            wanted = torch.autograd.grad(tdy.sum(), (a, b, tv), allow_unused=True)
            for got, want in zip(actual, wanted): self.same(got, want)

    def test_condition_versions_are_required_while_branch_and_output_values_are_not(self):
        # Original leaves stay selected when Torch rebases a mutated view.
        for changed in range(4):
            cb = fr.tensor([[True, False, True], [False, True, False]], dtype='bool')
            tcb = torch.tensor([[True, False, True], [False, True, False]])
            xb, ab = (make(lib, (2, 3), 'C', 'float32').requires_grad_(True) for lib in (fr, torch))
            yb, bb = (make(lib, (2, 3), 'C', 'float32', 2).requires_grad_(True) for lib in (fr, torch))
            x, y, a, b = xb.T, yb.T, ab.T, bb.T
            output, expected = fr.where(cb.T, x, y), torch.where(tcb.T, a, b)
            with fr.no_grad(): (cb, xb, yb, output.detach().T)[changed].fill_(0)
            with torch.no_grad(): (tcb, ab, bb, expected.detach().T)[changed].fill_(0)
            if changed == 0:
                with self.assertRaises((ValueError, RuntimeError)): fr.autograd.grad(output.sum(), (xb, yb))
                with self.assertRaises(RuntimeError): torch.autograd.grad(expected.sum(), (ab, bb))
            else:
                for got, want in zip(fr.autograd.grad(output.sum(), (xb, yb)), torch.autograd.grad(expected.sum(), (ab, bb))): self.same(got, want)

    def test_empty_promotion_avoids_huge_materialization_and_invalid_shapes_reject(self):
        for dtype, shapes in itertools.product(('float32', 'float64'), (((3, 0), (3, 0), (1, 1)), ((), (0, 3), (1, 3)), ((2, 0, 3), (), (1, 0, 1)))):
            c, tc = (make(lib, shapes[0], 'T', 'bool', 0) for lib in (fr, torch))
            x, a = (make(lib, shapes[1], 'T', 'float32').detach().requires_grad_(True) for lib in (fr, torch))
            y, b = (make(lib, shapes[2], 'C', dtype, 2).requires_grad_(True) for lib in (fr, torch))
            output, expected = fr.where(c, x, y), torch.where(tc, a, b)
            self.same(output, expected)
            actual = fr.autograd.grad(output, (x, y), fr.ones(output.shape, dtype=output.dtype))
            wanted = torch.autograd.grad(expected, (a, b), torch.ones(expected.shape, dtype=expected.dtype))
            for got, want in zip(actual, wanted): self.same(got, want)
        code = '''import ferro as f
x=f.tensor(1.).expand((2**60,1))
y=f.empty((1,0),dtype=f.float64)
z=f.where(f.tensor(True,dtype=f.bool),x,y)
assert z.shape==[2**60,0] and z.numel()==0 and z.dtype=="float64"
count=(2**63-1)//8+1
numeric=f.tensor(1.).expand((count,))
wide=f.tensor(1.,dtype=f.float64).expand((count,))
boolean=f.tensor(True,dtype=f.bool)
integer=f.tensor(1,dtype=f.int64).expand((count,))
for operation in (lambda:f.logical_and(numeric,numeric),lambda:f.where(wide,boolean,boolean),lambda:f.ge(integer,numeric)):
 try:operation()
 except ValueError:pass
 else:raise AssertionError('materialization capacity must reject before allocation')
'''
        environment = dict(os.environ); environment.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, '-c', code], check=True, timeout=10, env=environment)
        with self.assertRaises((ValueError, RuntimeError)): fr.where(fr.tensor([True, False], dtype='bool'), fr.ones(3), fr.ones(3))


if __name__ == '__main__': unittest.main()
