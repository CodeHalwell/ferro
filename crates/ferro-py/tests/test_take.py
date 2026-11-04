import math
import subprocess
import sys
import unittest

import ferro as fr
import torch


class TakeTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(tuple(actual.stride()), expected.stride())
        a = actual.reshape([actual.numel()]).tolist()
        b = expected.reshape(-1).tolist()
        for left, right in zip(a, b):
            if isinstance(right, float) and math.isnan(right):
                self.assertTrue(math.isnan(left))
            else:
                self.assertEqual(left, right)
                if isinstance(right, float) and right == 0:
                    self.assertEqual(math.copysign(1, left), math.copysign(1, right))

    def test_typed_strided_scalar_and_multidimensional_indices(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            data = [[1, 0, 3], [4, 5, 6]]
            if dtype == 'int64':
                data = [[2**63 - 1, -2**63, 2**53 + 1], [4, 5, 6]]
            elif dtype in ('float32', 'float64'):
                data = [[1., -0., math.nan], [math.inf, -math.inf, 6.]]
            a = fr.tensor(data, dtype=dtype).T
            b = torch.tensor(data, dtype=getattr(torch, dtype)).T
            for ids in (0, -1, [[-1, 1], [1, 0]], []):
                with self.subTest(dtype=dtype, ids=ids):
                    i = fr.tensor(ids, dtype='int64')
                    j = torch.tensor(ids, dtype=torch.int64)
                    self.same(fr.take(a, i), torch.take(b, j))
                    self.same(a.take(i), b.take(j))
        self.same(fr.tensor(0.25).take(fr.tensor([-1, 0])), torch.tensor(0.25).take(torch.tensor([-1, 0])))

    def test_first_order_gradients_and_source_layout(self):
        for dtype in ('float32', 'float64'):
            for layout in ('transpose', 'gapped', 'expand'):
                a = fr.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, requires_grad=True)
                b = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=getattr(torch, dtype), requires_grad=True)
                if layout == 'transpose':
                    a, b = a.T, b.T
                elif layout == 'gapped':
                    a, b = a[:, ::2], b[:, ::2]
                else:
                    a, b = a[:1].expand(2, 3), b[:1].expand(2, 3)
                i = fr.tensor([[1, 0], [-1, 1]])
                j = torch.tensor([[1, 0], [-1, 1]])
                seed = [[0.25, -0.5], [0.75, 1.]]
                ga, = fr.autograd.grad(a.take(i), a, fr.tensor(seed, dtype=dtype))
                gb, = torch.autograd.grad(b.take(j), b, torch.tensor(seed, dtype=getattr(torch, dtype)))
                self.same(ga, gb)
                self.assertIsNone(a.grad)

    def test_live_seed_higher_derivatives_and_disconnected_input(self):
        a = fr.tensor([0.25, 0.5, 0.75], requires_grad=True)
        b = torch.tensor([0.25, 0.5, 0.75], requires_grad=True)
        i, j = fr.tensor([1, 0, 1]), torch.tensor([1, 0, 1])
        ga = fr.tensor([2., 3., 4.], requires_grad=True)
        gb = torch.tensor([2., 3., 4.], requires_grad=True)
        da, = fr.autograd.grad(a.take(i), a, ga, create_graph=True)
        db, = torch.autograd.grad(b.take(j), b, gb, create_graph=True)
        self.same(da, db)
        aa, sa = fr.autograd.grad(da, (a, ga), fr.tensor([7., 11., 13.]), allow_unused=True, create_graph=True)
        ab, sb = torch.autograd.grad(db, (b, gb), torch.tensor([7., 11., 13.]), allow_unused=True, create_graph=True)
        self.assertIsNone(aa)
        self.assertIsNone(ab)
        self.same(sa, sb)

    def test_saved_input_index_seed_and_allowed_output_mutations(self):
        for changed in ('input', 'index', 'output'):
            a = fr.tensor([1., 2., 3.], requires_grad=True)
            i = fr.tensor([1, 0, 1])
            output = a.take(i)
            target = {'input': a, 'index': i, 'output': output}[changed]
            with fr.no_grad():
                target.detach().copy_(fr.zeros_like(target))
            if changed == 'output':
                gradient, = fr.autograd.grad(output, a, fr.ones_like(output))
                self.assertEqual(gradient.tolist(), [1., 2., 0.])
            else:
                with self.assertRaises(ValueError):
                    fr.autograd.grad(output, a, fr.ones_like(output))
        for changed in ('input', 'seed', 'index', 'gradient'):
            a = fr.tensor([1., 2., 3.], requires_grad=True)
            i = fr.tensor([1, 0, 1])
            seed = fr.tensor([2., 3., 4.], requires_grad=True)
            gradient, = fr.autograd.grad(a.take(i), a, seed, create_graph=True)
            target = {'input': a, 'seed': seed, 'index': i, 'gradient': gradient}[changed]
            with fr.no_grad():
                target.detach().copy_(fr.zeros_like(target))
            if changed in ('seed', 'index'):
                with self.assertRaises(ValueError):
                    fr.autograd.grad(gradient.sum(), seed)
            else:
                result, = fr.autograd.grad(gradient.sum(), seed)
                self.assertEqual(result.tolist(), [1., 1., 1.])

    def test_forward_directions_and_reverse_over_jvp(self):
        for dtype in ('float32', 'float64'):
            a = fr.tensor([1., 2., 3.], requires_grad=True)
            b = torch.tensor([1., 2., 3.], requires_grad=True)
            va = fr.tensor([0.25, 0.5, 0.75], dtype=dtype, requires_grad=True)
            vb = torch.tensor([0.25, 0.5, 0.75], dtype=getattr(torch, dtype), requires_grad=True)
            i, j = fr.tensor([1, 0, 1]), torch.tensor([1, 0, 1])
            pa, ta = fr.func.jvp(lambda x: x.take(i), (a,), (va,))
            pb, tb = torch.func.jvp(lambda x: x.take(j), (b,), (vb,))
            self.same(pa, pb)
            self.same(ta, tb)
            ga, = fr.autograd.grad(ta.sum(), va, retain_graph=True)
            gb, = torch.autograd.grad(tb.sum(), vb, retain_graph=True)
            self.same(ga, gb)
            with fr.no_grad():
                va.detach().copy_(fr.zeros_like(va))
            with self.assertRaises(ValueError):
                fr.autograd.grad(ta.sum(), va)

    def test_nonfinite_duplicate_scatter_and_no_grad_flags(self):
        a = fr.tensor([1., 2., 3.], requires_grad=True)
        b = torch.tensor([1., 2., 3.], requires_grad=True)
        i, j = fr.tensor([1, 1]), torch.tensor([1, 1])
        for values in ([math.inf, -math.inf], [math.nan, 1.], [-0., -0.]):
            ga, = fr.autograd.grad(a.take(i), a, fr.tensor(values))
            gb, = torch.autograd.grad(b.take(j), b, torch.tensor(values))
            self.same(ga, gb)
        with fr.no_grad():
            self.assertFalse(a.take(i).requires_grad)

    def test_empty_and_huge_broadcast_sources_are_bounded(self):
        program = '''
import ferro as fr
for dtype in ('float32','float64','int64','bool'):
    x=fr.tensor([1],dtype=dtype).expand(2**62)
    y=x.take(fr.tensor([0,-1]))
    assert y.tolist()==[True,True] if dtype=='bool' else y.tolist()==[1,1]
    ids=fr.empty((2**45,0),dtype='int64')
    y=x.take(ids)
    assert y.shape==[2**45,0] and y.numel()==0 and y.stride()==(1,1)
'''
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        a = fr.empty((2, 0), requires_grad=True)
        b = torch.empty((2, 0), requires_grad=True)
        ga, = fr.autograd.grad(a.take(fr.tensor([], dtype='int64')).sum(), a)
        gb, = torch.autograd.grad(b.take(torch.tensor([], dtype=torch.int64)).sum(), b)
        self.same(ga, gb)

    def test_invalid_indices_and_explicit_out_boundary(self):
        a = fr.tensor([1., 2., 3.])
        for values in ([3], [-4], [-2**63]):
            with self.assertRaises(ValueError):
                a.take(fr.tensor(values))
        with self.assertRaises(ValueError):
            a.take(fr.tensor([0.]))
        with self.assertRaises(TypeError):
            fr.take(a, [0])
        with self.assertRaises(NotImplementedError):
            fr.take(a, fr.tensor([0]), out=fr.zeros(1))


if __name__ == '__main__':
    unittest.main()
