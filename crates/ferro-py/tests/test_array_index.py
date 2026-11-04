import subprocess
import sys
import unittest

import ferro as fr
import torch


class ArrayIndexTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(tuple(actual.stride()), expected.stride())
        self.assertEqual(actual.tolist(), expected.detach().tolist())

    def test_none_and_ellipsis_produce_shared_basic_views(self):
        data = [[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]]
        keys = [None, (None, Ellipsis), (Ellipsis, None), (None, 1, None, Ellipsis, None),
                (slice(None), None, slice(1, None), None, 1), (None, None, Ellipsis, None)]
        for key in keys:
            a = fr.tensor(data, requires_grad=True)
            b = torch.tensor(data, requires_grad=True)
            actual, expected = a[key], b[key]
            self.same(actual, expected)
            self.assertTrue(actual._shares_storage(a))
            ga, = fr.autograd.grad(actual.sum(), a)
            gb, = torch.autograd.grad(expected.sum(), b)
            self.same(ga, gb)
        self.same(fr.tensor(0.5)[None, Ellipsis, None], torch.tensor(0.5)[None, Ellipsis, None])

    def test_multidimensional_tensor_indices_mix_with_basic_keys(self):
        keys = [lambda i: i, lambda i: (slice(None), i), lambda i: (1, slice(None), i),
                lambda i: (i, None, 1, Ellipsis), lambda i: (None, Ellipsis, i, None),
                lambda i: (slice(None), i, 1), lambda i: (None, i, Ellipsis)]
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            data = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
            if dtype == 'int64':
                data[0][0] = [2**63 - 1, -2**63]
            a = fr.tensor(data, dtype=dtype)
            b = torch.tensor(data, dtype=getattr(torch, dtype))
            for shape in ('vector', 'matrix', 'strided', 'empty'):
                if shape == 'vector':
                    i, j = fr.tensor([1, 0, -1]), torch.tensor([1, 0, -1])
                elif shape == 'empty':
                    i, j = fr.empty((2, 0), dtype='int64'), torch.empty((2, 0), dtype=torch.int64)
                else:
                    i, j = fr.tensor([[1, 0], [-1, 1]]), torch.tensor([[1, 0], [-1, 1]])
                    if shape == 'strided':
                        i, j = i.T, j.T
                for key in keys:
                    with self.subTest(dtype=dtype, index=shape, key=keys.index(key)):
                        actual, expected = a[key(i)], b[key(j)]
                        self.same(actual, expected)
                        self.assertFalse(actual._shares_storage(a))

    def test_duplicate_array_first_and_higher_derivatives_keep_live_seeds(self):
        for dtype in ('float32', 'float64'):
            a = fr.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, requires_grad=True)
            b = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=getattr(torch, dtype), requires_grad=True)
            i, j = fr.tensor([[2, 0], [2, 1]]), torch.tensor([[2, 0], [2, 1]])
            y, ty = a[None, :, i, None], b[None, :, j, None]
            seed = fr.ones_like(y, requires_grad=True)
            tseed = torch.ones_like(ty, requires_grad=True)
            graph = dtype == 'float32'
            ga, = fr.autograd.grad(y, a, seed, create_graph=graph)
            gb, = torch.autograd.grad(ty, b, tseed, create_graph=graph)
            self.same(ga, gb)
            if graph:
                next_a, = fr.autograd.grad(ga.sum(), seed, create_graph=True)
                next_b, = torch.autograd.grad(gb.sum(), tseed, create_graph=True)
                self.same(next_a, next_b)

    def test_index_guard_does_not_guard_input_or_seed_values(self):
        for changed in ('input', 'index', 'output', 'seed'):
            a = fr.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
            i = fr.tensor([[2, 0], [2, 1]])
            y = a[:, i]
            seed = fr.ones_like(y, requires_grad=True)
            gradient, = fr.autograd.grad(y, a, seed, create_graph=True)
            target = {'input': a, 'index': i, 'output': y, 'seed': seed}[changed]
            with fr.no_grad():
                target.detach().copy_(fr.zeros_like(target))
            if changed == 'index':
                with self.assertRaises(ValueError):
                    fr.autograd.grad(gradient.sum(), seed)
                with self.assertRaises(ValueError):
                    fr.autograd.grad(y, a, fr.ones_like(y))
            else:
                result, = fr.autograd.grad(gradient.sum(), seed)
                self.assertEqual(result.tolist(), [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]])

    def test_array_jvp_jacfwd_and_reverse_tangent_match_torch(self):
        a = fr.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        b = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        i, j = fr.tensor([[2, 0], [2, 1]]), torch.tensor([[2, 0], [2, 1]])
        for dtype in ('float32', 'float64'):
            va = fr.ones((2, 3), dtype=dtype, requires_grad=True)
            vb = torch.ones((2, 3), dtype=getattr(torch, dtype), requires_grad=True)
            pa, ta = fr.func.jvp(lambda x: x[None, :, i, None], (a,), (va,))
            pb, tb = torch.func.jvp(lambda x: x[None, :, j, None], (b,), (vb,))
            self.same(pa, pb)
            self.same(ta, tb)
            ga, = fr.autograd.grad(ta.sum(), va)
            gb, = torch.autograd.grad(tb.sum(), vb)
            self.same(ga, gb)
        self.same(fr.func.jacfwd(lambda x: x[:, i])(a), torch.func.jacfwd(lambda x: x[:, j])(b))

    def test_scalar_integer_tensor_is_a_basic_view(self):
        a = fr.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        b = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        i, j = fr.tensor(-1), torch.tensor(-1)
        for key_a, key_b in ((i, j), ((None, i, Ellipsis), (None, j, Ellipsis)), ((slice(None), i), (slice(None), j))):
            actual, expected = a[key_a], b[key_b]
            self.same(actual, expected)
            self.assertTrue(actual._shares_storage(a))

    def test_tuple_full_slice_alias_differs_from_bare_slice_exactly_as_torch(self):
        for key in (Ellipsis, (), (Ellipsis,), (slice(None),), (slice(0, 2, 1),)):
            a = fr.ones(2, 3, requires_grad=True)
            b = torch.ones(2, 3, requires_grad=True)
            seed = fr.ones_like(a, requires_grad=True)
            tseed = torch.ones_like(b, requires_grad=True)
            y, ty = a[key], b[key]
            self.assertFalse(y.is_leaf)
            ga, = fr.autograd.grad(y, a, seed)
            gb, = torch.autograd.grad(ty, b, tseed)
            self.assertIs(ga, seed)
            self.assertIs(gb, tseed)
            self.same(ga, gb)
            self.assertEqual(ga.requires_grad, gb.requires_grad)
        a = fr.ones(2, 3, requires_grad=True)
        b = torch.ones(2, 3, requires_grad=True)
        ga, = fr.autograd.grad(a[:].sum(), a)
        gb, = torch.autograd.grad(b[:].sum(), b)
        self.same(ga, gb)

    def test_empty_and_tiny_index_from_huge_expansion_are_bounded(self):
        program = '''
import ferro as fr
x=fr.tensor([[2.]]).expand(2**45,1)
i=fr.tensor([[0],[-1]])
y=x[None,i,None]
assert y.shape==[1,2,1,1,1] and y.tolist()==[[[[[2.]]],[[[2.]]]]]
for dtype in ('float32','float64','int64','bool'):
    x=fr.empty((0,3),dtype=dtype)
    i=fr.empty((2,0),dtype='int64')
    y=x[i]
    assert y.shape==[2,0,3] and y.numel()==0
'''
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_indices_and_remaining_multiple_array_boundary(self):
        a = fr.ones(2, 3)
        for key in ((Ellipsis, Ellipsis), (None, 0, 0, 0), fr.tensor([2]), fr.tensor([-3])):
            with self.assertRaises(ValueError):
                a[key]
        with self.assertRaises(TypeError):
            a[:, fr.tensor([0.])]
        with self.assertRaises(NotImplementedError):
            a[fr.tensor([0]), fr.tensor([0])]


if __name__ == '__main__':
    unittest.main()
