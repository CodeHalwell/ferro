"""CPU graph utility values and gradients against installed torch_geometric."""
import sys
import unittest

import torch
from torch_geometric import utils as pyg
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

import ferro as fr
from ferro import _graph_utils as graph


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def indices(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    dtype = torch.int64 if value.dtype == 'int64' else torch.float32
    return torch.tensor(value.tolist(), dtype=dtype).reshape(value.shape)


class PyGUtilityTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected, atol=2e-6, rtol=2e-5)

    def test_degree_typed_indices_and_isolated_nodes(self):
        self.assertIs(fr.graph.degree, graph.degree)
        self.assertIs(fr.graph.softmax, graph.softmax)
        for ids, n in (([3, 1, 1, 0, 3], 6), ([], 0), ([], 3)):
            for dtype in (None, 'float32', 'int64'):
                expected = pyg.degree(torch.tensor(ids, dtype=torch.int64), n, dtype=torch.int64 if dtype == 'int64' else torch.float32)
                actual = graph.degree(indices(ids), n, dtype)
                self.close(actual, expected)
                self.assertEqual(actual.dtype, dtype or 'float32')

    def test_grouped_softmax_values_strides_and_gradients(self):
        ids = torch.tensor([2, 0, 2, 1, 0])
        raw = torch.tensor([[.2, -.8, 1.1, .7, -1.3], [.4, .2, -.5, 1.1, .8]])
        a, b = tensor(raw, True), raw.clone().requires_grad_(True)
        out = graph.softmax(a.transpose(0, 1), indices(ids), num_nodes=5)
        ref = pyg.softmax(b.T, ids, num_nodes=5)
        self.close(out, ref)
        probe = torch.linspace(-.7, .9, 10).reshape(5, 2)
        (out * tensor(probe)).sum().backward()
        (ref * probe).sum().backward()
        self.close(a.grad, b.grad)
        for shape in ((0, 3), (4, 0)):
            x = tensor(torch.zeros(shape), True)
            out = graph.softmax(x, indices([] if shape[0] == 0 else [0, 1, 1, 2]), num_nodes=5)
            out.sum().backward()
            self.assertEqual(out.shape, list(shape))
            self.assertEqual(x.grad.shape, list(shape))

    def test_global_pools_values_gradients_and_missing_graphs(self):
        raw = torch.tensor([[-2., .5], [.7, -.9], [1.2, -.3], [.2, .8]])
        ids = torch.tensor([0, 2, 0, 2])
        for native, oracle in ((graph.global_add_pool, global_add_pool), (graph.global_mean_pool, global_mean_pool), (graph.global_max_pool, global_max_pool)):
            a, b = tensor(raw, True), raw.clone().requires_grad_(True)
            out, ref = native(a, indices(ids), size=4), oracle(b, ids, size=4)
            self.close(out, ref)
            (out * out).sum().backward()
            (ref * ref).sum().backward()
            self.close(a.grad, b.grad)
            self.close(native(tensor(raw), None), oracle(raw, None))
            x = tensor(torch.zeros(0, 2), True)
            out = native(x, indices([]), size=3)
            self.close(out, torch.zeros(3, 2))
            out.sum().backward()
            self.assertEqual(x.grad.shape, [0, 2])

    def test_max_ties_explicitly_distinguish_pyg_zero_initialization(self):
        for raw, expected in (([2., 2.], [.5, .5]), ([-1., 0.], [0., 1.])):
            x = tensor(raw, True)
            graph.global_max_pool(x, indices([0, 0])).sum().backward()
            self.close(x.grad, torch.tensor(expected))
        # PyG 2.8's CPU scatter_reduce includes its initial zero in the
        # derivative count. Ferro divides only among actual maximum inputs.
        x = torch.tensor([-1., 0.], requires_grad=True)
        global_max_pool(x, torch.tensor([0, 0])).sum().backward()
        self.assertEqual(x.grad.tolist(), [0., .5])
        empty = tensor(torch.empty(0, 2), True)
        mean = graph.global_mean_pool(empty, None)
        self.assertTrue(torch.isnan(values(mean)).all())
        mean.sum().backward()
        self.assertEqual(empty.grad.shape, [0, 2])
        with self.assertRaises(ValueError):
            graph.global_max_pool(empty, None)

    def test_self_loops_scalar_tensor_and_reduced_attributes(self):
        edge = torch.tensor([[0, 1, 2, 0, 1], [1, 1, 0, 1, 2]])
        attrs = torch.tensor([[.3, -.7], [-.2, .4], [.9, .8], [.6, -.1], [-.5, .2]])
        for fill in (None, .25, torch.tensor([.2, -.3]), 'add', 'mean', 'min', 'max'):
            a, b = tensor(attrs, True), attrs.clone().requires_grad_(True)
            nf = tensor(fill, True) if isinstance(fill, torch.Tensor) else fill
            tf = fill.clone().requires_grad_(True) if isinstance(fill, torch.Tensor) else fill
            actual, attr = graph.add_self_loops(indices(edge), a, nf, num_nodes=4)
            expected, ref = pyg.add_self_loops(edge, b, tf, num_nodes=4)
            self.close(actual, expected)
            self.close(attr, ref)
            probe = torch.linspace(.1, 1., attr.shape[0] * 2).reshape(attr.shape)
            (attr * tensor(probe)).sum().backward()
            (ref * probe).sum().backward()
            self.close(a.grad, b.grad)
            if isinstance(nf, fr.Tensor):
                self.close(nf.grad, tf.grad)
        a, b = tensor(attrs, True), attrs.clone().requires_grad_(True)
        out, attr = graph.remove_self_loops(indices(edge), a)
        expected, ref = pyg.remove_self_loops(edge, b)
        self.close(out, expected)
        attr.sum().backward(); ref.sum().backward()
        self.close(a.grad, b.grad)

    def test_rectangular_loops_and_empty_attributes(self):
        edge = torch.tensor([[0, 1, 0], [1, 3, 2]])
        attrs = torch.tensor([[.2], [.4], [.9]])
        for size in ((2, 5), (5, 4), (0, 4)):
            edges, data = (edge, attrs) if size[0] else (torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 1))
            actual, attr = graph.add_self_loops(indices(edges), tensor(data), 'mean', size)
            if edges.shape[1] and int(edges[1].max()) >= min(size):
                # PyG 2.8's loop utility reduces into min(size), which rejects
                # valid higher destination IDs. Compare the underlying PyG
                # scatter over the full target size for this supported extension.
                with self.assertRaises(RuntimeError):
                    pyg.add_self_loops(edges, data, 'mean', size)
                loops = torch.arange(min(size)).repeat(2, 1)
                expected = torch.cat([edges, loops], dim=1)
                ref = torch.cat([data, pyg.scatter(data, edges[1], dim=0, dim_size=size[1], reduce='mean')[:min(size)]])
            else:
                expected, ref = pyg.add_self_loops(edges, data, 'mean', size)
            self.close(actual, expected); self.close(attr, ref)
        native = fr.graph.EdgeIndex(5, [0], [1])
        self.assertEqual(graph.add_self_loops(native)[0].shape, [2, 6])

    def test_coalesce_duplicates_sorting_and_gradient_reductions(self):
        edge = torch.tensor([[2, 0, 2, 1, 0], [1, 2, 1, 0, 2]])
        attrs = torch.tensor([[.2, -.3], [-.4, .6], [.8, -.9], [-.2, .7], [.5, -.1]])
        for reduce in ('sum', 'mean', 'min', 'max'):
            for by_row in (True, False):
                a, b = tensor(attrs, True), attrs.clone().requires_grad_(True)
                out, attr = graph.coalesce(indices(edge), a, 4, reduce, sort_by_row=by_row)
                expected, ref = pyg.coalesce(edge, b, 4, reduce, sort_by_row=by_row)
                self.close(out, expected); self.close(attr, ref)
                (attr * attr).sum().backward(); (ref * ref).sum().backward()
                self.close(a.grad, b.grad)
        self.close(graph.coalesce(indices(edge)), pyg.coalesce(edge))
        out, attr = graph.coalesce(indices(edge), None)
        self.assertIsNone(attr)
        self.close(out, pyg.coalesce(edge))
        sorted_edge = pyg.coalesce(edge)
        self.close(graph.coalesce(indices(sorted_edge), is_sorted=True), sorted_edge)
        with self.assertRaisesRegex(ValueError, 'sorted'):
            graph.coalesce(indices(edge), is_sorted=True)

    def test_undirected_and_subgraph_preserve_attribute_gradients(self):
        edge = torch.tensor([[2, 0, 2, 1, 0, 3], [1, 2, 1, 0, 2, 3]])
        raw = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 7 + .1
        for operation in ('undirected', 'subgraph'):
            a, b = tensor(raw, True), raw.clone().requires_grad_(True)
            if operation == 'undirected':
                out, attr = graph.to_undirected(indices(edge), a, 5, 'mean')
                expected, ref = pyg.to_undirected(edge, b, 5, 'mean')
            else:
                out, attr = graph.subgraph(indices([2, 0, 1]), indices(edge), a, True, 5)
                expected, ref = pyg.subgraph(torch.tensor([2, 0, 1]), edge, b, True, 5)
            self.close(out, expected); self.close(attr, ref)
            (attr * attr).sum().backward(); (ref * ref).sum().backward()
            self.close(a.grad, b.grad)
        self.close(graph.to_undirected(indices(edge)), pyg.to_undirected(edge))
        out, attr = graph.subgraph([], indices(edge), tensor(raw), num_nodes=5)
        self.assertEqual(out.shape, [2, 0]); self.assertEqual(attr.shape, [0, 2])

    def test_strict_dtype_shape_bounds_and_unsupported_requests(self):
        edge = indices([[0, 1], [1, 0]])
        for bad in (tensor([[0, 1], [1, 0]]), tensor(torch.empty(2, 0)), indices([0, 1]), indices([[0], [-1]])):
            with self.assertRaises((TypeError, ValueError)):
                graph._edge_parts(bad)
        for size in ((1, 2), (2, 1), (-1, 2), (True, 2)):
            with self.assertRaises(ValueError):
                graph._edge_parts(edge, size=size)
        with self.assertRaises(ValueError):
            graph._edge_parts(fr.graph.EdgeIndex(3, [0], [1]), num_nodes=2)
        with self.assertRaises(ValueError):
            graph._edge_parts(indices([[2**63 - 1], [0]]))
        for bad in (indices([0, 2]), indices([-1, 0]), tensor([0, 1]), indices([[0, 1]])):
            with self.assertRaises(ValueError):
                graph.global_add_pool(tensor([[1.], [2.]]), bad, size=2)
        with self.assertRaises(ValueError):
            graph.degree(indices([]), num_nodes=sys.maxsize)
        with self.assertRaises(ValueError):
            graph.add_self_loops(edge, tensor([1.]))
        with self.assertRaises(NotImplementedError):
            graph.add_self_loops(edge, tensor([1., 2.]), 'mul')
        with self.assertRaises(NotImplementedError):
            graph.softmax(tensor([1., 2.]), ptr=indices([0, 2]))
        with self.assertRaises(TypeError):
            graph.subgraph([0], edge, return_edge_mask=1)
        with self.assertRaises(ValueError):
            graph.subgraph([0, 0], edge)
        with self.assertRaises(ValueError):
            graph.subgraph([True], edge)
        for operation in (lambda: graph.softmax(tensor([.2, .3]), indices([0, 0])), lambda: graph.global_add_pool(tensor([.2, .3]), indices([0, 0]))):
            with self.assertRaisesRegex(ValueError, 'capture'):
                fr.capture(operation)


if __name__ == '__main__':
    unittest.main()
