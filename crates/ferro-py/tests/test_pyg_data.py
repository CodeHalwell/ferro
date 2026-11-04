"""Ragged homogeneous Data/Batch compatibility and gradient ownership."""
import random
import unittest

import torch
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.loader import DataLoader as PyGLoader

import ferro as fr
from ferro._graph_data import Data, Batch, DataLoader
from ferro._graph_utils import global_add_pool


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def indices(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.int64 if value.dtype == 'int64' else torch.float32).reshape(value.shape)


def fixtures(grad=False):
    native, reference = [], []
    for i, (n, s, t) in enumerate(((3, [0, 1, 1], [1, 2, 2]), (2, [1], [0]), (0, [], []), (4, [], []))):
        x = (torch.arange(n * 2).reshape(n, 2).float() - i) / 5
        attr = torch.arange(len(s), dtype=torch.float32).reshape(len(s), 1) / 7 + .2
        edge = torch.tensor([s, t], dtype=torch.int64)
        label = torch.tensor([2**54 + i], dtype=torch.int64)
        native.append(Data(x=tensor(x, grad), edge_index=indices(edge), edge_attr=tensor(attr, grad), y=indices(label), num_nodes=n))
        reference.append(PyGData(x=x.clone().requires_grad_(grad), edge_index=edge, edge_attr=attr.clone().requires_grad_(grad), y=label, num_nodes=n))
    return native, reference


class PyGDataTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected, atol=2e-6, rtol=2e-5)

    def test_ragged_batch_matches_pyg_and_roundtrips_exact_labels(self):
        self.assertIs(fr.graph.Data, Data)
        self.assertIs(fr.graph.Batch, Batch)
        self.assertIs(fr.graph.DataLoader, DataLoader)
        source, reference = fixtures()
        actual, expected = Batch.from_data_list(source), PyGBatch.from_data_list(reference)
        self.assertEqual(actual.num_graphs, 4)
        self.assertEqual(actual.num_nodes, 9)
        self.assertEqual(actual.num_edges, 4)
        for key in ('x', 'edge_index', 'edge_attr', 'y', 'batch', 'ptr'):
            self.close(actual[key], expected[key])
        for key in ('edge_index', 'y', 'batch', 'ptr'):
            self.assertEqual(actual[key].dtype, 'int64')
        self.assertTrue(actual.validate())
        roundtrip = actual.to_data_list()
        for original, restored, oracle in zip(source, roundtrip, expected.to_data_list()):
            self.assertEqual(restored.num_nodes, original.num_nodes)
            self.assertEqual(set(restored.keys()), set(original.keys()))
            for key in original.keys():
                self.close(restored[key], oracle[key])
                self.assertEqual(restored[key].tolist(), original[key].tolist())

    def test_cat_pool_and_split_keep_all_feature_and_edge_gradients(self):
        source, reference = fixtures(True)
        actual, expected = Batch.from_data_list(source), PyGBatch.from_data_list(reference)
        split, reference_split = actual.to_data_list(), expected.to_data_list()
        pooled = global_add_pool(actual.x, actual.batch, size=4)
        loss = (pooled * pooled).sum()
        from torch_geometric.nn import global_add_pool as pool
        ref_loss = pool(expected.x, expected.batch, size=4).square().sum()
        for i, (a, b) in enumerate(zip(split, reference_split)):
            loss = loss + (a.x * (i + 1)).sum() + (a.edge_attr * a.edge_attr).sum()
            ref_loss = ref_loss + (b.x * (i + 1)).sum() + b.edge_attr.square().sum()
        loss.backward(); ref_loss.backward()
        for a, b in zip(source, reference):
            self.close(a.x.grad, b.x.grad)
            self.close(a.edge_attr.grad, b.edge_attr.grad)

    def test_scalar_and_ragged_graph_fields_and_explicit_node_labels(self):
        source, reference = [], []
        for n in (2, 3):
            y = torch.tensor([1, 2][:n] if n == 2 else [3, 4, 5], dtype=torch.int64)
            graph_value = torch.tensor(float(n))
            source.append(Data(x=tensor(torch.ones(n, 1)), y=indices(y), graph_value=tensor(graph_value, True), field_roles={'y': 'node', 'graph_value': 'graph'}))
            reference.append(PyGData(x=torch.ones(n, 1), y=y, graph_value=graph_value))
        actual, expected = Batch.from_data_list(source), PyGBatch.from_data_list(reference)
        self.close(actual.y, expected.y)
        self.close(actual.graph_value, expected.graph_value)
        split = actual.to_data_list()
        self.assertEqual([d.graph_value.shape for d in split], [[], []])
        (split[0].graph_value * 2 + split[1].graph_value * 3).backward()
        self.assertEqual([d.graph_value.grad.item() for d in source], [2., 3.])
        for original, restored in zip(source, split):
            self.assertEqual(original.y.tolist(), restored.y.tolist())

    def test_isolated_nodes_empty_graphs_and_native_topology(self):
        a = Data(edge_index=fr.graph.EdgeIndex(4, [0], [1]), y=indices([7]))
        b = Data(edge_index=fr.graph.EdgeIndex(0, [], []), y=indices([8]))
        c = Data(edge_index=fr.graph.EdgeIndex(2, [], []), y=indices([9]))
        batch = Batch.from_data_list([a, b, c])
        self.assertEqual(batch.ptr.tolist(), [0, 4, 4, 6])
        self.assertEqual(batch.batch.tolist(), [0, 0, 0, 0, 2, 2])
        self.assertEqual([d.num_nodes for d in batch.to_data_list()], [4, 0, 2])
        empty = Batch.from_data_list([Data(num_nodes=0), Data(num_nodes=0)])
        self.assertEqual(empty.ptr.tolist(), [0, 0, 0])
        self.assertEqual(empty.batch.dtype, 'int64')
        self.assertEqual([d.num_nodes for d in empty.to_data_list()], [0, 0])

    def test_ambiguous_custom_fields_require_explicit_roles(self):
        edge = indices([[0, 1], [1, 0]])
        bad = Data(x=tensor([[1.], [2.]]), edge_index=edge, custom=indices([4, 5]))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            Batch.from_data_list([bad])
        good = Data(x=tensor([[1.], [2.]]), edge_index=edge, custom=indices([4, 5]), field_roles={'custom': 'edge'})
        result = Batch.from_data_list([good, good])
        self.assertEqual(result.custom.tolist(), [4, 5, 4, 5])
        self.assertEqual(result.to_data_list()[1].custom.dtype, 'int64')

    def test_loader_collates_lazily_matches_pyg_and_counts_epochs(self):
        source, reference = fixtures()
        loader = DataLoader(source, batch_size=3)
        oracle = PyGLoader(reference, batch_size=3)
        self.assertEqual(len(loader), len(oracle))
        for actual, expected in zip(loader, oracle):
            for key in ('x', 'edge_index', 'edge_attr', 'y', 'batch', 'ptr'):
                self.close(actual[key], expected[key])
        self.assertEqual(loader.epoch, 1)
        shuffled = DataLoader(source, batch_size=2, shuffle=True, seed=19)
        for epoch in range(2):
            order = list(range(4)); random.Random(19 + epoch).shuffle(order)
            found = [label for batch in shuffled for label in batch.y.tolist()]
            self.assertEqual(found, [2**54 + i for i in order])
        self.assertEqual(shuffled.epoch, 2)
        self.assertEqual(len(list(DataLoader(source, batch_size=3, drop_last=True))), 1)
        self.assertEqual(list(DataLoader([], batch_size=2)), [])
        calls = []
        class Dataset:
            def __len__(self): return 4
            def __getitem__(self, i):
                calls.append(i)
                return source[i]
        it = iter(DataLoader(Dataset(), batch_size=2))
        self.assertEqual(calls, [])
        next(it)
        self.assertEqual(calls, [0, 1])

    def test_invalid_fields_bounds_and_loader_requests(self):
        with self.assertRaises(ValueError):
            Data(edge_index=tensor(torch.empty(2, 0)))
        bad = Data(x=tensor([[1.], [2.]]), edge_index=indices([[0], [2]]))
        self.assertFalse(bad.validate(raise_on_error=False))
        with self.assertRaises(ValueError):
            Batch.from_data_list([bad])
        for data in (Data(x=tensor([[1.]]), num_nodes=2), Data(edge_index=indices([[], []]), edge_attr=tensor([[1.]]), num_nodes=2), Data(x=tensor([[1.]]), field_roles={'missing': 'node'})):
            with self.assertRaises(ValueError):
                Batch.from_data_list([data])
        with self.assertRaises(ValueError):
            Batch.from_data_list([Data(x=tensor([[1.]])), Data(x=tensor([[1., 2.]]))])
        with self.assertRaises(ValueError):
            Batch.from_data_list([Data(x=tensor([[1.]])), Data(y=indices([1]))])
        with self.assertRaises(ValueError):
            Batch.from_data_list([])
        with self.assertRaises(TypeError):
            Data(y=[1])
        with self.assertRaises(ValueError):
            Data(batch=indices([0]))
        self.assertEqual(Data(x=fr.Tensor.from_f64([1.], [1, 1])).x.dtype, 'float64')
        for kwargs in ({'batch_size': True}, {'batch_size': 0}, {'num_workers': 1}, {'shuffle': 1}, {'seed': 1.5}):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                DataLoader([], **kwargs)


if __name__ == '__main__':
    unittest.main()
