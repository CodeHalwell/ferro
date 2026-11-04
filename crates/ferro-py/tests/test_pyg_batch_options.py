"""Actual PyG collation options and local link-index persistence."""
import tempfile
import unittest

import torch
from torch_geometric.data import Data as TorchData, Batch as TorchBatch, InMemoryDataset as TorchDataset
from torch_geometric.loader import DataLoader as TorchLoader

import ferro as fr
from ferro._graph_data import Data, Batch, DataLoader
from ferro._graph_dataset import InMemoryDataset
from ferro._graph_link import RandomLinkSplit


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def fixtures(grad=False):
    actual, expected = [], []
    for number, nodes in enumerate((3, 0, 2, 4)):
        edges = torch.tensor([[i for i in range(nodes - 1)], [i + 1 for i in range(nodes - 1)]], dtype=torch.int64)
        labels = torch.tensor([[i for i in range(nodes)], [nodes - i - 1 for i in range(nodes)]], dtype=torch.int64)
        raw = torch.arange(nodes * 2).reshape(2, nodes).float() / 7 + number
        attr = torch.arange(edges.shape[1]).float().reshape(-1, 1) / 11
        score = torch.arange(nodes).float() / 9
        roles = {'score': 'graph', 'edge_label': 'graph', 'pos_edge_label': 'graph', 'neg_edge_label': 'graph', 'tag': 'node'}
        graph = Data(x=tensor(raw, grad).transpose(0, 1), edge_index=index(edges), edge_attr=tensor(attr, grad),
                     edge_label_index=index(labels), pos_edge_label_index=index(labels[:, :1]), neg_edge_label_index=index(labels[:, 1:]),
                     edge_label=index([2**54 + i for i in range(nodes)]), pos_edge_label=index([1] if nodes else []),
                     neg_edge_label=index([0] * max(0, nodes - 1)), score=tensor(score, grad), y=tensor(number + .2, grad),
                     tag=fr.tensor([i % 2 == 0 for i in range(nodes)], dtype='bool'), num_nodes=nodes, field_roles=roles)
        reference = TorchData(x=raw.T.detach().requires_grad_(grad), edge_index=edges, edge_attr=attr.clone().requires_grad_(grad),
                              edge_label_index=labels, pos_edge_label_index=labels[:, :1], neg_edge_label_index=labels[:, 1:],
                              edge_label=torch.tensor([2**54 + i for i in range(nodes)], dtype=torch.int64), pos_edge_label=torch.tensor([1] if nodes else [], dtype=torch.int64),
                              neg_edge_label=torch.tensor([0] * max(0, nodes - 1), dtype=torch.int64), score=score.clone().requires_grad_(grad), y=torch.tensor(number + .2, requires_grad=grad),
                              tag=torch.tensor([i % 2 == 0 for i in range(nodes)], dtype=torch.bool), num_nodes=nodes)
        actual.append(graph)
        expected.append(reference)
    return actual, expected


class BatchOptionsTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=getattr(torch, actual.dtype)).reshape(actual.shape), expected.detach(), atol=3e-6, rtol=3e-5)

    def compare(self, actual, expected):
        self.assertTrue(actual.validate())
        self.assertEqual(actual.num_nodes, expected.num_nodes)
        for field in actual.keys():
            self.close(actual[field], expected[field])

    def test_standard_link_offsets_follow_vectors_and_roundtrip_match_pyg(self):
        actual, expected = fixtures()
        follow = ['x', 'edge_index', 'edge_attr', 'edge_label_index', 'pos_edge_label_index', 'neg_edge_label_index', 'score', 'y', 'tag', 'absent', 'x']
        batch = Batch.from_data_list(actual, follow_batch=follow)
        reference = TorchBatch.from_data_list(expected, follow_batch=follow)
        self.compare(batch, reference)
        self.assertEqual(batch.edge_label_index.shape, [2, 9])
        self.assertEqual(batch.edge_label_index_ptr.tolist(), [0, 3, 3, 5, 9])
        self.assertEqual(batch.y_ptr.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(batch.tag.dtype, 'bool')
        self.assertEqual(batch.edge_label.dtype, 'int64')
        for original, restored, wanted in zip(actual, batch.to_data_list(), reference.to_data_list()):
            # Ferro retains the original scalar shape; PyG extraction keeps [1].
            wanted.y = wanted.y.reshape([])
            self.compare(restored, wanted)
            self.assertEqual(restored.keys(), original.keys())
            self.assertTrue(all(not key.endswith('_ptr') for key in restored.keys()))

    def test_link_gather_loss_keeps_feature_edge_and_graph_gradients(self):
        actual, expected = fixtures(True)
        batch, reference = Batch.from_data_list(actual, follow_batch=['edge_label_index']), TorchBatch.from_data_list(expected)
        src, dst = batch.edge_label_index.tolist()
        loss = (batch.x.index_select(0, src) * batch.x.index_select(0, dst)).sum() + batch.edge_attr.square().sum() + batch.score.square().sum() + batch.y.sum()
        wanted = (reference.x[reference.edge_label_index[0]] * reference.x[reference.edge_label_index[1]]).sum() + reference.edge_attr.square().sum() + reference.score.square().sum() + reference.y.sum()
        self.close(loss, wanted)
        loss.backward()
        wanted.backward()
        for a, b in zip(actual, expected):
            for field in ('x', 'edge_attr', 'score', 'y'):
                self.close(a[field].grad, b[field].grad)

    def test_exclusion_matches_pyg_and_loader_forwards_options(self):
        actual, expected = fixtures()
        actual[0].unused = tensor([1., 2.])
        expected[0].unused = torch.tensor([1., 2.])
        follow, exclude = ['x', 'edge_attr', 'edge_label_index', 'absent'], ['edge_attr', 'unused', 'absent']
        batch = Batch.from_data_list(actual, follow, exclude)
        reference = TorchBatch.from_data_list(expected, follow_batch=follow, exclude_keys=exclude)
        self.compare(batch, reference)
        self.assertNotIn('edge_attr_batch', batch)
        self.assertTrue(all('unused' not in graph and graph.edge_attr is None for graph in batch.to_data_list()))
        loader = DataLoader(actual, 2, follow_batch=follow, exclude_keys=exclude)
        oracle = TorchLoader(expected, batch_size=2, follow_batch=follow, exclude_keys=exclude)
        for a, b in zip(loader, oracle):
            self.compare(a, b)
        entirely_excluded = Batch.from_data_list(actual, exclude_keys=set().union(*(graph.keys() for graph in actual)))
        self.assertEqual(entirely_excluded.ptr.tolist(), [0, 3, 3, 5, 9])
        self.assertEqual([graph.num_nodes for graph in entirely_excluded.to_data_list()], [3, 0, 2, 4])
        self.assertTrue(all(not graph.keys() for graph in entirely_excluded.to_data_list()))

    def test_inmemory_local_indices_slices_and_safe_archive_roundtrip(self):
        actual, expected = fixtures(True)
        dataset = InMemoryDataset(data_list=actual)
        reference, slices = TorchDataset.collate(expected)
        for field in ('edge_index', 'edge_label_index', 'pos_edge_label_index', 'neg_edge_label_index'):
            self.close(dataset.data[field], reference[field])
            self.close(dataset.slices[field], slices[field])
        for original, graph in zip(actual, dataset):
            self.assertTrue(graph.validate())
            for field in original.keys():
                self.assertEqual(original[field].tolist(), graph[field].tolist())
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/links.fgraph'
            InMemoryDataset.save(actual, path)
            loaded = InMemoryDataset()
            loaded.load(path)
            rebuilt = Batch.from_data_list(list(loaded), follow_batch=['edge_label_index'])
            self.compare(rebuilt, TorchBatch.from_data_list(expected, follow_batch=['edge_label_index']))
        loss = sum((graph.x.square().sum() for graph in dataset), tensor(0.))
        loss.backward()
        for graph in actual:
            self.close(graph.x.grad, torch.tensor(graph.x.tolist()).reshape(graph.x.shape) * 2)

    def test_random_link_split_results_batch_without_relabeling_supervision(self):
        graphs = []
        for nodes in (4, 6):
            rows = [[i for i in range(nodes)], [(i + 1) % nodes for i in range(nodes)]]
            graph = Data(x=tensor(torch.arange(nodes).float().reshape(nodes, 1)), edge_index=index(rows), num_nodes=nodes)
            graphs.append(RandomLinkSplit(num_val=0, num_test=0, seed=21)(graph)[0])
        batch = Batch.from_data_list(graphs, follow_batch=['edge_label_index'])
        self.assertTrue(batch.validate())
        self.assertEqual(batch.edge_label.tolist(), graphs[0].edge_label.tolist() + graphs[1].edge_label.tolist())
        for actual, original in zip(batch.to_data_list(), graphs):
            self.assertEqual(actual.edge_label_index.tolist(), original.edge_label_index.tolist())
            self.assertEqual(actual.edge_label.tolist(), original.edge_label.tolist())
        reference = TorchBatch.from_data_list([TorchData(**{key: torch.tensor(value.tolist(), dtype=getattr(torch, value.dtype)).reshape(value.shape)
                                                           for key, value in graph.items()}, num_nodes=graph.num_nodes) for graph in graphs])
        self.close(batch.edge_label_index, reference.edge_label_index)

    def test_invalid_index_dtype_bounds_shape_and_custom_hook_defaults(self):
        for label_index in (tensor([[0.], [1.]]), index([[0, 1]]), index([[0], [3]]), index([[-1], [1]])):
            graph = Data(num_nodes=3, edge_label_index=label_index)
            with self.assertRaises((ValueError, TypeError)):
                Batch.from_data_list([graph])
            with self.assertRaises((ValueError, TypeError)):
                InMemoryDataset(data_list=[graph])
            with tempfile.TemporaryDirectory() as directory, self.assertRaises((ValueError, TypeError)):
                InMemoryDataset.save([graph], directory + '/invalid.fgraph')
        graph = Data(num_nodes=2, edge_label_index=index([[0], [1]]), field_roles={'edge_label_index': 'node'})
        with self.assertRaisesRegex(ValueError, 'graph role'):
            Batch.from_data_list([graph])
        graph = Data(num_nodes=3, custom_index=index([[0], [1]]), field_roles={'custom_index': 'graph'})
        custom = Batch.from_data_list([graph, graph])
        self.assertEqual(custom.custom_index.tolist(), [[0, 3], [1, 4]])
        self.assertEqual(Batch.from_data_list([graph], exclude_keys=['custom_index']).num_nodes, 3)
        class CustomData(Data):
            def __inc__(self, key, value, *args):
                return 0
        graph = CustomData(num_nodes=2, edge_index=index([[0], [1]]))
        self.assertEqual(Batch.from_data_list([graph, graph]).edge_index.tolist(), [[0, 0], [1, 1]])

    def test_generated_metadata_collision_and_invalid_options_reject(self):
        actual, _ = fixtures()
        actual[0].x_batch = index([0, 1, 2])
        for graph in actual[1:]:
            graph.x_batch = index(range(graph.num_nodes))
        for graph in actual:
            graph._field_roles['x_batch'] = 'node'
        with self.assertRaisesRegex(ValueError, 'conflict'):
            Batch.from_data_list(actual, follow_batch=['x'])
        valid = Batch.from_data_list(actual, follow_batch=['x'], exclude_keys=['x_batch'])
        self.assertEqual(valid.x_batch.tolist(), valid.batch.tolist())
        for option in ('x', 2, [False], [1]):
            with self.assertRaises(TypeError):
                Batch.from_data_list(actual, follow_batch=option)
            with self.assertRaises(TypeError):
                DataLoader(actual, exclude_keys=option)
        with self.assertRaisesRegex(NotImplementedError, 'excluding edge_index'):
            Batch.from_data_list(actual, exclude_keys=['edge_index', 'x_batch'])
        self.assertTrue(Batch.from_data_list(actual, exclude_keys=['edge_index', 'edge_attr', 'x_batch']).validate())


if __name__ == '__main__':
    unittest.main()
