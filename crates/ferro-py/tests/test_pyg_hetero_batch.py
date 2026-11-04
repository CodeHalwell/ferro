"""Ragged heterogeneous batching against actual PyG."""
import unittest
import torch
from torch_geometric.data import HeteroData as TorchData, Batch as TorchBatch
from torch_geometric.nn import HeteroConv as TorchHeteroConv, SAGEConv as TorchSAGE
from torch_geometric.loader import DataLoader as TorchLoader
import ferro as fr
from ferro._graph_data import Batch, DataLoader
from ferro._graph_hetero import HeteroData
from ferro._graph_hetero_batch import HeteroBatch


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def fixtures(grad=False):
    actual, expected = [], []
    relations = [('a', 'to', 'b'), ('b', 'back', 'a'), ('z', 'empty', 'z')]
    for number, counts in enumerate(((3, 2, 0), (0, 1, 0), (2, 0, 1), (1, 3, 0))):
        left, right = HeteroData(), TorchData()
        for name, nodes in zip(('a', 'b', 'z'), counts):
            raw = torch.arange(nodes * 2).reshape(2, nodes).float() / 11 + number / 7
            left[name].x = tensor(raw, grad).transpose(0, 1)
            right[name].x = raw.T.detach().requires_grad_(grad)
            left[name].num_nodes = right[name].num_nodes = nodes
            left[name].y = index([2**55 + i for i in range(nodes)])
            right[name].y = torch.tensor([2**55 + i for i in range(nodes)], dtype=torch.int64)
            left[name].mask = fr.tensor([i % 2 == 0 for i in range(nodes)], dtype='bool')
            right[name].mask = torch.tensor([i % 2 == 0 for i in range(nodes)], dtype=torch.bool)
            left[name].scale = tensor(.2 + number / 10, grad)
            right[name].scale = torch.tensor(.2 + number / 10, requires_grad=grad)
        for relation in relations:
            ns, nt = left[relation[0]].num_nodes, left[relation[2]].num_nodes
            edges = [[0, 0, ns - 1], [0, nt - 1, 0]] if ns and nt and relation[0] != 'z' else [[], []]
            rows = torch.tensor(edges, dtype=torch.int64)
            attrs = torch.arange(rows.shape[1] * 2).reshape(-1, 2).float() / 13 + number / 17
            left[relation].edge_index, right[relation].edge_index = index(rows), rows
            left[relation].edge_attr = tensor(attrs, grad)
            right[relation].edge_attr = attrs.clone().requires_grad_(grad)
            left[relation].scale = tensor(.3 + number / 10, grad)
            right[relation].scale = torch.tensor(.3 + number / 10, requires_grad=grad)
        left.label, right.label = index(number), torch.tensor(number)
        left.context = tensor(torch.arange(number + 1).float().reshape(-1, 1), grad)
        right.context = torch.arange(number + 1).float().reshape(-1, 1).requires_grad_(grad)
        actual.append(left)
        expected.append(right)
    return actual, expected


class HeteroBatchTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=getattr(torch, actual.dtype)).reshape(actual.shape), expected.detach(), atol=4e-6, rtol=4e-5)

    def compare(self, actual, expected):
        self.assertTrue(actual.validate())
        self.assertEqual(actual.metadata(), expected.metadata())
        self.assertEqual(actual.num_nodes, expected.num_nodes)
        self.assertEqual(actual.num_edges, expected.num_edges)
        for key, value in actual._global.items():
            self.close(value, expected._global_store[key])
        for key in (*actual.node_types, *actual.edge_types):
            for field, value in actual[key].items():
                if field == 'num_nodes':
                    self.assertEqual(value, expected[key].num_nodes)
                else:
                    self.close(value, expected[key][field])

    def test_offsets_follow_metadata_types_and_roundtrip_match_pyg(self):
        actual, expected = fixtures()
        follow = ['x', 'edge_index', 'edge_attr', 'scale', 'label', 'context', 'absent']
        batch = HeteroBatch.from_data_list(actual, follow_batch=follow)
        reference = TorchBatch.from_data_list(expected, follow_batch=follow)
        self.compare(batch, reference)
        self.assertEqual(batch.num_graphs, 4)
        self.assertEqual(batch['a'].ptr.tolist(), [0, 3, 3, 5, 6])
        self.assertEqual(batch['b'].ptr.tolist(), [0, 2, 3, 3, 6])
        self.assertEqual(batch['z'].ptr.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(set(batch.batch_dict), {'a', 'b', 'z'})
        self.assertEqual(batch['a'].y.dtype, 'int64')
        self.assertEqual(batch['a'].mask.dtype, 'bool')
        for original, restored in zip(actual, batch.to_data_list()):
            self.assertTrue(restored.validate())
            self.assertEqual(restored.metadata(), original.metadata())
            self.assertEqual(restored.label.shape, [])
            self.assertEqual(restored.to_dict().keys(), original.to_dict().keys())
            for key in (*restored.node_types, *restored.edge_types):
                self.assertEqual(set(restored[key].keys()), set(original[key].keys()))
                for field in restored[key].keys():
                    if field != 'num_nodes':
                        self.assertEqual(restored[key][field].tolist(), original[key][field].tolist())
                        self.assertEqual(restored[key][field].shape, original[key][field].shape)

    def test_native_cat_split_and_heteroconv_preserve_all_gradients(self):
        actual, expected = fixtures(True)
        batch, reference = HeteroBatch.from_data_list(actual), TorchBatch.from_data_list(expected)
        relations = [('a', 'to', 'b'), ('b', 'back', 'a')]
        model = fr.graph.HeteroConv({key: fr.graph.SAGEConv((2, 2), 2) for key in relations}, aggr='sum')
        oracle = TorchHeteroConv({key: TorchSAGE((2, 2), 2) for key in relations}, aggr='sum')
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(dict(oracle.named_parameters())[name].detach()))
        outputs, targets = model(batch.x_dict, batch.edge_index_dict), oracle(reference.x_dict, reference.edge_index_dict)
        loss, expected_loss = batch.context.square().sum(), reference.context.square().sum()
        for key in outputs:
            self.close(outputs[key], targets[key])
            loss = loss + outputs[key].square().sum()
            expected_loss = expected_loss + targets[key].square().sum()
        for key in batch.node_types:
            loss = loss + batch[key].x.square().sum() + batch[key].scale.square().sum()
            expected_loss = expected_loss + reference[key].x.square().sum() + reference[key].scale.square().sum()
        for key in batch.edge_types:
            loss = loss + batch[key].edge_attr.square().sum() + batch[key].scale.square().sum()
            expected_loss = expected_loss + reference[key].edge_attr.square().sum() + reference[key].scale.square().sum()
        loss.backward()
        expected_loss.backward()
        for a, b in zip(actual, expected):
            self.close(a.context.grad, b.context.grad)
            for key in a.node_types:
                for field in ('x', 'scale'):
                    self.close(a[key][field].grad, b[key][field].grad)
            for key in a.edge_types:
                for field in ('edge_attr', 'scale'):
                    self.close(a[key][field].grad, b[key][field].grad)
        for name, parameter in model.named_parameters():
            self.close(parameter.tensor().grad, dict(oracle.named_parameters())[name].grad)

    def test_excluded_fields_and_graph_metadata_match_pyg(self):
        actual, expected = fixtures()
        actual[0]['a'].unused = tensor(torch.ones(3, 1))
        expected[0]['a'].unused = torch.ones(3, 1)
        follow, exclude = ['x', 'scale', 'edge_attr', 'missing'], ['scale', 'unused', 'edge_attr', 'missing']
        batch = HeteroBatch.from_data_list(actual, follow_batch=follow, exclude_keys=exclude)
        self.compare(batch, TorchBatch.from_data_list(expected, follow_batch=follow, exclude_keys=exclude))
        for restored in batch.to_data_list():
            for store in restored.stores:
                self.assertNotIn('scale', store)
                self.assertNotIn('unused', store)
                self.assertNotIn('edge_attr', store)

    def test_batch_factory_and_lazy_loader_dispatch_match_pyg(self):
        actual, expected = fixtures()
        batch = Batch.from_data_list(actual, follow_batch=['x', 'edge_index'], exclude_keys=['scale'])
        self.assertIsInstance(batch, HeteroBatch)
        self.compare(batch, TorchBatch.from_data_list(expected, follow_batch=['x', 'edge_index'], exclude_keys=['scale']))
        loader = DataLoader(actual, batch_size=3, follow_batch=['x'], exclude_keys=['scale'])
        oracle = TorchLoader(expected, batch_size=3, follow_batch=['x'], exclude_keys=['scale'])
        for left, right in zip(loader, oracle):
            self.assertIsInstance(left, HeteroBatch)
            self.compare(left, right)
        self.assertEqual(len(loader), 2)
        shuffled = DataLoader(actual, batch_size=2, shuffle=True, seed=37)
        twin = DataLoader(actual, batch_size=2, shuffle=True, seed=37)
        for _ in range(2):
            self.assertEqual([batch.label.tolist() for batch in shuffled], [batch.label.tolist() for batch in twin])
        self.assertEqual(len(list(DataLoader(actual, batch_size=3, drop_last=True))), 1)

    def test_empty_stores_and_all_empty_graphs_retain_graph_count(self):
        actual, expected = [], []
        for _ in range(3):
            a, b = HeteroData(), TorchData()
            a['x'].num_nodes = b['x'].num_nodes = 0
            a['x'].x, b['x'].x = tensor(torch.empty(0, 2), True), torch.empty(0, 2, requires_grad=True)
            a['x', 'r', 'x'].edge_index, b['x', 'r', 'x'].edge_index = index([[], []]), torch.empty(2, 0, dtype=torch.int64)
            a['x', 'r', 'x'].scale, b['x', 'r', 'x'].scale = tensor(.2, True), torch.tensor(.2, requires_grad=True)
            actual.append(a); expected.append(b)
        batch = HeteroBatch.from_data_list(actual, follow_batch=['x', 'edge_index'])
        self.compare(batch, TorchBatch.from_data_list(expected, follow_batch=['x', 'edge_index']))
        self.assertEqual(batch['x'].ptr.tolist(), [0, 0, 0, 0])
        self.assertEqual(batch['x', 'r', 'x'].num_edges, 0)
        batch['x'].x.sum().backward()
        self.assertTrue(all(graph['x'].x.grad.shape == [0, 2] for graph in actual))
        self.assertEqual(len(batch.to_data_list()), 3)

    def test_clone_keeps_autograd_and_independent_typed_fields(self):
        actual, _ = fixtures(True)
        batch = HeteroBatch.from_data_list(actual, follow_batch=['x'])
        copied = batch.clone()
        self.assertEqual(copied['a'].ptr.tolist(), batch['a'].ptr.tolist())
        copied['a'].x.sum().backward()
        self.assertTrue(all(graph['a'].x.grad.tolist() == torch.ones(graph['a'].x.shape).tolist() for graph in actual))
        copied['a'].y.copy_(fr.zeros(copied['a'].y.shape, dtype='int64'))
        self.assertNotEqual(copied['a'].y.tolist(), batch['a'].y.tolist())
        selected = batch.clone('x')
        self.assertIsNot(selected['a'].x, batch['a'].x)
        self.assertIs(selected['a'].y, batch['a'].y)

    def test_mismatched_schemas_reserved_metadata_and_custom_hooks_reject(self):
        actual, _ = fixtures()
        with self.assertRaises(ValueError):
            HeteroBatch.from_data_list([])
        with self.assertRaises(TypeError):
            HeteroBatch.from_data_list([actual[0], fr.graph.Data(num_nodes=0)])
        with self.assertRaises(TypeError):
            HeteroBatch.from_data_list([HeteroBatch.from_data_list(actual)])
        bad = actual[0].clone(); del bad['z']
        with self.assertRaises(ValueError):
            HeteroBatch.from_data_list([actual[0], bad])
        bad = actual[0].clone(); bad['a'].batch = index([0, 0, 0])
        with self.assertRaises(ValueError):
            HeteroBatch.from_data_list([bad])
        bad = actual[0].clone(); bad['a'].x_batch = index([0, 0, 0])
        with self.assertRaises(ValueError):
            HeteroBatch.from_data_list([bad], follow_batch=['x'])
        with self.assertRaises(NotImplementedError):
            HeteroBatch.from_data_list(actual, exclude_keys=['edge_index'])
        bad = actual[0].clone(); bad['a', 'to', 'b'].other_index = index([0, 1, 2])
        with self.assertRaises(NotImplementedError):
            HeteroBatch.from_data_list([bad])
        class Custom(HeteroData):
            def __inc__(self, key, value):
                return 0
        with self.assertRaises(NotImplementedError):
            HeteroBatch.from_data_list([Custom()])

    def test_metadata_mutation_and_capacity_fail_before_reconstruction(self):
        actual, _ = fixtures()
        batch = HeteroBatch.from_data_list(actual)
        batch['a'].ptr.copy_(index([0, 0, 0, 0, 6]))
        self.assertFalse(batch.validate(False))
        with self.assertRaises(ValueError):
            batch.to_data_list()
        batch = HeteroBatch.from_data_list(actual)
        edges = batch['a', 'to', 'b'].edge_index.tolist()
        edges[1][0] = 3
        batch['a', 'to', 'b'].edge_index.copy_(index(edges))
        self.assertFalse(batch.validate(False))
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            batch.to_data_list()
        huge = HeteroData(); huge['a'].num_nodes = 2**63 - 1
        with self.assertRaises(ValueError):
            HeteroBatch.from_data_list([huge])


if __name__ == '__main__':
    unittest.main()
