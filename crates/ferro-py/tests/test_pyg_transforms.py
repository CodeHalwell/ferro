"""Executable PyG transform/subgraph oracles and deterministic random invariants."""
import unittest

import torch
from torch_geometric.data import Data as TorchData
from torch_geometric import transforms as pt, utils as pu

import ferro as fr
from ferro._graph_data import Data
from ferro import _graph_transforms as gt


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    dtype = {'float32': torch.float32, 'int64': torch.int64, 'bool': torch.bool}[value.dtype]
    return torch.tensor(value.tolist(), dtype=dtype).reshape(value.shape)


class GraphTransformTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected, atol=2e-6, rtol=2e-5)

    def test_normalize_and_compose_values_strides_gradients_and_nonmutation(self):
        raw = torch.tensor([[-.8, .2, .4], [.3, -.2, .9]])
        source, reference = tensor(raw, True), raw.clone().requires_grad_(True)
        data = Data(x=source.transpose(0, 1))
        result = gt.Compose([gt.NormalizeFeatures()])(data)
        wanted = pt.Compose([pt.NormalizeFeatures()])(TorchData(x=reference.T))
        self.close(result.x, wanted.x)
        self.assertEqual(data.x.tolist(), raw.T.tolist())
        probe = torch.tensor([[.3, -.2], [.7, .4], [-.1, .9]])
        (result.x * tensor(probe)).sum().backward()
        (wanted.x * probe).sum().backward()
        self.close(source.grad, reference.grad)
        for shape in ((0, 3), (2, 0)):
            empty = tensor(torch.empty(shape), True)
            result = gt.NormalizeFeatures()(Data(x=empty))
            self.close(result.x, pt.NormalizeFeatures()(TorchData(x=torch.empty(shape))).x)
            result.x.sum().backward()
            self.assertEqual(empty.grad.shape, list(shape))
        self.assertEqual(len(gt.Compose([])([data, data])), 2)

    def test_loop_transforms_values_attributes_gradients_and_all_field_alignment(self):
        edge = torch.tensor([[0, 1, 1, 2], [1, 1, 0, 2]])
        raw = torch.tensor([[.2, .7], [.4, -.2], [-.8, .9], [.3, .1]])
        for operation in ('add', 'remove'):
            a, b = tensor(raw, True), raw.clone().requires_grad_(True)
            source = Data(edge_index=index(edge), edge_attr=a, num_nodes=4)
            if operation == 'add':
                result = gt.AddSelfLoops(attr='edge_attr', fill_value='mean')(source)
                wanted = pt.AddSelfLoops(attr='edge_attr', fill_value='mean')(TorchData(edge_index=edge, edge_attr=b, num_nodes=4))
            else:
                result = gt.RemoveSelfLoops(attr='edge_attr')(source)
                wanted = pt.RemoveSelfLoops(attr='edge_attr')(TorchData(edge_index=edge, edge_attr=b, num_nodes=4))
            self.close(result.edge_index, wanted.edge_index)
            self.close(result.edge_attr, wanted.edge_attr)
            (result.edge_attr * result.edge_attr).sum().backward()
            wanted.edge_attr.square().sum().backward()
            self.close(a.grad, b.grad)
            self.assertEqual(source.edge_index.tolist(), edge.tolist())
        source = Data(edge_index=index(edge), edge_weight=tensor([1., 2., 3., 4.]), edge_attr=index([8, 9, 10, 11]), edge_mask=fr.tensor([True, False, False, True]), num_nodes=4, field_roles={'edge_mask': 'edge'})
        result = gt.RemoveSelfLoops()(source)
        self.assertEqual(result.edge_attr.tolist(), [8, 10])
        self.assertEqual(result.edge_mask.tolist(), [True, False])
        with self.assertRaisesRegex(ValueError, 'other edge fields'):
            gt.AddSelfLoops()(source)

    def test_undirected_reduces_multiple_fields_against_pyg(self):
        edge = torch.tensor([[0, 1, 1, 2, 0], [1, 0, 2, 2, 1]])
        raw = torch.tensor([[.2, -.1], [.7, .5], [.9, -.3], [.8, .2], [-.2, .4]])
        for reduction in ('add', 'mean', 'min', 'max'):
            a, b = tensor(raw, True), raw.clone().requires_grad_(True)
            weight, tw = tensor([.3, .7, .9, .2, .4], True), torch.tensor([.3, .7, .9, .2, .4], requires_grad=True)
            source = Data(edge_index=index(edge), edge_attr=a, edge_weight=weight, num_nodes=4)
            result = gt.ToUndirected(reduction)(source)
            wanted = pt.ToUndirected(reduction)(TorchData(edge_index=edge, edge_attr=b, edge_weight=tw, num_nodes=4))
            self.close(result.edge_index, wanted.edge_index)
            self.close(result.edge_attr, wanted.edge_attr)
            self.close(result.edge_weight, wanted.edge_weight)
            (result.edge_attr.sum() + result.edge_weight.sum()).backward()
            (wanted.edge_attr.sum() + wanted.edge_weight.sum()).backward()
            self.close(a.grad, b.grad); self.close(weight.grad, tw.grad)

    def test_random_node_split_counts_class_balance_and_reproducible_calls(self):
        labels = torch.tensor([0, 1, 2] * 5)
        data = Data(x=tensor(torch.ones(15, 2)), y=index(labels), field_roles={'y': 'node'})
        for mode in ('train_rest', 'test_rest', 'random'):
            options = dict(split=mode, num_splits=3, num_train_per_class=2, num_val=.2, num_test=4)
            a, b = gt.RandomNodeSplit(**options, seed=7), gt.RandomNodeSplit(**options, seed=7)
            for _ in range(2):
                result, repeat = a(data), b(data)
                wanted = pt.RandomNodeSplit(**options)(TorchData(x=torch.ones(15, 2), y=labels))
                for name in ('train_mask', 'val_mask', 'test_mask'):
                    self.assertEqual(result[name].dtype, 'bool')
                    self.assertEqual(result[name].tolist(), repeat[name].tolist())
                    self.assertEqual(values(result[name]).sum(0).tolist(), wanted[name].sum(0).tolist())
                total = sum(values(result[k]).int() for k in ('train_mask', 'val_mask', 'test_mask'))
                self.assertTrue((total <= 1).all())
                if mode != 'train_rest':
                    for column in range(3):
                        for cls in range(3):
                            self.assertEqual(int(((labels == cls) & values(result.train_mask)[:, column]).sum()), 2)
                self.assertTrue(result.validate())
        self.assertNotIn('train_mask', data)
        empty = gt.RandomNodeSplit(key=None, num_val=0, num_test=0)(Data(num_nodes=0))
        self.assertEqual(empty.train_mask.shape, [0])

    def test_dropout_exact_extremes_and_sampled_masks(self):
        edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3], [1, 0, 2, 1, 3, 2, 3]])
        for p in (0., 1.):
            for undirected in (False, True):
                actual = gt.dropout_edge(index(edge), p, undirected, seed=11)
                wanted = pu.dropout_edge(edge, p, undirected)
                self.close(actual[0], wanted[0]); self.close(actual[1], wanted[1])
            actual = gt.dropout_node(index(edge), p, num_nodes=5, relabel_nodes=True, seed=11)
            wanted = pu.dropout_node(edge, p, num_nodes=5, relabel_nodes=True)
            for a, b in zip(actual, wanted): self.close(a, b)
        a = gt.dropout_node(index(edge), .35, num_nodes=5, relabel_nodes=True, seed=13)
        b = gt.dropout_node(index(edge), .35, num_nodes=5, relabel_nodes=True, seed=13)
        self.assertEqual([v.tolist() for v in a], [v.tolist() for v in b])
        expected, _, mask = pu.subgraph(values(a[2]), edge, relabel_nodes=True, return_edge_mask=True)
        self.close(a[0], expected); self.close(a[1], mask)
        edges, ids = gt.dropout_edge(index(edge), .3, True, seed=5)
        self.assertEqual(ids.dtype, 'int64')
        keep = ids.tolist()[:ids.shape[0] // 2]
        expected = torch.cat([edge[:, keep], edge[:, keep].flip(0)], 1)
        self.close(edges, expected)
        inactive = gt.dropout_edge(index(edge), .9, True, training=False)
        self.close(inactive[0], edge)
        self.assertEqual(inactive[1].dtype, 'bool')

    def test_k_hop_subgraph_both_flows_cycles_duplicates_empty_and_mappings(self):
        edge = torch.tensor([[0, 1, 2, 2, 3, 4, 1], [1, 0, 1, 1, 2, 3, 1]])
        for seeds in ([1, 0, 1], [5], []):
            for hops in (0, 1, 3):
                for flow in ('source_to_target', 'target_to_source'):
                    for directed in (False, True):
                        actual = gt.k_hop_subgraph(index(seeds), hops, index(edge), True, 6, flow, directed)
                        wanted = pu.k_hop_subgraph(torch.tensor(seeds, dtype=torch.int64), hops, edge, True, 6, flow, directed)
                        for a, b in zip(actual, wanted): self.close(a, b)
        result = gt.k_hop_subgraph(0, 2, index([[], []]), num_nodes=1)
        self.assertEqual(result[0].tolist(), [0])
        self.assertEqual(result[3].dtype, 'bool')

    def test_bipartite_subgraphs_masks_ordered_ids_and_edge_gradients(self):
        edge = torch.tensor([[0, 3, 1, 2, 3], [2, 0, 0, 3, 2]])
        raw = torch.linspace(-.5, .8, 10).reshape(5, 2)
        choices = [([3, 0, 1], [2, 0]), (torch.tensor([True, True, False, True, False]), torch.tensor([True, False, True, False]))]
        for left, right in choices:
            a, b = tensor(raw, True), raw.clone().requires_grad_(True)
            selections = tuple(fr.tensor(v.tolist()) if isinstance(v, torch.Tensor) else index(v) for v in (left, right))
            actual = gt.bipartite_subgraph(selections, index(edge), a, True, (5, 4), True)
            wanted = pu.bipartite_subgraph((left, right), edge, b, True, (5, 4), True)
            for x, y in zip(actual, wanted): self.close(x, y)
            (actual[1] * actual[1]).sum().backward(); wanted[1].square().sum().backward()
            self.close(a.grad, b.grad)
        masks = (fr.tensor([True, False, False, False, False]), fr.tensor([False, False, True, False]))
        actual = gt.bipartite_subgraph(masks, index(edge), return_edge_mask=True)
        wanted = pu.bipartite_subgraph(tuple(values(v) for v in masks), edge, return_edge_mask=True)
        self.close(actual[0], wanted[0]); self.close(actual[2], wanted[2])
        empty = gt.bipartite_subgraph(([], []), index([[], []]), tensor(torch.empty(0, 2), True), size=(0, 0), return_edge_mask=True)
        self.assertEqual(empty[0].shape, [2, 0]); self.assertEqual(empty[2].dtype, 'bool')

    def test_invalid_arguments_and_transform_alignment_reject(self):
        edge = index([[0], [1]])
        for kwargs in ({'p': -1}, {'p': float('nan')}, {'training': 1}):
            with self.assertRaises((ValueError, TypeError)):
                gt.dropout_edge(edge, **kwargs)
        with self.assertRaises(ValueError):
            gt.k_hop_subgraph([3], 1, edge, num_nodes=2)
        with self.assertRaises(ValueError):
            gt.k_hop_subgraph([0], -1, edge)
        with self.assertRaises(ValueError):
            gt.bipartite_subgraph(([0, 0], [1]), edge)
        with self.assertRaises(TypeError):
            gt.NormalizeFeatures('edge_attr')
        with self.assertRaises(ValueError):
            gt.ToUndirected()(Data(edge_index=edge, edge_attr=index([3]), num_nodes=2))
        with self.assertRaises(ValueError):
            gt.RandomNodeSplit(split='random', key=None)
        with self.assertRaises(ValueError):
            gt.RandomNodeSplit(split='random')(Data(x=tensor([[1.], [2.]]), y=tensor([0., 1.])))


if __name__ == '__main__':
    unittest.main()
