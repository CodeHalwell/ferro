"""Heterogeneous sampling alignment and executable PyG layer oracles.

No optional pyg-lib/torch-sparse backend is required or counted as validation.
Finite random samples have Ferro's documented frontier/RNG semantics.
"""
import sys
import tracemalloc
import unittest

import torch
from torch_geometric.data import HeteroData as TorchData
from torch_geometric.nn import HGTConv as TorchHGT

import ferro as fr
from ferro._graph_hetero import HeteroData
from ferro._graph_sampling import NeighborLoader, LinkNeighborLoader


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def graph(grad=False):
    data, reference = HeteroData(), TorchData()
    for offset, (name, count) in enumerate((('a', 5), ('b', 4), ('z', 0), ('unused', 1))):
        raw = torch.arange(count * 2).reshape(2, count).float() / 11 + offset / 7
        data[name].x = tensor(raw, grad).transpose(0, 1)
        reference[name].x = raw.T.clone().requires_grad_(grad)
        data[name].num_nodes = reference[name].num_nodes = count
        data[name].y = index([2**55 + i for i in range(count)])
        reference[name].y = torch.tensor([2**55 + i for i in range(count)], dtype=torch.int64)
        data[name].mask = fr.tensor([i % 2 == 0 for i in range(count)], dtype='bool')
        reference[name].mask = torch.tensor([i % 2 == 0 for i in range(count)], dtype=torch.bool)
        data[name].scale = tensor(.3, grad)
        reference[name].scale = torch.tensor(.3, requires_grad=grad)
    topologies = {
        ('a', 'writes', 'b'): [[0, 1, 1, 2, 3], [0, 1, 1, 2, 1]],
        ('b', 'reads', 'a'): [[0, 1, 1, 2], [1, 0, 2, 3]],
        ('a', 'links', 'a'): [[0, 1, 2, 3], [0, 0, 1, 2]],
        ('z', 'empty', 'z'): [[], []],
    }
    for relation, rows in topologies.items():
        edges = torch.tensor(rows, dtype=torch.int64)
        attrs = torch.arange(edges.shape[1]).reshape(-1, 1).float() / 9 + .2
        data[relation].edge_index = index(edges)
        reference[relation].edge_index = edges
        data[relation].edge_attr = tensor(attrs, grad)
        reference[relation].edge_attr = attrs.clone().requires_grad_(grad)
        data[relation].tag = index([2**55 + i for i in range(edges.shape[1])])
        reference[relation].tag = torch.tensor([2**55 + i for i in range(edges.shape[1])], dtype=torch.int64)
    data.graph_value = tensor(.8, grad)
    reference.graph_value = torch.tensor(.8, requires_grad=grad)
    return data, reference


def signature(batch):
    return ({name: batch[name].n_id.tolist() for name in batch.node_types},
            {key: batch[key].e_id.tolist() for key in batch.edge_types}, batch['b'].input_id.tolist())


class HeterogeneousNeighborhoodTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=4e-6, rtol=5e-5)

    def test_induced_alignment_matches_actual_pyg_subgraph(self):
        data, reference = graph(True)
        batch = next(iter(NeighborLoader(data, [-1, -1], input_nodes=('b', [1, 3]), batch_size=2, subgraph_type='induced')))
        selected = {name: torch.tensor(batch[name].n_id.tolist(), dtype=torch.int64) for name in batch.node_types}
        expected = reference.subgraph(selected)
        self.assertEqual(batch.metadata(), data.metadata())
        self.assertTrue(batch.validate())
        self.assertEqual(batch['b'].n_id.tolist()[:2], [1, 3])
        self.assertEqual(batch['b'].input_id.tolist(), [0, 1])
        self.assertEqual(batch['b'].batch_size, 2)
        self.assertEqual(batch['unused'].num_nodes, 0)
        for name in batch.node_types:
            for key in ('x', 'y', 'mask', 'scale'):
                self.assertEqual(batch[name][key].tolist(), expected[name][key].tolist())
                self.assertEqual(batch[name][key].dtype, str(expected[name][key].dtype).split('.')[-1])
        for relation in batch.edge_types:
            for key in ('edge_index', 'edge_attr', 'tag'):
                self.assertEqual(batch[relation][key].tolist(), expected[relation][key].tolist())
            ids = batch[relation].e_id.tolist()
            self.assertEqual(batch[relation].tag.tolist(), [2**55 + i for i in ids])
        self.assertEqual(batch.collect('input_id'), {'b': batch['b'].input_id})
        self.assertEqual(set(batch.collect('num_sampled_nodes')), set(data.node_types))

    def test_directional_manual_frontiers_and_duplicate_edge_identity(self):
        data, _ = graph()
        batch = next(iter(NeighborLoader(data, [-1, -1], input_nodes=('b', [1, 3]), batch_size=2)))
        self.assertEqual(batch['a'].n_id.tolist(), [1, 3, 2])
        self.assertEqual(batch['b'].n_id.tolist(), [1, 3, 0, 2])
        self.assertEqual(batch['a', 'writes', 'b'].e_id.tolist(), [1, 2, 4])
        self.assertEqual(batch['b', 'reads', 'a'].e_id.tolist(), [0, 3])
        self.assertEqual(batch['a', 'links', 'a'].e_id.tolist(), [2])
        self.assertEqual(batch['a'].num_sampled_nodes, [0, 2, 1])
        self.assertEqual(batch['b'].num_sampled_nodes, [2, 0, 2])
        self.assertEqual(batch['a', 'writes', 'b'].num_sampled_edges, [3, 0])
        for relation in batch.edge_types:
            src, _, dst = relation
            restored = [[batch[src].n_id.tolist()[i] for i in batch[relation].edge_index.tolist()[0]],
                        [batch[dst].n_id.tolist()[i] for i in batch[relation].edge_index.tolist()[1]]]
            original = data[relation].edge_index.tolist()
            self.assertEqual(restored, [[axis[i] for i in batch[relation].e_id.tolist()] for axis in original])

    def test_two_hop_hgt_matches_full_graph_seed_outputs_and_gradients(self):
        data, reference = graph(True)
        batch = next(iter(NeighborLoader(data, [-1, -1], input_nodes=('b', [1, 3]), batch_size=2)))
        torch.manual_seed(149)
        layers = [fr.graph.HGTConv(2, 2, data.metadata(), heads=2) for _ in range(2)]
        oracles = [TorchHGT(2, 2, data.metadata(), heads=2) for _ in range(2)]
        with fr.no_grad():
            for layer, oracle in zip(layers, oracles):
                expected = dict(oracle.named_parameters())
                for name, parameter in layer.named_parameters():
                    parameter.tensor().copy_(tensor(expected[name].detach()))
        actual, wanted = batch.x_dict, reference.x_dict
        for layer, oracle in zip(layers, oracles):
            actual = layer(actual, batch.edge_index_dict)
            wanted = oracle(wanted, reference.edge_index_dict)
        actual = actual['b'].narrow(0, 0, 2)
        wanted = wanted['b'][[1, 3]]
        self.close(actual, wanted)
        probe = torch.tensor([[.3, -.7], [.2, .8]])
        (actual * tensor(probe)).sum().backward()
        (wanted * probe).sum().backward()
        for name in data.node_types:
            if reference[name].x.grad is not None:
                self.close(data[name].x.grad, reference[name].x.grad)
        for layer, oracle in zip(layers, oracles):
            expected = dict(oracle.named_parameters())
            for name, parameter in layer.named_parameters():
                if expected[name].grad is not None:
                    self.close(parameter.tensor().grad, expected[name].grad)

    def test_edge_conditioned_two_hop_messages_preserve_all_field_gradients(self):
        data, reference = graph(True)
        batch = next(iter(NeighborLoader(data, [-1, -1], input_nodes=('b', [1, 3]), batch_size=2)))
        actual, wanted = batch.x_dict, reference.x_dict
        for _ in range(2):
            outputs = {name: x * .2 for name, x in actual.items()}
            targets = {name: x * .2 for name, x in wanted.items()}
            for relation in data.edge_types:
                src, _, dst = relation
                s, d = batch[relation].edge_index.tolist()
                messages = actual[src].index_select(0, s) * (1. + batch[relation].edge_attr)
                outputs[dst] = outputs[dst] + fr.graph.scatter(messages, d, batch[dst].num_nodes, 'sum')
                s, d = reference[relation].edge_index
                target_messages = wanted[src][s] * (1. + reference[relation].edge_attr)
                targets[dst] = targets[dst].index_add(0, d, target_messages)
            actual = {name: value.tanh() for name, value in outputs.items()}
            wanted = {name: value.tanh() for name, value in targets.items()}
        selected = actual['b'].narrow(0, 0, 2)
        self.close(selected, wanted['b'][[1, 3]])
        selected.sum().backward(); wanted['b'][[1, 3]].sum().backward()
        for name in data.node_types:
            if reference[name].x.grad is not None:
                self.close(data[name].x.grad, reference[name].x.grad)
        for relation in data.edge_types:
            expected = reference[relation].edge_attr.grad
            if expected is not None:
                self.close(data[relation].edge_attr.grad, expected)

    def test_finite_fanouts_replacement_induced_and_reproducible_epochs(self):
        data, _ = graph()
        fanouts = {relation: [1, 2] for relation in data.edge_types}
        for replace in (False, True):
            options = dict(input_nodes='b', batch_size=2, shuffle=True, replace=replace, seed=13)
            left, right = NeighborLoader(data, fanouts, **options), NeighborLoader(data, fanouts, **options)
            for _ in range(2):
                a, b = list(left), list(right)
                self.assertEqual([signature(v) for v in a], [signature(v) for v in b])
                self.assertEqual(sorted(i for value in a for i in value['b'].input_id.tolist()), list(range(4)))
                for value in a:
                    self.assertTrue(value.validate())
                    for relation in value.edge_types:
                        target = relation[2]
                        for hop, count in enumerate(value[relation].num_sampled_edges):
                            self.assertLessEqual(count, value[target].num_sampled_nodes[hop] * fanouts[relation][hop])
                        if not replace:
                            ids = value[relation].e_id.tolist()
                            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(left.epoch, 2)
        one = next(iter(NeighborLoader(data, [5], input_nodes=('b', [1]), replace=True, seed=4)))
        self.assertEqual(len(one['a', 'writes', 'b'].e_id.tolist()), 5)
        self.assertLess(len(set(one['a', 'writes', 'b'].e_id.tolist())), 5)

    def test_masks_remainders_empty_types_and_zero_hops(self):
        data, _ = graph()
        loader = NeighborLoader(data, [], input_nodes=('b', fr.tensor([True, False, True, True])), batch_size=2)
        self.assertEqual(len(loader), 2)
        batches = list(loader)
        self.assertEqual([batch['b'].n_id.tolist() for batch in batches], [[0, 2], [3]])
        self.assertEqual([batch['b'].input_id.tolist() for batch in batches], [[0, 1], [2]])
        self.assertTrue(all(batch[relation].num_edges == 0 for batch in batches for relation in data.edge_types))
        self.assertEqual(len(list(NeighborLoader(data, [0], input_nodes='b', batch_size=3, drop_last=True))), 1)
        self.assertEqual(list(NeighborLoader(data, [-1], input_nodes='z')), [])
        isolated = HeteroData()
        isolated['a'].num_nodes = 2
        isolated['a'].x = tensor([[1.], [2.]])
        value = next(iter(NeighborLoader(isolated, [-1, -1], input_nodes='a')))
        self.assertEqual(value['a'].num_sampled_nodes, [1, 0, 0])
        self.assertTrue(value.validate())

    def test_loader_fixes_topology_and_fields_without_detaching_values(self):
        data, _ = graph(True)
        original = data['b'].x
        loader = NeighborLoader(data, [0], input_nodes=('b', [1]), seed=1)
        data['b'].x = tensor(torch.zeros(4, 2))
        data['a', 'writes', 'b'].edge_index = index([[], []])
        batch = next(iter(loader))
        self.assertEqual(batch['b'].x.tolist(), original.index_select(0, [1]).tolist())
        (batch['b'].x.sum() + batch['b'].scale + batch.graph_value).backward()
        self.assertIsNotNone(original.grad)
        self.assertEqual(data['b'].scale.grad.item(), 1.)
        self.assertEqual(data.graph_value.grad.item(), 1.)
        self.assertIsNone(data['b'].x.grad)

    def test_invalid_metadata_fanouts_selectors_and_unsupported_modes(self):
        data, _ = graph()
        for options in ({'input_nodes': None}, {'input_nodes': 'unknown'}, {'input_nodes': ('b', [1, 1])},
                        {'input_nodes': ('b', [4])}, {'input_nodes': ('b', fr.tensor([True]))},
                        {'input_nodes': ('b', tensor([1.]))}, {'input_nodes': 'b', 'replace': 1},
                        {'input_nodes': 'b', 'subgraph_type': 'bidirectional'}, {'input_nodes': 'b', 'num_workers': 1},
                        {'input_nodes': 'b', 'disjoint': True}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError, NotImplementedError)):
                NeighborLoader(data, [1], **options)
        for fanouts in ({}, {key: [1] for key in data.edge_types[:-1]}, [True], [-2], [sys.maxsize],
                        {key: [1] * (i + 1) for i, key in enumerate(data.edge_types)}):
            with self.subTest(fanouts=fanouts), self.assertRaises((ValueError, TypeError)):
                NeighborLoader(data, fanouts, input_nodes='b')
        sampled = next(iter(NeighborLoader(data, [0], input_nodes='b')))
        with self.assertRaises(ValueError): NeighborLoader(sampled, [1], input_nodes='b')
        with self.assertRaises(TypeError): LinkNeighborLoader(data, [1])
        bad = HeteroData(data.to_dict())
        bad['a', 'writes', 'b'].edge_attr = tensor([[1.]])
        with self.assertRaises(ValueError): NeighborLoader(bad, [1], input_nodes='b')

    def test_sparse_large_cardinality_allocates_only_sampled_metadata(self):
        data = HeteroData()
        data['a'].num_nodes = data['b'].num_nodes = 10**8
        data['a', 'to', 'b'].edge_index = index([[10**8 - 1, 7, 7], [3, 3, 3]])
        tracemalloc.start()
        try:
            batch = next(iter(NeighborLoader(data, [-1, -1], input_nodes=('b', [3]), seed=2)))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 1_000_000)
        self.assertEqual(batch['a'].n_id.tolist(), [10**8 - 1, 7])
        self.assertEqual(batch['b'].n_id.tolist(), [3])
        self.assertEqual(batch['a', 'to', 'b'].e_id.tolist(), [0, 1, 2])
        self.assertTrue(batch.validate())


if __name__ == '__main__':
    unittest.main()
