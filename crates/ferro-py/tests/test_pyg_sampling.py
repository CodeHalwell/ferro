"""Neighborhood topology against executable PyG utilities and gradient oracles.

These checks do not require pyg-lib/torch-sparse. They execute PyG's pure tensor
k_hop_subgraph and SAGEConv, and test sampled invariants independently.
"""
import sys
import unittest

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import scatter as torch_scatter
from torch_geometric.nn import SAGEConv as TorchSAGE

import ferro as fr
from ferro._graph_data import Data
from ferro._graph_sampling import NeighborLoader, LinkNeighborLoader
from ferro._graph_link import RandomLinkSplit


def index(rows):
    value = torch.as_tensor(rows, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def pairs(edge):
    return list(zip(*edge.tolist()))


def graph(grad=False):
    edge = torch.tensor([[0, 2, 2, 3, 1, 4, 5, 6, 3, 0], [1, 1, 1, 2, 3, 3, 4, 5, 4, 6]])
    raw = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 13 - .7
    base = tensor(raw, grad)
    attrs = tensor(torch.arange(edge.shape[1], dtype=torch.float32).reshape(-1, 1) / 9, grad)
    data = Data(x=base.transpose(0, 1), edge_index=index(edge), edge_attr=attrs, y=index([10 + i for i in range(8)]), field_roles={'y': 'node'})
    return data, edge, base


class NeighborhoodTests(unittest.TestCase):
    def test_all_neighbors_match_pyg_k_hop_subgraph_and_metadata(self):
        data, edge, _ = graph()
        for kind in ('directional', 'induced'):
            for seeds in ([4, 1], [7], []):
                loader = NeighborLoader(data, [-1, -1], input_nodes=index(seeds), batch_size=2, subgraph_type=kind, seed=7)
                batches = list(loader)
                if not seeds:
                    self.assertEqual(batches, [])
                    continue
                batch = batches[0]
                subset, _, _, mask = k_hop_subgraph(torch.tensor(seeds), 2, edge, relabel_nodes=False, num_nodes=8, directed=kind == 'directional')
                self.assertEqual(set(batch.n_id.tolist()), set(subset.tolist()))
                self.assertEqual(set(batch.e_id.tolist()), set(mask.nonzero().reshape(-1).tolist()))
                self.assertEqual(batch.n_id.tolist()[:len(seeds)], seeds)
                self.assertEqual(batch.input_id.tolist(), list(range(len(seeds))))
                self.assertEqual(batch.batch_size, len(seeds))
                self.assertEqual(batch.y.tolist(), [10 + i for i in batch.n_id.tolist()])
                global_edges = [(batch.n_id.tolist()[s], batch.n_id.tolist()[d]) for s, d in pairs(batch.edge_index)]
                self.assertEqual(global_edges, [pairs(data.edge_index)[i] for i in batch.e_id.tolist()])
                self.assertEqual(batch.edge_attr.tolist(), [data.edge_attr.tolist()[i] for i in batch.e_id.tolist()])
                self.assertTrue(batch.validate())

    def test_all_neighbor_sage_values_and_original_input_gradients(self):
        data, edge, base = graph(True)
        batch = next(iter(NeighborLoader(data, [-1, -1], input_nodes=index([4, 1]), batch_size=2)))
        reference = TorchSAGE(3, 2)
        layer = fr.graph.SAGEConv(3, 2)
        with fr.no_grad():
            for name, parameter in layer.named_parameters():
                parameter.tensor().copy_(tensor(dict(reference.named_parameters())[name].detach()))
        raw = torch.tensor(base.tolist(), requires_grad=True)
        ids = batch.n_id.tolist()
        mapping = {node: i for i, node in enumerate(ids)}
        _, _, _, mask = k_hop_subgraph(torch.tensor([4, 1]), 2, edge, num_nodes=8, directed=True)
        selected_ids = mask.nonzero().reshape(-1).tolist()
        ref_edge = torch.tensor([[mapping[int(edge[0, i])] for i in selected_ids], [mapping[int(edge[1, i])] for i in selected_ids]])
        actual = layer(batch.x, batch.edge_index)
        wanted = reference(raw.T[ids], ref_edge)
        torch.testing.assert_close(torch.tensor(actual.tolist()), wanted.detach(), atol=2e-6, rtol=2e-5)
        probe = torch.tensor([[.3, -.2], [.6, .7]])
        (actual.narrow(0, 0, 2) * tensor(probe)).sum().backward()
        (wanted[:2] * probe).sum().backward()
        torch.testing.assert_close(torch.tensor(base.grad.tolist()), raw.grad, atol=2e-6, rtol=2e-5)
        for name, parameter in layer.named_parameters():
            torch.testing.assert_close(torch.tensor(parameter.tensor().grad.tolist()), dict(reference.named_parameters())[name].grad, atol=2e-6, rtol=2e-5)

    def test_two_layer_cyclic_all_neighbors_matches_full_graph_seed_gradients(self):
        edge = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4, 4, 5], [0, 1, 0, 1, 1, 1, 2, 3, 4, 4]])
        raw = torch.linspace(-.8, .9, 21).reshape(7, 3)
        edge_raw = torch.linspace(-.3, .7, 20).reshape(10, 2)
        data = Data(x=tensor(raw, True), edge_index=index(edge), edge_attr=tensor(edge_raw, True), num_nodes=7)
        sample = next(iter(NeighborLoader(data, [-1, -1], input_nodes=[1, 0, 6], batch_size=3)))
        torch.manual_seed(20)
        reference = [TorchSAGE(3, 2), TorchSAGE(2, 2)]
        layers = [fr.graph.SAGEConv(3, 2), fr.graph.SAGEConv(2, 2)]
        with fr.no_grad():
            for layer, oracle in zip(layers, reference):
                for name, parameter in layer.named_parameters():
                    parameter.tensor().copy_(tensor(dict(oracle.named_parameters())[name].detach()))
        # Edge-conditioned GraphSAGE: both layers also receive the mean incoming
        # edge attributes. This checks original edge-row gradients, including
        # parallel edges, against an independently executed full-graph oracle.
        target = sample.edge_index.tolist()[1]
        edge_mean = fr.graph.scatter(sample.edge_attr, target, sample.num_nodes, 'mean')
        hidden = (layers[0](sample.x, sample.edge_index) + edge_mean).relu()
        actual = layers[1](hidden, sample.edge_index) + edge_mean
        tx, ta = raw.clone().requires_grad_(True), edge_raw.clone().requires_grad_(True)
        ref_edge_mean = torch_scatter(ta, edge[1], dim=0, dim_size=7, reduce='mean')
        ref_hidden = (reference[0](tx, edge) + ref_edge_mean).relu()
        expected = reference[1](ref_hidden, edge) + ref_edge_mean
        seeds = [1, 0, 6]
        torch.testing.assert_close(torch.tensor(actual.tolist())[:3], expected[seeds].detach(), atol=2e-6, rtol=2e-5)
        probe = torch.tensor([[.3, -.8], [.7, .2], [-.6, .4]])
        (actual.narrow(0, 0, 3) * tensor(probe)).sum().backward()
        (expected[seeds] * probe).sum().backward()
        torch.testing.assert_close(torch.tensor(data.x.grad.tolist()), tx.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.tensor(data.edge_attr.grad.tolist()), ta.grad, atol=2e-6, rtol=2e-5)
        for layer, oracle in zip(layers, reference):
            for name, parameter in layer.named_parameters():
                torch.testing.assert_close(torch.tensor(parameter.tensor().grad.tolist()), dict(oracle.named_parameters())[name].grad, atol=2e-6, rtol=2e-5)

    def test_seed_epoch_determinism_fanout_frontiers_and_replacement(self):
        data, _, _ = graph()
        loaders = [NeighborLoader(data, [1, 2], input_nodes=index([4, 1, 3, 5, 6]), batch_size=2, shuffle=True, seed=21) for _ in range(2)]
        epochs = []
        for epoch in range(2):
            a, b = list(loaders[0]), list(loaders[1])
            self.assertEqual([(v.n_id.tolist(), v.e_id.tolist(), v.input_id.tolist()) for v in a], [(v.n_id.tolist(), v.e_id.tolist(), v.input_id.tolist()) for v in b])
            epochs.append([i for v in a for i in v.input_id.tolist()])
            for sample in a:
                self.assertEqual(sum(sample.num_sampled_nodes), sample.num_nodes)
                self.assertEqual(sum(sample.num_sampled_edges), sample.num_edges)
                self.assertEqual(sample.n_id.tolist()[:sample.batch_size], [[4, 1, 3, 5, 6][i] for i in sample.input_id.tolist()])
                start, edge_start = 0, 0
                local_edges = pairs(sample.edge_index)
                for hop, fanout in enumerate((1, 2)):
                    frontier_count = sample.num_sampled_nodes[hop]
                    ids = local_edges[edge_start:edge_start + sample.num_sampled_edges[hop]]
                    self.assertTrue(all(start <= d < start + frontier_count for _, d in ids))
                    self.assertTrue(all(sum(d == node for _, d in ids) <= fanout for node in range(start, start + frontier_count)))
                    start += frontier_count
                    edge_start += sample.num_sampled_edges[hop]
        self.assertNotEqual(epochs[0], epochs[1])
        single = Data(x=tensor([[2.], [3.]], True), edge_index=index([[0], [1]]), edge_attr=tensor([[.4]], True))
        repeated = next(iter(NeighborLoader(single, [4], input_nodes=[1], replace=True, seed=2)))
        self.assertEqual(repeated.e_id.tolist(), [0, 0, 0, 0])
        repeated.edge_attr.sum().backward()
        self.assertEqual(single.edge_attr.grad.tolist(), [[4.]])

    def test_induced_extra_edges_and_empty_fanout(self):
        data = Data(edge_index=index([[0, 1, 0], [1, 0, 0]]), num_nodes=3)
        directional = next(iter(NeighborLoader(data, [-1], input_nodes=[1])))
        induced = next(iter(NeighborLoader(data, [-1], input_nodes=[1], subgraph_type='induced')))
        self.assertEqual(directional.e_id.tolist(), [0])
        self.assertEqual(induced.e_id.tolist(), [0, 1, 2])
        empty = next(iter(NeighborLoader(data, [0], input_nodes=[1, 2], batch_size=2)))
        self.assertEqual(empty.n_id.tolist(), [1, 2])
        self.assertEqual(empty.edge_index.shape, [2, 0])
        self.assertEqual(empty.num_sampled_nodes, [2, 0])
        self.assertEqual(len(NeighborLoader(data, [], batch_size=2, drop_last=True)), 1)
        missing_topology = Data(x=tensor([[1.], [2.]]))
        self.assertEqual(next(iter(NeighborLoader(missing_topology, [-1]))).edge_index.shape, [2, 0])
        self.assertEqual(list(LinkNeighborLoader(missing_topology, [-1])), [])

    def test_boolean_input_node_masks_match_index_selection(self):
        data, _, _ = graph()
        mask = fr.tensor([False, True, False, False, True, False, False, False], dtype='bool')
        by_mask = next(iter(NeighborLoader(data, [-1, -1], input_nodes=mask, batch_size=2)))
        by_index = next(iter(NeighborLoader(data, [-1, -1], input_nodes=[1, 4], batch_size=2)))
        self.assertEqual(by_mask.n_id.tolist(), by_index.n_id.tolist())
        self.assertEqual(by_mask.e_id.tolist(), by_index.e_id.tolist())
        self.assertEqual(by_mask.input_id.tolist(), [0, 1])
        self.assertEqual(list(NeighborLoader(data, [-1], input_nodes=fr.tensor([False] * 8))), [])
        with self.assertRaises(ValueError):
            NeighborLoader(data, [-1], input_nodes=fr.tensor([True]))

    def test_link_endpoint_mapping_labels_negative_exclusion_and_gradients(self):
        data, _, base = graph(True)
        supervised = index([[0, 2, 7], [1, 3, 4]])
        loader = LinkNeighborLoader(data, [-1], supervised, index([0, 2, 1]), batch_size=2, neg_sampling_ratio=1., seed=19)
        batches = list(loader)
        known = set(pairs(data.edge_index)) | set(pairs(supervised))
        for batch in batches:
            ids, nodes = batch.input_id.tolist(), batch.n_id.tolist()
            endpoints = [(nodes[s], nodes[d]) for s, d in pairs(batch.edge_label_index)]
            self.assertEqual(endpoints[:len(ids)], [pairs(supervised)[i] for i in ids])
            self.assertTrue(all(p not in known and p[0] != p[1] for p in endpoints[len(ids):]))
            self.assertEqual(batch.edge_label.tolist(), [[1, 3, 2][i] for i in ids] + [0] * (len(endpoints) - len(ids)))
            self.assertEqual(batch.edge_label.dtype, 'int64')
            self.assertTrue(batch.validate())
        loss = sum(batch.x.sum() + batch.edge_attr.sum() for batch in batches)
        loss.backward()
        node_counts = [sum(i in batch.n_id.tolist() for batch in batches) for i in range(8)]
        self.assertEqual(base.grad.tolist(), [[float(n) for n in node_counts]] * 3)
        edge_counts = [sum(batch.e_id.tolist().count(i) for batch in batches) for i in range(data.num_edges)]
        self.assertEqual(data.edge_attr.grad.tolist(), [[float(n)] for n in edge_counts])

    def test_link_supervision_removal_including_reverse_duplicates(self):
        data = Data(edge_index=index([[0, 0, 1, 2, 1], [1, 1, 0, 1, 2]]), num_nodes=3)
        labels = index([[0], [1]])
        default = next(iter(LinkNeighborLoader(data, [-1], labels)))
        self.assertIn(0, default.e_id.tolist())
        excluded = next(iter(LinkNeighborLoader(data, [-1], labels, exclude_seed_edges=True, exclude_reverse_edges=True)))
        self.assertFalse(set(excluded.e_id.tolist()) & {0, 1, 2})
        self.assertNotIn('edge_label', excluded)
        empty = LinkNeighborLoader(data, [-1], index([[], []]))
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty), [])

    def test_disjoint_link_split_flows_into_loader_without_reintroducing_labels(self):
        data = Data(edge_index=index([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 4]]), num_nodes=5)
        train, _, _ = RandomLinkSplit(1, 1, disjoint_train_ratio=1, seed=5)(data)
        loader = LinkNeighborLoader(train, [-1, -1], train.edge_label_index, train.edge_label, batch_size=8)
        batch = next(iter(loader))
        nodes = batch.n_id.tolist()
        supervision = [(nodes[s], nodes[d]) for s, d in pairs(batch.edge_label_index)]
        adjacency = {(nodes[s], nodes[d]) for s, d in pairs(batch.edge_index)}
        self.assertEqual(supervision, pairs(train.edge_label_index))
        self.assertFalse(adjacency & {p for p, y in zip(supervision, batch.edge_label.tolist()) if y > 0})
        self.assertEqual(batch.edge_label.tolist(), train.edge_label.tolist())

    def test_invalid_and_unsupported_options_are_explicit(self):
        data, _, _ = graph()
        for kwargs in ({'input_nodes': [1, 1]}, {'input_nodes': [8]}, {'input_nodes': tensor([])}, {'batch_size': 0}, {'num_workers': 1}, {'subgraph_type': 'bidirectional'}, {'disjoint': True}, {'time_attr': 'time'}):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                NeighborLoader(data, [-1], **kwargs)
        for fanout in ([-2], [True], 2, {'edge': [1]}):
            with self.assertRaises(ValueError):
                NeighborLoader(data, fanout)
        with self.assertRaises(ValueError):
            NeighborLoader(data, [sys.maxsize], replace=True)
        with self.assertRaises(ValueError):
            LinkNeighborLoader(data, [-1], index([[0], [9]]))
        with self.assertRaises(ValueError):
            LinkNeighborLoader(data, [-1], index([[0], [1]]), tensor([1., 2.]))
        with self.assertRaises(ValueError):
            LinkNeighborLoader(data, [-1], exclude_reverse_edges=True)
        with self.assertRaises(NotImplementedError):
            LinkNeighborLoader(data, [-1], neg_sampling='triplet')


if __name__ == '__main__':
    unittest.main()
