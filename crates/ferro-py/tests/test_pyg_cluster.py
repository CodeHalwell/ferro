"""Actual PyG cluster collation with identical supplied partitions.

The reference _metis seam supplies assignments; no optional METIS backend is
executed or counted as validated. Ferro's balanced BFS is tested independently.
"""
import random
import sys
import tracemalloc
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData
from torch_geometric.loader import ClusterData as TorchClusterData, ClusterLoader as TorchClusterLoader

import ferro as fr
from ferro._graph_cluster import ClusterData, ClusterLoader
from ferro._graph_data import Data


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def fixture(grad=False):
    raw = torch.arange(18).float().reshape(2, 9) / 11 - .4
    edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 3, 0],
                          [1, 0, 0, 3, 2, 5, 6, 4, 7, 6, 4, 0]])
    attrs = torch.arange(24).float().reshape(12, 2) / 13 - .2
    tags = [2**55 + i for i in range(9)]
    mask = [i % 2 == 0 for i in range(9)]
    roles = {'n_id': 'node', 'e_id': 'edge', 'labels': 'node', 'mask': 'node', 'global_value': 'graph'}
    actual = Data(x=tensor(raw, grad).transpose(0, 1), edge_index=index(edges), edge_attr=tensor(attrs, grad),
                  n_id=index(range(9)), e_id=index(range(12)), labels=index(tags), mask=fr.tensor(mask, dtype='bool'),
                  global_value=tensor(.7, grad), num_nodes=9, field_roles=roles)
    expected = TorchData(x=raw.T.detach().requires_grad_(grad), edge_index=edges,
                         edge_attr=attrs.clone().requires_grad_(grad), n_id=torch.arange(9), e_id=torch.arange(12),
                         labels=torch.tensor(tags), mask=torch.tensor(mask), global_value=torch.tensor(.7, requires_grad=grad), num_nodes=9)
    return actual, expected


def oracle(data, assignments, parts, fmt='csr'):
    with patch.object(TorchClusterData, '_metis', return_value=torch.tensor(assignments, dtype=torch.int64)) as metis:
        result = TorchClusterData(data, parts, sparse_format=fmt, log=False)
    metis.assert_called_once()
    return result


class ClusterTests(unittest.TestCase):
    def close(self, actual, expected):
        dtype = getattr(torch, actual.dtype)
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=dtype).reshape(actual.shape), expected.detach(), atol=3e-6, rtol=3e-5)

    def same(self, actual, expected):
        self.assertTrue(actual.validate())
        self.assertEqual(actual.num_nodes, expected.num_nodes)
        for name in ('x', 'edge_index', 'edge_attr', 'n_id', 'e_id', 'labels', 'mask', 'global_value'):
            self.close(actual[name], expected[name])
        self.assertEqual(actual.labels.dtype, 'int64')
        self.assertEqual(actual.mask.dtype, 'bool')

    def test_partition_items_and_metadata_match_actual_pyg_csr_csc(self):
        assignments = [1, 1, 0, 0, 2, 2, 1, 0, 3]
        for fmt in ('csr', 'csc'):
            with self.subTest(format=fmt):
                actual, expected = fixture()
                clusters = ClusterData(actual, 5, partition=index(assignments), sparse_format=fmt)
                reference = oracle(expected, assignments, 5, fmt)
                for key in ('indptr', 'index', 'partptr', 'node_perm', 'edge_perm'):
                    self.close(getattr(clusters.partition, key), getattr(reference.partition, key))
                for part in range(5):
                    self.same(clusters[part], reference[part])
                self.assertEqual(clusters[-1].num_nodes, 0)
                self.assertEqual(clusters[3].num_edges, 0)

    def test_union_retains_cross_part_edges_and_duplicate_identity(self):
        assignments = [0, 1, 0, 1, 2, 2, 1, 0, 3]
        for fmt in ('csr', 'csc'):
            actual, expected = fixture()
            clusters = ClusterData(actual, 4, partition=assignments, sparse_format=fmt)
            reference = oracle(expected, assignments, 4, fmt)
            loader, reference_loader = ClusterLoader(clusters, batch_size=2), TorchClusterLoader(reference, batch_size=2)
            batches = list(loader)
            for batch, wanted in zip(batches, reference_loader):
                self.same(batch, wanted)
            self.assertGreater(batches[0].num_edges, clusters[0].num_edges + clusters[1].num_edges)
            self.assertIn(1, batches[0].e_id.tolist())
            self.assertIn(2, batches[0].e_id.tolist())
            reordered = clusters._collate([2, 0])
            self.same(reordered, reference_loader._collate([2, 0]))

    def test_feature_edge_and_graph_field_gradients_match_pyg(self):
        assignments = [0, 1, 0, 1, 2, 2, 1, 0, 3]
        for fmt in ('csr', 'csc'):
            actual, expected = fixture(True)
            clusters = ClusterData(actual, 4, partition=assignments, sparse_format=fmt)
            reference = oracle(expected, assignments, 4, fmt)
            loss, ref_loss = tensor(0.), torch.tensor(0.)
            for i, (batch, wanted) in enumerate(zip(ClusterLoader(clusters, 2), TorchClusterLoader(reference, batch_size=2))):
                self.same(batch, wanted)
                loss = loss + (batch.x * batch.x).sum() * (i + 1) + (batch.edge_attr * batch.edge_attr).sum() + batch.global_value * (i + 2)
                ref_loss = ref_loss + wanted.x.square().sum() * (i + 1) + wanted.edge_attr.square().sum() + wanted.global_value * (i + 2)
            loss.backward()
            ref_loss.backward()
            for field in ('x', 'edge_attr', 'global_value'):
                self.close(actual[field].grad, expected[field].grad)

    def test_all_parts_message_passing_matches_full_graph_and_gradients(self):
        actual, expected = fixture(True)
        clusters = ClusterData(actual, 4)
        batch = next(iter(ClusterLoader(clusters, 4)))
        src, dst = batch.edge_index.tolist()
        messages = batch.x.index_select(0, src) * batch.edge_attr
        result = fr.graph.scatter(messages, dst, batch.num_nodes, 'sum')
        ref_messages = expected.x[expected.edge_index[0]] * expected.edge_attr
        reference = torch.zeros_like(expected.x).index_add(0, expected.edge_index[1], ref_messages)
        wanted = reference[torch.tensor(batch.n_id.tolist())]
        self.close(result, wanted)
        (result * result).sum().backward()
        wanted.square().sum().backward()
        self.close(actual.x.grad, expected.x.grad)
        self.close(actual.edge_attr.grad, expected.edge_attr.grad)

    def test_balanced_bfs_is_deterministic_and_topology_order_independent(self):
        actual, _ = fixture()
        first = ClusterData(actual, 4)
        reversed_data = Data(num_nodes=9, edge_index=index([list(reversed(row)) for row in actual.edge_index.tolist()]))
        second = ClusterData(reversed_data, 4)
        self.assertEqual(first.partition.node_perm.tolist(), second.partition.node_perm.tolist())
        ptr = first.partition.partptr.tolist()
        self.assertEqual([b - a for a, b in zip(ptr, ptr[1:])], [3, 2, 2, 2])
        self.assertEqual(sorted(first.partition.node_perm.tolist()), list(range(9)))
        isolated = ClusterData(Data(num_nodes=7, edge_index=index([[], []])), 3)
        self.assertEqual(isolated.partition.node_perm.tolist(), list(range(7)))
        self.assertEqual(isolated.partition.partptr.tolist(), [0, 3, 5, 7])

    def test_exposed_partition_metadata_does_not_redirect_selections(self):
        actual, _ = fixture()
        clusters = ClusterData(actual, 3)
        expected = clusters._collate([0, 1])
        clusters.partition = replace(clusters.partition, node_perm=index(list(reversed(range(9)))),
                                     edge_perm=index([0] * 12), partptr=index([0, 0, 0, 9]))
        result = clusters._collate([0, 1])
        self.assertEqual(result.to_dict().keys(), expected.to_dict().keys())
        for field in result.keys():
            self.assertEqual(result[field].tolist(), expected[field].tolist())

    def test_empty_graph_partitions_typed_fields_and_empty_gradients(self):
        empty = Data(x=tensor(torch.empty(0, 2), True), edge_index=index([[], []]), edge_attr=tensor(torch.empty(0, 1), True), num_nodes=0)
        clusters = ClusterData(empty, 3)
        self.assertEqual(clusters.partition.indptr.tolist(), [0])
        self.assertEqual(clusters.partition.partptr.tolist(), [0, 0, 0, 0])
        batches = list(ClusterLoader(clusters, 2))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].edge_index.shape, [2, 0])
        self.assertTrue(batches[0].validate())
        (batches[0].x.sum() + batches[0].edge_attr.sum()).backward()
        self.assertEqual(empty.x.grad.shape, [0, 2])
        self.assertEqual(empty.edge_attr.grad.shape, [0, 1])

    def test_shuffle_seed_epoch_drop_last_and_lazy_selection(self):
        actual, _ = fixture()
        clusters = ClusterData(actual, 9, partition=list(range(9)))
        loader = ClusterLoader(clusters, 2, shuffle=True, seed=53)
        twin = ClusterLoader(clusters, 2, shuffle=True, seed=53)
        for epoch in range(2):
            order = list(range(9))
            random.Random(53 + epoch).shuffle(order)
            batches = list(loader)
            self.assertEqual([node for batch in batches for node in batch.n_id.tolist()], order)
            self.assertEqual([b.n_id.tolist() for b in batches], [b.n_id.tolist() for b in twin])
        self.assertEqual(loader.epoch, 2)
        self.assertEqual(len(ClusterLoader(clusters, 2, drop_last=True)), 4)
        self.assertEqual(len(list(ClusterLoader(clusters, 2, drop_last=True))), 4)
        with patch.object(clusters, '_collate', wraps=clusters._collate) as collate:
            iterator = iter(ClusterLoader(clusters, 2))
            self.assertEqual(collate.call_count, 0)
            next(iterator)
            self.assertEqual(collate.call_count, 1)

    def test_validation_and_capacity_precede_partition_allocation(self):
        actual, _ = fixture()
        for kwargs in ({'num_parts': 0}, {'num_parts': True}, {'num_parts': sys.maxsize},
                       {'num_parts': 3, 'partition': [0]}, {'num_parts': 3, 'partition': [3] * 9},
                       {'num_parts': 3, 'partition': tensor([0] * 9)}, {'num_parts': 3, 'partition': [False] * 9},
                       {'num_parts': 3, 'sparse_format': 'coo'}, {'num_parts': 3, 'log': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                ClusterData(actual, **kwargs)
        for kwargs in ({'recursive': True}, {'method': 'metis'}, {'save_dir': '/tmp/partition'}, {'filename': 'cache'}, {'keep_inter_cluster_edges': True}):
            with self.assertRaises(NotImplementedError):
                ClusterData(actual, 3, **kwargs)
        with self.assertRaises(ValueError):
            ClusterData(Data(num_nodes=sys.maxsize, edge_index=index([[], []])), 2)
        with self.assertRaises(ValueError):
            ClusterData(Data(num_nodes=3), 2)
        clusters = ClusterData(actual, 3)
        for invalid in (3, -4):
            with self.assertRaises(IndexError):
                clusters[invalid]
        with self.assertRaises(TypeError):
            clusters[True]
        for kwargs in ({'batch_size': 0}, {'shuffle': 1}, {'drop_last': 1}, {'seed': False}, {'num_workers': 1}):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                ClusterLoader(clusters, **kwargs)

    def test_sparse_partitioning_allocation_is_linear_not_node_squared(self):
        n = 5000
        data = Data(num_nodes=n, edge_index=index([list(range(n)), list(range(1, n)) + [0]]))
        tracemalloc.start()
        try:
            clusters = ClusterData(data, 20)
            batch = next(iter(ClusterLoader(clusters, 2)))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(batch.num_nodes, 500)
        self.assertLessEqual(batch.num_edges, 500)
        self.assertLess(peak, 8_000_000)
        self.assertEqual(clusters.partition.edge_perm.shape, [n])


if __name__ == '__main__':
    unittest.main()
