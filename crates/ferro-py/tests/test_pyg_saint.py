"""GraphSAINT parity without claiming optional torch_sparse execution.

Actual PyG _sample_nodes, _collate and _compute_norm methods run against a tiny
COO adapter for its sparse storage/induced-subgraph primitives. Random walks
also have an independent seeded structural oracle; random streams differ.
"""
import itertools
import random
import tracemalloc
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData
from torch_geometric.loader import GraphSAINTNodeSampler as TorchNode

import ferro as fr
from ferro._graph_data import Data
from ferro._graph_saint import GraphSAINTNodeSampler as Node, GraphSAINTRandomWalkSampler as Walk


def ft(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def fi(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


class SparseAdapter:
    """Only the sparse primitives used by PyG GraphSAINT's Python methods."""
    def __init__(self, row, col, value):
        order = sorted(range(len(row)), key=lambda i: (int(row[i]), int(col[i]), i))
        self.r, self.c, self.v = (torch.as_tensor(x, dtype=torch.int64)[order] for x in (row, col, value))
        self.storage = self

    def row(self):
        return self.r

    def value(self):
        return self.v

    def coo(self):
        return self.r, self.c, self.v

    def saint_subgraph(self, nodes):
        mapping = {int(node): i for i, node in enumerate(nodes)}
        ids = [i for i, (s, d) in enumerate(zip(self.r.tolist(), self.c.tolist())) if s in mapping and d in mapping]
        return SparseAdapter([mapping[int(self.r[i])] for i in ids],
                             [mapping[int(self.c[i])] for i in ids], self.v[ids]), None


def fixture(grad=False):
    raw = torch.arange(21).float().reshape(3, 7) / 13 - .4
    edge = torch.tensor([[3, 0, 2, 0, 5, 1, 3, 2, 1, 0, 4],
                         [1, 1, 3, 1, 5, 2, 0, 2, 0, 2, 3]])
    attrs = torch.arange(22).float().reshape(11, 2) / 19 - .2
    roles = dict(labels='node', mask='node', graph_value='graph')
    data = Data(x=ft(raw, grad).transpose(0, 1), edge_index=fi(edge), edge_attr=ft(attrs, grad),
                labels=fi([2**55 + i for i in range(7)]), mask=fr.tensor([True, False, True, False, True, False, True]),
                graph_value=ft([.7, -.2], grad), num_nodes=7, field_roles=roles)
    ref = TorchData(x=raw.T.detach().requires_grad_(grad), edge_index=edge, edge_attr=attrs.clone().requires_grad_(grad),
                    labels=torch.tensor([2**55 + i for i in range(7)]), mask=torch.tensor([True, False, True, False, True, False, True]),
                    graph_value=torch.tensor([.7, -.2], requires_grad=grad), num_nodes=7)
    return data, ref


def oracle(data, sampler):
    result = TorchNode.__new__(TorchNode)
    result.N, result.E, result.data = data.num_nodes, data.num_edges, data
    result.adj = SparseAdapter(data.edge_index[0], data.edge_index[1], torch.arange(data.num_edges))
    result.sample_coverage = sampler.sample_coverage
    result.num_steps, result.num_workers, result.log = sampler.num_steps, 0, False
    result._batch_size, result.batch_size = sampler.batch_size, 1
    if sampler.sample_coverage:
        result.node_norm = torch.tensor(sampler.node_norm.tolist())
        result.edge_norm = torch.tensor(sampler.edge_norm.tolist())
    return result


class GraphSAINTTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=getattr(torch, actual.dtype)).reshape(actual.shape),
                                   expected.detach(), atol=3e-6, rtol=3e-5)

    def compare(self, batch, expected, reference):
        nodes = torch.tensor(batch.n_id.tolist())
        subgraph, _ = reference.adj.saint_subgraph(nodes)
        want = reference._collate([(nodes, subgraph)])
        self.assertTrue(batch.validate())
        for name in ('x', 'edge_index', 'edge_attr', 'labels', 'mask', 'graph_value'):
            self.close(batch[name], want[name])
        self.close(batch.e_id, subgraph.v)
        if reference.sample_coverage:
            self.close(batch.node_norm, want.node_norm)
            self.close(batch.edge_norm, want.edge_norm)
        else:
            self.assertNotIn('node_norm', batch)
            self.assertNotIn('edge_norm', batch)
        return want

    def test_actual_pyg_collation_and_all_field_gradients(self):
        for factory, options in ((Node, {}), (Walk, dict(walk_length=3))):
            data, expected = fixture(True)
            sampler = factory(data, 5, num_steps=4, sample_coverage=3, seed=17, **options)
            reference = oracle(expected, sampler)
            loss, ref_loss = ft(0.), torch.tensor(0.)
            for step, batch in enumerate(sampler):
                want = self.compare(batch, expected, reference)
                loss = loss + (batch.x * batch.x * batch.node_norm.unsqueeze(1)).sum() + (batch.edge_attr * batch.edge_attr * batch.edge_norm.unsqueeze(1)).sum() + batch.graph_value.sum() * (step + 1)
                ref_loss = ref_loss + (want.x.square() * want.node_norm[:, None]).sum() + (want.edge_attr.square() * want.edge_norm[:, None]).sum() + want.graph_value.sum() * (step + 1)
            self.close(loss, ref_loss)
            loss.backward()
            ref_loss.backward()
            for name in ('x', 'edge_attr', 'graph_value'):
                self.close(data[name].grad, expected[name].grad)
            self.assertEqual(batch.labels.dtype, 'int64')
            self.assertEqual(batch.mask.dtype, 'bool')

    def test_empirical_normalization_executes_actual_pyg_method(self):
        for coverage in (1, 4):
            data, expected = fixture()
            draws = [[0, 1], [2, 3, 3], [0], [5]]
            class Controlled(Node):
                def _sample_nodes(self, rng):
                    if not hasattr(self, 'draws'):
                        self.draws = itertools.cycle(draws)
                    return next(self.draws)
            sampler = Controlled(data, 2, num_steps=3, sample_coverage=coverage, seed=2)
            reference = oracle(expected, sampler)
            sequence = itertools.cycle(draws)
            with patch.object(TorchNode, '_sample_nodes', side_effect=lambda count: torch.tensor(next(sequence))):
                node_norm, edge_norm = reference._compute_norm()
            self.close(sampler.node_norm, node_norm)
            self.close(sampler.edge_norm, edge_norm)
            self.assertEqual(sampler.normalization_samples % 3, 0)
            self.assertGreater(sampler.node_norm.tolist()[6], 1.)
            # Source observed, endpoint never co-sampled: finite clamp, not inf.
            self.assertIn(1e4, sampler.edge_norm.tolist())
            # Never-observed source and edge follows PyG's 0/0 -> .1 rule.
            self.assertAlmostEqual(sampler.edge_norm.tolist()[10], .1, places=6)

    def test_degree_weighted_node_draw_matches_actual_pyg_source(self):
        data, expected = fixture()
        sampler = Node(data, 5, seed=1)
        reference = oracle(expected, sampler)
        samples = [0, 1, 2, 6, 10]
        class Fixed:
            def __init__(self): self.values = iter(samples)
            def randrange(self, maximum):
                self.maximum = maximum
                return next(self.values)
        rng = Fixed()
        with patch('torch.randint', return_value=torch.tensor(samples).reshape(5, 1)) as call:
            want = reference._sample_nodes(5).reshape(-1).tolist()
        self.assertEqual(sampler._sample_nodes(rng), want)
        self.assertEqual(rng.maximum, data.num_edges)
        self.assertEqual(call.call_args.args[:2], (0, data.num_edges))
        nodes = sampler._sample_nodes(random.Random(3))
        self.assertNotIn(6, nodes)

    def test_seeded_walk_manual_oracle_outgoing_duplicates_and_sinks(self):
        data, _ = fixture()
        sampler = Walk(data, 5, 4, num_steps=3, seed=29)
        rng = random.Random(29)
        outgoing = {}
        src, dst = data.edge_index.tolist()
        for source, target in sorted(zip(src, dst)):
            outgoing.setdefault(source, []).append(target)
        for batch in sampler:
            roots = [rng.randrange(7) for _ in range(5)]
            visited = []
            for root in roots:
                visited.append(root)
                for _ in range(4):
                    targets = outgoing.get(root, [])
                    root = targets[rng.randrange(len(targets))] if targets else root
                    visited.append(root)
            self.assertEqual(batch.n_id.tolist(), sorted(set(visited)))
            selected = set(visited)
            ids = sorted([i for i, (s, d) in enumerate(zip(src, dst)) if s in selected and d in selected],
                         key=lambda i: (src[i], dst[i], i))
            self.assertEqual(batch.e_id.tolist(), ids)
            self.assertEqual((1 in ids), (3 in ids))  # Parallel original edges.
        # Explicit walk trace selects each duplicated outgoing row independently.
        tiny = Data(edge_index=fi([[0, 0, 0], [1, 1, 2]]), num_nodes=3)
        walk = Walk(tiny, 1, 1, seed=0)
        class Choice:
            def __init__(self, edge): self.values = iter([0, edge])
            def randrange(self, maximum): return next(self.values)
        self.assertEqual([walk._sample_nodes(Choice(i)) for i in range(3)], [[0, 1], [0, 1], [0, 2]])

    def test_epochs_calibration_and_lazy_iteration_are_deterministic(self):
        data, _ = fixture()
        a = Walk(data, 3, 2, num_steps=3, seed=91, sample_coverage=2)
        b = Walk(data, 3, 2, num_steps=3, seed=91, sample_coverage=0)
        def ids(loader): return [(x.n_id.tolist(), x.e_id.tolist()) for x in loader]
        self.assertEqual(a.epoch, 0)
        self.assertEqual(ids(a), ids(b))
        first = iter(a)
        self.assertEqual(a.epoch, 2)
        self.assertEqual([(x.n_id.tolist(), x.e_id.tolist()) for x in first], ids(b))
        self.assertEqual(len(a), 3)
        twin = Walk(data, 3, 2, num_steps=3, seed=91, sample_coverage=2)
        self.assertEqual(twin.node_norm.tolist(), a.node_norm.tolist())
        self.assertEqual(twin.edge_norm.tolist(), a.edge_norm.tolist())

    def test_edgeless_walks_isolates_zero_width_and_live_empty_edges(self):
        data = Data(x=ft(torch.empty(4, 0), True), edge_index=fi(torch.empty(2, 0, dtype=torch.int64)),
                    edge_attr=ft(torch.empty(0, 2), True), num_nodes=4)
        for length in (0, 3):
            sampler = Walk(data, 8, length, num_steps=2, sample_coverage=1, seed=3)
            loss = ft(0.)
            for batch in sampler:
                self.assertTrue(batch.validate())
                self.assertEqual(batch.edge_index.shape, [2, 0])
                self.assertEqual(batch.edge_norm.shape, [0])
                loss = loss + batch.x.sum() + batch.edge_attr.sum()
            loss.backward()
            self.assertEqual(data.x.grad.shape, [4, 0])
            self.assertEqual(data.edge_attr.grad.shape, [0, 2])
        with self.assertRaises(ValueError): Node(data, 1)

    def test_validation_and_capacity_reject_before_sampling(self):
        data, _ = fixture()
        for options in (dict(batch_size=0), dict(batch_size=True), dict(batch_size=2**62),
                        dict(batch_size=1, num_steps=0), dict(batch_size=1, sample_coverage=-1),
                        dict(batch_size=1, sample_coverage=1.5), dict(batch_size=1, seed=True),
                        dict(batch_size=1, num_workers=1), dict(batch_size=1, save_dir='cache'),
                        dict(batch_size=1, unknown=True), dict(batch_size=1, log=1)):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)): Node(data, **options)
        with self.assertRaises(ValueError): Walk(data, 2, 2**62)
        with self.assertRaises(ValueError): Walk(Data(edge_index=fi([[], []]), num_nodes=0), 1, 0)
        data['n_id'] = fi(range(7))
        with self.assertRaises(ValueError): Node(data, 1)
        data, _ = fixture()
        data['edge_label_index'] = fi([[0], [1]])
        with self.assertRaises(NotImplementedError): Node(data, 1)

    def test_topology_snapshot_and_graph_scalar_reference(self):
        data, _ = fixture()
        data.scalar = ft(.4, True)
        data._field_roles['scalar'] = 'graph'
        sampler = Node(data, 6, seed=12)
        data.edge_index = fi([[6], [6]])
        data.edge_attr = ft([[99., 99.]])
        batch = next(iter(sampler))
        self.assertNotIn(6, batch.n_id.tolist())
        self.assertIs(batch.scalar, data.scalar)
        self.assertEqual(batch.edge_attr.shape[1], 2)
        batch.scalar.backward()
        self.assertEqual(data.scalar.grad.item(), 1.)

    def test_large_sparse_graph_keeps_python_storage_bounded(self):
        count = 8000
        source = list(range(count))
        target = [(node + 1) % count for node in source]
        data = Data(edge_index=fi([source, target]), num_nodes=count)
        tracemalloc.start()
        try:
            sampler = Walk(data, 4, 3, num_steps=2, seed=7)
            for batch in sampler:
                self.assertLessEqual(batch.num_nodes, 16)
                self.assertLessEqual(batch.num_edges, 16)
                self.assertTrue(batch.validate())
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Measures Python adjacency/induction storage, not native allocations.
        # An N*N Python adjacency alone would exceed this by a wide margin.
        self.assertLess(peak, 16 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
