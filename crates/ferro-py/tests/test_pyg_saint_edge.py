"""Actual PyG 2.8.1 edge-sampling masses and induced-subgraph methods.

Uses the explicitly disclosed sparse primitive adapter from test_pyg_saint;
does not execute or validate an optional torch_sparse extension.
"""
import itertools
import random
import sys
import unittest
from collections import Counter
from unittest.mock import patch

import torch
from torch_geometric.loader import GraphSAINTEdgeSampler as TorchEdge

import ferro as fr
from ferro._graph_data import Data
from ferro._graph_saint import GraphSAINTEdgeSampler as Edge
import test_pyg_saint as _saint
from test_pyg_saint import SparseAdapter, fixture, fi, ft, oracle


class DegreeAdapter(SparseAdapter):
    def __init__(self, row, col, value, nodes):
        super().__init__(row, col, value)
        self.nodes = nodes

    def colcount(self):
        return torch.bincount(self.c, minlength=self.nodes)

    def rowcount(self):
        return torch.bincount(self.r, minlength=self.nodes)


def reference(data, sampler):
    result = TorchEdge.__new__(TorchEdge)
    result.__dict__.update(oracle(data, sampler).__dict__)
    result.adj = DegreeAdapter(data.edge_index[0], data.edge_index[1], torch.arange(data.num_edges), data.num_nodes)
    return result


class EdgeSAINTTests(unittest.TestCase):
    close = _saint.GraphSAINTTests.close
    compare = _saint.GraphSAINTTests.compare

    def test_actual_pyg_probability_mass_and_equivalent_categorical_cells(self):
        data, expected = fixture()
        sampler = Edge(data, 4, seed=2)
        wanted = reference(expected, sampler)
        uniform = torch.tensor([[.2 + ((i + j * 3) % 7) / 10 for i in range(data.num_edges)] for j in range(4)])
        observed = []
        class Logged:
            def __truediv__(self, denominator):
                observed.append(denominator.detach().clone())
                return uniform.log() / denominator
        class Draw:
            def log(self): return Logged()
        with patch('torch.rand', return_value=Draw()) as rand:
            endpoints = wanted._sample_nodes(4).reshape(-1).tolist()
        self.assertEqual(rand.call_args.args, (4, data.num_edges))
        masses = [right - left for left, right in zip([0.] + sampler._edge_cdf, sampler._edge_cdf)]
        torch.testing.assert_close(torch.tensor(masses), observed[0])
        selected = (uniform.log() / observed[0]).argmax(1).tolist()
        quantiles = [(sampler._edge_cdf[i] + (sampler._edge_cdf[i - 1] if i else 0.)) / (2 * sampler._edge_cdf[-1]) for i in selected]
        class Fixed:
            def __init__(self): self.values = iter(quantiles)
            def random(self): return next(self.values)
        # Different RNG transforms, identical selected categorical cells.
        self.assertEqual(sampler._sample_nodes(Fixed()), endpoints)

    def test_weighted_distribution_duplicates_and_zero_degree_floor(self):
        src, dst = [0, 0, 1, 2, 3, 4, 4, 6], [1, 1, 2, 1, 4, 3, 4, 7]
        data = Data(edge_index=fi([src, dst]), num_nodes=8)
        sampler = Edge(data, 1, seed=9)
        masses = [b - a for a, b in zip([0.] + sampler._edge_cdf, sampler._edge_cdf)]
        self.assertGreater(masses[-1], 0.)
        self.assertLess(masses[-1], 2e-10)
        rng, count = random.Random(9), Counter()
        for _ in range(30000):
            target, source = sampler._sample_nodes(rng)
            count[source, target] += 1
        expected = Counter()
        for edge, mass in zip(sampler._edge_order, masses):
            expected[src[edge], dst[edge]] += mass / sampler._edge_cdf[-1]
        for pair, probability in expected.items():
            self.assertAlmostEqual(count[pair] / 30000, probability, delta=.008)
        batch = next(iter(Edge(data, 200, seed=9)))
        self.assertIn(0, batch.e_id.tolist())
        self.assertIn(1, batch.e_id.tolist())
        self.assertNotIn(5, batch.n_id.tolist())

    def test_actual_pyg_induced_fields_and_live_node_edge_graph_gradients(self):
        data, expected = fixture(True)
        sampler = Edge(data, 3, num_steps=4, sample_coverage=2, seed=19)
        wanted = reference(expected, sampler)
        loss, ref_loss = ft(0.), torch.tensor(0.)
        for batch in sampler:
            ref = self.compare(batch, expected, wanted)
            loss = loss + (batch.x * batch.x * batch.node_norm.unsqueeze(1)).sum() + (batch.edge_attr * batch.edge_attr * batch.edge_norm.unsqueeze(1)).sum() + batch.graph_value.sum()
            ref_loss = ref_loss + (ref.x.square() * ref.node_norm[:, None]).sum() + (ref.edge_attr.square() * ref.edge_norm[:, None]).sum() + ref.graph_value.sum()
        self.close(loss, ref_loss)
        loss.backward(); ref_loss.backward()
        for key in ('x', 'edge_attr', 'graph_value'):
            self.close(data[key].grad, expected[key].grad)

    def test_calibration_matches_actual_pyg_and_does_not_consume_epochs(self):
        data, expected = fixture()
        draws = [[0, 1], [2, 3], [5], [1, 2, 3]]
        class Controlled(Edge):
            def _sample_nodes(self, rng):
                if not hasattr(self, 'draws'): self.draws = itertools.cycle(draws)
                return next(self.draws)
        sampler = Controlled(data, 2, num_steps=3, sample_coverage=3, seed=11)
        wanted = reference(expected, sampler)
        sequence = itertools.cycle(draws)
        with patch.object(TorchEdge, '_sample_nodes', side_effect=lambda count: torch.tensor(next(sequence))):
            node_norm, edge_norm = wanted._compute_norm()
        self.close(sampler.node_norm, node_norm)
        self.close(sampler.edge_norm, edge_norm)
        a, b = Edge(data, 2, num_steps=3, sample_coverage=2, seed=11), Edge(data, 2, num_steps=3, seed=11)
        for _ in range(2):
            self.assertEqual([(x.n_id.tolist(), x.e_id.tolist()) for x in a], [(x.n_id.tolist(), x.e_id.tolist()) for x in b])

    def test_empty_self_loop_and_capacity_contracts(self):
        with self.assertRaises(ValueError): Edge(Data(edge_index=fi([[], []]), num_nodes=2), 1)
        data = Data(edge_index=fi([[0, 0], [0, 0]]), num_nodes=1)
        batch = next(iter(Edge(data, 3, sample_coverage=1, seed=4)))
        self.assertEqual(batch.n_id.tolist(), [0])
        self.assertEqual(batch.e_id.tolist(), [0, 1])
        self.assertEqual(batch.node_norm.tolist(), [1.])
        self.assertEqual(batch.edge_norm.tolist(), [1., 1.])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            Edge(data, sys.maxsize // 16 + 1)


if __name__ == '__main__':
    unittest.main()
