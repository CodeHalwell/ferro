"""Actual PyG efficient graph basis/aggregator references."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import EGConv as TorchEG
import ferro as fr
from ferro.graph import EGConv
import test_pyg_modulated as helpers

native = helpers.native


class EfficientTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1877)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def test_aggregator_mixtures_heads_bases_and_all_gradients(self):
        for aggregators, heads, bases, loops, bias in ((None, 2, 3, True, True), (['sum', 'mean', 'symnorm'], 4, 2, False, False),
                                                       (['mean', 'max', 'min', 'var', 'std'], 1, 2, True, True),
                                                       (['std'], 2, 1, False, True), (['mean', 'mean'], 4, 3, False, True)):
            model = EGConv(3, 8, aggregators, num_heads=heads, num_bases=bases, add_self_loops=loops, bias=bias)
            oracle = TorchEG(3, 8, aggregators, num_heads=heads, num_bases=bases, add_self_loops=loops, bias=bias)
            self.copy(model, oracle)
            base = torch.randn(3, 6, requires_grad=True)
            nb = native(base, True)
            self.gradients(model, oracle, model(nb.transpose(0, 1), native(self.edge)), oracle(base.transpose(0, 1), self.edge), [('x', nb, base)])

    def test_topology_inferred_loops_and_std_floor(self):
        for aggr in (['mean'], ['symnorm'], ['std'], []):
            model, oracle = EGConv(3, 4, aggr, num_heads=2), TorchEG(3, 4, aggr, num_heads=2)
            self.copy(model, oracle)
            x = torch.randn(7, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, native(self.edge)), oracle(x, self.edge), [('x', nx, x)])
            if aggr == ['mean']:
                self.close(model(nx, native(self.edge)).narrow(0, 5, 2), oracle.bias.expand(2, 4))
            if aggr == ['std']:
                self.assertTrue(any(value != 0. for row in (model(nx, native(self.edge)) - model.bias.tensor()).narrow(0, 5, 2).tolist() for value in row))

    def test_empty_topology_nodes_native_edges_and_permutation(self):
        empty = torch.empty(2, 0, dtype=torch.int64)
        for nodes, aggregators, loops in ((0, ['symnorm'], True), (0, ['std', 'mean'], True),
                                        (5, ['symnorm'], False), (5, ['mean', 'std'], True)):
            model, oracle = EGConv(3, 4, aggregators, num_heads=2, add_self_loops=loops), TorchEG(3, 4, aggregators, num_heads=2, add_self_loops=loops)
            self.copy(model, oracle)
            x = torch.randn(nodes, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, fr.graph.EdgeIndex(nodes, [], [])), oracle(x, empty), [('x', nx, x)])
        model, oracle = EGConv(3, 8, ['mean', 'symnorm'], num_heads=2), TorchEG(3, 8, ['mean', 'symnorm'], num_heads=2)
        self.copy(model, oracle)
        x = torch.randn(12, 3, requires_grad=True)
        nx = native(x, True)
        edge = torch.cat([self.edge, self.edge + 6], 1)
        edge = edge[:, torch.randperm(edge.shape[1])]
        self.gradients(model, oracle, model(nx, native(edge)), oracle(x, edge), [('x', nx, x)])

    def test_exact_saved_optimizer_configuration_and_generator_continuation(self):
        def factory(aggregators=('mean', 'symnorm')): return EGConv(3, 8, aggregators, num_heads=2, num_bases=3)
        model, generator = factory(), fr.Generator(223)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        edge = native(self.edge)
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([6, 3], generator=generator), edge)
            loss = output.square().mean()
            result = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return result
        step(model, optimizer, generator)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'eg': optimizer}, generator=generator, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, generator) for _ in range(3)]
        other, rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.9)
        saved.restore(other, optimizers={'eg': other_opt}, generator=rng)
        self.assertEqual(expected, [step(other, other_opt, rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'eg': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'eg': other_opt}, generator=rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left: self.assertEqual(left[name].tolist(), right[name].tolist(), name)
        changed = factory(('sum', 'symnorm'))
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'eg': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_initialization_reset_config_capacity_and_eager_boundaries(self):
        model = EGConv(3, 8, ['mean', 'max'], num_heads=2, num_bases=3)
        self.assertEqual(model.bias.tensor().tolist(), [0.] * 8)
        self.assertTrue(any(value != 0. for value in model.comb_lin.bias.tensor().tolist()))
        self.assertTrue(all(abs(value) <= 3 ** -.5 for value in model.comb_lin.bias.tensor().tolist()))
        with self.assertRaises(AttributeError): model.aggregators = ('sum',)
        for constructor in (lambda: EGConv(3, 7), lambda: EGConv(3, 4, num_heads=0), lambda: EGConv(-1, 8),
                            lambda: EGConv(3, 8, ['bad']), lambda: EGConv(3, 8, 'sum')):
            with self.assertRaises((TypeError, ValueError)): constructor()
        with self.assertRaises(NotImplementedError): EGConv(3, 8, cached=True)
        with patch('ferro._graph_efficient.Linear', side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'): EGConv(1, 8, num_bases=sys.maxsize)
        empty = fr.Tensor.from_i64([], [2, 0])
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8, 1)
        with self.assertRaisesRegex(ValueError, 'capacity'): EGConv(1, 1, num_heads=1, num_bases=1)(huge, empty)
        wide = EGConv(1, 16, ['mean'], num_heads=4, num_bases=1, add_self_loops=False)
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 64 + 1, 1)
        with patch.object(wide.bases_lin, 'forward', side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'): wide(huge, empty)
        x = fr.Tensor.ones([6, 3])
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, native(self.edge)))
        with self.assertRaises(ValueError): model(x, fr.Tensor.from_i64([0, 6], [2, 1]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
