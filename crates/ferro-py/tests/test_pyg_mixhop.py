"""Actual PyG mixed adjacency powers, weighted gradients and native state."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import MixHopConv as TorchMix
import ferro as fr
from ferro.graph import MixHopConv
import test_pyg_modulated as helpers

native = helpers.native


class MixHopTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2467)
        self.edge = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 4], [1, 2, 0, 4, 3, 2, 2, 4]])

    def test_powers_order_duplicates_strides_weights_and_all_parameter_gradients(self):
        for powers, loops, bias, weighted in ((None, True, True, True), ([0], True, False, True),
                                             ([1, 3], False, True, True), ([2, 0, 2], True, True, False)):
            model, oracle = MixHopConv(3, 4, powers, loops, bias), TorchMix(3, 4, powers, loops, bias)
            self.copy(model, oracle)
            x = torch.randn(3, 6, requires_grad=True); nx = native(x, True)
            weights = (.4 + torch.rand(8)).requires_grad_() if weighted else None
            nw = None if weights is None else native(weights, True)
            self.gradients(model, oracle, model(nx.transpose(0, 1), native(self.edge), nw), oracle(x.transpose(0, 1), self.edge, weights), [('x', nx, x)] + ([] if weights is None else [('edge_weight', nw, weights)]))
            if powers == [0]: self.assertIsNone(nw.grad)

    def test_empty_graphs_missing_edges_native_topology_and_existing_duplicate_loop_boundary(self):
        for rows in (0, 5):
            for loops in (False, True):
                model, oracle = MixHopConv(3, 2, [3, 0], loops), TorchMix(3, 2, [3, 0], loops)
                self.copy(model, oracle)
                x = torch.randn(rows, 3, requires_grad=True); nx = native(x, True)
                edge, weight = torch.empty((2, 0), dtype=torch.long), torch.empty(0, requires_grad=True)
                nw = native(weight, True)
                self.gradients(model, oracle, model(nx, native(edge), nw), oracle(x, edge, weight), [('x', nx, x), ('weight', nw, weight)])
        duplicate = torch.cat([self.edge, torch.tensor([[2], [2]])], 1)
        model, oracle = MixHopConv(3, 2), TorchMix(3, 2)
        self.copy(model, oracle)
        x = torch.randn(6, 3, requires_grad=True); nx = native(x, True)
        topology = fr.graph.EdgeIndex(6, duplicate[0].tolist(), duplicate[1].tolist())
        self.gradients(model, oracle, model(nx, topology), oracle(x, duplicate), [('x', nx, x)])
        with self.assertRaisesRegex(NotImplementedError, 'duplicate'): model(native(x), native(duplicate), fr.Tensor.ones([9]))
        model, oracle = MixHopConv(3, 2, add_self_loops=False), TorchMix(3, 2, add_self_loops=False)
        self.copy(model, oracle)
        x, weight = torch.randn(6, 3, requires_grad=True), torch.rand(9, requires_grad=True)
        nx, nw = native(x, True), native(weight, True)
        self.gradients(model, oracle, model(nx, native(duplicate), nw), oracle(x, duplicate, weight), [('x', nx, x), ('weight', nw, weight)])

    def test_initializer_layout_bias_reset_and_independent_selected_slots(self):
        powers = [3, 1, 3]
        model, oracle = MixHopConv(3, 4, powers), TorchMix(3, 4, powers)
        self.assertEqual(set(dict(model.named_parameters())), set(dict(oracle.named_parameters())))
        self.assertIsInstance(model.lins[0], fr.nn.Identity)
        self.assertIsInstance(model.lins[2], fr.nn.Identity)
        self.assertIsNot(model.lins[1].weight, model.lins[3].weight)
        powers.reverse(); powers.append(8)
        self.assertEqual(model.powers, (3, 1, 3))
        with fr.no_grad():
            for parameter in model.parameters(): parameter.tensor().copy_(fr.Tensor.ones(parameter.tensor().shape) * 17.)
        model.reset_parameters()
        self.assertEqual(model.bias.tensor().tolist(), [0.] * 12)
        for index in (1, 3): self.assertTrue(all(abs(value) <= 3 ** -.5 + 1e-7 for value in model.lins[index].weight.tensor().flatten().tolist()))
        with self.assertRaises(AttributeError): model.powers = (0,)

    def test_serialized_shared_projection_optimizer_generator_and_power_config(self):
        def factory(changed=False):
            model = MixHopConv(3, 4, [2, 1, 0] if changed else [0, 1, 2])
            model.lins[2].weight = model.lins[0].weight
            return model
        def step(model, optimizer, rng):
            x, weight = fr.Tensor.randn([6, 3], generator=rng), fr.Tensor.rand([8], generator=rng) + .4
            output = model(x, native(self.edge), weight)
            loss = output.square().mean(); values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad(); return values
        model, rng = factory(), fr.Generator(479)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'mix': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'mix': other_opt}, generator=other_rng)
        self.assertIs(other.lins[0].weight, other.lins[2].weight)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'mix': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'mix': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed, changed_rng = factory(True), fr.Generator(1)
        changed_opt = fr.optim.AdamW(changed.parameters())
        def state(): return {key: value.tolist() for key, value in fr.checkpoint.snapshot_training(changed, optimizers={'mix': changed_opt}, generator=changed_rng).tensors().items()}
        before = state()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'mix': changed_opt}, generator=changed_rng)
        self.assertEqual(before, state())

    def test_metadata_capacity_and_invalid_normalization_before_projections(self):
        model, x = MixHopConv(3, 4), fr.Tensor.ones([6, 3])
        with patch.object(model.lins[0], 'forward', side_effect=AssertionError('projection before metadata rejection')):
            for value, topology, weights in ((fr.Tensor.ones([6, 2]), native(self.edge), None), (x, fr.Tensor.from_i64([0, 6], [2, 1]), None), (x, native(self.edge), fr.Tensor.ones([7]))):
                with self.assertRaises(ValueError): model(value, topology, weights)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, native(self.edge)))
        for powers in ([], [-1], [True], [1.5], 'bad'):
            with self.assertRaises(ValueError): MixHopConv(3, 4, powers)
        for kwargs in (dict(in_channels=0, out_channels=4), dict(in_channels=3, out_channels=0), dict(in_channels=3, out_channels=4, bias=1)):
            with self.assertRaises((ValueError, TypeError)): MixHopConv(**kwargs)
        with self.assertRaises(NotImplementedError): MixHopConv(3, 4, flow='target_to_source')
        with patch('ferro._graph_mixhop.Linear', side_effect=AssertionError('allocated parameters before capacity rejection')):
            with self.assertRaisesRegex(ValueError, 'capacity'): MixHopConv(3, 4, [sys.maxsize // 8])
            with self.assertRaisesRegex(ValueError, 'capacity'): MixHopConv(3, sys.maxsize // 4 + 1)
        huge_edges = fr.Tensor.from_i64([0], [1, 1]).expand([2, sys.maxsize // 16 + 1])
        with patch.object(fr.Tensor, 'tolist', side_effect=AssertionError('topology read before capacity rejection')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(x, huge_edges)
        virtual = fr.Tensor.ones([1, 3]).expand([sys.maxsize // 16 + 1, 3])
        with patch('ferro._graph_mixhop._gcn_weights', side_effect=AssertionError('normalization before loop capacity rejection')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(virtual, fr.Tensor.from_i64([], [2, 0]))


if __name__ == '__main__': unittest.main(verbosity=2)
