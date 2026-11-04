"""Actual PyG spectral pooling values, all losses, native gradients and state."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import dense_mincut_pool as torch_mincut, DMoNPooling as TorchDMoN
import ferro as fr
from ferro.graph import dense_mincut_pool, DMoNPooling
import test_pyg_modulated as helpers

native = helpers.native


class DenseSpectralPoolTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2213)

    def compare(self, model, oracle, actual, expected, inputs):
        for value, reference in zip(actual, expected): self.close(value, reference)
        self.gradients(model, oracle, fr.cat([value.flatten() for value in actual], 0), torch.cat([value.flatten() for value in expected], 0), inputs)

    def test_mincut_actual_pyg_all_outputs_input_adjacency_assignment_and_mask_gradients(self):
        for batch, width, clusters, temperature, weighted in ((1, 3, 3, 1., False), (2, 3, 2, .7, False), (3, 2, 4, 1.8, True), (2, 0, 2, 1., False)):
            x = torch.randn(batch, width, 5, requires_grad=True)
            adj = (torch.rand(batch, 5, 5) + .1).requires_grad_()
            assignments = torch.randn(batch, clusters, 5, requires_grad=True)
            mask = (torch.rand(batch, 5) + .2).requires_grad_() if weighted else torch.tensor([[True, True, True, False, True]] * batch)
            nx, na, ns = native(x, True), native(adj, True), native(assignments, True)
            nm = native(mask, True) if weighted else fr.Tensor.from_bool(mask.flatten().tolist(), list(mask.shape))
            tx, ts = x.transpose(1, 2), assignments.transpose(1, 2)
            fx, fs = nx.transpose(1, 2), ns.transpose(1, 2)
            if batch == 1: tx, ts, adj_, mask_ = tx.squeeze(0), ts.squeeze(0), adj.squeeze(0), mask.squeeze(0); fx, fs, na_, nm_ = fx.squeeze(0), fs.squeeze(0), na.squeeze(0), nm.squeeze(0)
            else: adj_, mask_, na_, nm_ = adj, mask, na, nm
            inputs = [('x', nx, x), ('adj', na, adj), ('assignments', ns, assignments)] + ([('mask', nm, mask)] if weighted else [])
            self.compare(fr.nn.Identity(), torch.nn.Identity(), dense_mincut_pool(fx, na_, fs, nm_, temperature), torch_mincut(tx, adj_, ts, mask_, temperature), inputs)

    def test_dmon_actual_pyg_affine_chain_all_losses_gradients_and_pairwise_count_normalization(self):
        for channels, clusters, batch, weighted in ((3, 1, 1, False), (3, 2, 1, False), ([3, 5, 4], 3, 2, True), ([3, 2], 4, 3, False)):
            model, oracle = DMoNPooling(channels, clusters), TorchDMoN(channels, clusters)
            self.copy(model, oracle)
            x = torch.randn(batch, 3, 5, requires_grad=True)
            adj = (torch.rand(batch, 5, 5) + .1).requires_grad_()
            mask = (torch.rand(batch, 5) + .2).requires_grad_() if weighted else torch.tensor([[True] * (5 - i) + [False] * i for i in range(batch)])
            nx, na = native(x, True), native(adj, True)
            nm = native(mask, True) if weighted else fr.Tensor.from_bool(mask.flatten().tolist(), list(mask.shape))
            actual, expected = model(nx.transpose(1, 2), na, nm), oracle(x.transpose(1, 2), adj, mask)
            self.compare(model, oracle, actual, expected, [('x', nx, x), ('adj', na, adj)] + ([('mask', nm, mask)] if weighted else []))
            sizes = expected[0].detach().sum(1).norm(dim=1)
            counts = mask.detach().sum(1, keepdim=True)
            pairwise = (sizes / counts * math.sqrt(clusters) - 1).mean()
            self.close(actual[-1], pairwise)
            if batch > 1:
                self.assertGreater(abs(float(pairwise - (sizes / counts.squeeze(1) * math.sqrt(clusters) - 1).mean())), 1e-6)

    def test_dmon_independent_same_dropout_mask_and_eval_stream(self):
        model, oracle = DMoNPooling([3, 4], 3, .3), TorchDMoN([3, 4], 3, .3)
        self.copy(model, oracle)
        sampler = fr.nn.Dropout(.3, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(model.assignment_dropout.rng_state)
        multiplier = helpers.torch_value(sampler(fr.Tensor.ones([2, 5, 3])))
        x, adj = torch.randn(2, 5, 3, requires_grad=True), (torch.rand(2, 5, 5) + .1).requires_grad_()
        nx, na = native(x, True), native(adj, True)
        original_dropout = torch.nn.functional.dropout
        def fixed_dropout(value, p=.5, training=True, inplace=False):
            if p == .3: return value * multiplier if training else value
            return original_dropout(value, p, training, inplace)
        with patch('torch.nn.functional.dropout', side_effect=fixed_dropout):
            self.compare(model, oracle, model(nx, na), oracle(x, adj), [('x', nx, x), ('adj', na, adj)])
        self.assertEqual(model.assignment_dropout.rng_state.tolist(), sampler.rng_state.tolist())
        model.eval(); oracle.eval()
        before = model.assignment_dropout.rng_state.tolist()
        for actual, expected in zip(model(native(x), native(adj)), oracle(x.detach(), adj.detach())): self.close(actual, expected)
        self.assertEqual(before, model.assignment_dropout.rng_state.tolist())

    def test_single_cluster_zero_norm_and_singular_graph_values_match_reference(self):
        for clusters, zero_adj, zero_mask in ((1, False, False), (3, True, False), (2, False, True)):
            x = torch.randn(1, 4, 3, requires_grad=True)
            adj = (torch.zeros(1, 4, 4) if zero_adj else torch.rand(1, 4, 4) + .1).requires_grad_()
            s = torch.randn(1, 4, clusters, requires_grad=True)
            mask = torch.zeros(1, 4, dtype=torch.bool) if zero_mask else None
            nx, na, ns = native(x, True), native(adj, True), native(s, True)
            nm = None if mask is None else fr.Tensor.from_bool(mask.flatten().tolist(), list(mask.shape))
            pairs = [(dense_mincut_pool(nx, na, ns, nm), torch_mincut(x, adj, s, mask))]
            model, oracle = DMoNPooling(3, clusters), TorchDMoN(3, clusters)
            self.copy(model, oracle)
            pairs.append((model(nx, na, nm), oracle(x, adj, mask)))
            for actual, expected in pairs:
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(helpers.torch_value(value), reference.detach(), atol=3e-5, rtol=4e-4, equal_nan=True)
            if clusters == 1:
                actual, expected = pairs[0]
                self.assertEqual(actual[-1].item(), 0.)
                self.compare(fr.nn.Identity(), torch.nn.Identity(), actual, expected, [('x', nx, x), ('adj', na, adj), ('s', ns, s)])

    def test_serialized_dmon_tied_mlp_optimizer_rng_dropout_and_reset(self):
        def factory(p=.2):
            model = DMoNPooling([3, 3], 3, p)
            model.mlp.lins[1].weight = model.mlp.lins[0].weight
            return model
        adjacency = fr.Tensor.ones([2, 5, 5])
        def step(model, optimizer, rng):
            output = model(fr.Tensor.randn([2, 5, 3], generator=rng), adjacency)
            loss = output[1].square().mean() + output[3] + output[4] + output[5]
            values = [value.tolist() for value in output], loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(331)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'dmon': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'dmon': other_opt}, generator=other_rng)
        self.assertIs(other.mlp.lins[0].weight, other.mlp.lins[1].weight)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'dmon': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'dmon': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'dmon': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])
        before = model.assignment_dropout.rng_state.tolist()
        model.reset_parameters()
        self.assertEqual(before, model.assignment_dropout.rng_state.tolist())
        for layer in model.mlp.lins:
            self.assertTrue(all(abs(value) <= layer.in_features ** -.5 for value in layer.weight.tensor().flatten().tolist()))
            self.assertTrue(any(value != 0. for value in layer.bias.tensor().tolist()))

    def test_metadata_capacity_and_invalid_calls_precede_dropout(self):
        model = DMoNPooling([3, 4], 2, .3)
        x, adj = fr.Tensor.ones([2, 5, 3]), fr.Tensor.ones([2, 5, 5])
        before = model.assignment_dropout.rng_state.tolist()
        calls = (lambda: model(x, fr.Tensor.ones([1, 5, 5])), lambda: model(x, adj, fr.Tensor.ones([5])),
                 lambda: model(fr.Tensor.ones([2, 5, 2]), adj), lambda: model(fr.Tensor.zeros([2, 0, 3]), fr.Tensor.zeros([2, 0, 0])),
                 lambda: model(x, adj, fr.Tensor.from_i64([1] * 10, [2, 5])))
        for call in calls:
            with self.assertRaises((ValueError, TypeError)): call()
            self.assertEqual(before, model.assignment_dropout.rng_state.tolist())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, adj))
        for temperature in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): dense_mincut_pool(x, adj, fr.Tensor.ones([2, 5, 2]), temp=temperature)
        with self.assertRaises(ValueError): dense_mincut_pool(x, adj, fr.Tensor.ones([1, 5, 2]))
        for channels, clusters in (([], 2), ([3, 0], 2), (3, 0)):
            with self.assertRaises(ValueError): DMoNPooling(channels, clusters)
        groups = math.isqrt(sys.maxsize // 4) + 1
        with patch.object(model.mlp, 'forward', side_effect=AssertionError('MLP reached before capacity check')):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                model(fr.Tensor.ones([1, 1, 3]).expand(groups, 1, 3), fr.Tensor.ones([1, 1, 1]).expand(groups, 1, 1))
        self.assertEqual(before, model.assignment_dropout.rng_state.tolist())
        with patch('ferro._graph_dense_pool.Linear', side_effect=AssertionError('parameters allocated')):
            with self.assertRaisesRegex(ValueError, 'capacity'): DMoNPooling(1, sys.maxsize // 8 + 1)


if __name__ == '__main__': unittest.main(verbosity=2)
