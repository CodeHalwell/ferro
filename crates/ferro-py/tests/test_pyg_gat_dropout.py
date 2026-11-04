"""Actual PyG GAT dropout, initialization and saved native-stream references."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import GATConv as TorchGAT, GATv2Conv as TorchGATv2
import ferro as fr
from ferro.graph import GATConv, GATv2Conv
import test_pyg_modulated as helpers

native = helpers.native


class GATDropoutTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1753)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def test_matched_nonzero_dropout_outputs_returned_alpha_and_all_gradients(self):
        cases = [(GATConv, TorchGAT, False, False), (GATConv, TorchGAT, True, False),
                 (GATv2Conv, TorchGATv2, False, True), (GATv2Conv, TorchGATv2, True, False)]
        for cls, ref, bipartite, shared in cases:
            with self.subTest(kind=cls.__name__, bipartite=bipartite, shared=shared):
                widths = (3, 2) if bipartite else 3
                options = dict(heads=2, concat=False, dropout=.3, residual=True, edge_dim=2, fill_value='add')
                if cls is GATv2Conv: options['share_weights'] = shared
                model, oracle = cls(widths, 4, **options), ref(widths, 4, **options)
                self.copy(model, oracle)
                x = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else x
                attrs = torch.randn(9, 2, requires_grad=True)
                nx, nt, na = native(x, True), None, native(attrs, True)
                nt = native(target, True) if bipartite else nx
                count = 7 + (5 if bipartite else 6)
                sampler = fr.nn.Dropout(.3, seed=1)
                with fr.no_grad(): sampler.rng_state.copy_(model.attention_dropout.rng_state)
                mask = torch.tensor(sampler(fr.Tensor.ones([count, 2])).tolist())
                calls = []
                def dropout(value, p=.5, training=True, inplace=False):
                    calls.append(value.shape)
                    self.assertEqual(list(value.shape), [count, 2]); self.assertEqual(p, .3)
                    return value * mask
                actual = model((nx, nt) if bipartite else nx, native(self.edge), na, return_attention_weights=True)
                with patch('torch.nn.functional.dropout', side_effect=dropout):
                    expected = oracle((x, target) if bipartite else x, self.edge, attrs, return_attention_weights=True)
                self.assertEqual(actual[1][0].tolist(), expected[1][0].tolist())
                inputs = [('x', nx, x), ('edge_attr', na, attrs)] + ([('target', nt, target)] if bipartite else [])
                self.gradients(model, oracle, actual[0], expected[0], inputs, actual[1][1], expected[1][1])
                self.assertEqual(len(calls), 1)
                self.assertEqual(model.attention_dropout.rng_state.tolist(), sampler.rng_state.tolist())

    def test_unit_dropout_eval_empty_edges_and_default_state_compatibility(self):
        for cls, ref in ((GATConv, TorchGAT), (GATv2Conv, TorchGATv2)):
            self.assertIsNone(getattr(cls(3, 4), 'attention_dropout', None))
            for p, training, empty in ((1., True, False), (.3, False, False), (.3, True, True)):
                model, oracle = cls(3, 4, heads=2, dropout=p, add_self_loops=False), ref(3, 4, heads=2, dropout=p, add_self_loops=False)
                model.train(training); oracle.train(training)
                self.copy(model, oracle)
                x = torch.randn(6, 3, requires_grad=True)
                nx = native(x, True)
                edge = torch.empty(2, 0, dtype=torch.int64) if empty else self.edge
                stream = model.attention_dropout.rng_state.tolist()
                actual, expected = model(nx, native(edge), return_attention_weights=True), oracle(x, edge, return_attention_weights=True)
                self.gradients(model, oracle, actual[0], expected[0], [('x', nx, x)], actual[1][1], expected[1][1])
                if not training or empty: self.assertEqual(stream, model.attention_dropout.rng_state.tolist())

    def test_projection_bias_initialization_and_reset_match_reference_distributions(self):
        model = GATv2Conv((3, 5), 4, heads=2)
        for layer, width in ((model.lin_l, 3), (model.lin_r, 5)):
            self.assertTrue(any(value != 0. for value in layer.bias.tensor().tolist()))
            self.assertTrue(all(abs(value) <= width ** -.5 for value in layer.bias.tensor().tolist()))
        self.assertEqual(model.bias.tensor().tolist(), [0.] * 8)
        model.reset_parameters()
        self.assertTrue(any(value != 0. for value in model.lin_l.bias.tensor().tolist()))
        with patch('ferro._graph_attention.Dropout', side_effect=AssertionError('stream construction reached')):
            with self.assertRaisesRegex(ValueError, 'equal source'): GATv2Conv((3, 5), 4, share_weights=True, dropout=.2)
        shared = GATv2Conv(3, 4, share_weights=True)
        self.assertIs(shared.lin_l, shared.lin_r)

    def test_exact_optimizer_rng_dropout_restart_and_validation_before_stream(self):
        def factory(p=.3): return GATv2Conv(3, 4, heads=2, dropout=p, share_weights=True, edge_dim=2)
        model, rng = factory(), fr.Generator(163)
        optimizer, edge = fr.optim.AdamW(model.parameters(), lr=.002), native(self.edge)
        def step(current, opt, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            attrs = fr.Tensor.rand([9, 2], generator=generator)
            output, (_, alpha) = current(x, edge, attrs, return_attention_weights=True)
            loss = output.square().mean() + alpha.square().mean()
            value = output.tolist(), alpha.tolist(), loss.item()
            loss.backward(); opt.step(); opt.zero_grad()
            return value
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'gat': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state')
            saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'gat': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.lin_l, other.lin_r)
        left = fr.checkpoint.snapshot_training(model, optimizers={'gat': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'gat': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertEqual((left[name].dtype, left[name].shape, left[name].tolist()),
                             (right[name].dtype, right[name].shape, right[name].tolist()), name)
        self.assertEqual(model.attention_dropout.rng_state.tolist(), other.attention_dropout.rng_state.tolist())
        changed = factory(.4)
        stream = changed.attention_dropout.rng_state.tolist()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'gat': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(stream, changed.attention_dropout.rng_state.tolist())
        x, attrs = fr.Tensor.ones([6, 3]), fr.Tensor.ones([9, 2])
        for call in (lambda: changed(x, edge, fr.Tensor.ones([8, 2])), lambda: changed(x, edge, attrs, return_attention_weights=1),
                     lambda: changed(fr.Tensor.ones([6, 4]), edge, attrs)):
            with self.assertRaises((ValueError, TypeError)): call()
            self.assertEqual(stream, changed.attention_dropout.rng_state.tolist())
        with self.assertRaisesRegex(ValueError, 'capture'): fr.capture(lambda: changed(x, edge, attrs))
        self.assertEqual(stream, changed.attention_dropout.rng_state.tolist())
        changed.dropout = .6
        with self.assertRaisesRegex(ValueError, 'dropout configuration'): changed(x, edge, attrs)
        self.assertEqual(stream, changed.attention_dropout.rng_state.tolist())
        changed.reset_parameters()
        self.assertEqual(stream, changed.attention_dropout.rng_state.tolist())



    def test_prefix_and_implicit_loop_topology_capacity_before_allocation(self):
        empty = fr.Tensor.from_i64([], [2, 0])
        for cls in (GATConv, GATv2Conv):
            for count, loops in ((sys.maxsize // 8, False), (sys.maxsize // 16 + 1, True)):
                model = cls(1, 1, add_self_loops=loops, dropout=.2)
                value = fr.Tensor.ones([1, 1]).expand(count, 1)
                stream = model.attention_dropout.rng_state.tolist()
                with patch('ferro._graph_attention.add_self_loops', side_effect=AssertionError('allocation reached')):
                    with self.assertRaisesRegex(ValueError, 'capacity'): model(value, empty)
                self.assertEqual(stream, model.attention_dropout.rng_state.tolist())

if __name__ == '__main__':
    unittest.main(verbosity=2)
