"""Actual PyG APPNP/TransformerConv dropout and native checkpoint references."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import APPNP as TorchAPPNP, TransformerConv as TorchTransformer
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import ferro as fr
from ferro.graph import APPNP, TransformerConv
import test_pyg_modulated as helpers

native = helpers.native


class PropagationDropoutTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1789)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def masks(self, layer, shape, count):
        sampler = fr.nn.Dropout(layer.p, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(layer.rng_state)
        masks = [torch.tensor(sampler(fr.Tensor.ones(shape)).tolist()).reshape(shape) for _ in range(count)]
        return sampler, masks

    def test_appnp_cumulative_masks_normalized_raw_and_edge_gradients(self):
        for normalize, weighted, steps in ((True, True, 3), (False, True, 4), (True, False, 2)):
            with self.subTest(normalize=normalize, weighted=weighted, steps=steps):
                model, oracle = APPNP(steps, .23, dropout=.2, normalize=normalize), TorchAPPNP(steps, .23, dropout=.2, normalize=normalize)
                x = torch.randn(6, 3, requires_grad=True)
                weight = (torch.rand(9) + .6).requires_grad_() if weighted else None
                nx, nw = native(x, True), native(weight, True) if weighted else None
                count = gcn_norm(self.edge, weight, 6)[0].shape[1] if normalize else 9
                sampler, masks = self.masks(model.edge_dropout, [count], steps)
                calls = []
                def dropout(value, p=.5, training=True, inplace=False):
                    calls.append(value.detach().clone())
                    self.assertEqual(p, .2)
                    return value * masks[len(calls) - 1]
                actual = model(nx, native(self.edge), nw)
                with patch('torch.nn.functional.dropout', side_effect=dropout):
                    expected = oracle(x, self.edge, weight)
                inputs = [('x', nx, x)] + ([('weight', nw, weight)] if weighted else [])
                self.gradients(model, oracle, actual, expected, inputs)
                self.assertEqual(len(calls), steps)
                for index in range(1, steps):
                    torch.testing.assert_close(calls[index], calls[index - 1] * masks[index - 1])
                self.assertEqual(model.edge_dropout.rng_state.tolist(), sampler.rng_state.tolist())

    def test_transformer_dropped_messages_raw_returned_attention_and_all_gradients(self):
        for bipartite, concat, root, beta in ((False, True, True, True), (True, False, True, False), (True, True, False, True)):
            options = dict(heads=2, concat=concat, root_weight=root, beta=beta, dropout=.35, edge_dim=2)
            channels = (3, 2) if bipartite else 3
            model, oracle = TransformerConv(channels, 4, **options), TorchTransformer(channels, 4, **options)
            self.copy(model, oracle)
            x = torch.randn(6, 3, requires_grad=True)
            target = torch.randn(5, 2, requires_grad=True) if bipartite else x
            attrs = torch.randn(9, 2, requires_grad=True)
            nx, nt, na = native(x, True), native(target, True) if bipartite else None, native(attrs, True)
            sampler, masks = self.masks(model.attention_dropout, [9, 2], 1)
            calls = []
            def dropout(value, p=.5, training=True, inplace=False):
                self.assertEqual(p, .35); calls.append(value.detach().clone())
                return value * masks[0]
            actual = model((nx, nt) if bipartite else nx, native(self.edge), na, return_attention_weights=False)
            with patch('torch.nn.functional.dropout', side_effect=dropout):
                expected = oracle((x, target) if bipartite else x, self.edge, attrs, return_attention_weights=False)
            inputs = [('x', nx, x), ('attrs', na, attrs)] + ([('target', nt, target)] if bipartite else [])
            self.gradients(model, oracle, actual[0], expected[0], inputs, actual[1][1], expected[1][1])
            self.assertEqual(len(calls), 1)
            self.close(actual[1][1], calls[0])
            self.assertEqual(model.attention_dropout.rng_state.tolist(), sampler.rng_state.tolist())

    def test_unit_eval_empty_and_zero_iteration_stream_behavior(self):
        for kind in ('appnp', 'transformer'):
            for p, training, empty, steps in ((1., True, False, 3), (.3, False, False, 2), (.3, True, True, 2), (.3, True, False, 0)):
                if kind == 'appnp':
                    model, oracle = APPNP(steps, .2, dropout=p, normalize=False), TorchAPPNP(steps, .2, dropout=p, normalize=False)
                    stream_layer = model.edge_dropout
                else:
                    if steps == 0: continue
                    model, oracle = TransformerConv(3, 4, heads=2, dropout=p), TorchTransformer(3, 4, heads=2, dropout=p)
                    stream_layer = model.attention_dropout
                    self.copy(model, oracle)
                model.train(training); oracle.train(training)
                stream = stream_layer.rng_state.tolist()
                edge = torch.empty(2, 0, dtype=torch.int64) if empty else self.edge
                x = torch.randn(6, 3, requires_grad=True)
                nx = native(x, True)
                if kind == 'appnp':
                    weight = torch.ones(edge.shape[1], requires_grad=True)
                    nw = native(weight, True)
                    actual, expected = model(nx, native(edge), nw), oracle(x, edge, weight)
                    self.gradients(model, oracle, actual, expected, [('x', nx, x), ('weight', nw, weight)])
                else:
                    actual, expected = model(nx, native(edge), return_attention_weights=True), oracle(x, edge, return_attention_weights=True)
                    self.gradients(model, oracle, actual[0], expected[0], [('x', nx, x)], actual[1][1], expected[1][1])
                if not training or empty or steps == 0:
                    self.assertEqual(stream, stream_layer.rng_state.tolist())
        self.assertFalse(hasattr(APPNP(2, .2), 'edge_dropout'))
        self.assertFalse(hasattr(TransformerConv(3, 4), 'attention_dropout'))

    def test_saved_combined_optimizer_generator_and_both_streams_continue_exactly(self):
        class Network(fr.nn.Module):
            p: float = .2
            def build(self):
                self.attention = TransformerConv(3, 3, heads=2, concat=False, dropout=self.p)
                self.propagate = APPNP(3, .2, dropout=self.p)
            def forward(self, x, edge):
                return self.propagate(self.attention(x, edge), edge)
        model, generator = Network(), fr.Generator(179)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        edge = native(self.edge)
        def step(model, optimizer, generator):
            value = model(fr.Tensor.rand([6, 3], generator=generator), edge)
            loss = value.square().mean()
            result = value.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return result
        step(model, optimizer, generator)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/snapshot')
            saved = fr.checkpoint.load(directory + '/snapshot')
        expected = [step(model, optimizer, generator) for _ in range(3)]
        other, other_generator = Network(), fr.Generator(1)
        other_optimizer = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'main': other_optimizer}, generator=other_generator)
        self.assertEqual(expected, [step(other, other_optimizer, other_generator) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'main': other_optimizer}, generator=other_generator, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertEqual((left[name].dtype, left[name].shape, left[name].tolist()),
                             (right[name].dtype, right[name].shape, right[name].tolist()), name)
        changed = Network(p=.3)
        before = changed.attention.attention_dropout.rng_state.tolist(), changed.propagate.edge_dropout.rng_state.tolist()
        with self.assertRaises(ValueError):
            saved.restore(changed, optimizers={'main': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, (changed.attention.attention_dropout.rng_state.tolist(), changed.propagate.edge_dropout.rng_state.tolist()))

    def test_invalid_calls_and_huge_prefix_reject_before_random_streams(self):
        x, edge = fr.Tensor.ones([6, 3]), native(self.edge)
        model = APPNP(2, .2, dropout=.3, normalize=False)
        stream = model.edge_dropout.rng_state.tolist()
        for call in (lambda: model(x, edge), lambda: model(x, edge, fr.Tensor.ones([8])),
                     lambda: fr.capture(lambda: model(x, edge, fr.Tensor.ones([9])))):
            with self.assertRaises(ValueError): call()
            self.assertEqual(stream, model.edge_dropout.rng_state.tolist())
        model.eval()
        self.close(model(x, edge), TorchAPPNP(2, .2, dropout=.3, normalize=False).eval()(torch.ones(6, 3), self.edge))
        model.reset_parameters()
        self.assertEqual(stream, model.edge_dropout.rng_state.tolist())
        transformer = TransformerConv(3, 4, dropout=.3, edge_dim=2)
        stream = transformer.attention_dropout.rng_state.tolist()
        for call in (lambda: transformer(x, edge), lambda: transformer(x, edge, fr.Tensor.ones([8, 2])),
                     lambda: transformer(x, edge, fr.Tensor.ones([9, 2]), return_attention_weights=1),
                     lambda: fr.capture(lambda: transformer(x, edge, fr.Tensor.ones([9, 2])))):
            with self.assertRaises((ValueError, TypeError)): call()
            self.assertEqual(stream, transformer.attention_dropout.rng_state.tolist())
        empty = fr.Tensor.from_i64([], [2, 0])
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8, 1)
        for model in (APPNP(2, .2, dropout=.3, add_self_loops=False), TransformerConv(1, 1, dropout=.3)):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(huge, empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
