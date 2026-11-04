"""Actual PyG hypergraph incidence, attention and all-adjoint references."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import HypergraphConv as TorchHypergraph
import ferro as fr
from ferro.graph import HypergraphConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class HypergraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1667)
        self.incidence = torch.tensor([[0, 1, 2, 1, 1, 3, 4, 0], [0, 0, 0, 1, 1, 2, 2, 2]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=4e-5, rtol=5e-4)

    def compare(self, model, reference, nodes=6, edges=4, incidence=None, weighted=True, attributes=True):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters(): parameter.tensor().copy_(native(parameters[name]))
        incidence = self.incidence if incidence is None else incidence
        x = torch.randn(nodes, 3, requires_grad=True)
        attr = torch.randn(edges, 3, requires_grad=True) if attributes else None
        weight = (torch.rand(edges) + .4).requires_grad_(True) if weighted else None
        nx, na, nw = native(x, True), None if attr is None else native(attr, True), None if weight is None else native(weight, True)
        actual = model(nx, native(incidence), nw, na, edges)
        expected = reference(x, incidence, weight, attr, edges)
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward(); (expected * probe).sum().backward()
        inputs = [('x', nx, x)] + ([] if attr is None else [('attributes', na, attr)]) + ([] if weight is None else [('weights', nw, weight)])
        inputs += [(name, parameter.tensor(), parameters[name]) for name, parameter in model.named_parameters()]
        for name, left, right in inputs:
            with self.subTest(gradient=name):
                if right.grad is None: self.assertIsNone(left.grad)
                else: self.close(left.grad, right.grad)

    def test_plain_incidence_weights_duplicates_isolates_and_ignored_attributes(self):
        for weighted, attributes, bias in ((False, False, True), (True, False, False), (True, True, True)):
            self.compare(HypergraphConv(3, 4, bias=bias), TorchHypergraph(3, 4, bias=bias), weighted=weighted, attributes=attributes)

    def test_attention_node_edge_normalization_heads_and_all_gradients(self):
        for mode, heads, concat, weighted, bias in (('node', 1, True, True, True), ('node', 3, False, False, False),
                                                   ('edge', 2, True, True, False), ('edge', 3, False, True, True)):
            with self.subTest(mode=mode, heads=heads, concat=concat):
                options = dict(use_attention=True, attention_mode=mode, heads=heads, concat=concat, bias=bias)
                self.compare(HypergraphConv(3, 4, **options), TorchHypergraph(3, 4, **options), weighted=weighted)

    def test_same_dropout_mask_reused_both_propagations_all_gradients(self):
        options = dict(use_attention=True, attention_mode='edge', heads=2, concat=False, dropout=.3)
        model, reference = HypergraphConv(3, 4, **options), TorchHypergraph(3, 4, **options)
        sampler = fr.nn.Dropout(.3, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(model.attention_dropout.rng_state)
        mask = torch.tensor(sampler(fr.Tensor.ones([8, 2])).tolist())
        calls = []
        def dropout(value, p=.5, training=True, inplace=False):
            calls.append(value.shape)
            self.assertEqual(list(value.shape), [8, 2]); self.assertEqual(p, .3)
            return value * mask
        with patch('torch.nn.functional.dropout', side_effect=dropout): self.compare(model, reference)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sampler.rng_state.tolist(), model.attention_dropout.rng_state.tolist())
        options['dropout'] = 1.
        self.compare(HypergraphConv(3, 4, **options), TorchHypergraph(3, 4, **options))
        options['dropout'] = .3
        model, reference = HypergraphConv(3, 4, **options).eval(), TorchHypergraph(3, 4, **options).eval()
        before = model.attention_dropout.rng_state.tolist()
        self.compare(model, reference)
        self.assertEqual(before, model.attention_dropout.rng_state.tolist())

    def test_empty_incidence_nodes_hyperedges_and_inferred_edge_count(self):
        incidence = torch.empty(2, 0, dtype=torch.int64)
        for nodes, edges in ((0, 0), (0, 3), (4, 0), (4, 3)):
            for attention in (False, True):
                with self.subTest(nodes=nodes, edges=edges, attention=attention):
                    options = dict(use_attention=attention, heads=2 if attention else 1)
                    self.compare(HypergraphConv(3, 4, **options), TorchHypergraph(3, 4, **options), nodes, edges, incidence)
        model, x = HypergraphConv(3, 4), fr.Tensor.ones([6, 3])
        self.assertEqual(model(x, native(self.incidence)).tolist(), model(x, native(self.incidence), num_edges=3).tolist())

    def test_checkpoint_dropout_optimizer_and_configuration(self):
        def factory(mode='node'):
            return HypergraphConv(3, 4, use_attention=True, attention_mode=mode, heads=2, dropout=.2)
        model, rng = factory(), fr.Generator(149)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        incidence = native(self.incidence)
        def step(current, opt, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            attrs = fr.Tensor.rand([4, 3], generator=generator)
            weights = fr.Tensor.rand([4], generator=generator) + .4
            output = current(x, incidence, weights, attrs, 4)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); opt.step(); opt.zero_grad()
            return values
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'hyper': optimizer}, generator=rng, step=1)
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'hyper': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        for left, right in zip(model.parameters(), other.parameters()): self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        self.assertEqual(model.attention_dropout.rng_state.tolist(), other.attention_dropout.rng_state.tolist())
        changed = factory('edge')
        stream = changed.attention_dropout.rng_state.tolist()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'hyper': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(stream, changed.attention_dropout.rng_state.tolist())

    def test_invalid_shapes_capacity_capture_before_random_state_advances(self):
        for constructor in (lambda: HypergraphConv(0, 2), lambda: HypergraphConv(3, 2, use_attention=1),
                            lambda: HypergraphConv(3, 2, attention_mode='wrong'), lambda: HypergraphConv(3, 2, dropout=2.),
                            lambda: HypergraphConv(3, 2, use_attention=True, heads=sys.maxsize)):
            with self.assertRaises((ValueError, TypeError)): constructor()
        with self.assertRaises(NotImplementedError): HypergraphConv(3, 2, heads=2)
        with patch('ferro._graph_hyper.Linear', side_effect=AssertionError('attention capacity must reject before Linear')):
            with self.assertRaises(ValueError): HypergraphConv(1, sys.maxsize // 8 + 1, use_attention=True)
        model, x, index = HypergraphConv(3, 4, use_attention=True, heads=2, dropout=.3), fr.Tensor.ones([6, 3]), native(self.incidence)
        attrs, weights = fr.Tensor.ones([4, 3]), fr.Tensor.ones([4])
        stream = model.attention_dropout.rng_state.tolist()
        for call in (lambda: model(x, index, weights, None, 4), lambda: model(x, index, weights, attrs, 2),
                     lambda: model(x, index, weights, fr.Tensor.ones([3, 3]), 4), lambda: model(x, index, fr.Tensor.ones([3]), attrs, 4),
                     lambda: model(x, native(torch.tensor([[6], [0]])), weights, attrs, 4)):
            with self.assertRaises((ValueError, TypeError)): call()
            self.assertEqual(stream, model.attention_dropout.rng_state.tolist())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, index, weights, attrs, 4))
        self.assertEqual(stream, model.attention_dropout.rng_state.tolist())
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaises(ValueError): HypergraphConv(1, 1)(huge, fr.Tensor.from_i64([], [2, 0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
