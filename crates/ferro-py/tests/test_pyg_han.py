"""Actual PyG HAN node/semantic attention, gradients and serialized state."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import HANConv as TorchHAN
import ferro as fr
from ferro.graph import HANConv
import test_pyg_modulated as helpers

native = helpers.native


class HANTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2423)
        self.nodes = ['a', 'b', 'c']
        self.relations = [('a', 'to', 'b'), ('b', 'back', 'a'), ('c', 'to', 'b')]
        self.widths = dict(a=3, b=2, c=4)
        self.edges = {
            self.relations[0]: torch.tensor([[0, 1, 2, 3, 0, 2], [0, 1, 2, 1, 2, 1]]),
            self.relations[1]: torch.tensor([[0, 1, 2, 1], [0, 1, 2, 3]]),
            self.relations[2]: torch.tensor([[0, 1, 1], [1, 0, 2]]),
        }
        self.rows = dict(a=4, b=3, c=2)

    def pair(self, **kwargs):
        model = HANConv(self.widths, 6, (self.nodes, self.relations), **kwargs)
        oracle = TorchHAN(self.widths, 6, (self.nodes, self.relations), **kwargs)
        self.copy(model, oracle)
        return model, oracle

    def compare(self, model, oracle, inputs, edges, *, strided=False):
        native_inputs = {name: native(value, True) for name, value in inputs.items()}
        x = {name: value.transpose(0, 1) for name, value in native_inputs.items()} if strided else native_inputs
        tx = {name: value.transpose(0, 1) for name, value in inputs.items()} if strided else inputs
        actual, semantic = model(x, {key: native(value) for key, value in edges.items()}, True)
        expected, reference_semantic = oracle(tx, edges, True)
        self.assertEqual(list(actual), list(expected)); self.assertEqual(list(semantic), list(reference_semantic))
        for key in actual:
            self.assertEqual(actual[key] is None, expected[key] is None)
            self.assertEqual(semantic[key] is None, reference_semantic[key] is None)
        values = [value.flatten() for value in actual.values() if value is not None]
        if not values:
            self.assertTrue(all(value is None for value in actual.values()))
            return actual, semantic
        target = [value.flatten() for value in expected.values() if value is not None]
        weights = [value for value in semantic.values() if value is not None]
        reference_weights = [value for value in reference_semantic.values() if value is not None]
        self.gradients(model, oracle, fr.cat(values, 0), torch.cat(target, 0), [(name, value, inputs[name]) for name, value in native_inputs.items()],
                       alpha=fr.cat(weights, 0) if weights else None, reference_alpha=torch.cat(reference_weights, 0) if weights else None)
        return actual, semantic

    def test_actual_pyg_relations_multihead_semantic_weights_strided_inputs_all_gradients(self):
        for heads, slope in ((1, .2), (2, .11), (3, -.13)):
            model, oracle = self.pair(heads=heads, negative_slope=slope)
            inputs = {name: torch.randn(self.widths[name], self.rows[name], requires_grad=True) for name in reversed(self.nodes)}
            output, attention = self.compare(model, oracle, inputs, self.edges, strided=True)
            self.assertIsNone(output['c']); self.assertEqual(attention['a'].shape, [1]); self.assertEqual(attention['b'].shape, [2])
            self.assertGreater(sum(abs(value) for value in model.q.tensor().grad.flatten().tolist()), 0.)

        relation = ('a', 'self', 'a')
        model, oracle = HANConv(3, 6, (['a'], [relation]), heads=2), TorchHAN(3, 6, (['a'], [relation]), heads=2)
        self.copy(model, oracle)
        edges = self.edges[self.relations[0]]
        topology = fr.graph.EdgeIndex(4, edges[0].tolist(), edges[1].tolist())
        x = torch.randn(4, 3, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model({'a': nx}, {relation: topology})['a'], oracle({'a': x}, {relation: edges})['a'], [('x', nx, x)])

    def test_dictionary_order_missing_relations_empty_targets_and_no_incoming_types(self):
        model, oracle = self.pair(heads=2)
        inputs = {name: torch.randn(self.rows[name], self.widths[name]) for name in self.nodes}
        ni = {key: native(value) for key, value in inputs.items()}
        first, attn1 = model(ni, {key: native(value) for key, value in self.edges.items()}, True)
        reverse = dict(reversed(list(self.edges.items())))
        second, attn2 = model(ni, {key: native(value) for key, value in reverse.items()}, True)
        self.close(first['b'], helpers.torch_value(second['b']))
        self.close(attn1['b'], helpers.torch_value(attn2['b']).flip(0))
        for rows_a, rows_b, supplied in ((4, 3, True), (0, 0, True), (0, 3, True), (4, 3, False)):
            model, oracle = self.pair(heads=2)
            inputs = dict(a=torch.randn(rows_a, 3, requires_grad=True), b=torch.randn(rows_b, 2, requires_grad=True))
            edges = {self.relations[0]: torch.empty((2, 0), dtype=torch.long)} if supplied else {}
            output, attention = self.compare(model, oracle, inputs, edges)
            self.assertIsNone(output['a']); self.assertIsNone(attention['a'])
            if supplied and rows_b == 0: self.assertIsNone(attention['b'])
            self.assertIsNone(model.proj['c'].weight.tensor().grad)
        model = HANConv(3, 6, (['a'], []))
        self.assertEqual(model({}, {}, True), ({}, {}))
        self.assertEqual(model({'a': fr.Tensor.ones([2, 3])}, {}, True), ({'a': None}, {'a': None}))

    def test_independent_same_relation_dropout_masks_and_eval_endpoint(self):
        model, oracle = self.pair(heads=2, dropout=.3)
        sampler = fr.nn.Dropout(.3, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(model.attn_dropout.rng_state)
        calls = []
        def dropout(value, p=.5, training=True, inplace=False):
            self.assertEqual(p, .3); self.assertTrue(training); self.assertFalse(inplace)
            calls.append(tuple(value.shape))
            return value * helpers.torch_value(sampler(fr.Tensor.ones(list(value.shape))))
        inputs = {name: torch.randn(self.rows[name], self.widths[name], requires_grad=True) for name in self.nodes}
        with patch('torch.nn.functional.dropout', side_effect=dropout): self.compare(model, oracle, inputs, self.edges)
        self.assertEqual(calls, [(6, 2), (4, 2), (3, 2)])
        self.assertEqual(model.attn_dropout.rng_state.tolist(), sampler.rng_state.tolist())
        model.eval(); oracle.eval(); before = model.attn_dropout.rng_state.tolist()
        values, attention = model({key: native(value) for key, value in inputs.items()}, {key: native(value) for key, value in self.edges.items()}, True)
        expected, weights = oracle(inputs, self.edges, True)
        for key in ('a', 'b'): self.close(values[key], expected[key]); self.close(attention[key], weights[key])
        self.assertEqual(model.attn_dropout.rng_state.tolist(), before)
        model, oracle = self.pair(heads=3, dropout=1.)
        self.compare(model, oracle, {name: torch.randn(self.rows[name], self.widths[name], requires_grad=True) for name in self.nodes}, self.edges)

    def test_initializer_glorot_shapes_reset_and_immutable_metadata(self):
        widths, nodes, relations = dict(self.widths), list(self.nodes), list(self.relations)
        model = HANConv(widths, 6, (nodes, relations), heads=2, dropout=.2)
        widths['a'] = 99; nodes.append('d'); relations.reverse()
        self.assertEqual(model.in_channels, self.widths)
        self.assertEqual(model.metadata, (tuple(self.nodes), tuple(self.relations)))
        public = model.in_channels; public['a'] = 77; self.assertEqual(model.in_channels, self.widths)
        for name, layer in model.proj.items():
            self.assertTrue(all(abs(value) <= self.widths[name] ** -.5 + 1e-7 for value in layer.weight.tensor().flatten().tolist()))
            self.assertTrue(any(value != 0. for value in layer.bias.tensor().tolist()))
        stream = model.attn_dropout.rng_state.tolist()
        with fr.no_grad():
            for parameter in model.parameters(): parameter.tensor().copy_(fr.Tensor.ones(parameter.tensor().shape) * 19.)
        model.reset_parameters()
        self.assertEqual(stream, model.attn_dropout.rng_state.tolist())
        for parameters in (model.lin_src, model.lin_dst):
            for parameter in parameters.values():
                self.assertEqual(parameter.tensor().shape, [1, 2, 3])
                self.assertTrue(all(abs(value) <= math.sqrt(6. / 5.) + 1e-7 for value in parameter.tensor().flatten().tolist()))
        self.assertTrue(all(abs(value) <= math.sqrt(6. / 7.) + 1e-7 for value in model.q.tensor().flatten().tolist()))
        with self.assertRaises(AttributeError): model.metadata = ((), ())

    def test_serialized_shared_relation_parameters_dropout_optimizer_rng_continuation(self):
        def factory(slope=.2):
            model = HANConv(self.widths, 6, (self.nodes, self.relations), heads=2, negative_slope=slope, dropout=.25)
            key = '__'.join(self.relations[0]); model.lin_dst[key] = model.lin_src[key]
            model.proj['c'].bias.trainable = False
            return model
        def step(model, optimizer, rng):
            inputs = {name: fr.Tensor.randn([self.rows[name], self.widths[name]], generator=rng) for name in self.nodes}
            output, semantic = model(inputs, {key: native(value) for key, value in self.edges.items()}, True)
            values = fr.cat([value.flatten() for value in output.values() if value is not None], 0)
            weights = fr.cat([value for value in semantic.values() if value is not None], 0)
            loss = values.square().mean() + weights.square().mean(); observed = values.tolist(), weights.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad(); return observed
        model, rng = factory(), fr.Generator(457)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'han': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'han': other_opt}, generator=other_rng)
        key = '__'.join(self.relations[0]); self.assertIs(other.lin_src[key], other.lin_dst[key])
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'han': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'han': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed, changed_rng = factory(.3), fr.Generator(1)
        changed_opt = fr.optim.AdamW(changed.parameters())
        def state(): return {key: value.tolist() for key, value in fr.checkpoint.snapshot_training(changed, optimizers={'han': changed_opt}, generator=changed_rng).tensors().items()}
        before = state()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'han': changed_opt}, generator=changed_rng)
        self.assertEqual(before, state())

    def test_preflight_all_dictionary_shapes_topology_and_capacity_before_streams(self):
        model = HANConv(self.widths, 6, (self.nodes, self.relations), heads=2, dropout=.3)
        inputs = {name: fr.Tensor.ones([self.rows[name], self.widths[name]]) for name in self.nodes}
        edges = {key: native(value) for key, value in self.edges.items()}
        before = model.attn_dropout.rng_state.tolist()
        bad_inputs = dict(inputs); bad_inputs['c'] = fr.Tensor.ones([2, 3])
        bad_edges = dict(edges); bad_edges[self.relations[2]] = fr.Tensor.from_i64([2, 0], [2, 1])
        for x, e in ((bad_inputs, edges), (inputs, bad_edges), ({'a': inputs['a']}, edges), (dict(inputs, unknown=inputs['a']), {}), (inputs, {('a', 'new', 'b'): edges[self.relations[0]]})):
            with patch.object(model.proj['a'], 'forward', side_effect=AssertionError('projection before metadata validation')):
                with self.assertRaises((ValueError, TypeError)): model(x, e)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(inputs, edges))
        with self.assertRaises(TypeError): model(inputs, edges, return_semantic_attention_weights=1)
        self.assertEqual(before, model.attn_dropout.rng_state.tolist())
        huge = fr.Tensor.ones([1, 3]).expand([sys.maxsize // 24 + 1, 3])
        with patch.object(model.proj['a'], 'forward', side_effect=AssertionError('projection before capacity validation')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model({'a': huge}, {})
        huge_edges = fr.Tensor.from_i64([0], [1, 1]).expand([2, sys.maxsize // 16 + 1])
        with patch.object(fr.Tensor, 'tolist', side_effect=AssertionError('topology read before capacity validation')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(inputs, {self.relations[0]: huge_edges})
        self.assertEqual(before, model.attn_dropout.rng_state.tolist())
        for kwargs in (dict(heads=4), dict(heads=0), dict(dropout=1.1), dict(negative_slope=float('nan'))):
            with self.assertRaises(ValueError): HANConv(self.widths, 6, (self.nodes, self.relations), **kwargs)
        for widths, meta in (({'a': 3}, (self.nodes, self.relations)), (3, (['a', 'a'], [])), (3, (['a'], [('a', 'to', 'b')])), (3, (['a'], [('a', 'x.y', 'a')]))):
            with self.assertRaises(ValueError): HANConv(widths, 6, meta)
        with self.assertRaises(NotImplementedError): HANConv(self.widths, 6, (self.nodes, self.relations), flow='target_to_source')
        with patch.object(fr.Tensor, 'zeros', side_effect=AssertionError('allocation before capacity validation')):
            with self.assertRaisesRegex(ValueError, 'capacity'): HANConv(3, sys.maxsize // 8 + 1, (['a'], []))


if __name__ == '__main__': unittest.main(verbosity=2)
