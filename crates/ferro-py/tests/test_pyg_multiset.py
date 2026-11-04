"""Actual PyG multiset pooling, learned seeds and native attention state."""
import math
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import GraphMultisetTransformer as TorchMultiset, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple
import ferro as fr
from ferro.graph import GraphMultisetTransformer
import test_pyg_modulated as helpers

native = helpers.native


class GraphMultisetTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2179)
        self.index = torch.tensor([0, 0, 1, 2, 2, 2, 4])

    def test_actual_pyg_two_pooling_blocks_encoder_heads_norm_truncation_and_gradients(self):
        cases = [(dict(k=1), None), (dict(k=3, heads=2), 2),
                 (dict(k=2, heads=2, layer_norm=True, num_encoder_blocks=2), 5),
                 (dict(k=4, num_encoder_blocks=0), 1)]
        for options, limit in cases:
            model, oracle = GraphMultisetTransformer(4, **options), TorchMultiset(4, **options)
            self.copy(model, oracle)
            base = torch.randn(4, 7, requires_grad=True); nb = native(base, True)
            actual = model(nb.transpose(0, 1), native(self.index), dim_size=6, max_num_elements=limit)
            expected = oracle(base.transpose(0, 1), self.index, dim_size=6, max_num_elements=limit)
            self.assertEqual(actual.shape[1], model.get_out_channels(4))
            self.gradients(model, oracle, actual, expected, [('x', nb, base)])
            if limit == 1: self.assertEqual(nb.grad.transpose(0, 1).index_select(0, [1, 4, 5]).tolist(), [[0.] * 4] * 3)

    def test_empty_groups_inputs_permutation_and_nonfinite_semantics(self):
        for rows, groups, index, limit in ((0, 3, torch.empty(0, dtype=torch.long), 2), (7, 6, self.index, None), (7, None, None, None)):
            model, oracle = GraphMultisetTransformer(4, 2, heads=2, layer_norm=True), TorchMultiset(4, 2, heads=2, layer_norm=True)
            self.copy(model, oracle)
            x = torch.randn(rows, 4, requires_grad=True); nx = native(x, True)
            actual = model(nx, None if index is None else native(index), dim_size=groups, max_num_elements=limit)
            expected = oracle(x, index, dim_size=groups, max_num_elements=limit)
            self.gradients(model, oracle, actual, expected, [('x', nx, x)])
        model = GraphMultisetTransformer(4, 3, heads=2)
        x = torch.randn(7, 4)
        before = model(native(x), native(self.index), dim_size=6)
        self.close(model(native(x[[1, 0, 2, 5, 3, 4, 6]]), native(self.index), dim_size=6), helpers.torch_value(before))
        oracle = TorchMultiset(4, 3, heads=2); self.copy(model, oracle)
        nonfinite = torch.full((7, 4), float('nan'))
        actual = helpers.torch_value(model(native(nonfinite), native(self.index), dim_size=6))
        expected = oracle(nonfinite, self.index, dim_size=6)
        torch.testing.assert_close(actual, expected, equal_nan=True)
        self.assertTrue(torch.isnan(actual).any())
    def test_matched_dropout_independent_torch_attention_algebra_all_gradients(self):
        model, oracle = GraphMultisetTransformer(4, 2, heads=2, layer_norm=True, dropout=.25), TorchMultiset(4, 2, heads=2, layer_norm=True, dropout=.25)
        self.copy(model, oracle)
        references = dict(oracle.named_modules())
        samplers = []
        def reference_attention(sampler):
            def forward(layer, query, key, value, key_padding_mask=None, need_weights=True, **kwargs):
                batch, length, width = query.shape
                source, heads = key.shape[1], layer.num_heads
                def project(x, index):
                    value = torch.nn.functional.linear(x, layer.in_proj_weight[index * width:(index + 1) * width], layer.in_proj_bias[index * width:(index + 1) * width])
                    return value.reshape(batch, -1, heads, width // heads).transpose(1, 2)
                q, k, v = project(query, 0), project(key, 1), project(value, 2)
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
                if key_padding_mask is not None: scores = scores.masked_fill(key_padding_mask[:, None, None, :], -float('inf'))
                live = ~torch.isneginf(scores).all(-1, keepdim=True)
                weights = torch.where(live, torch.softmax(torch.where(live, scores, torch.zeros_like(scores)), -1), torch.zeros_like(scores))
                multiplier = torch.tensor(sampler(fr.Tensor.ones([batch * heads, length, source])).tolist()).reshape(batch, heads, length, source)
                output = ((weights * multiplier) @ v).transpose(1, 2).reshape(batch, length, width)
                return torch.nn.functional.linear(output, layer.out_proj.weight, layer.out_proj.bias), None
            return forward
        for name, layer in model.named_modules():
            if isinstance(layer, fr.nn.MultiheadAttention):
                sampler = fr.nn.Dropout(.25, seed=1)
                with fr.no_grad(): sampler.rng_state.copy_(layer.attn_dropout.rng_state)
                samplers.append((layer, sampler))
                references[name].forward = types.MethodType(reference_attention(sampler), references[name])
        x = torch.randn(7, 4, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        for layer, sampler in samplers: self.assertEqual(layer.attn_dropout.rng_state.tolist(), sampler.rng_state.tolist())
        model.eval()
        before = [layer.attn_dropout.rng_state.tolist() for layer, _ in samplers]
        model(nx, native(self.index), dim_size=6)
        self.assertEqual(before, [layer.attn_dropout.rng_state.tolist() for layer, _ in samplers])

    def test_width_composition_and_weighted_message_gradients(self):
        model = fr.graph.MultiAggregation(['mean', 'graph_multiset_transformer'], aggrs_kwargs=[{}, dict(channels=4, k=2, heads=2)], mode='sum')
        oracle = TorchMulti(['mean', TorchMultiset(4, 2, heads=2)], mode='sum')
        self.copy(model, oracle)
        x = torch.randn(7, 4, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model = fr.graph.SimpleConv('graph_multiset_transformer', 'cat', aggr_kwargs=dict(channels=4, k=3))
        oracle = TorchSimple(TorchMultiset(4, 3), 'cat')
        self.copy(model, oracle)
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        x, target = torch.randn(6, 4, requires_grad=True), torch.randn(6, 1, requires_grad=True)
        weight = (torch.rand(7) + .5).requires_grad_()
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight), [('x', nx, x), ('target', nt, target), ('weight', nw, weight)])

    def test_saved_alias_optimizer_dropout_rng_configuration_and_reset(self):
        def factory(dropout=.2):
            model = GraphMultisetTransformer(4, 2, heads=2, layer_norm=True, dropout=dropout)
            model.pma2.lin.weight = model.pma1.lin.weight
            return model
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([7, 4], generator=generator), native(self.index), dim_size=6)
            loss = output.square().mean() + output.mean() * .13
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(313)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'multiset': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'multiset': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.pma1.lin.weight, other.pma2.lin.weight)
        left = fr.checkpoint.snapshot_training(model, optimizers={'multiset': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'multiset': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'multiset': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])
        streams, weights = {}, {}
        for name, layer in model.named_modules():
            if isinstance(layer, fr.nn.MultiheadAttention):
                streams[name] = layer.attn_dropout.rng_state.tolist()
                weights[name] = layer.out_proj.weight.tensor().tolist()
        model.reset_parameters()
        for pool, count in ((model.pma1, 2), (model.pma2, 1)):
            self.assertTrue(all(abs(value) <= math.sqrt(6. / ((count + 1) * 4)) for value in pool.seed.tensor().flatten().tolist()))
            self.assertTrue(any(value != 0. for value in pool.lin.bias.tensor().tolist()))
        for name, layer in model.named_modules():
            if isinstance(layer, fr.nn.MultiheadAttention):
                self.assertEqual(layer.attn_dropout.rng_state.tolist(), streams[name])
                self.assertEqual(layer.out_proj.weight.tensor().tolist(), weights[name])
                self.assertEqual(layer.in_proj_bias.tensor().tolist(), [0.] * 12)
                self.assertEqual(layer.out_proj.bias.tensor().tolist(), [0.] * 4)

    def test_validation_derived_capacity_and_nested_stream_preflight(self):
        for options in (dict(channels=0, k=1), dict(channels=3, k=1, heads=2), dict(channels=4, k=0),
                        dict(channels=4, k=2, num_encoder_blocks=-1), dict(channels=4, k=2, layer_norm=1), dict(channels=4, k=2, dropout=float('nan'))):
            with self.assertRaises((ValueError, TypeError)): GraphMultisetTransformer(**options)
        model, x = GraphMultisetTransformer(4, 2, heads=2, dropout=.2), fr.Tensor.ones([7, 4])
        def streams(): return [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)]
        before = streams()
        for call in (lambda: model(x, [0, 1, 0, 2, 2, 2, 4]), lambda: model(x, native(self.index), dim_size=4),
                     lambda: model(fr.Tensor.ones([7, 3]), native(self.index)), lambda: model(x, max_num_elements=0),
                     lambda: model(x, ptr=fr.Tensor.from_i64([0, 7], [2])), lambda: model(x, dim=-1),
                     lambda: model(fr.Tensor.zeros([0, 4]), fr.Tensor.from_i64([], [0]), dim_size=0, max_num_elements=2),
                     lambda: model(fr.Tensor.zeros([0, 4]), fr.Tensor.from_i64([], [0]), dim_size=3)):
            with self.assertRaises((ValueError, NotImplementedError)): call()
            self.assertEqual(before, streams())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x))
        with patch('ferro._graph_multiset.cat', side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(fr.Tensor.ones([1, 4]), [0], max_num_elements=sys.maxsize // 8 + 1)
        with patch('ferro._graph_multiset._SetPMA', side_effect=AssertionError('parameter allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'): GraphMultisetTransformer(4, sys.maxsize // 16 + 1)
        first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=4, out_channels=4, dropout=.2))
        combined = fr.graph.MultiAggregation([first, model])
        stream = first.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'sorted'): combined(x, [0, 1, 0, 2, 2, 2, 4])
        self.assertEqual(stream, first.multihead_attn.attn_dropout.rng_state.tolist())
        self.assertEqual(before, streams())


if __name__ == '__main__': unittest.main(verbosity=2)
