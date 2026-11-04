"""Actual PyG set-attention composition and independent fixed-mask references."""
import math
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import SetTransformerAggregation as TorchSet, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple
import ferro as fr
from ferro.graph import SetTransformerAggregation
import test_pyg_modulated as helpers

native = helpers.native


class SetTransformerTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1973)
        self.index = torch.tensor([0, 0, 1, 2, 2, 2, 4])

    def test_actual_pyg_blocks_heads_norm_concat_truncation_and_all_gradients(self):
        cases = [({}, None), (dict(num_seed_points=3, heads=2), 2),
                 (dict(num_seed_points=2, heads=2, layer_norm=True, concat=False, num_encoder_blocks=2, num_decoder_blocks=2), 5),
                 (dict(num_encoder_blocks=0, num_decoder_blocks=0), 1)]
        for options, limit in cases:
            model, oracle = SetTransformerAggregation(4, **options), TorchSet(4, **options)
            self.copy(model, oracle)
            base = torch.randn(4, 7, requires_grad=True)
            nb = native(base, True)
            actual = model(nb.transpose(0, 1), native(self.index), dim_size=6, max_num_elements=limit)
            expected = oracle(base.transpose(0, 1), self.index, dim_size=6, max_num_elements=limit)
            self.assertEqual(actual.shape[1], model.get_out_channels(4))
            self.gradients(model, oracle, actual, expected, [('x', nb, base)])
            if limit == 1: self.assertEqual(nb.grad.transpose(0, 1).index_select(0, [1, 4, 5]).tolist(), [[0.] * 4] * 3)

    def test_empty_groups_explicit_empty_input_and_permutation_invariance(self):
        for rows, groups, index, limit in ((0, 3, torch.empty(0, dtype=torch.long), 2), (7, 6, self.index, None), (7, None, None, None)):
            model, oracle = SetTransformerAggregation(4, 2, heads=2, layer_norm=True), TorchSet(4, 2, heads=2, layer_norm=True)
            self.copy(model, oracle)
            x = torch.randn(rows, 4, requires_grad=True)
            nx = native(x, True)
            actual = model(nx, None if index is None else native(index), dim_size=groups, max_num_elements=limit)
            expected = oracle(x, index, dim_size=groups, max_num_elements=limit)
            self.gradients(model, oracle, actual, expected, [('x', nx, x)])
        model = SetTransformerAggregation(4, 2, heads=2)
        x = torch.randn(7, 4)
        before = model(native(x), native(self.index), dim_size=6)
        permuted = x[[1, 0, 2, 5, 3, 4, 6]]
        self.close(model(native(permuted), native(self.index), dim_size=6), helpers.torch_value(before))
        oracle = TorchSet(4, 2, heads=2)
        self.copy(model, oracle)
        nonfinite = torch.full((7, 4), float('nan'))
        actual, expected = model(native(nonfinite), native(self.index), dim_size=6), oracle(nonfinite, self.index, dim_size=6)
        self.close(actual, expected)
        self.assertTrue(torch.isfinite(helpers.torch_value(actual)).all())

    def test_matched_dropout_independent_torch_attention_algebra_all_gradients(self):
        model, oracle = SetTransformerAggregation(4, 2, heads=2, layer_norm=True, dropout=.25), TorchSet(4, 2, heads=2, layer_norm=True, dropout=.25)
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

    def test_width_routing_in_multi_and_simple_composition(self):
        options = dict(aggrs_kwargs=[{}, dict(channels=4, num_seed_points=2, heads=2)], mode='proj', mode_kwargs=dict(in_channels=[4, 8], out_channels=3))
        model = fr.graph.MultiAggregation(['mean', 'set_transformer'], **options)
        oracle = TorchMulti(['mean', 'set_transformer'], **options)
        self.copy(model, oracle)
        x = torch.randn(7, 4, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model = fr.graph.SimpleConv('set_transformer', 'cat', aggr_kwargs=dict(channels=4, num_seed_points=2))
        oracle = TorchSimple('set_transformer', 'cat', aggr_kwargs=dict(channels=4, num_seed_points=2))
        self.copy(model, oracle)
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        x, target = torch.randn(6, 4, requires_grad=True), torch.randn(6, 1, requires_grad=True)
        weight = (torch.rand(7) + .5).requires_grad_()
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight), [('x', nx, x), ('target', nt, target), ('weight', nw, weight)])

    def test_saved_alias_configuration_optimizer_dropout_rng_and_reset(self):
        def factory(dropout=.2):
            model = SetTransformerAggregation(4, 2, heads=2, layer_norm=True, dropout=dropout)
            model.encoders[0].mab.lin.weight = model.pma.lin.weight
            return model
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([7, 4], generator=generator), native(self.index), dim_size=6)
            loss = output.square().mean() + output.mean() * .13
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(277)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'set': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'set': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.encoders[0].mab.lin.weight, other.pma.lin.weight)
        left = fr.checkpoint.snapshot_training(model, optimizers={'set': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'set': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'set': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])
        streams, preserved = {}, {}
        for name, layer in model.named_modules():
            if isinstance(layer, fr.nn.MultiheadAttention):
                streams[name] = layer.attn_dropout.rng_state.tolist()
                preserved[name] = layer.out_proj.weight.tensor().tolist()
        model.reset_parameters()
        bound = math.sqrt(6. / (3 * 4))
        self.assertTrue(all(abs(value) <= bound for value in model.pma.seed.tensor().flatten().tolist()))
        self.assertTrue(any(value != 0. for value in model.pma.lin.bias.tensor().tolist()))
        for name, layer in model.named_modules():
            if isinstance(layer, fr.nn.MultiheadAttention):
                self.assertEqual(layer.out_proj.weight.tensor().tolist(), preserved[name])
                self.assertEqual(layer.attn_dropout.rng_state.tolist(), streams[name])
                self.assertEqual(layer.in_proj_bias.tensor().tolist(), [0.] * 12)
                self.assertEqual(layer.out_proj.bias.tensor().tolist(), [0.] * 4)
            elif isinstance(layer, fr.nn.LayerNorm):
                self.assertEqual(layer.weight.tensor().tolist(), [1.] * 4)
                self.assertEqual(layer.bias.tensor().tolist(), [0.] * 4)

    def test_validation_capacity_and_invalid_calls_preserve_all_attention_streams(self):
        for options in (dict(channels=0), dict(channels=3, heads=2), dict(channels=4, num_seed_points=0),
                        dict(channels=4, num_encoder_blocks=-1), dict(channels=4, concat=1), dict(channels=4, dropout=float('nan'))):
            with self.assertRaises((ValueError, TypeError)): SetTransformerAggregation(**options)
        model, x = SetTransformerAggregation(4, heads=2, dropout=.2), fr.Tensor.ones([7, 4])
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
        with patch('ferro._graph_set_transformer.cat', side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                model(fr.Tensor.ones([1, 4]), [0], max_num_elements=math.isqrt(sys.maxsize // 8) + 1)
        self.assertEqual(before, streams())
        for groups, limit in ((0, 2), (3, None)):
            with self.assertRaises(RuntimeError): TorchSet(4, heads=2)(torch.empty(0, 4), torch.empty(0, dtype=torch.long), dim_size=groups, max_num_elements=limit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
