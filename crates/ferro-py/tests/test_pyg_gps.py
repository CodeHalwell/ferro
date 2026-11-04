"""Actual PyG GPS local/global attention, graph normalization and native state."""
import math
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import GPSConv as TorchGPS, GINEConv as TorchGINE
import ferro as fr
from ferro.graph import GPSConv
import test_pyg_modulated as helpers

native = helpers.native


class CountingConv(fr.nn.Module):
    bad_rows: bool = False

    def build(self):
        self.lin = fr.nn.Linear(4, 4)
        self._calls = 0

    def reset_parameters(self):
        with fr.no_grad(): self.lin.weight.tensor().copy_(fr.Tensor.ones([4, 4]))

    def forward(self, x, edge_index, **kwargs):
        self._calls += 1
        return self.lin(x.narrow(0, 0, 1) if self.bad_rows else x)


class GPSTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2267)
        self.edge = torch.tensor([[0, 1, 2, 3, 4, 5, 1], [1, 2, 0, 4, 5, 3, 3]])
        self.batch = torch.tensor([0, 0, 0, 2, 2, 2])

    def pair(self, local=True, **kwargs):
        conv = fr.graph.GINEConv(fr.nn.Linear(4, 4), edge_dim=2) if local else None
        oracle_conv = TorchGINE(torch.nn.Linear(4, 4), edge_dim=2) if local else None
        model, oracle = GPSConv(4, conv, heads=2, **kwargs), TorchGPS(4, oracle_conv, heads=2, **kwargs)
        self.copy(model, oracle)
        return model, oracle

    def test_actual_pyg_optional_local_gine_all_norms_strided_input_edge_and_parameter_gradients(self):
        for local in (False, True):
            for norm in (None, 'batch_norm', 'layer_norm', 'graph_norm'):
                model, oracle = self.pair(local, norm=norm)
                base = torch.randn(4, 6, requires_grad=True)
                attrs = torch.randn(7, 2, requires_grad=True)
                nb, na = native(base, True), native(attrs, True)
                actual = model(nb.transpose(0, 1), native(self.edge), native(self.batch), edge_attr=na)
                expected = oracle(base.transpose(0, 1), self.edge, self.batch, edge_attr=attrs)
                self.gradients(model, oracle, actual, expected, [('x', nb, base), ('edge_attr', na, attrs)])
                for number in (1, 2, 3):
                    layer, reference = getattr(model, 'norm' + str(number)), getattr(oracle, 'norm' + str(number))
                    if norm == 'batch_norm':
                        self.close(layer.module.running_mean, reference.module.running_mean)
                        self.close(layer.module.running_var, reference.module.running_var)
                        self.assertEqual(layer.module.num_batches_tracked.item(), reference.module.num_batches_tracked.item())

    def test_single_graph_activation_norm_options_extra_attention_tokens_and_eval(self):
        for options in (dict(norm=None, act='gelu', attn_kwargs=dict(bias=False)),
                        dict(norm='layer_norm', norm_kwargs=dict(mode='node', affine=False), act='silu'),
                        dict(norm='batch_norm', norm_kwargs=dict(momentum=None, affine=False), attn_kwargs=dict(add_bias_kv=True, add_zero_attn=True)),
                        dict(norm='batch_norm', norm_kwargs=dict(track_running_stats=False), act='leaky_relu', act_kwargs=dict(negative_slope=.13))):
            model, oracle = self.pair(False, **options)
            x = torch.randn(6, 4, requires_grad=True); nx = native(x, True)
            self.gradients(model, oracle, model(nx, None), oracle(x, None), [('x', nx, x)])
            model.eval(); oracle.eval()
            self.close(model(native(x), None), oracle(x.detach(), None))

    def test_independent_same_mask_attention_and_all_residual_mlp_dropouts(self):
        for local in (False, True):
            model, oracle = self.pair(local, norm='layer_norm', dropout=.25, attn_kwargs=dict(dropout=.2, add_bias_kv=True, add_zero_attn=True))
            outer = [model.local_dropout] if local else []
            outer += [model.global_dropout, model.mlp[2], model.mlp[4]]
            samplers = []
            for layer in outer + [model.attn.attn_dropout]:
                sampler = fr.nn.Dropout(layer.p, seed=1)
                with fr.no_grad(): sampler.rng_state.copy_(layer.rng_state)
                samplers.append(sampler)
            def attention(layer, query, key, value, key_padding_mask=None, need_weights=True, **kwargs):
                batch, length, width = query.shape
                projected = [torch.nn.functional.linear(tensor, layer.in_proj_weight[i * width:(i + 1) * width], layer.in_proj_bias[i * width:(i + 1) * width]) for i, tensor in enumerate((query, key, value))]
                if layer.bias_k is not None:
                    projected[1] = torch.cat([projected[1], layer.bias_k.expand(batch, 1, width)], 1)
                    projected[2] = torch.cat([projected[2], layer.bias_v.expand(batch, 1, width)], 1)
                    key_padding_mask = torch.cat([key_padding_mask, torch.zeros(batch, 1, dtype=torch.bool)], 1)
                if layer.add_zero_attn:
                    for i in (1, 2): projected[i] = torch.cat([projected[i], torch.zeros(batch, 1, width)], 1)
                    key_padding_mask = torch.cat([key_padding_mask, torch.zeros(batch, 1, dtype=torch.bool)], 1)
                q, k, v = [tensor.reshape(batch, -1, layer.num_heads, width // layer.num_heads).transpose(1, 2) for tensor in projected]
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // layer.num_heads)
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], -float('inf'))
                live = ~torch.isneginf(scores).all(-1, keepdim=True)
                weights = torch.where(live, torch.softmax(torch.where(live, scores, torch.zeros_like(scores)), -1), torch.zeros_like(scores))
                source = k.shape[-2]
                multiplier = helpers.torch_value(samplers[-1](fr.Tensor.ones([batch * layer.num_heads, length, source]))).reshape(batch, layer.num_heads, length, source)
                output = ((weights * multiplier) @ v).transpose(1, 2).reshape(batch, length, width)
                return torch.nn.functional.linear(output, layer.out_proj.weight, layer.out_proj.bias), None
            oracle.attn.forward = types.MethodType(attention, oracle.attn)
            cursor = [0]
            original_dropout = torch.nn.functional.dropout
            def dropout(value, p=.5, training=True, inplace=False):
                if p != .25: return original_dropout(value, p, training, inplace)
                sampler = samplers[cursor[0]]; cursor[0] += 1
                return value * helpers.torch_value(sampler(fr.Tensor.ones(list(value.shape))))
            x, attrs = torch.randn(6, 4, requires_grad=True), torch.randn(7, 2, requires_grad=True)
            nx, na = native(x, True), native(attrs, True)
            actual = model(nx, native(self.edge), native(self.batch), edge_attr=na)
            with patch('torch.nn.functional.dropout', side_effect=dropout):
                expected = oracle(x, self.edge, self.batch, edge_attr=attrs)
            self.gradients(model, oracle, actual, expected, [('x', nx, x), ('edge_attr', na, attrs)])
            self.assertEqual(cursor[0], len(outer))
            for layer, sampler in zip(outer + [model.attn.attn_dropout], samplers): self.assertEqual(layer.rng_state.tolist(), sampler.rng_state.tolist())
            model.eval()
            before = [layer.rng_state.tolist() for layer in outer + [model.attn.attn_dropout]]
            model(nx, native(self.edge), native(self.batch), edge_attr=na)
            self.assertEqual(before, [layer.rng_state.tolist() for layer in outer + [model.attn.attn_dropout]])

    def test_saved_bn_counters_shared_parameters_dropout_optimizer_and_rng(self):
        def factory(dropout=.2):
            conv = fr.graph.GINEConv(fr.nn.Linear(4, 4), edge_dim=2)
            model = GPSConv(4, conv, heads=2, dropout=dropout, norm_kwargs=dict(momentum=None), attn_kwargs=dict(dropout=.15))
            model.attn.out_proj.weight = model.conv.nn.weight
            model.norm3.module.weight = model.norm2.module.weight
            model.global_dropout = model.local_dropout
            return model
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([6, 4], generator=generator), native(self.edge), native(self.batch), edge_attr=fr.Tensor.randn([7, 2], generator=generator))
            loss = output.square().mean() + .13 * output.mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(359)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'gps': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'gps': other_opt}, generator=other_rng)
        self.assertIs(other.attn.out_proj.weight, other.conv.nn.weight)
        self.assertIs(other.norm2.module.weight, other.norm3.module.weight)
        self.assertIs(other.local_dropout, other.global_dropout)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'gps': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'gps': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        self.assertEqual([getattr(other, 'norm' + str(i)).module.num_batches_tracked.item() for i in (1, 2, 3)], [4, 4, 4])
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'gps': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_constructor_preserves_local_module_and_reset_matches_torch_scope(self):
        conv = CountingConv()
        before = conv.lin.weight.tensor().tolist()
        conv.eval()
        model = GPSConv(4, conv, dropout=.3, attn_kwargs=dict(dropout=.2, add_bias_kv=True), norm='layer_norm')
        self.assertIs(model.conv, conv)
        self.assertFalse(conv.training)
        self.assertEqual(before, conv.lin.weight.tensor().tolist())
        attention_weight = model.attn.out_proj.weight.tensor().tolist()
        streams = [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)]
        model.reset_parameters()
        self.assertEqual(conv.lin.weight.tensor().tolist(), [[1.] * 4] * 4)
        self.assertEqual(attention_weight, model.attn.out_proj.weight.tensor().tolist())
        self.assertEqual(streams, [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)])
        self.assertEqual(model.attn.in_proj_bias.tensor().tolist(), [0.] * 12)
        self.assertEqual(model.attn.out_proj.bias.tensor().tolist(), [0.] * 4)
        for i in (0, 3):
            layer = model.mlp[i]
            self.assertTrue(all(abs(value) <= layer.in_features ** -.5 for value in layer.weight.tensor().flatten().tolist()))
            self.assertTrue(any(value != 0. for value in layer.bias.tensor().tolist()))

    def test_global_metadata_capacity_and_bn_singleton_failures_precede_local_callback(self):
        conv = CountingConv()
        model = GPSConv(4, conv, heads=2, dropout=.2, attn_kwargs=dict(dropout=.2))
        x = fr.Tensor.ones([6, 4])
        def state(): return conv._calls, [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)], [getattr(model, 'norm' + str(i)).module.num_batches_tracked.item() for i in (1, 2, 3)]
        before = state()
        for call in (lambda: model(x, native(self.edge), [0, 1, 0, 1, 1, 1]), lambda: model(fr.Tensor.ones([6, 3]), native(self.edge)),
                     lambda: model(x, fr.Tensor.from_i64([0, 7], [2, 1])), lambda: model(x, native(self.edge), [0] * 5),
                     lambda: model(fr.Tensor.ones([1, 4]), fr.Tensor.from_i64([0, 0], [2, 1])),
                     lambda: model(fr.Tensor.zeros([0, 4]), fr.Tensor.from_i64([], [2, 0]))):
            with self.assertRaises(ValueError): call()
            self.assertEqual(before, state())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, native(self.edge)))
        with patch.object(conv, 'forward', side_effect=AssertionError('local callback reached before capacity validation')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(x, native(self.edge), [0, 0, 0, 0, 0, sys.maxsize // 8])
        self.assertEqual(before, state())
        with self.assertRaises(ValueError): GPSConv(4, CountingConv(bad_rows=True), norm=None)(x, native(self.edge))
        single = GPSConv(4, None, norm_kwargs=dict(allow_single_element=True))
        self.assertEqual(single(fr.Tensor.ones([1, 4]), None).shape, [1, 4])
        self.assertEqual(single.norm2.module.num_batches_tracked.item(), 0)
        for options in (dict(attn_type='performer'), dict(norm='instance_norm'), dict(act='unsupported')):
            with self.assertRaises(NotImplementedError): GPSConv(4, None, **options)
        with self.assertRaises(ValueError): GPSConv(4, None, attn_kwargs=dict(kdim=3))
        with self.assertRaises(ValueError): GPSConv(4, None, heads=3)


if __name__ == '__main__': unittest.main(verbosity=2)
