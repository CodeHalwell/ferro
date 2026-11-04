"""Set2Set registered aggregation, matching recurrent dropout, and saved state."""
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import Set2Set as TorchSet, SimpleConv as TorchSimple, GENConv as TorchGEN
from torch_geometric.nn.aggr import MultiAggregation as TorchMulti
import ferro as fr
from ferro.graph import Set2Set
import test_pyg_modulated as helpers

native = helpers.native


class Set2SetIntegrationTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1997)
        self.index = torch.tensor([0, 0, 1, 2, 2, 2, 4])
        self.edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])

    def test_width_aware_multi_simple_and_gen_actual_pyg_gradients(self):
        for mode, options in (('cat', {}), ('proj', dict(mode_kwargs=dict(in_channels=[3, 6], out_channels=4)))):
            parameters = dict(aggrs_kwargs=[{}, dict(in_channels=3, processing_steps=3)])
            model = fr.graph.MultiAggregation(['mean', 'set2set'], mode=mode, **parameters, **options)
            oracle = TorchMulti(['mean', 'set2set'], mode=mode, **parameters, **options)
            self.copy(model, oracle)
            x = torch.randn(7, 3, requires_grad=True)
            nx = native(x, True)
            actual, expected = model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6)
            self.assertEqual(model.get_out_channels(3), expected.shape[1])
            self.gradients(model, oracle, actual, expected, [('x', nx, x)])
        model = fr.graph.SimpleConv('set2set', 'cat', aggr_kwargs=dict(in_channels=3, processing_steps=2))
        oracle = TorchSimple('set2set', 'cat', aggr_kwargs=dict(in_channels=3, processing_steps=2))
        self.copy(model, oracle)
        x, target = torch.randn(6, 3, requires_grad=True), torch.randn(6, 2, requires_grad=True)
        weight = (torch.rand(7) + .4).requires_grad_()
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(self.edge), nw), oracle((x, target), self.edge, weight),
                       [('x', nx, x), ('target', nt, target), ('weight', nw, weight)])
        options = dict(mode='proj', mode_kwargs=dict(in_channels=[3, 6], out_channels=4))
        model = fr.graph.GENConv(3, 3, aggr=fr.graph.MultiAggregation(['mean', Set2Set(3, 2)], **options), norm=None, num_layers=1, bias=True)
        oracle = TorchGEN(3, 3, aggr=TorchMulti(['mean', TorchSet(3, 2)], **options), norm=None, num_layers=1, bias=True)
        self.copy(model, oracle)
        x, attr = torch.randn(6, 3, requires_grad=True), torch.randn(7, 3, requires_grad=True)
        nx, na = native(x, True), native(attr, True)
        self.gradients(model, oracle, model(nx, native(self.edge), na), oracle(x, self.edge, attr), [('x', nx, x), ('edge_attr', na, attr)])

    def test_same_dropout_masks_torch_layer_recurrence_and_all_gradients(self):
        model, oracle = Set2Set(3, 3, num_layers=3, dropout=.25), TorchSet(3, 3, num_layers=3, dropout=.25)
        self.copy(model, oracle)
        samplers = []
        for dropout in model.lstm.interlayer_dropout:
            sampler = fr.nn.Dropout(.25, seed=1)
            with fr.no_grad(): sampler.rng_state.copy_(dropout.rng_state)
            samplers.append(sampler)
        def forward(lstm, x, state):
            h, c = state
            outputs_h, outputs_c = [], []
            current = x
            for layer_index in range(3):
                layer = torch.nn.LSTM(6 if layer_index == 0 else 3, 3)
                for name in dict(layer.named_parameters()):
                    setattr(layer, name, getattr(lstm, name.replace('_l0', '_l' + str(layer_index))))
                current, (next_h, next_c) = layer(current, (h[layer_index:layer_index + 1], c[layer_index:layer_index + 1]))
                outputs_h.append(next_h); outputs_c.append(next_c)
                if layer_index < 2:
                    current = current * torch.tensor(samplers[layer_index](fr.Tensor.ones(list(current.shape))).tolist())
            return current, (torch.cat(outputs_h, 0), torch.cat(outputs_c, 0))
        oracle.lstm.forward = types.MethodType(forward, oracle.lstm)
        x = torch.randn(7, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        for dropout, sampler in zip(model.lstm.interlayer_dropout, samplers): self.assertEqual(dropout.rng_state.tolist(), sampler.rng_state.tolist())

    def test_serialized_composed_dropout_alias_optimizer_and_generator_continuation(self):
        def factory(dropout=.2):
            recurrent = Set2Set(3, 3, num_layers=2, dropout=dropout)
            recurrent.lstm.weight_hh_l1 = recurrent.lstm.weight_ih_l1
            return fr.graph.SimpleConv(recurrent, 'cat')
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([6, 3], generator=generator), native(self.edge))
            self.assertEqual(output.shape, [6, 9])
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(281)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'set2set': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'set2set': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.aggr_module.lstm.weight_ih_l1, other.aggr_module.lstm.weight_hh_l1)
        left = fr.checkpoint.snapshot_training(model, optimizers={'set2set': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'set2set': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'set2set': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_metadata_and_changed_width_capacity_precede_dropout(self):
        model = Set2Set(3, 3, num_layers=2, dropout=.2)
        state = model.lstm.interlayer_dropout[0].rng_state.tolist()
        x = fr.Tensor.ones([7, 3])
        for call in (lambda: model(fr.Tensor.ones([7, 2]), native(self.index)), lambda: model(x, native(self.index), dim_size=4),
                     lambda: model(x, ptr=fr.Tensor.from_i64([0, 7], [2]), dim_size=2), lambda: model(x, dim=0.0)):
            with self.assertRaises((ValueError, NotImplementedError)): call()
            self.assertEqual(state, model.lstm.interlayer_dropout[0].rng_state.tolist())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x))
        self.assertEqual(state, model.lstm.interlayer_dropout[0].rng_state.tolist())
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            Set2Set(1, 2)(fr.Tensor.zeros([0, 1]), fr.Tensor.from_i64([], [0]), dim_size=huge.shape[0])
        attention = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=2, out_channels=6, dropout=.2))
        combined = fr.graph.MultiAggregation([attention, model], mode='sum')
        first_stream = attention.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'width'): combined(fr.Tensor.ones([7, 2]), native(self.index), dim_size=6)
        self.assertEqual(first_stream, attention.multihead_attn.attn_dropout.rng_state.tolist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
