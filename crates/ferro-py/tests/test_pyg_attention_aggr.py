"""Attention readout composition with checked widths and registered streams."""
import tempfile
import unittest

import torch
from torch_geometric.nn import AttentionalAggregation as TorchAttention, SimpleConv as TorchSimple, GENConv as TorchGEN
from torch_geometric.nn.aggr import MultiAggregation as TorchMulti
import ferro as fr
from ferro.graph import AttentionalAggregation
import test_pyg_modulated as helpers

native = helpers.native


class LegacyGate(fr.nn.Module):
    def forward(self, x):
        return x.sin()


class LegacyDropoutGate(fr.nn.Module):
    def build(self):
        self.drop = fr.nn.Dropout(.3, seed=83)

    def forward(self, x):
        return self.drop(x).sin()


class AttentionAggregationTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2153)
        self.index = torch.tensor([2, 0, 2, 1, 2, 0, 4])
        self.edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])

    def test_actual_width_multi_simple_gen_and_all_input_parameter_edge_gradients(self):
        def aggregate(nn, constructor): return constructor(nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)), nn.Linear(3, 5))
        model = fr.graph.MultiAggregation(['mean', aggregate(fr.nn, AttentionalAggregation)], mode='cat')
        oracle = TorchMulti(['mean', aggregate(torch.nn, TorchAttention)], mode='cat')
        self.copy(model, oracle)
        x = torch.randn(7, 3, requires_grad=True); nx = native(x, True)
        named = fr.graph.MultiAggregation(['mean', 'attentional'], aggrs_kwargs=[{}, dict(gate_nn=fr.nn.Linear(3, 1))])
        self.assertEqual(named.get_out_channels(3), 6)
        self.assertEqual(model.get_out_channels(3), 8)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model = fr.graph.SimpleConv('attentional', 'cat', aggr_kwargs=dict(gate_nn=fr.nn.Sequential(fr.nn.Linear(3, 4), fr.nn.Tanh(), fr.nn.Linear(4, 1)), nn=fr.nn.Linear(3, 5)))
        oracle = TorchSimple(aggregate(torch.nn, TorchAttention), 'cat')
        self.copy(model, oracle)
        x, target, weight = torch.randn(6, 3, requires_grad=True), torch.randn(6, 2, requires_grad=True), torch.rand(7, requires_grad=True)
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(self.edge), nw), oracle((x, target), self.edge, weight), [('x', nx, x), ('target', nt, target), ('edge_weight', nw, weight)])
        model = fr.graph.GENConv(3, 3, aggr=aggregate(fr.nn, AttentionalAggregation), norm=None)
        oracle = TorchGEN(3, 3, aggr=aggregate(torch.nn, TorchAttention), norm=None)
        # Explicit width-routing extension: supply the projection that PyG
        # omits for arbitrary custom width-changing aggregation Modules.
        from torch_geometric.nn.dense.linear import Linear
        oracle.lin_aggr_out = Linear(5, 3, bias=False)
        self.copy(model, oracle)
        x = torch.randn(6, 3, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.edge)), oracle(x, self.edge), [('x', nx, x)])

    def test_featurewise_gating_same_mask_dropout_norm_and_empty_gradients(self):
        gate = fr.nn.Sequential(fr.nn.Linear(3, 4), fr.nn.Dropout(.25, seed=17), fr.nn.BatchNorm1d(4))
        values = fr.nn.Sequential(fr.nn.Linear(3, 4), fr.nn.Dropout(.2, seed=23))
        model = AttentionalAggregation(gate, values)
        oracle = TorchAttention(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Identity(), torch.nn.BatchNorm1d(4)),
                                torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Identity()))
        self.copy(model, oracle)
        masks = [fr.nn.Dropout(.25, seed=17), fr.nn.Dropout(.2, seed=23)]
        gate_mask = helpers.torch_value(masks[0](fr.Tensor.ones([7, 4])))
        value_mask = helpers.torch_value(masks[1](fr.Tensor.ones([7, 4])))
        oracle.gate_nn[1].register_forward_hook(lambda module, args, output: output * gate_mask)
        oracle.nn[1].register_forward_hook(lambda module, args, output: output * value_mask)
        x = torch.randn(7, 3, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        self.close(gate[2].running_mean, oracle.gate_nn[2].running_mean)
        self.close(gate[2].running_var, oracle.gate_nn[2].running_var)
        self.assertEqual(gate[1].rng_state.tolist(), masks[0].rng_state.tolist())
        self.assertEqual(values[1].rng_state.tolist(), masks[1].rng_state.tolist())
        for groups in (0, 4):
            model, oracle = AttentionalAggregation(fr.nn.Linear(3, 1), fr.nn.Linear(3, 5)), TorchAttention(torch.nn.Linear(3, 1), torch.nn.Linear(3, 5))
            self.copy(model, oracle)
            x = torch.empty(0, 3, requires_grad=True); nx = native(x, True)
            index = torch.empty(0, dtype=torch.long)
            self.gradients(model, oracle, model(nx, native(index), dim_size=groups), oracle(x, index, dim_size=groups), [('x', nx, x)])

    def test_legacy_custom_direct_forward_and_checked_composition_boundary(self):
        x = torch.randn(7, 3, requires_grad=True); nx = native(x, True)
        model = AttentionalAggregation(LegacyGate())
        class TorchGate(torch.nn.Module):
            def forward(self, x): return x.sin()
        oracle = TorchAttention(TorchGate())
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        with self.assertRaisesRegex(NotImplementedError, 'get_out_channels'): model.get_out_channels(3)
        with self.assertRaisesRegex(NotImplementedError, 'get_out_channels'): fr.graph.MultiAggregation([model, 'mean'])(native(x), native(self.index), dim_size=6)
        class DeclaredGate(fr.nn.Module):
            def get_out_channels(self, width): return width
            def forward(self, x): return x.sin()
        model = fr.graph.MultiAggregation([AttentionalAggregation(DeclaredGate()), 'mean'])
        self.assertEqual(model.get_out_channels(3), 6)
        self.assertEqual(model(native(x), native(self.index), dim_size=6).shape, [6, 6])

    def test_saved_composed_alias_optimizer_dropout_and_rng_continuation(self):
        def factory(p=.2):
            network = fr.nn.Sequential(fr.nn.Linear(3, 3), fr.nn.Tanh(), fr.nn.Dropout(p, seed=79))
            return fr.graph.SimpleConv(AttentionalAggregation(network, network))
        def step(model, optimizer, rng):
            output = model(fr.Tensor.randn([6, 3], generator=rng), native(self.edge))
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(79)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'attention': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'attention': other_opt}, generator=other_rng)
        self.assertIs(other.aggr_module.gate_nn, other.aggr_module.nn)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'attention': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'attention': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'attention': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_all_known_network_metadata_before_any_gate_or_value_stream(self):
        gate = fr.nn.Sequential(fr.nn.Linear(3, 1), fr.nn.Dropout(.2, seed=17))
        before = gate[1].rng_state.tolist()
        for network in (fr.nn.Linear(4, 2), fr.nn.LayerNorm(4), fr.nn.BatchNorm1d(3)):
            model = AttentionalAggregation(gate, network)
            x = fr.Tensor.ones([1 if isinstance(network, fr.nn.BatchNorm1d) else 7, 3])
            with self.assertRaises(ValueError): model(x)
            self.assertEqual(before, gate[1].rng_state.tolist())
        unknown_gate = LegacyDropoutGate()
        unknown_stream = unknown_gate.drop.rng_state.tolist()
        for network in (fr.nn.Linear(4, 3), fr.nn.BatchNorm1d(3)):
            model = AttentionalAggregation(unknown_gate, network)
            rows = 1 if isinstance(network, fr.nn.BatchNorm1d) else 7
            with self.assertRaises(ValueError): model(fr.Tensor.ones([rows, 3]))
            self.assertEqual(unknown_stream, unknown_gate.drop.rng_state.tolist())
        invalid = AttentionalAggregation(fr.nn.Sequential(fr.nn.Linear(3, 2), fr.nn.Dropout(.2, seed=19)), fr.nn.Linear(3, 4))
        stream = invalid.gate_nn[1].rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'gate width'): invalid(fr.Tensor.ones([7, 3]))
        self.assertEqual(stream, invalid.gate_nn[1].rng_state.tolist())
        first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=4, dropout=.2))
        combined = fr.graph.MultiAggregation([first, invalid])
        stream = first.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaises(ValueError): combined(fr.Tensor.ones([7, 3]))
        self.assertEqual(stream, first.multihead_attn.attn_dropout.rng_state.tolist())


if __name__ == '__main__': unittest.main(verbosity=2)
