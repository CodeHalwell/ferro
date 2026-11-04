"""Actual PyG DeepSets algebra, width metadata, and native continuation state."""
import sys
import tempfile
import unittest

import torch
from torch_geometric.nn.aggr import DeepSetsAggregation as TorchDeepSets, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple, GENConv as TorchGEN
import ferro as fr
from ferro.graph import DeepSetsAggregation
import test_pyg_modulated as helpers

native = helpers.native


class DeclaredNetwork(fr.nn.Module):
    channels: int
    behavior: str = 'normal'

    def build(self):
        self.lin = fr.nn.Linear(self.channels, self.channels)

    def get_out_channels(self, in_channels):
        if in_channels != self.channels: raise ValueError('wrong input width')
        return self.channels

    def forward(self, x):
        output = self.lin(x).sin()
        if self.behavior == 'rows': return output.narrow(0, 0, 1)
        if self.behavior == 'width': return output.narrow(1, 0, 1)
        return output


class TorchDeclared(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lin = torch.nn.Linear(channels, channels)

    def forward(self, x):
        return self.lin(x).sin()


class UnknownNetwork(fr.nn.Module):
    def forward(self, x):
        return x


class DeepSetsTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2131)
        self.index = torch.tensor([2, 0, 2, 1, 2, 0, 4])
        self.sorted_index = torch.tensor([0, 0, 1, 2, 2, 2, 4])
        self.ptr = torch.tensor([0, 2, 3, 6, 6, 7, 7])

    def test_actual_pyg_optional_networks_coo_csr_and_all_gradients(self):
        def networks(nn):
            return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4)), nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 2))
        for local_on, global_on in ((True, True), (True, False), (False, True), (False, False)):
            for csr in (False, True):
                local, global_ = networks(fr.nn)
                olocal, oglobal = networks(torch.nn)
                if not local_on:
                    global_, oglobal = fr.nn.Linear(3, 2), torch.nn.Linear(3, 2)
                model = DeepSetsAggregation(local if local_on else None, global_ if global_on else None)
                oracle = TorchDeepSets(olocal if local_on else None, oglobal if global_on else None)
                self.copy(model, oracle)
                base = torch.randn(3, 7, requires_grad=True)
                nb = native(base, True)
                args = dict(ptr=native(self.ptr)) if csr else dict(index=native(self.index), dim_size=6)
                targs = dict(ptr=self.ptr) if csr else dict(index=self.index, dim_size=6)
                actual, expected = model(nb.transpose(0, 1), **args), oracle(base.transpose(0, 1), **targs)
                self.assertEqual(actual.shape[1], model.get_out_channels(3))
                self.gradients(model, oracle, actual, expected, [('x', nb, base)])

    def test_empty_features_empty_groups_and_default_group(self):
        for rows, width, groups in ((0, 3, 0), (0, 3, 5), (4, 0, 5), (0, 0, 0)):
            for learned in (False, True) if width else (False,):
                model = DeepSetsAggregation(fr.nn.Linear(3, 4) if learned else None, fr.nn.Linear(4, 2) if learned else None)
                oracle = TorchDeepSets(torch.nn.Linear(3, 4) if learned else None, torch.nn.Linear(4, 2) if learned else None)
                self.copy(model, oracle)
                x = torch.randn(rows, width, requires_grad=True)
                index = torch.tensor([0, 2, 2, 3]) if rows else torch.empty(0, dtype=torch.long)
                nx = native(x, True)
                self.gradients(model, oracle, model(nx, native(index), dim_size=groups), oracle(x, index, dim_size=groups), [('x', nx, x)])
        for rows in (0, 4):
            x = torch.randn(rows, 3, requires_grad=True)
            nx = native(x, True)
            model, oracle = DeepSetsAggregation(), TorchDeepSets()
            self.gradients(model, oracle, model(nx), oracle(x), [('x', nx, x)])

    def test_preserved_constructor_state_recursive_reset_and_explicit_custom_metadata(self):
        local = fr.nn.Sequential(fr.nn.Linear(3, 3), fr.nn.LayerNorm(3), fr.nn.Dropout(.3, seed=43))
        local.eval()
        local[0].weight.trainable = False
        before = [p.tensor().tolist() for p in local.parameters()]
        stream = local[2].rng_state.tolist()
        model = DeepSetsAggregation(local, local)
        self.assertIs(model.local_nn, model.global_nn)
        self.assertEqual(before, [p.tensor().tolist() for p in model.parameters()])
        self.assertFalse(model.local_nn.training)
        self.assertFalse(model.local_nn[0].weight.trainable)
        model.reset_parameters()
        self.assertNotEqual(before[0], local[0].weight.tensor().tolist())
        self.assertEqual(stream, local[2].rng_state.tolist())
        self.assertEqual(local[1].weight.tensor().tolist(), [1.] * 3)
        model, oracle = DeepSetsAggregation(DeclaredNetwork(channels=3), DeclaredNetwork(channels=3)), TorchDeepSets(TorchDeclared(3), TorchDeclared(3))
        self.copy(model, oracle)
        x = torch.randn(7, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        with self.assertRaisesRegex(NotImplementedError, 'get_out_channels'): DeepSetsAggregation(UnknownNetwork())(native(x))
        for name in ('rows', 'width'):
            with self.assertRaises(ValueError): DeepSetsAggregation(DeclaredNetwork(channels=3, behavior=name))(native(x))
            with self.assertRaises(ValueError): DeepSetsAggregation(None, DeclaredNetwork(channels=3, behavior=name))(native(x), native(self.index))

    def test_norm_and_same_mask_dropout_all_gradients_and_running_state(self):
        local = fr.nn.Sequential(fr.nn.Linear(3, 4), fr.nn.Dropout(.25, seed=17), fr.nn.BatchNorm1d(4, momentum=None), fr.nn.Tanh())
        global_ = fr.nn.Sequential(fr.nn.Linear(4, 3), fr.nn.LayerNorm(3), fr.nn.Dropout(.2, seed=23))
        model = DeepSetsAggregation(local, global_)
        oracle = TorchDeepSets(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Identity(), torch.nn.BatchNorm1d(4, momentum=None), torch.nn.Tanh()),
                               torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3), torch.nn.Identity()))
        self.copy(model, oracle)
        masks = [fr.nn.Dropout(.25, seed=17), fr.nn.Dropout(.2, seed=23)]
        x = torch.randn(7, 3, requires_grad=True)
        nx = native(x, True)
        local_mask = helpers.torch_value(masks[0](fr.Tensor.ones([7, 4])))
        global_mask = helpers.torch_value(masks[1](fr.Tensor.ones([6, 3])))
        # Actual PyG reductions and original Torch parameters; masks come from
        # independent native streams, not the production forward's result.
        oracle.local_nn[1].register_forward_hook(lambda module, args, output: output * local_mask)
        oracle.global_nn[2].register_forward_hook(lambda module, args, output: output * global_mask)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        self.close(local[2].running_mean, oracle.local_nn[2].running_mean)
        self.close(local[2].running_var, oracle.local_nn[2].running_var)
        self.assertEqual(local[2].num_batches_tracked.item(), 1)
        self.assertEqual(local[1].rng_state.tolist(), masks[0].rng_state.tolist())
        self.assertEqual(global_[2].rng_state.tolist(), masks[1].rng_state.tolist())
        for norm, tnorm in ((fr.nn.RMSNorm(3), torch.nn.RMSNorm(3)), (fr.nn.LayerNorm((6, 3)), torch.nn.LayerNorm((6, 3)))):
            model, oracle = DeepSetsAggregation(None, norm), TorchDeepSets(None, tnorm)
            self.copy(model, oracle)
            x = torch.randn(7, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])

    def test_actual_width_composition_and_edge_gradients(self):
        def aggregate(nn, kind): return kind(nn.Sequential(nn.Linear(3, 5), nn.Tanh()), nn.Linear(5, 4))
        model = fr.graph.MultiAggregation(['mean', aggregate(fr.nn, DeepSetsAggregation)], mode='cat')
        oracle = TorchMulti(['mean', aggregate(torch.nn, TorchDeepSets)], mode='cat')
        self.copy(model, oracle)
        x = torch.randn(7, 3, requires_grad=True); nx = native(x, True)
        actual = model(nx, native(self.index), dim_size=6)
        self.assertEqual(fr.graph.MultiAggregation(['mean', 'deepsets']).get_out_channels(3), 6)
        self.assertEqual(model.get_out_channels(3), 7)
        self.gradients(model, oracle, actual, oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model = fr.graph.SimpleConv('deep_sets', 'cat', aggr_kwargs=dict(local_nn=fr.nn.Sequential(fr.nn.Linear(3, 5), fr.nn.Tanh()), global_nn=fr.nn.Linear(5, 4)))
        oracle = TorchSimple(aggregate(torch.nn, TorchDeepSets), 'cat')
        self.copy(model, oracle)
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        x, target, weight = torch.randn(6, 3, requires_grad=True), torch.randn(6, 2, requires_grad=True), torch.rand(7, requires_grad=True)
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight), [('x', nx, x), ('target', nt, target), ('weight', nw, weight)])
        model = fr.graph.GENConv(3, 3, aggr=aggregate(fr.nn, DeepSetsAggregation), norm=None)
        oracle = TorchGEN(3, 3, aggr=aggregate(torch.nn, TorchDeepSets), norm=None)
        # PyG does not consult arbitrary aggregators' width metadata; declare
        # only its missing output projection, preserving actual GEN algebra.
        from torch_geometric.nn.dense.linear import Linear
        oracle.lin_aggr_out = Linear(4, 3, bias=False)
        self.copy(model, oracle)
        x = torch.randn(6, 3, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(edge)), oracle(x, edge), [('x', nx, x)])

    def test_serialized_tied_network_dropout_optimizer_rng_and_config(self):
        def factory(eps=1e-5):
            network = fr.nn.Sequential(fr.nn.Linear(3, 3), fr.nn.LayerNorm(3, eps), fr.nn.Tanh(), fr.nn.Dropout(.2, seed=73))
            return DeepSetsAggregation(network, network)
        def step(model, optimizer, rng):
            output = model(fr.Tensor.randn([7, 3], generator=rng), native(self.index), dim_size=6)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(73)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'sets': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'sets': other_opt}, generator=other_rng)
        self.assertIs(other.local_nn, other.global_nn)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'sets': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'sets': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(1e-3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'sets': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_metadata_before_network_rng_and_nested_aggregation(self):
        x = fr.Tensor.ones([7, 3])
        local = fr.nn.Sequential(fr.nn.Linear(3, 4), fr.nn.Dropout(.2, seed=11))
        model = DeepSetsAggregation(local, fr.nn.BatchNorm1d(4))
        before = local[1].rng_state.tolist(), model.global_nn.num_batches_tracked.item()
        with self.assertRaisesRegex(ValueError, 'more than one'): model(x)
        self.assertEqual(before, (local[1].rng_state.tolist(), model.global_nn.num_batches_tracked.item()))
        for norm in (fr.nn.LayerNorm(5), fr.nn.RMSNorm((2, 4)), fr.nn.Softmax(2), fr.nn.Linear(3, 2)):
            bad = DeepSetsAggregation(local, norm)
            with self.assertRaises(ValueError): bad(x, native(self.index), dim_size=6)
            self.assertEqual(before[0], local[1].rng_state.tolist())
        with self.assertRaises(TypeError): DeepSetsAggregation(lambda x: x)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: DeepSetsAggregation()(x))
        with self.assertRaises(NotImplementedError): DeepSetsAggregation()(x, dim=1)
        with self.assertRaises(ValueError): DeepSetsAggregation()(x, ptr=fr.Tensor.from_i64([0, 7], [2]), dim_size=2)
        with self.assertRaisesRegex(ValueError, 'capacity'): DeepSetsAggregation()(x, [0] * 7, dim_size=sys.maxsize // 8 + 1)
        first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=4, dropout=.2))
        combined = fr.graph.MultiAggregation([first, model], mode='sum')
        stream = first.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaises(ValueError): combined(x)
        self.assertEqual(stream, first.multihead_attn.attn_dropout.rng_state.tolist())
        self.assertEqual(before[0], local[1].rng_state.tolist())


if __name__ == '__main__': unittest.main(verbosity=2)
