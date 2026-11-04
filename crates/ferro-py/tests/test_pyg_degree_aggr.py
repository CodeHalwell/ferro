"""Actual PyG degree scalers, learned normalization and registered state."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import DegreeScalerAggregation as TorchDegree, SoftmaxAggregation as TorchSoftmax, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple
import ferro as fr
from ferro.graph import DegreeScalerAggregation
import test_pyg_modulated as helpers

native = helpers.native
SCALERS = ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear']


class DegreeAggregationTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2039)
        self.index = torch.tensor([2, 0, 2, 1, 2, 0, 4])
        self.histogram = torch.tensor([3, 4, 2, 1])

    def test_all_degree_scalers_scalar_multi_and_normalization_gradients(self):
        cases = [(['mean', 'max', 'std'], SCALERS)] + [('mean', scaler) for scaler in SCALERS]
        for aggr, scalers in cases:
            for train in (False, True):
                model = DegreeScalerAggregation(aggr, scalers, native(self.histogram), train)
                oracle = TorchDegree(aggr, scalers, self.histogram, train)
                self.copy(model, oracle)
                base = torch.randn(3, 7, requires_grad=True)
                nb = native(base, True)
                actual = model(nb.transpose(0, 1), native(self.index), dim_size=6)
                expected = oracle(base.transpose(0, 1), self.index, dim_size=6)
                self.assertEqual(actual.shape[1], model.get_out_channels(3))
                self.gradients(model, oracle, actual, expected, [('x', nb, base)])

    def test_registered_learned_child_histogram_ownership_and_partial_reset(self):
        histogram = native(self.histogram)
        model = DegreeScalerAggregation(fr.graph.SoftmaxAggregation(.7, learn=True, channels=3), SCALERS, histogram, True)
        oracle = TorchDegree(TorchSoftmax(.7, learn=True, channels=3), SCALERS, self.histogram, True)
        self.copy(model, oracle)
        x = torch.randn(7, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        with fr.no_grad():
            histogram.copy_(fr.Tensor.from_i64([9, 1, 1, 1], [4]))
            model.aggr.t.tensor().copy_(fr.Tensor.ones([3]) * 2.5)
            model.avg_deg_lin.tensor().copy_(fr.Tensor.ones([1]) * 9.)
            model.avg_deg_log.tensor().copy_(fr.Tensor.ones([1]) * 7.)
        model.reset_parameters()
        self.assertEqual(model.histogram, tuple(self.histogram.tolist()))
        self.assertEqual(model.aggr.t.tensor().tolist(), [2.5] * 3)
        self.close(model.avg_deg_lin.tensor(), oracle.avg_deg_lin)
        self.close(model.avg_deg_log.tensor(), oracle.avg_deg_log)
        with self.assertRaises(AttributeError): model.scaler = ('identity',)
        # Accept scalar/list spelling independently of aggregation type. PyG's
        # current constructor tests aggr's type here instead of scaler's type.
        one = DegreeScalerAggregation(fr.graph.MeanAggregation(), 'linear', native(self.histogram))
        reference = TorchDegree(['mean'], ['linear'], self.histogram)
        self.close(one(native(x.detach()), native(self.index)), reference(x.detach(), self.index))

    def test_empty_features_and_groups_preserve_optional_normalization_gradients(self):
        for rows, width, groups in ((0, 3, 0), (0, 3, 5), (4, 0, 5)):
            model = DegreeScalerAggregation(['mean'], SCALERS, native(self.histogram), True)
            oracle = TorchDegree(['mean'], SCALERS, self.histogram, True)
            self.copy(model, oracle)
            x = torch.randn(rows, width, requires_grad=True)
            index = torch.tensor([0, 2, 2, 3]) if rows else torch.empty(0, dtype=torch.long)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, native(index), dim_size=groups), oracle(x, index, dim_size=groups), [('x', nx, x)])
        model = DegreeScalerAggregation(fr.graph.SortAggregation(0), SCALERS, native(self.histogram), True)
        x = fr.Tensor.ones([1, 1]).requires_grad_(True)
        with patch('ferro._graph_degree_aggr.Tensor', side_effect=AssertionError('empty readout materialized degree rows')):
            output = model(x, [0], dim_size=10**6)
            self.assertEqual(output.shape, [10**6, 0])
            output.sum().backward()
        self.assertEqual(x.grad.tolist(), [[0.]])
        self.assertEqual(model.avg_deg_lin.tensor().grad.tolist(), [0.])
        self.assertEqual(model.avg_deg_log.tensor().grad.tolist(), [0.])

    def test_width_composition_and_weighted_message_gradients(self):
        def aggregate(): return DegreeScalerAggregation(['mean'], ['identity', 'linear'], native(self.histogram), True)
        def reference(): return TorchDegree(['mean'], ['identity', 'linear'], self.histogram, True)
        options = dict(mode='proj', mode_kwargs=dict(in_channels=[3, 6], out_channels=4))
        model = fr.graph.MultiAggregation(['mean', aggregate()], **options)
        oracle = TorchMulti(['mean', reference()], **options)
        self.copy(model, oracle)
        x = torch.randn(7, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model = fr.graph.SimpleConv('degree_scaler', 'cat', aggr_kwargs=dict(aggr=['mean'], scaler=['identity', 'linear'], deg=native(self.histogram), train_norm=True))
        oracle = TorchSimple('degree_scaler', 'cat', aggr_kwargs=dict(aggr=['mean'], scaler=['identity', 'linear'], deg=self.histogram, train_norm=True))
        self.copy(model, oracle)
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        x, target = torch.randn(6, 3, requires_grad=True), torch.randn(6, 2, requires_grad=True)
        weight = (torch.rand(7) + .5).requires_grad_()
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight), [('x', nx, x), ('target', nt, target), ('weight', nw, weight)])

    def test_aliased_learned_norm_and_child_saved_optimizer_rng_and_config(self):
        def factory(histogram=None):
            model = DegreeScalerAggregation('softmax', 'linear', native(self.histogram if histogram is None else histogram), True, dict(learn=True, t=.7))
            model.avg_deg_lin = model.aggr.t
            return model
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([7, 3], generator=generator), native(self.index), dim_size=6)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(307)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'degree': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'degree': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.avg_deg_lin, other.aggr.t)
        left = fr.checkpoint.snapshot_training(model, optimizers={'degree': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'degree': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(torch.tensor([4, 3, 2, 1]))
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'degree': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_metadata_histogram_and_nested_state_preflight(self):
        for histogram in (torch.tensor([]), torch.tensor([0, 0]), torch.tensor([1, -1]), torch.tensor([1., .5]), torch.tensor([1., float('nan')])):
            with self.assertRaises(ValueError): DegreeScalerAggregation('mean', 'identity', native(histogram))
        for scaler in ('invalid', [], ['identity', 'bad']):
            with self.assertRaises(ValueError): DegreeScalerAggregation('mean', scaler, native(self.histogram))
        with self.assertRaises(ValueError): DegreeScalerAggregation('mean', 'linear', fr.Tensor.from_i64([3], [1]))
        model = DegreeScalerAggregation('mean', 'identity', native(self.histogram))
        x = fr.Tensor.ones([7, 3])
        with self.assertRaises(NotImplementedError): model(x, ptr=fr.Tensor.from_i64([0, 7], [2]))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x))
        with self.assertRaisesRegex(ValueError, 'capacity'): model(x, [0] * 7, dim_size=sys.maxsize // 8 + 1)
        for histogram in (fr.Tensor.from_i64([1], [1]).expand(sys.maxsize // 8 + 1), fr.Tensor.ones([1]).expand(sys.maxsize // 4 + 1)):
            with patch.object(fr.Tensor, 'tolist', side_effect=AssertionError('histogram materialized before capacity validation')):
                with self.assertRaisesRegex(ValueError, 'capacity'): DegreeScalerAggregation('mean', 'identity', histogram)
        first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=6, dropout=.2))
        later = DegreeScalerAggregation(fr.graph.LSTMAggregation(3, 3, num_layers=2, dropout=.2), ['identity', 'linear'], native(self.histogram))
        combined = fr.graph.MultiAggregation([first, later], mode='sum')
        before = first.multihead_attn.attn_dropout.rng_state.tolist(), later.aggr.lstm.interlayer_dropout[0].rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'sorted'): combined(x, native(self.index), dim_size=6)
        self.assertEqual(before, (first.multihead_attn.attn_dropout.rng_state.tolist(), later.aggr.lstm.interlayer_dropout[0].rng_state.tolist()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
