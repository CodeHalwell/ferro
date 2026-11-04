"""Actual PyG recurrent readout padding, gradient and native-state references."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import LSTMAggregation as TorchLSTM, GRUAggregation as TorchGRU
from torch_geometric.nn.aggr import MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple, GENConv as TorchGEN
from torch_geometric.utils import to_dense_batch
import ferro as fr
from ferro.graph import LSTMAggregation, GRUAggregation
import test_pyg_modulated as helpers

native = helpers.native


class RecurrentAggregationTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1931)
        self.index = torch.tensor([0, 0, 1, 2, 2, 2, 4], dtype=torch.int64)

    def test_actual_pyg_padding_truncation_bidirectional_projection_and_all_gradients(self):
        for cls, reference in ((LSTMAggregation, TorchLSTM), (GRUAggregation, TorchGRU)):
            cases = [(None, {}), (1, {'bias': False}), (5, {'num_layers': 2, 'bidirectional': True})]
            if cls is LSTMAggregation: cases.append((4, {'num_layers': 2, 'bidirectional': True, 'proj_size': 2}))
            for limit, options in cases:
                with self.subTest(kind=cls.__name__, limit=limit, options=options):
                    model, oracle = cls(3, 4, **options), reference(3, 4, **options)
                    self.copy(model, oracle)
                    base = torch.randn(3, 7, requires_grad=True)
                    nb = native(base, True)
                    actual = model(nb.transpose(0, 1), native(self.index), dim_size=6, max_num_elements=limit)
                    expected = oracle(base.transpose(0, 1), self.index, dim_size=6, max_num_elements=limit)
                    self.gradients(model, oracle, actual, expected, [('x', nb, base)])
                    if limit == 1:
                        self.assertEqual(nb.grad.transpose(0, 1).index_select(0, [1, 4, 5]).tolist(), [[0.] * 3] * 3)

    def test_padding_is_processed_and_within_group_order_is_observable(self):
        for cls, reference, name in ((LSTMAggregation, TorchLSTM, 'lstm'), (GRUAggregation, TorchGRU, 'gru')):
            model, oracle = cls(3, 4), reference(3, 4)
            self.copy(model, oracle)
            x = torch.randn(7, 3)
            actual = model(native(x), native(self.index), dim_size=6)
            expected = oracle(x, self.index, dim_size=6)
            self.close(actual, expected)
            dense, _ = to_dense_batch(x, self.index, batch_size=6)
            sequence, _ = getattr(oracle, name)(dense)
            self.assertGreater(float((sequence[1, -1] - sequence[1, 0]).detach().abs().max()), 1e-4)
            self.assertGreater(float(expected[3].detach().abs().max()), 1e-4)
            permuted = x.clone()
            permuted[[0, 1]] = permuted[[1, 0]]
            reordered = model(native(permuted), native(self.index), dim_size=6)
            self.assertGreater(float((torch.tensor(actual.tolist()) - torch.tensor(reordered.tolist())).abs().max()), 1e-4)

    def test_empty_groups_and_missing_index_match_actual_pyg(self):
        for cls, reference in ((LSTMAggregation, TorchLSTM), (GRUAggregation, TorchGRU)):
            for rows, groups, limit, no_index in ((0, 3, 2, False), (0, 0, 2, False), (4, None, None, True)):
                model, oracle = cls(3, 4), reference(3, 4)
                self.copy(model, oracle)
                x = torch.randn(rows, 3, requires_grad=True)
                nx = native(x, True)
                index = None if no_index else torch.empty(0, dtype=torch.int64)
                actual = model(nx, None if index is None else native(index), dim_size=groups, max_num_elements=limit)
                expected = oracle(x, index, dim_size=groups, max_num_elements=limit)
                self.gradients(model, oracle, actual, expected, [('x', nx, x)])
            with self.assertRaises((ValueError, RuntimeError)):
                cls(3, 4)(fr.Tensor.zeros([0, 3]), fr.Tensor.from_i64([], [0]))
            with self.assertRaises((ValueError, RuntimeError)):
                reference(3, 4)(torch.empty(0, 3), torch.empty(0, dtype=torch.int64))

    def test_matched_interlayer_dropout_against_independent_torch_recurrent_layers(self):
        for cls, reference, name, cell in ((LSTMAggregation, TorchLSTM, 'lstm', torch.nn.LSTM), (GRUAggregation, TorchGRU, 'gru', torch.nn.GRU)):
            options = dict(num_layers=3, bidirectional=True, dropout=.25)
            if cls is LSTMAggregation: options['proj_size'] = 2
            model, oracle = cls(3, 4, **options), reference(3, 4, **options)
            self.copy(model, oracle)
            recurrent, reference_recurrent = getattr(model, name), getattr(oracle, name)
            output_width = model.get_out_channels(3)
            masks, samplers = [], []
            for dropout in recurrent.interlayer_dropout:
                sampler = fr.nn.Dropout(.25, seed=1)
                with fr.no_grad(): sampler.rng_state.copy_(dropout.rng_state)
                masks.append(torch.tensor(sampler(fr.Tensor.ones([4, 6, output_width])).tolist()).transpose(0, 1))
                samplers.append(sampler)
            x = torch.randn(7, 3, requires_grad=True)
            nx = native(x, True)
            actual = model(nx, native(self.index), dim_size=6, max_num_elements=4)
            current, _ = to_dense_batch(x, self.index, batch_size=6, max_num_nodes=4)
            for index in range(3):
                kwargs = dict(bidirectional=True, batch_first=True)
                if cls is LSTMAggregation: kwargs['proj_size'] = 2
                layer = cell(3 if index == 0 else output_width, 4, **kwargs)
                for parameter_name in dict(layer.named_parameters()):
                    setattr(layer, parameter_name, getattr(reference_recurrent, parameter_name.replace('_l0', '_l' + str(index))))
                current, _ = layer(current)
                if index < 2: current = current * masks[index]
            self.gradients(model, oracle, actual, current[:, -1], [('x', nx, x)])
            for dropout, sampler in zip(recurrent.interlayer_dropout, samplers):
                self.assertEqual(dropout.rng_state.tolist(), sampler.rng_state.tolist())

    def test_saved_optimizer_aliases_dropout_configuration_and_rng_continuation(self):
        for cls, name in ((LSTMAggregation, 'lstm'), (GRUAggregation, 'gru')):
            def factory(bias=True):
                model = cls(3, 3, num_layers=2, dropout=.2, bias=bias)
                recurrent = getattr(model, name)
                recurrent.weight_hh_l0 = recurrent.weight_ih_l0
                return model
            model, generator = factory(), fr.Generator(229)
            optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
            index = native(self.index)
            def step(model, optimizer, generator):
                output = model(fr.Tensor.randn([7, 3], generator=generator), index, dim_size=6, max_num_elements=4)
                loss = output.square().mean()
                result = output.tolist(), loss.item()
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                return result
            step(model, optimizer, generator)
            saved = fr.checkpoint.snapshot_training(model, optimizers={'readout': optimizer}, generator=generator, step=1)
            with tempfile.TemporaryDirectory() as directory:
                saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
            expected = [step(model, optimizer, generator) for _ in range(3)]
            other, rng = factory(), fr.Generator(1)
            other_opt = fr.optim.AdamW(other.parameters(), lr=.7)
            saved.restore(other, optimizers={'readout': other_opt}, generator=rng)
            self.assertEqual(expected, [step(other, other_opt, rng) for _ in range(3)])
            self.assertIs(getattr(other, name).weight_hh_l0, getattr(other, name).weight_ih_l0)
            left = fr.checkpoint.snapshot_training(model, optimizers={'readout': optimizer}, generator=generator, step=4).tensors()
            right = fr.checkpoint.snapshot_training(other, optimizers={'readout': other_opt}, generator=rng, step=4).tensors()
            self.assertEqual(set(left), set(right))
            for key in left: self.assertEqual(left[key].tolist(), right[key].tolist(), key)
            changed = factory(False)
            with self.assertRaises(ValueError): saved.restore(changed, optimizers={'readout': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))

    def test_strict_sorted_shape_ptr_capacity_and_state_preflight(self):
        for cls, reference, name in ((LSTMAggregation, TorchLSTM, 'lstm'), (GRUAggregation, TorchGRU, 'gru')):
            for factory in (lambda: cls(0, 4), lambda: cls(3, 0), lambda: cls(3, 4, batch_first=False)):
                with self.assertRaises((TypeError, ValueError)): factory()
            model = cls(3, 4, num_layers=2, dropout=.2)
            stream = getattr(model, name).interlayer_dropout[0].rng_state.tolist()
            x, index = fr.Tensor.ones([7, 3]), native(self.index)
            calls = (lambda: model(x, fr.Tensor.from_i64([0, 1, 0, 2, 2, 2, 4], [7])), lambda: model(x, index, dim_size=4),
                     lambda: model(x, index, max_num_elements=0), lambda: model(x, index, dim=-1),
                     lambda: model(fr.Tensor.ones([7, 2]), index), lambda: model(x, ptr=fr.Tensor.from_i64([0, 7], [2])))
            for call in calls:
                with self.assertRaises((ValueError, TypeError, NotImplementedError)): call()
                self.assertEqual(stream, getattr(model, name).interlayer_dropout[0].rng_state.tolist())
            with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, index))
            with self.assertRaises(NotImplementedError): reference(3, 4)(torch.ones(7, 3), ptr=torch.tensor([0, 7]))
            empty, empty_index = fr.Tensor.zeros([0, 1]), fr.Tensor.from_i64([], [0])
            with patch('ferro._graph_recurrent_aggr.cat', side_effect=AssertionError('dense allocation reached')):
                with self.assertRaisesRegex(ValueError, 'capacity'): cls(1, 4)(empty, empty_index, dim_size=sys.maxsize // 16 + 1, max_num_elements=1)

    def test_multi_combination_uses_actual_recurrent_output_widths(self):
        for name in ('lstm', 'gru'):
            for mode in ('cat', 'proj', 'attn', 'mean'):
                width = 3 if mode == 'mean' else 5
                options = dict(aggrs_kwargs=[{}, dict(in_channels=3, out_channels=width)], mode=mode)
                if mode in ('proj', 'attn'):
                    options['mode_kwargs'] = dict(in_channels=[3, width], out_channels=4)
                model = fr.graph.MultiAggregation(['mean', name], **options)
                oracle = TorchMulti(['mean', name], **options)
                self.copy(model, oracle)
                x = torch.randn(7, 3, requires_grad=True)
                nx = native(x, True)
                actual = model(nx, native(self.index), dim_size=6)
                expected = oracle(x, self.index, dim_size=6)
                self.assertEqual(model.get_out_channels(3), expected.shape[1])
                self.gradients(model, oracle, actual, expected, [('x', nx, x)])
            for mode in ('cat', 'sum'):
                options = dict(aggrs_kwargs=[dict(in_channels=3, out_channels=5)], mode=mode)
                model, oracle = fr.graph.MultiAggregation([name], **options), TorchMulti([name], **options)
                self.copy(model, oracle)
                x = torch.randn(7, 3, requires_grad=True)
                nx = native(x, True)
                actual, expected = model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6)
                self.assertEqual(model.get_out_channels(3), 5)
                self.gradients(model, oracle, actual, expected, [('x', nx, x)])

    def test_simple_recurrent_width_root_and_edge_gradients(self):
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        for name in ('lstm', 'gru'):
            for combine in (None, 'cat', 'sum'):
                options = dict(aggr_kwargs=dict(in_channels=3, out_channels=5))
                model, oracle = fr.graph.SimpleConv(name, combine, **options), TorchSimple(name, combine, **options)
                self.copy(model, oracle)
                x = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(6, 1 if combine == 'sum' else 2, requires_grad=True)
                weight = (torch.rand(7) + .3).requires_grad_()
                nx, nt, nw = native(x, True), native(target, True), native(weight, True)
                self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight),
                               [('source', nx, x), ('target', nt, target), ('weight', nw, weight)])

    def test_gen_registered_projection_routes_recurrent_width(self):
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        for name in ('lstm', 'gru'):
            options = dict(aggrs_kwargs=[{}, dict(in_channels=3, out_channels=5)], mode='proj',
                           mode_kwargs=dict(in_channels=[3, 5], out_channels=4))
            aggregate = fr.graph.MultiAggregation(['mean', name], **options)
            reference = TorchMulti(['mean', name], **options)
            model = fr.graph.GENConv(3, 3, aggr=aggregate, norm=None, num_layers=1, bias=True)
            oracle = TorchGEN(3, 3, aggr=reference, norm=None, num_layers=1, bias=True)
            self.copy(model, oracle)
            self.assertEqual(model.lin_aggr_out.weight.tensor().shape, [3, 4])
            x, attr = torch.randn(6, 3, requires_grad=True), torch.randn(7, 3, requires_grad=True)
            nx, na = native(x, True), native(attr, True)
            self.gradients(model, oracle, model(nx, native(edge), na), oracle(x, edge, attr), [('x', nx, x), ('edge_attr', na, attr)])

    def test_nested_preflight_and_changed_width_capacity_precede_streams(self):
        first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn',
                                         mode_kwargs=dict(in_channels=3, out_channels=3, dropout=.2))
        later = LSTMAggregation(3, 3, num_layers=2, dropout=.2)
        model = fr.graph.MultiAggregation([first, later], mode='sum')
        streams = first.multihead_attn.attn_dropout.rng_state.tolist(), later.lstm.interlayer_dropout[0].rng_state.tolist()
        invalid = fr.Tensor.from_i64([0, 1, 0, 2, 2, 2, 4], [7])
        with self.assertRaisesRegex(ValueError, 'sorted'):
            model(fr.Tensor.ones([7, 3]), invalid, dim_size=6)
        self.assertEqual(streams, (first.multihead_attn.attn_dropout.rng_state.tolist(), later.lstm.interlayer_dropout[0].rng_state.tolist()))
        for mode, options in (('sum', {}), ('proj', dict(mode_kwargs=dict(in_channels=[3, 3], out_channels=4)))):
            recurrent = GRUAggregation(3, 5, num_layers=2, dropout=.2)
            mixed = fr.graph.MultiAggregation(['mean', recurrent], mode=mode, **options)
            stream = recurrent.gru.interlayer_dropout[0].rng_state.tolist()
            with self.assertRaisesRegex(ValueError, 'width'):
                mixed(fr.Tensor.ones([7, 3]), native(self.index), dim_size=6)
            self.assertEqual(stream, recurrent.gru.interlayer_dropout[0].rng_state.tolist())
        count = sys.maxsize // 16 + 1
        target = fr.Tensor.ones([1, 1]).expand(count, 1)
        simple = fr.graph.SimpleConv('gru', aggr_kwargs=dict(in_channels=1, out_channels=4))
        with patch('ferro._graph_recurrent_aggr.cat', side_effect=AssertionError('dense allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                simple((fr.Tensor.zeros([0, 1]), target), fr.Tensor.from_i64([], [2, 0]))
            combined = fr.graph.MultiAggregation(['mean', LSTMAggregation(1, 4)])
            with self.assertRaisesRegex(ValueError, 'capacity'):
                combined(fr.Tensor.zeros([0, 1]), fr.Tensor.from_i64([], [0]), dim_size=count)

    def test_composed_recurrent_readout_serialized_state_continuation(self):
        def factory():
            return fr.graph.SimpleConv(['mean', 'gru'], 'cat',
                aggr_kwargs=dict(aggrs_kwargs=[{}, dict(in_channels=3, out_channels=5, num_layers=2, dropout=.2)]))
        edge = native(torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index]))
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([6, 3], generator=generator), edge)
            self.assertEqual(output.shape, [6, 11])
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(251)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'readout': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(2)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'readout': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(2)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'readout': optimizer}, generator=rng, step=3).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'readout': other_opt}, generator=other_rng, step=3).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)


if __name__ == '__main__':
    unittest.main(verbosity=2)
