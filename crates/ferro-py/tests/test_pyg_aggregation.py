"""Actual PyG aggregation references with live inputs and learned exponents."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import aggr as reference
import ferro as fr
from ferro import graph as aggregation


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1543)
        self.index = torch.tensor([2, 0, 2, 0, 3, 2])

    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=6e-5, rtol=6e-4)

    def compare(self, model, oracle, x, index=None, ptr=None, count=None):
        parameters = dict(oracle.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters(): parameter.tensor().copy_(native(parameters[name]))
        nx = native(x, True)
        ni, np = None if index is None else native(index), None if ptr is None else native(ptr)
        actual, expected = model(nx, ni, np, count), oracle(x, index, ptr, count)
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward(); (expected * probe).sum().backward()
        inputs = [('x', nx, x)] + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]
        for name, left, right in inputs:
            with self.subTest(gradient=name):
                if right.grad is None: self.assertIsNone(left.grad)
                else: self.close(left.grad, right.grad)

    def test_sum_mean_max_min_index_ptr_and_default_group(self):
        for name in ('SumAggregation', 'MeanAggregation', 'MaxAggregation', 'MinAggregation'):
            for route in ('index', 'ptr', 'none'):
                with self.subTest(name=name, route=route):
                    x = torch.randn(6, 3, requires_grad=True)
                    model, oracle = getattr(aggregation, name)(), getattr(reference, name)()
                    index = self.index if route == 'index' else None
                    ptr = torch.tensor([0, 2, 2, 5, 6]) if route == 'ptr' else None
                    self.compare(model, oracle, x, index, ptr, 5 if route == 'index' else None)

    def test_variance_std_threshold_and_semi_gradient_reference(self):
        for name in ('VarAggregation', 'StdAggregation'):
            for semi in (False, True):
                with self.subTest(name=name, semi=semi):
                    x = torch.tensor([[1., 1.], [1., 1.0001], [-2., 3.], [2., 1.], [.1, .2]], requires_grad=True)
                    index = torch.tensor([0, 0, 1, 1, 2])
                    self.compare(getattr(aggregation, name)(semi), getattr(reference, name)(semi), x, index, count=4)

    def test_softmax_scalar_channel_temperatures_and_semi_gradient(self):
        for temperature, learned, semi, channels in ((1., False, False, 1), (-.7, False, True, 1),
                                                   (.3, True, False, 1), (1., True, False, 3)):
            with self.subTest(t=temperature, learn=learned, semi=semi, channels=channels):
                model, oracle = aggregation.SoftmaxAggregation(temperature, learned, semi, channels), reference.SoftmaxAggregation(temperature, learned, semi, channels)
                if channels == 3:
                    with torch.no_grad(): oracle.t.copy_(torch.tensor([-.4, .7, 1.2]))
                x = torch.randn(6, 3, requires_grad=True)
                self.compare(model, oracle, x, self.index, count=5)
        model = aggregation.SoftmaxAggregation(.7)
        self.assertEqual(model.t, .7)
        with self.assertRaises(AttributeError): model.t = 1.

    def test_powermean_constant_learned_channel_powers_and_clamp_gradients(self):
        for power, learn, channels, lower, upper in ((1., False, 1, 1e-4, 100.), (2., False, 1, .1, 2.),
                  (-2., False, 1, .1, 3.), (.7, True, 1, .1, 3.), (1., True, 3, .1, 2.), (2., False, 1, None, 10.)):
            with self.subTest(p=power, learn=learn, channels=channels, lower=lower):
                model = aggregation.PowerMeanAggregation(power, learn, channels, lower, upper)
                oracle = reference.PowerMeanAggregation(power, learn, channels, lower, upper)
                if channels == 3:
                    with torch.no_grad(): oracle.p.copy_(torch.tensor([.7, 1., 2.1]))
                x = (torch.rand(6, 3) * 3 + .03).requires_grad_(True)
                self.compare(model, oracle, x, self.index, count=5 if lower is not None else 4)
        model = aggregation.PowerMeanAggregation(2.)
        self.assertEqual(model.p, 2.)
        with self.assertRaises(AttributeError): model.p = .5

    def test_multiaggregation_dense_modes_and_projection_all_parameters(self):
        for mode in ('cat', 'sum', 'mean', 'max', 'min', 'logsumexp', 'std', 'var', 'proj', 'attn'):
            with self.subTest(mode=mode):
                kwargs = dict(in_channels=3, out_channels=4, bias=False) if mode == 'proj' else None
                if mode == 'attn': kwargs = dict(in_channels=3, out_channels=4, num_heads=2, bias=False)
                args = ['mean', 'softmax', 'powermean']
                options = [{}, {'t': .7, 'learn': True}, {'p': 1.3, 'learn': True}]
                model = aggregation.MultiAggregation(args, options, mode, kwargs)
                oracle = reference.MultiAggregation(args, options, mode, kwargs)
                x = (torch.rand(6, 3) * 2 + .4).requires_grad_(True)
                if mode in ('std', 'var'):
                    # Keep the std oracle away from almost-identical reduction
                    # outputs, where f32 roundoff is amplified by sqrt'(variance).
                    x = torch.tensor([[.2, 1., 4.], [.4, .6, 2.], [2., 3., 1.],
                                      [4., 3., .3], [1.2, 2., 3.], [3., .7, 2.]], requires_grad=True)
                self.compare(model, oracle, x, self.index, count=5)
                self.assertEqual(model.get_out_channels(3), oracle.get_out_channels(3))

    def test_attention_combine_same_dropout_mask_all_gradients_and_reset(self):
        options = dict(in_channels=3, out_channels=4, num_heads=2, dropout=.3)
        model = aggregation.MultiAggregation(['sum', 'mean', 'softmax'], mode='attn', mode_kwargs=options)
        oracle = reference.MultiAggregation(['sum', 'mean', 'softmax'], mode='attn', mode_kwargs=options)
        sampler = fr.nn.Dropout(.3, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(model.multihead_attn.attn_dropout.rng_state)
        mask = torch.tensor(sampler(fr.Tensor.ones([5, 2, 3, 3])).tolist())
        calls = []
        def dropout(value, p=.5, training=True, inplace=False):
            calls.append((list(value.shape), p, training))
            self.assertEqual(p, .3)
            return value * mask.reshape(value.shape)
        with patch('torch.nn.functional.dropout', side_effect=dropout):
            self.compare(model, oracle, torch.randn(6, 3, requires_grad=True), self.index, count=5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.multihead_attn.attn_dropout.rng_state.tolist(), sampler.rng_state.tolist())
        stream = model.multihead_attn.attn_dropout.rng_state.tolist()
        output_weight = model.multihead_attn.out_proj.weight.tensor().tolist()
        model.reset_parameters()
        self.assertEqual(output_weight, model.multihead_attn.out_proj.weight.tensor().tolist())
        self.assertEqual(stream, model.multihead_attn.attn_dropout.rng_state.tolist())

    def test_attention_combine_checkpoint_dropout_and_empty_batch(self):
        def factory(dropout=.25):
            return aggregation.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=4, num_heads=2, dropout=dropout))
        model, rng = factory(), fr.Generator(131)
        opt = fr.optim.AdamW(model.parameters(), lr=.002)
        index = native(self.index)
        def step(current, optimizer, generator):
            output = current(fr.Tensor.rand([6, 3], generator=generator), index, dim_size=5)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        step(model, opt, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'attention': opt}, generator=rng, step=1)
        expected = [step(model, opt, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        state.restore(other, optimizers={'attention': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertEqual(model.multihead_attn.attn_dropout.rng_state.tolist(), other.multihead_attn.attn_dropout.rng_state.tolist())
        changed = factory(.4)
        stream = changed.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaises(ValueError): state.restore(changed, optimizers={'attention': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(stream, changed.multihead_attn.attn_dropout.rng_state.tolist())
        for count in (0, 2):
            model, oracle = factory(0.), reference.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=4, num_heads=2))
            self.compare(model, oracle, torch.empty(0, 3, requires_grad=True), torch.empty(0, dtype=torch.int64), count=count)

    def test_multiaggregation_first_winner_ties_shared_modules_and_ptr(self):
        for mode in ('min', 'max'):
            model = aggregation.MultiAggregation(['sum', 'sum'], mode=mode)
            oracle = reference.MultiAggregation(['sum', 'sum'], mode=mode)
            self.compare(model, oracle, torch.randn(6, 3, requires_grad=True), ptr=torch.tensor([0, 2, 2, 6]))
        shared, oracle_shared = aggregation.SoftmaxAggregation(.7, learn=True), reference.SoftmaxAggregation(.7, learn=True)
        model, oracle = aggregation.MultiAggregation([shared, shared]), reference.MultiAggregation([oracle_shared, oracle_shared])
        self.compare(model, oracle, torch.randn(6, 3, requires_grad=True), self.index)
        self.assertIs(model.aggrs[0], model.aggrs[1])
        for mode in ('cat', 'sum', 'var', 'std'):
            model, oracle = aggregation.MultiAggregation(['mean'], mode=mode), reference.MultiAggregation(['mean'], mode=mode)
            self.compare(model, oracle, torch.randn(6, 3, requires_grad=True), self.index)

    def test_empty_nodes_empty_groups_zero_width_and_strided_inputs(self):
        constructors = [(aggregation.SumAggregation, reference.SumAggregation, {}),
                        (aggregation.StdAggregation, reference.StdAggregation, {}),
                        (aggregation.SoftmaxAggregation, reference.SoftmaxAggregation, {'learn': True}),
                        (aggregation.PowerMeanAggregation, reference.PowerMeanAggregation, {'p': 1.4, 'learn': True})]
        for cls, ref, options in constructors:
            for nodes, width in ((0, 3), (4, 0)):
                with self.subTest(name=cls.__name__, nodes=nodes, width=width):
                    x = torch.empty(nodes, width, requires_grad=True)
                    self.compare(cls(**options), ref(**options), x, torch.zeros(nodes, dtype=torch.int64), count=2)
        model, oracle = aggregation.MeanAggregation(), reference.MeanAggregation()
        x = torch.randn(3, 6).T.detach().requires_grad_(True)
        nx = native(x.T).transpose(0, 1).requires_grad_(True)
        result, expected = model(nx, native(self.index)), oracle(x, self.index)
        self.close(result, expected)
        result.square().sum().backward(); expected.square().sum().backward()
        self.close(nx.grad, x.grad)

    def test_native_optimizer_checkpoint_alias_and_initial_configuration(self):
        def factory(changed=False):
            shared = aggregation.SoftmaxAggregation(.8 if changed else .7, learn=True)
            return aggregation.MultiAggregation([shared, shared, aggregation.PowerMeanAggregation(1.2, learn=True)], mode='proj', mode_kwargs={'in_channels': 3, 'out_channels': 2})
        model, rng = factory(), fr.Generator(107)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        index = native(self.index)
        def step(current, opt, generator):
            x = fr.Tensor.rand([6, 3], generator=generator) + .4
            output = current(x, index, dim_size=5)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); opt.step(); opt.zero_grad()
            return values
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'aggr': optimizer}, generator=rng, step=1)
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'aggr': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.aggrs[0], other.aggrs[1])
        for left, right in zip(model.parameters(), other.parameters()): self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        changed = factory(True)
        before = [parameter.tensor().tolist() for parameter in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'aggr': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [parameter.tensor().tolist() for parameter in changed.parameters()])

    def test_validation_and_capacity_precede_work(self):
        for constructor in (lambda: aggregation.SoftmaxAggregation(learn=True, semi_grad=True), lambda: aggregation.SoftmaxAggregation(channels=2),
                            lambda: aggregation.PowerMeanAggregation(p=0.), lambda: aggregation.PowerMeanAggregation(clamp_min=2., clamp_max=1.),
                            lambda: aggregation.MultiAggregation([]), lambda: aggregation.MultiAggregation(['mean'], mode='proj'),
                            lambda: aggregation.MultiAggregation(['mean', 'sum'], mode='proj', mode_kwargs={'in_channels': 3})):
            with self.assertRaises((ValueError, TypeError)): constructor()
        with self.assertRaises(NotImplementedError): aggregation.PowerMeanAggregation(learn=True, clamp_min=None)
        with self.assertRaises(NotImplementedError): aggregation.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=4, add_zero_attn=True))
        x, index = native(torch.randn(6, 3)), native(self.index)
        for call in (lambda: aggregation.MeanAggregation()(x, index, dim_size=2), lambda: aggregation.SumAggregation()(x, native(torch.tensor([0, 1]))),
                     lambda: aggregation.SumAggregation()(x, ptr=native(torch.tensor([1, 6]))), lambda: aggregation.SumAggregation()(x, ptr=native(torch.tensor([0, 7, 6]))),
                     lambda: aggregation.SumAggregation()(x, ptr=native(torch.tensor([0, 6])), dim_size=2), lambda: aggregation.SumAggregation()(x, index, native(torch.tensor([0, 6]))),
                     lambda: aggregation.SoftmaxAggregation(learn=True, channels=2)(x), lambda: aggregation.PowerMeanAggregation(learn=True, channels=2)(x)):
            with self.assertRaises((ValueError, TypeError)): call()
        model = aggregation.SoftmaxAggregation()
        with self.assertRaises(NotImplementedError): model(x, dim=1)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, index))
        huge = fr.Tensor.ones([1, 1]).expand(1, sys.maxsize // 8 + 1)
        with patch('ferro._graph_aggregation.stack', side_effect=AssertionError('capacity must reject before stack')):
            with self.assertRaises(ValueError): aggregation.MultiAggregation(['sum', 'mean'], mode='max')(huge)
        with self.assertRaises(ValueError): aggregation.MeanAggregation()(x, index, dim_size=sys.maxsize)


if __name__ == '__main__':
    unittest.main(verbosity=2)
