"""Actual PyG quantiles, native stable sorting and live q interpolation."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import QuantileAggregation as TorchQuantile, MedianAggregation as TorchMedian, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple
import ferro as fr
from ferro.graph import QuantileAggregation, MedianAggregation
import test_pyg_modulated as helpers

native = helpers.native


class QuantileModel(fr.nn.Module):
    learned: bool = True

    def build(self):
        self.aggr = QuantileAggregation([.25, .5, .75])
        if self.learned:
            self.aggr.q = fr.nn.Parameter(self.aggr.q)
            self.q_alias = self.aggr.q
        self.projection = fr.nn.Linear(9, 2)

    def forward(self, x, index):
        return self.projection(self.aggr(x, index, dim_size=5))


class QuantileTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2281)
        self.index = torch.tensor([2, 0, 2, 1, 2, 0, 4, 4, 4])

    def test_all_interpolations_unsorted_multiple_quantiles_live_q_and_input_gradients(self):
        for interpolation in ('linear', 'lower', 'higher', 'nearest', 'midpoint'):
            for quantiles in (.2, [.1, .5, .9, .2]):
                model, oracle = QuantileAggregation(quantiles, interpolation, -2.), TorchQuantile(quantiles, interpolation, -2.)
                model.q.requires_grad_(True); oracle.q.requires_grad_()
                base = torch.randn(3, 9, requires_grad=True); nb = native(base, True)
                self.gradients(model, oracle, model(nb.transpose(0, 1), native(self.index), dim_size=6), oracle(base.transpose(0, 1), self.index, dim_size=6), [('x', nb, base), ('q', model.q, oracle.q)])
                self.assertEqual(model.get_out_channels(3), 3 * (len(quantiles) if isinstance(quantiles, list) else 1))

    def test_global_offset_rounding_nearest_half_even_and_lower_median(self):
        values = torch.tensor([[3., 7.], [0., 1.], [12., 20.], [6., 4.], [9., 16.]], requires_grad=True)
        index = torch.tensor([0, 0, 2, 2, 2])
        model, oracle = QuantileAggregation([.2, .5, 1.]), TorchQuantile([.2, .5, 1.])
        nx = native(values, True)
        self.gradients(model, oracle, model(nx, native(index), dim_size=4), oracle(values, index, dim_size=4), [('x', nx, values)])
        for count in (2, 4, 6, 8):
            x = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2).requires_grad_()
            for constructor, reference in ((lambda: QuantileAggregation(.5, 'nearest'), lambda: TorchQuantile(.5, 'nearest')), (MedianAggregation, TorchMedian)):
                model, oracle = constructor(), reference(); nx = native(x, True)
                self.gradients(model, oracle, model(nx), oracle(x), [('x', nx, x)])
                x.grad = None

    def test_empty_groups_empty_inputs_feature_zero_and_reference_failures(self):
        for rows, width, groups, index in ((0, 3, None, None), (9, 0, 6, self.index), (9, 3, 6, self.index)):
            model, oracle = QuantileAggregation(.3), TorchQuantile(.3)
            model.q.requires_grad_(True); oracle.q.requires_grad_()
            x = torch.randn(rows, width, requires_grad=True); nx = native(x, True)
            self.gradients(model, oracle, model(nx, None if index is None else native(index), dim_size=groups), oracle(x, index, dim_size=groups), [('x', nx, x), ('q', model.q, oracle.q)])
        model, oracle = QuantileAggregation([.2, .8]), TorchQuantile([.2, .8])
        model.q.requires_grad_(True); oracle.q.requires_grad_()
        x = torch.empty(9, 0, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x), ('q', model.q, oracle.q)])
        x = torch.randn(3, 2); index = torch.tensor([1, 1, 2])
        for interpolation in ('linear', 'lower', 'midpoint'):
            with self.assertRaises((IndexError, RuntimeError)): TorchQuantile(.2, interpolation)(x, index, dim_size=4)
            with self.assertRaisesRegex(ValueError, 'selected row'): QuantileAggregation(.2, interpolation)(native(x), native(index), dim_size=4)
        model, oracle = QuantileAggregation(.2, 'nearest', 7.), TorchQuantile(.2, 'nearest', 7.)
        self.close(model(native(x), native(index), dim_size=4), oracle(x, index, dim_size=4))
        for q, groups in (([.2, .5], None), (.5, 3)):
            with self.assertRaises(RuntimeError): TorchQuantile(q)(torch.empty(0, 2), dim_size=groups)
            with self.assertRaises(ValueError): QuantileAggregation(q)(fr.Tensor.zeros([0, 2]), dim_size=groups)

    def test_native_stable_tied_adjoints_and_nonfinite_value_semantics(self):
        x = torch.ones(64, 2, requires_grad=True)
        model = MedianAggregation(); nx = native(x, True)
        output = model(nx)
        reference = x.sort(dim=0, stable=True).values[31:32]
        self.gradients(model, torch.nn.Identity(), output, reference, [('x', nx, x)])
        expected = [[0., 0.]] * 64
        expected[31] = nx.grad.tolist()[31]
        self.assertEqual(nx.grad.tolist(), expected)
        signed = MedianAggregation()(fr.Tensor([-0., 0., 0., -0.], [2, 2]))
        self.assertEqual([math.copysign(1., value) for value in signed.flatten().tolist()], [-1., 1.])
        for interpolation in ('linear', 'lower', 'higher', 'nearest', 'midpoint'):
            x = torch.tensor([[-float('inf'), 1.], [2., float('nan')], [float('inf'), 3.], [4., -2.]])
            index = torch.tensor([0, 0, 0, 0])
            model, oracle = QuantileAggregation([0., .2, .5, .9, 1.], interpolation), TorchQuantile([0., .2, .5, .9, 1.], interpolation)
            torch.testing.assert_close(helpers.torch_value(model(native(x), native(index))), oracle(x, index), equal_nan=True)

    def test_width_composition_multi_and_simple_edge_weight_gradients(self):
        model = fr.graph.MultiAggregation(['mean', 'quantile'], aggrs_kwargs=[{}, dict(q=[.2, .5, .8])])
        oracle = TorchMulti(['mean', TorchQuantile([.2, .5, .8])])
        x = torch.randn(9, 3, requires_grad=True); nx = native(x, True)
        self.assertEqual(model.get_out_channels(3), 12)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model, oracle = fr.graph.SimpleConv('median', 'cat'), TorchSimple(TorchMedian(), 'cat')
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1, 0, 2]), self.index])
        x, target, weight = torch.randn(6, 3, requires_grad=True), torch.randn(6, 2, requires_grad=True), torch.rand(9, requires_grad=True)
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight), [('x', nx, x), ('target', nt, target), ('edge_weight', nw, weight)])

    def test_learned_q_alias_optimizer_checkpoint_rng_and_default_buffer_schema(self):
        def step(model, optimizer, rng):
            output = model(fr.Tensor.randn([9, 3], generator=rng), native(self.index))
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = QuantileModel(), fr.Generator(367)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.0003)
        before = model.aggr.q.tolist(); step(model, optimizer, rng)
        self.assertNotEqual(before, model.aggr.q.tolist())
        saved = fr.checkpoint.snapshot_training(model, optimizers={'quantile': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = QuantileModel(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'quantile': other_opt}, generator=other_rng)
        self.assertIs(other.aggr.q, other.q_alias)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'quantile': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'quantile': other_opt}, generator=other_rng, step=4).tensors()
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = QuantileModel(learned=False)
        before = changed.aggr.q.tolist(), [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'quantile': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, (changed.aggr.q.tolist(), [p.tensor().tolist() for p in changed.parameters()]))
        plain = QuantileAggregation([.2, .8])
        with fr.no_grad(): plain.q.copy_(fr.Tensor([.3, .6], [2, 1]))
        before = plain.q.tolist(); plain.reset_parameters(); self.assertEqual(before, plain.q.tolist())
        buffer_state = fr.checkpoint.snapshot_training(plain, optimizers={}, generator=fr.Generator(1), step=0)
        replacement = QuantileAggregation([.1, .9])
        buffer_state.restore(replacement, optimizers={}, generator=fr.Generator(2))
        self.assertEqual(before, replacement.q.tolist())
        incompatible = QuantileAggregation([.1, .9], 'lower')
        before_incompatible = incompatible.q.tolist()
        with self.assertRaises(ValueError): buffer_state.restore(incompatible, optimizers={}, generator=fr.Generator(2))
        self.assertEqual(before_incompatible, incompatible.q.tolist())

    def test_fill_scalar_overflow_precedes_nested_rng_but_nonfinite_fills_are_valid(self):
        x = torch.randn(9, 3)
        for fill in (-1e100, 1e100):
            with self.assertRaises(RuntimeError): TorchQuantile(.5, fill_value=fill)(x)
            with self.assertRaisesRegex(ValueError, 'overflows'): QuantileAggregation(.5, fill_value=fill)(native(x))
            first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=6, dropout=.2))
            combined = fr.graph.MultiAggregation([first, QuantileAggregation([.2, .8], fill_value=fill)], mode='sum')
            before = first.multihead_attn.attn_dropout.rng_state.tolist()
            with patch.object(fr.Tensor, 'sort', side_effect=AssertionError('sort reached before fill validation')):
                with self.assertRaisesRegex(ValueError, 'overflows'): combined(native(x), native(self.index), dim_size=6)
            self.assertEqual(before, first.multihead_attn.attn_dropout.rng_state.tolist())
        for fill in (float('inf'), -float('inf'), float('nan')):
            actual = QuantileAggregation([.2, .8], fill_value=fill)(native(x), native(self.index), dim_size=6)
            expected = TorchQuantile([.2, .8], fill_value=fill)(x, self.index, dim_size=6)
            torch.testing.assert_close(helpers.torch_value(actual), expected, equal_nan=True)

    def test_metadata_buffer_integrity_capacity_and_nested_state_preflight(self):
        for q in ([], -1., 2., float('nan'), [0., float('inf')]):
            with self.assertRaises(ValueError): QuantileAggregation(q)
        with self.assertRaises(ValueError): QuantileAggregation(.5, 'invalid')
        model, x = QuantileAggregation([.2, .8]), fr.Tensor.ones([9, 3])
        with self.assertRaises(NotImplementedError): model(x, ptr=fr.Tensor.from_i64([0, 9], [2]))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x))
        with self.assertRaises(ValueError): model(x, [0] * 8)
        with patch.object(fr.Tensor, 'sort', side_effect=AssertionError('sort reached before capacity check')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(x, [0] * 9, dim_size=sys.maxsize // 8 + 1)
        first = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=6, dropout=.2))
        combined = fr.graph.MultiAggregation([first, model], mode='sum')
        before = first.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'selected row'): combined(x, [1] * 9, dim_size=3)
        self.assertEqual(before, first.multihead_attn.attn_dropout.rng_state.tolist())
        with fr.no_grad(): model.q.copy_(fr.Tensor([float('nan'), .8], [2, 1]))
        with self.assertRaisesRegex(ValueError, 'q buffer'): combined(x, native(self.index), dim_size=6)
        self.assertEqual(before, first.multihead_attn.attn_dropout.rng_state.tolist())
        model.q = fr.Tensor.ones([2])
        with self.assertRaisesRegex(ValueError, 'q buffer shape'): model(x)


if __name__ == '__main__': unittest.main(verbosity=2)
