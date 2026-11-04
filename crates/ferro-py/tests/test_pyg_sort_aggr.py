"""Actual PyG sort-pooling values, selected gradients, and native saved state."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import SortAggregation as TorchSort, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple
import ferro as fr
from ferro.graph import SortAggregation
import test_pyg_modulated as helpers

native = helpers.native


class SortAggregationTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1951)
        self.index = torch.tensor([0, 0, 1, 2, 2, 2, 4], dtype=torch.int64)

    def test_actual_pyg_padding_truncation_strided_and_selected_input_gradients(self):
        for k, limit in ((1, None), (3, None), (5, None), (2, 1), (3, 5), (0, None), (2, 0)):
            model, oracle = SortAggregation(k), TorchSort(k)
            base = torch.randn(3, 7, requires_grad=True)
            nb = native(base, True)
            actual = model(nb.transpose(0, 1), native(self.index), dim_size=6, max_num_elements=limit)
            expected = oracle(base.transpose(0, 1), self.index, dim_size=6, max_num_elements=limit)
            self.assertEqual(model.get_out_channels(3), 3 * k)
            self.gradients(model, oracle, actual, expected, [('x', nb, base)])
            if limit == 1:
                self.assertEqual(nb.grad.transpose(0, 1).index_select(0, [1, 4, 5]).tolist(), [[0.] * 3] * 3)
        x = torch.randn(5, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(SortAggregation(3), TorchSort(3), SortAggregation(3)(nx), TorchSort(3)(x), [('x', nx, x)])

    def test_stable_tie_order_is_an_explicit_native_boundary(self):
        model = SortAggregation(5)
        x = torch.stack([torch.arange(40.), torch.ones(40)], 1).requires_grad_()
        nx = native(x, True)
        actual = model(nx)
        order = x[:, -1].sort(descending=True, stable=True).indices
        expected = x[order[:5]].reshape(1, 10)
        self.gradients(model, TorchSort(5), actual, expected, [('x', nx, x)])
        self.assertEqual(actual.tolist(), [[0., 1., 1., 1., 2., 1., 3., 1., 4., 1.]])

    def test_detached_sentinel_rounding_infinity_and_nan_match_pyg_values(self):
        cases = ([[-1e20, -1e20], [3., 2.]], [[-float('inf'), -float('inf')], [3., 2.]],
                 [[float('nan'), 1.], [3., 2.]], [[float('inf'), float('inf')], [float('inf'), float('inf')]])
        for values in cases:
            x = torch.tensor(values)
            actual = SortAggregation(3)(native(x), fr.Tensor.from_i64([0, 0], [2]), dim_size=2)
            expected = TorchSort(3)(x, torch.tensor([0, 0]), dim_size=2)
            torch.testing.assert_close(helpers.torch_value(actual), expected, equal_nan=True)
        x = torch.tensor([[-1e20, -1e20], [3., 2.]], requires_grad=True)
        nx = native(x, True)
        model, oracle = SortAggregation(3), TorchSort(3)
        self.gradients(model, oracle, model(nx), oracle(x), [('x', nx, x)])
        self.assertEqual(nx.grad.tolist()[0], [0., 0.])

    def test_width_aware_multi_and_simple_root_edge_gradient_composition(self):
        options = dict(aggrs_kwargs=[{}, dict(k=2)], mode='proj', mode_kwargs=dict(in_channels=[3, 6], out_channels=4))
        model = fr.graph.MultiAggregation(['mean', 'sort'], **options)
        oracle = TorchMulti(['mean', 'sort'], **options)
        self.copy(model, oracle)
        x = torch.randn(7, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.index), dim_size=6), oracle(x, self.index, dim_size=6), [('x', nx, x)])
        model, oracle = fr.graph.SimpleConv('sort', 'cat', aggr_kwargs=dict(k=2)), TorchSimple('sort', 'cat', aggr_kwargs=dict(k=2))
        edge = torch.stack([torch.tensor([0, 1, 2, 3, 4, 5, 1]), self.index])
        x, target = torch.randn(6, 3, requires_grad=True), torch.randn(6, 2, requires_grad=True)
        weight = (torch.rand(7) + .3).requires_grad_()
        nx, nt, nw = native(x, True), native(target, True), native(weight, True)
        self.gradients(model, oracle, model((nx, nt), native(edge), nw), oracle((x, target), edge, weight),
                       [('x', nx, x), ('target', nt, target), ('weight', nw, weight)])

    def test_config_checkpoint_and_optimizer_continuation(self):
        def factory(k=2): return fr.nn.Sequential(fr.nn.Linear(3, 3), SortAggregation(k))
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([7, 3], generator=generator))
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(263)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'sort': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'sort': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'sort': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'sort': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'sort': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_strict_metadata_bounds_capacity_and_nested_state_preflight(self):
        for k in (-1, True, 2.5):
            with self.assertRaises(ValueError): SortAggregation(k)
        model, x = SortAggregation(2), fr.Tensor.ones([7, 3])
        for call in (lambda: model(fr.Tensor.zeros([0, 3])), lambda: model(fr.Tensor.zeros([7, 0])),
                     lambda: model(x, [0, 1, 0, 2, 2, 2, 4]), lambda: model(x, native(self.index), dim_size=4),
                     lambda: model(x, ptr=fr.Tensor.from_i64([0, 7], [2])), lambda: model(x, max_num_elements=-1),
                     lambda: model(x, dim=-1)):
            with self.assertRaises((ValueError, NotImplementedError)): call()
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x))
        with self.assertRaises(RuntimeError): TorchSort(2)(torch.empty(0, 3))
        with patch('ferro._graph_sort_aggr.cat', side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                SortAggregation(4)(fr.Tensor.ones([1, 1]), [0], dim_size=sys.maxsize // 16 + 1)
        attention = fr.graph.MultiAggregation(['mean', 'sum'], mode='attn', mode_kwargs=dict(in_channels=3, out_channels=6, dropout=.2))
        combined = fr.graph.MultiAggregation([attention, model], mode='sum')
        stream = attention.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'sorted'): combined(x, [0, 1, 0, 2, 2, 2, 4])
        self.assertEqual(stream, attention.multihead_attn.attn_dropout.rng_state.tolist())

    def test_zero_selection_virtual_groups_keeps_connected_gradient_without_dense_work(self):
        x = fr.Tensor.ones([1, 1]).requires_grad_(True)
        with patch('ferro._graph_sort_aggr.cat', side_effect=AssertionError('zero selection reached dense work')):
            output = SortAggregation(0)(x, [0], dim_size=10**12, max_num_elements=0)
            self.assertEqual(output.shape, [10**12, 0])
            output.sum().backward()
            self.assertEqual(x.grad.tolist(), [[0.]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
