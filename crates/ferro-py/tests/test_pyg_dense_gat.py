"""Actual PyG dense graph attention, topology-only adjacencies and state."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import DenseGATConv as TorchDenseGAT
import ferro as fr
from ferro.graph import DenseGATConv
import test_pyg_modulated as helpers


def native(value, grad=False):
    if value.dtype == torch.bool: return fr.Tensor.from_bool(value.reshape(-1).tolist(), list(value.shape))
    return helpers.native(value, grad)


class DenseGATTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2521)
        self.adj = torch.tensor([[0., 2., -1., 0.], [3., 0., 0., 1.], [0., 0., 1., 1.], [2., 0., 1., 0.]])

    def test_actual_pyg_heads_concat_strides_topology_and_float_mask_all_gradients(self):
        for heads, concat, bias, loops in ((1, True, True, True), (2, True, False, True), (3, False, True, False)):
            model, oracle = DenseGATConv(3, 2, heads, concat, .13, bias=bias), TorchDenseGAT(3, 2, heads, concat, .13, bias=bias)
            self.copy(model, oracle)
            base = torch.randn(2, 3, 4, requires_grad=True)
            adjacency = self.adj.clone().requires_grad_(); mask = torch.rand(2, 4, requires_grad=True)
            nb, na, nm = native(base, True), native(adjacency, True), native(mask, True)
            self.gradients(model, oracle, model(nb.transpose(1, 2), na, nm, loops), oracle(base.transpose(1, 2), adjacency, mask, loops), [('x', nb, base), ('adj', na, adjacency), ('mask', nm, mask)])
            self.assertIsNone(na.grad)

    def test_typed_adjacency_broadcasting_empty_axes_and_pyg_concat_boundary(self):
        for dtype in (torch.float32, torch.bool, torch.int64):
            model, oracle = DenseGATConv(3, 2, heads=2), TorchDenseGAT(3, 2, heads=2)
            self.copy(model, oracle)
            x = torch.randn(4, 3, requires_grad=True); nx = native(x, True)
            adj = self.adj.to(dtype); mask = torch.tensor([[True, False, True, True], [True, True, False, True]])
            self.gradients(model, oracle, model(nx, native(adj), native(mask)), oracle(x, adj, mask), [('x', nx, x)])
        for bx, ba, nodes, concat in ((1, 2, 4, False), (0, 1, 4, True), (1, 0, 4, False), (1, 2, 0, True), (2, 1, 0, False), (0, 1, 0, True)):
            model, oracle = DenseGATConv(3, 2, heads=2, concat=concat), TorchDenseGAT(3, 2, heads=2, concat=concat)
            self.copy(model, oracle)
            x = torch.randn(bx, nodes, 3, requires_grad=True); adj = torch.ones(ba, nodes, nodes, requires_grad=True)
            nx, na = native(x, True), native(adj, True)
            self.gradients(model, oracle, model(nx, na), oracle(x, adj), [('x', nx, x), ('adj', na, adj)])
        model, oracle = DenseGATConv(3, 2, dropout=.2), TorchDenseGAT(3, 2, dropout=.2)
        before = model.attn_dropout.rng_state.tolist()
        with self.assertRaises(RuntimeError): oracle(torch.ones(1, 4, 3), torch.ones(2, 4, 4))
        with self.assertRaisesRegex(ValueError, 'concat'): model(fr.Tensor.ones([1, 4, 3]), fr.Tensor.ones([2, 4, 4]))
        self.assertEqual(before, model.attn_dropout.rng_state.tolist())

    def test_strided_mask_reshape_extension_matches_pyg_contiguous_copy_gradients(self):
        model, oracle = DenseGATConv(3, 2, heads=2), TorchDenseGAT(3, 2, heads=2)
        self.copy(model, oracle)
        x = torch.randn(2, 3, 3, requires_grad=True)
        base = torch.rand(2, 3, requires_grad=True)
        nx, nm = native(x, True), native(base, True)
        with self.assertRaises(RuntimeError): oracle(x, torch.ones(3, 3), base.t())
        actual = model(nx, fr.Tensor.ones([3, 3]), nm.transpose(0, 1))
        expected = oracle(x, torch.ones(3, 3), base.t().contiguous())
        self.gradients(model, oracle, actual, expected, [('x', nx, x), ('mask base', nm, base)])

    def test_independent_same_attention_mask_and_eval(self):
        model, oracle = DenseGATConv(3, 2, heads=2, dropout=.3), TorchDenseGAT(3, 2, heads=2, dropout=.3)
        self.copy(model, oracle)
        sampler = fr.nn.Dropout(.3, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(model.attn_dropout.rng_state)
        shapes = []
        def dropout(value, p=.5, training=True, inplace=False):
            self.assertEqual(p, .3); self.assertTrue(training); self.assertFalse(inplace)
            shapes.append(tuple(value.shape))
            return value * helpers.torch_value(sampler(fr.Tensor.ones(list(value.shape))))
        x = torch.randn(2, 4, 3, requires_grad=True); nx = native(x, True)
        actual = model(nx, native(self.adj))
        with patch('torch.nn.functional.dropout', side_effect=dropout): expected = oracle(x, self.adj)
        self.gradients(model, oracle, actual, expected, [('x', nx, x)])
        self.assertEqual(shapes, [(2, 4, 4, 2)])
        self.assertEqual(model.attn_dropout.rng_state.tolist(), sampler.rng_state.tolist())
        model.eval(); oracle.eval(); before = model.attn_dropout.rng_state.tolist()
        self.close(model(native(x), native(self.adj)), oracle(x, self.adj))
        self.assertEqual(before, model.attn_dropout.rng_state.tolist())

    def test_all_masked_rows_keep_actual_pyg_nonfinite_values_and_gradients(self):
        for p in (0., 1.):
            model, oracle = DenseGATConv(3, 2, dropout=p), TorchDenseGAT(3, 2, dropout=p)
            self.copy(model, oracle)
            x = torch.randn(1, 4, 3, requires_grad=True); nx = native(x, True)
            actual, expected = model(nx, fr.Tensor.zeros([4, 4]), add_loop=False), oracle(x, torch.zeros(4, 4), add_loop=False)
            torch.testing.assert_close(helpers.torch_value(actual), expected, equal_nan=True)
            actual.sum().backward(); expected.sum().backward()
            parameters = dict(oracle.named_parameters())
            for name, value, reference in [('x', nx, x)] + [(name, parameter.tensor(), parameters[name]) for name, parameter in model.named_parameters()]:
                torch.testing.assert_close(helpers.torch_value(value.grad), reference.grad, equal_nan=True, atol=3e-5, rtol=4e-4, msg=name)

    def test_serialized_alias_dropout_optimizer_generator_and_reset_bounds(self):
        def factory(slope=.2):
            model = DenseGATConv(3, 2, heads=2, negative_slope=slope, dropout=.25)
            model.att_dst = model.att_src
            return model
        def step(model, optimizer, rng):
            output = model(fr.Tensor.randn([2, 4, 3], generator=rng), native(self.adj))
            loss = output.square().mean(); values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad(); return values
        model, rng = factory(), fr.Generator(503)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'dense': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'dense': other_opt}, generator=other_rng)
        self.assertIs(other.att_src, other.att_dst)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'dense': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'dense': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed, changed_rng = factory(.3), fr.Generator(1)
        changed_opt = fr.optim.AdamW(changed.parameters())
        def state(): return {key: value.tolist() for key, value in fr.checkpoint.snapshot_training(changed, optimizers={'dense': changed_opt}, generator=changed_rng).tensors().items()}
        before = state()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'dense': changed_opt}, generator=changed_rng)
        self.assertEqual(before, state())
        stream = model.attn_dropout.rng_state.tolist(); model.reset_parameters()
        self.assertEqual(stream, model.attn_dropout.rng_state.tolist())
        self.assertEqual(model.bias.tensor().tolist(), [0.] * 4)
        self.assertTrue(all(abs(value) <= math.sqrt(6. / 7.) + 1e-7 for value in model.lin.weight.tensor().flatten().tolist()))
        self.assertTrue(all(abs(value) <= math.sqrt(6. / 4.) + 1e-7 for value in model.att_src.tensor().flatten().tolist()))

    def test_metadata_mask_and_score_capacity_preflight_before_dropout(self):
        model = DenseGATConv(3, 2, heads=2, dropout=.3)
        x, adj = fr.Tensor.ones([2, 4, 3]), native(self.adj)
        before = model.attn_dropout.rng_state.tolist()
        cases = [(x, fr.Tensor.ones([3, 3]), None), (fr.Tensor.ones([2, 4, 2]), adj, None),
                 (x, fr.Tensor.ones([3, 4, 4]), None), (x, adj, fr.Tensor.ones([3, 4])),
                 (x, adj, fr.Tensor.ones([5])), (fr.Tensor.ones([2, 0, 3]), fr.Tensor.ones([0, 0]), fr.Tensor.ones([0]))]
        with patch.object(model.lin, 'forward', side_effect=AssertionError('projection before metadata rejection')):
            for value, edges, mask in cases:
                with self.assertRaises(ValueError): model(value, edges, mask)
        self.assertEqual(before, model.attn_dropout.rng_state.tolist())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, adj))
        with self.assertRaises(TypeError): model(x, adj, add_loop=1)
        nodes = math.isqrt(sys.maxsize // 8) + 1
        virtual_x = fr.Tensor.ones([1, 1, 3]).expand([1, nodes, 3])
        virtual_adj = fr.Tensor.ones([1, 1]).expand([nodes, nodes])
        with patch.object(model.lin, 'forward', side_effect=AssertionError('projection before score capacity rejection')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(virtual_x, virtual_adj)
        with patch('ferro._graph_dense_gat.Linear', side_effect=AssertionError('parameters before capacity rejection')):
            with self.assertRaisesRegex(ValueError, 'capacity'): DenseGATConv(1, sys.maxsize // 8 + 1)
        for kwargs in (dict(heads=0), dict(dropout=float('nan')), dict(negative_slope=float('inf')), dict(concat=1), dict(bias=1)):
            with self.assertRaises((ValueError, TypeError)): DenseGATConv(3, 2, **kwargs)


if __name__ == '__main__': unittest.main(verbosity=2)
