"""Actual PyG anti-symmetric graph updates, all adjoints and native state."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import AntiSymmetricConv as TorchAnti, GCNConv as TorchGCN
import ferro as fr
from ferro.graph import AntiSymmetricConv
import test_pyg_modulated as helpers

native = helpers.native


class Local(fr.nn.Module):
    p: float = .2
    bad_rows: bool = False

    def build(self):
        self.lin, self.drop = fr.nn.Linear(3, 3), fr.nn.Dropout(self.p, seed=191)
        self._resets, self._calls = 0, 0

    def reset_parameters(self):
        self._resets += 1
        with fr.no_grad():
            self.lin.weight.tensor().copy_(fr.Tensor.ones([3, 3]) * .125)
            self.lin.bias.tensor().copy_(fr.Tensor.zeros([3]))

    def forward(self, x, edge_index, scale=1.):
        self._calls += 1
        return self.drop(self.lin(x.narrow(0, 0, 1) if self.bad_rows else x)) * scale


class TorchLocal(torch.nn.Module):
    def __init__(self, p=.2):
        super().__init__()
        self.lin, self.drop = torch.nn.Linear(3, 3), torch.nn.Dropout(p)
        self.resets = 0

    def reset_parameters(self):
        self.resets += 1
        with torch.no_grad():
            self.lin.weight.fill_(.125); self.lin.bias.zero_()

    def forward(self, x, edge_index, scale=1.):
        return self.drop(self.lin(x)) * scale


class AntiSymmetricTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2341)
        self.edge = torch.tensor([[0, 1, 2, 3, 1, 2, 4, 1], [1, 2, 0, 4, 2, 2, 3, 1]])

    def test_default_gcn_iterations_strides_weights_empty_and_all_gradients(self):
        for iters, bias, weighted, rows in ((0, True, True, 6), (1, True, True, 6), (3, False, False, 6), (2, True, True, 0)):
            model, oracle = AntiSymmetricConv(3, num_iters=iters, bias=bias), TorchAnti(3, num_iters=iters, bias=bias)
            self.copy(model, oracle)
            edge = self.edge if rows else torch.empty((2, 0), dtype=torch.long)
            weights = (.3 + torch.rand(edge.shape[1])).requires_grad_() if weighted else None
            nw = None if weights is None else native(weights, True)
            x = torch.randn(3, rows, requires_grad=True); nx = native(x, True)
            actual = model(nx.transpose(0, 1), native(edge), nw)
            expected = oracle(x.transpose(0, 1), edge, weights)
            self.gradients(model, oracle, actual, expected, [('x', nx, x)] + ([] if weights is None else [('edge_weight', nw, weights)]))
            if iters:
                grad = model.W.tensor().grad
                self.close(grad, -helpers.torch_value(grad).T)
                self.assertEqual(grad.diagonal().tolist(), [0.] * 3)
            else:
                self.assertIsNone(model.W.tensor().grad)
                value = native(x).transpose(0, 1)
                self.assertIs(model(value, object()), value)

    def test_custom_phi_parameter_activation_negative_coefficients_zero_step_and_buffers(self):
        for activation, kwargs in ((None, {}), ('leaky_relu', {'negative_slope': .17}), ('module', {})):
            for epsilon in (.07, 0.):
                phi, oracle_phi = fr.graph.GCNConv(3, 3, normalize=False, bias=True), TorchGCN(3, 3, normalize=False, bias=True)
                act = fr.nn.Linear(3, 3) if activation == 'module' else activation
                oracle_act = torch.nn.Linear(3, 3) if activation == 'module' else activation
                model = AntiSymmetricConv(3, phi, 3, epsilon, -.15, act, kwargs)
                oracle = TorchAnti(3, oracle_phi, 3, epsilon, -.15, oracle_act, kwargs)
                self.copy(model, oracle)
                identity = torch.randn(3, 3, requires_grad=True)
                with fr.no_grad(): model.eye.copy_(native(identity))
                model.eye.requires_grad_(True)
                oracle.eye = identity
                x, weights = torch.randn(6, 3, requires_grad=True), torch.randn(8, requires_grad=True)
                nx, nw = native(x, True), native(weights, True)
                self.gradients(model, oracle, model(nx, native(self.edge), nw), oracle(x, self.edge, weights), [('x', nx, x), ('weights', nw, weights), ('eye', model.eye, identity)])

    def test_supplied_phi_activation_constructor_reset_and_identity_buffer_preservation(self):
        phi, oracle_phi = Local(), TorchLocal()
        activation, oracle_activation = fr.nn.Linear(3, 3), torch.nn.Linear(3, 3)
        act_before = activation.weight.tensor().tolist()
        model, oracle = AntiSymmetricConv(3, phi, act=activation), TorchAnti(3, oracle_phi, act=oracle_activation)
        self.assertIs(model.phi, phi); self.assertIs(model.act, activation)
        self.assertEqual(phi._resets, oracle_phi.resets)
        self.assertEqual(act_before, activation.weight.tensor().tolist())
        self.assertEqual(model.bias.tensor().tolist(), [0.] * 3)
        self.close(model.eye, oracle.eye)
        with fr.no_grad():
            model.eye.copy_(fr.Tensor.ones([3, 3]) * .25)
            oracle.eye.fill_(.25)
            model.W.tensor().copy_(fr.Tensor.ones([3, 3]) * 19.)
        stream = phi.drop.rng_state.tolist()
        model.reset_parameters(); oracle.reset_parameters()
        self.assertEqual(phi._resets, oracle_phi.resets)
        self.assertEqual(phi.drop.rng_state.tolist(), stream)
        self.assertEqual(model.eye.tolist(), [[.25] * 3] * 3)
        self.assertEqual(act_before, activation.weight.tensor().tolist())
        self.assertTrue(all(abs(v) <= 3 ** -.5 + 1e-7 for v in model.W.tensor().flatten().tolist()))
        with self.assertRaises(AttributeError): model.gamma = .2

    def test_independent_same_masks_recurrent_custom_phi_gradients_and_stream(self):
        model, oracle = AntiSymmetricConv(3, Local(), 4), TorchAnti(3, TorchLocal(), 4)
        self.copy(model, oracle)
        sampler = fr.nn.Dropout(.2, seed=1)
        with fr.no_grad(): sampler.rng_state.copy_(model.phi.drop.rng_state)
        def dropout(value): return value * helpers.torch_value(sampler(fr.Tensor.ones(list(value.shape))))
        oracle.phi.drop.forward = dropout
        x = torch.randn(6, 3, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.edge), scale=.7), oracle(x, self.edge, scale=.7), [('x', nx, x)])
        self.assertEqual(model.phi._calls, 4)
        self.assertEqual(model.phi.drop.rng_state.tolist(), sampler.rng_state.tolist())
        model.eval(); before = model.phi.drop.rng_state.tolist()
        model(native(x), native(self.edge)); self.assertEqual(before, model.phi.drop.rng_state.tolist())

    def test_serialized_alias_buffer_dropout_optimizer_and_generator_continuation(self):
        def factory(gamma=.1):
            model = AntiSymmetricConv(3, Local(), 3, gamma=gamma)
            model.phi.lin.weight = model.W
            with fr.no_grad(): model.eye.copy_(fr.Tensor.ones([3, 3]) * .13)
            return model
        def step(model, optimizer, rng):
            output = model(fr.Tensor.randn([6, 3], generator=rng), native(self.edge), .8)
            loss = output.square().mean(); values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad(); return values
        model, rng = factory(), fr.Generator(419)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'graph': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'graph': other_opt}, generator=other_rng)
        self.assertIs(other.W, other.phi.lin.weight)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'graph': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'graph': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed, changed_rng = factory(.2), fr.Generator(1)
        changed_opt = fr.optim.AdamW(changed.parameters())
        def state(): return {key: value.tolist() for key, value in fr.checkpoint.snapshot_training(changed, optimizers={'graph': changed_opt}, generator=changed_rng).tensors().items()}
        before = state()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'graph': changed_opt}, generator=changed_rng)
        self.assertEqual(before, state())

    def test_metadata_and_capacity_before_callbacks_or_dropout_with_strict_output_shapes(self):
        model, x = AntiSymmetricConv(3, Local(), 2), fr.Tensor.ones([6, 3])
        initial = model.phi.drop.rng_state.tolist()
        for value, edge in ((fr.Tensor.ones([6, 2]), native(self.edge)), (x, fr.Tensor.from_i64([0, 6], [2, 1])), (fr.ones([6, 3], dtype='float64'), native(self.edge))):
            with self.assertRaises((ValueError, TypeError)): model(value, edge)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, native(self.edge)))
        self.assertEqual(model.phi._calls, 0)
        self.assertEqual(model.phi.drop.rng_state.tolist(), initial)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            model(fr.Tensor.ones([1, 3]).expand([sys.maxsize // 12 + 1, 3]), fr.Tensor.from_i64([], [2, 0]))
        self.assertEqual(model.phi._calls, 0)
        with self.assertRaisesRegex(ValueError, 'preserve'):
            AntiSymmetricConv(3, Local(bad_rows=True))(x, native(self.edge))
        with self.assertRaisesRegex(ValueError, 'shape'):
            AntiSymmetricConv(3, Local(), act=fr.nn.Linear(3, 2))(x, native(self.edge))
        for kwargs in (dict(in_channels=0), dict(in_channels=3, num_iters=-1), dict(in_channels=3, num_iters=True), dict(in_channels=3, gamma=float('nan')), dict(in_channels=3, epsilon=float('inf')), dict(in_channels=3, bias=1), dict(in_channels=3, phi=object()), dict(in_channels=3, act=None, act_kwargs={'alpha': 1.})):
            with self.assertRaises((ValueError, TypeError)): AntiSymmetricConv(**kwargs)
        with self.assertRaises(NotImplementedError): AntiSymmetricConv(3, act='unsupported')
        with patch.object(fr.Tensor, 'zeros', side_effect=AssertionError('allocation before capacity check')):
            with self.assertRaisesRegex(ValueError, 'capacity'): AntiSymmetricConv(sys.maxsize // 8 + 1)


if __name__ == '__main__': unittest.main(verbosity=2)
