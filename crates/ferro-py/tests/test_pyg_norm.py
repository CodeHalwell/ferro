"""Graph normalization against actual PyG, including singular norm gradients."""
import sys
import unittest

import torch
from torch_geometric.nn import (GraphNorm as TorchGraphNorm, PairNorm as TorchPairNorm,
    MeanSubtractionNorm as TorchMeanSubtractionNorm, GraphSizeNorm as TorchGraphSizeNorm,
    LayerNorm as TorchLayerNorm)
import ferro as fr
from ferro.graph import GraphNorm, PairNorm, MeanSubtractionNorm, GraphSizeNorm, LayerNorm


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class GraphNormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(877)

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4)

    def compare(self, model, reference, batch=None, batch_size=None, data=None, strided=False, randomize=True):
        if randomize:
            with torch.no_grad():
                for parameter in reference.parameters():
                    parameter.uniform_(-.5, 1.2)
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))
        data = torch.randn(7, 3) if data is None else data
        base = data.transpose(0, 1).clone().requires_grad_() if strided else data.clone().requires_grad_()
        x = base.transpose(0, 1) if strided else base
        nb = native(base, True)
        nx = nb.transpose(0, 1) if strided else nb
        actual = model(nx, None if batch is None else native(batch), batch_size)
        expected = reference(x, batch, batch_size)
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward()
        (expected * probe).sum().backward()
        self.close(nb.grad, base.grad)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.close(parameter.tensor().grad, parameters[name].grad)
        return actual, expected

    def test_graphnorm_learned_mean_scaling_unsorted_graphs_and_all_gradients(self):
        batch = torch.tensor([2, 0, 2, 4, 0, 2, 4])
        for groups, count in ((None, None), (batch, None), (batch, 7)):
            with self.subTest(batch=groups is not None, count=count):
                self.compare(GraphNorm(3), TorchGraphNorm(3), groups, count, strided=True)

    def test_pairnorm_individual_and_graph_scaling_all_gradients(self):
        batch = torch.tensor([2, 0, 2, 4, 0, 2, 4])
        for individual in (False, True):
            for groups, count in ((None, None), (batch, 7)):
                with self.subTest(individual=individual, batched=groups is not None):
                    self.compare(PairNorm(scale=-1.3, scale_individually=individual, eps=.002),
                                 TorchPairNorm(scale=-1.3, scale_individually=individual, eps=.002), groups, count)

    def test_mean_subtraction_and_size_scaling_batched_and_single_graph(self):
        batch = torch.tensor([2, 0, 2, 4, 0, 2, 4])
        for cls, reference in ((MeanSubtractionNorm, TorchMeanSubtractionNorm), (GraphSizeNorm, TorchGraphSizeNorm)):
            for groups, count in ((None, None), (batch, 7)):
                with self.subTest(norm=cls.__name__, batched=groups is not None):
                    self.compare(cls(), reference(), groups, count, strided=True)

    def test_graph_layernorm_modes_affine_and_epsilon_placement(self):
        batch = torch.tensor([2, 0, 2, 4, 0, 2, 4])
        for mode, affine, groups, count in (('graph', True, None, None), ('graph', False, batch, 7),
                                           ('node', True, batch, 7), ('node', False, None, None)):
            with self.subTest(mode=mode, affine=affine, batched=groups is not None):
                self.compare(LayerNorm(3, affine=affine, mode=mode), TorchLayerNorm(3, affine=affine, mode=mode), groups, count)
        data = torch.tensor([[.01, 0., .01], [0., .01, 0.]])
        unbatched, _ = self.compare(LayerNorm(3, eps=.1, affine=False), TorchLayerNorm(3, eps=.1, affine=False), data=data)
        batched, _ = self.compare(LayerNorm(3, eps=.1, affine=False), TorchLayerNorm(3, eps=.1, affine=False),
                                  batch=torch.tensor([0, 0]), batch_size=1, data=data)
        self.assertGreater((torch_value(unbatched) - torch_value(batched)).abs().max().item(), .02)
        model, reference = LayerNorm(3, mode='node'), TorchLayerNorm(3, mode='node')
        self.compare(model, reference, batch=torch.tensor([-1]), batch_size=-7)

    def test_constant_and_singleton_graphs_zero_norm_subgradients(self):
        data = torch.ones(5, 3) * 2.
        batch = torch.tensor([0, 1, 1, 2, 2])
        for model, reference in ((GraphNorm(3), TorchGraphNorm(3)),
                                 (PairNorm(scale_individually=True), TorchPairNorm(scale_individually=True)),
                                 (PairNorm(), TorchPairNorm()), (LayerNorm(3), TorchLayerNorm(3)),
                                 (MeanSubtractionNorm(), TorchMeanSubtractionNorm())):
            with self.subTest(norm=type(model).__name__):
                self.compare(model, reference, batch, 4, data=data, randomize=False)
        self.compare(LayerNorm(3), TorchLayerNorm(3), data=data, randomize=False)

    def test_empty_features_and_empty_graph_batches(self):
        for cls, reference in ((GraphNorm, TorchGraphNorm), (LayerNorm, TorchLayerNorm)):
            for batch, size in ((None, None), (torch.empty(0, dtype=torch.int64), 3)):
                with self.subTest(norm=cls.__name__, batch=batch is not None):
                    self.compare(cls(3), reference(3), batch, size, data=torch.empty(0, 3))
        for cls, reference in ((PairNorm, TorchPairNorm), (MeanSubtractionNorm, TorchMeanSubtractionNorm),
                               (GraphSizeNorm, TorchGraphSizeNorm)):
            for data, batch, size in ((torch.empty(0, 3), None, None),
                                      (torch.empty(0, 3), torch.empty(0, dtype=torch.int64), 0),
                                      (torch.empty(4, 0), torch.tensor([0, 2, 0, 2]), 4)):
                with self.subTest(norm=cls.__name__, shape=list(data.shape)):
                    self.compare(cls(), reference(), batch, size, data=data)

    def test_reset_and_checkpoint_configuration_aliases_and_optimizer_continuation(self):
        model, rng = GraphNorm(3), fr.Generator(43)
        model.mean_scale = model.weight
        model.bias.trainable = False
        opt = fr.optim.AdamW(model.parameters(), lr=.01)
        batch = native(torch.tensor([2, 0, 2, 4, 0, 2, 4]))
        def step(current, optimizer, generator):
            x = fr.Tensor.rand([7, 3], generator=generator)
            output = current(x, batch, 6)
            loss = output.square().mean()
            result = output.tolist(), loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return result
        step(model, opt, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'norm': opt}, generator=rng, step=1)
        expected = step(model, opt, rng)
        other, other_rng = GraphNorm(3), fr.Generator(999)
        other.mean_scale = other.weight
        other_opt = fr.optim.AdamW(other.parameters(), lr=.9)
        state.restore(other, optimizers={'norm': other_opt}, generator=other_rng)
        self.assertFalse(other.bias.trainable)
        self.assertIs(other.mean_scale, other.weight)
        self.assertEqual(expected, step(other, other_opt, other_rng))
        for p, q in zip(model.parameters(), other.parameters()):
            self.assertEqual(p.tensor().tolist(), q.tensor().tolist())
        self.assertEqual(fr.Tensor.rand([5], generator=rng).tolist(), fr.Tensor.rand([5], generator=other_rng).tolist())
        changed = GraphNorm(3, eps=.001)
        changed.mean_scale = changed.weight
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError):
            state.restore(changed, optimizers={'norm': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])
        model.reset_parameters()
        self.assertEqual(model.weight.tensor().tolist(), [1.] * 3)
        self.assertEqual(model.bias.tensor().tolist(), [0.] * 3)
        layer = LayerNorm(3)
        layer.reset_parameters()
        self.assertEqual(layer.weight.tensor().tolist(), [1.] * 3)

    def test_strict_shapes_ids_devices_configs_and_capture(self):
        for constructor in (lambda: GraphNorm(0), lambda: LayerNorm(sys.maxsize), lambda: PairNorm(scale=float('inf')),
                            lambda: GraphNorm(3, eps=0.), lambda: PairNorm(eps=-1.), lambda: LayerNorm(3, eps=float('nan')),
                            lambda: LayerNorm(3, mode='batch'), lambda: PairNorm(scale_individually=1)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        with self.assertRaises(NotImplementedError):
            GraphNorm(3, device='cuda')
        model = GraphNorm(3)
        x = native(torch.randn(7, 3))
        for call in (lambda: model(x, native(torch.ones(7))), lambda: model(x, native(torch.zeros(6, dtype=torch.int64))),
                     lambda: model(x, native(torch.tensor([0, 0, 0, 0, 0, 0, -1]))),
                     lambda: model(x, native(torch.ones(7, dtype=torch.int64)), 1),
                     lambda: model(x, native(torch.zeros(7, dtype=torch.int64)), sys.maxsize),
                     lambda: model(native(torch.empty(0, 3)), native(torch.empty(0, dtype=torch.int64))),
                     lambda: model(native(torch.randn(7, 4))), lambda: model([x, x]),
                     lambda: model(fr.tensor([[1., 2., 3.]], dtype='float64'))):
            with self.assertRaises((ValueError, TypeError)):
                call()
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(x))
        huge_empty_features = fr.Tensor.zeros([1, 0]).expand(sys.maxsize // 8 + 1, 0)
        with self.assertRaises(ValueError):
            GraphSizeNorm()(huge_empty_features)
        with self.assertRaises(ValueError):
            GraphSizeNorm()(fr.Tensor.zeros([0, 0]), fr.Tensor.from_i64([], [0]), sys.maxsize // 8 + 1)
        with self.assertRaises(AttributeError):
            model.eps = .1

    def test_parameterless_options_and_layer_mode_are_checkpoint_bound(self):
        for model, incompatible in ((PairNorm(scale=.7), PairNorm(scale=.8)),
                                    (PairNorm(), PairNorm(scale_individually=True)),
                                    (LayerNorm(3, mode='graph'), LayerNorm(3, mode='node'))):
            with self.subTest(norm=type(model).__name__):
                state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1))
                before = [parameter.tensor().tolist() for parameter in incompatible.parameters()]
                rng = fr.Generator(91)
                expected_rng = fr.Generator(91)
                with self.assertRaises(ValueError):
                    state.restore(incompatible, optimizers={}, generator=rng)
                self.assertEqual(before, [parameter.tensor().tolist() for parameter in incompatible.parameters()])
                self.assertEqual(fr.Tensor.rand([5], generator=rng).tolist(), fr.Tensor.rand([5], generator=expected_rng).tolist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
