"""Sparse spectral operators against actual PyG, including live edge gradients."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import SGConv as TorchSG, TAGConv as TorchTAG, ARMAConv as TorchARMA
import ferro as fr
from ferro.graph import SGConv, TAGConv, ARMAConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class SpectralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(857)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=2e-5, rtol=3e-4)

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))

    def compare(self, model, reference, edge=None, nodes=6, weighted=True, strided=False):
        self.copy(model, reference)
        edge = self.edge if edge is None else edge
        base = torch.randn((3, nodes) if strided else (nodes, 3), requires_grad=True)
        x = base.transpose(0, 1) if strided else base
        nb = native(base, True)
        nx = nb.transpose(0, 1) if strided else nb
        weight = (torch.rand(edge.shape[1]) + .4).requires_grad_() if weighted else None
        nw = native(weight, True) if weighted else None
        actual, expected = model(nx, native(edge), nw), reference(x, edge, weight)
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward()
        (expected * probe).sum().backward()
        parameters = dict(reference.named_parameters())
        pairs = [('x', nb, base)] + ([] if weight is None else [('edge_weight', nw, weight)])
        pairs += [(name, parameter.tensor(), parameters[name]) for name, parameter in model.named_parameters()]
        for name, actual, expected in pairs:
            with self.subTest(gradient=name):
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    self.assertIsNotNone(actual.grad)
                    self.close(actual.grad, expected.grad)

    def test_sg_propagation_hops_loops_bias_and_all_gradients(self):
        for hops, loops, bias, weighted in ((0, True, True, True), (1, True, False, False),
                                            (3, True, True, True), (2, False, False, True)):
            with self.subTest(K=hops, loops=loops, bias=bias, weighted=weighted):
                options = dict(K=hops, add_self_loops=loops, bias=bias)
                self.compare(SGConv(3, 4, **options), TorchSG(3, 4, **options), weighted=weighted, strided=True)

    def test_tag_polynomial_normalized_and_raw_all_gradients(self):
        for hops, normalize, bias, weighted in ((0, True, True, True), (1, False, False, False),
                                               (3, True, True, True), (2, False, False, True)):
            with self.subTest(K=hops, normalize=normalize, bias=bias, weighted=weighted):
                options = dict(K=hops, normalize=normalize, bias=bias)
                self.compare(TAGConv(3, 4, **options), TorchTAG(3, 4, **options), weighted=weighted)

    def test_arma_stacked_shared_and_unshared_all_gradients(self):
        for stacks, layers, shared, bias, act in ((1, 1, False, True, 'relu'), (3, 4, False, False, 'tanh'),
                                                (2, 3, True, True, None), (2, 1, True, False, 'relu')):
            with self.subTest(stacks=stacks, layers=layers, shared=shared, bias=bias, act=act):
                options = dict(num_stacks=stacks, num_layers=layers, shared_weights=shared, bias=bias)
                activation = None if act is None else torch.nn.Tanh() if act == 'tanh' else torch.nn.ReLU()
                model = ARMAConv(3, 4, act=act, **options)
                reference = TorchARMA(3, 4, act=activation, **options)
                self.compare(model, reference, strided=True)

    def test_arma_registered_trainable_activation_and_dropout_boundaries(self):
        self.compare(ARMAConv(3, 4, 2, 3, act=fr.nn.Linear(4, 4)),
                     TorchARMA(3, 4, 2, 3, act=torch.nn.Linear(4, 4)))
        for dropout in (.35, 1.):
            model, reference = ARMAConv(3, 4, 2, 3, dropout=dropout), TorchARMA(3, 4, 2, 3, dropout=dropout)
            if dropout < 1:
                model.eval()
                reference.eval()
            self.compare(model, reference)

    def test_arma_nonzero_dropout_matches_torch_with_identical_root_masks(self):
        model = ARMAConv(3, 4, 3, 4, dropout=.35, act='tanh')
        reference = TorchARMA(3, 4, 3, 4, dropout=.35, act=torch.nn.Tanh())
        sampler = fr.nn.Dropout(.35, seed=1)
        with fr.no_grad():
            sampler.rng_state.copy_(model.root_dropout.rng_state)
        masks = [torch_value(sampler(fr.Tensor.ones([1, 6, 3]))) for _ in range(4)]
        calls = []
        def fixed_dropout(value, p=.5, training=True, inplace=False):
            self.assertEqual(p, .35)
            self.assertTrue(training)
            self.assertFalse(inplace)
            self.assertEqual(list(value.shape), [1, 6, 3])
            mask = masks[len(calls)]
            calls.append(mask)
            return value * mask
        # The masks are shared external data here; Torch independently computes
        # graph propagation, stacked transforms and every derivative.
        with patch('torch.nn.functional.dropout', side_effect=fixed_dropout):
            self.compare(model, reference)
        self.assertEqual(len(calls), 4)
        self.assertEqual(model.root_dropout.rng_state.tolist(), sampler.rng_state.tolist())

    def test_empty_graphs_and_isolated_nodes_preserve_gradient_contract(self):
        edge = torch.empty(2, 0, dtype=torch.int64)
        for nodes in (0, 3):
            for model, reference in ((SGConv(3, 4, K=2), TorchSG(3, 4, K=2)),
                                     (TAGConv(3, 4, K=2), TorchTAG(3, 4, K=2)),
                                     (ARMAConv(3, 4, 2, 3), TorchARMA(3, 4, 2, 3))):
                with self.subTest(nodes=nodes, model=type(model).__name__):
                    self.compare(model, reference, edge=edge, nodes=nodes)

    def test_duplicate_self_loops_without_insertion_and_native_topology(self):
        edge = torch.tensor([[0, 0, 1, 2, 0, 1, 2], [0, 0, 1, 2, 1, 2, 0]])
        for model, reference in ((SGConv(3, 4, K=2, add_self_loops=False), TorchSG(3, 4, K=2, add_self_loops=False)),
                                 (TAGConv(3, 4), TorchTAG(3, 4)), (ARMAConv(3, 4, 2, 3), TorchARMA(3, 4, 2, 3))):
            self.compare(model, reference, edge=edge, nodes=3)
        x = native(torch.randn(3, 3))
        model = SGConv(3, 4)
        with self.assertRaises(NotImplementedError):
            model(x, native(edge), native(torch.ones(edge.shape[1])))
        edges = fr.graph.EdgeIndex(3, edge[0].tolist(), edge[1].tolist())
        self.assertEqual(model(x, edges).tolist(), model(x, native(edge)).tolist())

    def test_arma_dropout_checkpoint_exact_optimizer_and_stream_continuation(self):
        def factory():
            return ARMAConv(3, 4, 3, 4, shared_weights=True, dropout=.3, act='tanh')
        model, rng = factory(), fr.Generator(73)
        opt = fr.optim.AdamW(model.parameters(), lr=.015)
        x, edge, weight = native(torch.randn(6, 3)), native(self.edge), native(torch.rand(9) + .4)
        def step(model, optimizer, generator):
            current = x + fr.Tensor.rand([6, 3], generator=generator) * .02
            result = model(current, edge, weight)
            loss = result.square().mean()
            values = result.tolist(), loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return values
        step(model, opt, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            state.save(directory / 'mid')
            expected = [step(model, opt, rng) for _ in range(3)]
            final = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=4)
            clone, clone_rng = factory(), fr.Generator(999)
            clone_opt = fr.optim.AdamW(clone.parameters(), lr=.9)
            self.assertEqual(fr.checkpoint.load_training(directory / 'mid', clone, optimizers={'main': clone_opt}, generator=clone_rng), 1)
            self.assertEqual(expected, [step(clone, clone_opt, clone_rng) for _ in range(3)])
            actual = fr.checkpoint.snapshot_training(clone, optimizers={'main': clone_opt}, generator=clone_rng, step=4)
            for checkpoint, name in ((final, 'expected'), (actual, 'actual')):
                checkpoint.save(directory / name)
            records = [json.loads((directory / name / 'checkpoint.json').read_bytes()) for name in ('expected', 'actual')]
            payloads = [(directory / name / record.pop('generation') / 'model.safetensors').read_bytes() for name, record in zip(('expected', 'actual'), records)]
            self.assertEqual(records[0], records[1])
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(fr.Tensor.rand([8], generator=rng).tolist(), fr.Tensor.rand([8], generator=clone_rng).tolist())
        model.eval()
        before = model.root_dropout.rng_state.tolist()
        self.assertEqual(model(x, edge, weight).tolist(), model(x, edge, weight).tolist())
        self.assertEqual(before, model.root_dropout.rng_state.tolist())

    def test_checkpoint_aliases_modes_and_incompatible_configuration(self):
        model = TAGConv(3, 4, K=2)
        model.lins[2].weight = model.lins[0].weight
        model.lins[1].weight.trainable = False
        model.eval()
        state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1))
        other = TAGConv(3, 4, K=2)
        other.lins[2].weight = other.lins[0].weight
        state.restore(other, optimizers={}, generator=fr.Generator(2))
        self.assertFalse(other.training)
        self.assertFalse(other.lins[1].weight.trainable)
        self.assertIs(other.lins[0].weight, other.lins[2].weight)
        changed_config = TAGConv(3, 4, K=2, normalize=False)
        changed_config.lins[2].weight = changed_config.lins[0].weight
        for target in (TAGConv(3, 4, K=2), changed_config):
            before = [parameter.tensor().tolist() for parameter in target.parameters()]
            with self.assertRaises(ValueError):
                state.restore(target, optimizers={}, generator=fr.Generator(3))
            self.assertEqual(before, [parameter.tensor().tolist() for parameter in target.parameters()])
        arma = ARMAConv(3, 4, 2, 1, shared_weights=True)
        state = fr.checkpoint.snapshot_training(arma, optimizers={}, generator=fr.Generator(1))
        with self.assertRaises(ValueError):
            state.restore(ARMAConv(3, 4, 2, 1, shared_weights=False), optimizers={}, generator=fr.Generator(3))

    def test_strict_shapes_capacities_cache_options_and_invalid_stream_isolation(self):
        for constructor in (lambda: SGConv(-1, 3), lambda: TAGConv(3, 0), lambda: SGConv(3, 4, K=-1),
                            lambda: TAGConv(3, 4, K=True), lambda: ARMAConv(3, 4, num_layers=0),
                            lambda: ARMAConv(3, 4, num_stacks=False), lambda: ARMAConv(3, 4, dropout=float('nan')),
                            lambda: SGConv(sys.maxsize, 2), lambda: TAGConv(3, 4, K=sys.maxsize),
                            lambda: ARMAConv(3, 4, num_stacks=sys.maxsize)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        for constructor in (lambda: SGConv(3, 4, cached=True), lambda: TAGConv(3, 4, flow='target_to_source'),
                            lambda: ARMAConv(3, 4, act=lambda x: x.tanh()), lambda: SGConv(3, 4, aggr='mean')):
            with self.assertRaises(NotImplementedError):
                constructor()
        model = ARMAConv(3, 4, 2, 3, dropout=.4)
        x, edge = native(torch.randn(6, 3)), native(self.edge)
        before = model.root_dropout.rng_state.tolist()
        for call in (lambda: model(x, edge, native(torch.ones(8))), lambda: model(x, native(torch.tensor([[6], [0]]))),
                     lambda: model(native(torch.randn(6, 4)), edge), lambda: model([x, x], edge),
                     lambda: model(fr.tensor([[1., 2., 3.]], dtype='float64'), edge)):
            with self.assertRaises((ValueError, TypeError)):
                call()
            self.assertEqual(before, model.root_dropout.rng_state.tolist())
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(x, edge))
        huge_broadcast = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaises(ValueError):
            SGConv(1, 1)(huge_broadcast, fr.Tensor.from_i64([], [2, 0]))
        with self.assertRaises(AttributeError):
            model.num_layers = 5


if __name__ == '__main__':
    unittest.main(verbosity=2)
