"""Actual PyG signed propagation and native checkpoint gradient references."""
import sys
import tempfile
import unittest

import torch
from torch_geometric.nn import SignedConv as TorchSigned
import ferro as fr
from ferro.graph import SignedConv
import test_pyg_modulated as helpers

native = helpers.native


class SignedTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1811)
        self.positive = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])
        self.negative = torch.tensor([[4, 3, 2, 1, 1, 0], [2, 0, 4, 3, 3, 0]])

    def test_first_and_subsequent_homogeneous_bipartite_all_gradients(self):
        for first, bipartite, aggr, bias in ((True, False, 'mean', True), (False, True, 'mean', True),
                                            (True, True, 'add', False), (False, False, 'sum', False),
                                            (True, True, 'max', True), (False, False, 'min', True)):
            with self.subTest(first=first, bipartite=bipartite, aggr=aggr):
                model, oracle = SignedConv(3, 4, first, bias=bias, aggr=aggr), TorchSigned(3, 4, first, bias=bias, aggr=aggr)
                self.copy(model, oracle)
                width = 3 if first else 6
                base = torch.randn(width, 6, requires_grad=True)
                source = base.transpose(0, 1)
                target = torch.randn(5, width, requires_grad=True) if bipartite else source
                nb = native(base, True)
                ns, nt = nb.transpose(0, 1), native(target, True) if bipartite else None
                actual = model((ns, nt) if bipartite else ns, native(self.positive), native(self.negative))
                expected = oracle((source, target) if bipartite else source, self.positive, self.negative)
                self.gradients(model, oracle, actual, expected, [('source', nb, base)] + ([('target', nt, target)] if bipartite else []))

    def test_empty_sign_branches_nodes_and_native_topology(self):
        empty = torch.empty(2, 0, dtype=torch.int64)
        for first, nodes, positive, negative in ((True, 6, empty, self.negative), (False, 6, self.positive, empty),
                                                (True, 4, empty, empty), (False, 0, empty, empty)):
            model, oracle = SignedConv(3, 4, first), TorchSigned(3, 4, first)
            self.copy(model, oracle)
            x = torch.randn(nodes, 3 if first else 6, requires_grad=True)
            nx = native(x, True)
            edge = fr.graph.EdgeIndex(nodes, positive[0].tolist(), positive[1].tolist())
            self.gradients(model, oracle, model(nx, edge, native(negative)), oracle(x, positive, negative), [('x', nx, x)])

    def test_two_layer_signed_network_all_parameter_and_input_gradients(self):
        class Network(fr.nn.Module):
            def build(self):
                self.first = SignedConv(3, 4, True)
                self.second = SignedConv(4, 2, False)
            def forward(self, x, positive, negative):
                return self.second(self.first(x, positive, negative).tanh(), positive, negative)
        class Oracle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = TorchSigned(3, 4, True)
                self.second = TorchSigned(4, 2, False)
            def forward(self, x, positive, negative):
                return self.second(self.first(x, positive, negative).tanh(), positive, negative)
        model, oracle = Network(), Oracle()
        self.copy(model, oracle)
        x = torch.randn(6, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.positive), native(self.negative)), oracle(x, self.positive, self.negative), [('x', nx, x)])

    def test_disjoint_batch_and_edge_permutation_semantics(self):
        model, oracle = SignedConv(3, 4, False), TorchSigned(3, 4, False)
        self.copy(model, oracle)
        x = torch.randn(12, 6, requires_grad=True)
        positive, negative = torch.cat([self.positive, self.positive + 6], 1), torch.cat([self.negative, self.negative + 6], 1)
        positive, negative = positive[:, torch.randperm(positive.shape[1])], negative[:, torch.randperm(negative.shape[1])]
        nx = native(x, True)
        output = model(nx, native(positive), native(negative))
        self.gradients(model, oracle, output, oracle(x, positive, negative), [('x', nx, x)])
        self.close(output, torch.cat([oracle(x[:6], self.positive, self.negative), oracle(x[6:], self.positive, self.negative)]))

    def test_saved_optimizer_continuation_tied_modules_and_configuration(self):
        def factory(aggr='mean'):
            model = SignedConv(3, 4, True, aggr=aggr)
            model.lin_neg_l = model.lin_pos_l
            return model
        model, generator = factory(), fr.Generator(191)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        positive, negative = native(self.positive), native(self.negative)
        def step(model, optimizer, generator):
            output = model(fr.Tensor.rand([6, 3], generator=generator), positive, negative)
            loss = output.square().mean()
            result = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return result
        step(model, optimizer, generator)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'signed': optimizer}, generator=generator, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state')
            saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, generator) for _ in range(3)]
        other, rng = factory(), fr.Generator(2)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'signed': other_opt}, generator=rng)
        self.assertIs(other.lin_pos_l, other.lin_neg_l)
        self.assertEqual(expected, [step(other, other_opt, rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'signed': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'signed': other_opt}, generator=rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left: self.assertEqual(left[name].tolist(), right[name].tolist(), name)
        for changed in (factory('sum'), SignedConv(3, 4, True)):
            before = [p.tensor().tolist() for p in changed.parameters()]
            with self.assertRaises(ValueError):
                saved.restore(changed, optimizers={'signed': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
            self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_strict_shapes_topology_configuration_and_capacity(self):
        for factory in (lambda: SignedConv(-1, 4, True), lambda: SignedConv(3, 0, True),
                        lambda: SignedConv(3, 4, 1), lambda: SignedConv(3, 4, True, bias=1),
                        lambda: SignedConv(sys.maxsize, 1, False)):
            with self.assertRaises((TypeError, ValueError)): factory()
        for options in ({'flow': 'target_to_source'}, {'aggr': 'mul'}, {'node_dim': 1}):
            with self.assertRaises(NotImplementedError): SignedConv(3, 4, True, **options)
        model, x = SignedConv(3, 4, False), fr.Tensor.ones([6, 6])
        positive, negative = native(self.positive), native(self.negative)
        for call in (lambda: model(fr.Tensor.ones([6, 3]), positive, negative), lambda: model((x, None), positive, negative),
                     lambda: model(x, positive, fr.Tensor.from_i64([0, 6], [2, 1])),
                     lambda: model(x, positive, fr.Tensor.ones([2, 1])), lambda: model(fr.Tensor.ones([6, 6]).to(dtype='float64'), positive, negative)):
            with self.assertRaises((ValueError, TypeError)): call()
        with self.assertRaises(AttributeError): model.first_aggr = True
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, positive, negative))
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8, 1)
        empty = fr.Tensor.from_i64([], [2, 0])
        with self.assertRaisesRegex(ValueError, 'capacity'): SignedConv(1, 1, True)(huge, empty, empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
