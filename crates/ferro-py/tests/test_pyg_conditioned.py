"""Actual PyG references for learned edge filters and recurrent propagation."""
import sys
import unittest

import torch
from torch_geometric.nn import NNConv as TorchNN, EdgeConv as TorchEdge, GatedGraphConv as TorchGated
import ferro as fr
from ferro.graph import NNConv, ECConv, EdgeConv, GatedGraphConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class ConditionedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(887)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4)

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))

    def gradients(self, model, reference, actual, expected, inputs):
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward()
        (expected * probe).sum().backward()
        parameters = dict(reference.named_parameters())
        pairs = inputs + [(name, parameter.tensor(), parameters[name]) for name, parameter in model.named_parameters()]
        for name, actual, expected in pairs:
            with self.subTest(gradient=name):
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    self.assertIsNotNone(actual.grad)
                    self.close(actual.grad, expected.grad)

    def network(self, inputs, outputs):
        return (fr.nn.Sequential(fr.nn.Linear(inputs, 7), fr.nn.Tanh(), fr.nn.Linear(7, outputs)),
                torch.nn.Sequential(torch.nn.Linear(inputs, 7), torch.nn.Tanh(), torch.nn.Linear(7, outputs)))

    def test_nnconv_homogeneous_and_bipartite_all_gradients(self):
        for aggr, bipartite, root, bias in (('add', False, True, True), ('mean', True, True, False),
                                           ('max', True, False, True), ('min', False, True, False)):
            with self.subTest(aggr=aggr, bipartite=bipartite, root=root, bias=bias):
                network, torch_network = self.network(2, 12)
                widths = (3, 2) if bipartite else 3
                options = dict(aggr=aggr, root_weight=root, bias=bias)
                model, reference = NNConv(widths, 4, network, **options), TorchNN(widths, 4, torch_network, **options)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else source
                attributes = torch.randn(9, 2, requires_grad=True)
                ns, nt, na = native(source, True), None, native(attributes, True)
                nt = native(target, True) if bipartite else ns
                inputs = (ns, nt) if bipartite else ns
                torch_inputs = (source, target) if bipartite else source
                pairs = [('source', ns, source), ('edge_attr', na, attributes)] + ([('target', nt, target)] if bipartite else [])
                self.gradients(model, reference, model(inputs, native(self.edge), na), reference(torch_inputs, self.edge, attributes), pairs)

    def test_nnconv_missing_target_and_none_attribute_module(self):
        network, torch_network = self.network(2, 12)
        model, reference = NNConv((3, 2), 4, network), TorchNN((3, 2), 4, torch_network)
        self.copy(model, reference)
        source, attributes = torch.randn(6, 3, requires_grad=True), torch.randn(9, 2, requires_grad=True)
        ns, na = native(source, True), native(attributes, True)
        self.gradients(model, reference, model((ns, None), native(self.edge), na, size=(6, 7)),
                       reference((source, None), self.edge, attributes, size=(6, 7)), [('source', ns, source), ('edge_attr', na, attributes)])
        class Filters(fr.nn.Module):
            def build(self):
                self.weight = fr.nn.Parameter(fr.Tensor.rand([9, 12]))
            def forward(self, ignored):
                return self.weight.tensor()
        class TorchFilters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(9, 12))
            def forward(self, ignored):
                return self.weight
        model, reference = NNConv(3, 4, Filters()), TorchNN(3, 4, TorchFilters())
        self.copy(model, reference)
        x = torch.randn(6, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, reference, model(nx, native(self.edge)), reference(x, self.edge), [('x', nx, x)])
        self.assertIs(ECConv, NNConv)

    def test_edgeconv_pairwise_messages_all_aggregations_and_gradients(self):
        for aggr, bipartite in (('max', False), ('mean', True), ('add', True), ('min', False)):
            with self.subTest(aggr=aggr, bipartite=bipartite):
                network, torch_network = self.network(6, 4)
                model, reference = EdgeConv(network, aggr), TorchEdge(torch_network, aggr)
                self.copy(model, reference)
                base = torch.randn(3, 6, requires_grad=True)
                source = base.transpose(0, 1)
                target = torch.randn(5, 3, requires_grad=True) if bipartite else source
                nb = native(base, True)
                ns, nt = nb.transpose(0, 1), None
                nt = native(target, True) if bipartite else ns
                actual = model((ns, nt) if bipartite else ns, native(self.edge))
                expected = reference((source, target) if bipartite else source, self.edge)
                self.gradients(model, reference, actual, expected, [('source', nb, base)] + ([('target', nt, target)] if bipartite else []))

    def test_gatedgraph_recurrence_padding_bias_edge_and_all_parameter_gradients(self):
        for steps, width, aggr, bias, weighted in ((0, 3, 'add', True, True), (1, 4, 'mean', False, False),
                                                  (3, 3, 'add', True, True), (2, 0, 'max', True, True),
                                                  (3, 2, 'min', False, True)):
            with self.subTest(steps=steps, width=width, aggr=aggr, bias=bias):
                model, reference = GatedGraphConv(4, steps, aggr, bias), TorchGated(4, steps, aggr, bias)
                self.copy(model, reference)
                x = torch.randn(6, width, requires_grad=True)
                weight = torch.randn(9, requires_grad=True) if weighted else None
                nx, nw = native(x, True), None if weight is None else native(weight, True)
                self.gradients(model, reference, model(nx, native(self.edge), nw), reference(x, self.edge, weight),
                               [('x', nx, x)] + ([] if weight is None else [('edge_weight', nw, weight)]))

    def test_empty_graphs_isolates_and_duplicate_edges(self):
        edge = torch.empty(2, 0, dtype=torch.int64)
        for nodes in (0, 4):
            for kind in ('nn', 'edge', 'gated'):
                with self.subTest(nodes=nodes, kind=kind):
                    if kind == 'nn':
                        net, tnet = self.network(2, 12)
                        model, reference = NNConv(3, 4, net), TorchNN(3, 4, tnet)
                    elif kind == 'edge':
                        net, tnet = self.network(6, 4)
                        model, reference = EdgeConv(net), TorchEdge(tnet)
                    else:
                        model, reference = GatedGraphConv(4, 3), TorchGated(4, 3)
                    self.copy(model, reference)
                    x = torch.randn(nodes, 3, requires_grad=True)
                    nx = native(x, True)
                    args = [native(edge)]
                    targs = [edge]
                    pairs = [('x', nx, x)]
                    if kind == 'nn':
                        attr = torch.empty(0, 2, requires_grad=True)
                        na = native(attr, True)
                        args.append(na)
                        targs.append(attr)
                        pairs.append(('attributes', na, attr))
                    self.gradients(model, reference, model(nx, *args), reference(x, *targs), pairs)
        model, reference = GatedGraphConv(4, 2), TorchGated(4, 2)
        self.copy(model, reference)
        x = torch.randn(6, 3, requires_grad=True)
        nx = native(x, True)
        edges = fr.graph.EdgeIndex(6, self.edge[0].tolist(), self.edge[1].tolist())
        self.gradients(model, reference, model(nx, edges), reference(x, self.edge), [('x', nx, x)])

    def test_checkpoint_native_optimizer_aliases_and_configuration(self):
        network = fr.nn.Sequential(fr.nn.Linear(2, 12), fr.nn.Tanh())
        model, rng = NNConv(3, 4, network), fr.Generator(37)
        model.bias.trainable = False
        optimizer = fr.optim.AdamW(model.parameters(), lr=.01)
        x, edge = native(torch.randn(6, 3)), native(self.edge)
        def step(current, opt, generator):
            attributes = fr.Tensor.rand([9, 2], generator=generator)
            output = current(x, edge, attributes)
            loss = output.square().mean()
            result = output.tolist(), loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return result
        step(model, optimizer, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'edge': optimizer}, generator=rng, step=1)
        expected = step(model, optimizer, rng)
        other = NNConv(3, 4, fr.nn.Sequential(fr.nn.Linear(2, 12), fr.nn.Tanh()))
        other_rng, other_opt = fr.Generator(999), fr.optim.AdamW(other.parameters(), lr=.8)
        state.restore(other, optimizers={'edge': other_opt}, generator=other_rng)
        self.assertFalse(other.bias.trainable)
        self.assertEqual(expected, step(other, other_opt, other_rng))
        for left, right in zip(model.parameters(), other.parameters()):
            self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        changed = NNConv(3, 4, fr.nn.Sequential(fr.nn.Linear(2, 12), fr.nn.Tanh()), aggr='mean')
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError):
            state.restore(changed, optimizers={'edge': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])
        gated = GatedGraphConv(4, 2)
        gated.rnn.weight_hh = gated.rnn.weight_ih
        snapshot = fr.checkpoint.snapshot_training(gated, optimizers={}, generator=fr.Generator(1))
        clone = GatedGraphConv(4, 2)
        clone.rnn.weight_hh = clone.rnn.weight_ih
        snapshot.restore(clone, optimizers={}, generator=fr.Generator(2))
        self.assertIs(clone.rnn.weight_hh, clone.rnn.weight_ih)

    def test_strict_network_shapes_options_capacity_and_capture(self):
        for constructor in (lambda: NNConv(-1, 4, fr.nn.Linear(2, 12)), lambda: NNConv((3, 0), 4, fr.nn.Linear(2, 12)),
                            lambda: EdgeConv(lambda x: x), lambda: GatedGraphConv(0, 2), lambda: GatedGraphConv(4, -1),
                            lambda: GatedGraphConv(4, True), lambda: GatedGraphConv(sys.maxsize, 2)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        for constructor in (lambda: NNConv(3, 4, fr.nn.Linear(2, 12), flow='target_to_source'),
                            lambda: EdgeConv(fr.nn.Linear(6, 4), aggr='median'), lambda: GatedGraphConv(4, 2, node_dim=1)):
            with self.assertRaises(NotImplementedError):
                constructor()
        nn, edge, gated = NNConv(3, 4, fr.nn.Linear(2, 12)), EdgeConv(fr.nn.Linear(6, 4)), GatedGraphConv(4, 2)
        x, edges = native(torch.randn(6, 3)), native(self.edge)
        for call in (lambda: nn(x, edges, native(torch.randn(8, 2))),
                     lambda: NNConv(3, 4, fr.nn.Linear(2, 10))(x, edges, native(torch.randn(9, 2))),
                     lambda: edge((x, None), edges), lambda: edge((x, native(torch.randn(5, 2))), edges),
                     lambda: gated(native(torch.randn(6, 5)), edges), lambda: gated(x, edges, native(torch.ones(8))),
                     lambda: gated(x, native(torch.tensor([[6], [0]]))),
                     lambda: gated(fr.tensor([[1., 2., 3.]], dtype='float64'), edges)):
            with self.assertRaises((ValueError, TypeError)):
                call()
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: gated(x, edges))
        huge_empty_features = fr.Tensor.zeros([1, 0]).expand(sys.maxsize // 8 + 1, 0)
        with self.assertRaises(ValueError):
            GatedGraphConv(1, 1)(huge_empty_features, fr.Tensor.from_i64([], [2, 0]))
        with self.assertRaises(AttributeError):
            gated.num_layers = 7

    def test_gatedgraph_optimizer_checkpoint_continuation_and_reduction_config(self):
        model, rng = GatedGraphConv(4, 3), fr.Generator(67)
        opt = fr.optim.Adam(model.parameters(), lr=.02)
        edge = native(self.edge)
        def step(current, optimizer, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            weight = fr.Tensor.rand([9], generator=generator)
            output = current(x, edge, weight)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return values
        step(model, opt, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'gated': opt}, generator=rng, step=1)
        expected = [step(model, opt, rng) for _ in range(2)]
        other, other_rng = GatedGraphConv(4, 3), fr.Generator(99)
        other_opt = fr.optim.Adam(other.parameters(), lr=.7)
        state.restore(other, optimizers={'gated': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(2)])
        for left, right in zip(model.parameters(), other.parameters()):
            self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        self.assertEqual(fr.Tensor.rand([9], generator=rng).tolist(), fr.Tensor.rand([9], generator=other_rng).tolist())
        changed = GatedGraphConv(4, 3, aggr='mean')
        with self.assertRaises(ValueError):
            state.restore(changed, optimizers={'gated': fr.optim.Adam(changed.parameters())}, generator=fr.Generator(1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
