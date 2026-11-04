"""Actual PyG references for residual gates, cosine attention and differences."""
import sys
import unittest

import torch
from torch_geometric.nn import ResGatedGraphConv as TorchRes, AGNNConv as TorchAGNN, LEConv as TorchLE
import ferro as fr
from ferro.graph import ResGatedGraphConv, AGNNConv, LEConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(907)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4)

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))
            buffers = dict(reference.named_buffers())
            self.assertEqual(set(dict(model.named_buffers())), set(buffers))
            for name, buffer in model.named_buffers():
                buffer.copy_(native(buffers[name]))

    def gradients(self, model, reference, actual, expected, inputs):
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward()
        (expected * probe).sum().backward()
        parameters = dict(reference.named_parameters())
        for name, actual, expected in inputs + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    self.assertIsNotNone(actual.grad)
                    self.close(actual.grad, expected.grad)

    def test_residual_gates_node_and_edge_projection_all_gradients(self):
        for bipartite, edge_dim, root, bias, activation in ((False, None, True, True, 'sigmoid'),
                                                          (True, 2, True, False, 'tanh'),
                                                          (False, 2, False, True, 'sigmoid'),
                                                          (True, None, False, False, 'relu')):
            with self.subTest(bipartite=bipartite, edge_dim=edge_dim, root=root, bias=bias, activation=activation):
                widths = (3, 2) if bipartite else 3
                options = dict(edge_dim=edge_dim, root_weight=root, bias=bias)
                module = {'sigmoid': torch.nn.Sigmoid, 'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU}[activation]()
                model, reference = ResGatedGraphConv(widths, 4, act=activation, **options), TorchRes(widths, 4, act=module, **options)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else source
                ns, nt = native(source, True), None
                nt = native(target, True) if bipartite else ns
                attr = torch.randn(9, edge_dim, requires_grad=True) if edge_dim else None
                na = None if attr is None else native(attr, True)
                pair = (ns, nt) if bipartite else ns
                torch_pair = (source, target) if bipartite else source
                inputs = [('source', ns, source)] + ([('target', nt, target)] if bipartite else []) + ([] if attr is None else [('edge_attr', na, attr)])
                self.gradients(model, reference, model(pair, native(self.edge), na), reference(torch_pair, self.edge, attr), inputs)

    def test_residual_gate_trainable_activation_parameters(self):
        model = ResGatedGraphConv(3, 4, act=fr.nn.Linear(4, 4))
        reference = TorchRes(3, 4, act=torch.nn.Linear(4, 4))
        self.copy(model, reference)
        x = torch.randn(6, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, reference, model(nx, native(self.edge)), reference(x, self.edge), [('x', nx, x)])

    def test_cosine_attention_beta_buffers_loops_duplicates_and_zero_nodes(self):
        edge = torch.cat([self.edge, torch.tensor([[1, 1], [1, 1]])], 1)
        for trainable, loops, zero in ((True, True, False), (False, False, False), (True, False, True), (False, True, True)):
            with self.subTest(trainable=trainable, loops=loops, zero=zero):
                model, reference = AGNNConv(trainable, loops), TorchAGNN(trainable, loops)
                with torch.no_grad():
                    reference.beta.fill_(1.7)
                self.copy(model, reference)
                value = torch.randn(6, 3)
                if zero:
                    value[0] = 0.
                x = value.requires_grad_()
                nx = native(x, True)
                self.gradients(model, reference, model(nx, native(edge)), reference(x, edge), [('x', nx, x)])
        model = AGNNConv(False)
        with fr.no_grad():
            model.beta.copy_(fr.Tensor([2.], [1]))
        model.reset_parameters()
        self.assertEqual(model.beta.tolist(), [2.])

    def test_local_difference_weighted_bipartite_and_unweighted_gradients(self):
        for bipartite, bias, weighted in ((False, True, True), (True, False, True), (True, True, False)):
            with self.subTest(bipartite=bipartite, bias=bias, weighted=weighted):
                widths = (3, 2) if bipartite else 3
                model, reference = LEConv(widths, 4, bias=bias), TorchLE(widths, 4, bias=bias)
                self.copy(model, reference)
                base = torch.randn(3, 6, requires_grad=True)
                source = base.transpose(0, 1)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else source
                nb = native(base, True)
                ns = nb.transpose(0, 1)
                nt = native(target, True) if bipartite else ns
                weight = torch.randn(9, requires_grad=True) if weighted else None
                nw = None if weight is None else native(weight, True)
                inputs = [('source', nb, base)] + ([('target', nt, target)] if bipartite else []) + ([] if weight is None else [('edge_weight', nw, weight)])
                self.gradients(model, reference, model((ns, nt) if bipartite else ns, native(self.edge), nw),
                               reference((source, target) if bipartite else source, self.edge, weight), inputs)

    def test_empty_graphs_zero_feature_attention_and_native_topology(self):
        edges = torch.empty(2, 0, dtype=torch.int64)
        for nodes in (0, 4):
            for model, reference in ((ResGatedGraphConv(3, 4), TorchRes(3, 4)), (LEConv(3, 4), TorchLE(3, 4)),
                                     (AGNNConv(), TorchAGNN()), (AGNNConv(add_self_loops=False), TorchAGNN(add_self_loops=False))):
                with self.subTest(nodes=nodes, kind=type(model).__name__):
                    self.copy(model, reference)
                    x = torch.randn(nodes, 3, requires_grad=True)
                    nx = native(x, True)
                    self.gradients(model, reference, model(nx, native(edges)), reference(x, edges), [('x', nx, x)])
        model, reference = AGNNConv(), TorchAGNN()
        self.copy(model, reference)
        x = torch.empty(6, 0, requires_grad=True)
        nx = native(x, True)
        edge = fr.graph.EdgeIndex(6, self.edge[0].tolist(), self.edge[1].tolist())
        self.gradients(model, reference, model(nx, edge), reference(x, self.edge), [('x', nx, x)])

    def test_native_optimizer_checkpoint_continuation_aliases_and_config(self):
        def factory():
            model = ResGatedGraphConv(3, 4, edge_dim=2)
            model.lin_value.weight = model.lin_query.weight
            model.bias.trainable = False
            return model
        model, rng = factory(), fr.Generator(71)
        opt = fr.optim.AdamW(model.parameters(), lr=.01)
        edge = native(self.edge)
        def step(current, optimizer, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            attributes = fr.Tensor.rand([9, 2], generator=generator)
            output = current(x, edge, attributes)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return values
        step(model, opt, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'gates': opt}, generator=rng, step=1)
        expected = [step(model, opt, rng) for _ in range(2)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        state.restore(other, optimizers={'gates': other_opt}, generator=other_rng)
        self.assertIs(other.lin_value.weight, other.lin_query.weight)
        self.assertFalse(other.bias.trainable)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(2)])
        for a, b in zip(model.parameters(), other.parameters()):
            self.assertEqual(a.tensor().tolist(), b.tensor().tolist())
        for trainable in (False, True):
            model = AGNNConv(trainable, True)
            state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1))
            changed = AGNNConv(trainable, False)
            with self.assertRaises(ValueError):
                state.restore(changed, optimizers={}, generator=fr.Generator(2))

    def test_strict_shapes_options_capacities_and_capture(self):
        for constructor in (lambda: ResGatedGraphConv(-1, 4), lambda: ResGatedGraphConv(3, 4, edge_dim=0),
                            lambda: ResGatedGraphConv(3, 4, act=None), lambda: AGNNConv(requires_grad=1),
                            lambda: LEConv((3, 0), 4), lambda: LEConv(sys.maxsize, 4)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        for constructor in (lambda: AGNNConv(flow='target_to_source'), lambda: LEConv(3, 4, aggr='mean'),
                            lambda: ResGatedGraphConv(3, 4, act=lambda x: x.tanh())):
            with self.assertRaises(NotImplementedError):
                constructor()
        x, edges, attrs = native(torch.randn(6, 3)), native(self.edge), native(torch.randn(9, 2))
        for call in (lambda: ResGatedGraphConv(3, 4)(x, edges, attrs), lambda: ResGatedGraphConv(3, 4, edge_dim=2)(x, edges),
                     lambda: ResGatedGraphConv(3, 4, edge_dim=2)(x, edges, native(torch.randn(8, 2))),
                     lambda: ResGatedGraphConv(3, 4)((x, None), edges), lambda: LEConv(3, 4)((x, None), edges),
                     lambda: LEConv(3, 4)(x, edges, native(torch.randn(8))),
                     lambda: AGNNConv()(x, native(torch.tensor([[6], [0]]))),
                     lambda: AGNNConv()(fr.tensor([[1., 2., 3.]], dtype='float64'), edges)):
            with self.assertRaises((ValueError, TypeError)):
                call()
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: AGNNConv()(x, edges))
        huge_empty_features = fr.Tensor.zeros([1, 0]).expand(sys.maxsize // 8 + 1, 0)
        for loops in (False, True):
            with self.assertRaises(ValueError):
                AGNNConv(add_self_loops=loops)(huge_empty_features, fr.Tensor.from_i64([], [2, 0]))
        model = AGNNConv()
        with self.assertRaises(AttributeError):
            model.add_self_loops = False


if __name__ == '__main__':
    unittest.main(verbosity=2)
