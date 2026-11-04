"""Installed-wheel additional graph convolutions against actual optional PyG."""
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.graph import APPNP, ChebConv, EdgeIndex, GINEConv, TransformerConv

try:
    from torch_geometric.nn import APPNP as TorchAPPNP, ChebConv as TorchChebConv
    from torch_geometric.nn import GINEConv as TorchGINEConv, TransformerConv as TorchTransformerConv
except ImportError:
    TorchAPPNP = None


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def indices(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


def topology():
    return torch.tensor([[0, 1, 1, 2, 2, 3, 4, 0, 2], [1, 0, 2, 1, 3, 2, 4, 1, 2]], dtype=torch.int64)


@unittest.skipUnless(TorchAPPNP is not None, 'PyG is an optional test reference')
class ExtraConvTests(unittest.TestCase):
    def close(self, actual, wanted):
        torch.testing.assert_close(values(actual), wanted.detach(), atol=2e-5, rtol=2e-4)

    def copy_parameters(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(parameters[name].detach()))

    def gradients(self, model, reference, actual, wanted, inputs, attention=None):
        self.close(actual, wanted)
        generator = torch.Generator().manual_seed(819)
        probe = torch.randn(wanted.shape, generator=generator)
        loss, expected = (actual * tensor(probe)).sum(), (wanted * probe).sum()
        if attention is not None:
            alpha, alpha_reference = attention
            self.close(alpha, alpha_reference)
            alpha_probe = torch.randn(alpha_reference.shape, generator=generator)
            loss = loss + (alpha * tensor(alpha_probe)).sum() * .2
            expected = expected + (alpha_reference * alpha_probe).sum() * .2
        loss.backward()
        expected.backward()
        parameters = dict(reference.named_parameters())
        pairs = list(inputs) + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]
        for name, native, target in pairs:
            with self.subTest(gradient=name):
                if target.grad is None:
                    self.assertIsNone(native.grad)
                else:
                    self.assertIsNotNone(native.grad)
                    self.close(native.grad, target.grad)

    def test_gine_network_eps_and_edge_projection(self):
        for edge_dim, train_eps, paired, missing_target in ((None, False, False, False),
                                                          (2, True, False, False),
                                                          (2, True, True, False),
                                                          (3, False, True, True)):
            with self.subTest(edge_dim=edge_dim, train_eps=train_eps, paired=paired, missing_target=missing_target):
                torch.manual_seed(71)
                network = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
                reference = TorchGINEConv(network, eps=.13, train_eps=train_eps, edge_dim=edge_dim)
                model = GINEConv(fr.nn.Sequential(fr.nn.Linear(3, 5), fr.nn.Tanh(), fr.nn.Linear(5, 2)), eps=.13, train_eps=train_eps, edge_dim=edge_dim)
                self.copy_parameters(model, reference)
                generator = torch.Generator().manual_seed(139)
                tx = torch.randn(6, 3, generator=generator).requires_grad_(True)
                ty = torch.randn(4, 3, generator=generator).requires_grad_(True) if paired and not missing_target else None
                edge_index = topology()
                if paired:
                    edge_index = edge_index.clone()
                    edge_index[1] %= 4
                attributes = torch.randn(edge_index.shape[1], edge_dim or 3, generator=generator).requires_grad_(True)
                sx, sy, sa = tensor(tx.detach(), True), None if ty is None else tensor(ty.detach(), True), tensor(attributes.detach(), True)
                size = (6, 4) if paired else None
                actual = model((sx, sy) if paired else sx, indices(edge_index), sa, size=size)
                wanted = reference((tx, ty) if paired else tx, edge_index, attributes, size=size)
                pairs = [('source', sx, tx), ('edge_attr', sa, attributes)]
                if ty is not None:
                    pairs.append(('target', sy, ty))
                self.gradients(model, reference, actual, wanted, pairs)

    def test_gine_empty_edges_self_loops_and_missing_attributes(self):
        torch.manual_seed(72)
        reference = TorchGINEConv(torch.nn.Linear(3, 2), edge_dim=2, train_eps=True)
        model = GINEConv(fr.nn.Linear(3, 2), edge_dim=2, train_eps=True)
        self.copy_parameters(model, reference)
        for count in (0, 4):
            with self.subTest(nodes=count):
                tx = torch.linspace(-.6, .8, count * 3).reshape(count, 3).requires_grad_(True)
                ta = torch.empty(0, 2, requires_grad=True)
                sx, sa = tensor(tx.detach(), True), tensor(ta.detach(), True)
                edge_index = torch.empty(2, 0, dtype=torch.int64)
                self.gradients(model, reference, model(sx, indices(edge_index), sa), reference(tx, edge_index, ta), [('input', sx, tx), ('edge_attr', sa, ta)])
                for parameter in model.parameters():
                    parameter.zero_grad()
                reference.zero_grad(set_to_none=True)
        with self.assertRaises((ValueError, TypeError)):
            model(tensor(torch.ones(3, 3)), indices([[0], [1]]))

    def test_appnp_propagation_teleport_weights_and_normalization(self):
        cases = ((3, .15, True, True, True), (0, .2, True, True, True),
                 (4, 0., True, False, False), (2, 1., True, True, True),
                 (3, .3, False, True, True), (2, .1, False, False, False))
        for steps, alpha, normalize, loops, weighted in cases:
            with self.subTest(steps=steps, alpha=alpha, normalize=normalize, loops=loops, weighted=weighted):
                model = APPNP(steps, alpha, normalize=normalize, add_self_loops=loops)
                reference = TorchAPPNP(steps, alpha, normalize=normalize, add_self_loops=loops)
                tx = torch.linspace(-.7, .9, 18).reshape(6, 3).requires_grad_(True)
                edge_index = topology()
                tw = torch.linspace(.3, 1.1, edge_index.shape[1]).requires_grad_(True) if weighted else None
                sx, sw = tensor(tx.detach(), True), None if tw is None else tensor(tw.detach(), True)
                pairs = [('input', sx, tx)] + ([] if sw is None else [('edge_weight', sw, tw)])
                self.gradients(model, reference, model(sx, indices(edge_index), sw), reference(tx, edge_index, tw), pairs)

    def test_appnp_empty_and_duplicate_loop_contract(self):
        for nodes in (0, 4):
            for loops in (False, True):
                with self.subTest(nodes=nodes, loops=loops):
                    model, reference = APPNP(2, .2, add_self_loops=loops), TorchAPPNP(2, .2, add_self_loops=loops)
                    tx = torch.ones(nodes, 3, requires_grad=True)
                    sx = tensor(tx.detach(), True)
                    edge_index = torch.empty(2, 0, dtype=torch.int64)
                    self.gradients(model, reference, model(sx, indices(edge_index)), reference(tx, edge_index), [('input', sx, tx)])
        model = APPNP(2, .2)
        with self.assertRaisesRegex(NotImplementedError, 'duplicate self-loops'):
            model(tensor(torch.ones(2, 3)), indices([[0, 0], [0, 0]]), tensor([.4, .7]))

    def test_chebyshev_orders_laplacian_modes_and_edge_gradients(self):
        for order, normalization, lambda_max in ((1, 'sym', None), (2, 'sym', None), (3, 'sym', 2.3),
                                                 (4, 'rw', None), (3, None, None), (2, None, 3.7)):
            with self.subTest(order=order, normalization=normalization, lambda_max=lambda_max):
                torch.manual_seed(181)
                model, reference = ChebConv(3, 2, order, normalization), TorchChebConv(3, 2, order, normalization)
                self.copy_parameters(model, reference)
                tx = torch.linspace(-.9, .7, 18).reshape(6, 3).requires_grad_(True)
                edge_index = topology()
                tw = torch.linspace(.3, 1.1, edge_index.shape[1]).requires_grad_(True)
                sx, sw = tensor(tx.detach(), True), tensor(tw.detach(), True)
                actual, wanted = model(sx, indices(edge_index), sw, lambda_max=lambda_max), reference(tx, edge_index, tw, lambda_max=lambda_max)
                self.gradients(model, reference, actual, wanted, [('input', sx, tx), ('edge_weight', sw, tw)])

    def test_chebyshev_per_graph_lambda_gradients_and_empty_graphs(self):
        torch.manual_seed(182)
        model, reference = ChebConv(3, 2, 3), TorchChebConv(3, 2, 3)
        self.copy_parameters(model, reference)
        tx = torch.linspace(-.8, 1., 18).reshape(6, 3).requires_grad_(True)
        edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
        tw = torch.linspace(.4, 1.1, 8).requires_grad_(True)
        tl = torch.tensor([2.2, 3.1], requires_grad=True)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        sx, sw, sl = tensor(tx.detach(), True), tensor(tw.detach(), True), tensor(tl.detach(), True)
        actual = model(sx, indices(edge_index), sw, batch=indices(batch), lambda_max=sl)
        wanted = reference(tx, edge_index, tw, batch=batch, lambda_max=tl)
        self.gradients(model, reference, actual, wanted, [('input', sx, tx), ('edge_weight', sw, tw), ('lambda_max', sl, tl)])
        self.close(model(tensor(torch.empty(0, 3)), indices(torch.empty(2, 0, dtype=torch.int64)), lambda_max=2.),
                   reference(torch.empty(0, 3), torch.empty(2, 0, dtype=torch.int64), lambda_max=2.))
        with self.assertRaisesRegex(ValueError, 'explicit lambda_max'):
            model(tensor(torch.empty(0, 3)), indices(torch.empty(2, 0, dtype=torch.int64)))

    def test_transformer_heads_beta_edge_features_bipartite_and_gradients(self):
        cases = ((3, 1, True, False, None, True, True), (3, 3, True, True, 2, True, True),
                 ((3, 4), 2, False, True, 2, False, True), (3, 2, False, False, None, True, False))
        for channels, heads, concat, beta, edge_dim, bias, root in cases:
            with self.subTest(channels=channels, heads=heads, concat=concat, beta=beta, edge_dim=edge_dim, root=root):
                torch.manual_seed(183)
                options = dict(heads=heads, concat=concat, beta=beta, edge_dim=edge_dim, bias=bias, root_weight=root)
                model, reference = TransformerConv(channels, 2, **options), TorchTransformerConv(channels, 2, **options)
                self.copy_parameters(model, reference)
                paired = isinstance(channels, tuple)
                generator = torch.Generator().manual_seed(816)
                tx = torch.randn(6, 3, generator=generator).requires_grad_(True)
                ty = torch.randn(4, 4, generator=generator).requires_grad_(True) if paired else None
                sx, sy = tensor(tx.detach(), True), None if ty is None else tensor(ty.detach(), True)
                edge_index = topology()
                if paired:
                    edge_index = edge_index.clone()
                    edge_index[1] %= 4
                ta = torch.randn(edge_index.shape[1], edge_dim, generator=generator).requires_grad_(True) if edge_dim is not None else None
                sa = None if ta is None else tensor(ta.detach(), True)
                actual, (actual_edges, alpha) = model((sx, sy) if paired else sx, indices(edge_index), sa, return_attention_weights=False)
                wanted, (wanted_edges, reference_alpha) = reference((tx, ty) if paired else tx, edge_index, ta, return_attention_weights=False)
                self.assertEqual(actual_edges.tolist(), wanted_edges.tolist())
                pairs = [('source', sx, tx)] + ([] if ty is None else [('target', sy, ty)]) + ([] if ta is None else [('edge_attr', sa, ta)])
                self.gradients(model, reference, actual, wanted, pairs, (alpha, reference_alpha))

    def test_transformer_empty_edges_nodes_and_attention_return_flags(self):
        for nodes in (0, 4):
            with self.subTest(nodes=nodes):
                torch.manual_seed(184)
                options = dict(heads=2, beta=True, edge_dim=2, concat=False)
                model, reference = TransformerConv(3, 2, **options), TorchTransformerConv(3, 2, **options)
                self.copy_parameters(model, reference)
                tx, ta = torch.ones(nodes, 3, requires_grad=True), torch.empty(0, 2, requires_grad=True)
                sx, sa = tensor(tx.detach(), True), tensor(ta.detach(), True)
                edge_index = torch.empty(2, 0, dtype=torch.int64)
                self.gradients(model, reference, model(sx, indices(edge_index), sa), reference(tx, edge_index, ta), [('input', sx, tx), ('edge_attr', sa, ta)])
                for flag in (None, False, True):
                    actual = model(sx, indices(edge_index), sa, return_attention_weights=flag)
                    wanted = reference(tx, edge_index, ta, return_attention_weights=flag)
                    self.assertEqual(isinstance(actual, tuple), isinstance(wanted, tuple))

    def test_native_topology_disjoint_batch_and_noncontiguous_features(self):
        native_edges = EdgeIndex(3, [0, 1, 1, 2], [1, 0, 0, 1])
        edge_index = torch.tensor([native_edges.src, native_edges.dst])
        raw = torch.linspace(-.6, .9, 9).reshape(3, 3).requires_grad_(True)
        source = tensor(raw.detach(), True)
        model, reference = ChebConv(3, 2, 3), TorchChebConv(3, 2, 3)
        self.copy_parameters(model, reference)
        self.gradients(model, reference, model(source.transpose(0, 1), native_edges), reference(raw.T, edge_index), [('input', source, raw)])
        union, _ = EdgeIndex.batch([native_edges, native_edges])
        x = tensor(raw.detach().T)
        single = model(x, native_edges)
        self.close(model(tensor(torch.cat([values(x), values(x)])), union), torch.cat([values(single), values(single)]))

    def test_each_operator_disjoint_batches_match_actual_pyg(self):
        torch.manual_seed(521)
        edges = EdgeIndex(4, [0, 1, 1, 2, 2], [1, 0, 0, 1, 2])
        union, _ = EdgeIndex.batch([edges, edges])
        edge_index = torch.tensor([union.src, union.dst])
        x = torch.linspace(-.8, .7, 24).reshape(8, 3)
        attrs = torch.linspace(.2, 1.3, 20).reshape(10, 2)
        factories = ((lambda: GINEConv(fr.nn.Linear(3, 2), edge_dim=2), lambda: TorchGINEConv(torch.nn.Linear(3, 2), edge_dim=2)),
                     (lambda: APPNP(3, .2), lambda: TorchAPPNP(3, .2)),
                     (lambda: ChebConv(3, 2, 3, bias=False), lambda: TorchChebConv(3, 2, 3, bias=False)),
                     (lambda: TransformerConv(3, 2, heads=2, beta=True, edge_dim=2), lambda: TorchTransformerConv(3, 2, heads=2, beta=True, edge_dim=2)))
        for factory, reference_factory in factories:
            model, reference = factory(), reference_factory()
            with self.subTest(layer=type(model).__name__):
                self.copy_parameters(model, reference)
                tx, ta = x.clone().requires_grad_(True), attrs.clone().requires_grad_(True)
                sx, sa = tensor(tx.detach(), True), tensor(ta.detach(), True)
                edged = isinstance(model, (GINEConv, TransformerConv))
                args, reference_args = (sx, union), (tx, edge_index)
                if edged:
                    args, reference_args = (*args, sa), (*reference_args, ta)
                actual = model(*args)
                pairs = [('input', sx, tx)] + ([('edge_attr', sa, ta)] if edged else [])
                self.gradients(model, reference, actual, reference(*reference_args), pairs)
                singles = []
                for i in range(2):
                    arguments = (tensor(x[i * 4:(i + 1) * 4]), edges)
                    if edged:
                        arguments = (*arguments, tensor(attrs[i * 5:(i + 1) * 5]))
                    singles.append(values(model(*arguments)))
                self.close(actual, torch.cat(singles))

    def test_checkpoint_configuration_eps_and_optimizer_continuation(self):
        factories = (lambda: GINEConv(fr.nn.Linear(3, 2), eps=.2, train_eps=True, edge_dim=2),
                     lambda: ChebConv(3, 2, 3), lambda: TransformerConv(3, 2, heads=2, beta=True, edge_dim=2))
        x, edge_index = tensor(torch.linspace(-.7, .8, 9).reshape(3, 3)), indices([[0, 1, 1, 2], [1, 0, 0, 1]])
        attributes = tensor([[.2, .5], [-.3, .4], [.7, -.1], [.4, -.2]])
        for factory in factories:
            with self.subTest(layer=factory().__class__.__name__):
                model = factory()
                optimizer, rng = fr.optim.AdamW(model.parameters(), lr=.02), fr.Generator(19)
                args = (x, edge_index) if isinstance(model, ChebConv) else (x, edge_index, attributes)
                model(*args).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
                checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint.save(directory)
                    restored = factory()
                    other_optimizer = fr.optim.AdamW(restored.parameters(), lr=.9)
                    self.assertEqual(fr.checkpoint.load_training(directory, restored, optimizers={'main': other_optimizer}, generator=fr.Generator(999)), 1)
                    self.assertEqual(model(*args).tolist(), restored(*args).tolist())
                    for layer, opt in ((model, optimizer), (restored, other_optimizer)):
                        result = layer(*args)
                        (result * result).mean().backward()
                        opt.step()
                        opt.zero_grad()
                    self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in restored.parameters()])
                field = 'normalization' if isinstance(model, ChebConv) else 'initial_eps' if isinstance(model, GINEConv) else 'heads'
                with self.assertRaises(AttributeError):
                    setattr(model, field, 'modified')
        model = APPNP(3, .2)
        state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1), step=5)
        self.assertEqual(state.restore(APPNP(3, .2), optimizers={}, generator=fr.Generator(9)), 5)
        with self.assertRaises(ValueError):
            state.restore(APPNP(4, .2), optimizers={}, generator=fr.Generator(9))

    def test_same_shape_configuration_rejection_and_buffered_eps(self):
        pairs = ((lambda: GINEConv(fr.nn.Linear(3, 2), eps=.2), lambda: GINEConv(fr.nn.Linear(3, 2), eps=.3)),
                 (lambda: APPNP(2, .2), lambda: APPNP(2, .3)),
                 (lambda: ChebConv(3, 2, 3, 'sym'), lambda: ChebConv(3, 2, 3, 'rw')),
                 (lambda: TransformerConv(3, 2), lambda: TransformerConv(3, 2, root_weight=False)))
        for factory, mismatch_factory in pairs:
            source, target = factory(), mismatch_factory()
            with self.subTest(layer=type(source).__name__):
                rng = fr.Generator(917)
                state = fr.checkpoint.snapshot_training(source, optimizers={}, generator=fr.Generator(12), step=7)
                before = [p.tensor().tolist() for p in target.parameters()]
                with self.assertRaises(ValueError):
                    state.restore(target, optimizers={}, generator=rng)
                self.assertEqual(before, [p.tensor().tolist() for p in target.parameters()])
                expected_rng = fr.Generator(917)
                self.assertEqual(fr.Tensor.rand([4], generator=rng).tolist(), fr.Tensor.rand([4], generator=expected_rng).tolist())
                valid = factory()
                self.assertEqual(state.restore(valid, optimizers={}, generator=rng), 7)
                if isinstance(source, GINEConv):
                    self.assertEqual(valid.eps.tolist(), source.eps.tolist())

    def test_unsupported_and_invalid_requests_fail_explicitly(self):
        for factory in (lambda: APPNP(2, .2, cached=True), lambda: ChebConv(3, 2, 2, aggr='max')):
            with self.assertRaises(NotImplementedError):
                factory()
        for factory in (lambda: APPNP(-1, .2), lambda: APPNP(2, float('nan')), lambda: APPNP(2, .2, dropout=1.1),
                        lambda: TransformerConv(3, 2, dropout=-.1), lambda: ChebConv(-1, 2, 3),
                        lambda: ChebConv(3, 2, 0), lambda: ChebConv(3, 2, 2, 'bad'),
                        lambda: TransformerConv(3, 2, heads=0), lambda: TransformerConv((-1, 3), 2),
                        lambda: ChebConv(3, 2, sys.maxsize), lambda: TransformerConv(3, 2, heads=sys.maxsize),
                        lambda: GINEConv(fr.nn.Linear(3, 2), edge_dim=sys.maxsize)):
            with self.assertRaises((ValueError, TypeError)):
                factory()
        x, edge_index = tensor(torch.ones(3, 3)), indices([[0, 1], [1, 2]])
        for model in (APPNP(2, .2), ChebConv(3, 2, 3), TransformerConv(3, 2)):
            with self.subTest(layer=type(model).__name__), self.assertRaises((ValueError, TypeError)):
                model(x, indices([[0, 3], [1, 2]]))
        for maximum in (0., -1., float('inf'), tensor([]), tensor([2., 3.])):
            with self.subTest(maximum=maximum), self.assertRaises((ValueError, TypeError)):
                ChebConv(3, 2, 3)(x, edge_index, lambda_max=maximum)
        with self.assertRaises(NotImplementedError):
            TransformerConv(3, 2)(x, edge_index, tensor(torch.ones(2, 1)))
        with self.assertRaises((ValueError, TypeError)):
            TransformerConv(3, 2, edge_dim=2)(x, edge_index)
        with self.assertRaises(ValueError):
            TransformerConv((3, 3), 2)((x, None), edge_index)

    def test_dtype_and_eager_first_order_boundaries(self):
        from ferro.autograd import grad
        edge_index = indices([[0, 1], [1, 2]])
        for model in (GINEConv(fr.nn.Linear(3, 2), edge_dim=2), APPNP(2, .2),
                      ChebConv(3, 2, 3), TransformerConv(3, 2, edge_dim=2)):
            edged = isinstance(model, (GINEConv, TransformerConv))
            attributes = tensor(torch.ones(2, 2)) if edged else tensor([.5, .8])
            with self.subTest(layer=type(model).__name__):
                for dtype in ('int64', 'float64'):
                    with self.assertRaises(ValueError):
                        model(fr.tensor([[1, 2, 3]] * 3, dtype=dtype), edge_index, attributes)
                    with self.assertRaises(ValueError):
                        bad = fr.tensor([[1, 2]] * 2 if edged else [1, 2], dtype=dtype)
                        model(tensor(torch.ones(3, 3)), edge_index, bad)
                features = tensor(torch.ones(3, 3), True)
                with self.assertRaisesRegex(ValueError, 'capture'):
                    fr.capture(lambda: model(features, edge_index, attributes))
                with self.assertRaisesRegex(ValueError, 'create_graph|unsupported'):
                    grad(model(features, edge_index, attributes).sum(), [features], create_graph=True)
        features = tensor(torch.ones(3, 3), True)
        identity = APPNP(0, .2, normalize=False)
        self.assertIs(identity(features, edge_index), features)


if __name__ == '__main__':
    unittest.main(verbosity=2)
