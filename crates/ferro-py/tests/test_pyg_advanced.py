"""Advanced graph layers against actual optional PyG, with full gradients."""
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.graph import Data, EdgeIndex, GCN2Conv, HGTConv, PNAConv

try:
    from torch_geometric.nn import GCN2Conv as TorchGCN2Conv
    from torch_geometric.nn import HGTConv as TorchHGTConv, PNAConv as TorchPNAConv
except ImportError:
    TorchPNAConv = None


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def indices(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


def topology():
    return torch.tensor([[0, 1, 1, 2, 2, 3, 4, 0, 2], [1, 0, 2, 1, 3, 2, 4, 1, 2]])


@unittest.skipUnless(TorchPNAConv is not None, 'PyG is an optional test reference')
class AdvancedGraphTests(unittest.TestCase):
    def close(self, actual, wanted):
        torch.testing.assert_close(values(actual), wanted.detach(), atol=3e-5, rtol=3e-4)

    def copy_parameters(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(parameters[name].detach()))
            buffers = dict(reference.named_buffers())
            for name, buffer in model.named_buffers():
                buffer.copy_(tensor(buffers[name]))

    def gradients(self, model, reference, actual, wanted, inputs):
        if not isinstance(actual, dict):
            actual, wanted = {'output': actual}, {'output': wanted}
        self.assertEqual(set(actual), set(wanted))
        generator = torch.Generator().manual_seed(819)
        loss, expected = None, None
        for name, value in actual.items():
            self.close(value, wanted[name])
            probe = torch.randn(wanted[name].shape, generator=generator)
            term, target = (value * tensor(probe)).sum(), (wanted[name] * probe).sum()
            loss = term if loss is None else loss + term
            expected = target if expected is None else expected + target
        loss.backward()
        expected.backward()
        parameters = dict(reference.named_parameters())
        for name, native, target in list(inputs) + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if target.grad is None:
                    self.assertIsNone(native.grad)
                else:
                    self.assertIsNotNone(native.grad)
                    self.close(native.grad, target.grad)

    def test_pna_all_aggregators_scalers_towers_and_edge_gradients(self):
        aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']
        scalers = ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear']
        deg = torch.tensor([1, 3, 5, 2, 1])
        for towers, divide, train_norm, edge_dim, act in ((1, False, False, None, 'relu'),
                                                        (2, False, True, 2, 'tanh'),
                                                        (2, True, True, 3, 'gelu'),
                                                        (2, True, False, None, None)):
            with self.subTest(towers=towers, divide=divide, train_norm=train_norm, edge_dim=edge_dim, act=act):
                torch.manual_seed(191)
                config = dict(edge_dim=edge_dim, towers=towers, divide_input=divide, pre_layers=2, post_layers=2, train_norm=train_norm, act=act)
                model = PNAConv(4, 6, aggregators, scalers, indices(deg), **config)
                # PyG 2.8 inserts a non-callable None in multilayer networks.
                reference = TorchPNAConv(4, 6, aggregators, scalers, deg, **{**config, 'act': torch.nn.Identity() if act is None else act})
                self.copy_parameters(model, reference)
                x = torch.randn(6, 4, requires_grad=True)
                edge_index = topology()
                edge = torch.randn(edge_index.shape[1], edge_dim, requires_grad=True) if edge_dim else None
                sx, se = tensor(x.detach(), True), None if edge is None else tensor(edge.detach(), True)
                self.gradients(model, reference, model(sx, indices(edge_index), se), reference(x, edge_index, edge),
                               [('x', sx, x)] + ([] if edge is None else [('edge_attr', se, edge)]))

    def test_pna_empty_edges_nodes_and_degree_zero(self):
        deg = torch.tensor([2, 3, 4])
        for count in (0, 3):
            with self.subTest(nodes=count):
                torch.manual_seed(192)
                config = dict(edge_dim=2, train_norm=True)
                model = PNAConv(3, 2, ['mean', 'std', 'min', 'max'], ['identity', 'linear', 'attenuation'], indices(deg), **config)
                reference = TorchPNAConv(3, 2, ['mean', 'std', 'min', 'max'], ['identity', 'linear', 'attenuation'], deg, **config)
                self.copy_parameters(model, reference)
                x, edge = torch.randn(count, 3, requires_grad=True), torch.empty(0, 2, requires_grad=True)
                sx, se = tensor(x.detach(), True), tensor(edge.detach(), True)
                topology = torch.empty(2, 0, dtype=torch.int64)
                self.gradients(model, reference, model(sx, indices(topology), se), reference(x, topology, edge), [('x', sx, x), ('edge', se, edge)])

    def test_pna_degree_histogram_views_and_disjoint_graphs(self):
        from torch_geometric.data import Data as TorchData
        edge = topology()
        actual = PNAConv.get_degree_histogram([Data(edge_index=indices(edge), num_nodes=6), Data(edge_index=indices(edge), num_nodes=7)])
        expected = TorchPNAConv.get_degree_histogram([TorchData(edge_index=edge, num_nodes=6), TorchData(edge_index=edge, num_nodes=7)])
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(PNAConv.get_degree_histogram([]).tolist(), [0])
        torch.manual_seed(217)
        model = PNAConv(4, 4, ['mean', 'max', 'std'], ['identity', 'amplification'], actual)
        reference = TorchPNAConv(4, 4, ['mean', 'max', 'std'], ['identity', 'amplification'], expected)
        self.copy_parameters(model, reference)
        raw = torch.randn(4, 6, requires_grad=True)
        native = tensor(raw.detach(), True)
        self.gradients(model, reference, model(native.transpose(0, 1), indices(edge)), reference(raw.T, edge), [('strided input', native, raw)])
        single_edges = EdgeIndex(6, edge[0].tolist(), edge[1].tolist())
        union, _ = EdgeIndex.batch([single_edges, single_edges])
        x = tensor(raw.detach().T)
        one = model(x, single_edges)
        self.close(model(fr.cat([x, x]), union), torch.cat([values(one), values(one)]))

    def hgt_case(self, empty=False, partial=False, alias=False):
        metadata = (['author', 'paper', 'topic'], [('author', 'writes', 'paper'), ('paper', 'cites', 'paper'),
                                                  ('topic', 'describes', 'paper'), ('paper', 'has', 'topic')])
        widths = {'author': 3, 'paper': 4, 'topic': 4}
        torch.manual_seed(229)
        model, reference = HGTConv(widths, 4, metadata, heads=2), TorchHGTConv(widths, 4, metadata, heads=2)
        if alias:
            model.out_lin.lins['topic'] = model.out_lin.lins['paper']
            reference.out_lin.lins['topic'] = reference.out_lin.lins['paper']
            model.v_rel.weight = model.k_rel.weight
            reference.v_rel.weight = reference.k_rel.weight
        self.copy_parameters(model, reference)
        x = {name: torch.randn(count, widths[name], requires_grad=True) for name, count in (('topic', 3), ('author', 4), ('paper', 5))}
        edges = {metadata[1][1]: torch.tensor([[0, 1, 1, 3], [1, 0, 0, 2]]),
                 metadata[1][0]: torch.tensor([[0, 1, 2, 2], [1, 0, 1, 3]]),
                 metadata[1][2]: torch.tensor([[0, 1, 2], [0, 2, 4]]),
                 metadata[1][3]: torch.tensor([[0, 2, 3], [1, 0, 1]])}
        if partial:
            del edges[metadata[1][2]]
            del edges[metadata[1][3]]
            del x['topic']
        if empty:
            edges = {key: torch.empty(2, 0, dtype=torch.int64) for key in edges}
        native = {key: tensor(value.detach(), True) for key, value in x.items()}
        actual = model(native, {key: indices(value) for key, value in edges.items()})
        # PyG HeteroLinear(is_sorted=True) needs metadata-ordered relation types.
        wanted = reference(x, {key: edges[key] for key in metadata[1] if key in edges})
        reordered = model(native, {key: indices(value) for key, value in reversed(list(edges.items()))})
        for name in actual:
            self.close(reordered[name], values(actual[name]))
        self.gradients(model, reference, actual, wanted, [(key, native[key], x[key]) for key in x])
        return model, reference, native, edges

    def test_hgt_relations_joint_attention_and_all_gradients(self):
        self.hgt_case()
        self.hgt_case(partial=True)

    def test_hgt_empty_edges_and_shared_parameter_aliases(self):
        self.hgt_case(empty=True)
        self.hgt_case(alias=True)

    def test_hgt_empty_node_stores_and_unused_type(self):
        metadata = (['a', 'b', 'unused'], [('a', 'r', 'b')])
        for count in (0, 2):
            with self.subTest(nodes=count):
                torch.manual_seed(231)
                model, reference = HGTConv(4, 4, metadata, heads=2), TorchHGTConv(4, 4, metadata, heads=2)
                self.copy_parameters(model, reference)
                x = {'a': torch.empty(0, 4, requires_grad=True), 'b': torch.randn(count, 4, requires_grad=True),
                     'unused': torch.randn(2, 4, requires_grad=True)}
                native = {key: tensor(value.detach(), True) for key, value in x.items()}
                edge = torch.empty(2, 0, dtype=torch.int64)
                self.gradients(model, reference, model(native, {metadata[1][0]: indices(edge)}), reference(x, {metadata[1][0]: edge}),
                               [(key, native[key], x[key]) for key in x])

    def test_hgt_disjoint_graphs_and_alias_checkpoint(self):
        metadata = (['a', 'b'], [('a', 'r', 'b'), ('b', 's', 'a')])
        def factory(tied=True):
            model = HGTConv(4, 4, metadata, heads=2)
            if tied:
                model.v_rel.weight = model.k_rel.weight
                model.out_lin.lins['b'] = model.out_lin.lins['a']
            return model
        model = factory()
        x = {'a': tensor(torch.randn(3, 4)), 'b': tensor(torch.randn(2, 4))}
        edges = {metadata[1][0]: torch.tensor([[0, 1, 2], [1, 0, 1]]), metadata[1][1]: torch.tensor([[0, 1], [1, 2]])}
        single = model(x, {key: indices(value) for key, value in edges.items()})
        batched_edges = {key: indices(torch.cat([edge, edge + torch.tensor([[x[key[0]].shape[0]], [x[key[2]].shape[0]]])], dim=1)) for key, edge in edges.items()}
        batched = model({key: fr.cat([value, value]) for key, value in x.items()}, batched_edges)
        for name, output in single.items():
            self.close(batched[name], torch.cat([values(output), values(output)]))
        state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1), step=3)
        restored = factory()
        self.assertEqual(state.restore(restored, optimizers={}, generator=fr.Generator(2)), 3)
        self.assertIs(restored.k_rel.weight, restored.v_rel.weight)
        self.assertIs(restored.out_lin.lins['a'], restored.out_lin.lins['b'])
        with self.assertRaises(ValueError):
            state.restore(factory(False), optimizers={}, generator=fr.Generator(3))

    def test_gcn2_shared_separate_normalization_and_edge_gradients(self):
        for shared, normalize, loops, theta in ((True, True, True, None), (False, True, True, .7),
                                               (False, False, True, .5), (True, True, False, .8)):
            with self.subTest(shared=shared, normalize=normalize, loops=loops, theta=theta):
                torch.manual_seed(241)
                config = dict(shared_weights=shared, normalize=normalize, add_self_loops=loops, theta=theta, layer=None if theta is None else 3)
                model, reference = GCN2Conv(3, .2, **config), TorchGCN2Conv(3, .2, **config)
                self.copy_parameters(model, reference)
                x, initial = torch.randn(6, 3, requires_grad=True), torch.randn(8, 3, requires_grad=True)
                edge_index = topology()
                weight = torch.linspace(.3, 1.2, edge_index.shape[1]).requires_grad_(True)
                sx, si, sw = tensor(x.detach(), True), tensor(initial.detach(), True), tensor(weight.detach(), True)
                self.gradients(model, reference, model(sx, si, indices(edge_index), sw), reference(x, initial, edge_index, weight),
                               [('x', sx, x), ('x_0', si, initial), ('edge_weight', sw, weight)])

    def test_gcn2_empty_graphs_and_disjoint_batches(self):
        for nodes in (0, 4):
            with self.subTest(nodes=nodes):
                model, reference = GCN2Conv(3, .2), TorchGCN2Conv(3, .2)
                self.copy_parameters(model, reference)
                x, initial = torch.randn(nodes, 3, requires_grad=True), torch.randn(nodes, 3, requires_grad=True)
                sx, si = tensor(x.detach(), True), tensor(initial.detach(), True)
                edge = torch.empty(2, 0, dtype=torch.int64)
                self.gradients(model, reference, model(sx, si, indices(edge)), reference(x, initial, edge), [('x', sx, x), ('x_0', si, initial)])
        edge = EdgeIndex(6, topology()[0].tolist(), topology()[1].tolist())
        union, _ = EdgeIndex.batch([edge, edge])
        x, initial = tensor(torch.randn(6, 3)), tensor(torch.randn(6, 3))
        model = GCN2Conv(3, .2, theta=.6, layer=2)
        single = model(x, initial, edge)
        doubled = model(fr.cat([x, x]), fr.cat([initial, initial]), union)
        self.close(doubled, torch.cat([values(single), values(single)]))

    def test_checkpoint_roundtrip_optimizer_and_configuration(self):
        deg = indices([1, 3, 2])
        metadata = (['a', 'b'], [('a', 'r', 'b')])
        cases = [(lambda: PNAConv(3, 3, ['mean', 'std'], ['identity', 'linear'], deg, train_norm=True),
                  lambda m: m(tensor(torch.arange(12).reshape(4, 3) / 10), indices([[0, 1, 2], [1, 0, 1]]))),
                 (lambda: HGTConv({'a': 3, 'b': 3}, 3, metadata),
                  lambda m: m({'a': tensor(torch.arange(9).reshape(3, 3) / 10), 'b': tensor(torch.ones(2, 3))},
                              {('a', 'r', 'b'): indices([[0, 1, 2], [1, 0, 1]])})['b']),
                 (lambda: GCN2Conv(3, .2, shared_weights=False),
                  lambda m: m(tensor(torch.ones(3, 3)), tensor(torch.arange(9).reshape(3, 3) / 10), indices([[0, 1], [1, 2]])))]
        for factory, execute in cases:
            model = factory()
            with self.subTest(layer=type(model).__name__):
                opt, rng = fr.optim.Adam(model.parameters(), lr=.02), fr.Generator(7)
                execute(model).sum().backward()
                opt.step()
                opt.zero_grad()
                state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=1)
                with tempfile.TemporaryDirectory() as directory:
                    state.save(directory)
                    restored = factory()
                    other = fr.optim.Adam(restored.parameters(), lr=.9)
                    self.assertEqual(fr.checkpoint.load_training(directory, restored, optimizers={'main': other}, generator=fr.Generator(999)), 1)
                    self.assertEqual(execute(model).tolist(), execute(restored).tolist())
                    for layer, optimizer in ((model, opt), (restored, other)):
                        out = execute(layer)
                        (out * out).mean().backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in restored.parameters()])
        mismatches = [(PNAConv(3, 3, ['mean', 'std'], ['identity'], deg), PNAConv(3, 3, ['std', 'mean'], ['identity'], deg)),
                      (PNAConv(3, 3, ['mean'], ['identity'], deg), PNAConv(3, 3, ['mean'], ['identity'], indices([2, 2, 2]))),
                      (HGTConv(3, 3, metadata), HGTConv(3, 3, (['a', 'b'], [('a', 'other', 'b')]))),
                      (GCN2Conv(3, .2), GCN2Conv(3, .3))]
        for model, target in mismatches:
            state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(7))
            before = [p.tensor().tolist() for p in target.parameters()]
            with self.assertRaises(ValueError):
                state.restore(target, optimizers={}, generator=fr.Generator(9))
            self.assertEqual(before, [p.tensor().tolist() for p in target.parameters()])

    def test_invalid_metadata_options_shapes_and_capacity(self):
        deg = indices([1, 3, 2])
        for factory in (lambda: PNAConv(3, 3, ['bad'], ['identity'], deg),
                        lambda: PNAConv(3, 3, ['mean'], ['bad'], deg),
                        lambda: PNAConv(3, 3, ['mean'], ['linear'], indices([2])),
                        lambda: PNAConv(3, 3, ['mean'], ['identity'], indices([0, 0])),
                        lambda: PNAConv(3, 3, ['mean'], ['identity'], deg, towers=2),
                        lambda: PNAConv(3, 3, ['mean'], ['identity'], deg, pre_layers=sys.maxsize),
                        lambda: HGTConv(3, 3, (['a'], [('a', 'r', 'missing')])),
                        lambda: HGTConv({'a': 3}, 4, (['a', 'b'], [('a', 'r', 'b')])),
                        lambda: HGTConv(3, 3, (['a'], [('a', 'r', 'a')]), heads=2),
                        lambda: HGTConv(3, sys.maxsize, (['a'], [('a', 'r', 'a')])),
                        lambda: GCN2Conv(3, .2, theta=.5), lambda: GCN2Conv(3, float('nan')),
                        lambda: GCN2Conv(sys.maxsize, .2)):
            with self.assertRaises((ValueError, TypeError)):
                factory()
        for factory in (lambda: PNAConv(3, 3, ['mean'], ['identity'], deg, act='unknown'), lambda: GCN2Conv(3, .2, cached=True)):
            with self.assertRaises(NotImplementedError):
                factory()
        x = tensor(torch.ones(3, 3))
        with self.assertRaises(ValueError):
            PNAConv(3, 3, ['mean'], ['identity'], deg)(x, indices([[0, 3], [1, 2]]))
        with self.assertRaises(ValueError):
            GCN2Conv(3, .2)(x, tensor(torch.ones(2, 3)), indices([[0], [1]]))
        hgt = HGTConv(3, 3, (['a', 'b'], [('a', 'r', 'b')]))
        for features, edges in (({'a': x}, {('a', 'r', 'b'): indices([[0], [1]])}),
                                ({'a': x, 'b': x}, {('a', 'wrong', 'b'): indices([[0], [1]])}),
                                ({'a': x, 'b': x}, {('a', 'r', 'b'): indices([[0], [3]])})):
            with self.assertRaises(ValueError):
                hgt(features, edges)

    def test_cpu_f32_eager_boundaries_and_immutable_configuration(self):
        from ferro.autograd import grad
        x, edge = tensor(torch.ones(3, 3), True), indices([[0, 1], [1, 2]])
        pna = PNAConv(3, 3, ['mean'], ['identity'], indices([1, 2]))
        gcn = GCN2Conv(3, .2)
        hgt = HGTConv(3, 3, (['a'], [('a', 'r', 'a')]))
        for model, execute in ((pna, lambda m, v: m(v, edge)), (gcn, lambda m, v: m(v, v, edge)),
                               (hgt, lambda m, v: m({'a': v}, {('a', 'r', 'a'): edge})['a'])):
            with self.subTest(layer=type(model).__name__):
                for dtype in ('float64', 'int64'):
                    with self.assertRaises(ValueError):
                        execute(model, fr.tensor([[1, 2, 3]] * 3, dtype=dtype))
                with self.assertRaisesRegex(ValueError, 'capture'):
                    fr.capture(lambda: execute(model, x))
                with self.assertRaisesRegex(ValueError, 'unsupported|create_graph'):
                    grad(execute(model, x).sum(), [x], create_graph=True)
                field = 'aggregators' if model is pna else 'alpha' if model is gcn else 'metadata'
                with self.assertRaises(AttributeError):
                    setattr(model, field, 'changed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
