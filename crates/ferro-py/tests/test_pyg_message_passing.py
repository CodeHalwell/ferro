"""Real custom MessagePassing subclasses versus actual optional PyG."""
import inspect
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.graph import EdgeIndex, MessagePassing

try:
    from torch_geometric.nn import MessagePassing as TorchMessagePassing
except ImportError:
    TorchMessagePassing = None


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def indices(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


def eager(reference):
    # Compare this eager API with actual PyG's eager implementation. Its Jinja
    # templates specialize the first instance's flow and forward-call signature.
    reference.propagate = TorchMessagePassing.propagate.__get__(reference)
    reference.edge_updater = TorchMessagePassing.edge_updater.__get__(reference)
    return reference


class WeightedConv(MessagePassing):
    def __init__(self, channels=3, aggr='sum', flow='source_to_target'):
        super().__init__(aggr, flow=flow)
        self.channels = channels
        self.lin = fr.nn.Linear(2 * channels, 4)
        self.edge = fr.nn.Linear(2, 4, bias=False)
        self.post = fr.nn.Linear(4 * (len(aggr) if isinstance(aggr, (list, tuple)) else 1), 2)

    def forward(self, x, edge_index, edge_attr, root=None, size=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, root=root, size=size, gain=.7)

    def message(self, x_i, x_j, edge_attr, gain=1.):
        return self.lin(fr.cat([x_i, x_j], -1)).tanh() * gain + self.edge(edge_attr)

    def update(self, inputs, root=None):
        output = self.post(inputs)
        return output if root is None else output + root


class ElementConv(MessagePassing):
    def __init__(self, aggr='mean', flow='source_to_target', node_dim=-2):
        super().__init__(aggr, flow=flow, node_dim=node_dim)
        self.weight = fr.nn.Parameter(tensor([.63]))

    def forward(self, x, edge_index, edge_attr=None, size=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i=None, edge_attr=None):
        value = x_j * self.weight.tensor()
        if x_i is not None:
            value = value + x_i * .2
        return value if edge_attr is None else value + edge_attr

    def edge_update(self, x_j, x_i, edge_attr=None):
        output = x_i * self.weight.tensor() - x_j
        return output if edge_attr is None else output + edge_attr


if TorchMessagePassing is not None:
    class TorchWeightedConv(TorchMessagePassing):
        def __init__(self, channels=3, aggr='sum', flow='source_to_target'):
            super().__init__(aggr, flow=flow)
            self.channels = channels
            self.lin = torch.nn.Linear(2 * channels, 4)
            self.edge = torch.nn.Linear(2, 4, bias=False)
            self.post = torch.nn.Linear(4 * (len(aggr) if isinstance(aggr, (list, tuple)) else 1), 2)

        def forward(self, x, edge_index, edge_attr, root=None, size=None):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr, root=root, size=size, gain=.7)

        def message(self, x_i, x_j, edge_attr, gain=1.):
            return self.lin(torch.cat([x_i, x_j], -1)).tanh() * gain + self.edge(edge_attr)

        def update(self, inputs, root=None):
            output = self.post(inputs)
            return output if root is None else output + root

    class TorchElementConv(TorchMessagePassing):
        def __init__(self, aggr='mean', flow='source_to_target', node_dim=-2):
            super().__init__(aggr, flow=flow, node_dim=node_dim)
            self.weight = torch.nn.Parameter(torch.tensor([.63]))

        def forward(self, x, edge_index, edge_attr=None, size=None):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        def message(self, x_j, x_i=None, edge_attr=None):
            value = x_j * self.weight
            if x_i is not None:
                value = value + x_i * .2
            return value if edge_attr is None else value + edge_attr

        def edge_update(self, x_j, x_i, edge_attr=None):
            output = x_i * self.weight - x_j
            return output if edge_attr is None else output + edge_attr


@unittest.skipUnless(TorchMessagePassing is not None, 'PyG is an optional test reference')
class MessagePassingTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected.detach(), atol=2e-5, rtol=2e-4)

    def copy(self, model, reference):
        expected = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(expected))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(expected[name].detach()))

    def gradients(self, model, reference, actual, expected, inputs):
        self.close(actual, expected)
        probe = torch.randn(expected.shape, generator=torch.Generator().manual_seed(781))
        (actual * tensor(probe)).sum().backward()
        (expected * probe).sum().backward()
        parameters = dict(reference.named_parameters())
        for name, got, wanted in list(inputs) + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if wanted.grad is None:
                    self.assertIsNone(got.grad)
                else:
                    self.assertIsNotNone(got.grad)
                    self.close(got.grad, wanted.grad)

    def test_custom_parameterized_homogeneous_and_bipartite_routing(self):
        edges = torch.tensor([[0, 1, 1, 2, 3, 3], [1, 0, 1, 3, 2, 1]])
        for aggr in ('sum', 'mean', 'min', 'max', ['sum', 'mean', 'max']):
            for flow, paired in (('source_to_target', False), ('source_to_target', True), ('target_to_source', True)):
                with self.subTest(aggr=aggr, flow=flow, paired=paired):
                    torch.manual_seed(467)
                    model, reference = WeightedConv(aggr=aggr, flow=flow), eager(TorchWeightedConv(aggr=aggr, flow=flow))
                    self.copy(model, reference)
                    source = torch.randn(5, 3, requires_grad=True)
                    target = torch.randn(4, 3, requires_grad=True) if paired else source
                    attrs = torch.randn(6, 2, requires_grad=True)
                    rows = 5 if flow == 'target_to_source' or not paired else 4
                    root = torch.randn(rows, 2, requires_grad=True)
                    ss, st = tensor(source.detach(), True), None if not paired else tensor(target.detach(), True)
                    se, sr = tensor(attrs.detach(), True), tensor(root.detach(), True)
                    actual = model((ss, st) if paired else ss, indices(edges), se, sr)
                    wanted = reference((source, target) if paired else source, edges, attrs, root)
                    self.gradients(model, reference, actual, wanted, [('source', ss, source), ('edge', se, attrs), ('root', sr, root)] + ([] if not paired else [('target', st, target)]))

    def test_arbitrary_node_axis_rank_one_three_and_strided_features(self):
        edges = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 1, 2, 1]])
        for shape, axis in (((5,), 0), ((2, 5, 3), 1), ((5, 2, 3), 0), ((2, 3, 5), -1)):
            with self.subTest(shape=shape, axis=axis):
                torch.manual_seed(479)
                model, reference = ElementConv(node_dim=axis), eager(TorchElementConv(node_dim=axis))
                source = torch.randn(shape, requires_grad=True)
                native = tensor(source.detach(), True)
                self.gradients(model, reference, model(native, indices(edges)), reference(source, edges), [('x', native, source)])
        source = torch.randn(3, 5, requires_grad=True)
        native = tensor(source.detach(), True)
        model, reference = ElementConv(), eager(TorchElementConv())
        self.gradients(model, reference, model(native.transpose(0, 1), indices(edges)), reference(source.T, edges), [('strided', native, source)])

    def test_missing_target_explicit_sizes_and_empty_graphs(self):
        for nodes, targets, edges in ((4, 6, [[0, 1, 2], [1, 0, 4]]), (4, 6, [[], []]), (0, 0, [[], []])):
            with self.subTest(nodes=nodes, targets=targets):
                model, reference = ElementConv(aggr='sum'), eager(TorchElementConv(aggr='sum'))
                x = torch.randn(nodes, 3, requires_grad=True)
                native = tensor(x.detach(), True)
                edge = torch.tensor(edges, dtype=torch.int64)
                self.gradients(model, reference, model((native, None), indices(edge), size=(nodes, targets)),
                               reference((x, None), edge, size=(nodes, targets)), [('x', native, x)])

    def test_special_arguments_and_optional_defaults(self):
        class Special(MessagePassing):
            def message(self, x_j, index, edge_index_i, edge_index_j, size, size_i, size_j, dim_size, ptr, adj_t, edge_index, gain=.5):
                self._seen = (index.tolist(), edge_index_i.tolist(), edge_index_j.tolist(), list(size), size_i, size_j, dim_size, ptr, adj_t, edge_index.tolist())
                return x_j * gain
        class TorchSpecial(TorchMessagePassing):
            def message(self, x_j, index, edge_index_i, edge_index_j, size, size_i, size_j, dim_size, ptr, adj_t, edge_index, gain=.5):
                self._seen = (index.tolist(), edge_index_i.tolist(), edge_index_j.tolist(), list(size), size_i, size_j, dim_size, ptr, adj_t, edge_index.tolist())
                return x_j * gain
        edge = torch.tensor([[0, 1, 2], [1, 0, 1]])
        model, reference = Special(), eager(TorchSpecial())
        x = torch.randn(4, 3, requires_grad=True)
        native = tensor(x.detach(), True)
        self.gradients(model, reference, model.propagate(indices(edge), x=native), reference.propagate(edge, x=x, gain=.5), [('x', native, x)])
        self.assertEqual(model._seen, reference._seen)

    def test_custom_aggregation_and_edge_only_size_inference(self):
        class Custom(MessagePassing):
            def __init__(self):
                super().__init__(aggr=None)
            def message(self, edge_attr, factor=2.):
                return edge_attr * factor
            def aggregate(self, inputs, index, dim_size=None, offset=0.):
                return fr.graph.scatter(inputs, index.tolist(), dim_size if dim_size is not None else max(index.tolist(), default=-1) + 1, 'sum') + offset
        class TorchCustom(TorchMessagePassing):
            def __init__(self):
                super().__init__(aggr=None)
            def message(self, edge_attr, factor=2.):
                return edge_attr * factor
            def aggregate(self, inputs, index, dim_size=None, offset=0.):
                from torch_geometric.utils import scatter
                return scatter(inputs, index, dim_size=dim_size, reduce='sum') + offset
        edges = torch.tensor([[0, 8, 9], [1, 0, 1]])
        model, reference = Custom(), eager(TorchCustom())
        edge = torch.randn(3, 2, requires_grad=True)
        native = tensor(edge.detach(), True)
        self.gradients(model, reference, model.propagate(indices(edges), edge_attr=native, offset=.3),
                       reference.propagate(edges, edge_attr=edge, offset=.3, factor=2.), [('edge', native, edge)])

    def test_edge_updater_parameter_and_feature_gradients(self):
        edges = torch.tensor([[0, 1, 2], [1, 0, 1]])
        for flow in ('source_to_target', 'target_to_source'):
            with self.subTest(flow=flow):
                model, reference = ElementConv(flow=flow), eager(TorchElementConv(flow=flow))
                x, y, attrs = torch.randn(4, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)
                sx, sy, se = tensor(x.detach(), True), tensor(y.detach(), True), tensor(attrs.detach(), True)
                self.gradients(model, reference, model.edge_updater(indices(edges), x=(sx, sy), edge_attr=se),
                               reference.edge_updater(edges, x=(x, y), edge_attr=attrs), [('x', sx, x), ('y', sy, y), ('edge', se, attrs)])

    def test_integer_node_ids_lift_into_trainable_embedding(self):
        class EmbeddingConv(MessagePassing):
            def __init__(self):
                super().__init__(node_dim=0)
                self.embedding = fr.nn.Embedding(7, 3)
            def message(self, x_j):
                return self.embedding(x_j)
        class TorchEmbeddingConv(TorchMessagePassing):
            def __init__(self):
                super().__init__(node_dim=0)
                self.embedding = torch.nn.Embedding(7, 3)
            def message(self, x_j):
                return self.embedding(x_j)
        model, reference = EmbeddingConv(), eager(TorchEmbeddingConv())
        self.copy(model, reference)
        ids = torch.tensor([2, 1, 4, 2])
        edges = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 3]])
        self.gradients(model, reference, model.propagate(indices(edges), x=indices(ids)),
                       reference.propagate(edges, x=ids), [])

    def test_hooks_replace_inputs_outputs_remove_and_preserve_gradients(self):
        edges = torch.tensor([[0, 1, 2], [1, 0, 1]])
        model, reference = ElementConv(), eager(TorchElementConv())
        events, expected_events = [], []
        def attach(module, log):
            def propagate_pre(_, args):
                edge, size, kwargs = args
                log.append('propagate_pre')
                return edge, size, {**kwargs, 'edge_attr': kwargs['edge_attr'] * .8}
            def message_pre(_, args):
                log.append('message_pre')
                return ({**args[0], 'x_j': args[0]['x_j'] * 1.1},)
            def message_post(_, args, output):
                log.append('message_post')
                return output * .7
            def aggregate_pre(_, args):
                log.append('aggregate_pre')
                return args[0]
            def aggregate_post(_, args, output):
                log.append('aggregate_post')
                return output + .4
            def propagate_post(_, args, output):
                log.append(('propagate_post', list(args[1])))
                return output * .9
            return [getattr(module, name)(hook) for name, hook in (
                ('register_propagate_forward_pre_hook', propagate_pre), ('register_message_forward_pre_hook', message_pre),
                ('register_message_forward_hook', message_post), ('register_aggregate_forward_pre_hook', aggregate_pre),
                ('register_aggregate_forward_hook', aggregate_post), ('register_propagate_forward_hook', propagate_post))]
        handles, refs = attach(model, events), attach(reference, expected_events)
        x, attrs = torch.randn(4, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)
        sx, se = tensor(x.detach(), True), tensor(attrs.detach(), True)
        self.gradients(model, reference, model(sx, indices(edges), se), reference(x, edges, attrs), [('x', sx, x), ('edge', se, attrs)])
        self.assertEqual(events, expected_events)
        for handle in handles + refs:
            handle.remove()
            handle.remove()
        events.clear()
        self.close(model(tensor(x.detach()), indices(edges), tensor(attrs.detach())), reference(x.detach(), edges, attrs.detach()))
        self.assertEqual(events, [])
        with model.register_message_forward_hook(lambda module, args, output: output * 2):
            self.assertEqual(len(model._mp_hooks['message_post']), 1)
        self.assertEqual(len(model._mp_hooks['message_post']), 0)

    def test_edge_update_hooks_and_named_node_inputs(self):
        class Attention(MessagePassing):
            def edge_update(self, q_i, k_j, edge_bias=None):
                output = (q_i * k_j).sum_dim(1, False)
                return output if edge_bias is None else output + edge_bias
        class TorchAttention(TorchMessagePassing):
            def edge_update(self, q_i, k_j, edge_bias=None):
                output = (q_i * k_j).sum(dim=1)
                return output if edge_bias is None else output + edge_bias
        model, reference = Attention(), eager(TorchAttention())
        events, expected_events = [], []
        def attach(module, log):
            def before(_, args):
                edge, size, kwargs = args
                log.append(('before', size))
                return edge, size, {**kwargs, 'edge_bias': kwargs['edge_bias'] * .8}
            def after(_, args, output):
                log.append(('after', args[1]))
                return output * 1.1
            module.register_edge_update_forward_pre_hook(before)
            module.register_edge_update_forward_hook(after)
        attach(model, events)
        attach(reference, expected_events)
        edge = torch.tensor([[0, 1, 2], [1, 0, 1]])
        q, k, bias = torch.randn(3, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True), torch.randn(3, requires_grad=True)
        sq, sk, sb = tensor(q.detach(), True), tensor(k.detach(), True), tensor(bias.detach(), True)
        self.gradients(model, reference, model.edge_updater(indices(edge), size=(5, 3), q=sq, k=sk, edge_bias=sb),
                       reference.edge_updater(edge, size=(5, 3), q=q, k=k, edge_bias=bias), [('q', sq, q), ('k', sk, k), ('bias', sb, bias)])
        self.assertEqual(events, expected_events)

    def test_multi_aggregation_rank_three_and_checkpoint_parameter_ties(self):
        model, reference = ElementConv(aggr=['sum', 'mean', 'min'], node_dim=1), eager(TorchElementConv(aggr=['sum', 'mean', 'min'], node_dim=1))
        x = torch.randn(2, 4, 3, requires_grad=True)
        sx = tensor(x.detach(), True)
        edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.gradients(model, reference, model(sx, indices(edge)), reference(x, edge), [('x', sx, x)])
        class Tied(ElementConv):
            def __init__(self, tied=True):
                super().__init__()
                self.alias = self.weight if tied else fr.nn.Parameter(tensor([.63]))
        source, target = Tied(), Tied()
        state = fr.checkpoint.snapshot_training(source, optimizers={}, generator=fr.Generator(2), step=4)
        self.assertEqual(state.restore(target, optimizers={}, generator=fr.Generator(9)), 4)
        self.assertIs(target.alias, target.weight)
        with self.assertRaises(ValueError):
            state.restore(Tied(False), optimizers={}, generator=fr.Generator(9))

    def test_native_edge_index_zero_extrema_ties_and_checkpoint(self):
        edges = EdgeIndex(4, [0, 1, 2], [1, 0, 1])
        model = WeightedConv()
        self.assertIn('channels', str(inspect.signature(WeightedConv)))
        x, attrs = tensor(torch.randn(4, 3)), tensor(torch.randn(3, 2))
        optimizer = fr.optim.Adam(model.parameters(), lr=.01)
        model(x, edges, attrs).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=fr.Generator(7), step=1)
        with tempfile.TemporaryDirectory() as directory:
            snapshot.save(directory)
            restored = WeightedConv()
            other = fr.optim.Adam(restored.parameters(), lr=.9)
            self.assertEqual(fr.checkpoint.load_training(directory, restored, optimizers={'main': other}, generator=fr.Generator(99)), 1)
            self.assertEqual(model(x, edges, attrs).tolist(), restored(x, edges, attrs).tolist())
            for current, opt in ((model, optimizer), (restored, other)):
                output = current(x, edges, attrs)
                (output * output).sum().backward()
                opt.step()
                opt.zero_grad()
            self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in restored.parameters()])
        state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1))
        with self.assertRaises(ValueError):
            state.restore(WeightedConv(aggr='mean'), optimizers={}, generator=fr.Generator(2))
        maximum = MessagePassing('max', node_dim=0)
        messages = tensor([[0.], [0.]], True)
        maximum.aggregate(messages, indices([0, 0]), dim_size=2).sum().backward()
        self.assertEqual(messages.grad.tolist(), [[.5], [.5]])

    def test_strict_invalid_signatures_sizes_inputs_and_options(self):
        class Ordinary(fr.nn.Module):
            def __init__(self):
                super().__init__()
        self.assertTrue(Ordinary().training)
        with self.assertRaises(TypeError):
            class Variadic(MessagePassing):
                def message(self, **kwargs):
                    return kwargs['x_j']
            Variadic()
        for factory in (lambda: MessagePassing('median'), lambda: MessagePassing([]),
                        lambda: MessagePassing(decomposed_layers=2), lambda: MessagePassing(aggr_kwargs={'foo': 1})):
            with self.assertRaises(NotImplementedError):
                factory()
        with self.assertRaises(ValueError):
            MessagePassing(flow='unknown')
        with self.assertRaises(TypeError):
            MessagePassing(node_dim=True)
        x, edge = tensor(torch.ones(3, 2)), indices([[0, 1], [1, 2]])
        model = ElementConv()
        for call in (lambda: model(x, indices([[0, 3], [1, 2]])), lambda: model((x, x), edge, size=(3, 4)),
                     lambda: model(x, edge, size=(2, 3)), lambda: model((x,), edge),
                     lambda: model(fr.tensor([[1., 2.]] * 3, dtype='float64'), edge),
                     lambda: MessagePassing(node_dim=3).propagate(edge, x=x)):
            with self.assertRaises((ValueError, TypeError)):
                call()
        with self.assertRaises(TypeError):
            MessagePassing().propagate(edge)
        with self.assertRaises(NotImplementedError):
            MessagePassing(None).propagate(edge, x=x)
        with self.assertRaises(NotImplementedError):
            model.message_and_aggregate(None)
        with self.assertRaises(TypeError):
            model.register_message_forward_hook(42)
        with model.register_message_forward_pre_hook(lambda *args: 3):
            with self.assertRaises(TypeError):
                model(x, edge)
        with self.assertRaises(NotImplementedError):
            model.explain = True
        with self.assertRaises(AttributeError):
            model.flow = 'target_to_source'
        with self.assertRaises(ValueError):
            model(x, edge, size=(3, sys.maxsize))
        with self.assertRaisesRegex(ValueError, 'capture'):
            fr.capture(lambda: model(x, edge))
        from ferro.autograd import grad
        tracked = tensor(torch.ones(3, 2), True)
        with self.assertRaisesRegex(ValueError, 'unsupported|create_graph'):
            grad(model(tracked, edge).sum(), [tracked], create_graph=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
