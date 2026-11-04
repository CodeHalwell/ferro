"""Real PyG COO convolution parity: outputs, gradients and named state."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.nn import GCNConv as TorchGCN, SAGEConv as TorchSAGE, GINConv as TorchGIN

import ferro as fr
from ferro.graph import GCNConv, SAGEConv, GINConv, EdgeIndex

torch.set_num_threads(1)


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


def edges(source, target):
    return fr.Tensor.from_i64(source + target, [2, len(source)])


def initialize(model, reference):
    actual, expected = dict(model.named_parameters()), dict(reference.named_parameters())
    if set(actual) != set(expected):
        raise AssertionError(f'Parameter names differ: {set(actual)} != {set(expected)}')
    generator = torch.Generator().manual_seed(123)
    with fr.no_grad(), torch.no_grad():
        for name, target in expected.items():
            data = torch.randn(target.shape, generator=generator) * .35
            target.copy_(data)
            actual[name].tensor().copy_(tensor(data))


class PygConvolutionTests(unittest.TestCase):
    def close(self, actual, expected, atol=3e-6, rtol=3e-5):
        if actual is None or expected is None:
            self.assertIsNone(actual)
            self.assertIsNone(expected)
        else:
            torch.testing.assert_close(values(actual), expected.detach(), atol=atol, rtol=rtol)

    def compare(self, model, reference, actual_x, reference_x, topology, torch_topology, *, actual_weight=None, reference_weight=None, size=None):
        initialize(model, reference)
        if isinstance(model, GCNConv):
            actual = model(actual_x, topology, actual_weight)
            expected = reference(reference_x, torch_topology, reference_weight)
        else:
            actual = model(actual_x, topology, size=size)
            expected = reference(reference_x, torch_topology, size=size)
        self.close(actual, expected)
        probe = (torch.arange(expected.numel(), dtype=torch.float32).reshape(expected.shape) + 1) / max(1, expected.numel())
        (actual * tensor(probe)).sum().backward()
        (expected * probe).sum().backward()
        pairs = zip(actual_x, reference_x) if isinstance(actual_x, tuple) else ((actual_x, reference_x),)
        for left, right in pairs:
            if left is not None:
                self.close(left.grad, right.grad)
        reference_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.close(parameter.grad, reference_params[name].grad)
        if actual_weight is not None:
            self.close(actual_weight.grad, reference_weight.grad, atol=5e-6, rtol=5e-5)
        return actual, expected

    def test_gcn_normalization_weights_loops_and_parallel_edges(self):
        source, target = [0, 1, 1, 2, 3, 2, 2], [1, 0, 0, 1, 3, 2, 0]
        index = torch.tensor([source, target], dtype=torch.int64)
        data = (torch.arange(15).reshape(5, 3) - 6) / 7
        options = ({}, {'improved': True}, {'add_self_loops': False},
                   {'normalize': False}, {'normalize': False, 'bias': False})
        for config in options:
            for weighted in (False, True):
                for native in (False, True):
                    with self.subTest(config=config, weighted=weighted, native=native):
                        model, reference = GCNConv(3, 2, **config), TorchGCN(3, 2, **config)
                        x, tx = tensor(data, True), data.clone().requires_grad_(True)
                        tw = (torch.arange(len(source)) * .15 + .4).requires_grad_(True) if weighted else None
                        weight = tensor(tw.detach(), True) if weighted else None
                        topology = EdgeIndex(5, source, target) if native else edges(source, target)
                        self.compare(model, reference, x, tx, topology, index, actual_weight=weight, reference_weight=tw)
        duplicate_source, duplicate_target = [0, 0, 1, 1], [0, 0, 0, 0]
        self.compare(GCNConv(3, 2), TorchGCN(3, 2), tensor(data, True), data.clone().requires_grad_(True),
                     edges(duplicate_source, duplicate_target), torch.tensor([duplicate_source, duplicate_target]))
        with self.assertRaisesRegex(NotImplementedError, 'weighted duplicate self-loops'):
            GCNConv(3, 2)(tensor(data), edges(duplicate_source, duplicate_target), fr.Tensor.ones([4]))
        # With insertion disabled no self-loop overwrites occur, including weights.
        weights = torch.tensor([.3, .5, .7, .9], requires_grad=True)
        self.compare(GCNConv(3, 2, add_self_loops=False), TorchGCN(3, 2, add_self_loops=False),
                     tensor(data, True), data.clone().requires_grad_(True), edges(duplicate_source, duplicate_target),
                     torch.tensor([duplicate_source, duplicate_target]), actual_weight=tensor(weights.detach(), True), reference_weight=weights)

    def test_sage_homogeneous_aggregations_projection_and_normalization(self):
        source, target = [0, 1, 1, 2, 3, 2], [1, 0, 0, 1, 3, 0]
        index = torch.tensor([source, target], dtype=torch.int64)
        # Keep the exact-PyG matrix away from its documented zero-max backward
        # discrepancy, tested explicitly below with the unchanged native result.
        data = (torch.arange(15).reshape(5, 3) - 7) / 6 + .137
        for aggregation in ('mean', 'sum', 'add', 'max'):
            for options in ({}, {'project': True, 'normalize': True}, {'root_weight': False, 'bias': False}):
                with self.subTest(aggregation=aggregation, options=options):
                    self.compare(SAGEConv(3, 2, aggr=aggregation, **options), TorchSAGE(3, 2, aggr=aggregation, **options),
                                 tensor(data, True), data.clone().requires_grad_(True), edges(source, target), index)

    def test_max_zero_gradient_boundary_and_positive_ties(self):
        for data in ((-1., 0., 2.), (2., 2., 1.)):
            with self.subTest(data=data):
                model = SAGEConv(1, 1, aggr='max', root_weight=False, bias=False)
                reference = TorchSAGE(1, 1, aggr='max', root_weight=False, bias=False)
                with fr.no_grad(), torch.no_grad():
                    model.lin_l.weight.tensor().copy_(fr.Tensor.ones([1, 1]))
                    reference.lin_l.weight.fill_(1.)
                x = fr.Tensor(list(data), [3, 1]).requires_grad_(True)
                tx = torch.tensor(data).reshape(3, 1).requires_grad_(True)
                actual = model(x, edges([0, 1], [2, 2]))
                expected = reference(tx, torch.tensor([[0, 1], [2, 2]]))
                self.close(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                self.close(model.lin_l.weight.grad, reference.lin_l.weight.grad)
                if data[1] == 0.:
                    self.assertEqual(x.grad.tolist(), [[0.], [1.], [0.]])
                    self.assertEqual(tx.grad.tolist(), [[0.], [.5], [0.]])
                else:
                    self.close(x.grad, tx.grad)
                    self.assertEqual(x.grad.tolist(), [[.5], [.5], [0.]])

    def test_sage_bipartite_and_missing_root(self):
        source, target = [0, 1, 1, 2, 3], [0, 0, 2, 3, 3]
        index = torch.tensor([source, target], dtype=torch.int64)
        left, right = torch.linspace(-1., 1., 12).reshape(4, 3), torch.linspace(.2, 1.1, 10).reshape(5, 2)
        for root in (True, False):
            for missing in (True, False):
                with self.subTest(root=root, missing=missing):
                    self.compare(SAGEConv((3, 2), 4, root_weight=root, project=True, normalize=True),
                                 TorchSAGE((3, 2), 4, root_weight=root, project=True, normalize=True),
                                 (tensor(left, True), None if missing else tensor(right, True)),
                                 (left.clone().requires_grad_(True), None if missing else right.clone().requires_grad_(True)),
                                 edges(source, target), index, size=(None, 5))

    def test_gin_supplied_network_trainable_and_buffer_eps(self):
        source, target = [0, 1, 1, 2, 3, 2], [1, 0, 0, 1, 3, 0]
        index = torch.tensor([source, target], dtype=torch.int64)
        left, right = torch.linspace(-1., 1., 12).reshape(4, 3), torch.linspace(.1, 1.5, 15).reshape(5, 3)
        for trained in (True, False):
            for mode in ('homogeneous', 'bipartite', 'missing'):
                with self.subTest(train_eps=trained, mode=mode):
                    network = fr.nn.Sequential(fr.nn.Linear(3, 4), fr.nn.Tanh(), fr.nn.Linear(4, 2))
                    oracle = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
                    model, reference = GINConv(network, eps=.17, train_eps=trained), TorchGIN(oracle, eps=.17, train_eps=trained)
                    if mode == 'homogeneous':
                        x, tx, size = tensor(left, True), left.clone().requires_grad_(True), None
                    else:
                        missing = mode == 'missing'
                        x = (tensor(left, True), None if missing else tensor(right, True))
                        tx = (left.clone().requires_grad_(True), None if missing else right.clone().requires_grad_(True))
                        size = (4, 5)
                    self.compare(model, reference, x, tx, edges(source, target), index, size=size)
                    self.assertEqual(set(dict(model.named_buffers())), set() if trained else {'eps'})
                    self.assertEqual('eps' in dict(model.named_parameters()), trained)

    def test_empty_topology_nodes_and_isolated_normalization(self):
        for nodes in (0, 4):
            for kind in ('gcn', 'sage', 'gin'):
                with self.subTest(nodes=nodes, kind=kind):
                    if kind == 'gcn':
                        model, reference = GCNConv(3, 2, add_self_loops=False), TorchGCN(3, 2, add_self_loops=False)
                    elif kind == 'sage':
                        model = SAGEConv(3, 2, root_weight=False, bias=False, normalize=True)
                        reference = TorchSAGE(3, 2, root_weight=False, bias=False, normalize=True)
                    else:
                        model = GINConv(fr.nn.Linear(3, 2), train_eps=True)
                        reference = TorchGIN(torch.nn.Linear(3, 2), train_eps=True)
                    data = torch.ones(nodes, 3)
                    self.compare(model, reference, tensor(data, True), data.requires_grad_(True), edges([], []), torch.empty(2, 0, dtype=torch.int64))

    def test_node_and_edge_permutations(self):
        source, target = [0, 1, 1, 2, 3, 2], [1, 0, 0, 1, 3, 0]
        data = torch.linspace(-1., 1., 15).reshape(5, 3)
        permutation, order = [3, 0, 4, 1, 2], [5, 2, 0, 4, 1, 3]
        inverse = [permutation.index(i) for i in range(5)]
        for model in (GCNConv(3, 2), SAGEConv(3, 2, aggr='max'), GINConv(fr.nn.Linear(3, 2))):
            with self.subTest(layer=type(model).__name__), fr.no_grad():
                original = model(tensor(data), edges(source, target))
                changed = model(tensor(data[permutation]), edges([inverse[source[i]] for i in order], [inverse[target[i]] for i in order]))
                self.close(changed, values(original)[permutation])

    def test_named_state_eps_persistence_and_atomic_config_rejection(self):
        factories = (
            ('gcn', lambda changed=False: GCNConv(3, 2, normalize=not changed)),
            ('sage', lambda changed=False: SAGEConv(3, 2, aggr='sum' if changed else 'mean')),
            ('gin_buffer', lambda changed=False: GINConv(fr.nn.Linear(3, 2), eps=.4 if changed else .2)),
            ('gin_parameter', lambda changed=False: GINConv(fr.nn.Linear(3, 2), eps=.4 if changed else .2, train_eps=True)),
        )
        topology, x = edges([0, 1, 2, 2], [1, 0, 1, 2]), tensor(torch.linspace(-1., 1., 9).reshape(3, 3))
        for name, factory in factories:
            with self.subTest(layer=name), tempfile.TemporaryDirectory() as directory:
                model, rng = factory(), fr.Generator(17)
                optimizer = fr.optim.Adam(model.parameters(), lr=.01)
                if name == 'gin_buffer':
                    model.eps.copy_(tensor([.37]))
                loss = model(x, topology).square().mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
                checkpoint.save(Path(directory) / 'training')
                loaded = fr.checkpoint.load(Path(directory) / 'training')
                restored, restored_rng = factory(), fr.Generator(999)
                restored_opt = fr.optim.Adam(restored.parameters(), lr=.7)
                self.assertEqual(loaded.restore(restored, optimizers={'main': restored_opt}, generator=restored_rng), 1)
                self.close(restored(x, topology), values(model(x, topology)), atol=0, rtol=0)
                if name == 'gin_buffer':
                    self.assertEqual(restored.eps.tolist(), model.eps.tolist())
                for live, opt in ((model, optimizer), (restored, restored_opt)):
                    live(x, topology).square().mean().backward()
                    opt.step()
                    opt.zero_grad()
                first = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=2)
                second = fr.checkpoint.snapshot_training(restored, optimizers={'main': restored_opt}, generator=restored_rng, step=2)
                self.assertEqual(self.state(first), self.state(second))
                wrong, wrong_rng = factory(True), fr.Generator(77)
                wrong_opt = fr.optim.Adam(wrong.parameters(), lr=.3)
                before = fr.checkpoint.snapshot_training(wrong, optimizers={'main': wrong_opt}, generator=wrong_rng)
                with self.assertRaises(ValueError):
                    loaded.restore(wrong, optimizers={'main': wrong_opt}, generator=wrong_rng)
                after = fr.checkpoint.snapshot_training(wrong, optimizers={'main': wrong_opt}, generator=wrong_rng)
                self.assertEqual(self.state(before), self.state(after))

    @staticmethod
    def state(checkpoint):
        return {name: (tuple(value.shape), value.dtype, json.dumps(value.tolist(), allow_nan=False)) for name, value in checkpoint.tensors().items()}

    def test_reset_keeps_slots_and_configuration_is_immutable(self):
        layers = (GCNConv(3, 2), SAGEConv(3, 2, project=True), GINConv(fr.nn.Linear(3, 2), eps=.3, train_eps=True))
        for model in layers:
            with self.subTest(layer=type(model).__name__):
                slots = {name: id(parameter) for name, parameter in model.named_parameters()}
                model.reset_parameters()
                self.assertEqual(slots, {name: id(parameter) for name, parameter in model.named_parameters()})
                field = 'initial_eps' if isinstance(model, GINConv) else 'normalize'
                with self.assertRaises(AttributeError):
                    setattr(model, field, 123)

    def test_invalid_and_explicitly_unsupported_inputs(self):
        for build in (lambda: GCNConv(3, 2, cached=True), lambda: GCNConv(3, 2, flow='target_to_source'),
                      lambda: GCNConv(3, 2, aggr='mean'), lambda: SAGEConv(3, 2, aggr='unsupported'),
                      lambda: SAGEConv(3, 2, decomposed_layers=2), lambda: GINConv(fr.nn.Identity(), aggr='mean')):
            with self.assertRaises(NotImplementedError):
                build()
        for build in (lambda: GCNConv(-1, 2), lambda: SAGEConv((-1, 3), 2), lambda: GCNConv(3, 2, normalize=False, add_self_loops=True),
                      lambda: GINConv(fr.nn.Identity(), eps=float('nan'))):
            with self.assertRaises(ValueError):
                build()
        with self.assertRaises(TypeError):
            GINConv(torch.nn.Identity())
        for layer in (GCNConv(3, 2), SAGEConv(3, 2), GINConv(fr.nn.Linear(3, 2))):
            with self.subTest(layer=type(layer).__name__):
                with self.assertRaises(ValueError):
                    layer(fr.Tensor.from_i64([1] * 9, [3, 3]), edges([0], [1]))
                with self.assertRaises(ValueError):
                    layer(tensor(torch.zeros(3, 3)), fr.Tensor.zeros([2, 0]))
                with self.assertRaises(ValueError):
                    layer(tensor(torch.zeros(3, 3)), edges([3], [0]))
        with self.assertRaises(ValueError):
            GCNConv(3, 2)(tensor(torch.zeros(3, 3)), edges([0], [1]), fr.Tensor.ones([1, 1]))
        with self.assertRaises(ValueError):
            SAGEConv((3, 2), 2)((tensor(torch.zeros(3, 3)), tensor(torch.zeros(2, 2))), edges([0], [1]), size=(3, 3))


if __name__ == '__main__':
    unittest.main()
