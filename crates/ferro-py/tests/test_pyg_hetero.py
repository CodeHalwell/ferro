"""Actual PyG heterogeneous storage, relation aggregation and state parity."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import torch
from torch_geometric.data import HeteroData as TorchData
from torch_geometric.nn import HeteroConv as TorchHetero, SAGEConv as TorchSAGE, GCNConv as TorchGCN, RGCNConv as TorchRGCN

import ferro as fr
from ferro.graph import HeteroData, HeteroConv, SAGEConv, GCNConv, RGCNConv

torch.set_num_threads(1)
AB = ('a', 'writes', 'b')
BB = ('b', 'cites', 'b')
BA = ('b', 'returns', 'a')


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def topology(source, target):
    return fr.Tensor.from_i64(source + target, [2, len(source)])


def data_pair(grad=False):
    actual, expected = HeteroData(), TorchData()
    features = {'a': torch.linspace(-.8, .9, 6).reshape(3, 2),
                'b': torch.linspace(-.7, 1.3, 12).reshape(4, 3),
                'unused': torch.ones(2, 1)}
    for name, value in features.items():
        actual[name].x = tensor(value, grad)
        expected[name].x = value.clone().requires_grad_(grad)
    indices = {AB: ([0, 1, 1, 2], [1, 0, 0, 3]), BB: ([0, 1, 1, 2], [1, 0, 0, 2]),
               BA: ([0, 2, 3], [0, 1, 1])}
    for name, (src, dst) in indices.items():
        actual[name].edge_index = topology(src, dst)
        expected[name].edge_index = torch.tensor([src, dst], dtype=torch.int64)
    weight = torch.tensor([.7, 1.2, .9, 1.1])
    actual[BB].edge_weight = tensor(weight, grad)
    expected[BB].edge_weight = weight.clone().requires_grad_(grad)
    actual.y, expected.y = fr.Tensor.from_i64([4], [1]), torch.tensor([4])
    return actual, expected


def initialize(model, oracle):
    actual, expected = dict(model.named_parameters()), dict(oracle.named_parameters())
    if set(actual) != set(expected):
        raise AssertionError(f'parameter names differ: {set(actual)} != {set(expected)}')
    rng = torch.Generator().manual_seed(31)
    with fr.no_grad(), torch.no_grad():
        for name, parameter in expected.items():
            value = torch.randn(parameter.shape, generator=rng) * .4
            parameter.copy_(value)
            actual[name].tensor().copy_(tensor(value))


def hetero_pair(aggregation='sum'):
    actual = HeteroConv({AB: SAGEConv((2, 3), 2), BB: GCNConv(3, 2), BA: SAGEConv((3, 2), 2)}, aggr=aggregation)
    oracle = TorchHetero({AB: TorchSAGE((2, 3), 2), BB: TorchGCN(3, 2), BA: TorchSAGE((3, 2), 2)}, aggr=aggregation)
    initialize(actual, oracle)
    return actual, oracle


class Scale(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.Tensor.ones([1]))

    def forward(self, x, edge_index):
        return x * self.weight.tensor()


class TorchScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        return x * self.weight


class HeterogeneousGraphTests(unittest.TestCase):
    def close(self, actual, expected, atol=5e-6, rtol=5e-5):
        if actual is None or expected is None:
            self.assertIsNone(actual)
            self.assertIsNone(expected)
        else:
            got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(actual.shape)
            torch.testing.assert_close(got, expected.detach(), atol=atol, rtol=rtol)

    def test_data_stores_metadata_dicts_and_roundtrip(self):
        actual, expected = data_pair()
        actual['isolated'].num_nodes, expected['isolated'].num_nodes = 3, 3
        actual['a'].y, expected['a'].y = fr.Tensor.from_i64([0, 1, 0], [3]), torch.tensor([0, 1, 0])
        self.assertEqual(actual.metadata(), expected.metadata())
        self.assertEqual(len(actual.stores), len(expected.stores))
        self.assertEqual(actual.stores[0]['y'].tolist(), expected.stores[0]['y'].tolist())
        self.assertEqual(actual.num_nodes, expected.num_nodes)
        self.assertEqual(actual.num_edges, expected.num_edges)
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        self.assertEqual(len(actual), len(expected))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(actual.validate(), expected.validate())
        for name, value in actual.x_dict.items():
            self.close(value, expected.x_dict[name])
        for name, value in actual.edge_index_dict.items():
            self.close(value, expected.edge_index_dict[name], atol=0, rtol=0)
            self.assertEqual(actual[name].size(), expected[name].size())
        self.assertEqual(actual['a'].num_node_features, expected['a'].num_node_features)
        self.assertEqual(actual.collect('y').keys(), expected.collect('y').keys())
        encoded = actual.to_dict()
        self.assertEqual(set(encoded), set(expected.to_dict()))
        restored = HeteroData(encoded)
        self.assertEqual(restored.metadata(), actual.metadata())
        self.assertTrue(restored.validate())
        self.assertEqual(restored['isolated'].num_nodes, 3)
        self.assertEqual(restored.y.tolist(), [4])

    def test_clone_owns_fields_preserves_gradients_and_cpu_to(self):
        data, _ = data_pair()
        cloned = data.clone()
        self.assertIs(cloned.to('cpu'), cloned)
        cloned['a'].x.copy_(fr.Tensor.zeros([3, 2]))
        self.assertNotEqual(data['a'].x.tolist(), cloned['a'].x.tolist())
        cloned[AB].edge_index = topology([], [])
        self.assertEqual(data[AB].edge_index.shape, [2, 4])
        self.assertEqual(cloned.y.dtype, 'int64')
        with self.assertRaises(NotImplementedError):
            data.to('cuda:0')
        data, reference = data_pair(True)
        actual, expected = data.clone(), reference.clone()
        actual['a'].x.square().sum().backward()
        expected['a'].x.square().sum().backward()
        self.close(data['a'].x.grad, reference['a'].x.grad)
        self.assertIsNot(actual['a'].x, data['a'].x)
        subset = data.clone('x')
        self.assertIs(subset[AB].edge_index, data[AB].edge_index)

    def test_native_boolean_masks_are_typed_and_cloned(self):
        data, reference = data_pair()
        data['a'].train_mask = fr.Tensor.from_bool([True, False, True], [3])
        reference['a'].train_mask = torch.tensor([True, False, True])
        data[AB].edge_mask = fr.Tensor.from_bool([False, True, True, False], [4])
        reference[AB].edge_mask = torch.tensor([False, True, True, False])
        self.assertTrue(data.validate())
        clone = data.clone()
        self.assertEqual(clone['a'].train_mask.dtype, 'bool')
        self.assertIsNot(clone['a'].train_mask, data['a'].train_mask)
        self.close(clone['a'].train_mask, reference['a'].train_mask, atol=0, rtol=0)
        self.close(clone[AB].edge_mask, reference[AB].edge_mask, atol=0, rtol=0)

    def test_data_validation_errors_are_explicit(self):
        invalid = HeteroData()
        invalid[AB].edge_index = topology([0], [0])
        self.assertFalse(invalid.validate(False))
        with self.assertRaisesRegex(ValueError, 'undefined node type'):
            invalid.validate()
        invalid['a'].num_nodes, invalid['b'].num_nodes = 1, 1
        self.assertTrue(invalid.validate())
        invalid[AB].edge_index = topology([1], [0])
        self.assertFalse(invalid.validate(False))
        data, _ = data_pair()
        data['a'].num_nodes = 4
        with self.assertRaisesRegex(ValueError, 'row count'):
            data.validate()
        data['a'].num_nodes = 3
        data[AB].edge_attr = fr.Tensor.ones([3, 2])
        with self.assertRaisesRegex(ValueError, 'edge field row count'):
            data.validate()
        with self.assertRaises(ValueError):
            data[AB].edge_index = fr.Tensor.zeros([2, 0])
        data['a'].x = fr.Tensor.from_f64([1., 2.], [1, 2])
        self.assertEqual(data['a'].x.dtype, 'float64')
        with self.assertRaises(TypeError):
            data['a'].x = [[1., 2.]]
        with self.assertRaises(ValueError):
            data[('a', 'b')]
        with self.assertRaises(KeyError):
            HeteroData().x_dict
        self.assertEqual(HeteroData().collect('x', allow_empty=True), {})

    def test_real_pyg_all_aggregations_input_parameter_and_edge_gradients(self):
        for aggregation in ('sum', 'mean', 'min', 'max', 'cat', None):
            with self.subTest(aggregation=aggregation):
                data, reference = data_pair(True)
                model, oracle = hetero_pair(aggregation)
                actual = model(data.x_dict, data.edge_index_dict, edge_weight_dict=data.edge_weight_dict)
                expected = oracle(reference.x_dict, reference.edge_index_dict, edge_weight_dict=reference.edge_weight_dict)
                self.assertEqual(set(actual), {'a', 'b'})
                total, wanted = None, None
                for name, output in actual.items():
                    self.close(output, expected[name])
                    probe = (torch.arange(expected[name].numel()).reshape(expected[name].shape) + 1).float() / expected[name].numel()
                    term, target = (output * tensor(probe)).sum(), (expected[name] * probe).sum()
                    total = term if total is None else total + term
                    wanted = target if wanted is None else wanted + target
                total.backward()
                wanted.backward()
                for name in data.node_types:
                    self.close(data[name].x.grad, reference[name].x.grad)
                self.close(data[BB].edge_weight.grad, reference[BB].edge_weight.grad)
                expected_params = dict(oracle.named_parameters())
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        self.close(parameter.grad, expected_params[name].grad)

    def test_max_min_ties_follow_first_relation(self):
        keys = (('a', 'one', 'a'), ('a', 'two', 'a'))
        for aggregation in ('min', 'max'):
            with self.subTest(aggregation=aggregation):
                model = HeteroConv({key: Scale() for key in keys}, aggr=aggregation)
                oracle = TorchHetero({key: TorchScale() for key in keys}, aggr=aggregation)
                x, tx = tensor([[1., -2.]], True), torch.tensor([[1., -2.]], requires_grad=True)
                actual = model({'a': x}, {key: topology([], []) for key in keys})['a']
                expected = oracle({'a': tx}, {key: torch.empty(2, 0, dtype=torch.int64) for key in keys})['a']
                self.close(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                self.close(x.grad, tx.grad)
                for name, parameter in model.named_parameters():
                    self.close(parameter.grad, dict(oracle.named_parameters())[name].grad)
                self.assertEqual(model.convs[keys[0]].weight.grad.tolist(), [-1.])
                self.assertEqual(model.convs[keys[1]].weight.grad.tolist(), [0.])

    def test_missing_relations_empty_graphs_and_keyword_routing(self):
        data, reference = data_pair(True)
        model, oracle = hetero_pair(None)
        actual = model(data.x_dict, {AB: data[AB].edge_index})
        expected = oracle(reference.x_dict, {AB: reference[AB].edge_index})
        self.assertEqual(set(actual), {'b'})
        self.assertEqual(actual['b'].shape, [4, 1, 2])
        self.close(actual['b'], expected['b'])
        self.assertEqual(model(data.x_dict, {}), {})
        missing = model(x_dict={'a': data['a'].x}, edge_index_dict={AB: data[AB].edge_index}, size_dict={AB: (3, 4)})
        oracle_missing = oracle(x_dict={'a': reference['a'].x}, edge_index_dict={AB: reference[AB].edge_index}, size_dict={AB: (3, 4)})
        self.close(missing['b'], oracle_missing['b'])
        for aggregation in ('sum', 'max', None):
            with self.subTest(empty_aggregation=aggregation):
                model, oracle = hetero_pair(aggregation)
                features = {'a': fr.Tensor.zeros([0, 2]), 'b': fr.Tensor.zeros([0, 3])}
                tf = {'a': torch.empty(0, 2), 'b': torch.empty(0, 3)}
                ei = {key: topology([], []) for key in (AB, BB, BA)}
                tei = {key: torch.empty(2, 0, dtype=torch.int64) for key in ei}
                actual, expected = model(features, ei), oracle(tf, tei)
                for key in actual:
                    self.close(actual[key], expected[key])

    def test_shared_children_checkpoint_and_configuration_are_atomic(self):
        relations = (('a', 'one', 'b'), ('a', 'two', 'b'))
        def build(shared=True, reverse=False, aggr='sum'):
            layer = SAGEConv((2, 3), 2)
            keys = relations[::-1] if reverse else relations
            return HeteroConv({keys[0]: layer, keys[1]: layer if shared else SAGEConv((2, 3), 2)}, aggr=aggr)
        data, reference = data_pair(True)
        ei = {key: data[AB].edge_index for key in relations}
        model, rng = build(), fr.Generator(17)
        shared = TorchSAGE((2, 3), 2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            oracle = TorchHetero({key: shared for key in relations})
        initialize(model, oracle)
        self.assertIs(model.convs[relations[0]], model.convs[relations[1]])
        self.assertEqual(len(list(model.named_parameters(remove_duplicate=False))), 2 * len(list(model.named_parameters())))
        optimizer = fr.optim.AdamW(model.parameters(), lr=.01)
        actual = model(data.x_dict, ei)['b']
        expected = oracle(reference.x_dict, {key: reference[AB].edge_index for key in relations})['b']
        self.close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for name in data.node_types:
            self.close(data[name].x.grad, reference[name].x.grad)
        for name, parameter in model.named_parameters():
            self.close(parameter.grad, dict(oracle.named_parameters())[name].grad)
        optimizer.step()
        optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state'
            snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
            snapshot.save(path)
            loaded = fr.checkpoint.load(path)
            restored, rrng = build(), fr.Generator(999)
            ropt = fr.optim.AdamW(restored.parameters(), lr=.7)
            self.assertEqual(loaded.restore(restored, optimizers={'main': ropt}, generator=rrng), 1)
            for live, opt in ((model, optimizer), (restored, ropt)):
                live(data.x_dict, ei)['b'].square().mean().backward()
                opt.step()
                opt.zero_grad()
            first = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=2)
            second = fr.checkpoint.snapshot_training(restored, optimizers={'main': ropt}, generator=rrng, step=2)
            self.assertEqual(self.state(first), self.state(second))
            for config in ({'shared': False}, {'reverse': True}, {'aggr': 'mean'}):
                with self.subTest(mismatch=config):
                    wrong, wrng = build(**config), fr.Generator(23)
                    wopt = fr.optim.AdamW(wrong.parameters(), lr=.3)
                    before = fr.checkpoint.snapshot_training(wrong, optimizers={'main': wopt}, generator=wrng)
                    with self.assertRaises(ValueError):
                        loaded.restore(wrong, optimizers={'main': wopt}, generator=wrng)
                    after = fr.checkpoint.snapshot_training(wrong, optimizers={'main': wopt}, generator=wrng)
                    self.assertEqual(self.state(before), self.state(after))
        with self.assertRaises(AttributeError):
            model.aggr = 'mean'
        with self.assertRaises(AttributeError):
            model.convs.relations = relations[::-1]

    @staticmethod
    def state(checkpoint):
        return {key: (tuple(value.shape), value.dtype, json.dumps(value.tolist(), allow_nan=False)) for key, value in checkpoint.tensors().items()}

    def test_unsupported_and_invalid_relation_contracts(self):
        for key in ('a', ('a', 'b'), ('a', '', 'b')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                HeteroConv({key: SAGEConv(2, 2)})
        with self.assertRaises(TypeError):
            HeteroConv({AB: torch.nn.Linear(2, 2)})
        with self.assertRaisesRegex(ValueError, 'self-loop insertion'):
            HeteroConv({AB: GCNConv(2, 2)})
        with self.assertRaises(NotImplementedError):
            HeteroConv({}, aggr='lstm')
        with self.assertRaisesRegex(ValueError, 'collide'):
            HeteroConv({('a.b', 'r', 'c'): SAGEConv(2, 2), ('a#b', 'r', 'c'): SAGEConv(2, 2)})
        model, _ = hetero_pair()
        with self.assertRaisesRegex(ValueError, 'end with _dict'):
            model({}, {}, edge_weight={})
        with self.assertRaises(TypeError):
            model(fr.Tensor.zeros([2, 3]), {})

    def test_rgcn_real_pyg_relation_and_basis_gradients(self):
        src, dst, kinds = [0, 1, 1, 2, 3, 2], [1, 0, 0, 1, 3, 0], [1, 0, 0, 1, 2, 2]
        left, right = torch.linspace(-.7, .9, 12).reshape(4, 3), torch.linspace(-.2, 1., 10).reshape(5, 2)
        for bases in (None, 2):
            for aggregation in ('mean', 'sum', 'add'):
                for bipartite in (False, True):
                    with self.subTest(bases=bases, aggregation=aggregation, bipartite=bipartite):
                        width = (3, 2) if bipartite else 3
                        model = RGCNConv(width, 2, 4, num_bases=bases, aggr=aggregation)
                        oracle = TorchRGCN(width, 2, 4, num_bases=bases, aggr=aggregation)
                        initialize(model, oracle)
                        if bipartite:
                            x = (tensor(left, True), tensor(right, True))
                            tx = (left.clone().requires_grad_(True), right.clone().requires_grad_(True))
                        else:
                            x, tx = tensor(left, True), left.clone().requires_grad_(True)
                        actual = model(x, topology(src, dst), fr.Tensor.from_i64(kinds, [len(kinds)]))
                        expected = oracle(tx, torch.tensor([src, dst]), torch.tensor(kinds))
                        self.close(actual, expected)
                        probe = torch.linspace(.1, .9, expected.numel()).reshape(expected.shape)
                        (actual * tensor(probe)).sum().backward()
                        (expected * probe).sum().backward()
                        pairs = zip(x, tx) if bipartite else ((x, tx),)
                        for value, target in pairs:
                            self.close(value.grad, target.grad)
                        for name, parameter in model.named_parameters():
                            self.close(parameter.grad, dict(oracle.named_parameters())[name].grad)

    def test_rgcn_mean_is_normalized_separately_by_relation(self):
        model = RGCNConv(1, 1, 2, root_weight=False, bias=False)
        oracle = TorchRGCN(1, 1, 2, root_weight=False, bias=False)
        with fr.no_grad(), torch.no_grad():
            model.weight.tensor().copy_(fr.Tensor.ones([2, 1, 1]))
            oracle.weight.fill_(1.)
        source, target = tensor([[2.], [4.], [10.]], True), fr.Tensor.zeros([1, 1])
        ts, tt = torch.tensor([[2.], [4.], [10.]], requires_grad=True), torch.zeros(1, 1)
        actual = model((source, target), topology([0, 1, 2], [0, 0, 0]), fr.Tensor.from_i64([0, 0, 1], [3]))
        expected = oracle((ts, tt), torch.tensor([[0, 1, 2], [0, 0, 0]]), torch.tensor([0, 0, 1]))
        self.close(actual, expected)
        self.assertEqual(actual.tolist(), [[13.]])
        actual.sum().backward()
        expected.sum().backward()
        self.close(source.grad, ts.grad)
        self.assertEqual(source.grad.tolist(), [[.5], [.5], [1.]])

    def test_rgcn_empty_inputs_and_edge_permutations(self):
        for nodes in (0, 3):
            with self.subTest(nodes=nodes):
                model = RGCNConv(2, 3, 2, num_bases=1, root_weight=False, bias=False)
                oracle = TorchRGCN(2, 3, 2, num_bases=1, root_weight=False, bias=False)
                initialize(model, oracle)
                x, tx = fr.Tensor.ones([nodes, 2]).requires_grad_(True), torch.ones(nodes, 2, requires_grad=True)
                actual = model(x, topology([], []), fr.Tensor.from_i64([], [0]))
                expected = oracle(tx, torch.empty(2, 0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                self.close(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                self.close(x.grad, tx.grad)
                for name, parameter in model.named_parameters():
                    self.close(parameter.grad, dict(oracle.named_parameters())[name].grad)
        src, dst, kinds = [0, 1, 1, 2], [1, 0, 0, 1], [0, 1, 0, 1]
        model = RGCNConv(2, 3, 2, num_bases=1)
        x = tensor(torch.linspace(-.4, .8, 6).reshape(3, 2))
        with fr.no_grad():
            original = model(x, topology(src, dst), fr.Tensor.from_i64(kinds, [4]))
            order = [3, 1, 0, 2]
            changed = model(x, topology([src[i] for i in order], [dst[i] for i in order]), fr.Tensor.from_i64([kinds[i] for i in order], [4]))
            self.close(changed, torch.tensor(original.tolist()))

    def test_rgcn_checkpoint_config_and_invalid_ids(self):
        model, rng = RGCNConv(2, 3, 2, num_bases=1), fr.Generator(37)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.02)
        x, ei, rel = fr.Tensor.ones([3, 2]), topology([0, 1], [1, 2]), fr.Tensor.from_i64([0, 1], [2])
        model(x, ei, rel).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
        target, trng = RGCNConv(2, 3, 2, num_bases=1), fr.Generator(999)
        topt = fr.optim.AdamW(target.parameters(), lr=.7)
        saved.restore(target, optimizers={'main': topt}, generator=trng)
        restored = fr.checkpoint.snapshot_training(target, optimizers={'main': topt}, generator=trng, step=1)
        self.assertEqual(self.state(saved), self.state(restored))
        wrong, wrng = RGCNConv(2, 3, 2, num_bases=1, aggr='sum'), fr.Generator(13)
        wopt = fr.optim.AdamW(wrong.parameters(), lr=.8)
        before = fr.checkpoint.snapshot_training(wrong, optimizers={'main': wopt}, generator=wrng)
        with self.assertRaises(ValueError):
            saved.restore(wrong, optimizers={'main': wopt}, generator=wrng)
        self.assertEqual(self.state(before), self.state(fr.checkpoint.snapshot_training(wrong, optimizers={'main': wopt}, generator=wrng)))
        for ids in (fr.Tensor.from_i64([-1, 0], [2]), fr.Tensor.from_i64([0, 2], [2]), fr.Tensor.ones([2]), fr.Tensor.from_i64([0], [1])):
            with self.assertRaises(ValueError):
                model(x, ei, ids)
        for build in (lambda: RGCNConv(2, 3, 2, num_blocks=1), lambda: RGCNConv(2, 3, 2, aggr='max')):
            with self.assertRaises(NotImplementedError):
                build()
        for bases in (None, 1):
            with self.subTest(oversize_bases=bases), self.assertRaisesRegex(ValueError, 'capacity'):
                RGCNConv(2, 3, 1 << 62, num_bases=bases)
        with self.assertRaises(ValueError):
            model(fr.Tensor.from_i64([0, 1, 2], [3]), ei, rel)


if __name__ == '__main__':
    unittest.main()
