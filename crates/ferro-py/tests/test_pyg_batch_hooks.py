"""Actual PyG dense batching hooks, inverse offsets and native gradients."""
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData, Batch as TorchBatch
from torch_geometric.loader import DataLoader as TorchLoader

import ferro as fr
from ferro.graph import Data, Batch, DataLoader, InMemoryDataset


def tensor(value, dtype='float32', grad=False):
    value = torch.as_tensor(value, dtype=getattr(torch, dtype))
    return fr.tensor(value.reshape(-1).tolist(), dtype=dtype).reshape(list(value.shape)).requires_grad_(grad)


class PairData(Data):
    def __inc__(self, key, value, *args):
        if key == 'edge_index_s': return self.x_s.shape[0]
        if key == 'edge_index_t': return self.x_t.shape[0]
        return super().__inc__(key, value, *args)


class TorchPair(TorchData):
    def __inc__(self, key, value, *args):
        if key == 'edge_index_s': return self.x_s.shape[0]
        if key == 'edge_index_t': return self.x_t.shape[0]
        return super().__inc__(key, value, *args)


class BipartiteData(Data):
    def __inc__(self, key, value, *args):
        if key == 'edge_index':
            return tensor([[self.x_s.shape[0]], [self.x_t.shape[0]]], 'int64')
        return super().__inc__(key, value, *args)


class TorchBipartite(TorchData):
    def __inc__(self, key, value, *args):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.shape[0]], [self.x_t.shape[0]]])
        return super().__inc__(key, value, *args)


class AxesData(Data):
    def __cat_dim__(self, key, value, *args):
        if key == 'columns': return -1
        if key == 'graph_vector': return None
        return super().__cat_dim__(key, value, *args)

    def __inc__(self, key, value, *args):
        if key == 'columns': return tensor([[2], [3]], 'int64')
        return super().__inc__(key, value, *args)


class TorchAxes(TorchData):
    def __cat_dim__(self, key, value, *args):
        if key == 'columns': return -1
        if key == 'graph_vector': return None
        return super().__cat_dim__(key, value, *args)

    def __inc__(self, key, value, *args):
        if key == 'columns': return torch.tensor([[2], [3]])
        return super().__inc__(key, value, *args)


def pairs(dtype='float32', bipartite=False):
    actual, reference = [], []
    for number, (s, t) in enumerate(((3, 2), (0, 0), (2, 4))):
        xs = torch.arange(2 * s, dtype=getattr(torch, dtype)).reshape(2, s).T / 7 + number
        xt = torch.arange(2 * t, dtype=getattr(torch, dtype)).reshape(2, t).T / 9 - number
        if bipartite:
            edges = torch.tensor([[i % s for i in range(s + t)], [i % t for i in range(s + t)]], dtype=torch.int64).reshape(2, -1)
            fields = dict(edge_index=edges)
        else:
            fields = dict(edge_index_s=torch.tensor([[i for i in range(s - 1)], [i + 1 for i in range(s - 1)]], dtype=torch.int64),
                          edge_index_t=torch.tensor([[i for i in range(t - 1)], [i + 1 for i in range(t - 1)]], dtype=torch.int64))
        a = (BipartiteData if bipartite else PairData)(x_s=tensor(xs.T, dtype, True).transpose(0, 1),
                x_t=tensor(xt.T, dtype, True).transpose(0, 1), **{key: tensor(value, 'int64') for key, value in fields.items()})
        b = (TorchBipartite if bipartite else TorchPair)(x_s=xs.detach().requires_grad_(), x_t=xt.detach().requires_grad_(), **fields)
        actual.append(a)
        reference.append(b)
    return actual, reference


class BatchHooksTests(unittest.TestCase):
    def close(self, actual, expected):
        value = torch.tensor(actual.tolist(), dtype=getattr(torch, actual.dtype)).reshape(actual.shape)
        torch.testing.assert_close(value, expected.detach(), atol=2e-6 if actual.dtype == 'float32' else 2e-12, rtol=2e-6)

    def fields(self, actual, expected):
        self.assertEqual(set(actual.keys()), set(expected.keys()) - {'num_nodes'})
        for key in actual.keys(): self.close(actual[key], expected[key])

    def test_pair_independent_spaces_follow_exclusion_roundtrip_and_gradients(self):
        for dtype in ('float32', 'float64'):
            actual, reference = pairs(dtype)
            follow = ['x_s', 'x_t', 'edge_index_s', 'edge_index_t']
            batch = Batch.from_data_list(actual, follow_batch=follow)
            oracle = TorchBatch.from_data_list(reference, follow_batch=follow)
            self.fields(batch, oracle)
            self.assertIsInstance(batch, PairData)
            self.assertNotIn('batch', batch)
            self.assertEqual(batch.x_s_ptr.tolist(), [0, 3, 3, 5])
            self.assertEqual(batch.x_t_ptr.tolist(), [0, 2, 2, 6])
            src, dst = batch.edge_index_s.tolist()
            loss = (batch.x_s.index_select(0, src) * batch.x_s.index_select(0, dst)).sum() + batch.x_t.square().sum()
            expected = (oracle.x_s[oracle.edge_index_s[0]] * oracle.x_s[oracle.edge_index_s[1]]).sum() + oracle.x_t.square().sum()
            loss.backward(); expected.backward()
            for a, b in zip(actual, reference):
                self.close(a.x_s.grad, b.x_s.grad); self.close(a.x_t.grad, b.x_t.grad)
            for a, b in zip(batch.to_data_list(), oracle.to_data_list()):
                self.assertIsInstance(a, PairData)
                self.fields(a, b)
            batch = Batch.from_data_list(actual, follow_batch=follow, exclude_keys=['x_t', 'edge_index_t'])
            oracle = TorchBatch.from_data_list(reference, follow_batch=follow, exclude_keys=['x_t', 'edge_index_t'])
            self.fields(batch, oracle)
            for a, b in zip(batch.to_data_list(), oracle.to_data_list()): self.fields(a, b)

    def test_bipartite_tensor_row_offsets_and_current_value_inverse(self):
        actual, reference = pairs(bipartite=True)
        batch = Batch.from_data_list(actual, follow_batch=['edge_index', 'x_s', 'x_t'])
        oracle = TorchBatch.from_data_list(reference, follow_batch=['edge_index', 'x_s', 'x_t'])
        self.fields(batch, oracle)
        self.assertEqual(batch.edge_index[:, -1:].tolist(), oracle.edge_index[:, -1:].tolist())
        batch.x_s = batch.x_s * 2.3
        oracle.x_s = oracle.x_s * 2.3
        batch.edge_index = batch.edge_index + tensor([[1], [2]], 'int64')
        oracle.edge_index = oracle.edge_index + torch.tensor([[1], [2]])
        split, expected = batch.to_data_list(), oracle.to_data_list()
        for a, b in zip(split, expected):
            self.fields(a, b)
            self.assertIsInstance(a, BipartiteData)
        loss = sum((a.x_s.square().sum() + a.x_t.sum() for a in split), fr.tensor(0.))
        wanted = sum(b.x_s.square().sum() + b.x_t.sum() for b in expected)
        loss.backward(); wanted.backward()
        for a, b in zip(actual, reference):
            self.close(a.x_s.grad, b.x_s.grad); self.close(a.x_t.grad, b.x_t.grad)

    def test_custom_axes_stacked_vectors_float_offsets_and_empty_shapes(self):
        for dtype in ('float32', 'float64'):
            actual, reference = [], []
            for index, columns in enumerate((3, 0, 2)):
                raw = torch.arange(columns * 2, dtype=getattr(torch, dtype)).reshape(columns, 2).T / 5
                vector = torch.tensor([index + .1, index - .2], dtype=getattr(torch, dtype))
                actual.append(AxesData(num_nodes=2, columns=tensor(raw.T, dtype, True).transpose(0, 1), graph_vector=tensor(vector, dtype, True),
                                       score=tensor(index + .4, dtype, True), empty=tensor(torch.empty(0, 0), dtype, True)))
                reference.append(TorchAxes(num_nodes=2, columns=raw.detach().requires_grad_(), graph_vector=vector.requires_grad_(),
                                           score=torch.tensor(index + .4, dtype=getattr(torch, dtype), requires_grad=True), empty=torch.empty(0, 0, dtype=getattr(torch, dtype), requires_grad=True)))
            batch = Batch.from_data_list(actual, follow_batch=['columns', 'graph_vector', 'score', 'empty'])
            oracle = TorchBatch.from_data_list(reference, follow_batch=['columns', 'graph_vector', 'score', 'empty'])
            self.fields(batch, oracle)
            self.assertEqual(batch.graph_vector.shape, [3, 2])
            self.assertEqual(batch.columns_ptr.tolist(), [0, 3, 3, 5])
            split, wanted = batch.to_data_list(), oracle.to_data_list()
            for a, b in zip(split, wanted):
                b.score = b.score.reshape([])  # Ferro preserves original scalar rank.
                self.fields(a, b)
            loss = batch.columns.square().sum() + batch.graph_vector.square().sum() + batch.score.sum() + batch.empty.sum()
            expected = oracle.columns.square().sum() + oracle.graph_vector.square().sum() + oracle.score.sum() + oracle.empty.sum()
            loss.backward(); expected.backward()
            for a, b in zip(actual, reference):
                for key in ('columns', 'graph_vector', 'score', 'empty'): self.close(a[key].grad, b[key].grad)

    def test_default_index_substring_face_and_batch_increment_matches_pyg(self):
        actual, reference = [], []
        for nodes in (3, 4):
            fields = dict(face=torch.tensor([[0], [1], [2]]), custom_index=torch.tensor([[0, 1], [1, 2]]),
                          sample_index_value=torch.tensor([0, 2]), group_batch=torch.tensor([0, 2]),
                          exact=torch.tensor([2**60 + 3, -(2**60) + 1]))
            actual.append(Data(num_nodes=nodes, **{key: tensor(value, 'int64') for key, value in fields.items()}))
            reference.append(TorchData(num_nodes=nodes, **fields))
        batch, oracle = Batch.from_data_list(actual), TorchBatch.from_data_list(reference)
        self.fields(batch, oracle)
        for a, b in zip(batch.to_data_list(), oracle.to_data_list()): self.fields(a, b)
        self.assertEqual(batch.exact.tolist()[0], 2**60 + 3)
        face = torch.tensor([[0], [1], [2]])
        actual_face, reference_face = Data(face=tensor(face, 'int64')), TorchData(face=face)
        self.fields(Batch.from_data_list([actual_face, actual_face]), TorchBatch.from_data_list([reference_face, reference_face]))
        with self.assertRaisesRegex(ValueError, 'inferable num_nodes'):
            Batch.from_data_list([Data(custom_index=tensor([[0], [1]], 'int64'))])
        with self.assertRaisesRegex(RuntimeError, "infer 'num_nodes'"):
            TorchBatch.from_data_list([TorchData(custom_index=torch.tensor([[0], [1]]))])
        with self.assertRaisesRegex(ValueError, 'empty batch field'):
            Batch.from_data_list([Data(num_nodes=0, sample_batch=tensor([], 'int64'))])
        with self.assertRaises(RuntimeError):
            TorchBatch.from_data_list([TorchData(num_nodes=0, sample_batch=torch.empty(0, dtype=torch.int64))])

    def test_loader_subclass_required_constructor_and_metadata_ownership(self):
        class Required(Data):
            calls = 0
            def __init__(self, marker, **kwargs):
                type(self).calls += 1
                super().__init__(**kwargs)
                self._marker = marker
            def __cat_dim__(self, key, value, *args): return None if key == 'vector' else super().__cat_dim__(key, value, *args)
        data = [Required('a', num_nodes=1, vector=tensor([.1, .2])), Required('b', num_nodes=2, vector=tensor([.3, .4]))]
        batch = Batch.from_data_list(data)
        data[0]._marker = 'changed'
        first = batch.to_data_list()
        first[0]._field_roles['vector'] = 'node'
        second = batch.to_data_list()
        self.assertEqual(Required.calls, 2)
        self.assertEqual([item._marker for item in second], ['a', 'b'])
        self.assertNotIn('vector', second[0]._field_roles)
        a, b = pairs()
        actual = DataLoader(a, 2, follow_batch=['x_s', 'x_t'])
        expected = TorchLoader(b, batch_size=2, follow_batch=['x_s', 'x_t'])
        for got, wanted in zip(actual, expected): self.fields(got, wanted)

    def test_first_cat_hook_and_each_increment_receive_store_mapping(self):
        seen = []
        class Count(Data):
            def __cat_dim__(self, key, value, store):
                seen.append(('cat', key, store['value'].shape))
                return 0
            def __inc__(self, key, value, store):
                seen.append(('inc', key, store['value'].shape))
                return 1
        batch = Batch.from_data_list([Count(value=tensor([2.])), Count(value=tensor([5., 6.]))])
        self.assertEqual(seen, [('cat', 'value', [1]), ('inc', 'value', [1]), ('inc', 'value', [2])])
        wanted = []
        class TorchCount(TorchData):
            def __cat_dim__(self, key, value, store):
                wanted.append(('cat', key, list(store['value'].shape)))
                return 0
            def __inc__(self, key, value, store):
                wanted.append(('inc', key, list(store['value'].shape)))
                return 1
        oracle = TorchBatch.from_data_list([TorchCount(value=torch.tensor([2.])), TorchCount(value=torch.tensor([5., 6.]))])
        self.assertEqual(seen, wanted)
        self.fields(batch, oracle)
        self.assertEqual(batch.value.tolist(), [2., 6., 7.])
        self.assertEqual([item.value.tolist() for item in batch.to_data_list()], [[2.], [5., 6.]])

    def test_huge_empty_metadata_never_builds_nested_host_lists(self):
        import ferro._graph_data as implementation
        package_path = str(__import__('pathlib').Path(implementation.__file__).parent)
        script = '''
import sys, ferro
ferro.__path__.insert(0, sys.argv[1])
for name in list(sys.modules):
    if name == 'ferro.graph' or name.startswith('ferro._graph'): del sys.modules[name]
from ferro.graph import Data, Batch
class Empty(Data):
    def __inc__(self, key, value, *args):
        return ferro.tensor([], dtype='int64').reshape([1, 0]).expand([10**12, 0])
x = ferro.tensor([], dtype='int64').reshape([1, 0]).expand([10**12, 0])
result = Batch.from_data_list([Empty(value=x), Empty(value=x)])
if result.value.shape != [2 * 10**12, 0]: raise RuntimeError('wrong empty shape')
if any(item.value.shape != [10**12, 0] for item in result.to_data_list()): raise RuntimeError('wrong inverse shape')
print('huge-empty-ok')
'''
        result = subprocess.run([sys.executable, '-O', '-B', '-c', script, package_path], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('huge-empty-ok', result.stdout)

    def test_invalid_hook_metadata_capacity_and_overflow_precede_cat(self):
        class Bad(Data):
            axis, inc = 0, 0
            def __cat_dim__(self, key, value, *args): return self.axis
            def __inc__(self, key, value, *args): return self.inc
        with patch('ferro._graph_batch_hooks.cat', side_effect=AssertionError('must preflight')):
            for axis, inc, exc in (((), 0, NotImplementedError), (True, 0, NotImplementedError), (4, 0, ValueError),
                                   (0, -1, ValueError), (0, .5, TypeError), (0, tensor([1., 2.]), ValueError),
                                   (0, tensor([[1], [2]], 'int64'), ValueError), (0, 2**63, ValueError)):
                Bad.axis, Bad.inc = axis, inc
                with self.assertRaises(exc): Batch.from_data_list([Bad(value=tensor([1., 2.]))])
            Bad.axis, Bad.inc = 0, 1
            with self.assertRaisesRegex(ValueError, 'overflow'):
                Batch.from_data_list([Bad(value=tensor([0], 'int64')), Bad(value=tensor([2**63 - 1], 'int64'))])
            Bad.inc = tensor([0], 'int64').expand([sys.maxsize // 8 + 1])
            with self.assertRaisesRegex(ValueError, 'capacity'):
                Batch.from_data_list([Bad(value=tensor([1.]))])
            Bad.inc = 0
            with self.assertRaisesRegex(ValueError, 'capacity'):
                Batch.from_data_list([Bad(value=fr.tensor(1., dtype='float64').expand([sys.maxsize // 8 + 1]))])
            Bad.inc = 1
            with self.assertRaises(NotImplementedError): Batch.from_data_list([Bad(value=tensor([True], 'bool')), Bad(value=tensor([False], 'bool'))])
            with self.assertRaisesRegex(TypeError, 'one Data subclass'):
                Batch.from_data_list([Bad(value=tensor([1.])), Data(value=tensor([2.]))])

    def test_strict_persistence_conversion_and_existing_validation_boundaries(self):
        actual, _ = pairs()
        with self.assertRaisesRegex(NotImplementedError, 'dataset persistence'):
            InMemoryDataset(data_list=actual)
        with self.assertRaisesRegex(NotImplementedError, 'dataset persistence'):
            InMemoryDataset.collate(actual[:1])
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(NotImplementedError, 'dataset persistence'):
            InMemoryDataset.save(actual, directory + '/custom.fgraph')
        with self.assertRaisesRegex(NotImplementedError, 'custom graph conversion'):
            actual[0].to_heterogeneous()
        with self.assertRaisesRegex(ValueError, 'wrong row count'):
            Batch.from_data_list([AxesData(num_nodes=3, x=tensor([[1.], [2.]]))])
        with self.assertRaisesRegex(ValueError, 'same tensor field names'):
            Batch.from_data_list([AxesData(num_nodes=1, value=tensor([1.])), AxesData(num_nodes=1)])
        bad = AxesData(num_nodes=2, graph_vector=tensor([1., 2.]), graph_vector_batch=tensor([0], 'int64'))
        with self.assertRaisesRegex(ValueError, 'conflict'):
            Batch.from_data_list([bad], follow_batch=['graph_vector'])


if __name__ == '__main__':
    unittest.main()
