"""Actual PyG typed graph conversion, COO mapping and native-gradient checks."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData, HeteroData as TorchHetero
import ferro as fr
from ferro._graph_data import Data, Batch
from ferro._graph_hetero import HeteroData
from ferro._graph_hetero_batch import HeteroBatch


def native(value, grad=False):
    raw, shape = value.detach().reshape(-1).tolist(), list(value.shape)
    factory = {'float64': fr.Tensor.from_f64, 'int64': fr.Tensor.from_i64, 'bool': fr.Tensor.from_bool}.get(str(value.dtype).split('.')[-1], fr.Tensor)
    return factory(raw, shape).requires_grad_(grad)


def fixture(dtype=torch.float64, gradients=False):
    actual, expected, leaves = HeteroData(), TorchHetero(), []
    for name, count in (('a', 3), ('b', 2), ('empty', 0)):
        raw = (torch.arange(2 * count, dtype=dtype).reshape(2, count) / 7 + .3).requires_grad_(gradients)
        leaf = native(raw, gradients)
        actual[name].x, expected[name].x = leaf.transpose(0, 1), raw.T
        leaves.append((leaf, raw))
        actual[name].num_nodes = expected[name].num_nodes = count
        for key, value in (('label', torch.arange(count, dtype=torch.int64) + 2**55),
                           ('mask', torch.arange(count) % 2 == 0)):
            actual[name][key], expected[name][key] = native(value), value
    for relation, index in ((('a', 'to', 'b'), [[0, 0, 2], [1, 1, 0]]),
                            (('b', 'back', 'a'), [[0, 1], [2, 0]]),
                            (('empty', 'none', 'empty'), [[], []])):
        index = torch.tensor(index, dtype=torch.int64)
        actual[relation].edge_index, expected[relation].edge_index = native(index), index
        raw = (torch.arange(index.shape[1] * 2, dtype=dtype).reshape(-1, 2) / 13 + .2).requires_grad_(gradients)
        leaf = native(raw, gradients)
        actual[relation].edge_attr, expected[relation].edge_attr = leaf, raw
        leaves.append((leaf, raw))
    raw = torch.tensor([.25, -.5], dtype=dtype, requires_grad=gradients)
    actual.context, expected.context = native(raw, gradients), raw
    leaves.append((actual.context, raw))
    return actual, expected, leaves


class GraphConversionTests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        self.assertEqual(actual.shape, list(expected.shape))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=2e-7, rtol=2e-6, equal_nan=True)

    def homogeneous(self, actual, expected):
        self.assertTrue(actual.validate())
        self.assertEqual(actual.num_nodes, expected.num_nodes)
        self.assertEqual(actual.num_edges, expected.num_edges)
        self.assertEqual(set(actual.keys()), set(expected.keys()) - {'num_nodes'})
        for key in actual.keys():
            self.close(actual[key], expected[key])

    def heterogeneous(self, actual, expected):
        self.assertTrue(actual.validate())
        self.assertEqual(actual.metadata(), expected.metadata())
        for key in (*actual.node_types, *actual.edge_types):
            self.assertEqual(set(actual[key].keys()) - {'num_nodes'}, set(expected[key].keys()) - {'num_nodes'})
            if isinstance(key, str):
                self.assertEqual(actual[key].num_nodes, expected[key].num_nodes)
            for field, value in actual[key].items():
                if field != 'num_nodes':
                    self.close(value, expected[key][field])
        self.assertEqual(set(actual._global), set(expected._global_store))
        for key, value in actual._global.items():
            self.close(value, expected[key])

    def test_rectangular_duplicate_edges_typed_fields_and_empty_type_roundtrip(self):
        for dtype in (torch.float32, torch.float64):
            actual, expected, _ = fixture(dtype)
            flat, reference = actual.to_homogeneous(), expected.to_homogeneous()
            self.homogeneous(flat, reference)
            self.assertEqual(flat.node_type.tolist(), [0, 0, 0, 1, 1])
            self.assertEqual(flat.edge_type.tolist(), [0, 0, 0, 1, 1])
            self.assertEqual(flat.edge_index.tolist(), [[0, 0, 2, 3, 4], [4, 4, 3, 2, 0]])
            self.heterogeneous(flat.to_heterogeneous(), reference.to_heterogeneous())
            self.assertEqual(actual['a'].label.tolist()[0], 2**55)

    def test_concat_and_inverse_selection_preserve_every_live_gradient(self):
        for dtype in (torch.float32, torch.float64):
            actual, expected, leaves = fixture(dtype, True)
            flat, reference = actual.to_homogeneous(), expected.to_homogeneous()
            restored, torch_restored = flat.to_heterogeneous(), reference.to_heterogeneous()
            loss, torch_loss = flat.context.square().sum(), reference.context.square().sum()
            for number, key in enumerate((*restored.node_types, *restored.edge_types), 1):
                field = 'x' if isinstance(key, str) else 'edge_attr'
                loss = loss + (restored[key][field] * number).square().sum()
                torch_loss = torch_loss + (torch_restored[key][field] * number).square().sum()
            loss.backward(); torch_loss.backward()
            for left, right in leaves:
                self.close(left.grad, right.grad)

    def test_node_width_padding_dummy_values_and_source_nonmutation(self):
        actual, expected = HeteroData(), TorchHetero()
        for key, rows, width in (('a', 2, 2), ('b', 1, 3), ('empty', 0, 1)):
            value = torch.arange(rows * width, dtype=torch.float64).reshape(rows, width).requires_grad_(True)
            actual[key].x, expected[key].x = native(value, True), value
            actual[key].num_nodes = expected[key].num_nodes = rows
        for field, value in (('score', torch.tensor([[.2], [.7]], dtype=torch.float64)),
                             ('label', torch.tensor([2**55, 2**55 + 1])),
                             ('mask', torch.tensor([True, False]))):
            actual['a'][field], expected['a'][field] = native(value), value
        before = {key: set(actual[key].keys()) for key in actual.node_types}
        flat, reference = actual.to_homogeneous(), expected.to_homogeneous()
        self.homogeneous(flat, reference)
        self.assertEqual(flat.x.shape, [3, 3])
        self.assertEqual(flat.label.tolist(), [2**55, 2**55 + 1, -1])
        self.assertEqual(flat.mask.tolist(), [True, False, False])
        self.assertEqual(before, {key: set(actual[key].keys()) for key in actual.node_types})
        self.heterogeneous(flat.to_heterogeneous(), reference.to_heterogeneous())
        flat.x.square().sum().backward(); reference.x.square().sum().backward()
        for key in actual.node_types:
            self.close(actual[key].x.grad, expected[key].x.grad)

    def test_explicit_attributes_missing_policy_and_incompatible_shapes(self):
        actual, expected, _ = fixture()
        del actual['b'].mask; del expected['b'].mask
        actual['a'].only = native(torch.ones(3, 1)); expected['a'].only = torch.ones(3, 1)
        options = dict(node_attrs=['x', 'label'], edge_attrs=[], dummy_values=False)
        self.homogeneous(actual.to_homogeneous(**options), expected.to_homogeneous(**options))
        self.homogeneous(actual.to_homogeneous(dummy_values=False), expected.to_homogeneous(dummy_values=False))
        with self.assertRaises(ValueError): actual.to_homogeneous(node_attrs=['mask'], dummy_values=False)
        with self.assertRaises(ValueError): actual.to_homogeneous(node_attrs=['unknown'])
        actual['a'].scalar = native(torch.tensor(2.))
        with self.assertRaises(NotImplementedError): actual.to_homogeneous()
        self.assertNotIn('scalar', actual.to_homogeneous(node_attrs=['x']))

    def test_shuffled_type_ids_custom_roles_and_inferred_names(self):
        nodes = torch.tensor([1, 0, 1, 0, 2])
        index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        kinds = torch.tensor([0, 1, 0, 1])
        features = torch.arange(10).reshape(5, 2).double().requires_grad_(True)
        attrs = torch.arange(4).double().requires_grad_(True)
        actual = Data(x=native(features, True), edge_index=native(index), node_type=native(nodes),
                      edge_type=native(kinds), score=native(attrs, True), field_roles={'score': 'edge', 'edge_type': 'edge'})
        expected = TorchData(x=features, edge_index=index, node_type=nodes, edge_type=kinds, score=attrs)
        self.heterogeneous(actual.to_heterogeneous(), expected.to_heterogeneous())
        options = dict(node_type_names=['a', 'b', 'isolate'], edge_type_names=[('b', 'to', 'a'), ('a', 'back', 'b')])
        restored, reference = actual.to_heterogeneous(**options), expected.to_heterogeneous(**options)
        self.heterogeneous(restored, reference)
        self.assertEqual(restored['a'].x.tolist(), [[2., 3.], [6., 7.]])
        self.assertEqual(restored['b', 'to', 'a'].edge_index.tolist(), [[0, 1], [0, 1]])
        restored['a'].x.square().sum().backward(); reference['a'].x.square().sum().backward()
        self.close(actual.x.grad, features.grad)

    def test_empty_graphs_zero_width_fields_and_explicit_empty_type_ids(self):
        for counts in ((0, 0), (2, 0), (0, 3)):
            actual, expected = HeteroData(), TorchHetero()
            for key, count in zip(('a', 'b'), counts):
                value = torch.empty(count, 0)
                actual[key].x, expected[key].x = native(value), value
            for relation in (('a', 'to', 'b'), ('b', 'back', 'a')):
                index = torch.empty(2, 0, dtype=torch.int64)
                actual[relation].edge_index, expected[relation].edge_index = native(index), index
            flat, reference = actual.to_homogeneous(), expected.to_homogeneous()
            self.homogeneous(flat, reference)
            self.heterogeneous(flat.to_heterogeneous(), reference.to_heterogeneous())
        empty = HeteroData().to_homogeneous()
        self.assertEqual(empty.num_nodes, 0)
        self.assertEqual(empty.to_heterogeneous().metadata(), ([], []))
        data = Data(num_nodes=2, node_type=fr.tensor([0, 2], dtype='int64'))
        restored = data.to_heterogeneous(node_type_names=['a', 'empty', 'b'], edge_type_names=[])
        self.assertEqual([restored[key].num_nodes for key in restored.node_types], [1, 0, 1])

    def test_optional_type_vectors_require_explicit_reconstruction_metadata(self):
        actual, expected, _ = fixture()
        options = dict(add_node_type=False, add_edge_type=False)
        flat, reference = actual.to_homogeneous(**options), expected.to_homogeneous(**options)
        self.homogeneous(flat, reference)
        with self.assertRaises(ValueError): flat.to_heterogeneous()
        nodes = fr.tensor([0, 0, 0, 1, 1], dtype='int64')
        edges = fr.tensor([0, 0, 0, 1, 1], dtype='int64')
        restored = flat.to_heterogeneous(node_type=nodes, edge_type=edges)
        self.heterogeneous(restored, expected.to_homogeneous().to_heterogeneous())

    def test_single_type_node_copy_edge_alias_and_default_homogeneous_split(self):
        actual, expected = HeteroData(), TorchHetero()
        actual['a'].x, expected['a'].x = fr.tensor([[1.], [2.]], dtype='float64'), torch.tensor([[1.], [2.]], dtype=torch.float64)
        index, value = torch.tensor([[0], [1]]), torch.tensor([[.5]], dtype=torch.float64)
        actual['a', 'to', 'a'].edge_index, expected['a', 'to', 'a'].edge_index = native(index), index
        actual['a', 'to', 'a'].edge_attr, expected['a', 'to', 'a'].edge_attr = native(value), value
        flat, reference = actual.to_homogeneous(), expected.to_homogeneous()
        with fr.no_grad(), torch.no_grad():
            flat.x.copy_(fr.ones([2, 1], dtype='float64') * 7)
            reference.x.fill_(7)
            flat.edge_attr.copy_(fr.ones([1, 1], dtype='float64') * 9)
            reference.edge_attr.fill_(9)
        self.close(actual['a'].x, expected['a'].x)
        self.assertEqual(actual['a'].x.tolist(), [[1.], [2.]])
        self.close(actual['a', 'to', 'a'].edge_attr, expected['a', 'to', 'a'].edge_attr)
        self.assertEqual(actual['a', 'to', 'a'].edge_attr.tolist(), [[9.]])
        plain = Data(x=fr.ones([2, 1]), edge_index=native(index))
        torch_plain = TorchData(x=torch.ones(2, 1), edge_index=index)
        self.heterogeneous(plain.to_heterogeneous(), torch_plain.to_heterogeneous())

    def test_invalid_type_vectors_relations_collisions_and_unsupported_fields(self):
        actual, _, _ = fixture()
        flat = actual.to_homogeneous()
        for options in (dict(node_type=fr.ones(5)), dict(node_type=fr.tensor([0], dtype='int64')),
                        dict(node_type=fr.tensor([0, 0, -1, 1, 1], dtype='int64')),
                        dict(node_type_names=['a', 'a']), dict(edge_type_names=[('b', 'wrong', 'a')] * 3),
                        dict(edge_type_names=[('a', 'to', 'missing')]), dict(edge_type=fr.tensor([0, 0, 0, 0, 0], dtype='int64'))):
            with self.assertRaises((TypeError, ValueError)): flat.to_heterogeneous(**options)
        with self.assertRaises(TypeError): actual.to_homogeneous(add_node_type=1)
        actual.x = fr.ones(1)
        with self.assertRaises(ValueError): actual.to_homogeneous()
        del actual.x
        actual['a'].x = actual['a'].x.float()
        with self.assertRaises(ValueError): actual.to_homogeneous()
        graph = Data(num_nodes=2, edge_index=fr.tensor([[0], [1]], dtype='int64'), edge_label_index=fr.tensor([[0], [1]], dtype='int64'))
        with self.assertRaises(NotImplementedError): graph.to_heterogeneous()
        graph = Data(num_nodes=2, edge_index=fr.tensor([[0], [1]], dtype='int64'))
        with self.assertRaises(NotImplementedError): Batch.from_data_list([graph]).to_heterogeneous()
        source, _, _ = fixture()
        with self.assertRaises(NotImplementedError): HeteroBatch.from_data_list([source]).to_homogeneous()

    def test_capacity_checks_precede_materialization_or_metadata_lists(self):
        huge = sys.maxsize // 8 + 1
        source = HeteroData()
        source['a'].num_nodes = huge
        with patch('ferro._graph_convert.cat', side_effect=AssertionError('allocation before validation')):
            with self.assertRaises(ValueError): source.to_homogeneous()
            with self.assertRaises(ValueError): Data(num_nodes=huge).to_heterogeneous()
            source['a'].num_nodes = 0
            source['a'].x = fr.zeros([0, 1], dtype='float64').expand(0, huge)
            with self.assertRaises(ValueError): source.to_homogeneous()


if __name__ == '__main__':
    unittest.main(verbosity=2)
