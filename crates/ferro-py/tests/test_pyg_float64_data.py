"""Float64 graph data pipelines, distinct from float32 graph convolution kernels."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch as TorchBatch, Data as TorchData, HeteroData as TorchHeteroData

import ferro as fr
from ferro._graph_data import Batch, Data, DataLoader
from ferro._graph_dataset import InMemoryDataset
from ferro._graph_hetero import HeteroData
from ferro._graph_hetero_batch import HeteroBatch
from ferro._graph_sampling import NeighborLoader, LinkNeighborLoader
from ferro._graph_cluster import ClusterData, ClusterLoader
from ferro._graph_link import RandomLinkSplit
from ferro._graph_saint import GraphSAINTRandomWalkSampler


def native(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float64)
    return fr.Tensor.from_f64(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def fixture(nodes=4, grad=False):
    values = torch.arange(nodes * 2).double().reshape(2, nodes) / 7 + 1e-11
    edges = torch.tensor([[0, 1, 1, 2], [1, 2, 2, 0]]) if nodes else torch.empty(2, 0, dtype=torch.int64)
    attrs = torch.arange(edges.shape[1]).double().reshape(-1, 1) / 13 + 1e-12
    data = Data(x=native(values, grad).transpose(0, 1), edge_index=index(edges), edge_attr=native(attrs, grad),
                y=native([1e-300, 1.000000000000001], grad), num_nodes=nodes)
    reference = TorchData(x=values.T.detach().requires_grad_(grad), edge_index=edges, edge_attr=attrs.clone().requires_grad_(grad),
                          y=torch.tensor([1e-300, 1.000000000000001], dtype=torch.float64, requires_grad=grad), num_nodes=nodes)
    return data, reference


class Float64GraphTests(unittest.TestCase):
    def close(self, value, expected):
        self.assertEqual(value.dtype, 'float64')
        torch.testing.assert_close(torch.tensor(value.tolist(), dtype=torch.float64).reshape(value.shape), expected.detach(), atol=1e-13, rtol=1e-12)

    def test_homogeneous_ragged_batch_matches_actual_pyg_and_gradients(self):
        pairs = [fixture(n, True) for n in (4, 0, 5)]
        data, reference = zip(*pairs)
        batch = Batch.from_data_list(data, follow_batch=['x', 'edge_attr'])
        expected = TorchBatch.from_data_list(reference, follow_batch=['x', 'edge_attr'])
        for key in ('x', 'edge_attr', 'y'):
            self.close(batch[key], expected[key])
        for key in ('edge_index', 'batch', 'ptr', 'x_batch', 'x_ptr', 'edge_attr_ptr'):
            self.assertEqual(batch[key].tolist(), expected[key].tolist())
        loss, ref_loss = native(0.), torch.tensor(0., dtype=torch.float64)
        for i, (part, wanted) in enumerate(zip(batch.to_data_list(), expected.to_data_list())):
            loss = loss + (part.x * part.x).sum() * (i + 1) + part.edge_attr.sum() + part.y.sum()
            ref_loss = ref_loss + wanted.x.square().sum() * (i + 1) + wanted.edge_attr.sum() + wanted.y.sum()
        loss.backward()
        ref_loss.backward()
        for actual, wanted in pairs:
            for key in ('x', 'edge_attr', 'y'):
                self.close(actual[key].grad, wanted[key].grad)
        loaded = list(DataLoader(data, batch_size=2))
        self.assertEqual([value.x.dtype for value in loaded], ['float64', 'float64'])

    def test_heterogeneous_batch_clone_and_sampler_keep_float64_history(self):
        sources, references = [], []
        relation = ('a', 'to', 'b')
        for size in (3, 0, 4):
            data, expected = HeteroData(), TorchHeteroData()
            for name, count in (('a', size), ('b', 2)):
                raw = torch.arange(count * 2).double().reshape(count, 2) / 11 + 1e-10
                data[name].x, expected[name].x = native(raw, True), raw.requires_grad_(True)
                data[name].num_nodes = expected[name].num_nodes = count
            edges = [[0, 1, 1], [0, 1, 1]] if size else [[], []]
            data[relation].edge_index, expected[relation].edge_index = index(edges), torch.tensor(edges, dtype=torch.int64)
            raw = torch.arange(len(edges[0])).double().reshape(-1, 1) / 3 + 1e-11
            data[relation].edge_attr, expected[relation].edge_attr = native(raw, True), raw.requires_grad_(True)
            sources.append(data)
            references.append(expected)
        batch, expected = Batch.from_data_list(sources), TorchBatch.from_data_list(references)
        self.assertIsInstance(batch, HeteroBatch)
        clone = batch.clone()
        loss, ref_loss = native(0.), torch.tensor(0., dtype=torch.float64)
        for name in ('a', 'b'):
            self.close(clone[name].x, expected[name].x)
            loss = loss + (clone[name].x * clone[name].x).sum()
            ref_loss = ref_loss + expected[name].x.square().sum()
        self.close(clone[relation].edge_attr, expected[relation].edge_attr)
        loss = loss + clone[relation].edge_attr.sum()
        ref_loss = ref_loss + expected[relation].edge_attr.sum()
        loss.backward()
        ref_loss.backward()
        for data, ref in zip(sources, references):
            for name in ('a', 'b'):
                self.close(data[name].x.grad, ref[name].x.grad)
            self.close(data[relation].edge_attr.grad, ref[relation].edge_attr.grad)
        sample = next(iter(NeighborLoader(sources[0], [-1], input_nodes=('b', [1]), seed=3)))
        self.close(sample['a'].x, references[0]['a'].x[sample['a'].n_id.tolist()])
        self.close(sample[relation].edge_attr, references[0][relation].edge_attr[sample[relation].e_id.tolist()])

    def test_clone_keeps_sub_float32_values_signed_zero_and_independent_storage(self):
        data = HeteroData()
        data['a'].x = native([[-0., 1e-300, 1.000000000000001]], True)
        clone = data.clone()
        self.assertEqual(clone['a'].x.tolist(), data['a'].x.tolist())
        self.assertEqual(math.copysign(1., clone['a'].x.tolist()[0][0]), -1.)
        clone['a'].x.sum().backward()
        self.assertEqual(data['a'].x.grad.tolist(), [[1., 1., 1.]])
        with fr.no_grad():
            clone['a'].x.detach().copy_(native([[2., 3., 4.]]))
        self.assertEqual(data['a'].x.tolist()[0][1], 1e-300)

    def test_dataset_archive_preserves_dtype_precision_and_original_grad_flags(self):
        data = [fixture(4, True)[0], fixture(0, False)[0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'float64.zip'
            InMemoryDataset.save(data, path)
            restored = InMemoryDataset()
            restored.load(path)
            for original, current in zip(data, restored):
                for key in ('x', 'edge_attr', 'y'):
                    self.assertEqual(current[key].dtype, 'float64')
                    self.assertEqual(current[key].tolist(), original[key].tolist())
                    self.assertEqual(current[key].requires_grad, original[key].requires_grad)
            sample = restored[0]
            sample.x.sum().backward()
            self.assertEqual(sample.x.grad.tolist(), [[1., 1.]] * 4)
            self.assertIsNone(data[0].x.grad)  # Archive owns fresh leaves.

    def test_sampling_cluster_and_saint_fields_keep_values_and_gradients(self):
        factories = [lambda data: NeighborLoader(data, [-1, -1], input_nodes=[0, 2], batch_size=2),
                     lambda data: ClusterLoader(ClusterData(data, 2), batch_size=1),
                     lambda data: GraphSAINTRandomWalkSampler(data, 3, 2, num_steps=2, seed=7)]
        for factory in factories:
            data, reference = fixture(4, True)
            # Cluster has no generated IDs: retain explicitly aligned identity fields.
            if factory is factories[1]:
                data.ids = index(range(4)); data.edge_ids = index(range(4))
                data._field_roles.update(ids='node', edge_ids='edge')
            loss, expected = native(0.), torch.tensor(0., dtype=torch.float64)
            for batch in factory(data):
                nodes = batch.ids.tolist() if 'ids' in batch else batch.n_id.tolist()
                edges = batch.edge_ids.tolist() if 'edge_ids' in batch else batch.e_id.tolist()
                self.close(batch.x, reference.x[nodes])
                self.close(batch.edge_attr, reference.edge_attr[edges])
                loss = loss + (batch.x * batch.x).sum() + batch.edge_attr.sum()
                expected = expected + reference.x[nodes].square().sum() + reference.edge_attr[edges].sum()
            loss.backward(); expected.backward()
            self.close(data.x.grad, reference.x.grad)
            self.close(data.edge_attr.grad, reference.edge_attr.grad)

    def test_link_float64_labels_keep_live_history_and_negative_dtype(self):
        data, _ = fixture(4, True)
        labels = native([0., 1.000000000000001, .7, .8], True)
        batch = next(iter(LinkNeighborLoader(data, [-1], edge_label=labels, batch_size=4, neg_sampling_ratio=.5, seed=7)))
        self.assertEqual(batch.edge_label.dtype, 'float64')
        self.assertEqual(batch.edge_label.tolist()[:4], [1., 2.000000000000001, 1.7, 1.8])
        self.assertEqual(batch.edge_label.tolist()[4:], [0., 0.])
        batch.edge_label.sum().backward()
        self.assertEqual(labels.grad.tolist(), [1., 1., 1., 1.])
        # Split regression labels remain floating only when no negatives are added.
        data.edge_label = native([.1, .2, .2, .3], True)
        data._field_roles['edge_label'] = 'edge'
        parts = RandomLinkSplit(num_val=0, num_test=0, neg_sampling_ratio=0., seed=2)(data)
        self.assertEqual([part.edge_label.dtype for part in parts], ['float64'] * 3)
        parts[0].edge_label.sum().backward()
        self.assertEqual(data.edge_label.grad.tolist(), [1., 1., 0., 1.])

    def test_float64_capacity_preflight_and_conv_boundary(self):
        huge = native([1.]).expand(1, sys.maxsize // 8 + 1)
        data = Data(num_nodes=0, payload=huge, field_roles={'payload': 'graph'})
        with self.assertRaisesRegex(ValueError, 'graph output allocation capacity'):
            Batch.from_data_list([data])
        hetero = HeteroData()
        hetero['a'].num_nodes = 0
        hetero.payload = huge
        with self.assertRaisesRegex(ValueError, 'graph output allocation capacity'):
            HeteroBatch.from_data_list([hetero])
        data, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'float32'):
            fr.graph.GCNConv(2, 2)(data.x, data.edge_index)


if __name__ == '__main__':
    unittest.main()
