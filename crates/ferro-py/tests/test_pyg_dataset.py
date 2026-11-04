"""Graph dataset collation against PyG and safe, atomic owning persistence."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch
from torch_geometric.data import Data as TorchData, InMemoryDataset as TorchDataset

import ferro as fr
from ferro._graph_data import Data
from ferro._graph_dataset import Dataset, InMemoryDataset


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def index(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype={'float32': torch.float32, 'int64': torch.int64, 'bool': torch.bool}[value.dtype]).reshape(value.shape)


def fixtures(grad=False):
    actual, reference = [], []
    for i, (n, s, d) in enumerate(((3, [0, 1], [1, 2]), (0, [], []), (2, [1], [0]))):
        x = torch.arange(n * 2).reshape(n, 2).float() / 7 + i
        edge = torch.tensor([s, d], dtype=torch.int64)
        attrs = torch.arange(len(s)).reshape(len(s), 1).float() + .3
        mask = [j % 2 == 0 for j in range(n)]
        actual.append(Data(x=tensor(x, grad), edge_index=index(edge), edge_attr=tensor(attrs, grad), y=index([2**55 + i]), train_mask=fr.tensor(mask, dtype='bool'), num_nodes=n))
        reference.append(TorchData(x=x.clone().requires_grad_(grad), edge_index=edge, edge_attr=attrs.clone().requires_grad_(grad), y=torch.tensor([2**55 + i]), train_mask=torch.tensor(mask, dtype=torch.bool), num_nodes=n))
    return actual, reference


class GraphDatasetTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected, atol=2e-6, rtol=2e-5)

    def test_local_topology_collation_slices_and_roundtrip_match_pyg(self):
        source, reference = fixtures()
        dataset = InMemoryDataset(data_list=source)
        expected, slices = TorchDataset.collate(reference)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(set(dataset.slices), set(slices))
        for key in dataset.data.keys():
            self.close(dataset.data[key], expected[key])
            self.close(dataset.slices[key], slices[key])
        self.assertEqual(dataset.data.edge_index.tolist(), [[0, 1, 1], [1, 2, 0]])
        for i in range(3):
            restored = dataset.get(i)
            self.assertEqual(restored.num_nodes, source[i].num_nodes)
            for key in source[i].keys():
                self.assertEqual(restored[key].tolist(), source[i][key].tolist())
                self.assertEqual(restored[key].dtype, source[i][key].dtype)
            self.assertTrue(restored.validate())
        self.assertEqual(dataset.num_features, 2)

    def test_extracted_fields_retain_original_tensor_gradients(self):
        source, reference = fixtures(True)
        dataset = InMemoryDataset(data_list=source)
        loss = sum((dataset[i].x * (i + 1)).sum() + (dataset[i].edge_attr * dataset[i].edge_attr).sum() for i in range(3))
        expected = sum((item.x * (i + 1)).sum() + item.edge_attr.square().sum() for i, item in enumerate(reference))
        loss.backward(); expected.backward()
        for actual, wanted in zip(source, reference):
            self.close(actual.x.grad, wanted.x.grad)
            self.close(actual.edge_attr.grad, wanted.edge_attr.grad)

    def test_index_views_transform_filter_shuffle_and_scalar_labels(self):
        source, _ = fixtures()
        calls = []
        def transform(data):
            calls.append(data.num_nodes)
            data.x = data.x + 1.
            return data
        dataset = InMemoryDataset(data_list=source, transform=transform)
        view = dataset[[2, 0, 2]]
        self.assertEqual(len(view), 3)
        self.assertEqual([d.y.tolist() for d in (view[0], view[-1])], [[2**55 + 2], [2**55 + 2]])
        self.assertEqual(calls, [2, 2])
        self.assertEqual(source[2].x.tolist(), [[2., 2.142857074737549], [2.2857143878936768, 2.4285714626312256]])
        self.assertEqual(list(dataset[fr.tensor([True, False, True])].indices()), [0, 2])
        self.assertEqual(list(dataset[::-1].indices()), [2, 1, 0])
        shuffled, permutation = dataset.shuffle(return_perm=True, seed=7)
        self.assertEqual(list(shuffled.indices()), permutation.tolist())
        self.assertEqual(permutation.tolist(), dataset.shuffle(True, seed=7)[1].tolist())
        pre = []
        filtered = InMemoryDataset(data_list=source, pre_filter=lambda d: d.num_nodes > 0, pre_transform=lambda d: (pre.append(d.num_nodes) or d))
        self.assertEqual(pre, [3, 2]); self.assertEqual(len(filtered), 2)
        scalar = InMemoryDataset(data_list=[Data(y=index(2), num_nodes=0), Data(y=index(4), num_nodes=0)])
        self.assertEqual(scalar[0].y.shape, [])
        self.assertEqual(scalar.num_classes, 5)

    def test_dataset_base_subclass_and_empty_collections(self):
        source, _ = fixtures()
        class Custom(Dataset):
            def len(self): return len(source)
            def get(self, i): return source[i]
        dataset = Custom(root='/tmp/graphs')
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1:].indices(), (1, 2))
        self.assertEqual(dataset.raw_dir, '/tmp/graphs/raw')
        self.assertEqual(dataset.processed_dir, '/tmp/graphs/processed')
        empty = InMemoryDataset(data_list=[])
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.num_features, 0)
        self.assertEqual(len(empty[fr.tensor([], dtype='bool')]), 0)
        with self.assertRaises(IndexError): empty[0]
        with self.assertRaises(ValueError): dataset[fr.tensor([], dtype='bool')]
        with self.assertRaises(TypeError): dataset[True]
        with self.assertRaises(NotImplementedError): InMemoryDataset(force_reload=True)

    def test_safe_archive_owns_values_and_preserves_dtypes_shapes_and_flags(self):
        source, _ = fixtures(True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graphs.fgd'
            InMemoryDataset.save(source, path)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {'manifest.json', 'tensors.safetensors'})
                self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))
            loaded = InMemoryDataset()
            loaded.load(path)
            self.assertEqual(len(loaded), 3)
            for i, original in enumerate(source):
                restored = loaded[i]
                for key in original.keys():
                    self.assertEqual(restored[key].tolist(), original[key].tolist())
                    self.assertEqual(restored[key].dtype, original[key].dtype)
                    self.assertEqual(restored[key].requires_grad, original[key].requires_grad)
            saved = loaded[0].x.tolist()
            with fr.no_grad(): source[0].x.copy_(fr.Tensor.zeros(source[0].x.shape))
            self.assertEqual(loaded[0].x.tolist(), saved)
            loaded[0].x.sum().backward()
            self.assertIsNone(source[0].x.grad)
            empty_path = Path(directory) / 'empty.fgd'
            InMemoryDataset.save([], empty_path)
            loaded.load(empty_path)
            self.assertEqual(len(loaded), 0)

    def test_corrupt_archive_load_is_atomic_and_rejects_extra_members(self):
        source, _ = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'valid.fgd'
            InMemoryDataset.save(source, path)
            with zipfile.ZipFile(path) as archive:
                original_manifest = json.loads(archive.read('manifest.json'))
                payload = archive.read('tensors.safetensors')
            target = InMemoryDataset(data_list=source)[[2, 0]]
            before = [target[i].y.tolist() for i in range(len(target))]
            for change in ('dtype', 'shape', 'role', 'hash', 'extra', 'requires_grad', 'duplicate'):
                manifest = json.loads(json.dumps(original_manifest))
                field = manifest['graphs'][2]['fields'][0]
                if change == 'dtype': field['dtype'] = 'int64'
                elif change == 'shape': field['shape'] = [1, 4]
                elif change == 'role': field['role'] = 'invalid'
                elif change == 'hash': manifest['sha256'] = '0' * 64
                elif change == 'requires_grad': field['requires_grad'] = 1
                elif change == 'duplicate': manifest['graphs'][2]['fields'].append(field)
                bad = Path(directory) / (change + '.fgd')
                with zipfile.ZipFile(bad, 'w') as archive:
                    archive.writestr('manifest.json', json.dumps(manifest))
                    archive.writestr('tensors.safetensors', payload)
                    if change == 'extra': archive.writestr('../unexpected.py', 'raise RuntimeError()')
                with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                    target.load(bad)
                self.assertEqual([target[i].y.tolist() for i in range(len(target))], before)
                self.assertEqual(list(target.indices()), [2, 0])
            with self.assertRaises(ValueError): target.load(path, max_bytes=1)
            self.assertFalse((Path(directory).parent / 'unexpected.py').exists())

    def test_mixed_gradient_flags_survive_collation_and_persistence(self):
        source, _ = fixtures(True)
        source[1].x = source[1].x.detach()
        source[2].x = source[2].x.detach()
        source[0].edge_attr = source[0].edge_attr.detach()
        dataset = InMemoryDataset(data_list=source)
        for i, original in enumerate(source):
            for key in ('x', 'edge_attr'):
                self.assertEqual(dataset[i][key].requires_grad, original[key].requires_grad)
        dataset[0].x.sum().backward()
        self.close(source[0].x.grad, torch.ones(source[0].x.shape))
        self.assertIsNone(source[2].x.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.fgd'
            InMemoryDataset.save(source, path)
            loaded = InMemoryDataset()
            loaded.load(path)
            for i, original in enumerate(source):
                for key in ('x', 'edge_attr'):
                    self.assertEqual(loaded[i][key].requires_grad, original[key].requires_grad)
                    self.assertEqual(loaded[i][key].tolist(), original[key].tolist())
            loaded[0].x.sum().backward()
            self.close(source[0].x.grad, torch.ones(source[0].x.shape))

    def test_invalid_collation_and_save_do_not_replace_existing_artifact(self):
        source, _ = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.fgd'
            InMemoryDataset.save(source, path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                InMemoryDataset.save([Data(x=tensor([[1.]])), Data(x=tensor([[1., 2.]]))], path)
            self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            InMemoryDataset(data_list=[Data(x=tensor([[1.]])), Data(y=index([0]))])
        with self.assertRaises(ValueError):
            InMemoryDataset(data_list=[Data(x=tensor([[1.]]))])[fr.tensor([True, False])]


if __name__ == '__main__':
    unittest.main()
