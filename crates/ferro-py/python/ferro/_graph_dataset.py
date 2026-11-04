"""Small homogeneous graph datasets with explicit, pickle-free persistence.

Dataset supplies len/get, transformed indexing and subset views. InMemoryDataset
collates local topology without offsets, like PyG's internal storage, and
extracts exact original shapes. Save/load uses a versioned JSON+safetensors ZIP
with an integrity hash. Loading stages and validates every graph before replacing
dataset state. It restores owning leaves, not autograd history or pending grads.

Download registries, automatic raw/processed lifecycle, arbitrary Python objects,
heterogeneous stores and custom batching increments are not implemented.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import zipfile

from ._native import Tensor, EdgeIndex, cat, save_safetensors, load_safetensors
from ._graph_data import Data, Batch, _INDEX_FIELDS, _label_parts, _needs_hook_collation
from ._graph_utils import _count, _cpu_tensor
from ._graph_link import _graph_fields, _rng


def _data_copy(data):
    if isinstance(data, Data) and _needs_hook_collation(data):
        raise NotImplementedError('custom batching hooks/index fields require explicit DataLoader examples; dataset persistence is unsupported')
    roles = _graph_fields(data)
    result = Data(num_nodes=data.num_nodes, field_roles=roles)
    result._store = dict(data.items())
    return result


class Dataset:
    """Subclass len() and get(index); transform is applied on scalar indexing.

    List/slice/int64/Bool selections produce subset views, including repeated
    indices. root/raw/processed paths are conveniences, not automatic downloads.
    """
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, log=True, force_reload=False):
        for value in (transform, pre_transform, pre_filter):
            if value is not None and not callable(value):
                raise TypeError('dataset transforms and filters must be callable or None')
        if type(log) is not bool or type(force_reload) is not bool:
            raise TypeError('log and force_reload must be bool')
        if force_reload:
            raise NotImplementedError('automatic processing/reloading is unsupported; use explicit load/save')
        self.root = None if root is None else os.fsdecode(os.fspath(root))
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.log, self._indices = log, None

    def len(self):
        raise NotImplementedError('subclass Dataset.len')

    def get(self, index):
        raise NotImplementedError('subclass Dataset.get')

    def __len__(self):
        return self.len() if self._indices is None else len(self._indices)

    def indices(self):
        return range(self.len()) if self._indices is None else tuple(self._indices)

    def _index(self, index):
        if type(index) is not int:
            raise TypeError('graph index must be an integer')
        index = index + len(self) if index < 0 else index
        if not 0 <= index < len(self):
            raise IndexError('graph index is out of bounds')
        return index

    def __getitem__(self, index):
        if isinstance(index, Tensor) and index.dtype == 'int64' and index.shape == []:
            _cpu_tensor(index, 'index', 'int64')
            index = index.item()
        if type(index) is int:
            value = _data_copy(self.get(self.indices()[self._index(index)]))
            return value if self.transform is None else self.transform(value)
        return self.index_select(index)

    def index_select(self, index):
        if isinstance(index, slice):
            selected = list(range(len(self)))[index]
        else:
            is_mask = False
            if isinstance(index, Tensor):
                _cpu_tensor(index, 'index', dtype=None)
                if index.dtype not in ('int64', 'bool') or len(index.shape) != 1:
                    raise ValueError('dataset indices must be int64/Bool vectors')
                is_mask = index.dtype == 'bool'
                index = index.tolist()
            if not isinstance(index, (list, tuple)):
                raise TypeError('dataset selection must be a slice or integer/Bool sequence')
            if is_mask or (index and all(type(i) is bool for i in index)):
                if len(index) != len(self):
                    raise ValueError('dataset Bool mask length mismatch')
                selected = [i for i, keep in enumerate(index) if keep]
            else:
                selected = [self._index(i) for i in index]
        result = copy.copy(self)
        current = self.indices()
        result._indices = [current[i] for i in selected]
        return result

    def shuffle(self, return_perm=False, *, seed=None):
        if type(return_perm) is not bool:
            raise TypeError('return_perm must be bool')
        order = list(range(len(self)))
        _rng(seed).shuffle(order)
        result = self.index_select(order)
        return (result, Tensor.from_i64(order, [len(order)])) if return_perm else result

    @property
    def num_node_features(self):
        return 0 if not len(self) else self[0].num_node_features

    @property
    def num_features(self):
        return self.num_node_features

    @property
    def num_classes(self):
        labels = []
        for i in range(len(self)):
            y = self[i].y
            if y is None:
                continue
            _cpu_tensor(y, 'y', 'int64')
            if len(y.shape) > 1:
                raise NotImplementedError('num_classes supports scalar/vector int64 labels only')
            labels.extend([y.item()] if not y.shape else y.tolist())
        if any(y < 0 for y in labels):
            raise ValueError('num_classes requires nonnegative labels')
        return max(labels, default=-1) + 1

    @property
    def raw_dir(self):
        if self.root is None:
            raise ValueError('dataset has no root directory')
        return str(Path(self.root) / 'raw')

    @property
    def processed_dir(self):
        if self.root is None:
            raise ValueError('dataset has no root directory')
        return str(Path(self.root) / 'processed')


class InMemoryDataset(Dataset):
    """Explicit in-memory Data collection; construct with data_list=[...].

    pre_filter and pre_transform run once for supplied data_list, transform at
    indexed access. save(data_list, path) and load(path) use Ferro's documented
    safe archive, not torch.save/PyG .pt files. Load does not rerun preprocessing.
    Extracted fields preserve each original requires_grad flag, including mixed
    flags within a collated field; this is stricter than PyG's shared cat flag.
    """
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, log=True, force_reload=False, *, data_list=None):
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.data, self.slices = None, None
        if data_list is not None:
            prepared = []
            for source in data_list:
                item = _data_copy(source)
                if self.pre_filter is not None and not self.pre_filter(item):
                    continue
                prepared.append(item if self.pre_transform is None else self.pre_transform(item))
            self.data, self.slices = self.collate(prepared)

    @staticmethod
    def collate(data_list):
        items = list(data_list)
        for data in items:
            if isinstance(data, Data) and _needs_hook_collation(data):
                raise NotImplementedError('custom batching hooks/index fields require explicit DataLoader examples; dataset persistence is unsupported')
            _graph_fields(data)
            for key in _INDEX_FIELDS.intersection(data.keys()):
                _label_parts(data, key)
        if not items:
            result = Data(num_nodes=0)
            result._collated_nodes, result._collated_shapes = [], {}
            return result, None
        if len(items) == 1:
            result = _data_copy(items[0])
            result._collated_nodes = [items[0].num_nodes]
            result._collated_shapes = {k: [v.to_tensor().shape if isinstance(v, EdgeIndex) else v.shape] for k, v in items[0].items()}
            return result, None
        batch = Batch.from_data_list(items)
        result = Data(num_nodes=batch.num_nodes, field_roles=batch._field_roles)
        result._store = {key: batch[key] for key in batch._original_keys}
        slices = {key: Tensor.from_i64(offsets, [len(offsets)]) for key, offsets in batch._slices.items()}
        for key in (key for key in result.keys() if key == 'edge_index' or key in _INDEX_FIELDS):
            edges = [data[key].to_tensor() if isinstance(data[key], EdgeIndex) else data[key] for data in items]
            result._store[key] = cat(edges, 1)
            offsets = [0]
            for edge in edges:
                offsets.append(offsets[-1] + edge.shape[1])
            slices[key] = Tensor.from_i64(offsets, [len(offsets)])
        result._collated_nodes = list(batch._nodes)
        result._collated_shapes = dict(batch._shapes)
        result._collated_grad_flags = {
            key: [False if isinstance(data[key], EdgeIndex) else data[key].requires_grad for data in items]
            for key in result.keys()
        }
        return result, slices

    def len(self):
        return 0 if self.data is None else len(self.data._collated_nodes)

    def get(self, index):
        if type(index) is not int or not 0 <= index < self.len():
            raise IndexError('graph index is out of bounds')
        if self.slices is None:
            return _data_copy(self.data)
        result = Data(num_nodes=self.data._collated_nodes[index], field_roles=self.data._field_roles)
        for key, value in self.data.items():
            start, stop = self.slices[key].tolist()[index:index + 2]
            selected = value.narrow(1 if key == 'edge_index' or key in _INDEX_FIELDS else 0, start, stop - start)
            if key != 'edge_index' and not self.data._collated_shapes[key][index]:
                selected = selected.reshape([])
            if selected.requires_grad and not self.data._collated_grad_flags[key][index]:
                selected = selected.detach()
            result._store[key] = selected
        return result

    @classmethod
    def save(cls, data_list, path):
        """Atomically publish a JSON+safetensors archive with owning tensor values."""
        items = list(data_list)
        cls.collate(items)
        records, tensors = [], {}
        for i, data in enumerate(items):
            roles = _graph_fields(data)
            fields = []
            for j, (name, value) in enumerate(data.items()):
                value = value.to_tensor() if isinstance(value, EdgeIndex) else value
                identifier = f'g{i}.f{j}'
                tensors[identifier] = value
                fields.append(dict(name=name, tensor=identifier, role='topology' if name == 'edge_index' else roles[name], shape=value.shape, dtype=value.dtype, requires_grad=value.requires_grad))
            records.append(dict(num_nodes=data.num_nodes, fields=fields))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                tensor_path = Path(directory) / 'tensors.safetensors'
                save_safetensors(str(tensor_path), tensors)
                payload = tensor_path.read_bytes()
                manifest = dict(format='ferro.graph.dataset', version=1, sha256=hashlib.sha256(payload).hexdigest(), graphs=records)
                encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', delete=False) as handle:
                    temporary = Path(handle.name)
                    with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_STORED) as archive:
                        archive.writestr('manifest.json', encoded)
                        archive.writestr('tensors.safetensors', payload)
                os.replace(temporary, target)
                temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, path, data_cls=Data, *, max_bytes=1 << 30):
        """Validate the whole archive before replacing current data or subsets."""
        if data_cls is not Data:
            raise NotImplementedError('only homogeneous Data archives are supported')
        _count(max_bytes, 'max_bytes')
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 2 or {entry.filename for entry in entries} != {'manifest.json', 'tensors.safetensors'}:
                raise ValueError('dataset archive must contain exactly manifest.json and tensors.safetensors')
            if any(entry.compress_type != zipfile.ZIP_STORED for entry in entries) or sum(entry.file_size for entry in entries) > max_bytes:
                raise ValueError('dataset archive exceeds its byte limit or uses unsupported compression')
            if archive.getinfo('manifest.json').file_size > 16 * 1024 * 1024:
                raise ValueError('dataset manifest exceeds its metadata limit')
            manifest = json.loads(archive.read('manifest.json').decode('ascii'))
            payload = archive.read('tensors.safetensors')
        if type(manifest) is not dict or set(manifest) != {'format', 'version', 'sha256', 'graphs'} or manifest['format'] != 'ferro.graph.dataset' or type(manifest['version']) is not int or manifest['version'] != 1 or type(manifest['graphs']) is not list:
            raise ValueError('invalid dataset manifest schema')
        if manifest['sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError('dataset tensor integrity hash mismatch')
        with tempfile.TemporaryDirectory() as directory:
            tensor_path = Path(directory) / 'tensors.safetensors'
            tensor_path.write_bytes(payload)
            tensors = load_safetensors(str(tensor_path))
        records, referenced = [], set()
        for record in manifest['graphs']:
            if type(record) is not dict or set(record) != {'num_nodes', 'fields'} or type(record['fields']) is not list:
                raise ValueError('invalid graph record')
            node_count = _count(record['num_nodes'], 'num_nodes')
            values, roles = {}, {}
            for field in record['fields']:
                if type(field) is not dict or set(field) != {'name', 'tensor', 'role', 'shape', 'dtype', 'requires_grad'}:
                    raise ValueError('invalid tensor field record')
                name, identifier = field['name'], field['tensor']
                if type(name) is not str or type(identifier) is not str or name in values or identifier in referenced or identifier not in tensors:
                    raise ValueError('duplicate, missing or invalid tensor field')
                value = tensors[identifier]
                if type(field['shape']) is not list or any(type(n) is not int or n < 0 for n in field['shape']) or value.shape != field['shape'] or value.dtype != field['dtype']:
                    raise ValueError('tensor dtype/shape disagrees with dataset manifest')
                if type(field['requires_grad']) is not bool or (field['requires_grad'] and value.dtype not in ('float32', 'float64')):
                    raise ValueError('invalid requires_grad metadata')
                if name == 'edge_index':
                    if field['role'] != 'topology':
                        raise ValueError('edge_index must have topology role')
                elif field['role'] not in ('node', 'edge', 'graph'):
                    raise ValueError('invalid field role')
                else:
                    roles[name] = field['role']
                values[name] = value.requires_grad_(True) if field['requires_grad'] else value
                referenced.add(identifier)
            data = Data(num_nodes=node_count, field_roles=roles, **values)
            data.validate()
            records.append(data)
        if referenced != set(tensors):
            raise ValueError('dataset archive contains unreferenced tensors')
        staged_data, staged_slices = self.collate(records)
        self.data, self.slices, self._indices = staged_data, staged_slices, None
