"""Graph selection from saved dense batch slice and increment descriptions.

Negative indices use normal Python indexing. Boolean masks must contain exactly
num_graphs entries; Boolean sequences are also masks. These are deliberate
extensions/stricter checks over PyG's broken negative and permissive mask paths.
Tensor selectors remain CPU int64/bool. Selection never changes the batch or
returns a cached original graph: each field reads the current native tensor.
"""
import copy

from ._native import Tensor
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, _edge_tensor


def _count(batch):
    if not hasattr(batch, '_num_graphs') or not any(hasattr(batch, key) for key in ('_original_keys', '_hook_plans', '_field_keys')):
        raise ValueError('graph selection requires a batch built by from_data_list')
    return batch.num_graphs


def _one(index, count):
    if isinstance(index, Tensor):
        _cpu_tensor(index, 'graph index', 'int64')
        if index.shape:
            raise IndexError('a single graph index must be scalar')
        index = index.item()
    if type(index) is not int:
        raise IndexError('a single graph index must be an integer')
    index = index + count if index < 0 else index
    if not 0 <= index < count:
        raise IndexError('graph index is out of bounds')
    return index


def _many(index, count):
    if isinstance(index, slice):
        return list(range(count)[index])
    is_mask = False
    if isinstance(index, Tensor):
        _cpu_tensor(index, 'graph selection', dtype=None)
        if index.dtype not in ('int64', 'bool'):
            raise IndexError('graph selectors require int64 or bool tensors')
        _capacity(index.shape, 8 if index.dtype == 'int64' else 1)
        is_mask = index.dtype == 'bool'
        index = index.reshape([-1]).tolist()
    if not isinstance(index, (tuple, list)):
        raise IndexError('graph selectors require a slice, integer sequence or int64/bool tensor')
    if is_mask or (index and all(type(item) is bool for item in index)):
        if len(index) != count:
            raise IndexError('graph Boolean mask must contain num_graphs entries')
        return [i for i, keep in enumerate(index) if keep]
    return [_one(item, count) for item in index]


def _custom(batch, indices):
    from ._graph_batch_hooks import _shift
    output = []
    for number in indices:
        template = batch._hook_templates[number]
        data = copy.copy(template)
        data._store = {}
        data._field_roles = dict(template._field_roles)
        for key, (axis, stacked, slices, increment_shape, offsets, apply, shapes) in batch._hook_plans.items():
            a, b = slices[number:number + 2]
            value = batch[key].narrow(axis, a, b - a)
            if apply and (increment_shape or any(offsets[number])):
                value = _shift(value, increment_shape, offsets[number], subtract=True)
            data._store[key] = value.reshape(list(shapes[number])) if stacked else value
        output.append(data)
    return output


def _homogeneous(batch, indices):
    from ._graph_data import Data, _INDEX_FIELDS
    topology = {key: _edge_parts(batch[key], num_nodes=batch.num_nodes)[:2]
                for key in batch._original_keys if key == 'edge_index' or key in _INDEX_FIELDS}
    nodeptr, edgeptr = [0], [0]
    for n, e in zip(batch._nodes, batch._edges):
        nodeptr.append(nodeptr[-1] + n)
        edgeptr.append(edgeptr[-1] + e)
    output = []
    for number in indices:
        data = Data(num_nodes=batch._nodes[number], field_roles={key: role for key, role in batch._field_roles.items() if key in batch._original_keys})
        for key in batch._original_keys:
            if key in topology:
                src, dst = topology[key]
                a, b = edgeptr[number:number + 2] if key == 'edge_index' else batch._slices[key][number:number + 2]
                offset = nodeptr[number]
                data._store[key] = _edge_tensor([value - offset for value in src[a:b]], [value - offset for value in dst[a:b]])
            else:
                a, b = batch._slices[key][number:number + 2]
                value = batch[key].narrow(0, a, b - a)
                data._store[key] = value if batch._shapes[key][number] else value.reshape([])
        output.append(data)
    return output


def _heterogeneous(batch, indices):
    from ._graph_hetero import HeteroData
    from ._graph_hetero_batch import _mappings
    batch.validate()
    mappings = _mappings(batch)
    topology = {key: _edge_parts(batch[key].edge_index, size=batch[key].size())[:2]
                for key in batch.edge_types if 'edge_index' in batch._field_keys[key]}
    output = []
    for number in indices:
        data = HeteroData()
        for key, counts in batch._node_counts.items():
            data[key].num_nodes = counts[number]
        for key in batch.edge_types:
            data[key]
        fields = _mappings(data)
        for store_key, keys in batch._field_keys.items():
            for key in keys:
                a, b = batch._slices[store_key][key][number:number + 2]
                if key == 'edge_index' and isinstance(store_key, tuple):
                    src, dst = topology[store_key]
                    value = _edge_tensor([i - batch._nodeptr[store_key[0]][number] for i in src[a:b]],
                                         [i - batch._nodeptr[store_key[2]][number] for i in dst[a:b]])
                else:
                    value = mappings[store_key][key].narrow(0, a, b - a)
                    if not batch._shapes[store_key][key][number]:
                        value = value.reshape([])
                fields[store_key][key] = value
        output.append(data)
    return output


def _select(batch, indices):
    if not indices:
        return []
    if hasattr(batch, '_hook_plans'):
        return _custom(batch, indices)
    if hasattr(batch, '_field_keys'):
        return _heterogeneous(batch, indices)
    return _homogeneous(batch, indices)


def get_example(batch, index):
    return _select(batch, [_one(index, _count(batch))])[0]


def index_select(batch, index):
    return _select(batch, _many(index, _count(batch)))


def to_data_list(batch):
    return _select(batch, list(range(_count(batch))))


def getitem(batch, index):
    from ._graph_hetero_batch import HeteroBatch
    if isinstance(index, str):
        if isinstance(batch, HeteroBatch):
            return super(HeteroBatch, batch).__getitem__(index)
        from ._graph_data import Batch
        return super(Batch, batch).__getitem__(index)
    if isinstance(batch, HeteroBatch) and isinstance(index, tuple) and index and isinstance(index[0], str):
        return super(HeteroBatch, batch).__getitem__(index)
    if type(index) is int or isinstance(index, Tensor) and not index.shape:
        return get_example(batch, index)
    return index_select(batch, index)
