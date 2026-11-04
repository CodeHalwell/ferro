"""Typed container subgraphs with explicit field alignment and ownership.

Node selections preserve order and repetitions; edge endpoints refer to the
last selected occurrence of a repeated node, matching PyG. Edge selections
preserve order and repetitions and keep every original node. Native EdgeIndex
inputs become int64 COO fields with an explicit retained node count.

Only unbatched, unsampled eager dense CPU graphs with one node space per store are
supported. Custom batching hooks, pointer fields and additional index/face/link
supervision fields require separate semantic remapping and reject. Graph-level
fields and untouched stores' tensors remain aliased; container/store mappings
are independent. Explicit homogeneous field_roles resolve ambiguous lengths.
"""
import math
import sys
from collections.abc import Mapping

from ._native import Tensor, EdgeIndex, _is_capturing
from ._graph_clone import _shallow
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor


def _bytes(dtype):
    return 8 if dtype in ('float64', 'int64') else 1 if dtype == 'bool' else 4


def _selection(subset, count, edge_only=False):
    _cpu_tensor(subset, 'subset', dtype=None)
    if subset.dtype not in ('int64', 'bool') or subset.ndim != 1:
        raise ValueError('subsets require rank-one CPU int64 IDs or bool masks')
    _capacity(subset.shape, _bytes(subset.dtype))
    if edge_only and subset.dtype == 'int64':
        _capacity((2, subset.shape[0]), 8)
    if subset.dtype == 'bool':
        if subset.shape != [count]:
            raise ValueError('boolean subset length must match its node or edge count')
        return [i for i, take in enumerate(subset.tolist()) if take]
    ids = subset.tolist()
    if any(value < 0 or value >= count for value in ids):
        raise ValueError('subset IDs must be nonnegative and in bounds')
    return ids


def _fields(mapping):
    for key in mapping:
        if key != 'edge_index' and ('index' in key or key in ('face', 'batch', 'ptr') or key.endswith(('_batch', '_ptr'))):
            raise NotImplementedError('subgraph remapping of indexed or batched fields is unsupported: ' + key)


def _topology(value, size):
    _capacity((2, value.num_edges) if isinstance(value, EdgeIndex) else value.shape, 8)
    return _edge_parts(value, size=size)[:2]


def _gather_plan(value, ids, axis=0):
    _cpu_tensor(value, 'subgraph field', dtype=None)
    if value.dtype not in ('float32', 'float64', 'int64', 'bool'):
        raise ValueError('subgraph fields support CPU float32/float64/int64/bool only')
    shape = list(value.shape)
    shape[axis] = len(ids)
    for dim in shape:
        _count(dim, 'dimension')
    if math.prod(shape) > sys.maxsize // _bytes(value.dtype):
        raise ValueError('subgraph output allocation capacity exceeded')
    _capacity((len(ids),), 8)
    return value, axis, ids


def _gather(plan):
    value, axis, ids = plan
    return value.index_select(axis, ids)


def _copy(data):
    from ._graph_data import Data
    result = _shallow(data)
    if isinstance(data, Data):
        result._store = dict(data._store)
        result._field_roles = dict(data._field_roles)
        return result
    result._global = dict(data._global)
    result._global_view = _shallow(data._global_view)
    result._global_view._parent, result._global_view._mapping = result, result._global
    for name in ('_nodes', '_edges'):
        stores = {}
        for key, store in getattr(data, name).items():
            copied = _shallow(store)
            copied._parent, copied._mapping = result, dict(store._mapping)
            stores[key] = copied
        object.__setattr__(result, name, stores)
    return result


def _homogeneous(data, subset, edge_only):
    from ._graph_data import Batch, _custom_hooks, _has_node_count
    from ._graph_sampling import _SampledData
    if isinstance(data, (Batch, _SampledData)):
        raise NotImplementedError('extract individual graphs before subgraph selection; batch/sampler inverse metadata is not remapped')
    if _custom_hooks(data):
        raise NotImplementedError('subgraph selection with custom batching hooks is unsupported')
    _fields(data._store)
    if not edge_only and not _has_node_count(data):
        raise ValueError('node subgraphs require an inferable or explicit num_nodes')
    n = data.num_nodes
    src, dst = ([], []) if data.edge_index is None else _topology(data.edge_index, (n, n))
    data.validate()
    roles = {key: data._role(key, n, len(src)) for key in data.keys() if key != 'edge_index'}
    ids = _selection(subset, len(src) if edge_only else n, edge_only)
    if edge_only:
        edge_ids, nodes = ids, None
        source, target = [src[i] for i in ids], [dst[i] for i in ids]
    else:
        nodes = ids
        remap = {node: i for i, node in enumerate(nodes)}
        edge_ids = [i for i, (s, d) in enumerate(zip(src, dst)) if s in remap and d in remap]
        source, target = [remap[src[i]] for i in edge_ids], [remap[dst[i]] for i in edge_ids]
    _capacity((2, len(edge_ids)), 8)
    plans = {key: _gather_plan(data[key], edge_ids if role == 'edge' else nodes)
             for key, role in roles.items() if role == 'edge' or role == 'node' and nodes is not None}
    result = _copy(data)
    result._num_nodes = n if edge_only else len(nodes)
    result._field_roles = roles
    if data.edge_index is not None:
        result._store['edge_index'] = _edge_tensor(source, target)
    for key, plan in plans.items():
        result._store[key] = _gather(plan)
    return result


def _heterogeneous(data, subsets, edge_only):
    from ._graph_hetero_batch import HeteroBatch
    from ._graph_hetero_sampling import _SampledStore
    if isinstance(data, HeteroBatch) or any(isinstance(store, _SampledStore) for store in data.stores):
        raise NotImplementedError('extract individual graphs before subgraph selection; batch/sampler inverse metadata is not remapped')
    if any(hasattr(type(data), hook) for hook in ('__inc__', '__cat_dim__')):
        raise NotImplementedError('subgraph selection with custom heterogeneous hooks is unsupported')
    if not isinstance(subsets, Mapping):
        raise TypeError('heterogeneous subsets must map declared node or full relation keys to tensors')
    stores = data._edges if edge_only else data._nodes
    if any(key not in stores for key in subsets):
        raise KeyError('subsets must name existing node types or full relation triples')
    for store in data.stores:
        _fields(store._mapping)
    counts = {key: store.num_nodes for key, store in data._nodes.items()}
    if any(count is None for count in counts.values()):
        raise ValueError('typed subgraphs require an inferable or explicit node count for every type')
    topology = {}
    for key, store in data._edges.items():
        if 'edge_index' in store:
            topology[key] = _topology(store.edge_index, (counts[key[0]], counts[key[2]]))
        elif store.num_edges:
            raise NotImplementedError('nonempty relation stores require edge_index for subgraph selection')
    data.validate()
    ids = {key: _selection(subset, stores[key].num_edges if edge_only else counts[key], edge_only) for key, subset in subsets.items()}
    nodes = {} if edge_only else ids
    node_plans = {(key, field): _gather_plan(value, selected)
                  for key, selected in nodes.items() for field, value in data[key].items()
                  if isinstance(value, Tensor) and value.shape}
    edges, edge_plans = {}, {}
    for key, store in data._edges.items():
        if edge_only and key not in ids:
            continue
        src, dst = topology.get(key, ([], []))
        if edge_only:
            selected = ids[key]
            source, target = [src[i] for i in selected], [dst[i] for i in selected]
        else:
            left = None if key[0] not in nodes else {node: i for i, node in enumerate(nodes[key[0]])}
            right = None if key[2] not in nodes else {node: i for i, node in enumerate(nodes[key[2]])}
            selected = [i for i, (s, d) in enumerate(zip(src, dst)) if (left is None or s in left) and (right is None or d in right)]
            source = [src[i] if left is None else left[src[i]] for i in selected]
            target = [dst[i] if right is None else right[dst[i]] for i in selected]
        _capacity((2, len(selected)), 8)
        if key in topology:
            edges[key] = (source, target)
        for field, value in store.items():
            if field != 'edge_index' and value.shape:
                edge_plans[key, field] = _gather_plan(value, selected)
    result = _copy(data)
    for key, selected in nodes.items():
        result[key]._mapping['num_nodes'] = len(selected)
    for (key, field), plan in node_plans.items():
        result[key]._mapping[field] = _gather(plan)
    for key, (src, dst) in edges.items():
        result[key]._mapping['edge_index'] = _edge_tensor(src, dst)
    for (key, field), plan in edge_plans.items():
        result[key]._mapping[field] = _gather(plan)
    return result


def subgraph(data, subset):
    if _is_capturing():
        raise NotImplementedError('container subgraphs require eager execution')
    from ._graph_data import Data
    from ._graph_hetero import HeteroData
    if isinstance(data, Data):
        return _homogeneous(data, subset, False)
    if isinstance(data, HeteroData):
        return _heterogeneous(data, subset, False)
    raise TypeError('subgraph requires Data or HeteroData')


def edge_subgraph(data, subset):
    if _is_capturing():
        raise NotImplementedError('container subgraphs require eager execution')
    from ._graph_data import Data
    from ._graph_hetero import HeteroData
    if isinstance(data, Data):
        return _homogeneous(data, subset, True)
    if isinstance(data, HeteroData):
        return _heterogeneous(data, subset, True)
    raise TypeError('edge_subgraph requires Data or HeteroData')
