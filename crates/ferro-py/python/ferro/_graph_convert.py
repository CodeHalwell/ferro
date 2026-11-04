"""Typed CPU COO conversions using native concatenation and indexed selection.

Names retained on an in-memory homogeneous result include empty types. Generic
serialization need not preserve those private names; pass explicit names when
converting such a result back. Missing features use PyG's NaN/-1/False dummies;
two-dimensional node features may be padded with zeros to a common width.
Mixed field dtypes, scalar store metadata, custom index fields and batched graph
objects are unsupported. Original stores and tensors are never modified.
"""
from ._native import Tensor, cat
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor


def _bytes(dtype):
    return 8 if dtype in ('float64', 'int64') else 1 if dtype == 'bool' else 4


def _constant(shape, dtype, value):
    from ._tensor_api import full
    return full(shape, value, dtype=dtype)


def _names(names, name):
    if names is None:
        return None
    if not isinstance(names, (tuple, list)) or any(type(key) is not str or not key or (key.startswith('_') and name != 'node_type_names') for key in names):
        raise TypeError(name + ' must be a sequence of nonempty public field names')
    if len(set(names)) != len(names):
        raise ValueError(name + ' must not contain duplicates')
    return tuple(names)


def _feature_plan(stores, counts, names, dummy_values, node_fields):
    selected = _names(names, 'node_attrs' if node_fields else 'edge_attrs')
    automatic = selected is None
    if automatic:
        selected = tuple(dict.fromkeys(key for store in stores for key in store.keys()
                                       if key not in ('num_nodes', 'edge_index', 'ptr')))
    plans = {}
    for key in selected:
        if key in ('num_nodes', 'edge_index', 'ptr') or 'index' in key:
            raise NotImplementedError('conversion of custom index or pointer fields is unsupported: ' + key)
        values = [store._mapping.get(key) for store in stores]
        present = [value for value in values if value is not None]
        if not present:
            raise ValueError('requested conversion field does not exist: ' + key)
        if any(not value.shape for value in present):
            raise NotImplementedError('scalar node/relation metadata cannot be merged: ' + key)
        dtype = present[0].dtype
        if any(value.dtype != dtype for value in present):
            raise ValueError('conversion fields must have matching dtype across types: ' + key)
        tails = [tuple(value.shape[1:]) for value in present]
        missing = len(present) != len(stores)
        fillable = dummy_values and len(set(tails)) == 1
        compatible = len(set(tails)) == 1 or (node_fields and all(len(tail) == 1 for tail in tails))
        if not compatible or (missing and not fillable):
            if automatic:
                continue
            raise ValueError('requested field has incompatible or missing typed features: ' + key)
        tail = (max(tail[0] for tail in tails),) if node_fields and len(tails[0]) == 1 else tails[0]
        _capacity((sum(counts), *tail), _bytes(dtype))
        plans[key] = (tuple(values), tail, dtype)
    return plans


def _merge_fields(plans, counts, force_cat):
    fields = {}
    for key, (values, tail, dtype) in plans.items():
        pieces = []
        for value, count in zip(values, counts):
            if value is None:
                dummy = float('nan') if dtype in ('float32', 'float64') else False if dtype == 'bool' else -1
                value = _constant((count, *tail), dtype, dummy)
            elif tuple(value.shape[1:]) != tail:
                value = cat([value, _constant((count, tail[0] - value.shape[1]), dtype, 0)], 1)
            pieces.append(value)
        fields[key] = cat(pieces, 0) if force_cat or len(pieces) != 1 else pieces[0]
    return fields


def to_homogeneous(data, node_attrs=None, edge_attrs=None, add_node_type=True, add_edge_type=True, dummy_values=True):
    from ._graph_data import Data
    from ._graph_hetero_batch import HeteroBatch
    if isinstance(data, HeteroBatch):
        raise NotImplementedError('convert individual heterogeneous graphs before batching')
    if any(hasattr(type(data), hook) for hook in ('__inc__', '__cat_dim__')):
        raise NotImplementedError('custom graph conversion hooks are unsupported')
    if any(type(flag) is not bool for flag in (add_node_type, add_edge_type, dummy_values)):
        raise TypeError('conversion flags must be bool')
    data.validate()
    node_names, edge_names = tuple(data.node_types), tuple(data.edge_types)
    counts = tuple(data[key].num_nodes for key in node_names)
    edge_counts = tuple(data[key].num_edges for key in edge_names)
    nodes, edges = _count(sum(counts), 'total nodes'), _count(sum(edge_counts), 'total edges')
    _capacity((nodes,), 8)
    _capacity((2, edges), 8)
    node_plans = _feature_plan(data.node_stores, counts, node_attrs, dummy_values, True)
    edge_plans = _feature_plan(data.edge_stores, edge_counts, edge_attrs, dummy_values, False)
    roles = {**{key: 'graph' for key in data._global}, **{key: 'node' for key in node_plans},
             **{key: 'edge' for key in edge_plans}}
    keys = [*data._global, *node_plans, *edge_plans]
    if add_node_type:
        keys.append('node_type')
    if add_edge_type and edge_names:
        keys.append('edge_type')
    if len(set(keys)) != len(keys):
        raise ValueError('global, node, edge and generated type field names must not collide')
    if any(key in ('edge_index', 'num_nodes', 'batch', 'ptr', 'field_roles') or hasattr(Data, key) for key in keys):
        raise ValueError('a heterogeneous field conflicts with homogeneous graph metadata')
    offsets, cursor = {}, 0
    for key, count in zip(node_names, counts):
        offsets[key] = cursor
        cursor += count
    sources, targets = [], []
    for relation, count in zip(edge_names, edge_counts):
        store = data[relation]
        if 'edge_index' not in store:
            if count:
                raise ValueError('nonempty relation stores require explicit edge_index')
            continue
        src, dst, _ = _edge_parts(store.edge_index, size=store.size())
        sources.extend(value + offsets[relation[0]] for value in src)
        targets.extend(value + offsets[relation[2]] for value in dst)
    result = Data(num_nodes=nodes)
    for key, value in data._global.items():
        result[key] = value
    for plans, lengths, force_cat in ((node_plans, counts, True), (edge_plans, edge_counts, False)):
        for key, value in _merge_fields(plans, lengths, force_cat).items():
            result[key] = value
    if edge_names:
        result.edge_index = _edge_tensor(sources, targets)
    if add_node_type:
        result.node_type = Tensor.from_i64([kind for kind, count in enumerate(counts) for _ in range(count)], [nodes])
        roles['node_type'] = 'node'
    if add_edge_type and edge_names:
        result.edge_type = Tensor.from_i64([kind for kind, count in enumerate(edge_counts) for _ in range(count)], [edges])
        roles['edge_type'] = 'edge'
    object.__setattr__(result, '_field_roles', roles)
    object.__setattr__(result, '_node_type_names', node_names)
    object.__setattr__(result, '_edge_type_names', edge_names)
    result.validate()
    return result


def _type_ids(value, count, name):
    _capacity((count,), 8)
    if value is None:
        return [0] * count
    _cpu_tensor(value, name, 'int64')
    if value.shape != [count]:
        raise ValueError(name + ' must contain one int64 type ID per row')
    ids = value.tolist()
    if any(index < 0 for index in ids):
        raise ValueError(name + ' must contain nonnegative type IDs')
    return ids


def to_heterogeneous(data, node_type=None, edge_type=None, node_type_names=None, edge_type_names=None):
    from ._graph_data import Batch, _custom_hooks
    from ._graph_hetero import HeteroData, _Store, _relation
    if isinstance(data, Batch):
        raise NotImplementedError('convert individual homogeneous graphs before batching')
    if _custom_hooks(data):
        raise NotImplementedError('custom graph conversion hooks are unsupported')
    nodes = data.num_nodes
    _capacity((nodes,), 8)
    if isinstance(data.edge_index, Tensor):
        _capacity(data.edge_index.shape, 8)
    src, dst, _ = ([], [], (nodes, nodes)) if data.edge_index is None else _edge_parts(data.edge_index, num_nodes=nodes)
    edges = len(src)
    node_ids = _type_ids(data._store.get('node_type') if node_type is None else node_type, nodes, 'node_type')
    edge_ids = _type_ids(data._store.get('edge_type') if edge_type is None else edge_type, edges, 'edge_type')
    if node_type_names is None:
        node_type_names = getattr(data, '_node_type_names', None)
    if edge_type_names is None:
        edge_type_names = getattr(data, '_edge_type_names', None)
    if node_type_names is None:
        kinds = sorted(set(node_ids))
        if kinds != list(range(len(kinds))):
            raise ValueError('inferred node type IDs must be contiguous from zero; provide node_type_names for empty types')
        node_type_names = [str(kind) for kind in kinds]
    node_names = _names(node_type_names, 'node_type_names')
    if any(key == '_global_store' for key in node_names) or any(kind >= len(node_names) for kind in node_ids):
        raise ValueError('node_type IDs are outside the declared node type names')
    if nodes and len(node_names) > 1 and node_type is None and 'node_type' not in data:
        raise ValueError('multiple node types require an explicit node_type vector')
    if edge_type_names is None:
        kinds = sorted(set(edge_ids))
        if kinds != list(range(len(kinds))):
            raise ValueError('inferred edge type IDs must be contiguous from zero; provide edge_type_names for empty types')
        edge_type_names = []
        for kind in kinds:
            positions = [i for i, value in enumerate(edge_ids) if value == kind]
            source_types = {node_ids[src[i]] for i in positions}
            target_types = {node_ids[dst[i]] for i in positions}
            if len(source_types) != 1 or len(target_types) != 1:
                raise ValueError('one edge type cannot span multiple source or target node types')
            edge_type_names.append((node_names[next(iter(source_types))], str(kind), node_names[next(iter(target_types))]))
    if not isinstance(edge_type_names, (tuple, list)):
        raise TypeError('edge_type_names must be a sequence of relation triples')
    edge_names = tuple(_relation(key) for key in edge_type_names)
    if len(set(edge_names)) != len(edge_names) or any(kind >= len(edge_names) for kind in edge_ids):
        raise ValueError('edge type names must be unique and cover every edge_type ID')
    if edges and len(edge_names) > 1 and edge_type is None and 'edge_type' not in data:
        raise ValueError('multiple edge types require an explicit edge_type vector')
    if any(src_name not in node_names or dst_name not in node_names for src_name, _, dst_name in edge_names):
        raise ValueError('relation names reference an undefined node type')
    node_positions = [[] for _ in node_names]
    local = [0] * nodes
    for global_id, kind in enumerate(node_ids):
        local[global_id] = len(node_positions[kind])
        node_positions[kind].append(global_id)
    edge_positions = [[] for _ in edge_names]
    for position, kind in enumerate(edge_ids):
        relation = edge_names[kind]
        if (node_names[node_ids[src[position]]], node_names[node_ids[dst[position]]]) != (relation[0], relation[2]):
            raise ValueError('edge endpoints do not match the declared relation node types')
        edge_positions[kind].append(position)
    fields = {'node': {}, 'edge': {}, 'graph': {}}
    if any(key not in data._store or key == 'edge_index' for key in data._field_roles):
        raise ValueError('field_roles must name existing non-topology fields')
    for key, value in data.items():
        if key in ('edge_index', 'node_type', 'edge_type'):
            continue
        if key in ('batch', 'ptr') or 'index' in key:
            raise NotImplementedError('conversion of custom index or pointer fields is unsupported: ' + key)
        role = data._role(key, nodes, edges)
        count = nodes if role == 'node' else edges if role == 'edge' else None
        if count is not None and (not value.shape or value.shape[0] != count):
            raise ValueError(role + ' field has the wrong row count: ' + key)
        _capacity(value.shape, _bytes(value.dtype))
        fields[role][key] = value
    if any(key in node_names or hasattr(HeteroData, key) or key.endswith('_dict') for key in fields['graph']):
        raise ValueError('global fields conflict with heterogeneous type names or metadata')
    if any(key == 'num_nodes' or hasattr(_Store, key) for role in ('node', 'edge') for key in fields[role]):
        raise ValueError('a typed field conflicts with heterogeneous store metadata')
    result = HeteroData()
    for key, value in fields['graph'].items():
        result[key] = value
    for kind, name in enumerate(node_names):
        store = result[name]
        store.num_nodes = len(node_positions[kind])
        for key, value in fields['node'].items():
            store[key] = value.index_select(0, node_positions[kind])
    for kind, name in enumerate(edge_names):
        store, positions = result[name], edge_positions[kind]
        store.edge_index = _edge_tensor([local[src[i]] for i in positions], [local[dst[i]] for i in positions])
        for key, value in fields['edge'].items():
            store[key] = value.index_select(0, positions)
    result.validate()
    return result
