"""Dense homogeneous batching hooks with saved inverse slice/offset metadata.

Hook increments are nonnegative Python integers or CPU int64 tensors. Sparse
tuple axes, floating increments and nested fields are deliberately unsupported.
Custom callbacks own their side effects; all returned metadata is checked before
concatenation. Floating fields never leave native storage.
"""
import copy
import math

from ._native import Tensor, EdgeIndex, cat
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor


def _increment(value):
    if type(value) is int:
        return (), [_count(value, 'batch increment')]
    _cpu_tensor(value, 'batch increment', 'int64')
    _capacity(value.shape, 8)
    flat = value.reshape([-1]).tolist()
    for item in flat:
        _count(item, 'batch increment')
    return tuple(value.shape), flat


def _broadcast(shape, increment_shape):
    if len(increment_shape) > len(shape):
        raise ValueError('batch increment cannot add dimensions to its field')
    if any(a not in (1, b) for a, b in zip(reversed(increment_shape), reversed(shape))):
        raise ValueError('batch increment shape must broadcast to the field')


def _check_integer_shift(value, increment_shape, increment):
    """Check exact int64 addition without converting integer IDs to floats."""
    if value.dtype != 'int64' or not any(increment):
        return
    _capacity(value.shape, 8)
    shape = tuple(value.shape)
    padded = (1,) * (len(shape) - len(increment_shape)) + increment_shape
    strides = [math.prod(padded[axis + 1:]) for axis in range(len(shape))]
    for position, item in enumerate(value.reshape([-1]).tolist()):
        offset_index, remaining = 0, position
        for axis in range(len(shape) - 1, -1, -1):
            coordinate, remaining = remaining % shape[axis], remaining // shape[axis]
            if padded[axis] != 1:
                offset_index += coordinate * strides[axis]
        shifted = item + increment[offset_index]
        if not -(2**63) <= shifted < 2**63:
            raise ValueError('batch increment would overflow int64 field values')


def _shift(value, shape, offset, *, subtract=False):
    increment = Tensor.from_i64(offset, list(shape))
    if value.dtype != 'int64':
        increment = increment.to(dtype=value.dtype)
    return value - increment if subtract else value + increment


def collate(cls, items, follow, excluded):
    from ._graph_data import Data, _INDEX_FIELDS, _label_parts, _NODE_FIELDS, _EDGE_FIELDS, _has_node_count

    if len({type(data) for data in items}) != 1:
        raise TypeError('custom batching requires one Data subclass throughout')
    keys = [key for key in items[0].keys() if key not in excluded]
    if any(set(data.keys()) - excluded != set(keys) for data in items):
        raise ValueError('all graphs must have the same tensor field names')
    follow = tuple(key for key in follow if key in keys)
    generated = [key + suffix for key in follow for suffix in ('_batch', '_ptr')]
    if any(key in keys or key in ('batch', 'ptr') for key in generated) or len(set(generated)) != len(generated):
        raise ValueError('follow_batch generated names conflict with graph fields')

    # Preserve explicit row-role validation. Unknown custom fields can describe
    # several independent node spaces, so they are not inferred against one N.
    nodes, has_nodes, templates = [], [], []
    for data in items:
        n, e = data.num_nodes, data.num_edges
        _count(n, 'num_nodes')
        if data.edge_index is not None:
            _edge_parts(data.edge_index, num_nodes=n)
        if any(key not in data._store or key == 'edge_index' for key in data._field_roles):
            raise ValueError('field_roles must name existing non-topology fields')
        checked_roles = {key: ('node' if key in _NODE_FIELDS else 'edge')
                         for key in data.keys() if key in _NODE_FIELDS or key in _EDGE_FIELDS}
        checked_roles.update(data._field_roles)
        for key, role in checked_roles.items():
            value = data[key]
            count = n if role == 'node' else e if role == 'edge' else None
            if count is not None and (not value.shape or value.shape[0] != count):
                raise ValueError(f'{role} field {key!r} has the wrong row count')
            if role == 'edge' and key in keys and 'edge_index' not in keys and e:
                raise NotImplementedError('excluding edge_index also requires excluding nonempty edge fields')
        for key in _INDEX_FIELDS.intersection(keys):
            _label_parts(data, key)
        nodes.append(n)
        has_nodes.append(_has_node_count(data))
        template = copy.copy(data)
        template._store = {}
        template._field_roles = {key: value for key, value in data._field_roles.items() if key in keys}
        templates.append(template)
    if len(set(has_nodes)) != 1:
        raise ValueError('all custom examples must agree on whether a node count can be inferred')
    ptr = [0]
    for n in nodes:
        ptr.append(_count(ptr[-1] + n, 'total nodes'))
    if has_nodes[0]:
        _capacity((ptr[-1],), 8)
    _capacity((len(items) + 1,), 8)

    plans = {}
    for key in keys:
        values = []
        for data in items:
            value = data[key]
            if isinstance(value, EdgeIndex):
                src, dst, _ = _edge_parts(value)
                value = _edge_tensor(src, dst)
            _cpu_tensor(value, key, dtype=None)
            _capacity(value.shape, 8 if value.dtype in ('int64', 'float64') else 4)
            values.append(value)
        axis = items[0].__cat_dim__(key, values[0], items[0]._store)
        if axis is not None and type(axis) is not int:
            raise NotImplementedError('dense batching __cat_dim__ must return an integer or None')
        stacked = axis is None or not values[0].shape
        shapes = [[1] + value.shape if stacked else value.shape for value in values]
        rank = len(shapes[0])
        axis = 0 if stacked else axis
        if not -rank <= axis < rank:
            raise ValueError('batch concatenation axis is out of bounds')
        axis %= rank
        if key == 'edge_index' and axis != 1:
            raise ValueError('edge_index requires column concatenation')
        if any(value.dtype != values[0].dtype or len(shape) != rank or any(a != b for dim, (a, b) in enumerate(zip(shape, shapes[0])) if dim != axis) for value, shape in zip(values, shapes)):
            raise ValueError(f'field {key!r} must have matching dtype and non-concatenated dimensions')
        slices = [0]
        for shape in shapes:
            slices.append(_count(slices[-1] + shape[axis], 'total field entries'))
        output_shape = list(shapes[0])
        output_shape[axis] = slices[-1]
        _capacity(output_shape, 8 if values[0].dtype in ('int64', 'float64') else 4)
        increments = [_increment(data.__inc__(key, value.unsqueeze(0) if stacked else value, data._store)) for data, value in zip(items, values)]
        increment_shape = increments[0][0]
        if any(shape != increment_shape for shape, _ in increments):
            raise ValueError('batch increment shapes must agree across examples')
        for shape in shapes:
            _broadcast(shape, increment_shape)
        offsets, current = [], [0] * len(increments[0][1])
        for _, increment in increments:
            offsets.append(list(current))
            current = [_count(a + b, 'cumulative batch increment') for a, b in zip(current, increment)]
        apply = bool(increment_shape) or any(offsets[-1])
        if apply and values[0].dtype == 'bool':
            raise NotImplementedError('nonzero or tensor increments for bool fields are unsupported')
        if apply:
            for value, offset in zip(values, offsets):
                _check_integer_shift(value.unsqueeze(0) if stacked else value, increment_shape, offset)
        if key in follow:
            _capacity((slices[-1],), 8)
        plans[key] = values, axis, stacked, slices, increment_shape, offsets, apply

    data_cls = type(items[0])
    result_cls = cls if data_cls is Data else type(data_cls.__name__ + 'Batch', (cls, data_cls), {})
    result = object.__new__(result_cls)
    Data.__init__(result, num_nodes=ptr[-1] if has_nodes[0] else None)
    for key, (values, axis, stacked, slices, increment_shape, offsets, apply) in plans.items():
        shifted = []
        for value, offset in zip(values, offsets):
            value = value.unsqueeze(0) if stacked else value
            shifted.append(_shift(value, increment_shape, offset) if apply else value)
        result._store[key] = cat(shifted, axis)
        result._field_roles[key] = 'graph'
        if key in follow:
            result._store[key + '_batch'] = Tensor.from_i64([i for i, (a, b) in enumerate(zip(slices, slices[1:])) for _ in range(b - a)], [slices[-1]])
            result._store[key + '_ptr'] = Tensor.from_i64(slices, [len(slices)])
            result._field_roles[key + '_batch'] = result._field_roles[key + '_ptr'] = 'graph'
    if has_nodes[0]:
        result._store['batch'] = Tensor.from_i64([i for i, n in enumerate(nodes) for _ in range(n)], [ptr[-1]])
        result._store['ptr'] = Tensor.from_i64(ptr, [len(ptr)])
    # Save only immutable slice/offset descriptions and empty subclass templates;
    # reconstruction reads the CURRENT batch tensors, not the original fields.
    result._hook_plans = {key: (axis, stacked, tuple(slices), increment_shape, tuple(tuple(offset) for offset in offsets), apply, tuple(tuple(value.shape) for value in values))
                          for key, (values, axis, stacked, slices, increment_shape, offsets, apply) in plans.items()}
    result._hook_templates = templates
    result._num_graphs = len(items)
    return result


def separate(batch):
    from ._graph_batch_select import to_data_list
    return to_data_list(batch)
