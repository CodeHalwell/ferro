"""Homogeneous CPU graph transforms and extraction without Torch dependencies.

Transforms return new Data containers; tensor history remains live. Random
operations use local Python RNG state, not PyG's draw sequence. Heterogeneous
stores, arbitrary object metadata and implicit loader/scheduler RNG are outside
this module's contract.
"""
import math
import random

from ._native import Tensor, EdgeIndex
from ._graph_data import Data
from ._graph_utils import (_capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor,
                           _index_ids, _edge_attr, add_self_loops, to_undirected)
from ._graph_link import _graph_fields, _rng


def _copy(data):
    roles = _graph_fields(data)
    result = Data(num_nodes=data.num_nodes, field_roles=roles)
    result._store = dict(data.items())
    return result, roles


def _probability(p, training):
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must be finite and between zero and one')
    if type(training) is not bool:
        raise TypeError('training must be bool')


def _subset(value, count, name):
    if isinstance(value, Tensor) and value.dtype == 'bool':
        _cpu_tensor(value, name, 'bool')
        if value.shape != [count]:
            raise ValueError(name + ' bool mask must match the node count')
        return [i for i, take in enumerate(value.tolist()) if take]
    ids = _index_ids(value, name)
    if len(set(ids)) != len(ids) or any(i >= count for i in ids):
        raise ValueError(name + ' must contain unique in-bounds nodes')
    return ids


class Compose:
    def __init__(self, transforms):
        self.transforms = tuple(transforms)
        if any(not callable(t) for t in self.transforms):
            raise TypeError('transforms must contain callables')

    def __call__(self, data):
        result = [_copy(d)[0] for d in data] if isinstance(data, (list, tuple)) else _copy(data)[0]
        for transform in self.transforms:
            result = [transform(d) for d in result] if isinstance(result, (list, tuple)) else transform(result)
        return result


class NormalizeFeatures:
    """Subtract each field's global minimum, then divide by max(row sum, 1)."""
    def __init__(self, attrs=None):
        if attrs is not None and not isinstance(attrs, (list, tuple)):
            raise TypeError('attrs must be a list/tuple of field names')
        self.attrs = ['x'] if attrs is None else list(attrs)
        if any(not isinstance(key, str) for key in self.attrs):
            raise TypeError('attrs must contain field names')

    def __call__(self, data):
        result, _ = _copy(data)
        for key in self.attrs:
            if key not in result:
                continue
            value = _cpu_tensor(result[key], key)
            if not value.shape:
                raise ValueError('NormalizeFeatures fields must have a feature axis')
            if value.numel():
                shifted = value + (-value).max()
                result._store[key] = shifted / shifted.sum_dim(len(value.shape) - 1, True).clamp(1., float('inf'))
        return result


class AddSelfLoops:
    """Append loops and fill one named float32 edge field, preserving gradients.

    Other edge fields reject before transformation, since their loop values
    would be undefined. Use an explicit transformation to supply those values.
    """
    def __init__(self, attr='edge_weight', fill_value=1.):
        if attr is not None and (not isinstance(attr, str) or attr == 'edge_index'):
            raise ValueError('attr must name one edge field or be None')
        self.attr, self.fill_value = attr, fill_value

    def __call__(self, data):
        result, roles = _copy(data)
        if data.edge_index is None:
            return result
        if any(role == 'edge' and key != self.attr for key, role in roles.items()):
            raise ValueError('AddSelfLoops needs explicit fill values for other edge fields')
        if self.attr in roles and roles[self.attr] != 'edge':
            raise ValueError('attr must be an edge field')
        edges, attr = add_self_loops(data.edge_index, data._store.get(self.attr), self.fill_value, data.num_nodes)
        result._store['edge_index'] = edges
        if attr is not None:
            result._store[self.attr] = attr
        return result


class RemoveSelfLoops:
    """Remove diagonal edges and select every declared edge field consistently."""
    def __init__(self, attr='edge_weight'):
        if attr is not None and not isinstance(attr, str):
            raise TypeError('attr must be a field name or None')
        self.attr = attr

    def __call__(self, data):
        result, roles = _copy(data)
        if data.edge_index is None:
            return result
        src, dst, _ = _edge_parts(data.edge_index, num_nodes=data.num_nodes)
        keep = [i for i, (s, d) in enumerate(zip(src, dst)) if s != d]
        result._store['edge_index'] = _edge_tensor([src[i] for i in keep], [dst[i] for i in keep])
        for key, role in roles.items():
            if role == 'edge':
                result._store[key] = data[key].index_select(0, keep)
        return result


class ToUndirected:
    """Reverse and coalesce topology and every float32 edge field.

    merge is retained for homogeneous PyG compatibility; heterogeneous reverse
    relation stores and integer/bool attribute reductions are unsupported.
    """
    def __init__(self, reduce='add', merge=True):
        if reduce not in ('add', 'sum', 'mean', 'min', 'max'):
            raise NotImplementedError('reduce must be add/sum/mean/min/max')
        if type(merge) is not bool:
            raise TypeError('merge must be bool')
        self.reduce, self.merge = reduce, merge

    def __call__(self, data):
        result, roles = _copy(data)
        if data.edge_index is None:
            return result
        attrs = [key for key, role in roles.items() if role == 'edge']
        for key in attrs:
            _cpu_tensor(data[key], key)
        result._store['edge_index'] = to_undirected(data.edge_index, num_nodes=data.num_nodes, reduce=self.reduce)
        for key in attrs:
            _, result._store[key] = to_undirected(data.edge_index, data[key], data.num_nodes, self.reduce)
        return result


class RandomNodeSplit:
    """Deterministic seed+call node masks; supports train_rest/test_rest/random."""
    def __init__(self, split='train_rest', num_splits=1, num_train_per_class=20, num_val=500, num_test=1000, key='y', *, seed=None):
        if split not in ('train_rest', 'test_rest', 'random'):
            raise ValueError('split must be train_rest, test_rest or random')
        if type(num_splits) is not int or num_splits <= 0:
            raise ValueError('num_splits must be positive')
        _count(num_train_per_class, 'num_train_per_class')
        for value in (num_val, num_test):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (type(value) is float and value > 1):
                raise ValueError('split sizes must be nonnegative counts or ratios in [0,1]')
        if key is not None and not isinstance(key, str):
            raise TypeError('key must be a field name or None')
        if split != 'train_rest' and key is None:
            raise ValueError('class-balanced splits require a label key')
        _rng(seed)
        self.split, self.num_splits, self.num_train_per_class = split, num_splits, num_train_per_class
        self.num_val, self.num_test, self.key = num_val, num_test, key
        self.seed, self.calls = seed, 0

    def __call__(self, data):
        result, _ = _copy(data)
        if self.key is not None and self.key not in data:
            return result
        n = data.num_nodes
        _capacity((n, self.num_splits), 1)
        nv = round(n * self.num_val) if type(self.num_val) is float else self.num_val
        nt = round(n * self.num_test) if type(self.num_test) is float else self.num_test
        classes = {}
        if self.split != 'train_rest':
            labels = _cpu_tensor(data[self.key], self.key, 'int64')
            if labels.shape != [n] or any(v < 0 for v in labels.tolist()):
                raise ValueError('class labels must be nonnegative int64 with shape [num_nodes]')
            for i, value in enumerate(labels.tolist()):
                classes.setdefault(value, []).append(i)
        rng = _rng(None if self.seed is None else self.seed + self.calls)
        masks = [[], [], []]
        for _ in range(self.num_splits):
            if self.split == 'train_rest':
                order = list(range(n)); rng.shuffle(order)
                train, val, test = order[nv + nt:], order[:nv], order[nv:nv + nt]
            else:
                train = []
                for label in sorted(classes):
                    choices = list(classes[label]); rng.shuffle(choices)
                    train.extend(choices[:self.num_train_per_class])
                selected = set(train)
                remaining = [i for i in range(n) if i not in selected]; rng.shuffle(remaining)
                val = remaining[:nv]
                test = remaining[nv:] if self.split == 'test_rest' else remaining[nv:nv + nt]
            for output, ids in zip(masks, (train, val, test)):
                chosen = set(ids)
                output.append([i in chosen for i in range(n)])
        shape = [n] if self.num_splits == 1 else [n, self.num_splits]
        for key, columns in zip(('train_mask', 'val_mask', 'test_mask'), masks):
            result._store[key] = Tensor.from_bool([columns[j][i] for i in range(n) for j in range(self.num_splits)], shape)
            result._field_roles[key] = 'node'
        self.calls += 1
        return result


def dropout_edge(edge_index, p=.5, force_undirected=False, training=True, *, seed=None):
    """Drop edges; return a Bool mask, or repeated I64 IDs when symmetrizing.

    As in PyG, force_undirected samples only rows with source<=target and
    appends reverse copies. Disabled/zero dropout returns an all-true Bool mask.
    """
    _probability(p, training)
    if type(force_undirected) is not bool:
        raise TypeError('force_undirected must be bool')
    src, dst, _ = _edge_parts(edge_index)
    rng = _rng(seed)
    if not training or p == 0:
        return _edge_tensor(src, dst), Tensor.from_bool([True] * len(src), [len(src)])
    mask = [rng.random() >= p and (not force_undirected or s <= d) for s, d in zip(src, dst)]
    ids = [i for i, take in enumerate(mask) if take]
    a, b = [src[i] for i in ids], [dst[i] for i in ids]
    if force_undirected:
        return _edge_tensor(a + b, b + a), Tensor.from_i64(ids + ids, [2 * len(ids)])
    return _edge_tensor(a, b), Tensor.from_bool(mask, [len(mask)])


def dropout_node(edge_index, p=.5, num_nodes=None, training=True, relabel_nodes=False, *, seed=None):
    """Drop nodes and incident edges, returning aligned Bool edge/node masks."""
    _probability(p, training)
    if type(relabel_nodes) is not bool:
        raise TypeError('relabel_nodes must be bool')
    src, dst, size = _edge_parts(edge_index, num_nodes=num_nodes)
    _capacity((size[0],), 1)
    rng = _rng(seed)
    mask = [not training or p == 0 or rng.random() >= p for _ in range(size[0])]
    mapping = {node: i for i, node in enumerate(i for i, take in enumerate(mask) if take)}
    keep = [mask[s] and mask[d] for s, d in zip(src, dst)]
    a = [mapping[s] if relabel_nodes else s for s, take in zip(src, keep) if take]
    b = [mapping[d] if relabel_nodes else d for d, take in zip(dst, keep) if take]
    return _edge_tensor(a, b), Tensor.from_bool(keep, [len(keep)]), Tensor.from_bool(mask, [len(mask)])


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None, flow='source_to_target', directed=False):
    """PyG-style hop extraction: sorted nodes, topology, seed mapping, Bool mask."""
    _count(num_hops, 'num_hops')
    if type(relabel_nodes) is not bool or type(directed) is not bool:
        raise TypeError('relabel_nodes and directed must be bool')
    if flow not in ('source_to_target', 'target_to_source'):
        raise ValueError('unsupported flow')
    src, dst, size = _edge_parts(edge_index, num_nodes=num_nodes)
    seeds = [_count(node_idx, 'node_idx')] if type(node_idx) is int else _index_ids(node_idx, 'node_idx')
    if any(i >= size[0] for i in seeds):
        raise ValueError('seed node is out of bounds')
    row, col = (dst, src) if flow == 'source_to_target' else (src, dst)
    frontier, discovered, retained = set(seeds), set(seeds), set()
    for _ in range(num_hops):
        if not frontier:
            break
        current = [i for i, node in enumerate(row) if node in frontier]
        retained.update(current)
        frontier = {col[i] for i in current}
        discovered.update(frontier)
    nodes = sorted(discovered)
    mapping = {node: i for i, node in enumerate(nodes)}
    keep = [i in retained if directed else s in mapping and d in mapping for i, (s, d) in enumerate(zip(src, dst))]
    a = [mapping[s] if relabel_nodes else s for s, take in zip(src, keep) if take]
    b = [mapping[d] if relabel_nodes else d for d, take in zip(dst, keep) if take]
    return (Tensor.from_i64(nodes, [len(nodes)]), _edge_tensor(a, b),
            Tensor.from_i64([mapping[n] for n in seeds], [len(seeds)]), Tensor.from_bool(keep, [len(keep)]))


def bipartite_subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, size=None, return_edge_mask=False):
    """Induce a rectangular graph from two ordered ID lists or Bool masks."""
    if not isinstance(subset, (list, tuple)) or len(subset) != 2:
        raise ValueError('subset must contain source and target selections')
    if type(relabel_nodes) is not bool or type(return_edge_mask) is not bool:
        raise TypeError('relabel_nodes and return_edge_mask must be bool')
    src, dst, native_size = _edge_parts(edge_index)
    if size is None:
        inferred = native_size if isinstance(edge_index, EdgeIndex) else (max(src, default=-1) + 1, max(dst, default=-1) + 1)
        size = tuple(selection.shape[0] if isinstance(selection, Tensor) and selection.dtype == 'bool' and len(selection.shape) == 1 else inferred[i] for i, selection in enumerate(subset))
    src, dst, size = _edge_parts(edge_index, size=size)
    _edge_attr(edge_attr, len(src))
    left, right = _subset(subset[0], size[0], 'source subset'), _subset(subset[1], size[1], 'target subset')
    ms, mt = {n: i for i, n in enumerate(left)}, {n: i for i, n in enumerate(right)}
    mask = [s in ms and d in mt for s, d in zip(src, dst)]
    ids = [i for i, take in enumerate(mask) if take]
    edges = _edge_tensor([ms[src[i]] if relabel_nodes else src[i] for i in ids], [mt[dst[i]] if relabel_nodes else dst[i] for i in ids])
    result = edges, None if edge_attr is None else edge_attr.index_select(0, ids)
    return result + (Tensor.from_bool(mask, [len(mask)]),) if return_edge_mask else result
