"""Typed native graph clones with explicit field and container ownership.

Selected tensor fields receive native preserve-format clones; unselected fields
stay aliased, as in PyG. Internal batch slice dictionaries are independently
copied, a deliberate stronger isolation boundary than PyG's shallow metadata.
Custom Python classes and private attributes survive without constructor calls.
Arbitrary user-defined __copy__ callbacks are not invoked.
"""
import math
import sys
from types import MemberDescriptorType

from ._native import Tensor, EdgeIndex
from ._graph_utils import _capacity, _count, _cpu_tensor


def _shallow(value):
    result = object.__new__(type(value))
    result.__dict__.update(value.__dict__)
    for cls in type(value).__mro__:
        for descriptor in cls.__dict__.values():
            if isinstance(descriptor, MemberDescriptorType):
                try:
                    member = descriptor.__get__(value, type(value))
                except AttributeError:
                    continue
                descriptor.__set__(result, member)
    return result


def _metadata(value):
    if isinstance(value, dict):
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_metadata(item) for item in value)
    return value


def _preflight(value):
    if isinstance(value, EdgeIndex):
        _capacity((2, value.num_edges), 8)
    elif isinstance(value, Tensor):
        _cpu_tensor(value, 'graph clone field', dtype=None)
        if value.dtype not in ('float32', 'float64', 'int64', 'bool'):
            raise ValueError('graph clone fields require float32/float64/int64/bool')
        shape = tuple(_count(dim, 'dimension') for dim in value.shape)
        itemsize = 8 if value.dtype in ('float64', 'int64') else 1 if value.dtype == 'bool' else 4
        if math.prod(shape) > sys.maxsize // itemsize:
            raise ValueError('graph clone allocation capacity exceeded')


def _clone(value):
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, EdgeIndex):
        return EdgeIndex(value.num_nodes, value.src, value.dst)
    return value


def clone(data, *args):
    from ._graph_data import Data
    from ._graph_hetero import HeteroData

    if any(type(key) is not str for key in args):
        raise TypeError('graph clone field selectors must be strings')
    selected = set(args)
    if isinstance(data, Data):
        mappings = [data._store]
    elif isinstance(data, HeteroData):
        mappings = [data._global, *(store._mapping for store in data._nodes.values()), *(store._mapping for store in data._edges.values())]
    else:
        raise TypeError('clone requires Data or HeteroData')
    for mapping in mappings:
        for key, value in mapping.items():
            if not selected or key in selected:
                _preflight(value)
    result = _shallow(data)
    # Only engine-owned metadata is recursively copied. Arbitrary private user
    # objects remain shallow references, matching normal Python/PyG copying.
    for key in ('_field_roles', '_slices', '_shapes', '_original_keys', '_hook_plans',
                '_node_counts', '_nodeptr', '_edge_counts', '_edge_totals', '_field_keys', '_generated'):
        if key in data.__dict__:
            object.__setattr__(result, key, _metadata(data.__dict__[key]))
    if isinstance(data, Data):
        result._store = {key: _clone(value) if not selected or key in selected else value for key, value in data._store.items()}
        for key in ('_nodes', '_edges'):
            if key in data.__dict__:
                object.__setattr__(result, key, _metadata(data.__dict__[key]))
        if '_hook_templates' in data.__dict__:
            result._hook_templates = []
            for template in data._hook_templates:
                copied = _shallow(template)
                copied._store = dict(template._store)
                copied._field_roles = dict(template._field_roles)
                result._hook_templates.append(copied)
        return result
    result._global = {key: _clone(value) if not selected or key in selected else value for key, value in data._global.items()}
    result._global_view = _shallow(data._global_view)
    result._global_view._parent, result._global_view._mapping = result, result._global
    for name in ('_nodes', '_edges'):
        original, output = getattr(data, name), {}
        for key, store in original.items():
            copied = _shallow(store)
            copied._parent = result
            copied._mapping = {field: _clone(value) if not selected or field in selected else value for field, value in store.items()}
            output[key] = copied
        object.__setattr__(result, name, output)
    return result
