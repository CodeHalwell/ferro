"""In-place graph field ownership operations over native dense tensors.

Container identity, class, stores and private batch inverse metadata stay fixed.
Selected names are visited in argument order, including repeated names; absent
names are ignored. Generic callbacks run sequentially and may have partial
side effects if a later callback fails. They must return supported CPU Tensors
and remain responsible for graph alignment when changing shapes or topology.

Only Tensor fields participate. Explicit node counts, immutable native
EdgeIndex objects and private metadata are unchanged; nested Python values and
PyG's best-effort callbacks on arbitrary metadata are outside this typed API.
"""
from ._native import Tensor
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts


def _selectors(args):
    if any(type(key) is not str for key in args):
        raise TypeError('graph field selectors must be strings')


def _mappings(data):
    from ._graph_data import Data
    from ._graph_hetero import HeteroData
    if isinstance(data, Data):
        return [data._store]
    if isinstance(data, HeteroData):
        return [data._global, *(store._mapping for store in data._nodes.values()),
                *(store._mapping for store in data._edges.values())]
    raise TypeError('graph field operations require Data or HeteroData')


def apply(data, func, *args):
    _selectors(args)
    if not callable(func):
        raise TypeError('graph apply requires a callable')
    for mapping in _mappings(data):
        for key in args or tuple(mapping):
            value = mapping.get(key)
            if not isinstance(value, Tensor):
                continue
            result = func(value)
            _cpu_tensor(result, key, dtype=None)
            if result.dtype not in ('float32', 'float64', 'int64', 'bool'):
                raise ValueError('graph fields support CPU float32/float64/int64/bool only')
            if key == 'edge_index':
                _capacity(result.shape, 8)
                _edge_parts(result)
            # Replace only an existing field. Generated batch/ptr names cannot
            # use the public setter, which reserves them for ordinary writes.
            mapping[key] = result
    return data


def detach(data, *args):
    """Replace selected slots with detached views; external handles stay live."""
    return apply(data, lambda value: value.detach(), *args)


def contiguous(data, *args):
    """Materialize selected noncontiguous tensors using native contiguous."""
    return apply(data, lambda value: value.contiguous(), *args)


def cpu(data, *args):
    """Return this CPU-only container without changing its tensor handles."""
    _selectors(args)
    return data


def to(data, device, *args, non_blocking=False):
    """Only an identity CPU conversion is supported; no device migration."""
    if type(device) is not str or device != 'cpu':
        raise NotImplementedError('graph fields support CPU only; device migration is unsupported')
    if type(non_blocking) is not bool:
        raise TypeError('non_blocking must be bool')
    return cpu(data, *args)


def detach_(data, *args):
    raise NotImplementedError('in-place Tensor detachment is unsupported; use graph.detach() to replace field slots')
