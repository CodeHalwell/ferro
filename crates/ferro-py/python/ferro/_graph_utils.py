"""Homogeneous CPU graph utilities built from the existing tensor operations.

Floating computations support float32 and first-order gradients. Topology is
CPU int64; utilities return int64 [2, E] tensors even for native EdgeIndex input.
Max/min ties split gradients among actual inputs. PyG extensions may select one
winner; PyG's CPU scatter_reduce also counts its initial zero when the extremum
is zero. Bool subgraph masks are supported; CSR softmax is not implemented.
"""
import math
import sys

from ._native import Tensor, EdgeIndex, cat, scatter, segment_softmax


_MISSING = object()


def _cpu_tensor(value, name, dtype='float32'):
    if not isinstance(value, Tensor):
        raise TypeError(f'{name} must be a ferro Tensor')
    if value.device != 'cpu':
        raise ValueError(f'{name} requires CPU storage; move it explicitly')
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f'{name} requires {dtype}, got {value.dtype}')
    return value


def _count(value, name):
    if type(value) is not int or value < 0 or value > min(sys.maxsize, 2**63 - 1):
        raise ValueError(f'{name} must be a nonnegative int64-sized integer')
    return value


def _capacity(shape, itemsize=4):
    size = 1
    for dim in shape:
        size *= max(1, _count(dim, 'dimension'))
        if size > sys.maxsize // itemsize:
            raise ValueError('graph output allocation capacity exceeded')


def _index_ids(index, name='index', length=None):
    if isinstance(index, Tensor):
        _cpu_tensor(index, name, 'int64')
        if len(index.shape) != 1:
            raise ValueError(f'{name} must have shape [N]')
        ids = index.tolist()
    elif isinstance(index, (list, tuple)):
        ids = list(index)
    else:
        raise TypeError(f'{name} must be an int64 Tensor or integer sequence')
    if any(type(i) is not int or i < 0 or i > 2**63 - 1 for i in ids):
        raise ValueError(f'{name} must contain nonnegative int64 indices')
    if length is not None and len(ids) != length:
        raise ValueError(f'{name} must contain one index per input row')
    return ids


def _edge_parts(edge_index, num_nodes=None, size=None):
    """Return source IDs, destination IDs and validated (source, target) sizes."""
    if num_nodes is not None and size is not None:
        raise ValueError('provide num_nodes or size, not both')
    if isinstance(edge_index, EdgeIndex):
        src, dst = edge_index.src, edge_index.dst
        inferred = edge_index.num_nodes
    else:
        _cpu_tensor(edge_index, 'edge_index', 'int64')
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2, E]')
        src, dst = edge_index.tolist()
        if any(i < 0 for row in (src, dst) for i in row):
            raise ValueError('edge_index must contain nonnegative indices')
        inferred = max(max(src, default=-1), max(dst, default=-1)) + 1
    if size is None:
        n = _count(inferred if num_nodes is None else num_nodes, 'num_nodes')
        size = (n, n)
    else:
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError('size must be a pair (num_source_nodes, num_target_nodes)')
        size = tuple(_count(n, 'size') for n in size)
    if isinstance(edge_index, EdgeIndex) and size != (inferred, inferred):
        raise ValueError('native EdgeIndex node count must match the supplied homogeneous size')
    if any(i >= size[0] for i in src) or any(i >= size[1] for i in dst):
        raise ValueError('edge_index contains an out-of-bounds endpoint')
    _capacity((2, len(src)), 8)
    return src, dst, size


def _edge_tensor(src, dst):
    return Tensor.from_i64(src + dst, [2, len(src)])


def _edge_attr(edge_attr, edges):
    if edge_attr is not None:
        _cpu_tensor(edge_attr, 'edge_attr')
        if not edge_attr.shape or edge_attr.shape[0] != edges:
            raise ValueError('edge_attr must have shape [E, ...]')
    return edge_attr


def degree(index, num_nodes=None, dtype=None):
    """Count indices, returning float32 by default or exact int64 counts."""
    ids = _index_ids(index)
    n = max(ids, default=-1) + 1 if num_nodes is None else _count(num_nodes, 'num_nodes')
    if any(i >= n for i in ids):
        raise ValueError('index is out of bounds for num_nodes')
    if dtype not in (None, 'float32', 'int64'):
        raise ValueError('degree supports float32 or int64 dtype')
    _capacity((n,), 8 if dtype == 'int64' else 4)
    counts = [0] * n
    for i in ids:
        counts[i] += 1
    return Tensor.from_i64(counts, [n]) if dtype == 'int64' else Tensor(counts, [n])


def softmax(src, index=None, ptr=None, num_nodes=None, dim=0):
    """Stable softmax independently within each index group along dim zero."""
    _cpu_tensor(src, 'src')
    if ptr is not None:
        raise NotImplementedError('CSR ptr softmax is not supported; provide index')
    if dim != 0 or type(dim) is not int:
        raise NotImplementedError('grouped softmax supports dim=0 only')
    if not src.shape:
        raise ValueError('src must have shape [N, ...]')
    ids = _index_ids(index, length=src.shape[0])
    n = max(ids, default=-1) + 1 if num_nodes is None else _count(num_nodes, 'num_nodes')
    if any(i >= n for i in ids):
        raise ValueError('index is out of bounds for num_nodes')
    _capacity([n] + src.shape[1:])
    return segment_softmax(src, ids, n)


def _pool(x, batch, size, reduce):
    _cpu_tensor(x, 'x')
    if len(x.shape) not in (1, 2):
        raise ValueError('global pooling expects [N] or [N, F]')
    if size is not None:
        _count(size, 'size')
    ids = [0] * x.shape[0] if batch is None else _index_ids(batch, 'batch', x.shape[0])
    n = 1 if batch is None else (max(ids, default=-1) + 1 if size is None else size)
    if any(i >= n for i in ids):
        raise ValueError('batch index is out of bounds for size')
    _capacity([n] + x.shape[1:])
    if batch is None and x.shape[0] == 0:
        if reduce == 'max':
            raise ValueError('global_max_pool without batch requires a nonempty node axis')
        if reduce == 'mean':
            return scatter(x, ids, n, 'sum') / 0.0
    return scatter(x, ids, n, reduce)


def global_add_pool(x, batch, size=None):
    return _pool(x, batch, size, 'sum')


def global_mean_pool(x, batch, size=None):
    return _pool(x, batch, size, 'mean')


def global_max_pool(x, batch, size=None):
    """Pool maxima; missing graphs are zero and ties share gradients equally."""
    return _pool(x, batch, size, 'max')


def remove_self_loops(edge_index, edge_attr=None):
    src, dst, _ = _edge_parts(edge_index)
    _edge_attr(edge_attr, len(src))
    keep = [i for i, (s, d) in enumerate(zip(src, dst)) if s != d]
    edges = _edge_tensor([src[i] for i in keep], [dst[i] for i in keep])
    return edges, None if edge_attr is None else edge_attr.index_select(0, keep)


def add_self_loops(edge_index, edge_attr=None, fill_value=None, num_nodes=None):
    """Append every self loop, retaining existing loops as independent entries.

    A pair num_nodes=(Ns, Nt) adds min(Ns, Nt) loops. Attribute fill supports a
    scalar, a broadcastable feature Tensor, or add/sum/mean/min/max incoming
    reductions. Missing groups fill zero for reductions; the default fill is 1.
    """
    kw = {'size': num_nodes} if isinstance(num_nodes, (tuple, list)) else {'num_nodes': num_nodes}
    src, dst, size = _edge_parts(edge_index, **kw)
    _edge_attr(edge_attr, len(src))
    n = min(size)
    _capacity((2, len(src) + n), 8)
    loop_attr = None
    if edge_attr is not None:
        shape = [n] + edge_attr.shape[1:]
        _capacity(shape)
        if isinstance(fill_value, str):
            reduce = 'sum' if fill_value == 'add' else fill_value
            if reduce not in ('sum', 'mean', 'min', 'max'):
                raise NotImplementedError('loop attribute reduction supports add/sum/mean/min/max')
            _capacity([size[1]] + edge_attr.shape[1:])
            loop_attr = scatter(edge_attr, dst, size[1], reduce).narrow(0, 0, n)
        elif isinstance(fill_value, Tensor):
            _cpu_tensor(fill_value, 'fill_value')
            if len(fill_value.shape) > len(shape) or any(a not in (1, b) for a, b in zip(reversed(fill_value.shape), reversed(shape))):
                raise ValueError('fill_value cannot broadcast to loop attributes')
            loop_attr = Tensor.zeros(shape) + fill_value
        else:
            value = 1.0 if fill_value is None else fill_value
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('fill_value must be a finite scalar, float32 Tensor or reduction')
            loop_attr = Tensor.ones(shape) * value
    loops = list(range(n))
    return _edge_tensor(src + loops, dst + loops), None if edge_attr is None else cat([edge_attr, loop_attr], 0)


def coalesce(edge_index, edge_attr=_MISSING, num_nodes=None, reduce='sum', is_sorted=False, sort_by_row=True):
    """Sort edges and reduce duplicates; omitted attributes return only edges."""
    src, dst, _ = _edge_parts(edge_index, num_nodes=num_nodes)
    attr = None if edge_attr is _MISSING else _edge_attr(edge_attr, len(src))
    if type(is_sorted) is not bool or type(sort_by_row) is not bool:
        raise TypeError('is_sorted and sort_by_row must be bool')
    reduce = 'sum' if reduce == 'add' else reduce
    if reduce not in ('sum', 'mean', 'max', 'min'):
        raise NotImplementedError('coalesce supports add/sum/mean/max/min reductions')
    pairs = list(zip(src, dst))
    key = (lambda i: pairs[i]) if sort_by_row else (lambda i: (dst[i], src[i]))
    order = list(range(len(src))) if is_sorted else sorted(range(len(src)), key=key)
    if is_sorted and any(key(a) > key(b) for a, b in zip(order, order[1:])):
        raise ValueError('is_sorted=True requires sorted edge_index')
    unique, groups = [], []
    for i in order:
        if not unique or unique[-1] != pairs[i]:
            unique.append(pairs[i])
        groups.append(len(unique) - 1)
    edges = _edge_tensor([p[0] for p in unique], [p[1] for p in unique])
    if attr is not None:
        _capacity([len(unique)] + attr.shape[1:])
        attr = scatter(attr.index_select(0, order), groups, len(unique), reduce)
    return edges if edge_attr is _MISSING else (edges, attr)


def to_undirected(edge_index, edge_attr=_MISSING, num_nodes=None, reduce='add'):
    """Append reverse edges then coalesce, preserving attribute gradients."""
    src, dst, _ = _edge_parts(edge_index, num_nodes=num_nodes)
    attr = None if edge_attr is _MISSING else _edge_attr(edge_attr, len(src))
    _capacity((2, 2 * len(src)), 8)
    edges = _edge_tensor(src + dst, dst + src)
    if edge_attr is _MISSING:
        return coalesce(edges, num_nodes=num_nodes, reduce=reduce)
    return coalesce(edges, None if attr is None else cat([attr, attr], 0), num_nodes, reduce)


def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, num_nodes=None, *, return_edge_mask=False):
    """Induced subgraph with ordered integer IDs or a bool node mask.

    return_edge_mask adds an aligned native Bool Tensor to the result.
    Attribute selection remains differentiable.
    """
    if type(relabel_nodes) is not bool or type(return_edge_mask) is not bool:
        raise TypeError('relabel_nodes and return_edge_mask must be bool')
    if isinstance(subset, Tensor) and subset.dtype == 'bool':
        _cpu_tensor(subset, 'subset', 'bool')
        if subset.ndim != 1:
            raise ValueError('boolean subset must have shape [num_nodes]')
        if num_nodes is None:
            num_nodes = subset.shape[0]
    src, dst, size = _edge_parts(edge_index, num_nodes=num_nodes)
    _edge_attr(edge_attr, len(src))
    if isinstance(subset, Tensor) and subset.dtype == 'bool':
        _cpu_tensor(subset, 'subset', 'bool')
        if subset.shape != [size[0]]:
            raise ValueError('boolean subset must have shape [num_nodes]')
        nodes = [i for i, included in enumerate(subset.tolist()) if included]
    else:
        nodes = _index_ids(subset, 'subset')
    if len(set(nodes)) != len(nodes) or any(i >= size[0] for i in nodes):
        raise ValueError('subset must contain unique in-bounds node indices')
    mapping = {node: i for i, node in enumerate(nodes)}
    keep = [i for i, (s, d) in enumerate(zip(src, dst)) if s in mapping and d in mapping]
    source = [mapping[src[i]] if relabel_nodes else src[i] for i in keep]
    target = [mapping[dst[i]] if relabel_nodes else dst[i] for i in keep]
    result = (_edge_tensor(source, target), None if edge_attr is None else edge_attr.index_select(0, keep))
    if return_edge_mask:
        keep_set = set(keep)
        return result + (Tensor.from_bool([i in keep_set for i in range(len(src))], [len(src)]),)
    return result
