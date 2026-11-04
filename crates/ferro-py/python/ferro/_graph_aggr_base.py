"""Subclassable eager graph aggregation with native reductions and padding."""
import inspect
import math

from ._native import Tensor, cat, scatter
from ._graph_conditioned import _eager
from ._graph_conv import _configure
from ._graph_utils import _capacity, _count, _cpu_tensor, _index_ids
from .nn import Module


def _metadata(x, index, ptr, dim_size, dim):
    _eager(); _cpu_tensor(x, 'x')
    rank = len(x.shape)
    if type(dim) is not int or not -rank <= dim < rank: raise ValueError('dim is outside the source tensor rank')
    axis = dim % rank
    rows = x.shape[axis]
    _capacity(x.shape); _capacity((rows + 1,), 8)
    if dim_size is not None: _count(dim_size, 'dim_size')
    if index is not None and ptr is not None: raise ValueError('provide index or ptr, not both')
    if ptr is not None:
        _cpu_tensor(ptr, 'ptr', 'int64')
        _capacity(ptr.shape, 8)
        offsets = _index_ids(ptr, 'ptr')
        if not offsets or offsets[0] != 0 or offsets[-1] != rows or any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError('ptr must start at zero, be nondecreasing and end at the node count')
        count = len(offsets) - 1
        if dim_size is not None and dim_size != count: raise ValueError('dim_size must equal len(ptr)-1')
        ids = [group for group, (left, right) in enumerate(zip(offsets, offsets[1:])) for _ in range(right-left)]
    else:
        if isinstance(index, Tensor): _capacity(index.shape, 8)
        ids = [0] * rows if index is None else _index_ids(index, 'index', rows)
        count = max(ids, default=-1) + 1 if dim_size is None else dim_size
        if any(group >= count for group in ids): raise ValueError('index contains a group outside dim_size')
    _capacity((count + 1,), 8)
    output_shape = list(x.shape); output_shape[axis] = count; _capacity(output_shape)
    return ids, count, axis


class Aggregation(Module):
    """PyG-style custom aggregation over first-order CPU float32 native operations.

    Handwritten subclasses call super().__init__ and implement forward. Calls
    infer missing index/dim_size before standard Module hooks. Native reductions
    support arbitrary node axes and sum/add/mean/min/max; extrema retain Ferro's
    actual-neighbor tie convention. Rank-two sorted dense batching supports
    first-K truncation and live feature adjoints. Integer lists extend tensor
    indices; simultaneous index/ptr, sparse/fused paths and capture reject.

    Python callbacks and custom forward side effects remain caller-owned.
    get_out_channels is an explicit width protocol for graph compositions;
    width-changing subclasses must override it.
    """
    _builtin_init = True

    def __init_subclass__(cls, **kwargs):
        cls._builtin_init = True
        super().__init_subclass__(**kwargs)
        signature = inspect.signature(cls.__init__)
        cls.__signature__ = signature.replace(parameters=list(signature.parameters.values())[1:])

    def __init__(self):
        _configure(self)

    def __call__(self, x, index=None, ptr=None, dim_size=None, dim=-2, **kwargs):
        ids, count, _ = _metadata(x, index, ptr, dim_size, dim)
        if index is None and ptr is None: index = Tensor.from_i64(ids, [len(ids)])
        return super().__call__(x, index=index, ptr=ptr, dim_size=count, dim=dim, **kwargs)

    def reset_parameters(self):
        pass

    def get_out_channels(self, in_channels):
        return _count(in_channels, 'in_channels')

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, count, _ = _metadata(x, index, ptr, dim_size, dim)
        self.assert_two_dimensional_input(x, dim)
        width = self.get_out_channels(x.shape[1]); _count(width, 'aggregation output width')
        _capacity((count, width))

    def reduce(self, x, index=None, ptr=None, dim_size=None, dim=-2, reduce='sum'):
        if index is None and ptr is None: raise RuntimeError('reduce requires index or ptr')
        if reduce not in ('sum', 'add', 'mean', 'min', 'max'): raise NotImplementedError('native aggregation supports sum/add/mean/min/max')
        ids, count, axis = _metadata(x, index, ptr, dim_size, dim)
        order = [axis] + [position for position in range(len(x.shape)) if position != axis]
        rest = [x.shape[position] for position in order[1:]]
        flat = x.permute(order).reshape([x.shape[axis], math.prod(rest)])
        output = scatter(flat, ids, count, 'sum' if reduce == 'add' else reduce).reshape([count] + rest)
        return output.permute([order.index(position) for position in range(len(order))])

    def assert_index_present(self, index):
        if index is None: raise NotImplementedError('aggregation requires index')

    def assert_sorted_index(self, index):
        if index is not None:
            if isinstance(index, Tensor): _capacity(index.shape, 8)
            ids = _index_ids(index, 'index')
            if any(left > right for left, right in zip(ids, ids[1:])): raise ValueError('aggregation requires sorted index')

    def assert_two_dimensional_input(self, x, dim):
        _cpu_tensor(x, 'x')
        if len(x.shape) != 2 or type(dim) is not int or dim not in (0, -2): raise ValueError('dense batching requires rank-two x with dim=0/-2')

    def to_dense_batch(self, x, index=None, ptr=None, dim_size=None, dim=-2, fill_value=0., max_num_elements=None):
        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)
        ids, count, _ = _metadata(x, index, ptr, dim_size, dim)
        if any(left > right for left, right in zip(ids, ids[1:])): raise ValueError('dense batching requires sorted index')
        if type(fill_value) not in (int, float) or math.isfinite(fill_value) and abs(fill_value) > 3.4028234663852886e38:
            raise ValueError('fill_value must be representable as float32')
        if max_num_elements is not None: _count(max_num_elements, 'max_num_elements')
        if not ids and dim_size is None: raise ValueError('empty dense batches require explicit dim_size')
        if count == 0 and max_num_elements is None: raise ValueError('zero groups require explicit max_num_elements')
        sizes = {}
        for group in ids: sizes[group] = sizes.get(group, 0) + 1
        time = max(sizes.values(), default=0) if max_num_elements is None else max_num_elements
        rows, width = x.shape
        _capacity((count, time), 8); _capacity((count, time, width)); _capacity((rows + 1, width))
        if time == 0:
            empty = x.index_select(0, []).reshape([1, 0, width]).expand([count, 0, width])
            return empty, Tensor.from_bool([], [count, 0])
        positions, offsets = [rows] * (count * time), {}
        for row, group in enumerate(ids):
            column = offsets.get(group, 0); offsets[group] = column + 1
            if column < time: positions[group * time + column] = row
        padding = Tensor.ones([1, width]) * float(fill_value)
        output = cat([x, padding], 0).index_select(0, positions).reshape([count, time, width])
        return output, Tensor.from_bool([position != rows for position in positions], [count, time])
