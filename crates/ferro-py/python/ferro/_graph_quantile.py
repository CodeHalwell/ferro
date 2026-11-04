"""Native sorted grouped quantiles with live interpolation coefficients."""
import math

from ._native import Tensor
from ._graph_aggregation import _Aggregation, _layout
from ._graph_conv import _configure
from ._graph_utils import _capacity, _count, _cpu_tensor


class QuantileAggregation(_Aggregation):
    """Featurewise grouped quantiles using two native stable sorts.

    CPU float32 rank-two inputs accept unsorted index groups; ptr and capture
    are unsupported. Multiple quantiles concatenate in requested order. The
    registered float32 q buffer stays live in linear interpolation, including
    derivatives if its gradient flag is enabled. Global grouped offsets retain
    PyG's float32 interpolation rounding. Negative/out-of-range selected rows
    reject, preserving PyG's leading-empty-group boundary. Empty input supports
    only one quantile and zero output groups. Sorting ties retain input order;
    tied adjoints may differ from PyG's first, unstable value sort.
    """
    _builtin_init = True
    num_quantiles: int
    interpolation: str
    fill_value: float

    def __init__(self, q, interpolation='linear', fill_value=0.):
        values = tuple(q) if isinstance(q, (list, tuple)) else (q,)
        if not values or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError('q must contain finite quantile values in [0,1]')
        if interpolation not in ('linear', 'lower', 'higher', 'nearest', 'midpoint'):
            raise ValueError('unsupported quantile interpolation')
        if type(fill_value) not in (int, float): raise TypeError('fill_value must be a scalar')
        _capacity((len(values),), 8)
        values = tuple(float(value) for value in values)
        _configure(self, num_quantiles=len(values), interpolation=interpolation, fill_value=float(fill_value))
        self.register_buffer('q', Tensor(list(values), [len(values), 1]))

    def get_out_channels(self, in_channels):
        return _count(_count(in_channels, 'in_channels') * self.num_quantiles, 'quantile output width')

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        if ptr is not None: raise NotImplementedError('quantile aggregation requires index input; ptr is unsupported')
        ids, groups = _layout(x, index, None, dim_size, dim)
        if math.isfinite(self.fill_value) and abs(self.fill_value) > 3.4028234663852886e38:
            raise ValueError('fill_value overflows float32')
        quantiles = self.num_quantiles
        _cpu_tensor(self.q, 'q')
        if self.q.shape != [quantiles, 1]: raise ValueError('q buffer shape must match the configured quantile count')
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.q.flatten().tolist()):
            raise ValueError('q buffer values must remain finite and in [0,1]')
        _capacity(x.shape, 8)
        _capacity((groups, quantiles), 8)
        _capacity((groups, self.get_out_channels(x.shape[1])))
        if not x.shape[0] and (groups or quantiles != 1):
            raise ValueError('empty quantiles require one quantile and zero output groups')
        counts = [0] * groups
        for group in ids: counts[group] += 1
        offsets, offset = [], 0
        for count in counts:
            offsets.append(min(offset, x.shape[0] - 1) if dim_size is not None else offset)
            offset += count
        count = self.num_quantiles
        points = (self.q * Tensor([float(value - 1) for value in counts], [1, groups]) + Tensor([float(value) for value in offsets], [1, groups])).transpose(0, 1).reshape([groups * count])
        if self.interpolation == 'lower': selected = [points.floor()]
        elif self.interpolation == 'higher': selected = [points.ceil()]
        elif self.interpolation == 'nearest': selected = [points.round()]
        else: selected = [points.floor(), points.ceil()]
        rows = [[int(value) for value in part.tolist()] for part in selected]
        if any(row < 0 or row >= x.shape[0] for selection in rows for row in selection):
            raise ValueError('quantile selected row is outside the input; a leading empty group can cause this')
        return ids, groups, counts, points, rows

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, groups, counts, points, rows = self._preflight(x, index, ptr, dim_size, dim)
        count, width = self.num_quantiles, x.shape[1]
        values, permutation = x.sort(dim=0, descending=False, stable=True)
        types = Tensor.from_i64(ids, [x.shape[0], 1]).expand(x.shape[0], width).gather(0, permutation)
        _, permutation = types.sort(dim=0, descending=False, stable=True)
        values = values.gather(0, permutation)
        output = values.index_select(0, rows[0])
        if len(rows) == 2:
            upper = values.index_select(0, rows[1])
            output = output + (upper - output) * points.frac().reshape([groups * count, 1]) if self.interpolation == 'linear' else output * .5 + upper * .5
        missing = Tensor.from_bool([size == 0 for size in counts for _ in range(count)], [groups * count, 1])
        output = output.masked_fill(missing, self.fill_value)
        return output.reshape([groups, count * width])


class MedianAggregation(QuantileAggregation):
    """Featurewise lower median; even groups select the lower middle entry."""
    _builtin_init = True

    def __init__(self, fill_value=0.):
        super().__init__(.5, 'lower', fill_value)
