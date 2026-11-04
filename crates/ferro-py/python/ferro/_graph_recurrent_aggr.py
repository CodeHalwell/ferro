"""Order-sensitive recurrent aggregation over sorted sparse graph groups."""
from ._native import Tensor, cat
from ._graph_utils import _capacity, _count
from ._graph_conv import _configure, _positive
from ._graph_aggregation import _Aggregation, _layout
from .nn import LSTM, GRU


class _RecurrentAggregation(_Aggregation):
    in_channels: int
    out_channels: int

    def _build(self, in_channels, out_channels, constructor, options):
        _positive(in_channels, 'in_channels')
        _positive(out_channels, 'out_channels')
        if 'batch_first' in options:
            raise TypeError('recurrent graph aggregation fixes batch_first=True')
        _configure(self, in_channels=in_channels, out_channels=out_channels)
        setattr(self, self._kind, constructor(in_channels, out_channels, batch_first=True, **options))
        self.reset_parameters()

    def reset_parameters(self):
        getattr(self, self._kind).reset_parameters()

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        if in_channels != self.in_channels:
            raise ValueError('recurrent aggregation input width disagrees with in_channels')
        recurrent = getattr(self, self._kind)
        return (getattr(recurrent, 'proj_size', 0) or self.out_channels) * (2 if recurrent.bidirectional else 1)

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        if ptr is not None:
            raise NotImplementedError('recurrent graph aggregation requires sorted index input; ptr is unsupported')
        ids, groups = _layout(x, index, None, dim_size, dim)
        width = self.get_out_channels(x.shape[1])
        if any(left > right for left, right in zip(ids, ids[1:])):
            raise ValueError('recurrent graph aggregation requires sorted indices')
        counts = {}
        for group in ids: counts[group] = counts.get(group, 0) + 1
        time = max(counts.values(), default=0) if max_num_elements is None else _count(max_num_elements, 'max_num_elements')
        if time == 0:
            raise ValueError('recurrent aggregation sequence length must be positive; provide max_num_elements for empty input')
        recurrent = getattr(self, self._kind)
        _capacity((time,), 8)
        _capacity((groups, time), 8)
        _capacity((groups, time, self.in_channels))
        _capacity((x.shape[0] + 1, self.in_channels))
        _capacity((groups, time, width))
        _capacity((recurrent.num_layers, groups, (2 if recurrent.bidirectional else 1), self.out_channels, 4 if self._kind == 'lstm' else 3))
        return ids, groups, time, recurrent

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        ids, groups, time, recurrent = self._preflight(x, index, ptr, dim_size, dim, max_num_elements)
        positions = [x.shape[0]] * (groups * time)
        used = {}
        for row, group in enumerate(ids):
            position = used.get(group, 0)
            used[group] = position + 1
            if position < time: positions[group * time + position] = row
        padded = cat([x, Tensor.zeros([1, self.in_channels])], 0).index_select(0, positions).reshape([groups, time, self.in_channels])
        output, _ = recurrent(padded)
        return output.narrow(1, time - 1, 1).squeeze(1)


class LSTMAggregation(_RecurrentAggregation):
    """Aggregate sorted CPU f32 groups with a public native LSTM.

    Order matters. All groups run through the common zero-padded sequence,
    including padding after short groups, and return its last timestep. An
    explicit max_num_elements truncates each group to its first rows. Empty
    input requires an explicit positive max_num_elements. ptr rejects.
    LSTM kwargs include stacked/bidirectional layers, projections and saved
    interlayer dropout; batch_first is fixed and inputs are rank two.
    """
    _builtin_init = True
    _kind = 'lstm'

    def __init__(self, in_channels, out_channels, **kwargs):
        self._build(in_channels, out_channels, LSTM, kwargs)


class GRUAggregation(_RecurrentAggregation):
    """Aggregate sorted groups with GRU using LSTMAggregation's padding rules."""
    _builtin_init = True
    _kind = 'gru'

    def __init__(self, in_channels, out_channels, **kwargs):
        self._build(in_channels, out_channels, GRU, kwargs)
