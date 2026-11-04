"""Sorted sparse-group readout using native ordering and differentiable gather."""
from ._native import cat
from ._graph_utils import _capacity, _count
from ._graph_conv import _configure
from ._graph_aggregation import _Aggregation, _layout


class SortAggregation(_Aggregation):
    """Keep k rows sorted by their final feature, then flatten each group.

    CPU float32 rank-two inputs and sorted group indices are required; ptr
    rejects. max_num_elements truncates to each group's first rows before
    sorting. Missing rows are padded using PyG's detached min(x)-1 sentinel
    and all entries equal to that sentinel become zero. Empty inputs reject.
    Equal sort keys retain input order, including groups where Torch's
    unspecified sort tie order differs. Selection is discrete; selected
    feature gradients stay native. k=0 returns a connected empty readout.
    """
    _builtin_init = True
    k: int

    def __init__(self, k):
        _count(k, 'k')
        _capacity((k,))
        _configure(self, k=k)

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        if in_channels == 0:
            raise ValueError('SortAggregation requires at least one feature channel')
        return _count(self.k * in_channels, 'aggregation output width')

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        if ptr is not None:
            raise NotImplementedError('SortAggregation requires sorted index input; ptr is unsupported')
        ids, groups = _layout(x, index, None, dim_size, dim)
        width = x.shape[1]
        output_width = self.get_out_channels(width)
        if x.shape[0] == 0:
            raise ValueError('SortAggregation cannot reduce empty input')
        if any(left > right for left, right in zip(ids, ids[1:])):
            raise ValueError('SortAggregation requires sorted indices')
        counts = {}
        for group in ids: counts[group] = counts.get(group, 0) + 1
        time = max(counts.values()) if max_num_elements is None else _count(max_num_elements, 'max_num_elements')
        _capacity((groups, time, width), 8)
        _capacity((x.shape[0] + 1, width))
        _capacity((groups, max(time, self.k), width))
        _capacity((groups, output_width))
        return ids, groups, time, width

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        ids, groups, time, width = self._preflight(x, index, ptr, dim_size, dim, max_num_elements)
        if self.k == 0:
            return x.narrow(0, 0, 0).reshape([1, 0]).expand(groups, 0)
        positions = [x.shape[0]] * (groups * time)
        used = {}
        for row, group in enumerate(ids):
            position = used.get(group, 0)
            used[group] = position + 1
            if position < time: positions[group * time + position] = row
        fill = -(-x.detach()).max() - 1.
        padded = cat([x, fill.expand(1, width)], 0).index_select(0, positions).reshape([groups, time, width])
        order = padded.narrow(2, width - 1, 1).squeeze(2).topk(time, 1)[1]
        ordered = padded.gather(1, order.unsqueeze(2).expand(groups, time, width))
        if time >= self.k:
            ordered = ordered.narrow(1, 0, self.k)
        else:
            ordered = cat([ordered, fill.expand(groups, self.k - time, width)], 1)
        return ordered.masked_fill(ordered.eq(fill), 0.).reshape([groups, self.k * width])
