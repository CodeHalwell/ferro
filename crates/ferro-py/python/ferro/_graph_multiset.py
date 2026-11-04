"""Graph multiset pooling with registered native set-attention blocks."""
import math

from ._native import Tensor, cat
from ._graph_aggregation import _Aggregation, _layout
from ._graph_conv import _configure, _flags, _positive
from ._graph_utils import _capacity, _count
from ._graph_set_transformer import _SetPMA, _SetSAB
from .nn import ModuleList


class GraphMultisetTransformer(_Aggregation):
    """Pool to k learned representatives, mix them, then pool to one vector.

    CPU float32 rank-two inputs require sorted groups; ptr and capture reject.
    max_num_elements truncates before zero padding. Empty groups retain learned
    seed readouts. Empty inputs need positive dim_size and an explicit positive
    padding length. Native attention dropout streams are saved Module buffers.
    Unlike SetTransformerAggregation, this operator does not sanitize nonfinite
    results, matching PyG's graph-multiset forward.
    """
    _builtin_init = True
    channels: int
    k: int
    num_encoder_blocks: int
    heads: int
    layer_norm: bool
    dropout: float

    def __init__(self, channels, k, num_encoder_blocks=1, heads=1, layer_norm=False, dropout=0.):
        for value, name in ((channels, 'channels'), (k, 'k'), (heads, 'heads')): _positive(value, name)
        _count(num_encoder_blocks, 'num_encoder_blocks')
        _flags(layer_norm=layer_norm)
        if channels % heads: raise ValueError('channels must be divisible by heads')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        _capacity((num_encoder_blocks + 2, 6, channels, channels))
        _capacity((k + 1, channels))
        _configure(self, channels=channels, k=k, num_encoder_blocks=num_encoder_blocks,
                   heads=heads, layer_norm=layer_norm, dropout=float(dropout))
        self.pma1 = _SetPMA(channels, k, heads, layer_norm, float(dropout))
        self.encoders = ModuleList([_SetSAB(channels, heads, layer_norm, float(dropout)) for _ in range(num_encoder_blocks)])
        self.pma2 = _SetPMA(channels, 1, heads, layer_norm, float(dropout))

    def reset_parameters(self):
        self.pma1.reset_parameters()
        for encoder in self.encoders: encoder.reset_parameters()
        self.pma2.reset_parameters()

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        if in_channels != self.channels: raise ValueError('graph multiset input width must match channels')
        return self.channels

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        if ptr is not None: raise NotImplementedError('graph multiset aggregation requires sorted index input; ptr is unsupported')
        ids, groups = _layout(x, index, None, dim_size, dim)
        self.get_out_channels(x.shape[1])
        if any(left > right for left, right in zip(ids, ids[1:])): raise ValueError('graph multiset aggregation requires sorted indices')
        counts = {}
        for group in ids: counts[group] = counts.get(group, 0) + 1
        if not groups: raise ValueError('graph multiset aggregation requires at least one group')
        time = max(counts.values(), default=0) if max_num_elements is None else _count(max_num_elements, 'max_num_elements')
        if not time: raise ValueError('graph multiset aggregation requires a positive padding length')
        _capacity((groups, time), 8)
        _capacity((x.shape[0] + 1, self.channels))
        _capacity((groups, max(time, self.k), self.channels))
        for query, source in ((self.k, time), (1, self.k)):
            _capacity((groups, self.heads, query, source))
        if self.num_encoder_blocks: _capacity((groups, self.heads, self.k, self.k))
        return ids, groups, time

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        ids, groups, time = self._preflight(x, index, ptr, dim_size, dim, max_num_elements)
        positions = [x.shape[0]] * (groups * time)
        used = {}
        for row, group in enumerate(ids):
            position = used.get(group, 0)
            used[group] = position + 1
            if position < time: positions[group * time + position] = row
        mask = Tensor.from_bool([position != x.shape[0] for position in positions], [groups, time])
        output = cat([x, Tensor.zeros([1, self.channels])], 0).index_select(0, positions).reshape([groups, time, self.channels])
        output = self.pma1(output, mask)
        for encoder in self.encoders: output = encoder(output)
        return self.pma2(output).squeeze(1)
