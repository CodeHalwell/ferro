"""Native set-attention readout over explicitly grouped sparse feature rows."""
import math

from ._native import Tensor, cat, no_grad
from ._graph_aggregation import _Aggregation, _layout
from ._graph_utils import _capacity, _count
from ._graph_conv import _configure, _flags, _positive, _reset_linear
from .nn import LayerNorm, Linear, Module, ModuleList, MultiheadAttention, Parameter


class _SetMAB(Module):
    _builtin_init = True
    channels: int
    heads: int
    dropout: float

    def __init__(self, channels, heads, layer_norm, dropout):
        _configure(self, channels=channels, heads=heads, dropout=dropout)
        self.attn = MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.lin = Linear(channels, channels)
        _reset_linear(self.lin)
        self.layer_norm1 = LayerNorm(channels) if layer_norm else None
        self.layer_norm2 = LayerNorm(channels) if layer_norm else None

    def reset_parameters(self):
        with no_grad():
            self.attn.in_proj_weight.tensor().copy_((Tensor.rand([3 * self.channels, self.channels]) * 2 - 1) * math.sqrt(6. / (4 * self.channels)))
            self.attn.in_proj_bias.tensor().copy_(Tensor.zeros([3 * self.channels]))
            self.attn.out_proj.bias.tensor().copy_(Tensor.zeros([self.channels]))
        _reset_linear(self.lin)
        for norm in (self.layer_norm1, self.layer_norm2):
            if norm is not None: norm.reset_parameters()

    def forward(self, x, y, x_mask=None, y_mask=None):
        output, _ = self.attn(x, y, y, key_padding_mask=None if y_mask is None else y_mask.logical_not(), need_weights=False)
        if x_mask is not None: output = output.masked_fill(x_mask.logical_not().unsqueeze(2), 0.)
        output = output + x
        if self.layer_norm1 is not None: output = self.layer_norm1(output)
        output = output + self.lin(output).relu()
        return output if self.layer_norm2 is None else self.layer_norm2(output)


class _SetSAB(Module):
    _builtin_init = True

    def __init__(self, channels, heads, layer_norm, dropout):
        _configure(self)
        self.mab = _SetMAB(channels, heads, layer_norm, dropout)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x, mask=None):
        return self.mab(x, x, mask, mask)


class _SetPMA(Module):
    _builtin_init = True
    channels: int
    num_seed_points: int

    def __init__(self, channels, num_seed_points, heads, layer_norm, dropout):
        _configure(self, channels=channels, num_seed_points=num_seed_points)
        self.lin = Linear(channels, channels)
        self.seed = Parameter(Tensor.zeros([1, num_seed_points, channels]))
        self.mab = _SetMAB(channels, heads, layer_norm, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin)
        with no_grad():
            bound = math.sqrt(6. / ((self.num_seed_points + 1) * self.channels))
            self.seed.tensor().copy_((Tensor.rand([1, self.num_seed_points, self.channels]) * 2 - 1) * bound)
        self.mab.reset_parameters()

    def forward(self, x, mask=None):
        values = self.lin(x).relu()
        return self.mab(self.seed.tensor().expand(x.shape[0], self.num_seed_points, self.channels), values, y_mask=mask)


class SetTransformerAggregation(_Aggregation):
    """PyG encoder attention, learned-seed pooling, and decoder attention.

    Uses public native MultiheadAttention/LayerNorm/Linear and saved attention
    dropout streams. Inputs are rank-two CPU float32 with sorted group indices;
    ptr and capture reject. max_num_elements truncates each group before zero
    padding; seed embeddings are concatenated or averaged. Empty groups still
    attend from learned seeds. Empty input requires positive dim_size and an
    explicit positive padding length; zero-group/zero-length batches reject.
    Seed count is positive; encoder/decoder block counts may be zero.
    """
    _builtin_init = True
    channels: int
    num_seed_points: int
    num_encoder_blocks: int
    num_decoder_blocks: int
    heads: int
    concat: bool
    layer_norm: bool
    dropout: float

    def __init__(self, channels, num_seed_points=1, num_encoder_blocks=1, num_decoder_blocks=1,
                 heads=1, concat=True, layer_norm=False, dropout=0.):
        for value, name in ((channels, 'channels'), (num_seed_points, 'num_seed_points'), (heads, 'heads')): _positive(value, name)
        for value, name in ((num_encoder_blocks, 'num_encoder_blocks'), (num_decoder_blocks, 'num_decoder_blocks')): _count(value, name)
        _flags(concat=concat, layer_norm=layer_norm)
        if channels % heads: raise ValueError('channels must be divisible by heads')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        _capacity((num_encoder_blocks + num_decoder_blocks + 1, 6, channels, channels))
        _capacity((num_seed_points, channels))
        _configure(self, channels=channels, num_seed_points=num_seed_points, num_encoder_blocks=num_encoder_blocks,
                   num_decoder_blocks=num_decoder_blocks, heads=heads, concat=concat, layer_norm=layer_norm, dropout=float(dropout))
        self.encoders = ModuleList([_SetSAB(channels, heads, layer_norm, float(dropout)) for _ in range(num_encoder_blocks)])
        self.pma = _SetPMA(channels, num_seed_points, heads, layer_norm, float(dropout))
        self.decoders = ModuleList([_SetSAB(channels, heads, layer_norm, float(dropout)) for _ in range(num_decoder_blocks)])

    def reset_parameters(self):
        for encoder in self.encoders: encoder.reset_parameters()
        self.pma.reset_parameters()
        for decoder in self.decoders: decoder.reset_parameters()

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        if in_channels != self.channels: raise ValueError('set attention input width must match channels')
        return self.channels * (self.num_seed_points if self.concat else 1)

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        if ptr is not None: raise NotImplementedError('set attention aggregation requires sorted index input; ptr is unsupported')
        ids, groups = _layout(x, index, None, dim_size, dim)
        output_width = self.get_out_channels(x.shape[1])
        if any(left > right for left, right in zip(ids, ids[1:])): raise ValueError('set attention aggregation requires sorted indices')
        counts = {}
        for group in ids: counts[group] = counts.get(group, 0) + 1
        if not groups: raise ValueError('set attention aggregation requires at least one group')
        time = max(counts.values(), default=0) if max_num_elements is None else _count(max_num_elements, 'max_num_elements')
        if not time: raise ValueError('set attention aggregation requires a positive padding length')
        _capacity((groups, time), 8)
        _capacity((x.shape[0] + 1, self.channels))
        _capacity((groups, max(time, self.num_seed_points), self.channels))
        _capacity((groups, output_width))
        scores = [(self.num_seed_points, time)]
        if self.num_encoder_blocks: scores.append((time, time))
        if self.num_decoder_blocks: scores.append((self.num_seed_points, self.num_seed_points))
        for query, source in scores:
            _capacity((groups, self.heads, query, source))
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
        for encoder in self.encoders: output = encoder(output, mask)
        output = self.pma(output, mask)
        for decoder in self.decoders: output = decoder(output)
        output = output.masked_fill(output.ne(output), 0.).clamp(-3.4028234663852886e38, 3.4028234663852886e38)
        return output.reshape([groups, self.num_seed_points * self.channels]) if self.concat else output.mean_dim(1, False)
