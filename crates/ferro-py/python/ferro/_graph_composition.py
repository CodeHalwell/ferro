"""Graph representation composition and general learned neighbor messages."""
import math

from ._native import Tensor, cat, no_grad, scatter, segment_softmax, stack
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _pair, _positive, _reset_linear
from ._graph_conditioned import _eager, _reduction
from ._graph_residual import _widths
from .nn import Identity, Linear, LSTM, Module, Parameter


class JumpingKnowledge(Module):
    """Combine rank-two CPU f32 layer features by cat, max or LSTM attention.

    Max selects the first winning layer, including ties, as Torch does. In
    LSTM mode num_layers determines hidden width; the actual input list can
    contain a different positive number of layers, matching PyG.
    """
    _builtin_init = True
    mode: str
    channels: object
    num_layers: object

    def __init__(self, mode, channels=None, num_layers=None):
        if type(mode) is not str or mode.lower() not in ('cat', 'max', 'lstm'):
            raise ValueError('mode must be cat, max or lstm')
        mode = mode.lower()
        if mode == 'lstm':
            _positive(channels, 'channels')
            _positive(num_layers, 'num_layers')
            hidden = num_layers * channels // 2
            _positive(hidden, 'LSTM hidden width')
            _capacity((2, 4 * hidden, channels + hidden + 2))
        else:
            channels = num_layers = None
        _configure(self, mode=mode, channels=channels, num_layers=num_layers)
        self.lstm = LSTM(channels, hidden, bidirectional=True, batch_first=True) if mode == 'lstm' else None
        self.att = Linear(2 * hidden, 1) if mode == 'lstm' else None
        self.reset_parameters()

    def reset_parameters(self):
        if self.lstm is not None:
            self.lstm.reset_parameters()
            _reset_linear(self.att)

    def forward(self, xs):
        _eager()
        if not isinstance(xs, (tuple, list)) or not xs:
            raise ValueError('xs must be a nonempty list or tuple of feature tensors')
        for value in xs:
            _features(value, 'layer features', self.channels)
        nodes, width = xs[0].shape
        if any(value.shape[0] != nodes or (self.mode != 'cat' and value.shape[1] != width) for value in xs):
            raise ValueError('layer feature rows must agree, and max/lstm widths must agree')
        if self.mode == 'cat':
            _capacity((nodes, sum(value.shape[1] for value in xs)))
            return cat(xs, 1)
        _capacity((nodes, width, len(xs)))
        if self.mode == 'max':
            _capacity((nodes, width), 8)
            combined = stack(xs, 2)
            return combined.gather(2, combined.argmax(2, True)).squeeze(2)
        _capacity((nodes, len(xs), 2 * self.lstm.hidden_size))
        combined = stack(xs, 1)
        attention, _ = self.lstm(combined)
        weights = self.att(attention).squeeze(2).softmax(1)
        return (combined * weights.unsqueeze(2)).sum_dim(1, False)


class GeneralConv(Module):
    """Learned source/symmetric messages, optional attention, and a target skip.

    CPU f32 rank-two homogeneous or bipartite features. Symmetric messages and
    dot-product attention reuse source projections on targets, requiring equal
    feature widths. A tuple in_channels always builds a learned skip, matching
    PyG. Additive/dot attention uses finite native segment-softmax logits.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    in_edge_channels: object
    aggr: str
    skip_linear: bool
    directed_msg: bool
    heads: int
    attention: bool
    attention_type: str
    normalize_l2: bool

    def __init__(self, in_channels, out_channels, in_edge_channels=None, aggr='add', skip_linear=False,
                 directed_msg=True, heads=1, attention=False, attention_type='additive',
                 l2_normalize=False, bias=True, **kwargs):
        in_channels, widths = _widths(in_channels)
        _positive(out_channels, 'out_channels')
        _positive(heads, 'heads')
        _flags(skip_linear=skip_linear, directed_msg=directed_msg, attention=attention, l2_normalize=l2_normalize, bias=bias)
        if attention_type not in ('additive', 'dot_product'):
            raise ValueError('attention_type must be additive or dot_product')
        if (not directed_msg or (attention and attention_type == 'dot_product')) and widths[0] != widths[1]:
            raise ValueError('symmetric or dot-product messages require equal source/target widths')
        if in_edge_channels is not None:
            _positive(in_edge_channels, 'in_edge_channels')
            _capacity((in_edge_channels, heads, out_channels))
        _capacity((max(widths), heads, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, in_edge_channels=in_edge_channels,
                   aggr=_reduction(aggr, kwargs), skip_linear=skip_linear, directed_msg=directed_msg, heads=heads,
                   attention=attention, attention_type=attention_type, normalize_l2=l2_normalize)
        self._widths = widths
        self.lin_msg = Linear(widths[0], heads * out_channels, bias=bias)
        if not directed_msg:
            self.lin_msg_i = Linear(widths[0], heads * out_channels, bias=bias)
            _reset_linear(self.lin_msg_i)
        self.lin_self = Linear(widths[1], out_channels, bias=bias) if skip_linear or in_channels != out_channels else Identity()
        if in_edge_channels is not None:
            self.lin_edge = Linear(in_edge_channels, heads * out_channels, bias=bias)
        if attention:
            if attention_type == 'additive':
                self.att_msg = Parameter(Tensor.zeros([1, heads, out_channels]))
            else:
                self.register_buffer('scaler', Tensor([math.sqrt(out_channels)], []))
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin_msg)
        # PyG leaves lin_msg_i at its independently initialized values on reset.
        if isinstance(self.lin_self, Linear):
            _reset_linear(self.lin_self)
        if self.in_edge_channels is not None:
            _reset_linear(self.lin_edge)
        if self.attention and self.attention_type == 'additive':
            with no_grad():
                bound = math.sqrt(6. / (self.heads + self.out_channels))
                self.att_msg.tensor().copy_((Tensor.rand([1, self.heads, self.out_channels]) * 2 - 1) * bound)

    def _message(self, source, target, edge_attr):
        output = self.lin_msg(source)
        if not self.directed_msg:
            output = output + self.lin_msg_i(target)
        if edge_attr is not None:
            output = output + self.lin_edge(edge_attr)
        return output

    def forward(self, x, edge_index, edge_attr=None, size=None):
        _eager()
        source, target, size = _pair(x, size, *self._widths)
        if target is None:
            raise ValueError('GeneralConv requires target features for its skip path')
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size:
            _capacity((count + 1,), 8)
        _capacity((len(src), max(self._widths)))
        _capacity((len(src), self.heads, self.out_channels))
        _capacity((size[1], self.heads, self.out_channels))
        if edge_attr is not None:
            if self.in_edge_channels is None:
                raise ValueError('edge attributes require in_edge_channels at construction')
            _features(edge_attr, 'edge_attr', self.in_edge_channels)
            if edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must contain one row per edge')
        source, neighbors = source.index_select(0, src), target.index_select(0, dst)
        messages = self._message(source, neighbors, edge_attr).reshape([len(src), self.heads, self.out_channels])
        if self.attention:
            if self.attention_type == 'dot_product':
                reverse = self._message(neighbors, source, edge_attr).reshape([len(src), self.heads, self.out_channels])
                alpha = (reverse * messages).sum_dim(2, False) / self.scaler
            else:
                alpha = (messages * self.att_msg.tensor()).sum_dim(2, False)
            alpha = alpha.leaky_relu(.2)
            messages = messages * segment_softmax(alpha, dst, size[1]).unsqueeze(2)
        output = scatter(messages.reshape([len(src), self.heads * self.out_channels]), dst,
                         size[1], 'sum' if self.aggr == 'add' else self.aggr)
        output = output.reshape([size[1], self.heads, self.out_channels]).mean_dim(1, False) + self.lin_self(target)
        return output.normalize(-1, 1e-12) if self.normalize_l2 else output
