"""Edge-conditioned and recurrent graph layers over eager CPU f32 COO data.

All learned messages and recurrent updates use existing Ferro Modules and
tensor operations. No dynamic neighbor search or detached feature path is
introduced. Extremum ties use native scatter's actual-neighbor convention.
"""
from . import _native
from ._native import Tensor, cat, no_grad, scatter
from ._graph_utils import _capacity, _count, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear, _reset_network
from ._graph_extra_conv import _weights
from .nn import GRUCell, Linear, Module, Parameter


def _reduction(aggr, kwargs):
    if aggr not in ('add', 'sum', 'mean', 'max', 'min'):
        raise NotImplementedError('conditioned graph layers support add/sum/mean/max/min aggregation only')
    _options(kwargs, 'add')
    return aggr


def _eager():
    if _native._is_capturing():
        raise NotImplementedError('conditioned graph layers require eager execution')


def _network(nn):
    if not isinstance(nn, Module):
        raise TypeError('nn must be a Ferro Module')


class NNConv(Module):
    """PyG edge-conditioned filters with a registered edge network.

    The network must return [E, source_channels * out_channels]. Scalar edge
    attributes and arbitrary Python callables are unsupported. edge_attr=None
    is forwarded to a supplied Module that explicitly accepts it. Homogeneous
    and bipartite inputs, including missing target features, are supported.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    in_channels_l: int
    root_weight: bool
    aggr: str

    def __init__(self, in_channels, out_channels, nn, aggr='add', root_weight=True, bias=True, **kwargs):
        if type(in_channels) is int:
            widths = (in_channels, in_channels)
        elif isinstance(in_channels, (tuple, list)) and len(in_channels) == 2:
            widths = tuple(in_channels)
            in_channels = widths
        else:
            raise ValueError('in_channels must be a positive integer or source/target pair')
        for value in widths:
            _positive(value, 'in_channels')
        _positive(out_channels, 'out_channels')
        _flags(root_weight=root_weight, bias=bias)
        _network(nn)
        _capacity((max(widths), out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, in_channels_l=widths[0],
                   root_weight=root_weight, aggr=_reduction(aggr, kwargs))
        self._widths = widths
        self.nn = nn
        if root_weight:
            self.lin = Linear(widths[1], out_channels, bias=False)
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.nn)
        if self.root_weight:
            _reset_linear(self.lin)
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, edge_index, edge_attr=None, size=None):
        _eager()
        source, target, size = _pair(x, size, self._widths[0], self._widths[1] if self.root_weight else None)
        src, dst, _ = _edge_parts(edge_index, size=size)
        _capacity((size[1] + 1,), 8)
        if edge_attr is not None:
            _features(edge_attr, 'edge_attr')
            if edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must contain one row per edge')
        _capacity((len(src), self.in_channels_l, self.out_channels))
        _capacity((size[1], self.out_channels))
        weight = self.nn(edge_attr)
        _features(weight, 'edge network output', self.in_channels_l * self.out_channels)
        if weight.shape[0] != len(src):
            raise ValueError('edge network must return one filter per edge')
        message = source.index_select(0, src).unsqueeze(1).bmm(weight.reshape([len(src), self.in_channels_l, self.out_channels])).squeeze(1)
        out = scatter(message, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        if target is not None and self.root_weight:
            out = out + self.lin(target)
        return out if self.bias is None else out + self.bias.tensor()


ECConv = NNConv


class EdgeConv(Module):
    """Learned messages nn([target, source-target]) over explicit COO edges.

    Both node sets must have equal feature width. Missing targets reject;
    dynamic k-nearest-neighbor topology construction is not part of this layer.
    """
    _builtin_init = True
    aggr: str

    def __init__(self, nn, aggr='max', **kwargs):
        _network(nn)
        _configure(self, aggr=_reduction(aggr, kwargs))
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.nn)

    def forward(self, x, edge_index):
        _eager()
        source, target, size = _pair(x, None)
        if target is None or source.shape[1] != target.shape[1]:
            raise ValueError('EdgeConv requires source and target features with equal widths')
        src, dst, _ = _edge_parts(edge_index, size=size)
        _capacity((size[1] + 1,), 8)
        _capacity((len(src), 2, source.shape[1]))
        x_i = target.index_select(0, dst)
        message = self.nn(cat([x_i, source.index_select(0, src) - x_i], 1))
        _features(message, 'edge network output')
        if message.shape[0] != len(src):
            raise ValueError('edge network must return one message per edge')
        _capacity((size[1], message.shape[1]))
        return scatter(message, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)


class GatedGraphConv(Module):
    """Weighted graph messages followed by a shared public GRUCell update.

    Input width may be zero through out_channels; remaining channels are
    padded with zeros. num_layers=0 returns that padded input, leaving the
    registered weight tensor and GRU parameters unused, as in PyG.
    """
    _builtin_init = True
    out_channels: int
    num_layers: int
    aggr: str

    def __init__(self, out_channels, num_layers, aggr='add', bias=True, **kwargs):
        _positive(out_channels, 'out_channels')
        _count(num_layers, 'num_layers')
        _flags(bias=bias)
        _capacity((num_layers, out_channels, out_channels))
        _capacity((6, out_channels, out_channels))
        _configure(self, out_channels=out_channels, num_layers=num_layers, aggr=_reduction(aggr, kwargs))
        self.weight = Parameter(Tensor.zeros([num_layers, out_channels, out_channels]))
        self.rnn = GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            self.weight.tensor().copy_((Tensor.rand(list(self.weight.tensor().shape)) * 2 - 1) * self.out_channels ** -.5)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        _eager()
        _features(x, 'x')
        nodes, width = x.shape
        _capacity((nodes + 1,), 8)
        if width > self.out_channels:
            raise ValueError('input channels cannot exceed out_channels')
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        _weights(edge_weight, len(src))
        _capacity((nodes, 3, self.out_channels))
        _capacity((len(src), self.out_channels))
        if width < self.out_channels:
            x = cat([x, Tensor.zeros([nodes, self.out_channels - width])], 1)
        for iteration in range(self.num_layers):
            message = (x @ self.weight.tensor()[iteration]).index_select(0, src)
            if edge_weight is not None:
                message = message * edge_weight.reshape([len(src), 1])
            message = scatter(message, dst, nodes, 'sum' if self.aggr == 'add' else self.aggr)
            x = self.rnn(message, x)
        return x
