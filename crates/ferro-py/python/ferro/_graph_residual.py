"""Residual gates, cosine attention and local-difference graph propagation.

CPU f32 rank-two features and explicit COO topology use native first-order
operations. No implicit topology cache, sparse layout or dynamic search.
"""
from ._native import Tensor, cat, no_grad, scatter, segment_softmax
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear
from ._graph_extra_conv import _weights
from ._graph_conditioned import _eager
from ._graph_spectral import _activation
from .nn import Linear, Module, Parameter


def _widths(in_channels):
    if type(in_channels) is int:
        widths = (in_channels, in_channels)
    elif isinstance(in_channels, (tuple, list)) and len(in_channels) == 2:
        widths = tuple(in_channels)
        in_channels = widths
    else:
        raise ValueError('in_channels must be a positive integer or source/target pair')
    for value in widths:
        _positive(value, 'in_channels')
    return in_channels, widths


class ResGatedGraphConv(Module):
    """Learned source values multiplied by a learned target/source gate.

    With edge_dim, edge features enter all key/query/value projections. Without
    edge_dim, edge_attr must be absent. Supplied activation Modules register
    directly; known relu/tanh/sigmoid/gelu functions or names are also accepted.
    The bias flag controls the final bias; key/query/value biases remain, as in
    PyG. Root transformation can be independently disabled.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    edge_dim: object
    root_weight: bool
    aggr: str

    def __init__(self, in_channels, out_channels, act='sigmoid', edge_dim=None,
                 root_weight=True, bias=True, **kwargs):
        in_channels, widths = _widths(in_channels)
        _positive(out_channels, 'out_channels')
        _flags(root_weight=root_weight, bias=bias)
        if edge_dim is not None:
            _positive(edge_dim, 'edge_dim')
        activation = _activation(act)
        if activation is None:
            raise TypeError('ResGatedGraphConv requires a gating activation')
        extra = 0 if edge_dim is None else edge_dim
        _capacity((4, max(widths) + extra, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim,
                   root_weight=root_weight, aggr=_options(kwargs, 'add'))
        self._widths = widths
        self.act = activation
        self.lin_key = Linear(widths[1] + extra, out_channels)
        self.lin_query = Linear(widths[0] + extra, out_channels)
        self.lin_value = Linear(widths[0] + extra, out_channels)
        self.lin_skip = Linear(widths[1], out_channels, bias=False) if root_weight else None
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.lin_key, self.lin_query, self.lin_value, self.lin_skip):
            if layer is not None:
                _reset_linear(layer)
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, edge_index, edge_attr=None):
        _eager()
        source, target, size = _pair(x, None, *self._widths)
        if target is None:
            raise ValueError('ResGatedGraphConv requires target features for the gate')
        src, dst, _ = _edge_parts(edge_index, size=size)
        _capacity((size[1] + 1,), 8)
        if self.edge_dim is None:
            if edge_attr is not None:
                raise ValueError('edge_attr must be absent when edge_dim is None')
        else:
            _features(edge_attr, 'edge_attr', self.edge_dim)
            if edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must have shape [E, edge_dim]')
        for count in (*size, len(src)):
            _capacity((count, 3, max(*self._widths, self.out_channels) + (self.edge_dim or 0)))
        if edge_attr is None:
            key = self.lin_key(target).index_select(0, dst)
            query = self.lin_query(source).index_select(0, src)
            value = self.lin_value(source).index_select(0, src)
        else:
            target_edge = cat([target.index_select(0, dst), edge_attr], 1)
            source_edge = cat([source.index_select(0, src), edge_attr], 1)
            key, query, value = self.lin_key(target_edge), self.lin_query(source_edge), self.lin_value(source_edge)
        output = scatter(self.act(key + query) * value, dst, size[1], 'sum')
        if self.lin_skip is not None:
            output = output + self.lin_skip(target)
        return output if self.bias is None else output + self.bias.tensor()


class AGNNConv(Module):
    """Cosine graph attention with a learned or buffered scalar temperature.

    Duplicate non-self edges remain independent. add_self_loops removes all
    supplied diagonal edges and inserts exactly one per node. Native segment
    softmax requires finite logits. The constant-beta reset is a no-op, as PyG.
    """
    _builtin_init = True
    requires_grad: bool
    add_self_loops: bool
    aggr: str

    def __init__(self, requires_grad=True, add_self_loops=True, **kwargs):
        _flags(requires_grad=requires_grad, add_self_loops=add_self_loops)
        _configure(self, requires_grad=requires_grad, add_self_loops=add_self_loops, aggr=_options(kwargs, 'add'))
        if requires_grad:
            self.beta = Parameter(Tensor.ones([1]))
        else:
            self.register_buffer('beta', Tensor.ones([1]))

    def reset_parameters(self):
        if self.requires_grad:
            with no_grad():
                self.beta.tensor().copy_(Tensor.ones([1]))

    def forward(self, x, edge_index):
        _eager()
        _features(x, 'x')
        nodes, width = x.shape
        _capacity((nodes + 1,), 8)
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        _capacity((2, len(src) + (nodes if self.add_self_loops else 0)), 8)
        _capacity((len(src) + (nodes if self.add_self_loops else 0), width))
        if self.add_self_loops:
            keep = [i for i, (source, target) in enumerate(zip(src, dst)) if source != target]
            src = [src[i] for i in keep] + list(range(nodes))
            dst = [dst[i] for i in keep] + list(range(nodes))
        normalized = x.normalize(-1, 1e-12)
        cosine = (normalized.index_select(0, src) * normalized.index_select(0, dst)).sum_dim(-1, False)
        beta = self.beta.tensor() if self.requires_grad else self.beta
        attention = segment_softmax(beta * cosine, dst, nodes)
        messages = x.index_select(0, src) * attention.reshape([len(src), 1])
        return scatter(messages, dst, nodes, 'sum')


class LEConv(Module):
    """Projected source-minus-target messages plus a target skip projection.

    Signed scalar edge weights multiply each difference. Source and target
    channels may differ for a bipartite graph; target features are required.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    aggr: str

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        in_channels, widths = _widths(in_channels)
        _positive(out_channels, 'out_channels')
        _flags(bias=bias)
        _capacity((3, max(widths), out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, aggr=_options(kwargs, 'add'))
        self._widths = widths
        self.lin1 = Linear(widths[0], out_channels, bias=bias)
        self.lin2 = Linear(widths[1], out_channels, bias=False)
        self.lin3 = Linear(widths[1], out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.lin1, self.lin2, self.lin3):
            _reset_linear(layer)

    def forward(self, x, edge_index, edge_weight=None):
        _eager()
        source, target, size = _pair(x, None, *self._widths)
        if target is None:
            raise ValueError('LEConv requires target features')
        src, dst, _ = _edge_parts(edge_index, size=size)
        _capacity((size[1] + 1,), 8)
        _weights(edge_weight, len(src))
        for count in (*size, len(src)):
            _capacity((count, self.out_channels))
        message = self.lin1(source).index_select(0, src) - self.lin2(target).index_select(0, dst)
        if edge_weight is not None:
            message = message * edge_weight.reshape([len(src), 1])
        return scatter(message, dst, size[1], 'sum') + self.lin3(target)
