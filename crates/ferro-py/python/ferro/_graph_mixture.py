"""Degree mixtures, feature-steered kernels and averaged spectral propagation.

CPU f32 rank-two features, explicit COO topology and existing native first-order
operations. No sparse tensor, lazy width, feature cache or capture support.
"""
import math

from ._native import Tensor, cat, no_grad, scatter
from ._graph_utils import _capacity, _count, _edge_parts
from ._graph_conv import _configure, _flags, _options, _pair, _positive, _reset_linear
from ._graph_conditioned import _eager, _reduction
from ._graph_extra_conv import _gcn_weights
from ._graph_residual import _widths
from ._graph_spectral import _graph, _hops, _propagate
from .nn import Linear, Module, ModuleList, Parameter


class MFConv(Module):
    """Degree-specific neighbor and root maps, clamped at max_degree.

    Duplicate edges each contribute one to incoming degree. Missing target
    features omit the root map; explicit size then supplies the target count.
    All degree branches participate in the graph, including empty selections.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    max_degree: int
    aggr: str

    def __init__(self, in_channels, out_channels, max_degree=10, bias=True, **kwargs):
        in_channels, widths = _widths(in_channels)
        _positive(out_channels, 'out_channels')
        _count(max_degree, 'max_degree')
        _flags(bias=bias)
        _capacity((max_degree + 1, sum(widths) + 1, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels,
                   max_degree=max_degree, aggr=_reduction(kwargs.pop('aggr', 'add'), kwargs))
        self._widths = widths
        self.lins_l = ModuleList(Linear(widths[0], out_channels, bias=bias) for _ in range(max_degree + 1))
        self.lins_r = ModuleList(Linear(widths[1], out_channels, bias=False) for _ in range(max_degree + 1))
        self.reset_parameters()

    def reset_parameters(self):
        for left, right in zip(self.lins_l, self.lins_r):
            _reset_linear(left)
            _reset_linear(right)

    def forward(self, x, edge_index, size=None):
        _eager()
        source, target, size = _pair(x, size, *self._widths)
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size:
            _capacity((count + 1,), 8)
        _capacity((len(src), self._widths[0]))
        _capacity((size[1], max(self._widths[0], self.out_channels)))
        degree = [0] * size[1]
        for node in dst:
            degree[node] = min(self.max_degree, degree[node] + 1)
        neighbors = scatter(source.index_select(0, src), dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        rows, values = [], []
        for index, (left, right) in enumerate(zip(self.lins_l, self.lins_r)):
            selected = [node for node, count in enumerate(degree) if count == index]
            value = left(neighbors.index_select(0, selected))
            if target is not None:
                value = value + right(target.index_select(0, selected))
            rows.extend(selected)
            values.append(value)
        return scatter(cat(values, 0), rows, size[1], 'sum')


class FeaStConv(Module):
    """Feature-steered head mixtures with a headwise softmax per edge.

    Source and target feature widths must agree. Loop insertion replaces all
    diagonal edges and inserts one for each target, as PyG does; a rectangular
    graph with more targets than sources must disable loop insertion.
    Empty messages use explicit output widths; this also handles the empty
    reshape case rejected by PyG 2.8.1's inferred message dimension.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    heads: int
    add_self_loops: bool
    aggr: str

    def __init__(self, in_channels, out_channels, heads=1, add_self_loops=True, bias=True, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'), (heads, 'heads')):
            _positive(value, name)
        _flags(add_self_loops=add_self_loops, bias=bias)
        _capacity((in_channels, heads, out_channels + 1))
        _configure(self, in_channels=in_channels, out_channels=out_channels, heads=heads,
                   add_self_loops=add_self_loops, aggr=_reduction(kwargs.pop('aggr', 'mean'), kwargs))
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.u = Linear(in_channels, heads, bias=False)
        self.c = Parameter(Tensor.zeros([heads]))
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin)
        _reset_linear(self.u)
        with no_grad():
            self.c.tensor().copy_(Tensor.randn([self.heads]) * .1)
            if self.bias is not None:
                self.bias.tensor().copy_(Tensor.randn([self.out_channels]) * .1)

    def forward(self, x, edge_index):
        _eager()
        source, target, size = _pair(x, None, self.in_channels, self.in_channels)
        if target is None:
            raise ValueError('FeaStConv requires target features')
        if self.add_self_loops and size[1] > size[0]:
            raise ValueError('FeaStConv loop insertion requires at least as many sources as targets')
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size:
            _capacity((count + 1,), 8)
        count = len(src) + (size[1] if self.add_self_loops else 0)
        _capacity((2, count), 8)
        _capacity((count, self.heads, self.out_channels))
        _capacity((count, self.in_channels))
        _capacity((size[1], self.out_channels))
        if self.add_self_loops:
            selected = [i for i, (s, d) in enumerate(zip(src, dst)) if s != d]
            src = [src[i] for i in selected] + list(range(size[1]))
            dst = [dst[i] for i in selected] + list(range(size[1]))
        source_values = source.index_select(0, src)
        q = (self.u(source_values - target.index_select(0, dst)) + self.c.tensor()).softmax(1)
        projected = self.lin(source_values).reshape([len(src), self.heads, self.out_channels])
        messages = (projected * q.unsqueeze(2)).sum_dim(1, False)
        output = scatter(messages, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        return output if self.bias is None else output + self.bias.tensor()


class SSGConv(Module):
    """Initial alpha term plus the average of K normalized adjacency powers.

    K=0 returns lin(alpha*x), matching PyG's empty propagation loop. Alpha is
    any finite scalar; cached=True rejects rather than detaching stale features.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    alpha: float
    K: int
    cached: bool
    add_self_loops: bool
    aggr: str

    def __init__(self, in_channels, out_channels, alpha, K=1, cached=False, add_self_loops=True, bias=True, **kwargs):
        _positive(in_channels, 'in_channels')
        _positive(out_channels, 'out_channels')
        if type(alpha) not in (int, float) or not math.isfinite(alpha):
            raise ValueError('alpha must be a finite scalar')
        _hops(K)
        _flags(cached=cached, add_self_loops=add_self_loops, bias=bias)
        if cached:
            raise NotImplementedError('SSGConv supports cached=False only; no feature cache is retained')
        _capacity((in_channels, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, alpha=float(alpha),
                   K=K, cached=cached, add_self_loops=add_self_loops, aggr=_options(kwargs, 'add'))
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin)

    def forward(self, x, edge_index, edge_weight=None):
        nodes, src, dst = _graph(x, edge_index, edge_weight, self.in_channels, self.out_channels, self.add_self_loops)
        src, dst, weight = _gcn_weights(src, dst, nodes, edge_weight, self.add_self_loops)
        output = x * self.alpha
        for _ in range(self.K):
            x = _propagate(x, src, dst, nodes, weight)
            output = output + x * ((1. - self.alpha) / self.K)
        return self.lin(output)
