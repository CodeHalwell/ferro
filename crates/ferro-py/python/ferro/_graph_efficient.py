"""Adaptive graph basis filters built from native reductions and matrix products."""
from ._native import Tensor, no_grad, scatter, stack
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_extra_conv import _gcn_weights
from .nn import Linear, Module, Parameter


class EGConv(Module):
    """Learn per-node mixtures of graph aggregators and shared basis filters.

    Eager CPU f32 homogeneous COO input; sum/mean/symnorm/max/min/var/std are
    supported. COO std is sqrt(clamp(var,1e-5)), matching PyG. With symnorm,
    added loops cover every feature row. Without it, PyG infers added-loop
    count from topology, so trailing isolated rows retain no loop. Extrema use
    native actual-neighbor tie gradients. cached=True and sparse layouts reject.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    aggregators: tuple
    num_heads: int
    num_bases: int
    cached: bool
    add_self_loops: bool

    def __init__(self, in_channels, out_channels, aggregators=None, num_heads=8, num_bases=4,
                 cached=False, add_self_loops=True, bias=True, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'), (num_heads, 'num_heads'), (num_bases, 'num_bases')):
            _positive(value, name)
        _flags(cached=cached, add_self_loops=add_self_loops, bias=bias)
        if cached:
            raise NotImplementedError('EGConv supports cached=False only')
        if out_channels % num_heads:
            raise ValueError('out_channels must be divisible by num_heads')
        if aggregators is not None and type(aggregators) not in (list, tuple):
            raise TypeError('aggregators must be a list, tuple or None')
        aggregators = tuple(aggregators) if aggregators else ('symnorm',)
        if any(type(value) is not str or value not in ('sum', 'mean', 'symnorm', 'min', 'max', 'var', 'std') for value in aggregators):
            raise ValueError('unsupported EGConv aggregator')
        _options(kwargs, 'add')
        width = out_channels // num_heads
        _capacity((in_channels, width, num_bases))
        _capacity((in_channels, num_heads, num_bases, len(aggregators)))
        _capacity((out_channels,))
        _configure(self, in_channels=in_channels, out_channels=out_channels, aggregators=aggregators,
                   num_heads=num_heads, num_bases=num_bases, cached=cached, add_self_loops=add_self_loops)
        self.bases_lin = Linear(in_channels, width * num_bases, bias=False)
        self.comb_lin = Linear(in_channels, num_heads * num_bases * len(aggregators))
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.bases_lin, glorot=True)
        _reset_linear(self.comb_lin)
        if self.bias is not None:
            with no_grad(): self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, edge_index):
        _eager()
        _features(x, 'x', self.in_channels)
        nodes, width = x.shape[0], self.out_channels // self.num_heads
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        loops = nodes if 'symnorm' in self.aggregators else max(src + dst, default=-1) + 1
        _capacity((nodes + 1,), 8)
        _capacity((2, len(src) + (loops if self.add_self_loops else 0)), 8)
        _capacity((len(src) + (loops if self.add_self_loops else 0), width, self.num_bases))
        _capacity((nodes, len(self.aggregators), self.num_bases, max(width, self.num_heads)))
        _capacity((nodes, self.out_channels))
        symnorm_weight = None
        if 'symnorm' in self.aggregators:
            src, dst, symnorm_weight = _gcn_weights(src, dst, nodes, None, self.add_self_loops)
        elif self.add_self_loops:
            keep = [index for index, (source, target) in enumerate(zip(src, dst)) if source != target]
            src, dst = [src[index] for index in keep] + list(range(loops)), [dst[index] for index in keep] + list(range(loops))
        bases = self.bases_lin(x)
        weightings = self.comb_lin(x).reshape([nodes, self.num_heads, self.num_bases * len(self.aggregators)])
        message = bases.index_select(0, src)
        outputs = []
        for aggr in self.aggregators:
            if aggr == 'symnorm':
                value = scatter(message * symnorm_weight.unsqueeze(1), dst, nodes, 'sum')
            elif aggr in ('var', 'std'):
                mean = scatter(message, dst, nodes, 'mean')
                value = scatter(message.square(), dst, nodes, 'mean') - mean.square()
                if aggr == 'std': value = value.clamp(1e-5, float('inf')).sqrt()
            else:
                value = scatter(message, dst, nodes, aggr)
            outputs.append(value)
        aggregated = stack(outputs, 1) if len(outputs) > 1 else outputs[0]
        aggregated = aggregated.reshape([nodes, len(outputs) * self.num_bases, width])
        output = weightings.bmm(aggregated).reshape([nodes, self.out_channels])
        return output if self.bias is None else output + self.bias.tensor()
