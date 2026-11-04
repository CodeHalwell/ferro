"""Native mixed-power graph convolution over weighted GCN normalization."""
from ._native import Tensor, cat, no_grad, scatter
from ._graph_conditioned import _eager
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear
from ._graph_extra_conv import _gcn_weights, _weights
from ._graph_utils import _capacity, _edge_parts
from .nn import Identity, Linear, Module, ModuleList, Parameter


class MixHopConv(Module):
    """Concatenate independently projected powers of normalized adjacency.

    CPU float32 features have explicit positive widths. Power order and repeated
    powers are preserved, with one registered Linear for each selected power and
    Identity placeholders for intervening powers. Edge weights stay live through
    normalization and all propagation steps. Existing weighted duplicate loops
    with inserted self-loops, sparse layouts, lazy widths and capture reject.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    powers: tuple
    add_self_loops: bool
    aggr: str

    def __init__(self, in_channels, out_channels, powers=None, add_self_loops=True, bias=True, **kwargs):
        _positive(in_channels, 'in_channels'); _positive(out_channels, 'out_channels')
        _flags(add_self_loops=add_self_loops, bias=bias)
        if powers is None: powers = [0, 1, 2]
        if not isinstance(powers, (list, tuple)) or not powers or any(type(power) is not int or power < 0 for power in powers):
            raise ValueError('powers must be a nonempty sequence of nonnegative integers')
        powers, selected = tuple(powers), set(powers)
        _capacity((max(powers) + 1,), 8)
        _capacity((len(selected), in_channels, out_channels))
        _capacity((len(powers), out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, powers=powers,
                   add_self_loops=add_self_loops, aggr=_options(kwargs, 'add'))
        self.lins = ModuleList([Linear(in_channels, out_channels, bias=False) if power in selected else Identity() for power in range(max(powers) + 1)])
        self.bias = Parameter(Tensor.zeros([len(powers) * out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lins:
            if isinstance(layer, Linear): _reset_linear(layer)
        if self.bias is not None:
            with no_grad(): self.bias.tensor().copy_(Tensor.zeros([len(self.powers) * self.out_channels]))

    def forward(self, x, edge_index, edge_weight=None):
        _eager(); _features(x, 'x', self.in_channels)
        if isinstance(edge_index, Tensor): _capacity(edge_index.shape, 8)
        src, dst, _ = _edge_parts(edge_index, num_nodes=x.shape[0])
        _weights(edge_weight, len(src))
        edges = len(src) + (x.shape[0] if self.add_self_loops else 0)
        _capacity((2, edges), 8)
        _capacity(x.shape); _capacity((edges, self.in_channels))
        _capacity((x.shape[0], len(self.powers), self.out_channels))
        src, dst, weight = _gcn_weights(src, dst, x.shape[0], edge_weight, self.add_self_loops)
        outputs = [self.lins[0](x)]
        for index in range(1, len(self.lins)):
            x = scatter(x.index_select(0, src) * weight.reshape([len(src), 1]), dst, x.shape[0], 'sum')
            outputs.append(self.lins[index].forward(x))
        out = cat([outputs[power] for power in self.powers], 1)
        return out if self.bias is None else out + self.bias.tensor()
