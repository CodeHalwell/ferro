"""Signed positive/negative graph propagation through native eager tensors."""
from ._native import cat, scatter
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _flags, _pair, _positive, _reset_linear
from ._graph_conditioned import _eager, _reduction
from .nn import Linear, Module


class SignedConv(Module):
    """Learn balanced and unbalanced representations from two COO topologies.

    The first layer consumes F features; later layers consume concatenated
    positive/negative features of width 2*F. Every layer returns 2*out_channels.
    Homogeneous and bipartite CPU f32 inputs are supported; source and target
    feature widths agree, and target features are required for the root terms.
    Duplicate edges count independently and supplied self loops remain. Native
    max/min ties share gradients among actual neighbors. No sparse layouts,
    lazy widths, higher derivatives or capture are supported.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    first_aggr: bool
    aggr: str

    def __init__(self, in_channels, out_channels, first_aggr, bias=True, **kwargs):
        _positive(in_channels, 'in_channels')
        _positive(out_channels, 'out_channels')
        _flags(first_aggr=first_aggr, bias=bias)
        aggr = _reduction(kwargs.pop('aggr', 'mean'), kwargs)
        width = in_channels if first_aggr else 2 * in_channels
        _capacity((4, width, out_channels))
        _capacity((2, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, first_aggr=first_aggr, aggr=aggr)
        self.lin_pos_l = Linear(width, out_channels, bias=False)
        self.lin_pos_r = Linear(in_channels, out_channels, bias=bias)
        self.lin_neg_l = Linear(width, out_channels, bias=False)
        self.lin_neg_r = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.lin_pos_l, self.lin_pos_r, self.lin_neg_l, self.lin_neg_r):
            _reset_linear(layer)

    def forward(self, x, pos_edge_index, neg_edge_index):
        _eager()
        width = self.in_channels if self.first_aggr else 2 * self.in_channels
        source, target, size = _pair(x, None, width, width)
        if target is None:
            raise ValueError('SignedConv requires destination features for the root terms')
        positive = _edge_parts(pos_edge_index, size=size)
        negative = _edge_parts(neg_edge_index, size=size)
        _capacity((size[1] + 1,), 8)
        for count in (*size, len(positive[0]), len(negative[0])):
            _capacity((count, max(width, 2 * self.out_channels)))
        def aggregate(value, topology):
            src, dst, _ = topology
            return scatter(value.index_select(0, src), dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        if self.first_aggr:
            pos = self.lin_pos_l(aggregate(source, positive)) + self.lin_pos_r(target)
            neg = self.lin_neg_l(aggregate(source, negative)) + self.lin_neg_r(target)
        else:
            f = self.in_channels
            source_pos, source_neg = source.narrow(1, 0, f), source.narrow(1, f, f)
            target_pos, target_neg = target.narrow(1, 0, f), target.narrow(1, f, f)
            pos = cat([aggregate(source_pos, positive), aggregate(source_neg, negative)], 1)
            neg = cat([aggregate(source_neg, positive), aggregate(source_pos, negative)], 1)
            pos = self.lin_pos_l(pos) + self.lin_pos_r(target_pos)
            neg = self.lin_neg_l(neg) + self.lin_neg_r(target_neg)
        return cat([pos, neg], 1)
