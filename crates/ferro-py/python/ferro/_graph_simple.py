"""Native sparse light propagation and composable simple aggregation."""
from ._native import Tensor, cat, scatter
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _pair
from ._graph_conditioned import _eager, _reduction
from ._graph_extra_conv import _gcn_weights, _weights
from ._graph_gen import _aggregation
from .nn import Module


class LGConv(Module):
    """LightGCN propagation, with optional symmetric weighted normalization.

    CPU f32 rank-two features and explicit homogeneous COO topology. No loops
    are inserted and duplicates remain independent. Normalized degrees must
    be nonsingular for defined edge-weight gradients. Aggregation supports
    add/sum/mean/max/min with native actual-neighbor extrema tie gradients.
    """
    _builtin_init = True
    normalize: bool
    aggr: str

    def __init__(self, normalize=True, **kwargs):
        _flags(normalize=normalize)
        _configure(self, normalize=normalize, aggr=_reduction(kwargs.pop('aggr', 'add'), kwargs))

    def reset_parameters(self):
        """No parameters or graph cache are retained."""

    def forward(self, x, edge_index, edge_weight=None):
        _eager()
        _features(x, 'x')
        nodes, width = x.shape
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        _weights(edge_weight, len(src))
        _capacity((nodes + 1,), 8)
        for count in (nodes, len(src)):
            _capacity((count, width))
        if self.normalize:
            src, dst, edge_weight = _gcn_weights(src, dst, nodes, edge_weight, False)
        message = x.index_select(0, src)
        if edge_weight is not None:
            message = message * edge_weight.unsqueeze(1)
        return scatter(message, dst, nodes, 'sum' if self.aggr == 'add' else self.aggr)


class SimpleConv(Module):
    """Weighted sparse messages with an optional sum, cat or self-loop root.

    CPU f32 homogeneous or bipartite features, including missing target data
    with explicit size. Cat places target features before aggregated messages.
    self_loop requires homogeneous Tensor input and appends every node loop,
    preserving existing loops and filling new edge weights with one. Known
    aggregation names/lists/Modules use registered native aggregation state.
    Sparse layouts, arbitrary aggregation callbacks and capture reject.
    """
    _builtin_init = True
    combine_root: object

    def __init__(self, aggr='sum', combine_root=None, **kwargs):
        if combine_root not in (None, 'sum', 'cat', 'self_loop'):
            raise ValueError('combine_root must be None, sum, cat or self_loop')
        options = kwargs.pop('aggr_kwargs', None)
        if options is not None and type(options) is not dict:
            raise TypeError('aggr_kwargs must be a dictionary or None')
        _options(kwargs, 'add')
        aggregate = _aggregation(aggr, {} if options is None else dict(options))
        _configure(self, combine_root=combine_root)
        self.aggr_module = aggregate

    def reset_parameters(self):
        self.aggr_module.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        _eager()
        requested_size = size
        source, target, size = _pair(x, size)
        if self.combine_root == 'self_loop' and (not isinstance(x, Tensor) or (requested_size is not None and requested_size[0] != requested_size[1])):
            raise ValueError('self_loop root combination requires homogeneous input and size')
        src, dst, _ = _edge_parts(edge_index, size=size)
        _weights(edge_weight, len(src))
        loops = size[0] if self.combine_root == 'self_loop' else 0
        _capacity((size[1] + 1,), 8)
        _capacity((2, len(src) + loops), 8)
        width = source.shape[1]
        output_width = self.aggr_module.get_out_channels(width)
        if target is not None and self.combine_root == 'sum':
            target_width = target.shape[1]
            if output_width != target_width and output_width != 1 and target_width != 1:
                raise ValueError('sum root width must broadcast with aggregated features')
            output_width = target_width if output_width == 1 else output_width
        elif target is not None and self.combine_root == 'cat':
            output_width += target.shape[1]
        _capacity((len(src) + loops, width))
        _capacity((size[1], output_width))
        if loops:
            ids = list(range(loops))
            src, dst = src + ids, dst + ids
            if edge_weight is not None:
                edge_weight = cat([edge_weight, Tensor.ones([loops])], 0)
        message = source.index_select(0, src)
        if edge_weight is not None:
            message = message * edge_weight.unsqueeze(1)
        output = self.aggr_module(message, dst, dim_size=size[1])
        if target is not None:
            if self.combine_root == 'sum':
                output = output + target
            elif self.combine_root == 'cat':
                output = cat([target, output], 1)
        return output
