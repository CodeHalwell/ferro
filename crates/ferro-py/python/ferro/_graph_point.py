"""Point convolutions over caller-provided COO topology and live coordinates.

CPU f32 eager execution only. Neighbor discovery, radius/knn searches, sparse
formats and geometry preprocessing are deliberately separate from these layers.
"""
from ._native import Tensor, cat, scatter, segment_softmax
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear, _reset_network
from ._graph_conditioned import _eager, _reduction
from ._graph_residual import _widths
from .nn import Linear, Module


def _module(value, name):
    if value is not None and not isinstance(value, Module):
        raise TypeError(name + ' must be a Ferro Module or None')


def _positions(pos):
    if isinstance(pos, Tensor):
        source, target = pos, pos
    elif isinstance(pos, (tuple, list)) and len(pos) == 2:
        source, target = pos
    else:
        raise TypeError('pos must be a Tensor or (source, target) pair')
    _features(source, 'source positions')
    _features(target, 'target positions', source.shape[1])
    size = source.shape[0], target.shape[0]
    _capacity((max(size) + 1,), 8)
    return source, target, size


def _features_pair(x, size, optional, widths=None):
    if isinstance(x, Tensor) or (optional and x is None):
        source, target = x, None if optional else x
    elif isinstance(x, (tuple, list)) and len(x) == 2:
        source, target = x
    else:
        raise TypeError('x must be a Tensor or source/target pair' + (' or None' if optional else ''))
    for index, (value, name) in enumerate(((source, 'source features'), (target, 'target features'))):
        if optional and value is None:
            continue
        _features(value, name, None if widths is None else widths[index])
        if value.shape[0] != size[index]:
            raise ValueError(name + ' row count must match positions')
    return source, target


def _edges(edge_index, size, loops):
    src, dst, _ = _edge_parts(edge_index, size=size)
    diagonal = min(size) if loops else 0
    _capacity((2, len(src) + diagonal), 8)
    if loops:
        keep = [i for i, (source, target) in enumerate(zip(src, dst)) if source != target]
        src = [src[i] for i in keep] + list(range(diagonal))
        dst = [dst[i] for i in keep] + list(range(diagonal))
    return src, dst


def _output(value, rows, name, width=None):
    _features(value, name, width)
    if value.shape[0] != rows:
        raise ValueError(name + ' must preserve the input row count')
    return value


class PointNetConv(Module):
    """PointNet messages [source features, source position-target position].

    Source features may be None. Target features are validated but unused, as
    in PyG. local_nn and global_nn are registered directly. Max ties follow the
    native actual-neighbor convention; self-loop insertion uses min(Ns, Nt).
    """
    _builtin_init = True
    add_self_loops: bool
    aggr: str

    def __init__(self, local_nn=None, global_nn=None, add_self_loops=True, **kwargs):
        _module(local_nn, 'local_nn')
        _module(global_nn, 'global_nn')
        _flags(add_self_loops=add_self_loops)
        options = dict(kwargs)
        aggr = _reduction(options.pop('aggr', 'max'), options)
        _configure(self, add_self_loops=add_self_loops, aggr=aggr)
        self.local_nn, self.global_nn = local_nn, global_nn
        self.reset_parameters()

    def reset_parameters(self):
        for network in (self.local_nn, self.global_nn):
            if network is not None:
                _reset_network(network)

    def forward(self, x, pos, edge_index):
        _eager()
        source_pos, target_pos, size = _positions(pos)
        source, _ = _features_pair(x, size, True)
        src, dst = _edges(edge_index, size, self.add_self_loops)
        width = source_pos.shape[1] + (0 if source is None else source.shape[1])
        _capacity((len(src), width))
        message = source_pos.index_select(0, src) - target_pos.index_select(0, dst)
        if source is not None:
            message = cat([source.index_select(0, src), message], 1)
        if self.local_nn is not None:
            message = _output(self.local_nn(message), len(src), 'local_nn output')
        _capacity((size[1], message.shape[1]))
        result = scatter(message, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        return result if self.global_nn is None else _output(self.global_nn(result), size[1], 'global_nn output')


class PointTransformerConv(Module):
    """Per-channel point attention with learned relative-position embeddings.

    Positions enter pos_nn as target minus source. Default pos_nn maps three
    coordinate dimensions to out_channels; custom modules may accept any
    common coordinate width. Attention is normalized over incoming edges for
    each channel independently. Logits must be finite for native softmax.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    add_self_loops: bool
    position_dim: object
    aggr: str

    def __init__(self, in_channels, out_channels, pos_nn=None, attn_nn=None, add_self_loops=True, **kwargs):
        in_channels, widths = _widths(in_channels)
        _positive(out_channels, 'out_channels')
        _flags(add_self_loops=add_self_loops)
        _module(pos_nn, 'pos_nn')
        _module(attn_nn, 'attn_nn')
        _capacity((3, max(3, *widths), out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, add_self_loops=add_self_loops,
                   position_dim=3 if pos_nn is None else None, aggr=_options(kwargs, 'add'))
        self._widths = widths
        self.pos_nn = Linear(3, out_channels) if pos_nn is None else pos_nn
        self.attn_nn = attn_nn
        self.lin = Linear(widths[0], out_channels, bias=False)
        self.lin_src = Linear(widths[0], out_channels, bias=False)
        self.lin_dst = Linear(widths[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.pos_nn)
        if self.attn_nn is not None:
            _reset_network(self.attn_nn)
        for linear in (self.lin, self.lin_src, self.lin_dst):
            _reset_linear(linear)

    def forward(self, x, pos, edge_index):
        _eager()
        source_pos, target_pos, size = _positions(pos)
        if self.position_dim is not None and source_pos.shape[1] != self.position_dim:
            raise ValueError('default pos_nn requires three coordinate dimensions')
        source, target = _features_pair(x, size, False, self._widths)
        src, dst = _edges(edge_index, size, self.add_self_loops)
        for count in (*size, len(src)):
            _capacity((count, 3, max(self.out_channels, source_pos.shape[1])))
        delta = _output(self.pos_nn(target_pos.index_select(0, dst) - source_pos.index_select(0, src)),
                        len(src), 'pos_nn output', self.out_channels)
        alpha = self.lin_dst(target).index_select(0, dst) - self.lin_src(source).index_select(0, src) + delta
        if self.attn_nn is not None:
            alpha = _output(self.attn_nn(alpha), len(src), 'attn_nn output', self.out_channels)
        attention = segment_softmax(alpha, dst, size[1])
        message = attention * (self.lin(source).index_select(0, src) + delta)
        return scatter(message, dst, size[1], 'sum')
