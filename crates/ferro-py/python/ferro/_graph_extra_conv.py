"""Additional sparse PyG-style CPU f32 convolutions, eager first-order only.

No Torch/NumPy dependency, dense adjacency or cached normalization. Dropout
streams are explicit registered state included by native training checkpoints. COO topology is native EdgeIndex or CPU int64 [2,E].
"""
import math

from . import _native
from ._native import Tensor, cat, no_grad, scatter, segment_softmax
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, _index_ids
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear, _reset_network
from .nn import Dropout, Linear, Module, ModuleList, Parameter, Sequential


def _weights(edge_weight, count):
    if edge_weight is not None:
        _cpu_tensor(edge_weight, 'edge_weight')
        if edge_weight.shape != [count]:
            raise ValueError('edge_weight must have shape [E]')
    return edge_weight


def _gcn_weights(src, dst, nodes, edge_weight, loops):
    weight = _weights(edge_weight, len(src))
    if loops:
        keep = [i for i, (s, d) in enumerate(zip(src, dst)) if s != d]
        diagonal = [i for i, (s, d) in enumerate(zip(src, dst)) if s == d]
        if weight is not None:
            ids = [src[i] for i in diagonal]
            if len(ids) != len(set(ids)):
                raise NotImplementedError('weighted duplicate self-loops with APPNP loop insertion are unsupported')
            diagonal_weight = Tensor.ones([nodes]).scatter(0, Tensor.from_i64(ids, [len(ids)]), weight.index_select(0, diagonal))
            weight = cat([weight.index_select(0, keep), diagonal_weight], 0)
        src = [src[i] for i in keep] + list(range(nodes))
        dst = [dst[i] for i in keep] + list(range(nodes))
    if weight is None:
        weight = Tensor.ones([len(src)])
    degree = scatter(weight, dst, nodes, 'sum')
    inverse = degree.pow(-.5)
    inverse = inverse.masked_fill(Tensor([float(value == float('inf')) for value in inverse.tolist()], [nodes]), 0.)
    weight = inverse.index_select(0, src) * weight * inverse.index_select(0, dst)
    return src, dst, weight


class GINEConv(Module):
    """GIN with ReLU(source + projected edge) messages and learned/buffered eps.

    edge_attr is required and has shape [E,D]. Without edge_dim it must match
    node width. With edge_dim the input width is inferred from nn (or the first
    module of Sequential). Bipartite inputs and a missing target are supported.
    Supplied self-loops are retained; no additional loops are inserted.
    """
    _builtin_init = True
    initial_eps: float
    train_eps: bool
    edge_dim: object
    aggr: str

    def __init__(self, nn, eps=0., train_eps=False, edge_dim=None, **kwargs):
        if not isinstance(nn, Module):
            raise TypeError('GINEConv nn must be a ferro.nn.Module')
        if type(eps) not in (int, float) or not math.isfinite(eps):
            raise ValueError('eps must be finite')
        _flags(train_eps=train_eps)
        if edge_dim is not None:
            _positive(edge_dim, 'edge_dim')
            first = nn[0] if isinstance(nn, Sequential) and len(nn) else nn
            width = getattr(first, 'in_features', getattr(first, 'in_channels', None))
            _positive(width, 'nn input width')
            _capacity((edge_dim, width))
        else:
            width = None
        _configure(self, initial_eps=float(eps), train_eps=train_eps, edge_dim=edge_dim, aggr=_options(kwargs, 'add'))
        self.nn = nn
        self.lin = Linear(edge_dim, width) if edge_dim is not None else None
        if train_eps:
            self.eps = Parameter(Tensor([float(eps)], [1]))
        else:
            self.register_buffer('eps', Tensor([float(eps)], [1]))
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.nn)
        if self.lin is not None:
            _reset_linear(self.lin)
        with no_grad():
            (self.eps.tensor() if self.train_eps else self.eps).copy_(Tensor([self.initial_eps], [1]))

    def forward(self, x, edge_index, edge_attr=None, size=None):
        width = None if self.lin is None else self.lin.out_features
        source, target, size = _pair(x, size, width, width)
        if target is not None and target.shape[1] != source.shape[1]:
            raise ValueError('GINEConv source/target feature widths must match')
        src, dst, _ = _edge_parts(edge_index, size=size)
        _cpu_tensor(edge_attr, 'edge_attr')
        expected_width = source.shape[1] if self.edge_dim is None else self.edge_dim
        if edge_attr.shape != [len(src), expected_width]:
            raise ValueError('GINEConv edge_attr must have shape [E, edge_dim or node width]')
        for count in (*size, len(src)):
            _capacity((count, source.shape[1]))
        attributes = edge_attr if self.lin is None else self.lin(edge_attr)
        messages = (source.index_select(0, src) + attributes).relu()
        output = scatter(messages, dst, size[1], 'sum')
        if target is not None:
            eps = self.eps.tensor() if self.train_eps else self.eps
            output = output + (1. + eps) * target
        return self.nn(output)


class APPNP(Module):
    """K sparse personalized-PageRank propagations with teleport probability alpha.

    Uses PyG COO GCN normalization and differentiable edge weights. K=0 returns
    the input after validation. normalize=False leaves topology unchanged even
    when add_self_loops=True, matching PyG. cached=True rejects. In training,
    dropout repeatedly modifies the preceding edge weights, matching PyG;
    normalize=False then requires explicit edge weights when K is positive.
    Weighted duplicate self-loops with normalization and loop insertion reject
    because PyG's assignment backward is not last-writer differentiation.
    """
    _builtin_init = True
    K: int
    alpha: float
    dropout: float
    cached: bool
    add_self_loops: bool
    normalize: bool
    aggr: str

    def __init__(self, K, alpha, dropout=0., cached=False, add_self_loops=True, normalize=True, **kwargs):
        if type(K) is not int or K < 0:
            raise ValueError('K must be a nonnegative integer')
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be between zero and one')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be between zero and one')
        _flags(cached=cached, add_self_loops=add_self_loops, normalize=normalize)
        if cached:
            raise NotImplementedError('APPNP supports cached=False only')
        _configure(self, K=K, alpha=float(alpha), dropout=float(dropout), cached=cached,
                   add_self_loops=add_self_loops, normalize=normalize, aggr=_options(kwargs, 'add'))
        if dropout:
            self.edge_dropout = Dropout(dropout)

    def reset_parameters(self):
        """No parameters or cache; the explicit dropout stream is preserved."""

    def forward(self, x, edge_index, edge_weight=None):
        if _native._is_capturing():
            raise ValueError('APPNP capture is unsupported; use eager execution')
        if self.dropout and self.edge_dropout.p != self.dropout:
            raise ValueError('construct a new APPNP to change its dropout configuration')
        _features(x, 'x')
        nodes, width = x.shape
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        _weights(edge_weight, len(src))
        if self.K and self.training and self.dropout and not self.normalize and edge_weight is None:
            raise ValueError('training APPNP dropout without normalization requires explicit edge_weight')
        _capacity((nodes + 1,), 8)
        _capacity((2, len(src) + (nodes if self.normalize and self.add_self_loops else 0)), 8)
        _capacity((nodes, width))
        _capacity((len(src) + (nodes if self.normalize and self.add_self_loops else 0), width))
        if self.normalize:
            src, dst, weight = _gcn_weights(src, dst, nodes, edge_weight, self.add_self_loops)
        else:
            weight = Tensor.ones([len(src)]) if edge_weight is None else edge_weight
        initial = x
        for _ in range(self.K):
            if self.dropout and self.training:
                weight = self.edge_dropout(weight)
            messages = x.index_select(0, src) * weight.reshape([len(src), 1])
            x = scatter(messages, dst, nodes, 'sum') * (1. - self.alpha) + initial * self.alpha
        return x


class ChebConv(Module):
    """Sparse Chebyshev spectral filter with PyG None/sym/rw Laplacians.

    lambda_max may be a positive scalar or a per-graph float32 vector indexed
    by int64 batch. The default is twice the maximum Laplacian edge weight,
    including its diagonal, as in PyG. Empty graphs require explicit lambda_max;
    nonpositive/nonfinite lambda_max rejects instead of producing NaNs.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    K: int
    normalization: object
    aggr: str

    def __init__(self, in_channels, out_channels, K, normalization='sym', bias=True, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'), (K, 'K')):
            _positive(value, name)
        _flags(bias=bias)
        if normalization not in (None, 'sym', 'rw'):
            raise ValueError('normalization must be None, sym or rw')
        _capacity((K, in_channels, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, K=K,
                   normalization=normalization, aggr=_options(kwargs, 'add'))
        self.lins = ModuleList(Linear(in_channels, out_channels, bias=False) for _ in range(K))
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lins:
            _reset_linear(layer, glorot=True)
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def _laplacian(self, src, dst, nodes, edge_weight, batch, lambda_max):
        weight = _weights(edge_weight, len(src))
        keep = [i for i, (s, d) in enumerate(zip(src, dst)) if s != d]
        src, dst = [src[i] for i in keep], [dst[i] for i in keep]
        weight = Tensor.ones([len(src)]) if weight is None else weight.index_select(0, keep)
        degree = scatter(weight, src, nodes, 'sum')
        if self.normalization is None:
            laplacian = cat([-weight, degree], 0)
        else:
            inverse = degree.pow(-.5 if self.normalization == 'sym' else -1.)
            inverse = inverse.masked_fill(Tensor([float(value == float('inf')) for value in inverse.tolist()], [nodes]), 0.)
            normalized = inverse.index_select(0, src) * weight
            if self.normalization == 'sym':
                normalized = normalized * inverse.index_select(0, dst)
            laplacian = cat([-normalized, Tensor.ones([nodes])], 0)
        src, dst = src + list(range(nodes)), dst + list(range(nodes))
        if lambda_max is None:
            if not src:
                raise ValueError('empty ChebConv graph requires explicit lambda_max')
            maximum = laplacian.max() * 2.
        elif isinstance(lambda_max, Tensor):
            _cpu_tensor(lambda_max, 'lambda_max')
            if len(lambda_max.shape) > 1 or (lambda_max.shape and lambda_max.shape[0] == 0):
                raise ValueError('lambda_max must be a scalar or nonempty vector')
            maximum = lambda_max
        elif type(lambda_max) in (int, float):
            maximum = Tensor([float(lambda_max)], [1])
        else:
            raise TypeError('lambda_max must be a scalar or CPU float32 Tensor')
        number = maximum.shape[0] if maximum.shape else 1
        entries = maximum.tolist() if maximum.shape else [maximum.item()]
        if any(not math.isfinite(value) or value <= 0 for value in entries):
            raise ValueError('lambda_max must contain finite positive values')
        if batch is not None:
            ids = _index_ids(batch, 'batch', nodes)
            if number > 1:
                if any(i >= number for i in ids):
                    raise ValueError('batch index is out of bounds for lambda_max')
                maximum = maximum.index_select(0, [ids[i] for i in src])
        elif number > 1:
            raise ValueError('per-graph lambda_max requires batch')
        normalized = (laplacian * 2.) / maximum
        normalized = normalized.masked_fill(Tensor([float(value == float('inf')) for value in normalized.tolist()], [len(src)]), 0.)
        diagonal = Tensor([0.] * (len(src) - nodes) + [1.] * nodes, [len(src)])
        return src, dst, normalized - diagonal

    def forward(self, x, edge_index, edge_weight=None, batch=None, lambda_max=None):
        _features(x, 'x', self.in_channels)
        nodes = x.shape[0]
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        for count in (nodes, len(src) + nodes):
            _capacity((count, max(self.in_channels, self.out_channels)))
        src, dst, norm = self._laplacian(src, dst, nodes, edge_weight, batch, lambda_max)
        previous, current = x, x
        output = self.lins[0](x)
        for index, layer in enumerate(self.lins):
            if index == 0:
                continue
            propagated = scatter(current.index_select(0, src) * norm.reshape([len(src), 1]), dst, nodes, 'sum')
            following = propagated if index == 1 else propagated * 2. - previous
            output = output + layer(following)
            previous, current = current, following
        return output if self.bias is None else output + self.bias.tensor()


class TransformerConv(Module):
    """Sparse multi-head dot-product graph attention with optional beta skip gate.

    Edge projections are added to both keys and values. Both source and target
    features are required for bipartite input. edge_dim requires [E,edge_dim]
    attributes; attributes without edge_dim are explicitly unsupported. No self
    loops are inserted. True or False returns attention weights, matching this
    operator's PyG 2.8 convention (which differs from GAT/GATv2). Returned
    attention precedes dropout; messages use the dropped attention weights.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    heads: int
    concat: bool
    beta: bool
    dropout: float
    edge_dim: object
    root_weight: bool
    aggr: str

    def __init__(self, in_channels, out_channels, heads=1, concat=True, beta=False, dropout=0.,
                 edge_dim=None, bias=True, root_weight=True, **kwargs):
        widths = (in_channels, in_channels) if type(in_channels) is int else in_channels
        if not isinstance(widths, (tuple, list)) or len(widths) != 2:
            raise ValueError('in_channels must be positive or a source/target pair')
        widths = tuple(widths)
        for value, name in [(n, 'in_channels') for n in widths] + [(out_channels, 'out_channels'), (heads, 'heads')]:
            _positive(value, name)
        _flags(concat=concat, beta=beta, bias=bias, root_weight=root_weight)
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be between zero and one')
        if edge_dim is not None:
            _positive(edge_dim, 'edge_dim')
            _capacity((edge_dim, heads, out_channels))
        for width in widths:
            _capacity((width, heads, out_channels))
        total = out_channels * (heads if concat else 1)
        _capacity((3, total))
        _configure(self, in_channels=in_channels if type(in_channels) is int else widths,
                   out_channels=out_channels, heads=heads, concat=concat, beta=beta and root_weight,
                   dropout=float(dropout), edge_dim=edge_dim, root_weight=root_weight, aggr=_options(kwargs, 'add'))
        if dropout:
            self.attention_dropout = Dropout(dropout)
        self.lin_key = Linear(widths[0], heads * out_channels, bias=bias)
        self.lin_query = Linear(widths[1], heads * out_channels, bias=bias)
        self.lin_value = Linear(widths[0], heads * out_channels, bias=bias)
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False) if edge_dim is not None else None
        self.lin_skip = Linear(widths[1], total, bias=bias)
        self.lin_beta = Linear(3 * total, 1, bias=False) if self.beta else None
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.lin_key, self.lin_query, self.lin_value, self.lin_edge, self.lin_skip, self.lin_beta):
            if layer is not None:
                _reset_linear(layer)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        if _native._is_capturing():
            raise ValueError('TransformerConv capture is unsupported; use eager execution')
        if self.dropout and self.attention_dropout.p != self.dropout:
            raise ValueError('construct a new TransformerConv to change its dropout configuration')
        if return_attention_weights is not None and type(return_attention_weights) is not bool:
            raise TypeError('return_attention_weights must be bool or None')
        widths = (self.in_channels, self.in_channels) if type(self.in_channels) is int else self.in_channels
        source, target, size = _pair(x, None, *widths)
        if target is None:
            raise ValueError('TransformerConv requires destination features')
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in (*size, len(src)):
            _capacity((count, self.heads, self.out_channels))
        _capacity((size[1] + 1,), 8)
        if self.beta:
            _capacity((size[1], 3, self.out_channels * (self.heads if self.concat else 1)))
        if self.edge_dim is not None:
            _cpu_tensor(edge_attr, 'edge_attr')
            if edge_attr.shape != [len(src), self.edge_dim]:
                raise ValueError('edge_attr must have shape [E, edge_dim]')
        elif edge_attr is not None:
            raise NotImplementedError('TransformerConv edge_attr requires configured edge_dim')
        heads, width = self.heads, self.out_channels
        query = self.lin_query(target).reshape([size[1], heads, width]).index_select(0, dst)
        key = self.lin_key(source).reshape([size[0], heads, width]).index_select(0, src)
        value = self.lin_value(source).reshape([size[0], heads, width]).index_select(0, src)
        if self.lin_edge is not None:
            projected = self.lin_edge(edge_attr).reshape([len(src), heads, width])
            key, value = key + projected, value + projected
        logits = (query * key).sum_dim(2, False) / math.sqrt(width)
        alpha = segment_softmax(logits, dst, size[1])
        dropped_alpha = self.attention_dropout(alpha) if self.dropout else alpha
        aggregate = scatter(value * dropped_alpha.reshape([len(src), heads, 1]), dst, size[1], 'sum')
        output = aggregate.reshape([size[1], heads * width]) if self.concat else aggregate.mean_dim(1, False)
        if self.root_weight:
            skip = self.lin_skip(target)
            if self.lin_beta is None:
                output = output + skip
            else:
                beta = self.lin_beta(cat([output, skip, output - skip], 1)).sigmoid()
                output = beta * skip + (1. - beta) * output
        if type(return_attention_weights) is bool:
            return output, (Tensor.from_i64(src + dst, [2, len(src)]), alpha)
        return output
