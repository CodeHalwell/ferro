"""Feature modulation, crystal messages and frequency-adaptive propagation.

Dense CPU f32 node features, explicit COO topology and native first-order
operations only. No feature/normalization cache or alternative gradient engine.
"""
import math

from ._native import Tensor, Generator, cat, no_grad, scatter
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor, _index_ids
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear, _reset_network
from ._graph_extra_conv import _gcn_weights, _weights
from ._graph_conditioned import _eager, _reduction
from ._graph_residual import _widths
from ._graph_spectral import _activation
from .nn import BatchNorm1d, Dropout, GELU, Identity, Linear, Module, ModuleList, ReLU, Sequential, Sigmoid, Tanh
from .nn import functional as F


def _fresh_template(template):
    if type(template) is Linear:
        return Linear(template.in_features, template.out_features, bias=template.bias is not None)
    if type(template) is Sequential:
        return Sequential(*(_fresh_template(child) for child in template))
    if type(template) in (ReLU, Tanh, Sigmoid, Identity):
        return type(template)()
    if type(template) is GELU:
        return GELU(approximate=template.approximate)
    if type(template) is Dropout:
        return Dropout(template.p, seed=0)
    raise NotImplementedError('FiLM templates support exact Linear/Sequential/ReLU/Tanh/Sigmoid/GELU/Identity/Dropout modules only')


def _template_clones(template, count):
    if not isinstance(template, Module):
        raise TypeError('nn must be a Ferro Module or None')
    _capacity((count, sum(p.tensor().numel() for p in template.parameters()) + sum(b.numel() for _, b in template.named_buffers())))
    first = _fresh_template(template)
    from .checkpoint import snapshot_training
    snapshot = snapshot_training(template, optimizers={}, generator=Generator(0))
    result = []
    for index in range(count):
        clone = first if index == 0 else _fresh_template(template)
        snapshot.restore(clone, optimizers={}, generator=Generator(0))
        result.append(clone)
    return result


class FiLMConv(Module):
    """Relation-wise source projections modulated by target beta/gamma maps.

    A separate skip relation is always present. Known quiescent MLP templates
    are independently reconstructed with copied modes/freezing/dropout buffers;
    linear weights then reset as in PyG. Clear template gradients first. Tied or
    structurally modified templates reject. With one relation edge_type is
    ignored; multiple relations require bounded int64 [E] IDs.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    num_relations: int
    aggr: str

    def __init__(self, in_channels, out_channels, num_relations=1, nn=None, act='relu', aggr='mean', **kwargs):
        in_channels, widths = _widths(in_channels)
        _positive(out_channels, 'out_channels')
        _positive(num_relations, 'num_relations')
        activation = _activation(act)
        _capacity((num_relations + 1, max(widths), 3 * out_channels))
        clones = None if nn is None else _template_clones(nn, num_relations + 1)
        _configure(self, in_channels=in_channels, out_channels=out_channels, num_relations=num_relations, aggr=_reduction(aggr, kwargs))
        self._widths = widths
        self.act = activation
        self.lins = ModuleList(Linear(widths[0], out_channels, bias=False) for _ in range(num_relations))
        self.films = ModuleList(Linear(widths[1], 2 * out_channels) for _ in range(num_relations)) if clones is None else ModuleList(clones[:-1])
        self.lin_skip = Linear(widths[1], out_channels, bias=False)
        self.film_skip = Linear(widths[1], 2 * out_channels, bias=False) if clones is None else clones[-1]
        self.reset_parameters()

    def reset_parameters(self):
        for linear, film in zip(self.lins, self.films):
            _reset_linear(linear)
            _reset_network(film)
        _reset_linear(self.lin_skip)
        _reset_network(self.film_skip)

    def _modulation(self, module, x):
        values = module(x)
        _features(values, 'modulation output', 2 * self.out_channels)
        if values.shape[0] != x.shape[0]:
            raise ValueError('modulation network must preserve target node rows')
        return values.narrow(1, 0, self.out_channels), values.narrow(1, self.out_channels, self.out_channels)

    def _activate(self, value):
        return value if self.act is None else self.act(value)

    def forward(self, x, edge_index, edge_type=None):
        _eager()
        source, target, size = _pair(x, None, *self._widths)
        if target is None:
            raise ValueError('FiLMConv requires target features')
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size:
            _capacity((count + 1,), 8)
        for count in (*size, len(src)):
            _capacity((count, 2, self.out_channels))
        if self.num_relations > 1:
            _cpu_tensor(edge_type, 'edge_type', 'int64')
            relations = _index_ids(edge_type, 'edge_type', len(src))
            if any(index >= self.num_relations for index in relations):
                raise ValueError('edge_type contains an unknown relation')
        else:
            relations = None
        beta, gamma = self._modulation(self.film_skip, target)
        result = self._activate(gamma * self.lin_skip(target) + beta)
        for relation, (linear, film) in enumerate(zip(self.lins, self.films)):
            beta, gamma = self._modulation(film, target)
            selected = range(len(src)) if relations is None else [i for i, value in enumerate(relations) if value == relation]
            source_ids, target_ids = [src[i] for i in selected], [dst[i] for i in selected]
            message = gamma.index_select(0, target_ids) * linear(source).index_select(0, source_ids) + beta.index_select(0, target_ids)
            result = result + scatter(self._activate(message), target_ids, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        return result


class CGConv(Module):
    """Sigmoid-filtered softplus crystal messages plus the original target.

    Optional BatchNorm runs on aggregated messages before the target residual.
    Running statistics and the typed batch counter are checkpoint state. Edge
    dimension zero accepts no attributes or an [E,0] tensor.
    """
    _builtin_init = True
    channels: object
    dim: int
    batch_norm: bool
    aggr: str

    def __init__(self, channels, dim=0, aggr='add', batch_norm=False, bias=True, **kwargs):
        channels, widths = _widths(channels)
        _count(dim, 'dim')
        _flags(batch_norm=batch_norm, bias=bias)
        _capacity((2, sum(widths) + dim, widths[1]))
        _configure(self, channels=channels, dim=dim, batch_norm=batch_norm, aggr=_reduction(aggr, kwargs))
        self._widths = widths
        self.lin_f = Linear(sum(widths) + dim, widths[1], bias=bias)
        self.lin_s = Linear(sum(widths) + dim, widths[1], bias=bias)
        self.bn = BatchNorm1d(widths[1]) if batch_norm else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin_f)
        _reset_linear(self.lin_s)
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        _eager()
        source, target, size = _pair(x, None, *self._widths)
        if target is None:
            raise ValueError('CGConv requires target features')
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size:
            _capacity((count + 1,), 8)
        _capacity((len(src), sum(self._widths) + self.dim))
        if edge_attr is not None:
            _features(edge_attr, 'edge_attr', self.dim)
            if edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must contain one row per edge')
        elif self.dim:
            raise ValueError('positive dim requires edge_attr')
        values = [target.index_select(0, dst), source.index_select(0, src)]
        if edge_attr is not None:
            values.append(edge_attr)
        combined = cat(values, 1)
        message = self.lin_f(combined).sigmoid() * F.softplus(self.lin_s(combined))
        output = scatter(message, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        return (self.bn(output) if self.bn is not None else output) + target


class FAConv(Module):
    """Frequency-adaptive signed attention with checkpointed attention dropout.

    normalize=True requires no supplied weights and uses GCN normalization.
    normalize=False requires scalar edge weights and does not insert loops.
    cached=True rejects. A bool return flag, including False, returns raw
    pre-dropout attention plus the effective topology, matching PyG.
    """
    _builtin_init = True
    channels: int
    eps: float
    dropout: float
    cached: bool
    add_self_loops: bool
    normalize: bool
    aggr: str

    def __init__(self, channels, eps=.1, dropout=0., cached=False, add_self_loops=True, normalize=True, **kwargs):
        _positive(channels, 'channels')
        if type(eps) not in (int, float) or not math.isfinite(eps):
            raise ValueError('eps must be finite')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        _flags(cached=cached, add_self_loops=add_self_loops, normalize=normalize)
        if cached:
            raise NotImplementedError('FAConv supports cached=False only')
        _capacity((2, channels))
        _configure(self, channels=channels, eps=float(eps), dropout=float(dropout), cached=cached,
                   add_self_loops=add_self_loops, normalize=normalize, aggr=_options(kwargs, 'add'))
        self.att_l, self.att_r = Linear(channels, 1, bias=False), Linear(channels, 1, bias=False)
        self.attention_dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.att_l)
        _reset_linear(self.att_r)

    def forward(self, x, x_0, edge_index, edge_weight=None, return_attention_weights=None):
        _eager()
        _features(x, 'x', self.channels)
        _features(x_0, 'x_0', self.channels)
        if x.shape != x_0.shape:
            raise ValueError('x_0 must match x shape')
        if return_attention_weights is not None and type(return_attention_weights) is not bool:
            raise TypeError('return_attention_weights must be bool or None')
        nodes = x.shape[0]
        _capacity((nodes + 1,), 8)
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        count = len(src) + (nodes if self.normalize and self.add_self_loops else 0)
        _capacity((2, count), 8)
        _capacity((count, self.channels))
        if self.normalize:
            if edge_weight is not None:
                raise ValueError('normalized FAConv requires edge_weight=None')
            src, dst, edge_weight = _gcn_weights(src, dst, nodes, None, self.add_self_loops)
        else:
            if edge_weight is None:
                raise ValueError('unnormalized FAConv requires edge_weight')
            _weights(edge_weight, len(src))
        alpha = (self.att_l(x).index_select(0, src) + self.att_r(x).index_select(0, dst)).tanh().squeeze(-1)
        weighted = self.attention_dropout(alpha) * edge_weight
        output = scatter(x.index_select(0, src) * weighted.reshape([len(src), 1]), dst, nodes, 'sum')
        if self.eps:
            output = output + self.eps * x_0
        return output if return_attention_weights is None else (output, (_edge_tensor(src, dst), alpha))
