"""Native heterogeneous graph attention with learned relation-level weighting."""
import math
from collections.abc import Mapping

from ._native import Tensor, no_grad, scatter, segment_softmax, stack
from ._graph_conditioned import _eager
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear
from ._graph_hetero import _relation
from ._graph_utils import _capacity, _edge_parts
from .nn import Dropout, Linear, Module, ModuleDict, Parameter, ParameterDict


class HANConv(Module):
    """Relation-specific attention followed by learned semantic attention.

    Explicit fixed metadata and positive widths use eager CPU float32 features
    and COO topology. Input dictionaries may omit unused node types/relations.
    Outputs preserve input node order, with None for types without supplied
    incoming relations. Semantic attention entries follow incoming relation
    dictionary order, and are None for empty target types. One registered
    Dropout stream follows relation evaluation order. Native segment softmax
    requires finite logits. Sparse layouts, lazy widths and capture are unsupported.
    """
    _builtin_init = True
    channel_spec: tuple
    out_channels: int
    metadata: tuple
    heads: int
    negative_slope: float
    dropout: float
    aggr: str

    def __init__(self, in_channels, out_channels, metadata, heads=1, negative_slope=.2, dropout=0., **kwargs):
        _positive(out_channels, 'out_channels'); _positive(heads, 'heads')
        if out_channels % heads: raise ValueError('out_channels must be divisible by heads')
        if type(negative_slope) not in (int, float) or not math.isfinite(negative_slope): raise ValueError('negative_slope must be finite')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1: raise ValueError('dropout must be in [0,1]')
        if not isinstance(metadata, (tuple, list)) or len(metadata) != 2: raise ValueError('metadata must contain node types and edge types')
        nodes, relations = tuple(metadata[0]), tuple(_relation(key) for key in metadata[1])
        if not nodes or any(type(name) is not str or not name or '.' in name or name.startswith('_') for name in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError('node types must be unique public ModuleDict names')
        if len(set(relations)) != len(relations) or any(src not in nodes or dst not in nodes for src, _, dst in relations): raise ValueError('relations must be unique and reference declared node types')
        keys = tuple('__'.join(key) for key in relations)
        if any('.' in key for key in keys) or len(set(keys)) != len(keys): raise ValueError('encoded relation parameter keys must be distinct and contain no dots')
        widths = dict(in_channels) if isinstance(in_channels, Mapping) else {name: in_channels for name in nodes}
        if set(widths) != set(nodes): raise ValueError('in_channels keys must exactly match metadata node types')
        widths = {name: _positive(widths[name], 'in_channels') for name in nodes}
        for width in widths.values(): _capacity((width, out_channels))
        _capacity((out_channels, out_channels)); _capacity((2, len(relations), out_channels))
        aggr = _options(kwargs, 'add')
        _configure(self, channel_spec=tuple(widths.items()), out_channels=out_channels, metadata=(nodes, relations),
                   heads=heads, negative_slope=float(negative_slope), dropout=float(dropout), aggr=aggr)
        self.k_lin = Linear(out_channels, out_channels)
        self.q = Parameter(Tensor.zeros([1, out_channels]))
        self.proj = ModuleDict({name: Linear(width, out_channels) for name, width in widths.items()})
        self.lin_src, self.lin_dst = ParameterDict(), ParameterDict()
        for key in keys:
            self.lin_src[key] = Parameter(Tensor.zeros([1, heads, out_channels // heads]))
            self.lin_dst[key] = Parameter(Tensor.zeros([1, heads, out_channels // heads]))
        self.attn_dropout = Dropout(float(dropout))
        self.reset_parameters()

    @property
    def in_channels(self):
        return dict(self.channel_spec)

    def reset_parameters(self):
        for _, layer in self.proj.items(): _reset_linear(layer)
        _reset_linear(self.k_lin)
        with no_grad():
            bound = math.sqrt(6. / (self.heads + self.out_channels // self.heads))
            for module in (self.lin_src, self.lin_dst):
                for parameter in module.values():
                    parameter.tensor().copy_((Tensor.rand(parameter.tensor().shape) * 2 - 1) * bound)
            self.q.tensor().copy_((Tensor.rand([1, self.out_channels]) * 2 - 1) * math.sqrt(6. / (1 + self.out_channels)))

    def _preflight(self, x_dict, edge_index_dict):
        _eager()
        if not isinstance(x_dict, Mapping) or not isinstance(edge_index_dict, Mapping): raise TypeError('HAN inputs must be feature and relation dictionaries')
        widths, relations = dict(self.channel_spec), self.metadata[1]
        if any(name not in widths for name in x_dict): raise ValueError('undeclared input node type')
        if any(key not in relations for key in edge_index_dict): raise ValueError('undeclared input relation')
        for name, x in x_dict.items():
            _features(x, name, widths[name]); _capacity(x.shape); _capacity((x.shape[0], self.out_channels))
        validated, incoming = {}, {name: 0 for name in x_dict}
        for relation, edges in edge_index_dict.items():
            source, _, target = relation
            if source not in x_dict or target not in x_dict: raise ValueError('relation references a missing feature type')
            if isinstance(edges, Tensor): _capacity(edges.shape, 8)
            src, dst, _ = _edge_parts(edges, size=(x_dict[source].shape[0], x_dict[target].shape[0]))
            _capacity((len(src), self.out_channels))
            validated[relation] = src, dst
            incoming[target] += 1
        for name, count in incoming.items(): _capacity((count, x_dict[name].shape[0], self.out_channels))
        return validated

    def forward(self, x_dict, edge_index_dict, return_semantic_attention_weights=False):
        _flags(return_semantic_attention_weights=return_semantic_attention_weights)
        validated = self._preflight(x_dict, edge_index_dict)
        heads, width = self.heads, self.out_channels // self.heads
        projected = {name: self.proj[name](x).reshape([x.shape[0], heads, width]) for name, x in x_dict.items()}
        grouped = {name: [] for name in x_dict}
        for relation, (src, dst) in validated.items():
            source, _, target = relation; key = '__'.join(relation)
            xs, xt = projected[source], projected[target]
            alpha_source = (xs * self.lin_src[key].tensor()).sum_dim(2, False)
            alpha_target = (xt * self.lin_dst[key].tensor()).sum_dim(2, False)
            logits = alpha_source.index_select(0, src) + alpha_target.index_select(0, dst)
            alpha = self.attn_dropout(segment_softmax(logits.leaky_relu(self.negative_slope), dst, xt.shape[0]))
            messages = (xs.index_select(0, src) * alpha.reshape([len(src), heads, 1])).reshape([len(src), self.out_channels])
            grouped[target].append(scatter(messages, dst, xt.shape[0], 'sum').relu())
        output, semantic = {}, {}
        for name, values in grouped.items():
            if not values:
                output[name], semantic[name] = None, None
                continue
            value = stack(values, 0)
            if x_dict[name].shape[0] == 0:
                output[name], semantic[name] = value.reshape([0, self.out_channels]), None
                continue
            scores = (self.k_lin(value).tanh().mean_dim(1, False) * self.q.tensor()).sum_dim(1, False)
            attention = scores.softmax(0)
            output[name] = (value * attention.reshape([len(values), 1, 1])).sum_dim(0, False)
            semantic[name] = attention
        return (output, semantic) if return_semantic_attention_weights else output
