"""CPU f32 hypergraph incidence propagation with optional learned attention."""
import math

from ._native import Tensor, cat, no_grad, scatter, segment_softmax
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_extra_conv import _weights
from .nn import Dropout, Linear, Module, Parameter


class HypergraphConv(Module):
    """Propagate node features to hyperedges and back through COO incidence.

    hyperedge_index has int64 shape [2, incidences], with node IDs in row zero
    and hyperedge IDs in row one. Duplicates contribute independently. Node
    attention normalizes within each hyperedge; edge attention normalizes
    hyperedges incident on each node. One dropout mask is reused on both passes.
    Hyperedge weights affect weighted node degree, matching PyG 2.8.1's actual
    implementation. Gradients at singular weighted degree are undefined.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    use_attention: bool
    attention_mode: str
    heads: int
    concat: bool
    negative_slope: float
    dropout: float
    aggr: str

    def __init__(self, in_channels, out_channels, use_attention=False, attention_mode='node',
                 heads=1, concat=True, negative_slope=.2, dropout=0., bias=True, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'), (heads, 'heads')):
            _positive(value, name)
        _flags(use_attention=use_attention, concat=concat, bias=bias)
        if attention_mode not in ('node', 'edge'):
            raise ValueError('attention_mode must be node or edge')
        if type(negative_slope) not in (int, float) or not math.isfinite(negative_slope):
            raise ValueError('negative_slope must be finite')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        if not use_attention and heads != 1:
            raise NotImplementedError('non-attention HypergraphConv requires heads=1')
        concat = concat if use_attention else True
        _capacity((in_channels, heads, out_channels))
        if use_attention:
            _capacity((heads, 2 * out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, use_attention=use_attention,
                   attention_mode=attention_mode, heads=heads, concat=concat, negative_slope=float(negative_slope),
                   dropout=float(dropout), aggr=_options(kwargs, 'add'))
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(Tensor.zeros([1, heads, 2 * out_channels])) if use_attention else None
        self.bias = Parameter(Tensor.zeros([heads * out_channels if concat else out_channels])) if bias else None
        self.attention_dropout = Dropout(dropout) if use_attention else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin, glorot=True)
        with no_grad():
            if self.att is not None:
                bound = math.sqrt(6. / (self.heads + 2 * self.out_channels))
                self.att.tensor().copy_((Tensor.rand([1, self.heads, 2 * self.out_channels]) * 2 - 1) * bound)
            if self.bias is not None:
                self.bias.tensor().copy_(Tensor.zeros(list(self.bias.tensor().shape)))

    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None, num_edges=None):
        _eager()
        _features(x, 'x', self.in_channels)
        _cpu_tensor(hyperedge_index, 'hyperedge_index', 'int64')
        nodes, edge_ids, _ = _edge_parts(hyperedge_index)
        node_count = x.shape[0]
        edge_count = max(edge_ids, default=-1) + 1 if num_edges is None else _count(num_edges, 'num_edges')
        if any(node >= node_count for node in nodes) or any(edge >= edge_count for edge in edge_ids):
            raise ValueError('hyperedge incidence contains an out-of-bounds node or hyperedge')
        _weights(hyperedge_weight, edge_count)
        for count in (node_count, edge_count):
            _capacity((count + 1,), 8)
            _capacity((count, self.heads, self.out_channels))
        _capacity((len(nodes), self.heads, 2 * self.out_channels))
        if self.use_attention:
            _features(hyperedge_attr, 'hyperedge_attr', self.in_channels)
            if hyperedge_attr.shape[0] != edge_count:
                raise ValueError('hyperedge_attr must contain one row per hyperedge')
        weights = Tensor.ones([edge_count]) if hyperedge_weight is None else hyperedge_weight
        projected = self.lin(x).reshape([node_count, self.heads, self.out_channels])
        alpha = None
        if self.use_attention:
            attributes = self.lin(hyperedge_attr).reshape([edge_count, self.heads, self.out_channels])
            pair = cat([projected.index_select(0, nodes), attributes.index_select(0, edge_ids)], 2)
            logits = (pair * self.att.tensor()).sum_dim(2, False).leaky_relu(self.negative_slope)
            ids, count = (edge_ids, edge_count) if self.attention_mode == 'node' else (nodes, node_count)
            alpha = self.attention_dropout(segment_softmax(logits, ids, count)).unsqueeze(2)
        degree = scatter(weights.index_select(0, edge_ids), nodes, node_count, 'sum').reciprocal()
        degree = degree.masked_fill(degree.eq(float('inf')), 0.)
        cardinality = scatter(Tensor.ones([len(nodes)]), edge_ids, edge_count, 'sum').reciprocal()
        cardinality = cardinality.masked_fill(cardinality.eq(float('inf')), 0.)
        message = projected.index_select(0, nodes) * cardinality.index_select(0, edge_ids).reshape([len(nodes), 1, 1])
        if alpha is not None:
            message = alpha * message
        edges = scatter(message.reshape([len(nodes), self.heads * self.out_channels]), edge_ids, edge_count, 'sum')
        message = edges.index_select(0, edge_ids).reshape([len(nodes), self.heads, self.out_channels])
        message = degree.index_select(0, nodes).reshape([len(nodes), 1, 1]) * message
        if alpha is not None:
            message = alpha * message
        output = scatter(message.reshape([len(nodes), self.heads * self.out_channels]), nodes, node_count, 'sum')
        if not self.concat:
            output = output.reshape([node_count, self.heads, self.out_channels]).mean_dim(1, False)
        return output if self.bias is None else output + self.bias.tensor()
