"""Sparse CPU f32 PyG-style graph attention, without a Torch dependency."""
import math

from . import _native
from ._native import Tensor, segment_softmax, segment_sum
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, add_self_loops, remove_self_loops
from .nn import Dropout, Linear, Module, Parameter


class _AttentionBase(Module):
    """Multi-head additive graph attention with PyG 2.8-style arguments.

    Supports homogeneous or bipartite CPU f32 inputs and typed I64 [2,E]
    topology. Native EdgeIndex is accepted for homogeneous topology. Attention
    uses per-destination sparse softmax; duplicate edges remain independent.
    Self loops replace existing diagonal edges and derive their attributes
    using a scalar/Tensor or add/sum/mean/min/max fill reduction. Tensor fill
    values are persistent buffers; Parameter fills remain learned parameters. Both
    participate in checkpoints and preserve their values across resets.

    Attention dropout has its own checkpointed native random stream. Lazy
    channels, SparseTensor adjacency, custom aggregation and higher derivatives
    are unsupported. return_attention_weights=True
    returns (output, (edge_index, alpha)); False/None return only the output,
    matching the installed PyG 2.8.1 implementation. No dense adjacency is built.
    """
    _builtin_init = True

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2,
                 dropout=0.0, add_self_loops=True, edge_dim=None, fill_value='mean',
                 bias=True, residual=False, **kwargs):
        channels = (in_channels, in_channels) if type(in_channels) is int else in_channels
        if type(channels) is not tuple or len(channels) != 2 or any(type(n) is not int or n <= 0 for n in channels):
            raise ValueError('in_channels must be positive or a pair of positive integers; lazy channels are unsupported')
        if type(out_channels) is not int or out_channels <= 0 or type(heads) is not int or heads <= 0:
            raise ValueError('out_channels and heads must be positive integers')
        if any(type(value) is not bool for value in (concat, add_self_loops, bias, residual)):
            raise TypeError('concat, add_self_loops, bias and residual must be bool')
        if type(negative_slope) not in (int, float) or not math.isfinite(negative_slope):
            raise ValueError('negative_slope must be finite')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be between zero and one')
        if edge_dim is not None and (type(edge_dim) is not int or edge_dim <= 0):
            raise ValueError('edge_dim must be a positive integer or None')
        if isinstance(fill_value, Tensor):
            _cpu_tensor(fill_value, 'fill_value')
        elif type(fill_value) is str:
            if fill_value not in ('add', 'sum', 'mean', 'min', 'max'):
                raise NotImplementedError('fill_value reduction must be add/sum/mean/min/max')
        elif fill_value is not None and (type(fill_value) not in (int, float) or not math.isfinite(fill_value)):
            raise ValueError('fill_value must be a finite scalar, CPU f32 Tensor or supported reduction')
        for name, value in kwargs.items():
            if not ((name == 'aggr' and value in ('add', 'sum')) or
                    (name == 'flow' and value == 'source_to_target') or (name == 'node_dim' and value == 0)):
                raise NotImplementedError('Unsupported GATConv MessagePassing option: ' + name)
        for width in channels:
            _capacity((heads, out_channels, width))
        _capacity((heads, out_channels))
        if edge_dim is not None:
            _capacity((heads, out_channels, edge_dim))
        if residual:
            _capacity((out_channels * (heads if concat else 1), channels[1]))
        self._initialize()
        self.in_channels, self.out_channels, self.heads = in_channels, out_channels, heads
        self.concat, self.negative_slope, self.dropout = concat, float(negative_slope), float(dropout)
        self.add_self_loops, self.edge_dim, self.residual = add_self_loops, edge_dim, residual
        if dropout:
            self.attention_dropout = Dropout(dropout)
        if isinstance(fill_value, Parameter):
            self.fill_value = fill_value
        elif isinstance(fill_value, Tensor):
            self.register_buffer('fill_value', fill_value)
        else:
            self.fill_value = fill_value

    def reset_parameters(self):
        from ._native import no_grad
        with no_grad():
            for name, parameter in self.named_parameters(remove_duplicate=False):
                if name == 'fill_value':
                    continue
                shape = list(parameter.tensor().shape)
                if name in ('lin_l.bias', 'lin_r.bias'):
                    width = getattr(self, name.split('.')[0]).in_features
                    value = (Tensor.rand(shape) * 2. - 1.) * width ** -.5
                elif name == 'bias' or name.endswith('.bias'):
                    value = Tensor.zeros(shape)
                else:
                    bound = math.sqrt(6. / (shape[-2] + shape[-1]))
                    value = (Tensor.rand(shape) * 2. - 1.) * bound
                parameter.tensor().copy_(value)

    def _prepare(self, x, edge_index, edge_attr, size, return_attention_weights):
        if _native._is_capturing():
            raise ValueError('graph attention capture is unsupported; use eager execution')
        dropout = getattr(self, 'attention_dropout', None)
        if (dropout is None and self.dropout != 0.) or (dropout is not None and dropout.p != self.dropout):
            raise ValueError('construct a new graph attention layer to change its dropout configuration')
        if return_attention_weights is not None and type(return_attention_weights) is not bool:
            raise TypeError('return_attention_weights must be bool or None')
        if isinstance(x, Tensor):
            source = target = x
        elif type(x) in (tuple, list) and len(x) == 2:
            source, target = x
        else:
            raise TypeError('x must be a Tensor or a (source, destination) feature pair')
        channels = (self.in_channels, self.in_channels) if type(self.in_channels) is int else self.in_channels
        for value, width, name in ((source, channels[0], 'source features'), (target, channels[1], 'destination features')):
            if value is None and name == 'destination features':
                continue
            _cpu_tensor(value, name)
            if len(value.shape) != 2 or value.shape[1] != width:
                raise ValueError(name + ' must have shape [nodes, in_channels]')
        counts = (source.shape[0], target.shape[0] if target is not None else source.shape[0])
        if size is not None:
            if type(size) not in (tuple, list) or len(size) != 2 or any(n is not None and (type(n) is not int or n < 0) for n in size):
                raise ValueError('size must be a pair of nonnegative integers or None entries')
            requested = tuple(counts[i] if size[i] is None else size[i] for i in range(2))
            if requested[0] != counts[0] or (target is not None and requested[1] != counts[1]):
                raise ValueError('size disagrees with feature row counts')
            counts = requested
        sources, destinations, _ = _edge_parts(edge_index, size=counts)
        for count in counts:
            _capacity((count + 1,), 8)
        _capacity((2, len(sources) + (min(counts) if self.add_self_loops else 0)), 8)
        for count in (*counts, len(sources) + (min(counts) if self.add_self_loops else 0)):
            _capacity((count, self.heads, self.out_channels))
        if edge_attr is not None:
            _cpu_tensor(edge_attr, 'edge_attr')
            if len(edge_attr.shape) not in (1, 2) or edge_attr.shape[0] != len(sources):
                raise ValueError('edge_attr must have shape [edges] or [edges, edge_dim]')
            width = 1 if len(edge_attr.shape) == 1 else edge_attr.shape[1]
            if self.edge_dim is not None and width != self.edge_dim:
                raise ValueError('edge_attr width disagrees with edge_dim')
        topology = Tensor.from_i64(sources + destinations, [2, len(sources)])
        if self.add_self_loops:
            topology, edge_attr = remove_self_loops(topology, edge_attr)
            topology, edge_attr = add_self_loops(topology, edge_attr, fill_value=self.fill_value, num_nodes=counts)
            sources, destinations, _ = _edge_parts(topology, size=counts)
        return source, target, counts, topology, edge_attr, sources, destinations

    def _aggregate(self, projected_source, alpha, target, counts, topology, sources, destinations, return_attention_weights):
        heads, width = self.heads, self.out_channels
        messages = projected_source.index_select(0, sources) * alpha.reshape([len(sources), heads, 1])
        aggregated = segment_sum(messages, destinations, counts[1])
        output = aggregated.reshape([counts[1], heads * width]) if self.concat else aggregated.mean_dim(1, False)
        if self.res is not None and target is not None:
            output = output + self.res(target)
        if self.bias is not None:
            output = output + self.bias.tensor()
        return (output, (topology, alpha)) if return_attention_weights else output


class GATConv(_AttentionBase):
    """Sparse PyG-style additive attention; see the common supported contract.

    CPU f32, multi-head concat/mean, homogeneous/bipartite features, edge
    attributes, self-loop fill, residual, bias and checkpointed attention dropout
    are supported. Lazy channels, SparseTensor adjacency and custom aggregation reject.
    Execution is eager and first-order only.
    Only True returns attention weights, matching PyG 2.8.1. Tensor loop fills
    are persistent buffers. Rectangular loop reductions use the full target
    size; this also supports cases PyG's min-size reduction rejects.
    """
    _builtin_init = True

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2,
                 dropout=0.0, add_self_loops=True, edge_dim=None, fill_value='mean',
                 bias=True, residual=False, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout,
                         add_self_loops, edge_dim, fill_value, bias, residual, **kwargs)
        channels = (in_channels, in_channels) if type(in_channels) is int else in_channels
        self.lin = self.lin_src = self.lin_dst = None
        if type(in_channels) is int:
            self.lin = Linear(in_channels, heads * out_channels, bias=False)
        else:
            self.lin_src = Linear(channels[0], heads * out_channels, bias=False)
            self.lin_dst = Linear(channels[1], heads * out_channels, bias=False)
        self.att_src = Parameter(Tensor.zeros([1, heads, out_channels]))
        self.att_dst = Parameter(Tensor.zeros([1, heads, out_channels]))
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False) if edge_dim is not None else None
        self.att_edge = Parameter(Tensor.zeros([1, heads, out_channels])) if edge_dim is not None else None
        width = out_channels * (heads if concat else 1)
        self.res = Linear(channels[1], width, bias=False) if residual else None
        self.bias = Parameter(Tensor.zeros([width])) if bias else None
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=None):
        source, target, counts, topology, edge_attr, sources, destinations = self._prepare(
            x, edge_index, edge_attr, size, return_attention_weights)
        heads, width = self.heads, self.out_channels
        projected_source = (self.lin or self.lin_src)(source).reshape([counts[0], heads, width])
        projected_target = None if target is None else (self.lin or self.lin_dst)(target).reshape([counts[1], heads, width])
        alpha_source = (projected_source * self.att_src.tensor()).sum_dim(2, False)
        logits = alpha_source.index_select(0, sources)
        if projected_target is not None:
            alpha_target = (projected_target * self.att_dst.tensor()).sum_dim(2, False)
            logits = logits + alpha_target.index_select(0, destinations)
        if edge_attr is not None and self.lin_edge is not None and sources:
            attributes = edge_attr.reshape([len(sources), self.edge_dim])
            projected_edge = self.lin_edge(attributes).reshape([len(sources), heads, width])
            logits = logits + (projected_edge * self.att_edge.tensor()).sum_dim(2, False)
        if sources:
            alpha = segment_softmax(logits.leaky_relu(self.negative_slope), destinations, counts[1])
            if self.dropout:
                alpha = self.attention_dropout(alpha)
        else:
            alpha = logits
        return self._aggregate(projected_source, alpha, target, counts, topology, sources, destinations, return_attention_weights)


class GATv2Conv(_AttentionBase):
    """Dynamic sparse attention with leaky-ReLU before the attention projection.

    Supports the GATConv CPU/dropout/fill contract and separate or shared source
    and target transformations. Both feature tensors are required for bipartite
    inputs. share_weights=True requires equal input widths and preserves the
    shared parameter identity in optimizers and checkpoints. In contrast to
    GATConv, bias also controls the source/target linear biases, and edge_attr
    requires edge_dim. The forward signature follows PyG 2.8.1 (no size kwarg).
    """
    _builtin_init = True

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2,
                 dropout=0.0, add_self_loops=True, edge_dim=None, fill_value='mean',
                 bias=True, share_weights=False, residual=False, **kwargs):
        if type(share_weights) is not bool:
            raise TypeError('share_weights must be bool')
        if share_weights and type(in_channels) is tuple and len(in_channels) == 2 and in_channels[0] != in_channels[1]:
            raise ValueError('share_weights requires equal source and target feature widths')
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout,
                         add_self_loops, edge_dim, fill_value, bias, residual, **kwargs)
        channels = (in_channels, in_channels) if type(in_channels) is int else in_channels
        self.share_weights = share_weights
        self.lin_l = Linear(channels[0], heads * out_channels, bias=bias)
        self.lin_r = self.lin_l if share_weights else Linear(channels[1], heads * out_channels, bias=bias)
        self.att = Parameter(Tensor.zeros([1, heads, out_channels]))
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False) if edge_dim is not None else None
        width = out_channels * (heads if concat else 1)
        self.res = Linear(channels[1], width, bias=False) if residual else None
        self.bias = Parameter(Tensor.zeros([width])) if bias else None
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        if isinstance(x, (tuple, list)) and len(x) == 2 and x[1] is None:
            raise ValueError('GATv2Conv requires destination features')
        if edge_attr is not None and self.edge_dim is None:
            raise ValueError('GATv2Conv requires edge_dim when edge_attr is provided')
        source, target, counts, topology, edge_attr, sources, destinations = self._prepare(
            x, edge_index, edge_attr, None, return_attention_weights)
        heads, width = self.heads, self.out_channels
        projected_source = self.lin_l(source).reshape([counts[0], heads, width])
        projected_target = projected_source if isinstance(x, Tensor) and self.share_weights else self.lin_r(target).reshape([counts[1], heads, width])
        joint = projected_source.index_select(0, sources) + projected_target.index_select(0, destinations)
        if edge_attr is not None:
            joint = joint + self.lin_edge(edge_attr.reshape([len(sources), self.edge_dim])).reshape([len(sources), heads, width])
        logits = (joint.leaky_relu(self.negative_slope) * self.att.tensor()).sum_dim(2, False)
        alpha = segment_softmax(logits, destinations, counts[1])
        if self.dropout:
            alpha = self.attention_dropout(alpha)
        return self._aggregate(projected_source, alpha, target, counts, topology, sources, destinations, return_attention_weights)
