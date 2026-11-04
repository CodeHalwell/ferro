"""GENConv composition over native graph aggregators and normalization layers."""
import math

from ._native import Tensor, no_grad
from ._graph_aggr_base import Aggregation
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_residual import _widths
from ._graph_aggregation import (_Aggregation, SumAggregation, MeanAggregation, MinAggregation, MaxAggregation,
                                 VarAggregation, StdAggregation, SoftmaxAggregation, PowerMeanAggregation, MultiAggregation)
from ._graph_deep import MessageNorm
from .nn import BatchNorm1d, Dropout, LayerNorm, Linear, Module, ReLU, Sequential


def _aggregation(aggr, options):
    if isinstance(aggr, (list, tuple)):
        return MultiAggregation(aggr, **options)
    if isinstance(aggr, Aggregation):
        if options:
            raise ValueError('aggregation Module instances do not accept aggr_kwargs')
        return aggr
    constructors = dict(add=SumAggregation, sum=SumAggregation, mean=MeanAggregation,
                        min=MinAggregation, max=MaxAggregation, var=VarAggregation,
                        std=StdAggregation, softmax=SoftmaxAggregation, powermean=PowerMeanAggregation)
    if isinstance(aggr, str) and aggr in ('lstm', 'gru'):
        from ._graph_recurrent_aggr import LSTMAggregation, GRUAggregation
        constructors.update(lstm=LSTMAggregation, gru=GRUAggregation)
    if isinstance(aggr, str) and aggr in ('quantile', 'median'):
        from ._graph_quantile import QuantileAggregation, MedianAggregation
        constructors.update(quantile=QuantileAggregation, median=MedianAggregation)
    if isinstance(aggr, str) and aggr in ('variance_preserving', 'variancepreserving'):
        from ._graph_variance_aggr import VariancePreservingAggregation
        constructors.update(variance_preserving=VariancePreservingAggregation, variancepreserving=VariancePreservingAggregation)
    if aggr == 'sort':
        from ._graph_sort_aggr import SortAggregation
        constructors['sort'] = SortAggregation
    if aggr == 'graph_multiset_transformer':
        from ._graph_multiset import GraphMultisetTransformer
        constructors['graph_multiset_transformer'] = GraphMultisetTransformer
    if aggr == 'set_transformer':
        from ._graph_set_transformer import SetTransformerAggregation
        constructors['set_transformer'] = SetTransformerAggregation
    if aggr == 'set2set':
        from ._graph_advanced import Set2Set
        constructors['set2set'] = Set2Set
    if isinstance(aggr, str) and aggr in ('deep_sets', 'deepsets'):
        from ._graph_deep_sets import DeepSetsAggregation
        constructors.update(deep_sets=DeepSetsAggregation, deepsets=DeepSetsAggregation)
    if aggr == 'attentional':
        from ._graph_advanced import AttentionalAggregation
        constructors['attentional'] = AttentionalAggregation
    if aggr == 'degree_scaler':
        from ._graph_degree_aggr import DegreeScalerAggregation
        constructors['degree_scaler'] = DegreeScalerAggregation
    if not isinstance(aggr, str) or aggr not in constructors:
        raise NotImplementedError('GENConv requires a supported aggregation name, list or Module')
    return constructors[aggr](**options)


class GENConv(Module):
    """Positive messages with generalized aggregation, target skip and an MLP.

    CPU f32 rank-two homogeneous or bipartite features; target features may be
    absent with an explicit size. Known aggregation Modules remain registered
    directly. MLP normalization supports None, layer and batch; instance norm,
    sparse tensors, lazy widths, arbitrary callbacks and capture reject. Invalid
    singleton batch-statistics calls reject before attention streams or running
    state change, including evaluation with absent running statistics.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    eps: float
    edge_dim: object
    norm: object
    num_layers: int
    expansion: int

    def __init__(self, in_channels, out_channels, aggr='softmax', t=1., learn_t=False,
                 p=1., learn_p=False, msg_norm=False, learn_msg_scale=False, norm='batch',
                 num_layers=2, expansion=2, eps=1e-7, bias=False, edge_dim=None, **kwargs):
        in_channels, widths = _widths(in_channels)
        for value, name in ((out_channels, 'out_channels'), (num_layers, 'num_layers'), (expansion, 'expansion')):
            _positive(value, name)
        _flags(learn_t=learn_t, learn_p=learn_p, msg_norm=msg_norm, learn_msg_scale=learn_msg_scale, bias=bias)
        if norm not in (None, 'batch', 'layer'):
            raise NotImplementedError('GENConv supports norm=None/batch/layer only')
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps < 0:
            raise ValueError('eps must be finite and nonnegative')
        if edge_dim is not None:
            _positive(edge_dim, 'edge_dim')
            _capacity((edge_dim, out_channels))
        options = kwargs.pop('aggr_kwargs', None)
        _options(kwargs, 'add')
        semi_grad = aggr == 'softmax_sg'
        aggr = 'softmax' if semi_grad else 'powermean' if aggr == 'power' else aggr
        if options is None:
            options = dict(t=t, learn=learn_t, semi_grad=semi_grad) if aggr == 'softmax' else dict(p=p, learn=learn_p) if aggr == 'powermean' else {}
        elif type(options) is not dict:
            raise TypeError('aggr_kwargs must be a dictionary or None')
        aggregate = _aggregation(aggr, dict(options))
        aggregate_width = aggregate.get_out_channels(out_channels)
        inner = out_channels * expansion if num_layers > 1 else out_channels
        _capacity((num_layers, max(out_channels, inner), max(out_channels, inner)))
        _capacity((max(widths), out_channels))
        _capacity((aggregate_width, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, eps=float(eps),
                   edge_dim=edge_dim, norm=norm, num_layers=num_layers, expansion=expansion)
        self._widths = widths
        self._aggregate_width = aggregate_width
        self.aggr_module = aggregate
        for name, width in (('lin_src', widths[0]), ('lin_dst', widths[1]), ('lin_aggr_out', aggregate_width)):
            if width != out_channels:
                layer = Linear(width, out_channels, bias=bias)
                _reset_linear(layer)
                setattr(self, name, layer)
        if edge_dim is not None and edge_dim != out_channels:
            self.lin_edge = Linear(edge_dim, out_channels, bias=bias)
            _reset_linear(self.lin_edge)
        channels = [out_channels] + [inner] * (num_layers - 1) + [out_channels]
        layers = []
        for index, (source, target) in enumerate(zip(channels, channels[1:])):
            linear = Linear(source, target, bias=bias)
            _reset_linear(linear)
            layers.append(linear)
            if index < num_layers - 1:
                if norm == 'batch':
                    layers.append(BatchNorm1d(target, affine=True))
                elif norm == 'layer':
                    layers.append(LayerNorm(target, elementwise_affine=True))
                layers.extend((ReLU(), Dropout(0.)))
        self.mlp = Sequential(*layers)
        if msg_norm:
            self.msg_norm = MessageNorm(learn_msg_scale)

    def reset_parameters(self):
        self.aggr_module.reset_parameters()
        for name in ('lin_src', 'lin_dst', 'lin_aggr_out', 'lin_edge'):
            if hasattr(self, name):
                _reset_linear(getattr(self, name))
        for layer in self.mlp:
            if isinstance(layer, Linear):
                _reset_linear(layer)
            elif isinstance(layer, BatchNorm1d):
                layer.reset_parameters()
            elif isinstance(layer, LayerNorm):
                with no_grad():
                    layer.weight.tensor().copy_(Tensor.ones(list(layer.weight.tensor().shape)))
                    layer.bias.tensor().copy_(Tensor.zeros(list(layer.bias.tensor().shape)))
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None):
        _eager()
        source, target, size = _pair(x, size, *self._widths)
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size:
            _capacity((count + 1,), 8)
        _capacity((size[0], self.out_channels))
        inner = self.out_channels * self.expansion if self.num_layers > 1 else self.out_channels
        _capacity((size[1], max(self._aggregate_width, inner)))
        _capacity((len(src), self.out_channels))
        if edge_attr is not None:
            _features(edge_attr, 'edge_attr', self.out_channels if self.edge_dim is None else self.edge_dim)
            if edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must contain one row per edge')
        if hasattr(self, 'msg_norm') and target is None and size[0] != size[1]:
            raise ValueError('message normalization without target features requires equal node counts')
        if self.norm == 'batch' and self.num_layers > 1 and size[1] == 1 and any(
                isinstance(layer, BatchNorm1d) and (layer.training or (layer.running_mean is None and layer.running_var is None)) for layer in self.mlp):
            raise ValueError('BatchNorm using batch statistics requires more than one target row')
        projected = self.lin_src(source) if hasattr(self, 'lin_src') else source
        message = projected.index_select(0, src)
        if edge_attr is not None:
            message = message + (self.lin_edge(edge_attr) if hasattr(self, 'lin_edge') else edge_attr)
        output = self.aggr_module(message.relu() + self.eps, dst, dim_size=size[1])
        if hasattr(self, 'lin_aggr_out'):
            output = self.lin_aggr_out(output)
        if hasattr(self, 'msg_norm'):
            output = self.msg_norm(projected if target is None else target, output)
        if target is not None:
            output = output + (self.lin_dst(target) if hasattr(self, 'lin_dst') else target)
        return self.mlp(output)
