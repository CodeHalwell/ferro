"""Native CPU f32 grouped aggregation Modules for rank-two graph features.

Index inputs may be unsorted; CSR pointers describe contiguous groups. Both
routes use native scatter/segment operations and retain feature gradients.
Extrema scatter ties follow Ferro's actual-neighbor convention. Combine-mode
max/min choose the first aggregator, matching Torch's dimension reduction.
"""
import math

from ._native import Tensor, cat, no_grad, scatter, segment_softmax, stack
from ._graph_aggr_base import Aggregation
from ._graph_utils import _capacity, _count, _cpu_tensor, _index_ids
from ._graph_conv import _configure, _features, _flags, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_norm import _sqrt_with_zero_subgradient
from .nn import Linear, Module, ModuleDict, ModuleList, MultiheadAttention, Parameter


def _layout(x, index, ptr, dim_size, dim):
    _eager()
    _features(x, 'x')
    if type(dim) is not int or dim not in (0, -2):
        raise NotImplementedError('graph aggregations support rank-two inputs with dim=0/-2 only')
    nodes, width = x.shape
    _capacity((nodes + 1,), 8)
    _capacity((nodes, width))
    if dim_size is not None:
        _count(dim_size, 'dim_size')
    if ptr is not None:
        if index is not None:
            raise ValueError('provide index or ptr, not both')
        _cpu_tensor(ptr, 'ptr', 'int64')
        offsets = _index_ids(ptr, 'ptr')
        if not offsets or offsets[0] != 0 or offsets[-1] != nodes or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError('ptr must start at zero, be nondecreasing, and end at the input row count')
        count = len(offsets) - 1
        if dim_size is not None and dim_size != count:
            raise ValueError('dim_size must equal len(ptr)-1')
        ids = [group for group, (start, end) in enumerate(zip(offsets, offsets[1:])) for _ in range(end - start)]
    else:
        ids = [0] * nodes if index is None else _index_ids(index, 'index', nodes)
        count = max(ids, default=-1) + 1 if dim_size is None else dim_size
        if any(group >= count for group in ids):
            raise ValueError('index contains a group outside dim_size')
    _capacity((count + 1,), 8)
    _capacity((count, width))
    return ids, count


class _Aggregation(Aggregation):
    def reset_parameters(self):
        pass

    def get_out_channels(self, in_channels):
        return _count(in_channels, 'in_channels')

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        _layout(x, index, ptr, dim_size, dim)
        self.get_out_channels(x.shape[1])


class _Reduction(_Aggregation):
    _builtin_init = True
    reduction: str

    def __init__(self, reduction):
        _configure(self, reduction=reduction)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, count = _layout(x, index, ptr, dim_size, dim)
        return scatter(x, ids, count, self.reduction)


class SumAggregation(_Reduction):
    _builtin_init = True

    def __init__(self):
        super().__init__('sum')


class MeanAggregation(_Reduction):
    _builtin_init = True

    def __init__(self):
        super().__init__('mean')


class MaxAggregation(_Reduction):
    _builtin_init = True

    def __init__(self):
        super().__init__('max')


class MinAggregation(_Reduction):
    _builtin_init = True

    def __init__(self):
        super().__init__('min')


class VarAggregation(_Aggregation):
    _builtin_init = True
    semi_grad: bool

    def __init__(self, semi_grad=False):
        _flags(semi_grad=semi_grad)
        _configure(self, semi_grad=semi_grad)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, count = _layout(x, index, ptr, dim_size, dim)
        mean = scatter(x, ids, count, 'mean')
        square = x.square()
        mean2 = scatter(square.detach() if self.semi_grad else square, ids, count, 'mean')
        return mean2 - mean.square()


class StdAggregation(_Aggregation):
    _builtin_init = True

    def __init__(self, semi_grad=False):
        _configure(self)
        self.var_aggr = VarAggregation(semi_grad)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        variance = self.var_aggr(x, index, ptr, dim_size, dim)
        result = variance.clamp(1e-5, float('inf')).sqrt()
        return result.masked_fill(result.le(math.sqrt(1e-5)), 0.)


class _ScalarAggregation(_Aggregation):
    initial_value: float
    learn: bool
    channels: int

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        if self.channels != 1 and self.channels != in_channels:
            raise ValueError('channels must match the input feature width')
        return in_channels

    def __getattr__(self, name):
        if name == type(self)._scalar_name and not self.learn:
            return self.initial_value
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == type(self)._scalar_name and self.__dict__.get('learn') is False:
            raise AttributeError(name + ' is read-only configuration; construct a new aggregation')
        super().__setattr__(name, value)

    def reset_parameters(self):
        if self.learn:
            with no_grad():
                getattr(self, type(self)._scalar_name).tensor().copy_(Tensor.ones([self.channels]) * self.initial_value)

    def _value(self, x):
        if self.channels != 1 and self.channels != x.shape[1]:
            raise ValueError('channels must match the input feature width')
        return getattr(self, type(self)._scalar_name).tensor() if self.learn else self.initial_value


def _scalar(value, learn, channels, name):
    _flags(learn=learn)
    _positive(channels, 'channels')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite scalar')
    if not learn and channels != 1:
        raise ValueError('channels != 1 requires a learned parameter')
    _capacity((channels,))
    return float(value)


class SoftmaxAggregation(_ScalarAggregation):
    """Featurewise softmax weighting, optionally with learned temperatures."""
    _builtin_init = True
    _scalar_name = 't'
    semi_grad: bool

    def __init__(self, t=1., learn=False, semi_grad=False, channels=1):
        t = _scalar(t, learn, channels, 't')
        _flags(semi_grad=semi_grad)
        if learn and semi_grad:
            raise ValueError('learned temperature and semi_grad cannot both be enabled')
        _configure(self, initial_value=t, learn=learn, semi_grad=semi_grad, channels=channels)
        if learn:
            self.t = Parameter(Tensor.ones([channels]) * t)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, count = _layout(x, index, ptr, dim_size, dim)
        temperature = self._value(x)
        alpha = segment_softmax(x * temperature, ids, count)
        if self.semi_grad:
            alpha = alpha.detach()
        return scatter(x * alpha, ids, count, 'sum')


class PowerMeanAggregation(_ScalarAggregation):
    """Clamped generalized means with an optional learned nonzero exponent.

    A learned exponent uses native exp(log(x)*p), preserving its derivative.
    Learned exponents require a positive lower clamp. Optimizers must keep p
    away from zero for finite results. Constant
    p=1 bypasses clamping exactly as PyG; learned p=1 still uses the clamp path.
    """
    _builtin_init = True
    _scalar_name = 'p'
    min_value: object
    max_value: object

    def __init__(self, p=1., learn=False, channels=1, clamp_min=1e-4, clamp_max=100.):
        p = _scalar(p, learn, channels, 'p')
        if p == 0:
            raise ValueError('p must be nonzero')
        for value, name in ((clamp_min, 'clamp_min'), (clamp_max, 'clamp_max')):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(name + ' must be finite or None')
        if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
            raise ValueError('clamp_min cannot exceed clamp_max')
        if clamp_min is None and clamp_max is None and (learn or p != 1.):
            raise ValueError('at least one clamp bound is required on the power path')
        if learn and (clamp_min is None or clamp_min <= 0):
            raise NotImplementedError('learned powers require clamp_min > 0 for the native logarithmic derivative')
        _configure(self, initial_value=p, learn=learn, channels=channels,
                   min_value=clamp_min, max_value=clamp_max)
        if learn:
            self.p = Parameter(Tensor.ones([channels]) * p)

    def _clamp(self, x):
        return x.clamp(-float('inf') if self.min_value is None else self.min_value,
                       float('inf') if self.max_value is None else self.max_value)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, count = _layout(x, index, ptr, dim_size, dim)
        power = self._value(x)
        if not self.learn and power == 1.:
            return scatter(x, ids, count, 'mean')
        values = self._clamp(x)
        values = (values.log() * power).exp() if self.learn else values.pow(power)
        result = self._clamp(scatter(values, ids, count, 'mean'))
        return (result.log() / power).exp() if self.learn else result.pow(1. / power)


class _AggregationHeads(Module):
    _builtin_init = True
    in_channels: tuple
    out_channels: int

    def __init__(self, inputs, output):
        _configure(self, in_channels=inputs, out_channels=output)
        self.lins = ModuleDict({str(i): Linear(width, output) for i, width in enumerate(inputs)})
        self.reset_parameters()

    def reset_parameters(self):
        for _, layer in self.lins.items():
            _reset_linear(layer)

    def forward(self, values):
        return [self.lins[str(i)](value) for i, value in enumerate(values)]


class MultiAggregation(_Aggregation):
    """Registered aggregators combined by cat, a dense reduction or projection.

    Supported modes: cat/sum/mean/min/max/logsumexp/std/var/proj/attn. Projection
    and attention require in_channels and out_channels. Attention accepts
    num_heads/dropout/bias and uses checkpointed Ferro dropout streams.
    No fused reduction path or extra graph engine is used.
    """
    _builtin_init = True
    mode: str
    in_channels: object
    out_channels: object
    attention_heads: object
    attention_dropout: float
    attention_bias: bool

    def __init__(self, aggrs, aggrs_kwargs=None, mode='cat', mode_kwargs=None):
        if not isinstance(aggrs, (list, tuple)) or not aggrs:
            raise ValueError('aggrs must be a nonempty list or tuple')
        if aggrs_kwargs is None:
            aggrs_kwargs = [{} for _ in aggrs]
        if not isinstance(aggrs_kwargs, (list, tuple)) or len(aggrs_kwargs) != len(aggrs) or any(type(value) is not dict for value in aggrs_kwargs):
            raise ValueError('aggrs_kwargs must contain one dictionary per aggregation')
        if mode not in ('cat', 'sum', 'mean', 'min', 'max', 'logsumexp', 'std', 'var', 'proj', 'attn'):
            raise NotImplementedError('unsupported MultiAggregation mode')
        constructors = dict(sum=SumAggregation, add=SumAggregation, mean=MeanAggregation, min=MinAggregation,
                            max=MaxAggregation, var=VarAggregation, std=StdAggregation,
                            softmax=SoftmaxAggregation, powermean=PowerMeanAggregation)
        if any(isinstance(aggr, str) and aggr in ('lstm', 'gru') for aggr in aggrs):
            from ._graph_recurrent_aggr import LSTMAggregation, GRUAggregation
            constructors.update(lstm=LSTMAggregation, gru=GRUAggregation)
        if any(isinstance(aggr, str) and aggr in ('quantile', 'median') for aggr in aggrs):
            from ._graph_quantile import QuantileAggregation, MedianAggregation
            constructors.update(quantile=QuantileAggregation, median=MedianAggregation)
        if any(isinstance(aggr, str) and aggr in ('variance_preserving', 'variancepreserving') for aggr in aggrs):
            from ._graph_variance_aggr import VariancePreservingAggregation
            constructors.update(variance_preserving=VariancePreservingAggregation, variancepreserving=VariancePreservingAggregation)
        if 'sort' in aggrs:
            from ._graph_sort_aggr import SortAggregation
            constructors['sort'] = SortAggregation
        if 'graph_multiset_transformer' in aggrs:
            from ._graph_multiset import GraphMultisetTransformer
            constructors['graph_multiset_transformer'] = GraphMultisetTransformer
        if 'set_transformer' in aggrs:
            from ._graph_set_transformer import SetTransformerAggregation
            constructors['set_transformer'] = SetTransformerAggregation
        if 'set2set' in aggrs:
            from ._graph_advanced import Set2Set
            constructors['set2set'] = Set2Set
        if any(isinstance(aggr, str) and aggr in ('deep_sets', 'deepsets') for aggr in aggrs):
            from ._graph_deep_sets import DeepSetsAggregation
            constructors.update(deep_sets=DeepSetsAggregation, deepsets=DeepSetsAggregation)
        if 'attentional' in aggrs:
            from ._graph_advanced import AttentionalAggregation
            constructors['attentional'] = AttentionalAggregation
        if 'degree_scaler' in aggrs:
            from ._graph_degree_aggr import DegreeScalerAggregation
            constructors['degree_scaler'] = DegreeScalerAggregation
        modules = []
        for aggregation, kwargs in zip(aggrs, aggrs_kwargs):
            if isinstance(aggregation, str) and aggregation in constructors:
                modules.append(constructors[aggregation](**kwargs))
            elif isinstance(aggregation, Aggregation) and not kwargs:
                modules.append(aggregation)
            else:
                raise NotImplementedError('aggrs must contain supported names or aggregation Modules without extra kwargs')
        if mode_kwargs is not None and type(mode_kwargs) is not dict:
            raise TypeError('mode_kwargs must be a dictionary or None')
        options = dict(mode_kwargs or {})
        inputs, output, bias = None, None, True
        heads, dropout = None, 0.
        if mode in ('proj', 'attn'):
            if len(modules) < 2:
                raise ValueError(mode + ' requires multiple aggregations')
            inputs, output = options.pop('in_channels', None), options.pop('out_channels', None)
            inputs = (inputs,) * len(modules) if type(inputs) is int else tuple(inputs) if isinstance(inputs, (list, tuple)) else ()
            if len(inputs) != len(modules):
                raise ValueError('projection requires one in_channels value per aggregation')
            for width in inputs:
                _positive(width, 'in_channels')
            _positive(output, 'out_channels')
            bias = options.pop('bias', True)
            _flags(bias=bias)
            _capacity((sum(inputs), output))
            if mode == 'attn':
                heads, dropout = options.pop('num_heads', 1), options.pop('dropout', 0.)
                _positive(heads, 'num_heads')
                if output % heads:
                    raise ValueError('out_channels must be divisible by num_heads')
                if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
                    raise ValueError('dropout must be finite and in [0,1]')
                _capacity((4, output, output))
        if options:
            raise NotImplementedError('unsupported aggregation combine options: ' + ', '.join(sorted(options)))
        _configure(self, mode=mode, in_channels=inputs, out_channels=output,
                   attention_heads=heads, attention_dropout=float(dropout), attention_bias=bias)
        self.aggrs = ModuleList(modules)
        if mode == 'proj':
            self.lin = Linear(sum(inputs), output, bias=bias)
            _reset_linear(self.lin)
        elif mode == 'attn':
            self.lin_heads = _AggregationHeads(inputs, output)
            self.multihead_attn = MultiheadAttention(output, heads, dropout=dropout, bias=bias, batch_first=False)

    def reset_parameters(self):
        for aggregation in self.aggrs:
            aggregation.reset_parameters()
        if self.mode == 'proj':
            _reset_linear(self.lin)
        elif self.mode == 'attn':
            self.lin_heads.reset_parameters()
            # Torch MHA's reset redraws the packed input map, zeros biases and
            # preserves out_proj.weight; dropout streams are not reset.
            with no_grad():
                width = self.out_channels
                self.multihead_attn.in_proj_weight.tensor().copy_((Tensor.rand([3 * width, width]) * 2 - 1) * math.sqrt(6. / (4 * width)))
                if self.multihead_attn.in_proj_bias is not None:
                    self.multihead_attn.in_proj_bias.tensor().copy_(Tensor.zeros([3 * width]))
                    self.multihead_attn.out_proj.bias.tensor().copy_(Tensor.zeros([width]))

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        widths = tuple(_count(aggregation.get_out_channels(in_channels), 'aggregation output width') for aggregation in self.aggrs)
        if self.mode in ('proj', 'attn'):
            if widths != self.in_channels:
                raise ValueError('projection in_channels must match each aggregation output width')
            return self.out_channels
        if self.mode == 'cat':
            return sum(widths)
        if len(set(widths)) != 1:
            raise ValueError('aggregation reduction requires equal output widths')
        return widths[0]

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        _, count = _layout(x, index, ptr, dim_size, dim)
        widths = tuple(_count(aggregation.get_out_channels(x.shape[1]), 'aggregation output width') for aggregation in self.aggrs)
        output_width = self.get_out_channels(x.shape[1])
        _capacity((count, sum(widths)))
        _capacity((count, output_width))
        if self.mode in ('min', 'max'):
            _capacity((count, output_width), 8)
        if self.mode in ('proj', 'attn'):
            if self.mode == 'attn':
                _capacity((count, self.attention_heads, len(self.aggrs), len(self.aggrs)))
                _capacity((len(self.aggrs), count, self.out_channels))
        for aggregation in self.aggrs:
            aggregation._preflight(x, index, ptr, dim_size, dim)
        return count, widths

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        count, widths = self._preflight(x, index, ptr, dim_size, dim)
        values = [aggregation(x, index, ptr, dim_size, dim) for aggregation in self.aggrs]
        if any(value.shape != [count, width] for value, width in zip(values, widths)):
            raise ValueError('aggregation output shape disagrees with its declared channel width')
        if len(values) == 1:
            return values[0]
        if self.mode == 'cat':
            return cat(values, 1)
        if self.mode == 'proj':
            return self.lin(cat(values, 1))
        if self.mode == 'attn':
            combined = stack(self.lin_heads(values), 0)
            output, _ = self.multihead_attn(combined, combined, combined)
            return output.mean_dim(0, False)
        combined = stack(values, 0)
        if self.mode in ('min', 'max'):
            indices = combined.argmax(0, True) if self.mode == 'max' else (combined * -1).argmax(0, True)
            return combined.gather(0, indices).squeeze(0)
        if self.mode == 'logsumexp':
            return combined.logsumexp(0)
        if self.mode in ('std', 'var'):
            mean = combined.mean_dim(0, True)
            variance = (combined - mean).square().sum_dim(0, False) / (len(values) - 1)
            return _sqrt_with_zero_subgradient(variance) if self.mode == 'std' else variance
        return combined.sum_dim(0, False) if self.mode == 'sum' else combined.mean_dim(0, False)
