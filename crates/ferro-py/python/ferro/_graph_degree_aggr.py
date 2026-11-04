"""Reusable PNA degree scaling around registered native graph aggregators."""
import math

from ._native import Tensor, cat
from ._graph_aggregation import _Aggregation, _layout, MultiAggregation
from ._graph_advanced import _DegreeScalers
from ._graph_conv import _flags
from ._graph_utils import _capacity, _count, _cpu_tensor
from ._graph_gen import _aggregation


class DegreeScalerAggregation(_DegreeScalers, _Aggregation):
    """Aggregate messages and concatenate degree-scaled readouts.

    CPU float32 rank-two inputs use explicit groups or the default single
    group. ptr and capture reject. The training histogram is copied into
    immutable configuration; nonidentity scales require positive training
    degree. Learned normalization parameters must remain nonsingular.
    A scaler string or nonempty list works independently of aggregation type.
    reset_parameters resets only normalization, preserving child state as in
    PyG. Ordinary empty groups are supported by the selected aggregator.
    """
    _builtin_init = True
    scaler: tuple

    def __init__(self, aggr, scaler, deg, train_norm=False, aggr_kwargs=None):
        _flags(train_norm=train_norm)
        scalers = (scaler,) if isinstance(scaler, str) else tuple(scaler) if isinstance(scaler, (list, tuple)) else ()
        if not scalers or any(value not in ('identity', 'amplification', 'attenuation', 'linear', 'inverse_linear') for value in scalers):
            raise ValueError('scaler must contain supported degree scaling names')
        _cpu_tensor(deg, 'deg', dtype=None)
        if deg.dtype not in ('int64', 'float32') or len(deg.shape) != 1 or not deg.shape[0]:
            raise ValueError('deg must be a nonempty int64/float32 degree histogram')
        _capacity(deg.shape, 8 if deg.dtype == 'int64' else 4)
        histogram = tuple(deg.tolist())
        if any(not math.isfinite(value) or value < 0 or int(value) != value for value in histogram) or not sum(histogram) > 0:
            raise ValueError('degree histogram must contain nonnegative integer counts and nonzero total')
        histogram = tuple(int(value) for value in histogram)
        if not any(histogram[1:]) and any(value != 'identity' for value in scalers):
            raise ValueError('nonidentity scalers require positive training degree')
        _capacity((len(scalers),))
        if isinstance(aggr, (tuple, list)):
            aggregate = MultiAggregation(aggr, aggrs_kwargs=aggr_kwargs)
        else:
            if aggr_kwargs is not None and type(aggr_kwargs) is not dict:
                raise TypeError('a single aggregation requires dictionary aggr_kwargs')
            aggregate = _aggregation(aggr, {} if aggr_kwargs is None else dict(aggr_kwargs))
        _DegreeScalers.__init__(self, histogram, train_norm)
        object.__setattr__(self, 'scaler', scalers)
        self.aggr = aggregate

    def get_out_channels(self, in_channels):
        return _count(self.aggr.get_out_channels(in_channels) * len(self.scaler), 'aggregation output width')

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        if ptr is not None: raise NotImplementedError('degree scaling requires index input; ptr is unsupported')
        ids, groups = _layout(x, index, None, dim_size, dim)
        _capacity((groups, self.get_out_channels(x.shape[1])))
        self.aggr._preflight(x, index, None, dim_size, dim)
        return ids, groups

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, groups = self._preflight(x, index, ptr, dim_size, dim)
        output = self.aggr(x, index, None, dim_size, dim)
        if output.numel() == 0:
            degree = output.detach()
        else:
            counts = [0] * groups
            for group in ids: counts[group] += 1
            degree = Tensor([float(value) for value in counts], [groups, 1])
        average_linear, average_log = self.values()
        values = []
        for scaler in self.scaler:
            if scaler == 'identity': value = output
            elif scaler == 'amplification': value = output * ((degree + 1.).log() / average_log)
            elif scaler == 'attenuation': value = output * (average_log / (degree.clamp(1., float('inf')) + 1.).log())
            elif scaler == 'linear': value = output * (degree / average_linear)
            else: value = output * (average_linear / degree.clamp(1., float('inf')))
            values.append(value)
        return cat(values, 1) if len(values) > 1 else values[0]
