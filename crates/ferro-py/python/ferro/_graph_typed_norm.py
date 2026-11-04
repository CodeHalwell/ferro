"""CPU F32 graph/type normalization with explicit PyG state semantics.

InstanceNorm's explicit graph path keeps live running-statistic history, as
PyG does. That path cannot rebind registered buffers under Trainer's structure
lock. Resetting live statistics requires no_grad until general inplace history
updates are supported. Heterogeneous BatchNorm intentionally detaches its
statistics and uses a clamped biased variance, not ordinary BatchNorm algebra.
"""
import math

from ._native import Tensor, no_grad, is_grad_enabled, scatter, where
from ._graph_conv import _configure, _features, _flags
from ._graph_norm import _batch, _device, _epsilon
from ._graph_utils import _capacity, _count, _cpu_tensor, _index_ids
from ._graph_conditioned import _eager
from .nn import Module, Parameter
from .nn import functional as F


def _dimensions(channels, types=None):
    _count(channels, 'in_channels')
    if types is not None:
        _count(types, 'num_types')
        _capacity((types + 1,), 8)
        _capacity((types, channels))
    else:
        _capacity((channels,))


def _momentum(value):
    if value is not None and type(value) not in (int, float):
        raise TypeError('momentum must be a real scalar or None')
    return None if value is None else float(value)


def _affine(module, shape):
    module.weight = Parameter(Tensor.ones(shape)) if module.affine else None
    module.bias = Parameter(Tensor.zeros(shape)) if module.affine else None


def _buffers(module, shape):
    module.register_buffer('running_mean', Tensor.zeros(shape) if module.track_running_stats else None)
    module.register_buffer('running_var', Tensor.ones(shape) if module.track_running_stats else None)
    module.register_buffer('num_batches_tracked', Tensor.from_i64([0], []) if module.track_running_stats else None)


def _state(module, shape):
    if not module.track_running_stats:
        return
    for name in ('running_mean', 'running_var'):
        value = _cpu_tensor(getattr(module, name), name)
        if value.shape != list(shape):
            raise ValueError(name + ' shape differs from configured state')
    counter = _cpu_tensor(module.num_batches_tracked, 'num_batches_tracked', 'int64')
    if counter.shape:
        raise ValueError('num_batches_tracked must be a scalar int64 buffer')


def _reset_affine(module):
    with no_grad():
        if module.weight is not None:
            module.weight.tensor().fill_(1.)
            module.bias.tensor().fill_(0.)


def _types(x, type_vec, count):
    _cpu_tensor(type_vec, 'type_vec', 'int64')
    if type_vec.shape != [x.shape[0]]:
        raise ValueError('type_vec must contain one type ID per input row')
    _capacity(type_vec.shape, 8)
    ids = _index_ids(type_vec, 'type_vec', x.shape[0])
    if any(i >= count for i in ids):
        raise ValueError('type_vec contains a type outside num_types')
    return ids


class InstanceNorm(Module):
    """Per-graph normalization with native no-batch and live explicit-batch paths.

    Input is rank-two CPU F32 with the configured channel count. Explicit empty
    batches require batch_size; absent groups enter running-stat averages.
    No-batch singleton input-stat calls reject. The counter remains zero.
    """
    _builtin_init = True
    num_features: int
    eps: float
    momentum: object
    affine: bool
    track_running_stats: bool

    def __init__(self, in_channels, eps=1e-5, momentum=.1, affine=False, track_running_stats=False, device=None):
        _device(device); _dimensions(in_channels)
        _flags(affine=affine, track_running_stats=track_running_stats)
        _configure(self, num_features=in_channels, eps=_epsilon(eps), momentum=_momentum(momentum), affine=affine, track_running_stats=track_running_stats)
        _affine(self, [in_channels]); _buffers(self, [in_channels])

    def reset_running_stats(self):
        _state(self, [self.num_features])
        if self.track_running_stats:
            if is_grad_enabled() and (self.running_mean.requires_grad or self.running_var.requires_grad):
                raise NotImplementedError('reset live InstanceNorm statistics inside no_grad; inplace history updates are unsupported')
            with no_grad():
                self.running_mean.detach().fill_(0.)
                self.running_var.detach().fill_(1.)
                self.num_batches_tracked.copy_(Tensor.from_i64([0], []))

    def reset_parameters(self):
        self.reset_running_stats(); _reset_affine(self)

    def forward(self, x, batch=None, batch_size=None):
        _eager(); _features(x, 'x', self.num_features); _capacity(x.shape)
        _state(self, [self.num_features])
        use_stats = self.training or not self.track_running_stats
        if batch is None:
            if use_stats and x.shape[0] == 1:
                raise ValueError('InstanceNorm needs more than one spatial element')
            if self.momentum is None:
                raise TypeError('native no-batch InstanceNorm requires a numeric momentum')
            mean, variance = self.running_mean, self.running_var
            if not is_grad_enabled():
                mean = None if mean is None else mean.detach()
                variance = None if variance is None else variance.detach()
            if self.num_features == 0:
                if is_grad_enabled() and any(v is not None and v.requires_grad for v in (mean, variance)):
                    raise RuntimeError('native InstanceNorm does not differentiate running statistics')
                if self.weight is not None:
                    raise IndexError('native no-batch InstanceNorm does not support zero channels with affine tensors')
                return x.clone()
            return F.instance_norm(x.T.unsqueeze(0), mean, variance,
                None if self.weight is None else self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                use_stats, self.momentum, self.eps).squeeze(0).T
        ids, count = _batch(x, batch, batch_size, self.num_features, True)
        if use_stats:
            if self.track_running_stats and self.momentum is None:
                raise TypeError('tracked explicit-batch InstanceNorm requires numeric momentum in training')
            if self.track_running_stats and self._structure_locked:
                raise RuntimeError('live InstanceNorm running-stat rebinding is unsupported under Trainer structure lock')
            if self.num_features == 0:
                # No channel has a degree divisor; preserve the actual segment
                # graph without constructing an irrelevant count-sized vector.
                mean = scatter(x, ids, count, 'mean')
                centered = x - mean.index_select(0, ids)
                squares = scatter(centered.square(), ids, count, 'sum')
                variance, unbiased_variance = squares, squares
            else:
                degree = [0] * count
                for i in ids: degree[i] += 1
                divisor = Tensor([float(max(1, n)) for n in degree], [count, 1])
                unbiased = Tensor([float(max(1, n - 1)) for n in degree], [count, 1])
                mean = scatter(x, ids, count, 'sum') / divisor
                centered = x - mean.index_select(0, ids)
                squares = scatter(centered.square(), ids, count, 'sum')
                variance, unbiased_variance = squares / divisor, squares / unbiased
            if self.running_mean is not None:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.mean_dim(0, False)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_variance.mean_dim(0, False)
        else:
            mean = self.running_mean.reshape([1, self.num_features]).expand([count, self.num_features])
            variance = self.running_var.reshape([1, self.num_features]).expand([count, self.num_features])
            centered = x - mean.index_select(0, ids)
        output = centered / (variance + self.eps).sqrt().index_select(0, ids)
        return output if self.weight is None else output * self.weight.tensor().reshape([1, self.num_features]) + self.bias.tensor().reshape([1, self.num_features])


class HeteroBatchNorm(Module):
    """Per-type detached statistics, biased variance and present-type updates.

    The counter advances only with momentum=None, even on empty inputs, and
    follows native signed-int64 wraparound. Missing types retain saved values.
    Manually differentiable running statistics require no_grad for reset.
    """
    _builtin_init = True
    in_channels: int
    num_types: int
    eps: float
    momentum: object
    affine: bool
    track_running_stats: bool

    def __init__(self, in_channels, num_types, eps=1e-5, momentum=.1, affine=True, track_running_stats=True, device=None):
        _device(device); _dimensions(in_channels, num_types)
        _flags(affine=affine, track_running_stats=track_running_stats)
        _configure(self, in_channels=in_channels, num_types=num_types, eps=_epsilon(eps), momentum=_momentum(momentum), affine=affine, track_running_stats=track_running_stats)
        shape = [num_types, in_channels]; _affine(self, shape); _buffers(self, shape)

    def reset_running_stats(self):
        _state(self, [self.num_types, self.in_channels])
        if self.track_running_stats:
            if is_grad_enabled() and (self.running_mean.requires_grad or self.running_var.requires_grad):
                raise NotImplementedError('reset live HeteroBatchNorm statistics inside no_grad; inplace history updates are unsupported')
            with no_grad():
                self.running_mean.detach().fill_(0.); self.running_var.detach().fill_(1.)
                self.num_batches_tracked.copy_(Tensor.from_i64([0], []))

    def reset_parameters(self):
        self.reset_running_stats(); _reset_affine(self)

    def forward(self, x, type_vec):
        _eager(); _features(x, 'x', self.in_channels); _capacity(x.shape)
        _capacity((self.num_types, self.in_channels)); _capacity((self.num_types + 1,), 8)
        ids = _types(x, type_vec, self.num_types)
        _state(self, [self.num_types, self.in_channels])
        if self.track_running_stats and self.running_mean._shares_storage(self.running_var):
            raise NotImplementedError('running_mean and running_var require independent storage')
        if not self.training and self.track_running_stats:
            mean, variance = self.running_mean, self.running_var
        else:
            with no_grad():
                if self.in_channels == 0:
                    mean = scatter(x, ids, self.num_types, 'mean')
                    variance = scatter(x.square(), ids, self.num_types, 'mean') - mean.square()
                else:
                    degree = [0] * self.num_types
                    for i in ids: degree[i] += 1
                    divisor = Tensor([float(max(1, n)) for n in degree], [self.num_types, 1])
                    mean = scatter(x, ids, self.num_types, 'sum') / divisor
                    variance = scatter(x.square(), ids, self.num_types, 'sum') / divisor - mean.square()
        if self.training and self.track_running_stats:
            with no_grad():
                if self.momentum is None:
                    self.num_batches_tracked.copy_(self.num_batches_tracked + Tensor.from_i64([1], []))
                    factor = 1. / float(self.num_batches_tracked.item())
                else:
                    factor = self.momentum
                if self.in_channels == 0:
                    self.running_mean.copy_(self.running_mean)
                    self.running_var.copy_(self.running_var)
                else:
                    observed = set(ids)
                    present = Tensor.from_bool([i in observed for i in range(self.num_types)], [self.num_types, 1])
                    self.running_mean.copy_(where(present, (1 - factor) * self.running_mean + factor * mean, self.running_mean))
                    self.running_var.copy_(where(present, (1 - factor) * self.running_var + factor * variance, self.running_var))
        output = (x - mean.index_select(0, ids)) / variance.clamp(self.eps, float('inf')).sqrt().index_select(0, ids)
        return output if self.weight is None else output * self.weight.tensor().index_select(0, ids) + self.bias.tensor().index_select(0, ids)


class HeteroLayerNorm(Module):
    """Node LayerNorm with per-type affine parameters and checked grouping.

    type_ptr takes precedence when both selectors are provided. Pointers must
    completely cover every row with one segment per configured type. With
    affine=False, selectors are required but otherwise unused, as in PyG.
    """
    _builtin_init = True
    in_channels: int
    num_types: int
    eps: float
    affine: bool
    mode: str

    def __init__(self, in_channels, num_types, eps=1e-5, affine=True, mode='node', device=None):
        _device(device); _dimensions(in_channels, num_types); _flags(affine=affine)
        if mode != 'node': raise NotImplementedError('HeteroLayerNorm supports mode=node only')
        _configure(self, in_channels=in_channels, num_types=num_types, eps=_epsilon(eps), affine=affine, mode=mode)
        _affine(self, [num_types, in_channels])

    def reset_parameters(self):
        _reset_affine(self)

    def forward(self, x, type_vec=None, type_ptr=None):
        _eager(); _features(x, 'x', self.in_channels); _capacity(x.shape)
        if type_vec is None and type_ptr is None:
            raise ValueError('type_vec or type_ptr is required')
        ids = None
        if self.affine:
            _capacity((x.shape[0],), 8)
            if type_ptr is not None:
                if isinstance(type_ptr, Tensor):
                    _cpu_tensor(type_ptr, 'type_ptr', 'int64')
                    if type_ptr.shape != [self.num_types + 1]:raise ValueError('type_ptr must have num_types+1 entries')
                    _capacity(type_ptr.shape, 8)
                offsets = _index_ids(type_ptr, 'type_ptr', self.num_types + 1)
                if not offsets or offsets[0] != 0 or offsets[-1] != x.shape[0] or any(a > b for a,b in zip(offsets, offsets[1:])):
                    raise ValueError('type_ptr must cover every row with nondecreasing offsets')
                ids = [i for i, (a,b) in enumerate(zip(offsets, offsets[1:])) for _ in range(b-a)]
            else:
                ids = _types(x, type_vec, self.num_types)
            _capacity((x.shape[0],), 8)
        output = F.layer_norm(x, (self.in_channels,), None, None, self.eps)
        return output if not self.affine else output * self.weight.tensor().index_select(0, ids) + self.bias.tensor().index_select(0, ids)
