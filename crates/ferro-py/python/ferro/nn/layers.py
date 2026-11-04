"""torch-shaped layers over native ops. Initializers are time-seeded like Linear."""
import math
import os
from .._native import Tensor
from .module import Module, ModuleList, Parameter
from . import functional as F


def _positive(*values):
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values):
        raise ValueError('Layer sizes must be positive integers')


def _uniform(shape, bound):
    return (Tensor.rand(list(shape)) * 2 - 1) * bound


class Sequential(ModuleList):
    _builtin_init = True

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class Identity(Module):
    _builtin_init = True

    def __init__(self, *args, **kwargs):
        self._initialize()

    def forward(self, x):
        return x


class Flatten(Module):
    _builtin_init = True

    def __init__(self, start_dim=1, end_dim=-1):
        self._initialize()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


from .spatial import _Convolution, _MaxPool, _AvgPool


class Conv2d(_Convolution):
    """CPU grouped NCHW/CHW convolution with same/valid and all padding modes."""
    _builtin_init, _dimensions, _transposed = True, 2, False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        self._setup(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                    bias, padding_mode, 0, device, dtype)

    def forward(self, input):
        from .. import _spatial
        input, padding = _spatial._conv_padding(input, self.kernel_size, self.padding, self.dilation, self.padding_mode)
        return _spatial.conv2d(input, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                               self.stride, padding, self.dilation, self.groups)


class MaxPool2d(_MaxPool):
    """CPU rectangular max pooling with padding/dilation/ceil and typed indices."""
    _dimensions = 2


class AvgPool2d(_AvgPool):
    """CPU rectangular average pooling with padding/ceil and divisor options."""
    _dimensions = 2


class Dropout(Module):
    """Philox dropout with an exact persistent seed/counter buffer.

    CPU f32 buffer chunks encode all uint64 bits without float rounding. Masks
    reproduce after checkpoint restore; they do not match Torch's RNG sequence.
    """
    _builtin_init = True

    def __init__(self, p=0.5, inplace=False, *, seed=None):
        if type(p) not in (int, float) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError('dropout probability must be finite and in [0, 1]')
        if type(inplace) is not bool:
            raise TypeError('inplace must be bool')
        if inplace:
            raise NotImplementedError('inplace dropout is unsupported')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError('seed must be an unsigned 64-bit integer')
        self._initialize()
        self.p = float(p)
        self.inplace = False
        seed = int.from_bytes(os.urandom(8), 'little') if seed is None else seed
        self.register_buffer('rng_state', self._encode(seed, 0))

    @staticmethod
    def _encode(seed, offset):
        return Tensor([float((value >> shift) & 65535) for value in (seed, offset) for shift in (0, 16, 32, 48)], [8])

    @staticmethod
    def _validate_rng_buffer(state):
        if not isinstance(state, Tensor) or state.dtype != 'float32' or state.device != 'cpu' or state.shape != [8]:
            raise ValueError('dropout RNG state must be a CPU float32 buffer of shape [8]')
        values = state.tolist()
        if any(not math.isfinite(value) or value != int(value) or not 0 <= value <= 65535 for value in values):
            raise ValueError('invalid dropout RNG state chunks')
        return tuple(sum(int(values[base + i]) << shift for i, shift in enumerate((0, 16, 32, 48))) for base in (0, 4))

    def _stream(self):
        return Dropout._validate_rng_buffer(self.rng_state)

    @property
    def _seed(self):
        return self._stream()[0]

    @property
    def _offset(self):
        return self._stream()[1]

    def forward(self, input):
        if not self.training or self.p == 0.0:
            return F.dropout(input, self.p, self.training)
        if self.p == 1.0:
            return F.dropout(input, self.p, True)
        seed, offset = self._stream()
        y = F.dropout(input, self.p, True, seed=seed, offset=offset)
        if input.numel():
            from .._native import no_grad
            with no_grad():
                self.rng_state.copy_(self._encode(seed, offset + input.numel()))
        return y


class Embedding(Module):
    """CPU float32 or float64 parameter table with dense eager gradients.

    Supplied/pretrained weights share whole-contiguous CPU storage with a new
    leaf parameter and determine dtype/device independently of factory options.
    Sparse gradients and strided shared parameter slots reject.
    """
    _builtin_init = True
    padding_idx = None
    max_norm = None
    norm_type = 2.
    scale_grad_by_freq = False
    sparse = False

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2., scale_grad_by_freq=False, sparse=False, _weight=None,
                 _freeze=False, device=None, dtype=None):
        from .._tensor_api import zeros
        from ._embedding import options, parameter_options
        shape, dtype = parameter_options(num_embeddings, embedding_dim, _weight, device, dtype)
        num_embeddings, embedding_dim = shape
        if type(_freeze) is not bool:
            raise TypeError('_freeze must be bool')
        padding_idx, max_norm, norm_type = options(padding_idx, num_embeddings, max_norm, norm_type, scale_grad_by_freq, sparse)
        self._initialize()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        for name, value in (('padding_idx', padding_idx), ('max_norm', max_norm), ('norm_type', norm_type), ('scale_grad_by_freq', scale_grad_by_freq), ('sparse', sparse)):
            if value != getattr(type(self), name):
                setattr(self, name, value)
        initial = (Tensor.zeros(shape) if dtype == 'float32' else zeros(shape, dtype=dtype)) if _weight is None else _weight
        self.weight = Parameter._from_shared(initial, requires_grad=not _freeze)
        if _weight is None:
            self.reset_parameters()

    def reset_parameters(self):
        from .._native import no_grad
        from ._embedding import parameter_options
        weight = self.weight.tensor()
        shape, dtype = parameter_options(self.num_embeddings, self.embedding_dim, weight, None, None)
        with no_grad():
            weight.copy_(Tensor.randn(shape, dtype=dtype))
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None:
            from .._native import cat, no_grad
            from .._tensor_api import zeros
            with no_grad():
                weight = self.weight.tensor()
                zero = Tensor.zeros([1, self.embedding_dim]) if weight.dtype == 'float32' else zeros(1, self.embedding_dim, dtype=weight.dtype)
                parts = [weight.narrow(0, 0, self.padding_idx), zero,
                         weight.narrow(0, self.padding_idx + 1, self.num_embeddings - self.padding_idx - 1)]
                weight.copy_(cat(parts, 0))

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, max_norm=None,
                        norm_type=2., scale_grad_by_freq=False, sparse=False):
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise ValueError('from_pretrained expects a two-dimensional Tensor')
        return cls(*embeddings.shape, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=embeddings, _freeze=freeze)

    def forward(self, ids):
        return F.embedding(ids, self.weight.tensor(), self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)


class LayerNorm(Module):
    """Normalize a trailing shape with optional CPU float32/float64 parameters."""
    _builtin_init = True

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        from ._layer_norm import factory, epsilon
        from .._tensor_api import _dtype, ones, zeros
        shape = factory(normalized_shape, elementwise_affine, device, dtype, bias, affine_float64=True)
        dtype = _dtype(dtype)
        checked_eps = epsilon(eps, _dtype(dtype))
        self._initialize()
        self.normalized_shape, self.eps, self.elementwise_affine = shape, checked_eps, elementwise_affine
        self.weight = Parameter(Tensor.ones(list(shape)) if dtype == 'float32' else ones(shape, dtype=dtype)) if elementwise_affine else None
        self.bias = Parameter(Tensor.zeros(list(shape)) if dtype == 'float32' else zeros(shape, dtype=dtype)) if elementwise_affine and bias else None
        self.reset_parameters()

    def reset_parameters(self):
        from .._native import no_grad
        if self.elementwise_affine:
            with no_grad():
                self.weight.tensor().fill_(1.)
                if self.bias is not None:
                    self.bias.tensor().fill_(0.)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, None if self.weight is None else self.weight.tensor(),
                          None if self.bias is None else self.bias.tensor(), self.eps)


class RMSNorm(Module):
    """Normalize a trailing root-mean-square with CPU float32/float64 weights."""
    _builtin_init = True

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        from ._layer_norm import factory, epsilon
        from .._tensor_api import _dtype, ones
        shape = factory(normalized_shape, elementwise_affine, device, dtype, affine_float64=True)
        dtype = _dtype(dtype)
        checked_eps = None if eps is None else epsilon(eps, _dtype(dtype))
        self._initialize()
        self.normalized_shape, self.eps, self.elementwise_affine = shape, checked_eps, elementwise_affine
        self.weight = Parameter(Tensor.ones(list(shape)) if dtype == 'float32' else ones(shape, dtype=dtype)) if elementwise_affine else None
        self.reset_parameters()

    def reset_parameters(self):
        from .._native import no_grad
        if self.elementwise_affine:
            with no_grad():
                self.weight.tensor().fill_(1.)

    def forward(self, input):
        return F.rms_norm(input, self.normalized_shape, None if self.weight is None else self.weight.tensor(), self.eps)


class GroupNorm(Module):
    """CPU float32/float64 channel-group normalization over spatial dimensions."""
    _builtin_init = True

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None, *, bias=True):
        from ._layer_norm import factory, epsilon
        from .._tensor_api import _dtype, ones, zeros
        if type(num_groups) is not int or num_groups <= 0:
            raise ValueError('num_groups must be a positive integer')
        if type(num_channels) is not int or num_channels < 0 or num_channels % num_groups:
            raise ValueError('num_channels must be nonnegative and divisible by num_groups')
        factory((num_channels,), affine, device, dtype, bias, affine_float64=True)
        dtype = _dtype(dtype)
        checked_eps = epsilon(eps, _dtype(dtype))
        self._initialize()
        self.num_groups, self.num_channels, self.eps, self.affine = num_groups, num_channels, checked_eps, affine
        self.weight = Parameter(Tensor.ones([num_channels]) if dtype == 'float32' else ones([num_channels], dtype=dtype)) if affine else None
        self.bias = Parameter(Tensor.zeros([num_channels]) if dtype == 'float32' else zeros([num_channels], dtype=dtype)) if affine and bias else None
        self.reset_parameters()

    def reset_parameters(self):
        from .._native import no_grad
        if self.affine:
            with no_grad():
                self.weight.tensor().fill_(1.)
                if self.bias is not None:self.bias.tensor().fill_(0.)

    def forward(self, input):
        if not isinstance(input, Tensor) or input.ndim < 2 or input.shape[1] != self.num_channels:
            raise ValueError('GroupNorm expects [N,num_channels,*spatial] input')
        return F.group_norm(input, self.num_groups, None if self.weight is None else self.weight.tensor(),
                          None if self.bias is None else self.bias.tensor(), self.eps)


class _BatchNorm(Module):
    """CPU batch normalization with typed persistent running statistics.

    Float32/float64 affine parameters and computation are supported. Tracking
    uses an exact int64 counter, including empty batches
    and the training singleton failure, matching PyTorch's state machine.
    """
    _builtin_init = True
    _version = 2
    _ranks = ()

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None, *, bias=True):
        from numbers import Real
        from .._tensor_api import _device, _dtype, _shape, ones, zeros
        _positive(num_features)
        if any(type(value) is not bool for value in (affine, track_running_stats, bias)):
            raise TypeError('affine, track_running_stats and bias must be bool')
        if not isinstance(eps, Real) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('BatchNorm eps must be finite and positive')
        if momentum is not None and (not isinstance(momentum, Real) or isinstance(momentum, bool) or not math.isfinite(momentum)):
            raise ValueError('BatchNorm momentum must be finite or None')
        dtype, device = _dtype(dtype), _device(device)
        if device != 'cpu' or dtype not in ('float32', 'float64'):
            raise NotImplementedError('BatchNorm supports CPU float32/float64 storage')
        _shape([num_features], dtype)
        self._initialize()
        self.num_features, self.eps = num_features, float(eps)
        self.momentum = None if momentum is None else float(momentum)
        self.affine, self.track_running_stats = affine, track_running_stats
        self.weight = Parameter(ones([num_features], dtype=dtype)) if affine else None
        self.bias = Parameter(zeros([num_features], dtype=dtype)) if affine and bias else None
        self.register_buffer('running_mean', zeros([num_features], dtype=dtype) if track_running_stats else None)
        self.register_buffer('running_var', ones([num_features], dtype=dtype) if track_running_stats else None)
        self.register_buffer('num_batches_tracked', Tensor.from_i64([0], []) if track_running_stats else None)
        self.reset_parameters()

    def reset_running_stats(self):
        from .._native import no_grad
        from .._tensor_api import ones, zeros
        if self.track_running_stats:
            if self.running_mean is None or self.running_var is None or self.num_batches_tracked is None:
                raise ValueError('reset_running_stats requires allocated running buffers')
            with no_grad():
                self.running_mean.copy_(zeros(self.running_mean.shape, dtype=self.running_mean.dtype))
                self.running_var.copy_(ones(self.running_var.shape, dtype=self.running_var.dtype))
                self.num_batches_tracked.copy_(Tensor.from_i64([0], []))

    def reset_parameters(self):
        from .._native import no_grad
        self.reset_running_stats()
        if self.affine:
            with no_grad():
                self.weight.tensor().fill_(1.)
                if self.bias is not None:
                    self.bias.tensor().fill_(0.)

    def forward(self, input):
        from .. import _native
        if not isinstance(input, Tensor):
            raise TypeError('BatchNorm input must be a Tensor')
        if input.ndim not in self._ranks:
            raise ValueError(f'{type(self).__name__} expects input rank in {self._ranks}, got {input.shape}')
        if _native._is_capturing():
            raise NotImplementedError('BatchNorm requires eager execution')
        factor = 0. if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            counter = self.num_batches_tracked
            if counter.device != 'cpu' or counter.dtype != 'int64' or counter.shape != []:
                raise ValueError('num_batches_tracked must be a scalar CPU int64 buffer')
            with _native.no_grad():
                counter.copy_(counter + Tensor.from_i64([1], []))
            if self.momentum is None:
                factor = 1. / float(counter.item())
        use_batch = self.training or (self.running_mean is None and self.running_var is None)
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None
        return F.batch_norm(input, running_mean, running_var, None if self.weight is None else self.weight.tensor(),
                            None if self.bias is None else self.bias.tensor(), use_batch, factor, self.eps)


class BatchNorm1d(_BatchNorm):
    """Batch normalization over channels of [N,C] or [N,C,L] inputs."""
    _ranks = (2, 3)


class BatchNorm2d(_BatchNorm):
    """Batch normalization over channels of [N,C,H,W] inputs."""
    _ranks = (4,)


class BatchNorm3d(_BatchNorm):
    """Batch normalization over channels of [N,C,D,H,W] inputs."""
    _ranks = (5,)
