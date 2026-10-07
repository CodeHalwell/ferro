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


class Conv2d(Module):
    """CPU f32 NCHW convolution with stride/padding/dilation/groups and bias."""
    _builtin_init = True

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        _positive(in_channels, out_channels, groups)
        if in_channels % groups or out_channels % groups:
            raise ValueError('channels must be divisible by groups')
        self._initialize()
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size, self.stride = F._pair(kernel_size), F._pair(stride)
        self.padding, self.dilation = F._pair(padding), F._pair(dilation)
        _positive(*self.kernel_size, *self.stride, *self.dilation)
        fan_in = in_channels // groups * self.kernel_size[0] * self.kernel_size[1]
        self.weight = Parameter(_uniform((out_channels, in_channels // groups) + self.kernel_size, fan_in ** -0.5))
        self.bias = Parameter(_uniform((out_channels,), fan_in ** -0.5)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                        self.stride, self.padding, self.dilation, self.groups)


class _Pool2d(Module):
    _builtin_init = True

    def __init__(self, kernel_size, stride=None, padding=0):
        self._initialize()
        self.kernel_size = F._square(kernel_size, 'kernel_size')
        self.stride = self.kernel_size if stride is None else F._square(stride, 'stride')
        _positive(self.kernel_size, self.stride)
        if F._square(padding, 'padding'):
            raise ValueError('pooling padding is unsupported')


class MaxPool2d(_Pool2d):
    def forward(self, x):
        return x.max_pool2d(self.kernel_size, self.stride)


class AvgPool2d(_Pool2d):
    def forward(self, x):
        return x.avg_pool2d(self.kernel_size, self.stride)


class Dropout(Module):
    """Philox inverted dropout. Each training forward advances a private stream
    offset, so masks differ per step yet replay exactly for a given seed. The
    seed and stream offset are not part of training checkpoints."""
    _builtin_init = True

    def __init__(self, p=0.5, inplace=False, *, seed=None):
        if not isinstance(p, (int, float)) or not 0.0 <= p < 1.0:
            raise ValueError('dropout probability must be in [0, 1)')
        self._initialize()
        self.p = float(p)
        self._seed = int.from_bytes(os.urandom(8), 'little') if seed is None else seed
        self._offset = 0

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        y = x.dropout(self.p, True, self._seed, self._offset)
        self._offset += math.prod(x.shape)
        return y


class Embedding(Module):
    """N(0,1) table looked up by I64 ids of any shape."""
    _builtin_init = True

    def __init__(self, num_embeddings, embedding_dim):
        _positive(num_embeddings, embedding_dim)
        self._initialize()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(Tensor.randn([num_embeddings, embedding_dim]))

    def forward(self, ids):
        return F.embedding(ids, self.weight.tensor())


def _features(normalized_shape):
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if len(shape) != 1:
        raise ValueError('normalization over more than the last dimension is unsupported')
    _positive(*shape)
    return shape[0]


class LayerNorm(Module):
    """Normalizes over the last dimension."""
    _builtin_init = True

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        self._initialize()
        self.normalized_shape, self.eps = _features(normalized_shape), float(eps)
        self.weight = Parameter(Tensor.ones([self.normalized_shape])) if elementwise_affine else None
        self.bias = Parameter(Tensor.zeros([self.normalized_shape])) if elementwise_affine and bias else None

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, None if self.weight is None else self.weight.tensor(),
                            None if self.bias is None else self.bias.tensor(), self.eps)


class RMSNorm(Module):
    """Normalizes over the last dimension; eps=None uses f32 machine epsilon like torch."""
    _builtin_init = True

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        self._initialize()
        self.normalized_shape, self.eps = _features(normalized_shape), eps
        self.weight = Parameter(Tensor.ones([self.normalized_shape])) if elementwise_affine else None

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, None if self.weight is None else self.weight.tensor(), self.eps)


class GroupNorm(Module):
    """[N, C], [N, C, L] or NCHW input."""
    _builtin_init = True

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        _positive(num_groups, num_channels)
        if num_channels % num_groups:
            raise ValueError('num_channels must be divisible by num_groups')
        self._initialize()
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, float(eps)
        self.weight = Parameter(Tensor.ones([num_channels])) if affine else None
        self.bias = Parameter(Tensor.zeros([num_channels])) if affine else None

    def forward(self, x):
        if not 2 <= len(x.shape) <= 4 or x.shape[1] != self.num_channels:
            raise ValueError(f'GroupNorm expects [N, {self.num_channels}, *] input of rank 2-4, got {x.shape}')
        return F.group_norm(x, self.num_groups, None if self.weight is None else self.weight.tensor(),
                            None if self.bias is None else self.bias.tensor(), self.eps)


class _BatchNorm(Module):
    """CPU batch normalization with running statistics (unbiased running var).
    Training needs at least two elements per channel; eval uses the buffers."""
    _builtin_init = True
    _rank = None

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        _positive(num_features)
        self._initialize()
        self.num_features, self.eps, self.momentum = num_features, float(eps), float(momentum)
        self.weight = Parameter(Tensor.ones([num_features])) if affine else None
        self.bias = Parameter(Tensor.zeros([num_features])) if affine else None
        self.register_buffer('running_mean', Tensor.zeros([num_features]))
        self.register_buffer('running_var', Tensor.ones([num_features]))

    def forward(self, x):
        if len(x.shape) != self._rank:
            raise ValueError(f'{type(self).__name__} expects rank-{self._rank} input, got shape {x.shape}')
        return F.batch_norm(x, self.running_mean, self.running_var, None if self.weight is None else self.weight.tensor(),
                            None if self.bias is None else self.bias.tensor(), self.training, self.momentum, self.eps)


class BatchNorm1d(_BatchNorm):
    """[N, C] input only (no [N, C, L])."""
    _rank = 2


class BatchNorm2d(_BatchNorm):
    _rank = 4
