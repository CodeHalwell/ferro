"""Typed rearrangements and gated linear units over native tensor operations."""
from .._native import Tensor
from .._tensor_api import _shape
from .module import Module


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _tensor(input, rank):
    if not isinstance(input, Tensor) or input.ndim < rank:
        raise ValueError('input must be a Tensor of rank at least ' + str(rank))
    if input.device != 'cpu' or input.dtype not in ('float32', 'float64', 'int64', 'bool'):
        raise NotImplementedError('rearrangements support CPU float32/float64/int64/bool')
    _shape(input.shape, input.dtype)


def _copy(input):
    axis = min(range(input.ndim), key=lambda axis: input.shape[axis])
    return input.index_select(axis, list(range(input.shape[axis])))


def pixel_shuffle(input, upscale_factor):
    factor = _positive(upscale_factor, 'upscale_factor')
    _tensor(input, 3)
    shape = input.shape
    channels, height, width = shape[-3:]
    if channels % (factor * factor):
        raise ValueError('channels must be divisible by upscale_factor squared')
    leading = len(shape) - 3
    expanded = shape[:-3] + [channels // (factor * factor), factor, factor, height, width]
    order = list(range(leading)) + [leading, leading + 3, leading + 1, leading + 4, leading + 2]
    return _copy(input.reshape(expanded).permute(*order).reshape(shape[:-3] + [channels // (factor * factor), height * factor, width * factor]))


def pixel_unshuffle(input, downscale_factor):
    factor = _positive(downscale_factor, 'downscale_factor')
    _tensor(input, 3)
    shape = input.shape
    channels, height, width = shape[-3:]
    if height % factor or width % factor:
        raise ValueError('height and width must be divisible by downscale_factor')
    if input.numel() == 0:
        return _copy(input)
    leading = len(shape) - 3
    expanded = shape[:-3] + [channels, height // factor, factor, width // factor, factor]
    order = list(range(leading)) + [leading, leading + 2, leading + 4, leading + 1, leading + 3]
    return _copy(input.reshape(expanded).permute(*order).reshape(shape[:-3] + [channels * factor * factor, height // factor, width // factor]))


def channel_shuffle(input, groups):
    groups = _positive(groups, 'groups')
    _tensor(input, 3)
    batch, channels = input.shape[:2]
    if channels % groups:
        raise ValueError('channels must be divisible by groups')
    return _copy(input.reshape([batch, groups, channels // groups] + input.shape[2:]).transpose(1, 2).reshape(input.shape))


def glu(input, dim=-1):
    _tensor(input, 1)
    if input.dtype not in ('float32', 'float64'):
        raise TypeError('GLU requires a floating Tensor')
    if type(dim) is not int or not -input.ndim <= dim < input.ndim:
        raise ValueError('GLU dimension is out of range')
    dim %= input.ndim
    size = input.shape[dim]
    if size % 2:
        raise ValueError('GLU dimension must have even size')
    return input.narrow(dim, 0, size // 2) * input.narrow(dim, size // 2, size // 2).sigmoid()


class PixelShuffle(Module):
    _builtin_init = True
    upscale_factor: int

    def __init__(self, upscale_factor):
        factor = _positive(upscale_factor, 'upscale_factor')
        self._initialize()
        object.__setattr__(self, 'upscale_factor', factor)

    def forward(self, input):
        return pixel_shuffle(input, self.upscale_factor)


class PixelUnshuffle(Module):
    _builtin_init = True
    downscale_factor: int

    def __init__(self, downscale_factor):
        factor = _positive(downscale_factor, 'downscale_factor')
        self._initialize()
        object.__setattr__(self, 'downscale_factor', factor)

    def forward(self, input):
        return pixel_unshuffle(input, self.downscale_factor)


class ChannelShuffle(Module):
    _builtin_init = True
    groups: int

    def __init__(self, groups):
        groups = _positive(groups, 'groups')
        self._initialize()
        object.__setattr__(self, 'groups', groups)

    def forward(self, input):
        return channel_shuffle(input, self.groups)


class GLU(Module):
    _builtin_init = True
    dim: int

    def __init__(self, dim=-1):
        if type(dim) is not int:
            raise TypeError('dim must be an integer')
        self._initialize()
        object.__setattr__(self, 'dim', dim)

    def forward(self, input):
        return glu(input, self.dim)


class Unflatten(Module):
    _builtin_init = True
    dim: int
    unflattened_size: tuple

    def __init__(self, dim, unflattened_size):
        if type(dim) is not int:
            raise TypeError('dim must be an integer')
        if not isinstance(unflattened_size, (tuple, list)) or not unflattened_size:
            raise ValueError('unflattened_size must be a nonempty size sequence')
        shape = tuple(unflattened_size)
        if any(type(size) is not int or size < -1 for size in shape) or shape.count(-1) > 1:
            raise ValueError('unflattened_size accepts nonnegative sizes and one inferred dimension')
        self._initialize()
        object.__setattr__(self, 'dim', dim)
        object.__setattr__(self, 'unflattened_size', shape)

    def forward(self, input):
        if not isinstance(input, Tensor) or input.ndim < 1:
            raise ValueError('Unflatten requires a Tensor with at least one dimension')
        return input.unflatten(self.dim, self.unflattened_size)


class Softmax2d(Module):
    def forward(self, input):
        if not isinstance(input, Tensor) or input.ndim not in (3, 4):
            raise ValueError('Softmax2d requires rank-3 or rank-4 input')
        return input.softmax(-3)
