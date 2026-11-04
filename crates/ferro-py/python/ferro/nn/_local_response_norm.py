"""CPU cross-channel local response normalization over native tensor operations."""
import sys
from numbers import Real

from .. import _native
from .._native import Tensor
from .._ops import cat
from .._tensor_api import _shape
from .module import Module


def _options(size, alpha, beta, k):
    if type(size) is not int or not 0 < size <= sys.maxsize:
        raise ValueError('size must be a positive addressable integer')
    if any(not isinstance(value, Real) for value in (alpha, beta, k)):
        raise TypeError('alpha, beta and k must be real numbers')
    return size, float(alpha), float(beta), float(k)


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0):
    """Normalize with channel windows; even sizes include the extra left entry.

    CPU float32/float64 inputs and native first-order gradients are supported.
    The window size must be a positive addressable integer, including empties.
    Empty inputs are returned unchanged. Window sums use native reductions
    without materializing padding proportional to the configured window size.
    Strided higher-rank inputs extend Torch's contiguous-view restriction.
    """
    size, alpha, beta, k = _options(size, alpha, beta, k)
    if _native._is_capturing():
        raise NotImplementedError('local_response_norm requires eager execution')
    if not isinstance(input, Tensor) or input.device != 'cpu' or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('local_response_norm supports CPU float32/float64 tensors')
    if input.ndim < 3:
        raise ValueError('local_response_norm expects [N,C,*spatial] with rank at least three')
    if not input.numel():
        return input
    _shape(input.shape, input.dtype)
    channels = input.shape[1]
    _shape([channels], 'int64')
    squared = input * input
    windows = []
    for channel in range(channels):
        start = max(0, channel - size // 2)
        stop = min(channels, channel + (size - 1) // 2 + 1)
        windows.append(squared.narrow(1, start, stop - start).sum_dim(1, True))
    denominator = (cat(windows, 1) / size * alpha + k).pow(beta)
    return input / denominator


class LocalResponseNorm(Module):
    _builtin_init = True
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1.0):
        size, alpha, beta, k = _options(size, alpha, beta, k)
        self._initialize()
        for name, value in (('size', size), ('alpha', alpha), ('beta', beta), ('k', k)):
            object.__setattr__(self, name, value)

    def forward(self, input):
        return local_response_norm(input, self.size, self.alpha, self.beta, self.k)
