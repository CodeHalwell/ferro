"""Native typed cumulative operators."""
from ._native import Tensor
from collections import namedtuple

CummaxResult = namedtuple("CummaxResult", ("values", "indices"))
CumminResult = namedtuple("CumminResult", ("values", "indices"))


def cumsum(input, dim, *, dtype=None, out=None):
    """Cumulative sum with typed output and a differentiable reverse-prefix rule.

    CPU float32 prefixes use float64 accumulation. Bool defaults to int64;
    int64 arithmetic wraps. The original float32 host fallback remains valid.
    """
    if not isinstance(input, Tensor):
        raise TypeError('cumsum input must be a Tensor')
    if out is not None:
        raise NotImplementedError('cumsum out= buffers are unsupported')
    return input.cumsum(dim, dtype=dtype)


def cumprod(input, dim, *, dtype=None, out=None):
    """CPU cumulative product with zero-aware adjoints and typed accumulation.

    Bool defaults to int64; int64 products wrap. float32 prefixes accumulate
    in float64 and round each output. Zero-containing higher derivatives use
    a product-exclusion graph limited to 1048576 input-elements times axis
    width; ordinary derivatives and tangents do not have that work limit.
    """
    if not isinstance(input, Tensor):
        raise TypeError('cumprod input must be a Tensor')
    if out is not None:
        raise NotImplementedError('cumprod out= buffers are unsupported')
    return input.cumprod(dim, dtype=dtype)


def cummax(input, dim, *, out=None):
    """Typed CPU cumulative extrema with latest ties and exact int64 indices.

    Values retain the input dtype. Ordinary and higher floating derivatives
    use the selected indices; forward tangents retain their own floating dtype.
    """
    if not isinstance(input, Tensor):
        raise TypeError('cummax input must be a Tensor')
    if out is not None:
        raise NotImplementedError('cummax out= buffers are unsupported')
    return input.cummax(dim)


def cummin(input, dim, *, out=None):
    """Typed CPU cumulative extrema with latest ties and exact int64 indices.

    Values retain the input dtype. Ordinary and higher floating derivatives
    use the selected indices; forward tangents retain their own floating dtype.
    """
    if not isinstance(input, Tensor):
        raise TypeError('cummin input must be a Tensor')
    if out is not None:
        raise NotImplementedError('cummin out= buffers are unsupported')
    return input.cummin(dim)
