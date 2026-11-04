"""Functional p-normalization through the existing native norm and AD engine.

CPU float32/float64 support real non-NaN p and int/tuple/list/None axes.
The original float32 single-axis p=2 path remains available on its device.
These native reductions currently provide first-order eager derivatives only.
Stable norm accumulation may choose a different clamp branch than Torch when
the norm differs by one rounding unit exactly at eps.
"""
import math
from numbers import Real

from .. import _native
from .._native import Tensor
from .._tensor_api import _shape, full
from ..linalg import vector_norm


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    if out is not None:
        raise NotImplementedError('normalize out= is unsupported')
    if not isinstance(input, Tensor):
        raise TypeError('normalize input must be a Tensor')
    if input.dtype not in ('float32', 'float64'):
        raise ValueError('normalize input must have float32 or float64 dtype')
    if not isinstance(p, Real) or math.isnan(p):
        raise ValueError('normalize p must be a real number other than NaN')
    if not isinstance(eps, Real):
        raise TypeError('normalize eps must be a real number')
    p, eps = float(p), float(eps)
    _shape(input.shape, input.dtype)
    if input.dtype == 'float32' and math.isfinite(eps) and abs(eps) > 3.4028234663852886e38:
        raise OverflowError('normalize eps cannot be represented in float32')
    if _native._is_capturing():
        raise NotImplementedError('normalize requires eager execution')
    if input.dtype == 'float32' and p == 2. and type(dim) is int and input.ndim and math.isfinite(eps) and eps > 0:
        return input.normalize(dim, eps)
    if input.device != 'cpu':
        raise NotImplementedError('general p-normalization currently requires CPU inputs')
    norm = vector_norm(input, p, dim, True)
    lower = full((), eps, dtype=input.dtype)
    keep = norm.isnan().logical_or(norm.ge(lower))
    denominator = _native.where(keep, norm, lower)
    if math.isnan(eps):
        denominator = _native.where(norm.isnan(), norm, lower)
    return input / denominator
