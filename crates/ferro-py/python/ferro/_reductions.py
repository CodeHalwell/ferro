"""Typed eager CPU product reductions."""
from ._native import Tensor, _real_scalar

_MISSING = object()


def prod(input, dim=_MISSING, keepdim=_MISSING, *, dtype=None):
    """Multiply all entries or one dimension; floating accumulation uses its dtype.

    CPU Bool/I64/F32/F64 are supported. Integer products wrap. Floating
    reductions use a deterministic serial order, so SIMD/parallel Torch
    reductions can differ in rounding or extreme overflow/underflow. Ordinary
    floating gradients and existing F32 higher/JVP rules are supported;
    zero-containing higher prefix graphs retain cumprod's bounded-work limit.
    The current reverse-over-reverse Hessian transform can differ at nonfinite
    intermediates from Torch's default forward-over-reverse Hessian order.
    """
    if not isinstance(input, Tensor):
        raise TypeError('prod input must be a Tensor')
    if dtype not in (None, 'bool', 'int64', 'float32', 'float64'):
        raise ValueError('prod dtype must be bool, int64, float32 or float64')
    if dim is _MISSING:
        if keepdim is not _MISSING:
            raise TypeError('prod keepdim requires an explicit dimension')
        return input._prod(None, False, dtype)
    dim = _real_scalar(dim, scalar=True)
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError('prod dim must be an integer, not bool or None')
    if keepdim is _MISSING:
        keepdim = False
    if type(keepdim) is not bool:
        raise TypeError('prod keepdim must be bool')
    return input._prod(dim, keepdim, dtype)
