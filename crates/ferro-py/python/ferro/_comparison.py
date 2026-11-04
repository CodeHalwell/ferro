"""CPU comparisons for bool, int64, float32 and float64 native tensors."""
from ._native import Tensor, no_grad, _real_scalar


def _inputs(input, other, matching=False):
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise TypeError('input and other must be Tensors')
    if input.device != 'cpu' or other.device != 'cpu':
        raise NotImplementedError('comparison functions currently support CPU tensors')
    if input.dtype not in ('bool', 'int64', 'float32', 'float64') or other.dtype not in ('bool', 'int64', 'float32', 'float64'):
        raise NotImplementedError('comparison functions require bool, int64, float32 or float64 tensors')
    if matching and input.dtype != other.dtype:
        raise RuntimeError('input and other must have the same dtype')


def _tolerance(value, name):
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.requires_grad:
            raise TypeError(name + ' must be a real scalar or non-differentiable zero-dimensional tensor')
        value = value.item()
    value = _real_scalar(value)
    if not isinstance(value, (int, float)):
        raise TypeError(name + ' must be a real scalar')
    value = float(value)
    if not value >= 0:
        raise RuntimeError(name + ' must be nonnegative and not NaN')
    return value


def equal(input, other):
    """Return whether shapes and values agree, allowing different dtypes."""
    _inputs(input, other)
    if input.shape != other.shape:
        return False
    with no_grad():
        return bool(input.eq(other).all().item())


def isclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return broadcast Boolean comparisons with typed absolute/relative tolerances."""
    rtol, atol = _tolerance(rtol, 'rtol'), _tolerance(atol, 'atol')
    if type(equal_nan) is not bool:
        raise TypeError('equal_nan must be bool')
    _inputs(input, other, matching=True)
    with no_grad():
        result = input.eq(other)
        # Equality validates broadcast metadata; empty outputs read no values.
        if result.numel() == 0:
            return result
        if equal_nan and input.dtype in ('float32', 'float64'):
            result = result.logical_or(input.isnan().logical_and(other.isnan()))
        if rtol == 0 and atol == 0:
            return result
        left, right = input, other
        if input.dtype in ('bool', 'int64'):
            left, right = input.to(dtype='float32'), other.to(dtype='float32')
        actual_error = (left - right).abs()
        allowed_error = right.abs() * rtol + atol
        close = actual_error.isfinite().logical_and(actual_error.le(allowed_error))
        return result.logical_or(close)


def allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Return whether every broadcast element satisfies isclose."""
    return bool(isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan).all().item())
