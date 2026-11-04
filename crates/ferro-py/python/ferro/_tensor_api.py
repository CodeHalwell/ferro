"""Typed construction helpers; computation and autograd remain native."""
import math
import operator
import sys

from ._native import Tensor


_DTYPES = ('float32', 'float64', 'int64', 'bool')


def _dtype(dtype, default='float32'):
    dtype = default if dtype is None else dtype
    if dtype not in _DTYPES:
        raise TypeError('dtype must be float32, float64, int64, or bool')
    return dtype


def _device(device, default='cpu'):
    device = default if device is None else device
    if device == 'cpu':
        return device
    if device == 'cuda':
        return 'cuda:0'
    if isinstance(device, str) and device.startswith('cuda:'):
        index = device[5:]
        if index.isascii() and index.isdecimal() and int(index) < 2**32:
            return 'cuda:' + str(int(index))
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")


def _shape(size, dtype='float32'):
    if isinstance(size, int):
        size = (size,)
    if not isinstance(size, (tuple, list)):
        raise TypeError('size must be an integer or a list/tuple of integers')
    shape, capacity = [], 1
    for value in size:
        if isinstance(value, bool):
            raise TypeError('tensor dimensions must be integers, not bool')
        value = operator.index(value)
        if value < 0:
            raise ValueError('tensor dimensions must be nonnegative')
        capacity *= max(value, 1)
        if capacity > sys.maxsize:
            raise ValueError('shape exceeds addressable layout capacity')
        shape.append(value)
    count = 0 if 0 in shape else capacity
    width = 1 if dtype == 'bool' else 4 if dtype == 'float32' else 8
    if count > sys.maxsize // width:
        raise ValueError('shape exceeds addressable storage capacity')
    return shape, count


def _nested(data):
    flat, active = [], set()

    def visit(value):
        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in active:
                raise ValueError('cyclic tensor data is unsupported')
            active.add(identity)
            child_shape = None
            for child in value:
                shape = visit(child)
                if child_shape is not None and shape != child_shape:
                    raise ValueError('tensor data must be rectangular, not ragged')
                child_shape = shape
            active.remove(identity)
            return (len(value),) + (() if child_shape is None else child_shape)
        if not isinstance(value, (int, float)):
            raise TypeError('tensor data must contain Python integers or floats')
        flat.append(value)
        return ()

    shape = visit(data)
    if any(isinstance(value, float) for value in flat):
        inferred = 'float32'
    elif any(not isinstance(value, bool) for value in flat):
        inferred = 'int64'
    elif flat:
        inferred = 'bool'
    else:
        inferred = 'float32'
    return flat, shape, inferred


def _options(dtype, device, requires_grad):
    if not isinstance(requires_grad, bool):
        raise TypeError('requires_grad must be bool')
    if requires_grad and dtype not in ('float32', 'float64'):
        raise ValueError('autograd supports float32 and float64 tensors only')
    if dtype in ('float64', 'bool') and device != 'cpu':
        raise ValueError(f'{dtype} storage is supported on CPU only')


def _construct(flat, shape, dtype, device, requires_grad=False):
    _options(dtype, device, requires_grad)
    shape, _ = _shape(shape, dtype)
    if dtype == 'bool':
        result = Tensor.from_bool([bool(value) for value in flat], shape)
    elif dtype == 'int64':
        values = [int(value) for value in flat]
        if any(value < -(1 << 63) or value >= 1 << 63 for value in values):
            raise OverflowError('integer value is outside int64 range')
        result = Tensor.from_i64(values, shape)
    elif dtype == 'float64':
        result = Tensor.from_f64([float(value) for value in flat], shape)
    else:
        result = Tensor([float(value) for value in flat], shape)
    if device != 'cpu':
        result = result.to(device)
    return result.requires_grad_(True) if requires_grad else result


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    """Copy Python data or a Tensor into an independent, detached leaf.

    Boolean data defaults to bool, integers to int64, floating/empty to float32.
    Float32 and CPU float64 tensors support native autograd.
    """
    if isinstance(data, Tensor):
        dtype = _dtype(dtype, data.dtype)
        device = _device(device, data.device)
        _options(dtype, device, requires_grad)
        flat, _, _ = _nested(data.tolist())
        return _construct(flat, data.shape, dtype, device, requires_grad)
    flat, shape, inferred = _nested(data)
    dtype, device = _dtype(dtype, inferred), _device(device)
    return _construct(flat, shape, dtype, device, requires_grad)


def as_tensor(data, *, dtype=None, device=None):
    """Reuse a Tensor unchanged when its dtype/device match, including history.

    Floating CPU conversions preserve autograd; integer/bool conversions are
    nondifferentiable. Python sequences are copied into native storage.
    """
    if isinstance(data, Tensor):
        dtype, device = _dtype(dtype, data.dtype), _device(device, data.device)
        if dtype == data.dtype and device == data.device:
            return data
        return data.to(device=device, dtype=dtype)
    return tensor(data, dtype=dtype, device=device)


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False):
    if not isinstance(fill_value, (int, float)):
        raise TypeError('fill_value must be a Python integer or float')
    inferred = 'float32' if isinstance(fill_value, float) else 'bool' if isinstance(fill_value, bool) else 'int64'
    dtype, device = _dtype(dtype, inferred), _device(device)
    _options(dtype, device, requires_grad)
    fill_value = bool(fill_value) if dtype == 'bool' else int(fill_value) if dtype == 'int64' else float(fill_value)
    if dtype == 'int64' and not -(1 << 63) <= fill_value < (1 << 63):
        raise OverflowError('integer value is outside int64 range')
    if dtype == 'float32' and math.isfinite(fill_value) and abs(fill_value) > float.fromhex('0x1.fffffep+127'):
        raise OverflowError('fill_value is outside float32 range')
    shape, count = _shape(size, dtype)
    return _construct([fill_value] * count, shape, dtype, device, requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    if not size:
        raise TypeError('zeros requires a size; use () for a scalar')
    size = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    return full(size, 0, dtype=_dtype(dtype), device=device, requires_grad=requires_grad)


def empty(*size, dtype=None, device=None, requires_grad=False):
    """Allocate independent dense storage; Ferro initializes its bytes to zero.

    Callers must initialize values before depending on them, as for Torch
    empty. CPU supports float32/float64/int64/bool; CUDA factories retain their
    existing supported dtype/device boundaries.
    """
    return zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False):
    if not size:
        raise TypeError('ones requires a size; use () for a scalar')
    size = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    return full(size, 1, dtype=_dtype(dtype), device=device, requires_grad=requires_grad)


def arange(start, end=None, step=1, *, dtype=None, device=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    values = (start, end, step)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise TypeError('arange bounds and step must be Python integers or floats')
    if any(isinstance(value, float) and not math.isfinite(value) for value in values):
        raise ValueError('arange bounds and step must be finite')
    dtype = _dtype(dtype, 'float32' if any(isinstance(value, float) for value in values) else 'int64')
    if dtype == 'bool':
        raise TypeError('arange does not support bool dtype')
    device = _device(device)
    _options(dtype, device, requires_grad)
    if dtype == 'int64':
        start, end, step = (int(value) for value in values)
        if any(value < -(1 << 63) or value >= 1 << 63 for value in (start, end, step)):
            raise OverflowError('arange argument is outside int64 range')
    if step == 0 or (end > start and step < 0) or (end < start and step > 0):
        raise ValueError('arange step must be nonzero and point toward end')
    if all(isinstance(value, int) for value in (start, end, step)):
        count = len(range(start, end, step))
    else:
        count = math.ceil((end - start) / step)
    _shape((count,), dtype)
    return _construct([start + i * step for i in range(count)], [count], dtype, device, requires_grad)


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    dtype, device = _dtype(dtype), _device(device)
    _options(dtype, device, requires_grad)
    shape, count = _shape((n, n if m is None else m), dtype)
    values = [0] * count
    for i in range(min(shape)):
        values[i * shape[1] + i] = 1
    return _construct(values, shape, dtype, device, requires_grad)


def _random(normal, size, dtype, device, requires_grad, generator, seed):
    if not size:
        raise TypeError('random factories require a size; use () for a scalar')
    dtype, device = _dtype(dtype), _device(device)
    _options(dtype, device, requires_grad)
    if dtype not in ('float32', 'float64'):
        raise TypeError('native random factories support float32 and float64')
    if generator is not None and seed is not None:
        raise ValueError('pass generator or seed, not both')
    size = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    shape, _ = _shape(size, dtype)
    factory = Tensor.randn if normal else Tensor.rand
    result = factory(shape, seed=seed, generator=generator, device=device, dtype=dtype)
    return result.requires_grad_(True) if requires_grad else result


def rand(*size, dtype=None, device=None, requires_grad=False, generator=None, seed=None):
    """Native floating-point uniform samples; Generator sequences are Ferro-specific."""
    return _random(False, size, dtype, device, requires_grad, generator, seed)


def randn(*size, dtype=None, device=None, requires_grad=False, generator=None, seed=None):
    """Native floating-point normal samples; Generator sequences are Ferro-specific."""
    return _random(True, size, dtype, device, requires_grad, generator, seed)
