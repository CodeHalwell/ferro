"""Functional shape operations and typed reductions/matrix products.

These compose existing native operations. Operator dtype and device limits
remain explicit; out= destinations are unsupported.
"""
import math
from ._native import Tensor, cat as _cat
from . import _tensor_api as factories


def _tensor(value):
    if not isinstance(value, Tensor):
        raise TypeError('expected a Ferro Tensor')
    return value


def _float(value, dtype=None):
    _tensor(value)
    if value.dtype not in ('float32', 'float64') or dtype not in (None, value.dtype):
        raise NotImplementedError('floating arithmetic requires float32/float64 without implicit dtype conversion')
    return value


def _dims(dim, rank):
    values = (dim,) if type(dim) is int else tuple(dim)
    result = []
    for d in values:
        if type(d) is not int or not -max(rank, 1) <= d < max(rank, 1):
            raise ValueError('dimension out of range')
        d %= max(rank, 1)
        if d in result:
            raise ValueError('duplicate reduction dimension')
        result.append(d)
    return result


def is_tensor(value):
    return isinstance(value, Tensor)


def numel(input):
    return _tensor(input).numel()


def cat(tensors, dim=0):
    return _cat(list(tensors), dim)


concat = cat


def diagonal(input, offset=0, dim1=0, dim2=1):
    return _tensor(input).diagonal(offset=offset, dim1=dim1, dim2=dim2)


def trace(input):
    return _tensor(input).trace()


def reshape(input, shape):
    _tensor(input)
    shape = list(shape)
    if any(type(n) is not int or n < -1 for n in shape) or shape.count(-1) > 1:
        raise ValueError('shape must contain nonnegative integers and at most one inferred dimension')
    if -1 in shape:
        known = math.prod(n for n in shape if n != -1)
        if known == 0 or input.numel() % known:
            raise ValueError('cannot infer requested shape')
        shape[shape.index(-1)] = input.numel() // known
    return input.reshape(shape)


def flatten(input, start_dim=0, end_dim=-1):
    return _tensor(input).flatten(start_dim, end_dim)


def transpose(input, dim0, dim1):
    _tensor(input)
    if input.ndim == 0:
        _dims((dim0,), 0)
        _dims((dim1,), 0)
        return input.reshape([])
    return input.transpose(dim0, dim1)


def permute(input, dims):
    return _tensor(input).permute(*dims)


def squeeze(input, dim=None):
    _tensor(input)
    axes = list(range(input.ndim)) if dim is None else _dims(dim, input.ndim)
    if input.ndim == 0:
        return input.reshape([])
    output = input
    for axis in sorted(axes, reverse=True):
        if input.shape[axis] == 1:
            output = output.squeeze(axis)
    return output


def unsqueeze(input, dim):
    return _tensor(input).unsqueeze(dim)


def split(input, split_size_or_sections, dim=0):
    _tensor(input)
    if type(split_size_or_sections) is int and split_size_or_sections == 0:
        if input.ndim == 0:
            raise ValueError('cannot split a scalar')
        axis, = _dims(dim, input.ndim)
        if input.shape[axis] == 0:
            return (input.narrow(axis, 0, 0),)
    return tuple(input.split(split_size_or_sections, dim))


def chunk(input, chunks, dim=0):
    _tensor(input)
    if type(chunks) is not int or chunks <= 0:
        raise ValueError('chunks must be a positive integer')
    if input.ndim == 0:
        raise ValueError('cannot chunk a scalar')
    axis, = _dims(dim, input.ndim)
    if input.shape[axis] == 0:
        return tuple(input.split([0] * chunks, axis))
    return tuple(input.chunk(chunks, axis))


def unbind(input, dim=0):
    _tensor(input)
    if input.ndim == 0:
        raise ValueError('cannot unbind a scalar')
    axis, = _dims(dim, input.ndim)
    return tuple(input.narrow(axis, i, 1).squeeze(axis) for i in range(input.shape[axis]))


def _reduce(input, dim, keepdim, dtype, mean):
    if dtype is not None:
        _tensor(input)
        if dtype not in ('float32', 'float64', 'int64', 'bool'):
            raise NotImplementedError('reduction dtype is unsupported')
        if mean and dtype not in ('float32', 'float64'):
            raise ValueError('mean requires a floating-point output dtype')
        converted = input.to(dtype=dtype)
        reduced = _reduce(converted, dim, keepdim, None, mean)
        return reduced.to(dtype='bool') if dtype == 'bool' else reduced
    if isinstance(input, Tensor) and input.dtype == 'int64' and not mean:
        if dtype not in (None, 'int64'):
            raise NotImplementedError('integer sum supports int64 output without conversion')
        if type(keepdim) is not bool:
            raise TypeError('keepdim must be bool')
        axes = list(range(input.ndim)) if dim is None else _dims(dim, input.ndim)
        if not axes or input.ndim == 0:
            output = input.sum()
            return output.reshape([1] * input.ndim) if keepdim else output
        output = input
        for axis in sorted(axes, reverse=True):
            output = output.sum_dim(axis, keepdim)
        return output
    if isinstance(input, Tensor) and input.dtype == 'bool' and not mean:
        if dtype not in (None, 'int64'):
            raise NotImplementedError('boolean sum supports int64 output only')
        if type(keepdim) is not bool:
            raise TypeError('keepdim must be bool')
        axes = list(range(input.ndim)) if dim is None else _dims(dim, input.ndim)
        if not axes or input.ndim == 0:
            output = input.sum()
            return output.reshape([1] * input.ndim) if keepdim else output
        remaining = [axis for axis in range(input.ndim) if axis not in axes]
        packed = input.permute(*(remaining + axes)).reshape([input.shape[axis] for axis in remaining] + [math.prod(input.shape[axis] for axis in axes)])
        output = packed.sum_dim(-1, False)
        if keepdim:
            output = output.reshape([1 if axis in axes else input.shape[axis] for axis in range(input.ndim)])
        return output
    _float(input, dtype)
    if type(keepdim) is not bool:
        raise TypeError('keepdim must be bool')
    axes = list(range(input.ndim)) if dim is None else _dims(dim, input.ndim)
    if not axes or input.ndim == 0:
        output = input.mean() if mean else input.sum()
        return output.reshape([1] * input.ndim) if keepdim else output
    output = input
    for axis in sorted(axes, reverse=True):
        output = output.mean_dim(axis, keepdim) if mean else output.sum_dim(axis, keepdim)
    return output


def sum(input, dim=None, keepdim=False, *, dtype=None):
    return _reduce(input, dim, keepdim, dtype, False)


def mean(input, dim=None, keepdim=False, *, dtype=None):
    return _reduce(input, dim, keepdim, dtype, True)


def mm(input, mat2):
    _float(input)
    _float(mat2)
    if input.ndim != 2 or mat2.ndim != 2:
        raise ValueError('mm expects two matrices')
    return input.matmul(mat2)


def bmm(input, mat2):
    _float(input)
    _float(mat2)
    if input.ndim != 3 or mat2.ndim != 3:
        raise ValueError('bmm expects two rank-3 tensors')
    return input.bmm(mat2)


def matmul(input, other):
    _float(input)
    _float(other)
    if input.ndim == 0 or other.ndim == 0:
        raise ValueError('matmul operands must have at least one dimension')
    if input.device != other.device:
        raise ValueError('matmul operands must share a device')
    left_vector, right_vector = input.ndim == 1, other.ndim == 1
    left = input.unsqueeze(0) if left_vector else input
    right = other.unsqueeze(-1) if right_vector else other
    if left.shape[-1] != right.shape[-2]:
        raise ValueError('matmul inner dimensions do not match')
    rank = max(left.ndim, right.ndim) - 2
    a = [1] * (rank - left.ndim + 2) + left.shape[:-2]
    b = [1] * (rank - right.ndim + 2) + right.shape[:-2]
    batch = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError('matmul batch dimensions are not broadcastable')
        batch.append(y if x == 1 else x)
    m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
    if batch:
        left = left.expand(*(batch + [m, k])).reshape([math.prod(batch), m, k])
        right = right.expand(*(batch + [k, n])).reshape([math.prod(batch), k, n])
        output = left.bmm(right).reshape(batch + [m, n])
    else:
        output = left.matmul(right)
    if left_vector:
        output = output.squeeze(-2)
    if right_vector:
        output = output.squeeze(-1)
    return output


def full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False):
    _tensor(input)
    return factories.full(input.shape, fill_value, dtype=input.dtype if dtype is None else dtype,
                          device=input.device if device is None else device, requires_grad=requires_grad)


def zeros_like(input, *, dtype=None, device=None, requires_grad=False):
    return full_like(input, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def ones_like(input, *, dtype=None, device=None, requires_grad=False):
    return full_like(input, 1, dtype=dtype, device=device, requires_grad=requires_grad)


def rand_like(input, *, dtype=None, device=None, requires_grad=False):
    _tensor(input)
    return factories.rand(input.shape, dtype=input.dtype if dtype is None else dtype,
                          device=input.device if device is None else device, requires_grad=requires_grad)


def randn_like(input, *, dtype=None, device=None, requires_grad=False):
    _tensor(input)
    return factories.randn(input.shape, dtype=input.dtype if dtype is None else dtype,
                           device=input.device if device is None else device, requires_grad=requires_grad)
