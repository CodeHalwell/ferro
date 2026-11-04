"""Functional boolean operations over native typed CPU tensors."""
from ._native import Tensor, where as _where
from ._tensor_api import tensor


def _tensor(value):
    if not isinstance(value, Tensor):
        raise TypeError('expected a Ferro Tensor')
    return value


def eq(input, other):
    return _tensor(input).eq(other)


def ne(input, other):
    return _tensor(input).ne(other)


def lt(input, other):
    return _tensor(input).lt(other)


def le(input, other):
    return _tensor(input).le(other)


def gt(input, other):
    return _tensor(input).gt(other)


def ge(input, other):
    return _tensor(input).ge(other)


def logical_and(input, other):
    return _tensor(input).logical_and(other)


def logical_or(input, other):
    return _tensor(input).logical_or(other)


def logical_xor(input, other):
    return _tensor(input).logical_xor(other)


def logical_not(input):
    return _tensor(input).logical_not()


def isnan(input):
    return _tensor(input).isnan()


def isfinite(input):
    return _tensor(input).isfinite()


def _reduce(input, dim, keepdim, name):
    _tensor(input)
    if isinstance(dim, (list, tuple)):
        from ._ops import _dims
        axes = _dims(dim, input.ndim)
        if not axes:
            return input.bool()
        output = input
        for axis in sorted(axes, reverse=True):
            output = getattr(output, name)(axis, keepdim)
        return output
    return getattr(input, name)(dim, keepdim)


def any(input, dim=None, keepdim=False):
    return _reduce(input, dim, keepdim, 'any')


def all(input, dim=None, keepdim=False):
    return _reduce(input, dim, keepdim, 'all')


def nonzero(input, *, as_tuple=False):
    return _tensor(input).nonzero(as_tuple=as_tuple)


def masked_select(input, mask):
    return _tensor(input).masked_select(mask)


def where(condition, input=None, other=None):
    _tensor(condition)
    if input is None and other is None:
        return condition.nonzero(as_tuple=True)
    if input is None or other is None:
        raise TypeError('where requires both input and other')
    template = input if isinstance(input, Tensor) else other if isinstance(other, Tensor) else None
    if template is None:
        input, other = tensor(input), tensor(other)
    else:
        if not isinstance(input, Tensor):
            input = tensor(input, dtype=template.dtype, device=template.device)
        if not isinstance(other, Tensor):
            other = tensor(other, dtype=template.dtype, device=template.device)
    return _where(condition, input, other)
