"""Typed indexed selections backed by the native autograd engine."""
from ._native import Tensor


def take(input, index, *, out=None):
    if not isinstance(input, Tensor) or not isinstance(index, Tensor):
        raise TypeError('take requires Tensor input and index')
    if out is not None:
        raise NotImplementedError('take out destinations are not supported')
    return input.take(index)
