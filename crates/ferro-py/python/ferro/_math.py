"""Typed native pointwise operations with explicit output-destination limits."""
from ._native import Tensor


class _DefaultDecimals:
    def __repr__(self):
        return '0'


_DEFAULT_DECIMALS = _DefaultDecimals()


def _input(value, out):
    if not isinstance(value, Tensor):
        raise TypeError('input must be a Tensor')
    if out is not None:
        raise NotImplementedError('pointwise out destinations are unsupported')
    return value


def floor(input, *, out=None):
    return _input(input, out).floor()


def ceil(input, *, out=None):
    return _input(input, out).ceil()


def trunc(input, *, out=None):
    return _input(input, out).trunc()


def round(input, *, decimals=_DEFAULT_DECIMALS, out=None):
    input = _input(input, out)
    return input.round() if decimals is _DEFAULT_DECIMALS else input.round(decimals=decimals)


def frac(input, *, out=None):
    return _input(input, out).frac()


def rsqrt(input, *, out=None):
    return _input(input, out).rsqrt()


def sqrt(input, *, out=None):
    return _input(input, out).sqrt()
