"""Native CPU typed sorting; ties are deterministic when stable=True."""
from collections import namedtuple
from ._native import Tensor

_SortResult = namedtuple('sort', ('values', 'indices'))


def sort(input, dim=-1, descending=False, *, stable=False, out=None):
    """Sort F32/F64/I64/Bool values on CPU with exact I64 indices.

    Floating inputs have first-order gradients. Capture, forward AD and
    differentiable higher-order queries currently reject. Stable equal zeros
    and NaNs retain original order; unstable tied adjoints are unspecified.
    """
    if not isinstance(input, Tensor):
        raise TypeError('sort input must be a Tensor')
    if out is not None:
        raise NotImplementedError('sort out destinations are unsupported')
    return input.sort(dim, descending, stable=stable)


def argsort(input, dim=-1, descending=False, *, stable=False):
    if not isinstance(input, Tensor):
        raise TypeError('argsort input must be a Tensor')
    return input.argsort(dim, descending, stable=stable)
