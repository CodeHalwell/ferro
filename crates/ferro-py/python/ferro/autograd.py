"""Functional native derivatives. Unsupported higher-order VJPs fail explicitly."""
from ._native import Tensor


def grad(output, inputs, *, create_graph=False):
    if not isinstance(output, Tensor):
        raise TypeError('output must be a Tensor')
    return tuple(output.grad_wrt(list(inputs), create_graph=create_graph))


def vjp(output, inputs, seed, *, create_graph=False):
    return tuple(output.vjp_wrt(list(inputs), seed, create_graph=create_graph))
