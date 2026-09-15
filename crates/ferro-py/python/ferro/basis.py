"""Fixed-knot CPU f32 B-splines and coefficient-only KAN layers.

Basis appends a basis axis. KAN maps [batch,input] and [output,input,basis]
to [batch,output]. No hidden base activation or adaptive knots. Input paths
are first-order only; coefficient-only higher derivatives use core algebra.
"""
from ._native import BSplineBasis, kan, _kan_parameter
from .nn.module import Module


class KanLayer(Module):
    _builtin_init = True

    def __init__(self, basis, coefficients):
        self._initialize()
        self.basis = basis
        self.coefficients = _kan_parameter(basis, coefficients)

    def forward(self, x):
        return kan(x, self.coefficients.tensor(), self.basis)
