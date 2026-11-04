"""Fixed-knot CPU f32 B-splines and coefficient-only KAN layers.

Basis appends a basis axis. KAN maps [batch,input] and [output,input,basis]
to [batch,output]. No hidden base activation or adaptive knots. Input paths
are first-order only; coefficient-only higher derivatives use core algebra.
"""
from ._native import BSplineBasis, kan, _kan_parameter
from .nn.module import Module


class KanLayer(Module):
    """Trainable coefficients with immutable knots, degree and outside policy.

    Training checkpoints validate the complete basis configuration. ``basis``
    remains readable; construct a new layer to use a different basis.
    """
    _builtin_init = True
    knots: tuple
    degree: int
    outside: str

    def __init__(self, basis, coefficients):
        self._initialize()
        self.coefficients = _kan_parameter(basis, coefficients)
        self._basis = basis
        object.__setattr__(self, 'knots', tuple(basis.knots))
        object.__setattr__(self, 'degree', basis.degree)
        object.__setattr__(self, 'outside', basis.outside)

    @property
    def basis(self):
        return self._basis

    def forward(self, x):
        return kan(x, self.coefficients.tensor(), self.basis)
