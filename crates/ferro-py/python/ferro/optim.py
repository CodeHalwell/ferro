"""Optimizers update native parameter slots, never Python value copies."""
from ._native import _SGD
from .nn import Parameter


class SGD:
    def __init__(self, params, lr, momentum=0.0, nesterov=False):
        self.params = tuple(dict.fromkeys(params))
        if not all(isinstance(p, Parameter) for p in self.params):
            raise TypeError('SGD requires Parameter objects')
        self._native = _SGD(list(self.params), lr, momentum, nesterov)

    def zero_grad(self):
        self._native.zero_grad()

    def step(self):
        self._native.step()
