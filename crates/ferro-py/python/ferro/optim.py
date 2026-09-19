"""Native optimizers over stable parameter slots. Parameter groups are not supported."""
from ._native import _Optimizer
from .nn import Parameter


class _Base:
    def _initialize(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.0, momentum=0.0, nesterov=False):
        self.params = tuple(dict.fromkeys(params))
        if not all(isinstance(p, Parameter) for p in self.params):
            raise TypeError('Optimizers require Parameter objects; groups are unsupported')
        self._native = _Optimizer(type(self).__name__, list(self.params), lr,
                                  *betas, eps, weight_decay, momentum, nesterov)

    def zero_grad(self):
        self._native.zero_grad()

    def step(self):
        self._native.step()


class SGD(_Base):
    def __init__(self, params, lr, momentum=0.0, nesterov=False):
        self._initialize(params, lr, momentum=momentum, nesterov=nesterov)


class Adam(_Base):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self._initialize(params, lr, betas, eps)


class AdamW(_Base):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self._initialize(params, lr, betas, eps, weight_decay)
