"""Activation modules. `inplace` is accepted for torch source compatibility;
results are always out-of-place."""
from .module import Module
from . import functional as F


def _stateless(name, fn):
    def __init__(self, inplace=False):
        self._initialize()
    return type(name, (Module,), {'__init__': __init__, '_builtin_init': True, '__module__': __name__,
                                  '__qualname__': name, 'forward': lambda self, x: fn(x)})


ReLU = _stateless('ReLU', F.relu)
Sigmoid = _stateless('Sigmoid', F.sigmoid)
Tanh = _stateless('Tanh', F.tanh)
SiLU = _stateless('SiLU', F.silu)
Mish = _stateless('Mish', F.mish)
SELU = _stateless('SELU', F.selu)
ReLU6 = _stateless('ReLU6', F.relu6)
Hardsigmoid = _stateless('Hardsigmoid', F.hardsigmoid)
Hardswish = _stateless('Hardswish', F.hardswish)
Tanhshrink = _stateless('Tanhshrink', F.tanhshrink)
Softsign = _stateless('Softsign', F.softsign)
LogSigmoid = _stateless('LogSigmoid', F.logsigmoid)


class GELU(Module):
    _builtin_init = True

    def __init__(self, approximate='none'):
        if approximate not in ('none', 'tanh'):
            raise ValueError("approximate must be 'none' or 'tanh'")
        self._initialize()
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(x, self.approximate)


class ELU(Module):
    _builtin_init = True

    def __init__(self, alpha=1.0, inplace=False):
        self._initialize()
        self.alpha = float(alpha)

    def forward(self, x):
        return x.elu(self.alpha)


class CELU(Module):
    _builtin_init = True

    def __init__(self, alpha=1.0, inplace=False):
        self._initialize()
        self.alpha = float(alpha)

    def forward(self, x):
        return x.celu(self.alpha)


class LeakyReLU(Module):
    _builtin_init = True

    def __init__(self, negative_slope=0.01, inplace=False):
        self._initialize()
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        return x.leaky_relu(self.negative_slope)


class Hardtanh(Module):
    _builtin_init = True

    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        self._initialize()
        self.min_val, self.max_val = float(min_val), float(max_val)

    def forward(self, x):
        return x.hardtanh(self.min_val, self.max_val)


class Hardshrink(Module):
    _builtin_init = True

    def __init__(self, lambd=0.5):
        self._initialize()
        self.lambd = float(lambd)

    def forward(self, x):
        return x.hardshrink(self.lambd)


class Softshrink(Module):
    _builtin_init = True

    def __init__(self, lambd=0.5):
        self._initialize()
        self.lambd = float(lambd)

    def forward(self, x):
        return x.softshrink(self.lambd)


class Softplus(Module):
    _builtin_init = True

    def __init__(self, beta=1.0, threshold=20.0):
        if threshold != 20.0:
            raise ValueError('softplus threshold other than 20 is unsupported')
        self._initialize()
        self.beta, self.threshold = float(beta), float(threshold)

    def forward(self, x):
        return F.softplus(x, self.beta, self.threshold)


class Threshold(Module):
    _builtin_init = True

    def __init__(self, threshold, value, inplace=False):
        self._initialize()
        self.threshold, self.value = float(threshold), float(value)

    def forward(self, x):
        return x.threshold(self.threshold, self.value)


class _DimActivation(Module):
    _builtin_init = True

    def __init__(self, dim):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError('dim must be an integer (implicit dim is unsupported)')
        self._initialize()
        self.dim = dim


class Softmax(_DimActivation):
    def forward(self, x):
        return x.softmax(self.dim)


class LogSoftmax(_DimActivation):
    def forward(self, x):
        return x.log_softmax(self.dim)


class Softmin(_DimActivation):
    def forward(self, x):
        return x.softmin(self.dim)
