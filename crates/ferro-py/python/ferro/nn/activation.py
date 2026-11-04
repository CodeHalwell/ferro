"""Activation modules with explicit rejection of unsupported in-place mutation."""
from .module import Module
from . import functional as F
from .._native import Tensor, _real_scalar


class _InplaceActivation(Module):
    @property
    def inplace(self):
        return False


def _stateless(name, fn, inplace=False):
    if inplace:
        def __init__(self, inplace=False):
            F._activation_inplace(inplace)
            self._initialize()
    else:
        def __init__(self):
            self._initialize()
    return type(name, (_InplaceActivation if inplace else Module,), {'__init__': __init__, '_builtin_init': True, '__module__': __name__,
                                  '__qualname__': name, 'forward': lambda self, input: fn(input)})


ReLU = _stateless('ReLU', F.relu, True)
Sigmoid = _stateless('Sigmoid', F.sigmoid)
Tanh = _stateless('Tanh', F.tanh)
SiLU = _stateless('SiLU', F.silu, True)
Mish = _stateless('Mish', F.mish, True)
SELU = _stateless('SELU', F.selu, True)
ReLU6 = _stateless('ReLU6', F.relu6, True)
Hardsigmoid = _stateless('Hardsigmoid', F.hardsigmoid, True)
Hardswish = _stateless('Hardswish', F.hardswish, True)
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

    def forward(self, input):
        return F.gelu(input, self.approximate)


def _exponential_alpha(value):
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.requires_grad:
            raise TypeError('alpha must be a scalar without gradient history')
        value = value.item()
    value = _real_scalar(value, scalar=True)
    if not isinstance(value, (int, float)):
        raise TypeError('alpha must be a real numeric scalar')
    if isinstance(value, int) and not -(2**63) <= value < 2**64:
        raise OverflowError('alpha exceeds int64/uint64 range')
    # Preserve legacy configuration categories when conversion is exact.
    return float(value) if isinstance(value, int) and float(value) == value else value


class ELU(_InplaceActivation):
    """Typed exponential-linear activation; in-place mutation is unsupported."""
    _builtin_init = True

    def __init__(self, alpha=1.0, inplace=False):
        F._activation_inplace(inplace)
        alpha = _exponential_alpha(alpha)
        self._initialize()
        self.alpha = alpha

    def forward(self, input):
        return input.elu(self.alpha)


class CELU(_InplaceActivation):
    """CELU with Torch's separate forward/backward scalar conversion stages."""
    _builtin_init = True

    def __init__(self, alpha=1.0, inplace=False):
        F._activation_inplace(inplace)
        alpha = _exponential_alpha(alpha)
        self._initialize()
        self.alpha = alpha

    def forward(self, input):
        return input.celu(self.alpha)


class LeakyReLU(_InplaceActivation):
    _builtin_init = True

    def __init__(self, negative_slope=0.01, inplace=False):
        F._activation_inplace(inplace)
        self._initialize()
        self.negative_slope = float(negative_slope)

    def forward(self, input):
        return input.leaky_relu(self.negative_slope)


class Hardtanh(_InplaceActivation):
    _builtin_init = True

    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        F._activation_inplace(inplace)
        self._initialize()
        self.min_val, self.max_val = float(min_val), float(max_val)

    def forward(self, input):
        return input.hardtanh(self.min_val, self.max_val)


class Hardshrink(Module):
    _builtin_init = True

    def __init__(self, lambd=0.5):
        self._initialize()
        self.lambd = float(lambd)

    def forward(self, input):
        return input.hardshrink(self.lambd)


class Softshrink(Module):
    _builtin_init = True

    def __init__(self, lambd=0.5):
        self._initialize()
        self.lambd = float(lambd)

    def forward(self, input):
        return input.softshrink(self.lambd)


class Softplus(Module):
    _builtin_init = True

    def __init__(self, beta=1.0, threshold=20.0):
        options = []
        for value in (beta, threshold):
            if isinstance(value, Tensor):
                if value.ndim != 0 or value.requires_grad:
                    raise TypeError('softplus options must be scalars without gradient history')
                value = value.item()
            value = _real_scalar(value, scalar=True)
            if not isinstance(value, (int, float)):
                raise TypeError('softplus beta and threshold must be real numeric scalars')
            if isinstance(value, int) and not -(2**63) <= value < 2**64:
                raise OverflowError('softplus integer option exceeds int64/uint64 range')
            # Preserve legacy scalar schemas only when conversion is exact.
            if isinstance(value, int) and float(value) == value:
                value = float(value)
            options.append(value)
        self._initialize()
        self.beta, self.threshold = options

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)


class Threshold(_InplaceActivation):
    _builtin_init = True

    def __init__(self, threshold, value, inplace=False):
        F._activation_inplace(inplace)
        self._initialize()
        self.threshold, self.value = float(threshold), float(value)

    def forward(self, input):
        return input.threshold(self.threshold, self.value)


class _DimActivation(Module):
    _builtin_init = True

    def __init__(self, dim):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError('dim must be an integer (implicit dim is unsupported)')
        self._initialize()
        self.dim = dim


class Softmax(_DimActivation):
    def forward(self, input):
        return input.softmax(self.dim)


class LogSoftmax(_DimActivation):
    def forward(self, input):
        return input.log_softmax(self.dim)


class Softmin(_DimActivation):
    def forward(self, input):
        return input.softmin(self.dim)
