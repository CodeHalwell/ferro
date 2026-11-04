from .. import _native
from .._tensor_api import _dtype, _device, _shape, rand, zeros
from .module import Module, Parameter
from .functional import linear


class Linear(Module):
    _builtin_init = True

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (in_features, out_features)):
            raise ValueError('Linear dimensions must be positive integers')
        dtype, device = _dtype(dtype), _device(device)
        if dtype not in ('float32', 'float64'):
            raise TypeError('Linear parameters require float32 or float64')
        if device != 'cpu':
            raise ValueError('Linear parameter factories currently support CPU storage')
        _shape((out_features, in_features), dtype)
        self._check_mutation_mode()
        self._initialize()
        self.in_features, self.out_features = in_features, out_features
        bound = in_features ** -0.5
        self.weight = Parameter((rand(out_features, in_features, dtype=dtype) * 2 - 1) * bound)
        self.bias = Parameter(((rand(out_features, dtype=dtype) * 2 - 1) * bound
                               if dtype == 'float64' else zeros(out_features))) if bias else None

    @staticmethod
    def _check_mutation_mode():
        if _native._is_capturing() or _native._forward_ad_is_active():
            raise ValueError('Linear initialization/reset during capture or forward AD is unsupported')

    def reset_parameters(self):
        self._check_mutation_mode()
        weight = self.weight.tensor()
        bias = None if self.bias is None else self.bias.tensor()
        dtype = weight.dtype
        if dtype not in ('float32', 'float64') or weight.device != 'cpu':
            raise ValueError('Linear reset requires CPU floating-point parameters')
        if weight.shape != [self.out_features, self.in_features]:
            raise ValueError('Linear weight shape does not match its feature dimensions')
        if bias is not None and (bias.shape != [self.out_features] or bias.dtype != dtype or bias.device != 'cpu'):
            raise ValueError('Linear bias must match weight dtype, device and output features')
        bound = self.in_features ** -0.5
        new_weight = (rand(self.out_features, self.in_features, dtype=dtype) * 2 - 1) * bound
        new_bias = None if bias is None else (rand(self.out_features, dtype=dtype) * 2 - 1) * bound
        with _native.no_grad():
            weight.copy_(new_weight)
            if bias is not None:
                bias.copy_(new_bias)

    def forward(self, x):
        return linear(x, self.weight.tensor(), None if self.bias is None else self.bias.tensor())
