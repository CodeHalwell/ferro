from .._native import Tensor
from .module import Module, Parameter
from .functional import linear


class Linear(Module):
    _builtin_init = True

    def __init__(self, in_features, out_features, bias=True):
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (in_features, out_features)):
            raise ValueError('Linear dimensions must be positive integers')
        self._initialize()
        self.in_features, self.out_features = in_features, out_features
        bound = in_features ** -0.5
        self.weight = Parameter((Tensor.rand([out_features, in_features]) * 2 - 1) * bound)
        self.bias = Parameter(Tensor.zeros([out_features])) if bias else None

    def forward(self, x):
        return linear(x, self.weight.tensor(), None if self.bias is None else self.bias.tensor())
