"""Checked CPU learned activations, bilinear maps and Gaussian likelihoods."""
import math
import operator

from .. import _native
from .._native import Tensor, _real_scalar, no_grad, where
from .._tensor_api import _device, _dtype, _shape, full as _full
from .module import Module, Parameter


def _floating(value, name):
    if not isinstance(value, Tensor):
        raise TypeError(name + ' must be a Tensor')
    if value.device != 'cpu' or value.dtype not in ('float32', 'float64'):
        raise NotImplementedError(name + ' requires CPU float32 or float64 storage')
    return value


def _eager():
    if _native._is_capturing():
        raise NotImplementedError('these functional helpers require eager execution')


def _dimension(value, name, positive=False):
    if isinstance(value, bool):
        raise TypeError(name + ' must be an integer')
    value = operator.index(value)
    if value < int(positive):
        raise ValueError(name + (' must be positive' if positive else ' must be nonnegative'))
    return value


def _parameter_factory(device, dtype):
    if _device(device) != 'cpu' or _dtype(dtype) != 'float32':
        raise NotImplementedError('module parameters currently require CPU float32')


def prelu(input, weight):
    """CPU typed PReLU with native branch-wise adjoints and direct JVP.

    Float32 higher derivatives use a shared two-output backward operation.
    Float64 higher reverse gradients and inference capture remain unsupported.
    Direct JVP preserves absent sibling tangents and the measured >= zero
    boundary; ordinary first-order backward selects the > zero branch.
    """
    return _floating(input, 'input').prelu(_floating(weight, 'weight'))


class PReLU(Module):
    _builtin_init = True
    num_parameters: int
    init: float

    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        num_parameters = _dimension(num_parameters, 'num_parameters')
        _parameter_factory(device, dtype)
        _shape([num_parameters])
        init = _real_scalar(init, scalar=True)
        if not isinstance(init, (int, float)):
            raise TypeError('init must be a real scalar')
        if isinstance(init, int) and not -(1 << 63) <= init < 1 << 64:
            raise OverflowError('init exceeds int64/uint64 range')
        # Factory validation also checks finite float32 overflow on empty shapes.
        _full([0], init, dtype='float32')
        self._initialize()
        object.__setattr__(self, 'num_parameters', num_parameters)
        object.__setattr__(self, 'init', init)
        self.weight = Parameter(Tensor.zeros([num_parameters]))
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            self.weight.tensor().fill_(self.init)

    def forward(self, input):
        return prelu(input, self.weight.tensor())


def bilinear(input1, input2, weight, bias=None):
    """CPU float32/float64 bilinear maps with native tensor adjoints.

    Leading input shapes must match exactly; there is no batch broadcasting.
    Dense intermediates have shape [out_features, batch_size, in2_features].
    """
    _eager()
    values = [_floating(input1, 'input1'), _floating(input2, 'input2'), _floating(weight, 'weight')]
    if bias is not None:
        values.append(_floating(bias, 'bias'))
    if any(value.dtype != input1.dtype for value in values):
        raise ValueError('bilinear operands must have the same dtype')
    if input1.ndim == 0 or input2.ndim != input1.ndim:
        raise ValueError('bilinear inputs need matching positive ranks')
    if weight.ndim != 3:
        raise ValueError('bilinear weight must have shape [out_features, in1_features, in2_features]')
    output_width, first_width, second_width = weight.shape
    if first_width == 0 or second_width == 0:
        raise ValueError('bilinear feature widths must be nonzero')
    if input1.shape[:-1] != input2.shape[:-1] or input1.shape[-1] != first_width or input2.shape[-1] != second_width:
        raise ValueError('bilinear batch and feature dimensions do not match')
    if bias is not None and bias.shape != [output_width]:
        raise ValueError('bilinear bias must have shape [out_features]')
    prefix = input1.shape[:-1]
    batch = math.prod(prefix)
    for value in values:
        _shape(value.shape, value.dtype)
    _shape([output_width, batch, second_width], input1.dtype)
    _shape([*prefix, output_width], input1.dtype)
    first = input1.reshape([batch, first_width])
    second = input2.reshape([1, batch, second_width])
    projected = first.unsqueeze(0).matmul(weight)
    result = (projected * second).sum(dim=2).transpose(0, 1)
    # Empty transpose is contiguous by definition; the native bilinear result
    # still owns canonical output strides, so perform its real copy explicitly.
    result = result.clone(memory_format='contiguous_format') if result.numel() == 0 else result.contiguous()
    result = result.reshape([*prefix, output_width])
    return result if bias is None else result + bias


class Bilinear(Module):
    """CPU float32/float64 bilinear parameters and native tensor adjoints.

    Float64 supports ordinary first-order gradients and typed optimizer state;
    transform and higher-order support follows the underlying tensor APIs.
    """
    _builtin_init = True
    in1_features: int
    in2_features: int
    out_features: int

    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None, dtype=None):
        first = _dimension(in1_features, 'in1_features', True)
        second = _dimension(in2_features, 'in2_features')
        output = _dimension(out_features, 'out_features')
        if type(bias) is not bool:
            raise TypeError('bias must be bool')
        if _native._is_capturing() or _native._forward_ad_is_active():
            raise NotImplementedError('Bilinear initialization during capture or forward AD is unsupported')
        dtype = _dtype(dtype)
        if _device(device) != 'cpu' or dtype not in ('float32', 'float64'):
            raise NotImplementedError('Bilinear parameters require CPU float32 or float64')
        _shape([output, first, second], dtype)
        _shape([output], dtype)
        self._initialize()
        for name, value in (('in1_features', first), ('in2_features', second), ('out_features', output)):
            object.__setattr__(self, name, value)
        weight_shape = [output, first, second]
        self.weight = Parameter(Tensor.zeros(weight_shape) if dtype == 'float32' else _full(weight_shape, 0., dtype=dtype))
        self.bias = Parameter(Tensor.zeros([output]) if dtype == 'float32' else _full([output], 0., dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        if _native._is_capturing() or _native._forward_ad_is_active():
            raise NotImplementedError('Bilinear reset during capture or forward AD is unsupported')
        values = [self.weight.tensor()] + ([] if self.bias is None else [self.bias.tensor()])
        for value in values:
            _floating(value, 'Bilinear parameter')
            _shape(value.shape, value.dtype)
        bound = self.in1_features ** -0.5
        with no_grad():
            for value in values:
                value.copy_((Tensor.rand(value.shape, dtype=value.dtype) * 2 - 1) * bound)

    def forward(self, input1, input2):
        return bilinear(input1, input2, self.weight.tensor(), None if self.bias is None else self.bias.tensor())


def _gaussian_options(full, eps, reduction):
    if type(full) is not bool:
        raise TypeError('full must be bool')
    eps = _real_scalar(eps, scalar=True)
    if not isinstance(eps, (int, float)):
        raise TypeError('eps must be a real scalar')
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError("reduction must be 'none', 'mean' or 'sum'")
    return full, float(eps), reduction


def _broadcast(left, right):
    shape = []
    for a, b in zip([1] * max(0, len(right) - len(left)) + left,
                    [1] * max(0, len(left) - len(right)) + right):
        if a != b and a != 1 and b != 1:
            raise ValueError('input and target shapes are not broadcastable')
        shape.append(b if a == 1 else a)
    return shape


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction='mean'):
    """Gaussian NLL with an identity reverse derivative through variance clamp.

    CPU tensors with one matching float32/float64 dtype and a floating scalar
    variance are supported; mixed tensor dtypes are explicitly unsupported.
    Active forward AD rejects: Torch's no-grad in-place clamp has distinct
    tangent mutation semantics which this reverse composition does not model.
    """
    _eager()
    if _native._forward_ad_is_active():
        raise NotImplementedError('Gaussian variance-clamp forward AD is unsupported')
    _floating(input, 'input')
    _floating(target, 'target')
    full_term, eps, reduction = _gaussian_options(full, eps, reduction)
    if isinstance(var, float):
        if var < 0:
            raise ValueError('var has negative entries')
        var = _full(input.shape, var, dtype=input.dtype)
    _floating(var, 'var')
    if any(value.dtype != input.dtype for value in (target, var)):
        raise ValueError('Gaussian NLL tensors must have the same dtype')
    for value in (input, target, var):
        _shape(value.shape, value.dtype)
    shape = _broadcast(input.shape, target.shape)
    _shape(_broadcast(shape, var.shape if var.shape != input.shape[:-1] else [*var.shape, 1]), input.dtype)
    if var.shape != input.shape:
        if var.shape == input.shape[:-1]:
            var = var.unsqueeze(-1)
        elif var.ndim != input.ndim or sum(b for a, b in zip(input.shape, var.shape) if a != b) != 1:
            raise ValueError('var is of incorrect size')
    if bool((var < 0).any()):
        raise ValueError('var has negative entries')
    bound = _full([], eps, dtype=var.dtype)
    variance = var.clone().contiguous()
    with no_grad():
        clamped = where(variance.ge(bound).logical_or(variance.isnan()), variance, bound)
        variance.detach().copy_(clamped)
    loss = (variance.log() + (input - target).square() / variance) * 0.5
    if full_term:
        loss = loss + (0.5 * math.log(2 * math.pi))
    return loss if reduction == 'none' else loss.sum() if reduction == 'sum' else loss.mean()


class GaussianNLLLoss(Module):
    _builtin_init = True
    full: bool
    eps: float
    reduction: str

    def __init__(self, *, full=False, eps=1e-6, reduction='mean'):
        full, eps, reduction = _gaussian_options(full, eps, reduction)
        self._initialize()
        for name, value in (('full', full), ('eps', eps), ('reduction', reduction)):
            object.__setattr__(self, name, value)

    def forward(self, input, target, var):
        return gaussian_nll_loss(input, target, var, self.full, self.eps, self.reduction)
