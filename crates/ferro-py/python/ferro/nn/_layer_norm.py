"""Checked trailing-shape normalization through existing native operations."""
from numbers import Real

from .. import _native
from .._native import Tensor
from .._ops import mean
from .._tensor_api import _device, _dtype, _shape


def normalized_shape(value, dtype='float32'):
    shape, count = _shape(value, dtype)
    if not shape:
        raise ValueError('normalized_shape must contain at least one dimension')
    return tuple(shape), count


def epsilon(value, dtype, optional=False):
    if value is None and optional:
        return 1.1920928955078125e-7 if dtype == 'float32' else 2.220446049250313e-16
    if not isinstance(value, Real):
        raise TypeError('normalization eps must be a real number')
    return float(value)


def factory(shape, elementwise_affine, device, dtype, bias=True, *, affine_float64=False):
    if type(elementwise_affine) is not bool or type(bias) is not bool:
        raise TypeError('elementwise_affine and bias must be bool')
    dtype, device = _dtype(dtype), _device(device)
    if device != 'cpu' or dtype not in ('float32', 'float64'):
        raise NotImplementedError('normalization factories support CPU float32/float64 storage')
    if elementwise_affine and dtype != 'float32' and not affine_float64:
        raise NotImplementedError('normalization parameters require float32; use elementwise_affine=False for float64')
    return normalized_shape(shape, dtype)[0]


def validate(input, shape, weight, bias, eps, rms=False):
    if not isinstance(input, Tensor) or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('normalization supports float32/float64 inputs')
    shape, width = normalized_shape(shape, input.dtype)
    if input.ndim < len(shape) or input.shape[-len(shape):] != list(shape):
        raise ValueError('normalized_shape must match the trailing input dimensions')
    _shape(input.shape, input.dtype)
    if input.device != 'cpu' and (input.dtype != 'float32' or len(shape) != 1):
        raise NotImplementedError('multi-axis and float64 normalization currently require CPU')
    if input.dtype == 'float64' and _native._is_capturing():
        raise NotImplementedError('float64 normalization requires eager execution')
    for value, name in ((weight, 'weight'), (bias, 'bias')):
        if value is not None and (not isinstance(value, Tensor) or value.shape != list(shape)
                                  or value.dtype != input.dtype or value.device != input.device):
            raise ValueError(name + ' must match normalized_shape, input dtype and device')
    return shape, width, epsilon(eps, input.dtype, rms)


def apply_affine(value, weight, bias):
    if weight is not None:
        value = value * weight
    return value if bias is None else value + bias


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Normalize trailing dimensions; CPU float64 uses native primitive AD."""
    shape, width, eps = validate(input, normalized_shape, weight, bias, eps)
    if input.numel() == 0:
        return apply_affine(input.clone(), weight, bias)
    if input.dtype == 'float32':
        flat_shape = input.shape[:-len(shape)] + [width]
        flat = input if len(shape) == 1 else input.reshape(flat_shape)
        flattened_weight = weight if weight is None or len(shape) == 1 else weight.reshape([width])
        flattened_bias = bias if bias is None or len(shape) == 1 else bias.reshape([width])
        output = flat.layer_norm(flattened_weight, flattened_bias, eps)
        return output if len(shape) == 1 else output.reshape(input.shape)
    axes = tuple(range(input.ndim - len(shape), input.ndim))
    centered = input - mean(input, axes, True)
    normalized = centered / (mean(centered.square(), axes, True) + eps).sqrt()
    return apply_affine(normalized, weight, bias)


def rms_norm(input, normalized_shape, weight=None, eps=None):
    """Normalize trailing root-mean-square using input-dtype machine epsilon."""
    shape, width, eps = validate(input, normalized_shape, weight, None, eps, True)
    if input.numel() == 0:
        return apply_affine(input.clone(), weight, None)
    if input.dtype == 'float32':
        flat_shape = input.shape[:-len(shape)] + [width]
        flat = input if len(shape) == 1 else input.reshape(flat_shape)
        flattened_weight = weight if weight is None or len(shape) == 1 else weight.reshape([width])
        output = flat.rms_norm(flattened_weight, eps)
        return output if len(shape) == 1 else output.reshape(input.shape)
    axes = tuple(range(input.ndim - len(shape), input.ndim))
    normalized = input / (mean(input.square(), axes, True) + eps).sqrt()
    return apply_affine(normalized, weight, None)

