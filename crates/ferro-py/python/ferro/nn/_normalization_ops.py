"""Eager CPU batch normalization with native tensor gradients and state updates."""
from numbers import Real

from .. import _native
from .._native import Tensor, no_grad
from .._ops import mean
from .._tensor_api import _shape


def _channel_tensor(value, channels, dtype, name, running=False):
    if value is None:
        return
    if not isinstance(value, Tensor) or value.device != 'cpu' or value.dtype != dtype or value.shape != [channels]:
        raise ValueError(name + ' must have the input dtype on CPU with shape [channels]')
    if running and value.requires_grad:
        raise RuntimeError('batch_norm is not differentiable with respect to ' + name)


def _affine(value, weight, bias):
    shape = [1, value.shape[1]] + [1] * (value.ndim - 2)
    if weight is not None:
        value = value * weight.reshape(shape)
    return value if bias is None else value + bias.reshape(shape)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=.1, eps=1e-5):
    """Running-stat writes require independent whole-contiguous CPU buffers."""
    if _native._is_capturing():
        raise NotImplementedError('batch_norm requires eager execution')
    if not isinstance(input, Tensor) or input.device != 'cpu' or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('batch_norm supports CPU float32/float64 tensors')
    if input.ndim < 2 or input.shape[1] == 0:
        raise ValueError('batch_norm expects rank at least two and positive channels')
    if type(training) is not bool:
        raise TypeError('training must be bool')
    if not isinstance(momentum, Real) or not isinstance(eps, Real):
        raise TypeError('batch_norm momentum and eps must be real numbers')
    momentum, eps = float(momentum), float(eps)
    if training and eps <= 0:
        raise ValueError('batch_norm eps must be positive during training')
    channels = input.shape[1]
    _shape(input.shape, input.dtype)
    _shape([channels], input.dtype)
    for value, name, running in ((weight, 'weight', False), (bias, 'bias', False),
                                 (running_mean, 'running_mean', True), (running_var, 'running_var', True)):
        _channel_tensor(value, channels, input.dtype, name, running)
    if not training and (running_mean is None or running_var is None):
        raise ValueError('running_mean and running_var are required in evaluation mode')
    count = input.numel() // channels
    if training and count == 1:
        raise ValueError('batch_norm training needs more than one value per channel')
    if count == 0:
        return _affine(input.clone(), weight, bias)
    if training:
        if running_mean is not None and running_var is not None and running_mean._shares_storage(running_var):
            raise NotImplementedError('training running statistics must use independent storage')
        with no_grad():
            for value in (running_mean, running_var):
                if value is not None:
                    value.copy_(value)
    if input.dtype == 'float32' and not (0 < eps < 1e-30):
        w = Tensor.ones([channels]) if weight is None else weight
        b = Tensor.zeros([channels]) if bias is None else bias
        rm = Tensor.zeros([channels]) if running_mean is None else running_mean
        rv = Tensor.ones([channels]) if running_var is None else running_var
        order = [0] + list(range(2, input.ndim)) + [1]
        flat = input if input.ndim in (2, 4) else input.permute(*order).reshape([count, channels])
        output, updated_mean, updated_var = flat.batch_norm(w, b, rm, rv, training, momentum, eps)
        if flat is not input:
            shape = [input.shape[0]] + input.shape[2:] + [channels]
            inverse = [order.index(axis) for axis in range(input.ndim)]
            output = output.reshape(shape).permute(*inverse)
    else:
        working = input.to(dtype='float64') if input.dtype == 'float32' else input
        working_weight = None if weight is None else weight.to(dtype=working.dtype)
        working_bias = None if bias is None else bias.to(dtype=working.dtype)
        working_mean = None if running_mean is None else running_mean.to(dtype=working.dtype)
        working_var = None if running_var is None else running_var.to(dtype=working.dtype)
        shape = [1, channels] + [1] * (input.ndim - 2)
        if training:
            axes = tuple(axis for axis in range(input.ndim) if axis != 1)
            average = mean(working, axes, True)
            centered = working - average
            variance = mean(centered.square(), axes, True)
            output = _affine(centered / (variance + eps).sqrt(), working_weight, working_bias)
            with no_grad():
                updated_mean = None if working_mean is None else working_mean * (1 - momentum) + average.reshape([channels]) * momentum
                updated_var = None if working_var is None else working_var * (1 - momentum) + variance.reshape([channels]) * (momentum * count / (count - 1))
        else:
            output = _affine((working - working_mean.reshape(shape)) / (working_var.reshape(shape) + eps).sqrt(), working_weight, working_bias)
            updated_mean, updated_var = running_mean, running_var
        output = output.to(dtype=input.dtype)
    if training:
        with no_grad():
            if running_mean is not None:
                running_mean.copy_(updated_mean.to(dtype=running_mean.dtype))
            if running_var is not None:
                running_var.copy_(updated_var.to(dtype=running_var.dtype))
    return output
