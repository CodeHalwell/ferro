"""Checked CPU group normalization using existing native gradients."""
from .. import _native
from .._native import Tensor
from .._ops import mean
from .._tensor_api import _shape, ones, zeros
from ._layer_norm import epsilon


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """Normalize channels/spatial axes within each sample's channel groups.

    CPU float32/float64 first-order gradients are supported. Empty spatial
    inputs retain Torch's NaN affine-weight gradient through the actual
    scale/standard-deviation algebra; empty batches produce zero gradients.
    Bias-only gradients use the ordinary affine derivative, including the
    case where Torch2.13 CPU's direct backward raises without a weight tensor.
    """
    if _native._is_capturing():
        raise NotImplementedError('group_norm requires eager execution')
    if not isinstance(input, Tensor) or input.device != 'cpu' or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('group_norm supports CPU float32/float64 tensors')
    if input.ndim < 2:
        raise ValueError('group_norm expects [N,C,*spatial]')
    if type(num_groups) is not int or num_groups <= 0:
        raise ValueError('num_groups must be a positive integer')
    batch, channels = input.shape[:2]
    if channels % num_groups:
        raise ValueError('num_groups must divide channels')
    _shape(input.shape, input.dtype)
    _shape([channels], input.dtype)
    _shape([batch, num_groups], input.dtype)
    spatial = 1
    for value in input.shape[2:]:spatial *= value
    grouped = [batch, num_groups, channels // num_groups, spatial]
    _shape(grouped, input.dtype)
    if batch * (channels // num_groups) * spatial == 1:
        raise ValueError('group_norm requires more than one value per group across the batch')
    eps = epsilon(eps, input.dtype)
    for value, name in ((weight, 'weight'), (bias, 'bias')):
        if value is not None and (not isinstance(value, Tensor) or value.device != 'cpu'
                                  or value.dtype != input.dtype or value.shape != [channels]):
            raise ValueError(name + ' must have input dtype on CPU with shape [channels]')
    if input.dtype == 'float32' and input.numel():
        flat = input if input.ndim <= 4 else input.reshape([batch, channels, spatial])
        output = flat.group_norm(num_groups, ones([channels]) if weight is None else weight,
                                 zeros([channels]) if bias is None else bias, eps)
        return output if input.ndim <= 4 else output.reshape(input.shape)
    value = input.reshape(grouped)
    average = mean(value, (2,3), True)
    centered = value - average
    std = (mean(centered.square(), (2,3), True) + eps).sqrt()
    output = centered / std if weight is None else centered * (weight.reshape([1,num_groups,channels//num_groups,1]) / std)
    if bias is not None:output = output + bias.reshape([1,num_groups,channels//num_groups,1])
    return output.reshape(input.shape)

