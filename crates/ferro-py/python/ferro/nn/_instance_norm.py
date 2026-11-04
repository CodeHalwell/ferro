"""CPU instance normalization composed through the existing native BatchNorm."""
from numbers import Real
import warnings

from .. import _native
from .._native import Tensor, no_grad
from .._ops import mean
from .._tensor_api import _shape, full
from ._normalization_ops import _affine, _channel_tensor, batch_norm
from .layers import _BatchNorm


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=.1, eps=1e-5):
    """Normalize spatial values independently for each CPU sample/channel.

    Float32/float64 first-order native gradients are supported. A zero batch
    requires absent affine tensors; input statistics then write NaN to running
    buffers as in Torch2.13. Zero spatial size with a nonempty batch preserves
    statistics. Updating statistics requires independent whole CPU buffers.
    """
    if _native._is_capturing():
        raise NotImplementedError('instance_norm requires eager execution')
    if not isinstance(input, Tensor) or input.device != 'cpu' or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('instance_norm supports CPU float32/float64 tensors')
    if input.ndim < 3 or input.shape[1] == 0:
        raise ValueError('instance_norm expects [N,C,*spatial] with positive channels')
    if type(use_input_stats) is not bool:
        raise TypeError('use_input_stats must be bool')
    if not isinstance(momentum, Real) or not isinstance(eps, Real):
        raise TypeError('instance_norm momentum and eps must be real numbers')
    momentum, eps = float(momentum), float(eps)
    if use_input_stats and eps <= 0:
        raise ValueError('instance_norm eps must be positive when using input statistics')
    _shape(input.shape, input.dtype)
    batch, channels = input.shape[:2]
    spatial = 1
    for size in input.shape[2:]:
        spatial *= size
    if use_input_stats and spatial == 1:
        raise ValueError('instance_norm requires more than one spatial element when using input statistics')
    if (running_mean is None) != (running_var is None):
        raise ValueError('instance_norm running_mean and running_var must both be present or both be None')
    _shape([batch * channels], input.dtype)
    for value, name, running in ((weight, 'weight', False), (bias, 'bias', False),
                                 (running_mean, 'running_mean', True), (running_var, 'running_var', True)):
        _channel_tensor(value, channels, input.dtype, name, running)
    if batch == 0 and (weight is not None or bias is not None):
        raise ValueError('instance_norm with affine tensors does not support zero batch size')
    if not use_input_stats:
        return batch_norm(input, running_mean, running_var, weight, bias, False, momentum, eps)
    if spatial == 0 and batch != 0:
        return _affine(input.clone(), weight, bias)
    if running_mean is not None and running_var is not None and running_mean._shares_storage(running_var):
        raise NotImplementedError('instance_norm running statistics must use independent storage')
    with no_grad():
        for value in (running_mean, running_var):
            if value is not None:
                value.copy_(value)
    if batch == 0:
        with no_grad():
            for value in (running_mean, running_var):
                if value is not None:
                    value.copy_(full([channels], float('nan'), dtype=input.dtype))
        return input.clone()
    def repeat(value, owning=False):
        if value is None:
            return None
        repeated = value.reshape([1, channels]).expand(batch, channels).reshape([batch * channels])
        return repeated.clone() if owning else repeated
    repeated_mean, repeated_var = repeat(running_mean, True), repeat(running_var, True)
    output = batch_norm(input.reshape([1, batch * channels, spatial]), repeated_mean, repeated_var,
                        repeat(weight), repeat(bias), True, momentum, eps).reshape(input.shape)
    with no_grad():
        updated_mean = None if repeated_mean is None else mean(repeated_mean.reshape([batch, channels]), 0)
        updated_var = None if repeated_var is None else mean(repeated_var.reshape([batch, channels]), 0)
        if running_mean is not None:
            running_mean.copy_(updated_mean)
        if running_var is not None:
            running_var.copy_(updated_var)
    return output


class _InstanceNorm(_BatchNorm):
    _builtin_init = True
    _no_batch_rank = 0

    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=False, track_running_stats=False,
                 device=None, dtype=None, *, bias=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype, bias=bias)

    def forward(self, input):
        if not isinstance(input, Tensor):
            raise TypeError('InstanceNorm input must be a Tensor')
        if input.ndim not in self._ranks:
            raise ValueError(f'{type(self).__name__} expects input rank in {self._ranks}, got {input.shape}')
        feature_dim = input.ndim - self._no_batch_rank
        if input.shape[feature_dim] != self.num_features:
            if self.affine:
                raise ValueError('InstanceNorm input channels must match num_features when affine=True')
            warnings.warn('input channels do not match num_features, which is unused when affine=False', stacklevel=2)
        unbatched = input.ndim == self._no_batch_rank
        value = input.unsqueeze(0) if unbatched else input
        output = instance_norm(value, self.running_mean, self.running_var,
                               None if self.weight is None else self.weight.tensor(),
                               None if self.bias is None else self.bias.tensor(),
                               self.training or not self.track_running_stats,
                               0. if self.momentum is None else self.momentum, self.eps)
        return output.squeeze(0) if unbatched else output


class InstanceNorm1d(_InstanceNorm):
    """Per-sample channel normalization for [C,L] or [N,C,L] CPU inputs."""
    _ranks = (2, 3)
    _no_batch_rank = 2


class InstanceNorm2d(_InstanceNorm):
    """Per-sample channel normalization for [C,H,W] or [N,C,H,W] CPU inputs."""
    _ranks = (3, 4)
    _no_batch_rank = 3


class InstanceNorm3d(_InstanceNorm):
    """Per-sample channel normalization for [C,D,H,W] or [N,C,D,H,W] CPU inputs."""
    _ranks = (4, 5)
    _no_batch_rank = 4
