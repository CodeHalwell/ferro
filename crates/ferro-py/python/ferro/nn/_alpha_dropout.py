"""Self-normalizing dropout using native tensor operations and Philox masks."""
import math

from .. import _native
from .._native import Tensor, no_grad
from .._tensor_api import _shape
from .layers import Dropout
from . import functional as F


def _layout(input, p, training, inplace, feature):
    if not isinstance(input, Tensor): raise TypeError('alpha dropout input must be a Tensor')
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('dropout probability must be finite and in [0,1]')
    if type(training) is not bool or type(inplace) is not bool: raise TypeError('training and inplace must be bool')
    if inplace: raise NotImplementedError('inplace alpha dropout is unsupported')
    if input.device != 'cpu' or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('alpha dropout requires CPU float32/float64 tensors')
    active = training and p != 0. and input.numel() != 0
    if not active: return input.shape, False
    if feature and p != 1. and input.ndim < 2:
        raise ValueError('feature alpha dropout requires at least two input dimensions')
    shape = input.shape[:2] + [1] * (input.ndim - 2) if feature and input.ndim >= 2 else input.shape
    _shape(input.shape, input.dtype)
    _shape(shape, 'float64')
    if _native._is_capturing(): raise NotImplementedError('active alpha dropout requires eager execution')
    return shape, True


def _apply(input, p, training, inplace, feature, seed, offset):
    shape, active = _layout(input, p, training, inplace, feature)
    if not active: return input
    if p == 1.: return input * 0.
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError('seed must be an unsigned 64-bit integer')
    count = math.prod(shape)
    if type(offset) is not int or not 0 <= offset < 2**64 or count > 2**64 - 1 - offset:
        raise ValueError('dropout counter range is exhausted or invalid')
    ones = Tensor.ones(shape).to(dtype='float64')
    noise = F.dropout(ones, p, True, seed=seed, offset=offset).ne(0.).to(dtype=input.dtype)
    alpha = 1.7580993408473766
    scale = 1. / math.sqrt((alpha * alpha * p + 1.) * (1. - p))
    shift = (noise - 1.) * (alpha * scale) + alpha * scale * p
    return input * (noise * scale) + shift


def alpha_dropout(input, p=.5, training=False, inplace=False, *, seed=None, offset=0):
    """Elementwise alpha dropout; explicit seed/offset select a Philox stream."""
    return _apply(input, p, training, inplace, False, seed, offset)


def feature_alpha_dropout(input, p=.5, training=False, inplace=False, *, seed=None, offset=0):
    """Share each [N,C] mask over remaining spatial axes; no unbatched heuristic."""
    return _apply(input, p, training, inplace, True, seed, offset)


class _AlphaDropout(Dropout):
    """CPU float32/float64 alpha dropout with an owning persistent RNG stream.

    Masks and affine scaling use native operations. The stream advances per
    element or channel on nonempty stochastic calls, independently of Torch's
    RNG sequence. Inplace execution, active capture and stochastic module JVP
    are unsupported; seeded functional calls support JVP. Native products keep
    conservative saved-input version checks.
    """
    def forward(self, input):
        shape, active = _layout(input, self.p, self.training, self.inplace, self._feature)
        if not active or self.p == 1.:
            return _apply(input, self.p, self.training, self.inplace, self._feature, None, 0)
        if _native._forward_ad_is_active():
            raise NotImplementedError('stateful alpha dropout does not support forward AD; use the seeded functional form')
        seed, offset = self._stream()
        output = _apply(input, self.p, True, False, self._feature, seed, offset)
        with no_grad(): self.rng_state.copy_(self._encode(seed, offset + math.prod(shape)))
        return output


class AlphaDropout(_AlphaDropout):
    """Elementwise self-normalizing dropout with native checkpointed masks."""
    _feature = False


class FeatureAlphaDropout(_AlphaDropout):
    """Channel self-normalizing dropout for [N,C,*spatial] floating inputs."""
    _feature = True
