"""CPU channel dropout with saved Philox streams and native tensor adjoints."""
import math

from .. import _native
from .._native import Tensor, no_grad
from .._tensor_api import _shape
from .layers import Dropout
from . import functional as F


def _layout(input, dimensions, p, training, inplace):
    if not isinstance(input, Tensor): raise TypeError('channel dropout input must be a Tensor')
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('dropout probability must be finite and in [0,1]')
    if type(training) is not bool or type(inplace) is not bool:
        raise TypeError('training and inplace must be bool')
    if inplace: raise NotImplementedError('inplace channel dropout is unsupported')
    if input.device != 'cpu' or input.dtype not in ('float32', 'float64'):
        raise NotImplementedError('channel dropout requires CPU float32/float64 tensors')
    ranks = (2, 3) if dimensions == 1 else (3, 4) if dimensions == 2 else (4, 5)
    if input.ndim not in ranks: raise ValueError(f'dropout{dimensions}d requires rank {ranks[0]} or {ranks[1]}')
    unbatched = dimensions != 2 and input.ndim == ranks[0]
    shape = [1] + input.shape if unbatched else input.shape
    mask_shape = shape[:2] + [1] * (len(shape) - 2)
    active = training and p != 0. and input.numel() != 0
    if active:
        _shape(input.shape, input.dtype)
        _shape(mask_shape, 'float64')
        if _native._is_capturing(): raise NotImplementedError('active channel dropout requires eager execution')
    return unbatched, mask_shape, active


def _apply(input, p, training, inplace, dimensions, seed, offset):
    unbatched, mask_shape, active = _layout(input, dimensions, p, training, inplace)
    if active and p != 1.:
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError('seed must be an unsigned 64-bit integer')
        count = math.prod(mask_shape)
        if type(offset) is not int or not 0 <= offset < 2**64 or count > 2**64 - 1 - offset:
            raise ValueError('dropout counter range is exhausted or invalid')
    value = input.unsqueeze(0) if unbatched else input
    if not active: result = value
    elif p == 1.: result = value * 0.
    else:
        # Double sampling keeps the public probability precise even when float32
        # would round it to one. Torch scales F32 masks with an F32 denominator.
        ones = Tensor.ones(mask_shape).to(dtype='float64')
        mask = F.dropout(ones, p, True, seed=seed, offset=offset)
        if input.dtype == 'float32': mask = mask.ne(0.).float() / (1. - p)
        result = value * mask
    return result.squeeze(0) if unbatched else result


def dropout1d(input, p=.5, training=True, inplace=False, *, seed=None, offset=0):
    """Drop channels in CL/NCL inputs; seed/offset select a Philox stream."""
    return _apply(input, p, training, inplace, 1, seed, offset)


def dropout2d(input, p=.5, training=True, inplace=False, *, seed=None, offset=0):
    """Drop channels in NCHW or legacy NCL inputs (Torch's current rank-3 rule)."""
    return _apply(input, p, training, inplace, 2, seed, offset)


def dropout3d(input, p=.5, training=True, inplace=False, *, seed=None, offset=0):
    """Drop channels in CDHW/NCDHW inputs; seed/offset select a Philox stream."""
    return _apply(input, p, training, inplace, 3, seed, offset)


class _ChannelDropout(Dropout):
    """CPU float32/float64 channel masks with persistent native Philox state.

    The stream advances once per channel, only for nonempty stochastic calls.
    Checkpoints restore exact masks; Torch's RNG sequence is not reproduced.
    Inplace operations, active inference capture and stochastic module forward
    AD are unsupported; seeded functional calls support JVP. Native composition retains conservative saved-input version checks for products.
    """
    def forward(self, input):
        _, shape, active = _layout(input, self._dimensions, self.p, self.training, self.inplace)
        if not active or self.p == 1.:
            return _apply(input, self.p, self.training, self.inplace, self._dimensions, None, 0)
        if _native._forward_ad_is_active():
            raise NotImplementedError('stateful channel dropout does not support forward AD; use the seeded functional form')
        seed, offset = self._stream()
        output = _apply(input, self.p, True, False, self._dimensions, seed, offset)
        with no_grad(): self.rng_state.copy_(self._encode(seed, offset + math.prod(shape)))
        return output


class Dropout1d(_ChannelDropout):
    """Channel dropout for CL or NCL CPU float32/float64 tensors."""
    _dimensions = 1


class Dropout2d(_ChannelDropout):
    """Channel dropout for NCHW or legacy NCL (not unbatched CHW) tensors."""
    _dimensions = 2


class Dropout3d(_ChannelDropout):
    """Channel dropout for CDHW or NCDHW CPU float32/float64 tensors."""
    _dimensions = 3
