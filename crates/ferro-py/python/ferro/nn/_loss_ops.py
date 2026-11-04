"""Elementwise native losses and explicit reductions."""
import math
from .. import _native, linalg
from .._boolean_api import where
from .._ops import zeros_like
from .._ops import sum as tensor_sum
from .._native import Tensor


def reduce(value, reduction):
    if reduction == 'none':
        return value
    if reduction == 'sum':
        return value.sum()
    if reduction == 'mean':
        return value.mean()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _weight(value, input, name):
    if value is None:
        return None
    if not isinstance(value, Tensor) or value.device != input.device:
        raise ValueError(name + ' must be a Tensor on the input device')
    if value.dtype not in ('float32', 'float64', 'int64', 'bool') or (name == 'pos_weight' and value.dtype == 'bool'):
        raise TypeError(name + ' requires float32/float64/int64; bool is supported only for weight')
    if value.requires_grad and _native.is_grad_enabled():
        raise RuntimeError(name + ' is not differentiable')
    return value


def binary_cross_entropy(input, target, weight=None, reduction='mean', *, logits=False, pos_weight=None):
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(input, Tensor) or not isinstance(target, Tensor):
        raise TypeError('binary cross entropy expects Tensor input and target')
    weight = _weight(weight, input, 'weight')
    positive = _weight(pos_weight, input, 'pos_weight')
    return _native._binary_loss(input, target, logits, positive, weight, reduction)


def l1_loss(input, target, reduction='mean', weight=None):
    value = (input - target).abs()
    if weight is None:
        return reduce(value, reduction)
    if weight.shape != input.shape:
        raise ValueError('weight and input must have the same shape')
    value = value * weight
    return value.sum() / weight.sum() if reduction == 'mean' else reduce(value, reduction)


def huber_loss(input, target, reduction='mean', delta=1.0, weight=None):
    if input.dtype != target.dtype:
        raise NotImplementedError('huber_loss currently requires matching input and target dtypes')
    if not delta > 0:
        raise ValueError('huber_loss delta must be positive')
    d = input - target
    if math.isinf(delta) or (input.dtype == 'float32' and delta > 3.4028234663852886e38):
        value = .5 * d.square()
    else:
        value = where(d.abs() < delta, .5 * d.square(), delta * (d.abs() - .5 * delta))
    if weight is not None:
        if weight.shape != input.shape:
            raise ValueError('weight and input must have the same shape')
        value = value * weight
    return reduce(value, reduction)


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    if not beta >= 0:
        raise ValueError('smooth_l1_loss beta must be nonnegative')
    d = input - target
    value = d.abs() if beta == 0 else where(d.abs() < beta, .5 * d.square() / beta, d.abs() - .5 * beta)
    return reduce(value, reduction)


def kl_div(input, target, reduction='mean', log_target=False):
    if log_target:
        value = target.exp() * (target - input)
    else:
        entropy = target * target.log()
        value = where(target == 0, zeros_like(entropy), entropy) - target * input
    if reduction == 'batchmean':
        return value.sum() / (input.shape[0] if input.ndim else 1)
    return reduce(value, reduction)


def poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-8, reduction='mean'):
    value = input.exp() - target * input if log_input else input - target * (input + eps).log()
    if full:
        stirling = target * target.log() - target + .5 * (target * (2 * math.pi)).log()
        value = value + where(target > 1, stirling, zeros_like(stirling))
    return reduce(value, reduction)


def soft_margin_loss(input, target, reduction='mean'):
    if target.dtype in ('float32', 'float64') and input.dtype != target.dtype:
        raise NotImplementedError('soft_margin_loss currently requires matching floating input and target dtypes')
    if target.requires_grad and _native.is_grad_enabled():
        raise RuntimeError('soft_margin_loss is not differentiable with respect to target')
    z = -input * target
    return reduce(_native._binary_loss(z, zeros_like(z), True, None), reduction)


def hinge_embedding_loss(input, target, margin=1.0, reduction='mean'):
    positive = where(target != -1, input, zeros_like(input))
    negative = (margin - input).clamp(0., float('inf'))
    return reduce(positive + where(target != 1, negative, zeros_like(negative)), reduction)


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction='mean'):
    if input1.ndim != input2.ndim or input1.ndim != target.ndim:
        raise ValueError('margin_ranking_loss inputs and target must have matching ranks')
    return reduce((-target * (input1 - input2) + margin).clamp(0., float('inf')), reduction)


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction='mean', eps=9.999999960041972e-13):
    if input1.ndim not in (1, 2) or input2.ndim != input1.ndim or target.ndim != input1.ndim - 1:
        raise ValueError('cosine_embedding_loss expects one or two dimensional inputs and row labels')
    dot = tensor_sum(input1 * input2, -1)
    norm = ((tensor_sum(input1.square(), -1) + eps) * (tensor_sum(input2.square(), -1) + eps)).sqrt()
    cosine = dot / norm
    positive = where(target == 1, 1 - cosine, zeros_like(cosine))
    negative = where(target == -1, (cosine - margin).clamp(0., float('inf')), zeros_like(cosine))
    return reduce(positive + negative, reduction)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean'):
    if anchor.ndim != positive.ndim or anchor.ndim != negative.ndim:
        raise ValueError('triplet_margin_loss inputs must have matching ranks')
    if margin <= 0:
        raise ValueError('triplet_margin_loss margin must be positive')
    dpos = linalg.vector_norm(anchor - positive + eps, ord=p, dim=-1)
    dneg = linalg.vector_norm(anchor - negative + eps, ord=p, dim=-1)
    if swap:
        dpn = linalg.vector_norm(positive - negative + eps, ord=p, dim=-1)
        dneg = where(dpn < dneg, dpn, where(dpn > dneg, dneg, (dpn + dneg) * .5))
    return reduce((dpos - dneg + margin).clamp(0., float('inf')), reduction)
