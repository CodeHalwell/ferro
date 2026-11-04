"""Shared eager attention validation and CPU mask composition."""
import math
import sys
from .. import _native
from .._native import Tensor


def _metadata(query, key, value):
    inputs = (query, key, value)
    if any(not isinstance(x, Tensor) or x.dtype != 'float32' for x in inputs):
        raise ValueError('attention inputs must be float32 Tensors')
    rank = query.ndim
    if rank not in (2, 3, 4) or any(x.ndim != rank for x in inputs):
        raise ValueError('attention expects matching rank-2, rank-3 or rank-4 inputs')
    if any(x.device != query.device for x in inputs):
        raise ValueError('attention inputs must share a device')
    if any(x.shape[:-2] != query.shape[:-2] for x in inputs):
        raise ValueError('attention batch/head axes must match; broadcasting is unsupported')
    if not query.shape[-1] or query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError('attention query/key feature width must be positive and matching; key/value lengths must match')
    prefix = query.shape[:-2]
    scores_shape = prefix + [query.shape[-2], key.shape[-2]]
    output_shape = prefix + [query.shape[-2], value.shape[-1]]
    if any(math.prod(shape) > sys.maxsize // 4 for shape in (scores_shape, output_shape)):
        raise ValueError('attention allocation capacity exceeds addressable storage')
    return math.prod(prefix), scores_shape, output_shape


def attention_scores(query, key, attn_mask=None, is_causal=False, scale=None):
    batch = math.prod(query.shape[:-2])
    q = query.reshape([batch, query.shape[-2], query.shape[-1]])
    k = key.reshape([batch, key.shape[-2], key.shape[-1]])
    scores = (q.bmm(k.transpose(1, 2)) * (query.shape[-1] ** -.5 if scale is None else scale)).reshape(query.shape[:-1] + [key.shape[-2]])
    if attn_mask is not None:
        if not isinstance(attn_mask, Tensor) or attn_mask.dtype not in ('bool', 'float32'):
            raise ValueError('attention mask must be bool or float32')
        if attn_mask.device != query.device:
            raise ValueError('attention mask must share the input device')
        padded = [1] * (scores.ndim - attn_mask.ndim) + attn_mask.shape
        if attn_mask.ndim > scores.ndim or any(a not in (1, b) for a, b in zip(padded, scores.shape)):
            raise ValueError('attention mask is not broadcastable to the score shape')
        if attn_mask.dtype == 'bool':
            values = attn_mask.reshape([attn_mask.numel()]).tolist()
            attn_mask = Tensor([0. if value else -math.inf for value in values], attn_mask.shape)
        scores = scores + attn_mask
    if is_causal and all(scores.shape):
        length, source = scores.shape[-2:]
        mask = Tensor([0. if j <= i else -math.inf for i in range(length) for j in range(source)], [length, source]).to(scores.device)
        scores = scores + mask
    return scores


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, scale=None, *, dropout_fn=None):
    batch, _, output_shape = _metadata(query, key, value)
    if type(is_causal) is not bool:
        raise TypeError('is_causal must be bool')
    if is_causal and attn_mask is not None:
        raise ValueError('attn_mask and is_causal cannot be set together')
    if not isinstance(dropout_p, (int, float)) or not math.isfinite(dropout_p) or not 0 <= dropout_p <= 1:
        raise ValueError('dropout_p must be finite and between zero and one')
    if scale is not None and (type(scale) not in (int, float) or not math.isfinite(scale)):
        raise ValueError('scale must be a finite real number')
    if (attn_mask is not None or scale is not None or dropout_p) and _native._is_capturing():
        raise NotImplementedError('masked/custom-scale/stochastic attention requires eager execution')
    fold = lambda x: x.reshape([batch, x.shape[-2], x.shape[-1]])
    if attn_mask is None and scale is None and dropout_p == 0:
        return _native.scaled_dot_product_attention(fold(query), fold(key), fold(value), is_causal).reshape(output_shape)
    if query.device != 'cpu':
        raise NotImplementedError('masked/custom-scale attention currently supports CPU tensors only')
    scores = attention_scores(query, key, attn_mask, is_causal, scale)
    if not batch or query.shape[-2] == 0 or key.shape[-2] == 0:
        return _native.scaled_dot_product_attention(fold(query), fold(key), fold(value), False).reshape(output_shape)
    length, source = scores.shape[-2:]
    rows = scores.reshape([batch * length, source]).tolist()
    live = Tensor([0. if all(value == -math.inf for value in row) else 1. for row in rows], query.shape[:-1] + [1])
    safe = _native.where(live, scores, Tensor.zeros(scores.shape))
    weights = _native.where(live, safe.softmax(-1), Tensor.zeros(scores.shape))
    if dropout_p:
        if dropout_fn is None:
            from .functional import dropout
            weights = dropout(weights, dropout_p, True)
        else:
            weights = dropout_fn(weights)
    return weights.reshape([batch, length, source]).bmm(fold(value)).reshape(output_shape)
