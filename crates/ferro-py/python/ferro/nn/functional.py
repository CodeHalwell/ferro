"""Differentiable compositions of native tensor operations.

Native losses are mean-reduced; 'sum' rescales by the reduced element count
and 'none' is supported only where noted.
"""
import math
import os
from .. import _native
from .._native import Tensor, rnn_cell, gru_cell, lstm_cell, kan, embedding as _embedding, one_hot


def linear(x, weight, bias=None):
    lead = x.shape[:-1]
    if len(lead) > 1:
        x = x.reshape([math.prod(lead), x.shape[-1]])
    result = x @ weight.transpose(0, 1)
    result = result if bias is None else result + bias
    return result.reshape(lead + [weight.shape[0]]) if len(lead) > 1 else result


def mse_loss(prediction, target, reduction='mean'):
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    difference = prediction - target
    squared = difference * difference
    return squared if reduction == 'none' else getattr(squared, reduction)()


def _pair(value):
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in pair):
        raise ValueError('Expected an integer or pair of nonnegative integers')
    return pair


def _square(value, name):
    a, b = _pair(value)
    if a != b:
        raise ValueError(f'{name} must be square; native pooling has one size per axis pair')
    return a


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """CPU f32 NCHW grouped convolution, symmetric per-axis padding."""
    return _native.conv2d_options(x, weight, bias, _pair(stride), _pair(padding), _pair(dilation), groups)


def unfold2d(x, kernel_size, stride=1, padding=0, dilation=1):
    """CPU NCHW -> [N,C*KH*KW,OH*OW], zero-filled padding."""
    return _native.unfold2d(x, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation))


def fold2d(x, output_size, kernel_size, stride=1, padding=0, dilation=1):
    """CPU columns -> NCHW; overlaps sum, not average."""
    return _native.fold2d(x, _pair(output_size), _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation))


def max_pool2d(x, kernel_size, stride=None, padding=0):
    """NCHW, square kernel/stride, no padding (padding must be 0)."""
    if _square(padding, 'padding'):
        raise ValueError('max_pool2d padding is unsupported')
    k = _square(kernel_size, 'kernel_size')
    return x.max_pool2d(k, k if stride is None else _square(stride, 'stride'))


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    """NCHW, square kernel/stride, no padding (padding must be 0)."""
    if _square(padding, 'padding'):
        raise ValueError('avg_pool2d padding is unsupported')
    k = _square(kernel_size, 'kernel_size')
    return x.avg_pool2d(k, k if stride is None else _square(stride, 'stride'))


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def gelu(x, approximate='none'):
    if approximate not in ('none', 'tanh'):
        raise ValueError("approximate must be 'none' or 'tanh'")
    return x.gelu_erf() if approximate == 'none' else x.gelu()


def silu(x):
    return x.silu()


def mish(x):
    return x.mish()


def elu(x, alpha=1.0):
    return x.elu(alpha)


def celu(x, alpha=1.0):
    return x.celu(alpha)


def selu(x):
    return x.selu()


def leaky_relu(x, negative_slope=0.01):
    return x.leaky_relu(negative_slope)


def relu6(x):
    return x.relu6()


def hardtanh(x, min_val=-1.0, max_val=1.0):
    return x.hardtanh(min_val, max_val)


def hardsigmoid(x):
    return x.hardsigmoid()


def hardswish(x):
    return x.hardswish()


def hardshrink(x, lambd=0.5):
    return x.hardshrink(lambd)


def softshrink(x, lambd=0.5):
    return x.softshrink(lambd)


def tanhshrink(x):
    return x.tanhshrink()


def softsign(x):
    return x.softsign()


def logsigmoid(x):
    return x.log_sigmoid()


def softplus(x, beta=1.0, threshold=20.0):
    """Native threshold is fixed at 20 in beta*x units."""
    if threshold != 20.0:
        raise ValueError('softplus threshold other than 20 is unsupported')
    return x.softplus() if beta == 1.0 else (x * beta).softplus() / beta


def threshold(x, threshold, value):
    return x.threshold(threshold, value)


def softmax(x, dim):
    return x.softmax(dim)


def log_softmax(x, dim):
    return x.log_softmax(dim)


def softmin(x, dim):
    return x.softmin(dim)


def normalize(x, p=2.0, dim=1, eps=1e-12):
    if p != 2.0:
        raise ValueError('normalize supports p=2 only')
    return x.normalize(dim, eps)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    return x1.cosine_similarity(x2, dim, eps)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6):
    return x1.pairwise_distance(x2, p, eps)


def pad(x, pad, mode='constant', value=0.0):
    """torch order: (last_before, last_after, prev_before, prev_after, ...)."""
    if mode != 'constant':
        raise ValueError("pad supports mode='constant' only")
    pad = tuple(pad)
    ndim = len(x.shape)
    if len(pad) % 2 or len(pad) // 2 > ndim or any(not isinstance(p, int) or p < 0 for p in pad):
        raise ValueError('pad needs nonnegative (before, after) pairs for at most every dim')
    pads = [0] * (2 * ndim)
    for i in range(len(pad) // 2):
        d = ndim - 1 - i
        pads[2 * d], pads[2 * d + 1] = pad[2 * i], pad[2 * i + 1]
    return x.pad_constant(pads, value)


def dropout(x, p=0.5, training=True, *, seed=None, offset=0):
    """Counter-based dropout; seed=None draws a fresh OS-random seed per call."""
    if not training or p == 0.0:
        return x
    return x.dropout(p, True, int.from_bytes(os.urandom(8), 'little') if seed is None else seed, offset)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Normalizes over the last dimension only."""
    _last_dim(x, normalized_shape, 'layer_norm')
    return x.layer_norm(weight, bias, eps)


def rms_norm(x, normalized_shape, weight=None, eps=None):
    _last_dim(x, normalized_shape, 'rms_norm')
    return x.rms_norm(weight, 1.1920928955078125e-07 if eps is None else eps)


def _last_dim(x, normalized_shape, name):
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if len(shape) != 1 or x.shape[-1:] != list(shape):
        raise ValueError(f'{name} normalizes over the last dimension only; got normalized_shape={normalized_shape}')


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    c = x.shape[1]
    return x.group_norm(num_groups, Tensor.ones([c]) if weight is None else weight,
                        Tensor.zeros([c]) if bias is None else bias, eps)


def batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    """[N,C] or NCHW. Training updates running_mean/running_var in place."""
    from .._grad_mode import no_grad
    c = x.shape[1]
    y, mean, var = x.batch_norm(Tensor.ones([c]) if weight is None else weight,
                                Tensor.zeros([c]) if bias is None else bias,
                                running_mean, running_var, training, momentum, eps)
    if training:
        with no_grad():
            running_mean.copy_(mean)
            running_var.copy_(var)
    return y


def embedding(input, weight):
    """I64 ids of any shape -> ids.shape + [dim]."""
    flat = input if len(input.shape) == 1 else input.reshape([math.prod(input.shape)])
    return _embedding(weight, flat).reshape(list(input.shape) + [weight.shape[1]])


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """[B, L, E] or [B, H, L, E] inputs; no masks, dropout or custom scale."""
    if attn_mask is not None or dropout_p or scale is not None:
        raise NotImplementedError('attn_mask, dropout_p and scale are unsupported')
    if len(query.shape) == 3:
        return _native.scaled_dot_product_attention(query, key, value, is_causal)
    if len(query.shape) != 4:
        raise ValueError('scaled_dot_product_attention expects rank-3 or rank-4 inputs')
    fold = lambda t: t.reshape([t.shape[0] * t.shape[1]] + t.shape[2:])
    out = _native.scaled_dot_product_attention(fold(query), fold(key), fold(value), is_causal)
    return out.reshape(query.shape[:3] + [value.shape[-1]])


def _reduce(loss, reduction, count):
    if reduction == 'mean':
        return loss
    if reduction == 'sum':
        return loss * float(count)
    raise ValueError("reduction must be 'mean' or 'sum' (native losses are mean-reduced; 'none' is unsupported)")


def _numel(*tensors):
    """Element count of the broadcast of `tensors`, which native losses mean over."""
    rank = max(len(t.shape) for t in tensors)
    dims = [[1] * (rank - len(t.shape)) + list(t.shape) for t in tensors]
    return math.prod(max(col) if min(col) != 0 else 0 for col in zip(*dims))


def _rows(x):
    return x.shape[0] if len(x.shape) > 1 else 1


def cross_entropy(input, target, reduction='mean'):
    """CPU f32 [N,C] logits with I64 [N] class ids or f32 [N,C] probabilities.

    Requires nonempty batches/classes. No weights, smoothing or ignore_index.
    """
    return _reduce(_native.cross_entropy(input, target), reduction, input.shape[0])


def nll_loss(input, target, reduction='mean'):
    """[N,C] log-probabilities and I64 [N] class ids. No weights or ignore_index."""
    return _reduce(-input.gather(1, target.unsqueeze(1)).mean(), reduction, input.shape[0])


def l1_loss(input, target, reduction='mean'):
    return _reduce(_native.l1_loss(input, target), reduction, _numel(input, target))


def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    if weight is not None:
        raise NotImplementedError('weight is unsupported')
    return _reduce(_native.bce_loss(input, target), reduction, _numel(input, target))


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    if weight is not None or pos_weight is not None:
        raise NotImplementedError('weight and pos_weight are unsupported')
    return _reduce(_native.bce_with_logits_loss(input, target), reduction, _numel(input, target))


def huber_loss(input, target, reduction='mean', delta=1.0):
    return _reduce(_native.huber_loss(input, target, delta), reduction, _numel(input, target))


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    return _reduce(_native.smooth_l1_loss(input, target, beta), reduction, _numel(input, target))


def kl_div(input, target, reduction='mean', log_target=False):
    """input is log-probabilities, target probabilities; also 'batchmean'."""
    if log_target:
        raise NotImplementedError('log_target=True is unsupported')
    n = _numel(input, target)
    loss = _native.kl_div_loss(input, target)
    return loss * float(n / input.shape[0]) if reduction == 'batchmean' else _reduce(loss, reduction, n)


def poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-8, reduction='mean'):
    if not log_input or full:
        raise NotImplementedError('only log_input=True, full=False is supported')
    return _reduce(_native.poisson_nll_loss(input, target), reduction, _numel(input, target))


def soft_margin_loss(input, target, reduction='mean'):
    return _reduce(_native.soft_margin_loss(input, target), reduction, _numel(input, target))


def hinge_embedding_loss(input, target, margin=1.0, reduction='mean'):
    return _reduce(_native.hinge_embedding_loss(input, target, margin), reduction, _numel(input, target))


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction='mean'):
    return _reduce(_native.margin_ranking_loss(input1, input2, target, margin), reduction, _numel(input1, input2, target))


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction='mean', eps=1e-8):
    return _reduce(_native.cosine_embedding_loss(input1, input2, target, margin, eps), reduction, _rows(input1))


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean'):
    if swap:
        raise NotImplementedError('swap=True is unsupported')
    return _reduce(_native.triplet_margin_loss(anchor, positive, negative, margin, p, eps), reduction, _rows(anchor))
