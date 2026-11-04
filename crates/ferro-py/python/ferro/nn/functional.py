"""Differentiable compositions of native tensor operations.

Loss reductions and options are documented by their individual functions.
"""
import math
import os
from .. import _native
from .._native import Tensor, rnn_cell, gru_cell, lstm_cell, kan, embedding as _embedding, one_hot
from ._rearrange import pixel_shuffle, pixel_unshuffle, channel_shuffle, glu
from .._spatial import (conv1d, conv3d, conv_transpose1d, conv_transpose2d,
                       adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d,
                       conv_transpose3d, max_pool1d, max_pool3d, avg_pool1d, avg_pool3d,
                       adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d)


def linear(x, weight, bias=None):
    if not isinstance(x, Tensor) or not isinstance(weight, Tensor) or (bias is not None and not isinstance(bias, Tensor)):
        raise TypeError('linear expects Tensor input, weight and optional bias')
    if x.dtype not in ('float32', 'float64') or weight.dtype != x.dtype or (bias is not None and bias.dtype != x.dtype):
        raise TypeError('linear requires matching float32 or float64 input and parameters')
    if x.device != weight.device or (bias is not None and bias.device != x.device):
        raise ValueError('linear input and parameters must share a device')
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('linear expects input [..., in_features] and weight [out_features, in_features]')
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
    """CPU f32 NCHW/CHW grouped convolution, including same/valid padding."""
    from .._spatial import conv2d as apply
    return apply(x, weight, bias, stride, padding, dilation, groups)


def unfold2d(x, kernel_size, stride=1, padding=0, dilation=1):
    """CPU NCHW -> [N,C*KH*KW,OH*OW], zero-filled padding."""
    return _native.unfold2d(x, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation))


def fold2d(x, output_size, kernel_size, stride=1, padding=0, dilation=1):
    """CPU columns -> NCHW; overlaps sum, not average."""
    return _native.fold2d(x, _pair(output_size), _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation))


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """CPU NCHW/CHW pooling; bool flags follow the functional Torch order."""
    from .._spatial import max_pool2d as apply
    return apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    """CPU NCHW/CHW average pooling with rectangular windows and clipped padding."""
    from .._spatial import avg_pool2d as apply
    return apply(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


def _activation_inplace(inplace):
    if type(inplace) is not bool:
        raise TypeError('inplace must be bool')
    if inplace:
        raise NotImplementedError('inplace activations are unsupported')


def relu(input, inplace=False):
    _activation_inplace(inplace)
    return input.relu()


def sigmoid(input):
    return input.sigmoid()


def tanh(input):
    return input.tanh()


def gelu(input, approximate='none'):
    if approximate not in ('none', 'tanh'):
        raise ValueError("approximate must be 'none' or 'tanh'")
    return input.gelu_erf() if approximate == 'none' else input.gelu()


def silu(input, inplace=False):
    _activation_inplace(inplace)
    return input.silu()


def mish(input, inplace=False):
    _activation_inplace(inplace)
    return input.mish()


def elu(input, alpha=1.0, inplace=False):
    _activation_inplace(inplace)
    return input.elu(alpha)


def celu(input, alpha=1.0, inplace=False):
    _activation_inplace(inplace)
    return input.celu(alpha)


def selu(input, inplace=False):
    _activation_inplace(inplace)
    return input.selu()


def leaky_relu(input, negative_slope=0.01, inplace=False):
    _activation_inplace(inplace)
    return input.leaky_relu(negative_slope)


def relu6(input, inplace=False):
    _activation_inplace(inplace)
    return input.relu6()


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    _activation_inplace(inplace)
    return input.hardtanh(min_val, max_val)


def hardsigmoid(input, inplace=False):
    _activation_inplace(inplace)
    return input.hardsigmoid()


def hardswish(input, inplace=False):
    _activation_inplace(inplace)
    return input.hardswish()


def hardshrink(input, lambd=0.5):
    return input.hardshrink(lambd)


def softshrink(input, lambd=0.5):
    return input.softshrink(lambd)


def tanhshrink(input):
    return input.tanhshrink()


def softsign(input):
    return input.softsign()


def logsigmoid(input):
    return input.log_sigmoid()


def softplus(input, beta=1.0, threshold=20.0):
    """Typed native softplus with input-dtype threshold and beta arithmetic."""
    return input.softplus(beta, threshold)


def threshold(input, threshold, value, inplace=False):
    _activation_inplace(inplace)
    return input.threshold(threshold, value)


def softmax(input, dim):
    return input.softmax(dim)


def log_softmax(input, dim):
    return input.log_softmax(dim)


def softmin(input, dim):
    return input.softmin(dim)


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    """Native first-order p-normalization; general axes/dtypes use CPU storage."""
    from ._normalize import normalize as apply
    return apply(input, p, dim, eps, out)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    return x1.cosine_similarity(x2, dim, eps)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6):
    return x1.pairwise_distance(x2, p, eps)


def pad(input, pad, mode='constant', value=None):
    """CPU f32 Torch-order signed padding/cropping; constant/reflect/replicate/circular."""
    from .._spatial import pad as apply
    return apply(input, pad, mode, value)


def dropout(input, p=0.5, training=True, inplace=False, *, seed=None, offset=0):
    """Counter-based dropout; seed=None draws a fresh OS-random seed per call."""
    if not isinstance(input, Tensor):
        raise TypeError('dropout input must be a Tensor')
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('dropout probability must be finite and in [0, 1]')
    if type(training) is not bool or type(inplace) is not bool:
        raise TypeError('training and inplace must be bool')
    if inplace:
        raise NotImplementedError('inplace dropout is unsupported')
    if not training or p == 0.0 or input.numel() == 0:
        return input
    if p == 1.0:
        return input * 0.
    if _native._is_capturing():
        raise NotImplementedError('stochastic dropout requires eager execution')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError('seed must be an unsigned 64-bit integer')
    if type(offset) is not int or not 0 <= offset < 2**64 or input.numel() > 2**64 - 1 - offset:
        raise ValueError('dropout counter range is exhausted or invalid')
    _native._forward_ad_check_random()
    return input.dropout(p, True, int.from_bytes(os.urandom(8), 'little') if seed is None else seed, offset)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Eager CPU normalization over the requested trailing dimensions."""
    from ._layer_norm import layer_norm as apply
    return apply(input, normalized_shape, weight, bias, eps)


def rms_norm(input, normalized_shape, weight=None, eps=None):
    from ._layer_norm import rms_norm as apply
    return apply(input, normalized_shape, weight, eps)


def _last_dim(x, normalized_shape, name):
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if len(shape) != 1 or x.shape[-1:] != list(shape):
        raise ValueError(f'{name} normalizes over the last dimension only; got normalized_shape={normalized_shape}')


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    from ._group_norm import group_norm as apply
    return apply(input, num_groups, weight, bias, eps)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    """CPU floating batch normalization over all axes except channels."""
    from ._normalization_ops import batch_norm as apply
    return apply(input, running_mean, running_var, weight, bias, training, momentum, eps)


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=.1, eps=1e-5):
    """Eager CPU instance normalization with staged running-stat updates."""
    from ._instance_norm import instance_norm as apply
    return apply(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
    """Dense CPU F32/F64 lookup; renorm needs whole contiguous writable weights."""
    from ._embedding import embedding as apply
    return apply(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2., scale_grad_by_freq=False,
                  mode='mean', sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None):
    from ._embedding_bag import embedding_bag as apply
    return apply(input, weight, offsets, max_norm, norm_type, scale_grad_by_freq, mode, sparse,
                 per_sample_weights, include_last_offset, padding_idx)


def local_response_norm(input, size, alpha=1e-4, beta=.75, k=1.):
    from ._local_response_norm import local_response_norm as apply
    return apply(input, size, alpha, beta, k)


def dropout1d(input, p=.5, training=True, inplace=False, *, seed=None, offset=0):
    from ._channel_dropout import dropout1d as apply
    return apply(input, p, training, inplace, seed=seed, offset=offset)


def dropout2d(input, p=.5, training=True, inplace=False, *, seed=None, offset=0):
    from ._channel_dropout import dropout2d as apply
    return apply(input, p, training, inplace, seed=seed, offset=offset)


def dropout3d(input, p=.5, training=True, inplace=False, *, seed=None, offset=0):
    from ._channel_dropout import dropout3d as apply
    return apply(input, p, training, inplace, seed=seed, offset=offset)


def alpha_dropout(input, p=.5, training=False, inplace=False, *, seed=None, offset=0):
    from ._alpha_dropout import alpha_dropout as apply
    return apply(input, p, training, inplace, seed=seed, offset=offset)


def feature_alpha_dropout(input, p=.5, training=False, inplace=False, *, seed=None, offset=0):
    from ._alpha_dropout import feature_alpha_dropout as apply
    return apply(input, p, training, inplace, seed=seed, offset=offset)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Rank 2-4 attention; matched batch/head axes and positive query/key width.

    Bool masks use True for allowed positions; float masks add to logits.
    Masks/custom scale are CPU eager paths. Nonzero dropout is unsupported.
    """
    from ._attention_ops import scaled_dot_product_attention as apply
    return apply(query, key, value, attn_mask, dropout_p, is_causal, scale)


def _reduce(loss, reduction, count):
    if reduction == 'mean':
        return loss
    if reduction == 'sum':
        if count == 0:
            # An empty sum is 0, but the native mean over nothing is NaN; `where`
            # keeps the result on the graph so inputs still get (empty) grads.
            zero = Tensor.zeros(loss.shape)
            return _native.where(zero, loss, zero)
        return loss * float(count)
    raise ValueError("reduction must be 'mean' or 'sum' (native losses are mean-reduced; 'none' is unsupported)")


def _numel(*operands):
    """Element count of the broadcast of `operands` (tensors or shapes), which
    native losses mean over."""
    shapes = [list(t) if isinstance(t, (list, tuple)) else list(t.shape) for t in operands]
    rank = max(len(s) for s in shapes)
    dims = [[1] * (rank - len(s)) + s for s in shapes]
    return math.prod(max(col) if min(col) != 0 else 0 for col in zip(*dims))


def _classification_options(weight, ignore_index, reduction, label_smoothing=0.0):
    # The original Ferro API used reduction as the third positional argument.
    if isinstance(weight, str):
        if reduction != 'mean':
            raise TypeError('reduction was supplied twice')
        reduction, weight = weight, None
    if reduction not in ('none', 'sum', 'mean'):
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if type(ignore_index) is not int:
        raise TypeError('ignore_index must be an integer')
    if not isinstance(label_smoothing, (int, float)) or not math.isfinite(label_smoothing) or not 0 <= label_smoothing <= 1:
        raise ValueError('label_smoothing must be finite and between zero and one')
    return weight, reduction, float(label_smoothing)


def _classification_weight(weight, classes=None):
    if weight is None:
        return
    if not isinstance(weight, Tensor) or weight.dtype != 'float32':
        raise ValueError('class weights must be a float32 Tensor')
    if weight.device != 'cpu':
        raise ValueError('classification losses currently support CPU tensors only')
    if len(weight.shape) != 1 or not weight.shape[0] or (classes is not None and weight.shape != [classes]):
        raise ValueError('class weights must have shape [number of classes]')


def _classification_inputs(input, target, weight, ignore_index, allow_probabilities):
    if not isinstance(input, Tensor) or input.dtype != 'float32':
        raise ValueError('classification input must be a float32 Tensor')
    if not isinstance(target, Tensor):
        raise ValueError('classification target must be a Tensor')
    if input.device != 'cpu' or target.device != 'cpu':
        raise ValueError('classification losses currently support CPU tensors only')
    shape = input.shape
    if not shape:
        raise ValueError('classification input needs a class axis')
    classes = shape[0] if len(shape) == 1 else shape[1]
    if not classes:
        raise ValueError('classification input must have at least one class')
    output_shape = [] if len(shape) == 1 else shape[:1] + shape[2:]
    _classification_weight(weight, classes)
    probabilities = allow_probabilities and target.shape == shape
    if probabilities:
        if target.dtype != 'float32':
            raise ValueError('probability targets must have dtype float32')
        if ignore_index >= 0:
            raise ValueError('ignore_index is unsupported for probability targets')
        return classes, output_shape, None
    if target.dtype != 'int64' or target.shape != output_shape:
        raise ValueError('class targets must be int64 with the input class axis removed')
    if weight is not None and weight.requires_grad:
        raise ValueError('class weights cannot require gradients with index targets')
    labels = target.reshape([math.prod(output_shape)]).tolist()
    if any(label != ignore_index and not 0 <= label < classes for label in labels):
        raise ValueError('class target is out of bounds')
    return classes, output_shape, labels


def _classification_rows(input, classes):
    if len(input.shape) == 1:
        return input.reshape([1, classes])
    order = [0] + list(range(2, len(input.shape))) + [1]
    return input.permute(*order).reshape([math.prod(input.shape) // classes, classes])


def _classification_indices(log_probs, labels, weight, ignore_index, reduction, output_shape, smoothing):
    positions = [i for i, label in enumerate(labels) if label != ignore_index]
    ids = Tensor.from_i64([labels[i] for i in positions], [len(positions)])
    # Selecting valid rows keeps ignored entries out of the mean denominator
    # and gives exactly zero gradients even for an all-ignored mean (NaN).
    selected = log_probs.index_select(0, positions)
    losses = -selected.gather(1, ids.unsqueeze(1)).reshape([len(positions)])
    selected_weights = None if weight is None else weight.gather(0, ids)
    if selected_weights is not None:
        losses = losses * selected_weights
    smooth = None if not smoothing else -(selected if weight is None else selected * weight).sum_dim(1)
    if reduction == 'none':
        if smooth is not None:
            losses = losses * (1.0 - smoothing) + smooth * (smoothing / log_probs.shape[1])
        index = Tensor.from_i64(positions, [len(positions)])
        return Tensor.zeros([len(labels)]).scatter_add(0, index, losses).reshape(output_shape)
    total = losses.sum()
    smooth_total = None if smooth is None else smooth.sum()
    if reduction == 'mean':
        denominator = float(len(positions)) if selected_weights is None else selected_weights.sum()
        total = total / denominator
        if smooth_total is not None:
            smooth_total = smooth_total / denominator
    # Normalize the two terms separately, matching PyTorch even when selected
    # weights sum to zero: a NaN index term must not become a finite/Inf result.
    return total if smooth_total is None else total * (1.0 - smoothing) + smooth_total * (smoothing / log_probs.shape[1])


def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    """CPU float32 cross entropy, with class axis 1 (axis 0 when unbatched).

    Supports index or same-shape probability targets, weights, ignored indices,
    smoothing, and none/sum/mean reductions. Probability targets are not checked
    for normalization, as in PyTorch; their gradients remain live. Index means
    divide by selected class-weight sum, probability means by position count.
    Empty/all-ignored means are NaN. Positive class count is required; float64,
    device execution and capture are unsupported. A third positional reduction
    string remains accepted for compatibility with earlier Ferro versions.
    """
    weight, reduction, smoothing = _classification_options(weight, ignore_index, reduction, label_smoothing)
    classes, output_shape, labels = _classification_inputs(input, target, weight, ignore_index, True)
    log_probs = _classification_rows(input, classes).log_softmax(1)
    if labels is not None:
        return _classification_indices(log_probs, labels, weight, ignore_index, reduction, output_shape, smoothing)
    probabilities = _classification_rows(target, classes)
    if smoothing:
        probabilities = probabilities * (1.0 - smoothing) + smoothing / classes
    weighted = probabilities if weight is None else probabilities * weight
    losses = -(log_probs * weighted).sum_dim(1)
    if reduction == 'none':
        return losses.reshape(output_shape)
    return losses.sum() if reduction == 'sum' else losses.mean()


def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    """CPU float32 log-probabilities and int64 targets, class axis 1 (or 0).

    Supports spatial/unbatched inputs, weights, ignored indices and all three
    reductions, with the same index-target mean rules as cross_entropy.
    Float64, device execution and capture are unsupported. The historical third
    positional reduction string is accepted.
    """
    weight, reduction, _ = _classification_options(weight, ignore_index, reduction)
    classes, output_shape, labels = _classification_inputs(input, target, weight, ignore_index, False)
    return _classification_indices(_classification_rows(input, classes), labels, weight, ignore_index, reduction, output_shape, 0.0)


def l1_loss(input, target, reduction='mean', weight=None):
    from ._loss_ops import l1_loss as apply
    return apply(input, target, reduction, weight)


def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    """CPU floating probabilities, broadcast numeric weights and all reductions."""
    from ._loss_ops import binary_cross_entropy as apply
    return apply(input, target, weight, reduction)


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    """CPU float32/float64 logits and targets, numeric weights and positive-class weights."""
    from ._loss_ops import binary_cross_entropy as apply
    return apply(input, target, weight, reduction, logits=True, pos_weight=pos_weight)


def huber_loss(input, target, reduction='mean', delta=1.0, weight=None):
    from ._loss_ops import huber_loss as apply
    return apply(input, target, reduction, delta, weight)


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    from ._loss_ops import smooth_l1_loss as apply
    return apply(input, target, reduction, beta)


def kl_div(input, target, reduction='mean', log_target=False):
    from ._loss_ops import kl_div as apply
    return apply(input, target, reduction, log_target)


def poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-8, reduction='mean'):
    from ._loss_ops import poisson_nll_loss as apply
    return apply(input, target, log_input, full, eps, reduction)


def soft_margin_loss(input, target, reduction='mean'):
    from ._loss_ops import soft_margin_loss as apply
    return apply(input, target, reduction)


def hinge_embedding_loss(input, target, margin=1.0, reduction='mean'):
    from ._loss_ops import hinge_embedding_loss as apply
    return apply(input, target, margin, reduction)


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction='mean'):
    from ._loss_ops import margin_ranking_loss as apply
    return apply(input1, input2, target, margin, reduction)


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction='mean', eps=9.999999960041972e-13):
    from ._loss_ops import cosine_embedding_loss as apply
    return apply(input1, input2, target, margin, reduction, eps)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean'):
    from ._loss_ops import triplet_margin_loss as apply
    return apply(anchor, positive, negative, margin, p, eps, swap, reduction)


def prelu(input, weight):
    from ._catalog import prelu as apply
    return apply(input, weight)


def bilinear(input1, input2, weight, bias=None):
    from ._catalog import bilinear as apply
    return apply(input1, input2, weight, bias)


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction='mean'):
    from ._catalog import gaussian_nll_loss as apply
    return apply(input, target, var, full, eps, reduction)
