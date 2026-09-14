"""Differentiable compositions of native tensor operations."""
from .._native import rnn_cell, gru_cell, lstm_cell, kan

def linear(x, weight, bias=None):
    result = x @ weight.transpose(0, 1)
    return result if bias is None else result + bias


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


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """CPU f32 NCHW grouped convolution, symmetric per-axis padding."""
    from .._native import conv2d_options
    return conv2d_options(x, weight, bias, _pair(stride), _pair(padding), _pair(dilation), groups)


def unfold2d(x, kernel_size, stride=1, padding=0, dilation=1):
    """CPU NCHW -> [N,C*KH*KW,OH*OW], zero-filled padding."""
    from .._native import unfold2d as native
    return native(x, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation))


def fold2d(x, output_size, kernel_size, stride=1, padding=0, dilation=1):
    """CPU columns -> NCHW; overlaps sum, not average."""
    from .._native import fold2d as native
    return native(x, _pair(output_size), _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation))
