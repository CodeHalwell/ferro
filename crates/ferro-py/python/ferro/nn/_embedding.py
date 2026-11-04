"""Native eager CPU embedding options and explicit dense-gradient contracts."""
import operator

from .. import _native
from .._native import Tensor


def parameter_options(num_embeddings, embedding_dim, weight, device, dtype):
    from .._tensor_api import _device, _dtype, _shape
    check_mutation_mode()
    if weight is not None:
        if not isinstance(weight, Tensor) or weight.dtype not in ('float32', 'float64') or weight.device != 'cpu':
            raise ValueError('_weight must be a CPU float32 or float64 Tensor')
        dtype = weight.dtype
    else:
        dtype = _dtype(dtype)
        if dtype not in ('float32', 'float64') or _device(device) != 'cpu':
            raise NotImplementedError('Embedding parameter factories require CPU float32 or float64')
    shape, _ = _shape((num_embeddings, embedding_dim), dtype)
    if weight is not None and weight.shape != shape:
        raise ValueError('_weight must match the embedding shape')
    return shape, dtype


def check_mutation_mode():
    if _native._is_capturing() or _native._forward_ad_is_active():
        raise NotImplementedError('Embedding initialization/reset during capture or forward AD is unsupported')


def padding_index(value, rows):
    if value is None:
        return None
    value = operator.index(value)
    if value < 0:
        value += rows
    if not 0 <= value < rows:
        raise ValueError('padding_idx must be within num_embeddings')
    return value


def options(padding_idx, rows, max_norm, norm_type, scale_grad_by_freq, sparse):
    if type(scale_grad_by_freq) is not bool or type(sparse) is not bool:
        raise TypeError('scale_grad_by_freq and sparse must be bool')
    if sparse:
        raise NotImplementedError('embedding sparse gradients require sparse gradient storage')
    return padding_index(padding_idx, rows), None if max_norm is None else float(max_norm), float(norm_type)


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError('embedding input and weight must be Tensors')
    if weight.ndim != 2:
        raise ValueError('embedding weight must have shape [num_embeddings, embedding_dim]')
    padding_idx, max_norm, norm_type = options(padding_idx, weight.shape[0], max_norm, norm_type, scale_grad_by_freq, sparse)
    if weight.device != 'cpu':
        if padding_idx is not None or max_norm is not None or scale_grad_by_freq:
            raise NotImplementedError('embedding options currently require CPU tensors')
        flat = input.reshape([input.numel()])
        return _native.embedding(weight, flat).reshape(input.shape + [weight.shape[1]])
    return _native._embedding_with_options(weight, input, padding_idx, max_norm, norm_type, scale_grad_by_freq)
