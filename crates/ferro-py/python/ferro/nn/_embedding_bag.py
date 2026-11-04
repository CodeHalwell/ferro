"""CPU dense embedding bags with native typed reductions and adjoints."""
from .. import _native
from .._native import Tensor
from .._tensor_api import zeros
from ._embedding import options, parameter_options
from .layers import Embedding
from .module import Parameter


def _mode(mode, scale_grad_by_freq, include_last_offset):
    if mode not in ('sum', 'mean', 'max'):
        raise ValueError('mode must be sum, mean or max')
    if type(include_last_offset) is not bool:
        raise TypeError('include_last_offset must be bool')
    if mode == 'max' and scale_grad_by_freq:
        raise NotImplementedError('max mode does not support scale_grad_by_freq')


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2.,
                  scale_grad_by_freq=False, mode='mean', sparse=False,
                  per_sample_weights=None, include_last_offset=False, padding_idx=None):
    """Dense CPU float32/float64 bags, with first-order eager gradients.

    Offsets must start at zero and be monotone; a supplied final offset must
    equal the input length. Empty bags yield zero. Per-sample weights are live
    and sum-only. Max ties select the first non-padding occurrence. Frequency
    scaling follows the current Torch CPU sorted-ordinal count rule. Sparse
    gradients, capture and differentiable higher-order queries are unsupported.
    Renorm requires a whole-contiguous weight destination, as in embedding.
    """
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError('embedding_bag input and weight must be Tensors')
    if weight.ndim != 2:
        raise ValueError('embedding_bag weight must have shape [num_embeddings, embedding_dim]')
    if offsets is not None and not isinstance(offsets, Tensor):
        raise TypeError('offsets must be a Tensor or None')
    if per_sample_weights is not None and not isinstance(per_sample_weights, Tensor):
        raise TypeError('per_sample_weights must be a Tensor or None')
    padding_idx, max_norm, norm_type = options(padding_idx, weight.shape[0], max_norm, norm_type, scale_grad_by_freq, sparse)
    _mode(mode, scale_grad_by_freq, include_last_offset)
    return _native._embedding_bag(weight, input, offsets, mode, per_sample_weights,
                                  include_last_offset, padding_idx, max_norm, norm_type, scale_grad_by_freq)


class EmbeddingBag(Embedding):
    """CPU float32/float64 parameter table with native sum, mean and max bags.

    Pretrained weights retain their storage with an independent parameter leaf.
    Options and parameter flags participate in complete training checkpoints;
    sparse gradients reject. Supplied weights determine dtype and device.
    """
    _builtin_init = True

    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None,
                 include_last_offset=False, padding_idx=None, device=None, dtype=None):
        shape, dtype = parameter_options(num_embeddings, embedding_dim, _weight, device, dtype)
        padding_idx, max_norm, norm_type = options(padding_idx, shape[0], max_norm, norm_type, scale_grad_by_freq, sparse)
        _mode(mode, scale_grad_by_freq, include_last_offset)
        self._initialize()
        self.num_embeddings, self.embedding_dim = shape
        self.padding_idx, self.max_norm, self.norm_type = padding_idx, max_norm, norm_type
        self.scale_grad_by_freq, self.mode, self.sparse = scale_grad_by_freq, mode, sparse
        self.include_last_offset = include_last_offset
        initial = (Tensor.zeros(shape) if dtype == 'float32' else zeros(shape, dtype=dtype)) if _weight is None else _weight
        self.weight = Parameter._from_shared(initial, requires_grad=True)
        if _weight is None:
            self.reset_parameters()

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, max_norm=None, norm_type=2.,
                        scale_grad_by_freq=False, mode='mean', sparse=False,
                        include_last_offset=False, padding_idx=None):
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise ValueError('from_pretrained expects a two-dimensional Tensor')
        if type(freeze) is not bool:
            raise TypeError('freeze must be bool')
        module = cls(*embeddings.shape, max_norm, norm_type, scale_grad_by_freq, mode,
                     sparse, embeddings, include_last_offset, padding_idx)
        module.weight.requires_grad_(not freeze)
        return module

    def forward(self, input, offsets=None, per_sample_weights=None):
        return embedding_bag(input, self.weight.tensor(), offsets, self.max_norm, self.norm_type,
                             self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                             self.include_last_offset, self.padding_idx)
