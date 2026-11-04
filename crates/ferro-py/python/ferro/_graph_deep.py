"""Native message normalization for deeper graph compositions."""
import math

from ._native import Tensor, no_grad
from ._graph_utils import _capacity
from ._graph_conv import _configure, _features, _flags
from ._graph_conditioned import _eager
from .linalg import vector_norm
from .nn import Module, Parameter


class MessageNorm(Module):
    """Normalize messages and rescale by the original feature norm.

    Inputs are CPU f32 rank-two tensors with equal row counts; their feature
    widths may differ. Fixed scale remains a registered frozen Parameter and
    is read detached, while learned scale participates in differentiation.
    Changing scale.trainable switches that explicit read behavior as well.
    """
    _builtin_init = True
    learn_scale: bool

    def __init__(self, learn_scale=False, device=None):
        _flags(learn_scale=learn_scale)
        if device not in (None, 'cpu'):
            raise NotImplementedError('MessageNorm supports CPU initialization only')
        _configure(self, learn_scale=learn_scale)
        self.scale = Parameter(Tensor.ones([1]))
        self.scale.trainable = learn_scale

    def reset_parameters(self):
        with no_grad():
            self.scale.tensor().copy_(Tensor.ones([1]))

    def forward(self, x, msg, p=2.):
        _eager()
        _features(x, 'x')
        _features(msg, 'msg')
        if x.shape[0] != msg.shape[0]:
            raise ValueError('x and msg must have the same number of rows')
        if type(p) not in (int, float) or math.isnan(p):
            raise ValueError('p must be a real number other than NaN')
        _capacity(x.shape)
        _capacity(msg.shape)
        denominator = vector_norm(msg, ord=p, dim=1, keepdim=True).clamp(1e-12, float('inf'))
        source_norm = vector_norm(x, ord=p, dim=1, keepdim=True)
        scale = self.scale.tensor()
        if not self.scale.trainable:
            scale = scale.detach()
        return (msg / denominator) * source_norm * scale
