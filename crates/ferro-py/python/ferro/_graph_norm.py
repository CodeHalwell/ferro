"""Graph normalization over dense CPU f32 node features and int64 graph IDs.

These modules compose native tensor and segment operations. They retain no
topology, running statistics or graph cache. Arbitrary graph ID order and empty
groups are supported; differentiable input values never leave native tensors.
"""
import math

from . import _native
from ._native import Tensor, no_grad, scatter
from ._graph_conv import _configure, _features, _flags, _positive
from ._graph_utils import _capacity, _count, _index_ids
from .nn import Module, Parameter
from .nn import functional as F


def _epsilon(eps):
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    return float(eps)


def _device(device):
    if device not in (None, 'cpu'):
        raise NotImplementedError('graph normalization supports CPU only')


def _batch(x, batch, batch_size, width=None, require_empty_size=False):
    if _native._is_capturing():
        raise NotImplementedError('graph normalization requires eager execution')
    _features(x, 'x', width)
    _capacity((x.shape[0] + 1,), 8)
    if batch_size is not None:
        _count(batch_size, 'batch_size')
    if batch is None:
        ids, count = [0] * x.shape[0], 1
    else:
        ids = _index_ids(batch, 'batch', x.shape[0])
        if require_empty_size and not ids and batch_size is None:
            raise ValueError('empty batch requires an explicit batch_size')
        count = max(ids, default=-1) + 1 if batch_size is None else batch_size
        if any(value >= count for value in ids):
            raise ValueError('batch contains a graph ID outside batch_size')
    _capacity((count, x.shape[1]))
    _capacity((count + 1,), 8)
    return ids, count


def _center(x, ids, count):
    return x - scatter(x, ids, count, 'mean').index_select(0, ids)


def _sqrt_with_zero_subgradient(square):
    # Torch vector norm/std chooses zero at exact zero. Avoid sqrt'(0) while
    # preserving the live derivative at every positive input.
    zero = square.eq(0.)
    return square.masked_fill(zero, 1.).sqrt().masked_fill(zero, 0.)


def _reset_affine(module):
    with no_grad():
        if module.weight is not None:
            module.weight.tensor().copy_(Tensor.ones([module.in_channels]))
        if module.bias is not None:
            module.bias.tensor().copy_(Tensor.zeros([module.in_channels]))


class GraphNorm(Module):
    """PyG GraphNorm with learned channelwise mean scale, weight and bias."""
    _builtin_init = True
    in_channels: int
    eps: float

    def __init__(self, in_channels, eps=1e-5, device=None):
        _device(device)
        _positive(in_channels, 'in_channels')
        _capacity((3, in_channels))
        _configure(self, in_channels=in_channels, eps=_epsilon(eps))
        self.weight = Parameter(Tensor.ones([in_channels]))
        self.bias = Parameter(Tensor.zeros([in_channels]))
        self.mean_scale = Parameter(Tensor.ones([in_channels]))

    def reset_parameters(self):
        _reset_affine(self)
        with no_grad():
            self.mean_scale.tensor().copy_(Tensor.ones([self.in_channels]))

    def forward(self, x, batch=None, batch_size=None):
        ids, count = _batch(x, batch, batch_size, self.in_channels, True)
        mean = scatter(x, ids, count, 'mean')
        out = x - mean.index_select(0, ids) * self.mean_scale.tensor()
        variance = scatter(out.square(), ids, count, 'mean')
        std = (variance + self.eps).sqrt().index_select(0, ids)
        return self.weight.tensor() * out / std + self.bias.tensor()


class PairNorm(Module):
    """Centered node features with graphwise or individual L2 scaling.

    Graphwise scaling uses sqrt(mean squared norm + eps); individual scaling
    uses norm + eps, as in PyG. Exact zero vector norms use zero subgradients.
    """
    _builtin_init = True
    scale: float
    scale_individually: bool
    eps: float

    def __init__(self, scale=1., scale_individually=False, eps=1e-5):
        if type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError('scale must be finite')
        _flags(scale_individually=scale_individually)
        _configure(self, scale=float(scale), scale_individually=scale_individually, eps=_epsilon(eps))

    def forward(self, x, batch=None, batch_size=None):
        ids, count = _batch(x, batch, batch_size)
        centered = _center(x, ids, count)
        squared = centered.square().sum_dim(-1, True)
        if self.scale_individually:
            denominator = _sqrt_with_zero_subgradient(squared) + self.eps
        else:
            denominator = (scatter(squared, ids, count, 'mean') + self.eps).sqrt().index_select(0, ids)
        return self.scale * centered / denominator


class MeanSubtractionNorm(Module):
    """Subtract each graph's channelwise node-feature mean."""
    def forward(self, x, batch=None, dim_size=None):
        ids, count = _batch(x, batch, dim_size)
        return _center(x, ids, count)


class GraphSizeNorm(Module):
    """Scale each node by the inverse square root of its graph's node count."""
    def forward(self, x, batch=None, batch_size=None):
        ids, count = _batch(x, batch, batch_size)
        degrees = [0] * count
        for index in ids:
            degrees[index] += 1
        # Every selected graph has nonzero degree; unused groups need no divide.
        scale = Tensor([degrees[index] ** -.5 for index in ids], [x.shape[0], 1])
        return x * scale


class LayerNorm(Module):
    """PyG graph or per-node LayerNorm, distinct from ferro.nn.LayerNorm.

    Graph mode with batch=None uses std + eps. An explicit batch uses
    sqrt(variance + eps), matching PyG's observable epsilon placement. Node
    mode uses ordinary last-feature normalization and ignores graph IDs.
    """
    _builtin_init = True
    in_channels: int
    eps: float
    affine: bool
    mode: str

    def __init__(self, in_channels, eps=1e-5, affine=True, mode='graph', device=None):
        _device(device)
        _positive(in_channels, 'in_channels')
        _flags(affine=affine)
        if mode not in ('graph', 'node'):
            raise ValueError('mode must be graph or node')
        _capacity((2, in_channels))
        _configure(self, in_channels=in_channels, eps=_epsilon(eps), affine=affine, mode=mode)
        self.weight = Parameter(Tensor.ones([in_channels])) if affine else None
        self.bias = Parameter(Tensor.zeros([in_channels])) if affine else None

    def reset_parameters(self):
        _reset_affine(self)

    def forward(self, x, batch=None, batch_size=None):
        if self.mode == 'node':
            _batch(x, None, None, self.in_channels)
            return F.layer_norm(x, self.in_channels, None if self.weight is None else self.weight.tensor(),
                                None if self.bias is None else self.bias.tensor(), self.eps)
        ids, count = _batch(x, batch, batch_size, self.in_channels, True)
        if batch is None:
            centered = x - x.mean()
            denominator = _sqrt_with_zero_subgradient(centered.square().mean()) + self.eps
        else:
            degrees = [0] * count
            for index in ids:
                degrees[index] += 1
            denominator_count = Tensor([float(max(1, n) * self.in_channels) for n in degrees], [count, 1])
            mean = scatter(x, ids, count, 'sum').sum_dim(-1, True) / denominator_count
            centered = x - mean.index_select(0, ids)
            variance = scatter(centered.square(), ids, count, 'sum').sum_dim(-1, True) / denominator_count
            denominator = (variance + self.eps).sqrt().index_select(0, ids)
        result = centered / denominator
        return result if self.weight is None else result * self.weight.tensor() + self.bias.tensor()
