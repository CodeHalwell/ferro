"""Native dense spectral graph pooling and its differentiable objectives."""
import math

from ._native import Tensor
from ._tensor_api import eye
from ._graph_conditioned import _eager
from ._graph_conv import _configure, _positive, _reset_linear
from ._graph_utils import _capacity, _cpu_tensor
from .linalg import vector_norm
from .nn import Dropout, Linear, Module, ModuleList


def _inputs(x, adj, mask, clusters, width=None):
    _eager()
    values = []
    for value, name in ((x, 'x'), (adj, 'adj')):
        _cpu_tensor(value, name)
        if len(value.shape) not in (2, 3): raise ValueError('dense pooling inputs require rank two or three')
        values.append(value.unsqueeze(0) if len(value.shape) == 2 else value)
    x, adj = values
    batch, nodes, features = x.shape
    if not batch or not nodes or adj.shape != [batch, nodes, nodes] or (width is not None and features != width):
        raise ValueError('dense pooling requires matching nonempty batch/node axes and square adjacency')
    for shape in ((batch, nodes, nodes), (batch, nodes, features), (batch, nodes, clusters),
                  (batch, clusters, nodes), (batch, clusters, clusters), (batch, clusters, features)):
        _capacity(shape)
    _capacity((clusters, clusters), 8)
    if mask is not None:
        _cpu_tensor(mask, 'mask', dtype=None)
        if mask.dtype not in ('bool', 'float32') or mask.shape not in ([batch, nodes], [nodes] if batch == 1 else [batch, nodes]):
            raise ValueError('dense pooling mask requires bool/float32 [B,N], or [N] for one graph')
        mask = mask.reshape([batch, nodes, 1])
        if mask.dtype == 'bool': mask = mask.to(dtype='float32')
    return x, adj, mask


def _orthogonality(assignment, identity):
    gram = assignment.transpose(1, 2).bmm(assignment)
    normalized = gram / vector_norm(gram, dim=(1, 2), keepdim=True)
    return vector_norm(normalized - identity / math.sqrt(identity.shape[0]), dim=(1, 2)).mean()


def _normalized_adjacency(adjacency, identity):
    adjacency = adjacency.masked_fill(identity.ne(0.).unsqueeze(0), 0.)
    degree = adjacency.sum_dim(2, False).sqrt().unsqueeze(1) + 1e-15
    return (adjacency / degree) / degree.transpose(1, 2)


def dense_mincut_pool(x, adj, s, mask=None, temp=1.):
    """MinCut pooling, normalized coarsened adjacency and both auxiliary losses.

    CPU float32 rank-two inputs gain one batch axis; otherwise explicit batches
    must match. Positive finite temperatures and positive node/cluster counts
    are required. Bool or float32 masks retain the actual PyG multiplication
    semantics, including gradients for float masks. Singular zero-degree or
    zero-assignment objectives retain PyG's nonfinite results.
    """
    if type(temp) not in (int, float) or not math.isfinite(temp) or temp <= 0:
        raise ValueError('temp must be positive and finite')
    _cpu_tensor(s, 's')
    if len(s.shape) not in (2, 3): raise ValueError('assignments require rank two or three')
    s = s.unsqueeze(0) if len(s.shape) == 2 else s
    _positive(s.shape[2], 'cluster count')
    x, adj, mask = _inputs(x, adj, mask, s.shape[2])
    if s.shape[:2] != x.shape[:2]: raise ValueError('assignment batch/node axes must match features')
    assignment = (s if temp == 1. else s / temp).softmax(2)
    if mask is not None: x, assignment = x * mask, assignment * mask
    transposed = assignment.transpose(1, 2)
    output = transposed.bmm(x)
    adjacency = transposed.bmm(adj).bmm(assignment)
    numerator = adjacency.diagonal(dim1=1, dim2=2).sum_dim(1, False)
    # D is diagonal, so its product with S is a native rowwise multiply.
    degree_assignment = adj.sum_dim(2, False).unsqueeze(2) * assignment
    denominator = transposed.bmm(degree_assignment).diagonal(dim1=1, dim2=2).sum_dim(1, False)
    mincut = -(numerator / denominator).mean()
    identity = eye(s.shape[2])
    return output, _normalized_adjacency(adjacency, identity), mincut, _orthogonality(assignment, identity)


class _DMoNMLP(Module):
    _builtin_init = True
    channels: tuple

    def __init__(self, channels):
        _configure(self, channels=channels)
        self.lins = ModuleList([Linear(left, right) for left, right in zip(channels, channels[1:])])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lins: _reset_linear(layer)

    def forward(self, x):
        for layer in self.lins: x = layer(x)
        return x


class DMoNPooling(Module):
    """Native affine assignment network, modularity, orthogonality and size loss.

    Uses the current PyG no-activation/no-normalization MLP and dropout before
    assignment softmax. Float masks remain differentiable. The cluster loss
    preserves PyG's current [B] / [B,1] broadcast across graph counts. Invalid
    metadata rejects before assignment dropout or parameter work; singular
    graph objectives keep their nonfinite results. Inputs are CPU float32 with
    matching nonempty batch/node axes; capture and implicit batch broadcast
    are unsupported. Checkpoints include the registered assignment stream.
    """
    _builtin_init = True
    channels: tuple
    k: int
    dropout: float

    def __init__(self, channels, k, dropout=0.):
        channels = (channels,) if type(channels) is int else tuple(channels) if isinstance(channels, (list, tuple)) else ()
        if not channels: raise ValueError('channels must be a positive width or nonempty sequence')
        for width in channels: _positive(width, 'channel width')
        _positive(k, 'k')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        for left, right in zip(channels, channels[1:] + (k,)): _capacity((left, right))
        _capacity((k, k), 8)
        _configure(self, channels=channels, k=k, dropout=float(dropout))
        self.mlp = _DMoNMLP(channels + (k,))
        self.assignment_dropout = Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x, adj, mask=None):
        x, adj, mask = _inputs(x, adj, mask, self.k, self.channels[0])
        batch, nodes, _ = x.shape
        _capacity((batch, batch))
        for width in self.channels[1:]: _capacity((batch, nodes, width))
        if mask is None: mask = Tensor.ones([batch, nodes, 1])
        assignment = self.assignment_dropout(self.mlp(x)).softmax(2)
        x, assignment = x * mask, assignment * mask
        transposed = assignment.transpose(1, 2)
        output = transposed.bmm(x).selu()
        adjacency = transposed.bmm(adj).bmm(assignment)
        degrees = adj.sum_dim(2, False).unsqueeze(2) * mask
        mass = degrees.sum_dim(2, False).sum_dim(1, False) / 2.
        normalizer = transposed.bmm(degrees).bmm(degrees.transpose(1, 2).bmm(assignment)) / 2. / mass.reshape([batch, 1, 1])
        spectral = (-(adjacency - normalizer).diagonal(dim1=1, dim2=2).sum_dim(1, False) / 2. / mass).mean()
        identity = eye(self.k)
        orthogonality = _orthogonality(assignment, identity)
        sizes = vector_norm(assignment.sum_dim(1, False), dim=1)
        cluster = (sizes / mask.sum_dim(1, False) * math.sqrt(self.k) - 1.).mean()
        return assignment, output, _normalized_adjacency(adjacency, identity), spectral, orthogonality, cluster
