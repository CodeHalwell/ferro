"""Canonical dense CPU f32 graph layers with live adjacency gradients.

Rank-two inputs acquire a batch dimension. Dense adjacency uses row-target,
column-source multiplication except the documented PyG DenseGraphConv max
path. This module introduces no sparse conversion or custom gradient engine.
"""
import math

from ._native import Tensor, no_grad
from ._graph_utils import _capacity, _cpu_tensor
from ._graph_conv import _configure, _flags, _positive, _reset_linear, _reset_network
from ._graph_conditioned import _eager, _network
from .nn import Linear, Module, Parameter


def _broadcast(left, right):
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    raise ValueError('dense graph batch dimensions are not broadcastable')


def _prepare(x, adj, mask, width=None, graph_mask=False, max_mode=False, output_width=None):
    _eager()
    _cpu_tensor(x, 'x')
    _cpu_tensor(adj, 'adj')
    if x.ndim not in (2, 3) or adj.ndim not in (2, 3):
        raise ValueError('dense graph features and adjacency must have rank two or three')
    x = x.unsqueeze(0) if x.ndim == 2 else x
    adj = adj.unsqueeze(0) if adj.ndim == 2 else adj
    xb, nodes, features = x.shape
    ab, rows, columns = adj.shape
    if rows != nodes or columns != nodes:
        raise ValueError('adj must be square and match the feature node count')
    if width is not None and features != width:
        raise ValueError('feature width must match in_channels')
    batch = _broadcast(xb, ab)
    if max_mode and batch != xb:
        raise ValueError('DenseGraphConv max adjacency must broadcast to the feature batch')
    _capacity((batch, nodes, nodes))
    _capacity((batch, nodes, features))
    prepared_mask = None
    if mask is not None:
        _cpu_tensor(mask, 'mask', dtype=None)
        if mask.dtype not in ('bool', 'float32'):
            raise ValueError('dense graph masks require bool or float32 storage')
        if graph_mask:
            if not nodes or mask.numel() % nodes:
                raise ValueError('DenseGraphConv mask must reshape to [B,N,1] with N>0')
            mb = mask.numel() // nodes
            _broadcast(batch, mb)
        else:
            mb = ab
            if mask.numel() != mb * nodes:
                raise ValueError('mask element count must match adjacency batch times node count')
        _capacity((mb, nodes, 1))
        _capacity((_broadcast(batch, mb), nodes, features if output_width is None else output_width))
        prepared_mask = mask.reshape([mb, nodes, 1])
        if prepared_mask.dtype == 'bool':
            prepared_mask = prepared_mask.to(dtype='float32')
    return x.expand(batch, nodes, features), adj.expand(batch, nodes, nodes), prepared_mask


def _mask(result, mask):
    if mask is None:
        return result
    batch = _broadcast(result.shape[0], mask.shape[0])
    _capacity((batch, result.shape[1], result.shape[2]))
    return result * mask


def _dimensions(in_channels, out_channels, **flags):
    _positive(in_channels, 'in_channels')
    _positive(out_channels, 'out_channels')
    _flags(**flags)
    _capacity((2, in_channels, out_channels))


class DenseGCNConv(Module):
    """Symmetric row-degree normalization with optional diagonal replacement.

    add_loop overwrites every diagonal with 1 (2 for improved=True), so input
    diagonal gradients are zero on that path. Degrees clamp to at least one.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    improved: bool

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        _dimensions(in_channels, out_channels, improved=improved, bias=bias)
        _configure(self, in_channels=in_channels, out_channels=out_channels, improved=improved)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin, glorot=True)
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, adj, mask=None, add_loop=True):
        _flags(add_loop=add_loop)
        x, adj, mask = _prepare(x, adj, mask, self.in_channels, output_width=self.out_channels)
        batch, nodes, _ = adj.shape
        _capacity((batch, nodes, self.out_channels))
        if add_loop:
            _capacity((nodes, nodes), 8)
            diagonal = Tensor.from_bool([i == j for i in range(nodes) for j in range(nodes)], [1, nodes, nodes])
            adj = adj.masked_fill(diagonal.expand(batch, nodes, nodes), 2. if self.improved else 1.)
        inverse = adj.sum_dim(-1, False).clamp(1., float('inf')).pow(-.5)
        normalized = inverse.unsqueeze(-1) * adj * inverse.unsqueeze(-2)
        result = normalized.bmm(self.lin(x))
        if self.bias is not None:
            result = result + self.bias.tensor()
        return _mask(result, mask)


class DenseSAGEConv(Module):
    """Mean neighbor projection plus a separately biased root projection.

    Binary adjacency is the intended GraphSAGE input; floating adjacency is
    evaluated by the same weighted sum/degree formula as PyG.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    normalize: bool

    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        _dimensions(in_channels, out_channels, normalize=normalize, bias=bias)
        _configure(self, in_channels=in_channels, out_channels=out_channels, normalize=normalize)
        self.lin_rel = Linear(in_channels, out_channels, bias=False)
        self.lin_root = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin_rel)
        _reset_linear(self.lin_root)

    def forward(self, x, adj, mask=None):
        x, adj, mask = _prepare(x, adj, mask, self.in_channels, output_width=self.out_channels)
        _capacity((x.shape[0], x.shape[1], self.out_channels))
        neighbors = adj.bmm(x) / adj.sum_dim(-1, True).clamp(1., float('inf'))
        result = self.lin_rel(neighbors) + self.lin_root(x)
        if self.normalize:
            result = result.normalize(-1, 1e-12)
        return _mask(result, mask)


class DenseGINConv(Module):
    """A registered network applied to adjacency sum plus optional epsilon root."""
    _builtin_init = True
    initial_eps: float
    train_eps: bool

    def __init__(self, nn, eps=0., train_eps=False):
        _network(nn)
        _flags(train_eps=train_eps)
        if type(eps) not in (int, float) or not math.isfinite(eps):
            raise ValueError('eps must be finite')
        _configure(self, initial_eps=float(eps), train_eps=train_eps)
        self.nn = nn
        if train_eps:
            self.eps = Parameter(Tensor([float(eps)], [1]))
        else:
            self.register_buffer('eps', Tensor([float(eps)], [1]))
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.nn)
        with no_grad():
            (self.eps.tensor() if self.train_eps else self.eps).copy_(Tensor([self.initial_eps], [1]))

    def forward(self, x, adj, mask=None, add_loop=True):
        _flags(add_loop=add_loop)
        x, adj, mask = _prepare(x, adj, mask)
        result = adj.bmm(x)
        if add_loop:
            eps = self.eps.tensor() if self.train_eps else self.eps
            result = result + (1. + eps) * x
        result = self.nn(result)
        _cpu_tensor(result, 'nn output')
        if result.ndim != 3 or result.shape[:2] != x.shape[:2]:
            raise ValueError('nn must preserve batch and node dimensions')
        return _mask(result, mask)


class DenseGraphConv(Module):
    """Dense add/mean/max graph messages with a learned root map.

    Max matches PyG's distinct source-row/target-column adjacency orientation,
    uses nonzero entries as presence, and routes ties to the first source.
    Add/mean use row-target adjacency multiplication. Max adjacency receives
    no gradient; an empty node axis rejects because max has no identity.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    aggr: str

    def __init__(self, in_channels, out_channels, aggr='add', bias=True):
        _dimensions(in_channels, out_channels, bias=bias)
        if aggr not in ('add', 'mean', 'max'):
            raise ValueError('aggr must be add, mean or max')
        _configure(self, in_channels=in_channels, out_channels=out_channels, aggr=aggr)
        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin_rel)
        _reset_linear(self.lin_root)

    def forward(self, x, adj, mask=None):
        x, adj, mask = _prepare(x, adj, mask, self.in_channels, True, self.aggr == 'max', self.out_channels)
        batch, nodes, width = x.shape
        _capacity((batch, nodes, self.out_channels))
        if self.aggr == 'max':
            if not nodes:
                raise ValueError('DenseGraphConv max requires at least one node')
            _capacity((batch, nodes, nodes, width))
            messages = x.unsqueeze(2).expand(batch, nodes, nodes, width)
            excluded = adj.unsqueeze(-1).expand(batch, nodes, nodes, width).eq(0.)
            messages = messages.masked_fill(excluded, -float('inf'))
            winner = messages.argmax(1, True)
            result = messages.gather(1, winner).squeeze(1)
            result = result.masked_fill(result.eq(-float('inf')), 0.)
        else:
            result = adj.bmm(x)
            if self.aggr == 'mean':
                result = result / adj.sum_dim(-1, True).clamp(1., float('inf'))
        return _mask(self.lin_rel(result) + self.lin_root(x), mask)

