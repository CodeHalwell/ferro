"""Sparse CPU f32 spectral filters using existing eager tensor operations.

Inputs are rank-two features and native EdgeIndex or CPU int64 COO topology.
No sparse matrix, lazy width, feature cache, capture or higher-order engine is
introduced. Edge weights remain differentiable through normalization.
"""
import math

from . import _native
from ._native import Tensor, no_grad, scatter
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear
from ._graph_extra_conv import _gcn_weights, _weights
from .nn import Dropout, Linear, Module, ModuleList, Parameter, ReLU, Tanh, Sigmoid, GELU
from .nn import functional as F


def _hops(value):
    if type(value) is not int or value < 0:
        raise ValueError('K must be a nonnegative integer')


def _graph(x, edge_index, edge_weight, width, output, loops=False, stacks=1):
    if _native._is_capturing():
        raise NotImplementedError('spectral graph filters require eager execution')
    _features(x, 'x', width)
    nodes = x.shape[0]
    _capacity((nodes + 1,), 8)
    src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
    _weights(edge_weight, len(src))
    _capacity((2, len(src) + (nodes if loops else 0)), 8)
    _capacity((nodes, stacks, max(width, output)))
    _capacity((len(src) + (nodes if loops else 0), stacks, max(width, output)))
    return nodes, src, dst


def _propagate(x, src, dst, nodes, weight):
    message = x.index_select(0, src)
    if weight is not None:
        message = message * weight.reshape([len(src), 1])
    return scatter(message, dst, nodes, 'sum')


class SGConv(Module):
    """K normalized GCN propagations followed by one learned linear map.

    cached=True rejects: no stale or hidden detached feature cache is kept.
    K=0 still validates and normalizes topology, then applies the linear map.
    Weighted duplicate self-loops with loop insertion share the same explicit
    rejection as GCNConv because PyG's assignment derivative is ambiguous.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    K: int
    cached: bool
    add_self_loops: bool
    aggr: str

    def __init__(self, in_channels, out_channels, K=1, cached=False, add_self_loops=True, bias=True, **kwargs):
        _positive(in_channels, 'in_channels')
        _positive(out_channels, 'out_channels')
        _hops(K)
        _flags(cached=cached, add_self_loops=add_self_loops, bias=bias)
        if cached:
            raise NotImplementedError('SGConv supports cached=False only; no feature cache is retained')
        _capacity((in_channels, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, K=K, cached=cached,
                   add_self_loops=add_self_loops, aggr=_options(kwargs, 'add'))
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin)

    def forward(self, x, edge_index, edge_weight=None):
        nodes, src, dst = _graph(x, edge_index, edge_weight, self.in_channels, self.out_channels, self.add_self_loops)
        src, dst, weight = _gcn_weights(src, dst, nodes, edge_weight, self.add_self_loops)
        for _ in range(self.K):
            x = _propagate(x, src, dst, nodes, weight)
        return self.lin(x)


class TAGConv(Module):
    """Sum of K+1 learned maps of normalized adjacency powers, without loops."""
    _builtin_init = True
    in_channels: int
    out_channels: int
    K: int
    normalize: bool
    aggr: str

    def __init__(self, in_channels, out_channels, K=3, bias=True, normalize=True, **kwargs):
        _positive(in_channels, 'in_channels')
        _positive(out_channels, 'out_channels')
        _hops(K)
        _flags(bias=bias, normalize=normalize)
        _capacity((K + 1, in_channels, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, K=K,
                   normalize=normalize, aggr=_options(kwargs, 'add'))
        self.lins = ModuleList(Linear(in_channels, out_channels, bias=False) for _ in range(K + 1))
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.lins:
            _reset_linear(linear)
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, edge_index, edge_weight=None):
        nodes, src, dst = _graph(x, edge_index, edge_weight, self.in_channels, self.out_channels)
        if self.normalize:
            src, dst, edge_weight = _gcn_weights(src, dst, nodes, edge_weight, False)
        output = self.lins[0](x)
        for index in range(1, self.K + 1):
            x = _propagate(x, src, dst, nodes, edge_weight)
            output = output + self.lins[index](x)
        return output if self.bias is None else output + self.bias.tensor()


def _activation(act):
    if act is None or isinstance(act, Module):
        return act
    for name, function, cls in (('relu', F.relu, ReLU), ('tanh', F.tanh, Tanh),
                                ('sigmoid', F.sigmoid, Sigmoid), ('gelu', F.gelu, GELU)):
        if (isinstance(act, str) and act == name) or act is function:
            return cls()
    raise NotImplementedError('ARMAConv activation must be a Ferro Module, relu/tanh/sigmoid/gelu, or None')


class ARMAConv(Module):
    """Parallel recurrent ARMA stacks with root dropout and optional sharing.

    Parameters use PyG's tensor layouts and names. One root dropout mask is
    shared across stacks per iteration, with its native stream checkpointed.
    Dropout draws do not match Torch's global RNG. Supplied activation Modules
    are registered directly; arbitrary Python callables reject.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    num_stacks: int
    num_layers: int
    shared_weights: bool
    dropout: float
    aggr: str

    def __init__(self, in_channels, out_channels, num_stacks=1, num_layers=1,
                 shared_weights=False, act='relu', dropout=0., bias=True, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'),
                            (num_stacks, 'num_stacks'), (num_layers, 'num_layers')):
            _positive(value, name)
        _flags(shared_weights=shared_weights, bias=bias)
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        activation = _activation(act)
        count = 1 if shared_weights else num_layers
        _capacity((max(1, count - 1), num_stacks, out_channels, out_channels))
        _capacity((count + 1, num_stacks, in_channels, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, num_stacks=num_stacks,
                   num_layers=num_layers, shared_weights=shared_weights, dropout=float(dropout), aggr=_options(kwargs, 'add'))
        self.weight = Parameter(Tensor.zeros([max(1, count - 1), num_stacks, out_channels, out_channels]))
        self.init_weight = Parameter(Tensor.zeros([num_stacks, in_channels, out_channels]))
        self.root_weight = Parameter(Tensor.zeros([count, num_stacks, in_channels, out_channels]))
        self.bias = Parameter(Tensor.zeros([count, num_stacks, 1, out_channels])) if bias else None
        self.act = activation
        self.root_dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            for parameter in (self.weight, self.init_weight, self.root_weight):
                value = parameter.tensor()
                bound = math.sqrt(6. / (value.shape[-2] + value.shape[-1]))
                value.copy_((Tensor.rand(list(value.shape)) * 2 - 1) * bound)
            if self.bias is not None:
                self.bias.tensor().copy_(Tensor.zeros(list(self.bias.tensor().shape)))

    def forward(self, x, edge_index, edge_weight=None):
        nodes, src, dst = _graph(x, edge_index, edge_weight, self.in_channels, self.out_channels, stacks=self.num_stacks)
        src, dst, weight = _gcn_weights(src, dst, nodes, edge_weight, False)
        initial = x.unsqueeze(0)
        output = initial.expand(self.num_stacks, nodes, self.in_channels)
        for iteration in range(self.num_layers):
            index = 0 if self.shared_weights else iteration
            matrix = self.init_weight.tensor() if iteration == 0 else self.weight.tensor()[0 if self.shared_weights else iteration - 1]
            output = output.bmm(matrix)
            messages = output.permute(1, 0, 2).reshape([nodes, self.num_stacks * self.out_channels])
            output = _propagate(messages, src, dst, nodes, weight).reshape([nodes, self.num_stacks, self.out_channels]).permute(1, 0, 2)
            root = self.root_dropout(initial).expand(self.num_stacks, nodes, self.in_channels)
            output = output + root.bmm(self.root_weight.tensor()[index])
            if self.bias is not None:
                output = output + self.bias.tensor()[index]
            if self.act is not None:
                output = self.act(output)
        return output.mean_dim(0, False)
