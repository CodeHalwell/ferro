"""Gaussian and learned edge kernels over native CPU f32 graph operations."""
import math

from ._native import Tensor, no_grad, scatter
from ._graph_utils import _capacity, _edge_parts
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear
from ._graph_conditioned import _eager, _reduction
from ._graph_extra_conv import _gcn_weights
from ._graph_residual import _widths
from ._graph_spectral import _propagate
from .nn import Linear, Module, Parameter, ReLU, Sequential, Sigmoid


class GMMConv(Module):
    """Gaussian pseudocoordinate kernels, shared or separate per channel pair.

    Edge attributes are required CPU f32 [E, dim] pseudocoordinates. Input
    features may be bipartite with missing targets and an explicit size.
    root_weight=False installs an explicit None root; this avoids PyG 2.8.1's
    missing-root attribute error and implements the stated no-root operation.
    """
    _builtin_init = True
    in_channels: object
    out_channels: int
    dim: int
    kernel_size: int
    separate_gaussians: bool
    root_weight: bool
    rel_in_channels: int
    aggr: str

    def __init__(self, in_channels, out_channels, dim, kernel_size, separate_gaussians=False,
                 aggr='mean', root_weight=True, bias=True, **kwargs):
        in_channels, widths = _widths(in_channels)
        for value, name in ((out_channels, 'out_channels'), (dim, 'dim'), (kernel_size, 'kernel_size')):
            _positive(value, name)
        _flags(separate_gaussians=separate_gaussians, root_weight=root_weight, bias=bias)
        shape = [widths[0], out_channels, kernel_size, dim] if separate_gaussians else [kernel_size, dim]
        _capacity(shape)
        _capacity((max(widths), out_channels, kernel_size))
        _configure(self, in_channels=in_channels, out_channels=out_channels, dim=dim,
                   kernel_size=kernel_size, separate_gaussians=separate_gaussians,
                   root_weight=root_weight, rel_in_channels=widths[0], aggr=_reduction(aggr, kwargs))
        self._widths = widths
        self.g = Parameter(Tensor.zeros([widths[0], out_channels * kernel_size]))
        self.mu = Parameter(Tensor.zeros(shape))
        self.sigma = Parameter(Tensor.zeros(shape))
        self.root = Linear(widths[1], out_channels, bias=False) if root_weight else None
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            for parameter in (self.g, self.mu, self.sigma):
                shape = list(parameter.tensor().shape)
                bound = math.sqrt(6. / (shape[-2] + shape[-1]))
                parameter.tensor().copy_((Tensor.rand(shape) * 2 - 1) * bound)
            if self.bias is not None:
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))
        if self.root is not None:
            _reset_linear(self.root, glorot=True)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        _eager()
        source, target, size = _pair(x, size, *self._widths)
        src, dst, _ = _edge_parts(edge_index, size=size)
        _features(edge_attr, 'edge_attr', self.dim)
        edges, kernels, output_width = len(src), self.kernel_size, self.out_channels
        if edge_attr.shape[0] != edges:
            raise ValueError('edge_attr must contain one pseudocoordinate row per edge')
        for count in size:
            _capacity((count + 1,), 8)
        _capacity((size[1], output_width))
        if self.separate_gaussians:
            _capacity((edges, self.rel_in_channels, output_width, kernels, self.dim))
            coordinates = edge_attr.reshape([edges, 1, 1, 1, self.dim])
            mu = self.mu.tensor().unsqueeze(0)
            sigma = self.sigma.tensor().unsqueeze(0)
            gaussian = (((coordinates - mu).square() * -.5) / (sigma.square() + 1e-15)).sum_dim(4, False).exp()
            gaussian = (gaussian * self.g.tensor().reshape([1, self.rel_in_channels, output_width, kernels])).sum_dim(3, False)
            message = (source.index_select(0, src).unsqueeze(2) * gaussian).sum_dim(1, False)
        else:
            _capacity((size[0], kernels, output_width))
            _capacity((edges, kernels, max(self.dim, output_width)))
            coordinates = edge_attr.reshape([edges, 1, self.dim])
            mu, sigma = self.mu.tensor().unsqueeze(0), self.sigma.tensor().unsqueeze(0)
            gaussian = (((coordinates - mu).square() * -.5) / (sigma.square() + 1e-15)).sum_dim(2, False).exp()
            projected = source.matmul(self.g.tensor()).index_select(0, src).reshape([edges, kernels, output_width])
            message = (projected * gaussian.unsqueeze(2)).sum_dim(1, False)
        output = scatter(message, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        if target is not None and self.root is not None:
            output = output + self.root(target)
        return output if self.bias is None else output + self.bias.tensor()


class PDNConv(Module):
    """Sigmoid edge-network weights followed by optional GCN normalization.

    Without attributes, normalize=True uses unweighted GCN propagation and
    leaves the edge network unused. Without normalization, edge attributes
    are required. add_self_loops only applies when normalization is enabled.
    Weighted duplicate diagonals reject with loop insertion, as in GCNConv.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    edge_dim: int
    hidden_channels: int
    add_self_loops: bool
    normalize: bool
    aggr: str

    def __init__(self, in_channels, out_channels, edge_dim, hidden_channels,
                 add_self_loops=True, normalize=True, bias=True, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'),
                            (edge_dim, 'edge_dim'), (hidden_channels, 'hidden_channels')):
            _positive(value, name)
        _flags(add_self_loops=add_self_loops, normalize=normalize, bias=bias)
        _capacity((in_channels, out_channels))
        _capacity((edge_dim + 1, hidden_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim,
                   hidden_channels=hidden_channels, add_self_loops=add_self_loops,
                   normalize=normalize, aggr=_options(kwargs, 'add'))
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.mlp = Sequential(Linear(edge_dim, hidden_channels), ReLU(), Linear(hidden_channels, 1), Sigmoid())
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.lin, self.mlp[0], self.mlp[2]):
            _reset_linear(layer, glorot=True)
            if layer.bias is not None:
                with no_grad():
                    layer.bias.tensor().copy_(Tensor.zeros([layer.out_features]))
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, edge_index, edge_attr=None):
        _eager()
        _features(x, 'x', self.in_channels)
        nodes = x.shape[0]
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        _capacity((nodes + 1,), 8)
        count = len(src) + (nodes if self.normalize and self.add_self_loops else 0)
        _capacity((2, count), 8)
        _capacity((nodes, self.out_channels))
        _capacity((count, self.out_channels))
        _capacity((len(src), self.hidden_channels))
        if edge_attr is not None:
            _features(edge_attr, 'edge_attr', self.edge_dim)
            if edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must contain one row per edge')
        elif not self.normalize:
            raise ValueError('PDNConv requires edge_attr when normalize=False')
        weights = None if edge_attr is None else self.mlp(edge_attr).squeeze(1)
        if self.normalize:
            src, dst, weights = _gcn_weights(src, dst, nodes, weights, self.add_self_loops)
        output = _propagate(self.lin(x), src, dst, nodes, weights)
        return output if self.bias is None else output + self.bias.tensor()
