"""PyG-shaped CPU float32 graph convolutions using native first-order ops.

Inputs are rank-two features and native EdgeIndex or CPU int64 [2,E] topology.
Sparse matrices, lazy widths, cached normalization and general MessagePassing
options are unsupported. Weighted duplicate self-loops with GCN loop insertion
reject: PyG's duplicate assignment backward differs from last-writer calculus.
Max gradients share ties among actual neighbors; PyG's CPU zero-initialized
scatter destination can count as an extra backward tie at a zero maximum.
"""
import math

from ._native import Tensor, cat, no_grad, scatter
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts
from .nn import Linear, Module, Parameter


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer; lazy channels are unsupported')
    return value


def _flags(**values):
    for name, value in values.items():
        if type(value) is not bool:
            raise TypeError(f'{name} must be a bool')


def _options(kwargs, aggregation):
    options = dict(kwargs)
    if options.pop('flow', 'source_to_target') != 'source_to_target':
        raise NotImplementedError('graph convolutions support source_to_target flow only')
    if options.pop('node_dim', -2) not in (-2, 0):
        raise NotImplementedError('graph convolutions support rank-two features with node_dim=0/-2 only')
    aggr = options.pop('aggr', aggregation)
    if aggr not in ('add', 'sum'):
        raise NotImplementedError('this graph convolution supports add/sum aggregation only')
    if options:
        raise NotImplementedError('unsupported MessagePassing options: ' + ', '.join(sorted(options)))
    return aggr


def _configure(module, **values):
    module._initialize()
    for name, value in values.items():
        object.__setattr__(module, name, value)


def _features(value, name, width=None):
    _cpu_tensor(value, name)
    if len(value.shape) != 2 or (width is not None and value.shape[1] != width):
        raise ValueError(f'{name} must have shape [N, {width if width is not None else "F"}]')
    return value


def _pair(x, size, source_width=None, target_width=None):
    if isinstance(x, Tensor):
        source, target = x, x
    elif isinstance(x, (tuple, list)) and len(x) == 2:
        source, target = x
    else:
        raise TypeError('x must be a Tensor or (source, target) pair')
    _features(source, 'source features', source_width)
    if target is not None:
        _features(target, 'target features', target_width)
    inferred = (source.shape[0], source.shape[0] if target is None else target.shape[0])
    if size is None:
        size = inferred
    else:
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError('size must be a pair (num_source_nodes, num_target_nodes)')
        size = tuple(inferred[i] if value is None else value for i, value in enumerate(size))
        if any(type(n) is not int or n < 0 for n in size) or size[0] != inferred[0] or (target is not None and size[1] != inferred[1]):
            raise ValueError('size does not match source/target feature rows')
    return source, target, size


def _reset_linear(layer, glorot=False):
    bound = (6. / (layer.in_features + layer.out_features)) ** .5 if glorot else layer.in_features ** -.5
    with no_grad():
        layer.weight.tensor().copy_((Tensor.rand(list(layer.weight.tensor().shape)) * 2 - 1) * bound)
        if layer.bias is not None:
            layer.bias.tensor().copy_((Tensor.rand([layer.out_features]) * 2 - 1) * layer.in_features ** -.5)


def _reset_network(module):
    reset = getattr(module, 'reset_parameters', None)
    if callable(reset):
        reset()
    elif isinstance(module, Linear):
        _reset_linear(module)
    else:
        for child in module._registered.values():
            if isinstance(child, Module):
                _reset_network(child)


class GCNConv(Module):
    """PyG COO GCN normalization, including differentiable scalar edge weights.

Like PyG's COO path, improved=True changes newly inserted weighted self-loops;
without edge_weight every resulting edge is assigned weight one. cached=True
and weighted duplicate self-loops with loop insertion are explicitly rejected.
"""
    _builtin_init = True
    in_channels: int
    out_channels: int
    improved: bool
    cached: bool
    add_self_loops: bool
    normalize: bool
    aggr: str

    def __init__(self, in_channels, out_channels, improved=False, cached=False, add_self_loops=None, normalize=True, bias=True, **kwargs):
        _positive(in_channels, 'in_channels')
        _positive(out_channels, 'out_channels')
        _flags(improved=improved, cached=cached, normalize=normalize, bias=bias)
        if cached:
            raise NotImplementedError('GCNConv cached normalization is unsupported')
        loops = normalize if add_self_loops is None else add_self_loops
        _flags(add_self_loops=loops)
        if loops and not normalize:
            raise ValueError('GCNConv cannot add self-loops without normalization')
        aggr = _options(kwargs, 'add')
        _configure(self, in_channels=in_channels, out_channels=out_channels, improved=improved,
                   cached=cached, add_self_loops=loops, normalize=normalize, aggr=aggr)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin, glorot=True)
        if self.bias is not None:
            with no_grad():
                self.bias.tensor().copy_(Tensor.zeros([self.out_channels]))

    def forward(self, x, edge_index, edge_weight=None):
        _features(x, 'x', self.in_channels)
        n = x.shape[0]
        src, dst, _ = _edge_parts(edge_index, num_nodes=n)
        if edge_weight is not None:
            _cpu_tensor(edge_weight, 'edge_weight')
            if edge_weight.shape != [len(src)]:
                raise ValueError('edge_weight must have shape [E]')
        weight = edge_weight
        if self.normalize and self.add_self_loops:
            nonloops = [i for i, (s, d) in enumerate(zip(src, dst)) if s != d]
            loops = [i for i, (s, d) in enumerate(zip(src, dst)) if s == d]
            if weight is not None:
                nodes = [src[i] for i in loops]
                if len(nodes) != len(set(nodes)):
                    raise NotImplementedError('weighted duplicate self-loops with GCN loop insertion are unsupported')
                loop_weight = Tensor.ones([n]) * (2. if self.improved else 1.)
                loop_weight = loop_weight.scatter(0, Tensor.from_i64(nodes, [len(nodes)]), weight.index_select(0, loops))
                weight = cat([weight.index_select(0, nonloops), loop_weight], 0)
            src = [src[i] for i in nonloops] + list(range(n))
            dst = [dst[i] for i in nonloops] + list(range(n))
        if weight is None:
            weight = Tensor.ones([len(src)])
        if self.normalize:
            degree = scatter(weight, dst, n, 'sum')
            zero = Tensor([float(value == 0.) for value in degree.tolist()], [n])
            inv = degree.pow(-.5).masked_fill(zero, 0.)
            weight = inv.index_select(0, src) * weight * inv.index_select(0, dst)
        projected = self.lin(x)
        messages = projected.index_select(0, src) * weight.reshape([len(src), 1])
        output = scatter(messages, dst, n, 'sum')
        return output if self.bias is None else output + self.bias.tensor()


class SAGEConv(Module):
    """PyG GraphSAGE with plain or registered native aggregation Modules.

Optional source projection/ReLU, root transform and L2 output normalization
    use the same parameter names/layout as PyG. Homogeneous or bipartite CPU
    float32 features are supported. Learned names/lists/Modules use checked
    output widths; LSTM aggregation defaults to source input/hidden width and
    requires sorted destination IDs. Existing plain-reduction checkpoints keep
    their schema. Max/min ties share gradients among
    actual neighbors. Unlike PyG's CPU scatter_reduce, an initialization zero
    never takes a share of the gradient when the actual maximum is zero.
"""
    _builtin_init = True
    in_channels: object
    out_channels: int
    aggr: str
    normalize: bool
    root_weight: bool
    project: bool

    def __init__(self, in_channels, out_channels, aggr='mean', normalize=False, root_weight=True, project=False, bias=True, **kwargs):
        if type(in_channels) is int:
            widths = (in_channels, in_channels)
        elif isinstance(in_channels, (tuple, list)) and len(in_channels) == 2:
            widths = tuple(in_channels)
            in_channels = widths
        else:
            raise ValueError('in_channels must be a positive integer or source/target pair')
        for width in widths:
            _positive(width, 'in_channels')
        _positive(out_channels, 'out_channels')
        _flags(normalize=normalize, root_weight=root_weight, project=project, bias=bias)
        options = kwargs.pop('aggr_kwargs', None)
        if options is not None and type(options) is not dict:
            raise TypeError('aggr_kwargs must be a dictionary or None')
        options = {} if options is None else dict(options)
        _options(kwargs, 'add')
        _capacity((max(widths), out_channels))
        if project: _capacity((widths[0], widths[0]))
        aggregate, aggregate_width = None, widths[0]
        plain = isinstance(aggr, str) and aggr in ('mean', 'sum', 'add', 'max', 'min') and not options
        if not plain:
            from ._graph_gen import _aggregation
            if aggr == 'lstm':
                options.setdefault('in_channels', widths[0])
                options.setdefault('out_channels', widths[0])
            aggregate = _aggregation(aggr, options)
            aggregate_width = aggregate.get_out_channels(widths[0])
            _positive(aggregate_width, 'aggregation output width')
        _capacity((aggregate_width, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, aggr=aggr if isinstance(aggr, str) else 'module',
                   normalize=normalize, root_weight=root_weight, project=project)
        self._widths = widths
        if aggregate is not None: self.aggr_module = aggregate
        if project:
            self.lin = Linear(widths[0], widths[0], bias=True)
        self.lin_l = Linear(aggregate_width, out_channels, bias=bias)
        if root_weight:
            self.lin_r = Linear(widths[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'aggr_module'): self.aggr_module.reset_parameters()
        for layer in (getattr(self, 'lin', None), self.lin_l, getattr(self, 'lin_r', None)):
            if layer is not None:
                _reset_linear(layer)

    def forward(self, x, edge_index, size=None):
        source, target, size = _pair(x, size, self._widths[0], self._widths[1] if self.root_weight else None)
        src, dst, _ = _edge_parts(edge_index, size=size)
        for count in size: _capacity((count + 1,), 8)
        _capacity((size[0], self._widths[0]))
        _capacity((len(src), self._widths[0]))
        _capacity((size[1], max(self.lin_l.in_features, self.out_channels)))
        if hasattr(self, 'aggr_module') and self.aggr_module.get_out_channels(self._widths[0]) != self.lin_l.in_features:
            raise ValueError('aggregation output width must match the learned neighbor projection')
        if self.project:
            source = self.lin(source).relu()
        message = source.index_select(0, src)
        aggregate = self.aggr_module(message, dst, dim_size=size[1]) if hasattr(self, 'aggr_module') else scatter(message, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr)
        output = self.lin_l(aggregate)
        if self.root_weight and target is not None:
            output = output + self.lin_r(target)
        return output.normalize(-1, 1e-12) if self.normalize else output


class GINConv(Module):
    """PyG sum-neighbor GIN with a supplied Ferro Module and persistent eps."""
    _builtin_init = True
    initial_eps: float
    train_eps: bool
    aggr: str

    def __init__(self, nn, eps=0., train_eps=False, **kwargs):
        if not isinstance(nn, Module):
            raise TypeError('GINConv nn must be a ferro.nn.Module')
        if type(eps) not in (int, float) or not math.isfinite(eps):
            raise ValueError('eps must be a finite number')
        _flags(train_eps=train_eps)
        aggr = _options(kwargs, 'add')
        _configure(self, initial_eps=float(eps), train_eps=train_eps, aggr=aggr)
        self.nn = nn
        value = Tensor([float(eps)], [1])
        if train_eps:
            self.eps = Parameter(value)
        else:
            self.register_buffer('eps', value)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_network(self.nn)
        value = self.eps.tensor() if self.train_eps else self.eps
        with no_grad():
            value.copy_(Tensor([self.initial_eps], [1]))

    def forward(self, x, edge_index, size=None):
        source, target, size = _pair(x, size)
        if target is not None and source.shape[1] != target.shape[1]:
            raise ValueError('GINConv source/target feature widths must match')
        src, dst, _ = _edge_parts(edge_index, size=size)
        output = scatter(source.index_select(0, src), dst, size[1], 'sum')
        if target is not None:
            eps = self.eps.tensor() if self.train_eps else self.eps
            output = output + (1 + eps) * target
        return self.nn(output)
