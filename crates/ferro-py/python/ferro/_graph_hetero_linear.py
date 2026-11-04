"""Native typed and dictionary linear projections for heterogeneous graphs."""
import math

from ._native import Tensor, cat, no_grad
from ._graph_conditioned import _eager
from ._graph_conv import _configure, _features, _flags, _positive
from ._graph_utils import _capacity, _index_ids
from .nn import Linear, Module, ModuleDict, Parameter


def _initializers(kwargs):
    options = dict(kwargs)
    bias = options.pop('bias', True)
    weight = options.pop('weight_initializer', None)
    offset = options.pop('bias_initializer', None)
    _flags(bias=bias)
    if weight not in (None, 'glorot', 'uniform', 'kaiming_uniform'): raise ValueError('unsupported weight initializer')
    if offset not in (None, 'zeros'): raise ValueError('unsupported bias initializer')
    if options: raise NotImplementedError('unsupported heterogeneous linear options: ' + ', '.join(sorted(options)))
    return bias, weight, offset


def _reset(weight, bias, in_channels, out_channels, weight_initializer, bias_initializer):
    bound = math.sqrt(6. / (in_channels + out_channels)) if weight_initializer == 'glorot' else in_channels ** -.5
    with no_grad():
        weight.tensor().copy_((Tensor.rand(weight.tensor().shape) * 2 - 1) * bound)
        if bias is not None:
            value = Tensor.zeros(bias.tensor().shape) if bias_initializer == 'zeros' else (Tensor.rand(bias.tensor().shape) * 2 - 1) * in_channels ** -.5
            bias.tensor().copy_(value)


class HeteroLinear(Module):
    """Separate [in,out] projection per type with native gradient-preserving groups.

    CPU float32 rank-two features use int64/list type IDs. Widths and type count
    are explicit positive values; lazy initialization and compiled/fused kernels
    are unsupported. is_sorted=True validates the stated ordering. Missing
    types and empty inputs keep ordinary zero input and parameter adjoints.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    num_types: int
    is_sorted: bool
    weight_initializer: object
    bias_initializer: object

    def __init__(self, in_channels, out_channels, num_types, is_sorted=False, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'), (num_types, 'num_types')): _positive(value, name)
        _flags(is_sorted=is_sorted)
        bias, weight_initializer, bias_initializer = _initializers(kwargs)
        _capacity((num_types + 1,), 8)
        _capacity((num_types, in_channels, out_channels))
        _capacity((num_types, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, num_types=num_types,
                   is_sorted=is_sorted, weight_initializer=weight_initializer, bias_initializer=bias_initializer)
        self.weight = Parameter(Tensor.zeros([num_types, in_channels, out_channels]))
        self.bias = Parameter(Tensor.zeros([num_types, out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        _reset(self.weight, self.bias, self.in_channels, self.out_channels, self.weight_initializer, self.bias_initializer)

    def forward(self, x, type_vec):
        _eager(); _features(x, 'x', self.in_channels)
        _capacity((x.shape[0] + 1,), 8)
        _capacity((self.num_types + 1,), 8)
        _capacity(x.shape)
        _capacity((x.shape[0], self.out_channels))
        ids = _index_ids(type_vec, 'type_vec', x.shape[0])
        if any(value >= self.num_types for value in ids): raise ValueError('type_vec contains an unknown type')
        if self.is_sorted and any(left > right for left, right in zip(ids, ids[1:])): raise ValueError('is_sorted=True requires sorted type IDs')
        groups = [[] for _ in range(self.num_types)]
        for row, kind in enumerate(ids): groups[kind].append(row)
        order, values = [], []
        for kind, rows in enumerate(groups):
            order.extend(rows)
            values.append(x.index_select(0, rows).matmul(self.weight.tensor().narrow(0, kind, 1).squeeze(0)))
        inverse = [0] * len(ids)
        for position, row in enumerate(order): inverse[row] = position
        output = cat(values, 0).index_select(0, inverse)
        return output if self.bias is None else output + self.bias.tensor().index_select(0, ids)


class HeteroDictLinear(Module):
    """Registered per-key Linear layers with explicit widths and native state.

    Present known keys are returned in constructor order; unknown keys are
    ignored as in PyG's ordinary eager path. CPU float32 matrices may have
    different row counts, and absent types retain absent parameter gradients.
    Keys are nonempty strings without dots. Lazy widths and fused kernels are
    unsupported; initializer options match HeteroLinear.
    """
    _builtin_init = True
    channel_spec: tuple
    types: tuple
    out_channels: int
    weight_initializer: object
    bias_initializer: object

    def __init__(self, in_channels, out_channels, types=None, **kwargs):
        _positive(out_channels, 'out_channels')
        if types is not None and (not isinstance(types, (list, tuple)) or any(not isinstance(key, str) or not key or '.' in key for key in types) or len(set(types)) != len(types)):
            raise ValueError('types must contain distinct nonempty string keys without dots')
        if type(in_channels) is dict:
            channels = dict(in_channels)
            if types is not None and set(types) != set(channels): raise ValueError('types must match in_channels keys')
        else:
            _positive(in_channels, 'in_channels')
            if types is None: raise ValueError('integer in_channels requires types')
            channels = {key: in_channels for key in types}
        if any(not isinstance(key, str) or not key or '.' in key for key in channels): raise ValueError('channel keys must be nonempty strings without dots')
        for width in channels.values():
            _positive(width, 'input width'); _capacity((width, out_channels))
        bias, weight_initializer, bias_initializer = _initializers(kwargs)
        _configure(self, channel_spec=tuple(channels.items()), types=tuple(channels), out_channels=out_channels,
                   weight_initializer=weight_initializer, bias_initializer=bias_initializer)
        self.lins = ModuleDict({key: Linear(width, out_channels, bias=bias) for key, width in channels.items()})
        self.reset_parameters()

    @property
    def in_channels(self):
        return dict(self.channel_spec)

    def reset_parameters(self):
        for key, width in self.channel_spec:
            layer = self.lins[key]
            _reset(layer.weight, layer.bias, width, self.out_channels, self.weight_initializer, self.bias_initializer)

    def forward(self, x_dict):
        _eager()
        if type(x_dict) is not dict: raise TypeError('x_dict must be a dictionary')
        for key, width in self.channel_spec:
            if key in x_dict:
                _features(x_dict[key], key, width)
                _capacity(x_dict[key].shape)
                _capacity((x_dict[key].shape[0], self.out_channels))
                if self.lins[key].in_features != width or self.lins[key].out_features != self.out_channels:
                    raise ValueError('registered linear widths must match declared channels')
        return {key: self.lins[key](x_dict[key]) for key, _ in self.channel_spec if key in x_dict}
