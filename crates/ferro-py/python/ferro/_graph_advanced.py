"""Advanced eager CPU f32 graph layers composed from native tensor operations.

Topology is COO int64 or native EdgeIndex. No lazy channels, sparse layouts,
implicit stochastic state, generic MessagePassing or graph tracing is provided.
Min/max neighbor ties follow native scatter's shared-tie convention.
"""
import math
from collections.abc import Mapping

from ._native import Tensor, cat, no_grad, scatter, segment_softmax
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _index_ids
from ._graph_conv import _configure, _features, _flags, _options, _pair, _positive, _reset_linear, _reset_network
from ._graph_extra_conv import _gcn_weights, _weights
from ._graph_hetero import _relation
from ._graph_aggregation import _Aggregation, _layout
from .nn import Linear, LSTM, Module, ModuleDict, ModuleList, Parameter, Sequential


class _Activation(Module):
    kind: object

    def forward(self, x):
        if self.kind is None:
            return x
        return x.gelu_erf() if self.kind == 'gelu' else getattr(x, self.kind)()


class _DegreeScalers(Module):
    _builtin_init = True
    histogram: tuple
    train_norm: bool

    def __init__(self, histogram, train_norm):
        _configure(self, histogram=histogram, train_norm=train_norm)
        total = sum(histogram)
        self._initial = (sum(i * n for i, n in enumerate(histogram)) / total,
                         sum(math.log(i + 1) * n for i, n in enumerate(histogram)) / total)
        for name, value in zip(('avg_deg_lin', 'avg_deg_log'), self._initial):
            if train_norm:
                setattr(self, name, Parameter(Tensor([value], [1])))
            else:
                self.register_buffer(name, Tensor([value], [1]))

    def reset_parameters(self):
        with no_grad():
            for name, value in zip(('avg_deg_lin', 'avg_deg_log'), self._initial):
                target = getattr(self, name)
                (target.tensor() if self.train_norm else target).copy_(Tensor([value], [1]))

    def values(self):
        return tuple(value.tensor() if self.train_norm else value for value in (self.avg_deg_lin, self.avg_deg_log))


class PNAConv(Module):
    """Principal-neighborhood aggregation with towers and degree scalers.

    Aggregators: sum/mean/min/max/var/std. Scalers: identity/amplification/
    attenuation/linear/inverse_linear. Degree histogram is copied as immutable
    configuration; train_norm exposes the two normalization parameters.
    Supported MLP activations are relu/tanh/sigmoid/gelu/None, without kwargs.
    Standard deviation uses PyG's 1e-5 variance threshold. An all-isolated
    training histogram is allowed only with the identity scaler.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    aggregators: tuple
    scalers: tuple
    edge_dim: object
    towers: int
    pre_layers: int
    post_layers: int
    divide_input: bool
    act: object
    train_norm: bool

    def __init__(self, in_channels, out_channels, aggregators, scalers, deg, edge_dim=None,
                 towers=1, pre_layers=1, post_layers=1, divide_input=False, act='relu',
                 act_kwargs=None, train_norm=False, **kwargs):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'),
                            (towers, 'towers'), (pre_layers, 'pre_layers'), (post_layers, 'post_layers')):
            _positive(value, name)
        _flags(divide_input=divide_input, train_norm=train_norm)
        _options(kwargs, 'add')
        if out_channels % towers or (divide_input and in_channels % towers):
            raise ValueError('PNA channels must be divisible by towers')
        if not isinstance(aggregators, (tuple, list)) or not aggregators or any(v not in ('sum', 'mean', 'min', 'max', 'var', 'std') for v in aggregators):
            raise ValueError('PNA requires a nonempty list of supported aggregators')
        if not isinstance(scalers, (tuple, list)) or not scalers or any(v not in ('identity', 'amplification', 'attenuation', 'linear', 'inverse_linear') for v in scalers):
            raise ValueError('PNA requires a nonempty list of supported scalers')
        if act not in (None, 'relu', 'tanh', 'sigmoid', 'gelu') or act_kwargs:
            raise NotImplementedError('PNA supports relu/tanh/sigmoid/gelu/None without activation kwargs')
        _cpu_tensor(deg, 'deg', dtype=None)
        if deg.dtype not in ('int64', 'float32') or len(deg.shape) != 1 or not deg.shape[0]:
            raise ValueError('deg must be a nonempty int64/float32 degree histogram')
        histogram = tuple(deg.tolist())
        if any(not math.isfinite(v) or v < 0 or int(v) != v for v in histogram) or not sum(histogram) > 0:
            raise ValueError('degree histogram must contain nonnegative integer counts and nonzero total')
        histogram = tuple(int(v) for v in histogram)
        if not any(histogram[1:]) and any(scaler != 'identity' for scaler in scalers):
            raise ValueError('nonidentity PNA scalers require positive training degree')
        width = in_channels // towers if divide_input else in_channels
        output = out_channels // towers
        if edge_dim is not None:
            _positive(edge_dim, 'edge_dim')
            _capacity((edge_dim, width))
        _capacity((towers, pre_layers + post_layers, len(aggregators) * len(scalers) + 1, width, max(width, output)))
        _capacity((out_channels, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, aggregators=tuple(aggregators),
                   scalers=tuple(scalers), edge_dim=edge_dim, towers=towers, pre_layers=pre_layers,
                   post_layers=post_layers, divide_input=divide_input, act=act, train_norm=train_norm)
        self.aggr_module = _DegreeScalers(histogram, train_norm)
        self.edge_encoder = Linear(edge_dim, width) if edge_dim is not None else None
        self.pre_nns, self.post_nns = ModuleList(), ModuleList()
        for _ in range(towers):
            pre = [Linear((3 if edge_dim is not None else 2) * width, width)]
            post = [Linear((len(aggregators) * len(scalers) + 1) * width, output)]
            for _ in range(pre_layers - 1):
                pre.extend((_Activation(kind=act), Linear(width, width)))
            for _ in range(post_layers - 1):
                post.extend((_Activation(kind=act), Linear(output, output)))
            self.pre_nns.append(Sequential(*pre))
            self.post_nns.append(Sequential(*post))
        self.lin = Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.aggr_module.reset_parameters()
        for module in (self.edge_encoder, self.pre_nns, self.post_nns, self.lin):
            if module is not None:
                _reset_network(module)

    @staticmethod
    def get_degree_histogram(loader):
        """Count incoming degrees, including isolated nodes, across graph batches."""
        histogram = [0]
        for data in loader:
            _, destinations, size = _edge_parts(data.edge_index, num_nodes=data.num_nodes)
            _capacity((size[1],))
            degrees = [0] * size[1]
            for destination in destinations:
                degrees[destination] += 1
            required = max(degrees, default=0) + 1
            if required > len(histogram):
                histogram.extend([0] * (required - len(histogram)))
            for degree in degrees:
                histogram[degree] += 1
        return Tensor.from_i64(histogram, [len(histogram)])

    def forward(self, x, edge_index, edge_attr=None):
        _features(x, 'x', self.in_channels)
        nodes = x.shape[0]
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        width = self.in_channels // self.towers if self.divide_input else self.in_channels
        for count in (nodes, len(src)):
            _capacity((count, self.towers, width, len(self.aggregators), len(self.scalers)))
        if self.edge_dim is not None:
            _cpu_tensor(edge_attr, 'edge_attr')
            if edge_attr.shape != [len(src), self.edge_dim]:
                raise ValueError('PNA edge_attr must have shape [E,edge_dim]')
            attributes = self.edge_encoder(edge_attr)
        elif edge_attr is not None:
            raise ValueError('PNA edge_attr requires configured edge_dim')
        else:
            attributes = None
        degrees = [0] * nodes
        for i in dst:
            degrees[i] += 1
        degree = Tensor(degrees, [nodes, 1])
        log_degree = Tensor([math.log(d + 1) for d in degrees], [nodes, 1])
        safe_degree = Tensor([max(d, 1) for d in degrees], [nodes, 1])
        safe_log = Tensor([math.log(max(d, 1) + 1) for d in degrees], [nodes, 1])
        avg_lin, avg_log = self.aggr_module.values()
        outputs = []
        for tower in range(self.towers):
            features = x.narrow(1, tower * width, width) if self.divide_input else x
            parts = [features.index_select(0, dst), features.index_select(0, src)]
            if attributes is not None:
                parts.append(attributes)
            messages = self.pre_nns[tower](cat(parts, 1))
            reductions = []
            for aggregation in self.aggregators:
                if aggregation in ('var', 'std'):
                    mean = scatter(messages, dst, nodes, 'mean')
                    value = scatter(messages * messages, dst, nodes, 'mean') - mean * mean
                    if aggregation == 'std':
                        mask = Tensor([float(v <= 1e-5) for v in value.flatten().tolist()], list(value.shape))
                        value = value.clamp(1e-5, float('inf')).sqrt().masked_fill(mask, 0.)
                else:
                    value = scatter(messages, dst, nodes, aggregation)
                reductions.append(value)
            aggregate = cat(reductions, 1)
            scaled = []
            for scaler in self.scalers:
                if scaler == 'identity':
                    scaled.append(aggregate)
                else:
                    factor = (log_degree / avg_log if scaler == 'amplification' else avg_log / safe_log if scaler == 'attenuation'
                              else degree / avg_lin if scaler == 'linear' else avg_lin / safe_degree)
                    scaled.append(aggregate * factor)
            outputs.append(self.post_nns[tower](cat([features, *scaled], 1)))
        return self.lin(cat(outputs, 1))


class _TypedLinears(Module):
    _builtin_init = True
    widths: tuple
    out_channels: int

    def __init__(self, widths, out_channels):
        _configure(self, widths=tuple(widths.items()), out_channels=out_channels)
        self.lins = ModuleDict({name: Linear(width, out_channels) for name, width in widths.items()})


class _TypedParameters(Module):
    _builtin_init = True
    keys: tuple
    shape: tuple

    def __init__(self, keys, shape):
        _configure(self, keys=tuple(keys), shape=tuple(shape))
        encoded = [key.replace('.', '#') for key in keys]
        if len(set(encoded)) != len(encoded) or any(not name or name.startswith('_') or hasattr(self, name) for name in encoded):
            raise ValueError('typed parameter keys collide with attributes or each other')
        self._names = dict(zip(keys, encoded))
        for name in encoded:
            setattr(self, name, Parameter(Tensor.ones(list(shape))))

    def __getitem__(self, key):
        return getattr(self, self._names[key])


class _RelationWeights(Module):
    count: int
    width: int

    def build(self):
        self.weight = Parameter(Tensor.zeros([self.count, self.width, self.width]))


class HGTConv(Module):
    """Heterogeneous transformer with a joint softmax over incoming relations.

    Explicit positive node widths and complete fixed metadata are required.
    Input dictionaries may omit unused node types and relations. As in PyG 2.8,
    output keys are present input node types that are metadata destinations;
    empty incoming edge sets produce output bias plus the optional gated skip.
    Node names obey ModuleDict keys; encoded relation-prior collisions reject.
    """
    _builtin_init = True
    in_channels: tuple
    out_channels: int
    metadata: tuple
    heads: int
    aggr: str

    def __init__(self, in_channels, out_channels, metadata, heads=1, **kwargs):
        _positive(out_channels, 'out_channels')
        _positive(heads, 'heads')
        if out_channels % heads:
            raise ValueError('HGT out_channels must be divisible by heads')
        if not isinstance(metadata, (tuple, list)) or len(metadata) != 2:
            raise ValueError('metadata must be (node_types, edge_types)')
        nodes, relations = tuple(metadata[0]), tuple(_relation(key) for key in metadata[1])
        if not nodes or any(type(name) is not str or not name or '.' in name or name.startswith('_') for name in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError('HGT node types must be unique public ModuleDict names')
        if not relations or len(set(relations)) != len(relations) or any(src not in nodes or dst not in nodes for src, _, dst in relations):
            raise ValueError('HGT metadata requires unique relations with declared node types')
        widths = dict(in_channels) if isinstance(in_channels, Mapping) else {name: in_channels for name in nodes}
        if set(widths) != set(nodes):
            raise ValueError('HGT in_channels keys must exactly match metadata node types')
        widths = {name: _positive(widths[name], 'in_channels') for name in nodes}
        for width in widths.values():
            _capacity((3, width, out_channels))
        _capacity((len(nodes), out_channels, out_channels))
        _capacity((len(relations), heads, out_channels // heads, out_channels // heads))
        prior_keys = tuple('__'.join(key) for key in relations)
        if len(set(prior_keys)) != len(prior_keys):
            raise ValueError('HGT relation names collide after prior-key encoding')
        _configure(self, in_channels=tuple(widths.items()), out_channels=out_channels,
                   metadata=(nodes, relations), heads=heads, aggr=_options(kwargs, 'add'))
        self.kqv_lin = _TypedLinears(widths, 3 * out_channels)
        self.out_lin = _TypedLinears({name: out_channels for name in nodes}, out_channels)
        self.k_rel = _RelationWeights(count=heads * len(relations), width=out_channels // heads)
        self.v_rel = _RelationWeights(count=heads * len(relations), width=out_channels // heads)
        self.skip = _TypedParameters(nodes, (1,))
        self.p_rel = _TypedParameters(prior_keys, (1, heads))
        self.reset_parameters()

    def reset_parameters(self):
        for linears in (self.kqv_lin, self.out_lin):
            for _, layer in linears.lins.items():
                _reset_linear(layer)
        with no_grad():
            for relation in (self.k_rel, self.v_rel):
                weight = relation.weight.tensor()
                weight.copy_((Tensor.rand(list(weight.shape)) * 2 - 1) * relation.width ** -.5)
            for module in (self.skip, self.p_rel):
                for parameter in module.parameters():
                    parameter.tensor().copy_(Tensor.ones(list(parameter.tensor().shape)))

    def forward(self, x_dict, edge_index_dict):
        if not isinstance(x_dict, Mapping) or not isinstance(edge_index_dict, Mapping) or not x_dict:
            raise ValueError('HGT requires a nonempty feature mapping and relation mapping')
        widths = dict(self.in_channels)
        if any(name not in widths for name in x_dict):
            raise ValueError('HGT input contains an undeclared node type')
        relations = self.metadata[1]
        if any(key not in relations for key in edge_index_dict):
            raise ValueError('HGT input contains an undeclared relation')
        for name, features in x_dict.items():
            _features(features, name, widths[name])
            _capacity((features.shape[0], self.out_channels, 3))
        validated = {}
        for key, topology in edge_index_dict.items():
            src, _, dst = key
            if src not in x_dict or dst not in x_dict:
                raise ValueError('HGT relation references a missing feature type')
            validated[key] = _edge_parts(topology, size=(x_dict[src].shape[0], x_dict[dst].shape[0]))[:2]
        heads, width = self.heads, self.out_channels // self.heads
        _capacity((sum(len(src) for src, _ in validated.values()), heads, width))
        projections = {}
        for name, features in x_dict.items():
            values = self.kqv_lin.lins[name](features)
            projections[name] = tuple(values.narrow(1, i * self.out_channels, self.out_channels).reshape([features.shape[0], heads, width]) for i in range(3))
        offsets, total = {}, 0
        for name, features in x_dict.items():
            offsets[name], total = total, total + features.shape[0]
        keys, messages, priors, destinations = [], [], [], []
        for relation, (src, dst) in validated.items():
            source, _, target = relation
            index = relations.index(relation)
            _capacity((len(src), heads, width))
            key_parts, value_parts = [], []
            for head in range(heads):
                k = projections[source][0].narrow(1, head, 1).reshape([x_dict[source].shape[0], width])
                v = projections[source][2].narrow(1, head, 1).reshape([x_dict[source].shape[0], width])
                position = head * len(relations) + index
                kw = self.k_rel.weight.tensor().narrow(0, position, 1).reshape([width, width])
                vw = self.v_rel.weight.tensor().narrow(0, position, 1).reshape([width, width])
                key_parts.append((k @ kw).index_select(0, src).reshape([len(src), 1, width]))
                value_parts.append((v @ vw).index_select(0, src).reshape([len(src), 1, width]))
            keys.append(cat(key_parts, 1))
            messages.append(cat(value_parts, 1))
            priors.append(Tensor.ones([len(src), heads]) * self.p_rel['__'.join(relation)].tensor())
            destinations.extend(offsets[target] + i for i in dst)
        _capacity((total, heads, width))
        if keys:
            queries = cat([projections[name][1] for name in x_dict], 0).index_select(0, destinations)
            logits = (cat(keys, 0) * queries).sum_dim(2, False) * cat(priors, 0) / math.sqrt(width)
            attention = segment_softmax(logits, destinations, total)
            aggregate = scatter(cat(messages, 0) * attention.reshape([len(destinations), heads, 1]), destinations, total, 'sum').reshape([total, self.out_channels])
        else:
            aggregate = Tensor.zeros([total, self.out_channels])
        outputs = {}
        targets = {key[2] for key in relations}
        for name, features in x_dict.items():
            if name in targets:
                value = aggregate.narrow(0, offsets[name], features.shape[0])
                value = self.out_lin.lins[name](value.gelu_erf())
                if features.shape[1] == self.out_channels:
                    skip = self.skip[name].tensor().sigmoid()
                    value = skip * value + (1 - skip) * features
                outputs[name] = value
        return outputs


class GCN2Conv(Module):
    """GCNII initial residual and identity mapping, with differentiable weights.

    Supports shared or separate transforms and existing COO GCN normalization.
    No cached topology. Weighted duplicate self-loops with inserted loops reject.
    x_0 may have extra trailing rows, matching PyG's initial-state slicing.
    """
    _builtin_init = True
    channels: int
    alpha: float
    theta: object
    layer: object
    beta: float
    shared_weights: bool
    cached: bool
    add_self_loops: bool
    normalize: bool
    aggr: str

    def __init__(self, channels, alpha, theta=None, layer=None, shared_weights=True,
                 cached=False, add_self_loops=True, normalize=True, **kwargs):
        _positive(channels, 'channels')
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be finite and in [0,1]')
        if (theta is None) != (layer is None):
            raise ValueError('theta and layer must be specified together')
        if theta is not None:
            if type(theta) not in (int, float) or not math.isfinite(theta) or theta < 0:
                raise ValueError('theta must be finite and nonnegative')
            _positive(layer, 'layer')
        _flags(shared_weights=shared_weights, cached=cached, add_self_loops=add_self_loops, normalize=normalize)
        if cached:
            raise NotImplementedError('GCN2Conv supports cached=False only')
        _capacity((channels, channels))
        _configure(self, channels=channels, alpha=float(alpha), theta=None if theta is None else float(theta),
                   layer=layer, beta=1. if theta is None else math.log1p(theta / layer), shared_weights=shared_weights,
                   cached=cached, add_self_loops=add_self_loops, normalize=normalize, aggr=_options(kwargs, 'add'))
        self.weight1 = Parameter(Tensor.zeros([channels, channels]))
        self.weight2 = None if shared_weights else Parameter(Tensor.zeros([channels, channels]))
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            for parameter in (self.weight1, self.weight2):
                if parameter is not None:
                    parameter.tensor().copy_((Tensor.rand([self.channels, self.channels]) * 2 - 1) * math.sqrt(3 / self.channels))

    def forward(self, x, x_0, edge_index, edge_weight=None):
        _features(x, 'x', self.channels)
        _features(x_0, 'x_0', self.channels)
        if x_0.shape[0] < x.shape[0]:
            raise ValueError('x_0 must contain at least as many rows as x')
        nodes = x.shape[0]
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        _weights(edge_weight, len(src))
        _capacity((len(src) + nodes, self.channels))
        if self.normalize:
            src, dst, weight = _gcn_weights(src, dst, nodes, edge_weight, self.add_self_loops)
        else:
            weight = Tensor.ones([len(src)]) if edge_weight is None else edge_weight
        propagated = scatter(x.index_select(0, src) * weight.reshape([len(src), 1]), dst, nodes, 'sum') * (1 - self.alpha)
        initial = x_0.narrow(0, 0, nodes) * self.alpha
        if self.weight2 is None:
            combined = propagated + initial
            return combined * (1 - self.beta) + (combined @ self.weight1.tensor()) * self.beta
        return propagated * (1 - self.beta) + (propagated @ self.weight1.tensor()) * self.beta + initial * (1 - self.beta) + (initial @ self.weight2.tensor()) * self.beta


class GraphConv(Module):
    """Weighted sum/mean/max neighbors followed by affine and root transforms."""
    _builtin_init = True
    in_channels: object
    out_channels: int
    aggr: str

    def __init__(self, in_channels, out_channels, aggr='add', bias=True, **kwargs):
        widths = (in_channels, in_channels) if type(in_channels) is int else in_channels
        if not isinstance(widths, (tuple, list)) or len(widths) != 2:
            raise ValueError('in_channels must be positive or a source/target pair')
        widths = tuple(_positive(value, 'in_channels') for value in widths)
        _positive(out_channels, 'out_channels')
        _flags(bias=bias)
        _options(kwargs, 'add')
        if aggr not in ('add', 'sum', 'mean', 'max'):
            raise NotImplementedError('GraphConv supports add/sum/mean/max aggregation')
        for width in widths:
            _capacity((width, out_channels))
        _configure(self, in_channels=in_channels if type(in_channels) is int else widths, out_channels=out_channels, aggr=aggr)
        self.lin_rel = Linear(widths[0], out_channels, bias=bias)
        self.lin_root = Linear(widths[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin_rel)
        _reset_linear(self.lin_root)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        widths = (self.in_channels, self.in_channels) if type(self.in_channels) is int else self.in_channels
        source, target, size = _pair(x, size, *widths)
        src, dst, _ = _edge_parts(edge_index, size=size)
        _weights(edge_weight, len(src))
        for count in (*size, len(src)):
            _capacity((count, max(*widths, self.out_channels)))
        messages = source.index_select(0, src)
        if edge_weight is not None:
            messages = messages * edge_weight.reshape([len(src), 1])
        output = self.lin_rel(scatter(messages, dst, size[1], 'sum' if self.aggr == 'add' else self.aggr))
        return output if target is None else output + self.lin_root(target)


class _SelectTopK(Module):
    _builtin_init = True
    in_channels: int
    ratio: object
    min_score: object
    nonlinearity: str

    def __init__(self, in_channels, ratio, min_score, nonlinearity):
        _configure(self, in_channels=in_channels, ratio=ratio, min_score=min_score, nonlinearity=nonlinearity)
        self.weight = Parameter(Tensor.zeros([1, in_channels]))
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            self.weight.tensor().copy_((Tensor.rand([1, self.in_channels]) * 2 - 1) * self.in_channels ** -.5)

    def forward(self, features, groups, count):
        scores = (features * self.weight.tensor()).sum_dim(1, False)
        if self.min_score is None:
            norm = (self.weight.tensor() * self.weight.tensor()).sum().sqrt()
            if not math.isfinite(norm.item()) or norm.item() <= 0:
                raise ValueError('TopK projection weight must have finite nonzero norm')
            scores = getattr(scores / norm, self.nonlinearity)()
        else:
            scores = segment_softmax(scores, groups, count)
        values = scores.tolist()
        if any(not math.isfinite(value) for value in values):
            raise ValueError('TopK scores must be finite')
        if self.min_score is not None:
            threshold = (scatter(scores, groups, count, 'max') - 1e-7).clamp(float('-inf'), self.min_score).tolist()
            selected = [i for i, value in enumerate(values) if value > threshold[groups[i]]]
        else:
            partitions = [[] for _ in range(count)]
            for i, group in enumerate(groups):
                partitions[group].append(i)
            scaled = None if self.ratio >= 1 else (Tensor([len(part) for part in partitions], [count]) * self.ratio).tolist()
            selected = []
            for group, partition in enumerate(partitions):
                k = int(self.ratio) if self.ratio >= 1 else math.ceil(scaled[group])
                selected.extend(sorted(partition, key=lambda i: -values[i])[:k])
        return selected, scores.index_select(0, selected)


def _pool_config(in_channels, ratio, min_score, multiplier, nonlinearity):
    _positive(in_channels, 'in_channels')
    if ratio is not None and (type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0):
        raise ValueError('ratio must be finite and positive')
    if min_score is not None and (type(min_score) not in (int, float) or not math.isfinite(min_score) or not 0 <= min_score <= 1):
        raise ValueError('min_score must be in [0,1]')
    if ratio is None and min_score is None:
        raise ValueError('pooling requires ratio or min_score')
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier):
        raise ValueError('multiplier must be finite')
    if nonlinearity not in ('tanh', 'sigmoid', 'relu'):
        raise NotImplementedError('pooling supports tanh/sigmoid/relu nonlinearities')
    _capacity((1, in_channels))
    return dict(in_channels=in_channels, ratio=ratio, min_score=min_score, multiplier=float(multiplier), nonlinearity=nonlinearity)


class TopKPooling(Module):
    """Learned score selection and induced sparse subgraph, with score gradients.

    Selection ties keep original node order. PyTorch's unstable sort may choose
    another order at ties; away from ties outputs and derivatives match PyG.
    Ratios >=1 are absolute counts (including float 1.0), matching PyG 2.8.
    min_score selection retains original global order and at least one maximum
    per nonempty graph. Empty graphs return empty outputs. Zero projection norm
    rejects instead of producing NaNs. Arbitrary callable nonlinearities reject.
    """
    _builtin_init = True
    in_channels: int
    ratio: object
    min_score: object
    multiplier: float
    nonlinearity: str

    def __init__(self, in_channels, ratio=.5, min_score=None, multiplier=1., nonlinearity='tanh'):
        _configure(self, **_pool_config(in_channels, ratio, min_score, multiplier, nonlinearity))
        self.select = _SelectTopK(in_channels, ratio, min_score, nonlinearity)

    def reset_parameters(self):
        self.select.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        _features(x, 'x')
        nodes = x.shape[0]
        src, dst, _ = _edge_parts(edge_index, num_nodes=nodes)
        if edge_attr is not None:
            _cpu_tensor(edge_attr, 'edge_attr')
            if not edge_attr.shape or edge_attr.shape[0] != len(src):
                raise ValueError('edge_attr must have one row per edge')
        groups = [0] * nodes if batch is None else _index_ids(batch, 'batch', nodes)
        count = max(groups, default=-1) + 1
        _capacity((count, max(1, x.shape[1])))
        features = x if attn is None else attn
        _cpu_tensor(features, 'attn')
        if len(features.shape) == 1:
            features = features.reshape([features.shape[0], 1])
        _features(features, 'attn', self.in_channels)
        if features.shape[0] != nodes:
            raise ValueError('attn must have one row per node')
        if hasattr(self, 'gnn'):
            features = self.gnn(features, edge_index)
        selected, scores = self.select(features, groups, count)
        lookup = {old: new for new, old in enumerate(selected)}
        kept = [i for i, (source, target) in enumerate(zip(src, dst)) if source in lookup and target in lookup]
        topology = Tensor.from_i64([lookup[src[i]] for i in kept] + [lookup[dst[i]] for i in kept], [2, len(kept)])
        output = x.index_select(0, selected) * scores.reshape([len(selected), 1]) * self.multiplier
        return (output, topology, None if edge_attr is None else edge_attr.index_select(0, kept),
                Tensor.from_i64([groups[i] for i in selected], [len(selected)]),
                Tensor.from_i64(selected, [len(selected)]), scores)


class SAGPooling(TopKPooling):
    """Self-attention graph pooling with a registered GraphConv scorer by default.

    GNN may be another Ferro Module class accepting (in_channels, 1, **kwargs).
    The instantiated scorer's class/configuration is bound by checkpoints.
    """
    _builtin_init = True

    def __init__(self, in_channels, ratio=.5, GNN=GraphConv, min_score=None, multiplier=1., nonlinearity='tanh', **kwargs):
        if not isinstance(GNN, type) or not issubclass(GNN, Module):
            raise TypeError('SAGPooling GNN must be a Ferro Module class')
        _configure(self, **_pool_config(in_channels, ratio, min_score, multiplier, nonlinearity))
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.select = _SelectTopK(1, ratio, min_score, nonlinearity)

    def reset_parameters(self):
        _reset_network(self.gnn)
        self.select.reset_parameters()


def _groups(x, index, ptr, dim_size, dim):
    _features(x, 'x')
    if dim not in (0, -2):
        raise NotImplementedError('aggregation supports rank-two input and dim=0/-2')
    nodes = x.shape[0]
    if ptr is not None:
        if index is not None:
            raise ValueError('provide either index or ptr, not both')
        pointers = _index_ids(ptr, 'ptr')
        if not pointers or pointers[0] != 0 or pointers[-1] != nodes or any(a > b for a, b in zip(pointers, pointers[1:])):
            raise ValueError('ptr must be monotone from zero to the number of rows')
        groups = [group for group, (start, end) in enumerate(zip(pointers, pointers[1:])) for _ in range(end - start)]
        inferred = len(pointers) - 1
    else:
        groups = [0] * nodes if index is None else _index_ids(index, 'index', nodes)
        inferred = max(groups, default=-1) + 1
    count = inferred if dim_size is None else _count(dim_size, 'dim_size')
    if count < inferred:
        raise ValueError('dim_size is smaller than the provided group indices')
    _capacity((count, max(1, x.shape[1])))
    return groups, count


class AttentionalAggregation(_Aggregation):
    """Per-group softmax gating with registered gate and optional value Modules.

    Supports rank-two features and COO index or CSR ptr groups. Gate output is
    [N,1] or [N,F_out]; empty groups are zero. Shared gate/value Module aliases
    remain shared parameters and participate in checkpoint validation. Known
    native feature networks have checked widths and state preflight. Custom
    networks retain direct-forward support; composition requires their explicit
    get_out_channels metadata. MLPs receive original COO index/batch size; CSR
    paths retain PyG batch=None behavior. Older non-MLP schemas are unchanged.
    """
    _builtin_init = True

    def __init__(self, gate_nn, nn=None):
        if not isinstance(gate_nn, Module) or (nn is not None and not isinstance(nn, Module)):
            raise TypeError('attention gate/value networks must be Ferro Modules')
        from ._graph_mlp import MLP
        self._initialize()
        self.gate_nn=None if isinstance(gate_nn,MLP) else gate_nn
        self.nn=None if isinstance(nn,MLP) else nn
        if isinstance(gate_nn,MLP): self.gate_mlp=gate_nn
        if isinstance(nn,MLP): self.mlp=nn

    def _networks(self):
        return getattr(self,'gate_mlp',self.gate_nn),getattr(self,'mlp',self.nn)

    def reset_parameters(self):
        gate,value=self._networks()
        _reset_network(gate)
        if value is not None: _reset_network(value)

    def _widths(self, in_channels, rows=None, allow_unknown=False):
        from ._graph_deep_sets import _network_width
        width = _count(in_channels, 'in_channels')
        widths = []
        for module in self._networks():
            try:
                widths.append(_network_width(module, width, rows))
            except NotImplementedError:
                if not allow_unknown: raise
                widths.append(None)
        gate_width, value_width = widths
        if None not in widths and gate_width not in (1, value_width):
            raise ValueError('attention gate width must be one or the value output width')
        return gate_width, value_width

    def get_out_channels(self, in_channels):
        return self._widths(in_channels)[1]

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        groups, count = _layout(x, index, ptr, dim_size, dim)
        from ._graph_mlp import MLP
        _, width = self._widths(x.shape[1], x.shape[0])
        for module in self._networks():
            if isinstance(module,MLP): module._metadata(x.shape[1],x.shape[0],index,dim_size,True)
        _capacity((count, width))
        return groups, count

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        groups, count = _groups(x, index, ptr, dim_size, dim)
        widths = self._widths(x.shape[1], x.shape[0], allow_unknown=True)
        if widths[1] is not None:
            _capacity((count, widths[1]))
        from ._graph_mlp import MLP
        gate_module,value_module=self._networks()
        for module in (gate_module,value_module):
            if isinstance(module,MLP): module._metadata(x.shape[1],x.shape[0],index,dim_size,True)
        gate=gate_module(x,batch=index,batch_size=dim_size) if isinstance(gate_module,MLP) else gate_module(x)
        value=x if value_module is None else value_module(x,batch=index,batch_size=dim_size) if isinstance(value_module,MLP) else value_module(x)
        _features(value, 'value network output')
        _features(gate, 'gate network output')
        if value.shape[0] != x.shape[0] or gate.shape[0] != x.shape[0] or gate.shape[1] not in (1, value.shape[1]):
            raise ValueError('attention gate must be [N,1] or [N,F_out], and values [N,F_out]')
        if any(expected is not None and actual != expected for expected, actual in zip(widths, (gate.shape[1], value.shape[1]))):
            raise ValueError('attention network outputs do not match their declared widths')
        _capacity((count, value.shape[1]))
        attention = segment_softmax(gate, groups, count)
        return scatter(attention * value, groups, count, 'sum')


class Set2Set(_Aggregation):
    """Iterative content attention using the existing public CPU LSTM module.

    Supports stacked unidirectional LSTM with bias or no bias and saved
    interlayer dropout streams. Output width is twice the input width.
    Recurrent state is local to each forward, including empty groups; no hidden
    state or optimizer is retained. Input is rank two with index or CSR ptr.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    processing_steps: int

    def __init__(self, in_channels, processing_steps, **kwargs):
        _positive(in_channels, 'in_channels')
        _positive(processing_steps, 'processing_steps')
        options = dict(kwargs)
        if options.get('bidirectional', False) or options.get('batch_first', False) or options.get('proj_size', 0):
            raise NotImplementedError('Set2Set requires unidirectional, sequence-first, unprojected LSTM')
        layers = _positive(options.get('num_layers', 1), 'num_layers')
        _capacity((layers, 4, in_channels, 2 * in_channels))
        _capacity((processing_steps, 4, in_channels, 2 * in_channels))
        _configure(self, in_channels=in_channels, out_channels=2 * in_channels, processing_steps=processing_steps)
        self.lstm = LSTM(2 * in_channels, in_channels, **options)

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def get_out_channels(self, in_channels):
        _count(in_channels, 'in_channels')
        if in_channels != self.in_channels:
            raise ValueError('Set2Set input width must match in_channels')
        return self.out_channels

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        groups, count = _layout(x, index, ptr, dim_size, dim)
        self.get_out_channels(x.shape[1])
        _capacity((self.lstm.num_layers, count, self.out_channels))
        return groups, count

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        groups, count = self._preflight(x, index, ptr, dim_size, dim)
        hidden = (Tensor.zeros([self.lstm.num_layers, count, self.in_channels]), Tensor.zeros([self.lstm.num_layers, count, self.in_channels]))
        query = Tensor.zeros([count, self.out_channels])
        for _ in range(self.processing_steps):
            q, hidden = self.lstm(query.unsqueeze(0), hidden)
            q = q.reshape([count, self.in_channels])
            logits = (x * q.index_select(0, groups)).sum_dim(1, True)
            attention = segment_softmax(logits, groups, count)
            readout = scatter(attention * x, groups, count, 'sum')
            query = cat([q, readout], 1)
        return query


def dense_diff_pool(x, adj, s, mask=None, normalize=True):
    """Dense DiffPool outputs plus differentiable link and assignment entropy losses.

    Rank-two inputs acquire a batch dimension. All three inputs otherwise use
    matching explicit batch/node dimensions, with no implicit batch broadcasts.
    Bool or binary f32 masks are supported; float masks retain gradients. Empty
    batch/node axes reject because PyG's mean/normalized losses are undefined.
    At exact zero link residual, the norm uses Torch's zero-subgradient choice.
    """
    _flags(normalize=normalize)
    values = []
    for value, name in ((x, 'x'), (adj, 'adj'), (s, 's')):
        _cpu_tensor(value, name)
        if len(value.shape) not in (2, 3):
            raise ValueError('DiffPool inputs must have rank two or three')
        values.append(value.unsqueeze(0) if len(value.shape) == 2 else value)
    x, adj, s = values
    batch, nodes, width = x.shape
    if not batch or not nodes or not s.shape[2] or adj.shape != [batch, nodes, nodes] or s.shape[:2] != [batch, nodes]:
        raise ValueError('DiffPool requires nonempty matching batch/node axes, square adjacency and positive cluster count')
    _capacity((batch, nodes, nodes))
    _capacity((batch, s.shape[2], max(width, s.shape[2])))
    assignment = s.softmax(2)
    if mask is not None:
        _cpu_tensor(mask, 'mask', dtype=None)
        if mask.dtype not in ('bool', 'float32') or mask.shape not in ([batch, nodes], [nodes] if batch == 1 else [batch, nodes]):
            raise ValueError('DiffPool mask must be bool/binary float32 [B,N] (or [N] for one graph)')
        if mask.dtype == 'bool':
            mask = Tensor([float(v) for v in mask.flatten().tolist()], [batch, nodes, 1])
        else:
            if any(value not in (0., 1.) for value in mask.flatten().tolist()):
                raise ValueError('DiffPool float mask must be binary')
            mask = mask.reshape([batch, nodes, 1])
        x, assignment = x * mask, assignment * mask
    transposed = assignment.transpose(1, 2)
    output = transposed.bmm(x)
    adjacency = transposed.bmm(adj).bmm(assignment)
    residual = adj - assignment.bmm(transposed)
    squared = (residual * residual).sum()
    link = squared * 0. if squared.item() == 0. else squared.sqrt()
    if normalize:
        link = link / (batch * nodes * nodes)
    entropy = (-assignment * (assignment + 1e-15).log()).sum_dim(2, False).mean()
    return output, adjacency, link, entropy
