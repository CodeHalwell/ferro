"""Typed CPU heterogeneous graph stores and relation-wise module composition.

Fields use float32/float64/int64/bool tensors; graph layers can impose narrower
feature dtype contracts. Sparse layouts and automatic to_hetero/FX
conversion are not implemented. HeteroBatch handles disjoint batching separately.
Relation keys are full
(source_type, relation_type, target_type) triples. HeteroConv executes existing
Ferro Modules and native tensor operations; it does not introduce an AD engine.
"""
from collections.abc import Mapping

from ._native import Tensor, cat, stack, scatter, no_grad
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts
from ._graph_conv import _configure, _positive, _features, _flags
from .nn import Module, Parameter


def _relation(key):
    if type(key) is not tuple or len(key) != 3 or any(type(part) is not str or not part for part in key):
        raise ValueError('edge types must be (source, relation, target) triples of nonempty strings')
    return key


def _field(key, value):
    if type(key) is not str or not key or key.startswith('_'):
        raise ValueError('graph fields must have nonempty public string names')
    _cpu_tensor(value, key, dtype=None)
    if value.dtype not in ('float32', 'float64', 'int64', 'bool'):
        raise ValueError('heterogeneous graph fields support CPU float32/float64/int64/bool only')


class _Store:
    def __init__(self, parent, key):
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_key', key)
        object.__setattr__(self, '_mapping', {})

    def __getattr__(self, key):
        try:
            return self._mapping[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getitem__(self, key):
        return self._mapping[key]

    def __setitem__(self, key, value):
        if self._key is None:
            self._parent[key] = value
            return
        if key == 'num_nodes' and isinstance(self._key, str):
            if value is None:
                self._mapping.pop(key, None)
            else:
                self._mapping[key] = _count(value, 'num_nodes')
            return
        if type(key) is not str or hasattr(type(self), key) or key.startswith('_'):
            raise ValueError('unsupported or reserved heterogeneous field')
        if value is None:
            self._mapping.pop(key, None)
            return
        _field(key, value)
        if key == 'edge_index':
            if isinstance(self._key, str):
                raise ValueError('edge_index belongs in a relation store')
            _edge_parts(value)
        self._mapping[key] = value

    def __delitem__(self, key):
        del self._mapping[key]

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __contains__(self, key):
        return key in self._mapping

    def __len__(self):
        return len(self._mapping)

    def keys(self):
        return list(self._mapping)

    def items(self):
        return self._mapping.items()

    def to_dict(self):
        return dict(self._mapping)

    @property
    def num_nodes(self):
        if not isinstance(self._key, str):
            raise AttributeError('num_nodes is node-store metadata')
        if 'num_nodes' in self._mapping:
            return self._mapping['num_nodes']
        for name in ('x', 'pos', 'node_attr', 'node_type', *self._mapping):
            value = self._mapping.get(name)
            if isinstance(value, Tensor) and value.shape:
                return value.shape[0]
        return None

    @property
    def num_edges(self):
        if not isinstance(self._key, tuple):
            raise AttributeError('num_edges is relation-store metadata')
        if 'edge_index' in self._mapping:
            return self.edge_index.shape[1]
        for value in self._mapping.values():
            if isinstance(value, Tensor) and value.shape:
                return value.shape[0]
        return 0

    @property
    def num_node_features(self):
        if not isinstance(self._key, str):
            raise AttributeError('num_node_features is node-store metadata')
        value = self._mapping.get('x')
        return 0 if value is None else (value.shape[-1] if len(value.shape) > 1 else 1)

    @property
    def num_edge_features(self):
        if not isinstance(self._key, tuple):
            raise AttributeError('num_edge_features is relation-store metadata')
        value = self._mapping.get('edge_attr')
        return 0 if value is None else (value.shape[-1] if len(value.shape) > 1 else 1)

    def size(self):
        if not isinstance(self._key, tuple):
            raise TypeError('size() is relation-store metadata')
        return tuple(self._parent._nodes[key].num_nodes if key in self._parent._nodes else None
                     for key in (self._key[0], self._key[2]))


class HeteroData:
    """Node/relation stores with typed CPU tensor attributes and validation.

Set each node type's num_nodes when features cannot provide a node count.
Non-scalar store fields must agree with that store's node/edge cardinality;
scalar fields are metadata. clone preserves floating-point autograd connectivity.
to('cpu') is an in-place no-op; other devices require unsupported migration.
"""
    def __init__(self, mapping=None, **kwargs):
        object.__setattr__(self, '_nodes', {})
        object.__setattr__(self, '_edges', {})
        object.__setattr__(self, '_global', {})
        object.__setattr__(self, '_global_view', _Store(self, None))
        object.__setattr__(self._global_view, '_mapping', self._global)
        if mapping is not None:
            if not isinstance(mapping, Mapping):
                raise TypeError('HeteroData mapping must be a mapping')
            for key, value in mapping.items():
                if key == '_global_store':
                    if not isinstance(value, Mapping):
                        raise TypeError('_global_store must be a mapping')
                    for name, field in value.items():
                        self[name] = field
                elif isinstance(value, Mapping):
                    store = self[key]
                    for name, field in value.items():
                        store[name] = field
                else:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, key):
        if key in self._global:
            return self._global[key]
        if key.endswith('_dict'):
            return self.collect(key[:-5])
        raise AttributeError(key)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __getitem__(self, key):
        if type(key) is str:
            if not key or key == '_global_store':
                raise ValueError('node types must be nonempty strings')
            if key in self._global:
                return self._global[key]
            if key not in self._nodes:
                self._nodes[key] = _Store(self, key)
            return self._nodes[key]
        key = _relation(key)
        if key not in self._edges:
            self._edges[key] = _Store(self, key)
        return self._edges[key]

    def __setitem__(self, key, value):
        if type(key) is not str or key in self._nodes or hasattr(type(self), key) or key.endswith('_dict'):
            raise ValueError('set typed store fields through data[node_or_relation][field]')
        if value is None:
            self._global.pop(key, None)
        else:
            _field(key, value)
            self._global[key] = value

    def __delitem__(self, key):
        if key in self._global:
            del self._global[key]
        elif isinstance(key, str):
            del self._nodes[key]
        else:
            del self._edges[_relation(key)]

    def __delattr__(self, key):
        if key not in self._global:
            raise AttributeError(key)
        del self._global[key]

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return len(self.keys())

    def keys(self):
        return list(dict.fromkeys([*self._global, *(key for store in self.stores for key in store.keys())]))

    @property
    def node_types(self):
        return list(self._nodes)

    @property
    def edge_types(self):
        return list(self._edges)

    @property
    def node_stores(self):
        return list(self._nodes.values())

    @property
    def edge_stores(self):
        return list(self._edges.values())

    @property
    def stores(self):
        return [self._global_view] + self.node_stores + self.edge_stores

    @property
    def num_nodes(self):
        counts = [store.num_nodes for store in self.node_stores]
        return None if any(value is None for value in counts) else sum(counts)

    @property
    def num_edges(self):
        return sum(store.num_edges for store in self.edge_stores)

    def metadata(self):
        return self.node_types, self.edge_types

    def collect(self, key, allow_empty=False):
        values = {}
        for name, store in (*self._nodes.items(), *self._edges.items()):
            if hasattr(store, key):
                values[name] = getattr(store, key)
        if not values and not allow_empty:
            raise KeyError('no heterogeneous stores contain ' + str(key))
        return values

    def to_dict(self):
        return {'_global_store': dict(self._global), **{key: store.to_dict() for key, store in self._nodes.items()},
                **{key: store.to_dict() for key, store in self._edges.items()}}

    def to_homogeneous(self, node_attrs=None, edge_attrs=None, add_node_type=True, add_edge_type=True, dummy_values=True):
        """Concatenate typed stores with COO offsets and optional int64 type IDs.

        Floating fields keep native gradients. Missing fields use NaN/-1/False
        dummies; differing rank-two node widths are zero padded. Field dtypes
        must match; scalar store metadata and custom index fields are unsupported.
        """
        from ._graph_convert import to_homogeneous
        return to_homogeneous(self, node_attrs, edge_attrs, add_node_type, add_edge_type, dummy_values)

    def validate(self, raise_on_error=True):
        if type(raise_on_error) is not bool:
            raise TypeError('raise_on_error must be bool')
        try:
            for key, store in self._nodes.items():
                count = store.num_nodes
                if count is None:
                    raise ValueError('num_nodes is undefined for node type ' + key)
                for name, value in store.items():
                    if name in ('x', 'pos', 'node_attr', 'node_type') and not value.shape:
                        raise ValueError('node features require a node axis')
                    if isinstance(value, Tensor) and value.shape and value.shape[0] != count:
                        raise ValueError('node field row count differs from num_nodes: ' + key + '.' + name)
            for (src, relation, dst), store in self._edges.items():
                if src not in self._nodes or dst not in self._nodes:
                    raise ValueError('relation references an undefined node type')
                size = (self._nodes[src].num_nodes, self._nodes[dst].num_nodes)
                if 'edge_index' in store:
                    _edge_parts(store.edge_index, size=size)
                count = store.num_edges
                for name, value in store.items():
                    if name in ('edge_attr', 'edge_weight') and not value.shape:
                        raise ValueError('edge features require an edge axis')
                    if name != 'edge_index' and value.shape and value.shape[0] != count:
                        raise ValueError('edge field row count differs from num_edges: ' + relation + '.' + name)
            return True
        except (TypeError, ValueError):
            if raise_on_error:
                raise
            return False

    def subgraph(self, subset_dict):
        """Select typed nodes and induced relations, preserving omitted nodes.

        Each selected type takes ordered int64 IDs or a bool mask. Repeated
        nodes use their last occurrence for edge relabeling. Eager CPU only;
        batched, sampled and custom-index schemas require separate remapping.
        """
        from ._graph_subgraph import subgraph
        return subgraph(self, subset_dict)

    def edge_subgraph(self, subset_dict):
        """Select ordered edges per full relation key, retaining all nodes."""
        from ._graph_subgraph import edge_subgraph
        return edge_subgraph(self, subset_dict)

    def clone(self, *args):
        """Clone selected tensor fields, retaining stores and isolated batch metadata."""
        from ._graph_clone import clone
        return clone(self, *args)

    def apply(self, func, *args):
        """Replace selected tensor fields sequentially across global, node and edge stores.

        Node counts and private batch metadata stay fixed. Callbacks own graph
        alignment and partial side effects if a later call fails.
        """
        from ._graph_apply import apply
        return apply(self, func, *args)

    def detach(self, *args):
        """Replace selected slots with detached views; external handles stay live."""
        from ._graph_apply import detach
        return detach(self, *args)

    def detach_(self, *args):
        """True in-place Tensor detachment is unsupported."""
        from ._graph_apply import detach_
        return detach_(self, *args)

    def contiguous(self, *args):
        """Make selected tensor fields contiguous using native ownership rules."""
        from ._graph_apply import contiguous
        return contiguous(self, *args)

    def cpu(self, *args):
        """Return this CPU graph without changing selected tensor handles."""
        from ._graph_apply import cpu
        return cpu(self, *args)

    def to(self, device, *args, non_blocking=False):
        """Accept an identity CPU conversion; device migration is unsupported."""
        from ._graph_apply import to
        return to(self, device, *args, non_blocking=non_blocking)


class _RelationModules(Module):
    _builtin_init = True
    relations: tuple

    def __init__(self, convs):
        self._initialize()
        object.__setattr__(self, 'relations', tuple(convs))
        names = ['<' + '___'.join(key).replace('.', '#') + '>' for key in self.relations]
        if len(set(names)) != len(names):
            raise ValueError('relation names collide after PyG module-key encoding')
        object.__setattr__(self, '_names', dict(zip(self.relations, names)))
        for key, module in convs.items():
            setattr(self, self._names[key], module)

    def __getitem__(self, key):
        return getattr(self, self._names[_relation(key)])

    def __iter__(self):
        return iter(self.relations)

    def __len__(self):
        return len(self.relations)

    def keys(self):
        return list(self.relations)

    def items(self):
        return ((key, self[key]) for key in self.relations)

    def values(self):
        return (self[key] for key in self.relations)


class HeteroConv(Module):
    """Apply registered relation Modules, then group outputs by destination type.

Aggregations are sum/mean/min/max/cat or None (stack relations on axis one).
Max/min ties select the first relation, matching PyG's group reduction. Only
destinations of relations present in input dictionaries appear in the output.
Relation order and shared child aliases participate in checkpoint compatibility.
"""
    _builtin_init = True
    aggr: object

    def __init__(self, convs, aggr='sum'):
        if not isinstance(convs, Mapping):
            raise TypeError('convs must map relation triples to Ferro Modules')
        if aggr not in ('sum', 'mean', 'min', 'max', 'cat', None):
            raise NotImplementedError('HeteroConv aggregation supports sum/mean/min/max/cat/None')
        for key, module in convs.items():
            src, _, dst = _relation(key)
            if not isinstance(module, Module):
                raise TypeError('relation convolutions must be Ferro Modules')
            if src != dst and getattr(module, 'add_self_loops', False):
                raise ValueError('self-loop insertion is invalid for a bipartite relation')
        self._initialize()
        object.__setattr__(self, 'aggr', aggr)
        self.convs = _RelationModules(convs)

    def reset_parameters(self):
        for module in self.convs.values():
            reset = getattr(module, 'reset_parameters', None)
            if callable(reset):
                reset()

    def forward(self, *args_dict, **kwargs_dict):
        if any(not isinstance(value, Mapping) for value in (*args_dict, *kwargs_dict.values())):
            raise TypeError('HeteroConv arguments must be dictionaries indexed by node or relation type')
        if any(not key.endswith('_dict') for key in kwargs_dict):
            raise ValueError('HeteroConv keyword arguments must end with _dict')
        outputs = {}
        for key, module in self.convs.items():
            src, _, dst = key
            args, kwargs, edge_argument = [], {}, False
            for name, mapping in [(None, value) for value in args_dict] + list(kwargs_dict.items()):
                if key in mapping:
                    value = mapping[key]
                    edge_argument = True
                elif src == dst and src in mapping:
                    value = mapping[src]
                elif src in mapping or dst in mapping:
                    value = (mapping.get(src), mapping.get(dst))
                else:
                    continue
                if name is None:
                    args.append(value)
                else:
                    kwargs[name[:-5]] = value
            if not edge_argument:
                continue
            value = module(*args, **kwargs)
            _cpu_tensor(value, 'relation output')
            if not value.shape:
                raise ValueError('relation outputs require a node axis')
            outputs.setdefault(dst, []).append(value)
        result = {}
        for key, values in outputs.items():
            if self.aggr is None:
                result[key] = stack(values, 1)
            elif len(values) == 1:
                result[key] = values[0]
            elif self.aggr == 'cat':
                result[key] = cat(values, -1)
            else:
                combined = stack(values, 0)
                if self.aggr == 'sum':
                    result[key] = combined.sum_dim(0)
                elif self.aggr == 'mean':
                    result[key] = combined.mean_dim(0)
                elif self.aggr == 'max':
                    result[key] = combined.topk(1, 0)[0].squeeze(0)
                else:
                    result[key] = -(-combined).topk(1, 0)[0].squeeze(0)
        return result


class RGCNConv(Module):
    """CPU float32 relational GCN with per-relation sum/mean and optional bases.

Features are homogeneous or (source, target) rank-two matrices; edge_type is
CPU int64 [E]. Mean divides by each destination's degree within each relation,
then sums relation outputs. Sparse layouts, embedding-ID inputs and block
decomposition are unsupported. is_sorted is an optimization hint only: this
implementation explicitly groups relation IDs and needs no sorting assumption.
"""
    _builtin_init = True
    in_channels: object
    out_channels: int
    num_relations: int
    num_bases: object
    num_blocks: object
    aggr: str
    root_weight: bool
    is_sorted: bool

    def __init__(self, in_channels, out_channels, num_relations, num_bases=None, num_blocks=None,
                 aggr='mean', root_weight=True, is_sorted=False, bias=True, **kwargs):
        if type(in_channels) is int:
            widths = (in_channels, in_channels)
        elif isinstance(in_channels, (tuple, list)) and len(in_channels) == 2:
            widths = tuple(in_channels)
            in_channels = widths
        else:
            raise ValueError('in_channels must be an integer or source/target width pair')
        for width in widths:
            _positive(width, 'in_channels')
        _positive(out_channels, 'out_channels')
        _positive(num_relations, 'num_relations')
        if num_bases is not None:
            _positive(num_bases, 'num_bases')
        if num_blocks is not None:
            raise NotImplementedError('RGCNConv block decomposition is unsupported')
        if aggr not in ('mean', 'sum', 'add'):
            raise NotImplementedError('RGCNConv supports per-relation mean/sum/add only')
        _flags(root_weight=root_weight, is_sorted=is_sorted, bias=bias)
        if kwargs and (set(kwargs) != {'flow'} or kwargs['flow'] != 'source_to_target'):
            raise NotImplementedError('RGCNConv supports only source_to_target flow and no other MessagePassing options')
        _configure(self, in_channels=in_channels, out_channels=out_channels, num_relations=num_relations,
                   num_bases=num_bases, num_blocks=None, aggr=aggr, root_weight=root_weight, is_sorted=is_sorted)
        self._widths = widths
        count = num_relations if num_bases is None else num_bases
        _capacity((count, widths[0], out_channels))
        if num_bases is not None:
            _capacity((num_relations, num_bases))
            _capacity((num_relations, widths[0], out_channels))
        if root_weight:
            _capacity((widths[1], out_channels))
        self.weight = Parameter(Tensor.zeros([count, widths[0], out_channels]))
        self.comp = None if num_bases is None else Parameter(Tensor.zeros([num_relations, num_bases]))
        self.root = Parameter(Tensor.zeros([widths[1], out_channels])) if root_weight else None
        self.bias = Parameter(Tensor.zeros([out_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            for name, parameter in self.named_parameters():
                shape = list(parameter.tensor().shape)
                value = Tensor.zeros(shape) if name == 'bias' else (Tensor.rand(shape) * 2 - 1) * (6. / (shape[-2] + shape[-1])) ** .5
                parameter.tensor().copy_(value)

    def forward(self, x, edge_index, edge_type=None):
        if isinstance(x, Tensor):
            source = target = x
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            source, target = x
        else:
            raise TypeError('RGCNConv requires feature matrices; embedding-ID inputs are unsupported')
        _features(source, 'source features', self._widths[0])
        _features(target, 'target features', self._widths[1] if self.root_weight else None)
        src, dst, size = _edge_parts(edge_index, size=(source.shape[0], target.shape[0]))
        _capacity((size[1], self.out_channels))
        _capacity((size[1], self._widths[0]))
        _capacity((len(src), self._widths[0]))
        _cpu_tensor(edge_type, 'edge_type', 'int64')
        if edge_type.shape != [len(src)]:
            raise ValueError('edge_type must have shape [E]')
        relations = edge_type.tolist()
        if any(value < 0 or value >= self.num_relations for value in relations):
            raise ValueError('edge_type contains an out-of-range relation ID')
        weight = self.weight.tensor()
        if self.num_bases is not None:
            weight = (self.comp.tensor() @ weight.reshape([self.num_bases, self._widths[0] * self.out_channels])).reshape([self.num_relations, self._widths[0], self.out_channels])
        output = Tensor.zeros([size[1], self.out_channels])
        for relation in range(self.num_relations):
            selected = [index for index, kind in enumerate(relations) if kind == relation]
            sources, targets = [src[index] for index in selected], [dst[index] for index in selected]
            aggregate = scatter(source.index_select(0, sources), targets, size[1], 'sum' if self.aggr == 'add' else self.aggr)
            matrix = weight.index_select(0, [relation]).reshape([self._widths[0], self.out_channels])
            output = output + aggregate @ matrix
        if self.root is not None:
            output = output + target @ self.root.tensor()
        if self.bias is not None:
            output = output + self.bias.tensor()
        return output
