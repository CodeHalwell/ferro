"""Eager CPU COO message passing with inspected Python method signatures.

Custom layers use ordinary Python __init__/forward/message/aggregate/update
methods and existing native tensor/autograd operations. No tracing, generated
code, sparse fusion, explanation masks or feature decomposition is simulated.
"""
import inspect
import weakref

from ._native import Tensor, EdgeIndex, cat, scatter
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _index_ids
from ._graph_conv import _configure
from .nn import Module


_SPECIAL = frozenset(('edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'index',
                      'ptr', 'size', 'size_i', 'size_j', 'dim_size'))
_MISSING = inspect.Parameter.empty


class _HookHandle:
    def __init__(self, module, kind, identity):
        self._module, self._kind, self.id = weakref.ref(module), kind, identity

    def remove(self):
        module = self._module()
        if module is not None:
            module._mp_hooks[self._kind].pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


class MessagePassing(Module):
    """Subclassable native-op graph propagation with PyG-style argument routing.

    Lifted ``name_i``/``name_j`` arguments select destination/source rows from
    ``name`` supplied to propagate. A (source,target) pair supports bipartite
    graphs and missing endpoints; explicit size includes isolated destinations.
    Both flow directions and any valid node_dim are supported. Built-in
    reductions are sum/add/mean/min/max, or a sequence concatenated on the last
    feature dimension. aggr=None requires a custom aggregate implementation.

    Tensor inputs require CPU storage; lifted float64 values reject. Default
    aggregation requires f32 messages. Integer/bool node IDs can be lifted for
    custom embedding/mask computations. Duplicate edges remain independent.
    Native min/max ties share gradients among actual messages, including the
    documented difference from PyG's initial-zero tie for zero extrema.

    Subclasses may define __init__ and must call super().__init__. Ordinary
    public scalar/tuple attributes enter checkpoints, as with other built-ins;
    annotated configuration remains immutable. Hook functions and arbitrary
    Python method code are caller-owned behavior and are not serialized.
    """
    _builtin_init = True
    aggr: object
    flow: str
    node_dim: int
    decomposed_layers: int

    def __init_subclass__(cls, **kwargs):
        # Only MessagePassing descendants opt into handwritten constructors.
        cls._builtin_init = True
        super().__init_subclass__(**kwargs)
        signature = inspect.signature(cls.__init__)
        cls.__signature__ = signature.replace(parameters=list(signature.parameters.values())[1:])

    def __init__(self, aggr='sum', *, aggr_kwargs=None, flow='source_to_target', node_dim=-2, decomposed_layers=1):
        if aggr is not None:
            reductions = (aggr,) if type(aggr) is str else tuple(aggr) if isinstance(aggr, (tuple, list)) else ()
            if not reductions or any(reduction not in ('sum', 'add', 'mean', 'min', 'max') for reduction in reductions):
                raise NotImplementedError('MessagePassing supports sum/add/mean/min/max or a nonempty sequence')
            aggr = reductions[0] if type(aggr) is str else reductions
        if aggr_kwargs:
            raise NotImplementedError('custom aggregation options are unsupported; override aggregate instead')
        if aggr_kwargs is not None and not isinstance(aggr_kwargs, dict):
            raise TypeError('aggr_kwargs must be a dict or None')
        if flow not in ('source_to_target', 'target_to_source'):
            raise ValueError('flow must be source_to_target or target_to_source')
        if type(node_dim) is not int:
            raise TypeError('node_dim must be an integer')
        if type(decomposed_layers) is not int or decomposed_layers != 1:
            raise NotImplementedError('feature decomposition is unsupported; use decomposed_layers=1')
        _configure(self, aggr=aggr, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)
        self._mp_hooks = {kind + suffix: {} for kind in ('propagate', 'message', 'aggregate', 'edge_update') for suffix in ('_pre', '_post')}
        self._mp_next_hook = 0
        for name, skip in (('message', False), ('aggregate', True), ('update', True), ('edge_update', False)):
            self._parameters_for(name, skip)

    def _parameters_for(self, method, skip_first=False):
        parameters = list(inspect.signature(getattr(self, method)).parameters.values())
        if skip_first:
            if not parameters or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                raise TypeError(method + ' requires a first positional input parameter')
            parameters = parameters[1:]
        if any(parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) for parameter in parameters):
            raise TypeError(method + ' requires explicit named parameters; *args/**kwargs and positional-only routing are unsupported')
        return parameters

    @property
    def explain(self):
        return None

    @explain.setter
    def explain(self, value):
        if value is not None and value is not False:
            raise NotImplementedError('explanation masks are unsupported')

    def reset_parameters(self):
        """The base framework owns no parameters; subclasses reset their layers."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError('define forward in a MessagePassing subclass or call propagate directly')

    def _node_axis(self, value):
        _cpu_tensor(value, 'node features', dtype=None)
        if value.dtype not in ('float32', 'int64', 'bool'):
            raise ValueError('MessagePassing node features require float32/int64/bool')
        rank = len(value.shape)
        axis = self.node_dim + rank if self.node_dim < 0 else self.node_dim
        if not 0 <= axis < rank:
            raise ValueError('node_dim is out of bounds for the supplied tensor rank')
        return axis

    def _collect(self, parameters, edge_index, size, kwargs):
        source, destination, _ = _edge_parts(edge_index)
        edges = (source, destination)
        if size is None:
            mutable = [edge_index.num_nodes] * 2 if isinstance(edge_index, EdgeIndex) else [None, None]
        else:
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise ValueError('size must be a (source,target) pair with optional None entries')
            mutable = [None if count is None else _count(count, 'size') for count in size]
            if isinstance(edge_index, EdgeIndex):
                if any(count is not None and count != edge_index.num_nodes for count in mutable):
                    raise ValueError('native EdgeIndex size must match its homogeneous node count')
                mutable = [edge_index.num_nodes] * 2
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        names = tuple(dict.fromkeys(parameter.name for parameter in parameters))
        collected, lifted = {}, {}
        def observe(value, side):
            axis = self._node_axis(value)
            count = value.shape[axis]
            if mutable[side] is not None and mutable[side] != count:
                raise ValueError('node tensor size disagrees with inferred or explicit graph size')
            mutable[side] = count
        for name in names:
            if name in _SPECIAL:
                continue
            if name.endswith(('_i', '_j')):
                side = i if name.endswith('_i') else j
                value = kwargs.get(name[:-2], _MISSING)
                if isinstance(value, (tuple, list)):
                    if len(value) != 2:
                        raise ValueError(name[:-2] + ' must be a source/target pair')
                    for endpoint, item in enumerate(value):
                        if item is not None and not isinstance(item, Tensor):
                            raise TypeError('bipartite node values must be Tensors or None')
                        if isinstance(item, Tensor):
                            observe(item, endpoint)
                    value = value[side]
                if isinstance(value, Tensor):
                    observe(value, side)
                    lifted[name] = (value, side)
                elif value is not _MISSING and value is not None:
                    raise TypeError('lifted node arguments must be Tensors, pairs or None')
                collected[name] = value
            else:
                collected[name] = kwargs.get(name, _MISSING)
        resolved = [mutable[side] if mutable[side] is not None else mutable[1 - side] for side in range(2)]
        bounds = tuple(max(edges[side], default=-1) + 1 if count is None else count for side, count in enumerate(resolved))
        _edge_parts(edge_index, size=bounds)
        for name, (value, side) in lifted.items():
            axis = self._node_axis(value)
            shape = list(value.shape)
            shape[axis] = len(source)
            _capacity(shape, 8 if value.dtype == 'int64' else 4)
            collected[name] = value.index_select(axis, edges[side])
        topology = edge_index.to_tensor() if isinstance(edge_index, EdgeIndex) else edge_index
        ids_i, ids_j = Tensor.from_i64(edges[i], [len(source)]), Tensor.from_i64(edges[j], [len(source)])
        collected.update(edge_index=topology, adj_t=None, edge_index_i=ids_i, edge_index_j=ids_j,
                         index=ids_i, ptr=None, size=mutable, size_i=resolved[i], size_j=resolved[j], dim_size=resolved[i])
        return collected, mutable

    def _arguments(self, method, collected, skip_first=False):
        routed = {}
        for parameter in self._parameters_for(method, skip_first):
            value = collected.get(parameter.name, _MISSING)
            if value is _MISSING:
                if parameter.default is _MISSING:
                    raise TypeError('missing required ' + method + ' argument: ' + parameter.name)
                value = parameter.default
            routed[parameter.name] = value
        return routed

    def _before_stage(self, stage, arguments):
        for hook in tuple(self._mp_hooks[stage + '_pre'].values()):
            replacement = hook(self, (arguments,))
            if replacement is not None:
                if isinstance(replacement, tuple) and len(replacement) == 1:
                    replacement = replacement[0]
                if not isinstance(replacement, dict):
                    raise TypeError(stage + ' pre-hook must return a dict or one-element tuple containing a dict')
                arguments = replacement
        return arguments

    def _after_stage(self, stage, arguments, output):
        for hook in tuple(self._mp_hooks[stage + '_post'].values()):
            replacement = hook(self, arguments, output)
            if replacement is not None:
                output = replacement
        return output

    def _before_propagation(self, stage, edge_index, size, kwargs):
        for hook in tuple(self._mp_hooks[stage + '_pre'].values()):
            replacement = hook(self, (edge_index, size, kwargs))
            if replacement is not None:
                if not isinstance(replacement, (tuple, list)) or len(replacement) != 3 or not isinstance(replacement[2], dict):
                    raise TypeError(stage + ' pre-hook must return (edge_index, size, kwargs)')
                edge_index, size, kwargs = replacement
        return edge_index, size, kwargs

    def propagate(self, edge_index, size=None, **kwargs):
        edge_index, size, kwargs = self._before_propagation('propagate', edge_index, size, kwargs)
        parameters = self._parameters_for('message') + self._parameters_for('aggregate', True) + self._parameters_for('update', True)
        collected, mutable = self._collect(parameters, edge_index, size, kwargs)
        message = self._before_stage('message', self._arguments('message', collected))
        output = self.message(**message)
        output = self._after_stage('message', (message,), output)
        aggregate = self._before_stage('aggregate', self._arguments('aggregate', collected, True))
        output = self.aggregate(output, **aggregate)
        output = self._after_stage('aggregate', (aggregate,), output)
        output = self.update(output, **self._arguments('update', collected, True))
        return self._after_stage('propagate', (edge_index, mutable, kwargs), output)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        if self.aggr is None:
            raise NotImplementedError('aggr=None requires overriding aggregate')
        if ptr is not None:
            raise NotImplementedError('MessagePassing default aggregation supports COO index only')
        _cpu_tensor(inputs, 'messages')
        axis = self._node_axis(inputs)
        ids = _index_ids(index, 'index', inputs.shape[axis])
        inferred = max(ids, default=-1) + 1
        count = inferred if dim_size is None else _count(dim_size, 'dim_size')
        if count < inferred:
            raise ValueError('aggregation index exceeds dim_size')
        reductions = (self.aggr,) if type(self.aggr) is str else self.aggr
        shape = list(inputs.shape)
        shape[axis] = count
        if len(reductions) > 1:
            if len(shape) < 2 or axis == len(shape) - 1:
                raise ValueError('multi-aggregation requires a final feature axis distinct from node_dim')
            shape[-1] *= len(reductions)
        _capacity(shape)
        order = [axis] + [dim for dim in range(len(inputs.shape)) if dim != axis]
        inverse = [order.index(dim) for dim in range(len(order))]
        moved = inputs if axis == 0 else inputs.permute(*order)
        outputs = []
        for reduction in reductions:
            reduced = scatter(moved, ids, count, 'sum' if reduction == 'add' else reduction)
            outputs.append(reduced if axis == 0 else reduced.permute(*inverse))
        return outputs[0] if len(outputs) == 1 else cat(outputs, len(shape) - 1)

    def update(self, inputs):
        return inputs

    def edge_updater(self, edge_index, size=None, **kwargs):
        edge_index, size, kwargs = self._before_propagation('edge_update', edge_index, size, kwargs)
        collected, _ = self._collect(self._parameters_for('edge_update'), edge_index, size, kwargs)
        output = self.edge_update(**self._arguments('edge_update', collected))
        return self._after_stage('edge_update', (edge_index, size, kwargs), output)

    def edge_update(self):
        raise NotImplementedError('override edge_update to use edge_updater')

    def message_and_aggregate(self, adj_t):
        raise NotImplementedError('sparse/fused message passing is unsupported')

    def _register_hook(self, kind, hook):
        if not callable(hook):
            raise TypeError('hook must be callable')
        identity = self._mp_next_hook
        self._mp_next_hook += 1
        self._mp_hooks[kind][identity] = hook
        return _HookHandle(self, kind, identity)

    def register_propagate_forward_pre_hook(self, hook):
        return self._register_hook('propagate_pre', hook)

    def register_propagate_forward_hook(self, hook):
        return self._register_hook('propagate_post', hook)

    def register_message_forward_pre_hook(self, hook):
        return self._register_hook('message_pre', hook)

    def register_message_forward_hook(self, hook):
        return self._register_hook('message_post', hook)

    def register_aggregate_forward_pre_hook(self, hook):
        return self._register_hook('aggregate_pre', hook)

    def register_aggregate_forward_hook(self, hook):
        return self._register_hook('aggregate_post', hook)

    def register_edge_update_forward_pre_hook(self, hook):
        return self._register_hook('edge_update_pre', hook)

    def register_edge_update_forward_hook(self, hook):
        return self._register_hook('edge_update_post', hook)


MessagePassing.__signature__ = inspect.signature(MessagePassing.__init__).replace(
    parameters=list(inspect.signature(MessagePassing.__init__).parameters.values())[1:])
