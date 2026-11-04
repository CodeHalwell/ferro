"""Canonical edge, node and global graph-network callbacks over native tensors."""
from ._native import EdgeIndex, Tensor
from ._graph_conv import _configure, _features
from ._graph_conditioned import _eager
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, _edge_tensor, _index_ids
from .nn import Module


class MetaLayer(Module):
    """Run registered edge, node and global update Modules in that order.

    Inputs use rank-two CPU float32 features and explicit homogeneous COO.
    Native EdgeIndex inputs are converted to COO for callbacks. An edge's
    batch ID comes from its source node, including cross-group edges. Edge
    and node updates preserve row counts; a global update preserves supplied
    global rows or the groups implied by batch. Optional features may be None.
    Custom callback execution is ordinary eager Module execution, not tracing;
    callbacks own their shape/state contracts beyond these outer checks.
    """
    _builtin_init = True

    def __init__(self, edge_model=None, node_model=None, global_model=None):
        if any(value is not None and not isinstance(value, Module) for value in (edge_model, node_model, global_model)):
            raise TypeError('MetaLayer callbacks must be Ferro Modules or None')
        _configure(self)
        self.edge_model, self.node_model, self.global_model = edge_model, node_model, global_model
        self.reset_parameters()

    def reset_parameters(self):
        for model in (self.node_model, self.edge_model, self.global_model):
            reset = getattr(model, 'reset_parameters', None)
            if callable(reset): reset()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        _eager()
        _features(x, 'x')
        nodes, width = x.shape
        _capacity((nodes + 1,), 8)
        _capacity(x.shape)
        if isinstance(edge_index, Tensor): _capacity(edge_index.shape, 8)
        if batch is not None:
            _cpu_tensor(batch, 'batch', 'int64')
            if batch.shape != [nodes]: raise ValueError('batch must contain one entry per node')
        source, target, _ = _edge_parts(edge_index, num_nodes=nodes)
        edges = len(source)
        _capacity((edges, width))
        if edge_attr is not None:
            _features(edge_attr, 'edge_attr')
            if edge_attr.shape[0] != edges: raise ValueError('edge_attr must contain one row per edge')
            _capacity(edge_attr.shape)
        count = None
        if u is not None:
            _features(u, 'u')
            _capacity(u.shape)
            count = u.shape[0]
        if batch is not None:
            ids = _index_ids(batch, 'batch', nodes)
            required = max(ids, default=-1) + 1
            if count is not None and count < required: raise ValueError('batch contains a group outside global features')
            if count is None: count = required
            _capacity((count + 1,), 8)
        if isinstance(edge_index, EdgeIndex): edge_index = _edge_tensor(source, target)
        if self.edge_model is not None:
            edge_attr = self.edge_model(x.index_select(0, source), x.index_select(0, target), edge_attr, u,
                                       None if batch is None else batch.index_select(0, source))
            if edge_attr is not None:
                _features(edge_attr, 'edge_model output')
                if edge_attr.shape[0] != edges: raise ValueError('edge_model must preserve edge rows')
                _capacity(edge_attr.shape)
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)
            _features(x, 'node_model output')
            if x.shape[0] != nodes: raise ValueError('node_model must preserve node rows')
            _capacity(x.shape)
        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)
            if u is not None:
                _features(u, 'global_model output')
                if count is not None and u.shape[0] != count: raise ValueError('global_model must preserve graph rows')
                _capacity(u.shape)
        return x, edge_attr, u
