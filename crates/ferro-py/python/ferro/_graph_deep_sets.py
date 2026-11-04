"""Deep Sets readouts over registered native feature networks."""
from ._native import scatter
from ._graph_aggregation import _Aggregation, _layout
from ._graph_conv import _configure, _features, _reset_network
from ._graph_utils import _capacity, _count
from . import nn
from ._graph_mlp import MLP


_PRESERVING = (nn.Identity, nn.Dropout, nn.ReLU, nn.Sigmoid, nn.Tanh,
               nn.GELU, nn.SiLU, nn.Mish, nn.ELU, nn.CELU, nn.SELU,
               nn.LeakyReLU, nn.ReLU6, nn.Hardtanh, nn.Hardsigmoid,
               nn.Hardswish, nn.Hardshrink, nn.Softshrink, nn.Tanhshrink,
               nn.Softplus, nn.Softsign, nn.LogSigmoid, nn.Threshold)


def _network_width(module, width, rows=None):
    if module is None:
        return width
    if isinstance(module,MLP):
        width=module._metadata(width,rows,strict_state=True) if rows is not None else module.get_out_channels(width)
    elif type(module) is nn.Sequential:
        for child in module:
            width = _network_width(child, width, rows)
    elif type(module) is nn.Linear:
        if module.in_features != width:
            raise ValueError('network Linear input width does not match aggregation features')
        width = module.out_features
    elif type(module) in _PRESERVING:
        pass
    elif type(module) in (nn.Softmax, nn.LogSoftmax, nn.Softmin):
        if module.dim not in (0, 1, -1, -2):
            raise ValueError('network softmax dimension must address a rank-two input')
    elif type(module) in (nn.LayerNorm, nn.RMSNorm):
        shape = module.normalized_shape
        shape = (shape,) if type(shape) is int else tuple(shape)
        if len(shape) not in (1, 2) or shape[-1] != width or (len(shape) == 2 and rows is not None and shape[0] != rows):
            raise ValueError('network normalization shape does not match aggregation features')
    elif type(module) is nn.BatchNorm1d:
        if module.num_features != width:
            raise ValueError('network BatchNorm input width does not match aggregation features')
        if rows == 1 and (module.training or module.running_mean is None or module.running_var is None):
            raise ValueError('network BatchNorm requires more than one row while using input statistics')
    else:
        metadata = getattr(module, 'get_out_channels', None)
        if not callable(metadata):
            raise NotImplementedError('custom DeepSets networks must declare get_out_channels(in_channels)')
        width = _count(metadata(width), 'custom network output width')
    if rows is not None:
        _capacity((rows, width))
    return width


class DeepSetsAggregation(_Aggregation):
    """Transform rows, sum each group, then transform the group readouts.

    CPU float32 rank-two inputs support unsorted indices and CSR pointers.
    Native Linear/Sequential, activations, Dropout and feature normalization
    have checked output widths. Other Modules must declare get_out_channels
    and preserve the row count; their internal behavior remains their own
    contract. No speculative execution or tracing is used to infer shapes.
    Supplied Modules retain their weights, aliases, mode and random streams.
    Metadata, including known BatchNorm singleton failures, is validated before
    either network runs. MLPs receive the original index and dim_size as PyG
    does; CSR-only calls pass batch=None. A graph-normalized global MLP therefore
    rejects mismatched original node/group index lengths before local work.
    """
    _builtin_init = True

    def __init__(self, local_nn=None, global_nn=None):
        for name, module in (('local_nn', local_nn), ('global_nn', global_nn)):
            if module is not None and not isinstance(module, nn.Module):
                raise TypeError(name + ' must be a native Module or None')
        _configure(self)
        self.local_nn=None if isinstance(local_nn,MLP) else local_nn
        self.global_nn=None if isinstance(global_nn,MLP) else global_nn
        if isinstance(local_nn,MLP): self.local_mlp=local_nn
        if isinstance(global_nn,MLP): self.global_mlp=global_nn

    def _networks(self):
        return getattr(self,'local_mlp',self.local_nn),getattr(self,'global_mlp',self.global_nn)

    def reset_parameters(self):
        for module in self._networks():
            if module is not None:
                _reset_network(module)

    def get_out_channels(self, in_channels):
        local,global_=self._networks()
        width = _network_width(local, _count(in_channels, 'in_channels'))
        return _network_width(global_, width)

    def _preflight(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, groups = _layout(x, index, ptr, dim_size, dim)
        local,global_=self._networks()
        local_width = _network_width(local, x.shape[1], x.shape[0])
        if isinstance(local,MLP): local._metadata(x.shape[1],x.shape[0],index,dim_size,True)
        _capacity((groups, local_width))
        global_width = _network_width(global_, local_width, groups)
        if isinstance(global_,MLP): global_._metadata(local_width,groups,index,dim_size,True)
        _capacity((groups, global_width))
        return ids, groups, local_width, global_width

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, groups, local_width, global_width = self._preflight(x, index, ptr, dim_size, dim)
        local_module,global_module=self._networks()
        local=x if local_module is None else local_module(x,batch=index,batch_size=dim_size) if isinstance(local_module,MLP) else local_module(x)
        _features(local, 'local network output', local_width)
        if local.shape[0] != x.shape[0]:
            raise ValueError('local network must preserve the input row count')
        output = scatter(local, ids, groups, 'sum')
        output=output if global_module is None else global_module(output,batch=index,batch_size=dim_size) if isinstance(global_module,MLP) else global_module(output)
        _features(output, 'global network output', global_width)
        if output.shape[0] != groups:
            raise ValueError('global network must preserve the group row count')
        return output
