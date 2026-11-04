"""Bounded dense random-walk positional features with native tensor gradients."""
from ._native import Tensor,EdgeIndex,scatter,stack,cat,_is_capturing
from ._graph_data import Data,_RESERVED,_has_node_count
from ._graph_subgraph import _copy,_topology
from ._graph_utils import _capacity,_count,_cpu_tensor
from .linalg import diagonal


class AddRandomWalkPE:
    """Match PyG's dense path for homogeneous CPU graphs with at most 2000 nodes.

    The edge value is reciprocal clamped weighted out-degree, without a weight
    numerator, as in PyG. Parallel pairs overwrite. Live weighted duplicates
    reject: PyG's indexed overwrite sends gradients to every duplicate while
    native scatter differentiates the last writer. Unweighted/detached-weight
    duplicates retain reference values. Weights are F32; concatenation casts
    positional values to x's F32/F64/I64/Bool dtype. Zero walk_length produces
    one column, matching the reference. Negative lengths, sparse large graphs,
    heterogeneous data and capture reject. A fixed 2**27 dense-work budget and
    typed capacity checks precede allocation. No RNG is consumed.

    Calls shallow-copy Data/Batch containers and keep untouched fields/private
    inverse metadata shared. forward updates its supplied container. New fields
    receive explicit node roles; existing Batch slice metadata is not extended.
    Floating first-order gradients use native algebra; higher/forward AD remains
    bounded by the existing scatter operation's explicit unsupported scope.
    """
    _MAX_NODES=2000
    _MAX_WORK=1<<27

    def __init__(self,walk_length,attr_name='random_walk_pe'):
        self.walk_length=_count(walk_length,'walk_length')
        if attr_name is not None and not isinstance(attr_name,str):raise TypeError('attr_name must be a string or None')
        self.attr_name=attr_name

    def __repr__(self):return self.__class__.__name__+'()'

    def __call__(self,data):
        if not isinstance(data,Data):raise TypeError('AddRandomWalkPE requires homogeneous Data or Batch')
        return self.forward(_copy(data))

    def forward(self,data):
        if _is_capturing():raise NotImplementedError('AddRandomWalkPE requires eager execution')
        if not isinstance(data,Data):raise TypeError('AddRandomWalkPE requires homogeneous Data or Batch')
        if data.edge_index is None or not _has_node_count(data):raise ValueError('AddRandomWalkPE requires topology and a known node count')
        steps=max(1,_count(self.walk_length,'walk_length'))
        name='x' if self.attr_name is None else self.attr_name
        if not isinstance(name,str)or name.startswith('_')or name in _RESERVED or name=='edge_index'or hasattr(type(data),name):raise ValueError('unsupported or reserved positional field name')
        topology=data.edge_index
        if isinstance(topology,EdgeIndex):edges=topology.num_edges
        else:
            _cpu_tensor(topology,'edge_index','int64')
            if topology.ndim!=2 or topology.shape[0]!=2:raise ValueError('edge_index must have shape [2,E]')
            edges=topology.shape[1]
        _capacity((2,edges),8)
        if edges>self._MAX_WORK:raise ValueError('AddRandomWalkPE topology work capacity exceeded')
        nodes=data.num_nodes
        if nodes>self._MAX_NODES:raise NotImplementedError('AddRandomWalkPE supports the dense path with at most 2000 nodes')
        _capacity((nodes,nodes));_capacity((nodes,steps));_capacity((edges,),8)
        if edges+nodes*nodes+nodes*steps+(steps-1)*nodes**3>self._MAX_WORK:raise ValueError('AddRandomWalkPE dense work capacity exceeded')
        source,target=_topology(topology,(nodes,nodes))
        weight=data._store.get('edge_weight')
        if weight is not None:
            _cpu_tensor(weight,'edge_weight')
            if weight.shape!=[edges]:raise ValueError('edge_weight must have shape [E]')
            _capacity(weight.shape)
            if weight.requires_grad and len(set(zip(source,target)))!=edges:raise NotImplementedError('live edge weights with parallel pairs require unsupported overwrite gradients')
        x=data.x if self.attr_name is None else None
        if x is not None:
            _cpu_tensor(x,'x',dtype=None)
            if x.dtype not in ('float32','float64','int64','bool')or x.ndim not in (1,2)or x.shape[0]!=nodes:raise ValueError('x must be a supported typed [N] or [N,F] field')
            width=1 if x.ndim==1 else x.shape[1]
            size=8 if x.dtype in ('float64','int64') else 1 if x.dtype=='bool' else 4
            _capacity((nodes,width+steps),size);_capacity((nodes,steps),size)
        values=Tensor.ones([edges])if weight is None else weight
        degree=scatter(values.reshape([edges,1]),source,nodes,'sum').reshape([nodes]).clamp(1.,float('inf'))
        values=1./degree.index_select(0,source)
        index=Tensor.from_i64([s*nodes+t for s,t in zip(source,target)],[edges])
        adjacency=Tensor.zeros([nodes*nodes]).scatter(0,index,values).reshape([nodes,nodes])
        if nodes==0:
            pe=diagonal(adjacency).reshape([0,1]).expand([0,steps]).clone()
        else:
            output=adjacency;columns=[diagonal(output)]
            for _ in range(steps-1):
                output=output@adjacency;columns.append(diagonal(output))
            pe=stack(columns,dim=-1)
        if x is not None:
            feature=x.reshape([nodes,1])if x.ndim==1 else x
            pe=cat([feature,pe.to(dtype=x.dtype)],dim=-1)
        data[name]=pe
        data._field_roles[name]='node'
        return data
