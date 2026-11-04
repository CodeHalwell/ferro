"""Typed incidence cleanup over homogeneous and heterogeneous graph containers."""
from ._native import Tensor,EdgeIndex,_is_capturing
from ._graph_data import Data,Batch,_custom_hooks,_has_node_count
from ._graph_hetero import HeteroData
from ._graph_hetero_batch import HeteroBatch
from ._graph_subgraph import _copy,_fields,_topology,_gather_plan,_gather
from ._graph_utils import _capacity,_count,_edge_tensor


def _supported(data):
    if _is_capturing():raise NotImplementedError('graph cleanup requires eager execution')
    from ._graph_sampling import _SampledData
    from ._graph_hetero_sampling import _SampledStore
    if isinstance(data,(Batch,HeteroBatch,_SampledData))or isinstance(data,HeteroData)and any(isinstance(store,_SampledStore)for store in data.stores):
        raise NotImplementedError('extract individual unsampled graphs before cleanup; batch/sampler inverse metadata is not remapped')
    if isinstance(data,Data):
        if _custom_hooks(data):raise NotImplementedError('graph cleanup with custom batching hooks is unsupported')
        _fields(data._store)
    elif isinstance(data,HeteroData):
        if any(hasattr(type(data),hook)for hook in ('__inc__','__cat_dim__')):raise NotImplementedError('graph cleanup with custom heterogeneous hooks is unsupported')
        for store in data.stores:_fields(store._mapping)
    else:raise TypeError('graph cleanup requires Data or HeteroData')


def _nodes(mapping,ids):
    _capacity((len(ids),),8)
    lookup={old:new for new,old in enumerate(ids)}
    return [lookup[node]for node in mapping]


def _homogeneous(data):
    if not _has_node_count(data):raise ValueError('isolated-node cleanup requires an inferable or explicit num_nodes')
    if data.edge_index is not None:_capacity((2,data.edge_index.num_edges)if isinstance(data.edge_index,EdgeIndex)else data.edge_index.shape,8)
    n=_count(data.num_nodes,'num_nodes')
    source,target=([],[])if data.edge_index is None else _topology(data.edge_index,(n,n))
    ids=sorted(set(source)|set(target));edges=len(source);_capacity((2,edges),8)
    anchored=data._num_nodes is not None or any(key in data for key in ('x','pos','node_attr','node_type'))
    plans={};roles=dict(data._field_roles)
    for key,value in data.items():
        if key=='edge_index':continue
        explicit=roles.get(key)
        if explicit=='node':
            if not value.shape or value.shape[0]!=n:raise ValueError('declared node field must have one row per node')
            take=True
        elif explicit is not None:
            if explicit=='edge'and(not value.shape or value.shape[0]!=edges):raise ValueError('declared edge field must have one row per edge')
            take=False
        else:
            if not anchored and key not in ('edge_attr','edge_weight')and value.shape:
                raise NotImplementedError('topology-only node counts require explicit roles for other tensor fields')
            take=bool(value.shape)and value.shape[0]==n and(n!=edges or 'edge'not in key)
        if take:plans[key]=_gather_plan(value,ids);roles[key]='node'
        elif explicit is None:
            roles[key]='edge'if value.shape and value.shape[0]==edges and 'edge'in key else 'graph'
    edge=None if data.edge_index is None else _edge_tensor(_nodes(source,ids),_nodes(target,ids))
    fields={key:_gather(plan)for key,plan in plans.items()}
    if edge is not None:data._store['edge_index']=edge
    data._store.update(fields);data._field_roles=roles
    if data._num_nodes is not None or data.num_nodes!=len(ids):data._num_nodes=len(ids)
    return data


def _heterogeneous(data):
    counts={key:store.num_nodes for key,store in data._nodes.items()}
    if any(value is None for value in counts.values()):raise ValueError('isolated-node cleanup requires known counts for every node type')
    counts={key:_count(value,'num_nodes')for key,value in counts.items()};incident={key:set()for key in counts};topology={}
    for key,store in data._edges.items():
        if key[0]not in counts or key[2]not in counts:raise ValueError('relations must name existing node stores')
        if 'edge_index'not in store:
            if store.num_edges:raise NotImplementedError('nonempty relation stores require edge_index for cleanup')
            continue
        source,target=_topology(store.edge_index,(counts[key[0]],counts[key[2]]));topology[key]=(source,target)
        incident[key[0]].update(source);incident[key[2]].update(target)
    ids={key:sorted(nodes)for key,nodes in incident.items()};plans={}
    for key,store in data._nodes.items():
        for field,value in store.items():
            if field=='num_nodes':continue
            if not isinstance(value,Tensor):raise TypeError('node fields must be native tensors')
            if value.shape:
                if value.shape[0]!=counts[key]:raise ValueError('node-store fields must align with the node count')
                plans[key,field]=_gather_plan(value,ids[key])
    for key,store in data._edges.items():
        e=len(topology[key][0])if key in topology else 0
        for field,value in store.items():
            if field!='edge_index'and value.shape and value.shape[0]!=e:raise ValueError('edge-store fields must align with the edge count')
    edges={key:_edge_tensor(_nodes(source,ids[key[0]]),_nodes(target,ids[key[2]]))for key,(source,target)in topology.items()}
    fields={key:_gather(plan)for key,plan in plans.items()}
    for key,value in edges.items():data._edges[key]._mapping['edge_index']=value
    for(key,field),value in fields.items():data._nodes[key]._mapping[field]=value
    for key,store in data._nodes.items():
        if 'num_nodes'in store:store._mapping['num_nodes']=len(ids[key])
    return data


class RemoveIsolatedNodes:
    """Remove nodes absent from every edge, preserving ordered edges and fields.

    Self-loops count as incidence. Node IDs are sorted before compact remapping,
    across all relations for each heterogeneous type. CPU F32/F64/I64/Bool node
    fields use native gathers, while edge fields remain aliased. Homogeneous
    inferred roles follow PyG's row-count and edge-name rule, including node y;
    explicit Ferro field_roles take precedence. Unknown topology-only custom
    roles reject rather than depending on an already-remapped node count.

    Calls shallow-copy containers/stores; forward commits to the supplied
    container after metadata validation and every gather succeeds. Unrelated
    graph/private metadata stays shared. Batched/sampled graphs, custom hooks
    and indexed supervision require separate remapping and reject. No RNG,
    geometric search or additional differentiation engine is introduced.
    """
    def __call__(self,data):
        _supported(data)
        return self.forward(_copy(data))
    def forward(self,data):
        _supported(data)
        return _homogeneous(data)if isinstance(data,Data)else _heterogeneous(data)
    def __repr__(self):return self.__class__.__name__+'()'
