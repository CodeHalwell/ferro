"""Relational graph attention using native eager CPU F32 tensor operations."""
import math
from ._native import Tensor, EdgeIndex, no_grad, scatter, segment_softmax, where
from ._graph_conv import _configure, _features, _flags, _options, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, _edge_tensor, _index_ids
from .nn import Dropout, Linear, Module, Parameter, ReLU


class RGATConv(Module):
    """Learn per-relation messages and within/across-relation attention.

    Rank-two CPU F32 features, typed COO/native homogeneous topology, explicit
    relation IDs, basis/block decomposition and optional live edge attributes
    are supported. Returned attention precedes cardinality changes and Dropout.
    One registered stream controls attention Dropout. Logits must be finite;
    sparse layouts, integer embedding inputs, lazy channels and capture reject.

    PyG 2.8.1 leaves l2 uninitialized because its reset discards torch.full's
    result. This implementation initializes l2 to 1/out_channels. Subsequent
    reset calls preserve l2 and w, matching PyG's actual reset behavior.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    num_relations: int
    num_bases: object
    num_blocks: object
    mod: object
    attention_mechanism: str
    attention_mode: str
    heads: int
    dim: int
    concat: bool
    negative_slope: float
    dropout: float
    edge_dim: object
    aggr: str

    def __init__(self, in_channels, out_channels, num_relations, num_bases=None, num_blocks=None,
                 mod=None, attention_mechanism='across-relation', attention_mode='additive-self-attention',
                 heads=1, dim=1, concat=True, negative_slope=.2, dropout=0., edge_dim=None, bias=True, **kwargs):
        for name,value in [('in_channels',in_channels),('out_channels',out_channels),('num_relations',num_relations),('heads',heads),('dim',dim)]: _positive(value,name)
        _flags(concat=concat,bias=bias)
        if num_bases is not None: _positive(num_bases,'num_bases')
        if num_blocks is not None: _positive(num_blocks,'num_blocks')
        if num_bases is not None and num_blocks is not None: raise ValueError('basis and block decompositions are mutually exclusive')
        if edge_dim is not None: _positive(edge_dim,'edge_dim')
        if attention_mechanism not in ('within-relation','across-relation'): raise ValueError('invalid attention_mechanism')
        if attention_mode not in ('additive-self-attention','multiplicative-self-attention'): raise ValueError('invalid attention_mode')
        if attention_mode=='additive-self-attention' and dim!=1: raise ValueError('additive attention requires dim=1')
        if mod not in (None,'additive','scaled','f-additive','f-scaled'): raise ValueError('invalid cardinality mode')
        if type(dropout) not in (int,float) or not math.isfinite(dropout) or not 0<=dropout<=1: raise ValueError('dropout must be in [0,1]')
        if mod is not None and dropout>0: raise ValueError('cardinality modes require dropout=0')
        if type(negative_slope) not in (int,float) or not math.isfinite(negative_slope): raise ValueError('negative_slope must be finite')
        width=heads*out_channels
        if num_blocks is not None and (in_channels%num_blocks or width%num_blocks): raise ValueError('input and projected widths must be divisible by num_blocks')
        _capacity((width,heads*dim));_capacity((heads,dim,out_channels));_capacity((out_channels,out_channels))
        if num_blocks is None: _capacity((num_relations,in_channels,width))
        else: _capacity((num_relations,num_blocks,in_channels//num_blocks,width//num_blocks))
        if num_bases is not None: _capacity((num_bases,in_channels,width));_capacity((num_relations,num_bases))
        if edge_dim is not None: _capacity((edge_dim,width))
        aggr=_options(kwargs,'add')
        _configure(self,in_channels=in_channels,out_channels=out_channels,num_relations=num_relations,num_bases=num_bases,
                   num_blocks=num_blocks,mod=mod,attention_mechanism=attention_mechanism,attention_mode=attention_mode,
                   heads=heads,dim=dim,concat=concat,negative_slope=float(negative_slope),dropout=float(dropout),edge_dim=edge_dim,aggr=aggr)
        self.q=Parameter(Tensor.zeros([width,heads*dim]));self.k=Parameter(Tensor.zeros([width,heads*dim]))
        self.register_parameter('bias',Parameter(Tensor.zeros([(heads if concat else 1)*dim*out_channels])) if bias else None)
        self.lin_edge=Linear(edge_dim,width,bias=False) if edge_dim is not None else None
        self.register_parameter('e',Parameter(Tensor.zeros([width,heads*dim])) if edge_dim is not None else None)
        if num_bases is not None:
            self.att=Parameter(Tensor.zeros([num_relations,num_bases]));self.basis=Parameter(Tensor.zeros([num_bases,in_channels,width]))
        elif num_blocks is not None:
            self.weight=Parameter(Tensor.zeros([num_relations,num_blocks,in_channels//num_blocks,width//num_blocks]))
        else: self.weight=Parameter(Tensor.zeros([num_relations,in_channels,width]))
        self.w=Parameter(Tensor.ones([out_channels]));self.l1=Parameter(Tensor.ones([1,out_channels]));self.b1=Parameter(Tensor.zeros([1,out_channels]))
        self.l2=Parameter(Tensor.ones([out_channels,out_channels])/out_channels);self.b2=Parameter(Tensor.zeros([1,out_channels]))
        self.activation=ReLU();self.attn_dropout=Dropout(float(dropout));self.reset_parameters()

    def reset_parameters(self):
        weights=[self.q,self.k]+([self.basis,self.att] if self.num_bases is not None else [self.weight])
        if self.e is not None: weights.append(self.e)
        with no_grad():
            for p in weights:
                shape=p.shape;bound=math.sqrt(6./(shape[-2]+shape[-1]));p.tensor().copy_((Tensor.rand(shape)*2-1)*bound)
            if self.bias is not None: self.bias.tensor().fill_(0.)
            self.l1.tensor().fill_(1.);self.b1.tensor().fill_(0.);self.b2.tensor().fill_(0.)
        if self.lin_edge is not None: _reset_linear(self.lin_edge,True)

    def _preflight(self,x,edge_index,edge_type,edge_attr,size,return_attention_weights):
        _eager();_features(x,'x',self.in_channels);_capacity(x.shape)
        nodes=x.shape[0]
        if size is not None and (not isinstance(size,(tuple,list)) or len(size)!=2 or any(v is not None and (type(v)is not int or v!=nodes) for v in size)):
            raise ValueError('size must match the homogeneous feature row count')
        if return_attention_weights is not None and type(return_attention_weights)is not bool: raise TypeError('return_attention_weights must be bool or None')
        if isinstance(edge_index,Tensor): _capacity(edge_index.shape,8)
        src,dst,_=_edge_parts(edge_index,size=(nodes,nodes));edges=len(src)
        _cpu_tensor(edge_type,'edge_type','int64')
        if edge_type.shape!=[edges]: raise ValueError('edge_type must have shape [E]')
        _capacity(edge_type.shape,8);types=_index_ids(edge_type,'edge_type',edges)
        if any(r>=self.num_relations for r in types): raise ValueError('edge_type contains an undeclared relation')
        if edge_attr is not None:
            _cpu_tensor(edge_attr,'edge_attr')
            if self.edge_dim is None: raise ValueError('edge_attr requires configured edge_dim')
            if edge_attr.shape!=[edges,self.edge_dim] and not (self.edge_dim==1 and edge_attr.shape==[edges]): raise ValueError('edge_attr must have shape [E,edge_dim]')
            _capacity(edge_attr.shape)
        h,c,d=self.heads,self.out_channels,self.dim
        _capacity((nodes+1,),8);_capacity((edges+1,),8)
        for count in (nodes,edges): _capacity((count,h,d,c))
        _capacity((edges,self.in_channels));_capacity((edges,h*d))
        if self.num_blocks is None: _capacity((edges,self.in_channels,h*c))
        else: _capacity((edges,self.num_blocks,self.in_channels//self.num_blocks,h*c//self.num_blocks))
        return src,dst,types

    def forward(self,x,edge_index,edge_type=None,edge_attr=None,size=None,return_attention_weights=None):
        src,dst,types=self._preflight(x,edge_index,edge_type,edge_attr,size,return_attention_weights)
        n,e,h,c,d=x.shape[0],len(src),self.heads,self.out_channels,self.dim;width=h*c
        xi,xj=x.index_select(0,dst),x.index_select(0,src)
        if self.num_bases is not None: weight=(self.att.tensor()@self.basis.tensor().reshape([self.num_bases,self.in_channels*width])).reshape([self.num_relations,self.in_channels,width])
        else: weight=self.weight.tensor()
        weight=weight.index_select(0,types)
        if self.num_blocks is not None:
            blocks=self.num_blocks;weight=weight.reshape([e*blocks,self.in_channels//blocks,width//blocks])
            outi=xi.reshape([e*blocks,1,self.in_channels//blocks]).bmm(weight).reshape([e,width])
            outj=xj.reshape([e*blocks,1,self.in_channels//blocks]).bmm(weight).reshape([e,width])
        else:
            outi=xi.unsqueeze(1).bmm(weight).squeeze(1);outj=xj.unsqueeze(1).bmm(weight).squeeze(1)
        qi,kj=outi@self.q.tensor(),outj@self.k.tensor()
        addition=qi+kj if self.attention_mode=='additive-self-attention' else qi*kj
        if edge_attr is not None:
            edge=(self.lin_edge(edge_attr.reshape([e,self.edge_dim]))@self.e.tensor())
            addition=addition+edge if self.attention_mode=='additive-self-attention' else addition*edge
        if self.attention_mode=='additive-self-attention': addition=addition.leaky_relu(self.negative_slope)
        groups,count=dst,n
        if self.attention_mechanism=='within-relation':
            mapping={};groups=[]
            for relation,node in zip(types,dst):
                key=(relation,node)
                if key not in mapping: mapping[key]=len(mapping)
                groups.append(mapping[key])
            count=len(mapping)
        saved=segment_softmax(addition,groups,count);alpha=saved
        if self.mod in ('scaled','f-scaled'):
            degree=scatter(Tensor.ones([e,1]),dst,n,'sum').index_select(0,dst)
        if self.mod=='f-additive': alpha=where(alpha>0,alpha+1,alpha)
        elif self.mod=='f-scaled': alpha=alpha*degree
        elif self.mod is None: alpha=self.attn_dropout(alpha)
        message=outj.reshape([e,h,1,c])*alpha.reshape([e,h,d,1])
        if self.mod=='additive': message=message+outj.reshape([e,h,1,c])*self.w.tensor().reshape([1,1,1,c])
        elif self.mod=='scaled':
            degree=(self.activation(degree@self.l1.tensor()+self.b1.tensor())@self.l2.tensor())+self.b2.tensor()
            message=message*degree.reshape([e,1,1,c])
        out=scatter(message.reshape([e,h*d*c]),dst,n,'sum').reshape([n,h,d,c])
        out=out.reshape([n,h*d*c]) if self.concat else out.mean_dim(1,False).reshape([n,d*c])
        if self.bias is not None: out=out+self.bias.tensor()
        topology=edge_index.to_tensor() if isinstance(edge_index,EdgeIndex) else edge_index
        return (out,(topology,saved)) if isinstance(return_attention_weights,bool) else out
