"""PyG-shaped MLP composition with native state and per-layer Dropout streams."""
import inspect
import math
import warnings
from . import nn
from ._graph_conv import _configure, _features, _flags, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_norm import GraphNorm, LayerNorm
from ._graph_stateful_norm import BatchNorm
from ._graph_utils import _capacity, _count, _index_ids


def _activation(query,options):
    if query is None:
        return None
    if isinstance(query,nn.Module):
        if options: raise ValueError('activation Module instances do not accept act_kwargs')
        return query
    constructors={name.lower():getattr(nn,name) for name in ('ReLU','LeakyReLU','GELU','Tanh','SiLU','ELU','SELU','CELU','Sigmoid','Softplus','Softsign','LogSigmoid','ReLU6')}
    key=query.lower().replace('_','').replace('-','') if isinstance(query,str) else None
    if key not in constructors: raise NotImplementedError('MLP requires a supported activation name, native Module, or None')
    return constructors[key](**options)


class MLP(nn.Module):
    """Dense graph features with configurable activation/normalization ordering.

    Positive fixed widths and rank-two CPU F32 inputs are supported. Norms may
    be batch_norm, layer_norm, graph_norm, a registered Module instance, or None.
    A supplied norm instance is reused, preserving PyG's shared state. Standard
    batch metadata is validated before layer/dropout work; arbitrary Module
    callback effects remain its own contract. Unregistered callables, lazy/zero
    widths and capture are unsupported. Each layer has a checkpointed Dropout
    stream; random draws are not claimed to match Torch's generator sequence.
    """
    _builtin_init = True
    channel_list: tuple
    dropout: tuple
    act_first: bool
    plain_last: bool
    supports_norm_batch: bool

    def __init__(self,channel_list=None,*,in_channels=None,hidden_channels=None,out_channels=None,num_layers=None,
                 dropout=0.,act='relu',act_first=False,act_kwargs=None,norm='batch_norm',norm_kwargs=None,
                 plain_last=True,bias=True,**kwargs):
        unsupported=set(kwargs)-{'relu_first','batch_norm','batch_norm_kwargs'}
        if unsupported: raise NotImplementedError('unsupported MLP options: '+', '.join(sorted(unsupported)))
        _flags(act_first=act_first,plain_last=plain_last)
        if 'relu_first' in kwargs:
            _flags(relu_first=kwargs['relu_first']);act_first=act_first or kwargs['relu_first']
        legacy=kwargs.get('batch_norm')
        if legacy is not None:
            _flags(batch_norm=legacy);warnings.warn('batch_norm is deprecated; use norm',UserWarning,stacklevel=2)
            norm='batch_norm' if legacy else None;norm_kwargs=kwargs.get('batch_norm_kwargs') or {}
        for value,name in ((act_kwargs,'act_kwargs'),(norm_kwargs,'norm_kwargs')):
            if value is not None and type(value)is not dict: raise TypeError(name+' must be a dictionary or None')
        if type(channel_list)is int: in_channels=channel_list
        if in_channels is not None:
            _positive(in_channels,'in_channels');_positive(num_layers,'num_layers');_positive(out_channels,'out_channels')
            if num_layers>1: _positive(hidden_channels,'hidden_channels')
            _capacity((num_layers+1,),8)
            channel_list=[in_channels]+[hidden_channels]*(num_layers-1)+[out_channels]
        if not isinstance(channel_list,(list,tuple)) or len(channel_list)<2: raise ValueError('channel_list must contain at least input and output widths')
        channels=tuple(_positive(width,'channel width') for width in channel_list);count=len(channels)-1
        if type(dropout)in(int,float):
            probabilities=[float(dropout)]*count
            if plain_last: probabilities[-1]=0.
        elif isinstance(dropout,(tuple,list)):
            probabilities=list(dropout)
        else: raise TypeError('dropout must be numeric or a per-layer sequence')
        if len(probabilities)!=count: raise ValueError('dropout must contain one probability per layer')
        if any(type(p)not in(int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probabilities): raise ValueError('dropout probabilities must be finite and in [0,1]')
        probabilities=tuple(float(p)for p in probabilities)
        if type(bias)is bool: biases=[bias]*count
        elif isinstance(bias,(tuple,list)): biases=list(bias)
        else: raise TypeError('bias must be bool or a per-layer sequence')
        if len(biases)!=count or any(type(value)is not bool for value in biases): raise ValueError('bias must contain one bool per layer')
        for source,target in zip(channels,channels[1:]): _capacity((source,target))
        _capacity((count+1,),8)
        norms={'batchnorm':BatchNorm,'layernorm':LayerNorm,'graphnorm':GraphNorm}
        norm_key=norm.lower().replace('_','').replace('-','') if isinstance(norm,str) else None
        if norm_key in ('batch','layer','graph'): norm_key+='norm'
        if norm is not None and not isinstance(norm,nn.Module) and norm_key not in norms: raise NotImplementedError('MLP requires batch/layer/graph normalization, a Module instance, or None')
        if isinstance(norm,nn.Module) and norm_kwargs: raise ValueError('normalization Module instances do not accept norm_kwargs')
        activation=_activation(act,dict(act_kwargs or {}))
        widths=channels[1:-1] if plain_last else channels[1:]
        normalization=[nn.Identity() if norm is None else norm if isinstance(norm,nn.Module) else norms[norm_key](width,**dict(norm_kwargs or {})) for width in widths]
        supports=bool(normalization) and 'batch' in inspect.signature(normalization[0].forward).parameters
        _configure(self,channel_list=channels,dropout=probabilities,act_first=act_first,plain_last=plain_last,supports_norm_batch=supports)
        self.act=activation
        self.lins=nn.ModuleList([nn.Linear(source,target,bias=flag) for source,target,flag in zip(channels,channels[1:],biases)])
        self.norms=nn.ModuleList(normalization)
        self.dropouts=nn.ModuleList([nn.Dropout(p) for p in probabilities])
        self.reset_parameters()

    @property
    def in_channels(self): return self.channel_list[0]
    @property
    def out_channels(self): return self.channel_list[-1]
    @property
    def num_layers(self): return len(self.channel_list)-1

    def reset_parameters(self):
        for layer in self.lins: _reset_linear(layer)
        for layer in self.norms:
            reset=getattr(layer,'reset_parameters',None)
            if callable(reset): reset()

    def get_out_channels(self,in_channels):
        if _count(in_channels,'in_channels')!=self.in_channels: raise ValueError('MLP input width differs from declared channels')
        return self.out_channels

    def _metadata(self,in_channels,rows,batch=None,batch_size=None,strict_state=False):
        self.get_out_channels(in_channels);_count(rows,'rows');_capacity((rows,in_channels))
        for width in self.channel_list[1:]: _capacity((rows,width))
        groups=None
        for width,layer in zip(self.channel_list[1:],self.norms):
            if isinstance(layer,(BatchNorm,GraphNorm,LayerNorm)) and layer.in_channels!=width: raise ValueError('normalization width differs from the layer output')
            if isinstance(layer,(GraphNorm,LayerNorm)) and not (isinstance(layer,LayerNorm) and layer.mode=='node'):
                if groups is None:
                    _capacity((rows+1,),8)
                    if batch_size is not None: _count(batch_size,'batch_size')
                    if batch is None: groups=1
                    else:
                        ids=_index_ids(batch,'batch',rows);groups=max(ids,default=-1)+1 if batch_size is None else batch_size
                        if any(i>=groups for i in ids): raise ValueError('batch contains an ID outside batch_size')
                    _capacity((groups+1,),8)
                _capacity((groups,width))
            if strict_state and isinstance(layer,BatchNorm) and rows==1 and not layer.allow_single_element:
                state=layer.module
                if state.training or state.running_mean is None or state.running_var is None:
                    raise ValueError('readout MLP BatchNorm requires more than one row while using input statistics')
        return self.out_channels

    def _preflight(self,x,batch,batch_size,return_emb):
        _eager();_features(x,'x',self.in_channels)
        if return_emb is not None and type(return_emb)is not bool: raise TypeError('return_emb must be bool or None')
        self._metadata(x.shape[1],x.shape[0],batch,batch_size)

    def forward(self,x,batch=None,batch_size=None,return_emb=None):
        self._preflight(x,batch,batch_size,return_emb);emb=None
        for i,(linear,norm)in enumerate(zip(self.lins,self.norms)):
            x=linear(x)
            if self.act is not None and self.act_first: x=self.act(x)
            x=norm(x,batch,batch_size) if self.supports_norm_batch else norm(x)
            if self.act is not None and not self.act_first: x=self.act(x)
            x=self.dropouts[i](x)
            if return_emb is True: emb=x
        if self.plain_last: x=self.dropouts[-1](self.lins[-1](x))
        return (x,emb) if isinstance(return_emb,bool) else x
