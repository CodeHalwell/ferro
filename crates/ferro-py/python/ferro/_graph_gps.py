"""GPS local/global graph transformers over native graph and attention Modules."""
import math

from ._native import Tensor, cat, no_grad
from ._graph_aggregation import _layout
from ._graph_conv import _configure, _features, _positive, _reset_linear
from ._graph_utils import _capacity, _edge_parts
from ._graph_norm import GraphNorm, LayerNorm
from ._graph_stateful_norm import BatchNorm
from . import nn


class GPSConv(nn.Module):
    """Local message passing plus graphwise multihead attention and an MLP.

    CPU float32 features use sorted graph IDs, with native padding/unpadding.
    The supplied local Module is registered directly and keeps its initial
    state; it must return [N, channels]. Its kwargs and internal side effects
    remain its own contract. Known global metadata and normalization failures
    reject before the callback or saved dropout streams execute. Normalization
    supports batch_norm, layer_norm, graph_norm, or None. Performer attention,
    arbitrary normalization factories, empty inputs and capture are unsupported.
    """
    _builtin_init = True
    channels: int
    heads: int
    dropout: float
    act: str
    norm: object
    attn_type: str
    norm_with_batch: bool

    def __init__(self, channels, conv, heads=1, dropout=0., act='relu', act_kwargs=None,
                 norm='batch_norm', norm_kwargs=None, attn_type='multihead', attn_kwargs=None):
        _positive(channels, 'channels'); _positive(heads, 'heads')
        if channels % heads: raise ValueError('channels must be divisible by heads')
        if conv is not None and not isinstance(conv, nn.Module): raise TypeError('conv must be a native Module or None')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0,1]')
        if attn_type != 'multihead': raise NotImplementedError('GPSConv supports native multihead attention only')
        for value, name in ((act_kwargs, 'act_kwargs'), (norm_kwargs, 'norm_kwargs'), (attn_kwargs, 'attn_kwargs')):
            if value is not None and type(value) is not dict: raise TypeError(name + ' must be a dictionary or None')
        activations = dict(relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU, silu=nn.SiLU, elu=nn.ELU, leaky_relu=nn.LeakyReLU)
        if isinstance(act, nn.Module):
            if act_kwargs: raise ValueError('an activation Module does not accept act_kwargs')
            activation, act_name = act, 'module'
        elif isinstance(act, str) and act in activations:
            activation, act_name = activations[act](**(act_kwargs or {})), act
        else: raise NotImplementedError('GPSConv requires a supported activation name or native Module')
        norms = {'batch_norm': BatchNorm, 'layer_norm': LayerNorm, 'graph_norm': GraphNorm}
        if norm is not None and (not isinstance(norm, str) or norm not in norms):
            raise NotImplementedError('GPSConv normalization must be batch_norm, layer_norm, graph_norm, or None')
        if norm is None and norm_kwargs: raise ValueError('norm_kwargs require a normalization layer')
        attention = dict(attn_kwargs or {})
        if any(attention.get(key) not in (None, channels) for key in ('kdim', 'vdim')):
            raise ValueError('GPS self-attention key/value widths must match channels')
        _capacity((8, channels, channels))
        _configure(self, channels=channels, heads=heads, dropout=float(dropout), act=act_name,
                   norm=norm, attn_type=attn_type, norm_with_batch=norm in ('layer_norm', 'graph_norm'))
        self.conv = conv
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True, **attention)
        first, last = nn.Linear(channels, 2 * channels), nn.Linear(2 * channels, channels)
        _reset_linear(first); _reset_linear(last)
        self.mlp = nn.Sequential(first, activation, nn.Dropout(float(dropout)), last, nn.Dropout(float(dropout)))
        self.local_dropout, self.global_dropout = nn.Dropout(float(dropout)), nn.Dropout(float(dropout))
        self.norm1 = None if norm is None else norms[norm](channels, **(norm_kwargs or {}))
        self.norm2 = None if norm is None else norms[norm](channels, **(norm_kwargs or {}))
        self.norm3 = None if norm is None else norms[norm](channels, **(norm_kwargs or {}))

    def reset_parameters(self):
        if self.conv is not None:
            reset = getattr(self.conv, 'reset_parameters', None)
            if not callable(reset): raise AttributeError('local GPS conv has no reset_parameters method')
            reset()
        with no_grad():
            self.attn.in_proj_weight.tensor().copy_((Tensor.rand([3 * self.channels, self.channels]) * 2 - 1) * math.sqrt(6. / (4 * self.channels)))
            if self.attn.in_proj_bias is not None:
                self.attn.in_proj_bias.tensor().copy_(Tensor.zeros([3 * self.channels]))
                self.attn.out_proj.bias.tensor().copy_(Tensor.zeros([self.channels]))
            for parameter in (self.attn.bias_k, self.attn.bias_v):
                if parameter is not None:
                    parameter.tensor().copy_(Tensor.randn([1, 1, self.channels]) * math.sqrt(1. / self.channels))
        from ._graph_conv import _reset_network
        _reset_network(self.mlp)
        for norm in (self.norm1, self.norm2, self.norm3):
            if norm is not None: norm.reset_parameters()

    def _preflight(self, x, edge_index, batch):
        ids, groups = _layout(x, batch, None, None, -2)
        if x.shape[1] != self.channels or not x.shape[0]: raise ValueError('GPSConv requires nonempty [N, channels] features')
        if any(left > right for left, right in zip(ids, ids[1:])): raise ValueError('GPSConv requires sorted batch indices')
        if self.conv is not None: _edge_parts(edge_index, num_nodes=x.shape[0])
        counts = {}
        for group in ids: counts[group] = counts.get(group, 0) + 1
        time = max(counts.values())
        extra = int(self.attn.bias_k is not None) + int(self.attn.add_zero_attn)
        _capacity((x.shape[0] + 1, self.channels))
        _capacity((x.shape[0], 2 * self.channels))
        _capacity((groups, time), 8)
        _capacity((groups, time + extra, self.channels))
        _capacity((groups, self.heads, time, time + extra))
        active = (self.norm2, self.norm3) if self.conv is None else (self.norm1, self.norm2, self.norm3)
        for norm in active:
            if isinstance(norm, BatchNorm) and x.shape[0] == 1 and not norm.allow_single_element:
                layer = norm.module
                if layer.training or layer.running_mean is None or layer.running_var is None:
                    raise ValueError('GPS BatchNorm requires more than one row while using input statistics')
        return ids, groups, time

    def _normalize(self, value, norm, batch):
        if norm is None: return value
        return norm(value, batch=batch) if self.norm_with_batch else norm(value)

    def forward(self, x, edge_index, batch=None, **kwargs):
        ids, groups, time = self._preflight(x, edge_index, batch)
        local = None
        if self.conv is not None:
            local = self.conv(x, edge_index, **kwargs)
            _features(local, 'local GPS output', self.channels)
            if local.shape[0] != x.shape[0]: raise ValueError('local GPS conv must preserve the node count')
            local = self._normalize(self.local_dropout(local) + x, self.norm1, batch)
        positions, selected = [x.shape[0]] * (groups * time), []
        used = {}
        for row, group in enumerate(ids):
            position = used.get(group, 0)
            used[group] = position + 1
            flat = group * time + position
            positions[flat] = row; selected.append(flat)
        mask = Tensor.from_bool([position == x.shape[0] for position in positions], [groups, time])
        dense = cat([x, Tensor.zeros([1, self.channels])], 0).index_select(0, positions).reshape([groups, time, self.channels])
        output, _ = self.attn(dense, dense, dense, key_padding_mask=mask, need_weights=False)
        output = output.reshape([groups * time, self.channels]).index_select(0, selected)
        output = self._normalize(self.global_dropout(output) + x, self.norm2, batch)
        if local is not None: output = output + local
        output = output + self.mlp(output)
        return self._normalize(output, self.norm3, batch)
