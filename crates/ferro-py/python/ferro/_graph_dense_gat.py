"""Native dense graph attention with topology masks and saved dropout state."""
import math

from ._native import Tensor, no_grad
from ._tensor_api import eye
from ._graph_conditioned import _eager
from ._graph_conv import _configure, _flags, _positive, _reset_linear
from ._graph_utils import _capacity, _cpu_tensor
from .nn import Dropout, Linear, Module, Parameter


def _batch(left, right):
    if left == right: return left
    if left == 1: return right
    if right == 1: return left
    raise ValueError('batch dimensions are not broadcastable')


class DenseGATConv(Module):
    """Dense multihead GAT using nonzero adjacency as a nondifferentiable mask.

    CPU float32 features accept rank two or three; adjacency supports CPU
    float32, bool and int64. Batch broadcasting follows PyG, including the
    concat-mode restriction when adjacency expands a nonempty feature batch.
    Bool/float32 masks keep float adjoints; native reshape additionally accepts
    strided masks for which PyG's view rejects. Empty
    axes are supported without a mask; all-masked rows retain PyG NaNs. Lazy
    widths and capture are unsupported. One registered Dropout owns the stream.
    """
    _builtin_init = True
    in_channels: int
    out_channels: int
    heads: int
    concat: bool
    negative_slope: float
    dropout: float

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=.2, dropout=0., bias=True):
        for value, name in ((in_channels, 'in_channels'), (out_channels, 'out_channels'), (heads, 'heads')): _positive(value, name)
        _flags(concat=concat, bias=bias)
        if type(negative_slope) not in (int, float) or not math.isfinite(negative_slope): raise ValueError('negative_slope must be finite')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1: raise ValueError('dropout must be in [0,1]')
        _capacity((in_channels, heads, out_channels)); _capacity((2, heads, out_channels))
        _configure(self, in_channels=in_channels, out_channels=out_channels, heads=heads, concat=concat,
                   negative_slope=float(negative_slope), dropout=float(dropout))
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = Parameter(Tensor.zeros([1, 1, heads, out_channels]))
        self.att_dst = Parameter(Tensor.zeros([1, 1, heads, out_channels]))
        self.bias = Parameter(Tensor.zeros([heads * out_channels if concat else out_channels])) if bias else None
        self.attn_dropout = Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin, glorot=True)
        with no_grad():
            for parameter in (self.att_src, self.att_dst):
                parameter.tensor().copy_((Tensor.rand(parameter.tensor().shape) * 2 - 1) * math.sqrt(6. / (self.heads + self.out_channels)))
            if self.bias is not None: self.bias.tensor().copy_(Tensor.zeros(self.bias.tensor().shape))

    def _preflight(self, x, adj, mask, add_loop):
        _eager(); _flags(add_loop=add_loop)
        _cpu_tensor(x, 'x'); _cpu_tensor(adj, 'adj', dtype=None)
        if len(x.shape) not in (2, 3) or x.shape[-1] != self.in_channels: raise ValueError('x requires [N,F] or [B,N,F] with the declared width')
        if len(adj.shape) not in (2, 3) or adj.dtype not in ('float32', 'bool', 'int64'): raise ValueError('adj requires CPU float32/bool/int64 [N,N] or [B,N,N]')
        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        adj = adj.unsqueeze(0) if len(adj.shape) == 2 else adj
        original, nodes, _ = x.shape
        if adj.shape[-2:] != [nodes, nodes]: raise ValueError('adjacency node axes must match x')
        batch = _batch(original, adj.shape[0])
        if self.concat and nodes and batch != original: raise ValueError('concat mode cannot expand the feature batch through adjacency')
        output_batch = original if self.concat else batch
        width = self.heads * self.out_channels if self.concat else self.out_channels
        for shape in (x.shape, (batch, nodes, self.heads, self.out_channels), (batch, nodes, nodes, self.heads), (output_batch, nodes, width)):
            _capacity(shape)
        _capacity(adj.shape, 8 if adj.dtype == 'int64' else 1 if adj.dtype == 'bool' else 4)
        if add_loop: _capacity((nodes, nodes), 8)
        if mask is not None:
            _cpu_tensor(mask, 'mask', dtype=None)
            if mask.dtype not in ('float32', 'bool') or not nodes or math.prod(mask.shape) % nodes:
                raise ValueError('mask requires bool/float32 storage reshaped into complete nonempty node rows')
            _capacity(mask.shape)
            mask_batch = math.prod(mask.shape) // nodes
            _capacity((_batch(output_batch, mask_batch), nodes, width))
            mask = mask.reshape([mask_batch, nodes, 1]).to(dtype='float32')
        return x, adj, mask, batch

    def forward(self, x, adj, mask=None, add_loop=True):
        x, adj, mask, batch = self._preflight(x, adj, mask, add_loop)
        original, nodes, _ = x.shape
        if adj.dtype != 'float32': adj = adj.to(dtype='float32')
        if add_loop: adj = adj.masked_fill(eye(nodes).ne(0.).unsqueeze(0), 1)
        values = self.lin(x).reshape([original, nodes, self.heads, self.out_channels])
        source = (values * self.att_src.tensor()).sum_dim(3, False)
        target = (values * self.att_dst.tensor()).sum_dim(3, False)
        logits = (source.unsqueeze(1) + target.unsqueeze(2)).leaky_relu(self.negative_slope)
        logits = logits.expand([batch, nodes, nodes, self.heads])
        alpha = logits.masked_fill(adj.unsqueeze(3).eq(0), -float('inf')).softmax(2)
        alpha = self.attn_dropout(alpha)
        weights = alpha.permute([0, 3, 1, 2]).reshape([batch * self.heads, nodes, nodes])
        projected = values.permute([0, 2, 1, 3]).expand([batch, self.heads, nodes, self.out_channels]).reshape([batch * self.heads, nodes, self.out_channels])
        output = weights.bmm(projected).reshape([batch, self.heads, nodes, self.out_channels]).permute([0, 2, 1, 3])
        output = output.reshape([original, nodes, self.heads * self.out_channels]) if self.concat else output.mean_dim(2, False)
        if self.bias is not None: output = output + self.bias.tensor()
        return output if mask is None else output * mask
