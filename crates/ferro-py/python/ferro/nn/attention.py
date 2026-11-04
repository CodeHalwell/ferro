"""Attention layers composed over the native scaled_dot_product_attention."""
import math
import sys
from .. import _native
from .._native import Tensor
from .._ops import cat
from .module import Module, Parameter
from .linear import Linear
from .layers import RMSNorm, Dropout
from ._attention_ops import attention_scores, scaled_dot_product_attention as _sdpa
from . import functional as F


def _validate_heads(embed_dim, num_heads, num_kv_heads):
    if any(type(value) is not int or value <= 0 for value in (embed_dim, num_heads, num_kv_heads)):
        raise ValueError('attention dimensions and head counts must be positive integers')
    if embed_dim % num_heads or num_heads % num_kv_heads:
        raise ValueError('embed_dim must be divisible by num_heads, and num_heads by num_kv_heads')
    if 3 * embed_dim * embed_dim > sys.maxsize // 4:
        raise ValueError('attention parameter allocation exceeds addressable storage')


def _split_heads(x, heads):
    b, s, d = x.shape
    return x.reshape([b, s, heads, d // heads]).transpose(1, 2).reshape([b * heads, s, d // heads])


def _merge_heads(x, batch, heads):
    _, s, hd = x.shape
    return x.reshape([batch, heads, s, hd]).transpose(1, 2).reshape([batch, s, heads * hd])


class MultiheadAttention(Module):
    """Packed projections, batched/unbatched layouts and CPU mask support.

    The constructor follows Torch's argument order and sequence-first default.
    Nonzero dropout uses a saved
    Philox stream. Separate key/value widths and extra source tokens are supported;
    True masks exclude. Parameters currently require float32 initialization.
    """
    _builtin_init = True

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        if any(type(flag) is not bool for flag in (batch_first, bias, add_bias_kv, add_zero_attn)):
            raise TypeError('batch_first, bias, add_bias_kv and add_zero_attn must be bool')
        _validate_heads(embed_dim, num_heads, num_heads)
        kdim, vdim = embed_dim if kdim is None else kdim, embed_dim if vdim is None else vdim
        if any(type(size) is not int or size <= 0 for size in (kdim, vdim)):
            raise ValueError('kdim and vdim must be positive integers')
        if embed_dim * (embed_dim + kdim + vdim) > sys.maxsize // 4:
            raise ValueError('attention parameter allocation exceeds addressable storage')
        if device not in (None, 'cpu') or dtype not in (None, 'float32'):
            raise NotImplementedError('MultiheadAttention initialization supports CPU float32 only')
        self._initialize()
        self.embed_dim, self.num_heads, self.batch_first = embed_dim, num_heads, batch_first
        self.kdim, self.vdim, self.add_zero_attn = kdim, vdim, add_zero_attn
        self._qkv_same_embed_dim = kdim == embed_dim and vdim == embed_dim
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be finite and in [0, 1]')
        self.dropout = float(dropout)
        self.attn_dropout = Dropout(dropout) if dropout else None
        bound = (6.0 / (4 * embed_dim)) ** 0.5
        self.in_proj_weight = Parameter((Tensor.rand([3 * embed_dim, embed_dim]) * 2 - 1) * bound) if self._qkv_same_embed_dim else None
        self.q_proj_weight = None if self._qkv_same_embed_dim else Parameter((Tensor.rand([embed_dim, embed_dim]) * 2 - 1) * (3. / embed_dim) ** .5)
        self.k_proj_weight = None if self._qkv_same_embed_dim else Parameter((Tensor.rand([embed_dim, kdim]) * 2 - 1) * (6. / (embed_dim + kdim)) ** .5)
        self.v_proj_weight = None if self._qkv_same_embed_dim else Parameter((Tensor.rand([embed_dim, vdim]) * 2 - 1) * (6. / (embed_dim + vdim)) ** .5)
        self.in_proj_bias = Parameter(Tensor.zeros([3 * embed_dim])) if bias else None
        self.bias_k = Parameter(Tensor.randn([1, 1, embed_dim]) * embed_dim ** -.5) if add_bias_kv else None
        self.bias_v = Parameter(Tensor.randn([1, 1, embed_dim]) * embed_dim ** -.5) if add_bias_kv else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        e, h = self.embed_dim, self.num_heads
        if (self.attn_dropout is None and self.dropout != 0.) or (self.attn_dropout is not None and self.attn_dropout.p != self.dropout):
            raise ValueError('construct a new MultiheadAttention to change its dropout configuration')
        if any(type(flag) is not bool for flag in (need_weights, average_attn_weights, is_causal)):
            raise TypeError('attention flags must be bool')
        if is_causal and attn_mask is None:
            raise ValueError('is_causal requires an explicit attention mask')
        if any(not isinstance(x, Tensor) or x.dtype != 'float32' for x in (query, key, value)):
            raise ValueError('MultiheadAttention inputs must be float32 Tensors')
        unbatched = query.ndim == 2
        if any(x.ndim != (2 if unbatched else 3) for x in (query, key, value)):
            raise ValueError('MultiheadAttention inputs must have matching rank 2 or 3')
        if unbatched:
            query, key, value = (x.unsqueeze(0) for x in (query, key, value))
        elif not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        if any(len(x.shape) != 3 or x.shape[-1] != width for x, width in ((query, e), (key, self.kdim), (value, self.vdim))):
            raise ValueError('MultiheadAttention input widths must match embed_dim, kdim and vdim')
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError('MultiheadAttention batch axes and key/value lengths must match')
        if query.device != key.device or query.device != value.device:
            raise ValueError('MultiheadAttention inputs must share a device')
        batch, length, source = query.shape[0], query.shape[1], key.shape[1]
        causal_hint_only = is_causal and attn_mask is not None and key_padding_mask is None and not need_weights
        apply_causal = is_causal and (attn_mask is None or causal_hint_only)
        if (attn_mask is not None or key_padding_mask is not None) and _native._is_capturing():
            raise NotImplementedError('masked MultiheadAttention requires eager execution')
        mask = None
        for selected, padding in ((attn_mask, False), (key_padding_mask, True)):
            if selected is None:
                continue
            if not isinstance(selected, Tensor) or selected.dtype not in ('bool', 'float32') or selected.device != query.device:
                raise ValueError('attention masks must be bool/float32 on the input device')
            if query.device != 'cpu':
                raise NotImplementedError('masked MultiheadAttention currently supports CPU tensors only')
            allowed = ([source],) if unbatched and padding else ([batch, source],) if padding else ([length, source], [batch * h, length, source])
            if selected.shape not in allowed:
                raise ValueError('attention mask shape does not match input metadata')
            if selected.dtype == 'bool':
                values = selected.reshape([selected.numel()]).tolist()
                selected = Tensor([-math.inf if value else 0. for value in values], selected.shape)
            selected = selected.reshape([batch, 1, 1, source]) if padding else selected.reshape([batch, h, length, source]) if selected.ndim == 3 else selected
            mask = selected if mask is None else mask + selected
        if causal_hint_only:
            mask = None
        w = None if self.in_proj_weight is None else self.in_proj_weight.tensor()
        b = None if self.in_proj_bias is None else self.in_proj_bias.tensor()
        rows = lambda i: list(range(i * e, (i + 1) * e))
        projections = (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight)
        q, k, v = (F.linear(x, w.index_select(0, rows(i)) if w is not None else projections[i].tensor(),
                            None if b is None else b.index_select(0, rows(i))) for i, x in enumerate((query, key, value)))
        extra = int(self.bias_k is not None) + int(self.add_zero_attn)
        if extra:
            if apply_causal and not causal_hint_only and batch and length and source:
                causal = Tensor([0. if j <= i else -math.inf for i in range(length) for j in range(source)], [length, source])
                mask = causal if mask is None else mask + causal
            if not causal_hint_only:
                apply_causal = False
            if mask is not None:
                mask = F.pad(mask, (0, extra))
            if self.bias_k is not None:
                k = cat((k, self.bias_k.tensor().expand(batch, 1, e)), dim=1)
                v = cat((v, self.bias_v.tensor().expand(batch, 1, e)), dim=1)
            if self.add_zero_attn:
                k = cat((k, Tensor.zeros([batch, 1, e], device=k.device)), dim=1)
                v = cat((v, Tensor.zeros([batch, 1, e], device=v.device)), dim=1)
            source += extra
        q, k, v = (_split_heads(x, h) for x in (q, k, v))
        if not need_weights:
            if mask is not None and apply_causal and batch and length and source:
                causal = Tensor([0. if j <= i else -math.inf for i in range(length) for j in range(source)], [length, source])
                mask = mask + causal
            mask = None if mask is None else mask.expand(batch, h, length, source).reshape([batch * h, length, source])
            context = _sdpa(q, k, v, attn_mask=mask, is_causal=apply_causal and mask is None,
                            dropout_p=self.dropout if self.training else 0., dropout_fn=self.attn_dropout)
            weights = None
        else:
            q4, k4 = q.reshape([batch, h, length, e // h]), k.reshape([batch, h, source, e // h])
            weights = attention_scores(q4, k4, mask, apply_causal).softmax(-1)
            if self.training and self.attn_dropout is not None:
                weights = self.attn_dropout(weights)
            context = weights.reshape([batch * h, length, source]).bmm(v)
            if average_attn_weights:
                weights = weights.mean_dim(1)
        out = self.out_proj(_merge_heads(context, batch, h))
        if unbatched:
            out = out.squeeze(0)
            if weights is not None:
                weights = weights.squeeze(0)
        elif not self.batch_first:
            out = out.transpose(0, 1)
        return out, weights


class SelfAttention(Module):
    """LLaMA-shaped self-attention mirroring ferro_core::nn::MultiHeadAttention:
    q/k/v/o projections, optional grouped-query kv heads, optional RoPE at
    positions 0..seq, optional causal mask. Input and output [batch, seq, dim]."""
    _builtin_init = True

    def __init__(self, embed_dim, num_heads, num_kv_heads=None, causal=True, rope_base=None, bias=False):
        kv = num_heads if num_kv_heads is None else num_kv_heads
        _validate_heads(embed_dim, num_heads, kv)
        if rope_base is not None:
            rope_base = float(rope_base)
            if not math.isfinite(rope_base) or rope_base <= 0 or (embed_dim // num_heads) % 2:
                raise ValueError('RoPE requires a finite positive base and even head width')
        self._initialize()
        self.embed_dim, self.num_heads, self.num_kv_heads = embed_dim, num_heads, kv
        self.causal, self.rope_base = bool(causal), None if rope_base is None else float(rope_base)
        hd = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, kv * hd, bias=bias)
        self.v_proj = Linear(embed_dim, kv * hd, bias=bias)
        self.o_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError(f'SelfAttention expects [batch, seq, dim], got {x.shape}')
        b, s, _ = x.shape
        h, kv = self.num_heads, self.num_kv_heads
        q, k, v = _split_heads(self.q_proj(x), h), _split_heads(self.k_proj(x), kv), _split_heads(self.v_proj(x), kv)
        if self.rope_base is not None:
            q, k = q.rope_cached(self.rope_base), k.rope_cached(self.rope_base)
        if kv != h:
            # Query head i reads kv head i // group (torch/HF ordering).
            idx = [j for j in range(kv) for _ in range(h // kv)]
            expand = lambda t: t.reshape([b, kv] + t.shape[1:]).index_select(1, idx).reshape([b * h] + t.shape[1:])
            k, v = expand(k), expand(v)
        return self.o_proj(_merge_heads(F.scaled_dot_product_attention(q, k, v, is_causal=self.causal), b, h))


class TransformerBlock(Module):
    """Pre-norm causal decoder block mirroring ferro_core::nn::TransformerBlock:
    x + attn(rmsnorm(x)), then x + down(gelu_tanh(up(rmsnorm(x)))) with a 4x MLP.
    RoPE (base 10000) self-attention, RMSNorm eps 1e-5, no dropout. This is not
    torch.nn.TransformerEncoderLayer (post-norm LayerNorm, ReLU, dropout)."""
    _builtin_init = True

    def __init__(self, dim, num_heads):
        _validate_heads(dim, num_heads, num_heads)
        if (dim // num_heads) % 2:
            raise ValueError('TransformerBlock RoPE requires an even head width')
        self._initialize()
        self.dim, self.num_heads = dim, num_heads
        self.norm1 = RMSNorm(dim, eps=1e-5)
        self.attn = SelfAttention(dim, num_heads, causal=True, rope_base=10000.0)
        self.norm2 = RMSNorm(dim, eps=1e-5)
        self.up = Linear(dim, 4 * dim)
        self.down = Linear(4 * dim, dim)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        return h + self.down(self.up(self.norm2(h)).gelu())
