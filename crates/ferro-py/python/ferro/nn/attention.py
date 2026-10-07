"""Attention layers composed over the native scaled_dot_product_attention."""
from .._native import Tensor
from .module import Module, Parameter
from .linear import Linear
from .layers import RMSNorm
from . import functional as F


def _split_heads(x, heads):
    b, s, d = x.shape
    return x.reshape([b, s, heads, d // heads]).transpose(1, 2).reshape([b * heads, s, d // heads])


def _merge_heads(x, batch, heads):
    _, s, hd = x.shape
    return x.reshape([batch, heads, s, hd]).transpose(1, 2).reshape([batch, s, heads * hd])


class MultiheadAttention(Module):
    """torch.nn.MultiheadAttention subset: batch_first=True, packed in_proj,
    no masks (is_causal only), no dropout, averaged attention weights."""
    _builtin_init = True

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True):
        if not batch_first:
            raise NotImplementedError('only batch_first=True is supported')
        if num_heads <= 0 or embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        self._initialize()
        self.embed_dim, self.num_heads, self.batch_first = embed_dim, num_heads, True
        bound = (6.0 / (4 * embed_dim)) ** 0.5
        self.in_proj_weight = Parameter((Tensor.rand([3 * embed_dim, embed_dim]) * 2 - 1) * bound)
        self.in_proj_bias = Parameter(Tensor.zeros([3 * embed_dim])) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        if key_padding_mask is not None or attn_mask is not None or not average_attn_weights:
            raise NotImplementedError('masks and per-head attention weights are unsupported')
        e, h = self.embed_dim, self.num_heads
        w = self.in_proj_weight.tensor()
        b = None if self.in_proj_bias is None else self.in_proj_bias.tensor()
        rows = lambda i: list(range(i * e, (i + 1) * e))
        q, k, v = (_split_heads(F.linear(x, w.index_select(0, rows(i)), None if b is None else b.index_select(0, rows(i))), h)
                   for i, x in enumerate((query, key, value)))
        out = self.out_proj(_merge_heads(F.scaled_dot_product_attention(q, k, v, is_causal=is_causal), query.shape[0], h))
        if not need_weights:
            return out, None
        scores = q.bmm(k.transpose(1, 2)) * (e // h) ** -0.5
        if is_causal:
            l, s = scores.shape[1:]
            scores = scores + Tensor([-1e9 if j > i else 0.0 for i in range(l) for j in range(s)], [l, s])
        weights = scores.softmax(-1).reshape([query.shape[0], h] + scores.shape[1:]).mean_dim(1)
        return out, weights


class SelfAttention(Module):
    """LLaMA-shaped self-attention mirroring ferro_core::nn::MultiHeadAttention:
    q/k/v/o projections, optional grouped-query kv heads, optional RoPE at
    positions 0..seq, optional causal mask. Input and output [batch, seq, dim]."""
    _builtin_init = True

    def __init__(self, embed_dim, num_heads, num_kv_heads=None, causal=True, rope_base=None, bias=False):
        kv = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads <= 0 or embed_dim % num_heads or kv <= 0 or num_heads % kv:
            raise ValueError('embed_dim must divide into num_heads, and num_heads into num_kv_heads')
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
