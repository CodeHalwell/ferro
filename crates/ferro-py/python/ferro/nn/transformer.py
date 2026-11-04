"""Canonical dense CPU f32 Transformer layers built from existing Ferro modules.

No nested tensor, fused inference, compile or autograd engine is introduced.
Dropout owns checkpointed streams whose draws do not match Torch's global RNG.
"""
import math
import sys

from .. import _native
from .._native import Tensor, Generator, no_grad
from .module import Module, ModuleList
from .linear import Linear
from .layers import Dropout, LayerNorm
from .attention import MultiheadAttention
from .activation import ReLU, GELU
from . import functional as F


def _configure(module, **values):
    module._initialize()
    for name, value in values.items():
        object.__setattr__(module, name, value)


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _capacity(elements):
    if elements > sys.maxsize // 4:
        raise ValueError('Transformer allocation capacity exceeds addressable storage')


def _factory(device, dtype):
    if device not in (None, 'cpu') or dtype not in (None, 'float32'):
        raise NotImplementedError('Transformer modules support CPU float32 only')


def _activation_name(activation):
    if (isinstance(activation, str) and activation == 'relu') or activation is F.relu or type(activation) is ReLU:
        return 'relu'
    if (isinstance(activation, str) and activation == 'gelu') or activation is F.gelu:
        return 'gelu'
    if type(activation) is GELU:
        return 'gelu_tanh' if activation.approximate == 'tanh' else 'gelu'
    raise NotImplementedError('Transformer activation supports relu/gelu strings, Ferro functional relu/gelu, or ReLU/GELU modules')


class _TransformerActivation(Module):
    kind: str

    def forward(self, x):
        return x.relu() if self.kind == 'relu' else x.gelu_erf() if self.kind == 'gelu' else x.gelu()


def _layer_config(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype):
    _factory(device, dtype)
    for value, name in ((d_model, 'd_model'), (nhead, 'nhead'), (dim_feedforward, 'dim_feedforward')):
        _positive(value, name)
    if d_model % nhead:
        raise ValueError('d_model must be divisible by nhead')
    for flag, name in ((batch_first, 'batch_first'), (norm_first, 'norm_first'), (bias, 'bias')):
        if type(flag) is not bool:
            raise TypeError(name + ' must be bool')
    if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError('dropout must be finite and in [0,1]')
    if type(layer_norm_eps) not in (int, float) or not math.isfinite(layer_norm_eps) or layer_norm_eps < 0:
        raise ValueError('layer_norm_eps must be finite and nonnegative')
    _capacity(8 * d_model * d_model + 2 * d_model * dim_feedforward + dim_feedforward + 12 * d_model)
    return dict(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout_rate=float(dropout),
                activation_name=_activation_name(activation), layer_norm_eps=float(layer_norm_eps),
                batch_first=batch_first, norm_first=norm_first, bias=bias)


def _input(value, width, batch_first, name):
    if _native._is_capturing():
        raise NotImplementedError('Transformer modules require eager execution')
    if not isinstance(value, Tensor) or value.device != 'cpu' or value.dtype != 'float32' or value.ndim not in (2, 3):
        raise ValueError(name + ' must be a dense CPU float32 Tensor with rank two or three')
    if value.shape[-1] != width:
        raise ValueError(name + ' feature dimension must equal d_model')
    if value.ndim == 2:
        return 1, value.shape[0], True
    return (value.shape[0], value.shape[1], False) if batch_first else (value.shape[1], value.shape[0], False)


def _mask(mask, batch, length, source, heads, unbatched, padding=False):
    if mask is None:
        return
    if not isinstance(mask, Tensor) or mask.device != 'cpu' or mask.dtype not in ('bool', 'float32'):
        raise ValueError('Transformer masks must be CPU bool or float32 Tensors')
    allowed = ([source],) if padding and unbatched else ([batch, source],) if padding else ([length, source], [batch * heads, length, source])
    if mask.shape not in allowed:
        raise ValueError('Transformer mask shape does not match sequence, batch and head dimensions')


def _attention_preflight(module, query, key, mask, padding, causal):
    if type(causal) is not bool:
        raise TypeError('causal hints must be bool in Transformer layers')
    batch, length, unbatched = _input(query, module.embed_dim, module.batch_first, 'query')
    other_batch, source, other_unbatched = _input(key, module.embed_dim, module.batch_first, 'key')
    if batch != other_batch or unbatched != other_unbatched:
        raise ValueError('Transformer query and memory must have matching batch/layout metadata')
    _mask(mask, batch, length, source, module.num_heads, unbatched)
    _mask(padding, batch, length, source, module.num_heads, unbatched, True)
    if causal and mask is None:
        raise ValueError('a causal hint requires an explicit attention mask, as in Torch eager Transformer layers')


def _attention(module, query, key, value, mask, padding, causal):
    _attention_preflight(module, query, key, mask, padding, causal)
    result, _ = module(query, key, value, attn_mask=mask, key_padding_mask=padding,
                       need_weights=False, is_causal=causal)
    return result


def _causal_hint(mask, hint, length):
    if hint is not None:
        if type(hint) is not bool:
            raise TypeError('causal hint must be bool or None')
        return hint
    if mask is None:
        return False
    if not isinstance(mask, Tensor) or mask.device != 'cpu' or mask.dtype not in ('bool', 'float32'):
        raise ValueError('Transformer masks must be CPU bool or float32 Tensors')
    if mask.shape != [length, length]:
        return False
    entries = mask.tolist()
    return all(entries[i][j] == ((j > i) if mask.dtype == 'bool' else (-math.inf if j > i else 0.))
               for i in range(length) for j in range(length))


class _TransformerLayer(Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout_rate: float
    activation_name: str
    layer_norm_eps: float
    batch_first: bool
    norm_first: bool
    bias: bool

    def _build_common(self):
        self.self_attn = MultiheadAttention(self.d_model, self.nhead, bias=self.bias, batch_first=self.batch_first, dropout=self.dropout_rate)
        self.linear1 = Linear(self.d_model, self.dim_feedforward, bias=self.bias)
        self.dropout = Dropout(self.dropout_rate)
        self.linear2 = Linear(self.dim_feedforward, self.d_model, bias=self.bias)
        self.norm1 = LayerNorm(self.d_model, eps=self.layer_norm_eps, bias=self.bias)
        self.norm2 = LayerNorm(self.d_model, eps=self.layer_norm_eps, bias=self.bias)
        self.dropout1, self.dropout2 = Dropout(self.dropout_rate), Dropout(self.dropout_rate)
        self.activation = _TransformerActivation(kind=self.activation_name)

    def _feedforward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(_TransformerLayer):
    """Torch-shaped dense encoder with pre/post LayerNorm and saved dropout.

    Supports relu/exact-gelu and explicit Ferro ReLU/GELU activations. Arbitrary
    callables reject because their hidden state cannot be safely reconstructed.
    Causal flags are hints requiring a mask, following Torch's eager path.
    """
    _builtin_init = True

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=.1, activation='relu',
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        _configure(self, **_layer_config(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                       batch_first, norm_first, bias, device, dtype))
        self._build_common()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        _input(src, self.d_model, self.batch_first, 'src')
        x = src
        if self.norm_first:
            normalized = self.norm1(x)
            x = x + self.dropout1(_attention(self.self_attn, normalized, normalized, normalized, src_mask, src_key_padding_mask, is_causal))
            return x + self.dropout2(self._feedforward(self.norm2(x)))
        x = self.norm1(x + self.dropout1(_attention(self.self_attn, x, x, x, src_mask, src_key_padding_mask, is_causal)))
        return self.norm2(x + self.dropout2(self._feedforward(x)))


class TransformerDecoderLayer(_TransformerLayer):
    """Torch-shaped dense decoder with self-attention and cross-attention."""
    _builtin_init = True

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=.1, activation='relu',
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        _configure(self, **_layer_config(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                                       batch_first, norm_first, bias, device, dtype))
        self._build_common()
        self.multihead_attn = MultiheadAttention(d_model, nhead, bias=bias, batch_first=batch_first, dropout=dropout)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        # Reject invalid cross-attention metadata before self-attention consumes
        # any checkpointed dropout stream.
        _attention_preflight(self.self_attn, tgt, tgt, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        _attention_preflight(self.multihead_attn, tgt, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
        x = tgt
        if self.norm_first:
            normalized = self.norm1(x)
            x = x + self.dropout1(_attention(self.self_attn, normalized, normalized, normalized, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = x + self.dropout2(_attention(self.multihead_attn, self.norm2(x), memory, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            return x + self.dropout3(self._feedforward(self.norm3(x)))
        x = self.norm1(x + self.dropout1(_attention(self.self_attn, x, x, x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)))
        x = self.norm2(x + self.dropout2(_attention(self.multihead_attn, x, memory, memory, memory_mask, memory_key_padding_mask, memory_is_causal)))
        return self.norm3(x + self.dropout3(self._feedforward(x)))


def _clones(layer, count, expected):
    if type(layer) is not expected:
        raise NotImplementedError('Transformer stacks clone exact built-in layers only; arbitrary subclasses/custom layers are unsupported')
    _positive(count, 'num_layers')
    _capacity(count * (sum(p.tensor().numel() for p in layer.parameters()) + sum(b.numel() for _, b in layer.named_buffers())))
    # The public checkpoint transaction validates structure/config/storage ties,
    # copies buffers and freezing, and never aliases freshly constructed slots.
    from ..checkpoint import snapshot_training
    snapshot = snapshot_training(layer, optimizers={}, generator=Generator(0))
    activation = GELU(approximate='tanh') if layer.activation_name == 'gelu_tanh' else layer.activation_name
    result = ModuleList()
    for _ in range(count):
        clone = expected(layer.d_model, layer.nhead, layer.dim_feedforward, layer.dropout_rate, activation,
                         layer.layer_norm_eps, layer.batch_first, layer.norm_first, layer.bias)
        snapshot.restore(clone, optimizers={}, generator=Generator(0))
        result.append(clone)
    return result


class TransformerEncoder(Module):
    """Independent copies of one quiescent canonical encoder-layer template.

    Clear template gradients before construction. Copies preserve template
    weights, modes/freezing and buffered dropout state in distinct storage.
    Modified/tied template structures reject. norm is registered directly.
    enable_nested_tensor/mask_check are accepted optimization preferences;
    execution always uses the dense eager path and use_nested_tensor is False.
    """
    _builtin_init = True
    num_layers: int
    d_model: int
    batch_first: bool
    enable_nested_tensor: bool
    mask_check: bool
    use_nested_tensor: bool

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        if type(enable_nested_tensor) is not bool or type(mask_check) is not bool:
            raise TypeError('nested optimization flags must be bool')
        if norm is not None and not isinstance(norm, Module):
            raise TypeError('norm must be a Ferro Module or None')
        layers = _clones(encoder_layer, num_layers, TransformerEncoderLayer)
        _configure(self, num_layers=num_layers, d_model=encoder_layer.d_model, batch_first=encoder_layer.batch_first,
                   enable_nested_tensor=enable_nested_tensor, mask_check=mask_check, use_nested_tensor=False)
        self.layers, self.norm = layers, norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        _, length, _ = _input(src, self.d_model, self.batch_first, 'src')
        causal = _causal_hint(mask, is_causal, length)
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=causal)
        return output if self.norm is None else self.norm(output)


class TransformerDecoder(Module):
    """Independent canonical decoder-layer copies; see TransformerEncoder."""
    _builtin_init = True
    num_layers: int
    d_model: int
    batch_first: bool

    def __init__(self, decoder_layer, num_layers, norm=None):
        if norm is not None and not isinstance(norm, Module):
            raise TypeError('norm must be a Ferro Module or None')
        layers = _clones(decoder_layer, num_layers, TransformerDecoderLayer)
        _configure(self, num_layers=num_layers, d_model=decoder_layer.d_model, batch_first=decoder_layer.batch_first)
        self.layers, self.norm = layers, norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
        _, length, _ = _input(tgt, self.d_model, self.batch_first, 'tgt')
        _input(memory, self.d_model, self.batch_first, 'memory')
        causal = _causal_hint(tgt_mask, tgt_is_causal, length)
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           tgt_is_causal=causal, memory_is_causal=memory_is_causal)
        return output if self.norm is None else self.norm(output)


class Transformer(Module):
    """Canonical encoder-decoder Transformer with dense eager execution.

    Custom encoder/decoder Ferro Modules are registered directly, not cloned;
    they must accept the standard forward keywords. Initialization applies
    Xavier-uniform to every matrix parameter, including custom modules, as
    Torch does. Default stacks require positive layer counts.
    """
    _builtin_init = True
    d_model: int
    nhead: int
    batch_first: bool

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=.1, activation='relu', custom_encoder=None,
                 custom_decoder=None, layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 bias=True, device=None, dtype=None):
        _layer_config(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                      batch_first, norm_first, bias, device, dtype)
        if any(module is not None and not isinstance(module, Module) for module in (custom_encoder, custom_decoder)):
            raise TypeError('custom_encoder/custom_decoder must be Ferro Modules')
        if custom_encoder is None:
            _positive(num_encoder_layers, 'num_encoder_layers')
        if custom_decoder is None:
            _positive(num_decoder_layers, 'num_decoder_layers')
        count = (num_encoder_layers if custom_encoder is None else 0) + (num_decoder_layers if custom_decoder is None else 0)
        _capacity(count * (8 * d_model * d_model + 2 * d_model * dim_feedforward + dim_feedforward + 12 * d_model))
        _configure(self, d_model=d_model, nhead=nhead, batch_first=batch_first)
        args = (d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias)
        self.encoder = custom_encoder if custom_encoder is not None else TransformerEncoder(
            TransformerEncoderLayer(*args), num_encoder_layers, LayerNorm(d_model, eps=layer_norm_eps, bias=bias))
        self.decoder = custom_decoder if custom_decoder is not None else TransformerDecoder(
            TransformerDecoderLayer(*args), num_decoder_layers, LayerNorm(d_model, eps=layer_norm_eps, bias=bias))
        self._reset_parameters()

    def _reset_parameters(self):
        with no_grad():
            for parameter in self.parameters():
                value = parameter.tensor()
                if value.ndim > 1 and value.numel():
                    receptive = math.prod(value.shape[2:])
                    bound = math.sqrt(6. / ((value.shape[0] + value.shape[1]) * receptive))
                    value.copy_((Tensor.rand(list(value.shape)) * 2 - 1) * bound)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, src_is_causal=None,
                tgt_is_causal=None, memory_is_causal=False):
        source_batch, source_length, source_unbatched = _input(src, self.d_model, self.batch_first, 'src')
        target_batch, target_length, target_unbatched = _input(tgt, self.d_model, self.batch_first, 'tgt')
        if source_batch != target_batch or source_unbatched != target_unbatched:
            raise ValueError('Transformer source and target batch/layout metadata must match')
        # Built-in stacks preserve sequence shape. Check their masks before the
        # encoder advances streams. Custom modules retain their own contract.
        if type(self.encoder) is TransformerEncoder and type(self.decoder) is TransformerDecoder:
            source_causal = _causal_hint(src_mask, src_is_causal, source_length)
            target_causal = _causal_hint(tgt_mask, tgt_is_causal, target_length)
            _attention_preflight(self.encoder.layers[0].self_attn, src, src, src_mask, src_key_padding_mask, source_causal)
            _attention_preflight(self.decoder.layers[0].self_attn, tgt, tgt, tgt_mask, tgt_key_padding_mask, target_causal)
            _attention_preflight(self.decoder.layers[0].multihead_attn, tgt, src, memory_mask, memory_key_padding_mask, memory_is_causal)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=src_is_causal)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)

    @staticmethod
    def generate_square_subsequent_mask(sz, device=None, dtype=None):
        if type(sz) is not int or sz < 0:
            raise ValueError('mask size must be a nonnegative integer')
        if device not in (None, 'cpu') or dtype not in (None, 'float32', 'bool'):
            raise NotImplementedError('Transformer masks support CPU float32 or bool')
        _capacity(sz * sz)
        if dtype == 'bool':
            return Tensor.from_bool([j > i for i in range(sz) for j in range(sz)], [sz, sz])
        return Tensor([-math.inf if j > i else 0. for i in range(sz) for j in range(sz)], [sz, sz])
