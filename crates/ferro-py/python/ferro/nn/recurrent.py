"""CPU f32 recurrent modules with PyTorch parameter names and state layouts.

RNN/GRU/LSTM support dense or packed sequences, stacked bidirectional layers,
interlayer dropout and LSTM projections. Dropout streams are Ferro's Philox
streams, not Torch's RNG sequence. State remains caller-owned.
"""
import math

from .._native import Tensor, cat, stack, no_grad
from .._tensor_api import _shape
from ..recurrent import RecurrentState, rnn_cell, gru_cell, lstm_cell, unroll
from .module import Module, ModuleList, Parameter
from .layers import Dropout
from .utils.rnn import PackedSequence


def _sizes(*sizes):
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError('recurrent dimensions and layer count must be positive integers')


def _factory(device, dtype):
    if device not in (None, 'cpu'):
        raise NotImplementedError('recurrent modules support CPU tensors only')
    if dtype is not None:
        raise NotImplementedError('recurrent modules currently use the default f32 dtype; dtype overrides are unsupported')


def _tensor(value, name):
    if not isinstance(value, Tensor):
        raise TypeError(name + ' must be a Tensor')
    if value.device != 'cpu':
        raise NotImplementedError('recurrent modules support CPU tensors only')
    if value.dtype != 'float32':
        raise ValueError(name + ' must have float32 dtype')


def _state(value, expected, name):
    _tensor(value, name)
    if value.shape != expected:
        raise ValueError(f'{name} must have shape {expected}, got {value.shape}')
    return value


def _step(kind, inputs, initial, weights, nonlinearity):
    wi, wh, bi, bh = weights[:4]
    if kind == 'lstm':
        if len(weights) == 5:
            gates = inputs @ wi.transpose(0, 1) + initial.h @ wh.transpose(0, 1)
            if bi is not None:
                gates = gates + bi + bh
            width = initial.c.shape[1]
            i = gates.narrow(1, 0, width).sigmoid()
            f = gates.narrow(1, width, width).sigmoid()
            g = gates.narrow(1, 2 * width, width).tanh()
            o = gates.narrow(1, 3 * width, width).sigmoid()
            c = f * initial.c + i * g
            h = (o * c.tanh()) @ weights[4].transpose(0, 1)
            return RecurrentState(h, c)
        h, c = lstm_cell(inputs, initial.h, initial.c, wi, wh, bi, bh)
        return RecurrentState(h, c)
    if kind == 'gru':
        return RecurrentState(gru_cell(inputs, initial.h, wi, wh, bi, bh))
    if nonlinearity == 'tanh':
        return RecurrentState(rnn_cell(inputs, initial.h, wi, wh, bi, bh))
    input_affine = inputs @ wi.transpose(0, 1)
    hidden_affine = initial.h @ wh.transpose(0, 1)
    if bi is not None:
        input_affine = input_affine + bi
        hidden_affine = hidden_affine + bh
    return RecurrentState((input_affine + hidden_affine).relu())


class _RecurrentParameters(Module):
    def reset_parameters(self, *, generator=None):
        """Uniform(-1/sqrt(H), 1/sqrt(H)); an optional explicit Generator is supported."""
        bound = self.hidden_size ** -0.5
        with no_grad():
            for parameter in self.parameters():
                tensor = parameter.tensor()
                tensor.copy_((Tensor.rand(list(tensor.shape), generator=generator) * 2 - 1) * bound)

    def _register_weights(self, suffix, inputs, gates, projection=0):
        _shape([gates * self.hidden_size, inputs])
        _shape([gates * self.hidden_size, projection or self.hidden_size])
        if projection:
            _shape([projection, self.hidden_size])
        names = ['weight_ih' + suffix, 'weight_hh' + suffix]
        setattr(self, names[0], Parameter(Tensor.zeros([gates * self.hidden_size, inputs])))
        setattr(self, names[1], Parameter(Tensor.zeros([gates * self.hidden_size, projection or self.hidden_size])))
        if self.bias:
            for side in ('ih', 'hh'):
                name = 'bias_' + side + suffix
                setattr(self, name, Parameter(Tensor.zeros([gates * self.hidden_size])))
                names.append(name)
        if projection:
            name = 'weight_hr' + suffix
            setattr(self, name, Parameter(Tensor.zeros([projection, self.hidden_size])))
            names.append(name)
        return tuple(names)

    def _weights(self, names):
        values = [getattr(self, name).tensor() for name in names]
        return values if self.bias else values[:2] + [None, None] + values[2:]


class _Recurrent(_RecurrentParameters):
    def __setattr__(self, name, value):
        if name == 'dropout':
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('dropout must be a finite probability in [0,1]')
            value = float(value)
            if '_registered' in self.__dict__ and 'num_layers' in self.__dict__:
                existing = self.__dict__.get('interlayer_dropout')
                if value and self.num_layers > 1:
                    if existing is None:
                        children = ModuleList(Dropout(value) for _ in range(self.num_layers - 1))
                        children.train(self.training)
                        super().__setattr__('interlayer_dropout', children)
                    else:
                        for layer in existing:
                            layer.p = value
                elif existing is not None:
                    super().__delattr__('interlayer_dropout')
        super().__setattr__(name, value)

    def _setup(self, kind, input_size, hidden_size, num_layers, bias, batch_first, dropout,
               bidirectional, nonlinearity, device, dtype, projection=0):
        _sizes(input_size, hidden_size, num_layers)
        _factory(device, dtype)
        if any(type(flag) is not bool for flag in (bias, batch_first, bidirectional)):
            raise TypeError('bias, batch_first and bidirectional must be bool')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('dropout must be a finite probability in [0,1]')
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('RNN nonlinearity must be tanh or relu')
        directions, gates = 2 if bidirectional else 1, {'rnn': 1, 'gru': 3, 'lstm': 4}[kind]
        width = projection or hidden_size
        first = gates * hidden_size * (input_size + width + (2 if bias else 0)) + projection * hidden_size
        later = gates * hidden_size * (directions * width + width + (2 if bias else 0)) + projection * hidden_size
        _shape([directions * (first + (num_layers - 1) * later) + (8 * (num_layers - 1) if dropout else 0)])
        self._initialize()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.bias, self.batch_first, self.dropout, self.bidirectional = bias, batch_first, float(dropout), bidirectional
        self._kind = kind
        if kind == 'lstm':
            self.proj_size = projection
        if kind == 'rnn':
            self.nonlinearity = nonlinearity
        self._all_weights = []
        for layer in range(num_layers):
            inputs = input_size if layer == 0 else (projection or hidden_size) * directions
            for direction in range(directions):
                suffix = f'_l{layer}' + ('_reverse' if direction else '')
                self._all_weights.append(self._register_weights(suffix, inputs, gates, projection))
        self.reset_parameters()

    @property
    def all_weights(self):
        return [[getattr(self, name) for name in names] for names in self._all_weights]

    def flatten_parameters(self):
        """Compatibility no-op: these eager CPU modules do not pack parameters."""

    def _run(self, inputs, initial, names):
        weights = self._weights(names)
        time, batch, features = inputs.shape
        nonlinearity = self.nonlinearity if self._kind == 'rnn' else 'tanh'
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('RNN nonlinearity must be tanh or relu')
        if nonlinearity == 'relu' or batch == 0 or getattr(self, 'proj_size', 0):
            # Native unroll skips empty batches. Applying the cells retains the
            # empty input/parameter gradient paths expected by the module API.
            outputs, current = [], initial
            for index in range(time):
                token = inputs.index_select(0, [index]).reshape([batch, features])
                current = _step(self._kind, token, current, weights, nonlinearity)
                outputs.append(current.h)
            return stack(outputs, 0), current
        output = unroll(self._kind, inputs, initial, *weights, lengths=[time] * batch)
        return output.outputs, output.state

    def _run_packed(self, data, sizes, initial, names, reverse):
        weights = self._weights(names)
        nonlinearity = self.nonlinearity if self._kind == 'rnn' else 'tanh'
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('RNN nonlinearity must be tanh or relu')
        offsets, offset = [], 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        outputs, current = [None] * len(sizes), initial
        times = range(len(sizes) - 1, -1, -1) if reverse else range(len(sizes))
        for step in times:
            size = sizes[step]
            state = RecurrentState(current.h.narrow(0, 0, size),
                                   None if current.c is None else current.c.narrow(0, 0, size))
            updated = _step(self._kind, data.narrow(0, offsets[step], size), state, weights, nonlinearity)
            outputs[step] = updated.h
            inactive = sizes[0] - size
            if inactive:
                h = cat([updated.h, current.h.narrow(0, size, inactive)], 0)
                c = None if updated.c is None else cat([updated.c, current.c.narrow(0, size, inactive)], 0)
                current = RecurrentState(h, c)
            else:
                current = updated
        return cat(outputs, 0), current

    def forward(self, input, hx=None):
        packed = isinstance(input, PackedSequence)
        if packed:
            input = PackedSequence(*input)
            _tensor(input.data, 'packed data')
            if input.data.ndim != 2:
                raise ValueError('recurrent packed data must have shape [sum(lengths), input_size]')
            sizes = input.batch_sizes.tolist()
            if not sizes:
                raise ValueError('recurrent packed sequence must not be empty')
            inputs, unbatched = input.data, False
            time, batch, features = len(sizes), sizes[0], inputs.shape[1]
        else:
            _tensor(input, 'input')
            if len(input.shape) not in (2, 3):
                raise ValueError('recurrent input must be [T,I], [T,B,I], or [B,T,I]')
            unbatched = len(input.shape) == 2
            inputs = input.unsqueeze(1) if unbatched else input.transpose(0, 1) if self.batch_first else input
            time, batch, features = inputs.shape
        if features != self.input_size:
            raise ValueError(f'input last dimension must be {self.input_size}, got {features}')
        if time == 0:
            raise ValueError('recurrent sequence length must be greater than zero')
        _shape([time], 'int64')
        directions = 2 if self.bidirectional else 1
        output_size = getattr(self, 'proj_size', 0) or self.hidden_size
        _shape([self.num_layers * directions, batch, output_size])
        _shape([batch, {'rnn': 1, 'gru': 3, 'lstm': 4}[self._kind] * self.hidden_size])
        _shape([inputs.shape[0], output_size * directions] if packed else [time, batch, output_size * directions])
        if self._kind == 'lstm':
            _shape([self.num_layers * directions, batch, self.hidden_size])
        prefix = [self.num_layers * directions] + ([] if unbatched else [batch])
        h_shape, c_shape = prefix + [output_size], prefix + [self.hidden_size]
        if hx is None:
            h = Tensor.zeros([self.num_layers * directions, batch, output_size])
            c = Tensor.zeros([self.num_layers * directions, batch, self.hidden_size]) if self._kind == 'lstm' else None
        elif self._kind == 'lstm':
            if not isinstance(hx, (tuple, list)) or len(hx) != 2:
                raise TypeError('LSTM state must be an (h_0, c_0) pair')
            h, c = _state(hx[0], h_shape, 'h_0'), _state(hx[1], c_shape, 'c_0')
            if unbatched:
                h, c = h.unsqueeze(1), c.unsqueeze(1)
        else:
            h, c = _state(hx, h_shape, 'h_0'), None
            if unbatched:
                h = h.unsqueeze(1)
        if packed and input.sorted_indices is not None:
            order = input.sorted_indices.tolist()
            h = h.index_select(1, order)
            if c is not None:
                c = c.index_select(1, order)
        final_h, final_c, current = [], [], inputs
        reverse = list(range(time - 1, -1, -1)) if self.bidirectional else None
        for layer in range(self.num_layers):
            outputs = []
            for direction in range(directions):
                index = layer * directions + direction
                initial = RecurrentState(h.index_select(0, [index]).squeeze(0),
                                         None if c is None else c.index_select(0, [index]).squeeze(0))
                if packed:
                    output, state = self._run_packed(current, sizes, initial, self._all_weights[index], bool(direction))
                    outputs.append(output)
                else:
                    sequence = current.index_select(0, reverse) if direction else current
                    output, state = self._run(sequence, initial, self._all_weights[index])
                    outputs.append(output.index_select(0, reverse) if direction else output)
                final_h.append(state.h)
                if state.c is not None:
                    final_c.append(state.c)
            current = cat(outputs, 1 if packed else 2) if self.bidirectional else outputs[0]
            if self.dropout and layer < self.num_layers - 1:
                current = self.interlayer_dropout[layer](current)
        h_n = stack(final_h, 0)
        c_n = stack(final_c, 0) if final_c else None
        if packed:
            if input.unsorted_indices is not None:
                inverse = input.unsorted_indices.tolist()
                h_n = h_n.index_select(1, inverse)
                if c_n is not None:
                    c_n = c_n.index_select(1, inverse)
            current = PackedSequence(current, input.batch_sizes, input.sorted_indices, input.unsorted_indices)
        elif unbatched:
            current, h_n = current.squeeze(1), h_n.squeeze(1)
            if c_n is not None:
                c_n = c_n.squeeze(1)
        elif self.batch_first:
            current = current.transpose(0, 1)
        return current, (h_n, c_n) if self._kind == 'lstm' else h_n


class RNN(_Recurrent):
    """Dense tanh/ReLU RNN; hidden state is [layers*directions, batch, hidden]."""
    _builtin_init = True

    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True,
                 batch_first=False, dropout=0., bidirectional=False, device=None, dtype=None):
        self._setup('rnn', input_size, hidden_size, num_layers, bias, batch_first, dropout,
                    bidirectional, nonlinearity, device, dtype)


class GRU(_Recurrent):
    """Reset-after GRU with PyTorch reset/update/new gate ordering."""
    _builtin_init = True

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False, device=None, dtype=None):
        self._setup('gru', input_size, hidden_size, num_layers, bias, batch_first, dropout,
                    bidirectional, 'tanh', device, dtype)


class LSTM(_Recurrent):
    """LSTM with input/forget/candidate/output gates and an explicit (h, c) pair."""
    _builtin_init = True

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False, proj_size=0, device=None, dtype=None):
        if type(proj_size) is not int or proj_size < 0:
            raise ValueError('proj_size must be a nonnegative integer')
        if type(hidden_size) is int and proj_size >= hidden_size:
            raise ValueError('proj_size must be smaller than hidden_size')
        self._setup('lstm', input_size, hidden_size, num_layers, bias, batch_first, dropout,
                    bidirectional, 'tanh', device, dtype, proj_size)


class _Cell(_RecurrentParameters):
    def _setup(self, kind, input_size, hidden_size, bias, nonlinearity, device, dtype):
        _sizes(input_size, hidden_size)
        _factory(device, dtype)
        if type(bias) is not bool:
            raise TypeError('bias must be bool')
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('RNNCell nonlinearity must be tanh or relu')
        self._initialize()
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self._kind = kind
        if kind == 'rnn':
            self.nonlinearity = nonlinearity
        self._names = self._register_weights('', input_size, {'rnn': 1, 'gru': 3, 'lstm': 4}[kind])
        if not bias:
            self.bias_ih, self.bias_hh = None, None
        self.reset_parameters()

    def forward(self, input, hx=None):
        _tensor(input, 'input')
        if len(input.shape) not in (1, 2):
            raise ValueError('recurrent cell input must be [I] or [B,I]')
        unbatched = len(input.shape) == 1
        inputs = input.unsqueeze(0) if unbatched else input
        batch, features = inputs.shape
        if features != self.input_size:
            raise ValueError(f'input last dimension must be {self.input_size}, got {features}')
        _shape([batch, {'rnn': 1, 'gru': 3, 'lstm': 4}[self._kind] * self.hidden_size])
        shape = [self.hidden_size] if unbatched else [batch, self.hidden_size]
        if hx is None:
            h = Tensor.zeros([batch, self.hidden_size])
            c = Tensor.zeros(list(h.shape)) if self._kind == 'lstm' else None
        elif self._kind == 'lstm':
            if not isinstance(hx, (tuple, list)) or len(hx) != 2:
                raise TypeError('LSTMCell state must be an (h, c) pair')
            h, c = _state(hx[0], shape, 'h'), _state(hx[1], shape, 'c')
            if unbatched:
                h, c = h.unsqueeze(0), c.unsqueeze(0)
        else:
            h, c = _state(hx, shape, 'h'), None
            if unbatched:
                h = h.unsqueeze(0)
        nonlinearity = self.nonlinearity if self._kind == 'rnn' else 'tanh'
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('RNNCell nonlinearity must be tanh or relu')
        state = _step(self._kind, inputs, RecurrentState(h, c), self._weights(self._names), nonlinearity)
        h, c = state.h, state.c
        if unbatched:
            h, c = h.squeeze(0), None if c is None else c.squeeze(0)
        return (h, c) if self._kind == 'lstm' else h


class RNNCell(_Cell):
    _builtin_init = True

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None):
        self._setup('rnn', input_size, hidden_size, bias, nonlinearity, device, dtype)


class GRUCell(_Cell):
    _builtin_init = True

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        self._setup('gru', input_size, hidden_size, bias, 'tanh', device, dtype)


class LSTMCell(_Cell):
    _builtin_init = True

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        self._setup('lstm', input_size, hidden_size, bias, 'tanh', device, dtype)
