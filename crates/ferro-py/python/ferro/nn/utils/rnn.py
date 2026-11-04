"""CPU packed-sequence utilities preserving native tensor autograd paths."""
from collections import namedtuple

from ..._native import Tensor, cat, stack
from ..._tensor_api import full, _shape


def _tensor(value, name):
    if not isinstance(value, Tensor):
        raise TypeError(name + ' must be a Tensor')
    if value.device != 'cpu':
        raise NotImplementedError('packed-sequence utilities currently support CPU tensors only')


def _integers(value, name):
    if isinstance(value, Tensor):
        if value.device != 'cpu' or value.dtype != 'int64' or value.ndim != 1:
            raise ValueError(name + ' must be a one-dimensional CPU int64 tensor')
        return value.tolist()
    try:
        result = list(value)
    except TypeError:
        raise TypeError(name + ' must be an integer sequence or CPU int64 tensor') from None
    if any(type(item) is not int for item in result):
        raise TypeError(name + ' must contain integers')
    return result


def _permutation(value, batch, name):
    if not isinstance(value, Tensor):
        raise TypeError(name + ' must be a CPU int64 tensor')
    values = _integers(value, name)
    if sorted(values) != list(range(batch)):
        raise ValueError(name + ' must be a permutation of batch indices')
    return values


class PackedSequence(namedtuple('_PackedSequence', 'data batch_sizes sorted_indices unsorted_indices')):
    """Time-major packed rows plus validated CPU int64 packing metadata.

    Construct with packing helpers in ordinary use. Unsorted helpers use stable
    ties; sorting tied lengths is not a cross-runtime ordering guarantee.
    """
    __slots__ = ()

    def __new__(cls, data, batch_sizes, sorted_indices=None, unsorted_indices=None):
        _tensor(data, 'data')
        if data.ndim == 0:
            raise ValueError('packed data must have at least one dimension')
        if not isinstance(batch_sizes, Tensor):
            raise TypeError('batch_sizes must be a CPU int64 tensor')
        sizes = _integers(batch_sizes, 'batch_sizes')
        if any(size <= 0 for size in sizes) or any(a < b for a, b in zip(sizes, sizes[1:])):
            raise ValueError('batch_sizes must be positive and nonincreasing')
        if sum(sizes) != data.shape[0]:
            raise ValueError('sum(batch_sizes) must match packed data rows')
        batch = sizes[0] if sizes else 0
        if sorted_indices is None:
            if unsorted_indices is not None:
                raise ValueError('unsorted_indices requires sorted_indices')
        else:
            order = _permutation(sorted_indices, batch, 'sorted_indices')
            inverse = [0] * batch
            for sorted_index, original_index in enumerate(order):
                inverse[original_index] = sorted_index
            if unsorted_indices is None:
                unsorted_indices = Tensor.from_i64(inverse, [batch])
            elif _permutation(unsorted_indices, batch, 'unsorted_indices') != inverse:
                raise ValueError('unsorted_indices must invert sorted_indices')
        return super().__new__(cls, data, batch_sizes, sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        return False

    def is_pinned(self):
        return False

    def to(self, device):
        if device != 'cpu':
            raise NotImplementedError('PackedSequence currently supports CPU tensors only')
        return self

    def cpu(self):
        return self


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    """Pack positive lengths, truncating padded tails without detaching input."""
    _tensor(input, 'input')
    if input.ndim < 2:
        raise ValueError('padded input must have time and batch dimensions')
    if type(batch_first) is not bool or type(enforce_sorted) is not bool:
        raise TypeError('batch_first and enforce_sorted must be bool')
    x = input.transpose(0, 1) if batch_first else input
    values = _integers(lengths, 'lengths')
    time, batch = x.shape[:2]
    if len(values) != batch or batch == 0 or any(length <= 0 or length > time for length in values):
        raise ValueError('lengths must contain one positive length <= time for each nonempty batch member')
    order = sorted(range(batch), key=lambda index: -values[index])
    if enforce_sorted and any(a < b for a, b in zip(values, values[1:])):
        raise ValueError('lengths must be sorted in decreasing order when enforce_sorted=True')
    sorted_indices = None if enforce_sorted else Tensor.from_i64(order, [batch])
    if not enforce_sorted:
        x = x.index_select(1, order)
        values = [values[index] for index in order]
    _shape([values[0]], 'int64')
    sizes = [sum(length > step for length in values) for step in range(values[0])]
    rows = [x.narrow(0, step, 1).squeeze(0).narrow(0, 0, size) for step, size in enumerate(sizes)]
    return PackedSequence(cat(rows, 0), Tensor.from_i64(sizes, [len(sizes)]), sorted_indices)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0., total_length=None):
    """Restore original batch order and return padded values plus int64 lengths."""
    if not isinstance(sequence, PackedSequence):
        raise TypeError('sequence must be a PackedSequence')
    # _replace() is inherited from tuple; validate metadata again at use.
    sequence = PackedSequence(*sequence)
    if type(batch_first) is not bool:
        raise TypeError('batch_first must be bool')
    sizes = sequence.batch_sizes.tolist()
    if not sizes:
        raise ValueError('cannot pad an empty PackedSequence')
    time, batch, features = len(sizes), sizes[0], sequence.data.shape[1:]
    if total_length is not None and (type(total_length) is not int or total_length < time):
        raise ValueError('total_length must be an integer at least the longest sequence length')
    total = time if total_length is None else total_length
    _shape([total, batch, *features], sequence.data.dtype)
    rows, offset = [], 0
    for size in sizes:
        row = sequence.data.narrow(0, offset, size)
        if size < batch:
            row = cat([row, full([batch - size, *features], padding_value, dtype=sequence.data.dtype)], 0)
        rows.append(row)
        offset += size
    padded = stack(rows, 0)
    if total > time:
        padded = cat([padded, full([total - time, batch, *features], padding_value, dtype=sequence.data.dtype)], 0)
    lengths = [sum(size > index for size in sizes) for index in range(batch)]
    if sequence.unsorted_indices is not None:
        inverse = sequence.unsorted_indices.tolist()
        padded = padded.index_select(1, inverse)
        lengths = [lengths[index] for index in inverse]
    return (padded.transpose(0, 1) if batch_first else padded), Tensor.from_i64(lengths, [batch])


def pad_sequence(sequences, batch_first=False, padding_value=0., padding_side='right'):
    """Pad a nonempty sequence list, including individual zero-length tensors."""
    if type(batch_first) is not bool:
        raise TypeError('batch_first must be bool')
    if padding_side not in ('left', 'right'):
        raise ValueError('padding_side must be left or right')
    sequences = list(sequences)
    if not sequences:
        raise ValueError('sequences must not be empty')
    for sequence in sequences:
        _tensor(sequence, 'sequence')
        if sequence.ndim == 0:
            raise ValueError('each sequence must have a length dimension')
    first = sequences[0]
    if any(value.shape[1:] != first.shape[1:] or value.dtype != first.dtype for value in sequences):
        raise ValueError('sequence feature shapes and dtypes must match')
    time = max(value.shape[0] for value in sequences)
    _shape([time, len(sequences), *first.shape[1:]], first.dtype)
    padded = []
    for sequence in sequences:
        missing = time - sequence.shape[0]
        if missing:
            padding = full([missing, *first.shape[1:]], padding_value, dtype=first.dtype)
            sequence = cat([sequence, padding] if padding_side == 'right' else [padding, sequence], 0)
        padded.append(sequence)
    return stack(padded, 0 if batch_first else 1)


def pack_sequence(sequences, enforce_sorted=True):
    sequences = list(sequences)
    padded = pad_sequence(sequences)
    return pack_padded_sequence(padded, [value.shape[0] for value in sequences], enforce_sorted=enforce_sorted)


def unpad_sequence(padded_sequences, lengths, batch_first=False):
    """Return differentiable views of right-padded sequences in batch order."""
    _tensor(padded_sequences, 'padded_sequences')
    if padded_sequences.ndim < 2 or type(batch_first) is not bool:
        raise ValueError('padded input needs time/batch dimensions and a boolean batch_first')
    x = padded_sequences if batch_first else padded_sequences.transpose(0, 1)
    values = _integers(lengths, 'lengths')
    if len(values) != x.shape[0] or any(value < 0 or value > x.shape[1] for value in values):
        raise ValueError('lengths must contain one length within the padded time dimension per batch member')
    return [x.narrow(0, index, 1).squeeze(0).narrow(0, 0, length) for index, length in enumerate(values)]
