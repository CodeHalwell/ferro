"""Native TensorDataset batching and lazy synchronous Python datasets.

The native TensorDataset worker path retains its existing sampler/order contract
and materializes an epoch. Synchronous datasets/iterables collate one batch at a time;
workers for arbitrary Python callbacks are unsupported. Shuffle sequences follow
Ferro's seed+epoch convention, not Torch's RNG sequence.
"""
import bisect
import itertools
import math
import os
import random
from collections.abc import Mapping, Sequence
from ._native import Tensor, TensorDataset, Generator, _load_epoch, _sample_order, stack
from ._tensor_api import tensor


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError('Dataset subclasses must implement __getitem__')

    def __add__(self, other):
        return ConcatDataset([self, other])


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError('IterableDataset subclasses must implement __iter__')

    def __add__(self, other):
        raise NotImplementedError('concatenating iterable datasets is unsupported')


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, tuple(indices)
        if any(type(index) is not int for index in self.indices):
            raise TypeError('subset indices must be integers')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.dataset[i] for i in self.indices[index]]
        return self.dataset[self.indices[index]]


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = tuple(datasets)
        if not self.datasets or any(isinstance(d, IterableDataset) for d in self.datasets):
            raise ValueError('ConcatDataset requires nonempty map-style datasets')
        self.cumulative_sizes = list(itertools.accumulate(len(d) for d in self.datasets))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError('dataset index must be an integer')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('dataset index out of range')
        part = bisect.bisect_right(self.cumulative_sizes, index)
        previous = self.cumulative_sizes[part - 1] if part else 0
        return self.datasets[part][index - previous]


def random_split(dataset, lengths, generator=None, *, seed=None):
    lengths = list(lengths)
    count = len(dataset)
    if not lengths or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in lengths):
        raise ValueError('split lengths must be finite nonnegative numbers')
    if math.isclose(sum(lengths), 1.) and sum(lengths) <= 1.:
        if any(n > 1 for n in lengths):
            raise ValueError('split fractions must be between zero and one')
        lengths = [math.floor(count * n) for n in lengths]
        for i in range(count - sum(lengths)):
            lengths[i % len(lengths)] += 1
    if any(type(n) is not int for n in lengths) or sum(lengths) != count:
        raise ValueError('split lengths must sum to the dataset length')
    if generator is not None and (not isinstance(generator, Generator) or seed is not None):
        raise ValueError('provide one Ferro Generator or seed')
    if seed is not None and type(seed) is not int:
        raise TypeError('seed must be an integer')
    indices = list(range(count))
    if generator is None:
        random.Random(seed).shuffle(indices)
    else:
        for i in range(count - 1, 0, -1):
            j = min(i, int(Tensor.rand([1], generator=generator).item() * (i + 1)))
            indices[i], indices[j] = indices[j], indices[i]
    offsets = list(itertools.accumulate([0] + lengths))
    return [Subset(dataset, indices[offsets[i]:offsets[i + 1]]) for i in range(len(lengths))]


def default_convert(value):
    if isinstance(value, Mapping):
        return {key: default_convert(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(default_convert(item) for item in value))
    if isinstance(value, (list, tuple)):
        return [default_convert(item) for item in value]
    return value


def default_collate(batch):
    if not batch:
        raise ValueError('cannot collate an empty batch')
    first = batch[0]
    if isinstance(first, Tensor):
        if any(not isinstance(item, Tensor) or item.dtype != first.dtype or item.device != first.device or item.shape != first.shape for item in batch):
            raise ValueError('tensor samples must share shape, dtype and device')
        return stack(batch, 0)
    if type(first) in (bool, int, float):
        if any(type(item) is not type(first) for item in batch):
            raise TypeError('scalar samples must share a type')
        return tensor(batch, dtype={bool: 'bool', int: 'int64', float: 'float64'}[type(first)])
    if isinstance(first, (str, bytes)):
        if any(type(item) is not type(first) for item in batch):
            raise TypeError('string samples must share a type')
        return list(batch)
    if isinstance(first, Mapping):
        if any(not isinstance(item, Mapping) or set(item) != set(first) for item in batch):
            raise ValueError('mapping samples must share keys')
        return {key: default_collate([item[key] for item in batch]) for key in first}
    if isinstance(first, Sequence):
        if any(not isinstance(item, Sequence) or len(item) != len(first) for item in batch):
            raise ValueError('sequence samples must have equal lengths')
        columns = [default_collate(items) for items in zip(*batch)]
        return type(first)(*columns) if isinstance(first, tuple) and hasattr(first, '_fields') else columns
    raise TypeError('unsupported sample type for default_collate: ' + type(first).__name__)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, seed=None,
                 *, collate_fn=None, sampler=None, batch_sampler=None):
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError('batch_size must be a positive integer or None')
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ValueError('num_workers must be a nonnegative integer')
        if type(shuffle) is not bool or type(drop_last) is not bool:
            raise TypeError('shuffle and drop_last must be bool')
        if seed is not None and type(seed) is not int:
            raise TypeError('seed must be an integer')
        if collate_fn is not None and not callable(collate_fn):
            raise TypeError('collate_fn must be callable')
        if sampler is not None and shuffle:
            raise ValueError('sampler and shuffle are mutually exclusive')
        if batch_sampler is not None and (batch_size != 1 or shuffle or drop_last or sampler is not None):
            raise ValueError('batch_sampler is mutually exclusive with batching/sampling options')
        iterable = isinstance(dataset, IterableDataset)
        if iterable and (shuffle or sampler is not None or batch_sampler is not None):
            raise ValueError('IterableDataset does not support shuffle or custom samplers')
        if not iterable and (not hasattr(dataset, '__getitem__') or not hasattr(dataset, '__len__')):
            raise TypeError('dataset must be map-style or an IterableDataset')
        if batch_size is None and drop_last:
            raise ValueError('drop_last requires automatic batching')
        native = isinstance(dataset, TensorDataset) and collate_fn is None and sampler is None and batch_sampler is None and batch_size is not None
        if num_workers and not native:
            raise NotImplementedError('Python datasets/collation support num_workers=0 only')
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.drop_last, self.num_workers = drop_last, num_workers
        self.collate_fn = collate_fn if collate_fn is not None else (default_convert if batch_size is None and batch_sampler is None else default_collate)
        if native and num_workers == 0:
            self.collate_fn = lambda batch: tuple(default_collate(batch))
        self.sampler, self.batch_sampler = sampler, batch_sampler
        self._native, self._iterable = native and num_workers > 0, iterable
        self._native_sampler = native
        self.seed = int.from_bytes(os.urandom(8), 'little') if seed is None else seed
        self.epoch = 0

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        n = len(self.sampler) if self.sampler is not None else len(self.dataset)
        if self.batch_size is None:
            return n
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        seed = (self.seed + self.epoch) % 2**64 if self.shuffle else None
        self.epoch += 1
        if self._native:
            return iter(_load_epoch(self.dataset, self.batch_size, seed, self.drop_last, self.num_workers))
        if self.batch_sampler is not None:
            return (self.collate_fn([self.dataset[i] for i in indices]) for indices in self.batch_sampler)
        if self._iterable:
            samples = iter(self.dataset)
        else:
            indices = range(len(self.dataset)) if self.sampler is None else self.sampler
            if self.shuffle:
                if self._native_sampler:
                    indices = _sample_order(len(self.dataset), seed)
                else:
                    indices = list(indices)
                    random.Random(seed).shuffle(indices)
            samples = (self.dataset[i] for i in indices)
        if self.batch_size is None:
            return map(self.collate_fn, samples)
        return self._batches(samples)

    def _batches(self, samples):
        while True:
            batch = list(itertools.islice(samples, self.batch_size))
            if not batch or self.drop_last and len(batch) < self.batch_size:
                return
            yield self.collate_fn(batch)
