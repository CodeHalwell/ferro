"""Native in-memory batching over ferro_core::data::{TensorDataset, DataLoader}.

Only TensorDataset is supported: native workers cannot call Python
__getitem__ without serializing on the GIL. Each epoch is collated natively
(GIL released, worker threads when num_workers > 0) before batches are yielded,
so one epoch of stacked batches is held in memory. shuffle=True reshuffles per
epoch from (seed + epoch); pass seed for reproducible orders.
"""
import os
from ._native import TensorDataset, _load_epoch


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, seed=None):
        if not isinstance(dataset, TensorDataset):
            raise TypeError('DataLoader supports ferro.data.TensorDataset only')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ValueError('num_workers must be a nonnegative integer')
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, bool(shuffle)
        self.drop_last, self.num_workers = bool(drop_last), num_workers
        self.seed = int.from_bytes(os.urandom(8), 'little') if seed is None else seed
        self.epoch = 0

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        seed = (self.seed + self.epoch) % 2**64 if self.shuffle else None
        self.epoch += 1
        return iter(_load_epoch(self.dataset, self.batch_size, seed, self.drop_last, self.num_workers))
