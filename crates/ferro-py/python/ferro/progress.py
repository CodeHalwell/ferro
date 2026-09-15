"""Synchronous scalar metric reporting; never performs training operations."""
from contextlib import contextmanager
from numbers import Real
import math
import sys


class Progress:
    """Weighted means over explicitly logged scalars, with persistent counters.

    Use ``with progress.epoch(): for batch in progress.batches(source): ...``.
    Call update() only AFTER a successful optimizer step. log() accepts Python
    scalars, not tensors: tensor.item() and its synchronization remain explicit.
    Accuracy is a user-supplied fraction; no predictions or labels are inferred.
    """
    def __init__(self, *, log_every=1, stream=None, enabled=True, tensor_metrics=False):
        if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1:
            raise ValueError('log_every must be a positive integer')
        self.tensor_metrics = tensor_metrics
        self.log_every, self.stream, self.enabled = log_every, stream, enabled
        self.epoch_count = self.batch = self.global_batch = self.updates = 0
        self.phase = 'train'
        self.total = None
        self._active = self._iterating = False
        self._epoch_token = None
        self._sums, self._weights = {}, {}

    def reset(self):
        """Reset all counters and metrics; forbidden during an epoch."""
        if self._active:
            raise RuntimeError("Cannot reset an active epoch")
        self.epoch_count = self.batch = self.global_batch = self.updates = 0
        self.total = None
        self._sums, self._weights = {}, {}

    @property
    def metrics(self):
        return {name: value / self._weights[name] for name, value in self._sums.items()}

    @property
    def should_log(self):
        """Cadence hint for explicit scalar extraction; not a logging requirement."""
        return self.enabled and self.batch > 0 and self.batch % self.log_every == 0

    @contextmanager
    def epoch(self, *, phase='train'):
        if self._active:
            raise RuntimeError('An epoch is already active')
        if not isinstance(phase, str) or not phase:
            raise ValueError('phase must be a nonempty string')
        self._active = True
        self._epoch_token = object()
        self.phase = phase
        if phase == 'train':
            self.epoch_count += 1
        self._exhausted = False
        self.batch, self.total = 0, None
        self._sums, self._weights = {}, {}
        try:
            yield self
        except BaseException:
            raise
        else:
            self._print('complete' if self._exhausted else 'stopped')
        finally:
            self._epoch_token = None
            self._active = self._iterating = False

    def batches(self, source):
        if not self._active or self._iterating:
            raise RuntimeError('batches requires an active epoch and no other iterator')
        return self._batches(source, self._epoch_token)

    def _batches(self, source, token):
        if token is not self._epoch_token or not self._active or self._iterating:
            raise RuntimeError('batches iterator is expired or another iterator is active')
        self._iterating = True
        try:
            try:
                self.total = len(source)
            except TypeError:
                self.total = None
            iterator = iter(source)
            while True:
                if token is not self._epoch_token or not self._active:
                    raise RuntimeError('batches iterator has outlived its epoch')
                try:
                    value = next(iterator)
                except StopIteration:
                    self._exhausted = True
                    break
                self.batch += 1
                self.global_batch += 1
                yield value
                if token is not self._epoch_token or not self._active:
                    raise RuntimeError('batches iterator has outlived its epoch')
                if self.should_log:
                    self._print()
        finally:
            if token is self._epoch_token:
                self._iterating = False

    def log(self, metrics=None, *, weight=1, **values):
        if not self._active:
            raise RuntimeError('log requires an active epoch')
        merged = dict(metrics or {})
        merged.update(values)
        if not isinstance(weight, Real) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('weight must be a positive finite scalar')
        for name, value in merged.items():
            if not isinstance(name, str) or not name:
                raise TypeError('Metric names must be nonempty strings')
            if not isinstance(value, Real):
                raise TypeError('Metrics must be explicit Python scalars; use tensor.item() at your chosen cadence')
        for name, value in merged.items():
            self._sums[name] = self._sums.get(name, 0.) + float(value) * weight
            self._weights[name] = self._weights.get(name, 0.) + weight

    def update(self, count=1):
        if not self._active:
            raise RuntimeError('update requires an active epoch')
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('update count must be a nonnegative integer')
        self.updates += count

    def _print(self, status=''):
        if not self.enabled:
            return
        batch = str(self.batch) if self.total is None else f'{self.batch}/{self.total}'
        parts = [self.phase, f'epoch={self.epoch_count}', f'batch={batch}', f'updates={self.updates}']
        for name, value in self.metrics.items():
            parts.append(f'{name}={value:.2%}' if name == 'accuracy' else f'{name}={value:.4f}')
        if status:
            parts.append(status)
        print(' '.join(parts), file=self.stream if self.stream is not None else sys.stdout)
